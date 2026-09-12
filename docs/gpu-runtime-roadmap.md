# GPU Runtime

## Purpose

The GPU runtime architecture: residency rules, the GPU-first output
path, and the central `RuntimeGpuBudget` pressure policy. This
document records the current contracts and the remaining extension
slices.

Related documents:

- [architecture.md](architecture.md)
- [threading-model.md](threading-model.md)
- [vulkano-renderer-design.md](vulkano-renderer-design.md)
- [output-interop.md](output-interop.md)
- [profiling.md](profiling.md)
- [tracking-v2-design.md](tracking-v2-design.md)

## Runtime Shape

```text
camera frame
   │
   ▼
tracking worker ── DirectML sessions
   │              RTMW3D / YOLOX / FaceMesh
   │ compact pose + expression data only
   ▼
app thread
   │ pose, weights, simulation controls
   ▼
render thread ─── Vulkan
   │  static base SSBOs / morph delta SSBOs / cloth pos+norm SSBOs
   │  per-primitive transform compute prepass (`transform_cs`)
   │  ↓ writes world-space `GpuVertex` SSBO
   │  graphics pipelines (forward + outline) read it as vertex buffer
   ▼
exportable GPU image
   │
   ├── local preview path
   └── output worker via GpuFrameToken + explicit sync
```

### Ownership rule

- **CPU owns** scene intent: pose targets, morph weights, calibration,
  scheduling, UI state.
- **GPU owns** large transient fields: vertex deltas, render targets,
  cloth particle state (GPU backend), exported frames.
- **Cross-thread handoff owns only tokens**: compact metadata plus
  explicit synchronization, never naked renderer internals.

### Design principles

1. **GPU residency first** — data that is large, reused every frame,
   or consumed only by GPU stages is created once and kept resident.
2. **Rebuild on topology change, update on state change** — avatar
   import rebuilds GPU resources; a morph weight change rewrites a
   1 KB control UBO; a cloth step rewrites existing state, not
   resource identity. No per-frame `VkBuffer` allocation in the draw
   path.
3. **CPU readback is a service path** — screenshots, thumbnails,
   image-sequence export, unsupported sinks, diagnostics. Normal
   output rides `GpuFrameToken` (see output-interop.md for the
   lease/sidecar protocol).
4. **Synchronization is part of the API** — a frame is ready only when
   its fence/semaphore/token says so.

## Core Mechanisms

- **Compute prepass** — skinning + morph + cloth fused into
  `pipeline::transform_cs`; per-frame host writes bounded to the
  control UBO and (on solver version bumps) the cloth SSBOs.
  `GPU_RUNTIME` counters (`transform_resources`, `ubo_writes`,
  `cloth_writes`, `cloth_creations`, `readback_bytes`) make
  allocation stability observable — see profiling.md.
- **GPU-first output** — `output_export.rs` returns
  `ExportedPixelData::{GpuFrameToken, CpuReadback}`; sinks consume
  tokens from a bounded export image pool with explicit lease
  bookkeeping; CPU readback remains the visible fallback. Active
  handoff mode + dropped frames surface in the Output inspector.
- **GPU cloth backend** — persistent `ClothGpuSlot` buffers, compute
  kernels (`cloth_verlet_cs`, `cloth_constraint_accumulate_cs` /
  `cloth_constraint_apply_cs`, `cloth_normal_cs`); when
  `ClothDeformSnapshot::solver_backend == Gpu` the renderer dispatches
  them and `transform_cs` reads GPU-filled cloth SSBOs. Formula parity
  is locked by `cloth_gpu_boundary::tests` (Rust mirrors of each GLSL
  body, asserted within `1e-3` of the CPU PBD reference). The CPU
  solver remains the default production backend; nothing in
  `RuntimeGpuBudget` flips the backend yet.
- **Failure-recovery loop** — GPU export failures drive
  `EmergencyCpu`, which forces `RenderExportMode::CpuReadback` until
  the clean-streak recovery clears.

## RuntimeGpuBudget — State Machine

`src/app/runtime_gpu_budget.rs` is the single place that decides
system-level cadence; consumers read its outputs and never hard-code
their own thresholds.

### State machine

| Mode | Trigger to enter | Trigger to leave | Render FPS clamp | Pose Hz | YOLOX skip | Depth refresh | FaceMesh EP |
|---|---|---|---|---|---|---|---|
| `Healthy` | initial / recovered | render dt > 1.2× target dt **or** drops/sec ≥ 5 (5 s sustained) | user choice | 30 | every 4 | every 1 | Auto |
| `PressureLight` | sustained light pressure for 5 s | clean 30 s **or** light→heavy escalation | min(user, 45) | 25 | every 6 | every 2 | Auto |
| `PressureHeavy` | sustained light pressure another 5 s **or** drops/sec ≥ 20 (bypass) | clean 30 s | min(user, 30) | 20 | every 8 | every 3 | Cpu |
| `EmergencyCpu` | 2 GPU export failures in the recent window | clean 30 s | min(user, 30) | 15 | every 12 | every 4 | Cpu |

Depth refresh: the metric cloud rebuild runs every Nth *estimated* frame;
skipped frames hand the provider a clone of the last cloud (stale by at
most one refresh interval). Healthy stays at 1 (= the pre-budget
per-frame rebuild) so the budget never changes nominal tracking
behaviour — staleness only grows once the system is already degrading.
The original design table sketched 4/6/8/12 including Healthy; that
would have silently degraded nominal tracking without bench backing and
was changed when the wiring landed.

Hysteresis is the point. Recovery requires a 30-second clean streak per
step — a one-frame stutter does not immediately bounce you up a level,
and a one-frame all-clean does not race you back to `Healthy` from
`EmergencyCpu`. The 20-drops/sec bypass is the only short-circuit
upward; everything else needs the 5 s dwell.

### Consumer plumbing

- **Render FPS**: forwarded each frame from
  `Application::update_runtime_gpu_budget` to
  `OutputRouter::set_target_fps`. The output throttle gate enforces the
  clamp without any other party knowing about it.
- **YOLOX skip period**: published to
  `tracking::rtmw3d::YOLOX_REFRESH_PERIOD` (a `pub static AtomicU64`).
  `Rtmw3dInference::process_pose` loads it on every frame inside the
  submit guard. A single shared atomic is used in preference to an
  `Arc<AtomicU64>` plumbing path because the value is conceptually
  global — one Application + one Rtmw3dInference per session.
- **Pose Hz**: published to `tracking::worker::POSE_HZ_TARGET` and read
  in the worker's estimate loop. Frames that fail the pacing check
  (`pose_throttle_allows`, a device-clock accumulator so the effective
  rate converges to the target from any camera rate) are dropped before
  the inference pass; the mailbox keeps the last published estimate.
  The throttle is disabled while the session recorder is active so
  `pose.jsonl` stays gap-free.
- **Depth refresh period**: published to
  `tracking::worker::DEPTH_REFRESH_PERIOD`; the metric-cloud rebuild
  runs on every Nth estimated frame and skipped frames clone the last
  cloud into the provider.
- **FaceMesh ONNX EP**: published to
  `tracking::face_mediapipe::FACEMESH_EP_CPU`, consulted at
  `rtmw3d::from_models_dir_with_options` session-build time — a flip
  takes effect on the next tracking start, the same latency class as
  the user-facing `force_cpu` toggle.

### Inputs the budget reads

`RuntimeMeasurements`, populated each frame from:

- `render_dt_ema` — 5-frame EMA of render-thread frame dt
- `OutputRouter::dropped_count()` → drops/sec across the sampling window
- `OutputDiagnostics::export_pool` → leased / capacity counts
- recent GPU export failures via
  `OutputRouter::take_gpu_export_failure_count` — a take-and-reset
  counter; ≥ 2 failures inside a single budget tick window drives
  `EmergencyCpu`.

### Why centralise

Render cadence, YOLOX skip period, depth refresh, and FaceMesh EP are
system-level policy: no single subsystem can see the others' load.
Kept per-subsystem, they degrade independently — one drops frames
while another stays at full cost, and the remedy is manual re-tuning.
The budget makes degraded modes intentional and visible in one place.

## Remaining Extension Slices

Every wiring change must land with a real consumer in the same
commit, so the budget doesn't accumulate dead policy outputs. Wired so
far: render FPS, YOLOX skip period, pose Hz, depth refresh period,
FaceMesh EP preference (see "Consumer plumbing" above). Remaining
slices:

- **GPU cloth as default** — promote the GPU cloth backend from
  parity-tested option to production default (potentially budget-driven),
  shrinking the CPU snapshot vector to a metadata-only descriptor.
  Opt-in today via `VULVATAR_CLOTH_GPU=1` (attach-time selection;
  dynamic pin targets that follow the avatar's bones are wired through
  `ClothGpuDispatchControl::pin_positions`). Blocked on a GPU collider
  stage (capsules still live only in the CPU solver) and, for CPU-side
  consumers, a GPU→CPU readback path.
- **GPU-local preview** — egui still consumes CPU pixels for the
  viewport; that is a distinct preview fallback path and must not
  define the output architecture.

## Verification Matrix

| Test | Why |
|---|---|
| face tracking on/off | proves the morph path stays allocation-stable; `transform_resources` counter must not climb |
| cloth on/off | proves cloth writes reuse buffers; `cloth_creations` counter ticks once per primitive at first cloth-on, then plateaus |
| 640p / 720p / 1080p output | exposes readback scaling |
| CPU fallback vs GPU token path | proves fallback parity |
| 30 min soak | catches allocator leaks / device loss |
| provider variants | reveals DirectML/Vulkan contention |
| OBS handoff color test | protects metadata + color space |

## What Not To Do

- Do not optimize the CPU readback path into permanence.
- Do not pass renderer-owned Vulkan images directly into output worker
  code without a lease/token contract.
- Do not scatter GPU cadence decisions across RTMW3D, YOLOX, GUI,
  and output code — `RuntimeGpuBudget` is the one place that describes
  the system-level policy.
