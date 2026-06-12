# GPU Runtime

## Purpose

The GPU runtime architecture: residency rules, the GPU-first output
path, and the central `RuntimeGpuBudget` pressure policy. The original
three-phase roadmap (stabilize baseline → GPU-first output → GPU
simulation + pacing) has fully landed; this document now records the
landed contracts and the remaining extension slices.

Related documents:

- [architecture.md](architecture.md)
- [threading-model.md](threading-model.md)
- [vulkano-renderer-design.md](vulkano-renderer-design.md)
- [output-interop.md](output-interop.md)
- [profiling.md](profiling.md)
- [onnx-tracking-pipeline.md](onnx-tracking-pipeline.md)

## Runtime Shape

```text
camera frame
   │
   ▼
tracking worker ── DirectML sessions
   │              RTMW3D / YOLOX / FaceMesh / depth
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

## Landed Milestones

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

## RuntimeGpuBudget — Landed State Machine

`src/app/runtime_gpu_budget.rs` is the single place that decides
system-level cadence; consumers read its outputs and never hard-code
their own thresholds.

### State machine

| Mode | Trigger to enter | Trigger to leave | Render FPS clamp | Pose Hz | YOLOX skip | Depth skip | FaceMesh EP |
|---|---|---|---|---|---|---|---|
| `Healthy` | initial / recovered | render dt > 1.2× target dt **or** drops/sec ≥ 5 (5 s sustained) | user choice | 30 | every 4 | every 4 | Auto |
| `PressureLight` | sustained light pressure for 5 s | clean 30 s **or** light→heavy escalation | min(user, 45) | 25 | every 6 | every 6 | Auto |
| `PressureHeavy` | sustained light pressure another 5 s **or** drops/sec ≥ 20 (bypass) | clean 30 s | min(user, 30) | 20 | every 8 | every 8 | Cpu |
| `EmergencyCpu` | 2 GPU export failures in the recent window | clean 30 s | min(user, 30) | 15 | every 12 | every 12 | Cpu |

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
- **Pose Hz / Depth skip / FaceMesh EP**: budget fields are populated
  and surfaced in the inspector; the consumer wiring lands when each
  subsystem starts honouring the budget. The YOLOX skip period is the
  reference pattern (`pub static AtomicU64` read in the submit guard).

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

Before this landed, render cadence lived on `OutputRouter`, YOLOX skip
period was a `const` in `tracking::rtmw3d`, depth refresh was a
separate const, and facemesh EP was chosen at startup with no runtime
adjustment. None of them could see the others. Under sustained load
the system would either (a) drop frames in one subsystem while another
remained at full cost, or (b) require manual re-tuning. The budget
makes degraded modes intentional and visible in one place.

## Remaining Extension Slices

Each is a small wiring change that must land with a real consumer in
the same commit, so the budget doesn't accumulate dead policy outputs:

- **Pose worker Hz throttle** — `pub static AtomicU32` next to the pose
  worker's loop tick, written from
  `Application::update_runtime_gpu_budget`.
- **Depth refresh period** — same pattern beside the depth provider's
  per-frame guard.
- **FaceMesh ONNX EP preference** — currently hard-coded per provider
  (`Auto` for standalone rtmw3d, `ForceCpu` for the depth-colocated
  paths). Plumbing the preference means re-init on change OR an
  `AtomicU8` consulted at next session build.
- **GPU cloth as default** — promote the GPU cloth backend from
  parity-tested option to production default (potentially budget-driven),
  shrinking the CPU snapshot vector to a metadata-only descriptor.
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
- Do not scatter GPU cadence decisions across RTMW3D, YOLOX, DAv2, GUI,
  and output code — `RuntimeGpuBudget` is the one place that describes
  the system-level policy.
