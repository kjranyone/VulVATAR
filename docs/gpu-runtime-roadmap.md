# GPU Runtime Roadmap

## Purpose

This document turns the current Vulkan + DirectML runtime into one coherent
optimization plan.

The renderer has now crossed an important boundary:

- static mesh data stays resident on the GPU
- skinning, morph blending, and cloth deformation are fused into a single
  per-primitive **compute prepass** (`pipeline::transform_cs`) that writes
  world-space vertices into a persistent device-local SSBO; the graphics
  pipelines bind that SSBO as their vertex buffer and only apply
  view × projection
- per-frame CPU writes are bounded to a 1 KB control UBO and (only when
  the cloth solver bumps its version) the in-place cloth position / normal
  SSBOs — there is no per-frame `VkBuffer` allocation in the draw path

The next phase is to make the rest of the runtime obey the same rule:

> Large, frame-shaped data should stay on the GPU.  
> The CPU should move compact control signals, not whole images or whole meshes.

Related documents:

- [architecture.md](architecture.md)
- [threading-model.md](threading-model.md)
- [renderer-api-contract.md](renderer-api-contract.md)
- [output-interop.md](output-interop.md)
- [profiling.md](profiling.md)
- [onnx-tracking-pipeline.md](onnx-tracking-pipeline.md)

## Current State

### What is already in the right shape

| Area | Current state | Why it is good |
|---|---|---|
| Base meshes | per-primitive `GpuVertexBase` SSBO (`base_ssbo`) + IBO, allocated once on first encounter | no per-frame rebuild, no CPU re-bake of skinned/morphed vertices |
| Transform | single compute prepass (`pipeline::transform_cs`, `local_size_x=64`) that fuses skinning, morph blend, and cloth override per primitive | one dispatch per (instance, primitive); graphics shaders are reduced to `view * proj` |
| Morphs | immutable per-primitive `morph_deltas` SSBO + a 1 KB `TransformControl` UBO that the render loop rewrites with `weights[0..target_count]` and `has_cloth`/`has_cloth_normals` flags | dynamic expressions allocate nothing; only the UBO is touched per frame |
| Cloth draw path | persistent host-visible `cloth_pos_ssbo` / `cloth_norm_ssbo` rewritten in place when the solver bumps `version`; consumed by the same compute prepass | per-frame allocator churn is removed; cloth and morph share the same dispatch |
| Tracking result handoff | mailbox of compact skeleton / expression data | CPU sees small semantic results, not large tensors |
| Render submission | snapshot-based `RenderFrameInput` | renderer boundary is explicit |

### What still fights the target architecture

| Area | Current state | Cost |
|---|---|---|
| Output | every rendered frame is copied image → staging buffer → CPU readback buffer | large GPU→CPU traffic remains the default path |
| Preview | GUI consumes CPU pixels and uploads them again to egui | one extra round trip remains in the local preview path |
| Cloth simulation | CPU solver produces full deformed vertex arrays | draw upload is fixed, but simulation is still CPU-owned |
| GPU scheduling | Vulkan rendering and DirectML inference share one GPU without a unified budget policy | contention is handled locally, not globally |

The important asymmetry is this:

```text
draw path:   now mostly GPU-resident
output path: still CPU-readback-centred
```

The next architectural win is therefore not another draw-path micro-
optimization. It is making output GPU-first.

## Target Runtime Shape

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
  future cloth particle state, exported frames.
- **Cross-thread handoff owns only tokens**: compact metadata plus explicit
  synchronization, never naked renderer internals.

## Design Principles

### 1. GPU residency first

If data is:

- large,
- reused every frame, or
- consumed only by GPU stages,

then it should be created once and kept resident there.

### 2. Rebuild on topology change, update on state change

Examples:

- avatar import / mesh topology change → allocate or rebuild GPU resources
- morph weight change → rewrite a small UBO
- cloth step → rewrite existing state, not resource identity

### 3. CPU readback is a service path

CPU readback is still useful for:

- screenshots
- PNG thumbnail generation
- unsupported sinks
- diagnostics

It should not be the normal path for every preview/output frame.

### 4. Synchronization is part of the API

A frame is not "ready" because render submission happened. It is ready only
when an explicit fence/semaphore/token says so.

## Phase Plan

## Phase 1 — Stabilize the new baseline

Goal: lock in the improvements that removed allocator churn and make the
runtime diagnosable before larger surgery.

### Work

1. Keep the new compute-prepass render path.
2. Keep the DirectML session contract explicit:
   - sequential execution
   - memory pattern disabled
3. Refuse overlapping tracking-worker restarts if an old worker has not
   actually stopped.
4. Add durable metrics (the renderer already emits the first four as the
   60-frame `GPU_RUNTIME` log via `GpuRuntimeCounters`):
   - render frame time
   - `TransformGpuData` creations (`transform_resources`) — should plateau
     at primitive count after avatar warm-up
   - `control_ubo` writes (`ubo_writes`) — bounded by visible primitives
     per frame
   - `cloth_pos_ssbo` writes (`cloth_writes`) — only ticks when the cloth
     solver bumps `version`
   - readback bytes/frame
   - output queue depth
   - tracking provider / EP labels
5. Add a stress profile:
   - face tracking on
   - cloth on
   - 1080p output
   - 30–60 minute soak

### Exit criteria

- no per-frame `VkBuffer` allocation in steady-state morph/cloth draws —
  the only per-frame host writes are the `control_ubo` and the cloth
  SSBOs on solver version bumps; `transform_resources` and
  `cloth_creations` counters are flat after warm-up
- no overlapping tracking workers
- stable 30-minute run without device loss or OS crash
- a profile log can explain where frame time goes

## Phase 2 — GPU-first output

Goal: make `GpuFrameToken` the normal output product and demote CPU readback
to fallback.

### Work

1. Introduce a renderer-owned exportable image pool:
   - bounded size
   - explicit in-flight state
   - recycle only after sink completion
2. Make `output_export.rs` return one of:

```rust
enum ExportedPixelData {
    GpuFrameToken(GpuFrameToken),
    CpuReadback(Arc<Vec<u8>>),
}
```

3. Route normal output sinks through `GpuFrameToken`.
4. Keep CPU readback only for:
   - thumbnails
   - image-sequence export
   - unsupported machines / failed GPU interop
5. Expose active handoff mode in diagnostics:
   - GPU token
   - CPU fallback
   - dropped frame count

### Preferred migration order

1. Define the token/lifetime contract.
2. Implement the image pool and sync bookkeeping.
3. Preserve current CPU path as fallback.
4. Switch sinks one by one.
5. Only then make GPU handoff the default.

### Exit criteria

- live output no longer requires per-frame CPU readback on supported systems
- fallback remains visible and correct
- output queue backpressure never blocks app or render threads

## Phase 3 — GPU simulation and pacing

Goal: make dynamic runtime work GPU-native end to end, then coordinate the
shared GPU budget rather than letting subsystems fight invisibly.

### Work

#### Cloth

Move cloth simulation toward compute in stages:

1. persistent GPU particle/state buffers — landed (`ClothGpuSlot`)
2. compute kernels for integration + constraints + normals — landed
   (`cloth_verlet_cs` / `cloth_constraint_accumulate_cs` +
   `cloth_constraint_apply_cs` / `cloth_normal_cs`)
3. compute-produced deform buffers consumed directly by draw passes —
   landed: when `ClothDeformSnapshot::solver_backend == Gpu`, the
   renderer dispatches the cloth compute pipelines and `transform_cs`
   reads `cloth_pos_ssbo` / `cloth_norm_ssbo` filled by the GPU. The
   per-frame host write into those SSBOs is suppressed.
4. CPU readback only for diagnostics/editor inspection — pending
   (current CPU solver remains the default backend, runs in parallel
   when `Cpu` is selected; `Gpu` backend skips it)

Formula parity locked by `cloth_gpu_boundary::tests`: each shader has a
pure-Rust mirror of its GLSL body and is asserted equal (within
`1e-3`) to the CPU PBD reference over 60 frames / 64 iterations / etc.
Bit-for-bit SPIR-V vs CPU parity needs a hardware integration test
outside the lib boundary.

`ClothDeformSnapshot` is now primitive-scoped (`target_primitive_id`,
`target_mesh_id`, `vertex_offset`, `vertex_count`) and
`RenderAvatarInstance::cloth_deforms` is a `Vec`, so the renderer
already gates `has_cloth` per primitive and applies the deform only
to the snapshot's vertex subset. Multi-primitive cloth instances
no longer collapse body vertices to the origin. When step 3
completes (compute kernel writing the per-primitive cloth SSBO
directly), the CPU snapshot vector can shrink to a metadata-only
descriptor.

#### Preview

Prefer a GPU-local preview surface when the GUI stack allows it. If egui keeps
requiring CPU pixels in the near term, treat that as a distinct preview
fallback path rather than allowing it to define the output architecture.

#### GPU scheduling

Create an explicit runtime budget policy for a shared GPU:

- render cadence target
- pose inference cadence
- YOLOX refresh period
- depth refresh period
- face cascade EP selection
- output queue pressure

The scheduler need not be elaborate first. A small policy object that chooses
provider cadence from live timings is already better than many local constants
that cannot see each other.

### Exit criteria

- cloth deformation no longer requires CPU-owned full-frame geometry in the
  main path
- GPU contention is visible in one place
- degraded modes are intentional, not accidental

## RuntimeGpuBudget — Landed State Machine

Phase 3's GPU scheduling exit criterion landed as
`src/app/runtime_gpu_budget.rs`. The budget is the single place that
decides system-level cadence; consumers read its outputs and never
hard-code their own thresholds.

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
  subsystem starts honouring the budget (these are tracked under
  `plan/runtime-gpu-budget.md`).

### Inputs the budget reads

`RuntimeMeasurements`, populated each frame from:

- `render_dt_ema` — 5-frame EMA of render-thread frame dt
- `OutputRouter::dropped_count()` → drops/sec across the sampling window
- `OutputDiagnostics::export_pool` → leased / capacity counts
- recent GPU export failures (currently 0; a small failure-window
  counter is the remaining work — `EmergencyCpu` will only fire once
  that counter is populated)

### Why centralise

Before this landed, render cadence lived on `OutputRouter`, YOLOX skip
period was a `const` in `tracking::rtmw3d`, depth refresh was a
separate const, and facemesh EP was chosen at startup with no runtime
adjustment. None of them could see the others. Under sustained load
the system would either (a) drop frames in one subsystem while another
remained at full cost, or (b) require manual re-tuning. The budget
makes degraded modes intentional and visible in one place.

## Recommended Concrete Types

These are the types that would make the intended boundaries harder to violate:

```rust
struct GpuFrameToken {
    image_id: ExportedResourceId,
    extent: [u32; 2],
    format: OutputPixelFormat,
    alpha_mode: RenderOutputAlpha,
    ready_sync: OutputSyncToken,
    lease: FrameLease,
}

struct ExportImagePool {
    slots: Vec<ExportImageSlot>,
    policy: FramePoolPolicy,
}

struct RuntimeGpuBudget {
    render_target_fps: u32,
    pose_target_hz: u32,
    yolox_period: u64,
    depth_period: u64,
    face_ep: FaceMeshEp,
}

struct ClothGpuState {
    particles: StorageBuffer,
    constraints: StorageBuffer,
    deform_vertices: StorageBuffer,
}
```

The exact names can evolve. The point is the boundary:

- token, not raw image
- pool, not ad hoc allocation
- budget, not scattered magic numbers
- GPU state, not per-frame CPU geometry ownership

## Changes To Existing Modules

### `src/renderer/mod.rs`

Should gradually lose:

- CPU-readback-centric frame completion assumptions
- output lifetime policy

Should keep:

- render orchestration
- draw submission
- resource-cache ownership

### `src/renderer/output_export.rs`

Should become the center of:

- export image pool
- sync token creation
- CPU fallback choice

### `src/output/`

Should consume:

- `GpuFrameToken` in the normal path
- `CpuReadback` only in fallback paths

### `src/simulation/`

Should keep CPU cloth solver while it is still authoritative, but expose a
future-compatible interface so the renderer can later consume GPU-produced
deform buffers without changing avatar ownership rules.

### `src/tracking/`

Should gain a higher-level cadence/budget policy rather than letting each
provider encode permanent timing assumptions in isolation.

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
- Do not pass renderer-owned Vulkan images directly into output worker code
  without a lease/token contract.
- Do not move cloth to compute while output still quietly forces every frame
  back through CPU unless there is a deliberate milestone reason.
- Do not scatter GPU cadence decisions across RTMW3D, YOLOX, DAv2, GUI, and
  output code without one place that can describe the system-level policy.

## Immediate Next Implementation Task

The Phase 2 GPU export pool, Phase 3 cloth compute migration, and
Phase 3 runtime budget have all landed. The remaining immediate work is:

> Add a recent-window GPU-export-failure counter to `OutputDiagnostics`
> so `RuntimeGpuBudget::EmergencyCpu` can actually fire (today the
> input is hardcoded to 0). Then plumb `pose_hz_target`, `depth_skip_period`,
> and `facemesh_ep_preference` to their respective consumers, the same
> way YOLOX skip is wired through `tracking::rtmw3d::YOLOX_REFRESH_PERIOD`.

These complete the runtime budget loop: today the budget is fully
expressive on its outputs but only one consumer (YOLOX) actually
honours its decisions cross-thread.
