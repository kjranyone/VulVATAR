# Runtime GPU budget policy (P3-03)

Date: 2026-05-20
Roadmap reference: [`gpu-runtime-roadmap-tasks.md` P3-03](./gpu-runtime-roadmap-tasks.md)

## Goal

Consolidate the scattered render / tracking / cloth / FaceMesh cadence
decisions into one `RuntimeGpuBudget` policy module fed by live measurements.
GPU contention (Vulkano render queue + DirectML inference + future compute
cloth + future GPU shared-frame export all sharing one physical adapter)
should be reasoned about and tuned from a single place instead of being
embedded as folklore across `tracking::provider`, `app::render`, and
`renderer::mod`.

## Why

Today the project has at least five independently tuned cadences:

| Knob | Where it lives | Driven by |
|---|---|---|
| Render FPS target | `OutputRouter::set_target_fps` | GUI combo box |
| Pose Hz target | hardcoded in solver / mailbox staleness check | constants |
| YOLOX refresh period | `tracking/yolox.rs` cadence | constants |
| Depth refresh period | `tracking/cigpose_metric_depth.rs` cadence | constants |
| FaceMesh EP choice | `tracking/face_mediapipe.rs` provider selection | startup-time decision |

There is no central place that knows "the GPU is contested right now, drop
YOLOX to every 4th frame and stay on CPU FaceMesh." Today a user who hits
GPU saturation just sees frame drops and worker restarts without any single
authority to ease pressure.

Once compute cloth (P3-02) and GPU shared-frame export (the P2-05 Win32
work) actually use the GPU concurrently with DirectML inference, the
implicit budget will start visibly fighting itself. We want the central
policy in place *before* that contention lands.

## Status

- [x] `RuntimeGpuBudget` struct + module (`src/app/runtime_gpu_budget.rs`)
- [x] Live-measurement intake (render dt EMA, output drops/sec window,
      export pool leased/capacity). Provider dt + GPU export failure
      window are stubs in `RuntimeMeasurements` — the budget reads
      them, the producer needs more plumbing (deferred)
- [x] Degraded-mode definitions (4 levels) with hysteresis (5 s dwell
      Light→Heavy, 30 s recovery dwell per step)
- [x] First production consumer: render FPS clamp via
      `Application::set_user_render_fps` → `budget.render_fps_target()`
      → `OutputRouter::set_target_fps`
- [x] **B4** YOLOX skip period consumer: `YOLOX_REFRESH_PERIOD` is
      now a `pub static AtomicU64` in `tracking::rtmw3d`. The
      `Application::update_runtime_gpu_budget` tick stores the budget's
      `yolox_skip_period` into this atomic each frame;
      `Rtmw3dInference::process_pose` loads it (Relaxed) inside the
      submit guard. Single shared atomic over an Arc plumbing path
      because the value is conceptually global. New test
      `all_modes_emit_nonzero_yolox_skip_period` asserts every mode
      arm produces a non-zero period (review pass 2). `.max(1)` on the
      read site is defence-in-depth.
- [x] **B7** docs/gpu-runtime-roadmap.md "RuntimeGpuBudget — Landed
      State Machine" section: full state-machine table with
      thresholds, per-mode targets, consumer plumbing matrix, and
      rationale notes. EmergencyCpu unreachability documented as
      pending the recent-window GPU-export-failure counter.

## Architecture target

```text
                       ┌────────────────────────────┐
   render dt   ──────► │                            │
   pose dt     ──────► │     RuntimeGpuBudget       │ ─► render FPS clamp
   output      ──────► │                            │ ─► pose Hz target
   queue/pool  ──────► │  (current + smoothed       │ ─► YOLOX skip period
   provider dt ──────► │   measurements +           │ ─► depth skip period
                       │   degraded-mode policy)    │ ─► FaceMesh EP request
                       └────────────────────────────┘
```

The budget is **advisory**: each consumer reads the latest target and
reconciles on its own next tick. No consumer is allowed to ask the budget
to reach back into them — keeps the dependency arrow one-way.

## Compact state

```rust
pub struct RuntimeGpuBudget {
    pub render_fps_target: u32,
    pub pose_hz_target: u32,
    pub yolox_skip_period: u32,
    pub depth_skip_period: u32,
    pub facemesh_ep_preference: FaceMeshEpPreference,
    pub degraded_mode: DegradedMode,
}

pub enum DegradedMode {
    Healthy,
    PressureLight,    // 1+ signals are persistently elevated
    PressureHeavy,    // GPU saturation; aggressive cadence cuts
    EmergencyCpu,     // GPU export failing; fall back to CPU sinks
}
```

The four `DegradedMode` levels are intentionally coarse — fine-grained
auto-tuning is a trap. The user (or a future tuning pass) can override the
mode if the policy gets it wrong.

## Inputs and how they map

| Input | Source | Influences |
|---|---|---|
| `render_dt_ema` | render thread timing | `render_fps_target` clamp |
| `provider_dt_ema` (per provider) | tracking workers | `pose_hz_target`, EP preference |
| `output_queue_depth` | `OutputRouter::diagnostics` | `degraded_mode` escalation |
| `export_pool.leased_slots` | `OutputDiagnostics::export_pool` | `degraded_mode` escalation |
| `dropped_frame_count` delta | `OutputRouter::dropped_count` | `degraded_mode` escalation |
| `cloth_backend` distribution | `ClothSolverBackend` per cloth | informational; future input |

## Degraded-mode transition rules (first cut)

- `Healthy → PressureLight` if any of: render_dt > 1.2 × target, output
  drops/sec > 5, export pool leased ≥ capacity for >1 s.
- `PressureLight → PressureHeavy` if pressure persists > 5 s or
  output drops/sec > 20.
- `PressureHeavy → EmergencyCpu` if GPU export fails (handle creation /
  fence wait) twice within 10 s.
- Every mode auto-recovers to one level lower after 30 s of clean signals.

These thresholds are first guesses; the production tuning pass picks
real numbers after observing soak runs.

## Scope

In:
- new module `src/app/runtime_gpu_budget.rs` (or `src/runtime/budget.rs`
  if a new top-level runtime crate makes sense once more policies land)
- `Application` owns the budget; updates it once per frame after drains
- one or two consumers cut over in the first slice (suggest:
  `OutputRouter::set_target_fps` and the YOLOX skip counter), prove the
  shape, then migrate others incrementally

Out:
- automatic mode escalation tuning — first cut uses round numbers; tuning
  happens after live data
- per-avatar policies — single global budget for now
- writing-to-disk telemetry — diagnostics surface in the inspector first,
  recording can come later

## Tasks

- [x] **B1** define `RuntimeGpuBudget` + `DegradedMode` in
      `src/app/runtime_gpu_budget.rs`
- [x] **B2** per-frame measurement intake on `Application`
      (`update_runtime_gpu_budget(now)`) called from `run_frame`
- [x] **B3** GUI calls `Application::set_user_render_fps`; the budget
      stores user intent and `OutputRouter::set_target_fps` reads the
      (possibly-clamped) `budget.render_fps_target()` each frame
- [ ] **B4** migrate YOLOX skip period (`tracking/yolox.rs`) — value
      is computed and shown in inspector; consumer wiring deferred
      (needs `Arc<AtomicU64>` plumbed Application → TrackingSource →
      PoseProvider → Rtmw3dInference)
- [x] **B5** budget state surfaced in
      `src/gui/inspector/output.rs::draw_output` under "Runtime GPU
      Budget" collapsing header: degraded mode, render target
      (user→clamped), YOLOX skip period, last transition reason
- [x] **B6** unit tests cover all four transitions and recovery
      (9 tests in `runtime_gpu_budget::tests`)
- [ ] **B7** docs update — deferred until thresholds tune against
      live data

## Risks

- **Coupling creep**: every existing consumer wants its own special-case
  input. Keep the budget's surface area narrow — if a consumer needs
  fine-grained control, expose a knob on the consumer itself rather than
  expanding `RuntimeGpuBudget`.
- **Hidden hysteresis bugs**: round-number thresholds plus 30-second
  recovery windows are easy to oscillate. The unit test on the synthetic
  stream needs to cover at least one steady-state and one oscillation
  scenario.
- **GUI dissonance**: if the user sets render FPS to 60 and the budget
  clamps it to 45 under pressure, the inspector must show the active
  value AND the reason (degraded mode).

## Done when

- one render-time consumer and one tracking consumer read from
  `RuntimeGpuBudget` instead of from local constants
- the inspector shows the current `DegradedMode` and the chief cause
- unit tests cover the four mode transitions and at least one steady
  state
- no consumer reaches into another consumer's pacing logic; everything
  routes through the budget
