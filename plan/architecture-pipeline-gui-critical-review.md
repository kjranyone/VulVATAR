# VulVATAR critical architecture / pipeline / GUI review

Date: 2026-05-06
Last progress update: 2026-05-07

## Progress at a glance

The original review (below) was written on 2026-05-06 against a clean
working tree. The following findings have since been addressed; the rest
are still open. Each section under "Findings" carries an explicit
`Status:` line so the next agent doesn't have to cross-reference this
table.

| Finding | Severity | Status (2026-05-07) |
|---------|----------|----------------------|
| #1  Live output is CPU-readback-centered             | Critical | OPEN — needs a real GPU handoff (Phase 6). Phase 1 below has been completed so the runtime no longer *claims* GPU interop. |
| #2  `SharedTexture` misleading name and diagnostic   | Critical | DONE — renamed `FrameSink::SharedTextureFileStub`, diagnostics report actual `OutputFrame.handoff_path`, GUI shows real handoff + CPU fallback warning. |
| #3  Renderer ↔ output bidirectional dependency       | High     | OPEN — `OutputFrame.gpu_image: Option<Arc<vulkano::Image>>` still leaks into `output/`. |
| #4  Render thread backpressure partly accidental     | High     | OPEN — result delivery still uses blocking `result_tx.send`. |
| #5  Rapier branch bypasses spring/cloth toggles      | High     | DONE — `PhysicsWorld::step_all` now takes `SimulationStepOptions`; both branches honour the GUI toggles. |
| #6  Sim clock consumed per avatar, not per frame     | High     | DONE — `Application::sim_clock.advance` runs once per frame and `(fixed_dt, substeps)` are reused; redundant `PhysicsWorld::sim_clock` removed. |
| #7  Stale tracking causes hard pose reset            | High     | OPEN — still treats stale samples as `None`. |
| #8  Tracking mailbox clones heavy GUI data           | Medium   | OPEN. |
| #9  Lower-body provider toggle is a no-op            | Medium   | DONE — relabelled "Drive Leg Bones", dead `prefer_lower_body` plumbing removed end-to-end. |
| #10 GUI state has grown into a broad mutable object  | High     | OPEN. |
| #11 Repaint gating comment doesn't match runtime     | Medium   | DONE — `!self.paused` removed; condition now keys off real activity (tracking / lipsync / load / notifications / clip animation). |
| #12 Renderer remains a very large module             | Medium   | OPEN. |
| #13 Avatar instance carries render-facing primitives | Medium   | OPEN. |

### What changed in code (2026-05-07)

- `src/simulation/mod.rs`
  - New `SimulationStepOptions { spring_enabled, cloth_enabled }`.
  - `PhysicsWorld::step_all(fixed_dt, substeps, avatar, options)` —
    Rapier and non-Rapier paths both gate on `options`.
  - Removed `PhysicsWorld::sim_clock` field + `simulation_clock()`
    accessor (the duplicate clock was the source of #6's per-avatar
    bug). The single source of truth is `Application::sim_clock`.
  - Tests added (`#[cfg(test)] mod tests`):
    - `sim_clock_advance_does_not_regenerate_substeps_within_a_frame`
    - `step_all_with_spring_disabled_does_not_mutate_spring_state`
    - `step_all_with_spring_enabled_does_mutate_spring_state`
- `src/app/render.rs`
  - `sim_clock.advance(frame_dt)` hoisted out of the avatar loop.
  - Constructs `SimulationStepOptions` per avatar (cloth gated by both
    runtime toggle AND `avatar.cloth_enabled`).
- `src/output/frame_sink.rs`
  - `FrameSink::SharedTexture` → `FrameSink::SharedTextureFileStub`.
  - Dead `SharedTextureSink` struct deleted.
  - Header comment on `SharedMemoryFileSink` rewritten so it stops
    claiming GPU interop.
- `src/output/mod.rs`
  - `OutputRouter.last_handoff_path: Option<HandoffPath>` tracked on
    publish.
  - `OutputDiagnostics.active_handoff_path` is now `Option<HandoffPath>`
    derived from the most recent published frame, not the sink variant.
  - Test added: `diagnostics_handoff_path_follows_published_frames`.
- `src/gui/inspector/output.rs`
  - Synchronization + Diagnostics sections read live `diagnostics()`
    and colour-code GPU vs CPU readback vs SharedMemory; surface a
    "CPU readback fallback in use" warning when active.
- `src/gui/mod.rs`
  - Repaint gate: `!self.paused` removed; replaced with
    `animation_playing` (any avatar has `active_clip`) plus the
    existing tracking/lipsync/load/notification flags.
- `src/tracking/mod.rs`, `src/app/tracking_lifecycle.rs`,
  `src/gui/inspector/tracking.rs`
  - Dropped the `_full` suffixes and the dead `prefer_lower_body`
    parameter chain (the value was `let _ = prefer_lower_body;`).
  - Old `start` / `start_with_params` / `start_tracking` /
    `start_tracking_with_params` no-op wrappers deleted.
- `locales/{en,ja,ko,zh}.yml`
  - `sink_shared_texture` labelled "(file stub)" / "(ファイルスタブ)".
  - New keys: `handoff_active_gpu`, `handoff_active_cpu_readback`,
    `handoff_active_shared_memory`, `handoff_pending`,
    `handoff_fallback_warning`, `connection_pending` (en + ja
    translated; ko/zh keep English placeholders matching prior policy
    in those files).
  - `tracking.lower_body` and `lower_body_tooltip` rewritten to
    describe the actual solver behaviour, not model selection.

`cargo test`: 165 passed / 1 ignored / 0 failed (was 161 / 1 / 0).

### Re-review of follow-up fixes (2026-05-07, seventh pass)

A seventh-pass review caught one more depth-pipeline gap and a stale
comment. Both fixed:

- **Depth-aware re-calibration was blocked by the old persisted
  calibration's plausibility band.** The previous passes routed
  `calibration_mode_hint` into anchor selection, but
  `Rtmw3dWithDepthProvider::estimate_pose_internal` still passed
  `self.pose_calibration.as_ref()` directly into `calibrate_scale`
  for the depth-plausibility gate (`anchor_depth_m ± max_deviation`).
  When the user re-captures from a different distance than the saved
  calibration, every frame trips `PlausibilityRejected`, the EMA
  starves out, and the recapture session falls through to the
  RTMW3D synthetic-z fallback — meaning a fresh depth-aware
  calibration cannot be produced. Now the provider passes `None` to
  `calibrate_scale` whenever `calibration_mode_hint` is `Some`
  (modal open), suppressing the gate during recapture; `resolve_origin_metric`
  / `build_skeleton` still receive the persisted calibration (they
  use it for orthogonal channels — torso depth template, shoulder
  span — that don't depend on recapture distance). Once the modal
  closes and `persist_calibration` writes the new value, the gate
  re-engages with the updated band.
  Files: `src/tracking/rtmw3d_with_depth.rs`.

- **`set_calibration` comment still described the inner RTMW3D as
  `force_shoulder_anchor || hint`.** The inner inference's
  `process_pose` was rewritten to use override semantics three passes
  ago; the wrapper provider's mirror-comment never caught up. Reads
  as a contradiction to anyone tracing the contract from the wrapper
  inwards. Comment now spells out the inner override match
  (`Some(UpperBody) → true`, `Some(FullBody) → false`,
  `None → persisted`) and explains why the mirror is needed for the
  `None` arm.
  Files: `src/tracking/rtmw3d_with_depth.rs`.

### Re-review of follow-up fixes (2026-05-07, sixth pass)

A sixth-pass review caught one more anchor-contract gap. Fixed:

- **`Rtmw3dWithDepthProvider` fallbacks bypassed the anchor sync.**
  The wrapper provider stored `pose_calibration` and
  `calibration_mode_hint` on itself and passed them into
  `build_options_from_calibration` for the depth-built skeleton — but
  the two fallback paths (`return rtmw_est;` on RTMW3D keypoint
  shortage, and the same on DAv2 cold-start scale-calibration
  failure) return the inner `Rtmw3dInference::estimate_pose` output
  unchanged. That inner inference builds its source skeleton off its
  *own* `force_shoulder_anchor || hint` flags, which were never
  written by the wrapper. Result: under fallback the first
  `UpperBody` capture still hallucinated hip, and the
  `UpperBody → FullBody` re-calibration was still impossible — the
  exact bugs the previous passes were supposed to close. Fixed by
  mirroring both writes in
  `Rtmw3dWithDepthProvider::set_calibration` /
  `set_calibration_mode_hint` onto the inner `Rtmw3dInference` so
  every code path inside the wrapper sees the same anchor contract.
  Files: `src/tracking/rtmw3d_with_depth.rs`.

### Re-review of follow-up fixes (2026-05-07, fifth pass)

A fifth-pass review caught two more issues. Both fixed:

- **Calibration-mode hint stuck on after the modal closed.** The
  hint sync in `update()` ran *before* `draw_modal`. Modal
  transitions caused by Cancel / Esc / Done auto-close happen
  *inside* `draw_modal`, so by the time the sync next ran the modal
  was already `Closed` — except the next `update()` would only fire
  if some other repaint reason (animation, render-in-flight, dirty)
  was set. On a static avatar with no autosave pending, that meant
  the worker kept the previously-pushed `Some(UpperBody|FullBody)`
  hint forever, overriding the persisted calibration past the user's
  cancel. Extracted the sync into `GuiApp::sync_calibration_mode_hint`
  and call it twice per `update`: once before `run_frame` (catches
  Idle-state mode-button switches) and once after `draw_modal`
  (catches the close transitions).
  Files: `src/gui/mod.rs`.

- **`Rtmw3dWithDepthProvider` call-site comment still read OR-ish.**
  The depth provider's `process_pose` comment described the hint as
  "set when either the persisted calibration is `UpperBody` or the
  open calibration modal currently has `UpperBody` selected" — that's
  the OR semantics that the v1 implementation had. The v2 override
  semantics (and the current code) need both directions. Comment
  rewritten to spell out the override contract and point at
  `build_options_from_calibration` for the canonical statement.
  Files: `src/tracking/rtmw3d_with_depth.rs`.

### Re-review of follow-up fixes (2026-05-07, fourth pass)

A fourth-pass review caught the deferred depth-provider work plus a
stale trait comment. Both fixed:

- **Depth providers now honour `set_calibration_mode_hint`.** Both
  `Rtmw3dWithDepthProvider` and `CigposeMetricDepthProvider` gained a
  `calibration_mode_hint: Option<CalibrationMode>` field, an
  override-semantics impl of the trait method, and pass the hint into
  `build_options_from_calibration` alongside the persisted calibration.
  `build_options_from_calibration` itself was changed to take both
  arguments and apply the same override logic as RTMW3D
  (`Some(UpperBody)` → force, `Some(FullBody)` → don't force,
  `None` → fall back to persisted). The deferred TODO from the
  second-pass review is now resolved end-to-end.
  Files: `src/tracking/skeleton_from_depth.rs`,
  `src/tracking/rtmw3d_with_depth.rs`,
  `src/tracking/cigpose_metric_depth.rs`.

- **Trait comment for `set_calibration_mode_hint` aligned with the
  override contract.** The doc still said "treat the hint as an OR
  with whatever the confirmed calibration says" — true for the v1
  bool implementation, false for the v2 `Option<CalibrationMode>`
  override implementation. Future `PoseProvider` implementors
  reading the trait would have re-introduced the
  `UpperBody → FullBody re-calibration impossible` bug. Comment now
  spells out the override semantics in both directions.
  Files: `src/tracking/provider.rs`.

### Re-review of follow-up fixes (2026-05-07, third pass)

A third-pass review found two more issues from the previous round.
Both fixed:

- **Calibration mode hint couldn't override a stale persisted
  `force_shoulder_anchor`.** The hint was a `bool`, OR'd with the
  persisted flag at the call site. That meant `Some(FullBody)` could
  not clear `force_shoulder_anchor=true` left over from a previous
  `UpperBody` calibration — re-calibrating from `UpperBody` to
  `FullBody` was impossible because the GUI's
  `pose.root_anchor_is_hip` gate kept rejecting every sample. Changed
  the field to `Option<CalibrationMode>` and applied **override**
  semantics in `process_pose`: `Some(UpperBody) → true`,
  `Some(FullBody) → false`, `None → persisted`.
  Files: `src/tracking/rtmw3d/mod.rs`, `src/tracking/provider.rs`.

- **Repaint gate's autosave-throttle wake-up delayed render-input
  changes by up to 250 ms.** The previous fix used
  `request_repaint_after(throttle_remaining)` for `project_dirty` so
  the gate didn't hot-loop. But that meant a checkbox flip on a
  static avatar — where `render_in_flight` resolved within the same
  frame — wouldn't repaint until the autosave wake-up fired up to
  ~250 ms later, defeating the immediate-feedback intent. Moved
  `project_dirty` / `profiles_dirty` into the immediate-repaint arm
  (`ctx.request_repaint()`). The autosave throttle still lives in
  `autosave_tick` and governs disk writes; the gate burns ~15 frames
  per click waiting for the save to land, which is acceptable.
  Pathological case (permanently failing autosave) burns frames at
  the display rate, documented and accepted.
  Files: `src/gui/mod.rs`.

### Re-review of follow-up fixes (2026-05-07, second pass)

A second-pass review of the follow-ups found three more bugs in the
code that landed during the first follow-up pass. All fixed:

- **`render_results_pending` leaked when paused or with no avatars.**
  The drain loop was nested inside `if !self.avatars.is_empty()` and
  the whole `run_frame` was gated by `if !self.paused`. Either of
  those conditions becoming false between submit and result would pin
  the counter > 0, which then pinned the new `render_in_flight` repaint
  gate at full rate forever. Split the drain into a public
  `Application::drain_render_results()` and call it from
  `GuiApp::update()` even on the paused path.
  Files: `src/app/render.rs`, `src/gui/mod.rs`.

- **`render_results_pending` leaked on render error.** The render
  thread's `RenderFrame` arm did `continue` on `renderer.render(&input)`
  failure, never sending a result. Submit had already incremented the
  counter, so every render error permanently inflated the counter and
  pinned the GUI to full repaint. Now sends a sentinel
  `RenderResult { exported_frame: None, .. }`; `process_render_result`
  already handled the `None` arm gracefully (clears `rendered_pixels`,
  returns), so the in-flight balance is preserved without surfacing a
  fake successful frame to the output worker.
  Files: `src/app/render_thread.rs`.

- **`request_repaint_after(0)` is not the same as `request_repaint()`,
  and `profiles_dirty` has no autosave throttle.** Original gate did
  `request_repaint_after(throttle - elapsed)` for both
  `project_dirty` *and* `profiles_dirty`. (a) `profiles_dirty` is
  saved immediately — no throttle — so it shouldn't wait up to 250 ms
  behind the project deadline. (b) When `elapsed >= throttle` the
  remaining `Duration::ZERO` schedules egui's idle wakeup (potentially
  seconds), not the next frame. Now: `profiles_dirty` always calls
  `request_repaint()` immediately; `project_dirty` falls back to
  `request_repaint()` when its remaining throttle is zero.
  Files: `src/gui/mod.rs`.

#### Deferred from the second-pass review

(The "depth providers don't honour `set_calibration_mode_hint`" item
that previously sat here was picked up in the fourth-pass review
above and is now done.)

- **`CalibrationModalState::active_mode()` returns `Some` in `Idle`.**
  This means opening the modal in `UpperBody` to read instructions
  immediately suppresses hip-anchor framing on the *live* tracking
  feed, even before the user hits Start. That can be argued either
  way (it's a "preview the mode" UX), so the trade-off is documented
  rather than reverted. If a user complains the avatar's lower body
  collapses the moment they open the modal, narrow `active_mode()` to
  the `WaitingForPose` / `Collecting` states only.

### Follow-up review fixes (2026-05-07)

A self-review of the changes above surfaced three additional issues
that have also been fixed:

- **Output diagnostics over-claimed delivery.** `OutputRouter::publish`
  was updating `last_publish_timestamp` / `last_handoff_path`
  unconditionally before the local queue / `try_send` step, so a
  saturated worker channel still showed up as "connected". Moved the
  commit so it only fires inside the `Ok(())` arm of `try_send`. Test
  updated to retry-until-observed so `sync_channel(1)` saturation
  doesn't race the assertion.
  Files: `src/output/mod.rs`.

- **Repaint gate stalled GUI mutations on a static avatar.** The Phase
  #11 fix dropped `!self.paused` but kept too narrow a set of activity
  signals — a checkbox flip on an idle avatar produced one input-driven
  frame, then the render result and any `project_dirty` autosave
  stalled until the next user input. Added `Application::has_pending_render_result()`
  (counter incremented on a successful `RenderThread::submit` and
  decremented per drained `RenderResult`), included it plus
  `calibration_modal.is_open()` in the gate, and used
  `request_repaint_after(throttle_remaining)` for `project_dirty` /
  `profiles_dirty` so an idle dirty save still hits disk within
  `AUTOSAVE_THROTTLE`. `RenderThread::submit` now returns `bool` so
  callers can know whether the command actually queued.
  Files: `src/app/mod.rs`, `src/app/render.rs`, `src/app/render_thread.rs`,
  `src/gui/mod.rs`.

- **First-pass UpperBody calibration could collect zero samples.**
  (Pre-existing issue in user's WIP; surfaced during the review of
  the Finding #11 work.) `Rtmw3dPoseProvider::set_calibration` only
  fires *after* `persist_calibration`, so until the first capture
  succeeds the provider's `force_shoulder_anchor` stays `false`.
  Combined with the GUI's `pose.root_anchor_is_hip == false`
  collection gate in `refresh.rs`, every sample was being dropped
  whenever the model hallucinated a confident hip — making the
  *first* `UpperBody` capture impossible by construction. Added a
  transient mailbox channel `pending_calibration_mode_hint` plus a
  new `PoseProvider::set_calibration_mode_hint` trait method (default
  no-op). The GUI pushes the hint on edges from `update()` based on
  `CalibrationModalState::active_mode()`; the worker forwards it; the
  RTMW3D provider applies it as an **override** of the persisted
  `force_shoulder_anchor` (`Some(UpperBody)` → true,
  `Some(FullBody)` → false, `None` → persisted) so re-calibrating
  in *either* direction works. (The third-pass review caught a
  bug where the hint was originally a `bool` and could only set the
  flag, not clear it — see "third pass" section below.)
  Files: `src/tracking/mod.rs`, `src/tracking/provider.rs`,
  `src/tracking/rtmw3d/mod.rs`, `src/gui/calibration/state.rs`,
  `src/gui/mod.rs`.

### Suggested order for the remaining work

1. **#7 stale tracking policy** — most user-visible runtime issue still
   open; isolated to `app::run_frame` + a small bit of state.
2. **#3 + #4 contracts and backpressure** — these clean up the
   renderer/output boundary and unblock #1 (Phase 6 GPU handoff).
3. **#1 real GPU handoff** — the big one. Needs the boundary cleanup
   in #3 to not paint itself into a corner.
4. **#10 GUI state split** — large mechanical refactor; do after the
   runtime issues so the split lands on a stable contract.
5. **#8, #12, #13** — incremental cleanups, can be picked up any time.

## Scope

This note is a critical review of the current VulVATAR application from three
angles:

- architecture and subsystem boundaries
- inference / retargeting / simulation / output pipeline
- GUI state ownership and user-facing workflow

No code changes were made for this review.

Validation commands run:

```powershell
cargo test --no-run -q
cargo test -q
```

Result:

- `cargo test --no-run -q`: passed
- `cargo test -q`: passed, `161 passed`, `1 ignored`

The fact that tests pass should not be read as architectural health. The main
risks below are runtime behavior, performance, maintainability, and misleading
contracts rather than type-level breakage.

## Executive Summary

The documents describe a clean design: app thread owns scene state, renderer
owns GPU resources, tracking publishes latest normalized input, and output
consumes explicit frame tokens. The implementation is moving in that direction
in some places, but several important runtime paths already contradict that
architecture.

The highest-risk issue is output. The design says GPU-resident handoff is the
preferred path and CPU readback is a fallback. The current app hardcodes
`RenderExportMode::CpuReadback` for live frames and then writes CPU pixel data
to shared-memory/file-backed sinks. The `SharedTexture` sink name and output
diagnostics imply GPU interop, but the implementation is still a CPU file
protocol.

The second highest-risk issue is state ownership. `Application` and `GuiApp`
have become broad coordination objects. They are not just shells; they hold
runtime state, user intent, persistence shadows, render settings, output
lifecycle, tracking lifecycle, calibration, and GUI-only widget buffers. This
will make future changes brittle unless the state model is tightened.

The third highest-risk issue is behavior mismatch. Some user-facing toggles
and diagnostics do not match runtime behavior: `SharedTexture` is not a shared
texture, Rapier-enabled simulation can bypass spring/cloth toggles, stale
tracking samples can snap back to rest pose rather than fade or hold, and the
GUI repaint gating still repaints continuously when unpaused.

## Findings

### 1. Live output is CPU-readback-centered despite the design

Severity: Critical

Status (2026-05-07): **OPEN.** No real GPU handoff yet — every live
frame still goes through `RenderExportMode::CpuReadback`. Phase 1 of
the repair plan has been completed so the system no longer *advertises*
GPU interop falsely (see #2), but the underlying CPU-bound path is
unchanged. Phase 6 of the repair order is the work item.

Design intent:

- `docs/output-interop.md`: GPU-resident handoff should be preferred.
- CPU readback should be a fallback path.

Current implementation:

- `src/app/render.rs:494` sets live frame export to
  `RenderExportMode::CpuReadback`.
- `src/renderer/mod.rs:1197` to `src/renderer/mod.rs:1252` records
  `copy_image_to_buffer`, then stores a pending readback fence and buffer.
- `src/app/render.rs:195` to `src/app/render.rs:214` forwards CPU pixel data
  to the output frame when available.

Problem:

Every live output frame goes through GPU-to-CPU transfer and CPU-side shared
memory/file output. At 1920x1080 RGBA this is about 8 MB per frame before any
copies inside the sink. At 60 fps that is roughly 500 MB/s of raw frame data,
not counting staging, CPU cache effects, file-backed mapping writes, or GUI
texture upload.

This directly conflicts with the stated architecture and is likely to become
the dominant performance cost once real avatars, morphs, cloth, and inference
run together.

Recommendation:

1. Introduce a real live-output mode selection:
   - preview-only CPU readback for egui viewport
   - GPU token path for OBS / virtual-camera-capable sinks
   - explicit CPU fallback when GPU interop is unavailable
2. Make `OutputTargetRequest.export_mode` derive from the selected sink and
   platform capability, not be hardcoded in `Application::build_frame_input_multi`.
3. Split preview readback from output handoff. The GUI may still need CPU pixels
   for egui, but the OBS path should not inherit that constraint.

### 2. `SharedTexture` is currently a misleading name and diagnostic

Severity: Critical

Status (2026-05-07): **DONE.** Variant renamed
`FrameSink::SharedTextureFileStub`; legacy `SharedTextureSink` struct
deleted; `SharedMemoryFileSink` header comment rewritten to admit it is
a CPU-only file dump. `OutputRouter` tracks the most recently published
frame's `handoff_path` and `OutputDiagnostics.active_handoff_path` is
now `Option<HandoffPath>` derived from that, not from the sink variant.
The Synchronization panel surfaces the real path with a warning row
when CPU readback fallback is active. Locale labels updated. Regression
test:
`output::tests::diagnostics_handoff_path_follows_published_frames`.
Follow-up (also DONE): `last_publish_timestamp` /
`last_handoff_path` are now committed only after a successful
`worker_tx.try_send`, so a saturated worker channel no longer paints
the GUI as "connected / GPU active".

Current implementation (pre-fix, kept for context):

- `src/output/frame_sink.rs:9` exposes `FrameSink::SharedTexture`.
- `src/output/frame_sink.rs:1014` maps it to `SharedMemoryFileSink::new()`.
- `src/output/frame_sink.rs:548` labels that writer as a GPU shared texture
  file protocol, but `write_frame` requires `frame.pixel_data`.
- `src/output/mod.rs:373` to `src/output/mod.rs:384` reports
  `SharedTexture` as `GpuSharedFrame`.

Problem:

The user and diagnostics can believe GPU interop is active when the sink is
actually a CPU byte stream. This is worse than an unimplemented feature because
it creates false confidence during profiling and QA.

Recommendation:

- Rename the current sink to something honest, such as `SharedTextureFileStub`
  or `SharedMemoryTextureProtocol`, until it exports a real GPU handle.
- Make diagnostics report the actual per-frame `OutputFrame.handoff_path`, not
  infer it from the selected sink name.
- Treat `SharedTexture` as unavailable unless `OutputFrame.gpu_token` contains
  a valid external handle and sync contract.

### 3. Renderer and output depend on each other's internals

Severity: High

Status (2026-05-07): **OPEN.** `OutputFrame.gpu_image:
Option<Arc<vulkano::image::Image>>` still exposes a Vulkano type to
`output/`, and `output_export.rs` still imports
`crate::output::HandoffPath`. The bidirectional dependency is unchanged.

Current implementation:

- `src/renderer/output_export.rs:43` stores `crate::output::HandoffPath`.
- `src/output/mod.rs:77` stores `Option<Arc<vulkano::image::Image>>`.

Problem:

The design says renderer owns GPU resources and output consumes exported frame
tokens. The code has a bidirectional conceptual dependency:

- renderer knows output-level handoff enum details
- output knows Vulkano image types

This makes it hard to enforce lifetime, synchronization, and platform-specific
handoff policy. It also makes future non-Vulkano output tests harder.

Recommendation:

- Move shared frame contract types into a neutral module, for example
  `src/frame_handoff.rs` or `src/interop.rs`.
- Replace `OutputFrame.gpu_image: Option<Arc<Image>>` with an opaque token that
  includes resource id, handle type, format, extent, color space, alpha mode,
  sync primitive, and lifetime policy.
- Keep raw `vulkano::Image` private to renderer code.

### 4. Render thread backpressure is partly explicit, partly accidental

Severity: High

Status (2026-05-07): **OPEN.** Result delivery still uses blocking
`result_tx.send(result)`. Unchanged.

Current implementation:

- `src/app/render_thread.rs:47` uses `sync_channel::<RenderCommand>(2)`.
- `src/app/render_thread.rs:79` uses `try_send`, dropping commands when full.
- `src/app/render_thread.rs:153` uses blocking `result_tx.send(result)`.

Problem:

Command submission is latest-ish and non-blocking, but result submission can
block the render thread if the app thread stops draining. In normal operation
the app drains results after submit, but modal loops, long UI work, asset load
polling, or future code can change timing. This is an implicit stall point.

Recommendation:

- Make result delivery also use explicit latest-frame semantics:
  `sync_channel(1)` + `try_send` and replace/drop stale result.
- Or use a mailbox (`Arc<Mutex<Option<RenderResult>>>`) so the app always reads
  the newest completed frame.
- Track dropped render results in diagnostics.

### 5. Rapier-enabled branch bypasses spring / cloth toggles

Severity: High

Status (2026-05-07): **DONE.** `PhysicsWorld::step_all` now takes
`SimulationStepOptions { spring_enabled, cloth_enabled }` and gates the
spring / cloth integrators on those fields, so the GUI toggles work the
same with or without Rapier initialised. Regression tests:
`step_all_with_spring_disabled_does_not_mutate_spring_state` and
`step_all_with_spring_enabled_does_mutate_spring_state`. (Cloth-disabled
test is implicit via the existing `step_cloth_does_nothing_when_avatar_cloth_disabled`
plus the new gate; a dedicated test was not added because building a
full cloth state in a unit test is heavy. If a regression appears,
extend the existing avatar test fixture.)

Current implementation:

- `src/app/render.rs:107` checks `self.physics.rapier_initialized()`.
- If true, it calls `self.physics.step_all(frame_dt, avatar)`.
- `src/simulation/mod.rs:121` to `src/simulation/mod.rs:131` always steps
  spring, cloth, and Rapier inside `step_all`.

Problem:

Default features include `rapier`. `PhysicsWorld::attach_avatar` initializes
Rapier when the feature is enabled. After an avatar is attached, GUI toggles for
spring and cloth can be bypassed because the Rapier branch no longer checks
`RuntimeToggles`.

Recommendation:

- Pass `RuntimeToggles` or a smaller `SimulationStepOptions` into physics.
- Keep fixed-timestep calculation outside the Rapier/non-Rapier branch.
- Ensure tests cover:
  - spring disabled means no spring mutation
  - cloth disabled means no cloth mutation
  - same behavior with and without Rapier initialized

### 6. Simulation clock is consumed per avatar, not per frame

Severity: High

Status (2026-05-07): **DONE.** `Application::sim_clock.advance(frame_dt)`
is hoisted out of the avatar loop in `src/app/render.rs`; the resulting
`(fixed_dt, substeps)` pair is reused for every avatar. The duplicate
`PhysicsWorld::sim_clock` field (different `fixed_dt` of 1/120 s,
unconditionally `reset()` after each `step_all`) was removed entirely
so there is one source of truth. Regression test:
`sim_clock_advance_does_not_regenerate_substeps_within_a_frame`.

Current implementation:

- `src/app/render.rs:77` loops over avatars.
- `src/app/render.rs:111` calls `self.sim_clock.advance(frame_dt)` inside the
  loop.

Problem:

`SimulationClock` is global to the application, but it is advanced once per
avatar. With multiple avatars, the first avatar may consume the accumulated
substeps and later avatars may get fewer or zero substeps. This also makes
behavior dependent on avatar order.

Recommendation:

- Advance the simulation clock once per frame before the avatar loop.
- Reuse the computed `substeps` for every avatar.
- If per-avatar simulation clocks are desired later, store them on the avatar
  or simulation instance explicitly.

### 7. Stale tracking samples can cause hard pose reset

Severity: High

Status (2026-05-07): **OPEN.** Unchanged. Likely the highest-value
remaining user-visible fix.

Current implementation:

- `src/app/render.rs:42` treats stale tracking as `None`.
- `src/app/render.rs:78` rebuilds base/rest pose every frame.
- Solver is skipped when tracking is `None`.

Problem:

When tracking temporarily stalls, the avatar can snap toward rest pose instead
of holding or fading from the latest valid pose. The design documents call for
latest-valid reuse and graceful fallback.

Recommendation:

- Add a latched tracking sample with age metadata.
- Use policy states:
  - fresh: solve normally
  - stale but within hold window: reuse last sample with decaying confidence
  - expired: fade from last solved pose toward animation/rest
- Do not let one stale frame invalidate all pose channels.

### 8. Tracking mailbox clones relatively heavy GUI data under one mutex

Severity: Medium

Status (2026-05-07): **OPEN.** Unchanged.

Current implementation:

- `src/tracking/mod.rs:132` stores pose, webcam preview frame, annotation,
  calibration, and torso template in one `Arc<Mutex<TrackingMailboxInner>>`.
- `src/tracking/mod.rs:248` snapshots and clones pose, frame, and annotation
  under that lock.
- `src/tracking/mod.rs:895` publishes a downscaled webcam frame every inference
  frame.

Problem:

The mailbox is simple and correct enough for now, but it mixes hot pose data,
GUI preview pixels, diagnostics, and calibration side channels. GUI preview can
increase lock time and allocation pressure for the same mutex the app uses to
read pose.

Recommendation:

- Split into separate mailboxes:
  - pose mailbox
  - camera preview mailbox
  - diagnostics/error mailbox
  - calibration command mailbox
- Use `Arc<Vec<u8>>` for preview frame payloads if snapshots continue to clone
  them.

### 9. Lower-body provider toggle is exposed but effectively no-op

Severity: Medium

Status (2026-05-07): **DONE.** UI relabelled "Drive Leg Bones" with a
tooltip that describes the actual solver behaviour and explicitly notes
"the underlying pose model is unchanged". The dead `prefer_lower_body`
parameter chain was removed end-to-end (`TrackingWorker::start_with_params_full`,
`Application::start_tracking_with_params_full`, `worker_loop`,
`run_webcam`); the no-op `let _ = prefer_lower_body;` is gone. Two
no-caller `start*` wrappers were deleted along the way.

Current implementation (pre-fix):

- GUI exposes lower-body tracking state.
- `src/tracking/mod.rs:797` says `prefer_lower_body` has been a no-op since
  the move to whole-body 3D models.

Problem:

User-facing controls that do not alter the active pipeline erode trust. The
toggle may still affect solver behavior in `FrameConfig`, but it no longer
selects a different provider/model as the comments imply.

Recommendation:

- Rename the UI control to what it actually does, such as `Drive leg bones`.
- Remove provider/model-selection wording.
- If model/provider selection is still desired, expose `VULVATAR_POSE_PROVIDER`
  or a real runtime provider selector.

### 10. GUI state has grown into a broad mutable object

Severity: High

Status (2026-05-07): **OPEN.** Unchanged. The
`output.rs` inspector did get a small piece of "read live diagnostics
instead of mutating GuiApp" treatment as part of #2, which is a sliver
of the same refactor; the broader split has not been attempted.

Current implementation:

- `src/gui/mod.rs:343` starts `GuiApp`.
- It contains project state, app runtime, persistence flags, notifications,
  tracking widgets, rendering widgets, output widgets, calibration textures,
  lipsync state, cloth authoring buffers, thumbnail jobs, folder watchers,
  viewport drag state, avatar library controls, and more.
- `src/gui/mod.rs:1007` to `src/gui/mod.rs:1243` handles input, hotkeys,
  polling, syncing GUI state into application, running the frame, autosave,
  modal rendering, panels, viewport, toasts, and repaint policy.

Problem:

This is a maintainability risk. It is hard to tell whether a field is:

- persistent project state
- persistent profile state
- runtime reality
- requested user intent
- GUI-only temporary buffer
- async job handle
- diagnostic cache

The project already has bugs that come from requested-vs-active confusion in
the output/lipsync areas. Without stronger boundaries, more of these will
appear.

Recommendation:

- Introduce explicit state groups:
  - `ProjectUiModel`
  - `RuntimeStatus`
  - `TrackingPanelState`
  - `RenderingPanelState`
  - `OutputPanelState`
  - `CalibrationUiState`
  - `LibraryUiState`
- Keep `GuiApp::update` as an orchestration method only.
- Move "sync GUI to app" into small reconciliation functions with tests.

### 11. Repaint gating comment does not match runtime behavior

Severity: Medium

Status (2026-05-07): **DONE.** `!self.paused` removed from the gate
(it was the entire reason the gate ran continuously in the default
state). Replaced with `animation_playing` (any avatar carries an
`active_clip`) so the existing intent — repaint only when something
visibly changes tick-to-tick — actually holds. Comment rewritten to
match. Note: "output sink actively publishing" was deliberately *not*
added because there is no user-facing output disable today (output is
hardcoded `output_enabled: true` in `build_frame_input_multi`); revisit
when that lands.
Follow-up (also DONE): the first version of the fix was *too*
narrow. A static avatar with no animation could mutate state through a
slider drag and have the result render-pipeline stall (no follow-up
frame to drain the `RenderResult`) and the autosave timer never tick
(no condition was forcing repaint while `project_dirty`). Added
`Application::has_pending_render_result()` (counter incremented on a
successful `RenderThread::submit`, decremented per drained result),
included it plus `calibration_modal.is_open()` in the immediate gate,
plus `project_dirty` / `profiles_dirty` (after the third-pass review,
the original `request_repaint_after(throttle_remaining)` scheduling
turned out to delay render-input changes by up to 250 ms; the
autosave throttle now lives only in `autosave_tick`).
`RenderThread::submit` now returns `bool` so the caller knows whether
to bump the in-flight counter.

Current implementation (pre-fix):

- `src/gui/mod.rs:1224` explains that repaint is requested only when something
  animates.
- `src/gui/mod.rs:1235` includes `|| !self.paused`.

Problem:

The app requests continuous repaint whenever it is not paused. That means idle
CPU/battery savings described by the comment mostly do not happen in the
default state.

Recommendation:

- Decide whether "unpaused" should mean "continuous render loop" even without
  tracking/avatar/output activity.
- If not, change the condition to actual activity:
  - tracking active
  - animation playing
  - lipsync active
  - avatar load job active
  - notifications visible
  - output sink actively publishing
- Update the comment to match the chosen policy.

### 12. Renderer remains a very large module

Severity: Medium

Status (2026-05-07): **OPEN.** Unchanged.

Current implementation:

- `src/renderer/mod.rs` is roughly 2364 lines.
- It owns initialization, pipeline selection, render pass, resource caches,
  readback, thumbnails, debug-ish paths, texture resolution, mesh upload,
  morph/cloth vertex generation, and placeholder geometry.

Problem:

The renderer API boundary is reasonably narrow from app to renderer, but the
implementation module has too many reasons to change. This makes regressions
likely when touching color space, thumbnail rendering, output export, or mesh
upload.

Recommendation:

- Split implementation by execution path:
  - `device.rs` or `context.rs`
  - `frame.rs`
  - `readback.rs`
  - `draw.rs`
  - `thumbnail.rs` already exists but still shares substantial code in `mod.rs`
  - `mesh_cache.rs`
  - `texture_cache.rs`
- Keep `VulkanRenderer` as the facade.

### 13. Avatar instance stores duplicated render-facing primitive data

Severity: Medium

Status (2026-05-07): **OPEN.** Unchanged.

Current implementation:

- `src/avatar/instance.rs:39` stores `primitive_arcs`.
- `src/avatar/instance.rs:125` clones primitives into `Arc<MeshPrimitiveAsset>`.
- `src/app/render.rs:417` to `src/app/render.rs:431` forwards those arcs into
  renderer snapshots.

Problem:

This is a pragmatic optimization to avoid cloning vertex data every frame, but
it means avatar runtime now carries renderer-snapshot optimization data. It is
not a GPU type leak, but it blurs the asset/runtime/render boundary.

Recommendation:

- Move primitive arc/cache ownership into immutable asset construction, not
  avatar instance runtime.
- Or create a `RenderAssetView` built once per loaded `AvatarAsset` and shared
  by render snapshot construction.

## Architecture Assessment

### What is strong

- The high-level design documents are coherent and mostly point in the right
  direction.
- `RenderFrameInput` is a useful snapshot boundary.
- Tracking provider abstraction exists and supports multiple pipelines.
- Tracking inference is off the GUI/app thread.
- Output has an explicit worker and bounded-ish intent.
- Project persistence, requested-vs-active state, and recovery have started to
  receive serious attention.

### What is weak

- The output path has diverged furthest from the architecture.
- The app shell is becoming a service locator and state bag.
- GUI controls and diagnostics can misrepresent actual runtime paths.
- The simulation step order is mostly right, but fixed timestep ownership is
  not robust for multi-avatar.
- Some "Phase" comments describe planned or partial plumbing rather than
  completed behavior; this makes the code read more finished than it is.

## Pipeline Review

### Current high-level frame path

1. GUI updates `Application` fields.
2. `Application::run_frame` handles input update.
3. App checks tracking mailbox freshness.
4. App rebuilds base pose for every avatar.
5. App applies tracking pose if available.
6. App steps simulation.
7. App builds skinning matrices.
8. App builds `RenderFrameInput`.
9. App submits render command to render thread.
10. Render thread renders and schedules CPU readback.
11. Next render harvests previous readback.
12. App publishes CPU pixels to output worker.
13. GUI uploads CPU pixels to egui texture.

### Main critique

The pipeline is functional for a POC, but it serializes too much around CPU
pixels. The renderer produces an image, then the system immediately turns it
into CPU bytes for both preview and output. That makes the GUI preview path
dictate the OBS path.

The better shape is:

```text
renderer
  -> GPU output token for external consumers
  -> optional low-priority CPU preview/readback for egui
```

Preview and broadcast output should be siblings, not the same CPU payload.

## GUI Review

### Current strengths

- The application has clear modes: avatar, preview, tracking, rendering,
  output, cloth authoring, settings.
- The right inspector changes by mode.
- There are visible diagnostics for tracking and output.
- The app exposes calibration and model-library workflows rather than hiding
  everything in logs.

### Main critique

The GUI is structurally convenient but not stable. Most panels mutate `GuiApp`
directly, and `GuiApp::update` then reconciles many of those fields into
`Application` every frame. That pattern is easy to start with and hard to keep
correct as runtime failures and requested/active distinctions accumulate.

The GUI should become less like "the owner of all mutable state" and more like:

- persistent model editor
- command issuer
- runtime status viewer
- temporary widget state owner

Those are different roles and should not all share one large struct.

## Recommended Repair Order

### Phase 1: Make output truthfully represented — DONE (2026-05-07)

1. ~~Rename or relabel the current `SharedTexture` path so it does not claim GPU
   interop.~~ — `FrameSink::SharedTextureFileStub`.
2. ~~Change diagnostics to report actual `OutputFrame.handoff_path`.~~ —
   `OutputDiagnostics.active_handoff_path: Option<HandoffPath>` from the
   most recently published frame.
3. ~~Add a warning/status row when CPU readback fallback is active.~~ —
   `inspector.handoff_fallback_warning` shown in the Synchronization
   panel when `fallback_active`.
4. ~~Ensure `SharedTexture` cannot be selected as a real GPU path until a real
   exported handle exists.~~ — diagnostics derive `gpu_interop_enabled`
   from the actual frame `handoff_path`, not the sink variant; the
   variant is renamed `SharedTextureFileStub` so the GUI never claims
   GPU interop based on selection alone.

### Phase 2: Fix simulation step correctness — DONE (2026-05-07)

1. ~~Move `self.sim_clock.advance(frame_dt)` outside the avatar loop.~~
2. ~~Make `PhysicsWorld::step_all` accept toggles/options.~~ — now
   `step_all(fixed_dt, substeps, avatar, options: SimulationStepOptions)`.
3. ~~Add tests for spring/cloth disabled behavior with Rapier initialized.~~
   — see #5 status note for the cloth-test caveat (gate is shared with
   spring, no dedicated cloth test added).

### Phase 3: Improve tracking stale policy — OPEN

1. Store last valid tracking sample and timestamp in `Application`.
2. Implement hold/fade policy instead of immediate `None`.
3. Surface tracking age in GUI diagnostics.

Note: `Application::last_tracking_pose` already exists (set by
`run_frame` whenever a fresh sample arrives) but is never *consumed*
when the next frame's sample is stale. The hold/fade policy can build
on that field rather than introducing a new one.

### Phase 4: Untangle renderer/output contracts — OPEN

1. Move handoff types to a neutral module.
2. Remove direct `vulkano::Image` from `OutputFrame`.
3. Make renderer produce either:
   - `CpuReadbackFrame`
   - `GpuFrameToken`
   through one neutral enum.

### Phase 5: Split GUI state — OPEN (small slice done as part of #2)

1. Extract output panel state and reconciliation first, because output already
   has requested-vs-active complexity.
2. Extract tracking panel state next.
3. Keep `GuiApp` as a coordinator and owner of top-level sub-states only.

Note: `gui/inspector/output.rs` was partially converted to read live
`OutputDiagnostics` rather than mutating `GuiApp`, but `GuiApp` itself
was not split into the suggested sub-states.

### Phase 6: Real GPU handoff — OPEN

1. Define the external handle type and sync primitive for the first supported
   OBS path.
2. Implement GPU export for that path.
3. Make CPU readback a visible fallback.
4. Benchmark:
   - render-only
   - render + GUI preview readback
   - render + output CPU fallback
   - render + GPU handoff

## Suggested Tests / Checks

Add focused tests where possible:

- ~~`Application::run_frame` advances simulation clock once per frame for all
  avatars.~~ — covered by
  `simulation::tests::sim_clock_advance_does_not_regenerate_substeps_within_a_frame`.
  An end-to-end `run_frame` test was *not* added because constructing a
  full `Application` (render thread, output worker, tracking source) in
  a unit test is heavy; the clock-level test is sufficient because the
  call site in `run_frame` is now a single hoisted `advance` followed
  by reuse — there is no way for it to regenerate substeps per avatar.
- Rapier initialized + cloth disabled does not mutate cloth state.
  *(Open. The shared `SimulationStepOptions.cloth_enabled` gate has the
  same shape as the spring gate, but a dedicated test was deferred —
  building cloth state in a unit test requires a `ClothSimState` plus
  particles, which the existing test fixture in
  `cloth_solver/tests.rs` already does. Wiring that into a
  `PhysicsWorld::step_all` call is the remaining work.)*
- ~~Rapier initialized + spring disabled does not mutate spring state.~~
  — `simulation::tests::step_all_with_spring_disabled_does_not_mutate_spring_state`
  + `…_enabled_does_mutate…` counterpart.
- ~~Output diagnostics reflect actual frame handoff path.~~ —
  `output::tests::diagnostics_handoff_path_follows_published_frames`.
- ~~Selecting `SharedTexture` without GPU handle support reports unavailable or
  fallback, not `GpuSharedFrame`.~~ — implicit in the diagnostics test
  above (`active_handoff_path` is `None` until a frame publishes; once
  one publishes with `HandoffPath::CpuReadback` it is reported as such
  regardless of the sink variant).
- Stale tracking sample enters hold/fade policy rather than immediate rest.
  *(Open — Phase 3 work item.)*

Manual checks:

- Run with a real VRM at 1920x1080 and inspect CPU usage with:

```powershell
RUST_LOG=vulvatar=info cargo run
```

- Toggle output sinks and confirm the GUI reports the real path.
- Start tracking, then block/stop camera input briefly and watch whether pose
  snaps to rest or degrades gracefully.
- Toggle spring/cloth with a loaded avatar after Rapier initialization and
  verify the toggles actually gate simulation.

## Bottom Line

The project is not "just a scaffold" anymore. It is a real application with a
large amount of runtime behavior, but some architecture comments still describe
future intent rather than current truth.

The most important near-term discipline is to make the runtime contract honest:

- if a path is CPU readback, call it CPU readback
- if a sink is file-backed, call it file-backed
- if a GUI toggle is only solver-side, do not describe it as model selection
- if diagnostics say GPU interop, require a real GPU token

Once the labels match reality, the next priority is to stop the CPU preview path
from becoming the output architecture.
