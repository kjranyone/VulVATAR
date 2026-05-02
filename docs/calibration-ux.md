# Pose Calibration UX

## Why

The depth-aware tracking providers (`rtmw3d-with-depth`, `cigpose-metric-depth`) read the subject's pelvic anchor from a per-pixel depth map (DAv2 / MoGe). When something other than the subject occupies depth pixels near the keypoint sampling window — a desk in front of the user, a monitor between hips and lens, a chair back, a tall plant — the metric depth read for the hip drifts by 0.3–1.5 m. That biased reading propagates through:

1. The calibration scale `c` (DAv2's metric refinement) → all joint Z values
2. Body yaw computation (uses shoulder Z spread relative to hip)
3. Root translation (now drives Hips position)
4. Any downstream physics / IK that depends on world-space hip position

A small bias in one frame is masked by the EMA / Kalman filtering. A persistent bias from a stationary obstacle is **not** masked — it shifts the entire reference frame and the avatar tracks the wrong baseline forever.

Our current safeguards (`MIN_C/MAX_C` clamp, `MIN_ANCHOR_2D_SEPARATION`) reject extreme single-frame failures but don't recover from a consistently-biased setup.

This document specifies an explicit calibration step, modeled on existing VTuber webcam tools (e.g. VSeeFace's "Calibrate VRM"), that lets the user lock in their setup-specific reference values once per session.

## User flow

### Entry point
A `Calibrate Pose ▼` split button next to the existing `Recalibrate` button in the Tracking inspector. The dropdown offers two modes:

- **Full Body** — subject visible from feet to head; calibration captures the hip pelvic anchor.
- **Upper Body Only** — subject visible from waist (or chest) up; calibration captures the shoulder midpoint anchor.

Picking a mode opens a fullscreen-modal egui `Window` that suspends the normal viewport interaction.

### Modal layout

```
┌───────────────────────────────────────────────────────────┐
│   Pose Calibration — Full Body                            │
│                                                           │
│   ┌──────────────────────┐    Step 1 of 1                 │
│   │                      │                                │
│   │   Webcam preview     │    Stand in T-pose             │
│   │   (with skeleton     │    (arms horizontal,           │
│   │    annotations)      │     palms forward,             │
│   │                      │     legs shoulder-width)       │
│   │                      │                                │
│   │                      │    ▓▓▓▓▓▓▓░░░  2.1 s left      │
│   │                      │                                │
│   └──────────────────────┘    Confidence: 0.97 ✓          │
│                               Hip depth: 1.84 m (DAv2)    │
│                               Samples captured: 18 / 60   │
│                                                           │
│         [Cancel]                          [Capture Now]   │
└───────────────────────────────────────────────────────────┘
```

The webcam preview is the same `RenderAnnotations` overlay the camera-wipe already produces; we re-route it to a larger ImageWidget when the modal is open.

### Capture sequence

Per mode:

| Mode       | Pose                                                      | Anchor measured                  |
|------------|-----------------------------------------------------------|----------------------------------|
| Full Body  | T-pose (arms horizontal, palms forward, feet shoulder w.) | Hip pair midpoint (COCO 11/12)   |
| Upper Body | Hands at sides, shoulders relaxed, facing camera          | Shoulder pair midpoint (COCO 5/6)|

Capture is a **2-second collection window** (≈ 60 frames at 30 fps) preceded by a **1-second hold-still countdown**. During collection:

- Per-frame samples are appended only when the source skeleton's overall confidence exceeds 0.5 **and** the relevant anchor pair clears `KEYPOINT_VISIBILITY_FLOOR`.
- If 1 second passes with no samples meeting the floor, surface "Pose not detected — adjust framing" and pause the timer.
- `Capture Now` finishes the collection window early using whatever samples were gathered (minimum 5).

After collection, compute the **per-axis median** of the captured samples and stash it in `PoseCalibration`.

### Cancellation

`Cancel` discards everything and closes the modal — no calibration data is written. Existing calibration (if any) is preserved.

### Re-entry

Re-running calibration overwrites the previous `PoseCalibration` for the current project. The status line in the Tracking inspector shows when the active calibration was captured (`Calibrated 2 minutes ago` / `not calibrated`).

## Data model

```rust
// src/tracking/calibration.rs (new module — split from current single-struct
// `TrackingCalibration` so the face calibration and pose calibration can
// evolve independently).

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CalibrationMode {
    FullBody,
    UpperBody,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PoseCalibration {
    pub mode: CalibrationMode,
    /// ISO-8601 timestamp of the capture, for the "calibrated N minutes ago"
    /// status line. `String` rather than `SystemTime` because serde_json
    /// handles the former cleanly.
    pub captured_at: String,
    /// Number of frames whose median produced these values. < 5 means the
    /// capture was a `Capture Now` cut-short with sparse samples.
    pub frame_count: usize,
    /// Image-centre-relative anchor X in source-skeleton units
    /// (`x ∈ [-aspect, +aspect]`). Used by `pose_solver` as the seed for
    /// the root-translation EMA reference, replacing the auto-EMA's first
    /// hip-visible frame.
    pub anchor_x: f32,
    /// Image-centre-relative anchor Y in source-skeleton units
    /// (`y ∈ [-1, +1]`).
    pub anchor_y: f32,
    /// Camera-space metric depth of the anchor pair midpoint, in meters.
    /// `None` for `rtmw3d` (no depth pipeline) — the rtmw3d-only solver
    /// path uses `anchor_x` / `anchor_y` only.
    pub anchor_depth_m: Option<f32>,
    /// Median per-frame overall confidence from the captured samples.
    /// Surfaced on the inspector to flag setups where calibration was
    /// taken under marginal conditions.
    pub confidence: f32,
    /// Anchor-depth jitter (per-frame standard deviation) during the
    /// collection window. Used by `skeleton_from_depth` as a dynamic
    /// `MIN_C / MAX_C` clamp range — `± 3 σ` of measured jitter rather
    /// than a hardcoded `[0.3, 15.0]`.
    pub anchor_depth_jitter_m: Option<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct TrackingCalibration {
    pub neutral_face_pose: FacePose,
    /// Optional pose calibration. `None` means the user has never
    /// calibrated this project — the solver uses its existing auto-EMA
    /// behaviour.
    pub pose: Option<PoseCalibration>,
}
```

## Solver / provider integration

### Anchor selection (`skeleton_from_depth::resolve_origin_metric`)

When a `PoseCalibration` is active, the resolution policy changes:

| Mode       | Behaviour                                                      |
|------------|----------------------------------------------------------------|
| Full Body  | Hip preferred → shoulder fallback (current behaviour, unchanged). |
| Upper Body | **Force** shoulder anchor regardless of hip visibility — even when the hip pair clears the floor, treat it as untrustworthy (likely picking up a desk surface). |

### Calibration scale clamp

`skeleton_from_depth` already clamps the DAv2 calibration scale `c` to `[MIN_C, MAX_C] = [0.3, 15.0]`. With a calibration available, narrow that range to a per-setup band:

```text
let c_baseline = calibration.anchor_depth_m / measured_depth_at_capture;
let c_min = (c_baseline * 0.5).max(MIN_C);
let c_max = (c_baseline * 2.0).min(MAX_C);
```

The 0.5× / 2.0× band tolerates the subject moving meaningfully closer / further from the camera but rejects the desk-suddenly-fills-the-depth-window case.

### Root translation EMA seed (`pose_solver`)

The EMA reference for `root_offset` (currently auto-initialised on the first hip-visible frame) is replaced by the calibration value:

```rust
state.root_reference.get_or_insert_with(|| {
    calibration
        .pose
        .as_ref()
        .map(|c| [c.anchor_x, c.anchor_y, c.anchor_depth_m.unwrap_or(0.0)])
        .unwrap_or(raw_offset)  // fallback to current auto-EMA path
});
```

The EMA continues to drift slowly so the calibration value is a *seed*, not a hard lock — the user can drift their stance over a long session without the avatar yo-yo-ing back to the calibrated centre.

### `rtmw3d` (no depth) path

The same calibration is honoured but the depth field is ignored. The `anchor_x` / `anchor_y` seed alone is enough to give the rtmw3d solver a meaningful root-translation reference (the existing auto-EMA logic is the fallback when no calibration was ever captured).

## Persistence

### Project file

```rust
// src/persistence.rs — TrackingConfig
#[serde(default)]
pub pose_calibration: Option<PoseCalibrationDto>,

// New DTO (mirrors PoseCalibration; separated so persistence can evolve
// the on-disk schema independently of the in-memory struct):
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PoseCalibrationDto {
    pub mode: String,           // "full_body" / "upper_body"
    pub captured_at: String,    // ISO-8601
    pub frame_count: usize,
    pub anchor_x: f32,
    pub anchor_y: f32,
    pub anchor_depth_m: Option<f32>,
    pub confidence: f32,
    pub anchor_depth_jitter_m: Option<f32>,
}
```

### Restore on project load

`GuiApp::load_state` (the existing project loader) populates
`Application::tracking_calibration.pose` from the DTO. From there,
`apply_calibration` (already invoked per-frame on the source skeleton)
forwards the calibration to the solver and to `skeleton_from_depth`.

## UI status line

Tracking inspector adds a single-line status under the calibrate button:

```
Calibrated 2 minutes ago — Full Body, depth 1.84 m, conf 0.97
                                             ^^^^^^
                                             omitted for rtmw3d
```

When no calibration exists:

```
Pose not calibrated — auto-EMA active
```

## Implementation phases

| Phase | Deliverable                                                            |
|-------|------------------------------------------------------------------------|
| A     | This doc + `tracking::calibration` module + DTO scaffolding (no behaviour change) |
| B     | Modal scaffold: open/close, preview routing, countdown UI, Capture Now button |
| C     | Capture loop: frame collection, median computation, confidence gating  |
| D     | Solver / provider integration: anchor mode, c-clamp, EMA seed         |
| E     | Persistence: save/load + status line + "calibrated N min ago" rendering |

## Open questions / future work

- **Per-profile calibration**: today the top-bar Profile concept is global — calibration is project-scoped. If profiles ever become per-user, the calibration field should move under the profile.
- **Multi-step capture**: future modes ("step left, step right" for X range, "lean in / out" for Z range) can extend `PoseCalibration` with per-axis ranges. The single T-pose / hands-at-sides flow is the MVP.
- **Person segmentation** as an alternative to calibration: a future option could replace this UX with an automatic per-pixel mask that excludes background depth. Calibration remains useful as an explicit baseline regardless.
