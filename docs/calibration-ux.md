# Pose Calibration UX

## Why

The tracking pipeline's depth stage reads the subject's pelvic anchor from a per-pixel depth map (DAv2). When something other than the subject occupies depth pixels near the keypoint sampling window — a desk in front of the user, a monitor between hips and lens, a chair back, a tall plant — the metric depth read for the hip drifts by 0.3–1.5 m. That biased reading propagates through:

1. The calibration scale `c` (DAv2's metric refinement) → all joint Z values
2. Body yaw computation (uses shoulder Z spread relative to hip)
3. Root translation (now drives Hips position)
4. Any downstream physics / IK that depends on world-space hip position

A small bias in one frame is masked by the EMA / Kalman filtering. A persistent bias from a stationary obstacle is **not** masked — it shifts the entire reference frame and the avatar tracks the wrong baseline forever.

Our current safeguards (`MIN_C/MAX_C` clamp, `MIN_ANCHOR_2D_SEPARATION`) reject extreme single-frame failures but don't recover from a consistently-biased setup.

This document specifies an explicit calibration step, modeled on existing VTuber webcam tools (e.g. VSeeFace's "Calibrate VRM"), that lets the user lock in their setup-specific reference values once per session.

## User flow

### Entry point
A `Calibrate Pose ▼` split button in the Tracking inspector. (An earlier prototype lived next to a `Recalibrate` button; that vestigial button was removed once `Calibrate Pose ▼` itself became the canonical capture entry point — running it again overwrites the active profile's calibration.) The dropdown offers two modes:

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

The camera preview is the same `RenderAnnotations` overlay the camera-wipe already produces; we re-route it to a larger ImageWidget when the modal is open.

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

### Per-profile storage

Calibration lives on `StreamProfile.pose_calibration`
(`src/gui/profile.rs`), which is persisted to
`%APPDATA%\VulVATAR\profiles.json` (Windows) /
`~/Library/Application Support/VulVATAR/profiles.json` (macOS) /
`~/.local/share/VulVATAR/profiles.json` (Linux). The profile library
follows the *user* across projects, so a single user with separate
"home desk" / "office desk" / "show stage" profiles can carry one
captured calibration per setup and switching the profile dropdown
snaps the depth pipeline to the right baseline immediately.

Save flow: `Calibrate Pose ▼ → finalize → persist_calibration()`
writes to (a) `Application.tracking_calibration.pose` for the live
solver, (b) the tracking mailbox for the depth-pipeline provider's
c-clamp / anchor-mode, and (c) the active profile + sets
`profiles_dirty`. The per-frame autosave block flushes
`profiles.json` on the next tick.

Load flow: `GuiApp` constructor calls
`crate::persistence::load_profiles()` (falling back to built-in
presets on parse failure) and seeds the active profile's
calibration into Application + mailbox. Switching profiles via
the top-bar dropdown invokes `apply_profile` which does the same
push for the newly-active profile (with `None` deliberately
clearing any prior setup's calibration so the auto-EMA falls back
to its first-frame seed instead of carrying over a stale baseline).

### Legacy project-file rescue

Earlier versions stored calibration on the project file
(`ProjectState.pose_calibration` / `TrackingConfig.pose_calibration`).
The on-disk DTO field is still defined but marked
`#[serde(default, skip_serializing_if = "Option::is_none")]` — new
saves never emit it, but old project files can still be parsed.
When `GuiApp::load_state` sees a legacy `Some(_)` *and* the
currently-active profile has no calibration yet, it migrates the
value onto the profile and marks the library dirty so the rescue
lands in `profiles.json` on the next autosave. When the profile
already has a calibration, the legacy value is discarded — the
user has since calibrated under the per-profile model and that
takes precedence.

### DTO

```rust
// src/persistence.rs — PoseCalibrationDto. The storage location moved
// from the project file to profiles.json (see "Per-profile storage"
// above); the shape evolved to add the optional torso depth template
// for depth-aware providers and the shoulder span used by the
// scale-invariant arm-length recovery in skeleton_from_depth.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PoseCalibrationDto {
    pub mode: String,           // "full_body" / "upper_body"
    pub captured_at: String,    // ISO-8601
    pub captured_at_unix: u64,
    pub frame_count: usize,
    pub anchor_x: f32,
    pub anchor_y: f32,
    pub anchor_depth_m: Option<f32>,
    pub confidence: f32,
    pub anchor_depth_jitter_m: Option<f32>,
    /// Median shoulder-pair separation in meters during the capture
    /// window. Used by `skeleton_from_depth` as the reference for the
    /// subject-specific arm-length scale recovery — without it the
    /// solver falls back to a population-average length.
    pub shoulder_span_m: Option<f32>,
    pub x_range_observed: Option<f32>,
    pub z_range_observed: Option<f32>,
    /// Optional small (e.g. 32×32) per-pixel depth grid sampled over
    /// the captured torso bbox. Lets the depth-aware providers
    /// detect frame-to-frame deviation from the calibrated torso
    /// shape (desk, chair back, monitor) and fall back to anchor-only
    /// rather than tracking a contaminated metric depth read.
    pub torso_depth_template: Option<TorsoDepthTemplateDto>,
}
```

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
| F     | Multi-step capture: optional X/Z range step → per-axis sensitivity in solver |
| G     | Per-profile storage: calibration moves from project file to `profiles.json` so each setup carries its own baseline |
| H     | Bust-up framing fallback: UpperBody wait-gate switches to anchor-stillness when elbows are cropped out (see below) |
| I     | Neutral body yaw: oblique camera placement maps the user's habitual forward to avatar-forward (see below) |

## Multi-step capture (Phase F)

After the anchor capture finishes, the modal pauses on an
`AnchorDone` prompt offering two choices:

- **Skip & Finish** — commits the anchor-only calibration and closes
  the modal. The solver falls back to its static
  `[0.6, 0.6, 0.3]` per-axis sensitivity for root translation.
- **Capture Range ▶** — enters a 10-second range capture preceded
  by a 1.5-second hold-still pre-roll. The user steps left, right,
  leans toward the camera, then back. Per-frame source `(x, z)` is
  folded into running min/max; on completion the per-axis
  peak-to-peak deltas become `x_range_observed` /
  `z_range_observed` on the in-flight `PoseCalibration`.

The solver derives per-axis sensitivity from the observed range:

```text
sens_x = clamp(TARGET_AVATAR_X / (range_x / 2), [0.3, 2.5])
sens_z = clamp(TARGET_AVATAR_Z / (range_z / 2), [0.15, 1.0])
```

with `TARGET_AVATAR_X = 0.5` and `TARGET_AVATAR_Z = 0.3` so the
avatar's hip reaches roughly `±TARGET` units when the subject hits
their captured extremes. A user with a narrow room (small ±X) gets
a higher gain so a 30 cm step still reads as meaningful avatar
motion; a user with a large studio gets a lower gain so a full sweep
still keeps the avatar within frame.

Floors:
- `MIN_RANGE_OBSERVED = 0.05` — captured sweeps below this are
  treated as "user barely moved" and the field is left as `None`.
- `MIN_RANGE_SAMPLES = 10` — the entire range step is discarded
  below this so a near-empty 10-second window doesn't poison the
  derived sensitivity.

The Z range field is only emitted when the anchor was already
metric (depth-aware providers); rtmw3d-only paths leave it `None`.

## Bust-up framing fallback (Phase H)

### Problem

The `WaitingForPose → Collecting` gate scores the live pose against the
mode's target pose via **shoulder→elbow direction vectors**
(`pose_match::pose_match_score`). Both elbows must be visible with
≥ 0.3 keypoint confidence or the score is pinned at 0 — the capture
never starts.

Typical webcam-streamer framing (bust-up: head to slightly below the
shoulders) crops both elbows out **permanently**. The existing
`no_lower_arms_since` hint ("step back so your elbows are visible")
assumes the framing is adjustable; for a streamer whose scene layout
fixes the camera crop, it is not. A T-pose is equally impossible, so
`FullBody` is not an alternative. The only current escape hatch is the
`Capture Now` bypass button, which skips the quality gate entirely and
is not discoverable as *the* intended path.

### What a bust-up capture can and cannot measure

The gate is the only real blocker — the data the calibration actually
needs is almost fully available without elbows:

| Captured value                          | Available at bust-up framing |
|-----------------------------------------|------------------------------|
| Anchor (shoulder-pair midpoint x/y/z)   | ✓ (UpperBody already forces the shoulder anchor) |
| `shoulder_span_m`                       | ✓ (shoulder pair only)       |
| `neutral_expressions`, `neutral_face_ypr_{mesh,body}` | ✓ (face only) |
| Torso depth template                    | △ shrinks to shoulder-line → frame bottom; occluder rejection still works on the covered cells |
| X/Z range step (Phase F)                | ✓ (anchor-only sweep)        |
| Arm-direction pose-match gate           | ✗ — this is the blocker      |

### Design: automatic stillness-gate fallback inside UpperBody

No third `CalibrationMode` is added. Adding a `BustUp` variant would
force the user to self-diagnose their framing before calibrating,
ripple through serde / persistence / anchor policy for zero data-model
benefit (the captured values and the anchor policy are identical to
UpperBody), and leave the failure mode in place for anyone who picks
the wrong mode. Instead, `UpperBody`'s wait state detects the framing
and swaps its *gate*, not its mode:

1. **Engage condition** — in `WaitingForPose { mode: UpperBody }`,
   when the shoulder anchor has been continuously visible while both
   lower-arm keypoints have been continuously missing for
   `NO_ANCHOR_HINT_SECONDS` (4 s, same constant the framing hint
   already uses), the wait state latches `stillness_fallback = true`.
2. **Latching** — once engaged, the fallback stays engaged until the
   state machine leaves `WaitingForPose` (capture start, cancel, or
   retry). Elbows flickering back in at the frame edge must not bounce
   the user between two different instructions / progress semantics
   mid-wait. A fresh capture attempt re-evaluates from scratch.
3. **Stillness gate** — while engaged, the per-frame score is no
   longer arm direction but **anchor stillness**: each time the
   tracking mailbox sequence advances, the anchor (`root_offset` x/y,
   source units) is compared against the previous tracking frame's
   position. Displacement below `STILLNESS_EPS` per tracking frame
   scores toward the gate; at or above it resets `frames_at_match`,
   exactly mirroring the arm-direction gate's hold semantics.
   `REQUIRED_STABLE_FRAMES` (15 tracking frames ≈ 0.5 s) is shared.
   The score shown on the progress bar is
   `1 − (displacement / STILLNESS_EPS)` clamped to `[0, 1]`, so the
   bar keeps its "closer to ready" meaning.
   - `STILLNESS_EPS = 0.010` source units per tracking frame. Source
     space is y ∈ [−1, 1], so this is 0.5 % of frame height —
     generous enough for breathing / keypoint jitter at typical
     desk distance, tight enough that leaning or reaching resets
     the hold. Tracked per *mailbox sequence*, not per GUI repaint:
     the GUI repaints faster than the tracker emits frames, and
     per-repaint deltas would read as near-zero motion at any speed.
   - Frames where the anchor is missing or `overall_confidence <
     MIN_FRAME_CONF` reset the hold (same bar the Collecting window
     applies to sample admission — no point starting a capture from
     frames that Collecting would immediately reject).
4. **Guidance text stays coherent end-to-end** — every string the
   user reads while the fallback is engaged describes the stillness
   gate, never the arm pose they can't perform:
   - *Idle*: the UpperBody instruction body ends with a heads-up that
     capture auto-switches to hold-still detection when the elbows
     aren't in frame, so a bust-up user knows pressing Start is fine.
   - *WaitingForPose*: the step heading swaps from "get into the
     pose" to "face the camera and hold still", the instruction body
     to the bust-up variant, and the progress-bar label from
     "Match N%" to "Stillness N%" (the score is motion-derived, not
     pose similarity). The post-lock "nice, hold it…" heading and the
     lock-in fill are shared — they mean the same thing in both gates.
   - *Framing hint*: the "step back so your elbows are visible" row
     is replaced by "switched to hold-still detection — no pose
     needed" (the step-back advice is unactionable at fixed streamer
     framing). The `FullBody` path keeps the step-back hint
     unchanged — at full-body framing, missing elbows really do mean
     the camera is too close for the required T-pose.
   - *Collecting*: the latch is carried into the `Collecting` variant
     (display-only field) so the body text keeps saying "hold still"
     through the sample window instead of reverting to hands-at-sides
     instructions.
5. **Everything downstream is unchanged** — `Collecting` already
   admits samples on anchor visibility + confidence alone (elbows are
   never consulted), the UpperBody target-pose reference render stays
   hands-at-sides (the un-framed part of the pose is irrelevant), the
   `PoseCalibration` schema is untouched, and the Phase F range step
   works as-is on the shoulder anchor.

### Non-goals

- Detecting bust-up framing *before* the user presses Start (Idle has
  no wait loop; 4 s after Start is soon enough and avoids flicker on
  the mode picker).
- Persisting "this calibration was captured at bust-up framing" — no
  consumer needs it; `torso_depth_template` cell coverage already
  reflects the reduced torso window organically.

## Neutral body yaw — oblique camera placement (Phase I)

### Problem

A desk-mounted D435 rarely sits dead ahead of where the user actually
faces (monitor centre); 15–45° of horizontal offset is the norm. Today
only the **head** is neutralised for this (`neutral_face_ypr_*`,
subtracted in `apply_calibration`). The **torso** is not:
`compute_shoulder_yaw_rotation` aims the avatar's shoulder line at the
observed camera-space shoulder line every frame, so a 30°-oblique
camera bakes a constant 30° body yaw into the avatar — producing the
"head faces the viewer, torso faces sideways" composite once the face
neutral kicks in. Translation axes are similarly camera-aligned: a
side-step at the desk reads as a diagonal x+z move.

### Semantic contract (decide this first — everything follows)

**Neutral means: the user's habitual working orientation maps to
avatar-forward.** This is the same contract `neutral_face_ypr` already
implements ("camera mounted off eye line ... otherwise bakes a
constant head turn into the avatar"). Consequence for the capture UX:
the hold instruction must say **"face your usual forward (your
monitor), in your natural working posture"** — NOT "face the camera".
The current UpperBody / bust-up strings say "face the camera", which
coincides with the contract only when the camera happens to be
frontal; at oblique placement the two instructions produce *different
calibrations* and the face neutral has silently depended on users
misreading the instruction in the favourable way. Fixing this wording
is part of Phase I regardless of the yaw work.

### The core design decision: de-rotate the *scene*, not the root

The naive fix — subtract a captured neutral angle inside
`compute_shoulder_yaw_rotation` — is **wrong** and was rejected:

- **Rejected A: root-only yaw subtraction.** The solver's driven-bone
  loop aims every bone at *source-space* directions and the arm
  two-bone IK targets *source-space* wrist positions. De-rotating only
  the Hips leaves all of those targets in the oblique camera frame:
  the pelvis faces forward while spine, shoulders and arms still chase
  30°-rotated directions — a permanently twisted torso and arms that
  point wrong by exactly the neutral angle. Any per-consumer patch
  scheme (subtract here, subtract there) reintroduces this class of
  bug every time a new consumer of source directions lands.
- **Rejected B: de-rotation inside the provider
  (`skeleton_from_depth`).** The tracking mailbox deliberately carries
  the RAW pose; calibration is applied at solve time on a clone (see
  `refresh.rs`: "The snapshot pose is raw ... no risk of subtracting a
  previous calibration twice"). The calibration modal's capture loop
  reads the mailbox — if the provider published de-rotated skeletons,
  every *re*-calibration would measure anchors/yaw in
  already-corrected space, converging the stored neutral toward zero
  across recaptures (the double-subtraction bug class the raw-mailbox
  invariant exists to prevent). Provider internals (ray-IK along
  observation rays, L/R swap correction, arm_z, torso-template
  registration) are camera-geometry algorithms and must keep operating
  in camera space anyway.

**Chosen: a single rigid re-expression of the published sample at
solve time**, in `TrackingCalibration::apply_calibration` — the same
insertion point, clone semantics and staleness model as the face
neutral. Conceptually: *re-express the observation as if the camera
had been placed frontally at the same distance*, i.e. rotate the
scene by `−θ` about the vertical axis through the calibrated anchor
`a0`:

```text
p' = a0 + R_y(−θ) · (p − a0)
```

One transform, applied uniformly, keeps every downstream consumer
consistent by construction instead of by per-call-site vigilance.

### Consumer inventory (what rotates, what must NOT)

| Sample field | Rotate? | Why |
|---|---|---|
| `joints` positions | ✓ | isotropic source units (normalised by `mpsu`) — rotation is geometrically valid |
| `fingertips` | ✓ | same space as joints |
| `left/right_hand_orientation` | ✓ (quaternion pre-multiply by `R_y(−θ)`) | palm frames are source-space; forgetting this leaves wrist twist off by θ — easy to miss because the error is subtle at small angles |
| `root_offset` | ✓ | on the D435 path this is a raw-metres camera-space vector (`pose_solver` "Metric translation" contract), so the rotation is unit-safe; see units audit below |
| `face` / `face_body_raw` | ✗ | head neutrality is owned by `neutral_face_ypr_*` (per-estimator residuals); rotating the face pose *and* subtracting its neutral would double-count θ. `face_body_raw` additionally must stay raw (capture-only contract) |
| `metric_frame_info` (`anchor_cam_m`, `intrinsics`, `mpsu`, spans) | ✗ | documented as RAW camera space; the sensor-matched mirror render and any camera-geometry consumer need the true camera frame |

Both-neutral consistency check: when the user turns their whole body
by φ, body yaw and camera-relative face yaw each change by φ, and each
stream subtracts its own captured constant — head stays glued to the
torso with no double subtraction, because the face pose is *excluded*
from the rigid rotation.

### Units audit (the trap that would surface months later)

- `joints` / `fingertips`: isotropic source frame (metric skeleton
  scaled by `mpsu`) — rotation valid.
- `root_offset` (D435 path): raw metres, all three axes — rotation
  valid. The stale "x in source units, z in metres" wording on the
  `SourceSkeleton::root_offset` doc comment predates the metric-path
  rework and must be reconciled during implementation, not worked
  around.
- Pivot `a0`: stored as `anchor_x/anchor_y` (source units) +
  `anchor_depth_m` (metres). Expressing it in each rotated field's own
  frame needs the per-frame `mpsu` — available on
  `metric_frame_info`.
- **Non-metric path (`metric_frame_info == None`): do not capture and
  do not apply.** Rotating an `(x, y, 0)` offset about Y fabricates
  depth from nothing, and monocular z is not trustworthy enough to
  measure θ in the first place. D435 is the sole capture backend;
  the guard is for synthetic/bench producers.

### Measurement (capture window)

Per admitted `Collecting` frame, compute the shoulder-line horizontal
yaw `θ = atan2(Δz, Δx)`-style deviation from frontal, from the same
`LeftUpperArm`/`RightUpperArm` joints the solver uses, with the same
`MIN_HORIZ_SPAN` degeneracy floor. Aggregate as the median; store
`neutral_body_yaw: Option<f32>` (radians) with `#[serde(default)]` —
`None` (old saves, insufficient samples, non-metric) is a strict
no-op, preserving today's behaviour exactly.

Floors and clamps:
- ≥ 5 finite samples, else `None` (matches `FACE_NEUTRAL_MIN_SAMPLES`).
- `|θ|` clamp at 60°: beyond that the far shoulder is
  occlusion-shadowed in depth and L/R swap risk dominates — a larger
  reading is more likely garbage than geometry.
- Inspector warning above 45°: "camera very oblique — tracking quality
  degrades" (surfaced, not silently clamped).
- Sign convention gets dedicated unit tests against synthetic
  skeletons — the selfie-mirror x-flip (`pose_match.rs` convention
  reminder: avatar bone names ≠ subject anatomical sides) makes yaw
  sign the single most likely silent-wrong constant in this design.

### Interactions audited

- **Range step (Phase F)**: `RangeCollecting` folds raw mailbox
  `(x, z)` into min/max. With de-rotation active at runtime, raw-axis
  extremes describe a camera-aligned box while runtime offsets live in
  body frame — so rotate **each sample** by `R_y(−θ)` (θ is already on
  the in-flight calibration carried in the variant) *before* folding.
  Rotating the finished box instead would under/over-state the range
  (an axis-aligned box is not rotation-equivariant).
- **Per-axis sensitivity** consequently becomes body-frame — which is
  the semantics the instruction already implies ("step left, lean
  toward the camera" are body-relative actions).
- **Sensor-matched overlays** (camera-wipe PIP, `validate_pipeline`
  composites): these show the true camera view; a de-rotated avatar
  *intentionally* no longer matches the photo's viewing angle when a
  neutral yaw is active. Documented here so nobody "fixes" the
  mismatch later by rotating the render back. Benches run without
  calibration and are unaffected by default.
- **EMA / 完全ミラー root reference**: the reference seeds from the
  calibration anchor and (default) freezes after lock-in; under the
  single-pivot transform, `dev = R·(c − a0)` — deviations rotate
  cleanly, no re-seeding needed.
- **Head**: see consumer table — no interaction beyond the constant
  each stream already owns.

### Rollout (measure first, then change behaviour)

1. **I1 — measure + surface, zero behaviour change**: capture
   `neutral_body_yaw` during the existing hold, persist it, show it on
   the inspector status line ("yaw −28°"), and log it in the
   signal-quality / live-debug channel. This is the probe that
   confirms the real-setup magnitude (and sign!) against the physical
   camera angle before any solve-path change.
2. **I2 — solve-time rigid de-rotation** of joints / fingertips / hand
   orientations / root_offset, with the round-trip unit tests below.
3. **I3 — range-sample rotation + capture-instruction rewording**
   ("face your usual forward") across all four locales.

Each phase lands complete (no interim fallbacks); the split exists
because I1's reading validates the constant before I2 spends it.

### Test plan

- Unit: yaw-measurement sign on synthetic skeletons (camera-left vs
  camera-right placement, mirrored-x convention).
- Unit: rigid round-trip — build a frontal reference sample, rotate
  the scene by θ about `a0` (joints, fingertips, hand quats,
  root_offset), apply a calibration carrying `neutral_body_yaw = θ`,
  assert the result equals the frontal reference within epsilon.
- Unit: `None` neutral is a bitwise no-op; non-metric samples are
  never rotated even with `Some(θ)`.
- Unit: range folding of rotated samples vs rotating the folded box
  (must differ; folding-of-rotated is the spec).
- Live: `diagnose_video_replay` (CPU inference, non-competing) on
  oblique footage; live-debug protocol to read the measured θ and
  post-I2 residual shoulder yaw on the user's actual setup.

## Open questions / future work

- **Person segmentation** as an alternative to calibration: a future option could replace this UX with an automatic per-pixel mask that excludes background depth. Calibration remains useful as an explicit baseline regardless.
- **Camera pitch / roll neutral** (Phase I deliberately covers yaw only):
  the shoulder line constrains yaw and roll but carries **zero pitch
  information**, and at bust-up framing there is no visible spine axis
  to measure pitch from — a "pitch neutral" would be fit from noise.
  Camera pitch is also observationally entangled with the user's
  habitual lean (a D435 has no IMU to break the tie; the D435i does),
  and shoulder-line roll is already routed into the UpperChest lean by
  the solver rather than the root. If pitch neutralisation is ever
  wanted, it needs either an IMU-equipped camera or a dedicated
  capture step with the user's posture constrained — not a bolt-on to
  the Phase I transform.
