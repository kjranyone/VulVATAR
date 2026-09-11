# Pose Calibration UX

> The captured `PoseCalibration` is applied at solve time via
> `TrackingCalibration::apply_calibration` (neutral face / expression /
> body-yaw). Wiring the same capture into the fusion estimator's
> `q_neutral` is future work ([tracking-v2-design.md](tracking-v2-design.md) §6).
> The code is the source of truth for exact thresholds and field sets
> (`src/tracking/calibration.rs`, `src/gui/calibration/`); this
> document carries the UX flow, the data contract, and the design
> rationale.

## Why

The tracking pipeline reads the subject's anchor (hip or shoulder midpoint) from the RealSense D435 per-pixel depth map. When something other than the subject occupies depth pixels near the keypoint sampling window — a desk in front of the user, a monitor between hips and lens, a chair back — the depth read for the anchor drifts by 0.3–1.5 m. That biased reading propagates through:

1. Body yaw computation (uses shoulder Z spread)
2. Root translation (drives Hips position)
3. Any downstream physics / IK that depends on world-space hip position

A small bias in one frame is masked by the filtering. A persistent bias from a stationary obstacle is **not** masked — it shifts the entire reference frame and the avatar tracks the wrong baseline forever. Per-frame plausibility gates reject extreme single-frame failures but can't recover from a consistently-biased setup.

Calibration is the explicit fix, modeled on existing VTuber tools (e.g. VSeeFace's "Calibrate VRM"): the user locks in their setup-specific reference values once per setup (profile).

## User flow

### Entry point
A `Calibrate Pose` button in the Tracking inspector (and mirrored in the top bar, disabled with a "start the camera first" tooltip while tracking is off). Clicking opens a fullscreen-modal egui `Window` that suspends the normal viewport interaction; it reopens at the mode the active profile was last calibrated with (FullBody for a first-time capture). The mode is picked inside the modal on the Idle pane via segmented buttons:

- **Full Body** — subject visible from feet to head; calibration captures the hip pelvic anchor.
- **Upper Body Only** — subject visible from waist (or chest) up; calibration captures the shoulder midpoint anchor.

### Modal layout

Left: live camera preview with the detection-annotation overlay (same
`DetectionAnnotation` data the camera-wipe PIP uses) beside a one-shot
offscreen render of the avatar in the mode's target pose ("match this
pose"). Right: the state-driven status pane — instructions, progress
bar, live telemetry (confidence / anchor-detected), contextual hints,
and the step's action buttons.

### Capture sequence (state machine)

Per mode:

| Mode       | Pose                                                             | Anchor measured                  |
|------------|------------------------------------------------------------------|----------------------------------|
| Full Body  | T-pose (arms horizontal, palms forward, feet shoulder w.)        | Hip pair midpoint (COCO 11/12)   |
| Upper Body | Hands at sides, shoulders relaxed, facing your usual forward (monitor) | Shoulder pair midpoint (COCO 5/6)|

States (`src/gui/calibration/state.rs`):

1. **Idle** — mode picker + instructions; nothing is sampled until the
   user presses **Start**.
2. **WaitingForPose** — event-driven gate, no fixed countdown: the
   per-frame pose-match score (shoulder→elbow direction vs the target
   pose) must hold above threshold for ~0.5 s of tracking frames.
   In UpperBody, if the elbows stay cropped out while the shoulder
   anchor is visible, the gate latches onto the **stillness fallback**
   (see "Bust-up framing fallback" below). `Capture Now` bypasses the
   gate.
3. **Collecting** — 2-second sample window. Per-frame admission:
   mailbox sequence advanced, anchor kind matches the mode, overall
   confidence ≥ `MIN_FRAME_CONF`. Alongside the anchor samples the
   window accumulates the neutral expressions, the per-source neutral
   face poses, and the shoulder-line yaw (see "Neutral body yaw"
   below). Finalisation
   computes per-axis **medians** (minimum `MIN_SAMPLES` frames, else
   the capture fails with a retry prompt).
4. **AnchorDone** — the anchor calibration is already persisted;
   the user chooses `Capture Range ▶` (optional 10-second movement
   sweep) or `Skip & Finish`, or `Retry`.
5. **RangeHoldStill / RangeCollecting** — optional per-axis
   peak-to-peak movement capture (see "Optional movement-range
   capture" below).
6. **Done** — capture summary (frames/depth, shoulder span, movement
   range, face-neutral status) shown until the user closes the modal.

### Cancellation

`Cancel` discards everything and closes the modal — no calibration data is written. Existing calibration (if any) is preserved.

### Re-entry

Re-running calibration overwrites the previous `PoseCalibration` for the current project. The status line in the Tracking inspector shows when the active calibration was captured (`Calibrated 2 minutes ago` / `not calibrated`).

## Data model

Authoritative definitions: `src/tracking/calibration.rs`
(`CalibrationMode`, `PoseCalibration`) and `src/tracking/mod.rs`
(`TrackingCalibration { pose: Option<PoseCalibration> }`). This doc
deliberately does not duplicate the struct bodies — the field docs
there carry the per-field rationale and unit contracts. Summary of
what one capture stores:

| Field group | Contents | Consumer |
|---|---|---|
| Identity | `mode`, `captured_at`, `captured_at_unix`, `frame_count`, `confidence` | inspector status line |
| Anchor | `anchor_x/y` (median `root_offset`; metres), `anchor_depth_m` (positive forward metres), `anchor_depth_jitter_m` (*diagnostic only*) | body-yaw pivot in `apply_calibration` |
| Body scale | `shoulder_span_m` | Done-pane summary + persistence |
| Movement range | `x_range_observed`, `z_range_observed` (body-frame) | Done-pane summary + persistence |
| Neutrals | `neutral_expressions`, `neutral_face_ypr_mesh/_body`, `neutral_body_yaw` | `TrackingCalibration::apply_calibration` at solve time |

## Solve-time transforms (`TrackingCalibration::apply_calibration`)

Applied to a clone of the sample just before retarget each frame
(the mailbox always carries the RAW sample — the recapture-invariant
that prevents double subtraction): the neutral-body-yaw scene
de-rotation, the per-source neutral face-pose subtraction, and the
neutral-expression rescale. The fusion estimator does not yet consume
the capture (`q_neutral` is future work). D435 depth is measured, not
estimated — there is no scale-calibration step. `anchor_depth_jitter_m`
is a capture-quality diagnostic only (inspector status detail).

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
`apply_calibration` path, (b) the tracking mailbox, and (c) the
active profile + sets `profiles_dirty`. The per-frame autosave block
flushes `profiles.json` on the next tick.

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

`src/persistence.rs::PoseCalibrationDto` mirrors `PoseCalibration`
field-for-field (mode as a stable string, plus a compatibility
single-source `neutral_face_ypr` field — written as a mirror of the
mesh value, and loaded into `neutral_face_ypr_mesh` when the
per-source fields are absent). Newer fields
(`shoulder_span_m`, ranges, torso template, the neutral family,
`neutral_body_yaw`) are all `#[serde(default)]` so every older save
loads with them absent rather than failing; load-time coercions
(positive `anchor_depth_m`, body-yaw plausibility re-clamp) live in
`dto_to_pose_calibration`. The profile round-trip test in
`src/gui/profile.rs` asserts every field survives save→load — add the
assertion when adding a field.

`shoulder_span_m` is stored in **real camera metres**, not the
normalised source units the published `SourceSkeleton` uses (whose
shoulder span equals the fixed `TARGET_SRC_SHOULDER_SPAN` (0.75)).
Measuring it raw would store that constant for every subject; consuming
it as metres then inflates every anthropometric bone by ~1.8× (depth-hole
joints fly off along their ray) while shrinking the published skeleton
to ~58% of the range the solver's dead-zones are tuned for. The capture
path converts back with `MetricFrameInfo::mpsu`, and the aggregate,
loader and consumer all gate on `tracking::shoulder_span_plausible`
(0.20–0.60 m), so a pre-fix profile loads as "not captured" instead of
poisoning the session.

## UI status line

Tracking inspector adds a single-line status under the calibrate button:

```
✓ Calibrated 2 min ago — Full Body
  Depth 1.84 m, jitter ±0.025 m, 47 samples, conf 0.97   ← depth parts omitted for non-depth captures
  Body yaw neutral −28° (oblique-camera correction active) ← only when a neutral body yaw is stored
```

The body-yaw line turns into an amber "camera very oblique" warning
above 45°. When the live anchor kind contradicts the calibrated mode
the headline becomes an amber mismatch warning; with no calibration:

```
Pose not calibrated — auto-EMA active
```

## Optional movement-range capture

After the anchor capture finishes, the modal pauses on an
`AnchorDone` prompt offering two choices:

- **Skip & Finish** — commits the anchor-only calibration; the range
  fields stay `None`.
- **Capture Range ▶** — enters a 10-second range capture preceded
  by a 1.5-second hold-still pre-roll. The user steps left, right,
  leans toward their usual forward, then back. Per-frame `(x, z)` is
  folded into running min/max (with a neutral body yaw active, each
  sample is rotated into the body frame first — see "Neutral body
  yaw" below); on
  completion the per-axis peak-to-peak deltas become
  `x_range_observed` / `z_range_observed` on the in-flight
  `PoseCalibration`.

**Consumer status**: no solver path consumes the ranges. Metric root
translation is **1:1** (a 30 cm side-step is 30 cm of avatar travel —
no room-size gain, no clamp), so there is no per-axis sensitivity to
derive from them. The ranges exist for the Done-pane summary and
persistence; any future consumer (e.g. an optional "fit my room"
scaling toggle) inherits body-frame-consistent values.

Floors (`finalize_range_collection`):
- `MIN_RANGE_OBSERVED = 0.05` — captured sweeps below this are
  treated as "user barely moved" and the field is left as `None`.
- `MIN_RANGE_SAMPLES = 10` — the entire range step is discarded
  below this so a near-empty 10-second window doesn't emit
  meaningless extremes.

The Z range field is only emitted when the captured z was actually
metric; non-metric captures leave it `None`.

## Bust-up framing fallback (stillness gate)

### Why it exists

The arm-direction gate (`pose_match::pose_match_score`, which drives
`WaitingForPose → Collecting`) needs both elbows visible with ≥ 0.3
keypoint confidence — without them the score is pinned at 0 and
capture can never start on that gate alone.

Typical desk-delivery framing (bust-up: head to slightly below the
shoulders) crops both elbows out **permanently**: the "step back so
your elbows are visible" framing hint is unactionable when the scene
layout fixes the camera crop, and a T-pose is equally impossible, so
`FullBody` is not an alternative. The `Capture Now` bypass skips the
quality gate entirely, which is an escape hatch — not a quality path.
Hence the stillness gate below.

### What a bust-up capture can and cannot measure

The gate is the only real blocker — the data the calibration actually
needs is almost fully available without elbows:

| Captured value                          | Available at bust-up framing |
|-----------------------------------------|------------------------------|
| Anchor (shoulder-pair midpoint x/y/z)   | ✓ (UpperBody already forces the shoulder anchor) |
| `shoulder_span_m`                       | ✓ (shoulder pair only)       |
| `neutral_expressions`, `neutral_face_ypr_{mesh,body}` | ✓ (face only) |
| Torso depth template                    | △ shrinks to shoulder-line → frame bottom; occluder rejection still works on the covered cells |
| X/Z range step (optional sweep)         | ✓ (anchor-only sweep)        |
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
     pose" to "face your usual forward and hold still", the instruction body
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
   `PoseCalibration` schema is untouched, and the movement-range step
   works as-is on the shoulder anchor.

### Non-goals

- Detecting bust-up framing *before* the user presses Start (Idle has
  no wait loop; 4 s after Start is soon enough and avoids flicker on
  the mode picker).
- Persisting "this calibration was captured at bust-up framing" — no
  consumer needs it.

## Neutral body yaw — oblique camera placement

### Why it exists

A desk-mounted D435 rarely sits dead ahead of where the user actually
faces (monitor centre); 15–45° of horizontal offset is the norm. The
**head** is neutralised for this by `neutral_face_ypr_*` (subtracted
in `apply_calibration`). Without a matching **torso** neutral,
`compute_shoulder_yaw_rotation` would aim the avatar's shoulder line
at the observed camera-space shoulder line every frame — a 30°-oblique
camera would bake a constant 30° body yaw into the avatar, producing a
"head faces the viewer, torso faces sideways" composite, and a
side-step at the desk would read as a diagonal x+z move. The neutral
body yaw removes that constant for the whole scene.

### Semantic contract (everything follows from this)

**Neutral means: the user's habitual working orientation maps to
avatar-forward.** This is the same contract `neutral_face_ypr` already
implements ("camera mounted off eye line ... otherwise bakes a
constant head turn into the avatar"). Consequence for the capture UX:
every hold instruction says **"face your usual forward (your
monitor), in your natural working posture"** — never "face the
camera". The two coincide only when the camera happens to be frontal;
at oblique placement they produce *different calibrations*.

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
- **Rejected B: de-rotation inside the provider.** The tracking
  mailbox deliberately carries the RAW pose; calibration is applied
  at solve time on a clone (see `refresh.rs`: "The snapshot pose is
  raw ... no risk of subtracting a previous calibration twice"). The
  calibration modal's capture loop reads the mailbox — if the
  provider published de-rotated skeletons, every *re*-calibration
  would measure anchors/yaw in already-corrected space, converging
  the stored neutral toward zero across recaptures (the
  double-subtraction bug class the raw-mailbox invariant exists to
  prevent).

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

Implementation note: the depth builder publishes joints/fingertips
**anchor-centred** (the anchor is the joint-space origin), so the
single pivoted transform decomposes into two pivot-free pieces —
joints rotate about their own origin, and `root_offset` (an absolute
source-oriented metric position) pivots about the calibrated anchor
`[anchor_x, anchor_y, −anchor_depth_m]` (the same convention as the
solver's root-reference seed). No `mpsu` unit conversion is needed
anywhere.

### Consumer inventory (what rotates, what must NOT)

| Sample field | Rotate? | Why |
|---|---|---|
| `joints` positions | ✓ | isotropic source units (normalised by `mpsu`) — rotation is geometrically valid |
| `fingertips` | ✓ | same space as joints |
| `root_offset` | ✓ | raw-metres camera-space vector; rotation is unit-safe; see units audit below |
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
- `root_offset` (D435 path): raw metres, all three axes (the unit
  contract on `SourceSkeleton::root_offset`) — rotation valid.
- Pivot `a0`: `[anchor_x, anchor_y, −anchor_depth_m]` is in the same
  frame and units as `root_offset` (the anchor fields are captured
  medians of that very value), so the offset pivot needs no unit
  conversion; joints/fingertips are anchor-centred and pivot about
  their own origin. No `mpsu` conversion appears anywhere in the
  transform.
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

Floors and gates:
- ≥ 5 finite samples, else `None` (matches `FACE_NEUTRAL_MIN_SAMPLES`).
- MAD ≤ 0.15 rad across the window (same stability gate as the face
  neutrals): a torso that swivels mid-window must not bake a
  transient heading into every subsequent frame.
- `|θ|` beyond 60° → **reject** (`None`), not clamp: beyond that the
  far shoulder is occlusion-shadowed in depth and L/R swap risk
  dominates, so the reading is more likely garbage than geometry —
  and a clamped value would apply a wrong-*magnitude* rotation every
  frame, which is worse than no rotation. The load path re-clamps
  defensively (hand-edited profile ±180° would flip the scene behind
  the camera).
- Inspector warning above 45°: "camera very oblique — tracking quality
  degrades" (surfaced, not silently altered).
- Sign convention gets dedicated unit tests against synthetic
  skeletons — the selfie-mirror x-flip (`pose_match.rs` convention
  reminder: avatar bone names ≠ subject anatomical sides) makes yaw
  sign the single most likely silent-wrong constant in this design.

### Interactions audited

- **Movement-range step**: `RangeCollecting` folds mailbox `(x, z)`
  into min/max. With de-rotation active at runtime, raw-axis extremes
  describe a camera-aligned box while runtime offsets live in body
  frame — so rotate **each sample** by the de-rotation
  (`rotate_xz(·, θ)`; θ is already on the in-flight calibration
  carried in the variant) *before* folding. Rotating the finished box
  instead would under/over-state the range (an axis-aligned box is
  not rotation-equivariant). The stored ranges are therefore
  **body-frame** — the semantics the instruction implies ("step left,
  lean toward your usual forward" are body-relative actions). Today
  the ranges are summary/persistence-only (see the movement-range
  consumer status above), so this keeps the *data* future-proof
  rather than changing live behaviour.
- **Sensor-matched overlays** (camera-wipe PIP): these show the true camera view; a de-rotated avatar
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

### Surfacing

The measured yaw is shown on the inspector status line ("Body yaw
neutral −28°", amber warning above 45°) — the sanity probe for
checking the reading's sign and magnitude against the physical camera
placement. Outstanding validation is **live**: calibrate at a
physically-known camera offset and compare — if the sign is inverted,
flip `shoulder_line_yaw` *and* its normative sign tests as one
change.

### Verification

- Unit: yaw-measurement sign on synthetic skeletons (both placements;
  mirrored-x convention) — `tracking::calibration::body_yaw_tests`.
- Unit: rigid round-trip — build a frontal reference sample, rotate
  the scene by θ about `a0` (joints, fingertips, hand basis vectors,
  root_offset), apply a calibration carrying `neutral_body_yaw = θ`,
  assert the result equals the frontal reference within epsilon —
  `tracking::calibration_apply_tests`.
- Unit: `None` neutral is a no-op; non-metric samples are never
  rotated even with `Some(θ)`; the face pose is never touched by the
  body-yaw rotation; a subject at the calibrated anchor doesn't move.
- Unit: median/stability/plausibility gates of the yaw aggregate, and
  fold-of-rotated-samples ≠ rotate-the-folded-box —
  `gui::calibration::finalize::tests`.
- Live: `diagnose_fusion_replay` (offline replay) on
  oblique footage; live-debug protocol to read the measured θ and
  the post-de-rotation residual shoulder yaw on the user's actual
  setup.

## Open questions / future work

- **Person segmentation** as an alternative to calibration: a future option could replace this UX with an automatic per-pixel mask that excludes background depth. Calibration remains useful as an explicit baseline regardless.
- **Camera pitch / roll neutral** (the body-yaw neutral deliberately covers yaw only):
  the shoulder line constrains yaw and roll but carries **zero pitch
  information**, and at bust-up framing there is no visible spine axis
  to measure pitch from — a "pitch neutral" would be fit from noise.
  Camera pitch is also observationally entangled with the user's
  habitual lean (a D435 has no IMU to break the tie; the D435i does),
  and shoulder-line roll is already routed into the UpperChest lean by
  the solver rather than the root. If pitch neutralisation is ever
  wanted, it needs either an IMU-equipped camera or a dedicated
  capture step with the user's posture constrained — not a bolt-on to
  the yaw de-rotation transform.
