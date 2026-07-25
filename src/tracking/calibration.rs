//! Per-session pose calibration types.
//!
//! See `docs/calibration-ux.md` for the rationale. The short version:
//! the depth-aware tracking provider reads the subject's anchor from
//! a per-pixel depth map, and a desk / monitor / chair occluding the
//! depth window biases that anchor by 0.3–1.5 m. The bias propagates
//! through every downstream consumer (body yaw, root translation).
//! This module carries the explicit reference values the user
//! captures via the `Calibrate Pose` modal so the solver and
//! `skeleton_from_depth` can seed / gate against a known-good
//! baseline rather than the auto-EMA's first frame.
//!
//! [`super::TrackingCalibration`] carries an optional
//! [`PoseCalibration`]; consumers fall back to the auto-EMA /
//! default behaviour whenever it is absent or
//! [`PoseCalibration::is_active`] returns false.

use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Which pose the subject took during the calibration capture window.
/// Drives both the on-screen instructions and the anchor-selection
/// policy in [`super::skeleton_from_depth`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationMode {
    /// Subject visible from feet to head, T-pose. Hip pair midpoint
    /// (COCO 11/12) is the captured anchor; `skeleton_from_depth`
    /// continues to prefer hip with shoulder fallback.
    FullBody,
    /// Subject visible from chest / waist up, hands at sides. Shoulder
    /// pair midpoint (COCO 5/6) is the captured anchor; once active,
    /// `skeleton_from_depth` is *forced* onto the shoulder anchor even
    /// when the hip pair clears the visibility floor — when the user
    /// said "upper body only", an apparent hip detection is almost
    /// certainly the desk surface or chair seat.
    UpperBody,
}

impl CalibrationMode {
    /// Stable string for serialization. Decoupled from the enum's
    /// `Debug` impl so the on-disk schema is independent of Rust
    /// formatting choices.
    pub fn as_str(self) -> &'static str {
        match self {
            CalibrationMode::FullBody => "full_body",
            CalibrationMode::UpperBody => "upper_body",
        }
    }
}

/// Returned by [`CalibrationMode`]'s [`FromStr`] when the string
/// doesn't match any known variant. Callers (currently
/// [`crate::persistence`]) treat this as "drop the calibration and
/// log a warning" rather than fail-loud, so the existing project
/// file isn't lost on a future schema change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnknownCalibrationMode;

impl FromStr for CalibrationMode {
    type Err = UnknownCalibrationMode;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "full_body" => Ok(CalibrationMode::FullBody),
            "upper_body" => Ok(CalibrationMode::UpperBody),
            _ => Err(UnknownCalibrationMode),
        }
    }
}

/// One calibration capture's median-aggregated values.
///
/// Built by the modal's capture loop from a 2-second collection
/// window of source-skeleton samples; consumed by the solver
/// (root-reference seed, solve-time neutrals via `apply_calibration`)
/// and the depth-aware skeleton builder (anchor forcing, legacy-path
/// scale plausibility).
///
/// Anchor fields are a straight copy of the captured
/// `SourceSkeleton::root_offset` medians and share its unit contract:
/// on the metric D435 path `anchor_x`/`anchor_y` are source-oriented
/// metres; on the legacy rtmw3d-only path they are image-relative
/// source units. `anchor_depth_m` is always the positive camera-space
/// forward distance (source z negated).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoseCalibration {
    pub mode: CalibrationMode,
    /// ISO-8601 capture timestamp. `String` rather than `SystemTime`
    /// because the on-disk schema needs a stable, human-readable
    /// representation that survives `serde_json` / version changes.
    pub captured_at: String,
    /// Unix seconds at capture. Persisted alongside [`Self::captured_at`]
    /// so the inspector's "Calibrated N min ago" relative-time
    /// rendering doesn't have to parse the ISO string each frame.
    /// Both fields are written together by `aggregate()` and consumed
    /// together; if they disagree (manually-edited project file) the
    /// unix value wins.
    #[serde(default)]
    pub captured_at_unix: u64,
    /// Number of frames whose median produced these values. Below 5
    /// indicates a `Capture Now` cut-short with sparse samples; the
    /// inspector flags the calibration as "low-sample" in that case.
    pub frame_count: usize,
    /// Image-centre-relative anchor X.
    pub anchor_x: f32,
    /// Image-centre-relative anchor Y (Y-up convention).
    pub anchor_y: f32,
    /// Camera-space metric depth of the anchor pair midpoint.
    /// `None` for `rtmw3d` (no depth pipeline) — the rtmw3d-only
    /// solver path uses `anchor_x` / `anchor_y` only.
    pub anchor_depth_m: Option<f32>,
    /// Median per-frame `overall_confidence` from the captured samples.
    /// Surfaced on the inspector to flag setups where calibration was
    /// taken under marginal lighting / framing conditions.
    pub confidence: f32,
    /// Per-frame stddev of the captured anchor depth. A
    /// capture-quality indicator (inspector status detail), persisted
    /// for diagnostics — no solver/provider consumer (the D435 metric
    /// path needs no scale calibration). `None` for non-depth
    /// captures.
    pub anchor_depth_jitter_m: Option<f32>,
    /// Median per-subject 3D shoulder span (LeftShoulder ↔
    /// RightShoulder distance) in metres, gathered across the
    /// calibration window. The depth-aware skeleton builder uses it
    /// as `reference_span_m` — the subject's true body scale — which
    /// feeds both the isotropic `mpsu` source-frame normalisation and
    /// the solver's metres→avatar-units factor for 1:1 root
    /// placement. `None` when fewer than 3 frames produced a finite
    /// reading or for older saves that predate the field (the builder
    /// then falls back to the frame's measured span / anatomical
    /// mean).
    #[serde(default)]
    pub shoulder_span_m: Option<f32>,
    /// Optional per-axis peak-to-peak movement range captured by the
    /// optional range step (step left/right, lean in/out). No solver
    /// consumer: metric root translation is 1:1 (a 30 cm side-step is
    /// 30 cm of avatar travel, no room-size gain), so there is no
    /// per-axis sensitivity to derive. Captured for the Done-pane
    /// summary and persistence, stored **body-frame** (samples
    /// rotated by the neutral body yaw before folding) so any future
    /// consumer matches the de-rotated runtime offsets. `None` when
    /// the user skipped the range step.
    #[serde(default)]
    pub x_range_observed: Option<f32>,
    #[serde(default)]
    pub z_range_observed: Option<f32>,
    /// Optional torso depth template captured during calibration.
    /// Stores the median DAv2 depth across the calibration window for
    /// each cell of a low-resolution grid covering the torso bounding
    /// box (between shoulder and hip keypoints). When present,
    /// `skeleton_from_depth` computes a per-frame "torso depth bias"
    /// (median of `current_depth − template_depth` over visible cells)
    /// and corrects shoulder / hip samples with it, recovering the
    /// real torso depth even when an occluder (hand crossing in front
    /// of chest, palms covering shoulders) dominates the keypoint's
    /// own sample window. See `docs/calibration-ux.md` for the
    /// rationale and `Step 1` minimal-version implementation notes.
    #[serde(default)]
    pub torso_depth_template: Option<TorsoDepthTemplate>,
    /// Resting (neutral) expression weights sampled during the same
    /// calibration window, keyed by VRM expression name (e.g. "aa",
    /// "blink"). Captured only when face tracking is active while the
    /// user holds the neutral calibration pose — empty otherwise.
    ///
    /// [`crate::tracking::TrackingCalibration::apply_calibration`]
    /// subtracts this baseline from the live source expressions and
    /// rescales the remaining headroom, so a face that rests with a
    /// slightly open mouth or narrowed eyes maps to a neutral avatar
    /// and still reaches full open/blink. Per-person eye/mouth aperture
    /// differs, so a fixed `weight = raw` mapping is wrong without this.
    /// Stored as an ordered `Vec` (not a map) for stable on-disk JSON.
    #[serde(default)]
    pub neutral_expressions: Vec<(String, f32)>,
    /// Resting head pose `[yaw, pitch, roll]` (radians) captured from
    /// the **FaceMesh** estimator across the calibration window while
    /// the user holds their natural "facing forward" posture.
    /// Subtracted from live *mesh-sourced* face poses by
    /// [`crate::tracking::TrackingCalibration::apply_calibration`] so
    /// "neutral" really is `yaw = pitch = roll = 0` *for this person's
    /// setup*: a camera mounted off eye line (moderately — the atan
    /// saturation and mesh profile collapse cap the correctable offset
    /// at roughly ±20°) otherwise bakes a constant head turn/tilt into
    /// the avatar.
    ///
    /// Per-source split rationale: the mesh and body estimators carry
    /// *different* systematic residuals (different landmark sets and
    /// hardcoded anatomical neutrals), so a neutral measured from one
    /// must never be subtracted from the other — that turns every
    /// runtime source switch into a visible head step. `None` when too
    /// few confident mesh frames were seen during the hold; subtraction
    /// for that source is then a no-op (the hardcoded anatomical
    /// neutral inside the estimator still applies).
    #[serde(default)]
    pub neutral_face_ypr_mesh: Option<[f32; 3]>,
    /// Resting head pose from the **body** (RTMW3D ear/eye-line)
    /// estimator over the same window — see
    /// [`Self::neutral_face_ypr_mesh`] for the per-source rationale.
    /// Accumulated from `SourceSkeleton::face_body_raw` so a frontal
    /// hold (where the mesh wins selection nearly every frame) still
    /// measures the body-path residual.
    #[serde(default)]
    pub neutral_face_ypr_body: Option<[f32; 3]>,
    /// Neutral **body** yaw (radians): the horizontal angle of the
    /// user's shoulder line relative to camera-frontal while they hold
    /// their habitual working orientation ("face your usual forward",
    /// which at an oblique camera placement is NOT the camera). Median
    /// of [`shoulder_line_yaw`] across the capture window; the torso
    /// counterpart of [`Self::neutral_face_ypr_mesh`]. Consumed by
    /// [`super::TrackingCalibration::apply_calibration`], which
    /// rigidly re-expresses the whole published sample by
    /// [`rotate_xz`]`(·, θ)` — "as if the camera had been frontal" —
    /// so the solver's shoulder-line yaw, direction-matched bones and
    /// arm IK targets all see a de-rotated scene consistently. See
    /// `docs/calibration-ux.md` "Neutral body yaw" for why root-only subtraction
    /// and provider-side rotation were rejected.
    ///
    /// Sign convention: positive when the user's shoulder line tilts
    /// toward +z at its +x end (see [`shoulder_line_yaw`]); pinned by
    /// unit tests because the selfie-mirror x-flip makes this the most
    /// likely silently-wrong constant in the pipeline.
    ///
    /// `None` for captures without a metric depth backend, captures
    /// with fewer than [`BODY_YAW_MIN_SAMPLES`] valid frames, and all
    /// older saves that predate the field — each of which must behave exactly like
    /// today (no rotation applied).
    #[serde(default)]
    pub neutral_body_yaw: Option<f32>,
}

/// Minimum per-frame yaw readings for a capture to publish a
/// [`PoseCalibration::neutral_body_yaw`]. Matches the face-neutral
/// floor: below this the estimate is one noisy reading, and baking it
/// into every subsequent frame is worse than the uncorrected constant.
pub const BODY_YAW_MIN_SAMPLES: usize = 5;

/// Hard cap on a stored neutral body yaw. Beyond ~60° the far
/// shoulder is occlusion-shadowed in the depth map and L/R-swap risk
/// dominates: a larger reading is more likely detector garbage than
/// camera geometry. The GUI additionally warns (without clamping)
/// above [`BODY_YAW_WARN_RAD`].
pub const BODY_YAW_MAX_RAD: f32 = std::f32::consts::PI / 3.0;

/// Inspector-warning threshold (45°): tracking still runs but depth
/// shadowing measurably degrades the far arm; the user should know
/// their camera is very oblique rather than silently getting worse
/// output.
pub const BODY_YAW_WARN_RAD: f32 = std::f32::consts::PI / 4.0;

/// Minimum horizontal (x, z) shoulder-line length, in isotropic
/// source units, for its yaw to be well-defined. Same floor as the
/// solver's `compute_shoulder_yaw_rotation` — a near-vertical
/// shoulder line has no meaningful heading.
const YAW_MIN_HORIZ_SPAN: f32 = 0.20;

/// Horizontal shoulder-line yaw of a published sample, in radians:
/// `atan2(Δz, Δx)` of `LeftUpperArm − RightUpperArm` (avatar bone
/// names — the source frame is selfie-mirrored, so `LeftUpperArm`
/// sits at +x for a camera-facing subject and a frontal hold reads
/// `0`). Positive = the +x shoulder is tilted toward +z (source +z
/// points toward the camera, so the +x shoulder is the *closer* one).
/// The unit tests are the normative statement of the sign.
///
/// `None` when the sample has no metric backend (monocular z is not
/// trustworthy enough to measure a constant we then subtract forever),
/// either shoulder is missing or below the keypoint floor, or the
/// horizontal span is degenerate.
pub fn shoulder_line_yaw(sample: &super::SourceSkeleton) -> Option<f32> {
    use crate::asset::HumanoidBone;
    sample.metric_frame_info.as_ref()?;
    let l = sample.joints.get(&HumanoidBone::LeftUpperArm)?;
    let r = sample.joints.get(&HumanoidBone::RightUpperArm)?;
    if l.confidence < 0.3 || r.confidence < 0.3 {
        return None;
    }
    let dx = l.position[0] - r.position[0];
    let dz = l.position[2] - r.position[2];
    let span = (dx * dx + dz * dz).sqrt();
    if !span.is_finite() || span < YAW_MIN_HORIZ_SPAN {
        return None;
    }
    Some(dz.atan2(dx))
}

/// Rotate `v` by `theta` radians about the +Y (vertical) axis, i.e.
/// in the horizontal (x, z) plane: `x' = x·cosθ + z·sinθ`,
/// `z' = −x·sinθ + z·cosθ`. Chosen so that
/// `rotate_xz(shoulder_line, shoulder_line_yaw(...))` maps the
/// shoulder line back to frontal (+x, z = 0) — the de-rotation
/// direction. Y passes through untouched.
pub fn rotate_xz(v: [f32; 3], theta: f32) -> [f32; 3] {
    let (s, c) = theta.sin_cos();
    [v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c]
}

/// Calibrated torso surface depth profile. Captured during the
/// calibration window from the DAv2 depth map within the torso
/// bounding box (between shoulder and hip keypoints), median-
/// aggregated across all frames where the four torso anchors
/// (shoulders + hips) cleared the visibility floor.
///
/// **Why a small grid (32×32)**: the torso surface is approximately
/// rigid relative to the body (clothing introduces millimetre-scale
/// variation, but the chest / belly / shoulder geometry is
/// per-individual fixed). 32×32 is enough resolution to capture
/// the chest-to-belly slope and the rough left-right asymmetry of
/// typical poses (one shoulder slightly forward) without bloating
/// the calibration file (4 KB / capture). At inference time, only
/// the cells whose current-frame depth lands within a "person
/// surface band" of the template depth contribute to the bias
/// computation — occluder pixels are automatically rejected.
///
/// **Coordinate system**: cells are indexed in row-major order with
/// `(0, 0)` at the bbox top-left in image space. `bbox_normalized`
/// records the bbox extents in image-relative coordinates (matching
/// the keypoints' nx/ny convention) so the template can be
/// re-projected onto the same image region at inference time even
/// when the subject moves slightly. Cells where the template depth
/// could not be recovered (depth-map void, occluder during the
/// entire calibration window) hold `f32::NAN` and are skipped by
/// the inference-time bias computation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TorsoDepthTemplate {
    /// Cell-grid resolution. Always 32 × 32 for the Step 1 minimal
    /// version; expressed as fields rather than a const so future
    /// experiments with finer grids (e.g. 64×64 for richer registration
    /// in Step 3) can land without a serde-format break.
    pub width: u32,
    pub height: u32,
    /// Per-cell median depth in metres, row-major, length =
    /// `width × height`. `f32::NAN` for cells with no valid sample
    /// (depth-map void or persistent occluder during calibration);
    /// inference-time consumers must skip these cells.
    pub depths_m: Vec<f32>,
    /// Bounding box of the captured torso region in image-relative
    /// coordinates `(min_nx, min_ny, max_nx, max_ny)` with the same
    /// nx/ny convention as the keypoints (nx ∈ [0, 1], ny ∈ [0, 1],
    /// origin top-left). The inference-time consumer re-uses the
    /// *current frame's* shoulder/hip keypoints to define a fresh
    /// bbox, then re-samples the template by mapping each
    /// current-bbox cell back into template-bbox coordinates.
    pub bbox_normalized: [f32; 4],
}

impl TorsoDepthTemplate {
    /// Cell-grid resolution used by Step 1. Exposed as a constant so
    /// the GUI / provider capture loops can pre-size the accumulation
    /// buffers without re-checking the field at every frame.
    pub const GRID_SIZE: u32 = 32;
}

#[cfg(test)]
mod body_yaw_tests {
    use super::*;
    use crate::asset::HumanoidBone;
    use crate::tracking::source_skeleton::{SourceJoint, SourceSkeleton};

    fn skeleton_with_shoulders(l: [f32; 3], r: [f32; 3]) -> SourceSkeleton {
        let mut sk = SourceSkeleton::default();
        for (bone, pos) in [
            (HumanoidBone::LeftUpperArm, l),
            (HumanoidBone::RightUpperArm, r),
        ] {
            sk.joints.insert(
                bone,
                SourceJoint {
                    position: pos,
                    confidence: 0.9,
                    metric_depth_m: None,
                },
            );
        }
        sk.stamp_synthetic_metric_frame();
        sk
    }

    #[test]
    fn frontal_hold_reads_zero() {
        let sk = skeleton_with_shoulders([0.25, 0.0, 0.0], [-0.25, 0.0, 0.0]);
        let yaw = shoulder_line_yaw(&sk).unwrap();
        assert!(yaw.abs() < 1e-6, "frontal must be 0, got {yaw}");
    }

    #[test]
    fn yaw_sign_is_positive_when_plus_x_shoulder_is_at_plus_z() {
        // Normative sign statement: +x shoulder pushed toward +z →
        // positive yaw. Everything downstream (de-rotation direction,
        // range folding) is defined against this.
        let sk = skeleton_with_shoulders([0.25, 0.0, 0.1], [-0.25, 0.0, -0.1]);
        let yaw = shoulder_line_yaw(&sk).unwrap();
        assert!(yaw > 0.05, "expected positive yaw, got {yaw}");
        // Mirror-image placement flips the sign.
        let sk = skeleton_with_shoulders([0.25, 0.0, -0.1], [-0.25, 0.0, 0.1]);
        let yaw = shoulder_line_yaw(&sk).unwrap();
        assert!(yaw < -0.05, "expected negative yaw, got {yaw}");
    }

    #[test]
    fn rotate_by_measured_yaw_restores_frontal() {
        // De-rotation contract: rotate_xz(·, θ) with θ measured by
        // shoulder_line_yaw maps the shoulder line back to frontal.
        let sk = skeleton_with_shoulders([0.23, 0.0, 0.13], [-0.21, 0.0, -0.09]);
        let yaw = shoulder_line_yaw(&sk).unwrap();
        let l = rotate_xz([0.23, 0.0, 0.13], yaw);
        let r = rotate_xz([-0.21, 0.0, -0.09], yaw);
        let dz = l[2] - r[2];
        assert!(dz.abs() < 1e-6, "residual Δz after de-rotation: {dz}");
        assert!(l[0] - r[0] > 0.0, "shoulder line must stay +x");
    }

    #[test]
    fn non_metric_sample_measures_none() {
        let mut sk = skeleton_with_shoulders([0.25, 0.0, 0.1], [-0.25, 0.0, -0.1]);
        sk.metric_frame_info = None;
        assert!(shoulder_line_yaw(&sk).is_none());
    }

    #[test]
    fn degenerate_span_measures_none() {
        // Near-vertical shoulder line (profile view collapse): heading
        // undefined, must refuse rather than emit ±90° garbage.
        let sk = skeleton_with_shoulders([0.02, 0.3, 0.01], [-0.02, -0.3, -0.01]);
        assert!(shoulder_line_yaw(&sk).is_none());
    }

    #[test]
    fn rotate_xz_preserves_y_and_length() {
        let v = [0.3, -0.7, 0.2];
        let out = rotate_xz(v, 0.83);
        assert_eq!(out[1], v[1]);
        let len_in = (v[0] * v[0] + v[2] * v[2]).sqrt();
        let len_out = (out[0] * out[0] + out[2] * out[2]).sqrt();
        assert!((len_in - len_out).abs() < 1e-6);
    }
}

impl PoseCalibration {
    /// `true` once a capture has completed and at least one frame was
    /// gathered. Used by consumers to decide between calibration-aware
    /// and auto-EMA / hardcoded-clamp behaviour.
    pub fn is_active(&self) -> bool {
        self.frame_count > 0
    }

    /// The neutral head pose measured for `source`, if the calibration
    /// hold gathered enough confident frames from that estimator.
    /// Subtraction must be keyed on the live pose's own source — see
    /// [`Self::neutral_face_ypr_mesh`].
    pub fn neutral_face_ypr_for(
        &self,
        source: super::source_skeleton::FaceSource,
    ) -> Option<[f32; 3]> {
        match source {
            super::source_skeleton::FaceSource::Mesh => self.neutral_face_ypr_mesh,
            super::source_skeleton::FaceSource::Body => self.neutral_face_ypr_body,
        }
    }
}
