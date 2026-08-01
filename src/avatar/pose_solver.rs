//! Solve avatar bone rotations from a [`SourceSkeleton`].
//!
//! The solver consumes 3D joint positions directly (the upstream tracker
//! is responsible for producing depth — see
//! [`crate::tracking::rtmw3d::Rtmw3dInference`], which drives the
//! RTMW3D whole-body 3D pipeline). Three ideas:
//!
//! 1. **Rest-relative.** Every output rotation is expressed as a delta
//!    from the avatar's rest pose, so models whose bones have non-identity
//!    rest rotations (most real VRMs) are not destroyed by the write.
//!
//! 2. **Direction matching.** For chain bones (upper/lower arm, leg,
//!    spine) we read the bone's *rest world direction* (from its world
//!    position to its tip's world position) and rotate it to match the
//!    source skeleton's observed direction in camera space. The
//!    minimum-arc quaternion is written as the world-space delta; it is
//!    then conjugated into the bone's parent-local frame so the result
//!    respects the skeleton hierarchy.
//!
//! 3. **Face pose is independent.** The head rotation comes from facial
//!    keypoints (yaw/pitch/roll), not from the nose-to-shoulder vector.
//!    Those angles are absolute in the source (camera) frame, so the Head
//!    bone's WORLD orientation is set from them directly (the parent
//!    chain's rotation is divided out rather than composed with) — see
//!    `apply_face_pose`. The neck/spine chain still carries torso posture;
//!    it just no longer double-counts into where the head looks.
//!
//! The solver assumes `local_transforms` has already been reset to the
//! skeleton's rest pose (via `AvatarInstance::build_base_pose`). It only
//! writes rotations; translations and scales are left untouched.
//!
//! ## Source coordinate convention
//!
//! `SourceJoint::position` is `[x, y, z]` with x ∈ [-aspect, +aspect],
//! y ∈ [-1, 1] Y-up, and z in the same units with `+z` toward the
//! camera. The Y180 root flip baked onto VRM 0.x rigs already aligns
//! "avatar faces camera" with `+z = toward camera`, so direction
//! matching against rest_world directions works uniformly across VRM
//! 0.x and 1.x without a separate handedness toggle.

use std::collections::HashMap;
use std::time::Instant;

use crate::asset::{HumanoidBone, HumanoidMap, NodeId, Quat, SkeletonAsset, Transform, Vec3};
use crate::math_utils::{
    quat_conjugate, quat_from_basis_pair, quat_from_euler_ypr, quat_from_vectors, quat_mul,
    quat_normalize, quat_rotate_vec3, swing_twist_decompose, vec3_add, vec3_cross, vec3_dot,
    vec3_length, vec3_normalize, vec3_scale, vec3_sub,
};
use crate::tracking::source_skeleton::{FacePose, HandOrientation, SourceSkeleton};

/// Reference frame period used to convert the GUI's frame-rate-naive
/// `rotation_blend` slider into a time-constant. With the slider at
/// `b` and the system actually running at `dt_ref`, the per-frame α
/// becomes `1 − exp(−dt/τ)` with `τ = −dt_ref / ln(1 − b)`. The
/// slider semantics stay intact — `b` reads as "what α you'd see at
/// the reference rate" — but the actual smoothing is now invariant
/// to frame-rate jitter.
const ROTATION_BLEND_REFERENCE_DT: f32 = 1.0 / 30.0;

/// Schmitt hysteresis ratio. A joint becomes "active" when its
/// confidence rises above `threshold` and stays active until it
/// drops below `threshold * SCHMITT_EXIT_RATIO`. Prevents chatter
/// when raw confidence wiggles around the threshold boundary.
const SCHMITT_EXIT_RATIO: f32 = 0.8;

/// 1€ filter constants. `min_cutoff` is the cutoff at zero motion
/// (lower → smoother at rest); `beta` is the velocity-coupling gain
/// (higher → faster response when the joint moves); `d_cutoff` is
/// the cutoff used to low-pass the velocity estimate. Defaults
/// follow the original 1€ paper's pose-tracking recommendations.
const ONE_EURO_MIN_CUTOFF_HZ: f32 = 1.0;
const ONE_EURO_BETA: f32 = 0.05;
const ONE_EURO_D_CUTOFF_HZ: f32 = 1.0;

/// Per-channel 1€ tuning for the wrists + fingers. Monocular hand
/// keypoints measure 3–5× noisier than the torso at the detector
/// (measured via `diagnose_signal_quality`: SRC `jitter_hf` 0.09–0.11
/// on hands/forearms vs 0.02–0.04 on head/torso), and after the solver
/// the hands are still the jitteriest bones (AV `jitter_hf` ~0.036 and
/// `wander_lf` ~0.11, ~3–4× the torso). A uniform filter can't win
/// that: the cutoff that keeps the latency-critical head crisp leaves
/// the hands buzzing. So the hand tier gets a lower rest cutoff (more
/// smoothing of the in-place oscillation) and a higher velocity gain so
/// intentional fast motion — a strum, a wave — still tracks through the
/// 1€ speed coupling, which the hand can afford far more than the head.
const HAND_MIN_CUTOFF_HZ: f32 = 0.5;
const HAND_BETA: f32 = 0.10;

/// Rest deadband radius for the hand tier, in source-skeleton units.
/// The 1€ tuning above is *lever-arm-limited* — it cut the solved hand
/// jitter only ~5% because the hand's world position is dominated by
/// upstream arm-angle noise amplified over a ~0.5 m lever, not by the
/// wrist keypoint directly. The deadband attacks what's left at the
/// source: while a joint is genuinely at rest, freeze any per-frame
/// movement under this radius so both the micro-oscillation ("same
/// place but shaking") and the slow drift stop leaking through. `0`
/// disables it (the body default) — the head/torso must stay live.
const HAND_DEAD_RADIUS: f32 = 0.02;

/// Rest-gate speed band (source units/sec) over which the deadband and
/// the arm rotation hold fade out. Below `LO` the joint is fully at rest
/// (hold at full strength); above `HI` it is moving and the hold is off,
/// smoothstep-faded in between so there is no chatter or snap.
///
/// The gate keys off the 1€ *output* speed — the filter's own smoothed
/// position step per frame — NOT the d_cutoff-smoothed raw velocity in
/// `vel`. Measured on real footage, the raw hand keypoint is so noisy
/// that even after the 1 Hz velocity low-pass its speed reads median
/// ~2.3 units/s *at rest* (jitter does not fully cancel), so a gate on
/// `vel` almost never opened (full hold on only 10% of frames). The
/// filter *output* separates cleanly: at rest it converges and its step
/// speed collapses, while intentional motion tracks through. The band is
/// sized from the measured class split on real footage (wrist/elbow
/// output speed: at-rest median ≈ 0.10, p75 ≈ 0.14–0.18; in-motion
/// median ≈ 0.31–0.42): below `LO` is firmly inside the rest cluster,
/// above `HI` firmly in motion, and the overlap zone gets a partial,
/// smoothstep-faded hold.
///
/// `HI` sits at the *bottom* of the motion cluster (≈ the in-motion
/// median), not above it: a band that ran past the motion speeds would
/// leave `rest > 0` during ordinary deliberate gestures, and the hold's
/// soft threshold would then shrink small per-frame rotation steps of a
/// *real* slow move — the arm would lag and catch up in jumps. `LO`
/// sits at the rest cluster's p75 so the full hold spans the whole rest
/// distribution. The narrow overlap is intentional (the two clusters
/// nearly touch); the smoothstep keeps the transition chatter-free.
const REST_SPEED_LO: f32 = 0.18;
const REST_SPEED_HI: f32 = 0.32;

/// End-effector rotation hold for the arm chain, in radians of per-frame
/// rotation step frozen while the wrist is at rest. The source-keypoint
/// filters (1€ + deadband) hit a ceiling on hand *world* stability
/// because the hand sits at the end of a ~0.5 m lever: tiny angle noise
/// on the shoulder/upper-arm — bones the source filters never touch — is
/// amplified into large hand displacement (measured: the rest deadband
/// cut forearm rest jitter −6% but the hand only −0%, because the hand is
/// upstream-dominated). This hold attacks it in rotation space, *after*
/// that amplification: while the wrist reads at rest, any per-frame
/// rotation step below this angle on the arm chain (upper arm → forearm →
/// hand) is frozen, so upstream jitter can no longer reach the hand. It
/// gates the freeze on step *magnitude*, so a large intentional pose
/// change always passes (its step dwarfs this radius) — only the residual
/// buzz of a converged, held pose is frozen. The rest gate reuses
/// [`REST_SPEED_LO`]/[`REST_SPEED_HI`] on the wrist filter's output speed
/// (same signal, same rest concept), and the release is a smoothstep, so
/// a deliberate move is never frozen and there is no snap on release.
/// Sized to the measured buzz: the at-rest UpperArm residual works out
/// to ~2.4° of equivalent per-frame rotation (0.0105 units over a
/// ~0.25 m bone), so the first attempt at 0.5° froze almost none of it.
const ARM_HOLD_ANG_DEAD: f32 = 0.035; // ≈ 2.0°

/// A-pose idle angle for an UNTRACKED arm, measured from straight-down (0°) in
/// the frontal plane. A relaxed A-pose sits the arm ~40° out from the body; the
/// alternative — the model's T-pose bind — juts it straight out (~90°) and reads
/// as broken the moment tracking drops. See `apply_idle_arm_pose`.
const IDLE_ARM_ANGLE_FROM_DOWN_DEG: f32 = 40.0;

/// Rest deadband on expression weights (`solve_expressions`, eye/brow
/// path). The FaceMesh blink/eye blendshapes jitter ±0.04 per frame at
/// rest — the visible eye "twitch" — while a real blink is a Δ≈1.0 jump.
/// A soft threshold on the per-frame weight *delta* freezes sub-threshold
/// flutter and passes real expressions unshrunk; because it gates on the
/// delta magnitude it is self-gating (no separate rest-speed signal, which
/// a 0–1 weight channel doesn't have). Sized just above the measured
/// flutter so genuine expression onsets are untouched.
const EXPR_REST_DEAD: f32 = 0.06;

/// Resolved 1€ + deadband tuning for one channel. The XY plane and the
/// Z (depth) axis are tuned independently: the source frame is
/// camera-aligned (+Z = away from camera), and monocular Z is the
/// lowest-information axis — measured per-axis on real footage, source
/// Z jitter runs 1.4–3× the XY axes (head worst at 3×) and the at-rest
/// slow wander is Z-dominated on every bone. Z is also the axis where
/// latency is perceptually cheapest (a depth lag is nearly invisible
/// on screen), so Z can afford a much lower cutoff and a wider deadband
/// without the response cost that would make XY feel draggy.
#[derive(Clone, Copy)]
struct OneEuroTuning {
    min_cutoff: f32,
    beta: f32,
    /// Rest deadband radius on the XY step (source units); `0` disables.
    dead_radius: f32,
    /// 1€ rest cutoff for the Z axis.
    z_min_cutoff: f32,
    /// 1€ velocity gain for the Z axis. Sized against the *noise floor*
    /// of the smoothed Z velocity (≈1.0 units/s on the hands, ≈0.45 on
    /// the head — pure detector jitter that never cancels), so noise
    /// alone cannot open the filter but a genuinely fast depth move
    /// (a lean, a reach toward the camera) still raises the cutoff.
    z_beta: f32,
    /// Rest deadband radius on the Z step (source units); `0` disables.
    z_dead_radius: f32,
}

impl OneEuroTuning {
    /// The body default: standard 1€ on XY plus a small velocity-gated
    /// XY rest deadband, heavy Z smoothing + small rest deadband.
    ///
    /// The XY deadband is the shared-root attack on rest jitter: the torso
    /// / shoulder / neck / head keypoints carry a ~0.008–0.011 (AV
    /// jitter_hf) floor that every downstream bone inherits — it reaches
    /// the head directly and the hands amplified over the arm lever, so a
    /// still torso quiets both. It is gated on the joint's own filter
    /// output speed (see [`REST_SPEED_LO`]/`HI`), so it only freezes a
    /// genuinely at-rest joint and releases the instant real motion
    /// starts — this is compatible with keeping the torso "live" for head
    /// latency (that constraint forbids a heavier *low-pass*, not a
    /// rest-only freeze). Sized well below the hand tier's 0.02 since the
    /// torso keypoints are the cleanest.
    const BODY: Self = Self {
        min_cutoff: ONE_EURO_MIN_CUTOFF_HZ,
        beta: ONE_EURO_BETA,
        dead_radius: 0.012,
        z_min_cutoff: 0.15,
        z_beta: 0.15,
        z_dead_radius: 0.01,
    };
    /// The wrist/forearm/finger tier: smoother XY 1€ + rest deadband,
    /// and the heaviest Z treatment (hands measure the noisiest Z).
    const HAND: Self = Self {
        min_cutoff: HAND_MIN_CUTOFF_HZ,
        beta: HAND_BETA,
        dead_radius: HAND_DEAD_RADIUS,
        z_min_cutoff: 0.10,
        z_beta: 0.15,
        z_dead_radius: 0.02,
    };
    /// The root-translation channel (`root_offset`). The root is a
    /// presence channel, not a gesture channel — per the signal-quality
    /// design discussion it can afford to be sluggish, and its wobble
    /// moves the *whole* avatar, so it gets low cutoffs and a deadband
    /// on both planes.
    const ROOT: Self = Self {
        min_cutoff: 0.3,
        beta: 0.05,
        dead_radius: 0.01,
        z_min_cutoff: 0.1,
        z_beta: 0.1,
        z_dead_radius: 0.02,
    };
    /// The face-pose angle tier (yaw, pitch, roll), packed so roll rides
    /// the `z_*` slot. The head-orientation channel had *no* 1€ at all —
    /// raw yaw/pitch/roll went straight to the head bone — and measures
    /// the noisiest angular signal on the avatar (SRC roll ≈ 4°/frame at
    /// rest, velocity fully noise-dominated). yaw/pitch get a standard
    /// rest cutoff; roll gets the heaviest smoothing because it is both
    /// the noisiest and the least intentional (people rarely roll their
    /// head deliberately). `beta` is kept low so the noise-dominated
    /// angular velocity cannot open the cutoff on jitter alone.
    const FACE: Self = Self {
        min_cutoff: 1.0,
        beta: 0.15,
        dead_radius: 0.0,
        z_min_cutoff: 0.5,
        z_beta: 0.10,
        z_dead_radius: 0.0,
    };
}

#[inline]
fn smoothstep(x: f32, lo: f32, hi: f32) -> f32 {
    if hi <= lo {
        return if x < lo { 0.0 } else { 1.0 };
    }
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// True for the distal-arm bones that get the [`HAND_MIN_CUTOFF_HZ`] /
/// [`HAND_BETA`] 1€ tuning instead of the body default: the wrists, the
/// **forearms/elbows** (`*LowerArm`), and all fingers. The forearms are
/// included because the avatar hand's world position is
/// `elbow + L·normalize(wrist − elbow)` — the elbow keypoint's noise
/// (measured just as high as the wrist's, SRC `jitter_hf` ~0.09)
/// contributes to the hand position *twice* (elbow origin + forearm
/// direction), so smoothing only the wrist barely moved the hand (−5%
/// in the first pass). The `*UpperArm`/`*Shoulder` bones stay on the
/// body default — they measure clean (~0.022) and are more structural.
/// The fingertip auxiliary map is keyed by the distal finger bones, so
/// it resolves through this predicate too.
fn is_hand_or_finger(bone: HumanoidBone) -> bool {
    use HumanoidBone::*;
    matches!(
        bone,
        LeftHand
            | RightHand
            | LeftLowerArm
            | RightLowerArm
            | LeftThumbProximal
            | LeftThumbIntermediate
            | LeftThumbDistal
            | LeftIndexProximal
            | LeftIndexIntermediate
            | LeftIndexDistal
            | LeftMiddleProximal
            | LeftMiddleIntermediate
            | LeftMiddleDistal
            | LeftRingProximal
            | LeftRingIntermediate
            | LeftRingDistal
            | LeftLittleProximal
            | LeftLittleIntermediate
            | LeftLittleDistal
            | RightThumbProximal
            | RightThumbIntermediate
            | RightThumbDistal
            | RightIndexProximal
            | RightIndexIntermediate
            | RightIndexDistal
            | RightMiddleProximal
            | RightMiddleIntermediate
            | RightMiddleDistal
            | RightRingProximal
            | RightRingIntermediate
            | RightRingDistal
            | RightLittleProximal
            | RightLittleIntermediate
            | RightLittleDistal
    )
}

/// 1€ + deadband tuning for a bone: the hand tier for wrists/forearms/
/// fingers, the body default for everything else.
fn one_euro_params_for(bone: HumanoidBone) -> OneEuroTuning {
    if is_hand_or_finger(bone) {
        OneEuroTuning::HAND
    } else {
        OneEuroTuning::BODY
    }
}

/// Input knobs for the solver. All fields have sensible defaults; callers
/// only need to tweak when surfacing UI sliders.
#[derive(Clone, Debug)]
pub struct SolverParams {
    /// Per-frame blend factor toward the new rotation, in `[0, 1]`. `0.0`
    /// keeps the previous rotation (no motion), `1.0` snaps instantly to
    /// the solver's output.
    pub rotation_blend: f32,
    /// Minimum source-joint confidence below which a joint is ignored.
    pub joint_confidence_threshold: f32,
    /// Minimum face-pose confidence below which the face pose is ignored.
    pub face_confidence_threshold: f32,
    /// Drive the wrist + 30 finger bones from the source skeleton. When
    /// false the wrist is left at whatever shortest-arc the LowerArm
    /// chain settled it at and the fingers stay at rest pose.
    pub hand_tracking_enabled: bool,
    /// Apply the head-bone face-pose rotation derived from facial
    /// landmarks. When false the head bone is left at rest (parented
    /// to whatever the spine chain produced).
    pub face_tracking_enabled: bool,
    /// Drive the upper / lower leg and foot bones. When false the legs
    /// stay at rest pose regardless of what the tracker emits — useful
    /// when only the upper body is in frame and the user wants to keep
    /// the avatar's legs static rather than have hallucinated COCO
    /// 11-22 keypoints flail them around.
    pub lower_body_tracking_enabled: bool,
    /// Translate the avatar's `Hips` bone in addition to rotating it,
    /// so the avatar follows the subject's side-step / lean / crouch.
    /// When false the avatar pivots in place — useful when the rendered
    /// output should stay framed regardless of how the subject drifts
    /// in front of the camera.
    pub root_translation_enabled: bool,
    /// Optional pose-calibration capture from the `Calibrate Pose ▼`
    /// modal. When present, the solver uses the captured anchor as
    /// the explicit EMA reference seed for root translation —
    /// replacing the auto-EMA's "first hip-visible frame" heuristic
    /// with a user-acknowledged neutral position. Falls through to
    /// the auto-EMA when `None`. See `docs/calibration-ux.md`.
    pub pose_calibration: Option<crate::tracking::PoseCalibration>,
    /// Fill a dropped elbow with a two-bone IK toward the wrist so a
    /// recognised hand doesn't leave the arm at a bind T-pose (see
    /// `compute_arm_reach_elbows`). Default `true`; exposed so the live debug
    /// channel can A/B it.
    pub arm_reach_ik_enabled: bool,
    /// Blend the arm directions toward a positional two-bone solve when the
    /// hands are close, so touching hands meet (see `compute_arm_contact_ik`).
    /// Default `true`; exposed so the live debug channel can isolate whether
    /// this stage is the one crossing the arms at the midline.
    pub contact_ik_enabled: bool,
    /// Horizon (seconds) over which the root-translation neutral re-centres
    /// toward the subject's current position. `None` (default) = **full mirror**:
    /// after a brief startup lock-in the neutral FREEZES, so a real side-step /
    /// lean / walk-in persists on the avatar instead of drifting back to centre
    /// (metric depth is absolute — there is no drift to absorb, so the old ~10 s
    /// self-recentring fought the true distance signal). `Some(h)` restores that
    /// self-recentring EMA for a framed-avatar mode that should stay centred.
    pub root_recenter_horizon_s: Option<f32>,
    /// Rest an UNTRACKED arm in a relaxed A-pose (arms angled down) instead of
    /// leaving it at the model's bind, which for the common T-pose rig juts the
    /// arm straight out and snaps in the instant the arm leaves frame. When on,
    /// an undriven arm fades to the A-pose via the same blend as a tracked one,
    /// so drop-out and re-acquire are smooth. Tracked arms are unaffected (their
    /// direction-match is bind-relative). Default `true`; exposed for A/B.
    pub idle_arm_apose_enabled: bool,
}

impl Default for SolverParams {
    fn default() -> Self {
        // Mirrors `TrackingSmoothingParams::default`: confidence
        // thresholds default to `0.0` so the GUI sliders are the
        // single source of truth for filtering. See the explanation
        // there for the rationale. Retargeting flags default to `true`
        // so non-GUI callers (image-based diagnose scripts, tests) get
        // the full pipeline; the GUI passes whatever the user toggled.
        Self {
            rotation_blend: 0.7,
            joint_confidence_threshold: 0.0,
            face_confidence_threshold: 0.0,
            hand_tracking_enabled: true,
            face_tracking_enabled: true,
            lower_body_tracking_enabled: true,
            root_translation_enabled: true,
            pose_calibration: None,
            arm_reach_ik_enabled: true,
            contact_ik_enabled: true,
            root_recenter_horizon_s: None,
            idle_arm_apose_enabled: true,
        }
    }
}

/// Per-frame state carried between solver calls for temporal smoothing.
///
/// With the upstream tracker now emitting 3D positions directly, this
/// no longer needs the per-bone running max calibration / depth sign
/// tracking that the old 2D→3D lift required. What remains is the 1€
/// filter (smooths 3D positions before they hit direction matching),
/// Schmitt hysteresis (prevents confidence-flicker chatter), and the
/// previous-solve timestamp used to derive `dt`.
#[derive(Clone, Debug, Default)]
pub struct PoseSolverState {
    /// Per-bone 1€ filter state for joint position smoothing.
    joint_filters: HashMap<HumanoidBone, OneEuroFilterState>,
    /// Separate filter state map for fingertip auxiliary positions
    /// (keyed by the distal bone, same as `SourceSkeleton::fingertips`).
    fingertip_filters: HashMap<HumanoidBone, OneEuroFilterState>,
    /// 1€ filter state for the hand-orientation forward/up vectors,
    /// indexed `[left_forward, left_up, right_forward, right_up]`. The
    /// wrist rotation reads `HandOrientation` directly and the palm-normal
    /// cross product is very noisy, so held finger poses (thumbs-up,
    /// pointing) jittered until this smoothing was added. Smoothed
    /// per-component then re-normalised; at rest (≈zero velocity) the 1€
    /// filter collapses to its min-cutoff and damps the jitter hard.
    hand_orient_filters: [OneEuroFilterState; 4],
    /// Per-bone Schmitt hysteresis state for joint confidence.
    joint_active: HashMap<HumanoidBone, bool>,
    /// Wall-clock timestamp of the previous solve, used to derive the
    /// *render-side* `dt` for the dt-aware rotation blends. The 1€
    /// filters do NOT use this when the sample carries a capture
    /// timestamp — see `filtered_sample_cache`. `None` until the first
    /// call.
    last_solve_instant: Option<Instant>,
    /// The filtered output of `preprocess_source` for the most recent
    /// *distinct* camera sample, keyed by `filtered_sample_key`. The
    /// render loop solves at display rate (60+ fps) while the camera
    /// produces 30 fps — without this cache every camera sample passes
    /// through the 1€ filters once per *render* frame, so `out_speed`
    /// alternates between ~2× the true speed (fresh sample over a half-
    /// length wall dt) and ~0 (repeat sample), which defeats every
    /// speed-calibrated rest gate and makes smoothing depend on the
    /// viewer's monitor refresh rate. On a repeated sample the cached
    /// geometry is reused verbatim (filters do not advance) and only
    /// the confidence channels are refreshed from the live sample so
    /// the hold/fade decay still reaches the gates. Engaged only when
    /// the sample carries `capture_timestamp_ms` (the real camera
    /// path); synthetic producers keep the legacy solve-every-call
    /// behaviour.
    filtered_sample_cache: Option<SourceSkeleton>,
    /// `(source_timestamp, capture_timestamp_ms bits)` of the cached
    /// filtered sample above.
    filtered_sample_key: Option<(u64, u64)>,
    /// Capture timestamp (ms, device clock) of the previous *distinct*
    /// sample — the true inter-capture interval `dt` for the 1€ filters.
    last_capture_timestamp_ms: Option<f64>,
    /// Previous frame's solved local rotation per skeleton node.
    /// `run_frame` rebuilds the base pose every frame, so the value
    /// sitting in `local_transforms` at solve time is REST, not the
    /// previous frame's pose — blending from it makes
    /// `rotation_blend < 1.0` display a permanent `alpha` fraction of
    /// every rotation (measured: a 150-deg arm fold showed a constant
    /// ~45-deg shortfall at the 0.7 default) instead of exponentially
    /// converging. The solver blends from THIS map and writes the
    /// result back, restoring true time-constant smoothing.
    prev_local_rotations: std::collections::HashMap<usize, Quat>,
    /// Previous frame's solved Hips local translation (same rationale
    /// as `prev_local_rotations` for the root-translation lerp).
    prev_hips_translation: Option<[f32; 3]>,
    /// 1€ + deadband state for the raw `root_offset` samples, applied at
    /// the channel's entry (before the reference EMA / deviation math).
    /// The reference EMA is deliberately slow (~10 s) — it defines
    /// "neutral", it does not filter jitter — and at the production
    /// `rotation_blend = 1.0` the hips lerp is a passthrough, so without
    /// this filter the detector's root wobble lands on the avatar's Hips
    /// verbatim and sways the entire body.
    root_offset_filter: OneEuroFilterState,
    /// 1€ filter state for the face pose angles (yaw, pitch, roll),
    /// applied in `preprocess_source` before the head bone is driven.
    /// The face track emits head orientation with heavy per-frame jitter
    /// (measured SRC roll ≈ 4°/frame at rest) and `apply_face_pose`
    /// otherwise passes it straight through — this is the only smoothing
    /// on the head-orientation channel, which the position filters never
    /// touch. Packed `[yaw, pitch, roll]` so roll rides the `z_*` (heavier)
    /// slot of [`OneEuroTuning::FACE`].
    face_angle_filter: OneEuroFilterState,
    /// Slow EMA of the source-skeleton `root_offset`, used as the
    /// "where the subject normally stands" reference. The avatar's
    /// Hips is translated by `(root_offset − reference) * sensitivity`
    /// so the feature self-calibrates to the user's typical pose.
    /// `None` until the first hip-visible frame.
    root_reference: Option<[f32; 3]>,
    /// Frames since `root_reference` was seeded. Drives the full-mirror
    /// lock-in: the neutral converges for the first `ROOT_REFERENCE_LOCK_IN_FRAMES`
    /// (averaging out a noisy startup frame), then freezes.
    root_reference_frames: u32,
    /// `root_reference` was seeded from an explicit pose calibration → trust it
    /// immediately (freeze with no lock-in convergence toward the current pose).
    root_reference_calibrated: bool,
    /// Last stable elbow-swivel plane per arm (`[left, right]`): the
    /// unit perpendicular (⟂ shoulder→wrist axis) the arm-reach IK last
    /// bent the elbow along. Read when the swivel input degenerates —
    /// the pole collapses onto the chain axis (fully folded / fully
    /// stretched view) — so the elbow keeps its previous bend plane
    /// instead of snapping to an arbitrary world-axis fallback; updated
    /// only while the bend radius is well-conditioned.
    arm_swivel_hold: [Option<Vec3>; 2],
    /// EMA state for the camera-driven mouth visemes (aa/ih/ou/ee/oh),
    /// keyed by expression name. The image lip-sync path takes the raw
    /// FaceMesh blendshape, which is noisy frame-to-frame; the eye/brow
    /// path is already smoothed by `expression_blend` inside
    /// [`solve_expressions`], but the mouth viseme policy bypasses that
    /// blend, so it is eased here instead. Empty until the first frame a
    /// viseme is seen; reset with the rest of the motion smoothing.
    mouth_viseme_ema: HashMap<String, f32>,
}

impl PoseSolverState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all per-frame state. With the lift gone there is no
    /// long-running calibration to preserve, so this is equivalent to
    /// [`Self::reset_motion_smoothing`] — the two methods exist for API
    /// compatibility with earlier versions.
    pub fn reset(&mut self) {
        self.reset_motion_smoothing();
    }

    /// Discard the per-frame motion smoothing state (1€ filter,
    /// hysteresis active flags, last-solve timestamp). Useful when
    /// the next solve call represents a new subject pose that should
    /// not blend with whatever was last seen — e.g. between a
    /// calibration-priming frame and the actual test frame in
    /// `diagnose_pose`.
    pub fn reset_motion_smoothing(&mut self) {
        self.joint_filters.clear();
        self.fingertip_filters.clear();
        self.hand_orient_filters = Default::default();
        self.mouth_viseme_ema.clear();
        self.joint_active.clear();
        self.last_solve_instant = None;
        self.filtered_sample_cache = None;
        self.filtered_sample_key = None;
        self.last_capture_timestamp_ms = None;
        self.prev_local_rotations.clear();
        self.prev_hips_translation = None;
        self.root_reference = None;
        self.root_reference_frames = 0;
        self.root_reference_calibrated = false;
        self.root_offset_filter = Default::default();
        self.face_angle_filter = Default::default();
        self.arm_swivel_hold = [None, None];
    }
}

/// 1€ filter state for a single 3D keypoint stream. See
/// [Casiez et al. 2012](https://hal.inria.fr/hal-00670496).
#[derive(Clone, Copy, Debug, Default)]
struct OneEuroFilterState {
    initialized: bool,
    pos: [f32; 3],
    vel: [f32; 3],
    /// Speed of the filter's *output* over the last frame (units/sec) —
    /// the rest-gate signal for the deadband and the arm rotation hold.
    /// See [`REST_SPEED_LO`] for why the output (not `vel`) is gated on.
    out_speed: f32,
}

impl OneEuroFilterState {
    /// Apply the filter to a raw position and return the smoothed
    /// position. `dt` is seconds since the previous update for this
    /// stream. `min_cutoff` is the rest cutoff (lower → smoother when
    /// still) and `beta` the velocity-coupling gain (higher → more
    /// responsive in motion) — passed per call so different channels
    /// (hands vs head) get different tunings from one filter. Self-
    /// initializes on the first call (no smoothing).
    fn apply(&mut self, raw: [f32; 3], dt: f32, t: OneEuroTuning) -> [f32; 3] {
        if !self.initialized || dt <= 0.0 {
            self.initialized = true;
            self.pos = raw;
            self.vel = [0.0, 0.0, 0.0];
            self.out_speed = 0.0;
            return raw;
        }
        // Velocity from raw delta, then low-pass it via d_cutoff.
        let raw_vel = [
            (raw[0] - self.pos[0]) / dt,
            (raw[1] - self.pos[1]) / dt,
            (raw[2] - self.pos[2]) / dt,
        ];
        let alpha_d = one_euro_alpha(dt, ONE_EURO_D_CUTOFF_HZ);
        for (vel, raw_vel) in self.vel.iter_mut().zip(raw_vel.iter()) {
            *vel += alpha_d * (raw_vel - *vel);
        }
        // Position cutoff scales with velocity magnitude — high-speed
        // motion gets a higher cutoff (less smoothing, more responsive).
        // XY and Z are cut independently (see [`OneEuroTuning`]): each
        // plane's cutoff opens only on *its own* speed, so Z's heavy
        // smoothing is not defeated by a fast XY gesture and vice versa.
        let speed_xy = (self.vel[0] * self.vel[0] + self.vel[1] * self.vel[1]).sqrt();
        let speed_z = self.vel[2].abs();
        let alpha_xy = one_euro_alpha(dt, t.min_cutoff + t.beta * speed_xy);
        let alpha_z = one_euro_alpha(dt, t.z_min_cutoff + t.z_beta * speed_z);

        // 1€ candidate movement for this frame (before the deadband).
        let mut delta = [
            alpha_xy * (raw[0] - self.pos[0]),
            alpha_xy * (raw[1] - self.pos[1]),
            alpha_z * (raw[2] - self.pos[2]),
        ];
        let mag = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        // The rest-gate signal: the filter output's own speed. At rest
        // the filter converges and this collapses toward zero (raw
        // jitter is already absorbed by the cutoff above); in motion it
        // tracks the true speed. See [`REST_SPEED_LO`] for why this is
        // used instead of the noisier `vel`.
        self.out_speed = mag / dt;
        let rest = 1.0 - smoothstep(self.out_speed, REST_SPEED_LO, REST_SPEED_HI);

        // Rest deadband, per plane. Faded in by how *at-rest* the joint
        // is: `rest` = 1 below REST_SPEED_LO, 0 above REST_SPEED_HI, so
        // slow intentional motion (sustained output speed) is never
        // frozen — only true rest jitter sees the full radius. The shrink
        // is a soft threshold (`(mag - dead)/mag`) so movement past the
        // radius passes through continuously — no snap when it releases.
        if t.dead_radius > 0.0 && rest > 0.0 {
            let mag_xy = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
            let keep = soft_threshold_keep(mag_xy, t.dead_radius * rest);
            delta[0] *= keep;
            delta[1] *= keep;
        }
        if t.z_dead_radius > 0.0 && rest > 0.0 {
            let keep = soft_threshold_keep(delta[2].abs(), t.z_dead_radius * rest);
            delta[2] *= keep;
        }

        for (pos, d) in self.pos.iter_mut().zip(delta.iter()) {
            *pos += *d;
        }
        self.pos
    }

    /// The filter's last output, without advancing it. Used on repeated
    /// samples (same capture frame observed on a later render frame):
    /// re-`apply`ing the identical raw value would collapse `out_speed`
    /// toward zero and make the rest gates see a render-rate-dependent
    /// speed signal. `None` before the first `apply`.
    fn last_output(&self) -> Option<[f32; 3]> {
        if self.initialized {
            Some(self.pos)
        } else {
            None
        }
    }
}

/// Soft-threshold retention factor for a rest deadband: 0 when the
/// movement `mag` is at/below `dead`, ramping to 1 as it exceeds it via
/// `(mag - dead) / mag`, so crossing the boundary is continuous (no snap
/// on release). Shared by the position deadband (XY / Z planes) and the
/// arm rotation hold so all three gates use identical falloff — change
/// the shape here and every gate stays consistent.
#[inline]
fn soft_threshold_keep(mag: f32, dead: f32) -> f32 {
    if mag <= dead {
        0.0
    } else {
        (mag - dead) / mag
    }
}

#[inline]
fn one_euro_alpha(dt: f32, cutoff_hz: f32) -> f32 {
    let tau = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz.max(1e-6));
    let raw_alpha = 1.0 / (1.0 + tau / dt.max(1e-6));
    raw_alpha.clamp(0.0, 1.0)
}

/// Source points that define a bone's *tip* — what the bone is pointing
/// at. For most bones it's the next humanoid joint down the chain; torso
/// bones use the shoulder midpoint; distal finger bones use the fingertip
/// auxiliary map on [`SourceSkeleton::fingertips`].
#[derive(Clone, Copy, Debug)]
enum Tip {
    Joint(HumanoidBone),
    ShoulderMidpoint,
    /// Tip lives in `SourceSkeleton::fingertips`, keyed by the bone itself.
    Fingertip,
    /// Clavicle (HumanoidBone::LeftShoulder / RightShoulder) — the
    /// bone whose tip in the rig sits at the **outer** end (the
    /// shoulder joint, where the upper arm starts) and whose base
    /// sits at the **inner** end (sternum top). The COCO source
    /// keypoints place the shoulder at the OUTER end (idx 5/6); the
    /// inner end has no dedicated keypoint, so we approximate it as
    /// the L/R shoulder midpoint. Also overrides the source-side base
    /// (which would otherwise default to the bone's own keypoint and
    /// collapse the direction to zero, since that keypoint **is** the
    /// outer end).
    ///
    /// `outer_source` = source keypoint at the clavicle's outer end
    /// (e.g. `LeftShoulder` for the left clavicle). `avatar_outer` =
    /// avatar bone whose rest position marks the clavicle's outer
    /// end in the rig (e.g. `LeftUpperArm`).
    Clavicle {
        outer_source: HumanoidBone,
        avatar_outer: HumanoidBone,
    },
    /// UpperChest / Neck — the upper-spine + cervical chain that
    /// captures chin-thrust (head poking forward). COCO has no
    /// keypoints for individual spine vertebrae or for the neck base,
    /// so source-side base = L/R-shoulder midpoint and source-side
    /// tip = `source.joints[Head]`, which the source builder injects
    /// from the ear midpoint (or nose fallback when an ear is
    /// occluded). Avatar tip = the Head bone's rest world position.
    ///
    /// Both `UpperChest` and `Neck` are driven by the same source
    /// signal; the FK-aware `rest_dir_current` rebase distributes the
    /// rotation across them based on each bone's rest-pose lever
    /// arm, so the upper spine bows slightly while the neck bends
    /// further — without us having to fabricate intermediate
    /// keypoints we don't actually observe.
    HeadFromShoulders,
}

/// Static description of which bones the solver will try to drive and
/// what defines their direction. Ordered parent-first so the forward
/// kinematics pass always sees the parent's updated world rotation before
/// touching the child.
const DRIVEN_BONES: &[(HumanoidBone, Tip)] = &[
    // Spine chain. Ordered parent-first along the actual VRM
    // hierarchy (Spine -> Chest -> UpperChest -> {Neck, clavicles})
    // so the FK pass sees each parent's updated world rotation
    // before touching its children. `Chest` shares Spine's
    // hip->shoulder signal; FK's `rest_dir_current` rebase makes
    // its delta near-identity in practice (Spine has already
    // absorbed the bend), but listing it keeps the bone in the
    // driven set so any small residual still gets applied instead
    // of being silently swallowed by the rest pose.
    (HumanoidBone::Spine, Tip::ShoulderMidpoint),
    (HumanoidBone::Chest, Tip::ShoulderMidpoint),
    (HumanoidBone::UpperChest, Tip::HeadFromShoulders),
    // Clavicles. Listed before the upper arms so the FK pass updates
    // them first — when the clavicle picks up a shrug, the upper
    // arm's *current* world rotation needs the parent's already-
    // applied rotation baked in, which the parent-first ordering
    // gives us for free.
    (
        HumanoidBone::LeftShoulder,
        Tip::Clavicle {
            outer_source: HumanoidBone::LeftShoulder,
            avatar_outer: HumanoidBone::LeftUpperArm,
        },
    ),
    (
        HumanoidBone::RightShoulder,
        Tip::Clavicle {
            outer_source: HumanoidBone::RightShoulder,
            avatar_outer: HumanoidBone::RightUpperArm,
        },
    ),
    // Neck. Sibling of the clavicles under UpperChest. Order vs
    // clavicles is irrelevant (sibling) — placed after them so the
    // arm cluster stays contiguous in this list.
    (HumanoidBone::Neck, Tip::HeadFromShoulders),
    // Upper body
    (
        HumanoidBone::LeftUpperArm,
        Tip::Joint(HumanoidBone::LeftLowerArm),
    ),
    (
        HumanoidBone::LeftLowerArm,
        Tip::Joint(HumanoidBone::LeftHand),
    ),
    (
        HumanoidBone::RightUpperArm,
        Tip::Joint(HumanoidBone::RightLowerArm),
    ),
    (
        HumanoidBone::RightLowerArm,
        Tip::Joint(HumanoidBone::RightHand),
    ),
    // Lower body
    (
        HumanoidBone::LeftUpperLeg,
        Tip::Joint(HumanoidBone::LeftLowerLeg),
    ),
    (
        HumanoidBone::LeftLowerLeg,
        Tip::Joint(HumanoidBone::LeftFoot),
    ),
    (
        HumanoidBone::RightUpperLeg,
        Tip::Joint(HumanoidBone::RightLowerLeg),
    ),
    (
        HumanoidBone::RightLowerLeg,
        Tip::Joint(HumanoidBone::RightFoot),
    ),
    // Feet — driven from a toe-tip auxiliary keypoint stashed in
    // `SourceSkeleton::fingertips` (same auxiliary slot the finger
    // distal bones use). Without this the foot bone is stuck in
    // rest orientation and toes point straight forward regardless
    // of how the lower leg has rotated.
    (HumanoidBone::LeftFoot, Tip::Fingertip),
    (HumanoidBone::RightFoot, Tip::Fingertip),
    // Left fingers — each finger is a 3-bone chain whose tip comes from
    // the next joint down the finger (the Distal bone uses the fingertip
    // auxiliary position).
    (
        HumanoidBone::LeftThumbProximal,
        Tip::Joint(HumanoidBone::LeftThumbIntermediate),
    ),
    (
        HumanoidBone::LeftThumbIntermediate,
        Tip::Joint(HumanoidBone::LeftThumbDistal),
    ),
    (HumanoidBone::LeftThumbDistal, Tip::Fingertip),
    (
        HumanoidBone::LeftIndexProximal,
        Tip::Joint(HumanoidBone::LeftIndexIntermediate),
    ),
    (
        HumanoidBone::LeftIndexIntermediate,
        Tip::Joint(HumanoidBone::LeftIndexDistal),
    ),
    (HumanoidBone::LeftIndexDistal, Tip::Fingertip),
    (
        HumanoidBone::LeftMiddleProximal,
        Tip::Joint(HumanoidBone::LeftMiddleIntermediate),
    ),
    (
        HumanoidBone::LeftMiddleIntermediate,
        Tip::Joint(HumanoidBone::LeftMiddleDistal),
    ),
    (HumanoidBone::LeftMiddleDistal, Tip::Fingertip),
    (
        HumanoidBone::LeftRingProximal,
        Tip::Joint(HumanoidBone::LeftRingIntermediate),
    ),
    (
        HumanoidBone::LeftRingIntermediate,
        Tip::Joint(HumanoidBone::LeftRingDistal),
    ),
    (HumanoidBone::LeftRingDistal, Tip::Fingertip),
    (
        HumanoidBone::LeftLittleProximal,
        Tip::Joint(HumanoidBone::LeftLittleIntermediate),
    ),
    (
        HumanoidBone::LeftLittleIntermediate,
        Tip::Joint(HumanoidBone::LeftLittleDistal),
    ),
    (HumanoidBone::LeftLittleDistal, Tip::Fingertip),
    // Right fingers
    (
        HumanoidBone::RightThumbProximal,
        Tip::Joint(HumanoidBone::RightThumbIntermediate),
    ),
    (
        HumanoidBone::RightThumbIntermediate,
        Tip::Joint(HumanoidBone::RightThumbDistal),
    ),
    (HumanoidBone::RightThumbDistal, Tip::Fingertip),
    (
        HumanoidBone::RightIndexProximal,
        Tip::Joint(HumanoidBone::RightIndexIntermediate),
    ),
    (
        HumanoidBone::RightIndexIntermediate,
        Tip::Joint(HumanoidBone::RightIndexDistal),
    ),
    (HumanoidBone::RightIndexDistal, Tip::Fingertip),
    (
        HumanoidBone::RightMiddleProximal,
        Tip::Joint(HumanoidBone::RightMiddleIntermediate),
    ),
    (
        HumanoidBone::RightMiddleIntermediate,
        Tip::Joint(HumanoidBone::RightMiddleDistal),
    ),
    (HumanoidBone::RightMiddleDistal, Tip::Fingertip),
    (
        HumanoidBone::RightRingProximal,
        Tip::Joint(HumanoidBone::RightRingIntermediate),
    ),
    (
        HumanoidBone::RightRingIntermediate,
        Tip::Joint(HumanoidBone::RightRingDistal),
    ),
    (HumanoidBone::RightRingDistal, Tip::Fingertip),
    (
        HumanoidBone::RightLittleProximal,
        Tip::Joint(HumanoidBone::RightLittleIntermediate),
    ),
    (
        HumanoidBone::RightLittleIntermediate,
        Tip::Joint(HumanoidBone::RightLittleDistal),
    ),
    (HumanoidBone::RightLittleDistal, Tip::Fingertip),
];

/// Solve a single frame.
///
/// `local_transforms` is modified in place. Rotations for bones that the
/// solver could not drive (source joint absent / low confidence) are left
/// untouched — callers are expected to have reset them to rest in the same
/// frame via `AvatarInstance::build_base_pose`.
pub fn solve_avatar_pose(
    source: &SourceSkeleton,
    skeleton: &SkeletonAsset,
    humanoid: Option<&HumanoidMap>,
    local_transforms: &mut [Transform],
    params: &SolverParams,
    state: &mut PoseSolverState,
) {
    let Some(humanoid) = humanoid else {
        return;
    };

    // Render-side dt against the previous solve call (wall clock);
    // clamped so a long pause (debugger, GC stall) does not produce a
    // huge α that snaps the avatar to whatever stale or noisy data
    // arrived first. On the very first call we have no reference, so
    // substitute the 30 fps reference period — the slider value then
    // maps to itself on frame zero, matching the pre-smoothing
    // behaviour. Used ONLY for the display-side dt-aware rotation
    // blends: the measurement-side filters use the capture-clock dt
    // below.
    let now = Instant::now();
    let dt = match state.last_solve_instant {
        Some(prev) => (now - prev).as_secs_f32().clamp(0.0, 0.25),
        None => ROTATION_BLEND_REFERENCE_DT,
    };
    state.last_solve_instant = Some(now);

    // Sample identity + measurement dt. A sample from the real camera
    // path carries `capture_timestamp_ms`; solving the SAME sample again
    // on a later render frame must not advance any measurement filter
    // (see `PoseSolverState::filtered_sample_cache`). The filters' dt is
    // the device-clock interval between distinct captures — the render
    // cadence is irrelevant to how far the subject actually moved.
    let sample_key = source
        .capture_timestamp_ms
        .map(|ts| (source.source_timestamp, ts.to_bits()));
    let is_repeat_sample = sample_key.is_some()
        && sample_key == state.filtered_sample_key
        && state.filtered_sample_cache.is_some();
    let sample_dt = match (source.capture_timestamp_ms, state.last_capture_timestamp_ms) {
        (Some(cur), Some(prev)) if cur > prev => (((cur - prev) / 1000.0) as f32).clamp(1e-3, 0.25),
        // First camera sample, non-monotonic device clock, or a
        // synthetic producer without capture timestamps: fall back to
        // the wall dt (the legacy behaviour, exact for bench drivers
        // that solve once per sample).
        _ => dt,
    };

    // Apply the 1€ filter + Schmitt hysteresis on every keypoint
    // before any geometry runs. Working on a local clone keeps
    // `&SourceSkeleton` immutable for callers (tests, other consumers).
    // On a repeated sample, reuse the cached filtered geometry and only
    // refresh the confidence channels — the hold/fade policy decays
    // confidences on a held sample without re-capturing, and that decay
    // must still reach the solver's gates.
    let mut source_owned = if is_repeat_sample {
        let mut cached = state
            .filtered_sample_cache
            .clone()
            .expect("is_repeat_sample checked cache presence");
        refresh_confidence_channels(&mut cached, source);
        gate_joint_confidences(&mut cached, state, params);
        cached
    } else {
        let filtered = preprocess_source(source, state, sample_dt, params);
        if let Some(key) = sample_key {
            state.filtered_sample_cache = Some(filtered.clone());
            state.filtered_sample_key = Some(key);
            state.last_capture_timestamp_ms = source.capture_timestamp_ms;
        }
        filtered
    };

    // Rest-pose world transforms, computed from the skeleton's rest_local
    // values. These are the ground truth we measure bone directions against;
    // if the avatar has any offset rotations baked into rest pose (common
    // for VRM rigs) they are respected.
    let rest_world = compute_world_transforms(skeleton, |node_idx| {
        skeleton.nodes[node_idx].rest_local.clone()
    });

    // D435-exclusive: every joint carries faithful camera-space 3D, so torso
    // orientation is read directly from the real shoulder / hip lines in the
    // Hips block below. There is no monocular foreshortening yaw to compute
    // and no running-max width to maintain — when no 3D torso line is
    // available the Hips are left at rest (forward) rather than guessing.

    // Two-bone IK: disabled. The reconstruction biases the elbow /
    // knee toward a "natural bend" pole, which corrupts genuine
    // straight-arm poses like calref_003 (T-pose) where the source
    // actually has straight arms. Goal (B) is satisfied better by
    // driving each bone toward the source's raw direction so a
    // straight-arm input remains straight; the cost is that
    // deskcrop / off-frame wrist cases stay foreshortened
    // (separate fix needed at the inference layer).
    let _ = (&mut source_owned, &rest_world);

    let source = &source_owned;

    // Current-frame world transforms, starting from whatever is in
    // local_transforms right now (i.e. rest, unless the caller has already
    // applied animation clips). Updated incrementally as we drive each
    // bone so that a child's new local rotation is computed relative to
    // its parent's updated world rotation.
    let mut current_world = compute_world_transforms(skeleton, |i| local_transforms[i].clone());

    // Hips rotation — derived **directly from the source's L/R UpperLeg
    // direction**. The per-bone benchmark compares the avatar's
    // `RightUpperLeg → LeftUpperLeg` world direction against the source's,
    // and the only way to match that is to align Hips so the two vectors
    // coincide. We use `quat_from_vectors` (shortest-arc 3D rotation) so the
    // alignment is *exact* — including the small Y component of the source
    // hip vector.
    //
    // Torso orientation: hip line preferred, then the real 3D shoulder line
    // when the hips are out of frame (desk-up). Both are exact shortest-arc
    // alignments of the avatar rest line to the source line. A missing 3D
    // line leaves the Hips at rest (forward) rather than guessing — the fix
    // for the desk-up wild-yaw.
    let torso_align = compute_hip_align_rotation(
        &source_owned,
        &rest_world,
        humanoid,
        params.joint_confidence_threshold,
    )
    .or_else(|| {
        // Hips out of frame (desk-up): the shoulder line gives the torso facing,
        // but only its HORIZONTAL (yaw) part goes to the root Hips — applying the
        // full roll here would spin the whole avatar about the pelvis (the
        // "zero-g rotation" a user sees when tilting their shoulders on an
        // upper-body framing). The shoulder ROLL is applied to the upper torso
        // in the driven-bone loop (UpperChest alignment), so the pelvis stays
        // grounded and the spine leans instead.
        compute_shoulder_yaw_rotation(
            &source_owned,
            &rest_world,
            humanoid,
            params.joint_confidence_threshold,
        )
    });
    let hips_target_world_rot = torso_align.and_then(|align_q| {
        // Rotates the rest torso line to the source torso line; compose
        // with the rest hips world rotation.
        humanoid
            .bone_map
            .get(&HumanoidBone::Hips)
            .map(|n| quat_mul(&align_q, &rest_world[n.0 as usize].rotation))
    });

    if let (Some(hips_node), Some(target_world)) = (
        humanoid.bone_map.get(&HumanoidBone::Hips).copied(),
        hips_target_world_rot,
    ) {
        let hips_idx = hips_node.0 as usize;
        if hips_idx < skeleton.nodes.len() {
            let parent_world_rot = skeleton.nodes[hips_idx]
                .parent
                .map(|NodeId(p)| current_world[p as usize].rotation)
                .unwrap_or([0.0, 0.0, 0.0, 1.0]);
            let new_local_rot =
                quat_normalize(&quat_mul(&quat_conjugate(&parent_world_rot), &target_world));
            local_transforms[hips_idx].rotation = blend_local_rotation(
                state,
                hips_idx,
                local_transforms[hips_idx].rotation,
                &new_local_rot,
                dt_aware_blend(params.rotation_blend, dt),
            );
            let updated_world =
                quat_mul(&parent_world_rot, &local_transforms[hips_idx].rotation);
            current_world[hips_idx].rotation = updated_world;
        }
    }

    // Body root translation: shift the avatar's `Hips` so the avatar
    // follows the subject's side-step / lean-in / crouch instead of
    // pivoting in place. We compare the current `root_offset` against
    // a slow EMA reference (the user's "neutral standing position"),
    // multiply the deviation by the per-axis sensitivity, and add it
    // to the rest local position. The EMA self-calibrates over the
    // first few seconds, then drifts slowly enough that intentional
    // motion still reads while accidental drift gets absorbed.
    if params.root_translation_enabled {
        if let Some(raw_offset) = source.root_offset {
            // Filter the raw channel at its entry — see
            // `PoseSolverState::root_offset_filter` for why the EMA and
            // the blend below cannot do this job. Measurement-side
            // state: advances only on a distinct camera sample, with
            // the capture-clock dt; a repeated sample reuses the last
            // filtered output (re-applying the identical raw value
            // would zero `out_speed` and double-run the deadband at
            // render rate).
            let raw_offset = if is_repeat_sample {
                state
                    .root_offset_filter
                    .last_output()
                    .unwrap_or(raw_offset)
            } else {
                state
                    .root_offset_filter
                    .apply(raw_offset, sample_dt, OneEuroTuning::ROOT)
            };
            // Calibration-aware EMA seed. When the user has captured
            // an explicit pose calibration (see docs/calibration-ux.md)
            // *and* the runtime anchor type matches the calibration's
            // mode (hip ↔ FullBody, shoulder ↔ UpperBody), use the
            // calibrated anchor as the reference instead of letting
            // the auto-EMA latch onto whatever the first frame
            // happened to read. Mismatched anchor types fall through
            // to the auto-EMA path — calibrating in T-pose then
            // sitting at a desk yields hip→shoulder anchor switching,
            // and the calibrated hip values would no longer apply.
            if state.root_reference.is_none() {
                if let Some(cal) = params.pose_calibration.as_ref() {
                    let anchor_type_matches = match cal.mode {
                        crate::tracking::CalibrationMode::FullBody => {
                            source.root_anchor_is_hip
                        }
                        crate::tracking::CalibrationMode::UpperBody => {
                            !source.root_anchor_is_hip
                        }
                    };
                    if anchor_type_matches {
                        // `anchor_x` / `anchor_y` are stored in
                        // source-space (matches `raw_offset` directly),
                        // but `anchor_depth_m` is documented as
                        // camera-space metric depth (positive forward
                        // distance). `raw_offset[2]` is source-space
                        // (= negative for forward subjects), so we
                        // negate here to get the EMA seed onto the
                        // same convention as the per-frame samples
                        // it'll blend against.
                        state.root_reference = Some([
                            cal.anchor_x,
                            cal.anchor_y,
                            -cal.anchor_depth_m.unwrap_or(0.0),
                        ]);
                        state.root_reference_calibrated = true;
                    }
                }
            }

            // Neutral-reference update. Full-mirror (default,
            // `root_recenter_horizon_s == None`): seed on the first hip-visible
            // frame, converge for a brief lock-in that averages out a noisy
            // startup frame, then FREEZE — so a real side-step / lean / walk-in
            // persists on the avatar instead of drifting back to centre. Metric
            // depth is absolute, so there is nothing to self-recentre against.
            // A calibrated seed is trusted immediately (no lock-in). `Some(h)`
            // restores the old self-recentring EMA (h-second horizon).
            // Reference update is measurement-side state: it advances
            // once per distinct CAMERA sample. Counting render frames
            // here made the lock-in window (and the `Some(h)` EMA
            // horizon) scale with the viewer's monitor refresh rate.
            const ROOT_REFERENCE_LOCK_IN_FRAMES: u32 = 30; // ~1 s of 30 fps camera samples
            let new_ref = if is_repeat_sample && state.root_reference.is_some() {
                state.root_reference.unwrap()
            } else {
                let alpha = if state.root_reference.is_none() {
                    1.0
                } else if let Some(h) = params.root_recenter_horizon_s.filter(|h| *h > 0.0) {
                    (sample_dt / h).clamp(0.0, 0.2)
                } else if state.root_reference_calibrated
                    || state.root_reference_frames >= ROOT_REFERENCE_LOCK_IN_FRAMES
                {
                    0.0 // frozen → full mirror
                } else {
                    0.1 // lock-in: converge over ~1 s, then freeze
                };
                let prev_ref = state.root_reference.unwrap_or(raw_offset);
                let new_ref = [
                    prev_ref[0] + alpha * (raw_offset[0] - prev_ref[0]),
                    prev_ref[1] + alpha * (raw_offset[1] - prev_ref[1]),
                    prev_ref[2] + alpha * (raw_offset[2] - prev_ref[2]),
                ];
                state.root_reference = Some(new_ref);
                state.root_reference_frames = state.root_reference_frames.saturating_add(1);
                new_ref
            };

            let dev = [
                raw_offset[0] - new_ref[0],
                raw_offset[1] - new_ref[1],
                raw_offset[2] - new_ref[2],
            ];
            // Metric translation (D435): `root_offset` is RAW metres, so `dev`
            // is a real-world displacement of the subject from their neutral
            // position. Map it 1:1 into avatar units — scaled only by the
            // avatar's proportion relative to the subject, so a human-sized rig
            // stays ≈1:1 — with NO room-size sensitivity gain and NO clamp.
            // This is the "camera-space projects straight to the avatar"
            // contract: a 30 cm side-step moves the hips 30 cm.
            //
            // Avatar proportion = rest shoulder span / subject shoulder span.
            // Both are metres-equivalent (the rig's rest world is authored
            // ~1 unit ≈ 1 m), so this is ≈1.0 for a human-sized avatar.
            // Defaults to true 1:1 when either span is unknown.
            let avatar_rest_shoulder_span = {
                let li = humanoid
                    .bone_map
                    .get(&HumanoidBone::LeftUpperArm)
                    .map(|n| n.0 as usize);
                let ri = humanoid
                    .bone_map
                    .get(&HumanoidBone::RightUpperArm)
                    .map(|n| n.0 as usize);
                match (
                    li.and_then(|i| rest_world.get(i)),
                    ri.and_then(|i| rest_world.get(i)),
                ) {
                    (Some(l), Some(r)) => {
                        let d = vec3_length(&vec3_sub(&l.position, &r.position));
                        (d > 0.05).then_some(d)
                    }
                    _ => None,
                }
            };
            let subject_to_avatar_scale = source
                .metric_frame_info
                .as_ref()
                .map(|m| m.reference_span_m)
                .filter(|s| *s > 0.05)
                .and_then(|ref_span| avatar_rest_shoulder_span.map(|av| av / ref_span))
                .unwrap_or(1.0);
            let translation_delta = [
                dev[0] * subject_to_avatar_scale,
                dev[1] * subject_to_avatar_scale,
                dev[2] * subject_to_avatar_scale,
            ];

            if let Some(hips_node) = humanoid.bone_map.get(&HumanoidBone::Hips).copied() {
                let hips_idx = hips_node.0 as usize;
                if hips_idx < skeleton.nodes.len() {
                    let rest_pos = skeleton.nodes[hips_idx].rest_local.translation;
                    let target = [
                        rest_pos[0] + translation_delta[0],
                        rest_pos[1] + translation_delta[1],
                        rest_pos[2] + translation_delta[2],
                    ];
                    // Same dt-aware blend the rotation pass uses, so
                    // translation responsiveness scales with the user's
                    // `rotation_blend` setting (one fewer slider).
                    let blend = dt_aware_blend(params.rotation_blend, dt);
                    // Blend from the previous frame's solved value,
                    // not the freshly-rebuilt base pose — see
                    // `PoseSolverState::prev_hips_translation`.
                    let prev = state
                        .prev_hips_translation
                        .unwrap_or(local_transforms[hips_idx].translation);
                    let blended = [
                        prev[0] + blend * (target[0] - prev[0]),
                        prev[1] + blend * (target[1] - prev[1]),
                        prev[2] + blend * (target[2] - prev[2]),
                    ];
                    state.prev_hips_translation = Some(blended);
                    local_transforms[hips_idx].translation = blended;
                    // Refresh the world transform so any downstream
                    // bone that reads `current_world[hips_idx].position`
                    // sees the translated origin.
                    let parent_world_pos = skeleton.nodes[hips_idx]
                        .parent
                        .map(|NodeId(p)| current_world[p as usize].position)
                        .unwrap_or([0.0, 0.0, 0.0]);
                    current_world[hips_idx].position = [
                        parent_world_pos[0] + local_transforms[hips_idx].translation[0],
                        parent_world_pos[1] + local_transforms[hips_idx].translation[1],
                        parent_world_pos[2] + local_transforms[hips_idx].translation[2],
                    ];
                }
            }
        }
    }

    // Arm-reach IK: when the hand is recognised but the elbow keypoint dropped,
    // synthesise the elbow toward the wrist so the arm reaches the hand instead
    // of collapsing to a bind-pose T. Augments a local copy of the source with
    // the solved `*LowerArm` joints; everything below (contact IK + the body
    // chain) then drives the arm normally. No-op (no clone) when both elbows are
    // adequately observed, so well-tracked frames are untouched.
    let arm_reach_elbows = if params.arm_reach_ik_enabled {
        compute_arm_reach_elbows(
            source,
            &rest_world,
            humanoid,
            params,
            &mut state.arm_swivel_hold,
        )
    } else {
        [None, None]
    };
    let augmented_source;
    let source: &SourceSkeleton = if arm_reach_elbows[0].is_some() || arm_reach_elbows[1].is_some() {
        let mut s = source.clone();
        if let Some(j) = arm_reach_elbows[0] {
            s.joints.insert(HumanoidBone::LeftLowerArm, j);
        }
        if let Some(j) = arm_reach_elbows[1] {
            s.joints.insert(HumanoidBone::RightLowerArm, j);
        }
        augmented_source = s;
        &augmented_source
    } else {
        source
    };

    // Hands-contact arm IK (see `compute_arm_contact_ik`): when the
    // subject's wrists are close, the four arm-bone directions blend
    // toward a two-bone positional solve so the avatar's hands
    // actually meet despite proportion differences.
    let arm_contact_ik = if params.contact_ik_enabled {
        compute_arm_contact_ik(source, humanoid, &rest_world, params)
    } else {
        None
    };

    // Body chain: direction-match each driven bone using 3D source positions.
    // Wrist orientation pass is fired just-in-time at the body→fingers
    // transition: the wrist bone is not in DRIVEN_BONES, so without
    // this hook every finger MCP starts from a wrist whose twist around
    // its length axis is whatever the LowerArm shortest-arc happened
    // to leave it at — and since `quat_from_vectors` picks twist=0
    // around the new direction, the chain typically ends up rotated
    // 90° at the finger base (palm sideways instead of where the
    // subject's palm is actually pointing). The just-in-time pass
    // installs a full 3-DoF wrist rotation derived from the four MCPs
    // (palm plane normal) before the first finger entry runs.
    let mut wrists_oriented = false;
    // Which arm bones the loop actually drove this frame — an undriven arm bone
    // gets the A-pose idle below instead of snapping to the T-pose bind.
    // Index: 0 = LeftUpperArm, 1 = LeftLowerArm, 2 = RightUpperArm, 3 = RightLowerArm.
    let mut arm_driven = [false; 4];
    for &(bone, tip) in DRIVEN_BONES {
        // GUI retargeting toggles. `hand_tracking_enabled` skips both the
        // wrist 3-DoF orientation pass and every finger bone, so the hand
        // chain stays at rest pose. `lower_body_tracking_enabled` skips the
        // upper/lower leg + foot direction-match entries so the legs stay
        // at rest pose regardless of whether the tracker emitted leg
        // keypoints — the user's "I only have my upper body in frame"
        // intent is honoured even when COCO 11-22 happen to have score.
        if is_finger_bone(bone) && !params.hand_tracking_enabled {
            continue;
        }
        if is_lower_body_bone(bone) && !params.lower_body_tracking_enabled {
            continue;
        }
        if !wrists_oriented && is_finger_bone(bone) {
            solve_wrist_orientation(
                HumanoidBone::LeftHand,
                HumanoidBone::LeftMiddleProximal,
                HumanoidBone::LeftIndexProximal,
                HumanoidBone::LeftLittleProximal,
                source.left_hand_orientation,
                skeleton,
                humanoid,
                &rest_world,
                &mut current_world,
                local_transforms,
                params,
                state,
                dt,
            );
            solve_wrist_orientation(
                HumanoidBone::RightHand,
                HumanoidBone::RightMiddleProximal,
                HumanoidBone::RightIndexProximal,
                HumanoidBone::RightLittleProximal,
                source.right_hand_orientation,
                skeleton,
                humanoid,
                &rest_world,
                &mut current_world,
                local_transforms,
                params,
                state,
                dt,
            );
            wrists_oriented = true;
        }
        let Some(source_base) = source.joints.get(&bone) else {
            continue;
        };
        if source_base.confidence < params.joint_confidence_threshold {
            continue;
        }
        let source_tip_pos = match tip {
            Tip::Joint(tip_bone) => {
                let Some(tip_joint) = source.joints.get(&tip_bone) else {
                    continue;
                };
                if tip_joint.confidence < params.joint_confidence_threshold {
                    continue;
                }
                tip_joint.position
            }
            Tip::ShoulderMidpoint => {
                let (Some(l), Some(r)) = (
                    source.joints.get(&HumanoidBone::LeftShoulder),
                    source.joints.get(&HumanoidBone::RightShoulder),
                ) else {
                    continue;
                };
                if l.confidence < params.joint_confidence_threshold
                    || r.confidence < params.joint_confidence_threshold
                {
                    continue;
                }
                midpoint(&l.position, &r.position)
            }
            Tip::Fingertip => {
                let Some(tip_joint) = source.fingertips.get(&bone) else {
                    continue;
                };
                if tip_joint.confidence < params.joint_confidence_threshold {
                    continue;
                }
                tip_joint.position
            }
            Tip::Clavicle { outer_source, .. } => {
                let Some(tip_joint) = source.joints.get(&outer_source) else {
                    continue;
                };
                if tip_joint.confidence < params.joint_confidence_threshold {
                    continue;
                }
                tip_joint.position
            }
            Tip::HeadFromShoulders => {
                let Some(tip_joint) = source.joints.get(&HumanoidBone::Head) else {
                    continue;
                };
                if tip_joint.confidence < params.joint_confidence_threshold {
                    continue;
                }
                tip_joint.position
            }
        };

        // For most bones the bone's own keypoint is the direction's base.
        // Two exceptions both override to the L/R-shoulder midpoint:
        //   * Clavicle — the bone keypoint sits at the OUTER end (=
        //     the same point we just took as the tip), so without an
        //     override the direction collapses to zero. Midpoint
        //     approximates the inner clavicle anchor (sternum top).
        //   * HeadFromShoulders — UpperChest / Neck have no dedicated
        //     COCO keypoint; the natural source-side base for the
        //     "shoulder -> head" chin-thrust signal is the shoulder
        //     midpoint itself, regardless of which bone in the chain
        //     is being driven.
        let source_base_pos = match tip {
            Tip::Clavicle { .. } | Tip::HeadFromShoulders => {
                let (Some(l), Some(r)) = (
                    source.joints.get(&HumanoidBone::LeftShoulder),
                    source.joints.get(&HumanoidBone::RightShoulder),
                ) else {
                    continue;
                };
                if l.confidence < params.joint_confidence_threshold
                    || r.confidence < params.joint_confidence_threshold
                {
                    continue;
                }
                midpoint(&l.position, &r.position)
            }
            _ => source_base.position,
        };

        let Some(node_id) = humanoid.bone_map.get(&bone).copied() else {
            continue;
        };
        let node_idx = node_id.0 as usize;
        if node_idx >= skeleton.nodes.len() {
            continue;
        }

        // Tip in the avatar's rest skeleton. For Tip::Joint we use the tip
        // humanoid bone's world position; for ShoulderMidpoint we average
        // the two shoulders; for Fingertip we walk the bone's own children
        // (VRM rigs expose the fingertip as a non-humanoid child of the
        // Distal bone, so there is no HumanoidBone for it).
        let rest_tip_pos = match tip {
            Tip::Joint(tip_bone) => {
                let Some(tip_node) = humanoid.bone_map.get(&tip_bone).copied() else {
                    continue;
                };
                rest_world[tip_node.0 as usize].position
            }
            Tip::ShoulderMidpoint => {
                let (Some(l), Some(r)) = (
                    humanoid.bone_map.get(&HumanoidBone::LeftShoulder).copied(),
                    humanoid.bone_map.get(&HumanoidBone::RightShoulder).copied(),
                ) else {
                    continue;
                };
                midpoint(
                    &rest_world[l.0 as usize].position,
                    &rest_world[r.0 as usize].position,
                )
            }
            Tip::Clavicle { avatar_outer, .. } => {
                let Some(tip_node) = humanoid.bone_map.get(&avatar_outer).copied() else {
                    continue;
                };
                rest_world[tip_node.0 as usize].position
            }
            Tip::HeadFromShoulders => {
                let Some(tip_node) = humanoid.bone_map.get(&HumanoidBone::Head).copied() else {
                    continue;
                };
                rest_world[tip_node.0 as usize].position
            }
            Tip::Fingertip => {
                // Follow the distal bone's first-child descendant until we
                // hit a leaf — that gives the fingertip position in the
                // rest pose. If the distal has no children, fall back to
                // extrapolating along the parent → distal direction so we
                // still have *some* non-zero rest direction.
                let mut cursor = node_idx;
                loop {
                    let children = &skeleton.nodes[cursor].children;
                    if children.is_empty() {
                        break;
                    }
                    cursor = children[0].0 as usize;
                    if cursor >= skeleton.nodes.len() {
                        break;
                    }
                }
                if cursor == node_idx {
                    // Distal has no children — synthesise a tip by
                    // extending the parent→distal vector one more segment.
                    let parent_idx = match skeleton.nodes[node_idx].parent {
                        Some(NodeId(p)) => p as usize,
                        None => continue,
                    };
                    let p = rest_world[parent_idx].position;
                    let d = rest_world[node_idx].position;
                    [d[0] + (d[0] - p[0]), d[1] + (d[1] - p[1]), d[2] + (d[2] - p[2])]
                } else {
                    rest_world[cursor].position
                }
            }
        };
        // For Tip::HeadFromShoulders the source side anchors at the
        // L/R-shoulder midpoint (see `source_base_pos` override above).
        // Mirror that on the rest side so the direction vectors compare
        // like-for-like — otherwise the solver fits
        // `(rest.Head - rest.Neck) → (src.Head - src.mid)`, leaving a
        // constant ~2.5° residual driven entirely by the rest-pose
        // Neck-to-shoulder offset on standard VRM rigs.
        let rest_base_pos = match tip {
            Tip::HeadFromShoulders => {
                match (
                    humanoid.bone_map.get(&HumanoidBone::LeftUpperArm).copied(),
                    humanoid.bone_map.get(&HumanoidBone::RightUpperArm).copied(),
                ) {
                    (Some(l), Some(r)) => midpoint(
                        &rest_world[l.0 as usize].position,
                        &rest_world[r.0 as usize].position,
                    ),
                    _ => rest_world[node_idx].position,
                }
            }
            _ => rest_world[node_idx].position,
        };
        let rest_dir = vec3_normalize(&vec3_sub(&rest_tip_pos, &rest_base_pos));
        if rest_dir == [0.0, 0.0, 0.0] {
            continue;
        }

        let source_dir = vec3_normalize(&vec3_sub(&source_tip_pos, &source_base_pos));
        if source_dir == [0.0, 0.0, 0.0] {
            continue;
        }

        // Hands-contact override: blend the raw source direction
        // toward the positional-IK direction for the four arm bones.
        let source_dir = match bone {
            HumanoidBone::LeftUpperArm => contact_ik_dir(&arm_contact_ik, 0, 0, source_dir),
            HumanoidBone::LeftLowerArm => contact_ik_dir(&arm_contact_ik, 0, 1, source_dir),
            HumanoidBone::RightUpperArm => contact_ik_dir(&arm_contact_ik, 1, 0, source_dir),
            HumanoidBone::RightLowerArm => contact_ik_dir(&arm_contact_ik, 1, 1, source_dir),
            _ => source_dir,
        };

        // The bone's *current* rest direction must include any rotation
        // the parent has already picked up this frame (notably the Hips
        // body-yaw applied above). Without this rebase, a +y spine
        // matching a +y source-dir under a 180°-yawed Hips would still
        // emit a 180° local twist to "cancel out" the parent yaw —
        // mirroring the spine's local frame and inverting all the
        // child bones' geometry, leaving the back-pose avatar with its
        // arms folded inside the torso.
        let parent_world_rot = skeleton.nodes[node_idx]
            .parent
            .map(|NodeId(p)| current_world[p as usize].rotation)
            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let parent_rest_world_rot = skeleton.nodes[node_idx]
            .parent
            .map(|NodeId(p)| rest_world[p as usize].rotation)
            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let parent_yaw_delta =
            quat_mul(&parent_world_rot, &quat_conjugate(&parent_rest_world_rot));
        let rest_dir_current = quat_rotate_vec3(&parent_yaw_delta, &rest_dir);
        let rest_world_rot = rest_world[node_idx].rotation;
        let current_rest_world_rot = quat_mul(&parent_yaw_delta, &rest_world_rot);

        let delta_world = quat_from_vectors(&rest_dir_current, &source_dir);
        let mut new_world_rot = quat_mul(&delta_world, &current_rest_world_rot);

        // Asymmetric-shoulder alignment. After HeadFromShoulders drives
        // pitch/yaw, the avatar's clavicle-base separation stays along
        // its rest direction (horizontal in body frame). For poses with
        // one shoulder up + other down, this leaves the post-clavicle
        // `LeftUpperArm - RightUpperArm` direction misaligned with
        // source — the bone rotations move the tips outward but the
        // joint bases dominate, so shoulder_line residual stays large
        // (~5-13°) even with everything else driven correctly. Rotating
        // UpperChest so its rest clavicle_sep direction matches source's
        // shoulder_line direction fixes this. Uses quat_from_vectors so
        // it works under arbitrary body yaw (front, back, profile alike)
        // — no facing detection needed.
        if matches!(bone, HumanoidBone::UpperChest) {
            // Prefer the clavicle (Shoulder) bones, but fall back to the UpperArm
            // roots — which every rig has and which ARE the shoulder joints — so
            // this shoulder-line alignment still carries a shoulder TILT onto the
            // upper torso on clavicle-less rigs. This is now the primary tilt
            // path in upper-body mode, where the root Hips takes only the yaw.
            let l_node = humanoid
                .bone_map
                .get(&HumanoidBone::LeftShoulder)
                .or_else(|| humanoid.bone_map.get(&HumanoidBone::LeftUpperArm))
                .copied();
            let r_node = humanoid
                .bone_map
                .get(&HumanoidBone::RightShoulder)
                .or_else(|| humanoid.bone_map.get(&HumanoidBone::RightUpperArm))
                .copied();
            let l_src = source
                .joints
                .get(&HumanoidBone::LeftShoulder)
                .or_else(|| source.joints.get(&HumanoidBone::LeftUpperArm));
            let r_src = source
                .joints
                .get(&HumanoidBone::RightShoulder)
                .or_else(|| source.joints.get(&HumanoidBone::RightUpperArm));
            if let (Some(l_node), Some(r_node), Some(l_src), Some(r_src)) =
                (l_node, r_node, l_src, r_src)
            {
                if l_src.confidence >= params.joint_confidence_threshold
                    && r_src.confidence >= params.joint_confidence_threshold
                {
                    let clav_sep_rest = vec3_sub(
                        &rest_world[l_node.0 as usize].position,
                        &rest_world[r_node.0 as usize].position,
                    );
                    let clav_sep_local = quat_rotate_vec3(
                        &quat_conjugate(&rest_world[node_idx].rotation),
                        &clav_sep_rest,
                    );
                    let clav_sep_new_world =
                        vec3_normalize(&quat_rotate_vec3(&new_world_rot, &clav_sep_local));
                    let src_shoulder_line =
                        vec3_normalize(&vec3_sub(&l_src.position, &r_src.position));
                    if clav_sep_new_world != [0.0; 3] && src_shoulder_line != [0.0; 3] {
                        // Gate on the actual angular discrepancy between
                        // avatar's current clav_sep direction and the
                        // source's shoulder_line direction. Below the
                        // head_lean test's lever-arm threshold (~3°), the
                        // standard HeadFromShoulders rotation IS the
                        // mechanism producing the head shift; cancelling
                        // it would leave head shifts under-driven. Above
                        // it, the over-tilt is large enough that the
                        // alignment correction outweighs the head-shift
                        // cost.
                        let dot = (clav_sep_new_world[0] * src_shoulder_line[0]
                            + clav_sep_new_world[1] * src_shoulder_line[1]
                            + clav_sep_new_world[2] * src_shoulder_line[2])
                            .clamp(-1.0, 1.0);
                        let discrepancy_deg = dot.acos().to_degrees();
                        if discrepancy_deg >= 3.0 {
                            let align_delta =
                                quat_from_vectors(&clav_sep_new_world, &src_shoulder_line);
                            new_world_rot = quat_mul(&align_delta, &new_world_rot);
                        }
                    }
                }
            }
        }

        // Neck HeadFromShoulders pivot correction. The standard math
        // aligns `(Head - Neck.world)` with `source_dir`, but the score
        // (and the semantic intent) measures `(Head - shoulder_mid)`
        // direction. Because the Neck bone pivots at Neck.world while
        // shoulder_mid sits a few cm forward/below, those two directions
        // diverge — leaving a residual of up to ~12° for poses with
        // significant UpperChest tilt. Apply a corrective rotation that
        // aligns the actual `(Head - shoulder_mid)` direction with
        // `source_dir`. One iteration is sufficient to drop the residual
        // to well under 1° in practice.
        if matches!(bone, HumanoidBone::Neck) && matches!(tip, Tip::HeadFromShoulders) {
            if let (Some(head_node), Some(l_ua), Some(r_ua)) = (
                humanoid.bone_map.get(&HumanoidBone::Head).copied(),
                humanoid.bone_map.get(&HumanoidBone::LeftUpperArm).copied(),
                humanoid.bone_map.get(&HumanoidBone::RightUpperArm).copied(),
            ) {
                let head_local =
                    skeleton.nodes[head_node.0 as usize].rest_local.translation;
                let neck_pos = rest_world[node_idx].position;
                let head_world = vec3_add(
                    &neck_pos,
                    &quat_rotate_vec3(&new_world_rot, &head_local),
                );
                let shoulder_mid_rest = midpoint(
                    &rest_world[l_ua.0 as usize].position,
                    &rest_world[r_ua.0 as usize].position,
                );
                // Shift rest shoulder_mid by UpperChest's delta-from-rest
                // to approximate the current world position (clavicle
                // rotations are ignored — small effect compared to the
                // UpperChest tilt).
                let shoulder_mid_offset =
                    vec3_sub(&shoulder_mid_rest, &neck_pos);
                let shoulder_mid_current = vec3_add(
                    &neck_pos,
                    &quat_rotate_vec3(&parent_yaw_delta, &shoulder_mid_offset),
                );
                let actual_dir = vec3_normalize(&vec3_sub(
                    &head_world,
                    &shoulder_mid_current,
                ));
                if actual_dir != [0.0; 3] && source_dir != [0.0; 3] {
                    let corrective =
                        quat_from_vectors(&actual_dir, &source_dir);
                    new_world_rot = quat_mul(&corrective, &new_world_rot);
                }
            }
        }

        let new_local_rot = quat_normalize(&quat_mul(&quat_conjugate(&parent_world_rot), &new_world_rot));


        // End-effector rotation hold: freeze the arm chain's residual
        // rotation jitter while its driving joint is at rest, defeating
        // the lever-arm amplification of upstream angle noise that the
        // source-keypoint filters can't reach. Non-arm bones get
        // `rest = 0`, which is exactly `blend_local_rotation`. See
        // [`ARM_HOLD_ANG_DEAD`] / [`arm_hold_rest`].
        let hold_rest = arm_hold_rest(state, bone);
        local_transforms[node_idx].rotation = blend_arm_rotation(
            state,
            node_idx,
            local_transforms[node_idx].rotation,
            &new_local_rot,
            dt_aware_blend(params.rotation_blend, dt),
            hold_rest,
            arm_step_cap(bone),
        );

        // Keep the world rotation cache in sync so child bones in this
        // same pass see the updated parent orientation.
        let updated_world = quat_mul(&parent_world_rot, &local_transforms[node_idx].rotation);
        current_world[node_idx].rotation = updated_world;

        match bone {
            HumanoidBone::LeftUpperArm => arm_driven[0] = true,
            HumanoidBone::LeftLowerArm => arm_driven[1] = true,
            HumanoidBone::RightUpperArm => arm_driven[2] = true,
            HumanoidBone::RightLowerArm => arm_driven[3] = true,
            _ => {}
        }
    }

    // A-pose idle: any arm bone the loop could not drive (arm out of frame /
    // wrist not tracked) rests in a relaxed A-pose rather than the T-pose bind.
    if params.idle_arm_apose_enabled {
        let idle_alpha = dt_aware_blend(params.rotation_blend, dt);
        apply_idle_arm_pose(
            HumanoidBone::LeftUpperArm,
            HumanoidBone::LeftLowerArm,
            arm_driven[0],
            arm_driven[1],
            skeleton,
            humanoid,
            &rest_world,
            &mut current_world,
            local_transforms,
            state,
            idle_alpha,
        );
        apply_idle_arm_pose(
            HumanoidBone::RightUpperArm,
            HumanoidBone::RightLowerArm,
            arm_driven[2],
            arm_driven[3],
            skeleton,
            humanoid,
            &rest_world,
            &mut current_world,
            local_transforms,
            state,
            idle_alpha,
        );
    }

    // Face pose: drive the Head bone independently. The face track
    // emits head orientation reliably even at oblique views, so the
    // old "gate by shoulder ratio" sentinel is no longer needed — we
    // trust whatever the tracker emits if its confidence clears the
    // threshold. The GUI's `face_tracking_enabled` toggle additionally
    // gates this whole block — when off, the head bone stays parented
    // to whatever rotation the spine chain produced.
    if params.face_tracking_enabled {
        if let Some(face) = source.face {
            if face.confidence >= params.face_confidence_threshold {
                if let Some(head_node) = humanoid.bone_map.get(&HumanoidBone::Head).copied() {
                    let head_idx = head_node.0 as usize;
                    if head_idx < skeleton.nodes.len() {
                        apply_face_pose(
                            face,
                            head_idx,
                            skeleton,
                            local_transforms,
                            &current_world,
                            &rest_world,
                            state,
                            dt_aware_blend(params.rotation_blend, dt),
                        );
                    }
                }
            }
        }
    }

    if thumb_debug_enabled() {
        log_thumb_diagnostics(source, skeleton, humanoid, local_transforms, &rest_world);
    }
}

/// Cached `VULVATAR_DEBUG_THUMB` flag. Reading the env var costs a
/// syscall on every solve; a `OnceLock` lets us amortise that to once
/// per process while still allowing tests to run without the env set.
fn thumb_debug_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("VULVATAR_DEBUG_THUMB").is_some())
}

/// Per-frame thumb diagnostic log. Emits one line per hand at info
/// level so the user can see it without raising RUST_LOG to debug.
/// Goal is to disambiguate "RTMW3D failed to detect thumb keypoints"
/// from "keypoints are detected but the rotation is too small".
///
/// For each side we log:
/// * The 3D source position of CMC / MCP / IP / tip (or `none` if
///   the joint was gated out).
/// * The metacarpal and phalanx segment lengths in source-space units
///   — should be ~0.04-0.10 for an in-frame hand.
/// * The angle between the bone's current local rotation and its rest
///   local rotation, in degrees. Direction-match writes the local
///   rotation; if the angle is small while the user is doing a thumbs-up,
///   the math attenuated the signal. If the angle is reasonable but the
///   visible avatar thumb still doesn't move, the rest-pose orientation
///   of the bone may be wrong (rigging issue).
fn log_thumb_diagnostics(
    source: &SourceSkeleton,
    skeleton: &SkeletonAsset,
    humanoid: &HumanoidMap,
    local_transforms: &[Transform],
    _rest_world: &[WorldXform],
) {
    use HumanoidBone::*;
    fn fmt_pos(p: Option<&[f32; 3]>) -> String {
        match p {
            Some(v) => format!("({:+.3},{:+.3},{:+.3})", v[0], v[1], v[2]),
            None => "none".to_string(),
        }
    }
    fn seg_len(a: Option<&[f32; 3]>, b: Option<&[f32; 3]>) -> String {
        match (a, b) {
            (Some(a), Some(b)) => {
                let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                format!("{:.3}", (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt())
            }
            _ => "—".to_string(),
        }
    }
    fn rot_angle_deg(
        bone: HumanoidBone,
        skeleton: &SkeletonAsset,
        humanoid: &HumanoidMap,
        local_transforms: &[Transform],
    ) -> String {
        let Some(node) = humanoid.bone_map.get(&bone).copied() else {
            return "—".to_string();
        };
        let idx = node.0 as usize;
        if idx >= skeleton.nodes.len() {
            return "—".to_string();
        }
        let cur = local_transforms[idx].rotation;
        let rest = skeleton.nodes[idx].rest_local.rotation;
        // Rotation that takes rest into cur: delta = cur * conjugate(rest).
        let delta = quat_mul(&cur, &quat_conjugate(&rest));
        // Quaternion angle is 2 * acos(|w|); abs() picks the shortest arc.
        let w = delta[3].abs().min(1.0);
        let angle = 2.0 * w.acos() * (180.0 / std::f32::consts::PI);
        format!("{:5.1}°", angle)
    }

    for (label, prox, inter, dist) in [
        ("L", LeftThumbProximal, LeftThumbIntermediate, LeftThumbDistal),
        ("R", RightThumbProximal, RightThumbIntermediate, RightThumbDistal),
    ] {
        let p_cmc = source.joints.get(&prox).map(|j| &j.position);
        let p_mcp = source.joints.get(&inter).map(|j| &j.position);
        let p_ip = source.joints.get(&dist).map(|j| &j.position);
        let p_tip = source.fingertips.get(&dist).map(|j| &j.position);
        log::info!(
            "thumb[{}] src CMC={} MCP={} IP={} tip={} | seg meta={} prox={} dist={} | rot prox={} inter={} dist={}",
            label,
            fmt_pos(p_cmc),
            fmt_pos(p_mcp),
            fmt_pos(p_ip),
            fmt_pos(p_tip),
            seg_len(p_cmc, p_mcp),
            seg_len(p_mcp, p_ip),
            seg_len(p_ip, p_tip),
            rot_angle_deg(prox, skeleton, humanoid, local_transforms),
            rot_angle_deg(inter, skeleton, humanoid, local_transforms),
            rot_angle_deg(dist, skeleton, humanoid, local_transforms),
        );
    }
}

/// Body yaw, derived from the 3D torso-width vector.
///
/// At rest the avatar's left shoulder sits at +X (camera-right under
/// the VRM Y180 flip) and the right shoulder at −X, so the
/// `Left − Right` shoulder vector projected onto the XZ plane points
/// along +X at rest. As the subject rotates around their vertical axis
/// the 3D-native tracker reports this same vector tilted in XZ.
///
/// VulVATAR is selfie-style: subject's left/right are swapped against
/// the avatar's left/right via the mirror mapping in
/// `tracking::rtmw3d::COCO_BODY`, so the body yaw must be applied in
/// the **opposite** sense. Subject rotating
/// CCW (their left side coming toward camera) should make the avatar
/// rotate CW from the user's POV looking at the screen — i.e. the
/// avatar's apparent rotation in selfie video matches what the user
/// would see in a mirror.
///
/// `atan2(bz, bx)` (note: not `-bz`) gives `+θ_subject` mirrored to
/// `−θ_subject` for the avatar. Front and back poses are unchanged
/// (yaw 0 and ±π are sign-invariant); only the intermediate angles
/// flip, which is exactly where the previous formula was producing
/// the wrong-direction body rotation (e.g. 315° three-quarter view
/// rendering the avatar as a 45° puppet instead of a 45° mirror).
///
/// **Pair selection.** Shoulders are the primary signal: they have
/// the longest torso-width lever arm in source space, so for the same
/// quaternion of yaw they produce the largest `bz` magnitude and the
/// best signal-to-noise. The hip pair acts as a fallback only — if
/// the shoulder pair is missing or below the confidence threshold
/// (typical when the user's frame is cropped to the shoulders, or in
/// off-camera-roll poses), we read the hip pair instead.
///
/// We do **not** average the two pairs. The earlier "shoulder + hip
/// confidence-weighted mean" (Option A in
/// `plan/body-twist-yaw-investigation.md`) assumed the two pairs carry
/// independent noisy samples of the same yaw angle, but the 45°
/// measurement row in that plan shows hip Δz disagreeing in **sign**
/// with shoulder Δz at a three-quarter view — the hip pair's reading
/// at that pose was anatomically inconsistent (likely RTMW3D depth
/// noise rather than spine twist). Averaging therefore introduced a
/// sign-flip regression at 45°/315°. Until we have a reliable
/// independent torso-yaw signal (Option B/C from the plan), the
/// shoulder pair stays authoritative and hips only paper over the
/// "shoulders not visible" case.
///
/// Returns `None` when neither pair clears the confidence threshold
/// so the caller can skip the rotation entirely instead of snapping
/// the body to whatever stale value comes through.
/// Shortest-arc rotation that takes the avatar's rest-pose
/// `R_UpperLeg → L_UpperLeg` direction to the source skeleton's
/// equivalent direction. Returns `None` when either source endpoint
/// is missing / below confidence or both hip positions are
/// coincident in 3D (the resulting direction would be pure noise).
///
/// Composing this with the rest world rotation of Hips and pushing
/// it back to local space yields a hips orientation whose
/// `R_UpperLeg → L_UpperLeg` vector matches the source's exactly,
/// which is what the validation benchmark scores against.
fn compute_hip_align_rotation(
    source: &SourceSkeleton,
    rest_world: &[WorldXform],
    humanoid: &HumanoidMap,
    threshold: f32,
) -> Option<Quat> {
    use HumanoidBone::*;
    let l = source.joints.get(&LeftUpperLeg)?;
    let r = source.joints.get(&RightUpperLeg)?;
    if l.confidence < threshold || r.confidence < threshold {
        return None;
    }
    // Source's hip line in source space (L − R).
    let src_dir = vec3_sub(&l.position, &r.position);
    let src_len = vec3_length(&src_dir);
    // Reject literally-coincident hips (would normalise to noise).
    // We pass through every other case so the avatar follows the
    // teacher even when the hip pair span is small.
    if src_len < 1.0e-4 {
        return None;
    }
    let src_n = [src_dir[0] / src_len, src_dir[1] / src_len, src_dir[2] / src_len];

    // Avatar's rest-pose hip line: R_UpperLeg world → L_UpperLeg
    // world, taken straight from `rest_world` so the rotation
    // computed here is *additive* on top of the rest Hips rotation.
    let l_idx = humanoid.bone_map.get(&LeftUpperLeg).map(|n| n.0 as usize)?;
    let r_idx = humanoid.bone_map.get(&RightUpperLeg).map(|n| n.0 as usize)?;
    let l_world = rest_world.get(l_idx)?.position;
    let r_world = rest_world.get(r_idx)?.position;
    let av_dir = vec3_sub(&l_world, &r_world);
    let av_len = vec3_length(&av_dir);
    if av_len < 1.0e-4 {
        return None;
    }
    let av_n = [av_dir[0] / av_len, av_dir[1] / av_len, av_dir[2] / av_len];

    Some(quat_from_vectors(&av_n, &src_n))
}

/// Upper-body counterpart of [`compute_hip_align_rotation`]: align the
/// avatar's rest `R_UpperArm → L_UpperArm` line to the source skeleton's,
/// giving the torso orientation directly from the real 3D shoulder line. The
/// D435 shoulder positions carry true camera-space depth, so this is a plain
/// shortest-arc alignment — no foreshortening inference, no running-max
/// ratchet (the bug that made a corner-clamped shoulder swing the whole
/// body). Returns `None` when either shoulder is missing / below confidence
/// (e.g. gated out of frame) or the two are coincident, so the caller leaves
/// the Hips at rest (forward) rather than guessing. The UpperArm roots ARE the
/// shoulder joints and are present in every humanoid rig, so this is more
/// robust than keying on the optional clavicle bones.
fn compute_shoulder_align_rotation(
    source: &SourceSkeleton,
    rest_world: &[WorldXform],
    humanoid: &HumanoidMap,
    threshold: f32,
) -> Option<Quat> {
    use HumanoidBone::*;
    let l = source.joints.get(&LeftUpperArm)?;
    let r = source.joints.get(&RightUpperArm)?;
    if l.confidence < threshold || r.confidence < threshold {
        return None;
    }
    let src_dir = vec3_sub(&l.position, &r.position);
    let src_len = vec3_length(&src_dir);
    // Plausibility floor. The metric skeleton is normalised so the shoulder
    // span is ≈ `TARGET_SRC_SHOULDER_SPAN` (0.75). A span far below that means
    // the two shoulder keypoints collapsed onto the same point (a degenerate
    // detection whose direction is pure noise — seen on the wave replay at
    // span ≈ 0.03). Reject it so the caller leaves the hips at rest rather
    // than snapping the torso to a garbage yaw.
    const MIN_SHOULDER_SPAN: f32 = 0.30;
    if src_len < MIN_SHOULDER_SPAN {
        return None;
    }
    let src_n = [src_dir[0] / src_len, src_dir[1] / src_len, src_dir[2] / src_len];

    let l_idx = humanoid.bone_map.get(&LeftUpperArm).map(|n| n.0 as usize)?;
    let r_idx = humanoid.bone_map.get(&RightUpperArm).map(|n| n.0 as usize)?;
    let l_world = rest_world.get(l_idx)?.position;
    let r_world = rest_world.get(r_idx)?.position;
    let av_dir = vec3_sub(&l_world, &r_world);
    let av_len = vec3_length(&av_dir);
    if av_len < 1.0e-4 {
        return None;
    }
    let av_n = [av_dir[0] / av_len, av_dir[1] / av_len, av_dir[2] / av_len];

    Some(quat_from_vectors(&av_n, &src_n))
}

/// Upper-body-mode facing for the root Hips: the HORIZONTAL (yaw) part of the
/// shoulder-line alignment only. Applying the full shoulder-line rotation
/// (which includes the roll of a shoulder tilt) to the root spins the WHOLE
/// avatar about the pelvis — the "zero-g rotation" artefact when only the upper
/// body is framed. Projecting both shoulder lines onto the horizontal plane
/// keeps the pelvis upright and grounded; the shoulder ROLL is applied to the
/// upper torso (see the `UpperChest` alignment in the driven-bone loop) so the
/// spine leans instead. Returns `None` if either shoulder is missing / the
/// projected line is degenerate (near-vertical), leaving the Hips at rest.
fn compute_shoulder_yaw_rotation(
    source: &SourceSkeleton,
    rest_world: &[WorldXform],
    humanoid: &HumanoidMap,
    threshold: f32,
) -> Option<Quat> {
    use HumanoidBone::*;
    let l = source.joints.get(&LeftUpperArm)?;
    let r = source.joints.get(&RightUpperArm)?;
    if l.confidence < threshold || r.confidence < threshold {
        return None;
    }
    // Horizontal projection (drop Y) of the source shoulder line.
    let src_h = [l.position[0] - r.position[0], 0.0, l.position[2] - r.position[2]];
    let src_len = vec3_length(&src_h);
    // A near-vertical shoulder line has almost no horizontal component — its yaw
    // is ill-defined; keep the previous facing rather than snapping.
    const MIN_HORIZ_SPAN: f32 = 0.20;
    if src_len < MIN_HORIZ_SPAN {
        return None;
    }
    let src_n = [src_h[0] / src_len, 0.0, src_h[2] / src_len];

    let l_idx = humanoid.bone_map.get(&LeftUpperArm).map(|n| n.0 as usize)?;
    let r_idx = humanoid.bone_map.get(&RightUpperArm).map(|n| n.0 as usize)?;
    let l_world = rest_world.get(l_idx)?.position;
    let r_world = rest_world.get(r_idx)?.position;
    let av_h = [l_world[0] - r_world[0], 0.0, l_world[2] - r_world[2]];
    let av_len = vec3_length(&av_h);
    if av_len < 1.0e-4 {
        return None;
    }
    let av_n = [av_h[0] / av_len, 0.0, av_h[2] / av_len];

    Some(quat_from_vectors(&av_n, &src_n))
}

/// A-pose idle world direction for an arm whose rest bone axis is `rest_dir`,
/// under torso rotation `torso_delta` (rest → current). Down-and-out at
/// `IDLE_ARM_ANGLE_FROM_DOWN_DEG` from straight-down, in the frontal plane, with
/// the horizontal sign taken from `rest_dir` so each arm stays on its own side.
fn idle_arm_direction(rest_dir: Vec3, torso_delta: Quat) -> Vec3 {
    let s = if rest_dir[0] >= 0.0 { 1.0 } else { -1.0 };
    let th = IDLE_ARM_ANGLE_FROM_DOWN_DEG.to_radians();
    let idle_body = [s * th.sin(), -th.cos(), 0.0];
    quat_rotate_vec3(&torso_delta, &idle_body)
}

/// Rest an untracked arm in a relaxed A-pose instead of the T-pose bind. Bones
/// the direction-match loop drove are flagged and skipped here; only a bone the
/// loop left at rest is touched. The upper arm is aimed
/// `IDLE_ARM_ANGLE_FROM_DOWN_DEG` out from straight-down — in the CURRENT torso
/// frame so it follows body yaw/tilt — and the forearm fades straight. Both go
/// through `blend_arm_rotation`, so losing the arm eases into the A-pose and
/// re-acquiring it eases back out (no snap, unlike the bind which appears the
/// instant the loop stops writing the bone). Tracked arms are untouched: their
/// direction-match is bind-relative, so A-pose vs T-pose bind is invisible while
/// tracking holds.
#[allow(clippy::too_many_arguments)]
fn apply_idle_arm_pose(
    ua_bone: HumanoidBone,
    la_bone: HumanoidBone,
    ua_driven: bool,
    la_driven: bool,
    skeleton: &SkeletonAsset,
    humanoid: &HumanoidMap,
    rest_world: &[WorldXform],
    current_world: &mut [WorldXform],
    local_transforms: &mut [Transform],
    state: &mut PoseSolverState,
    alpha: f32,
) {
    if ua_driven && la_driven {
        return;
    }
    let (Some(ua), Some(la)) = (
        humanoid.bone_map.get(&ua_bone).copied(),
        humanoid.bone_map.get(&la_bone).copied(),
    ) else {
        return;
    };
    let (ua_idx, la_idx) = (ua.0 as usize, la.0 as usize);
    if ua_idx >= rest_world.len() || la_idx >= rest_world.len() {
        return;
    }

    // Torso delta (rest → current) so the idle arm follows body yaw / tilt.
    let torso_delta = humanoid
        .bone_map
        .get(&HumanoidBone::Hips)
        .map(|h| h.0 as usize)
        .filter(|&i| i < rest_world.len())
        .map(|i| quat_mul(&current_world[i].rotation, &quat_conjugate(&rest_world[i].rotation)))
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);

    let parent_world = |idx: usize, cw: &[WorldXform]| -> Quat {
        skeleton.nodes[idx]
            .parent
            .map(|NodeId(p)| cw[p as usize].rotation)
            .unwrap_or([0.0, 0.0, 0.0, 1.0])
    };

    if !ua_driven {
        // Rest bone axis (shoulder → elbow); its X sign keeps the arm on its own
        // side so the A-pose is left/right correct on any rig.
        let rest_dir = vec3_normalize(&vec3_sub(
            &rest_world[la_idx].position,
            &rest_world[ua_idx].position,
        ));
        if rest_dir != [0.0; 3] {
            let rest_dir_w = quat_rotate_vec3(&torso_delta, &rest_dir);
            let idle_dir_w = idle_arm_direction(rest_dir, torso_delta);
            let q = quat_from_vectors(&rest_dir_w, &idle_dir_w);
            let target_world =
                quat_mul(&q, &quat_mul(&torso_delta, &rest_world[ua_idx].rotation));
            let pw = parent_world(ua_idx, current_world);
            let new_local = quat_normalize(&quat_mul(&quat_conjugate(&pw), &target_world));
            local_transforms[ua_idx].rotation = blend_arm_rotation(
                state,
                ua_idx,
                local_transforms[ua_idx].rotation,
                &new_local,
                alpha,
                0.0,
                arm_max_step(),
            );
            current_world[ua_idx].rotation = quat_mul(&pw, &local_transforms[ua_idx].rotation);
        }
    }

    if !la_driven {
        // Straighten the forearm: fade toward its rest-local (a continuation of
        // the upper arm) so it hangs straight instead of holding a stale bend.
        let rest_local = skeleton.nodes[la_idx].rest_local.rotation;
        local_transforms[la_idx].rotation = blend_arm_rotation(
            state,
            la_idx,
            local_transforms[la_idx].rotation,
            &rest_local,
            alpha,
            0.0,
            arm_max_step(),
        );
        let pw = parent_world(la_idx, current_world);
        current_world[la_idx].rotation = quat_mul(&pw, &local_transforms[la_idx].rotation);
    }
}

fn apply_face_pose(
    face: FacePose,
    head_idx: usize,
    skeleton: &SkeletonAsset,
    local_transforms: &mut [Transform],
    current_world: &[WorldXform],
    rest_world: &[WorldXform],
    state: &mut PoseSolverState,
    blend: f32,
) {
    // Face pose is an ABSOLUTE head orientation in the source (camera) frame:
    // every channel is measured against the image, not against the torso —
    // yaw from the ear line's XZ angle, pitch from the face's own vertical
    // landmark geometry, roll from the eye line — and `apply_calibration`
    // subtracts the user's neutral from it. So the Head bone's WORLD
    // rotation is the face pose applied to the head's REST world
    // orientation; the parent's accumulated rotation must be divided out,
    // not stacked on.
    //
    // Stacking (the previous `head_world = face * parent_world`) let every
    // upstream rotation double-count into gaze. Live measurement
    // (2026-07-27, 1123 frames): the depth builder's source Head (ear
    // midpoint) sits a stable +0.20 shoulder-spans in front of the shoulder
    // midpoint — part real forward-head desk posture, part depth-surface
    // sampling geometry — which `Tip::HeadFromShoulders` turns into a ~21 deg
    // forward bow of the UpperChest/Neck chain. Stacked, that bow landed on
    // the head as a PERMANENT 21 deg of look-down: with the face track
    // reporting −5 deg (chin slightly up) the avatar's head forward axis
    // measured −11 deg (down), and the regression over the whole capture read
    // `head_elev = −0.79*pitch − 0.156*yaw − 21.5`. Absolute placement leaves
    // the neck bow in the torso (it is real posture) while gaze follows the
    // face track alone, and the constant term goes away.
    //
    // Note this is a no-op whenever the parent chain sits at its rest
    // orientation: `conj(rest_parent) * rest_head_world == rest_local`, so
    // an upright torso reproduces the old rest-relative result exactly.
    let target_world = quat_mul(
        &quat_from_euler_ypr(face.pitch, face.yaw, face.roll),
        &rest_world[head_idx].rotation,
    );
    let parent_world_rot = skeleton.nodes[head_idx]
        .parent
        .map(|NodeId(p)| current_world[p as usize].rotation)
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let target_local = quat_normalize(&quat_mul(&quat_conjugate(&parent_world_rot), &target_world));
    local_transforms[head_idx].rotation = blend_local_rotation(
        state,
        head_idx,
        local_transforms[head_idx].rotation,
        &target_local,
        blend,
    );
}

// ---------------------------------------------------------------------------
// Forward kinematics helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct WorldXform {
    position: Vec3,
    rotation: Quat,
}

/// Compute world-space rotation + translation for every node by a
/// depth-first walk from each scene root. The caller supplies each node's
/// local transform via a closure (so we can reuse this code for both rest
/// and current poses).
fn compute_world_transforms(
    skeleton: &SkeletonAsset,
    local_for: impl Fn(usize) -> Transform,
) -> Vec<WorldXform> {
    let n = skeleton.nodes.len();
    let mut out = vec![
        WorldXform {
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
        };
        n
    ];
    let mut stack: Vec<(usize, Option<usize>)> = Vec::with_capacity(n);
    for root in skeleton.root_nodes.iter().rev() {
        stack.push((root.0 as usize, None));
    }
    while let Some((idx, parent)) = stack.pop() {
        if idx >= n {
            continue;
        }
        let local = local_for(idx);
        let (pos, rot) = match parent {
            Some(p) => {
                let parent_w = &out[p];
                // position = parent_pos + parent_rot * local.translation
                let rotated = quat_rotate_vec3(&parent_w.rotation, &local.translation);
                (
                    [
                        parent_w.position[0] + rotated[0],
                        parent_w.position[1] + rotated[1],
                        parent_w.position[2] + rotated[2],
                    ],
                    quat_normalize(&quat_mul(&parent_w.rotation, &local.rotation)),
                )
            }
            None => (local.translation, quat_normalize(&local.rotation)),
        };
        out[idx] = WorldXform {
            position: pos,
            rotation: rot,
        };
        for child in skeleton.nodes[idx].children.iter().rev() {
            stack.push((child.0 as usize, Some(idx)));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn midpoint(a: &Vec3, b: &Vec3) -> Vec3 {
    [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ]
}

// ---------------------------------------------------------------------------
// Hands-contact arm IK
// ---------------------------------------------------------------------------

/// Per-side IK-corrected arm directions for the hands-contact pass.
/// Produced by [`compute_arm_contact_ik`], consumed via
/// [`contact_ik_dir`] inside the body-chain loop.
struct ArmContactIk {
    /// Proximity blend weight 0..1 (0 = pure direction-match).
    weight: f32,
    /// `[left, right]` → (UpperArm dir, LowerArm dir), source space.
    dirs: [Option<(Vec3, Vec3)>; 2],
}

/// Hands-contact positional correction. Direction-match reproduces the
/// subject's bone DIRECTIONS, but touching hands are a POSITION
/// constraint: with avatar proportions (shoulder span : arm length)
/// different from the subject's, perfectly copied directions leave the
/// avatar's wrists apart when the subject's palms meet. When the two
/// source wrists are close, each arm is re-solved as a two-bone IK
/// toward the wrist positions mapped through ONE affine source→avatar
/// map — a single map preserves coincidence (identical targets stay
/// identical), which per-side scaling would break. The resulting
/// directions replace the raw source directions with a proximity-
/// blended weight, so open poses keep pure direction parity (and the
/// direction-parity validation metrics stay meaningful there).
fn compute_arm_contact_ik(
    source: &SourceSkeleton,
    humanoid: &HumanoidMap,
    rest_world: &[WorldXform],
    params: &SolverParams,
) -> Option<ArmContactIk> {
    let thr = params.joint_confidence_threshold;
    let joint = |b: HumanoidBone| -> Option<Vec3> {
        source
            .joints
            .get(&b)
            .filter(|x| x.confidence >= thr)
            .map(|x| x.position)
    };
    let s_l = joint(HumanoidBone::LeftUpperArm)?;
    let s_r = joint(HumanoidBone::RightUpperArm)?;
    let e_l = joint(HumanoidBone::LeftLowerArm)?;
    let e_r = joint(HumanoidBone::RightLowerArm)?;
    let w_l = joint(HumanoidBone::LeftHand)?;
    let w_r = joint(HumanoidBone::RightHand)?;

    let span_src = vec3_length(&vec3_sub(&s_l, &s_r));
    if span_src < 1e-4 {
        return None;
    }
    // Proximity gate: full weight when the wrists sit within 0.25×span
    // (palms together measures ≲0.2×span on the MCP-centroid wrists),
    // fading to zero by 0.45×span; smoothstep keeps the takeover
    // pop-free as the hands approach.
    let d_w = vec3_length(&vec3_sub(&w_l, &w_r));
    let t = ((d_w / span_src) - 0.25) / (0.45 - 0.25);
    let raw_w = 1.0 - t.clamp(0.0, 1.0);
    if raw_w <= 0.0 {
        return None;
    }
    let weight = raw_w * raw_w * (3.0 - 2.0 * raw_w);

    let rest_pos = |b: HumanoidBone| -> Option<Vec3> {
        humanoid
            .bone_map
            .get(&b)
            .map(|n| n.0 as usize)
            .filter(|&i| i < rest_world.len())
            .map(|i| rest_world[i].position)
    };
    let rs_l = rest_pos(HumanoidBone::LeftUpperArm)?;
    let rs_r = rest_pos(HumanoidBone::RightUpperArm)?;
    let re_l = rest_pos(HumanoidBone::LeftLowerArm)?;
    let re_r = rest_pos(HumanoidBone::RightLowerArm)?;
    let rw_l = rest_pos(HumanoidBone::LeftHand)?;
    let rw_r = rest_pos(HumanoidBone::RightHand)?;

    // Single affine map: anchored at the shoulder midpoints, scaled by
    // the half-armspan ratio (span/2 + arm length — the natural reach
    // normaliser: pure span ratio leaves long-armed subjects mapping
    // forward reaches beyond a short-armed avatar's reach sphere),
    // rotated by the shoulder-girdle alignment so it stays valid under
    // body yaw.
    let arm_src = 0.5
        * ((vec3_length(&vec3_sub(&e_l, &s_l)) + vec3_length(&vec3_sub(&w_l, &e_l)))
            + (vec3_length(&vec3_sub(&e_r, &s_r)) + vec3_length(&vec3_sub(&w_r, &e_r))));
    let arm_av = 0.5
        * ((vec3_length(&vec3_sub(&re_l, &rs_l)) + vec3_length(&vec3_sub(&rw_l, &re_l)))
            + (vec3_length(&vec3_sub(&re_r, &rs_r)) + vec3_length(&vec3_sub(&rw_r, &re_r))));
    let reach_src = 0.5 * span_src + arm_src;
    let reach_av = 0.5 * vec3_length(&vec3_sub(&rs_l, &rs_r)) + arm_av;
    if reach_av < 1e-4 || reach_src < 1e-4 {
        return None;
    }
    let s_per_av = reach_src / reach_av;
    let c_src = midpoint(&s_l, &s_r);
    let c_av = midpoint(&rs_l, &rs_r);
    let rest_line = vec3_normalize(&vec3_sub(&rs_l, &rs_r));
    let src_line = vec3_normalize(&vec3_sub(&s_l, &s_r));
    if rest_line == [0.0; 3] || src_line == [0.0; 3] {
        return None;
    }
    let girdle = quat_from_vectors(&rest_line, &src_line);

    let solve_side = |rs: Vec3, re: Vec3, rw: Vec3, e_src: Vec3, w_src: Vec3| {
        let offset = vec3_scale(&quat_rotate_vec3(&girdle, &vec3_sub(&rs, &c_av)), s_per_av);
        let anchor = vec3_add(&c_src, &offset);
        let l1 = vec3_length(&vec3_sub(&re, &rs)) * s_per_av;
        let l2 = vec3_length(&vec3_sub(&rw, &re)) * s_per_av;
        two_bone_ik(&anchor, &w_src, l1, l2, &e_src)
    };

    Some(ArmContactIk {
        weight,
        dirs: [
            solve_side(rs_l, re_l, rw_l, e_l, w_l),
            solve_side(rs_r, re_r, rw_r, e_r, w_r),
        ],
    })
}

/// Arm-reach IK: solve the whole arm by two-bone IK to the OBSERVED wrist so
/// the avatar's hand actually reaches the recognised hand.
///
/// The body-chain retarget is otherwise pure direction-match, which copies the
/// USER'S joint angles onto the avatar's bones. That has two failure modes this
/// solve fixes:
///   1. A *dropped* elbow keypoint (depth hole / off-frame) leaves BOTH arm
///      bones undriven — `LeftUpperArm` needs the elbow as its tip and
///      `LeftLowerArm` needs it as its base — so they stay at bind (a T-pose)
///      even though the wrist was tracked fine ("hand recognised, arm T-posed").
///   2. Even with the elbow observed, direction-match reaches only
///      `avatar_bone_length` along each copied direction. When avatar and user
///      proportions differ, the hand lands SHORT of the observed wrist: the
///      arms collapse toward the chest and, near the midline, the residual
///      jitters sign so the forearms appear to cross (measured: hands-together
///      spread ~8% of source, occasional sign flip — the "arms cross when the
///      hands meet" artefact).
///
/// So whenever the shoulder + wrist are confident we place the elbow with a
/// two-bone IK toward the wrist (avatar rest bone-lengths scaled into source
/// space; the observed elbow, if any, is only the swivel pole, else a natural
/// behind-the-line bend). The caller inserts the result as the `*LowerArm`
/// source joint so the existing loop drives the whole arm to the hand — now
/// re-proportioned to the avatar, so the hand reaches. Undriven only when the
/// wrist itself is missing/weak (arm at rest). Returns `[left, right]`.
/// Where the elbow bends when nothing observes it.
///
/// The swivel pole only picks a *direction* around the shoulder→wrist axis;
/// its magnitude is irrelevant. The shipping default pushed the pole to
/// `-z` — straight behind the subject — which is why clasping the hands in
/// front swung the avatar's elbows backwards: live capture (2026-07-28) put
/// the avatar's elbow 24 cm BEHIND its shoulder while the hand was in front
/// of it, on frames where that arm's source elbow was unobserved.
///
/// The replacement direction is measured, not assumed. Over 1311 frames
/// where the shoulder, elbow AND wrist were all genuinely observed, the
/// elbow's component perpendicular to the shoulder→wrist axis pointed:
///
/// | side  | lateral | vertical | depth |
/// |-------|---------|----------|-------|
/// | left  | +0.93 (outward) | +0.11 | +0.36 (toward camera) |
/// | right | −0.73 (outward) | −0.67 | +0.15 (toward camera) |
///
/// Outward dominates on both sides and the depth component is positive on
/// both — the opposite of the old bias. Vertical disagrees between sides
/// (it tracks what each hand was doing), so it takes the mean of the two, a
/// mild downward lean that matches how arms actually hang. `sh` is in source
/// coords where the shoulder midpoint is the origin, so the sign of its `x`
/// IS the outward direction for that side.
fn synthetic_elbow_pole(sh: &Vec3, wr: &Vec3, src_span: f32) -> Vec3 {
    /// Perpendicular offset shape: outward-dominant, slightly down, slightly
    /// forward — normalised, then scaled by the same 0.4 spans the previous
    /// default used (the magnitude never mattered, only the direction).
    const OUTWARD: f32 = 0.85;
    const DOWN: f32 = -0.35;
    const FORWARD: f32 = 0.25;
    let mid = midpoint(sh, wr);
    let outward = if sh[0] >= 0.0 { 1.0 } else { -1.0 };
    let k = 0.4 * src_span;
    [
        mid[0] + outward * OUTWARD * k,
        mid[1] + DOWN * k,
        mid[2] + FORWARD * k,
    ]
}

fn compute_arm_reach_elbows(
    source: &SourceSkeleton,
    rest_world: &[WorldXform],
    humanoid: &HumanoidMap,
    params: &SolverParams,
    swivel_hold: &mut [Option<Vec3>; 2],
) -> [Option<crate::tracking::SourceJoint>; 2] {
    let thr = params.joint_confidence_threshold;
    let rest_pos = |b: HumanoidBone| -> Option<Vec3> {
        humanoid
            .bone_map
            .get(&b)
            .map(|n| n.0 as usize)
            .filter(|&i| i < rest_world.len())
            .map(|i| rest_world[i].position)
    };
    let av_span = match (
        rest_pos(HumanoidBone::LeftUpperArm),
        rest_pos(HumanoidBone::RightUpperArm),
    ) {
        (Some(a), Some(b)) => vec3_length(&vec3_sub(&a, &b)),
        _ => return [None, None],
    };
    // Source-side scale reference. The per-frame raw shoulder span
    // foreshortens with torso yaw (and collapses on shoulder depth
    // holes), which shrinks l1/l2 and re-introduces exactly the
    // under-reach this pass exists to fix — but only in twisted poses,
    // where it's hardest to notice in review. The metric pipeline
    // already maintains a stabilised subject span (TorsoScaleStabilizer
    // → `MetricFrameInfo.reference_span_m`, metres) and the
    // metres→source conversion (`mpsu`), so prefer that and fall back
    // to the raw pair only for non-metric providers.
    let raw_span = match (
        source.joints.get(&HumanoidBone::LeftUpperArm),
        source.joints.get(&HumanoidBone::RightUpperArm),
    ) {
        (Some(a), Some(b)) => Some(vec3_length(&vec3_sub(&a.position, &b.position))),
        _ => None,
    };
    let stable_span = source
        .metric_frame_info
        .as_ref()
        .filter(|m| m.mpsu > 1e-6)
        .map(|m| m.reference_span_m / m.mpsu)
        .filter(|s| *s > 1e-4);
    let Some(src_span) = stable_span.or(raw_span).filter(|s| *s > 1e-4) else {
        return [None, None];
    };
    if av_span < 1e-4 {
        return [None, None];
    }
    let scale = src_span / av_span;

    let sides = [
        (
            HumanoidBone::LeftUpperArm,
            HumanoidBone::LeftLowerArm,
            HumanoidBone::LeftHand,
            HumanoidBone::LeftMiddleProximal,
        ),
        (
            HumanoidBone::RightUpperArm,
            HumanoidBone::RightLowerArm,
            HumanoidBone::RightHand,
            HumanoidBone::RightMiddleProximal,
        ),
    ];
    let mut out = [None, None];
    for (side, &(sh_b, el_b, wr_b, palm_b)) in sides.iter().enumerate() {
        // Require a confident shoulder + wrist. Given both, solve the whole arm
        // by IK to the observed wrist EVEN WHEN the elbow is also observed:
        // direction-match copies the user's angles at avatar bone lengths and
        // lands the hand short of the wrist when proportions differ (collapse /
        // midline cross). IK re-proportions the chain so the hand reaches; the
        // observed elbow, if any, becomes the swivel pole below.
        let Some(sh) = source.joints.get(&sh_b).filter(|x| x.confidence >= thr) else {
            continue;
        };
        let Some(wr) = source.joints.get(&wr_b).filter(|x| x.confidence >= thr) else {
            continue;
        };
        let (Some(rsh), Some(rel), Some(rwr)) = (rest_pos(sh_b), rest_pos(el_b), rest_pos(wr_b))
        else {
            continue;
        };
        let l1 = vec3_length(&vec3_sub(&rel, &rsh)) * scale;
        // The IK target is the source wrist joint, which the trackers
        // place at the four-MCP centroid (≈ half a palm PAST the
        // anatomical wrist — see `attach_hand`). With a bare forearm
        // length the chain is asked to cover forearm + palm with l2 =
        // forearm, so reachable targets over-flex the elbow and
        // extended ones read as unreachable. Extend l2 by the avatar's
        // own rest wrist→middle-proximal distance (the same palm
        // segment, exactly), with an anthropometric fraction as the
        // fallback for finger-less rigs.
        const PALM_FRACTION_OF_FOREARM: f32 = 0.35;
        let fore_rest = vec3_length(&vec3_sub(&rwr, &rel));
        let palm_rest = rest_pos(palm_b)
            .map(|p| vec3_length(&vec3_sub(&p, &rwr)))
            .filter(|l| *l > 1e-4)
            .unwrap_or(PALM_FRACTION_OF_FOREARM * fore_rest);
        let l2 = (fore_rest + palm_rest) * scale;
        if l1 < 1e-4 || l2 < 1e-4 {
            continue;
        }
        // Swivel pole: the observed elbow ONLY when it is confidently tracked
        // (it steers the bend direction without dictating the reach). A
        // low-confidence elbow is trusted for NOTHING geometric: when the elbow
        // leaves the frame the keypoint clamps to the image edge with a fabricated
        // depth, and steering the swivel toward that point wrenches the forearm
        // sideways — the "arm fractures when the elbow is out of frame" artefact
        // (手首は見えているのに肘が視界外だと腕が骨折する). Below threshold we fall
        // back to a natural bend biased behind the shoulder→wrist line (toward the
        // body, −Z is away from the camera) so the elbow does not hyper-extend
        // straight — the same default used when no elbow keypoint exists at all.
        // Continuous confidence blend instead of a hard threshold: a
        // step gate teleports the pole (and with it the forearm swivel)
        // the frame the elbow confidence crosses `thr`. The observed
        // elbow fades in over [thr, thr + POLE_CONF_BLEND_BAND].
        const POLE_CONF_BLEND_BAND: f32 = 0.15;
        let synth_pole = synthetic_elbow_pole(&sh.position, &wr.position, src_span);
        let pole = match source.joints.get(&el_b) {
            Some(el) => {
                let t = ((el.confidence - thr) / POLE_CONF_BLEND_BAND).clamp(0.0, 1.0);
                [
                    synth_pole[0] + (el.position[0] - synth_pole[0]) * t,
                    synth_pole[1] + (el.position[1] - synth_pole[1]) * t,
                    synth_pole[2] + (el.position[2] - synth_pole[2]) * t,
                ]
            }
            None => synth_pole,
        };
        // Swivel continuity: when the pole degenerates onto the
        // shoulder→wrist axis, prefer the held bend plane from the last
        // well-conditioned frame over two_bone_ik's arbitrary
        // world-axis fallback (which flips the elbow at random under
        // micro-noise near full fold / full extension).
        let to_t = vec3_sub(&wr.position, &sh.position);
        let d = vec3_length(&to_t);
        let pole = if d > 1e-5 {
            let axis = vec3_scale(&to_t, 1.0 / d);
            let p = vec3_sub(&pole, &sh.position);
            let p_perp = vec3_sub(&p, &vec3_scale(&axis, vec3_dot(&p, &axis)));
            if vec3_length(&p_perp) < 0.05 * d {
                match swivel_hold[side] {
                    Some(held) => vec3_add(&sh.position, &vec3_scale(&held, d)),
                    None => pole,
                }
            } else {
                pole
            }
        } else {
            pole
        };
        let Some((upper_dir, _lower_dir)) =
            two_bone_ik(&sh.position, &wr.position, l1, l2, &pole)
        else {
            continue;
        };
        let elbow_pos = vec3_add(&sh.position, &vec3_scale(&upper_dir, l1));
        // Refresh the held bend plane only while it is well-conditioned
        // (meaningful bend radius) — near full extension the achieved
        // perpendicular is numerical noise and must not overwrite a
        // good hold.
        if d > 1e-5 {
            let axis = vec3_scale(&to_t, 1.0 / d);
            let e = vec3_sub(&elbow_pos, &sh.position);
            let e_perp = vec3_sub(&e, &vec3_scale(&axis, vec3_dot(&e, &axis)));
            let r = vec3_length(&e_perp);
            if r > 0.05 * l1 {
                swivel_hold[side] = Some(vec3_scale(&e_perp, 1.0 / r));
            }
        }
        out[side] = Some(crate::tracking::SourceJoint {
            position: elbow_pos,
            confidence: wr.confidence.min(sh.confidence),
            metric_depth_m: None,
        });
    }
    out
}

/// Analytic two-bone IK: place the elbow so the chain
/// (anchor → elbow → target) keeps the given segment lengths, choosing
/// the elbow swivel closest to `pole` (the tracked elbow position).
/// Returns the (upper, lower) unit directions; the lower direction
/// points at the TRUE target even when the target is out of reach and
/// the elbow placement had to clamp, so the hand still aims correctly.
fn two_bone_ik(
    anchor: &Vec3,
    target: &Vec3,
    l1: f32,
    l2: f32,
    pole: &Vec3,
) -> Option<(Vec3, Vec3)> {
    let to_t = vec3_sub(target, anchor);
    let d = vec3_length(&to_t);
    if d < 1e-5 || l1 < 1e-5 || l2 < 1e-5 {
        return None;
    }
    let axis = vec3_scale(&to_t, 1.0 / d);
    let d_cl = d.clamp((l1 - l2).abs() + 1e-4, (l1 + l2 - 1e-4).max((l1 - l2).abs() + 2e-4));
    let a = (l1 * l1 - l2 * l2 + d_cl * d_cl) / (2.0 * d_cl);
    let r = (l1 * l1 - a * a).max(0.0).sqrt();

    let p = vec3_sub(pole, anchor);
    let p_perp = vec3_sub(&p, &vec3_scale(&axis, vec3_dot(&p, &axis)));
    let p_len = vec3_length(&p_perp);
    let perp = if p_len > 1e-5 {
        vec3_scale(&p_perp, 1.0 / p_len)
    } else {
        // Pole sits on the shoulder→target axis (fully folded or fully
        // stretched view) — any perpendicular keeps the chain valid.
        let alt = if axis[1].abs() < 0.9 {
            [0.0, 1.0, 0.0]
        } else {
            [1.0, 0.0, 0.0]
        };
        let c = vec3_normalize(&vec3_cross(&axis, &alt));
        if c == [0.0; 3] {
            return None;
        }
        c
    };
    let elbow = vec3_add(
        &vec3_add(anchor, &vec3_scale(&axis, a)),
        &vec3_scale(&perp, r),
    );
    let upper = vec3_normalize(&vec3_sub(&elbow, anchor));
    let lower = vec3_normalize(&vec3_sub(target, &elbow));
    if upper == [0.0; 3] || lower == [0.0; 3] {
        return None;
    }
    Some((upper, lower))
}

#[cfg(test)]
mod arm_contact_ik_tests {
    use super::*;

    fn assert_close(a: Vec3, b: Vec3, eps: f32) {
        let d = vec3_length(&vec3_sub(&a, &b));
        assert!(d < eps, "expected {a:?} ≈ {b:?} (|Δ|={d})");
    }

    /// Reachable target with the pole exactly on the solution circle:
    /// the elbow lands on the pole and both directions are exact.
    #[test]
    fn two_bone_ik_reachable_matches_pole() {
        let (upper, lower) = two_bone_ik(
            &[0.0, 0.0, 0.0],
            &[0.3, 0.0, 0.0],
            0.25,
            0.25,
            &[0.15, -0.2, 0.0],
        )
        .expect("solvable");
        assert_close(upper, [0.6, -0.8, 0.0], 1e-3);
        assert_close(lower, [0.6, 0.8, 0.0], 1e-3);
    }

    /// Out-of-reach target: the chain straightens and the lower
    /// direction still points at the TRUE target.
    #[test]
    fn two_bone_ik_clamps_out_of_reach_toward_target() {
        let (upper, lower) = two_bone_ik(
            &[0.0, 0.0, 0.0],
            &[0.6, 0.0, 0.0],
            0.25,
            0.25,
            &[0.2, -0.1, 0.0],
        )
        .expect("solvable");
        assert!(
            vec3_dot(&upper, &[1.0, 0.0, 0.0]) > 0.99,
            "upper should straighten toward the target, got {upper:?}"
        );
        assert!(
            vec3_dot(&lower, &[1.0, 0.0, 0.0]) > 0.99,
            "lower should aim at the true target, got {lower:?}"
        );
    }

    use crate::asset::HumanoidMap;
    use std::collections::HashMap;

    /// Build a HumanoidMap + rest_world for a symmetric T-pose-ish
    /// avatar, arm bones at node indices 0..=5.
    fn arm_rig() -> (HumanoidMap, Vec<WorldXform>) {
        use crate::asset::NodeId;
        let mut bone_map = HashMap::new();
        bone_map.insert(HumanoidBone::LeftUpperArm, NodeId(0));
        bone_map.insert(HumanoidBone::LeftLowerArm, NodeId(1));
        bone_map.insert(HumanoidBone::LeftHand, NodeId(2));
        bone_map.insert(HumanoidBone::RightUpperArm, NodeId(3));
        bone_map.insert(HumanoidBone::RightLowerArm, NodeId(4));
        bone_map.insert(HumanoidBone::RightHand, NodeId(5));
        let xf = |p: Vec3| WorldXform { position: p, rotation: [0.0, 0.0, 0.0, 1.0] };
        // T-pose: arms out along ±x at shoulder height.
        let rest = vec![
            xf([0.15, 1.4, 0.0]),  // 0 L upper
            xf([0.40, 1.4, 0.0]),  // 1 L lower (elbow)
            xf([0.65, 1.4, 0.0]),  // 2 L hand
            xf([-0.15, 1.4, 0.0]), // 3 R upper
            xf([-0.40, 1.4, 0.0]), // 4 R lower
            xf([-0.65, 1.4, 0.0]), // 5 R hand
        ];
        (HumanoidMap { bone_map }, rest)
    }

    fn src_joint(p: Vec3) -> crate::tracking::SourceJoint {
        crate::tracking::SourceJoint { position: p, confidence: 1.0, metric_depth_m: None }
    }

    /// REPRO: hands brought together at the midline in front (the
    /// "fingertips touching" pose the user reports breaking). The
    /// contact IK must NOT cross the arms — the left elbow/wrist stay
    /// on the +x (left) side and the right on −x.
    #[test]
    fn hands_together_does_not_cross_arms() {
        let (humanoid, rest) = arm_rig();
        let mut source = SourceSkeleton::empty(0);
        let put = |sk: &mut SourceSkeleton, b, p| {
            sk.joints.insert(b, src_joint(p));
        };
        // Shoulders apart, elbows out + forward, wrists meeting near
        // the midline in front of the chest.
        put(&mut source, HumanoidBone::LeftUpperArm, [0.20, 0.40, 0.0]);
        put(&mut source, HumanoidBone::RightUpperArm, [-0.20, 0.40, 0.0]);
        put(&mut source, HumanoidBone::LeftLowerArm, [0.28, 0.15, 0.25]);
        put(&mut source, HumanoidBone::RightLowerArm, [-0.28, 0.15, 0.25]);
        put(&mut source, HumanoidBone::LeftHand, [0.03, 0.0, 0.5]);
        put(&mut source, HumanoidBone::RightHand, [-0.03, 0.0, 0.5]);

        let params = SolverParams { joint_confidence_threshold: 0.0, ..Default::default() };
        let ik = compute_arm_contact_ik(&source, &humanoid, &rest, &params)
            .expect("contact IK engages when hands meet");

        let (ul, ll) = ik.dirs[0].expect("L dirs");
        let (ur, lr) = ik.dirs[1].expect("R dirs");
        eprintln!("weight={:.3}", ik.weight);
        eprintln!("L upper={ul:?} lower={ll:?}");
        eprintln!("R upper={ur:?} lower={lr:?}");
        // Crossing signature: left upper arm pointing toward −x (right).
        assert!(ul[0] > -0.2, "L upper crosses to the right: {ul:?}");
        assert!(ur[0] < 0.2, "R upper crosses to the left: {ur:?}");
        // Lower (forearm) should aim inward toward the midline but not
        // past it dramatically: left forearm points −x (inward), right
        // +x — that's correct convergence, not crossing.
        // Convergence, not crossing: left forearm aims −x (inward to
        // the midline), right +x; neither overshoots to the far side.
        assert!(ll[0] < 0.0 && lr[0] > 0.0, "forearms should converge inward");
    }

    /// Coincident targets from two different anchors resolve to the
    /// SAME wrist position: anchor + l1·upper + l2·lower must agree —
    /// the property that makes the avatar's palms actually meet.
    #[test]
    fn two_bone_ik_coincident_targets_meet() {
        let target = [0.05, -0.1, 0.3];
        let (u_l, l_l) =
            two_bone_ik(&[0.2, 0.0, 0.0], &target, 0.25, 0.25, &[0.25, -0.2, 0.1]).unwrap();
        let (u_r, l_r) =
            two_bone_ik(&[-0.2, 0.0, 0.0], &target, 0.25, 0.25, &[-0.25, -0.2, 0.1]).unwrap();
        let wrist = |anchor: Vec3, u: Vec3, l: Vec3| {
            vec3_add(
                &vec3_add(&anchor, &vec3_scale(&u, 0.25)),
                &vec3_scale(&l, 0.25),
            )
        };
        let w_l = wrist([0.2, 0.0, 0.0], u_l, l_l);
        let w_r = wrist([-0.2, 0.0, 0.0], u_r, l_r);
        assert_close(w_l, target, 1e-3);
        assert_close(w_r, target, 1e-3);
    }

    /// REPRO: the elbow leaves the camera frame while the hand stays
    /// tracked ("手を認識してるのに肘が視界外だと腕が骨折する"). Off-frame the
    /// elbow keypoint clamps to the image edge with a fabricated position and
    /// only a weak score; `compute_arm_reach_elbows` must NOT steer the swivel
    /// toward that garbage — a low-confidence elbow is ignored as the pole and
    /// the arm falls back to a natural behind-the-line bend (no fracture).
    #[test]
    fn offframe_elbow_ignored_as_swivel_pole() {
        let (humanoid, rest) = arm_rig();
        let mut source = SourceSkeleton::empty(0);
        // Confident shoulders (span 0.30 == avatar span → scale 1) and a
        // confident left wrist reaching forward-down at the midline.
        source.joints.insert(HumanoidBone::LeftUpperArm, src_joint([0.15, 1.40, 0.0]));
        source.joints.insert(HumanoidBone::RightUpperArm, src_joint([-0.15, 1.40, 0.0]));
        source.joints.insert(HumanoidBone::LeftHand, src_joint([0.15, 1.25, 0.15]));
        // Off-frame elbow: clamped far out to +x (image edge) with a weak score.
        source.joints.insert(
            HumanoidBone::LeftLowerArm,
            crate::tracking::SourceJoint {
                position: [0.90, 1.40, 0.0],
                confidence: 0.2,
                metric_depth_m: None,
            },
        );

        let params = SolverParams { joint_confidence_threshold: 0.5, ..Default::default() };
        let mut hold = [None, None];
        let out = compute_arm_reach_elbows(&source, &rest, &humanoid, &params, &mut hold);
        let elbow = out[0].expect("left arm reaches the tracked wrist").position;

        // The contract is "a sub-threshold elbow contributes NOTHING", so the
        // reference is the same solve with no elbow keypoint at all — not a
        // hardcoded position, which would silently re-pin the synthetic
        // default's own geometry (it changed once already: the old
        // straight-back bias is what swung the avatar's elbows behind it when
        // the hands came together in front).
        let mut without = source.clone();
        without.joints.remove(&HumanoidBone::LeftLowerArm);
        let mut hold = [None, None];
        let baseline = compute_arm_reach_elbows(&without, &rest, &humanoid, &params, &mut hold)[0]
            .expect("left arm reaches the tracked wrist")
            .position;
        let drift = vec3_length(&vec3_sub(&elbow, &baseline));
        assert!(
            drift < 1e-4,
            "a sub-threshold elbow must not steer the swivel: {elbow:?} vs {baseline:?}"
        );
        // And it must be nowhere near the garbage keypoint at x = 0.90.
        assert!(
            elbow[0] < 0.5,
            "elbow wrenched sideways toward the off-frame garbage pole: {elbow:?}"
        );
    }

    /// Torso yaw foreshortens the raw shoulder span; the stabilised
    /// metric span (`MetricFrameInfo.reference_span_m`) must win so the
    /// arm keeps its true length instead of shrinking with the twist
    /// (the under-reach → hands-collapse-to-midline regression class).
    #[test]
    fn reach_scale_prefers_stabilized_metric_span() {
        let (humanoid, rest) = arm_rig();
        let mut source = SourceSkeleton::empty(0);
        // Yawed torso: raw span reads 0.15, half the avatar's 0.30.
        source.joints.insert(HumanoidBone::LeftUpperArm, src_joint([0.075, 1.40, 0.0]));
        source.joints.insert(HumanoidBone::RightUpperArm, src_joint([-0.075, 1.40, 0.0]));
        source.joints.insert(HumanoidBone::LeftHand, src_joint([0.075, 1.00, 0.10]));
        let params = SolverParams { joint_confidence_threshold: 0.0, ..Default::default() };
        let sh = [0.075, 1.40, 0.0];

        // No metric info → raw-span fallback: upper arm halves (0.125).
        let mut hold = [None, None];
        let raw = compute_arm_reach_elbows(&source, &rest, &humanoid, &params, &mut hold);
        let raw_l1 = vec3_length(&vec3_sub(&raw[0].expect("raw path solves").position, &sh));
        assert!((raw_l1 - 0.125).abs() < 1e-3, "raw fallback l1 = {raw_l1}, want 0.125");

        // Stabilised span says the subject really spans 0.30 → scale 1,
        // full-length upper arm despite the foreshortened raw pair.
        source.stamp_synthetic_metric_frame();
        source.metric_frame_info.as_mut().unwrap().reference_span_m = 0.30;
        let mut hold = [None, None];
        let stab = compute_arm_reach_elbows(&source, &rest, &humanoid, &params, &mut hold);
        let stab_l1 = vec3_length(&vec3_sub(&stab[0].expect("metric path solves").position, &sh));
        assert!((stab_l1 - 0.25).abs() < 1e-3, "stabilised l1 = {stab_l1}, want 0.25");
    }

    /// The IK target is the MCP-centroid wrist (≈ a palm past the
    /// anatomical wrist), so l2 must cover forearm + palm. With finger
    /// rest bones present the palm segment comes from the avatar's own
    /// wrist→middle-proximal distance.
    #[test]
    fn l2_covers_forearm_plus_palm() {
        use crate::asset::NodeId;
        let (mut humanoid, mut rest) = arm_rig();
        humanoid
            .bone_map
            .insert(HumanoidBone::LeftMiddleProximal, NodeId(6));
        rest.push(WorldXform { position: [0.75, 1.4, 0.0], rotation: [0.0, 0.0, 0.0, 1.0] });

        let mut source = SourceSkeleton::empty(0);
        source.joints.insert(HumanoidBone::LeftUpperArm, src_joint([0.15, 1.40, 0.0]));
        source.joints.insert(HumanoidBone::RightUpperArm, src_joint([-0.15, 1.40, 0.0]));
        // Bent-reachable target for l1 = 0.25, l2 = 0.25 + 0.10.
        source.joints.insert(HumanoidBone::LeftHand, src_joint([0.15, 1.05, 0.15]));
        let params = SolverParams { joint_confidence_threshold: 0.0, ..Default::default() };
        let mut hold = [None, None];
        let out = compute_arm_reach_elbows(&source, &rest, &humanoid, &params, &mut hold);
        let elbow = out[0].expect("solves").position;
        let l1 = vec3_length(&vec3_sub(&elbow, &[0.15, 1.40, 0.0]));
        let l2 = vec3_length(&vec3_sub(&[0.15, 1.05, 0.15], &elbow));
        assert!((l1 - 0.25).abs() < 1e-3, "l1 = {l1}");
        assert!(
            (l2 - 0.35).abs() < 1e-3,
            "l2 must include the rest palm segment (0.25 + 0.10), got {l2}"
        );
    }

    /// Elbow confidence crossing the gate must ROTATE the swivel
    /// smoothly, not teleport it: just-below vs just-above threshold
    /// solutions stay close, while the full blend still spans the real
    /// synthetic↔observed gap.
    #[test]
    fn pole_confidence_blend_is_continuous_at_threshold() {
        let (humanoid, rest) = arm_rig();
        let thr = 0.5_f32;
        let params = SolverParams { joint_confidence_threshold: thr, ..Default::default() };
        let solve_with_conf = |conf: f32| {
            let mut source = SourceSkeleton::empty(0);
            source.joints.insert(HumanoidBone::LeftUpperArm, src_joint([0.15, 1.40, 0.0]));
            source.joints.insert(HumanoidBone::RightUpperArm, src_joint([-0.15, 1.40, 0.0]));
            source.joints.insert(HumanoidBone::LeftHand, src_joint([0.15, 1.00, 0.0]));
            source.joints.insert(
                HumanoidBone::LeftLowerArm,
                crate::tracking::SourceJoint {
                    // Deliberately opposite the synthetic default (which
                    // leans outward / forward): inboard and behind, so the
                    // blend has a real gap to traverse.
                    position: [0.02, 1.20, -0.25],
                    confidence: conf,
                    metric_depth_m: None,
                },
            );
            let mut hold = [None, None];
            compute_arm_reach_elbows(&source, &rest, &humanoid, &params, &mut hold)[0]
                .expect("solves")
                .position
        };
        let below = solve_with_conf(thr - 0.01);
        let above = solve_with_conf(thr + 0.01);
        let full = solve_with_conf(thr + 0.20);
        let step = vec3_length(&vec3_sub(&above, &below));
        let span = vec3_length(&vec3_sub(&full, &below));
        assert!(
            step < 0.05,
            "crossing the confidence gate must not teleport the elbow: step {step}"
        );
        assert!(
            span > 0.15,
            "sanity: the blend spans a real synthetic↔observed gap, got {span}"
        );
    }

    /// When the observed pole collapses onto the shoulder→wrist axis
    /// (fully folded / fully stretched view), the held bend plane from
    /// the previous well-conditioned frame must steer the elbow — not
    /// the arbitrary world-axis fallback.
    #[test]
    fn degenerate_pole_keeps_previous_swivel_plane() {
        let (humanoid, rest) = arm_rig();
        let params = SolverParams { joint_confidence_threshold: 0.0, ..Default::default() };
        let frame = |elbow_z: f32| {
            let mut source = SourceSkeleton::empty(0);
            source.joints.insert(HumanoidBone::LeftUpperArm, src_joint([0.15, 1.40, 0.0]));
            source.joints.insert(HumanoidBone::RightUpperArm, src_joint([-0.15, 1.40, 0.0]));
            source.joints.insert(HumanoidBone::LeftHand, src_joint([0.15, 1.00, 0.0]));
            source.joints.insert(HumanoidBone::LeftLowerArm, src_joint([0.15, 1.20, elbow_z]));
            source
        };

        // Fresh state, degenerate pole (elbow exactly on the axis):
        // documents the arbitrary fallback (bends +z here).
        let mut hold = [None, None];
        let cold = compute_arm_reach_elbows(&frame(0.0), &rest, &humanoid, &params, &mut hold)[0]
            .expect("solves")
            .position;
        assert!(cold[2] > 0.0, "world-axis fallback bends +z in this rig: {cold:?}");

        // Seeded by a −z bend the frame before, the SAME degenerate
        // frame must keep bending −z.
        let mut hold = [None, None];
        let seeded = compute_arm_reach_elbows(&frame(-0.25), &rest, &humanoid, &params, &mut hold)[0]
            .expect("solves")
            .position;
        assert!(seeded[2] < 0.0, "seed frame bends −z: {seeded:?}");
        let held = compute_arm_reach_elbows(&frame(0.0), &rest, &humanoid, &params, &mut hold)[0]
            .expect("solves")
            .position;
        assert!(
            held[2] < 0.0,
            "degenerate pole must keep the held −z bend plane, got {held:?}"
        );
    }

    /// The A-pose idle direction sits an untracked arm down-and-out at the
    /// configured angle (40° from straight-down), on its own side, and follows
    /// a torso yaw — replacing the T-pose bind that would jut the arm out flat.
    #[test]
    fn idle_arm_direction_is_a_pose() {
        let ident = [0.0, 0.0, 0.0, 1.0];
        let down = [0.0, -1.0, 0.0];

        // Left arm (rest axis +x): down-and-out to the left, 40° off vertical.
        let l = idle_arm_direction([1.0, 0.0, 0.0], ident);
        assert!(l[1] < 0.0, "left idle arm should point downward: {l:?}");
        assert!(l[0] > 0.0, "left idle arm should stay on the +x side: {l:?}");
        let ang_l = vec3_dot(&vec3_normalize(&l), &down).clamp(-1.0, 1.0).acos().to_degrees();
        assert!((ang_l - 40.0).abs() < 0.5, "left arm angle from down = {ang_l}, want 40");

        // Right arm (rest axis −x): mirror — stays on the −x side.
        let r = idle_arm_direction([-1.0, 0.0, 0.0], ident);
        assert!(r[1] < 0.0 && r[0] < 0.0, "right idle arm should point down-right: {r:?}");

        // Under a 90° torso yaw (about +y) the arm still points down, but its
        // outward component rotates out of the x axis into z (sign is the quat
        // handedness convention — assert magnitude, not sign).
        let yaw90 = [0.0, (std::f32::consts::FRAC_PI_4).sin(), 0.0, (std::f32::consts::FRAC_PI_4).cos()];
        let ly = idle_arm_direction([1.0, 0.0, 0.0], yaw90);
        assert!(ly[1] < 0.0, "yawed idle arm still points down: {ly:?}");
        assert!(
            ly[0].abs() < 0.2 && ly[2].abs() > 0.4,
            "yawed idle arm's outward axis should rotate from x into z: {ly:?}"
        );
    }
}

/// Blend `raw` toward the contact-IK direction for `side` (0 = left,
/// 1 = right) and `seg` (0 = UpperArm, 1 = LowerArm) by the pass's
/// proximity weight. Identity when the pass is inactive.
fn contact_ik_dir(ik: &Option<ArmContactIk>, side: usize, seg: usize, raw: Vec3) -> Vec3 {
    let Some(ik) = ik else {
        return raw;
    };
    let Some((upper, lower)) = ik.dirs[side] else {
        return raw;
    };
    let target = if seg == 0 { upper } else { lower };
    let w = ik.weight;
    let mixed = [
        raw[0] + (target[0] - raw[0]) * w,
        raw[1] + (target[1] - raw[1]) * w,
        raw[2] + (target[2] - raw[2]) * w,
    ];
    let n = vec3_normalize(&mixed);
    if n == [0.0; 3] {
        raw
    } else {
        n
    }
}

/// Convert the GUI's frame-rate-naive `rotation_blend` slider value
/// into a dt-aware α. The slider's intent is "what fraction of the
/// new pose to absorb each frame at the reference rate"; reframing
/// it as a time-constant means the same slider value behaves
/// identically whether the camera ships 30 or 60 fps and whether
/// the renderer hitches occasionally.
/// Slerp toward `target` from the PREVIOUS FRAME's solved rotation
/// for this node (falling back to `fallback` — the rest-pose value —
/// on the first frame or after a reset), recording the result for
/// the next frame. See `PoseSolverState::prev_local_rotations` for
/// why blending from `local_transforms` directly is wrong.
#[inline]
fn blend_local_rotation(
    state: &mut PoseSolverState,
    node_idx: usize,
    fallback: Quat,
    target: &Quat,
    alpha: f32,
) -> Quat {
    let from = state
        .prev_local_rotations
        .get(&node_idx)
        .copied()
        .unwrap_or(fallback);
    let out = quat_slerp_short(&from, target, alpha);
    state.prev_local_rotations.insert(node_idx, out);
    out
}

/// Largest angle an arm-chain bone may rotate in ONE solved frame.
///
/// This is a *continuity* bound, not smoothing. The arm's target is not a
/// continuous signal: it switches between "driven by the observed elbow /
/// wrist" and "idle A-pose" whenever the observation set changes, and in this
/// user's framing (hands below frame most of the time) that happens every few
/// frames. The two targets are far apart — live capture (2026-07-28) measured
/// the avatar's hand 0.4-0.7 m away across a single such switch. A per-frame
/// lerp cannot hide that: `rotation_blend` moves 70 % of the way to the
/// current target each frame, so a discontinuous target yields a
/// 70 %-of-the-gap step, which at 20-30 fps reads as a teleport.
///
/// Capping the step turns any target discontinuity into a short traversal
/// (a 60 deg switch takes ~4 frames ≈ 0.2 s) while leaving real motion
/// untouched: 15 deg/frame is 300-450 deg/s at the shoulder, far above human
/// arm speed, and the measured p95 of the avatar hand's per-frame travel in
/// ordinary use is 1.5 mm — three orders of magnitude below the artefact.
const ARM_MAX_STEP_PER_FRAME: f32 = 0.26; // 15 degrees

/// Bones the step cap applies to: the arm chain, whose target is the one that
/// switches discontinuously. Fingers ride the hand and the spine/head are
/// driven from continuously-observed geometry, so neither needs (or should
/// get) a rate limit.
/// Per-frame rotation cap for `bone`: the arm-chain bound, or unbounded for
/// bones whose driving target is continuous.
fn arm_step_cap(bone: HumanoidBone) -> f32 {
    if is_arm_chain_bone(bone) {
        arm_max_step()
    } else {
        f32::INFINITY
    }
}

/// The arm-chain step bound, liftable via `VULVATAR_DIAG_DISABLE=stepcap`
/// so the replay bisect harness can attribute artefacts to (or exonerate)
/// the continuity bound. Live builds never set the variable.
fn arm_max_step() -> f32 {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let disabled = *DISABLED.get_or_init(|| {
        std::env::var("VULVATAR_DIAG_DISABLE")
            .map(|v| v.split(',').any(|t| t.trim() == "stepcap"))
            .unwrap_or(false)
    });
    if disabled {
        f32::INFINITY
    } else {
        ARM_MAX_STEP_PER_FRAME
    }
}

fn is_arm_chain_bone(bone: HumanoidBone) -> bool {
    use HumanoidBone::*;
    matches!(
        bone,
        LeftUpperArm | LeftLowerArm | LeftHand | RightUpperArm | RightLowerArm | RightHand
    )
}

/// [`blend_local_rotation`] plus the end-effector rotation hold: when
/// `rest > 0` (the arm is at rest, see [`ARM_HOLD_ANG_DEAD`]) the blend's
/// per-frame rotation step is soft-thresholded, freezing the residual
/// jitter of a converged pose while letting any larger — i.e. intentional
/// — step through. `rest == 0` reduces exactly to [`blend_local_rotation`],
/// so non-arm bones are provably unaffected.
#[inline]
fn blend_arm_rotation(
    state: &mut PoseSolverState,
    node_idx: usize,
    fallback: Quat,
    target: &Quat,
    alpha: f32,
    rest: f32,
    max_step: f32,
) -> Quat {
    let from = state
        .prev_local_rotations
        .get(&node_idx)
        .copied()
        .unwrap_or(fallback);
    let mut eff_alpha = alpha;
    if rest > 0.0 {
        // The slerp would rotate `from` toward `target` by `ang * alpha`
        // this frame. Freeze that step below `dead`; soft-threshold above
        // it so crossing the boundary is continuous (no snap on release).
        let ang = quat_angle_between(&from, target);
        let step = ang * alpha;
        let keep = soft_threshold_keep(step, ARM_HOLD_ANG_DEAD * rest);
        eff_alpha = alpha * keep;
    }
    // Continuity bound (see `ARM_MAX_STEP_PER_FRAME`). `max_step` is
    // infinite for bones whose target is continuous, making this a no-op.
    if max_step.is_finite() {
        let ang = quat_angle_between(&from, target);
        if ang * eff_alpha > max_step {
            eff_alpha = (max_step / ang.max(1e-6)).min(eff_alpha);
        }
    }
    let out = quat_slerp_short(&from, target, eff_alpha);
    state.prev_local_rotations.insert(node_idx, out);
    out
}

/// Rest factor for `joint`'s filter output speed: 1 at rest
/// (≤ [`REST_SPEED_LO`]), 0 moving (≥ [`REST_SPEED_HI`]). The filter
/// output is the key — at rest it converges so its speed collapses
/// toward zero, while intentional motion sustains it — so this gate never
/// freezes a deliberate move, only true rest buzz. Falls back to 0 (no
/// hold) when the joint filter has no sample yet.
fn arm_rest_factor(state: &PoseSolverState, joint: HumanoidBone) -> f32 {
    match state.joint_filters.get(&joint) {
        Some(f) if f.initialized => 1.0 - smoothstep(f.out_speed, REST_SPEED_LO, REST_SPEED_HI),
        _ => 0.0,
    }
}

/// The rest gate for one arm bone's rotation hold, keyed by the joint
/// that *drives* that bone's direction match — the upper arm follows the
/// elbow keypoint and the forearm follows the wrist (`DRIVEN_BONES`'
/// `Tip::Joint` targets). Gating the upper arm on the wrist was measured
/// wrong: the elbow can orbit the shoulder while the wrist stays put in
/// world space, which would freeze the upper arm mid-gesture. Non-arm
/// bones return 0 (no hold).
fn arm_hold_rest(state: &PoseSolverState, bone: HumanoidBone) -> f32 {
    let driving_joint = match bone {
        HumanoidBone::LeftUpperArm => HumanoidBone::LeftLowerArm,
        HumanoidBone::LeftLowerArm | HumanoidBone::LeftHand => HumanoidBone::LeftHand,
        HumanoidBone::RightUpperArm => HumanoidBone::RightLowerArm,
        HumanoidBone::RightLowerArm | HumanoidBone::RightHand => HumanoidBone::RightHand,
        _ => return 0.0,
    };
    arm_rest_factor(state, driving_joint)
}

#[inline]
fn dt_aware_blend(slider_blend: f32, dt: f32) -> f32 {
    if dt <= 0.0 {
        return 0.0;
    }
    let b = slider_blend.clamp(0.0, 1.0);
    if b >= 1.0 {
        return 1.0;
    }
    if b <= 0.0 {
        return 0.0;
    }
    let tau = -ROTATION_BLEND_REFERENCE_DT / (1.0 - b).ln();
    1.0 - (-dt / tau).exp()
}

/// Apply the 1€ filter to every joint and fingertip position, then
/// gate confidences via per-bone Schmitt hysteresis. Returns a
/// fresh `SourceSkeleton` so `&SourceSkeleton` callers don't need to
/// hand over a mutable reference.
fn preprocess_source(
    source: &SourceSkeleton,
    state: &mut PoseSolverState,
    dt: f32,
    params: &SolverParams,
) -> SourceSkeleton {
    let mut out = source.clone();

    for (bone, joint) in out.joints.iter_mut() {
        let tuning = one_euro_params_for(*bone);
        let filt = state.joint_filters.entry(*bone).or_default();
        joint.position = filt.apply(joint.position, dt, tuning);
    }
    // Schmitt hysteresis: a single dropout below `enter` does not turn
    // the joint off as long as it stays above `exit`. Shared with the
    // repeated-sample path, which refreshes confidences from the live
    // (possibly hold-decayed) sample and must gate them identically.
    gate_joint_confidences(&mut out, state, params);

    for (bone, joint) in out.fingertips.iter_mut() {
        let tuning = one_euro_params_for(*bone);
        let filt = state.fingertip_filters.entry(*bone).or_default();
        joint.position = filt.apply(joint.position, dt, tuning);
        // Fingertips don't go through the body-bone DRIVEN_BONES list
        // for confidence gating directly — the distal bone owns the
        // gate — so we only smooth the position here.
    }

    // Smooth the hand orientation (forward/up) the same way the joints are
    // smoothed. The wrist rotation consumes `HandOrientation` directly and
    // the palm-normal cross product is noisy, so held finger poses jittered
    // when it was used raw. Filter each vector per-component, then
    // re-normalise (the 1€ filter doesn't preserve unit length).
    let normalize3 = |v: [f32; 3]| -> [f32; 3] {
        let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if m > 1e-6 {
            [v[0] / m, v[1] / m, v[2] / m]
        } else {
            v
        }
    };
    // Palm forward/up are the hand tier (the wrist's own orientation
    // stream), so they use the smoother 1€ cutoff — but NOT the
    // positional deadband (unit direction vectors are a different scale
    // from the position radius; freezing them would need an angular
    // threshold — left as future work) and NOT the Z anisotropy (the
    // third component of a unit direction is not camera depth).
    let orient = OneEuroTuning {
        dead_radius: 0.0,
        z_min_cutoff: HAND_MIN_CUTOFF_HZ,
        z_beta: HAND_BETA,
        z_dead_radius: 0.0,
        ..OneEuroTuning::HAND
    };
    if let Some(o) = out.left_hand_orientation.as_mut() {
        o.forward = normalize3(state.hand_orient_filters[0].apply(o.forward, dt, orient));
        o.up = normalize3(state.hand_orient_filters[1].apply(o.up, dt, orient));
    }
    if let Some(o) = out.right_hand_orientation.as_mut() {
        o.forward = normalize3(state.hand_orient_filters[2].apply(o.forward, dt, orient));
        o.up = normalize3(state.hand_orient_filters[3].apply(o.up, dt, orient));
    }

    // Smooth the face pose angles. The head-orientation channel is
    // otherwise unfiltered — `apply_face_pose` passes raw yaw/pitch/roll
    // straight to the head bone — and it is the noisiest angular signal on
    // the avatar (measured SRC roll ≈ 4°/frame at rest), so at rest the
    // head visibly wobbles. Pack `[yaw, pitch, roll]` so roll takes the
    // heavier `z_*` slot of the FACE tuning; the 1€ collapses to its low
    // rest cutoff when the head is still and opens on a genuine turn.
    if let Some(face) = out.face.as_mut() {
        let s = state
            .face_angle_filter
            .apply([face.yaw, face.pitch, face.roll], dt, OneEuroTuning::FACE);
        face.yaw = s[0];
        face.pitch = s[1];
        face.roll = s[2];
    }

    // Anatomical finger constraints (run on the smoothed keypoints): clamp
    // each finger's curl to a single hinge plane with physiological angle
    // limits, removing the impossible twist / lateral wobble that noisy
    // landmarks otherwise feed into the unconstrained shortest-arc solve.
    constrain_fingers(&mut out);

    out
}

/// Per-bone Schmitt confidence gate (enter at the threshold, exit at
/// `SCHMITT_EXIT_RATIO ×` it). Factored out of [`preprocess_source`] so
/// the repeated-sample path can re-gate refreshed confidences without
/// re-running the position filters.
fn gate_joint_confidences(
    out: &mut SourceSkeleton,
    state: &mut PoseSolverState,
    params: &SolverParams,
) {
    let enter = params.joint_confidence_threshold;
    let exit = enter * SCHMITT_EXIT_RATIO;
    for (bone, joint) in out.joints.iter_mut() {
        let active = state.joint_active.entry(*bone).or_insert(false);
        let raw = joint.confidence;
        if *active {
            if raw < exit {
                *active = false;
                joint.confidence = 0.0;
            }
        } else if raw >= enter {
            *active = true;
        } else {
            joint.confidence = 0.0;
        }
    }
}

/// Copy every confidence channel (and the expression weights) from the
/// live sample onto a cached filtered skeleton. Used when the render
/// loop observes the SAME camera sample again: geometry comes from the
/// cache (measurement filters must not advance) but the hold/fade
/// policy may have decayed the live sample's confidences, and that
/// decay must still reach the solver's gates.
fn refresh_confidence_channels(cached: &mut SourceSkeleton, live: &SourceSkeleton) {
    for (bone, joint) in cached.joints.iter_mut() {
        if let Some(l) = live.joints.get(bone) {
            joint.confidence = l.confidence;
        }
    }
    for (bone, tip) in cached.fingertips.iter_mut() {
        if let Some(l) = live.fingertips.get(bone) {
            tip.confidence = l.confidence;
        }
    }
    if let (Some(cf), Some(lf)) = (cached.face.as_mut(), live.face.as_ref()) {
        cf.confidence = lf.confidence;
    }
    cached.face_mesh_confidence = live.face_mesh_confidence;
    if let (Some(c), Some(l)) = (
        cached.left_hand_orientation.as_mut(),
        live.left_hand_orientation.as_ref(),
    ) {
        c.confidence = l.confidence;
    }
    if let (Some(c), Some(l)) = (
        cached.right_hand_orientation.as_mut(),
        live.right_hand_orientation.as_ref(),
    ) {
        c.confidence = l.confidence;
    }
    cached.overall_confidence = live.overall_confidence;
    cached.expressions = live.expressions.clone();
}

/// Re-project each finger's PIP/DIP joints onto a single anatomical curl
/// plane and clamp the bend magnitudes. The hinge axis is the knuckle
/// (medio-lateral) axis = `palm_normal × proximal_phalanx`; fingers flex
/// in the plane spanned by the phalanx and the palm normal. This removes
/// the out-of-plane twist / lateral jitter that unconstrained shortest-arc
/// direction matching produced from noisy MediaPipe keypoints — the
/// dominant cause of thumbs-up / pointing-pose wobble. Runs on the already
/// 1€-smoothed source, so bend angles are temporally stable before they're
/// clamped. The proximal (MCP) joint is left as measured: it is a genuine
/// 2-DoF joint carrying flexion *and* abduction/spread, which the curl
/// plane must not flatten. Requires the hand orientation (palm normal);
/// fingers on a hand with no orientation this frame are left untouched.
fn constrain_fingers(out: &mut SourceSkeleton) {
    use HumanoidBone::*;
    // (proximal, intermediate, distal, is_left)
    let fingers: [(HumanoidBone, HumanoidBone, HumanoidBone, bool); 10] = [
        (LeftThumbProximal, LeftThumbIntermediate, LeftThumbDistal, true),
        (LeftIndexProximal, LeftIndexIntermediate, LeftIndexDistal, true),
        (LeftMiddleProximal, LeftMiddleIntermediate, LeftMiddleDistal, true),
        (LeftRingProximal, LeftRingIntermediate, LeftRingDistal, true),
        (LeftLittleProximal, LeftLittleIntermediate, LeftLittleDistal, true),
        (RightThumbProximal, RightThumbIntermediate, RightThumbDistal, false),
        (RightIndexProximal, RightIndexIntermediate, RightIndexDistal, false),
        (RightMiddleProximal, RightMiddleIntermediate, RightMiddleDistal, false),
        (RightRingProximal, RightRingIntermediate, RightRingDistal, false),
        (RightLittleProximal, RightLittleIntermediate, RightLittleDistal, false),
    ];
    // Physiological flex cap (radians, magnitude only — the measured sign
    // carries the curl direction, which flips with the hand's palm normal).
    const MAX_FLEX: f32 = 2.0; // ~115°

    for (prox, inter, dist, is_left) in fingers {
        let palm_normal = match if is_left {
            out.left_hand_orientation.as_ref()
        } else {
            out.right_hand_orientation.as_ref()
        } {
            Some(o) => o.up,
            None => continue,
        };
        let (Some(p0), Some(p1), Some(p2)) = (
            out.joints.get(&prox).map(|j| j.position),
            out.joints.get(&inter).map(|j| j.position),
            out.joints.get(&dist).map(|j| j.position),
        ) else {
            continue;
        };
        let Some(p3) = out.fingertips.get(&dist).map(|j| j.position) else {
            continue;
        };

        let seg1 = vec3_sub(&p1, &p0);
        let seg2 = vec3_sub(&p2, &p1);
        let seg3 = vec3_sub(&p3, &p2);
        let len2 = vec3_length(&seg2);
        let len3 = vec3_length(&seg3);
        if vec3_length(&seg1) < 1e-5 || len2 < 1e-5 || len3 < 1e-5 {
            continue;
        }
        let dir1 = vec3_normalize(&seg1);
        let hinge = vec3_cross(&palm_normal, &dir1);
        if vec3_length(&hinge) < 1e-5 {
            continue; // phalanx parallel to palm normal — axis undefined.
        }
        let hinge = vec3_normalize(&hinge);

        // PIP: signed bend of seg2 vs seg1 about the hinge, clamped, then
        // re-projected exactly into the hinge plane.
        let pip = signed_angle(&dir1, &vec3_normalize(&seg2), &hinge).clamp(-MAX_FLEX, MAX_FLEX);
        let dir2 = rotate_about_axis(&dir1, &hinge, pip);
        // DIP: clamped and softly coupled — it cannot bend much past the
        // PIP (tendon coupling) and stays in the same plane.
        let dip_cap = (pip.abs() * 1.3).min(MAX_FLEX);
        let dip = signed_angle(&vec3_normalize(&seg2), &vec3_normalize(&seg3), &hinge)
            .clamp(-dip_cap, dip_cap);
        let dir3 = rotate_about_axis(&dir2, &hinge, dip);

        let p2_new = [
            p1[0] + dir2[0] * len2,
            p1[1] + dir2[1] * len2,
            p1[2] + dir2[2] * len2,
        ];
        let p3_new = [
            p2_new[0] + dir3[0] * len3,
            p2_new[1] + dir3[1] * len3,
            p2_new[2] + dir3[2] * len3,
        ];
        if let Some(j) = out.joints.get_mut(&dist) {
            j.position = p2_new;
        }
        if let Some(j) = out.fingertips.get_mut(&dist) {
            j.position = p3_new;
        }
    }
}

/// Rotate `v` about unit axis `k` by `theta` radians (Rodrigues).
fn rotate_about_axis(v: &[f32; 3], k: &[f32; 3], theta: f32) -> [f32; 3] {
    let (s, c) = theta.sin_cos();
    let kxv = vec3_cross(k, v);
    let kdv = vec3_dot(k, v) * (1.0 - c);
    [
        v[0] * c + kxv[0] * s + k[0] * kdv,
        v[1] * c + kxv[1] * s + k[1] * kdv,
        v[2] * c + kxv[2] * s + k[2] * kdv,
    ]
}

/// Signed angle from unit vector `a` to unit vector `b` about unit `axis`.
fn signed_angle(a: &[f32; 3], b: &[f32; 3], axis: &[f32; 3]) -> f32 {
    let cross = vec3_cross(a, b);
    vec3_dot(&cross, axis).atan2(vec3_dot(a, b))
}

/// True for the six lower-body bones the solver drives (per-side
/// upper-leg, lower-leg, foot). Gated by
/// `SolverParams::lower_body_tracking_enabled` so the user can keep
/// the avatar's legs locked to rest pose when the camera only frames
/// the upper body.
fn is_lower_body_bone(b: HumanoidBone) -> bool {
    use HumanoidBone::*;
    matches!(
        b,
        LeftUpperLeg | LeftLowerLeg | LeftFoot | RightUpperLeg | RightLowerLeg | RightFoot
    )
}

/// True for the 30 finger bones (5 fingers × 3 phalanges × 2 hands).
/// Used by `solve_avatar_pose` to detect the body→finger transition
/// in the `DRIVEN_BONES` iteration so the wrist orientation pass can
/// fire once at the right moment.
fn is_finger_bone(b: HumanoidBone) -> bool {
    use HumanoidBone::*;
    matches!(
        b,
        LeftThumbProximal | LeftThumbIntermediate | LeftThumbDistal
        | LeftIndexProximal | LeftIndexIntermediate | LeftIndexDistal
        | LeftMiddleProximal | LeftMiddleIntermediate | LeftMiddleDistal
        | LeftRingProximal | LeftRingIntermediate | LeftRingDistal
        | LeftLittleProximal | LeftLittleIntermediate | LeftLittleDistal
        | RightThumbProximal | RightThumbIntermediate | RightThumbDistal
        | RightIndexProximal | RightIndexIntermediate | RightIndexDistal
        | RightMiddleProximal | RightMiddleIntermediate | RightMiddleDistal
        | RightRingProximal | RightRingIntermediate | RightRingDistal
        | RightLittleProximal | RightLittleIntermediate | RightLittleDistal
    )
}

/// Drive a wrist bone with a full 3-DoF rotation derived from the
/// tracker's `HandOrientation` (forward + palm-normal pair). Without
/// this pass the wrist's twist axis is unconstrained — the LowerArm
/// chain settles a wrist *position* via direction-match but its
/// rotation around the LowerArm→wrist axis is whatever shortest-arc
/// picked, leaving the avatar's fingers rotated 90° at the base.
///
/// Both rest and source bases are computed the same way (cross-
/// orthogonalised forward+up from wrist + 3 MCPs) so the bias from
/// using the hand-track wrist (which has noisy Z under occlusion)
/// cancels between the two basis pairs.
#[allow(clippy::too_many_arguments)]
fn solve_wrist_orientation(
    wrist_bone: HumanoidBone,
    middle_proximal: HumanoidBone,
    index_proximal: HumanoidBone,
    pinky_proximal: HumanoidBone,
    source_orientation: Option<HandOrientation>,
    skeleton: &SkeletonAsset,
    humanoid: &HumanoidMap,
    rest_world: &[WorldXform],
    current_world: &mut [WorldXform],
    local_transforms: &mut [Transform],
    params: &SolverParams,
    state: &mut PoseSolverState,
    dt: f32,
) {
    let Some(orient) = source_orientation else {
        return;
    };
    if orient.confidence < params.joint_confidence_threshold {
        return;
    }

    // Same rest hold as the arm chain in the main loop: this pass is the
    // final writer of the forearm-twist and hand rotations, so without
    // the hold here the wrist's residual orientation buzz would bypass
    // the arm hold entirely.
    let hold_rest = arm_rest_factor(state, wrist_bone);

    let Some(wrist_node) = humanoid.bone_map.get(&wrist_bone).copied() else {
        return;
    };
    let Some(middle_node) = humanoid.bone_map.get(&middle_proximal).copied() else {
        return;
    };
    let Some(index_node) = humanoid.bone_map.get(&index_proximal).copied() else {
        return;
    };
    let Some(pinky_node) = humanoid.bone_map.get(&pinky_proximal).copied() else {
        return;
    };

    let wrist_idx = wrist_node.0 as usize;
    let middle_idx = middle_node.0 as usize;
    let index_idx = index_node.0 as usize;
    let pinky_idx = pinky_node.0 as usize;
    if wrist_idx >= skeleton.nodes.len()
        || middle_idx >= rest_world.len()
        || index_idx >= rest_world.len()
        || pinky_idx >= rest_world.len()
    {
        return;
    }

    // Rest basis: forward = middle MCP relative to wrist, up = palm
    // plane normal. Same construction as `attach_hand` so the basis
    // pair maps cleanly with `quat_from_basis_pair`.
    let rest_wrist = rest_world[wrist_idx].position;
    let rest_middle = rest_world[middle_idx].position;
    let rest_index = rest_world[index_idx].position;
    let rest_pinky = rest_world[pinky_idx].position;
    let rest_raw_fwd = vec3_sub(&rest_middle, &rest_wrist);
    let rest_across = vec3_sub(&rest_index, &rest_pinky);
    let rest_normal_raw = vec3_cross(&rest_across, &rest_raw_fwd);
    let rest_up = vec3_normalize(&rest_normal_raw);
    if rest_up == [0.0; 3] {
        return;
    }
    let rest_fwd = vec3_normalize(&vec3_cross(&rest_up, &rest_across));
    if rest_fwd == [0.0; 3] {
        return;
    }

    // Walk the parent's accumulated yaw delta into the rest basis,
    // mirroring the body chain's `parent_yaw_delta` rebase. Without
    // this the wrist would fight the spine yaw for the chain's
    // overall rotation when the subject turns their body.
    let parent_world_rot = skeleton.nodes[wrist_idx]
        .parent
        .map(|NodeId(p)| current_world[p as usize].rotation)
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let parent_rest_world_rot = skeleton.nodes[wrist_idx]
        .parent
        .map(|NodeId(p)| rest_world[p as usize].rotation)
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let parent_yaw_delta =
        quat_mul(&parent_world_rot, &quat_conjugate(&parent_rest_world_rot));
    let rest_fwd_current = quat_rotate_vec3(&parent_yaw_delta, &rest_fwd);
    let rest_up_current = quat_rotate_vec3(&parent_yaw_delta, &rest_up);
    let current_rest_world_rot =
        quat_mul(&parent_yaw_delta, &rest_world[wrist_idx].rotation);

    // Full 3-DoF basis-pair alignment.
    let delta_world = quat_from_basis_pair(
        &rest_fwd_current,
        &rest_up_current,
        &orient.forward,
        &orient.up,
    );
    let new_world_rot = quat_mul(&delta_world, &current_rest_world_rot);

    // Anatomical twist redistribution. The radius rotates over the
    // ulna inside the forearm, so what looks like "wrist rotation" to
    // a viewer is actually shared between the forearm and the wrist
    // joint — split roughly 50/50 in a real arm. If we apply the
    // entire `delta_world` to the Hand bone alone (which is what the
    // body chain's separate LowerArm direction-match leaves us with),
    // pronation past ~30° looks like the wrist has been snapped off
    // its parent: the hand twists while the forearm sleeve stays put.
    //
    // Decompose `delta_world` into (swing, twist) around the forearm
    // length axis (LowerArm origin → wrist origin in world space) and
    // hand half of the twist over to the LowerArm. The Hand still
    // ends up at `new_world_rot` because its local rotation is
    // recomputed against the updated parent.
    //
    // The LowerArm is the wrist's direct parent in standard humanoid
    // rigs (UpperArm → LowerArm → Hand). When that's not the case
    // (or the forearm length is degenerate) we skip the redistribution
    // and apply the full delta to the Hand alone — the same behaviour
    // that shipped before this change.
    const FOREARM_TWIST_SHARE: f32 = 0.5;
    let blend = dt_aware_blend(params.rotation_blend, dt);

    let lower_arm_idx = skeleton.nodes[wrist_idx].parent.map(|NodeId(p)| p as usize);
    // Forearm twist axis in WORLD space, derived from the LowerArm's
    // *current* world rotation rather than `current_world.position`.
    // The latter never gets re-FK'd after the per-bone direction-match
    // pass updates rotations — it still holds rest-pose positions, so
    // for any pose where the forearm rotated significantly off T-pose
    // (hands-on-hips, hands-behind-head, "I dunno" shrug) the wrist
    // position reads the REST world location and the resulting axis
    // points along the rest-pose forearm (≈ ±X) instead of the
    // current bone axis. Pre-multiplying `delta_world` by a "twist"
    // that's actually a swing around world X then knocks the
    // LowerArm out of its just-set direction, dragging the hand
    // toward the body midline. Compute the axis from
    // `current_world[la_idx].rotation` (which IS kept up-to-date) and
    // the bone's rest forearm direction instead.
    let twist_axis = lower_arm_idx.and_then(|la_idx| {
        // The bone vector in the LowerArm's OWN (parent-of-wrist)
        // space is the wrist's rest LOCAL translation. The previous
        // formulation rotated the rest *world* direction
        // (rest_world[wrist] − rest_world[la]) by the LowerArm's
        // current world rotation — that double-applies the
        // LowerArm's rest world rotation (the world rest direction
        // already contains it), tilting the axis on any rig whose
        // rest rotations are not identity. A tilted axis lets swing
        // leak into the "twist" half that's transferred to the
        // LowerArm, dragging the hand off the just-solved forearm
        // direction (measured: 0.41 foreshortening error on the
        // lean-left open-palm validation pose; 0.02 with the correct
        // axis).
        let rest_dir_local = vec3_normalize(&skeleton.nodes[wrist_idx].rest_local.translation);
        if rest_dir_local == [0.0; 3] {
            return None;
        }
        let current_world_dir =
            quat_rotate_vec3(&current_world[la_idx].rotation, &rest_dir_local);
        let normalized = vec3_normalize(&current_world_dir);
        if normalized == [0.0; 3] {
            None
        } else {
            Some(normalized)
        }
    });

    if let (Some(la_idx), Some(axis)) = (lower_arm_idx, twist_axis) {
        let (_, twist_world) = swing_twist_decompose(&delta_world, &axis);
        let half_twist = quat_slerp_short(&[0.0, 0.0, 0.0, 1.0], &twist_world, FOREARM_TWIST_SHARE);

        // LowerArm's new world rotation = pre-multiply its current
        // world by the half-twist (twist axis is in world space, so
        // the twist composes on the LEFT).
        let lower_arm_world_old = current_world[la_idx].rotation;
        let lower_arm_world_new = quat_mul(&half_twist, &lower_arm_world_old);

        // Recompose into LowerArm.local against ITS parent (UpperArm).
        let upper_arm_world_rot = skeleton.nodes[la_idx]
            .parent
            .map(|NodeId(p)| current_world[p as usize].rotation)
            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let lower_arm_local_target = quat_normalize(&quat_mul(
            &quat_conjugate(&upper_arm_world_rot),
            &lower_arm_world_new,
        ));
        local_transforms[la_idx].rotation = blend_arm_rotation(
            state,
            la_idx,
            local_transforms[la_idx].rotation,
            &lower_arm_local_target,
            blend,
            hold_rest,
            arm_max_step(),
        );
        let actual_lower_arm_world =
            quat_mul(&upper_arm_world_rot, &local_transforms[la_idx].rotation);
        current_world[la_idx].rotation = actual_lower_arm_world;

        // Hand.local against the now-twisted LowerArm world. The
        // Hand's *world* target is unchanged (`new_world_rot`); only
        // the parent moved, so the local representation shifts to
        // compensate and the visible chain still hits the desired
        // wrist orientation.
        let hand_local_target = quat_normalize(&quat_mul(
            &quat_conjugate(&actual_lower_arm_world),
            &new_world_rot,
        ));
        local_transforms[wrist_idx].rotation = blend_arm_rotation(
            state,
            wrist_idx,
            local_transforms[wrist_idx].rotation,
            &hand_local_target,
            blend,
            hold_rest,
            arm_max_step(),
        );
        let updated_world =
            quat_mul(&actual_lower_arm_world, &local_transforms[wrist_idx].rotation);
        current_world[wrist_idx].rotation = updated_world;
    } else {
        // Fallback: forearm degenerate or no parent — apply the full
        // delta to the Hand alone, preserving the pre-redistribution
        // behaviour for edge cases.
        let new_local_rot =
            quat_normalize(&quat_mul(&quat_conjugate(&parent_world_rot), &new_world_rot));
        local_transforms[wrist_idx].rotation = blend_arm_rotation(
            state,
            wrist_idx,
            local_transforms[wrist_idx].rotation,
            &new_local_rot,
            blend,
            hold_rest,
            arm_max_step(),
        );
        let updated_world =
            quat_mul(&parent_world_rot, &local_transforms[wrist_idx].rotation);
        current_world[wrist_idx].rotation = updated_world;
    }
}
// Slerp / lerp
/// Shortest-arc angle between two unit quaternions, in radians. Uses the
/// absolute dot so `q` and `−q` (same orientation) read as zero angle.
#[inline]
fn quat_angle_between(a: &Quat, b: &Quat) -> f32 {
    let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]).abs().clamp(0.0, 1.0);
    2.0 * dot.acos()
}

fn quat_slerp_short(a: &Quat, b: &Quat, t: f32) -> Quat {
    // Ensure shortest-arc blend.
    let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let sign = if dot < 0.0 {
        dot = -dot;
        -1.0
    } else {
        1.0
    };
    if t <= 0.0 {
        return *a;
    }
    if t >= 1.0 {
        return [sign * b[0], sign * b[1], sign * b[2], sign * b[3]];
    }
    // Use lerp for cheapness and numerical stability when dot > 0.995;
    // slerp is not a hot path given we call this ~20 times per frame but
    // the branch keeps the math correct for large angle diffs.
    if dot > 0.9995 {
        let inv_t = 1.0 - t;
        return quat_normalize(&[
            inv_t * a[0] + t * sign * b[0],
            inv_t * a[1] + t * sign * b[1],
            inv_t * a[2] + t * sign * b[2],
            inv_t * a[3] + t * sign * b[3],
        ]);
    }
    let theta = dot.acos();
    let sin_theta = theta.sin();
    let w_a = ((1.0 - t) * theta).sin() / sin_theta;
    let w_b = (t * theta).sin() / sin_theta;
    [
        w_a * a[0] + w_b * sign * b[0],
        w_a * a[1] + w_b * sign * b[1],
        w_a * a[2] + w_b * sign * b[2],
        w_a * a[3] + w_b * sign * b[3],
    ]
}

/// Resolved expression weight (unchanged from the pre-refactor API, kept
/// so downstream rendering code does not need to learn a new type name).
#[derive(Clone, Debug)]
pub struct ResolvedExpressionWeight {
    pub name: String,
    pub weight: f32,
}

/// Resolve tracking expression weights against the avatar's named blend
/// shapes. Expression names must match the VRM 1.0 canonical identifiers
/// (e.g. "aa", "blink", "blinkLeft"); the VRM loader already canonicalises
/// 0.x presetName values into this space.
pub fn solve_expressions(
    source: &SourceSkeleton,
    avatar_expressions: &crate::asset::ExpressionAssetSet,
    previous: Option<&[ResolvedExpressionWeight]>,
    expression_blend: f32,
    face_confidence_threshold: f32,
    mouth_source: crate::tracking::MouthSource,
    state: &mut PoseSolverState,
) -> Vec<ResolvedExpressionWeight> {
    use crate::tracking::MouthSource;
    // VRM mouth visemes whose driver (audio lip-sync vs camera) is
    // selectable. Everything else (eyes / brows / emotions) always comes
    // from the camera.
    const MOUTH_VISEMES: [&str; 5] = ["aa", "ih", "ou", "ee", "oh"];
    // Gate on the face mesh's own confidence (the FaceMesh model
    // outputs an "is this a face" sigmoid). When the gate fails we
    // return the previous weights verbatim — leaves the avatar's
    // expression frozen at the last good value rather than snapping
    // to neutral, which is what users expect when the face briefly
    // turns away from camera.
    let face_conf = source.face_mesh_confidence.unwrap_or(0.0);
    if face_conf < face_confidence_threshold {
        return previous.map(|p| p.to_vec()).unwrap_or_default();
    }

    let prev_map: HashMap<&str, f32> = previous
        .map(|p| p.iter().map(|w| (w.name.as_str(), w.weight)).collect())
        .unwrap_or_default();

    avatar_expressions
        .expressions
        .iter()
        .filter_map(|expr_def| {
            let tracking = source
                .expressions
                .iter()
                .find(|e| e.name == expr_def.name)?;
            let raw = tracking.weight.clamp(0.0, 1.0);
            let prev_w = prev_map.get(expr_def.name.as_str()).copied().unwrap_or(raw);
            // Rest deadband (eye/brow path): shrink a sub-threshold
            // per-frame delta toward zero so blink/eye micro-flutter is
            // frozen, while a real blink (Δ≈1.0) passes almost unshrunk.
            // Soft-thresholded so crossing the boundary is continuous.
            let delta = raw - prev_w;
            let deadbanded = prev_w + delta * soft_threshold_keep(delta.abs(), EXPR_REST_DEAD);
            let blended = prev_w + expression_blend * (deadbanded - prev_w);
            let weight = if MOUTH_VISEMES.contains(&expr_def.name.as_str()) {
                // `prev_w` carries the audio lip-sync value: `step_lipsync`
                // runs just before the face solve each frame and writes the
                // mouth visemes into the weights `previous` points at. So the
                // source policy mixes camera against audio (`prev_w`).
                //
                // The camera viseme is eased here with its own EMA: the
                // eye/brow path above is smoothed by `blended`, but the mouth
                // policy bypasses that, so the raw FaceMesh blendshape would
                // otherwise jitter the mouth frame-to-frame. The audio side is
                // already smoothed upstream by the lip-sync `smoothing`.
                let cam = {
                    let ema = state
                        .mouth_viseme_ema
                        .entry(expr_def.name.clone())
                        .or_insert(raw);
                    *ema += expression_blend * (raw - *ema);
                    *ema
                };
                match mouth_source {
                    MouthSource::Audio => prev_w,
                    MouthSource::Image => cam,
                    MouthSource::Both => cam.max(prev_w),
                }
            } else {
                blended
            };
            Some(ResolvedExpressionWeight {
                name: expr_def.name.clone(),
                weight: weight.clamp(0.0, 1.0),
            })
        })
        .collect()
}

#[cfg(test)]
mod metric_orientation_tests {
    use super::*;
    use crate::asset::{HumanoidMap, NodeId};
    use crate::math_utils::quat_rotate_vec3;
    use crate::tracking::source_skeleton::SourceJoint;
    use std::collections::HashMap;

    fn put(sk: &mut SourceSkeleton, bone: HumanoidBone, pos: [f32; 3], confidence: f32) {
        sk.joints
            .insert(bone, SourceJoint { position: pos, confidence, metric_depth_m: None });
    }

    /// Minimal rig for [`compute_shoulder_align_rotation`], which reads only
    /// the two upper-arm bones: LeftUpperArm at node 0, RightUpperArm at
    /// node 1, with the given rest world positions.
    fn rest_rig(l: [f32; 3], r: [f32; 3]) -> (HumanoidMap, Vec<WorldXform>) {
        let mut bone_map = HashMap::new();
        bone_map.insert(HumanoidBone::LeftUpperArm, NodeId(0));
        bone_map.insert(HumanoidBone::RightUpperArm, NodeId(1));
        let rest_world = vec![
            WorldXform { position: l, rotation: [0.0, 0.0, 0.0, 1.0] },
            WorldXform { position: r, rotation: [0.0, 0.0, 0.0, 1.0] },
        ];
        (HumanoidMap { bone_map }, rest_world)
    }

    /// Frontal subject whose shoulder line matches the avatar's rest line →
    /// (near-)identity: the metric torso-align adds no yaw when the subject
    /// faces the camera square-on. This is the real-3D counterpart to the
    /// monocular `shoulder_only_falls_back_to_single_pair` front case, with
    /// NO foreshortening/running-max machinery in the path.
    #[test]
    fn frontal_shoulders_give_identity() {
        let (humanoid, rest_world) = rest_rig([0.15, 1.4, 0.0], [-0.15, 1.4, 0.0]);
        let mut sk = SourceSkeleton::empty(0);
        put(&mut sk, HumanoidBone::LeftUpperArm, [0.2, 0.5, 0.0], 1.0);
        put(&mut sk, HumanoidBone::RightUpperArm, [-0.2, 0.5, 0.0], 1.0);
        let q = compute_shoulder_align_rotation(&sk, &rest_world, &humanoid, 0.1)
            .expect("both shoulders clear threshold");
        // Rotating the avatar's rest shoulder dir (+X) by q must land back on
        // the source dir (+X) — i.e. q ≈ identity.
        let rotated = quat_rotate_vec3(&q, &[1.0, 0.0, 0.0]);
        assert!(
            (rotated[0] - 1.0).abs() < 1e-3 && rotated[1].abs() < 1e-3 && rotated[2].abs() < 1e-3,
            "frontal → identity, got {rotated:?}"
        );
    }

    /// 90° yaw: the subject's real shoulder line runs along the camera Z
    /// axis. The alignment rotation carries the avatar's +X rest line straight
    /// onto that −Z line — the 3D shoulder direction drives yaw directly.
    #[test]
    fn quarter_turn_maps_rest_line_onto_source_line() {
        let (humanoid, rest_world) = rest_rig([0.15, 1.4, 0.0], [-0.15, 1.4, 0.0]);
        let mut sk = SourceSkeleton::empty(0);
        // Left shoulder behind, right in front → shoulder dir L−R = (0,0,−0.4).
        put(&mut sk, HumanoidBone::LeftUpperArm, [0.0, 0.5, -0.2], 1.0);
        put(&mut sk, HumanoidBone::RightUpperArm, [0.0, 0.5, 0.2], 1.0);
        let q = compute_shoulder_align_rotation(&sk, &rest_world, &humanoid, 0.1).expect("shoulders");
        let rotated = quat_rotate_vec3(&q, &[1.0, 0.0, 0.0]);
        assert!(
            rotated[2] < -0.99 && rotated[0].abs() < 1e-2 && rotated[1].abs() < 1e-2,
            "quarter-turn should map +X→−Z, got {rotated:?}"
        );
    }

    /// A shoulder below the confidence threshold (e.g. gated out of frame by
    /// the 2D border gate) → `None`, so the caller leaves the hips at rest
    /// ("unmeasured") rather than inventing an orientation from one point.
    #[test]
    fn low_confidence_shoulder_returns_none() {
        let (humanoid, rest_world) = rest_rig([0.15, 1.4, 0.0], [-0.15, 1.4, 0.0]);
        let mut sk = SourceSkeleton::empty(0);
        put(&mut sk, HumanoidBone::LeftUpperArm, [0.2, 0.5, 0.0], 1.0);
        put(&mut sk, HumanoidBone::RightUpperArm, [-0.2, 0.5, 0.0], 0.02);
        assert!(compute_shoulder_align_rotation(&sk, &rest_world, &humanoid, 0.1).is_none());
    }
}

#[cfg(test)]
mod clavicle_tests {
    //! Regression tests for the `Tip::Clavicle` driving path. The
    //! shoulder humanoid bones (the clavicles) used to be absent from
    //! `DRIVEN_BONES` entirely, so a shrug never reached the avatar.
    //! These tests pin the contract: a raised source-side shoulder
    //! produces a non-identity rotation on that side's clavicle bone,
    //! and the rotation tilts the bone toward the raised direction.
    use super::*;
    use crate::asset::{NodeId, SkeletonAsset, SkeletonNode, Transform};
    use crate::tracking::source_skeleton::SourceJoint;
    use std::collections::HashMap;

    /// Build a tiny but topologically real upper-body rig:
    ///   Hips -> Spine -> {LeftShoulder -> LeftUpperArm -> LeftLowerArm,
    ///                     RightShoulder -> RightUpperArm -> RightLowerArm}
    /// Rest pose keeps shoulder line and clavicles flat along ±X so
    /// `rest_dir` for each clavicle equals the rest source-side
    /// direction (= ±X). That alignment lets us assert clean identity
    /// rotations on rest-pose inputs without a "rig vs source rest
    /// pose mismatch" baseline angle leaking in.
    fn build_upper_body_rig() -> (SkeletonAsset, HumanoidMap) {
        let bones: &[(HumanoidBone, Option<HumanoidBone>, [f32; 3])] = &[
            (HumanoidBone::Hips, None, [0.0, 0.9, 0.0]),
            (HumanoidBone::Spine, Some(HumanoidBone::Hips), [0.0, 0.5, 0.0]),
            (
                HumanoidBone::LeftShoulder,
                Some(HumanoidBone::Spine),
                [0.05, 0.0, 0.0],
            ),
            (
                HumanoidBone::LeftUpperArm,
                Some(HumanoidBone::LeftShoulder),
                [0.16, 0.0, 0.0],
            ),
            (
                HumanoidBone::LeftLowerArm,
                Some(HumanoidBone::LeftUpperArm),
                [0.29, 0.0, 0.0],
            ),
            (
                HumanoidBone::RightShoulder,
                Some(HumanoidBone::Spine),
                [-0.05, 0.0, 0.0],
            ),
            (
                HumanoidBone::RightUpperArm,
                Some(HumanoidBone::RightShoulder),
                [-0.16, 0.0, 0.0],
            ),
            (
                HumanoidBone::RightLowerArm,
                Some(HumanoidBone::RightUpperArm),
                [-0.29, 0.0, 0.0],
            ),
        ];

        let bone_to_idx: HashMap<HumanoidBone, usize> = bones
            .iter()
            .enumerate()
            .map(|(i, (b, _, _))| (*b, i))
            .collect();

        let mut nodes: Vec<SkeletonNode> = bones
            .iter()
            .enumerate()
            .map(|(i, (bone, _parent, translation))| SkeletonNode {
                id: NodeId(i as u64),
                name: format!("{bone:?}"),
                parent: None,
                children: Vec::new(),
                rest_local: Transform {
                    translation: *translation,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                },
                humanoid_bone: Some(*bone),
            })
            .collect();

        for (i, (_, parent, _)) in bones.iter().enumerate() {
            if let Some(parent_bone) = parent {
                let parent_idx = bone_to_idx[parent_bone];
                nodes[i].parent = Some(NodeId(parent_idx as u64));
                nodes[parent_idx].children.push(NodeId(i as u64));
            }
        }

        let skeleton = SkeletonAsset {
            nodes,
            root_nodes: vec![NodeId(0)],
            inverse_bind_matrices: Vec::new(),
        };
        let humanoid = HumanoidMap {
            bone_map: bone_to_idx
                .iter()
                .map(|(b, i)| (*b, NodeId(*i as u64)))
                .collect(),
        };
        (skeleton, humanoid)
    }

    /// Initial local_transforms = clone of rest_local, matching the
    /// post-`build_base_pose` precondition the solver expects.
    fn rest_local_transforms(skeleton: &SkeletonAsset) -> Vec<Transform> {
        skeleton.nodes.iter().map(|n| n.rest_local.clone()).collect()
    }

    fn put(sk: &mut SourceSkeleton, bone: HumanoidBone, pos: [f32; 3]) {
        sk.joints.insert(
            bone,
            SourceJoint {
                position: pos,
                confidence: 1.0,
                metric_depth_m: None,
            },
        );
    }

    fn quat_is_identity(q: &Quat) -> bool {
        q[0].abs() < 1e-4 && q[1].abs() < 1e-4 && q[2].abs() < 1e-4 && (q[3].abs() - 1.0).abs() < 1e-4
    }

    fn solve_with(source: &SourceSkeleton) -> (Vec<Transform>, HumanoidMap, SkeletonAsset) {
        let (skeleton, humanoid) = build_upper_body_rig();
        let mut local = rest_local_transforms(&skeleton);
        let params = SolverParams {
            // Snap directly — no smoothing — so the assertion measures
            // the solver's output, not blend lag.
            rotation_blend: 1.0,
            joint_confidence_threshold: 0.05,
            ..Default::default()
        };
        let mut state = PoseSolverState::new();
        solve_avatar_pose(source, &skeleton, Some(&humanoid), &mut local, &params, &mut state);
        (local, humanoid, skeleton)
    }

    /// Hands clasped in front must not throw the elbows behind the body.
    ///
    /// Live regression (2026-07-28): with the source elbow unobserved, the
    /// synthetic swivel pole pointed straight back, and the avatar's elbow
    /// solved 24 cm BEHIND its shoulder while the hand was in front of it —
    /// "手を前で組むと腕が後ろに回る". The measured direction of a real
    /// elbow (1311 observed shoulder/elbow/wrist triples) is outward and
    /// slightly toward the camera, never behind.
    #[test]
    fn synthetic_elbow_pole_bends_outward_and_forward_not_backward() {
        let span = 0.75;
        // Hands clasped in front of the chest: wrist near the midline,
        // forward of the shoulder (+z is toward the camera).
        for (side, shoulder) in [("left", [0.33, 0.0, 0.0]), ("right", [-0.33, 0.0, 0.0])] {
            let wrist = [0.02, -0.05, 0.20];
            let pole = synthetic_elbow_pole(&shoulder, &wrist, span);
            let mid = midpoint(&shoulder, &wrist);
            let off = vec3_sub(&pole, &mid);
            assert!(
                off[2] > 0.0,
                "{side}: pole must not sit behind the subject, z offset {}",
                off[2]
            );
            let outward = if shoulder[0] >= 0.0 { off[0] } else { -off[0] };
            assert!(outward > 0.0, "{side}: pole must lean outward, got {outward}");
            assert!(
                outward > off[2] && outward > off[1].abs(),
                "{side}: outward must dominate (measured 0.93 / 0.73)"
            );
            assert!(off[1] < 0.0, "{side}: elbows hang, not rise");
        }
    }

    fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    /// A hand target that vanishes must not teleport the avatar's arm.
    ///
    /// Live regression (2026-07-28): with the user's hands below frame, the
    /// arm's target switched between observation-driven and idle every few
    /// frames, and each switch moved the avatar's hand 0.4-0.7 m in ONE
    /// frame. The per-frame lerp cannot fix that — it just makes the step
    /// 70 % of the gap — so the arm chain carries a rotation-rate bound.
    #[test]
    fn losing_the_arm_target_eases_instead_of_teleporting() {
        let (skeleton, humanoid) = build_upper_body_rig();
        let mut local: Vec<Transform> =
            skeleton.nodes.iter().map(|n| n.rest_local.clone()).collect();
        // Snap blend deliberately: `dt_aware_blend` scales the production
        // 0.7 by wall-clock dt, and a test loop runs in microseconds, so a
        // "realistic" blend would measure the harness's clock rather than the
        // solver. At 1.0 the ONLY thing between a discontinuous target and a
        // one-frame teleport is the step cap — which is what this pins.
        let params = SolverParams { rotation_blend: 1.0, ..Default::default() };
        let mut state = PoseSolverState::new();

        let mut driven = SourceSkeleton::empty(0);
        put(&mut driven, HumanoidBone::LeftShoulder, [0.21, 0.60, 0.0]);
        put(&mut driven, HumanoidBone::RightShoulder, [-0.21, 0.60, 0.0]);
        // Left arm reaching forward and up — far from the idle A-pose.
        put(&mut driven, HumanoidBone::LeftUpperArm, [0.21, 0.60, 0.0]);
        put(&mut driven, HumanoidBone::LeftHand, [0.36, 0.80, 0.34]);
        put(&mut driven, HumanoidBone::LeftLowerArm, [0.30, 0.72, 0.28]);

        let hand_pos = |local: &Vec<Transform>| -> [f32; 3] {
            let idx = humanoid.bone_map[&HumanoidBone::LeftLowerArm].0 as usize;
            let world = compute_world_transforms(&skeleton, |i| local[i].clone());
            world[idx].position
        };

        // Settle on the driven pose.
        for _ in 0..30 {
            solve_avatar_pose(&driven, &skeleton, Some(&humanoid), &mut local, &params, &mut state);
        }
        let settled = hand_pos(&local);

        // The observation disappears: only the torso remains.
        let mut lost = SourceSkeleton::empty(0);
        put(&mut lost, HumanoidBone::LeftShoulder, [0.21, 0.60, 0.0]);
        put(&mut lost, HumanoidBone::RightShoulder, [-0.21, 0.60, 0.0]);

        let mut prev = settled;
        let mut max_step = 0.0_f32;
        let mut total = 0.0_f32;
        for _ in 0..30 {
            solve_avatar_pose(&lost, &skeleton, Some(&humanoid), &mut local, &params, &mut state);
            let now = hand_pos(&local);
            let step = dist3(prev, now);
            max_step = max_step.max(step);
            total += step;
            prev = now;
        }

        assert!(
            total > 0.05,
            "the arm must actually move to idle, travelled {total:.3} m"
        );
        // ARM_MAX_STEP_PER_FRAME (15 deg) over a ~0.3 m chain bounds a single
        // frame's travel well under the measured 0.4-0.7 m teleport.
        assert!(
            max_step < 0.10,
            "losing the target must ease, largest single-frame move {max_step:.3} m"
        );
    }

    /// Symmetric "rest" shoulders → clavicles must end at identity (or
    /// near-identity given quaternion arithmetic). This is the
    /// regression baseline: if the new path were buggy and emitted a
    /// rotation even for a rest pose, downstream rendering would
    /// twitch on every neutral frame.
    #[test]
    fn rest_shoulders_leave_clavicles_at_identity() {
        let mut sk = SourceSkeleton::empty(0);
        put(&mut sk, HumanoidBone::LeftShoulder, [0.21, 1.40, 0.0]);
        put(&mut sk, HumanoidBone::RightShoulder, [-0.21, 1.40, 0.0]);

        let (local, humanoid, _skel) = solve_with(&sk);

        let l_idx = humanoid.bone_map[&HumanoidBone::LeftShoulder].0 as usize;
        let r_idx = humanoid.bone_map[&HumanoidBone::RightShoulder].0 as usize;
        assert!(
            quat_is_identity(&local[l_idx].rotation),
            "rest pose: left clavicle should stay near identity, got {:?}",
            local[l_idx].rotation
        );
        assert!(
            quat_is_identity(&local[r_idx].rotation),
            "rest pose: right clavicle should stay near identity, got {:?}",
            local[r_idx].rotation
        );
    }

    /// Left shoulder raised 10 cm above the right (one-sided shrug):
    /// the left clavicle bone must rotate to tilt toward +Y. Before
    /// the `Tip::Clavicle` wiring this rotation was identity for any
    /// input — the bone wasn't in `DRIVEN_BONES`, so the avatar's
    /// shoulder line stayed flat regardless of what the tracker saw.
    #[test]
    fn left_shrug_lifts_left_clavicle_toward_plus_y() {
        let mut sk = SourceSkeleton::empty(0);
        put(&mut sk, HumanoidBone::LeftShoulder, [0.21, 1.50, 0.0]);
        put(&mut sk, HumanoidBone::RightShoulder, [-0.21, 1.40, 0.0]);

        let (local, humanoid, _skel) = solve_with(&sk);
        let l_idx = humanoid.bone_map[&HumanoidBone::LeftShoulder].0 as usize;
        let r_idx = humanoid.bone_map[&HumanoidBone::RightShoulder].0 as usize;

        assert!(
            !quat_is_identity(&local[l_idx].rotation),
            "shrug should produce a non-identity left-clavicle rotation, got {:?}",
            local[l_idx].rotation
        );

        // Apply the local rotation to the clavicle's rest direction
        // (= +X in this rig). The result must move *upward* in Y —
        // that's what makes the shoulder visibly rise on the avatar.
        let rest_dir = [1.0, 0.0, 0.0];
        let rotated = quat_rotate_vec3(&local[l_idx].rotation, &rest_dir);
        assert!(
            rotated[1] > 0.05,
            "left shrug should tilt left clavicle upward in Y; rotated={:?}",
            rotated
        );

        // The opposite side's clavicle moves the *other* way, because
        // the midpoint also lifts when only the left rises — this
        // matches what the tracker observes and what we want the
        // avatar to mimic.
        let r_rest_dir = [-1.0, 0.0, 0.0];
        let r_rotated = quat_rotate_vec3(&local[r_idx].rotation, &r_rest_dir);
        assert!(
            r_rotated[1] < -0.01,
            "left-only shrug should tilt right clavicle slightly down; rotated={:?}",
            r_rotated
        );
    }

    /// Bilateral shrug: both shoulders rise the same amount → midpoint
    /// rises too, so neither clavicle's direction relative to the
    /// midpoint changes. The clavicle bones therefore stay at
    /// identity. This is the "user shrugged both shoulders" case —
    /// in practice the avatar's overall body height (Hips translation)
    /// is what reflects this, not the clavicles.
    #[test]
    fn bilateral_shrug_leaves_clavicles_near_identity() {
        let mut sk = SourceSkeleton::empty(0);
        put(&mut sk, HumanoidBone::LeftShoulder, [0.21, 1.50, 0.0]);
        put(&mut sk, HumanoidBone::RightShoulder, [-0.21, 1.50, 0.0]);

        let (local, humanoid, _skel) = solve_with(&sk);
        let l_idx = humanoid.bone_map[&HumanoidBone::LeftShoulder].0 as usize;
        let r_idx = humanoid.bone_map[&HumanoidBone::RightShoulder].0 as usize;

        assert!(
            quat_is_identity(&local[l_idx].rotation),
            "bilateral shrug: left clavicle should stay near identity, got {:?}",
            local[l_idx].rotation
        );
        assert!(
            quat_is_identity(&local[r_idx].rotation),
            "bilateral shrug: right clavicle should stay near identity, got {:?}",
            local[r_idx].rotation
        );
    }
}

#[cfg(test)]
mod chin_chain_tests {
    //! Regression tests for the `Tip::HeadFromShoulders` driving path.
    //! Before this wiring, Neck/UpperChest/Chest were absent from
    //! `DRIVEN_BONES` and `source.joints` had no Head entry, so a
    //! chin-thrust (head poking forward toward the camera) produced
    //! no avatar response at all.
    use super::*;
    use crate::asset::{NodeId, SkeletonAsset, SkeletonNode, Transform};
    use crate::tracking::source_skeleton::SourceJoint;
    use std::collections::HashMap;

    /// Full upper-body rig with the spine chain populated:
    ///   Hips -> Spine -> Chest -> UpperChest -> {Neck -> Head,
    ///                                            LeftShoulder ->
    ///                                              LeftUpperArm,
    ///                                            RightShoulder ->
    ///                                              RightUpperArm}
    /// All rest translations are pure Y (or pure ±X for clavicles)
    /// so rest_dir for each chain bone is +Y — straight upright
    /// spine, no natural curvature. That keeps the chin-thrust math
    /// simple to reason about: any forward Z in the source direction
    /// must come out as a forward pitch on UpperChest (Neck residual
    /// goes near identity in this clean case because parent already
    /// absorbed the full bend).
    fn build_full_upper_body_rig() -> (SkeletonAsset, HumanoidMap) {
        let bones: &[(HumanoidBone, Option<HumanoidBone>, [f32; 3])] = &[
            (HumanoidBone::Hips, None, [0.0, 0.9, 0.0]),
            (HumanoidBone::Spine, Some(HumanoidBone::Hips), [0.0, 0.10, 0.0]),
            (HumanoidBone::Chest, Some(HumanoidBone::Spine), [0.0, 0.10, 0.0]),
            (
                HumanoidBone::UpperChest,
                Some(HumanoidBone::Chest),
                [0.0, 0.10, 0.0],
            ),
            (
                HumanoidBone::Neck,
                Some(HumanoidBone::UpperChest),
                [0.0, 0.10, 0.0],
            ),
            (HumanoidBone::Head, Some(HumanoidBone::Neck), [0.0, 0.10, 0.0]),
            (
                HumanoidBone::LeftShoulder,
                Some(HumanoidBone::UpperChest),
                [0.05, 0.0, 0.0],
            ),
            (
                HumanoidBone::LeftUpperArm,
                Some(HumanoidBone::LeftShoulder),
                [0.16, 0.0, 0.0],
            ),
            (
                HumanoidBone::RightShoulder,
                Some(HumanoidBone::UpperChest),
                [-0.05, 0.0, 0.0],
            ),
            (
                HumanoidBone::RightUpperArm,
                Some(HumanoidBone::RightShoulder),
                [-0.16, 0.0, 0.0],
            ),
        ];

        let bone_to_idx: HashMap<HumanoidBone, usize> = bones
            .iter()
            .enumerate()
            .map(|(i, (b, _, _))| (*b, i))
            .collect();

        let mut nodes: Vec<SkeletonNode> = bones
            .iter()
            .enumerate()
            .map(|(i, (bone, _parent, translation))| SkeletonNode {
                id: NodeId(i as u64),
                name: format!("{bone:?}"),
                parent: None,
                children: Vec::new(),
                rest_local: Transform {
                    translation: *translation,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                },
                humanoid_bone: Some(*bone),
            })
            .collect();

        for (i, (_, parent, _)) in bones.iter().enumerate() {
            if let Some(parent_bone) = parent {
                let parent_idx = bone_to_idx[parent_bone];
                nodes[i].parent = Some(NodeId(parent_idx as u64));
                nodes[parent_idx].children.push(NodeId(i as u64));
            }
        }

        let skeleton = SkeletonAsset {
            nodes,
            root_nodes: vec![NodeId(0)],
            inverse_bind_matrices: Vec::new(),
        };
        let humanoid = HumanoidMap {
            bone_map: bone_to_idx
                .iter()
                .map(|(b, i)| (*b, NodeId(*i as u64)))
                .collect(),
        };
        (skeleton, humanoid)
    }

    fn put(sk: &mut SourceSkeleton, bone: HumanoidBone, pos: [f32; 3]) {
        sk.joints.insert(
            bone,
            SourceJoint {
                position: pos,
                confidence: 1.0,
                metric_depth_m: None,
            },
        );
    }

    /// Mirror `inject_spine_chain_proxies` from the source builders
    /// so the test source carries the same dual-inserts the real
    /// pipeline emits. Without these the solver's
    /// `source.joints.get(&bone)` lookup fails for UpperChest / Neck
    /// and the new driving path is silently skipped.
    fn inject_chain_proxies(sk: &mut SourceSkeleton) {
        let l = *sk
            .joints
            .get(&HumanoidBone::LeftShoulder)
            .expect("test source must include LeftShoulder");
        let r = *sk
            .joints
            .get(&HumanoidBone::RightShoulder)
            .expect("test source must include RightShoulder");
        let mid = SourceJoint {
            position: [
                (l.position[0] + r.position[0]) * 0.5,
                (l.position[1] + r.position[1]) * 0.5,
                (l.position[2] + r.position[2]) * 0.5,
            ],
            confidence: l.confidence.min(r.confidence),
            metric_depth_m: None,
        };
        sk.joints.insert(HumanoidBone::UpperChest, mid);
        sk.joints.insert(HumanoidBone::Neck, mid);
    }

    /// Like clavicle_tests::quat_is_identity but with a slightly
    /// looser tolerance — the chin-chain bones inherit a 1€ filter
    /// pass + parent-rotation rebase that round-trip through enough
    /// quaternion arithmetic to push the absolute error past 1e-4.
    fn quat_is_near_identity(q: &Quat) -> bool {
        q[0].abs() < 1e-3 && q[1].abs() < 1e-3 && q[2].abs() < 1e-3 && (q[3].abs() - 1.0).abs() < 1e-3
    }

    fn solve_with(source: &SourceSkeleton) -> (Vec<Transform>, HumanoidMap, SkeletonAsset) {
        let (skeleton, humanoid) = build_full_upper_body_rig();
        let mut local: Vec<Transform> =
            skeleton.nodes.iter().map(|n| n.rest_local.clone()).collect();
        let params = SolverParams {
            rotation_blend: 1.0,
            joint_confidence_threshold: 0.05,
            ..Default::default()
        };
        let mut state = PoseSolverState::new();
        solve_avatar_pose(source, &skeleton, Some(&humanoid), &mut local, &params, &mut state);
        (local, humanoid, skeleton)
    }

    /// Head proxy directly above the shoulder midpoint = no chin
    /// thrust. UpperChest and Neck must stay at rest. This is the
    /// regression baseline — if the path were buggy, every neutral
    /// frame would inject a small spurious rotation.
    #[test]
    fn rest_head_above_shoulders_leaves_chain_at_identity() {
        let mut sk = SourceSkeleton::empty(0);
        put(&mut sk, HumanoidBone::LeftShoulder, [0.21, 0.60, 0.0]);
        put(&mut sk, HumanoidBone::RightShoulder, [-0.21, 0.60, 0.0]);
        // ShoulderMid = (0, 0.60, 0); Head above it by 0.30 (no Z) —
        // matches the rest-pose chain (all +Y) so source_dir == rest_dir.
        put(&mut sk, HumanoidBone::Head, [0.0, 0.90, 0.0]);
        inject_chain_proxies(&mut sk);

        let (local, humanoid, _skel) = solve_with(&sk);
        let uc_idx = humanoid.bone_map[&HumanoidBone::UpperChest].0 as usize;
        let n_idx = humanoid.bone_map[&HumanoidBone::Neck].0 as usize;

        assert!(
            quat_is_near_identity(&local[uc_idx].rotation),
            "rest head: UpperChest should stay near identity, got {:?}",
            local[uc_idx].rotation
        );
        assert!(
            quat_is_near_identity(&local[n_idx].rotation),
            "rest head: Neck should stay near identity, got {:?}",
            local[n_idx].rotation
        );
    }

    /// Chin thrust: head moved 10 cm forward (+Z toward camera) while
    /// shoulders stay put. UpperChest must pitch forward enough to
    /// move the Head bone's world position toward +Z. Before this
    /// commit the entire chain was at rest pose and the avatar's
    /// head pivot stayed motionless regardless of how far the
    /// subject pushed their face out — this test pins the new path.
    #[test]
    fn chin_thrust_pitches_chain_forward() {
        let mut sk = SourceSkeleton::empty(0);
        put(&mut sk, HumanoidBone::LeftShoulder, [0.21, 0.60, 0.0]);
        put(&mut sk, HumanoidBone::RightShoulder, [-0.21, 0.60, 0.0]);
        // Head pushed 10 cm forward of where it sits at rest.
        put(&mut sk, HumanoidBone::Head, [0.0, 0.90, 0.10]);
        inject_chain_proxies(&mut sk);

        let (local, humanoid, skeleton) = solve_with(&sk);
        let uc_idx = humanoid.bone_map[&HumanoidBone::UpperChest].0 as usize;

        assert!(
            !quat_is_near_identity(&local[uc_idx].rotation),
            "chin thrust: UpperChest must rotate, got {:?}",
            local[uc_idx].rotation
        );

        // The avatar's Head bone world position should have moved
        // forward in Z. Recompute world transforms after the solve to
        // confirm — the UpperChest rotation must propagate up the
        // chain to Neck and Head positions via FK.
        let head_idx = humanoid.bone_map[&HumanoidBone::Head].0 as usize;
        let world = compute_world_transforms(&skeleton, |i| local[i].clone());
        let head_z = world[head_idx].position[2];
        assert!(
            head_z > 0.04,
            "chin thrust: Head world Z must move forward (>4 cm), got {head_z}"
        );

        // Sanity: the Head should still be roughly at its rest height.
        // A pitch around X drops the Y a little (cos(θ) factor); 10 cm
        // forward over a 30 cm chain gives θ ≈ 19°, cos(19°) ≈ 0.946,
        // so Y drops by ~5%. Allow generous slack for FK distribution.
        let head_y = world[head_idx].position[1];
        assert!(
            head_y > 1.10 && head_y < 1.50,
            "chin thrust: Head world Y should stay near rest height (~1.20), got {head_y}"
        );
    }

    /// Lateral head shift (subject leans head left): the chain must
    /// roll, not pitch. Verifies that the direction-matching is fully
    /// 3D — pitching for Z, rolling for X — and that we didn't
    /// accidentally hard-code a chin-thrust-only axis.
    #[test]
    fn head_lean_left_rolls_chain_in_x() {
        let mut sk = SourceSkeleton::empty(0);
        put(&mut sk, HumanoidBone::LeftShoulder, [0.21, 0.60, 0.0]);
        put(&mut sk, HumanoidBone::RightShoulder, [-0.21, 0.60, 0.0]);
        // Head shifted +0.10 in X (subject's anatomical right per
        // selfie-mirror — not important for the test, only the
        // direction matters).
        put(&mut sk, HumanoidBone::Head, [0.10, 0.90, 0.0]);
        inject_chain_proxies(&mut sk);

        let (local, humanoid, skeleton) = solve_with(&sk);
        let head_idx = humanoid.bone_map[&HumanoidBone::Head].0 as usize;
        let world = compute_world_transforms(&skeleton, |i| local[i].clone());

        let head_x = world[head_idx].position[0];
        let head_z = world[head_idx].position[2];
        assert!(
            head_x > 0.04,
            "head lean: Head world X must move toward +X (>4 cm), got {head_x}"
        );
        assert!(
            head_z.abs() < 0.02,
            "head lean: Head world Z must stay near zero (no spurious pitch), got {head_z}"
        );
    }

    /// Solve the same chin-thrust source with a face pose attached and
    /// report the Head bone's world forward (+Z) elevation in degrees —
    /// positive = looking up, negative = looking down.
    fn head_forward_elevation_deg(head_forward_z: f32, face_pitch: f32) -> f32 {
        let mut sk = SourceSkeleton::empty(0);
        put(&mut sk, HumanoidBone::LeftShoulder, [0.21, 0.60, 0.0]);
        put(&mut sk, HumanoidBone::RightShoulder, [-0.21, 0.60, 0.0]);
        put(&mut sk, HumanoidBone::Head, [0.0, 0.90, head_forward_z]);
        inject_chain_proxies(&mut sk);
        sk.face = Some(FacePose {
            yaw: 0.0,
            pitch: face_pitch,
            roll: 0.0,
            confidence: 1.0,
            ..Default::default()
        });

        let (local, humanoid, skeleton) = solve_with(&sk);
        let head_idx = humanoid.bone_map[&HumanoidBone::Head].0 as usize;
        let world = compute_world_transforms(&skeleton, |i| local[i].clone());
        let fwd = quat_rotate_vec3(&world[head_idx].rotation, &[0.0, 0.0, 1.0]);
        fwd[1].clamp(-1.0, 1.0).asin().to_degrees()
    }

    /// The face track owns gaze. A forward-head posture bows the neck
    /// chain, and that bow must NOT also tilt where the head looks.
    ///
    /// Live regression (2026-07-27): the depth builder's source Head sits
    /// a steady +0.20 shoulder-spans in front of the shoulder midpoint
    /// (measured over 1123 frames), so `Tip::HeadFromShoulders` bows the
    /// chain ~21°. While that bow was composed with the face pose, the
    /// avatar looked 11° DOWN on a face track reporting 5° UP — the user's
    /// "I'm looking above the horizon and the avatar is looking down".
    #[test]
    fn neck_bow_does_not_tilt_gaze() {
        // 0.084 forward over a 0.30 chain ≈ the measured 0.20·span bow.
        let bowed = head_forward_elevation_deg(0.084, 0.0);
        assert!(
            bowed.abs() < 1.0,
            "a bowed neck must leave gaze level, got {bowed}°"
        );
        let upright = head_forward_elevation_deg(0.0, 0.0);
        assert!(
            (bowed - upright).abs() < 1.0,
            "gaze must not depend on the neck bow: bowed {bowed}° vs upright {upright}°"
        );
    }

    /// ...and the face pose itself still drives gaze one-for-one, with
    /// `+pitch` = chin down = looking down.
    #[test]
    fn face_pitch_drives_gaze_one_for_one() {
        let up = head_forward_elevation_deg(0.084, -10.0_f32.to_radians());
        let down = head_forward_elevation_deg(0.084, 10.0_f32.to_radians());
        assert!(
            (up - 10.0).abs() < 1.0,
            "10° chin-up must read as 10° of upward gaze, got {up}°"
        );
        assert!(
            (down + 10.0).abs() < 1.0,
            "10° chin-down must read as 10° of downward gaze, got {down}°"
        );
    }
}

/// The render loop solves at display rate while the camera captures at
/// 30 fps — these tests pin down that the *measurement* side of the
/// solver (1€ filters, confidence gates) advances once per distinct
/// camera sample, so the solved pose can't depend on the viewer's
/// monitor refresh rate.
#[cfg(test)]
mod sample_dedup_tests {
    use super::*;
    use crate::asset::{HumanoidBone, HumanoidMap, NodeId, SkeletonAsset, SkeletonNode};
    use crate::tracking::source_skeleton::SourceJoint;
    use std::collections::HashMap;

    fn build_arm_rig() -> (SkeletonAsset, HumanoidMap) {
        let bones: &[(HumanoidBone, Option<HumanoidBone>, [f32; 3])] = &[
            (HumanoidBone::Hips, None, [0.0, 0.9, 0.0]),
            (HumanoidBone::Spine, Some(HumanoidBone::Hips), [0.0, 0.5, 0.0]),
            (
                HumanoidBone::LeftUpperArm,
                Some(HumanoidBone::Spine),
                [0.2, 0.0, 0.0],
            ),
            (
                HumanoidBone::LeftLowerArm,
                Some(HumanoidBone::LeftUpperArm),
                [0.29, 0.0, 0.0],
            ),
        ];
        let bone_to_idx: HashMap<HumanoidBone, usize> = bones
            .iter()
            .enumerate()
            .map(|(i, (b, _, _))| (*b, i))
            .collect();
        let mut nodes: Vec<SkeletonNode> = bones
            .iter()
            .enumerate()
            .map(|(i, (bone, _, translation))| SkeletonNode {
                id: NodeId(i as u64),
                name: format!("{bone:?}"),
                parent: None,
                children: Vec::new(),
                rest_local: Transform {
                    translation: *translation,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                },
                humanoid_bone: Some(*bone),
            })
            .collect();
        for (i, (_, parent, _)) in bones.iter().enumerate() {
            if let Some(parent_bone) = parent {
                let parent_idx = bone_to_idx[parent_bone];
                nodes[i].parent = Some(NodeId(parent_idx as u64));
                nodes[parent_idx].children.push(NodeId(i as u64));
            }
        }
        let skeleton = SkeletonAsset {
            nodes,
            root_nodes: vec![NodeId(0)],
            inverse_bind_matrices: Vec::new(),
        };
        let humanoid = HumanoidMap {
            bone_map: bone_to_idx
                .iter()
                .map(|(b, i)| (*b, NodeId(*i as u64)))
                .collect(),
        };
        (skeleton, humanoid)
    }

    /// A moving-elbow sample stamped with a device capture time, as the
    /// real D435 path publishes it.
    fn camera_sample(step: u64, elbow_y: f32) -> SourceSkeleton {
        let mut sk = SourceSkeleton::empty(step);
        sk.capture_timestamp_ms = Some(step as f64 * 33.3);
        for (bone, pos) in [
            (HumanoidBone::LeftUpperArm, [0.2, 1.4, 0.0]),
            (HumanoidBone::LeftLowerArm, [0.45, elbow_y, 0.0]),
        ] {
            sk.joints.insert(
                bone,
                SourceJoint {
                    position: pos,
                    confidence: 1.0,
                    metric_depth_m: None,
                },
            );
        }
        sk.overall_confidence = 1.0;
        sk
    }

    fn params() -> SolverParams {
        SolverParams {
            rotation_blend: 1.0,
            joint_confidence_threshold: 0.05,
            // Keep the untracked-arm idler out of the way: these tests
            // assert what the *tracking* path does and does not drive.
            idle_arm_apose_enabled: false,
            ..Default::default()
        }
    }

    /// Solving the same capture sequence once-per-sample (30 Hz render)
    /// and twice-per-sample (60 Hz render) must land on the same pose:
    /// the measurement filters advance per *capture*, not per solve
    /// call. Pre-fix, every repeat pushed the identical raw value
    /// through the 1€ filters again, so the result depended on how many
    /// render frames each camera frame was displayed for.
    #[test]
    fn double_rate_resolve_matches_single_rate() {
        let (skeleton, humanoid) = build_arm_rig();
        let samples: Vec<SourceSkeleton> = (0..8)
            .map(|i| camera_sample(i, 1.4 - 0.05 * i as f32))
            .collect();

        let run = |repeats: usize| -> Vec<Transform> {
            let mut state = PoseSolverState::new();
            let mut local: Vec<Transform> = skeleton
                .nodes
                .iter()
                .map(|n| n.rest_local.clone())
                .collect();
            for s in &samples {
                for _ in 0..repeats {
                    // Rebuild base pose each render frame, as run_frame does.
                    for (t, n) in local.iter_mut().zip(skeleton.nodes.iter()) {
                        *t = n.rest_local.clone();
                    }
                    solve_avatar_pose(s, &skeleton, Some(&humanoid), &mut local, &params(), &mut state);
                }
            }
            local
        };

        let once = run(1);
        let twice = run(2);
        for (idx, (a, b)) in once.iter().zip(twice.iter()).enumerate() {
            for c in 0..4 {
                assert!(
                    (a.rotation[c] - b.rotation[c]).abs() < 1e-4,
                    "node {idx}: solving each camera sample twice (60 Hz render) must not \
                     change the pose vs once (30 Hz render); rotation component {c} \
                     differs: {} vs {}",
                    a.rotation[c],
                    b.rotation[c],
                );
            }
        }
    }

    /// The hold/fade policy re-publishes the SAME capture with decayed
    /// confidences. The cached-geometry fast path must still let that
    /// decay reach the confidence gates — a fully decayed repeat leaves
    /// the arm at rest even though the cache holds a bent-arm pose.
    #[test]
    fn confidence_decay_on_repeated_sample_still_gates_joints() {
        let (skeleton, humanoid) = build_arm_rig();
        let mut state = PoseSolverState::new();
        let rest: Vec<Transform> = skeleton
            .nodes
            .iter()
            .map(|n| n.rest_local.clone())
            .collect();

        // Fresh solve with a clearly bent arm.
        let bent = camera_sample(0, 1.0);
        let mut local = rest.clone();
        solve_avatar_pose(&bent, &skeleton, Some(&humanoid), &mut local, &params(), &mut state);
        let upper_idx = humanoid.bone_map[&HumanoidBone::LeftUpperArm].0 as usize;
        let bent_rot = local[upper_idx].rotation;
        assert!(
            bent_rot[2].abs() > 0.01 || bent_rot[0].abs() > 0.01 || bent_rot[1].abs() > 0.01,
            "precondition: bent arm must rotate the upper arm, got {bent_rot:?}"
        );

        // Same capture, confidences decayed to zero (hold window expiry).
        let mut decayed = bent.clone();
        decayed.scale_confidence(0.0);
        let mut local = rest.clone();
        solve_avatar_pose(&decayed, &skeleton, Some(&humanoid), &mut local, &params(), &mut state);
        let r = local[upper_idx].rotation;
        assert!(
            r[0].abs() < 1e-5 && r[1].abs() < 1e-5 && r[2].abs() < 1e-5,
            "fully decayed repeat must leave the arm at rest (gated off), got {r:?}"
        );
    }
}
