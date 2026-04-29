//! Solve avatar bone rotations from a [`SourceSkeleton`].
//!
//! The solver is driven by four ideas:
//!
//! 1. **Tree-walked 2D→3D lift.** Per-bone direction matching alone leaks
//!    detector noise across the kinematic chain — each child reconstructs
//!    its own Z independently and the implied position diverges from
//!    `parent_3d + bone_3d`. Instead we lift every joint into 3D *once
//!    per frame* in tree order, anchoring each child's depth to its
//!    parent's depth (`child_z = parent_z + ±√(L² − d²)`, sign chosen for
//!    temporal continuity). A bone's source direction is then
//!    `child_3d − parent_3d`, which is by construction consistent with the
//!    parent's already-resolved 3D position.
//!
//! 2. **Rest-relative.** Every output rotation is expressed as a delta from
//!    the avatar's rest pose, so models whose bones have non-identity rest
//!    rotations (most real VRMs) are not destroyed by the write.
//!
//! 3. **Direction matching.** For chain bones (upper/lower arm, leg, spine)
//!    we read the bone's *rest world direction* (from its world position to
//!    its tip's world position) and rotate it to match the source skeleton's
//!    observed direction in camera space. The minimum-arc quaternion is
//!    written as the world-space delta; it is then conjugated into the
//!    bone's parent-local frame so the result respects the skeleton
//!    hierarchy.
//!
//! 4. **Face pose is independent.** The head rotation comes from facial
//!    keypoints (yaw/pitch/roll), not from the nose-to-shoulder vector.
//!    It is applied as a rest-relative local delta on the Head bone.
//!
//! The solver assumes `local_transforms` has already been reset to the
//! skeleton's rest pose (via `AvatarInstance::build_base_pose`). It only
//! writes rotations; translations and scales are left untouched.

use std::collections::HashMap;
use std::time::Instant;

use crate::asset::{HumanoidBone, HumanoidMap, NodeId, Quat, SkeletonAsset, Transform, Vec3};
use crate::math_utils::{
    quat_conjugate, quat_from_euler_ypr, quat_from_vectors, quat_mul, quat_normalize,
    quat_rotate_vec3, vec3_normalize, vec3_sub,
};
use crate::tracking::source_skeleton::{FacePose, SourceSkeleton};

/// Reference frame period used to convert the GUI's frame-rate-naive
/// `rotation_blend` slider into a time-constant. With the slider at
/// `b` and the system actually running at `dt_ref`, the per-frame α
/// becomes `1 − exp(−dt/τ)` with `τ = −dt_ref / ln(1 − b)`. The
/// slider semantics stay intact — `b` reads as "what α you'd see at
/// the reference rate" — but the actual smoothing is now invariant
/// to frame-rate jitter.
const ROTATION_BLEND_REFERENCE_DT: f32 = 1.0 / 30.0;

/// Time constant (seconds) for exponential decay of per-bone running
/// max calibration. Spike-resistance: a single large outlier max
/// loses 1/e of its lead in this many seconds while a sustained real
/// peak gets refreshed every frame and never decays. Tuned so a
/// minute-long pose gap (user steps away) still has most of the
/// calibration intact when they return.
const RUNNING_MAX_DECAY_TAU_SEC: f32 = 60.0;

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
}

impl Default for SolverParams {
    fn default() -> Self {
        Self {
            rotation_blend: 0.7,
            joint_confidence_threshold: crate::tracking::DEFAULT_CONFIDENCE_THRESHOLD,
            face_confidence_threshold: crate::tracking::DEFAULT_CONFIDENCE_THRESHOLD,
        }
    }
}

/// Per-bone calibration state carried between solver frames so we can
/// recover Z (depth toward/away from camera) from 2D foreshortening.
///
/// The fundamental problem: a single 2D view does not reveal depth, so
/// when the subject extends an arm forward the 2D shoulder→wrist vector
/// becomes very short and the solver — without this state — collapses
/// the avatar's arm to lie in the image plane regardless of the true
/// pose. Recovering Z requires knowing the bone's *un-foreshortened* 2D
/// length so we can compute `|Z| = sqrt(L_full² − L_observed²)`.
///
/// `L_full` is learned per-bone as a running max of observed 2D length
/// on that bone. The very first observation seeds the calibration to
/// itself, so the initial frame has Z = 0 (matching the previous
/// stateless behaviour); as the subject moves and at some point fully
/// extends the limb in the image plane the max grows and unlocks Z
/// recovery for subsequent foreshortened frames.
///
/// **Why running max and not avatar bone length:** an anime/stylised
/// avatar's arm-to-shoulder ratio is usually different from a human's,
/// so calibrating against `rest_world` distances systematically
/// over-shoots `expected_2d > observed_2d` and folds limbs forward
/// even at T-pose. Calibrating against the *subject's own* observed
/// extents is invariant to avatar choice.
///
/// **Single-shot callers** (tests, diagnose CLI) accumulate no history
/// and so always run with Z = 0 unless they explicitly prime the state
/// with a calibration frame first (see `diagnose_pose --prime`).
#[derive(Clone, Debug, Default)]
pub struct PoseSolverState {
    /// Maximum 2D length (in source image-space units) ever observed for
    /// each bone. Initialized empty; entries appear lazily as bones are
    /// solved.
    bone_max_2d: HashMap<HumanoidBone, f32>,
    /// Maximum 2D shoulder span (||LeftShoulder − RightShoulder||) ever
    /// observed. Used to detect side / rear views: when the current
    /// span is much smaller than the max the body has yawed away from
    /// the camera and face-pose keypoints are unreliable.
    shoulder_span_max_2d: f32,
    /// Per-bone 1€ filter state for joint position smoothing.
    joint_filters: HashMap<HumanoidBone, OneEuroFilterState>,
    /// Separate filter state map for fingertip auxiliary positions
    /// (keyed by the distal bone, same as `SourceSkeleton::fingertips`).
    fingertip_filters: HashMap<HumanoidBone, OneEuroFilterState>,
    /// Per-bone Schmitt hysteresis state for joint confidence.
    joint_active: HashMap<HumanoidBone, bool>,
    /// Last frame's signed Z offset along each bone in the 2D→3D lift,
    /// keyed by the *parent* of the lift step (which is the bone being
    /// driven by that step). Used to pick the sign of the next frame's
    /// Z reconstruction for chain continuity: sqrt() always yields a
    /// non-negative magnitude, but the limb may be in front of or
    /// behind its parent. We pick whichever sign keeps the offset
    /// closest to last frame's value, so a limb that is "forward" stays
    /// forward across detector noise rather than flipping back and
    /// forth between ±Z.
    bone_prev_z_offset: HashMap<HumanoidBone, f32>,
    /// Same idea for fingertips, keyed by the distal bone whose tip
    /// they represent (matches `SourceSkeleton::fingertips`).
    fingertip_prev_z_offset: HashMap<HumanoidBone, f32>,
    /// Wall-clock timestamp of the previous solve, used to derive `dt`
    /// for the dt-aware blend / decay / 1€ filter. `None` until the
    /// first call.
    last_solve_instant: Option<Instant>,
}

impl PoseSolverState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all per-bone calibration. Call when the subject changes
    /// (different person, or large jump after a tracking dropout) so
    /// stale max values do not pin Z high for a small new subject.
    pub fn reset(&mut self) {
        self.bone_max_2d.clear();
        self.shoulder_span_max_2d = 0.0;
        self.joint_filters.clear();
        self.fingertip_filters.clear();
        self.joint_active.clear();
        self.bone_prev_z_offset.clear();
        self.fingertip_prev_z_offset.clear();
        self.last_solve_instant = None;
    }

    /// Discard the per-frame motion smoothing state (1€ filter,
    /// hysteresis active flags, last-solve timestamp, sign continuity
    /// of the lift's Z offsets) without touching the longer-running
    /// calibration (`bone_max_2d`, `shoulder_span_max_2d`). Useful
    /// when the next solve call represents a new subject pose that
    /// should not blend with whatever was last seen — e.g. between a
    /// calibration-priming frame and the actual test frame in
    /// `diagnose_pose`.
    pub fn reset_motion_smoothing(&mut self) {
        self.joint_filters.clear();
        self.fingertip_filters.clear();
        self.joint_active.clear();
        self.bone_prev_z_offset.clear();
        self.fingertip_prev_z_offset.clear();
        self.last_solve_instant = None;
    }
}

/// 1€ filter state for a single 2D keypoint stream. See
/// [Casiez et al. 2012](https://hal.inria.fr/hal-00670496).
#[derive(Clone, Copy, Debug, Default)]
struct OneEuroFilterState {
    initialized: bool,
    pos: [f32; 2],
    vel: [f32; 2],
}

impl OneEuroFilterState {
    /// Apply the filter to a raw position and return the smoothed
    /// position. `dt` is seconds since the previous update for this
    /// stream. Self-initializes on the first call (no smoothing).
    fn apply(&mut self, raw: [f32; 2], dt: f32) -> [f32; 2] {
        if !self.initialized || dt <= 0.0 {
            self.initialized = true;
            self.pos = raw;
            self.vel = [0.0, 0.0];
            return raw;
        }
        // Velocity from raw delta, then low-pass it via d_cutoff.
        let raw_vel = [(raw[0] - self.pos[0]) / dt, (raw[1] - self.pos[1]) / dt];
        let alpha_d = one_euro_alpha(dt, ONE_EURO_D_CUTOFF_HZ);
        self.vel[0] += alpha_d * (raw_vel[0] - self.vel[0]);
        self.vel[1] += alpha_d * (raw_vel[1] - self.vel[1]);
        // Position cutoff scales with velocity magnitude — high-speed
        // motion gets a higher cutoff (less smoothing, more responsive).
        let speed = (self.vel[0] * self.vel[0] + self.vel[1] * self.vel[1]).sqrt();
        let cutoff = ONE_EURO_MIN_CUTOFF_HZ + ONE_EURO_BETA * speed;
        let alpha = one_euro_alpha(dt, cutoff);
        self.pos[0] += alpha * (raw[0] - self.pos[0]);
        self.pos[1] += alpha * (raw[1] - self.pos[1]);
        self.pos
    }
}

#[inline]
fn one_euro_alpha(dt: f32, cutoff_hz: f32) -> f32 {
    let tau = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz.max(1e-6));
    let raw_alpha = 1.0 / (1.0 + tau / dt.max(1e-6));
    raw_alpha.clamp(0.0, 1.0)
}

/// Body-yaw foreshortening fraction. `1.0` means shoulders fully visible
/// (facing camera); `0.0` means side profile. Below this threshold the
/// face-pose detector is unreliable (only one ear/eye visible) and we
/// suppress face yaw/pitch/roll. Tuned so 45° three-quarter views still
/// pass.
const FACE_GATE_SHOULDER_RATIO: f32 = 0.45;

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
}

/// Static description of which bones the solver will try to drive and
/// what defines their direction. Ordered parent-first so the forward
/// kinematics pass always sees the parent's updated world rotation before
/// touching the child.
const DRIVEN_BONES: &[(HumanoidBone, Tip)] = &[
    (HumanoidBone::Spine, Tip::ShoulderMidpoint),
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
    // Feet — driven from COCO big-toe keypoints stashed in
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

/// Result of the per-frame 2D→3D lift. Each joint's 3D position is
/// computed in tree order so a child's depth is anchored to its
/// parent's depth: `child_3d.z = parent_3d.z + ±√(L² − d²)`. The sign
/// is chosen for temporal continuity (closest to last frame's offset),
/// so the limb does not flip across the image plane on detector noise.
///
/// `joints_3d` covers the body skeleton (Hips, shoulders, arms, legs,
/// fingers). `fingertips_3d` covers the auxiliary fingertip / toe tips
/// keyed by their distal bone (matches `SourceSkeleton::fingertips`).
/// `shoulder_mid_3d` is the lifted shoulder midpoint, used as the tip
/// of the Spine bone.
#[derive(Default)]
struct LiftedSkeleton {
    joints_3d: HashMap<HumanoidBone, [f32; 3]>,
    fingertips_3d: HashMap<HumanoidBone, [f32; 3]>,
    shoulder_mid_3d: Option<[f32; 3]>,
}

/// Tree-walk order for the lift. Each entry is `(parent, child,
/// calib_key)` — the calibration key is usually the bone being driven
/// (i.e. the parent for body chains, the distal bone for finger
/// chains), and is used to look up the running max 2D length in
/// `state.bone_max_2d`.
///
/// Order matters: a child must come after its parent so the parent's
/// 3D position is already filled in when we reach the child. We do not
/// include the shoulders themselves here — they are handled inline
/// because they share the shoulder midpoint's Z by construction (the
/// shoulder line is roughly horizontal in body frame).
const LIFT_CHAINS: &[(HumanoidBone, HumanoidBone, HumanoidBone)] = &[
    // Arms: parent is the shoulder/upper-arm joint already placed by
    // the shoulder-mid step. Calibration key is the bone being driven
    // (UpperArm direction comes from UpperArm→LowerArm length, etc.).
    (
        HumanoidBone::LeftUpperArm,
        HumanoidBone::LeftLowerArm,
        HumanoidBone::LeftUpperArm,
    ),
    (
        HumanoidBone::LeftLowerArm,
        HumanoidBone::LeftHand,
        HumanoidBone::LeftLowerArm,
    ),
    (
        HumanoidBone::RightUpperArm,
        HumanoidBone::RightLowerArm,
        HumanoidBone::RightUpperArm,
    ),
    (
        HumanoidBone::RightLowerArm,
        HumanoidBone::RightHand,
        HumanoidBone::RightLowerArm,
    ),
    // Legs.
    (
        HumanoidBone::Hips,
        HumanoidBone::LeftUpperLeg,
        HumanoidBone::LeftUpperLeg,
    ),
    (
        HumanoidBone::LeftUpperLeg,
        HumanoidBone::LeftLowerLeg,
        HumanoidBone::LeftUpperLeg,
    ),
    (
        HumanoidBone::LeftLowerLeg,
        HumanoidBone::LeftFoot,
        HumanoidBone::LeftLowerLeg,
    ),
    (
        HumanoidBone::Hips,
        HumanoidBone::RightUpperLeg,
        HumanoidBone::RightUpperLeg,
    ),
    (
        HumanoidBone::RightUpperLeg,
        HumanoidBone::RightLowerLeg,
        HumanoidBone::RightUpperLeg,
    ),
    (
        HumanoidBone::RightLowerLeg,
        HumanoidBone::RightFoot,
        HumanoidBone::RightLowerLeg,
    ),
    // Fingers (left).
    (
        HumanoidBone::LeftHand,
        HumanoidBone::LeftThumbProximal,
        HumanoidBone::LeftHand,
    ),
    (
        HumanoidBone::LeftThumbProximal,
        HumanoidBone::LeftThumbIntermediate,
        HumanoidBone::LeftThumbProximal,
    ),
    (
        HumanoidBone::LeftThumbIntermediate,
        HumanoidBone::LeftThumbDistal,
        HumanoidBone::LeftThumbIntermediate,
    ),
    (
        HumanoidBone::LeftHand,
        HumanoidBone::LeftIndexProximal,
        HumanoidBone::LeftHand,
    ),
    (
        HumanoidBone::LeftIndexProximal,
        HumanoidBone::LeftIndexIntermediate,
        HumanoidBone::LeftIndexProximal,
    ),
    (
        HumanoidBone::LeftIndexIntermediate,
        HumanoidBone::LeftIndexDistal,
        HumanoidBone::LeftIndexIntermediate,
    ),
    (
        HumanoidBone::LeftHand,
        HumanoidBone::LeftMiddleProximal,
        HumanoidBone::LeftHand,
    ),
    (
        HumanoidBone::LeftMiddleProximal,
        HumanoidBone::LeftMiddleIntermediate,
        HumanoidBone::LeftMiddleProximal,
    ),
    (
        HumanoidBone::LeftMiddleIntermediate,
        HumanoidBone::LeftMiddleDistal,
        HumanoidBone::LeftMiddleIntermediate,
    ),
    (
        HumanoidBone::LeftHand,
        HumanoidBone::LeftRingProximal,
        HumanoidBone::LeftHand,
    ),
    (
        HumanoidBone::LeftRingProximal,
        HumanoidBone::LeftRingIntermediate,
        HumanoidBone::LeftRingProximal,
    ),
    (
        HumanoidBone::LeftRingIntermediate,
        HumanoidBone::LeftRingDistal,
        HumanoidBone::LeftRingIntermediate,
    ),
    (
        HumanoidBone::LeftHand,
        HumanoidBone::LeftLittleProximal,
        HumanoidBone::LeftHand,
    ),
    (
        HumanoidBone::LeftLittleProximal,
        HumanoidBone::LeftLittleIntermediate,
        HumanoidBone::LeftLittleProximal,
    ),
    (
        HumanoidBone::LeftLittleIntermediate,
        HumanoidBone::LeftLittleDistal,
        HumanoidBone::LeftLittleIntermediate,
    ),
    // Fingers (right).
    (
        HumanoidBone::RightHand,
        HumanoidBone::RightThumbProximal,
        HumanoidBone::RightHand,
    ),
    (
        HumanoidBone::RightThumbProximal,
        HumanoidBone::RightThumbIntermediate,
        HumanoidBone::RightThumbProximal,
    ),
    (
        HumanoidBone::RightThumbIntermediate,
        HumanoidBone::RightThumbDistal,
        HumanoidBone::RightThumbIntermediate,
    ),
    (
        HumanoidBone::RightHand,
        HumanoidBone::RightIndexProximal,
        HumanoidBone::RightHand,
    ),
    (
        HumanoidBone::RightIndexProximal,
        HumanoidBone::RightIndexIntermediate,
        HumanoidBone::RightIndexProximal,
    ),
    (
        HumanoidBone::RightIndexIntermediate,
        HumanoidBone::RightIndexDistal,
        HumanoidBone::RightIndexIntermediate,
    ),
    (
        HumanoidBone::RightHand,
        HumanoidBone::RightMiddleProximal,
        HumanoidBone::RightHand,
    ),
    (
        HumanoidBone::RightMiddleProximal,
        HumanoidBone::RightMiddleIntermediate,
        HumanoidBone::RightMiddleProximal,
    ),
    (
        HumanoidBone::RightMiddleIntermediate,
        HumanoidBone::RightMiddleDistal,
        HumanoidBone::RightMiddleIntermediate,
    ),
    (
        HumanoidBone::RightHand,
        HumanoidBone::RightRingProximal,
        HumanoidBone::RightHand,
    ),
    (
        HumanoidBone::RightRingProximal,
        HumanoidBone::RightRingIntermediate,
        HumanoidBone::RightRingProximal,
    ),
    (
        HumanoidBone::RightRingIntermediate,
        HumanoidBone::RightRingDistal,
        HumanoidBone::RightRingIntermediate,
    ),
    (
        HumanoidBone::RightHand,
        HumanoidBone::RightLittleProximal,
        HumanoidBone::RightHand,
    ),
    (
        HumanoidBone::RightLittleProximal,
        HumanoidBone::RightLittleIntermediate,
        HumanoidBone::RightLittleProximal,
    ),
    (
        HumanoidBone::RightLittleIntermediate,
        HumanoidBone::RightLittleDistal,
        HumanoidBone::RightLittleIntermediate,
    ),
];

/// Distal bones whose fingertip / toe tip lives in
/// `SourceSkeleton::fingertips`. Each one's lifted 3D parent is the
/// distal bone itself.
const LIFT_FINGERTIP_PARENTS: &[HumanoidBone] = &[
    HumanoidBone::LeftThumbDistal,
    HumanoidBone::LeftIndexDistal,
    HumanoidBone::LeftMiddleDistal,
    HumanoidBone::LeftRingDistal,
    HumanoidBone::LeftLittleDistal,
    HumanoidBone::RightThumbDistal,
    HumanoidBone::RightIndexDistal,
    HumanoidBone::RightMiddleDistal,
    HumanoidBone::RightRingDistal,
    HumanoidBone::RightLittleDistal,
    HumanoidBone::LeftFoot,
    HumanoidBone::RightFoot,
];

/// Lift the source skeleton's 2D joints to 3D in tree order.
///
/// Each child's depth is anchored to its parent's depth. The 2D length
/// of each bone is calibrated against a per-bone running max so the
/// foreshortening Z is `√(L_max² − d_observed²)`, with the sign
/// (positive = toward camera, negative = away from camera) chosen to
/// minimise the change from the previous frame's signed offset. The
/// magnitude alone is fully determined by 2D; only the sign carries
/// across frames, and its decision is local (per bone) — enough to
/// stop a small detector wiggle at d≈L from oscillating between ±Z
/// while still letting a real limb flip cleanly when the wiggle is
/// large enough to dominate.
fn lift_2d_to_3d(
    source: &SourceSkeleton,
    state: &mut PoseSolverState,
    threshold: f32,
) -> LiftedSkeleton {
    let mut lifted = LiftedSkeleton::default();

    // 1. Hips at z=0 — the lift's anchor. Without Hips we cannot
    //    propagate any chain.
    let Some(hips) = source.joints.get(&HumanoidBone::Hips) else {
        return lifted;
    };
    if hips.confidence < threshold {
        return lifted;
    }
    lifted
        .joints_3d
        .insert(HumanoidBone::Hips, [hips.position[0], hips.position[1], 0.0]);

    // 2. Spine: shoulder midpoint as the Spine bone's tip. Calibrate
    //    against the Spine bone's running max. Both shoulders inherit
    //    the midpoint's Z (we do not separately resolve the inter-
    //    shoulder Z difference here — body_yaw on Hips already accounts
    //    for the rotated shoulder line, and the per-arm lift then picks
    //    up the residual depth).
    if let (Some(ls), Some(rs)) = (
        source.joints.get(&HumanoidBone::LeftShoulder),
        source.joints.get(&HumanoidBone::RightShoulder),
    ) {
        if ls.confidence >= threshold && rs.confidence >= threshold {
            let mid_2d = [
                (ls.position[0] + rs.position[0]) * 0.5,
                (ls.position[1] + rs.position[1]) * 0.5,
            ];
            let hips_3d = lifted.joints_3d[&HumanoidBone::Hips];
            let mid_3d = lift_step(state, &hips_3d, mid_2d, HumanoidBone::Spine);
            lifted.shoulder_mid_3d = Some(mid_3d);
            // Shoulders and upper arms share Z with the midpoint.
            for (bone, pos_2d) in [
                (HumanoidBone::LeftShoulder, ls.position),
                (HumanoidBone::LeftUpperArm, ls.position),
                (HumanoidBone::RightShoulder, rs.position),
                (HumanoidBone::RightUpperArm, rs.position),
            ] {
                lifted
                    .joints_3d
                    .insert(bone, [pos_2d[0], pos_2d[1], mid_3d[2]]);
            }
        }
    }

    // 3. Walk the rest of the chain in declared order.
    for &(parent, child, calib_key) in LIFT_CHAINS {
        let Some(parent_3d) = lifted.joints_3d.get(&parent).copied() else {
            continue;
        };
        let Some(child_joint) = source.joints.get(&child) else {
            continue;
        };
        if child_joint.confidence < threshold {
            continue;
        }
        let child_3d = lift_step(state, &parent_3d, child_joint.position, calib_key);
        lifted.joints_3d.insert(child, child_3d);
    }

    // 4. Fingertips / toe tips. Each one's parent is the distal bone
    //    itself (e.g. LeftThumbDistal → fingertip). The calibration
    //    key is the parent (matching the existing Tip::Fingertip path
    //    that uses `state.bone_max_2d.get(&distal_bone)`).
    for &distal in LIFT_FINGERTIP_PARENTS {
        let Some(parent_3d) = lifted.joints_3d.get(&distal).copied() else {
            continue;
        };
        let Some(tip_joint) = source.fingertips.get(&distal) else {
            continue;
        };
        if tip_joint.confidence < threshold {
            continue;
        }
        let tip_3d = lift_step_fingertip(state, &parent_3d, tip_joint.position, distal);
        lifted.fingertips_3d.insert(distal, tip_3d);
    }

    lifted
}

/// Lift one body-skeleton step: place `child_2d` in 3D anchored to
/// `parent_3d`. Updates `state.bone_max_2d[calib_key]` (running max of
/// observed 2D length) and `state.bone_prev_z_offset[calib_key]`
/// (signed Z offset, persisted for sign continuity).
fn lift_step(
    state: &mut PoseSolverState,
    parent_3d: &[f32; 3],
    child_2d: [f32; 2],
    calib_key: HumanoidBone,
) -> [f32; 3] {
    let dx = child_2d[0] - parent_3d[0];
    let dy = child_2d[1] - parent_3d[1];
    let observed_sq = dx * dx + dy * dy;
    let observed = observed_sq.sqrt();
    let calibration = state
        .bone_max_2d
        .get(&calib_key)
        .copied()
        .unwrap_or(0.0)
        .max(observed);
    state.bone_max_2d.insert(calib_key, calibration);
    let delta_sq = calibration * calibration - observed_sq;
    let z_mag = if delta_sq > 1e-8 { delta_sq.sqrt() } else { 0.0 };
    let prev = state.bone_prev_z_offset.get(&calib_key).copied().unwrap_or(0.0);
    let z_signed = pick_continuous_z(prev, z_mag);
    state.bone_prev_z_offset.insert(calib_key, z_signed);
    [child_2d[0], child_2d[1], parent_3d[2] + z_signed]
}

/// Same as [`lift_step`] but writes into the fingertip Z map so finger
/// continuity does not collide with the body-bone keys (a finger
/// distal bone keys both into `bone_max_2d` for its own length and
/// `bone_prev_z_offset` — the fingertip extension uses the same
/// `bone_max_2d` slot but a separate prev-z slot to keep its sign
/// independent from the bone's own Z motion).
fn lift_step_fingertip(
    state: &mut PoseSolverState,
    parent_3d: &[f32; 3],
    tip_2d: [f32; 2],
    calib_key: HumanoidBone,
) -> [f32; 3] {
    let dx = tip_2d[0] - parent_3d[0];
    let dy = tip_2d[1] - parent_3d[1];
    let observed_sq = dx * dx + dy * dy;
    let observed = observed_sq.sqrt();
    let calibration = state
        .bone_max_2d
        .get(&calib_key)
        .copied()
        .unwrap_or(0.0)
        .max(observed);
    state.bone_max_2d.insert(calib_key, calibration);
    let delta_sq = calibration * calibration - observed_sq;
    let z_mag = if delta_sq > 1e-8 { delta_sq.sqrt() } else { 0.0 };
    let prev = state
        .fingertip_prev_z_offset
        .get(&calib_key)
        .copied()
        .unwrap_or(0.0);
    let z_signed = pick_continuous_z(prev, z_mag);
    state.fingertip_prev_z_offset.insert(calib_key, z_signed);
    [tip_2d[0], tip_2d[1], parent_3d[2] + z_signed]
}

/// Choose the sign of `z_mag` (≥ 0) that keeps the offset as close as
/// possible to `prev`. On a fresh state (`prev == 0`) this defaults to
/// `+z_mag`, matching the original "always toward camera" behaviour.
#[inline]
fn pick_continuous_z(prev: f32, z_mag: f32) -> f32 {
    if z_mag <= 0.0 {
        return 0.0;
    }
    let diff_pos = (z_mag - prev).abs();
    let diff_neg = (-z_mag - prev).abs();
    if diff_pos <= diff_neg {
        z_mag
    } else {
        -z_mag
    }
}

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

    // Compute per-frame dt against the previous solve; clamp so a
    // long pause (debugger, GC stall) does not produce a huge α that
    // snaps the avatar to whatever stale or noisy data arrived first.
    // On the very first call we have no reference, so substitute the
    // 30 fps reference period — the slider value then maps to itself
    // on frame zero, matching the pre-smoothing behaviour.
    let now = Instant::now();
    let dt = match state.last_solve_instant {
        Some(prev) => (now - prev).as_secs_f32().clamp(0.0, 0.25),
        None => ROTATION_BLEND_REFERENCE_DT,
    };
    state.last_solve_instant = Some(now);

    // Apply the 1€ filter + Schmitt hysteresis on every keypoint
    // before any geometry runs. Working on a local clone keeps
    // `&SourceSkeleton` immutable for callers (tests, other consumers).
    let source_owned = preprocess_source(source, state, dt, params);
    let source = &source_owned;

    // Decay the per-bone running max so transient detector spikes do
    // not pin Z high indefinitely. Real, sustained peaks get
    // refreshed every frame and never decay; outliers wash out over
    // RUNNING_MAX_DECAY_TAU_SEC. Also decays shoulder-span max.
    if dt > 0.0 {
        let factor = (-dt / RUNNING_MAX_DECAY_TAU_SEC).exp();
        for v in state.bone_max_2d.values_mut() {
            *v *= factor;
        }
        state.shoulder_span_max_2d *= factor;
    }

    // 2D → 3D lift (tree walk). Each child's depth is anchored to its
    // parent's depth, so the source direction we feed into the per-
    // bone direction match is by construction consistent with the
    // parent's already-resolved 3D position. The lift updates the
    // running-max calibration as a side effect.
    let lifted = lift_2d_to_3d(source, state, params.joint_confidence_threshold);

    // Update shoulder-span max so we can later detect side/rear views.
    // Computed every frame (regardless of any per-bone gating) so the
    // running max tracks the subject's true shoulder span across
    // sessions of front-facing motion.
    let shoulder_span_now = current_shoulder_span_2d(source);
    if let Some(span) = shoulder_span_now {
        if span > state.shoulder_span_max_2d {
            state.shoulder_span_max_2d = span;
        }
    }
    let body_facing_camera_ratio = match (shoulder_span_now, state.shoulder_span_max_2d) {
        (Some(now), max) if max > 1e-6 => (now / max).clamp(0.0, 1.0),
        _ => 1.0,
    };

    // Body yaw: how far the subject's torso has rotated around the
    // vertical axis. We don't see this directly from 2D, but two
    // signals together pin it down:
    //
    //   * Shoulder-span foreshortening gives magnitude. A full span
    //     means yaw = 0 (facing camera) or yaw = ±180° (facing away);
    //     a fully-collapsed span means side profile (yaw = ±90°).
    //
    //   * The horizontal sign of (RightShoulder.x − LeftShoulder.x)
    //     flips between front-half and rear-half of the rotation
    //     because COCO's left/right labels are subject-POV: when the
    //     subject faces away their left side now lands on camera-left.
    //
    //   * For left/right disambiguation (90° vs 270°, etc.) we use
    //     the *sign* of the face yaw when the face is still visible
    //     enough for the inference to return a pose; otherwise we fall
    //     back to a default sign.
    let body_yaw = compute_body_yaw_signed(source, state, body_facing_camera_ratio);

    // Rest-pose world transforms, computed from the skeleton's rest_local
    // values. These are the ground truth we measure bone directions against;
    // if the avatar has any offset rotations baked into rest pose (common
    // for VRM rigs) they are respected.
    let rest_world = compute_world_transforms(skeleton, |node_idx| {
        skeleton.nodes[node_idx].rest_local.clone()
    });

    // Current-frame world transforms, starting from whatever is in
    // local_transforms right now (i.e. rest, unless the caller has already
    // applied animation clips). Updated incrementally as we drive each
    // bone so that a child's new local rotation is computed relative to
    // its parent's updated world rotation.
    let mut current_world = compute_world_transforms(skeleton, |i| local_transforms[i].clone());

    // --- 0. Body root yaw: rotate Hips around Y so the avatar's torso
    //         faces the same way as the subject. Skipped when calibration
    //         has not yet seen a wider shoulder span than the current
    //         frame (running max == observed → ratio = 1 → yaw = 0).
    if body_yaw.abs() > 0.01 {
        if let Some(hips_node) = humanoid.bone_map.get(&HumanoidBone::Hips).copied() {
            let hips_idx = hips_node.0 as usize;
            if hips_idx < skeleton.nodes.len() {
                let yaw_delta_world = quat_from_euler_ypr(0.0, body_yaw, 0.0);
                let rest_world_rot = rest_world[hips_idx].rotation;
                let new_world_rot = quat_mul(&yaw_delta_world, &rest_world_rot);
                let parent_world_rot = skeleton.nodes[hips_idx]
                    .parent
                    .map(|NodeId(p)| current_world[p as usize].rotation)
                    .unwrap_or([0.0, 0.0, 0.0, 1.0]);
                let new_local_rot = quat_normalize(&quat_mul(
                    &quat_conjugate(&parent_world_rot),
                    &new_world_rot,
                ));
                let prev_local_rot = local_transforms[hips_idx].rotation;
                local_transforms[hips_idx].rotation =
                    quat_slerp_short(&prev_local_rot, &new_local_rot, dt_aware_blend(params.rotation_blend, dt));
                let updated_world = quat_mul(&parent_world_rot, &local_transforms[hips_idx].rotation);
                current_world[hips_idx].rotation = updated_world;
            }
        }
    }

    // --- 1. Body chain: direction-match each driven bone. ---
    //
    // Each bone's source direction comes from the lifted 3D positions
    // (`child_3d − parent_3d`) rather than per-bone reconstruction.
    // This guarantees chain consistency: if the upper-arm has been
    // placed at z = +0.15, the lower-arm's source direction starts
    // from that same z (rather than independently picking up its own
    // arbitrary z), so kinematic-chain noise — legs zig-zagging in
    // 3D, finger curling out of a closed fist — is suppressed at the
    // lift step instead of leaking into per-bone rotation matches.
    //
    // The Y-flip baked onto VRM 0.x roots already handles the
    // "avatar faces camera" orientation, so comparing rest_world
    // directions with camera-space lifted vectors works uniformly
    // across VRM 0.x and 1.x.
    for &(bone, tip) in DRIVEN_BONES {
        let Some(source_base_3d) = source_3d_for_bone(bone, &lifted) else {
            continue;
        };
        let source_tip_3d = match tip {
            Tip::Joint(tip_bone) => source_3d_for_bone(tip_bone, &lifted),
            Tip::ShoulderMidpoint => lifted.shoulder_mid_3d,
            Tip::Fingertip => lifted.fingertips_3d.get(&bone).copied(),
        };
        let Some(source_tip_3d) = source_tip_3d else {
            continue;
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
        let rest_base_pos = rest_world[node_idx].position;
        let rest_dir = vec3_normalize(&vec3_sub(&rest_tip_pos, &rest_base_pos));
        if rest_dir == [0.0, 0.0, 0.0] {
            continue;
        }

        let source_dir = vec3_normalize(&vec3_sub(&source_tip_3d, &source_base_3d));
        if source_dir == [0.0, 0.0, 0.0] {
            continue;
        }

        let delta_world = quat_from_vectors(&rest_dir, &source_dir);
        let rest_world_rot = rest_world[node_idx].rotation;
        let new_world_rot = quat_mul(&delta_world, &rest_world_rot);

        // Convert the world-space new rotation into a parent-local one
        // using the parent's *current* world rotation (which already
        // reflects any earlier driven bones in this frame).
        let parent_world_rot = skeleton.nodes[node_idx]
            .parent
            .map(|NodeId(p)| current_world[p as usize].rotation)
            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let new_local_rot = quat_normalize(&quat_mul(&quat_conjugate(&parent_world_rot), &new_world_rot));

        let prev_local_rot = local_transforms[node_idx].rotation;
        local_transforms[node_idx].rotation = quat_slerp_short(
            &prev_local_rot,
            &new_local_rot,
            dt_aware_blend(params.rotation_blend, dt),
        );

        // Keep the world rotation cache in sync so child bones in this
        // same pass see the updated parent orientation.
        let updated_world = quat_mul(&parent_world_rot, &local_transforms[node_idx].rotation);
        current_world[node_idx].rotation = updated_world;
    }

    // --- 2. Face pose: drive the Head bone independently. ---
    //
    // Skip when the body has yawed enough that the face-pose detector
    // can no longer see both eyes/ears (single-side landmarks produce
    // wildly wrong yaw/pitch/roll, which otherwise rotate the head into
    // a back-of-camera position at side profile / rear views).
    if body_facing_camera_ratio >= FACE_GATE_SHOULDER_RATIO {
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
                            dt_aware_blend(params.rotation_blend, dt),
                        );
                    }
                }
            }
        }
    }
}

/// Current 2D shoulder span (||L − R|| in source image coords), if both
/// shoulders are present.
fn current_shoulder_span_2d(source: &SourceSkeleton) -> Option<f32> {
    let l = source.joints.get(&HumanoidBone::LeftShoulder)?;
    let r = source.joints.get(&HumanoidBone::RightShoulder)?;
    let dx = l.position[0] - r.position[0];
    let dy = l.position[1] - r.position[1];
    Some((dx * dx + dy * dy).sqrt())
}

/// Estimate the subject's body yaw (rotation around the vertical
/// world axis) in radians.
///
/// Convention: 0 = facing camera, ±π = facing away, +π/2 = subject
/// rotated to their right (camera sees their left side), −π/2 = the
/// mirror.
///
/// Two signals combine:
/// * `body_facing_camera_ratio` (current shoulder span / running max)
///   gives the magnitude through `acos(ratio)`. Front/rear are
///   disambiguated by the horizontal sign of `R.x − L.x`: COCO labels
///   are subject-POV so the sign flips at 90°/270° crossings.
/// * Left/right (90° vs 270°) is taken from the sign of the face-pose
///   yaw when one was detected; otherwise default to +.
fn compute_body_yaw_signed(
    source: &SourceSkeleton,
    _state: &PoseSolverState,
    ratio: f32,
) -> f32 {
    let (Some(l), Some(r)) = (
        source.joints.get(&HumanoidBone::LeftShoulder),
        source.joints.get(&HumanoidBone::RightShoulder),
    ) else {
        return 0.0;
    };
    // Magnitude of yaw based on how foreshortened the shoulders are.
    let mag_front_half = ratio.clamp(0.0, 1.0).acos();
    let lr_dx = r.position[0] - l.position[0];
    // Empirically (see sample_data/all_diag/*/source_skeleton.txt) the
    // detector emits `LeftShoulder` on camera-left and `RightShoulder`
    // on camera-right at front view: `R.x − L.x` is positive at 0°,
    // sweeps through ~0 at ±90° (side profile), and goes negative at
    // 180° (rear). So front-half ⇔ lr_dx ≥ 0, rear-half ⇔ lr_dx < 0.
    let yaw_unsigned = if lr_dx >= 0.0 {
        mag_front_half
    } else {
        std::f32::consts::PI - mag_front_half
    };
    // Sign: +ve = avatar rotates to its right (mirror of subject
    // turning to their right). Two complementary signals:
    //
    //   * Face yaw (when detected). A face turned to camera-+X has
    //     positive yaw; the subject is therefore rotating to their
    //     left. Avatar should rotate to its right → +ve sign. So
    //     `sign = sign(face.yaw)`.
    //
    //   * At side / rear views the face inference rejects, but the
    //     hand confidence asymmetry survives: the visible side has
    //     the high-confidence hand. Empirically (sample_data
    //     rotation series) `LeftHand` retains high confidence at
    //     "left-side profile" 90° rotations and `RightHand` at the
    //     mirror-symmetric 270° rotations. We use the difference to
    //     pick the sign.
    // The face inference returns `confidence = 0` on rear views to
    // signal "magnitude unreliable but sign of yaw still tracks the
    // subject's rotation direction" — so we accept yaw sign even at
    // zero confidence here, while the head-bone driver has already
    // skipped via the face-confidence threshold.
    let sign = if let Some(face) = source.face {
        if face.yaw.abs() > 0.05 {
            face.yaw.signum()
        } else {
            hand_visibility_sign(source)
        }
    } else {
        hand_visibility_sign(source)
    };
    sign * yaw_unsigned
}

/// Pick a body-yaw sign from hand-confidence asymmetry. Side-profile
/// occludes one hand entirely so the diff can be large; rear-3/4
/// keeps both hands visible and the diff shrinks but its sign still
/// tracks which side has rotated forward toward the camera.
///
/// Returns +1 when the LEFT-labelled hand wins (subject's left side
/// is closer to camera), -1 when RIGHT wins, and +1 as the
/// fallback when the diff is below the noise floor.
fn hand_visibility_sign(source: &SourceSkeleton) -> f32 {
    const ASYM_MIN: f32 = 0.02;
    let l = source
        .joints
        .get(&HumanoidBone::LeftHand)
        .map(|j| j.confidence)
        .unwrap_or(0.0);
    let r = source
        .joints
        .get(&HumanoidBone::RightHand)
        .map(|j| j.confidence)
        .unwrap_or(0.0);
    let diff = l - r;
    if diff.abs() < ASYM_MIN {
        1.0
    } else if diff > 0.0 {
        1.0
    } else {
        -1.0
    }
}

fn apply_face_pose(
    face: FacePose,
    head_idx: usize,
    skeleton: &SkeletonAsset,
    local_transforms: &mut [Transform],
    current_world: &[WorldXform],
    blend: f32,
) {
    // Face pose is in body-local Y-up convention: a positive yaw turns the
    // head to the avatar's left in its own frame. Compose the delta in the
    // Head bone's parent-local frame by rotating it through the parent's
    // world rotation: the avatar's local +Y in world space is the parent's
    // Y axis rotated by the parent's world rotation, so we build the delta
    // in world space and then conjugate it back into parent-local.
    let delta_world = quat_from_euler_ypr(face.pitch, face.yaw, face.roll);

    // Rest-relative: the new local rotation is rest_local * delta_local,
    // where delta_local is delta_world expressed in the Head bone's
    // parent-local frame.
    let parent_world_rot = skeleton.nodes[head_idx]
        .parent
        .map(|NodeId(p)| current_world[p as usize].rotation)
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let delta_local = quat_mul(
        &quat_mul(&quat_conjugate(&parent_world_rot), &delta_world),
        &parent_world_rot,
    );
    let rest_local_rot = skeleton.nodes[head_idx].rest_local.rotation;
    let target_local = quat_normalize(&quat_mul(&rest_local_rot, &delta_local));
    let prev = local_transforms[head_idx].rotation;
    local_transforms[head_idx].rotation = quat_slerp_short(&prev, &target_local, blend);
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

/// Look up a humanoid bone's lifted 3D position. Returns `None` when
/// the lift skipped the bone (low confidence, missing parent in the
/// chain, etc.).
fn source_3d_for_bone(bone: HumanoidBone, lifted: &LiftedSkeleton) -> Option<[f32; 3]> {
    lifted.joints_3d.get(&bone).copied()
}

fn midpoint(a: &Vec3, b: &Vec3) -> Vec3 {
    [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ]
}

/// Convert the GUI's frame-rate-naive `rotation_blend` slider value
/// into a dt-aware α. The slider's intent is "what fraction of the
/// new pose to absorb each frame at the reference rate"; reframing
/// it as a time-constant means the same slider value behaves
/// identically whether the camera ships 30 or 60 fps and whether
/// the renderer hitches occasionally.
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

    let enter = params.joint_confidence_threshold;
    let exit = enter * SCHMITT_EXIT_RATIO;

    for (bone, joint) in out.joints.iter_mut() {
        let filt = state.joint_filters.entry(*bone).or_default();
        let smoothed = filt.apply(joint.position, dt);
        joint.position = smoothed;
        // Schmitt hysteresis: a single dropout below `enter` does
        // not turn the joint off as long as it stays above `exit`.
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

    for (bone, joint) in out.fingertips.iter_mut() {
        let filt = state.fingertip_filters.entry(*bone).or_default();
        joint.position = filt.apply(joint.position, dt);
        // Fingertips don't go through the body-bone DRIVEN_BONES list
        // for confidence gating directly — the distal bone owns the
        // gate — so we only smooth the position here.
    }

    out
}

// Slerp / lerp
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
) -> Vec<ResolvedExpressionWeight> {
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
            let blended = prev_w + expression_blend * (raw - prev_w);
            Some(ResolvedExpressionWeight {
                name: expr_def.name.clone(),
                weight: blended.clamp(0.0, 1.0),
            })
        })
        .collect()
}
