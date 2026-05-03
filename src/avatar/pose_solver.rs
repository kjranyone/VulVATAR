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
//!    It is applied as a rest-relative local delta on the Head bone.
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
    quat_normalize, quat_rotate_vec3, swing_twist_decompose, vec3_cross, vec3_dot, vec3_length,
    vec3_normalize, vec3_sub,
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
    /// Per-axis sensitivity for the root translation. Multiplied
    /// against the source-skeleton `root_offset` deviation (relative
    /// to a slow EMA reference) before being added to the Hips rest
    /// local position. `1.0` follows the subject 1:1 in source-space
    /// units (note that depth-aware providers emit METERS while
    /// rtmw3d-only emits normalised image units, so the comfortable
    /// value differs per provider). Defaults pick a moderate follow.
    pub root_translation_sensitivity: [f32; 3],
    /// Optional pose-calibration capture from the `Calibrate Pose ▼`
    /// modal. When present, the solver uses the captured anchor as
    /// the explicit EMA reference seed for root translation —
    /// replacing the auto-EMA's "first hip-visible frame" heuristic
    /// with a user-acknowledged neutral position. Falls through to
    /// the auto-EMA when `None`. See `docs/calibration-ux.md`.
    pub pose_calibration: Option<crate::tracking::PoseCalibration>,
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
            // 0.6 horizontal/vertical, 0.3 depth: in-plane motion reads
            // most naturally at near-1:1 follow, while depth (already in
            // metric metres for depth-aware providers) needs to be
            // dampened so a 0.5 m forward step doesn't shove the avatar
            // through the camera plane.
            root_translation_sensitivity: [0.6, 0.6, 0.3],
            pose_calibration: None,
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
    /// Per-bone Schmitt hysteresis state for joint confidence.
    joint_active: HashMap<HumanoidBone, bool>,
    /// Wall-clock timestamp of the previous solve, used to derive `dt`
    /// for the dt-aware blend / 1€ filter. `None` until the first call.
    last_solve_instant: Option<Instant>,
    /// Slow EMA of the source-skeleton `root_offset`, used as the
    /// "where the subject normally stands" reference. The avatar's
    /// Hips is translated by `(root_offset − reference) * sensitivity`
    /// so the feature self-calibrates to the user's typical pose.
    /// `None` until the first hip-visible frame.
    root_reference: Option<[f32; 3]>,
    /// Running max of `|L.shoulder.x − R.shoulder.x|` observed across
    /// frames. This proxies the subject's true frontal shoulder width:
    /// the X-span shrinks as the body yaws away from frontal, so the
    /// historical maximum approximates the un-foreshortened width.
    /// [`compute_body_yaw_3d`] uses this to recover the yaw magnitude
    /// from X-foreshortening, which is more stable than RTMW3D's
    /// per-frame Δz signal at intermediate (~30°–60°) rotations.
    /// `0.0` until the first confident shoulder pair arrives.
    running_shoulder_x_span_max: f32,
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
        self.joint_active.clear();
        self.last_solve_instant = None;
        self.root_reference = None;
        // `running_shoulder_x_span_max` is intentionally NOT reset
        // here — it captures the subject's anatomical shoulder width
        // and persists across smoothing resets so a calibration-prime
        // frame's reading still informs subsequent test frames in the
        // diagnose_pose workflow (and survives view switches in live
        // tracking, since the user's anatomy has not changed).
    }
}

/// 1€ filter state for a single 3D keypoint stream. See
/// [Casiez et al. 2012](https://hal.inria.fr/hal-00670496).
#[derive(Clone, Copy, Debug, Default)]
struct OneEuroFilterState {
    initialized: bool,
    pos: [f32; 3],
    vel: [f32; 3],
}

impl OneEuroFilterState {
    /// Apply the filter to a raw position and return the smoothed
    /// position. `dt` is seconds since the previous update for this
    /// stream. Self-initializes on the first call (no smoothing).
    fn apply(&mut self, raw: [f32; 3], dt: f32) -> [f32; 3] {
        if !self.initialized || dt <= 0.0 {
            self.initialized = true;
            self.pos = raw;
            self.vel = [0.0, 0.0, 0.0];
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
        let speed = (self.vel[0] * self.vel[0]
            + self.vel[1] * self.vel[1]
            + self.vel[2] * self.vel[2])
            .sqrt();
        let cutoff = ONE_EURO_MIN_CUTOFF_HZ + ONE_EURO_BETA * speed;
        let alpha = one_euro_alpha(dt, cutoff);
        for (pos, raw) in self.pos.iter_mut().zip(raw.iter()) {
            *pos += alpha * (raw - *pos);
        }
        self.pos
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
    let mut source_owned = preprocess_source(source, state, dt, params);

    // Rest-pose world transforms, computed from the skeleton's rest_local
    // values. These are the ground truth we measure bone directions against;
    // if the avatar has any offset rotations baked into rest pose (common
    // for VRM rigs) they are respected.
    let rest_world = compute_world_transforms(skeleton, |node_idx| {
        skeleton.nodes[node_idx].rest_local.clone()
    });

    // Source body yaw, computed early so the IK pole hints (which
    // describe limb-bend direction in body-local terms — knee bends
    // forward, elbow bends down) can be rotated into the source frame
    // before IK runs. Without this rotation the leg IK assumes the
    // subject is facing the camera and bends knees toward +Z even when
    // the body has rotated 45°+ to the side, producing a visibly
    // dislocated leg in three-quarter / profile poses.
    let body_yaw_source = compute_body_yaw_3d(
        &source_owned,
        params.joint_confidence_threshold,
        &mut state.running_shoulder_x_span_max,
    );

    // Two-bone IK: the tracker's mid-limb keypoint (elbow / knee) is
    // unreliable for foreshortened poses (raised hand, knee-near-camera).
    // Trust the more-reliable shoulder/hip + wrist/ankle keypoints and
    // reconstruct the elbow / knee position geometrically using the
    // avatar's anatomical bone-length ratios. See `apply_2bone_ik_chain`
    // for the full rationale and math; the call list below pins which
    // chains get IK'd and what pole hint each uses.
    apply_2bone_ik_arms_and_legs(
        &mut source_owned,
        &rest_world,
        humanoid,
        params.joint_confidence_threshold,
        body_yaw_source.unwrap_or(0.0),
    );

    let source = &source_owned;

    // Current-frame world transforms, starting from whatever is in
    // local_transforms right now (i.e. rest, unless the caller has already
    // applied animation clips). Updated incrementally as we drive each
    // bone so that a child's new local rotation is computed relative to
    // its parent's updated world rotation.
    let mut current_world = compute_world_transforms(skeleton, |i| local_transforms[i].clone());

    // Body root yaw: rotate Hips around Y so the avatar's torso faces
    // the same way as the subject. With a 3D-native tracker this falls
    // out cleanly from the shoulder line in the XZ plane: at rest the
    // shoulder line points along +X (avatar's left → camera-right),
    // and as the subject rotates around their vertical axis the line
    // tilts in XZ so its yaw against the rest +X axis encodes the
    // body's facing direction directly.
    if let Some(body_yaw) = body_yaw_source {
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
                    local_transforms[hips_idx].rotation = quat_slerp_short(
                        &prev_local_rot,
                        &new_local_rot,
                        dt_aware_blend(params.rotation_blend, dt),
                    );
                    let updated_world =
                        quat_mul(&parent_world_rot, &local_transforms[hips_idx].rotation);
                    current_world[hips_idx].rotation = updated_world;
                }
            }
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
                        state.root_reference = Some([
                            cal.anchor_x,
                            cal.anchor_y,
                            cal.anchor_depth_m.unwrap_or(0.0),
                        ]);
                    }
                }
            }

            // EMA blend factor: the reference should drift slowly
            // (~10 s effective horizon at 30 fps). Faster on the very
            // first hip-visible frame (no calibration-derived seed
            // either) so the auto-EMA locks in quickly without
            // introducing a big initial jump.
            let alpha = if state.root_reference.is_none() {
                1.0
            } else {
                (dt / 10.0).clamp(0.0, 0.2)
            };
            let prev_ref = state.root_reference.unwrap_or(raw_offset);
            let new_ref = [
                prev_ref[0] + alpha * (raw_offset[0] - prev_ref[0]),
                prev_ref[1] + alpha * (raw_offset[1] - prev_ref[1]),
                prev_ref[2] + alpha * (raw_offset[2] - prev_ref[2]),
            ];
            state.root_reference = Some(new_ref);

            let dev = [
                raw_offset[0] - new_ref[0],
                raw_offset[1] - new_ref[1],
                raw_offset[2] - new_ref[2],
            ];
            // Per-axis sensitivity. The static defaults
            // (`[0.6, 0.6, 0.3]`) are the right floor when we know
            // nothing about the user's room, but a calibration that
            // captured an explicit X/Z range (multi-step calibration:
            // step left/right, lean in/out) lets us derive the gain
            // from the *observed* envelope — a user with a narrow
            // room (small ±X) gets a higher gain so a 30 cm step
            // still moves the avatar across half its world-space
            // horizontal envelope, while a large studio gets a lower
            // gain so motion stays within frame.
            //
            // Formula: `sens = TARGET / (range / 2)` so the avatar's
            // hip reaches `±TARGET` units when the subject reaches
            // their captured extremes. The half-range is what's
            // relevant because the EMA reference sits in the middle
            // of the captured sweep.
            //
            // Clamp keeps a tiny captured range (user barely moved)
            // from producing runaway gain, and a huge one from
            // dropping below the static default.
            const TARGET_AVATAR_X: f32 = 0.5;
            const TARGET_AVATAR_Z: f32 = 0.3;
            const SENS_X_CLAMP: (f32, f32) = (0.3, 2.5);
            const SENS_Z_CLAMP: (f32, f32) = (0.15, 1.0);

            let static_sens = params.root_translation_sensitivity;
            let derived_sens_x = params
                .pose_calibration
                .as_ref()
                .and_then(|c| c.x_range_observed)
                .map(|range| {
                    let half = (range * 0.5).max(0.05);
                    (TARGET_AVATAR_X / half).clamp(SENS_X_CLAMP.0, SENS_X_CLAMP.1)
                })
                .unwrap_or(static_sens[0]);
            let derived_sens_z = params
                .pose_calibration
                .as_ref()
                .and_then(|c| c.z_range_observed)
                .map(|range| {
                    let half = (range * 0.5).max(0.05);
                    (TARGET_AVATAR_Z / half).clamp(SENS_Z_CLAMP.0, SENS_Z_CLAMP.1)
                })
                .unwrap_or(static_sens[2]);

            let translation_delta = [
                dev[0] * derived_sens_x,
                dev[1] * static_sens[1],
                dev[2] * derived_sens_z,
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
                    let prev = local_transforms[hips_idx].translation;
                    local_transforms[hips_idx].translation = [
                        prev[0] + blend * (target[0] - prev[0]),
                        prev[1] + blend * (target[1] - prev[1]),
                        prev[2] + blend * (target[2] - prev[2]),
                    ];
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
        let rest_base_pos = rest_world[node_idx].position;
        let rest_dir = vec3_normalize(&vec3_sub(&rest_tip_pos, &rest_base_pos));
        if rest_dir == [0.0, 0.0, 0.0] {
            continue;
        }

        let source_dir = vec3_normalize(&vec3_sub(&source_tip_pos, &source_base_pos));
        if source_dir == [0.0, 0.0, 0.0] {
            continue;
        }

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
        let new_world_rot = quat_mul(&delta_world, &current_rest_world_rot);

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
fn compute_body_yaw_3d(
    source: &SourceSkeleton,
    threshold: f32,
    running_shoulder_x_span_max: &mut f32,
) -> Option<f32> {
    use HumanoidBone::*;
    // Shoulders first; hips are a fallback for shoulder-cropped frames.
    let pairs = [
        (LeftShoulder, RightShoulder),
        (LeftUpperLeg, RightUpperLeg),
    ];
    for (left_bone, right_bone) in pairs {
        let Some(l) = source.joints.get(&left_bone) else {
            continue;
        };
        let Some(r) = source.joints.get(&right_bone) else {
            continue;
        };
        if l.confidence < threshold || r.confidence < threshold {
            continue;
        }
        let bx = l.position[0] - r.position[0];
        let bz = l.position[2] - r.position[2];
        let span_xz = (bx * bx + bz * bz).sqrt();
        // Dead-zone: with the torso-width vector tiny (subject in
        // exact side profile or hip pair degenerate), atan2 amplifies
        // any noise into wild yaw flips. Require a minimum span
        // before we trust the angle, and let the next pair (or
        // None) take over otherwise.
        if span_xz < 0.05 {
            continue;
        }
        // Update the running max only on the shoulder pair — shoulders
        // have stable anatomy and clean keypoints; hips are a noisy
        // fallback signal we don't want polluting the historical
        // reference. Use `|bx|` so a back-facing pose (bx negative)
        // also contributes its full extent.
        let abs_bx = bx.abs();
        if matches!(left_bone, LeftShoulder) && abs_bx > *running_shoulder_x_span_max {
            *running_shoulder_x_span_max = abs_bx;
        }
        // Yaw recovery from X-foreshortening:
        //   |bx| / true_x_span = |cos(yaw)|
        // The "true x-span" is the X-axis shoulder width at frontal pose
        // (yaw = 0). RTMW3D's per-frame Δz under-estimates depth at
        // intermediate yaw (~30°–60°) where bz is small; the running
        // max of the observed X-span captures the un-foreshortened
        // width and gives a more stable magnitude.
        let target_span = running_shoulder_x_span_max.max(span_xz);
        let cos_yaw = (bx / target_span).clamp(-1.0, 1.0);
        let yaw_magnitude = cos_yaw.acos();
        // Sign convention: positive yaw = standard right-hand-rule
        // CCW rotation around +Y (= subject turns to **her left**,
        // bringing her right side toward camera). For the source's
        // unmirrored Y-up frame this corresponds to L.shoulder going
        // *back* (-Z) and R.shoulder coming *forward* (+Z), i.e.
        // bz = L.z - R.z is **negative**. Hence `yaw_sign = -sign(bz)`.
        // bz = 0 (exact frontal/back) yields zero magnitude so the
        // sign choice is moot; pick `+` deterministically.
        let yaw_sign = if bz <= 0.0 { 1.0 } else { -1.0 };
        return Some(yaw_sign * yaw_magnitude);
    }
    None
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
// 2-bone IK
// ---------------------------------------------------------------------------

/// Apply two-bone IK to all four limb chains (left/right arm,
/// left/right leg). Each call replaces the source skeleton's
/// `middle` joint position (elbow / knee) with one that's
/// anatomically consistent with the avatar's rest-pose bone lengths
/// and the trusted base/tip endpoints.
///
/// **Why limb-chain IK at all**
///
/// The downstream bone-rotation loop in `solve_avatar_pose` consumes
/// *direction-only* signals: for each bone it computes
/// `dir = normalize(tip - base)` and rotates the avatar's bone of
/// fixed (rest-pose) length to point in that direction. There is no
/// length-preservation or end-effector constraint downstream, so a
/// wrong middle-keypoint position propagates verbatim into the
/// avatar: the upper-arm rotates to point at the (wrong) elbow, the
/// forearm rotates from there to point at the (correctish) wrist,
/// and the avatar's hand lands at avatar_shoulder + L_upper *
/// dir_upper + L_lower * dir_lower — geometrically inconsistent with
/// where the *real* hand was in the source frame.
///
/// The validation case that exposed this: subject raises right hand
/// to upper chest (palm to camera). RTMW3D detects shoulder and
/// wrist 2D positions correctly, but the elbow keypoint lands at
/// hip-mid level — well below where it physically is. The implied
/// source bone lengths come out forearm > upper-arm (anatomically
/// reversed), and the avatar's rendered hand lands ~at the hip
/// instead of upper chest because of the bad elbow direction
/// dragging the upper-arm down.
///
/// **Why arms get pole = -Y, legs get pole = +Z**
///
/// The IK has two solutions for the elbow/knee position (both sides
/// of the chord). The pole hint disambiguates. Default arm pose has
/// the elbow bending "down/back" relative to the shoulder-wrist
/// chord (think wave gesture, hand-on-hip, arms-folded — all bend
/// the elbow toward -Y in source space). Default leg pose has the
/// knee bending forward (running, sitting, kneeling — all bend the
/// knee toward +Z, i.e. away from the camera in source-space
/// convention where +Z is toward camera). These hints break down for
/// extreme poses (back-handed wave, knee bending sideways), but
/// `apply_2bone_ik_chain` adapts the hint toward the original
/// keypoint position when one is available — so a slightly-off
/// elbow detection still preserves the side it was on.
fn apply_2bone_ik_arms_and_legs(
    source: &mut SourceSkeleton,
    rest_world: &[WorldXform],
    humanoid: &HumanoidMap,
    threshold: f32,
    body_yaw_source: f32,
) {
    // Avatar-to-source unit conversion: source positions are in
    // image-normalised coordinates while rest_world bone lengths are
    // in avatar-space (typically metres for VRM 1.0). Use shoulder
    // span as the calibration: it's the most reliably-detected pair
    // across pose variations and is essentially the same anatomical
    // measurement on both sides of the conversion.
    let Some(scale) = compute_avatar_to_source_scale(source, rest_world, humanoid)
    else {
        return;
    };

    // Rotate body-local pole hints by the source body yaw so they
    // describe the limb's bend direction in **source space** for the
    // current frame. At rest the subject faces the camera so body-
    // local +Z (forward) coincides with source +Z; once the body
    // rotates around Y this no longer holds and the static hint pulls
    // the knee or elbow toward the wrong half-space.
    let cos_y = body_yaw_source.cos();
    let sin_y = body_yaw_source.sin();
    // R_y(yaw) · (0, 0, 1) = (sin yaw, 0, cos yaw). Body-local +Z
    // becomes this in source space.
    let leg_pole = [sin_y, 0.0, cos_y];
    // Arms: body-local -Y (elbow bends down). Y is the rotation axis
    // so it's invariant under yaw — same vector in source space.
    let arm_pole = [0.0, -1.0, 0.0];

    apply_2bone_ik_chain(
        source,
        rest_world,
        humanoid,
        HumanoidBone::LeftShoulder,
        HumanoidBone::LeftLowerArm,
        HumanoidBone::LeftHand,
        arm_pole,
        threshold,
        scale,
    );
    apply_2bone_ik_chain(
        source,
        rest_world,
        humanoid,
        HumanoidBone::RightShoulder,
        HumanoidBone::RightLowerArm,
        HumanoidBone::RightHand,
        arm_pole,
        threshold,
        scale,
    );

    apply_2bone_ik_chain(
        source,
        rest_world,
        humanoid,
        HumanoidBone::LeftUpperLeg,
        HumanoidBone::LeftLowerLeg,
        HumanoidBone::LeftFoot,
        leg_pole,
        threshold,
        scale,
    );
    apply_2bone_ik_chain(
        source,
        rest_world,
        humanoid,
        HumanoidBone::RightUpperLeg,
        HumanoidBone::RightLowerLeg,
        HumanoidBone::RightFoot,
        leg_pole,
        threshold,
        scale,
    );
}

/// Single-chain 2-bone IK: shoulder → elbow → wrist (or hip → knee →
/// ankle). Replaces `middle`'s position in `source` with one that
/// satisfies the avatar's rest-pose bone-length ratios while keeping
/// the base + tip endpoints fixed.
///
/// No-op when:
/// - The base or tip is missing or below `threshold` confidence
/// - Avatar bone lengths can't be looked up from `rest_world`
/// - Chord (base→tip) is degenerately short (< 1mm) or exceeds the
///   maximum reach by more than 5%; the latter typically means the
///   keypoint detector wildly misplaced one of the endpoints, and IK
///   forced into a degenerate elbow at the chord midpoint would look
///   worse than leaving the original wrong elbow alone.
/// - The pole hint is parallel to the chord (e.g. arm pointing
///   straight down with pole = -Y), so the perpendicular component
///   collapses and the elbow's bend direction is undetermined.
///
/// `pole_hint_source` is in source coords. The function projects it
/// onto the plane perpendicular to the chord and uses the projection
/// as the elbow's bend direction. If the original middle position is
/// available, the hint is biased toward whichever side of the chord
/// the original was on — so a slightly-off keypoint that's on the
/// correct side still produces the right bend direction.
#[allow(clippy::too_many_arguments)]
fn apply_2bone_ik_chain(
    source: &mut SourceSkeleton,
    rest_world: &[WorldXform],
    humanoid: &HumanoidMap,
    base: HumanoidBone,
    middle: HumanoidBone,
    tip: HumanoidBone,
    pole_hint_source: Vec3,
    threshold: f32,
    avatar_to_source_scale: f32,
) {
    let Some(base_joint) = source.joints.get(&base).copied() else {
        return;
    };
    if base_joint.confidence < threshold {
        return;
    }
    let Some(tip_joint) = source.joints.get(&tip).copied() else {
        return;
    };
    if tip_joint.confidence < threshold {
        return;
    }
    let base_pos = base_joint.position;
    let tip_pos = tip_joint.position;

    // Avatar's anatomical bone-length **ratio** from rest pose. We
    // deliberately don't use avatar bone lengths *directly*: the
    // global avatar-to-source scale derived from shoulder span is
    // accurate at the shoulder line but doesn't transfer cleanly to
    // the leg chain (a person's leg-to-shoulder ratio varies by ±10%
    // per individual, and per-frame perspective compresses it
    // differently). Forcing avatar-scaled bone lengths onto a
    // standing leg whose source-implied length doesn't match (chord
    // < scaled max_reach) creates a phantom bend at the knee — the
    // IK has to bend *somewhere* to fit the bones into the chord.
    //
    // Instead, take the source's **implied total chain length**
    // (l_upper_implied + l_lower_implied) and re-distribute it
    // using only the avatar's *ratio*. This preserves the source's
    // body-scale signal while fixing the proportion error that
    // RTMW3D's keypoint-detection inaccuracy introduced.
    let Some(base_node) = humanoid.bone_map.get(&base).copied() else {
        return;
    };
    let Some(middle_node) = humanoid.bone_map.get(&middle).copied() else {
        return;
    };
    let Some(tip_node) = humanoid.bone_map.get(&tip).copied() else {
        return;
    };
    let base_idx = base_node.0 as usize;
    let middle_idx = middle_node.0 as usize;
    let tip_idx = tip_node.0 as usize;
    if base_idx >= rest_world.len()
        || middle_idx >= rest_world.len()
        || tip_idx >= rest_world.len()
    {
        return;
    }
    let l_upper_avatar = vec3_length(&vec3_sub(
        &rest_world[middle_idx].position,
        &rest_world[base_idx].position,
    ));
    let l_lower_avatar = vec3_length(&vec3_sub(
        &rest_world[tip_idx].position,
        &rest_world[middle_idx].position,
    ));
    if l_upper_avatar < 1e-4 || l_lower_avatar < 1e-4 {
        return;
    }
    let avatar_ratio_upper_total = l_upper_avatar / (l_upper_avatar + l_lower_avatar);

    // Source's implied total chain length, derived from the
    // currently-detected (possibly inaccurate) middle position.
    // We use this as the budget the IK must redistribute over the
    // upper + lower bones. The middle keypoint's *position* may be
    // wrong, but its summed-distance-from-base-via-itself-to-tip
    // tracks the actual subject geometry better than any avatar
    // scaling does — see the rationale block above.
    let original_middle = source.joints.get(&middle).map(|j| j.position);
    let total_implied = if let Some(orig) = original_middle {
        vec3_length(&vec3_sub(&orig, &base_pos))
            + vec3_length(&vec3_sub(&tip_pos, &orig))
    } else {
        return; // No middle keypoint to derive total from; can't IK.
    };
    let l_upper = total_implied * avatar_ratio_upper_total;
    let l_lower = total_implied - l_upper;

    // The avatar_to_source_scale parameter is intentionally unused
    // here — see the rationale block above. Kept on the signature so
    // callers don't have to recompute it per chain in the future
    // (e.g. for a metric-aware IK variant if root translation needs
    // it).
    let _ = avatar_to_source_scale;

    // Chord = base → tip in source coords.
    let chord = vec3_sub(&tip_pos, &base_pos);
    let chord_len = vec3_length(&chord);
    let max_reach = l_upper + l_lower;
    if chord_len < 1e-3 || chord_len > max_reach * 1.05 {
        return;
    }
    // Clamp slightly under max_reach to avoid sin_alpha → 0 numerical
    // collapse at full extension (pole_unit becomes meaningless).
    let chord_eff = chord_len.min(max_reach * 0.999);
    let chord_unit = [
        chord[0] / chord_len,
        chord[1] / chord_len,
        chord[2] / chord_len,
    ];

    // Law of cosines: angle α at base between chord and base→middle.
    let cos_alpha = ((l_upper * l_upper + chord_eff * chord_eff
        - l_lower * l_lower)
        / (2.0 * l_upper * chord_eff))
        .clamp(-1.0, 1.0);
    let sin_alpha = (1.0 - cos_alpha * cos_alpha).max(0.0).sqrt();

    // Project the static pole hint onto the plane perpendicular to
    // the chord — that gives the actual bend direction in source
    // space. We deliberately don't adapt the hint to the original
    // middle's offset: when the keypoint is correctly detected and
    // sits near the chord (e.g. arm hanging straight at side, chord
    // ≈ -Y, original elbow with tiny perpendicular component), the
    // adaptive hint is dominated by detection noise and produces a
    // wrong-direction bend that visibly distorts the avatar's pose.
    // The static hint stays consistent and degrades gracefully:
    // for chords parallel to the hint (arm straight down with hint
    // = -Y) the perpendicular collapses and IK aborts, leaving the
    // already-correct keypoint untouched.
    let dot = vec3_dot(&pole_hint_source, &chord_unit);
    let pole_perp = [
        pole_hint_source[0] - chord_unit[0] * dot,
        pole_hint_source[1] - chord_unit[1] * dot,
        pole_hint_source[2] - chord_unit[2] * dot,
    ];
    let pole_len = vec3_length(&pole_perp);
    if pole_len < 1e-4 {
        // Pole parallel to chord; bend direction undetermined.
        return;
    }
    let pole_unit = [
        pole_perp[0] / pole_len,
        pole_perp[1] / pole_len,
        pole_perp[2] / pole_len,
    ];

    let middle_pos = [
        base_pos[0] + l_upper * (cos_alpha * chord_unit[0] + sin_alpha * pole_unit[0]),
        base_pos[1] + l_upper * (cos_alpha * chord_unit[1] + sin_alpha * pole_unit[1]),
        base_pos[2] + l_upper * (cos_alpha * chord_unit[2] + sin_alpha * pole_unit[2]),
    ];

    if let Some(j) = source.joints.get_mut(&middle) {
        j.position = middle_pos;
    }
}

/// Avatar-to-source coordinate-scale factor, derived from the ratio
/// of source-vs-avatar shoulder spans on the **horizontal X axis only**.
/// Used by `apply_2bone_ik_chain` to convert avatar-unit bone lengths
/// (metres for VRM 1.0) into source-coordinate-unit bone lengths so
/// the IK math stays in one consistent space.
///
/// X-only (rather than full 3D) because the source skeleton's Z is
/// the (relative) RTMW3D nz signal and includes per-frame depth
/// noise / bias, which inflates the 3D shoulder span estimate.
/// Avatar rest-pose shoulders sit at the same Z (T-pose), so the
/// avatar side has no Z component anyway — comparing 3D source
/// against 1D avatar would systematically over-scale and inflate
/// the IK bone-length, pushing the elbow further out than it
/// belongs.
///
/// Returns `None` when either shoulder is missing or the spans are
/// degenerate (subject in extreme side profile, bad detection
/// frame). Caller skips IK on that frame in either case.
fn compute_avatar_to_source_scale(
    source: &SourceSkeleton,
    rest_world: &[WorldXform],
    humanoid: &HumanoidMap,
) -> Option<f32> {
    let l_node = humanoid.bone_map.get(&HumanoidBone::LeftShoulder)?;
    let r_node = humanoid.bone_map.get(&HumanoidBone::RightShoulder)?;
    let l_idx = l_node.0 as usize;
    let r_idx = r_node.0 as usize;
    if l_idx >= rest_world.len() || r_idx >= rest_world.len() {
        return None;
    }
    let avatar_span_x =
        (rest_world[l_idx].position[0] - rest_world[r_idx].position[0]).abs();
    if avatar_span_x < 1e-4 {
        return None;
    }
    let l_src = source.joints.get(&HumanoidBone::LeftShoulder)?;
    let r_src = source.joints.get(&HumanoidBone::RightShoulder)?;
    let source_span_x = (l_src.position[0] - r_src.position[0]).abs();
    if source_span_x < 1e-4 {
        return None;
    }
    Some(source_span_x / avatar_span_x)
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
        joint.position = filt.apply(joint.position, dt);
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
    dt: f32,
) {
    let Some(orient) = source_orientation else {
        return;
    };
    if orient.confidence < params.joint_confidence_threshold {
        return;
    }

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
    let twist_axis = lower_arm_idx.and_then(|la_idx| {
        let lower_arm_pos = current_world[la_idx].position;
        let wrist_pos = current_world[wrist_idx].position;
        let raw = vec3_sub(&wrist_pos, &lower_arm_pos);
        let normalized = vec3_normalize(&raw);
        if normalized == [0.0; 3] { None } else { Some(normalized) }
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
        let lower_arm_prev_local = local_transforms[la_idx].rotation;
        local_transforms[la_idx].rotation =
            quat_slerp_short(&lower_arm_prev_local, &lower_arm_local_target, blend);
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
        let prev_local_rot = local_transforms[wrist_idx].rotation;
        local_transforms[wrist_idx].rotation =
            quat_slerp_short(&prev_local_rot, &hand_local_target, blend);
        let updated_world =
            quat_mul(&actual_lower_arm_world, &local_transforms[wrist_idx].rotation);
        current_world[wrist_idx].rotation = updated_world;
    } else {
        // Fallback: forearm degenerate or no parent — apply the full
        // delta to the Hand alone, preserving the pre-redistribution
        // behaviour for edge cases.
        let new_local_rot =
            quat_normalize(&quat_mul(&quat_conjugate(&parent_world_rot), &new_world_rot));
        let prev_local_rot = local_transforms[wrist_idx].rotation;
        local_transforms[wrist_idx].rotation =
            quat_slerp_short(&prev_local_rot, &new_local_rot, blend);
        let updated_world =
            quat_mul(&parent_world_rot, &local_transforms[wrist_idx].rotation);
        current_world[wrist_idx].rotation = updated_world;
    }
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
    face_confidence_threshold: f32,
) -> Vec<ResolvedExpressionWeight> {
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
            let blended = prev_w + expression_blend * (raw - prev_w);
            Some(ResolvedExpressionWeight {
                name: expr_def.name.clone(),
                weight: blended.clamp(0.0, 1.0),
            })
        })
        .collect()
}

#[cfg(test)]
mod body_yaw_tests {
    use super::*;
    use crate::tracking::source_skeleton::SourceJoint;

    fn put(sk: &mut SourceSkeleton, bone: HumanoidBone, pos: [f32; 3], confidence: f32) {
        sk.joints.insert(bone, SourceJoint { position: pos, confidence, metric_depth_m: None });
    }

    /// Three-quarter view (~45°): shoulder Δz is tiny (+0.004) but
    /// **correctly signed** for the yaw direction. The hip pair's Δz at
    /// the same pose carries the **opposite sign** (-0.014) per the
    /// plan/body-twist-yaw-investigation.md row 045°, so any averaging
    /// would flip the result. This regression test pins us to "shoulders
    /// first": the magnitude is small but the sign must be right, never
    /// wrong. Sign convention: bz > 0 (LEFT shoulder forward) ⇒ subject
    /// turned to her RIGHT ⇒ R_y CW from above ⇒ negative yaw.
    #[test]
    fn three_quarter_view_keeps_shoulder_sign() {
        let mut sk = SourceSkeleton::empty(0);
        put(&mut sk, HumanoidBone::LeftShoulder, [0.059, 0.486, -0.083], 1.0);
        put(&mut sk, HumanoidBone::RightShoulder, [-0.155, 0.486, -0.087], 1.0);
        put(&mut sk, HumanoidBone::LeftUpperLeg, [0.046, -0.004, -0.007], 1.0);
        put(&mut sk, HumanoidBone::RightUpperLeg, [-0.046, 0.004, 0.007], 1.0);

        let mut span_max = 0.0;
        let yaw = compute_body_yaw_3d(&sk, 0.1, &mut span_max).expect("shoulder pair clears threshold");
        // Shoulder-only: bx=+0.214, bz=+0.004 → small magnitude (~1°),
        // sign negative (subject turned right).
        assert!(
            yaw < 0.0 && yaw > -0.05,
            "45° shoulder yaw should be small -ε, got {yaw}"
        );
    }

    /// 90° side profile: large yaw (≈ -48°). Subject's left shoulder
    /// forward (bz > 0) ⇒ subject turned to her right ⇒ negative yaw.
    /// Sanity-check that the function recovers a real yaw signal once
    /// Δz comes off the noise floor.
    #[test]
    fn side_profile_90_gives_large_negative_yaw() {
        let mut sk = SourceSkeleton::empty(0);
        put(&mut sk, HumanoidBone::LeftShoulder, [0.054, 0.534, 0.019], 1.0);
        put(&mut sk, HumanoidBone::RightShoulder, [-0.029, 0.531, -0.075], 1.0);

        let mut span_max = 0.0;
        let yaw_deg = compute_body_yaw_3d(&sk, 0.1, &mut span_max)
            .expect("shoulders")
            .to_degrees();
        assert!(
            (-55.0..-45.0).contains(&yaw_deg),
            "90° side profile expected ≈-48°, got {yaw_deg:.1}°"
        );
    }

    /// 270° side profile: large yaw (≈ -62°). Different magnitude than
    /// 90° because RTMW3D's Z asymmetry biases the two side views
    /// unequally; both lie in the same negative quadrant.
    #[test]
    fn side_profile_270_gives_large_negative_yaw() {
        let mut sk = SourceSkeleton::empty(0);
        put(&mut sk, HumanoidBone::LeftShoulder, [0.031, 0.523, 0.012], 1.0);
        put(&mut sk, HumanoidBone::RightShoulder, [-0.017, 0.523, -0.078], 1.0);

        let mut span_max = 0.0;
        let yaw_deg = compute_body_yaw_3d(&sk, 0.1, &mut span_max)
            .expect("shoulders")
            .to_degrees();
        assert!(
            (-68.0..-58.0).contains(&yaw_deg),
            "270° side profile expected ≈-62°, got {yaw_deg:.1}°"
        );
    }

    /// Back-facing (180°): bx is strongly negative so atan2 lands near
    /// ±π regardless of Δz noise. Pin |yaw| > 170° to confirm the wrap.
    #[test]
    fn back_facing_yaw_near_pi() {
        let mut sk = SourceSkeleton::empty(0);
        put(&mut sk, HumanoidBone::LeftShoulder, [-0.166, 0.504, 0.061], 1.0);
        put(&mut sk, HumanoidBone::RightShoulder, [0.140, 0.520, 0.047], 1.0);

        let mut span_max = 0.0;
        let yaw_deg = compute_body_yaw_3d(&sk, 0.1, &mut span_max)
            .expect("shoulders")
            .to_degrees();
        assert!(
            yaw_deg.abs() > 170.0,
            "back-facing yaw should be near ±180°, got {yaw_deg:.1}°"
        );
    }

    /// Both pairs confident but with **disagreeing sign** on Δz — the
    /// pathology that broke Option A averaging. Shoulders give a small
    /// negative yaw (subject turned right); hips alone would give a
    /// positive one (opposite sign). The shoulders-first contract
    /// requires the result to come out negative.
    #[test]
    fn shoulders_authoritative_when_hips_disagree() {
        let mut sk = SourceSkeleton::empty(0);
        // Shoulder Δz=+0.004, Δx=+0.214 → small magnitude, sign neg.
        put(&mut sk, HumanoidBone::LeftShoulder, [0.107, 0.486, -0.085], 1.0);
        put(&mut sk, HumanoidBone::RightShoulder, [-0.107, 0.486, -0.089], 1.0);
        // Hip Δz=-0.014, Δx=+0.092 → opposite sign (positive).
        put(&mut sk, HumanoidBone::LeftUpperLeg, [0.046, 0.0, -0.007], 1.0);
        put(&mut sk, HumanoidBone::RightUpperLeg, [-0.046, 0.0, 0.007], 1.0);

        let mut span_max = 0.0;
        let yaw = compute_body_yaw_3d(&sk, 0.1, &mut span_max).expect("shoulders win");
        assert!(
            yaw < 0.0,
            "shoulders must dominate even when hips disagree in sign, got {yaw}"
        );
    }

    /// Shoulder-only fallback (lower-body cropped out of frame). Front
    /// pose has bx large and bz near zero → yaw ≈ 0.
    #[test]
    fn shoulder_only_falls_back_to_single_pair() {
        let mut sk = SourceSkeleton::empty(0);
        put(&mut sk, HumanoidBone::LeftShoulder, [0.21, 1.4, 0.0], 1.0);
        put(&mut sk, HumanoidBone::RightShoulder, [-0.21, 1.4, 0.0], 1.0);

        let mut span_max = 0.0;
        let yaw = compute_body_yaw_3d(&sk, 0.1, &mut span_max).expect("shoulder-only fallback");
        assert!(
            yaw.abs() < 0.05,
            "front-pose shoulder span should give ~0 yaw, got {yaw}"
        );
    }

    /// No torso joints visible at all → `None` so the caller doesn't
    /// snap the avatar to a stale value.
    #[test]
    fn no_torso_joints_returns_none() {
        let sk = SourceSkeleton::empty(0);
        let mut span_max = 0.0;
        assert!(compute_body_yaw_3d(&sk, 0.1, &mut span_max).is_none());
    }

    /// Foreshortening recovery: once the running max of `|Δx|` has
    /// captured the subject's true frontal shoulder width, a subsequent
    /// rotated frame should report a **larger** yaw magnitude than the
    /// pure-atan2 first call would. This is the live-tracking win — we
    /// trade single-shot atan2 (matches first-frame behavior in tests
    /// above) for cumulative magnitude correction across a session.
    #[test]
    fn running_max_amplifies_under_rotated_yaw() {
        let mut span_max = 0.0;

        // Frame 1: subject roughly frontal — primes the running max
        // with a wide |Δx|.
        let mut frontal = SourceSkeleton::empty(0);
        put(&mut frontal, HumanoidBone::LeftShoulder, [0.20, 0.5, 0.0], 1.0);
        put(&mut frontal, HumanoidBone::RightShoulder, [-0.20, 0.5, 0.0], 1.0);
        let _ = compute_body_yaw_3d(&frontal, 0.1, &mut span_max);
        assert!(
            (span_max - 0.40).abs() < 1e-4,
            "frontal frame should prime running max to 0.40, got {span_max}"
        );

        // Frame 2: same subject rotated; bx shrinks (foreshortening)
        // but bz is small / underestimated by RTMW3D. Pure atan2 would
        // give ≈ atan2(0.05, 0.20) ≈ 14° magnitude, but with running
        // max we recover ≈ acos(0.20/0.40) = 60°.
        let mut rotated = SourceSkeleton::empty(0);
        put(&mut rotated, HumanoidBone::LeftShoulder, [0.10, 0.5, -0.025], 1.0);
        put(&mut rotated, HumanoidBone::RightShoulder, [-0.10, 0.5, 0.025], 1.0);
        let yaw_deg = compute_body_yaw_3d(&rotated, 0.1, &mut span_max)
            .expect("shoulders")
            .to_degrees();
        // Foreshortening recovery: |bx|=0.20, target=0.40,
        // cos⁻¹(0.5) = 60°. Sign convention: bz<0 (LEFT back, RIGHT
        // forward) ⇒ subject turned LEFT (CCW) ⇒ POSITIVE yaw.
        assert!(
            (yaw_deg - 60.0).abs() < 2.0,
            "foreshortening recovery should yield ≈ +60°, got {yaw_deg:.1}°"
        );
    }

    /// Shoulders present but below the confidence threshold while hips
    /// are confident: the function falls through to the hip pair. This
    /// is the asymmetric robustness gain over a shoulder-only baseline.
    #[test]
    fn low_confidence_shoulders_falls_through_to_hips() {
        let mut sk = SourceSkeleton::empty(0);
        put(&mut sk, HumanoidBone::LeftShoulder, [0.21, 1.4, 0.0], 0.05);
        put(&mut sk, HumanoidBone::RightShoulder, [-0.21, 1.4, 0.0], 0.05);
        put(&mut sk, HumanoidBone::LeftUpperLeg, [0.18, 0.0, 0.05], 1.0);
        put(&mut sk, HumanoidBone::RightUpperLeg, [-0.18, 0.0, -0.05], 1.0);

        let mut span_max = 0.0;
        let yaw = compute_body_yaw_3d(&sk, 0.1, &mut span_max).expect("hip pair carries the signal");
        // Hip-only: bx=0.36, bz=+0.10 (LEFT forward → subject right turn)
        // ⇒ negative yaw, magnitude ≈ atan2(0.10, 0.36) ≈ 0.27 rad.
        assert!(
            (yaw - (-0.27)).abs() < 0.02,
            "hip-only yaw mismatch: {yaw}"
        );
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
}
