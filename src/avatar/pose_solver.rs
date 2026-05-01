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
    quat_normalize, quat_rotate_vec3, vec3_cross, vec3_normalize, vec3_sub,
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
    let source_owned = preprocess_source(source, state, dt, params);
    let source = &source_owned;

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

    // Body root yaw: rotate Hips around Y so the avatar's torso faces
    // the same way as the subject. With a 3D-native tracker this falls
    // out cleanly from the shoulder line in the XZ plane: at rest the
    // shoulder line points along +X (avatar's left → camera-right),
    // and as the subject rotates around their vertical axis the line
    // tilts in XZ so its yaw against the rest +X axis encodes the
    // body's facing direction directly.
    if let Some(body_yaw) = compute_body_yaw_3d(source, params.joint_confidence_threshold) {
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

        let source_dir = vec3_normalize(&vec3_sub(&source_tip_pos, &source_base.position));
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
/// **Sample combination.** RTMW3D's per-joint depth resolution for two
/// close points collapses Δz to the noise floor at ~45°/315° three-
/// quarter views (see `plan/body-twist-yaw-investigation.md`): the
/// shoulder pair alone reports Δz ≈ 0.004 in selfie space, yielding
/// avatar yaw of ~1° instead of the expected ~45°. We mitigate by
/// also reading the hip pair as an independent vote on torso-width
/// orientation: hips sit anatomically below the shoulders so their
/// per-joint depth noise is uncorrelated at this scale, and a
/// confidence-weighted mean of the two `(bx, bz)` samples roughly
/// halves the noise floor. When only one pair clears the confidence
/// gate the function falls back to that single pair.
///
/// Returns `None` when neither pair clears the confidence threshold
/// so the caller can skip the rotation entirely instead of snapping
/// the body to whatever stale value comes through.
fn compute_body_yaw_3d(source: &SourceSkeleton, threshold: f32) -> Option<f32> {
    use HumanoidBone::*;
    let pairs = [
        (LeftShoulder, RightShoulder),
        (LeftUpperLeg, RightUpperLeg),
    ];
    let mut sum_bx = 0.0_f32;
    let mut sum_bz = 0.0_f32;
    let mut sum_w = 0.0_f32;
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
        let weight = l.confidence.min(r.confidence);
        sum_bx += (l.position[0] - r.position[0]) * weight;
        sum_bz += (l.position[2] - r.position[2]) * weight;
        sum_w += weight;
    }
    if sum_w <= 0.0 {
        return None;
    }
    let bx = sum_bx / sum_w;
    let bz = sum_bz / sum_w;
    // Dead-zone: with the torso-width vector tiny (subject in exact
    // side profile), atan2 amplifies any noise into wild yaw flips.
    // Require a minimum span before we trust the angle.
    if (bx * bx + bz * bz).sqrt() < 0.05 {
        return None;
    }
    Some(bz.atan2(bx))
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
    let new_local_rot =
        quat_normalize(&quat_mul(&quat_conjugate(&parent_world_rot), &new_world_rot));

    let prev_local_rot = local_transforms[wrist_idx].rotation;
    local_transforms[wrist_idx].rotation = quat_slerp_short(
        &prev_local_rot,
        &new_local_rot,
        dt_aware_blend(params.rotation_blend, dt),
    );

    // Refresh the world cache so the finger entries about to run
    // see the wrist's full orientation, not the stale rest pose.
    let updated_world =
        quat_mul(&parent_world_rot, &local_transforms[wrist_idx].rotation);
    current_world[wrist_idx].rotation = updated_world;
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
        sk.joints.insert(bone, SourceJoint { position: pos, confidence });
    }

    /// Three-quarter view (~45°): the shoulder-only signal collapses to
    /// the SimCC noise floor (Δz ≈ 0.004 between shoulders) and the old
    /// implementation produced sub-degree yaw, but the hip Δz is an
    /// order of magnitude larger at this pose so the combined signal
    /// must register a clearly non-zero yaw. Numbers here are the
    /// measured 45° row from `plan/body-twist-yaw-investigation.md`.
    #[test]
    fn three_quarter_view_uses_hip_signal_when_shoulders_collapse() {
        let mut sk = SourceSkeleton::empty(0);
        put(&mut sk, HumanoidBone::LeftShoulder, [0.059, 0.486, -0.083], 1.0);
        put(&mut sk, HumanoidBone::RightShoulder, [-0.155, 0.486, -0.087], 1.0);
        put(&mut sk, HumanoidBone::LeftUpperLeg, [0.046, -0.004, -0.007], 1.0);
        put(&mut sk, HumanoidBone::RightUpperLeg, [-0.046, 0.004, 0.007], 1.0);

        let yaw = compute_body_yaw_3d(&sk, 0.1).expect("torso pair clears threshold");
        let yaw_deg = yaw.to_degrees().abs();
        // Old shoulder-only output for this pose was ~0.93°; the
        // weighted hip+shoulder mean must lift it well past that.
        assert!(
            yaw_deg > 1.5,
            "combined-pair yaw should beat shoulder-only floor of ~1°, got {yaw_deg:.2}°"
        );
    }

    /// When only the shoulder pair is present (lower-body cropped out
    /// of frame), the function must still return that pair's reading
    /// rather than `None`. Front pose has bx large and bz near zero so
    /// yaw should be ≈0.
    #[test]
    fn shoulder_only_falls_back_to_single_pair() {
        let mut sk = SourceSkeleton::empty(0);
        put(&mut sk, HumanoidBone::LeftShoulder, [0.21, 1.4, 0.0], 1.0);
        put(&mut sk, HumanoidBone::RightShoulder, [-0.21, 1.4, 0.0], 1.0);

        let yaw = compute_body_yaw_3d(&sk, 0.1).expect("shoulder-only fallback");
        assert!(yaw.abs() < 0.05, "front-pose shoulder span should give ~0 yaw, got {yaw}");
    }

    /// No torso joints visible at all → `None` so the caller doesn't
    /// snap the avatar to a stale value.
    #[test]
    fn no_torso_joints_returns_none() {
        let sk = SourceSkeleton::empty(0);
        assert!(compute_body_yaw_3d(&sk, 0.1).is_none());
    }

    /// Both shoulders present but below the confidence threshold while
    /// hips are confident: the function must skip the shoulders and
    /// return the hip-only reading.
    #[test]
    fn low_confidence_pair_is_skipped() {
        let mut sk = SourceSkeleton::empty(0);
        put(&mut sk, HumanoidBone::LeftShoulder, [0.21, 1.4, 0.0], 0.05);
        put(&mut sk, HumanoidBone::RightShoulder, [-0.21, 1.4, 0.0], 0.05);
        put(&mut sk, HumanoidBone::LeftUpperLeg, [0.18, 0.0, 0.05], 1.0);
        put(&mut sk, HumanoidBone::RightUpperLeg, [-0.18, 0.0, -0.05], 1.0);

        let yaw = compute_body_yaw_3d(&sk, 0.1).expect("hip pair carries the signal");
        // Hip-only: bx=0.36, bz=0.10 → atan2(0.10, 0.36) ≈ 0.27 rad.
        assert!((yaw - 0.27).abs() < 0.02, "hip-only yaw mismatch: {yaw}");
    }
}
