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
        for i in 0..3 {
            self.vel[i] += alpha_d * (raw_vel[i] - self.vel[i]);
        }
        // Position cutoff scales with velocity magnitude — high-speed
        // motion gets a higher cutoff (less smoothing, more responsive).
        let speed = (self.vel[0] * self.vel[0]
            + self.vel[1] * self.vel[1]
            + self.vel[2] * self.vel[2])
            .sqrt();
        let cutoff = ONE_EURO_MIN_CUTOFF_HZ + ONE_EURO_BETA * speed;
        let alpha = one_euro_alpha(dt, cutoff);
        for i in 0..3 {
            self.pos[i] += alpha * (raw[i] - self.pos[i]);
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
    for &(bone, tip) in DRIVEN_BONES {
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
    // threshold.
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

/// Body yaw, derived from the 3D shoulder line.
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
/// Returns `None` when either shoulder is below the confidence
/// threshold so the caller can skip the rotation entirely instead of
/// snapping the body to whatever stale value comes through.
fn compute_body_yaw_3d(source: &SourceSkeleton, threshold: f32) -> Option<f32> {
    let l = source.joints.get(&HumanoidBone::LeftShoulder)?;
    let r = source.joints.get(&HumanoidBone::RightShoulder)?;
    if l.confidence < threshold || r.confidence < threshold {
        return None;
    }
    let bx = l.position[0] - r.position[0];
    let bz = l.position[2] - r.position[2];
    // Dead-zone: with both shoulders close together (subject in
    // exact side profile), bx is tiny and atan2 amplifies any noise
    // into wild yaw flips. Require a minimum span before we trust
    // the angle.
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
