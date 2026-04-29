//! Solve avatar bone rotations from a [`SourceSkeleton`].
//!
//! The solver is driven by three ideas:
//!
//! 1. **Rest-relative.** Every output rotation is expressed as a delta from
//!    the avatar's rest pose, so models whose bones have non-identity rest
//!    rotations (most real VRMs) are not destroyed by the write.
//!
//! 2. **Direction matching.** For chain bones (upper/lower arm, leg, spine)
//!    we read the bone's *rest world direction* (from its world position to
//!    its tip's world position) and rotate it to match the source skeleton's
//!    observed direction in camera space. The minimum-arc quaternion is
//!    written as the world-space delta; it is then conjugated into the
//!    bone's parent-local frame so the result respects the skeleton
//!    hierarchy.
//!
//! 3. **Face pose is independent.** The head rotation comes from facial
//!    keypoints (yaw/pitch/roll), not from the nose-to-shoulder vector.
//!    It is applied as a rest-relative local delta on the Head bone.
//!
//! The solver assumes `local_transforms` has already been reset to the
//! skeleton's rest pose (via `AvatarInstance::build_base_pose`). It only
//! writes rotations; translations and scales are left untouched.

use std::collections::HashMap;

use crate::asset::{HumanoidBone, HumanoidMap, NodeId, Quat, SkeletonAsset, Transform, Vec3};
use crate::math_utils::{
    quat_conjugate, quat_from_euler_ypr, quat_from_vectors, quat_mul, quat_normalize,
    quat_rotate_vec3, vec3_normalize, vec3_sub,
};
use crate::tracking::source_skeleton::{FacePose, SourceSkeleton};

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
    }
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

    // --- 1. Body chain: direction-match each driven bone. ---
    for &(bone, tip) in DRIVEN_BONES {
        let Some(source_base) = get_joint_2d(source, bone) else {
            continue;
        };
        let source_tip_2d = match tip {
            Tip::Joint(tip_bone) => get_joint_2d(source, tip_bone),
            Tip::ShoulderMidpoint => shoulder_midpoint(source),
            Tip::Fingertip => source.fingertips.get(&bone).map(|j| JointRef {
                position: j.position,
                confidence: j.confidence,
            }),
        };
        let Some(source_tip) = source_tip_2d else {
            continue;
        };
        if source_base.confidence < params.joint_confidence_threshold
            || source_tip.confidence < params.joint_confidence_threshold
        {
            continue;
        }

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

        // Recover the source direction with Z (depth toward/away from
        // camera). The 2D detector only gives dx/dy in the image plane,
        // but a fully T-posed arm pointing toward the camera projects to
        // a tiny 2D vector — without the Z component the solver folds
        // the avatar's arm to lie in the image plane regardless.
        //
        // Z is recovered from foreshortening: if the bone's
        // un-foreshortened 2D length is `L`, and we observe length `d`,
        // then `|Z| = sqrt(L² − d²)` in source units. `L` comes from
        // (a) per-frame anthropometric scaling against torso 2D length,
        // and (b) running max of observed length on this bone in
        // `state.bone_max_2d` — whichever is larger. The previous
        // approach used the avatar's rest-pose bone length as `L`, which
        // breaks for stylised rigs whose proportions differ from the
        // subject; this one calibrates against the *subject's own*
        // proportions and so is invariant to avatar choice.
        //
        // Sign of Z defaults to +Z (toward camera), matching the typical
        // VTuber gesture of reaching toward the lens. Arms behind the
        // body are usually occluded with low confidence and are dropped
        // earlier by the threshold check.
        //
        // The Y-flip baked onto VRM 0.x roots already handles the
        // "avatar faces camera" orientation, so comparing rest_world
        // directions with camera-space source vectors works uniformly
        // across VRM 0.x and 1.x.
        let dx = source_tip.position[0] - source_base.position[0];
        let dy = source_tip.position[1] - source_base.position[1];
        let observed_2d_sq = dx * dx + dy * dy;
        if observed_2d_sq < 1e-8 {
            continue;
        }
        let observed_2d = observed_2d_sq.sqrt();

        let calibration = state
            .bone_max_2d
            .get(&bone)
            .copied()
            .unwrap_or(0.0)
            .max(observed_2d);
        state.bone_max_2d.insert(bone, calibration);
        let delta_sq = calibration * calibration - observed_2d_sq;
        let z = if delta_sq > 1e-8 {
            delta_sq.sqrt()
        } else {
            0.0
        };
        let source_dir = vec3_normalize(&[dx, dy, z]);

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
        local_transforms[node_idx].rotation =
            quat_slerp_short(&prev_local_rot, &new_local_rot, params.rotation_blend);

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
                            params.rotation_blend,
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

/// Tiny view struct used internally so every bone endpoint (base or tip) goes
/// through the same confidence-gating path regardless of whether it comes
/// from a single joint or a derived midpoint.
struct JointRef {
    position: [f32; 2],
    confidence: f32,
}

type TipRef = JointRef;

fn get_joint_2d(source: &SourceSkeleton, bone: HumanoidBone) -> Option<JointRef> {
    source.joints.get(&bone).map(|j| JointRef {
        position: j.position,
        confidence: j.confidence,
    })
}

fn shoulder_midpoint(source: &SourceSkeleton) -> Option<TipRef> {
    let (Some(l), Some(r)) = (
        source.joints.get(&HumanoidBone::LeftShoulder),
        source.joints.get(&HumanoidBone::RightShoulder),
    ) else {
        return None;
    };
    Some(JointRef {
        position: [
            (l.position[0] + r.position[0]) * 0.5,
            (l.position[1] + r.position[1]) * 0.5,
        ],
        confidence: l.confidence.min(r.confidence),
    })
}

fn midpoint(a: &Vec3, b: &Vec3) -> Vec3 {
    [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ]
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
