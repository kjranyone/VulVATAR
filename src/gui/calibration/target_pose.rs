//! Build the avatar's skinning matrices for the calibration target
//! pose (T-pose / hands-at-sides), used by the modal's "this is what
//! you should look like" reference snapshot. The output is fed to a
//! one-shot offscreen render via the existing thumbnail render path.
//!
//! The rest pose of a VRM is *defined* to be T-pose (arms horizontal,
//! palms forward, feet shoulder-width), so [`CalibrationMode::FullBody`]
//! just snapshots the rest pose — no bone manipulation required.
//! [`CalibrationMode::UpperBody`] aims each `UpperArm` bone at a
//! target *world*-space direction (slightly outward, mostly down) so
//! the arms hang at the avatar's sides.
//!
//! We compute the rotation in world space via shortest-arc to the
//! target direction, then conjugate by the parent's world rotation
//! (`Q_local = R_parent⁻¹ * Q_world * R_parent`) to recover a
//! parent-local pre-multiplier. This is rig-agnostic: it works
//! regardless of how the upper arm's parent (shoulder/chest) is
//! oriented in the rest pose, and regardless of which axis the bone
//! itself uses as its "length" axis. Earlier versions applied a
//! fixed-angle rotation around a chosen local axis (Z, then X, then
//! Y) and each one came out wrong on at least some rigs because the
//! parent shoulder isn't axis-aligned to world; aim-at-target
//! sidesteps that entirely.

use crate::asset::{identity_matrix, AvatarAsset, HumanoidBone, Mat4, SkeletonAsset, Transform};
use crate::avatar::pose as pose_helpers;
use crate::math_utils::{
    mat4_rotation_to_quat, mat4_translation, quat_conjugate, quat_from_vectors, quat_mul,
    vec3_length_sq, vec3_normalize, vec3_sub, Vec3,
};
use crate::tracking::CalibrationMode;

/// Bend the arm this many degrees outward from straight-down (world
/// −Y) in the target pose. 10° matches the natural at-ease standing
/// posture: arms hang almost vertical with elbows just slightly clear
/// of the hips. 0° (pure vertical) reads as military/stiff and tends
/// to clip the wrists into the thighs on most VRM proportions.
const UPPER_BODY_ARM_OUTWARD_DEGREES: f32 = 10.0;

/// Build the avatar's skinning matrices for the calibration mode's
/// target pose. Pure function over the asset (no mutation, no shared
/// state) so it's safe to call from the snapshot-request path
/// without affecting the live render's avatar pose.
pub fn target_pose_skinning_matrices(
    asset: &AvatarAsset,
    mode: CalibrationMode,
) -> Vec<Mat4> {
    let skeleton = &asset.skeleton;

    let mut locals: Vec<Transform> = skeleton
        .nodes
        .iter()
        .map(|n| n.rest_local.clone())
        .collect();

    if let CalibrationMode::UpperBody = mode {
        if let Some(humanoid) = &asset.humanoid {
            let rest_globals = compute_globals(skeleton, &locals);
            for (upper, lower) in [
                (HumanoidBone::LeftUpperArm, HumanoidBone::LeftLowerArm),
                (HumanoidBone::RightUpperArm, HumanoidBone::RightLowerArm),
            ] {
                let upper_idx = match humanoid.bone_map.get(&upper) {
                    Some(n) => n.0 as usize,
                    None => continue,
                };
                let lower_idx = match humanoid.bone_map.get(&lower) {
                    Some(n) => n.0 as usize,
                    None => continue,
                };
                apply_arm_down_rotation(skeleton, &mut locals, &rest_globals, upper_idx, lower_idx);
            }
        }
    }

    let globals = compute_globals(skeleton, &locals);
    build_skinning_from_globals(skeleton, &globals)
}

/// Aim the arm at a target *world* direction (slightly-outward,
/// mostly-down) using shortest-arc rotation, then convert to a
/// parent-local pre-multiplier via the standard conjugation
/// `Q_local = R_parent⁻¹ * Q_world * R_parent`.
///
/// We don't apply a fixed-angle Z rotation because not every rig has
/// the upper arm pointing exactly horizontal in the rest pose — some
/// rest at a slight downward angle, in which case rotating by a fixed
/// 80° would leave the arm tipped *past* vertical (and toward the
/// opposite side, producing the "arms folded at waist" symptom).
/// Aiming-at-target is invariant to where the arm starts.
///
/// Sign of the outward bend (which side the elbow clears the body
/// on) is auto-detected from the rest-pose arm's world X — the rig
/// itself tells us which arm is which, so we don't need to hard-code
/// "left arm goes this way, right arm goes the other way".
fn apply_arm_down_rotation(
    skeleton: &SkeletonAsset,
    locals: &mut [Transform],
    rest_globals: &[Mat4],
    upper_idx: usize,
    lower_idx: usize,
) {
    if upper_idx >= locals.len()
        || upper_idx >= rest_globals.len()
        || lower_idx >= rest_globals.len()
    {
        return;
    }
    let upper_pos = mat4_translation(&rest_globals[upper_idx]);
    let lower_pos = mat4_translation(&rest_globals[lower_idx]);
    let arm_world = vec3_sub(&lower_pos, &upper_pos);
    if vec3_length_sq(&arm_world) < 1e-8 {
        return;
    }
    let arm_dir = vec3_normalize(&arm_world);

    let outward_sign = if arm_dir[0] >= 0.0 { 1.0 } else { -1.0 };
    let bend = UPPER_BODY_ARM_OUTWARD_DEGREES.to_radians();
    let target_dir: Vec3 = [outward_sign * bend.sin(), -bend.cos(), 0.0];
    let q_world = quat_from_vectors(&arm_dir, &target_dir);

    let parent_idx = skeleton.nodes[upper_idx].parent.map(|p| p.0 as usize);
    let parent_rot = match parent_idx {
        Some(pi) if pi < rest_globals.len() => mat4_rotation_to_quat(&rest_globals[pi]),
        _ => [0.0, 0.0, 0.0, 1.0],
    };
    let parent_rot_inv = quat_conjugate(&parent_rot);
    let q_pre = quat_mul(&quat_mul(&parent_rot_inv, &q_world), &parent_rot);

    locals[upper_idx].rotation = quat_mul(&q_pre, &locals[upper_idx].rotation);
}

/// Hierarchy traversal that produces global transforms from per-node
/// locals. Owns its output `Vec<Mat4>` so the snapshot path doesn't
/// need a `&mut AvatarInstance` — wraps the in-place pure helper in
/// [`crate::avatar::pose::compute_global_transforms`].
fn compute_globals(skeleton: &SkeletonAsset, locals: &[Transform]) -> Vec<Mat4> {
    let mut globals = vec![identity_matrix(); skeleton.nodes.len()];
    pose_helpers::compute_global_transforms(skeleton, locals, &mut globals);
    globals
}

/// Owning wrapper around [`crate::avatar::pose::build_skinning_matrices`]
/// — same rationale as [`compute_globals`].
fn build_skinning_from_globals(skeleton: &SkeletonAsset, globals: &[Mat4]) -> Vec<Mat4> {
    let mut out = vec![identity_matrix(); skeleton.nodes.len()];
    pose_helpers::build_skinning_matrices(skeleton, globals, &mut out);
    out
}
