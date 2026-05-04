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

use crate::asset::{
    identity_matrix, AvatarAsset, HumanoidBone, Mat4, Quat, SkeletonAsset, Transform,
};
use crate::math_utils::{
    mat4_mul, mat4_translation, quat_conjugate, quat_from_vectors, quat_mul, quat_normalize,
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
        Some(pi) if pi < rest_globals.len() => mat3_to_quat(&rest_globals[pi]),
        _ => [0.0, 0.0, 0.0, 1.0],
    };
    let parent_rot_inv = quat_conjugate(&parent_rot);
    let q_pre = quat_mul(&quat_mul(&parent_rot_inv, &q_world), &parent_rot);

    locals[upper_idx].rotation = quat_mul(&q_pre, &locals[upper_idx].rotation);
}

/// Extract the rotation component from a column-major affine 4×4 as
/// a unit quaternion. Normalises each basis column to strip uniform
/// scale; non-uniform scale will produce a slightly off-axis result
/// but VRM rigs don't use non-uniform scale on humanoid bones.
fn mat3_to_quat(m: &Mat4) -> Quat {
    let col0 = vec3_normalize(&[m[0][0], m[0][1], m[0][2]]);
    let col1 = vec3_normalize(&[m[1][0], m[1][1], m[1][2]]);
    let col2 = vec3_normalize(&[m[2][0], m[2][1], m[2][2]]);
    let m00 = col0[0];
    let m10 = col0[1];
    let m20 = col0[2];
    let m01 = col1[0];
    let m11 = col1[1];
    let m21 = col1[2];
    let m02 = col2[0];
    let m12 = col2[1];
    let m22 = col2[2];

    let trace = m00 + m11 + m22;
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        let inv = 1.0 / s;
        quat_normalize(&[
            (m21 - m12) * inv,
            (m02 - m20) * inv,
            (m10 - m01) * inv,
            0.25 * s,
        ])
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        let inv = 1.0 / s;
        quat_normalize(&[
            0.25 * s,
            (m01 + m10) * inv,
            (m02 + m20) * inv,
            (m21 - m12) * inv,
        ])
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        let inv = 1.0 / s;
        quat_normalize(&[
            (m01 + m10) * inv,
            0.25 * s,
            (m12 + m21) * inv,
            (m02 - m20) * inv,
        ])
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        let inv = 1.0 / s;
        quat_normalize(&[
            (m02 + m20) * inv,
            (m12 + m21) * inv,
            0.25 * s,
            (m10 - m01) * inv,
        ])
    }
}

/// Hierarchy traversal that produces global transforms from per-node
/// locals. Stack-based so it doesn't blow the call stack on deep
/// rigs. Identical algorithm to
/// `AvatarInstance::compute_global_pose`; duplicated here so the
/// snapshot path doesn't need a `&mut AvatarInstance`.
fn compute_globals(skeleton: &SkeletonAsset, locals: &[Transform]) -> Vec<Mat4> {
    let n = skeleton.nodes.len();
    let mut globals = vec![identity_matrix(); n];
    let mut stack: Vec<(usize, Option<usize>)> = Vec::new();
    for root in skeleton.root_nodes.iter().rev() {
        stack.push((root.0 as usize, None));
    }
    while let Some((idx, parent_idx)) = stack.pop() {
        let local_mat = locals[idx].to_matrix();
        let global_mat = match parent_idx {
            Some(pi) => mat4_mul(&globals[pi], &local_mat),
            None => local_mat,
        };
        globals[idx] = global_mat;
        for child in skeleton.nodes[idx].children.iter().rev() {
            stack.push((child.0 as usize, Some(idx)));
        }
    }
    globals
}

/// `skinning[i] = globals[i] * inverse_bind[i]`. Identical to
/// `AvatarInstance::build_skinning_matrices`; same duplication
/// rationale (see [`compute_globals`]).
fn build_skinning_from_globals(skeleton: &SkeletonAsset, globals: &[Mat4]) -> Vec<Mat4> {
    let n = skeleton.nodes.len();
    let mut out = vec![identity_matrix(); n];
    for i in 0..n {
        let ibm = if i < skeleton.inverse_bind_matrices.len() {
            &skeleton.inverse_bind_matrices[i]
        } else {
            continue;
        };
        if i < globals.len() {
            out[i] = mat4_mul(&globals[i], ibm);
        }
    }
    out
}
