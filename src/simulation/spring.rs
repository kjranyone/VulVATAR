use crate::asset::ColliderShape;
use crate::avatar::AvatarInstance;
use crate::math_utils::{
    closest_point_on_segment, mat4_translation, quat_from_vectors, quat_mul, quat_normalize,
    quat_rotate_vec3, vec3_add, vec3_cross, vec3_dot, vec3_length, vec3_length_sq, vec3_scale,
    vec3_sub, Vec3,
};
use crate::simulation::cloth::ResolvedCollider;

/// User-facing spring-bone tuning, layered on top of the VRM asset's
/// authored per-chain/per-joint values at simulation time (the asset is
/// never mutated). Edited in the Rendering inspector, persisted with the
/// project.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpringTuning {
    /// How freely the chains swing. `1.0` = as authored. Applied as an
    /// *inverse* scale on both stiffness (rest-pose pull) and drag
    /// (velocity damping): raising sway softens the pull and lets
    /// oscillation persist; lowering it stiffens and deadens the chain
    /// until, near the bottom of the range, hair is effectively rigid.
    /// One knob moves both because they push the same percept in
    /// opposite directions — splitting them exposes tuning traps
    /// (zero stiffness + low drag never settles) for no expressive gain.
    pub sway_scale: f32,
    /// Additive adjustment to every joint's authored `gravityPower`,
    /// clamped so the effective power never goes negative. Additive —
    /// not a multiplier — because VRM 1.0's *default* gravityPower is
    /// 0.0: many models ship with no authored gravity at all, and a
    /// multiplier would be a dead knob on exactly the models whose hair
    /// most needs the droop.
    pub gravity_offset: f32,
}

impl SpringTuning {
    pub const SWAY_RANGE: std::ops::RangeInclusive<f32> = 0.05..=2.0;
    pub const GRAVITY_RANGE: std::ops::RangeInclusive<f32> = -1.0..=1.0;
}

impl Default for SpringTuning {
    fn default() -> Self {
        Self {
            sway_scale: 1.0,
            gravity_offset: 0.0,
        }
    }
}

/// Verlet integration-based spring bone solver.
///
/// For each spring chain defined in the avatar asset, this solver:
/// 1. Iterates joints tail-to-tip
/// 2. Computes velocity via Verlet (current - previous) with drag
/// 3. Applies stiffness pull toward rest pose, plus gravity
/// 4. Normalizes the result to preserve bone length
/// 5. Resolves sphere/capsule collider penetrations
/// 6. Writes solved rotations back into the avatar's local transforms
///
/// Verlet position update for a single joint.
///
/// Computes the next position from the current and previous positions using
/// velocity (with drag), stiffness pull toward the rest pose, and gravity.
#[allow(clippy::too_many_arguments)]
fn verlet_integrate_joint(
    current: &Vec3,
    previous: &Vec3,
    rest_world_target: &Vec3,
    gravity_dir: &Vec3,
    drag: f32,
    stiffness: f32,
    gravity_power: f32,
    dt: f32,
    dt2: f32,
) -> Vec3 {
    let velocity = vec3_scale(&vec3_sub(current, previous), (1.0 - drag).max(0.0));
    let gravity = vec3_scale(gravity_dir, gravity_power * dt2);
    let stiffness_force = vec3_scale(&vec3_sub(rest_world_target, current), stiffness * dt);
    vec3_add(
        &vec3_add(&vec3_add(current, &velocity), &stiffness_force),
        &gravity,
    )
}

/// Normalize the joint position to maintain the original bone length from
/// the parent.
fn enforce_bone_length(next: &Vec3, parent_world_pos: &Vec3, bone_length: f32) -> Vec3 {
    let dir = vec3_sub(next, parent_world_pos);
    let dir_len = vec3_length(&dir);
    if dir_len > 1e-8 {
        let normalized = vec3_scale(&dir, bone_length / dir_len);
        vec3_add(parent_world_pos, &normalized)
    } else {
        *next
    }
}

/// Resolve sphere collision for a joint, pushing the position out of the
/// sphere and re-normalizing to bone length.
fn resolve_sphere_collision(
    next: &Vec3,
    collider_center: &Vec3,
    collider_radius: f32,
    bone_radius: f32,
    parent_world_pos: &Vec3,
    bone_length: f32,
) -> Vec3 {
    let total_radius = collider_radius + bone_radius;
    let diff = vec3_sub(next, collider_center);
    let dist = vec3_length(&diff);
    if dist < total_radius {
        let normal = if dist > 1e-8 {
            vec3_scale(&diff, 1.0 / dist)
        } else {
            [0.0, 0.0, 1.0]
        };
        let pushed = vec3_add(collider_center, &vec3_scale(&normal, total_radius));
        let constrained = enforce_bone_length(&pushed, parent_world_pos, bone_length);
        let diff_c = vec3_sub(&constrained, collider_center);
        let dist_c = vec3_length(&diff_c);
        if dist_c < total_radius {
            let normal_c = if dist_c > 1e-8 {
                vec3_scale(&diff_c, 1.0 / dist_c)
            } else {
                normal
            };
            vec3_add(collider_center, &vec3_scale(&normal_c, total_radius))
        } else {
            constrained
        }
    } else {
        *next
    }
}

/// Resolve capsule collision for a joint.  The capsule is defined by a
/// center, axis direction, half-height, and radius.
#[allow(clippy::too_many_arguments)]
fn resolve_capsule_collision(
    next: &Vec3,
    collider_center: &Vec3,
    collider_axis: &Vec3,
    half_height: f32,
    collider_radius: f32,
    bone_radius: f32,
    parent_world_pos: &Vec3,
    bone_length: f32,
) -> Vec3 {
    let seg_a = vec3_add(collider_center, &vec3_scale(collider_axis, -half_height));
    let seg_b = vec3_add(collider_center, &vec3_scale(collider_axis, half_height));
    let closest = closest_point_on_segment(&seg_a, &seg_b, next);
    let total_radius = collider_radius + bone_radius;
    let diff = vec3_sub(next, &closest);
    let dist = vec3_length(&diff);
    if dist < total_radius {
        let normal = if dist > 1e-8 {
            vec3_scale(&diff, 1.0 / dist)
        } else {
            [0.0, 0.0, 1.0]
        };
        let pushed = vec3_add(&closest, &vec3_scale(&normal, total_radius));
        let constrained = enforce_bone_length(&pushed, parent_world_pos, bone_length);
        let closest_c = closest_point_on_segment(&seg_a, &seg_b, &constrained);
        let diff_c = vec3_sub(&constrained, &closest_c);
        let dist_c = vec3_length(&diff_c);
        if dist_c < total_radius {
            let normal_c = if dist_c > 1e-8 {
                vec3_scale(&diff_c, 1.0 / dist_c)
            } else {
                normal
            };
            vec3_add(&closest_c, &vec3_scale(&normal_c, total_radius))
        } else {
            constrained
        }
    } else {
        *next
    }
}

/// Compute the local rotation that maps the rest-pose direction to the
/// solved world direction and write it back into the avatar's local
/// transforms.  Returns `true` if the rotation was successfully written.
fn compute_rotation_writeback(
    next: &Vec3,
    parent_world_pos: &Vec3,
    rest_local_translation: &Vec3,
    parent_global_transform: &[[f32; 4]; 4],
    rest_rotation: &[f32; 4],
    local_rotation_out: &mut [f32; 4],
) -> bool {
    let solved_dir_world = vec3_sub(next, parent_world_pos);
    let rest_dir_len = vec3_length(rest_local_translation);
    if rest_dir_len < 1e-8 {
        return false;
    }
    let rest_dir_norm = vec3_scale(rest_local_translation, 1.0 / rest_dir_len);

    let solved_dir_local =
        mat4_inverse_transform_direction(parent_global_transform, &solved_dir_world);
    let solved_dir_local_len = vec3_length(&solved_dir_local);
    if solved_dir_local_len < 1e-8 {
        return false;
    }
    let solved_dir_local_norm = vec3_scale(&solved_dir_local, 1.0 / solved_dir_local_len);

    let rot = quat_from_to(&rest_dir_norm, &solved_dir_local_norm);
    let final_rot = quat_mul(&rot, rest_rotation);
    *local_rotation_out = quat_normalize(&final_rot);
    true
}

pub fn step_spring_bones(
    dt: f32,
    avatar: &mut AvatarInstance,
    world_colliders: &[ResolvedCollider],
    tuning: &SpringTuning,
    // Scene gravity, resolved into this avatar's local frame: `gravity_dir`
    // is a unit down vector that *reorients* each chain's authored
    // `gravity_dir` (via a delta rotation from default-down, so the global
    // direction control turns all chains uniformly while preserving their
    // relative authored offsets), and `gravity_scale` is the scene
    // strength multiplier folded into every joint's power. Default gravity
    // (down, strength 1) reproduces the authored behaviour for an upright
    // avatar — including chains whose authored direction is not straight
    // down.
    gravity_dir: [f32; 3],
    gravity_scale: f32,
) {
    let chain_count = avatar.secondary_motion.spring_states.len();
    if chain_count == 0 || dt <= 0.0 {
        return;
    }

    let dt2 = dt * dt;
    // User tuning (see [`SpringTuning`]): sway inversely scales
    // stiffness/drag. Stiffness is additionally capped at `1/dt` so a
    // low sway setting cannot push the per-step rest pull past 1.0 and
    // flip the Verlet integration into overshoot oscillation.
    let sway_inv = 1.0 / tuning.sway_scale.clamp(0.05, 2.0);
    let max_stiffness = 1.0 / dt;

    // Scene gravity direction layers *on top of* each chain's authored
    // `gravity_dir` rather than replacing it: this is the rotation that
    // carries the default down axis onto the scene's (avatar-local) down,
    // and it is applied to every authored direction below. When the scene
    // gravity is default-down for an upright avatar `gravity_dir` is
    // `[0,-1,0]` and this delta is identity, so authored per-chain
    // directions (e.g. side-swept hair) are reproduced exactly; tilting
    // the global gravity reorients all chains uniformly while preserving
    // their relative authored offsets.
    let gravity_delta = quat_from_vectors(&[0.0, -1.0, 0.0], &gravity_dir);

    for chain_idx in 0..chain_count {
        if chain_idx >= avatar.asset.spring_bones.len() {
            continue;
        }
        let spring_asset = &avatar.asset.spring_bones[chain_idx];
        let chain_stiffness = spring_asset.stiffness;
        let chain_drag = spring_asset.drag_force;
        // Authored per-chain direction, reoriented by the scene-gravity
        // delta above (identity when the global gravity is default-down).
        let gravity_dir = quat_rotate_vec3(&gravity_delta, &spring_asset.gravity_dir);
        let chain_gravity_power = spring_asset.gravity_power;
        let bone_radius = spring_asset.radius;
        let colliders: Vec<&crate::asset::ColliderAsset> = spring_asset
            .collider_refs
            .iter()
            .filter_map(|r| avatar.asset.colliders.iter().find(|c| c.id == r.id))
            .collect();

        let joints = avatar.secondary_motion.spring_states[chain_idx]
            .joints
            .clone();

        if joints.len() < 2 {
            continue;
        }

        for j in 1..joints.len() {
            let stiffness = (spring_asset
                .joint_stiffness
                .get(j)
                .copied()
                .unwrap_or(chain_stiffness)
                * sway_inv)
                .min(max_stiffness);
            let drag = (spring_asset
                .joint_drag
                .get(j)
                .copied()
                .unwrap_or(chain_drag)
                * sway_inv)
                .clamp(0.0, 1.0);
            // authored power + user offset, then scaled by the scene
            // gravity strength; clamped non-negative.
            let gravity_power = ((spring_asset
                .joint_gravity_power
                .get(j)
                .copied()
                .unwrap_or(chain_gravity_power)
                + tuning.gravity_offset)
                * gravity_scale)
                .max(0.0);
            let node_idx = joints[j].0 as usize;
            let parent_idx = joints[j - 1].0 as usize;

            // Retrieve current / previous positions from per-joint state.
            let current = avatar.secondary_motion.spring_states[chain_idx]
                .positions
                .get(j)
                .copied()
                .unwrap_or([0.0, 0.0, 0.0]);
            let previous = avatar.secondary_motion.spring_states[chain_idx]
                .previous_positions
                .get(j)
                .copied()
                .unwrap_or(current);

            // Handle uninitialized positions.
            let current = if vec3_length_sq(&current) < 1e-12 {
                mat4_translation(&avatar.pose.global_transforms[node_idx])
            } else {
                current
            };
            let previous = if vec3_length_sq(&previous) < 1e-12 {
                current
            } else {
                previous
            };

            let parent_world_pos = mat4_translation(&avatar.pose.global_transforms[parent_idx]);
            let rest_local_translation =
                avatar.asset.skeleton.nodes[node_idx].rest_local.translation;
            let bone_length = vec3_length(&rest_local_translation).max(0.001);

            let rest_world_dir = mat4_transform_direction(
                &avatar.pose.global_transforms[parent_idx],
                &rest_local_translation,
            );
            let rest_world_target = vec3_add(&parent_world_pos, &rest_world_dir);

            // 1. Verlet integration
            let mut next = verlet_integrate_joint(
                &current,
                &previous,
                &rest_world_target,
                &gravity_dir,
                drag,
                stiffness,
                gravity_power,
                dt,
                dt2,
            );

            // 2. Enforce bone length
            next = enforce_bone_length(&next, &parent_world_pos, bone_length);

            // 3. Collider resolution
            for collider in &colliders {
                let collider_node = collider.node.0 as usize;
                if collider_node >= avatar.pose.global_transforms.len() {
                    continue;
                }
                let collider_world_pos =
                    mat4_translation(&avatar.pose.global_transforms[collider_node]);
                let rotated_offset = mat4_transform_direction(
                    &avatar.pose.global_transforms[collider_node],
                    &collider.offset,
                );
                let collider_center = vec3_add(&collider_world_pos, &rotated_offset);

                match collider.shape {
                    ColliderShape::Sphere { radius } => {
                        next = resolve_sphere_collision(
                            &next,
                            &collider_center,
                            radius,
                            bone_radius,
                            &parent_world_pos,
                            bone_length,
                        );
                    }
                    ColliderShape::Capsule { radius, height } => {
                        let up = mat4_transform_direction(
                            &avatar.pose.global_transforms[collider_node],
                            &[0.0, 1.0, 0.0],
                        );
                        let up_len = vec3_length(&up);
                        let up_norm = if up_len > 1e-8 {
                            vec3_scale(&up, 1.0 / up_len)
                        } else {
                            [0.0, 1.0, 0.0]
                        };
                        next = resolve_capsule_collision(
                            &next,
                            &collider_center,
                            &up_norm,
                            height * 0.5,
                            radius,
                            bone_radius,
                            &parent_world_pos,
                            bone_length,
                        );
                    }
                }
            }

            // 3b. World collider resolution (scene-level colliders from Rapier/PhysicsWorld)
            for wc in world_colliders {
                match wc {
                    ResolvedCollider::Sphere { center, radius } => {
                        next = resolve_sphere_collision(
                            &next,
                            center,
                            *radius,
                            bone_radius,
                            &parent_world_pos,
                            bone_length,
                        );
                    }
                    ResolvedCollider::Capsule {
                        center,
                        radius,
                        half_height,
                        axis,
                    } => {
                        next = resolve_capsule_collision(
                            &next,
                            center,
                            axis,
                            *half_height,
                            *radius,
                            bone_radius,
                            &parent_world_pos,
                            bone_length,
                        );
                    }
                }
            }

            // 4. Update per-joint state
            if let Some(pos) = avatar.secondary_motion.spring_states[chain_idx]
                .positions
                .get_mut(j)
            {
                *pos = next;
            }
            if let Some(prev) = avatar.secondary_motion.spring_states[chain_idx]
                .previous_positions
                .get_mut(j)
            {
                *prev = current;
            }

            // 5. Rotation writeback
            let rest_rot = avatar.asset.skeleton.nodes[node_idx].rest_local.rotation;
            compute_rotation_writeback(
                &next,
                &parent_world_pos,
                &rest_local_translation,
                &avatar.pose.global_transforms[parent_idx],
                &rest_rot,
                &mut avatar.pose.local_transforms[node_idx].rotation,
            );
        }
    }

    // Update collision cache contact count (informational).
    avatar.secondary_motion.collision_cache.contact_count = 0;
}

// ---------------------------------------------------------------------------
// Mat4 helpers (column-major) -- specific to spring bone solver
// ---------------------------------------------------------------------------

/// Transform a direction vector (no translation) by the upper-left 3x3 of a
/// column-major 4x4 matrix.
#[inline]
fn mat4_transform_direction(m: &[[f32; 4]; 4], d: &Vec3) -> Vec3 {
    [
        m[0][0] * d[0] + m[1][0] * d[1] + m[2][0] * d[2],
        m[0][1] * d[0] + m[1][1] * d[1] + m[2][1] * d[2],
        m[0][2] * d[0] + m[1][2] * d[1] + m[2][2] * d[2],
    ]
}

/// Inverse-transform a direction by the upper-left 3x3 of a column-major 4x4
/// matrix. Assumes the 3x3 is orthonormal (rotation only or uniform scale),
/// so the inverse is the transpose.
#[inline]
fn mat4_inverse_transform_direction(m: &[[f32; 4]; 4], d: &Vec3) -> Vec3 {
    // Transpose of the 3x3 block: rows become columns.
    [
        m[0][0] * d[0] + m[0][1] * d[1] + m[0][2] * d[2],
        m[1][0] * d[0] + m[1][1] * d[1] + m[1][2] * d[2],
        m[2][0] * d[0] + m[2][1] * d[1] + m[2][2] * d[2],
    ]
}

/// Build a quaternion that rotates unit vector `from` to unit vector `to`.
fn quat_from_to(from: &Vec3, to: &Vec3) -> [f32; 4] {
    let d = vec3_dot(from, to);
    if d > 0.9999 {
        // Nearly identical directions: identity quaternion.
        return [0.0, 0.0, 0.0, 1.0];
    }
    if d < -0.9999 {
        // Nearly opposite: rotate 180 degrees around an arbitrary perpendicular axis.
        let mut perp = vec3_cross(from, &[1.0, 0.0, 0.0]);
        if vec3_length_sq(&perp) < 1e-6 {
            perp = vec3_cross(from, &[0.0, 1.0, 0.0]);
        }
        let len = vec3_length(&perp);
        let perp = vec3_scale(&perp, 1.0 / len);
        // 180-degree rotation: w=0, axis=perp
        return [perp[0], perp[1], perp[2], 0.0];
    }
    let axis = vec3_cross(from, to);
    let w = 1.0 + d;
    quat_normalize(&[axis[0], axis[1], axis[2], w])
}
