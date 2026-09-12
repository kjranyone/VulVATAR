use crate::asset::ColliderShape;
use crate::avatar::AvatarInstance;
use crate::math_utils::{
    closest_point_on_segment, mat4_rotation_to_quat, mat4_translation, quat_conjugate,
    quat_from_vectors, quat_mul, quat_normalize, quat_rotate_vec3, vec3_add, vec3_cross,
    vec3_dot, vec3_length, vec3_length_sq, vec3_scale, vec3_sub, Vec3,
};
use crate::simulation::cloth::ResolvedCollider;
use crate::simulation::SceneGravity;

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
    /// Physically-scaled gravity (default on). Gravity parameters become
    /// fractions of Earth gravity with per-chain floors
    /// (`SpringBoneAsset::gravity_floor`, set for hair categories at
    /// asset build) and stiffness becomes an exponential approach rate —
    /// strands re-hang toward world-down on human-natural timescales
    /// (~0.3–0.5 s) even when the model author left `gravityPower` at 0.
    /// Off reproduces the legacy unitless force mix bit-for-bit
    /// (gravityPower treated as m/s², positional `stiffness*dt` pull) —
    /// the authored-faithful behaviour.
    pub natural_gravity: bool,
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
            natural_gravity: true,
        }
    }
}

/// Natural-gravity model: how the authored stiffness parameter (the
/// 0..1-ish VRC/VRM scale) maps to an exponential approach rate in 1/s.
/// 7.0 puts a typical hair chain (stiffness 0.4) at ~2.8 /s — a settle
/// time constant of ~0.36 s, inside the natural re-hang band. Under the
/// 0.15 g hair floor the equilibrium sag per joint lands around
/// 10–18 mm, so styled strands keep their shape while still hanging
/// toward world-down as the head tilts.
const NATURAL_STIFFNESS_RATE_GAIN: f32 = 7.0;

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
/// Computes the next position from the current and previous positions
/// using velocity (with drag), a positional pull toward the rest target,
/// and a gravity displacement. The pull and gravity magnitudes are
/// precomputed by the caller (they differ between the natural and
/// legacy force models — see the joint loop):
/// - `stiffness_factor` is the fraction of the rest-target offset to
///   close this step (`stiffness * dt` legacy, `1 - exp(-rate*dt)`
///   natural — both unconditionally < 1, so no overshoot),
/// - `gravity_step` is the along-direction displacement this step
///   (`power * dt²` legacy, `frac * g * dt²` natural).
fn verlet_integrate_joint(
    current: &Vec3,
    previous: &Vec3,
    rest_world_target: &Vec3,
    gravity_dir: &Vec3,
    drag: f32,
    stiffness_factor: f32,
    gravity_step: f32,
) -> Vec3 {
    let velocity = vec3_scale(&vec3_sub(current, previous), (1.0 - drag).max(0.0));
    let gravity = vec3_scale(gravity_dir, gravity_step);
    let stiffness_force = vec3_scale(&vec3_sub(rest_world_target, current), stiffness_factor);
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
    // stiffness/drag. In the legacy force model the positional
    // stiffness pull is capped at `1/dt` so a low sway setting cannot
    // push the per-step rest pull past 1.0 and flip the Verlet
    // integration into overshoot oscillation; the natural model's
    // `1 - exp(-rate*dt)` pull is unconditionally < 1 and needs no cap.
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

        // Carried frame for the joint loop. Iteration `j` solves the
        // segment (joints[j-1] → joints[j]): its anchor is the origin of
        // `joints[j-1]` and the solved rotation is written into
        // `joints[j-1]` — the segment's *head*. Under the engine's T·R
        // local compose order (`Transform::to_matrix` puts translation
        // outside the rotation block) a node's rotation aims the segment
        // to its child, not the node's own offset, so writing the tail
        // node instead left the first segment of every chain frozen at
        // rest and displaced each swing one segment down.
        //
        // For `j == 1` the head is the chain root, whose pose global is
        // fresh this frame (tracking). For `j >= 2` the head was solved
        // earlier in this loop, so we carry its solved origin
        // (`positions[j-1]`, == `next` of the previous iteration) and its
        // post-solve basis (parent basis ∘ written rotation). The pose
        // globals are only recomputed after the whole solver, so reading
        // them here would anchor bent chains to rest-derived transforms
        // and make the solved chain diverge from the rendered one.
        let head_idx = joints[0].0 as usize;
        let mut anchor = mat4_translation(&avatar.pose.global_transforms[head_idx]);
        let mut head_parent_basis = match avatar.asset.skeleton.nodes[head_idx].parent {
            Some(p) => {
                mat4_rotation_to_quat(&avatar.pose.global_transforms[p.0 as usize])
            }
            None => [0.0, 0.0, 0.0, 1.0],
        };

        for j in 1..joints.len() {
            let stiffness_param = spring_asset
                .joint_stiffness
                .get(j)
                .copied()
                .unwrap_or(chain_stiffness)
                * sway_inv;
            let drag = (spring_asset
                .joint_drag
                .get(j)
                .copied()
                .unwrap_or(chain_drag)
                * sway_inv)
                .clamp(0.0, 1.0);
            let gravity_param = spring_asset
                .joint_gravity_power
                .get(j)
                .copied()
                .unwrap_or(chain_gravity_power);

            // Force model. Natural gravity (default): gravity parameters
            // are fractions of Earth gravity — authored power plus the
            // user offset, floored per chain by `gravity_floor` — and
            // stiffness is an exponential approach rate. Strands re-hang
            // toward world-down on human-natural timescales even when
            // the model author left `gravityPower` at 0. Legacy (toggle
            // off) reproduces the original unitless mix bit-for-bit:
            // gravityPower treated as m/s², positional `stiffness * dt`
            // pull capped by `max_stiffness`.
            let (stiffness_factor, gravity_step) = if tuning.natural_gravity {
                let rate = stiffness_param * NATURAL_STIFFNESS_RATE_GAIN;
                let factor = -((-(rate * dt)).exp_m1());
                let frac = (gravity_param + tuning.gravity_offset)
                    .clamp(0.0, 1.5)
                    .max(spring_asset.gravity_floor);
                (
                    factor,
                    frac * SceneGravity::EARTH_G * gravity_scale * dt2,
                )
            } else {
                let stiffness = stiffness_param.min(max_stiffness);
                let power = (gravity_param + tuning.gravity_offset).max(0.0) * gravity_scale;
                (stiffness * dt, power * dt2)
            };
            let node_idx = joints[j].0 as usize;
            let head_idx = joints[j - 1].0 as usize;

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

            // Anchor from the carried frame (see the comment above the
            // loop): the head joint's solved origin, not its rest-derived
            // pose global.
            let parent_world_pos = anchor;
            let rest_local_translation =
                avatar.asset.skeleton.nodes[node_idx].rest_local.translation;
            let bone_length = vec3_length(&rest_local_translation).max(0.001);

            // Segment rest direction in the head node's local frame
            // (head rest rotation applied to the bone axis), and the
            // stiffness pull target following the solved bend of the
            // parent segment.
            let head_rest_rot = avatar.asset.skeleton.nodes[head_idx].rest_local.rotation;
            let rest_dir_local = quat_rotate_vec3(&head_rest_rot, &rest_local_translation);
            let rest_world_dir = quat_rotate_vec3(&head_parent_basis, &rest_dir_local);
            let rest_world_target = vec3_add(&parent_world_pos, &rest_world_dir);

            // 1. Verlet integration
            let mut next = verlet_integrate_joint(
                &current,
                &previous,
                &rest_world_target,
                &gravity_dir,
                drag,
                stiffness_factor,
                gravity_step,
            );

            // 2. Enforce bone length
            next = enforce_bone_length(&next, &parent_world_pos, bone_length);

            // 3. Collider resolution. Snapshot the pre-collision position
            // so the total projection can be subtracted from the implicit
            // velocity in step 4.
            let pre_collision = next;
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

            // 4. Update per-joint state. Verlet stores velocity
            // implicitly as (current - previous); a collider projection
            // that moves only `current` converts this step's penetration
            // depth into an outward kick next step — the hair bouncing
            // off a shoulder instead of draping on it. Shifting
            // `previous` by the same correction makes the contact
            // inelastic: the chain keeps its tangential slide, loses the
            // normal restitution.
            let correction = vec3_sub(&next, &pre_collision);
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
                *prev = vec3_add(&current, &correction);
            }

            // 5. Rotation writeback into the segment's HEAD node
            // (joints[j-1]): its rotation must aim the segment at the
            // solved tail. `rest_dir_local` is the segment's rest
            // direction in the head's local frame, so the delta rotation
            // maps rest → solved in the head's parent frame, composed
            // before the head's own rest rotation. Also stored in the
            // chain state so frames that skip the solver (zero
            // substeps) can re-apply it over the per-frame rest reset in
            // `build_base_pose`.
            let solved_dir_world = vec3_sub(&next, &parent_world_pos);
            let rest_dir_len = vec3_length(&rest_dir_local);
            let solved_dir_len = vec3_length(&solved_dir_world);
            if rest_dir_len > 1e-8 && solved_dir_len > 1e-8 {
                let solved_dir_local = quat_rotate_vec3(
                    &quat_conjugate(&head_parent_basis),
                    &solved_dir_world,
                );
                let rot = quat_from_to(
                    &vec3_scale(&rest_dir_local, 1.0 / rest_dir_len),
                    &vec3_scale(&solved_dir_local, 1.0 / vec3_length(&solved_dir_local)),
                );
                let written = quat_normalize(&quat_mul(&rot, &head_rest_rot));
                avatar.pose.local_transforms[head_idx].rotation = written;
                if let Some(slot) = avatar.secondary_motion.spring_states[chain_idx]
                    .solved_rotations
                    .get_mut(j - 1)
                {
                    *slot = written;
                }
                // Carry the head's post-solve basis (parent basis ∘
                // written rotation) and the solved tail as the next
                // segment's anchor.
                head_parent_basis = quat_mul(&head_parent_basis, &written);
            }
            anchor = next;
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
