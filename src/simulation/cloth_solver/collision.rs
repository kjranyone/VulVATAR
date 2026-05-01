//! External collision, self-collision, and pin enforcement for the cloth solver.

use crate::math_utils::{
    closest_point_on_segment, vec3_add, vec3_dot, vec3_length, vec3_scale, vec3_sub, Vec3,
};
use crate::simulation::cloth::{ClothSimState, ResolvedCollider};

// =========================================================================
// Collision (M3: sphere + capsule, E6: collision_margin)
// =========================================================================

pub(super) fn collide(sim: &mut ClothSimState, colliders: &[ResolvedCollider]) {
    let collision_margin = sim.collision_margin;

    for p in sim.particles.iter_mut() {
        if p.pinned {
            continue;
        }
        for collider in colliders {
            match collider {
                ResolvedCollider::Sphere { center, radius } => {
                    let effective_radius = radius + collision_margin;
                    let diff = vec3_sub(&p.position, center);
                    let dist = vec3_length(&diff);
                    if dist < effective_radius && dist > 1e-12 {
                        let normal = vec3_scale(&diff, 1.0 / dist);
                        p.position = vec3_add(center, &vec3_scale(&normal, effective_radius));
                    }
                }
                ResolvedCollider::Capsule {
                    center,
                    radius,
                    half_height,
                    axis,
                } => {
                    let effective_radius = radius + collision_margin;
                    // Compute segment endpoints from center, axis, and half_height
                    let seg_a = vec3_sub(center, &vec3_scale(axis, *half_height));
                    let seg_b = vec3_add(center, &vec3_scale(axis, *half_height));
                    // Closest point on segment to particle
                    let closest = closest_point_on_segment(&seg_a, &seg_b, &p.position);
                    let diff = vec3_sub(&p.position, &closest);
                    let dist = vec3_length(&diff);
                    if dist < effective_radius && dist > 1e-12 {
                        let normal = vec3_scale(&diff, 1.0 / dist);
                        p.position = vec3_add(&closest, &vec3_scale(&normal, effective_radius));
                    }
                }
            }
        }
    }
}

// =========================================================================
// Self-collision via spatial hashing
// =========================================================================

/// Detect and resolve particle-particle self-collisions using a spatial hash
/// grid.  Two particles collide when their distance is less than
/// `2 * self_collision_radius`.  Pairs connected by a distance constraint are
/// skipped since they are expected to be close.
pub(super) fn resolve_self_collisions(sim: &mut ClothSimState) {
    if !sim.self_collision {
        return;
    }

    let radius = sim.self_collision_radius;
    let min_dist = 2.0 * radius;
    let min_dist_sq = min_dist * min_dist;

    // Rebuild the spatial hash grid with current positions.
    sim.spatial_hash.clear();
    for (i, p) in sim.particles.iter().enumerate() {
        sim.spatial_hash.insert(i, p.position);
    }

    // We cannot borrow `sim.particles` mutably while iterating, so we collect
    // corrections into a temporary buffer and apply them afterwards.
    let particle_count = sim.particles.len();
    // Stack-allocated would be nice, but particle_count is dynamic.
    let mut corrections: Vec<Vec3> = vec![[0.0; 3]; particle_count];
    let mut correction_counts: Vec<u32> = vec![0; particle_count];

    for i in 0..particle_count {
        if sim.particles[i].pinned {
            continue;
        }

        let pos_i = sim.particles[i].position;
        let neighbors = sim.spatial_hash.query_neighbors(pos_i, min_dist);

        for &j in &neighbors {
            // Only process each pair once (i < j) and skip self.
            if j <= i {
                continue;
            }
            if sim.particles[j].pinned {
                continue;
            }

            // Skip pairs connected by a distance constraint.
            let lo = i.min(j);
            let hi = i.max(j);
            if sim.connected_pairs.contains(&(lo, hi)) {
                continue;
            }

            let pos_j = sim.particles[j].position;
            let diff = vec3_sub(&pos_j, &pos_i);
            let dist_sq = vec3_dot(&diff, &diff);

            if dist_sq < min_dist_sq && dist_sq > 1e-24 {
                let dist = dist_sq.sqrt();
                let overlap = min_dist - dist;
                let dir = vec3_scale(&diff, 1.0 / dist);

                // Push each particle half the overlap distance apart.
                let half_push = vec3_scale(&dir, overlap * 0.5);

                corrections[i] = vec3_sub(&corrections[i], &half_push);
                correction_counts[i] += 1;

                corrections[j] = vec3_add(&corrections[j], &half_push);
                correction_counts[j] += 1;
            }
        }
    }

    // Apply averaged corrections.
    for i in 0..particle_count {
        if correction_counts[i] > 0 && !sim.particles[i].pinned {
            let avg = vec3_scale(&corrections[i], 1.0 / correction_counts[i] as f32);
            sim.particles[i].position = vec3_add(&sim.particles[i].position, &avg);
        }
    }
}

// =========================================================================
// Pin resolution
// =========================================================================

/// Resolve pin targets to world-space positions from skeleton global
/// transforms and write them to the pinned particles.
pub(super) fn apply_pin_targets(
    sim: &mut ClothSimState,
    global_transforms: &[crate::asset::Mat4],
) {
    for pin in &sim.pin_targets {
        let node_idx = pin.node_index;
        if node_idx >= global_transforms.len() {
            continue;
        }
        let mat = &global_transforms[node_idx];
        let [ox, oy, oz] = pin.offset;
        let wx = mat[0][0] * ox + mat[1][0] * oy + mat[2][0] * oz + mat[3][0];
        let wy = mat[0][1] * ox + mat[1][1] * oy + mat[2][1] * oz + mat[3][1];
        let wz = mat[0][2] * ox + mat[1][2] * oy + mat[2][2] * oz + mat[3][2];

        let world_pos: Vec3 = [wx, wy, wz];
        for &pi in &pin.particle_indices {
            if pi < sim.particles.len() {
                sim.particles[pi].position = world_pos;
                sim.particles[pi].prev_position = world_pos;
            }
        }
    }
}

/// After constraint projection and collision, re-snap pinned particles.
pub(super) fn enforce_pins(sim: &mut ClothSimState) {
    // Pin targets already wrote the correct positions in apply_pin_targets.
    // If constraint projection or collision moved them, reset to pin position
    // stored in prev_position (which was also set by apply_pin_targets).
    for p in sim.particles.iter_mut() {
        if p.pinned {
            p.position = p.prev_position;
        }
    }
}
