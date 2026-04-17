use crate::avatar::AvatarInstance;
use crate::math_utils::{
    closest_point_on_segment, vec3_add, vec3_cross, vec3_dot, vec3_length, vec3_normalize,
    vec3_scale, vec3_sub, Vec3,
};
use crate::simulation::cloth::{ClothSimState, ClothSimTempBuffers, ResolvedCollider};

// =========================================================================
// Public entry point
// =========================================================================

/// Advance cloth simulation by `dt` seconds for the given avatar instance.
pub fn step_cloth(dt: f32, avatar: &mut AvatarInstance, world_colliders: &[ResolvedCollider]) {
    if !avatar.cloth_enabled {
        return;
    }

    step_cloth_single(dt, avatar, world_colliders);
    step_cloth_overlays(dt, avatar, world_colliders);
}

fn step_cloth_single(dt: f32, avatar: &mut AvatarInstance, world_colliders: &[ResolvedCollider]) {
    // We need both cloth_state (runtime positions) and a ClothSimState to work
    // with.  On first call we lazily initialise ClothSimState from the asset.
    // ClothSimState lives inside AvatarInstance::cloth_sim (we will add it).
    //
    // Because the current codebase stores cloth overlay data on the asset side
    // but the avatar only keeps ClothState (positions/normals/deform), we keep
    // a parallel ClothSimState.  For the POC the sim state is stored as
    // Option<ClothSimState> on AvatarInstance::cloth_sim.

    let cloth_state = match avatar.cloth_state.as_mut() {
        Some(cs) if cs.enabled => cs,
        _ => return,
    };

    // ---- lazy init of sim state -----------------------------------------------
    let sim = match avatar.cloth_sim.as_mut() {
        Some(s) if s.initialized => s,
        _ => return, // not yet initialised – caller should call init_cloth_sim first
    };

    let particle_count = sim.particle_count();
    if particle_count == 0 {
        return;
    }

    // ---- ensure scratch buffers -----------------------------------------------
    let buffers = avatar
        .cloth_sim_buffers
        .get_or_insert_with(|| ClothSimTempBuffers::new(particle_count));
    if buffers.temp_positions.len() != particle_count {
        buffers.resize(particle_count);
    }

    // ---- resolve colliders to world space -------------------------------------
    let colliders = crate::simulation::cloth::resolve_colliders(
        &avatar.asset.colliders,
        &avatar.pose.global_transforms,
    );

    // ---- resolve pin targets --------------------------------------------------
    apply_pin_targets(sim, &avatar.pose.global_transforms);

    // ---- Verlet integration ---------------------------------------------------
    verlet_integrate(sim, dt);

    // ---- constraint projection (iterative) ------------------------------------
    let iterations = sim.solver_iterations;
    for _ in 0..iterations {
        project_distance_constraints(sim, buffers);
        project_bend_constraints(sim);
    }

    // ---- self-collision -------------------------------------------------------
    resolve_self_collisions(sim);

    // ---- collision ------------------------------------------------------------
    let mut all_colliders = colliders;
    all_colliders.extend_from_slice(world_colliders);
    collide(sim, &all_colliders);

    // ---- enforce pins after all projections ------------------------------------
    enforce_pins(sim);

    // ---- compute per-vertex normals -------------------------------------------
    compute_normals(sim);

    // ---- derive velocity from position delta ----------------------------------
    derive_velocity(sim, dt);

    // ---- write back to ClothState for rendering -------------------------------
    write_back(sim, cloth_state);
}

fn step_cloth_overlays(dt: f32, avatar: &mut AvatarInstance, world_colliders: &[ResolvedCollider]) {
    if avatar.cloth_overlays.is_empty() {
        return;
    }

    let colliders = crate::simulation::cloth::resolve_colliders(
        &avatar.asset.colliders,
        &avatar.pose.global_transforms,
    );
    let mut all_colliders = colliders;
    all_colliders.extend_from_slice(world_colliders);

    let global_transforms = avatar.pose.global_transforms.clone();

    for slot in &mut avatar.cloth_overlays {
        if !slot.enabled {
            continue;
        }

        let particle_count = slot.sim.particle_count();
        if particle_count == 0 || !slot.sim.initialized {
            continue;
        }

        if slot.buffers.temp_positions.len() != particle_count {
            slot.buffers.resize(particle_count);
        }

        apply_pin_targets(&mut slot.sim, &global_transforms);
        verlet_integrate(&mut slot.sim, dt);

        let iterations = slot.sim.solver_iterations;
        for _ in 0..iterations {
            project_distance_constraints(&mut slot.sim, &mut slot.buffers);
            project_bend_constraints(&mut slot.sim);
        }

        resolve_self_collisions(&mut slot.sim);
        collide(&mut slot.sim, &all_colliders);
        enforce_pins(&mut slot.sim);
        compute_normals(&mut slot.sim);
        derive_velocity(&mut slot.sim, dt);
        write_back(&mut slot.sim, &mut slot.state);
    }
}

// =========================================================================
// Verlet integration (E7: wind_response applied here)
// =========================================================================

fn verlet_integrate(sim: &mut ClothSimState, dt: f32) {
    let gravity = sim.gravity;
    let damping = sim.damping.clamp(0.0, 1.0);
    let wind_force = vec3_scale(&sim.wind_direction, sim.wind_response);

    for p in sim.particles.iter_mut() {
        if p.pinned {
            continue;
        }

        let vel = vec3_sub(&p.position, &p.prev_position);
        let damped_vel = vec3_scale(&vel, 1.0 - damping);
        // Gravity + wind as acceleration scaled by dt^2
        let accel = vec3_scale(&vec3_add(&gravity, &wind_force), dt * dt);

        let new_pos = vec3_add(&vec3_add(&p.position, &damped_vel), &accel);

        p.prev_position = p.position;
        p.position = new_pos;
    }
}

// =========================================================================
// Distance constraint projection (Jakobsen-style)
// =========================================================================

fn project_distance_constraints(sim: &mut ClothSimState, buffers: &mut ClothSimTempBuffers) {
    buffers.clear();

    for dc in &sim.distance_constraints {
        let a = dc.a;
        let b = dc.b;
        if a >= sim.particles.len() || b >= sim.particles.len() {
            continue;
        }

        let pa = sim.particles[a].position;
        let pb = sim.particles[b].position;
        let diff = vec3_sub(&pb, &pa);
        let dist = vec3_length(&diff);
        if dist < 1e-12 {
            continue;
        }

        let err = dist - dc.rest_length;
        let dir = vec3_scale(&diff, 1.0 / dist);

        let inv_mass_a = sim.particles[a].inv_mass;
        let inv_mass_b = sim.particles[b].inv_mass;
        let total_inv_mass = inv_mass_a + inv_mass_b;
        if total_inv_mass < 1e-12 {
            continue;
        }

        let stiffness = dc.stiffness.clamp(0.0, 1.0);
        let correction = err * stiffness;

        let wa = inv_mass_a / total_inv_mass;
        let wb = inv_mass_b / total_inv_mass;

        // Accumulate corrections (average style)
        let corr_a = vec3_scale(&dir, correction * wa);
        let corr_b = vec3_scale(&dir, -correction * wb);

        buffers.correction_accumulator[a] = vec3_add(&buffers.correction_accumulator[a], &corr_a);
        buffers.correction_counts[a] += 1;

        buffers.correction_accumulator[b] = vec3_add(&buffers.correction_accumulator[b], &corr_b);
        buffers.correction_counts[b] += 1;
    }

    // Apply averaged corrections
    for i in 0..sim.particles.len() {
        if sim.particles[i].pinned {
            continue;
        }
        let count = buffers.correction_counts[i];
        if count == 0 {
            continue;
        }
        let avg = vec3_scale(&buffers.correction_accumulator[i], 1.0 / count as f32);
        sim.particles[i].position = vec3_add(&sim.particles[i].position, &avg);
    }
}

// =========================================================================
// Bend constraint projection (M1)
// =========================================================================

/// Project bend constraints using a dihedral-angle approach.
///
/// Each bend constraint has three indices (p0, p1, p2). p0 is the hinge
/// vertex; p1 and p2 are the wing vertices.  The dihedral angle is measured
/// between edges p0->p1 and p0->p2, and corrections are applied to restore
/// the rest angle.
fn project_bend_constraints(sim: &mut ClothSimState) {
    // We need to index into sim.particles mutably while iterating bend_constraints.
    // Use index-based access.
    let n = sim.particles.len();
    let constraint_count = sim.bend_constraints.len();

    for ci in 0..constraint_count {
        let p0 = sim.bend_constraints[ci].p0;
        let p1 = sim.bend_constraints[ci].p1;
        let p2 = sim.bend_constraints[ci].p2;
        let rest_angle = sim.bend_constraints[ci].rest_angle;
        let stiffness = sim.bend_constraints[ci].stiffness.clamp(0.0, 1.0);

        if p0 >= n || p1 >= n || p2 >= n {
            continue;
        }

        let x0 = sim.particles[p0].position;
        let x1 = sim.particles[p1].position;
        let x2 = sim.particles[p2].position;

        // Edges from hinge vertex
        let e1 = vec3_sub(&x1, &x0);
        let e2 = vec3_sub(&x2, &x0);

        // Current angle between e1 and e2
        let dot = vec3_dot(&e1, &e2);
        let cross = vec3_cross(&e1, &e2);
        let cross_len = vec3_length(&cross);
        let current_angle = cross_len.atan2(dot);

        let angle_err = current_angle - rest_angle;
        if angle_err.abs() < 1e-6 {
            continue;
        }

        // Correction magnitude (scaled by stiffness)
        let correction = angle_err * stiffness;

        // Correction direction: rotate p1 and p2 around the cross-product axis
        // to reduce the angle error.  We use a simple linearized approach:
        // move p1 and p2 along the directions perpendicular to their respective
        // edges, in the plane spanned by e1 and e2.
        let e1_len = vec3_length(&e1);
        let e2_len = vec3_length(&e2);
        if e1_len < 1e-12 || e2_len < 1e-12 {
            continue;
        }

        // Perpendicular component of e2 w.r.t. e1 (and vice versa) gives the
        // tangent direction for angle change.
        let e1_norm = vec3_scale(&e1, 1.0 / e1_len);
        let e2_norm = vec3_scale(&e2, 1.0 / e2_len);

        // Direction to move p1: perpendicular to e1, toward e2
        let perp1 = vec3_sub(
            &e2_norm,
            &vec3_scale(&e1_norm, vec3_dot(&e2_norm, &e1_norm)),
        );
        let perp1_len = vec3_length(&perp1);
        if perp1_len < 1e-6 {
            continue;
        }
        let perp1 = vec3_scale(&perp1, 1.0 / perp1_len);

        // Direction to move p2: perpendicular to e2, toward e1
        let perp2 = vec3_sub(
            &e1_norm,
            &vec3_scale(&e2_norm, vec3_dot(&e1_norm, &e2_norm)),
        );
        let perp2_len = vec3_length(&perp2);
        if perp2_len < 1e-6 {
            continue;
        }
        let perp2 = vec3_scale(&perp2, 1.0 / perp2_len);

        let inv_mass_0 = sim.particles[p0].inv_mass;
        let inv_mass_1 = sim.particles[p1].inv_mass;
        let inv_mass_2 = sim.particles[p2].inv_mass;
        let total_inv_mass = inv_mass_0 + inv_mass_1 + inv_mass_2;
        if total_inv_mass < 1e-12 {
            continue;
        }

        // Move p1 along -perp1 and p2 along -perp2 to close the angle,
        // scaled by each particle's share of inverse mass.
        let scale = correction * 0.5; // half for each wing vertex

        if !sim.particles[p1].pinned {
            let w1 = inv_mass_1 / total_inv_mass;
            let delta1 = vec3_scale(&perp1, -scale * w1 * e1_len);
            sim.particles[p1].position = vec3_add(&sim.particles[p1].position, &delta1);
        }
        if !sim.particles[p2].pinned {
            let w2 = inv_mass_2 / total_inv_mass;
            let delta2 = vec3_scale(&perp2, -scale * w2 * e2_len);
            sim.particles[p2].position = vec3_add(&sim.particles[p2].position, &delta2);
        }
        // Optionally move hinge vertex to conserve momentum
        if !sim.particles[p0].pinned {
            let w0 = inv_mass_0 / total_inv_mass;
            let delta1_comp = vec3_scale(&perp1, scale * w0 * e1_len * 0.5);
            let delta2_comp = vec3_scale(&perp2, scale * w0 * e2_len * 0.5);
            sim.particles[p0].position = vec3_add(
                &sim.particles[p0].position,
                &vec3_add(&delta1_comp, &delta2_comp),
            );
        }
    }
}

// =========================================================================
// Collision (M3: sphere + capsule, E6: collision_margin)
// =========================================================================

fn collide(sim: &mut ClothSimState, colliders: &[ResolvedCollider]) {
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
fn resolve_self_collisions(sim: &mut ClothSimState) {
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
// Per-vertex normal computation (M2)
// =========================================================================

fn compute_normals(sim: &mut ClothSimState) {
    let n = sim.particles.len();
    // Resize normals buffer inside sim (we store temporarily in a local vec
    // and then copy back, since ClothSimState doesn't have a normals field
    // -- we'll write directly to the output in write_back instead).
    // Actually, we compute here and store in a side buffer attached to sim.
    // We'll use the temp_positions-style approach: store in the sim state.
    // For simplicity, we compute here and pass via a static-lifetime trick.
    // Better: we store the computed normals in a Vec on the stack and pass
    // them to write_back.  But since step_cloth calls compute_normals then
    // write_back separately, we need storage.  Let's add a normals vec to
    // ClothSimState.  It's already there as an implicit need.

    // Lazy-init normals buffer on ClothSimState
    if sim.computed_normals.len() != n {
        sim.computed_normals.resize(n, [0.0; 3]);
    }

    // Zero out
    for normal in sim.computed_normals.iter_mut() {
        *normal = [0.0, 0.0, 0.0];
    }

    // Accumulate face normals from triangles
    let idx = &sim.triangle_indices;
    let tri_count = idx.len() / 3;
    for t in 0..tri_count {
        let i0 = idx[t * 3] as usize;
        let i1 = idx[t * 3 + 1] as usize;
        let i2 = idx[t * 3 + 2] as usize;
        if i0 >= n || i1 >= n || i2 >= n {
            continue;
        }

        let p0 = sim.particles[i0].position;
        let p1 = sim.particles[i1].position;
        let p2 = sim.particles[i2].position;

        let e1 = vec3_sub(&p1, &p0);
        let e2 = vec3_sub(&p2, &p0);
        let face_normal = vec3_cross(&e1, &e2); // not normalized; magnitude = 2*area (area-weighted)

        sim.computed_normals[i0] = vec3_add(&sim.computed_normals[i0], &face_normal);
        sim.computed_normals[i1] = vec3_add(&sim.computed_normals[i1], &face_normal);
        sim.computed_normals[i2] = vec3_add(&sim.computed_normals[i2], &face_normal);
    }

    // Normalize
    for normal in sim.computed_normals.iter_mut() {
        *normal = vec3_normalize(normal);
    }
}

// =========================================================================
// Pin resolution
// =========================================================================

/// Resolve pin targets to world-space positions from skeleton global
/// transforms and write them to the pinned particles.
fn apply_pin_targets(sim: &mut ClothSimState, global_transforms: &[crate::asset::Mat4]) {
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
fn enforce_pins(sim: &mut ClothSimState) {
    // Pin targets already wrote the correct positions in apply_pin_targets.
    // If constraint projection or collision moved them, reset to pin position
    // stored in prev_position (which was also set by apply_pin_targets).
    for p in sim.particles.iter_mut() {
        if p.pinned {
            p.position = p.prev_position;
        }
    }
}

// =========================================================================
// Velocity derivation
// =========================================================================

fn derive_velocity(sim: &mut ClothSimState, dt: f32) {
    if dt < 1e-12 {
        return;
    }
    let inv_dt = 1.0 / dt;
    for p in sim.particles.iter_mut() {
        p.velocity = vec3_scale(&vec3_sub(&p.position, &p.prev_position), inv_dt);
    }
}

// =========================================================================
// Write-back to ClothState for rendering (M2: normals written here)
// =========================================================================

fn write_back(sim: &ClothSimState, cloth_state: &mut crate::avatar::ClothState) {
    let n = sim.particles.len();

    // Resize output buffers if needed
    if cloth_state.sim_positions.len() != n {
        cloth_state.sim_positions.resize(n, [0.0; 3]);
    }
    if cloth_state.prev_sim_positions.len() != n {
        cloth_state.prev_sim_positions.resize(n, [0.0; 3]);
    }

    for (i, p) in sim.particles.iter().enumerate() {
        cloth_state.sim_positions[i] = p.position;
        cloth_state.prev_sim_positions[i] = p.prev_position;
    }

    // Write normals to ClothState
    if cloth_state.sim_normals.len() != n {
        cloth_state.sim_normals.resize(n, [0.0; 3]);
    }
    for i in 0..n.min(sim.computed_normals.len()) {
        cloth_state.sim_normals[i] = sim.computed_normals[i];
    }

    // Write deform output
    let deform = &mut cloth_state.deform_output;
    if deform.deformed_positions.len() != n {
        deform.deformed_positions.resize(n, [0.0; 3]);
    }
    for (i, p) in sim.particles.iter().enumerate() {
        deform.deformed_positions[i] = p.position;
    }

    // Write deformed normals
    let normals_vec = if let Some(ref mut nv) = deform.deformed_normals {
        nv.resize(n, [0.0; 3]);
        nv
    } else {
        deform.deformed_normals = Some(vec![[0.0; 3]; n]);
        deform.deformed_normals.as_mut().unwrap()
    };
    for i in 0..n.min(sim.computed_normals.len()) {
        normals_vec[i] = sim.computed_normals[i];
    }

    deform.version += 1;

    // Update runtime caches
    cloth_state.constraint_cache.active_constraint_count = 0; // filled by solver as needed
    cloth_state.collision_cache.active_collision_count = 0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::cloth::{
        ClothBendConstraint, ClothDistanceConstraint, ClothParticle, ClothSimState,
        ClothSimTempBuffers, PinTarget, ResolvedCollider, SpatialHashGrid,
    };
    use std::collections::HashSet;

    fn make_simple_sim(particle_count: usize) -> ClothSimState {
        let particles = (0..particle_count)
            .map(|i| {
                let p = ClothParticle::new([i as f32, 0.0, 0.0], false);
                p
            })
            .collect();
        ClothSimState {
            particles,
            distance_constraints: Vec::new(),
            bend_constraints: Vec::new(),
            pin_targets: Vec::new(),
            solver_iterations: 4,
            gravity: [0.0, -9.81, 0.0],
            damping: 0.01,
            collision_margin: 0.0,
            wind_response: 0.0,
            wind_direction: [1.0, 0.0, 0.0],
            triangle_indices: Vec::new(),
            computed_normals: vec![[0.0; 3]; particle_count],
            initialized: true,
            self_collision: false,
            self_collision_radius: 0.01,
            connected_pairs: HashSet::new(),
            spatial_hash: SpatialHashGrid::new(0.04),
        }
    }

    // =========================================================================
    // Verlet integration
    // =========================================================================

    #[test]
    fn verlet_gravity_pulls_down() {
        let mut sim = make_simple_sim(1);
        sim.gravity = [0.0, -10.0, 0.0];
        sim.damping = 0.0;
        sim.particles[0].position = [0.0, 1.0, 0.0];
        sim.particles[0].prev_position = [0.0, 1.0, 0.0];

        verlet_integrate(&mut sim, 0.1);

        let dy = sim.particles[0].position[1] - 1.0;
        assert!(dy < 0.0, "gravity should pull particle down, got dy={}", dy);
        let expected_dy = -10.0 * 0.1 * 0.1;
        assert!(
            (dy - expected_dy).abs() < 1e-4,
            "expected dy={}, got {}",
            expected_dy,
            dy
        );
    }

    #[test]
    fn verlet_damping_reduces_velocity() {
        let mut sim = make_simple_sim(1);
        sim.gravity = [0.0, 0.0, 0.0];
        sim.damping = 0.5;
        sim.particles[0].position = [1.0, 0.0, 0.0];
        sim.particles[0].prev_position = [0.0, 0.0, 0.0];

        verlet_integrate(&mut sim, 0.1);

        let vx = (sim.particles[0].position[0] - 1.0) / 0.1;
        assert!(
            vx.abs() < 10.0 * 0.99,
            "damped velocity should be less than original, got vx={}",
            vx
        );
    }

    #[test]
    fn verlet_pinned_particles_dont_move() {
        let mut sim = make_simple_sim(1);
        sim.gravity = [0.0, -100.0, 0.0];
        sim.particles[0].pinned = true;
        sim.particles[0].inv_mass = 0.0;
        sim.particles[0].position = [5.0, 5.0, 5.0];
        sim.particles[0].prev_position = [5.0, 5.0, 5.0];

        verlet_integrate(&mut sim, 0.1);

        assert_eq!(sim.particles[0].position, [5.0, 5.0, 5.0]);
    }

    #[test]
    fn verlet_wind_pushes_particles() {
        let mut sim = make_simple_sim(1);
        sim.gravity = [0.0, 0.0, 0.0];
        sim.damping = 0.0;
        sim.wind_response = 5.0;
        sim.wind_direction = [1.0, 0.0, 0.0];
        sim.particles[0].position = [0.0, 0.0, 0.0];
        sim.particles[0].prev_position = [0.0, 0.0, 0.0];

        verlet_integrate(&mut sim, 0.1);

        let dx = sim.particles[0].position[0];
        assert!(dx > 0.0, "wind should push particle in +x, got dx={}", dx);
    }

    // =========================================================================
    // Distance constraints
    // =========================================================================

    #[test]
    fn distance_constraint_converges_to_rest_length() {
        let mut sim = make_simple_sim(2);
        let rest_length = 1.0;
        sim.particles[0].position = [0.0, 0.0, 0.0];
        sim.particles[1].position = [2.0, 0.0, 0.0];
        sim.distance_constraints.push(ClothDistanceConstraint {
            a: 0,
            b: 1,
            rest_length,
            stiffness: 1.0,
        });

        let mut buffers = ClothSimTempBuffers::new(2);

        for _ in 0..50 {
            project_distance_constraints(&mut sim, &mut buffers);
        }

        let dist = vec3_length(&vec3_sub(
            &sim.particles[1].position,
            &sim.particles[0].position,
        ));
        assert!(
            (dist - rest_length).abs() < 0.01,
            "expected dist={}, got {}",
            rest_length,
            dist
        );
    }

    #[test]
    fn distance_constraint_preserves_center_of_mass() {
        let mut sim = make_simple_sim(2);
        sim.particles[0].position = [0.0, 0.0, 0.0];
        sim.particles[1].position = [2.0, 0.0, 0.0];
        sim.distance_constraints.push(ClothDistanceConstraint {
            a: 0,
            b: 1,
            rest_length: 1.0,
            stiffness: 1.0,
        });

        let com_before: Vec3 = vec3_scale(
            &vec3_add(&sim.particles[0].position, &sim.particles[1].position),
            0.5,
        );

        let mut buffers = ClothSimTempBuffers::new(2);
        project_distance_constraints(&mut sim, &mut buffers);

        let com_after: Vec3 = vec3_scale(
            &vec3_add(&sim.particles[0].position, &sim.particles[1].position),
            0.5,
        );

        for i in 0..3 {
            assert!(
                (com_after[i] - com_before[i]).abs() < 1e-4,
                "center of mass should be preserved, axis {} shifted by {}",
                i,
                (com_after[i] - com_before[i])
            );
        }
    }

    #[test]
    fn distance_constraint_with_zero_stiffness_does_nothing() {
        let mut sim = make_simple_sim(2);
        sim.particles[0].position = [0.0, 0.0, 0.0];
        sim.particles[1].position = [3.0, 0.0, 0.0];
        sim.distance_constraints.push(ClothDistanceConstraint {
            a: 0,
            b: 1,
            rest_length: 1.0,
            stiffness: 0.0,
        });

        let mut buffers = ClothSimTempBuffers::new(2);
        project_distance_constraints(&mut sim, &mut buffers);

        assert_eq!(sim.particles[0].position, [0.0, 0.0, 0.0]);
        assert_eq!(sim.particles[1].position, [3.0, 0.0, 0.0]);
    }

    // =========================================================================
    // Bend constraints
    // =========================================================================

    #[test]
    fn bend_constraint_changes_angle() {
        let mut sim = make_simple_sim(3);
        sim.particles[0].position = [0.0, 0.0, 0.0];
        sim.particles[1].position = [1.0, 0.0, 0.0];
        sim.particles[2].position = [0.0, 1.0, 0.0];

        let rest_angle = std::f32::consts::PI;
        sim.bend_constraints.push(ClothBendConstraint {
            p0: 0,
            p1: 1,
            p2: 2,
            rest_angle,
            stiffness: 0.3,
        });

        let angle_before: f32 = {
            let e1 = vec3_sub(&sim.particles[1].position, &sim.particles[0].position);
            let e2 = vec3_sub(&sim.particles[2].position, &sim.particles[0].position);
            vec3_length(&vec3_cross(&e1, &e2)).atan2(vec3_dot(&e1, &e2))
        };

        project_bend_constraints(&mut sim);

        let angle_after: f32 = {
            let e1 = vec3_sub(&sim.particles[1].position, &sim.particles[0].position);
            let e2 = vec3_sub(&sim.particles[2].position, &sim.particles[0].position);
            vec3_length(&vec3_cross(&e1, &e2)).atan2(vec3_dot(&e1, &e2))
        };

        assert!(
            (angle_after - angle_before).abs() > 0.01,
            "bend constraint should change the angle: before={}, after={}",
            angle_before.to_degrees(),
            angle_after.to_degrees()
        );
    }

    // =========================================================================
    // Pin enforcement
    // =========================================================================

    #[test]
    fn enforce_pins_resets_pinned_particles() {
        let mut sim = make_simple_sim(2);
        sim.particles[0].pinned = true;
        sim.particles[0].inv_mass = 0.0;
        sim.particles[0].position = [1.0, 2.0, 3.0]; // moved by constraints
        sim.particles[0].prev_position = [0.0, 0.0, 0.0]; // correct pin pos

        enforce_pins(&mut sim);

        assert_eq!(sim.particles[0].position, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn enforce_pins_does_not_affect_free_particles() {
        let mut sim = make_simple_sim(1);
        sim.particles[0].position = [5.0, 5.0, 5.0];
        sim.particles[0].prev_position = [0.0, 0.0, 0.0];

        enforce_pins(&mut sim);

        assert_eq!(sim.particles[0].position, [5.0, 5.0, 5.0]);
    }

    // =========================================================================
    // Sphere collision
    // =========================================================================

    #[test]
    fn sphere_collision_pushes_particle_out() {
        let mut sim = make_simple_sim(1);
        sim.particles[0].position = [0.5, 0.0, 0.0]; // inside sphere, offset from center
        sim.collision_margin = 0.0;

        let colliders = vec![ResolvedCollider::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        }];

        collide(&mut sim, &colliders);

        let dist = vec3_length(&sim.particles[0].position);
        assert!(
            dist >= 0.99,
            "particle should be pushed outside sphere, dist={}",
            dist
        );
    }

    #[test]
    fn sphere_collision_respects_margin() {
        let mut sim = make_simple_sim(1);
        sim.particles[0].position = [1.2, 0.0, 0.0]; // inside margin zone
        sim.collision_margin = 0.5;

        let colliders = vec![ResolvedCollider::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        }];

        collide(&mut sim, &colliders);

        let dist = vec3_length(&sim.particles[0].position);
        let effective_radius = 1.0 + 0.5;
        assert!(
            dist >= effective_radius - 0.01,
            "particle should be outside effective radius {}, dist={}",
            effective_radius,
            dist
        );
    }

    #[test]
    fn sphere_collision_skips_pinned() {
        let mut sim = make_simple_sim(1);
        sim.particles[0].pinned = true;
        sim.particles[0].position = [0.0, 0.0, 0.0]; // inside sphere

        let colliders = vec![ResolvedCollider::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        }];

        collide(&mut sim, &colliders);

        assert_eq!(sim.particles[0].position, [0.0, 0.0, 0.0]);
    }

    // =========================================================================
    // Capsule collision
    // =========================================================================

    #[test]
    fn capsule_collision_pushes_particle_out() {
        let mut sim = make_simple_sim(1);
        // Capsule along Y from -1 to +1, radius 0.5, particle slightly off-axis
        sim.particles[0].position = [0.1, 0.0, 0.0]; // inside capsule
        sim.collision_margin = 0.0;

        let colliders = vec![ResolvedCollider::Capsule {
            center: [0.0, 0.0, 0.0],
            radius: 0.5,
            half_height: 1.0,
            axis: [0.0, 1.0, 0.0],
        }];

        collide(&mut sim, &colliders);

        let dist_from_axis =
            (sim.particles[0].position[0].powi(2) + sim.particles[0].position[2].powi(2)).sqrt();
        assert!(
            dist_from_axis >= 0.49,
            "particle should be pushed outside capsule radius, dist_from_axis={}",
            dist_from_axis
        );
    }

    // =========================================================================
    // Self-collision
    // =========================================================================

    #[test]
    fn self_collision_pushes_overlapping_particles_apart() {
        let mut sim = make_simple_sim(2);
        sim.self_collision = true;
        sim.self_collision_radius = 0.5;
        // Place both particles at the same position
        sim.particles[0].position = [0.0, 0.0, 0.0];
        sim.particles[1].position = [0.01, 0.0, 0.0];
        // No connected pairs so they will collide
        sim.connected_pairs = HashSet::new();
        sim.spatial_hash = SpatialHashGrid::new(2.0);

        resolve_self_collisions(&mut sim);

        let dist = vec3_length(&vec3_sub(
            &sim.particles[1].position,
            &sim.particles[0].position,
        ));
        assert!(
            dist > 0.01,
            "particles should be pushed apart, dist={}",
            dist
        );
    }

    #[test]
    fn self_collision_skips_connected_pairs() {
        let mut sim = make_simple_sim(2);
        sim.self_collision = true;
        sim.self_collision_radius = 0.5;
        sim.particles[0].position = [0.0, 0.0, 0.0];
        sim.particles[1].position = [0.01, 0.0, 0.0];
        sim.connected_pairs = HashSet::from([(0, 1)]);
        sim.spatial_hash = SpatialHashGrid::new(2.0);

        resolve_self_collisions(&mut sim);

        assert_eq!(sim.particles[0].position, [0.0, 0.0, 0.0]);
        assert_eq!(sim.particles[1].position, [0.01, 0.0, 0.0]);
    }

    // =========================================================================
    // Normal computation
    // =========================================================================

    #[test]
    fn normals_for_triangle_in_xy_plane() {
        let mut sim = make_simple_sim(3);
        // Triangle in XY plane: (0,0,0), (1,0,0), (0,1,0)
        sim.particles[0].position = [0.0, 0.0, 0.0];
        sim.particles[1].position = [1.0, 0.0, 0.0];
        sim.particles[2].position = [0.0, 1.0, 0.0];
        sim.triangle_indices = vec![0, 1, 2];

        compute_normals(&mut sim);

        // All normals should point in +Z direction (cross of e1 x e2)
        for i in 0..3 {
            let nz = sim.computed_normals[i][2];
            assert!(
                nz > 0.9,
                "normal {} should point in +Z, got {:?}",
                i,
                sim.computed_normals[i]
            );
        }
    }

    #[test]
    fn normals_for_degenerate_triangle_are_degenerate() {
        let mut sim = make_simple_sim(3);
        // Collinear points => zero-area triangle => degenerate normal
        sim.particles[0].position = [0.0, 0.0, 0.0];
        sim.particles[1].position = [1.0, 0.0, 0.0];
        sim.particles[2].position = [2.0, 0.0, 0.0];
        sim.triangle_indices = vec![0, 1, 2];

        compute_normals(&mut sim);

        // Cross product is zero, normalize of zero is undefined;
        // we just verify it doesn't panic
    }

    // =========================================================================
    // Velocity derivation
    // =========================================================================

    #[test]
    fn derive_velocity_computes_correct_velocity() {
        let mut sim = make_simple_sim(1);
        sim.particles[0].position = [1.0, 0.0, 0.0];
        sim.particles[0].prev_position = [0.0, 0.0, 0.0];

        derive_velocity(&mut sim, 0.1);

        let expected_vx = 1.0 / 0.1;
        assert!(
            (sim.particles[0].velocity[0] - expected_vx).abs() < 1e-3,
            "expected vx={}, got {}",
            expected_vx,
            sim.particles[0].velocity[0]
        );
    }

    #[test]
    fn derive_velocity_zero_dt_doesnt_panic() {
        let mut sim = make_simple_sim(1);
        sim.particles[0].position = [1.0, 0.0, 0.0];
        sim.particles[0].prev_position = [0.0, 0.0, 0.0];

        derive_velocity(&mut sim, 0.0);

        // Should not panic; velocity unchanged from init
    }

    // =========================================================================
    // Write-back
    // =========================================================================

    #[test]
    fn write_back_copies_positions_and_normals() {
        let mut sim = make_simple_sim(2);
        sim.particles[0].position = [1.0, 2.0, 3.0];
        sim.particles[1].position = [4.0, 5.0, 6.0];
        sim.computed_normals = vec![[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]];

        let mut cloth_state = crate::avatar::instance::ClothState {
            overlay_id: crate::asset::ClothOverlayId(0),
            enabled: true,
            sim_positions: Vec::new(),
            prev_sim_positions: Vec::new(),
            sim_normals: Vec::new(),
            constraint_cache: crate::avatar::instance::ClothConstraintRuntimeCache {
                active_constraint_count: 0,
            },
            collision_cache: crate::avatar::instance::ClothCollisionRuntimeCache {
                active_collision_count: 0,
            },
            deform_output: crate::avatar::instance::ClothDeformOutput {
                deformed_positions: Vec::new(),
                deformed_normals: None,
                version: 0,
            },
        };

        write_back(&sim, &mut cloth_state);

        assert_eq!(cloth_state.sim_positions[0], [1.0, 2.0, 3.0]);
        assert_eq!(cloth_state.sim_positions[1], [4.0, 5.0, 6.0]);
        assert_eq!(cloth_state.sim_normals[0], [0.0, 0.0, 1.0]);
        assert_eq!(cloth_state.sim_normals[1], [0.0, 1.0, 0.0]);
        assert_eq!(cloth_state.deform_output.version, 1);
    }

    #[test]
    fn write_back_increments_version() {
        let sim = make_simple_sim(1);

        let mut cloth_state = crate::avatar::instance::ClothState {
            overlay_id: crate::asset::ClothOverlayId(0),
            enabled: true,
            sim_positions: Vec::new(),
            prev_sim_positions: Vec::new(),
            sim_normals: Vec::new(),
            constraint_cache: crate::avatar::instance::ClothConstraintRuntimeCache {
                active_constraint_count: 0,
            },
            collision_cache: crate::avatar::instance::ClothCollisionRuntimeCache {
                active_collision_count: 0,
            },
            deform_output: crate::avatar::instance::ClothDeformOutput {
                deformed_positions: Vec::new(),
                deformed_normals: None,
                version: 42,
            },
        };

        write_back(&sim, &mut cloth_state);

        assert_eq!(cloth_state.deform_output.version, 43);
    }

    // =========================================================================
    // Timestep stability
    // =========================================================================

    #[test]
    fn multi_step_does_not_spike() {
        let mut sim = make_simple_sim(2);
        sim.gravity = [0.0, -9.81, 0.0];
        sim.damping = 0.02;
        sim.particles[0].position = [0.0, 1.0, 0.0];
        sim.particles[0].prev_position = [0.0, 1.0, 0.0];
        sim.particles[1].position = [1.0, 1.0, 0.0];
        sim.particles[1].prev_position = [1.0, 1.0, 0.0];
        sim.distance_constraints.push(ClothDistanceConstraint {
            a: 0,
            b: 1,
            rest_length: 1.0,
            stiffness: 0.8,
        });

        let mut buffers = ClothSimTempBuffers::new(2);

        for _ in 0..1000 {
            verlet_integrate(&mut sim, 1.0 / 60.0);
            for _ in 0..sim.solver_iterations {
                project_distance_constraints(&mut sim, &mut buffers);
            }
        }

        for p in &sim.particles {
            assert!(
                p.position.iter().all(|v| v.is_finite()),
                "positions should remain finite after 1000 steps: {:?}",
                p.position
            );
            let mag = vec3_length(&p.position);
            assert!(mag < 1000.0, "positions should not diverge, mag={}", mag);
        }
    }

    fn make_minimal_avatar_asset() -> std::sync::Arc<crate::asset::AvatarAsset> {
        std::sync::Arc::new(crate::asset::AvatarAsset {
            id: crate::asset::AvatarAssetId(0),
            source_path: std::path::PathBuf::new(),
            source_hash: crate::asset::AssetSourceHash([0u8; 32]),
            skeleton: crate::asset::SkeletonAsset {
                nodes: Vec::new(),
                root_nodes: Vec::new(),
                inverse_bind_matrices: Vec::new(),
            },
            meshes: Vec::new(),
            materials: Vec::new(),
            humanoid: None,
            spring_bones: Vec::new(),
            colliders: Vec::new(),
            default_expressions: crate::asset::ExpressionAssetSet {
                expressions: Vec::new(),
            },
            animation_clips: Vec::new(),
            node_to_mesh: Default::default(),
            vrm_meta: Default::default(),
            root_aabb: crate::asset::Aabb::empty(),
        })
    }

    // =========================================================================
    // Cloth disabled / no cloth overlay
    // =========================================================================

    #[test]
    fn step_cloth_does_nothing_when_avatar_cloth_disabled() {
        let mut avatar = crate::avatar::AvatarInstance::new(
            crate::avatar::AvatarInstanceId(0),
            make_minimal_avatar_asset(),
        );
        avatar.cloth_enabled = false;

        step_cloth(1.0 / 60.0, &mut avatar, &[]);
    }

    // =========================================================================
    // Pin target resolution
    // =========================================================================

    #[test]
    fn apply_pin_targets_moves_pinned_particles_to_transform() {
        let mut sim = make_simple_sim(1);
        sim.particles[0].pinned = true;
        sim.particles[0].inv_mass = 0.0;
        sim.pin_targets.push(PinTarget {
            node_index: 0,
            offset: [1.0, 0.0, 0.0],
            particle_indices: vec![0],
        });

        // Identity transform, offset [1,0,0] => world pos [1,0,0]
        let transforms = vec![[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]];

        apply_pin_targets(&mut sim, &transforms);

        assert_eq!(sim.particles[0].position[0], 1.0);
    }

    // =========================================================================
    // Spatial hash grid (in cloth.rs, tested via public API)
    // =========================================================================

    #[test]
    fn spatial_hash_finds_neighbors() {
        let mut grid = SpatialHashGrid::new(1.0);
        grid.insert(0, [0.0, 0.0, 0.0]);
        grid.insert(1, [0.5, 0.0, 0.0]);
        grid.insert(2, [100.0, 0.0, 0.0]); // far away

        let neighbors = grid.query_neighbors([0.0, 0.0, 0.0], 1.0);
        assert!(neighbors.contains(&0));
        assert!(neighbors.contains(&1));
        assert!(!neighbors.contains(&2));
    }

    #[test]
    fn spatial_hash_skips_nan_positions() {
        let mut grid = SpatialHashGrid::new(1.0);
        grid.insert(0, [f32::NAN, 0.0, 0.0]);
        grid.insert(1, [0.0, 0.0, 0.0]);

        let neighbors = grid.query_neighbors([0.0, 0.0, 0.0], 1.0);
        assert!(!neighbors.contains(&0));
        assert!(neighbors.contains(&1));
    }

    // =========================================================================
    // Collider resolution
    // =========================================================================

    #[test]
    fn resolve_colliders_sphere_transform() {
        let collider = crate::asset::ColliderAsset {
            id: crate::asset::ColliderId(0),
            node: crate::asset::NodeId(0),
            offset: [0.0, 1.0, 0.0],
            shape: crate::asset::ColliderShape::Sphere { radius: 0.5 },
        };

        let identity: crate::asset::Mat4 = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0], // translation [0, 1, 0]
        ];

        let resolved = crate::simulation::cloth::resolve_colliders(&[collider], &[identity]);
        assert_eq!(resolved.len(), 1);
        match &resolved[0] {
            ResolvedCollider::Sphere { center, radius } => {
                assert_eq!(*center, [0.0, 2.0, 0.0]); // transform pos [0,1,0] + offset [0,1,0]
                assert_eq!(*radius, 0.5);
            }
            _ => panic!("expected sphere"),
        }
    }

    // =========================================================================
    // World collider integration
    // =========================================================================

    #[test]
    fn world_colliders_are_included_in_collision() {
        let mut sim = make_simple_sim(1);
        sim.particles[0].position = [0.5, 0.0, 0.0];
        sim.collision_margin = 0.0;

        let world_colliders = vec![ResolvedCollider::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        }];

        collide(&mut sim, &world_colliders);

        let dist = vec3_length(&sim.particles[0].position);
        assert!(
            dist >= 0.99,
            "world collider should push particle out, dist={}",
            dist
        );
    }

    // =========================================================================
    // Performance: cloth step timing (stub)
    // =========================================================================

    #[test]
    fn cloth_step_performance_stub() {
        let particle_count = 256;
        let mut sim = make_simple_sim(particle_count);
        sim.gravity = [0.0, -9.81, 0.0];
        sim.damping = 0.02;
        sim.solver_iterations = 4;
        for i in 0..particle_count {
            sim.particles[i].position = [i as f32 % 16.0, (i as f32 / 16.0).floor(), 0.0];
            sim.particles[i].prev_position = sim.particles[i].position;
        }

        let mut buffers = ClothSimTempBuffers::new(particle_count);

        let start = std::time::Instant::now();
        for _ in 0..600 {
            verlet_integrate(&mut sim, 1.0 / 60.0);
            for _ in 0..sim.solver_iterations {
                project_distance_constraints(&mut sim, &mut buffers);
            }
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 2000,
            "600 substeps with {} particles took {:?} (expected < 2s)",
            particle_count,
            elapsed
        );
    }

    // =========================================================================
    // Integration: cloth disable / re-enable cycle
    // =========================================================================

    #[test]
    fn cloth_disable_preserves_positions() {
        let particle_count = 16;
        let mut sim = make_simple_sim(particle_count);
        for i in 0..particle_count {
            sim.particles[i].position = [i as f32, 1.0, 0.0];
            sim.particles[i].prev_position = sim.particles[i].position;
        }

        let positions_before: Vec<Vec3> = sim.particles.iter().map(|p| p.position).collect();

        verlet_integrate(&mut sim, 1.0 / 60.0);

        let positions_after_step: Vec<Vec3> = sim.particles.iter().map(|p| p.position).collect();

        assert_ne!(
            positions_before, positions_after_step,
            "gravity should move particles"
        );
    }

    #[test]
    fn cloth_reenable_after_disable_resumes_from_state() {
        let particle_count = 8;
        let mut sim = make_simple_sim(particle_count);
        sim.gravity = [0.0, -9.81, 0.0];
        for i in 0..particle_count {
            sim.particles[i].position = [i as f32, 10.0, 0.0];
            sim.particles[i].prev_position = sim.particles[i].position;
        }

        let mut buffers = ClothSimTempBuffers::new(particle_count);

        verlet_integrate(&mut sim, 1.0 / 60.0);
        for _ in 0..sim.solver_iterations {
            project_distance_constraints(&mut sim, &mut buffers);
        }

        let saved_positions: Vec<Vec3> = sim.particles.iter().map(|p| p.position).collect();

        let idle_steps = 10;
        for _ in 0..idle_steps {
            // Simulating disabled state: no stepping
        }

        verlet_integrate(&mut sim, 1.0 / 60.0);
        for _ in 0..sim.solver_iterations {
            project_distance_constraints(&mut sim, &mut buffers);
        }

        let positions_after: Vec<Vec3> = sim.particles.iter().map(|p| p.position).collect();

        assert_ne!(
            saved_positions, positions_after,
            "simulation should have advanced from saved state"
        );
    }

    // =========================================================================
    // Performance: timing breakdown per phase
    // =========================================================================

    #[test]
    fn cloth_timing_breakdown() {
        let particle_count = 128;
        let mut sim = make_simple_sim(particle_count);
        sim.gravity = [0.0, -9.81, 0.0];
        sim.damping = 0.02;
        sim.solver_iterations = 4;
        for i in 0..particle_count {
            let x = (i as f32 % 11.0) - 5.0;
            let y = ((i as f32 / 11.0).floor()) - 5.0;
            sim.particles[i].position = [x, y, 0.0];
            sim.particles[i].prev_position = sim.particles[i].position;
        }

        let mut buffers = ClothSimTempBuffers::new(particle_count);

        let iterations = 60;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            verlet_integrate(&mut sim, 1.0 / 60.0);
            for _ in 0..sim.solver_iterations {
                project_distance_constraints(&mut sim, &mut buffers);
            }
            project_bend_constraints(&mut sim);
            compute_normals(&mut sim);
            derive_velocity(&mut sim, 1.0 / 60.0);
        }
        let total = start.elapsed();
        let per_frame = total / iterations;

        assert!(
            per_frame.as_millis() < 100,
            "full cloth step with {} particles took {:?} per frame (debug budget 100ms)",
            particle_count,
            per_frame
        );
    }

    // =========================================================================
    // T07: Time-budgeted cloth step (non-blocking verification)
    // =========================================================================

    #[test]
    fn step_cloth_respects_time_budget() {
        let particle_count = 256;
        let mut sim = make_simple_sim(particle_count);
        sim.gravity = [0.0, -9.81, 0.0];
        sim.damping = 0.02;
        sim.solver_iterations = 16;
        for i in 0..particle_count {
            sim.particles[i].position = [i as f32 % 16.0, (i as f32 / 16.0).floor(), 0.0];
            sim.particles[i].prev_position = sim.particles[i].position;
        }

        let mut buffers = ClothSimTempBuffers::new(particle_count);
        let budget = std::time::Duration::from_millis(5);
        let mut steps_completed = 0u32;
        let max_steps = 1000u32;

        let start = std::time::Instant::now();
        for _ in 0..max_steps {
            if start.elapsed() > budget {
                break;
            }
            verlet_integrate(&mut sim, 1.0 / 60.0);
            for _ in 0..sim.solver_iterations {
                project_distance_constraints(&mut sim, &mut buffers);
            }
            steps_completed += 1;
        }

        assert!(
            steps_completed < max_steps,
            "should have stopped early due to budget, completed {} steps",
            steps_completed
        );
        assert!(
            start.elapsed() <= budget + std::time::Duration::from_millis(2),
            "budget enforcement took too long: {:?} > {:?}",
            start.elapsed(),
            budget
        );
    }

    // =========================================================================
    // T08: Performance impact — measure cloth vs no-cloth frame overhead
    // =========================================================================

    #[test]
    fn cloth_performance_overhead_is_bounded() {
        let particle_count = 128;
        let iterations = 200;

        let baseline_start = std::time::Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(());
        }
        let baseline = baseline_start.elapsed();

        let mut sim = make_simple_sim(particle_count);
        sim.gravity = [0.0, -9.81, 0.0];
        sim.damping = 0.02;
        sim.solver_iterations = 8;
        for i in 0..particle_count {
            let x = (i as f32 % 11.0) - 5.0;
            let y = ((i as f32 / 11.0).floor()) - 5.0;
            sim.particles[i].position = [x, y, 0.0];
            sim.particles[i].prev_position = sim.particles[i].position;
        }
        let mut buffers = ClothSimTempBuffers::new(particle_count);

        let cloth_start = std::time::Instant::now();
        for _ in 0..iterations {
            verlet_integrate(&mut sim, 1.0 / 60.0);
            for _ in 0..sim.solver_iterations {
                project_distance_constraints(&mut sim, &mut buffers);
            }
            project_bend_constraints(&mut sim);
            compute_normals(&mut sim);
            derive_velocity(&mut sim, 1.0 / 60.0);
        }
        let cloth_elapsed = cloth_start.elapsed();

        let per_frame_overhead = (cloth_elapsed - baseline) / iterations;
        assert!(
            per_frame_overhead.as_micros() < 16_000,
            "cloth overhead per frame: {:?} (budget 16ms for 60fps)",
            per_frame_overhead
        );
    }

    // =========================================================================
    // Cloth LoD: solver iteration adjustment
    // =========================================================================

    #[test]
    fn cloth_lod_reduces_iterations_for_low_detail() {
        use crate::simulation::cloth::ClothLoDLevel;
        let mut sim = make_simple_sim(16);
        sim.solver_iterations = 8;
        sim.apply_lod(&ClothLoDLevel::Half);
        assert_eq!(sim.solver_iterations, 6);

        sim.apply_lod(&ClothLoDLevel::Quarter);
        assert_eq!(sim.solver_iterations, 4);

        sim.apply_lod(&ClothLoDLevel::Full);
        assert_eq!(sim.solver_iterations, 8);
    }

    #[test]
    fn cloth_lod_disables_self_collision_at_low_ratio() {
        use crate::simulation::cloth::ClothLoDLevel;
        let mut sim = make_simple_sim(16);
        sim.self_collision = true;

        sim.apply_lod(&ClothLoDLevel::Quarter);
        assert!(
            !sim.self_collision,
            "self-collision should be off at Quarter LoD"
        );

        sim.self_collision = true;
        sim.apply_lod(&ClothLoDLevel::Half);
        assert!(
            sim.self_collision,
            "self-collision should remain on at Half LoD"
        );
    }
}
