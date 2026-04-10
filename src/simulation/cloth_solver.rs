use crate::avatar::AvatarInstance;
use crate::math_utils::{
    closest_point_on_segment, vec3_add, vec3_cross, vec3_dot, vec3_length, vec3_normalize,
    vec3_scale, vec3_sub, Vec3,
};
use crate::simulation::cloth::{ClothSimState, ClothSimTempBuffers, ResolvedCollider};

// Convenience aliases: the cloth solver originally used by-value signatures
// (v3_add(a, b)).  The shared math_utils takes references.  These thin wrappers
// keep the diff small and avoid touching every single call site.
#[inline]
fn v3_add(a: Vec3, b: Vec3) -> Vec3 { vec3_add(&a, &b) }
#[inline]
fn v3_sub(a: Vec3, b: Vec3) -> Vec3 { vec3_sub(&a, &b) }
#[inline]
fn v3_scale(a: Vec3, s: f32) -> Vec3 { vec3_scale(&a, s) }
#[inline]
fn v3_dot(a: Vec3, b: Vec3) -> f32 { vec3_dot(&a, &b) }
#[inline]
fn v3_cross(a: Vec3, b: Vec3) -> Vec3 { vec3_cross(&a, &b) }
#[inline]
fn v3_length(a: Vec3) -> f32 { vec3_length(&a) }
#[inline]
fn v3_normalize(a: Vec3) -> Vec3 { vec3_normalize(&a) }
#[inline]
fn v3_closest_point_on_segment(a: Vec3, b: Vec3, p: Vec3) -> Vec3 {
    closest_point_on_segment(&a, &b, &p)
}

// =========================================================================
// Public entry point
// =========================================================================

/// Advance cloth simulation by `dt` seconds for the given avatar instance.
pub fn step_cloth(dt: f32, avatar: &mut AvatarInstance) {
    if !avatar.cloth_enabled {
        return;
    }

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
    let buffers = avatar.cloth_sim_buffers.get_or_insert_with(|| {
        ClothSimTempBuffers::new(particle_count)
    });
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
    collide(sim, &colliders);

    // ---- enforce pins after all projections ------------------------------------
    enforce_pins(sim);

    // ---- compute per-vertex normals -------------------------------------------
    compute_normals(sim);

    // ---- derive velocity from position delta ----------------------------------
    derive_velocity(sim, dt);

    // ---- write back to ClothState for rendering -------------------------------
    write_back(sim, cloth_state);
}

// =========================================================================
// Verlet integration (E7: wind_response applied here)
// =========================================================================

fn verlet_integrate(sim: &mut ClothSimState, dt: f32) {
    let gravity = sim.gravity;
    let damping = sim.damping.clamp(0.0, 1.0);
    let wind_force = v3_scale(sim.wind_direction, sim.wind_response);

    for p in sim.particles.iter_mut() {
        if p.pinned {
            continue;
        }

        let vel = v3_sub(p.position, p.prev_position);
        let damped_vel = v3_scale(vel, 1.0 - damping);
        // Gravity + wind as acceleration scaled by dt^2
        let accel = v3_scale(v3_add(gravity, wind_force), dt * dt);

        let new_pos = v3_add(v3_add(p.position, damped_vel), accel);

        p.prev_position = p.position;
        p.position = new_pos;
    }
}

// =========================================================================
// Distance constraint projection (Jakobsen-style)
// =========================================================================

fn project_distance_constraints(
    sim: &mut ClothSimState,
    buffers: &mut ClothSimTempBuffers,
) {
    buffers.clear();

    for dc in &sim.distance_constraints {
        let a = dc.a;
        let b = dc.b;
        if a >= sim.particles.len() || b >= sim.particles.len() {
            continue;
        }

        let pa = sim.particles[a].position;
        let pb = sim.particles[b].position;
        let diff = v3_sub(pb, pa);
        let dist = v3_length(diff);
        if dist < 1e-12 {
            continue;
        }

        let err = dist - dc.rest_length;
        let dir = v3_scale(diff, 1.0 / dist);

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
        let corr_a = v3_scale(dir, correction * wa);
        let corr_b = v3_scale(dir, -correction * wb);

        buffers.correction_accumulator[a] = v3_add(buffers.correction_accumulator[a], corr_a);
        buffers.correction_counts[a] += 1;

        buffers.correction_accumulator[b] = v3_add(buffers.correction_accumulator[b], corr_b);
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
        let avg = v3_scale(buffers.correction_accumulator[i], 1.0 / count as f32);
        sim.particles[i].position = v3_add(sim.particles[i].position, avg);
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
        let e1 = v3_sub(x1, x0);
        let e2 = v3_sub(x2, x0);

        // Current angle between e1 and e2
        let dot = v3_dot(e1, e2);
        let cross = v3_cross(e1, e2);
        let cross_len = v3_length(cross);
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
        let e1_len = v3_length(e1);
        let e2_len = v3_length(e2);
        if e1_len < 1e-12 || e2_len < 1e-12 {
            continue;
        }

        // Perpendicular component of e2 w.r.t. e1 (and vice versa) gives the
        // tangent direction for angle change.
        let e1_norm = v3_scale(e1, 1.0 / e1_len);
        let e2_norm = v3_scale(e2, 1.0 / e2_len);

        // Direction to move p1: perpendicular to e1, toward e2
        let perp1 = v3_sub(e2_norm, v3_scale(e1_norm, v3_dot(e2_norm, e1_norm)));
        let perp1 = v3_normalize(perp1);

        // Direction to move p2: perpendicular to e2, toward e1
        let perp2 = v3_sub(e1_norm, v3_scale(e2_norm, v3_dot(e1_norm, e2_norm)));
        let perp2 = v3_normalize(perp2);

        // Check that perpendiculars are valid
        if v3_length(perp1) < 1e-6 || v3_length(perp2) < 1e-6 {
            continue;
        }

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
            let delta1 = v3_scale(perp1, -scale * w1 * e1_len);
            sim.particles[p1].position = v3_add(sim.particles[p1].position, delta1);
        }
        if !sim.particles[p2].pinned {
            let w2 = inv_mass_2 / total_inv_mass;
            let delta2 = v3_scale(perp2, -scale * w2 * e2_len);
            sim.particles[p2].position = v3_add(sim.particles[p2].position, delta2);
        }
        // Optionally move hinge vertex to conserve momentum
        if !sim.particles[p0].pinned {
            let w0 = inv_mass_0 / total_inv_mass;
            let delta1_comp = v3_scale(perp1, scale * w0 * e1_len * 0.5);
            let delta2_comp = v3_scale(perp2, scale * w0 * e2_len * 0.5);
            sim.particles[p0].position =
                v3_add(sim.particles[p0].position, v3_add(delta1_comp, delta2_comp));
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
                    let diff = v3_sub(p.position, *center);
                    let dist = v3_length(diff);
                    if dist < effective_radius && dist > 1e-12 {
                        let normal = v3_scale(diff, 1.0 / dist);
                        p.position = v3_add(*center, v3_scale(normal, effective_radius));
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
                    let seg_a = v3_sub(*center, v3_scale(*axis, *half_height));
                    let seg_b = v3_add(*center, v3_scale(*axis, *half_height));
                    // Closest point on segment to particle
                    let closest = v3_closest_point_on_segment(seg_a, seg_b, p.position);
                    let diff = v3_sub(p.position, closest);
                    let dist = v3_length(diff);
                    if dist < effective_radius && dist > 1e-12 {
                        let normal = v3_scale(diff, 1.0 / dist);
                        p.position = v3_add(closest, v3_scale(normal, effective_radius));
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
            let diff = v3_sub(pos_j, pos_i);
            let dist_sq = v3_dot(diff, diff);

            if dist_sq < min_dist_sq && dist_sq > 1e-24 {
                let dist = dist_sq.sqrt();
                let overlap = min_dist - dist;
                let dir = v3_scale(diff, 1.0 / dist);

                // Push each particle half the overlap distance apart.
                let half_push = v3_scale(dir, overlap * 0.5);

                corrections[i] = v3_sub(corrections[i], half_push);
                correction_counts[i] += 1;

                corrections[j] = v3_add(corrections[j], half_push);
                correction_counts[j] += 1;
            }
        }
    }

    // Apply averaged corrections.
    for i in 0..particle_count {
        if correction_counts[i] > 0 && !sim.particles[i].pinned {
            let avg = v3_scale(corrections[i], 1.0 / correction_counts[i] as f32);
            sim.particles[i].position = v3_add(sim.particles[i].position, avg);
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

        let e1 = v3_sub(p1, p0);
        let e2 = v3_sub(p2, p0);
        let face_normal = v3_cross(e1, e2); // not normalized; magnitude = 2*area (area-weighted)

        sim.computed_normals[i0] = v3_add(sim.computed_normals[i0], face_normal);
        sim.computed_normals[i1] = v3_add(sim.computed_normals[i1], face_normal);
        sim.computed_normals[i2] = v3_add(sim.computed_normals[i2], face_normal);
    }

    // Normalize
    for normal in sim.computed_normals.iter_mut() {
        *normal = v3_normalize(*normal);
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
        p.velocity = v3_scale(v3_sub(p.position, p.prev_position), inv_dt);
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
