//! Per-vertex normal computation and write-back to the renderable ClothState.

use crate::math_utils::{vec3_add, vec3_cross, vec3_normalize, vec3_sub};
use crate::simulation::cloth::ClothSimState;

// =========================================================================
// Per-vertex normal computation (M2)
// =========================================================================

pub(super) fn compute_normals(sim: &mut ClothSimState) {
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
// Write-back to ClothState for rendering (M2: normals written here)
// =========================================================================

pub(super) fn write_back(sim: &ClothSimState, cloth_state: &mut crate::avatar::ClothState) {
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
    let copy_len = n.min(sim.computed_normals.len());
    normals_vec[..copy_len].copy_from_slice(&sim.computed_normals[..copy_len]);

    deform.version += 1;

    // Update runtime caches
    cloth_state.constraint_cache.active_constraint_count = 0; // filled by solver as needed
    cloth_state.collision_cache.active_collision_count = 0;
}
