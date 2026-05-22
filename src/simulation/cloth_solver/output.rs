//! Per-vertex normal computation and write-back to the renderable ClothState.
//!
//! Normals use **angle-weighted** face-normal accumulation (Max 1999,
//! "Weights for Computing Vertex Normals from Facet Normals"). Each
//! triangle's contribution to a vertex is its unit face-normal scaled
//! by the incident angle at that vertex, summed across all incident
//! triangles, then normalised once per vertex.
//!
//! Compared with the prior area-weighted accumulation (face-normal scaled
//! by 2 × triangle area):
//!
//! - On regular meshes the two converge to the same answer.
//! - On **irregular** triangulations — long thin triangles next to short
//!   compact ones, T-junctions, fans where one triangle dominates the
//!   1-ring area — angle weighting gives noticeably smoother shading
//!   because each triangle's contribution is bounded by `[0, π]` rather
//!   than by its (potentially unbounded) area. The area-weighted scheme
//!   skews the normal toward whichever triangle happens to be largest.
//!
//! Cost per triangle is one extra `acos` for each of the three vertices
//! plus three edge normalisations — negligible against the constraint
//! solver and well within the cloth path's per-frame budget.

use crate::math_utils::{vec3_add, vec3_cross, vec3_dot, vec3_length, vec3_normalize, vec3_scale, vec3_sub};
use crate::simulation::cloth::ClothSimState;

// =========================================================================
// Per-vertex normal computation (Max 1999, angle-weighted)
// =========================================================================

/// Angle between two unit-or-zero vectors `u`, `v`. Returns 0 if either
/// side is degenerate (effectively dropping the triangle's contribution
/// to this vertex, which is correct — a degenerate edge can't define
/// an angle).
fn safe_angle_between(u: [f32; 3], v: [f32; 3]) -> f32 {
    let lu = vec3_length(&u);
    let lv = vec3_length(&v);
    if lu < 1e-9 || lv < 1e-9 {
        return 0.0;
    }
    let dot = vec3_dot(&u, &v) / (lu * lv);
    // Clamp to `[-1, 1]` so a tiny float drift past the unit-circle
    // doesn't make `acos` return NaN.
    dot.clamp(-1.0, 1.0).acos()
}

pub(crate) fn compute_normals(sim: &mut ClothSimState) {
    let n = sim.particles.len();
    // Lazy-init normals buffer.
    if sim.computed_normals.len() != n {
        sim.computed_normals.resize(n, [0.0; 3]);
    }
    for normal in sim.computed_normals.iter_mut() {
        *normal = [0.0, 0.0, 0.0];
    }

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

        // Face normal — normalised so the angle weights are the sole
        // contribution magnitude. Skip degenerate triangles whose
        // cross product is below the noise floor (collinear points,
        // duplicate vertices, etc.).
        let e01 = vec3_sub(&p1, &p0);
        let e02 = vec3_sub(&p2, &p0);
        let raw_normal = vec3_cross(&e01, &e02);
        let face_area2 = vec3_length(&raw_normal);
        if face_area2 < 1e-12 {
            continue;
        }
        let face_normal = vec3_scale(&raw_normal, 1.0 / face_area2);

        // Three incident angles. By construction
        // `α0 + α1 + α2 = π`, but rounding sometimes drifts; the
        // sum-to-π identity is not load-bearing for shading.
        let e10 = vec3_sub(&p0, &p1);
        let e12 = vec3_sub(&p2, &p1);
        let e20 = vec3_sub(&p0, &p2);
        let e21 = vec3_sub(&p1, &p2);
        let a0 = safe_angle_between(e01, e02);
        let a1 = safe_angle_between(e10, e12);
        let a2 = safe_angle_between(e20, e21);

        sim.computed_normals[i0] =
            vec3_add(&sim.computed_normals[i0], &vec3_scale(&face_normal, a0));
        sim.computed_normals[i1] =
            vec3_add(&sim.computed_normals[i1], &vec3_scale(&face_normal, a1));
        sim.computed_normals[i2] =
            vec3_add(&sim.computed_normals[i2], &vec3_scale(&face_normal, a2));
    }

    // Normalise.
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
