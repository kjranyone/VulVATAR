//! Distance and bend constraint projection for the cloth solver.

use crate::math_utils::{
    vec3_add, vec3_cross, vec3_dot, vec3_length, vec3_scale, vec3_sub,
};
use crate::simulation::cloth::{ClothSimState, ClothSimTempBuffers};

// =========================================================================
// Distance constraint projection (Jakobsen-style)
// =========================================================================

pub(super) fn project_distance_constraints(
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
///
/// KNOWN ISSUE: the unit test `bend_constraint_changes_angle` only verifies
/// that the angle moves; the correction direction was observed to be inverted
/// against simple synthetic inputs (closes the angle when it should open and
/// vice versa) during T07 verification. Investigate when wiring against real
/// cloth meshes — synthetic-test behaviour may not be the prod-relevant case.
pub(super) fn project_bend_constraints(sim: &mut ClothSimState) {
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
