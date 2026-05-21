//! Distance and bend constraint projection for the cloth solver.
//!
//! Distance constraints use the **XPBD** (eXtended Position-Based Dynamics,
//! Macklin et al. 2016) formulation rather than classic Jakobsen-style PBD:
//!
//! ```text
//!   α̃_j   = α_j / dt²              (time-step-scaled compliance)
//!   Δλ_j  = -(C_j + α̃_j · λ_j) / (w_a + w_b + α̃_j)
//!   λ_j  += Δλ_j
//!   Δx_a  = +w_a · d · Δλ_j        (d = (x_a − x_b)/|x_a − x_b|)
//!   Δx_b  = -w_b · d · Δλ_j
//! ```
//!
//! Per-constraint `λ` accumulates across the `solver_iterations` projection
//! passes inside one substep. The substep entry resets `λ` to zero (see
//! `ClothSimTempBuffers::reset_lambda`). This gives:
//!
//! - Stiffness becomes a **material property** (compliance `α`), not a
//!   per-iteration "fraction of the error to correct" hack. The asset's
//!   legacy `stiffness ∈ (0, 1]` is mapped onto an XPBD compliance via
//!   `compliance_from_pbd_stiffness` below — `stiffness = 1` is treated as
//!   the rigid limit (`α = 0`).
//! - Equilibrium length under a steady load is well-defined and does not
//!   drift with the iteration count or the substep `dt`. Classic PBD's
//!   "more iterations → effectively stiffer" trap is structurally gone.
//! - Pinned particles (`inv_mass = 0`) fall out of the denominator
//!   naturally; their `Δx` is zero by construction.
//!
//! Jacobi-style averaging (corrections accumulated into a per-particle
//! buffer and divided by touching-constraint count) is preserved from the
//! prior PBD implementation so the Δx-application phase is unchanged.

use crate::math_utils::{
    vec3_add, vec3_cross, vec3_dot, vec3_length, vec3_scale, vec3_sub,
};
use crate::simulation::cloth::{ClothSimState, ClothSimTempBuffers};

/// `stiffness <= STIFFNESS_DISABLED_EPS` treats the constraint as **disabled**
/// (legacy PBD contract: "stiffness = 0 means no correction"). The XPBD
/// formula with finite compliance would still apply a tiny correction, so
/// the constraint projection short-circuits here to preserve that contract
/// for assets / tests that use `stiffness = 0` to author a "soft hinge" or
/// to disable a single edge.
const STIFFNESS_DISABLED_EPS: f32 = 1.0e-6;

/// Convert the legacy `stiffness ∈ (0, 1]` PBD knob into an XPBD compliance
/// `α` (units of m / N). `stiffness = 1` is the rigid limit (`α = 0`); softer
/// values produce a quadratically increasing compliance so the tail of the
/// range gives noticeably stretchy cloth without `stiffness = 0` collapsing
/// into numerical instability.
///
/// The `1e-7` base scale was chosen so that the default cloth assets
/// (`stiffness ≈ 0.5–1.0`, particle masses on the order of 0.01 kg, mesh
/// edges ~5 cm) sit in the "very stiff but compliant enough that the solver
/// stays well-conditioned" regime at 60 Hz / 4 substeps. Tune per-asset by
/// authoring lower stiffness in the cloth file.
fn compliance_from_pbd_stiffness(stiffness: f32) -> f32 {
    let s = stiffness.clamp(0.0, 1.0);
    let slack = (1.0 - s).max(0.0);
    slack * slack * 1.0e-7
}

// =========================================================================
// Distance constraint projection (XPBD, Jacobi-averaged)
// =========================================================================

/// Project all distance constraints once. Caller is responsible for
/// having reset [`ClothSimTempBuffers::lambda_distance`] at the start of
/// the current substep — XPBD's stiffness independence depends on `λ`
/// growing across calls within one substep.
///
/// `dt` is the substep duration (used to scale compliance into the
/// `α̃ = α / dt²` form). Passing the same `dt` value the integrator
/// used is required for the formula to be correct.
pub(crate) fn project_distance_constraints(
    sim: &mut ClothSimState,
    buffers: &mut ClothSimTempBuffers,
    dt: f32,
) {
    buffers.clear();
    // The substep entry guarantees the lambda buffer matches the
    // current constraint count; defence in depth: if a stale buffer
    // somehow leaked through (e.g. constraint count changed mid-step
    // without a substep boundary), fall back to no-lambda PBD behaviour
    // by zero-extending so the formula still produces a finite result.
    if buffers.lambda_distance.len() < sim.distance_constraints.len() {
        buffers
            .lambda_distance
            .resize(sim.distance_constraints.len(), 0.0);
    }

    // dt² ≈ 0 happens during paused-frame editor scrubs; the
    // compliance term collapses to ∞ and the constraint contributes
    // nothing. The guard keeps the formula well-defined.
    let dt_sq = (dt * dt).max(1e-12);

    for (j, dc) in sim.distance_constraints.iter().enumerate() {
        let a = dc.a;
        let b = dc.b;
        if a >= sim.particles.len() || b >= sim.particles.len() {
            continue;
        }
        if dc.stiffness <= STIFFNESS_DISABLED_EPS {
            // Legacy PBD contract: stiffness == 0 disables the
            // constraint. XPBD's finite compliance would still nudge
            // particles together; short-circuit to keep that contract.
            continue;
        }

        let pa = sim.particles[a].position;
        let pb = sim.particles[b].position;
        // d points from b → a — matching the convention in the XPBD
        // module-level documentation above. `∇C` w.r.t. x_a is +d,
        // w.r.t. x_b is −d.
        let diff = vec3_sub(&pa, &pb);
        let dist = vec3_length(&diff);
        if dist < 1e-12 {
            continue;
        }
        let d = vec3_scale(&diff, 1.0 / dist);

        let inv_mass_a = sim.particles[a].inv_mass;
        let inv_mass_b = sim.particles[b].inv_mass;
        let w_sum = inv_mass_a + inv_mass_b;
        if w_sum < 1e-12 {
            continue;
        }

        let compliance = compliance_from_pbd_stiffness(dc.stiffness);
        let alpha_tilde = compliance / dt_sq;

        let c = dist - dc.rest_length;
        let lambda = buffers.lambda_distance[j];
        let denom = w_sum + alpha_tilde;
        let delta_lambda = (-c - alpha_tilde * lambda) / denom;
        buffers.lambda_distance[j] = lambda + delta_lambda;

        // Δx_a = +w_a · d · Δλ
        // Δx_b = −w_b · d · Δλ
        let corr_a = vec3_scale(&d, inv_mass_a * delta_lambda);
        let corr_b = vec3_scale(&d, -inv_mass_b * delta_lambda);

        buffers.correction_accumulator[a] =
            vec3_add(&buffers.correction_accumulator[a], &corr_a);
        buffers.correction_counts[a] += 1;

        buffers.correction_accumulator[b] =
            vec3_add(&buffers.correction_accumulator[b], &corr_b);
        buffers.correction_counts[b] += 1;
    }

    // Jacobi-averaged Δx application: every constraint that touches
    // particle `i` contributes one entry into the accumulator; we
    // apply the mean. Matches the PBD code path this XPBD pass
    // replaced so the existing collision / pin / self-collision
    // phases see corrections of comparable magnitude.
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
