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

use crate::math_utils::{vec3_add, vec3_cross, vec3_dot, vec3_length, vec3_scale, vec3_sub};
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
        // 1e-9 m = 1 nm. Below this both endpoints are numerically
        // coincident and the constraint direction is undefined. Same
        // threshold used by the GPU mirror at the lambda-update +
        // accumulate passes.
        if dist < 1e-9 {
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

        buffers.correction_accumulator[a] = vec3_add(&buffers.correction_accumulator[a], &corr_a);
        buffers.correction_accumulator[b] = vec3_add(&buffers.correction_accumulator[b], &corr_b);
        buffers.correction_count[a] += 1;
        buffers.correction_count[b] += 1;
    }

    // Under-relaxed Jacobi application: Δx_i = Σ corr / (n_i + 1).
    // The plain sum (the textbook Jacobi form, and what the GLSL
    // accumulate pass did until now) diverges once per-particle
    // constraint degree grows past a couple of edges — measured on
    // Yumeka's welded skirt (degree ≈ 6): first-step positions at
    // ±1e8. Relaxing by 1/(n+1) is the standard stable Jacobi cloth
    // factor; λ stays per-constraint and un-averaged, only the
    // position step is relaxed.
    for i in 0..sim.particles.len() {
        if sim.particles[i].pinned {
            continue;
        }
        let corr = buffers.correction_accumulator[i];
        if corr[0] == 0.0 && corr[1] == 0.0 && corr[2] == 0.0 {
            continue;
        }
        let relax = 1.0 / (buffers.correction_count[i] as f32 + 1.0);
        let relaxed = vec3_scale(&corr, relax);
        sim.particles[i].position = vec3_add(&sim.particles[i].position, &relaxed);
    }
}

// =========================================================================
// Bend constraint projection (M1)
// =========================================================================

/// Project bend constraints — three-point **edge-angle hinge**.
///
/// MODEL CONTRACT (T09): the constrained quantity is the angle at the
/// hinge vertex `p0` between the edges to the wing vertices `p1` and
/// `p2`, measured in the plane the two edges span. This is NOT the
/// two-triangle dihedral angle along a shared edge; generation code
/// must not feed shared-edge triangle pairs here expecting dihedral
/// semantics. Auto-cloth generates no bend constraints; authored
/// `.vvtcloth` files get this edge-angle model.
///
/// CORRECTION MODEL (linearised, both directions): moving wing `p1` by
/// `δ` along the in-plane unit perpendicular toward `p2` changes the
/// angle at the exact first-order rate `−δ/|p0p1|` (symmetric for
/// `p2`), so a free wing's step is
/// `delta = perp · (w / w_free) · stiffness · err · |edge|` with `w`
/// the wing's inv-mass share among FREE wings only. Both signs of
/// `err` strictly reduce `|err|` in the linear regime; repeated
/// projection converges to `rest_angle` (unit-tested: monotone error
/// decay, both directions).
///
/// Fixed points and degeneracies:
/// - The hinge vertex `p0` is NEVER moved by this constraint. Its
///   second-order positional drift is repaired by the distance
///   constraints on the following passes. (The pre-T09 hinge
///   compensation term fought the wing correction and, with both wings
///   pinned, drove the free hinge away without bound — measured by
///   `bend_constraint_pinned_wings_is_a_fixed_point` against the old
///   implementation.)
/// - Pinned wings never move and carry no share; two pinned wings make
///   the constraint inert.
/// - Exactly-collinear edges (sin ≈ 0) or a zero-length edge have no
///   correction direction — the constraint stays inert and NaN-free.
///
/// The GPU cloth stages currently implement distance constraints only.
/// When a bend stage is ported it must mirror THIS model — the
/// pre-T09 implementation had the correction direction inverted and
/// must not be used as the reference.
pub(super) fn project_bend_constraints(sim: &mut ClothSimState) {
    let n = sim.particles.len();

    for ci in 0..sim.bend_constraints.len() {
        let p0 = sim.bend_constraints[ci].p0;
        let p1 = sim.bend_constraints[ci].p1;
        let p2 = sim.bend_constraints[ci].p2;
        let rest_angle = sim.bend_constraints[ci].rest_angle;
        let stiffness = sim.bend_constraints[ci].stiffness.clamp(0.0, 1.0);
        if stiffness <= 0.0 {
            continue;
        }

        if p0 >= n || p1 >= n || p2 >= n {
            continue;
        }

        let x0 = sim.particles[p0].position;
        let x1 = sim.particles[p1].position;
        let x2 = sim.particles[p2].position;

        let e1 = vec3_sub(&x1, &x0);
        let e2 = vec3_sub(&x2, &x0);

        let e1_len = vec3_length(&e1);
        let e2_len = vec3_length(&e2);
        if e1_len < 1e-12 || e2_len < 1e-12 {
            continue;
        }

        let dot = vec3_dot(&e1, &e2);
        let cross_len = vec3_length(&vec3_cross(&e1, &e2));
        // |perp| equals sin(angle) for unit edges — the same quantity the
        // perpendicular directions below normalise by. Near-collinear
        // edges leave no stable correction direction; stay inert.
        if cross_len < 1e-6 * e1_len * e2_len {
            continue;
        }
        let current_angle = cross_len.atan2(dot);

        let err = current_angle - rest_angle;
        if err.abs() < 1e-6 {
            continue;
        }

        // Unit in-plane perpendiculars: p1's points toward p2's side,
        // p2's toward p1's side. Moving p1 along +perp1 lowers the
        // angle at rate 1/|e1| (and symmetrically for p2).
        let e1_norm = vec3_scale(&e1, 1.0 / e1_len);
        let e2_norm = vec3_scale(&e2, 1.0 / e2_len);
        let perp1 = vec3_sub(&e2_norm, &vec3_scale(&e1_norm, dot / (e1_len * e2_len)));
        let perp1_len = vec3_length(&perp1);
        if perp1_len < 1e-6 {
            continue;
        }
        let perp1 = vec3_scale(&perp1, 1.0 / perp1_len);
        let perp2 = vec3_sub(&e1_norm, &vec3_scale(&e2_norm, dot / (e1_len * e2_len)));
        let perp2_len = vec3_length(&perp2);
        if perp2_len < 1e-6 {
            continue;
        }
        let perp2 = vec3_scale(&perp2, 1.0 / perp2_len);

        // Inv-mass shares among FREE wings only. Pinned wings move
        // nothing and dilute nothing; the hinge is never moved.
        let w1 = if sim.particles[p1].pinned {
            0.0
        } else {
            sim.particles[p1].inv_mass
        };
        let w2 = if sim.particles[p2].pinned {
            0.0
        } else {
            sim.particles[p2].inv_mass
        };
        let w_sum = w1 + w2;
        if w_sum < 1e-12 {
            continue;
        }

        // Linearised step: Δθ = −(w/w_sum) · stiffness · err per wing,
        // so the free wings jointly remove `stiffness · err` each pass.
        let scale = stiffness * err;
        if w1 > 0.0 {
            let delta1 = vec3_scale(&perp1, (w1 / w_sum) * scale * e1_len);
            sim.particles[p1].position = vec3_add(&sim.particles[p1].position, &delta1);
        }
        if w2 > 0.0 {
            let delta2 = vec3_scale(&perp2, (w2 / w_sum) * scale * e2_len);
            sim.particles[p2].position = vec3_add(&sim.particles[p2].position, &delta2);
        }
    }
}
