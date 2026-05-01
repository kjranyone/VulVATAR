//! Verlet integration and velocity derivation for the cloth solver.

use crate::math_utils::{vec3_add, vec3_scale, vec3_sub};
use crate::simulation::cloth::ClothSimState;

// =========================================================================
// Verlet integration (E7: wind_response applied here)
// =========================================================================

pub(super) fn verlet_integrate(sim: &mut ClothSimState, dt: f32) {
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
// Velocity derivation
// =========================================================================

pub(super) fn derive_velocity(sim: &mut ClothSimState, dt: f32) {
    if dt < 1e-12 {
        return;
    }
    let inv_dt = 1.0 / dt;
    for p in sim.particles.iter_mut() {
        p.velocity = vec3_scale(&vec3_sub(&p.position, &p.prev_position), inv_dt);
    }
}
