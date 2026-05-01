//! Cloth solver orchestration: wires together integrator, constraints,
//! collision, and output phases for both the avatar's primary cloth state
//! and any extra overlay slots.

use crate::avatar::AvatarInstance;
use crate::simulation::cloth::{ClothSimTempBuffers, ResolvedCollider};

mod collision;
mod constraints;
mod integrator;
mod output;

#[cfg(test)]
mod tests;

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
    collision::apply_pin_targets(sim, &avatar.pose.global_transforms);

    // ---- Verlet integration ---------------------------------------------------
    integrator::verlet_integrate(sim, dt);

    // ---- constraint projection (iterative) ------------------------------------
    let iterations = sim.solver_iterations;
    for _ in 0..iterations {
        constraints::project_distance_constraints(sim, buffers);
        constraints::project_bend_constraints(sim);
    }

    // ---- self-collision -------------------------------------------------------
    collision::resolve_self_collisions(sim);

    // ---- collision ------------------------------------------------------------
    let mut all_colliders = colliders;
    all_colliders.extend_from_slice(world_colliders);
    collision::collide(sim, &all_colliders);

    // ---- enforce pins after all projections ------------------------------------
    collision::enforce_pins(sim);

    // ---- compute per-vertex normals -------------------------------------------
    output::compute_normals(sim);

    // ---- derive velocity from position delta ----------------------------------
    integrator::derive_velocity(sim, dt);

    // ---- write back to ClothState for rendering -------------------------------
    output::write_back(sim, cloth_state);
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

        collision::apply_pin_targets(&mut slot.sim, &global_transforms);
        integrator::verlet_integrate(&mut slot.sim, dt);

        let iterations = slot.sim.solver_iterations;
        for _ in 0..iterations {
            constraints::project_distance_constraints(&mut slot.sim, &mut slot.buffers);
            constraints::project_bend_constraints(&mut slot.sim);
        }

        collision::resolve_self_collisions(&mut slot.sim);
        collision::collide(&mut slot.sim, &all_colliders);
        collision::enforce_pins(&mut slot.sim);
        output::compute_normals(&mut slot.sim);
        integrator::derive_velocity(&mut slot.sim, dt);
        output::write_back(&slot.sim, &mut slot.state);
    }
}
