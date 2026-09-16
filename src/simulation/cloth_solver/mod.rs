//! Cloth solver orchestration: wires together integrator, constraints,
//! collision, and output phases for both the avatar's primary cloth state
//! and any extra overlay slots.

use crate::avatar::AvatarInstance;
use crate::simulation::cloth::{ClothSimState, ClothSimTempBuffers, ResolvedCollider};

mod collision;
// `pub(crate)` so the GPU cloth boundary's formula-parity tests can
// call them directly to compare the GLSL shaders' formulas against
// the CPU XPBD references.
pub(crate) mod constraints;
pub(crate) mod integrator;
pub(crate) mod output;

#[cfg(test)]
mod tests;

// =========================================================================
// Public entry point
// =========================================================================

/// Advance cloth simulation by `dt` seconds for the given avatar instance.
///
/// `body_sdf` is the posed body's distance field (as consumed by the
/// spring solver); it only engages for slots with `sdf_contact > 0`.
pub fn step_cloth(
    dt: f32,
    avatar: &mut AvatarInstance,
    world_colliders: &[ResolvedCollider],
    body_sdf: Option<&crate::simulation::sdf::SdfField>,
) {
    if !avatar.cloth_enabled {
        return;
    }

    step_cloth_single(dt, avatar, world_colliders, body_sdf);
    step_cloth_overlays(dt, avatar, world_colliders, body_sdf);
}

fn step_cloth_single(
    dt: f32,
    avatar: &mut AvatarInstance,
    world_colliders: &[ResolvedCollider],
    body_sdf: Option<&crate::simulation::sdf::SdfField>,
) {
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

    // For Gpu-backed cloth the renderer dispatches the integration /
    // constraint projection / normal recomputation compute pipelines;
    // running the CPU XPBD path on top would waste CPU cycles and the
    // resulting deform_output would just be ignored by the renderer.
    if cloth_state.solver_backend == crate::simulation::cloth_gpu_boundary::ClothSolverBackend::Gpu
    {
        return;
    }

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
        &avatar.collider_enabled,
    );
    let mut all_colliders = colliders;
    all_colliders.extend_from_slice(world_colliders);

    // ---- settle-sleep gate (idle z-fight campaign, 2026-09-15) ----------------
    // A quiet sim with bit-identical inputs keeps its solved positions
    // verbatim; skipping the pin write too is required for that (the
    // GPU path's frozen-frame contract behaves the same). The merged
    // collider list is part of the key — a scene collider appearing or
    // moving must wake the sim.
    let sleep_hash =
        cloth_cpu_inputs_hash(dt, sim, &all_colliders, &avatar.pose.global_transforms);
    if !settle_gate_cpu(&mut cloth_state.settle, sleep_hash) {
        return;
    }

    // ---- resolve pin targets --------------------------------------------------
    collision::apply_pin_targets(sim, &avatar.pose.global_transforms);

    // ---- Verlet integration ---------------------------------------------------
    integrator::verlet_integrate(sim, dt);

    // ---- reset XPBD Lagrange multipliers for this substep --------------------
    // The lambda buffer accumulates across `solver_iterations` projection
    // passes inside one substep; the next substep starts fresh.
    buffers.reset_lambda(sim.distance_constraints.len());

    // ---- constraint projection (iterative) ------------------------------------
    let iterations = sim.solver_iterations;
    for _ in 0..iterations {
        constraints::project_distance_constraints(sim, buffers, dt);
        constraints::project_bend_constraints(sim);
    }

    // ---- self-collision -------------------------------------------------------
    collision::resolve_self_collisions(sim);

    // ---- collision ------------------------------------------------------------
    collision::collide(sim, &all_colliders, body_sdf);

    // ---- enforce pins after all projections ------------------------------------
    collision::enforce_pins(sim);

    // ---- compute per-vertex normals -------------------------------------------
    output::compute_normals(sim);

    // ---- derive velocity from position delta ----------------------------------
    integrator::derive_velocity(sim, dt);

    // ---- settle-sleep accounting ----------------------------------------------
    settle_account_cpu(&mut cloth_state.settle, sim);

    // ---- write back to ClothState for rendering -------------------------------
    output::write_back(sim, cloth_state);
}

fn step_cloth_overlays(
    dt: f32,
    avatar: &mut AvatarInstance,
    world_colliders: &[ResolvedCollider],
    body_sdf: Option<&crate::simulation::sdf::SdfField>,
) {
    if avatar.cloth_overlays.is_empty() {
        return;
    }

    let colliders = crate::simulation::cloth::resolve_colliders(
        &avatar.asset.colliders,
        &avatar.pose.global_transforms,
        &avatar.collider_enabled,
    );
    let mut all_colliders = colliders;
    all_colliders.extend_from_slice(world_colliders);

    let global_transforms = avatar.pose.global_transforms.clone();

    for slot in &mut avatar.cloth_overlays {
        if !slot.enabled {
            continue;
        }
        // Skip CPU PBD for Gpu-backed overlays — renderer drives them.
        if slot.state.solver_backend
            == crate::simulation::cloth_gpu_boundary::ClothSolverBackend::Gpu
        {
            continue;
        }

        let particle_count = slot.sim.particle_count();
        if particle_count == 0 || !slot.sim.initialized {
            continue;
        }

        if slot.buffers.temp_positions.len() != particle_count {
            slot.buffers.resize(particle_count);
        }

        // Settle-sleep gate — same contract as `step_cloth_single`.
        let sleep_hash =
            cloth_cpu_inputs_hash(dt, &slot.sim, &all_colliders, &global_transforms);
        if !settle_gate_cpu(&mut slot.state.settle, sleep_hash) {
            continue;
        }

        collision::apply_pin_targets(&mut slot.sim, &global_transforms);
        integrator::verlet_integrate(&mut slot.sim, dt);

        // XPBD lambda reset per substep — see `step_cloth_single` for
        // the lifecycle rationale.
        slot.buffers
            .reset_lambda(slot.sim.distance_constraints.len());

        let iterations = slot.sim.solver_iterations;
        for _ in 0..iterations {
            constraints::project_distance_constraints(&mut slot.sim, &mut slot.buffers, dt);
            constraints::project_bend_constraints(&mut slot.sim);
        }

        collision::resolve_self_collisions(&mut slot.sim);
        collision::collide(&mut slot.sim, &all_colliders, body_sdf);
        collision::enforce_pins(&mut slot.sim);
        output::compute_normals(&mut slot.sim);
        integrator::derive_velocity(&mut slot.sim, dt);
        settle_account_cpu(&mut slot.state.settle, &slot.sim);
        output::write_back(&slot.sim, &mut slot.state);
    }
}

// =========================================================================
// Settle-sleep (idle z-fight campaign, 2026-09-15) — CPU solver side
// =========================================================================

/// Full-precision fingerprint of every input one CPU substep consumes
/// besides particle state. See `simulation/settle.rs` for why the
/// hash must be bit-exact (no quantizing) and `app/render.rs`
/// `cloth_gpu_inputs_hash` for the GPU twin.
fn cloth_cpu_inputs_hash(
    dt: f32,
    sim: &ClothSimState,
    colliders: &[ResolvedCollider],
    global_transforms: &[crate::asset::Mat4],
) -> u64 {
    use crate::math_utils::vec3_scale;
    use crate::simulation::settle::SettleHasher;
    let mut h = SettleHasher::new();
    h.write_f32(dt);
    h.write_f32(sim.damping);
    h.write_f32(sim.sdf_contact);
    h.write_f32s(&sim.gravity);
    h.write_f32s(&vec3_scale(&sim.wind_direction, sim.wind_response));
    h.write_u32(sim.solver_iterations);
    h.write_f32(sim.collision_margin);
    h.write_bool(sim.self_collision);
    h.write_f32(sim.self_collision_radius);
    h.write_u32(sim.pin_targets.len() as u32);
    for pin in &sim.pin_targets {
        h.write_u32(pin.node_index as u32);
        h.write_f32s(&pin.offset);
    }
    h.write_u32(global_transforms.len() as u32);
    for m in global_transforms {
        for col in m {
            h.write_f32s(col);
        }
    }
    h.write_u32(colliders.len() as u32);
    for c in colliders {
        match c {
            ResolvedCollider::Sphere { center, radius } => {
                h.write_u32(0);
                h.write_f32s(center);
                h.write_f32(*radius);
            }
            ResolvedCollider::Capsule {
                center,
                radius,
                half_height,
                axis,
            } => {
                h.write_u32(1);
                h.write_f32s(center);
                h.write_f32(*radius);
                h.write_f32(*half_height);
                h.write_f32s(axis);
            }
        }
    }
    h.finish()
}

/// Gate one CPU substep: `false` = the sim is asleep on unchanged
/// inputs, skip the whole substep (positions stay bit-stable). Any
/// input change wakes the state and re-arms the fingerprint.
fn settle_gate_cpu(settle: &mut crate::simulation::settle::ClothSettleSleep, hash: u64) -> bool {
    if settle.last_inputs != Some(hash) {
        settle.sleeping = false;
        settle.quiet_frames = 0;
        settle.last_inputs = Some(hash);
        true
    } else {
        !settle.sleeping
    }
}

/// Quiet accounting after one CPU substep. `prev_position` holds the
/// pre-integration position at this point, so
/// `|position - prev_position|` is the substep's total motion
/// (integration + constraint/collision corrections).
fn settle_account_cpu(
    settle: &mut crate::simulation::settle::ClothSettleSleep,
    sim: &ClothSimState,
) {
    use crate::math_utils::{vec3_length, vec3_sub};
    let max_move = sim
        .particles
        .iter()
        .map(|p| vec3_length(&vec3_sub(&p.position, &p.prev_position)))
        .fold(0.0f32, f32::max);
    let (quiet_frames, sleeping) =
        crate::simulation::settle::settle_bump(settle.quiet_frames, max_move);
    settle.quiet_frames = quiet_frames;
    settle.sleeping = sleeping;
    settle.last_max_delta = max_move;
}
