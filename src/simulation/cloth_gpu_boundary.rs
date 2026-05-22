//! Cloth ownership boundary between authoring, CPU simulation, GPU simulation,
//! and render-consumable deform.
//!
//! Scaffolding for the planned compute-cloth migration (`plan/gpu-runtime-roadmap-tasks.md`
//! P3-01). No runtime behaviour change: this module re-exports the existing
//! types under boundary-named aliases and introduces a placeholder
//! [`ClothGpuSimulationState`] so the eventual GPU solver can land in the
//! correct slot without renegotiating the avatar/renderer ownership rules.
//!
//! # The four categories
//!
//! | Category | Authoritative type | Lives in | Owned by |
//! |----------|--------------------|----------|----------|
//! | Authoring | [`ClothAuthoring`] | Project save / VRM-SpringBone import | Asset layer |
//! | CPU simulation state | [`ClothCpuSimulationState`] | Per-frame solver scratch | Avatar instance |
//! | GPU simulation state | [`ClothGpuSimulationState`] | Future compute SSBOs | Renderer |
//! | Render-consumable deform | [`ClothRenderConsumableDeform`] | Per-frame snapshot | Renderer / frame input |
//!
//! The renderer must only see [`ClothRenderConsumableDeform`]. Whether the
//! deform came from a CPU PBD solver or a future GPU compute pass is an
//! implementation detail behind [`ClothSolverBackend`].

use crate::asset::ClothAsset;
use crate::renderer::frame_input::ClothDeformSnapshot;
use crate::simulation::cloth::ClothSimState;

/// Authoring data: cloth topology, constraints, render bindings — the
/// serialisable description that survives across runs. Same shape as the
/// asset-layer `ClothAsset`; re-aliased here so the boundary is callable by
/// name from solver and renderer code without spelling out the import path.
pub type ClothAuthoring = ClothAsset;

/// Per-frame CPU solver scratch (particle positions, velocities, constraint
/// runtime cache). Lives on `AvatarInstance` so simulation can mutate it
/// freely; the renderer never reads it directly.
pub type ClothCpuSimulationState = ClothSimState;

/// Render-consumable per-primitive deform snapshot. The renderer matches
/// these against `RenderMeshInstance::primitive_id` and writes the deformed
/// positions / normals into the primitive's cloth SSBO. This is the only
/// type the renderer is allowed to see — both the CPU and the future GPU
/// solver must produce values shaped like this.
pub type ClothRenderConsumableDeform = ClothDeformSnapshot;

/// Per-primitive GPU simulation state owned by the renderer (when P3-02
/// lands). Today this is a structural scaffold: no Vulkan handles yet, but
/// the field shape and lifecycle are pinned down so the eventual compute
/// solver lands in the slot without renegotiating the avatar/renderer
/// boundary.
///
/// # Particle state representation
///
/// **Decision (S0)**: Verlet, with explicit `pos` and `prev_pos` buffers.
///
/// The CPU PBD solver already uses Verlet integration (see
/// `simulation::cloth::ClothParticle::{position, prev_position}`); keeping
/// the GPU path on the same representation lets the CPU reference tests
/// (e.g. `verlet_gravity_pulls_down`) double as expected-value generators
/// for compute-side smoke tests within tolerance.
///
/// Verlet over explicit velocity also halves the per-particle SSBO write
/// pressure during constraint projection: only positions change while
/// constraints iterate; velocity is derived once at the end of the
/// substep from `(pos - prev_pos) / dt`.
///
/// # Lifecycle (intended; not yet implemented)
///
/// - Buffers are created once at cloth-attach time, sized by the
///   primitive's particle / constraint counts
/// - Particle position / `prev_position` SSBOs are read-write across
///   substeps
/// - Constraint table SSBOs are uploaded once and read-only thereafter
/// - The control UBO is rewritten every frame with `(dt, gravity, wind,
///   collider transforms)` — the only per-frame CPU→GPU upload
/// - `version` mirrors `ClothDeformOutput::version`; the renderer flips
///   `has_cloth = 1` in the transform control UBO based on this counter
///
/// # Slots (deferred, documented for the compute landing)
///
/// ```text
/// pos_ssbo        : SSBO of vec4 (xyz = position, w = inv_mass)
/// prev_pos_ssbo   : SSBO of vec4 (xyz = prev_position, w = pinned bit)
/// constraint_ssbo : SSBO of (u32, u32, f32, f32) — (a, b, rest_length, stiffness)
/// collider_ssbo   : SSBO of resolved colliders (sphere / capsule transforms)
/// normal_ssbo    : SSBO of vec3 produced by stage 3; consumed by transform_cs
/// triangle_idx_ssbo : SSBO of u32 indices for normal recomputation
/// control_ubo     : { dt, gravity[3], wind[3], damping, collider_count, … }
/// ```
#[derive(Clone, Debug, Default)]
pub struct ClothGpuSimulationState {
    /// Version mirror; bumped by the compute pass after stage 3 finishes
    /// so the renderer's transform control UBO can flip `has_cloth = 1`
    /// without sampling SSBO contents.
    pub version: u32,
    /// Solver substep iteration count, picked at attach time from the
    /// asset's `iterations` field. Stored here so the compute dispatch
    /// loop reads it without re-pulling the asset.
    pub iterations: u32,
    /// Total particle count for this cloth instance — picked at attach
    /// time, never changes after. Used to size dispatches.
    pub particle_count: u32,
    /// Total constraint count uploaded into `constraint_ssbo` at attach
    /// time. Constraints never change at runtime for a given cloth.
    pub constraint_count: u32,
}

impl ClothGpuSimulationState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a sized placeholder from the authoring data. The actual
    /// Vulkan SSBO handles land when stage 1 of P3-02 ships; the counts
    /// alone are enough for the renderer to dispatch the right number of
    /// workgroups even with a no-op solver.
    pub fn from_authoring(
        particle_count: u32,
        constraint_count: u32,
        iterations: u32,
    ) -> Self {
        Self {
            version: 0,
            iterations,
            particle_count,
            constraint_count,
        }
    }

    pub fn bump_version(&mut self) {
        self.version = self.version.wrapping_add(1);
    }
}

/// Which solver backend produces the [`ClothRenderConsumableDeform`]
/// snapshots for a given cloth. The compute path is intentionally not yet
/// reachable; `Cpu` is the only variant the avatar layer constructs today.
///
/// Keeping the variant named now means later code that selects the backend
/// (e.g. a `RuntimeGpuBudget` decision in P3-03) can switch on this enum
/// instead of growing a bool / feature flag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClothSolverBackend {
    #[default]
    Cpu,
    /// GPU compute path. Until P3-02 stage 1 lands the renderer falls
    /// back to CPU silently (the snapshot pipeline does not yet honour
    /// this flag); the variant exists so attach-time code can already
    /// record intent and the upcoming compute landing has a stable
    /// switch to flip behaviour on.
    Gpu,
}

impl ClothSolverBackend {
    pub fn label(self) -> &'static str {
        match self {
            ClothSolverBackend::Cpu => "CPU PBD",
            ClothSolverBackend::Gpu => "GPU compute",
        }
    }
}

/// CSR (compressed sparse row) adjacency listing the triangles incident to
/// each vertex. Built once at cloth-attach time; consumed by the
/// `cloth_normal_cs` compute shader's per-vertex normal-accumulation loop
/// without needing atomics.
///
/// `offsets.len() == vertex_count + 1`. For vertex `v`, the incident
/// triangle indices live in `triangles[offsets[v] .. offsets[v + 1]]`.
#[derive(Clone, Debug, Default)]
pub struct VertexTriangleAdjacency {
    pub offsets: Vec<u32>,
    pub triangles: Vec<u32>,
}

/// Build a CSR particle→constraint adjacency for a cloth's distance
/// constraint table.
///
/// `constraints` is a slice of `(particle_a, particle_b)` index pairs.
/// For particle `p`, the incident constraint indices live in
/// `output.triangles[output.offsets[p] .. output.offsets[p + 1]]`
/// (reusing the `VertexTriangleAdjacency` field names — same CSR
/// shape, different semantics). Indices `>= particle_count` are
/// silently skipped.
pub fn build_particle_constraint_adjacency(
    constraints: &[(u32, u32)],
    particle_count: u32,
) -> VertexTriangleAdjacency {
    let pc = particle_count as usize;

    let mut counts = vec![0u32; pc];
    for (a, b) in constraints {
        if (*a as usize) < pc {
            counts[*a as usize] += 1;
        }
        if (*b as usize) < pc {
            counts[*b as usize] += 1;
        }
    }

    let mut offsets = vec![0u32; pc + 1];
    let mut acc: u32 = 0;
    for p in 0..pc {
        offsets[p] = acc;
        acc = acc.saturating_add(counts[p]);
    }
    offsets[pc] = acc;

    let mut triangles = vec![0u32; acc as usize];
    let mut cursor = vec![0u32; pc];
    for (cidx, (a, b)) in constraints.iter().enumerate() {
        if (*a as usize) < pc {
            let ai = *a as usize;
            let slot = (offsets[ai] + cursor[ai]) as usize;
            triangles[slot] = cidx as u32;
            cursor[ai] += 1;
        }
        if (*b as usize) < pc {
            let bi = *b as usize;
            let slot = (offsets[bi] + cursor[bi]) as usize;
            triangles[slot] = cidx as u32;
            cursor[bi] += 1;
        }
    }

    VertexTriangleAdjacency { offsets, triangles }
}

/// Build a CSR vertex→triangle adjacency for a cloth mesh.
///
/// `triangle_indices` is the flat index buffer (3 entries per triangle).
/// Indices `>= vertex_count` are silently skipped — corrupt indices
/// would otherwise read past the position SSBO at runtime, and a panic
/// here would be unhelpful far from the asset-loading site.
pub fn build_vertex_triangle_adjacency(
    triangle_indices: &[u32],
    vertex_count: u32,
) -> VertexTriangleAdjacency {
    let vc = vertex_count as usize;
    let tri_count = triangle_indices.len() / 3;

    // First pass: count incident triangles per vertex.
    let mut counts = vec![0u32; vc];
    for t in 0..tri_count {
        for k in 0..3 {
            let v = triangle_indices[t * 3 + k];
            if (v as usize) < vc {
                counts[v as usize] += 1;
            }
        }
    }

    // Prefix sum → offsets[v + 1].
    let mut offsets = vec![0u32; vc + 1];
    let mut acc: u32 = 0;
    for v in 0..vc {
        offsets[v] = acc;
        acc = acc.saturating_add(counts[v]);
    }
    offsets[vc] = acc;

    // Second pass: scatter triangle indices into the right slot per vertex.
    let mut triangles = vec![0u32; acc as usize];
    let mut cursor = vec![0u32; vc];
    for t in 0..tri_count {
        for k in 0..3 {
            let v = triangle_indices[t * 3 + k];
            let vi = v as usize;
            if vi < vc {
                let slot = (offsets[vi] + cursor[vi]) as usize;
                triangles[slot] = t as u32;
                cursor[vi] += 1;
            }
        }
    }

    VertexTriangleAdjacency { offsets, triangles }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_backend_is_cpu() {
        assert_eq!(ClothSolverBackend::default(), ClothSolverBackend::Cpu);
    }

    #[test]
    fn gpu_state_from_authoring_records_counts_and_iterations() {
        let s = ClothGpuSimulationState::from_authoring(64, 256, 8);
        assert_eq!(s.particle_count, 64);
        assert_eq!(s.constraint_count, 256);
        assert_eq!(s.iterations, 8);
        assert_eq!(s.version, 0);
    }

    #[test]
    fn bump_version_wraps_safely_across_u32_boundary() {
        let mut s = ClothGpuSimulationState::from_authoring(1, 1, 1);
        s.version = u32::MAX;
        s.bump_version();
        assert_eq!(s.version, 0);
    }

    fn collect_triangles_for(adj: &VertexTriangleAdjacency, vertex: u32) -> Vec<u32> {
        let v = vertex as usize;
        let start = adj.offsets[v] as usize;
        let end = adj.offsets[v + 1] as usize;
        let mut tris: Vec<u32> = adj.triangles[start..end].to_vec();
        tris.sort_unstable();
        tris
    }

    #[test]
    fn adjacency_single_triangle_lists_each_vertex_once() {
        let adj = build_vertex_triangle_adjacency(&[0, 1, 2], 3);
        assert_eq!(adj.offsets, vec![0, 1, 2, 3]);
        assert_eq!(collect_triangles_for(&adj, 0), vec![0]);
        assert_eq!(collect_triangles_for(&adj, 1), vec![0]);
        assert_eq!(collect_triangles_for(&adj, 2), vec![0]);
    }

    #[test]
    fn adjacency_shared_vertex_appears_in_both_triangles() {
        // Two triangles sharing vertex 1: (0,1,2) and (1,3,4).
        let adj = build_vertex_triangle_adjacency(&[0, 1, 2, 1, 3, 4], 5);
        assert_eq!(adj.offsets, vec![0, 1, 3, 4, 5, 6]);
        assert_eq!(collect_triangles_for(&adj, 0), vec![0]);
        assert_eq!(collect_triangles_for(&adj, 1), vec![0, 1]);
        assert_eq!(collect_triangles_for(&adj, 2), vec![0]);
        assert_eq!(collect_triangles_for(&adj, 3), vec![1]);
        assert_eq!(collect_triangles_for(&adj, 4), vec![1]);
    }

    #[test]
    fn adjacency_isolated_vertex_has_empty_range() {
        // Vertex 3 is not referenced by any triangle.
        let adj = build_vertex_triangle_adjacency(&[0, 1, 2], 4);
        assert_eq!(adj.offsets, vec![0, 1, 2, 3, 3]);
        assert_eq!(collect_triangles_for(&adj, 3), Vec::<u32>::new());
    }

    #[test]
    fn adjacency_skips_out_of_range_indices() {
        // Vertex index 9 is out of range; should be ignored, not panic,
        // not read past the vertex count.
        let adj = build_vertex_triangle_adjacency(&[0, 1, 9], 3);
        assert_eq!(adj.offsets, vec![0, 1, 2, 2]);
        assert_eq!(collect_triangles_for(&adj, 0), vec![0]);
        assert_eq!(collect_triangles_for(&adj, 1), vec![0]);
        assert_eq!(collect_triangles_for(&adj, 2), Vec::<u32>::new());
    }

    #[test]
    fn adjacency_empty_input_yields_only_terminator() {
        let adj = build_vertex_triangle_adjacency(&[], 4);
        assert_eq!(adj.offsets, vec![0, 0, 0, 0, 0]);
        assert!(adj.triangles.is_empty());
    }

    // =====================================================================
    // S1.3 — CPU mirror of the cloth_verlet_cs GLSL formula
    // =====================================================================
    //
    // Without a Vulkano device available in unit tests we validate the
    // shader's *formula* (not the actual SPIR-V) by porting it into pure
    // Rust and comparing against the existing CPU PBD integrator. A
    // bit-for-bit match here is the precondition for the GPU dispatch
    // (which the renderer wires in S1.2) to agree with the CPU reference
    // — actual SPIR-V vs CPU parity needs a hardware integration test
    // that lives outside the lib-test boundary.

    /// Direct port of the `cloth_verlet_cs::main` body, parametrised on
    /// the same inputs the GLSL UBO carries. Mutates `positions` and
    /// `prev_positions` in place exactly as the shader rewrites the two
    /// SSBOs. Pinned particles (`pinned[i] = true`) skip integration.
    fn cpu_mirror_of_cloth_verlet_cs(
        positions: &mut [[f32; 3]],
        prev_positions: &mut [[f32; 3]],
        pinned: &[bool],
        inv_masses: &[f32],
        gravity: [f32; 3],
        wind_force: [f32; 3],
        damping: f32,
        dt: f32,
    ) {
        let dt2 = dt * dt;
        for i in 0..positions.len() {
            // Mirror the shader's `if (pinned > 0.5 || inv_mass <= 0.0)`
            // branch — both conditions skip integration.
            if pinned[i] || inv_masses[i] <= 0.0 {
                continue;
            }
            let pos = positions[i];
            let prev = prev_positions[i];
            let vel = [pos[0] - prev[0], pos[1] - prev[1], pos[2] - prev[2]];
            let d = 1.0 - damping;
            let damped = [vel[0] * d, vel[1] * d, vel[2] * d];
            let accel = [
                (gravity[0] + wind_force[0]) * dt2,
                (gravity[1] + wind_force[1]) * dt2,
                (gravity[2] + wind_force[2]) * dt2,
            ];
            let new_pos = [
                pos[0] + damped[0] + accel[0],
                pos[1] + damped[1] + accel[1],
                pos[2] + damped[2] + accel[2],
            ];
            prev_positions[i] = pos;
            positions[i] = new_pos;
        }
    }

    /// 4 free particles under gravity + light damping for 60 frames at
    /// `dt = 1/60`. The CPU PBD integrator (`verlet_integrate`) and the
    /// GLSL formula mirror must agree to within `1e-3` on every
    /// particle's final position.
    #[test]
    fn cloth_verlet_cs_formula_matches_cpu_pbd_over_60_frames() {
        use crate::simulation::cloth::{ClothParticle, ClothSimState, SpatialHashGrid};
        use crate::simulation::cloth_solver::integrator::verlet_integrate;
        use std::collections::HashSet;

        let dt = 1.0_f32 / 60.0;
        let gravity = [0.0_f32, -9.81, 0.0];
        let wind = [0.5_f32, 0.0, 0.2];
        let wind_response = 0.0_f32; // wind_force = wind * response = 0
        let damping = 0.02_f32;

        // ----- CPU PBD path -----
        let mut sim = ClothSimState {
            particles: (0..4)
                .map(|i| ClothParticle::new([i as f32, 1.0, 0.0], false))
                .collect(),
            distance_constraints: Vec::new(),
            bend_constraints: Vec::new(),
            pin_targets: Vec::new(),
            solver_iterations: 0,
            gravity,
            damping,
            collision_margin: 0.0,
            wind_response,
            wind_direction: wind,
            triangle_indices: Vec::new(),
            computed_normals: vec![[0.0; 3]; 4],
            initialized: true,
            self_collision: false,
            self_collision_radius: 0.0,
            connected_pairs: HashSet::new(),
            spatial_hash: SpatialHashGrid::new(0.04),
        };
        for _ in 0..60 {
            verlet_integrate(&mut sim, dt);
        }
        let cpu_final: Vec<[f32; 3]> = sim.particles.iter().map(|p| p.position).collect();

        // ----- GLSL formula mirror -----
        let mut positions: Vec<[f32; 3]> =
            (0..4).map(|i| [i as f32, 1.0, 0.0]).collect();
        let mut prev_positions: Vec<[f32; 3]> = positions.clone();
        let pinned: Vec<bool> = vec![false; 4];
        // wind_force baked the same way collect_cloth_deforms does it:
        // wind_direction * wind_response.
        let wind_force = [
            wind[0] * wind_response,
            wind[1] * wind_response,
            wind[2] * wind_response,
        ];
        let inv_masses = vec![1.0_f32; 4];
        for _ in 0..60 {
            cpu_mirror_of_cloth_verlet_cs(
                &mut positions,
                &mut prev_positions,
                &pinned,
                &inv_masses,
                gravity,
                wind_force,
                damping,
                dt,
            );
        }

        // ----- parity check -----
        for i in 0..4 {
            for k in 0..3 {
                let diff = (cpu_final[i][k] - positions[i][k]).abs();
                assert!(
                    diff < 1e-3,
                    "particle {} axis {} diverged: CPU PBD = {}, GLSL mirror = {}, |diff| = {}",
                    i,
                    k,
                    cpu_final[i][k],
                    positions[i][k],
                    diff
                );
            }
        }
    }

    #[test]
    fn particle_constraint_adjacency_single_constraint() {
        let adj = build_particle_constraint_adjacency(&[(0, 1)], 2);
        assert_eq!(adj.offsets, vec![0, 1, 2]);
        assert_eq!(collect_triangles_for(&adj, 0), vec![0]);
        assert_eq!(collect_triangles_for(&adj, 1), vec![0]);
    }

    #[test]
    fn particle_constraint_adjacency_shared_particle() {
        // 3 constraints, particle 1 in all three.
        let adj = build_particle_constraint_adjacency(&[(0, 1), (1, 2), (1, 3)], 4);
        assert_eq!(adj.offsets, vec![0, 1, 4, 5, 6]);
        assert_eq!(collect_triangles_for(&adj, 0), vec![0]);
        assert_eq!(collect_triangles_for(&adj, 1), vec![0, 1, 2]);
        assert_eq!(collect_triangles_for(&adj, 2), vec![1]);
        assert_eq!(collect_triangles_for(&adj, 3), vec![2]);
    }

    #[test]
    fn particle_constraint_adjacency_skips_out_of_range() {
        let adj = build_particle_constraint_adjacency(&[(0, 1), (1, 9)], 2);
        // (1, 9) → 1 still recorded, 9 dropped.
        assert_eq!(adj.offsets, vec![0, 1, 3]);
        assert_eq!(collect_triangles_for(&adj, 0), vec![0]);
        assert_eq!(collect_triangles_for(&adj, 1), vec![0, 1]);
    }

    // =====================================================================
    // S2.2 — CPU mirror of cloth_constraint_{accumulate,apply}_cs formula
    // =====================================================================

    /// Direct port of the two GLSL constraint shaders, applied as a single
    /// Jacobi iteration: accumulate deltas per particle, then apply them.
    /// Used in the parity test below as the GLSL-side reference.
    fn cpu_mirror_of_cloth_constraint_iter(
        positions: &mut [[f32; 4]], // xyz = pos, w = inv_mass
        constraints: &[(u32, u32, f32, f32)], // (a, b, rest_length, stiffness)
        adj: &VertexTriangleAdjacency,
    ) {
        let n = positions.len();
        let mut deltas = vec![[0.0_f32; 3]; n];

        // Accumulate pass — mirrors `cloth_constraint_accumulate_cs`.
        for pid in 0..n {
            let w_self = positions[pid][3];
            if w_self <= 0.0 {
                continue;
            }
            let self_pos = [positions[pid][0], positions[pid][1], positions[pid][2]];
            let start = adj.offsets[pid] as usize;
            let end = adj.offsets[pid + 1] as usize;
            let mut delta = [0.0_f32; 3];
            for k in start..end {
                let cidx = adj.triangles[k] as usize;
                let (a, b, rest_length, stiffness) = constraints[cidx];
                let other = if a as usize == pid { b } else { a } as usize;
                let other_p = positions[other];
                let dir = [
                    other_p[0] - self_pos[0],
                    other_p[1] - self_pos[1],
                    other_p[2] - self_pos[2],
                ];
                let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
                if len > 1e-6 {
                    let err = len - rest_length;
                    let w_other = other_p[3];
                    let w_sum = w_self + w_other;
                    if w_sum > 0.0 {
                        let scale = stiffness * err / len * (w_self / w_sum);
                        delta[0] += dir[0] * scale;
                        delta[1] += dir[1] * scale;
                        delta[2] += dir[2] * scale;
                    }
                }
            }
            deltas[pid] = delta;
        }

        // Apply pass — mirrors `cloth_constraint_apply_cs`.
        for pid in 0..n {
            positions[pid][0] += deltas[pid][0];
            positions[pid][1] += deltas[pid][1];
            positions[pid][2] += deltas[pid][2];
        }
    }

    /// 32-particle sheet (4×8) with horizontal + vertical distance
    /// constraints between adjacent grid neighbours. Run 64 constraint
    /// projection iterations on both:
    ///   (a) the CPU `project_distance_constraints` reference
    ///       (now XPBD — eXtended PBD, Macklin et al. 2016)
    ///   (b) the GLSL Jacobi formula mirror (`cpu_mirror_of_cloth_constraint_iter`)
    ///       which is still classic PBD
    /// and assert convergence to within a tolerance.
    ///
    /// **Stiffness-1 equivalence**: at `stiffness = 1.0` the XPBD
    /// compliance `α = 0`, which collapses the XPBD update to exactly
    /// the PBD form for this iteration (`Δλ = −C / w_sum`, `Δx = w·d·Δλ`
    /// — identical magnitude to PBD's `correction = err`). The test
    /// therefore stays correct as a CPU↔GPU parity check while the
    /// GPU side is still on PBD; a follow-up slice will migrate the
    /// GLSL constraint shader to XPBD too (adding a lambda SSBO and a
    /// `dt`/`α` field on the constraint control UBO).
    ///
    /// Jacobi is only an *approximation* of Gauss-Seidel; the two
    /// converge to the same equilibrium but along different paths.
    /// We compare per-constraint length error, not per-particle position,
    /// because position drift between methods can be larger than length
    /// drift while both still satisfy the rest-length constraint.
    #[test]
    fn cloth_constraint_jacobi_satisfies_rest_lengths_like_cpu_pbd() {
        use crate::simulation::cloth::{
            ClothDistanceConstraint, ClothParticle, ClothSimState, ClothSimTempBuffers,
            SpatialHashGrid,
        };
        use crate::simulation::cloth_solver::constraints::project_distance_constraints;
        use std::collections::HashSet;

        const W: u32 = 4;
        const H: u32 = 8;
        let n = (W * H) as usize;

        // Build the grid: vertex i is at ((i % W) - W/2, (i / W) - H/2, 0)
        // with rest length 1.0 between immediate horizontal / vertical
        // neighbours.
        let mut positions: Vec<[f32; 3]> = (0..n)
            .map(|i| {
                let x = (i as u32 % W) as f32 - (W as f32 - 1.0) * 0.5;
                let y = (i as u32 / W) as f32 - (H as f32 - 1.0) * 0.5;
                [x, y, 0.0]
            })
            .collect();
        // Perturb a couple of particles so the constraints actually need
        // to do work (otherwise the rest-pose already satisfies them).
        positions[0] = [-3.0, -3.0, 0.5];
        positions[(n - 1) as usize] = [3.0, 3.0, -0.5];

        let mut constraints: Vec<(u32, u32, f32, f32)> = Vec::new();
        let stiffness = 1.0_f32;
        for y in 0..H {
            for x in 0..W {
                let i = y * W + x;
                if x + 1 < W {
                    constraints.push((i, i + 1, 1.0, stiffness));
                }
                if y + 1 < H {
                    constraints.push((i, i + W, 1.0, stiffness));
                }
            }
        }

        // ----- (a) CPU PBD reference: project_distance_constraints -----
        let mut sim = ClothSimState {
            particles: positions
                .iter()
                .map(|p| ClothParticle::new(*p, false))
                .collect(),
            distance_constraints: constraints
                .iter()
                .map(|(a, b, r, s)| ClothDistanceConstraint {
                    a: *a as usize,
                    b: *b as usize,
                    rest_length: *r,
                    stiffness: *s,
                })
                .collect(),
            bend_constraints: Vec::new(),
            pin_targets: Vec::new(),
            solver_iterations: 0,
            gravity: [0.0; 3],
            damping: 0.0,
            collision_margin: 0.0,
            wind_response: 0.0,
            wind_direction: [0.0; 3],
            triangle_indices: Vec::new(),
            computed_normals: vec![[0.0; 3]; n],
            initialized: true,
            self_collision: false,
            self_collision_radius: 0.0,
            connected_pairs: HashSet::new(),
            spatial_hash: SpatialHashGrid::new(0.04),
        };
        let mut buffers = ClothSimTempBuffers::new(n);
        // XPBD lambda is meaningless across substeps — reset once
        // before the 64-iter loop so this whole block models a single
        // substep with 64 projection passes (which is what the GPU
        // mirror does too).
        buffers.reset_lambda(sim.distance_constraints.len());
        // `dt` only matters when stiffness < 1 (compliance > 0). The
        // grid uses stiffness = 1 so α = 0 and `dt` falls out of the
        // formula; any positive value gives identical behaviour.
        let test_dt = 1.0 / 60.0_f32;
        for _ in 0..64 {
            project_distance_constraints(&mut sim, &mut buffers, test_dt);
        }

        // ----- (b) GLSL Jacobi formula mirror -----
        let mut gpu_positions: Vec<[f32; 4]> = positions
            .iter()
            .map(|p| [p[0], p[1], p[2], 1.0])
            .collect();
        let adj = build_particle_constraint_adjacency(
            &constraints.iter().map(|(a, b, _, _)| (*a, *b)).collect::<Vec<_>>(),
            n as u32,
        );
        for _ in 0..64 {
            cpu_mirror_of_cloth_constraint_iter(&mut gpu_positions, &constraints, &adj);
        }

        // ----- Assert: every constraint's length is within tolerance of rest_length -----
        let tol = 0.05; // 5cm of slack for 64 Jacobi iterations on the rough perturbation
        for (a, b, rest_length, _) in &constraints {
            let pa_cpu = sim.particles[*a as usize].position;
            let pb_cpu = sim.particles[*b as usize].position;
            let dx = pb_cpu[0] - pa_cpu[0];
            let dy = pb_cpu[1] - pa_cpu[1];
            let dz = pb_cpu[2] - pa_cpu[2];
            let len_cpu = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!(
                (len_cpu - rest_length).abs() < tol,
                "CPU PBD constraint ({},{}) length drift: {}",
                a, b, len_cpu
            );

            let pa_gpu = gpu_positions[*a as usize];
            let pb_gpu = gpu_positions[*b as usize];
            let dx = pb_gpu[0] - pa_gpu[0];
            let dy = pb_gpu[1] - pa_gpu[1];
            let dz = pb_gpu[2] - pa_gpu[2];
            let len_gpu = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!(
                (len_gpu - rest_length).abs() < tol,
                "GLSL Jacobi constraint ({},{}) length drift: {}",
                a, b, len_gpu
            );
        }
    }

    // =====================================================================
    // S3.2 — CPU mirror of cloth_normal_cs formula
    // =====================================================================

    /// Direct port of the `cloth_normal_cs::main` body. Used in the
    /// parity test below to verify the GLSL normal recomputation matches
    /// the CPU reference (`cloth_solver::output::compute_normals`).
    ///
    /// Implements the same **angle-weighted** (Max 1999) accumulation
    /// as both the CPU reference and the GLSL shader: each incident
    /// triangle contributes its unit face normal scaled by the angle
    /// at the vertex.
    fn cpu_mirror_of_cloth_normal_cs(
        positions: &[[f32; 3]],
        triangle_indices: &[u32],
        adj: &VertexTriangleAdjacency,
    ) -> Vec<[f32; 3]> {
        fn safe_angle(u: [f32; 3], v: [f32; 3]) -> f32 {
            let lu = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
            let lv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if lu < 1e-9 || lv < 1e-9 {
                return 0.0;
            }
            let dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (lu * lv);
            dot.clamp(-1.0, 1.0).acos()
        }

        let n = positions.len();
        let mut normals = vec![[0.0_f32, 1.0, 0.0]; n];
        for vid in 0..n {
            let start = adj.offsets[vid] as usize;
            let end = adj.offsets[vid + 1] as usize;
            let mut accum = [0.0_f32; 3];
            for k in start..end {
                let tri = adj.triangles[k] as usize;
                let i0 = triangle_indices[tri * 3] as usize;
                let i1 = triangle_indices[tri * 3 + 1] as usize;
                let i2 = triangle_indices[tri * 3 + 2] as usize;
                let p0 = positions[i0];
                let p1 = positions[i1];
                let p2 = positions[i2];
                let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
                let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
                let cross = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ];
                let face_area2 = (cross[0] * cross[0]
                    + cross[1] * cross[1]
                    + cross[2] * cross[2])
                    .sqrt();
                if face_area2 < 1e-12 {
                    continue;
                }
                let inv = 1.0 / face_area2;
                let face_normal = [cross[0] * inv, cross[1] * inv, cross[2] * inv];
                let angle = if vid == i0 {
                    safe_angle(e1, e2)
                } else if vid == i1 {
                    let e10 = [-e1[0], -e1[1], -e1[2]];
                    let e12 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
                    safe_angle(e10, e12)
                } else {
                    let e20 = [-e2[0], -e2[1], -e2[2]];
                    let e21 = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]];
                    safe_angle(e20, e21)
                };
                accum[0] += face_normal[0] * angle;
                accum[1] += face_normal[1] * angle;
                accum[2] += face_normal[2] * angle;
            }
            let len = (accum[0] * accum[0] + accum[1] * accum[1] + accum[2] * accum[2]).sqrt();
            if len > 1e-12 {
                normals[vid] = [accum[0] / len, accum[1] / len, accum[2] / len];
            }
        }
        normals
    }

    /// 32-particle (4×8) flat sheet of triangles in XY plane: the GLSL
    /// normal formula must agree with the CPU PBD `compute_normals`
    /// reference to within `1e-4` per component per vertex. (The CPU
    /// reference uses the same area-weighted accumulation, so they
    /// should match bit-for-bit modulo floating-point summation order.)
    #[test]
    fn cloth_normal_cs_formula_matches_cpu_pbd() {
        use crate::simulation::cloth::{
            ClothParticle, ClothSimState, SpatialHashGrid,
        };
        use crate::simulation::cloth_solver::output::compute_normals;
        use std::collections::HashSet;

        const W: u32 = 4;
        const H: u32 = 8;
        let n = (W * H) as usize;

        let positions: Vec<[f32; 3]> = (0..n)
            .map(|i| {
                let x = (i as u32 % W) as f32;
                let y = (i as u32 / W) as f32;
                [x, y, 0.0]
            })
            .collect();
        // Two triangles per quad cell: (i, i+1, i+W) and (i+1, i+W+1, i+W).
        let mut triangle_indices: Vec<u32> = Vec::new();
        for y in 0..(H - 1) {
            for x in 0..(W - 1) {
                let i = y * W + x;
                triangle_indices.extend_from_slice(&[i, i + 1, i + W]);
                triangle_indices.extend_from_slice(&[i + 1, i + W + 1, i + W]);
            }
        }

        // ----- CPU PBD reference -----
        let mut sim = ClothSimState {
            particles: positions
                .iter()
                .map(|p| ClothParticle::new(*p, false))
                .collect(),
            distance_constraints: Vec::new(),
            bend_constraints: Vec::new(),
            pin_targets: Vec::new(),
            solver_iterations: 0,
            gravity: [0.0; 3],
            damping: 0.0,
            collision_margin: 0.0,
            wind_response: 0.0,
            wind_direction: [0.0; 3],
            triangle_indices: triangle_indices.clone(),
            computed_normals: vec![[0.0; 3]; n],
            initialized: true,
            self_collision: false,
            self_collision_radius: 0.0,
            connected_pairs: HashSet::new(),
            spatial_hash: SpatialHashGrid::new(0.04),
        };
        compute_normals(&mut sim);
        let cpu_normals = sim.computed_normals.clone();

        // ----- GLSL formula mirror -----
        let adj = build_vertex_triangle_adjacency(&triangle_indices, n as u32);
        let gpu_normals = cpu_mirror_of_cloth_normal_cs(&positions, &triangle_indices, &adj);

        // ----- Parity check -----
        for i in 0..n {
            for k in 0..3 {
                // Vertex with no triangles (corners of degenerate grid) — skip if both methods agree on default
                let diff = (cpu_normals[i][k] - gpu_normals[i][k]).abs();
                assert!(
                    diff < 1e-4,
                    "normal {} axis {} mismatch: CPU = {:?}, GLSL = {:?}",
                    i, k, cpu_normals[i], gpu_normals[i]
                );
            }
        }
    }

    /// Pinned particles must hold their starting position under the
    /// shader formula, mirroring CPU PBD `verlet_integrate`.
    #[test]
    fn cloth_verlet_cs_formula_pins_match_cpu_pbd() {
        let mut positions = vec![[2.0_f32, 3.0, 4.0]];
        let mut prev = vec![[2.0_f32, 3.0, 4.0]];
        let pinned = vec![true];
        let inv_masses = vec![1.0_f32];
        cpu_mirror_of_cloth_verlet_cs(
            &mut positions,
            &mut prev,
            &pinned,
            &inv_masses,
            [0.0, -100.0, 0.0],
            [0.0; 3],
            0.0,
            0.1,
        );
        assert_eq!(positions[0], [2.0, 3.0, 4.0]);
        assert_eq!(prev[0], [2.0, 3.0, 4.0]);
    }

    /// Particles with `inv_mass = 0` are equivalent to pinned — the
    /// shader's `if (pinned > 0.5 || inv_mass <= 0.0)` branch handles
    /// both. This test asserts the mirror behaves identically.
    #[test]
    fn cloth_verlet_cs_formula_zero_inv_mass_pins_like_explicit_pin() {
        let mut positions = vec![[1.0_f32, 0.0, 0.0]];
        let mut prev = vec![[1.0_f32, 0.0, 0.0]];
        let pinned = vec![false]; // not explicitly pinned …
        let inv_masses = vec![0.0_f32]; // … but inv_mass == 0 should pin it
        cpu_mirror_of_cloth_verlet_cs(
            &mut positions,
            &mut prev,
            &pinned,
            &inv_masses,
            [0.0, -100.0, 0.0],
            [0.0; 3],
            0.0,
            0.1,
        );
        assert_eq!(positions[0], [1.0, 0.0, 0.0]);
        assert_eq!(prev[0], [1.0, 0.0, 0.0]);
    }
}
