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
        gravity: [f32; 3],
        wind_force: [f32; 3],
        damping: f32,
        dt: f32,
    ) {
        let dt2 = dt * dt;
        for i in 0..positions.len() {
            if pinned[i] {
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
        for _ in 0..60 {
            cpu_mirror_of_cloth_verlet_cs(
                &mut positions,
                &mut prev_positions,
                &pinned,
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

    /// Pinned particles must hold their starting position under the
    /// shader formula, mirroring CPU PBD `verlet_integrate`.
    #[test]
    fn cloth_verlet_cs_formula_pins_match_cpu_pbd() {
        let mut positions = vec![[2.0_f32, 3.0, 4.0]];
        let mut prev = vec![[2.0_f32, 3.0, 4.0]];
        let pinned = vec![true];
        cpu_mirror_of_cloth_verlet_cs(
            &mut positions,
            &mut prev,
            &pinned,
            [0.0, -100.0, 0.0],
            [0.0; 3],
            0.0,
            0.1,
        );
        assert_eq!(positions[0], [2.0, 3.0, 4.0]);
        assert_eq!(prev[0], [2.0, 3.0, 4.0]);
    }
}
