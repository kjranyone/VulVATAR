//! Cloth ownership boundary between authoring, CPU simulation, GPU simulation,
//! and render-consumable deform.
//!
//! Cloth GPU-state boundary (the compute-cloth migration's P3-01 slot).
//! No runtime behaviour change: this module re-exports the existing
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
    pub fn from_authoring(particle_count: u32, constraint_count: u32, iterations: u32) -> Self {
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

/// CSR adjacency mapping particle → the particles it shares a distance
/// constraint with (each endpoint lists the other). The GPU
/// self-collision resolve pass uses it to skip constraint-connected
/// pairs, mirroring the CPU solver's `connected_pairs` set. Same CSR
/// shape as [`build_particle_constraint_adjacency`] but with particle
/// indices in the scatter array (that one carries constraint indices,
/// which the constraint accumulate pass needs instead).
pub fn build_particle_particle_adjacency(
    pairs: &[(u32, u32)],
    particle_count: u32,
) -> VertexTriangleAdjacency {
    let n = particle_count as usize;
    let in_range =
        |v: u32| (v as usize) < n;
    let mut degree = vec![0u32; n + 1];
    for &(a, b) in pairs {
        // Pairs with either endpoint out of range are dropped entirely
        // — degrees and scatter must agree.
        if in_range(a) && in_range(b) {
            degree[a as usize + 1] += 1;
            degree[b as usize + 1] += 1;
        }
    }
    for i in 0..n {
        degree[i + 1] += degree[i];
    }
    let mut scatter = vec![0u32; degree[n] as usize];
    let mut cursor = degree.clone();
    for &(a, b) in pairs {
        if (a as usize) < n && (b as usize) < n {
            scatter[cursor[a as usize] as usize] = b;
            cursor[a as usize] += 1;
            scatter[cursor[b as usize] as usize] = a;
            cursor[b as usize] += 1;
        }
    }
    VertexTriangleAdjacency {
        offsets: degree,
        triangles: scatter,
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
    /// GPU compute path. Opt-in at attach time via
    /// [`solver_backend_from_env`] (`VULVATAR_CLOTH_GPU=1`); the
    /// renderer dispatches verlet + XPBD + normals on the GPU and the
    /// CPU solver skips its XPBD pass. Known gaps vs `Cpu`: no GPU
    /// collider stage (cloth can pass through the body), and
    /// CPU-side consumers (`ClothState::deform_output`, the cloth
    /// inspector's live particle view, the manual Step button) see
    /// the attach-time rest pose because nothing reads positions back.
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

/// One-shot backend selection for freshly attached cloth
/// (`VULVATAR_CLOTH_GPU=1` → `Gpu`, anything else / unset → `Cpu`).
/// Cached in a `OnceLock` because the selection is per-attach anyway
/// and a mid-session env flip must not produce mixed backends on
/// overlay slots. Promoting `Gpu` to the default (or driving it from
/// `RuntimeGpuBudget`) is deliberately not done yet: see the collider
/// / readback gaps on [`ClothSolverBackend::Gpu`].
/// Runtime backend request for freshly attached cloth, published from
/// the persisted `AppSettings::cloth_gpu_backend` before the first
/// attach of the session: `None` keeps the `VULVATAR_CLOTH_GPU` env
/// decision, `Some(true/false)` overrides it. A plain static (one
/// Application writes before any attach; attach sites only read) —
/// same shape as the tracking cadence atomics.
static CLOTH_BACKEND_REQUESTED: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0); // 0 = unset, 1 = force CPU, 2 = force GPU

/// Publish the persisted backend preference. Must run before the first
/// cloth attach of the session (attach caches its decision per cloth).
pub fn set_cloth_backend_request(request: Option<bool>) {
    use std::sync::atomic::Ordering;
    CLOTH_BACKEND_REQUESTED.store(
        match request {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}

pub fn solver_backend_from_env() -> ClothSolverBackend {
    static CACHE: std::sync::OnceLock<ClothSolverBackend> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        match CLOTH_BACKEND_REQUESTED.load(std::sync::atomic::Ordering::Relaxed) {
            1 => ClothSolverBackend::Cpu,
            2 => ClothSolverBackend::Gpu,
            _ => match std::env::var_os("VULVATAR_CLOTH_GPU") {
                Some(v) if v != "0" => ClothSolverBackend::Gpu,
                _ => ClothSolverBackend::Cpu,
            },
        }
    })
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

        // ----- CPU XPBD path -----
        let mut sim = ClothSimState {
            particles: (0..4)
                .map(|i| ClothParticle::new([i as f32, 1.0, 0.0], false))
                .collect(),
            distance_constraints: Vec::new(),
            bend_constraints: Vec::new(),
            pin_targets: Vec::new(),
            solver_iterations: 0,
            gravity,
            gravity_scale: 1.0,
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
            sdf_contact: 0.0,
        };
        for _ in 0..60 {
            verlet_integrate(&mut sim, dt);
        }
        let cpu_final: Vec<[f32; 3]> = sim.particles.iter().map(|p| p.position).collect();

        // ----- GLSL formula mirror -----
        let mut positions: Vec<[f32; 3]> = (0..4).map(|i| [i as f32, 1.0, 0.0]).collect();
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

    /// Direct port of the three GLSL XPBD constraint shaders, applied
    /// as a single iteration: per-constraint lambda update + Δλ_j write,
    /// per-particle Δx accumulate (reading Δλ), then apply.
    ///
    /// `lambda` accumulates across calls within one substep — caller
    /// is responsible for zeroing it at the start of each substep,
    /// matching the renderer's `cmd.fill_buffer(lambda_ssbo, 0)` reset
    /// in `mod.rs`.
    fn cpu_mirror_of_cloth_constraint_iter(
        positions: &mut [[f32; 4]],           // xyz = pos, w = inv_mass
        constraints: &[(u32, u32, f32, f32)], // (a, b, rest_length, stiffness)
        adj: &VertexTriangleAdjacency,
        lambda: &mut [f32],
        dt: f32,
    ) {
        fn compliance(stiffness: f32) -> f32 {
            let s = stiffness.clamp(0.0, 1.0);
            let slack = (1.0 - s).max(0.0);
            slack * slack * 1.0e-7
        }

        let n = positions.len();
        let dt_sq = (dt * dt).max(1.0e-12);
        let mut dlambda = vec![0.0_f32; constraints.len()];

        // Pass 1 — per-constraint Δλ_j (mirror of
        // `cloth_constraint_lambda_update_cs`).
        for (j, &(a, b, rest_length, stiffness)) in constraints.iter().enumerate() {
            if stiffness <= 1.0e-6 {
                dlambda[j] = 0.0;
                continue;
            }
            let pa = positions[a as usize];
            let pb = positions[b as usize];
            let diff = [pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]];
            let dist = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
            // 1e-9 m = 1 nm. Below this both endpoints are numerically
            // coincident; matches the GLSL + CPU XPBD threshold.
            if dist < 1.0e-9 {
                dlambda[j] = 0.0;
                continue;
            }
            let w_a = pa[3];
            let w_b = pb[3];
            let w_sum = w_a + w_b;
            if w_sum < 1.0e-12 {
                dlambda[j] = 0.0;
                continue;
            }
            let alpha_tilde = compliance(stiffness) / dt_sq;
            let c = dist - rest_length;
            let lambda_old = lambda[j];
            let denom = w_sum + alpha_tilde;
            let dl = (-c - alpha_tilde * lambda_old) / denom;
            lambda[j] = lambda_old + dl;
            dlambda[j] = dl;
        }

        // Pass 2 — per-particle Δx accumulate (mirror of
        // `cloth_constraint_accumulate_cs`).
        let mut deltas = vec![[0.0_f32; 3]; n];
        for pid in 0..n {
            let w_self = positions[pid][3];
            if w_self <= 0.0 {
                continue;
            }
            let self_pos = [positions[pid][0], positions[pid][1], positions[pid][2]];
            let start = adj.offsets[pid] as usize;
            let end = adj.offsets[pid + 1] as usize;
            let mut delta = [0.0_f32; 3];
            // Under-relaxed Jacobi denominator (n + 1) — mirrors the
            // GLSL accumulate pass and the CPU apply loop.
            let mut n_rel = 1.0_f32;
            for k in start..end {
                let cidx = adj.triangles[k] as usize;
                let (a, b, _, _) = constraints[cidx];
                let other = if a as usize == pid { b } else { a } as usize;
                let other_p = positions[other];
                let dir = [
                    other_p[0] - self_pos[0],
                    other_p[1] - self_pos[1],
                    other_p[2] - self_pos[2],
                ];
                let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
                if len > 1.0e-9 {
                    // Δx_self = -d_unit · w_self · Δλ_j
                    //   where d_unit = (other - self) / len
                    let dl = dlambda[cidx];
                    let scale = -w_self * dl / len;
                    delta[0] += dir[0] * scale;
                    delta[1] += dir[1] * scale;
                    delta[2] += dir[2] * scale;
                    n_rel += 1.0;
                }
            }
            deltas[pid] = [
                delta[0] / n_rel,
                delta[1] / n_rel,
                delta[2] / n_rel,
            ];
        }

        // Pass 3 — apply Δx (mirror of `cloth_constraint_apply_cs`).
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
    ///       (XPBD — eXtended PBD, Macklin et al. 2016)
    ///   (b) the GLSL XPBD formula mirror (`cpu_mirror_of_cloth_constraint_iter`)
    ///       which mirrors the three-pass lambda-update / accumulate /
    ///       apply dispatch in the GLSL shaders.
    ///
    /// Sweeps `stiffness ∈ {1.0, 0.7, 0.3}` to cover the rigid limit
    /// (`α = 0`, XPBD ≡ PBD) and two compliance regimes where the
    /// XPBD-specific `λ` accumulation matters.
    ///
    /// Without external load, every stiffness setting wants to drive
    /// the constraints to their rest length — the compliance shifts
    /// only the per-iteration step size and the iteration count to
    /// reach a given tolerance, not the asymptote. 64 iterations is
    /// enough that all three stiffness values land inside the 5cm
    /// tolerance.
    ///
    /// Jacobi is only an *approximation* of Gauss-Seidel; the two
    /// converge to the same equilibrium but along different paths.
    /// We compare per-constraint length error, not per-particle
    /// position, because position drift between methods can be
    /// larger than length drift while both still satisfy the
    /// rest-length constraint.
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
        // neighbours. The perturbation moves two corners off the rest
        // pose so the constraints actually need to do work.
        let mut base_positions: Vec<[f32; 3]> = (0..n)
            .map(|i| {
                let x = (i as u32 % W) as f32 - (W as f32 - 1.0) * 0.5;
                let y = (i as u32 / W) as f32 - (H as f32 - 1.0) * 0.5;
                [x, y, 0.0]
            })
            .collect();
        base_positions[0] = [-3.0, -3.0, 0.5];
        base_positions[(n - 1) as usize] = [3.0, 3.0, -0.5];

        // Build the constraint list once; the stiffness sweep below
        // mutates the per-edge `stiffness` per iteration.
        let mut constraint_pairs: Vec<(u32, u32, f32)> = Vec::new();
        for y in 0..H {
            for x in 0..W {
                let i = y * W + x;
                if x + 1 < W {
                    constraint_pairs.push((i, i + 1, 1.0));
                }
                if y + 1 < H {
                    constraint_pairs.push((i, i + W, 1.0));
                }
            }
        }
        let adj = build_particle_constraint_adjacency(
            &constraint_pairs
                .iter()
                .map(|(a, b, _)| (*a, *b))
                .collect::<Vec<_>>(),
            n as u32,
        );

        for &stiffness in &[1.0_f32, 0.7, 0.3] {
            let constraints: Vec<(u32, u32, f32, f32)> = constraint_pairs
                .iter()
                .map(|(a, b, r)| (*a, *b, *r, stiffness))
                .collect();
            let test_dt = 1.0 / 60.0_f32;

            // ----- (a) CPU XPBD reference -----
            let mut sim = ClothSimState {
                particles: base_positions
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
                gravity_scale: 1.0,
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
            sdf_contact: 0.0,
            };
            let mut buffers = ClothSimTempBuffers::new(n);
            buffers.reset_lambda(sim.distance_constraints.len());
            for _ in 0..64 {
                project_distance_constraints(&mut sim, &mut buffers, test_dt);
            }

            // ----- (b) GLSL XPBD formula mirror -----
            let mut gpu_positions: Vec<[f32; 4]> = base_positions
                .iter()
                .map(|p| [p[0], p[1], p[2], 1.0])
                .collect();
            let mut gpu_lambda = vec![0.0_f32; constraints.len()];
            for _ in 0..64 {
                cpu_mirror_of_cloth_constraint_iter(
                    &mut gpu_positions,
                    &constraints,
                    &adj,
                    &mut gpu_lambda,
                    test_dt,
                );
            }

            // ----- Assert CPU XPBD and GPU mirror agree per-particle -----
            // After review pass 10 dropped the CPU's Jacobi averaging,
            // both paths run identical XPBD updates and should match
            // within float-rounding noise. The constraint-length
            // assertion below is the convergence sanity check; the
            // per-particle assertion is the actual parity guarantee.
            let pos_tol = 1.0e-4_f32; // ~0.1 mm — float-summation noise.
            for i in 0..n as usize {
                let pc = sim.particles[i].position;
                let pg = gpu_positions[i];
                let dx = pg[0] - pc[0];
                let dy = pg[1] - pc[1];
                let dz = pg[2] - pc[2];
                let drift = (dx * dx + dy * dy + dz * dz).sqrt();
                assert!(
                    drift < pos_tol,
                    "CPU XPBD vs GLSL XPBD parity broke @ particle {} stiffness={}: drift={}",
                    i,
                    stiffness,
                    drift
                );
            }

            // ----- Assert every constraint's length lands inside tol -----
            let tol = 0.05; // 5 cm slack across stiffness range, 64 iters.
            for (a, b, rest_length, _) in &constraints {
                let pa_cpu = sim.particles[*a as usize].position;
                let pb_cpu = sim.particles[*b as usize].position;
                let dx = pb_cpu[0] - pa_cpu[0];
                let dy = pb_cpu[1] - pa_cpu[1];
                let dz = pb_cpu[2] - pa_cpu[2];
                let len_cpu = (dx * dx + dy * dy + dz * dz).sqrt();
                assert!(
                    (len_cpu - rest_length).abs() < tol,
                    "CPU XPBD constraint ({},{}) length drift @ stiffness={}: {}",
                    a,
                    b,
                    stiffness,
                    len_cpu
                );

                let pa_gpu = gpu_positions[*a as usize];
                let pb_gpu = gpu_positions[*b as usize];
                let dx = pb_gpu[0] - pa_gpu[0];
                let dy = pb_gpu[1] - pa_gpu[1];
                let dz = pb_gpu[2] - pa_gpu[2];
                let len_gpu = (dx * dx + dy * dy + dz * dz).sqrt();
                assert!(
                    (len_gpu - rest_length).abs() < tol,
                    "GLSL XPBD constraint ({},{}) length drift @ stiffness={}: {}",
                    a,
                    b,
                    stiffness,
                    len_gpu
                );
            }
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
                let face_area2 =
                    (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
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
    /// normal formula must agree with the CPU `compute_normals`
    /// reference to within `1e-4` per component per vertex. Both paths
    /// use angle-weighted accumulation (Max 1999), so they should match
    /// bit-for-bit modulo floating-point summation order. Isolated /
    /// degenerate vertices fall back to `(0, 1, 0)` on both sides.
    #[test]
    fn cloth_normal_cs_formula_matches_cpu_pbd() {
        use crate::simulation::cloth::{ClothParticle, ClothSimState, SpatialHashGrid};
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

        // ----- CPU XPBD reference -----
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
            gravity_scale: 1.0,
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
            sdf_contact: 0.0,
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
                    i,
                    k,
                    cpu_normals[i],
                    gpu_normals[i]
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

    /// Rust mirror of `cloth_collide_cs`'s per-particle body: sequential
    /// capsule loop with the position mutating across iterations, exactly
    /// as the shader does (closest-point degenerate-segment handling,
    /// radius+margin projection, 1e-12 guards).
    fn collide_cs_project(mut pos: [f32; 3], capsules: &[[f32; 7]], margin: f32) -> [f32; 3] {
        for cap in capsules {
            let (a, b) = ([cap[0], cap[1], cap[2]], [cap[3], cap[4], cap[5]]);
            let radius = cap[6] + margin;
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ap = [pos[0] - a[0], pos[1] - a[1], pos[2] - a[2]];
            let ab_len_sq = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
            let closest = if ab_len_sq >= 1.0e-12 {
                let t = ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / ab_len_sq)
                    .clamp(0.0, 1.0);
                [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t]
            } else {
                a
            };
            let diff = [pos[0] - closest[0], pos[1] - closest[1], pos[2] - closest[2]];
            let dist = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
            if dist < radius && dist > 1.0e-12 {
                let n = [diff[0] / dist, diff[1] / dist, diff[2] / dist];
                pos = [
                    closest[0] + n[0] * radius,
                    closest[1] + n[1] * radius,
                    closest[2] + n[2] * radius,
                ];
            }
        }
        pos
    }

    /// CPU reference: transcription of `cloth_solver::collision::collide`
    /// for one particle (sphere + capsule variants, same guards), using
    /// the shared closest-point helper.
    fn cpu_collide_particle(
        mut pos: [f32; 3],
        capsules: &[[f32; 7]],
        margin: f32,
    ) -> [f32; 3] {
        use crate::math_utils::{closest_point_on_segment, vec3_add, vec3_scale, vec3_sub};
        for cap in capsules {
            let a = [cap[0], cap[1], cap[2]];
            let b = [cap[3], cap[4], cap[5]];
            let radius = cap[6] + margin;
            let closest = closest_point_on_segment(&a, &b, &pos);
            let diff = vec3_sub(&pos, &closest);
            let dist = crate::math_utils::vec3_length(&diff);
            if dist < radius && dist > 1e-12 {
                let normal = vec3_scale(&diff, 1.0 / dist);
                pos = vec3_add(&closest, &vec3_scale(&normal, radius));
            }
        }
        pos
    }

    /// Deterministic LCG so failures reproduce without a seed dependency.
    fn lcg(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as f32 / (u32::MAX >> 1) as f32) - 1.0
    }

    #[test]
    fn cloth_collide_cs_formula_matches_cpu_collision() {
        let mut rng = 789_101_112u64;
        for case in 0..200 {
            let n_caps = 1 + (case % 4);
            let mut capsules = Vec::new();
            for _ in 0..n_caps {
                let degenerate = lcg(&mut rng) > 0.5; // sphere ⇔ p0 == p1
                let p0 = [lcg(&mut rng), lcg(&mut rng), lcg(&mut rng)];
                let p1 = if degenerate {
                    p0
                } else {
                    [lcg(&mut rng), lcg(&mut rng), lcg(&mut rng)]
                };
                let radius = 0.05 + lcg(&mut rng).abs() * 0.3;
                capsules.push([p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], radius]);
            }
            let margin = (lcg(&mut rng).abs() * 0.05).max(0.0);
            // Half the particles start inside a capsule so the projection
            // path (not just the miss path) is exercised.
            let base = if case % 2 == 0 {
                [capsules[0][0], capsules[0][1], capsules[0][2]]
            } else {
                [lcg(&mut rng), lcg(&mut rng), lcg(&mut rng)]
            };
            let pos = [
                base[0] + lcg(&mut rng) * 0.1,
                base[1] + lcg(&mut rng) * 0.1,
                base[2] + lcg(&mut rng) * 0.1,
            ];
            let gpu = collide_cs_project(pos, &capsules, margin);
            let cpu = cpu_collide_particle(pos, &capsules, margin);
            for k in 0..3 {
                assert!(
                    (gpu[k] - cpu[k]).abs() < 1e-5,
                    "case {case} axis {k}: gpu {gpu:?} vs cpu {cpu:?}"
                );
            }
        }
    }

    #[test]
    fn particle_particle_adjacency_lists_both_endpoints() {
        // Chain 0-1-2 plus an out-of-range pair that must be dropped.
        let adj = build_particle_particle_adjacency(
            &[(0, 1), (1, 2), (2, 9)],
            3,
        );
        assert_eq!(&adj.offsets, &[0, 1, 3, 4]);
        assert_eq!(&adj.triangles, &[1, 0, 2, 1]);
    }

    /// GPU mirror of `cloth_selfcol_resolve_cs`'s per-particle body
    /// with the grid replaced by a brute-force scan over all particles
    /// — identical pair set (the 27-cell neighbourhood is lossless for
    /// distances < min_dist), so formula parity is exact.
    fn selfcol_resolve_mirror(
        positions: &[[f32; 3]],
        inv_mass: &[f32],
        connected: &std::collections::HashSet<(usize, usize)>,
        radius: f32,
    ) -> Vec<[f32; 3]> {
        let n = positions.len();
        let min_dist = 2.0 * radius;
        let mut out = positions.to_vec();
        for i in 0..n {
            if inv_mass[i] <= 0.0 {
                continue;
            }
            let mut acc = [0.0f32; 3];
            let mut hits = 0usize;
            for j in 0..n {
                if j == i || inv_mass[j] <= 0.0 {
                    continue;
                }
                let lo = i.min(j);
                let hi = i.max(j);
                if connected.contains(&(lo, hi)) {
                    continue;
                }
                let diff = [
                    positions[j][0] - positions[i][0],
                    positions[j][1] - positions[i][1],
                    positions[j][2] - positions[i][2],
                ];
                let d2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
                if d2 < min_dist * min_dist && d2 > 1.0e-24 {
                    let d = d2.sqrt();
                    let overlap = min_dist - d;
                    let dir = [diff[0] / d, diff[1] / d, diff[2] / d];
                    acc[0] -= dir[0] * (overlap * 0.5);
                    acc[1] -= dir[1] * (overlap * 0.5);
                    acc[2] -= dir[2] * (overlap * 0.5);
                    hits += 1;
                }
            }
            if hits > 0 {
                for k in 0..3 {
                    out[i][k] += acc[k] / hits as f32;
                }
            }
        }
        out
    }

    /// CPU reference: transcription of
    /// `cloth_solver::collision::resolve_self_collisions` with the
    /// spatial hash replaced by the same brute-force pair set.
    fn selfcol_resolve_cpu(
        positions: &[[f32; 3]],
        pinned: &[bool],
        connected: &std::collections::HashSet<(usize, usize)>,
        radius: f32,
    ) -> Vec<[f32; 3]> {
        let n = positions.len();
        let min_dist = 2.0 * radius;
        let mut corrections = vec![[0.0f32; 3]; n];
        let mut counts = vec![0u32; n];
        for i in 0..n {
            if pinned[i] {
                continue;
            }
            for j in (i + 1)..n {
                if pinned[j] {
                    continue;
                }
                if connected.contains(&(i, j)) {
                    continue;
                }
                let diff = [
                    positions[j][0] - positions[i][0],
                    positions[j][1] - positions[i][1],
                    positions[j][2] - positions[i][2],
                ];
                let d2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
                if d2 < min_dist * min_dist && d2 > 1.0e-24 {
                    let d = d2.sqrt();
                    let overlap = min_dist - d;
                    let dir = [diff[0] / d, diff[1] / d, diff[2] / d];
                    let half = [dir[0] * overlap * 0.5, dir[1] * overlap * 0.5, dir[2] * overlap * 0.5];
                    for k in 0..3 {
                        corrections[i][k] -= half[k];
                        corrections[j][k] += half[k];
                    }
                    counts[i] += 1;
                    counts[j] += 1;
                }
            }
        }
        let mut out = positions.to_vec();
        for i in 0..n {
            if counts[i] > 0 && !pinned[i] {
                for k in 0..3 {
                    out[i][k] += corrections[i][k] / counts[i] as f32;
                }
            }
        }
        out
    }

    #[test]
    fn cloth_selfcol_resolve_formula_matches_cpu() {
        let mut rng_state = 42_424_242u64;
        let lcg = |s: &mut u64| -> f32 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*s >> 33) as f32 / (u32::MAX >> 1) as f32) - 1.0
        };
        for case in 0..60 {
            let n = 6 + (case % 10);
            let radius = 0.01 + lcg(&mut rng_state).abs() * 0.02;
            // Cluster particles inside one min_dist ball so the push
            // paths actually fire, plus a few spread far away.
            let mut positions = Vec::with_capacity(n);
            for i in 0..n {
                if i % 3 == 0 {
                    positions.push([
                        lcg(&mut rng_state) * radius,
                        lcg(&mut rng_state) * radius,
                        lcg(&mut rng_state) * radius,
                    ]);
                } else {
                    positions.push([
                        lcg(&mut rng_state) * 10.0,
                        lcg(&mut rng_state) * 10.0,
                        lcg(&mut rng_state) * 10.0,
                    ]);
                }
            }
            let mut pinned = vec![false; n];
            for (i, p) in pinned.iter_mut().enumerate() {
                *p = i % 5 == 0; // some pinned particles on both sides
            }
            let inv_mass: Vec<f32> = pinned.iter().map(|&p| if p { 0.0 } else { 1.0 }).collect();
            // Constraint connectivity mirrors the CPU `connected_pairs`
            // construction (both endpoint orderings in the set).
            let mut connected = std::collections::HashSet::new();
            for i in 0..n.saturating_sub(1) {
                if i % 2 == 0 {
                    connected.insert((i, i + 1));
                }
            }

            let gpu = selfcol_resolve_mirror(&positions, &inv_mass, &connected, radius);
            let cpu = selfcol_resolve_cpu(&positions, &pinned, &connected, radius);
            for i in 0..n {
                for k in 0..3 {
                    assert!(
                        (gpu[i][k] - cpu[i][k]).abs() < 1e-5,
                        "case {case} particle {i} axis {k}: gpu {:?} vs cpu {:?}",
                        gpu[i],
                        cpu[i]
                    );
                }
            }
        }
    }
}
