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
}
