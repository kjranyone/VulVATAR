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

/// Placeholder for the future compute-cloth state: per-primitive particle
/// SSBOs, constraint index buffers, etc. Currently empty; the type exists so
/// the renderer's cloth-binding code can carry an
/// `Option<ClothGpuSimulationState>` slot ahead of the solver landing.
///
/// When the GPU solver lands (P3-02), this struct will own:
/// - particle position / velocity SSBO handles
/// - constraint index / weight SSBO handles
/// - the version counter that mirrors `ClothDeformOutput::version`
#[derive(Clone, Debug, Default)]
pub struct ClothGpuSimulationState {
    _placeholder: (),
}

impl ClothGpuSimulationState {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Which solver backend produces the [`ClothRenderConsumableDeform`]
/// snapshots for a given cloth. The compute path is intentionally not yet
/// reachable; `Cpu` is the only variant the avatar layer constructs today.
///
/// Keeping the variant named now means later code that selects the backend
/// (e.g. a `RuntimeGpuBudget` decision in P3-03) can switch on this enum
/// instead of growing a bool / feature flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClothSolverBackend {
    Cpu,
    Gpu,
}

impl Default for ClothSolverBackend {
    fn default() -> Self {
        Self::Cpu
    }
}
