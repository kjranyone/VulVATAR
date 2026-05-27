use crate::asset::Mat4;
use crate::avatar::AvatarInstanceId;
use crate::renderer::material::MaterialUploadRequest;

#[derive(Clone, Debug)]
pub struct RenderFrameInput {
    pub camera: CameraState,
    pub lighting: LightingState,
    pub instances: Vec<RenderAvatarInstance>,
    pub output_request: OutputTargetRequest,
    pub background_image_path: Option<std::path::PathBuf>,
    pub show_ground_grid: bool,
    /// Solid clear color for the main render pass when not transparent.
    /// `(r, g, b, 1.0)` is written into the colour attachment's clear value.
    /// Ignored when [`Self::transparent_background`] is true (the render
    /// target is cleared to `(0,0,0,0)` instead).
    pub background_color: [f32; 3],
    pub transparent_background: bool,
    /// Global avatar opacity multiplier (1.0 = fully opaque). Drives the
    /// fade-out-when-no-person-detected feature; the renderer folds it into
    /// the per-frame camera uniform and multiplies it into every avatar
    /// fragment's alpha.
    pub avatar_opacity: f32,
}

#[derive(Clone, Debug)]
pub struct RenderAvatarInstance {
    pub instance_id: AvatarInstanceId,
    pub world_transform: crate::asset::Transform,
    pub mesh_instances: Vec<RenderMeshInstance>,
    pub skinning_matrices: Vec<Mat4>,
    /// Per-frame cloth snapshots, scoped per (target primitive, vertex
    /// subset). Each entry applies its `deformed_positions` to the
    /// `[vertex_offset .. vertex_offset + vertex_count)` range of the
    /// primitive identified by `target_primitive_id`. Empty when the
    /// avatar has no cloth or the cloth has no render binding yet.
    pub cloth_deforms: Vec<ClothDeformSnapshot>,
    pub debug_flags: RenderDebugFlags,
}

#[derive(Clone, Debug, Default)]
pub struct RenderDebugFlags {
    pub show_skeleton: bool,
    pub show_colliders: bool,
    pub show_cloth_mesh: bool,
    pub show_normals: bool,
    pub material_mode_override: Option<crate::renderer::material::MaterialShaderMode>,
}

#[derive(Clone, Debug)]
pub struct RenderMeshInstance {
    pub mesh_id: crate::asset::MeshId,
    pub primitive_id: crate::asset::PrimitiveId,
    pub material_binding: MaterialUploadRequest,
    pub bounds: crate::asset::Aabb,
    pub alpha_mode: RenderAlphaMode,
    pub cull_mode: RenderCullMode,
    pub outline: OutlineSnapshot,
    /// The actual mesh primitive data for GPU upload. When `None`, a placeholder
    /// triangle is drawn instead.
    pub primitive_data: Option<std::sync::Arc<crate::asset::MeshPrimitiveAsset>>,
    /// Per-morph-target weights for this primitive. Indices correspond to
    /// `MeshPrimitiveAsset::morph_targets`. Empty when no morph targets are active.
    pub morph_weights: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct ClothDeformSnapshot {
    /// Primitive the cloth applies to. Globally unique within an avatar,
    /// so this is enough on its own for the renderer to match snapshots
    /// against `RenderMeshInstance::primitive_id`.
    pub target_primitive_id: crate::asset::PrimitiveId,
    /// Mesh that owns the target primitive. Cosmetic/diagnostic — the
    /// renderer never reads it. `None` for legacy snapshots whose
    /// `ClothRenderRegionBinding` was authored before the mesh field
    /// landed.
    pub target_mesh_id: Option<crate::asset::MeshId>,
    /// Vertex range inside the target primitive that this snapshot
    /// covers, copied from `ClothRenderRegionBinding::vertex_subset`.
    /// The renderer writes `deformed_positions[i]` into the primitive's
    /// cloth SSBO at index `vertex_offset + i` and leaves vertices
    /// outside `[vertex_offset .. vertex_offset + vertex_count)` at
    /// their rest pose.
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub deformed_positions: Vec<crate::asset::Vec3>,
    pub deformed_normals: Option<Vec<crate::asset::Vec3>>,
    pub version: u64,
    /// Which solver produced this snapshot. For `Cpu`, the renderer
    /// copies `deformed_positions` / `deformed_normals` into the
    /// per-primitive cloth SSBO. For `Gpu`, the renderer skips the
    /// CPU snapshot copy and instead dispatches the cloth compute
    /// pipelines, which write the SSBOs in place.
    pub solver_backend: crate::simulation::cloth_gpu_boundary::ClothSolverBackend,
    /// Per-frame control data consumed by the GPU cloth compute
    /// pipelines. `Some` when `solver_backend == Gpu`; `None`
    /// otherwise. `deformed_positions` doubles as the first-frame
    /// initialiser for `cloth_pos_ssbo` / `prev_pos_ssbo`, so this
    /// struct only carries the simulation parameters.
    pub gpu_control: Option<ClothGpuDispatchControl>,
    /// One-shot attach data for the GPU cloth solver — constraints,
    /// triangle indices, masses, pinned flags. Carried on every frame
    /// `solver_backend == Gpu` (cloning is cheap relative to the
    /// dispatch cost); the renderer only reads it when lazily
    /// allocating `ClothGpuSlot` on the first frame.
    pub gpu_attach: Option<ClothGpuAttachData>,
}

/// Per-frame control data uploaded to `cloth_verlet_cs`'s `Control`
/// UBO. Populated by `collect_cloth_deforms` for GPU-backed cloths
/// from the avatar's `ClothSimState`.
#[derive(Clone, Copy, Debug)]
pub struct ClothGpuDispatchControl {
    /// Substep dt. The renderer dispatches the verlet + constraint
    /// pipeline `substeps` times per frame with this dt, matching the
    /// CPU path's `for _ in 0..substeps { step_cloth(fixed_dt) }`
    /// loop. Was previously frame_dt (multiple substeps' worth of
    /// time integrated in one shot) which made gravity·dt² roughly
    /// `substeps²` too large and `α̃ = α/dt²` roughly `substeps²`
    /// too small — CPU and GPU ran qualitatively different physics.
    pub dt: f32,
    /// Number of verlet + constraint dispatch substeps to run this
    /// frame. Comes from `SimClock::advance(frame_dt)`.
    pub substeps: u32,
    pub damping: f32,
    pub gravity: [f32; 3],
    /// `wind_direction * wind_response` baked into a single vector.
    pub wind_force: [f32; 3],
    /// XPBD constraint iteration count per substep. Mirror of
    /// `ClothSimState::solver_iterations`; the renderer runs the
    /// lambda-update + accumulate + apply compute passes this many
    /// times *within each substep*. Normal recomputation runs once
    /// per frame, after all substeps complete.
    pub solver_iterations: u32,
}

/// One-shot data uploaded to the GPU at cloth-attach time: constraint
/// table, triangle indices, per-particle inverse mass and pinned flag.
/// Carried on `ClothDeformSnapshot.gpu_attach` so the renderer's lazy
/// slot allocation can seed the static SSBOs.
#[derive(Clone, Debug)]
pub struct ClothGpuAttachData {
    /// `(particle_a, particle_b, rest_length, stiffness)` tuples. Built
    /// from `ClothSimState::distance_constraints`; uploaded once into
    /// the constraint SSBO.
    pub constraints: Vec<(u32, u32, f32, f32)>,
    /// Flat triangle index buffer (3 per triangle). Built from
    /// `ClothSimState::triangle_indices` (cast u32). Uploaded once
    /// into the triangle-index SSBO.
    pub triangle_indices: Vec<u32>,
    /// Per-particle inverse mass. `0.0` marks a pinned / immobile
    /// particle. Written into the `w` component of `cloth_pos_ssbo`
    /// at allocation time.
    pub inv_masses: Vec<f32>,
    /// Per-particle pinned flag (`true` ⇒ shader's `pinned > 0.5`
    /// branch wins, particle holds its previous position). Written
    /// into the `w` component of `prev_pos_ssbo`.
    pub pinned: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RenderAlphaMode {
    Opaque,
    Cutout,
    Blend,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RenderCullMode {
    BackFace,
    DoubleSided,
    FrontFace,
}

#[derive(Clone, Debug)]
pub struct OutlineSnapshot {
    pub enabled: bool,
    pub width: f32,
    pub color: crate::asset::Vec3,
}

impl Default for OutlineSnapshot {
    fn default() -> Self {
        Self {
            enabled: false,
            width: 0.0,
            color: [0.0, 0.0, 0.0],
        }
    }
}

#[derive(Clone, Debug)]
pub struct CameraState {
    pub view: Mat4,
    pub projection: Mat4,
    pub position_ws: crate::asset::Vec3,
    pub viewport_extent: crate::asset::UVec2,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            view: crate::asset::identity_matrix(),
            projection: crate::asset::identity_matrix(),
            position_ws: [0.0, 0.0, -5.0],
            viewport_extent: [1920, 1080],
        }
    }
}

#[derive(Clone, Debug)]
pub struct LightingState {
    pub main_light_dir_ws: crate::asset::Vec3,
    pub main_light_color: crate::asset::Vec3,
    pub main_light_intensity: f32,
    pub ambient_term: crate::asset::Vec3,
}

impl Default for LightingState {
    fn default() -> Self {
        Self {
            main_light_dir_ws: [0.5, -1.0, 0.3],
            main_light_color: [1.0, 1.0, 1.0],
            main_light_intensity: 1.0,
            ambient_term: [0.2, 0.2, 0.2],
        }
    }
}

#[derive(Clone, Debug)]
pub struct OutputTargetRequest {
    pub preview_enabled: bool,
    pub output_enabled: bool,
    pub extent: crate::asset::UVec2,
    pub color_space: RenderColorSpace,
    pub alpha_mode: RenderOutputAlpha,
    pub export_mode: RenderExportMode,
    /// Requested multisample anti-aliasing level. The renderer clamps this
    /// to what the physical device's framebuffer sample counts support and
    /// rebuilds the render pass / pipelines / targets when it changes.
    pub msaa: MsaaMode,
}

impl Default for OutputTargetRequest {
    fn default() -> Self {
        Self {
            preview_enabled: true,
            output_enabled: false,
            extent: [1920, 1080],
            color_space: RenderColorSpace::Srgb,
            alpha_mode: RenderOutputAlpha::Premultiplied,
            export_mode: RenderExportMode::None,
            msaa: MsaaMode::Off,
        }
    }
}

/// Multisample anti-aliasing level for the offscreen render target.
///
/// `Off` keeps the historical single-sample path (no resolve attachment).
/// The other levels add an N-sample colour + depth target that the render
/// pass resolves into the single-sample readback / export image, so the
/// downstream readback (`copy_image_to_buffer`) and the MF virtual-camera
/// external-memory share both keep seeing a single-sample image regardless
/// of the MSAA level. The renderer clamps the requested level down to the
/// device's `framebuffer_color_sample_counts & framebuffer_depth_sample_counts`
/// support before building anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MsaaMode {
    #[default]
    Off,
    X2,
    X4,
    X8,
}

impl MsaaMode {
    /// Vulkan `rasterization_samples` / attachment sample count. `Off` maps
    /// to 1 (the no-resolve, no-MSAA path).
    pub fn sample_count(self) -> u32 {
        match self {
            MsaaMode::Off => 1,
            MsaaMode::X2 => 2,
            MsaaMode::X4 => 4,
            MsaaMode::X8 => 8,
        }
    }

    /// Map a GUI combo index (0=Off, 1=2x, 2=4x, 3=8x) to a mode.
    /// Out-of-range indices fall back to `Off`.
    pub fn from_index(index: usize) -> Self {
        match index {
            1 => MsaaMode::X2,
            2 => MsaaMode::X4,
            3 => MsaaMode::X8,
            _ => MsaaMode::Off,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RenderColorSpace {
    Srgb,
    LinearSrgb,
}

impl RenderColorSpace {
    /// Stable lowercase tag used in logs and in `ExportMetadata` for sinks
    /// that still want a string (e.g. JSON manifests). The matching MF media
    /// type attributes for the virtual camera path are picked up directly
    /// from the enum, not from this string.
    pub fn as_str(&self) -> &'static str {
        match self {
            RenderColorSpace::Srgb => "srgb",
            RenderColorSpace::LinearSrgb => "linear-srgb",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RenderOutputAlpha {
    Opaque,
    Premultiplied,
    Straight,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RenderExportMode {
    None,
    GpuExport,
    CpuReadback,
}

#[cfg(test)]
mod tests {
    use super::MsaaMode;

    #[test]
    fn msaa_mode_sample_count_maps_to_vulkan_levels() {
        assert_eq!(MsaaMode::Off.sample_count(), 1);
        assert_eq!(MsaaMode::X2.sample_count(), 2);
        assert_eq!(MsaaMode::X4.sample_count(), 4);
        assert_eq!(MsaaMode::X8.sample_count(), 8);
    }

    #[test]
    fn msaa_mode_from_index_matches_gui_combo_order() {
        // The GUI combo order (0=Off, 1=2x, 2=4x, 3=8x) is the persisted
        // contract; this guards against the mapping drifting from the
        // inspector's `selectable_value` indices.
        assert_eq!(MsaaMode::from_index(0), MsaaMode::Off);
        assert_eq!(MsaaMode::from_index(1), MsaaMode::X2);
        assert_eq!(MsaaMode::from_index(2), MsaaMode::X4);
        assert_eq!(MsaaMode::from_index(3), MsaaMode::X8);
    }

    #[test]
    fn msaa_mode_from_index_out_of_range_falls_back_to_off() {
        assert_eq!(MsaaMode::from_index(4), MsaaMode::Off);
        assert_eq!(MsaaMode::from_index(usize::MAX), MsaaMode::Off);
    }

    #[test]
    fn msaa_mode_default_is_off() {
        // Default must keep existing projects on the historical 1× path.
        assert_eq!(MsaaMode::default(), MsaaMode::Off);
        assert_eq!(MsaaMode::default().sample_count(), 1);
    }
}
