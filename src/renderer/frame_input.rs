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
    pub bloom: BloomSettings,
    pub generative_background: GenerativeBackgroundSettings,
    /// Tracking-derived signals consumed by the generative background
    /// shader. `valid == false` (the default) zeroes every reactive term.
    pub background_tracking: BackgroundTracking,
    /// Background animation clock in seconds. The app accumulates wall time
    /// and wraps it at 4096 s to preserve f32 precision; validation binaries
    /// pass a fixed value so renders stay deterministic.
    pub time_seconds: f32,
}

/// Generative (procedural) background parameters, forwarded per frame.
/// Changing any field never rebuilds pipelines — everything reaches the
/// fullscreen background shader as push constants. While `enabled` is
/// false the background draw is skipped entirely and the scene keeps the
/// historical clear-color / transparent background.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerativeBackgroundSettings {
    pub enabled: bool,
    /// Linear-space brightness multiplier. Streak highlights exceed 1.0 so
    /// the bloom chain picks them up at its default threshold.
    pub intensity: f32,
    pub speed: f32,
    /// Noise/domain scale of the flow field.
    pub scale: f32,
    /// Strength of the tracking-reactive terms (hand swirl, head wave,
    /// mouth pulse). 0.0 turns the background into a pure flow field.
    pub reactivity: f32,
    pub color_a: [f32; 3],
    pub color_b: [f32; 3],
}

impl Default for GenerativeBackgroundSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 1.0,
            speed: 1.0,
            scale: 2.0,
            reactivity: 1.0,
            color_a: [0.02, 0.05, 0.18],
            color_b: [0.10, 0.85, 1.00],
        }
    }
}

/// Avatar-derived world-space anchors for the reactive background. Bone
/// positions live in avatar-root space (the same space the renderer draws
/// vertices in — `RenderAvatarInstance::world_transform` is never applied),
/// so the renderer projects them with the frame's view/projection as-is.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BackgroundTracking {
    pub head_ws: [f32; 3],
    pub left_hand_ws: [f32; 3],
    pub right_hand_ws: [f32; 3],
    /// Mouth-open amount in `[0, 1]` — the max of the five VRM mouth
    /// viseme expression weights ("aa", "ih", "ou", "ee", "oh").
    pub mouth_open: f32,
    /// False when no avatar (or no humanoid map) is loaded; the renderer
    /// then drops every reactive term.
    pub valid: bool,
}

/// Bloom post-effect parameters, forwarded per frame. Changing any field
/// never rebuilds pipelines — the renderer feeds them to the bloom passes as
/// push constants and skips the down/upsample chain entirely while
/// `enabled` is false.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BloomSettings {
    pub enabled: bool,
    /// Linear-space multiplier applied to the blurred bloom contribution at
    /// composite time (both RGB and alpha, so the glow halo carries
    /// fractional alpha over a transparent background).
    pub intensity: f32,
    /// Soft-knee luminance threshold in linear space. 1.0 means only
    /// HDR-bright pixels (emissive, strong lights) feed the bloom chain.
    pub threshold: f32,
}

impl Default for BloomSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 0.6,
            threshold: 1.0,
        }
    }
}

/// Request for the body-surface distance field splat on this instance.
/// The renderer voxel-splats the skinned primitive into a grid covering
/// the avatar's rest AABB (expanded by the collision shell) every frame
/// and ships the field back via `RenderResult::sdf_fields`, where the
/// spring solver picks it up — see `simulation/sdf.rs`.
#[derive(Clone, Debug)]
pub struct BodySdfPlan {
    /// Primitives to splat: the body surface plus any face/head
    /// surface (often a separate primitive; hair bangs and twintails
    /// collide against it). Each splats into the shared field through
    /// its own dispatch.
    pub prims: Vec<(crate::asset::MeshId, crate::asset::PrimitiveId)>,
    pub grid: crate::simulation::sdf::SdfGrid,
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
    /// `Some` when the instance's spring chains want body collision —
    /// the renderer splats the named primitive into `grid`'s field as
    /// part of the compute prepass. `None` when the asset has no body
    /// primitive or the spring solver is toggled off.
    pub body_sdf: Option<BodySdfPlan>,
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

impl RenderMeshInstance {
    /// Standard construction from an asset primitive: the shared material
    /// resolution chain (primitive's material → avatar's first material →
    /// built-in default), the `alpha_mode` / `cull_mode` mapping from the
    /// resolved material, a material-driven outline snapshot, and morph
    /// weights from the avatar's current expression weights
    /// ([`crate::avatar::AvatarInstance::morph_weights_for_prim`]).
    ///
    /// This is the ONE authoritative mapping — the live frame-input
    /// builder, thumbnail snapshots, and the offline / diagnostic
    /// renderers all start here and then override the pieces they differ
    /// on (shading mode, debug view, disabled outline for depth-sensitive
    /// benches) through field assignment.
    pub fn from_primitive(
        avatar: &crate::avatar::AvatarInstance,
        mesh_id: crate::asset::MeshId,
        prim: &std::sync::Arc<crate::asset::MeshPrimitiveAsset>,
    ) -> Self {
        let material_binding = avatar
            .asset
            .materials
            .iter()
            .find(|m| m.id == prim.material_id)
            .map(MaterialUploadRequest::from_asset_material)
            .or_else(|| {
                avatar
                    .asset
                    .materials
                    .first()
                    .map(|m| MaterialUploadRequest::from_asset_material(m))
            })
            .unwrap_or_else(MaterialUploadRequest::default_material);

        let alpha_mode = match material_binding.alpha_mode {
            crate::asset::AlphaMode::Opaque => RenderAlphaMode::Opaque,
            crate::asset::AlphaMode::Mask(_) => RenderAlphaMode::Cutout,
            crate::asset::AlphaMode::Blend => RenderAlphaMode::Blend,
        };
        let cull_mode = if material_binding.double_sided {
            RenderCullMode::DoubleSided
        } else {
            RenderCullMode::BackFace
        };
        let outline = OutlineSnapshot {
            enabled: material_binding.outline_width > 0.0,
            width: material_binding.outline_width,
            color: material_binding.outline_color,
        };

        Self {
            mesh_id,
            primitive_id: prim.id,
            material_binding,
            bounds: prim.bounds,
            alpha_mode,
            cull_mode,
            outline,
            primitive_data: Some(std::sync::Arc::clone(prim)),
            morph_weights: avatar.morph_weights_for_prim(prim),
        }
    }
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

/// Per-frame control data for the GPU cloth dispatch. Populated by
/// `collect_cloth_deforms` for GPU-backed cloths from the avatar's
/// `ClothSimState`.
#[derive(Clone, Debug)]
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
    /// Collision margin added to every capsule radius, mirroring
    /// `ClothSimState::collision_margin` on the CPU path.
    pub collision_margin: f32,
    /// Whether the GPU self-collision passes run (mirrors
    /// `ClothSimState::self_collision`; off by default, matching the
    /// CPU solver).
    pub self_collision: bool,
    /// Self-collision particle radius (m). Particles closer than
    /// `2 * radius` push apart; `ClothSimState::self_collision_radius`.
    pub self_collision_radius: f32,
    /// Body-SDF contact radius (m; 0 = stage off). Mirrors
    /// `ClothSimState::sdf_contact`; the collide kernel projects free
    /// particles onto the isosurface along the smooth SDF gradient.
    pub sdf_contact: f32,
    /// World-space collision capsules for THIS frame (avatar-node
    /// colliders resolved from the current pose; spheres encoded as
    /// degenerate capsules with `p0 == p1`). Scene colliders and
    /// self-collision remain CPU-solver-only.
    pub colliders: Vec<ClothGpuCollider>,
    /// Per-particle pin world targets for THIS frame — the GPU twin
    /// of the CPU solver's `apply_pin_targets` (`T(node) · offset`
    /// per pin binding, expanded to particle index space). The
    /// renderer writes pinned particles' `pos`/`prev_pos` SSBO rows
    /// to these targets before the substep dispatches so pinned
    /// cloth follows the avatar's bones. Entries for unpinned
    /// particles are zero and skipped via `gpu_attach.pinned`.
    /// Empty when the cloth has no pins.
    pub pin_positions: Vec<[f32; 3]>,
}

/// One world-space collision capsule for the GPU cloth stage. Spheres
/// are encoded as degenerate capsules (`p0 == p1`); the closest-point-
/// on-segment math handles that case identically to a sphere.
#[derive(Clone, Copy, Debug)]
pub struct ClothGpuCollider {
    /// First segment endpoint.
    pub p0: [f32; 3],
    /// Second segment endpoint (equal to `p0` for spheres).
    pub p1: [f32; 3],
    /// Capsule radius, before the per-cloth collision margin.
    pub radius: f32,
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
    /// Edge-angle bend constraints (T09 model, rest angles
    /// precomputed). Empty when the garment has none — the renderer's
    /// bend kernels stay unallocated and undispached.
    pub bend: Vec<crate::renderer::pipeline::ClothBendGpu>,
    /// Bend-wing CSR over particles: `bend_adj_offsets[v] ..
    /// [v+1]` indexes into `bend_adj_constraints` (constraint rows
    /// where `v` is a wing). The hinge never appears — it is never
    /// moved.
    pub bend_adj_offsets: Vec<u32>,
    pub bend_adj_constraints: Vec<u32>,
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
    use super::{BloomSettings, GenerativeBackgroundSettings, MsaaMode};

    #[test]
    fn generative_background_default_is_disabled() {
        // Existing projects (and every construction site that fills the
        // field with `Default::default()`) must keep the historical
        // clear-color background.
        let bg = GenerativeBackgroundSettings::default();
        assert!(!bg.enabled);
    }

    #[test]
    fn bloom_settings_default_is_disabled() {
        // Existing projects (and every diagnostic binary that fills the
        // field with `Default::default()`) must keep the historical
        // no-post-processing output.
        let bloom = BloomSettings::default();
        assert!(!bloom.enabled);
    }

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
