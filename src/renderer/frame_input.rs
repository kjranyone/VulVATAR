#![allow(dead_code)]
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
}

#[derive(Clone, Debug)]
pub struct RenderAvatarInstance {
    pub instance_id: AvatarInstanceId,
    pub world_transform: crate::asset::Transform,
    pub mesh_instances: Vec<RenderMeshInstance>,
    pub skinning_matrices: Vec<Mat4>,
    pub cloth_deform: Option<ClothDeformSnapshot>,
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
}

#[derive(Clone, Debug)]
pub struct ClothDeformSnapshot {
    pub deformed_positions: Vec<crate::asset::Vec3>,
    pub deformed_normals: Option<Vec<crate::asset::Vec3>>,
    pub version: u64,
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
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RenderColorSpace {
    Srgb,
    LinearSrgb,
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
