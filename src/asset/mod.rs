#![allow(dead_code)]
pub mod vrm;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

pub type Vec3 = [f32; 3];
pub type Vec4 = [f32; 4];
pub type Quat = [f32; 4];
pub type Mat4 = [[f32; 4]; 4];
pub type UVec2 = [u32; 2];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AvatarAssetId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MeshId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrimitiveId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MaterialId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClothOverlayId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ColliderId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AnimationClipId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AttachmentClassId(pub u64);

// ---------------------------------------------------------------------------
// Animation clip types
// ---------------------------------------------------------------------------

/// Which transform property a channel targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimationProperty {
    Translation,
    Rotation,
    Scale,
}

/// Interpolation mode between keyframes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolationMode {
    Step,
    Linear,
    CubicSpline,
}

/// A single keyframe within an animation channel.
#[derive(Clone, Debug)]
pub struct Keyframe {
    /// Time in seconds from the start of the clip.
    pub time: f32,
    /// The keyframe value.
    /// - Translation / Scale: `[x, y, z, 0.0]`
    /// - Rotation (quaternion): `[x, y, z, w]`
    pub value: Vec4,
    /// Tangent data used only for `CubicSpline` interpolation.
    /// `[in_tangent, out_tangent]` each stored as `Vec4`.
    pub tangents: Option<[Vec4; 2]>,
    pub interpolation: InterpolationMode,
}

/// One channel targets a single property of a single node.
#[derive(Clone, Debug)]
pub struct AnimationChannel {
    pub target_node: NodeId,
    pub property: AnimationProperty,
    pub keyframes: Vec<Keyframe>,
}

/// An animation clip containing one or more channels.
#[derive(Clone, Debug)]
pub struct AnimationClip {
    pub id: AnimationClipId,
    pub name: String,
    /// Total duration in seconds (max keyframe time across all channels).
    pub duration: f32,
    pub channels: Vec<AnimationChannel>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClothRegionTag(pub u64);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssetSourceHash(pub [u8; 32]);

#[derive(Clone, Debug)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

impl Transform {
    /// Build a 4x4 TRS matrix in **column-major** layout (`m[col][row]`).
    ///
    /// Translation occupies column 3 (indices `[3][0..3]`). GPU shaders
    /// expect column-major data, so this can be uploaded directly.
    pub fn to_matrix(&self) -> Mat4 {
        let [x, y, z, w] = self.rotation;
        let [sx, sy, sz] = self.scale;

        let x2 = x + x;
        let y2 = y + y;
        let z2 = z + z;
        let xx = x * x2;
        let xy = x * y2;
        let xz = x * z2;
        let yy = y * y2;
        let yz = y * z2;
        let zz = z * z2;
        let wx = w * x2;
        let wy = w * y2;
        let wz = w * z2;

        [
            [(1.0 - (yy + zz)) * sx, (xy + wz) * sx, (xz - wy) * sx, 0.0],
            [(xy - wz) * sy, (1.0 - (xx + zz)) * sy, (yz + wx) * sy, 0.0],
            [(xz + wy) * sz, (yz - wx) * sz, (1.0 - (xx + yy)) * sz, 0.0],
            [
                self.translation[0],
                self.translation[1],
                self.translation[2],
                1.0,
            ],
        ]
    }
}

#[derive(Clone, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

#[derive(Clone, Debug)]
pub struct AvatarAsset {
    pub id: AvatarAssetId,
    pub source_path: PathBuf,
    pub source_hash: AssetSourceHash,
    pub skeleton: SkeletonAsset,
    pub meshes: Vec<MeshAsset>,
    pub materials: Vec<MaterialAsset>,
    pub humanoid: Option<HumanoidMap>,
    pub spring_bones: Vec<SpringBoneAsset>,
    pub colliders: Vec<ColliderAsset>,
    pub default_expressions: ExpressionAssetSet,
    pub animation_clips: Vec<AnimationClip>,
}

#[derive(Clone, Debug)]
pub struct SkeletonAsset {
    pub nodes: Vec<SkeletonNode>,
    pub root_nodes: Vec<NodeId>,
    pub inverse_bind_matrices: Vec<Mat4>,
}

#[derive(Clone, Debug)]
pub struct SkeletonNode {
    pub id: NodeId,
    pub name: String,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    pub rest_local: Transform,
    pub humanoid_bone: Option<HumanoidBone>,
}

#[derive(Clone, Debug)]
pub struct MeshAsset {
    pub id: MeshId,
    pub name: String,
    pub primitives: Vec<MeshPrimitiveAsset>,
}

#[derive(Clone, Debug)]
pub struct VertexData {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<[f32; 2]>,
    pub joint_indices: Vec<[u16; 4]>,
    pub joint_weights: Vec<Vec4>,
}

#[derive(Clone, Debug)]
pub struct MeshPrimitiveAsset {
    pub id: PrimitiveId,
    pub vertex_count: u32,
    pub index_count: u32,
    pub material_id: MaterialId,
    pub skin: Option<SkinBinding>,
    pub bounds: Aabb,
    pub vertices: Option<VertexData>,
    pub indices: Option<Vec<u32>>,
}

#[derive(Clone, Debug)]
pub struct SkinBinding {
    pub joint_nodes: Vec<NodeId>,
    pub inverse_bind_matrices: Vec<Mat4>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AlphaMode {
    Opaque,
    Mask(f32),
    Blend,
}

#[derive(Clone, Debug)]
pub struct MaterialAsset {
    pub id: MaterialId,
    pub name: String,
    pub base_mode: MaterialMode,
    pub base_color: Vec4,
    pub alpha_mode: AlphaMode,
    pub double_sided: bool,
    pub texture_bindings: MaterialTextureSet,
    pub toon_params: ToonMaterialParams,
    pub mtoon_params: Option<MtoonStagedParams>,
}

#[derive(Clone, Debug)]
pub enum MaterialMode {
    Unlit,
    SimpleLit,
    ToonLike,
}

#[derive(Clone, Debug)]
pub struct MaterialTextureSet {
    pub base_color_texture: Option<TextureBinding>,
    pub normal_map_texture: Option<TextureBinding>,
    pub shade_ramp_texture: Option<TextureBinding>,
    pub emissive_texture: Option<TextureBinding>,
    pub matcap_texture: Option<TextureBinding>,
}

#[derive(Clone, Debug)]
pub struct TextureBinding {
    pub uri: String,
    pub pixel_data: Option<Arc<Vec<u8>>>,
    pub dimensions: (u32, u32),
}

#[derive(Clone, Debug)]
pub struct ToonMaterialParams {
    pub ramp_threshold: f32,
    pub shadow_softness: f32,
    pub outline_width: f32,
    pub outline_color: Vec3,
}

#[derive(Clone, Debug)]
pub struct HumanoidMap {
    pub bone_map: std::collections::HashMap<HumanoidBone, NodeId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum HumanoidBone {
    Hips,
    Spine,
    Chest,
    UpperChest,
    Neck,
    Head,
    LeftShoulder,
    LeftUpperArm,
    LeftLowerArm,
    LeftHand,
    RightShoulder,
    RightUpperArm,
    RightLowerArm,
    RightHand,
    LeftUpperLeg,
    LeftLowerLeg,
    LeftFoot,
    RightUpperLeg,
    RightLowerLeg,
    RightFoot,
}

#[derive(Clone, Debug)]
pub struct ExpressionAssetSet {
    pub expressions: Vec<ExpressionDef>,
}

#[derive(Clone, Debug)]
pub struct ExpressionDef {
    pub name: String,
    pub weight: f32,
}

#[derive(Clone, Debug)]
pub struct SpringBoneAsset {
    pub chain_root: NodeId,
    pub joints: Vec<NodeId>,
    pub stiffness: f32,
    pub drag_force: f32,
    pub gravity_dir: Vec3,
    pub gravity_power: f32,
    pub radius: f32,
    pub collider_refs: Vec<ColliderRef>,
    /// Per-joint stiffness overrides. When non-empty, index corresponds to `joints`.
    pub joint_stiffness: Vec<f32>,
    /// Per-joint drag overrides. When non-empty, index corresponds to `joints`.
    pub joint_drag: Vec<f32>,
    /// Per-joint gravity power overrides. When non-empty, index corresponds to `joints`.
    pub joint_gravity_power: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct ColliderRef {
    pub id: ColliderId,
}

#[derive(Clone, Debug)]
pub struct ColliderAsset {
    pub id: ColliderId,
    pub node: NodeId,
    pub shape: ColliderShape,
    pub offset: Vec3,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ColliderShape {
    Sphere { radius: f32 },
    Capsule { radius: f32, height: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SceneColliderId(pub u64);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneColliderAsset {
    pub id: SceneColliderId,
    pub position: Vec3,
    pub shape: ColliderShape,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClothAsset {
    pub id: ClothOverlayId,
    pub target_avatar: AvatarAssetId,
    pub target_avatar_hash: AssetSourceHash,
    pub stable_refs: ClothStableRefSet,
    pub simulation_mesh: ClothSimulationMesh,
    pub render_bindings: Vec<ClothRenderRegionBinding>,
    pub mesh_mapping: ClothMeshMapping,
    pub pins: Vec<ClothPin>,
    pub constraints: ClothConstraintSet,
    pub collision_bindings: Vec<ClothCollisionBinding>,
    pub lods: Vec<ClothLod>,
    pub solver_params: ClothSolverParams,
    pub metadata: ClothOverlayMetadata,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClothStableRefSet {
    pub node_refs: Vec<NodeRef>,
    pub mesh_refs: Vec<MeshRef>,
    pub primitive_refs: Vec<PrimitiveRef>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeRef {
    pub id: NodeId,
    pub name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshRef {
    pub id: MeshId,
    pub name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrimitiveRef {
    pub id: PrimitiveId,
    pub name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClothSimulationMesh {
    pub vertices: Vec<ClothSimVertex>,
    pub indices: Vec<u32>,
    pub rest_lengths: Vec<f32>,
    pub attachment_classes: Vec<AttachmentClassId>,
    pub region_tags: Vec<ClothRegionTag>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClothSimVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: [f32; 2],
    pub pinned: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClothRenderRegionBinding {
    pub primitive: PrimitiveRef,
    pub vertex_subset: VertexSubsetRef,
    pub mapping_region: ClothRegionTag,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VertexSubsetRef {
    pub offset: u32,
    pub count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClothMeshMapping {
    pub mapping_mode: ClothMappingMode,
    pub entries: Vec<ClothMappingEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ClothMappingMode {
    Barycentric,
    Nearest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClothMappingEntry {
    pub sim_vertex: u32,
    pub render_vertex: u32,
    pub weight: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClothPin {
    pub sim_vertex_indices: Vec<u32>,
    pub binding_node: NodeRef,
    pub offset: Vec3,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClothConstraintSet {
    pub distance_constraints: Vec<DistanceConstraint>,
    pub bend_constraints: Vec<BendConstraint>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DistanceConstraint {
    pub indices: [u32; 2],
    pub rest_length: f32,
    pub stiffness: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BendConstraint {
    pub indices: [u32; 3],
    pub stiffness: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClothCollisionBinding {
    pub proxy_shape: ColliderShape,
    pub binding_node: NodeRef,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClothLod {
    pub distance_threshold: f32,
    pub active_region: ClothRegionTag,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClothSolverParams {
    pub substeps: u32,
    pub iterations: u32,
    pub gravity_scale: f32,
    pub damping: f32,
    pub self_collision: bool,
    pub collision_margin: f32,
    pub wind_response: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClothOverlayMetadata {
    pub name: String,
    pub format_version: u32,
    pub created_with: String,
    pub last_saved_with: String,
}

impl ClothAsset {
    /// Create a minimal valid empty `ClothAsset` targeting the given avatar.
    pub fn new_empty(
        id: ClothOverlayId,
        avatar_id: AvatarAssetId,
        avatar_hash: AssetSourceHash,
    ) -> Self {
        Self {
            id,
            target_avatar: avatar_id,
            target_avatar_hash: avatar_hash,
            stable_refs: ClothStableRefSet {
                node_refs: vec![],
                mesh_refs: vec![],
                primitive_refs: vec![],
            },
            simulation_mesh: ClothSimulationMesh {
                vertices: vec![],
                indices: vec![],
                rest_lengths: vec![],
                attachment_classes: vec![],
                region_tags: vec![],
            },
            render_bindings: vec![],
            mesh_mapping: ClothMeshMapping {
                mapping_mode: ClothMappingMode::Barycentric,
                entries: vec![],
            },
            pins: vec![],
            constraints: ClothConstraintSet {
                distance_constraints: vec![],
                bend_constraints: vec![],
            },
            collision_bindings: vec![],
            lods: vec![],
            solver_params: ClothSolverParams {
                substeps: 1,
                iterations: 4,
                gravity_scale: 1.0,
                damping: 0.99,
                self_collision: false,
                collision_margin: 0.01,
                wind_response: 0.5,
            },
            metadata: ClothOverlayMetadata {
                name: "New Overlay".to_string(),
                format_version: 1,
                created_with: format!("VulVATAR {}", env!("CARGO_PKG_VERSION")),
                last_saved_with: format!("VulVATAR {}", env!("CARGO_PKG_VERSION")),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// MToon staged parameters (pure data, no GPU types)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct MtoonStagedParams {
    pub shade_color: Vec4,
    pub shade_shift: f32,
    pub shade_toony: f32,
    pub lit_color: Vec4,
    pub gi_equalization: f32,
    pub matcap_texture: Option<MtoonTextureSlot>,
    pub rim_texture: Option<MtoonTextureSlot>,
    pub rim_color: Vec4,
    pub rim_lighting_mix: f32,
    pub rim_fresnel_power: f32,
    pub rim_lift: f32,
    pub emissive_texture: Option<MtoonTextureSlot>,
    pub emissive_color: Vec4,
    pub outline_width_mode: MtoonOutlineWidthMode,
    pub outline_color: Vec3,
    pub outline_width: f32,
    pub uv_anim_mask_texture: Option<MtoonTextureSlot>,
    pub uv_anim_scroll_x_speed: f32,
    pub uv_anim_scroll_y_speed: f32,
    pub uv_anim_rotation_speed: f32,
}

#[derive(Clone, Debug)]
pub struct MtoonTextureSlot {
    pub uri: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MtoonOutlineWidthMode {
    None,
    WorldCoordinates,
    ScreenCoordinates,
}

impl Default for MtoonOutlineWidthMode {
    fn default() -> Self {
        Self::None
    }
}

pub fn identity_matrix() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}
