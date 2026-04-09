pub mod vrm;

use std::path::PathBuf;

pub type Vec3 = [f32; 3];
pub type Vec4 = [f32; 4];
pub type Quat = [f32; 4];
pub type Mat4 = [[f32; 4]; 4];
pub type UVec2 = [u32; 2];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AvatarAssetId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrimitiveId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaterialId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClothOverlayId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColliderId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AnimationClipId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttachmentClassId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClothRegionTag(pub u64);

#[derive(Clone, Debug)]
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
pub struct MeshPrimitiveAsset {
    pub id: PrimitiveId,
    pub vertex_count: u32,
    pub index_count: u32,
    pub material_id: MaterialId,
    pub skin: Option<SkinBinding>,
    pub bounds: Aabb,
}

#[derive(Clone, Debug)]
pub struct SkinBinding {
    pub joint_nodes: Vec<NodeId>,
    pub inverse_bind_matrices: Vec<Mat4>,
}

#[derive(Clone, Debug)]
pub struct MaterialAsset {
    pub id: MaterialId,
    pub name: String,
    pub base_mode: MaterialMode,
    pub base_color: Vec4,
    pub texture_bindings: MaterialTextureSet,
    pub toon_params: ToonMaterialParams,
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
}

#[derive(Clone, Debug)]
pub struct TextureBinding {
    pub uri: String,
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

#[derive(Clone, Debug)]
pub enum ColliderShape {
    Sphere { radius: f32 },
    Capsule { radius: f32, height: f32 },
}

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
pub struct ClothStableRefSet {
    pub node_refs: Vec<NodeRef>,
    pub mesh_refs: Vec<MeshRef>,
    pub primitive_refs: Vec<PrimitiveRef>,
}

#[derive(Clone, Debug)]
pub struct NodeRef {
    pub id: NodeId,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct MeshRef {
    pub id: MeshId,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct PrimitiveRef {
    pub id: PrimitiveId,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct ClothSimulationMesh {
    pub vertices: Vec<ClothSimVertex>,
    pub indices: Vec<u32>,
    pub rest_lengths: Vec<f32>,
    pub attachment_classes: Vec<AttachmentClassId>,
    pub region_tags: Vec<ClothRegionTag>,
}

#[derive(Clone, Debug)]
pub struct ClothSimVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: [f32; 2],
    pub pinned: bool,
}

#[derive(Clone, Debug)]
pub struct ClothRenderRegionBinding {
    pub primitive: PrimitiveRef,
    pub vertex_subset: VertexSubsetRef,
    pub mapping_region: ClothRegionTag,
}

#[derive(Clone, Debug)]
pub struct VertexSubsetRef {
    pub offset: u32,
    pub count: u32,
}

#[derive(Clone, Debug)]
pub struct ClothMeshMapping {
    pub mapping_mode: ClothMappingMode,
    pub entries: Vec<ClothMappingEntry>,
}

#[derive(Clone, Debug)]
pub enum ClothMappingMode {
    Barycentric,
    Nearest,
}

#[derive(Clone, Debug)]
pub struct ClothMappingEntry {
    pub sim_vertex: u32,
    pub render_vertex: u32,
    pub weight: f32,
}

#[derive(Clone, Debug)]
pub struct ClothPin {
    pub sim_vertex_indices: Vec<u32>,
    pub binding_node: NodeRef,
    pub offset: Vec3,
}

#[derive(Clone, Debug)]
pub struct ClothConstraintSet {
    pub distance_constraints: Vec<DistanceConstraint>,
    pub bend_constraints: Vec<BendConstraint>,
}

#[derive(Clone, Debug)]
pub struct DistanceConstraint {
    pub indices: [u32; 2],
    pub rest_length: f32,
    pub stiffness: f32,
}

#[derive(Clone, Debug)]
pub struct BendConstraint {
    pub indices: [u32; 3],
    pub stiffness: f32,
}

#[derive(Clone, Debug)]
pub struct ClothCollisionBinding {
    pub proxy_shape: ColliderShape,
    pub binding_node: NodeRef,
}

#[derive(Clone, Debug)]
pub struct ClothLod {
    pub distance_threshold: f32,
    pub active_region: ClothRegionTag,
}

#[derive(Clone, Debug)]
pub struct ClothSolverParams {
    pub substeps: u32,
    pub iterations: u32,
    pub gravity_scale: f32,
    pub damping: f32,
    pub self_collision: bool,
    pub collision_margin: f32,
    pub wind_response: f32,
}

#[derive(Clone, Debug)]
pub struct ClothOverlayMetadata {
    pub name: String,
    pub format_version: u32,
    pub created_with: String,
    pub last_saved_with: String,
}

pub fn identity_matrix() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}
