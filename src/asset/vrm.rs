use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::asset::*;

mod gltf_ext {
    use serde::Deserialize;

    #[derive(Debug, Clone, Deserialize)]
    pub struct HumanoidBone {
        pub node: u32,
        #[serde(default)]
        pub use_default_value: Option<bool>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct Humanoid {
        pub human_bones: std::collections::HashMap<String, HumanoidBone>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct Expression {
        pub name: String,
        #[serde(default)]
        pub is_binary: Option<bool>,
        #[serde(default)]
        pub override_blink: Option<String>,
        #[serde(default)]
        pub override_look_at: Option<String>,
        #[serde(default)]
        pub override_mouth: Option<String>,
        #[serde(default)]
        pub binds: Option<Vec<ExpressionBind>>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct ExpressionBind {
        pub node: u32,
        pub property: String,
        pub value: f32,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct Expressions {
        #[serde(default)]
        pub preset: Option<Preset>,
        #[serde(default)]
        pub custom: Option<Vec<Expression>>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct Preset {
        #[serde(default)]
        pub happy: Option<Expression>,
        #[serde(default)]
        pub angry: Option<Expression>,
        #[serde(default)]
        pub sad: Option<Expression>,
        #[serde(default)]
        pub relaxed: Option<Expression>,
        #[serde(default)]
        pub surprised: Option<Expression>,
        #[serde(default)]
        pub aa: Option<Expression>,
        #[serde(default)]
        pub ih: Option<Expression>,
        #[serde(default)]
        pub ou: Option<Expression>,
        #[serde(default)]
        pub ee: Option<Expression>,
        #[serde(default)]
        pub oh: Option<Expression>,
        #[serde(default)]
        pub blink: Option<Expression>,
        #[serde(default)]
        pub blink_left: Option<Expression>,
        #[serde(default)]
        pub blink_right: Option<Expression>,
        #[serde(default)]
        pub look_up: Option<Expression>,
        #[serde(default)]
        pub look_down: Option<Expression>,
        #[serde(default)]
        pub look_left: Option<Expression>,
        #[serde(default)]
        pub look_right: Option<Expression>,
        #[serde(default)]
        pub neutral: Option<Expression>,
    }

    impl Preset {
        pub fn all_expressions(&self) -> Vec<&Expression> {
            let mut result = Vec::new();
            if let Some(ref e) = self.blink {
                result.push(e);
            }
            if let Some(ref e) = self.blink_left {
                result.push(e);
            }
            if let Some(ref e) = self.blink_right {
                result.push(e);
            }
            if let Some(ref e) = self.look_up {
                result.push(e);
            }
            if let Some(ref e) = self.look_down {
                result.push(e);
            }
            if let Some(ref e) = self.look_left {
                result.push(e);
            }
            if let Some(ref e) = self.look_right {
                result.push(e);
            }
            if let Some(ref e) = self.happy {
                result.push(e);
            }
            if let Some(ref e) = self.angry {
                result.push(e);
            }
            if let Some(ref e) = self.sad {
                result.push(e);
            }
            if let Some(ref e) = self.relaxed {
                result.push(e);
            }
            if let Some(ref e) = self.surprised {
                result.push(e);
            }
            if let Some(ref e) = self.aa {
                result.push(e);
            }
            if let Some(ref e) = self.ih {
                result.push(e);
            }
            if let Some(ref e) = self.ou {
                result.push(e);
            }
            if let Some(ref e) = self.ee {
                result.push(e);
            }
            if let Some(ref e) = self.oh {
                result.push(e);
            }
            if let Some(ref e) = self.neutral {
                result.push(e);
            }
            result
        }
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct VrmExtension {
        pub humanoid: Humanoid,
        #[serde(default)]
        pub expressions: Option<Expressions>,
    }

    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct ColliderShapeDef {
        #[serde(default)]
        pub sphere: Option<SphereShapeDef>,
        #[serde(default)]
        pub capsule: Option<CapsuleShapeDef>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct SpringBoneCollider {
        pub node: u32,
        #[serde(default)]
        pub shape: ColliderShapeDef,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct SphereShapeDef {
        #[serde(default)]
        pub radius: f32,
        #[serde(default)]
        pub offset: Option<[f32; 3]>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct CapsuleShapeDef {
        #[serde(default)]
        pub radius: f32,
        #[serde(default)]
        pub offset: Option<[f32; 3]>,
        #[serde(default)]
        pub tail: Option<[f32; 3]>,
    }

    fn default_stiffness() -> f32 {
        1.0
    }
    fn default_gravity_power() -> f32 {
        0.0
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct SpringBoneJoint {
        pub node: u32,
        #[serde(default)]
        pub hit_radius: Option<f32>,
        #[serde(default = "default_stiffness")]
        pub stiffness: f32,
        #[serde(default = "default_gravity_power")]
        pub gravity_power: f32,
        #[serde(default)]
        pub gravity_dir: Option<[f32; 3]>,
        #[serde(default)]
        pub drag_force: Option<f32>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct SpringBoneColliderGroup {
        pub colliders: Vec<u32>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct SpringBoneSpring {
        pub name: String,
        pub joints: Vec<SpringBoneJoint>,
        #[serde(default)]
        pub collider_groups: Option<Vec<u32>>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct SpringBoneExtension {
        #[serde(default)]
        pub springs: Vec<SpringBoneSpring>,
        #[serde(default)]
        pub collider_groups: Option<Vec<SpringBoneColliderGroup>>,
        #[serde(default)]
        pub colliders: Option<Vec<SpringBoneCollider>>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct RootExtension {
        #[serde(rename = "VRMC_vrm")]
        pub vrmc_vrm: Option<VrmExtension>,
        #[serde(rename = "VRMC_springBone")]
        pub vrmc_spring_bone: Option<SpringBoneExtension>,
    }
}

#[derive(Debug)]
pub enum VrmLoadError {
    Io(std::io::Error),
    Gltf(gltf::Error),
    MissingExtension,
    InvalidHumanoidBone(String),
    JsonParse(serde_json::Error),
}

impl std::fmt::Display for VrmLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Gltf(e) => write!(f, "glTF error: {}", e),
            Self::MissingExtension => write!(f, "missing VRMC_vrm extension"),
            Self::InvalidHumanoidBone(name) => write!(f, "invalid humanoid bone: {}", name),
            Self::JsonParse(e) => write!(f, "JSON parse error: {}", e),
        }
    }
}

impl std::error::Error for VrmLoadError {}

impl From<std::io::Error> for VrmLoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<gltf::Error> for VrmLoadError {
    fn from(e: gltf::Error) -> Self {
        Self::Gltf(e)
    }
}

impl From<serde_json::Error> for VrmLoadError {
    fn from(e: serde_json::Error) -> Self {
        Self::JsonParse(e)
    }
}

pub struct VrmAssetLoader;

impl VrmAssetLoader {
    pub fn new() -> Self {
        Self
    }

    pub fn load(&self, path: &str) -> Result<Arc<AvatarAsset>, VrmLoadError> {
        let file_data = std::fs::read(path)?;
        self.load_from_bytes(&file_data, PathBuf::from(path))
    }

    pub fn load_from_bytes(
        &self,
        data: &[u8],
        source_path: PathBuf,
    ) -> Result<Arc<AvatarAsset>, VrmLoadError> {
        let source_hash = compute_hash(data);

        let gltf_doc = gltf::Gltf::from_slice(data)?;
        let root_ext: Option<gltf_ext::RootExtension> = gltf_doc
            .document
            .extensions()
            .and_then(|ext| serde_json::from_value(serde_json::Value::Object(ext.clone())).ok());

        let vrm_ext = root_ext.as_ref().and_then(|r| r.vrmc_vrm.as_ref());
        let spring_ext = root_ext.as_ref().and_then(|r| r.vrmc_spring_bone.as_ref());

        let (nodes, node_name_map) = Self::build_skeleton_nodes(&gltf_doc.document);
        let humanoid = vrm_ext
            .map(|ext| Self::build_humanoid_map(ext, &node_name_map))
            .transpose()?;

        let meshes = Self::build_meshes(&gltf_doc.document);
        let materials = Self::build_materials(&gltf_doc.document);
        let skin_bindings = Self::build_skin_bindings(&gltf_doc.document);
        let meshes_with_skin = Self::assign_skins_to_meshes(meshes, skin_bindings);

        let root_nodes: Vec<NodeId> = gltf_doc
            .document
            .scenes()
            .next()
            .map(|scene| scene.nodes().map(|n| NodeId(n.index() as u64)).collect())
            .unwrap_or_default();

        let inv_bind_count = nodes.len();
        let inverse_bind_matrices = vec![identity_matrix(); inv_bind_count];

        let (spring_bones, colliders) = if let Some(spring_ext) = spring_ext {
            Self::build_spring_bones(spring_ext)
        } else {
            (vec![], vec![])
        };

        let expressions = vrm_ext
            .map(|ext| Self::build_expressions(ext))
            .unwrap_or_else(|| ExpressionAssetSet {
                expressions: vec![],
            });

        let asset = AvatarAsset {
            id: AvatarAssetId(1),
            source_path,
            source_hash,
            skeleton: SkeletonAsset {
                nodes,
                root_nodes,
                inverse_bind_matrices,
            },
            meshes: meshes_with_skin,
            materials,
            humanoid,
            spring_bones,
            colliders,
            default_expressions: expressions,
        };

        Ok(Arc::new(asset))
    }

    fn build_skeleton_nodes(doc: &gltf::Document) -> (Vec<SkeletonNode>, HashMap<u32, String>) {
        let gltf_nodes: Vec<gltf::Node<'_>> = doc.nodes().collect();
        let node_count = gltf_nodes.len();

        let mut nodes = Vec::with_capacity(node_count);
        let mut name_map = HashMap::new();

        for gnode in &gltf_nodes {
            let idx = gnode.index();
            let name = gnode.name().unwrap_or("").to_string();
            name_map.insert(idx as u32, name.clone());

            let (t, r, s) = gnode.transform().decomposed();
            let parent = Self::find_parent(&gltf_nodes, idx);
            let children: Vec<NodeId> =
                gnode.children().map(|c| NodeId(c.index() as u64)).collect();

            nodes.push(SkeletonNode {
                id: NodeId(idx as u64),
                name,
                parent,
                children,
                rest_local: Transform {
                    translation: t,
                    rotation: r,
                    scale: s,
                },
                humanoid_bone: None,
            });
        }

        (nodes, name_map)
    }

    fn find_parent(gltf_nodes: &[gltf::Node<'_>], target_idx: usize) -> Option<NodeId> {
        for gnode in gltf_nodes {
            for child in gnode.children() {
                if child.index() == target_idx {
                    return Some(NodeId(gnode.index() as u64));
                }
            }
        }
        None
    }

    fn build_humanoid_map(
        ext: &gltf_ext::VrmExtension,
        _node_name_map: &HashMap<u32, String>,
    ) -> Result<HumanoidMap, VrmLoadError> {
        let mut bone_map = HashMap::new();
        for (bone_name, bone_def) in &ext.humanoid.human_bones {
            let humanoid_bone = parse_humanoid_bone(bone_name)
                .ok_or_else(|| VrmLoadError::InvalidHumanoidBone(bone_name.to_string()))?;
            bone_map.insert(humanoid_bone, NodeId(bone_def.node as u64));
        }
        Ok(HumanoidMap { bone_map })
    }

    fn build_meshes(doc: &gltf::Document) -> Vec<MeshAsset> {
        let mut next_mesh_id: u64 = 1;
        let mut next_prim_id: u64 = 1;

        doc.meshes()
            .map(|gmesh| {
                let mesh_id = MeshId(next_mesh_id);
                next_mesh_id += 1;

                let primitives: Vec<MeshPrimitiveAsset> = gmesh
                    .primitives()
                    .map(|prim| {
                        let prim_id = PrimitiveId(next_prim_id);
                        next_prim_id += 1;

                        let reader = prim.reader(|_| None);
                        let vertex_count =
                            reader.read_positions().map(|p| p.count()).unwrap_or(0) as u32;
                        let index_count = prim.indices().map(|i| i.count()).unwrap_or(0) as u32;

                        let material_id = prim
                            .material()
                            .index()
                            .map(|i| MaterialId(i as u64 + 1))
                            .unwrap_or(MaterialId(0));

                        let bb = prim.bounding_box();
                        let bounds = Aabb {
                            min: bb.min,
                            max: bb.max,
                        };

                        MeshPrimitiveAsset {
                            id: prim_id,
                            vertex_count,
                            index_count,
                            material_id,
                            skin: None,
                            bounds,
                        }
                    })
                    .collect();

                MeshAsset {
                    id: mesh_id,
                    name: gmesh.name().unwrap_or("unnamed").to_string(),
                    primitives,
                }
            })
            .collect()
    }

    fn build_materials(doc: &gltf::Document) -> Vec<MaterialAsset> {
        doc.materials()
            .enumerate()
            .map(|(i, gmat)| {
                let pbr = gmat.pbr_metallic_roughness();
                let base_color = pbr.base_color_factor();

                let alpha_mode = gmat.alpha_mode();
                let base_mode = if matches!(alpha_mode, gltf::material::AlphaMode::Blend) {
                    MaterialMode::Unlit
                } else if pbr.metallic_factor() > 0.0 {
                    MaterialMode::SimpleLit
                } else {
                    MaterialMode::ToonLike
                };

                let base_color_texture = pbr.base_color_texture().map(|info| {
                    let uri = info
                        .texture()
                        .source()
                        .name()
                        .unwrap_or("unknown")
                        .to_string();
                    TextureBinding { uri }
                });

                let normal_map_texture = gmat.normal_texture().map(|info| {
                    let uri = info
                        .texture()
                        .source()
                        .name()
                        .unwrap_or("unknown")
                        .to_string();
                    TextureBinding { uri }
                });

                MaterialAsset {
                    id: MaterialId(i as u64 + 1),
                    name: gmat.name().unwrap_or("unnamed").to_string(),
                    base_mode,
                    base_color,
                    texture_bindings: MaterialTextureSet {
                        base_color_texture,
                        normal_map_texture,
                    },
                    toon_params: ToonMaterialParams {
                        ramp_threshold: 0.5,
                        shadow_softness: 0.1,
                        outline_width: 0.01,
                        outline_color: [0.0, 0.0, 0.0],
                    },
                }
            })
            .collect()
    }

    fn build_skin_bindings(doc: &gltf::Document) -> Vec<(usize, SkinBinding)> {
        doc.skins()
            .map(|skin| {
                let joint_nodes: Vec<NodeId> =
                    skin.joints().map(|j| NodeId(j.index() as u64)).collect();

                let reader = skin.reader(|_| None);
                let inverse_bind_matrices: Vec<Mat4> = reader
                    .read_inverse_bind_matrices()
                    .map(|iter| iter.map(|m| m).collect())
                    .unwrap_or_else(|| {
                        let node_count = joint_nodes.len();
                        vec![identity_matrix(); node_count]
                    });

                let skeleton_node_index = skin.skeleton().map(|n| n.index()).unwrap_or(0);

                (
                    skeleton_node_index,
                    SkinBinding {
                        joint_nodes,
                        inverse_bind_matrices,
                    },
                )
            })
            .collect()
    }

    fn assign_skins_to_meshes(
        meshes: Vec<MeshAsset>,
        skins: Vec<(usize, SkinBinding)>,
    ) -> Vec<MeshAsset> {
        let skin_by_skeleton: HashMap<usize, &SkinBinding> =
            skins.iter().map(|(idx, binding)| (*idx, binding)).collect();

        meshes
            .into_iter()
            .map(|mut mesh| {
                for prim in &mut mesh.primitives {
                    if let Some(skin) = skin_by_skeleton.get(&0) {
                        prim.skin = Some((*skin).clone());
                    } else if let Some((_, skin)) = skins.first() {
                        prim.skin = Some(skin.clone());
                    }
                }
                mesh
            })
            .collect()
    }

    fn build_spring_bones(
        ext: &gltf_ext::SpringBoneExtension,
    ) -> (Vec<SpringBoneAsset>, Vec<ColliderAsset>) {
        let mut collider_id: u64 = 1;
        let colliders: Vec<ColliderAsset> = ext
            .colliders
            .as_ref()
            .map(|collider_defs| {
                collider_defs
                    .iter()
                    .map(|c| {
                        let (shape, offset) = if let Some(ref sphere) = c.shape.sphere {
                            (
                                ColliderShape::Sphere {
                                    radius: sphere.radius,
                                },
                                sphere.offset.unwrap_or([0.0; 3]),
                            )
                        } else if let Some(ref capsule) = c.shape.capsule {
                            (
                                ColliderShape::Capsule {
                                    radius: capsule.radius,
                                    height: 0.0,
                                },
                                capsule.offset.unwrap_or([0.0; 3]),
                            )
                        } else {
                            (ColliderShape::Sphere { radius: 0.0 }, [0.0; 3])
                        };

                        let id = ColliderId(collider_id);
                        collider_id += 1;
                        ColliderAsset {
                            id,
                            node: NodeId(c.node as u64),
                            shape,
                            offset,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        let collider_id_map: HashMap<u32, ColliderId> = ext
            .colliders
            .as_ref()
            .map(|defs| {
                defs.iter()
                    .enumerate()
                    .map(|(i, _)| (i as u32, ColliderId(i as u64 + 1)))
                    .collect()
            })
            .unwrap_or_default();

        let collider_group_map: HashMap<u32, Vec<ColliderId>> = ext
            .collider_groups
            .as_ref()
            .map(|groups| {
                groups
                    .iter()
                    .enumerate()
                    .map(|(gi, group)| {
                        let ids: Vec<ColliderId> = group
                            .colliders
                            .iter()
                            .filter_map(|ci| collider_id_map.get(ci).copied())
                            .collect();
                        (gi as u32, ids)
                    })
                    .collect()
            })
            .unwrap_or_default();

        let spring_bones: Vec<SpringBoneAsset> = ext
            .springs
            .iter()
            .map(|spring| {
                let chain_root = spring
                    .joints
                    .first()
                    .map(|j| NodeId(j.node as u64))
                    .unwrap_or(NodeId(0));

                let joints: Vec<NodeId> = spring
                    .joints
                    .iter()
                    .map(|j| NodeId(j.node as u64))
                    .collect();

                let collider_refs: Vec<ColliderRef> = spring
                    .collider_groups
                    .as_ref()
                    .map(|groups| {
                        groups
                            .iter()
                            .filter_map(|gi| {
                                collider_group_map
                                    .get(gi)
                                    .map(|ids| ids.iter().map(|&id| ColliderRef { id }))
                            })
                            .flatten()
                            .collect()
                    })
                    .unwrap_or_default();

                let stiffness = spring.joints.first().map(|j| j.stiffness).unwrap_or(1.0);
                let drag_force = spring
                    .joints
                    .first()
                    .and_then(|j| j.drag_force)
                    .unwrap_or(0.5);
                let gravity_dir = spring
                    .joints
                    .first()
                    .and_then(|j| j.gravity_dir)
                    .unwrap_or([0.0, -1.0, 0.0]);
                let gravity_power = spring
                    .joints
                    .first()
                    .map(|j| j.gravity_power)
                    .unwrap_or(0.0);
                let radius = spring
                    .joints
                    .first()
                    .and_then(|j| j.hit_radius)
                    .unwrap_or(0.05);

                SpringBoneAsset {
                    chain_root,
                    joints,
                    stiffness,
                    drag_force,
                    gravity_dir,
                    gravity_power,
                    radius,
                    collider_refs,
                }
            })
            .collect();

        (spring_bones, colliders)
    }

    fn build_expressions(ext: &gltf_ext::VrmExtension) -> ExpressionAssetSet {
        let mut expressions = Vec::new();

        if let Some(ref expr_section) = ext.expressions {
            if let Some(ref preset) = expr_section.preset {
                for expr in preset.all_expressions() {
                    expressions.push(ExpressionDef {
                        name: expr.name.clone(),
                        weight: 0.0,
                    });
                }
            }

            if let Some(ref custom) = expr_section.custom {
                for expr in custom {
                    expressions.push(ExpressionDef {
                        name: expr.name.clone(),
                        weight: 0.0,
                    });
                }
            }
        }

        ExpressionAssetSet { expressions }
    }
}

fn compute_hash(data: &[u8]) -> AssetSourceHash {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    AssetSourceHash(hash)
}

fn parse_humanoid_bone(name: &str) -> Option<HumanoidBone> {
    match name {
        "hips" => Some(HumanoidBone::Hips),
        "spine" => Some(HumanoidBone::Spine),
        "chest" => Some(HumanoidBone::Chest),
        "upperChest" => Some(HumanoidBone::UpperChest),
        "neck" => Some(HumanoidBone::Neck),
        "head" => Some(HumanoidBone::Head),
        "leftShoulder" => Some(HumanoidBone::LeftShoulder),
        "leftUpperArm" => Some(HumanoidBone::LeftUpperArm),
        "leftLowerArm" => Some(HumanoidBone::LeftLowerArm),
        "leftHand" => Some(HumanoidBone::LeftHand),
        "rightShoulder" => Some(HumanoidBone::RightShoulder),
        "rightUpperArm" => Some(HumanoidBone::RightUpperArm),
        "rightLowerArm" => Some(HumanoidBone::RightLowerArm),
        "rightHand" => Some(HumanoidBone::RightHand),
        "leftUpperLeg" => Some(HumanoidBone::LeftUpperLeg),
        "leftLowerLeg" => Some(HumanoidBone::LeftLowerLeg),
        "leftFoot" => Some(HumanoidBone::LeftFoot),
        "rightUpperLeg" => Some(HumanoidBone::RightUpperLeg),
        "rightLowerLeg" => Some(HumanoidBone::RightLowerLeg),
        "rightFoot" => Some(HumanoidBone::RightFoot),
        _ => None,
    }
}
