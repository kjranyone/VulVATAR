use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::asset::*;

static NEXT_AVATAR_ID: AtomicU64 = AtomicU64::new(1);

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
        let blob = gltf_doc.blob.as_deref();
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

        let meshes = Self::build_meshes(&gltf_doc.document, blob);
        let materials = Self::build_materials(&gltf_doc.document);
        let skin_bindings = Self::build_skin_bindings(&gltf_doc.document, blob);
        let meshes_with_skin = Self::assign_skins_to_meshes(
            &gltf_doc.document,
            meshes,
            &skin_bindings,
        );

        let root_nodes: Vec<NodeId> = gltf_doc
            .document
            .scenes()
            .next()
            .map(|scene| scene.nodes().map(|n| NodeId(n.index() as u64)).collect())
            .unwrap_or_default();

        // Collect inverse bind matrices from all skins
        let inverse_bind_matrices = if let Some((_, first_skin)) = skin_bindings.first() {
            first_skin.inverse_bind_matrices.clone()
        } else {
            vec![identity_matrix(); nodes.len()]
        };

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
            id: AvatarAssetId(NEXT_AVATAR_ID.fetch_add(1, Ordering::Relaxed)),
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
            animation_clips: Self::build_animation_clips(&gltf_doc.document, blob),
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

    fn build_meshes(doc: &gltf::Document, blob: Option<&[u8]>) -> Vec<MeshAsset> {
        let mut next_mesh_id: u64 = 1;
        let mut next_prim_id: u64 = 1;

        let buffers: Vec<gltf::buffer::Data> = doc
            .buffers()
            .map(|buf| {
                let data = match buf.source() {
                    gltf::buffer::Source::Bin => blob.unwrap_or(&[]).to_vec(),
                    gltf::buffer::Source::Uri(_uri) => {
                        eprintln!(
                            "[vrm] External buffer URIs not supported for mesh loading, \
                             buffer {} will be empty",
                            buf.index()
                        );
                        vec![]
                    }
                };
                gltf::buffer::Data(data)
            })
            .collect();

        doc.meshes()
            .map(|gmesh| {
                let mesh_id = MeshId(next_mesh_id);
                next_mesh_id += 1;

                let primitives: Vec<MeshPrimitiveAsset> = gmesh
                    .primitives()
                    .map(|prim| {
                        let prim_id = PrimitiveId(next_prim_id);
                        next_prim_id += 1;

                        let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

                        let positions: Vec<[f32; 3]> = reader
                            .read_positions()
                            .map(|iter| iter.collect())
                            .unwrap_or_default();
                        let vertex_count = positions.len() as u32;

                        let normals: Vec<[f32; 3]> = reader
                            .read_normals()
                            .map(|iter| iter.collect())
                            .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; vertex_count as usize]);

                        let uvs: Vec<[f32; 2]> = reader
                            .read_tex_coords(0)
                            .map(|tc| tc.into_f32().collect())
                            .unwrap_or_else(|| vec![[0.0, 0.0]; vertex_count as usize]);

                        let joint_indices: Vec<[u16; 4]> = reader
                            .read_joints(0)
                            .map(|j| j.into_u16().collect())
                            .unwrap_or_default();

                        let joint_weights: Vec<[f32; 4]> = reader
                            .read_weights(0)
                            .map(|w| w.into_f32().collect())
                            .unwrap_or_default();

                        let indices: Vec<u32> = reader
                            .read_indices()
                            .map(|idx| idx.into_u32().collect())
                            .unwrap_or_default();
                        let index_count = indices.len() as u32;

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

                        let vertices = if vertex_count > 0 {
                            Some(VertexData {
                                positions,
                                normals,
                                uvs,
                                joint_indices,
                                joint_weights,
                            })
                        } else {
                            None
                        };

                        let idx_data = if index_count > 0 {
                            Some(indices)
                        } else {
                            None
                        };

                        MeshPrimitiveAsset {
                            id: prim_id,
                            vertex_count,
                            index_count,
                            material_id,
                            skin: None,
                            bounds,
                            vertices,
                            indices: idx_data,
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

                let gltf_alpha = gmat.alpha_mode();
                let alpha_mode = match gltf_alpha {
                    gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                    gltf::material::AlphaMode::Mask => {
                        AlphaMode::Mask(gmat.alpha_cutoff().unwrap_or(0.5))
                    }
                    gltf::material::AlphaMode::Blend => AlphaMode::Blend,
                };
                let double_sided = gmat.double_sided();

                // Extract MToon extension if present
                let mtoon_ext = gmat.extensions().and_then(|ext| ext.get("VRMC_materials_mtoon"));
                let has_mtoon = mtoon_ext.is_some();

                let base_mode = if has_mtoon {
                    MaterialMode::ToonLike
                } else if matches!(gltf_alpha, gltf::material::AlphaMode::Blend) {
                    MaterialMode::Unlit
                } else if pbr.metallic_factor() > 0.0 {
                    MaterialMode::SimpleLit
                } else {
                    MaterialMode::ToonLike
                };

                let base_color_texture = pbr.base_color_texture().map(|info| {
                    Self::texture_binding_from_info(&info.texture())
                });

                let normal_map_texture = gmat.normal_texture().map(|info| {
                    Self::texture_binding_from_info(&info.texture())
                });

                // Parse MToon parameters
                let (toon_params, mtoon_params) = if let Some(mtoon) = mtoon_ext {
                    Self::parse_mtoon_params(mtoon)
                } else {
                    (
                        ToonMaterialParams {
                            ramp_threshold: 0.5,
                            shadow_softness: 0.1,
                            outline_width: 0.01,
                            outline_color: [0.0, 0.0, 0.0],
                        },
                        None,
                    )
                };

                MaterialAsset {
                    id: MaterialId(i as u64 + 1),
                    name: gmat.name().unwrap_or("unnamed").to_string(),
                    base_mode,
                    base_color,
                    alpha_mode,
                    double_sided,
                    texture_bindings: MaterialTextureSet {
                        base_color_texture,
                        normal_map_texture,
                        shade_ramp_texture: None,
                        emissive_texture: None,
                    },
                    toon_params,
                    mtoon_params,
                }
            })
            .collect()
    }

    fn texture_binding_from_info(texture: &gltf::Texture<'_>) -> TextureBinding {
        let uri = texture
            .source()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|| format!("texture_{}", texture.index()));
        TextureBinding { uri }
    }

    fn parse_mtoon_params(
        mtoon: &serde_json::Value,
    ) -> (ToonMaterialParams, Option<MtoonStagedParams>) {
        use crate::asset::{MtoonOutlineWidthMode, MtoonStagedParams};

        let get_f32 = |key: &str| mtoon.get(key).and_then(|v| v.as_f64()).map(|v| v as f32);
        let get_color3 = |key: &str| -> [f32; 3] {
            mtoon
                .get(key)
                .and_then(|v| v.as_array())
                .map(|arr| {
                    [
                        arr.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        arr.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        arr.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                    ]
                })
                .unwrap_or([0.0; 3])
        };
        let get_color4 = |key: &str| -> [f32; 4] {
            mtoon
                .get(key)
                .and_then(|v| v.as_array())
                .map(|arr| {
                    [
                        arr.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        arr.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        arr.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        arr.get(3).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                    ]
                })
                .unwrap_or([0.0, 0.0, 0.0, 1.0])
        };

        let shade_color = get_color4("shadeColorFactor");
        let shade_shift = get_f32("shadingShiftFactor").unwrap_or(0.0);
        let shade_toony = get_f32("shadingToonyFactor").unwrap_or(0.9);
        let gi_equalization = get_f32("giEqualizationFactor").unwrap_or(0.9);

        let outline_width_mode_str = mtoon
            .get("outlineWidthMode")
            .and_then(|v| v.as_str())
            .unwrap_or("none");
        let outline_width_mode = match outline_width_mode_str {
            "worldCoordinates" => MtoonOutlineWidthMode::WorldCoordinates,
            "screenCoordinates" => MtoonOutlineWidthMode::ScreenCoordinates,
            _ => MtoonOutlineWidthMode::None,
        };
        let outline_width = get_f32("outlineWidthFactor").unwrap_or(0.0);
        let outline_color = get_color3("outlineColorFactor");

        let rim_color = get_color4("parametricRimColorFactor");
        let rim_fresnel_power = get_f32("parametricRimFresnelPowerFactor").unwrap_or(5.0);
        let rim_lift = get_f32("parametricRimLiftFactor").unwrap_or(0.0);
        let rim_lighting_mix = get_f32("rimLightingMixFactor").unwrap_or(1.0);

        let emissive_color = get_color4("emissiveFactor");

        let uv_anim_scroll_x = get_f32("uvAnimationScrollXSpeedFactor").unwrap_or(0.0);
        let uv_anim_scroll_y = get_f32("uvAnimationScrollYSpeedFactor").unwrap_or(0.0);
        let uv_anim_rotation = get_f32("uvAnimationRotationSpeedFactor").unwrap_or(0.0);

        let toon_params = ToonMaterialParams {
            ramp_threshold: (1.0 - shade_toony).max(0.01),
            shadow_softness: (1.0 - shade_toony) * 0.5,
            outline_width,
            outline_color,
        };

        let staged = MtoonStagedParams {
            shade_color,
            shade_shift,
            shade_toony,
            lit_color: [1.0, 1.0, 1.0, 1.0],
            gi_equalization,
            matcap_texture: None,
            rim_texture: None,
            rim_color,
            rim_lighting_mix,
            rim_fresnel_power,
            rim_lift,
            emissive_texture: None,
            emissive_color,
            outline_width_mode,
            outline_color,
            outline_width,
            uv_anim_mask_texture: None,
            uv_anim_scroll_x_speed: uv_anim_scroll_x,
            uv_anim_scroll_y_speed: uv_anim_scroll_y,
            uv_anim_rotation_speed: uv_anim_rotation,
        };

        (toon_params, Some(staged))
    }

    fn build_skin_bindings(
        doc: &gltf::Document,
        blob: Option<&[u8]>,
    ) -> Vec<(usize, SkinBinding)> {
        let buffers: Vec<gltf::buffer::Data> = doc
            .buffers()
            .map(|buf| {
                let data = match buf.source() {
                    gltf::buffer::Source::Bin => blob.unwrap_or(&[]).to_vec(),
                    gltf::buffer::Source::Uri(_uri) => {
                        eprintln!(
                            "[vrm] External buffer URIs not supported for skin loading, \
                             buffer {} will be empty",
                            buf.index()
                        );
                        vec![]
                    }
                };
                gltf::buffer::Data(data)
            })
            .collect();

        doc.skins()
            .enumerate()
            .map(|(skin_index, skin)| {
                let joint_nodes: Vec<NodeId> =
                    skin.joints().map(|j| NodeId(j.index() as u64)).collect();

                let reader = skin.reader(|buffer| Some(&buffers[buffer.index()]));
                let inverse_bind_matrices: Vec<Mat4> = reader
                    .read_inverse_bind_matrices()
                    .map(|iter| iter.collect())
                    .unwrap_or_else(|| {
                        eprintln!(
                            "[vrm] Skin {} has no inverse bind matrices, using identity",
                            skin_index
                        );
                        vec![identity_matrix(); joint_nodes.len()]
                    });

                (
                    skin_index,
                    SkinBinding {
                        joint_nodes,
                        inverse_bind_matrices,
                    },
                )
            })
            .collect()
    }

    fn assign_skins_to_meshes(
        doc: &gltf::Document,
        meshes: Vec<MeshAsset>,
        skins: &[(usize, SkinBinding)],
    ) -> Vec<MeshAsset> {
        // Build a map from glTF mesh index to skin index by examining nodes.
        // Each node can reference both a mesh and a skin.
        let mut mesh_to_skin: HashMap<usize, usize> = HashMap::new();
        for node in doc.nodes() {
            if let (Some(mesh), Some(skin)) = (node.mesh(), node.skin()) {
                mesh_to_skin.insert(mesh.index(), skin.index());
            }
        }

        meshes
            .into_iter()
            .enumerate()
            .map(|(mesh_idx, mut mesh)| {
                let skin_binding = mesh_to_skin
                    .get(&mesh_idx)
                    .and_then(|skin_idx| skins.get(*skin_idx).map(|(_, s)| s));

                if let Some(skin) = skin_binding {
                    for prim in &mut mesh.primitives {
                        prim.skin = Some(skin.clone());
                    }
                } else if let Some((_, fallback_skin)) = skins.first() {
                    // Fallback: if no explicit mapping, use the first skin (common in VRM)
                    for prim in &mut mesh.primitives {
                        prim.skin = Some(fallback_skin.clone());
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
                            let offset = capsule.offset.unwrap_or([0.0; 3]);
                            // Compute capsule height from the distance between
                            // the offset (head) and the tail position. If tail
                            // is not specified, fall back to zero height.
                            let height = if let Some(tail) = capsule.tail {
                                let dx = tail[0] - offset[0];
                                let dy = tail[1] - offset[1];
                                let dz = tail[2] - offset[2];
                                (dx * dx + dy * dy + dz * dz).sqrt()
                            } else {
                                0.0
                            };
                            (
                                ColliderShape::Capsule {
                                    radius: capsule.radius,
                                    height,
                                },
                                offset,
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

                // Per-joint parameter storage: each joint in the chain can have
                // its own stiffness, drag and gravity_power values.
                let joint_stiffness: Vec<f32> = spring
                    .joints
                    .iter()
                    .map(|j| j.stiffness)
                    .collect();
                let joint_drag: Vec<f32> = spring
                    .joints
                    .iter()
                    .map(|j| j.drag_force.unwrap_or(drag_force))
                    .collect();
                let joint_gravity_power: Vec<f32> = spring
                    .joints
                    .iter()
                    .map(|j| j.gravity_power)
                    .collect();

                SpringBoneAsset {
                    chain_root,
                    joints,
                    stiffness,
                    drag_force,
                    gravity_dir,
                    gravity_power,
                    radius,
                    collider_refs,
                    joint_stiffness,
                    joint_drag,
                    joint_gravity_power,
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

    fn build_animation_clips(doc: &gltf::Document, blob: Option<&[u8]>) -> Vec<AnimationClip> {
        let buffers: Vec<gltf::buffer::Data> = doc
            .buffers()
            .map(|buf| {
                let data = match buf.source() {
                    gltf::buffer::Source::Bin => blob.unwrap_or(&[]).to_vec(),
                    gltf::buffer::Source::Uri(_) => vec![],
                };
                gltf::buffer::Data(data)
            })
            .collect();

        let mut clips = Vec::new();
        for (anim_idx, anim) in doc.animations().enumerate() {
            let name = anim
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|| format!("Animation_{}", anim_idx));

            let mut channels = Vec::new();
            let mut duration: f32 = 0.0;

            for channel in anim.channels() {
                let target = channel.target();
                let target_node = NodeId(target.node().index() as u64);

                let property = match target.property() {
                    gltf::animation::Property::Translation => AnimationProperty::Translation,
                    gltf::animation::Property::Rotation => AnimationProperty::Rotation,
                    gltf::animation::Property::Scale => AnimationProperty::Scale,
                    gltf::animation::Property::MorphTargetWeights => continue,
                };

                let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));

                let timestamps: Vec<f32> = reader
                    .read_inputs()
                    .map(|iter| iter.collect())
                    .unwrap_or_default();

                if let Some(&last_t) = timestamps.last() {
                    if last_t > duration {
                        duration = last_t;
                    }
                }

                let sampler = channel.sampler();
                let interp = match sampler.interpolation() {
                    gltf::animation::Interpolation::Step => InterpolationMode::Step,
                    gltf::animation::Interpolation::Linear => InterpolationMode::Linear,
                    gltf::animation::Interpolation::CubicSpline => InterpolationMode::CubicSpline,
                };

                let outputs = reader.read_outputs();
                let keyframes: Vec<Keyframe> = match outputs {
                    Some(gltf::animation::util::ReadOutputs::Translations(iter)) => {
                        let values: Vec<[f32; 3]> = iter.collect();
                        Self::build_keyframes_vec3(&timestamps, &values, interp)
                    }
                    Some(gltf::animation::util::ReadOutputs::Rotations(rotations)) => {
                        let values: Vec<[f32; 4]> = rotations.into_f32().collect();
                        Self::build_keyframes_vec4(&timestamps, &values, interp)
                    }
                    Some(gltf::animation::util::ReadOutputs::Scales(iter)) => {
                        let values: Vec<[f32; 3]> = iter.collect();
                        Self::build_keyframes_vec3(&timestamps, &values, interp)
                    }
                    _ => continue,
                };

                channels.push(AnimationChannel {
                    target_node,
                    property,
                    keyframes,
                });
            }

            if channels.is_empty() {
                continue;
            }

            clips.push(AnimationClip {
                id: AnimationClipId(anim_idx as u64 + 1),
                name,
                duration,
                channels,
            });
        }

        clips
    }

    /// Build keyframes from vec3 outputs (translation/scale).
    /// For CubicSpline, every 3 consecutive values are [in_tangent, value, out_tangent].
    fn build_keyframes_vec3(
        timestamps: &[f32],
        values: &[[f32; 3]],
        interp: InterpolationMode,
    ) -> Vec<Keyframe> {
        if interp == InterpolationMode::CubicSpline {
            // CubicSpline: 3 values per keyframe (in_tangent, value, out_tangent)
            timestamps
                .iter()
                .enumerate()
                .filter_map(|(i, &time)| {
                    let base = i * 3;
                    if base + 2 >= values.len() {
                        return None;
                    }
                    let in_t = values[base];
                    let val = values[base + 1];
                    let out_t = values[base + 2];
                    Some(Keyframe {
                        time,
                        value: [val[0], val[1], val[2], 0.0],
                        tangents: Some([
                            [in_t[0], in_t[1], in_t[2], 0.0],
                            [out_t[0], out_t[1], out_t[2], 0.0],
                        ]),
                        interpolation: interp,
                    })
                })
                .collect()
        } else {
            timestamps
                .iter()
                .zip(values.iter())
                .map(|(&time, val)| Keyframe {
                    time,
                    value: [val[0], val[1], val[2], 0.0],
                    tangents: None,
                    interpolation: interp,
                })
                .collect()
        }
    }

    /// Build keyframes from vec4 outputs (rotation quaternion).
    fn build_keyframes_vec4(
        timestamps: &[f32],
        values: &[[f32; 4]],
        interp: InterpolationMode,
    ) -> Vec<Keyframe> {
        if interp == InterpolationMode::CubicSpline {
            timestamps
                .iter()
                .enumerate()
                .filter_map(|(i, &time)| {
                    let base = i * 3;
                    if base + 2 >= values.len() {
                        return None;
                    }
                    Some(Keyframe {
                        time,
                        value: values[base + 1],
                        tangents: Some([values[base], values[base + 2]]),
                        interpolation: interp,
                    })
                })
                .collect()
        } else {
            timestamps
                .iter()
                .zip(values.iter())
                .map(|(&time, &val)| Keyframe {
                    time,
                    value: val,
                    tangents: None,
                    interpolation: interp,
                })
                .collect()
        }
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
