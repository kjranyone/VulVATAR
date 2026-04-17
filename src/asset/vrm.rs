#![allow(dead_code)]
use image::GenericImageView;
use log::warn;
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::asset::*;

static NEXT_AVATAR_ID: AtomicU64 = AtomicU64::new(1);

/// Coarse-grained progress reported by [`VrmAssetLoader::load_with_progress`].
#[derive(Clone, Debug)]
pub enum LoadStage {
    Reading,
    Parsing,
    Skeleton,
    Meshes,
    /// Per-material progress. `current` reaches `total` when done.
    Materials { current: usize, total: usize },
    SpringBones,
    Finalizing,
}

impl LoadStage {
    /// Human-readable label for display in the loading UI.
    pub fn label(&self) -> String {
        match self {
            LoadStage::Reading => "Reading file…".into(),
            LoadStage::Parsing => "Parsing glTF…".into(),
            LoadStage::Skeleton => "Building skeleton…".into(),
            LoadStage::Meshes => "Decoding meshes…".into(),
            LoadStage::Materials { current, total } => {
                format!("Decoding materials ({}/{})", current, total)
            }
            LoadStage::SpringBones => "Building spring bones…".into(),
            LoadStage::Finalizing => "Finalizing…".into(),
        }
    }
}

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
        #[serde(default, rename = "morphTargetBinds")]
        pub morph_target_binds: Option<Vec<MorphTargetBind>>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct ExpressionBind {
        pub node: u32,
        pub property: String,
        pub value: f32,
    }

    /// VRM 1.0 morph target bind (inside `morphTargetBinds`).
    #[derive(Debug, Clone, Deserialize)]
    pub struct MorphTargetBind {
        pub node: u32,
        pub index: u32,
        pub weight: f32,
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
        #[serde(default, rename = "specVersion")]
        pub spec_version: Option<String>,
        #[serde(default)]
        pub meta: Option<Meta>,
    }

    /// VRM 1.x `VRMC_vrm.meta`.
    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct Meta {
        #[serde(default)]
        pub name: Option<String>,
        #[serde(default)]
        pub version: Option<String>,
        #[serde(default)]
        pub authors: Option<Vec<String>>,
        #[serde(default, rename = "copyrightInformation")]
        pub copyright_information: Option<String>,
        #[serde(default, rename = "contactInformation")]
        pub contact_information: Option<String>,
        #[serde(default)]
        pub references: Option<Vec<String>>,
        #[serde(default, rename = "licenseUrl")]
        pub license_url: Option<String>,
        #[serde(default, rename = "thirdPartyLicenses")]
        pub third_party_licenses: Option<String>,
    }

    /// VRM 0.x `VRM.meta`. Field names differ from 1.x.
    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct LegacyMeta {
        #[serde(default)]
        pub title: Option<String>,
        #[serde(default)]
        pub version: Option<String>,
        #[serde(default)]
        pub author: Option<String>,
        #[serde(default, rename = "contactInformation")]
        pub contact_information: Option<String>,
        #[serde(default)]
        pub reference: Option<String>,
        #[serde(default, rename = "licenseName")]
        pub license_name: Option<String>,
        #[serde(default, rename = "otherLicenseUrl")]
        pub other_license_url: Option<String>,
    }

    /// VRM 0.x root `VRM` extension. We only parse the metadata here — the
    /// rest (humanoid, blendshapes, secondaryAnimation) is unsupported by
    /// this loader.
    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct LegacyVrmExtension {
        #[serde(default, rename = "exporterVersion")]
        pub exporter_version: Option<String>,
        #[serde(default, rename = "specVersion")]
        pub spec_version: Option<String>,
        #[serde(default)]
        pub meta: Option<LegacyMeta>,
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
        /// Legacy VRM 0.x root extension (metadata only).
        #[serde(rename = "VRM")]
        pub legacy_vrm: Option<LegacyVrmExtension>,
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

struct MtoonTextureIndices {
    shade_color_texture: Option<usize>,
    emissive_texture: Option<usize>,
    rim_texture: Option<usize>,
    matcap_texture: Option<usize>,
    uv_anim_mask_texture: Option<usize>,
}

impl VrmAssetLoader {
    pub fn new() -> Self {
        Self
    }

    pub fn load(&self, path: &str) -> Result<Arc<AvatarAsset>, VrmLoadError> {
        self.load_with_progress(path, |_| {})
    }

    /// Like [`load`], but invokes `on_progress` at each major stage so the
    /// caller can update a loading UI. The callback runs on the same thread
    /// as the caller — typically a worker thread spawned by the GUI.
    pub fn load_with_progress(
        &self,
        path: &str,
        on_progress: impl Fn(LoadStage),
    ) -> Result<Arc<AvatarAsset>, VrmLoadError> {
        on_progress(LoadStage::Reading);
        let file_data = std::fs::read(path)?;
        self.load_from_bytes_with_progress(&file_data, PathBuf::from(path), on_progress)
    }

    pub fn load_from_bytes(
        &self,
        data: &[u8],
        source_path: PathBuf,
    ) -> Result<Arc<AvatarAsset>, VrmLoadError> {
        self.load_from_bytes_with_progress(data, source_path, |_| {})
    }

    pub fn load_from_bytes_with_progress(
        &self,
        data: &[u8],
        source_path: PathBuf,
        on_progress: impl Fn(LoadStage),
    ) -> Result<Arc<AvatarAsset>, VrmLoadError> {
        let source_hash = compute_hash(data);

        on_progress(LoadStage::Parsing);
        let gltf_doc = gltf::Gltf::from_slice(data)?;
        let blob = gltf_doc.blob.as_deref();
        let root_ext: Option<gltf_ext::RootExtension> = gltf_doc
            .document
            .extensions()
            .and_then(|ext| serde_json::from_value(serde_json::Value::Object(ext.clone())).ok());

        let vrm_ext = root_ext.as_ref().and_then(|r| r.vrmc_vrm.as_ref());
        let spring_ext = root_ext.as_ref().and_then(|r| r.vrmc_spring_bone.as_ref());
        let legacy_ext = root_ext.as_ref().and_then(|r| r.legacy_vrm.as_ref());
        let vrm_meta = Self::build_vrm_meta(vrm_ext, legacy_ext);

        on_progress(LoadStage::Skeleton);
        let (mut nodes, node_name_map) = Self::build_skeleton_nodes(&gltf_doc.document);
        let humanoid = vrm_ext
            .map(|ext| Self::build_humanoid_map(ext, &node_name_map))
            .transpose()?;

        on_progress(LoadStage::Meshes);
        let meshes = Self::build_meshes(&gltf_doc.document, blob);
        let materials =
            Self::build_materials(&gltf_doc.document, blob, &source_path, &on_progress);
        let skin_bindings = Self::build_skin_bindings(&gltf_doc.document, blob);
        let meshes_with_skin =
            Self::assign_skins_to_meshes(&gltf_doc.document, meshes, &skin_bindings);

        let root_nodes: Vec<NodeId> = gltf_doc
            .document
            .scenes()
            .next()
            .map(|scene| scene.nodes().map(|n| NodeId(n.index() as u64)).collect())
            .unwrap_or_default();

        // VRM 0.x stores the model facing -Z; VRM 1.x (and the rest of this
        // engine) expects +Z. Bake a 180° Y rotation into each root node so
        // the skeleton, spring bones and retargeting all see the 1.x
        // orientation without touching the mesh vertex data.
        if vrm_meta.spec_version == VrmSpecVersion::V0 {
            Self::bake_y_flip_on_roots(&mut nodes, &root_nodes);
        }

        // Collect inverse bind matrices from all skins
        let inverse_bind_matrices = if let Some((_, first_skin)) = skin_bindings.first() {
            first_skin.inverse_bind_matrices.clone()
        } else {
            vec![identity_matrix(); nodes.len()]
        };

        on_progress(LoadStage::SpringBones);
        let (spring_bones, colliders) = if let Some(spring_ext) = spring_ext {
            Self::build_spring_bones(spring_ext)
        } else {
            (vec![], vec![])
        };

        on_progress(LoadStage::Finalizing);

        let expressions = vrm_ext
            .map(|ext| Self::build_expressions(ext))
            .unwrap_or_else(|| ExpressionAssetSet {
                expressions: vec![],
            });

        // Build node → mesh index mapping for morph target expression binding.
        let node_to_mesh: std::collections::HashMap<usize, usize> = gltf_doc
            .document
            .nodes()
            .filter_map(|node| {
                node.mesh().map(|m| (node.index(), m.index()))
            })
            .collect();

        let mut root_aabb = Aabb::empty();
        for mesh in &meshes_with_skin {
            for prim in &mesh.primitives {
                root_aabb.expand(&prim.bounds);
            }
        }

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
            node_to_mesh,
            vrm_meta,
            root_aabb,
        };

        Ok(Arc::new(asset))
    }

    /// Pre-multiply each root node's rest_local by a 180° Y rotation, so the
    /// whole skeleton subtree appears rotated 180° around Y in world space.
    /// Equivalent to `three-vrm`'s `VRMUtils.rotateVRM0()`.
    fn bake_y_flip_on_roots(nodes: &mut [SkeletonNode], root_nodes: &[NodeId]) {
        // Y-axis 180° quaternion in xyzw layout: (0, 1, 0, 0).
        let y_flip: [f32; 4] = [0.0, 1.0, 0.0, 0.0];
        for root_id in root_nodes {
            let idx = root_id.0 as usize;
            if idx >= nodes.len() {
                continue;
            }
            let node = &mut nodes[idx];
            // new_rot = Y180 * old_rot
            node.rest_local.rotation =
                crate::math_utils::quat_mul(&y_flip, &node.rest_local.rotation);
            // new_t = Y180 * old_t  →  flip X and Z, keep Y.
            node.rest_local.translation = [
                -node.rest_local.translation[0],
                node.rest_local.translation[1],
                -node.rest_local.translation[2],
            ];
        }
    }

    fn build_vrm_meta(
        vrm_ext: Option<&gltf_ext::VrmExtension>,
        legacy_ext: Option<&gltf_ext::LegacyVrmExtension>,
    ) -> VrmMeta {
        if let Some(ext) = vrm_ext {
            let meta = ext.meta.clone().unwrap_or_default();
            return VrmMeta {
                spec_version: VrmSpecVersion::V1,
                spec_version_raw: ext.spec_version.clone(),
                title: meta.name,
                authors: meta.authors.unwrap_or_default(),
                model_version: meta.version,
                contact_information: meta.contact_information,
                references: meta.references.unwrap_or_default(),
                license: meta.license_url,
                copyright_information: meta.copyright_information,
            };
        }
        if let Some(ext) = legacy_ext {
            let meta = ext.meta.clone().unwrap_or_default();
            // VRM 0.x `author` is a single string; convention is
            // comma-separated for multiple authors. Split so Inspector and
            // library entries treat each as its own name.
            let authors: Vec<String> = meta
                .author
                .map(|s| {
                    s.split(',')
                        .map(|a| a.trim().to_string())
                        .filter(|a| !a.is_empty())
                        .collect()
                })
                .unwrap_or_default();
            let references = meta.reference.into_iter().collect();
            return VrmMeta {
                spec_version: VrmSpecVersion::V0,
                spec_version_raw: ext.spec_version.clone(),
                title: meta.title,
                authors,
                model_version: meta.version,
                contact_information: meta.contact_information,
                references,
                license: meta.license_name.or(meta.other_license_url),
                copyright_information: None,
            };
        }
        VrmMeta::default()
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
                        warn!(
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

                        let idx_data = if index_count > 0 { Some(indices) } else { None };

                        // Parse morph targets (blend shapes).
                        let target_names: Vec<String> = gmesh
                            .extras()
                            .as_ref()
                            .and_then(|raw| {
                                let v: serde_json::Value =
                                    serde_json::from_str(raw.get()).ok()?;
                                let arr = v.get("targetNames")?.as_array()?;
                                Some(
                                    arr.iter()
                                        .filter_map(|s| s.as_str().map(String::from))
                                        .collect(),
                                )
                            })
                            .unwrap_or_default();

                        let get_buf_data = |buffer: gltf::Buffer<'_>| -> Option<&[u8]> {
                            buffers.get(buffer.index()).map(|d| d.0.as_slice())
                        };

                        let morph_targets: Vec<MorphTargetDelta> = prim
                            .morph_targets()
                            .enumerate()
                            .map(|(ti, mt)| {
                                let pos_d: Vec<[f32; 3]> = mt
                                    .positions()
                                    .and_then(|acc| {
                                        let iter = gltf::accessor::Iter::<[f32; 3]>::new(
                                            acc,
                                            get_buf_data.clone(),
                                        )?;
                                        Some(iter.collect())
                                    })
                                    .unwrap_or_default();
                                let norm_d: Vec<[f32; 3]> = mt
                                    .normals()
                                    .and_then(|acc| {
                                        let iter = gltf::accessor::Iter::<[f32; 3]>::new(
                                            acc,
                                            get_buf_data.clone(),
                                        )?;
                                        Some(iter.collect())
                                    })
                                    .unwrap_or_default();
                                let name = target_names
                                    .get(ti)
                                    .cloned()
                                    .unwrap_or_else(|| format!("target_{}", ti));
                                MorphTargetDelta {
                                    name,
                                    position_deltas: pos_d,
                                    normal_deltas: norm_d,
                                }
                            })
                            .collect();

                        if !morph_targets.is_empty() {
                            log::info!(
                                "[vrm] mesh '{}' prim {} has {} morph targets",
                                gmesh.name().unwrap_or("unnamed"),
                                prim_id.0,
                                morph_targets.len(),
                            );
                        }

                        MeshPrimitiveAsset {
                            id: prim_id,
                            vertex_count,
                            index_count,
                            material_id,
                            skin: None,
                            bounds,
                            vertices,
                            indices: idx_data,
                            morph_targets,
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

    fn resolve_texture_index(
        doc: &gltf::Document,
        index: usize,
        blob: Option<&[u8]>,
        source_path: &Path,
    ) -> Option<TextureBinding> {
        doc.textures()
            .nth(index)
            .map(|tex| Self::texture_binding_from_info(&tex, blob, source_path))
    }

    fn build_materials(
        doc: &gltf::Document,
        blob: Option<&[u8]>,
        source_path: &Path,
        on_progress: &dyn Fn(LoadStage),
    ) -> Vec<MaterialAsset> {
        let total = doc.materials().count();
        on_progress(LoadStage::Materials { current: 0, total });
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

                let mtoon_ext = gmat
                    .extensions()
                    .and_then(|ext| ext.get("VRMC_materials_mtoon"));
                let has_mtoon = mtoon_ext.is_some();

                let base_mode = if has_mtoon {
                    MaterialMode::ToonLike
                } else {
                    MaterialMode::SimpleLit
                };

                let base_color_texture = pbr.base_color_texture().map(|info| {
                    Self::texture_binding_from_info(&info.texture(), blob, source_path)
                });

                let normal_map_texture = gmat.normal_texture().map(|info| {
                    Self::texture_binding_from_info(&info.texture(), blob, source_path)
                });

                let mut shade_ramp_texture = None;
                let mut emissive_texture_binding = None;
                let mut matcap_texture_binding = None;

                let (toon_params, mtoon_params) = if let Some(mtoon) = mtoon_ext {
                    let (toon, mut staged, tex_indices) = Self::parse_mtoon_params(mtoon);

                    shade_ramp_texture = tex_indices
                        .shade_color_texture
                        .and_then(|idx| Self::resolve_texture_index(doc, idx, blob, source_path));

                    emissive_texture_binding = tex_indices
                        .emissive_texture
                        .and_then(|idx| Self::resolve_texture_index(doc, idx, blob, source_path));

                    matcap_texture_binding = tex_indices
                        .matcap_texture
                        .and_then(|idx| Self::resolve_texture_index(doc, idx, blob, source_path));

                    staged.emissive_texture =
                        emissive_texture_binding
                            .as_ref()
                            .map(|tb| MtoonTextureSlot {
                                uri: tb.uri.clone(),
                            });
                    staged.rim_texture = tex_indices
                        .rim_texture
                        .and_then(|idx| Self::resolve_texture_index(doc, idx, blob, source_path))
                        .map(|tb| MtoonTextureSlot { uri: tb.uri });
                    staged.matcap_texture =
                        matcap_texture_binding.as_ref().map(|tb| MtoonTextureSlot {
                            uri: tb.uri.clone(),
                        });
                    staged.uv_anim_mask_texture = tex_indices
                        .uv_anim_mask_texture
                        .and_then(|idx| Self::resolve_texture_index(doc, idx, blob, source_path))
                        .map(|tb| MtoonTextureSlot { uri: tb.uri });

                    (toon, Some(staged))
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

                let asset = MaterialAsset {
                    id: MaterialId(i as u64 + 1),
                    name: gmat.name().unwrap_or("unnamed").to_string(),
                    base_mode,
                    base_color,
                    alpha_mode,
                    double_sided,
                    texture_bindings: MaterialTextureSet {
                        base_color_texture,
                        normal_map_texture,
                        shade_ramp_texture,
                        emissive_texture: emissive_texture_binding,
                        matcap_texture: matcap_texture_binding,
                    },
                    toon_params,
                    mtoon_params,
                };
                on_progress(LoadStage::Materials {
                    current: i + 1,
                    total,
                });
                asset
            })
            .inspect(|mat| {
                let tb = &mat.texture_bindings;
                let base = tb.base_color_texture.as_ref();
                log::info!(
                    "[vrm] material #{} '{}' mode={:?} base_tex={} ({}x{})",
                    mat.id.0,
                    mat.name,
                    mat.base_mode,
                    base.map_or("none", |t| &t.uri),
                    base.map_or(0, |t| t.dimensions.0),
                    base.map_or(0, |t| t.dimensions.1),
                );
            })
            .collect()
    }

    fn texture_binding_from_info(
        texture: &gltf::Texture<'_>,
        blob: Option<&[u8]>,
        source_path: &Path,
    ) -> TextureBinding {
        let base_dir = source_path.parent();
        let default_name = format!("texture_{}", texture.index());
        let texture_name = texture.source().name().unwrap_or(default_name.as_str());

        let source = texture.source().source();
        let (uri, pixel_data, dimensions) = match source {
            gltf::image::Source::View { view, .. } => {
                let start = view.offset();
                let end = start + view.length();
                let cache_key = format!("{}#{}", source_path.display(), texture_name);
                match blob.and_then(|b| b.get(start..end)) {
                    Some(encoded) => match image::load_from_memory(encoded) {
                        Ok(img) => {
                            let (w, h) = img.dimensions();
                            log::info!(
                                "[vrm] texture '{}' loaded from buffer: {}x{} ({} bytes)",
                                cache_key,
                                w,
                                h,
                                encoded.len()
                            );
                            (cache_key, Some(Arc::new(img.to_rgba8().into_raw())), (w, h))
                        }
                        Err(e) => {
                            log::warn!("[vrm] texture '{}' decode failed: {}", cache_key, e);
                            (cache_key, None, (0, 0))
                        }
                    },
                    None => {
                        log::warn!(
                            "[vrm] texture '{}' buffer range {}..{} out of bounds",
                            cache_key,
                            start,
                            end
                        );
                        (cache_key, None, (0, 0))
                    }
                }
            }
            gltf::image::Source::Uri { uri: file_uri, .. } => {
                let resolved_path = base_dir
                    .map(|dir| dir.join(file_uri))
                    .unwrap_or_else(|| PathBuf::from(file_uri));
                let cache_key = resolved_path.to_string_lossy().into_owned();
                match image::open(&resolved_path) {
                    Ok(img) => {
                        let (w, h) = img.dimensions();
                        log::info!(
                            "[vrm] texture '{}' loaded from uri '{}': {}x{}",
                            texture_name,
                            resolved_path.display(),
                            w,
                            h
                        );
                        (cache_key, Some(Arc::new(img.to_rgba8().into_raw())), (w, h))
                    }
                    Err(e) => {
                        log::warn!(
                            "[vrm] texture '{}' file open failed for '{}': {}",
                            texture_name,
                            resolved_path.display(),
                            e
                        );
                        (cache_key, None, (0, 0))
                    }
                }
            }
        };

        TextureBinding {
            uri,
            pixel_data,
            dimensions,
        }
    }

    fn parse_mtoon_params(
        mtoon: &serde_json::Value,
    ) -> (ToonMaterialParams, MtoonStagedParams, MtoonTextureIndices) {
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
        let get_color3_as_4 = |key: &str| -> [f32; 4] {
            mtoon
                .get(key)
                .and_then(|v| v.as_array())
                .map(|arr| {
                    [
                        arr.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        arr.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        arr.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        1.0f32,
                    ]
                })
                .unwrap_or([0.0, 0.0, 0.0, 0.0])
        };

        let shade_color = get_color3_as_4("shadeColorFactor");
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

        let rim_color = get_color3_as_4("parametricRimColorFactor");
        let rim_fresnel_power = get_f32("parametricRimFresnelPowerFactor").unwrap_or(5.0);
        let rim_lift = get_f32("parametricRimLiftFactor").unwrap_or(0.0);
        let rim_lighting_mix = get_f32("rimLightingMixFactor").unwrap_or(1.0);

        let emissive_color = get_color3_as_4("emissiveFactor");

        let uv_anim_scroll_x = get_f32("uvAnimationScrollXSpeedFactor").unwrap_or(0.0);
        let uv_anim_scroll_y = get_f32("uvAnimationScrollYSpeedFactor").unwrap_or(0.0);
        let uv_anim_rotation = get_f32("uvAnimationRotationSpeedFactor").unwrap_or(0.0);

        let get_tex_index = |key: &str| {
            mtoon.get(key).and_then(|v| {
                v.as_u64()
                    .or_else(|| v.get("index").and_then(|idx| idx.as_u64()))
                    .map(|v| v as usize)
            })
        };

        let tex_indices = MtoonTextureIndices {
            shade_color_texture: get_tex_index("shadeColorTexture"),
            emissive_texture: get_tex_index("emissiveTexture"),
            rim_texture: get_tex_index("rimTexture"),
            matcap_texture: get_tex_index("matcapTexture"),
            uv_anim_mask_texture: get_tex_index("uvAnimMaskTexture"),
        };

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

        (toon_params, staged, tex_indices)
    }

    fn build_skin_bindings(doc: &gltf::Document, blob: Option<&[u8]>) -> Vec<(usize, SkinBinding)> {
        let buffers: Vec<gltf::buffer::Data> = doc
            .buffers()
            .map(|buf| {
                let data = match buf.source() {
                    gltf::buffer::Source::Bin => blob.unwrap_or(&[]).to_vec(),
                    gltf::buffer::Source::Uri(_uri) => {
                        warn!(
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
                        warn!(
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
                let joint_stiffness: Vec<f32> = spring.joints.iter().map(|j| j.stiffness).collect();
                let joint_drag: Vec<f32> = spring
                    .joints
                    .iter()
                    .map(|j| j.drag_force.unwrap_or(drag_force))
                    .collect();
                let joint_gravity_power: Vec<f32> =
                    spring.joints.iter().map(|j| j.gravity_power).collect();

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

    fn build_expression_def(expr: &gltf_ext::Expression) -> ExpressionDef {
        let morph_binds = expr
            .morph_target_binds
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .map(|b| ExpressionMorphBind {
                node_index: b.node as usize,
                morph_target_index: b.index as usize,
                weight: b.weight,
            })
            .collect();

        ExpressionDef {
            name: expr.name.clone(),
            weight: 0.0,
            morph_binds,
        }
    }

    fn build_expressions(ext: &gltf_ext::VrmExtension) -> ExpressionAssetSet {
        let mut expressions = Vec::new();

        if let Some(ref expr_section) = ext.expressions {
            if let Some(ref preset) = expr_section.preset {
                for expr in preset.all_expressions() {
                    expressions.push(Self::build_expression_def(expr));
                }
            }

            if let Some(ref custom) = expr_section.custom {
                for expr in custom {
                    expressions.push(Self::build_expression_def(expr));
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
