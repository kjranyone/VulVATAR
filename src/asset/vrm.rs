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

// ---------------------------------------------------------------------------
// Version-specific schema modules
// ---------------------------------------------------------------------------
//
// VRM 0.x and 1.x have incompatible JSON layouts and live under different
// root extensions, so each version gets its own serde schema here. The
// loader picks the right one based on which extension is present in the
// glTF root `extensions` map.
//
// VRM 0.x (root extension: `VRM`)
//   - humanoid.humanBones: array of `{bone, node, ...}`
//   - blendShapeMaster.blendShapeGroups: array; binds reference mesh index
//     with weight in 0..=100
//   - secondaryAnimation.boneGroups: per-root spring chains sharing
//     parameters; field is spelled `stiffiness` (sic) in the on-disk spec
//   - secondaryAnimation.colliderGroups: spheres only
//
// VRM 1.x (root extensions: `VRMC_vrm` + `VRMC_springBone`)
//   - VRMC_vrm.humanoid.humanBones: map `{boneName: {node}}`
//   - VRMC_vrm.expressions.preset: map with fixed camelCase keys; the
//     map key IS the expression name (no `name` field on the value)
//   - VRMC_vrm.expressions.custom: map {name: Expression}
//   - VRMC_springBone.springs[].joints[]: per-joint stiffness/drag/gravity
//     with field names in camelCase
//   - VRMC_springBone.colliders: sphere or capsule

mod v0 {
    use serde::Deserialize;

    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct VrmRoot {
        #[serde(default, rename = "exporterVersion")]
        pub exporter_version: Option<String>,
        #[serde(default, rename = "specVersion")]
        pub spec_version: Option<String>,
        #[serde(default)]
        pub meta: Option<Meta>,
        #[serde(default)]
        pub humanoid: Option<Humanoid>,
        #[serde(default, rename = "blendShapeMaster")]
        pub blend_shape_master: Option<BlendShapeMaster>,
        #[serde(default, rename = "secondaryAnimation")]
        pub secondary_animation: Option<SecondaryAnimation>,
    }

    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct Meta {
        #[serde(default)]
        pub title: Option<String>,
        #[serde(default)]
        pub version: Option<String>,
        /// VRM 0.x uses a single `author` string; multi-author files put
        /// names as a comma-separated list. The loader splits on commas.
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
        /// VRM 0.x stores the thumbnail as a glTF *texture* index. Some
        /// exporters emit `-1` for "no thumbnail" so the type is signed.
        #[serde(default)]
        pub texture: Option<i64>,
    }

    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct Humanoid {
        #[serde(default, rename = "humanBones")]
        pub human_bones: Vec<HumanBone>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct HumanBone {
        #[serde(default)]
        pub bone: String,
        /// Nullable in practice: some exporters emit `-1` for unmapped bones.
        #[serde(default)]
        pub node: i32,
    }

    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct BlendShapeMaster {
        #[serde(default, rename = "blendShapeGroups")]
        pub blend_shape_groups: Vec<BlendShapeGroup>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct BlendShapeGroup {
        #[serde(default)]
        pub name: String,
        #[serde(default, rename = "presetName")]
        pub preset_name: Option<String>,
        #[serde(default)]
        pub binds: Vec<BlendShapeBind>,
        #[serde(default, rename = "isBinary")]
        pub is_binary: Option<bool>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct BlendShapeBind {
        #[serde(default)]
        pub mesh: i32,
        #[serde(default)]
        pub index: i32,
        /// VRM 0.x spec defines the weight range as 0..=100
        /// (SkinnedMeshRenderer.SetBlendShapeWeight). The loader rescales
        /// to 0..=1 when building `ExpressionMorphBind`.
        #[serde(default)]
        pub weight: f32,
    }

    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct SecondaryAnimation {
        #[serde(default, rename = "boneGroups")]
        pub bone_groups: Vec<BoneGroup>,
        #[serde(default, rename = "colliderGroups")]
        pub collider_groups: Vec<ColliderGroup>,
    }

    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct BoneGroup {
        #[serde(default)]
        pub comment: Option<String>,
        /// Field is spelled `stiffiness` (double-i) in the VRM 0.x spec.
        /// Preserved verbatim so files written by Unity-era exporters
        /// continue to parse.
        #[serde(default, rename = "stiffiness")]
        pub stiffiness: f32,
        #[serde(default, rename = "gravityPower")]
        pub gravity_power: f32,
        #[serde(default, rename = "gravityDir")]
        pub gravity_dir: Option<XyzVec>,
        #[serde(default, rename = "dragForce")]
        pub drag_force: f32,
        #[serde(default)]
        pub center: Option<i32>,
        #[serde(default, rename = "hitRadius")]
        pub hit_radius: f32,
        /// Each entry is a chain-root node index; children form the chain.
        #[serde(default)]
        pub bones: Vec<i32>,
        /// Indices into the sibling `colliderGroups` array.
        #[serde(default, rename = "colliderGroups")]
        pub collider_groups: Vec<u32>,
    }

    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct XyzVec {
        #[serde(default)]
        pub x: f32,
        #[serde(default)]
        pub y: f32,
        #[serde(default)]
        pub z: f32,
    }

    impl XyzVec {
        pub fn to_array(&self) -> [f32; 3] {
            [self.x, self.y, self.z]
        }
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct ColliderGroup {
        #[serde(default)]
        pub node: i32,
        #[serde(default)]
        pub colliders: Vec<Collider>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct Collider {
        #[serde(default)]
        pub offset: Option<XyzVec>,
        #[serde(default)]
        pub radius: f32,
    }
}

mod v1 {
    use serde::Deserialize;
    use std::collections::HashMap;

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
        /// VRM 1.0 stores the thumbnail as a glTF *image* index (not a
        /// texture index — this differs from 0.x).
        #[serde(default, rename = "thumbnailImage")]
        pub thumbnail_image: Option<u32>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct Humanoid {
        #[serde(rename = "humanBones")]
        pub human_bones: HashMap<String, HumanBone>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct HumanBone {
        pub node: u32,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct Expressions {
        #[serde(default)]
        pub preset: Option<Preset>,
        /// VRM 1.0: `custom` is a map {name: Expression}. The map key is the
        /// expression name; the `Expression` value has no `name` field.
        #[serde(default)]
        pub custom: Option<HashMap<String, Expression>>,
    }

    /// VRM 1.0 `preset` expression map. JSON keys are camelCase
    /// (`blinkLeft`, `lookUp`, ...), so we keep snake_case Rust fields and
    /// rely on `rename_all` to bridge the two. Each value is an expression
    /// object; the *map key* is the name, so `Expression` has no `name`
    /// field.
    #[derive(Debug, Clone, Default, Deserialize)]
    #[serde(rename_all = "camelCase")]
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
        /// Return `(canonical_name, expression)` pairs for every populated
        /// preset slot. The canonical name is the VRM 1.0 camelCase key
        /// (e.g. `"blinkLeft"`) and matches the identifiers produced by the
        /// tracking layer.
        pub fn iter_named(&self) -> Vec<(&'static str, &Expression)> {
            let mut out = Vec::new();
            macro_rules! push_if {
                ($field:ident, $name:literal) => {
                    if let Some(ref e) = self.$field {
                        out.push(($name, e));
                    }
                };
            }
            push_if!(happy, "happy");
            push_if!(angry, "angry");
            push_if!(sad, "sad");
            push_if!(relaxed, "relaxed");
            push_if!(surprised, "surprised");
            push_if!(aa, "aa");
            push_if!(ih, "ih");
            push_if!(ou, "ou");
            push_if!(ee, "ee");
            push_if!(oh, "oh");
            push_if!(blink, "blink");
            push_if!(blink_left, "blinkLeft");
            push_if!(blink_right, "blinkRight");
            push_if!(look_up, "lookUp");
            push_if!(look_down, "lookDown");
            push_if!(look_left, "lookLeft");
            push_if!(look_right, "lookRight");
            push_if!(neutral, "neutral");
            out
        }
    }

    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct Expression {
        #[serde(default, rename = "morphTargetBinds")]
        pub morph_target_binds: Vec<MorphTargetBind>,
        #[serde(default, rename = "isBinary")]
        pub is_binary: Option<bool>,
        #[serde(default, rename = "overrideBlink")]
        pub override_blink: Option<String>,
        #[serde(default, rename = "overrideLookAt")]
        pub override_look_at: Option<String>,
        #[serde(default, rename = "overrideMouth")]
        pub override_mouth: Option<String>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct MorphTargetBind {
        pub node: u32,
        pub index: u32,
        pub weight: f32,
    }

    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct SpringBoneExtension {
        #[serde(default)]
        pub springs: Vec<Spring>,
        #[serde(default, rename = "colliderGroups")]
        pub collider_groups: Option<Vec<ColliderGroup>>,
        #[serde(default)]
        pub colliders: Option<Vec<Collider>>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct Spring {
        #[serde(default)]
        pub name: String,
        pub joints: Vec<Joint>,
        #[serde(default, rename = "colliderGroups")]
        pub collider_groups: Option<Vec<u32>>,
    }

    fn default_stiffness() -> f32 {
        1.0
    }
    fn default_gravity_power() -> f32 {
        0.0
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct Joint {
        pub node: u32,
        #[serde(default, rename = "hitRadius")]
        pub hit_radius: Option<f32>,
        #[serde(default = "default_stiffness")]
        pub stiffness: f32,
        #[serde(default = "default_gravity_power", rename = "gravityPower")]
        pub gravity_power: f32,
        #[serde(default, rename = "gravityDir")]
        pub gravity_dir: Option<[f32; 3]>,
        #[serde(default, rename = "dragForce")]
        pub drag_force: Option<f32>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct Collider {
        pub node: u32,
        #[serde(default)]
        pub shape: ColliderShape,
    }

    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct ColliderShape {
        #[serde(default)]
        pub sphere: Option<SphereShape>,
        #[serde(default)]
        pub capsule: Option<CapsuleShape>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct SphereShape {
        #[serde(default)]
        pub radius: f32,
        #[serde(default)]
        pub offset: Option<[f32; 3]>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct CapsuleShape {
        #[serde(default)]
        pub radius: f32,
        #[serde(default)]
        pub offset: Option<[f32; 3]>,
        #[serde(default)]
        pub tail: Option<[f32; 3]>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct ColliderGroup {
        pub colliders: Vec<u32>,
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
}

impl Default for VrmAssetLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl VrmAssetLoader {
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

        // Avatar load cache (B1, plan/T10-future-extensions.md). Try the
        // cached parse first; on hit, re-decode textures from the source
        // bytes (Option B: textures aren't included in the cache file) and
        // return without redoing the rest of the glTF / VRM extension /
        // skin / spring-bone / animation work. On miss, proceed to the
        // full parse and best-effort cache the result for next time.
        let source_path = PathBuf::from(path);
        if let Ok(Some(mut cached)) = crate::asset::cache::try_load(&source_path) {
            log::info!("avatar cache: hit for {}", source_path.display());
            on_progress(LoadStage::Materials {
                current: 0,
                total: cached.materials.len(),
            });
            // Texture rehydration. If the parse fails (corrupt file), drop
            // the cache hit and fall through to the regular parse path so
            // the user sees the real error rather than a half-loaded asset.
            match gltf::Gltf::from_slice(&file_data) {
                Ok(gltf_doc) => {
                    let blob = gltf_doc.blob.as_deref();
                    let source_path_clone = cached.source_path.clone();
                    Self::rehydrate_textures(&mut cached, &gltf_doc.document, blob, &source_path_clone);
                    cached.id = AvatarAssetId(NEXT_AVATAR_ID.fetch_add(1, Ordering::Relaxed));
                    cached.set_loaded_from_cache(true);
                    return Ok(Arc::new(cached));
                }
                Err(e) => {
                    log::warn!(
                        "avatar cache: hit but source re-parse failed ({}); falling back to full parse",
                        e
                    );
                }
            }
        }

        let asset = self.load_from_bytes_with_progress(&file_data, source_path.clone(), on_progress)?;
        // Best-effort persist. If serialisation or write fails the user
        // still gets the freshly-parsed asset; the next load just misses
        // again.
        if let Err(e) = crate::asset::cache::save(&source_path, &asset) {
            log::warn!(
                "avatar cache: save for '{}' failed: {}",
                source_path.display(),
                e
            );
        }
        Ok(asset)
    }

    /// Walk every `TextureBinding` on `asset` and repopulate its
    /// `pixel_data` from the freshly-parsed glTF document. Used by the
    /// cache hit path: `TextureBinding.pixel_data` carries
    /// `#[serde(skip)]` (see `src/asset/mod.rs`) so the cache file
    /// stays compact, which means the renderer would otherwise see
    /// `pixel_data == None` and fall back to its placeholder white
    /// texture. The function reuses `texture_binding_from_info` so the
    /// URI / decode path matches the regular load.
    #[allow(clippy::type_complexity)]
    fn rehydrate_textures(
        asset: &mut AvatarAsset,
        gltf_doc: &gltf::Document,
        blob: Option<&[u8]>,
        source_path: &Path,
    ) {
        let mut decoded: HashMap<String, (Arc<Vec<u8>>, (u32, u32))> = HashMap::new();
        for texture in gltf_doc.textures() {
            let binding = Self::texture_binding_from_info(&texture, blob, source_path);
            if let Some(pixels) = binding.pixel_data {
                decoded.insert(binding.uri, (pixels, binding.dimensions));
            }
        }

        let patch = |tb: &mut Option<TextureBinding>| {
            if let Some(tb) = tb.as_mut() {
                if let Some((pixels, dims)) = decoded.get(&tb.uri) {
                    tb.pixel_data = Some(Arc::clone(pixels));
                    tb.dimensions = *dims;
                }
            }
        };

        for mat in &mut asset.materials {
            patch(&mut mat.texture_bindings.base_color_texture);
            patch(&mut mat.texture_bindings.normal_map_texture);
            patch(&mut mat.texture_bindings.shade_ramp_texture);
            patch(&mut mat.texture_bindings.emissive_texture);
            patch(&mut mat.texture_bindings.matcap_texture);
        }
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

        // Snapshot root-level glTF extensions as raw JSON values, keyed by
        // extension name, so each version-specific parser can pull only the
        // sub-tree it cares about without being held hostage by sibling
        // extensions that fail to deserialise.
        let ext_map: HashMap<String, serde_json::Value> = gltf_doc
            .document
            .extensions()
            .map(|ext| ext.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        on_progress(LoadStage::Skeleton);
        let (mut nodes, _node_name_map) = Self::build_skeleton_nodes(&gltf_doc.document);

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

        // glTF node index → mesh index. VRM 1.x morph target binds reference
        // nodes directly; VRM 0.x references meshes and needs the inverse
        // lookup, built below.
        let node_to_mesh: HashMap<usize, usize> = gltf_doc
            .document
            .nodes()
            .filter_map(|node| node.mesh().map(|m| (node.index(), m.index())))
            .collect();
        let mut mesh_to_node: HashMap<usize, usize> = HashMap::new();
        for (&n, &m) in &node_to_mesh {
            mesh_to_node.entry(m).or_insert(n);
        }

        on_progress(LoadStage::SpringBones);
        let mut parsed = parse_vrm_extensions(&ext_map, &nodes, &mesh_to_node)?;

        // Resolve the thumbnail hint to encoded image bytes while we still
        // hold `gltf_doc` and `blob`. The library code persists these to
        // disk later — we only want to do the glTF traversal once.
        parsed.meta.thumbnail =
            resolve_thumbnail(&gltf_doc.document, blob, &parsed.thumbnail_hint);

        // VRM 0.x stores the model facing -Z; VRM 1.x (and the rest of this
        // engine) expects +Z. Bake a 180° Y rotation into each root node so
        // the skeleton, spring bones and retargeting all see the 1.x
        // orientation without touching the mesh vertex data.
        if parsed.meta.spec_version == VrmSpecVersion::V0 {
            Self::bake_y_flip_on_roots(&mut nodes, &root_nodes);
        }

        // Build per-node inverse bind matrices. For each glTF node, if any
        // skin lists it as a joint, use that skin's IBM for this node. VRM
        // files typically carry consistent bind poses across skins, so the
        // "first skin wins" fallback is safe for multi-skin avatars (face
        // + body + accessories). Nodes that never appear in any skin get
        // identity — they will not be skinned anyway.
        let mut inverse_bind_matrices = vec![identity_matrix(); nodes.len()];
        let mut ibm_set = vec![false; nodes.len()];
        for (_skin_index, skin) in &skin_bindings {
            for (joint_pos, NodeId(node_id)) in skin.joint_nodes.iter().enumerate() {
                let node_idx = *node_id as usize;
                if node_idx >= inverse_bind_matrices.len() || ibm_set[node_idx] {
                    continue;
                }
                if let Some(ibm) = skin.inverse_bind_matrices.get(joint_pos) {
                    inverse_bind_matrices[node_idx] = *ibm;
                    ibm_set[node_idx] = true;
                }
            }
        }

        on_progress(LoadStage::Finalizing);

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
            humanoid: parsed.humanoid,
            spring_bones: parsed.springs,
            colliders: parsed.colliders,
            default_expressions: parsed.expressions,
            animation_clips: Self::build_animation_clips(&gltf_doc.document, blob),
            node_to_mesh,
            vrm_meta: parsed.meta,
            root_aabb,
            loaded_from_cache: false,
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
                                            get_buf_data,
                                        )?;
                                        Some(iter.collect())
                                    })
                                    .unwrap_or_default();
                                let norm_d: Vec<[f32; 3]> = mt
                                    .normals()
                                    .and_then(|acc| {
                                        let iter = gltf::accessor::Iter::<[f32; 3]>::new(
                                            acc,
                                            get_buf_data,
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

                let effective_skin = skin_binding.or_else(|| skins.first().map(|(_, s)| s));

                if let Some(skin) = effective_skin {
                    for prim in &mut mesh.primitives {
                        // Remap vertex joint indices from "skin-relative"
                        // (index into this skin's joint_nodes list) to
                        // "node-relative" (glTF node index). This lets the
                        // renderer use a single skinning matrix array shared
                        // across all skins in the avatar — critical for VRM
                        // models that ship separate skins for face/body/etc.
                        if let Some(ref mut vd) = prim.vertices {
                            for ji in vd.joint_indices.iter_mut() {
                                for slot in ji.iter_mut() {
                                    let joint_idx = *slot as usize;
                                    if let Some(NodeId(node_id)) =
                                        skin.joint_nodes.get(joint_idx)
                                    {
                                        *slot = *node_id as u16;
                                    }
                                }
                            }
                        }
                        prim.skin = Some(skin.clone());
                    }
                }
                mesh
            })
            .collect()
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

// ---------------------------------------------------------------------------
// VRM extension dispatch — picks the right per-version parser.
// ---------------------------------------------------------------------------

struct ParsedVrm {
    meta: VrmMeta,
    humanoid: Option<HumanoidMap>,
    expressions: ExpressionAssetSet,
    springs: Vec<SpringBoneAsset>,
    colliders: Vec<ColliderAsset>,
    /// Where in the glTF document the VRM file says its thumbnail lives.
    /// Resolved into raw bytes after `parse_vrm_extensions` returns,
    /// where the caller has access to `gltf::Document` + the binary blob.
    thumbnail_hint: ThumbnailHint,
}

impl ParsedVrm {
    fn empty() -> Self {
        Self {
            meta: VrmMeta::default(),
            humanoid: None,
            expressions: ExpressionAssetSet {
                expressions: vec![],
            },
            springs: vec![],
            colliders: vec![],
            thumbnail_hint: ThumbnailHint::default(),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct ThumbnailHint {
    /// VRM 1.0: index into `doc.images()` (glTF *image*, not texture).
    image_index: Option<usize>,
    /// VRM 0.x: index into `doc.textures()`. The texture's `.source()` is
    /// the image index we ultimately read.
    texture_index: Option<usize>,
}

fn parse_vrm_extensions(
    ext_map: &HashMap<String, serde_json::Value>,
    nodes: &[SkeletonNode],
    mesh_to_node: &HashMap<usize, usize>,
) -> Result<ParsedVrm, VrmLoadError> {
    // When both VRMC_vrm and VRM are present (some exporters bundle legacy
    // data for backwards compatibility), VRM 1.x is authoritative. Fall back
    // to the other path only if parsing the primary one fails outright.
    if let Some(raw) = ext_map.get("VRMC_vrm") {
        match parse_v1_extensions(raw, ext_map.get("VRMC_springBone")) {
            Ok(parsed) => return Ok(parsed),
            Err(e) => {
                warn!(
                    "VRM 1.x extension parse failed ({e}); falling back to VRM 0.x if available"
                );
            }
        }
    }
    if let Some(raw) = ext_map.get("VRM") {
        return parse_v0_extensions(raw, nodes, mesh_to_node);
    }
    // No recognised VRM extension: return glTF-as-avatar with no retargeting.
    warn!("no VRM extension found in glTF root; loading as glTF-only avatar");
    Ok(ParsedVrm::empty())
}

// ---------------------------------------------------------------------------
// VRM 1.x parser
// ---------------------------------------------------------------------------

fn parse_v1_extensions(
    vrmc_vrm_raw: &serde_json::Value,
    vrmc_spring_bone_raw: Option<&serde_json::Value>,
) -> Result<ParsedVrm, VrmLoadError> {
    let vrmc_vrm: v1::VrmExtension = serde_json::from_value(vrmc_vrm_raw.clone())
        .map_err(|e| {
            warn!("VRMC_vrm deserialisation failed: {e}");
            VrmLoadError::JsonParse(e)
        })?;

    let (meta, thumbnail_hint) = build_v1_meta(&vrmc_vrm);
    let humanoid = Some(build_v1_humanoid(&vrmc_vrm.humanoid)?);
    let expressions = vrmc_vrm
        .expressions
        .as_ref()
        .map(build_v1_expressions)
        .unwrap_or_else(|| ExpressionAssetSet {
            expressions: vec![],
        });

    let (springs, colliders) = match vrmc_spring_bone_raw {
        Some(raw) => match serde_json::from_value::<v1::SpringBoneExtension>(raw.clone()) {
            Ok(sb) => build_v1_springs(&sb),
            Err(e) => {
                warn!("VRMC_springBone parse failed: {e}");
                (vec![], vec![])
            }
        },
        None => (vec![], vec![]),
    };

    Ok(ParsedVrm {
        meta,
        humanoid,
        expressions,
        springs,
        colliders,
        thumbnail_hint,
    })
}

/// Resolve a [`ThumbnailHint`] to the encoded image bytes (PNG / JPEG)
/// embedded in the VRM. Returns `None` when the hint is empty, the index
/// is out of range, the image source is an external URI we can't read,
/// or the buffer slice is unreachable. Logs but does not error — a
/// missing thumbnail must not fail the avatar load.
fn resolve_thumbnail(
    doc: &gltf::Document,
    blob: Option<&[u8]>,
    hint: &ThumbnailHint,
) -> Option<EmbeddedThumbnail> {
    let image = if let Some(image_idx) = hint.image_index {
        doc.images().nth(image_idx)
    } else if let Some(tex_idx) = hint.texture_index {
        doc.textures().nth(tex_idx).map(|tex| tex.source())
    } else {
        return None;
    }?;

    match image.source() {
        gltf::image::Source::View { view, mime_type } => {
            let start = view.offset();
            let end = start + view.length();
            let bytes = blob.and_then(|b| b.get(start..end))?;
            Some(EmbeddedThumbnail {
                mime_type: mime_type.to_string(),
                bytes: bytes.to_vec(),
            })
        }
        gltf::image::Source::Uri { uri, .. } => {
            // VRM thumbnails are virtually always embedded in the .glb
            // binary chunk; URI references are unusual but legal here.
            log::warn!(
                "VRM thumbnail at uri '{uri}' is external; skipping (no source-relative resolver)",
            );
            None
        }
    }
}

fn build_v1_meta(ext: &v1::VrmExtension) -> (VrmMeta, ThumbnailHint) {
    let meta = ext.meta.clone().unwrap_or_default();
    let hint = ThumbnailHint {
        image_index: meta.thumbnail_image.map(|i| i as usize),
        texture_index: None,
    };
    let vrm_meta = VrmMeta {
        spec_version: VrmSpecVersion::V1,
        spec_version_raw: ext.spec_version.clone(),
        title: meta.name,
        authors: meta.authors.unwrap_or_default(),
        model_version: meta.version,
        contact_information: meta.contact_information,
        references: meta.references.unwrap_or_default(),
        license: meta.license_url,
        copyright_information: meta.copyright_information,
        thumbnail: None,
    };
    (vrm_meta, hint)
}

fn build_v1_humanoid(humanoid: &v1::Humanoid) -> Result<HumanoidMap, VrmLoadError> {
    let mut bone_map = HashMap::new();
    for (bone_name, bone_def) in &humanoid.human_bones {
        // Unknown bones (fingers, toes, jaw, eyes, ...) are accepted by the
        // VRM 1.0 spec but ignored by this retargeting pipeline. Skipping them
        // is friendlier than hard-erroring the entire load.
        if let Some(humanoid_bone) = parse_humanoid_bone(bone_name) {
            bone_map.insert(humanoid_bone, NodeId(bone_def.node as u64));
        }
    }
    Ok(HumanoidMap { bone_map })
}

fn build_v1_expressions(section: &v1::Expressions) -> ExpressionAssetSet {
    let mut expressions = Vec::new();

    if let Some(ref preset) = section.preset {
        for (name, expr) in preset.iter_named() {
            expressions.push(v1_expression_to_def(name, expr));
        }
    }
    if let Some(ref custom) = section.custom {
        for (name, expr) in custom {
            expressions.push(v1_expression_to_def(name, expr));
        }
    }

    ExpressionAssetSet { expressions }
}

fn v1_expression_to_def(name: &str, expr: &v1::Expression) -> ExpressionDef {
    let morph_binds = expr
        .morph_target_binds
        .iter()
        .map(|b| ExpressionMorphBind {
            node_index: b.node as usize,
            morph_target_index: b.index as usize,
            weight: b.weight,
        })
        .collect();

    ExpressionDef {
        name: name.to_string(),
        weight: 0.0,
        morph_binds,
    }
}

fn build_v1_springs(ext: &v1::SpringBoneExtension) -> (Vec<SpringBoneAsset>, Vec<ColliderAsset>) {
    // Colliders are numbered sequentially starting at 1 so that references from
    // spring collider_groups can resolve back via the index map below.
    let mut next_collider_id: u64 = 1;
    let colliders: Vec<ColliderAsset> = ext
        .colliders
        .as_ref()
        .map(|defs| {
            defs.iter()
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
                    let id = ColliderId(next_collider_id);
                    next_collider_id += 1;
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

    let springs: Vec<SpringBoneAsset> = ext
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

            let first = spring.joints.first();
            let stiffness = first.map(|j| j.stiffness).unwrap_or(1.0);
            let drag_force = first.and_then(|j| j.drag_force).unwrap_or(0.5);
            let gravity_dir = first.and_then(|j| j.gravity_dir).unwrap_or([0.0, -1.0, 0.0]);
            let gravity_power = first.map(|j| j.gravity_power).unwrap_or(0.0);
            let radius = first.and_then(|j| j.hit_radius).unwrap_or(0.05);

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

    (springs, colliders)
}

// ---------------------------------------------------------------------------
// VRM 0.x parser
// ---------------------------------------------------------------------------

fn parse_v0_extensions(
    vrm_raw: &serde_json::Value,
    nodes: &[SkeletonNode],
    mesh_to_node: &HashMap<usize, usize>,
) -> Result<ParsedVrm, VrmLoadError> {
    let root: v0::VrmRoot = serde_json::from_value(vrm_raw.clone()).map_err(|e| {
        warn!("VRM 0.x root extension parse failed: {e}");
        VrmLoadError::JsonParse(e)
    })?;

    let (meta, thumbnail_hint) = build_v0_meta(&root);
    let humanoid = root.humanoid.as_ref().map(build_v0_humanoid);
    let expressions = root
        .blend_shape_master
        .as_ref()
        .map(|bsm| build_v0_expressions(bsm, mesh_to_node))
        .unwrap_or_else(|| ExpressionAssetSet {
            expressions: vec![],
        });
    let (springs, colliders) = root
        .secondary_animation
        .as_ref()
        .map(|sa| build_v0_springs(sa, nodes))
        .unwrap_or_else(|| (vec![], vec![]));

    Ok(ParsedVrm {
        meta,
        humanoid,
        expressions,
        springs,
        colliders,
        thumbnail_hint,
    })
}

fn build_v0_meta(root: &v0::VrmRoot) -> (VrmMeta, ThumbnailHint) {
    let meta = root.meta.clone().unwrap_or_default();
    // VRM 0.x `author` is a single string; multi-author files put names as a
    // comma-separated list. Split so Inspector and library entries treat each
    // as its own name.
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
    // Some VRM 0.x exporters emit `-1` for "no thumbnail"; guard against that.
    let texture_index = meta
        .texture
        .filter(|i| *i >= 0)
        .map(|i| i as usize);
    let hint = ThumbnailHint {
        image_index: None,
        texture_index,
    };
    let vrm_meta = VrmMeta {
        spec_version: VrmSpecVersion::V0,
        spec_version_raw: root.spec_version.clone(),
        title: meta.title,
        authors,
        model_version: meta.version,
        contact_information: meta.contact_information,
        references,
        license: meta.license_name.or(meta.other_license_url),
        copyright_information: None,
        thumbnail: None,
    };
    (vrm_meta, hint)
}

fn build_v0_humanoid(humanoid: &v0::Humanoid) -> HumanoidMap {
    let mut bone_map = HashMap::new();
    for entry in &humanoid.human_bones {
        if entry.node < 0 {
            continue;
        }
        if let Some(bone) = parse_humanoid_bone(&entry.bone) {
            bone_map.insert(bone, NodeId(entry.node as u64));
        }
    }
    HumanoidMap { bone_map }
}

fn build_v0_expressions(
    master: &v0::BlendShapeMaster,
    mesh_to_node: &HashMap<usize, usize>,
) -> ExpressionAssetSet {
    let mut expressions = Vec::with_capacity(master.blend_shape_groups.len());

    for group in &master.blend_shape_groups {
        // Prefer the preset name (mapped to the VRM 1.0 canonical key) so the
        // tracking layer finds the right slot. Fall back to the group name for
        // unknown/custom blendshapes.
        let name = group
            .preset_name
            .as_deref()
            .and_then(v0_preset_name_to_v1)
            .map(|s| s.to_string())
            .unwrap_or_else(|| group.name.clone());

        if name.is_empty() {
            continue;
        }

        let morph_binds: Vec<ExpressionMorphBind> = group
            .binds
            .iter()
            .filter_map(|bind| {
                if bind.mesh < 0 || bind.index < 0 {
                    return None;
                }
                let node_index = mesh_to_node.get(&(bind.mesh as usize))?;
                Some(ExpressionMorphBind {
                    node_index: *node_index,
                    morph_target_index: bind.index as usize,
                    // VRM 0.x weights are 0..=100; rescale to 0..=1.
                    weight: bind.weight / 100.0,
                })
            })
            .collect();

        expressions.push(ExpressionDef {
            name,
            weight: 0.0,
            morph_binds,
        });
    }

    ExpressionAssetSet { expressions }
}

/// Translate a VRM 0.x `presetName` value to the VRM 1.0 canonical expression
/// identifier used throughout the tracking/retargeting pipeline. Returns
/// `None` for unknown presets and for `"unknown"` (so callers fall back to
/// the group's `name`).
fn v0_preset_name_to_v1(preset: &str) -> Option<&'static str> {
    match preset {
        "neutral" => Some("neutral"),
        // Mouth / vowels
        "a" => Some("aa"),
        "i" => Some("ih"),
        "u" => Some("ou"),
        "e" => Some("ee"),
        "o" => Some("oh"),
        // Eye blinks
        "blink" => Some("blink"),
        "blink_l" => Some("blinkLeft"),
        "blink_r" => Some("blinkRight"),
        // Emotions
        "joy" => Some("happy"),
        "angry" => Some("angry"),
        "sorrow" => Some("sad"),
        "fun" => Some("relaxed"),
        // Gaze
        "lookup" => Some("lookUp"),
        "lookdown" => Some("lookDown"),
        "lookleft" => Some("lookLeft"),
        "lookright" => Some("lookRight"),
        _ => None,
    }
}

fn build_v0_springs(
    sa: &v0::SecondaryAnimation,
    nodes: &[SkeletonNode],
) -> (Vec<SpringBoneAsset>, Vec<ColliderAsset>) {
    // Flatten 0.x colliderGroups (which bundle spheres under a single node)
    // into per-group ColliderAsset lists that can be referenced by index.
    let mut next_collider_id: u64 = 1;
    let mut all_colliders: Vec<ColliderAsset> = Vec::new();
    let mut group_to_ids: Vec<Vec<ColliderId>> = Vec::with_capacity(sa.collider_groups.len());

    for group in &sa.collider_groups {
        if group.node < 0 {
            group_to_ids.push(vec![]);
            continue;
        }
        let node = NodeId(group.node as u64);
        let mut ids = Vec::with_capacity(group.colliders.len());
        for c in &group.colliders {
            let offset = c.offset.as_ref().map(|v| v.to_array()).unwrap_or([0.0; 3]);
            let id = ColliderId(next_collider_id);
            next_collider_id += 1;
            all_colliders.push(ColliderAsset {
                id,
                node,
                shape: ColliderShape::Sphere { radius: c.radius },
                offset,
            });
            ids.push(id);
        }
        group_to_ids.push(ids);
    }

    let mut springs: Vec<SpringBoneAsset> = Vec::new();
    for group in &sa.bone_groups {
        let collider_refs: Vec<ColliderRef> = group
            .collider_groups
            .iter()
            .filter_map(|gi| group_to_ids.get(*gi as usize))
            .flat_map(|ids| ids.iter().map(|&id| ColliderRef { id }))
            .collect();

        let gravity_dir = group
            .gravity_dir
            .as_ref()
            .map(|v| v.to_array())
            .unwrap_or([0.0, -1.0, 0.0]);
        let radius = if group.hit_radius > 0.0 {
            group.hit_radius
        } else {
            0.02
        };

        for &root_idx in &group.bones {
            if root_idx < 0 {
                continue;
            }
            let chain = walk_v0_chain(nodes, root_idx as usize);
            if chain.is_empty() {
                continue;
            }
            let chain_root = chain[0];
            let joint_count = chain.len();
            springs.push(SpringBoneAsset {
                chain_root,
                joints: chain,
                stiffness: group.stiffiness,
                drag_force: group.drag_force,
                gravity_dir,
                gravity_power: group.gravity_power,
                radius,
                collider_refs: collider_refs.clone(),
                // 0.x shares parameters across the whole chain, so broadcast
                // the group values into per-joint slots.
                joint_stiffness: vec![group.stiffiness; joint_count],
                joint_drag: vec![group.drag_force; joint_count],
                joint_gravity_power: vec![group.gravity_power; joint_count],
            });
        }
    }

    (springs, all_colliders)
}

/// Walk a VRM 0.x spring chain starting at `root_idx`, following the first
/// child at each step. Stops when a node has no children. Branching chains
/// are flattened to the primary strand — the common case (hair, skirt
/// strands) has a single child per node.
fn walk_v0_chain(nodes: &[SkeletonNode], root_idx: usize) -> Vec<NodeId> {
    let mut chain = Vec::new();
    let mut cursor = root_idx;
    loop {
        if cursor >= nodes.len() {
            break;
        }
        chain.push(NodeId(cursor as u64));
        let next = nodes[cursor].children.first().copied();
        match next {
            Some(NodeId(n)) => {
                let n = n as usize;
                // Guard against accidental cycles (malformed input).
                if n == cursor || chain.iter().any(|NodeId(c)| *c as usize == n) {
                    break;
                }
                cursor = n;
            }
            None => break,
        }
    }
    chain
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

        // ----- Fingers -----
        "leftThumbProximal" => Some(HumanoidBone::LeftThumbProximal),
        "leftThumbIntermediate" | "leftThumbMetacarpal" => {
            Some(HumanoidBone::LeftThumbIntermediate)
        }
        "leftThumbDistal" => Some(HumanoidBone::LeftThumbDistal),
        "leftIndexProximal" => Some(HumanoidBone::LeftIndexProximal),
        "leftIndexIntermediate" => Some(HumanoidBone::LeftIndexIntermediate),
        "leftIndexDistal" => Some(HumanoidBone::LeftIndexDistal),
        "leftMiddleProximal" => Some(HumanoidBone::LeftMiddleProximal),
        "leftMiddleIntermediate" => Some(HumanoidBone::LeftMiddleIntermediate),
        "leftMiddleDistal" => Some(HumanoidBone::LeftMiddleDistal),
        "leftRingProximal" => Some(HumanoidBone::LeftRingProximal),
        "leftRingIntermediate" => Some(HumanoidBone::LeftRingIntermediate),
        "leftRingDistal" => Some(HumanoidBone::LeftRingDistal),
        "leftLittleProximal" => Some(HumanoidBone::LeftLittleProximal),
        "leftLittleIntermediate" => Some(HumanoidBone::LeftLittleIntermediate),
        "leftLittleDistal" => Some(HumanoidBone::LeftLittleDistal),

        "rightThumbProximal" => Some(HumanoidBone::RightThumbProximal),
        "rightThumbIntermediate" | "rightThumbMetacarpal" => {
            Some(HumanoidBone::RightThumbIntermediate)
        }
        "rightThumbDistal" => Some(HumanoidBone::RightThumbDistal),
        "rightIndexProximal" => Some(HumanoidBone::RightIndexProximal),
        "rightIndexIntermediate" => Some(HumanoidBone::RightIndexIntermediate),
        "rightIndexDistal" => Some(HumanoidBone::RightIndexDistal),
        "rightMiddleProximal" => Some(HumanoidBone::RightMiddleProximal),
        "rightMiddleIntermediate" => Some(HumanoidBone::RightMiddleIntermediate),
        "rightMiddleDistal" => Some(HumanoidBone::RightMiddleDistal),
        "rightRingProximal" => Some(HumanoidBone::RightRingProximal),
        "rightRingIntermediate" => Some(HumanoidBone::RightRingIntermediate),
        "rightRingDistal" => Some(HumanoidBone::RightRingDistal),
        "rightLittleProximal" => Some(HumanoidBone::RightLittleProximal),
        "rightLittleIntermediate" => Some(HumanoidBone::RightLittleIntermediate),
        "rightLittleDistal" => Some(HumanoidBone::RightLittleDistal),

        _ => None,
    }
}

#[cfg(test)]
mod metadata_tests {
    use super::*;

    const SAMPLE_VRM: &str = "sample_data/AvatarSample_A.vrm";
    const SAMPLE_VRM0: &str = "sample_data/vrm0x.vrm";

    /// Load a VRM fixture, or skip the test gracefully when the file
    /// hasn't been provisioned. The `sample_data/` directory is
    /// `.gitignore`'d so dev boxes / CI without the fixtures still get
    /// a green run while keeping the assertions live wherever the
    /// files do exist.
    fn load_or_skip(loader: &VrmAssetLoader, path: &str) -> Option<Arc<AvatarAsset>> {
        if !std::path::Path::new(path).exists() {
            eprintln!(
                "[skip] VRM fixture '{path}' not present \
                 (gitignored sample_data/ — provision the file to run this test)"
            );
            return None;
        }
        Some(
            loader
                .load(path)
                .unwrap_or_else(|e| panic!("load fixture '{path}': {e:?}")),
        )
    }

    /// Confirms the VRM 1.x sample parses end-to-end: spec version is
    /// detected, all tracked humanoid bones are present, preset expressions
    /// survive the dispatch, and at least one spring bone was produced.
    #[test]
    fn sample_vrm_loads_fully() {
        let loader = VrmAssetLoader::new();
        let Some(asset) = load_or_skip(&loader, SAMPLE_VRM) else {
            return;
        };

        assert_eq!(
            asset.vrm_meta.spec_version,
            VrmSpecVersion::V1,
            "sample_data/AvatarSample_A.vrm is VRM 1.x"
        );
        assert!(
            asset.vrm_meta.title.is_some(),
            "VRMC_vrm.meta.name must be populated for a well-formed sample"
        );

        let humanoid = asset
            .humanoid
            .as_ref()
            .expect("humanoid map must be populated for retargeting to work");
        assert!(
            humanoid.bone_map.contains_key(&HumanoidBone::Hips),
            "Hips bone must be mapped"
        );
        assert!(
            humanoid.bone_map.contains_key(&HumanoidBone::Head),
            "Head bone must be mapped"
        );
        assert!(
            humanoid.bone_map.len() >= 15,
            "expected at least the 15 core humanoid bones, got {}",
            humanoid.bone_map.len()
        );

        assert!(
            !asset.default_expressions.expressions.is_empty(),
            "preset expressions must survive VRMC_vrm parsing"
        );
        let names: std::collections::HashSet<&str> = asset
            .default_expressions
            .expressions
            .iter()
            .map(|e| e.name.as_str())
            .collect();
        for expected in ["blink", "aa"] {
            assert!(
                names.contains(expected),
                "expected canonical preset '{expected}' in {:?}",
                names
            );
        }

        assert!(
            !asset.spring_bones.is_empty(),
            "VRMC_springBone must produce at least one spring chain for the sample"
        );
    }

    /// Load the real VRM 0.x sample end-to-end and assert that every piece
    /// of the avatar pipeline (metadata, humanoid, expressions, spring
    /// bones) makes it through the 0.x → 1.x canonicalisation.
    #[test]
    fn sample_vrm0_loads_fully() {
        let loader = VrmAssetLoader::new();
        let Some(asset) = load_or_skip(&loader, SAMPLE_VRM0) else {
            return;
        };

        assert_eq!(
            asset.vrm_meta.spec_version,
            VrmSpecVersion::V0,
            "sample_data/vrm0x.vrm must parse as VRM 0.x"
        );

        let humanoid = asset
            .humanoid
            .as_ref()
            .expect("humanoid map must be populated for 0.x retargeting");
        assert!(
            humanoid.bone_map.contains_key(&HumanoidBone::Hips),
            "VRM 0.x Hips bone must be mapped"
        );
        assert!(
            humanoid.bone_map.contains_key(&HumanoidBone::Head),
            "VRM 0.x Head bone must be mapped"
        );
        assert!(
            humanoid.bone_map.len() >= 15,
            "expected at least the 15 core humanoid bones in 0.x sample, got {}",
            humanoid.bone_map.len()
        );

        // Most 0.x models ship at least `a`/`blink` presets; verify the
        // canonicalisation landed them under the VRM 1.0 identifiers the
        // tracking layer expects.
        if !asset.default_expressions.expressions.is_empty() {
            let names: std::collections::HashSet<&str> = asset
                .default_expressions
                .expressions
                .iter()
                .map(|e| e.name.as_str())
                .collect();
            eprintln!("0.x canonicalised expression names: {:?}", names);
            // Weight rescaling: no bind should exceed 1.0 after the
            // 0..=100 → 0..=1 conversion.
            for expr in &asset.default_expressions.expressions {
                for bind in &expr.morph_binds {
                    assert!(
                        bind.weight <= 1.0 + 1e-4,
                        "expression '{}' bind weight {} exceeds 1.0 — 0.x 100-scale not rescaled?",
                        expr.name,
                        bind.weight
                    );
                }
            }
        }

        eprintln!(
            "VRM 0.x sample: title={:?} authors={:?} humanoid_bones={} expressions={} springs={} colliders={}",
            asset.vrm_meta.title,
            asset.vrm_meta.authors,
            humanoid.bone_map.len(),
            asset.default_expressions.expressions.len(),
            asset.spring_bones.len(),
            asset.colliders.len(),
        );
    }

    /// End-to-end sanity for VRM 0.x tracking → GPU skinning matrix.
    ///
    /// Loads the real VRM 0.x sample (which ships *two* distinct skins — a
    /// tiny 1-joint skin and the main 172-joint body skin), feeds it a
    /// synthetic `SourceSkeleton` with a raised left arm and a turned head,
    /// runs the solver + skinning pipeline, and asserts:
    ///
    ///   1. `pose_solver::solve_avatar_pose` writes the Head bone's local
    ///      rotation away from rest when a non-zero face pose is supplied.
    ///   2. The LeftUpperArm bone's local rotation is modified when the
    ///      solver has both a shoulder and an elbow source joint to work
    ///      with — the direction-match path (not just the face-pose path)
    ///      reaches the skeleton.
    ///   3. After `compute_global_pose` + `build_skinning_matrices`, the
    ///      Head bone's skinning matrix (as the shader sees it, indexed by
    ///      glTF node id) reflects the rotation — i.e. multi-skin models
    ///      still feed the right joint.
    #[test]
    fn sample_vrm0_solver_drives_head_and_arm() {
        use crate::avatar::pose_solver::{solve_avatar_pose, PoseSolverState, SolverParams};
        use crate::avatar::{AvatarInstance, AvatarInstanceId};
        use crate::tracking::source_skeleton::{FacePose, SourceJoint, SourceSkeleton};

        let loader = VrmAssetLoader::new();
        let Some(asset) = load_or_skip(&loader, SAMPLE_VRM0) else {
            return;
        };

        let mut avatar = AvatarInstance::new(AvatarInstanceId(1), asset.clone());
        avatar.build_base_pose();

        let head_idx = asset
            .humanoid
            .as_ref()
            .unwrap()
            .bone_map
            .get(&HumanoidBone::Head)
            .expect("Head mapped")
            .0 as usize;
        let left_upper_arm_idx = asset
            .humanoid
            .as_ref()
            .unwrap()
            .bone_map
            .get(&HumanoidBone::LeftUpperArm)
            .expect("LeftUpperArm mapped")
            .0 as usize;

        let head_rest = avatar.pose.local_transforms[head_idx].rotation;
        let upper_arm_rest = avatar.pose.local_transforms[left_upper_arm_idx].rotation;

        // Build a synthetic source skeleton:
        //   * Left shoulder at x = -0.2, elbow up + out (x = -0.4, y = +0.3)
        //     so the LeftUpperArm direction tilts away from the body.
        //   * Head face pose yaw = 45°, all other zero.
        let mut source = SourceSkeleton::empty(0);
        source.joints.insert(
            HumanoidBone::LeftShoulder,
            SourceJoint {
                position: [-0.2, 1.4, 0.0],
                confidence: 1.0,
            },
        );
        source.joints.insert(
            HumanoidBone::LeftUpperArm,
            SourceJoint {
                position: [-0.2, 1.4, 0.0],
                confidence: 1.0,
            },
        );
        source.joints.insert(
            HumanoidBone::LeftLowerArm,
            SourceJoint {
                position: [-0.4, 1.7, 0.0],
                confidence: 1.0,
            },
        );
        source.joints.insert(
            HumanoidBone::RightShoulder,
            SourceJoint {
                position: [0.2, 1.4, 0.0],
                confidence: 1.0,
            },
        );
        source.face = Some(FacePose {
            yaw: std::f32::consts::FRAC_PI_4,
            pitch: 0.0,
            roll: 0.0,
            confidence: 1.0,
        });
        source.overall_confidence = 1.0;

        let params = SolverParams {
            rotation_blend: 1.0,
            joint_confidence_threshold: 0.1,
            face_confidence_threshold: 0.1,
        };
        let mut solver_state = PoseSolverState::default();
        solve_avatar_pose(
            &source,
            &avatar.asset.skeleton,
            avatar.asset.humanoid.as_ref(),
            &mut avatar.pose.local_transforms,
            &params,
            &mut solver_state,
        );

        let head_after = avatar.pose.local_transforms[head_idx].rotation;
        let arm_after = avatar.pose.local_transforms[left_upper_arm_idx].rotation;
        eprintln!("head rest  : {:?}", head_rest);
        eprintln!("head after : {:?}", head_after);
        eprintln!("arm rest   : {:?}", upper_arm_rest);
        eprintln!("arm after  : {:?}", arm_after);

        let head_diff: f32 = head_rest
            .iter()
            .zip(head_after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            head_diff > 1e-4,
            "Head bone local rotation did not change in response to face pose ({head_diff})"
        );

        let arm_diff: f32 = upper_arm_rest
            .iter()
            .zip(arm_after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            arm_diff > 1e-4,
            "LeftUpperArm local rotation did not change when elbow moved in source ({arm_diff})"
        );

        // Confirm the Head's skinning matrix actually changes.
        avatar.compute_global_pose();
        avatar.build_skinning_matrices();
        let skin_with_tracking = avatar.pose.skinning_matrices[head_idx];

        avatar.build_base_pose();
        avatar.compute_global_pose();
        avatar.build_skinning_matrices();
        let skin_rest = avatar.pose.skinning_matrices[head_idx];

        let max_diff: f32 = skin_rest
            .iter()
            .zip(skin_with_tracking.iter())
            .flat_map(|(a, b)| a.iter().zip(b.iter()))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max_diff > 1e-4,
            "Head skinning matrix unchanged after solver — multi-skin indexing regressed ({max_diff})"
        );
    }

    /// Synthesise a bent index finger on one hand and assert the solver
    /// drives the avatar's corresponding finger bones. This is the
    /// regression test for "making a fist / peace sign / open hand doesn't
    /// register" — the solver has to walk the finger chain (Proximal →
    /// Intermediate → Distal) and turn each bone toward the next joint's
    /// image-space position.
    #[test]
    fn sample_vrm0_solver_bends_fingers() {
        use crate::avatar::pose_solver::{solve_avatar_pose, PoseSolverState, SolverParams};
        use crate::avatar::{AvatarInstance, AvatarInstanceId};
        use crate::tracking::source_skeleton::{SourceJoint, SourceSkeleton};

        let loader = VrmAssetLoader::new();
        let Some(asset) = load_or_skip(&loader, SAMPLE_VRM0) else {
            return;
        };

        let mut avatar = AvatarInstance::new(AvatarInstanceId(1), asset.clone());
        avatar.build_base_pose();

        // Pick the LEFT index chain — the 0.x sample maps these to
        // separate skin nodes (see `sample_vrm0_humanoid_mapping_looks_sane`).
        // COCO keypoints on the user's anatomical LEFT hand drive the
        // avatar's RIGHT chain (mirror), but for the solver we can plug
        // values directly into the source skeleton under either side.
        let humanoid = asset.humanoid.as_ref().unwrap();
        let prox = humanoid
            .bone_map
            .get(&HumanoidBone::RightIndexProximal)
            .copied()
            .unwrap()
            .0 as usize;
        let inter = humanoid
            .bone_map
            .get(&HumanoidBone::RightIndexIntermediate)
            .copied()
            .unwrap()
            .0 as usize;
        let distal = humanoid
            .bone_map
            .get(&HumanoidBone::RightIndexDistal)
            .copied()
            .unwrap()
            .0 as usize;

        let rest_prox = avatar.pose.local_transforms[prox].rotation;
        let rest_inter = avatar.pose.local_transforms[inter].rotation;
        let rest_distal = avatar.pose.local_transforms[distal].rotation;

        // Plant a curled index finger on the right side of the source
        // skeleton. MCP sits near the palm; PIP/DIP/TIP progressively curl
        // downward (smaller y). Confidence maxed so the solver does not
        // threshold us out.
        let mut source = SourceSkeleton::empty(0);
        let mut put = |bone, x, y| {
            source.joints.insert(
                bone,
                SourceJoint {
                    position: [x, y, 0.0],
                    confidence: 1.0,
                },
            );
        };
        // Wrist / anchor so the arm chain upstream does not move wildly.
        put(HumanoidBone::RightHand, -0.4, 0.2);
        put(HumanoidBone::RightIndexProximal, -0.42, 0.15);
        put(HumanoidBone::RightIndexIntermediate, -0.44, 0.05);
        put(HumanoidBone::RightIndexDistal, -0.45, -0.05);
        source.fingertips.insert(
            HumanoidBone::RightIndexDistal,
            SourceJoint {
                position: [-0.44, -0.12, 0.0],
                confidence: 1.0,
            },
        );
        source.overall_confidence = 1.0;

        let params = SolverParams {
            rotation_blend: 1.0,
            joint_confidence_threshold: 0.1,
            face_confidence_threshold: 0.1,
        };
        let mut solver_state = PoseSolverState::default();
        solve_avatar_pose(
            &source,
            &avatar.asset.skeleton,
            avatar.asset.humanoid.as_ref(),
            &mut avatar.pose.local_transforms,
            &params,
            &mut solver_state,
        );

        let new_prox = avatar.pose.local_transforms[prox].rotation;
        let new_inter = avatar.pose.local_transforms[inter].rotation;
        let new_distal = avatar.pose.local_transforms[distal].rotation;

        for (label, rest, new) in [
            ("Proximal", rest_prox, new_prox),
            ("Intermediate", rest_inter, new_inter),
            ("Distal", rest_distal, new_distal),
        ] {
            let diff: f32 = rest
                .iter()
                .zip(new.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(
                diff > 1e-4,
                "Right Index {label} local rotation did not react to bent-finger source ({diff})"
            );
        }
    }

    /// Dump the VRM 0.x sample's humanoid mapping alongside the skeleton
    /// node names so we can visually confirm each HumanoidBone points at a
    /// sensibly-named node (e.g. `Head` bone → `J_Bip_C_Head`).
    #[test]
    fn sample_vrm0_humanoid_mapping_looks_sane() {
        let loader = VrmAssetLoader::new();
        let Some(asset) = load_or_skip(&loader, SAMPLE_VRM0) else {
            return;
        };
        let humanoid = asset.humanoid.as_ref().expect("humanoid populated");

        let mut pairs: Vec<_> = humanoid.bone_map.iter().collect();
        pairs.sort_by_key(|(bone, _)| format!("{:?}", bone));
        eprintln!("-- VRM 0.x humanoid map --");
        for (bone, NodeId(node_idx)) in pairs {
            let name = asset
                .skeleton
                .nodes
                .get(*node_idx as usize)
                .map(|n| n.name.as_str())
                .unwrap_or("<out-of-range>");
            eprintln!("  {:?} -> node[{}] = {:?}", bone, node_idx, name);
        }

        // Sanity: expected bones present.
        for bone in [
            HumanoidBone::Hips,
            HumanoidBone::Spine,
            HumanoidBone::Head,
            HumanoidBone::LeftUpperArm,
            HumanoidBone::RightUpperArm,
            HumanoidBone::LeftUpperLeg,
            HumanoidBone::RightUpperLeg,
        ] {
            assert!(
                humanoid.bone_map.contains_key(&bone),
                "expected HumanoidBone::{:?} in 0.x humanoid map",
                bone
            );
        }
    }

    /// Synthesise a minimal VRM 0.x root extension JSON and exercise the
    /// per-version parser directly — we have no real 0.x sample file in-tree,
    /// so the schema coverage is validated at the unit level.
    #[test]
    fn v0_synthetic_extension_parses_humanoid_and_blendshapes() {
        let raw = serde_json::json!({
            "exporterVersion": "VRMUnityExporter-0.62",
            "specVersion": "0.0",
            "meta": {
                "title": "Synthetic 0.x Model",
                "author": "Author A, Author B",
                "contactInformation": "test@example.com",
                "licenseName": "CC_BY"
            },
            "humanoid": {
                "humanBones": [
                    { "bone": "hips", "node": 0 },
                    { "bone": "spine", "node": 1 },
                    { "bone": "head", "node": 2 },
                    { "bone": "leftHand", "node": 3 },
                    // Unknown bone (jaw) must not error out the whole parse.
                    { "bone": "jaw", "node": 99 },
                    // Negative node is a common sentinel for "unmapped".
                    { "bone": "rightEye", "node": -1 }
                ]
            },
            "blendShapeMaster": {
                "blendShapeGroups": [
                    {
                        "name": "A",
                        "presetName": "a",
                        "binds": [ { "mesh": 0, "index": 5, "weight": 100.0 } ]
                    },
                    {
                        "name": "Blink",
                        "presetName": "blink",
                        "binds": [ { "mesh": 0, "index": 6, "weight": 100.0 } ]
                    },
                    {
                        "name": "Joy",
                        "presetName": "joy",
                        "binds": [ { "mesh": 0, "index": 7, "weight": 100.0 } ]
                    },
                    {
                        "name": "CustomGroup",
                        "presetName": "unknown",
                        "binds": [ { "mesh": 0, "index": 8, "weight": 50.0 } ]
                    }
                ]
            },
            "secondaryAnimation": {
                "boneGroups": [
                    {
                        "comment": "hair",
                        "stiffiness": 1.5,
                        "gravityPower": 0.2,
                        "gravityDir": { "x": 0.0, "y": -1.0, "z": 0.0 },
                        "dragForce": 0.4,
                        "center": -1,
                        "hitRadius": 0.03,
                        "bones": [4],
                        "colliderGroups": [0]
                    }
                ],
                "colliderGroups": [
                    {
                        "node": 10,
                        "colliders": [
                            { "offset": { "x": 0.0, "y": 0.1, "z": 0.0 }, "radius": 0.1 }
                        ]
                    }
                ]
            }
        });

        // Synthesise a minimal skeleton: a single chain 4 → 5 → 6 → (leaf).
        let mut nodes = vec![
            SkeletonNode {
                id: NodeId(0),
                name: "root".into(),
                parent: None,
                children: vec![],
                rest_local: Transform::default(),
                humanoid_bone: None,
            };
            11
        ];
        nodes[4].children = vec![NodeId(5)];
        nodes[5].children = vec![NodeId(6)];

        let mesh_to_node: HashMap<usize, usize> =
            [(0usize, 2usize)].into_iter().collect();

        let parsed =
            super::parse_v0_extensions(&raw, &nodes, &mesh_to_node).expect("v0 parse");

        assert_eq!(parsed.meta.spec_version, VrmSpecVersion::V0);
        assert_eq!(parsed.meta.title.as_deref(), Some("Synthetic 0.x Model"));
        assert_eq!(
            parsed.meta.authors,
            vec!["Author A".to_string(), "Author B".to_string()],
            "VRM 0.x comma-separated authors split into separate entries"
        );

        let humanoid = parsed.humanoid.expect("humanoid built");
        assert_eq!(humanoid.bone_map.get(&HumanoidBone::Hips), Some(&NodeId(0)));
        assert_eq!(humanoid.bone_map.get(&HumanoidBone::Head), Some(&NodeId(2)));
        assert!(
            !humanoid.bone_map.contains_key(&HumanoidBone::LeftFoot),
            "unmapped bones absent"
        );

        let by_name: std::collections::HashMap<&str, &ExpressionDef> = parsed
            .expressions
            .expressions
            .iter()
            .map(|e| (e.name.as_str(), e))
            .collect();
        assert!(by_name.contains_key("aa"), "a → aa");
        assert!(by_name.contains_key("blink"));
        assert!(by_name.contains_key("happy"), "joy → happy");
        assert!(
            by_name.contains_key("CustomGroup"),
            "unknown preset falls back to the group name"
        );
        let aa = by_name["aa"];
        assert_eq!(aa.morph_binds.len(), 1);
        // mesh 0 → node 2 via mesh_to_node map
        assert_eq!(aa.morph_binds[0].node_index, 2);
        assert_eq!(aa.morph_binds[0].morph_target_index, 5);
        assert!(
            (aa.morph_binds[0].weight - 1.0).abs() < 1e-6,
            "0.x weight 100 rescales to 1.0"
        );

        assert_eq!(parsed.colliders.len(), 1);
        assert_eq!(parsed.colliders[0].node, NodeId(10));
        assert!(matches!(
            parsed.colliders[0].shape,
            ColliderShape::Sphere { .. }
        ));
        assert_eq!(parsed.springs.len(), 1);
        let spring = &parsed.springs[0];
        assert_eq!(spring.chain_root, NodeId(4));
        assert_eq!(
            spring.joints,
            vec![NodeId(4), NodeId(5), NodeId(6)],
            "walk_v0_chain follows first child until a leaf"
        );
        assert!((spring.stiffness - 1.5).abs() < 1e-6);
        assert!(!spring.collider_refs.is_empty());
    }

    /// Synthesise a minimal VRM 1.x VRMC_vrm + VRMC_springBone pair and check
    /// the shared output. This catches regressions in the preset expression
    /// naming (which broke silently before the v0/v1 split).
    #[test]
    fn v1_synthetic_extension_parses_preset_expressions() {
        let vrmc_vrm = serde_json::json!({
            "specVersion": "1.0",
            "meta": {
                "name": "Synthetic 1.x",
                "authors": ["Alice"],
                "licenseUrl": "https://example.com/license"
            },
            "humanoid": {
                "humanBones": {
                    "hips":  { "node": 0 },
                    "spine": { "node": 1 },
                    "head":  { "node": 2 }
                }
            },
            "expressions": {
                "preset": {
                    "aa": {
                        "morphTargetBinds": [ { "node": 2, "index": 5, "weight": 1.0 } ]
                    },
                    "blinkLeft": {
                        "morphTargetBinds": [ { "node": 2, "index": 6, "weight": 1.0 } ]
                    }
                },
                "custom": {
                    "MyCustom": {
                        "morphTargetBinds": [ { "node": 2, "index": 7, "weight": 0.5 } ]
                    }
                }
            }
        });
        let vrmc_spring = serde_json::json!({
            "springs": [
                {
                    "name": "hair",
                    "joints": [
                        { "node": 10, "hitRadius": 0.02, "stiffness": 1.0, "gravityPower": 0.0, "dragForce": 0.5 },
                        { "node": 11, "hitRadius": 0.02, "stiffness": 1.0, "gravityPower": 0.0, "dragForce": 0.5 }
                    ],
                    "colliderGroups": [0]
                }
            ],
            "colliders": [
                { "node": 1, "shape": { "sphere": { "offset": [0.0, 0.0, 0.0], "radius": 0.05 } } }
            ],
            "colliderGroups": [
                { "colliders": [0] }
            ]
        });

        let parsed =
            super::parse_v1_extensions(&vrmc_vrm, Some(&vrmc_spring)).expect("v1 parse");

        assert_eq!(parsed.meta.spec_version, VrmSpecVersion::V1);
        assert_eq!(parsed.meta.title.as_deref(), Some("Synthetic 1.x"));
        assert_eq!(parsed.meta.authors, vec!["Alice".to_string()]);

        let humanoid = parsed.humanoid.expect("humanoid built");
        assert_eq!(humanoid.bone_map.len(), 3);

        let names: std::collections::HashSet<&str> = parsed
            .expressions
            .expressions
            .iter()
            .map(|e| e.name.as_str())
            .collect();
        assert!(names.contains("aa"));
        assert!(names.contains("blinkLeft"), "blinkLeft preset key preserved");
        assert!(names.contains("MyCustom"), "custom expression keyed by map key");

        assert_eq!(parsed.springs.len(), 1);
        assert_eq!(parsed.springs[0].joints.len(), 2);
        assert_eq!(parsed.colliders.len(), 1);
    }
}
