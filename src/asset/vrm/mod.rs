use std::collections::HashMap;
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

mod extensions;
mod gltf_decode;
mod mtoon;
mod v0;
mod v1;

use extensions::{parse_vrm_extensions, resolve_thumbnail};

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
                    gltf_decode::rehydrate_textures(&mut cached, &gltf_doc.document, blob, &source_path_clone);
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
        let (mut nodes, _node_name_map) = gltf_decode::build_skeleton_nodes(&gltf_doc.document);

        on_progress(LoadStage::Meshes);
        let meshes = gltf_decode::build_meshes(&gltf_doc.document, blob);
        let materials =
            gltf_decode::build_materials(&gltf_doc.document, blob, &source_path, &on_progress);
        let skin_bindings = gltf_decode::build_skin_bindings(&gltf_doc.document, blob);
        let meshes_with_skin =
            gltf_decode::assign_skins_to_meshes(&gltf_doc.document, meshes, &skin_bindings);

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
            gltf_decode::bake_y_flip_on_roots(&mut nodes, &root_nodes);
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
            animation_clips: gltf_decode::build_animation_clips(&gltf_doc.document, blob),
            node_to_mesh,
            vrm_meta: parsed.meta,
            root_aabb,
            loaded_from_cache: false,
        };

        Ok(Arc::new(asset))
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

#[cfg(test)]
mod metadata_tests;
