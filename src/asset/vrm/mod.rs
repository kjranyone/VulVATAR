#![allow(dead_code)]
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
            ..Default::default()
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
            ..Default::default()
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
            super::extensions::parse_v0_extensions(&raw, &nodes, &mesh_to_node).expect("v0 parse");

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
            super::extensions::parse_v1_extensions(&vrmc_vrm, Some(&vrmc_spring)).expect("v1 parse");

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
