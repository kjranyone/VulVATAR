//! VRChat PhysBone and UnityPackage integration.
//!
//! Extracts `VRCPhysBone` and `VRCPhysBoneCollider` configurations from `.unitypackage`
//! archives (such as those shipped with BOOTH / VRChat avatars) and converts them into
//! VulVATAR's native [`SpringBoneAsset`] and [`ColliderAsset`].

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;
use log::info;
use tar::Archive;

use crate::asset::{
    ColliderAsset, ColliderId, ColliderRef, ColliderShape, HumanoidBone, NodeId, SkeletonAsset,
    SpringBoneAsset, Vec3,
};

/// Extracted VRCPhysBone parameters from a Unity Prefab YAML.
#[derive(Clone, Debug)]
pub struct VrcPhysBoneData {
    pub file_id: i64,
    pub go_file_id: i64,
    pub root_transform_id: i64,
    pub pull: f32,
    pub spring: f32,
    pub stiffness: f32,
    pub gravity: f32,
    pub gravity_falloff: f32,
    pub radius: f32,
    pub immobile: f32,
    pub collider_refs: Vec<i64>,
}

/// Extracted VRCPhysBoneCollider parameters from a Unity Prefab YAML.
#[derive(Clone, Debug)]
pub struct VrcColliderData {
    pub file_id: i64,
    pub name: String,
    pub parent_tf_id: Option<i64>,
    pub shape_type: u32, // 0 = Sphere, 1 = Capsule, 2 = Plane
    pub radius: f32,
    pub height: f32,
    pub position: Vec3,
    pub rotation: [f32; 4],
}

/// Complete parsed VRC physics dataset from a `.unitypackage`.
#[derive(Clone, Debug, Default)]
pub struct ParsedVrcData {
    pub phys_bones: Vec<VrcPhysBoneData>,
    pub colliders: Vec<VrcColliderData>,
    /// Maps stripped GameObject / Transform fileID -> Corresponding source fileID
    pub stripped_map: HashMap<i64, i64>,
    /// Maps Transform fileID -> father Transform fileID
    pub transform_fathers: HashMap<i64, i64>,
}

/// Look for a `.unitypackage` in the given directory or its parent.
pub fn find_unitypackage_in_dir(base_dir: &Path) -> Option<PathBuf> {
    // 1. Check base_dir directly
    if let Ok(entries) = std::fs::read_dir(base_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("unitypackage"))
                    .unwrap_or(false)
            {
                return Some(p);
            }
        }
    }

    // 2. Check parent directory
    if let Some(parent) = base_dir.parent() {
        if let Ok(entries) = std::fs::read_dir(parent) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_file()
                    && p.extension()
                        .and_then(|e| e.to_str())
                        .map(|e| e.eq_ignore_ascii_case("unitypackage"))
                        .unwrap_or(false)
                {
                    return Some(p);
                }
            }
        }
    }

    None
}

/// Open a `.unitypackage` archive and parse VRC PhysBones and Colliders from the main prefab.
pub fn parse_unitypackage(package_path: &Path) -> Result<ParsedVrcData, String> {
    info!("vrc: inspecting unitypackage: {:?}", package_path);
    let file = File::open(package_path)
        .map_err(|e| format!("failed to open unitypackage '{:?}': {e}", package_path))?;
    let gz = GzDecoder::new(file);
    let mut archive = Archive::new(gz);

    let mut pathnames: HashMap<String, String> = HashMap::new();
    let mut prefab_assets: HashMap<String, Vec<u8>> = HashMap::new();

    for entry in archive
        .entries()
        .map_err(|e| format!("failed to read tar entries: {e}"))?
    {
        let mut entry = entry.map_err(|e| format!("tar entry error: {e}"))?;
        let path = entry
            .path()
            .map_err(|e| format!("invalid tar entry path: {e}"))?
            .to_string_lossy()
            .replace('\\', "/");

        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() >= 2 {
            let guid = parts[0].to_string();
            let file_name = parts[1];

            if file_name == "pathname" {
                let mut buf = String::new();
                if entry.read_to_string(&mut buf).is_ok() {
                    pathnames.insert(guid, buf.trim().to_string());
                }
            } else if file_name == "asset" {
                let mut buf = Vec::new();
                if entry.read_to_end(&mut buf).is_ok() {
                    prefab_assets.insert(guid, buf);
                }
            }
        }
    }

    // Find the primary avatar prefab:
    // Filter pathnames ending with `.prefab`, prioritize ones not starting with "sample" or "demo",
    // and choose the largest one (avatars with full PhysBone setup are typically larger).
    let mut candidate_prefabs: Vec<(String, String, usize)> = Vec::new();
    for (guid, path) in &pathnames {
        if path.to_lowercase().ends_with(".prefab") {
            let size = prefab_assets.get(guid).map(|b| b.len()).unwrap_or(0);
            candidate_prefabs.push((guid.clone(), path.clone(), size));
        }
    }

    if candidate_prefabs.is_empty() {
        return Err("no .prefab found in unitypackage".to_string());
    }

    // Sort: largest size first
    candidate_prefabs.sort_by(|a, b| b.2.cmp(&a.2));
    let (best_guid, best_path, best_size) = &candidate_prefabs[0];
    info!(
        "vrc: selected primary prefab: '{}' (guid: {}, size: {} bytes)",
        best_path, best_guid, best_size
    );

    let raw_bytes = prefab_assets
        .get(best_guid)
        .ok_or_else(|| format!("missing asset data for guid {}", best_guid))?;
    let yaml_str = String::from_utf8_lossy(raw_bytes);

    let parsed = parse_prefab_yaml(&yaml_str);
    info!(
        "vrc: parsed {} PhysBones and {} Colliders from prefab",
        parsed.phys_bones.len(),
        parsed.colliders.len()
    );

    Ok(parsed)
}

/// Parse a Unity YAML `.prefab` content into [`ParsedVrcData`].
pub fn parse_prefab_yaml(yaml: &str) -> ParsedVrcData {
    let mut out = ParsedVrcData::default();

    // Map GameObject fileID -> GameObject Name
    let mut go_names: HashMap<i64, String> = HashMap::new();
    // Map Transform fileID -> GameObject fileID
    let mut tf_to_go: HashMap<i64, i64> = HashMap::new();

    // Split YAML documents by `--- !u!`
    let docs = yaml.split("\n--- !u!");

    for doc in docs {
        let lines: Vec<&str> = doc.lines().collect();
        if lines.is_empty() {
            continue;
        }

        let header = lines[0].trim();
        let is_stripped = header.contains("stripped");
        let header_parts: Vec<&str> = header.split_whitespace().collect();
        if header_parts.len() < 2 {
            continue;
        }

        let class_id: u32 = match header_parts[0].parse() {
            Ok(c) => c,
            Err(_) => continue,
        };
        let file_id_str = header_parts[1].trim_start_matches('&');
        let file_id: i64 = match file_id_str.parse() {
            Ok(id) => id,
            Err(_) => continue,
        };

        if is_stripped {
            for line in &lines {
                let trimmed = line.trim();
                if trimmed.starts_with("m_CorrespondingSourceObject:") {
                    if let Some(src_id) = extract_file_id_from_curly(trimmed) {
                        out.stripped_map.insert(file_id, src_id);
                    }
                }
            }
            continue;
        }

        match class_id {
            1 => {
                // GameObject
                for line in &lines {
                    let trimmed = line.trim();
                    if trimmed.starts_with("m_Name:") {
                        let name = trimmed.trim_start_matches("m_Name:").trim();
                        go_names.insert(file_id, name.to_string());
                    }
                }
            }
            4 => {
                // Transform
                let mut go_id: Option<i64> = None;
                let mut father_id: Option<i64> = None;
                for line in &lines {
                    let trimmed = line.trim();
                    if trimmed.starts_with("m_GameObject:") {
                        go_id = extract_file_id_from_curly(trimmed);
                    } else if trimmed.starts_with("m_Father:") {
                        father_id = extract_file_id_from_curly(trimmed);
                    }
                }
                if let Some(gid) = go_id {
                    tf_to_go.insert(file_id, gid);
                }
                if let Some(fid) = father_id {
                    out.transform_fathers.insert(file_id, fid);
                }
            }
            114 => {
                // MonoBehaviour (VRCPhysBone or VRCPhysBoneCollider)
                let is_phys_bone = doc.contains("pull:") && doc.contains("spring:");
                let is_collider = doc.contains("shapeType:") && doc.contains("radius:");

                if is_phys_bone {
                    let mut go_id: i64 = 0;
                    let mut root_id: i64 = 0;
                    let mut pull: f32 = 0.2;
                    let mut spring: f32 = 0.5;
                    let mut stiffness: f32 = 0.2;
                    let mut gravity: f32 = 0.0;
                    let mut gravity_falloff: f32 = 0.0;
                    let mut radius: f32 = 0.0;
                    let mut immobile: f32 = 0.0;
                    let mut colliders = Vec::new();

                    let mut in_colliders = false;

                    for line in &lines {
                        let trimmed = line.trim();
                        if trimmed.starts_with("m_GameObject:") {
                            if let Some(id) = extract_file_id_from_curly(trimmed) {
                                go_id = id;
                            }
                        } else if trimmed.starts_with("rootTransform:") {
                            if let Some(id) = extract_file_id_from_curly(trimmed) {
                                root_id = id;
                            }
                        } else if trimmed.starts_with("pull:") {
                            pull = parse_f32_after_colon(trimmed, pull);
                        } else if trimmed.starts_with("spring:") {
                            spring = parse_f32_after_colon(trimmed, spring);
                        } else if trimmed.starts_with("stiffness:") {
                            stiffness = parse_f32_after_colon(trimmed, stiffness);
                        } else if trimmed.starts_with("gravity:") {
                            gravity = parse_f32_after_colon(trimmed, gravity);
                        } else if trimmed.starts_with("gravityFalloff:") {
                            gravity_falloff = parse_f32_after_colon(trimmed, gravity_falloff);
                        } else if trimmed.starts_with("radius:") {
                            radius = parse_f32_after_colon(trimmed, radius);
                        } else if trimmed.starts_with("immobile:") {
                            immobile = parse_f32_after_colon(trimmed, immobile);
                        } else if trimmed.starts_with("colliders:") {
                            in_colliders = true;
                        } else if in_colliders {
                            if trimmed.starts_with("-") {
                                if let Some(id) = extract_file_id_from_curly(trimmed) {
                                    colliders.push(id);
                                }
                            } else if !trimmed.is_empty() {
                                in_colliders = false;
                            }
                        }
                    }

                    out.phys_bones.push(VrcPhysBoneData {
                        file_id,
                        go_file_id: go_id,
                        root_transform_id: root_id,
                        pull,
                        spring,
                        stiffness,
                        gravity,
                        gravity_falloff,
                        radius,
                        immobile,
                        collider_refs: colliders,
                    });
                } else if is_collider {
                    let mut go_id: i64 = 0;
                    let mut shape_type: u32 = 0;
                    let mut radius: f32 = 0.05;
                    let mut height: f32 = 0.0;
                    let mut pos = [0.0f32; 3];
                    let mut rot = [0.0f32, 0.0, 0.0, 1.0];

                    for line in &lines {
                        let trimmed = line.trim();
                        if trimmed.starts_with("m_GameObject:") {
                            if let Some(id) = extract_file_id_from_curly(trimmed) {
                                go_id = id;
                            }
                        } else if trimmed.starts_with("shapeType:") {
                            shape_type = parse_u32_after_colon(trimmed, 0);
                        } else if trimmed.starts_with("radius:") {
                            radius = parse_f32_after_colon(trimmed, radius);
                        } else if trimmed.starts_with("height:") {
                            height = parse_f32_after_colon(trimmed, height);
                        } else if trimmed.starts_with("position:") {
                            pos = extract_vec3_from_curly(trimmed, pos);
                        } else if trimmed.starts_with("rotation:") {
                            rot = extract_quat_from_curly(trimmed, rot);
                        }
                    }

                    let name = go_names
                        .get(&go_id)
                        .cloned()
                        .unwrap_or_else(|| format!("Collider_{file_id}"));

                    out.colliders.push(VrcColliderData {
                        file_id,
                        name,
                        parent_tf_id: None,
                        shape_type,
                        radius,
                        height,
                        position: pos,
                        rotation: rot,
                    });
                }
            }
            _ => {}
        }
    }

    out
}

fn extract_file_id_from_curly(line: &str) -> Option<i64> {
    if let Some(idx) = line.find("fileID:") {
        let rest = &line[idx + 7..];
        let num_str: String = rest
            .chars()
            .skip_while(|c| c.is_whitespace())
            .take_while(|c| c.is_ascii_digit() || *c == '-')
            .collect();
        return num_str.parse().ok();
    }
    None
}

fn parse_f32_after_colon(line: &str, default: f32) -> f32 {
    line.split(':')
        .nth(1)
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

fn parse_u32_after_colon(line: &str, default: u32) -> u32 {
    line.split(':')
        .nth(1)
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

fn extract_vec3_from_curly(line: &str, default: Vec3) -> Vec3 {
    let mut res = default;
    if let Some(x_idx) = line.find("x:") {
        let s = &line[x_idx + 2..];
        let num: String = s
            .chars()
            .skip_while(|c| c.is_whitespace())
            .take_while(|c| c.is_ascii_digit() || *c == '-' || *c == '.')
            .collect();
        if let Ok(v) = num.parse() {
            res[0] = v;
        }
    }
    if let Some(y_idx) = line.find("y:") {
        let s = &line[y_idx + 2..];
        let num: String = s
            .chars()
            .skip_while(|c| c.is_whitespace())
            .take_while(|c| c.is_ascii_digit() || *c == '-' || *c == '.')
            .collect();
        if let Ok(v) = num.parse() {
            res[1] = v;
        }
    }
    if let Some(z_idx) = line.find("z:") {
        let s = &line[z_idx + 2..];
        let num: String = s
            .chars()
            .skip_while(|c| c.is_whitespace())
            .take_while(|c| c.is_ascii_digit() || *c == '-' || *c == '.')
            .collect();
        if let Ok(v) = num.parse() {
            res[2] = v;
        }
    }
    res
}

fn extract_quat_from_curly(line: &str, default: [f32; 4]) -> [f32; 4] {
    let mut res = default;
    for (k, idx_target) in [("x:", 0), ("y:", 1), ("z:", 2), ("w:", 3)] {
        if let Some(idx) = line.find(k) {
            let s = &line[idx + 2..];
            let num: String = s
                .chars()
                .skip_while(|c| c.is_whitespace())
                .take_while(|c| c.is_ascii_digit() || *c == '-' || *c == '.')
                .collect();
            if let Ok(v) = num.parse() {
                res[idx_target] = v;
            }
        }
    }
    res
}

// ---------------------------------------------------------------------------
// Bone Mapping & SpringBone Generation
// ---------------------------------------------------------------------------

/// Categorise a non-humanoid bone chain by name into avatar parts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChainCategory {
    HairFront,
    HairSide,
    HairTwintale,
    HairBack,
    HairRibbon,
    HairWing,
    Skirt,
    Breast,
    Tail,
    Wing,
    Belt,
    Other,
}

impl ChainCategory {
    pub fn classify(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("bang") {
            ChainCategory::HairFront
        } else if lower.contains("twintale") || lower.contains("twintail") {
            ChainCategory::HairTwintale
        } else if lower.contains("side") {
            ChainCategory::HairSide
        } else if lower.contains("back") {
            ChainCategory::HairBack
        } else if lower.contains("ribbon") {
            ChainCategory::HairRibbon
        } else if lower.contains("wing") && lower.contains("hair") {
            ChainCategory::HairWing
        } else if lower.contains("skirt") {
            ChainCategory::Skirt
        } else if lower.contains("breast") {
            ChainCategory::Breast
        } else if lower.contains("tail") {
            ChainCategory::Tail
        } else if lower.contains("wing") {
            ChainCategory::Wing
        } else if lower.contains("belt") {
            ChainCategory::Belt
        } else if lower.contains("hair") {
            ChainCategory::HairBack
        } else {
            ChainCategory::Other
        }
    }
}

/// Check if a bone name belongs to facial features, limbs, or utility nodes
/// that should NEVER be simulated as dynamic spring bones (which cause tongue/cheek distortion).
fn is_excluded_bone_name(name: &str) -> bool {
    let lower = name.to_lowercase();
    // Head & facial components (e.g. Tongue, Cheek, Eye)
    lower.contains("tongue")
        || lower.contains("cheek")
        || lower.contains("eye")
        || lower.contains("pupil")
        || lower.contains("brow")
        || lower.contains("lip")
        || lower.contains("jaw")
        || lower.contains("mouth")
        || lower.contains("face")
        || lower.contains("nose")
        || lower.contains("ear")
        || lower.contains("teeth")
        || lower.contains("tooth")
        // Limbs & skeletal twist / helper bones
        || lower.contains("twist")
        || lower.contains("finger")
        || lower.contains("thumb")
        || lower.contains("index")
        || lower.contains("middle")
        || lower.contains("ring")
        || lower.contains("little")
        || lower.contains("pinky")
        || lower.contains("hand")
        || lower.contains("foot")
        || lower.contains("arm")
        || lower.contains("leg")
        || lower.contains("toe")
        // Utility & helpers
        || lower.contains("col")
        || lower.contains("mesh")
        || lower.contains("target")
        || lower.contains("ik")
        || lower.contains("pole")
        || lower.contains("socket")
        || lower.contains("anchor")
        || lower.contains("armature")
        || lower.contains("null")
        || lower.contains("dummy")
        || lower.contains("center")
        || lower.contains("scale")
}

/// Discovered non-humanoid bone chain from the skeleton hierarchy.
pub struct DiscoveredChain {
    pub root_node_idx: usize,
    pub category: ChainCategory,
    pub name: String,
    pub joint_node_indices: Vec<usize>,
}

/// Scan `skeleton` to discover non-humanoid bone chains suitable for dynamics.
pub fn discover_bone_chains(skeleton: &SkeletonAsset) -> Vec<DiscoveredChain> {
    let mut chains = Vec::new();

    for (node_idx, node) in skeleton.nodes.iter().enumerate() {
        if node.humanoid_bone.is_some() || node.children.is_empty() {
            continue;
        }

        // Never turn facial bones (tongue, cheek, eye) or extremities into spring chains
        if is_excluded_bone_name(&node.name) {
            continue;
        }

        // Must be a chain root: parent is either humanoid, None, or a container like Hair_root / Skirt_root
        let is_chain_root = match node.parent {
            None => false,
            Some(p_id) => {
                let p_idx = p_id.0 as usize;
                if p_idx < skeleton.nodes.len() {
                    let p_node = &skeleton.nodes[p_idx];
                    let p_name_lower = p_node.name.to_lowercase();
                    p_node.humanoid_bone.is_some()
                        || p_name_lower.contains("root")
                        || p_name_lower == "armature"
                } else {
                    false
                }
            }
        };

        if !is_chain_root {
            continue;
        }

        let cat = ChainCategory::classify(&node.name);

        // Collect all joints down this chain (single child branch or first path)
        let mut joints = vec![node_idx];
        let mut curr_idx = node_idx;
        while skeleton.nodes[curr_idx].children.len() == 1 {
            let next_id = skeleton.nodes[curr_idx].children[0];
            let next_idx = next_id.0 as usize;
            if next_idx < skeleton.nodes.len() {
                joints.push(next_idx);
                curr_idx = next_idx;
            } else {
                break;
            }
        }

        // If any descendant joint in the chain is an excluded bone, skip the entire chain
        if joints.iter().any(|&j_idx| is_excluded_bone_name(&skeleton.nodes[j_idx].name)) {
            continue;
        }

        chains.push(DiscoveredChain {
            root_node_idx: node_idx,
            category: cat,
            name: node.name.clone(),
            joint_node_indices: joints,
        });
    }

    chains
}

/// Convert parsed VRC PhysBones and Colliders into VulVATAR [`SpringBoneAsset`]s and [`ColliderAsset`]s.
pub fn build_spring_bones_and_colliders(
    skeleton: &SkeletonAsset,
    vrc_data: &ParsedVrcData,
) -> (Vec<SpringBoneAsset>, Vec<ColliderAsset>) {
    let mut spring_bones = Vec::new();
    let mut colliders = Vec::new();

    // 1. Build Colliders: attach leg / body colliders to humanoid bones
    let mut leg_collider_refs = Vec::new();

    let left_upper_leg = skeleton
        .nodes
        .iter()
        .position(|n| n.humanoid_bone == Some(HumanoidBone::LeftUpperLeg));
    let right_upper_leg = skeleton
        .nodes
        .iter()
        .position(|n| n.humanoid_bone == Some(HumanoidBone::RightUpperLeg));

    for c in &vrc_data.colliders {
        let lower = c.name.to_lowercase();
        let target_node_idx = if lower.contains("_l") || lower.contains("left") {
            left_upper_leg
        } else if lower.contains("_r") || lower.contains("right") {
            right_upper_leg
        } else {
            None
        };

        if let Some(node_idx) = target_node_idx {
            let col_id = ColliderId(colliders.len() as u64 + 1);
            let shape = if c.height > 0.01 {
                ColliderShape::Capsule {
                    radius: c.radius.max(0.02),
                    height: c.height,
                }
            } else {
                ColliderShape::Sphere {
                    radius: c.radius.max(0.02),
                }
            };
            colliders.push(ColliderAsset {
                id: col_id,
                node: NodeId(node_idx as u64),
                shape,
                offset: c.position,
            });
            leg_collider_refs.push(ColliderRef { id: col_id });
        }
    }

    // 2. Discover bone chains from skeleton
    let chains = discover_bone_chains(skeleton);
    info!(
        "vrc: discovered {} non-humanoid bone chains in skeleton",
        chains.len()
    );

    // Group PhysBones by category / characteristics:
    let mut skirt_pbs: Vec<&VrcPhysBoneData> = Vec::new();
    let mut tail_pbs: Vec<&VrcPhysBoneData> = Vec::new();
    let mut front_pbs: Vec<&VrcPhysBoneData> = Vec::new();
    let mut back_pbs: Vec<&VrcPhysBoneData> = Vec::new();
    let mut side_pbs: Vec<&VrcPhysBoneData> = Vec::new();
    let mut twintail_pbs: Vec<&VrcPhysBoneData> = Vec::new();
    let mut ribbon_pbs: Vec<&VrcPhysBoneData> = Vec::new();
    let mut breast_pbs: Vec<&VrcPhysBoneData> = Vec::new();
    let mut wing_pbs: Vec<&VrcPhysBoneData> = Vec::new();
    let mut belt_pbs: Vec<&VrcPhysBoneData> = Vec::new();
    let mut other_pbs: Vec<&VrcPhysBoneData> = Vec::new();

    for pb in &vrc_data.phys_bones {
        if pb.gravity >= 0.8 && !pb.collider_refs.is_empty() {
            skirt_pbs.push(pb);
        } else if (pb.gravity - 0.22).abs() < 0.05 && pb.collider_refs.len() == 1 {
            tail_pbs.push(pb);
        } else if pb.immobile >= 0.95 && pb.gravity == 0.0 {
            breast_pbs.push(pb);
        } else if pb.radius >= 0.03 && pb.collider_refs.len() >= 2 {
            twintail_pbs.push(pb);
        } else if pb.pull >= 0.5 {
            ribbon_pbs.push(pb);
        } else if pb.gravity >= 0.5 {
            back_pbs.push(pb);
        } else if (pb.pull - 0.20).abs() < 0.01 && (pb.spring - 0.603).abs() < 0.01 {
            front_pbs.push(pb);
        } else if (pb.spring - 0.20).abs() < 0.01 {
            belt_pbs.push(pb);
        } else if (pb.pull - 0.25).abs() < 0.01 {
            wing_pbs.push(pb);
        } else if pb.radius > 0.03 {
            side_pbs.push(pb);
        } else {
            other_pbs.push(pb);
        }
    }

    // Default fallback parameters if no matching PhysBone is found:
    let default_hair = VrcPhysBoneData {
        file_id: 0,
        go_file_id: 0,
        root_transform_id: 0,
        pull: 0.2,
        spring: 0.5,
        stiffness: 0.2,
        gravity: 0.0,
        gravity_falloff: 0.0,
        radius: 0.02,
        immobile: 0.5,
        collider_refs: Vec::new(),
    };

    let default_skirt = VrcPhysBoneData {
        file_id: 0,
        go_file_id: 0,
        root_transform_id: 0,
        pull: 0.15,
        spring: 0.4,
        stiffness: 0.2,
        gravity: 0.8,
        gravity_falloff: 0.0,
        radius: 0.02,
        immobile: 0.8,
        collider_refs: Vec::new(),
    };

    for chain in &chains {
        let pb = match chain.category {
            ChainCategory::HairFront => front_pbs.first().copied().unwrap_or(&default_hair),
            ChainCategory::HairSide => side_pbs.first().copied().unwrap_or(&default_hair),
            ChainCategory::HairTwintale => twintail_pbs.first().copied().unwrap_or(&default_hair),
            ChainCategory::HairBack => back_pbs.first().copied().unwrap_or(&default_hair),
            ChainCategory::HairRibbon => ribbon_pbs.first().copied().unwrap_or(&default_hair),
            ChainCategory::HairWing => wing_pbs.first().copied().unwrap_or(&default_hair),
            ChainCategory::Skirt => skirt_pbs.first().copied().unwrap_or(&default_skirt),
            ChainCategory::Breast => breast_pbs.first().copied().unwrap_or(&default_hair),
            ChainCategory::Tail => tail_pbs.first().copied().unwrap_or(&default_skirt),
            ChainCategory::Wing => wing_pbs.first().copied().unwrap_or(&default_hair),
            ChainCategory::Belt => belt_pbs.first().copied().unwrap_or(&default_hair),
            ChainCategory::Other => other_pbs.first().copied().unwrap_or(&default_hair),
        };

        // Convert VRC parameters to SpringBoneAsset
        let stiffness = (pb.pull * 2.0).clamp(0.05, 1.0);
        let drag_force = (1.0 - pb.spring * 0.8).clamp(0.1, 0.9);
        let gravity_power = pb.gravity.clamp(0.0, 1.5);
        let radius = pb.radius.max(0.01);

        let col_refs = if chain.category == ChainCategory::Skirt {
            leg_collider_refs.clone()
        } else {
            Vec::new()
        };

        let joint_nodes: Vec<NodeId> = chain
            .joint_node_indices
            .iter()
            .map(|&idx| NodeId(idx as u64))
            .collect();

        spring_bones.push(SpringBoneAsset {
            chain_root: NodeId(chain.root_node_idx as u64),
            joints: joint_nodes,
            stiffness,
            drag_force,
            gravity_dir: [0.0, -1.0, 0.0],
            gravity_power,
            radius,
            collider_refs: col_refs,
            joint_stiffness: Vec::new(),
            joint_drag: Vec::new(),
            joint_gravity_power: Vec::new(),
        });
    }

    (spring_bones, colliders)
}
