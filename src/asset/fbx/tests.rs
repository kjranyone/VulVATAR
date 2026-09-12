use std::path::Path;

/// Resolve the Yumeka sample FBX on this machine. The pinned v1.0.1
/// sample takes precedence when present; otherwise the newest
/// `YUMEKA_v*` directory under `sample_data/` wins, so the clearance
/// tests keep running against whatever version is actually installed
/// (the local sample here is v1.0.3, and the v1.0.1-only tests below
/// silently skip).
fn yumeka_fbx_path() -> Option<String> {
    let pinned = "sample_data/YUMEKA_v1.0.1/FBX/Yumeka_v1.0.fbx";
    if Path::new(pinned).exists() {
        return Some(pinned.to_string());
    }
    let mut best: Option<(String, String)> = None;
    let Ok(entries) = std::fs::read_dir("sample_data") else {
        return None;
    };
    for entry in entries.flatten() {
        let dir_name = entry.file_name().to_string_lossy().to_string();
        if !dir_name.starts_with("YUMEKA_v") {
            continue;
        }
        let Ok(fbx_dir) = std::fs::read_dir(entry.path().join("FBX")) else {
            continue;
        };
        for f in fbx_dir.flatten() {
            if f.path().extension().and_then(|e| e.to_str()) != Some("fbx") {
                continue;
            }
            let is_newer = match &best {
                Some((cur, _)) => dir_name > *cur,
                None => true,
            };
            if is_newer {
                if let Some(p) = f.path().to_str() {
                    best = Some((dir_name.clone(), p.to_string()));
                }
            }
        }
    }
    best.map(|(_, p)| p)
}

/// Find a skeleton node by humanoid classification, falling back to the
/// Yumeka bone naming (`UpperLeg_L` etc.) when the map has no entry.
fn find_humanoid_node(
    asset: &crate::asset::AvatarAsset,
    bone: crate::asset::HumanoidBone,
    name_suffix: &str,
) -> Option<usize> {
    asset
        .skeleton
        .nodes
        .iter()
        .position(|n| n.humanoid_bone == Some(bone))
        .or_else(|| {
            asset
                .skeleton
                .nodes
                .iter()
                .position(|n| n.name.ends_with(name_suffix))
        })
}

#[test]
fn test_load_yumeka_fbx() {
    let fbx_path = "sample_data/YUMEKA_v1.0.1/FBX/Yumeka_v1.0.fbx";
    if !Path::new(fbx_path).exists() {
        return;
    }

    let opts = ufbx::LoadOpts::default();
    let scene = ufbx::load_file(fbx_path, opts).unwrap();

    // Load avatar using FbxAssetLoader
    let loader = crate::asset::fbx::FbxAssetLoader::new();
    let asset = loader.load_with_progress(fbx_path, |_| {}).unwrap();
    println!(
        "Loaded AvatarAsset successfully! nodes: {}",
        asset.skeleton.nodes.len()
    );

    let mut globals = vec![crate::asset::identity_matrix(); asset.skeleton.nodes.len()];
    let locals: Vec<crate::asset::Transform> = asset
        .skeleton
        .nodes
        .iter()
        .map(|n| n.rest_local.clone())
        .collect();
    crate::avatar::pose::compute_global_transforms(&asset.skeleton, &locals, &mut globals);

    let mut max_diff = 0.0f32;
    for (i, node) in scene.nodes.iter().enumerate() {
        let u_n2w = &node.node_to_world;
        let g = &globals[i];
        // g is column-major: g[col][row]
        // u_n2w is row-major: u.m[row][col]
        let diff = (g[0][0] - u_n2w.m00 as f32).abs()
            + (g[1][0] - u_n2w.m01 as f32).abs()
            + (g[2][0] - u_n2w.m02 as f32).abs()
            + (g[3][0] - u_n2w.m03 as f32).abs()
            + (g[3][1] - u_n2w.m13 as f32).abs()
            + (g[3][2] - u_n2w.m23 as f32).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 1e-2 && i < 10 {
            println!(
                "Transform diff at node {} ('{}'): diff={}",
                i, node.element.name, diff
            );
            println!("  g col3 (t): [{}, {}, {}]", g[3][0], g[3][1], g[3][2]);
            println!("  u t: [{}, {}, {}]", u_n2w.m03, u_n2w.m13, u_n2w.m23);
        }
    }
    println!(
        "Max transform diff between compute_global_transforms and ufbx: {}",
        max_diff
    );
    assert!(
        max_diff < 1e-4,
        "Global transforms diverge from ufbx: {}",
        max_diff
    );

    // Verify inverse bind matrices
    let hips_node_idx = asset
        .skeleton
        .nodes
        .iter()
        .position(|n| n.name == "Hips")
        .unwrap();
    let ibm_hips = &asset.skeleton.inverse_bind_matrices[hips_node_idx];
    assert_ne!(
        ibm_hips,
        &crate::asset::identity_matrix(),
        "Hips IBM must not be identity!"
    );

    // Verify vertex joint indices
    let mut found_hips_joint = false;
    for mesh in &asset.meshes {
        for prim in &mesh.primitives {
            if let Some(ref vd) = prim.vertices {
                for (ji, jw) in vd.joint_indices.iter().zip(&vd.joint_weights) {
                    for (&slot, &weight) in ji.iter().zip(jw) {
                        if weight > 0.01 {
                            assert!(
                                (slot as usize) < asset.skeleton.nodes.len(),
                                "Joint index {} out of bounds ({})",
                                slot,
                                asset.skeleton.nodes.len()
                            );
                            if slot as usize == hips_node_idx {
                                found_hips_joint = true;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        found_hips_joint,
        "Expected at least one vertex to be bound to Hips"
    );

    // Verify node_to_mesh references valid node indices
    for &node_idx in asset.node_to_mesh.keys() {
        assert!(
            node_idx < asset.skeleton.nodes.len(),
            "node_to_mesh key {} out of bounds ({})",
            node_idx,
            asset.skeleton.nodes.len()
        );
    }

    // Verify materials
    for mat in &asset.materials {
        assert!(
            mat.double_sided,
            "Material {} should be double_sided",
            mat.name
        );
        let lower = mat.name.to_lowercase();
        if lower.contains("transparent")
            || lower.contains("trans")
            || lower.contains("alpha")
            || lower.contains("blend")
        {
            assert_eq!(mat.alpha_mode, crate::asset::AlphaMode::Mask(0.05));
        } else {
            assert_eq!(mat.alpha_mode, crate::asset::AlphaMode::Opaque);
        }
    }

    // Verify wing material resolved its mask texture via token matching
    let wing_mat = asset
        .materials
        .iter()
        .find(|m| m.name.to_lowercase().contains("wing"))
        .expect("Wing mat must exist");
    assert!(
        wing_mat.texture_bindings.base_color_texture.is_some(),
        "Wing texture must be resolved"
    );

    // Verify triangulation generated proper 3-vertex polygons for all primitives
    let mut total_indices = 0;
    for mesh in &asset.meshes {
        for prim in &mesh.primitives {
            assert_eq!(
                prim.index_count % 3,
                0,
                "Indices count must be multiple of 3"
            );
            total_indices += prim.index_count;
        }
    }
    assert!(
        total_indices > 20000,
        "Indices count should be well populated (got {})",
        total_indices
    );

    // Verify humanoid mapping
    let humanoid = asset.humanoid.as_ref().expect("Humanoid map must exist");
    assert!(humanoid
        .bone_map
        .contains_key(&crate::asset::HumanoidBone::Hips));
    assert!(humanoid
        .bone_map
        .contains_key(&crate::asset::HumanoidBone::Head));

    // Verify expressions / blendshapes
    let exprs = &asset.default_expressions.expressions;
    assert!(!exprs.is_empty(), "Expressions must not be empty");
    assert!(
        exprs.iter().any(|e| e.name == "aa"),
        "Viseme 'aa' must be mapped"
    );
    assert!(
        exprs.iter().any(|e| e.name == "blink"),
        "Blink must be mapped"
    );

    // Verify height (Y-up)
    let size = asset.root_aabb.size();
    assert!(
        size[1] > 1.0 && size[1] < 1.8,
        "Model height along Y axis should be ~1.34m (got {:.2}m)",
        size[1]
    );

    let mats_with_textures = asset
        .materials
        .iter()
        .filter(|m| {
            m.texture_bindings
                .base_color_texture
                .as_ref()
                .is_some_and(|t| t.pixel_data.is_some())
        })
        .count();
    assert_eq!(
        mats_with_textures,
        asset.materials.len(),
        "All materials should have resolved textures"
    );

    // Second load (must hit cache and rehydrate textures)
    let cached_asset = loader
        .load(fbx_path)
        .expect("Failed to load cached Yumeka FBX avatar");
    assert!(
        cached_asset.loaded_from_cache,
        "Second load must be flagged as loaded_from_cache"
    );
    let cached_mats_with_textures = cached_asset
        .materials
        .iter()
        .filter(|m| {
            m.texture_bindings
                .base_color_texture
                .as_ref()
                .is_some_and(|t| t.pixel_data.is_some())
        })
        .count();
    assert_eq!(
        cached_mats_with_textures, mats_with_textures,
        "Rehydrated textures count in cached load must match initial parse"
    );

    // Verify spring bones & colliders were extracted and cached
    assert!(
        !asset.spring_bones.is_empty(),
        "Expected spring bones to be populated from unitypackage/heuristic"
    );
    assert_eq!(
        cached_asset.spring_bones.len(),
        asset.spring_bones.len(),
        "Cached spring_bones count must match initial parse"
    );
    assert_eq!(
        cached_asset.colliders.len(),
        asset.colliders.len(),
        "Cached colliders count must match initial parse"
    );

    // Verify spring bone simulation step runs without panic or NaN
    let mut instance = crate::avatar::AvatarInstance::new(
        crate::avatar::AvatarInstanceId(1),
        std::sync::Arc::clone(&asset),
    );
    assert_eq!(
        instance.secondary_motion.spring_states.len(),
        asset.spring_bones.len()
    );
    let tuning = crate::simulation::spring::SpringTuning::default();
    for _ in 0..10 {
        crate::simulation::spring::step_spring_bones(
            1.0 / 60.0,
            &mut instance,
            &[],
            &tuning,
            [0.0, -1.0, 0.0],
            1.0,
        );
    }
    // Verify positions are finite (no NaN / Inf)
    for state in &instance.secondary_motion.spring_states {
        for pos in &state.positions {
            assert!(pos[0].is_finite() && pos[1].is_finite() && pos[2].is_finite());
        }
    }
}

#[test]
fn test_find_avatar_file_in_dir() {
    let dir = Path::new("sample_data/YUMEKA_v1.0.1");
    if !dir.exists() {
        return;
    }

    let found = crate::asset::find_avatar_file_in_dir(dir);
    assert!(
        found.is_some(),
        "Should find an avatar file in sample_data/YUMEKA_v1.0.1"
    );
    let found_path = found.unwrap();
    assert!(
        found_path.ends_with("Yumeka_v1.0.fbx"),
        "Should find Yumeka_v1.0.fbx, got: {:?}",
        found_path
    );
}

#[test]
fn test_inspect_mouth() {
    let fbx_path = "sample_data/YUMEKA_v1.0.1/FBX/Yumeka_v1.0.fbx";
    if !Path::new(fbx_path).exists() {
        return;
    }

    let loader = crate::asset::fbx::FbxAssetLoader::new();
    let asset = loader.load(fbx_path).unwrap();

    // 1. Verify facial bones (tongue, cheek, eye, mouth) are NEVER in spring bones
    for sb in &asset.spring_bones {
        let root_name = &asset.skeleton.nodes[sb.chain_root.0 as usize].name;
        let root_lower = root_name.to_lowercase();
        assert!(
            !root_lower.contains("tongue")
                && !root_lower.contains("cheek")
                && !root_lower.contains("eye")
                && !root_lower.contains("jaw"),
            "Facial bone '{}' must not be a spring bone chain root",
            root_name
        );
        for &j in &sb.joints {
            let j_name = &asset.skeleton.nodes[j.0 as usize].name;
            let j_lower = j_name.to_lowercase();
            assert!(
                !j_lower.contains("tongue")
                    && !j_lower.contains("cheek")
                    && !j_lower.contains("eye")
                    && !j_lower.contains("jaw"),
                "Facial bone '{}' must not be in spring bone joints",
                j_name
            );
        }
    }

    // 2. Verify standard preset 'aa' binds only the best morph target (no duplicate application)
    let aa_expr = asset
        .default_expressions
        .expressions
        .iter()
        .find(|e| e.name == "aa")
        .expect("Preset 'aa' must exist");
    assert_eq!(
        aa_expr.morph_binds.len(),
        1,
        "Preset 'aa' must have exactly 1 morph bind per mesh node, got: {:?}",
        aa_expr.morph_binds
    );
    // Target 0 is vrc.v_aa
    assert_eq!(
        aa_expr.morph_binds[0].morph_target_index, 0,
        "Preset 'aa' should prioritize vrc.v_aa"
    );
}

#[test]
fn test_inspect_skirt_meshes() {
    let fbx_path = "sample_data/YUMEKA_v1.0.1/FBX/Yumeka_v1.0.fbx";
    if !Path::new(fbx_path).exists() {
        return;
    }

    let loader = crate::asset::fbx::FbxAssetLoader::new();
    let asset = loader.load(fbx_path).unwrap();

    let skirt_nodes: Vec<(usize, &str)> = asset
        .skeleton
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.name.to_lowercase().contains("skirt"))
        .map(|(i, n)| (i, n.name.as_str()))
        .collect();
    println!(
        "=== SKIRT BONES IN SKELETON (total: {}) ===",
        skirt_nodes.len()
    );
    for &(idx, name) in &skirt_nodes {
        println!("  Node {}: '{}'", idx, name);
    }

    println!("=== MESHES WEIGHTED TO SKIRT BONES ===");
    for mesh in &asset.meshes {
        for (pi, prim) in mesh.primitives.iter().enumerate() {
            let mut skirt_vert_count = 0usize;
            let mut total_vert_count = 0usize;
            if let Some(ref vd) = prim.vertices {
                total_vert_count = vd.positions.len();
                for (ji, jw) in vd.joint_indices.iter().zip(&vd.joint_weights) {
                    let mut is_skirt = false;
                    for (&slot, &w) in ji.iter().zip(jw) {
                        if w > 0.05 && skirt_nodes.iter().any(|&(s_idx, _)| slot as usize == s_idx)
                        {
                            is_skirt = true;
                            break;
                        }
                    }
                    if is_skirt {
                        skirt_vert_count += 1;
                    }
                }
            }
            if skirt_vert_count > 0 {
                let mat_name = asset
                    .materials
                    .iter()
                    .find(|m| m.id == prim.material_id)
                    .map(|m| m.name.as_str())
                    .unwrap_or("unknown");
                let raw_y_min = prim
                    .vertices
                    .as_ref()
                    .map(|v| {
                        v.positions
                            .iter()
                            .map(|p| p[1])
                            .fold(f32::INFINITY, f32::min)
                    })
                    .unwrap_or(0.0);
                let raw_y_max = prim
                    .vertices
                    .as_ref()
                    .map(|v| {
                        v.positions
                            .iter()
                            .map(|p| p[1])
                            .fold(f32::NEG_INFINITY, f32::max)
                    })
                    .unwrap_or(0.0);
                println!(
                    "  Mesh '{}' (id={:?}) prim {} (id={:?}): {} / {} vertices weighted to skirt (mat='{}', bounds={:?}..{:?}, raw_y={:.4}..{:.4})",
                    mesh.name, mesh.id, pi, prim.id, skirt_vert_count, total_vert_count, mat_name, prim.bounds.min, prim.bounds.max, raw_y_min, raw_y_max
                );
            }
        }
    }

    let circle_056 = asset.meshes.iter().find(|m| m.name == "Circle.056");
    assert!(
        circle_056.is_some(),
        "Yumeka should have Circle.056 mesh for skirt"
    );
    let circle_056 = circle_056.unwrap();
    assert_eq!(circle_056.primitives.len(), 1);
    let prim = &circle_056.primitives[0];
    assert_eq!(prim.vertex_count, 2460);
    assert!(
        prim.bounds.min[1] > 0.55 && prim.bounds.max[1] < 0.85,
        "Skirt bounds must be around hips/thighs"
    );
}

#[test]
fn test_inspect_hair_and_colliders() {
    let fbx_path = "sample_data/YUMEKA_v1.0.1/FBX/Yumeka_v1.0.fbx";
    if !Path::new(fbx_path).exists() {
        return;
    }

    let loader = crate::asset::fbx::FbxAssetLoader::new();
    let asset = loader.load(fbx_path).unwrap();

    println!(
        "=== TOTAL COLLIDERS IN ASSET: {} ===",
        asset.colliders.len()
    );
    for (i, c) in asset.colliders.iter().enumerate() {
        let node_name = &asset.skeleton.nodes[c.node.0 as usize].name;
        println!(
            "  Collider {}: node='{}' (id={:?}), shape={:?}, offset={:?}",
            i, node_name, c.node, c.shape, c.offset
        );
    }

    let mut avatar = crate::avatar::AvatarInstance::new(
        crate::avatar::AvatarInstanceId(1),
        std::sync::Arc::clone(&asset),
    );
    avatar.build_base_pose();
    avatar.compute_global_pose();

    println!("=== HUMANOID BONE WORLD POSITIONS ===");
    for (i, node) in asset.skeleton.nodes.iter().enumerate() {
        if let Some(hb) = node.humanoid_bone {
            let pos = crate::math_utils::mat4_translation(&avatar.pose.global_transforms[i]);
            println!(
                "  Node {}: {:?} ('{}') -> world pos: [{:.4}, {:.4}, {:.4}]",
                i, hb, node.name, pos[0], pos[1], pos[2]
            );
        }
    }

    println!("=== HAIR CHAIN REST POSITIONS ===");
    for (i, sb) in asset.spring_bones.iter().enumerate() {
        let root_name = &asset.skeleton.nodes[sb.chain_root.0 as usize].name;
        if root_name.to_lowercase().contains("hair") {
            let root_pos = crate::math_utils::mat4_translation(
                &avatar.pose.global_transforms[sb.chain_root.0 as usize],
            );
            let last_joint = sb.joints.last().copied().unwrap_or(sb.chain_root).0 as usize;
            let tip_pos =
                crate::math_utils::mat4_translation(&avatar.pose.global_transforms[last_joint]);
            println!("  Hair Chain {}: '{}' root=[{:.4}, {:.4}, {:.4}] -> tip=[{:.4}, {:.4}, {:.4}] (joints={})",
                i, root_name, root_pos[0], root_pos[1], root_pos[2], tip_pos[0], tip_pos[1], tip_pos[2], sb.joints.len()
            );
        }
    }

    assert!(
        asset.colliders.len() >= 8,
        "Expected at least 8 body colliders (Legs, Chest, Spine, Arms, Head), got {}",
        asset.colliders.len()
    );

    // Verify hair chains have colliders assigned
    let hair_springs: Vec<&crate::asset::SpringBoneAsset> = asset
        .spring_bones
        .iter()
        .filter(|sb| {
            asset.skeleton.nodes[sb.chain_root.0 as usize]
                .name
                .to_lowercase()
                .contains("hair")
        })
        .collect();
    assert!(
        !hair_springs.is_empty(),
        "Yumeka must have hair spring bones"
    );
    for sb in &hair_springs {
        let name = &asset.skeleton.nodes[sb.chain_root.0 as usize].name;
        assert!(
            !sb.collider_refs.is_empty(),
            "Hair chain '{}' must have colliders assigned to prevent costume penetration",
            name
        );
    }

    // Step spring bones with forward gravity and verify simulation stability
    let tuning = crate::simulation::spring::SpringTuning::default();
    for _ in 0..15 {
        crate::simulation::spring::step_spring_bones(
            1.0 / 60.0,
            &mut avatar,
            &[],
            &tuning,
            [0.0, -0.707, 0.707], // forward/downward gravity
            1.0,
        );
    }
    for state in &avatar.secondary_motion.spring_states {
        for pos in &state.positions {
            assert!(pos[0].is_finite() && pos[1].is_finite() && pos[2].is_finite());
        }
    }
}

#[test]
fn test_inspect_thigh_colliders_and_skirt() {
    fn mat4_dir(m: &[[f32; 4]; 4], d: &[f32; 3]) -> [f32; 3] {
        [
            m[0][0] * d[0] + m[1][0] * d[1] + m[2][0] * d[2],
            m[0][1] * d[0] + m[1][1] * d[1] + m[2][1] * d[2],
            m[0][2] * d[0] + m[1][2] * d[1] + m[2][2] * d[2],
        ]
    }
    let fbx_path = "sample_data/YUMEKA_v1.0.1/FBX/Yumeka_v1.0.fbx";
    if !Path::new(fbx_path).exists() {
        return;
    }

    let loader = crate::asset::fbx::FbxAssetLoader::new();
    let asset = loader.load(fbx_path).unwrap();

    let mut avatar = crate::avatar::AvatarInstance::new(
        crate::avatar::AvatarInstanceId(1),
        std::sync::Arc::clone(&asset),
    );
    avatar.build_base_pose();
    avatar.compute_global_pose();

    println!("=== SKELETON LEG NODES ===");
    for (i, node) in asset.skeleton.nodes.iter().enumerate() {
        if matches!(
            node.humanoid_bone,
            Some(crate::asset::HumanoidBone::Hips)
                | Some(crate::asset::HumanoidBone::LeftUpperLeg)
                | Some(crate::asset::HumanoidBone::LeftLowerLeg)
                | Some(crate::asset::HumanoidBone::RightUpperLeg)
                | Some(crate::asset::HumanoidBone::RightLowerLeg)
        ) {
            let pos = crate::math_utils::mat4_translation(&avatar.pose.global_transforms[i]);
            let up = mat4_dir(&avatar.pose.global_transforms[i], &[0.0, 1.0, 0.0]);
            println!(
                "  Node {}: {:?} ('{}') -> world pos: [{:.4}, {:.4}, {:.4}], local Y up in world: [{:.4}, {:.4}, {:.4}]",
                i, node.humanoid_bone, node.name, pos[0], pos[1], pos[2], up[0], up[1], up[2]
            );
        }
    }

    println!("=== THIGH COLLIDERS IN WORLD SPACE ===");
    for (i, c) in asset.colliders.iter().enumerate() {
        let node_idx = c.node.0 as usize;
        let node = &asset.skeleton.nodes[node_idx];
        if matches!(
            node.humanoid_bone,
            Some(crate::asset::HumanoidBone::LeftUpperLeg)
                | Some(crate::asset::HumanoidBone::RightUpperLeg)
        ) {
            let node_pos =
                crate::math_utils::mat4_translation(&avatar.pose.global_transforms[node_idx]);
            let rotated_offset = mat4_dir(&avatar.pose.global_transforms[node_idx], &c.offset);
            let center = crate::math_utils::vec3_add(&node_pos, &rotated_offset);
            let up = mat4_dir(&avatar.pose.global_transforms[node_idx], &[0.0, 1.0, 0.0]);
            let up_len = crate::math_utils::vec3_length(&up);
            let up_norm = [up[0] / up_len, up[1] / up_len, up[2] / up_len];

            if let crate::asset::ColliderShape::Capsule { radius, height } = c.shape {
                let half_h = height * 0.5;
                let seg_a = crate::math_utils::vec3_sub(
                    &center,
                    &crate::math_utils::vec3_scale(&up_norm, half_h),
                );
                let seg_b = crate::math_utils::vec3_add(
                    &center,
                    &crate::math_utils::vec3_scale(&up_norm, half_h),
                );
                println!(
                    "  Thigh Collider {}: node='{}', radius={:.4}, height={:.4}\n    center=[{:.4}, {:.4}, {:.4}]\n    seg_a=[{:.4}, {:.4}, {:.4}]\n    seg_b=[{:.4}, {:.4}, {:.4}]",
                    i, node.name, radius, height, center[0], center[1], center[2], seg_a[0], seg_a[1], seg_a[2], seg_b[0], seg_b[1], seg_b[2]
                );
            }
        }
    }

    println!("=== SKIRT SPRING CHAINS ===");
    let mut skirt_chain_count = 0;
    for (i, sb) in asset.spring_bones.iter().enumerate() {
        let root_name = &asset.skeleton.nodes[sb.chain_root.0 as usize].name;
        if root_name.to_lowercase().contains("skirt") {
            skirt_chain_count += 1;
            let col_names: Vec<String> = sb
                .collider_refs
                .iter()
                .map(|r| {
                    let c = asset.colliders.iter().find(|c| c.id == r.id).unwrap();
                    let n = &asset.skeleton.nodes[c.node.0 as usize].name;
                    format!("{}(id={})", n, r.id.0)
                })
                .collect();
            println!(
                "  Skirt Chain {}: '{}' radius={:.4}, grav={:.4}, colliders={:?}",
                i, root_name, sb.radius, sb.gravity_power, col_names
            );

            // Assert: Skirt chains do NOT carry rigid thigh colliders (anti-penetration is handled by GPU Skin-Anchor clearance field)
            assert!(
                sb.collider_refs.is_empty(),
                "Skirt chain '{}' should not have rigid capsule colliders attached",
                root_name
            );
        }
    }
    assert!(
        skirt_chain_count >= 10,
        "Expected at least 10 skirt chains in Yumeka"
    );

    // 4. Test natural skirt hanging under downward gravity
    let mut avatar = crate::avatar::AvatarInstance::new(
        crate::avatar::AvatarInstanceId(1),
        std::sync::Arc::clone(&asset),
    );
    avatar.compute_global_pose();

    // Step physics with downward gravity
    let tuning = crate::simulation::spring::SpringTuning::default();
    for _ in 0..20 {
        crate::simulation::spring::step_spring_bones(
            1.0 / 60.0,
            &mut avatar,
            &[],
            &tuning,
            [0.0, -1.0, 0.0],
            1.0,
        );
    }

    // Verify skirt chains hang naturally downward, not flipping or pointing upward
    for (i, sb) in asset.spring_bones.iter().enumerate() {
        let root_name = &asset.skeleton.nodes[sb.chain_root.0 as usize].name;
        if root_name == "Skirt_2_L" || root_name == "Skirt_1" {
            let last_pos = avatar.secondary_motion.spring_states[i]
                .positions
                .last()
                .copied()
                .unwrap();
            println!(
                "  Natural hang, {} tip pos: [{:.4}, {:.4}, {:.4}]",
                root_name, last_pos[0], last_pos[1], last_pos[2]
            );
            // Tip Y must be lower than root (hangs downward, Y < 0.70)
            assert!(
                last_pos[1] < 0.70,
                "Skirt chain '{}' tip Y ({}) should hang naturally downward, not pointing upward",
                root_name,
                last_pos[1]
            );
        }
    }
}

#[test]
fn test_inspect_all_meshes_in_yumeka() {
    let fbx_path = "sample_data/YUMEKA_v1.0.1/FBX/Yumeka_v1.0.fbx";
    if !Path::new(fbx_path).exists() {
        return;
    }
    let loader = crate::asset::fbx::FbxAssetLoader::new();
    let asset = loader.load(fbx_path).unwrap();
    println!("=== ALL MESHES IN YUMEKA ({}) ===", asset.meshes.len());
    for (i, m) in asset.meshes.iter().enumerate() {
        for (pi, p) in m.primitives.iter().enumerate() {
            let mat_name = asset
                .materials
                .iter()
                .find(|mat| mat.id == p.material_id)
                .map(|mat| mat.name.as_str())
                .unwrap_or("?");
            println!(
                "  Mesh {} '{}' prim {} (id={:?}) verts={} mat='{}' bounds={:?}..{:?}",
                i, m.name, pi, p.id, p.vertex_count, mat_name, p.bounds.min, p.bounds.max
            );
        }
    }
}

#[test]
fn test_yumeka_skin_anchors_generation() {
    let Some(fbx_path) = yumeka_fbx_path() else {
        return;
    };
    let loader = crate::asset::fbx::FbxAssetLoader::new();
    let asset = loader.load(&fbx_path).unwrap();

    assert!(
        asset.body_primitive_id.is_some(),
        "Yumeka must have an identified body primitive"
    );
    let body_pid = asset.body_primitive_id.unwrap();

    let body_prim = asset
        .meshes
        .iter()
        .flat_map(|m| &m.primitives)
        .find(|p| p.id == body_pid)
        .expect("body primitive must exist");
    assert!(
        body_prim.vertex_count > 10_000,
        "Yumeka body primitive should be the largest mesh (got {} verts)",
        body_prim.vertex_count
    );

    let clearance = crate::asset::clearance::SKIN_ANCHOR_CLEARANCE;
    let containment = crate::asset::clearance::SKIN_ANCHOR_CONTAINMENT;

    for m in &asset.meshes {
        for p in &m.primitives {
            let clearance_bound = p
                .skin_anchors
                .as_ref()
                .map(|a| a.iter().filter(|x| x.body_vertex_idx != u32::MAX).count())
                .unwrap_or(0);
            let containment_bound = p
                .containment_anchors
                .as_ref()
                .map(|a| a.iter().filter(|x| x.body_vertex_idx != u32::MAX).count())
                .unwrap_or(0);
            match (clearance_bound, containment_bound) {
                (0, 0) => println!(
                    "Mesh '{}' prim {:?} verts={}: NO anchors",
                    m.name, p.id, p.vertex_count
                ),
                _ => println!(
                    "Mesh '{}' prim {:?} verts={}: clearance {} / {}, containment {} / {}",
                    m.name,
                    p.id,
                    p.vertex_count,
                    clearance_bound,
                    p.skin_anchors.as_ref().map(|a| a.len()).unwrap_or(0),
                    containment_bound,
                    p.containment_anchors.as_ref().map(|a| a.len()).unwrap_or(0)
                ),
            }
            // Clearance anchors (Phase 1 + outer layers) hold a positive
            // minimum floor and reference `body_primitive_id`;
            // containment anchors hold a rest-derived target of either
            // sign and reference `containment_primitive_id`.
            for a in p
                .skin_anchors
                .iter()
                .flatten()
                .filter(|x| x.body_vertex_idx != u32::MAX)
            {
                assert_eq!(a.mode, clearance, "skin_anchors must be clearance-mode");
                assert!(
                    a.min_clearance >= 0.002,
                    "clearance anchor below 2mm floor: {}",
                    a.min_clearance
                );
            }
            for a in p
                .containment_anchors
                .iter()
                .flatten()
                .filter(|x| x.body_vertex_idx != u32::MAX)
            {
                assert_eq!(
                    a.mode, containment,
                    "containment_anchors must be containment-mode"
                );
            }
            // Each anchor set indexes its OWN parent's vertex buffer —
            // the middle layer of a 3-layer stack has two different
            // parents (clearance vs containment).
            let parent_verts = |ppid: Option<crate::asset::PrimitiveId>| {
                ppid.and_then(|pid| {
                    asset
                        .meshes
                        .iter()
                        .flat_map(|pm| &pm.primitives)
                        .find(|pp| pp.id == pid)
                        .and_then(|pp| pp.vertices.as_ref())
                        .map(|vd| vd.positions.len())
                })
            };
            for (anchors, parent_len) in [
                (&p.skin_anchors, parent_verts(p.body_primitive_id)),
                (
                    &p.containment_anchors,
                    parent_verts(p.containment_primitive_id),
                ),
            ] {
                let Some(parent_len) = parent_len else {
                    continue;
                };
                if let Some(ref anc) = anchors {
                    let max_idx = anc
                        .iter()
                        .filter(|x| x.body_vertex_idx != u32::MAX)
                        .map(|x| x.body_vertex_idx)
                        .max()
                        .unwrap_or(0);
                    assert!(
                        (max_idx as usize) < parent_len,
                        "anchor index {} out of parent bounds (parent has {} verts)",
                        max_idx,
                        parent_len
                    );
                }
            }
        }
    }

    // Phase 1 must bind at least one garment against the body surface.
    let body_anchored: usize = asset
        .meshes
        .iter()
        .flat_map(|m| &m.primitives)
        .filter(|p| p.body_primitive_id == Some(body_pid))
        .map(|p| {
            p.skin_anchors
                .as_ref()
                .map(|a| a.iter().filter(|x| x.body_vertex_idx != u32::MAX).count())
                .unwrap_or(0)
        })
        .sum();
    assert!(
        body_anchored > 1000,
        "Expected a garment with >1000 body-bound anchors (skirt), got {}",
        body_anchored
    );
}

#[test]
fn test_yumeka_anti_penetration_projection_on_leg_lift() {
    let Some(fbx_path) = yumeka_fbx_path() else {
        return;
    };

    let loader = crate::asset::fbx::FbxAssetLoader::new();
    let asset = loader.load(&fbx_path).expect("Failed to load Yumeka FBX");

    let body_pid = asset
        .body_primitive_id
        .expect("body_primitive_id must be identified");
    let body_prim = asset
        .meshes
        .iter()
        .flat_map(|m| &m.primitives)
        .find(|p| p.id == body_pid)
        .expect("body primitive must exist");

    // The Phase-1 garment with the most body-bound anchors is the skirt.
    let (skirt_mesh_name, skirt_prim_id) = asset
        .meshes
        .iter()
        .flat_map(|m| m.primitives.iter().map(move |p| (&m.name, p)))
        .filter(|(_, p)| p.body_primitive_id == Some(body_pid))
        .max_by_key(|(_, p)| {
            p.skin_anchors
                .as_ref()
                .map(|a| a.iter().filter(|x| x.body_vertex_idx != u32::MAX).count())
                .unwrap_or(0)
        })
        .map(|(name, p)| (name.clone(), p.id))
        .expect("at least one garment must be anchored against the body");
    let skirt_prim = asset
        .meshes
        .iter()
        .flat_map(|m| &m.primitives)
        .find(|p| p.id == skirt_prim_id)
        .unwrap();
    let anchors = skirt_prim
        .skin_anchors
        .as_ref()
        .expect("garment must have skin anchors");
    println!(
        "Phase-1 garment under test: '{}' (prim {:?})",
        skirt_mesh_name, skirt_prim_id
    );

    // 1. Create avatar instance and pose: lift LeftUpperLeg forward by 45 degrees
    let mut avatar = crate::avatar::AvatarInstance::new(
        crate::avatar::AvatarInstanceId(1),
        std::sync::Arc::clone(&asset),
    );

    let l_leg_idx = find_humanoid_node(
        &asset,
        crate::asset::HumanoidBone::LeftUpperLeg,
        "UpperLeg_L",
    )
    .expect("left upper leg node must be identifiable");

    let angle_rad = 45.0f32.to_radians();
    let sin_half = (angle_rad * 0.5).sin();
    let cos_half = (angle_rad * 0.5).cos();
    let rot_x = [sin_half, 0.0, 0.0, cos_half];
    avatar.pose.local_transforms[l_leg_idx].rotation =
        crate::math_utils::quat_mul(&rot_x, &asset.skeleton.nodes[l_leg_idx].rest_local.rotation);
    avatar.compute_global_pose();
    avatar.build_skinning_matrices();

    // 2. Compute skinned world vertices for both body and skirt under this lifted pose
    let body_world = crate::asset::clearance::compute_rest_world_vertices(
        body_prim,
        &avatar.pose.skinning_matrices,
    );
    let skirt_world = crate::asset::clearance::compute_rest_world_vertices(
        skirt_prim,
        &avatar.pose.skinning_matrices,
    );

    assert_eq!(body_world.len(), body_prim.vertex_count as usize);
    assert_eq!(skirt_world.len(), skirt_prim.vertex_count as usize);

    // 3. CPU mirror of the transform_cs clearance projection — including
    // the fold-region normal flip and the parent-position sanity band the
    // shader applies (pipeline.rs transform_cs skin-anchor block).
    let mut penetration_threat_count = 0usize;
    let mut max_push_distance = 0.0f32;

    for (vid, &(mut skirt_p, _skirt_n)) in skirt_world.iter().enumerate() {
        let anc = anchors[vid];
        if anc.body_vertex_idx != u32::MAX && anc.weight > 1e-4 {
            let (bp, bn_raw) = body_world[anc.body_vertex_idx as usize];
            let bn_len =
                (bn_raw[0] * bn_raw[0] + bn_raw[1] * bn_raw[1] + bn_raw[2] * bn_raw[2]).sqrt();
            let bp_len = crate::math_utils::vec3_length(&bp);
            if bn_len <= 1e-4 || bp_len <= 1e-3 || bp_len >= 10.0 {
                continue;
            }
            let bn = [bn_raw[0] / bn_len, bn_raw[1] / bn_len, bn_raw[2] / bn_len];
            let diff = [skirt_p[0] - bp[0], skirt_p[1] - bp[1], skirt_p[2] - bp[2]];
            let clearance = crate::math_utils::vec3_dot(&diff, &bn);

            if clearance < anc.min_clearance {
                penetration_threat_count += 1;
                let push = (anc.min_clearance - clearance) * anc.weight;
                max_push_distance = max_push_distance.max(push);
                skirt_p[0] += bn[0] * push;
                skirt_p[1] += bn[1] * push;
                skirt_p[2] += bn[2] * push;

                // After projection, clearance must satisfy min_clearance
                let new_diff = [skirt_p[0] - bp[0], skirt_p[1] - bp[1], skirt_p[2] - bp[2]];
                let new_clearance = crate::math_utils::vec3_dot(&new_diff, &bn);
                assert!(
                    new_clearance >= anc.min_clearance - 1e-4,
                    "After projection, clearance ({}) must be >= min_clearance ({})",
                    new_clearance,
                    anc.min_clearance
                );
            }
        }
    }

    println!(
        "Yumeka leg-lift anti-penetration: {} vertices guarded from body penetration, max push = {:.2} mm",
        penetration_threat_count,
        max_push_distance * 1000.0
    );

    assert!(
        penetration_threat_count > 0,
        "Expected leg lift to trigger anti-penetration constraints on skirt vertices"
    );
}

#[test]
fn test_inspect_shirt_blazer_elbow() {
    let fbx_path = "sample_data/YUMEKA_v1.0.1/FBX/Yumeka_v1.0.fbx";
    if !Path::new(fbx_path).exists() {
        return;
    }

    let loader = crate::asset::fbx::FbxAssetLoader::new();
    let asset = loader.load(fbx_path).expect("Failed to load Yumeka FBX");

    let m51 = asset
        .meshes
        .iter()
        .find(|m| m.name == "Circle.051")
        .unwrap();
    let m57 = asset
        .meshes
        .iter()
        .find(|m| m.name == "Circle.057")
        .unwrap();

    let p51 = &m51.primitives[0];
    let p57 = &m57.primitives[0];

    let l_forearm = asset
        .skeleton
        .nodes
        .iter()
        .position(|n| n.name == "LowerArm_L")
        .unwrap();
    let _l_upperarm = asset
        .skeleton
        .nodes
        .iter()
        .position(|n| n.name == "UpperArm_L")
        .unwrap();

    let node_count = asset.skeleton.nodes.len();
    let locals: Vec<_> = asset
        .skeleton
        .nodes
        .iter()
        .map(|n| n.rest_local.clone())
        .collect();
    let mut globals = vec![crate::asset::identity_matrix(); node_count];
    crate::avatar::pose::compute_global_transforms(&asset.skeleton, &locals, &mut globals);
    let mut skinning = vec![crate::asset::identity_matrix(); node_count];
    crate::avatar::pose::build_skinning_matrices(&asset.skeleton, &globals, &mut skinning);

    let v51_rest = crate::asset::clearance::compute_rest_world_vertices(p51, &skinning);
    let v57_rest = crate::asset::clearance::compute_rest_world_vertices(p57, &skinning);

    let elbow_pos = [
        globals[l_forearm][3][0],
        globals[l_forearm][3][1],
        globals[l_forearm][3][2],
    ];
    println!("Elbow world pos at rest: {:?}", elbow_pos);

    let mut dist51: Vec<f32> = Vec::new();
    for &(p, _) in &v51_rest {
        let diff = crate::math_utils::vec3_sub(&p, &elbow_pos);
        let d = crate::math_utils::vec3_length(&diff);
        if d < 0.08 {
            dist51.push(d);
        }
    }
    let mut dist57: Vec<f32> = Vec::new();
    for &(p, _) in &v57_rest {
        let diff = crate::math_utils::vec3_sub(&p, &elbow_pos);
        let d = crate::math_utils::vec3_length(&diff);
        if d < 0.08 {
            dist57.push(d);
        }
    }
    dist51.sort_by(|a: &f32, b: &f32| a.partial_cmp(b).unwrap());
    dist57.sort_by(|a: &f32, b: &f32| a.partial_cmp(b).unwrap());

    println!("Within 8cm of elbow: Circle.051 has {} verts (median dist={:?}), Circle.057 has {} verts (median dist={:?})",
        dist51.len(), dist51.get(dist51.len() / 2),
        dist57.len(), dist57.get(dist57.len() / 2)
    );

    let is_51_inner = dist51.get(dist51.len() / 2) < dist57.get(dist57.len() / 2);
    let (inner_p, outer_p) = if is_51_inner { (p57, p51) } else { (p51, p57) };
    let (inner_name, outer_name) = if is_51_inner {
        ("Circle.057 (shirt)", "Circle.051 (blazer)")
    } else {
        ("Circle.051 (shirt)", "Circle.057 (blazer)")
    };
    println!("Identification: inner={}, outer={}", inner_name, outer_name);

    // Test elbow curling along pitch/yaw/roll axes
    // In human anatomy, elbow flexion is around the local axis perpendicular to bone length
    for (axis_name, axis) in [
        ("local_X", [1.0f32, 0.0, 0.0]),
        ("local_Y", [0.0, 1.0f32, 0.0]),
        ("local_Z", [0.0, 0.0, 1.0f32]),
    ] {
        for deg in [45.0f32, 90.0f32, 120.0f32] {
            let mut avatar = crate::avatar::AvatarInstance::new(
                crate::avatar::AvatarInstanceId(1),
                std::sync::Arc::clone(&asset),
            );
            let angle_rad = deg.to_radians();
            let sin_half = (angle_rad * 0.5).sin();
            let cos_half = (angle_rad * 0.5).cos();
            let q = [
                axis[0] * sin_half,
                axis[1] * sin_half,
                axis[2] * sin_half,
                cos_half,
            ];
            avatar.pose.local_transforms[l_forearm].rotation = crate::math_utils::quat_mul(
                &q,
                &asset.skeleton.nodes[l_forearm].rest_local.rotation,
            );
            avatar.compute_global_pose();
            avatar.build_skinning_matrices();

            let inner_bent = crate::asset::clearance::compute_rest_world_vertices(
                inner_p,
                &avatar.pose.skinning_matrices,
            );
            let outer_bent = crate::asset::clearance::compute_rest_world_vertices(
                outer_p,
                &avatar.pose.skinning_matrices,
            );

            // Check distance of bent forearm node
            let cur_elbow = [
                avatar.pose.global_transforms[l_forearm][3][0],
                avatar.pose.global_transforms[l_forearm][3][1],
                avatar.pose.global_transforms[l_forearm][3][2],
            ];

            // For inner vertices near elbow, find nearest outer vertex and check if inner is outside outer
            let mut penetrations = 0usize;
            let mut max_pen_dist = 0.0f32;

            for (_vi, &(ip, _inrm)) in inner_bent.iter().enumerate() {
                let diff_e = crate::math_utils::vec3_sub(&ip, &cur_elbow);
                if crate::math_utils::vec3_length(&diff_e) > 0.07 {
                    continue;
                }
                let mut best_outer = None;
                let mut min_d = f32::MAX;
                for (_oi, &(op, onrm)) in outer_bent.iter().enumerate() {
                    let d = crate::math_utils::vec3_length(&crate::math_utils::vec3_sub(&ip, &op));
                    if d < min_d {
                        min_d = d;
                        best_outer = Some((op, onrm));
                    }
                }
                if let Some((op, onrm)) = best_outer {
                    let diff_io = crate::math_utils::vec3_sub(&ip, &op);
                    let pen = crate::math_utils::vec3_dot(&diff_io, &onrm);
                    if pen > 0.001 {
                        penetrations += 1;
                        if pen > max_pen_dist {
                            max_pen_dist = pen;
                        }
                    }
                }
            }

            if penetrations > 0 {
                println!(
                    "  Bend {} by {} deg: {} inner verts penetrated outer, max poke = {:.2} mm",
                    axis_name,
                    deg,
                    penetrations,
                    max_pen_dist * 1000.0
                );
            }
        }
    }

    // Weight comparison around elbow for Circle.051 and Circle.057
    println!("=== ELBOW WEIGHT DISTRIBUTION COMPARISON ===");
    let ivd = inner_p.vertices.as_ref().unwrap();
    let ovd = outer_p.vertices.as_ref().unwrap();

    let mut inner_elbow_weights = Vec::new();
    for (vi, _p) in ivd.positions.iter().enumerate() {
        let mut w_upper = 0.0f32;
        let mut w_lower = 0.0f32;
        let mut w_twist = 0.0f32;
        for s in 0..4 {
            let ji = ivd.joint_indices[vi][s] as usize;
            let w = ivd.joint_weights[vi][s];
            if ji < asset.skeleton.nodes.len() {
                let name = &asset.skeleton.nodes[ji].name;
                if name.contains("UpperArm") && !name.contains("twist") {
                    w_upper += w;
                } else if name.contains("LowerArm") && !name.contains("twist") {
                    w_lower += w;
                } else if name.contains("twist") {
                    w_twist += w;
                }
            }
        }
        if w_upper > 0.05 && w_lower > 0.05 {
            inner_elbow_weights.push((vi, w_upper, w_lower, w_twist));
        }
    }

    let mut outer_elbow_weights = Vec::new();
    for (vi, _p) in ovd.positions.iter().enumerate() {
        let mut w_upper = 0.0f32;
        let mut w_lower = 0.0f32;
        let mut w_twist = 0.0f32;
        for s in 0..4 {
            let ji = ovd.joint_indices[vi][s] as usize;
            let w = ovd.joint_weights[vi][s];
            if ji < asset.skeleton.nodes.len() {
                let name = &asset.skeleton.nodes[ji].name;
                if name.contains("UpperArm") && !name.contains("twist") {
                    w_upper += w;
                } else if name.contains("LowerArm") && !name.contains("twist") {
                    w_lower += w;
                } else if name.contains("twist") {
                    w_twist += w;
                }
            }
        }
        if w_upper > 0.05 && w_lower > 0.05 {
            outer_elbow_weights.push((vi, w_upper, w_lower, w_twist));
        }
    }

    println!(
        "Joint transition vertices (UpperArm > 0.05 and LowerArm > 0.05): Inner (shirt) = {}, Outer (blazer) = {}",
        inner_elbow_weights.len(), outer_elbow_weights.len()
    );
    if let (Some(iw), Some(ow)) = (inner_elbow_weights.first(), outer_elbow_weights.first()) {
        println!(
            "Sample Inner weight: Upper={:.3}, Lower={:.3}, Twist={:.3}",
            iw.1, iw.2, iw.3
        );
        println!(
            "Sample Outer weight: Upper={:.3}, Lower={:.3}, Twist={:.3}",
            ow.1, ow.2, ow.3
        );
    }
}

#[test]
fn test_layered_clothing_clearance_e2e() {
    let Some(fbx_path) = yumeka_fbx_path() else {
        return;
    };

    let loader = crate::asset::fbx::FbxAssetLoader::new();
    let asset = loader.load_with_progress(&fbx_path, |_| {}).unwrap();

    println!("=== AVATAR MESH INVENTORY ===");
    for m in &asset.meshes {
        for p in &m.primitives {
            let mat_name = asset
                .materials
                .iter()
                .find(|mat| mat.id == p.material_id)
                .map(|mat| mat.name.as_str())
                .unwrap_or("unknown");
            println!(
                "Mesh '{:20}' prim_id={:?} verts={:5} mat='{}' parent={:?}",
                m.name, p.id, p.vertex_count, mat_name, p.body_primitive_id
            );
        }
    }
    println!("=============================");

    // Dynamically discover the layered pair that matters for the
    // shirt-through-blazer symptom: the INNER layer is a primitive
    // carrying containment anchors whose OUTER parent carries no
    // containment of its own (i.e. the outermost garment of the stack).
    // The outer must in turn carry clearance anchors pointing back at
    // the inner (the mutual pairing Phase 2 establishes).
    let containment = crate::asset::clearance::SKIN_ANCHOR_CONTAINMENT;

    let prim_of = |pid: crate::asset::PrimitiveId| {
        asset
            .meshes
            .iter()
            .flat_map(|m| &m.primitives)
            .find(|p| p.id == pid)
    };
    let bound_count = |anchors: &Option<Vec<crate::asset::SkinAnchor>>| {
        anchors
            .as_ref()
            .map(|a| a.iter().filter(|x| x.body_vertex_idx != u32::MAX).count())
            .unwrap_or(0)
    };

    let inner_prim = asset
        .meshes
        .iter()
        .flat_map(|m| &m.primitives)
        .find(|p| {
            p.containment_anchors.is_some()
                && p.containment_primitive_id
                    .and_then(|outer_pid| {
                        prim_of(outer_pid).map(|o| o.containment_anchors.is_none())
                    })
                    .unwrap_or(false)
        })
        .expect("no inner layer carries containment anchors against an outermost garment");
    let inner_pid = inner_prim.id;
    let outer_pid = inner_prim
        .containment_primitive_id
        .expect("containment-carrying primitive must reference its outer layer");
    let outer_prim = prim_of(outer_pid).expect("outer layer primitive must exist");
    assert!(
        outer_prim.skin_anchors.is_some(),
        "outer layer must carry clearance anchors"
    );
    assert_eq!(
        outer_prim.body_primitive_id,
        Some(inner_pid),
        "outer layer must reference the inner layer back (mutual pairing)"
    );
    assert!(
        inner_prim
            .containment_anchors
            .as_ref()
            .and_then(|a| a.iter().find(|x| x.body_vertex_idx != u32::MAX))
            .map(|x| x.mode)
            == Some(containment),
        "containment anchors must be flagged with the containment mode"
    );

    let mesh_name_of = |pid| {
        asset
            .meshes
            .iter()
            .find(|m| m.primitives.iter().any(|p| p.id == pid))
            .map(|m| m.name.clone())
            .unwrap_or_default()
    };
    let inner_anchors = inner_prim.containment_anchors.as_ref().unwrap();
    let outer_anchors = outer_prim.skin_anchors.as_ref().unwrap();
    let inner_bound = bound_count(&inner_prim.containment_anchors);
    let outer_bound = bound_count(&outer_prim.skin_anchors);
    println!(
        "HGCF pairing: outer='{}' ({}/{} clearance anchors) <-> inner='{}' ({}/{} containment anchors)",
        mesh_name_of(outer_pid),
        outer_bound,
        outer_anchors.len(),
        mesh_name_of(inner_pid),
        inner_bound,
        inner_anchors.len()
    );
    assert!(
        outer_bound > 2000,
        "Expected >2000 clearance anchors on the outer layer, got {}",
        outer_bound
    );
    assert!(
        inner_bound > 2000,
        "Expected >2000 containment anchors on the inner layer, got {}",
        inner_bound
    );

    // Bend the left forearm 90 degrees around local X.
    let l_forearm = find_humanoid_node(
        &asset,
        crate::asset::HumanoidBone::LeftLowerArm,
        "LowerArm_L",
    )
    .expect("left forearm node must be identifiable");
    let mut avatar = crate::avatar::AvatarInstance::new(
        crate::avatar::AvatarInstanceId(1),
        std::sync::Arc::clone(&asset),
    );
    avatar.pose.local_transforms = asset
        .skeleton
        .nodes
        .iter()
        .map(|n| n.rest_local.clone())
        .collect();
    let angle_rad = 90.0f32.to_radians();
    let sin_half = (angle_rad * 0.5).sin();
    let cos_half = (angle_rad * 0.5).cos();
    let q = [1.0 * sin_half, 0.0, 0.0, cos_half];
    avatar.pose.local_transforms[l_forearm].rotation =
        crate::math_utils::quat_mul(&q, &asset.skeleton.nodes[l_forearm].rest_local.rotation);
    avatar.compute_global_pose();
    avatar.build_skinning_matrices();

    let inner_bent = crate::asset::clearance::compute_rest_world_vertices(
        inner_prim,
        &avatar.pose.skinning_matrices,
    );
    let outer_bent = crate::asset::clearance::compute_rest_world_vertices(
        outer_prim,
        &avatar.pose.skinning_matrices,
    );

    // ---- GPU mirror pass A: outer clearance projection (transform_cs
    // mode-0 branch: sanity band + fold-region normal flip + push out).
    let mut outer_projected = outer_bent.clone();
    let mut pushed_count = 0usize;
    let mut max_push_dist = 0.0f32;
    for (vi, anc) in outer_anchors.iter().enumerate() {
        if anc.body_vertex_idx != u32::MAX && anc.weight > 1e-4 {
            let (bp, bn_raw) = inner_bent[anc.body_vertex_idx as usize];
            let wp = outer_projected[vi].0;
            let bn_len = crate::math_utils::vec3_length(&bn_raw);
            let bp_len = crate::math_utils::vec3_length(&bp);
            if bn_len <= 1e-4 || bp_len <= 1e-3 || bp_len >= 10.0 {
                continue;
            }
            let bn = [bn_raw[0] / bn_len, bn_raw[1] / bn_len, bn_raw[2] / bn_len];
            let diff = [wp[0] - bp[0], wp[1] - bp[1], wp[2] - bp[2]];
            let c = crate::math_utils::vec3_dot(&diff, &bn);
            if c < anc.min_clearance {
                let push = (anc.min_clearance - c) * anc.weight;
                outer_projected[vi].0[0] += bn[0] * push;
                outer_projected[vi].0[1] += bn[1] * push;
                outer_projected[vi].0[2] += bn[2] * push;
                pushed_count += 1;
                if push > max_push_dist {
                    max_push_dist = push;
                }
            }
        }
    }
    println!(
        "GPU clearance projection: pushed {} outer vertices outward (max push = {:.2} mm)",
        pushed_count,
        max_push_dist * 1000.0
    );

    // ---- GPU mirror pass B: inner containment clamp (transform_cs
    // mode-1 branch). On the GPU this reads the parent's previous-frame
    // VBO; under a held pose that converges to the same-frame projected
    // outer surface used here.
    let mut inner_contained = inner_bent.clone();
    let mut clamp_count = 0usize;
    let mut max_clamp_dist = 0.0f32;
    for (vi, anc) in inner_anchors.iter().enumerate() {
        if anc.body_vertex_idx != u32::MAX && anc.weight > 1e-4 {
            let (op, on_raw) = outer_projected[anc.body_vertex_idx as usize];
            let ip = inner_contained[vi].0;
            let on_len = crate::math_utils::vec3_length(&on_raw);
            let op_len = crate::math_utils::vec3_length(&op);
            if on_len <= 1e-4 || op_len <= 1e-3 || op_len >= 10.0 {
                continue;
            }
            let on = [on_raw[0] / on_len, on_raw[1] / on_len, on_raw[2] / on_len];
            let diff = [ip[0] - op[0], ip[1] - op[1], ip[2] - op[2]];
            let c = crate::math_utils::vec3_dot(&diff, &on);
            if c > anc.min_clearance {
                let push = (c - anc.min_clearance) * anc.weight;
                inner_contained[vi].0[0] -= on[0] * push;
                inner_contained[vi].0[1] -= on[1] * push;
                inner_contained[vi].0[2] -= on[2] * push;
                clamp_count += 1;
                if push > max_clamp_dist {
                    max_clamp_dist = push;
                }
            }
        }
    }
    println!(
        "GPU containment clamp: clamped {} inner vertices back inside (max clamp = {:.2} mm)",
        clamp_count,
        max_clamp_dist * 1000.0
    );

    // Measure penetrations near the elbow: an inner vertex counts as
    // poking out when it sits beyond its nearest outer vertex along the
    // outer normal.
    let cur_elbow = [
        avatar.pose.global_transforms[l_forearm][3][0],
        avatar.pose.global_transforms[l_forearm][3][1],
        avatar.pose.global_transforms[l_forearm][3][2],
    ];

    fn measure_pen(
        inner: &[([f32; 3], [f32; 3])],
        outer: &[([f32; 3], [f32; 3])],
        elbow: [f32; 3],
    ) -> (usize, f32) {
        let mut pen = 0usize;
        let mut max_pen = 0.0f32;
        for &(ip, _) in inner {
            let d = crate::math_utils::vec3_length(&crate::math_utils::vec3_sub(&ip, &elbow));
            if d > 0.07 {
                continue;
            }
            let mut best: Option<([f32; 3], [f32; 3])> = None;
            let mut min_d = f32::MAX;
            for &(op, onrm) in outer {
                let dd = crate::math_utils::vec3_length(&crate::math_utils::vec3_sub(&ip, &op));
                if dd < min_d {
                    min_d = dd;
                    best = Some((op, onrm));
                }
            }
            if let Some((op, onrm)) = best {
                let diff_io = crate::math_utils::vec3_sub(&ip, &op);
                let p = crate::math_utils::vec3_dot(&diff_io, &onrm);
                if p > 0.0005 {
                    pen += 1;
                    if p > max_pen {
                        max_pen = p;
                    }
                }
            }
        }
        (pen, max_pen)
    }

    let (pen_before, max_before) = measure_pen(&inner_bent, &outer_bent, cur_elbow);
    let (pen_clearance_only, _) = measure_pen(&inner_bent, &outer_projected, cur_elbow);
    let (pen_after, max_after) = measure_pen(&inner_contained, &outer_projected, cur_elbow);

    // Diagnose the worst surviving pokes: is the clamp target
    // legitimately positive (cuff-style rest exposure), or does the
    // rest correspondence point somewhere unrepresentative after the
    // bend?
    let mut worst: Vec<(f32, usize)> = Vec::new();
    for (vi, &(ip, _)) in inner_contained.iter().enumerate() {
        let d = crate::math_utils::vec3_length(&crate::math_utils::vec3_sub(&ip, &cur_elbow));
        if d > 0.07 {
            continue;
        }
        let mut best: Option<([f32; 3], [f32; 3])> = None;
        let mut min_d = f32::MAX;
        for &(op, onrm) in outer_projected.iter() {
            let dd = crate::math_utils::vec3_length(&crate::math_utils::vec3_sub(&ip, &op));
            if dd < min_d {
                min_d = dd;
                best = Some((op, onrm));
            }
        }
        if let Some((op, onrm)) = best {
            let p = crate::math_utils::vec3_dot(&crate::math_utils::vec3_sub(&ip, &op), &onrm);
            if p > 0.0005 {
                worst.push((p, vi));
            }
        }
    }
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    // Rest-skinned geometry of both layers, for classifying residual
    // pokes and asserting rest-pose stability.
    let node_count = asset.skeleton.nodes.len();
    let rest_locals: Vec<_> = asset
        .skeleton
        .nodes
        .iter()
        .map(|n| n.rest_local.clone())
        .collect();
    let mut rest_globals = vec![crate::asset::identity_matrix(); node_count];
    crate::avatar::pose::compute_global_transforms(
        &asset.skeleton,
        &rest_locals,
        &mut rest_globals,
    );
    let mut rest_skinning = vec![crate::asset::identity_matrix(); node_count];
    crate::avatar::pose::build_skinning_matrices(
        &asset.skeleton,
        &rest_globals,
        &mut rest_skinning,
    );
    let inner_rest_skinned =
        crate::asset::clearance::compute_rest_world_vertices(inner_prim, &rest_skinning);
    let outer_rest_skinned =
        crate::asset::clearance::compute_rest_world_vertices(outer_prim, &rest_skinning);

    // Rest-pose stability: the containment targets are rest-derived
    // (`rest_clearance + slack`), so at rest no vertex may move.
    let mut rest_clamp_max = 0.0f32;
    for (vi, anc) in inner_anchors.iter().enumerate() {
        if anc.body_vertex_idx == u32::MAX || anc.weight <= 1e-4 {
            continue;
        }
        let (ip, _) = inner_rest_skinned[vi];
        let (op, on) = outer_rest_skinned[anc.body_vertex_idx as usize];
        let on_len = crate::math_utils::vec3_length(&on);
        if on_len <= 1e-4 {
            continue;
        }
        let on = [on[0] / on_len, on[1] / on_len, on[2] / on_len];
        let c = crate::math_utils::vec3_dot(&crate::math_utils::vec3_sub(&ip, &op), &on);
        if c > anc.min_clearance {
            rest_clamp_max = rest_clamp_max.max(c - anc.min_clearance);
        }
    }
    assert!(
        rest_clamp_max < 1e-6,
        "containment must never fire on the rest pose (max displacement {} mm)",
        rest_clamp_max * 1000.0
    );

    // Classify residual pokes by anchor relationship. The poke metric
    // measures against the NEAREST bent outer vertex's tangent plane,
    // which misfires in fold regions; the mechanism's actual guarantee
    // is per-anchor: after clamping, the vertex sits at most
    // `min_clearance` outside its ANCHORED outer vertex's plane. A
    // residual is therefore only a mechanism failure when it violates
    // its own anchor plane (c > target); otherwise it is either a fold
    // metric artifact (anchor relationship intact) or an open-corner
    // region (armpit gusset / cuff edges — the outer garment carries
    // no surface to contain against).
    let mut residual_violations = 0usize;
    let mut residual_fold_artifacts = 0usize;
    let mut residual_open_corner = 0usize;
    for &(pen, vi) in &worst {
        let anc = &inner_anchors[vi];
        let anchored_near = anc.body_vertex_idx != u32::MAX && {
            let (rp, _) = inner_rest_skinned[vi];
            let (op, _) = outer_rest_skinned[anc.body_vertex_idx as usize];
            crate::math_utils::vec3_length(&crate::math_utils::vec3_sub(&rp, &op)) < 0.03
        };
        if !anchored_near {
            residual_open_corner += 1;
            continue;
        }
        let (op, on) = outer_projected[anc.body_vertex_idx as usize];
        let ip = inner_contained[vi].0;
        let c = crate::math_utils::vec3_dot(&crate::math_utils::vec3_sub(&ip, &op), &on);
        if c > anc.min_clearance + 1e-4 {
            residual_violations += 1;
            println!(
                "  MECHANISM VIOLATION: vi={} pen={:.1}mm c={:.1}mm target={:.1}mm",
                vi,
                pen * 1000.0,
                c * 1000.0,
                anc.min_clearance * 1000.0
            );
        } else {
            residual_fold_artifacts += 1;
        }
    }
    println!(
        "Residual pokes: {} total, {} mechanism violations (must be 0), {} fold metric artifacts (anchor plane satisfied), {} open-corner (no outer surface)",
        worst.len(), residual_violations, residual_fold_artifacts, residual_open_corner
    );

    println!(
        "Penetration near elbow: BEFORE (no constraints) = {} verts (max poke = {:.2} mm)",
        pen_before,
        max_before * 1000.0
    );
    println!(
        "Penetration near elbow: clearance only          = {} verts",
        pen_clearance_only
    );
    println!(
        "Penetration near elbow: AFTER  clr + containment = {} verts (max poke = {:.2} mm)",
        pen_after,
        max_after * 1000.0
    );
    assert!(
        pen_before > 50,
        "Must reproduce elbow penetration before clearance"
    );
    assert_eq!(
        residual_violations, 0,
        "every anchored vertex must satisfy its containment plane after clamping"
    );
}
#[test]
fn test_skin_through_sleeve_at_elbow_bend() {
    let Some(fbx_path) = yumeka_fbx_path() else {
        return;
    };
    let loader = crate::asset::fbx::FbxAssetLoader::new();
    let asset = loader.load(&fbx_path).unwrap();

    let body_pid = asset.body_primitive_id.expect("body primitive");
    let body_prim = asset
        .meshes
        .iter()
        .flat_map(|m| &m.primitives)
        .find(|p| p.id == body_pid)
        .unwrap();

    // The layered stack (reuse the e2e discovery): outermost garment and
    // its inner (the middle layer).
    let prim_of = |pid: crate::asset::PrimitiveId| {
        asset
            .meshes
            .iter()
            .flat_map(|m| &m.primitives)
            .find(|p| p.id == pid)
    };
    let mid_prim = asset
        .meshes
        .iter()
        .flat_map(|m| &m.primitives)
        .find(|p| {
            p.containment_anchors.is_some()
                && p.containment_primitive_id
                    .and_then(|o| prim_of(o).map(|op| op.containment_anchors.is_none()))
                    .unwrap_or(false)
        })
        .expect("layered stack present");
    let outer_pid = mid_prim.containment_primitive_id.unwrap();
    let outer_prim = prim_of(outer_pid).unwrap();
    let outer_anchors = outer_prim.skin_anchors.as_ref().unwrap();
    let mid_anchors = mid_prim.containment_anchors.as_ref().unwrap();
    println!(
        "stack: body <- mid({}) <- outer({})",
        prim_name(&asset, mid_prim.id),
        prim_name(&asset, outer_pid)
    );

    // Bend BOTH elbows.
    let mut avatar = crate::avatar::AvatarInstance::new(
        crate::avatar::AvatarInstanceId(1),
        std::sync::Arc::clone(&asset),
    );
    avatar.pose.local_transforms = asset.skeleton.nodes.iter().map(|n| n.rest_local.clone()).collect();
    let angle = 90.0f32.to_radians();
    let (s, c) = ((angle * 0.5).sin(), (angle * 0.5).cos());
    let mut elbows = Vec::new();
    for (bone, suffix) in [
        (crate::asset::HumanoidBone::LeftLowerArm, "LowerArm_L"),
        (crate::asset::HumanoidBone::RightLowerArm, "LowerArm_R"),
    ] {
        let idx = find_humanoid_node(&asset, bone, suffix).expect("forearm node");
        let q = [s, 0.0, 0.0, c];
        avatar.pose.local_transforms[idx].rotation = crate::math_utils::quat_mul(
            &q,
            &asset.skeleton.nodes[idx].rest_local.rotation,
        );
    }
    avatar.compute_global_pose();
    avatar.build_skinning_matrices();

    let skin = crate::asset::clearance::compute_rest_world_vertices(body_prim, &avatar.pose.skinning_matrices);
    let mid_bent = crate::asset::clearance::compute_rest_world_vertices(mid_prim, &avatar.pose.skinning_matrices);
    let outer_bent = crate::asset::clearance::compute_rest_world_vertices(outer_prim, &avatar.pose.skinning_matrices);

    for bone in [crate::asset::HumanoidBone::LeftLowerArm, crate::asset::HumanoidBone::RightLowerArm] {
        let idx = find_humanoid_node(&asset, bone, "").expect("forearm node");
        elbows.push([
            avatar.pose.global_transforms[idx][3][0],
            avatar.pose.global_transforms[idx][3][1],
            avatar.pose.global_transforms[idx][3][2],
        ]);
    }

    // GPU mirror, matching transform_cs's sequential application:
    //   mid:   clearance vs BODY (skin) first, then containment vs outer
    //   outer: clearance vs the final mid
    let mid_clear_anchors = mid_prim.skin_anchors.as_ref();
    let mut mid_clearanced = mid_bent.clone();
    if let Some(anchors) = mid_clear_anchors {
        for (vi, anc) in anchors.iter().enumerate() {
            if anc.body_vertex_idx == u32::MAX || anc.weight <= 1e-4 {
                continue;
            }
            let (bp, bn_raw) = skin[anc.body_vertex_idx as usize];
            let bn_len = crate::math_utils::vec3_length(&bn_raw);
            let bp_len = crate::math_utils::vec3_length(&bp);
            if bn_len <= 1e-4 || bp_len <= 1e-3 || bp_len >= 10.0 {
                continue;
            }
            let bn = [bn_raw[0] / bn_len, bn_raw[1] / bn_len, bn_raw[2] / bn_len];
            let diff = crate::math_utils::vec3_sub(&mid_clearanced[vi].0, &bp);
            let clr = crate::math_utils::vec3_dot(&diff, &bn);
            if clr < anc.min_clearance {
                let push = (anc.min_clearance - clr) * anc.weight;
                for k in 0..3 {
                    mid_clearanced[vi].0[k] += bn[k] * push;
                }
            }
        }
    }

    // outer clearance off mid_bent, then mid containment vs
    // projected outer (converged same-frame stand-in).
    let mut outer_proj = outer_bent.clone();
    for (vi, anc) in outer_anchors.iter().enumerate() {
        if anc.body_vertex_idx == u32::MAX || anc.weight <= 1e-4 {
            continue;
        }
        let (bp, bn_raw) = mid_bent[anc.body_vertex_idx as usize];
        let bn_len = crate::math_utils::vec3_length(&bn_raw);
        let bp_len = crate::math_utils::vec3_length(&bp);
        if bn_len <= 1e-4 || bp_len <= 1e-3 || bp_len >= 10.0 {
            continue;
        }
        let bn = [bn_raw[0] / bn_len, bn_raw[1] / bn_len, bn_raw[2] / bn_len];
        let diff = crate::math_utils::vec3_sub(&outer_proj[vi].0, &bp);
        let clr = crate::math_utils::vec3_dot(&diff, &bn);
        if clr < anc.min_clearance {
            let push = (anc.min_clearance - clr) * anc.weight;
            for k in 0..3 {
                outer_proj[vi].0[k] += bn[k] * push;
            }
        }
    }
    let mut mid_contained = mid_clearanced.clone();
    for (vi, anc) in mid_anchors.iter().enumerate() {
        if anc.body_vertex_idx == u32::MAX || anc.weight <= 1e-4 {
            continue;
        }
        let (op, on_raw) = outer_proj[anc.body_vertex_idx as usize];
        let on_len = crate::math_utils::vec3_length(&on_raw);
        let op_len = crate::math_utils::vec3_length(&op);
        if on_len <= 1e-4 || op_len <= 1e-3 || op_len >= 10.0 {
            continue;
        }
        let on = [on_raw[0] / on_len, on_raw[1] / on_len, on_raw[2] / on_len];
        let diff = crate::math_utils::vec3_sub(&mid_contained[vi].0, &op);
        let c2 = crate::math_utils::vec3_dot(&diff, &on);
        if c2 > anc.min_clearance {
            let push = (c2 - anc.min_clearance) * anc.weight;
            for k in 0..3 {
                mid_contained[vi].0[k] -= on[k] * push;
            }
        }
    }

    // Skin-through measurement: skin verts near an elbow vs the nearest
    // garment vertex tangent plane.
    fn skin_through(
        skin: &[([f32; 3], [f32; 3])],
        garment: &[([f32; 3], [f32; 3])],
        elbows: &[[f32; 3]],
        radius: f32,
    ) -> (usize, f32) {
        skin_through_banded(skin, garment, elbows, radius, 0.0)
    }

    // `inner_band` restricts the count to skin verts whose distance to
    // the nearest elbow node is within [inner_band, radius) — the elbow
    // core vs the armhole/cuff periphery.
    #[allow(clippy::too_many_arguments)]
    fn skin_through_banded(
        skin: &[([f32; 3], [f32; 3])],
        garment: &[([f32; 3], [f32; 3])],
        elbows: &[[f32; 3]],
        radius: f32,
        inner_band: f32,
    ) -> (usize, f32) {
        let mut count = 0usize;
        let mut max_pen = 0.0f32;
        for &(sp, _) in skin {
            let ed = elbows
                .iter()
                .map(|e| crate::math_utils::vec3_length(&crate::math_utils::vec3_sub(&sp, e)))
                .fold(f32::MAX, f32::min);
            if ed >= radius || ed < inner_band {
                continue;
            }
            let mut best: Option<([f32; 3], [f32; 3])> = None;
            let mut min_d = f32::MAX;
            for &(gp, gn) in garment {
                let d = crate::math_utils::vec3_length(&crate::math_utils::vec3_sub(&sp, &gp));
                if d < min_d {
                    min_d = d;
                    best = Some((gp, gn));
                }
            }
            if let Some((gp, gn)) = best {
                let pen = crate::math_utils::vec3_dot(&crate::math_utils::vec3_sub(&sp, &gp), &gn);
                if pen > 0.0005 {
                    count += 1;
                    max_pen = max_pen.max(pen);
                }
            }
        }
        (count, max_pen)
    }

    // Diagnose: worst pokes AFTER body clearance — where are they and
    // what does their anchor look like?
    {
        let anchors = mid_prim.skin_anchors.as_ref().unwrap();
        let bound = anchors.iter().filter(|a| a.body_vertex_idx != u32::MAX).count();
        println!("mid body-anchors bound: {} / {}", bound, anchors.len());
        let mut worst: Vec<(f32, [f32; 3], usize)> = Vec::new();
        for &(sp, _) in &skin {
            if !elbows.iter().any(|e| crate::math_utils::vec3_length(&crate::math_utils::vec3_sub(&sp, e)) < 0.07) {
                continue;
            }
            let mut best: Option<([f32; 3], [f32; 3])> = None;
            let mut min_d = f32::MAX;
            for &(gp, gn) in &mid_clearanced {
                let d = crate::math_utils::vec3_length(&crate::math_utils::vec3_sub(&sp, &gp));
                if d < min_d {
                    min_d = d;
                    best = Some((gp, gn));
                }
            }
            if let Some((gp, gn)) = best {
                let pen = crate::math_utils::vec3_dot(&crate::math_utils::vec3_sub(&sp, &gp), &gn);
                if pen > 0.005 {
                    worst.push((pen, sp, 0));
                }
            }
        }
        worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        println!("worst skin-through pokes after body-clr (skin side): {}", worst.len());
        for &(pen, sp, _) in worst.iter().take(5) {
            // nearest mid vert -> its anchor
            let mut best_vi = 0usize;
            let mut min_d = f32::MAX;
            for (vi, &(mp, _)) in mid_clearanced.iter().enumerate() {
                let d = crate::math_utils::vec3_length(&crate::math_utils::vec3_sub(&sp, &mp));
                if d < min_d {
                    min_d = d;
                    best_vi = vi;
                }
            }
            let anc = &anchors[best_vi];
            let elbow_d = elbows
                .iter()
                .map(|e| crate::math_utils::vec3_length(&crate::math_utils::vec3_sub(&sp, e)))
                .fold(f32::MAX, f32::min);
            if anc.body_vertex_idx == u32::MAX {
                println!("  skin poke pen={:.1}mm at [{:.2} {:.2} {:.2}] (elbow_d={:.0}mm): nearest mid vert UNBOUND", pen * 1000.0, sp[0], sp[1], sp[2], elbow_d * 1000.0);
            } else {
                let (bp, _) = skin[anc.body_vertex_idx as usize];
                let anchor_d = crate::math_utils::vec3_length(&crate::math_utils::vec3_sub(&mid_clearanced[best_vi].0, &bp));
                println!(
                    "  skin poke pen={:.1}mm at [{:.2} {:.2} {:.2}] (elbow_d={:.0}mm): nearest mid vert {} anchor dist={:.0}mm min_clr={:.1}mm",
                    pen * 1000.0, sp[0], sp[1], sp[2], elbow_d * 1000.0, best_vi, anchor_d * 1000.0, anc.min_clearance * 1000.0
                );
            }
        }
    }

    // The assertion targets the ELBOW CORE (within 35mm of the joint)
    // — the reported symptom. The 35-70mm band is the armhole /
    // upper-arm periphery where the shirt carries no sensible coverage
    // and tangent-plane violations reshuffle with any sleeve
    // displacement; those are printed as reference values only.
    let (core_pure, _) = skin_through_banded(&skin, &mid_bent, &elbows, 0.035, 0.0);
    let (core_fixed, _) = skin_through_banded(&skin, &mid_clearanced, &elbows, 0.035, 0.0);
    let (core_contained, _) = skin_through_banded(&skin, &mid_contained, &elbows, 0.035, 0.0);
    println!(
        "elbow core (<35mm): pure={} +body-clr={} +containment={}",
        core_pure, core_fixed, core_contained
    );
    assert!(
        core_pure > 20,
        "90-degree bend must reproduce skin-through-sleeve at the elbow core (got {})",
        core_pure
    );
    assert_eq!(
        core_fixed, 0,
        "body clearance must eliminate elbow-core skin-through (got {})",
        core_fixed
    );
    for radius in [0.07_f32, 0.12] {
        let (a_cnt, a_max) = skin_through(&skin, &mid_bent, &elbows, radius);
        let (b_cnt, b_max) = skin_through(&skin, &mid_clearanced, &elbows, radius);
        let (c_cnt, c_max) = skin_through(&skin, &mid_contained, &elbows, radius);
        let (ob_cnt, ob_max) = skin_through(&skin, &outer_bent, &elbows, radius);
        let (op_cnt, op_max) = skin_through(&skin, &outer_proj, &elbows, radius);
        println!(
            "radius {:.0}mm: skin-through-mid: pure={} (max {:.1}mm) +body-clr={} (max {:.1}mm) +containment={} (max {:.1}mm) | skin-through-outer: pure={} (max {:.1}mm) cleared={} (max {:.1}mm)",
            radius * 1000.0,
            a_cnt, a_max * 1000.0,
            b_cnt, b_max * 1000.0,
            c_cnt, c_max * 1000.0,
            ob_cnt, ob_max * 1000.0,
            op_cnt, op_max * 1000.0
        );
        for band in [0.0_f32, 0.035] {
            let (ba, _) = skin_through_banded(&skin, &mid_bent, &elbows, radius, band);
            let (bb, _) = skin_through_banded(&skin, &mid_clearanced, &elbows, radius, band);
            let (bc, _) = skin_through_banded(&skin, &mid_contained, &elbows, radius, band);
            println!(
                "  band {:.0}-{:.0}mm: pure={} +body-clr={} +containment={}",
                band * 1000.0,
                radius * 1000.0,
                ba,
                bb,
                bc
            );
        }
    }
}

fn prim_name(asset: &crate::asset::AvatarAsset, pid: crate::asset::PrimitiveId) -> String {
    asset
        .meshes
        .iter()
        .find(|m| m.primitives.iter().any(|p| p.id == pid))
        .map(|m| m.name.clone())
        .unwrap_or_default()
}
