use std::path::Path;

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
    println!("Loaded AvatarAsset successfully! nodes: {}", asset.skeleton.nodes.len());

    let mut globals = vec![crate::asset::identity_matrix(); asset.skeleton.nodes.len()];
    let locals: Vec<crate::asset::Transform> = asset.skeleton.nodes.iter().map(|n| n.rest_local.clone()).collect();
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
            println!("Transform diff at node {} ('{}'): diff={}", i, node.element.name, diff);
            println!("  g col3 (t): [{}, {}, {}]", g[3][0], g[3][1], g[3][2]);
            println!("  u t: [{}, {}, {}]", u_n2w.m03, u_n2w.m13, u_n2w.m23);
        }
    }
    println!("Max transform diff between compute_global_transforms and ufbx: {}", max_diff);
    assert!(max_diff < 1e-4, "Global transforms diverge from ufbx: {}", max_diff);


    // Verify inverse bind matrices
    let hips_node_idx = asset.skeleton.nodes.iter().position(|n| n.name == "Hips").unwrap();
    let ibm_hips = &asset.skeleton.inverse_bind_matrices[hips_node_idx];
    assert_ne!(ibm_hips, &crate::asset::identity_matrix(), "Hips IBM must not be identity!");

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
    assert!(found_hips_joint, "Expected at least one vertex to be bound to Hips");

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
        assert!(mat.double_sided, "Material {} should be double_sided", mat.name);
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
    assert!(humanoid.bone_map.contains_key(&crate::asset::HumanoidBone::Hips));
    assert!(humanoid.bone_map.contains_key(&crate::asset::HumanoidBone::Head));

    // Verify expressions / blendshapes
    let exprs = &asset.default_expressions.expressions;
    assert!(!exprs.is_empty(), "Expressions must not be empty");
    assert!(exprs.iter().any(|e| e.name == "aa"), "Viseme 'aa' must be mapped");
    assert!(exprs.iter().any(|e| e.name == "blink"), "Blink must be mapped");

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
        .filter(|m| m.texture_bindings.base_color_texture.as_ref().is_some_and(|t| t.pixel_data.is_some()))
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
        .filter(|m| m.texture_bindings.base_color_texture.as_ref().is_some_and(|t| t.pixel_data.is_some()))
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
    assert!(found.is_some(), "Should find an avatar file in sample_data/YUMEKA_v1.0.1");
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
            !root_lower.contains("tongue") && !root_lower.contains("cheek") && !root_lower.contains("eye") && !root_lower.contains("jaw"),
            "Facial bone '{}' must not be a spring bone chain root", root_name
        );
        for &j in &sb.joints {
            let j_name = &asset.skeleton.nodes[j.0 as usize].name;
            let j_lower = j_name.to_lowercase();
            assert!(
                !j_lower.contains("tongue") && !j_lower.contains("cheek") && !j_lower.contains("eye") && !j_lower.contains("jaw"),
                "Facial bone '{}' must not be in spring bone joints", j_name
            );
        }
    }

    // 2. Verify standard preset 'aa' binds only the best morph target (no duplicate application)
    let aa_expr = asset.default_expressions.expressions.iter().find(|e| e.name == "aa").expect("Preset 'aa' must exist");
    assert_eq!(
        aa_expr.morph_binds.len(), 1,
        "Preset 'aa' must have exactly 1 morph bind per mesh node, got: {:?}", aa_expr.morph_binds
    );
    // Target 0 is vrc.v_aa
    assert_eq!(aa_expr.morph_binds[0].morph_target_index, 0, "Preset 'aa' should prioritize vrc.v_aa");
}

#[test]
fn test_inspect_skirt_meshes() {
    let fbx_path = "sample_data/YUMEKA_v1.0.1/FBX/Yumeka_v1.0.fbx";
    if !Path::new(fbx_path).exists() {
        return;
    }

    let loader = crate::asset::fbx::FbxAssetLoader::new();
    let asset = loader.load(fbx_path).unwrap();

    let skirt_nodes: Vec<(usize, &str)> = asset.skeleton.nodes.iter().enumerate()
        .filter(|(_, n)| n.name.to_lowercase().contains("skirt"))
        .map(|(i, n)| (i, n.name.as_str()))
        .collect();
    println!("=== SKIRT BONES IN SKELETON (total: {}) ===", skirt_nodes.len());
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
                        if w > 0.05 && skirt_nodes.iter().any(|&(s_idx, _)| slot as usize == s_idx) {
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
                let mat_name = asset.materials.iter()
                    .find(|m| m.id == prim.material_id)
                    .map(|m| m.name.as_str())
                    .unwrap_or("unknown");
                let raw_y_min = prim.vertices.as_ref().map(|v| v.positions.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min)).unwrap_or(0.0);
                let raw_y_max = prim.vertices.as_ref().map(|v| v.positions.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max)).unwrap_or(0.0);
                println!(
                    "  Mesh '{}' (id={:?}) prim {} (id={:?}): {} / {} vertices weighted to skirt (mat='{}', bounds={:?}..{:?}, raw_y={:.4}..{:.4})",
                    mesh.name, mesh.id, pi, prim.id, skirt_vert_count, total_vert_count, mat_name, prim.bounds.min, prim.bounds.max, raw_y_min, raw_y_max
                );
            }
        }
    }

    let circle_056 = asset.meshes.iter().find(|m| m.name == "Circle.056");
    assert!(circle_056.is_some(), "Yumeka should have Circle.056 mesh for skirt");
    let circle_056 = circle_056.unwrap();
    assert_eq!(circle_056.primitives.len(), 1);
    let prim = &circle_056.primitives[0];
    assert_eq!(prim.vertex_count, 2460);
    assert!(prim.bounds.min[1] > 0.55 && prim.bounds.max[1] < 0.85, "Skirt bounds must be around hips/thighs");
}

#[test]
fn test_inspect_hair_and_colliders() {
    let fbx_path = "sample_data/YUMEKA_v1.0.1/FBX/Yumeka_v1.0.fbx";
    if !Path::new(fbx_path).exists() {
        return;
    }

    let loader = crate::asset::fbx::FbxAssetLoader::new();
    let asset = loader.load(fbx_path).unwrap();

    println!("=== TOTAL COLLIDERS IN ASSET: {} ===", asset.colliders.len());
    for (i, c) in asset.colliders.iter().enumerate() {
        let node_name = &asset.skeleton.nodes[c.node.0 as usize].name;
        println!("  Collider {}: node='{}' (id={:?}), shape={:?}, offset={:?}", i, node_name, c.node, c.shape, c.offset);
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
            println!("  Node {}: {:?} ('{}') -> world pos: [{:.4}, {:.4}, {:.4}]", i, hb, node.name, pos[0], pos[1], pos[2]);
        }
    }

    println!("=== HAIR CHAIN REST POSITIONS ===");
    for (i, sb) in asset.spring_bones.iter().enumerate() {
        let root_name = &asset.skeleton.nodes[sb.chain_root.0 as usize].name;
        if root_name.to_lowercase().contains("hair") {
            let root_pos = crate::math_utils::mat4_translation(&avatar.pose.global_transforms[sb.chain_root.0 as usize]);
            let last_joint = sb.joints.last().copied().unwrap_or(sb.chain_root).0 as usize;
            let tip_pos = crate::math_utils::mat4_translation(&avatar.pose.global_transforms[last_joint]);
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
        .filter(|sb| asset.skeleton.nodes[sb.chain_root.0 as usize].name.to_lowercase().contains("hair"))
        .collect();
    assert!(!hair_springs.is_empty(), "Yumeka must have hair spring bones");
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
            Some(crate::asset::HumanoidBone::LeftUpperLeg) | Some(crate::asset::HumanoidBone::RightUpperLeg)
        ) {
            let node_pos = crate::math_utils::mat4_translation(&avatar.pose.global_transforms[node_idx]);
            let rotated_offset = mat4_dir(&avatar.pose.global_transforms[node_idx], &c.offset);
            let center = crate::math_utils::vec3_add(&node_pos, &rotated_offset);
            let up = mat4_dir(&avatar.pose.global_transforms[node_idx], &[0.0, 1.0, 0.0]);
            let up_len = crate::math_utils::vec3_length(&up);
            let up_norm = [up[0] / up_len, up[1] / up_len, up[2] / up_len];

            if let crate::asset::ColliderShape::Capsule { radius, height } = c.shape {
                let half_h = height * 0.5;
                let seg_a = crate::math_utils::vec3_sub(&center, &crate::math_utils::vec3_scale(&up_norm, half_h));
                let seg_b = crate::math_utils::vec3_add(&center, &crate::math_utils::vec3_scale(&up_norm, half_h));
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
            let col_names: Vec<String> = sb.collider_refs.iter().map(|r| {
                let c = asset.colliders.iter().find(|c| c.id == r.id).unwrap();
                let n = &asset.skeleton.nodes[c.node.0 as usize].name;
                format!("{}(id={})", n, r.id.0)
            }).collect();
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
    assert!(skirt_chain_count >= 10, "Expected at least 10 skirt chains in Yumeka");

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
            let last_pos = avatar.secondary_motion.spring_states[i].positions.last().copied().unwrap();
            println!("  Natural hang, {} tip pos: [{:.4}, {:.4}, {:.4}]", root_name, last_pos[0], last_pos[1], last_pos[2]);
            // Tip Y must be lower than root (hangs downward, Y < 0.70)
            assert!(
                last_pos[1] < 0.70,
                "Skirt chain '{}' tip Y ({}) should hang naturally downward, not pointing upward",
                root_name, last_pos[1]
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
            let mat_name = asset.materials.iter().find(|mat| mat.id == p.material_id).map(|mat| mat.name.as_str()).unwrap_or("?");
            println!("  Mesh {} '{}' prim {} (id={:?}) verts={} mat='{}' bounds={:?}..{:?}", i, m.name, pi, p.id, p.vertex_count, mat_name, p.bounds.min, p.bounds.max);
        }
    }
}

#[test]
fn test_yumeka_skin_anchors_generation() {
    let fbx_path = "sample_data/YUMEKA_v1.0.1/FBX/Yumeka_v1.0.fbx";
    if !Path::new(fbx_path).exists() {
        return;
    }
    let loader = crate::asset::fbx::FbxAssetLoader::new();
    let asset = loader.load(fbx_path).unwrap();

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
    assert_eq!(
        body_prim.vertex_count, 79116,
        "Yumeka body primitive has 79116 vertices"
    );

    for m in &asset.meshes {
        for p in &m.primitives {
            if let Some(ref anc) = p.skin_anchors {
                let bound = anc.iter().filter(|a| a.body_vertex_idx != u32::MAX).count();
                println!("Mesh '{}' prim {:?} verts={}: bound skin anchors = {} / {}", m.name, p.id, p.vertex_count, bound, anc.len());
            } else {
                println!("Mesh '{}' prim {:?} verts={}: NO skin anchors", m.name, p.id, p.vertex_count);
            }
        }
    }

    // Find skirt mesh (Circle.056)
    let skirt_mesh = asset
        .meshes
        .iter()
        .find(|m| m.name == "Circle.056")
        .expect("Yumeka has Circle.056 skirt mesh");
    let skirt_prim = &skirt_mesh.primitives[0];

    assert!(
        skirt_prim.skin_anchors.is_some(),
        "Skirt primitive must have generated skin anchors"
    );
    assert_eq!(
        skirt_prim.body_primitive_id,
        Some(body_pid),
        "Skirt primitive must point to body primitive"
    );

    let anchors = skirt_prim.skin_anchors.as_ref().unwrap();
    assert_eq!(anchors.len(), 2460);

    let mut bound_count = 0usize;
    for anc in anchors {
        if anc.body_vertex_idx != u32::MAX {
            bound_count += 1;
            assert!(
                anc.min_clearance >= 0.002,
                "Min clearance must be at least 2mm (0.002m), got {}",
                anc.min_clearance
            );
            assert!(
                (anc.body_vertex_idx as usize) < body_prim.vertex_count as usize,
                "Body vertex index must be within body primitive bounds"
            );
        }
    }

    println!(
        "Yumeka skirt skin anchors: {} / 2460 bound to body (body_pid={:?})",
        bound_count, body_pid
    );
    assert!(
        bound_count >= 2000,
        "Expected at least 2000 skirt vertices bound to body surface, got {}",
        bound_count
    );
}

#[test]
fn test_yumeka_anti_penetration_projection_on_leg_lift() {
    let fbx_path = "sample_data/YUMEKA_v1.0.1/FBX/Yumeka_v1.0.fbx";
    if !Path::new(fbx_path).exists() {
        return;
    }

    let loader = crate::asset::fbx::FbxAssetLoader::new();
    let asset = loader.load(fbx_path).expect("Failed to load Yumeka FBX");

    let body_pid = asset.body_primitive_id.expect("body_primitive_id must be identified");
    let body_prim = asset
        .meshes
        .iter()
        .flat_map(|m| &m.primitives)
        .find(|p| p.id == body_pid)
        .expect("body primitive must exist");

    let skirt_mesh = asset
        .meshes
        .iter()
        .find(|m| m.name == "Circle.056")
        .expect("Circle.056 skirt mesh must exist");
    let skirt_prim = &skirt_mesh.primitives[0];
    let anchors = skirt_prim.skin_anchors.as_ref().expect("Skirt must have skin anchors");

    // 1. Create avatar instance and pose: lift LeftUpperLeg forward by 45 degrees
    let mut avatar = crate::avatar::AvatarInstance::new(
        crate::avatar::AvatarInstanceId(1),
        std::sync::Arc::clone(&asset),
    );

    let l_leg_idx = asset
        .skeleton
        .nodes
        .iter()
        .position(|n| n.name == "UpperLeg_L")
        .expect("UpperLeg_L must exist");

    let angle_rad = 45.0f32.to_radians();
    let sin_half = (angle_rad * 0.5).sin();
    let cos_half = (angle_rad * 0.5).cos();
    let rot_x = [sin_half, 0.0, 0.0, cos_half];
    avatar.pose.local_transforms[l_leg_idx].rotation = crate::math_utils::quat_mul(
        &rot_x,
        &asset.skeleton.nodes[l_leg_idx].rest_local.rotation,
    );
    avatar.compute_global_pose();
    avatar.build_skinning_matrices();

    // 2. Compute skinned world vertices for both body and skirt under this lifted pose
    let body_world = crate::asset::clearance::compute_rest_world_vertices(body_prim, &avatar.pose.skinning_matrices);
    let skirt_world = crate::asset::clearance::compute_rest_world_vertices(skirt_prim, &avatar.pose.skinning_matrices);

    assert_eq!(body_world.len(), body_prim.vertex_count as usize);
    assert_eq!(skirt_world.len(), skirt_prim.vertex_count as usize);

    // 3. Simulate GPU Compute Shader transform_cs clearance projection logic
    let mut penetration_threat_count = 0usize;
    let mut max_push_distance = 0.0f32;

    for (vid, &(mut skirt_p, _skirt_n)) in skirt_world.iter().enumerate() {
        let anc = anchors[vid];
        if anc.body_vertex_idx != u32::MAX && anc.weight > 1e-4 {
            let (bp, bn) = body_world[anc.body_vertex_idx as usize];
            let nlen = (bn[0] * bn[0] + bn[1] * bn[1] + bn[2] * bn[2]).sqrt();
            if nlen > 1e-4 {
                let unit_bn = [bn[0] / nlen, bn[1] / nlen, bn[2] / nlen];
                let diff = [skirt_p[0] - bp[0], skirt_p[1] - bp[1], skirt_p[2] - bp[2]];
                let clearance = diff[0] * unit_bn[0] + diff[1] * unit_bn[1] + diff[2] * unit_bn[2];

                if clearance < anc.min_clearance {
                    penetration_threat_count += 1;
                    let push = (anc.min_clearance - clearance) * anc.weight;
                    max_push_distance = max_push_distance.max(push);
                    skirt_p[0] += unit_bn[0] * push;
                    skirt_p[1] += unit_bn[1] * push;
                    skirt_p[2] += unit_bn[2] * push;

                    // After projection, clearance must satisfy min_clearance
                    let new_diff = [skirt_p[0] - bp[0], skirt_p[1] - bp[1], skirt_p[2] - bp[2]];
                    let new_clearance = new_diff[0] * unit_bn[0] + new_diff[1] * unit_bn[1] + new_diff[2] * unit_bn[2];
                    assert!(
                        new_clearance >= anc.min_clearance - 1e-4,
                        "After projection, clearance ({}) must be >= min_clearance ({})",
                        new_clearance,
                        anc.min_clearance
                    );
                }
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

