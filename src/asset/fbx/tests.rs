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





