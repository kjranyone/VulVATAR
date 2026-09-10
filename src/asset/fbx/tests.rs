use std::path::Path;

#[test]
fn test_load_yumeka_fbx() {
    let fbx_path = "sample_data/YUMEKA_v1.0.1/FBX/Yumeka_v1.0.fbx";
    if !Path::new(fbx_path).exists() {
        return;
    }

    let opts = ufbx::LoadOpts::default();
    let scene = ufbx::load_file(fbx_path, opts).unwrap();

    // Check mesh instances / nodes
    println!("Node 1 (Armature): name='{}' rot={:?} t={:?}", scene.nodes[1].element.name, scene.nodes[1].local_transform.rotation, scene.nodes[1].local_transform.translation);
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
        assert_eq!(mat.alpha_mode, crate::asset::AlphaMode::Mask(0.5));
    }

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
    assert!(
        mats_with_textures > 0,
        "At least one material must have resolved texture from Texture/PNG"
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
}
