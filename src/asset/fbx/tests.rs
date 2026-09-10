use std::path::Path;
use crate::asset::HumanoidBone;
use super::loader::FbxAssetLoader;

#[test]
fn test_load_yumeka_fbx() {
    let fbx_path = "sample_data/YUMEKA_v1.0.1/FBX/Yumeka_v1.0.fbx";
    if !Path::new(fbx_path).exists() {
        eprintln!("Skipping test_load_yumeka_fbx: sample file not found at {}", fbx_path);
        return;
    }

    let loader = FbxAssetLoader::new();

    // First load (parses FBX & saves cache)
    let asset = loader
        .load(fbx_path)
        .expect("Failed to load Yumeka FBX avatar");

    // 1. Verify skeleton & nodes
    assert!(!asset.skeleton.nodes.is_empty(), "Skeleton nodes must not be empty");
    assert!(!asset.skeleton.root_nodes.is_empty(), "Root nodes must not be empty");
    println!("Loaded {} skeleton nodes", asset.skeleton.nodes.len());

    // 2. Verify humanoid mapping
    let humanoid = asset.humanoid.as_ref().expect("Humanoid map must exist");
    assert!(
        humanoid.bone_map.contains_key(&HumanoidBone::Hips),
        "Hips bone must be mapped"
    );
    assert!(
        humanoid.bone_map.contains_key(&HumanoidBone::Head),
        "Head bone must be mapped"
    );
    println!("Mapped {} humanoid bones", humanoid.bone_map.len());

    // 3. Verify meshes and primitives
    assert!(!asset.meshes.is_empty(), "Meshes must not be empty");
    let total_prims: usize = asset.meshes.iter().map(|m| m.primitives.len()).sum();
    let total_verts: usize = asset
        .meshes
        .iter()
        .flat_map(|m| m.primitives.iter())
        .map(|p| p.vertex_count as usize)
        .sum();
    println!("Loaded {} meshes, {} primitives, {} total vertices", asset.meshes.len(), total_prims, total_verts);
    assert!(total_prims > 0, "Primitives count must be > 0");
    assert!(total_verts > 0, "Vertex count must be > 0");

    // 4. Verify materials & textures
    assert!(!asset.materials.is_empty(), "Materials must not be empty");
    let mats_with_textures = asset
        .materials
        .iter()
        .filter(|m| m.texture_bindings.base_color_texture.as_ref().is_some_and(|t| t.pixel_data.is_some()))
        .count();
    println!(
        "Loaded {} materials, {} with base color texture resolved & decoded",
        asset.materials.len(),
        mats_with_textures
    );
    assert!(
        mats_with_textures > 0,
        "At least one material must have resolved texture from Texture/PNG"
    );

    // 5. Verify expressions / blendshapes
    let exprs = &asset.default_expressions.expressions;
    println!("Loaded {} expressions / blendshapes", exprs.len());
    assert!(!exprs.is_empty(), "Expressions must not be empty");

    let has_aa = exprs.iter().any(|e| e.name == "aa");
    let has_blink = exprs.iter().any(|e| e.name == "blink");
    println!("Has 'aa' preset: {}, has 'blink' preset: {}", has_aa, has_blink);
    assert!(has_aa, "Viseme 'aa' must be mapped from vrc.v_aa");
    assert!(has_blink, "Blink must be mapped from eye_close / blink");

    // 6. Verify bounds (Y-up)
    assert!(!asset.root_aabb.is_empty(), "Root AABB must not be empty");
    let size = asset.root_aabb.size();
    println!("Avatar root_aabb min: {:?}, max: {:?}", asset.root_aabb.min, asset.root_aabb.max);
    println!("Avatar size: x={:.2}, y={:.2}, z={:.2}", size[0], size[1], size[2]);
    assert!(
        size[1] > 1.0 && size[1] < 1.8,
        "Model height along Y axis should be ~1.34m (got {:.2}m)",
        size[1]
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
    println!("Cache hit & texture rehydration test PASSED!");
}
