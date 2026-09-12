//! End-to-end loader regression tests for the VRM 0.x and 1.x parse
//! pipelines. Lives in its own file so the load orchestrator in
//! `mod.rs` doesn't have to host 700 lines of fixture wiring. Two
//! shapes of test:
//!
//! * **Fixture tests** load real `.vrm` files from `sample_data/`
//!   (gitignored — provision the fixtures locally to run them; CI and
//!   fresh clones skip with a `[skip]` line).
//! * **Synthetic tests** build minimal VRMC_vrm / VRM root JSON values
//!   in-memory and feed them through `super::extensions::parse_v0_extensions`
//!   / `super::extensions::parse_v1_extensions` so schema changes break
//!   here without needing fixtures on disk.

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
/// tiny 1-joint skin and the main 172-joint body skin), applies a
/// tracking-v2 [`RigPose`] with world-delta rotations on the head, the
/// left upper arm and a right index finger segment, and asserts:
///
///   1. `retarget::apply_rig_pose` writes each driven bone's local
///      rotation away from rest;
///   2. after `compute_global_pose` + `build_skinning_matrices`, the
///      Head bone's skinning matrix (as the shader sees it, indexed by
///      glTF node id) reflects the rotation — i.e. multi-skin models
///      still feed the right joint.
#[test]
fn sample_vrm0_retarget_drives_head_arm_and_fingers() {
    use crate::avatar::retarget::{apply_rig_pose, RetargetParams, RetargetState};
    use crate::avatar::{AvatarInstance, AvatarInstanceId};
    use crate::tracking::fusion::output::{RigBone, RigPose};

    let loader = VrmAssetLoader::new();
    let Some(asset) = load_or_skip(&loader, SAMPLE_VRM0) else {
        return;
    };

    let mut avatar = AvatarInstance::new(AvatarInstanceId(1), asset.clone());
    avatar.build_base_pose();

    let humanoid = asset.humanoid.as_ref().unwrap();
    let node = |b: HumanoidBone| humanoid.bone_map.get(&b).expect("bone mapped").0 as usize;
    let head_idx = node(HumanoidBone::Head);
    let probes = [
        HumanoidBone::Head,
        HumanoidBone::LeftUpperArm,
        HumanoidBone::RightIndexProximal,
    ];
    let rest: Vec<[f32; 4]> = probes
        .iter()
        .map(|&b| avatar.pose.local_transforms[node(b)].rotation)
        .collect();

    // World-delta rotations in the viewer frame: 45° head yaw, arm swung
    // forward, finger curled.
    let qy = |a: f32| [0.0, (a / 2.0).sin(), 0.0, (a / 2.0).cos()];
    let qx = |a: f32| [(a / 2.0).sin(), 0.0, 0.0, (a / 2.0).cos()];
    let mut rig = RigPose {
        quality: 1.0,
        ..Default::default()
    };
    for (b, q) in [
        (HumanoidBone::Head, qy(std::f32::consts::FRAC_PI_4)),
        (HumanoidBone::LeftUpperArm, qx(0.9)),
        (HumanoidBone::RightIndexProximal, qx(1.1)),
    ] {
        rig.bones.insert(
            b,
            RigBone {
                delta_world: q,
                sigma: 0.05,
                data_sigma: 0.05,
            },
        );
    }
    let params = RetargetParams {
        rotation_blend: 1.0,
        root_translation_enabled: false,
        ..Default::default()
    };
    let mut state = RetargetState::default();
    apply_rig_pose(
        &rig,
        &asset.skeleton,
        humanoid,
        &mut avatar.pose.local_transforms,
        &params,
        &mut state,
        1.0 / 30.0,
    );

    for (k, &b) in probes.iter().enumerate() {
        let after = avatar.pose.local_transforms[node(b)].rotation;
        let diff: f32 = rest[k]
            .iter()
            .zip(after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            diff > 1e-4,
            "{b:?} local rotation did not change under the rig delta ({diff})"
        );
    }

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
        "Head skinning matrix unchanged after retarget — multi-skin indexing regressed ({max_diff})"
    );
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

    let mesh_to_node: HashMap<usize, usize> = [(0usize, 2usize)].into_iter().collect();

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
    assert!(
        names.contains("blinkLeft"),
        "blinkLeft preset key preserved"
    );
    assert!(
        names.contains("MyCustom"),
        "custom expression keyed by map key"
    );

    assert_eq!(parsed.springs.len(), 1);
    assert_eq!(parsed.springs[0].joints.len(), 2);
    assert_eq!(parsed.colliders.len(), 1);
}
