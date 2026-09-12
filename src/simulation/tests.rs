use super::*;
use crate::asset::{
    identity_matrix, AssetSourceHash, AvatarAssetId, ColliderAsset, ColliderId, ColliderRef,
    ColliderShape, ExpressionAssetSet, NodeId, SkeletonAsset, SkeletonNode, SpringBoneAsset,
    Transform,
};
use crate::avatar::{AvatarInstance, AvatarInstanceId};
use std::sync::Arc;

/// Regression test for Finding #6: a single frame must yield the same
/// `(fixed_dt, substeps)` for every avatar. After the first per-frame
/// `advance` call the accumulator is drained, so repeating the call with
/// `dt=0` must produce zero substeps. The previous code path called
/// `advance(frame_dt)` once per avatar, so the second avatar onwards saw
/// `substeps=0`.
#[test]
fn sim_clock_advance_does_not_regenerate_substeps_within_a_frame() {
    let mut clock = SimulationClock::new(1.0 / 60.0, 8);
    assert_eq!(clock.advance(1.0 / 60.0), 1);
    assert_eq!(clock.advance(0.0), 0);
    assert_eq!(clock.advance(0.0), 0);
}

fn make_two_joint_spring_avatar() -> AvatarInstance {
    // Vertical chain: bone axis parallel to gravity, so gravity has no
    // angular effect — right for toggle-gating tests that only need
    // "does the state advance at all".
    make_two_joint_spring_avatar_with_offset([0.0, 1.0, 0.0])
}

/// Two-joint spring chain with a configurable per-bone rest offset.
/// Pass a horizontal offset (e.g. `[1,0,0]`) when the test needs
/// gravity to actually swing the chain: with the default vertical
/// layout the pull is parallel to the bone and `enforce_bone_length`
/// cancels it exactly.
fn make_two_joint_spring_avatar_with_offset(offset: [f32; 3]) -> AvatarInstance {
    make_two_joint_spring_avatar_with(offset, [0.0, -1.0, 0.0])
}

fn make_two_joint_spring_avatar_with(offset: [f32; 3], gravity_dir: [f32; 3]) -> AvatarInstance {
    let nodes = vec![
        SkeletonNode {
            id: NodeId(0),
            name: "root".into(),
            parent: None,
            children: vec![NodeId(1)],
            rest_local: Transform::default(),
            humanoid_bone: None,
        },
        SkeletonNode {
            id: NodeId(1),
            name: "j_a".into(),
            parent: Some(NodeId(0)),
            children: vec![NodeId(2)],
            rest_local: Transform {
                translation: offset,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            },
            humanoid_bone: None,
        },
        SkeletonNode {
            id: NodeId(2),
            name: "j_b".into(),
            parent: Some(NodeId(1)),
            children: vec![],
            rest_local: Transform {
                translation: offset,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            },
            humanoid_bone: None,
        },
    ];
    let skeleton = SkeletonAsset {
        nodes,
        root_nodes: vec![NodeId(0)],
        inverse_bind_matrices: vec![identity_matrix(); 3],
    };
    let spring = SpringBoneAsset {
        chain_root: NodeId(0),
        joints: vec![NodeId(1), NodeId(2)],
        stiffness: 1.0,
        drag_force: 0.4,
        gravity_dir,
        gravity_power: 1.0,
        gravity_floor: 0.0,
        radius: 0.0,
        collider_refs: vec![],
        joint_stiffness: vec![],
        joint_drag: vec![],
        joint_gravity_power: vec![],
    };

    let asset = Arc::new(AvatarAsset {
        id: AvatarAssetId(0),
        source_path: std::path::PathBuf::from("test.vrm"),
        source_hash: AssetSourceHash([0u8; 32]),
        skeleton,
        meshes: vec![],
        materials: vec![],
        humanoid: None,
        spring_bones: vec![spring],
        colliders: vec![],
        default_expressions: ExpressionAssetSet {
            expressions: vec![],
        },
        animation_clips: vec![],
        node_to_mesh: Default::default(),
        vrm_meta: Default::default(),
        root_aabb: crate::asset::Aabb::empty(),
        body_primitive_id: None,
        loaded_from_cache: false,
    });

    let mut inst = AvatarInstance::new(AvatarInstanceId(1), asset);
    inst.build_base_pose();
    inst.compute_global_pose();
    inst
}

/// Generalized spring-avatar builder for collider / chain-consistency
/// tests. `joint_offsets[i]` is spring joint `i + 1`'s rest translation
/// (node 1 is the chain anchor — `joints[0]`, never solved; nodes 2..
/// are solved tail joints). `collider` places a single collider shape
/// at `offset` on the root node, referenced by the chain.
fn make_spring_avatar_with_colliders(
    joint_offsets: &[[f32; 3]],
    collider: Option<(ColliderShape, [f32; 3])>,
    gravity_power: f32,
    gravity_floor: f32,
) -> AvatarInstance {
    let n = joint_offsets.len();
    let mut nodes = vec![SkeletonNode {
        id: NodeId(0),
        name: "root".into(),
        parent: None,
        children: vec![NodeId(1)],
        rest_local: Transform::default(),
        humanoid_bone: None,
    }];
    for (i, off) in joint_offsets.iter().enumerate() {
        let id = (i + 1) as u64;
        nodes.push(SkeletonNode {
            id: NodeId(id),
            name: format!("j_{id}"),
            parent: Some(NodeId(id - 1)),
            children: if (id as usize) < n {
                vec![NodeId(id + 1)]
            } else {
                vec![]
            },
            rest_local: Transform {
                translation: *off,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            },
            humanoid_bone: None,
        });
    }
    let skeleton = SkeletonAsset {
        nodes,
        root_nodes: vec![NodeId(0)],
        inverse_bind_matrices: vec![identity_matrix(); n + 1],
    };
    let spring = SpringBoneAsset {
        chain_root: NodeId(1),
        joints: (1..=n as u64).map(NodeId).collect(),
        stiffness: 1.0,
        drag_force: 0.4,
        gravity_dir: [0.0, -1.0, 0.0],
        gravity_power,
        gravity_floor,
        radius: 0.02,
        collider_refs: collider
            .as_ref()
            .map(|_| vec![ColliderRef { id: ColliderId(1) }])
            .unwrap_or_default(),
        joint_stiffness: vec![],
        joint_drag: vec![],
        joint_gravity_power: vec![],
    };
    let colliders = collider
        .map(|(shape, offset)| {
            vec![ColliderAsset {
                id: ColliderId(1),
                node: NodeId(0),
                shape,
                offset,
            }]
        })
        .unwrap_or_default();

    let asset = Arc::new(AvatarAsset {
        id: AvatarAssetId(0),
        source_path: std::path::PathBuf::from("test.vrm"),
        source_hash: AssetSourceHash([0u8; 32]),
        skeleton,
        meshes: vec![],
        materials: vec![],
        humanoid: None,
        spring_bones: vec![spring],
        colliders,
        default_expressions: ExpressionAssetSet {
            expressions: vec![],
        },
        animation_clips: vec![],
        node_to_mesh: Default::default(),
        vrm_meta: Default::default(),
        root_aabb: crate::asset::Aabb::empty(),
        body_primitive_id: None,
        loaded_from_cache: false,
    });

    let mut inst = AvatarInstance::new(AvatarInstanceId(1), asset);
    inst.build_base_pose();
    inst.compute_global_pose();
    inst
}

/// Regression test for Finding #5: when `spring_enabled` is false the
/// Rapier-bearing `step_all` must not advance spring state. Before the
/// fix the Rapier branch ignored runtime toggles entirely.
#[test]
fn step_all_with_spring_disabled_does_not_mutate_spring_state() {
    let mut world = PhysicsWorld::new();
    let mut avatar = make_two_joint_spring_avatar();
    let sentinel = [9.0_f32, 9.0, 9.0];
    for state in &mut avatar.secondary_motion.spring_states {
        for p in &mut state.positions {
            *p = sentinel;
        }
        for p in &mut state.previous_positions {
            *p = sentinel;
        }
    }
    let before: Vec<Vec<[f32; 3]>> = avatar
        .secondary_motion
        .spring_states
        .iter()
        .map(|s| s.positions.clone())
        .collect();

    world.step_all(
        1.0 / 120.0,
        4,
        &mut avatar,
        SimulationStepOptions {
            spring_enabled: false,
            cloth_enabled: false,
        },
        &spring::SpringTuning::default(),
        &SceneGravity::default(),
    );

    let after: Vec<Vec<[f32; 3]>> = avatar
        .secondary_motion
        .spring_states
        .iter()
        .map(|s| s.positions.clone())
        .collect();
    assert_eq!(before, after);
}

/// Counterpart to the disabled-toggle test: makes sure the disabled case
/// is non-trivially gated, i.e. the same setup *does* mutate spring
/// positions when the toggle is on.
#[test]
fn step_all_with_spring_enabled_does_mutate_spring_state() {
    let mut world = PhysicsWorld::new();
    let mut avatar = make_two_joint_spring_avatar();
    let before: Vec<Vec<[f32; 3]>> = avatar
        .secondary_motion
        .spring_states
        .iter()
        .map(|s| s.positions.clone())
        .collect();

    world.step_all(
        1.0 / 120.0,
        4,
        &mut avatar,
        SimulationStepOptions {
            spring_enabled: true,
            cloth_enabled: false,
        },
        &spring::SpringTuning::default(),
        &SceneGravity::default(),
    );

    let after: Vec<Vec<[f32; 3]>> = avatar
        .secondary_motion
        .spring_states
        .iter()
        .map(|s| s.positions.clone())
        .collect();
    assert_ne!(before, after);
}

/// Runs the two-joint chain for `steps` ticks under `tuning` and
/// returns the final tip position.
fn settle_tip(tuning: &spring::SpringTuning, steps: u32) -> [f32; 3] {
    let mut world = PhysicsWorld::new();
    // Horizontal chain — see `make_two_joint_spring_avatar_with_offset`:
    // gravity must be perpendicular to the bone to produce droop.
    let mut avatar = make_two_joint_spring_avatar_with_offset([1.0, 0.0, 0.0]);
    for _ in 0..steps {
        world.step_springs(1.0 / 60.0, 1, &mut avatar, tuning, &SceneGravity::default());
        avatar.compute_global_pose();
    }
    *avatar.secondary_motion.spring_states[0]
        .positions
        .last()
        .unwrap()
}

/// `gravity_offset` is additive on the authored per-joint power: a
/// negative offset large enough to cancel the asset's `1.0` must
/// leave the tip hanging higher (less droop) than the authored run.
#[test]
fn spring_tuning_gravity_offset_changes_droop() {
    let authored = settle_tip(&spring::SpringTuning::default(), 120);
    let no_gravity = settle_tip(
        &spring::SpringTuning {
            gravity_offset: -1.0,
            ..Default::default()
        },
        120,
    );
    assert!(
        no_gravity[1] > authored[1] + 1e-4,
        "cancelling gravity should reduce droop: authored y={} no-gravity y={}",
        authored[1],
        no_gravity[1]
    );
}

/// Low sway stiffens the chain: under identical gravity the rigid
/// setting must stay closer to the rest pose than the loose one.
#[test]
fn spring_tuning_sway_scale_controls_stiffness() {
    let mut avatar = make_two_joint_spring_avatar_with_offset([1.0, 0.0, 0.0]);
    avatar.compute_global_pose();
    let rest_tip = {
        let node = avatar.asset.spring_bones[0].joints[1].0 as usize;
        crate::math_utils::mat4_translation(&avatar.pose.global_transforms[node])
    };
    let dist = |a: &[f32; 3], b: &[f32; 3]| {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let rigid = settle_tip(
        &spring::SpringTuning {
            sway_scale: 0.05,
            ..Default::default()
        },
        120,
    );
    let loose = settle_tip(
        &spring::SpringTuning {
            sway_scale: 2.0,
            ..Default::default()
        },
        120,
    );
    assert!(
        dist(&rigid, &rest_tip) < dist(&loose, &rest_tip),
        "rigid sway must hold the tip nearer rest: rigid={:?} loose={:?} rest={:?}",
        rigid,
        loose,
        rest_tip
    );
}

// -- Scene gravity ----------------------------------------------------

fn approx(a: [f32; 3], b: [f32; 3]) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() < 1e-4)
}

/// Default scene gravity (down, strength 1) must reproduce the old
/// hardcoded Rapier / cloth vector exactly — no behaviour change for
/// existing projects.
#[test]
fn scene_gravity_default_is_earth_down() {
    let g = SceneGravity::default();
    assert!(approx(g.world_accel(), [0.0, -9.81, 0.0]));
    // identity avatar rotation => local == world
    assert!(approx(
        g.local_accel(&[0.0, 0.0, 0.0, 1.0]),
        [0.0, -9.81, 0.0]
    ));
    assert!(approx(g.local_dir(&[0.0, 0.0, 0.0, 1.0]), [0.0, -1.0, 0.0]));
}

/// Strength is a linear multiplier; 0 is weightless, 2 doubles.
#[test]
fn scene_gravity_strength_scales_linearly() {
    let weightless = SceneGravity {
        strength: 0.0,
        ..Default::default()
    };
    assert!(approx(weightless.world_accel(), [0.0, 0.0, 0.0]));
    assert_eq!(weightless.spring_power_scale(), 0.0);
    let heavy = SceneGravity {
        strength: 2.0,
        ..Default::default()
    };
    assert!(approx(heavy.world_accel(), [0.0, -19.62, 0.0]));
}

/// World-space direction is inverse-rotated into the avatar's local
/// frame: an avatar yawed 90° about Y should see world-down stay down
/// in local Y (rotation about the gravity axis leaves it unchanged),
/// while a 90° roll about Z maps world-down onto local ±X.
#[test]
fn scene_gravity_direction_is_world_space() {
    let g = SceneGravity::default();
    // 90° about Y (quat = [0, sin45, 0, cos45]): down stays down.
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let yaw90 = [0.0, s, 0.0, s];
    assert!(approx(g.local_accel(&yaw90), [0.0, -9.81, 0.0]));
    // 90° about Z: world down rotates into local +X (magnitude kept).
    let roll90 = [0.0, 0.0, s, s];
    let la = g.local_accel(&roll90);
    assert!(
        (la[0].abs() - 9.81).abs() < 1e-3,
        "expected ~9.81 on X, got {la:?}"
    );
    assert!(
        la[1].abs() < 1e-3 && la[2].abs() < 1e-3,
        "off-axis leak: {la:?}"
    );
}

/// Regression: a spring chain whose authored `gravity_dir` is not
/// straight down must keep pulling along its authored direction under
/// default scene gravity — the scene direction *reorients* (delta from
/// down), it does not overwrite. A vertical chain with authored
/// gravity_dir = +X should swing sideways (tip.x grows); if the code
/// forced the scene down axis instead, gravity would lie along the
/// bone and `enforce_bone_length` would cancel it (tip.x ~ 0).
#[test]
fn spring_preserves_authored_gravity_dir_at_default_scene() {
    let mut world = PhysicsWorld::new();
    // Vertical bones (offset +Y), authored gravity swept to +X.
    let mut avatar = make_two_joint_spring_avatar_with([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]);
    avatar.compute_global_pose();
    let soft = spring::SpringTuning {
        sway_scale: 2.0,
        ..Default::default()
    };
    let strong = SceneGravity {
        direction: [0.0, -1.0, 0.0],
        strength: 3.0,
    };
    for _ in 0..240 {
        world.step_springs(1.0 / 60.0, 1, &mut avatar, &soft, &strong);
        avatar.compute_global_pose();
    }
    let tip = *avatar.secondary_motion.spring_states[0]
        .positions
        .last()
        .unwrap();
    // Layered (correct): authored +X preserved → tip swings to
    // x ≈ 0.10. Override (the bug): gravity forced to scene-down lies
    // along the bone → enforce_bone_length cancels it → x ≈ 0. The
    // 0.03 threshold sits far from both.
    assert!(
        tip[0] > 0.03,
        "authored +X gravity must swing the tip sideways, got tip = {tip:?}"
    );
}

/// `apply_cloth_gravity` with default scene gravity reproduces the old
/// `[0, -9.81 * scale, 0]` per-cloth formula.
#[test]
fn apply_cloth_gravity_default_matches_legacy() {
    let mut avatar = make_two_joint_spring_avatar();
    avatar.cloth_sim = Some(cloth::ClothSimState {
        gravity_scale: 1.5,
        ..Default::default()
    });
    apply_cloth_gravity(&mut avatar, &SceneGravity::default());
    let g = avatar.cloth_sim.as_ref().unwrap().gravity;
    assert!(approx(g, [0.0, -9.81 * 1.5, 0.0]), "got {g:?}");
}

/// Regression: a frame where the simulation clock yields zero substeps
/// skips the spring solver, but `build_base_pose` has already reset
/// every local transform to rest. Before the solved-rotation store the
/// spring joints rendered that frame in their rest pose — the
/// single-frame hair-clip symptom. `reapply_spring_rotations` must put
/// the last solved rotation back exactly.
#[test]
fn zero_substep_frame_preserves_last_solved_spring_rotation() {
    let quat_dist = |a: [f32; 4], b: [f32; 4]| {
        (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>()).sqrt()
    };
    let mut world = PhysicsWorld::new();
    // Horizontal chain so gravity swings it away from rest; loose sway
    // so the settled rotation clearly departs from the rest rotation.
    let mut avatar = make_two_joint_spring_avatar_with_offset([1.0, 0.0, 0.0]);
    let soft = spring::SpringTuning {
        sway_scale: 2.0,
        ..Default::default()
    };
    for _ in 0..120 {
        world.step_springs(1.0 / 60.0, 1, &mut avatar, &soft, &SceneGravity::default());
        avatar.compute_global_pose();
    }
    let written_node = avatar.asset.spring_bones[0].joints[0].0 as usize;
    let solved = avatar.pose.local_transforms[written_node].rotation;
    let rest = avatar.asset.skeleton.nodes[written_node]
        .rest_local
        .rotation;
    assert!(
        quat_dist(solved, rest) > 1e-3,
        "test premise: the swing must move the joint away from rest \
             (solved={solved:?} rest={rest:?})"
    );

    // Zero-substep frame: base-pose reset, solver skipped, restore runs.
    avatar.build_base_pose();
    assert!(
        quat_dist(avatar.pose.local_transforms[written_node].rotation, rest) < 1e-6,
        "build_base_pose resets the spring joint to rest (bug entry point)"
    );
    avatar.reapply_spring_rotations();
    assert!(
        quat_dist(avatar.pose.local_transforms[written_node].rotation, solved) < 1e-6,
        "reapply must restore the last solved rotation exactly"
    );
}

/// The solved-rotation store starts at the rest rotations, so
/// re-applying before the solver has ever run must be a no-op rather
/// than a garbage write.
#[test]
fn reapply_spring_rotations_before_first_solve_is_noop() {
    let mut avatar = make_two_joint_spring_avatar();
    avatar.build_base_pose();
    avatar.reapply_spring_rotations();
    for (local, node) in avatar
        .pose
        .local_transforms
        .iter()
        .zip(avatar.asset.skeleton.nodes.iter())
    {
        assert_eq!(local.rotation, node.rest_local.rotation);
    }
}

/// Resting contact on a collider must not inject outward velocity.
/// Verlet stores velocity implicitly as (current - previous); a
/// collider projection that moves only `current` converts that
/// step's penetration depth into an outward kick next step — the
/// "hair hits the shoulder and bounces off" symptom. Measures the
/// solved joint's outward excursion beyond the contact radius after
/// first touch: a bounce flies clear of the surface, a damped
/// contact stays on it.
#[test]
fn collider_contact_does_not_bounce() {
    let mut world = PhysicsWorld::new();
    // Anchor at (1,0,0), tip rest at (2,0,0); sphere centred directly
    // under the tip so gravity drops the tip onto its top.
    let mut avatar = make_spring_avatar_with_colliders(
        &[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        Some((ColliderShape::Sphere { radius: 0.15 }, [2.0, -0.15, 0.0])),
        1.0,
        0.0,
    );
    let bone_radius = avatar.asset.spring_bones[0].radius;
    let total_radius = 0.15 + bone_radius;
    let center = [2.0_f32, -0.15, 0.0];
    let mut contacted = false;
    let mut max_excursion = 0.0_f32;
    let mut settled_dist = 0.0_f32;
    for _ in 0..240 {
        world.step_springs(
            1.0 / 60.0,
            1,
            &mut avatar,
            &spring::SpringTuning::default(),
            &SceneGravity::default(),
        );
        avatar.compute_global_pose();
        let tip = avatar.secondary_motion.spring_states[0].positions[1];
        let d = ((tip[0] - center[0]).powi(2)
            + (tip[1] - center[1]).powi(2)
            + (tip[2] - center[2]).powi(2))
        .sqrt();
        if !contacted && d <= total_radius * 1.001 {
            // The solver projects the joint exactly onto the contact
            // boundary, so "in contact" means at-or-inside by a hair.
            contacted = true;
        }
        if contacted {
            max_excursion = max_excursion.max(d - total_radius);
            settled_dist = d;
        }
    }
    assert!(contacted, "test premise: the chain must reach the collider");
    assert!(
            max_excursion < 0.01,
            "contact must damp out; the joint instead bounced {max_excursion:.4} m clear of the surface (settled at {settled_dist:.4}, radius {total_radius:.4})"
        );
}

/// Every solved spring joint's stored world position must match the
/// rendered global transform after `compute_global_pose`. Anchoring
/// later joints to the previous joint's *rest* global transform (the
/// pose globals are only recomputed after the whole solver) makes the
/// rendered chain diverge from the solved chain exactly when the
/// chain is bent — e.g. hair draped over a shoulder, where contact
/// corrections then flicker.
#[test]
fn solved_positions_match_rendered_globals_when_bent() {
    let mut world = PhysicsWorld::new();
    let mut avatar = make_spring_avatar_with_colliders(
        &[[0.35, 0.0, 0.0], [0.35, 0.0, 0.0], [0.35, 0.0, 0.0]],
        None,
        1.0,
        0.0,
    );
    // Loose sway + gravity: the chain bends well away from rest.
    let soft = spring::SpringTuning {
        sway_scale: 2.0,
        ..Default::default()
    };
    for _ in 0..240 {
        world.step_springs(1.0 / 60.0, 1, &mut avatar, &soft, &SceneGravity::default());
    }
    avatar.compute_global_pose();
    let joints = &avatar.asset.spring_bones[0].joints;
    for j in 1..joints.len() {
        let solved = avatar.secondary_motion.spring_states[0].positions[j];
        let rendered = crate::math_utils::mat4_translation(
            &avatar.pose.global_transforms[joints[j].0 as usize],
        );
        let d = ((solved[0] - rendered[0]).powi(2)
            + (solved[1] - rendered[1]).powi(2)
            + (solved[2] - rendered[2]).powi(2))
        .sqrt();
        assert!(
            d < 2e-3,
            "joint {j}: solved {solved:?} vs rendered {rendered:?} diverged by {d:.4} m"
        );
    }
}

// -- Natural gravity force model ----------------------------------------

/// Run a small horizontal chain (5 cm bones, avatar-realistic scale)
/// until it settles and return the tip's vertical drop below its rest
/// position. Used by the natural-gravity tests below.
fn settle_tip_drop(tuning: &spring::SpringTuning, power: f32, floor: f32) -> f32 {
    let mut world = PhysicsWorld::new();
    let offsets = [[0.05, 0.0, 0.0], [0.05, 0.0, 0.0], [0.05, 0.0, 0.0]];
    let mut avatar = make_spring_avatar_with_colliders(&offsets, None, power, floor);
    let rest_tip_y = crate::math_utils::mat4_translation(
        &avatar.pose.global_transforms[avatar.asset.spring_bones[0].joints[2].0 as usize],
    )[1];
    for _ in 0..600 {
        world.step_springs(1.0 / 60.0, 1, &mut avatar, tuning, &SceneGravity::default());
    }
    let tip_y = avatar.secondary_motion.spring_states[0].positions[2];
    rest_tip_y - tip_y[1]
}

/// The natural-gravity floor is what makes authored-zero hair strands
/// (Yumeka's bangs, sides, twintales) re-hang under head motion: without
/// it a `gravity_power == 0` chain has no world-down force at all, and a
/// model author's zero leaves the strand gravity-less forever.
#[test]
fn natural_gravity_floor_droops_zero_power_chain() {
    let tuning = spring::SpringTuning::default();
    // Authored zero, floor on: the strand must visibly sag.
    let dropped = settle_tip_drop(&tuning, 0.0, 0.15);
    assert!(
        dropped > 0.003,
        "floor must produce visible droop, got {dropped:.4} m"
    );
    // Authored zero, floor zero: nothing pulls the strand down.
    let floating = settle_tip_drop(&tuning, 0.0, 0.0);
    assert!(
        floating.abs() < 0.0005,
        "zero power + zero floor must stay at rest, got {floating:.4} m"
    );
    // The floor is defeated by the toggle: legacy mode reproduces the
    // authored-faithful zero-gravity behaviour.
    let legacy = settle_tip_drop(
        &spring::SpringTuning {
            natural_gravity: false,
            ..Default::default()
        },
        0.0,
        0.15,
    );
    assert!(
        legacy.abs() < 0.0005,
        "legacy mode must ignore the floor, got {legacy:.4} m"
    );
}

/// The natural model must converge to the closed-form equilibrium sag
/// `gravity_step / stiffness_factor` per joint (drag only affects the
/// transient). This pins the force model's math, not just its direction.
#[test]
fn natural_gravity_settles_at_closed_form_sag() {
    let dt = 1.0f32 / 60.0;
    let tuning = spring::SpringTuning::default();
    // Stiffness 1.0 (builder) -> rate 7/s; authored fraction 1.0 -> g.
    let factor = -((-(7.0f32 * dt)).exp_m1());
    let per_joint = 9.81 * dt * dt * factor.recip();
    // Two solved joints sag ~2x the per-joint equilibrium (each target
    // follows the previous joint's sag); allow generous 35% tolerance
    // for the constraint projection and drag interplay.
    let drop = settle_tip_drop(&tuning, 1.0, 0.0);
    let expected = 2.0 * per_joint;
    assert!(
        ((drop - expected) / expected).abs() < 0.35,
        "settled drop {drop:.4} m vs closed form {expected:.4} m (per joint {per_joint:.4})"
    );
}

/// The gravity_offset slider stays meaningful in the natural model: a
/// full negative offset drives the fraction to the clamp at zero (the
/// floor wins — disabling gravity entirely is the toggle's job, not the
/// trim slider's), and a positive offset deepens the sag.
#[test]
fn natural_gravity_offset_modulates_sag() {
    let minus = settle_tip_drop(
        &spring::SpringTuning {
            gravity_offset: -1.0,
            ..Default::default()
        },
        0.6,
        0.0,
    );
    let plain = settle_tip_drop(&spring::SpringTuning::default(), 0.6, 0.0);
    assert!(
        plain > minus + 0.001,
        "positive-authorised power must sag more than offset-cancelled: {plain:.4} vs {minus:.4}"
    );
}
