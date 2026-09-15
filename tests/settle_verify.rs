//! TEMPORARY verification harness for the settle-sleep campaign
//! (2026-09-15). Runs the same scenarios as the in-crate tests in
//! `simulation/tests.rs` / `cloth_solver/tests.rs` through the public
//! API, because the in-crate test build is currently blocked by
//! unrelated in-progress work in `tracking/fusion`. Delete once
//! `cargo test --lib` compiles again.

use std::collections::HashSet;
use std::sync::Arc;
use vulvatar_lib::asset::{
    identity_matrix, AssetSourceHash, AvatarAsset, AvatarAssetId, ColliderShape,
    ColliderShape as Shape, ExpressionAssetSet, NodeId, SceneColliderAsset, SceneColliderId,
    SkeletonAsset, SkeletonNode, SpringBoneAsset, Transform, Aabb,
};
use vulvatar_lib::avatar::{AvatarInstance, AvatarInstanceId};
use vulvatar_lib::math_utils::{vec3_length, vec3_sub};
use vulvatar_lib::simulation::cloth::{
    ClothParticle, ClothSimState, ResolvedCollider, SpatialHashGrid,
};
use vulvatar_lib::simulation::cloth_gpu_boundary::ClothSolverBackend;
use vulvatar_lib::simulation::cloth_solver;
use vulvatar_lib::simulation::sdf::{SdfField, SdfGrid, SENTINEL};
use vulvatar_lib::simulation::spring::{self, SpringTuning};
use vulvatar_lib::simulation::{PhysicsWorld, SceneGravity};

// ---------------------------------------------------------------------
// Fixtures (public-API twins of the in-crate builders)
// ---------------------------------------------------------------------

fn spring_asset(chain_root: NodeId, joints: Vec<NodeId>, gravity_dir: [f32; 3]) -> SpringBoneAsset {
    SpringBoneAsset {
        chain_root,
        joints,
        stiffness: 1.0,
        drag_force: 0.4,
        gravity_dir,
        gravity_power: 1.0,
        gravity_floor: 0.0,
        radius: 0.0,
        collider_refs: vec![],
        body_collision: false,
        joint_stiffness: vec![],
        joint_drag: vec![],
        joint_gravity_power: vec![],
    }
}

fn two_joint_avatar(offset: [f32; 3], gravity_dir: [f32; 3], body_collision: bool) -> AvatarInstance {
    let mut nodes = vec![SkeletonNode {
        id: NodeId(0),
        name: "root".into(),
        parent: None,
        children: vec![NodeId(1)],
        rest_local: Transform::default(),
        humanoid_bone: None,
    }];
    for id in 1..=2u64 {
        nodes.push(SkeletonNode {
            id: NodeId(id),
            name: format!("j_{id}"),
            parent: Some(NodeId(id - 1)),
            children: if id < 2 { vec![NodeId(id + 1)] } else { vec![] },
            rest_local: Transform {
                translation: offset,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            },
            humanoid_bone: None,
        });
    }
    let mut asset_spring = spring_asset(NodeId(0), vec![NodeId(1), NodeId(2)], gravity_dir);
    asset_spring.body_collision = body_collision;
    if body_collision {
        asset_spring.radius = 0.02;
    }
    let asset = Arc::new(AvatarAsset {
        id: AvatarAssetId(0),
        source_path: std::path::PathBuf::from("test.vrm"),
        source_hash: AssetSourceHash([0u8; 32]),
        skeleton: SkeletonAsset {
            nodes,
            root_nodes: vec![NodeId(0)],
            inverse_bind_matrices: vec![identity_matrix(); 3],
        },
        meshes: vec![],
        materials: vec![],
        humanoid: None,
        spring_bones: vec![asset_spring],
        colliders: vec![],
        default_expressions: ExpressionAssetSet { expressions: vec![] },
        animation_clips: vec![],
        node_to_mesh: Default::default(),
        vrm_meta: Default::default(),
        root_aabb: Aabb::empty(),
        body_primitive_id: None,
        loaded_from_cache: false,
    });
    let mut inst = AvatarInstance::new(AvatarInstanceId(1), asset);
    inst.build_base_pose();
    inst.compute_global_pose();
    inst
}

fn test_sdf(surface: bool) -> SdfField {
    // Covers the WHOLE solved chain (the solved joint sits at x≈2.0,
    // y≈-0.03) — samples outside the grid are SENTINEL everywhere, so
    // a grid that misses the joints looks "unchanged" even with a
    // surface planted. The zero plane sits one cell BELOW z=0: a joint
    // near z=0 samples d≈0.01 with a clean +z gradient (a slab centred
    // exactly on the joint is a symmetric pit — zero gradient, the
    // ambiguous-minimum case `resolve` deliberately skips).
    let grid = SdfGrid {
        origin: [-0.05, -0.3, -0.08],
        voxel: 0.01,
        dims: [215, 48, 16],
    };
    let mut data = vec![SENTINEL; grid.cell_count()];
    if surface {
        for x in 5..210 {
            for y in 8..44 {
                data[grid.index(x, y, 6)] = 0.01;
                data[grid.index(x, y, 7)] = 0.0;
                data[grid.index(x, y, 8)] = 0.01;
            }
        }
    }
    SdfField::new(grid, Arc::new(data))
}
fn sleeping(avatar: &AvatarInstance) -> bool {
    avatar.secondary_motion.spring_states[0].sleeping
}

fn tip(avatar: &AvatarInstance) -> [f32; 3] {
    *avatar.secondary_motion.spring_states[0]
        .positions
        .last()
        .unwrap()
}

fn settle_steps(world: &mut PhysicsWorld, avatar: &mut AvatarInstance, sdf: Option<&SdfField>) {
    let tuning = SpringTuning::default();
    let gravity = SceneGravity::default();
    for _ in 0..60 {
        world.step_springs(1.0 / 60.0, 1, avatar, &tuning, &gravity, sdf);
    }
}

// ---------------------------------------------------------------------
// Spring state machine
// ---------------------------------------------------------------------

#[test]
fn spring_settled_chain_sleeps_and_output_is_bit_stable() {
    let mut world = PhysicsWorld::new();
    let mut avatar = two_joint_avatar([1.0, 0.0, 0.0], [0.0, -1.0, 0.0], false);
    settle_steps(&mut world, &mut avatar, None);
    assert!(sleeping(&avatar), "chain must sleep once settled");
    let frozen = avatar.secondary_motion.spring_states[0].positions.clone();
    let tuning = SpringTuning::default();
    let gravity = SceneGravity::default();
    for _ in 0..5 {
        world.step_springs(1.0 / 60.0, 1, &mut avatar, &tuning, &gravity, None);
    }
    assert_eq!(
        avatar.secondary_motion.spring_states[0].positions, frozen,
        "sleeping chain must reproduce its vertex stream bit-for-bit"
    );
}

/// Latch regression: the first version latched `sleeping` forever, so a
/// driver that stopped bit-stable mid-swing froze the chain in place.
#[test]
fn spring_sleep_resettles_after_driver_stops_instead_of_freezing() {
    let mut world = PhysicsWorld::new();
    let mut avatar = two_joint_avatar([1.0, 0.0, 0.0], [0.0, -1.0, 0.0], false);
    let tuning = SpringTuning::default();
    let gravity = SceneGravity::default();
    settle_steps(&mut world, &mut avatar, None);
    assert!(sleeping(&avatar));

    for k in 0..8 {
        let off = if k % 2 == 0 { 0.08 } else { -0.08 };
        avatar.pose.local_transforms[0].translation = [off, 0.0, 0.0];
        avatar.compute_global_pose();
        world.step_springs(1.0 / 60.0, 1, &mut avatar, &tuning, &gravity, None);
    }
    assert!(!sleeping(&avatar), "a driven chain must be awake");

    let mut moving_steps = 0;
    let mut prev = avatar.secondary_motion.spring_states[0].positions.clone();
    for _ in 0..120 {
        world.step_springs(1.0 / 60.0, 1, &mut avatar, &tuning, &gravity, None);
        let cur = &avatar.secondary_motion.spring_states[0].positions;
        let moved = cur
            .iter()
            .zip(prev.iter())
            .any(|(a, b)| vec3_length(&vec3_sub(a, b)) > 1e-4);
        if moved {
            moving_steps += 1;
        }
        prev = cur.clone();
    }
    assert!(
        moving_steps >= 3,
        "chain must keep settling after the driver stops (moved in \
         {moving_steps} steps), not freeze mid-swing"
    );
    assert!(sleeping(&avatar), "chain must re-sleep once settled");
}

/// Key-store regression: side-authored gravity chains never matched the
/// key (stored the reoriented per-chain vector, compared the scene
/// down vector) and never slept.
#[test]
fn spring_sleep_key_matches_for_side_authored_gravity() {
    let mut world = PhysicsWorld::new();
    let mut avatar = two_joint_avatar([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], false);
    settle_steps(&mut world, &mut avatar, None);
    assert!(
        sleeping(&avatar),
        "side-gravity chain must sleep once quiet (key must compare like with like)"
    );
}

#[test]
fn spring_sleep_wakes_when_body_sdf_intrudes() {
    let mut world = PhysicsWorld::new();
    let mut avatar = two_joint_avatar([1.0, 0.0, 0.0], [0.0, -1.0, 0.0], true);
    let tuning = SpringTuning::default();
    let gravity = SceneGravity::default();
    let far = test_sdf(false);
    settle_steps(&mut world, &mut avatar, Some(&far));
    assert!(sleeping(&avatar));
    let frozen = avatar.secondary_motion.spring_states[0].positions.clone();

    let near = test_sdf(true);
    let mut woke = false;
    for _ in 0..10 {
        world.step_springs(1.0 / 60.0, 1, &mut avatar, &tuning, &gravity, Some(&near));
        if avatar.secondary_motion.spring_states[0].positions != frozen {
            woke = true;
        }
    }
    assert!(woke, "SDF intrusion must wake the chain and move joints");
    assert!(!sleeping(&avatar));
    // Surface withdrawn: the chain re-settles to its free hang and
    // re-sleeps. (Resting in sustained contact under full gravity moves
    // ~g*dt^2 per step — legitimately awake, the same class as the
    // offline tail's sustained swing.)
    let mut reslept = false;
    for _ in 0..240 {
        world.step_springs(1.0 / 60.0, 1, &mut avatar, &tuning, &gravity, Some(&far));
        if sleeping(&avatar) {
            reslept = true;
            break;
        }
    }
    assert!(reslept, "chain must re-sleep once the intrusion is gone");
}

#[test]
fn spring_sleep_wakes_when_scene_collider_appears() {
    let mut world = PhysicsWorld::new();
    let mut avatar = two_joint_avatar([1.0, 0.0, 0.0], [0.0, -1.0, 0.0], false);
    settle_steps(&mut world, &mut avatar, None);
    assert!(sleeping(&avatar));
    let frozen = tip(&avatar);

    world.add_scene_collider(SceneColliderAsset {
        id: SceneColliderId(1),
        position: [2.0, 0.0, 0.0],
        shape: Shape::Sphere { radius: 0.06 },
    });
    let tuning = SpringTuning::default();
    let gravity = SceneGravity::default();
    world.step_springs(1.0 / 60.0, 1, &mut avatar, &tuning, &gravity, None);
    assert_ne!(
        tip(&avatar), frozen,
        "new scene collider must wake the chain and push the tip"
    );
    assert!(!sleeping(&avatar));
}

// ---------------------------------------------------------------------
// CPU cloth gate
// ---------------------------------------------------------------------

fn cloth_avatar(gravity: [f32; 3]) -> AvatarInstance {
    let asset = Arc::new(AvatarAsset {
        id: AvatarAssetId(0),
        source_path: std::path::PathBuf::new(),
        source_hash: AssetSourceHash([0u8; 32]),
        skeleton: SkeletonAsset {
            nodes: Vec::new(),
            root_nodes: Vec::new(),
            inverse_bind_matrices: Vec::new(),
        },
        meshes: vec![],
        materials: vec![],
        humanoid: None,
        spring_bones: vec![],
        colliders: vec![],
        default_expressions: ExpressionAssetSet { expressions: vec![] },
        animation_clips: vec![],
        node_to_mesh: Default::default(),
        vrm_meta: Default::default(),
        root_aabb: Aabb::empty(),
        body_primitive_id: None,
        loaded_from_cache: false,
    });
    let mut avatar = AvatarInstance::new(AvatarInstanceId(0), asset);
    avatar.attach_cloth(vulvatar_lib::asset::ClothOverlayId(0));
    if let Some(cs) = avatar.cloth_state.as_mut() {
        cs.solver_backend = ClothSolverBackend::Cpu;
    }
    let mut sim = ClothSimState {
        particles: (0..4)
            .map(|i| ClothParticle::new([i as f32, 0.0, 0.0], false))
            .collect(),
        distance_constraints: Vec::new(),
        bend_constraints: Vec::new(),
        pin_targets: Vec::new(),
        solver_iterations: 4,
        gravity,
        gravity_scale: 1.0,
        damping: 0.01,
        collision_margin: 0.0,
        wind_response: 0.0,
        wind_direction: [1.0, 0.0, 0.0],
        triangle_indices: Vec::new(),
        computed_normals: vec![[0.0; 3]; 4],
        initialized: true,
        self_collision: false,
        self_collision_radius: 0.01,
        connected_pairs: HashSet::new(),
        spatial_hash: SpatialHashGrid::new(0.04),
    };
    sim.gravity = gravity;
    avatar.cloth_sim = Some(sim);
    avatar.cloth_enabled = true;
    avatar
}

#[test]
fn cpu_cloth_settles_to_sleep_and_freezes_bitwise() {
    let mut avatar = cloth_avatar([0.0, 0.0, 0.0]);
    for _ in 0..30 {
        cloth_solver::step_cloth(1.0 / 60.0, &mut avatar, &[]);
    }
    let cs = avatar.cloth_state.as_ref().unwrap();
    assert!(
        cs.settle.sleeping,
        "still cloth must sleep (quiet={}, max_delta={})",
        cs.settle.quiet_frames, cs.settle.last_max_delta
    );
    let frozen = cs.sim_positions.clone();
    for _ in 0..10 {
        cloth_solver::step_cloth(1.0 / 60.0, &mut avatar, &[]);
    }
    assert_eq!(
        avatar.cloth_state.as_ref().unwrap().sim_positions,
        frozen,
        "sleeping CPU cloth must skip the substep entirely"
    );
}

#[test]
fn cpu_cloth_wakes_when_inputs_change() {
    // (a) gravity turns on.
    let mut avatar = cloth_avatar([0.0, 0.0, 0.0]);
    for _ in 0..30 {
        cloth_solver::step_cloth(1.0 / 60.0, &mut avatar, &[]);
    }
    assert!(avatar.cloth_state.as_ref().unwrap().settle.sleeping);
    let frozen = avatar.cloth_state.as_ref().unwrap().sim_positions.clone();
    avatar.cloth_sim.as_mut().unwrap().gravity = [0.0, -9.81, 0.0];
    cloth_solver::step_cloth(1.0 / 60.0, &mut avatar, &[]);
    let cs = avatar.cloth_state.as_ref().unwrap();
    assert_ne!(cs.sim_positions, frozen, "gravity change must wake the cloth");
    assert!(!cs.settle.sleeping);

    // (b) a world collider appears — offset from the particle's exact
    // centre: `collide` skips the degenerate dist==0 case.
    let mut avatar = cloth_avatar([0.0, 0.0, 0.0]);
    for _ in 0..30 {
        cloth_solver::step_cloth(1.0 / 60.0, &mut avatar, &[]);
    }
    assert!(avatar.cloth_state.as_ref().unwrap().settle.sleeping);
    let frozen = avatar.cloth_state.as_ref().unwrap().sim_positions.clone();
    let colliders = vec![ResolvedCollider::Sphere {
        center: [2.96, 0.0, 0.0], // 4 cm from particle 3, inside its 5 cm radius
        radius: 0.05,
    }];
    cloth_solver::step_cloth(1.0 / 60.0, &mut avatar, &colliders);
    let cs = avatar.cloth_state.as_ref().unwrap();
    assert_ne!(
        cs.sim_positions, frozen,
        "new scene collider must wake the cloth and push particles"
    );
    assert!(!cs.settle.sleeping);
}

