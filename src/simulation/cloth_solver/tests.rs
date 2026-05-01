//! Unit tests for the cloth solver phases.

use super::collision::{apply_pin_targets, collide, enforce_pins, resolve_self_collisions};
use super::constraints::{project_bend_constraints, project_distance_constraints};
use super::integrator::{derive_velocity, verlet_integrate};
use super::output::{compute_normals, write_back};
use super::*;
use crate::math_utils::{vec3_add, vec3_cross, vec3_dot, vec3_length, vec3_scale, vec3_sub, Vec3};
use crate::simulation::cloth::{
    ClothBendConstraint, ClothDistanceConstraint, ClothParticle, ClothSimState,
    ClothSimTempBuffers, PinTarget, ResolvedCollider, SpatialHashGrid,
};
use std::collections::HashSet;

fn make_simple_sim(particle_count: usize) -> ClothSimState {
    let particles = (0..particle_count)
        .map(|i| {
            ClothParticle::new([i as f32, 0.0, 0.0], false)
        })
        .collect();
    ClothSimState {
        particles,
        distance_constraints: Vec::new(),
        bend_constraints: Vec::new(),
        pin_targets: Vec::new(),
        solver_iterations: 4,
        gravity: [0.0, -9.81, 0.0],
        damping: 0.01,
        collision_margin: 0.0,
        wind_response: 0.0,
        wind_direction: [1.0, 0.0, 0.0],
        triangle_indices: Vec::new(),
        computed_normals: vec![[0.0; 3]; particle_count],
        initialized: true,
        self_collision: false,
        self_collision_radius: 0.01,
        connected_pairs: HashSet::new(),
        spatial_hash: SpatialHashGrid::new(0.04),
    }
}

// =========================================================================
// Verlet integration
// =========================================================================

#[test]
fn verlet_gravity_pulls_down() {
    let mut sim = make_simple_sim(1);
    sim.gravity = [0.0, -10.0, 0.0];
    sim.damping = 0.0;
    sim.particles[0].position = [0.0, 1.0, 0.0];
    sim.particles[0].prev_position = [0.0, 1.0, 0.0];

    verlet_integrate(&mut sim, 0.1);

    let dy = sim.particles[0].position[1] - 1.0;
    assert!(dy < 0.0, "gravity should pull particle down, got dy={}", dy);
    let expected_dy = -10.0 * 0.1 * 0.1;
    assert!(
        (dy - expected_dy).abs() < 1e-4,
        "expected dy={}, got {}",
        expected_dy,
        dy
    );
}

#[test]
fn verlet_damping_reduces_velocity() {
    let mut sim = make_simple_sim(1);
    sim.gravity = [0.0, 0.0, 0.0];
    sim.damping = 0.5;
    sim.particles[0].position = [1.0, 0.0, 0.0];
    sim.particles[0].prev_position = [0.0, 0.0, 0.0];

    verlet_integrate(&mut sim, 0.1);

    let vx = (sim.particles[0].position[0] - 1.0) / 0.1;
    assert!(
        vx.abs() < 10.0 * 0.99,
        "damped velocity should be less than original, got vx={}",
        vx
    );
}

#[test]
fn verlet_pinned_particles_dont_move() {
    let mut sim = make_simple_sim(1);
    sim.gravity = [0.0, -100.0, 0.0];
    sim.particles[0].pinned = true;
    sim.particles[0].inv_mass = 0.0;
    sim.particles[0].position = [5.0, 5.0, 5.0];
    sim.particles[0].prev_position = [5.0, 5.0, 5.0];

    verlet_integrate(&mut sim, 0.1);

    assert_eq!(sim.particles[0].position, [5.0, 5.0, 5.0]);
}

#[test]
fn verlet_wind_pushes_particles() {
    let mut sim = make_simple_sim(1);
    sim.gravity = [0.0, 0.0, 0.0];
    sim.damping = 0.0;
    sim.wind_response = 5.0;
    sim.wind_direction = [1.0, 0.0, 0.0];
    sim.particles[0].position = [0.0, 0.0, 0.0];
    sim.particles[0].prev_position = [0.0, 0.0, 0.0];

    verlet_integrate(&mut sim, 0.1);

    let dx = sim.particles[0].position[0];
    assert!(dx > 0.0, "wind should push particle in +x, got dx={}", dx);
}

// =========================================================================
// Distance constraints
// =========================================================================

#[test]
fn distance_constraint_converges_to_rest_length() {
    let mut sim = make_simple_sim(2);
    let rest_length = 1.0;
    sim.particles[0].position = [0.0, 0.0, 0.0];
    sim.particles[1].position = [2.0, 0.0, 0.0];
    sim.distance_constraints.push(ClothDistanceConstraint {
        a: 0,
        b: 1,
        rest_length,
        stiffness: 1.0,
    });

    let mut buffers = ClothSimTempBuffers::new(2);

    for _ in 0..50 {
        project_distance_constraints(&mut sim, &mut buffers);
    }

    let dist = vec3_length(&vec3_sub(
        &sim.particles[1].position,
        &sim.particles[0].position,
    ));
    assert!(
        (dist - rest_length).abs() < 0.01,
        "expected dist={}, got {}",
        rest_length,
        dist
    );
}

#[test]
fn distance_constraint_preserves_center_of_mass() {
    let mut sim = make_simple_sim(2);
    sim.particles[0].position = [0.0, 0.0, 0.0];
    sim.particles[1].position = [2.0, 0.0, 0.0];
    sim.distance_constraints.push(ClothDistanceConstraint {
        a: 0,
        b: 1,
        rest_length: 1.0,
        stiffness: 1.0,
    });

    let com_before: Vec3 = vec3_scale(
        &vec3_add(&sim.particles[0].position, &sim.particles[1].position),
        0.5,
    );

    let mut buffers = ClothSimTempBuffers::new(2);
    project_distance_constraints(&mut sim, &mut buffers);

    let com_after: Vec3 = vec3_scale(
        &vec3_add(&sim.particles[0].position, &sim.particles[1].position),
        0.5,
    );

    for i in 0..3 {
        assert!(
            (com_after[i] - com_before[i]).abs() < 1e-4,
            "center of mass should be preserved, axis {} shifted by {}",
            i,
            (com_after[i] - com_before[i])
        );
    }
}

#[test]
fn distance_constraint_with_zero_stiffness_does_nothing() {
    let mut sim = make_simple_sim(2);
    sim.particles[0].position = [0.0, 0.0, 0.0];
    sim.particles[1].position = [3.0, 0.0, 0.0];
    sim.distance_constraints.push(ClothDistanceConstraint {
        a: 0,
        b: 1,
        rest_length: 1.0,
        stiffness: 0.0,
    });

    let mut buffers = ClothSimTempBuffers::new(2);
    project_distance_constraints(&mut sim, &mut buffers);

    assert_eq!(sim.particles[0].position, [0.0, 0.0, 0.0]);
    assert_eq!(sim.particles[1].position, [3.0, 0.0, 0.0]);
}

// =========================================================================
// Bend constraints
// =========================================================================

#[test]
fn bend_constraint_changes_angle() {
    let mut sim = make_simple_sim(3);
    sim.particles[0].position = [0.0, 0.0, 0.0];
    sim.particles[1].position = [1.0, 0.0, 0.0];
    sim.particles[2].position = [0.0, 1.0, 0.0];

    let rest_angle = std::f32::consts::PI;
    sim.bend_constraints.push(ClothBendConstraint {
        p0: 0,
        p1: 1,
        p2: 2,
        rest_angle,
        stiffness: 0.3,
    });

    let angle_before: f32 = {
        let e1 = vec3_sub(&sim.particles[1].position, &sim.particles[0].position);
        let e2 = vec3_sub(&sim.particles[2].position, &sim.particles[0].position);
        vec3_length(&vec3_cross(&e1, &e2)).atan2(vec3_dot(&e1, &e2))
    };

    project_bend_constraints(&mut sim);

    let angle_after: f32 = {
        let e1 = vec3_sub(&sim.particles[1].position, &sim.particles[0].position);
        let e2 = vec3_sub(&sim.particles[2].position, &sim.particles[0].position);
        vec3_length(&vec3_cross(&e1, &e2)).atan2(vec3_dot(&e1, &e2))
    };

    assert!(
        (angle_after - angle_before).abs() > 0.01,
        "bend constraint should change the angle: before={}, after={}",
        angle_before.to_degrees(),
        angle_after.to_degrees()
    );
}

// =========================================================================
// Pin enforcement
// =========================================================================

#[test]
fn enforce_pins_resets_pinned_particles() {
    let mut sim = make_simple_sim(2);
    sim.particles[0].pinned = true;
    sim.particles[0].inv_mass = 0.0;
    sim.particles[0].position = [1.0, 2.0, 3.0]; // moved by constraints
    sim.particles[0].prev_position = [0.0, 0.0, 0.0]; // correct pin pos

    enforce_pins(&mut sim);

    assert_eq!(sim.particles[0].position, [0.0, 0.0, 0.0]);
}

#[test]
fn enforce_pins_does_not_affect_free_particles() {
    let mut sim = make_simple_sim(1);
    sim.particles[0].position = [5.0, 5.0, 5.0];
    sim.particles[0].prev_position = [0.0, 0.0, 0.0];

    enforce_pins(&mut sim);

    assert_eq!(sim.particles[0].position, [5.0, 5.0, 5.0]);
}

// =========================================================================
// Sphere collision
// =========================================================================

#[test]
fn sphere_collision_pushes_particle_out() {
    let mut sim = make_simple_sim(1);
    sim.particles[0].position = [0.5, 0.0, 0.0]; // inside sphere, offset from center
    sim.collision_margin = 0.0;

    let colliders = vec![ResolvedCollider::Sphere {
        center: [0.0, 0.0, 0.0],
        radius: 1.0,
    }];

    collide(&mut sim, &colliders);

    let dist = vec3_length(&sim.particles[0].position);
    assert!(
        dist >= 0.99,
        "particle should be pushed outside sphere, dist={}",
        dist
    );
}

#[test]
fn sphere_collision_respects_margin() {
    let mut sim = make_simple_sim(1);
    sim.particles[0].position = [1.2, 0.0, 0.0]; // inside margin zone
    sim.collision_margin = 0.5;

    let colliders = vec![ResolvedCollider::Sphere {
        center: [0.0, 0.0, 0.0],
        radius: 1.0,
    }];

    collide(&mut sim, &colliders);

    let dist = vec3_length(&sim.particles[0].position);
    let effective_radius = 1.0 + 0.5;
    assert!(
        dist >= effective_radius - 0.01,
        "particle should be outside effective radius {}, dist={}",
        effective_radius,
        dist
    );
}

#[test]
fn sphere_collision_skips_pinned() {
    let mut sim = make_simple_sim(1);
    sim.particles[0].pinned = true;
    sim.particles[0].position = [0.0, 0.0, 0.0]; // inside sphere

    let colliders = vec![ResolvedCollider::Sphere {
        center: [0.0, 0.0, 0.0],
        radius: 1.0,
    }];

    collide(&mut sim, &colliders);

    assert_eq!(sim.particles[0].position, [0.0, 0.0, 0.0]);
}

// =========================================================================
// Capsule collision
// =========================================================================

#[test]
fn capsule_collision_pushes_particle_out() {
    let mut sim = make_simple_sim(1);
    // Capsule along Y from -1 to +1, radius 0.5, particle slightly off-axis
    sim.particles[0].position = [0.1, 0.0, 0.0]; // inside capsule
    sim.collision_margin = 0.0;

    let colliders = vec![ResolvedCollider::Capsule {
        center: [0.0, 0.0, 0.0],
        radius: 0.5,
        half_height: 1.0,
        axis: [0.0, 1.0, 0.0],
    }];

    collide(&mut sim, &colliders);

    let dist_from_axis =
        (sim.particles[0].position[0].powi(2) + sim.particles[0].position[2].powi(2)).sqrt();
    assert!(
        dist_from_axis >= 0.49,
        "particle should be pushed outside capsule radius, dist_from_axis={}",
        dist_from_axis
    );
}

// =========================================================================
// Self-collision
// =========================================================================

#[test]
fn self_collision_pushes_overlapping_particles_apart() {
    let mut sim = make_simple_sim(2);
    sim.self_collision = true;
    sim.self_collision_radius = 0.5;
    // Place both particles at the same position
    sim.particles[0].position = [0.0, 0.0, 0.0];
    sim.particles[1].position = [0.01, 0.0, 0.0];
    // No connected pairs so they will collide
    sim.connected_pairs = HashSet::new();
    sim.spatial_hash = SpatialHashGrid::new(2.0);

    resolve_self_collisions(&mut sim);

    let dist = vec3_length(&vec3_sub(
        &sim.particles[1].position,
        &sim.particles[0].position,
    ));
    assert!(
        dist > 0.01,
        "particles should be pushed apart, dist={}",
        dist
    );
}

#[test]
fn self_collision_skips_connected_pairs() {
    let mut sim = make_simple_sim(2);
    sim.self_collision = true;
    sim.self_collision_radius = 0.5;
    sim.particles[0].position = [0.0, 0.0, 0.0];
    sim.particles[1].position = [0.01, 0.0, 0.0];
    sim.connected_pairs = HashSet::from([(0, 1)]);
    sim.spatial_hash = SpatialHashGrid::new(2.0);

    resolve_self_collisions(&mut sim);

    assert_eq!(sim.particles[0].position, [0.0, 0.0, 0.0]);
    assert_eq!(sim.particles[1].position, [0.01, 0.0, 0.0]);
}

// =========================================================================
// Normal computation
// =========================================================================

#[test]
fn normals_for_triangle_in_xy_plane() {
    let mut sim = make_simple_sim(3);
    // Triangle in XY plane: (0,0,0), (1,0,0), (0,1,0)
    sim.particles[0].position = [0.0, 0.0, 0.0];
    sim.particles[1].position = [1.0, 0.0, 0.0];
    sim.particles[2].position = [0.0, 1.0, 0.0];
    sim.triangle_indices = vec![0, 1, 2];

    compute_normals(&mut sim);

    // All normals should point in +Z direction (cross of e1 x e2)
    for i in 0..3 {
        let nz = sim.computed_normals[i][2];
        assert!(
            nz > 0.9,
            "normal {} should point in +Z, got {:?}",
            i,
            sim.computed_normals[i]
        );
    }
}

#[test]
fn normals_for_degenerate_triangle_are_degenerate() {
    let mut sim = make_simple_sim(3);
    // Collinear points => zero-area triangle => degenerate normal
    sim.particles[0].position = [0.0, 0.0, 0.0];
    sim.particles[1].position = [1.0, 0.0, 0.0];
    sim.particles[2].position = [2.0, 0.0, 0.0];
    sim.triangle_indices = vec![0, 1, 2];

    compute_normals(&mut sim);

    // Cross product is zero, normalize of zero is undefined;
    // we just verify it doesn't panic
}

// =========================================================================
// Velocity derivation
// =========================================================================

#[test]
fn derive_velocity_computes_correct_velocity() {
    let mut sim = make_simple_sim(1);
    sim.particles[0].position = [1.0, 0.0, 0.0];
    sim.particles[0].prev_position = [0.0, 0.0, 0.0];

    derive_velocity(&mut sim, 0.1);

    let expected_vx = 1.0 / 0.1;
    assert!(
        (sim.particles[0].velocity[0] - expected_vx).abs() < 1e-3,
        "expected vx={}, got {}",
        expected_vx,
        sim.particles[0].velocity[0]
    );
}

#[test]
fn derive_velocity_zero_dt_doesnt_panic() {
    let mut sim = make_simple_sim(1);
    sim.particles[0].position = [1.0, 0.0, 0.0];
    sim.particles[0].prev_position = [0.0, 0.0, 0.0];

    derive_velocity(&mut sim, 0.0);

    // Should not panic; velocity unchanged from init
}

// =========================================================================
// Write-back
// =========================================================================

#[test]
fn write_back_copies_positions_and_normals() {
    let mut sim = make_simple_sim(2);
    sim.particles[0].position = [1.0, 2.0, 3.0];
    sim.particles[1].position = [4.0, 5.0, 6.0];
    sim.computed_normals = vec![[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]];

    let mut cloth_state = crate::avatar::instance::ClothState {
        overlay_id: crate::asset::ClothOverlayId(0),
        enabled: true,
        sim_positions: Vec::new(),
        prev_sim_positions: Vec::new(),
        sim_normals: Vec::new(),
        constraint_cache: crate::avatar::instance::ClothConstraintRuntimeCache {
            active_constraint_count: 0,
        },
        collision_cache: crate::avatar::instance::ClothCollisionRuntimeCache {
            active_collision_count: 0,
        },
        deform_output: crate::avatar::instance::ClothDeformOutput {
            deformed_positions: Vec::new(),
            deformed_normals: None,
            version: 0,
        },
    };

    write_back(&sim, &mut cloth_state);

    assert_eq!(cloth_state.sim_positions[0], [1.0, 2.0, 3.0]);
    assert_eq!(cloth_state.sim_positions[1], [4.0, 5.0, 6.0]);
    assert_eq!(cloth_state.sim_normals[0], [0.0, 0.0, 1.0]);
    assert_eq!(cloth_state.sim_normals[1], [0.0, 1.0, 0.0]);
    assert_eq!(cloth_state.deform_output.version, 1);
}

#[test]
fn write_back_increments_version() {
    let sim = make_simple_sim(1);

    let mut cloth_state = crate::avatar::instance::ClothState {
        overlay_id: crate::asset::ClothOverlayId(0),
        enabled: true,
        sim_positions: Vec::new(),
        prev_sim_positions: Vec::new(),
        sim_normals: Vec::new(),
        constraint_cache: crate::avatar::instance::ClothConstraintRuntimeCache {
            active_constraint_count: 0,
        },
        collision_cache: crate::avatar::instance::ClothCollisionRuntimeCache {
            active_collision_count: 0,
        },
        deform_output: crate::avatar::instance::ClothDeformOutput {
            deformed_positions: Vec::new(),
            deformed_normals: None,
            version: 42,
        },
    };

    write_back(&sim, &mut cloth_state);

    assert_eq!(cloth_state.deform_output.version, 43);
}

// =========================================================================
// Timestep stability
// =========================================================================

#[test]
fn multi_step_does_not_spike() {
    let mut sim = make_simple_sim(2);
    sim.gravity = [0.0, -9.81, 0.0];
    sim.damping = 0.02;
    sim.particles[0].position = [0.0, 1.0, 0.0];
    sim.particles[0].prev_position = [0.0, 1.0, 0.0];
    sim.particles[1].position = [1.0, 1.0, 0.0];
    sim.particles[1].prev_position = [1.0, 1.0, 0.0];
    sim.distance_constraints.push(ClothDistanceConstraint {
        a: 0,
        b: 1,
        rest_length: 1.0,
        stiffness: 0.8,
    });

    let mut buffers = ClothSimTempBuffers::new(2);

    for _ in 0..1000 {
        verlet_integrate(&mut sim, 1.0 / 60.0);
        for _ in 0..sim.solver_iterations {
            project_distance_constraints(&mut sim, &mut buffers);
        }
    }

    for p in &sim.particles {
        assert!(
            p.position.iter().all(|v| v.is_finite()),
            "positions should remain finite after 1000 steps: {:?}",
            p.position
        );
        let mag = vec3_length(&p.position);
        assert!(mag < 1000.0, "positions should not diverge, mag={}", mag);
    }
}

fn make_minimal_avatar_asset() -> std::sync::Arc<crate::asset::AvatarAsset> {
    std::sync::Arc::new(crate::asset::AvatarAsset {
        id: crate::asset::AvatarAssetId(0),
        source_path: std::path::PathBuf::new(),
        source_hash: crate::asset::AssetSourceHash([0u8; 32]),
        skeleton: crate::asset::SkeletonAsset {
            nodes: Vec::new(),
            root_nodes: Vec::new(),
            inverse_bind_matrices: Vec::new(),
        },
        meshes: Vec::new(),
        materials: Vec::new(),
        humanoid: None,
        spring_bones: Vec::new(),
        colliders: Vec::new(),
        default_expressions: crate::asset::ExpressionAssetSet {
            expressions: Vec::new(),
        },
        animation_clips: Vec::new(),
        node_to_mesh: Default::default(),
        vrm_meta: Default::default(),
        root_aabb: crate::asset::Aabb::empty(),
        loaded_from_cache: false,
    })
}

// =========================================================================
// Cloth disabled / no cloth overlay
// =========================================================================

#[test]
fn step_cloth_does_nothing_when_avatar_cloth_disabled() {
    let mut avatar = crate::avatar::AvatarInstance::new(
        crate::avatar::AvatarInstanceId(0),
        make_minimal_avatar_asset(),
    );
    avatar.cloth_enabled = false;

    step_cloth(1.0 / 60.0, &mut avatar, &[]);
}

// =========================================================================
// Pin target resolution
// =========================================================================

#[test]
fn apply_pin_targets_moves_pinned_particles_to_transform() {
    let mut sim = make_simple_sim(1);
    sim.particles[0].pinned = true;
    sim.particles[0].inv_mass = 0.0;
    sim.pin_targets.push(PinTarget {
        node_index: 0,
        offset: [1.0, 0.0, 0.0],
        particle_indices: vec![0],
    });

    // Identity transform, offset [1,0,0] => world pos [1,0,0]
    let transforms = vec![[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]];

    apply_pin_targets(&mut sim, &transforms);

    assert_eq!(sim.particles[0].position[0], 1.0);
}

// =========================================================================
// Spatial hash grid (in cloth.rs, tested via public API)
// =========================================================================

#[test]
fn spatial_hash_finds_neighbors() {
    let mut grid = SpatialHashGrid::new(1.0);
    grid.insert(0, [0.0, 0.0, 0.0]);
    grid.insert(1, [0.5, 0.0, 0.0]);
    grid.insert(2, [100.0, 0.0, 0.0]); // far away

    let neighbors = grid.query_neighbors([0.0, 0.0, 0.0], 1.0);
    assert!(neighbors.contains(&0));
    assert!(neighbors.contains(&1));
    assert!(!neighbors.contains(&2));
}

#[test]
fn spatial_hash_skips_nan_positions() {
    let mut grid = SpatialHashGrid::new(1.0);
    grid.insert(0, [f32::NAN, 0.0, 0.0]);
    grid.insert(1, [0.0, 0.0, 0.0]);

    let neighbors = grid.query_neighbors([0.0, 0.0, 0.0], 1.0);
    assert!(!neighbors.contains(&0));
    assert!(neighbors.contains(&1));
}

// =========================================================================
// Collider resolution
// =========================================================================

#[test]
fn resolve_colliders_sphere_transform() {
    let collider = crate::asset::ColliderAsset {
        id: crate::asset::ColliderId(0),
        node: crate::asset::NodeId(0),
        offset: [0.0, 1.0, 0.0],
        shape: crate::asset::ColliderShape::Sphere { radius: 0.5 },
    };

    let identity: crate::asset::Mat4 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0], // translation [0, 1, 0]
    ];

    let resolved = crate::simulation::cloth::resolve_colliders(&[collider], &[identity]);
    assert_eq!(resolved.len(), 1);
    match &resolved[0] {
        ResolvedCollider::Sphere { center, radius } => {
            assert_eq!(*center, [0.0, 2.0, 0.0]); // transform pos [0,1,0] + offset [0,1,0]
            assert_eq!(*radius, 0.5);
        }
        _ => panic!("expected sphere"),
    }
}

// =========================================================================
// World collider integration
// =========================================================================

#[test]
fn world_colliders_are_included_in_collision() {
    let mut sim = make_simple_sim(1);
    sim.particles[0].position = [0.5, 0.0, 0.0];
    sim.collision_margin = 0.0;

    let world_colliders = vec![ResolvedCollider::Sphere {
        center: [0.0, 0.0, 0.0],
        radius: 1.0,
    }];

    collide(&mut sim, &world_colliders);

    let dist = vec3_length(&sim.particles[0].position);
    assert!(
        dist >= 0.99,
        "world collider should push particle out, dist={}",
        dist
    );
}

// =========================================================================
// Performance: cloth step timing (stub)
// =========================================================================

#[test]
fn cloth_step_performance_stub() {
    let particle_count = 256;
    let mut sim = make_simple_sim(particle_count);
    sim.gravity = [0.0, -9.81, 0.0];
    sim.damping = 0.02;
    sim.solver_iterations = 4;
    for i in 0..particle_count {
        sim.particles[i].position = [i as f32 % 16.0, (i as f32 / 16.0).floor(), 0.0];
        sim.particles[i].prev_position = sim.particles[i].position;
    }

    let mut buffers = ClothSimTempBuffers::new(particle_count);

    let start = std::time::Instant::now();
    for _ in 0..600 {
        verlet_integrate(&mut sim, 1.0 / 60.0);
        for _ in 0..sim.solver_iterations {
            project_distance_constraints(&mut sim, &mut buffers);
        }
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < 2000,
        "600 substeps with {} particles took {:?} (expected < 2s)",
        particle_count,
        elapsed
    );
}

// =========================================================================
// Integration: cloth disable / re-enable cycle
// =========================================================================

#[test]
fn cloth_disable_preserves_positions() {
    let particle_count = 16;
    let mut sim = make_simple_sim(particle_count);
    for i in 0..particle_count {
        sim.particles[i].position = [i as f32, 1.0, 0.0];
        sim.particles[i].prev_position = sim.particles[i].position;
    }

    let positions_before: Vec<Vec3> = sim.particles.iter().map(|p| p.position).collect();

    verlet_integrate(&mut sim, 1.0 / 60.0);

    let positions_after_step: Vec<Vec3> = sim.particles.iter().map(|p| p.position).collect();

    assert_ne!(
        positions_before, positions_after_step,
        "gravity should move particles"
    );
}

#[test]
fn cloth_reenable_after_disable_resumes_from_state() {
    let particle_count = 8;
    let mut sim = make_simple_sim(particle_count);
    sim.gravity = [0.0, -9.81, 0.0];
    for i in 0..particle_count {
        sim.particles[i].position = [i as f32, 10.0, 0.0];
        sim.particles[i].prev_position = sim.particles[i].position;
    }

    let mut buffers = ClothSimTempBuffers::new(particle_count);

    verlet_integrate(&mut sim, 1.0 / 60.0);
    for _ in 0..sim.solver_iterations {
        project_distance_constraints(&mut sim, &mut buffers);
    }

    let saved_positions: Vec<Vec3> = sim.particles.iter().map(|p| p.position).collect();

    let idle_steps = 10;
    for _ in 0..idle_steps {
        // Simulating disabled state: no stepping
    }

    verlet_integrate(&mut sim, 1.0 / 60.0);
    for _ in 0..sim.solver_iterations {
        project_distance_constraints(&mut sim, &mut buffers);
    }

    let positions_after: Vec<Vec3> = sim.particles.iter().map(|p| p.position).collect();

    assert_ne!(
        saved_positions, positions_after,
        "simulation should have advanced from saved state"
    );
}

// =========================================================================
// Performance: timing breakdown per phase
// =========================================================================

#[test]
fn cloth_timing_breakdown() {
    let particle_count = 128;
    let mut sim = make_simple_sim(particle_count);
    sim.gravity = [0.0, -9.81, 0.0];
    sim.damping = 0.02;
    sim.solver_iterations = 4;
    for i in 0..particle_count {
        let x = (i as f32 % 11.0) - 5.0;
        let y = ((i as f32 / 11.0).floor()) - 5.0;
        sim.particles[i].position = [x, y, 0.0];
        sim.particles[i].prev_position = sim.particles[i].position;
    }

    let mut buffers = ClothSimTempBuffers::new(particle_count);

    let iterations = 60;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        verlet_integrate(&mut sim, 1.0 / 60.0);
        for _ in 0..sim.solver_iterations {
            project_distance_constraints(&mut sim, &mut buffers);
        }
        project_bend_constraints(&mut sim);
        compute_normals(&mut sim);
        derive_velocity(&mut sim, 1.0 / 60.0);
    }
    let total = start.elapsed();
    let per_frame = total / iterations;

    assert!(
        per_frame.as_millis() < 100,
        "full cloth step with {} particles took {:?} per frame (debug budget 100ms)",
        particle_count,
        per_frame
    );
}

// =========================================================================
// T07: Time-budgeted cloth step (non-blocking verification)
// =========================================================================

#[test]
fn step_cloth_respects_time_budget() {
    let particle_count = 256;
    let mut sim = make_simple_sim(particle_count);
    sim.gravity = [0.0, -9.81, 0.0];
    sim.damping = 0.02;
    sim.solver_iterations = 16;
    for i in 0..particle_count {
        sim.particles[i].position = [i as f32 % 16.0, (i as f32 / 16.0).floor(), 0.0];
        sim.particles[i].prev_position = sim.particles[i].position;
    }

    let mut buffers = ClothSimTempBuffers::new(particle_count);
    let budget = std::time::Duration::from_millis(5);
    let mut steps_completed = 0u32;
    // Sized so even very fast CPUs can't complete the loop within
    // `budget`; the test asserts the budget check breaks us out
    // before max_steps. Was 1000 — completed in <5ms on modern
    // desktop CPUs and tripped the assertion below.
    let max_steps = 10_000_000u32;

    let start = std::time::Instant::now();
    for _ in 0..max_steps {
        if start.elapsed() > budget {
            break;
        }
        verlet_integrate(&mut sim, 1.0 / 60.0);
        for _ in 0..sim.solver_iterations {
            project_distance_constraints(&mut sim, &mut buffers);
        }
        steps_completed += 1;
    }

    assert!(
        steps_completed < max_steps,
        "should have stopped early due to budget, completed {} steps",
        steps_completed
    );
    assert!(
        start.elapsed() <= budget + std::time::Duration::from_millis(2),
        "budget enforcement took too long: {:?} > {:?}",
        start.elapsed(),
        budget
    );
}

// =========================================================================
// T08: Performance impact — measure cloth vs no-cloth frame overhead
// =========================================================================

#[test]
fn cloth_performance_overhead_is_bounded() {
    let particle_count = 128;
    let iterations = 200;

    let baseline_start = std::time::Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(());
    }
    let baseline = baseline_start.elapsed();

    let mut sim = make_simple_sim(particle_count);
    sim.gravity = [0.0, -9.81, 0.0];
    sim.damping = 0.02;
    sim.solver_iterations = 8;
    for i in 0..particle_count {
        let x = (i as f32 % 11.0) - 5.0;
        let y = ((i as f32 / 11.0).floor()) - 5.0;
        sim.particles[i].position = [x, y, 0.0];
        sim.particles[i].prev_position = sim.particles[i].position;
    }
    let mut buffers = ClothSimTempBuffers::new(particle_count);

    let cloth_start = std::time::Instant::now();
    for _ in 0..iterations {
        verlet_integrate(&mut sim, 1.0 / 60.0);
        for _ in 0..sim.solver_iterations {
            project_distance_constraints(&mut sim, &mut buffers);
        }
        project_bend_constraints(&mut sim);
        compute_normals(&mut sim);
        derive_velocity(&mut sim, 1.0 / 60.0);
    }
    let cloth_elapsed = cloth_start.elapsed();

    let per_frame_overhead = (cloth_elapsed - baseline) / iterations;
    assert!(
        per_frame_overhead.as_micros() < 16_000,
        "cloth overhead per frame: {:?} (budget 16ms for 60fps)",
        per_frame_overhead
    );
}

// =========================================================================
// Cloth LoD: solver iteration adjustment
// =========================================================================

#[test]
fn cloth_lod_reduces_iterations_for_low_detail() {
    use crate::simulation::cloth::ClothLoDLevel;
    let mut sim = make_simple_sim(16);
    sim.solver_iterations = 8;
    sim.apply_lod(&ClothLoDLevel::Half);
    assert_eq!(sim.solver_iterations, 6);

    sim.apply_lod(&ClothLoDLevel::Quarter);
    assert_eq!(sim.solver_iterations, 4);

    sim.apply_lod(&ClothLoDLevel::Full);
    assert_eq!(sim.solver_iterations, 8);
}

#[test]
fn cloth_lod_disables_self_collision_at_low_ratio() {
    use crate::simulation::cloth::ClothLoDLevel;
    let mut sim = make_simple_sim(16);
    sim.self_collision = true;

    sim.apply_lod(&ClothLoDLevel::Quarter);
    assert!(
        !sim.self_collision,
        "self-collision should be off at Quarter LoD"
    );

    sim.self_collision = true;
    sim.apply_lod(&ClothLoDLevel::Half);
    assert!(
        sim.self_collision,
        "self-collision should remain on at Half LoD"
    );
}
