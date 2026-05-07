pub mod cloth;
pub mod cloth_solver;
pub mod spring;

use log::info;

use crate::asset::AvatarAsset;
use crate::avatar::AvatarInstance;
use crate::simulation::cloth::ClothLoDConfig;

pub struct SimulationClock {
    accumulator: f32,
    fixed_dt: f32,
    max_substeps: u32,
}

impl SimulationClock {
    pub fn new(fixed_dt: f32, max_substeps: u32) -> Self {
        Self {
            accumulator: 0.0,
            fixed_dt,
            max_substeps,
        }
    }

    pub fn advance(&mut self, frame_dt: f32) -> u32 {
        self.accumulator += frame_dt;
        // Clamp accumulator before the loop to prevent spiral of death.
        self.accumulator = self
            .accumulator
            .min(self.fixed_dt * self.max_substeps as f32);
        let mut substeps = 0u32;
        while self.accumulator >= self.fixed_dt && substeps < self.max_substeps {
            self.accumulator -= self.fixed_dt;
            substeps += 1;
        }
        substeps
    }

    pub fn fixed_dt(&self) -> f32 {
        self.fixed_dt
    }

    pub fn reset(&mut self) {
        self.accumulator = 0.0;
    }
}

/// Per-frame gates for the secondary-motion solvers driven from the GUI's
/// `RuntimeToggles`. Kept separate from `RuntimeToggles` so the simulation
/// crate doesn't depend on `app::*`, and so the Rapier/non-Rapier code paths
/// can both honour the same set of switches uniformly.
#[derive(Clone, Copy, Debug, Default)]
pub struct SimulationStepOptions {
    pub spring_enabled: bool,
    pub cloth_enabled: bool,
}

pub struct PhysicsWorld {
    collider_count: usize,
    spring_chain_count: usize,
    scene_colliders: Vec<crate::asset::SceneColliderAsset>,
    cloth_lod_presets: Vec<ClothLoDConfig>,
    #[cfg(feature = "rapier")]
    rapier: Option<RapierWorld>,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            collider_count: 0,
            spring_chain_count: 0,
            scene_colliders: Vec::new(),
            cloth_lod_presets: ClothLoDConfig::presets(),
            #[cfg(feature = "rapier")]
            rapier: None,
        }
    }

    pub fn attach_avatar(&mut self, asset: &AvatarAsset) {
        self.collider_count = asset.colliders.len();
        self.spring_chain_count = asset.spring_bones.len();
        info!(
            "physics: attached avatar with {} collider(s) and {} spring chain(s)",
            self.collider_count, self.spring_chain_count
        );

        #[cfg(feature = "rapier")]
        {
            if self.rapier.is_none() {
                self.init_rapier();
            }
            if let Some(ref mut rapier) = self.rapier {
                rapier.add_character_body([0.0, 0.0, 0.0], 0.3, 0.5);
            }
        }
    }

    pub fn step_springs(&mut self, fixed_dt: f32, substeps: u32, avatar: &mut AvatarInstance) {
        let world_colliders = cloth::resolve_scene_colliders(&self.scene_colliders);
        for _ in 0..substeps {
            spring::step_spring_bones(fixed_dt, avatar, &world_colliders);
        }
    }

    pub fn step_cloth(&mut self, dt: f32, avatar: &mut AvatarInstance) {
        let world_colliders = cloth::resolve_scene_colliders(&self.scene_colliders);
        cloth_solver::step_cloth(dt, avatar, &world_colliders);
    }

    pub fn step_cloth_with_camera_distance(
        &mut self,
        dt: f32,
        avatar: &mut AvatarInstance,
        camera_distance: f32,
    ) {
        let world_colliders = cloth::resolve_scene_colliders(&self.scene_colliders);
        if let Some(ref mut sim) = avatar.cloth_sim {
            let lod = ClothLoDConfig::select_for_distance(&self.cloth_lod_presets, camera_distance);
            sim.apply_lod(&lod);
        }
        for slot in &mut avatar.cloth_overlays {
            let lod = ClothLoDConfig::select_for_distance(&self.cloth_lod_presets, camera_distance);
            slot.sim.apply_lod(&lod);
        }
        cloth_solver::step_cloth(dt, avatar, &world_colliders);
    }

    /// Step all enabled secondary-motion solvers for `substeps` ticks of
    /// `fixed_dt` seconds each. The frame-level [`SimulationClock`] lives in
    /// `Application` so its accumulator advances once per frame and the
    /// resulting `(fixed_dt, substeps)` are reused for every avatar; passing
    /// them in here avoids the per-avatar consumption bug where the first
    /// avatar drains the accumulator and later avatars get zero substeps.
    ///
    /// `options` gates the individual solvers so the GUI's spring/cloth
    /// toggles work the same way whether or not the Rapier branch is taken.
    pub fn step_all(
        &mut self,
        fixed_dt: f32,
        substeps: u32,
        avatar: &mut AvatarInstance,
        options: SimulationStepOptions,
    ) {
        if substeps == 0 {
            return;
        }
        let world_colliders = cloth::resolve_scene_colliders(&self.scene_colliders);
        for _ in 0..substeps {
            if options.spring_enabled {
                spring::step_spring_bones(fixed_dt, avatar, &world_colliders);
            }
            if options.cloth_enabled {
                cloth_solver::step_cloth(fixed_dt, avatar, &world_colliders);
            }
            #[cfg(feature = "rapier")]
            if let Some(ref mut rapier) = self.rapier {
                rapier.step(fixed_dt);
            }
        }
    }

    pub fn add_scene_collider(&mut self, collider: crate::asset::SceneColliderAsset) {
        #[cfg(feature = "rapier")]
        if let Some(ref mut rapier) = self.rapier {
            match collider.shape {
                crate::asset::ColliderShape::Sphere { radius } => {
                    rapier.add_static_sphere(collider.position, radius);
                }
                crate::asset::ColliderShape::Capsule { radius, height } => {
                    rapier.add_static_capsule(collider.position, radius, height * 0.5);
                }
            }
        }
        info!(
            "physics: added scene collider {:?} at {:?}",
            collider.id, collider.position
        );
        self.scene_colliders.push(collider);
    }

    pub fn scene_collider_count(&self) -> usize {
        self.scene_colliders.len()
    }

    pub fn clear_scene_colliders(&mut self) {
        self.scene_colliders.clear();
    }

    /// Initialise Rapier world (requires the `rapier` feature).
    #[cfg(feature = "rapier")]
    pub fn init_rapier(&mut self) {
        self.rapier = Some(RapierWorld::new());
        info!("physics: rapier world initialised");
    }

    /// No-op when the rapier feature is not enabled.
    #[cfg(not(feature = "rapier"))]
    pub fn init_rapier(&mut self) {
        // Rapier feature not compiled in; nothing to do.
    }

    #[cfg(feature = "rapier")]
    pub fn rapier_initialized(&self) -> bool {
        self.rapier.is_some()
    }

    #[cfg(not(feature = "rapier"))]
    pub fn rapier_initialized(&self) -> bool {
        false
    }

    #[cfg(feature = "rapier")]
    pub fn character_position(&self) -> Option<[f32; 3]> {
        self.rapier.as_ref().and_then(|r| r.character_position())
    }

    #[cfg(not(feature = "rapier"))]
    pub fn character_position(&self) -> Option<[f32; 3]> {
        None
    }

    #[cfg(feature = "rapier")]
    pub fn remove_character_bodies(&mut self, _path: &str) {
        if let Some(ref mut rapier) = self.rapier {
            rapier.remove_character_body();
        }
        self.collider_count = 0;
        self.spring_chain_count = 0;
        info!("physics: detached avatar");
    }

    #[cfg(not(feature = "rapier"))]
    pub fn remove_character_bodies(&mut self, _path: &str) {
        self.collider_count = 0;
        self.spring_chain_count = 0;
        info!("physics: detached avatar");
    }

    /// Add a static sphere collider to the Rapier world (e.g. avatar body part).
    #[cfg(feature = "rapier")]
    pub fn add_static_collider(&mut self, position: [f32; 3], radius: f32) {
        if let Some(ref mut rapier) = self.rapier {
            rapier.add_static_sphere(position, radius);
        }
    }

    #[cfg(not(feature = "rapier"))]
    pub fn add_static_collider(&mut self, _position: [f32; 3], _radius: f32) {
        // No-op without rapier feature.
    }

    /// Query contacts within a radius around a point.
    /// Returns (contact_point, contact_normal, penetration_depth) tuples.
    #[cfg(feature = "rapier")]
    pub fn query_contacts(
        &self,
        position: [f32; 3],
        radius: f32,
    ) -> Vec<([f32; 3], [f32; 3], f32)> {
        if let Some(ref rapier) = self.rapier {
            rapier.query_contacts(position, radius)
        } else {
            Vec::new()
        }
    }

    #[cfg(not(feature = "rapier"))]
    pub fn query_contacts(
        &self,
        _position: [f32; 3],
        _radius: f32,
    ) -> Vec<([f32; 3], [f32; 3], f32)> {
        Vec::new()
    }
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Rapier integration (behind feature flag)
// ===========================================================================

#[cfg(feature = "rapier")]
use rapier3d::prelude::*;

/// Thin wrapper around Rapier3D rigid-body and collider sets plus a physics
/// pipeline.  Used for optional world-level collision (scene colliders,
/// character controller body) rather than as the primary cloth/spring solver.
#[cfg(feature = "rapier")]
pub struct RapierWorld {
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,
    character_body: Option<RigidBodyHandle>,
}

#[cfg(feature = "rapier")]
impl RapierWorld {
    pub fn new() -> Self {
        Self {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            character_body: None,
        }
    }

    /// Add a fixed (static) sphere collider at the given world position.
    pub fn add_static_sphere(&mut self, position: [f32; 3], radius: f32) {
        let rb = RigidBodyBuilder::fixed()
            .translation(vector![position[0], position[1], position[2]])
            .build();
        let rb_handle = self.rigid_body_set.insert(rb);

        let collider = ColliderBuilder::ball(radius).build();
        self.collider_set
            .insert_with_parent(collider, rb_handle, &mut self.rigid_body_set);
    }

    /// Add a fixed (static) capsule collider at the given world position.
    pub fn add_static_capsule(&mut self, position: [f32; 3], radius: f32, half_height: f32) {
        let rb = RigidBodyBuilder::fixed()
            .translation(vector![position[0], position[1], position[2]])
            .build();
        let rb_handle = self.rigid_body_set.insert(rb);

        let collider = ColliderBuilder::capsule_y(half_height, radius).build();
        self.collider_set
            .insert_with_parent(collider, rb_handle, &mut self.rigid_body_set);
    }

    /// Step the Rapier pipeline by one tick.
    pub fn step(&mut self, dt: f32) {
        self.integration_parameters.dt = dt;
        self.physics_pipeline.step(
            &vector![0.0, -9.81, 0.0],
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            None,
            &(),
            &(),
        );
        self.query_pipeline.update(&self.collider_set);
    }

    /// Query contacts: find colliders intersecting a sphere at `position`
    /// with given `radius`.  Returns a list of (contact_point, normal, depth).
    pub fn query_contacts(
        &self,
        position: [f32; 3],
        radius: f32,
    ) -> Vec<([f32; 3], [f32; 3], f32)> {
        let mut results = Vec::new();
        let shape = rapier3d::geometry::Ball::new(radius);
        let shape_pos = Isometry::translation(position[0], position[1], position[2]);

        self.query_pipeline.intersections_with_shape(
            &self.rigid_body_set,
            &self.collider_set,
            &shape_pos,
            &shape,
            QueryFilter::default(),
            |handle| {
                // For each intersecting collider, compute a contact pair.
                if let Some(collider) = self.collider_set.get(handle) {
                    let col_pos = collider.position();
                    let t = col_pos.translation;
                    // Approximate contact point as collider centre
                    // (precise contact requires shape-specific logic).
                    let contact_point = [t.x, t.y, t.z];
                    let dx = position[0] - t.x;
                    let dy = position[1] - t.y;
                    let dz = position[2] - t.z;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-12);
                    let normal = [dx / dist, dy / dist, dz / dist];
                    let depth = (radius - dist).max(0.0);
                    results.push((contact_point, normal, depth));
                }
                true // continue iteration
            },
        );
        results
    }

    pub fn add_character_body(&mut self, position: [f32; 3], radius: f32, half_height: f32) {
        let rb = RigidBodyBuilder::dynamic()
            .translation(vector![position[0], position[1], position[2]])
            .lock_rotations()
            .build();
        let rb_handle = self.rigid_body_set.insert(rb);

        let capsule = ColliderBuilder::capsule_y(half_height, radius)
            .restitution(0.0)
            .friction(0.7)
            .build();
        self.collider_set
            .insert_with_parent(capsule, rb_handle, &mut self.rigid_body_set);

        self.character_body = Some(rb_handle);
        info!(
            "physics: added character body at {:?} (radius={}, half_height={})",
            position, radius, half_height
        );
    }

    pub fn remove_character_body(&mut self) {
        if let Some(handle) = self.character_body.take() {
            self.rigid_body_set.remove(
                handle,
                &mut self.island_manager,
                &mut self.collider_set,
                &mut self.impulse_joint_set,
                &mut self.multibody_joint_set,
                true,
            );
        }
    }

    pub fn move_character(&mut self, velocity: [f32; 3]) {
        if let Some(handle) = self.character_body {
            if let Some(body) = self.rigid_body_set.get_mut(handle) {
                body.set_linvel(vector![velocity[0], velocity[1], velocity[2]], true);
            }
        }
    }

    pub fn character_position(&self) -> Option<[f32; 3]> {
        self.character_body.and_then(|handle| {
            self.rigid_body_set.get(handle).map(|body| {
                let t = body.translation();
                [t.x, t.y, t.z]
            })
        })
    }

    pub fn query_sphere_cast(
        &self,
        origin: [f32; 3],
        direction: [f32; 3],
        max_distance: f32,
        radius: f32,
    ) -> Option<([f32; 3], [f32; 3])> {
        let shape = rapier3d::geometry::Ball::new(radius);
        let origin_iso = Isometry::translation(origin[0], origin[1], origin[2]);
        let dir = vector![direction[0], direction[1], direction[2]];
        let options = rapier3d::parry::query::ShapeCastOptions {
            max_time_of_impact: max_distance,
            stop_at_penetration: true,
            ..Default::default()
        };
        let (_handle, hit) = self.query_pipeline.cast_shape(
            &self.rigid_body_set,
            &self.collider_set,
            &origin_iso,
            &dir,
            &shape,
            options,
            QueryFilter::default(),
        )?;
        let toi = hit.time_of_impact;
        let hit_point = [
            origin[0] + direction[0] * toi,
            origin[1] + direction[1] * toi,
            origin[2] + direction[2] * toi,
        ];
        let normal = [-direction[0], -direction[1], -direction[2]];
        Some((hit_point, normal))
    }
}

#[cfg(feature = "rapier")]
impl Default for RapierWorld {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::{
        identity_matrix, AssetSourceHash, AvatarAssetId, ExpressionAssetSet, NodeId,
        SkeletonAsset, SkeletonNode, SpringBoneAsset, Transform,
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
                    translation: [0.0, 1.0, 0.0],
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
                    translation: [0.0, 1.0, 0.0],
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
            gravity_dir: [0.0, -1.0, 0.0],
            gravity_power: 1.0,
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
        );

        let after: Vec<Vec<[f32; 3]>> = avatar
            .secondary_motion
            .spring_states
            .iter()
            .map(|s| s.positions.clone())
            .collect();
        assert_ne!(before, after);
    }
}
