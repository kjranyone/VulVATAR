pub mod cloth;
pub mod cloth_gpu_boundary;
pub mod cloth_solver;
pub mod spring;

use log::info;

use crate::asset::AvatarAsset;
use crate::avatar::AvatarInstance;
use crate::math_utils::{quat_conjugate, quat_rotate_vec3, vec3_normalize, vec3_scale, Quat, Vec3};
use crate::simulation::cloth::ClothLoDConfig;

/// Scene-wide gravity shared by every secondary-motion solver (spring
/// bones, cloth, Rapier). One source of truth replaces the three
/// independent gravities the solvers used to hardcode.
///
/// `direction` is a **world-space** vector (need not be unit; normalised
/// on use). `strength` is a **dimensionless multiplier** on Earth gravity
/// — 1.0 = normal, 0.0 = weightless, 2.0 = heavy. A dimensionless
/// multiplier (not m/s²) is deliberate: spring bones integrate a unitless
/// `gravityPower` gain, not an acceleration, so a single m/s² number
/// cannot feed all three solvers. Instead every solver keeps its authored
/// baseline (VRM `gravityPower`, cloth `gravity_scale`) and this scales
/// them uniformly.
///
/// The sims run in **avatar-local** space (`compute_global_transforms`
/// does not fold in `world_transform`), so for a rotated avatar the
/// world-space direction is converted into that avatar's local frame with
/// the inverse of its `world_transform` rotation. Rapier runs in world
/// space and takes [`Self::world_accel`] directly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneGravity {
    pub direction: [f32; 3],
    pub strength: f32,
}

impl SceneGravity {
    /// Earth gravity magnitude (m/s²), the baseline `strength == 1.0`
    /// scales. Cloth/Rapier are in m/s²; spring folds `strength` into its
    /// unitless power directly (see [`Self::spring_power_scale`]).
    pub const EARTH_G: f32 = 9.81;
    pub const STRENGTH_RANGE: std::ops::RangeInclusive<f32> = 0.0..=3.0;

    /// Unit direction, falling back to straight down if `direction` is
    /// degenerate (so a zeroed control never yields NaNs).
    fn dir_unit(&self) -> Vec3 {
        let n = vec3_normalize(&self.direction);
        if n == [0.0, 0.0, 0.0] {
            [0.0, -1.0, 0.0]
        } else {
            n
        }
    }

    /// World-space acceleration vector (m/s²) for Rapier.
    pub fn world_accel(&self) -> Vec3 {
        vec3_scale(&self.dir_unit(), Self::EARTH_G * self.strength)
    }

    /// Avatar-local acceleration vector (m/s²) for cloth, given the
    /// avatar's `world_transform` rotation (inverse-rotated into local).
    pub fn local_accel(&self, world_rot: &Quat) -> Vec3 {
        quat_rotate_vec3(&quat_conjugate(world_rot), &self.world_accel())
    }

    /// Avatar-local unit down direction for spring bones (the sim applies
    /// its own unitless power along this axis).
    pub fn local_dir(&self, world_rot: &Quat) -> Vec3 {
        vec3_normalize(&quat_rotate_vec3(&quat_conjugate(world_rot), &self.dir_unit()))
    }

    /// Multiplier folded into each spring joint's authored `gravityPower`.
    pub fn spring_power_scale(&self) -> f32 {
        self.strength
    }
}

impl Default for SceneGravity {
    fn default() -> Self {
        Self {
            direction: [0.0, -1.0, 0.0],
            strength: 1.0,
        }
    }
}

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

/// Derive each cloth sim's gravity from the scene gravity, once per frame
/// before integration. Cloth runs in avatar-local space, so the world
/// gravity is inverse-rotated by the avatar's `world_transform`; each
/// sim's own `gravity_scale` is preserved as a relative multiplier so
/// per-garment heaviness survives. With the default gravity (down,
/// strength 1) this reproduces the old `[0, -9.81*scale, 0]` exactly.
///
/// Public so the Cloth Authoring panel's manual "Step" button (which
/// calls `cloth_solver::step_cloth` directly, bypassing the
/// [`PhysicsWorld`] wrappers) can recompute `sim.gravity` from the edited
/// `gravity_scale` before stepping — otherwise a paused-authoring Step
/// would integrate a stale gravity and the slider would look dead.
pub fn apply_cloth_gravity(avatar: &mut AvatarInstance, gravity: &SceneGravity) {
    let accel = gravity.local_accel(&avatar.world_transform.rotation);
    if let Some(ref mut sim) = avatar.cloth_sim {
        sim.gravity = vec3_scale(&accel, sim.gravity_scale);
    }
    for slot in &mut avatar.cloth_overlays {
        slot.sim.gravity = vec3_scale(&accel, slot.sim.gravity_scale);
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

    pub fn step_springs(
        &mut self,
        fixed_dt: f32,
        substeps: u32,
        avatar: &mut AvatarInstance,
        tuning: &spring::SpringTuning,
        gravity: &SceneGravity,
    ) {
        let world_colliders = cloth::resolve_scene_colliders(&self.scene_colliders);
        let dir = gravity.local_dir(&avatar.world_transform.rotation);
        let scale = gravity.spring_power_scale();
        for _ in 0..substeps {
            spring::step_spring_bones(fixed_dt, avatar, &world_colliders, tuning, dir, scale);
        }
    }

    pub fn step_cloth(&mut self, dt: f32, avatar: &mut AvatarInstance, gravity: &SceneGravity) {
        let world_colliders = cloth::resolve_scene_colliders(&self.scene_colliders);
        apply_cloth_gravity(avatar, gravity);
        cloth_solver::step_cloth(dt, avatar, &world_colliders);
    }

    pub fn step_cloth_with_camera_distance(
        &mut self,
        dt: f32,
        avatar: &mut AvatarInstance,
        camera_distance: f32,
        gravity: &SceneGravity,
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
        apply_cloth_gravity(avatar, gravity);
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
        spring_tuning: &spring::SpringTuning,
        gravity: &SceneGravity,
    ) {
        if substeps == 0 {
            return;
        }
        let world_colliders = cloth::resolve_scene_colliders(&self.scene_colliders);
        let spring_dir = gravity.local_dir(&avatar.world_transform.rotation);
        let spring_scale = gravity.spring_power_scale();
        #[cfg(feature = "rapier")]
        let rapier_g = gravity.world_accel();
        for _ in 0..substeps {
            if options.spring_enabled {
                spring::step_spring_bones(
                    fixed_dt,
                    avatar,
                    &world_colliders,
                    spring_tuning,
                    spring_dir,
                    spring_scale,
                );
            }
            if options.cloth_enabled {
                apply_cloth_gravity(avatar, gravity);
                cloth_solver::step_cloth(fixed_dt, avatar, &world_colliders);
            }
            #[cfg(feature = "rapier")]
            if let Some(ref mut rapier) = self.rapier {
                rapier.step(fixed_dt, rapier_g);
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

    /// Step the Rapier pipeline by one tick under `gravity` (world-space
    /// m/s², from the scene gravity).
    pub fn step(&mut self, dt: f32, gravity: Vec3) {
        self.integration_parameters.dt = dt;
        self.physics_pipeline.step(
            &vector![gravity[0], gravity[1], gravity[2]],
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
        assert!(approx(g.local_accel(&[0.0, 0.0, 0.0, 1.0]), [0.0, -9.81, 0.0]));
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
        assert!((la[0].abs() - 9.81).abs() < 1e-3, "expected ~9.81 on X, got {la:?}");
        assert!(la[1].abs() < 1e-3 && la[2].abs() < 1e-3, "off-axis leak: {la:?}");
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
}
