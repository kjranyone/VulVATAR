pub mod cloth;
pub mod cloth_solver;
pub mod spring;

use crate::asset::AvatarAsset;
use crate::avatar::AvatarInstance;

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
        self.accumulator = self.accumulator.min(self.fixed_dt * self.max_substeps as f32);
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

pub struct PhysicsWorld {
    collider_count: usize,
    spring_chain_count: usize,
    #[cfg(feature = "rapier")]
    rapier: Option<RapierWorld>,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            collider_count: 0,
            spring_chain_count: 0,
            #[cfg(feature = "rapier")]
            rapier: None,
        }
    }

    pub fn attach_avatar(&mut self, asset: &AvatarAsset) {
        self.collider_count = asset.colliders.len();
        self.spring_chain_count = asset.spring_bones.len();
        println!(
            "physics: attached avatar with {} collider(s) and {} spring chain(s)",
            self.collider_count, self.spring_chain_count
        );
    }

    pub fn step_springs(&mut self, dt: f32, avatar: &mut AvatarInstance) {
        spring::step_spring_bones(dt, avatar);
    }

    pub fn step_cloth(&mut self, dt: f32, avatar: &mut AvatarInstance) {
        cloth_solver::step_cloth(dt, avatar);
    }

    /// Initialise Rapier world (requires the `rapier` feature).
    #[cfg(feature = "rapier")]
    pub fn init_rapier(&mut self) {
        self.rapier = Some(RapierWorld::new());
        println!("physics: rapier world initialised");
    }

    /// No-op when the rapier feature is not enabled.
    #[cfg(not(feature = "rapier"))]
    pub fn init_rapier(&mut self) {
        // Rapier feature not compiled in; nothing to do.
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

    /// Step the Rapier pipeline by one tick.
    #[allow(dead_code)]
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
        self.query_pipeline
            .update(&self.collider_set);
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
}
