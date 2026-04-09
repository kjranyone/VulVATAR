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
        let mut substeps = 0u32;
        while self.accumulator >= self.fixed_dt && substeps < self.max_substeps {
            self.accumulator -= self.fixed_dt;
            substeps += 1;
        }
        if self.accumulator > self.fixed_dt * self.max_substeps as f32 {
            self.accumulator = 0.0;
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
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            collider_count: 0,
            spring_chain_count: 0,
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
}
