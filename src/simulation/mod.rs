pub mod cloth;

use crate::avatar::AvatarInstance;

pub struct PhysicsWorld {
    collider_count: usize,
    cloth_region_count: usize,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            collider_count: 0,
            cloth_region_count: 0,
        }
    }

    pub fn attach_avatar(&mut self, avatar: &AvatarInstance) {
        self.collider_count = avatar.asset.colliders.len();
        self.cloth_region_count = avatar.cloth_state.regions.len();
        println!(
            "physics: attached avatar '{}' with {} collider(s) and {} cloth region(s)",
            avatar.asset.name, self.collider_count, self.cloth_region_count
        );
    }

    pub fn step(&mut self, dt_seconds: f32, avatar: &mut AvatarInstance) {
        println!(
            "physics: step {:.4} sec, {} spring chain(s), {} cloth region(s), {} collider(s)",
            dt_seconds,
            avatar.spring_state.chain_states.len(),
            avatar.cloth_state.regions.len(),
            self.collider_count
        );
    }
}
