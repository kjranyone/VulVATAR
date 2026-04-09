use crate::avatar::AvatarInstance;

pub fn step_spring_bones(dt: f32, avatar: &mut AvatarInstance) {
    let chain_count = avatar.secondary_motion.spring_states.len();
    let _ = dt;
    let _ = chain_count;
}
