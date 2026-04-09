use crate::avatar::AvatarInstance;

pub fn step_cloth(dt: f32, avatar: &mut AvatarInstance) {
    if !avatar.cloth_enabled {
        return;
    }
    let _ = dt;
}
