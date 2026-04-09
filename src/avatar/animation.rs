use crate::asset::AnimationClipId;

#[derive(Clone, Debug)]
pub struct AnimationState {
    pub active_clip: Option<AnimationClipId>,
    pub playhead_seconds: f32,
    pub speed: f32,
    pub looping: bool,
}

impl Default for AnimationState {
    fn default() -> Self {
        Self {
            active_clip: None,
            playhead_seconds: 0.0,
            speed: 1.0,
            looping: true,
        }
    }
}
