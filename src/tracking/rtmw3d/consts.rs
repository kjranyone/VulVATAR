//! RTMW3D input-pixel constants and the structural keypoint-visibility
//! floor used by the face cascade.

pub(super) const INPUT_W: u32 = 288;
pub(super) const INPUT_H: u32 = 384;

/// ImageNet mean / std on `[0, 255]` RGB. Mirrors rtmlib's preprocessing.
pub(super) const MEAN_RGB: [f32; 3] = [123.675, 116.28, 103.53];
pub(super) const STD_RGB: [f32; 3] = [58.395, 57.12, 57.375];

/// "Detection actually exists" floor. Used for structural gates in the
/// face cascade (body-face keypoints present, bbox from face-68).
/// RTMW3D's SimCC max-bin scores after sigmoid put background /
/// occluded joints near zero, so a pinhole-sized floor rejects
/// "the model emitted something for an invisible point" without
/// hiding real signal from the fusion estimator.
pub(super) const KEYPOINT_VISIBILITY_FLOOR: f32 = 0.05;

// COCO-Wholebody 133 layout (rtmlib / mmpose): 0..=16 body, 17..=22
// foot, 23..=90 face-68, 91..=111 left hand, 112..=132 right hand.

