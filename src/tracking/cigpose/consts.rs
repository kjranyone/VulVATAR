//! CIGPose preprocessing constants.
//!
//! Mirrors RTMW3D's input contract: 288×384 NCHW float32, ImageNet
//! mean/std on 0..255 RGB. Kept separate from `rtmw3d::consts` so the
//! two providers can diverge (e.g. if a future CIGPose checkpoint
//! ships at a different resolution) without coupling.

pub(super) const INPUT_W: u32 = 288;
pub(super) const INPUT_H: u32 = 384;

pub(super) const MEAN_RGB: [f32; 3] = [123.675, 116.28, 103.53];
pub(super) const STD_RGB: [f32; 3] = [58.395, 57.12, 57.375];
