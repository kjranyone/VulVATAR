//! ViTPose-Wholebody preprocessing constants.
//!
//! ViTPose-S accepts 192×256 NCHW float32, ImageNet mean/std on
//! `[0, 255]` RGB. The 3:4 aspect mirrors RTMW3D and CIGPose, so the
//! shared `pad_and_clamp_bbox` (`288:384`) produces a crop whose
//! aspect is identical to ViTPose's target — no extra letterboxing
//! needed beyond the bilinear squash already done by the preprocess
//! pass.

pub(super) const INPUT_W: u32 = 192;
pub(super) const INPUT_H: u32 = 256;

pub(super) const MEAN_RGB: [f32; 3] = [123.675, 116.28, 103.53];
pub(super) const STD_RGB: [f32; 3] = [58.395, 57.12, 57.375];
