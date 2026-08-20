//! Static keypoint-index → `HumanoidBone` mappings and the model's
//! input-pixel preprocessing constants. Pulled out of `mod.rs` so the
//! 130-line COCO-Wholebody mapping table doesn't bury the inference
//! orchestration logic.
//!
//! Selfie mirror is baked into these tables — every COCO-Wholebody
//! "left" landmark drives the avatar's `Right*` bone and vice versa.
//! See the table headers in this file for the per-section reasoning.


pub(super) const INPUT_W: u32 = 288;
pub(super) const INPUT_H: u32 = 384;

/// ImageNet mean / std on `[0, 255]` RGB. Mirrors rtmlib's preprocessing.
pub(super) const MEAN_RGB: [f32; 3] = [123.675, 116.28, 103.53];
pub(super) const STD_RGB: [f32; 3] = [58.395, 57.12, 57.375];

/// "Detection actually exists" floor. Used only for *structural* gates
/// inside this module — choosing whether hips are visible enough to
/// anchor the source-skeleton origin, whether a hand has enough MCPs
/// for centroid, etc. **Not** a per-joint quality threshold: those
/// belong on the solver (`SolverParams::joint_confidence_threshold`)
/// so the GUI slider actually does something. RTMW3D's SimCC max-bin
/// scores after sigmoid put background / occluded joints near zero,
/// so a pinhole-sized floor is enough to reject "the model emitted
/// something for an invisible point" without hiding real signal from
/// the downstream solver.
pub(super) const KEYPOINT_VISIBILITY_FLOOR: f32 = 0.05;

// ---------------------------------------------------------------------------
// COCO-Wholebody 133 keypoint indices
//
// Layout (rtmlib / mmpose convention):
//   0..=16   body 17        (COCO body)
//   17..=22  foot 6         (big toe / small toe / heel × L,R)
//   23..=90  face 68        (dlib 68-point convention)
//   91..=111 left hand 21   (subject's left)
//   112..=132 right hand 21 (subject's right)
//
// We use selfie mirror: subject's anatomical-left limb drives the
// avatar's `Right*` bones so the on-screen avatar mirrors the user.
// ---------------------------------------------------------------------------

