//! Static keypoint-index → `HumanoidBone` mappings and the model's
//! input-pixel preprocessing constants. Pulled out of `mod.rs` so the
//! 130-line COCO-Wholebody mapping table doesn't bury the inference
//! orchestration logic.
//!
//! Selfie mirror is baked into these tables — every COCO-Wholebody
//! "left" landmark drives the avatar's `Right*` bone and vice versa.
//! See the table headers in this file for the per-section reasoning.

use crate::asset::HumanoidBone;

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

pub(super) const COCO_BODY: &[(usize, HumanoidBone)] = &[
    // (index, avatar bone). Subject's left → avatar Right.
    //
    // Wrist slots (`LeftHand`, `RightHand`) are *not* listed here.
    // The body track and the dedicated hand track within RTMW3D
    // disagree about wrist position by up to ~0.4 in normalised z
    // when an object occludes the wrist (e.g. the body track
    // hallucinated a guitar-strumming wrist behind the torso while
    // the hand track placed it correctly at chest level). The hand
    // track is trained on hand-specific data and is more reliable in
    // those scenarios, so we drive `LeftHand` / `RightHand` from
    // the hand subset's wrist keypoint (index 91 / 112) inside
    // `attach_hand` below.
    (5, HumanoidBone::RightShoulder),
    (5, HumanoidBone::RightUpperArm),
    (6, HumanoidBone::LeftShoulder),
    (6, HumanoidBone::LeftUpperArm),
    (7, HumanoidBone::RightLowerArm),
    (8, HumanoidBone::LeftLowerArm),
    (11, HumanoidBone::RightUpperLeg),
    (12, HumanoidBone::LeftUpperLeg),
    (13, HumanoidBone::RightLowerLeg),
    (14, HumanoidBone::LeftLowerLeg),
    (15, HumanoidBone::RightFoot),
    (16, HumanoidBone::LeftFoot),
];

/// Big-toe landmarks → fingertips slot. COCO 17 = left big toe, 20 =
/// right big toe (subject's frame). Selfie mirror.
pub(super) const COCO_FOOT_TIPS: &[(usize, HumanoidBone)] = &[
    (17, HumanoidBone::RightFoot), // subject left big toe → avatar Right foot tip
    (20, HumanoidBone::LeftFoot),
];

/// MediaPipe-compatible hand landmark indices (within a single hand):
///   0 wrist
///   1..=4  thumb  (CMC, MCP, IP, TIP)
///   5..=8  index  (MCP, PIP, DIP, TIP)
///   9..=12 middle
///   13..=16 ring
///   17..=20 pinky
///
/// `HAND_PHALANGES` lists the 15 phalanx positions; `HAND_TIPS` lists
/// the five tips that go into `SourceSkeleton::fingertips` keyed by the
/// distal bone whose tip they represent.
///
/// All bones below are the **avatar Right** chain — apply selfie mirror
/// at the call site to get Left.
pub(super) const HAND_PHALANGES_RIGHT: &[(usize, HumanoidBone)] = &[
    (1, HumanoidBone::RightThumbProximal),
    (2, HumanoidBone::RightThumbIntermediate),
    (3, HumanoidBone::RightThumbDistal),
    (5, HumanoidBone::RightIndexProximal),
    (6, HumanoidBone::RightIndexIntermediate),
    (7, HumanoidBone::RightIndexDistal),
    (9, HumanoidBone::RightMiddleProximal),
    (10, HumanoidBone::RightMiddleIntermediate),
    (11, HumanoidBone::RightMiddleDistal),
    (13, HumanoidBone::RightRingProximal),
    (14, HumanoidBone::RightRingIntermediate),
    (15, HumanoidBone::RightRingDistal),
    (17, HumanoidBone::RightLittleProximal),
    (18, HumanoidBone::RightLittleIntermediate),
    (19, HumanoidBone::RightLittleDistal),
];
pub(super) const HAND_TIPS_RIGHT: &[(usize, HumanoidBone)] = &[
    (4, HumanoidBone::RightThumbDistal),
    (8, HumanoidBone::RightIndexDistal),
    (12, HumanoidBone::RightMiddleDistal),
    (16, HumanoidBone::RightRingDistal),
    (20, HumanoidBone::RightLittleDistal),
];
pub(super) const HAND_PHALANGES_LEFT: &[(usize, HumanoidBone)] = &[
    (1, HumanoidBone::LeftThumbProximal),
    (2, HumanoidBone::LeftThumbIntermediate),
    (3, HumanoidBone::LeftThumbDistal),
    (5, HumanoidBone::LeftIndexProximal),
    (6, HumanoidBone::LeftIndexIntermediate),
    (7, HumanoidBone::LeftIndexDistal),
    (9, HumanoidBone::LeftMiddleProximal),
    (10, HumanoidBone::LeftMiddleIntermediate),
    (11, HumanoidBone::LeftMiddleDistal),
    (13, HumanoidBone::LeftRingProximal),
    (14, HumanoidBone::LeftRingIntermediate),
    (15, HumanoidBone::LeftRingDistal),
    (17, HumanoidBone::LeftLittleProximal),
    (18, HumanoidBone::LeftLittleIntermediate),
    (19, HumanoidBone::LeftLittleDistal),
];
pub(super) const HAND_TIPS_LEFT: &[(usize, HumanoidBone)] = &[
    (4, HumanoidBone::LeftThumbDistal),
    (8, HumanoidBone::LeftIndexDistal),
    (12, HumanoidBone::LeftMiddleDistal),
    (16, HumanoidBone::LeftRingDistal),
    (20, HumanoidBone::LeftLittleDistal),
];
