//! Source skeleton: the avatar-agnostic intermediate representation emitted
//! by the tracking backends and consumed by [`crate::avatar::pose_solver`].
//!
//! A `SourceSkeleton` carries sparse 2D joint positions (in normalised image
//! coordinates with aspect pre-applied) plus a face pose derived from
//! facial keypoints and any expression blend-shape weights. It deliberately
//! does not attempt to compute per-bone rotations — that is the solver's
//! job and depends on the target avatar's rest pose.

use std::collections::HashMap;

use crate::asset::HumanoidBone;

/// One tracked joint in image-space.
///
/// * `position` — normalised camera coords: x in `[-aspect, +aspect]`,
///   y in `[-1, +1]` (y-up). Aspect is already applied so Euclidean distance
///   in x/y is comparable.
/// * `confidence` — detector-reported confidence in `[0, 1]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SourceJoint {
    pub position: [f32; 2],
    pub confidence: f32,
}

/// Head orientation derived from facial keypoints (nose + eyes + ears).
///
/// Angles are in radians with neutral at `yaw = pitch = roll = 0`. Sign
/// conventions are chosen so the values feed straight into
/// [`crate::math_utils::quat_from_euler_ypr`] without a sign flip:
///
/// * **`yaw`** — positive turns the head around +Y toward camera-space +X.
/// * **`pitch`** — positive tilts the head around +X so the top of the
///   head moves toward +Z (i.e. the chin drops — "looking down").
/// * **`roll`** — positive rolls around +Z so the head leans toward +X.
#[derive(Clone, Copy, Debug, Default)]
pub struct FacePose {
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub confidence: f32,
}

/// A named expression blend-shape weight (e.g. `"blink" → 0.6`). Names use the
/// VRM 1.0 canonical identifier space; the avatar-side retarget maps them to
/// matching `ExpressionDef` entries at solve time.
#[derive(Clone, Debug)]
pub struct SourceExpression {
    pub name: String,
    pub weight: f32,
}

/// One tracker sample.
///
/// Body joints are sparse: only the humanoid bones for which the detector
/// produced (or the tracker derived) a 2D position are populated. Face pose,
/// if `Some`, takes precedence over any rotation that might otherwise be
/// computed from the Head/Neck joint pair — the head is driven by facial
/// keypoint geometry, not by the nose-to-shoulder direction.
#[derive(Clone, Debug, Default)]
pub struct SourceSkeleton {
    pub source_timestamp: u64,
    pub joints: HashMap<HumanoidBone, SourceJoint>,
    /// Auxiliary positions for finger *tips* (the keypoint beyond the
    /// `*Distal` bone). Keyed by the distal bone whose tip it represents —
    /// e.g. `fingertips[LeftIndexDistal]` is the 2D position of the left
    /// index fingertip. Used by the solver to drive the distal bone's
    /// orientation; the tip itself is not a humanoid bone.
    pub fingertips: HashMap<HumanoidBone, SourceJoint>,
    pub face: Option<FacePose>,
    pub expressions: Vec<SourceExpression>,
    /// Overall scalar detection confidence (used for stale/quality gating
    /// upstream). `0.0` means "no person detected".
    pub overall_confidence: f32,
}

impl SourceSkeleton {
    pub fn empty(source_timestamp: u64) -> Self {
        Self {
            source_timestamp,
            joints: HashMap::new(),
            fingertips: HashMap::new(),
            face: None,
            expressions: Vec::new(),
            overall_confidence: 0.0,
        }
    }

    /// Insert a joint only if its confidence clears `min_conf`.
    pub fn put_joint(&mut self, bone: HumanoidBone, joint: SourceJoint, min_conf: f32) {
        if joint.confidence >= min_conf {
            self.joints.insert(bone, joint);
        }
    }
}
