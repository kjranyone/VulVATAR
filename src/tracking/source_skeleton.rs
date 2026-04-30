//! Source skeleton: the avatar-agnostic intermediate representation emitted
//! by the tracking backends and consumed by [`crate::avatar::pose_solver`].
//!
//! A `SourceSkeleton` carries sparse 3D joint positions (image-space x/y
//! plus a depth z relative to the body anchor), per-side hand landmark
//! arrays, and either a face pose or ARKit-compatible blendshape weights.
//! It deliberately does not attempt to compute per-bone rotations — that
//! is the solver's job and depends on the target avatar's rest pose.
//!
//! ## Coordinate convention
//!
//! * `x` ∈ `[-aspect, +aspect]` (camera-space, aspect pre-applied)
//! * `y` ∈ `[-1, +1]`, Y-up (image bottom is `-1`, top is `+1`)
//! * `z` ∈ depth in the same units as x/y, with `+z` pointing *toward the
//!   camera*. The body's hip midpoint is the depth origin (`z = 0`); a
//!   joint reaching toward the lens has `z > 0`.
//!
//! With a 3D-native backend (RTMW3D whole-body) the solver consumes
//! the depth directly — no foreshortening reconstruction needed.
//! Backends that only output 2D must populate `z = 0` and accept that
//! the solver cannot disambiguate forward / backward limbs.

use std::collections::HashMap;

use crate::asset::HumanoidBone;

/// One tracked joint in image-space, with depth.
///
/// * `position` — `[x, y, z]` in normalised camera coords (see module
///   docs for axis conventions). For backends that only produce 2D
///   keypoints, set `z = 0`.
/// * `confidence` — detector-reported confidence in `[0, 1]`. For
///   SimCC-style outputs (RTMW3D) this is the sigmoid of the heatmap
///   peak, so a single threshold gates "joint actually in frame and
///   well-localised".
#[derive(Clone, Copy, Debug, Default)]
pub struct SourceJoint {
    pub position: [f32; 3],
    pub confidence: f32,
}

/// Head orientation derived from facial keypoints.
///
/// Angles are in radians with neutral at `yaw = pitch = roll = 0`. Sign
/// conventions are chosen so the values feed straight into
/// [`crate::math_utils::quat_from_euler_ypr`] without a sign flip:
///
/// * **`yaw`** — positive turns the head around +Y toward camera-space +X.
/// * **`pitch`** — positive tilts the head around +X so the top of the
///   head moves toward +Z (i.e. the chin drops — "looking down").
/// * **`roll`** — positive rolls around +Z so the head leans toward +X.
///
/// The face track is currently empty — RTMW3D emits 68 face landmarks
/// (indices 23-90) but they are not yet decoded into yaw/pitch/roll;
/// the head bone falls back to spine direction. Phase C (SMIRK) will
/// drive this field along with ARKit blendshape weights.
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

/// Full orientation of a tracked hand. `forward` is the wrist→fingers
/// axis (typically wrist → middle MCP) and `up` is the palm normal
/// (away from the back of the hand, toward where the fingernails
/// face when curled up). Both are unit vectors in source-skeleton
/// coords. The pair is computed by the tracker from the four MCP
/// landmarks and used by the solver to drive the wrist bone's full
/// 3-DoF rotation — without `up`, finger MCPs end up with arbitrary
/// 90° twist because the wrist→tip direction alone leaves the
/// twist axis under-constrained.
#[derive(Clone, Copy, Debug)]
pub struct HandOrientation {
    pub forward: [f32; 3],
    pub up: [f32; 3],
    pub confidence: f32,
}

/// One tracker sample.
///
/// Body joints are sparse: only the humanoid bones for which the detector
/// produced (or the tracker derived) a 3D position are populated. Face pose,
/// if `Some`, takes precedence over any rotation that might otherwise be
/// computed from the Head/Neck joint pair — the head is driven by facial
/// keypoint geometry, not by the nose-to-shoulder direction.
///
/// Hand finger joints (thumb/index/middle/ring/little × proximal/
/// intermediate/distal) live in `joints` alongside body bones once a
/// hand landmarker has run; the auxiliary `fingertips` map carries the
/// tip beyond each distal bone, since the tip itself does not have a
/// humanoid bone slot.
#[derive(Clone, Debug, Default)]
pub struct SourceSkeleton {
    pub source_timestamp: u64,
    pub joints: HashMap<HumanoidBone, SourceJoint>,
    /// Auxiliary positions for finger *tips* (the keypoint beyond the
    /// `*Distal` bone). Keyed by the distal bone whose tip it represents —
    /// e.g. `fingertips[LeftIndexDistal]` is the 3D position of the left
    /// index fingertip. Used by the solver to drive the distal bone's
    /// orientation; the tip itself is not a humanoid bone. The same slot
    /// is reused for foot toe tips (`fingertips[LeftFoot]`).
    pub fingertips: HashMap<HumanoidBone, SourceJoint>,
    pub face: Option<FacePose>,
    pub expressions: Vec<SourceExpression>,
    /// Full 3-DoF orientation of the left hand (wrist) when the hand
    /// track produced enough MCP landmarks to define a palm plane.
    /// `None` when the hand is not tracked or the MCPs are too
    /// degenerate to extract a palm normal.
    pub left_hand_orientation: Option<HandOrientation>,
    pub right_hand_orientation: Option<HandOrientation>,
    /// Optional FaceMesh-model "is this a face" confidence (post sigmoid)
    /// for the frame's face crop. Distinct from `face.confidence`, which
    /// is body-derived (min over the 5 COCO face landmarks): this one
    /// is the dedicated face-mesh model's own output and is what the
    /// solver gates `expressions` on via `face_confidence_threshold`.
    /// `None` when no face mesh ran (no face crop, no model).
    pub face_mesh_confidence: Option<f32>,
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
            left_hand_orientation: None,
            right_hand_orientation: None,
            face_mesh_confidence: None,
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
