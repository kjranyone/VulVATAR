//! Observation inputs of one frame: pinhole intrinsics, the model-point
//! addressing enum, and the per-channel observation records (2-D / 3-D
//! keypoints, orientation, shoulder yaw) aggregated into `FrameObs`.
//! Everything public is re-exported from the estimator root.

use crate::tracking::fusion::math::*;

/// Pinhole intrinsics of the observing (colour) camera, pixels.
#[derive(Clone, Copy, Debug)]
pub struct Intrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub width: f64,
    pub height: f64,
}

impl Intrinsics {
    #[inline]
    pub fn project(&self, p: V3) -> Option<[f64; 2]> {
        if p[2] <= 0.05 {
            return None;
        }
        Some([
            self.fx * p[0] / p[2] + self.cx,
            self.fy * p[1] / p[2] + self.cy,
        ])
    }
    #[inline]
    pub fn deproject(&self, u: f64, v: f64, z: f64) -> V3 {
        [(u - self.cx) / self.fx * z, (v - self.cy) / self.fy * z, z]
    }
}

/// A model point an observation refers to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModelPoint {
    Joint(usize),
    Site(usize),
    /// A point rigidly attached to `joint` at `local` (metres, joint frame,
    /// already scaled — used for learned face landmarks).
    Attached {
        joint: usize,
        local: V3,
    },
}

/// One 2-D keypoint observation.
#[derive(Clone, Copy, Debug)]
pub struct Kp2d {
    pub point: ModelPoint,
    pub u: f64,
    pub v: f64,
    /// Isotropic pixel σ.
    pub sigma: f64,
}

/// One 3-D point observation (camera frame, metres).
#[derive(Clone, Copy, Debug)]
pub struct Kp3d {
    pub point: ModelPoint,
    pub p: V3,
    pub sigma: f64,
    /// Multiplier on the LATERAL (x/y, ≈ across-ray) σ. 1.0 = isotropic.
    /// Depth-lifted landmark points carry an honest z (the sensor) but
    /// lateral coordinates inherited from the 2-D landmark — when that
    /// landmark systematically frontalizes (dense face mesh), the lateral
    /// part must not outvote real orientation evidence.
    pub lat_scale: f64,
}

/// A direct orientation observation of one joint's WORLD (camera-frame)
/// rotation — e.g. the FaceMesh-derived head pose. Residual is the SO(3)
/// log of `R_world(joint) · R_targetᵀ`, whitened by `sigma` (rad).
#[derive(Clone, Copy, Debug)]
pub struct OriObs {
    pub joint: usize,
    pub target: M3,
    pub sigma: f64,
    /// How many joints up the kinematic chain (starting at `joint`) this
    /// observation is allowed to move. A head-pose obs with `depth: 2`
    /// adjusts head+neck only — a saturated / wrong face pose can then
    /// never twist the torso, whose orientation is owned by the body
    /// keypoints. `usize::MAX` = whole chain incl. root.
    pub chain_depth: usize,
}

/// Torso-yaw observation: the angle of the shoulder line in the camera
/// x/z plane, measured from the chest's depth slope. This is the only
/// channel that carries torso yaw from depth — a torso CAPSULE is
/// rotationally symmetric about its own axis, so the surface term is
/// blind to yaw by construction, leaving the 2-D shoulder pixels alone
/// to fix it (measured +17° of over-rotation against the depth
/// reference on a live desk session).
#[derive(Clone, Copy, Debug)]
pub struct ShoulderYawObs {
    /// Joint whose position is the LEFT end of the line.
    pub left: usize,
    /// … and the right end.
    pub right: usize,
    /// atan2(Δz_cam, Δx_cam) of the measured line, radians.
    pub yaw: f64,
    pub sigma: f64,
}

/// Everything observed at one capture time.
#[derive(Clone, Debug, Default)]
pub struct FrameObs {
    /// Capture time, seconds (device clock).
    pub t: f64,
    pub intr: Option<Intrinsics>,
    pub kp2d: Vec<Kp2d>,
    pub kp3d: Vec<Kp3d>,
    /// Direct world-orientation observations (see [`OriObs`]).
    pub ori: Vec<OriObs>,
    /// Torso yaw from the chest depth slope (see [`ShoulderYawObs`]).
    pub shoulder_yaw: Option<ShoulderYawObs>,
    /// Coarse torso reference (camera metres, e.g. shoulder-mid from
    /// depth) used only to seed the root on (re)acquisition.
    pub torso_hint: Option<V3>,
    /// Sparse surface points (camera metres, σ): depth samples under body
    /// keypoints, interpreted as "some body surface is here" (z-buffer
    /// semantics — an occluding arm in front of a shoulder is evidence for
    /// the arm, not a wrong shoulder). Associated to the nearest capsule of
    /// any part.
    pub surface: Vec<(V3, f64)>,
    /// Per-capsule association permission for `surface` (empty = all).
    /// Surface evidence may only be claimed by parts that are observed:
    /// the trunk and head always, a limb capsule only while its distal
    /// joint carries data. A point whose nearest capsule is a forbidden
    /// one is DROPPED, not re-assigned — an unobserved arm hanging at the
    /// side still occludes the trunk there. Without this, free
    /// (unobserved) arm capsules slide onto the chest and explain the
    /// torso surface at zero cost (measured: 1 400 of 4 800 points on the
    /// upper arms, trunk yaw 27° off).
    pub surf_allow: Vec<bool>,
}
