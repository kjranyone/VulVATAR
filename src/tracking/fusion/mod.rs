//! Tracking v2 — observation-fusion body fitter.
//!
//! See `docs/tracking-v2-design.md`. Every observation (2-D keypoints with
//! σ, depth-lifted 3-D points, the D435 point cloud) is fused into one
//! articulated body model by a robust Levenberg–Marquardt MAP fit whose
//! posterior covariance is carried between frames. The output is joint
//! angles + root pose + per-joint σ.

pub mod canonical_face;
pub mod coherence;
#[cfg(feature = "inference")]
mod detector_thread;
pub mod estimator;
#[cfg(feature = "inference")]
pub mod hands;
#[cfg(feature = "inference")]
pub mod head_ori;
pub mod math;
pub mod model;
pub mod observe;
pub mod output;
#[cfg(feature = "inference")]
pub mod provider;
pub mod seed;
pub mod visibility;
