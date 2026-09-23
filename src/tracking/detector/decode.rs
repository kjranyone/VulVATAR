//! Shared decoded-keypoint struct. The RTMW3D SimCC decoder that used
//! to live here was removed with the detector migration — YOLO26-pose
//! decodes its own outputs in `yolo26.rs` and fills this struct.

/// One decoded keypoint in normalised model space:
/// `nx, ny ∈ [0, 1]` (image-relative, NY top-to-bottom),
/// `nz ∈ [0, 1]` (model depth axis), `score` is the per-joint
/// confidence after sigmoid of the x/y heatmap peaks.
#[derive(Clone, Copy, Debug, Default)]
pub struct DecodedJoint {
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub score: f32,
    /// Localisation σ of the x / y peak in the same normalised units as
    /// `nx` / `ny` (posterior std of the softmax-normalised SimCC
    /// distribution in a window around the peak, RTMW3D era). A sharp
    /// peak yields ~1 bin; a flat or bimodal distribution yields tens of
    /// bins. This is the per-keypoint uncertainty the fusion estimator
    /// consumes.
    pub sx: f32,
    pub sy: f32,
    /// Peak-shape statistics of the raw SimCC vectors, kept for the
    /// visibility calibration (see `fusion::visibility`). All are
    /// scale-invariant in the logit values.
    ///
    /// Ratio of the strongest response OUTSIDE the main peak's window
    /// to the main peak: ≈0 for a unimodal confident joint, →1 for a
    /// bimodal / flat one.
    pub second_x: f32,
    pub second_y: f32,
    /// Fraction of bins at or above half of the peak value. A nominal
    /// peak covers a few bins; a flat response covers most of the axis.
    pub half_x: f32,
    pub half_y: f32,
    /// Sigmoid of the z-axis peak (the depth head's own confidence).
    /// YOLO26 has no z head; `yolo26.rs` reuses the kpt confidence so
    /// the visibility calibration's zscore term keeps working.
    pub zscore: f32,
}
