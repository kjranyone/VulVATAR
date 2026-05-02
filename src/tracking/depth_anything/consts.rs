//! Depth Anything V2 Small input geometry.
//!
//! The HF-hosted ONNX is fully dynamic-shape, but DirectML recompiles
//! the graph per-shape so we lock to one resolution. Constraints:
//!
//! * Multiples of 14 (DPT / DINOv2 patch size).
//! * Aspect close to typical webcam 4:3 (subject framing) without
//!   incurring the full 518² square cost on every frame.
//!
//! 392 × 294 = 14·28 × 14·21 = 588 image tokens. The token count is
//! quadratic in self-attention cost — at 588 vs 1036 we expect ~57% of
//! the inference time of the 518×392 default. Calibration values
//! (`d_raw`) are scale-invariant under resolution so the body-anchor
//! solve is unaffected; only sub-keypoint depth detail degrades, which
//! is fine because [`super::DepthAnythingFrame::sample_inverse_depth`]
//! median-filters a small window anyway.
//!
//! Smaller candidates (already validated to be valid DPT shapes):
//!
//! | W×H      | tokens | rel. cost |
//! |----------|--------|-----------|
//! | 518×392  | 1036   | 100%      |
//! | 392×294  |  588   |  57%      | ← current
//! | 308×224  |  352   |  34%      |
//! | 252×196  |  252   |  24%      |
//!
//! Output spatial dims match input.

pub const INPUT_W: u32 = 392;
pub const INPUT_H: u32 = 294;

/// ImageNet stats applied to `[0, 1]`-scaled RGB. DAv2's preprocessor
/// rescales 1/255 *then* normalises with these — we fold both into a
/// single `(rgb / 255 - mean) / std` step in `preprocess`.
pub(super) const MEAN_RGB: [f32; 3] = [0.485, 0.456, 0.406];
pub(super) const STD_RGB: [f32; 3] = [0.229, 0.224, 0.225];
