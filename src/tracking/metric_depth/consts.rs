//! MoGe-2 ViT-S input geometry.
//!
//! The HF-hosted ONNX export is dynamic-shape, but DirectML recompiles
//! the graph on every distinct input dimension. We lock to one
//! resolution that's both (a) a multiple of the DINOv2 patch size (14)
//! so the encoder doesn't need to internally re-resize, and (b) close
//! to a webcam-friendly 4:3 aspect.
//!
//! 644 × 476 = 14·46 × 14·34 → 1564 image tokens. That's inside MoGe's
//! recommended 1200..3600 token range and roughly the "best
//! quality/speed tradeoff" sweet spot the upstream README cites for
//! webcam-class inputs.

pub const INPUT_W: u32 = 644;
pub const INPUT_H: u32 = 476;

/// Number of ViT image tokens for the chosen input resolution. Passed
/// to the model as the second input — the encoder uses it to compute
/// `(token_rows, token_cols)` and bilinearly resizes accordingly. Must
/// satisfy `token_rows · token_cols ≈ NUM_TOKENS` at the input aspect.
pub(super) const NUM_TOKENS: i64 = (INPUT_W as i64 / 14) * (INPUT_H as i64 / 14);

/// Mask threshold from the upstream `infer()` reference. Pixels with a
/// mask probability below this are out-of-domain (sky, far background,
/// or where the encoder is not confident about geometry).
pub(super) const MASK_THRESHOLD: f32 = 0.5;
