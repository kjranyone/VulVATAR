//! MoGe-2 metric monocular geometry inference.
//!
//! Used by [`super::cigpose_metric_depth`] to obtain a per-pixel
//! metric point cloud at every frame, which the provider then samples
//! at each CIGPose 2D keypoint to lift the joint into 3D.
//!
//! ## Pipeline
//!
//! 1. **Resize** source RGB to a fixed 644×476 — multiples of the
//!    DINOv2 patch size (14) so the encoder doesn't internally
//!    re-resize. Fixed shape avoids DirectML per-shape recompiles.
//! 2. **Run** the model with two inputs: `image` (1, 3, 476, 644) in
//!    `[0, 1]` and `num_tokens` int64 scalar (= `(644/14)·(476/14) =
//!    1564`).
//! 3. **Extract** four outputs: `points` (1, 476, 644, 3) BHWC,
//!    `normal` (auxiliary, ignored here), `mask` (1, 476, 644)
//!    sigmoid in `[0, 1]`, `metric_scale` (1,) scalar.
//! 4. **Postprocess**: `points * metric_scale` → metres, mask < 0.5
//!    pixels → NaN. Skip the upstream `recover_focal_shift` 2-DoF
//!    Levenberg-Marquardt optimisation — its absolute-z correction is
//!    invisible after the provider re-anchors every joint to hip mid.
//!
//! Output is owned by [`MoGeFrame`] in MoGe's input pixel grid; the
//! provider remaps CIGPose 2D keypoints from frame coords into this
//! grid before sampling.

#[cfg(feature = "inference")]
mod consts;
#[cfg(feature = "inference")]
mod decode;
#[cfg(feature = "inference")]
mod preprocess;

#[cfg(feature = "inference")]
pub use consts::{INPUT_H, INPUT_W};
#[cfg(feature = "inference")]
pub use decode::MoGeFrame;

#[cfg(feature = "inference")]
use log::{error, info, warn};
#[cfg(feature = "inference")]
use ndarray::{Array, IxDyn};
#[cfg(feature = "inference")]
use ort::session::Session;
#[cfg(feature = "inference")]
use ort::value::TensorRef;
#[cfg(feature = "inference")]
use std::path::Path;

#[cfg(feature = "inference")]
use super::rtmw3d::{build_session, InferenceBackend};

#[cfg(feature = "inference")]
use consts::NUM_TOKENS;

/// Names of the four output tensors as declared by the
/// `Ruicheng/moge-2-vits-normal-onnx` export: `points`, `normal`,
/// `mask`, `scale`. The upstream PyTorch `forward()` calls the metric
/// scalar `metric_scale`, but the published ONNX file shortens it to
/// `scale`. We name-match first; if the file ever ships under
/// different names the positional fallback (first 4 outputs in
/// declaration order) keeps the code working.
#[cfg(feature = "inference")]
const OUTPUT_NAME_POINTS: &str = "points";
#[cfg(feature = "inference")]
const OUTPUT_NAME_MASK: &str = "mask";
#[cfg(feature = "inference")]
const OUTPUT_NAME_METRIC_SCALE: &str = "scale";

#[cfg(feature = "inference")]
pub struct MoGe2Inference {
    session: Session,
    image_input_name: String,
    num_tokens_input_name: Option<String>,
    /// Resolved output tensor names (`points`, `mask`, `metric_scale`)
    /// — looked up by the canonical name first, falling back to
    /// position 0 / 2 / 3 if the model's name table differs from the
    /// upstream export. (`normal` is skipped; we don't consume it.)
    output_points: String,
    output_mask: String,
    output_metric_scale: String,
    backend: InferenceBackend,
}

#[cfg(feature = "inference")]
impl MoGe2Inference {
    pub fn from_model_path(model_path: impl AsRef<Path>) -> Result<Self, String> {
        let path = model_path.as_ref();
        if !path.is_file() {
            return Err(format!(
                "MoGe-2 model not found at {}. Run dev.ps1 setup (experimental flow) to fetch it.",
                path.display()
            ));
        }
        info!("Loading MoGe-2 from {}", path.display());

        let path_str = path.to_string_lossy().into_owned();
        let (session, backend) = build_session(&path_str, 4, "MoGe-2")?;

        let inputs: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        if inputs.is_empty() {
            return Err("MoGe-2 model declares no inputs".to_string());
        }
        let image_input_name = inputs[0].clone();
        let num_tokens_input_name = inputs.get(1).cloned();
        info!(
            "MoGe-2 I/O: inputs={:?}, image='{}', num_tokens={:?}",
            inputs, image_input_name, num_tokens_input_name
        );

        let outputs: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        if outputs.len() < 4 {
            return Err(format!(
                "MoGe-2 model declares {} outputs, expected at least 4 (points, normal, mask, metric_scale)",
                outputs.len()
            ));
        }
        // Resolve by canonical name; if absent (custom export), fall
        // back to declaration order from the upstream export script:
        // [points, normal, mask, metric_scale].
        let output_points = outputs
            .iter()
            .find(|n| n.as_str() == OUTPUT_NAME_POINTS)
            .cloned()
            .unwrap_or_else(|| outputs[0].clone());
        let output_mask = outputs
            .iter()
            .find(|n| n.as_str() == OUTPUT_NAME_MASK)
            .cloned()
            .unwrap_or_else(|| outputs[2].clone());
        let output_metric_scale = outputs
            .iter()
            .find(|n| n.as_str() == OUTPUT_NAME_METRIC_SCALE)
            .cloned()
            .unwrap_or_else(|| outputs[3].clone());
        info!(
            "MoGe-2 outputs: points='{}', mask='{}', metric_scale='{}' (declared {:?})",
            output_points, output_mask, output_metric_scale, outputs
        );

        Ok(Self {
            session,
            image_input_name,
            num_tokens_input_name,
            output_points,
            output_mask,
            output_metric_scale,
            backend,
        })
    }

    pub fn backend(&self) -> &InferenceBackend {
        &self.backend
    }

    /// Run inference on a whole-frame RGB buffer. The returned
    /// [`MoGeFrame`] is in MoGe's input pixel grid (`INPUT_W ×
    /// INPUT_H`).
    pub fn estimate(&mut self, rgb: &[u8], src_w: u32, src_h: u32) -> Option<MoGeFrame> {
        let expected = (src_w as usize)
            .saturating_mul(src_h as usize)
            .saturating_mul(3);
        if rgb.len() < expected || src_w == 0 || src_h == 0 {
            return None;
        }
        let image_tensor = preprocess::preprocess(rgb, src_w, src_h);
        let image = match TensorRef::from_array_view(&image_tensor) {
            Ok(v) => v,
            Err(e) => {
                error!("MoGe-2: image TensorRef creation failed: {}", e);
                return None;
            }
        };

        // num_tokens: 0-d int64 scalar matching the upstream export's
        // `torch.tensor(num_tokens, dtype=torch.int64)` shape. ndarray's
        // dynamic-rank zero-d view encodes that as `IxDyn(&[])`.
        let num_tokens_arr = Array::<i64, _>::from_shape_vec(IxDyn(&[]), vec![NUM_TOKENS])
            .expect("0-d Array from single-element vec");
        let num_tokens = match TensorRef::from_array_view(&num_tokens_arr) {
            Ok(v) => v,
            Err(e) => {
                error!("MoGe-2: num_tokens TensorRef creation failed: {}", e);
                return None;
            }
        };

        let outputs_result = match self.num_tokens_input_name.as_deref() {
            Some(n) => self.session.run(ort::inputs![
                self.image_input_name.as_str() => image,
                n => num_tokens,
            ]),
            None => self
                .session
                .run(ort::inputs![self.image_input_name.as_str() => image]),
        };
        let outputs = match outputs_result {
            Ok(o) => o,
            Err(e) => {
                error!("MoGe-2: ONNX run failed: {}", e);
                return None;
            }
        };

        let pixel_count = (INPUT_W as usize) * (INPUT_H as usize);
        let points = match outputs.get(self.output_points.as_str()) {
            Some(v) => match v.try_extract_tensor::<f32>() {
                Ok((_, data)) if data.len() >= pixel_count * 3 => data.to_vec(),
                Ok((_, data)) => {
                    error!(
                        "MoGe-2: 'points' tensor too small ({} < {})",
                        data.len(),
                        pixel_count * 3
                    );
                    return None;
                }
                Err(e) => {
                    error!("MoGe-2: 'points' extract failed: {}", e);
                    return None;
                }
            },
            None => {
                error!("MoGe-2: 'points' output missing");
                return None;
            }
        };
        let mask = match outputs.get(self.output_mask.as_str()) {
            Some(v) => match v.try_extract_tensor::<f32>() {
                Ok((_, data)) if data.len() >= pixel_count => data.to_vec(),
                Ok((_, data)) => {
                    error!(
                        "MoGe-2: 'mask' tensor too small ({} < {})",
                        data.len(),
                        pixel_count
                    );
                    return None;
                }
                Err(e) => {
                    error!("MoGe-2: 'mask' extract failed: {}", e);
                    return None;
                }
            },
            None => {
                error!("MoGe-2: 'mask' output missing");
                return None;
            }
        };
        let metric_scale = match outputs.get(self.output_metric_scale.as_str()) {
            Some(v) => match v.try_extract_tensor::<f32>() {
                Ok((_, data)) if !data.is_empty() => data[0],
                Ok(_) => {
                    error!("MoGe-2: 'metric_scale' tensor empty");
                    return None;
                }
                Err(e) => {
                    error!("MoGe-2: 'metric_scale' extract failed: {}", e);
                    return None;
                }
            },
            None => {
                error!("MoGe-2: 'metric_scale' output missing");
                return None;
            }
        };
        drop(outputs);

        let frame = decode::build_frame(&points, &mask, metric_scale, INPUT_W, INPUT_H);
        if frame.is_none() {
            warn!(
                "MoGe-2: postprocess rejected outputs (scale={})",
                metric_scale
            );
        }
        frame
    }
}
