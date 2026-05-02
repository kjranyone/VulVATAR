//! CIGPose 2D whole-body pose inference.
//!
//! Sibling to RTMW3D, used by the experimental
//! [`super::cigpose_metric_depth`] provider. Unlike RTMW3D, CIGPose only
//! emits 2D keypoints — depth comes from a separate MoGe metric-depth
//! pass and the two are stitched together by the provider.
//!
//! ## Pipeline
//!
//! 1. **Resize / normalise** the source frame to 288×384 NCHW with
//!    ImageNet mean/std (same constants as RTMW3D).
//! 2. **Infer**: two SimCC outputs (X: `(1, 133, 288·s)`, Y: `(1, 133,
//!    384·s)`) where `s = split_ratio` from the model's
//!    `cigpose_meta` ONNX custom-metadata field (default 2.0 →
//!    bin counts 576 / 768, identical to RTMW3D's X/Y outputs).
//! 3. **Decode**: argmax X / Y, score = `min(max_x, max_y)` raw
//!    logit (passed through sigmoid here so the downstream visibility
//!    floor uses the same `[0, 1]` scale as RTMW3D).
//!
//! Whole-frame inference (no YOLOX crop). The CIGPose+MetricDepth
//! provider feeds the same source frame to MoGe so they share a pixel
//! coordinate system, and inserting a YOLOX crop only on the CIGPose
//! side would force a coord transform between 2D keypoints and the
//! depth point cloud for every joint sample. Keep them aligned;
//! revisit if accuracy on small / off-centre subjects becomes an
//! issue.

#[cfg(feature = "inference")]
mod consts;
#[cfg(feature = "inference")]
mod decode;
#[cfg(feature = "inference")]
mod preprocess;

#[cfg(feature = "inference")]
pub use decode::{DecodedJoint2d, NUM_JOINTS};

#[cfg(feature = "inference")]
use log::{error, info, warn};
#[cfg(feature = "inference")]
use ort::session::Session;
#[cfg(feature = "inference")]
use ort::value::TensorRef;
#[cfg(feature = "inference")]
use std::path::Path;

#[cfg(feature = "inference")]
use super::rtmw3d::{build_session, InferenceBackend};

#[cfg(feature = "inference")]
pub struct CigposeInference {
    session: Session,
    input_name: String,
    /// (X, Y) tensor names from the model. Order is the runtime output
    /// order; the model declares them as `simcc_x`, `simcc_y` but we
    /// don't rely on the literal names — we take the first two outputs.
    output_names: [String; 2],
    /// SimCC bin counts derived from `split_ratio` in the model's
    /// `cigpose_meta` custom metadata. Pre-cached so per-frame decode
    /// can slice without re-reading the metadata each call.
    simcc_x_bins: usize,
    simcc_y_bins: usize,
    backend: InferenceBackend,
}

#[cfg(feature = "inference")]
impl CigposeInference {
    pub fn from_model_path(model_path: impl AsRef<Path>) -> Result<Self, String> {
        let path = model_path.as_ref();
        if !path.is_file() {
            return Err(format!(
                "CIGPose model not found at {}. Run dev.ps1 setup (experimental flow) to fetch it.",
                path.display()
            ));
        }
        info!("Loading CIGPose from {}", path.display());

        let path_str = path.to_string_lossy().into_owned();
        let (session, backend) = build_session(&path_str, 4, "CIGPose")?;

        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input".to_string());
        let outputs: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        if outputs.len() < 2 {
            return Err(format!(
                "CIGPose model declares {} outputs, expected at least 2 (SimCC X/Y)",
                outputs.len()
            ));
        }
        let output_names = [outputs[0].clone(), outputs[1].clone()];
        info!(
            "CIGPose I/O: input='{}', outputs={:?}",
            input_name, output_names
        );

        // split_ratio: read from the model's custom metadata. Default
        // 2.0 matches mmpose's SimCC export default; if the metadata
        // is missing or unparseable we fall through to that.
        let split_ratio = read_split_ratio_from_metadata(&session).unwrap_or(2.0);
        if (split_ratio - 2.0).abs() > 1e-3 {
            info!(
                "CIGPose split_ratio = {} (from cigpose_meta metadata)",
                split_ratio
            );
        }
        let simcc_x_bins = (consts::INPUT_W as f32 * split_ratio).round() as usize;
        let simcc_y_bins = (consts::INPUT_H as f32 * split_ratio).round() as usize;

        Ok(Self {
            session,
            input_name,
            output_names,
            simcc_x_bins,
            simcc_y_bins,
            backend,
        })
    }

    pub fn backend(&self) -> &InferenceBackend {
        &self.backend
    }

    /// Run inference on a whole-frame RGB buffer and return one
    /// decoded 2D joint per COCO-Wholebody index. Keypoint coordinates
    /// are normalised image-relative (`nx, ny ∈ [0, 1]`).
    pub fn estimate(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> Vec<DecodedJoint2d> {
        let expected = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(3);
        if rgb.len() < expected || width == 0 || height == 0 {
            return Vec::new();
        }
        let tensor = preprocess::preprocess(rgb, width, height);
        let input = match TensorRef::from_array_view(&tensor) {
            Ok(v) => v,
            Err(e) => {
                error!("CIGPose: TensorRef creation failed: {}", e);
                return Vec::new();
            }
        };

        let x_name = self.output_names[0].clone();
        let y_name = self.output_names[1].clone();
        let outputs = match self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
        {
            Ok(o) => o,
            Err(e) => {
                error!("CIGPose: ONNX run failed: {}", e);
                return Vec::new();
            }
        };

        let extract = |name: &str, expected_len: usize| -> Option<Vec<f32>> {
            let v = outputs.get(name)?;
            let (_, data) = v.try_extract_tensor::<f32>().ok()?;
            if data.len() < expected_len {
                None
            } else {
                Some(data.to_vec())
            }
        };

        let simcc_x = match extract(&x_name, NUM_JOINTS * self.simcc_x_bins) {
            Some(d) => d,
            None => {
                error!(
                    "CIGPose: SimCC X output '{}' missing or wrong size (expected {} floats)",
                    x_name,
                    NUM_JOINTS * self.simcc_x_bins
                );
                return Vec::new();
            }
        };
        let simcc_y = match extract(&y_name, NUM_JOINTS * self.simcc_y_bins) {
            Some(d) => d,
            None => {
                error!(
                    "CIGPose: SimCC Y output '{}' missing or wrong size (expected {} floats)",
                    y_name,
                    NUM_JOINTS * self.simcc_y_bins
                );
                return Vec::new();
            }
        };
        drop(outputs);

        decode::decode_simcc_xy(&simcc_x, &simcc_y, self.simcc_x_bins, self.simcc_y_bins)
    }
}

/// Pull `split_ratio` out of the ONNX model's `cigpose_meta` custom
/// metadata field. The author's reference Python loader looks here, so
/// we follow the same contract — metadata is the canonical source and
/// `2.0` is only a last-resort fallback for malformed exports.
#[cfg(feature = "inference")]
fn read_split_ratio_from_metadata(session: &Session) -> Option<f32> {
    let meta = session.metadata().ok()?;
    let raw = meta.custom("cigpose_meta")?;
    // Metadata payload is a JSON blob like `{"input_w":288,"input_h":384,"split_ratio":2.0}`.
    // Pull `split_ratio` out without taking a serde dependency: it's a
    // single numeric scalar, so a minimal substring scan is enough.
    let key = "\"split_ratio\"";
    let kpos = raw.find(key)?;
    let after = &raw[kpos + key.len()..];
    let colon = after.find(':')?;
    let tail = after[colon + 1..].trim_start();
    let end = tail
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E'))
        .unwrap_or(tail.len());
    let num = tail[..end].trim();
    let parsed: f32 = num.parse().ok()?;
    if parsed.is_finite() && parsed > 0.0 {
        Some(parsed)
    } else {
        warn!("CIGPose: cigpose_meta split_ratio={} is not a positive finite number; falling back to 2.0", parsed);
        None
    }
}
