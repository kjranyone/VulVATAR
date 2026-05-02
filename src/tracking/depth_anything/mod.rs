//! Depth Anything V2 Small inference — relative monocular depth.
//!
//! Used by [`super::rtmw3d_with_depth`] as the depth source for the
//! 30-fps streaming pipeline. RTMW3D supplies fast 2D body / hand /
//! face landmarks; this module supplies a per-pixel depth signal
//! that the provider then calibrates to approximate metric scale via
//! a known body anchor (shoulder span ≈ 0.40 m).
//!
//! ## Why DAv2 (Small) and not MoGe-2
//!
//! MoGe-2 emits true metric points but its ViT-S backbone with full
//! self-attention runs at ~200 ms on DirectML — incompatible with
//! 30 fps. DAv2-Small is also DPT-architected, but the DINOv2-S
//! backbone (~21 M params) plus a much smaller token count (1036 vs
//! 1564 here) brings inference to ~30 ms range. The trade-off is
//! the output is *relative* (inverse-depth-proportional) rather than
//! metric, which the calibration step in `rtmw3d_with_depth` solves
//! for using the body-anchor 1-DoF fit:
//!
//! ```text
//!   z_metric = c / d_raw,   c = body_distance / sqrt(geom_term)
//! ```
//!
//! ## Pipeline
//!
//! 1. **Resize** source RGB to `INPUT_W × INPUT_H` (multiples of 14)
//!    NCHW. Rescale 1/255 then ImageNet-normalize.
//! 2. **Run** ONNX: input `pixel_values` `(1, 3, H, W)` →
//!    output `predicted_depth` `(1, H, W)` of inverse-depth-style
//!    relative values (larger ≈ closer to camera).
//! 3. Return [`DepthAnythingFrame`] holding the raw output for the
//!    provider to calibrate. No metric conversion or back-projection
//!    happens here — that requires camera intrinsics + an anchor,
//!    which the provider knows but this module does not.

#[cfg(feature = "inference")]
pub mod consts;
#[cfg(feature = "inference")]
mod preprocess;

#[cfg(feature = "inference")]
pub use consts::{INPUT_H, INPUT_W};

#[cfg(feature = "inference")]
use log::{error, info};
#[cfg(feature = "inference")]
use ort::session::Session;
#[cfg(feature = "inference")]
use ort::value::TensorRef;
#[cfg(feature = "inference")]
use std::path::Path;

#[cfg(feature = "inference")]
use super::rtmw3d::{build_session, InferenceBackend};

/// Per-frame DAv2 output. `relative` is row-major
/// `width × height` floats — DAv2 emits inverse-depth-style values
/// where larger ≈ closer to camera. `0.0` is the background asymptote.
#[cfg(feature = "inference")]
pub struct DepthAnythingFrame {
    pub width: u32,
    pub height: u32,
    pub relative: Vec<f32>,
}

#[cfg(feature = "inference")]
impl DepthAnythingFrame {
    /// Sample raw inverse-depth at a frame-relative position. Local
    /// median over a small window suppresses single-pixel artefacts
    /// the same way [`super::cigpose_metric_depth::sample_metric_point`]
    /// does for MoGe.
    pub fn sample_inverse_depth(&self, nx: f32, ny: f32, radius: i32) -> Option<f32> {
        if !nx.is_finite() || !ny.is_finite() {
            return None;
        }
        let w = self.width as i32;
        let h = self.height as i32;
        let cx = (nx * self.width as f32).round() as i32;
        let cy = (ny * self.height as f32).round() as i32;
        if cx < 0 || cy < 0 || cx >= w || cy >= h {
            return None;
        }
        let mut values = Vec::with_capacity(((radius * 2 + 1) * (radius * 2 + 1)) as usize);
        for yy in (cy - radius).max(0)..=(cy + radius).min(h - 1) {
            for xx in (cx - radius).max(0)..=(cx + radius).min(w - 1) {
                let idx = yy as usize * self.width as usize + xx as usize;
                let v = self.relative[idx];
                if v.is_finite() && v > 0.0 {
                    values.push(v);
                }
            }
        }
        if values.is_empty() {
            return None;
        }
        values.sort_by(|a, b| a.total_cmp(b));
        Some(values[values.len() / 2])
    }
}

/// ONNX output tensor name. Verified via `inspect_onnx` against the
/// `onnx-community/depth-anything-v2-small` export. Falls back to
/// position 0 if the file ever ships under a different name.
#[cfg(feature = "inference")]
const OUTPUT_NAME_DEPTH: &str = "predicted_depth";

#[cfg(feature = "inference")]
pub struct DepthAnythingV2Inference {
    session: Session,
    input_name: String,
    output_name: String,
    backend: InferenceBackend,
}

#[cfg(feature = "inference")]
impl DepthAnythingV2Inference {
    pub fn from_model_path(model_path: impl AsRef<Path>) -> Result<Self, String> {
        let path = model_path.as_ref();
        if !path.is_file() {
            return Err(format!(
                "Depth Anything V2 model not found at {}. Run dev.ps1 setup (rtmw3d-with-depth) to fetch it.",
                path.display()
            ));
        }
        info!("Loading Depth Anything V2 from {}", path.display());

        let path_str = path.to_string_lossy().into_owned();
        let (session, backend) = build_session(&path_str, 4, "DAv2-Small")?;

        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "pixel_values".to_string());
        let outputs: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        if outputs.is_empty() {
            return Err("DAv2 model declares no outputs".to_string());
        }
        let output_name = outputs
            .iter()
            .find(|n| n.as_str() == OUTPUT_NAME_DEPTH)
            .cloned()
            .unwrap_or_else(|| outputs[0].clone());
        info!(
            "DAv2 I/O: input='{}', output='{}' (declared {:?})",
            input_name, output_name, outputs
        );

        Ok(Self {
            session,
            input_name,
            output_name,
            backend,
        })
    }

    pub fn backend(&self) -> &InferenceBackend {
        &self.backend
    }

    /// Run inference on a whole-frame RGB buffer. The returned
    /// [`DepthAnythingFrame`] is in DAv2's input pixel grid
    /// (`INPUT_W × INPUT_H`). The provider remaps RTMW3D 2D
    /// keypoints from frame-relative `[0, 1]` coords into this grid
    /// before sampling.
    pub fn estimate(&mut self, rgb: &[u8], src_w: u32, src_h: u32) -> Option<DepthAnythingFrame> {
        let expected = (src_w as usize)
            .saturating_mul(src_h as usize)
            .saturating_mul(3);
        if rgb.len() < expected || src_w == 0 || src_h == 0 {
            return None;
        }
        let tensor = preprocess::preprocess(rgb, src_w, src_h);
        let input = match TensorRef::from_array_view(&tensor) {
            Ok(v) => v,
            Err(e) => {
                error!("DAv2: TensorRef creation failed: {}", e);
                return None;
            }
        };

        let outputs = match self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
        {
            Ok(o) => o,
            Err(e) => {
                error!("DAv2: ONNX run failed: {}", e);
                return None;
            }
        };

        let pixel_count = (INPUT_W as usize) * (INPUT_H as usize);
        let depth_data = match outputs.get(self.output_name.as_str()) {
            Some(v) => match v.try_extract_tensor::<f32>() {
                Ok((_, data)) if data.len() >= pixel_count => data.to_vec(),
                Ok((_, data)) => {
                    error!(
                        "DAv2: '{}' tensor too small ({} < {})",
                        self.output_name,
                        data.len(),
                        pixel_count
                    );
                    return None;
                }
                Err(e) => {
                    error!("DAv2: '{}' extract failed: {}", self.output_name, e);
                    return None;
                }
            },
            None => {
                error!("DAv2: '{}' output missing", self.output_name);
                return None;
            }
        };
        drop(outputs);

        Some(DepthAnythingFrame {
            width: INPUT_W,
            height: INPUT_H,
            relative: depth_data,
        })
    }
}
