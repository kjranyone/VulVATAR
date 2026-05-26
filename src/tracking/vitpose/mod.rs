//! ViTPose-Wholebody 2D pose inference.
//!
//! Sibling to RTMW3D / CIGPose. Loads
//! `models/vitpose-s-wholebody.onnx` (downloaded by
//! `dev.ps1 Install-VitposeWholebody`, originally
//! `JunkyByte/easy_ViTPose` on Hugging Face), emits 133
//! COCO-Wholebody keypoints in 2D. Pair with a metric depth model
//! (DAv2 / MoGe-2) for full 3D — first integration is 2D-only
//! (`nz = 0.5` planar fallback) so we can A/B against RTMW3D on the
//! same `validate_pipeline` harness before adding depth.
//!
//! ## Pipeline
//!
//! 1. **Resize / normalise** the (already YOLOX-cropped) frame to
//!    192×256 NCHW with ImageNet mean/std.
//! 2. **Infer**: one output `(1, 133, H, W)` heatmap tensor. ViTPose-S
//!    emits `H = 64, W = 48`, but we read the shape at runtime so
//!    swapping in -B / -L / -H variants needs no code change.
//! 3. **Decode**: per-channel argmax + MMPose-style ±0.25 sub-pixel
//!    nudge → normalised crop-relative `(nx, ny)` + clamped score.

#[cfg(feature = "inference")]
mod consts;
#[cfg(feature = "inference")]
mod decode;
#[cfg(feature = "inference")]
mod preprocess;

#[cfg(feature = "inference")]
pub use decode::{DecodedJoint2d, NUM_JOINTS};

/// Picks which easy_ViTPose checkpoint to load. The variants are
/// shape-compatible (256×192 NCHW input, `(133, H, W)` heatmap
/// output with the same H/W per variant) so the surrounding pipeline
/// doesn't need to branch — only the filename changes. Selectable
/// via the `VITPOSE_VARIANT` env var; default is `s` (smallest,
/// fastest, ~97 MB).
///
/// Sizes (wholebody ONNX from JunkyByte/easy_ViTPose):
/// - `S` — 97 MB
/// - `B` — 360 MB
/// - `L` — 1.2 GB
/// - `H` — 2.55 GB (graph + 393 external data files in
///   `vitpose-h-wholebody_onnx/`)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VitposeVariant {
    S,
    B,
    L,
    H,
}

impl VitposeVariant {
    pub fn from_env() -> Self {
        match std::env::var("VITPOSE_VARIANT")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("b") | Some("base") => Self::B,
            Some("l") | Some("large") => Self::L,
            Some("h") | Some("huge") => Self::H,
            _ => Self::S,
        }
    }

    /// ONNX filename under `models/` for this variant. H uses
    /// external data, so the graph file lives inside the
    /// `_onnx` subdir alongside its weight tensors — keeping them
    /// colocated lets `ort` resolve the `location:` references in
    /// the proto without extra runtime configuration.
    pub fn filename(self) -> &'static str {
        match self {
            Self::S => "vitpose-s-wholebody.onnx",
            Self::B => "vitpose-b-wholebody.onnx",
            Self::L => "vitpose-l-wholebody.onnx",
            Self::H => "vitpose-h-wholebody_onnx/vitpose-h-wholebody.onnx",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::S => "ViTPose-S-Wholebody",
            Self::B => "ViTPose-B-Wholebody",
            Self::L => "ViTPose-L-Wholebody",
            Self::H => "ViTPose-H-Wholebody",
        }
    }
}

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

#[cfg(feature = "inference")]
pub struct VitposeInference {
    session: Session,
    input_name: String,
    output_name: String,
    backend: InferenceBackend,
}

#[cfg(feature = "inference")]
impl VitposeInference {
    pub fn from_model_path(model_path: impl AsRef<Path>) -> Result<Self, String> {
        Self::from_model_path_with_label(model_path, "ViTPose-Wholebody")
    }

    pub fn from_model_path_with_label(
        model_path: impl AsRef<Path>,
        label: &str,
    ) -> Result<Self, String> {
        let path = model_path.as_ref();
        if !path.is_file() {
            return Err(format!(
                "{} model not found at {}. Run dev.ps1 Install-VitposeWholebody to fetch it.",
                label,
                path.display()
            ));
        }
        info!("Loading {} from {}", label, path.display());

        let path_str = path.to_string_lossy().into_owned();
        let (session, backend) = build_session(&path_str, 4, label)?;

        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input_0".to_string());
        let outputs: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        if outputs.is_empty() {
            return Err(format!("{} model declared no outputs", label));
        }
        let output_name = outputs[0].clone();
        info!(
            "{} I/O: input='{}', output='{}'",
            label, input_name, output_name
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

    /// Run inference on an RGB crop (expected to already be a YOLOX-
    /// padded person bbox; the caller remaps the returned keypoints
    /// back to whole-frame coords). Returns one decoded 2D joint per
    /// COCO-Wholebody index. Keypoint coords are crop-relative
    /// `(nx, ny) ∈ [0, 1]²`.
    pub fn estimate(&mut self, rgb: &[u8], width: u32, height: u32) -> Vec<DecodedJoint2d> {
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
                error!("ViTPose: TensorRef creation failed: {}", e);
                return Vec::new();
            }
        };
        let out_name = self.output_name.clone();
        let outputs = match self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
        {
            Ok(o) => o,
            Err(e) => {
                error!("ViTPose: ONNX run failed: {}", e);
                return Vec::new();
            }
        };
        let Some(value) = outputs.get(out_name.as_str()) else {
            error!("ViTPose: output '{}' missing from session results", out_name);
            return Vec::new();
        };
        let (shape, data) = match value.try_extract_tensor::<f32>() {
            Ok(s) => s,
            Err(e) => {
                error!("ViTPose: failed to extract heatmap tensor: {}", e);
                return Vec::new();
            }
        };
        // Expected shape: (1, 133, H, W). The first two dims are
        // fixed for this model; the spatial dims are dynamic so we
        // pick them up from the runtime tensor instead of hardcoding
        // 64×48. Defensive: tolerate `(133, H, W)` with the batch
        // already squeezed.
        let (h, w) = match shape.len() {
            4 => (shape[2] as usize, shape[3] as usize),
            3 => (shape[1] as usize, shape[2] as usize),
            other => {
                error!(
                    "ViTPose: unexpected output rank {} (shape={:?}), expected 3 or 4",
                    other, shape
                );
                return Vec::new();
            }
        };
        let owned = data.to_vec();
        drop(outputs);
        decode::decode_heatmaps(&owned, h, w)
    }
}
