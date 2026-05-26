//! MediaPipe BlazePalm detector — finds hand bounding boxes directly
//! in an RGB image. CPU-only inference to keep the DirectML queue
//! uncontested while HMR2 runs on GPU.
//!
//! Model: `models/mediapipe_palm_detection.onnx` (4 MB, Apache 2.0,
//! OpenCV-converted from MediaPipe's TFLite). Input is `(1, 192, 192, 3)`
//! NHWC float32 in `[0, 1]`. Outputs:
//! - `Identity`: `(1, 2016, 18)` — per-anchor `[dx, dy, dw, dh, 7×(kp_x, kp_y)]`
//!   in 192-pixel units relative to the anchor centre.
//! - `Identity_1`: `(1, 2016, 1)` — logit scores; sigmoid for confidence.
//!
//! ## Anchor layout (BlazePalm 192)
//!
//! 2016 anchors come from two feature-map layers:
//! - **Layer 0**: stride 8 → 24×24 grid, 2 anchors per cell = 1152.
//! - **Layer 1**: stride 16 → 12×12 grid, 6 anchors per cell = 864.
//!
//! Each anchor's centre is at the centre of its cell in normalised
//! coords `[0, 1]`. `fixed_anchor_size = true` means we don't store
//! per-anchor `(w, h)`; the bbox dimensions come entirely from the
//! model's `dw, dh` predictions.

use std::path::Path;

use log::{error, info};
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;

use super::rtmw3d::{build_session, InferenceBackend};

const INPUT_SIZE: usize = 192;
const NUM_ANCHORS: usize = 2016;

/// One detected palm. `bbox` is in *source-image pixel* coords (NOT
/// the 192×192 model input). `score` is the post-sigmoid confidence.
#[derive(Clone, Debug)]
pub struct PalmDetection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

pub struct PalmDetector {
    session: Session,
    input_name: String,
    output_names: [String; 2],
    anchors: Vec<(f32, f32)>,
    #[allow(dead_code)]
    backend: InferenceBackend,
}

impl PalmDetector {
    pub fn from_model_path(model_path: impl AsRef<Path>) -> Result<Self, String> {
        let path = model_path.as_ref();
        if !path.is_file() {
            return Err(format!(
                "Palm detector ONNX not found at {}. Fetch from opencv/palm_detection_mediapipe.",
                path.display()
            ));
        }
        info!("Loading MediaPipe Palm Detector from {}", path.display());

        let path_str = path.to_string_lossy().into_owned();
        let (session, backend) = build_session(&path_str, 4, "MediaPipePalm-CPU")?;

        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input_1".to_string());

        let output_names_vec: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        if output_names_vec.len() < 2 {
            return Err(format!(
                "Palm detector declared {} outputs, expected 2",
                output_names_vec.len()
            ));
        }
        let output_names = [output_names_vec[0].clone(), output_names_vec[1].clone()];

        Ok(Self {
            session,
            input_name,
            output_names,
            anchors: generate_anchors(),
            backend,
        })
    }

    /// Detect hands in an RGB image. Returns bboxes in the source
    /// image's pixel coords (not the 192 model space). Up to 4
    /// hands, sorted by score descending.
    pub fn detect(&mut self, rgb: &[u8], src_w: u32, src_h: u32) -> Vec<PalmDetection> {
        if rgb.len() < (src_w as usize * src_h as usize * 3) || src_w == 0 || src_h == 0 {
            return Vec::new();
        }
        // Letterbox-resize to 192×192, keeping the aspect ratio so a
        // wide YOLOX crop doesn't get squashed (palms would lose their
        // square aspect, hurting detection confidence). Track the
        // padding bias so we can un-letterbox the bbox afterwards.
        let scale = INPUT_SIZE as f32 / src_w.max(src_h) as f32;
        let resized_w = (src_w as f32 * scale).round() as usize;
        let resized_h = (src_h as f32 * scale).round() as usize;
        let pad_x = (INPUT_SIZE - resized_w) / 2;
        let pad_y = (INPUT_SIZE - resized_h) / 2;

        let mut tensor = Array4::<f32>::zeros((1, INPUT_SIZE, INPUT_SIZE, 3));
        let src_w_f = src_w as f32;
        let src_h_f = src_h as f32;
        let scale_x = src_w_f / resized_w as f32;
        let scale_y = src_h_f / resized_h as f32;
        let stride = src_w as usize * 3;
        for dy in 0..resized_h {
            let sy = (((dy as f32) + 0.5) * scale_y - 0.5)
                .clamp(0.0, src_h_f - 1.0) as usize;
            let row = sy * stride;
            for dx in 0..resized_w {
                let sx = (((dx as f32) + 0.5) * scale_x - 0.5)
                    .clamp(0.0, src_w_f - 1.0) as usize;
                let i = row + sx * 3;
                tensor[(0, dy + pad_y, dx + pad_x, 0)] = rgb[i] as f32 / 255.0;
                tensor[(0, dy + pad_y, dx + pad_x, 1)] = rgb[i + 1] as f32 / 255.0;
                tensor[(0, dy + pad_y, dx + pad_x, 2)] = rgb[i + 2] as f32 / 255.0;
            }
        }

        let input = match TensorRef::from_array_view(&tensor) {
            Ok(v) => v,
            Err(e) => {
                error!("PalmDetector: TensorRef creation failed: {}", e);
                return Vec::new();
            }
        };
        let outputs = match self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
        {
            Ok(o) => o,
            Err(e) => {
                error!("PalmDetector: ONNX run failed: {}", e);
                return Vec::new();
            }
        };
        let extract = |name: &str, expected: usize| -> Option<Vec<f32>> {
            let v = outputs.get(name)?;
            let (_, data) = v.try_extract_tensor::<f32>().ok()?;
            if data.len() < expected {
                None
            } else {
                Some(data.to_vec())
            }
        };
        let boxes = match extract(&self.output_names[0], NUM_ANCHORS * 18) {
            Some(b) => b,
            None => {
                error!("PalmDetector: boxes output missing or short");
                return Vec::new();
            }
        };
        let scores_raw = match extract(&self.output_names[1], NUM_ANCHORS) {
            Some(s) => s,
            None => {
                error!("PalmDetector: scores output missing or short");
                return Vec::new();
            }
        };
        drop(outputs);

        // Decode every anchor above the score threshold, then NMS.
        const SCORE_THRESHOLD: f32 = 0.5;
        const NMS_IOU: f32 = 0.3;
        let mut candidates: Vec<PalmDetection> = Vec::new();
        let input_size_f = INPUT_SIZE as f32;
        let pad_x_f = pad_x as f32;
        let pad_y_f = pad_y as f32;
        let unscale_x = src_w_f / resized_w as f32;
        let unscale_y = src_h_f / resized_h as f32;
        for i in 0..NUM_ANCHORS {
            let logit = scores_raw[i];
            let score = sigmoid(logit);
            if score < SCORE_THRESHOLD {
                continue;
            }
            let (anchor_x, anchor_y) = self.anchors[i];
            // boxes layout: [dx, dy, dw, dh, kp1_x, kp1_y, ..., kp7_x, kp7_y]
            let base = i * 18;
            let dx = boxes[base] / input_size_f;
            let dy = boxes[base + 1] / input_size_f;
            let dw = boxes[base + 2] / input_size_f;
            let dh = boxes[base + 3] / input_size_f;
            let cx = dx + anchor_x; // bbox centre in 192-normalised coords
            let cy = dy + anchor_y;
            let x1_192 = (cx - dw * 0.5) * input_size_f;
            let y1_192 = (cy - dh * 0.5) * input_size_f;
            let x2_192 = (cx + dw * 0.5) * input_size_f;
            let y2_192 = (cy + dh * 0.5) * input_size_f;
            // Un-letterbox: subtract pad, then scale back to source-image
            // pixels.
            let x1 = (x1_192 - pad_x_f) * unscale_x;
            let y1 = (y1_192 - pad_y_f) * unscale_y;
            let x2 = (x2_192 - pad_x_f) * unscale_x;
            let y2 = (y2_192 - pad_y_f) * unscale_y;
            candidates.push(PalmDetection {
                x1: x1.clamp(0.0, src_w_f),
                y1: y1.clamp(0.0, src_h_f),
                x2: x2.clamp(0.0, src_w_f),
                y2: y2.clamp(0.0, src_h_f),
                score,
            });
        }
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        nms(candidates, NMS_IOU, 4)
    }
}

/// Generate BlazePalm 192's 2016 anchors. Layer 0 (stride 8) puts 2
/// anchors at each of 24×24 = 576 cells (1152 total); layer 1
/// (stride 16) puts 6 anchors at each of 12×12 = 144 cells (864).
/// Both layers use cell-centre coords; `fixed_anchor_size = true`
/// in the original config means we don't track per-anchor sizes.
fn generate_anchors() -> Vec<(f32, f32)> {
    let mut anchors = Vec::with_capacity(NUM_ANCHORS);
    // MediaPipe's anchor calculator iterates row-major over each
    // feature-map layer: outer Y, inner X (matches the model's
    // flattened output where the same flatten order is used for the
    // score / box predictions).
    // Stride 8 layer.
    for y in 0..24 {
        for x in 0..24 {
            let cx = (x as f32 + 0.5) / 24.0;
            let cy = (y as f32 + 0.5) / 24.0;
            anchors.push((cx, cy));
            anchors.push((cx, cy));
        }
    }
    // Stride 16 layer (repeated 3 times in the original 4-layer config,
    // collapsed here to "6 anchors per cell" because the spatial
    // positions match across layers 1/2/3 and only the anchor scale
    // varies — which the model handles via `dw, dh` predictions, not
    // the anchor data we store).
    for y in 0..12 {
        for x in 0..12 {
            let cx = (x as f32 + 0.5) / 12.0;
            let cy = (y as f32 + 0.5) / 12.0;
            for _ in 0..6 {
                anchors.push((cx, cy));
            }
        }
    }
    debug_assert_eq!(anchors.len(), NUM_ANCHORS);
    anchors
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn iou(a: &PalmDetection, b: &PalmDetection) -> f32 {
    let inter_x1 = a.x1.max(b.x1);
    let inter_y1 = a.y1.max(b.y1);
    let inter_x2 = a.x2.min(b.x2);
    let inter_y2 = a.y2.min(b.y2);
    let iw = (inter_x2 - inter_x1).max(0.0);
    let ih = (inter_y2 - inter_y1).max(0.0);
    let inter = iw * ih;
    let area_a = (a.x2 - a.x1).max(0.0) * (a.y2 - a.y1).max(0.0);
    let area_b = (b.x2 - b.x1).max(0.0) * (b.y2 - b.y1).max(0.0);
    let union = area_a + area_b - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

fn nms(mut detections: Vec<PalmDetection>, iou_thresh: f32, max_keep: usize) -> Vec<PalmDetection> {
    let mut kept: Vec<PalmDetection> = Vec::new();
    while let Some(top) = detections.first().cloned() {
        kept.push(top.clone());
        if kept.len() >= max_keep {
            break;
        }
        detections.retain(|d| iou(d, &top) < iou_thresh && d.score > 0.0);
        if detections.is_empty() || detections.first().map(|d| d.score) == Some(top.score) {
            // drop the head we already kept
            if !detections.is_empty() {
                detections.remove(0);
            }
        }
    }
    kept
}
