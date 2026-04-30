//! YOLOX-m human-art person detector — pre-stage to RTMW3D so we can
//! crop the camera frame to the actual subject before feeding it to the
//! 288×384 pose model.
//!
//! The mmpose `rtmposev1` ONNX SDK ships an `end2end.onnx` for
//! YOLOX-m fine-tuned on Human-Art. The export bakes NMS into the graph,
//! so the post-processing here is just a max-score gather rather than a
//! full NMS implementation.
//!
//! ## Why bbox first
//!
//! RTMW3D is a top-down model — accuracy degrades roughly linearly with
//! how much of the model input is "person pixels" vs background. With a
//! whole 16:9 webcam frame squashed into 288×384, a subject that fills
//! 60% of the camera height ends up filling ~60% of model height, which
//! is fine. But on inputs where the subject is smaller (jumping mid-air,
//! camera further back, or just not centred), the model gets a tiny
//! person and confidence drops. YOLOX gives us a tight crop so the
//! subject always fills the model input.
//!
//! ## Schema
//!
//!   input  `(1, 3, 640, 640)` Float32 — BGR, raw [0, 255], letterboxed
//!   dets   `(1, N, 5)` Float32       — `[x1, y1, x2, y2, score]` post-NMS
//!   labels `(1, N)` Int64            — class index (0 = person)
//!
//! Coordinates are in letterbox space; we undo the letterbox here before
//! returning a bbox in original-image pixel coords.

#![cfg(feature = "inference")]
#![allow(dead_code)]

use log::warn;
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

use super::rtmw3d::{build_session, InferenceBackend};

const INPUT_SIZE: u32 = 640;
const PAD_VALUE: u8 = 114;
/// Person bbox detection floor. The mmdeploy export bakes NMS in with a
/// score threshold of 0.05 — we'd rather discard low-confidence persons
/// than feed a noisy bbox to RTMW3D.
const PERSON_SCORE_THRESHOLD: f32 = 0.30;
const PERSON_CLASS_ID: i64 = 0;

/// Detected person bounding box in *original image pixel* coords, post
/// letterbox-undo.
#[derive(Clone, Copy, Debug)]
pub struct PersonBbox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

impl PersonBbox {
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }
    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }
}

pub struct YoloxPersonDetector {
    session: Session,
    input_name: String,
    /// Output names in declared order: typically `["dets", "labels"]`.
    output_names: Vec<String>,
    backend: InferenceBackend,
}

impl YoloxPersonDetector {
    /// Try to load `models/yolox.onnx`. Returns `Ok(None)` if the file
    /// is missing — caller should treat detection as optional and fall
    /// back to whole-frame inference.
    pub fn try_from_models_dir(dir: &Path) -> Result<Option<Self>, String> {
        let path = dir.join("yolox.onnx");
        if !path.exists() {
            return Ok(None);
        }
        let model_path = path
            .to_str()
            .ok_or_else(|| "YOLOX: model path is not valid UTF-8".to_string())?;
        let (session, backend) = build_session(model_path, 1, "YOLOX")
            .map_err(|e| format!("YOLOX: failed to build session: {}", e))?;
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .ok_or_else(|| "YOLOX: no inputs declared".to_string())?;
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        if output_names.len() < 2 {
            return Err(format!(
                "YOLOX: expected ≥2 outputs (dets, labels), got {}",
                output_names.len()
            ));
        }
        Ok(Some(Self {
            session,
            input_name,
            output_names,
            backend,
        }))
    }

    pub fn backend(&self) -> &InferenceBackend {
        &self.backend
    }

    /// Run detection on the RGB camera frame. Returns the highest-score
    /// person bbox, or `None` if nothing was detected above
    /// [`PERSON_SCORE_THRESHOLD`].
    ///
    /// `rgb` is laid out as `H × W × 3` u8 with RGB channel order.
    pub fn detect_largest_person(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> Option<PersonBbox> {
        if width == 0 || height == 0 {
            return None;
        }
        let expected_len = (width as usize) * (height as usize) * 3;
        if rgb.len() < expected_len {
            return None;
        }

        let (tensor, scale, pad_x, pad_y) = preprocess(rgb, width, height);
        let input = match TensorRef::from_array_view(&tensor) {
            Ok(t) => t,
            Err(e) => {
                warn!("YOLOX: TensorRef creation failed: {}", e);
                return None;
            }
        };

        let dets_name = self.output_names[0].clone();
        let labels_name = self.output_names[1].clone();
        let outputs = match self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
        {
            Ok(out) => out,
            Err(e) => {
                warn!("YOLOX: ONNX run failed: {}", e);
                return None;
            }
        };

        let dets = outputs.get(dets_name.as_str())?;
        let (dets_shape, dets_data) = dets.try_extract_tensor::<f32>().ok()?;
        let labels = outputs.get(labels_name.as_str())?;
        let (labels_shape, labels_data) = labels.try_extract_tensor::<i64>().ok()?;

        // dets: (1, N, 5), labels: (1, N).
        if dets_shape.len() != 3 || dets_shape[2] != 5 {
            warn!("YOLOX: unexpected dets shape {:?}", dets_shape);
            return None;
        }
        let n = dets_shape[1] as usize;
        if labels_shape.len() < 2 || (labels_shape[1] as usize) != n {
            warn!(
                "YOLOX: dets/labels mismatch: dets={:?} labels={:?}",
                dets_shape, labels_shape
            );
            return None;
        }
        if dets_data.len() < n * 5 || labels_data.len() < n {
            return None;
        }

        // Pick the highest-score *person* (label == 0) above threshold.
        // We use score as the picker rather than bbox area because the
        // mmdeploy NMS already returns dets sorted by score; in practice
        // the top-scoring person is the "main subject" we want to track.
        let mut best: Option<PersonBbox> = None;
        for i in 0..n {
            if labels_data[i] != PERSON_CLASS_ID {
                continue;
            }
            let base = i * 5;
            let score = dets_data[base + 4];
            if score < PERSON_SCORE_THRESHOLD {
                continue;
            }
            // Letterbox-space → original-image-space.
            let lx1 = dets_data[base];
            let ly1 = dets_data[base + 1];
            let lx2 = dets_data[base + 2];
            let ly2 = dets_data[base + 3];
            let inv = 1.0 / scale.max(1e-6);
            let x1 = ((lx1 - pad_x) * inv).clamp(0.0, width as f32);
            let y1 = ((ly1 - pad_y) * inv).clamp(0.0, height as f32);
            let x2 = ((lx2 - pad_x) * inv).clamp(0.0, width as f32);
            let y2 = ((ly2 - pad_y) * inv).clamp(0.0, height as f32);
            if x2 <= x1 || y2 <= y1 {
                continue;
            }
            let bbox = PersonBbox {
                x1,
                y1,
                x2,
                y2,
                score,
            };
            match best {
                None => best = Some(bbox),
                Some(prev) if score > prev.score => best = Some(bbox),
                _ => {}
            }
        }
        best
    }
}

/// Letterbox to 640×640 with 114 padding, BGR raw [0, 255]. Returns the
/// NCHW tensor plus the scale + pad offsets needed to map detections
/// back to original image coords.
fn preprocess(rgb: &[u8], src_w: u32, src_h: u32) -> (Array4<f32>, f32, f32, f32) {
    let dst = INPUT_SIZE as usize;
    let mut tensor = Array4::<f32>::from_elem((1, 3, dst, dst), PAD_VALUE as f32);
    let scale = (INPUT_SIZE as f32 / src_w as f32).min(INPUT_SIZE as f32 / src_h as f32);
    let new_w = ((src_w as f32) * scale).round() as usize;
    let new_h = ((src_h as f32) * scale).round() as usize;
    let pad_x = ((dst - new_w) / 2) as f32;
    let pad_y = ((dst - new_h) / 2) as f32;
    let pad_x_us = pad_x as usize;
    let pad_y_us = pad_y as usize;
    let src_w_us = src_w as usize;
    let src_h_us = src_h as usize;
    let max_x = src_w_us.saturating_sub(1);
    let max_y = src_h_us.saturating_sub(1);
    let stride = src_w_us * 3;
    let inv_scale = 1.0 / scale.max(1e-6);

    // Bilinear resample, OpenCV-style pixel-centred sampling.
    for dy in 0..new_h {
        let fy = ((dy as f32) + 0.5) * inv_scale - 0.5;
        let y0f = fy.floor();
        let wy1 = (fy - y0f).clamp(0.0, 1.0);
        let wy0 = 1.0 - wy1;
        let y0 = (y0f as i32).clamp(0, max_y as i32) as usize;
        let y1 = ((y0f as i32) + 1).clamp(0, max_y as i32) as usize;
        let row0 = y0 * stride;
        let row1 = y1 * stride;
        let out_y = dy + pad_y_us;
        for dx in 0..new_w {
            let fx = ((dx as f32) + 0.5) * inv_scale - 0.5;
            let x0f = fx.floor();
            let wx1 = (fx - x0f).clamp(0.0, 1.0);
            let wx0 = 1.0 - wx1;
            let x0 = (x0f as i32).clamp(0, max_x as i32) as usize;
            let x1 = ((x0f as i32) + 1).clamp(0, max_x as i32) as usize;
            let i00 = row0 + x0 * 3;
            let i01 = row0 + x1 * 3;
            let i10 = row1 + x0 * 3;
            let i11 = row1 + x1 * 3;
            let w00 = wy0 * wx0;
            let w01 = wy0 * wx1;
            let w10 = wy1 * wx0;
            let w11 = wy1 * wx1;
            // RGB → BGR. mmdeploy YOLOX expects BGR raw [0, 255].
            let r = rgb[i00] as f32 * w00
                + rgb[i01] as f32 * w01
                + rgb[i10] as f32 * w10
                + rgb[i11] as f32 * w11;
            let g = rgb[i00 + 1] as f32 * w00
                + rgb[i01 + 1] as f32 * w01
                + rgb[i10 + 1] as f32 * w10
                + rgb[i11 + 1] as f32 * w11;
            let b = rgb[i00 + 2] as f32 * w00
                + rgb[i01 + 2] as f32 * w01
                + rgb[i10 + 2] as f32 * w10
                + rgb[i11 + 2] as f32 * w11;
            let out_x = dx + pad_x_us;
            tensor[(0, 0, out_y, out_x)] = b;
            tensor[(0, 1, out_y, out_x)] = g;
            tensor[(0, 2, out_y, out_x)] = r;
        }
    }
    (tensor, scale, pad_x, pad_y)
}
