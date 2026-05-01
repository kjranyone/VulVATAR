//! YOLOX person detector — pre-stage to RTMW3D so we can crop the camera
//! frame to the actual subject before feeding it to the 288×384 pose model.
//!
//! Uses the standard mmpose/mmdeploy YOLOX-m export with NMS baked into
//! the graph (`models/yolox.onnx`). Older raw-head YOLOX-nano exports are
//! still accepted as `models/yolox_nano.onnx` for local experiments.
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
//! ## Supported schemas
//!
//! Bundled mmpose/mmdeploy YOLOX-m:
//!
//!   input   `(1, 3, 640, 640)` Float32 — BGR, raw [0, 255], letterboxed
//!   dets    `(1, N, 5)` Float32        — `[x1, y1, x2, y2, score]`
//!   labels  `(1, N)` Int64             — COCO class id
//!
//! Legacy raw YOLOX-nano:
//!
//!   input   `(1, 3, 416, 416)` Float32 — BGR, raw [0, 255], letterboxed
//!   output  `(1, 3549, 85)` Float32    — per-anchor head output:
//!                                       `[cx, cy, w, h, obj, cls0..cls79]`
//!
//! `3549 = 52² + 26² + 13²` — one row per anchor at strides 8 / 16 / 32
//! over a 416² input, matching YOLOX's anchor-free decoupled head. `cx`,
//! `cy`, `w`, `h` are *raw* (need `+ grid * stride` and `exp() * stride`
//! respectively to land in letterbox pixel space). `obj` and `cls_i`
//! arrive post-sigmoid.
//!
//! Coordinates land in letterbox space; we undo the letterbox here
//! before returning a bbox in original-image pixel coords.
//!
//! The `cfg(feature = "inference")` gate lives on the `mod yolox`
//! declaration in `tracking::mod.rs` — duplicating it here as an inner
//! attribute caused clippy `duplicated_attributes` once both layers were
//! enabled, so we keep the gate single-sourced.


use log::{info, warn};
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

use super::rtmw3d::{build_session, InferenceBackend};

const MMDEPLOY_INPUT_SIZE: u32 = 640;
const RAW_INPUT_SIZE: u32 = 416;
const PAD_VALUE: u8 = 114;
/// YOLOX strides for the three FPN levels at the standard config.
/// Matches the official `tools/export_onnx.py` head.
const STRIDES: [u32; 3] = [8, 16, 32];
/// Person bbox detection floor. Combined `obj × cls[person]` score, both
/// of which arrive post-sigmoid, so this is a probability threshold.
const PERSON_SCORE_THRESHOLD: f32 = 0.30;
/// COCO class index for "person".
const PERSON_CLASS_ID: usize = 0;
/// Total channels per anchor: 4 bbox + 1 obj + 80 COCO classes.
const ANCHOR_CHANNELS: usize = 85;

/// Captured letterbox transform parameters from `preprocess`. Both
/// decode paths use these to map detection bboxes from the model's
/// letterbox-space back into original image pixel coords. Bundled into
/// one struct so the decoders take 3–4 args instead of 8.
#[derive(Clone, Copy, Debug)]
struct LetterboxFrame {
    /// Resize scale applied to the source frame (`min(W/src_w, H/src_h)`).
    scale: f32,
    /// Horizontal padding added inside the model input to centre the
    /// scaled source.
    pad_x: f32,
    /// Vertical padding (counterpart to `pad_x`).
    pad_y: f32,
    /// Original source width — bbox max is clamped to this.
    width: u32,
    /// Original source height — bbox max is clamped to this.
    height: u32,
}

impl LetterboxFrame {
    /// Map a letterbox-space bbox `(lx1, ly1, lx2, ly2)` back to original
    /// image pixel coords, clamped to `[0, width] × [0, height]`. Returns
    /// `None` if the resulting box is empty.
    fn undo(&self, lx1: f32, ly1: f32, lx2: f32, ly2: f32) -> Option<(f32, f32, f32, f32)> {
        let inv = 1.0 / self.scale.max(1e-6);
        let x1 = ((lx1 - self.pad_x) * inv).clamp(0.0, self.width as f32);
        let y1 = ((ly1 - self.pad_y) * inv).clamp(0.0, self.height as f32);
        let x2 = ((lx2 - self.pad_x) * inv).clamp(0.0, self.width as f32);
        let y2 = ((ly2 - self.pad_y) * inv).clamp(0.0, self.height as f32);
        if x2 <= x1 || y2 <= y1 {
            None
        } else {
            Some((x1, y1, x2, y2))
        }
    }
}

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
    schema: YoloxOutputSchema,
    input_size: u32,
    backend: InferenceBackend,
}

#[derive(Clone, Copy, Debug)]
enum YoloxOutputSchema {
    MmdeployNms,
    RawHead,
}

impl YoloxPersonDetector {
    /// Try to load a YOLOX person detector. Returns `Ok(None)` if no
    /// supported model is present, so the caller can fall back to
    /// whole-frame RTMW3D.
    pub fn try_from_models_dir(dir: &Path) -> Result<Option<Self>, String> {
        let (path, schema, input_size) = {
            let mmdeploy = dir.join("yolox.onnx");
            if mmdeploy.exists() {
                (
                    mmdeploy,
                    YoloxOutputSchema::MmdeployNms,
                    MMDEPLOY_INPUT_SIZE,
                )
            } else {
                let raw = dir.join("yolox_nano.onnx");
                (raw, YoloxOutputSchema::RawHead, RAW_INPUT_SIZE)
            }
        };
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
        if output_names.is_empty() {
            return Err("YOLOX: no outputs declared".to_string());
        }
        match schema {
            YoloxOutputSchema::MmdeployNms if output_names.len() < 2 => {
                return Err(format!(
                    "YOLOX: {} looks like mmdeploy NMS export but has outputs {:?}",
                    path.display(),
                    output_names
                ));
            }
            _ => {}
        }
        info!(
            "YOLOX loaded {} as {:?} input={} outputs={:?}",
            path.display(),
            schema,
            input_size,
            output_names
        );
        Ok(Some(Self {
            session,
            input_name,
            output_names,
            schema,
            input_size,
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

        let (tensor, scale, pad_x, pad_y) = preprocess(rgb, width, height, self.input_size);
        let input = match TensorRef::from_array_view(&tensor) {
            Ok(t) => t,
            Err(e) => {
                warn!("YOLOX: TensorRef creation failed: {}", e);
                return None;
            }
        };

        let output_name = self.output_names[0].clone();
        let labels_name = self.output_names.get(1).cloned();
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

        let lbf = LetterboxFrame {
            scale,
            pad_x,
            pad_y,
            width,
            height,
        };
        match self.schema {
            YoloxOutputSchema::MmdeployNms => decode_mmdeploy_nms(
                &outputs,
                output_name.as_str(),
                labels_name.as_deref()?,
                &lbf,
            ),
            YoloxOutputSchema::RawHead => {
                decode_raw_head(&outputs, output_name.as_str(), self.input_size, &lbf)
            }
        }
    }
}

fn decode_mmdeploy_nms(
    outputs: &ort::session::SessionOutputs,
    dets_name: &str,
    labels_name: &str,
    lbf: &LetterboxFrame,
) -> Option<PersonBbox> {
    let dets = outputs.get(dets_name)?;
    let labels = outputs.get(labels_name)?;
    let (dets_shape, dets_data) = dets.try_extract_tensor::<f32>().ok()?;
    let (labels_shape, labels_data) = labels.try_extract_tensor::<i64>().ok()?;

    if dets_shape.len() != 3 || dets_shape[2] as usize != 5 {
        warn!(
            "YOLOX: unexpected dets shape {:?} (want [1, N, 5])",
            dets_shape
        );
        return None;
    }
    let n = dets_shape[1] as usize;
    if labels_shape.len() != 2 || labels_shape[1] as usize != n {
        warn!(
            "YOLOX: labels shape {:?} does not match dets count {}",
            labels_shape, n
        );
        return None;
    }
    if dets_data.len() < n * 5 || labels_data.len() < n {
        return None;
    }

    let mut best: Option<PersonBbox> = None;
    for (i, &label) in labels_data.iter().take(n).enumerate() {
        if label != PERSON_CLASS_ID as i64 {
            continue;
        }
        let base = i * 5;
        let score = dets_data[base + 4];
        if score < PERSON_SCORE_THRESHOLD {
            continue;
        }
        let Some((x1, y1, x2, y2)) = lbf.undo(
            dets_data[base],
            dets_data[base + 1],
            dets_data[base + 2],
            dets_data[base + 3],
        ) else {
            continue;
        };
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

fn decode_raw_head(
    outputs: &ort::session::SessionOutputs,
    output_name: &str,
    input_size: u32,
    lbf: &LetterboxFrame,
) -> Option<PersonBbox> {
    let head = outputs.get(output_name)?;
    let (head_shape, head_data) = head.try_extract_tensor::<f32>().ok()?;

    // Expect (1, N, 85) where N = Σ (input/stride)² over strides.
    if head_shape.len() != 3 || head_shape[2] as usize != ANCHOR_CHANNELS {
        warn!(
            "YOLOX: unexpected output shape {:?} (want [1, N, {}])",
            head_shape, ANCHOR_CHANNELS
        );
        return None;
    }
    let n = head_shape[1] as usize;
    let expected_n: usize = STRIDES
        .iter()
        .map(|s| {
            let g = (input_size / s) as usize;
            g * g
        })
        .sum();
    if n != expected_n {
        warn!(
            "YOLOX: anchor count mismatch: got {}, expected {} for {}x{} with strides {:?}",
            n, expected_n, input_size, input_size, STRIDES
        );
        return None;
    }
    if head_data.len() < n * ANCHOR_CHANNELS {
        return None;
    }

    let mut best: Option<PersonBbox> = None;
    let mut anchor_idx: usize = 0;
    for &stride in &STRIDES {
        let grid = (input_size / stride) as usize;
        let stride_f = stride as f32;
        for gy in 0..grid {
            for gx in 0..grid {
                let base = anchor_idx * ANCHOR_CHANNELS;
                anchor_idx += 1;

                let obj = head_data[base + 4];
                let cls = head_data[base + 5 + PERSON_CLASS_ID];
                let score = obj * cls;
                if score < PERSON_SCORE_THRESHOLD {
                    continue;
                }

                let raw_cx = head_data[base];
                let raw_cy = head_data[base + 1];
                let raw_w = head_data[base + 2];
                let raw_h = head_data[base + 3];
                let cx = (raw_cx + gx as f32) * stride_f;
                let cy = (raw_cy + gy as f32) * stride_f;
                let w = raw_w.exp() * stride_f;
                let h = raw_h.exp() * stride_f;

                let Some((x1, y1, x2, y2)) =
                    lbf.undo(cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h)
                else {
                    continue;
                };
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
        }
    }
    best
}

/// Letterbox to `input_size × input_size` with 114 padding, BGR raw
/// [0, 255]. Returns the NCHW tensor plus the scale + pad offsets
/// needed to map detections back to original image coords.
fn preprocess(rgb: &[u8], src_w: u32, src_h: u32, input_size: u32) -> (Array4<f32>, f32, f32, f32) {
    let dst = input_size as usize;
    let mut tensor = Array4::<f32>::from_elem((1, 3, dst, dst), PAD_VALUE as f32);
    let scale = (input_size as f32 / src_w as f32).min(input_size as f32 / src_h as f32);
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
