//! Shared YOLO26-pose ONNX geometry: Ultralytics letterbox preprocessing,
//! square-crop NCHW tensor fill, and the channel-major `(1, 5+3K, A)`
//! pose-head decode.
//!
//! Every Ultralytics pose export — the whole-frame body model (K = 17,
//! `super::yolo26`) and the crop-local hand (K = 21, `fusion::hands`)
//! backends — shares the same
//! head layout: box `cx/cy/w/h` at channels 0..4, objectness at channel
//! 4, then K×(x, y, conf). Channel `c`, anchor `a` lives at
//! `data[c * A + a]`. Coordinates are model-input pixels, so consumers
//! un-letterbox with the same `r` / pad the preprocessing produced (for
//! a square-crop model `r = 1`, pads = 0).
//!
//! `pub` (not `pub(crate)`): the `diagnose_yolo26_pose` bench bin
//! consumes these helpers directly so its copies can't drift.

use ndarray::Array4;

/// Head channels before the K×(x, y, conf) block: box cx/cy/w/h +
/// objectness.
pub const POSE_HEAD_CHANNELS: usize = 5;

/// Total output channels of a pose export with `kps` keypoints.
#[inline]
pub fn pose_channels(kps: usize) -> usize {
    POSE_HEAD_CHANNELS + 3 * kps
}

/// Anchor count of a flat channel-major pose output, or `None` if the
/// byte length doesn't tile into `kps`-keypoint channels.
#[inline]
pub fn pose_anchor_count(data_len: usize, kps: usize) -> Option<usize> {
    let channels = pose_channels(kps);
    if data_len % channels != 0 {
        None
    } else {
        Some(data_len / channels)
    }
}

/// Best-scoring anchor by objectness (channel 4) at or above
/// `min_score`. The production detectors take the argmax without NMS
/// (single-subject crops / single-user desk frames).
pub fn best_pose_anchor(
    data: &[f32],
    anchors: usize,
    min_score: f32,
) -> Option<(usize, f32)> {
    let at = |c: usize, a: usize| data[c * anchors + a];
    let mut best: Option<(usize, f32)> = None;
    for a in 0..anchors {
        let s = at(4, a);
        if s >= min_score && best.map_or(true, |(_, bs)| s > bs) {
            best = Some((a, s));
        }
    }
    best
}

/// Un-letterbox the box of anchor `a`: model-input pixels → frame
/// pixels `(x0, y0, x1, y1)`.
pub fn anchor_box(
    data: &[f32],
    anchors: usize,
    a: usize,
    pad_x: f32,
    pad_y: f32,
    r: f32,
) -> [f32; 4] {
    let at = |c: usize| data[c * anchors + a];
    let (cx, cy, w, h) = (at(0), at(1), at(2), at(3));
    [
        (cx - w / 2.0 - pad_x) / r,
        (cy - h / 2.0 - pad_y) / r,
        (cx + w / 2.0 - pad_x) / r,
        (cy + h / 2.0 - pad_y) / r,
    ]
}

/// Un-letterbox the K keypoints of anchor `a` into frame pixels
/// `[x, y, conf]`.
pub fn anchor_keypoints(
    data: &[f32],
    anchors: usize,
    a: usize,
    kps: usize,
    pad_x: f32,
    pad_y: f32,
    r: f32,
) -> Vec<[f32; 3]> {
    let at = |c: usize| data[c * anchors + a];
    (0..kps)
        .map(|k| {
            let base = POSE_HEAD_CHANNELS + k * 3;
            [
                (at(base) - pad_x) / r,
                (at(base + 1) - pad_y) / r,
                at(base + 2),
            ]
        })
        .collect()
}

/// Ultralytics letterbox: bilinear resize so the long side fits `s`,
/// pad the short side with grey (114/255), NCHW RGB 0..1.
pub fn letterbox(rgb: &[u8], w: u32, h: u32, s: u32) -> Array4<f32> {
    let r = (s as f32 / w as f32).min(s as f32 / h as f32);
    let nw = (w as f32 * r).round() as i64;
    let nh = (h as f32 * r).round() as i64;
    let pad_x = ((s as i64 - nw) / 2).max(0);
    let pad_y = ((s as i64 - nh) / 2).max(0);
    let mut arr = Array4::<f32>::from_elem((1, 3, s as usize, s as usize), 114.0 / 255.0);
    for y in 0..nh {
        let sy = ((y as f32 + 0.5) / r - 0.5).max(0.0);
        let y0 = sy.floor() as u32;
        let y1 = (y0 + 1).min(h - 1);
        let fy = (sy - y0 as f32).clamp(0.0, 1.0);
        for x in 0..nw {
            let sx = ((x as f32 + 0.5) / r - 0.5).max(0.0);
            let x0 = sx.floor() as u32;
            let x1 = (x0 + 1).min(w - 1);
            let fx = (sx - x0 as f32).clamp(0.0, 1.0);
            for c in 0..3usize {
                let p00 = rgb[(y0 as usize * w as usize + x0 as usize) * 3 + c] as f32;
                let p01 = rgb[(y0 as usize * w as usize + x1 as usize) * 3 + c] as f32;
                let p10 = rgb[(y1 as usize * w as usize + x0 as usize) * 3 + c] as f32;
                let p11 = rgb[(y1 as usize * w as usize + x1 as usize) * 3 + c] as f32;
                let top = p00 * (1.0 - fx) + p01 * fx;
                let bot = p10 * (1.0 - fx) + p11 * fx;
                let v = (top * (1.0 - fy) + bot * fy) / 255.0;
                let dx = (x + pad_x) as usize;
                let dy = (y + pad_y) as usize;
                arr[[0, c, dy, dx]] = v;
            }
        }
    }
    arr
}

/// Bilinear-resample the square crop `(x0, y0, size)` of an RGB frame
/// into `tensor` `(1, 3, s, s)` NCHW, RGB `[0, 1]`. Pixels falling
/// outside the frame read as 0 — the crop-local backends size their
/// crops so hands/faces sit well inside, and the edge case only
/// darkens the border ring the way the letterbox pad does.
pub fn fill_crop_tensor(
    rgb: &[u8],
    width: u32,
    height: u32,
    crop: (f32, f32, f32),
    tensor: &mut Array4<f32>,
) {
    let (x0, y0, size) = crop;
    let s = tensor.dim().3 as u32;
    let scale = size / s as f32;
    let (w, h) = (width as i64, height as i64);
    for ty in 0..s {
        let sy = y0 + (ty as f32 + 0.5) * scale - 0.5;
        for tx in 0..s {
            let sx = x0 + (tx as f32 + 0.5) * scale - 0.5;
            let (fx, fy) = (sx.floor(), sy.floor());
            let (ax, ay) = (sx - fx, sy - fy);
            let (ix, iy) = (fx as i64, fy as i64);
            let mut acc = [0.0f32; 3];
            let mut wsum = 0.0f32;
            for (dy, wy) in [(0i64, 1.0 - ay), (1, ay)] {
                for (dx, wx) in [(0i64, 1.0 - ax), (1, ax)] {
                    let (px, py) = (ix + dx, iy + dy);
                    if px < 0 || py < 0 || px >= w || py >= h {
                        continue;
                    }
                    let wgt = wx * wy;
                    let i = ((py * w + px) * 3) as usize;
                    acc[0] += rgb[i] as f32 * wgt;
                    acc[1] += rgb[i + 1] as f32 * wgt;
                    acc[2] += rgb[i + 2] as f32 * wgt;
                    wsum += wgt;
                }
            }
            let inv = if wsum > 0.0 { 1.0 / (255.0 * wsum) } else { 0.0 };
            let (ty, tx) = (ty as usize, tx as usize);
            tensor[(0, 0, ty, tx)] = acc[0] * inv;
            tensor[(0, 1, ty, tx)] = acc[1] * inv;
            tensor[(0, 2, ty, tx)] = acc[2] * inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic (1, 5+3K, A) output with one strong anchor.
    fn synth(kps: usize, anchors: usize, a: usize, kp: [f32; 3]) -> Vec<f32> {
        let channels = pose_channels(kps);
        let mut data = vec![0.0f32; channels * anchors];
        data[4 * anchors + a] = 0.9; // objectness
        // Box centred at (100, 100), 50×50.
        data[0 * anchors + a] = 100.0;
        data[1 * anchors + a] = 100.0;
        data[2 * anchors + a] = 50.0;
        data[3 * anchors + a] = 50.0;
        for k in 0..kps {
            let base = (POSE_HEAD_CHANNELS + k * 3) * anchors + a;
            data[base] = kp[0];
            data[base + 1] = kp[1];
            data[base + 2] = kp[2];
        }
        data
    }

    #[test]
    fn best_anchor_picks_objectness_argmax_above_threshold() {
        let data = synth(21, 4, 2, [10.0, 20.0, 0.7]);
        // No anchor above threshold → None.
        assert!(best_pose_anchor(&data, 4, 0.95).is_none());
        let (a, s) = best_pose_anchor(&data, 4, 0.25).unwrap();
        assert_eq!(a, 2);
        assert!((s - 0.9).abs() < 1e-6);
    }

    #[test]
    fn anchor_count_rejects_wrong_channel_tiling() {
        assert_eq!(pose_anchor_count(pose_channels(21) * 7, 21), Some(7));
        assert_eq!(pose_anchor_count(pose_channels(21) * 7 + 1, 21), None);
        // A 17-kp length misread as a 21-kp model fails to tile unless by
        // coincidence (5+51=56 vs 5+63=68 share no small multiple).
        assert_eq!(pose_anchor_count(56 * 100, 21), None);
    }

    #[test]
    fn keypoints_unletterbox_back_to_frame_pixels() {
        // Whole-frame convention: 640 model input, 1280×640 frame →
        // r = 0.5, pad_x = 160, pad_y = 0. A keypoint at model (320, 100)
        // maps to frame ((320−160)/0.5, 100/0.5) = (320, 200).
        let data = synth(17, 1, 0, [320.0, 100.0, 0.8]);
        let kps = anchor_keypoints(&data, 1, 0, 17, 160.0, 0.0, 0.5);
        assert_eq!(kps.len(), 17);
        assert!((kps[0][0] - 320.0).abs() < 1e-4);
        assert!((kps[0][1] - 200.0).abs() < 1e-4);
        assert!((kps[0][2] - 0.8).abs() < 1e-6);
        let kps = anchor_keypoints(&data, 1, 0, 17, 0.0, 0.0, 1.0);
        assert!((kps[0][0] - 320.0).abs() < 1e-4);
        assert!((kps[0][1] - 100.0).abs() < 1e-4);
    }

    #[test]
    fn fill_crop_tensor_maps_crop_origin_to_tensor_origin() {
        // 4×2 grey frame (value 255), crop (1, 0, 2) sampled into 2×2:
        // every output pixel samples inside the crop → 1.0 everywhere.
        let w = 4u32;
        let h = 2u32;
        let rgb = vec![255u8; (w * h * 3) as usize];
        let mut tensor = Array4::<f32>::zeros((1, 3, 2, 2));
        fill_crop_tensor(&rgb, w, h, (1.0, 0.0, 2.0), &mut tensor);
        for v in tensor.iter() {
            assert!((v - 1.0).abs() < 1e-4, "got {v}");
        }
        // Crop entirely past the right edge: every sample lands outside
        // the frame → zeros.
        let mut tensor = Array4::<f32>::zeros((1, 3, 2, 2));
        fill_crop_tensor(&rgb, w, h, (4.0, 0.0, 2.0), &mut tensor);
        for v in tensor.iter() {
            assert!((v - 0.0).abs() < 1e-4, "outside crop must read zero, got {v}");
        }
    }
}
