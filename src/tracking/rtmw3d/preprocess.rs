//! Frame preprocessing for RTMW3D inference.
//!
//! Two related concerns live here:
//!
//! * The **YOLOX crop** helpers (`pad_bbox_to_aspect`, `crop_rgb_padded`)
//!   take a raw `PersonBbox` and produce a contiguous RGB slice at
//!   RTMW3D's exact 288:384 input aspect. The crop is allowed to run past
//!   the frame edges (zero-padded) so the subject's aspect is preserved —
//!   the mmpose `TopdownAffine` contract. The crop's SEED (the
//!   self-tracking bbox in `mod.rs`) is clamped to the frame so this
//!   beyond-frame crop cannot feed an unbounded zoom-out loop.
//! * The **288×384 NCHW resize** (`preprocess`) does OpenCV-style
//!   bilinear resampling + ImageNet mean/std normalization. Because the
//!   crop is already 288:384, the resize is a pure scale with no aspect
//!   squash (the earlier clamp-to-frame crop left a landscape rectangle
//!   that squashed a frame-filling subject and flattened the heatmaps).

use ndarray::Array4;

use super::super::yolox::PersonBbox;
use super::consts::{INPUT_H, INPUT_W, MEAN_RGB, STD_RGB};

// ---------------------------------------------------------------------------
// YOLOX crop helpers
// ---------------------------------------------------------------------------

/// Aspect-correct the padded bbox to the model's input ratio WITHOUT clamping
/// to the frame — out-of-frame regions are zero-padded later by
/// [`crop_rgb_padded`]. This is the mmpose `TopdownAffine` contract the model
/// was trained on: the crop preserves the subject's aspect so the squash-resize
/// into `INPUT_W × INPUT_H` does not horizontally compress a frame-filling
/// subject. A clamp-to-frame crop grows the deficient dimension to hit the
/// aspect and THEN clamps, which for a 16:9 desk-mirror frame leaves a landscape
/// crop (aspect ~1.3) that squashes the subject to ~55% width — flattening
/// RTMW3D's heatmaps to uniform low confidence and edge-clamped elbows. Returns
/// `(x1, y1, x2, y2)` in source pixels, possibly negative or past `width`/
/// `height`. The SEED bbox is clamped upstream (`derive_self_track_bbox`) so
/// this beyond-frame crop cannot drive an unbounded zoom-out feedback loop.
pub(in crate::tracking) fn pad_bbox_to_aspect(bbox: &PersonBbox, pad_ratio: f32) -> (f32, f32, f32, f32) {
    let mut bw = (bbox.x2 - bbox.x1).max(1.0) * (1.0 + pad_ratio);
    let mut bh = (bbox.y2 - bbox.y1).max(1.0) * (1.0 + pad_ratio);
    let cx = (bbox.x1 + bbox.x2) * 0.5;
    let cy = (bbox.y1 + bbox.y2) * 0.5;
    let target_aspect = INPUT_W as f32 / INPUT_H as f32;
    if bw > bh * target_aspect {
        bh = bw / target_aspect;
    } else {
        bw = bh * target_aspect;
    }
    let (half_w, half_h) = (bw * 0.5, bh * 0.5);
    (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
}

/// Copy an aspect-correct rectangle that MAY extend past the frame bounds into a
/// tightly-packed `cw × ch × 3` buffer, zero-padding (black) any out-of-frame
/// pixels. `ox`/`oy` are the top-left in source pixels (may be negative). Black
/// matches mmpose's `warpAffine` border and normalises to a constant the model
/// learned to ignore. One `copy_from_slice` per row over the in-bounds x-span.
pub(in crate::tracking) fn crop_rgb_padded(
    rgb: &[u8],
    width: u32,
    height: u32,
    ox: i32,
    oy: i32,
    cw: u32,
    ch: u32,
) -> Vec<u8> {
    let stride = width as usize * 3;
    let crop_stride = cw as usize * 3;
    let mut out = vec![0u8; crop_stride * ch as usize];
    let x_start = ox.max(0);
    let x_end = (ox + cw as i32).min(width as i32);
    if x_end <= x_start {
        return out;
    }
    let n = ((x_end - x_start) * 3) as usize;
    for dy in 0..ch as i32 {
        let sy = oy + dy;
        if sy < 0 || sy >= height as i32 {
            continue;
        }
        let src = sy as usize * stride + x_start as usize * 3;
        let dst = dy as usize * crop_stride + ((x_start - ox) as usize) * 3;
        out[dst..dst + n].copy_from_slice(&rgb[src..src + n]);
    }
    out
}

// ---------------------------------------------------------------------------
// 288×384 NCHW resize + normalize
// ---------------------------------------------------------------------------

/// Resize source RGB to `INPUT_W × INPUT_H` NCHW with ImageNet
/// normalization. Aspect is squashed (no letterbox); the model is
/// tolerant of it for centred-subject input. Bilinear sampling so
/// small / distant subjects don't lose keypoint detail to nearest's
/// quantisation.
pub(super) fn preprocess(rgb: &[u8], src_w: u32, src_h: u32) -> Array4<f32> {
    let dst_w = INPUT_W as usize;
    let dst_h = INPUT_H as usize;
    let mut tensor = Array4::<f32>::zeros((1, 3, dst_h, dst_w));
    if src_w == 0 || src_h == 0 {
        return tensor;
    }
    let src_w_us = src_w as usize;
    let src_h_us = src_h as usize;
    let max_x = src_w_us.saturating_sub(1);
    let max_y = src_h_us.saturating_sub(1);
    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;
    let stride = src_w_us * 3;
    let expected_len = stride * src_h_us;
    if rgb.len() < expected_len {
        return tensor;
    }

    let inv_std_r = 1.0 / STD_RGB[0];
    let inv_std_g = 1.0 / STD_RGB[1];
    let inv_std_b = 1.0 / STD_RGB[2];

    for dy in 0..dst_h {
        // OpenCV-style pixel-centre sampling: dst (dy + 0.5) maps to
        // src ((dy + 0.5) * scale_y - 0.5).
        let fy = ((dy as f32) + 0.5) * scale_y - 0.5;
        let y0f = fy.floor();
        let wy1 = (fy - y0f).clamp(0.0, 1.0);
        let wy0 = 1.0 - wy1;
        let y0 = (y0f as i32).clamp(0, max_y as i32) as usize;
        let y1 = ((y0f as i32) + 1).clamp(0, max_y as i32) as usize;
        let row0 = y0 * stride;
        let row1 = y1 * stride;
        for dx in 0..dst_w {
            let fx = ((dx as f32) + 0.5) * scale_x - 0.5;
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
            tensor[(0, 0, dy, dx)] = (r - MEAN_RGB[0]) * inv_std_r;
            tensor[(0, 1, dy, dx)] = (g - MEAN_RGB[1]) * inv_std_g;
            tensor[(0, 2, dy, dx)] = (b - MEAN_RGB[2]) * inv_std_b;
        }
    }
    tensor
}
