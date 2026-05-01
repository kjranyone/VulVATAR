//! Frame preprocessing for RTMW3D inference.
//!
//! Two related concerns live here:
//!
//! * The **YOLOX crop** helpers (`pad_and_clamp_bbox`, `crop_rgb`) take
//!   a raw `PersonBbox` from the YOLOX detector and produce a
//!   contiguous RGB slice that gets fed into the model. The crop is
//!   sized to RTMW3D's 288:384 input aspect so the squash inside
//!   `preprocess` is minimal.
//! * The **288×384 NCHW resize** (`preprocess`) does OpenCV-style
//!   bilinear resampling + ImageNet mean/std normalization. Aspect is
//!   squashed (no letterbox) — the model is robust to it for
//!   centred-subject inputs and the YOLOX crop above already brought
//!   the source aspect close to the model's.

use ndarray::Array4;

use super::consts::{INPUT_H, INPUT_W, MEAN_RGB, STD_RGB};
use super::super::yolox::PersonBbox;

// ---------------------------------------------------------------------------
// YOLOX crop helpers
// ---------------------------------------------------------------------------

/// Pad the YOLOX-detected bbox, adjust it toward RTMW3D's 288:384 input
/// aspect, and clamp to the frame. Shared with the CIGPose+MoGe-2
/// provider, which also feeds CIGPose at 288×384 — keep the same
/// aspect target so the helper stays single-sourced.
pub(in crate::tracking) fn pad_and_clamp_bbox(
    bbox: &PersonBbox,
    width: u32,
    height: u32,
    pad_ratio: f32,
) -> (u32, u32, u32, u32) {
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
    let half_w = bw * 0.5;
    let half_h = bh * 0.5;
    let x1 = (cx - half_w).max(0.0).floor() as u32;
    let y1 = (cy - half_h).max(0.0).floor() as u32;
    let x2 = (cx + half_w).min(width as f32).ceil() as u32;
    let y2 = (cy + half_h).min(height as f32).ceil() as u32;
    (x1, y1, x2, y2)
}

/// Copy the rectangle `[(ox, oy), (ox+cw, oy+ch))` out of `rgb`
/// (`width × height × 3` u8) into a contiguous tightly-packed buffer.
/// Caller has already clamped the rectangle into the source bounds.
/// Shared with the CIGPose+MoGe-2 provider for the same person-crop
/// flow.
pub(in crate::tracking) fn crop_rgb(
    rgb: &[u8],
    width: u32,
    _height: u32,
    ox: u32,
    oy: u32,
    cw: u32,
    ch: u32,
) -> Vec<u8> {
    let stride = (width as usize) * 3;
    let crop_stride = (cw as usize) * 3;
    let mut out = Vec::with_capacity(crop_stride * (ch as usize));
    for y in 0..(ch as usize) {
        let src_row = (oy as usize + y) * stride + (ox as usize) * 3;
        out.extend_from_slice(&rgb[src_row..src_row + crop_stride]);
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
