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
use super::decode::DecodedJoint;

/// Remap decoded crop-space joints back to original-frame normalised
/// coords. `nx`/`ny` get the affine crop→frame transform; `nz` is
/// multiplied by the caller-supplied `z_gain`.
///
/// The nz scaling is the x/y/z unit contract: RTMW3D's SimCC-z is a
/// *metric* root-relative axis (`RTMW3D_Z_RANGE` = 2.17 m, rtmlib
/// convention) while Δnx/Δny shrink with the subject's apparent size,
/// so nz needs an apparent-size gain to stay proportionate to x/y in
/// frame space. For a crop that tracks the subject, `ch / height` is
/// that gain (crop height ∝ apparent subject size). For the
/// whole-frame letterbox fallback the crop height says nothing about
/// the subject — that is why the gain is a separate parameter: the
/// caller passes the last *tracked* gain for continuity instead of the
/// fallback's constant `≈2.37` (see the call site in `mod.rs`). Only
/// relative Δnz matters downstream (`to_source` subtracts the anchor's
/// nz), so a plain multiply is sufficient.
pub(super) fn remap_crop_joints(
    joints: &mut [DecodedJoint],
    ox: f32,
    oy: f32,
    cw: f32,
    ch: f32,
    width: u32,
    height: u32,
    z_gain: f32,
) {
    let inv_w = 1.0 / (width.max(1) as f32);
    let inv_h = 1.0 / (height.max(1) as f32);
    for j in joints.iter_mut() {
        j.nx = (ox + j.nx * cw) * inv_w;
        j.ny = (oy + j.ny * ch) * inv_h;
        j.nz *= z_gain;
        j.sx *= cw * inv_w;
        j.sy *= ch * inv_h;
    }
}

/// Apparent-size z gain for a subject-tracking crop: crop height as a
/// fraction of frame height. Valid ONLY when the crop follows the
/// subject (bbox / self-track) — see [`remap_crop_joints`].
pub(super) fn tracked_z_gain(ch: f32, height: u32) -> f32 {
    ch / (height.max(1) as f32)
}

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
pub(super) fn pad_bbox_to_aspect(bbox: &PersonBbox, pad_ratio: f32) -> (f32, f32, f32, f32) {
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
pub(super) fn crop_rgb_padded(
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

    // Column invariants of the bilinear resample, hoisted out of the row
    // loop: `scale_x` is constant, so every dst column has fixed source
    // neighbours and weights. Per-pixel arithmetic (weight products, the
    // channel sums and their order) is unchanged — the outputs stay
    // bit-identical to the un-hoisted loop.
    let mut col_w = Vec::with_capacity(dst_w * 4);
    let mut col_ix = Vec::with_capacity(dst_w * 2);
    for dx in 0..dst_w {
        let fx = ((dx as f32) + 0.5) * scale_x - 0.5;
        let x0f = fx.floor();
        let wx1 = (fx - x0f).clamp(0.0, 1.0);
        let wx0 = 1.0 - wx1;
        let x0 = (x0f as i32).clamp(0, max_x as i32) as usize;
        let x1 = ((x0f as i32) + 1).clamp(0, max_x as i32) as usize;
        col_ix.push((x0, x1));
        col_w.push((wx0, wx1));
    }

    let samples = tensor.as_slice_mut().unwrap();
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
            let (x0, x1) = col_ix[dx];
            let (wx0, wx1) = col_w[dx];
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
            // Row-major NCHW: `Array4::zeros` is contiguous, so the flat
            // write is `((c * dst_h) + dy) * dst_w + dx`.
            let o = (dy * dst_w + dx) as usize;
            samples[o] = (r - MEAN_RGB[0]) * inv_std_r;
            samples[dst_h * dst_w + o] = (g - MEAN_RGB[1]) * inv_std_g;
            samples[2 * dst_h * dst_w + o] = (b - MEAN_RGB[2]) * inv_std_b;
        }
    }
    tensor
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dj(nx: f32, ny: f32, nz: f32) -> DecodedJoint {
        DecodedJoint {
            nx,
            ny,
            nz,
            score: 0.9,
            sx: 0.002,
            sy: 0.002,
            ..Default::default()
        }
    }

    /// Reference copy of the pre-hoisting preprocess loop. The hoisted
    /// version must be BIT-identical: same per-pixel weight products and
    /// channel sums in the same order, only the per-column invariants
    /// (source indices, x weights) lifted out of the row loop.
    fn preprocess_reference(rgb: &[u8], src_w: u32, src_h: u32) -> Array4<f32> {
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
        if rgb.len() < stride * src_h_us {
            return tensor;
        }
        let inv_std_r = 1.0 / STD_RGB[0];
        let inv_std_g = 1.0 / STD_RGB[1];
        let inv_std_b = 1.0 / STD_RGB[2];
        for dy in 0..dst_h {
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

    /// The hoisted resample must produce bit-identical tensors to the
    /// un-hoisted loop, including the sub-pixel edge cases (1-px source,
    /// extreme upscale, non-integer scale factors).
    #[test]
    fn preprocess_hoist_is_bit_identical() {
        // Deterministic pseudo-random RGB covering the full byte range.
        let mut seed = 0x12345678u64;
        let mut next = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u8
        };
        for (w, h) in [(1u32, 1u32), (2, 3), (63, 47), (288, 384), (640, 480), (1920, 1080)] {
            let rgb: Vec<u8> = (0..(w as usize * h as usize * 3)).map(|_| next()).collect();
            let a = preprocess(&rgb, w, h);
            let b = preprocess_reference(&rgb, w, h);
            assert_eq!(
                a.as_slice().unwrap(),
                b.as_slice().unwrap(),
                "preprocess mismatch at {w}x{h}"
            );
        }
    }

    /// The same physical pose decoded from a small crop and from a
    /// large crop must remap to identical relative x:y:z proportions —
    /// the distance-invariance contract that motivated the nz scaling.
    #[test]
    fn remap_scales_z_by_the_same_ratio_as_y() {
        let (w, h) = (1280u32, 720u32);
        // Two joints separated in all three axes, in crop space.
        let base = [dj(0.30, 0.30, 0.40), dj(0.70, 0.80, 0.65)];

        // Near subject: crop is most of the frame.
        let mut near = base;
        remap_crop_joints(
            &mut near,
            100.0,
            0.0,
            540.0,
            720.0,
            w,
            h,
            tracked_z_gain(720.0, h),
        );
        // Far subject: same pose, crop four times smaller.
        let mut far = base;
        remap_crop_joints(
            &mut far,
            500.0,
            200.0,
            135.0,
            180.0,
            w,
            h,
            tracked_z_gain(180.0, h),
        );

        let ratios = |pair: &[DecodedJoint; 2]| {
            let dx = pair[1].nx - pair[0].nx;
            let dy = pair[1].ny - pair[0].ny;
            let dz = pair[1].nz - pair[0].nz;
            (dz / dy, dx / dy)
        };
        let (near_zy, near_xy) = ratios(&near);
        let (far_zy, far_xy) = ratios(&far);
        assert!(
            (near_zy - far_zy).abs() < 1e-5,
            "z:y proportion must be crop-size invariant, near {near_zy} far {far_zy}"
        );
        assert!(
            (near_xy - far_xy).abs() < 1e-5,
            "x:y proportion must be crop-size invariant, near {near_xy} far {far_xy}"
        );
    }

    /// Identity crop (whole frame, no offset) must leave nx/ny/nz
    /// untouched — the zero-cost baseline the old code special-cased.
    #[test]
    fn remap_identity_crop_is_a_noop() {
        let (w, h) = (640u32, 480u32);
        let mut joints = [dj(0.25, 0.75, 0.5)];
        remap_crop_joints(
            &mut joints,
            0.0,
            0.0,
            640.0,
            480.0,
            w,
            h,
            tracked_z_gain(480.0, h),
        );
        assert!((joints[0].nx - 0.25).abs() < 1e-6);
        assert!((joints[0].ny - 0.75).abs() < 1e-6);
        assert!((joints[0].nz - 0.5).abs() < 1e-6);
    }

    /// Full-frame letterbox bbox: `pad_bbox_to_aspect` with zero pad on
    /// a 16:9 frame must produce a portrait 288:384 virtual crop that
    /// contains the whole frame (no squash).
    #[test]
    fn whole_frame_aspect_pad_contains_frame_without_squash() {
        let bbox = PersonBbox {
            x1: 0.0,
            y1: 0.0,
            x2: 1280.0,
            y2: 720.0,
            score: 1.0,
        };
        let (x1, y1, x2, y2) = pad_bbox_to_aspect(&bbox, 0.0);
        let cw = x2 - x1;
        let ch = y2 - y1;
        let aspect = cw / ch;
        assert!(
            (aspect - INPUT_W as f32 / INPUT_H as f32).abs() < 1e-3,
            "virtual crop must match the model input aspect, got {aspect}"
        );
        assert!(x1 <= 0.0 && x2 >= 1280.0 && y1 <= 0.0 && y2 >= 720.0);
    }
}
