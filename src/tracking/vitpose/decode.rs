//! ViTPose heatmap decoder.
//!
//! Input: one tensor of shape `(133, H, W)` (the batch dim already
//! stripped) where each `(H, W)` slice is the unnormalised heatmap
//! logit for a single COCO-Wholebody keypoint. ViTPose-S emits
//! `H = 64, W = 48` for our 256×192 input (input ÷ 4) but the
//! decode is shape-agnostic.
//!
//! Decode = per-channel argmax over the spatial grid, plus the
//! standard MMPose-style ±0.25 sub-pixel offset toward the brighter
//! immediate neighbour on each axis. The peak value itself doubles
//! as the per-joint score; we clamp into `[0, 1]` so downstream
//! gates that were tuned for sigmoid'd RTMW3D scores still bite at
//! the same threshold.

pub const NUM_JOINTS: usize = 133;

#[derive(Clone, Copy, Debug, Default)]
pub struct DecodedJoint2d {
    /// Image-relative `[0, 1]`, x-right (within the inference crop —
    /// caller is expected to remap to whole-frame coords).
    pub nx: f32,
    /// Image-relative `[0, 1]`, y-down.
    pub ny: f32,
    /// Per-joint confidence in `[0, 1]`.
    pub score: f32,
}

/// Decode `(num_joints, h, w)` heatmaps. `data` is row-major
/// (`joint * h * w + y * w + x`). Returns one [`DecodedJoint2d`] per
/// joint in declared order. Joints whose argmax sits on the grid
/// border or whose peak is finite-but-non-positive get `score = 0`
/// so the downstream visibility floor treats them as missing.
pub fn decode_heatmaps(data: &[f32], h: usize, w: usize) -> Vec<DecodedJoint2d> {
    let mut out = Vec::with_capacity(NUM_JOINTS);
    let stride = h * w;
    if data.len() < NUM_JOINTS * stride || h == 0 || w == 0 {
        for _ in 0..NUM_JOINTS {
            out.push(DecodedJoint2d::default());
        }
        return out;
    }
    let inv_w = 1.0 / w as f32;
    let inv_h = 1.0 / h as f32;
    for j in 0..NUM_JOINTS {
        let base = j * stride;
        let slice = &data[base..base + stride];
        let (mut peak_x, mut peak_y, peak_v) = argmax_2d(slice, w, h);

        // Sub-pixel refinement: MMPose's TopdownHeatmapDecoder uses
        // a ±0.25 nudge toward the brighter immediate neighbour on
        // each axis. We only apply it when the neighbour exists and
        // is strictly larger than the peak's opposite-side neighbour
        // — at the grid edge we skip and accept the integer bin.
        if peak_x > 0 && peak_x + 1 < w {
            let left = slice[peak_y * w + peak_x - 1];
            let right = slice[peak_y * w + peak_x + 1];
            let dx = if right > left { 0.25 } else if left > right { -0.25 } else { 0.0 };
            peak_x_subpix(&mut peak_x, dx);
        }
        if peak_y > 0 && peak_y + 1 < h {
            let up = slice[(peak_y - 1) * w + peak_x];
            let down = slice[(peak_y + 1) * w + peak_x];
            let dy = if down > up { 0.25 } else if up > down { -0.25 } else { 0.0 };
            peak_y_subpix(&mut peak_y, dy);
        }

        // Convert bin centre to image-relative [0, 1]. `+0.5` shifts
        // to the cell centre; dividing by grid size yields the
        // normalised crop coord. Caller remaps to whole-frame.
        let nx = ((peak_x as f32) + 0.5) * inv_w;
        let ny = ((peak_y as f32) + 0.5) * inv_h;
        let score = peak_v.clamp(0.0, 1.0);
        out.push(DecodedJoint2d { nx, ny, score });
    }
    out
}

#[inline]
fn argmax_2d(slice: &[f32], w: usize, h: usize) -> (usize, usize, f32) {
    let mut max_v = f32::NEG_INFINITY;
    let mut mx = 0usize;
    let mut my = 0usize;
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let v = slice[row + x];
            if v > max_v {
                max_v = v;
                mx = x;
                my = y;
            }
        }
    }
    (mx, my, max_v)
}

/// Helper to fold the float `dx` back into the integer-bin peak via
/// rounding-toward-half. We bias the *integer* peak by the rounded
/// nudge so downstream consumers that already store peak as `usize`
/// keep their semantics — the +0.5 cell-centre shift below already
/// gives sub-bin resolution in image coords.
#[inline]
fn peak_x_subpix(peak_x: &mut usize, dx: f32) {
    if dx >= 0.5 {
        *peak_x = peak_x.saturating_add(1);
    } else if dx <= -0.5 {
        *peak_x = peak_x.saturating_sub(1);
    }
}

#[inline]
fn peak_y_subpix(peak_y: &mut usize, dy: f32) {
    if dy >= 0.5 {
        *peak_y = peak_y.saturating_add(1);
    } else if dy <= -0.5 {
        *peak_y = peak_y.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_emits_one_joint_per_keypoint() {
        let h = 4;
        let w = 4;
        let data = vec![0.0_f32; NUM_JOINTS * h * w];
        let joints = decode_heatmaps(&data, h, w);
        assert_eq!(joints.len(), NUM_JOINTS);
    }

    #[test]
    fn argmax_of_single_peak() {
        let h = 4;
        let w = 4;
        let mut data = vec![0.0_f32; NUM_JOINTS * h * w];
        // Place a 0.9 peak at (y=2, x=3) for joint 0.
        data[2 * w + 3] = 0.9;
        let joints = decode_heatmaps(&data, h, w);
        let j0 = joints[0];
        assert!((j0.score - 0.9).abs() < 1e-6);
        // (x + 0.5) / w = 3.5 / 4 = 0.875
        assert!((j0.nx - 0.875).abs() < 1e-6);
        // (y + 0.5) / h = 2.5 / 4 = 0.625
        assert!((j0.ny - 0.625).abs() < 1e-6);
    }
}
