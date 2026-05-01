//! MoGe-2 4-output postprocess.
//!
//! The exported model emits raw `forward()` outputs only — the
//! upstream `infer()` does focal-and-shift recovery via a 2-DoF
//! Levenberg-Marquardt fit on a downsampled grid before producing the
//! final metric point cloud. We deliberately *skip* that recovery for
//! v1 and just multiply the affine point map by the predicted
//! `metric_scale`. The resulting points have a small absolute z bias
//! (the unrecovered shift), but [`super::super::cigpose_metric_depth`]
//! anchors every joint to the hip mid-point and the avatar solver
//! consumes only relative bone directions, so the bias cancels out
//! within one frame. Worth revisiting if absolute-depth-sensitive
//! features (e.g. ground-plane detection) get added later.

use super::consts::MASK_THRESHOLD;

/// Decoded MoGe frame: per-pixel metric point cloud, in metres, in
/// MoGe's input pixel grid (`width × height`). Invalid pixels (mask
/// below the threshold or non-finite output) are stored as `NaN` in
/// `points_m` so the consumer's local-window median sampling
/// naturally rejects them. Per-pixel metric Z is `points_m[i][2]` —
/// no separate depth map is kept since the only current consumer
/// samples the full xyz, not depth alone.
pub struct MoGeFrame {
    pub width: u32,
    pub height: u32,
    pub points_m: Vec<[f32; 3]>,
}

/// Post-process the four MoGe outputs into a [`MoGeFrame`].
///
/// * `points_bhw3` — `(1, H, W, 3)` row-major affine point map (xyz per pixel).
/// * `mask_bhw`   — `(1, H, W)` row-major sigmoid mask in `[0, 1]`.
/// * `metric_scale` — scalar (already `exp()`-applied by `forward()`).
pub fn build_frame(
    points_bhw3: &[f32],
    mask_bhw: &[f32],
    metric_scale: f32,
    width: u32,
    height: u32,
) -> Option<MoGeFrame> {
    let pixel_count = (width as usize) * (height as usize);
    if pixel_count == 0 {
        return None;
    }
    if points_bhw3.len() < pixel_count * 3 || mask_bhw.len() < pixel_count {
        return None;
    }
    if !metric_scale.is_finite() || metric_scale <= 0.0 {
        return None;
    }

    let mut points_m = Vec::with_capacity(pixel_count);
    // Index-driven loop because each iteration touches three slices
    // (mask + 3 floats per pixel from points); enumerate-on-one would
    // still require indexed access for the other.
    #[allow(clippy::needless_range_loop)]
    for i in 0..pixel_count {
        let valid = mask_bhw[i] >= MASK_THRESHOLD;
        let base = i * 3;
        let px = points_bhw3[base];
        let py = points_bhw3[base + 1];
        let pz = points_bhw3[base + 2];
        if valid && px.is_finite() && py.is_finite() && pz.is_finite() {
            points_m.push([px * metric_scale, py * metric_scale, pz * metric_scale]);
        } else {
            points_m.push([f32::NAN, f32::NAN, f32::NAN]);
        }
    }

    Some(MoGeFrame {
        width,
        height,
        points_m,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_frame_applies_scale_and_mask() {
        // 2×1 frame, two pixels.
        let points = [
            1.0, 2.0, 3.0, // pixel 0
            4.0, 5.0, 6.0, // pixel 1
        ];
        let mask = [0.9, 0.1]; // pixel 1 is below threshold → NaN
        let frame = build_frame(&points, &mask, 2.0, 2, 1).unwrap();
        assert_eq!(frame.points_m[0], [2.0, 4.0, 6.0]);
        assert!(frame.points_m[1][0].is_nan());
    }

    #[test]
    fn build_frame_rejects_zero_scale() {
        let points = [0.0_f32; 3];
        let mask = [1.0_f32];
        assert!(build_frame(&points, &mask, 0.0, 1, 1).is_none());
    }
}
