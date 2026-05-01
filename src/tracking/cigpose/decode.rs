//! SimCC X / Y decode for CIGPose.
//!
//! Two-axis variant of RTMW3D's three-axis SimCC decode: argmax along
//! each axis, divide by `split_ratio`-derived bin count to recover
//! normalised image-relative coords, and pass the per-axis logit peak
//! through sigmoid so the score lives in `[0, 1]` (matches the
//! convention `KEYPOINT_VISIBILITY_FLOOR` was tuned against).
//!
//! The reference Python loader compares the raw logit min directly to
//! a threshold (~0.6), but using sigmoid here keeps the visibility
//! floor consistent across providers — the solver's
//! `joint_confidence_threshold` slider is the single user-facing knob.

pub const NUM_JOINTS: usize = 133;

#[derive(Clone, Copy, Debug, Default)]
pub struct DecodedJoint2d {
    /// Image-relative `[0, 1]`, x-right.
    pub nx: f32,
    /// Image-relative `[0, 1]`, y-down.
    pub ny: f32,
    /// Sigmoid of the per-axis logit min — `[0, 1]`.
    pub score: f32,
}

#[inline]
fn argmax_with_score(slice: &[f32]) -> (usize, f32) {
    let mut max_v = f32::NEG_INFINITY;
    let mut max_i = 0usize;
    for (i, &v) in slice.iter().enumerate() {
        if v > max_v {
            max_v = v;
            max_i = i;
        }
    }
    (max_i, max_v)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn decode_simcc_xy(
    simcc_x: &[f32],
    simcc_y: &[f32],
    x_bins: usize,
    y_bins: usize,
) -> Vec<DecodedJoint2d> {
    let mut out = Vec::with_capacity(NUM_JOINTS);
    let inv_x = 1.0 / x_bins as f32;
    let inv_y = 1.0 / y_bins as f32;
    for j in 0..NUM_JOINTS {
        let x_off = j * x_bins;
        let y_off = j * y_bins;
        if x_off + x_bins > simcc_x.len() || y_off + y_bins > simcc_y.len() {
            out.push(DecodedJoint2d::default());
            continue;
        }
        let x_slice = &simcc_x[x_off..x_off + x_bins];
        let y_slice = &simcc_y[y_off..y_off + y_bins];
        let (xi, xs) = argmax_with_score(x_slice);
        let (yi, ys) = argmax_with_score(y_slice);
        let score = sigmoid(xs.min(ys));
        out.push(DecodedJoint2d {
            nx: xi as f32 * inv_x,
            ny: yi as f32 * inv_y,
            score,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_of_one_hot() {
        let x = [0.0_f32, 0.5, 5.0, 0.1];
        let (i, s) = argmax_with_score(&x);
        assert_eq!(i, 2);
        assert!((s - 5.0).abs() < 1e-6);
    }

    #[test]
    fn decode_simcc_xy_one_joint_centre() {
        // Single joint, peak at the centre of each axis.
        let x_bins = 8;
        let y_bins = 4;
        let mut x = vec![0.0_f32; NUM_JOINTS * x_bins];
        let mut y = vec![0.0_f32; NUM_JOINTS * y_bins];
        x[0 * x_bins + 4] = 10.0;
        y[0 * y_bins + 2] = 10.0;
        let joints = decode_simcc_xy(&x, &y, x_bins, y_bins);
        assert_eq!(joints.len(), NUM_JOINTS);
        assert!((joints[0].nx - 0.5).abs() < 1e-6);
        assert!((joints[0].ny - 0.5).abs() < 1e-6);
        // sigmoid(10) ≈ 0.99995
        assert!(joints[0].score > 0.999);
    }
}
