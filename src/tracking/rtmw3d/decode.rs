//! SimCC heatmap decode — argmax + sigmoid + per-joint score
//! aggregation. Pure-function module: takes the three raw output
//! slices from RTMW3D and returns one [`DecodedJoint`] per
//! COCO-Wholebody index.

use super::consts::INPUT_H;

pub(super) const NUM_JOINTS: usize = 133;
pub(super) const SIMCC_X_BINS: usize = 576;
pub(super) const SIMCC_Y_BINS: usize = 768;
pub(super) const SIMCC_Z_BINS: usize = 576;

pub(super) const RTMW3D_Z_RANGE: f32 = 2.1744869;
pub(super) const RTMW3D_SOURCE_Z_SCALE: f32 =
    (SIMCC_Z_BINS as f32 / INPUT_H as f32) * RTMW3D_Z_RANGE;

/// One decoded keypoint in normalised model space:
/// `nx, ny ∈ [0, 1]` (image-relative, NY top-to-bottom),
/// `nz ∈ [0, 1]` (model depth axis), `score` is the per-joint
/// confidence after sigmoid.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct DecodedJoint {
    pub(super) nx: f32,
    pub(super) ny: f32,
    pub(super) nz: f32,
    pub(super) score: f32,
}

/// SimCC argmax with subpixel score. Returns `(bin, max_value)`.
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

pub(super) fn decode_simcc(
    simcc_x: &[f32],
    simcc_y: &[f32],
    simcc_z: &[f32],
) -> Vec<DecodedJoint> {
    let mut out = Vec::with_capacity(NUM_JOINTS);
    for j in 0..NUM_JOINTS {
        let x_slice = &simcc_x[j * SIMCC_X_BINS..(j + 1) * SIMCC_X_BINS];
        let y_slice = &simcc_y[j * SIMCC_Y_BINS..(j + 1) * SIMCC_Y_BINS];
        let z_slice = &simcc_z[j * SIMCC_Z_BINS..(j + 1) * SIMCC_Z_BINS];
        let (xi, xs) = argmax_with_score(x_slice);
        let (yi, ys) = argmax_with_score(y_slice);
        let (zi, _zs) = argmax_with_score(z_slice);
        // Per-joint score: take the smaller of x/y heatmap peaks
        // (z is depth — its peak does not localise the joint
        // detection, only its depth) and pass through sigmoid so
        // values land in `[0, 1]` for the threshold.
        let score = sigmoid(xs.min(ys));
        out.push(DecodedJoint {
            nx: xi as f32 / SIMCC_X_BINS as f32,
            ny: yi as f32 / SIMCC_Y_BINS as f32,
            nz: zi as f32 / SIMCC_Z_BINS as f32,
            score,
        });
    }
    out
}
