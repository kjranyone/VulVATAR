//! SimCC heatmap decode — sub-pixel argmax + sigmoid + per-joint score
//! aggregation. Pure-function module: takes the three raw output
//! slices from RTMW3D and returns one [`DecodedJoint`] per
//! COCO-Wholebody index.

pub(in crate::tracking) const NUM_JOINTS: usize = 133;
pub(super) const SIMCC_X_BINS: usize = 576;
pub(super) const SIMCC_Y_BINS: usize = 768;
pub(super) const SIMCC_Z_BINS: usize = 576;

/// One decoded keypoint in normalised model space:
/// `nx, ny ∈ [0, 1]` (image-relative, NY top-to-bottom),
/// `nz ∈ [0, 1]` (model depth axis), `score` is the per-joint
/// confidence after sigmoid of the x/y heatmap peaks.
#[derive(Clone, Copy, Debug, Default)]
pub struct DecodedJoint {
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub score: f32,
    /// Localisation σ of the x / y peak in the same normalised units as
    /// `nx` / `ny` (posterior std of the softmax-normalised SimCC
    /// distribution in a window around the peak). A sharp peak yields
    /// ~1 bin; a flat or bimodal distribution yields tens of bins. This
    /// is the per-keypoint uncertainty the fusion estimator consumes.
    pub sx: f32,
    pub sy: f32,
    /// Peak-shape statistics of the raw SimCC vectors, kept for the
    /// visibility calibration (see `fusion::visibility`). All are
    /// scale-invariant in the logit values.
    ///
    /// Ratio of the strongest response OUTSIDE the main peak's window
    /// (±`NOMINAL_FWHM_BINS`) to the main peak: ≈0 for a unimodal
    /// confident joint, →1 for a bimodal / flat one.
    pub second_x: f32,
    pub second_y: f32,
    /// Fraction of bins at or above half of the peak value. A nominal
    /// peak covers ≈ FWHM / bins; a flat response covers most of the axis.
    pub half_x: f32,
    pub half_y: f32,
    /// Sigmoid of the z-axis peak (the depth head's own confidence).
    pub zscore: f32,
}

/// SimCC argmax. Returns `(bin, max_value)`.
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

/// Sub-pixel peak refinement: fit a parabola through the argmax bin and
/// its two neighbours and return the fractional bin of the vertex,
/// clamped to ±0.5 bins. Standard mmpose-style post-processing: the raw
/// argmax quantises to 1/576 of the crop, which on a small crop becomes
/// a visible per-frame jitter step that the downstream One-Euro filters
/// otherwise have to absorb. Border bins (no two neighbours) are
/// returned unrefined.
#[inline]
fn refine_peak(slice: &[f32], peak: usize) -> f32 {
    if peak == 0 || peak + 1 >= slice.len() {
        return peak as f32;
    }
    let l = slice[peak - 1];
    let m = slice[peak];
    let r = slice[peak + 1];
    let denom = l - 2.0 * m + r;
    if denom.abs() < 1e-9 {
        return peak as f32;
    }
    let offset = (0.5 * (l - r) / denom).clamp(-0.5, 0.5);
    peak as f32 + offset
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Localisation σ (in bins) of the SimCC peak at `peak`, from the peak's
/// full width at half maximum: SimCC heads are trained against Gaussian
/// label vectors, so a confident joint yields a narrow bump (FWHM ≈ 2.355σ
/// of the label kernel, ~6 bins) while an ambiguous or occluded joint
/// yields a broad or multi-modal one. Scale-invariant in the output values
/// (unlike a softmax, whose sharpness depends on the unknown logit scale).
/// Floored at 0.5 bin, capped at `SIGMA_WINDOW`.
const SIGMA_WINDOW: usize = 96;
#[inline]
fn peak_sigma_bins(slice: &[f32], peak: usize) -> f32 {
    let m = slice[peak];
    if !(m > 0.0) {
        return SIGMA_WINDOW as f32;
    }
    let half = 0.5 * m;
    // Walk left / right until the value drops below half max (or the
    // window ends). Sub-bin interpolation at the crossing.
    let mut left = 0.0f32;
    for k in 1..=SIGMA_WINDOW {
        if peak < k {
            left = k as f32 - 1.0;
            break;
        }
        let v = slice[peak - k];
        if v < half {
            let prev = slice[peak - k + 1];
            let frac = if prev > v {
                (prev - half) / (prev - v)
            } else {
                0.0
            };
            left = (k - 1) as f32 + frac.clamp(0.0, 1.0);
            break;
        }
        if k == SIGMA_WINDOW {
            left = SIGMA_WINDOW as f32;
        }
    }
    let mut right = 0.0f32;
    for k in 1..=SIGMA_WINDOW {
        if peak + k >= slice.len() {
            right = k as f32 - 1.0;
            break;
        }
        let v = slice[peak + k];
        if v < half {
            let prev = slice[peak + k - 1];
            let frac = if prev > v {
                (prev - half) / (prev - v)
            } else {
                0.0
            };
            right = (k - 1) as f32 + frac.clamp(0.0, 1.0);
            break;
        }
        if k == SIGMA_WINDOW {
            right = SIGMA_WINDOW as f32;
        }
    }
    let fwhm = left + right;
    // A confident RTMW3D peak measures ≈ NOMINAL_FWHM bins wide (the
    // label kernel); the sub-bin argmax of such a bump localises to a few
    // bins. Broader / plateaued bumps mean ambiguity: scale the base
    // precision by the squared broadness ratio.
    let ratio = (fwhm / NOMINAL_FWHM_BINS).clamp(1.0, 4.0);
    (BASE_SIGMA_BINS * ratio * ratio).min(SIGMA_WINDOW as f32)
}
/// Measured FWHM (bins) of a confident RTMW3D SimCC peak (2026-08-19,
/// D435 1280×720 replays: half-max at ±22 bins).
const NOMINAL_FWHM_BINS: f32 = 44.0;
/// Localisation σ (bins) of a nominal peak after sub-bin refinement.
const BASE_SIGMA_BINS: f32 = 2.5;

/// Peak-shape statistics for one SimCC axis: `(second, half)` where
/// `second` is the strongest response outside ±`NOMINAL_FWHM_BINS` of
/// the argmax divided by the argmax value (0 when the peak is ≤ 0), and
/// `half` is the fraction of all bins at or above half the peak value.
#[inline]
fn peak_shape(slice: &[f32], peak: usize) -> (f32, f32) {
    let m = slice[peak];
    if !(m > 0.0) {
        return (1.0, 1.0);
    }
    let w = NOMINAL_FWHM_BINS as usize;
    let lo = peak.saturating_sub(w);
    let hi = (peak + w).min(slice.len() - 1);
    let half = 0.5 * m;
    let mut second = f32::NEG_INFINITY;
    let mut n_half = 0usize;
    for (i, &v) in slice.iter().enumerate() {
        if v >= half {
            n_half += 1;
        }
        if (i < lo || i > hi) && v > second {
            second = v;
        }
    }
    let second = if second.is_finite() {
        (second / m).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (second, n_half as f32 / slice.len() as f32)
}

pub(super) fn decode_simcc(simcc_x: &[f32], simcc_y: &[f32], simcc_z: &[f32]) -> Vec<DecodedJoint> {
    let mut out = Vec::with_capacity(NUM_JOINTS);
    for j in 0..NUM_JOINTS {
        let x_slice = &simcc_x[j * SIMCC_X_BINS..(j + 1) * SIMCC_X_BINS];
        let y_slice = &simcc_y[j * SIMCC_Y_BINS..(j + 1) * SIMCC_Y_BINS];
        let z_slice = &simcc_z[j * SIMCC_Z_BINS..(j + 1) * SIMCC_Z_BINS];
        let (xi, xs) = argmax_with_score(x_slice);
        let (yi, ys) = argmax_with_score(y_slice);
        let (zi, zs) = argmax_with_score(z_slice);
        // Per-joint score: take the smaller of x/y heatmap peaks
        // (z is depth — its peak does not localise the joint
        // detection, only its depth) and pass through sigmoid so
        // values land in `[0, 1]`.
        let score = sigmoid(xs.min(ys));
        let (second_x, half_x) = peak_shape(x_slice, xi);
        let (second_y, half_y) = peak_shape(y_slice, yi);
        out.push(DecodedJoint {
            nx: refine_peak(x_slice, xi) / SIMCC_X_BINS as f32,
            ny: refine_peak(y_slice, yi) / SIMCC_Y_BINS as f32,
            nz: refine_peak(z_slice, zi) / SIMCC_Z_BINS as f32,
            score,
            sx: peak_sigma_bins(x_slice, xi) / SIMCC_X_BINS as f32,
            sy: peak_sigma_bins(y_slice, yi) / SIMCC_Y_BINS as f32,
            second_x,
            second_y,
            half_x,
            half_y,
            zscore: sigmoid(zs),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build one flat SimCC plane with a triangular peak at `peak`,
    /// whose neighbours are asymmetric so the true (continuous) maximum
    /// sits off-centre of the argmax bin.
    fn plane(bins: usize, peak: usize, left: f32, mid: f32, right: f32) -> Vec<f32> {
        let mut v = vec![-8.0; bins];
        v[peak - 1] = left;
        v[peak] = mid;
        v[peak + 1] = right;
        v
    }

    fn full(bins: usize, peak: usize, left: f32, mid: f32, right: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(NUM_JOINTS * bins);
        for _ in 0..NUM_JOINTS {
            out.extend_from_slice(&plane(bins, peak, left, mid, right));
        }
        out
    }

    #[test]
    fn subpixel_refinement_moves_toward_stronger_neighbour() {
        // Right neighbour stronger than left → vertex sits right of the
        // argmax bin centre; symmetric case stays exactly on it.
        let x = full(SIMCC_X_BINS, 100, 1.0, 4.0, 3.0);
        let y = full(SIMCC_Y_BINS, 200, 2.0, 4.0, 2.0);
        let z = full(SIMCC_Z_BINS, 300, 3.0, 4.0, 1.0);
        let joints = decode_simcc(&x, &y, &z);
        let j = joints[0];
        let nx_bins = j.nx * SIMCC_X_BINS as f32;
        let ny_bins = j.ny * SIMCC_Y_BINS as f32;
        let nz_bins = j.nz * SIMCC_Z_BINS as f32;
        assert!(
            nx_bins > 100.0 && nx_bins <= 100.5,
            "x vertex must shift toward the stronger right neighbour, got {nx_bins}"
        );
        assert!(
            (ny_bins - 200.0).abs() < 1e-4,
            "symmetric neighbours must not shift the peak, got {ny_bins}"
        );
        assert!(
            nz_bins < 300.0 && nz_bins >= 299.5,
            "z vertex must shift toward the stronger left neighbour, got {nz_bins}"
        );
    }

    #[test]
    fn refinement_recovers_continuous_peak_between_bins() {
        // A sampled parabola with vertex at bin + 0.25 must decode to
        // within a small tolerance of that fractional position.
        let vertex = 150.25_f32;
        let f = |b: usize| -((b as f32 - vertex) * (b as f32 - vertex));
        let x = full(SIMCC_X_BINS, 150, f(149), f(150), f(151));
        let y = full(SIMCC_Y_BINS, 150, f(149), f(150), f(151));
        let z = full(SIMCC_Z_BINS, 150, f(149), f(150), f(151));
        let joints = decode_simcc(&x, &y, &z);
        let nx_bins = joints[0].nx * SIMCC_X_BINS as f32;
        assert!(
            (nx_bins - vertex).abs() < 1e-3,
            "parabolic vertex must be recovered exactly, got {nx_bins} want {vertex}"
        );
    }
}
