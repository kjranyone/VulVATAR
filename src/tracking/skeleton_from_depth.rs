//! Shared "2D keypoints + per-pixel metric depth → SourceSkeleton"
//! pipeline used by both [`super::cigpose_metric_depth`] and
//! [`super::rtmw3d_with_depth`].
//!
//! The two providers feed differently-sourced 2D keypoints (CIGPose
//! and RTMW3D respectively) and differently-derived depth maps (MoGe
//! true-metric vs DAv2 calibrated-relative), but both produce a
//! [`super::metric_depth::MoGeFrame`]-shaped point cloud and consume
//! [`super::cigpose::DecodedJoint2d`]-shaped 2D landmarks. The
//! skeleton-building math (origin selection, axis flips, hand chain
//! attachment, head-pose derivation) is identical past that point —
//! keeping it single-sourced here avoids the two providers drifting.

use crate::asset::HumanoidBone;

use super::cigpose::DecodedJoint2d;
use super::face_mediapipe::{derive_face_bbox, FaceBbox};
use super::metric_depth::MoGeFrame;
use super::source_skeleton::HandOrientation;
use super::{FacePose, SourceJoint, SourceSkeleton};

/// Per-keypoint visibility floor — anything below is treated as "no
/// detection" rather than a low-confidence detection. The user-tunable
/// `joint_confidence_threshold` slider lives in the solver.
pub const KEYPOINT_VISIBILITY_FLOOR: f32 = 0.05;

/// Half-size of the square depth-sampling window when reading metric
/// depth at a 2D keypoint. Small enough to track fingers without
/// dragging in neighbouring depth, large enough that single-pixel
/// model artefacts get out-voted by the median.
pub const SAMPLE_RADIUS_PX: i32 = 3;

const NUM_JOINTS: usize = 133;

// ---------------------------------------------------------------------------
// COCO-Wholebody index → HumanoidBone tables. Same convention RTMW3D
// uses; selfie mirror = subject's anatomical-left → avatar `Right*`
// bone. Single-sourced here so any future tweak (e.g. excluding a
// noisy keypoint) lands in both providers.
// ---------------------------------------------------------------------------

pub(super) const COCO_BODY: &[(usize, HumanoidBone)] = &[
    (5, HumanoidBone::RightShoulder),
    (5, HumanoidBone::RightUpperArm),
    (6, HumanoidBone::LeftShoulder),
    (6, HumanoidBone::LeftUpperArm),
    (7, HumanoidBone::RightLowerArm),
    (8, HumanoidBone::LeftLowerArm),
    (11, HumanoidBone::RightUpperLeg),
    (12, HumanoidBone::LeftUpperLeg),
    (13, HumanoidBone::RightLowerLeg),
    (14, HumanoidBone::LeftLowerLeg),
    (15, HumanoidBone::RightFoot),
    (16, HumanoidBone::LeftFoot),
];

pub(super) const COCO_FOOT_TIPS: &[(usize, HumanoidBone)] = &[
    (17, HumanoidBone::RightFoot),
    (20, HumanoidBone::LeftFoot),
];

pub(super) const HAND_PHALANGES_RIGHT: &[(usize, HumanoidBone)] = &[
    (1, HumanoidBone::RightThumbProximal),
    (2, HumanoidBone::RightThumbIntermediate),
    (3, HumanoidBone::RightThumbDistal),
    (5, HumanoidBone::RightIndexProximal),
    (6, HumanoidBone::RightIndexIntermediate),
    (7, HumanoidBone::RightIndexDistal),
    (9, HumanoidBone::RightMiddleProximal),
    (10, HumanoidBone::RightMiddleIntermediate),
    (11, HumanoidBone::RightMiddleDistal),
    (13, HumanoidBone::RightRingProximal),
    (14, HumanoidBone::RightRingIntermediate),
    (15, HumanoidBone::RightRingDistal),
    (17, HumanoidBone::RightLittleProximal),
    (18, HumanoidBone::RightLittleIntermediate),
    (19, HumanoidBone::RightLittleDistal),
];

pub(super) const HAND_TIPS_RIGHT: &[(usize, HumanoidBone)] = &[
    (4, HumanoidBone::RightThumbDistal),
    (8, HumanoidBone::RightIndexDistal),
    (12, HumanoidBone::RightMiddleDistal),
    (16, HumanoidBone::RightRingDistal),
    (20, HumanoidBone::RightLittleDistal),
];

pub(super) const HAND_PHALANGES_LEFT: &[(usize, HumanoidBone)] = &[
    (1, HumanoidBone::LeftThumbProximal),
    (2, HumanoidBone::LeftThumbIntermediate),
    (3, HumanoidBone::LeftThumbDistal),
    (5, HumanoidBone::LeftIndexProximal),
    (6, HumanoidBone::LeftIndexIntermediate),
    (7, HumanoidBone::LeftIndexDistal),
    (9, HumanoidBone::LeftMiddleProximal),
    (10, HumanoidBone::LeftMiddleIntermediate),
    (11, HumanoidBone::LeftMiddleDistal),
    (13, HumanoidBone::LeftRingProximal),
    (14, HumanoidBone::LeftRingIntermediate),
    (15, HumanoidBone::LeftRingDistal),
    (17, HumanoidBone::LeftLittleProximal),
    (18, HumanoidBone::LeftLittleIntermediate),
    (19, HumanoidBone::LeftLittleDistal),
];

pub(super) const HAND_TIPS_LEFT: &[(usize, HumanoidBone)] = &[
    (4, HumanoidBone::LeftThumbDistal),
    (8, HumanoidBone::LeftIndexDistal),
    (12, HumanoidBone::LeftMiddleDistal),
    (16, HumanoidBone::LeftRingDistal),
    (20, HumanoidBone::LeftLittleDistal),
];

// ---------------------------------------------------------------------------
// Depth sampling
// ---------------------------------------------------------------------------

/// Sample a metric-depth frame's point cloud at a frame-relative
/// `(nx, ny)` keypoint with a local-window median. Returns `None` when
/// the keypoint falls outside the frame or every sample in the window
/// is masked / non-finite.
pub fn sample_metric_point(frame: &MoGeFrame, nx: f32, ny: f32) -> Option<[f32; 3]> {
    sample_metric_point_with_radius(frame, nx, ny, SAMPLE_RADIUS_PX)
}

/// Variant of [`sample_metric_point`] that takes the median window
/// half-size as an argument. Used by diagnostics to compare radius=0
/// (single pixel) / 1 (3×3) / 3 (7×7, the runtime default).
pub fn sample_metric_point_with_radius(
    frame: &MoGeFrame,
    nx: f32,
    ny: f32,
    radius_px: i32,
) -> Option<[f32; 3]> {
    if !nx.is_finite() || !ny.is_finite() {
        return None;
    }
    let w = frame.width as i32;
    let h = frame.height as i32;
    let cx = (nx * frame.width as f32).round() as i32;
    let cy = (ny * frame.height as f32).round() as i32;
    if cx < 0 || cy < 0 || cx >= w || cy >= h {
        return None;
    }
    let cap = ((radius_px * 2 + 1) * (radius_px * 2 + 1)) as usize;
    let mut xs = Vec::with_capacity(cap);
    let mut ys = Vec::with_capacity(cap);
    let mut zs = Vec::with_capacity(cap);
    for yy in (cy - radius_px).max(0)..=(cy + radius_px).min(h - 1) {
        for xx in (cx - radius_px).max(0)..=(cx + radius_px).min(w - 1) {
            let idx = yy as usize * frame.width as usize + xx as usize;
            let [px, py, pz] = frame.points_m[idx];
            if px.is_finite() && py.is_finite() && pz.is_finite() && pz > 0.0 {
                xs.push(px);
                ys.push(py);
                zs.push(pz);
            }
        }
    }
    Some([median(xs)?, median(ys)?, median(zs)?])
}

fn median(mut values: Vec<f32>) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    Some(values[values.len() / 2])
}

/// Sample a torso-anchor keypoint (shoulder / hip) using the **back
/// percentile** of the depth window instead of the median. The
/// `BACK_PERCENTILE_FRACTION` deepest fraction of pixels in the window
/// is selected, and the median of *those* pixels is returned.
///
/// **Why**: the unmasked median fails when something close to the
/// camera (a hand crossing in front of the chest, a palm resting on
/// the shoulder) covers the majority of the sample window. The
/// median then collapses onto the occluder's depth, and the shoulder
/// reads as `z > 0` (in front of the camera plane) — physically
/// impossible for a torso anchor. The back-percentile approach
/// exploits the anatomical prior that **the torso is the deepest
/// part of any near-keypoint geometry**: even when a hand covers
/// most of the window, the few pixels at the edges that still see
/// the actual shoulder live at the *back* of the depth distribution.
/// Taking the median of the deepest 25% gives those pixels the
/// floor.
///
/// **Why this isn't safe for non-torso keypoints**: a wrist held in
/// front of the body has `wrist_depth < torso_depth`. Applying the
/// back-percentile to the wrist would push it backward onto the
/// torso's depth — a regression. So this function is wired *only*
/// to shoulder (COCO 5, 6) and hip (COCO 11, 12) samples in
/// `resolve_origin_metric` and `build_skeleton::resolve`. The other
/// 13 body keypoints, plus all face / hand keypoints, continue to
/// use the vanilla median sample which trusts the keypoint position.
pub fn sample_metric_point_back_percentile(
    frame: &MoGeFrame,
    nx: f32,
    ny: f32,
) -> Option<[f32; 3]> {
    sample_metric_point_back_percentile_with_radius(frame, nx, ny, SAMPLE_RADIUS_PX)
}

/// Fraction of *person-surface* pixels (counting from the deepest)
/// considered for the back-percentile median. 25% means: after the
/// background-rejection pass narrows the window to person pixels,
/// take the deepest 25% of those and median them. Tuned so the
/// occluder-dominant case (hand covers >50% of window) still
/// recovers via the rear edge pixels, while the no-occlusion case
/// (uniform shoulder depth across the whole window) reduces to
/// approximately the unmasked median.
const BACK_PERCENTILE_FRACTION: f32 = 0.25;

/// Half-width of the "person surface" depth band centred on the
/// window's coarse median. Pixels outside this band (typically
/// background — wall behind subject, often 1–3 m further than the
/// torso) are excluded from the back-percentile selection so the
/// "deepest fraction" doesn't end up reading the wall instead of
/// the shoulder. 0.50 m comfortably covers a person's front-to-back
/// thickness (~30 cm at the chest) while rejecting any meaningful
/// background gap. Larger values (1.0+ m) start admitting walls
/// in tight rooms; smaller values (0.20 m) start clipping the body
/// itself when an occluder hand pulls the coarse median forward.
const SURFACE_BAND_HALF_M: f32 = 0.50;

/// Variant of [`sample_metric_point_back_percentile`] that takes the
/// window half-size as an argument. Mirrors the
/// [`sample_metric_point_with_radius`] split for diagnostic use.
pub fn sample_metric_point_back_percentile_with_radius(
    frame: &MoGeFrame,
    nx: f32,
    ny: f32,
    radius_px: i32,
) -> Option<[f32; 3]> {
    if !nx.is_finite() || !ny.is_finite() {
        return None;
    }
    let w = frame.width as i32;
    let h = frame.height as i32;
    let cx = (nx * frame.width as f32).round() as i32;
    let cy = (ny * frame.height as f32).round() as i32;
    if cx < 0 || cy < 0 || cx >= w || cy >= h {
        return None;
    }
    let cap = ((radius_px * 2 + 1) * (radius_px * 2 + 1)) as usize;
    let mut samples: Vec<[f32; 3]> = Vec::with_capacity(cap);
    for yy in (cy - radius_px).max(0)..=(cy + radius_px).min(h - 1) {
        for xx in (cx - radius_px).max(0)..=(cx + radius_px).min(w - 1) {
            let idx = yy as usize * frame.width as usize + xx as usize;
            let [px, py, pz] = frame.points_m[idx];
            if px.is_finite() && py.is_finite() && pz.is_finite() && pz > 0.0 {
                samples.push([px, py, pz]);
            }
        }
    }
    if samples.is_empty() {
        return None;
    }

    // Step 1 — coarse person-surface depth from the unmasked median.
    // This is the same value `sample_metric_point` would return; we
    // reuse it as an anchor for the surface-band filter below. When
    // an occluder dominates the window the coarse median sits on
    // the occluder's depth, but the *band* around it (±0.5 m) still
    // reaches back to the shoulder behind it; the band only excludes
    // genuine background (wall 1+ m further).
    let mut all_z: Vec<f32> = samples.iter().map(|p| p[2]).collect();
    all_z.sort_by(|a, b| a.total_cmp(b));
    let coarse_z = all_z[all_z.len() / 2];

    // Step 2 — keep only pixels within the person-surface band. The
    // band is centred on `coarse_z` and ±SURFACE_BAND_HALF_M wide.
    // Background pixels (wall behind subject, depth-map void filled
    // with implausible values) fall outside this band and are
    // discarded so they don't poison the back-percentile in step 3.
    let surface: Vec<&[f32; 3]> = samples
        .iter()
        .filter(|p| (p[2] - coarse_z).abs() <= SURFACE_BAND_HALF_M)
        .collect();

    // Empty surface band is unreachable in practice — at minimum
    // the pixel that produced `coarse_z` itself is in the band — but
    // guard against numerical edge cases (NaN propagation, etc.) by
    // falling back to the unmasked sample.
    if surface.is_empty() {
        return sample_metric_point_with_radius(frame, nx, ny, radius_px);
    }

    // Step 3 — sort surface pixels by depth and take the deepest
    // BACK_PERCENTILE_FRACTION. With background gone, this reads
    // the actual shoulder/hip surface at the rear edge of the
    // person's silhouette, which is robust to a foreground hand /
    // arm occluding most of the window.
    let mut surface_owned: Vec<[f32; 3]> = surface.iter().map(|&&p| p).collect();
    surface_owned.sort_by(|a, b| a[2].total_cmp(&b[2]));
    let back_count = ((surface_owned.len() as f32) * BACK_PERCENTILE_FRACTION)
        .ceil()
        .max(1.0) as usize;
    let back_count = back_count.min(surface_owned.len());
    let back_slice = &surface_owned[surface_owned.len() - back_count..];

    // Per-axis median of the back slice. Sorting each axis
    // independently rather than picking the centroid of the same
    // pixel: x / y of the same back-most z pixel can land on the
    // shoulder edge rather than the centre, while the per-axis
    // median centres the (x, y, z) result on the back slice's
    // centre of mass.
    let xs: Vec<f32> = back_slice.iter().map(|p| p[0]).collect();
    let ys: Vec<f32> = back_slice.iter().map(|p| p[1]).collect();
    let zs: Vec<f32> = back_slice.iter().map(|p| p[2]).collect();
    Some([median(xs)?, median(ys)?, median(zs)?])
}

// ---------------------------------------------------------------------------
// Torso depth template capture
// ---------------------------------------------------------------------------

/// Running accumulator for `TorsoDepthTemplate` capture during a
/// calibration window. Each `add_frame` call samples one cell per
/// grid position and appends the depth to the matching per-cell
/// bucket; `finalize` reduces each bucket via median to produce
/// the on-disk template.
///
/// Memory cost is bounded: `GRID_SIZE × GRID_SIZE × frames × 4
/// bytes`. At 32×32 cells × 60 frames (2 s × 30 fps) × 4 = 240 KB
/// per capture window, well below the 4 KB final on-disk size and
/// utterly insignificant for a transient buffer.
pub struct TorsoCaptureBuffer {
    /// Per-cell accumulated depths. Length = `GRID_SIZE × GRID_SIZE`,
    /// row-major. Cells with no admitted samples (depth-map void or
    /// no qualifying frame in the window) finalize to `f32::NAN`.
    cells: Vec<Vec<f32>>,
    /// Bbox extents averaged across admitted frames. We average rather
    /// than take a single frame's because the subject drifts slightly
    /// during the 2-second window — averaging gives the template a
    /// "centred" bbox over the capture period.
    bbox_acc: [f32; 4],
    /// Number of frames whose data was admitted into `cells` /
    /// `bbox_acc`. Used at finalize time to (a) bail with `None` if
    /// zero, and (b) divide `bbox_acc` for the average.
    frame_count: u32,
}

impl TorsoCaptureBuffer {
    /// Grid resolution. Mirrors `TorsoDepthTemplate::GRID_SIZE` so
    /// callers can pre-size their loops without coupling to the
    /// calibration module's constant.
    pub const GRID: usize =
        crate::tracking::TorsoDepthTemplate::GRID_SIZE as usize;

    pub fn new() -> Self {
        let cell_count = Self::GRID * Self::GRID;
        Self {
            cells: (0..cell_count).map(|_| Vec::new()).collect(),
            bbox_acc: [0.0; 4],
            frame_count: 0,
        }
    }

    /// Sample one frame into the buffer. Returns `true` when the
    /// frame met the bar (all four torso anchors above floor, bbox
    /// non-degenerate, at least one cell got a finite sample) and
    /// `false` when it was skipped.
    ///
    /// **What "all four torso anchors" means**: shoulders (COCO 5/6)
    /// plus hips (COCO 11/12). When the user calibrates in Upper
    /// Body mode the hips might be hidden — for now we still require
    /// all four, since the torso bbox needs the hip pair to know
    /// where the bottom of the torso ends. A future Upper Body-only
    /// variant could fall back to "shoulder line + a fixed offset"
    /// or to MediaPipe's torso segmentation, but that's outside the
    /// minimum-version scope.
    pub fn add_frame(
        &mut self,
        joints: &[DecodedJoint2d],
        frame: &MoGeFrame,
    ) -> bool {
        if joints.len() < NUM_JOINTS {
            return false;
        }
        let ls = &joints[5];
        let rs = &joints[6];
        let lh = &joints[11];
        let rh = &joints[12];
        let min_score = ls.score.min(rs.score).min(lh.score).min(rh.score);
        if min_score < KEYPOINT_VISIBILITY_FLOOR {
            return false;
        }

        // Bbox in image-relative coordinates. Pad slightly outward
        // (5%) so cells near the keypoints capture a bit of skin /
        // collar / waistband margin rather than being clipped at
        // the exact joint pixel — small padding tolerates the
        // running-mean bbox drift across the capture window without
        // shifting cells off the body itself.
        const BBOX_PAD: f32 = 0.05;
        let min_nx_raw = ls.nx.min(rs.nx).min(lh.nx).min(rh.nx);
        let max_nx_raw = ls.nx.max(rs.nx).max(lh.nx).max(rh.nx);
        let min_ny_raw = ls.ny.min(rs.ny).min(lh.ny).min(rh.ny);
        let max_ny_raw = ls.ny.max(rs.ny).max(lh.ny).max(rh.ny);
        let bbox_w = max_nx_raw - min_nx_raw;
        let bbox_h = max_ny_raw - min_ny_raw;
        if bbox_w < 0.05 || bbox_h < 0.05 {
            return false;
        }
        let pad_x = bbox_w * BBOX_PAD;
        let pad_y = bbox_h * BBOX_PAD;
        let min_nx = (min_nx_raw - pad_x).clamp(0.0, 1.0);
        let max_nx = (max_nx_raw + pad_x).clamp(0.0, 1.0);
        let min_ny = (min_ny_raw - pad_y).clamp(0.0, 1.0);
        let max_ny = (max_ny_raw + pad_y).clamp(0.0, 1.0);

        // Sample one pixel per cell at the cell's centre. Single-
        // pixel sample is fine here because the median over the
        // capture window absorbs per-frame depth-map noise — using
        // the 7×7 window-median per cell would just be redundant
        // smoothing.
        let grid = Self::GRID;
        let grid_f = grid as f32;
        let mut frame_admitted = false;
        for gy in 0..grid {
            for gx in 0..grid {
                // Cell centre in image-relative coords.
                let cell_nx = min_nx + ((gx as f32 + 0.5) / grid_f) * (max_nx - min_nx);
                let cell_ny = min_ny + ((gy as f32 + 0.5) / grid_f) * (max_ny - min_ny);
                let cx = (cell_nx * frame.width as f32).round() as i32;
                let cy = (cell_ny * frame.height as f32).round() as i32;
                if cx < 0 || cy < 0 || cx >= frame.width as i32 || cy >= frame.height as i32 {
                    continue;
                }
                let pixel_idx = cy as usize * frame.width as usize + cx as usize;
                let [_, _, pz] = frame.points_m[pixel_idx];
                if pz.is_finite() && pz > 0.0 {
                    self.cells[gy * grid + gx].push(pz);
                    frame_admitted = true;
                }
            }
        }

        if !frame_admitted {
            return false;
        }
        self.bbox_acc[0] += min_nx;
        self.bbox_acc[1] += min_ny;
        self.bbox_acc[2] += max_nx;
        self.bbox_acc[3] += max_ny;
        self.frame_count += 1;
        true
    }

    pub fn frame_count(&self) -> u32 {
        self.frame_count
    }

    /// Reduce the per-cell accumulated depths via median and return
    /// the immutable on-disk template. `None` when no frames were
    /// admitted — caller should treat that as "no template available
    /// for this calibration" and fall back to anchor-only behaviour.
    pub fn finalize(self) -> Option<crate::tracking::TorsoDepthTemplate> {
        if self.frame_count == 0 {
            return None;
        }
        let n = self.frame_count as f32;
        let bbox = [
            self.bbox_acc[0] / n,
            self.bbox_acc[1] / n,
            self.bbox_acc[2] / n,
            self.bbox_acc[3] / n,
        ];
        let depths: Vec<f32> = self
            .cells
            .into_iter()
            .map(|mut bucket| {
                if bucket.is_empty() {
                    f32::NAN
                } else {
                    bucket.sort_by(|a, b| a.total_cmp(b));
                    bucket[bucket.len() / 2]
                }
            })
            .collect();
        Some(crate::tracking::TorsoDepthTemplate {
            width: Self::GRID as u32,
            height: Self::GRID as u32,
            depths_m: depths,
            bbox_normalized: bbox,
        })
    }
}

impl Default for TorsoCaptureBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the current frame's torso bounding box from the four
/// torso anchor keypoints (shoulders 5/6 + hips 11/12). Returns
/// `[min_nx, min_ny, max_nx, max_ny]` with the same 5% padding the
/// capture path uses, or `None` when any anchor is below the
/// visibility floor or the bbox is degenerate.
///
/// Single-sourced so the inference-time bias path
/// (`compute_torso_bias`) and the capture path
/// (`TorsoCaptureBuffer::add_frame`) agree on bbox extents bit-for-bit;
/// a divergence here would shift the cell-grid registration and the
/// bias would be reading the wrong template cells.
pub fn current_torso_bbox(joints: &[DecodedJoint2d]) -> Option<[f32; 4]> {
    if joints.len() < NUM_JOINTS {
        return None;
    }
    let ls = &joints[5];
    let rs = &joints[6];
    let lh = &joints[11];
    let rh = &joints[12];
    let min_score = ls.score.min(rs.score).min(lh.score).min(rh.score);
    if min_score < KEYPOINT_VISIBILITY_FLOOR {
        return None;
    }
    const BBOX_PAD: f32 = 0.05;
    let min_nx_raw = ls.nx.min(rs.nx).min(lh.nx).min(rh.nx);
    let max_nx_raw = ls.nx.max(rs.nx).max(lh.nx).max(rh.nx);
    let min_ny_raw = ls.ny.min(rs.ny).min(lh.ny).min(rh.ny);
    let max_ny_raw = ls.ny.max(rs.ny).max(lh.ny).max(rh.ny);
    let bbox_w = max_nx_raw - min_nx_raw;
    let bbox_h = max_ny_raw - min_ny_raw;
    if bbox_w < 0.05 || bbox_h < 0.05 {
        return None;
    }
    let pad_x = bbox_w * BBOX_PAD;
    let pad_y = bbox_h * BBOX_PAD;
    Some([
        (min_nx_raw - pad_x).clamp(0.0, 1.0),
        (min_ny_raw - pad_y).clamp(0.0, 1.0),
        (max_nx_raw + pad_x).clamp(0.0, 1.0),
        (max_ny_raw + pad_y).clamp(0.0, 1.0),
    ])
}

/// Maximum allowed deviation between current-frame and template depth
/// for a cell to contribute to the bias estimate. Cells outside this
/// band are treated as occluder (closer than template) or background
/// (much further than template) and silently dropped. Same physical
/// reasoning as `SURFACE_BAND_HALF_M` in the back-percentile sample:
/// the human torso is ~30 cm front-to-back, ±0.5 m comfortably
/// covers that plus moderate forward/back drift since calibration.
const TEMPLATE_BIAS_BAND_M: f32 = 0.5;

/// Compute the current-frame torso depth bias against a calibrated
/// template. Returns `Some(bias_m)` such that
/// `current_torso_depth ≈ template_torso_depth + bias` over the
/// pixels that are actually showing the torso (occluders + background
/// rejected by the `TEMPLATE_BIAS_BAND_M` gate).
///
/// **Algorithm**:
/// 1. Build current-frame torso bbox from shoulders + hips.
/// 2. For each template cell:
///    a. Skip if `template_depth` is NaN (the cell never got a
///       valid sample during calibration).
///    b. Compute the cell's centre position in image coords by
///       projecting `(rel_x, rel_y)` through the *current* bbox.
///    c. Sample the current frame's depth at that pixel.
///    d. Skip if the sample is non-finite or out of frame.
///    e. Skip if `|current − template| > TEMPLATE_BIAS_BAND_M`
///       (occluder / background reject).
///    f. Otherwise, append `current − template` to the bias pool.
/// 3. Return median(pool) if pool is non-empty, else `None`.
///
/// **Why median over mean**: occluder pixels that *just barely*
/// fall inside the band (e.g. a hand whose depth is 40 cm closer
/// than the chest behind it) will skew the mean noticeably; the
/// median absorbs them as long as they're a minority. With ~half
/// the torso typically visible even under heavy occlusion, the
/// median is robust.
pub fn compute_torso_bias(
    template: &crate::tracking::TorsoDepthTemplate,
    joints: &[DecodedJoint2d],
    frame: &MoGeFrame,
) -> Option<f32> {
    let bbox = current_torso_bbox(joints)?;
    let cur_w = bbox[2] - bbox[0];
    let cur_h = bbox[3] - bbox[1];
    if cur_w < 0.05 || cur_h < 0.05 {
        return None;
    }
    let grid_w = template.width as usize;
    let grid_h = template.height as usize;
    if template.depths_m.len() != grid_w * grid_h {
        return None;
    }

    let mut diffs: Vec<f32> = Vec::with_capacity(grid_w * grid_h);
    let grid_w_f = grid_w as f32;
    let grid_h_f = grid_h as f32;
    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let template_z = template.depths_m[gy * grid_w + gx];
            if !template_z.is_finite() {
                continue;
            }
            // Cell centre as a position within the current bbox.
            let rel_x = (gx as f32 + 0.5) / grid_w_f;
            let rel_y = (gy as f32 + 0.5) / grid_h_f;
            let cell_nx = bbox[0] + rel_x * cur_w;
            let cell_ny = bbox[1] + rel_y * cur_h;
            let cx = (cell_nx * frame.width as f32).round() as i32;
            let cy = (cell_ny * frame.height as f32).round() as i32;
            if cx < 0 || cy < 0 || cx >= frame.width as i32 || cy >= frame.height as i32 {
                continue;
            }
            let pixel_idx = cy as usize * frame.width as usize + cx as usize;
            let [_, _, current_z] = frame.points_m[pixel_idx];
            if !current_z.is_finite() || current_z <= 0.0 {
                continue;
            }
            let diff = current_z - template_z;
            if diff.abs() > TEMPLATE_BIAS_BAND_M {
                continue;
            }
            diffs.push(diff);
        }
    }

    if diffs.is_empty() {
        return None;
    }
    diffs.sort_by(|a, b| a.total_cmp(b));
    Some(diffs[diffs.len() / 2])
}

/// Look up a torso anchor's depth from the calibrated template
/// at the keypoint's current image position. The keypoint's
/// `(nx, ny)` is mapped into the *current* bbox's local coordinate
/// system, then re-projected onto the template grid (the template
/// stores its own bbox so the mapping is a pure rescale, not a
/// per-frame fit). Returns `None` when the keypoint falls outside
/// the bbox or the corresponding template cell is NaN (no
/// calibration sample admitted there).
pub fn template_depth_at_keypoint(
    template: &crate::tracking::TorsoDepthTemplate,
    bbox: &[f32; 4],
    nx: f32,
    ny: f32,
) -> Option<f32> {
    let bbox_w = bbox[2] - bbox[0];
    let bbox_h = bbox[3] - bbox[1];
    if bbox_w < 1e-4 || bbox_h < 1e-4 {
        return None;
    }
    // Position within the current bbox, [0, 1] × [0, 1].
    let rel_x = (nx - bbox[0]) / bbox_w;
    let rel_y = (ny - bbox[1]) / bbox_h;
    if !(0.0..=1.0).contains(&rel_x) || !(0.0..=1.0).contains(&rel_y) {
        return None;
    }
    // Round to the nearest grid cell. Single-cell lookup rather than
    // bilinear: at 32×32 the cell footprint is ~3% of bbox width,
    // smaller than a typical shoulder keypoint's per-frame jitter,
    // so interpolation buys nothing. Larger grids in future Step 3
    // experiments may want bilinear.
    let gx = (rel_x * template.width as f32) as usize;
    let gy = (rel_y * template.height as f32) as usize;
    let gx = gx.min(template.width as usize - 1);
    let gy = gy.min(template.height as usize - 1);
    let cell_z = template.depths_m[gy * template.width as usize + gx];
    cell_z.is_finite().then_some(cell_z)
}

// ---------------------------------------------------------------------------
// Skeleton building
// ---------------------------------------------------------------------------

/// Per-call options that the depth-aware providers thread through
/// from the tracking-worker (and ultimately from
/// `Application::tracking_calibration`) into the skeleton builder.
///
/// Wired here as a struct rather than positional args so future
/// calibration-derived knobs (dynamic c-clamp range, anchor-bias
/// rejection thresholds) can be added without touching every call
/// site again.
#[derive(Clone, Copy, Debug, Default)]
pub struct BuildOptions {
    /// `true` when the user calibrated in `Upper Body Only` mode and
    /// hip detection should be skipped — even when the hip pair clears
    /// the visibility floor, treat it as untrustworthy (almost
    /// certainly a desk surface or chair seat masquerading as a hip
    /// keypoint). `false` preserves the existing
    /// hip-preferred / shoulder-fallback behaviour.
    pub force_shoulder_anchor: bool,
    /// Per-subject expected upper-arm length (shoulder → elbow) in
    /// metres, derived from `PoseCalibration::shoulder_span_m`. Drives
    /// the bone-length-aware Z reconstruction in
    /// [`reconstruct_arm_z_magnitudes`]. `None` falls back to the
    /// `EXPECTED_UPPER_ARM_M` adult-average constant.
    pub subject_upper_arm_m: Option<f32>,
    /// Per-subject expected lower-arm length (elbow → wrist) in
    /// metres, derived from `PoseCalibration::shoulder_span_m`. `None`
    /// falls back to the `EXPECTED_LOWER_ARM_M` adult-average constant.
    pub subject_lower_arm_m: Option<f32>,
}

/// Build [`BuildOptions`] from the depth-providers' cached
/// `pose_calibration`. Single-sourced so both depth providers
/// (rtmw3d-with-depth, cigpose-metric-depth) compute the same opts
/// and any future calibration-derived knob (c-clamp from jitter,
/// anchor-bias rejection) lands in one place.
pub(super) fn build_options_from_calibration(
    calibration: Option<&super::PoseCalibration>,
) -> BuildOptions {
    let force_shoulder_anchor = calibration
        .map(|c| matches!(c.mode, super::CalibrationMode::UpperBody))
        .unwrap_or(false);
    // Adult anatomy ratios pinned to shoulder span: upper arm = 0.75 ×,
    // forearm = 0.625 ×. These are population averages; per-subject
    // variance is ~±10% which is well within reconstruction's
    // tolerance (we only use the magnitude to disambiguate sign-noise
    // from real depth signals, so a 10% body-scale mismatch shifts the
    // reconstructed elbow Z by 3 cm — invisible at the avatar scale).
    let (subject_upper_arm_m, subject_lower_arm_m) = calibration
        .and_then(|c| c.shoulder_span_m)
        .map(|span| (Some(span * 0.75), Some(span * 0.625)))
        .unwrap_or((None, None));
    BuildOptions {
        force_shoulder_anchor,
        subject_upper_arm_m,
        subject_lower_arm_m,
    }
}

/// Resolve the body anchor origin in metric camera-space coords.
/// Hip mid is preferred (CIGPose 11/12); falls back to shoulder mid
/// (5/6) for upper-body crops. Returns `None` when neither pair is
/// usable — the frame produces no skeleton.
///
/// `opts.force_shoulder_anchor = true` skips the hip branch entirely;
/// see [`BuildOptions`] for the defensive rationale.
pub(super) fn resolve_origin_metric(
    joints: &[DecodedJoint2d],
    depth: &MoGeFrame,
    opts: BuildOptions,
    calibration: Option<&crate::tracking::PoseCalibration>,
) -> Option<([f32; 3], /*anchor_was_hip*/ bool, /*anchor_score*/ f32)> {
    if joints.len() < NUM_JOINTS {
        return None;
    }
    // Pre-compute torso bias if the calibration carries a template.
    // Cached once per frame because both anchor pairs (hip + shoulder
    // fallback) share the same bias — the bias describes the whole
    // torso's z drift since calibration, not a per-keypoint offset.
    let template_bias = calibration
        .and_then(|c| c.torso_depth_template.as_ref())
        .and_then(|t| {
            let bbox = current_torso_bbox(joints)?;
            compute_torso_bias(t, joints, depth).map(|b| (t, bbox, b))
        });

    // Anchor pairs (hip 11/12 and shoulder 5/6) use either:
    //   (a) **template + bias** when calibration has a torso template
    //       *and* the keypoint maps to a non-NaN template cell. This
    //       is robust to heavy occlusion (palms covering shoulders,
    //       arms crossed over chest) because the bias is computed
    //       over the *visible* torso surface and the template
    //       supplies the per-anchor depth from calibration.
    //   (b) the **back-percentile** sample as a fallback. Used when
    //       no calibration exists, or when the template cell at the
    //       keypoint is NaN, or the bias couldn't be computed (no
    //       visible torso pixels matched the band).
    let sample = |kpt: &DecodedJoint2d| -> Option<[f32; 3]> {
        if let Some((template, bbox, bias)) = template_bias.as_ref() {
            if let Some(template_z) =
                template_depth_at_keypoint(template, bbox, kpt.nx, kpt.ny)
            {
                // Keep x / y from the back-percentile sample (image-
                // plane location is what we trust the keypoint
                // detector for), but replace z with the calibrated
                // template depth shifted by the current frame's
                // bias. This is what survives occlusion of the
                // keypoint itself.
                let xy = sample_metric_point_back_percentile(depth, kpt.nx, kpt.ny)?;
                return Some([xy[0], xy[1], template_z + bias]);
            }
        }
        sample_metric_point_back_percentile(depth, kpt.nx, kpt.ny)
    };

    let try_pair = |a: &DecodedJoint2d, b: &DecodedJoint2d| -> Option<[f32; 3]> {
        let l = sample(a)?;
        let r = sample(b)?;
        Some([
            (l[0] + r[0]) * 0.5,
            (l[1] + r[1]) * 0.5,
            (l[2] + r[2]) * 0.5,
        ])
    };

    if !opts.force_shoulder_anchor {
        let lh = &joints[11];
        let rh = &joints[12];
        let hip_score = lh.score.min(rh.score);
        if hip_score >= KEYPOINT_VISIBILITY_FLOOR {
            if let Some(o) = try_pair(lh, rh) {
                return Some((o, true, hip_score));
            }
        }
    }

    let ls = &joints[5];
    let rs = &joints[6];
    let shoulder_score = ls.score.min(rs.score);
    if shoulder_score >= KEYPOINT_VISIBILITY_FLOOR {
        if let Some(o) = try_pair(ls, rs) {
            return Some((o, false, shoulder_score));
        }
    }
    None
}

/// Build a [`SourceSkeleton`] from 2D keypoints + sampled metric
/// depth using a pre-resolved origin. Camera-space metric `(x-right,
/// y-down, z-forward)` is flipped on every axis to match the
/// source-space convention `(selfie-mirror x, y-up, z toward
/// camera)`. Selfie mirror: subject's anatomical-left landmarks
/// drive avatar `Right*` bones.
///
/// Wrist position = centroid of the four MCPs (rationale: the
/// hand-track wrist landmark is unreliable under occlusion). Face
/// pose / expressions are intentionally left empty here — the caller
/// populates them via [`derive_face_pose_metric`] and the optional
/// FaceMesh cascade.
pub(super) fn build_skeleton(
    frame_index: u64,
    joints: &[DecodedJoint2d],
    depth: &MoGeFrame,
    origin: [f32; 3],
    anchor_was_hip: bool,
    anchor_score: f32,
    opts: BuildOptions,
    calibration: Option<&crate::tracking::PoseCalibration>,
) -> SourceSkeleton {
    let mut sk = SourceSkeleton::empty(frame_index);
    if joints.len() < NUM_JOINTS {
        return sk;
    }

    let to_source = |p: [f32; 3]| -> [f32; 3] {
        [
            -(p[0] - origin[0]),
            -(p[1] - origin[1]),
            -(p[2] - origin[2]),
        ]
    };

    // Same template-bias precompute as in `resolve_origin_metric`.
    // Torso bbox + bias depend only on the current frame's joints
    // and depth map, so caching here avoids re-computing per anchor
    // sample.
    let template_bias = calibration
        .and_then(|c| c.torso_depth_template.as_ref())
        .and_then(|t| {
            let bbox = current_torso_bbox(joints)?;
            compute_torso_bias(t, joints, depth).map(|b| (t, bbox, b))
        });

    // Sample policy by keypoint:
    //   * Shoulders (5, 6) and hips (11, 12) — torso anchors. When a
    //     calibrated template is available *and* the keypoint maps to
    //     a non-NaN template cell, use template-z + bias. Otherwise
    //     fall back to back-percentile (which itself falls back to
    //     vanilla median when surface band collapses).
    //   * Every other body keypoint (face / elbows / wrists / knees /
    //     ankles) uses the vanilla median. Applying the
    //     back-percentile or template-bias correction to a wrist
    //     held in front of the body would push it backward onto the
    //     torso depth — a regression in the no-occlusion case.
    let resolve = |idx: usize| -> Option<SourceJoint> {
        if idx >= joints.len() {
            return None;
        }
        let j = &joints[idx];
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            return None;
        }
        let pos_cam = if matches!(idx, 5 | 6 | 11 | 12) {
            // Torso anchor path.
            if let Some((template, bbox, bias)) = template_bias.as_ref() {
                if let Some(template_z) =
                    template_depth_at_keypoint(template, bbox, j.nx, j.ny)
                {
                    let xy = sample_metric_point_back_percentile(depth, j.nx, j.ny)?;
                    [xy[0], xy[1], template_z + bias]
                } else {
                    sample_metric_point_back_percentile(depth, j.nx, j.ny)?
                }
            } else {
                sample_metric_point_back_percentile(depth, j.nx, j.ny)?
            }
        } else {
            sample_metric_point(depth, j.nx, j.ny)?
        };
        Some(SourceJoint {
            position: to_source(pos_cam),
            confidence: j.score,
        })
    };

    // Hips + Spine share the pelvic origin. Spine has no COCO keypoint
    // but the solver drives its bone via `Hips → ShoulderMidpoint`,
    // so it needs a source.joints entry for the *base* lookup or
    // pose_solver silently skips spine bend (no waist bend, no forward
    // lean) — the same gap that existed in `rtmw3d::skeleton`.
    if anchor_was_hip {
        let hip_joint = SourceJoint {
            position: [0.0, 0.0, 0.0],
            confidence: anchor_score,
        };
        sk.joints.insert(HumanoidBone::Hips, hip_joint);
        sk.joints.insert(HumanoidBone::Spine, hip_joint);
    }

    // Root offset: anchor mid in metric-camera-space (METERS for the
    // depth-aware path) with axes flipped to match the source-skeleton
    // convention (selfie-mirror x, y-up, z toward camera). Emitted
    // whenever any anchor was detected — hip preferred, with shoulder
    // fallback for upper-body framing. `root_anchor_is_hip` tells
    // downstream consumers (calibration mode-matching, EMA seed
    // selection) which anchor produced the value.
    //
    // Solver subtracts a slow EMA reference so the translation feature
    // is per-setup self-calibrating — what the avatar follows is
    // *deviation* from where the subject normally stands, not absolute
    // camera coords.
    sk.root_offset = Some([-origin[0], -origin[1], -origin[2]]);
    sk.root_anchor_is_hip = anchor_was_hip;

    const LOWER_BODY_FIRST_IDX: usize = 11;
    for &(idx, bone) in COCO_BODY {
        if !anchor_was_hip && idx >= LOWER_BODY_FIRST_IDX {
            continue;
        }
        if let Some(joint) = resolve(idx) {
            sk.joints.insert(bone, joint);
        }
    }

    if anchor_was_hip {
        for &(idx, bone) in COCO_FOOT_TIPS {
            if let Some(joint) = resolve(idx) {
                sk.fingertips.insert(bone, joint);
            }
        }
    }

    attach_hand(
        &mut sk,
        joints,
        depth,
        91,
        HumanoidBone::RightHand,
        HAND_PHALANGES_RIGHT,
        HAND_TIPS_RIGHT,
        &to_source,
    );
    attach_hand(
        &mut sk,
        joints,
        depth,
        112,
        HumanoidBone::LeftHand,
        HAND_PHALANGES_LEFT,
        HAND_TIPS_LEFT,
        &to_source,
    );

    // Order matters: reconstruction must run on the ORIGINAL depth
    // values (small dz from skin/cloth low contrast → reconstruct to
    // anatomical magnitude). The clamp afterwards caps any remaining
    // huge values from occluded keypoints. Reversing the order would
    // make reconstruction see clamp-enlarged dz=0.45 as "already
    // plausible" and skip — defeating the outfit-normalisation.
    reconstruct_arm_z_magnitudes(&mut sk, opts);
    apply_anatomical_z_clamps(&mut sk);

    let mut sum = 0.0_f32;
    let mut n = 0_u32;
    for j in joints.iter().take(17) {
        sum += j.score;
        n += 1;
    }
    sk.overall_confidence = if n > 0 { sum / n as f32 } else { 0.0 };

    sk.face = None;
    sk.expressions = Vec::new();
    sk
}

#[allow(clippy::too_many_arguments)]
fn attach_hand<F>(
    sk: &mut SourceSkeleton,
    joints: &[DecodedJoint2d],
    depth: &MoGeFrame,
    base_index: usize,
    wrist_bone: HumanoidBone,
    phalanges: &[(usize, HumanoidBone)],
    tips: &[(usize, HumanoidBone)],
    to_source: &F,
) where
    F: Fn([f32; 3]) -> [f32; 3],
{
    // Sample the four MCP knuckles in source-space and the
    // hand-track wrist (idx 0) — the wrist position emitted to the
    // skeleton is the MCP centroid (more reliable under occlusion
    // than the hand-track wrist), but the wrist landmark itself is
    // needed to build the palm-orientation basis below.
    const INDEX_MCP_LOCAL: usize = 5;
    const MIDDLE_MCP_LOCAL: usize = 9;
    const PINKY_MCP_LOCAL: usize = 17;
    const MCP_LOCALS: [usize; 4] = [INDEX_MCP_LOCAL, MIDDLE_MCP_LOCAL, 13, PINKY_MCP_LOCAL];

    let sample_local = |local: usize| -> Option<([f32; 3], f32)> {
        let global = base_index + local;
        if global >= joints.len() {
            return None;
        }
        let j = &joints[global];
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            return None;
        }
        let p_cam = sample_metric_point(depth, j.nx, j.ny)?;
        Some((to_source(p_cam), j.score))
    };

    let mut sum = [0.0_f32; 3];
    let mut min_conf = f32::INFINITY;
    let mut found = 0_u32;
    let mut index_mcp_pos: Option<[f32; 3]> = None;
    let mut middle_mcp_pos: Option<[f32; 3]> = None;
    let mut pinky_mcp_pos: Option<[f32; 3]> = None;
    for &local in &MCP_LOCALS {
        if let Some((p, score)) = sample_local(local) {
            sum[0] += p[0];
            sum[1] += p[1];
            sum[2] += p[2];
            min_conf = min_conf.min(score);
            found += 1;
            match local {
                INDEX_MCP_LOCAL => index_mcp_pos = Some(p),
                MIDDLE_MCP_LOCAL => middle_mcp_pos = Some(p),
                PINKY_MCP_LOCAL => pinky_mcp_pos = Some(p),
                _ => {}
            }
        }
    }
    if found < 3 {
        return;
    }
    let inv = 1.0 / found as f32;
    sk.joints.insert(
        wrist_bone,
        SourceJoint {
            position: [sum[0] * inv, sum[1] * inv, sum[2] * inv],
            confidence: min_conf,
        },
    );

    // Palm orientation basis. Mirrors the standard rtmw3d hand
    // orientation derivation (see `tracking::rtmw3d::skeleton`):
    //
    //   raw_forward = middle_mcp − wrist        (wrist → middle finger)
    //   across      = index_mcp  − pinky_mcp    (across the palm)
    //   normal      = across × raw_forward       (palm plane normal)
    //   forward     = normal × across            (re-orthogonalised)
    //
    // Cross-orthogonalising forward against the normal means a noisy
    // hand-track wrist Z (occluded wrists routinely land behind the
    // torso) only perturbs the normal slightly without tilting the
    // in-palm forward direction. Without `HandOrientation` set, the
    // solver falls back to shortest-arc twist which is undefined
    // around the wrist↔fingertip axis — visually random hand roll.
    if let (Some(wri_p), Some(mid_p), Some(idx_p), Some(pin_p)) = (
        sample_local(0).map(|(p, _)| p),
        middle_mcp_pos,
        index_mcp_pos,
        pinky_mcp_pos,
    ) {
        let raw_forward = sub3(mid_p, wri_p);
        let across = sub3(idx_p, pin_p);
        let normal_raw = cross3(across, raw_forward);
        if let Some(normal) = normalize3(normal_raw) {
            let forward_raw = cross3(normal, across);
            if let Some(forward) = normalize3(forward_raw) {
                let orientation = HandOrientation {
                    forward,
                    up: normal,
                    confidence: min_conf,
                };
                match wrist_bone {
                    HumanoidBone::LeftHand => sk.left_hand_orientation = Some(orientation),
                    HumanoidBone::RightHand => sk.right_hand_orientation = Some(orientation),
                    _ => {}
                }
            }
        }
    }

    for &(local_idx, bone) in phalanges {
        let global = base_index + local_idx;
        if global >= joints.len() {
            continue;
        }
        let j = &joints[global];
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            continue;
        }
        if let Some(p_cam) = sample_metric_point(depth, j.nx, j.ny) {
            sk.joints.insert(
                bone,
                SourceJoint {
                    position: to_source(p_cam),
                    confidence: j.score,
                },
            );
        }
    }
    for &(local_idx, bone) in tips {
        let global = base_index + local_idx;
        if global >= joints.len() {
            continue;
        }
        let j = &joints[global];
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            continue;
        }
        if let Some(p_cam) = sample_metric_point(depth, j.nx, j.ny) {
            sk.fingertips.insert(
                bone,
                SourceJoint {
                    position: to_source(p_cam),
                    confidence: j.score,
                },
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Anatomical Z-plausibility clamps
// ---------------------------------------------------------------------------

/// Per-bone maximum |Δz| between a child joint and its parent joint
/// in source-space. These are the z-axis projections of the bone's
/// full length — reached only when the bone points exactly at or
/// directly away from the camera. Loose adult-body upper bounds keep
/// genuine fully-extended reaches untouched.
///
/// Why this exists: in side-profile / rear-facing poses, RTMW3D still
/// emits a high-confidence keypoint for the occluded elbow / wrist
/// (the model is trained to predict joint position even under
/// occlusion). DAv2's per-pixel depth at that 2D position can land
/// on a near-zero `d_raw` (background, clothing shadow, hair). The
/// metric back-projection `z = c / d_raw` then explodes to tens of
/// metres, and the avatar solver interprets the implausible elbow as
/// an arm pointing 20 m through the chest. Bulk diagnostic over
/// rotation sequences measured 5–10 such anomalies per 16-frame run
/// — exactly the frames where the user turns sideways briefly.
///
/// The clamp preserves the *sign* of the depth-derived Z (so the arm
/// still moves the right direction) and only caps the *magnitude* to
/// the anatomical bound. Less information loss than snap-to-parent.
const MAX_TORSO_DZ_M: f32 = 0.50; // hip mid → shoulder
const MAX_UPPER_ARM_DZ_M: f32 = 0.45; // shoulder → elbow
const MAX_LOWER_ARM_DZ_M: f32 = 0.40; // elbow → hand-wrist
const MAX_UPPER_LEG_DZ_M: f32 = 0.55; // hip → knee
const MAX_LOWER_LEG_DZ_M: f32 = 0.50; // knee → ankle
const MAX_FOOT_DZ_M: f32 = 0.25; // ankle → toe tip

/// Clamp `child.position[2]` so it lies within `parent.z ± max_dz`.
/// No-op when either joint is missing or the constraint is already
/// satisfied. Sign of `(child.z − parent.z)` is preserved.
fn clamp_z_to_parent(
    sk: &mut SourceSkeleton,
    parent: HumanoidBone,
    child: HumanoidBone,
    max_dz: f32,
) {
    let parent_z = match sk.joints.get(&parent) {
        Some(p) => p.position[2],
        None => return,
    };
    if let Some(c) = sk.joints.get_mut(&child) {
        let dz = c.position[2] - parent_z;
        if dz.abs() > max_dz {
            c.position[2] = parent_z + dz.signum() * max_dz;
        }
    }
}

/// Same as [`clamp_z_to_parent`] but for fingertip / toe-tip entries
/// stored in `sk.fingertips` (keyed by the bone they extend from).
fn clamp_fingertip_z_to_parent(
    sk: &mut SourceSkeleton,
    parent: HumanoidBone,
    fingertip_key: HumanoidBone,
    max_dz: f32,
) {
    let parent_z = match sk.joints.get(&parent) {
        Some(p) => p.position[2],
        None => return,
    };
    if let Some(c) = sk.fingertips.get_mut(&fingertip_key) {
        let dz = c.position[2] - parent_z;
        if dz.abs() > max_dz {
            c.position[2] = parent_z + dz.signum() * max_dz;
        }
    }
}

/// Walk the skeleton from origin outward, capping each bone's Z
/// projection against its parent's. Hips at z=0 by construction (when
/// hip-anchored); shoulders at z=0 in upper-body framing. Either way
/// the chain originates at zero, so each subsequent clamp narrows the
/// reachable Z range to anatomical bounds.
fn apply_anatomical_z_clamps(sk: &mut SourceSkeleton) {
    // Torso: shoulders against hips. RightShoulder + RightUpperArm
    // alias to the same depth sample (idx 5), and likewise for left
    // (idx 6) — clamp both for consistency.
    clamp_z_to_parent(sk, HumanoidBone::Hips, HumanoidBone::LeftShoulder, MAX_TORSO_DZ_M);
    clamp_z_to_parent(sk, HumanoidBone::Hips, HumanoidBone::RightShoulder, MAX_TORSO_DZ_M);
    clamp_z_to_parent(sk, HumanoidBone::Hips, HumanoidBone::LeftUpperArm, MAX_TORSO_DZ_M);
    clamp_z_to_parent(sk, HumanoidBone::Hips, HumanoidBone::RightUpperArm, MAX_TORSO_DZ_M);

    // Arms: elbow vs shoulder, hand vs elbow.
    clamp_z_to_parent(
        sk,
        HumanoidBone::LeftShoulder,
        HumanoidBone::LeftLowerArm,
        MAX_UPPER_ARM_DZ_M,
    );
    clamp_z_to_parent(
        sk,
        HumanoidBone::RightShoulder,
        HumanoidBone::RightLowerArm,
        MAX_UPPER_ARM_DZ_M,
    );
    clamp_z_to_parent(
        sk,
        HumanoidBone::LeftLowerArm,
        HumanoidBone::LeftHand,
        MAX_LOWER_ARM_DZ_M,
    );
    clamp_z_to_parent(
        sk,
        HumanoidBone::RightLowerArm,
        HumanoidBone::RightHand,
        MAX_LOWER_ARM_DZ_M,
    );

    // Legs: knee vs hip, ankle vs knee.
    clamp_z_to_parent(sk, HumanoidBone::Hips, HumanoidBone::LeftUpperLeg, MAX_TORSO_DZ_M);
    clamp_z_to_parent(sk, HumanoidBone::Hips, HumanoidBone::RightUpperLeg, MAX_TORSO_DZ_M);
    clamp_z_to_parent(
        sk,
        HumanoidBone::LeftUpperLeg,
        HumanoidBone::LeftLowerLeg,
        MAX_UPPER_LEG_DZ_M,
    );
    clamp_z_to_parent(
        sk,
        HumanoidBone::RightUpperLeg,
        HumanoidBone::RightLowerLeg,
        MAX_UPPER_LEG_DZ_M,
    );
    clamp_z_to_parent(
        sk,
        HumanoidBone::LeftLowerLeg,
        HumanoidBone::LeftFoot,
        MAX_LOWER_LEG_DZ_M,
    );
    clamp_z_to_parent(
        sk,
        HumanoidBone::RightLowerLeg,
        HumanoidBone::RightFoot,
        MAX_LOWER_LEG_DZ_M,
    );

    // Toe tips (in `sk.fingertips`, keyed by the foot bone).
    clamp_fingertip_z_to_parent(
        sk,
        HumanoidBone::LeftFoot,
        HumanoidBone::LeftFoot,
        MAX_FOOT_DZ_M,
    );
    clamp_fingertip_z_to_parent(
        sk,
        HumanoidBone::RightFoot,
        HumanoidBone::RightFoot,
        MAX_FOOT_DZ_M,
    );
}

// ---------------------------------------------------------------------------
// Anatomical bone-length Z reconstruction (arms only)
// ---------------------------------------------------------------------------

/// Adult anatomical reference for the upper arm (shoulder → elbow).
/// Pinned to the same body scale as `ASSUMED_SHOULDER_SPAN_M` =
/// 0.40 m used by the depth calibrator: shoulder-span × 0.75.
const EXPECTED_UPPER_ARM_M: f32 = 0.30;
/// Adult anatomical reference for the forearm (elbow → wrist/hand).
/// shoulder-span × 0.625.
const EXPECTED_LOWER_ARM_M: f32 = 0.25;

/// If the depth-derived 3D bone length is below this fraction of the
/// anatomical expectation, treat it as a magnitude underestimate
/// (the typical sleeveless-vs-long-sleeve case where DAv2 sees less
/// depth contrast at the elbow) and reconstruct.
const UNDERESTIMATE_RATIO: f32 = 0.7;

/// Minimum |Δz| to trust the depth signal's *sign*. Below this, the
/// raw value is dominated by noise and reconstructing to a full
/// expected magnitude would force the bone forward or backward
/// arbitrarily. Leaving the small dz alone lets the bone sit near
/// the camera plane, which matches what a noisy depth signal
/// physically represents.
///
/// Empirically tuned via the high_skin vs frilly outfit comparison
/// (9 pose cells × 2 sides = 18 measurements per setting):
///   floor 0.02 → mean angle diff 26.8° / max 69° (best)
///   floor 0.08 → 33.7° / 111° (sign-flip risk on near-zero readings)
///   no reconstruction → 35.1° / 93°
/// 0.02 wins because the depth-pipeline rarely produces |dz| < 2 cm
/// without genuinely meaning "arm in image plane" — the threshold
/// catches that case while leaving room to amplify the 5–10 cm
/// sleeveless underestimates.
const MIN_DZ_FOR_SIGN: f32 = 0.02;

/// For each arm bone (shoulder→elbow, elbow→hand), check whether the
/// depth-derived 3D length is suspiciously short. If so, keep the
/// 2D component (X/Y, well-tracked from RTMW3D) and reconstruct the
/// Z magnitude from the anatomical bone length, taking the sign from
/// the raw depth signal.
///
/// **Why**: bulk diagnostic on the high_skin vs frilly outfit
/// comparison (same scene, same pose, only sleeve difference) measured
/// |Δz_elbow_high − Δz_elbow_frilly| up to 17 cm. DAv2-Small sees
/// less depth contrast on bare skin than on a fabric-vs-skin
/// boundary, so sleeveless arms underestimate the elbow's forward
/// extension by 25–73 %. The depth signal's *sign* is reliable
/// (forward arms read forward); only the *magnitude* is sleeve-
/// dependent. Reconstructing magnitude from anatomy makes the
/// avatar's bone direction outfit-independent.
///
/// Skipped when the raw depth length is already plausible (≥ 70 % of
/// expected; long-sleeve / good-contrast case) and when the raw |Δz|
/// is below the noise floor (sign unreliable).
///
/// Arms only: legs are typically clothed (pants/skirt depth contrast
/// is good), and the user's reported issue was elbow Z specifically.
fn reconstruct_arm_z_magnitudes(sk: &mut SourceSkeleton, opts: BuildOptions) {
    let upper = opts.subject_upper_arm_m.unwrap_or(EXPECTED_UPPER_ARM_M);
    let lower = opts.subject_lower_arm_m.unwrap_or(EXPECTED_LOWER_ARM_M);
    reconstruct_bone_z(
        sk,
        HumanoidBone::LeftShoulder,
        HumanoidBone::LeftLowerArm,
        upper,
    );
    reconstruct_bone_z(
        sk,
        HumanoidBone::RightShoulder,
        HumanoidBone::RightLowerArm,
        upper,
    );
    reconstruct_bone_z(
        sk,
        HumanoidBone::LeftLowerArm,
        HumanoidBone::LeftHand,
        lower,
    );
    reconstruct_bone_z(
        sk,
        HumanoidBone::RightLowerArm,
        HumanoidBone::RightHand,
        lower,
    );
}

fn reconstruct_bone_z(
    sk: &mut SourceSkeleton,
    parent: HumanoidBone,
    child: HumanoidBone,
    expected_length_m: f32,
) {
    let parent_pos = match sk.joints.get(&parent) {
        Some(p) => p.position,
        None => return,
    };
    let cj = match sk.joints.get_mut(&child) {
        Some(c) => c,
        None => return,
    };
    let dx = cj.position[0] - parent_pos[0];
    let dy = cj.position[1] - parent_pos[1];
    let dz_raw = cj.position[2] - parent_pos[2];
    let xy_len_sq = dx * dx + dy * dy;
    let depth_3d_len_sq = xy_len_sq + dz_raw * dz_raw;
    let exp_sq = expected_length_m * expected_length_m;
    let threshold_sq = (UNDERESTIMATE_RATIO * expected_length_m).powi(2);

    if depth_3d_len_sq >= threshold_sq {
        // Depth-derived bone is roughly anatomical length. Trust it
        // — long-sleeve / good-contrast case.
        return;
    }
    if dz_raw.abs() < MIN_DZ_FOR_SIGN {
        // Depth signal too weak to trust the sign. Leave dz near zero
        // — corresponds physically to "arm in image plane".
        return;
    }
    if xy_len_sq >= exp_sq {
        // 2D extent already accounts for the full bone length — the
        // arm is exactly perpendicular to the camera. No Z component.
        cj.position[2] = parent_pos[2];
        return;
    }
    // Reconstruct: keep 2D position, set Z so the 3D bone length
    // equals the anatomical expectation. Sign follows the depth
    // signal so forward arms stay forward.
    let dz_target = (exp_sq - xy_len_sq).sqrt();
    let sign = if dz_raw >= 0.0 { 1.0 } else { -1.0 };
    cj.position[2] = parent_pos[2] + sign * dz_target;
}

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn normalize3(v: [f32; 3]) -> Option<[f32; 3]> {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-6 {
        Some([v[0] / len, v[1] / len, v[2] / len])
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Face cascade helpers
// ---------------------------------------------------------------------------

/// Derive head yaw / pitch / roll from face keypoints 0..=4 (nose /
/// left_eye / right_eye / left_ear / right_ear), each lifted into 3D
/// via metric-depth sampling and routed through the same axis flip +
/// hip-origin convention as the body skeleton.
///
/// Logic mirrors [`super::rtmw3d::face::derive_face_pose_from_body`]:
/// ear-line yaw, eye-line roll, eye-mid → nose pitch. Confidence is
/// the min of the 5 input scores. Includes the back-ear reflection
/// fallback for >45° head turns.
pub(super) fn derive_face_pose_metric(
    joints: &[DecodedJoint2d],
    depth: &MoGeFrame,
    origin: [f32; 3],
) -> Option<FacePose> {
    if joints.len() < 5 {
        return None;
    }
    for j in joints.iter().take(5) {
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            return None;
        }
    }

    let to_src = |idx: usize| -> Option<[f32; 3]> {
        let j = &joints[idx];
        let p_cam = sample_metric_point(depth, j.nx, j.ny)?;
        Some([
            -(p_cam[0] - origin[0]),
            -(p_cam[1] - origin[1]),
            -(p_cam[2] - origin[2]),
        ])
    };

    let nose = to_src(0)?;
    // Selfie mirror: subject's left maps to avatar's right.
    let avatar_right_eye = to_src(1)?;
    let avatar_left_eye = to_src(2)?;
    let avatar_right_ear = to_src(3)?;
    let avatar_left_ear = to_src(4)?;

    const EAR_OCCLUSION_RATIO: f32 = 0.5;
    let s_right_ear = joints[3].score;
    let s_left_ear = joints[4].score;
    let (right_ear_xz, left_ear_xz) = if s_right_ear < s_left_ear * EAR_OCCLUSION_RATIO {
        let recon = [
            2.0 * nose[0] - avatar_left_ear[0],
            avatar_left_ear[1],
            2.0 * nose[2] - avatar_left_ear[2],
        ];
        (recon, avatar_left_ear)
    } else if s_left_ear < s_right_ear * EAR_OCCLUSION_RATIO {
        let recon = [
            2.0 * nose[0] - avatar_right_ear[0],
            avatar_right_ear[1],
            2.0 * nose[2] - avatar_right_ear[2],
        ];
        (avatar_right_ear, recon)
    } else {
        (avatar_right_ear, avatar_left_ear)
    };
    let ear_dx = left_ear_xz[0] - right_ear_xz[0];
    let ear_dz = left_ear_xz[2] - right_ear_xz[2];
    let ear_dist = (ear_dx * ear_dx + ear_dz * ear_dz).sqrt();
    if ear_dist < 0.02 {
        return None;
    }
    let yaw = ear_dz.atan2(ear_dx);

    let eye_dx = avatar_left_eye[0] - avatar_right_eye[0];
    let eye_dy = avatar_left_eye[1] - avatar_right_eye[1];
    let roll = eye_dy.atan2(eye_dx);

    let mid_eye_y = (avatar_left_eye[1] + avatar_right_eye[1]) * 0.5;
    let face_width = (eye_dx * eye_dx + eye_dy * eye_dy).sqrt().max(0.02);
    let pitch_signal = (mid_eye_y - nose[1]) / face_width;
    let pitch = pitch_signal.clamp(-2.0, 2.0).atan();

    let conf = joints[..5].iter().map(|j| j.score).fold(1.0_f32, f32::min);
    Some(FacePose {
        yaw,
        pitch,
        roll,
        confidence: conf,
    })
}

/// Pixel-space face bbox derived from face-68 landmarks (indices
/// 23..=90). Mirrors [`super::rtmw3d::face::build_face_bbox_from_joints`]
/// but consumes the [`DecodedJoint2d`] type.
pub(super) fn build_face_bbox_from_joints_2d(
    joints: &[DecodedJoint2d],
    width: u32,
    height: u32,
) -> Option<FaceBbox> {
    let mut points_px: Vec<(f32, f32)> = Vec::with_capacity(68);
    for i in 23..=90 {
        let Some(j) = joints.get(i) else { continue };
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            continue;
        }
        points_px.push((j.nx * width as f32, j.ny * height as f32));
    }
    derive_face_bbox(&points_px, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_frame(width: u32, height: u32, point: [f32; 3]) -> MoGeFrame {
        let pixels = (width * height) as usize;
        MoGeFrame {
            width,
            height,
            points_m: vec![point; pixels],
        }
    }

    #[test]
    fn sample_metric_point_returns_pixel() {
        let frame = dummy_frame(644, 476, [0.5, 0.4, 1.2]);
        let p = sample_metric_point(&frame, 0.5, 0.5).unwrap();
        assert!((p[0] - 0.5).abs() < 1e-6);
        assert!((p[1] - 0.4).abs() < 1e-6);
        assert!((p[2] - 1.2).abs() < 1e-6);
    }

    #[test]
    fn sample_outside_frame_returns_none() {
        let frame = dummy_frame(644, 476, [0.0, 0.0, 1.0]);
        assert!(sample_metric_point(&frame, -0.1, 0.5).is_none());
        assert!(sample_metric_point(&frame, 1.1, 0.5).is_none());
    }

    fn jt(z: f32) -> SourceJoint {
        SourceJoint {
            position: [0.0, 0.0, z],
            confidence: 1.0,
        }
    }

    #[test]
    fn z_clamp_caps_runaway_elbow_to_anatomical_bound() {
        // Simulates an occluded elbow whose DAv2 sample landed on a
        // near-zero d_raw and back-projected to z = -22 m. Shoulder
        // at 0.30 m forward of hips. Expected: clamped to
        // shoulder ± MAX_UPPER_ARM_DZ_M, sign preserved (-22 → forward),
        // so elbow.z = shoulder.z + (-MAX_UPPER_ARM_DZ_M) = -0.15.
        let mut sk = SourceSkeleton::empty(0);
        sk.joints.insert(HumanoidBone::Hips, jt(0.0));
        sk.joints.insert(HumanoidBone::LeftShoulder, jt(0.30));
        sk.joints.insert(HumanoidBone::LeftLowerArm, jt(-22.0));
        apply_anatomical_z_clamps(&mut sk);
        let elbow_z = sk.joints[&HumanoidBone::LeftLowerArm].position[2];
        let expected = 0.30 + (-1.0) * MAX_UPPER_ARM_DZ_M;
        assert!(
            (elbow_z - expected).abs() < 1e-5,
            "elbow.z {} != expected {}",
            elbow_z,
            expected
        );
    }

    #[test]
    fn z_clamp_no_op_when_within_bounds() {
        // Realistic arms-forward pose: elbow 22 cm forward of shoulder.
        // Well within MAX_UPPER_ARM_DZ_M = 0.45 m, so untouched.
        let mut sk = SourceSkeleton::empty(0);
        sk.joints.insert(HumanoidBone::Hips, jt(0.0));
        sk.joints.insert(HumanoidBone::LeftShoulder, jt(0.10));
        sk.joints.insert(HumanoidBone::LeftLowerArm, jt(0.32));
        apply_anatomical_z_clamps(&mut sk);
        assert!((sk.joints[&HumanoidBone::LeftLowerArm].position[2] - 0.32).abs() < 1e-5);
    }

    #[test]
    fn z_clamp_skips_missing_parent() {
        // Upper-body framing: hips absent. Shoulder/elbow clamps that
        // depend on hips become no-ops; shoulder→elbow clamp still runs.
        let mut sk = SourceSkeleton::empty(0);
        sk.joints.insert(HumanoidBone::LeftShoulder, jt(2.0)); // would be capped if hips present
        sk.joints.insert(HumanoidBone::LeftLowerArm, jt(2.10));
        apply_anatomical_z_clamps(&mut sk);
        // Shoulder unchanged (no parent in sk.joints).
        assert!((sk.joints[&HumanoidBone::LeftShoulder].position[2] - 2.0).abs() < 1e-5);
        // Elbow within bound of shoulder (Δ=0.10 < 0.45) → unchanged.
        assert!((sk.joints[&HumanoidBone::LeftLowerArm].position[2] - 2.10).abs() < 1e-5);
    }

    #[test]
    fn z_reconstruct_amplifies_underestimated_forward_arm() {
        // Sleeveless case: shoulder at origin, elbow with weak forward
        // depth (10 cm — above the sign-trust floor of 8 cm) but in 2D
        // the elbow is almost directly in front of the shoulder (xy ≈ 0).
        // Anatomical upper arm = 0.30 m, so elbow Z should be
        // reconstructed to +0.30 m forward.
        let mut sk = SourceSkeleton::empty(0);
        sk.joints.insert(HumanoidBone::LeftShoulder, jt(0.0));
        sk.joints.insert(
            HumanoidBone::LeftLowerArm,
            SourceJoint {
                position: [0.0, 0.0, 0.10],
                confidence: 1.0,
            },
        );
        reconstruct_arm_z_magnitudes(&mut sk, BuildOptions::default());
        let z = sk.joints[&HumanoidBone::LeftLowerArm].position[2];
        assert!(
            (z - EXPECTED_UPPER_ARM_M).abs() < 1e-5,
            "elbow z should be reconstructed to {}, got {}",
            EXPECTED_UPPER_ARM_M,
            z,
        );
    }

    #[test]
    fn z_reconstruct_uses_subject_arm_length_when_provided() {
        // PoseCalibration with measured shoulder span = 0.50 m
        // (taller-than-average user). Derived upper arm = 0.375 m.
        // With xy ≈ 0 and a weak forward signal, reconstruction should
        // target 0.375, not the 0.30 hardcoded default.
        let mut sk = SourceSkeleton::empty(0);
        sk.joints.insert(HumanoidBone::LeftShoulder, jt(0.0));
        sk.joints.insert(
            HumanoidBone::LeftLowerArm,
            SourceJoint {
                position: [0.0, 0.0, 0.10],
                confidence: 1.0,
            },
        );
        let opts = BuildOptions {
            subject_upper_arm_m: Some(0.375),
            subject_lower_arm_m: Some(0.3125),
            ..Default::default()
        };
        reconstruct_arm_z_magnitudes(&mut sk, opts);
        let z = sk.joints[&HumanoidBone::LeftLowerArm].position[2];
        assert!(
            (z - 0.375).abs() < 1e-5,
            "elbow z should match subject_upper_arm_m=0.375, got {}",
            z,
        );
    }

    #[test]
    fn z_reconstruct_skips_when_depth_already_plausible() {
        // Long-sleeve case: depth-derived 3D length 0.226 m (xy=0.05,
        // dz=0.22). 0.226 ≥ 0.7 × 0.30 = 0.21, so reconstruction
        // skips — depth is trusted.
        let mut sk = SourceSkeleton::empty(0);
        sk.joints.insert(HumanoidBone::LeftShoulder, jt(0.0));
        sk.joints.insert(
            HumanoidBone::LeftLowerArm,
            SourceJoint {
                position: [0.05, 0.0, 0.22],
                confidence: 1.0,
            },
        );
        reconstruct_arm_z_magnitudes(&mut sk, BuildOptions::default());
        assert!((sk.joints[&HumanoidBone::LeftLowerArm].position[2] - 0.22).abs() < 1e-5);
    }

    #[test]
    fn z_reconstruct_skips_when_dz_below_sign_floor() {
        // Arm in image plane with negligible Z signal. Reconstruction
        // would otherwise jam the arm forward arbitrarily; instead we
        // leave the small dz alone.
        let mut sk = SourceSkeleton::empty(0);
        sk.joints.insert(HumanoidBone::LeftShoulder, jt(0.0));
        sk.joints.insert(
            HumanoidBone::LeftLowerArm,
            SourceJoint {
                position: [0.05, 0.05, 0.005], // |dz| < MIN_DZ_FOR_SIGN
                confidence: 1.0,
            },
        );
        reconstruct_arm_z_magnitudes(&mut sk, BuildOptions::default());
        assert!((sk.joints[&HumanoidBone::LeftLowerArm].position[2] - 0.005).abs() < 1e-5);
    }

    #[test]
    fn z_reconstruct_t_pose_unchanged() {
        // T-pose arm: xy distance ≈ upper arm length, dz tiny.
        // depth_3d_len ≈ xy ≈ 0.30m, well above the 0.21m
        // underestimate threshold → reconstruction skips, the dz
        // signal (whatever it is) passes through untouched.
        let mut sk = SourceSkeleton::empty(0);
        sk.joints.insert(HumanoidBone::LeftShoulder, jt(0.0));
        sk.joints.insert(
            HumanoidBone::LeftLowerArm,
            SourceJoint {
                position: [EXPECTED_UPPER_ARM_M, 0.0, 0.05],
                confidence: 1.0,
            },
        );
        reconstruct_arm_z_magnitudes(&mut sk, BuildOptions::default());
        assert!((sk.joints[&HumanoidBone::LeftLowerArm].position[2] - 0.05).abs() < 1e-5);
    }

    #[test]
    fn z_reconstruct_preserves_backward_sign() {
        // Arm reaching back (negative Z in source space): dz_raw = -0.10
        // above the sign-trust floor, sleeveless underestimate of the
        // true 0.30 m. Reconstructed magnitude should also be negative.
        let mut sk = SourceSkeleton::empty(0);
        sk.joints.insert(HumanoidBone::LeftShoulder, jt(0.0));
        sk.joints.insert(
            HumanoidBone::LeftLowerArm,
            SourceJoint {
                position: [0.0, 0.0, -0.10],
                confidence: 1.0,
            },
        );
        reconstruct_arm_z_magnitudes(&mut sk, BuildOptions::default());
        let z = sk.joints[&HumanoidBone::LeftLowerArm].position[2];
        assert!(
            (z + EXPECTED_UPPER_ARM_M).abs() < 1e-5,
            "elbow z should be -{}, got {}",
            EXPECTED_UPPER_ARM_M,
            z,
        );
    }

    #[test]
    fn z_clamp_propagates_through_chain() {
        // Hip 0, Shoulder clamped 0→0.50, Elbow then clamped against
        // (clamped) shoulder to 0.95, Hand against (clamped) elbow to 1.35.
        let mut sk = SourceSkeleton::empty(0);
        sk.joints.insert(HumanoidBone::Hips, jt(0.0));
        sk.joints.insert(HumanoidBone::LeftShoulder, jt(5.0));
        sk.joints.insert(HumanoidBone::LeftUpperArm, jt(5.0));
        sk.joints.insert(HumanoidBone::LeftLowerArm, jt(5.0));
        sk.joints.insert(HumanoidBone::LeftHand, jt(5.0));
        apply_anatomical_z_clamps(&mut sk);
        assert!((sk.joints[&HumanoidBone::LeftShoulder].position[2] - 0.50).abs() < 1e-5);
        assert!((sk.joints[&HumanoidBone::LeftLowerArm].position[2] - 0.95).abs() < 1e-5);
        assert!((sk.joints[&HumanoidBone::LeftHand].position[2] - 1.35).abs() < 1e-5);
    }
}
