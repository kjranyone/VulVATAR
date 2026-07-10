//! Shared "2D keypoints + per-pixel metric depth → SourceSkeleton"
//! pipeline, consumed by [`super::rtmw3d_with_depth`].
//!
//! The provider feeds RTMW3D-sourced 2D keypoints and an absolute
//! metric depth map (RealSense D435) shaped as a [`MetricDepthFrame`]
//! point cloud with [`DecodedJoint2d`]-shaped 2D landmarks. The
//! skeleton-building math (origin selection, axis flips, hand chain
//! attachment, head-pose derivation) is single-sourced here.
//!
//! The skeleton *builders* run only when a depth source is present —
//! i.e. the `realsense` feature. Without it the module is still compiled
//! (it is `inference`-gated for the shared [`MetricDepthFrame`] /
//! [`DecodedJoint2d`] / `TorsoCaptureBuffer` types) but the builders are
//! dormant, hence the dead-code allowance below.
#![cfg_attr(not(feature = "realsense"), allow(dead_code))]

use crate::asset::HumanoidBone;

use super::source_skeleton::{CameraIntrinsics, HandOrientation, MetricFrameInfo};
use super::{SourceJoint, SourceSkeleton};

/// Number of COCO-Wholebody keypoints every 2D decode emits.
pub const NUM_JOINTS: usize = 133;

/// A decoded whole-frame 2D keypoint, normalised image-relative.
#[derive(Clone, Copy, Debug, Default)]
pub struct DecodedJoint2d {
    /// Image-relative `[0, 1]`, x-right.
    pub nx: f32,
    /// Image-relative `[0, 1]`, y-down.
    pub ny: f32,
    /// Confidence in `[0, 1]`.
    pub score: f32,
}

/// Describes the region of the full camera frame that a depth map
/// covers when the depth model was fed a cropped (person-only) input
/// rather than the full frame. All values are in normalised
/// full-frame coordinates `[0, 1]`. `None` on `MetricDepthFrame::crop`
/// means the depth map covers the entire frame.
#[derive(Clone, Debug)]
pub struct FrameCrop {
    /// Left edge of the crop in normalised full-frame X.
    pub x1_frac: f32,
    /// Top edge of the crop in normalised full-frame Y.
    pub y1_frac: f32,
    /// Width of the crop as a fraction of full-frame width.
    pub w_frac: f32,
    /// Height of the crop as a fraction of full-frame height.
    pub h_frac: f32,
}

/// Per-pixel metric point cloud, in metres, in the depth model's
/// input pixel grid (`width × height`). Invalid pixels (mask below
/// threshold or non-finite output) are stored as `NaN` in `points_m`
/// so the consumer's local-window median sampling naturally rejects
/// them. Per-pixel metric Z is `points_m[i][2]` — no separate depth
/// map is kept since the only current consumer samples the full xyz,
/// not depth alone.
#[derive(Clone)]
pub struct MetricDepthFrame {
    pub width: u32,
    pub height: u32,
    pub points_m: Vec<[f32; 3]>,
    /// If this frame was produced from a cropped input (e.g. DAv2 fed
    /// a YOLOX person crop), the crop region in full-frame normalised
    /// coordinates. `None` means the depth map covers the entire frame.
    pub crop: Option<FrameCrop>,
    /// Pinhole intrinsics of the colour image this depth was aligned to,
    /// carried through so the built skeleton can expose them on
    /// [`MetricFrameInfo`] for the 1:1 sensor-matched render. `None` when
    /// the frame was synthesised without a real sensor (tests).
    pub intrinsics: Option<CameraIntrinsics>,
}

/// Per-keypoint visibility floor — anything below is treated as "no
/// detection" rather than a low-confidence detection. The user-tunable
/// `joint_confidence_threshold` slider lives in the solver.
pub const KEYPOINT_VISIBILITY_FLOOR: f32 = 0.05;

/// Half-size of the square depth-sampling window when reading metric
/// depth at a 2D keypoint. Small enough to track fingers without
/// dragging in neighbouring depth, large enough that single-pixel
/// model artefacts get out-voted by the median.
pub const SAMPLE_RADIUS_PX: i32 = 3;

/// Target shoulder span, in source units, that the metric skeleton is
/// normalised to (`position *= 1/mpsu`, `mpsu = shoulder_span_m / this`).
/// Chosen to reproduce the numeric range the monocular RTMW3D source frame
/// produced so the solver's position-magnitude tunings (dead-zones, 1€
/// filter cutoffs) keep holding. Rotation is scale-invariant so this never
/// affects orientation — only keeps thresholds in-range.
/// TODO(verify Tier 1): confirm against a real monocular frame's measured
/// shoulder span; seeded from the `sx = -(Δnx)·2·aspect` geometry.
/// `pub` so the offline `diagnose_depth_replay` bin locks the value against
/// a real recorded frame (the Tier 1 verification the TODO refers to).
pub const TARGET_SRC_SHOULDER_SPAN: f32 = 0.75;

/// Remap full-frame normalised `(nx, ny)` to crop-local normalised
/// coordinates when the depth frame covers only a sub-region of the
/// camera frame. Returns `None` if the point falls outside the crop.
#[inline]
fn remap_to_crop(crop: &FrameCrop, nx: f32, ny: f32) -> Option<(f32, f32)> {
    let cnx = (nx - crop.x1_frac) / crop.w_frac;
    let cny = (ny - crop.y1_frac) / crop.h_frac;
    if cnx < 0.0 || cnx > 1.0 || cny < 0.0 || cny > 1.0 {
        return None;
    }
    Some((cnx, cny))
}

// ---------------------------------------------------------------------------
// COCO-Wholebody index → HumanoidBone tables. Same convention RTMW3D
// uses; selfie mirror = subject's anatomical-left → avatar `Right*`
// bone. Single-sourced here so any future tweak (e.g. excluding a
// noisy keypoint) lands in both providers.
// ---------------------------------------------------------------------------

// The COCO body/leg index → bone mapping is now applied inline in
// `build_skeleton` (torso from the surface fit, limbs from the ray–sphere
// chain: 5/6 shoulders, 7/8 elbows, 11/12 hips, 13/14 knees, 15/16 ankles,
// 17/20 toe-tips — selfie mirror), so only the hand-block tables live here.
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
pub fn sample_metric_point(frame: &MetricDepthFrame, nx: f32, ny: f32) -> Option<[f32; 3]> {
    sample_metric_point_with_radius(frame, nx, ny, SAMPLE_RADIUS_PX)
}

/// Variant of [`sample_metric_point`] that takes the median window
/// half-size as an argument. Used by diagnostics to compare radius=0
/// (single pixel) / 1 (3×3) / 3 (7×7, the runtime default).
pub fn sample_metric_point_with_radius(
    frame: &MetricDepthFrame,
    nx: f32,
    ny: f32,
    radius_px: i32,
) -> Option<[f32; 3]> {
    if !nx.is_finite() || !ny.is_finite() {
        return None;
    }
    let (nx, ny) = match &frame.crop {
        Some(c) => match remap_to_crop(c, nx, ny) {
            Some(remapped) => remapped,
            None => return None,
        },
        None => (nx, ny),
    };
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
    pub const GRID: usize = crate::tracking::TorsoDepthTemplate::GRID_SIZE as usize;

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
    pub fn add_frame(&mut self, joints: &[DecodedJoint2d], frame: &MetricDepthFrame) -> bool {
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
}

/// Build [`BuildOptions`] from the provider's cached
/// `pose_calibration` and the GUI's transient calibration-modal mode
/// hint. Single-sourced so any future calibration-derived knob
/// (c-clamp from jitter, anchor-bias rejection) lands in one place.
///
/// `mode_hint` carries the mode the user has selected in the
/// **currently open** calibration modal. When `Some`, it **overrides**
/// the persisted calibration's mode for the `force_shoulder_anchor`
/// decision in either direction:
///
/// * `Some(UpperBody)` → force shoulder anchor (suppress phantom hip),
///   even if the persisted calibration is `FullBody` or absent. This
///   is what makes the *first* `UpperBody` capture possible — until
///   `persist_calibration` runs there is no persisted flag to flip.
/// * `Some(FullBody)`  → do **not** force shoulder anchor, even if
///   the persisted calibration is `UpperBody`. Without this branch,
///   re-calibrating from `UpperBody` to `FullBody` is impossible: the
///   persisted flag stays true, hip never appears as the anchor, and
///   every collected sample is rejected by the GUI's
///   `pose.root_anchor_is_hip` gate.
/// * `None`            → modal closed, fall back to the persisted
///   calibration's mode.
pub(super) fn build_options_from_calibration(
    calibration: Option<&super::PoseCalibration>,
    mode_hint: Option<super::CalibrationMode>,
) -> BuildOptions {
    let force_shoulder_anchor = match mode_hint {
        Some(super::CalibrationMode::UpperBody) => true,
        Some(super::CalibrationMode::FullBody) => false,
        None => calibration
            .map(|c| matches!(c.mode, super::CalibrationMode::UpperBody))
            .unwrap_or(false),
    };
    BuildOptions {
        force_shoulder_anchor,
    }
}

// ---------------------------------------------------------------------------
// Torso surface fit (geometric model)
// ---------------------------------------------------------------------------

/// Fit the torso as a **surface** from the depth point cloud, rather than
/// sampling depth at each torso keypoint's own (possibly-background or
/// occluded) pixel. Torso keypoints then get their metric position by
/// intersecting their pixel ray with the fitted plane, so a shoulder
/// keypoint that lands on the wall behind the subject still resolves onto
/// the body surface. This is the metric-native replacement for the whole
/// back-percentile / template-bias / z-band / border-gate stack.
pub(super) mod torso_fit {
    use super::{
        BuildOptions, DecodedJoint2d, MetricDepthFrame, KEYPOINT_VISIBILITY_FLOOR, NUM_JOINTS,
    };

    /// Torso solution consumed by `build_skeleton`: each joint is a direct
    /// windowed-median depth sample at its keypoint (camera-space metres,
    /// x-right / y-down / z-forward), with the shoulder pair occlusion-guarded.
    #[derive(Clone, Copy, Debug)]
    pub(in crate::tracking) struct TorsoFit {
        /// COCO 5 → avatar Right shoulder.
        pub r_shoulder_cam: Option<[f32; 3]>,
        /// COCO 6 → avatar Left shoulder.
        pub l_shoulder_cam: Option<[f32; 3]>,
        /// COCO 11 → avatar RightUpperLeg (hip).
        pub r_hip_cam: Option<[f32; 3]>,
        /// COCO 12 → avatar LeftUpperLeg (hip).
        pub l_hip_cam: Option<[f32; 3]>,
        /// Anchor origin (hip-mid preferred, shoulder-mid fallback).
        pub anchor_cam: [f32; 3],
        pub anchor_is_hip: bool,
        pub anchor_score: f32,
    }

    #[inline]
    fn visible(joints: &[DecodedJoint2d], idx: usize) -> Option<&DecodedJoint2d> {
        joints
            .get(idx)
            .filter(|j| j.score >= KEYPOINT_VISIBILITY_FLOOR)
    }

    #[inline]
    fn mid(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5]
    }

    /// Move a camera-space point along its observation ray (origin→point) to a
    /// new depth `z`. Used to re-seat a foreground-occluded shoulder onto the
    /// body surface without changing its 2D pixel direction.
    #[inline]
    fn ray_to_depth(p: [f32; 3], z: f32) -> [f32; 3] {
        if p[2].abs() < 1e-6 {
            return p;
        }
        let s = z / p[2];
        [p[0] * s, p[1] * s, p[2] * s]
    }

    /// Depth gap (m) between the two shoulders above which the shallower one is
    /// taken to be **foreground-occluded** (a hand held in front of it) rather
    /// than a genuine torso turn. A desk-up self-view is front-facing by prior;
    /// even a 40° turn separates the shoulders by only ~0.12 m in depth, so a
    /// gap past this is the sensor seeing the occluder, not the shoulder.
    const OCCLUSION_GAP_M: f32 = 0.20;
    /// Nominal biacromial (shoulder) span (m) used as the rigidity reference.
    /// The two shoulders are a rigid segment ~this long; a sampled pair whose
    /// 3-D separation collapses far below this is not two shoulders but the
    /// detector drawing both onto the central hands (namaste) — unobservable.
    const ANATOMICAL_SPAN_M: f32 = 0.36;
    /// Reject the shoulder pair as a valid orientation cue when its measured
    /// 3-D separation falls below this fraction of [`ANATOMICAL_SPAN_M`].
    const MIN_RIGID_SPAN_FRAC: f32 = 0.7;

    /// Solve the torso from **direct local depth samples** at the shoulder/hip
    /// keypoints (the sensor's own measurement), with a front-facing occlusion
    /// clamp so a hand in front of a shoulder can't tilt the torso.
    ///
    /// This supersedes the earlier plane-fit: on the desk-up geometry the PCA
    /// plane fitted to the (perspective-foreshortened, desk-extended) region is
    /// near-horizontal and unstable frame-to-frame (its normal's y-component
    /// swings from +0.9 to −1.0), so intersecting the shoulder rays with it
    /// extrapolates them metres away and injects a spurious ±60° yaw. A
    /// windowed-median sample at each shoulder is both more faithful and far
    /// steadier. `None` only when no shoulder (nor hip, full-body) yields a
    /// sample — the sole "emit empty skeleton" case.
    pub(in crate::tracking) fn fit_torso(
        frame: &MetricDepthFrame,
        joints: &[DecodedJoint2d],
        opts: BuildOptions,
    ) -> Option<TorsoFit> {
        if joints.len() < NUM_JOINTS {
            return None;
        }
        let _ = frame.intrinsics?; // the metric path always carries intrinsics

        // Direct windowed-median depth sample at a torso keypoint, deprojected
        // to camera-space metres. Robust to a few edge pixels; no plane needed.
        let sample = |idx: usize| -> Option<[f32; 3]> {
            let j = visible(joints, idx)?;
            super::sample_metric_point(frame, j.nx, j.ny)
        };
        let mut r_shoulder_cam = sample(5);
        let mut l_shoulder_cam = sample(6);

        // Occlusion / degeneracy guard on the shoulder pair. A valid pair is a
        // rigid ~one-span segment with both shoulders on the body surface. Two
        // failure modes make the torso orientation unobservable:
        //   • foreground occlusion — a hand in front of one shoulder pulls its
        //     sample metres nearer → a large DEPTH GAP (palms-front);
        //   • collapse — the detector draws both shoulders onto the central
        //     hands → the 3-D SEPARATION shrinks far below a span (namaste).
        // In either case re-seat both shoulders onto the body (deeper) depth
        // along their own rays, so the torso reads frontal (dz≈0) instead of a
        // spurious ±60–90° yaw. Both are kept (the arms still root on them);
        // the front-facing prior is the safe default for an unseen orientation.
        let mut occluded = 0u8;
        if let (Some(r), Some(l)) = (r_shoulder_cam, l_shoulder_cam) {
            let d = [r[0] - l[0], r[1] - l[1], r[2] - l[2]];
            let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let depth_gap = (r[2] - l[2]).abs();
            let body_z = r[2].max(l[2]);
            if dist < MIN_RIGID_SPAN_FRAC * ANATOMICAL_SPAN_M {
                // Collapsed pair: the detector drew both shoulders onto the
                // central hands, so BOTH lateral positions are unreliable — a
                // depth-only re-seat would leave the x's crossed (±180° yaw).
                // Fabricate a canonical frontal pair about the shoulder mid
                // (avatar-Right at larger camera-x), which the front-facing
                // prior makes the safe default for an unobservable orientation.
                let m = mid(r, l);
                let half = 0.5 * ANATOMICAL_SPAN_M;
                r_shoulder_cam = Some([m[0] + half, m[1], body_z]);
                l_shoulder_cam = Some([m[0] - half, m[1], body_z]);
                occluded = 1;
            } else if depth_gap > OCCLUSION_GAP_M {
                // Foreground occlusion of ONE shoulder (a hand in front of it):
                // its depth is wrong but its 2-D ray is fine, so re-seat the
                // shallower one onto the body (deeper) depth along its own ray.
                r_shoulder_cam = Some(ray_to_depth(r, body_z));
                l_shoulder_cam = Some(ray_to_depth(l, body_z));
                occluded = 1;
            }
        }

        let hips_allowed = !opts.force_shoulder_anchor;
        let r_hip_cam = if hips_allowed { sample(11) } else { None };
        let l_hip_cam = if hips_allowed { sample(12) } else { None };

        let (anchor_cam, anchor_is_hip, anchor_score) =
            if let (Some(rh), Some(lh)) = (r_hip_cam, l_hip_cam) {
                (mid(rh, lh), true, joints[11].score.min(joints[12].score))
            } else if let (Some(rs), Some(ls)) = (r_shoulder_cam, l_shoulder_cam) {
                (mid(rs, ls), false, joints[5].score.min(joints[6].score))
            } else if let Some(s) = r_shoulder_cam.or(l_shoulder_cam) {
                (s, false, joints[5].score.max(joints[6].score))
            } else {
                return None;
            };

        // Env-gated diagnostic (VULVATAR_TORSO_DEBUG=1): one line per fit so a
        // replay can be correlated with the CSV yaw. Surfaces the sampled +
        // occlusion-clamped shoulder cam positions and the resulting local yaw.
        if std::env::var("VULVATAR_TORSO_DEBUG").is_ok() {
            use std::sync::atomic::{AtomicU64, Ordering};
            static CTR: AtomicU64 = AtomicU64::new(0);
            let n = CTR.fetch_add(1, Ordering::Relaxed);
            let fmt = |o: Option<[f32; 3]>| {
                o.map(|p| format!("[{:+.3},{:+.3},{:+.3}]", p[0], p[1], p[2]))
                    .unwrap_or_else(|| "—".into())
            };
            let yaw = match (r_shoulder_cam, l_shoulder_cam) {
                (Some(r), Some(l)) => (-(l[2] - r[2])).atan2(-(l[0] - r[0])).to_degrees(),
                _ => f32::NAN,
            };
            eprintln!(
                "TORSODBG#{n} occ={occluded} Rsh={} Lsh={} yaw={:+.1} anchor=[{:+.3},{:+.3},{:+.3}]{}",
                fmt(r_shoulder_cam),
                fmt(l_shoulder_cam),
                yaw,
                anchor_cam[0],
                anchor_cam[1],
                anchor_cam[2],
                if anchor_is_hip { " hip" } else { " sh" },
            );
        }

        Some(TorsoFit {
            r_shoulder_cam,
            l_shoulder_cam,
            r_hip_cam,
            l_hip_cam,
            anchor_cam,
            anchor_is_hip,
            anchor_score,
        })
    }
}

// ---------------------------------------------------------------------------
// Bone-length-constrained limb solve
// ---------------------------------------------------------------------------

/// Anthropometric segment lengths (metres) derived from the subject's
/// shoulder (biacromial) span. Upper-arm 0.75 and forearm 0.625 are the
/// ratios from the removed depth arm-z reconstruction; the leg/foot ratios
/// come from Winter's segment/stature fractions renormalised to span
/// (span ≈ 0.22·stature). See `docs/ray-ik-depth-solve.md`.
#[derive(Clone, Copy, Debug)]
pub(super) struct BoneLengths {
    pub upper_arm_m: f32,
    pub forearm_m: f32,
    pub thigh_m: f32,
    pub shin_m: f32,
    pub foot_m: f32,
}

/// Derive all bone lengths from the calibrated shoulder span. Reuses the
/// existing single-scalar `shoulder_span_m` calibration — no new capture UI.
pub(super) fn anthropometric_bones(shoulder_span_m: f32) -> BoneLengths {
    let s = shoulder_span_m;
    BoneLengths {
        upper_arm_m: 0.75 * s,
        forearm_m: 0.625 * s,
        thigh_m: 1.10 * s,
        shin_m: 1.05 * s,
        foot_m: 0.65 * s,
    }
}

/// Intersection of a pixel ray `P(t) = t·[a, b, 1]` with a bone-length
/// sphere. `t` is camera depth (the ray direction's z-component is 1).
#[derive(Clone, Copy, Debug, PartialEq)]
enum RaySphere {
    Two(f32, f32),
    Tangent(f32),
    /// The ray misses the sphere; `closest_t` is the ray param at closest
    /// approach — the "reach" case where 2D says the limb extends past its
    /// anatomical length (foreshortening lost or 2D error).
    Miss { closest_t: f32 },
}

/// Solve `|t·d − c|² = l²` for `d = [a, b, 1]`, sphere centre `c`, radius `l`.
fn ray_sphere_depths(a: f32, b: f32, c: [f32; 3], l: f32) -> RaySphere {
    let aa = a * a + b * b + 1.0;
    let bq = a * c[0] + b * c[1] + c[2]; // d·c
    let cq = c[0] * c[0] + c[1] * c[1] + c[2] * c[2] - l * l; // |c|² − l²
    let disc = bq * bq - aa * cq;
    if disc > 1e-9 {
        let root = disc.sqrt();
        RaySphere::Two((bq - root) / aa, (bq + root) / aa)
    } else if disc >= -1e-9 {
        RaySphere::Tangent(bq / aa)
    } else {
        RaySphere::Miss {
            closest_t: bq / aa,
        }
    }
}

/// Place a limb child joint on its observation ray at `bone_len_m` from the
/// already-solved `parent_cam`. **The bone length — not the depth sample —
/// sets the distance**, so a background/occluded depth reading at the child
/// pixel can no longer teleport the joint (a wrist over the far wall stays
/// pinned one forearm from the elbow). `depth_prior` (a valid local median
/// depth) is used *only* to disambiguate the two ray–sphere roots, and only
/// when it lands near one of them. Returns the child camera-space point, or
/// `None` for an off-frame keypoint or a grossly inconsistent 2D→3D reach.
pub(super) fn solve_limb_joint(
    intr: &CameraIntrinsics,
    nx: f32,
    ny: f32,
    parent_cam: [f32; 3],
    bone_len_m: f32,
    depth_prior: Option<f32>,
) -> Option<[f32; 3]> {
    if !nx.is_finite() || !ny.is_finite() || !(bone_len_m > 1e-4) {
        return None;
    }
    let u = nx * intr.width as f32;
    let v = ny * intr.height as f32;
    let a = (u - intr.cx) / intr.fx;
    let b = (v - intr.cy) / intr.fy;
    let point_at = |t: f32| [t * a, t * b, t];

    /// Reject a root closer than this to the lens (nonsensical for a limb).
    const MIN_T_M: f32 = 0.15;
    /// Max implied bone stretch in the ray-misses-sphere reach case.
    const STRETCH_MAX: f32 = 1.6;
    /// A depth prior only disambiguates if it lands within this of a root.
    const DEPTH_PRIOR_TOL_M: f32 = 0.30;

    let prior = depth_prior.filter(|z| z.is_finite() && *z > MIN_T_M);
    let pick = |roots: &[f32]| -> Option<f32> {
        let valid: Vec<f32> = roots.iter().copied().filter(|t| *t > MIN_T_M).collect();
        if valid.is_empty() {
            return None;
        }
        if let Some(z) = prior {
            let nearest = *valid
                .iter()
                .min_by(|x, y| (**x - z).abs().total_cmp(&(**y - z).abs()))
                .unwrap();
            if (nearest - z).abs() < DEPTH_PRIOR_TOL_M {
                return Some(nearest);
            }
        }
        // No usable prior → toward-camera (smaller positive root). The far
        // root would sink the joint behind the torso; limbs reach forward.
        Some(valid.iter().copied().fold(f32::INFINITY, f32::min))
    };

    let t = match ray_sphere_depths(a, b, parent_cam, bone_len_m) {
        RaySphere::Two(t0, t1) => pick(&[t0, t1])?,
        RaySphere::Tangent(t) => {
            if t > MIN_T_M {
                t
            } else {
                return None;
            }
        }
        RaySphere::Miss { closest_t } => {
            if closest_t <= MIN_T_M {
                return None;
            }
            let p = point_at(closest_t);
            let d = [
                p[0] - parent_cam[0],
                p[1] - parent_cam[1],
                p[2] - parent_cam[2],
            ];
            let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if dist > STRETCH_MAX * bone_len_m {
                // 2D is grossly inconsistent with the anatomical length —
                // trust the depth prior if present, otherwise drop the joint.
                match prior {
                    Some(z) => z,
                    None => return None,
                }
            } else {
                closest_t
            }
        }
    };
    Some(point_at(t))
}

/// Every left/right bone pair a whole-body 2D-detector transposition swaps
/// together — body chain + hand/finger chain. Used by
/// [`correct_upper_body_lr_swap`] to put the whole skeleton back.
const LR_SWAP_PAIRS: &[(HumanoidBone, HumanoidBone)] = &[
    (HumanoidBone::LeftShoulder, HumanoidBone::RightShoulder),
    (HumanoidBone::LeftUpperArm, HumanoidBone::RightUpperArm),
    (HumanoidBone::LeftLowerArm, HumanoidBone::RightLowerArm),
    (HumanoidBone::LeftUpperLeg, HumanoidBone::RightUpperLeg),
    (HumanoidBone::LeftLowerLeg, HumanoidBone::RightLowerLeg),
    (HumanoidBone::LeftFoot, HumanoidBone::RightFoot),
    (HumanoidBone::LeftHand, HumanoidBone::RightHand),
    (HumanoidBone::LeftThumbProximal, HumanoidBone::RightThumbProximal),
    (HumanoidBone::LeftThumbIntermediate, HumanoidBone::RightThumbIntermediate),
    (HumanoidBone::LeftThumbDistal, HumanoidBone::RightThumbDistal),
    (HumanoidBone::LeftIndexProximal, HumanoidBone::RightIndexProximal),
    (HumanoidBone::LeftIndexIntermediate, HumanoidBone::RightIndexIntermediate),
    (HumanoidBone::LeftIndexDistal, HumanoidBone::RightIndexDistal),
    (HumanoidBone::LeftMiddleProximal, HumanoidBone::RightMiddleProximal),
    (HumanoidBone::LeftMiddleIntermediate, HumanoidBone::RightMiddleIntermediate),
    (HumanoidBone::LeftMiddleDistal, HumanoidBone::RightMiddleDistal),
    (HumanoidBone::LeftRingProximal, HumanoidBone::RightRingProximal),
    (HumanoidBone::LeftRingIntermediate, HumanoidBone::RightRingIntermediate),
    (HumanoidBone::LeftRingDistal, HumanoidBone::RightRingDistal),
    (HumanoidBone::LeftLittleProximal, HumanoidBone::RightLittleProximal),
    (HumanoidBone::LeftLittleIntermediate, HumanoidBone::RightLittleIntermediate),
    (HumanoidBone::LeftLittleDistal, HumanoidBone::RightLittleDistal),
];

fn swap_source_pair(
    map: &mut std::collections::HashMap<HumanoidBone, SourceJoint>,
    l: HumanoidBone,
    r: HumanoidBone,
) {
    let lv = map.remove(&l);
    let rv = map.remove(&r);
    if let Some(v) = rv {
        map.insert(l, v);
    }
    if let Some(v) = lv {
        map.insert(r, v);
    }
}

/// Correct a whole-body left/right transposition from the 2D detector on the
/// metric skeleton. RTMW3D intermittently swaps its entire L/R keypoint
/// assignment under motion / symmetry — proven on the wave replay, where
/// adjacent frames flip between correct and x-reversed shoulders, which no
/// real turn can do. Distinct from [`super::rtmw3d::arm_z::correct_hand_lr_swap`]
/// (hands cross, shoulders correct): here the SHOULDERS themselves are
/// reversed, sending torso yaw to ±180°.
///
/// Source +x is the subject's anatomical left, so for a front-facing subject
/// the left joints sit at larger x. When the shoulders are x-reversed AND a
/// majority of confident L/R body pairs agree, the whole block was transposed
/// and we swap every L/R pair back. Front-facing prior: this is a desk-up
/// self-view app; a genuine >90° turn (which also reverses x-order) is out of
/// scope, and the metric z channel carries turns up to that point without it.
fn correct_upper_body_lr_swap(sk: &mut SourceSkeleton) {
    let g = |b: HumanoidBone| sk.joints.get(&b).map(|j| j.position);
    let (Some(sl), Some(sr)) = (
        g(HumanoidBone::LeftUpperArm),
        g(HumanoidBone::RightUpperArm),
    ) else {
        return;
    };
    // Shoulders must themselves be x-reversed for the swap to apply.
    if sl[0] - sr[0] >= 0.0 {
        return;
    }
    let span = {
        let d = [sl[0] - sr[0], sl[1] - sr[1], sl[2] - sr[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    };
    if span < 1e-3 {
        return;
    }
    let thresh = 0.15 * span;
    // Vote across confident body pairs (hands excluded — they legitimately
    // cross). Shoulders are already confirmed inverted above.
    let vote = [
        (HumanoidBone::LeftUpperArm, HumanoidBone::RightUpperArm),
        (HumanoidBone::LeftLowerArm, HumanoidBone::RightLowerArm),
        (HumanoidBone::LeftUpperLeg, HumanoidBone::RightUpperLeg),
    ];
    let (mut inverted, mut total) = (0i32, 0i32);
    for (l, r) in vote {
        if let (Some(lp), Some(rp)) = (g(l), g(r)) {
            let d = lp[0] - rp[0];
            if d.abs() > thresh {
                total += 1;
                if d < 0.0 {
                    inverted += 1;
                }
            }
        }
    }
    // With ≥2 confident pairs, require a majority inverted so a single noisy
    // shoulder frame can't flip the whole body. On a DEGRADED frame (arm across
    // the face → only the shoulders confidently present, <2 vote pairs) fall
    // back to the shoulders alone, but only when they are CLEARLY reversed
    // (past half a shoulder span): a front-facing self-view has no in-scope
    // >90° turn, so a clean x-reversal is an RTMW3D block transposition, not a
    // real turn — leaving it sends torso yaw to ±180° (the frame-200 spike).
    let clear_shoulder_reversal = (sr[0] - sl[0]) > 0.5 * span;
    if total >= 2 {
        if inverted * 2 <= total {
            return;
        }
    } else if !clear_shoulder_reversal {
        return;
    }
    for &(l, r) in LR_SWAP_PAIRS {
        swap_source_pair(&mut sk.joints, l, r);
        swap_source_pair(&mut sk.fingertips, l, r);
    }
    std::mem::swap(&mut sk.left_hand_orientation, &mut sk.right_hand_orientation);
}

/// Build a [`SourceSkeleton`] from 2D keypoints + a fitted torso surface
/// ([`torso_fit`]) via a **geometric model fit**, not per-keypoint depth
/// sampling. Torso joints ride the fitted plane; limb joints are placed by
/// bone-length-constrained ray–sphere intersection from their parent.
/// Camera-space metric `(x-right, y-down, z-forward)` is flipped on every
/// axis to match the source-space convention `(selfie-mirror x, y-up, z
/// toward camera)`. Selfie mirror: subject's anatomical-left landmarks drive
/// avatar `Right*` bones.
///
/// The wrist is re-pinned to the forearm chain (elbow + forearm length), not
/// the raw MCP-centroid depth, so a hand over the background stays one
/// forearm from the elbow instead of teleporting. Face pose / expressions are
/// left empty here — the caller populates them via the FaceMesh cascade.
pub(super) fn build_skeleton(
    frame_index: u64,
    joints: &[DecodedJoint2d],
    depth: &MetricDepthFrame,
    fit: torso_fit::TorsoFit,
    calibration: Option<&crate::tracking::PoseCalibration>,
) -> SourceSkeleton {
    let mut sk = SourceSkeleton::empty(frame_index);
    if joints.len() < NUM_JOINTS {
        return sk;
    }
    // The geometric fit forms observation rays from the intrinsics; the
    // metric (D435) path always carries them. Without them there is no model
    // to fit — emit an empty skeleton (matches the old no-anchor case).
    let intr = match depth.intrinsics {
        Some(i) => i,
        None => return sk,
    };

    let origin = fit.anchor_cam;
    let anchor_was_hip = fit.anchor_is_hip;
    let anchor_score = fit.anchor_score;

    // Camera metres → source frame (selfie-mirror x, y-up, z toward camera),
    // anchored at the torso origin. `&to_source` flows to the KEEP helpers.
    let to_source = |p: [f32; 3]| -> [f32; 3] {
        [-(p[0] - origin[0]), -(p[1] - origin[1]), -(p[2] - origin[2])]
    };
    // Insert a camera-space joint, surfacing its raw metric depth.
    let src_joint = |cam: [f32; 3], score: f32| -> SourceJoint {
        SourceJoint {
            position: [-(cam[0] - origin[0]), -(cam[1] - origin[1]), -(cam[2] - origin[2])],
            confidence: score,
            metric_depth_m: Some(cam[2]),
        }
    };

    // Bone lengths from the calibrated shoulder span (fallback: this frame's
    // measured 3D span, else an anatomical mean). This one scalar also drives
    // the isotropic normalization below as `reference_span_m`.
    let reference_span_m = calibration
        .and_then(|c| c.shoulder_span_m)
        .filter(|s| *s > 0.05)
        .or_else(|| {
            let r = fit.r_shoulder_cam?;
            let l = fit.l_shoulder_cam?;
            let d = ((l[0] - r[0]).powi(2) + (l[1] - r[1]).powi(2) + (l[2] - r[2]).powi(2)).sqrt();
            (d > 0.05).then_some(d)
        })
        .unwrap_or(0.38);
    let bones = anthropometric_bones(reference_span_m);

    // A local median depth prior only disambiguates the two ray–sphere roots.
    let depth_prior = |nx: f32, ny: f32| -> Option<f32> {
        sample_metric_point_with_radius(depth, nx, ny, SAMPLE_RADIUS_PX).map(|p| p[2])
    };
    // Solve a limb child on its observation ray at `bone_len` from `parent`.
    let solve_child = |parent: [f32; 3], coco_idx: usize, bone_len: f32| -> Option<([f32; 3], f32)> {
        let j = joints.get(coco_idx)?;
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            return None;
        }
        let z = depth_prior(j.nx, j.ny);
        let cam = solve_limb_joint(&intr, j.nx, j.ny, parent, bone_len, z)?;
        Some((cam, j.score))
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
            metric_depth_m: None,
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

    // --- Torso joints: on the fitted surface (COCO 5/6 shoulders, 11/12 hips) ---
    if let Some(c) = fit.r_shoulder_cam {
        sk.joints.insert(HumanoidBone::RightShoulder, src_joint(c, joints[5].score));
        sk.joints.insert(HumanoidBone::RightUpperArm, src_joint(c, joints[5].score));
    }
    if let Some(c) = fit.l_shoulder_cam {
        sk.joints.insert(HumanoidBone::LeftShoulder, src_joint(c, joints[6].score));
        sk.joints.insert(HumanoidBone::LeftUpperArm, src_joint(c, joints[6].score));
    }
    if anchor_was_hip {
        if let Some(c) = fit.r_hip_cam {
            sk.joints.insert(HumanoidBone::RightUpperLeg, src_joint(c, joints[11].score));
        }
        if let Some(c) = fit.l_hip_cam {
            sk.joints.insert(HumanoidBone::LeftUpperLeg, src_joint(c, joints[12].score));
        }
    }

    // --- Limb chains: bone-length-constrained ray–sphere from the parent ---
    // Arms: shoulder → elbow (upper arm). The wrist is solved inside
    // `attach_hand` from the elbow, so the hand block's MCP centroid drives
    // its direction; keep the elbow position to pass along.
    let r_elbow_cam = fit.r_shoulder_cam.and_then(|sh| {
        solve_child(sh, 7, bones.upper_arm_m).map(|(c, s)| {
            sk.joints.insert(HumanoidBone::RightLowerArm, src_joint(c, s));
            c
        })
    });
    let l_elbow_cam = fit.l_shoulder_cam.and_then(|sh| {
        solve_child(sh, 8, bones.upper_arm_m).map(|(c, s)| {
            sk.joints.insert(HumanoidBone::LeftLowerArm, src_joint(c, s));
            c
        })
    });

    // Legs: hip → knee (thigh) → ankle (shin) → toe-tip (foot). Hip-anchored
    // only; the desk-up shoulder-anchored path has no legs in frame.
    if anchor_was_hip {
        if let Some(hip) = fit.r_hip_cam {
            if let Some((knee, ks)) = solve_child(hip, 13, bones.thigh_m) {
                sk.joints.insert(HumanoidBone::RightLowerLeg, src_joint(knee, ks));
                if let Some((ankle, ascore)) = solve_child(knee, 15, bones.shin_m) {
                    sk.joints.insert(HumanoidBone::RightFoot, src_joint(ankle, ascore));
                    if let Some((toe, ts)) = solve_child(ankle, 17, bones.foot_m) {
                        sk.fingertips.insert(HumanoidBone::RightFoot, src_joint(toe, ts));
                    }
                }
            }
        }
        if let Some(hip) = fit.l_hip_cam {
            if let Some((knee, ks)) = solve_child(hip, 14, bones.thigh_m) {
                sk.joints.insert(HumanoidBone::LeftLowerLeg, src_joint(knee, ks));
                if let Some((ankle, ascore)) = solve_child(knee, 16, bones.shin_m) {
                    sk.joints.insert(HumanoidBone::LeftFoot, src_joint(ankle, ascore));
                    if let Some((toe, ts)) = solve_child(ankle, 20, bones.foot_m) {
                        sk.fingertips.insert(HumanoidBone::LeftFoot, src_joint(toe, ts));
                    }
                }
            }
        }
    }

    // Spine-chain proxies + Head, paralleling `rtmw3d::skeleton`. See
    // `inject_spine_chain_proxies` there for the rationale; on the
    // depth-aware path the Z component is metric (meters) instead of
    // RTMW3D-normalised, but the geometry the solver consumes is the
    // *direction* between joints, so the units cancel and the same
    // injection scheme works unchanged.
    inject_spine_chain_proxies(
        &mut sk,
        joints,
        depth,
        &to_source,
        anchor_was_hip,
        anchor_score,
    );

    attach_hand(
        &mut sk,
        joints,
        depth,
        &intr,
        91,
        HumanoidBone::RightHand,
        HAND_PHALANGES_RIGHT,
        HAND_TIPS_RIGHT,
        &to_source,
        r_elbow_cam,
        bones.forearm_m,
    );
    attach_hand(
        &mut sk,
        joints,
        depth,
        &intr,
        112,
        HumanoidBone::LeftHand,
        HAND_PHALANGES_LEFT,
        HAND_TIPS_LEFT,
        &to_source,
        l_elbow_cam,
        bones.forearm_m,
    );

    // Undo a whole-body left/right transposition from the 2D detector before
    // anything reads the L/R geometry. RTMW3D intermittently swaps its entire
    // L/R block under motion, x-reversing the shoulders and sending torso yaw
    // to ±180° (proven on the wave replay). Runs on raw metres; the isotropic
    // normalisation below is order-independent so it doesn't matter which side.
    correct_upper_body_lr_swap(&mut sk);

    // Normalise the whole metric skeleton (raw metres, camera-space) into the
    // source frame's numeric range by ONE isotropic scale, so the solver's
    // position-magnitude tunings keep holding. z stays REAL relative depth
    // (same scale as x/y) → the rotation path reads the true 3D shoulder/hip
    // line, no foreshortening reconstruction. `reference_span_m` was resolved
    // above (calibration → this frame's measured span → anatomical mean).
    // metric_depth_m and root_offset stay RAW metres.
    let mpsu = reference_span_m / TARGET_SRC_SHOULDER_SPAN;
    if mpsu > 1e-6 {
        let inv = 1.0 / mpsu;
        for j in sk.joints.values_mut() {
            j.position = [j.position[0] * inv, j.position[1] * inv, j.position[2] * inv];
        }
        for t in sk.fingertips.values_mut() {
            t.position = [t.position[0] * inv, t.position[1] * inv, t.position[2] * inv];
        }
    }

    // The single metric-native signal: its presence flips the solver + render
    // onto the faithful camera-space path. Carries the raw metric anchor (1:1
    // root placement), the normalization scale, and the intrinsics (mirror
    // render). `intr` is always present on the metric path (asserted above).
    sk.metric_frame_info = Some(MetricFrameInfo {
        anchor_cam_m: origin,
        anchor_is_hip: anchor_was_hip,
        mpsu,
        reference_span_m,
        intrinsics: intr,
    });

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
    depth: &MetricDepthFrame,
    intr: &CameraIntrinsics,
    base_index: usize,
    wrist_bone: HumanoidBone,
    phalanges: &[(usize, HumanoidBone)],
    tips: &[(usize, HumanoidBone)],
    to_source: &F,
    elbow_cam: Option<[f32; 3]>,
    forearm_len_m: f32,
) where
    F: Fn([f32; 3]) -> [f32; 3],
{
    const INDEX_MCP_LOCAL: usize = 5;
    const MIDDLE_MCP_LOCAL: usize = 9;
    const PINKY_MCP_LOCAL: usize = 17;
    const MCP_LOCALS: [usize; 4] = [INDEX_MCP_LOCAL, MIDDLE_MCP_LOCAL, 13, PINKY_MCP_LOCAL];
    /// A finger sample farther than this from the MCP centroid is a
    /// background/occlusion outlier (a human hand spans ~0.20 m). This is the
    /// wrist-relative replacement for the old anchor-relative z-band.
    const MAX_HAND_SPAN_M: f32 = 0.30;

    // Sample the MCP knuckles in CAMERA space — the ray–sphere wrist solve and
    // the centroid both need camera coords. No anchor-relative z gate here;
    // plausibility is wrist-relative (below).
    let sample_cam = |local: usize| -> Option<([f32; 3], f32)> {
        let j = joints.get(base_index + local)?;
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            return None;
        }
        let p = sample_metric_point(depth, j.nx, j.ny)?;
        Some((p, j.score))
    };

    let mut sum = [0.0_f32; 3];
    let mut sum_nx = 0.0_f32;
    let mut sum_ny = 0.0_f32;
    let mut min_conf = f32::INFINITY;
    let mut found = 0_u32;
    let mut index_mcp: Option<[f32; 3]> = None;
    let mut middle_mcp: Option<[f32; 3]> = None;
    let mut pinky_mcp: Option<[f32; 3]> = None;
    for &local in &MCP_LOCALS {
        if let Some((p, score)) = sample_cam(local) {
            sum[0] += p[0];
            sum[1] += p[1];
            sum[2] += p[2];
            let j = &joints[base_index + local];
            sum_nx += j.nx;
            sum_ny += j.ny;
            min_conf = min_conf.min(score);
            found += 1;
            match local {
                INDEX_MCP_LOCAL => index_mcp = Some(p),
                MIDDLE_MCP_LOCAL => middle_mcp = Some(p),
                PINKY_MCP_LOCAL => pinky_mcp = Some(p),
                _ => {}
            }
        }
    }
    if found < 3 {
        // No usable hand samples → the hand rests (no wrist/finger joints).
        return;
    }
    let inv = 1.0 / found as f32;
    let mcp_centroid_cam = [sum[0] * inv, sum[1] * inv, sum[2] * inv];
    let mcp_nx = sum_nx * inv;
    let mcp_ny = sum_ny * inv;

    // Re-pin the wrist to the forearm chain: place it at `forearm_len` from
    // the elbow along the MCP-centroid ray, disambiguated by the sampled
    // depth. A background depth can no longer drag the wrist off — the bone
    // length sets the distance. Falls back to the raw centroid with no elbow.
    let wrist_cam = match elbow_cam {
        Some(elbow) if forearm_len_m > 1e-4 => {
            solve_limb_joint(intr, mcp_nx, mcp_ny, elbow, forearm_len_m, Some(mcp_centroid_cam[2]))
                .unwrap_or(mcp_centroid_cam)
        }
        _ => mcp_centroid_cam,
    };

    // Sanity bound: a wrist implausibly far from the torso anchor is a
    // background/occlusion sample. This can only arise from the raw-centroid
    // fallback (no elbow chain) — a bone-pinned wrist is already bounded to
    // one forearm from the elbow. Rest the hand instead of letting it
    // teleport to the far wall (the -4 m outliers in the palms/namaste replay).
    let wrist_src = to_source(wrist_cam);
    let wrist_reach =
        (wrist_src[0] * wrist_src[0] + wrist_src[1] * wrist_src[1] + wrist_src[2] * wrist_src[2])
            .sqrt();
    if wrist_reach > 5.0 * forearm_len_m.max(0.1) {
        return;
    }

    // Rigid shift that carries the sampled hand onto the re-pinned wrist.
    let delta = [
        wrist_cam[0] - mcp_centroid_cam[0],
        wrist_cam[1] - mcp_centroid_cam[1],
        wrist_cam[2] - mcp_centroid_cam[2],
    ];

    sk.joints.insert(
        wrist_bone,
        SourceJoint {
            position: to_source(wrist_cam),
            confidence: min_conf,
            metric_depth_m: Some(wrist_cam[2]),
        },
    );

    // Palm orientation basis (KEEP). Computed in SOURCE space so the
    // handedness matches the solver (camera space would flip `forward`):
    //   raw_forward = middle_mcp − wrist ; across = index_mcp − pinky_mcp
    //   normal = across × raw_forward   ; forward = normal × across
    // Cross-orthogonalising forward against the normal keeps a noisy wrist Z
    // from tilting the in-palm forward. Directions are delta-invariant, so the
    // un-shifted samples are fine.
    if let (Some((wri_cam, _)), Some(mid), Some(idx), Some(pin)) =
        (sample_cam(0), middle_mcp, index_mcp, pinky_mcp)
    {
        let wri_p = to_source(wri_cam);
        let raw_forward = sub3(to_source(mid), wri_p);
        let across = sub3(to_source(idx), to_source(pin));
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

    // Fingers: wrist-relative plausibility bound (replaces the anchor z-band),
    // then rigid-shifted onto the re-pinned wrist so the hand stays coherent.
    let within_hand = |p: [f32; 3]| -> bool {
        let d = [
            p[0] - mcp_centroid_cam[0],
            p[1] - mcp_centroid_cam[1],
            p[2] - mcp_centroid_cam[2],
        ];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() <= MAX_HAND_SPAN_M
    };
    let shifted_src = |p: [f32; 3]| to_source([p[0] + delta[0], p[1] + delta[1], p[2] + delta[2]]);
    for &(local_idx, bone) in phalanges {
        if let Some((p, score)) = sample_cam(local_idx) {
            if !within_hand(p) {
                continue;
            }
            sk.joints.insert(
                bone,
                SourceJoint {
                    position: shifted_src(p),
                    confidence: score,
                    metric_depth_m: None,
                },
            );
        }
    }
    for &(local_idx, bone) in tips {
        if let Some((p, score)) = sample_cam(local_idx) {
            if !within_hand(p) {
                continue;
            }
            sk.fingertips.insert(
                bone,
                SourceJoint {
                    position: shifted_src(p),
                    confidence: score,
                    metric_depth_m: None,
                },
            );
        }
    }
}

/// Depth-aware twin of `rtmw3d::skeleton::inject_spine_chain_proxies`.
/// Same dual-insert scheme (Chest at hip-mid, UpperChest+Neck at
/// shoulder midpoint, Head at ear midpoint with nose fallback). Z is
/// metric (meters) instead of RTMW3D-normalised, but the solver only
/// reads bone *directions*, so units cancel.
fn inject_spine_chain_proxies<F>(
    sk: &mut SourceSkeleton,
    joints: &[DecodedJoint2d],
    depth: &MetricDepthFrame,
    to_source: &F,
    anchor_was_hip: bool,
    anchor_score: f32,
) where
    F: Fn([f32; 3]) -> [f32; 3],
{
    if anchor_was_hip {
        sk.joints.insert(
            HumanoidBone::Chest,
            SourceJoint {
                position: [0.0, 0.0, 0.0],
                confidence: anchor_score,
                metric_depth_m: None,
            },
        );
    }

    if let (Some(l), Some(r)) = (
        sk.joints.get(&HumanoidBone::LeftShoulder).copied(),
        sk.joints.get(&HumanoidBone::RightShoulder).copied(),
    ) {
        let mid = SourceJoint {
            position: [
                (l.position[0] + r.position[0]) * 0.5,
                (l.position[1] + r.position[1]) * 0.5,
                (l.position[2] + r.position[2]) * 0.5,
            ],
            confidence: l.confidence.min(r.confidence),
            metric_depth_m: None,
        };
        sk.joints.insert(HumanoidBone::UpperChest, mid);
        sk.joints.insert(HumanoidBone::Neck, mid);
    }

    const NOSE_IDX: usize = 0;
    const LEFT_EAR_IDX: usize = 3;
    const RIGHT_EAR_IDX: usize = 4;

    let sample_face = |idx: usize| -> Option<SourceJoint> {
        let j = joints.get(idx)?;
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            return None;
        }
        let p_cam = sample_metric_point(depth, j.nx, j.ny)?;
        Some(SourceJoint {
            position: to_source(p_cam),
            confidence: j.score,
            metric_depth_m: None,
        })
    };

    let head = match (sample_face(LEFT_EAR_IDX), sample_face(RIGHT_EAR_IDX)) {
        (Some(l), Some(r)) => Some(SourceJoint {
            position: [
                (l.position[0] + r.position[0]) * 0.5,
                (l.position[1] + r.position[1]) * 0.5,
                (l.position[2] + r.position[2]) * 0.5,
            ],
            confidence: l.confidence.min(r.confidence),
            metric_depth_m: None,
        }),
        _ => sample_face(NOSE_IDX),
    };

    if let Some(h) = head {
        sk.joints.insert(HumanoidBone::Head, h);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_frame(width: u32, height: u32, point: [f32; 3]) -> MetricDepthFrame {
        let pixels = (width * height) as usize;
        MetricDepthFrame {
            width,
            height,
            points_m: vec![point; pixels],
            crop: None,
            intrinsics: None,
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

    fn dj(nx: f32, ny: f32, score: f32) -> DecodedJoint2d {
        DecodedJoint2d { nx, ny, score }
    }

    // -----------------------------------------------------------------------
    // Torso surface fit
    // -----------------------------------------------------------------------

    fn intr_640() -> CameraIntrinsics {
        CameraIntrinsics {
            fx: 600.0,
            fy: 600.0,
            cx: 320.0,
            cy: 240.0,
            width: 640,
            height: 480,
        }
    }

    /// Deproject every pixel onto `plane` (point `p0`, unit normal `n`),
    /// giving a synthetic torso surface. Rays parallel to the plane → NaN.
    fn plane_frame(intr: CameraIntrinsics, p0: [f32; 3], n: [f32; 3]) -> MetricDepthFrame {
        let (w, h) = (intr.width, intr.height);
        let mut pts = vec![[f32::NAN; 3]; (w * h) as usize];
        let np0 = n[0] * p0[0] + n[1] * p0[1] + n[2] * p0[2];
        for v in 0..h {
            for u in 0..w {
                let d = [
                    (u as f32 - intr.cx) / intr.fx,
                    (v as f32 - intr.cy) / intr.fy,
                    1.0,
                ];
                let nd = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
                if nd.abs() < 1e-6 {
                    continue;
                }
                let t = np0 / nd;
                if t > 0.0 {
                    pts[(v * w + u) as usize] = [t * d[0], t * d[1], t * d[2]];
                }
            }
        }
        MetricDepthFrame {
            width: w,
            height: h,
            points_m: pts,
            crop: None,
            intrinsics: Some(intr),
        }
    }

    /// Overwrite a ±12 px block around a normalised keypoint with a surface at
    /// depth `z` (a foreground hand/occluder), for the occlusion-guard tests.
    fn stamp_block(frame: &mut MetricDepthFrame, intr: &CameraIntrinsics, nx: f32, ny: f32, z: f32) {
        let cx = (nx * intr.width as f32).round() as i32;
        let cy = (ny * intr.height as f32).round() as i32;
        for v in (cy - 12).max(0)..=(cy + 12).min(intr.height as i32 - 1) {
            for u in (cx - 12).max(0)..=(cx + 12).min(intr.width as i32 - 1) {
                let d = [(u as f32 - intr.cx) / intr.fx, (v as f32 - intr.cy) / intr.fy, 1.0];
                frame.points_m[(v as u32 * intr.width + u as u32) as usize] = [d[0] * z, d[1] * z, z];
            }
        }
    }

    #[test]
    fn fit_torso_frontal_shoulders_and_hips() {
        // A fronto-parallel synthetic surface at z=2.0: every keypoint's local
        // windowed-median depth sample lands on it. idx5 → avatar Right
        // (image-left, cam x<0), idx6 → avatar Left (image-right, cam x>0).
        let intr = intr_640();
        let frame = plane_frame(intr, [0.0, 0.0, 2.0], [0.0, 0.0, -1.0]);
        let mut joints = vec![dj(0.5, 0.5, 0.0); NUM_JOINTS];
        joints[5] = dj(0.42, 0.40, 0.9);
        joints[6] = dj(0.58, 0.40, 0.9);
        joints[11] = dj(0.45, 0.60, 0.9);
        joints[12] = dj(0.55, 0.60, 0.9);
        let opts = BuildOptions {
            force_shoulder_anchor: false,
        };
        let fit = torso_fit::fit_torso(&frame, &joints, opts).expect("torso fit");
        assert!(fit.anchor_is_hip, "hip anchor expected");
        for c in [
            fit.r_shoulder_cam,
            fit.l_shoulder_cam,
            fit.r_hip_cam,
            fit.l_hip_cam,
        ] {
            assert!((c.expect("torso joint")[2] - 2.0).abs() < 1e-3);
        }
        assert!(fit.r_shoulder_cam.unwrap()[0] < 0.0);
        assert!(fit.l_shoulder_cam.unwrap()[0] > 0.0);
    }

    /// A hand in front of ONE shoulder drags its depth sample metres nearer;
    /// the occlusion guard re-seats it onto the body (deeper) depth so the two
    /// shoulders share a depth (frontal) instead of yielding a spurious yaw.
    #[test]
    fn fit_torso_reseats_foreground_occluded_shoulder() {
        let intr = intr_640();
        let mut frame = plane_frame(intr, [0.0, 0.0, 2.0], [0.0, 0.0, -1.0]);
        // idx5 shoulder pixel is covered by a near-camera hand at z=1.2.
        stamp_block(&mut frame, &intr, 0.42, 0.40, 1.2);
        let mut joints = vec![dj(0.5, 0.5, 0.0); NUM_JOINTS];
        joints[5] = dj(0.42, 0.40, 0.9);
        joints[6] = dj(0.58, 0.40, 0.9);
        let opts = BuildOptions {
            force_shoulder_anchor: true,
        };
        let fit = torso_fit::fit_torso(&frame, &joints, opts).expect("torso fit");
        let r = fit.r_shoulder_cam.unwrap();
        let l = fit.l_shoulder_cam.unwrap();
        // Re-seated to the body depth → equal z (frontal), not 1.2 vs 2.0.
        assert!((r[2] - l[2]).abs() < 0.05, "r={:?} l={:?}", r, l);
        assert!(r[2] > 1.8, "occluded shoulder pulled to body depth: {:?}", r);
    }

    /// When the detector collapses both shoulders onto the central hands (3-D
    /// separation far below a span), the guard fabricates a canonical frontal
    /// pair (avatar-Right at larger camera-x) rather than trust crossed samples
    /// that would read ±180°.
    #[test]
    fn fit_torso_frontal_fallback_on_collapsed_pair() {
        let intr = intr_640();
        let frame = plane_frame(intr, [0.0, 0.0, 2.0], [0.0, 0.0, -1.0]);
        let mut joints = vec![dj(0.5, 0.5, 0.0); NUM_JOINTS];
        // Both shoulders within a few px of centre → collapsed 3-D separation.
        joints[5] = dj(0.49, 0.45, 0.9);
        joints[6] = dj(0.51, 0.45, 0.9);
        let opts = BuildOptions {
            force_shoulder_anchor: true,
        };
        let fit = torso_fit::fit_torso(&frame, &joints, opts).expect("torso fit");
        let r = fit.r_shoulder_cam.unwrap();
        let l = fit.l_shoulder_cam.unwrap();
        assert!((r[2] - l[2]).abs() < 1e-4, "equal depth expected");
        assert!(r[0] > l[0], "avatar-Right at larger camera-x for frontal");
        assert!((r[0] - l[0]).abs() > 0.25, "span ~0.36 expected, got {}", r[0] - l[0]);
    }

    // -----------------------------------------------------------------------
    // Bone-length-constrained limb solve
    // -----------------------------------------------------------------------

    fn project_px(intr: &CameraIntrinsics, p: [f32; 3]) -> (f32, f32) {
        (
            (intr.cx + p[0] / p[2] * intr.fx) / intr.width as f32,
            (intr.cy + p[1] / p[2] * intr.fy) / intr.height as f32,
        )
    }

    fn bone_len(a: [f32; 3], b: [f32; 3]) -> f32 {
        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }

    #[test]
    fn anthropometric_ratios() {
        let b = anthropometric_bones(0.4);
        assert!((b.upper_arm_m - 0.30).abs() < 1e-6);
        assert!((b.forearm_m - 0.25).abs() < 1e-6);
        assert!((b.thigh_m - 0.44).abs() < 1e-6);
        assert!((b.shin_m - 0.42).abs() < 1e-6);
        assert!((b.foot_m - 0.26).abs() < 1e-6);
    }

    #[test]
    fn limb_solve_recovers_child_toward_camera() {
        let intr = intr_640();
        let parent = [0.0, 0.0, 2.0];
        // Child reaches toward the camera (z < parent z) → the near root, so
        // both the depth-prior and the toward-camera default recover it.
        let child = [0.15, -0.05, 1.7];
        let bone = bone_len(parent, child);
        let (nx, ny) = project_px(&intr, child);

        for prior in [Some(child[2]), None] {
            let got = solve_limb_joint(&intr, nx, ny, parent, bone, prior).expect("solved");
            assert!(
                (got[0] - child[0]).abs() < 1e-3
                    && (got[1] - child[1]).abs() < 1e-3
                    && (got[2] - child[2]).abs() < 1e-3,
                "prior={:?} got={:?} child={:?}",
                prior,
                got,
                child
            );
        }
    }

    #[test]
    fn limb_solve_ignores_background_prior_stays_bone_pinned() {
        let intr = intr_640();
        let parent = [0.0, 0.0, 2.0];
        let child = [0.15, -0.05, 1.7];
        let bone = bone_len(parent, child);
        let (nx, ny) = project_px(&intr, child);
        // A background depth (3.8 m) is far from both roots → ignored; the
        // joint stays pinned at bone length (near root = the child).
        let got = solve_limb_joint(&intr, nx, ny, parent, bone, Some(3.8)).expect("solved");
        assert!((bone_len(parent, got) - bone).abs() < 1e-2, "dist={}", bone_len(parent, got));
        assert!((got[2] - child[2]).abs() < 1e-2, "got={:?}", got);
    }

    #[test]
    fn limb_solve_miss_drops_without_prior_uses_prior_when_present() {
        let intr = intr_640();
        let parent = [0.0, 0.0, 2.0];
        // Off-axis ray with a tiny bone → ray misses the sphere by far.
        let nx = 0.9;
        let ny = 0.5;
        let bone = 0.05;
        assert!(solve_limb_joint(&intr, nx, ny, parent, bone, None).is_none());
        let got = solve_limb_joint(&intr, nx, ny, parent, bone, Some(1.7)).expect("prior");
        assert!((got[2] - 1.7).abs() < 1e-3, "got={:?}", got);
    }
}
