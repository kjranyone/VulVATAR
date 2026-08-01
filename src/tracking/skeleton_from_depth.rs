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
    /// Device capture timestamp of the colour/depth frameset, in
    /// milliseconds on the sensor's clock (D435 hardware timestamp).
    /// Consecutive-frame differences drive the dt-normalised temporal
    /// estimators ([`TorsoScaleStabilizer`], the face-source crossfade,
    /// the arm-length leaky maxima) so their time constants hold under
    /// frame drops / non-30-fps streams. `None` for synthetic frames
    /// (tests, offline benches without recorded timestamps) — consumers
    /// fall back to a nominal 30 fps step.
    pub timestamp_ms: Option<f64>,
}

/// Derives the inter-frame dt (seconds) from consecutive device capture
/// timestamps, with a nominal-30-fps fallback when timestamps are absent
/// (synthetic frames, offline benches) or non-monotonic (device clock
/// reset after a reconnect). Clamped so a single pathological gap can't
/// blow up a rate gate or an EMA step.
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::tracking) struct FrameDtTracker {
    last_ts_ms: Option<f64>,
}

/// Nominal frame step used whenever a real capture dt is unavailable.
/// All dt-normalised constants reproduce their historical 30-fps
/// behaviour exactly at this step.
pub(in crate::tracking) const NOMINAL_FRAME_DT_S: f32 = 1.0 / 30.0;

impl FrameDtTracker {
    /// dt bounds: 5 ms (~200 fps — anything faster is duplicate/burst
    /// delivery, not subject motion) to 100 ms (~10 fps — beyond that a
    /// gap is a stall, and stretching rate gates further would let a
    /// contamination jump masquerade as plausible motion).
    const DT_MIN_S: f32 = 0.005;
    const DT_MAX_S: f32 = 0.100;

    pub(in crate::tracking) fn tick(&mut self, ts_ms: Option<f64>) -> f32 {
        let dt = match (self.last_ts_ms, ts_ms) {
            (Some(prev), Some(cur)) if cur > prev => ((cur - prev) / 1000.0) as f32,
            _ => NOMINAL_FRAME_DT_S,
        };
        if ts_ms.is_some() {
            self.last_ts_ms = ts_ms;
        }
        dt.clamp(Self::DT_MIN_S, Self::DT_MAX_S)
    }

    pub(in crate::tracking) fn reset(&mut self) {
        self.last_ts_ms = None;
    }
}

/// Per-keypoint visibility floor — anything below is treated as "no
/// detection" rather than a low-confidence detection. The user-tunable
/// `joint_confidence_threshold` slider lives in the solver.
pub const KEYPOINT_VISIBILITY_FLOOR: f32 = 0.05;

/// Sustained-visibility gate for the hand landmark blocks.
///
/// Desk framing — a programming streamer with head + shoulders in frame and
/// hands on the keyboard BELOW the frame — keeps the wrist/elbow keypoints
/// hovering around the bottom image border. A border-riding hand pops in and
/// out of the depth-observable region several times a second, and each pop
/// snapped the avatar's arm between rest and the observed position (live
/// trace: 17–23 hand jumps of 0.35–0.55 m world in 20 s while the subject
/// held still). The gate tracks each end effector's recent in-frame duty
/// (an EMA over ≈ [`Self::DUTY_TAU_S`] of *capture* time) and lets it drive
/// the skeleton only while that duty is high ([`Self::DUTY_ON`], with a
/// Schmitt band down to [`Self::DUTY_OFF`]): a genuine "raise hands into
/// frame" gesture engages in ≈ 0.3 s, imperceptible against the raise
/// itself; border peeks at a low duty never engage. Duty — not
/// consecutive dwell — because the 2D hand detector alternates between
/// the real hand and a below-frame phantom even on well-framed footage
/// (palms replay: in-2/out-1), and a phantom frame yields no depth sample
/// anyway, so the per-frame drop still protects the pose.
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::tracking) struct ArmEngageGate {
    /// Per hand landmark block (91-111, 112-132) — judged on the
    /// block's own MCP knuckles, the same joints `attach_hand`
    /// needs 3-of-4 of.
    ///
    /// Elbows are deliberately NOT gated: an off-frame elbow is already
    /// dropped per frame (no depth pixel), a briefly-visible elbow only
    /// refines the two-bone pole for that frame (no rest⇔observed
    /// position snap — the arm still aims at the wrist), and gating it
    /// measurably starved the elbow chain on the palms/namaste replays.
    hand: [EngageChannel; 2],
}

/// One end effector's engage state: an exponentially-weighted in-frame
/// duty ratio with a Schmitt trigger.
///
/// Consecutive-dwell hysteresis was tried first and failed on real
/// footage: even with the hands held statically at the chest (palms
/// replay) the 2D hand detector alternates between the real hand and a
/// below-frame phantom at duty cycles like in-2/out-1 — a "must be
/// continuously visible" gate never re-engages there, while the
/// hand was genuinely trackable two frames out of three. The
/// discriminating signal is the *visibility duty over a short window*:
/// real hand use keeps it well above one-half; a desk-hidden hand
/// peeking over the bottom border stays far below it.
#[derive(Clone, Copy, Debug, Default)]
struct EngageChannel {
    /// EMA of the in-frame indicator (0/1), time constant
    /// [`ArmEngageGate::DUTY_TAU_S`].
    duty: f32,
    engaged: bool,
}

impl EngageChannel {
    /// `in_frame` / `confident` are this frame's observability verdicts.
    fn advance(&mut self, in_frame: bool, confident: bool, dt_s: f32) {
        if in_frame && !confident {
            // In-frame detection dip: pause — bridging momentary noise
            // on a visibly-present hand is the solver hold policy's job.
            return;
        }
        let target = if in_frame { 1.0 } else { 0.0 };
        let alpha = 1.0 - (-dt_s / ArmEngageGate::DUTY_TAU_S).exp();
        self.duty += alpha * (target - self.duty);
        if self.duty >= ArmEngageGate::DUTY_ON {
            self.engaged = true;
        } else if self.duty <= ArmEngageGate::DUTY_OFF {
            self.engaged = false;
        }
        // Between the thresholds: keep the previous state (Schmitt).
    }
}

impl ArmEngageGate {
    /// Time constant of the visibility-duty EMA. A fully-visible hand
    /// engages from cold in ≈ 0.32 s (`-τ·ln(1-DUTY_ON)`); a hand
    /// dropped back to the keyboard releases in ≈ 0.34 s.
    const DUTY_TAU_S: f32 = 0.40;
    /// Engage when the recent in-frame duty exceeds this. Above the 50%
    /// a symmetric 1-in/1-out border flicker converges to, below the
    /// ~67% of the palms replay's worst real-hand phantom-flip zone.
    const DUTY_ON: f32 = 0.55;
    /// Release when the duty falls below this. The wide Schmitt band
    /// keeps a borderline hand from flapping the engagement.
    const DUTY_OFF: f32 = 0.30;
    /// A keypoint this close to any image border counts as out of frame:
    /// RTMW3D border-clamps some off-frame joints to nx/ny ≈ 0.998, and a
    /// depth window at the border is half background anyway.
    const BORDER_MARGIN: f32 = 0.01;
    /// Hand landmark block base index per side.
    const HAND_BASES: [usize; 2] = [91, 112];
    /// The four MCP knuckle locals `attach_hand` reads.
    const MCP_LOCALS: [usize; 4] = [5, 9, 13, 17];

    pub(in crate::tracking) fn reset(&mut self) {
        *self = Self::default();
    }

    /// Advance the engage channels by `dt_s` and return a copy of
    /// `joints` with every disengaged end effector's scores zeroed
    /// (below the visibility floor ⇒ all consumers treat it as
    /// undetected). Torso / leg / face keypoints pass through untouched.
    ///
    /// Loss-mode bookkeeping: LEAVING THE FRAME resets the engage dwell
    /// (the border-flicker this gate exists to suppress), an in-frame
    /// confidence dip merely pauses it — momentary detection noise on a
    /// hand plainly in view is the solver hold policy's job.
    pub(in crate::tracking) fn tick_and_gate(
        &mut self,
        joints: &[DecodedJoint2d],
        dt_s: f32,
    ) -> Vec<DecodedJoint2d> {
        let in_frame = |idx: usize| -> bool {
            joints.get(idx).is_some_and(|j| {
                let m = Self::BORDER_MARGIN;
                (m..=1.0 - m).contains(&j.nx) && (m..=1.0 - m).contains(&j.ny)
            })
        };
        let confident = |idx: usize| -> bool {
            joints
                .get(idx)
                .is_some_and(|j| j.score >= KEYPOINT_VISIBILITY_FLOOR)
        };
        let mut gated = joints.to_vec();
        for side in 0..2 {
            // Hand block, judged on the MCP knuckles `attach_hand` needs
            // 3-of-4 of: in frame while >= 3 knuckles are, confident
            // while >= 3 are above the floor.
            let base = Self::HAND_BASES[side];
            let count = |pred: &dyn Fn(usize) -> bool| {
                Self::MCP_LOCALS.iter().filter(|&&l| pred(base + l)).count()
            };
            self.hand[side].advance(count(&in_frame) >= 3, count(&confident) >= 3, dt_s);
            if std::env::var("VULVATAR_ENGAGE_DEBUG").is_ok() {
                let j = &joints[base + Self::MCP_LOCALS[1]];
                eprintln!(
                    "ENGAGE side={side} dt={dt_s:.3} hand={:?} mcp_if={} mcp_conf={} mid=({:.2},{:.2},{:.2})",
                    self.hand[side],
                    count(&in_frame),
                    count(&confident),
                    j.nx,
                    j.ny,
                    j.score,
                );
            }
            if !self.hand[side].engaged {
                for j in gated.iter_mut().skip(base).take(21) {
                    j.score = 0.0;
                }
            }
        }
        gated
    }
}

/// Is this whole-frame-normalised 2D keypoint inside the camera image? The
/// aligned depth frame only spans `[0, 1]²`; a keypoint the detector places
/// OUTSIDE it (RTMW3D extrapolates a limb that has left the frame — e.g. an
/// elbow below a head-and-shoulders crop, `ny > 1`) has NO depth pixel, so its
/// metric 3D cannot be observed and must not be fabricated. Callers use this to
/// deny the bone-length ray fallback for off-frame joints (an in-frame depth
/// HOLE is different — the joint is visible, its IR return was just absorbed).
pub(super) fn keypoint_in_frame(nx: f32, ny: f32) -> bool {
    (0.0..=1.0).contains(&nx) && (0.0..=1.0).contains(&ny)
}

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

/// How far toward the camera (metres) from the torso a keypoint may sit and
/// still count as "on the person" — a fully forward-reached hand.
const PERSON_NEAR_M: f32 = 0.75;
/// How far behind the torso (metres) a keypoint may sit — a leaned-back or
/// trailing limb. Anything deeper than this is background.
const PERSON_FAR_M: f32 = 0.45;

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
    sample_window(frame, nx, ny, radius_px, None)
}

/// **Person-aware** depth sample: like [`sample_metric_point_with_radius`] but
/// keeps only window samples whose depth lies on the person — within
/// `[z_ref - PERSON_NEAR_M, z_ref + PERSON_FAR_M]`, where `z_ref` is the
/// torso-anchor depth (metres). This is the root fix for the head/hand
/// teleport: a *correctly-placed* keypoint that sits at the body's occluding
/// silhouette (hair, headphone, hand edge) has a window full of the wall
/// metres behind it — the naive median then returns background and the joint
/// flies off. Rejecting the background at the sample boundary means a joint
/// whose depth is genuinely unobservable returns `None` (caller rests/holds)
/// instead of teleporting.
pub fn sample_metric_point_person(
    frame: &MetricDepthFrame,
    nx: f32,
    ny: f32,
    radius_px: i32,
    z_ref: f32,
) -> Option<[f32; 3]> {
    let band = z_ref
        .is_finite()
        .then(|| (z_ref - PERSON_NEAR_M, z_ref + PERSON_FAR_M));
    sample_window(frame, nx, ny, radius_px, band)
}

/// Shared window-median sampler. `z_band`, when set, restricts admitted
/// samples to `[lo, hi]` metres of depth (the person band) so background /
/// silhouette-edge pixels are discarded.
fn sample_window(
    frame: &MetricDepthFrame,
    nx: f32,
    ny: f32,
    radius_px: i32,
    z_band: Option<(f32, f32)>,
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
            if !(px.is_finite() && py.is_finite() && pz.is_finite() && pz > 0.0) {
                continue;
            }
            if let Some((lo, hi)) = z_band {
                if pz < lo || pz > hi {
                    continue;
                }
            }
            xs.push(px);
            ys.push(py);
            zs.push(pz);
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

        // Torso depth reference from the four anchor keypoints' own pixels —
        // the band the cells are gated against. Temporal median can absorb
        // RANDOM noise, but background pixels (armpit gaps, above-shoulder
        // corners of the rectangular bbox) sit at the same place every frame,
        // so without a band they'd survive the median and be baked into the
        // template as "torso".
        let mut anchor_depths: Vec<f32> = [ls, rs, lh, rh]
            .iter()
            .filter_map(|j| {
                let cx = (j.nx * frame.width as f32).round() as i32;
                let cy = (j.ny * frame.height as f32).round() as i32;
                if cx < 0 || cy < 0 || cx >= frame.width as i32 || cy >= frame.height as i32 {
                    return None;
                }
                let pz = frame.points_m[cy as usize * frame.width as usize + cx as usize][2];
                (pz.is_finite() && pz > 0.0).then_some(pz)
            })
            .collect();
        if anchor_depths.len() < 2 {
            return false;
        }
        anchor_depths.sort_by(|a, b| a.total_cmp(b));
        let z_ref = anchor_depths[anchor_depths.len() / 2];
        /// A torso surface spans well under this around its anchor depth; a
        /// cell pixel outside it is background / a foreground limb, not torso.
        const TORSO_BAND_M: f32 = 0.35;

        // Sample one pixel per cell at the cell's centre. Single-
        // pixel sample is fine here because the median over the
        // capture window absorbs per-frame depth-map noise.
        let grid = Self::GRID;
        let grid_f = grid as f32;
        let mut valid_cells = 0usize;
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
                if pz.is_finite() && pz > 0.0 && (pz - z_ref).abs() <= TORSO_BAND_M {
                    self.cells[gy * grid + gx].push(pz);
                    valid_cells += 1;
                }
            }
        }

        // Admit the frame only when a meaningful fraction of the grid saw the
        // torso — previously ONE valid cell out of 1024 admitted a frame, so
        // near-void frames still dragged the bbox average around.
        const MIN_VALID_CELL_FRAC: f32 = 0.30;
        if (valid_cells as f32) < MIN_VALID_CELL_FRAC * (grid * grid) as f32 {
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
        /// The shoulder pair was FABRICATED as a canonical frontal pair
        /// (collapse guard): its geometry is a constant, not a measurement.
        /// Consumers must not feed its span into any scale estimator, and
        /// the built joints are marked [`crate::tracking::source_skeleton::JointOrigin::Synthesized`].
        pub pair_fabricated: bool,
        /// The shoulder pair was RE-SEATED along its rays onto the body depth
        /// (foreground-occlusion guard). The 2-D rays are real but the depth —
        /// and hence the 3-D span — is inferred, so span estimators skip it.
        pub pair_reseated: bool,
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
    ///
    /// `person_z_ref` is the PREVIOUS frame's stabilised anchor depth
    /// (metres). When present, the shoulder/hip samples themselves are
    /// person-band gated — closing the hole where a shoulder keypoint on the
    /// silhouette sampled the wall 4 m back, the occlusion guard then declared
    /// the wall "the body" (deeper = body) and re-seated BOTH shoulders onto
    /// it, teleporting the whole torso. The first frame (`None`) seeds
    /// unbanded, exactly like the head/hand z-band bootstraps.
    pub(in crate::tracking) fn fit_torso(
        frame: &MetricDepthFrame,
        joints: &[DecodedJoint2d],
        opts: BuildOptions,
        person_z_ref: Option<f32>,
    ) -> Option<TorsoFit> {
        if joints.len() < NUM_JOINTS {
            return None;
        }
        let _ = frame.intrinsics?; // the metric path always carries intrinsics

        // Direct windowed-median depth sample at a torso keypoint, deprojected
        // to camera-space metres. Robust to a few edge pixels; no plane needed.
        // Person-band gated against the previous stabilised anchor when
        // available, so a silhouette-edge torso keypoint over the far wall
        // returns `None` (unobservable) instead of the wall.
        let z_ref = person_z_ref.filter(|z| z.is_finite() && *z > 0.0);
        let sample = |idx: usize| -> Option<[f32; 3]> {
            let j = visible(joints, idx)?;
            match z_ref {
                Some(z) => super::sample_metric_point_person(
                    frame,
                    j.nx,
                    j.ny,
                    super::SAMPLE_RADIUS_PX,
                    z,
                ),
                None => super::sample_metric_point(frame, j.nx, j.ny),
            }
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
        let mut pair_fabricated = false;
        let mut pair_reseated = false;
        if let (Some(r), Some(l)) = (r_shoulder_cam, l_shoulder_cam) {
            let d = [r[0] - l[0], r[1] - l[1], r[2] - l[2]];
            let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let depth_gap = (r[2] - l[2]).abs();
            // Which of the two depths is "the body"? With a person reference
            // available, the one CLOSER to it — "deeper = body" silently
            // declared the wall behind the subject the body whenever a
            // background pixel slipped through. Without a reference (first
            // frame) keep the deeper-is-body prior: the common contaminant
            // there is a hand in FRONT of a shoulder.
            let body_z = match z_ref {
                Some(z) => {
                    if (r[2] - z).abs() <= (l[2] - z).abs() {
                        r[2]
                    } else {
                        l[2]
                    }
                }
                None => r[2].max(l[2]),
            };
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
                pair_fabricated = true;
            } else if depth_gap > OCCLUSION_GAP_M {
                // Occlusion of ONE shoulder (a hand in front of it, or — with
                // a person reference — a background pixel past it): its depth
                // is wrong but its 2-D ray is fine, so re-seat both onto the
                // body depth along their own rays.
                r_shoulder_cam = Some(ray_to_depth(r, body_z));
                l_shoulder_cam = Some(ray_to_depth(l, body_z));
                pair_reseated = true;
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
            } else if let Some(s) = r_shoulder_cam {
                // Score of the side that actually produced the sample — not
                // simply the larger of the two.
                (s, false, joints[5].score)
            } else if let Some(s) = l_shoulder_cam {
                (s, false, joints[6].score)
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
            let occluded = u8::from(pair_fabricated) + 2 * u8::from(pair_reseated);
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
            pair_fabricated,
            pair_reseated,
        })
    }
}

// ---------------------------------------------------------------------------
// Global scale + root stabiliser (zoom-glitch fix)
// ---------------------------------------------------------------------------

/// Temporal stabiliser for the avatar's GLOBAL scale and root distance.
///
/// The subject's shoulder span and torso anchor are near-constant physically,
/// but a hand crossing in front of a shoulder contaminates that shoulder's
/// depth sample — swinging the raw per-frame span (measured 0.26–0.42 m across
/// palms/namaste) and the anchor depth (frame-to-frame jumps up to 0.4 m).
/// Because `reference_span_m` scales the whole avatar
/// (`avatar_scale = rest_span / reference_span_m`, and `mpsu` normalises every
/// joint by it) and the anchor sets the avatar's distance, that raw jitter
/// makes the character zoom in/out and lurch toward/away — the reported glitch.
/// Both quantities change slowly in reality, so hold them with an EMA: the span
/// with outlier rejection (a contaminated frame reads far off and is ignored),
/// the anchor with an adaptive rate (follow a genuine lean, hold sampling
/// jitter). Provider-owned; reset per session.
/// NOTE ON TIME UNITS: every temporal constant below is expressed in
/// seconds (EMA time constants, rejection windows) or per-second rates
/// (the anchor spike gate) and integrated against the real capture dt
/// supplied by the caller ([`FrameDtTracker`]). At the nominal 30 fps
/// step each reproduces the historically calibrated per-frame behaviour
/// exactly (the old per-frame value is quoted in each doc comment).
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::tracking) struct TorsoScaleStabilizer {
    span_m: Option<f32>,
    anchor_cam: Option<[f32; 3]>,
    /// Accumulated wall time (s) of consecutive span measurements rejected
    /// as outliers. At [`Self::SPAN_RESEED_S`] the estimator re-seeds
    /// instead of rejecting forever — the escape hatch for a bad first seed
    /// (finger-merge frame) or a subject swap, both of which previously
    /// dead-locked the scale: every subsequent true reading differed from
    /// the bogus held span by more than the outlier fraction and was
    /// discarded for the whole session.
    span_reject_time_s: f32,
    /// Accumulated wall time (s) of consecutive anchor measurements
    /// rejected as spikes.
    anchor_reject_time_s: f32,
    /// Ring buffer of the most recent REJECTED raw anchors, used as the
    /// independent evidence for re-seeding: only when the rejected readings
    /// are mutually consistent (the subject really is somewhere else, not
    /// noise) does the anchor jump there.
    anchor_recent: [[f32; 3]; Self::ANCHOR_RING],
    anchor_recent_len: usize,
    anchor_recent_head: usize,
}

impl TorsoScaleStabilizer {
    /// Plausible biacromial span band (m) used to reject a contaminated seed.
    const SPAN_SEED_LO: f32 = 0.28;
    const SPAN_SEED_HI: f32 = 0.48;
    /// Anatomical fallback span used when the first reading is implausible.
    const SPAN_DEFAULT: f32 = 0.38;
    /// A raw span this far (fraction) from the held value is contamination.
    /// Tight, because a real biacromial span barely changes frame to frame.
    const SPAN_OUTLIER_FRAC: f32 = 0.25;
    /// EMA time constant (s) for easing toward an accepted span
    /// (α = 1 − exp(−dt/τ); 0.05/frame at 30 fps). Long: the span is a
    /// physical constant, so once seeded it should hold near-locked,
    /// letting the avatar scale stay put instead of breathing with
    /// shoulder-depth noise.
    const SPAN_TAU_S: f32 = 0.650;
    /// An anchor moving faster than this (m/s) is physically impossible for
    /// a torso → it is shoulder-depth contamination (a hand crossing in
    /// front), NOT a real move, so REJECT it (hold). (0.12 m/frame at
    /// 30 fps.) A real lean / walk-in is well under this (~1 m/s) and passes
    /// through almost untouched, because the subject's distance (root_offset,
    /// hence the avatar's near/far) MUST reflect real movement — the downstream
    /// 1€ ROOT filter, not this gate, owns jitter smoothing. The old gate had
    /// this inverted (followed big jumps, held small ones) and so lagged genuine
    /// approach/retreat by ~25% while partly chasing contamination spikes.
    const ANCHOR_SPIKE_M_PER_S: f32 = 3.6;
    /// Follow time constant (s) for plausible movement: near pass-through,
    /// ~0 lag (α = 1 − exp(−dt/τ); 0.80/frame at 30 fps).
    const ANCHOR_FOLLOW_TAU_S: f32 = 0.0207;
    /// Sustained span-rejection time (s) before the estimator re-seeds.
    /// Long enough that a held namaste (whose fabricated span is
    /// already filtered upstream) or a burst of contamination can't re-seed,
    /// short enough that a bad seed / subject swap recovers within seconds.
    const SPAN_RESEED_S: f32 = 1.5;
    /// Sustained anchor-rejection time (s) before re-seed is CONSIDERED.
    /// The predecessor design instead crept 15% toward every rejected
    /// reading, which meant ~20 frames of a held pose (namaste, a leaning
    /// cheek-on-hand) walked the anchor fully onto the contamination — and the
    /// person z-band with it. Rejection now HOLDS; convergence to a genuinely
    /// changed position goes through the consistency check below.
    const ANCHOR_RESEED_S: f32 = 2.0;
    /// Rejected-anchor ring size used for the consistency check.
    const ANCHOR_RING: usize = 8;
    /// Re-seed only if every recent rejected reading sits within this of
    /// their mean — i.e. the "new place" is stable, not flicker.
    const ANCHOR_RESEED_SPREAD_M: f32 = 0.10;

    pub(in crate::tracking) fn reset(&mut self) {
        *self = Self::default();
    }

    /// Depth (camera z, metres) of the held anchor — the person-band
    /// reference for the NEXT frame's torso sampling. `None` until seeded.
    pub(in crate::tracking) fn anchor_z(&self) -> Option<f32> {
        self.anchor_cam.map(|a| a[2])
    }

    /// Fold a raw measured shoulder span into the stable estimate and return
    /// it. `raw = None` (shoulder pair missing or fabricated this frame)
    /// returns the HELD value — previously a single dropped shoulder popped
    /// the scale to the hard-coded anatomical default and back. Seeds within
    /// the anatomical band (a wildly off first frame seeds the default
    /// instead), eases toward plausible readings, ignores gross outliers so a
    /// hand-contaminated frame can't shrink/grow the avatar — and re-seeds
    /// after [`Self::SPAN_RESEED_S`] of sustained rejection, so a bogus
    /// seed or a subject swap can't dead-lock the scale forever.
    ///
    /// `dt_s` is the capture-timestamp frame step ([`FrameDtTracker`]) —
    /// the EMA step and the rejection window integrate real time, so a
    /// 15 fps stall or a 60 fps stream keeps the same wall-clock
    /// behaviour as the calibrated 30 fps baseline.
    pub(in crate::tracking) fn stable_span(&mut self, raw: Option<f32>, dt_s: f32) -> Option<f32> {
        let raw = match raw {
            Some(r) if r.is_finite() && r > 0.0 => r,
            _ => return self.span_m,
        };
        let next = match self.span_m {
            None => {
                self.span_reject_time_s = 0.0;
                if (Self::SPAN_SEED_LO..=Self::SPAN_SEED_HI).contains(&raw) {
                    raw
                } else {
                    Self::SPAN_DEFAULT
                }
            }
            Some(cur) => {
                if (raw - cur).abs() <= Self::SPAN_OUTLIER_FRAC * cur {
                    self.span_reject_time_s = 0.0;
                    let alpha = 1.0 - (-dt_s / Self::SPAN_TAU_S).exp();
                    cur + alpha * (raw - cur)
                } else {
                    self.span_reject_time_s += dt_s;
                    if self.span_reject_time_s >= Self::SPAN_RESEED_S {
                        // The "outliers" have outlasted anything a transient
                        // contamination produces — the held value is what's
                        // wrong. Re-seed like a first frame.
                        self.span_reject_time_s = 0.0;
                        if (Self::SPAN_SEED_LO..=Self::SPAN_SEED_HI).contains(&raw) {
                            raw
                        } else {
                            Self::SPAN_DEFAULT
                        }
                    } else {
                        cur
                    }
                }
            }
        };
        self.span_m = Some(next);
        Some(next)
    }

    /// Spike-reject the torso anchor (camera metres): a real lean / walk-in
    /// passes through nearly untouched so the avatar's distance tracks it,
    /// while a physically-impossible one-frame jump (shoulder depth
    /// contaminated by a hand in front) is rejected — and, unlike the old
    /// creep, a SUSTAINED contamination (a pose held for seconds) keeps being
    /// rejected instead of slowly capitulating. Recovery from a genuine
    /// discontinuity (subject swap, tracking re-acquired elsewhere) requires
    /// independent evidence: [`Self::ANCHOR_RESEED_S`] of rejections
    /// whose readings agree with each other, at which point the anchor
    /// re-seeds to their mean. This is a contamination gate, NOT a jitter
    /// smoother — the downstream 1€ ROOT filter owns jitter.
    ///
    /// `dt_s` is the capture-timestamp frame step ([`FrameDtTracker`]): the
    /// spike gate compares a *velocity* (dev/dt vs
    /// [`Self::ANCHOR_SPIKE_M_PER_S`]), so a dropped frame doesn't turn a
    /// legitimate 0.5 m/s approach into a "physically impossible" jump, and
    /// the follow easing + rejection window integrate real time.
    pub(in crate::tracking) fn stable_anchor(&mut self, raw: [f32; 3], dt_s: f32) -> [f32; 3] {
        let next = match self.anchor_cam {
            None => raw,
            Some(cur) => {
                let d = [raw[0] - cur[0], raw[1] - cur[1], raw[2] - cur[2]];
                let dev = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if dev <= Self::ANCHOR_SPIKE_M_PER_S * dt_s {
                    self.anchor_reject_time_s = 0.0;
                    self.anchor_recent_len = 0;
                    self.anchor_recent_head = 0;
                    let alpha = 1.0 - (-dt_s / Self::ANCHOR_FOLLOW_TAU_S).exp();
                    [
                        cur[0] + alpha * d[0],
                        cur[1] + alpha * d[1],
                        cur[2] + alpha * d[2],
                    ]
                } else {
                    self.anchor_reject_time_s += dt_s;
                    self.anchor_recent[self.anchor_recent_head] = raw;
                    self.anchor_recent_head = (self.anchor_recent_head + 1) % Self::ANCHOR_RING;
                    self.anchor_recent_len = (self.anchor_recent_len + 1).min(Self::ANCHOR_RING);
                    if self.anchor_reject_time_s >= Self::ANCHOR_RESEED_S
                        && self.anchor_recent_len == Self::ANCHOR_RING
                        && Self::ring_is_consistent(
                            &self.anchor_recent,
                            Self::ANCHOR_RESEED_SPREAD_M,
                        )
                    {
                        self.anchor_reject_time_s = 0.0;
                        self.anchor_recent_len = 0;
                        self.anchor_recent_head = 0;
                        Self::ring_mean(&self.anchor_recent)
                    } else {
                        cur
                    }
                }
            }
        };
        self.anchor_cam = Some(next);
        next
    }

    fn ring_mean(ring: &[[f32; 3]; Self::ANCHOR_RING]) -> [f32; 3] {
        let mut m = [0.0f32; 3];
        for p in ring {
            m[0] += p[0];
            m[1] += p[1];
            m[2] += p[2];
        }
        let inv = 1.0 / Self::ANCHOR_RING as f32;
        [m[0] * inv, m[1] * inv, m[2] * inv]
    }

    fn ring_is_consistent(ring: &[[f32; 3]; Self::ANCHOR_RING], tol: f32) -> bool {
        let mean = Self::ring_mean(ring);
        ring.iter().all(|p| {
            let d = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() <= tol
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

fn swap_source_pair<V>(
    map: &mut std::collections::HashMap<HumanoidBone, V>,
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

/// Temporal hysteresis for [`correct_upper_body_lr_swap`]. RTMW3D's block
/// transposition genuinely flips per frame (the wave replay shows adjacent
/// frames alternating), so the *correction* must stay per-frame — the latch
/// only owns the WEAK-EVIDENCE zone, where the instantaneous geometry can't
/// tell (shoulder reversal below the clear margin, vote tie): there the
/// previous frame's decision repeats instead of the whole body flapping
/// left/right at frame rate around the threshold. Provider-owned, reset per
/// session.
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::tracking) struct LrSwapLatch {
    last_swapped: bool,
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
fn correct_upper_body_lr_swap(sk: &mut SourceSkeleton, latch: &mut LrSwapLatch) {
    /// Shoulders clearly NOT reversed past this fraction of a span → strong
    /// "no swap" regardless of the latch.
    const AMBIG_LO_FRAC: f32 = -0.15;
    /// Shoulders reversed past half a span → strong "swap" (vetoable by a
    /// clear vote majority the other way). Same value as the old
    /// `clear_shoulder_reversal` gate.
    const CLEAR_FRAC: f32 = 0.5;
    /// Inside the ambiguous band, a vote majority only decides when the
    /// reversal itself is also past this margin in the agreeing direction;
    /// otherwise the previous frame's decision holds (hysteresis).
    const VOTE_MARGIN_FRAC: f32 = 0.15;

    let g = |b: HumanoidBone| sk.joints.get(&b).map(|j| j.position);
    let (Some(sl), Some(sr)) = (
        g(HumanoidBone::LeftUpperArm),
        g(HumanoidBone::RightUpperArm),
    ) else {
        return;
    };
    let span = {
        let d = [sl[0] - sr[0], sl[1] - sr[1], sl[2] - sr[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    };
    if span < 1e-3 {
        return;
    }
    // Positive = shoulders x-reversed (front-facing left should sit at
    // larger source x).
    let rev_frac = (sr[0] - sl[0]) / span;
    let thresh = 0.15 * span;
    // Vote across confident body pairs (hands excluded — they legitimately
    // cross).
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
    let vote_majority = (total >= 2).then(|| inverted * 2 > total);
    // Three-zone decision with hysteresis. The transposition genuinely flips
    // per frame, so STRONG evidence still decides instantly in both
    // directions; only the ambiguous middle repeats the previous decision —
    // previously a rev_frac oscillating around a hard threshold swapped and
    // un-swapped the ENTIRE body (16 bone pairs + hand orientations) at frame
    // rate.
    let decision = if rev_frac < AMBIG_LO_FRAC {
        false
    } else if rev_frac > CLEAR_FRAC {
        // Clear x-reversal. A confident vote majority the other way (the limbs
        // say "not transposed") still vetoes, matching the old majority gate.
        vote_majority.unwrap_or(true)
    } else {
        match vote_majority {
            Some(true) if rev_frac > VOTE_MARGIN_FRAC => true,
            Some(false) if rev_frac < VOTE_MARGIN_FRAC => false,
            _ => latch.last_swapped,
        }
    };
    latch.last_swapped = decision;
    if !decision {
        return;
    }
    for &(l, r) in LR_SWAP_PAIRS {
        swap_source_pair(&mut sk.joints, l, r);
        swap_source_pair(&mut sk.fingertips, l, r);
        swap_source_pair(&mut sk.joint_origins, l, r);
    }
    std::mem::swap(&mut sk.left_hand_orientation, &mut sk.right_hand_orientation);
}

/// Build a [`SourceSkeleton`] from 2D keypoints + a fitted torso surface
/// ([`torso_fit`]) via a **geometric model fit**, not per-keypoint depth
/// sampling. Torso joints ride the fitted plane; limb joints are placed by
/// their own directly-observed metric depth sample (person-aware, so a
/// silhouette/occluder pixel is rejected at the source rather than read as the
/// joint); a bone-length ray–sphere from the parent is only a fallback for a
/// joint whose depth is a hole. Camera-space metric `(x-right, y-down,
/// z-forward)` is flipped on every axis to match the source-space convention
/// `(selfie-mirror x, y-up, z toward camera)`. Selfie mirror: subject's
/// anatomical-left landmarks drive avatar `Right*` bones.
///
/// The wrist/hand is the observed wrist keypoint's metric 3D (knuckle-centroid
/// fallback), NOT a forearm-length re-pin — the sensor's measured forward reach
/// is the pose, so an extended hand keeps its depth instead of floating to face
/// height. Face pose / expressions are left empty here — the caller populates
/// them via the FaceMesh cascade.
pub(super) fn build_skeleton(
    frame_index: u64,
    joints: &[DecodedJoint2d],
    depth: &MetricDepthFrame,
    fit: torso_fit::TorsoFit,
    calibration: Option<&crate::tracking::PoseCalibration>,
    reference_span_override: Option<f32>,
    swap_latch: &mut LrSwapLatch,
) -> SourceSkeleton {
    use crate::tracking::source_skeleton::JointOrigin;
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
        // Provider's temporally-stabilised span (uncalibrated path): holds the
        // avatar's scale steady against per-frame shoulder-depth contamination.
        .or_else(|| reference_span_override.filter(|s| *s > 0.05))
        .or_else(|| {
            let r = fit.r_shoulder_cam?;
            let l = fit.l_shoulder_cam?;
            let d = ((l[0] - r[0]).powi(2) + (l[1] - r[1]).powi(2) + (l[2] - r[2]).powi(2)).sqrt();
            (d > 0.05).then_some(d)
        })
        .unwrap_or(0.38);
    let bones = anthropometric_bones(reference_span_m);

    // Torso-anchor depth = the person-depth reference every keypoint sample is
    // gated against (shoulders/hips sample cleanly, so this is reliable).
    let person_z_ref = origin[2];
    // Place a limb child at its DIRECTLY OBSERVED metric 3D — the sensor's own
    // person-aware depth sample at the keypoint IS the pose. Sampling already
    // rejects background/occluder depth, so the point is on-body; using it keeps
    // a forward-reached limb's TRUE depth instead of discarding it for an
    // anthropometric bone length (the old ray-sphere pin threw the measured
    // reach away, floating extended arms to the wrong height). Bone-length
    // ray-sphere (`solve_limb_joint`) is a FALLBACK only — when the child depth
    // is missing (a hole / off-frame) the joint is pinned one bone from the
    // parent along its 2D ray instead of dropped.
    // Returns `(cam, score, origin)` — `origin` is `Extrapolated` when the
    // position came from the bone-length ray fallback rather than an actual
    // depth sample, so statistical consumers can tell measurement from
    // inference.
    let solve_child = |parent: [f32; 3],
                       coco_idx: usize,
                       bone_len: f32|
     -> Option<([f32; 3], f32, JointOrigin)> {
        let j = joints.get(coco_idx)?;
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            return None;
        }
        let (cam, origin) = match sample_metric_point_person(
            depth,
            j.nx,
            j.ny,
            SAMPLE_RADIUS_PX,
            person_z_ref,
        ) {
            Some(observed) => (observed, JointOrigin::Observed),
            // Depth miss. The bone-length ray solve is a fallback for an
            // IN-FRAME hole (dark hair / clothing eats the IR) — it pins the
            // joint one bone from its parent along the observed 2D ray. For an
            // OFF-FRAME keypoint there is no ray to trust: the joint has left
            // the image and `solve_limb_joint` would fabricate a false
            // toward-camera depth, planting the elbow FORWARD of the hand and
            // folding the forearm (the "hands spread in front of the chest but
            // the avatar breaks" case — the elbow is below a head+shoulders
            // crop at ny>1). Drop it; the arm solver then bends a natural elbow
            // toward the still-visible wrist instead.
            None if keypoint_in_frame(j.nx, j.ny) => (
                solve_limb_joint(&intr, j.nx, j.ny, parent, bone_len, None)?,
                JointOrigin::Extrapolated,
            ),
            None => return None,
        };
        Some((cam, j.score, origin))
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
    // Provenance: a guard-processed shoulder pair is not a measurement —
    // fabricated canonical pairs are `Synthesized`, ray re-seated ones
    // `Extrapolated` — so scale/anchor estimators can refuse to ingest them.
    let shoulder_origin = if fit.pair_fabricated {
        JointOrigin::Synthesized
    } else if fit.pair_reseated {
        JointOrigin::Extrapolated
    } else {
        JointOrigin::Observed
    };
    if let Some(c) = fit.r_shoulder_cam {
        sk.joints.insert(HumanoidBone::RightShoulder, src_joint(c, joints[5].score));
        sk.joints.insert(HumanoidBone::RightUpperArm, src_joint(c, joints[5].score));
        sk.mark_origin(HumanoidBone::RightShoulder, shoulder_origin);
        sk.mark_origin(HumanoidBone::RightUpperArm, shoulder_origin);
    }
    if let Some(c) = fit.l_shoulder_cam {
        sk.joints.insert(HumanoidBone::LeftShoulder, src_joint(c, joints[6].score));
        sk.joints.insert(HumanoidBone::LeftUpperArm, src_joint(c, joints[6].score));
        sk.mark_origin(HumanoidBone::LeftShoulder, shoulder_origin);
        sk.mark_origin(HumanoidBone::LeftUpperArm, shoulder_origin);
    }
    if anchor_was_hip {
        if let Some(c) = fit.r_hip_cam {
            sk.joints.insert(HumanoidBone::RightUpperLeg, src_joint(c, joints[11].score));
        }
        if let Some(c) = fit.l_hip_cam {
            sk.joints.insert(HumanoidBone::LeftUpperLeg, src_joint(c, joints[12].score));
        }
    }

    // --- Limb chains: DIRECT observed depth, bone-length only as fallback ---
    // Arms: shoulder → elbow (upper arm). `solve_child` returns the elbow's own
    // observed metric 3D (bone-length ray-sphere only when its depth is a hole);
    // the wrist is likewise observed inside `attach_hand`, so no elbow position
    // is threaded between them any more.
    // Elbow camera positions are kept for `attach_hand`: when the hand's
    // depth is fully holed, the wrist is ray-solved a forearm length from
    // its elbow (same fallback family as `solve_child` itself).
    let mut r_elbow_cam = None;
    let mut l_elbow_cam = None;
    if let Some((c, s, o)) = fit.r_shoulder_cam.and_then(|sh| solve_child(sh, 7, bones.upper_arm_m)) {
        sk.joints.insert(HumanoidBone::RightLowerArm, src_joint(c, s));
        sk.mark_origin(HumanoidBone::RightLowerArm, o);
        r_elbow_cam = Some(c);
    }
    if let Some((c, s, o)) = fit.l_shoulder_cam.and_then(|sh| solve_child(sh, 8, bones.upper_arm_m)) {
        sk.joints.insert(HumanoidBone::LeftLowerArm, src_joint(c, s));
        sk.mark_origin(HumanoidBone::LeftLowerArm, o);
        l_elbow_cam = Some(c);
    }

    // Legs: hip → knee (thigh) → ankle (shin) → toe-tip (foot). Hip-anchored
    // only; the desk-up shoulder-anchored path has no legs in frame.
    if anchor_was_hip {
        if let Some(hip) = fit.r_hip_cam {
            if let Some((knee, ks, ko)) = solve_child(hip, 13, bones.thigh_m) {
                sk.joints.insert(HumanoidBone::RightLowerLeg, src_joint(knee, ks));
                sk.mark_origin(HumanoidBone::RightLowerLeg, ko);
                if let Some((ankle, ascore, ao)) = solve_child(knee, 15, bones.shin_m) {
                    sk.joints.insert(HumanoidBone::RightFoot, src_joint(ankle, ascore));
                    sk.mark_origin(HumanoidBone::RightFoot, ao);
                    if let Some((toe, ts, _)) = solve_child(ankle, 17, bones.foot_m) {
                        sk.fingertips.insert(HumanoidBone::RightFoot, src_joint(toe, ts));
                    }
                }
            }
        }
        if let Some(hip) = fit.l_hip_cam {
            if let Some((knee, ks, ko)) = solve_child(hip, 14, bones.thigh_m) {
                sk.joints.insert(HumanoidBone::LeftLowerLeg, src_joint(knee, ks));
                sk.mark_origin(HumanoidBone::LeftLowerLeg, ko);
                if let Some((ankle, ascore, ao)) = solve_child(knee, 16, bones.shin_m) {
                    sk.joints.insert(HumanoidBone::LeftFoot, src_joint(ankle, ascore));
                    sk.mark_origin(HumanoidBone::LeftFoot, ao);
                    if let Some((toe, ts, _)) = solve_child(ankle, 20, bones.foot_m) {
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
        person_z_ref,
    );

    attach_hand(
        &mut sk,
        joints,
        depth,
        91,
        HumanoidBone::RightHand,
        HAND_PHALANGES_RIGHT,
        HAND_TIPS_RIGHT,
        &to_source,
        bones.forearm_m,
        person_z_ref,
        r_elbow_cam,
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
        bones.forearm_m,
        person_z_ref,
        l_elbow_cam,
    );

    // Undo a whole-body left/right transposition from the 2D detector before
    // anything reads the L/R geometry. RTMW3D intermittently swaps its entire
    // L/R block under motion, x-reversing the shoulders and sending torso yaw
    // to ±180° (proven on the wave replay). Runs on raw metres; the isotropic
    // normalisation below is order-independent so it doesn't matter which side.
    correct_upper_body_lr_swap(&mut sk, swap_latch);

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
    base_index: usize,
    wrist_bone: HumanoidBone,
    phalanges: &[(usize, HumanoidBone)],
    tips: &[(usize, HumanoidBone)],
    to_source: &F,
    forearm_len_m: f32,
    person_z_ref: f32,
    elbow_cam: Option<[f32; 3]>,
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
    // the centroid both need camera coords. Person-aware: a knuckle keypoint at
    // the hand's silhouette against a far wall must not sample the wall.
    let sample_cam = |local: usize| -> Option<([f32; 3], f32)> {
        let j = joints.get(base_index + local)?;
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            return None;
        }
        let p = sample_metric_point_person(depth, j.nx, j.ny, SAMPLE_RADIUS_PX, person_z_ref)?;
        Some((p, j.score))
    };

    // Replay-bisect probe (`VULVATAR_DIAG_HAND`): classify per frame WHY a
    // hand fails to attach — 2D score below floor, keypoint off-frame, or
    // person-band depth sample miss — so low hand duty can be attributed to
    // the detector vs the depth sampler without guessing.
    let diag = std::env::var("VULVATAR_DIAG_HAND").is_ok();
    let mut n_score = 0_u32; // MCPs with a usable 2D score
    let mut n_depth = 0_u32; // ... that also produced a person-band sample
    let mut n_raw = 0_u32; // ... that produced ANY depth sample (no band)
    let mut n_wide = 0_u32; // ... person-band sample with a 21x21 window
    let mut mean_ny = 0.0_f32;
    let mut z_probe: Option<f32> = None; // unbanded z of the middle MCP
    if diag {
        for &local in &MCP_LOCALS {
            if let Some(j) = joints.get(base_index + local) {
                if j.score >= KEYPOINT_VISIBILITY_FLOOR {
                    n_score += 1;
                    mean_ny += j.ny;
                    if sample_metric_point_person(
                        depth,
                        j.nx,
                        j.ny,
                        SAMPLE_RADIUS_PX,
                        person_z_ref,
                    )
                    .is_some()
                    {
                        n_depth += 1;
                    }
                    if let Some(p) =
                        sample_metric_point_with_radius(depth, j.nx, j.ny, SAMPLE_RADIUS_PX)
                    {
                        n_raw += 1;
                        if local == MIDDLE_MCP_LOCAL {
                            z_probe = Some(p[2]);
                        }
                    }
                    if sample_metric_point_person(depth, j.nx, j.ny, 10, person_z_ref).is_some() {
                        n_wide += 1;
                    }
                }
            }
        }
        if n_score > 0 {
            mean_ny /= n_score as f32;
        }
    }

    let mut sum = [0.0_f32; 3];
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
    if diag {
        let raw_score = joints
            .get(base_index + MIDDLE_MCP_LOCAL)
            .map(|j| j.score)
            .unwrap_or(-1.0);
        eprintln!(
            "HANDDIAG side={:?} ns={n_score} nd={n_depth} nr={n_raw} nw={n_wide} zmid={:.3} zref={person_z_ref:.3} found={found} ny={mean_ny:.3} msc={raw_score:.3}",
            wrist_bone,
            z_probe.unwrap_or(f32::NAN)
        );
    }
    // 2D admissibility. Depth is NOT required to attach the hand any more —
    // the 2026-07-28 replay bisect measured the hand's 2D at 4/4 MCP scores
    // on every frame of the palms/wave replays while the depth under those
    // same knuckles was a hole in half of them (hands near the D435's
    // near limit / in its projector shadow). Requiring depth made the hand
    // BLINK at frame rate, and every downstream arm artefact was that blink
    // amplified. The elbow chain has had a ray fallback all along
    // (`solve_limb_joint`); the hand gets the same degradation path.
    //
    // What IS required for the depth-free paths: at least 3 confident MCP
    // keypoints (fewer means the detector — or the engage gate — says the
    // hand is unobservable), the wrist keypoint itself, and every one of
    // them safely INSIDE the frame. RTMW3D border-clamps an off-frame hand
    // to the image edge; a real depth sample proves such a pixel is on the
    // person, but without depth an edge-riding keypoint must not be pinned
    // in 3D (that would fabricate a hand for the desk-typing envelope).
    const FALLBACK_BORDER_MARGIN: f32 = 0.01;
    let inside = |j: &DecodedJoint2d| {
        j.nx >= FALLBACK_BORDER_MARGIN
            && j.nx <= 1.0 - FALLBACK_BORDER_MARGIN
            && j.ny >= FALLBACK_BORDER_MARGIN
            && j.ny <= 1.0 - FALLBACK_BORDER_MARGIN
    };
    let scored = |local: usize| {
        joints
            .get(base_index + local)
            .filter(|j| j.score >= KEYPOINT_VISIBILITY_FLOOR)
    };
    let mcp_kps: Vec<&DecodedJoint2d> =
        MCP_LOCALS.iter().filter_map(|&l| scored(l)).collect();
    let wrist_kp = scored(0);
    let conf_2d = mcp_kps.iter().map(|j| j.score).fold(f32::INFINITY, f32::min);
    let fallback_ok = mcp_kps.len() >= 3
        && mcp_kps.iter().all(|j| inside(j))
        && wrist_kp.is_some_and(inside);

    let mcp_centroid_cam = if found > 0 {
        let inv = 1.0 / found as f32;
        [sum[0] * inv, sum[1] * inv, sum[2] * inv]
    } else {
        [0.0; 3]
    };

    // Deproject a full-frame-normalised keypoint at an assumed depth — the
    // hand-plane reconstruction for keypoints whose own depth is a hole.
    let deproject = |nx: f32, ny: f32, z: f32| -> Option<[f32; 3]> {
        let intr = depth.intrinsics.as_ref()?;
        Some([
            (nx * intr.width as f32 - intr.cx) / intr.fx * z,
            (ny * intr.height as f32 - intr.cy) / intr.fy * z,
            z,
        ])
    };

    // The wrist's DIRECTLY OBSERVED metric 3D IS the hand position when the
    // depth cooperates: its own person-aware sample, else the knuckle
    // centroid. Person-aware sampling already keeps these on-body, so the
    // old forearm-length re-pin — which discarded the measured forward reach
    // and floated an extended hand up to face height — stays gone.
    // Sampled once and reused by the palm-orientation basis below.
    use crate::tracking::source_skeleton::JointOrigin;
    let wrist_sample = sample_cam(0);
    let (wrist_cam, wrist_origin) = if let Some((p, _)) = wrist_sample {
        (p, JointOrigin::Observed)
    } else if found >= 3 {
        (mcp_centroid_cam, JointOrigin::Observed)
    } else if fallback_ok && found >= 1 {
        // Partial knuckle depth: the sampled knuckle(s) fix the hand's depth
        // plane, the wrist keypoint fixes its ray.
        let w = wrist_kp.expect("fallback_ok requires the wrist keypoint");
        match deproject(w.nx, w.ny, mcp_centroid_cam[2]) {
            Some(p) => (p, JointOrigin::Extrapolated),
            None => (mcp_centroid_cam, JointOrigin::Extrapolated),
        }
    } else if fallback_ok {
        // Depth fully holed: pin the forearm length along the observed wrist
        // ray from the elbow — the limb chain's own hole fallback. No depth
        // prior: the toward-camera root matches how a reaching arm leaves
        // the near limit in the first place.
        let w = wrist_kp.expect("fallback_ok requires the wrist keypoint");
        match (depth.intrinsics.as_ref(), elbow_cam) {
            (Some(intr), Some(elbow)) => {
                match solve_limb_joint(intr, w.nx, w.ny, elbow, forearm_len_m, None) {
                    Some(p) => (p, JointOrigin::Extrapolated),
                    None => {
                        if diag {
                            eprintln!("HANDDIAG side={wrist_bone:?} RAY_FAIL");
                        }
                        return;
                    }
                }
            }
            _ => {
                if diag {
                    eprintln!("HANDDIAG side={wrist_bone:?} NO_ELBOW");
                }
                return;
            }
        }
    } else {
        // Too few 2D keypoints, or an edge-riding hand with no depth proof.
        if diag {
            eprintln!(
                "HANDDIAG side={wrist_bone:?} NOT_ADMISSIBLE nkp={} inside={} wrist_kp={} wrist_inside={}",
                mcp_kps.len(),
                mcp_kps.iter().all(|j| inside(j)),
                wrist_kp.is_some(),
                wrist_kp.is_some_and(inside),
            );
        }
        return;
    };
    let min_conf = if min_conf.is_finite() { min_conf } else { conf_2d };
    if !min_conf.is_finite() {
        return;
    }

    // Sanity bound: an observed wrist implausibly far from the torso anchor is a
    // stray sample the person-aware gate let through. Rest the hand instead of
    // letting it fly out (the -4 m outliers in the palms/namaste replay).
    let wrist_src = to_source(wrist_cam);
    let wrist_reach =
        (wrist_src[0] * wrist_src[0] + wrist_src[1] * wrist_src[1] + wrist_src[2] * wrist_src[2])
            .sqrt();
    if wrist_reach > 5.0 * forearm_len_m.max(0.1) {
        if diag {
            eprintln!(
                "HANDDIAG side={:?} REACH_REJECT reach={wrist_reach:.3} limit={:.3}",
                wrist_bone,
                5.0 * forearm_len_m.max(0.1)
            );
        }
        return;
    }

    sk.joints.insert(
        wrist_bone,
        SourceJoint {
            position: to_source(wrist_cam),
            confidence: min_conf,
            metric_depth_m: Some(wrist_cam[2]),
        },
    );
    sk.mark_origin(wrist_bone, wrist_origin);

    // Resolve a hand keypoint to camera space: its own person-band depth
    // sample when it exists, else (fallback-admissible frames only) the 2D
    // ray dropped onto the hand's depth plane. A hand spans ~10 cm in z, so
    // the plane approximation bounds the error well below the 0.30 m
    // outlier clamp while keeping palm orientation and finger curl alive
    // through depth holes.
    let hand_plane_z = wrist_cam[2];
    let resolve = |local: usize, sampled: Option<[f32; 3]>| -> Option<[f32; 3]> {
        if sampled.is_some() {
            return sampled;
        }
        if !fallback_ok {
            return None;
        }
        let j = scored(local).filter(|j| inside(j))?;
        deproject(j.nx, j.ny, hand_plane_z)
    };

    // Palm orientation basis (KEEP). Computed in SOURCE space so the
    // handedness matches the solver (camera space would flip `forward`):
    //   raw_forward = middle_mcp − wrist ; across = index_mcp − pinky_mcp
    //   normal = across × raw_forward   ; forward = normal × across
    // Cross-orthogonalising forward against the normal keeps a noisy wrist Z
    // from tilting the in-palm forward. The wrist end of the basis is the
    // resolved `wrist_cam` (not the raw sample), so orientation survives
    // the same depth holes the position now does.
    if let (Some(mid), Some(idx), Some(pin)) = (
        resolve(MIDDLE_MCP_LOCAL, middle_mcp),
        resolve(INDEX_MCP_LOCAL, index_mcp),
        resolve(PINKY_MCP_LOCAL, pinky_mcp),
    ) {
        let wri_p = to_source(wrist_cam);
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

    // Fingers: wrist-relative plausibility bound (replaces the anchor z-band).
    // Wrist and fingers are observed in the same camera frame, so they are
    // already coherent — the old rigid shift onto a re-pinned wrist is gone
    // along with the re-pin itself.
    // Outlier clamp reference: the sampled-knuckle centroid when any depth
    // exists, else the (ray-solved) wrist — plane-resolved fingers are near
    // it by construction, real samples still get vetted against it.
    let hand_ref = if found > 0 { mcp_centroid_cam } else { wrist_cam };
    let within_hand = |p: [f32; 3]| -> bool {
        let d = [p[0] - hand_ref[0], p[1] - hand_ref[1], p[2] - hand_ref[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() <= MAX_HAND_SPAN_M
    };
    // Finger position + its 2D score: sampled depth first, hand plane else.
    let finger_cam = |local: usize| -> Option<([f32; 3], f32)> {
        let sampled = sample_cam(local);
        let score = sampled
            .map(|(_, s)| s)
            .or_else(|| scored(local).map(|j| j.score))?;
        resolve(local, sampled.map(|(p, _)| p)).map(|p| (p, score))
    };
    for &(local_idx, bone) in phalanges {
        if let Some((p, score)) = finger_cam(local_idx) {
            if !within_hand(p) {
                continue;
            }
            sk.joints.insert(
                bone,
                SourceJoint {
                    position: to_source(p),
                    confidence: score,
                    metric_depth_m: None,
                },
            );
        }
    }
    for &(local_idx, bone) in tips {
        if let Some((p, score)) = finger_cam(local_idx) {
            if !within_hand(p) {
                continue;
            }
            sk.fingertips.insert(
                bone,
                SourceJoint {
                    position: to_source(p),
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
    person_z_ref: f32,
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

    // Head position from the ear/nose depth samples. The 2-D keypoints are
    // located correctly, but an ear at the head's occluding silhouette (against
    // a bright far wall) or on IR-absorbing hair / a headphone cup has a
    // sampling window full of the background — which the naive median returned
    // as the head depth, flinging the head metres away (measured: a correct
    // left-ear keypoint whose 7x7 window was 40/40 background at ~4.8 m). The
    // person-aware sampler discards those samples at the source, so a genuinely
    // unobservable ear returns `None` and we fall back to the visible ear / the
    // nose / the anatomical rest above the neck instead of teleporting.
    let neck_pos = sk.joints.get(&HumanoidBone::Neck).map(|j| j.position);

    // Camera-space face sample, so the one-ear lateral correction below can
    // work in metres before the source-frame flip.
    let sample_face_cam = |idx: usize| -> Option<([f32; 3], f32)> {
        let j = joints.get(idx)?;
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            return None;
        }
        let p_cam = sample_metric_point_person(depth, j.nx, j.ny, SAMPLE_RADIUS_PX, person_z_ref)?;
        Some((p_cam, j.score))
    };
    let src_head = |cam: [f32; 3], conf: f32| SourceJoint {
        position: to_source(cam),
        confidence: conf,
        metric_depth_m: None,
    };

    use crate::tracking::source_skeleton::JointOrigin;
    let (head, head_origin) = match (
        sample_face_cam(LEFT_EAR_IDX),
        sample_face_cam(RIGHT_EAR_IDX),
    ) {
        (Some((l, ls)), Some((r, rs))) => (
            Some(src_head(
                [
                    (l[0] + r[0]) * 0.5,
                    (l[1] + r[1]) * 0.5,
                    (l[2] + r[2]) * 0.5,
                ],
                ls.min(rs),
            )),
            JointOrigin::Observed,
        ),
        // One ear occluded (common at desk-up): drive off the visible one,
        // shifted half an ear-to-ear width toward the head centre. Without
        // the shift the head jumps ~8 cm sideways every time the far ear
        // flickers across the visibility floor (both-ears midpoint ⇄ raw
        // near-ear). The lateral direction comes from the 2-D nose keypoint
        // (always the centre-ward side of an ear); when the nose is absent
        // too, the uncorrected ear is still better than no head.
        (Some((ear, s)), None) => (
            Some(src_head(one_ear_head(ear, LEFT_EAR_IDX, NOSE_IDX, joints), s)),
            JointOrigin::Extrapolated,
        ),
        (None, Some((ear, s))) => (
            Some(src_head(one_ear_head(ear, RIGHT_EAR_IDX, NOSE_IDX, joints), s)),
            JointOrigin::Extrapolated,
        ),
        (None, None) => (
            sample_face_cam(NOSE_IDX).map(|(p, s)| src_head(p, s)),
            JointOrigin::Observed,
        ),
    };

    // No trustworthy face depth this frame → anatomical rest above the neck,
    // so the head holds a sane upright pose instead of teleporting. The
    // position is pure fabrication (a constant offset), so it carries a
    // DECAYED confidence and a `Synthesized` mark — previously it inherited
    // the neck's full confidence and downstream smoothing blended real⇄
    // fabricated heads at full weight, bobbing the head on every flicker.
    let (head, head_origin) = match head {
        Some(h) => (Some(h), head_origin),
        None => (
            neck_pos.map(|n| SourceJoint {
                position: [n[0], n[1] + HEAD_ABOVE_NECK_M, n[2]],
                confidence: sk
                    .joints
                    .get(&HumanoidBone::Neck)
                    .map(|j| j.confidence * FABRICATED_HEAD_CONF_SCALE)
                    .unwrap_or(0.0),
                metric_depth_m: None,
            }),
            JointOrigin::Synthesized,
        ),
    };

    if let Some(h) = head {
        sk.joints.insert(HumanoidBone::Head, h);
        sk.mark_origin(HumanoidBone::Head, head_origin);
    }
}

/// Half the ear-to-ear width (m): the lateral correction applied when only
/// one ear has a trustworthy depth, so the head estimate stays at the head
/// CENTRE instead of snapping to the near ear.
const EAR_HALF_WIDTH_M: f32 = 0.075;

/// Shift a single visible ear's camera-space sample half an ear-to-ear width
/// toward the head centre. The lateral direction comes from the 2-D nose
/// keypoint (always centre-ward of an ear); with no confident nose the raw
/// ear is returned unshifted.
fn one_ear_head(
    ear_cam: [f32; 3],
    ear_idx: usize,
    nose_idx: usize,
    joints: &[DecodedJoint2d],
) -> [f32; 3] {
    match (joints.get(ear_idx), joints.get(nose_idx)) {
        (Some(e), Some(n)) if n.score >= KEYPOINT_VISIBILITY_FLOOR => {
            let toward_centre = (n.nx - e.nx).signum();
            [
                ear_cam[0] + toward_centre * EAR_HALF_WIDTH_M,
                ear_cam[1],
                ear_cam[2],
            ]
        }
        _ => ear_cam,
    }
}

/// Confidence multiplier for the anatomical (neck + offset) fallback head —
/// fabricated geometry must not carry measurement-grade confidence.
const FABRICATED_HEAD_CONF_SCALE: f32 = 0.5;

/// Anatomical offset (metres, pre-normalisation) of the ear-midpoint "head"
/// keypoint above the neck (shoulder midpoint) — the fallback head height
/// when no face keypoint has a trustworthy depth this frame.
const HEAD_ABOVE_NECK_M: f32 = 0.16;

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

    /// Nominal 30 fps step — the baseline all historical calibrations
    /// were measured at.
    const DT: f32 = NOMINAL_FRAME_DT_S;
    /// Frame counts equivalent to the reseed horizons at the 30 fps step.
    const SPAN_RESEED_FRAMES_AT_30: u32 = 45;
    const ANCHOR_RESEED_FRAMES_AT_30: u32 = 60;

    /// Full keypoint set with confident in-frame shoulders and one hand
    /// block's MCP knuckles at `(hand_nx, hand_ny)`, scores 0.6 (the
    /// score RTMW3D gives extrapolated joints).
    fn engage_joints(hand_nx: f32, hand_ny: f32) -> Vec<DecodedJoint2d> {
        let mut joints = vec![DecodedJoint2d::default(); NUM_JOINTS];
        let mk = |nx: f32, ny: f32| DecodedJoint2d { nx, ny, score: 0.6 };
        joints[5] = mk(0.40, 0.45);
        joints[6] = mk(0.60, 0.45);
        joints[7] = mk(0.45, 0.85); // elbow: NOT gated, must always pass
        for &local in &ArmEngageGate::MCP_LOCALS {
            joints[91 + local] = mk(hand_nx, hand_ny);
        }
        joints
    }

    #[test]
    fn engage_gate_border_flicker_never_engages() {
        // A hand popping across the bottom border every other frame (the
        // desk-framing live trace) sits at ~50% duty — below DUTY_ON:
        // its hand block stays gated on every frame.
        let mut g = ArmEngageGate::default();
        let inside = engage_joints(0.5, 0.95);
        let outside = engage_joints(0.5, 1.15);
        for i in 0..60 {
            let src = if i % 2 == 0 { &inside } else { &outside };
            let gated = g.tick_and_gate(src, DT);
            assert_eq!(gated[91 + 9].score, 0.0, "flickering hand block must stay gated (frame {i})");
            // Torso keypoints and the elbow always pass.
            assert_eq!(gated[5].score, 0.6);
            assert_eq!(gated[6].score, 0.6);
            assert_eq!(gated[7].score, 0.6, "elbows are not gated");
        }
    }

    #[test]
    fn engage_gate_sustained_visibility_engages_and_lingers_through_phantom_flips() {
        let mut g = ArmEngageGate::default();
        let inside = engage_joints(0.5, 0.80);
        // First frame in view: still gated (duty barely off zero).
        let first = g.tick_and_gate(&inside, DT);
        assert_eq!(first[91 + 9].score, 0.0, "cold start → gated");
        // Hold in frame ~1 s: duty ≈ 0.9, engaged, passes through.
        let mut last = first;
        for _ in 0..30 {
            last = g.tick_and_gate(&inside, DT);
        }
        assert_eq!(last[91 + 9].score, 0.6, "sustained in-frame hand block engages");
        // The palms replay's worst real-hand zone: the 2D detector flips
        // to a below-frame phantom one frame in three. Duty ≈ 2/3 stays
        // above DUTY_OFF (and converges above DUTY_ON), so the hand keeps
        // driving the avatar on its visible frames.
        let outside = engage_joints(0.5, 1.15);
        for i in 0..30 {
            let src = if i % 3 == 2 { &outside } else { &inside };
            let gated = g.tick_and_gate(src, DT);
            if i % 3 != 2 {
                assert_eq!(gated[91 + 9].score, 0.6, "in-2/out-1 phantom flip keeps the hand engaged (frame {i})");
            }
        }
        // Hands back on the keyboard: sustained absence releases, and a
        // 1-frame re-entry peek stays gated.
        for _ in 0..30 {
            g.tick_and_gate(&outside, DT);
        }
        let re = g.tick_and_gate(&inside, DT);
        assert_eq!(re[91 + 9].score, 0.0, "sustained absence releases; a peek does not re-engage");
    }

    #[test]
    fn engage_gate_in_frame_confidence_dip_keeps_engagement() {
        // A hand plainly in view whose detection score dips for a frame
        // (motion blur) must NOT lose its engagement — bridging that gap
        // is the solver hold policy's job. Only leaving the frame counts
        // against the duty.
        let mut g = ArmEngageGate::default();
        let inside = engage_joints(0.5, 0.80);
        for _ in 0..30 {
            g.tick_and_gate(&inside, DT);
        }
        let mut dip = inside.clone();
        for &local in &ArmEngageGate::MCP_LOCALS {
            dip[91 + local].score = 0.01;
        }
        g.tick_and_gate(&dip, DT);
        let back = g.tick_and_gate(&inside, DT);
        assert_eq!(back[91 + 9].score, 0.6, "in-frame score dip must not release the hand");
    }

    #[test]
    fn scale_stabilizer_holds_span_through_contamination() {
        // A steady 0.36 m span with an occasional hand-contaminated frame
        // (0.20 m / 0.55 m) must not swing the held span: the outlier is
        // ignored and the avatar scale stays put.
        let mut s = TorsoScaleStabilizer::default();
        let seed = s.stable_span(Some(0.36), DT).unwrap();
        assert!((seed - 0.36).abs() < 1e-6, "seeds on the first plausible span");
        let after_low = s.stable_span(Some(0.20), DT).unwrap(); // hand in front → far too small
        let after_high = s.stable_span(Some(0.55), DT).unwrap(); // finger merge → far too large
        assert!(
            (after_low - 0.36).abs() < 0.02 && (after_high - 0.36).abs() < 0.02,
            "contaminated frames are rejected, span holds near 0.36 (got {after_low}, {after_high})"
        );
        // A genuine (plausible) reading eases the estimate slightly.
        let eased = s.stable_span(Some(0.34), DT).unwrap();
        assert!(eased < 0.36 && eased > 0.34, "plausible reading eases in slowly");
    }

    #[test]
    fn scale_stabilizer_seeds_default_on_bad_first_frame() {
        // If the very first frame is contaminated (implausible span), seed the
        // anatomical default rather than locking onto garbage.
        let mut s = TorsoScaleStabilizer::default();
        let seed = s.stable_span(Some(0.12), DT).unwrap();
        assert!((seed - 0.38).abs() < 1e-6, "implausible first span → default seed");
    }

    #[test]
    fn scale_stabilizer_span_holds_through_missing_measurement() {
        // A dropped shoulder (or a fabricated pair) contributes no
        // measurement: the held span must be returned, not a hard-coded
        // default — previously this popped the avatar scale to 0.38 and back.
        let mut s = TorsoScaleStabilizer::default();
        s.stable_span(Some(0.44), DT);
        let held = s.stable_span(None, DT);
        assert_eq!(held, Some(0.44), "missing measurement returns the held span");
        // And before any seed, None stays None (caller falls back explicitly).
        let mut fresh = TorsoScaleStabilizer::default();
        assert_eq!(fresh.stable_span(None, DT), None);
    }

    #[test]
    fn scale_stabilizer_span_reseeds_after_sustained_rejection() {
        // Dead-lock regression: a bogus first seed (0.46, finger-merge frame)
        // with a true span of 0.32 — |0.32−0.46| > 25% of 0.46 — used to
        // reject EVERY subsequent true reading forever, freezing the avatar at
        // the wrong scale for the whole session. After SPAN_RESEED_S of
        // consistent "outliers" the estimator must re-seed onto them. Also
        // covers a mid-session subject swap.
        let mut s = TorsoScaleStabilizer::default();
        s.stable_span(Some(0.46), DT);
        let mut last = 0.0;
        for _ in 0..(SPAN_RESEED_FRAMES_AT_30 + 1) {
            last = s.stable_span(Some(0.32), DT).unwrap();
        }
        assert!(
            (last - 0.32).abs() < 1e-6,
            "sustained true readings must re-seed the span (got {last})"
        );
        // An INTERMITTENT outlier burst shorter than the horizon still holds.
        let mut s2 = TorsoScaleStabilizer::default();
        s2.stable_span(Some(0.36), DT);
        for _ in 0..10 {
            s2.stable_span(Some(0.55), DT);
        }
        let held = s2.stable_span(Some(0.36), DT).unwrap();
        assert!((held - 0.36).abs() < 0.02, "short burst does not re-seed (got {held})");
    }

    #[test]
    fn scale_stabilizer_span_reseed_time_is_framerate_invariant() {
        // The reseed horizon is wall time, not a frame count: at 15 fps
        // (dt = 1/15) half as many rejected frames cover the same 1.5 s,
        // so the re-seed must fire after ~half the frames — and NOT fire
        // while the accumulated time is still short of the horizon.
        let dt15 = 2.0 * DT;
        let mut s = TorsoScaleStabilizer::default();
        s.stable_span(Some(0.46), dt15);
        // 20 rejects × (1/15 s) ≈ 1.33 s < 1.5 s → still held.
        let mut last = 0.0;
        for _ in 0..20 {
            last = s.stable_span(Some(0.32), dt15).unwrap();
        }
        assert!(
            (last - 0.46).abs() < 1e-6,
            "below the wall-time horizon the bogus seed still holds (got {last})"
        );
        // 3 more (≈1.53 s total) → re-seeded. At the old 45-frame count
        // this would have needed 22 more frames.
        for _ in 0..3 {
            last = s.stable_span(Some(0.32), dt15).unwrap();
        }
        assert!(
            (last - 0.32).abs() < 1e-6,
            "the horizon elapses in wall time at 15 fps (got {last})"
        );
    }

    #[test]
    fn scale_stabilizer_span_ema_is_framerate_invariant() {
        // Same wall-clock duration of the same plausible reading must land
        // on (nearly) the same span regardless of the frame rate the
        // duration was delivered at: 30 frames at 30 fps vs 15 frames at
        // 15 fps, both 1.0 s of easing 0.36 → 0.40.
        let mut s30 = TorsoScaleStabilizer::default();
        s30.stable_span(Some(0.36), DT);
        let mut a = 0.0;
        for _ in 0..30 {
            a = s30.stable_span(Some(0.40), DT).unwrap();
        }
        let mut s15 = TorsoScaleStabilizer::default();
        s15.stable_span(Some(0.36), 2.0 * DT);
        let mut b = 0.0;
        for _ in 0..15 {
            b = s15.stable_span(Some(0.40), 2.0 * DT).unwrap();
        }
        assert!(
            (a - b).abs() < 0.002,
            "1 s of easing must be frame-rate invariant (30 fps → {a}, 15 fps → {b})"
        );
    }

    #[test]
    fn scale_stabilizer_passes_real_move_rejects_spike() {
        let mut s = TorsoScaleStabilizer::default();
        let a0 = s.stable_anchor([0.0, 0.0, 1.60], DT);
        assert_eq!(a0, [0.0, 0.0, 1.60], "seeds on the first anchor");
        // A realistic walk-in step (~0.03 m/frame ≈ 1 m/s) passes through almost
        // fully: the avatar's near/far MUST reflect real movement.
        let step = s.stable_anchor([0.0, 0.0, 1.57], DT);
        assert!(step[2] < 1.585, "a real approach step is tracked promptly (got {})", step[2]);
        // A physically-impossible one-frame jump (a hand contaminating shoulder
        // depth) is rejected — the anchor barely moves.
        let spike = s.stable_anchor([0.0, 0.0, 1.05], DT)[2]; // ~0.5 m in one frame
        assert!(spike > 1.45, "a contamination spike is rejected (got {spike})");
    }

    #[test]
    fn scale_stabilizer_spike_gate_is_a_velocity_not_a_step() {
        // 0.16 m in one frame is contamination at 30 fps (4.8 m/s > 3.6)
        // but plausible motion after a dropped frame (dt = 1/15 → 2.4 m/s).
        // The frame-count gate rejected the latter, so a frame drop during
        // a genuine approach froze the avatar's distance.
        let mut fast = TorsoScaleStabilizer::default();
        fast.stable_anchor([0.0, 0.0, 1.60], DT);
        let z30 = fast.stable_anchor([0.0, 0.0, 1.44], DT)[2];
        assert!(
            (z30 - 1.60).abs() < 1e-6,
            "0.16 m in 1/30 s is rejected as contamination (got {z30})"
        );
        let mut slow = TorsoScaleStabilizer::default();
        slow.stable_anchor([0.0, 0.0, 1.60], 2.0 * DT);
        let z15 = slow.stable_anchor([0.0, 0.0, 1.44], 2.0 * DT)[2];
        assert!(
            z15 < 1.55,
            "the same step across a dropped frame (1/15 s) is followed (got {z15})"
        );
    }

    #[test]
    fn scale_stabilizer_anchor_holds_through_sustained_contamination() {
        // A pose HELD for seconds (namaste, cheek-on-hand) keeps producing the
        // same contaminated anchor. The old 15% reject-creep fully converged
        // onto it within ~20 frames (<1 s), dragging the person z-band along.
        // Rejection must now HOLD for the whole reseed horizon.
        let mut s = TorsoScaleStabilizer::default();
        s.stable_anchor([0.0, 0.0, 1.60], DT);
        let contaminated = [0.0, 0.0, 1.20]; // hand 0.4 m in front, held
        let mut z = 0.0;
        for _ in 0..(ANCHOR_RESEED_FRAMES_AT_30 - 2) {
            z = s.stable_anchor(contaminated, DT)[2];
        }
        assert!(
            (z - 1.60).abs() < 1e-6,
            "held contamination must not creep the anchor (got {z})"
        );
        // But past the horizon, mutually-consistent readings ARE the new
        // truth (subject swap / re-acquire) → re-seed.
        for _ in 0..4 {
            z = s.stable_anchor(contaminated, DT)[2];
        }
        assert!(
            (z - 1.20).abs() < 1e-3,
            "consistent readings past the horizon re-seed the anchor (got {z})"
        );
    }

    #[test]
    fn scale_stabilizer_anchor_does_not_reseed_onto_flicker() {
        // Sustained rejection whose readings DISAGREE (alternating between two
        // contaminants) is noise, not a new position — never re-seed onto it.
        let mut s = TorsoScaleStabilizer::default();
        s.stable_anchor([0.0, 0.0, 1.60], DT);
        let mut z = 0.0;
        for i in 0..(ANCHOR_RESEED_FRAMES_AT_30 + 20) {
            let raw = if i % 2 == 0 { [0.0, 0.0, 1.20] } else { [0.3, 0.0, 0.9] };
            z = s.stable_anchor(raw, DT)[2];
        }
        assert!(
            (z - 1.60).abs() < 1e-6,
            "inconsistent rejections must never re-seed (got {z})"
        );
    }

    #[test]
    fn scale_stabilizer_anchor_z_exposes_held_depth() {
        let mut s = TorsoScaleStabilizer::default();
        assert_eq!(s.anchor_z(), None);
        s.stable_anchor([0.1, 0.2, 1.5], DT);
        assert!((s.anchor_z().unwrap() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn scale_stabilizer_tracks_sustained_approach_without_lag() {
        // The old heavy EMA lagged a real approach ~0.16 m and dropped ~25% of
        // the motion (the "near/far not reflected" regression). A sustained ramp
        // must now converge to within a few cm.
        let mut s = TorsoScaleStabilizer::default();
        let mut z = 1.60;
        s.stable_anchor([0.0, 0.0, z], DT);
        let mut last = z;
        for _ in 0..60 {
            z -= 0.01; // 1.60 -> 1.00 over 2 s at 30 fps (~0.3 m/s)
            last = s.stable_anchor([0.0, 0.0, z], DT)[2];
        }
        assert!((last - 1.00).abs() < 0.03, "ramp tracked with <3 cm lag (got {last})");
    }

    #[test]
    fn frame_dt_tracker_derives_clamps_and_falls_back() {
        let mut t = FrameDtTracker::default();
        // No previous timestamp → nominal step.
        assert_eq!(t.tick(Some(1000.0)), NOMINAL_FRAME_DT_S);
        // Real consecutive-timestamp difference.
        assert!((t.tick(Some(1066.667)) - 0.066667).abs() < 1e-4);
        // Non-monotonic (device clock reset) → nominal, and the new
        // timestamp becomes the reference.
        assert_eq!(t.tick(Some(50.0)), NOMINAL_FRAME_DT_S);
        assert!((t.tick(Some(83.333)) - 0.033333).abs() < 1e-4);
        // Absent timestamps (synthetic frames) → nominal, reference kept.
        assert_eq!(t.tick(None), NOMINAL_FRAME_DT_S);
        // Pathological gap clamps to the stall ceiling.
        assert_eq!(t.tick(Some(9999.0)), 0.100);
        t.reset();
        assert_eq!(t.tick(Some(5.0)), NOMINAL_FRAME_DT_S);
    }

    fn dummy_frame(width: u32, height: u32, point: [f32; 3]) -> MetricDepthFrame {
        let pixels = (width * height) as usize;
        MetricDepthFrame {
            width,
            height,
            points_m: vec![point; pixels],
            crop: None,
            intrinsics: None,
            timestamp_ms: None,
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

    #[test]
    fn person_aware_sample_rejects_background_keeps_person() {
        // 5x5 frame; fill with background (5 m), then paint a person patch
        // (0.6 m) covering the left half. A window centred on the person edge
        // must return the person depth, not the background median.
        let mut points = vec![[0.0f32, 0.0, 5.0]; 25];
        for y in 0..5 {
            for x in 0..3 {
                points[y * 5 + x] = [0.0, 0.0, 0.6];
            }
        }
        let frame = MetricDepthFrame {
            width: 5, height: 5, points_m: points, crop: None, intrinsics: None,
            timestamp_ms: None,
        };
        // Centre pixel (x=2) sits at the silhouette: naive median leans toward
        // whichever half dominates, but person-aware with z_ref=0.6 keeps 0.6.
        let p = sample_metric_point_person(&frame, 0.5, 0.5, 2, 0.6).unwrap();
        assert!((p[2] - 0.6).abs() < 1e-6, "person-aware sample kept background: {}", p[2]);
        // A keypoint whose whole window is background (no person within band)
        // is unobservable → None, so the caller rests instead of teleporting.
        let bg = MetricDepthFrame {
            width: 5, height: 5, points_m: vec![[0.0, 0.0, 5.0]; 25], crop: None, intrinsics: None,
            timestamp_ms: None,
        };
        assert!(sample_metric_point_person(&bg, 0.5, 0.5, 2, 0.6).is_none());
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
            timestamp_ms: None,
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
        let fit = torso_fit::fit_torso(&frame, &joints, opts, None).expect("torso fit");
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
        let fit = torso_fit::fit_torso(&frame, &joints, opts, None).expect("torso fit");
        let r = fit.r_shoulder_cam.unwrap();
        let l = fit.l_shoulder_cam.unwrap();
        // Re-seated to the body depth → equal z (frontal), not 1.2 vs 2.0.
        assert!((r[2] - l[2]).abs() < 0.05, "r={:?} l={:?}", r, l);
        assert!(r[2] > 1.8, "occluded shoulder pulled to body depth: {:?}", r);
        assert!(fit.pair_reseated && !fit.pair_fabricated, "re-seat must be flagged");
    }

    /// The torso-teleport hole: a shoulder keypoint at the silhouette samples
    /// the wall 4 m back. "Deeper = body" then re-seated BOTH shoulders onto
    /// the wall. With the person band (previous stabilised anchor) the
    /// background sample is rejected at the source: that shoulder returns no
    /// sample and the fit anchors on the remaining one at body depth.
    #[test]
    fn fit_torso_band_rejects_background_shoulder() {
        let intr = intr_640();
        let mut frame = plane_frame(intr, [0.0, 0.0, 2.0], [0.0, 0.0, -1.0]);
        // idx5's pixels are the far wall at 5.0 m (silhouette overshoot).
        stamp_block(&mut frame, &intr, 0.42, 0.40, 5.0);
        let mut joints = vec![dj(0.5, 0.5, 0.0); NUM_JOINTS];
        joints[5] = dj(0.42, 0.40, 0.9);
        joints[6] = dj(0.58, 0.40, 0.9);
        let opts = BuildOptions { force_shoulder_anchor: true };
        let fit =
            torso_fit::fit_torso(&frame, &joints, opts, Some(2.0)).expect("torso fit");
        assert!(
            fit.r_shoulder_cam.is_none(),
            "background shoulder must be rejected by the band, got {:?}",
            fit.r_shoulder_cam
        );
        let anchor_z = fit.anchor_cam[2];
        assert!(
            (anchor_z - 2.0).abs() < 0.05,
            "anchor stays at body depth (got {anchor_z})"
        );
        // Un-banded first frame (no reference yet) keeps the legacy behaviour.
        let legacy = torso_fit::fit_torso(&frame, &joints, opts, None).expect("fit");
        assert!(legacy.r_shoulder_cam.is_some());
    }

    /// An elbow the detector places BELOW a head+shoulders crop (`ny > 1`,
    /// off-frame) has no depth pixel; it must be DROPPED, not fabricated at a
    /// false toward-camera depth. The fabricated forward elbow is what folds the
    /// forearm and breaks "hands spread in front of the chest". coco 7 → avatar
    /// RightLowerArm, coco 8 → LeftLowerArm.
    #[test]
    fn offframe_elbow_is_dropped_not_fabricated() {
        let intr = intr_640();
        let frame = plane_frame(intr, [0.0, 0.0, 0.6], [0.0, 0.0, -1.0]);
        let mut joints = vec![dj(0.5, 0.5, 0.0); NUM_JOINTS];
        joints[5] = dj(0.42, 0.40, 0.9); // R shoulder (image-left)
        joints[6] = dj(0.58, 0.40, 0.9); // L shoulder (image-right)
        joints[9] = dj(0.55, 0.80, 0.7); // L wrist, in frame
        joints[10] = dj(0.45, 0.80, 0.7); // R wrist, in frame
        joints[7] = dj(0.62, 1.05, 0.7); // R elbow BELOW the frame
        joints[8] = dj(0.38, 1.05, 0.7); // L elbow BELOW the frame
        let opts = BuildOptions { force_shoulder_anchor: true };
        let fit = torso_fit::fit_torso(&frame, &joints, opts, None).expect("torso fit");
        let mut latch = LrSwapLatch::default();
        let sk = build_skeleton(0, &joints, &frame, fit, None, None, &mut latch);
        assert!(
            sk.joints.get(&HumanoidBone::RightLowerArm).is_none(),
            "off-frame right elbow (coco7) must be dropped, got {:?}",
            sk.joints.get(&HumanoidBone::RightLowerArm)
        );
        assert!(
            sk.joints.get(&HumanoidBone::LeftLowerArm).is_none(),
            "off-frame left elbow (coco8) must be dropped, got {:?}",
            sk.joints.get(&HumanoidBone::LeftLowerArm)
        );
    }

    /// Erase every depth pixel in a normalised rect (a D435 near-limit /
    /// projector-shadow hole over the hand).
    fn hole_rect(frame: &mut MetricDepthFrame, nx0: f32, nx1: f32, ny0: f32, ny1: f32) {
        let (w, h) = (frame.width, frame.height);
        for v in ((ny0 * h as f32) as u32)..((ny1 * h as f32).ceil() as u32).min(h) {
            for u in ((nx0 * w as f32) as u32)..((nx1 * w as f32).ceil() as u32).min(w) {
                frame.points_m[(v * w + u) as usize] = [f32::NAN; 3];
            }
        }
    }

    /// Shoulders + right elbow on the body plane, right-hand block (base 91:
    /// wrist local 0, MCP locals 5/9/13/17) at the given centre.
    fn hand_scene_joints(hand_nx: f32, hand_ny: f32) -> Vec<DecodedJoint2d> {
        let mut joints = vec![dj(0.5, 0.5, 0.0); NUM_JOINTS];
        joints[5] = dj(0.42, 0.40, 0.9); // R shoulder (image-left)
        joints[6] = dj(0.58, 0.40, 0.9); // L shoulder
        joints[7] = dj(0.44, 0.58, 0.8); // R elbow, in frame on the body plane
        joints[91] = dj(hand_nx, hand_ny - 0.03, 0.7); // wrist
        for &l in &[5usize, 9, 13, 17] {
            joints[91 + l] = dj(hand_nx + (l as f32 - 11.0) * 0.002, hand_ny, 0.7);
        }
        joints
    }

    /// The 2026-07-28 bisect root cause: the hand's 2D is perfect while the
    /// depth under every hand keypoint is a hole (palms replay: 54 % of
    /// frames). The hand must ATTACH anyway — ray-solved a forearm length
    /// from the elbow and marked `Extrapolated` — instead of blinking out.
    #[test]
    fn hand_attaches_through_full_depth_hole_via_forearm_ray() {
        let intr = intr_640();
        let mut frame = plane_frame(intr, [0.0, 0.0, 0.6], [0.0, 0.0, -1.0]);
        hole_rect(&mut frame, 0.34, 0.50, 0.66, 0.82);
        let joints = hand_scene_joints(0.42, 0.72);
        let opts = BuildOptions { force_shoulder_anchor: true };
        let fit = torso_fit::fit_torso(&frame, &joints, opts, None).expect("torso fit");
        let mut latch = LrSwapLatch::default();
        let sk = build_skeleton(0, &joints, &frame, fit, None, None, &mut latch);
        let wrist = sk
            .joints
            .get(&HumanoidBone::RightHand)
            .expect("hand must attach through a full depth hole");
        assert_eq!(
            sk.joint_origins.get(&HumanoidBone::RightHand),
            Some(&crate::tracking::source_skeleton::JointOrigin::Extrapolated),
            "ray-solved wrist must be marked Extrapolated"
        );
        let z = wrist.metric_depth_m.expect("wrist carries its camera z");
        assert!(
            (0.15..0.65).contains(&z),
            "ray-solved wrist depth must be plausible (toward-camera), got {z}"
        );
    }

    /// Partial knuckle depth (one MCP sampled, the rest holed): the sampled
    /// knuckle fixes the hand's depth plane and the wrist keypoint its ray —
    /// the hand attaches at that plane instead of resting.
    #[test]
    fn hand_attaches_on_partial_knuckle_depth_at_the_sampled_plane() {
        let intr = intr_640();
        let mut frame = plane_frame(intr, [0.0, 0.0, 0.6], [0.0, 0.0, -1.0]);
        hole_rect(&mut frame, 0.25, 0.60, 0.66, 0.82);
        // Spread the MCPs wide enough that a tiny real surface under ONE of
        // them (the middle MCP, local 9 at −0.04) leaves the rest holed.
        let mut joints = hand_scene_joints(0.42, 0.72);
        for &l in &[5usize, 9, 13, 17] {
            joints[91 + l] = dj(0.42 + (l as f32 - 11.0) * 0.02, 0.72, 0.7);
        }
        let mid = &joints[91 + 9];
        let (mu, mv) = (
            (mid.nx * intr.width as f32).round() as i32,
            (mid.ny * intr.height as f32).round() as i32,
        );
        for v in (mv - 3)..=(mv + 3) {
            for u in (mu - 3)..=(mu + 3) {
                let d = [(u as f32 - intr.cx) / intr.fx, (v as f32 - intr.cy) / intr.fy, 1.0];
                frame.points_m[(v as u32 * intr.width + u as u32) as usize] =
                    [d[0] * 0.35, d[1] * 0.35, 0.35];
            }
        }
        let opts = BuildOptions { force_shoulder_anchor: true };
        let fit = torso_fit::fit_torso(&frame, &joints, opts, None).expect("torso fit");
        let mut latch = LrSwapLatch::default();
        let sk = build_skeleton(0, &joints, &frame, fit, None, None, &mut latch);
        let wrist = sk
            .joints
            .get(&HumanoidBone::RightHand)
            .expect("hand must attach on partial knuckle depth");
        let z = wrist.metric_depth_m.expect("wrist carries its camera z");
        assert!(
            (z - 0.35).abs() < 0.06,
            "wrist must sit on the sampled knuckle plane (~0.35 m), got {z}"
        );
        assert_eq!(
            sk.joint_origins.get(&HumanoidBone::RightHand),
            Some(&crate::tracking::source_skeleton::JointOrigin::Extrapolated),
        );
        assert!(
            sk.right_hand_orientation.is_some(),
            "palm orientation must survive the plane fallback"
        );
    }

    /// A border-riding hand block with NO depth proof must not be pinned in
    /// 3D: RTMW3D border-clamps genuinely off-frame (desk-typing) hands to
    /// the image edge, and fabricating those would re-open the very artefact
    /// the fallback exists to fix.
    #[test]
    fn borderclamped_hand_without_depth_stays_rested() {
        let intr = intr_640();
        let mut frame = plane_frame(intr, [0.0, 0.0, 0.6], [0.0, 0.0, -1.0]);
        hole_rect(&mut frame, 0.30, 0.55, 0.90, 1.0);
        let mut joints = hand_scene_joints(0.42, 0.995);
        joints[91] = dj(0.42, 0.995, 0.7); // wrist border-clamped too
        let opts = BuildOptions { force_shoulder_anchor: true };
        let fit = torso_fit::fit_torso(&frame, &joints, opts, None).expect("torso fit");
        let mut latch = LrSwapLatch::default();
        let sk = build_skeleton(0, &joints, &frame, fit, None, None, &mut latch);
        assert!(
            sk.joints.get(&HumanoidBone::RightHand).is_none(),
            "border-clamped hand with no depth must rest, got {:?}",
            sk.joints.get(&HumanoidBone::RightHand)
        );
    }

    /// Full depth available → the wrist keeps its directly observed sample
    /// and stays `Observed` (no origin entry), exactly as before the
    /// fallback existed.
    #[test]
    fn hand_with_full_depth_stays_observed() {
        let intr = intr_640();
        let frame = plane_frame(intr, [0.0, 0.0, 0.6], [0.0, 0.0, -1.0]);
        let joints = hand_scene_joints(0.42, 0.72);
        let opts = BuildOptions { force_shoulder_anchor: true };
        let fit = torso_fit::fit_torso(&frame, &joints, opts, None).expect("torso fit");
        let mut latch = LrSwapLatch::default();
        let sk = build_skeleton(0, &joints, &frame, fit, None, None, &mut latch);
        let wrist = sk
            .joints
            .get(&HumanoidBone::RightHand)
            .expect("hand attaches on full depth");
        assert!(
            sk.joint_origins.get(&HumanoidBone::RightHand).is_none(),
            "fully sampled wrist stays Observed"
        );
        let z = wrist.metric_depth_m.expect("wrist camera z");
        assert!((z - 0.6).abs() < 0.05, "wrist on the body plane, got {z}");
    }

    /// Contrast: an IN-FRAME elbow whose depth window is a HOLE (a dark sleeve
    /// eats the IR) is still recovered by the bone-length ray fallback — the
    /// off-frame gate must not regress the in-frame-hole path.
    #[test]
    fn inframe_hole_elbow_is_recovered_by_bone_length() {
        let intr = intr_640();
        let mut frame = plane_frame(intr, [0.0, 0.0, 0.6], [0.0, 0.0, -1.0]);
        let mut joints = vec![dj(0.5, 0.5, 0.0); NUM_JOINTS];
        joints[5] = dj(0.42, 0.40, 0.9);
        joints[6] = dj(0.58, 0.40, 0.9);
        let (enx, eny) = (0.62_f32, 0.60_f32); // R elbow, IN frame
        joints[7] = dj(enx, eny, 0.7);
        // Punch a NaN depth hole around the elbow pixel (leaves shoulders valid).
        let (cx, cy) = ((enx * intr.width as f32) as i32, (eny * intr.height as f32) as i32);
        for v in (cy - 8).max(0)..=(cy + 8).min(intr.height as i32 - 1) {
            for u in (cx - 8).max(0)..=(cx + 8).min(intr.width as i32 - 1) {
                frame.points_m[(v as u32 * intr.width + u as u32) as usize] = [f32::NAN; 3];
            }
        }
        let opts = BuildOptions { force_shoulder_anchor: true };
        let fit = torso_fit::fit_torso(&frame, &joints, opts, None).expect("torso fit");
        let mut latch = LrSwapLatch::default();
        let sk = build_skeleton(0, &joints, &frame, fit, None, None, &mut latch);
        assert!(
            sk.joints.get(&HumanoidBone::RightLowerArm).is_some(),
            "in-frame elbow at a depth hole should be recovered by the bone-length fallback"
        );
        // Provenance: a ray-fallback elbow is inference, not measurement.
        assert_eq!(
            sk.origin(HumanoidBone::RightLowerArm),
            crate::tracking::source_skeleton::JointOrigin::Extrapolated,
            "bone-length fallback must be marked Extrapolated"
        );
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
        let fit = torso_fit::fit_torso(&frame, &joints, opts, None).expect("torso fit");
        let r = fit.r_shoulder_cam.unwrap();
        let l = fit.l_shoulder_cam.unwrap();
        assert!((r[2] - l[2]).abs() < 1e-4, "equal depth expected");
        assert!(r[0] > l[0], "avatar-Right at larger camera-x for frontal");
        assert!((r[0] - l[0]).abs() > 0.25, "span ~0.36 expected, got {}", r[0] - l[0]);
        assert!(fit.pair_fabricated, "canonical pair must be flagged fabricated");
        // And the built skeleton carries the Synthesized provenance mark so
        // the span EMA / any statistical consumer can refuse it.
        let mut latch = LrSwapLatch::default();
        let sk = build_skeleton(0, &joints, &frame, fit, None, None, &mut latch);
        assert_eq!(
            sk.origin(HumanoidBone::LeftUpperArm),
            crate::tracking::source_skeleton::JointOrigin::Synthesized,
        );
        assert_eq!(
            sk.origin(HumanoidBone::RightUpperArm),
            crate::tracking::source_skeleton::JointOrigin::Synthesized,
        );
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

    // -----------------------------------------------------------------------
    // L/R swap hysteresis
    // -----------------------------------------------------------------------

    fn sk_with(joints: &[(HumanoidBone, [f32; 3])]) -> SourceSkeleton {
        let mut sk = SourceSkeleton::empty(0);
        for &(b, p) in joints {
            sk.joints.insert(
                b,
                SourceJoint {
                    position: p,
                    confidence: 0.9,
                    metric_depth_m: None,
                },
            );
        }
        sk
    }

    #[test]
    fn lr_swap_strong_reversal_still_corrects_instantly() {
        // A clear transposition must be undone on THIS frame (the detector
        // genuinely flips per frame) — the latch must not delay it.
        let mut latch = LrSwapLatch::default();
        let mut sk = sk_with(&[
            (HumanoidBone::LeftUpperArm, [-0.15, 0.0, 0.0]),
            (HumanoidBone::RightUpperArm, [0.15, 0.0, 0.0]),
            (HumanoidBone::LeftLowerArm, [-0.2, -0.3, 0.0]),
            (HumanoidBone::RightLowerArm, [0.2, -0.3, 0.0]),
        ]);
        correct_upper_body_lr_swap(&mut sk, &mut latch);
        assert!(
            sk.joints[&HumanoidBone::LeftUpperArm].position[0] > 0.0,
            "clear transposition corrected on the same frame"
        );
    }

    #[test]
    fn lr_swap_weak_zone_repeats_previous_decision() {
        // Ambiguous geometry (near-tied shoulders, no confident votes) must
        // repeat the previous frame's decision instead of flapping the whole
        // body at frame rate around the threshold.
        let weak = || {
            sk_with(&[
                (HumanoidBone::LeftUpperArm, [0.02, 0.0, 0.0]),
                (HumanoidBone::RightUpperArm, [-0.02, 0.3, 0.0]),
            ])
        };

        // Fresh latch (never swapped) → weak evidence keeps NOT swapping.
        let mut latch = LrSwapLatch::default();
        let mut sk = weak();
        correct_upper_body_lr_swap(&mut sk, &mut latch);
        assert!(
            (sk.joints[&HumanoidBone::LeftUpperArm].position[0] - 0.02).abs() < 1e-6,
            "weak evidence with a cold latch must not swap"
        );

        // Engage the latch with one strong frame…
        let mut strong = sk_with(&[
            (HumanoidBone::LeftUpperArm, [-0.15, 0.0, 0.0]),
            (HumanoidBone::RightUpperArm, [0.15, 0.0, 0.0]),
            (HumanoidBone::LeftLowerArm, [-0.2, -0.3, 0.0]),
            (HumanoidBone::RightLowerArm, [0.2, -0.3, 0.0]),
        ]);
        correct_upper_body_lr_swap(&mut strong, &mut latch);
        // …then the SAME weak frame now keeps the swap decision (hysteresis).
        let mut sk2 = weak();
        correct_upper_body_lr_swap(&mut sk2, &mut latch);
        assert!(
            (sk2.joints[&HumanoidBone::LeftUpperArm].position[0] + 0.02).abs() < 1e-6,
            "weak evidence with an engaged latch keeps swapping"
        );

        // Strongly-normal geometry releases the latch instantly.
        let mut normal = sk_with(&[
            (HumanoidBone::LeftUpperArm, [0.15, 0.0, 0.0]),
            (HumanoidBone::RightUpperArm, [-0.15, 0.0, 0.0]),
        ]);
        correct_upper_body_lr_swap(&mut normal, &mut latch);
        assert!(
            (normal.joints[&HumanoidBone::LeftUpperArm].position[0] - 0.15).abs() < 1e-6,
            "clearly-normal frame is left untouched"
        );
        let mut sk3 = weak();
        correct_upper_body_lr_swap(&mut sk3, &mut latch);
        assert!(
            (sk3.joints[&HumanoidBone::LeftUpperArm].position[0] - 0.02).abs() < 1e-6,
            "latch released by the clearly-normal frame"
        );
    }

    // -----------------------------------------------------------------------
    // Head: one-ear lateral correction + fabricated-head provenance
    // -----------------------------------------------------------------------

    #[test]
    fn one_ear_head_is_shifted_toward_centre() {
        let intr = intr_640();
        let frame = plane_frame(intr, [0.0, 0.0, 2.0], [0.0, 0.0, -1.0]);
        let base = |joints: &mut Vec<DecodedJoint2d>| {
            joints[5] = dj(0.42, 0.40, 0.9);
            joints[6] = dj(0.58, 0.40, 0.9);
            joints[0] = dj(0.50, 0.33, 0.9); // nose
        };
        let opts = BuildOptions { force_shoulder_anchor: true };

        // Both ears visible → reference head at the ear midpoint.
        let mut j_both = vec![dj(0.5, 0.5, 0.0); NUM_JOINTS];
        base(&mut j_both);
        j_both[3] = dj(0.46, 0.35, 0.9);
        j_both[4] = dj(0.54, 0.35, 0.9);
        let fit = torso_fit::fit_torso(&frame, &j_both, opts, None).unwrap();
        let mut latch = LrSwapLatch::default();
        let sk_both = build_skeleton(0, &j_both, &frame, fit, None, None, &mut latch);
        let head_both = sk_both.joints[&HumanoidBone::Head].position;

        // Only the left ear (idx 3) visible → head must sit near the centre,
        // not at the raw ear (which is ~0.19 source units off-centre here).
        let mut j_one = vec![dj(0.5, 0.5, 0.0); NUM_JOINTS];
        base(&mut j_one);
        j_one[3] = dj(0.46, 0.35, 0.9);
        j_one[4] = dj(0.54, 0.35, 0.0); // below visibility floor
        let fit = torso_fit::fit_torso(&frame, &j_one, opts, None).unwrap();
        let mut latch = LrSwapLatch::default();
        let sk_one = build_skeleton(0, &j_one, &frame, fit, None, None, &mut latch);
        let head_one = sk_one.joints[&HumanoidBone::Head].position;

        assert!(
            (head_one[0] - head_both[0]).abs() < 0.06,
            "one-ear head stays near the centre: both={:?} one={:?}",
            head_both,
            head_one
        );
        assert_eq!(
            sk_one.origin(HumanoidBone::Head),
            crate::tracking::source_skeleton::JointOrigin::Extrapolated,
            "one-ear head is extrapolated, not observed"
        );
    }

    #[test]
    fn fabricated_fallback_head_has_decayed_confidence() {
        let intr = intr_640();
        let frame = plane_frame(intr, [0.0, 0.0, 2.0], [0.0, 0.0, -1.0]);
        let mut joints = vec![dj(0.5, 0.5, 0.0); NUM_JOINTS];
        joints[5] = dj(0.42, 0.40, 0.9);
        joints[6] = dj(0.58, 0.40, 0.9);
        // Nose / ears all below the visibility floor → anatomical fallback.
        let opts = BuildOptions { force_shoulder_anchor: true };
        let fit = torso_fit::fit_torso(&frame, &joints, opts, None).unwrap();
        let mut latch = LrSwapLatch::default();
        let sk = build_skeleton(0, &joints, &frame, fit, None, None, &mut latch);
        let head = sk.joints[&HumanoidBone::Head];
        let neck = sk.joints[&HumanoidBone::Neck];
        assert!(
            head.confidence <= neck.confidence * FABRICATED_HEAD_CONF_SCALE + 1e-6,
            "fabricated head must not carry measurement-grade confidence \
             (head {} vs neck {})",
            head.confidence,
            neck.confidence
        );
        assert_eq!(
            sk.origin(HumanoidBone::Head),
            crate::tracking::source_skeleton::JointOrigin::Synthesized,
        );
    }

    // -----------------------------------------------------------------------
    // Torso capture buffer: band + admit ratio
    // -----------------------------------------------------------------------

    #[test]
    fn capture_buffer_bands_out_background_and_requires_coverage() {
        // 64×64 frame: person plane at 1.0 m in the central region, wall at
        // 5.0 m elsewhere. Anchors sit on the person.
        let (w, h) = (64u32, 64u32);
        let mut points = vec![[0.0f32, 0.0, 5.0]; (w * h) as usize];
        for y in 16..56 {
            for x in 20..44 {
                points[y * w as usize + x] = [0.0, 0.0, 1.0];
            }
        }
        // An "armpit gap": a see-through hole INSIDE the torso bbox where the
        // wall shows every frame — the structural contaminant the temporal
        // median can't reject (it's not random noise).
        for y in 18..30 {
            for x in 30..34 {
                points[y * w as usize + x] = [0.0, 0.0, 5.0];
            }
        }
        let frame = MetricDepthFrame {
            width: w,
            height: h,
            points_m: points,
            crop: None,
            intrinsics: None,
            timestamp_ms: None,
        };
        let mut joints = vec![dj(0.5, 0.5, 0.0); NUM_JOINTS];
        joints[5] = dj(0.34, 0.28, 0.9); // shoulders
        joints[6] = dj(0.66, 0.28, 0.9);
        joints[11] = dj(0.38, 0.84, 0.9); // hips
        joints[12] = dj(0.62, 0.84, 0.9);
        let mut buf = TorsoCaptureBuffer::new();
        assert!(buf.add_frame(&joints, &frame), "torso frame admitted");
        let tpl = buf.finalize().expect("template");
        // Background pixels sit at the same cells every frame: without the
        // band they'd survive the median and be baked in as torso. Banded,
        // they must be NaN, and person cells must read the person depth.
        let has_person = tpl.depths_m.iter().any(|d| d.is_finite() && (d - 1.0).abs() < 0.05);
        assert!(has_person, "person cells captured");
        assert!(
            !tpl.depths_m.iter().any(|d| d.is_finite() && *d > 2.0),
            "no background depth may survive into the template"
        );
        assert!(
            tpl.depths_m.iter().any(|d| d.is_nan()),
            "the armpit-gap cells must finalize to NaN, not wall depth"
        );

        // A frame whose anchors see only the wall (subject gone) is refused
        // outright — its cells would all be out-of-band.
        let wall = MetricDepthFrame {
            width: w,
            height: h,
            points_m: vec![[0.0, 0.0, 5.0]; (w * h) as usize],
            crop: None,
            intrinsics: None,
            timestamp_ms: None,
        };
        let mut buf2 = TorsoCaptureBuffer::new();
        // Anchors read 5.0 → z_ref 5.0; every cell is in-band at 5.0, so this
        // frame IS admitted (the buffer can't know it's a wall) — coverage
        // still gates. But with the person present and anchors on the person,
        // a mostly-void depth map must be refused:
        let mut sparse = wall.points_m.clone();
        for p in sparse.iter_mut() {
            *p = [f32::NAN; 3];
        }
        // leave a single valid pixel at an anchor so z_ref exists
        let ax = (0.34 * w as f32).round() as usize;
        let ay = (0.28 * h as f32).round() as usize;
        sparse[ay * w as usize + ax] = [0.0, 0.0, 1.0];
        let bx = (0.66 * w as f32).round() as usize;
        sparse[ay * w as usize + bx] = [0.0, 0.0, 1.0];
        let sparse_frame = MetricDepthFrame {
            width: w,
            height: h,
            points_m: sparse,
            crop: None,
            intrinsics: None,
            timestamp_ms: None,
        };
        assert!(
            !buf2.add_frame(&joints, &sparse_frame),
            "a near-void frame (two valid pixels) must be refused by the coverage gate"
        );
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
