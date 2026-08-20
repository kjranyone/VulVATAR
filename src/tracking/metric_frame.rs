//! Shared metric-depth frame types: the aligned D435 point cloud handed
//! to the pose provider each frame ([`MetricDepthFrame`]) and the
//! capture-clock dt tracker ([`FrameDtTracker`]). Consumed by the fusion
//! estimator (`tracking::fusion`) and the offline replay benches.

use super::source_skeleton::CameraIntrinsics;

/// Per-pixel metric point cloud, in metres, in the depth model's
/// input pixel grid (`width × height`). Invalid pixels (mask below
/// threshold or non-finite output) are stored as `NaN` in `points_m`
/// so consumers' local-window median sampling naturally rejects
/// them. Per-pixel metric Z is `points_m[i][2]` — no separate depth
/// map is kept since the only current consumer samples the full xyz,
/// not depth alone. Always full-frame and aligned 1:1 to the colour
/// image the keypoints were detected in.
#[derive(Clone)]
pub struct MetricDepthFrame {
    pub width: u32,
    pub height: u32,
    pub points_m: Vec<[f32; 3]>,
    /// Pinhole intrinsics of the colour image this depth was aligned to,
    /// carried through so the built skeleton can expose them on
    /// [`MetricFrameInfo`] for the 1:1 sensor-matched render. `None` when
    /// the frame was synthesised without a real sensor (tests).
    pub intrinsics: Option<CameraIntrinsics>,
    /// Device capture timestamp of the colour/depth frameset, in
    /// milliseconds on the sensor's clock (D435 hardware timestamp).
    /// Consecutive-frame differences drive the dt-normalised temporal
    /// estimators (face-source crossfade, fusion process noise) so
    /// their time constants hold under frame drops / non-30-fps streams.
    /// `None` for synthetic frames
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
pub struct FrameDtTracker {
    last_ts_ms: Option<f64>,
}

/// Nominal frame step used whenever a real capture dt is unavailable.
/// All dt-normalised constants reproduce their historical 30-fps
/// behaviour exactly at this step.
pub const NOMINAL_FRAME_DT_S: f32 = 1.0 / 30.0;

impl FrameDtTracker {
    /// dt bounds: 5 ms (~200 fps — anything faster is duplicate/burst
    /// delivery, not subject motion) to 100 ms (~10 fps — beyond that a
    /// gap is a stall, and stretching rate gates further would let a
    /// contamination jump masquerade as plausible motion).
    const DT_MIN_S: f32 = 0.005;
    const DT_MAX_S: f32 = 0.100;

    pub fn tick(&mut self, ts_ms: Option<f64>) -> f32 {
        let dt = match (self.last_ts_ms, ts_ms) {
            (Some(prev), Some(cur)) if cur > prev => ((cur - prev) / 1000.0) as f32,
            _ => NOMINAL_FRAME_DT_S,
        };
        if ts_ms.is_some() {
            self.last_ts_ms = ts_ms;
        }
        dt.clamp(Self::DT_MIN_S, Self::DT_MAX_S)
    }

    pub fn reset(&mut self) {
        self.last_ts_ms = None;
    }
}

/// Wrap a D435 color-aligned depth frame into a [`MetricDepthFrame`]:
/// deproject every pixel through the *real* camera intrinsics into a
/// metric point cloud (metres, x-right / y-down / z-forward), marking
/// no-return pixels as `NaN`. There is no learned scale and no
/// centered-principal-point assumption: the D435 supplies absolute
/// metres and a true `(cx, cy)`. The depth is full-frame, aligned 1:1
/// to the color image the keypoints were detected in.
#[cfg(feature = "realsense")]
pub fn build_metric_frame_from_d435(
    frame: &crate::tracking::realsense::RealSenseFrame,
) -> MetricDepthFrame {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let mut points_m = Vec::with_capacity(w * h);
    for v in 0..h {
        for u in 0..w {
            match frame.point_m(u, v) {
                Some(p) => points_m.push(p),
                None => points_m.push([f32::NAN, f32::NAN, f32::NAN]),
            }
        }
    }
    let intr = frame.intrinsics;
    MetricDepthFrame {
        width: frame.width,
        height: frame.height,
        points_m,
        intrinsics: Some(CameraIntrinsics {
            fx: intr.fx,
            fy: intr.fy,
            cx: intr.cx,
            cy: intr.cy,
            width: intr.width,
            height: intr.height,
        }),
        timestamp_ms: Some(frame.timestamp_ms),
    }
}
