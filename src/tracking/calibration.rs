//! Per-session pose calibration types.
//!
//! See `docs/calibration-ux.md` for the rationale. The short version:
//! depth-aware tracking providers read the subject's pelvic anchor from
//! a per-pixel depth map, and a desk / monitor / chair occluding the
//! depth window biases that anchor by 0.3–1.5 m. The bias propagates
//! through every downstream consumer (calibration scale, body yaw,
//! root translation). This module carries the explicit reference
//! values the user captures via the `Calibrate Pose ▼` modal so the
//! solver and `skeleton_from_depth` can clamp / seed against a
//! known-good baseline rather than the auto-EMA's first frame.
//!
//! Phase A (this commit) only defines the types and the
//! `apply_calibration` plumbing; the modal UI (Phase B), capture loop
//! (Phase C), and solver / provider integration (Phase D) land in
//! follow-up commits. The `TrackingCalibration` struct in
//! [`super::TrackingCalibration`] now carries an optional
//! [`PoseCalibration`] so persisted projects with calibration data can
//! already be loaded into memory; consumers fall back to the existing
//! auto-EMA / hardcoded clamp behaviour as long as
//! [`PoseCalibration::is_active`] returns false.

use serde::{Deserialize, Serialize};

/// Which pose the subject took during the calibration capture window.
/// Drives both the on-screen instructions and the anchor-selection
/// policy in [`super::skeleton_from_depth`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationMode {
    /// Subject visible from feet to head, T-pose. Hip pair midpoint
    /// (COCO 11/12) is the captured anchor; `skeleton_from_depth`
    /// continues to prefer hip with shoulder fallback.
    FullBody,
    /// Subject visible from chest / waist up, hands at sides. Shoulder
    /// pair midpoint (COCO 5/6) is the captured anchor; once active,
    /// `skeleton_from_depth` is *forced* onto the shoulder anchor even
    /// when the hip pair clears the visibility floor — when the user
    /// said "upper body only", an apparent hip detection is almost
    /// certainly the desk surface or chair seat.
    UpperBody,
}

impl CalibrationMode {
    /// Stable string for serialization. Decoupled from the enum's
    /// `Debug` impl so the on-disk schema is independent of Rust
    /// formatting choices.
    pub fn as_str(self) -> &'static str {
        match self {
            CalibrationMode::FullBody => "full_body",
            CalibrationMode::UpperBody => "upper_body",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "full_body" => Some(CalibrationMode::FullBody),
            "upper_body" => Some(CalibrationMode::UpperBody),
            _ => None,
        }
    }
}

/// One calibration capture's median-aggregated values.
///
/// Built by the modal's capture loop (Phase C) from a 2-second
/// collection window of source-skeleton samples; consumed by the
/// solver (root-translation EMA seed) and `skeleton_from_depth`
/// (calibration-scale clamp + anchor mode).
///
/// Fields are stored in source-skeleton coordinates (`x ∈
/// [-aspect, +aspect]`, `y ∈ [-1, +1]`, depth in metres) so consumers
/// can drop them in without unit conversion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoseCalibration {
    pub mode: CalibrationMode,
    /// ISO-8601 capture timestamp. `String` rather than `SystemTime`
    /// because the on-disk schema needs a stable, human-readable
    /// representation that survives `serde_json` / version changes.
    pub captured_at: String,
    /// Unix seconds at capture. Persisted alongside [`Self::captured_at`]
    /// so the inspector's "Calibrated N min ago" relative-time
    /// rendering doesn't have to parse the ISO string each frame.
    /// Both fields are written together by `aggregate()` and consumed
    /// together; if they disagree (manually-edited project file) the
    /// unix value wins.
    #[serde(default)]
    pub captured_at_unix: u64,
    /// Number of frames whose median produced these values. Below 5
    /// indicates a `Capture Now` cut-short with sparse samples; the
    /// inspector flags the calibration as "low-sample" in that case.
    pub frame_count: usize,
    /// Image-centre-relative anchor X.
    pub anchor_x: f32,
    /// Image-centre-relative anchor Y (Y-up convention).
    pub anchor_y: f32,
    /// Camera-space metric depth of the anchor pair midpoint.
    /// `None` for `rtmw3d` (no depth pipeline) — the rtmw3d-only
    /// solver path uses `anchor_x` / `anchor_y` only.
    pub anchor_depth_m: Option<f32>,
    /// Median per-frame `overall_confidence` from the captured samples.
    /// Surfaced on the inspector to flag setups where calibration was
    /// taken under marginal lighting / framing conditions.
    pub confidence: f32,
    /// Per-frame stddev of the captured anchor depth. Drives the
    /// dynamic `MIN_C / MAX_C` clamp in `skeleton_from_depth`.
    /// `None` for non-depth providers.
    pub anchor_depth_jitter_m: Option<f32>,
    /// Optional per-axis range data captured by the multi-step
    /// calibration extension (steps 2–5: step left/right, lean
    /// in/out). When present, the solver derives per-axis
    /// translation sensitivity from the *observed* range — a user
    /// with a narrow room (small ±X) gets a higher gain so a
    /// 30 cm step still moves the avatar across half its
    /// world-space horizontal envelope. `None` when the user
    /// skipped the extended steps; solver falls back to the
    /// static [0.6, 0.6, 0.3] default.
    #[serde(default)]
    pub x_range_observed: Option<f32>,
    #[serde(default)]
    pub z_range_observed: Option<f32>,
}

impl PoseCalibration {
    /// `true` once a capture has completed and at least one frame was
    /// gathered. Used by consumers to decide between calibration-aware
    /// and auto-EMA / hardcoded-clamp behaviour.
    pub fn is_active(&self) -> bool {
        self.frame_count > 0
    }
}
