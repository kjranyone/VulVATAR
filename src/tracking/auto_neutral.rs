//! Session-automatic neutral estimation — the ritual-free counterpart of
//! the `Calibrate Pose` capture.
//!
//! The neutralisation layer (oblique-camera body yaw, per-source habitual
//! head pose, calibrated root anchor) is critical-path correctness for any
//! camera that is not dead-frontal, yet it used to activate only after the
//! user completed the calibration modal — which a desk-framed streamer
//! could not even start for most of this project's life. This module
//! watches the RAW sample stream during ordinary use and, per channel,
//! seeds the same [`PoseCalibration`] fields the modal would have
//! captured, as soon as a sliding window of samples passes the SAME
//! stability gates the modal's finalize step applies (median aggregation,
//! MAD stillness, plausibility caps). Root-reference lock-in already
//! works this way; this extends the idea to the remaining neutrals.
//!
//! Contract:
//! * An EXPLICIT calibration (modal capture or persisted project data)
//!   always wins — callers consult [`AutoNeutral::calibration`] only when
//!   `TrackingCalibration::pose` is `None`.
//! * Each channel seeds ONCE per session and then freezes (matching the
//!   root reference's "seed → lock-in → freeze" philosophy): a neutral
//!   that keeps re-estimating would absorb intentional posture changes.
//! * Session-only: nothing here is persisted. A fresh session re-seeds
//!   within seconds; the explicit capture remains the way to make a
//!   neutral durable.

use std::collections::VecDeque;

use log::info;

use super::calibration::{shoulder_line_yaw, CalibrationMode, PoseCalibration, BODY_YAW_MAX_RAD};
use super::source_skeleton::FaceSource;
use super::SourceSkeleton;

/// Sliding-window length per channel. ~5 s at the 30 fps capture rate —
/// deliberately longer than the modal's 2 s hold, since ordinary use is
/// noisier than a deliberate hold and a wrong auto-neutral is worse than
/// a late one.
const WINDOW: usize = 150;

/// Per-channel MAD stillness caps — identical to the modal finalize
/// gates (`FACE_NEUTRAL_MAX_MAD_RAD` / `BODY_YAW_MAX_MAD_RAD`), so the
/// auto path cannot bake in anything the manual path would reject.
const MAX_MAD_RAD: f32 = 0.15;
/// Anchor stillness cap (metres, per axis). A working user breathes and
/// types (~1-2 cm); a lean or a chair scoot is far beyond this and slides
/// the window until the posture settles.
const ANCHOR_MAX_MAD_M: f32 = 0.03;
/// Confidence floor for face-pose samples entering the window.
const FACE_SAMPLE_MIN_CONF: f32 = 0.5;

/// One scalar channel with a capped sliding window that freezes on seed.
#[derive(Debug, Default, Clone)]
struct ScalarChannel {
    window: VecDeque<f32>,
    seeded: Option<f32>,
}

impl ScalarChannel {
    /// Push a sample; when the window is full and its MAD passes, seed
    /// (once) with the window median. `cap` bounds the |median|;
    /// exceeding it discards the window (detector garbage, not geometry).
    fn push(&mut self, v: f32, mad_max: f32, cap: Option<f32>) {
        if self.seeded.is_some() || !v.is_finite() {
            return;
        }
        self.window.push_back(v);
        if self.window.len() > WINDOW {
            self.window.pop_front();
        }
        if self.window.len() < WINDOW {
            return;
        }
        let mut vals: Vec<f32> = self.window.iter().copied().collect();
        let median = median_inplace(&mut vals);
        let mut dev: Vec<f32> = self.window.iter().map(|s| (s - median).abs()).collect();
        if median_inplace(&mut dev) > mad_max {
            return; // window slides until the posture settles
        }
        if let Some(cap) = cap {
            if median.abs() > cap {
                self.window.clear();
                return;
            }
        }
        self.seeded = Some(median);
    }
}

/// Three scalar channels seeded together (a face pose's yaw/pitch/roll —
/// per-axis windows would seed a chimera of different moments).
#[derive(Debug, Default, Clone)]
struct TripleChannel {
    window: VecDeque<[f32; 3]>,
    seeded: Option<[f32; 3]>,
}

impl TripleChannel {
    fn push(&mut self, v: [f32; 3], mad_max: [f32; 3]) {
        if self.seeded.is_some() || v.iter().any(|c| !c.is_finite()) {
            return;
        }
        self.window.push_back(v);
        if self.window.len() > WINDOW {
            self.window.pop_front();
        }
        if self.window.len() < WINDOW {
            return;
        }
        let mut medians = [0.0f32; 3];
        for axis in 0..3 {
            let mut vals: Vec<f32> = self.window.iter().map(|s| s[axis]).collect();
            medians[axis] = median_inplace(&mut vals);
        }
        for axis in 0..3 {
            if mad_max[axis].is_finite() {
                let mut dev: Vec<f32> =
                    self.window.iter().map(|s| (s[axis] - medians[axis]).abs()).collect();
                if median_inplace(&mut dev) > mad_max[axis] {
                    return;
                }
            }
        }
        self.seeded = Some(medians);
    }
}

/// Continuous, per-channel automatic neutral estimation over the RAW
/// sample stream. One instance lives on `Application`; `ingest` runs
/// once per distinct capture sample, and [`Self::calibration`] yields a
/// synthesised [`PoseCalibration`] once the base (anchor) channel has
/// seeded.
#[derive(Debug, Default)]
pub struct AutoNeutral {
    /// Dedupe key of the last ingested sample (`source_timestamp`,
    /// capture-timestamp bits) — `ingest` is called from the render loop,
    /// which runs faster than the camera.
    last_key: Option<(u64, u64)>,
    /// Anchor position (source-oriented metres) + the anchor type the
    /// window was collected under. An anchor-type flip mid-window resets
    /// it: the two anchors sit ~0.4 m apart and must not be medianed
    /// together.
    anchor_x: ScalarChannel,
    anchor_y: ScalarChannel,
    anchor_z: ScalarChannel,
    anchor_is_hip: Option<bool>,
    confidence: ScalarChannel,
    body_yaw: ScalarChannel,
    face_mesh: TripleChannel,
    face_body: TripleChannel,
    /// Rebuilt whenever a channel seeds; `None` until the anchor seeds.
    cached: Option<PoseCalibration>,
    logged_channels: u8,
}

impl AutoNeutral {
    /// Forget everything — call on tracking session restart so a
    /// previous camera placement can't leak into the new session.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// The synthesised calibration, once enough of the scene has proven
    /// stable. `None` until the anchor channel seeds. Callers must
    /// prefer an explicit `TrackingCalibration::pose` over this.
    pub fn calibration(&self) -> Option<&PoseCalibration> {
        self.cached.as_ref()
    }

    /// Feed one RAW (pre-`apply_calibration`) sample. Dedupes repeated
    /// deliveries of the same capture frame internally.
    pub fn ingest(&mut self, sample: &SourceSkeleton) {
        // Metric-only: every neutral this module estimates is defined on
        // the metric path (yaw needs real z, the anchor is metres).
        if sample.metric_frame_info.is_none() {
            return;
        }
        let key = (
            sample.source_timestamp,
            sample.capture_timestamp_ms.map(f64::to_bits).unwrap_or(0),
        );
        if self.last_key == Some(key) {
            return;
        }
        self.last_key = Some(key);

        if let Some(root) = sample.root_offset {
            // Anchor-type consistency: a hip↔shoulder flip resets the
            // positional windows (different anchors, ~0.4 m apart).
            if self.anchor_is_hip != Some(sample.root_anchor_is_hip) {
                if self.cached.is_none() {
                    self.anchor_x = ScalarChannel::default();
                    self.anchor_y = ScalarChannel::default();
                    self.anchor_z = ScalarChannel::default();
                    self.confidence = ScalarChannel::default();
                    self.anchor_is_hip = Some(sample.root_anchor_is_hip);
                }
            }
            if self.anchor_is_hip == Some(sample.root_anchor_is_hip) {
                self.anchor_x.push(root[0], ANCHOR_MAX_MAD_M, None);
                self.anchor_y.push(root[1], ANCHOR_MAX_MAD_M, None);
                self.anchor_z.push(root[2], ANCHOR_MAX_MAD_M, None);
                self.confidence.push(sample.overall_confidence, f32::INFINITY, None);
            }
        }

        if let Some(yaw) = shoulder_line_yaw(sample) {
            self.body_yaw.push(yaw, MAX_MAD_RAD, Some(BODY_YAW_MAX_RAD));
        }

        // Face neutrals, per source — mirroring the modal capture:
        // mesh from the published pose on steady (non-crossfade) mesh
        // frames, body from the always-published raw body candidate.
        if let Some(face) = sample.face {
            if face.source == FaceSource::Mesh
                && face.blend.is_none()
                && face.confidence >= FACE_SAMPLE_MIN_CONF
            {
                self.face_mesh.push(
                    [face.yaw, face.pitch, face.roll],
                    // Roll wanders least and is the least damaging to
                    // get slightly wrong — same waiver as the modal.
                    [MAX_MAD_RAD, MAX_MAD_RAD, f32::INFINITY],
                );
            }
        }
        if let Some(body) = sample.face_body_raw {
            if body.confidence >= FACE_SAMPLE_MIN_CONF {
                self.face_body.push(
                    [body.yaw, body.pitch, body.roll],
                    [MAX_MAD_RAD, MAX_MAD_RAD, f32::INFINITY],
                );
            }
        }

        self.rebuild_cache();
    }

    fn rebuild_cache(&mut self) {
        // The anchor is the base: without it there is no mode, no root
        // seed and no yaw pivot, and the other neutrals are pure no-ops
        // in `apply_calibration` anyway.
        let (Some(ax), Some(ay), Some(az), Some(is_hip)) = (
            self.anchor_x.seeded,
            self.anchor_y.seeded,
            self.anchor_z.seeded,
            self.anchor_is_hip,
        ) else {
            return;
        };
        let next = PoseCalibration {
            mode: if is_hip {
                CalibrationMode::FullBody
            } else {
                CalibrationMode::UpperBody
            },
            captured_at: "auto".to_string(),
            captured_at_unix: 0,
            frame_count: WINDOW,
            anchor_x: ax,
            anchor_y: ay,
            // Source z is negative-forward; the field is positive
            // camera-forward metres (same convention as the modal).
            anchor_depth_m: Some(az.abs()),
            confidence: self.confidence.seeded.unwrap_or(0.0),
            anchor_depth_jitter_m: None,
            shoulder_span_m: None,
            x_range_observed: None,
            z_range_observed: None,
            torso_depth_template: None,
            neutral_expressions: Vec::new(),
            neutral_face_ypr_mesh: self.face_mesh.seeded,
            neutral_face_ypr_body: self.face_body.seeded,
            neutral_body_yaw: self.body_yaw.seeded,
        };
        // One log line per newly seeded channel — the only user-visible
        // trace of the automatic path.
        let channels = u8::from(next.neutral_body_yaw.is_some())
            | u8::from(next.neutral_face_ypr_mesh.is_some()) << 1
            | u8::from(next.neutral_face_ypr_body.is_some()) << 2
            | 1 << 3; // anchor
        if channels != self.logged_channels {
            self.logged_channels = channels;
            info!(
                "auto-neutral seeded: anchor=[{:+.3},{:+.3},{:+.3}] ({}) body_yaw={} face_mesh={} face_body={}",
                ax,
                ay,
                az,
                if is_hip { "hip" } else { "shoulder" },
                next.neutral_body_yaw
                    .map(|y| format!("{:+.1}°", y.to_degrees()))
                    .unwrap_or_else(|| "-".into()),
                next.neutral_face_ypr_mesh
                    .map(|y| format!("{:+.1}°", y[0].to_degrees()))
                    .unwrap_or_else(|| "-".into()),
                next.neutral_face_ypr_body
                    .map(|y| format!("{:+.1}°", y[0].to_degrees()))
                    .unwrap_or_else(|| "-".into()),
            );
        }
        self.cached = Some(next);
    }
}

/// In-place median (same contract as the modal finalize helper).
fn median_inplace(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::HumanoidBone;
    use crate::tracking::source_skeleton::{
        CameraIntrinsics, FacePose, MetricFrameInfo, SourceJoint,
    };

    fn metric_sample(n: u64, root: [f32; 3], yaw_z: f32) -> SourceSkeleton {
        let mut sk = SourceSkeleton::empty(n);
        sk.capture_timestamp_ms = Some(n as f64 * 33.3);
        sk.root_offset = Some(root);
        sk.root_anchor_is_hip = false;
        sk.overall_confidence = 0.8;
        sk.metric_frame_info = Some(MetricFrameInfo {
            anchor_cam_m: [-root[0], -root[1], -root[2]],
            anchor_is_hip: false,
            mpsu: 1.0,
            reference_span_m: 0.38,
            intrinsics: CameraIntrinsics {
                fx: 900.0,
                fy: 900.0,
                cx: 640.0,
                cy: 360.0,
                width: 1280,
                height: 720,
            },
        });
        // Shoulder line with a z separation → shoulder_line_yaw reads a
        // stable oblique heading.
        let j = |p: [f32; 3]| SourceJoint { position: p, confidence: 0.9, metric_depth_m: None };
        sk.joints.insert(HumanoidBone::LeftUpperArm, j([0.19, 0.0, yaw_z]));
        sk.joints.insert(HumanoidBone::RightUpperArm, j([-0.19, 0.0, -yaw_z]));
        sk.face_body_raw = Some(FacePose {
            yaw: -0.7,
            pitch: 0.05,
            roll: 0.0,
            confidence: 0.9,
            ..Default::default()
        });
        sk
    }

    #[test]
    fn stable_stream_seeds_an_auto_calibration() {
        let mut auto = AutoNeutral::default();
        for n in 0..(WINDOW as u64 + 5) {
            auto.ingest(&metric_sample(n, [0.02, -0.15, -0.62], 0.11));
        }
        let cal = auto.calibration().expect("stable window must seed");
        assert_eq!(cal.mode, CalibrationMode::UpperBody);
        assert!((cal.anchor_x - 0.02).abs() < 1e-4);
        assert!((cal.anchor_depth_m.unwrap() - 0.62).abs() < 1e-4);
        let yaw = cal.neutral_body_yaw.expect("shoulder yaw must seed");
        assert!(yaw.abs() > 0.2, "oblique shoulder line must produce a non-zero neutral, got {yaw}");
        let body = cal.neutral_face_ypr_body.expect("body face neutral must seed");
        assert!((body[0] - -0.7).abs() < 1e-4);
        assert_eq!(cal.captured_at, "auto");
    }

    #[test]
    fn moving_subject_does_not_seed() {
        // A subject wandering ±10 cm never presents a still window.
        let mut auto = AutoNeutral::default();
        for n in 0..(WINDOW as u64 * 3) {
            let sway = if n % 2 == 0 { 0.1 } else { -0.1 };
            auto.ingest(&metric_sample(n, [sway, -0.15, -0.62], 0.11));
        }
        assert!(auto.calibration().is_none(), "unstable anchor must not seed");
    }

    #[test]
    fn seeded_values_freeze_for_the_session() {
        let mut auto = AutoNeutral::default();
        for n in 0..(WINDOW as u64 + 5) {
            auto.ingest(&metric_sample(n, [0.02, -0.15, -0.62], 0.11));
        }
        let before = auto.calibration().unwrap().anchor_x;
        // The user then leans somewhere else and stays there — the
        // session neutral must NOT chase them (full-mirror philosophy).
        for n in 1000..(1000 + WINDOW as u64 * 2) {
            auto.ingest(&metric_sample(n, [0.30, -0.15, -0.62], 0.11));
        }
        assert_eq!(auto.calibration().unwrap().anchor_x, before);
    }

    #[test]
    fn repeated_delivery_of_one_frame_counts_once() {
        // The render loop re-delivers the same capture frame at display
        // rate; a 150-sample window must represent 150 CAMERA frames.
        let mut auto = AutoNeutral::default();
        let sample = metric_sample(7, [0.02, -0.15, -0.62], 0.11);
        for _ in 0..(WINDOW * 4) {
            auto.ingest(&sample);
        }
        assert!(auto.calibration().is_none(), "duplicates must not fill the window");
    }

    #[test]
    fn anchor_type_flip_resets_the_positional_window() {
        let mut auto = AutoNeutral::default();
        for n in 0..(WINDOW as u64 - 10) {
            auto.ingest(&metric_sample(n, [0.02, -0.15, -0.62], 0.11));
        }
        // Hips appear (mode flip) just before the window would fill —
        // the shoulder-anchored positions must not be medianed with
        // hip-anchored ones.
        let mut hips = metric_sample(WINDOW as u64, [0.02, -0.55, -0.62], 0.11);
        hips.root_anchor_is_hip = true;
        auto.ingest(&hips);
        assert!(auto.calibration().is_none());
        // Continuing in hip mode seeds the hip anchor cleanly.
        for n in 0..(WINDOW as u64 + 5) {
            let mut s = metric_sample(2000 + n, [0.02, -0.55, -0.62], 0.11);
            s.root_anchor_is_hip = true;
            auto.ingest(&s);
        }
        let cal = auto.calibration().expect("hip window must seed");
        assert_eq!(cal.mode, CalibrationMode::FullBody);
        assert!((cal.anchor_y - -0.55).abs() < 1e-4);
    }
}
