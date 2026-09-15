//! Display-rate rig-pose interpolation.
//!
//! The fusion estimator publishes at its own compute-bound cadence
//! (~20-30 Hz, `rig.diag.solve_ms`-bound), but `run_frame` renders at the
//! display refresh (60+ Hz). Replaying the newest [`RigPose`] as-is turned
//! the avatar motion into a staircase: frozen for the full inter-sample
//! interval, then a multi-millimetre jump the frame the next sample
//! landed (measured live 2026-09-15: 20.1 Hz solves, inter-frame deltas
//! p95 3-5 mm body / up to 36 mm head, between updates ~0.4 mm).
//!
//! [`PoseTimeline`] keeps the last few raw samples and answers
//! [`PoseTimeline::sample_at`] with a slerped [`RigPose`] positioned
//! `delay_frac` of an inter-sample interval behind the newest one. With
//! the default `delay_frac = 0.5` the interpolator renders half an
//! interval in the past and extrapolates (bounded by `max_extrap`)
//! across the remainder, so steady-state motion stays continuous while
//! added latency stays at half the sample interval (~25 ms at 20 Hz).
//! When the extrapolation budget runs out (tracking stall) the pose
//! freezes — the same hold behaviour the un-interpolated path had.
//!
//! Everything runs in the quaternion domain (`slerp_short`, which
//! extrapolates for `t > 1`); Euler angles are never involved.
//! The retarget's 1€ filters then see a genuinely continuous signal —
//! which also dissolves the old "filter stepped at render rate against
//! an unchanged sample" quirk, because the input now changes every frame.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use super::retarget::slerp_short;
use crate::asset::HumanoidBone;
use crate::tracking::fusion::output::{RigBone, RigPose};

/// Samples retained for interpolation. Two are used; the rest are
/// spill head-room so a late consumer never sees an empty buffer.
const MAX_SAMPLES: usize = 4;
/// Nominal prior for the very first sample's interval (no previous
/// arrival to measure against).
const NOMINAL_INTERVAL_S: f32 = 1.0 / 30.0;
/// Inter-sample intervals outside this window are treated as garbage
/// (burst catch-up, inference stall, device clock resync) and replaced
/// with the wall-clock arrival delta.
const MIN_CAPTURE_INTERVAL_S: f32 = 0.001;
const MAX_CAPTURE_INTERVAL_S: f32 = 0.25;
/// Clamp floor for the interval the interpolation clock divides by —
/// keeps `alpha` finite even when ingest saw a degenerate cadence.
const MIN_INTERVAL_S: f32 = 0.005;

/// Kill-switch for benches / A-B: `VULVATAR_POSE_INTERP=0` makes
/// [`PoseTimeline::sample_at`] return the newest raw sample unchanged
/// (the pre-interpolation behaviour). Read once per process.
fn interp_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var("VULVATAR_POSE_INTERP").as_deref() == Ok("0"))
}

/// Tuning for [`PoseTimeline::sample_at`].
#[derive(Clone, Copy, Debug)]
pub struct PoseInterpParams {
    /// Interpolation delay as a fraction of the latest inter-sample
    /// interval. `0.5` renders half an interval behind the newest sample
    /// and extrapolates across the rest; `1.0` is pure interpolation
    /// (never extrapolates in steady state, one full interval of latency).
    pub delay_frac: f32,
    /// Extrapolation cap past the newest sample, in the same units
    /// (fractions of an interval). Also the freeze point when tracking
    /// stalls: `alpha` pins at `1 + max_extrap` until the next sample.
    pub max_extrap: f32,
}

impl Default for PoseInterpParams {
    fn default() -> Self {
        Self {
            delay_frac: 0.5,
            max_extrap: 0.5,
        }
    }
}

#[derive(Debug)]
struct TimedRig {
    rig: Arc<RigPose>,
    /// Capture time of this sample (device-clock seconds; `0` when the
    /// producer doesn't stamp it — synthetic benches). Also the dedup key.
    capture_t: f64,
    /// Interval to the previous sample (seconds), measured on the capture
    /// clock when plausible, else on the arrival clock. This is the unit
    /// the interpolation clock advances in.
    interval: f32,
    /// Monotonic arrival stamp at the consumer (≈ publish time + one
    /// render frame of poll latency; the fixed offset cancels in alpha).
    arrived: Instant,
}

/// Ring of recent rig samples + the interpolator over them. Owned by the
/// [`Application`](crate::app::Application), fed from `run_frame`'s fresh
/// tracking branch, one interpolation per frame shared by all avatars.
#[derive(Debug, Default)]
pub struct PoseTimeline {
    samples: VecDeque<TimedRig>,
}

impl PoseTimeline {
    /// Record a freshly published sample. Re-pulls of the same sample
    /// (the mailbox hands out the newest value until the worker publishes
    /// again) are recognised by their capture time and dropped; samples
    /// without a capture stamp (`t == 0`, synthetic benches) are always
    /// taken because a fresh mailbox pull is then the only novelty signal.
    pub fn ingest(&mut self, rig: &Arc<RigPose>) {
        self.ingest_at(rig, Instant::now());
    }

    /// [`Self::ingest`] with an explicit arrival clock (tests).
    pub fn ingest_at(&mut self, rig: &Arc<RigPose>, arrived: Instant) {
        let capture_t = rig.t;
        if let Some(prev) = self.samples.back() {
            if prev.capture_t > 0.0 && capture_t == prev.capture_t {
                return;
            }
        }
        let interval = match self.samples.back() {
            None => NOMINAL_INTERVAL_S,
            Some(prev) => {
                let capture_dt = if prev.capture_t > 0.0 && capture_t > prev.capture_t {
                    Some((capture_t - prev.capture_t) as f32)
                } else {
                    None
                };
                let arrival_dt = now_delta(prev.arrived, arrived);
                match capture_dt {
                    Some(dt) if dt > MIN_CAPTURE_INTERVAL_S && dt <= MAX_CAPTURE_INTERVAL_S => dt,
                    _ => arrival_dt.clamp(MIN_CAPTURE_INTERVAL_S, MAX_CAPTURE_INTERVAL_S),
                }
            }
        };
        self.samples.push_back(TimedRig {
            rig: Arc::clone(rig),
            capture_t,
            interval,
            arrived,
        });
        while self.samples.len() > MAX_SAMPLES {
            self.samples.pop_front();
        }
    }

    /// The rig the avatar should render at wall-clock time `now`:
    /// a slerp/lerp between the bracketing samples positioned
    /// `delay_frac · interval` behind the newest one, extrapolated up to
    /// `max_extrap` intervals past it and frozen beyond that. With fewer
    /// than two samples (or the `VULVATAR_POSE_INTERP=0` kill-switch)
    /// this is the newest raw sample unchanged.
    pub fn sample_at(&self, params: PoseInterpParams, now: Instant) -> RigPose {
        let Some(latest) = self.samples.back() else {
            return RigPose::default();
        };
        if self.samples.len() < 2 || interp_disabled() {
            return latest.rig.as_ref().clone();
        }
        let prev = &self.samples[self.samples.len() - 2];
        let interval = latest.interval.max(MIN_INTERVAL_S);
        let delay = params.delay_frac.clamp(0.0, 1.0) * interval;
        let age = now.saturating_duration_since(latest.arrived).as_secs_f32();
        let alpha = 1.0 + (age - delay) / interval;
        // Extrapolation cap ≙ tracking-stall freeze: past the budget the
        // pose holds still (what the un-interpolated path did all along)
        // instead of flying off on stale velocity.
        let alpha = alpha.clamp(0.0, 1.0 + params.max_extrap.clamp(0.0, 1.0));
        interpolate_rigs(&prev.rig, &latest.rig, alpha)
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.samples.len()
    }
}

fn now_delta(from: Instant, to: Instant) -> f32 {
    to.saturating_duration_since(from).as_secs_f32()
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]
}

fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

/// Blend two rig samples. Rotations take the shortest arc and may
/// extrapolate (`alpha > 1`); every scalar channels lerp the same way,
/// with `[0,1]` clamps where the value is a confidence/quality.
fn interpolate_rigs(prev: &RigPose, latest: &RigPose, alpha: f32) -> RigPose {
    let mut bones: HashMap<HumanoidBone, RigBone> = HashMap::with_capacity(latest.bones.len());
    for (bone, b) in &latest.bones {
        let blended = match prev.bones.get(bone) {
            Some(a) => RigBone {
                delta_world: slerp_short(&a.delta_world, &b.delta_world, alpha),
                sigma: lerp(a.sigma, b.sigma, alpha),
                data_sigma: lerp(a.data_sigma, b.data_sigma, alpha),
            },
            // Bone absent from the older sample (gate churn, shape still
            // warming up): no arc to walk — the newer value wins and the
            // retarget's own σ gates smooth the transition.
            None => *b,
        };
        bones.insert(*bone, blended);
    }
    RigPose {
        t: prev.t + (latest.t - prev.t) * alpha as f64,
        bones,
        root_cam_m: lerp3(prev.root_cam_m, latest.root_cam_m, alpha),
        root_sigma_m: lerp(prev.root_sigma_m, latest.root_sigma_m, alpha),
        quality: clamp01(lerp(prev.quality, latest.quality, alpha)),
        shape_confidence: clamp01(lerp(prev.shape_confidence, latest.shape_confidence, alpha)),
        hand_confidence: [
            clamp01(lerp(prev.hand_confidence[0], latest.hand_confidence[0], alpha)),
            clamp01(lerp(prev.hand_confidence[1], latest.hand_confidence[1], alpha)),
        ],
        shoulder_span_m: lerp(prev.shoulder_span_m, latest.shoulder_span_m, alpha),
        head_local_wrists: [
            lerp3(prev.head_local_wrists[0], latest.head_local_wrists[0], alpha),
            lerp3(prev.head_local_wrists[1], latest.head_local_wrists[1], alpha),
        ],
        face_proximity: [
            clamp01(lerp(prev.face_proximity[0], latest.face_proximity[0], alpha)),
            clamp01(lerp(prev.face_proximity[1], latest.face_proximity[1], alpha)),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qy(rad: f32) -> [f32; 4] {
        [0.0, (rad / 2.0).sin(), 0.0, (rad / 2.0).cos()]
    }

    fn quat_angle_y(q: [f32; 4]) -> f32 {
        let ang = 2.0 * q[3].clamp(-1.0, 1.0).acos();
        let ang = if ang > std::f32::consts::PI {
            2.0 * std::f32::consts::PI - ang
        } else {
            ang
        };
        // Signed: the quaternion's Y component disambiguates the direction.
        if q[1] >= 0.0 { ang } else { -ang }
    }

    fn rig(bones: &[(HumanoidBone, f32)], quality: f32, t: f64) -> Arc<RigPose> {
        let mut r = RigPose {
            quality,
            t,
            root_cam_m: [t as f32, 0.0, 0.0],
            ..Default::default()
        };
        for (bone, yaw) in bones {
            r.bones.insert(
                *bone,
                RigBone {
                    delta_world: qy(*yaw),
                    sigma: 0.05,
                    data_sigma: 0.05,
                },
            );
        }
        Arc::new(r)
    }

    /// Two samples 50 ms apart on the capture clock: rig A at `base-100ms`
    /// (Hips yaw 0), rig B at `base-50ms` (Hips yaw 0.5). Default params
    /// (delay 25 ms, cap 1.5).
    fn two_sample_timeline(base: Instant) -> PoseTimeline {
        let mut tl = PoseTimeline::default();
        tl.ingest_at(&rig(&[(HumanoidBone::Hips, 0.0)], 0.4, 1.00), base - std::time::Duration::from_millis(100));
        tl.ingest_at(&rig(&[(HumanoidBone::Hips, 0.5)], 0.8, 1.05), base - std::time::Duration::from_millis(50));
        tl
    }

    #[test]
    fn midpoint_brackets_interpolate_halfway() {
        let base = Instant::now();
        let tl = two_sample_timeline(base);
        let out = tl.sample_at(
            PoseInterpParams::default(),
            base - std::time::Duration::from_millis(50),
        );
        // age = 0 → alpha = 1 - 25/50 = 0.5 → exactly halfway.
        let hips = out.bones[&HumanoidBone::Hips].delta_world;
        assert!((quat_angle_y(hips) - 0.25).abs() < 1e-3, "got {}", quat_angle_y(hips));
        // Scalars ride the same alpha.
        assert!((out.quality - 0.6).abs() < 1e-5);
        assert!((out.root_cam_m[0] - 1.025).abs() < 1e-4);
        assert!((out.t - 1.025).abs() < 1e-9);
    }

    #[test]
    fn full_delay_renders_the_newest_sample_then_walks_back() {
        let base = Instant::now();
        let tl = two_sample_timeline(base);
        let p = PoseInterpParams {
            delay_frac: 1.0,
            max_extrap: 0.5,
        };
        // age = 0, delay = 50 ms → alpha = 0 → exactly the previous sample.
        let out = tl.sample_at(p, base - std::time::Duration::from_millis(50));
        assert!((quat_angle_y(out.bones[&HumanoidBone::Hips].delta_world)).abs() < 1e-4);
        // age = 40 ms → alpha = 1 + (40-50)/50 = 0.8.
        let out = tl.sample_at(p, base - std::time::Duration::from_millis(10));
        assert!((quat_angle_y(out.bones[&HumanoidBone::Hips].delta_world) - 0.4).abs() < 1e-3);
    }

    #[test]
    fn stall_pins_alpha_at_the_extrapolation_cap_and_freezes() {
        let base = Instant::now();
        let tl = two_sample_timeline(base);
        let far = base + std::time::Duration::from_secs(10);
        let a = tl.sample_at(PoseInterpParams::default(), far);
        let b = tl.sample_at(PoseInterpParams::default(), far + std::time::Duration::from_secs(10));
        let qa = a.bones[&HumanoidBone::Hips].delta_world;
        let qb = b.bones[&HumanoidBone::Hips].delta_world;
        // alpha pinned at 1.5: yaw extrapolates to 0.75 and then holds.
        assert!((quat_angle_y(qa) - 0.75).abs() < 1e-3, "got {}", quat_angle_y(qa));
        assert_eq!(qa, qb, "stalled timeline must freeze, not keep drifting");
    }

    #[test]
    fn extrapolated_alpha_stays_continuous_across_a_sample_arrival() {
        // Constant-velocity motion (0.5 rad per 50 ms sample): the pose just
        // before the next sample lands (alpha = 1.5, extrapolated) must
        // match the pose just after (alpha = 0.5 of the new pair) — that is
        // the property that makes the 20 Hz staircase invisible.
        let base = Instant::now();
        let mut tl = PoseTimeline::default();
        tl.ingest_at(&rig(&[(HumanoidBone::Hips, 0.0)], 1.0, 1.00), base - std::time::Duration::from_millis(100));
        tl.ingest_at(&rig(&[(HumanoidBone::Hips, 0.5)], 1.0, 1.05), base - std::time::Duration::from_millis(50));
        // Exactly at the arrival instant the old pair sits at the
        // extrapolation cap (alpha = 1.5).
        let before = tl.sample_at(PoseInterpParams::default(), base);
        tl.ingest_at(&rig(&[(HumanoidBone::Hips, 1.0)], 1.0, 1.10), base);
        // The new pair starts at alpha = 0.5. Constant velocity: same pose.
        let after = tl.sample_at(PoseInterpParams::default(), base);
        let a = quat_angle_y(before.bones[&HumanoidBone::Hips].delta_world);
        let b = quat_angle_y(after.bones[&HumanoidBone::Hips].delta_world);
        assert!((a - b).abs() < 1e-3, "discontinuity at sample arrival: {a} vs {b}");
    }

    #[test]
    fn bone_missing_from_the_older_sample_takes_the_newer_value() {
        let base = Instant::now();
        let mut tl = PoseTimeline::default();
        tl.ingest_at(&rig(&[(HumanoidBone::Hips, 0.0)], 1.0, 1.00), base - std::time::Duration::from_millis(100));
        tl.ingest_at(
            &rig(&[(HumanoidBone::Hips, 0.5), (HumanoidBone::Spine, 0.9)], 1.0, 1.05),
            base - std::time::Duration::from_millis(50),
        );
        let out = tl.sample_at(PoseInterpParams::default(), base - std::time::Duration::from_millis(50));
        let spine = out.bones[&HumanoidBone::Spine].delta_world;
        assert!((quat_angle_y(spine) - 0.9).abs() < 1e-4);
        assert!(out.bones[&HumanoidBone::Hips].delta_world.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn synthetic_zero_capture_clock_dedups_on_nothing_and_falls_back_to_arrival_intervals() {
        let base = Instant::now();
        let mut tl = PoseTimeline::default();
        // All samples share capture_t == 0: nothing to dedup on, and the
        // interval comes from the arrival clock.
        tl.ingest_at(&rig(&[(HumanoidBone::Hips, 0.0)], 1.0, 0.0), base - std::time::Duration::from_millis(100));
        tl.ingest_at(&rig(&[(HumanoidBone::Hips, 0.5)], 1.0, 0.0), base - std::time::Duration::from_millis(50));
        assert_eq!(tl.len(), 2);
        tl.ingest_at(&rig(&[(HumanoidBone::Hips, 1.0)], 1.0, 0.0), base - std::time::Duration::from_millis(10));
        assert_eq!(tl.len(), 3);
        // Latest interval = 40 ms (arrival), delay = 20 ms, age = 0 →
        // alpha = 1 - 20/40 = 0.5 of the last pair → 0.75.
        let out = tl.sample_at(PoseInterpParams::default(), base - std::time::Duration::from_millis(10));
        assert!((quat_angle_y(out.bones[&HumanoidBone::Hips].delta_world) - 0.75).abs() < 1e-3);
    }

    #[test]
    fn stamped_capture_clock_dedups_republished_samples() {
        let base = Instant::now();
        let mut tl = PoseTimeline::default();
        tl.ingest_at(&rig(&[(HumanoidBone::Hips, 0.0)], 1.0, 1.00), base - std::time::Duration::from_millis(100));
        tl.ingest_at(&rig(&[(HumanoidBone::Hips, 0.5)], 1.0, 1.05), base - std::time::Duration::from_millis(50));
        // Same capture time re-pulled from the mailbox: dropped.
        tl.ingest_at(&rig(&[(HumanoidBone::Hips, 9.9)], 1.0, 1.05), base - std::time::Duration::from_millis(1));
        assert_eq!(tl.len(), 2);
    }

    #[test]
    fn degenerate_sample_cadence_uses_a_floored_interval() {
        // Two samples 1 ms apart on the capture clock (burst catch-up):
        // below MIN_CAPTURE_INTERVAL_S, so the arrival delta (50 ms) wins
        // and alpha stays finite and sane.
        let base = Instant::now();
        let mut tl = PoseTimeline::default();
        tl.ingest_at(&rig(&[(HumanoidBone::Hips, 0.0)], 1.0, 1.000), base - std::time::Duration::from_millis(100));
        tl.ingest_at(&rig(&[(HumanoidBone::Hips, 0.5)], 1.0, 1.001), base - std::time::Duration::from_millis(50));
        let out = tl.sample_at(PoseInterpParams::default(), base - std::time::Duration::from_millis(50));
        let hips = out.bones[&HumanoidBone::Hips].delta_world;
        assert!((quat_angle_y(hips) - 0.25).abs() < 1e-3, "got {}", quat_angle_y(hips));
    }

    #[test]
    fn fewer_than_two_samples_passes_through() {
        let base = Instant::now();
        let empty = PoseTimeline::default();
        assert_eq!(empty.sample_at(PoseInterpParams::default(), base).quality, 0.0);

        let mut tl = PoseTimeline::default();
        let only = rig(&[(HumanoidBone::Hips, 0.3)], 0.7, 2.0);
        tl.ingest_at(&only, base);
        let out = tl.sample_at(PoseInterpParams::default(), base + std::time::Duration::from_secs(5));
        assert!((out.quality - 0.7).abs() < 1e-6);
        assert!((quat_angle_y(out.bones[&HumanoidBone::Hips].delta_world) - 0.3).abs() < 1e-4);
    }
}
