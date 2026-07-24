//! Live pose-vs-target scoring. Drives the `WaitingForPose →
//! Collecting` gate: the per-frame match score is computed in
//! [`super::refresh::refresh_anchor_telemetry`], and once the score
//! holds above [`POSE_MATCH_THRESHOLD`] for [`REQUIRED_STABLE_FRAMES`]
//! consecutive frames, [`super::transitions::advance_state`] flips
//! the state machine forward.

use crate::tracking::{CalibrationMode, SourceSkeleton};

/// Live pose-match score above which a frame counts toward the
/// "stable in pose" tally. 0.6 is calibrated against the squared
/// cosine-similarity scoring in [`pose_match_score`] — both arms in
/// roughly the right direction (≈ ±35° from target) clear this. Lower
/// would let near-misses trip capture; higher demands a cleaner pose
/// than most users can hold at the first try.
pub(super) const POSE_MATCH_THRESHOLD: f32 = 0.6;

/// Number of consecutive matched frames required before transitioning
/// from `WaitingForPose` to `Collecting`. At 30 fps this is ≈ 0.5 s of
/// stable above-threshold match — enough to filter out a single
/// wobbly frame the user happened to be in T-pose during, not so long
/// that the user feels the system is laggy after they pose correctly.
pub(super) const REQUIRED_STABLE_FRAMES: u32 = 15;

/// Per-tracking-frame anchor displacement (source units) below which
/// a frame counts as "still" for the bust-up stillness fallback gate
/// (see `docs/calibration-ux.md`, Phase H). Source space is
/// `y ∈ [-1, 1]`, so 0.010 is 0.5 % of frame height per frame —
/// generous enough for breathing and keypoint jitter at desk
/// distance, tight enough that leaning or reaching resets the hold.
/// Must be evaluated against per-*mailbox-sequence* deltas, not
/// per-repaint deltas: the GUI repaints faster than the tracker
/// emits frames and per-repaint displacement reads as near-zero at
/// any real motion speed.
pub(super) const STILLNESS_EPS: f32 = 0.010;

/// Stillness score in `[0.0, 1.0]` for the bust-up fallback gate:
/// `1 − displacement / STILLNESS_EPS`, clamped. `> 0.0` means the
/// frame counts toward the stable-hold tally; `0.0` (displacement at
/// or beyond the epsilon) resets it — mirroring how the
/// arm-direction gate treats sub-threshold match scores. The linear
/// ramp doubles as the progress-bar fill so the bar keeps its
/// "closer to ready" meaning while the fallback is engaged.
pub(super) fn stillness_score(prev: [f32; 2], curr: [f32; 2]) -> f32 {
    let dx = curr[0] - prev[0];
    let dy = curr[1] - prev[1];
    let disp = (dx * dx + dy * dy).sqrt();
    if !disp.is_finite() {
        return 0.0;
    }
    (1.0 - disp / STILLNESS_EPS).clamp(0.0, 1.0)
}

/// Score in `[0.0, 1.0]` measuring how well the user's live pose
/// matches the calibration mode's *target* pose:
///
/// - `FullBody`  → T-pose: arms extended horizontally to the sides.
/// - `UpperBody` → arms relaxed at the sides (= sitting/standing
///   neutral). Works for desk-sitting users who can't physically
///   T-pose at their current framing.
///
/// Computed as the geometric mean of the two arm-direction matches.
/// Each arm produces a unit-vector from shoulder→elbow (lower-arm
/// joint, in image space) and is compared to the target direction
/// for that arm via squared cosine similarity. Returns `0.0` when
/// either shoulder or elbow is missing / low-confidence — caller
/// treats that as "not in pose yet" and waits.
///
/// Convention reminder (`source_skeleton` module docs): image space
/// is `+x = mirrored-right`, `+y = up`. T-pose right arm therefore
/// has the (subject's) **right** elbow at large negative x relative
/// to the shoulder; due to the selfie-mirror this is stored on the
/// `LeftShoulder` / `LeftLowerArm` bones. Be careful when reading
/// this code — "left" / "right" here are *avatar bone names*, not
/// the subject's anatomical sides.
pub(super) fn pose_match_score(pose: &SourceSkeleton, mode: CalibrationMode) -> f32 {
    // Target arm directions in image-space `(x, y)` (y-up).
    //
    // - T-pose: avatar `LeftShoulder` → `LeftLowerArm` extends to the
    //   subject's anatomical right (= mirrored-left on screen = +x in
    //   the source-space convention). Avatar `RightShoulder` →
    //   `RightLowerArm` extends to subject's anatomical left = −x.
    //   Both with y ≈ 0 (horizontal).
    // - UpperBody (hands at sides): both arms point straight down
    //   (= negative y in y-up source space).
    let (left_target, right_target) = match mode {
        CalibrationMode::FullBody => ([1.0_f32, 0.0_f32], [-1.0_f32, 0.0_f32]),
        CalibrationMode::UpperBody => ([0.0_f32, -1.0_f32], [0.0_f32, -1.0_f32]),
    };

    let l = arm_direction(pose, true);
    let r = arm_direction(pose, false);
    let (Some(l), Some(r)) = (l, r) else {
        return 0.0;
    };
    let l_score = direction_match(l, left_target);
    let r_score = direction_match(r, right_target);
    // Geometric mean — both arms have to be in pose, not just one.
    // sqrt(0 * x) = 0 so a single bad arm tanks the whole score, which
    // is the desired behaviour: don't trip capture if only one arm
    // is in T-pose.
    (l_score * r_score).sqrt()
}

/// Unit-vector from shoulder to lower-arm (= elbow position) in
/// image-space `(x, y)`. Returns `None` when either keypoint is
/// missing or carries < 0.3 confidence (= keypoint detector is
/// unsure — using its position would dilute the match score with
/// noise).
fn arm_direction(pose: &SourceSkeleton, avatar_left: bool) -> Option<[f32; 2]> {
    use crate::asset::HumanoidBone;
    let (shoulder_bone, elbow_bone) = if avatar_left {
        (HumanoidBone::LeftShoulder, HumanoidBone::LeftLowerArm)
    } else {
        (HumanoidBone::RightShoulder, HumanoidBone::RightLowerArm)
    };
    let s = pose.joints.get(&shoulder_bone)?;
    let e = pose.joints.get(&elbow_bone)?;
    if s.confidence < 0.3 || e.confidence < 0.3 {
        return None;
    }
    let dx = e.position[0] - s.position[0];
    let dy = e.position[1] - s.position[1];
    let len = (dx * dx + dy * dy).sqrt();
    if !len.is_finite() || len < 1e-3 {
        return None;
    }
    Some([dx / len, dy / len])
}

/// Squared-cosine direction similarity, mapped to `[0.0, 1.0]`.
/// `1.0` when `actual == target`, `0.0` when opposite. Squaring
/// sharpens the threshold so middling matches (≈ 60° off) score
/// noticeably below near-matches (≈ 20° off) — clearer "closer to
/// the right pose" signal on the progress bar.
fn direction_match(actual: [f32; 2], target: [f32; 2]) -> f32 {
    let dot = actual[0] * target[0] + actual[1] * target[1];
    (((dot + 1.0) * 0.5).clamp(0.0, 1.0)).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stillness_score_perfect_hold_is_one() {
        assert_eq!(stillness_score([0.2, -0.5], [0.2, -0.5]), 1.0);
    }

    #[test]
    fn stillness_score_at_epsilon_is_zero() {
        // Displacement exactly STILLNESS_EPS must reset the hold —
        // the gate condition is `score > 0.0`.
        assert_eq!(stillness_score([0.0, 0.0], [STILLNESS_EPS, 0.0]), 0.0);
        assert_eq!(stillness_score([0.0, 0.0], [0.0, STILLNESS_EPS * 3.0]), 0.0);
    }

    #[test]
    fn stillness_score_sub_epsilon_jitter_counts() {
        // Half-epsilon jitter (breathing / keypoint noise) must keep
        // the hold alive with a midway progress-bar fill.
        let s = stillness_score([0.0, 0.0], [STILLNESS_EPS * 0.5, 0.0]);
        assert!((s - 0.5).abs() < 1e-6, "got {s}");
    }

    #[test]
    fn stillness_score_non_finite_resets() {
        assert_eq!(stillness_score([f32::NAN, 0.0], [0.0, 0.0]), 0.0);
    }
}
