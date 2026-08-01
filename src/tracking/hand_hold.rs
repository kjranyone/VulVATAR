//! Short-term hold for the hand / forearm source joints.
//!
//! The depth builder emits a hand only when at least 3 of the 4 MCP knuckles
//! yield a person-aware depth sample (`attach_hand`). That is the right test
//! for *acquiring* a hand, but as a per-frame existence test it makes the
//! joint blink: live capture (2026-07-28, 109 s of ordinary desk work)
//! recorded the source `RightHand` appearing and disappearing at ~15 Hz while
//! its own 2-D keypoints held to ±2 mm and the raw depth under them was 49/49
//! valid pixels. Downstream, every blink was a full-amplitude pose change —
//! the avatar's hands teleported 0.45-0.60 m in a single frame, 34 times in
//! that capture, which is the "arms are all over the place" the user sees.
//!
//! A momentary sampling miss is missing data, not a hand that ceased to
//! exist. This module keeps publishing the last observed joint for a short
//! window, with a confidence that decays to zero across it, so:
//!
//! * a 1-3 frame dropout is invisible — the arm keeps its observed pose;
//! * a genuine exit (hand leaves frame and stays out) still releases, and
//!   releases *gradually*: the decaying confidence crosses the solver's
//!   `joint_confidence_threshold` partway through, so the idle A-pose fade
//!   takes over from a plausible pose instead of snapping from a stale one;
//! * nothing is fabricated — a held joint is the last real measurement,
//!   marked [`JointOrigin::Extrapolated`] so statistical consumers can tell
//!   it from a fresh sample.
//!
//! Deliberately NOT a position filter: while the hand is observed this is a
//! pass-through. It only fills gaps.

use crate::asset::HumanoidBone;

use super::source_skeleton::{JointOrigin, SourceJoint, SourceSkeleton};

/// How long a hand may be republished after its last real observation.
///
/// Sized from the measured dropout statistics rather than taste: in the live
/// capture the gaps between consecutive real observations of a hand that was
/// plainly still in view ran 1-7 frames (~50-350 ms at the ~20 Hz inference
/// rate). 0.35 s covers those without covering a real exit, which in the same
/// capture lasted tens of seconds.
const HOLD_S: f32 = 0.35;

/// Joints held together. The forearm is included because it blinks with the
/// hand (measured 0.29 toggles/s for the left) and holding the hand while the
/// elbow vanishes would leave the chain solving a two-bone reach against a
/// missing pole.
const HELD_BONES: [(HumanoidBone, HumanoidBone); 2] = [
    (HumanoidBone::LeftHand, HumanoidBone::LeftLowerArm),
    (HumanoidBone::RightHand, HumanoidBone::RightLowerArm),
];

/// Consecutive real observations before a joint becomes holdable.
///
/// A hold bridges gaps in an ESTABLISHED track. Latching whatever appeared
/// last frame does the opposite: live capture (2026-07-28, t=46.17) caught a
/// right elbow that existed for exactly one frame, and the first version of
/// this module held it for the full window — dragging the avatar's right hand
/// 0.4 m across the body and keeping it there for 7 frames instead of 1.
/// Requiring the track to prove itself first mirrors `ArmEngageGate`'s
/// visibility duty on the entry side.
const ESTABLISH_FRAMES: u32 = 3;

#[derive(Clone, Copy, Debug, Default)]
struct Held {
    joint: Option<SourceJoint>,
    /// Seconds since the last real observation; `>= HOLD_S` means released.
    age_s: f32,
    /// Consecutive observed frames behind `joint`. Below
    /// [`ESTABLISH_FRAMES`] the track is not yet worth bridging.
    streak: u32,
}

impl Held {
    /// Advance one frame against this frame's observation, returning the
    /// joint to publish (`None` = leave absent).
    fn step(&mut self, observed: Option<SourceJoint>, dt_s: f32) -> Option<SourceJoint> {
        match observed {
            Some(j) => {
                self.joint = Some(j);
                self.age_s = 0.0;
                self.streak = self.streak.saturating_add(1);
                Some(j)
            }
            None => {
                let held = self.joint?;
                if self.streak < ESTABLISH_FRAMES {
                    // Never established — a flash, not a track. Forget it
                    // rather than bridging from it.
                    self.joint = None;
                    self.streak = 0;
                    return None;
                }
                self.age_s += dt_s.max(0.0);
                if self.age_s >= HOLD_S {
                    self.joint = None;
                    self.streak = 0;
                    return None;
                }
                // Linear confidence decay across the window. The solver gates
                // joints on confidence, so this is what turns a hard
                // disappearance into a fade — and it keeps a held joint from
                // outranking a real one anywhere that compares confidences.
                let decay = 1.0 - self.age_s / HOLD_S;
                Some(SourceJoint {
                    confidence: held.confidence * decay,
                    ..held
                })
            }
        }
    }
}

/// Per-side hold state for both arms. One instance lives on the provider and
/// is stepped once per inference frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct HandHold {
    sides: [[Held; 2]; 2],
}

impl HandHold {
    /// Drop all held joints — call when tracking restarts / the subject is
    /// re-acquired, so a stale pose from before the gap can't reappear.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Fill hand / forearm gaps in `sk` from the recent past. `dt_s` is the
    /// capture-timestamp step between inference frames.
    pub fn apply(&mut self, sk: &mut SourceSkeleton, dt_s: f32) {
        for (side, (hand, forearm)) in HELD_BONES.iter().enumerate() {
            for (slot, bone) in [(0usize, *hand), (1usize, *forearm)] {
                let observed = sk.joints.get(&bone).copied();
                let was_observed = observed.is_some();
                match self.sides[side][slot].step(observed, dt_s) {
                    Some(j) => {
                        sk.joints.insert(bone, j);
                        if !was_observed {
                            sk.mark_origin(bone, JointOrigin::Extrapolated);
                        }
                    }
                    None => {
                        sk.joints.remove(&bone);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 20.0;

    fn joint(x: f32) -> SourceJoint {
        SourceJoint {
            position: [x, 0.1, 0.2],
            confidence: 0.8,
            metric_depth_m: Some(0.42),
        }
    }

    fn skeleton_with(hand: Option<SourceJoint>) -> SourceSkeleton {
        let mut sk = SourceSkeleton::empty(0);
        if let Some(j) = hand {
            sk.joints.insert(HumanoidBone::RightHand, j);
        }
        sk
    }

    /// Feed `n` observed frames of the same joint, establishing the track.
    fn establish(hold: &mut HandHold, x: f32, n: u32) {
        for _ in 0..n {
            let mut sk = skeleton_with(Some(joint(x)));
            hold.apply(&mut sk, DT);
        }
    }

    #[test]
    fn a_one_frame_flash_is_never_held() {
        // Live regression (t=46.17): a right elbow that existed for a single
        // frame was latched for the whole window, dragging the avatar's hand
        // 0.4 m across the body for 7 frames instead of 1. An unestablished
        // track must be forgotten, not bridged.
        let mut hold = HandHold::default();
        let mut sk = skeleton_with(Some(joint(0.3)));
        hold.apply(&mut sk, DT);
        let mut sk = skeleton_with(None);
        hold.apply(&mut sk, DT);
        assert!(
            !sk.joints.contains_key(&HumanoidBone::RightHand),
            "a single-frame observation must not arm the hold"
        );
    }

    #[test]
    fn brief_dropout_keeps_the_last_observed_hand() {
        // THE live failure: the hand blinks for a few frames while plainly
        // still in view. Before the hold, each blink swung the avatar's hand
        // 0.45 m and back.
        let mut hold = HandHold::default();
        establish(&mut hold, 0.3, ESTABLISH_FRAMES);

        for frame in 0..3 {
            let mut sk = skeleton_with(None);
            hold.apply(&mut sk, DT);
            let j = sk
                .joints
                .get(&HumanoidBone::RightHand)
                .unwrap_or_else(|| panic!("frame {frame}: hand must be held"));
            assert_eq!(j.position, [0.3, 0.1, 0.2], "held pose is the last observation");
            assert!(j.confidence < 0.8, "held confidence must decay");
            assert!(j.confidence > 0.0);
        }
    }

    #[test]
    fn sustained_absence_releases_the_hand() {
        let mut hold = HandHold::default();
        establish(&mut hold, 0.3, ESTABLISH_FRAMES);
        // Past the window the joint must be gone, so the idle A-pose owns the
        // arm instead of a stale target being held forever.
        for _ in 0..(HOLD_S / DT).ceil() as u32 {
            let mut sk = skeleton_with(None);
            hold.apply(&mut sk, DT);
        }
        let mut sk = skeleton_with(None);
        hold.apply(&mut sk, DT);
        assert!(!sk.joints.contains_key(&HumanoidBone::RightHand));
    }

    #[test]
    fn confidence_decays_monotonically_to_release() {
        // The release must be a ramp, not a cliff: the solver's confidence
        // gate should cross partway through the window.
        let mut hold = HandHold::default();
        establish(&mut hold, 0.3, ESTABLISH_FRAMES);
        let mut prev = 1.0_f32;
        let mut seen = 0;
        loop {
            let mut sk = skeleton_with(None);
            hold.apply(&mut sk, DT);
            let Some(j) = sk.joints.get(&HumanoidBone::RightHand) else {
                break;
            };
            assert!(j.confidence < prev, "confidence must strictly decay");
            prev = j.confidence;
            seen += 1;
            assert!(seen < 100, "hold must terminate");
        }
        assert!(seen >= 3, "hold should span several frames, got {seen}");
    }

    #[test]
    fn a_fresh_observation_resets_the_window() {
        let mut hold = HandHold::default();
        establish(&mut hold, 0.3, ESTABLISH_FRAMES);
        for _ in 0..3 {
            let mut sk = skeleton_with(None);
            hold.apply(&mut sk, DT);
        }
        // Re-observed at a new place: full confidence, new position, and the
        // window starts over.
        establish(&mut hold, 0.5, ESTABLISH_FRAMES);
        let mut sk = skeleton_with(Some(joint(0.5)));
        hold.apply(&mut sk, DT);
        let j = sk.joints[&HumanoidBone::RightHand];
        assert_eq!(j.position, [0.5, 0.1, 0.2]);
        assert!((j.confidence - 0.8).abs() < 1e-6, "observed frames pass through");

        let mut sk = skeleton_with(None);
        hold.apply(&mut sk, DT);
        let held = sk.joints[&HumanoidBone::RightHand];
        assert_eq!(held.position, [0.5, 0.1, 0.2], "holds the NEW observation");
        // One frame into a restarted window: 0.8 · (1 − DT/HOLD_S) = 0.686.
        // The point is that it restarted — had the earlier gap carried over,
        // this frame would already be near release.
        let expected = 0.8 * (1.0 - DT / HOLD_S);
        assert!(
            (held.confidence - expected).abs() < 1e-6,
            "window must restart from full confidence, got {}",
            held.confidence
        );
    }

    #[test]
    fn never_invents_a_hand_that_was_never_seen() {
        let mut hold = HandHold::default();
        for _ in 0..10 {
            let mut sk = skeleton_with(None);
            hold.apply(&mut sk, DT);
            assert!(!sk.joints.contains_key(&HumanoidBone::RightHand));
        }
    }

    #[test]
    fn reset_drops_held_state() {
        let mut hold = HandHold::default();
        establish(&mut hold, 0.3, ESTABLISH_FRAMES);
        hold.reset();
        let mut sk = skeleton_with(None);
        hold.apply(&mut sk, DT);
        assert!(!sk.joints.contains_key(&HumanoidBone::RightHand));
    }
}
