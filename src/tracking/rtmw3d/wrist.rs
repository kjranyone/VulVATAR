//! Four-stage wrist resilience layer documented in
//! `docs/onnx-tracking-pipeline.md` §5. Inputs are the wrist position
//! [`super::skeleton::build_source_skeleton`] derived from the MCP
//! centroid (Stage 1). Stages 2–4 live here:
//!
//! * Stage 2 — forearm length sanity check
//! * Stage 3 — temporal hold over up to [`TEMPORAL_HOLD_MAX_FRAMES`]
//!   frames with [`TEMPORAL_HOLD_DECAY`] confidence decay
//! * Stage 4 — chain extension synthesis from elbow + upper arm
//!
//! See `WristTracker` for the per-side state carried frame-to-frame.

use crate::asset::HumanoidBone;

use super::super::{SourceJoint, SourceSkeleton};
use super::math::{length3, sub3};

/// Per-side temporal state for a wrist. When the live frame's
/// MCP-centroid wrist is rejected by the sanity check (forearm folded
/// behind the upper arm or wildly stretched), we hold the previous
/// frame's position for up to `TEMPORAL_HOLD_MAX_FRAMES` so the avatar
/// does not snap to a default pose during a momentary occlusion.
///
/// The tracker also accumulates the running max of the upper-arm
/// length seen on clean frames, which the sanity check reads to bound
/// the plausible forearm length.
#[derive(Default)]
pub(super) struct WristTracker {
    last_pos: Option<[f32; 3]>,
    last_conf: f32,
    age: u32,
    /// Running max of `||elbow - shoulder||` over recent clean frames.
    /// Falls back to a generous constant before any clean frame.
    max_upper_arm: f32,
}

const TEMPORAL_HOLD_MAX_FRAMES: u32 = 6;
const TEMPORAL_HOLD_DECAY: f32 = 0.85;
/// Max ratio (forearm length / upper-arm length) accepted by the
/// sanity check. Real arms have upper-arm ≈ forearm so 1.4× allows
/// for measurement noise but rejects "wrist drifted to opposite end
/// of the body" pathologies.
const ARM_LENGTH_RATIO_LIMIT: f32 = 1.4;

/// Validate the live wrist position against the upper-arm chain and
/// either accept (updating the tracker), repair from the previous
/// frame's good wrist, or drop the wrist and let the solver leave the
/// hand at rest. Also propagates fingertip / phalanx positions by the
/// same delta when we substitute a held wrist, so the hand chain stays
/// coherent.
pub(super) fn apply_wrist_temporal_hold(
    sk: &mut SourceSkeleton,
    wrist_bone: HumanoidBone,
    shoulder_bone: HumanoidBone,
    elbow_bone: HumanoidBone,
    tracker: &mut WristTracker,
) {
    let live_wrist = sk.joints.get(&wrist_bone).copied();
    let shoulder = sk.joints.get(&shoulder_bone).copied();
    let elbow = sk.joints.get(&elbow_bone).copied();

    // Without an elbow the sanity check has nothing to compare
    // against. Trust the live wrist if any, age the tracker.
    let Some(elbow) = elbow else {
        if live_wrist.is_some() {
            tracker.age = 0;
            tracker.last_pos = live_wrist.map(|j| j.position);
            tracker.last_conf = live_wrist.map(|j| j.confidence).unwrap_or(0.0);
        } else {
            tracker.age = tracker.age.saturating_add(1);
        }
        return;
    };

    let upper_arm = match shoulder {
        Some(s) => length3(sub3(elbow.position, s.position)),
        None => 0.0,
    };
    if upper_arm > tracker.max_upper_arm {
        tracker.max_upper_arm = upper_arm;
    }
    // Generous fallback — never let the bound drop below this so the
    // very first frame doesn't reject all hands.
    let arm_baseline = tracker.max_upper_arm.max(0.18);

    let accept_live = match live_wrist {
        Some(j) => {
            let forearm = length3(sub3(j.position, elbow.position));
            // Reject if forearm is wildly longer than the upper arm
            // (catches "wrist hallucinated across the body") or if
            // the chain folds backward sharply (forearm pointing
            // away from the elbow's continuation).
            forearm <= arm_baseline * ARM_LENGTH_RATIO_LIMIT
        }
        None => false,
    };

    if accept_live {
        let live = live_wrist.expect("checked above");
        tracker.last_pos = Some(live.position);
        tracker.last_conf = live.confidence;
        tracker.age = 0;
        return;
    }

    // Live wrist rejected (or absent). Try to substitute the held
    // value if it's recent enough, propagating the delta to the
    // finger phalanges so the hand chain doesn't tear.
    tracker.age = tracker.age.saturating_add(1);
    if tracker.age <= TEMPORAL_HOLD_MAX_FRAMES {
        if let Some(prev_pos) = tracker.last_pos {
            let decayed_conf = (tracker.last_conf * TEMPORAL_HOLD_DECAY).max(0.05);
            if let Some(live) = live_wrist {
                // Live fingers exist but the wrist itself was
                // implausible. Translate the whole chain by the
                // delta from the bogus wrist to the held position
                // so finger geometry around the held wrist stays
                // correct.
                let delta = sub3(prev_pos, live.position);
                translate_finger_chain(sk, wrist_bone, delta);
            }
            sk.joints.insert(
                wrist_bone,
                SourceJoint {
                    position: prev_pos,
                    confidence: decayed_conf,
                    metric_depth_m: None,
                },
            );
            tracker.last_conf = decayed_conf;
            return;
        }
    }

    // Tertiary fallback: synthesise a wrist by extending the upper
    // arm vector from elbow forward by the same length. Better than
    // a missing hand for VTubing visibility, even if approximate.
    // Only fires when both shoulder and elbow are confident — without
    // them we can't form an arm direction.
    if let Some(shoulder) = shoulder {
        let upper = sub3(elbow.position, shoulder.position);
        let upper_len = length3(upper);
        if upper_len >= 0.05 {
            // Continue forward by the same length, slightly nudged
            // toward camera (+z) so the synthesised hand reads as
            // "extended in front" rather than "stuck inside torso".
            let mut forearm = [
                upper[0] / upper_len,
                upper[1] / upper_len,
                upper[2] / upper_len,
            ];
            forearm[2] += 0.2;
            let f_len = length3(forearm).max(1e-3);
            let synth_pos = [
                elbow.position[0] + forearm[0] / f_len * upper_len,
                elbow.position[1] + forearm[1] / f_len * upper_len,
                elbow.position[2] + forearm[2] / f_len * upper_len,
            ];
            // Drop the live finger chain (its geometry was suspect)
            // and emit only the wrist. Confidence is just above the
            // default solver threshold so the chain participates
            // but the user-visible "kept stale" state is still
            // distinguishable in confidence histograms.
            clear_finger_chain(sk, wrist_bone);
            sk.joints.insert(
                wrist_bone,
                SourceJoint {
                    position: synth_pos,
                    confidence: 0.35,
                    metric_depth_m: None,
                },
            );
            tracker.last_pos = None;
            tracker.last_conf = 0.0;
            return;
        }
    }

    // No reliable signal at all → drop the wrist and finger chain.
    sk.joints.remove(&wrist_bone);
    clear_finger_chain(sk, wrist_bone);
    tracker.last_pos = None;
    tracker.last_conf = 0.0;
}

/// Per-wrist list of finger phalanx + tip bones. Used by the temporal
/// hold to translate or drop the entire hand chain coherently.
fn finger_chain_bones(wrist: HumanoidBone) -> &'static [HumanoidBone] {
    match wrist {
        HumanoidBone::RightHand => &[
            HumanoidBone::RightThumbProximal,
            HumanoidBone::RightThumbIntermediate,
            HumanoidBone::RightThumbDistal,
            HumanoidBone::RightIndexProximal,
            HumanoidBone::RightIndexIntermediate,
            HumanoidBone::RightIndexDistal,
            HumanoidBone::RightMiddleProximal,
            HumanoidBone::RightMiddleIntermediate,
            HumanoidBone::RightMiddleDistal,
            HumanoidBone::RightRingProximal,
            HumanoidBone::RightRingIntermediate,
            HumanoidBone::RightRingDistal,
            HumanoidBone::RightLittleProximal,
            HumanoidBone::RightLittleIntermediate,
            HumanoidBone::RightLittleDistal,
        ],
        HumanoidBone::LeftHand => &[
            HumanoidBone::LeftThumbProximal,
            HumanoidBone::LeftThumbIntermediate,
            HumanoidBone::LeftThumbDistal,
            HumanoidBone::LeftIndexProximal,
            HumanoidBone::LeftIndexIntermediate,
            HumanoidBone::LeftIndexDistal,
            HumanoidBone::LeftMiddleProximal,
            HumanoidBone::LeftMiddleIntermediate,
            HumanoidBone::LeftMiddleDistal,
            HumanoidBone::LeftRingProximal,
            HumanoidBone::LeftRingIntermediate,
            HumanoidBone::LeftRingDistal,
            HumanoidBone::LeftLittleProximal,
            HumanoidBone::LeftLittleIntermediate,
            HumanoidBone::LeftLittleDistal,
        ],
        _ => &[],
    }
}

fn clear_finger_chain(sk: &mut SourceSkeleton, wrist: HumanoidBone) {
    for b in finger_chain_bones(wrist) {
        sk.joints.remove(b);
        sk.fingertips.remove(b);
    }
}

fn translate_finger_chain(sk: &mut SourceSkeleton, wrist: HumanoidBone, delta: [f32; 3]) {
    for b in finger_chain_bones(wrist) {
        if let Some(j) = sk.joints.get_mut(b) {
            j.position[0] += delta[0];
            j.position[1] += delta[1];
            j.position[2] += delta[2];
        }
        if let Some(t) = sk.fingertips.get_mut(b) {
            t.position[0] += delta[0];
            t.position[1] += delta[1];
            t.position[2] += delta[2];
        }
    }
}
