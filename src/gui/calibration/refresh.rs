//! Per-frame mailbox snapshot fold-in. Computes the live anchor-
//! visibility flag, overall confidence, pose-match score, and folds
//! per-frame keypoint data into `Collecting`'s sample vec /
//! `RangeCollecting`'s min/max counters. Called once per frame from
//! [`super::panes::draw_preview_pane`] (which already takes the
//! mailbox snapshot for the webcam preview, so we piggyback on its
//! work).

use std::time::Instant;

use crate::gui::GuiApp;
use crate::tracking::{CalibrationMode, MailboxSnapshot};

use super::finalize::measure_shoulder_span;
use super::pose_match::{pose_match_score, POSE_MATCH_THRESHOLD};
use super::state::{relevant_mode, AnchorSample, CalibrationModalState};
use super::MIN_FRAME_CONF;

/// Update the cached "anchor visible / confidence / pose-match"
/// telemetry on the modal state from the latest mailbox snapshot.
/// Also drives the per-frame sample collection in `Collecting` and
/// per-axis min/max in `RangeCollecting`.
pub(super) fn refresh_anchor_telemetry(state: &mut GuiApp, snap: &MailboxSnapshot) {
    let confidence = snap
        .pose
        .as_ref()
        .map(|p| p.overall_confidence)
        .unwrap_or(0.0);
    // Anchor visibility: hip pair for FullBody, shoulder pair for
    // UpperBody. Hits the joints map directly so we read the same
    // floors `skeleton_from_depth` uses.
    let anchor_seen = snap
        .pose
        .as_ref()
        .map(|p| {
            use crate::asset::HumanoidBone;
            match relevant_mode(&state.calibration.modal) {
                Some(CalibrationMode::FullBody) => {
                    p.joints.contains_key(&HumanoidBone::Hips)
                }
                Some(CalibrationMode::UpperBody) => {
                    p.joints.contains_key(&HumanoidBone::LeftShoulder)
                        && p.joints.contains_key(&HumanoidBone::RightShoulder)
                }
                None => false,
            }
        })
        .unwrap_or(false);
    // Lower-arm visibility: both elbows must clear the keypoint floor
    // for `pose_match_score` to return non-zero (see
    // `pose_match::arm_direction`). When the user can't get past 0%
    // match because the camera is too close, this is almost always
    // the cause — surface a dedicated framing hint in `WaitingForPose`
    // rather than letting the user wonder why they're stuck at zero.
    let lower_arms_seen = snap
        .pose
        .as_ref()
        .map(|p| {
            use crate::asset::HumanoidBone;
            p.joints.contains_key(&HumanoidBone::LeftLowerArm)
                && p.joints.contains_key(&HumanoidBone::RightLowerArm)
        })
        .unwrap_or(false);

    let now = Instant::now();
    match &mut state.calibration.modal {
        CalibrationModalState::Idle {
            last_anchor_seen,
            last_confidence,
            ..
        }
        | CalibrationModalState::Collecting {
            last_anchor_seen,
            last_confidence,
            ..
        } => {
            *last_anchor_seen = anchor_seen;
            *last_confidence = confidence;
        }
        CalibrationModalState::WaitingForPose {
            mode,
            last_score,
            frames_at_match,
            last_anchor_seen,
            last_confidence,
            no_anchor_since,
            no_lower_arms_since,
        } => {
            *last_anchor_seen = anchor_seen;
            *last_confidence = confidence;
            // Track sustained anchor-loss so the status pane can
            // surface the mode-switch suggestion after
            // NO_ANCHOR_HINT_SECONDS without recovery.
            if anchor_seen {
                *no_anchor_since = None;
            } else if no_anchor_since.is_none() {
                *no_anchor_since = Some(now);
            }
            // Same edge-trigger pattern for elbow visibility, but
            // only meaningful while the anchor is in frame. If the
            // anchor dropped too, the anchor hint takes priority and
            // the framing hint would just add noise.
            if !anchor_seen || lower_arms_seen {
                *no_lower_arms_since = None;
            } else if no_lower_arms_since.is_none() {
                *no_lower_arms_since = Some(now);
            }
            // Pose-match scoring. Drives the progress bar fill *and*
            // the auto-transition to Collecting. If no live pose this
            // frame, score collapses to 0 and `frames_at_match`
            // resets — a brief tracking dropout doesn't leak into a
            // stale "almost-matching" state.
            let score = snap
                .pose
                .as_ref()
                .map(|p| pose_match_score(p, *mode))
                .unwrap_or(0.0);
            *last_score = score;
            if score >= POSE_MATCH_THRESHOLD {
                *frames_at_match = frames_at_match.saturating_add(1);
            } else {
                *frames_at_match = 0;
            }
        }
        CalibrationModalState::RangeHoldStill {
            last_confidence, ..
        } => {
            *last_confidence = confidence;
        }
        CalibrationModalState::RangeCollecting {
            last_confidence, ..
        } => {
            *last_confidence = confidence;
        }
        _ => {}
    }

    // During the collection window, append the per-frame anchor +
    // confidence to the median pool. Three gates apply:
    //   1. The mailbox sequence must have advanced — otherwise we'd
    //      double-count the same frame each repaint.
    //   2. The anchor must match the chosen mode (hip for FullBody,
    //      shoulder for UpperBody) so a partially-visible subject
    //      doesn't mix anchor types in the median.
    //   3. The frame's overall confidence must clear MIN_FRAME_CONF
    //      so noisy keypoints don't drag the median around.
    if let CalibrationModalState::Collecting {
        mode,
        ref mut samples,
        ref mut expr_accum,
        ref mut last_seq_consumed,
        ..
    } = state.calibration.modal
    {
        if snap.sequence > *last_seq_consumed {
            *last_seq_consumed = snap.sequence;
            if let Some(pose) = snap.pose.as_ref() {
                let anchor_kind_matches = match mode {
                    CalibrationMode::FullBody => pose.root_anchor_is_hip,
                    CalibrationMode::UpperBody => !pose.root_anchor_is_hip,
                };
                if anchor_kind_matches && pose.overall_confidence >= MIN_FRAME_CONF {
                    if let Some(pos) = pose.root_offset {
                        samples.push(AnchorSample {
                            position: pos,
                            confidence: pose.overall_confidence,
                            shoulder_span_m: measure_shoulder_span(pose),
                        });
                    }
                    // Fold the resting expression weights into the running
                    // mean (independent of `root_offset` — a desk-distance
                    // upper-body capture still has a face). When face
                    // tracking is off, `pose.expressions` is empty and this
                    // leaves `expr_accum` untouched → no neutral baseline.
                    for expr in &pose.expressions {
                        let acc = expr_accum.entry(expr.name.clone()).or_insert((0.0, 0));
                        acc.0 += expr.weight;
                        acc.1 += 1;
                    }
                }
            }
        }
    }

    // Range-capture path: track per-axis min/max instead of medianing.
    // We *don't* gate on anchor-kind match here — if the user
    // calibrated full body but happens to crouch out of hip
    // visibility mid-sweep, the residual shoulder-anchor sample is
    // still useful for the X range (and z stays None on rtmw3d-only
    // anyway). The confidence floor is the only quality gate.
    if let CalibrationModalState::RangeCollecting {
        ref mut x_min,
        ref mut x_max,
        ref mut z_min,
        ref mut z_max,
        ref mut samples_seen,
        ref mut last_seq_consumed,
        ..
    } = state.calibration.modal
    {
        if snap.sequence > *last_seq_consumed {
            *last_seq_consumed = snap.sequence;
            if let Some(pose) = snap.pose.as_ref() {
                if pose.overall_confidence >= MIN_FRAME_CONF {
                    if let Some(pos) = pose.root_offset {
                        if pos[0] < *x_min {
                            *x_min = pos[0];
                        }
                        if pos[0] > *x_max {
                            *x_max = pos[0];
                        }
                        if pos[2] < *z_min {
                            *z_min = pos[2];
                        }
                        if pos[2] > *z_max {
                            *z_max = pos[2];
                        }
                        *samples_seen += 1;
                    }
                }
            }
        }
    }
}
