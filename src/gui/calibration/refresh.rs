//! Per-frame mailbox snapshot fold-in. Computes the live anchor-
//! visibility flag, overall confidence, pose-match score, and folds
//! per-frame keypoint data into `Collecting`'s sample vec /
//! `RangeCollecting`'s min/max counters. Called once per frame from
//! [`super::panes::draw_preview_pane`] (which already takes the
//! mailbox snapshot for the camera preview, so we piggyback on its
//! work).

use std::time::Instant;

use crate::gui::GuiApp;
use crate::tracking::{CalibrationMode, MailboxSnapshot};

use super::finalize::measure_shoulder_span;
use super::pose_match::{pose_match_score, stillness_score, POSE_MATCH_THRESHOLD};
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
    // floors the v1 depth pipeline used.
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
            stillness_fallback,
            last_anchor_pos,
            last_seq_consumed,
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
            // Bust-up framing fallback (docs/calibration-ux.md):
            // shoulders visible but elbows cropped out for
            // the whole hint window means the arm-direction gate can
            // never fire at this framing. Latch onto the stillness
            // gate instead of leaving the user stuck at 0 % with a
            // "step back" hint they can't act on. UpperBody only —
            // at full-body framing, missing elbows really do mean
            // "camera too close for a T-pose".
            if !*stillness_fallback && *mode == CalibrationMode::UpperBody {
                let elbows_gone_for = no_lower_arms_since
                    .map(|t| now.duration_since(t).as_secs_f32())
                    .unwrap_or(0.0);
                if elbows_gone_for >= super::NO_ANCHOR_HINT_SECONDS {
                    *stillness_fallback = true;
                    *last_score = 0.0;
                    *frames_at_match = 0;
                    *last_anchor_pos = None;
                }
            }
            if *stillness_fallback {
                // Stillness gate: score anchor displacement per
                // *tracking frame* (mailbox sequence), never per GUI
                // repaint — repaint-rate deltas read as near-zero
                // motion at any speed.
                if snap.sequence > *last_seq_consumed {
                    *last_seq_consumed = snap.sequence;
                    // Same admission bar Collecting applies to its
                    // samples: no point starting a capture from
                    // frames the collection window would reject.
                    let admitted = snap.pose.as_ref().and_then(|p| {
                        (anchor_seen && p.overall_confidence >= MIN_FRAME_CONF)
                            .then_some(p.root_offset)
                            .flatten()
                    });
                    if let Some(pos) = admitted {
                        let curr = [pos[0], pos[1]];
                        if let Some(prev) = *last_anchor_pos {
                            let score = stillness_score(prev, curr);
                            *last_score = score;
                            if score > 0.0 {
                                *frames_at_match = frames_at_match.saturating_add(1);
                            } else {
                                *frames_at_match = 0;
                            }
                        }
                        *last_anchor_pos = Some(curr);
                    } else {
                        // Dropout: reset the hold *and* the previous
                        // position so we don't compare across a gap.
                        *last_score = 0.0;
                        *frames_at_match = 0;
                        *last_anchor_pos = None;
                    }
                }
            } else {
                // Pose-match scoring. Drives the progress bar fill
                // *and* the auto-transition to Collecting. If no live
                // pose this frame, score collapses to 0 and
                // `frames_at_match` resets — a brief tracking dropout
                // doesn't leak into a stale "almost-matching" state.
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
        ref mut face_accum_mesh,
        ref mut face_accum_body,
        ref mut body_yaw_accum,
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
                    // Shoulder-line yaw for the neutral-body-yaw median
                    // (oblique camera placement). The helper
                    // refuses non-metric frames, missing shoulders and
                    // degenerate spans, so a `Some` here is a reading
                    // worth aggregating. Raw-mailbox invariant holds:
                    // the snapshot is never calibration-rotated, so a
                    // recapture measures the true camera angle.
                    if let Some(yaw) = crate::tracking::shoulder_line_yaw(pose) {
                        body_yaw_accum.push(yaw);
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
                    // Resting head pose for `neutral_face_ypr_{mesh,body}`.
                    // The snapshot pose is raw (calibration is applied at
                    // solve time on a clone), so this captures the
                    // absolute camera-relative angles — no risk of
                    // subtracting a previous calibration twice. Gate on
                    // the face's own confidence: a collapsed FaceMesh /
                    // occluded ear-line pose must not drag the median.
                    //
                    // Accumulated per estimator: the published `face`
                    // feeds the accumulator of whichever source produced
                    // it, and the always-raw body pose feeds the body
                    // accumulator so both neutrals are measured in one
                    // frontal hold (where the mesh wins selection nearly
                    // every frame).
                    if let Some(face) = pose.face {
                        // Mid-crossfade frames carry the TARGET source's
                        // tag over MIXED angles — admitting them would
                        // pollute the target source's neutral with the
                        // other estimator's residual. Skip them; the
                        // window has ~60 steady frames to spare 6.
                        if face.confidence >= 0.5 && face.blend.is_none() {
                            match face.source {
                                crate::tracking::FaceSource::Mesh => {
                                    face_accum_mesh.push([face.yaw, face.pitch, face.roll])
                                }
                                crate::tracking::FaceSource::Body => {
                                    face_accum_body.push([face.yaw, face.pitch, face.roll])
                                }
                            }
                        }
                    }
                    if let Some(body) = pose.face_body_raw {
                        // Avoid double-pushing when the body pose was
                        // also the published selection this frame.
                        let published_is_body = pose
                            .face
                            .map(|f| f.source == crate::tracking::FaceSource::Body)
                            .unwrap_or(false);
                        if !published_is_body && body.confidence >= 0.5 {
                            face_accum_body.push([body.yaw, body.pitch, body.roll]);
                        }
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
        ref calibration,
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
                    if let Some(raw_pos) = pose.root_offset {
                        // With a neutral body yaw on the in-flight
                        // calibration, runtime offsets live in the
                        // de-rotated (body) frame — so fold range
                        // extremes over samples rotated the same way.
                        // Rotating EACH sample (not the finished box)
                        // matters: an axis-aligned min/max box is not
                        // rotation-equivariant. The pivot is irrelevant
                        // for peak-to-peak deltas, so rotate about the
                        // origin. Metric-gated like the runtime path.
                        let pos = match calibration.neutral_body_yaw {
                            Some(theta) if pose.metric_frame_info.is_some() => {
                                crate::tracking::rotate_xz(raw_pos, theta)
                            }
                            _ => raw_pos,
                        };
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
