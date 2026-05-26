//! State-machine transitions for the calibration modal. Edges that
//! correspond to user button-presses (`Start`, `Capture Now`, `Retry`,
//! `Skip & Finish`, `Capture Range ▶`) plus the time-/event-driven
//! `advance_state` tick called every frame from
//! [`super::panes::draw_modal`]. Pure state-graph logic — no rendering.

use std::time::Instant;

use crate::gui::GuiApp;
use crate::tracking::{CalibrationMode, PoseCalibration};

use super::finalize::{finalize_collection, finalize_range_collection};
use super::pose_match::REQUIRED_STABLE_FRAMES;
use super::state::{CalibrationModalState, DoneOutcome};
use super::{COLLECTION_SECONDS, RANGE_COLLECTION_SECONDS, RANGE_HOLD_STILL_SECONDS};

/// User hit `Start` from the Idle pane: enter the pose-match wait.
/// The mode currently selected on the segmented buttons is carried
/// into `WaitingForPose`; from this point on the mode is locked (the
/// segmented buttons are no longer rendered) so a mid-capture switch
/// can't invalidate samples. `_now` is unused because the wait state
/// is event-driven (pose-match), not time-driven — kept for symmetry
/// with `begin_range_capture` and as a hook for any future "started
/// at" telemetry on the variant.
pub(super) fn begin_capture(state: &mut GuiApp) {
    let _now = Instant::now();
    let next = match &state.calibration.modal {
        CalibrationModalState::Idle { mode, .. } => {
            Some(CalibrationModalState::WaitingForPose {
                mode: *mode,
                last_score: 0.0,
                frames_at_match: 0,
                last_anchor_seen: false,
                last_confidence: 0.0,
                no_anchor_since: None,
                no_lower_arms_since: None,
            })
        }
        _ => None,
    };
    if let Some(s) = next {
        state.calibration.modal = s;
    }
}

/// User chose `Retry` from the AnchorDone prompt: discard the
/// in-flight modal state and re-enter Idle at the same mode so they
/// can capture again without leaving the modal and re-clicking the
/// inspector button.
///
/// The previously persisted anchor calibration on
/// `Application::tracking_calibration.pose` (and the active profile)
/// is *not* reverted — completing the retry will overwrite it via the
/// next `persist_calibration` call. If the user instead Cancels after
/// hitting Retry, the previous record remains in place. This matches
/// "I want to redo this" intent: clicking Retry doesn't itself destroy
/// data; only landing a fresh capture does.
///
/// Also flips the worker's torso-capture toggle off and bumps the
/// already-consumed template seq, so any template the worker was
/// about to publish from the previous capture window doesn't leak
/// into the new attempt's `poll_torso_template`.
pub(super) fn retry_capture(state: &mut GuiApp) {
    let mode = match &state.calibration.modal {
        CalibrationModalState::AnchorDone { mode, .. } => Some(*mode),
        _ => None,
    };
    if let Some(mode) = mode {
        state.app.tracking.mailbox().set_torso_capture(false);
        state.calibration.torso_template_seq =
            state.app.tracking.mailbox().torso_template_seq();
        state.calibration.modal.open(mode);
    }
}

/// User chose `Skip & Finish` from the AnchorDone prompt: jump
/// straight to the success-display Done state with the existing
/// anchor-only calibration. The calibration was already persisted by
/// `finalize_collection`, so there's nothing else to write.
pub(super) fn skip_range(state: &mut GuiApp) {
    let now = Instant::now();
    let next = match &state.calibration.modal {
        CalibrationModalState::AnchorDone {
            mode, calibration, ..
        } => Some(transition_to_done_success(*mode, calibration.clone(), now)),
        _ => None,
    };
    if let Some(s) = next {
        state.calibration.modal = s;
    }
}

/// User chose `Capture Range ▶` from the AnchorDone prompt: roll the
/// state machine into the range-capture pre-roll. The calibration
/// from the anchor step is carried forward via the variant payload
/// so `finalize_range_collection` can update it in place.
pub(super) fn begin_range_capture(state: &mut GuiApp) {
    let now = Instant::now();
    let next = match &state.calibration.modal {
        CalibrationModalState::AnchorDone {
            mode, calibration, ..
        } => Some(CalibrationModalState::RangeHoldStill {
            mode: *mode,
            started_at: now,
            calibration: calibration.clone(),
            last_confidence: 0.0,
        }),
        _ => None,
    };
    if let Some(s) = next {
        state.calibration.modal = s;
    }
}

/// Drive the state machine forward by elapsed time / pose-match
/// score. Called every frame from `panes::draw_modal`. Edge cases:
///
/// - `WaitingForPose`         → `Collecting` once the pose-match
///   gate fires. Triggers torso-template capture on the depth
///   provider via the mailbox toggle.
/// - `Collecting`             → `AnchorDone | Done{Insufficient}` on
///   timer expiry, via [`finalize_collection`].
/// - `RangeHoldStill`         → `RangeCollecting` after the brief
///   pre-roll; seeds the per-axis min/max sentinels.
/// - `RangeCollecting`        → `Done` via [`finalize_range_collection`].
pub(super) fn advance_state(state: &mut GuiApp) {
    let now = Instant::now();
    let next = match &state.calibration.modal {
        CalibrationModalState::WaitingForPose {
            mode,
            frames_at_match,
            ..
        } if *frames_at_match >= REQUIRED_STABLE_FRAMES => {
            // Pose-match gate cleared: user has held the target pose
            // stably for ~0.5 s. Tell the depth-pipeline provider to
            // start accumulating per-frame torso depth samples for the
            // template. The worker forwards this to
            // `set_torso_capture(true)` on its next iteration; the
            // actual capture work runs inside `estimate_pose` for the
            // duration of the Collecting window.
            state.app.tracking.mailbox().set_torso_capture(true);
            Some(CalibrationModalState::Collecting {
                mode: *mode,
                started_at: now,
                samples: Vec::with_capacity(64),
                last_seq_consumed: 0,
                last_anchor_seen: false,
                last_confidence: 0.0,
            })
        }
        CalibrationModalState::Collecting {
            mode, started_at, ..
        } if now.duration_since(*started_at).as_secs_f32() >= COLLECTION_SECONDS => {
            // Anchor window expired: finalise. `finalize_collection`
            // either transitions us to `AnchorDone` (with the
            // user-prompt for the optional range step) on success,
            // or straight to `Done { Insufficient }` when too few
            // frames cleared the quality bar.
            Some(finalize_collection(state, *mode, now))
        }
        CalibrationModalState::RangeHoldStill {
            mode,
            started_at,
            calibration,
            ..
        } if now.duration_since(*started_at).as_secs_f32() >= RANGE_HOLD_STILL_SECONDS => {
            // Seed min/max with +∞ / −∞ so the first incoming sample
            // unconditionally tightens the bounds. f32::INFINITY here
            // is fine — the finalize step gates on `samples_seen`
            // before consuming any of these values.
            Some(CalibrationModalState::RangeCollecting {
                mode: *mode,
                started_at: now,
                calibration: calibration.clone(),
                x_min: f32::INFINITY,
                x_max: f32::NEG_INFINITY,
                z_min: f32::INFINITY,
                z_max: f32::NEG_INFINITY,
                samples_seen: 0,
                last_seq_consumed: 0,
                last_confidence: 0.0,
            })
        }
        CalibrationModalState::RangeCollecting {
            mode, started_at, ..
        } if now.duration_since(*started_at).as_secs_f32() >= RANGE_COLLECTION_SECONDS => {
            // Range window expired: fold whatever we observed into
            // the in-flight calibration and transition to the
            // success-display.
            Some(finalize_range_collection(state, *mode, now))
        }
        _ => None,
    };
    if let Some(s) = next {
        state.calibration.modal = s;
    }
}

/// Force the active capture step to terminate or kick over to the
/// next phase. Triggered by the `Capture Now` button. Transition
/// rules:
///
/// - `WaitingForPose`       → bypass the pose-match gate and start
///   the 2-second sample window. The user will still need to wait
///   for samples to accumulate, but they're not blocked on the
///   per-frame pose-match score crossing the threshold (escape hatch
///   for users whose body proportions sit outside the
///   [`super::pose_match::pose_match_score`] tolerance — e.g. very
///   narrow shoulders that don't reach the "extended" criterion in
///   T-pose).
/// - `Collecting`           → finalize anchor with current samples
///   (same `MIN_SAMPLES` floor — clicking too early can still produce
///   `Insufficient`).
/// - `RangeHoldStill`       → skip range, finish with anchor only.
/// - `RangeCollecting`      → finalize range with whatever min/max
///   we accumulated.
pub(super) fn finish_capture(state: &mut GuiApp) {
    let now = Instant::now();
    let next = match &state.calibration.modal {
        CalibrationModalState::WaitingForPose { mode, .. } => {
            // Bypass the gate: kick off Collecting straight away so
            // the 2-second sample window starts now. Same torso-capture
            // toggle as the natural pose-match path.
            state.app.tracking.mailbox().set_torso_capture(true);
            Some(CalibrationModalState::Collecting {
                mode: *mode,
                started_at: now,
                samples: Vec::with_capacity(64),
                last_seq_consumed: 0,
                last_anchor_seen: false,
                last_confidence: 0.0,
            })
        }
        CalibrationModalState::Collecting { mode, .. } => {
            Some(finalize_collection(state, *mode, now))
        }
        CalibrationModalState::RangeHoldStill {
            mode, calibration, ..
        } => Some(transition_to_done_success(*mode, calibration.clone(), now)),
        CalibrationModalState::RangeCollecting { mode, .. } => {
            Some(finalize_range_collection(state, *mode, now))
        }
        _ => None,
    };
    if let Some(s) = next {
        state.calibration.modal = s;
    }
}

/// Compose a `Done { Success }` state from an already-aggregated
/// calibration. Used by the `Skip & Finish` and `RangeHoldStill +
/// Capture Now` paths — both already wrote the anchor calibration to
/// `Application` during the original `finalize_collection` call, so
/// here we only need to surface the success display.
pub(super) fn transition_to_done_success(
    mode: CalibrationMode,
    calibration: PoseCalibration,
    now: Instant,
) -> CalibrationModalState {
    CalibrationModalState::Done {
        mode,
        shown_at: now,
        outcome: DoneOutcome::Success {
            frame_count: calibration.frame_count,
            calibration,
        },
    }
}

