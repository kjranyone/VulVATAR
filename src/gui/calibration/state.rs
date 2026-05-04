//! State-machine types for the calibration modal: the variant enum
//! itself, the per-frame `AnchorSample` rows captured during
//! `Collecting`, and the result variant `DoneOutcome`. Pure data —
//! no GUI rendering, no state mutation logic. Transitions live in
//! [`super::transitions`].

use std::time::Instant;

use crate::tracking::{CalibrationMode, PoseCalibration};

/// One per-frame anchor reading collected during the capture window.
/// Stored as `Vec<AnchorSample>` on the `Collecting` variant; the
/// transition to `Done` walks the vec to compute medians + stddev.
#[derive(Clone, Copy, Debug)]
pub struct AnchorSample {
    /// Source-space anchor position (matches `SourceSkeleton::root_offset`).
    /// For depth-aware paths z is metric metres; for rtmw3d-only z is 0.
    pub position: [f32; 3],
    /// Per-frame `overall_confidence`. Aggregated as a separate median
    /// so the surfaced calibration confidence reflects the gathered
    /// samples, not just the first frame's reading.
    pub confidence: f32,
    /// Per-frame measured shoulder span (3D distance between
    /// LeftShoulder and RightShoulder joints, in source-space metres
    /// for depth-aware providers). `None` when either shoulder was
    /// missing this frame, or for rtmw3d-only paths where source
    /// positions are not metric. Aggregated by `aggregate()` as the
    /// median of finite samples to give a per-subject body-scale
    /// reference that the depth pipeline uses for bone-length-aware
    /// Z reconstruction (see
    /// [`crate::tracking::skeleton_from_depth::reconstruct_arm_z_magnitudes`]).
    pub shoulder_span_m: Option<f32>,
}

/// Internal state of the modal. `Closed` means it is not rendered at
/// all; the other variants drive the per-frame UI under
/// [`super::panes::draw_modal`].
#[derive(Clone, Debug, Default)]
pub enum CalibrationModalState {
    #[default]
    Closed,
    /// User-input gated pre-capture state. The modal is open and the
    /// user can pick a mode + verify their framing, but no sample
    /// collection is happening yet. Pressing **Start** transitions to
    /// `WaitingForPose`. This state replaces the prior "open ==
    /// HoldStill counts down immediately" behavior — the implicit
    /// countdown made the modal feel like it had ambushed the user,
    /// since the instruction pane and the active sampling timer were
    /// both fighting for attention from the moment the dialog
    /// appeared.
    Idle {
        mode: CalibrationMode,
        last_anchor_seen: bool,
        last_confidence: f32,
    },
    /// Active pose-match wait. Replaces the prior `HoldStill` fixed-
    /// timer pre-roll with an event-driven gate: the user takes the
    /// target pose at their own pace, the per-frame
    /// [`super::pose_match::pose_match_score`] is computed against
    /// [`super::pose_match::POSE_MATCH_THRESHOLD`], and once the score
    /// stays above the threshold for
    /// [`super::pose_match::REQUIRED_STABLE_FRAMES`] consecutive
    /// frames we transition to `Collecting`. This solves three prior
    /// pain points: (a) users had no time to read the instruction
    /// text before the fixed countdown ran out; (b) users in the
    /// wrong mode (e.g. FullBody at a desk where hips are off-frame)
    /// could fail capture without ever realising they couldn't
    /// physically achieve the requested pose; (c) the timer didn't
    /// care whether the user had actually struck the pose.
    ///
    /// `frames_at_match` resets to 0 whenever the score drops below
    /// the threshold, so the user has to *hold* the pose stably — a
    /// momentary glance through T-pose en route to scratching their
    /// nose doesn't trip capture.
    ///
    /// `no_anchor_since` records the first frame on which the
    /// mode-relevant anchor went missing. After
    /// [`super::NO_ANCHOR_HINT_SECONDS`] without a recovery, the
    /// status pane surfaces the mode-switch suggestion (see
    /// `panes::draw_capturing_pane`).
    WaitingForPose {
        mode: CalibrationMode,
        last_score: f32,
        frames_at_match: u32,
        last_anchor_seen: bool,
        last_confidence: f32,
        no_anchor_since: Option<Instant>,
    },
    /// Active anchor sample collection. Per-frame `AnchorSample` rows
    /// are appended each time the mailbox sequence advances *and* the
    /// frame meets the quality bar (anchor visible, anchor type
    /// matches mode, confidence ≥ `MIN_FRAME_CONF`). The transition
    /// to `AnchorDone` walks `samples` to compute the median
    /// calibration values; failure goes straight to `Done`.
    Collecting {
        mode: CalibrationMode,
        started_at: Instant,
        samples: Vec<AnchorSample>,
        /// Tracks the last mailbox sequence we admitted a sample for,
        /// so we don't double-count when the GUI repaints faster than
        /// the tracking thread emits new frames.
        last_seq_consumed: u64,
        last_anchor_seen: bool,
        last_confidence: f32,
    },
    /// Anchor capture succeeded — calibration is already written to
    /// `Application::tracking_calibration.pose`. Pauses the state
    /// machine on a "Capture Range?" prompt: user can hit
    /// `Capture Range ▶` to enter the optional 10-second range step
    /// (records per-axis min/max, derives `x_range_observed` /
    /// `z_range_observed`), or `Skip & Finish` to commit the
    /// anchor-only calibration immediately.
    ///
    /// `calibration` is held in this variant rather than re-derived
    /// on the user's choice so the same aggregate is forwarded to the
    /// `Done` state (success-display) regardless of the range path.
    AnchorDone {
        mode: CalibrationMode,
        shown_at: Instant,
        calibration: PoseCalibration,
    },
    /// Pre-roll countdown for the range capture. Shorter than the
    /// anchor pre-roll because the user already settled into the
    /// modal and just needs a beat to start moving.
    RangeHoldStill {
        mode: CalibrationMode,
        started_at: Instant,
        calibration: PoseCalibration,
        last_confidence: f32,
    },
    /// 10-second range capture. Per-frame source `(x, z)` is folded
    /// into running min/max; on transition to `Done` the per-axis
    /// peak-to-peak deltas become `x_range_observed` /
    /// `z_range_observed` on the in-flight `PoseCalibration`. The
    /// solver picks those up to derive per-axis sensitivity from the
    /// observed range.
    ///
    /// `last_seq_consumed` matches `Collecting`'s — same
    /// "GUI repaints faster than the tracking thread emits frames"
    /// reason. `samples_seen` is a non-aggregating counter for the
    /// status pane so the user can tell capture is alive.
    RangeCollecting {
        mode: CalibrationMode,
        started_at: Instant,
        calibration: PoseCalibration,
        x_min: f32,
        x_max: f32,
        z_min: f32,
        z_max: f32,
        samples_seen: usize,
        last_seq_consumed: u64,
        last_confidence: f32,
    },
    /// Calibration finished — either the timer ran out or the user
    /// hit Capture Now / Skip & Finish. Either succeeded (anchor
    /// captured, optionally enriched with range data) or was rejected
    /// (`outcome == DoneOutcome::Insufficient`); status pane shows
    /// the matching message before the modal auto-closes.
    Done {
        mode: CalibrationMode,
        shown_at: Instant,
        outcome: DoneOutcome,
    },
}

/// Result of a `Collecting → Done` transition, surfaced on the status
/// pane so the user knows whether they need to retry.
#[derive(Clone, Debug)]
pub enum DoneOutcome {
    /// Capture succeeded. Carries the aggregated calibration that was
    /// just written to `Application::tracking_calibration.pose`, used
    /// only for the post-capture status display.
    Success {
        frame_count: usize,
        calibration: PoseCalibration,
    },
    /// Capture failed. `samples_collected` is below `MIN_SAMPLES` —
    /// the modal shows "Capture failed — N samples (need M)" so the
    /// user can adjust framing / lighting and retry.
    Insufficient { samples_collected: usize },
}

impl CalibrationModalState {
    pub fn is_open(&self) -> bool {
        !matches!(self, CalibrationModalState::Closed)
    }

    pub fn open(&mut self, mode: CalibrationMode) {
        *self = CalibrationModalState::Idle {
            mode,
            last_anchor_seen: false,
            last_confidence: 0.0,
        };
    }

    /// Update the calibration mode while the modal is in `Idle`
    /// (= before the user has hit Start). No-op in any other state —
    /// once the capture timer is running, the segmented buttons are
    /// hidden anyway, but this guard makes the contract explicit and
    /// protects against a future caller that doesn't know about the
    /// hidden-UI convention.
    ///
    /// Clears `last_anchor_seen` / `last_confidence` on a successful
    /// switch. These are anchor-specific (hip pair vs shoulder pair)
    /// and would otherwise display stale "anchor visible ✓" reflecting
    /// the *previous* mode's anchor for one frame. Resetting forces the
    /// indicator to ✗ until the next mailbox snapshot confirms the new
    /// anchor type is in frame — that flicker is the visual "your mode
    /// switch was registered" feedback.
    pub fn set_mode(&mut self, new_mode: CalibrationMode) {
        if let CalibrationModalState::Idle {
            mode,
            last_anchor_seen,
            last_confidence,
        } = self
        {
            *mode = new_mode;
            *last_anchor_seen = false;
            *last_confidence = 0.0;
        }
    }

    pub fn close(&mut self) {
        *self = CalibrationModalState::Closed;
    }
}

/// Mode currently driving the modal, if any. Used by the panes (mode
/// picker, anchor-visibility check) and the refresh path (which
/// anchor pair to look for).
pub(super) fn relevant_mode(state: &CalibrationModalState) -> Option<CalibrationMode> {
    match state {
        CalibrationModalState::Idle { mode, .. }
        | CalibrationModalState::WaitingForPose { mode, .. }
        | CalibrationModalState::Collecting { mode, .. }
        | CalibrationModalState::AnchorDone { mode, .. }
        | CalibrationModalState::RangeHoldStill { mode, .. }
        | CalibrationModalState::RangeCollecting { mode, .. }
        | CalibrationModalState::Done { mode, .. } => Some(*mode),
        CalibrationModalState::Closed => None,
    }
}
