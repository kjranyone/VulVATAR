//! Pose-calibration fullscreen modal.
//!
//! State machine: `Closed → Idle → WaitingForPose → Collecting →
//! AnchorDone → (RangeHoldStill → RangeCollecting →)? Done → Closed`.
//! See `state.rs` for the variant-by-variant contract.
//!
//! Per-frame samples are accumulated from the source-skeleton mailbox
//! during the `Collecting` window, the per-axis **median** + per-frame
//! **stddev** are computed on transition to `Done`, and the resulting
//! [`crate::tracking::PoseCalibration`] is written into
//! `Application::tracking_calibration.pose` so persistence (auto-save
//! + project file) picks it up on the next round-trip.
//!
//! The capture rejects low-quality frames at intake — anchor must be
//! detected, the anchor type must match the chosen mode, and the
//! source-skeleton overall confidence must clear `MIN_FRAME_CONF`.
//! "Capture Now" finishes early but still requires `MIN_SAMPLES`
//! gathered samples to avoid writing a 1-frame "calibration" that's
//! no better than the auto-EMA's first frame.
//!
//! The modal is a true overlay: a dimmed full-viewport rect underneath
//! a centred `egui::Window` so all background interaction (orbit
//! controls, inspector clicks) is blocked while calibration is active.
//!
//! Module layout:
//! - [`state`]       — state-machine enum, sample types, simple impls.
//! - [`pose_match`]  — live pose vs. target-pose scoring (drives the
//!   `WaitingForPose → Collecting` gate).
//! - [`transitions`] — state-graph edges (begin / finish / retry /
//!   skip / range / time-driven advance).
//! - [`finalize`]    — sample aggregation + writing the resulting
//!   [`PoseCalibration`] into Application + active profile + mailbox.
//! - [`refresh`]     — per-frame mailbox snapshot fold-in (anchor
//!   visibility, confidence, pose-match score, sample collection).
//! - [`panes`]       — egui rendering, public `draw_modal` entry
//!   point. The other modules are pure logic; this is the only module
//!   that knows about `egui` types.

mod finalize;
mod panes;
mod pose_match;
mod refresh;
mod state;
mod target_pose;
mod transitions;

pub use panes::draw_modal;
pub use state::CalibrationModalState;
pub use target_pose::target_pose_skinning_matrices;

/// Anchor-capture window duration after the pose is recognised. At
/// 30 fps this yields ~60 sample frames; the median over that window
/// absorbs per-frame keypoint jitter.
pub(super) const COLLECTION_SECONDS: f32 = 2.0;

/// Brief prep countdown before the optional range capture. The user
/// already settled into the modal for step 1, so step 2 only needs a
/// beat to start moving.
pub(super) const RANGE_HOLD_STILL_SECONDS: f32 = 1.5;

/// Range-capture window. Long enough for the user to step left,
/// right, lean in, lean back without rushing — the per-axis min/max
/// is what gets recorded so a single full sweep is enough.
pub(super) const RANGE_COLLECTION_SECONDS: f32 = 10.0;

/// Minimum source-skeleton overall confidence for a sample to be
/// admitted into the median pool. Below this, the keypoints are too
/// noisy to give a reliable anchor reading and would skew the
/// aggregate even if the median weights down individual outliers.
pub(super) const MIN_FRAME_CONF: f32 = 0.5;

/// Smallest sample count the modal will accept as a valid capture.
/// Below this the auto-EMA's first-frame seed is no worse than what
/// the median can offer, so we reject and force the user to retry.
pub(super) const MIN_SAMPLES: usize = 5;

/// How long the relevant anchor (hip pair / shoulder pair) can be
/// missing in `WaitingForPose` before the modal surfaces a "your
/// selected mode's anchor isn't visible — switch?" hint. Picked so a
/// brief framing wobble doesn't trigger the hint, but a sustained
/// "I picked Full Body but I'm sitting at a desk" misconfiguration
/// surfaces within a few seconds.
pub(super) const NO_ANCHOR_HINT_SECONDS: f32 = 4.0;
