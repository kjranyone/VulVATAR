#[cfg(feature = "realsense")]
pub mod realsense;
/// Full-rate raw-capture recorder (colour + aligned depth + intrinsics) for
/// offline estimator evaluation. Needs the D435 frame type, hence the gate.
#[cfg(feature = "realsense")]
pub mod sequence_recorder;

pub(crate) mod latest_cell;
#[cfg(feature = "inference")]
pub mod metric_frame;
mod pose_estimation;
pub mod provider;
pub mod stagelog;

pub mod debug_channel;
pub mod devices;
pub mod face_mediapipe;
pub mod face_rtmtface;
pub mod fusion;
pub mod mailbox;
pub mod detector;
pub mod session_record;
pub mod source_skeleton;
pub mod worker;
#[cfg(feature = "inference")]

pub use devices::{
    camera_fps_for_index, camera_fps_index_for, camera_resolution_for_index,
    camera_resolution_index_for, d400_product_name, enumerate_cameras, usable_capture_device,
    usb_link_too_slow, CameraDeviceInfo,
};
pub use mailbox::{DetectionAnnotation, HandCropDiag, MailboxSnapshot, PreviewFrame, TrackingMailbox};
pub use source_skeleton::{
    CameraIntrinsics, FacePose, FaceSource, MetricFrameInfo, SourceExpression, SourceJoint,
    SourceSkeleton,
};
pub use worker::{PoseEstimate, TrackingSource, TrackingWorker, CAPTURE_BACKEND_LABEL};

/// Smoothing / threshold params consumed by
/// [`crate::avatar::retarget::apply_rig_pose`] via
/// [`RetargetParams`](crate::avatar::retarget::RetargetParams).
///
/// The [`Default`] values below are tuned to lean on the fusion
/// estimator's process noise (see the `Default` impl), so most users
/// never need to touch these. They are now surfaced in the Tracking
/// inspector's *Advanced smoothing* section for the per-camera cases the
/// defaults don't cover (jittery expression rigs): the GUI holds the
/// live values on `TrackingGuiState::smoothing` and passes them through
/// `FrameConfig::smoothing` each frame. The GUI does not expose
/// `stale_timeout_nanos` — it is a hold-policy timing knob, not a
/// smoothing control — so it always keeps its default.
#[derive(Clone, Debug)]
pub struct TrackingSmoothingParams {
    /// Per-frame blend factor toward the new rotation. Maps directly to
    /// `RetargetParams::rotation_blend`.
    pub rotation_blend: f32,
    /// Render-rate rig interpolation (staircase removal). The estimator
    /// publishes at its compute-bound cadence (~20-30 Hz, solve-time
    /// bound); with this on, `run_frame` slerps between the last two rig
    /// samples every render frame (see `avatar::pose_timeline`) instead of
    /// replaying the newest sample as-is. Off = the pre-interpolation
    /// behaviour exactly.
    pub pose_interp_enabled: bool,
    /// Interpolation delay as a fraction of the latest inter-sample
    /// interval. 0.5 renders half an interval behind the newest sample
    /// and extrapolates (bounded) across the rest — low added latency;
    /// 1.0 is pure interpolation with a full interval of latency and no
    /// steady-state extrapolation.
    pub pose_interp_delay_frac: f32,
    /// Per-frame blend factor toward new expression weights.
    pub expression_blend: f32,
    /// Minimum face-pose confidence for the head to react.
    pub face_confidence_threshold: f32,
    pub stale_timeout_nanos: u64,
}

impl Default for TrackingSmoothingParams {
    fn default() -> Self {
        // `rotation_blend = 1.0` snaps each frame straight to the
        // retarget output. The fusion estimator's process noise already
        // smooths jitter adaptively, so a separate per-frame rotation
        // LPF on top just adds blanket lag. Joint gating is the
        // retarget's σ-rest (not a keypoint-confidence floor).
        // `expression_blend` stays smoothed because there is no
        // equivalent filter on expression weights — without this LPF,
        // ARKit blendshapes chatter visibly.
        Self {
            rotation_blend: 1.0,
            pose_interp_enabled: true,
            pose_interp_delay_frac: 0.5,
            expression_blend: 0.8,
            face_confidence_threshold: 0.0,
            stale_timeout_nanos: 200_000_000,
        }
    }
}

/// Which signal drives the avatar's mouth visemes (`aa`/`ih`/`ou`/`ee`/`oh`).
/// Audio lip-sync and the camera (FaceMesh) both produce mouth shapes; this
/// selects how they combine so the camera-based path doesn't silently
/// override audio (or vice-versa). Only the mouth visemes are affected —
/// eyes / brows / emotions always come from the camera. Consumed by
/// [`crate::avatar::expressions::solve_expressions`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MouthSource {
    /// Audio lip-sync only — the camera's mouth visemes are ignored.
    Audio,
    /// Camera (FaceMesh / "image lip-sync") only — audio is ignored.
    Image,
    /// Whichever is stronger per viseme (`max`) — mouth opens for speech
    /// *or* a visibly open mouth. The default.
    #[default]
    Both,
}

impl MouthSource {
    /// Stable index for GUI combo boxes / persistence. 0=Audio, 1=Image, 2=Both.
    pub fn to_index(self) -> usize {
        match self {
            MouthSource::Audio => 0,
            MouthSource::Image => 1,
            MouthSource::Both => 2,
        }
    }
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => MouthSource::Audio,
            1 => MouthSource::Image,
            _ => MouthSource::Both,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TrackingErrorLevel {
    Warning,
    Blocking,
}
