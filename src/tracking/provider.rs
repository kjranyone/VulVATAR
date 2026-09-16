//! Pose provider boundary.
//!
//! `TrackingWorker` talks to this module instead of binding directly to
//! a specific model. There is a single production pipeline: YOLO11-pose
//! perception feeding the fusion estimator. Metric depth is supplied
//! externally by a RealSense D435 (`set_external_depth`); without it
//! the estimator still runs on 2-D keypoints alone.

use std::path::Path;

use super::PoseEstimate;

/// User-facing pipeline configuration, bound once at tracking start.
/// Surfaced as toggles in the Tracking inspector and persisted with
/// the project; the safe-mode banner substitutes
/// [`TrackingPipelineConfig::safe_mode`] after an unclean exit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrackingPipelineConfig {
    /// Force every ONNX session onto the CPU EP, keeping DirectML —
    /// and the GPU driver's compute queue — out of the tracking
    /// pipeline entirely. Slower, but isolates tracking from
    /// GPU-driver instability (see the 2026-06-11 freeze incident).
    pub force_cpu: bool,
}

impl Default for TrackingPipelineConfig {
    fn default() -> Self {
        Self {
            force_cpu: false,
        }
    }
}

impl TrackingPipelineConfig {
    /// Degraded configuration offered after an unclean exit: no
    /// DirectML sessions, no depth stage, no person crop. Trades
    /// accuracy and latency for the most conservative driver load.
    pub fn safe_mode() -> Self {
        Self {
            force_cpu: true,
        }
    }
}

/// Runtime pose-estimation implementation used by the tracking worker.
pub trait PoseProvider {
    /// User-visible provider/backend label.
    fn label(&self) -> String;

    /// Warnings collected during provider setup. They are shown once by the UI.
    fn take_load_warnings(&mut self) -> Vec<String> {
        Vec::new()
    }

    fn estimate_pose(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> PoseEstimate;

    /// Reset per-session temporal state (self-tracking crop, YOLOX
    /// sticky result, fusion estimator). Called between *unrelated*
    /// inputs — `validate_gt` calls this before every pose so pose N's
    /// temporal state can't contaminate pose N+1. Live tracking never
    /// calls it mid-session.
    fn reset_temporal_state(&mut self) {}

    /// Supply a metric depth frame captured by an external sensor (e.g. a
    /// RealSense D435) for the *next* [`Self::estimate_pose`] call,
    /// replacing any previous depth. The tracking worker calls
    /// this each frame with depth aligned to the color image it is about
    /// to hand to `estimate_pose`. Providers without a depth stage ignore
    /// it. Consumed once: the provider clears it after the next estimate.
    ///
    /// Gated on `inference`, not `realsense`: the depth is consumed by the
    /// fusion estimator, and offline
    /// benches inject a recorded/synthetic `MetricDepthFrame` here without
    /// the native realsense toolchain. Only the live D435 *source*
    /// (`build_metric_frame_from_d435`) needs the `realsense` feature.
    #[cfg(feature = "inference")]
    fn set_external_depth(&mut self, _depth: crate::tracking::metric_frame::MetricDepthFrame) {}

    /// Live-tracking entry point that tolerates a pipelined provider:
    /// submit THIS frame's detector work and return the freshest finished
    /// result, or `None` when the detector has nothing new yet — the
    /// worker then skips publishing for that capture and the camera paces
    /// the retry. The returned estimate may describe an EARLIER frame
    /// than `frame_index` (the published pose carries its own
    /// `capture_timestamp_ms` — consumers must not overwrite it with the
    /// capture's time). The default delegates to [`Self::estimate_pose`],
    /// which never skips.
    fn estimate_pose_latest(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> Option<PoseEstimate> {
        Some(self.estimate_pose(rgb_data, width, height, frame_index))
    }
}

/// Build the production pose provider: YOLO11-pose, shaped by the user's
/// pipeline configuration. Metric depth, when present, is fed in
/// externally via [`PoseProvider::set_external_depth`].
pub fn create_pose_provider(
    models_dir: impl AsRef<Path>,
    config: TrackingPipelineConfig,
) -> Result<Box<dyn PoseProvider>, String> {
    #[cfg(feature = "inference")]
    {
        super::fusion::provider::FusionProvider::from_models_dir_with_config(models_dir, config)
            .map(|p| Box::new(p) as Box<dyn PoseProvider>)
    }
    #[cfg(not(feature = "inference"))]
    {
        let _ = (models_dir, config);
        Err("pose provider requires the `inference` cargo feature".to_string())
    }
}

/// Live-tracking variant of [`create_pose_provider`]: runs the detector
/// stage (YOLO11-pose + FaceMesh) on its own thread so it overlaps the solver
/// stage — live tracking is camera-paced, and the two stages together
/// (~48 ms) exceed the 33 ms frame period while each alone does not.
/// The pair with [`PoseProvider::estimate_pose_latest`].
///
/// Falls back to the synchronous provider in safe mode (`force_cpu`) or
/// with `VULVATAR_NO_PIPELINE=1` (rollback switch). Offline harnesses
/// keep using [`create_pose_provider`], whose behaviour is unchanged.
pub fn create_pose_provider_live(
    models_dir: impl AsRef<Path>,
    config: TrackingPipelineConfig,
) -> Result<Box<dyn PoseProvider>, String> {
    if std::env::var_os("VULVATAR_NO_PIPELINE").is_some() {
        return create_pose_provider(models_dir, config);
    }
    #[cfg(feature = "inference")]
    {
        super::fusion::provider::FusionProvider::from_models_dir_live(models_dir, config)
            .map(|p| Box::new(p) as Box<dyn PoseProvider>)
    }
    #[cfg(not(feature = "inference"))]
    {
        let _ = (models_dir, config);
        Err("pose provider requires the `inference` cargo feature".to_string())
    }
}
