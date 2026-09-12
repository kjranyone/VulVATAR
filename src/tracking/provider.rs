//! Pose provider boundary.
//!
//! `TrackingWorker` talks to this module instead of binding directly to
//! a specific model. There is a single production pipeline: RTMW3D
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
    /// Run the YOLOX person-crop stage; off falls back to whole-frame
    /// RTMW3D inference.
    pub yolox_enabled: bool,
}

impl Default for TrackingPipelineConfig {
    fn default() -> Self {
        Self {
            force_cpu: false,
            yolox_enabled: true,
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
            yolox_enabled: false,
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

    /// Update the provider's view of the user's pose calibration. The
    /// tracking worker calls this each iteration with the latest value
    /// from `Application::tracking_calibration.pose` (forwarded via
    /// the calibration mailbox); the depth pipeline uses it to switch
    /// anchor selection (`Upper Body` mode forces shoulder anchor) and
    /// to clamp the metric calibration scale against the captured
    /// jitter range.
    fn set_calibration(&mut self, _calibration: Option<crate::tracking::PoseCalibration>) {}

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
}

/// Build the production pose provider: RTMW3D, shaped by the user's
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
