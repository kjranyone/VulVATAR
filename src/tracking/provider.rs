//! Pose provider boundary.
//!
//! `TrackingWorker` talks to this module instead of binding directly to
//! a specific model. There is a single production pipeline: RTMW3D
//! (body + hands + face). Metric depth is supplied externally by a
//! RealSense D435 (`set_external_depth`, `realsense` feature); without
//! it the provider runs RTMW3D-only and z falls back to RTMW3D's
//! body-prior synthetic.

use std::path::Path;

use super::rtmw3d_with_depth::Rtmw3dWithDepthProvider;
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

    /// Toggle per-frame torso depth capture on/off. While enabled the
    /// depth stage accumulates one `TorsoDepthGridFrame` per inference
    /// frame (subject to the visibility-floor gate) into an internal
    /// buffer. The GUI calibration-modal flips this on at the start of
    /// `Collecting`, off when the window closes, then calls
    /// [`take_torso_template`] to harvest the median-aggregated result.
    ///
    /// **Why a stateful capture mode rather than streaming the depth
    /// map every frame**: the depth map is large (frame.width ×
    /// frame.height × 12 bytes for the point cloud) and the calibration
    /// window only happens for ~2 seconds out of an entire session.
    /// Keeping the capture local to the provider means we don't pay
    /// the per-frame mailbox-cloning cost for the 99.9% of frames
    /// where no calibration is in flight.
    ///
    /// [`take_torso_template`]: Self::take_torso_template
    fn set_torso_capture(&mut self, _enabled: bool) {}

    /// Drain the accumulated torso depth template from the provider
    /// and return its median-aggregated form, or `None` when no
    /// frames cleared the visibility floor during the most recent
    /// capture window (or the depth stage is disabled). Resets the
    /// internal buffer so a subsequent capture starts clean.
    ///
    /// The GUI calibration-modal calls this once when transitioning
    /// `Collecting → AnchorDone`, stitches the result onto the
    /// `PoseCalibration` it just finalized, and re-publishes the
    /// enriched calibration via the mailbox.
    fn take_torso_template(&mut self) -> Option<crate::tracking::TorsoDepthTemplate> {
        None
    }

    /// Reset per-session temporal state: wrist temporal holds,
    /// smoothing EMAs, self-tracking crop. Called between *unrelated*
    /// inputs — `validate_pipeline` calls this before every image so
    /// image N's temporal holds can't contaminate image N+1's
    /// skeleton. Live tracking never calls it mid-session.
    fn reset_temporal_state(&mut self) {}

    /// Hint from the GUI about the calibration mode the user is *currently
    /// collecting* (modal open, samples about to flow), independent of any
    /// confirmed `PoseCalibration`. Used to bridge the gap where
    /// [`Self::set_calibration`] only fires *after* `persist_calibration`:
    /// during the very first `UpperBody` capture there is no confirmed
    /// calibration yet, so without this hint the provider would still
    /// hallucinate hip keypoints with high confidence and the GUI would
    /// reject every collected sample (`pose.root_anchor_is_hip == true`).
    ///
    /// Contract: implementations MUST treat the hint as an **override**
    /// of the persisted calibration's mode, in either direction:
    ///
    /// * `Some(UpperBody)` → force shoulder-anchor framing, even if the
    ///   persisted calibration is `FullBody` or absent.
    /// * `Some(FullBody)` → do **not** force shoulder anchor, even if
    ///   the persisted calibration is `UpperBody`. (OR semantics break
    ///   here: re-calibrating from `UpperBody` to `FullBody` would be
    ///   impossible because the persisted flag would keep suppressing
    ///   hip and the GUI's `pose.root_anchor_is_hip` gate would reject
    ///   every collected sample.)
    /// * `None` → modal closed, fall back to the persisted calibration's
    ///   mode.
    fn set_calibration_mode_hint(&mut self, _hint: Option<crate::tracking::CalibrationMode>) {}

    /// Supply a metric depth frame captured by an external sensor (e.g. a
    /// RealSense D435) for the *next* [`Self::estimate_pose`] call,
    /// replacing the internal DAv2 depth stage. The tracking worker calls
    /// this each frame with depth aligned to the color image it is about
    /// to hand to `estimate_pose`. Providers without a depth stage ignore
    /// it. Consumed once: the provider clears it after the next estimate.
    ///
    /// Gated on `inference`, not `realsense`: the depth is consumed by the
    /// inference-stage skeleton lift (`skeleton_from_depth`), and offline
    /// benches inject a recorded/synthetic `MetricDepthFrame` here without
    /// the native realsense toolchain. Only the live D435 *source*
    /// (`build_metric_frame_from_d435`) needs the `realsense` feature.
    #[cfg(feature = "inference")]
    fn set_external_depth(
        &mut self,
        _depth: crate::tracking::skeleton_from_depth::MetricDepthFrame,
    ) {
    }
}

/// Build the production pose provider: RTMW3D, shaped by the user's
/// pipeline configuration. Metric depth, when present, is fed in
/// externally via [`PoseProvider::set_external_depth`].
pub fn create_pose_provider(
    models_dir: impl AsRef<Path>,
    config: TrackingPipelineConfig,
) -> Result<Box<dyn PoseProvider>, String> {
    Rtmw3dWithDepthProvider::from_models_dir_with_config(models_dir, config)
        .map(|p| Box::new(p) as Box<dyn PoseProvider>)
}
