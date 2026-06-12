//! Pose provider boundary.
//!
//! `TrackingWorker` talks to this module instead of binding directly to
//! a specific model. There is a single production pipeline: RTMW3D
//! (body + hands + face) with an async Depth Anything V2 metric-depth
//! stage. When `dav2_small.onnx` is absent the provider runs the same
//! pipeline without the depth stage (z falls back to RTMW3D's
//! body-prior synthetic), so one provider covers both the fast and the
//! accurate configuration.

use std::path::Path;

use super::rtmw3d_with_depth::Rtmw3dWithDepthProvider;
use super::PoseEstimate;

/// User-facing pipeline configuration, bound once at tracking start.
/// Surfaced as toggles in the Tracking inspector and persisted with
/// the project; the safe-mode banner substitutes
/// [`TrackingPipelineConfig::safe_mode`] after an unclean exit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrackingPipelineConfig {
    /// Run the async DAv2 metric-depth stage (requires
    /// `models/dav2_small.onnx`; missing model degrades with a
    /// visible warning).
    pub depth_enabled: bool,
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
            depth_enabled: true,
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
            depth_enabled: false,
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

    /// Reset per-session temporal state: wrist temporal holds, the
    /// sticky depth outbox, smoothing EMAs, self-tracking crop.
    /// Called between *unrelated* inputs — `validate_pipeline` calls
    /// this before every image so image N's sticky depth map and
    /// temporal holds can't contaminate image N+1's skeleton (the
    /// depth path otherwise reuses the previous image's depth for 3
    /// of every 4 frames via `DEPTH_REFRESH_PERIOD`). Live tracking
    /// never calls it mid-session.
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
}

/// Build the production pose provider: RTMW3D with the async DAv2
/// metric-depth stage, shaped by the user's pipeline configuration.
pub fn create_pose_provider(
    models_dir: impl AsRef<Path>,
    config: TrackingPipelineConfig,
) -> Result<Box<dyn PoseProvider>, String> {
    Rtmw3dWithDepthProvider::from_models_dir_with_config(models_dir, config)
        .map(|p| Box::new(p) as Box<dyn PoseProvider>)
}
