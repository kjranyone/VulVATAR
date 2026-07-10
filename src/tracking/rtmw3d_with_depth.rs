//! RTMW3D pose provider — the single production pose pipeline, sized for
//! 30-fps virtual-camera output.
//!
//! RTMW3D's 2D body / hand / face landmarks are excellent and run in
//! ~38 ms; its baked-in z is a body-prior fiction that doesn't capture
//! forward / backward limb motion. Metric depth for the arm / hand
//! chains comes from a RealSense D435 (`realsense` feature): the
//! tracking worker aligns depth to the color frame and hands it in via
//! `PoseProvider::set_external_depth` before each `estimate_pose`. The
//! frame is absolute metres with a true principal point, so
//! `estimate_from_external_depth` back-projects the skeleton straight
//! from it — no scale calibration, no lateral-bias correction. Without
//! external depth the provider runs RTMW3D-only and z stays the
//! body-prior synthetic.

use std::path::Path;

use super::provider::PoseProvider;
use super::{DetectionAnnotation, PoseEstimate, SourceSkeleton};

#[cfg(feature = "inference")]
use super::rtmw3d::Rtmw3dInference;
#[cfg(feature = "inference")]
use super::skeleton_from_depth::{DecodedJoint2d, NUM_JOINTS};
#[cfg(feature = "realsense")]
use super::skeleton_from_depth::{
    build_options_from_calibration, build_skeleton, torso_fit, MetricDepthFrame,
};
#[cfg(feature = "inference")]
use log::info;
#[cfg(feature = "realsense")]
use log::warn;

pub struct Rtmw3dWithDepthProvider {
    #[cfg(feature = "inference")]
    rtmw3d: Rtmw3dInference,
    load_warnings: Vec<String>,
    /// Latest pose calibration pushed from the GUI via the tracking
    /// mailbox. `None` means no calibration is active — the depth
    /// pipeline keeps its default hip-preferred / shoulder-fallback
    /// anchor selection. When `Some` and the mode is `UpperBody`,
    /// `BuildOptions::force_shoulder_anchor` is set so a hallucinated
    /// hip detection (e.g. desk surface in foreground) can't override
    /// the user-acknowledged "I'm only showing my upper body"
    /// declaration.
    pose_calibration: Option<crate::tracking::PoseCalibration>,
    /// Transient mode hint pushed by the GUI while the calibration
    /// modal is open. **Overrides** `pose_calibration.mode` for the
    /// `force_shoulder_anchor` decision so re-calibration in either
    /// direction works (see
    /// [`super::skeleton_from_depth::build_options_from_calibration`]
    /// for the full contract). `None` whenever the modal is closed.
    calibration_mode_hint: Option<crate::tracking::CalibrationMode>,
    /// Active torso-template capture buffer, set by the GUI while
    /// the calibration modal is in `Collecting`. `Some` means each
    /// successful frame contributes one sample per cell to the
    /// running median; `None` means the per-frame capture path is
    /// skipped (no overhead in normal streaming).
    #[cfg(feature = "inference")]
    torso_capture: Option<super::skeleton_from_depth::TorsoCaptureBuffer>,
    /// Metric depth frame supplied by an external sensor (RealSense D435)
    /// for the *next* `estimate_pose`, via
    /// [`super::provider::PoseProvider::set_external_depth`]. When `Some`,
    /// the entire DAv2 acquisition + scale-calibration + Phase-7 lateral-
    /// bias-correction path is bypassed: the frame is already absolute
    /// metric depth aligned to the color image with a true principal
    /// point, so `estimate_from_external_depth` builds the skeleton
    /// straight from it. Taken (consumed) each frame.
    #[cfg(feature = "realsense")]
    external_depth: Option<super::skeleton_from_depth::MetricDepthFrame>,
}

impl Rtmw3dWithDepthProvider {
    #[cfg(feature = "inference")]
    pub fn from_models_dir(models_dir: impl AsRef<Path>) -> Result<Self, String> {
        Self::from_models_dir_with_config(
            models_dir,
            super::provider::TrackingPipelineConfig::default(),
        )
    }

    #[cfg(feature = "inference")]
    pub fn from_models_dir_with_config(
        models_dir: impl AsRef<Path>,
        config: super::provider::TrackingPipelineConfig,
    ) -> Result<Self, String> {
        let dir = models_dir.as_ref();

        // No internal depth model: metric depth arrives externally from a
        // RealSense D435 via `set_external_depth`. FaceMesh runs on the
        // Auto EP (DirectML ~3 ms) — with no depth model there is no GPU
        // command-queue contention. (`force_cpu` overrides this inside
        // `Rtmw3dInference`.)
        let mut rtmw3d = Rtmw3dInference::from_models_dir_with_options(
            dir,
            super::rtmw3d::Rtmw3dOptions {
                face_ep: super::face_mediapipe::FaceMeshEp::Auto,
                force_cpu: config.force_cpu,
                yolox_enabled: config.yolox_enabled,
            },
        )?;
        let warnings = rtmw3d.take_load_warnings();
        info!("RTMW3D provider ready ({})", rtmw3d.backend().label());

        Ok(Self {
            rtmw3d,
            load_warnings: warnings,
            pose_calibration: None,
            calibration_mode_hint: None,
            torso_capture: None,
            #[cfg(feature = "realsense")]
            external_depth: None,
        })
    }

    #[cfg(not(feature = "inference"))]
    pub fn from_models_dir(_: impl AsRef<Path>) -> Result<Self, String> {
        Err("RTMW3D provider requires the `inference` cargo feature".to_string())
    }
}

impl PoseProvider for Rtmw3dWithDepthProvider {
    fn label(&self) -> String {
        #[cfg(feature = "inference")]
        {
            format!("RTMW3D / {}", self.rtmw3d.backend().label())
        }
        #[cfg(not(feature = "inference"))]
        {
            "RTMW3D (inference disabled)".to_string()
        }
    }

    fn take_load_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.load_warnings)
    }

    fn set_calibration(&mut self, calibration: Option<crate::tracking::PoseCalibration>) {
        // Mirror the persisted `force_shoulder_anchor` onto the inner
        // `Rtmw3dInference` so the fallback paths in
        // `estimate_pose_internal` (RTMW3D keypoint shortage at the
        // top, DAv2 cold-start scale-calibration failure later) see
        // the same anchor contract as the depth-built skeleton. Both
        // fallbacks return `rtmw_est` straight through from the inner
        // RTMW3D, which builds its source skeleton via the override
        // logic in `Rtmw3dInference::process_pose`
        // (`hint Some(UpperBody) → true`,
        // `hint Some(FullBody) → false`,
        // `hint None → persisted force_shoulder_anchor`). Without this
        // mirror the persisted side of that match would always read
        // `false`, so a fallback frame outside the modal would build
        // a hip-anchored skeleton even when the user calibrated
        // `UpperBody`.
        #[cfg(feature = "inference")]
        {
            self.rtmw3d.force_shoulder_anchor = calibration
                .as_ref()
                .map(|c| matches!(c.mode, crate::tracking::CalibrationMode::UpperBody))
                .unwrap_or(false);
        }
        self.pose_calibration = calibration;
    }

    fn set_calibration_mode_hint(&mut self, hint: Option<crate::tracking::CalibrationMode>) {
        // Override the persisted calibration's mode for the
        // `force_shoulder_anchor` decision while the modal is open —
        // see `build_options_from_calibration` for the contract.
        self.calibration_mode_hint = hint;
        // Also mirror onto the inner Rtmw3dInference so fallback frames
        // (see `set_calibration` above for the rationale) honour the
        // same override the depth-built path does.
        #[cfg(feature = "inference")]
        {
            self.rtmw3d.force_shoulder_anchor_hint = hint;
        }
    }

    fn reset_temporal_state(&mut self) {
        #[cfg(feature = "inference")]
        {
            self.rtmw3d.reset_temporal_state();
        }
    }

    fn set_torso_capture(&mut self, enabled: bool) {
        #[cfg(feature = "inference")]
        {
            // Toggling on starts a fresh buffer; toggling off
            // discards any in-flight buffer (the canonical exit is
            // through `take_torso_template`, which moves the buffer
            // out and finalises it). This drop-on-off path covers
            // user-initiated cancel of the calibration modal mid-
            // capture, where we don't want stale partial data to
            // leak into the next capture.
            self.torso_capture = if enabled {
                Some(super::skeleton_from_depth::TorsoCaptureBuffer::new())
            } else {
                None
            };
        }
        #[cfg(not(feature = "inference"))]
        {
            let _ = enabled;
        }
    }

    fn take_torso_template(&mut self) -> Option<crate::tracking::TorsoDepthTemplate> {
        #[cfg(feature = "inference")]
        {
            self.torso_capture.take().and_then(|buf| buf.finalize())
        }
        #[cfg(not(feature = "inference"))]
        {
            None
        }
    }

    fn estimate_pose(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> PoseEstimate {
        #[cfg(feature = "inference")]
        return self.estimate_pose_internal(rgb_data, width, height, frame_index);

        #[cfg(not(feature = "inference"))]
        {
            let _ = (rgb_data, width, height);
            empty_estimate(frame_index)
        }
    }

    #[cfg(feature = "realsense")]
    fn set_external_depth(
        &mut self,
        depth: super::skeleton_from_depth::MetricDepthFrame,
    ) {
        self.external_depth = Some(depth);
    }
}

#[cfg(feature = "inference")]
impl Rtmw3dWithDepthProvider {
    fn estimate_pose_internal(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> PoseEstimate {
        let expected_len = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(3);
        if rgb_data.len() < expected_len || width == 0 || height == 0 {
            return empty_estimate(frame_index);
        }

        // Phase 1: RTMW3D — full pipeline (YOLOX + RTMW3D + FaceMesh).
        // We keep its `annotation.keypoints` (whole-frame normalised 2D
        // + score) and `skeleton.expressions` / `face_mesh_confidence`
        // (the FaceMesh cascade). When external depth is present the rest
        // of `rtmw_est.skeleton` is rebuilt from it below; otherwise the
        // RTMW3D estimate passes through unchanged.
        let rtmw_est = self
            .rtmw3d
            .estimate_pose(rgb_data, width, height, frame_index);

        let joints_2d: Vec<DecodedJoint2d> = rtmw_est
            .annotation
            .keypoints
            .iter()
            .map(|&(nx, ny, score)| DecodedJoint2d { nx, ny, score })
            .collect();
        if joints_2d.len() < NUM_JOINTS {
            // RTMW3D failed to emit a full keypoint set — pass its
            // empty estimate through unchanged.
            return rtmw_est;
        }

        // External metric depth (RealSense D435) supplied for THIS color
        // frame short-circuits the entire DAv2 acquisition + scale-
        // calibration + Phase-7 lateral-bias-correction path: the depth is
        // already absolute metric, aligned to this exact frame, with a
        // true principal point — so build the skeleton straight from it.
        // `take` clears it so a dropped next frame can't reuse stale depth.
        #[cfg(feature = "realsense")]
        if let Some(metric_frame) = self.external_depth.take() {
            return self.estimate_from_external_depth(
                frame_index,
                rtmw_est,
                &joints_2d,
                metric_frame,
            );
        }

        // No external depth this frame: the RTMW3D estimate, with its
        // body-prior synthetic z, IS the result.
        rtmw_est
    }

    /// Build a pose estimate from an externally-supplied metric depth
    /// frame (RealSense D435). Unlike the DAv2 path this needs no scale
    /// calibration (the depth is absolute metres) and no Phase-7 lateral-
    /// bias correction (the metric XY is accurate from the real
    /// intrinsics), so it is just: torso capture -> resolve origin ->
    /// build skeleton -> inherit RTMW3D's face. The DAv2 guess-layer
    /// (ray-IK, arm_z, fold, contact) is intentionally absent — the
    /// metric depth replaces it.
    #[cfg(feature = "realsense")]
    fn estimate_from_external_depth(
        &mut self,
        frame_index: u64,
        mut rtmw_est: PoseEstimate,
        joints_2d: &[DecodedJoint2d],
        metric_frame: super::skeleton_from_depth::MetricDepthFrame,
    ) -> PoseEstimate {
        // No 2D border gate here any more: the torso surface fit (`fit_torso`)
        // is robust to a single border-clamped shoulder — it reads the whole
        // torso point cloud, not that one pixel — so the old symptom patch is
        // gone. `joints_2d` is consumed as-received.

        // Torso-template capture (calibration): retained plumbing. The
        // inference consumers (template-bias) were removed with the geometric
        // fit; this now only feeds the persisted template. Follow-up: retire
        // the capture UI. No-op when no capture window is active.
        if let Some(buf) = self.torso_capture.as_mut() {
            let _ = buf.add_frame(joints_2d, &metric_frame);
        }

        let mut opts = build_options_from_calibration(
            self.pose_calibration.as_ref(),
            self.calibration_mode_hint,
        );
        // RealSense desk-up default: with no pose calibration at all, anchor
        // on the shoulders rather than trusting a phantom desk-edge / chair
        // "hip" — or emitting an empty skeleton when the hips are simply out
        // of frame. An explicit calibration (UpperBody or FullBody) is
        // honoured as-is by `build_options_from_calibration`; this only fills
        // the fully-uncalibrated gap, so a FullBody user is never overridden.
        if self.pose_calibration.is_none() && self.calibration_mode_hint.is_none() {
            opts.force_shoulder_anchor = true;
        }
        let mut skeleton = match torso_fit::fit_torso(&metric_frame, joints_2d, opts) {
            Some(fit) => build_skeleton(
                frame_index,
                joints_2d,
                &metric_frame,
                fit,
                self.pose_calibration.as_ref(),
            ),
            None => {
                warn!("RTMW3D+D435: no torso fit — emitting empty skeleton");
                SourceSkeleton::empty(frame_index)
            }
        };

        // Inherit RTMW3D's face pose + FaceMesh cascade output (the face
        // track is body-derived and unrelated to the depth source).
        skeleton.face = rtmw_est.skeleton.face;
        skeleton.expressions = std::mem::take(&mut rtmw_est.skeleton.expressions);
        skeleton.face_mesh_confidence = rtmw_est.skeleton.face_mesh_confidence;
        if let (Some(ref mut fp), Some(mesh_conf)) =
            (skeleton.face.as_mut(), skeleton.face_mesh_confidence)
        {
            fp.confidence = fp.confidence.max(mesh_conf);
        }

        // Hand orientation: prefer the depth-built palm (metric MCPs); fall
        // back to RTMW3D's only where the depth path produced none. (There is
        // no more Phase-7 position overwrite or metric-arm re-placement — the
        // whole skeleton already carries faithful metric 3D from
        // `build_skeleton`, which is the entire point of this rebuild.)
        if let Some(rt) = rtmw_est.skeleton.left_hand_orientation {
            skeleton.left_hand_orientation.get_or_insert(rt);
        }
        if let Some(rt) = rtmw_est.skeleton.right_hand_orientation {
            skeleton.right_hand_orientation.get_or_insert(rt);
        }

        // No torso-anchor temporal EMA any more: the plane fit's spatial
        // average over hundreds of torso points already delivers a stable
        // per-frame anchor, so the single-pixel jitter the EMA smoothed no
        // longer exists. Residual smoothing, if any, belongs to the solver's
        // 1€ filters, not a patch here.

        PoseEstimate {
            annotation: rtmw_est.annotation,
            skeleton,
        }
    }
}

fn empty_estimate(frame_index: u64) -> PoseEstimate {
    PoseEstimate {
        annotation: DetectionAnnotation {
            keypoints: Vec::new(),
            skeleton: Vec::new(),
            bounding_box: None,
        },
        skeleton: SourceSkeleton::empty(frame_index),
    }
}

/// Build a `MetricDepthFrame` from a RealSense D435 color-aligned frame:
/// deproject every pixel through the *real* camera intrinsics into a
/// metric point cloud (metres, x-right / y-down / z-forward), marking
/// no-return pixels as `NaN` so `sample_metric_point`'s window median
/// rejects them. There is no learned scale and no centered-principal-
/// point assumption: the D435 supplies absolute metres and a true
/// `(cx, cy)`, so this is a straight per-pixel deprojection. `crop` is
/// `None` — the depth is full-frame, aligned 1:1 to the color image the
/// keypoints were detected in.
/// aligned 1:1 to the color image the keypoints were detected in.
#[cfg(feature = "realsense")]
pub fn build_metric_frame_from_d435(
    frame: &crate::tracking::realsense::RealSenseFrame,
) -> MetricDepthFrame {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let mut points_m = Vec::with_capacity(w * h);
    for v in 0..h {
        for u in 0..w {
            match frame.point_m(u, v) {
                Some(p) => points_m.push(p),
                None => points_m.push([f32::NAN, f32::NAN, f32::NAN]),
            }
        }
    }
    let intr = frame.intrinsics;
    MetricDepthFrame {
        width: frame.width,
        height: frame.height,
        points_m,
        crop: None,
        intrinsics: Some(crate::tracking::source_skeleton::CameraIntrinsics {
            fx: intr.fx,
            fy: intr.fy,
            cx: intr.cx,
            cy: intr.cy,
            width: intr.width,
            height: intr.height,
        }),
    }
}
