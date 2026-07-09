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
#[cfg(feature = "realsense")]
use crate::asset::HumanoidBone;

#[cfg(feature = "inference")]
use super::rtmw3d::Rtmw3dInference;
#[cfg(feature = "inference")]
use super::skeleton_from_depth::{DecodedJoint2d, NUM_JOINTS};
#[cfg(feature = "realsense")]
use super::skeleton_from_depth::{
    build_options_from_calibration, build_skeleton, resolve_origin_metric, MetricDepthFrame,
};
#[cfg(feature = "inference")]
use log::info;
#[cfg(feature = "realsense")]
use log::{debug, warn};

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
    /// Last-valid metric forearm vector (source units, elbow→wrist)
    /// per side, with its age in frames. DAv2 refreshes every 2–4
    /// frames; on the STALE frames in between, the depth builder
    /// samples the current keypoints against the old depth map and
    /// the forearm sample is garbage, failing `fa_valid` in
    /// `replace_arm_chains_from_metric`. The old fallback reverted to
    /// the nz-flat reconstruction, which for a raised palm pointing
    /// at the camera is wrong — so the wrist z flip-flopped forward
    /// (fresh) ↔ flat (stale) at the refresh period, snapping a
    /// raised hand toward the face every other frame (caught only by
    /// real-footage replay, f96–136). Holding the last-valid metric
    /// direction across the gap keeps the depth dimension coherent
    /// while the elbow still tracks in 2D every frame.
    #[cfg(feature = "inference")]
    metric_forearm_hold: [Option<([f32; 3], u32)>; 2],
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
            metric_forearm_hold: [None, None],
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
            self.metric_forearm_hold = [None, None];
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
        // Torso-template capture (calibration), same visibility-gated path
        // as the DAv2 branch. No-op when no capture window is active.
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
        let mut skeleton = match resolve_origin_metric(
            joints_2d,
            &metric_frame,
            opts,
            self.pose_calibration.as_ref(),
        ) {
            Some(anchor) => build_skeleton(
                frame_index,
                joints_2d,
                &metric_frame,
                anchor,
                opts,
                self.pose_calibration.as_ref(),
            ),
            None => {
                warn!("RTMW3D+D435: no body anchor — emitting empty skeleton");
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

        // `build_skeleton` emits camera-metric positions (metres, y-down).
        // The pose solver expects the normalised source frame and reads
        // `position[xyz]` (it never reads `metric_depth_m`), so — exactly
        // as the retired DAv2 path did — map into the source frame here.
        // These two phases are NOT DAv2-specific: they turn *any* metric
        // point cloud into a solver-ready skeleton.

        // Snapshot the metric arm geometry + shoulder span BEFORE the
        // overwrite: the metric relative vectors (elbow-shoulder,
        // wrist-elbow) are the perspective-true depth signal, and the
        // shoulder metric-X gives the metres-per-source-unit conversion.
        let shoulder_metric_x_pre = {
            let l = skeleton
                .joints
                .get(&HumanoidBone::LeftUpperArm)
                .map(|j| j.position[0]);
            let r = skeleton
                .joints
                .get(&HumanoidBone::RightUpperArm)
                .map(|j| j.position[0]);
            l.zip(r)
        };
        let metric_arms = snapshot_metric_arms(&skeleton);

        // Phase 7: overwrite each joint's position with RTMW3D's own
        // (unbiased, normalised) source coordinates; UNION in any joint the
        // depth builder dropped so a missing metric sample never deletes a
        // limb RTMW3D tracked fine; inherit hand orientation.
        for (bone, joint) in skeleton.joints.iter_mut() {
            if let Some(rtmw_joint) = rtmw_est.skeleton.joints.get(bone) {
                joint.position = rtmw_joint.position;
            }
        }
        for (bone, joint) in skeleton.fingertips.iter_mut() {
            if let Some(rtmw_joint) = rtmw_est.skeleton.fingertips.get(bone) {
                joint.position = rtmw_joint.position;
            }
        }
        for (bone, joint) in rtmw_est.skeleton.joints.iter() {
            skeleton.joints.entry(*bone).or_insert(*joint);
        }
        for (bone, joint) in rtmw_est.skeleton.fingertips.iter() {
            skeleton.fingertips.entry(*bone).or_insert(*joint);
        }
        if let Some(rt) = rtmw_est.skeleton.left_hand_orientation {
            skeleton.left_hand_orientation = Some(rt);
        }
        if let Some(rt) = rtmw_est.skeleton.right_hand_orientation {
            skeleton.right_hand_orientation = Some(rt);
        }

        // Phase 7.6: re-place the arm chains from the metric relative
        // vectors — perspective-true elbow/wrist in all three axes, which
        // is the whole point of the depth camera. Unlike the DAv2 path this
        // needs no lateral-bias correction, so the DAv2-band `inject_*_bz`
        // hacks are intentionally omitted; the D435 metric samples are used
        // directly.
        if let Some((sl_metric_x, sr_metric_x)) = shoulder_metric_x_pre {
            replace_arm_chains_from_metric(
                &mut skeleton,
                &metric_arms,
                sl_metric_x,
                sr_metric_x,
                joints_2d,
                &mut self.metric_forearm_hold,
            );
        }

        PoseEstimate {
            annotation: rtmw_est.annotation,
            skeleton,
        }
    }
}

/// Metric (depth-built) arm-chain snapshot taken before the Phase-7
/// position overwrite: per side, the shoulder / elbow / wrist
/// camera-relative source positions as the depth back-projection
/// produced them, plus per-joint confidence.
#[cfg(feature = "realsense")]
#[derive(Default)]
struct MetricArmSnapshot {
    /// [left, right] → (shoulder, elbow, wrist) positions.
    sides: [Option<([f32; 3], [f32; 3], [f32; 3])>; 2],
}

#[cfg(feature = "realsense")]
fn snapshot_metric_arms(sk: &SourceSkeleton) -> MetricArmSnapshot {
    const MIN_CONF: f32 = 0.4;
    let mut out = MetricArmSnapshot::default();
    for (i, (sh, el, wr)) in [
        (
            HumanoidBone::LeftUpperArm,
            HumanoidBone::LeftLowerArm,
            HumanoidBone::LeftHand,
        ),
        (
            HumanoidBone::RightUpperArm,
            HumanoidBone::RightLowerArm,
            HumanoidBone::RightHand,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let g = |b: HumanoidBone, require_metric_flag: bool| -> Option<[f32; 3]> {
            let j = sk.joints.get(&b)?;
            if j.confidence < MIN_CONF {
                return None;
            }
            // Body joints whose position came from a depth sample
            // carry `metric_depth_m`; a fallback-built joint must not
            // masquerade as measured geometry. The wrist is exempt:
            // the depth builder constructs it as the MCP-centroid of
            // metric samples but surfaces no scalar — in the depth
            // skeleton it is depth-built by construction.
            if require_metric_flag {
                j.metric_depth_m?;
            }
            Some(j.position)
        };
        let (gs, ge, gw) = (g(sh, true), g(el, true), g(wr, false));
        debug!(
            "metric-arm snapshot side {}: shoulder={} elbow={} wrist={}",
            i,
            gs.is_some(),
            ge.is_some(),
            gw.is_some()
        );
        if let (Some(s), Some(e), Some(w)) = (gs, ge, gw) {
            out.sides[i] = Some((s, e, w));
        }
    }
    out
}

/// Re-place the elbow and wrist from the metric relative segment
/// vectors. The nz-based shoulder position (Phase 7) stays the
/// anchor; the metric vectors are scaled into source units via the
/// shoulder-span conversion factor and bounded by anatomical segment
/// lengths so a hand-occluded or background-contaminated sample
/// can't fling the chain.
#[cfg(feature = "realsense")]
fn replace_arm_chains_from_metric(
    sk: &mut SourceSkeleton,
    arms: &MetricArmSnapshot,
    shoulder_l_metric_x: f32,
    shoulder_r_metric_x: f32,
    joints_2d: &[DecodedJoint2d],
    forearm_hold: &mut [Option<([f32; 3], u32)>; 2],
) {
    /// Max frames a held metric forearm vector stays usable. DAv2's
    /// refresh period is 2–4 frames; 6 covers a doubled gap (a
    /// dropped depth result) without letting a truly stale direction
    /// persist into a genuinely new arm pose.
    const MAX_FOREARM_HOLD_FRAMES: u32 = 6;
    /// Anatomical bounds on metric segment lengths (metres). The
    /// upper caps reject "sampled the bookshelf" outliers; the lower
    /// floors reject occlusion collapse — a wrist depth sample
    /// landing on the elbow/torso surface yields an impossibly short
    /// segment (observed 0.11 m forearm on the lean-left validation
    /// image) that would fold the avatar's arm flat.
    const MAX_UPPER_ARM_M: f32 = 0.45;
    const MAX_FOREARM_M: f32 = 0.42;
    const MIN_UPPER_ARM_M: f32 = 0.15;
    const MIN_FOREARM_M: f32 = 0.15;

    let metric_x_span = (shoulder_l_metric_x - shoulder_r_metric_x).abs();
    if metric_x_span < 1e-3 {
        return;
    }
    let source_x_span = match (
        sk.joints.get(&HumanoidBone::LeftUpperArm),
        sk.joints.get(&HumanoidBone::RightUpperArm),
    ) {
        (Some(l), Some(r)) => (l.position[0] - r.position[0]).abs(),
        _ => return,
    };
    if source_x_span < 1e-3 {
        return;
    }
    let mpsu = metric_x_span / source_x_span;

    for (i, (sh_bone, el_bone, wr_bone)) in [
        (
            HumanoidBone::LeftUpperArm,
            HumanoidBone::LeftLowerArm,
            HumanoidBone::LeftHand,
        ),
        (
            HumanoidBone::RightUpperArm,
            HumanoidBone::RightLowerArm,
            HumanoidBone::RightHand,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let Some((s_m, e_m, w_m)) = arms.sides[i] else {
            continue;
        };
        // Skip sides whose wrist has left the frame: the depth was
        // sampled at the hallucinated in-frame stump position and
        // would override the edge-exit extrapolation with garbage.
        // The forearm-length reference is unknown here; 0.20
        // frame-normalised (~0.4 source units on a typical deskcrop
        // span) errs toward keeping the metric path active.
        let get = |k: usize| -> Option<(f32, f32, f32)> {
            joints_2d.get(k).map(|j| (j.nx, j.ny, j.score))
        };
        if crate::tracking::rtmw3d::wrist_out_of_frame(&get, i, 0.20) {
            continue;
        }
        let seg =
            |a: [f32; 3], b: [f32; 3]| -> [f32; 3] { [b[0] - a[0], b[1] - a[1], b[2] - a[2]] };
        let len = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        // NOTE: the snapshot positions are already in source units
        // (the depth builder back-projects then divides by its own
        // calibration), so the segment vectors are source-space and
        // the metre caps are applied after converting BACK to metres
        // via mpsu.
        let ua = seg(s_m, e_m);
        let fa = seg(e_m, w_m);
        debug!(
            "metric-arm side {}: ua={:.3}m fa={:.3}m (caps {:.2}/{:.2}) ua_vec=({:+.2},{:+.2},{:+.2}) fa_vec=({:+.2},{:+.2},{:+.2})",
            i,
            len(ua) * mpsu,
            len(fa) * mpsu,
            MAX_UPPER_ARM_M,
            MAX_FOREARM_M,
            ua[0], ua[1], ua[2],
            fa[0], fa[1], fa[2],
        );
        let ua_m = len(ua) * mpsu;
        let fa_m = len(fa) * mpsu;
        let ua_valid = (MIN_UPPER_ARM_M..=MAX_UPPER_ARM_M).contains(&ua_m);
        let fa_valid = (MIN_FOREARM_M..=MAX_FOREARM_M).contains(&fa_m);
        debug!("metric-arm side {i}: ua_valid={ua_valid} fa_valid={fa_valid}");
        if !ua_valid {
            continue; // no trustworthy anchor for the chain
        }
        let Some(sh_now) = sk.joints.get(&sh_bone).map(|j| j.position) else {
            continue;
        };
        let old_elbow = sk.joints.get(&el_bone).map(|j| j.position);
        let new_elbow = [sh_now[0] + ua[0], sh_now[1] + ua[1], sh_now[2] + ua[2]];
        // Wrist forearm vector, in priority order:
        //   1. this frame's metric vector when plausible (fresh depth)
        //      — also refreshes the temporal hold;
        //   2. the last-valid metric vector held across the stale-
        //      depth gap (carries the forward extension that the
        //      nz-flat reconstruction loses, killing the every-other-
        //      frame forward↔flat flip);
        //   3. the nz + bone-length reconstruction, when no recent
        //      metric direction exists at all.
        // All three are carried on the freshly-relocated (2D-tracked)
        // elbow, so the wrist responds in-plane every frame regardless
        // of which depth source supplies its forward extension.
        let forearm_vec = if fa_valid {
            forearm_hold[i] = Some((fa, 0));
            Some(fa)
        } else if let Some((held, age)) = forearm_hold[i] {
            if age < MAX_FOREARM_HOLD_FRAMES {
                forearm_hold[i] = Some((held, age + 1));
                debug!("metric-arm side {i}: forearm hold reused (age {})", age + 1);
                Some(held)
            } else {
                forearm_hold[i] = None;
                None
            }
        } else {
            None
        };
        let new_wrist = if let Some(fa_vec) = forearm_vec {
            [
                new_elbow[0] + fa_vec[0],
                new_elbow[1] + fa_vec[1],
                new_elbow[2] + fa_vec[2],
            ]
        } else {
            match (old_elbow, sk.joints.get(&wr_bone).map(|j| j.position)) {
                (Some(oe), Some(ow)) => [
                    new_elbow[0] + (ow[0] - oe[0]),
                    new_elbow[1] + (ow[1] - oe[1]),
                    new_elbow[2] + (ow[2] - oe[2]),
                ],
                _ => continue,
            }
        };
        let wrist_delta = match sk.joints.get(&wr_bone) {
            Some(j) => [
                new_wrist[0] - j.position[0],
                new_wrist[1] - j.position[1],
                new_wrist[2] - j.position[2],
            ],
            None => [0.0; 3],
        };
        if let Some(j) = sk.joints.get_mut(&el_bone) {
            j.position = new_elbow;
        }
        if let Some(j) = sk.joints.get_mut(&wr_bone) {
            j.position = new_wrist;
        }
        // Carry the finger chain with the wrist so hand orientation
        // and finger solving stay coherent.
        let left = i == 0;
        for (bone, joint) in sk.joints.iter_mut() {
            if metric_is_finger_side(*bone, left) {
                joint.position[0] += wrist_delta[0];
                joint.position[1] += wrist_delta[1];
                joint.position[2] += wrist_delta[2];
            }
        }
        for (bone, joint) in sk.fingertips.iter_mut() {
            if metric_is_finger_side(*bone, left) {
                joint.position[0] += wrist_delta[0];
                joint.position[1] += wrist_delta[1];
                joint.position[2] += wrist_delta[2];
            }
        }
    }
}

#[cfg(feature = "realsense")]
fn metric_is_finger_side(bone: HumanoidBone, left: bool) -> bool {
    let name = format!("{bone:?}");
    let side_ok = if left {
        name.starts_with("Left")
    } else {
        name.starts_with("Right")
    };
    side_ok
        && (name.contains("Thumb")
            || name.contains("Index")
            || name.contains("Middle")
            || name.contains("Ring")
            || name.contains("Little"))
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
    MetricDepthFrame {
        width: frame.width,
        height: frame.height,
        points_m,
        crop: None,
    }
}
