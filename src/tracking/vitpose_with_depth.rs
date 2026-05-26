//! ViTPose-Wholebody + DAv2 metric-depth provider.
//!
//! ViTPose is 2D-only. Pairing it with Depth Anything V2 Small
//! recovers the z signal so poses with Z-axis motion (jumping,
//! squatting, knee-bend in flight) actually deflect the avatar's
//! limbs in depth instead of collapsing onto the camera plane.
//!
//! ## Pipeline
//!
//! 1. **YOLOX person crop** → 25 % pad onto 3:4 (same as `vitpose`).
//! 2. **ViTPose** on the crop → 133 2D keypoints, remapped to
//!    whole-frame coords.
//! 3. **DAv2-Small** on the whole frame → 392 × 294 relative depth.
//! 4. **Calibrate scale `c`** so `z = c / d_raw` matches `0.40 m`
//!    shoulder span (or `0.32 m` hip fallback) — reuses
//!    [`super::rtmw3d_with_depth::calibrate_scale`].
//! 5. **Build metric frame** by back-projecting every DAv2 pixel via
//!    `(c, h_half, v_half)` — reuses
//!    [`super::rtmw3d_with_depth::build_metric_frame_from_dav2`].
//! 6. **Skeleton from depth** via the shared
//!    [`super::skeleton_from_depth::build_skeleton`] — same code
//!    path the cigpose+moge provider uses, just fed DAv2-derived
//!    metric instead of MoGe true-metric.
//!
//! ## Why sync instead of async
//!
//! The streaming variant `rtmw3d-with-depth` runs DAv2 on a worker
//! thread and tolerates 1-frame staleness. For the per-image
//! `validate_pipeline` use case (and any future "best-quality"
//! offline rendering path) the latency cost doesn't matter and a
//! synchronous call keeps the code shape symmetric with
//! `cigpose-metric-depth`. If realtime streaming becomes a goal for
//! ViTPose, copy the async worker pattern from `rtmw3d_with_depth`.

use std::path::Path;

use super::provider::PoseProvider;
use super::{DetectionAnnotation, PoseEstimate, SourceSkeleton};

#[cfg(feature = "inference")]
use super::depth_anything::DepthAnythingV2Inference;
#[cfg(feature = "inference")]
use super::rtmw3d::{crop_rgb, pad_and_clamp_bbox};
#[cfg(feature = "inference")]
use super::rtmw3d_with_depth::{build_metric_frame_from_dav2, calibrate_scale};
#[cfg(feature = "inference")]
use super::skeleton_from_depth::{
    build_face_bbox_from_joints_2d, build_options_from_calibration, build_skeleton,
    derive_face_pose_metric, resolve_origin_metric,
};
#[cfg(feature = "inference")]
use super::vitpose::{DecodedJoint2d, VitposeInference, VitposeVariant, NUM_JOINTS};
#[cfg(feature = "inference")]
use super::yolox::YoloxPersonDetector;
#[cfg(feature = "inference")]
use log::{debug, info, warn};

pub struct VitposeWithDepthProvider {
    #[cfg(feature = "inference")]
    vitpose: VitposeInference,
    #[cfg(feature = "inference")]
    dav2: DepthAnythingV2Inference,
    #[cfg(feature = "inference")]
    person_detector: Option<YoloxPersonDetector>,
    #[cfg(feature = "inference")]
    face_mesh: Option<super::face_mediapipe::FaceMeshInference>,
    load_warnings: Vec<String>,
    pose_calibration: Option<crate::tracking::PoseCalibration>,
    calibration_mode_hint: Option<crate::tracking::CalibrationMode>,
}

impl VitposeWithDepthProvider {
    #[cfg(feature = "inference")]
    pub(super) fn from_models_dir(models_dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = models_dir.as_ref();
        let variant = VitposeVariant::from_env();
        let vitpose = VitposeInference::from_model_path_with_label(
            dir.join(variant.filename()),
            variant.label(),
        )?;
        let dav2 = DepthAnythingV2Inference::from_model_path(dir.join("dav2_small.onnx"))?;

        let mut load_warnings = Vec::new();
        let person_detector = match YoloxPersonDetector::try_from_models_dir(dir) {
            Ok(Some(d)) => {
                info!("YOLOX person detector loaded ({})", d.backend().label());
                Some(d)
            }
            Ok(None) => {
                let msg = "YOLOX person detector not found; ViTPose will run whole-frame and is likely to produce empty heatmaps.".to_string();
                warn!("{}", msg);
                load_warnings.push(msg);
                None
            }
            Err(e) => {
                let msg = format!(
                    "YOLOX person detector failed to load: {}. ViTPose will run whole-frame.",
                    e
                );
                warn!("{}", msg);
                load_warnings.push(msg);
                None
            }
        };

        // Force FaceMesh on CPU EP — same rationale as
        // `cigpose-metric-depth`: a small face cascade running on a
        // crowded DirectML queue (DAv2 already there) inflates from
        // ~3 ms to ~13 ms. CPU is the better lane.
        let face_mesh = match super::face_mediapipe::FaceMeshInference::try_from_models_dir(
            dir,
            super::face_mediapipe::FaceMeshEp::ForceCpu,
        ) {
            Ok(opt) => opt,
            Err(e) => {
                let msg = format!(
                    "Face inference failed to load: {}. Expressions disabled.",
                    e
                );
                warn!("{}", msg);
                load_warnings.push(msg);
                None
            }
        };

        info!(
            "ViTPose+DAv2 provider ready (ViTPose: {}, DAv2: {}, YOLOX: {}, FaceMesh: {})",
            vitpose.backend().label(),
            dav2.backend().label(),
            if person_detector.is_some() { "on" } else { "off" },
            if face_mesh.is_some() { "on" } else { "off" },
        );

        Ok(Self {
            vitpose,
            dav2,
            person_detector,
            face_mesh,
            load_warnings,
            pose_calibration: None,
            calibration_mode_hint: None,
        })
    }

    #[cfg(not(feature = "inference"))]
    pub(super) fn from_models_dir(_: impl AsRef<Path>) -> Result<Self, String> {
        Err("ViTPose+DAv2 provider requires the `inference` cargo feature".to_string())
    }
}

impl PoseProvider for VitposeWithDepthProvider {
    fn label(&self) -> String {
        #[cfg(feature = "inference")]
        {
            format!(
                "ViTPose+DAv2 / {} + {}",
                self.vitpose.backend().label(),
                self.dav2.backend().label(),
            )
        }
        #[cfg(not(feature = "inference"))]
        {
            "ViTPose+DAv2 (inference disabled)".to_string()
        }
    }

    fn take_load_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.load_warnings)
    }

    fn set_calibration(&mut self, calibration: Option<crate::tracking::PoseCalibration>) {
        self.pose_calibration = calibration;
    }

    fn set_calibration_mode_hint(&mut self, hint: Option<crate::tracking::CalibrationMode>) {
        self.calibration_mode_hint = hint;
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
}

#[cfg(feature = "inference")]
impl VitposeWithDepthProvider {
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

        let t_total = std::time::Instant::now();

        // Phase 1: YOLOX person crop. ViTPose needs a tight bbox or
        // the heatmaps go empty; same constraint as the 2D-only
        // ViTPose provider.
        let t_yolox = std::time::Instant::now();
        let (cropped, infer_slice, infer_w, infer_h, crop_origin, annotation_bbox) = {
            let bbox_opt = self
                .person_detector
                .as_mut()
                .and_then(|d| d.detect_largest_person(rgb_data, width, height));
            match bbox_opt {
                Some(bbox) => {
                    let (cx1, cy1, cx2, cy2) = pad_and_clamp_bbox(&bbox, width, height, 0.25);
                    let cw = cx2 - cx1;
                    let ch = cy2 - cy1;
                    debug!(
                        "ViTPose+DAv2: YOLOX bbox=[{:.0},{:.0},{:.0},{:.0}] score={:.2} → crop {}x{}",
                        bbox.x1, bbox.y1, bbox.x2, bbox.y2, bbox.score, cw, ch
                    );
                    if cw < 32 || ch < 32 {
                        (None, rgb_data, width, height, None, None)
                    } else {
                        let crop = crop_rgb(rgb_data, width, height, cx1, cy1, cw, ch);
                        let ann = Some((
                            bbox.x1 / width as f32,
                            bbox.y1 / height as f32,
                            bbox.x2 / width as f32,
                            bbox.y2 / height as f32,
                        ));
                        (
                            Some(crop),
                            &[][..],
                            cw,
                            ch,
                            Some((cx1 as f32, cy1 as f32, cw as f32, ch as f32)),
                            ann,
                        )
                    }
                }
                None => (None, rgb_data, width, height, None, None),
            }
        };
        let infer_rgb: &[u8] = match &cropped {
            Some(buf) => buf.as_slice(),
            None => infer_slice,
        };
        let dt_yolox = t_yolox.elapsed();

        // Phase 2: ViTPose 2D inference.
        let t_vit = std::time::Instant::now();
        let mut joints_2d = self.vitpose.estimate(infer_rgb, infer_w, infer_h);
        let dt_vit = t_vit.elapsed();
        if joints_2d.len() < NUM_JOINTS {
            return empty_estimate(frame_index);
        }

        // Remap crop-relative `(nx, ny)` to whole-frame coords so DAv2
        // (which always runs on the original frame) and ViTPose share
        // a single coordinate system.
        if let Some((ox, oy, cw, ch)) = crop_origin {
            let inv_w = 1.0 / width as f32;
            let inv_h = 1.0 / height as f32;
            for j in joints_2d.iter_mut() {
                j.nx = (ox + j.nx * cw) * inv_w;
                j.ny = (oy + j.ny * ch) * inv_h;
            }
        }

        // Phase 3: DAv2 on the whole frame.
        let t_dav2 = std::time::Instant::now();
        let dav2_frame = match self.dav2.estimate(rgb_data, width, height) {
            Some(f) => f,
            None => {
                warn!("ViTPose+DAv2: depth pass returned no frame this tick");
                return empty_estimate(frame_index);
            }
        };
        let dt_dav2 = t_dav2.elapsed();

        // Phase 4: calibrate `c` from the shoulder pair (or hip
        // fallback). The cigpose+moge sibling skips this step because
        // MoGe is already metric; DAv2 is relative so we do need it.
        // The cigpose_metric_depth-style `joints_2d` shape (
        // `Vec<DecodedJoint2d>` with `(nx, ny, score)`) is exactly
        // what `calibrate_scale` consumes.
        //
        // While the calibration modal is open we suppress the saved
        // plausibility band — same logic as `rtmw3d_with_depth`'s
        // recalibration path.
        let plausibility_cal = if self.calibration_mode_hint.is_some() {
            None
        } else {
            self.pose_calibration.as_ref()
        };
        let calib_joints = adapt_for_calibrate(&joints_2d);
        let t_calib = std::time::Instant::now();
        let calib = match calibrate_scale(&calib_joints, &dav2_frame, None, width, height, plausibility_cal) {
            Ok(c) => c,
            Err(failure) => {
                warn!(
                    "ViTPose+DAv2: scale calibration failed; falling back to 2D-only ({})",
                    failure
                );
                // Fallback: emit a 2D-only skeleton via the shared
                // 2D path so the caller still sees *some* pose.
                return fallback_planar_estimate(&joints_2d, width, height, frame_index, annotation_bbox);
            }
        };
        let dt_calib = t_calib.elapsed();

        // Phase 5: build metric frame from DAv2 + calibration.
        let t_metric = std::time::Instant::now();
        let metric_frame = build_metric_frame_from_dav2(&dav2_frame, &calib);
        let dt_metric = t_metric.elapsed();

        // Phase 6: skeleton from depth — same code path the
        // cigpose+moge sibling uses, just with DAv2-derived metric.
        let t_skel = std::time::Instant::now();
        let opts = build_options_from_calibration(
            self.pose_calibration.as_ref(),
            self.calibration_mode_hint,
        );
        let (mut skeleton, origin_opt) = match resolve_origin_metric(
            &calib_joints,
            &metric_frame,
            opts,
            self.pose_calibration.as_ref(),
        ) {
            Some(anchor) => {
                let origin = anchor.origin;
                let sk = build_skeleton(
                    frame_index,
                    &calib_joints,
                    &metric_frame,
                    anchor,
                    opts,
                    self.pose_calibration.as_ref(),
                );
                (sk, Some(origin))
            }
            None => (SourceSkeleton::empty(frame_index), None),
        };
        let dt_skel = t_skel.elapsed();

        // Phase 7: face pose. Prefer FaceMesh's dense-landmark pose
        // when available; otherwise the 5-face-landmark DAv2 sample.
        let t_face = std::time::Instant::now();
        if let Some(origin) = origin_opt {
            skeleton.face = derive_face_pose_metric(&calib_joints, &metric_frame, origin);
        }
        if let Some(face_mesh) = self.face_mesh.as_mut() {
            if let Some(bbox) = build_face_bbox_from_joints_2d(&calib_joints, width, height) {
                if let Some((exprs, mesh_conf, mesh_face_pose)) =
                    face_mesh.estimate(rgb_data, width, height, &bbox)
                {
                    skeleton.expressions = exprs;
                    skeleton.face_mesh_confidence = Some(mesh_conf);
                    if let Some(mut fp) = mesh_face_pose {
                        fp.confidence = mesh_conf;
                        skeleton.face = Some(fp);
                    } else if let Some(ref mut fp) = skeleton.face {
                        fp.confidence = fp.confidence.max(mesh_conf);
                    }
                }
            }
        }
        let dt_face = t_face.elapsed();

        let dt_total = t_total.elapsed();
        debug!(
            "ViTPose+DAv2 timing: total={:>5.1}ms yolox={:>4.1}ms vit={:>5.1}ms dav2={:>5.1}ms calib={:>4.2}ms metric={:>4.1}ms skel={:>4.1}ms face={:>4.1}ms (c={:.3} anchor={})",
            dt_total.as_secs_f32() * 1000.0,
            dt_yolox.as_secs_f32() * 1000.0,
            dt_vit.as_secs_f32() * 1000.0,
            dt_dav2.as_secs_f32() * 1000.0,
            dt_calib.as_secs_f32() * 1000.0,
            dt_metric.as_secs_f32() * 1000.0,
            dt_skel.as_secs_f32() * 1000.0,
            dt_face.as_secs_f32() * 1000.0,
            calib.scale_c,
            calib.anchor_label,
        );

        let mut annotation = build_annotation(&joints_2d);
        annotation.bounding_box = annotation_bbox;
        PoseEstimate {
            annotation,
            skeleton,
        }
    }
}

#[cfg(feature = "inference")]
fn adapt_for_calibrate(joints: &[DecodedJoint2d]) -> Vec<super::cigpose::DecodedJoint2d> {
    // `calibrate_scale` and the depth helpers were written against
    // `cigpose::DecodedJoint2d`. Memory layout / fields are identical
    // (`nx, ny, score`), so this is a pure renaming pass.
    joints
        .iter()
        .map(|j| super::cigpose::DecodedJoint2d {
            nx: j.nx,
            ny: j.ny,
            score: j.score,
        })
        .collect()
}

#[cfg(feature = "inference")]
fn fallback_planar_estimate(
    joints_2d: &[DecodedJoint2d],
    width: u32,
    height: u32,
    frame_index: u64,
    annotation_bbox: Option<(f32, f32, f32, f32)>,
) -> PoseEstimate {
    // When DAv2 calibration fails (side-profile shoulders, desk-in-
    // foreground hip pair, …), emit the same 2D-only skeleton the
    // standalone `vitpose-wholebody` provider would produce. Better
    // to ship a planar pose than to drop the frame entirely.
    use super::rtmw3d::{build_source_skeleton_from_coco, DecodedJoint};
    let coco = joints_2d
        .iter()
        .map(|j| DecodedJoint {
            nx: j.nx,
            ny: j.ny,
            nz: 0.5,
            score: j.score,
        })
        .collect::<Vec<_>>();
    let skeleton = build_source_skeleton_from_coco(frame_index, &coco, width, height, false);
    let mut annotation = build_annotation(joints_2d);
    annotation.bounding_box = annotation_bbox;
    PoseEstimate {
        annotation,
        skeleton,
    }
}

#[cfg(feature = "inference")]
fn build_annotation(joints_2d: &[DecodedJoint2d]) -> DetectionAnnotation {
    DetectionAnnotation {
        keypoints: joints_2d.iter().map(|j| (j.nx, j.ny, j.score)).collect(),
        skeleton: Vec::new(),
        bounding_box: None,
    }
}

fn empty_estimate(frame_index: u64) -> PoseEstimate {
    PoseEstimate {
        annotation: DetectionAnnotation::default(),
        skeleton: SourceSkeleton::empty(frame_index),
    }
}
