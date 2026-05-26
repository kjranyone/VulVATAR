//! ViTPose-Wholebody 2D pose provider.
//!
//! Drop-in alternative to `rtmw3d` for the rotation path. ViTPose-S
//! transformer-based 2D keypoints land *visually* more accurate than
//! RTMW3D on the failure modes documented in [[pose-pipeline-iter]]
//! (asymmetric T-pose wrist detection, arm forward-vs-lateral
//! confusion, deskcrop with off-frame hands). Pipeline:
//!
//! 1. **YOLOX person crop** (mandatory — ViTPose was trained on tight
//!    person crops; whole-frame inference yields a near-empty
//!    heatmap). 25% pad onto a 3:4 aspect, same as RTMW3D.
//! 2. **ViTPose** 192×256 NCHW → `(133, H, W)` heatmap tensor.
//! 3. **Decode** argmax + ±0.25 sub-pixel nudge → 133 `(nx, ny)`
//!    crop-relative keypoints + per-joint score.
//! 4. **Remap** to whole-frame coords (same `crop_origin` math as
//!    `cigpose_metric_depth`).
//! 5. **Skeleton build** via the shared
//!    [`super::rtmw3d::build_source_skeleton_from_coco`] using
//!    `nz = 0.5` (= `position[2] = 0` after the source-space
//!    centring), so the resulting `SourceSkeleton` is planar.
//!
//! ## Why 2D-only first
//!
//! The pose-iter memory [[pose-pipeline-iter]] notes the visual
//! failure modes are 2D-localisation problems, not depth problems.
//! Hips-direct + IK-skip already drops the rotation path to 0.8°
//! mean error on RTMW3D; the remaining visual gap is from ViTPose's
//! 2D output being a better fit to the photo subject's pose, not
//! from improved Z. DAv2 pairing (option `b` in the memory) is a
//! future iteration once we've confirmed the 2D win is real.

use std::path::Path;

use super::provider::PoseProvider;
use super::{DetectionAnnotation, PoseEstimate, SourceSkeleton};

#[cfg(feature = "inference")]
use super::rtmw3d::{
    build_source_skeleton_from_coco, crop_rgb, pad_and_clamp_bbox, DecodedJoint,
};
#[cfg(feature = "inference")]
use super::vitpose::{DecodedJoint2d, VitposeInference, VitposeVariant, NUM_JOINTS};
#[cfg(feature = "inference")]
use super::yolox::YoloxPersonDetector;
#[cfg(feature = "inference")]
use log::{debug, info, warn};

pub struct VitposeWholebodyProvider {
    #[cfg(feature = "inference")]
    vitpose: VitposeInference,
    #[cfg(feature = "inference")]
    person_detector: Option<YoloxPersonDetector>,
    load_warnings: Vec<String>,
    /// See `Rtmw3dWithDepthProvider::pose_calibration` — drives
    /// `force_shoulder_anchor` so a phantom hip detection in
    /// upper-body framing can't override the user's declaration.
    pose_calibration: Option<crate::tracking::PoseCalibration>,
    /// Transient mode hint pushed by the GUI while the calibration
    /// modal is open. Same override contract as the other providers.
    calibration_mode_hint: Option<crate::tracking::CalibrationMode>,
}

impl VitposeWholebodyProvider {
    #[cfg(feature = "inference")]
    pub(super) fn from_models_dir(models_dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = models_dir.as_ref();
        let variant = VitposeVariant::from_env();
        let vitpose = VitposeInference::from_model_path_with_label(
            dir.join(variant.filename()),
            variant.label(),
        )?;

        let mut load_warnings = Vec::new();
        let person_detector = match YoloxPersonDetector::try_from_models_dir(dir) {
            Ok(Some(d)) => {
                info!("YOLOX person detector loaded ({})", d.backend().label());
                Some(d)
            }
            Ok(None) => {
                let msg = "YOLOX person detector not found; ViTPose will run whole-frame and is likely to produce empty heatmaps. Install via dev.ps1.".to_string();
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

        info!(
            "ViTPose-Wholebody provider ready (ViTPose: {}, YOLOX: {})",
            vitpose.backend().label(),
            if person_detector.is_some() {
                "on"
            } else {
                "off"
            },
        );

        Ok(Self {
            vitpose,
            person_detector,
            load_warnings,
            pose_calibration: None,
            calibration_mode_hint: None,
        })
    }

    #[cfg(not(feature = "inference"))]
    pub(super) fn from_models_dir(_: impl AsRef<Path>) -> Result<Self, String> {
        Err("ViTPose-Wholebody provider requires the `inference` cargo feature".to_string())
    }
}

impl PoseProvider for VitposeWholebodyProvider {
    fn label(&self) -> String {
        #[cfg(feature = "inference")]
        {
            format!("ViTPose-Wholebody / {}", self.vitpose.backend().label())
        }
        #[cfg(not(feature = "inference"))]
        {
            "ViTPose-Wholebody (inference disabled)".to_string()
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
impl VitposeWholebodyProvider {
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

        // Phase 1: YOLOX person crop. ViTPose was trained on tight
        // person crops — without this stage the heatmaps are nearly
        // empty for any non-centred subject.
        let t_yolox = std::time::Instant::now();
        let (cropped, infer_slice, infer_w, infer_h, crop_origin, annotation_bbox) = {
            let bbox_opt = self
                .person_detector
                .as_mut()
                .and_then(|d| d.detect_largest_person(rgb_data, width, height));
            match bbox_opt {
                Some(bbox) => {
                    let (cx1, cy1, cx2, cy2) =
                        pad_and_clamp_bbox(&bbox, width, height, 0.25);
                    let cw = cx2 - cx1;
                    let ch = cy2 - cy1;
                    debug!(
                        "ViTPose: YOLOX bbox=[{:.0},{:.0},{:.0},{:.0}] score={:.2} → crop {}x{} (orig {}x{})",
                        bbox.x1, bbox.y1, bbox.x2, bbox.y2, bbox.score, cw, ch, width, height
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

        // Phase 2 + 3: ViTPose inference + heatmap decode.
        let t_vit = std::time::Instant::now();
        let mut joints_2d = self.vitpose.estimate(infer_rgb, infer_w, infer_h);
        let dt_vit = t_vit.elapsed();
        if joints_2d.len() < NUM_JOINTS {
            return empty_estimate(frame_index);
        }

        // Phase 4: remap crop-relative `(nx, ny)` back to whole-frame.
        if let Some((ox, oy, cw, ch)) = crop_origin {
            let inv_w = 1.0 / width as f32;
            let inv_h = 1.0 / height as f32;
            for j in joints_2d.iter_mut() {
                j.nx = (ox + j.nx * cw) * inv_w;
                j.ny = (oy + j.ny * ch) * inv_h;
            }
        }

        // Phase 5: source skeleton. ViTPose is 2D-only — synthesise
        // `nz = 0.5` for every joint so the rtmw3d-shared builder's
        // origin-subtract math centres `position[2] = 0` for every
        // joint (planar). The hip-direct + IK-skip path in
        // `pose_solver` already handles the rotation correctly from
        // 2D-only signal; depth pairing comes later.
        let force_shoulder = match self.calibration_mode_hint {
            Some(crate::tracking::CalibrationMode::UpperBody) => true,
            Some(crate::tracking::CalibrationMode::FullBody) => false,
            None => self
                .pose_calibration
                .as_ref()
                .map(|c| matches!(c.mode, crate::tracking::CalibrationMode::UpperBody))
                .unwrap_or(false),
        };
        let coco_joints = to_coco_joints(&joints_2d);
        let skeleton = build_source_skeleton_from_coco(
            frame_index,
            &coco_joints,
            width,
            height,
            force_shoulder,
        );

        let dt_total = t_total.elapsed();
        debug!(
            "ViTPose timing: total={:>5.1}ms yolox={:>5.1}ms infer={:>5.1}ms",
            dt_total.as_secs_f32() * 1000.0,
            dt_yolox.as_secs_f32() * 1000.0,
            dt_vit.as_secs_f32() * 1000.0,
        );

        let annotation = build_annotation(&joints_2d, annotation_bbox);
        PoseEstimate {
            annotation,
            skeleton,
        }
    }
}

#[cfg(feature = "inference")]
fn to_coco_joints(joints_2d: &[DecodedJoint2d]) -> Vec<DecodedJoint> {
    joints_2d
        .iter()
        .map(|j| DecodedJoint {
            nx: j.nx,
            ny: j.ny,
            // 0.5 = depth-bin midpoint. After the source-skeleton
            // origin subtraction in `rtmw3d::skeleton::to_source`,
            // every joint's `position[2]` collapses to 0. The
            // rotation path consumes only relative directions, so
            // a planar input means the solver ignores the Z axis
            // entirely and lets `compute_hip_align_rotation` drive
            // body yaw from XY alone.
            nz: 0.5,
            score: j.score,
        })
        .collect()
}

#[cfg(feature = "inference")]
fn build_annotation(
    joints_2d: &[DecodedJoint2d],
    bounding_box: Option<(f32, f32, f32, f32)>,
) -> DetectionAnnotation {
    DetectionAnnotation {
        keypoints: joints_2d.iter().map(|j| (j.nx, j.ny, j.score)).collect(),
        skeleton: Vec::new(),
        bounding_box,
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
