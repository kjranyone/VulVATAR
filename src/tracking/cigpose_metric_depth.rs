//! Experimental CIGPose 2D + MoGe-2 metric-depth tracking pipeline.
//!
//! CIGPose emits 133 COCO-Wholebody keypoints in 2D image coordinates;
//! MoGe-2 emits a per-pixel metric point cloud (in metres). For each
//! 2D keypoint we sample MoGe's point cloud at the corresponding
//! pixel and use the resulting `(X, Y, Z)` as the joint's
//! source-space position.
//!
//! ## Trade-off vs other providers
//!
//! * **vs `rtmw3d` (default)** — RTMW3D's z is hip-relative synthetic
//!   from a body prior baked into the model. CIGPose+MoGe uses
//!   *measured* monocular depth, so forward / backward limb motion is
//!   more faithful. Cost: two ONNX sessions, much slower than RTMW3D.
//! * **vs `rtmw3d-with-depth`** — The 30-fps streaming sibling. That
//!   one trades MoGe's true-metric output for DAv2-Small's
//!   relative-then-calibrated-metric depth and gets a ~5× speedup,
//!   at the cost of an absolute-scale guess (shoulder span = 0.40 m).
//!   This provider keeps MoGe's true-metric path and is the right
//!   reference when accuracy matters more than fps.
//!
//! ## Pipeline
//!
//! 1. **YOLOX** person crop (optional). 1.25× pad, same pre-stage as
//!    RTMW3D. CIGPose runs on the crop; 2D keypoints get remapped to
//!    whole-frame normalised coords below so MoGe (always whole-frame)
//!    can be sampled in a single coordinate system.
//! 2. **CIGPose** 2D inference → 133 COCO-Wholebody keypoints.
//! 3. **MoGe-2** metric depth on whole frame → 644×476 point cloud.
//! 4. **Skeleton** built via [`super::skeleton_from_depth`]: hip mid
//!    as origin, selfie-mirror axis flips, hand chains.
//! 5. **Head pose** derived from face landmarks 0..=4 + MoGe samples.
//! 6. **FaceMesh + BlendshapeV2** cascade (optional) → expressions.

use std::path::{Path, PathBuf};

use super::provider::PoseProvider;
use super::{DetectionAnnotation, PoseEstimate, SourceSkeleton};

#[cfg(feature = "inference")]
use super::cigpose::{CigposeInference, NUM_JOINTS};
#[cfg(feature = "inference")]
use super::face_mediapipe::FaceMeshInference;
#[cfg(feature = "inference")]
use super::metric_depth::MoGe2Inference;
#[cfg(feature = "inference")]
use super::skeleton_from_depth::{
    build_face_bbox_from_joints_2d, build_skeleton, derive_face_pose_metric, resolve_origin_metric,
};
#[cfg(feature = "inference")]
use super::yolox::YoloxPersonDetector;
#[cfg(feature = "inference")]
use log::{debug, info, warn};

pub struct CigposeMetricDepthProvider {
    #[cfg(feature = "inference")]
    cigpose: CigposeInference,
    #[cfg(feature = "inference")]
    moge: MoGe2Inference,
    #[cfg(feature = "inference")]
    person_detector: Option<YoloxPersonDetector>,
    #[cfg(feature = "inference")]
    face_mesh: Option<FaceMeshInference>,
    #[allow(dead_code)]
    cigpose_model_path: PathBuf,
    #[allow(dead_code)]
    moge_model_path: PathBuf,
    load_warnings: Vec<String>,
}

impl CigposeMetricDepthProvider {
    #[cfg(feature = "inference")]
    pub(super) fn from_models_dir(models_dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = models_dir.as_ref();
        let cigpose_model_path = dir.join("cigpose.onnx");
        let moge_model_path = dir.join("moge_v2.onnx");

        let cigpose = CigposeInference::from_model_path(&cigpose_model_path)?;
        let moge = MoGe2Inference::from_model_path(&moge_model_path)?;

        let mut load_warnings = Vec::new();
        let person_detector = match YoloxPersonDetector::try_from_models_dir(dir) {
            Ok(Some(d)) => {
                info!("YOLOX person detector loaded ({})", d.backend().label());
                Some(d)
            }
            Ok(None) => {
                info!("YOLOX person detector not found — CIGPose will run whole-frame");
                None
            }
            Err(e) => {
                let msg = format!(
                    "YOLOX person detector failed to load: {}. Falling back to whole-frame CIGPose.",
                    e
                );
                warn!("{}", msg);
                load_warnings.push(msg);
                None
            }
        };

        // Force CPU EP for FaceMesh: MoGe-2 (~175 ms) on DirectML
        // would otherwise inflate the small face cascade via GPU
        // queue contention.
        let face_mesh = match FaceMeshInference::try_from_models_dir(
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
            "CIGPose+MoGe-2 provider ready (CIGPose: {}, MoGe-2: {}, YOLOX: {}, FaceMesh: {})",
            cigpose.backend().label(),
            moge.backend().label(),
            if person_detector.is_some() { "on" } else { "off" },
            if face_mesh.is_some() { "on" } else { "off" },
        );

        Ok(Self {
            cigpose,
            moge,
            person_detector,
            face_mesh,
            cigpose_model_path,
            moge_model_path,
            load_warnings,
        })
    }

    #[cfg(not(feature = "inference"))]
    pub(super) fn from_models_dir(_: impl AsRef<Path>) -> Result<Self, String> {
        Err("CIGPose+MoGe-2 provider requires the `inference` cargo feature".to_string())
    }
}

impl PoseProvider for CigposeMetricDepthProvider {
    fn label(&self) -> String {
        #[cfg(feature = "inference")]
        {
            format!("CIGPose+MoGe-2 / {}", self.cigpose.backend().label())
        }
        #[cfg(not(feature = "inference"))]
        {
            "CIGPose+MoGe-2 (inference disabled)".to_string()
        }
    }

    fn take_load_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.load_warnings)
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
impl CigposeMetricDepthProvider {
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

        let t_yolox = std::time::Instant::now();
        let (cig_owned, cig_input_slice, cig_w, cig_h, crop_origin, annotation_bbox) = {
            let bbox_opt = self
                .person_detector
                .as_mut()
                .and_then(|d| d.detect_largest_person(rgb_data, width, height));
            match bbox_opt {
                Some(bbox) => {
                    let (cx1, cy1, cx2, cy2) = super::rtmw3d::pad_and_clamp_bbox(
                        &bbox, width, height, 0.25,
                    );
                    let cw = cx2 - cx1;
                    let ch = cy2 - cy1;
                    debug!(
                        "CIGPose+MoGe: YOLOX bbox=[{:.0},{:.0},{:.0},{:.0}] score={:.2} → crop {}x{} (orig {}x{})",
                        bbox.x1, bbox.y1, bbox.x2, bbox.y2, bbox.score, cw, ch, width, height
                    );
                    if cw < 32 || ch < 32 {
                        (None, rgb_data, width, height, None, None)
                    } else {
                        let crop = super::rtmw3d::crop_rgb(rgb_data, width, height, cx1, cy1, cw, ch);
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
        let cig_input: &[u8] = match &cig_owned {
            Some(buf) => buf.as_slice(),
            None => cig_input_slice,
        };
        let dt_yolox = t_yolox.elapsed();

        let t_cig = std::time::Instant::now();
        let mut joints_2d = self.cigpose.estimate(cig_input, cig_w, cig_h);
        let dt_cig = t_cig.elapsed();
        if joints_2d.len() < NUM_JOINTS {
            return empty_estimate(frame_index);
        }

        if let Some((ox, oy, cw, ch)) = crop_origin {
            let inv_w = 1.0 / width as f32;
            let inv_h = 1.0 / height as f32;
            for j in joints_2d.iter_mut() {
                j.nx = (ox + j.nx * cw) * inv_w;
                j.ny = (oy + j.ny * ch) * inv_h;
            }
        }

        let t_moge = std::time::Instant::now();
        let depth_frame = match self.moge.estimate(rgb_data, width, height) {
            Some(f) => f,
            None => {
                warn!("CIGPose+MoGe-2: depth pass returned no frame this tick");
                return empty_estimate(frame_index);
            }
        };
        let dt_moge = t_moge.elapsed();

        let t_skel = std::time::Instant::now();
        let (mut skeleton, origin_opt) = match resolve_origin_metric(&joints_2d, &depth_frame) {
            Some((origin, anchor_was_hip, anchor_score)) => {
                let sk = build_skeleton(
                    frame_index,
                    &joints_2d,
                    &depth_frame,
                    origin,
                    anchor_was_hip,
                    anchor_score,
                );
                (sk, Some(origin))
            }
            None => (SourceSkeleton::empty(frame_index), None),
        };
        let dt_skel = t_skel.elapsed();

        let t_face = std::time::Instant::now();
        if let Some(origin) = origin_opt {
            skeleton.face = derive_face_pose_metric(&joints_2d, &depth_frame, origin);
        }

        if let Some(face_mesh) = self.face_mesh.as_mut() {
            if let Some(bbox) = build_face_bbox_from_joints_2d(&joints_2d, width, height) {
                if let Some((exprs, mesh_conf, mesh_face_pose)) =
                    face_mesh.estimate(rgb_data, width, height, &bbox)
                {
                    skeleton.expressions = exprs;
                    skeleton.face_mesh_confidence = Some(mesh_conf);
                    // Prefer FaceMesh's dense-landmark pose over the
                    // CIGPose+MoGe-derived one — same rationale as
                    // the rtmw3d-with-depth path: the 5 face
                    // landmarks DAv2-sampled from a small head region
                    // are too noisy for stable yaw across rotations.
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

        let mut annotation = build_annotation(&joints_2d);
        annotation.bounding_box = annotation_bbox;

        let dt_total = t_total.elapsed();
        debug!(
            "CIGPose+MoGe timing: total={:>5.1}ms yolox={:>5.1}ms cig={:>5.1}ms moge={:>5.1}ms skel={:>4.1}ms face={:>4.1}ms",
            dt_total.as_secs_f32() * 1000.0,
            dt_yolox.as_secs_f32() * 1000.0,
            dt_cig.as_secs_f32() * 1000.0,
            dt_moge.as_secs_f32() * 1000.0,
            dt_skel.as_secs_f32() * 1000.0,
            dt_face.as_secs_f32() * 1000.0,
        );

        PoseEstimate {
            annotation,
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

#[cfg(feature = "inference")]
fn build_annotation(joints: &[super::cigpose::DecodedJoint2d]) -> DetectionAnnotation {
    let keypoints = joints
        .iter()
        .map(|j| (j.nx, j.ny, j.score))
        .collect::<Vec<_>>();
    DetectionAnnotation {
        keypoints,
        skeleton: Vec::new(),
        bounding_box: None,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn empty_estimate_is_well_formed() {
        let est = super::empty_estimate(42);
        assert_eq!(est.skeleton.source_timestamp, 42);
        assert!(est.skeleton.joints.is_empty());
        assert!(est.annotation.keypoints.is_empty());
    }
}
