//! Experimental CIGPose 2D + MoGe-2 metric-depth tracking pipeline.
//!
//! CIGPose emits 133 COCO-Wholebody keypoints in 2D image coordinates;
//! MoGe-2 emits a per-pixel metric point cloud (in metres). For each
//! 2D keypoint we sample MoGe's point cloud at the corresponding pixel
//! (with a small local-window median to suppress single-pixel spikes
//! around wrists/fingers) and use the resulting `(X, Y, Z)` directly
//! as the joint's source-space position.
//!
//! Compared to the default RTMW3D path:
//!
//! * RTMW3D produces 3D keypoints from a single forward pass with z
//!   inferred from a body-prior baked into the model. CIGPose+MoGe
//!   uses *measured* monocular depth, so forward/backward limb motion
//!   is more faithful in principle — at the cost of two ONNX
//!   sessions instead of one.
//! * No YOLOX person crop. CIGPose runs whole-frame so its 2D
//!   keypoints share the same pixel coord system as MoGe's depth map,
//!   keeping the pixel-to-3D lookup trivial.
//! * No wrist resilience layer or face cascade in v1 — body and hand
//!   chains only. The face bone falls back to the spine direction in
//!   the avatar solver, identical to RTMW3D's pre-FaceMesh state.
//!
//! Coordinate convention: MoGe outputs camera-space `(x-right, y-down,
//! z-forward)` in metres. After hip-mid origin subtraction we flip
//! every axis to land in source-space `(selfie-mirror x, y-up, z
//! toward camera)` per the [`crate::tracking::source_skeleton`]
//! contract. Absolute scale is metric (metres) rather than image-
//! relative — the avatar solver consumes only relative bone
//! directions, so the unit difference vs RTMW3D is invisible.

use std::path::{Path, PathBuf};

use super::provider::PoseProvider;
use super::{DetectionAnnotation, PoseEstimate, SourceJoint, SourceSkeleton};
use crate::asset::HumanoidBone;

#[cfg(feature = "inference")]
use super::cigpose::{CigposeInference, DecodedJoint2d, NUM_JOINTS};
#[cfg(feature = "inference")]
use super::face_mediapipe::{derive_face_bbox, FaceBbox, FaceMeshInference};
#[cfg(feature = "inference")]
use super::metric_depth::{MoGe2Inference, MoGeFrame};
#[cfg(feature = "inference")]
use super::yolox::YoloxPersonDetector;
#[cfg(feature = "inference")]
use super::FacePose;
#[cfg(feature = "inference")]
use log::{debug, info, warn};

/// Per-keypoint visibility floor — same as RTMW3D's structural gate.
/// User-facing "is this keypoint quality enough" filtering still
/// happens at the solver via `joint_confidence_threshold`.
#[cfg(feature = "inference")]
const KEYPOINT_VISIBILITY_FLOOR: f32 = 0.05;

/// Local-window radius (pixels in MoGe's input grid) for sampling the
/// point cloud at a 2D keypoint. Small enough to track fingers without
/// dragging in neighbouring depth, large enough that single-pixel
/// MoGe artefacts get out-voted by the median.
#[cfg(feature = "inference")]
const SAMPLE_RADIUS_PX: i32 = 3;

// ---------------------------------------------------------------------------
// COCO-Wholebody index → HumanoidBone tables. Duplicated from
// `rtmw3d::consts` rather than promoted to a shared module — both
// pipelines target the same 133-keypoint topology, but the two
// providers may diverge on which keypoints they trust (e.g. CIGPose
// has no body-internal Z to disagree with the hand track wrist), so
// keeping the tables next to their consumer makes that local. Selfie
// mirror: subject's anatomical-left landmarks drive avatar `Right*`
// bones.
// ---------------------------------------------------------------------------

#[cfg(feature = "inference")]
const COCO_BODY: &[(usize, HumanoidBone)] = &[
    (5, HumanoidBone::RightShoulder),
    (5, HumanoidBone::RightUpperArm),
    (6, HumanoidBone::LeftShoulder),
    (6, HumanoidBone::LeftUpperArm),
    (7, HumanoidBone::RightLowerArm),
    (8, HumanoidBone::LeftLowerArm),
    (11, HumanoidBone::RightUpperLeg),
    (12, HumanoidBone::LeftUpperLeg),
    (13, HumanoidBone::RightLowerLeg),
    (14, HumanoidBone::LeftLowerLeg),
    (15, HumanoidBone::RightFoot),
    (16, HumanoidBone::LeftFoot),
];

#[cfg(feature = "inference")]
const COCO_FOOT_TIPS: &[(usize, HumanoidBone)] = &[
    (17, HumanoidBone::RightFoot),
    (20, HumanoidBone::LeftFoot),
];

#[cfg(feature = "inference")]
const HAND_PHALANGES_RIGHT: &[(usize, HumanoidBone)] = &[
    (1, HumanoidBone::RightThumbProximal),
    (2, HumanoidBone::RightThumbIntermediate),
    (3, HumanoidBone::RightThumbDistal),
    (5, HumanoidBone::RightIndexProximal),
    (6, HumanoidBone::RightIndexIntermediate),
    (7, HumanoidBone::RightIndexDistal),
    (9, HumanoidBone::RightMiddleProximal),
    (10, HumanoidBone::RightMiddleIntermediate),
    (11, HumanoidBone::RightMiddleDistal),
    (13, HumanoidBone::RightRingProximal),
    (14, HumanoidBone::RightRingIntermediate),
    (15, HumanoidBone::RightRingDistal),
    (17, HumanoidBone::RightLittleProximal),
    (18, HumanoidBone::RightLittleIntermediate),
    (19, HumanoidBone::RightLittleDistal),
];

#[cfg(feature = "inference")]
const HAND_TIPS_RIGHT: &[(usize, HumanoidBone)] = &[
    (4, HumanoidBone::RightThumbDistal),
    (8, HumanoidBone::RightIndexDistal),
    (12, HumanoidBone::RightMiddleDistal),
    (16, HumanoidBone::RightRingDistal),
    (20, HumanoidBone::RightLittleDistal),
];

#[cfg(feature = "inference")]
const HAND_PHALANGES_LEFT: &[(usize, HumanoidBone)] = &[
    (1, HumanoidBone::LeftThumbProximal),
    (2, HumanoidBone::LeftThumbIntermediate),
    (3, HumanoidBone::LeftThumbDistal),
    (5, HumanoidBone::LeftIndexProximal),
    (6, HumanoidBone::LeftIndexIntermediate),
    (7, HumanoidBone::LeftIndexDistal),
    (9, HumanoidBone::LeftMiddleProximal),
    (10, HumanoidBone::LeftMiddleIntermediate),
    (11, HumanoidBone::LeftMiddleDistal),
    (13, HumanoidBone::LeftRingProximal),
    (14, HumanoidBone::LeftRingIntermediate),
    (15, HumanoidBone::LeftRingDistal),
    (17, HumanoidBone::LeftLittleProximal),
    (18, HumanoidBone::LeftLittleIntermediate),
    (19, HumanoidBone::LeftLittleDistal),
];

#[cfg(feature = "inference")]
const HAND_TIPS_LEFT: &[(usize, HumanoidBone)] = &[
    (4, HumanoidBone::LeftThumbDistal),
    (8, HumanoidBone::LeftIndexDistal),
    (12, HumanoidBone::LeftMiddleDistal),
    (16, HumanoidBone::LeftRingDistal),
    (20, HumanoidBone::LeftLittleDistal),
];

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

pub struct CigposeMetricDepthProvider {
    #[cfg(feature = "inference")]
    cigpose: CigposeInference,
    #[cfg(feature = "inference")]
    moge: MoGe2Inference,
    /// Optional YOLOX person detector. When present, CIGPose runs on
    /// a 1.25× padded crop around the largest detected person (matches
    /// the RTMW3D pre-stage). MoGe-2 still runs on the whole frame so
    /// the keypoint→depth lookup stays in a single pixel-coord system.
    #[cfg(feature = "inference")]
    person_detector: Option<YoloxPersonDetector>,
    /// Optional MediaPipe FaceMesh + BlendshapeV2 cascade. When
    /// present, the face bbox is built from CIGPose's face-68 (indices
    /// 23..=90) and the result populates `skeleton.expressions` +
    /// `skeleton.face_mesh_confidence`.
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

        // Optional sub-pipelines: same load-or-warn pattern as RTMW3D.
        // Both fall through to "feature off" rather than blocking the
        // provider, so a partial models/ directory still gives the user
        // the body track.
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

        let face_mesh = match FaceMeshInference::try_from_models_dir(dir) {
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

        // YOLOX person crop. Same pre-stage as RTMW3D: 25% pad to give
        // the pose head some slack for outstretched limbs. CIGPose then
        // runs on the crop; the 2D keypoints come back in crop space
        // and get remapped to whole-frame normalised coords below so
        // MoGe (which always runs on the whole frame) can be sampled
        // at the same coordinate system without an extra transform.
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

        // Remap crop-space (nx, ny) → original-frame (nx, ny). When
        // there was no crop this is a no-op. Note CIGPose has no Z so
        // depth comes from MoGe per-keypoint sampling later.
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

        // Head pose from CIGPose face landmarks 0..=4 (nose / eyes /
        // ears) lifted into 3D via MoGe sampling. Same yaw/pitch/roll
        // math as `rtmw3d::face::derive_face_pose_from_body`, but
        // operating on metric coords directly so we don't need a
        // RTMW3D-style nz scaling factor.
        let t_face = std::time::Instant::now();
        if let Some(origin) = origin_opt {
            skeleton.face = derive_face_pose_metric(&joints_2d, &depth_frame, origin);
        }

        // FaceMesh + BlendshapeV2: only runs when the cascade is
        // loaded *and* CIGPose's face-68 produced a valid bbox. The
        // mesh's own confidence boosts the body-derived face_pose so
        // the solver gate doesn't reject head rotation just because
        // the 5 face anchor scores happened to be low this frame.
        if let Some(face_mesh) = self.face_mesh.as_mut() {
            if let Some(bbox) = build_face_bbox_from_joints_2d(&joints_2d, width, height) {
                if let Some((exprs, mesh_conf)) =
                    face_mesh.estimate(rgb_data, width, height, &bbox)
                {
                    skeleton.expressions = exprs;
                    skeleton.face_mesh_confidence = Some(mesh_conf);
                    if let Some(ref mut fp) = skeleton.face {
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
fn build_annotation(joints: &[DecodedJoint2d]) -> DetectionAnnotation {
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

// ---------------------------------------------------------------------------
// Skeleton building
// ---------------------------------------------------------------------------

/// Sample MoGe's metric point cloud at a frame-relative `(nx, ny)`
/// keypoint position with a local-window median. Returns `None` when
/// the keypoint falls outside the frame or every sample in the window
/// is masked.
#[cfg(feature = "inference")]
fn sample_metric_point(frame: &MoGeFrame, nx: f32, ny: f32) -> Option<[f32; 3]> {
    if !nx.is_finite() || !ny.is_finite() {
        return None;
    }
    let w = frame.width as i32;
    let h = frame.height as i32;
    let cx = (nx * frame.width as f32).round() as i32;
    let cy = (ny * frame.height as f32).round() as i32;
    if cx < 0 || cy < 0 || cx >= w || cy >= h {
        return None;
    }
    let mut xs = Vec::with_capacity(((SAMPLE_RADIUS_PX * 2 + 1) * (SAMPLE_RADIUS_PX * 2 + 1)) as usize);
    let mut ys = Vec::with_capacity(xs.capacity());
    let mut zs = Vec::with_capacity(xs.capacity());
    for yy in (cy - SAMPLE_RADIUS_PX).max(0)..=(cy + SAMPLE_RADIUS_PX).min(h - 1) {
        for xx in (cx - SAMPLE_RADIUS_PX).max(0)..=(cx + SAMPLE_RADIUS_PX).min(w - 1) {
            let idx = yy as usize * frame.width as usize + xx as usize;
            let [px, py, pz] = frame.points_m[idx];
            if px.is_finite() && py.is_finite() && pz.is_finite() && pz > 0.0 {
                xs.push(px);
                ys.push(py);
                zs.push(pz);
            }
        }
    }
    Some([median(xs)?, median(ys)?, median(zs)?])
}

#[cfg(feature = "inference")]
fn median(mut values: Vec<f32>) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    Some(values[values.len() / 2])
}

/// Resolve the body anchor origin in metric camera-space coords.
///
/// Hip mid (CIGPose 11/12) is preferred. Joint *score* alone isn't
/// enough: even confident 2D keypoints can land on a pixel that MoGe
/// masked out (jacket print with low geometry confidence, etc.), so we
/// also require a successful point-cloud sample before accepting the
/// anchor. If the hip pair fails, fall back to shoulder mid (5/6) and
/// signal `anchor_was_hip = false` so the consumer can suppress
/// lower-body keypoints (anything CIGPose emits for legs / feet on an
/// upper-body crop is hallucinated).
///
/// Returns `None` when neither pair is usable — the frame produces no
/// skeleton.
#[cfg(feature = "inference")]
fn resolve_origin_metric(
    joints: &[DecodedJoint2d],
    depth: &MoGeFrame,
) -> Option<(/*origin*/ [f32; 3], /*anchor_was_hip*/ bool, /*anchor_score*/ f32)> {
    if joints.len() < NUM_JOINTS {
        return None;
    }
    let try_pair = |a: &DecodedJoint2d, b: &DecodedJoint2d| -> Option<[f32; 3]> {
        let l = sample_metric_point(depth, a.nx, a.ny)?;
        let r = sample_metric_point(depth, b.nx, b.ny)?;
        Some([
            (l[0] + r[0]) * 0.5,
            (l[1] + r[1]) * 0.5,
            (l[2] + r[2]) * 0.5,
        ])
    };

    let lh = &joints[11];
    let rh = &joints[12];
    let hip_score = lh.score.min(rh.score);
    if hip_score >= KEYPOINT_VISIBILITY_FLOOR {
        if let Some(o) = try_pair(lh, rh) {
            return Some((o, true, hip_score));
        }
    }

    let ls = &joints[5];
    let rs = &joints[6];
    let shoulder_score = ls.score.min(rs.score);
    if shoulder_score >= KEYPOINT_VISIBILITY_FLOOR {
        if let Some(o) = try_pair(ls, rs) {
            return Some((o, false, shoulder_score));
        }
    }
    None
}

/// Build a [`SourceSkeleton`] from CIGPose 2D keypoints + sampled MoGe
/// metric depth using a pre-resolved origin. Mirrors the structural
/// decisions in [`super::rtmw3d::skeleton::build_source_skeleton`]:
///
/// * Anatomical-left keypoints drive avatar `Right*` bones (selfie
///   mirror) — every camera-space axis is flipped on the way out so
///   the source-space convention `(x toward avatar +x, y-up, z toward
///   camera)` is preserved.
/// * Hands: wrist position = centroid of the four MCPs (locals 5, 9,
///   13, 17). Same rationale as RTMW3D — hand-track wrist landmarks
///   are unreliable under occlusion.
/// * No hand orientation derivation in v1; the avatar solver leaves
///   the wrist twist at rest.
///
/// Face / expressions are left empty — the caller is expected to
/// populate them via [`derive_face_pose_metric`] and the optional
/// FaceMesh cascade.
#[cfg(feature = "inference")]
fn build_skeleton(
    frame_index: u64,
    joints: &[DecodedJoint2d],
    depth: &MoGeFrame,
    origin: [f32; 3],
    anchor_was_hip: bool,
    anchor_score: f32,
) -> SourceSkeleton {
    let mut sk = SourceSkeleton::empty(frame_index);
    if joints.len() < NUM_JOINTS {
        return sk;
    }

    let to_source = |p: [f32; 3]| -> [f32; 3] {
        [
            -(p[0] - origin[0]),
            -(p[1] - origin[1]),
            -(p[2] - origin[2]),
        ]
    };

    let resolve = |idx: usize| -> Option<SourceJoint> {
        if idx >= joints.len() {
            return None;
        }
        let j = &joints[idx];
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            return None;
        }
        let pos_cam = sample_metric_point(depth, j.nx, j.ny)?;
        Some(SourceJoint {
            position: to_source(pos_cam),
            confidence: j.score,
        })
    };

    if anchor_was_hip {
        sk.joints.insert(
            HumanoidBone::Hips,
            SourceJoint {
                position: [0.0, 0.0, 0.0],
                confidence: anchor_score,
            },
        );
    }

    const LOWER_BODY_FIRST_IDX: usize = 11;
    for &(idx, bone) in COCO_BODY {
        if !anchor_was_hip && idx >= LOWER_BODY_FIRST_IDX {
            continue;
        }
        if let Some(joint) = resolve(idx) {
            sk.joints.insert(bone, joint);
        }
    }

    if anchor_was_hip {
        for &(idx, bone) in COCO_FOOT_TIPS {
            if let Some(joint) = resolve(idx) {
                sk.fingertips.insert(bone, joint);
            }
        }
    }

    attach_hand(
        &mut sk,
        joints,
        depth,
        91,
        HumanoidBone::RightHand,
        HAND_PHALANGES_RIGHT,
        HAND_TIPS_RIGHT,
        &to_source,
    );
    attach_hand(
        &mut sk,
        joints,
        depth,
        112,
        HumanoidBone::LeftHand,
        HAND_PHALANGES_LEFT,
        HAND_TIPS_LEFT,
        &to_source,
    );

    let mut sum = 0.0_f32;
    let mut n = 0_u32;
    for j in joints.iter().take(17) {
        sum += j.score;
        n += 1;
    }
    sk.overall_confidence = if n > 0 { sum / n as f32 } else { 0.0 };

    sk.face = None;
    sk.expressions = Vec::new();
    sk
}

#[cfg(feature = "inference")]
#[allow(clippy::too_many_arguments)]
fn attach_hand<F>(
    sk: &mut SourceSkeleton,
    joints: &[DecodedJoint2d],
    depth: &MoGeFrame,
    base_index: usize,
    wrist_bone: HumanoidBone,
    phalanges: &[(usize, HumanoidBone)],
    tips: &[(usize, HumanoidBone)],
    to_source: &F,
) where
    F: Fn([f32; 3]) -> [f32; 3],
{
    const MCP_LOCALS: [usize; 4] = [5, 9, 13, 17];
    let mut sum = [0.0_f32; 3];
    let mut min_conf = f32::INFINITY;
    let mut found = 0_u32;
    for &local in &MCP_LOCALS {
        let global = base_index + local;
        if global >= joints.len() {
            continue;
        }
        let j = &joints[global];
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            continue;
        }
        let p_cam = match sample_metric_point(depth, j.nx, j.ny) {
            Some(p) => p,
            None => continue,
        };
        let p = to_source(p_cam);
        sum[0] += p[0];
        sum[1] += p[1];
        sum[2] += p[2];
        min_conf = min_conf.min(j.score);
        found += 1;
    }
    if found < 3 {
        return;
    }
    let inv = 1.0 / found as f32;
    sk.joints.insert(
        wrist_bone,
        SourceJoint {
            position: [sum[0] * inv, sum[1] * inv, sum[2] * inv],
            confidence: min_conf,
        },
    );

    for &(local_idx, bone) in phalanges {
        let global = base_index + local_idx;
        if global >= joints.len() {
            continue;
        }
        let j = &joints[global];
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            continue;
        }
        if let Some(p_cam) = sample_metric_point(depth, j.nx, j.ny) {
            sk.joints.insert(
                bone,
                SourceJoint {
                    position: to_source(p_cam),
                    confidence: j.score,
                },
            );
        }
    }
    for &(local_idx, bone) in tips {
        let global = base_index + local_idx;
        if global >= joints.len() {
            continue;
        }
        let j = &joints[global];
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            continue;
        }
        if let Some(p_cam) = sample_metric_point(depth, j.nx, j.ny) {
            sk.fingertips.insert(
                bone,
                SourceJoint {
                    position: to_source(p_cam),
                    confidence: j.score,
                },
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Face cascade helpers
// ---------------------------------------------------------------------------

/// Derive head yaw / pitch / roll from CIGPose's body face keypoints
/// 0..=4 (nose / left_eye / right_eye / left_ear / right_ear), each
/// lifted into 3D via MoGe-sampled metric depth and routed through
/// the same axis-flip + hip-origin convention as the body skeleton.
///
/// Logic mirrors [`super::rtmw3d::face::derive_face_pose_from_body`]:
///
/// * Ear-line direction in XZ → yaw (with reflected-back-ear fallback
///   when one ear is occluded).
/// * Eye-line direction in XY → roll.
/// * Eye-mid → nose vertical disparity normalised by face width →
///   pitch.
///
/// Operates on metric coords (metres). Confidence is the min of the
/// 5 input scores; if any keypoint is below the visibility floor the
/// face pose is treated as unreliable and `None` is returned so the
/// solver leaves the head bone at rest rather than latching onto a
/// hallucinated 5-landmark cluster.
#[cfg(feature = "inference")]
fn derive_face_pose_metric(
    joints: &[DecodedJoint2d],
    depth: &MoGeFrame,
    origin: [f32; 3],
) -> Option<FacePose> {
    if joints.len() < 5 {
        return None;
    }
    for j in joints.iter().take(5) {
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            return None;
        }
    }

    // Sample each face landmark in MoGe and convert to source-space
    // (selfie-mirror x, y-up, z toward camera) using the same hip-mid
    // origin as the body skeleton.
    let to_src = |idx: usize| -> Option<[f32; 3]> {
        let j = &joints[idx];
        let p_cam = sample_metric_point(depth, j.nx, j.ny)?;
        Some([
            -(p_cam[0] - origin[0]),
            -(p_cam[1] - origin[1]),
            -(p_cam[2] - origin[2]),
        ])
    };

    let nose = to_src(0)?;
    // Selfie mirror: subject's left maps to avatar's right.
    let avatar_right_eye = to_src(1)?;
    let avatar_left_eye = to_src(2)?;
    let avatar_right_ear = to_src(3)?;
    let avatar_left_ear = to_src(4)?;

    // Back-ear robustness: when the head turns past ~45° one ear
    // becomes occluded and CIGPose still emits a position for it,
    // but MoGe will pin the depth to whatever surface is at that
    // pixel — typically the visible side of the head, collapsing
    // ear_dz toward zero. Detect occlusion by score asymmetry and
    // reconstruct the back ear as the visible ear reflected through
    // the nose's XZ position.
    const EAR_OCCLUSION_RATIO: f32 = 0.5;
    let s_right_ear = joints[3].score;
    let s_left_ear = joints[4].score;
    let (right_ear_xz, left_ear_xz) = if s_right_ear < s_left_ear * EAR_OCCLUSION_RATIO {
        let recon = [
            2.0 * nose[0] - avatar_left_ear[0],
            avatar_left_ear[1],
            2.0 * nose[2] - avatar_left_ear[2],
        ];
        (recon, avatar_left_ear)
    } else if s_left_ear < s_right_ear * EAR_OCCLUSION_RATIO {
        let recon = [
            2.0 * nose[0] - avatar_right_ear[0],
            avatar_right_ear[1],
            2.0 * nose[2] - avatar_right_ear[2],
        ];
        (avatar_right_ear, recon)
    } else {
        (avatar_right_ear, avatar_left_ear)
    };
    let ear_dx = left_ear_xz[0] - right_ear_xz[0];
    let ear_dz = left_ear_xz[2] - right_ear_xz[2];
    let ear_dist = (ear_dx * ear_dx + ear_dz * ear_dz).sqrt();
    // Threshold in metres — adult ear-to-ear is ~0.15 m, so 0.02 m
    // is "no horizontal information".
    if ear_dist < 0.02 {
        return None;
    }
    let yaw = ear_dz.atan2(ear_dx);

    let eye_dx = avatar_left_eye[0] - avatar_right_eye[0];
    let eye_dy = avatar_left_eye[1] - avatar_right_eye[1];
    let roll = eye_dy.atan2(eye_dx);

    let mid_eye_y = (avatar_left_eye[1] + avatar_right_eye[1]) * 0.5;
    let face_width = (eye_dx * eye_dx + eye_dy * eye_dy).sqrt().max(0.02);
    let pitch_signal = (mid_eye_y - nose[1]) / face_width;
    let pitch = pitch_signal.clamp(-2.0, 2.0).atan();

    let conf = joints[..5].iter().map(|j| j.score).fold(1.0_f32, f32::min);
    Some(FacePose {
        yaw,
        pitch,
        roll,
        confidence: conf,
    })
}

/// Build a pixel-space face bbox from CIGPose's face-68 landmarks
/// (indices 23..=90, dlib 68-point convention). Same idea as
/// [`super::rtmw3d::face::build_face_bbox_from_joints`] — body face
/// keypoints 0..=4 cluster on the upper half of the face and would
/// produce a bbox that clips the chin under the standard 1.7× pad,
/// while the face-68 set spans jaw / brows / nose / eyes / mouth and
/// gives a centroid on the actual face centre.
#[cfg(feature = "inference")]
fn build_face_bbox_from_joints_2d(
    joints: &[DecodedJoint2d],
    width: u32,
    height: u32,
) -> Option<FaceBbox> {
    let mut points_px: Vec<(f32, f32)> = Vec::with_capacity(68);
    for i in 23..=90 {
        let Some(j) = joints.get(i) else { continue };
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            continue;
        }
        points_px.push((j.nx * width as f32, j.ny * height as f32));
    }
    derive_face_bbox(&points_px, width, height)
}

// Silence unused-import warning when the `inference` feature is off:
// the `MOGE_W` / `MOGE_H` re-exports become dead in that build but
// aren't referenced anywhere user-facing.
#[cfg(all(not(feature = "inference"), test))]
mod tests {
    #[test]
    fn empty_estimate_is_well_formed() {
        let est = super::empty_estimate(42);
        assert_eq!(est.skeleton.source_timestamp, 42);
        assert!(est.skeleton.joints.is_empty());
        assert!(est.annotation.keypoints.is_empty());
    }
}

#[cfg(feature = "inference")]
#[cfg(test)]
mod tests_inference {
    use super::super::metric_depth::{INPUT_H as MOGE_H, INPUT_W as MOGE_W};
    use super::*;

    fn dummy_frame(width: u32, height: u32, point: [f32; 3]) -> MoGeFrame {
        let pixels = (width * height) as usize;
        let points = vec![point; pixels];
        MoGeFrame {
            width,
            height,
            points_m: points,
        }
    }

    #[test]
    fn sample_metric_point_returns_pixel() {
        let frame = dummy_frame(MOGE_W, MOGE_H, [0.5, 0.4, 1.2]);
        let p = sample_metric_point(&frame, 0.5, 0.5).unwrap();
        assert!((p[0] - 0.5).abs() < 1e-6);
        assert!((p[1] - 0.4).abs() < 1e-6);
        assert!((p[2] - 1.2).abs() < 1e-6);
    }

    #[test]
    fn sample_outside_frame_returns_none() {
        let frame = dummy_frame(MOGE_W, MOGE_H, [0.0, 0.0, 1.0]);
        assert!(sample_metric_point(&frame, -0.1, 0.5).is_none());
        assert!(sample_metric_point(&frame, 1.1, 0.5).is_none());
        assert!(sample_metric_point(&frame, 0.5, -0.1).is_none());
        assert!(sample_metric_point(&frame, 0.5, 1.1).is_none());
    }

    #[test]
    fn empty_estimate_is_well_formed() {
        let est = empty_estimate(42);
        assert_eq!(est.skeleton.source_timestamp, 42);
        assert!(est.skeleton.joints.is_empty());
        assert!(est.annotation.keypoints.is_empty());
    }
}
