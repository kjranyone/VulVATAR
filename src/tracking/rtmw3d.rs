//! RTMW3D — single-model whole-body 3D pose inference.
//!
//! This is the successor to `tracking::mediapipe`. RTMW3D-x (~370 MB)
//! emits 133 COCO-Wholebody keypoints in 3D from a single forward pass:
//! 17 body + 6 foot + 68 face + 21 + 21 hand. All keypoints share the
//! same coordinate system, so wrist→finger continuity is built-in
//! without per-hand crops or coordinate-frame fiddling.
//!
//! ## Pipeline
//!
//! Following Kinemotion's approach, we feed the camera frame directly
//! to the model — no person-bbox detector. VulVATAR is single-subject
//! VTubing so the implicit "centre subject" assumption holds.
//!
//! 1. **Resize** RGB to 288×384 (W×H) NCHW. Aspect is squashed (the
//!    model is robust to it; bbox-based crop would be cleaner but
//!    requires a YOLOX detector and per-frame proposal logic).
//! 2. **Normalise** with ImageNet mean/std as the rtmlib reference
//!    implementation does.
//! 3. **Infer** the model: one input, three SimCC heatmap outputs
//!    (X: `(1, 133, 576)`, Y: `(1, 133, 768)`, Z: `(1, 133, 576)`).
//! 4. **Decode** by argmax per joint per axis to get a normalised
//!    `(nx, ny, nz)` triple in `[0, 1]³`. The score at the argmax bin
//!    is the per-joint confidence.
//! 5. **Map** the 133 indices to `HumanoidBone` slots (selfie mirror)
//!    and emit a `SourceSkeleton` whose hip mid-point is the depth
//!    origin, in source-space `[-aspect, +aspect] × [-1, 1] × z`
//!    coords with `+z` toward the camera.
//!
//! Z is left in normalised model units (Kinemotion-style) — we *do not*
//! apply the rtmlib `z_range = 2.1745 m` metric conversion, because the
//! pose solver consumes relative bone directions and works fine with
//! consistently-scaled normalised z. Skipping the conversion also
//! sidesteps "what is the model's true z-range" model-by-model
//! variability.

#![allow(dead_code)]

use super::PoseEstimate;
#[cfg(feature = "inference")]
use super::face_mediapipe::FaceMeshInference;
#[cfg(feature = "inference")]
use super::yolox::{PersonBbox, YoloxPersonDetector};
#[cfg(feature = "inference")]
use super::{DetectionAnnotation, FacePose, SourceJoint, SourceSkeleton};
#[cfg(feature = "inference")]
use crate::asset::HumanoidBone;
#[cfg(feature = "inference")]
use log::{error, info, warn};
#[cfg(feature = "inference")]
use ndarray::Array4;
#[cfg(feature = "inference")]
use ort::session::Session;
#[cfg(feature = "inference")]
use ort::value::TensorRef;
#[cfg(feature = "inference")]
use std::path::Path;

/// Which ONNX Runtime execution provider the session ended up on.
/// Surfaced to the GUI so users can tell whether the GPU path is active
/// or whether DirectML failed and tracking quietly fell back to CPU.
#[derive(Clone, Debug)]
pub enum InferenceBackend {
    DirectMl,
    Cpu,
    CpuFromDirectMlFailure { reason: String },
}

impl InferenceBackend {
    pub fn label(&self) -> String {
        match self {
            Self::DirectMl => "DirectML".to_string(),
            Self::Cpu => "CPU".to_string(),
            Self::CpuFromDirectMlFailure { reason } => {
                format!("CPU (DirectML unavailable: {})", reason)
            }
        }
    }
}

#[cfg(feature = "inference")]
const INPUT_W: u32 = 288;
#[cfg(feature = "inference")]
const INPUT_H: u32 = 384;

#[cfg(feature = "inference")]
const NUM_JOINTS: usize = 133;
#[cfg(feature = "inference")]
const SIMCC_X_BINS: usize = 576;
#[cfg(feature = "inference")]
const SIMCC_Y_BINS: usize = 768;
#[cfg(feature = "inference")]
const SIMCC_Z_BINS: usize = 576;

/// ImageNet mean / std on `[0, 255]` RGB. Mirrors rtmlib's preprocessing.
#[cfg(feature = "inference")]
const MEAN_RGB: [f32; 3] = [123.675, 116.28, 103.53];
#[cfg(feature = "inference")]
const STD_RGB: [f32; 3] = [58.395, 57.12, 57.375];

/// Per-joint confidence floor. SimCC max scores cluster around 1–10 for
/// confident detections; the heatmaps are pre-softmax logits, so we map
/// to `[0, 1]` via sigmoid before thresholding so this constant matches
/// `DEFAULT_CONFIDENCE_THRESHOLD` semantics.
#[cfg(feature = "inference")]
const JOINT_CONFIDENCE_THRESHOLD: f32 = 0.3;

// ---------------------------------------------------------------------------
// COCO-Wholebody 133 keypoint indices
//
// Layout (rtmlib / mmpose convention):
//   0..=16   body 17        (COCO body)
//   17..=22  foot 6         (big toe / small toe / heel × L,R)
//   23..=90  face 68        (dlib 68-point convention)
//   91..=111 left hand 21   (subject's left)
//   112..=132 right hand 21 (subject's right)
//
// We use selfie mirror: subject's anatomical-left limb drives the
// avatar's `Right*` bones so the on-screen avatar mirrors the user.
// ---------------------------------------------------------------------------

#[cfg(feature = "inference")]
const COCO_BODY: &[(usize, HumanoidBone)] = &[
    // (index, avatar bone). Subject's left → avatar Right.
    //
    // Wrist slots (`LeftHand`, `RightHand`) are *not* listed here.
    // The body track and the dedicated hand track within RTMW3D
    // disagree about wrist position by up to ~0.4 in normalised z
    // when an object occludes the wrist (e.g. the body track
    // hallucinated a guitar-strumming wrist behind the torso while
    // the hand track placed it correctly at chest level). The hand
    // track is trained on hand-specific data and is more reliable in
    // those scenarios, so we drive `LeftHand` / `RightHand` from
    // the hand subset's wrist keypoint (index 91 / 112) inside
    // `attach_hand` below.
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

/// Big-toe landmarks → fingertips slot. COCO 17 = left big toe, 20 =
/// right big toe (subject's frame). Selfie mirror.
#[cfg(feature = "inference")]
const COCO_FOOT_TIPS: &[(usize, HumanoidBone)] = &[
    (17, HumanoidBone::RightFoot), // subject left big toe → avatar Right foot tip
    (20, HumanoidBone::LeftFoot),
];

/// MediaPipe-compatible hand landmark indices (within a single hand):
///   0 wrist
///   1..=4  thumb  (CMC, MCP, IP, TIP)
///   5..=8  index  (MCP, PIP, DIP, TIP)
///   9..=12 middle
///   13..=16 ring
///   17..=20 pinky
///
/// `HAND_PHALANGES` lists the 15 phalanx positions; `HAND_TIPS` lists
/// the five tips that go into `SourceSkeleton::fingertips` keyed by the
/// distal bone whose tip they represent.
///
/// All bones below are the **avatar Right** chain — apply selfie mirror
/// at the call site to get Left.
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

/// Per-side temporal state for a wrist. When the live frame's
/// MCP-centroid wrist is rejected by the sanity check (forearm folded
/// behind the upper arm or wildly stretched), we hold the previous
/// frame's position for up to `TEMPORAL_HOLD_MAX_FRAMES` so the avatar
/// does not snap to a default pose during a momentary occlusion.
///
/// The tracker also accumulates the running max of the upper-arm
/// length seen on clean frames, which the sanity check reads to bound
/// the plausible forearm length.
#[cfg(feature = "inference")]
#[derive(Default)]
struct WristTracker {
    last_pos: Option<[f32; 3]>,
    last_conf: f32,
    age: u32,
    /// Running max of `||elbow - shoulder||` over recent clean frames.
    /// Falls back to a generous constant before any clean frame.
    max_upper_arm: f32,
}

#[cfg(feature = "inference")]
const TEMPORAL_HOLD_MAX_FRAMES: u32 = 6;
#[cfg(feature = "inference")]
const TEMPORAL_HOLD_DECAY: f32 = 0.85;
/// Max ratio (forearm length / upper-arm length) accepted by the
/// sanity check. Real arms have upper-arm ≈ forearm so 1.4× allows
/// for measurement noise but rejects "wrist drifted to opposite end
/// of the body" pathologies.
#[cfg(feature = "inference")]
const ARM_LENGTH_RATIO_LIMIT: f32 = 1.4;

pub struct Rtmw3dInference {
    #[cfg(feature = "inference")]
    session: Session,
    #[cfg(feature = "inference")]
    input_name: String,
    #[cfg(feature = "inference")]
    output_names: Vec<String>,
    /// Optional MediaPipe FaceMeshV2 + BlendshapeV2 pipeline. Loaded
    /// when `face_landmark.onnx` and `face_blendshapes.onnx` are
    /// present in the same `models/` directory; absent → expression
    /// channel stays empty and the head bone falls back to whatever
    /// the spine direction implies.
    #[cfg(feature = "inference")]
    face_mesh: Option<FaceMeshInference>,
    /// Optional YOLOX-m human-art person detector. When present the
    /// camera frame is cropped to the largest detected person before
    /// being fed to RTMW3D — this is the principled fix for
    /// small-subject / off-centre / partial-occlusion cases. Absent →
    /// whole-frame inference (Kinemotion-style).
    #[cfg(feature = "inference")]
    person_detector: Option<YoloxPersonDetector>,
    #[cfg(feature = "inference")]
    left_wrist: WristTracker,
    #[cfg(feature = "inference")]
    right_wrist: WristTracker,
    #[cfg(feature = "inference")]
    load_warnings: Vec<String>,
    #[cfg(feature = "inference")]
    backend: InferenceBackend,
}

impl Rtmw3dInference {
    /// Instantiate from `models_dir` (typically `models/`). Looks for
    /// `rtmw3d.onnx` and tries to register the DirectML EP, falling
    /// back to CPU on EP failure.
    #[cfg(feature = "inference")]
    pub fn from_models_dir(models_dir: impl AsRef<Path>) -> Result<Self, String> {
        let models_dir = models_dir.as_ref();
        let model_path = models_dir.join("rtmw3d.onnx");
        if !model_path.is_file() {
            return Err(format!(
                "RTMW3D model not found at {}. Run dev.ps1 setup to fetch it.",
                model_path.display()
            ));
        }

        info!("Loading RTMW3D from {}", model_path.display());
        let (session, backend) =
            build_session(&model_path.to_string_lossy(), 4, "RTMW3D")?;

        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input".to_string());
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        info!(
            "RTMW3D I/O: input='{}', outputs={:?}",
            input_name, output_names
        );
        if output_names.len() < 3 {
            warn!(
                "RTMW3D reports {} outputs, expected 3 (SimCC X/Y/Z)",
                output_names.len()
            );
        }

        let mut load_warnings = Vec::new();
        let face_mesh = match FaceMeshInference::try_from_models_dir(models_dir) {
            Ok(opt) => opt,
            Err(e) => {
                let msg = format!("Face inference failed to load: {}. Expressions disabled.", e);
                warn!("{}", msg);
                load_warnings.push(msg);
                None
            }
        };

        let person_detector = match YoloxPersonDetector::try_from_models_dir(models_dir) {
            Ok(Some(d)) => {
                info!("YOLOX person detector loaded ({})", d.backend().label());
                Some(d)
            }
            Ok(None) => {
                info!("YOLOX person detector not found — using whole-frame RTMW3D");
                None
            }
            Err(e) => {
                let msg = format!(
                    "YOLOX person detector failed to load: {}. Falling back to whole-frame.",
                    e
                );
                warn!("{}", msg);
                load_warnings.push(msg);
                None
            }
        };

        Ok(Self {
            session,
            input_name,
            output_names,
            face_mesh,
            person_detector,
            left_wrist: WristTracker::default(),
            right_wrist: WristTracker::default(),
            load_warnings,
            backend,
        })
    }

    #[cfg(not(feature = "inference"))]
    pub fn from_models_dir(_: impl AsRef<std::path::Path>) -> Result<Self, String> {
        Err("RTMW3D inference requires the `inference` cargo feature".to_string())
    }

    pub fn take_load_warnings(&mut self) -> Vec<String> {
        #[cfg(feature = "inference")]
        return std::mem::take(&mut self.load_warnings);
        #[cfg(not(feature = "inference"))]
        Vec::new()
    }

    pub fn backend(&self) -> &InferenceBackend {
        #[cfg(feature = "inference")]
        return &self.backend;
        #[cfg(not(feature = "inference"))]
        unreachable!("Rtmw3dInference cannot be constructed without the `inference` feature")
    }

    pub fn estimate_pose(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> PoseEstimate {
        #[cfg(feature = "inference")]
        return self.estimate_pose_internal(rgb_data, width, height, frame_index);

        #[cfg(not(feature = "inference"))]
        super::pose_estimation::estimate_pose(rgb_data, width, height, frame_index)
    }

    #[cfg(feature = "inference")]
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

        // YOLOX-m person crop. When a person is detected we crop the
        // frame around them (with padding) before feeding it to RTMW3D
        // — this is the principled fix for small-subject / off-centre /
        // partial-occlusion inputs. Joints come back in *crop* nx/ny
        // and are remapped to original-frame coords below so the rest
        // of the pipeline (face bbox, source skeleton, solver) sees a
        // consistent coordinate system regardless of crop state.
        //
        // Falls through to whole-frame inference when no person is
        // detected or YOLOX is unavailable.
        let (infer_rgb_owned, infer_rgb_slice, infer_w, infer_h, crop_origin) = {
            let bbox_opt = self
                .person_detector
                .as_mut()
                .and_then(|d| d.detect_largest_person(rgb_data, width, height));
            match bbox_opt {
                Some(bbox) => {
                    // 25% pad: YOLOX bboxes are tight to the visible
                    // body silhouette, but RTMW3D needs slack for
                    // outstretched arms and cross-step legs that may
                    // briefly extend past the bbox between frames.
                    // Tested at 10% — produced regressions on walking
                    // / cross-step poses where the leading limb fell
                    // outside the crop.
                    let (cx1, cy1, cx2, cy2) = pad_and_clamp_bbox(&bbox, width, height, 0.25);
                    let cw = (cx2 - cx1) as u32;
                    let ch = (cy2 - cy1) as u32;
                    info!(
                        "RTMW3D: YOLOX bbox=[{:.0},{:.0},{:.0},{:.0}] score={:.2} → crop {}x{} (orig {}x{})",
                        bbox.x1, bbox.y1, bbox.x2, bbox.y2, bbox.score, cw, ch, width, height
                    );
                    if cw < 32 || ch < 32 {
                        // Bbox too small to crop usefully — fall through.
                        (None, rgb_data, width, height, None)
                    } else {
                        let crop = crop_rgb(rgb_data, width, height, cx1, cy1, cw, ch);
                        (Some(crop), &[][..], cw, ch, Some((cx1 as f32, cy1 as f32, cw as f32, ch as f32)))
                    }
                }
                None => (None, rgb_data, width, height, None),
            }
        };
        let infer_rgb: &[u8] = match &infer_rgb_owned {
            Some(buf) => buf.as_slice(),
            None => infer_rgb_slice,
        };

        // Preprocess to 288×384 NCHW with ImageNet normalization.
        let tensor = preprocess(infer_rgb, infer_w, infer_h);
        let input = match TensorRef::from_array_view(&tensor) {
            Ok(v) => v,
            Err(e) => {
                error!("RTMW3D: TensorRef creation failed: {}", e);
                return empty_estimate(frame_index);
            }
        };
        // Snapshot output names so the immutable borrow of self.output_names
        // does not conflict with the &mut Session borrow inside `run`.
        let x_name = self.output_names.first().cloned();
        let y_name = self.output_names.get(1).cloned();
        let z_name = self.output_names.get(2).cloned();

        let outputs = match self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
        {
            Ok(out) => out,
            Err(e) => {
                error!("RTMW3D: ONNX run failed: {}", e);
                return empty_estimate(frame_index);
            }
        };

        let extract = |name: &Option<String>, expected_len: usize| -> Option<Vec<f32>> {
            let n = name.as_deref()?;
            let v = outputs.get(n)?;
            let (_, data) = v.try_extract_tensor::<f32>().ok()?;
            if data.len() < expected_len {
                None
            } else {
                Some(data.to_vec())
            }
        };
        let simcc_x = match extract(&x_name, NUM_JOINTS * SIMCC_X_BINS) {
            Some(d) => d,
            None => {
                error!("RTMW3D: SimCC X output missing or wrong size");
                return empty_estimate(frame_index);
            }
        };
        let simcc_y = match extract(&y_name, NUM_JOINTS * SIMCC_Y_BINS) {
            Some(d) => d,
            None => {
                error!("RTMW3D: SimCC Y output missing or wrong size");
                return empty_estimate(frame_index);
            }
        };
        let simcc_z = match extract(&z_name, NUM_JOINTS * SIMCC_Z_BINS) {
            Some(d) => d,
            None => {
                error!("RTMW3D: SimCC Z output missing or wrong size");
                return empty_estimate(frame_index);
            }
        };
        drop(outputs);

        let mut joints = decode_simcc(&simcc_x, &simcc_y, &simcc_z);
        if let Some((ox, oy, cw, ch)) = crop_origin {
            // Remap crop-space (nx, ny) ∈ [0, 1]² back to original-frame
            // (nx, ny). nz is depth — left as-is (model emits it
            // hip-centred and pose_solver consumes only relative bone
            // directions, so the absolute z scale is not load-bearing).
            let inv_w = 1.0 / width as f32;
            let inv_h = 1.0 / height as f32;
            for j in joints.iter_mut() {
                j.nx = (ox + j.nx * cw) * inv_w;
                j.ny = (oy + j.ny * ch) * inv_h;
            }
        }
        let mut skeleton = build_source_skeleton(frame_index, &joints, width, height);

        // Per-side wrist sanity check + temporal hold. Run before
        // the face cascade so the wrist is settled when the solver
        // reads it.
        apply_wrist_temporal_hold(
            &mut skeleton,
            HumanoidBone::RightHand,
            HumanoidBone::RightShoulder,
            HumanoidBone::RightLowerArm,
            &mut self.right_wrist,
        );
        apply_wrist_temporal_hold(
            &mut skeleton,
            HumanoidBone::LeftHand,
            HumanoidBone::LeftShoulder,
            HumanoidBone::LeftLowerArm,
            &mut self.left_wrist,
        );

        // Head pose (yaw/pitch/roll) from RTMW3D's body face keypoints
        // 0..=4. These are already in source-skeleton 3D coords, so
        // the angles fall straight into the solver's frame without
        // any unprojection.
        skeleton.face = derive_face_pose_from_body(&skeleton, &joints);

        // Face crop → MediaPipe FaceMesh → BlendshapeV2. The bbox is
        // derived from RTMW3D's face-68 landmarks (indices 23..=90)
        // so the cascade does not need its own face detector.
        if let Some(face) = self.face_mesh.as_mut() {
            if let Some(bbox) = build_face_bbox_from_joints(&joints, width, height) {
                if let Some(exprs) = face.estimate(rgb_data, width, height, &bbox) {
                    skeleton.expressions = exprs;
                }
            }
        }

        let annotation = build_annotation(&joints);
        PoseEstimate {
            annotation,
            skeleton,
        }
    }
}

#[cfg(feature = "inference")]
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

// ---------------------------------------------------------------------------
// YOLOX crop helpers
// ---------------------------------------------------------------------------

/// Pad the YOLOX-detected bbox by `pad_ratio` on each side and clamp to
/// the frame. 10% padding works in practice — tight enough to keep the
/// subject filling the model input, loose enough to capture limbs that
/// briefly extend past the YOLOX box (waving arms, cross-step legs).
#[cfg(feature = "inference")]
fn pad_and_clamp_bbox(
    bbox: &PersonBbox,
    width: u32,
    height: u32,
    pad_ratio: f32,
) -> (u32, u32, u32, u32) {
    let bw = (bbox.x2 - bbox.x1).max(1.0);
    let bh = (bbox.y2 - bbox.y1).max(1.0);
    let cx = (bbox.x1 + bbox.x2) * 0.5;
    let cy = (bbox.y1 + bbox.y2) * 0.5;
    let half_w = bw * 0.5 * (1.0 + pad_ratio);
    let half_h = bh * 0.5 * (1.0 + pad_ratio);
    let x1 = (cx - half_w).max(0.0).floor() as u32;
    let y1 = (cy - half_h).max(0.0).floor() as u32;
    let x2 = (cx + half_w).min(width as f32).ceil() as u32;
    let y2 = (cy + half_h).min(height as f32).ceil() as u32;
    (x1, y1, x2, y2)
}

/// Copy the rectangle `[(ox, oy), (ox+cw, oy+ch))` out of `rgb`
/// (`width × height × 3` u8) into a contiguous tightly-packed buffer.
/// Caller has already clamped the rectangle into the source bounds.
#[cfg(feature = "inference")]
fn crop_rgb(rgb: &[u8], width: u32, _height: u32, ox: u32, oy: u32, cw: u32, ch: u32) -> Vec<u8> {
    let stride = (width as usize) * 3;
    let crop_stride = (cw as usize) * 3;
    let mut out = Vec::with_capacity(crop_stride * (ch as usize));
    for y in 0..(ch as usize) {
        let src_row = (oy as usize + y) * stride + (ox as usize) * 3;
        out.extend_from_slice(&rgb[src_row..src_row + crop_stride]);
    }
    out
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

/// Resize source RGB to `INPUT_W × INPUT_H` NCHW with ImageNet
/// normalization. Aspect is squashed (no letterbox); the model is
/// tolerant of it for centred-subject input. Bilinear sampling so
/// small / distant subjects don't lose keypoint detail to nearest's
/// quantisation.
#[cfg(feature = "inference")]
fn preprocess(rgb: &[u8], src_w: u32, src_h: u32) -> Array4<f32> {
    let dst_w = INPUT_W as usize;
    let dst_h = INPUT_H as usize;
    let mut tensor = Array4::<f32>::zeros((1, 3, dst_h, dst_w));
    if src_w == 0 || src_h == 0 {
        return tensor;
    }
    let src_w_us = src_w as usize;
    let src_h_us = src_h as usize;
    let max_x = src_w_us.saturating_sub(1);
    let max_y = src_h_us.saturating_sub(1);
    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;
    let stride = src_w_us * 3;
    let expected_len = stride * src_h_us;
    if rgb.len() < expected_len {
        return tensor;
    }

    let inv_std_r = 1.0 / STD_RGB[0];
    let inv_std_g = 1.0 / STD_RGB[1];
    let inv_std_b = 1.0 / STD_RGB[2];

    for dy in 0..dst_h {
        // OpenCV-style pixel-centre sampling: dst (dy + 0.5) maps to
        // src ((dy + 0.5) * scale_y - 0.5).
        let fy = ((dy as f32) + 0.5) * scale_y - 0.5;
        let y0f = fy.floor();
        let wy1 = (fy - y0f).clamp(0.0, 1.0);
        let wy0 = 1.0 - wy1;
        let y0 = (y0f as i32).clamp(0, max_y as i32) as usize;
        let y1 = ((y0f as i32) + 1).clamp(0, max_y as i32) as usize;
        let row0 = y0 * stride;
        let row1 = y1 * stride;
        for dx in 0..dst_w {
            let fx = ((dx as f32) + 0.5) * scale_x - 0.5;
            let x0f = fx.floor();
            let wx1 = (fx - x0f).clamp(0.0, 1.0);
            let wx0 = 1.0 - wx1;
            let x0 = (x0f as i32).clamp(0, max_x as i32) as usize;
            let x1 = ((x0f as i32) + 1).clamp(0, max_x as i32) as usize;
            let i00 = row0 + x0 * 3;
            let i01 = row0 + x1 * 3;
            let i10 = row1 + x0 * 3;
            let i11 = row1 + x1 * 3;
            let w00 = wy0 * wx0;
            let w01 = wy0 * wx1;
            let w10 = wy1 * wx0;
            let w11 = wy1 * wx1;
            let r = rgb[i00] as f32 * w00
                + rgb[i01] as f32 * w01
                + rgb[i10] as f32 * w10
                + rgb[i11] as f32 * w11;
            let g = rgb[i00 + 1] as f32 * w00
                + rgb[i01 + 1] as f32 * w01
                + rgb[i10 + 1] as f32 * w10
                + rgb[i11 + 1] as f32 * w11;
            let b = rgb[i00 + 2] as f32 * w00
                + rgb[i01 + 2] as f32 * w01
                + rgb[i10 + 2] as f32 * w10
                + rgb[i11 + 2] as f32 * w11;
            tensor[(0, 0, dy, dx)] = (r - MEAN_RGB[0]) * inv_std_r;
            tensor[(0, 1, dy, dx)] = (g - MEAN_RGB[1]) * inv_std_g;
            tensor[(0, 2, dy, dx)] = (b - MEAN_RGB[2]) * inv_std_b;
        }
    }
    tensor
}

// ---------------------------------------------------------------------------
// SimCC decode
// ---------------------------------------------------------------------------

/// One decoded keypoint in normalised model space:
/// `nx, ny ∈ [0, 1]` (image-relative, NY top-to-bottom),
/// `nz ∈ [0, 1]` (model depth axis), `score` is the per-joint
/// confidence after sigmoid.
#[cfg(feature = "inference")]
#[derive(Clone, Copy, Debug, Default)]
struct DecodedJoint {
    nx: f32,
    ny: f32,
    nz: f32,
    score: f32,
}

/// SimCC argmax with subpixel score. Returns `(bin, max_value)`.
#[cfg(feature = "inference")]
#[inline]
fn argmax_with_score(slice: &[f32]) -> (usize, f32) {
    let mut max_v = f32::NEG_INFINITY;
    let mut max_i = 0usize;
    for (i, &v) in slice.iter().enumerate() {
        if v > max_v {
            max_v = v;
            max_i = i;
        }
    }
    (max_i, max_v)
}

#[cfg(feature = "inference")]
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(feature = "inference")]
fn decode_simcc(simcc_x: &[f32], simcc_y: &[f32], simcc_z: &[f32]) -> Vec<DecodedJoint> {
    let mut out = Vec::with_capacity(NUM_JOINTS);
    for j in 0..NUM_JOINTS {
        let x_slice = &simcc_x[j * SIMCC_X_BINS..(j + 1) * SIMCC_X_BINS];
        let y_slice = &simcc_y[j * SIMCC_Y_BINS..(j + 1) * SIMCC_Y_BINS];
        let z_slice = &simcc_z[j * SIMCC_Z_BINS..(j + 1) * SIMCC_Z_BINS];
        let (xi, xs) = argmax_with_score(x_slice);
        let (yi, ys) = argmax_with_score(y_slice);
        let (zi, _zs) = argmax_with_score(z_slice);
        // Per-joint score: take the smaller of x/y heatmap peaks
        // (z is depth — its peak does not localise the joint
        // detection, only its depth) and pass through sigmoid so
        // values land in `[0, 1]` for the threshold.
        let score = sigmoid(xs.min(ys));
        out.push(DecodedJoint {
            nx: xi as f32 / SIMCC_X_BINS as f32,
            ny: yi as f32 / SIMCC_Y_BINS as f32,
            nz: zi as f32 / SIMCC_Z_BINS as f32,
            score,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// SourceSkeleton construction
// ---------------------------------------------------------------------------

#[cfg(feature = "inference")]
fn build_source_skeleton(
    frame_index: u64,
    joints: &[DecodedJoint],
    width: u32,
    height: u32,
) -> SourceSkeleton {
    let mut sk = SourceSkeleton::empty(frame_index);
    if joints.len() < NUM_JOINTS {
        return sk;
    }

    // Hip mid in normalised model space (COCO 11 = subject's left hip,
    // 12 = subject's right hip).
    let lh = &joints[11];
    let rh = &joints[12];
    let hip_score = lh.score.min(rh.score);
    if hip_score < JOINT_CONFIDENCE_THRESHOLD {
        // No reliable hip → no usable origin → bail.
        return sk;
    }
    let hip_nx = (lh.nx + rh.nx) * 0.5;
    let hip_ny = (lh.ny + rh.ny) * 0.5;
    let hip_nz = (lh.nz + rh.nz) * 0.5;

    // Source-space coord conversion. Aspect is recovered by scaling
    // the x axis: image x in [0, 1] maps to source x in
    // `[-aspect, +aspect]` where aspect = width/height. y is flipped
    // (image y-down → source y-up). z is centred on hip mid; sign
    // chosen so `+z` is toward the camera.
    //
    // The x axis is *negated* on top of that: COCO's anatomical-left
    // keypoints (e.g. index 5 LEFT_SHOULDER) sit at high image-x
    // because the subject faces the camera mirrored. Selfie mirror
    // routes those into avatar `Right*` bones, whose rest position
    // (post-VRM-Y180 root flip) is at source `-x`. So subject-left
    // (high norm_x) must end up at source `-x`. The negation makes
    // that hold.
    let aspect = width as f32 / height.max(1) as f32;
    let to_source = |j: &DecodedJoint| -> [f32; 3] {
        let sx = -(j.nx - hip_nx) * (2.0 * aspect);
        let sy = -(j.ny - hip_ny) * 2.0;
        let sz = -(j.nz - hip_nz) * 2.0;
        [sx, sy, sz]
    };

    // Mark width/height used so the unused-warning is silenced
    // when the code path skips downstream consumers of those values.
    let _ = (width, height);

    // Hips
    sk.joints.insert(
        HumanoidBone::Hips,
        SourceJoint {
            position: [0.0, 0.0, 0.0],
            confidence: hip_score,
        },
    );

    // Body bones.
    for &(idx, bone) in COCO_BODY {
        if idx >= joints.len() {
            continue;
        }
        let j = &joints[idx];
        if j.score < JOINT_CONFIDENCE_THRESHOLD {
            continue;
        }
        sk.joints.insert(
            bone,
            SourceJoint {
                position: to_source(j),
                confidence: j.score,
            },
        );
    }

    // Foot tips.
    for &(idx, bone) in COCO_FOOT_TIPS {
        if idx >= joints.len() {
            continue;
        }
        let j = &joints[idx];
        if j.score < JOINT_CONFIDENCE_THRESHOLD {
            continue;
        }
        sk.fingertips.insert(
            bone,
            SourceJoint {
                position: to_source(j),
                confidence: j.score,
            },
        );
    }

    // Hands. COCO Wholebody indices 91..=111 are the subject's left
    // hand, and 112..=132 are the subject's right hand. Within each
    // group the local index runs 0..=20 (0 = wrist). Selfie mirror:
    // subject's left → avatar Right* fingers, and the wrist landmark
    // becomes the avatar's `RightHand` / `LeftHand` bone position.
    attach_hand(
        &mut sk,
        joints,
        91,
        HumanoidBone::RightHand,
        HAND_PHALANGES_RIGHT,
        HAND_TIPS_RIGHT,
        &to_source,
    );
    attach_hand(
        &mut sk,
        joints,
        112,
        HumanoidBone::LeftHand,
        HAND_PHALANGES_LEFT,
        HAND_TIPS_LEFT,
        &to_source,
    );

    // Overall confidence: average of the body 17 keypoint scores —
    // gives the GUI status bar a coarse quality signal.
    let mut sum = 0.0_f32;
    let mut n = 0_u32;
    for j in joints.iter().take(17) {
        sum += j.score;
        n += 1;
    }
    sk.overall_confidence = if n > 0 { sum / n as f32 } else { 0.0 };

    // Face pose / blendshapes are intentionally left at None / empty
    // for v1 — the head bone falls back to spine direction. Blendshape
    // expressions need the dedicated face model (Phase C: SMIRK).
    sk.face = None;
    sk.expressions = Vec::new();
    sk
}

#[cfg(feature = "inference")]
fn attach_hand<F>(
    sk: &mut SourceSkeleton,
    joints: &[DecodedJoint],
    base_index: usize,
    wrist_bone: HumanoidBone,
    phalanges: &[(usize, HumanoidBone)],
    tips: &[(usize, HumanoidBone)],
    to_source: &F,
) where
    F: Fn(&DecodedJoint) -> [f32; 3],
{
    // Wrist position = centroid of the four finger MCP joints (index,
    // middle, ring, little — local indices 5, 9, 13, 17). The thumb
    // CMC sits anatomically away from the MCP line so it's excluded.
    //
    // We *do not* use either the body track's wrist keypoint
    // (index 9 / 10) or the hand-track's own wrist (index 91 / 112)
    // because both are unreliable when the wrist is occluded — e.g.
    // a guitar strumming hand whose wrist is hidden behind the
    // instrument lands ~0.3 units behind the torso in normalised z.
    // The MCPs anchor on the visible knuckles, so averaging four of
    // them produces a stable hand position that respects the actual
    // monocular geometry. The centroid sits ~half a hand-length
    // forward of the true wrist; that's biomechanically slightly
    // wrong but visually invisible and the avatar's solver re-anchors
    // the chain at the same point regardless.
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
        if j.score < JOINT_CONFIDENCE_THRESHOLD {
            continue;
        }
        let p = to_source(j);
        sum[0] += p[0];
        sum[1] += p[1];
        sum[2] += p[2];
        min_conf = min_conf.min(j.score);
        found += 1;
    }
    if found < 3 {
        // Need at least 3 of 4 MCPs to call this hand tracked.
        // Fewer than that → drop the whole hand chain for the frame
        // so the solver leaves fingers at rest.
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
        if j.score < JOINT_CONFIDENCE_THRESHOLD {
            continue;
        }
        sk.joints.insert(
            bone,
            SourceJoint {
                position: to_source(j),
                confidence: j.score,
            },
        );
    }
    for &(local_idx, bone) in tips {
        let global = base_index + local_idx;
        if global >= joints.len() {
            continue;
        }
        let j = &joints[global];
        if j.score < JOINT_CONFIDENCE_THRESHOLD {
            continue;
        }
        sk.fingertips.insert(
            bone,
            SourceJoint {
                position: to_source(j),
                confidence: j.score,
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Wrist sanity check + temporal hold
// ---------------------------------------------------------------------------

/// Validate the live wrist position against the upper-arm chain and
/// either accept (updating the tracker), repair from the previous
/// frame's good wrist, or drop the wrist and let the solver leave the
/// hand at rest. Also propagates fingertip / phalanx positions by the
/// same delta when we substitute a held wrist, so the hand chain stays
/// coherent.
#[cfg(feature = "inference")]
fn apply_wrist_temporal_hold(
    sk: &mut SourceSkeleton,
    wrist_bone: HumanoidBone,
    shoulder_bone: HumanoidBone,
    elbow_bone: HumanoidBone,
    tracker: &mut WristTracker,
) {
    let live_wrist = sk.joints.get(&wrist_bone).copied();
    let shoulder = sk.joints.get(&shoulder_bone).copied();
    let elbow = sk.joints.get(&elbow_bone).copied();

    // Without an elbow the sanity check has nothing to compare
    // against. Trust the live wrist if any, age the tracker.
    let Some(elbow) = elbow else {
        if live_wrist.is_some() {
            tracker.age = 0;
            tracker.last_pos = live_wrist.map(|j| j.position);
            tracker.last_conf = live_wrist.map(|j| j.confidence).unwrap_or(0.0);
        } else {
            tracker.age = tracker.age.saturating_add(1);
        }
        return;
    };

    let upper_arm = match shoulder {
        Some(s) => length3(sub3(elbow.position, s.position)),
        None => 0.0,
    };
    if upper_arm > tracker.max_upper_arm {
        tracker.max_upper_arm = upper_arm;
    }
    // Generous fallback — never let the bound drop below this so the
    // very first frame doesn't reject all hands.
    let arm_baseline = tracker.max_upper_arm.max(0.18);

    let accept_live = match live_wrist {
        Some(j) => {
            let forearm = length3(sub3(j.position, elbow.position));
            // Reject if forearm is wildly longer than the upper arm
            // (catches "wrist hallucinated across the body") or if
            // the chain folds backward sharply (forearm pointing
            // away from the elbow's continuation).
            forearm <= arm_baseline * ARM_LENGTH_RATIO_LIMIT
        }
        None => false,
    };

    if accept_live {
        let live = live_wrist.expect("checked above");
        tracker.last_pos = Some(live.position);
        tracker.last_conf = live.confidence;
        tracker.age = 0;
        return;
    }

    // Live wrist rejected (or absent). Try to substitute the held
    // value if it's recent enough, propagating the delta to the
    // finger phalanges so the hand chain doesn't tear.
    tracker.age = tracker.age.saturating_add(1);
    if tracker.age <= TEMPORAL_HOLD_MAX_FRAMES {
        if let Some(prev_pos) = tracker.last_pos {
            let decayed_conf = (tracker.last_conf * TEMPORAL_HOLD_DECAY).max(0.05);
            if let Some(live) = live_wrist {
                // Live fingers exist but the wrist itself was
                // implausible. Translate the whole chain by the
                // delta from the bogus wrist to the held position
                // so finger geometry around the held wrist stays
                // correct.
                let delta = sub3(prev_pos, live.position);
                translate_finger_chain(sk, wrist_bone, delta);
            }
            sk.joints.insert(
                wrist_bone,
                SourceJoint {
                    position: prev_pos,
                    confidence: decayed_conf,
                },
            );
            tracker.last_conf = decayed_conf;
            return;
        }
    }

    // Tertiary fallback: synthesise a wrist by extending the upper
    // arm vector from elbow forward by the same length. Better than
    // a missing hand for VTubing visibility, even if approximate.
    // Only fires when both shoulder and elbow are confident — without
    // them we can't form an arm direction.
    if let Some(shoulder) = shoulder {
        let upper = sub3(elbow.position, shoulder.position);
        let upper_len = length3(upper);
        if upper_len >= 0.05 {
            // Continue forward by the same length, slightly nudged
            // toward camera (+z) so the synthesised hand reads as
            // "extended in front" rather than "stuck inside torso".
            let mut forearm = [
                upper[0] / upper_len,
                upper[1] / upper_len,
                upper[2] / upper_len,
            ];
            forearm[2] += 0.2;
            let f_len = length3(forearm).max(1e-3);
            let synth_pos = [
                elbow.position[0] + forearm[0] / f_len * upper_len,
                elbow.position[1] + forearm[1] / f_len * upper_len,
                elbow.position[2] + forearm[2] / f_len * upper_len,
            ];
            // Drop the live finger chain (its geometry was suspect)
            // and emit only the wrist. Confidence is just above the
            // default solver threshold so the chain participates
            // but the user-visible "kept stale" state is still
            // distinguishable in confidence histograms.
            clear_finger_chain(sk, wrist_bone);
            sk.joints.insert(
                wrist_bone,
                SourceJoint {
                    position: synth_pos,
                    confidence: 0.35,
                },
            );
            tracker.last_pos = None;
            tracker.last_conf = 0.0;
            return;
        }
    }

    // No reliable signal at all → drop the wrist and finger chain.
    sk.joints.remove(&wrist_bone);
    clear_finger_chain(sk, wrist_bone);
    tracker.last_pos = None;
    tracker.last_conf = 0.0;
}

#[cfg(feature = "inference")]
#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[cfg(feature = "inference")]
#[inline]
fn length3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Per-wrist list of finger phalanx + tip bones. Used by the temporal
/// hold to translate or drop the entire hand chain coherently.
#[cfg(feature = "inference")]
fn finger_chain_bones(wrist: HumanoidBone) -> &'static [HumanoidBone] {
    match wrist {
        HumanoidBone::RightHand => &[
            HumanoidBone::RightThumbProximal,
            HumanoidBone::RightThumbIntermediate,
            HumanoidBone::RightThumbDistal,
            HumanoidBone::RightIndexProximal,
            HumanoidBone::RightIndexIntermediate,
            HumanoidBone::RightIndexDistal,
            HumanoidBone::RightMiddleProximal,
            HumanoidBone::RightMiddleIntermediate,
            HumanoidBone::RightMiddleDistal,
            HumanoidBone::RightRingProximal,
            HumanoidBone::RightRingIntermediate,
            HumanoidBone::RightRingDistal,
            HumanoidBone::RightLittleProximal,
            HumanoidBone::RightLittleIntermediate,
            HumanoidBone::RightLittleDistal,
        ],
        HumanoidBone::LeftHand => &[
            HumanoidBone::LeftThumbProximal,
            HumanoidBone::LeftThumbIntermediate,
            HumanoidBone::LeftThumbDistal,
            HumanoidBone::LeftIndexProximal,
            HumanoidBone::LeftIndexIntermediate,
            HumanoidBone::LeftIndexDistal,
            HumanoidBone::LeftMiddleProximal,
            HumanoidBone::LeftMiddleIntermediate,
            HumanoidBone::LeftMiddleDistal,
            HumanoidBone::LeftRingProximal,
            HumanoidBone::LeftRingIntermediate,
            HumanoidBone::LeftRingDistal,
            HumanoidBone::LeftLittleProximal,
            HumanoidBone::LeftLittleIntermediate,
            HumanoidBone::LeftLittleDistal,
        ],
        _ => &[],
    }
}

#[cfg(feature = "inference")]
fn clear_finger_chain(sk: &mut SourceSkeleton, wrist: HumanoidBone) {
    for b in finger_chain_bones(wrist) {
        sk.joints.remove(b);
        sk.fingertips.remove(b);
    }
}

#[cfg(feature = "inference")]
fn translate_finger_chain(sk: &mut SourceSkeleton, wrist: HumanoidBone, delta: [f32; 3]) {
    for b in finger_chain_bones(wrist) {
        if let Some(j) = sk.joints.get_mut(b) {
            j.position[0] += delta[0];
            j.position[1] += delta[1];
            j.position[2] += delta[2];
        }
        if let Some(t) = sk.fingertips.get_mut(b) {
            t.position[0] += delta[0];
            t.position[1] += delta[1];
            t.position[2] += delta[2];
        }
    }
}

// ---------------------------------------------------------------------------
// Head pose & face bbox derivation
// ---------------------------------------------------------------------------

/// Derive head yaw / pitch / roll from RTMW3D's body face keypoints
/// (0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear) in
/// source-skeleton 3D coords. Returns `None` if any keypoint is below
/// the confidence floor or the eye/ear spans are too small to give a
/// reliable angle.
///
/// All values are already mirrored at the source level (subject's
/// left ear ends up at source `+x`), so the sign conventions match
/// `quat_from_euler_ypr`.
#[cfg(feature = "inference")]
fn derive_face_pose_from_body(skeleton: &SourceSkeleton, joints: &[DecodedJoint]) -> Option<FacePose> {
    if joints.len() < 5 {
        return None;
    }
    for j in joints.iter().take(5) {
        if j.score < JOINT_CONFIDENCE_THRESHOLD {
            return None;
        }
    }
    // We need positions in source-skeleton coords. The COCO body
    // keypoints 0..=4 are not stored in `skeleton.joints` (only
    // shoulders / arms / legs are), so reconstruct via the same
    // `to_source` math. We extract hip mid from the stored Hips
    // joint (which is `(0, 0, 0)` by definition of the origin),
    // then read the 5 face keypoints in normalised space and run
    // them through the same axis flip used elsewhere.
    let hip = (joints[11].nx + joints[12].nx) * 0.5;
    let aspect = skeleton
        .joints
        .get(&HumanoidBone::LeftShoulder)
        .or_else(|| skeleton.joints.get(&HumanoidBone::RightShoulder))
        .map(|s| s.position[0].abs())
        .unwrap_or(0.0)
        / ((joints[5].nx - joints[6].nx).abs() / 2.0).max(1e-3);
    // Conservative aspect estimate: if we can't read it from the
    // stored shoulders fall back to assuming square (1.0).
    let aspect = if aspect.is_finite() && aspect > 0.5 && aspect < 4.0 { aspect } else { 1.0 };
    let to_src = |j: &DecodedJoint| -> [f32; 3] {
        let sx = -(j.nx - hip) * aspect;
        let sy = -(j.ny - (joints[11].ny + joints[12].ny) * 0.5);
        let sz = -(j.nz - (joints[11].nz + joints[12].nz) * 0.5);
        [sx, sy, sz]
    };
    // Subject-POV indices map to avatar-POV after `to_src` (selfie
    // mirror): subject's left → avatar's right and vice versa. Bind
    // the avatar-frame names directly so the geometry below reads
    // without sign confusion.
    let nose = to_src(&joints[0]);
    let avatar_right_eye = to_src(&joints[1]); // subject left eye
    let avatar_left_eye = to_src(&joints[2]);  // subject right eye
    let avatar_right_ear = to_src(&joints[3]); // subject left ear
    let avatar_left_ear = to_src(&joints[4]);  // subject right ear

    // Ear line points avatar_left_ear → avatar_right_ear along +x at
    // rest, so dx = left.x - right.x is positive when the head faces
    // camera. As the head turns to camera-right (avatar yaw +) the
    // line tilts in XZ so dz becomes positive too. yaw = atan2(dz, dx).
    let ear_dx = avatar_left_ear[0] - avatar_right_ear[0];
    let ear_dz = avatar_left_ear[2] - avatar_right_ear[2];
    let ear_dist = (ear_dx * ear_dx + ear_dz * ear_dz).sqrt();
    if ear_dist < 0.01 {
        return None;
    }
    let yaw = ear_dz.atan2(ear_dx);

    // Eye line: same handedness. Roll positive = head tilts toward
    // avatar's left ear (avatar's left eye drops below the right).
    let eye_dx = avatar_left_eye[0] - avatar_right_eye[0];
    let eye_dy = avatar_left_eye[1] - avatar_right_eye[1];
    let roll = eye_dy.atan2(eye_dx);

    let mid_eye_y = (avatar_left_eye[1] + avatar_right_eye[1]) * 0.5;
    let face_width = (eye_dx * eye_dx + eye_dy * eye_dy).sqrt().max(0.01);
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

/// Pixel-space face bbox derived from RTMW3D's 68 face landmarks
/// (indices 23..=90, dlib 68-point convention with jawline / brows /
/// nose / eyes / mouth). Body keypoints 0..=4 (nose / eyes / ears)
/// give a centroid that's too high on the face — they cluster on the
/// upper half so 1.7× padding overshoots above the head and clips
/// the chin. The face-68 set spans the whole face including mouth
/// and jaw, so its centroid lands on the actual face centre. Returns
/// `None` when too few face keypoints clear the confidence floor.
#[cfg(feature = "inference")]
fn build_face_bbox_from_joints(
    joints: &[DecodedJoint],
    width: u32,
    height: u32,
) -> Option<super::face_mediapipe::FaceBbox> {
    let mut points_px: Vec<(f32, f32)> = Vec::with_capacity(68);
    for i in 23..=90 {
        let Some(j) = joints.get(i) else { continue };
        if j.score < JOINT_CONFIDENCE_THRESHOLD {
            continue;
        }
        points_px.push((j.nx * width as f32, j.ny * height as f32));
    }
    super::face_mediapipe::derive_face_bbox(&points_px, width, height)
}

#[cfg(feature = "inference")]
fn build_annotation(joints: &[DecodedJoint]) -> DetectionAnnotation {
    // Emit the body 17 in image-normalised coords for the GUI overlay.
    // Edges follow the COCO body topology so the existing renderer can
    // draw it without changes.
    let mut keypoints = Vec::with_capacity(17);
    for j in joints.iter().take(17) {
        keypoints.push((j.nx, j.ny, j.score));
    }
    DetectionAnnotation {
        keypoints,
        skeleton: coco_body_edges(),
        bounding_box: None,
    }
}

#[cfg(feature = "inference")]
fn coco_body_edges() -> Vec<(usize, usize)> {
    vec![
        // Face
        (0, 1), (0, 2), (1, 3), (2, 4),
        // Torso
        (5, 6), (5, 11), (6, 12), (11, 12),
        // Arms
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        // Legs
        (11, 13), (13, 15),
        (12, 14), (14, 16),
    ]
}

// ---------------------------------------------------------------------------
// ONNX session helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "inference-gpu")]
pub(super) fn build_session(
    model_path: &str,
    intra_threads: usize,
    label: &str,
) -> Result<(Session, InferenceBackend), String> {
    match try_build_directml_session(model_path, intra_threads) {
        Ok(session) => {
            info!(
                "{} session created via DirectML execution provider ({})",
                label, model_path
            );
            Ok((session, InferenceBackend::DirectMl))
        }
        Err(e) => {
            warn!(
                "{} DirectML EP registration failed for '{}': {}. Falling back to CPU.",
                label, model_path, e
            );
            let session = build_cpu_session(model_path, intra_threads, label)?;
            Ok((
                session,
                InferenceBackend::CpuFromDirectMlFailure { reason: e },
            ))
        }
    }
}

#[cfg(all(feature = "inference", not(feature = "inference-gpu")))]
pub(super) fn build_session(
    model_path: &str,
    intra_threads: usize,
    label: &str,
) -> Result<(Session, InferenceBackend), String> {
    let session = build_cpu_session(model_path, intra_threads, label)?;
    Ok((session, InferenceBackend::Cpu))
}

#[cfg(feature = "inference")]
fn build_cpu_session(
    model_path: &str,
    intra_threads: usize,
    label: &str,
) -> Result<Session, String> {
    let session = Session::builder()
        .map_err(|e| format!("ORT build error ({}): {}", label, e))?
        .with_intra_threads(intra_threads)
        .map_err(|e| format!("ORT threads error ({}): {}", label, e))?
        .commit_from_file(model_path)
        .map_err(|e| format!("ORT load error ({}): {}", label, e))?;
    info!("{} session created on CPU EP ({})", label, model_path);
    Ok(session)
}

#[cfg(feature = "inference-gpu")]
fn try_build_directml_session(
    model_path: &str,
    intra_threads: usize,
) -> Result<Session, String> {
    let builder = Session::builder().map_err(|e| format!("builder: {}", e))?;
    let builder = builder
        .with_execution_providers([ort::ep::DirectML::default().build()])
        .map_err(|e| format!("with_execution_providers: {}", e))?;
    let mut builder = builder
        .with_intra_threads(intra_threads)
        .map_err(|e| format!("with_intra_threads: {}", e))?;
    builder
        .commit_from_file(model_path)
        .map_err(|e| format!("commit_from_file: {}", e))
}
