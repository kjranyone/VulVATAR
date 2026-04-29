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
use super::{DetectionAnnotation, SourceJoint, SourceSkeleton};
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

pub struct Rtmw3dInference {
    #[cfg(feature = "inference")]
    session: Session,
    #[cfg(feature = "inference")]
    input_name: String,
    #[cfg(feature = "inference")]
    output_names: Vec<String>,
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

        Ok(Self {
            session,
            input_name,
            output_names,
            load_warnings: Vec::new(),
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

        // Preprocess to 288×384 NCHW with ImageNet normalization.
        let tensor = preprocess(rgb_data, width, height);
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

        let joints = decode_simcc(&simcc_x, &simcc_y, &simcc_z);
        let skeleton = build_source_skeleton(frame_index, &joints, width, height);
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
// Preprocessing
// ---------------------------------------------------------------------------

/// Resize source RGB to `INPUT_W × INPUT_H` NCHW with ImageNet
/// normalization. Aspect is squashed (no letterbox); the model is
/// tolerant of it for centred-subject input.
#[cfg(feature = "inference")]
fn preprocess(rgb: &[u8], src_w: u32, src_h: u32) -> Array4<f32> {
    let dst_w = INPUT_W as usize;
    let dst_h = INPUT_H as usize;
    let mut tensor = Array4::<f32>::zeros((1, 3, dst_h, dst_w));
    let inv_dx = src_w as f32 / dst_w as f32;
    let inv_dy = src_h as f32 / dst_h as f32;
    for dy in 0..dst_h {
        let sy = ((dy as f32 + 0.5) * inv_dy) as usize;
        let sy = sy.min((src_h as usize).saturating_sub(1));
        for dx in 0..dst_w {
            let sx = ((dx as f32 + 0.5) * inv_dx) as usize;
            let sx = sx.min((src_w as usize).saturating_sub(1));
            let si = (sy * src_w as usize + sx) * 3;
            if si + 2 >= rgb.len() {
                continue;
            }
            let r = rgb[si] as f32;
            let g = rgb[si + 1] as f32;
            let b = rgb[si + 2] as f32;
            tensor[(0, 0, dy, dx)] = (r - MEAN_RGB[0]) / STD_RGB[0];
            tensor[(0, 1, dy, dx)] = (g - MEAN_RGB[1]) / STD_RGB[1];
            tensor[(0, 2, dy, dx)] = (b - MEAN_RGB[2]) / STD_RGB[2];
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
fn build_session(
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
fn build_session(
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
