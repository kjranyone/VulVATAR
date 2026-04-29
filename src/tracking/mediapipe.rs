//! MediaPipe Holistic ONNX inference.
//!
//! Phase 1 (current): body only — drives the avatar from MediaPipe Pose
//! Landmarker's 33 3D landmarks (`pose_landmark.onnx`, sourced from
//! [opencv/pose_estimation_mediapipe](https://huggingface.co/opencv/pose_estimation_mediapipe)).
//! Hands and face are deferred to P2 / P3 — the corresponding ONNX
//! sessions are not loaded yet.
//!
//! ## Pipeline
//!
//! 1. **Letterbox** the incoming RGB frame to 256×256 (preserve aspect
//!    by padding with zero pixels). Produces an NHWC `[1, 256, 256, 3]`
//!    tensor in `[0, 1]`.
//!
//! 2. **Run** the pose landmarker. The model emits five outputs (in
//!    upstream order):
//!
//!    | # | name        | shape           | role                    |
//!    |---|-------------|-----------------|-------------------------|
//!    | 0 | landmarks   | (1, 195)        | 39 × (x, y, z, vis, pres) — screen coords (256-px space) |
//!    | 1 | conf        | (1, 1)          | overall pose confidence |
//!    | 2 | mask        | (1, 256, 256, 1)| body segmentation (unused) |
//!    | 3 | heatmap     | (1, 256, 256, 39)| per-joint heatmaps (unused) |
//!    | 4 | landmarks_word | (1, 117)     | 39 × (x, y, z) — metric, hip-relative |
//!
//!    Visibility/presence come pre-sigmoid, so we apply `1 / (1 +
//!    exp(-x))` before thresholding.
//!
//! 3. **Mirror & retarget.** MediaPipe labels are subject-POV (their
//!    LEFT shoulder is on the subject's left side). VulVATAR uses
//!    selfie-style mirror so the avatar's RIGHT side is on the subject's
//!    LEFT side — see [`MEDIAPIPE_TO_HUMANOID`] for the exact mapping.
//!    Output positions are emitted in source-space coords:
//!    x ∈ [-aspect, +aspect], y ∈ [-1, +1] Y-up, z relative to the hip
//!    midpoint with `+z` toward the camera.

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
    /// DirectML EP registered successfully. GPU-accelerated.
    DirectMl,
    /// CPU EP, deliberately — either the `inference-gpu` cargo feature
    /// is off, or DirectML was never attempted for this build.
    Cpu,
    /// CPU EP because DirectML failed at registration (no DX12
    /// hardware, missing DLL, driver issue). Surfaced verbatim so the
    /// user can act on it.
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

/// Input dimensions for the pose landmarker. Hard-coded because the
/// OpenCV ONNX export pins them at 256×256 and we do not want to
/// support multiple variants until there is a reason to.
#[cfg(feature = "inference")]
const POSE_INPUT_SIZE: u32 = 256;

/// Number of landmarks emitted by the BlazePose Heavy model. Of these,
/// the first 33 are the standard humanoid keypoints (face + body + feet)
/// and the last 6 are auxiliary palm-orientation / hand-anchor points
/// that are not used by the solver — VulVATAR's hand articulation comes
/// from the dedicated MediaPipe Hand Landmarker in P2.
#[cfg(feature = "inference")]
const POSE_LANDMARK_COUNT: usize = 39;

/// Visibility / presence threshold below which a body landmark is
/// suppressed. MediaPipe outputs sigmoid-pre values; `0.5` after
/// sigmoid corresponds to `0.0` pre-sigmoid. We default to 0.3 (after
/// sigmoid) to match the project-wide `DEFAULT_CONFIDENCE_THRESHOLD`.
#[cfg(feature = "inference")]
const POSE_VISIBILITY_THRESHOLD: f32 = 0.3;

/// Input dimensions for the hand landmarker (OpenCV ONNX export).
#[cfg(feature = "inference")]
const HAND_INPUT_SIZE: u32 = 224;

/// Number of landmarks emitted by the hand model.
#[cfg(feature = "inference")]
const HAND_LANDMARK_COUNT: usize = 21;

/// Confidence floor for accepting a hand inference result.
#[cfg(feature = "inference")]
const HAND_CONFIDENCE_THRESHOLD: f32 = 0.4;

/// Padding multiplier when deriving the hand bbox from body
/// landmarks. The wrist + finger auxiliary points sit at the inner
/// edge of the hand crop, so we pad outward enough to also include
/// the fingertips.
#[cfg(feature = "inference")]
const HAND_BBOX_PAD: f32 = 1.6;

/// Input dimensions for the FaceMeshV2 landmarker (PINTO export of
/// MediaPipe `face_landmarks_detector` — input is `(1, 3, 256, 256)` NCHW).
#[cfg(feature = "inference")]
const FACE_INPUT_SIZE: u32 = 256;

/// Number of landmarks emitted by FaceMeshV2 (the dense face mesh).
#[cfg(feature = "inference")]
const FACE_LANDMARK_COUNT: usize = 478;

/// Number of landmarks the BlendshapeV2 input expects (a fixed subset
/// of the 478 mesh landmarks; see [`FACE_BLENDSHAPE_LANDMARK_SUBSET`]).
#[cfg(feature = "inference")]
const FACE_BLENDSHAPE_INPUT_COUNT: usize = 146;

/// Number of ARKit-style blendshape coefficients emitted by
/// BlendshapeV2 (matches Apple's 52-blendshape spec, with index 0 the
/// `_neutral` baseline that we drop).
#[cfg(feature = "inference")]
const FACE_BLENDSHAPE_COUNT: usize = 52;

/// Padding multiplier when deriving the face bbox from BlazePose's
/// face landmarks (nose / eyes / ears / mouth corners). BlazePose does
/// not return forehead or chin tips, so we pad generously to ensure
/// the full head fits in the crop.
#[cfg(feature = "inference")]
const FACE_BBOX_PAD: f32 = 1.7;

/// Confidence floor for accepting a face inference result (post sigmoid).
#[cfg(feature = "inference")]
const FACE_CONFIDENCE_THRESHOLD: f32 = 0.4;

/// Indices into FaceMeshV2's 478 landmarks that BlendshapeV2 expects
/// as its 146-point input. Sourced verbatim from MediaPipe's
/// `face_blendshapes_graph.cc` (`kLandmarksSubsetIdxs`).
#[cfg(feature = "inference")]
const FACE_BLENDSHAPE_LANDMARK_SUBSET: [u16; FACE_BLENDSHAPE_INPUT_COUNT] = [
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58,
    61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107,
    109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157,
    158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197,
    234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293,
    295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
    336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384,
    385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466, 468,
    469, 470, 471, 472, 473, 474, 475, 476, 477,
];

/// ARKit-compatible blendshape names in BlendshapeV2's output order.
/// Index 0 (`_neutral`) is a baseline scalar the model emits but that
/// no avatar consumes directly — we keep it in the array so positional
/// indexing matches the model and the array length is a clean 52.
#[cfg(feature = "inference")]
const FACE_BLENDSHAPE_NAMES: [&str; FACE_BLENDSHAPE_COUNT] = [
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
];

pub struct MediaPipeInference {
    #[cfg(feature = "inference")]
    pose_session: Session,
    #[cfg(feature = "inference")]
    pose_input_name: String,
    /// Output names captured at load time. The OpenCV ONNX export
    /// labels them as `Identity`/`Identity_1`/.../`Identity_4` in
    /// graph order; we rely on positional order so we read the model's
    /// declared output list directly rather than hard-coding strings.
    #[cfg(feature = "inference")]
    pose_output_names: Vec<String>,
    /// Optional MediaPipe Hand Landmarker session. Loaded when
    /// `hand_landmark.onnx` is present in the models dir; absent → hand
    /// inference is silently skipped (fingers stay at rest pose).
    #[cfg(feature = "inference")]
    hand_session: Option<Session>,
    #[cfg(feature = "inference")]
    hand_input_name: Option<String>,
    /// Hand model output order: (landmarks, conf, handedness, landmarks_word).
    #[cfg(feature = "inference")]
    hand_output_names: Vec<String>,
    /// Optional MediaPipe FaceMeshV2 + BlendshapeV2 sessions. Loaded
    /// when both `face_landmark.onnx` and `face_blendshapes.onnx` are
    /// present; if either is missing the avatar's head and expressions
    /// stay at neutral. Shape: face_landmarks output[0] is
    /// `(1, 1, 1, 1434)` — 478 × (x, y, z); blendshapes input
    /// `(1, 146, 2)`, output `(52,)`.
    #[cfg(feature = "inference")]
    face_session: Option<Session>,
    #[cfg(feature = "inference")]
    face_input_name: Option<String>,
    #[cfg(feature = "inference")]
    face_output_names: Vec<String>,
    #[cfg(feature = "inference")]
    blendshape_session: Option<Session>,
    #[cfg(feature = "inference")]
    blendshape_input_name: Option<String>,
    #[cfg(feature = "inference")]
    blendshape_output_names: Vec<String>,
    #[cfg(feature = "inference")]
    load_warnings: Vec<String>,
    #[cfg(feature = "inference")]
    backend: InferenceBackend,
}

impl MediaPipeInference {
    /// Instantiate from `models_dir` (typically `models/`). Looks for
    /// `pose_landmark.onnx` and tries to register the DirectML EP,
    /// falling back to CPU if DirectML is not available.
    #[cfg(feature = "inference")]
    pub fn from_models_dir(models_dir: impl AsRef<Path>) -> Result<Self, String> {
        let models_dir = models_dir.as_ref();
        let pose_path = models_dir.join("pose_landmark.onnx");
        if !pose_path.is_file() {
            return Err(format!(
                "MediaPipe pose model not found at {}. Run `dev.ps1` setup to fetch it.",
                pose_path.display()
            ));
        }

        info!("Loading MediaPipe Pose Landmarker from {}", pose_path.display());
        let (session, backend) = build_session(&pose_path.to_string_lossy(), 4, "MediaPipe Pose")?;

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
            "MediaPipe Pose I/O: input='{}', outputs={:?}",
            input_name, output_names
        );
        if output_names.len() < 5 {
            warn!(
                "MediaPipe Pose Landmarker reports {} outputs, expected 5",
                output_names.len()
            );
        }

        // Optional hand landmarker. Missing file → skip silently with a
        // warning surfaced to the GUI; the avatar will pose body but
        // leave fingers at rest until the user runs Install-Models.
        let hand_path = models_dir.join("hand_landmark.onnx");
        let mut load_warnings = Vec::new();
        let (hand_session, hand_input_name, hand_output_names) = if hand_path.is_file() {
            info!(
                "Loading MediaPipe Hand Landmarker from {}",
                hand_path.display()
            );
            match build_session(&hand_path.to_string_lossy(), 2, "MediaPipe Hand") {
                Ok((session, _hand_backend)) => {
                    let h_input = session
                        .inputs()
                        .first()
                        .map(|i| i.name().to_string())
                        .unwrap_or_else(|| "input".to_string());
                    let h_outputs: Vec<String> = session
                        .outputs()
                        .iter()
                        .map(|o| o.name().to_string())
                        .collect();
                    info!(
                        "MediaPipe Hand I/O: input='{}', outputs={:?}",
                        h_input, h_outputs
                    );
                    if h_outputs.len() < 4 {
                        warn!(
                            "MediaPipe Hand Landmarker reports {} outputs, expected 4",
                            h_outputs.len()
                        );
                    }
                    (Some(session), Some(h_input), h_outputs)
                }
                Err(e) => {
                    let msg = format!("Hand landmarker failed to load: {}. Fingers will not animate.", e);
                    warn!("{}", msg);
                    load_warnings.push(msg);
                    (None, None, Vec::new())
                }
            }
        } else {
            let msg = format!(
                "Hand landmarker model not found at {}. Fingers will not animate. Run dev.ps1 setup.",
                hand_path.display()
            );
            warn!("{}", msg);
            load_warnings.push(msg);
            (None, None, Vec::new())
        };

        // Optional FaceMeshV2 + BlendshapeV2. Both must be present for
        // expression weights to flow; if either is missing the
        // expression channel stays empty and the head bone falls back
        // to whatever the spine implies.
        let face_path = models_dir.join("face_landmark.onnx");
        let blendshape_path = models_dir.join("face_blendshapes.onnx");
        let (face_session, face_input_name, face_output_names) = match face_path.is_file() {
            true => {
                info!("Loading MediaPipe FaceMeshV2 from {}", face_path.display());
                match build_session(&face_path.to_string_lossy(), 2, "MediaPipe Face") {
                    Ok((s, _b)) => {
                        let f_input = s
                            .inputs()
                            .first()
                            .map(|i| i.name().to_string())
                            .unwrap_or_else(|| "input".to_string());
                        let f_outputs: Vec<String> = s
                            .outputs()
                            .iter()
                            .map(|o| o.name().to_string())
                            .collect();
                        info!(
                            "MediaPipe Face I/O: input='{}', outputs={:?}",
                            f_input, f_outputs
                        );
                        (Some(s), Some(f_input), f_outputs)
                    }
                    Err(e) => {
                        let msg = format!(
                            "FaceMeshV2 failed to load: {}. Expressions will not animate.",
                            e
                        );
                        warn!("{}", msg);
                        load_warnings.push(msg);
                        (None, None, Vec::new())
                    }
                }
            }
            false => {
                let msg = format!(
                    "FaceMeshV2 model not found at {}. Expressions will not animate. Run dev.ps1 setup.",
                    face_path.display()
                );
                warn!("{}", msg);
                load_warnings.push(msg);
                (None, None, Vec::new())
            }
        };
        let (blendshape_session, blendshape_input_name, blendshape_output_names) =
            if face_session.is_some() && blendshape_path.is_file() {
                info!(
                    "Loading MediaPipe BlendshapeV2 from {}",
                    blendshape_path.display()
                );
                match build_session(&blendshape_path.to_string_lossy(), 1, "MediaPipe Blendshape") {
                    Ok((s, _b)) => {
                        let b_input = s
                            .inputs()
                            .first()
                            .map(|i| i.name().to_string())
                            .unwrap_or_else(|| "input_points".to_string());
                        let b_outputs: Vec<String> =
                            s.outputs().iter().map(|o| o.name().to_string()).collect();
                        info!(
                            "MediaPipe Blendshape I/O: input='{}', outputs={:?}",
                            b_input, b_outputs
                        );
                        (Some(s), Some(b_input), b_outputs)
                    }
                    Err(e) => {
                        let msg = format!(
                            "BlendshapeV2 failed to load: {}. Expressions will not animate.",
                            e
                        );
                        warn!("{}", msg);
                        load_warnings.push(msg);
                        (None, None, Vec::new())
                    }
                }
            } else {
                if face_session.is_some() {
                    let msg = format!(
                        "BlendshapeV2 model not found at {}. Expressions will not animate.",
                        blendshape_path.display()
                    );
                    warn!("{}", msg);
                    load_warnings.push(msg);
                }
                (None, None, Vec::new())
            };

        Ok(Self {
            pose_session: session,
            pose_input_name: input_name,
            pose_output_names: output_names,
            hand_session,
            hand_input_name,
            hand_output_names,
            face_session,
            face_input_name,
            face_output_names,
            blendshape_session,
            blendshape_input_name,
            blendshape_output_names,
            load_warnings,
            backend,
        })
    }

    #[cfg(not(feature = "inference"))]
    pub fn from_models_dir(_models_dir: impl AsRef<std::path::Path>) -> Result<Self, String> {
        Err("Inference feature is not enabled. Build with --features inference".to_string())
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
        unreachable!("MediaPipeInference cannot be constructed without the `inference` feature")
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
        {
            super::pose_estimation::estimate_pose(rgb_data, width, height, frame_index)
        }
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

        // Letterbox the input to 256×256 in NHWC, [0, 1].
        let (tensor, letterbox) =
            preprocess_letterbox(rgb_data, width, height, POSE_INPUT_SIZE);

        let input = match TensorRef::from_array_view(&tensor) {
            Ok(v) => v,
            Err(e) => {
                error!("MediaPipe Pose: TensorRef creation failed: {}", e);
                return empty_estimate(frame_index);
            }
        };
        // Snapshot the output-name list locally so we can keep an
        // immutable borrow of those strings while `outputs` borrows
        // `self.pose_session` mutably (Session::run takes `&mut self`).
        let landmarks_name = self.pose_output_names.first().cloned();
        let conf_name = self.pose_output_names.get(1).cloned();
        let world_name = self.pose_output_names.get(4).cloned();

        let outputs = match self
            .pose_session
            .run(ort::inputs![self.pose_input_name.as_str() => input])
        {
            Ok(out) => out,
            Err(e) => {
                error!("MediaPipe Pose: ONNX run failed: {}", e);
                return empty_estimate(frame_index);
            }
        };

        // Output positional access: order set at load time by the model
        // graph. We need outputs[0] (landmarks) and outputs[4]
        // (landmarks_word). `conf` (outputs[1]) is the overall pose
        // confidence — used to gate the whole skeleton.
        //
        // We clone every borrowed tensor slice to owned `Vec<f32>` here
        // so `outputs` (which holds a mutable borrow of `pose_session`,
        // and transitively of `self`) can be dropped before we kick off
        // hand inference further down — `hand_session.run` needs its
        // own mutable borrow of `self`.
        let landmarks_screen_data: Vec<f32> = match landmarks_name
            .as_deref()
            .and_then(|n| outputs.get(n))
            .and_then(|v| v.try_extract_tensor::<f32>().ok())
        {
            Some((_, data)) => data.to_vec(),
            None => {
                error!("MediaPipe Pose: missing or unreadable landmarks output");
                return empty_estimate(frame_index);
            }
        };
        let conf_value = conf_name
            .as_deref()
            .and_then(|n| outputs.get(n))
            .and_then(|v| v.try_extract_tensor::<f32>().ok())
            .and_then(|(_, data)| data.first().copied());
        let world_data: Option<Vec<f32>> = world_name
            .as_deref()
            .and_then(|n| outputs.get(n))
            .and_then(|v| v.try_extract_tensor::<f32>().ok())
            .map(|(_, data)| data.to_vec());
        // Release the &mut borrow on self.pose_session before later
        // mutable use of self (notably hand inference below).
        drop(outputs);

        let pose_conf = conf_value.map(sigmoid).unwrap_or(0.0);
        if pose_conf < POSE_VISIBILITY_THRESHOLD {
            // No person, or detector too unsure — emit empty skeleton
            // but tag it with frame index so upstream sees a fresh tick.
            return empty_estimate(frame_index);
        }

        // Decode 39 × 5 (screen) and 39 × 3 (world). Apply sigmoid to
        // the visibility/presence pair on the screen output.
        let screen = decode_landmarks_screen(&landmarks_screen_data);
        let world = world_data.as_deref().map(decode_landmarks_world);

        // Map MediaPipe landmarks → source-space SourceSkeleton.
        let mut skeleton = build_source_skeleton(
            frame_index,
            pose_conf,
            &screen,
            world.as_deref(),
            width,
            height,
            &letterbox,
        );

        // Hand inference per side, if the hand model is loaded and the
        // body wrist for that side is tracked. Failures degrade
        // gracefully: a missing hand leaves the avatar's fingers at
        // rest pose for that frame.
        if self.hand_session.is_some() {
            for side in [HandSide::SubjectLeft, HandSide::SubjectRight] {
                let avatar_wrist_bone = side.avatar_wrist();
                let Some(wrist_joint) = skeleton.joints.get(&avatar_wrist_bone).copied()
                else {
                    continue;
                };
                self.run_hand_for_side(
                    rgb_data,
                    width,
                    height,
                    &screen,
                    &letterbox,
                    side,
                    wrist_joint.position,
                    &mut skeleton,
                );
            }
        }

        // Face inference (FaceMeshV2 → 478 landmarks, then BlendshapeV2
        // → 52 ARKit blendshape weights). Same graceful-skip semantics
        // as hand inference: any failure leaves expressions empty for
        // this frame.
        if self.face_session.is_some() {
            self.run_face(rgb_data, width, height, &screen, &letterbox, &mut skeleton);
        }

        // Build an annotation overlay so the GUI's keypoint debug view
        // still works. We emit the 33 body landmarks; auxiliary 33-39
        // are skipped to avoid cluttering the overlay.
        let annotation = build_annotation(&screen, &letterbox, width, height);

        PoseEstimate {
            annotation,
            skeleton,
        }
    }

    /// Run the hand landmarker for a single side, populating finger
    /// joints + fingertips on `sk`. Silent on failure — missing hand
    /// inference just leaves the corresponding fingers at rest pose.
    #[cfg(feature = "inference")]
    fn run_hand_for_side(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        screen: &[ScreenLandmark],
        letterbox: &LetterboxRect,
        side: HandSide,
        body_wrist_pos: [f32; 3],
        sk: &mut SourceSkeleton,
    ) {
        let bbox = match derive_hand_bbox(screen, side, letterbox, width, height) {
            Some(b) => b,
            None => return,
        };
        let tensor = crop_hand_to_tensor(rgb_data, width, height, &bbox);
        let input = match TensorRef::from_array_view(&tensor) {
            Ok(v) => v,
            Err(e) => {
                error!("MediaPipe Hand: TensorRef creation failed: {}", e);
                return;
            }
        };
        let input_name = match self.hand_input_name.as_deref() {
            Some(n) => n.to_string(),
            None => return,
        };
        // Snapshot output names so the immutable borrow of self.hand_output_names
        // does not conflict with the &mut Session borrow inside `run`.
        let world_name = self.hand_output_names.get(3).cloned();
        let conf_name = self.hand_output_names.get(1).cloned();

        let hand_session = match self.hand_session.as_mut() {
            Some(s) => s,
            None => return,
        };
        let outputs = match hand_session.run(ort::inputs![input_name.as_str() => input]) {
            Ok(out) => out,
            Err(e) => {
                error!("MediaPipe Hand: ONNX run failed ({:?}): {}", side, e);
                return;
            }
        };
        let conf_value = conf_name
            .as_deref()
            .and_then(|n| outputs.get(n))
            .and_then(|v| v.try_extract_tensor::<f32>().ok())
            .and_then(|(_, data)| data.first().copied());
        let hand_conf = match conf_value {
            Some(c) => sigmoid(c),
            None => 0.0,
        };
        if hand_conf < HAND_CONFIDENCE_THRESHOLD {
            return;
        }
        let world_data = world_name
            .as_deref()
            .and_then(|n| outputs.get(n))
            .and_then(|v| v.try_extract_tensor::<f32>().ok())
            .map(|(_, data)| data.to_vec());
        let world_data = match world_data {
            Some(d) => d,
            None => return,
        };
        let hand_world = decode_hand_world(&world_data);
        attach_hand_to_skeleton(sk, &hand_world, body_wrist_pos, side, hand_conf);
    }

    /// Run FaceMeshV2 → BlendshapeV2, populating `sk.face` and
    /// `sk.expressions`. Same graceful-skip semantics as the hand
    /// path: any failure leaves both fields untouched.
    #[cfg(feature = "inference")]
    fn run_face(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        screen: &[ScreenLandmark],
        letterbox: &LetterboxRect,
        sk: &mut SourceSkeleton,
    ) {
        let bbox = match derive_face_bbox(screen, letterbox, width, height) {
            Some(b) => b,
            None => return,
        };
        let face_tensor = crop_face_to_tensor(rgb_data, width, height, &bbox);
        let face_input_view = match TensorRef::from_array_view(&face_tensor) {
            Ok(v) => v,
            Err(e) => {
                error!("MediaPipe Face: TensorRef creation failed: {}", e);
                return;
            }
        };
        let face_input_name = match self.face_input_name.as_deref() {
            Some(n) => n.to_string(),
            None => return,
        };
        let landmarks_out_name = self.face_output_names.first().cloned();
        let conf_out_name = self.face_output_names.get(1).cloned();

        let face_session = match self.face_session.as_mut() {
            Some(s) => s,
            None => return,
        };
        let face_outputs = match face_session.run(
            ort::inputs![face_input_name.as_str() => face_input_view],
        ) {
            Ok(out) => out,
            Err(e) => {
                error!("MediaPipe Face: ONNX run failed: {}", e);
                return;
            }
        };
        let conf_value = conf_out_name
            .as_deref()
            .and_then(|n| face_outputs.get(n))
            .and_then(|v| v.try_extract_tensor::<f32>().ok())
            .and_then(|(_, data)| data.first().copied());
        let face_conf = conf_value.map(sigmoid).unwrap_or(0.0);
        if face_conf < FACE_CONFIDENCE_THRESHOLD {
            return;
        }
        let landmarks_data: Option<Vec<f32>> = landmarks_out_name
            .as_deref()
            .and_then(|n| face_outputs.get(n))
            .and_then(|v| v.try_extract_tensor::<f32>().ok())
            .map(|(_, data)| data.to_vec());
        // Drop the face outputs (and the session borrow they hold)
        // before we touch the blendshape session.
        drop(face_outputs);
        let landmarks_data = match landmarks_data {
            Some(d) => d,
            None => return,
        };
        let landmarks = decode_face_landmarks(&landmarks_data);
        // Suppress unused-warning while head-pose is body-driven.
        let _ = (&bbox, width, height);

        // Head pose (yaw / pitch / roll) is intentionally **not**
        // populated here. FaceMeshV2's Z is in arbitrary
        // face-relative pixel-equivalent units, so deriving an
        // accurate yaw from it without per-frame camera-intrinsic
        // unprojection would only fight the body model's own face
        // landmarks. The avatar's head bone falls back to the spine
        // direction in the absence of `sk.face`, which is good
        // enough for v1 — proper head pose is a follow-up using
        // BlazePose's 3D world face landmarks (indices 0..=10).

        // Blendshapes
        if self.blendshape_session.is_some() {
            let bs_input = prepare_blendshape_input(&landmarks);
            let bs_input_view = match TensorRef::from_array_view(&bs_input) {
                Ok(v) => v,
                Err(e) => {
                    error!("MediaPipe Blendshape: TensorRef creation failed: {}", e);
                    return;
                }
            };
            let bs_input_name = match self.blendshape_input_name.as_deref() {
                Some(n) => n.to_string(),
                None => return,
            };
            let bs_output_name = self.blendshape_output_names.first().cloned();
            let bs_session = match self.blendshape_session.as_mut() {
                Some(s) => s,
                None => return,
            };
            let bs_outputs = match bs_session.run(
                ort::inputs![bs_input_name.as_str() => bs_input_view],
            ) {
                Ok(out) => out,
                Err(e) => {
                    error!("MediaPipe Blendshape: ONNX run failed: {}", e);
                    return;
                }
            };
            let bs_data: Option<Vec<f32>> = bs_output_name
                .as_deref()
                .and_then(|n| bs_outputs.get(n))
                .and_then(|v| v.try_extract_tensor::<f32>().ok())
                .map(|(_, data)| data.to_vec());
            drop(bs_outputs);
            if let Some(d) = bs_data {
                let weights = decode_blendshapes(&d);
                sk.expressions = map_blendshapes_to_expressions(&weights);
            }
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

/// Letterbox an RGB frame to a square `target × target` buffer,
/// preserving aspect by padding the shorter side with black. Returns
/// the NHWC tensor `[1, target, target, 3]` in `[0, 1]` and a
/// [`LetterboxRect`] describing where the original image landed inside
/// the square so we can map landmarks back later.
#[cfg(feature = "inference")]
fn preprocess_letterbox(
    rgb: &[u8],
    width: u32,
    height: u32,
    target: u32,
) -> (Array4<f32>, LetterboxRect) {
    let target_f = target as f32;
    let scale = (target_f / width as f32).min(target_f / height as f32);
    let new_w = ((width as f32) * scale).round() as u32;
    let new_h = ((height as f32) * scale).round() as u32;
    let pad_x = ((target - new_w) / 2) as i32;
    let pad_y = ((target - new_h) / 2) as i32;

    let mut tensor = Array4::<f32>::zeros((1, target as usize, target as usize, 3));

    // Bilinear-ish (nearest, since the cost saving matters and the model
    // is robust to a few pixels' rounding) — we resample the source
    // image into the centred letterbox region.
    let inv_scale = 1.0 / scale;
    for dy in 0..(new_h as i32) {
        let sy = ((dy as f32) * inv_scale).floor() as i32;
        if sy < 0 || sy >= height as i32 {
            continue;
        }
        let row_dst_y = pad_y + dy;
        if row_dst_y < 0 || row_dst_y >= target as i32 {
            continue;
        }
        for dx in 0..(new_w as i32) {
            let sx = ((dx as f32) * inv_scale).floor() as i32;
            if sx < 0 || sx >= width as i32 {
                continue;
            }
            let row_dst_x = pad_x + dx;
            if row_dst_x < 0 || row_dst_x >= target as i32 {
                continue;
            }
            let src_idx = ((sy as usize) * (width as usize) + sx as usize) * 3;
            if src_idx + 2 >= rgb.len() {
                continue;
            }
            let r = rgb[src_idx] as f32 / 255.0;
            let g = rgb[src_idx + 1] as f32 / 255.0;
            let b = rgb[src_idx + 2] as f32 / 255.0;
            tensor[(0, row_dst_y as usize, row_dst_x as usize, 0)] = r;
            tensor[(0, row_dst_y as usize, row_dst_x as usize, 1)] = g;
            tensor[(0, row_dst_y as usize, row_dst_x as usize, 2)] = b;
        }
    }

    let rect = LetterboxRect {
        target,
        pad_x,
        pad_y,
        scaled_w: new_w,
        scaled_h: new_h,
        scale,
    };
    (tensor, rect)
}

/// Maps coordinates from the 256×256 letterbox space back to original
/// image coordinates and on into VulVATAR's source-space convention.
#[cfg(feature = "inference")]
#[derive(Clone, Copy, Debug)]
struct LetterboxRect {
    /// Side length of the square the source was letterboxed into.
    target: u32,
    /// Pixel offset of the scaled image within the square.
    pad_x: i32,
    pad_y: i32,
    /// Size of the scaled image inside the square (i.e. before padding).
    scaled_w: u32,
    scaled_h: u32,
    /// Source-pixel-to-target-pixel scale factor.
    scale: f32,
}

#[cfg(feature = "inference")]
impl LetterboxRect {
    /// Map (x_target, y_target, z_target) — landmark coords as emitted
    /// by the model, with x/y in `[0, target]` and z in arbitrary
    /// model-space units — back to source-image pixel coordinates plus
    /// an unchanged z value. Returns `None` when the target-space x/y
    /// fall outside the actual scaled-image area (i.e. inside the
    /// letterbox padding), which usually means a low-confidence
    /// landmark we will drop anyway.
    fn target_to_source(&self, lx: f32, ly: f32) -> Option<(f32, f32)> {
        let inside_x = lx - self.pad_x as f32;
        let inside_y = ly - self.pad_y as f32;
        if inside_x < 0.0 || inside_x > self.scaled_w as f32 {
            return None;
        }
        if inside_y < 0.0 || inside_y > self.scaled_h as f32 {
            return None;
        }
        Some((inside_x / self.scale, inside_y / self.scale))
    }
}

// ---------------------------------------------------------------------------
// Output decoding
// ---------------------------------------------------------------------------

#[cfg(feature = "inference")]
#[derive(Clone, Copy, Debug, Default)]
struct ScreenLandmark {
    x: f32, // pixel space, 0..target
    y: f32,
    z: f32, // depth in pixel-equivalent units, hip-relative
    visibility: f32, // post-sigmoid
    presence: f32,   // post-sigmoid
}

/// Decode the (1, 195) screen-landmarks output into 39 × `ScreenLandmark`.
/// Visibility / presence are sigmoid-pre on disk; we sigmoid them here.
#[cfg(feature = "inference")]
fn decode_landmarks_screen(data: &[f32]) -> Vec<ScreenLandmark> {
    let mut out = Vec::with_capacity(POSE_LANDMARK_COUNT);
    for i in 0..POSE_LANDMARK_COUNT {
        let base = i * 5;
        if base + 4 >= data.len() {
            break;
        }
        out.push(ScreenLandmark {
            x: data[base],
            y: data[base + 1],
            z: data[base + 2],
            visibility: sigmoid(data[base + 3]),
            presence: sigmoid(data[base + 4]),
        });
    }
    out
}

/// Decode the (1, 117) world-landmarks output into 39 × `[x, y, z]` in
/// metric units (hip-relative). We use these for the z component of
/// the source skeleton — they are roughly Euclidean and survive the
/// foreshortening that screen-z does not.
#[cfg(feature = "inference")]
fn decode_landmarks_world(data: &[f32]) -> Vec<[f32; 3]> {
    let mut out = Vec::with_capacity(POSE_LANDMARK_COUNT);
    for i in 0..POSE_LANDMARK_COUNT {
        let base = i * 3;
        if base + 2 >= data.len() {
            break;
        }
        out.push([data[base], data[base + 1], data[base + 2]]);
    }
    out
}

#[cfg(feature = "inference")]
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// MediaPipe → HumanoidBone mapping
// ---------------------------------------------------------------------------

/// Mirror-aware mapping from MediaPipe BlazePose landmark indices to
/// VulVATAR [`HumanoidBone`] slots. Selfie convention: subject's left
/// side (MediaPipe `LEFT_*`, indices 11/13/15/...) drives the avatar's
/// `Right*` bones, so when the user lifts their left arm the on-screen
/// avatar (which faces them) lifts the arm on the same screen side.
///
/// Body / face / arm / leg subset only — finger-aux landmarks
/// (17–22, pinky/index/thumb tips) are intentionally not mapped here:
/// finger articulation comes from the dedicated MediaPipe Hand
/// Landmarker in Phase 2.
#[cfg(feature = "inference")]
const MEDIAPIPE_TO_HUMANOID: &[(usize, HumanoidBone)] = &[
    // Shoulders / arms
    (11, HumanoidBone::RightShoulder),
    (11, HumanoidBone::RightUpperArm),
    (12, HumanoidBone::LeftShoulder),
    (12, HumanoidBone::LeftUpperArm),
    (13, HumanoidBone::RightLowerArm),
    (14, HumanoidBone::LeftLowerArm),
    (15, HumanoidBone::RightHand),
    (16, HumanoidBone::LeftHand),
    // Hips / legs
    (23, HumanoidBone::RightUpperLeg),
    (24, HumanoidBone::LeftUpperLeg),
    (25, HumanoidBone::RightLowerLeg),
    (26, HumanoidBone::LeftLowerLeg),
    (27, HumanoidBone::RightFoot),
    (28, HumanoidBone::LeftFoot),
];

/// Toe-tip landmarks. Stashed in `SourceSkeleton::fingertips` so the
/// solver's `Tip::Fingertip` path picks them up to drive the foot
/// bone's orientation (heel-to-toe, in the avatar's own foot frame).
#[cfg(feature = "inference")]
const MEDIAPIPE_FOOT_TIPS: &[(usize, HumanoidBone)] = &[
    (31, HumanoidBone::RightFoot), // left_foot_index → mirror → RightFoot
    (32, HumanoidBone::LeftFoot),
];

/// Build a `SourceSkeleton` from the decoded MediaPipe landmarks.
///
/// Coordinate system out: x ∈ [-aspect, +aspect], y Y-up ∈ [-1, +1],
/// z relative to hip with `+z` toward camera (matching
/// `SourceSkeleton`'s schema in [`super::source_skeleton`]).
#[cfg(feature = "inference")]
fn build_source_skeleton(
    frame_index: u64,
    pose_conf: f32,
    screen: &[ScreenLandmark],
    world: Option<&[[f32; 3]]>,
    width: u32,
    height: u32,
    _rect: &LetterboxRect,
) -> SourceSkeleton {
    let mut sk = SourceSkeleton::empty(frame_index);
    sk.overall_confidence = pose_conf;
    if screen.is_empty() {
        return sk;
    }

    // Compute hip midpoint in *world* coords so we can rebase z to it.
    // World output is hip-relative already (per MediaPipe docs) but the
    // model can drift; subtracting the hip midpoint each frame makes
    // the depth signal cleaner.
    let hip_world_mid = world.and_then(|w| {
        let l = w.get(23).copied()?;
        let r = w.get(24).copied()?;
        Some([
            (l[0] + r[0]) * 0.5,
            (l[1] + r[1]) * 0.5,
            (l[2] + r[2]) * 0.5,
        ])
    });

    // Use world coordinates for the entire 3D position — mixing
    // screen-x/y with world-z splits the source between two model
    // heads that do not perfectly agree, which slumps the rendered
    // avatar's depth into ill-defined hybrid units. World output is
    // already hip-relative metric, so the solver's direction-matching
    // gets a self-consistent 3D vector. We still keep screen for the
    // GUI overlay (separate code path).
    //
    // MediaPipe world conventions (real-world Cartesian, hip-relative):
    //   * x: subject's left (anatomical) → MediaPipe +x
    //   * y: down                        → MediaPipe +y (Y-down)
    //   * z: away from camera            → MediaPipe +z
    //
    // VulVATAR source-space (avatar convention, post Y180 root flip):
    //   * x: avatar's left  / camera-right  → +x
    //   * y: up                              → +y
    //   * z: toward camera                   → +z
    //
    // Selfie mirror: subject's anatomical left side (e.g. their left
    // hand on the right side of the screen for the user) drives the
    // *avatar's* right bones — so MediaPipe LEFT_* (index 11/13/15,
    // subject's left) is mapped to avatar Right* in
    // [`MEDIAPIPE_TO_HUMANOID`]. After that mapping, the SourceSkeleton
    // entry called "RightShoulder" carries the world position of the
    // subject's anatomical *left* shoulder, which sits at MediaPipe +x.
    // The avatar's right shoulder rest_world is at world −x (avatar's
    // anatomical right under the VRM Y180 flip), so the source
    // x-axis must be **negated** for the bone-direction match to land
    // the right way round. Y is negated to flip Y-down → Y-up, and Z
    // is negated so `+z = toward camera`.
    let to_source = |_sl: &ScreenLandmark, mp_index: usize| -> Option<SourceJoint> {
        // Visibility * presence acts as the joint confidence — both
        // need to be high for the point to be trusted.
        let conf = _sl.visibility.min(_sl.presence);
        let world_pos = world?.get(mp_index).copied()?;
        let hip = hip_world_mid?;
        let nx = -(world_pos[0] - hip[0]);
        let ny = -(world_pos[1] - hip[1]);
        let nz = -(world_pos[2] - hip[2]);
        Some(SourceJoint {
            position: [nx, ny, nz],
            confidence: conf,
        })
    };
    // Width/height parameters retained for future use (e.g. rendering
    // the source skeleton overlaid on the input image); silence the
    // dead-code warning explicitly.
    let _ = (width, height);

    // Body bones.
    for &(mp_idx, bone) in MEDIAPIPE_TO_HUMANOID {
        let Some(sl) = screen.get(mp_idx) else { continue };
        let Some(joint) = to_source(sl, mp_idx) else { continue };
        if joint.confidence >= POSE_VISIBILITY_THRESHOLD {
            sk.joints.insert(bone, joint);
        }
    }

    // Hips: midpoint of left/right hip in source space.
    if let (Some(rl), Some(rr)) = (
        sk.joints.get(&HumanoidBone::RightUpperLeg).copied(),
        sk.joints.get(&HumanoidBone::LeftUpperLeg).copied(),
    ) {
        let hips = SourceJoint {
            position: [
                (rl.position[0] + rr.position[0]) * 0.5,
                (rl.position[1] + rr.position[1]) * 0.5,
                (rl.position[2] + rr.position[2]) * 0.5,
            ],
            confidence: rl.confidence.min(rr.confidence),
        };
        sk.joints.insert(HumanoidBone::Hips, hips);
    }

    // Toe tips → fingertips slot keyed by foot bone.
    for &(mp_idx, foot_bone) in MEDIAPIPE_FOOT_TIPS {
        let Some(sl) = screen.get(mp_idx) else { continue };
        let Some(joint) = to_source(sl, mp_idx) else { continue };
        if joint.confidence >= POSE_VISIBILITY_THRESHOLD {
            sk.fingertips.insert(foot_bone, joint);
        }
    }

    // Face pose is left empty in Phase 1 — the head bone will follow
    // the body's spine direction until P3 wires up the dedicated face
    // landmarker. Setting `face = None` means the solver's face block
    // is skipped, leaving the head at rest_local rotation relative to
    // the (now spine-driven) neck.
    sk.face = None;
    sk.expressions = Vec::new();
    sk
}

#[cfg(feature = "inference")]
fn build_annotation(
    screen: &[ScreenLandmark],
    rect: &LetterboxRect,
    width: u32,
    height: u32,
) -> DetectionAnnotation {
    // Emit the 33 body landmarks in normalised image-coords [0, 1] for
    // the GUI overlay. The skeleton edge list mirrors COCO so the
    // existing renderer can draw it without a schema change.
    let mut keypoints = Vec::with_capacity(33);
    for i in 0..33.min(screen.len()) {
        let sl = &screen[i];
        let conf = sl.visibility.min(sl.presence);
        if let Some((sx, sy)) = rect.target_to_source(sl.x, sl.y) {
            let nx = (sx / width as f32).clamp(0.0, 1.0);
            let ny = (sy / height as f32).clamp(0.0, 1.0);
            keypoints.push((nx, ny, conf));
        } else {
            keypoints.push((0.0, 0.0, 0.0));
        }
    }
    let skeleton = blazepose_skeleton_edges();
    DetectionAnnotation {
        keypoints,
        skeleton,
        bounding_box: None,
    }
}

// ---------------------------------------------------------------------------
// Hand inference (Phase 2)
//
// The MediaPipe Hand Landmarker is a 224×224 single-hand 3D landmarker.
// We run it once per detected hand (up to twice per frame) using a
// body-derived bounding box: BlazePose emits four auxiliary landmarks
// per hand (wrist + pinky/index/thumb knuckles) which are tight enough
// to skip the standalone palm detector for non-overlapping hand poses.
//
// Outputs (positional, set by the OpenCV ONNX export):
//   0. landmarks (1, 63)        — 21 × (x, y, z) screen, x/y in 224-pixel space
//   1. conf      (1, 1)         — overall hand confidence (post-sigmoid)
//   2. handedness(1, 1)         — 0 = left, 1 = right (we ignore: bbox tells us)
//   3. world     (1, 63)        — 21 × (x, y, z) metric, hand-center origin
//
// We use the WORLD output for finger geometry (same coord system as
// body world: x right, y down, z away from camera, all in meters).
// After axis flip and a translation that snaps `wrist[0]` onto the
// body's wrist position, the per-finger directions feed the solver
// using the same "joint position = bone start" convention as the body
// chain — the existing finger entries in [`POSE_BONES_TIPS`] handle
// the rest.
// ---------------------------------------------------------------------------

#[cfg(feature = "inference")]
#[derive(Clone, Copy, Debug)]
enum HandSide {
    /// Subject's anatomical left hand. Body landmarks 15/17/19/21
    /// drive the bbox; the result is mapped to avatar **Right** fingers
    /// (selfie mirror).
    SubjectLeft,
    /// Subject's anatomical right hand → avatar Left fingers.
    SubjectRight,
}

#[cfg(feature = "inference")]
impl HandSide {
    fn body_landmark_indices(self) -> [usize; 4] {
        match self {
            // wrist, pinky, index, thumb (BlazePose body model)
            HandSide::SubjectLeft => [15, 17, 19, 21],
            HandSide::SubjectRight => [16, 18, 20, 22],
        }
    }

    fn avatar_wrist(self) -> HumanoidBone {
        match self {
            HandSide::SubjectLeft => HumanoidBone::RightHand,
            HandSide::SubjectRight => HumanoidBone::LeftHand,
        }
    }

    fn joint_mapping(self) -> &'static [(usize, HumanoidBone)] {
        match self {
            HandSide::SubjectLeft => HAND_JOINTS_AVATAR_RIGHT,
            HandSide::SubjectRight => HAND_JOINTS_AVATAR_LEFT,
        }
    }

    fn tip_mapping(self) -> &'static [(usize, HumanoidBone)] {
        match self {
            HandSide::SubjectLeft => HAND_TIPS_AVATAR_RIGHT,
            HandSide::SubjectRight => HAND_TIPS_AVATAR_LEFT,
        }
    }
}

/// Hand-landmark index → avatar bone, for the avatar's *right* finger
/// chain. Subject's left hand inference fills these slots (mirror).
///
/// MediaPipe hand landmark indices:
///   0      = WRIST (handled separately as anchor)
///   1..=4  = THUMB:  CMC, MCP, IP, TIP
///   5..=8  = INDEX:  MCP, PIP, DIP, TIP
///   9..=12 = MIDDLE: MCP, PIP, DIP, TIP
///   13..=16= RING:   MCP, PIP, DIP, TIP
///   17..=20= PINKY:  MCP, PIP, DIP, TIP
///
/// The `*Distal` bone's tip is in [`HAND_TIPS_*`] (keyed by distal
/// bone). The proximal/intermediate joints carry positions of the
/// joints they *start at*, matching the body convention where
/// `joints[Bone].position = start of Bone`.
#[cfg(feature = "inference")]
const HAND_JOINTS_AVATAR_RIGHT: &[(usize, HumanoidBone)] = &[
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
const HAND_TIPS_AVATAR_RIGHT: &[(usize, HumanoidBone)] = &[
    (4, HumanoidBone::RightThumbDistal),
    (8, HumanoidBone::RightIndexDistal),
    (12, HumanoidBone::RightMiddleDistal),
    (16, HumanoidBone::RightRingDistal),
    (20, HumanoidBone::RightLittleDistal),
];

#[cfg(feature = "inference")]
const HAND_JOINTS_AVATAR_LEFT: &[(usize, HumanoidBone)] = &[
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
const HAND_TIPS_AVATAR_LEFT: &[(usize, HumanoidBone)] = &[
    (4, HumanoidBone::LeftThumbDistal),
    (8, HumanoidBone::LeftIndexDistal),
    (12, HumanoidBone::LeftMiddleDistal),
    (16, HumanoidBone::LeftRingDistal),
    (20, HumanoidBone::LeftLittleDistal),
];

/// Square crop of the source image, in source-pixel coordinates.
#[cfg(feature = "inference")]
#[derive(Clone, Copy, Debug)]
struct HandBbox {
    /// Top-left corner in source pixels (may be negative if the bbox
    /// extends past the left/top edge of the source — we clamp later).
    x: f32,
    y: f32,
    /// Side length in source pixels (square).
    size: f32,
}

/// Derive a square hand bounding box from the body's auxiliary hand
/// landmarks (wrist, pinky, index, thumb). Returns `None` when the
/// landmarks are too occluded or the bbox would be degenerately small.
///
/// Strategy: take the four hand-region body landmarks in source-pixel
/// space, compute their centroid, then expand to a square that
/// contains all four points padded by [`HAND_BBOX_PAD`]. The pad
/// multiplier accounts for the fact that BlazePose's "pinky/index/thumb"
/// markers sit at the *knuckles* — fingertips extend further out and
/// would otherwise be cropped off.
#[cfg(feature = "inference")]
fn derive_hand_bbox(
    screen: &[ScreenLandmark],
    side: HandSide,
    rect: &LetterboxRect,
    source_w: u32,
    source_h: u32,
) -> Option<HandBbox> {
    let indices = side.body_landmark_indices();
    let mut points: Vec<(f32, f32)> = Vec::with_capacity(4);
    let mut min_conf = 1.0_f32;
    for idx in indices {
        let sl = screen.get(idx)?;
        let conf = sl.visibility.min(sl.presence);
        if conf < POSE_VISIBILITY_THRESHOLD {
            return None;
        }
        min_conf = min_conf.min(conf);
        let (sx, sy) = rect.target_to_source(sl.x, sl.y)?;
        points.push((sx, sy));
    }
    if points.is_empty() {
        return None;
    }

    let cx = points.iter().map(|p| p.0).sum::<f32>() / points.len() as f32;
    let cy = points.iter().map(|p| p.1).sum::<f32>() / points.len() as f32;
    let max_d = points
        .iter()
        .map(|(x, y)| ((x - cx).powi(2) + (y - cy).powi(2)).sqrt())
        .fold(0.0_f32, f32::max);
    let half = (max_d * HAND_BBOX_PAD).max(8.0);
    let size = half * 2.0;

    // Reject bboxes wholly outside the image (the wrist might be
    // tracked but visually clipped).
    if cx + half < 0.0 || cy + half < 0.0 {
        return None;
    }
    if cx - half > source_w as f32 || cy - half > source_h as f32 {
        return None;
    }
    Some(HandBbox {
        x: cx - half,
        y: cy - half,
        size,
    })
}

/// Crop the source RGB image to the hand bbox, resizing to 224×224
/// NHWC `[0,1]`. Out-of-image samples are zero-padded (model is robust
/// to black borders).
#[cfg(feature = "inference")]
fn crop_hand_to_tensor(
    rgb: &[u8],
    width: u32,
    height: u32,
    bbox: &HandBbox,
) -> Array4<f32> {
    let target = HAND_INPUT_SIZE as usize;
    let mut tensor = Array4::<f32>::zeros((1, target, target, 3));
    let stride = (width as usize) * 3;
    let scale = bbox.size / target as f32;
    for dy in 0..target {
        let sy = (bbox.y + (dy as f32 + 0.5) * scale) as i32;
        if sy < 0 || sy >= height as i32 {
            continue;
        }
        for dx in 0..target {
            let sx = (bbox.x + (dx as f32 + 0.5) * scale) as i32;
            if sx < 0 || sx >= width as i32 {
                continue;
            }
            let src_idx = (sy as usize) * stride + (sx as usize) * 3;
            if src_idx + 2 >= rgb.len() {
                continue;
            }
            tensor[(0, dy, dx, 0)] = rgb[src_idx] as f32 / 255.0;
            tensor[(0, dy, dx, 1)] = rgb[src_idx + 1] as f32 / 255.0;
            tensor[(0, dy, dx, 2)] = rgb[src_idx + 2] as f32 / 255.0;
        }
    }
    tensor
}

/// Decode the (1, 63) hand world-landmarks output into 21 × `[x, y, z]`.
/// Units: metres, hand-center origin (per MediaPipe spec). Coordinate
/// axes match the body world output (X right, Y down, Z away from
/// camera) so the same axis flip applies before adding into the source
/// skeleton.
#[cfg(feature = "inference")]
fn decode_hand_world(data: &[f32]) -> Vec<[f32; 3]> {
    let mut out = Vec::with_capacity(HAND_LANDMARK_COUNT);
    for i in 0..HAND_LANDMARK_COUNT {
        let base = i * 3;
        if base + 2 >= data.len() {
            break;
        }
        out.push([data[base], data[base + 1], data[base + 2]]);
    }
    out
}

/// Translate the 21 hand-center-relative world landmarks so that
/// landmark 0 (wrist) coincides with the body's wrist joint, then
/// insert the per-bone joints + per-distal tips into `sk` keyed by
/// avatar-side bones.
///
/// `body_wrist_pos` must already be in source-skeleton coords (the
/// post-flip frame: x ∈ [-aspect, +aspect], y up, +z toward camera).
/// The hand world axes are flipped here to match.
#[cfg(feature = "inference")]
fn attach_hand_to_skeleton(
    sk: &mut SourceSkeleton,
    hand_world: &[[f32; 3]],
    body_wrist_pos: [f32; 3],
    side: HandSide,
    hand_conf: f32,
) {
    if hand_world.len() < HAND_LANDMARK_COUNT {
        return;
    }
    // Same axis flip as body world: meters with X right/Y down/Z away
    // → source-skeleton X left+ / Y up / Z toward-camera.
    let flip = |p: [f32; 3]| -> [f32; 3] { [-p[0], -p[1], -p[2]] };
    let wrist_flipped = flip(hand_world[0]);
    let offset = [
        body_wrist_pos[0] - wrist_flipped[0],
        body_wrist_pos[1] - wrist_flipped[1],
        body_wrist_pos[2] - wrist_flipped[2],
    ];
    let to_source = |idx: usize| -> [f32; 3] {
        let f = flip(hand_world[idx]);
        [f[0] + offset[0], f[1] + offset[1], f[2] + offset[2]]
    };

    for &(mp_idx, bone) in side.joint_mapping() {
        let pos = to_source(mp_idx);
        sk.joints.insert(
            bone,
            SourceJoint {
                position: pos,
                confidence: hand_conf,
            },
        );
    }
    for &(mp_idx, distal_bone) in side.tip_mapping() {
        let pos = to_source(mp_idx);
        sk.fingertips.insert(
            distal_bone,
            SourceJoint {
                position: pos,
                confidence: hand_conf,
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Face inference (Phase 3)
//
// FaceMeshV2 (478 landmarks, 256×256 NCHW) + BlendshapeV2 (146 subset
// → 52 ARKit blendshape coefficients). The face bbox is derived from
// BlazePose's head landmarks (0=nose, 1..=8 eyes/ears, 9..=10 mouth
// corners) — generously padded because BlazePose has no forehead /
// chin point and the bbox would otherwise clip the top of the head.
//
// Head pose (yaw / pitch / roll) is computed from BlazePose's *world*
// face landmarks (3D, hip-relative) — not the FaceMeshV2 output, whose
// Z is in arbitrary pixel-depth units. Blendshape weights are mapped
// to VRM 1.0 expression names where there is a clean correspondence
// (blink, mouth presets) and passed through verbatim otherwise so the
// avatar's expression set can pick up names like `mouthSmileLeft`
// directly when the rig defines them.
// ---------------------------------------------------------------------------

#[cfg(feature = "inference")]
#[derive(Clone, Copy, Debug)]
struct FaceBbox {
    /// Top-left source pixel.
    x: f32,
    y: f32,
    /// Square side length in source pixels.
    size: f32,
}

/// Derive a face bbox from BlazePose's 11 face-region landmarks
/// (indices 0..=10: nose, eyes, ears, mouth corners). Returns `None`
/// when fewer than 4 landmarks clear the visibility threshold or when
/// the bbox would lie wholly outside the source image.
#[cfg(feature = "inference")]
fn derive_face_bbox(
    screen: &[ScreenLandmark],
    rect: &LetterboxRect,
    source_w: u32,
    source_h: u32,
) -> Option<FaceBbox> {
    let mut points: Vec<(f32, f32)> = Vec::with_capacity(11);
    for idx in 0..=10 {
        let Some(sl) = screen.get(idx) else { continue };
        let conf = sl.visibility.min(sl.presence);
        if conf < POSE_VISIBILITY_THRESHOLD {
            continue;
        }
        let Some((sx, sy)) = rect.target_to_source(sl.x, sl.y) else { continue };
        points.push((sx, sy));
    }
    if points.len() < 4 {
        return None;
    }
    let cx = points.iter().map(|p| p.0).sum::<f32>() / points.len() as f32;
    let cy = points.iter().map(|p| p.1).sum::<f32>() / points.len() as f32;
    let max_d = points
        .iter()
        .map(|(x, y)| ((x - cx).powi(2) + (y - cy).powi(2)).sqrt())
        .fold(0.0_f32, f32::max);
    let half = (max_d * FACE_BBOX_PAD).max(16.0);
    let size = half * 2.0;
    if cx + half < 0.0 || cy + half < 0.0 {
        return None;
    }
    if cx - half > source_w as f32 || cy - half > source_h as f32 {
        return None;
    }
    Some(FaceBbox {
        x: cx - half,
        y: cy - half,
        size,
    })
}

/// Crop the source RGB image to the face bbox and produce an NCHW
/// `[1, 3, 256, 256]` float tensor in `[0, 1]`. Out-of-image samples
/// are zero-padded.
///
/// Note: shape is **NCHW** (channels-first) for FaceMeshV2 — the body
/// and hand models above use NHWC. The PINTO export of
/// `face_landmarks_detector` keeps the canonical TensorFlow Lite
/// channels-last layout repacked, but the OpenCV TensorFlow.js export
/// flipped it; we match the PINTO copy we ship.
#[cfg(feature = "inference")]
fn crop_face_to_tensor(
    rgb: &[u8],
    width: u32,
    height: u32,
    bbox: &FaceBbox,
) -> Array4<f32> {
    let target = FACE_INPUT_SIZE as usize;
    let mut tensor = Array4::<f32>::zeros((1, 3, target, target));
    let stride = (width as usize) * 3;
    let scale = bbox.size / target as f32;
    for dy in 0..target {
        let sy = (bbox.y + (dy as f32 + 0.5) * scale) as i32;
        if sy < 0 || sy >= height as i32 {
            continue;
        }
        for dx in 0..target {
            let sx = (bbox.x + (dx as f32 + 0.5) * scale) as i32;
            if sx < 0 || sx >= width as i32 {
                continue;
            }
            let src_idx = (sy as usize) * stride + (sx as usize) * 3;
            if src_idx + 2 >= rgb.len() {
                continue;
            }
            tensor[(0, 0, dy, dx)] = rgb[src_idx] as f32 / 255.0;
            tensor[(0, 1, dy, dx)] = rgb[src_idx + 1] as f32 / 255.0;
            tensor[(0, 2, dy, dx)] = rgb[src_idx + 2] as f32 / 255.0;
        }
    }
    tensor
}

/// Decode the FaceMeshV2 landmarks output `(1, 1, 1, 1434)` into 478
/// × `[x, y, z]`. Coordinates are in 256-pixel space (matching the
/// model input), with z in pixel-equivalent depth units relative to
/// the face's geometric centre.
#[cfg(feature = "inference")]
fn decode_face_landmarks(data: &[f32]) -> Vec<[f32; 3]> {
    let mut out = Vec::with_capacity(FACE_LANDMARK_COUNT);
    for i in 0..FACE_LANDMARK_COUNT {
        let base = i * 3;
        if base + 2 >= data.len() {
            break;
        }
        out.push([data[base], data[base + 1], data[base + 2]]);
    }
    out
}

/// Build the BlendshapeV2 input tensor `(1, 146, 2)` by selecting the
/// 146-landmark subset and dividing x/y by the face crop input size
/// to get normalised `[0, 1]` coordinates.
#[cfg(feature = "inference")]
fn prepare_blendshape_input(landmarks: &[[f32; 3]]) -> ndarray::Array3<f32> {
    let mut tensor = ndarray::Array3::<f32>::zeros((1, FACE_BLENDSHAPE_INPUT_COUNT, 2));
    let scale = 1.0 / FACE_INPUT_SIZE as f32;
    for (out_i, &mp_idx) in FACE_BLENDSHAPE_LANDMARK_SUBSET.iter().enumerate() {
        if let Some(p) = landmarks.get(mp_idx as usize) {
            tensor[(0, out_i, 0)] = p[0] * scale;
            tensor[(0, out_i, 1)] = p[1] * scale;
        }
    }
    tensor
}

/// Decode the BlendshapeV2 output `(52,)` into a fixed array.
#[cfg(feature = "inference")]
fn decode_blendshapes(data: &[f32]) -> [f32; FACE_BLENDSHAPE_COUNT] {
    let mut out = [0.0_f32; FACE_BLENDSHAPE_COUNT];
    let n = data.len().min(FACE_BLENDSHAPE_COUNT);
    out[..n].copy_from_slice(&data[..n]);
    out
}

/// Map the 52 ARKit-style blendshape weights to a list of
/// [`SourceExpression`] entries for the avatar to consume. We emit:
///
/// * Verbatim ARKit names for the 51 non-`_neutral` channels — VRM 1.0
///   custom expressions can match these exactly when the rig defines
///   them.
/// * VRM 1.0 preset aggregations (`blinkLeft`, `blinkRight`, `aa`,
///   `happy`, `sad`, `surprised`) so even a stock VRM with no custom
///   blendshapes shows live facial motion. Mirror convention applies:
///   the subject's left eye drives `blinkRight`.
#[cfg(feature = "inference")]
fn map_blendshapes_to_expressions(
    weights: &[f32; FACE_BLENDSHAPE_COUNT],
) -> Vec<super::SourceExpression> {
    let mut out: Vec<super::SourceExpression> = Vec::with_capacity(FACE_BLENDSHAPE_COUNT + 8);
    // 1. Pass-through ARKit names (skip index 0 = `_neutral`).
    for (i, name) in FACE_BLENDSHAPE_NAMES.iter().enumerate().skip(1) {
        out.push(super::SourceExpression {
            name: (*name).to_string(),
            weight: weights[i].clamp(0.0, 1.0),
        });
    }
    // 2. VRM 1.0 preset aggregations. These are *additional* entries
    //    so a rig that maps the ARKit names directly is unaffected;
    //    a rig that only knows VRM presets still gets sensible motion.
    let blink_left_subj = weights[9].clamp(0.0, 1.0);   // eyeBlinkLeft (subject)
    let blink_right_subj = weights[10].clamp(0.0, 1.0); // eyeBlinkRight
    // Selfie mirror: subject's left eye → avatar's right eye.
    out.push(super::SourceExpression { name: "blinkRight".into(), weight: blink_left_subj });
    out.push(super::SourceExpression { name: "blinkLeft".into(),  weight: blink_right_subj });
    out.push(super::SourceExpression {
        name: "blink".into(),
        weight: ((blink_left_subj + blink_right_subj) * 0.5).clamp(0.0, 1.0),
    });
    let jaw_open = weights[25].clamp(0.0, 1.0);
    out.push(super::SourceExpression { name: "aa".into(), weight: jaw_open });
    let smile = ((weights[44] + weights[45]) * 0.5).clamp(0.0, 1.0);
    out.push(super::SourceExpression { name: "happy".into(), weight: smile });
    let frown = ((weights[30] + weights[31]) * 0.5).clamp(0.0, 1.0);
    out.push(super::SourceExpression { name: "sad".into(), weight: frown });
    // Surprise = brow inner up *and* mouth open, AND-gated.
    let brow_up = weights[3].clamp(0.0, 1.0);
    let surprise = (brow_up * jaw_open).sqrt();
    out.push(super::SourceExpression { name: "surprised".into(), weight: surprise });
    out
}

/// Derive a [`FacePose`] (yaw / pitch / roll) from BlazePose's *world*
/// face landmarks. We use the body model's 3D output rather than
/// FaceMeshV2 because the latter's Z axis is arbitrary face-relative
/// pixels — not aligned with the source-skeleton frame. BlazePose's
/// face landmarks are noisy but consistent with the rest of the body
/// world, which is what the head bone solver needs.
///
/// * yaw — angle of the line from `right_ear → left_ear` projected on
///   the XZ plane.
/// * roll — tilt of the line `right_eye → left_eye` in the XY plane.
/// * pitch — vertical position of the nose relative to the mid-eye
///   point, normalised by ear-to-ear distance.
///
/// All angles are returned in radians with the sign convention
/// expected by `quat_from_euler_ypr`.
#[cfg(feature = "inference")]
#[allow(dead_code)]
fn derive_face_pose_from_body_world(
    body_world: &[[f32; 3]],
    overall_conf: f32,
) -> Option<super::FacePose> {
    let nose = body_world.get(0)?;
    let l_eye = body_world.get(2)?;
    let r_eye = body_world.get(5)?;
    let l_ear = body_world.get(7)?;
    let r_ear = body_world.get(8)?;
    // Apply the same axis flip as build_source_skeleton (negate xyz).
    let flip = |p: &[f32; 3]| [-p[0], -p[1], -p[2]];
    let nose = flip(nose);
    let l_eye = flip(l_eye);
    let r_eye = flip(r_eye);
    let l_ear = flip(l_ear);
    let r_ear = flip(r_ear);

    // Selfie mirror: subject's "left ear" sits on the source +x side
    // (after flip), which is the *avatar's* left. So the ear vector
    // points from right_ear (-x side) to left_ear (+x side).
    let ear_dx = l_ear[0] - r_ear[0];
    let ear_dz = l_ear[2] - r_ear[2];
    let ear_dist = (ear_dx * ear_dx + ear_dz * ear_dz).sqrt();
    if ear_dist < 0.02 {
        return None;
    }
    // yaw: positive = head turning around +Y so the subject looks
    // toward camera-right (the avatar's left). atan2(-dz, dx) gives
    // exactly that sign.
    let yaw = (-ear_dz).atan2(ear_dx);
    // roll: angle of the eye line in the XY plane. Positive roll =
    // head tilts toward the avatar's left ear (toward +x).
    let eye_dx = l_eye[0] - r_eye[0];
    let eye_dy = l_eye[1] - r_eye[1];
    let roll = eye_dy.atan2(eye_dx);
    // pitch: signed offset of the nose below the mid-eye line,
    // normalised by face width. Negative offset (nose above eyes) →
    // head tilted back; positive offset → chin down.
    let mid_eye_y = (l_eye[1] + r_eye[1]) * 0.5;
    let face_width = (eye_dx * eye_dx + eye_dy * eye_dy).sqrt().max(0.01);
    let pitch_signal = (mid_eye_y - nose[1]) / face_width;
    let pitch = pitch_signal.clamp(-2.0, 2.0).atan();

    Some(super::FacePose {
        yaw,
        pitch,
        roll,
        confidence: overall_conf.clamp(0.0, 1.0),
    })
}

/// Edge list for the BlazePose 33-point skeleton, suitable for the
/// keypoint-overlay renderer. Indices match MediaPipe's documented
/// landmark numbering (0=nose, 11/12=shoulders, etc.).
#[cfg(feature = "inference")]
fn blazepose_skeleton_edges() -> Vec<(usize, usize)> {
    vec![
        // Face contour (eyes, ears, mouth)
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        // Torso
        (11, 12),
        (11, 23),
        (12, 24),
        (23, 24),
        // Left arm
        (11, 13),
        (13, 15),
        // Right arm
        (12, 14),
        (14, 16),
        // Left leg
        (23, 25),
        (25, 27),
        (27, 29),
        (29, 31),
        // Right leg
        (24, 26),
        (26, 28),
        (28, 30),
        (30, 32),
    ]
}

// ---------------------------------------------------------------------------
// ONNX session helpers (duplicated from the legacy inference module so
// this file is self-contained — the legacy file is on the P4 chopping
// block and we do not want a transient cross-module dependency).
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
