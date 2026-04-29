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

        Ok(Self {
            pose_session: session,
            pose_input_name: input_name,
            pose_output_names: output_names,
            load_warnings: Vec::new(),
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
        let landmarks_screen_data = match landmarks_name
            .as_deref()
            .and_then(|n| outputs.get(n))
            .and_then(|v| v.try_extract_tensor::<f32>().ok())
        {
            Some((_, data)) => data,
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
        let world_data = world_name
            .as_deref()
            .and_then(|n| outputs.get(n))
            .and_then(|v| v.try_extract_tensor::<f32>().ok())
            .map(|(_, data)| data.to_vec());

        let pose_conf = conf_value.map(sigmoid).unwrap_or(0.0);
        if pose_conf < POSE_VISIBILITY_THRESHOLD {
            // No person, or detector too unsure — emit empty skeleton
            // but tag it with frame index so upstream sees a fresh tick.
            return empty_estimate(frame_index);
        }

        // Decode 39 × 5 (screen) and 39 × 3 (world). Apply sigmoid to
        // the visibility/presence pair on the screen output.
        let screen = decode_landmarks_screen(landmarks_screen_data);
        let world = world_data.as_deref().map(decode_landmarks_world);

        // Map MediaPipe landmarks → source-space SourceSkeleton.
        let skeleton = build_source_skeleton(
            frame_index,
            pose_conf,
            &screen,
            world.as_deref(),
            width,
            height,
            &letterbox,
        );

        // Build an annotation overlay so the GUI's keypoint debug view
        // still works. We emit the 33 body landmarks; auxiliary 33-39
        // are skipped to avoid cluttering the overlay.
        let annotation = build_annotation(&screen, &letterbox, width, height);

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
    rect: &LetterboxRect,
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

    let aspect = (width as f32) / (height as f32);
    let to_source = |sl: &ScreenLandmark, mp_index: usize| -> Option<SourceJoint> {
        // Visibility * presence acts as the joint confidence — both
        // need to be high for the point to be trusted.
        let conf = sl.visibility.min(sl.presence);
        let (sx, sy) = rect.target_to_source(sl.x, sl.y)?;
        // Normalize source pixels into the project's [-aspect, +aspect]
        // / [-1, +1] convention (Y-up). Mirror flip horizontally —
        // VTuber selfie style: subject moves left → avatar moves to
        // its right (camera-left of screen).
        let nx = -((sx / width as f32) * 2.0 - 1.0) * aspect;
        let ny = 1.0 - (sy / height as f32) * 2.0;

        // World-z is in metric units (~ same scale as world-x/y, and
        // world-x is in roughly torso half-widths). Pull the z from
        // world output and rebase to hip; flip sign so +z points
        // toward camera (MediaPipe's convention is +z = away from
        // camera).
        let nz = world
            .and_then(|w| w.get(mp_index).copied())
            .zip(hip_world_mid)
            .map(|(p, h)| -(p[2] - h[2]))
            .unwrap_or(0.0);

        Some(SourceJoint {
            position: [nx, ny, nz],
            confidence: conf,
        })
    };

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
