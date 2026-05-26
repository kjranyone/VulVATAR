//! MediaPipe Hand Landmarker (CPU EP) — 21 keypoint per hand, 3D output.
//!
//! Pairs with HMR2 body to add per-finger pose without piling another
//! GPU workload onto DirectML (HMR2 already owns the DML queue; adding
//! a second DML session would stack DXGI commands and TDR Intel Arc
//! the way DAv2 + RTMW3D used to — see [[dav2-directml-tdr]]).
//!
//! Model: `models/mediapipe_hand_landmark.onnx` from OpenCV-converted
//! MediaPipe `handpose_estimation_mediapipe_2023feb.onnx` (Apache 2.0).
//! - Input: `(1, 224, 224, 3)` NHWC, float32 in `[0, 1]`
//! - Outputs:
//!   * `Identity`: `(1, 63)` — 21 keypoints × `(x, y, z)` in crop pixel
//!     space (`x, y ∈ [0, 224]`, `z` relative depth)
//!   * `Identity_1`: `(1, 1)` — hand presence confidence
//!   * `Identity_2`: `(1, 1)` — handedness (0 = right, 1 = left, in model
//!     terms — the model output is mirror-sensitive so the caller must
//!     decide which hand is being passed)
//!   * `Identity_3`: `(1, 63)` — 21 keypoints in world coords (relative
//!     to wrist, meters)
//!
//! CPU inference: ~1.3 ms / hand on Python/ORT — same number is
//! expected from Rust/ort.

use std::path::Path;

use log::{error, info};
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;

use super::rtmw3d::{build_session, InferenceBackend};

/// MediaPipe Hand Landmarker — 21 3D keypoints per hand.
pub const NUM_HAND_KEYPOINTS: usize = 21;

/// Hand landmark layout (0..20). Useful for the finger-bone mapping in
/// `hmr2_provider`.
#[derive(Clone, Copy, Debug)]
pub enum HandKp {
    Wrist = 0,
    ThumbCmc = 1,
    ThumbMcp = 2,
    ThumbIp = 3,
    ThumbTip = 4,
    IndexMcp = 5,
    IndexPip = 6,
    IndexDip = 7,
    IndexTip = 8,
    MiddleMcp = 9,
    MiddlePip = 10,
    MiddleDip = 11,
    MiddleTip = 12,
    RingMcp = 13,
    RingPip = 14,
    RingDip = 15,
    RingTip = 16,
    PinkyMcp = 17,
    PinkyPip = 18,
    PinkyDip = 19,
    PinkyTip = 20,
}

/// Output of one hand inference.
///
/// - `screen`: pixel coords inside the 224×224 hand crop + relative
///   depth. `x, y ∈ [0, 224]`; `z` is monocular depth-like (in
///   roughly the same scale as the in-crop pixel dimensions). USE
///   THIS for finger bone positions in the body's source frame — the
///   pixel coords map cleanly to image space alongside the body.
/// - `world`: hand-canonical 3D landmarks (origin at wrist, axes
///   aligned with the hand's frame, NOT the camera frame). USE THIS
///   only for deriving palm orientation (forward + normal), where the
///   relative geometry inside the hand is what matters.
#[derive(Clone, Debug, Default)]
pub struct HandLandmarks {
    pub screen: [[f32; 3]; NUM_HAND_KEYPOINTS],
    pub world: [[f32; 3]; NUM_HAND_KEYPOINTS],
    pub presence: f32,
    pub handedness_score: f32,
}

pub struct HandLandmarker {
    session: Session,
    input_name: String,
    output_names: Vec<String>,
    #[allow(dead_code)]
    backend: InferenceBackend,
}

impl HandLandmarker {
    pub fn from_model_path(model_path: impl AsRef<Path>) -> Result<Self, String> {
        let path = model_path.as_ref();
        if !path.is_file() {
            return Err(format!(
                "MediaPipe hand landmarker not found at {}. Run dev.ps1 Install-MediaPipeHand or fetch from opencv/handpose_estimation_mediapipe.",
                path.display()
            ));
        }
        info!("Loading MediaPipe Hand Landmarker from {}", path.display());

        let path_str = path.to_string_lossy().into_owned();
        // Force CPU EP — the entire point is GPU isolation from HMR2's
        // DML session. `build_session` here defaults to CPU because
        // the hand model is tiny enough that DML setup overhead
        // dominates anyway.
        let (session, backend) = build_session(&path_str, 4, "MediaPipeHand-CPU")?;

        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input_1".to_string());

        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        if output_names.len() < 4 {
            return Err(format!(
                "MediaPipe hand model declared {} outputs, expected 4",
                output_names.len()
            ));
        }

        Ok(Self {
            session,
            input_name,
            output_names,
            backend,
        })
    }

    /// Run the landmark model on a 224×224 RGB crop. Returns landmarks
    /// in world space (relative to wrist) plus confidence/handedness.
    pub fn estimate(&mut self, rgb_224: &[u8]) -> Option<HandLandmarks> {
        if rgb_224.len() < 224 * 224 * 3 {
            return None;
        }
        // NHWC, float32 in [0, 1]. The MediaPipe TFLite default. No
        // ImageNet mean/std — the original model bundles those into
        // the input layer's batch-norm, post-conversion the ONNX
        // expects raw [0, 1].
        let mut tensor = Array4::<f32>::zeros((1, 224, 224, 3));
        for y in 0..224 {
            for x in 0..224 {
                let i = (y * 224 + x) * 3;
                tensor[(0, y, x, 0)] = rgb_224[i] as f32 / 255.0;
                tensor[(0, y, x, 1)] = rgb_224[i + 1] as f32 / 255.0;
                tensor[(0, y, x, 2)] = rgb_224[i + 2] as f32 / 255.0;
            }
        }

        let input = match TensorRef::from_array_view(&tensor) {
            Ok(v) => v,
            Err(e) => {
                error!("MediaPipeHand: TensorRef creation failed: {}", e);
                return None;
            }
        };
        let outputs = match self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
        {
            Ok(o) => o,
            Err(e) => {
                error!("MediaPipeHand: ONNX run failed: {}", e);
                return None;
            }
        };

        let extract = |name: &str, expected: usize| -> Option<Vec<f32>> {
            let v = outputs.get(name)?;
            let (_, data) = v.try_extract_tensor::<f32>().ok()?;
            if data.len() < expected {
                None
            } else {
                Some(data[..expected].to_vec())
            }
        };

        // Output naming after conversion: Identity = screen kp,
        // Identity_1 = presence, Identity_2 = handedness,
        // Identity_3 = world kp. Order is stable across the
        // opencv/handpose_estimation_mediapipe checkpoint family.
        let screen_flat = extract(&self.output_names[0], 63)?;
        let presence = extract(&self.output_names[1], 1).map(|v| v[0]).unwrap_or(0.0);
        let handedness_score =
            extract(&self.output_names[2], 1).map(|v| v[0]).unwrap_or(0.0);
        let world_flat = extract(&self.output_names[3], 63)?;
        drop(outputs);

        let mut screen = [[0.0_f32; 3]; NUM_HAND_KEYPOINTS];
        let mut world = [[0.0_f32; 3]; NUM_HAND_KEYPOINTS];
        for i in 0..NUM_HAND_KEYPOINTS {
            screen[i][0] = screen_flat[i * 3];
            screen[i][1] = screen_flat[i * 3 + 1];
            screen[i][2] = screen_flat[i * 3 + 2];
            world[i][0] = world_flat[i * 3];
            world[i][1] = world_flat[i * 3 + 1];
            world[i][2] = world_flat[i * 3 + 2];
        }

        Some(HandLandmarks {
            screen,
            world,
            presence,
            handedness_score,
        })
    }
}
