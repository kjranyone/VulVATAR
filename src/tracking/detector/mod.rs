//! The pose detector stage: Ultralytics YOLO26-pose (single-stage
//! whole-frame person + COCO-17) plus the shared perception pieces every
//! detector feeds — the 133-entry COCO-Wholebody keypoint record, the
//! MediaPipe face-cascade orchestration, and the DirectML/CPU session
//! builder.
//!
//! Replaced RTMW3D (SimCC wholebody, 2026-09): the fusion estimator's
//! contract (`DetectorAux` — 133 keypoints + FaceMesh landmarks + person
//! crop) is unchanged; YOLO26 fills the body-17 block and the estimator's
//! visibility calibration takes over the rest. See `yolo26.rs` for the
//! score/σ calibration notes and `docs/tracking-v2-design.md` for the
//! estimator side.

#[cfg(feature = "inference")]
pub(crate) mod annotation;
#[cfg(feature = "inference")]
pub(crate) mod decode;
#[cfg(feature = "inference")]
pub(crate) mod face;
#[cfg(feature = "inference")]
pub(crate) mod session;
#[cfg(feature = "inference")]
pub(crate) mod yolo26;

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

/// Raw perception outputs of one detector frame, for observation-level
/// consumers (the fusion estimator) that want the keypoints *before* any
/// skeleton building.
#[cfg(feature = "inference")]
#[derive(Clone, Debug, Default)]
pub(crate) struct DetectorAux {
    /// 133 COCO-Wholebody keypoints, whole-frame normalised `[0,1]`.
    /// YOLO26 fills the body 17; every other block carries score 0,
    /// which the estimator's gates treat as absent.
    pub joints: Vec<decode::DecodedJoint>,
    /// FaceMesh 478 landmarks in frame pixels (`z` in pixel scale) and
    /// the mesh confidence, when the face cascade ran this frame.
    pub face_mesh: Option<(Vec<[f32; 3]>, f32)>,
    /// Person box fed to the estimator `(x, y, w, h)` in frame pixels.
    pub crop: Option<(f32, f32, f32, f32)>,
}

/// Construction options for the pose detector. Lets the GUI's pipeline
/// settings (and the safe-mode degraded configuration) control the
/// execution provider.
#[derive(Clone, Copy, Debug)]
pub struct DetectorOptions {
    /// Execution provider for the FaceMesh + Blendshape cascade.
    /// Overridden to CPU when `force_cpu` is set.
    pub face_ep: crate::tracking::face_mediapipe::FaceMeshEp,
    /// Run the detector and FaceMesh on the CPU EP, keeping DirectML —
    /// and therefore the GPU driver's compute queue — completely out of
    /// the tracking pipeline. YOLO26-pose is fast enough on CPU (~34 ms)
    /// for this to be a usable degraded mode.
    pub force_cpu: bool,
}

impl Default for DetectorOptions {
    fn default() -> Self {
        Self {
            face_ep: crate::tracking::face_mediapipe::FaceMeshEp::Auto,
            force_cpu: false,
        }
    }
}

#[cfg(feature = "inference")]
pub use decode::DecodedJoint;
