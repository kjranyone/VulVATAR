//! HMR2 (4D-Humans) SMPL parametric pose inference.
//!
//! Single-image 3D body fitting that emits SMPL pose params — joint
//! rotation matrices for the 23 body joints plus the root (Pelvis). The
//! ONNX is exported FP16 with static batch=1 (see `hmr2_export/`); on
//! Intel Arc B570 + DirectML this runs at ~13 ms / frame (77 FPS).
//!
//! ## Why parametric over 2D / 2.5D
//!
//! ViTPose-L 2D keypoints + RTMW3D's coarse Z classifier both miss the
//! 3D body rotation needed for subtle torso twists and arms reaching
//! forward (deskcrop) — the 2D projection collapses anything not
//! orthogonal to the image plane. SMPL fitting reasons about the body
//! as a rigid kinematic chain under a perspective camera and outputs
//! per-joint axis-angle rotations that, by construction, encode the
//! 3D pose the photo depicts.
//!
//! ## Output shape
//!
//! - `global_orient`: (1, 1, 3, 3) — Pelvis world rotation
//! - `body_pose`:     (1, 23, 3, 3) — local rotations for joints 1..=23
//! - `betas`:         (1, 10) — SMPL shape coefficients (unused here)
//! - `pred_cam`:      (1, 3) — weak-perspective `(scale, tx, ty)`

#[cfg(feature = "inference")]
mod preprocess;

pub mod smpl;

#[cfg(feature = "inference")]
pub use preprocess::IMAGE_SIZE;

#[cfg(feature = "inference")]
use log::{error, info};
#[cfg(feature = "inference")]
use ort::session::Session;
#[cfg(feature = "inference")]
use ort::value::TensorRef;
#[cfg(feature = "inference")]
use std::path::Path;

#[cfg(feature = "inference")]
use super::rtmw3d::{build_session, InferenceBackend};

/// Decoded SMPL pose: rotation matrices for Pelvis + 23 body joints,
/// plus weak-perspective camera. Rotations are *local* to each joint's
/// parent in the SMPL kinematic tree (see `smpl::PARENTS`).
#[derive(Clone, Debug)]
pub struct Hmr2Output {
    /// Pelvis world rotation (3x3 matrix, row-major).
    pub global_orient: [[f32; 3]; 3],
    /// 23 SMPL body joint local rotations.
    pub body_pose: [[[f32; 3]; 3]; 23],
    /// SMPL shape parameters. Solver currently uses fixed rest joint
    /// positions so betas are advisory only.
    pub betas: [f32; 10],
    /// Weak-perspective camera: `(scale, translation_x, translation_y)`.
    /// Maps the SMPL mesh to the bbox crop frame; `pred_cam_t.z` in
    /// the 4D-Humans demo is derived from `scale` + focal length.
    pub pred_cam: [f32; 3],
}

#[cfg(feature = "inference")]
pub struct Hmr2Inference {
    session: Session,
    input_name: String,
    output_names: [String; 4],
    backend: InferenceBackend,
}

#[cfg(feature = "inference")]
impl Hmr2Inference {
    pub fn from_model_path(model_path: impl AsRef<Path>) -> Result<Self, String> {
        let path = model_path.as_ref();
        if !path.is_file() {
            return Err(format!(
                "HMR2 model not found at {}. Run hmr2_export/export_onnx.py + fp16.py to build it.",
                path.display()
            ));
        }
        info!("Loading HMR2 from {}", path.display());

        let path_str = path.to_string_lossy().into_owned();
        let (session, backend) = build_session(&path_str, 4, "HMR2")?;

        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "image".to_string());

        let outputs: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        if outputs.len() < 4 {
            return Err(format!(
                "HMR2 model declared {} outputs, expected 4 (global_orient/body_pose/betas/pred_cam)",
                outputs.len()
            ));
        }
        let output_names = [
            outputs[0].clone(),
            outputs[1].clone(),
            outputs[2].clone(),
            outputs[3].clone(),
        ];
        info!(
            "HMR2 I/O: input='{}', outputs={:?}",
            input_name, output_names
        );

        Ok(Self {
            session,
            input_name,
            output_names,
            backend,
        })
    }

    pub fn backend(&self) -> &InferenceBackend {
        &self.backend
    }

    /// Infer SMPL pose from a 256×256 RGB image (whole frame is fine
    /// for testing; production callers should YOLOX-crop first to put
    /// the subject roughly centered).
    pub fn estimate(&mut self, rgb: &[u8], width: u32, height: u32) -> Option<Hmr2Output> {
        let expected = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(3);
        if rgb.len() < expected || width == 0 || height == 0 {
            return None;
        }
        let tensor = preprocess::preprocess(rgb, width, height);
        let input = match TensorRef::from_array_view(&tensor) {
            Ok(v) => v,
            Err(e) => {
                error!("HMR2: TensorRef creation failed: {}", e);
                return None;
            }
        };
        let outputs = match self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
        {
            Ok(o) => o,
            Err(e) => {
                error!("HMR2: ONNX run failed: {}", e);
                return None;
            }
        };

        let extract = |name: &str, expected_len: usize| -> Option<Vec<f32>> {
            let v = outputs.get(name)?;
            let (_, data) = v.try_extract_tensor::<f32>().ok()?;
            if data.len() < expected_len {
                None
            } else {
                Some(data.to_vec())
            }
        };
        let go_data = extract(&self.output_names[0], 9)?;
        let bp_data = extract(&self.output_names[1], 23 * 9)?;
        let betas_data = extract(&self.output_names[2], 10)?;
        let cam_data = extract(&self.output_names[3], 3)?;
        drop(outputs);

        // Repack flat tensors into structured matrices. Row-major
        // layout matches the PyTorch tensor reshape((1, *, 3, 3)).
        let mut global_orient = [[0.0_f32; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                global_orient[r][c] = go_data[r * 3 + c];
            }
        }
        let mut body_pose = [[[0.0_f32; 3]; 3]; 23];
        for j in 0..23 {
            for r in 0..3 {
                for c in 0..3 {
                    body_pose[j][r][c] = bp_data[j * 9 + r * 3 + c];
                }
            }
        }
        let mut betas = [0.0_f32; 10];
        betas.copy_from_slice(&betas_data[..10]);
        let mut pred_cam = [0.0_f32; 3];
        pred_cam.copy_from_slice(&cam_data[..3]);

        Some(Hmr2Output {
            global_orient,
            body_pose,
            betas,
            pred_cam,
        })
    }
}
