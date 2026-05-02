//! MediaPipe FaceMeshV2 + BlendshapeV2 face landmark / ARKit
//! blendshape inference.
//!
//! VulVATAR's primary tracker is RTMW3D, which already emits 68 face
//! landmarks at indices 23..=90 of its 133-keypoint output. Those
//! landmarks are useful for *bbox derivation* (we use them to crop
//! the face for this module's input) and for head-pose euler
//! computation, but they do not by themselves deliver ARKit-style
//! blendshape coefficients (`mouthSmileLeft`, `eyeBlinkLeft`,
//! `jawOpen`, …) that drive a VRM's expression channels.
//!
//! That gap is filled here. The cascade is:
//!
//!   RTMW3D body kp 0..=4 (nose / eyes / ears) →
//!     face bbox (1.7× pad) →
//!       FaceMeshV2  (256×256 NCHW input → 478 dense landmarks)  →
//!         BlendshapeV2  (146-landmark subset → 52 ARKit weights) →
//!           SourceExpression list
//!
//! Both ONNX files are sourced from PINTO_model_zoo's Wasabi mirror
//! (Apache-2). They are tiny — 4.8 MB + 1.8 MB combined — so the
//! module is cheap to load alongside the 370 MB RTMW3D session.
//!
//! ## Why not SMIRK or a FLAME regressor?
//!
//! SMIRK (CVPR 2024) regresses FLAME parameters that we would then
//! retarget into ARKit's 52-blendshape space. The benefit is the
//! "smirk" direction in the expression manifold — asymmetric mouth
//! poses — but stylised VRM avatars rarely carry asymmetric
//! blendshape shapes anyway, so the gain is small and the cost
//! (PyTorch → ONNX self-export, FLAME → ARKit retarget matrix) is
//! significant. BlendshapeV2 maps directly into ARKit, the path is
//! load-tested, and the model is Apache-2.


// FaceBbox and SourceExpression are referenced by both the real
// (inference) and stub (no-inference) `estimate` impls, so they must
// be visible regardless of feature flag — otherwise the
// no-default-features build cannot resolve the stub's signature.
use super::source_skeleton::FacePose;
use super::SourceExpression;
#[cfg(feature = "inference")]
use log::{error, info, warn};
#[cfg(feature = "inference")]
use ndarray::{Array3, Array4};
#[cfg(feature = "inference")]
use ort::session::Session;
#[cfg(feature = "inference")]
use ort::value::TensorRef;
#[cfg(feature = "inference")]
use std::path::Path;

use super::rtmw3d::InferenceBackend;

/// Input dimensions for the FaceMeshV2 landmarker (PINTO export of
/// MediaPipe `face_landmarks_detector` — input is `(1, 3, 256, 256)`
/// NCHW).
#[cfg(feature = "inference")]
const FACE_INPUT_SIZE: u32 = 256;

#[cfg(feature = "inference")]
const FACE_LANDMARK_COUNT: usize = 478;

#[cfg(feature = "inference")]
const FACE_BLENDSHAPE_INPUT_COUNT: usize = 146;

#[cfg(feature = "inference")]
const FACE_BLENDSHAPE_COUNT: usize = 52;

/// Padding multiplier when deriving the face bbox from RTMW3D's
/// face-68 landmarks (indices 23..=90). The 68-point set spans
/// jawline / brows / mouth so the centroid lands on the face centre
/// and `max_d` already reaches close to the face edge — we only pad
/// a touch to ensure the FaceMesh model gets a small margin around
/// the face contour.
#[cfg(feature = "inference")]
const FACE_BBOX_PAD: f32 = 1.3;

// (Previously a hardcoded `FACE_CONFIDENCE_THRESHOLD = 0.4` lived here
// and silently dropped any blendshape result the FaceMesh model was
// less than 40 % confident about. That hid the GUI-exposed
// `face_confidence_threshold` slider behind a stricter floor the user
// could not lower — exactly the same dual-filter pattern that broke
// upper-body framing on the body track. The gate is gone; `estimate`
// surfaces the raw post-sigmoid confidence to the caller, who feeds it
// into the solver alongside the body-derived face confidence so the
// GUI slider is the single source of truth.)

/// Indices into FaceMeshV2's 478 landmarks that BlendshapeV2 expects
/// as its 146-point input. Sourced verbatim from MediaPipe's
/// `face_blendshapes_graph.cc` (`kLandmarksSubsetIdxs`).
#[cfg(feature = "inference")]
const FACE_BLENDSHAPE_LANDMARK_SUBSET: [u16; FACE_BLENDSHAPE_INPUT_COUNT] = [
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65,
    66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136,
    144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172,
    173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283,
    284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324,
    332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385,
    386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466, 468, 469, 470, 471, 472,
    473, 474, 475, 476, 477,
];

/// ARKit-compatible blendshape names in BlendshapeV2's output order.
/// Index 0 (`_neutral`) is a baseline scalar the model emits but that
/// no avatar consumes directly — we keep it in the array so positional
/// indexing matches the model and the array length is a clean 52.
#[cfg(feature = "inference")]
const FACE_BLENDSHAPE_NAMES: [&str; FACE_BLENDSHAPE_COUNT] = [
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
];

/// Pixel-space face bbox (image-relative).
///
/// Visible regardless of the `inference` feature flag so the stub
/// `estimate()` impl below can refer to it without a build break in
/// the `--no-default-features` configuration.
#[derive(Clone, Copy, Debug)]
pub struct FaceBbox {
    pub x: f32,
    pub y: f32,
    pub size: f32,
}

/// Which execution provider to use for FaceMesh + Blendshape.
///
/// `Auto` is right when nothing else heavy uses DirectML (standalone
/// `rtmw3d` provider): GPU runs the small face cascade in ~3 ms.
/// `ForceCpu` is right when DAv2 / MoGe is colocated on DirectML
/// (depth-aware providers): ~7 ms on CPU avoids the 13 ms-tier
/// inflation we observed under DirectML EP queue contention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceMeshEp {
    Auto,
    ForceCpu,
}

pub struct FaceMeshInference {
    #[cfg(feature = "inference")]
    face_session: Session,
    #[cfg(feature = "inference")]
    face_input_name: String,
    #[cfg(feature = "inference")]
    face_output_names: Vec<String>,
    #[cfg(feature = "inference")]
    blendshape_session: Session,
    #[cfg(feature = "inference")]
    blendshape_input_name: String,
    #[cfg(feature = "inference")]
    blendshape_output_names: Vec<String>,
    #[cfg(feature = "inference")]
    backend: InferenceBackend,
}

impl FaceMeshInference {
    /// Try to load both ONNX files. Returns `Ok(None)` if either is
    /// missing — face inference is optional, the avatar still tracks
    /// body and hands without it. Returns `Err` only on a load
    /// failure that the caller should surface to the user.
    ///
    /// `face_ep` selects DirectML (Auto) vs CPU. See [`FaceMeshEp`].
    #[cfg(feature = "inference")]
    pub fn try_from_models_dir(
        models_dir: impl AsRef<Path>,
        face_ep: FaceMeshEp,
    ) -> Result<Option<Self>, String> {
        let models_dir = models_dir.as_ref();
        let face_path = models_dir.join("face_landmark.onnx");
        let blend_path = models_dir.join("face_blendshapes.onnx");
        if !face_path.is_file() || !blend_path.is_file() {
            warn!(
                "Face inference disabled — missing {} or {}. \
                 Run dev.ps1 setup to fetch them.",
                face_path.display(),
                blend_path.display()
            );
            return Ok(None);
        }

        let ep_label = match face_ep {
            FaceMeshEp::Auto => "Auto",
            FaceMeshEp::ForceCpu => "CPU EP",
        };
        info!("Loading FaceMeshV2 from {} ({})", face_path.display(), ep_label);
        let (face_session, backend) = match face_ep {
            FaceMeshEp::Auto => {
                super::rtmw3d::build_session(&face_path.to_string_lossy(), 2, "FaceMeshV2")?
            }
            FaceMeshEp::ForceCpu => super::rtmw3d::build_session_cpu_only(
                &face_path.to_string_lossy(),
                2,
                "FaceMeshV2",
            )?,
        };
        let face_input_name = face_session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input".to_string());
        let face_output_names: Vec<String> = face_session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        info!(
            "FaceMeshV2 I/O: input='{}', outputs={:?}",
            face_input_name, face_output_names
        );

        info!(
            "Loading BlendshapeV2 from {} ({})",
            blend_path.display(),
            ep_label
        );
        let (blendshape_session, _) = match face_ep {
            FaceMeshEp::Auto => {
                super::rtmw3d::build_session(&blend_path.to_string_lossy(), 2, "BlendshapeV2")?
            }
            FaceMeshEp::ForceCpu => super::rtmw3d::build_session_cpu_only(
                &blend_path.to_string_lossy(),
                2,
                "BlendshapeV2",
            )?,
        };
        let blendshape_input_name = blendshape_session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input".to_string());
        let blendshape_output_names: Vec<String> = blendshape_session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        info!(
            "BlendshapeV2 I/O: input='{}', outputs={:?}",
            blendshape_input_name, blendshape_output_names
        );

        Ok(Some(Self {
            face_session,
            face_input_name,
            face_output_names,
            blendshape_session,
            blendshape_input_name,
            blendshape_output_names,
            backend,
        }))
    }

    #[cfg(not(feature = "inference"))]
    pub fn try_from_models_dir(
        _: impl AsRef<std::path::Path>,
        _: FaceMeshEp,
    ) -> Result<Option<Self>, String> {
        Ok(None)
    }

    /// Run FaceMesh + Blendshape on `rgb_data` cropped to `bbox`.
    /// Returns `None` on low confidence or any inference error so the
    /// caller can leave the avatar's expression channel empty.
    /// Run FaceMeshV2 + BlendshapeV2 on the face crop. Returns the
    /// 52-blendshape mapping plus the FaceMesh model's own
    /// "is this even a face" confidence (post-sigmoid, in `[0, 1]`).
    /// Returns `None` only on actual ONNX errors — confidence-based
    /// filtering is the caller's responsibility.
    #[cfg(feature = "inference")]
    pub fn estimate(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        bbox: &FaceBbox,
    ) -> Option<(Vec<SourceExpression>, f32, Option<FacePose>)> {
        let face_tensor = crop_face_to_tensor(rgb_data, width, height, bbox);
        let face_input = match TensorRef::from_array_view(&face_tensor) {
            Ok(v) => v,
            Err(e) => {
                error!("FaceMesh: TensorRef creation failed: {}", e);
                return None;
            }
        };
        let landmarks_name = self.face_output_names.first().cloned();
        let conf_name = self.face_output_names.get(1).cloned();

        let face_outputs = match self
            .face_session
            .run(ort::inputs![self.face_input_name.as_str() => face_input])
        {
            Ok(out) => out,
            Err(e) => {
                error!("FaceMesh: ONNX run failed: {}", e);
                return None;
            }
        };
        let conf_value = conf_name
            .as_deref()
            .and_then(|n| face_outputs.get(n))
            .and_then(|v| v.try_extract_tensor::<f32>().ok())
            .and_then(|(_, data)| data.first().copied());
        let face_conf = conf_value.map(sigmoid).unwrap_or(0.0);
        let landmarks_data: Vec<f32> = landmarks_name
            .as_deref()
            .and_then(|n| face_outputs.get(n))
            .and_then(|v| v.try_extract_tensor::<f32>().ok())
            .map(|(_, data)| data.to_vec())?;
        // Drop the face outputs (and its session borrow) before we
        // run the blendshape session.
        drop(face_outputs);
        let landmarks = decode_face_landmarks(&landmarks_data);

        // Run BlendshapeV2 on the 146-landmark subset.
        let bs_input = prepare_blendshape_input(&landmarks);
        let bs_input_view = match TensorRef::from_array_view(&bs_input) {
            Ok(v) => v,
            Err(e) => {
                error!("Blendshape: TensorRef creation failed: {}", e);
                return None;
            }
        };
        let bs_output_name = self.blendshape_output_names.first().cloned();
        let bs_outputs = match self
            .blendshape_session
            .run(ort::inputs![self.blendshape_input_name.as_str() => bs_input_view])
        {
            Ok(out) => out,
            Err(e) => {
                error!("Blendshape: ONNX run failed: {}", e);
                return None;
            }
        };
        let bs_data: Vec<f32> = bs_output_name
            .as_deref()
            .and_then(|n| bs_outputs.get(n))
            .and_then(|v| v.try_extract_tensor::<f32>().ok())
            .map(|(_, data)| data.to_vec())?;
        drop(bs_outputs);
        let weights = decode_blendshapes(&bs_data);
        // Head pose from the dense 478 landmarks. Much higher fidelity
        // than the 5-landmark body-derived pose: FaceMesh runs on a
        // dedicated 256×256 face crop and emits depth at each landmark
        // anchored on a learnt canonical face mesh, so the cheek and
        // ear-line baselines have real Z separation even at 3/4 turns
        // (the dead-zone where the body-derived pose's `ear_dz`
        // collapses to noise). `face_conf` already gates this — the
        // caller folds it into the FacePose's confidence.
        let face_pose = derive_face_pose_from_landmarks(&landmarks);
        Some((map_blendshapes_to_expressions(&weights), face_conf, face_pose))
    }

    #[cfg(not(feature = "inference"))]
    pub fn estimate(
        &mut self,
        _: &[u8],
        _: u32,
        _: u32,
        _: &FaceBbox,
    ) -> Option<(Vec<SourceExpression>, f32, Option<FacePose>)> {
        None
    }

    pub fn backend(&self) -> &InferenceBackend {
        #[cfg(feature = "inference")]
        return &self.backend;
        #[cfg(not(feature = "inference"))]
        unreachable!()
    }
}

/// Derive a face bbox from a list of pixel-space face-region points
/// (typically the five RTMW3D body keypoints 0..=4: nose, left/right
/// eye, left/right ear). Returns `None` if too few points were
/// available or the bbox falls outside the image.
#[cfg(feature = "inference")]
pub fn derive_face_bbox(
    points_px: &[(f32, f32)],
    source_w: u32,
    source_h: u32,
) -> Option<FaceBbox> {
    if points_px.len() < 4 {
        return None;
    }
    let cx = points_px.iter().map(|p| p.0).sum::<f32>() / points_px.len() as f32;
    let cy = points_px.iter().map(|p| p.1).sum::<f32>() / points_px.len() as f32;
    let max_d = points_px
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

#[cfg(feature = "inference")]
fn crop_face_to_tensor(rgb: &[u8], width: u32, height: u32, bbox: &FaceBbox) -> Array4<f32> {
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

/// Derive head pose (yaw / pitch / roll) from the dense 478 FaceMesh
/// landmarks.
///
/// FaceMesh runs on a dedicated 256×256 face crop and emits depth at
/// each landmark anchored on a learnt canonical face mesh, so the
/// cheek-to-cheek baseline has real Z separation across the full 0–
/// 360° head turn — including the 22.5° / 45° / 67.5° "3/4 view"
/// dead-zone where the body-derived `derive_face_pose_from_body`
/// collapses (its 5-landmark ear/eye line has near-uniform nz at
/// those poses, so `ear_dz.atan2(ear_dx)` returns near zero).
///
/// MediaPipe FaceMesh is view-invariant: indices reference fixed
/// anatomical positions on the canonical face regardless of head
/// orientation or input mirroring. Coordinates are in 256-pixel face-
/// crop space with z in pixel-equivalent depth units (smaller z =
/// closer to camera, per MediaPipe's documented convention).
///
/// Sign conventions match the source-skeleton frame
/// (see `source_skeleton::FacePose`): `+yaw` turns the head toward
/// camera-space `+X` (subject's anatomical right after the standard
/// `to_source` flip), `+pitch` drops the chin (looking down),
/// `+roll` leans the head toward `+X`.
#[cfg(feature = "inference")]
pub(super) fn derive_face_pose_from_landmarks(landmarks: &[[f32; 3]]) -> Option<FacePose> {
    if landmarks.len() < 478 {
        return None;
    }

    // MediaPipe FaceMesh canonical landmark indices.
    const NOSE_TIP: usize = 1;
    const RIGHT_EYE_OUTER: usize = 33; // subject's anatomical right
    const LEFT_EYE_OUTER: usize = 263; // subject's anatomical left
    const RIGHT_CHEEK: usize = 234;
    const LEFT_CHEEK: usize = 454;

    let nose = landmarks[NOSE_TIP];
    let r_eye = landmarks[RIGHT_EYE_OUTER];
    let l_eye = landmarks[LEFT_EYE_OUTER];
    let r_cheek = landmarks[RIGHT_CHEEK];
    let l_cheek = landmarks[LEFT_CHEEK];

    // YAW from 2D nose offset relative to cheek midpoint.
    //
    // The PINTO export's z signal turns out to be ~4× weaker than
    // x/y in scale on this network — a measured 45° head turn in
    // the validation set yields only ~13° via the clean
    // `cheek_dz.atan2(cheek_dx)` formula. So z-based yaw, while
    // sign-correct and noise-free, is too dampened to be useful.
    //
    // Instead exploit the rigid-perspective observation that as
    // the head rotates around its vertical axis the nose tip
    // sweeps left/right relative to the cheek midpoint:
    //
    //   yaw_signal = (nose.x − cheek_midpoint.x) / face_half_width
    //
    // Source convention: `+yaw = head toward +X` (subject's anat
    // right, after the standard `to_source` flip on non-mirrored
    // camera input). At rest the right cheek (idx 234) sits at
    // LOW image X and the left cheek (idx 454) at HIGH image X,
    // so when the subject turns toward their anat right the nose
    // sweeps toward LOW image X (`nose.x − center < 0`). We negate
    // the signal so this case yields positive source yaw.
    let cheek_dx = l_cheek[0] - r_cheek[0];
    if cheek_dx.abs() < 5.0 {
        return None;
    }
    let face_half_width = (cheek_dx.abs() * 0.5).max(10.0);
    let cheek_mid_x = (r_cheek[0] + l_cheek[0]) * 0.5;
    let nose_offset_x = nose[0] - cheek_mid_x;
    // Clamp at ±2.0 (atan ≈ 63°) rather than ±3.0 (atan ≈ 72°): the
    // larger range only mattered for back/3/4 views where face
    // confidence collapses to ~0 anyway, and 72° saturation produced
    // visually jarring overshoot when the FaceMesh confidence briefly
    // spiked during head transitions through profile.
    let yaw_signal = nose_offset_x / face_half_width;
    let yaw = -yaw_signal.clamp(-2.0, 2.0).atan();

    // PITCH from nose vs eye-mid Y. Image y-down: when subject looks
    // down, the nose drops below the eye line (`y[nose] - eye_mid_y
    // > 0`). Source `+pitch` = chin drops, so the sign matches
    // directly without a flip.
    //
    // Anatomical baseline subtraction: at a frontal neutral pose the
    // nose tip naturally sits below the eye line by ~50% of inter-eye
    // distance, which under the raw formula maps to a constant
    // `+25–27°` pitch reading. Measured across the validation set:
    //   gothic 0°/shoulders/wrists/torso (8 neutral frames): mean ≈ 26°
    //   tan(26°) ≈ 0.49
    // Subtracting that constant from the *signal* (not the angle)
    // preserves linearity through the atan and yields ≈ 0° at neutral
    // and ≈ −20° for the head_pitch_up_020 test frame.
    const PITCH_NEUTRAL_SIGNAL: f32 = 0.49;
    let eye_mid_y = (r_eye[1] + l_eye[1]) * 0.5;
    let face_width = ((l_eye[0] - r_eye[0]).powi(2) + (l_eye[1] - r_eye[1]).powi(2))
        .sqrt()
        .max(20.0);
    let pitch_signal = (nose[1] - eye_mid_y) / face_width - PITCH_NEUTRAL_SIGNAL;
    let pitch = pitch_signal.clamp(-2.0, 2.0).atan();

    // ROLL from eye-line Y-X. Image y-down: when subject tilts head
    // to their anatomical right, the right eye (idx 33, low image X)
    // drops in image (`y[33] > y[263]`). Source `+roll` = head leans
    // toward +X (anat right), so we want positive in this case;
    // `atan2(y[33] − y[263], x[263] − x[33])` does this directly
    // (both numerator and denominator positive).
    let roll = (r_eye[1] - l_eye[1]).atan2(l_eye[0] - r_eye[0]);

    Some(FacePose {
        yaw,
        pitch,
        roll,
        // Confidence overwritten by caller with `face_mesh_confidence`
        // (the FaceMesh model's "is this a face" sigmoid). The pose
        // itself doesn't carry an independent quality signal.
        confidence: 1.0,
    })
}

/// Decode the FaceMeshV2 landmarks output into 478 × `[x, y, z]`.
/// Coordinates are in 256-pixel space; z is in pixel-equivalent depth
/// units relative to the face's geometric centre.
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
fn prepare_blendshape_input(landmarks: &[[f32; 3]]) -> Array3<f32> {
    let mut tensor = Array3::<f32>::zeros((1, FACE_BLENDSHAPE_INPUT_COUNT, 2));
    let scale = 1.0 / FACE_INPUT_SIZE as f32;
    for (out_i, &mp_idx) in FACE_BLENDSHAPE_LANDMARK_SUBSET.iter().enumerate() {
        if let Some(p) = landmarks.get(mp_idx as usize) {
            tensor[(0, out_i, 0)] = p[0] * scale;
            tensor[(0, out_i, 1)] = p[1] * scale;
        }
    }
    tensor
}

#[cfg(feature = "inference")]
fn decode_blendshapes(data: &[f32]) -> [f32; FACE_BLENDSHAPE_COUNT] {
    let mut out = [0.0_f32; FACE_BLENDSHAPE_COUNT];
    let n = data.len().min(FACE_BLENDSHAPE_COUNT);
    out[..n].copy_from_slice(&data[..n]);
    out
}

#[cfg(feature = "inference")]
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Map the 52 ARKit-style blendshape weights to a list of
/// `SourceExpression` entries. Emits both verbatim ARKit names (so
/// VRM 1.0 rigs that define the matching custom expressions get the
/// signal directly) and VRM 1.0 preset aggregations (`blinkLeft`,
/// `blinkRight`, `aa`, `happy`, `sad`, `surprised`) so a stock rig
/// with no ARKit-named expressions still shows live facial motion.
/// Selfie mirror is honoured: the subject's left eye drives
/// `blinkRight`.
#[cfg(feature = "inference")]
fn map_blendshapes_to_expressions(weights: &[f32; FACE_BLENDSHAPE_COUNT]) -> Vec<SourceExpression> {
    let mut out: Vec<SourceExpression> = Vec::with_capacity(FACE_BLENDSHAPE_COUNT + 8);
    for (i, name) in FACE_BLENDSHAPE_NAMES.iter().enumerate().skip(1) {
        out.push(SourceExpression {
            name: (*name).to_string(),
            weight: weights[i].clamp(0.0, 1.0),
        });
    }
    let blink_left_subj = weights[9].clamp(0.0, 1.0); // eyeBlinkLeft (subject)
    let blink_right_subj = weights[10].clamp(0.0, 1.0);
    out.push(SourceExpression {
        name: "blinkRight".into(),
        weight: blink_left_subj,
    });
    out.push(SourceExpression {
        name: "blinkLeft".into(),
        weight: blink_right_subj,
    });
    out.push(SourceExpression {
        name: "blink".into(),
        weight: ((blink_left_subj + blink_right_subj) * 0.5).clamp(0.0, 1.0),
    });
    let jaw_open = weights[25].clamp(0.0, 1.0);
    out.push(SourceExpression {
        name: "aa".into(),
        weight: jaw_open,
    });
    let smile = ((weights[44] + weights[45]) * 0.5).clamp(0.0, 1.0);
    out.push(SourceExpression {
        name: "happy".into(),
        weight: smile,
    });
    let frown = ((weights[30] + weights[31]) * 0.5).clamp(0.0, 1.0);
    out.push(SourceExpression {
        name: "sad".into(),
        weight: frown,
    });
    let brow_up = weights[3].clamp(0.0, 1.0);
    let surprise = (brow_up * jaw_open).sqrt();
    out.push(SourceExpression {
        name: "surprised".into(),
        weight: surprise,
    });
    out
}
