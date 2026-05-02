//! RTMW3D — single-model whole-body 3D pose inference.
//!
//! Successor to `tracking::mediapipe`. RTMW3D-x (~370 MB) emits 133
//! COCO-Wholebody keypoints in 3D from a single forward pass:
//! 17 body, 6 foot, 68 face, plus 21 per hand. All keypoints share
//! the same coordinate system, so wrist→finger continuity is built-in
//! without per-hand crops or coordinate-frame fiddling.
//!
//! ## Pipeline
//!
//! 1. **YOLOX person crop** (optional). Adapts the source frame to a
//!    bbox around the largest detected person, padded toward RTMW3D's
//!    288:384 input aspect. Falls through to whole-frame when YOLOX is
//!    absent or no person clears its threshold.
//! 2. **Resize / normalise** to 288×384 NCHW with ImageNet stats —
//!    mirrors rtmlib's reference preprocess.
//! 3. **Infer** the model: one input, three SimCC heatmap outputs
//!    (X: `(1, 133, 576)`, Y: `(1, 133, 768)`, Z: `(1, 133, 576)`).
//! 4. **Decode** by argmax + sigmoid into normalised `(nx, ny, nz)`
//!    triples in `[0, 1]³` with per-joint scores.
//! 5. **Map** the 133 indices to `HumanoidBone` slots (selfie mirror)
//!    and build a `SourceSkeleton` whose hip mid-point is the depth
//!    origin, in source-space `[-aspect, +aspect] × [-1, 1] × z` coords
//!    with `+z` toward the camera.
//! 6. **Wrist resilience layer** validates the per-hand wrist position
//!    against the upper-arm chain and synthesises a held / extended
//!    fallback when the live track is implausible (see `wrist`).
//! 7. **Face cascade**: head pose from body face landmarks 0..=4, then
//!    optional MediaPipe FaceMesh + Blendshape on a face-68-derived
//!    bbox for expression channels.
//!
//! ## Module layout
//!
//! Each pipeline stage lives in its own sibling module so this file
//! only orchestrates. The sibling modules (`consts`, `math`, `decode`,
//! `preprocess`, `skeleton`, `wrist`, `face`, `annotation`, `session`)
//! are private to this directory; `session::build_session` is re-exported
//! as `pub(in crate::tracking)` for the YOLOX + FaceMesh sessions to
//! share the DirectML-fall-back-to-CPU EP-selection logic.

#[cfg(feature = "inference")]
mod annotation;
#[cfg(feature = "inference")]
mod consts;
#[cfg(feature = "inference")]
mod decode;
#[cfg(feature = "inference")]
mod face;
#[cfg(feature = "inference")]
mod math;
#[cfg(feature = "inference")]
mod preprocess;
#[cfg(feature = "inference")]
mod session;
#[cfg(feature = "inference")]
mod skeleton;
#[cfg(feature = "inference")]
mod wrist;
#[cfg(feature = "inference")]
mod yolox_worker;

#[cfg(feature = "inference")]
pub(in crate::tracking) use session::{build_session, build_session_cpu_only};
// Crop helpers shared with the CIGPose+MoGe-2 provider's optional
// YOLOX person-crop pre-stage. The functions don't depend on RTMW3D
// internals; they're here purely because that's where they were
// originally written. Promotion-only re-export, no logic change.
#[cfg(feature = "inference")]
pub(in crate::tracking) use preprocess::{crop_rgb, pad_and_clamp_bbox};

#[cfg(feature = "inference")]
use super::face_mediapipe::FaceMeshInference;
#[cfg(feature = "inference")]
use super::yolox::YoloxPersonDetector;
#[cfg(feature = "inference")]
use yolox_worker::YoloxWorker;
use super::PoseEstimate;
#[cfg(feature = "inference")]
use super::{DetectionAnnotation, SourceSkeleton};
#[cfg(feature = "inference")]
use crate::asset::HumanoidBone;
#[cfg(feature = "inference")]
use log::{debug, error, info, warn};
#[cfg(feature = "inference")]
use ort::session::Session;
#[cfg(feature = "inference")]
use ort::value::TensorRef;
#[cfg(feature = "inference")]
use std::path::Path;

#[cfg(feature = "inference")]
use decode::{NUM_JOINTS, SIMCC_X_BINS, SIMCC_Y_BINS, SIMCC_Z_BINS};

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

/// Submit a frame to the YOLOX worker on every Nth call. Detection
/// itself runs asynchronously in a background thread, so the main
/// pipeline never waits for it past the cold-start frame — we just
/// read the worker's sticky outbox each call and use whatever bbox
/// it last produced (typically 1–2 frames stale).
///
/// At 30 fps and N=4, YOLOX submits at 7.5 fps. The 25% downstream
/// pad on the bbox absorbs the few-pixel subject motion that
/// accumulates between submits.
#[cfg(feature = "inference")]
const YOLOX_REFRESH_PERIOD: u64 = 4;

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
    /// Optional YOLOX-m human-art person detector running on a
    /// background thread. When present the camera frame is cropped
    /// to the largest detected person before being fed to RTMW3D —
    /// the principled fix for small-subject / off-centre /
    /// partial-occlusion cases. Absent → whole-frame inference
    /// (Kinemotion-style). The worker pattern (mpsc inbox + sticky
    /// `Arc<DetectResult>` outbox) means the main pipeline never
    /// blocks on detection past the cold-start frame, hiding YOLOX's
    /// ~22 ms cost behind the rest of the pipeline.
    #[cfg(feature = "inference")]
    yolox_worker: Option<YoloxWorker>,
    #[cfg(feature = "inference")]
    left_wrist: wrist::WristTracker,
    #[cfg(feature = "inference")]
    right_wrist: wrist::WristTracker,
    #[cfg(feature = "inference")]
    load_warnings: Vec<String>,
    #[cfg(feature = "inference")]
    backend: InferenceBackend,
}

impl Rtmw3dInference {
    /// Instantiate from `models_dir` (typically `models/`). Looks for
    /// `rtmw3d.onnx` and tries to register the DirectML EP, falling
    /// back to CPU on EP failure. FaceMesh + Blendshape go on
    /// DirectML (Auto) — best when no other heavy GPU model is
    /// active. Depth-aware providers should use
    /// [`Self::from_models_dir_with_face_ep`] with `ForceCpu` to
    /// avoid GPU contention.
    #[cfg(feature = "inference")]
    pub fn from_models_dir(models_dir: impl AsRef<Path>) -> Result<Self, String> {
        Self::from_models_dir_with_face_ep(
            models_dir,
            super::face_mediapipe::FaceMeshEp::Auto,
        )
    }

    /// Same as [`Self::from_models_dir`] but with explicit FaceMesh
    /// EP. Used by `rtmw3d-with-depth` to force FaceMesh onto CPU,
    /// where it runs uncontended at ~7 ms instead of 13+ ms when
    /// fighting DAv2 for the DirectML command queue.
    #[cfg(feature = "inference")]
    pub fn from_models_dir_with_face_ep(
        models_dir: impl AsRef<Path>,
        face_ep: super::face_mediapipe::FaceMeshEp,
    ) -> Result<Self, String> {
        let models_dir = models_dir.as_ref();
        let model_path = models_dir.join("rtmw3d.onnx");
        if !model_path.is_file() {
            return Err(format!(
                "RTMW3D model not found at {}. Run dev.ps1 setup to fetch it.",
                model_path.display()
            ));
        }

        info!("Loading RTMW3D from {}", model_path.display());
        let (session, backend) = build_session(&model_path.to_string_lossy(), 4, "RTMW3D")?;

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
        let face_mesh = match FaceMeshInference::try_from_models_dir(models_dir, face_ep) {
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

        let person_detector = match YoloxPersonDetector::try_from_models_dir(models_dir) {
            Ok(Some(d)) => {
                info!(
                    "YOLOX person detector loaded ({}) — running on background thread",
                    d.backend().label()
                );
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
        let yolox_worker = match person_detector {
            Some(d) => Some(YoloxWorker::spawn(d)?),
            None => None,
        };

        Ok(Self {
            session,
            input_name,
            output_names,
            face_mesh,
            yolox_worker,
            left_wrist: wrist::WristTracker::default(),
            right_wrist: wrist::WristTracker::default(),
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

        let t_total = std::time::Instant::now();

        // YOLOX-nano person crop. When a person is detected we crop the
        // frame around them (with padding) before feeding it to RTMW3D
        // — this is the principled fix for small-subject / off-centre /
        // partial-occlusion inputs. Joints come back in *crop* nx/ny
        // and are remapped to original-frame coords below so the rest
        // of the pipeline (face bbox, source skeleton, solver) sees a
        // consistent coordinate system regardless of crop state.
        //
        // Falls through to whole-frame inference when no person is
        // detected or YOLOX is unavailable.
        let t_yolox = std::time::Instant::now();
        let (infer_rgb_owned, infer_rgb_slice, infer_w, infer_h, crop_origin, annotation_bbox) = {
            // Submit a detect request to the YOLOX worker on every
            // `YOLOX_REFRESH_PERIOD`-th frame (or always on cold
            // start so the very first call doesn't deadlock waiting
            // on `wait_latest`). The worker drains older queued
            // requests so it always processes the freshest submitted
            // frame. Either way, we then read the latest sticky
            // result — typically from 1–2 frames ago, which is well
            // within the 25% downstream pad.
            let bbox_opt = if let Some(worker) = self.yolox_worker.as_ref() {
                let cold_start = !worker.has_result();
                if cold_start || frame_index % YOLOX_REFRESH_PERIOD == 0 {
                    worker.submit(rgb_data, width, height);
                }
                worker.wait_latest().bbox
            } else {
                None
            };
            match bbox_opt {
                Some(bbox) => {
                    // 25% pad: YOLOX bboxes are tight to the visible
                    // body silhouette, but RTMW3D needs slack for
                    // outstretched arms and cross-step legs that may
                    // briefly extend past the bbox between frames.
                    // Tested at 10% — produced regressions on walking
                    // / cross-step poses where the leading limb fell
                    // outside the crop.
                    let (cx1, cy1, cx2, cy2) =
                        preprocess::pad_and_clamp_bbox(&bbox, width, height, 0.25);
                    let cw = cx2 - cx1;
                    let ch = cy2 - cy1;
                    debug!(
                        "RTMW3D: YOLOX bbox=[{:.0},{:.0},{:.0},{:.0}] score={:.2} → crop {}x{} (orig {}x{})",
                        bbox.x1, bbox.y1, bbox.x2, bbox.y2, bbox.score, cw, ch, width, height
                    );
                    if cw < 32 || ch < 32 {
                        // Bbox too small to crop usefully — fall through.
                        (None, rgb_data, width, height, None, None)
                    } else {
                        let crop = preprocess::crop_rgb(rgb_data, width, height, cx1, cy1, cw, ch);
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
        let infer_rgb: &[u8] = match &infer_rgb_owned {
            Some(buf) => buf.as_slice(),
            None => infer_rgb_slice,
        };
        let dt_yolox = t_yolox.elapsed();

        // Preprocess to 288×384 NCHW with ImageNet normalization.
        let t_pre = std::time::Instant::now();
        let tensor = preprocess::preprocess(infer_rgb, infer_w, infer_h);
        let dt_pre = t_pre.elapsed();

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

        let t_run = std::time::Instant::now();
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
        let dt_run = t_run.elapsed();

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

        let t_decode = std::time::Instant::now();
        let mut joints = decode::decode_simcc(&simcc_x, &simcc_y, &simcc_z);
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
        let mut skeleton = skeleton::build_source_skeleton(frame_index, &joints, width, height);

        // Per-side wrist sanity check + temporal hold. Run before
        // the face cascade so the wrist is settled when the solver
        // reads it.
        wrist::apply_wrist_temporal_hold(
            &mut skeleton,
            HumanoidBone::RightHand,
            HumanoidBone::RightShoulder,
            HumanoidBone::RightLowerArm,
            &mut self.right_wrist,
        );
        wrist::apply_wrist_temporal_hold(
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
        skeleton.face = face::derive_face_pose_from_body(&skeleton, &joints);
        let dt_decode = t_decode.elapsed();

        // Face crop → MediaPipe FaceMesh → BlendshapeV2. The bbox is
        // derived from RTMW3D's face-68 landmarks (indices 23..=90)
        // so the cascade does not need its own face detector. The face
        // mesh's own confidence is folded into `skeleton.face.confidence`
        // by taking the max of the body-derived value (5 face landmarks)
        // and the mesh-model value — they're independent "is the face
        // visible" signals so the higher one is the better evidence.
        // The solver's `face_confidence_threshold` then gates head
        // rotation *and* expressions against this single number.
        let t_face = std::time::Instant::now();
        if let Some(face_mesh) = self.face_mesh.as_mut() {
            if let Some(bbox) = face::build_face_bbox_from_joints(&joints, width, height) {
                if let Some((exprs, mesh_conf, mesh_face_pose)) =
                    face_mesh.estimate(rgb_data, width, height, &bbox)
                {
                    skeleton.expressions = exprs;
                    skeleton.face_mesh_confidence = Some(mesh_conf);
                    // Prefer FaceMesh's dense-landmark pose when
                    // available — it has real Z separation across
                    // the full 360° head turn including the 22.5°–
                    // 67.5° "3/4 view" dead-zone where the body-
                    // derived ear-line pose collapses to ~0 yaw
                    // (the 5 RTMW3D face landmarks lose nz contrast
                    // at those angles). Fall back to the body-derived
                    // pose otherwise. Stamp the FaceMesh model's
                    // confidence onto the result so the solver can
                    // still gate against it.
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

        let dt_total = t_total.elapsed();
        debug!(
            "RTMW3D timing: total={:>5.1}ms yolox={:>5.1}ms pre={:>4.1}ms run={:>5.1}ms decode={:>4.1}ms face={:>4.1}ms",
            dt_total.as_secs_f32() * 1000.0,
            dt_yolox.as_secs_f32() * 1000.0,
            dt_pre.as_secs_f32() * 1000.0,
            dt_run.as_secs_f32() * 1000.0,
            dt_decode.as_secs_f32() * 1000.0,
            dt_face.as_secs_f32() * 1000.0,
        );

        let mut detection = annotation::build_annotation(&joints);
        detection.bounding_box = annotation_bbox;
        PoseEstimate {
            annotation: detection,
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
