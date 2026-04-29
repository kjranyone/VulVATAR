#![allow(dead_code)]
use super::PoseEstimate;
#[cfg(feature = "inference")]
use super::face_expression::{FaceExpressionSmoother, FACE_LANDMARK_COUNT};
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
use ort::value::{Shape, TensorRef};
#[cfg(feature = "inference")]
use serde_json::Value as JsonValue;
#[cfg(feature = "inference")]
use std::fs;
#[cfg(feature = "inference")]
use std::path::{Path, PathBuf};

/// Which ONNX Runtime execution provider the session ended up on. Surfaced
/// to the GUI so users can tell whether the GPU path is active or whether
/// DirectML failed to register and tracking quietly fell back to CPU
/// (which is much slower).
#[derive(Clone, Debug)]
pub enum InferenceBackend {
    /// DirectML EP registered successfully. GPU-accelerated.
    DirectMl,
    /// CPU EP, deliberately — either the `inference-gpu` cargo feature is
    /// off, or DirectML was never attempted for this build.
    Cpu,
    /// CPU EP because DirectML failed at registration (no DX12 hardware,
    /// missing DLL, driver issue). Slower than `DirectMl` and slower than
    /// the user expects from a GPU-feature build, so we surface the
    /// reason verbatim.
    CpuFromDirectMlFailure { reason: String },
}

impl InferenceBackend {
    /// One-line label suitable for the inspector. Mirrors the wording in
    /// the plan spec ("Backend: DirectML" / "Backend: CPU (DirectML
    /// unavailable)").
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

pub struct CigPoseInference {
    #[cfg(feature = "inference")]
    session: Session,
    #[cfg(feature = "inference")]
    input_name: String,
    #[cfg(feature = "inference")]
    detector: Option<YoloxDetector>,
    #[cfg(feature = "inference")]
    load_warnings: Vec<String>,
    #[cfg(feature = "inference")]
    input_w: u32,
    #[cfg(feature = "inference")]
    input_h: u32,
    #[cfg(feature = "inference")]
    split_ratio: f32,
    #[cfg(feature = "inference")]
    face_smoother: FaceExpressionSmoother,
    #[cfg(feature = "inference")]
    backend: InferenceBackend,
}

impl CigPoseInference {
    #[cfg(feature = "inference")]
    pub fn new(model_path: &str) -> std::result::Result<Self, String> {
        Self::new_with_detector(model_path, None)
    }

    #[cfg(feature = "inference")]
    pub fn new_with_detector(
        model_path: &str,
        detector_path: Option<&str>,
    ) -> std::result::Result<Self, String> {
        info!("Loading CIGPose model from {}", model_path);

        let (session, backend) = build_session(model_path, 4, "CIGPose")?;

        let input_name = session
            .inputs()
            .first()
            .map(|input| input.name().to_string())
            .unwrap_or_else(|| "input".to_string());
        let config = read_pose_model_config(&session, model_path);
        info!(
            "CIGPose model config: {}x{}, split_ratio={}",
            config.input_w, config.input_h, config.split_ratio
        );

        let mut load_warnings = Vec::new();
        let detector = match detector_path {
            Some(path) => match YoloxDetector::new(path) {
                Ok(detector) => {
                    info!("Loaded YOLOX person detector from {}", path);
                    Some(detector)
                }
                Err(e) => {
                    let warning = format!("YOLOX detector disabled: {}", e);
                    warn!("{}", warning);
                    load_warnings.push(warning);
                    None
                }
            },
            None => {
                let warning = "YOLOX detector not found; CIGPose will use a full-frame person crop"
                    .to_string();
                warn!("{}", warning);
                load_warnings.push(warning);
                None
            }
        };

        Ok(Self {
            session,
            input_name,
            detector,
            load_warnings,
            input_w: config.input_w,
            input_h: config.input_h,
            split_ratio: config.split_ratio,
            face_smoother: FaceExpressionSmoother::new(),
            backend,
        })
    }

    #[cfg(feature = "inference")]
    pub fn from_models_dir(models_dir: impl AsRef<Path>) -> std::result::Result<Self, String> {
        Self::from_models_dir_with_pref(models_dir, false)
    }

    /// Same as [`Self::from_models_dir`], but the `prefer_lower_body`
    /// flag flips the default model preference (ubody → wholebody) so
    /// hip/knee/ankle keypoints actually get emitted. Use when the GUI
    /// "Track lower body" toggle is on. Trade-off: wholebody variants
    /// are typically smaller-resolution and detect hands less precisely
    /// than the same-tier ubody model, so leg tracking costs hand
    /// fidelity.
    #[cfg(feature = "inference")]
    pub fn from_models_dir_with_pref(
        models_dir: impl AsRef<Path>,
        prefer_lower_body: bool,
    ) -> std::result::Result<Self, String> {
        let models_dir = models_dir.as_ref();
        let model_path = find_best_pose_model(models_dir, prefer_lower_body).ok_or_else(|| {
            format!(
                "No CIGPose ONNX model found in {}. Run ./dev.ps1 setup.",
                models_dir.display()
            )
        })?;
        let detector_path = models_dir.join("yolox_nano.onnx");
        let detector = detector_path
            .is_file()
            .then(|| detector_path.to_string_lossy().to_string());
        Self::new_with_detector(&model_path.to_string_lossy(), detector.as_deref())
    }

    #[cfg(feature = "inference")]
    pub fn take_load_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.load_warnings)
    }

    #[cfg(feature = "inference")]
    pub fn backend(&self) -> &InferenceBackend {
        &self.backend
    }

    #[cfg(not(feature = "inference"))]
    pub fn new(_model_path: &str) -> Result<Self, String> {
        Err("Inference feature is not enabled. Build with --features inference".to_string())
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

        let bbox = self
            .detector
            .as_mut()
            .and_then(|detector| detector.detect_best_person(rgb_data, width, height))
            .unwrap_or_else(|| PersonBBox::full_frame(width, height));

        let (tensor, crop_region) =
            preprocess_person(rgb_data, width, height, bbox, self.input_w, self.input_h);

        let input = match TensorRef::from_array_view(&tensor) {
            Ok(v) => v,
            Err(e) => {
                error!("TensorRef creation failed: {}", e);
                return empty_estimate(frame_index);
            }
        };

        let outputs = match self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
        {
            Ok(out) => out,
            Err(e) => {
                error!("ONNX inference failed: {}", e);
                return empty_estimate(frame_index);
            }
        };

        if outputs.len() < 2 {
            error!(
                "ONNX inference returned {} outputs, expected at least 2",
                outputs.len()
            );
            return empty_estimate(frame_index);
        }

        let output_x = outputs.get("simcc_x").unwrap_or(&outputs[0]);
        let output_y = outputs.get("simcc_y").unwrap_or(&outputs[1]);

        let (shape_x, data_x) = match output_x.try_extract_tensor::<f32>() {
            Ok(v) => v,
            Err(e) => {
                error!("ONNX simcc_x extraction failed: {}", e);
                return empty_estimate(frame_index);
            }
        };
        let (_shape_y, data_y) = match output_y.try_extract_tensor::<f32>() {
            Ok(v) => v,
            Err(e) => {
                error!("ONNX simcc_y extraction failed: {}", e);
                return empty_estimate(frame_index);
            }
        };

        let model_kpts = decode_simcc_from_slice(data_x, data_y, shape_x, self.split_ratio);
        let frame_kpts =
            remap_keypoints_to_frame(model_kpts, crop_region, self.input_w, self.input_h);

        let aspect = width as f32 / height as f32;
        let mut skeleton =
            map_to_source_skeleton(&frame_kpts, width, height, aspect, frame_index);

        // Drive expression weights from the face block via the smoother. The
        // smoother holds asymmetric attack/release state per channel, so a
        // raw "open mouth" frame ramps up over a few frames instead of
        // snapping. Coordinates are normalised to the same y-up image space
        // used by the rest of the skeleton.
        let face_block = face_block_from_keypoints(&frame_kpts, width, height, aspect);
        skeleton.expressions = self.face_smoother.update(&face_block);

        let annotation = build_annotation(frame_kpts, width, height, Some(bbox));

        PoseEstimate {
            skeleton,
            annotation,
        }
    }
}

#[cfg(feature = "inference")]
#[derive(Clone, Copy, Debug)]
struct PoseModelConfig {
    input_w: u32,
    input_h: u32,
    split_ratio: f32,
}

/// Build an ORT [`Session`] with a small thread pool, optionally
/// routing through the DirectML execution provider when the
/// `inference-gpu` cargo feature is enabled. If DirectML registration
/// fails (no DX12 hardware, missing DLL, driver issue), we emit a
/// warning and retry on a CPU-only builder so a broken GPU path
/// doesn't kill tracking.
///
/// Returns the session alongside the EP that was actually used so the
/// caller can surface this to the UI — a silent CPU fallback is the
/// difference between "tracking is fine" and "tracking is fine but
/// taking 10× longer than the user expected".
#[cfg(all(feature = "inference", feature = "inference-gpu"))]
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
                "{} DirectML EP registration failed for '{}': {}. \
                 Falling back to CPU.",
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

/// CPU-only build path used when the `inference-gpu` feature is off.
/// Mirrors the GPU build's fallback behavior so the EP list is identical.
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

/// Attempt the DirectML EP path. Kept separate so the error-type
/// shuffling between `with_execution_providers` (BuilderResult) and
/// `commit_from_file` (ort::Result) doesn't leak into the caller.
#[cfg(feature = "inference-gpu")]
fn try_build_directml_session(
    model_path: &str,
    intra_threads: usize,
) -> std::result::Result<Session, String> {
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

#[cfg(feature = "inference")]
fn read_pose_model_config(session: &Session, model_path: &str) -> PoseModelConfig {
    let fallback = config_from_filename(model_path);
    if let Ok(metadata) = session.metadata() {
        if let Some(meta_str) = metadata.custom("cigpose_meta") {
            if let Ok(meta) = serde_json::from_str::<JsonValue>(&meta_str) {
                let input_w = meta
                    .get("input_w")
                    .and_then(JsonValue::as_u64)
                    .map(|v| v as u32)
                    .unwrap_or(fallback.input_w);
                let input_h = meta
                    .get("input_h")
                    .and_then(JsonValue::as_u64)
                    .map(|v| v as u32)
                    .unwrap_or(fallback.input_h);
                let split_ratio = meta
                    .get("split_ratio")
                    .and_then(JsonValue::as_f64)
                    .map(|v| v as f32)
                    .unwrap_or(fallback.split_ratio);
                return PoseModelConfig {
                    input_w,
                    input_h,
                    split_ratio,
                };
            }
        }
    }
    fallback
}

#[cfg(feature = "inference")]
fn config_from_filename(model_path: &str) -> PoseModelConfig {
    let name = Path::new(model_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    if name.contains("384x288") {
        PoseModelConfig {
            input_w: 288,
            input_h: 384,
            split_ratio: 2.0,
        }
    } else {
        PoseModelConfig {
            input_w: 192,
            input_h: 256,
            split_ratio: 2.0,
        }
    }
}

#[cfg(feature = "inference")]
fn find_best_pose_model(models_dir: &Path, prefer_lower_body: bool) -> Option<PathBuf> {
    let entries = fs::read_dir(models_dir).ok()?;
    entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .filter_map(|path| {
            let score = pose_model_score(&path, prefer_lower_body)?;
            Some((score, path))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, path)| path)
}

#[cfg(feature = "inference")]
fn pose_model_score(path: &Path, prefer_lower_body: bool) -> Option<i32> {
    let name = path.file_name()?.to_str()?.to_ascii_lowercase();
    if !name.starts_with("cigpose-") || !name.ends_with(".onnx") {
        return None;
    }

    let mut score = 0;
    score += if name.contains("384x288") {
        10_000
    } else if name.contains("256x192") {
        5_000
    } else {
        1_000
    };
    // Body coverage preference: ubody is higher fidelity for the upper
    // body (preferred default for sitting VTubers), wholebody adds legs
    // at some cost. The prefer_lower_body flag flips which one wins.
    let (ubody_bonus, wholebody_bonus) = if prefer_lower_body {
        (300, 500)
    } else {
        (500, 300)
    };
    score += if name.contains("coco-ubody") {
        ubody_bonus
    } else if name.contains("wholebody") {
        wholebody_bonus
    } else {
        0
    };
    score += if name.starts_with("cigpose-x") {
        300
    } else if name.starts_with("cigpose-l") {
        200
    } else if name.starts_with("cigpose-m") {
        100
    } else {
        0
    };
    Some(score)
}

#[cfg(feature = "inference")]
fn empty_estimate(frame_index: u64) -> PoseEstimate {
    PoseEstimate {
        skeleton: SourceSkeleton::empty(frame_index),
        annotation: DetectionAnnotation::default(),
    }
}

#[cfg(feature = "inference")]
#[derive(Clone, Copy, Debug)]
struct PersonBBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

#[cfg(feature = "inference")]
impl PersonBBox {
    fn full_frame(width: u32, height: u32) -> Self {
        Self {
            x1: 0.0,
            y1: 0.0,
            x2: width as f32,
            y2: height as f32,
        }
    }

    fn area(&self) -> f32 {
        (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0)
    }
}

#[cfg(feature = "inference")]
#[derive(Clone, Copy, Debug)]
struct CropRegion {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

#[cfg(feature = "inference")]
fn preprocess_person(
    rgb_data: &[u8],
    src_w: u32,
    src_h: u32,
    bbox: PersonBBox,
    dst_w: u32,
    dst_h: u32,
) -> (Array4<f32>, CropRegion) {
    let mut tensor = Array4::<f32>::zeros((1, 3, dst_h as usize, dst_w as usize));

    let crop = person_crop_region(bbox, src_w, src_h, dst_w, dst_h);
    let crop_w = (crop.x2 - crop.x1).max(1.0);
    let crop_h = (crop.y2 - crop.y1).max(1.0);

    // CIGPose/MMPose ImageNet normalization in 0..255 RGB space.
    let mean = [123.675, 116.28, 103.53];
    let std = [58.395, 57.12, 57.375];

    for y in 0..dst_h {
        let v = if dst_h > 1 {
            y as f32 / (dst_h - 1) as f32
        } else {
            0.0
        };
        let src_y = crop.y1 + v * (crop_h - 1.0);
        for x in 0..dst_w {
            let u = if dst_w > 1 {
                x as f32 / (dst_w - 1) as f32
            } else {
                0.0
            };
            let src_x = crop.x1 + u * (crop_w - 1.0);
            let [r, g, b] = sample_bilinear_rgb(rgb_data, src_w, src_h, src_x, src_y);

            tensor[[0, 0, y as usize, x as usize]] = (r - mean[0]) / std[0];
            tensor[[0, 1, y as usize, x as usize]] = (g - mean[1]) / std[1];
            tensor[[0, 2, y as usize, x as usize]] = (b - mean[2]) / std[2];
        }
    }

    (tensor, crop)
}

#[cfg(feature = "inference")]
fn person_crop_region(
    bbox: PersonBBox,
    src_w: u32,
    src_h: u32,
    model_w: u32,
    model_h: u32,
) -> CropRegion {
    let cx = (bbox.x1 + bbox.x2) * 0.5;
    let cy = (bbox.y1 + bbox.y2) * 0.5;
    let mut bw = (bbox.x2 - bbox.x1).max(1.0);
    let mut bh = (bbox.y2 - bbox.y1).max(1.0);
    let target_aspect = model_w as f32 / model_h as f32;

    if bw / bh > target_aspect {
        bh = bw / target_aspect;
    } else {
        bw = bh * target_aspect;
    }
    bw *= 1.25;
    bh *= 1.25;

    CropRegion {
        x1: (cx - bw * 0.5).clamp(0.0, src_w.saturating_sub(1) as f32),
        y1: (cy - bh * 0.5).clamp(0.0, src_h.saturating_sub(1) as f32),
        x2: (cx + bw * 0.5).clamp(1.0, src_w as f32),
        y2: (cy + bh * 0.5).clamp(1.0, src_h as f32),
    }
}

#[cfg(feature = "inference")]
fn sample_bilinear_rgb(rgb_data: &[u8], width: u32, height: u32, x: f32, y: f32) -> [f32; 3] {
    let x = x.clamp(0.0, width.saturating_sub(1) as f32);
    let y = y.clamp(0.0, height.saturating_sub(1) as f32);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;

    let p00 = pixel_rgb(rgb_data, width, x0, y0);
    let p10 = pixel_rgb(rgb_data, width, x1, y0);
    let p01 = pixel_rgb(rgb_data, width, x0, y1);
    let p11 = pixel_rgb(rgb_data, width, x1, y1);

    let mut out = [0.0; 3];
    for c in 0..3 {
        let top = p00[c] * (1.0 - tx) + p10[c] * tx;
        let bottom = p01[c] * (1.0 - tx) + p11[c] * tx;
        out[c] = top * (1.0 - ty) + bottom * ty;
    }
    out
}

#[cfg(feature = "inference")]
fn pixel_rgb(rgb_data: &[u8], width: u32, x: u32, y: u32) -> [f32; 3] {
    let i = ((y * width + x) * 3) as usize;
    if i + 2 >= rgb_data.len() {
        return [0.0, 0.0, 0.0];
    }
    [
        rgb_data[i] as f32,
        rgb_data[i + 1] as f32,
        rgb_data[i + 2] as f32,
    ]
}

#[cfg(feature = "inference")]
struct YoloxDetector {
    session: Session,
    input_name: String,
    input_size: u32,
    conf_thresh: f32,
    grids: Vec<[f32; 2]>,
    strides: Vec<f32>,
}

#[cfg(feature = "inference")]
impl YoloxDetector {
    fn new(model_path: &str) -> std::result::Result<Self, String> {
        // YOLOX runs on the same EP as CIGPose in practice (DirectML
        // registration is process-wide); the backend tag from this build
        // is intentionally discarded — CigPoseInference's tag is the
        // canonical one shown in the UI.
        let (session, _backend) = build_session(model_path, 2, "YOLOX")?;
        let input_name = session
            .inputs()
            .first()
            .map(|input| input.name().to_string())
            .unwrap_or_else(|| "input".to_string());
        let input_size = 416;
        let (grids, strides) = build_yolox_grids(input_size);
        Ok(Self {
            session,
            input_name,
            input_size,
            conf_thresh: 0.5,
            grids,
            strides,
        })
    }

    fn detect_best_person(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
    ) -> Option<PersonBBox> {
        let (tensor, ratio) = preprocess_yolox_letterbox(rgb_data, width, height, self.input_size);
        let input = match TensorRef::from_array_view(&tensor) {
            Ok(v) => v,
            Err(e) => {
                warn!("YOLOX TensorRef creation failed: {}", e);
                return None;
            }
        };
        let outputs = match self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
        {
            Ok(out) => out,
            Err(e) => {
                warn!("YOLOX inference failed: {}", e);
                return None;
            }
        };
        if outputs.len() == 0 {
            warn!("YOLOX inference returned no outputs");
            return None;
        }
        let first = &outputs[0];
        let (shape, data) = match first.try_extract_tensor::<f32>() {
            Ok(v) => v,
            Err(e) => {
                warn!("YOLOX output extraction failed: {}", e);
                return None;
            }
        };
        decode_best_yolox_person(
            data,
            shape,
            &self.grids,
            &self.strides,
            ratio,
            width,
            height,
            self.conf_thresh,
        )
    }
}

#[cfg(feature = "inference")]
fn build_yolox_grids(input_size: u32) -> (Vec<[f32; 2]>, Vec<f32>) {
    let mut grids = Vec::new();
    let mut strides = Vec::new();
    for stride in [8u32, 16, 32] {
        let g = input_size / stride;
        for y in 0..g {
            for x in 0..g {
                grids.push([x as f32, y as f32]);
                strides.push(stride as f32);
            }
        }
    }
    (grids, strides)
}

#[cfg(feature = "inference")]
fn preprocess_yolox_letterbox(
    rgb_data: &[u8],
    src_w: u32,
    src_h: u32,
    input_size: u32,
) -> (Array4<f32>, f32) {
    let mut tensor =
        Array4::<f32>::from_elem((1, 3, input_size as usize, input_size as usize), 114.0);
    let ratio = (input_size as f32 / src_h as f32).min(input_size as f32 / src_w as f32);
    let dst_w = ((src_w as f32 * ratio) as u32).max(1).min(input_size);
    let dst_h = ((src_h as f32 * ratio) as u32).max(1).min(input_size);

    for y in 0..dst_h {
        let src_y = y as f32 / ratio;
        for x in 0..dst_w {
            let src_x = x as f32 / ratio;
            let [r, g, b] = sample_bilinear_rgb(rgb_data, src_w, src_h, src_x, src_y);
            tensor[[0, 0, y as usize, x as usize]] = r;
            tensor[[0, 1, y as usize, x as usize]] = g;
            tensor[[0, 2, y as usize, x as usize]] = b;
        }
    }

    (tensor, ratio)
}

#[cfg(feature = "inference")]
#[allow(clippy::too_many_arguments)]
fn decode_best_yolox_person(
    data: &[f32],
    shape: &Shape,
    grids: &[[f32; 2]],
    strides: &[f32],
    ratio: f32,
    frame_w: u32,
    frame_h: u32,
    conf_thresh: f32,
) -> Option<PersonBBox> {
    if shape.len() < 3 || ratio <= 0.0 {
        return None;
    }
    let rows = shape[1] as usize;
    let cols = shape[2] as usize;
    if rows == 0 || cols < 6 || data.len() < rows.saturating_mul(cols) {
        return None;
    }

    let frame_area = (frame_w as f32 * frame_h as f32).max(1.0);
    let frame_center = [frame_w as f32 * 0.5, frame_h as f32 * 0.5];
    let half_diag = (frame_center[0].hypot(frame_center[1])).max(1.0);
    let mut best: Option<(f32, PersonBBox)> = None;
    for row in 0..rows.min(grids.len()).min(strides.len()) {
        let base = row * cols;
        let objectness = data[base + 4];
        let person_class = data[base + 5];
        let score = objectness * person_class;
        if score < conf_thresh {
            continue;
        }

        let stride = strides[row];
        let grid = grids[row];
        let cx = (data[base] + grid[0]) * stride;
        let cy = (data[base + 1] + grid[1]) * stride;
        let bw = data[base + 2].clamp(-10.0, 10.0).exp() * stride;
        let bh = data[base + 3].clamp(-10.0, 10.0).exp() * stride;

        let bbox = PersonBBox {
            x1: ((cx - bw * 0.5) / ratio).clamp(0.0, frame_w as f32),
            y1: ((cy - bh * 0.5) / ratio).clamp(0.0, frame_h as f32),
            x2: ((cx + bw * 0.5) / ratio).clamp(0.0, frame_w as f32),
            y2: ((cy + bh * 0.5) / ratio).clamp(0.0, frame_h as f32),
        };
        if bbox.area() < 16.0 {
            continue;
        }

        // VulVATAR is a single-operator app: prefer the confident person
        // closest to the camera center, with a mild penalty for tiny boxes.
        let bbox_center = [(bbox.x1 + bbox.x2) * 0.5, (bbox.y1 + bbox.y2) * 0.5];
        let center_dist = ((bbox_center[0] - frame_center[0])
            .hypot(bbox_center[1] - frame_center[1])
            / half_diag)
            .clamp(0.0, 1.0);
        let center_weight = (1.0 - 0.45 * center_dist).clamp(0.2, 1.0);
        let area_weight = (bbox.area() / frame_area).sqrt().clamp(0.25, 1.0);
        let selection_score = score * center_weight * area_weight;

        match best {
            Some((best_score, _)) if best_score >= selection_score => {}
            _ => best = Some((selection_score, bbox)),
        }
    }

    best.map(|(_, bbox)| bbox)
}

#[cfg(feature = "inference")]
fn decode_simcc_from_slice(
    data_x: &[f32],
    data_y: &[f32],
    shape_x: &Shape,
    split_ratio: f32,
) -> Vec<(f32, f32, f32)> {
    let num_keypoints = shape_x[1] as usize;
    let x_channels = shape_x[2] as usize;

    let y_channels = if data_y.len() >= num_keypoints * x_channels && num_keypoints > 0 {
        data_y.len() / num_keypoints
    } else {
        256 * 2
    };

    let mut kpts = Vec::with_capacity(num_keypoints);

    for k in 0..num_keypoints {
        let x_start = k * x_channels;
        let x_end = (k + 1) * x_channels;
        let x_slice = &data_x[x_start..x_end.min(data_x.len())];

        let mut max_x_idx = 0;
        let mut max_x_val = f32::MIN;
        for (i, &val) in x_slice.iter().enumerate() {
            if val > max_x_val {
                max_x_val = val;
                max_x_idx = i;
            }
        }

        let y_start = k * y_channels;
        let y_end = (k + 1) * y_channels;
        let y_slice = &data_y[y_start..y_end.min(data_y.len())];

        let mut max_y_idx = 0;
        let mut max_y_val = f32::MIN;
        for (i, &val) in y_slice.iter().enumerate() {
            if val > max_y_val {
                max_y_val = val;
                max_y_idx = i;
            }
        }

        let x = max_x_idx as f32 / split_ratio;
        let y = max_y_idx as f32 / split_ratio;
        let conf = max_x_val.min(max_y_val);
        kpts.push((x, y, conf));
    }
    kpts
}

#[cfg(feature = "inference")]
fn remap_keypoints_to_frame(
    kpts: Vec<(f32, f32, f32)>,
    crop: CropRegion,
    model_w: u32,
    model_h: u32,
) -> Vec<(f32, f32, f32)> {
    let scale_x = (crop.x2 - crop.x1) / model_w as f32;
    let scale_y = (crop.y2 - crop.y1) / model_h as f32;
    kpts.into_iter()
        .map(|(x, y, c)| (x * scale_x + crop.x1, y * scale_y + crop.y1, c))
        .collect()
}

/// COCO-wholebody body-only skeleton connections (for the GUI overlay).
const SKELETON_CONNECTIONS: [(usize, usize); 16] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
];

#[cfg(feature = "inference")]
fn build_annotation(
    kpts: Vec<(f32, f32, f32)>,
    orig_w: u32,
    orig_h: u32,
    bbox: Option<PersonBBox>,
) -> DetectionAnnotation {
    let ow = orig_w as f32;
    let oh = orig_h as f32;
    let norm_kpts: Vec<(f32, f32, f32)> = kpts
        .into_iter()
        .map(|(x, y, c)| (x / ow, y / oh, c))
        .collect();
    let bounding_box = bbox.map(|b| (b.x1 / ow, b.y1 / oh, b.x2 / ow, b.y2 / oh));
    DetectionAnnotation {
        keypoints: norm_kpts,
        skeleton: SKELETON_CONNECTIONS.to_vec(),
        bounding_box,
    }
}

// COCO-WholeBody keypoint indices.
//
// Layout (133 keypoints):
//   0..=16    body
//   17..=22   feet
//   23..=90   face (68)
//   91..=111  left hand (21, MediaPipe layout)
//   112..=132 right hand (21)
//
// The hand layout per 21-point block is the MediaPipe convention:
//   +0  WRIST
//   +1..+4   THUMB  (CMC, MCP, IP, TIP)
//   +5..+8   INDEX  (MCP, PIP, DIP, TIP)
//   +9..+12  MIDDLE (MCP, PIP, DIP, TIP)
//   +13..+16 RING   (MCP, PIP, DIP, TIP)
//   +17..+20 LITTLE (MCP, PIP, DIP, TIP)
#[cfg(feature = "inference")]
mod coco {
    pub const NOSE: usize = 0;
    pub const LEFT_EYE: usize = 1;
    pub const RIGHT_EYE: usize = 2;
    pub const LEFT_EAR: usize = 3;
    pub const RIGHT_EAR: usize = 4;
    pub const LEFT_SHOULDER: usize = 5;
    pub const RIGHT_SHOULDER: usize = 6;
    pub const LEFT_ELBOW: usize = 7;
    pub const RIGHT_ELBOW: usize = 8;
    pub const LEFT_WRIST: usize = 9;
    pub const RIGHT_WRIST: usize = 10;
    pub const LEFT_HIP: usize = 11;
    pub const RIGHT_HIP: usize = 12;
    pub const LEFT_KNEE: usize = 13;
    pub const RIGHT_KNEE: usize = 14;
    pub const LEFT_ANKLE: usize = 15;
    pub const RIGHT_ANKLE: usize = 16;
    pub const LEFT_BIG_TOE: usize = 17;
    pub const LEFT_SMALL_TOE: usize = 18;
    pub const LEFT_HEEL: usize = 19;
    pub const RIGHT_BIG_TOE: usize = 20;
    pub const RIGHT_SMALL_TOE: usize = 21;
    pub const RIGHT_HEEL: usize = 22;

    /// Start index of the dense iBUG 68-point face block.
    pub const FACE_BLOCK_START: usize = 23;
    pub const LEFT_HAND_START: usize = 91;
    pub const RIGHT_HAND_START: usize = 112;

    /// Offsets into a single 21-point hand block.
    pub mod hand {
        pub const THUMB_CMC: usize = 1;
        pub const THUMB_MCP: usize = 2;
        pub const THUMB_IP: usize = 3;
        pub const THUMB_TIP: usize = 4;
        pub const INDEX_MCP: usize = 5;
        pub const INDEX_PIP: usize = 6;
        pub const INDEX_DIP: usize = 7;
        pub const INDEX_TIP: usize = 8;
        pub const MIDDLE_MCP: usize = 9;
        pub const MIDDLE_PIP: usize = 10;
        pub const MIDDLE_DIP: usize = 11;
        pub const MIDDLE_TIP: usize = 12;
        pub const RING_MCP: usize = 13;
        pub const RING_PIP: usize = 14;
        pub const RING_DIP: usize = 15;
        pub const RING_TIP: usize = 16;
        pub const LITTLE_MCP: usize = 17;
        pub const LITTLE_PIP: usize = 18;
        pub const LITTLE_DIP: usize = 19;
        pub const LITTLE_TIP: usize = 20;
    }
}

/// Convert a flat keypoint list into a [`SourceSkeleton`].
///
/// Positions are normalised to the camera's image plane:
/// * x ∈ `[-aspect, +aspect]` (aspect = image_w / image_h)
/// * y ∈ `[-1, +1]`, **y-up** (the +Y axis points toward the top of the image)
///
/// Body joints come straight from the COCO-WholeBody body subset. The Head
/// rotation is *not* derived from the nose position (that only measures
/// where the face is in the frame, not where it is pointed); instead the
/// nose + eyes + ears geometry is fed into [`face_pose_from_keypoints`] to
/// yield yaw/pitch/roll.
#[cfg(feature = "inference")]
fn map_to_source_skeleton(
    kpts: &[(f32, f32, f32)],
    input_w: u32,
    input_h: u32,
    aspect: f32,
    frame_index: u64,
) -> SourceSkeleton {
    let kpt = |i: usize| -> (f32, f32, f32) { kpts.get(i).copied().unwrap_or((0.0, 0.0, 0.0)) };
    let to_norm_x = |x: f32| ((x / input_w as f32) * 2.0 - 1.0) * aspect;
    let to_norm_y = |y: f32| 1.0 - (y / input_h as f32) * 2.0;
    // The CIGPose backend is 2D-only — the new 3D `SourceJoint` schema
    // is filled in with `z = 0` here. This entire `inference.rs` is
    // scheduled for replacement by the MediaPipe Holistic pipeline in
    // P28-P32; until then the avatar simply runs without depth (legs
    // and arms collapse to the image plane), which is acceptable for
    // the migration window because lift-based depth recovery has
    // already been removed.
    let to_joint = |k: (f32, f32, f32)| SourceJoint {
        position: [to_norm_x(k.0), to_norm_y(k.1), 0.0],
        confidence: k.2,
    };

    let mut sk = SourceSkeleton::empty(frame_index);

    // Body keypoints → humanoid joint positions. The solver uses parent
    // positions (shoulder/hip) plus child positions (elbow/knee/etc.) to
    // reconstruct each bone direction; we populate the hierarchy entries
    // the solver expects to see.
    let min_body_conf = 0.0; // store everything, solver filters by threshold
    let shoulders_l = kpt(coco::LEFT_SHOULDER);
    let shoulders_r = kpt(coco::RIGHT_SHOULDER);
    let elbow_l = kpt(coco::LEFT_ELBOW);
    let elbow_r = kpt(coco::RIGHT_ELBOW);
    let wrist_l = kpt(coco::LEFT_WRIST);
    let wrist_r = kpt(coco::RIGHT_WRIST);
    let hip_l = kpt(coco::LEFT_HIP);
    let hip_r = kpt(coco::RIGHT_HIP);
    let knee_l = kpt(coco::LEFT_KNEE);
    let knee_r = kpt(coco::RIGHT_KNEE);
    let ankle_l = kpt(coco::LEFT_ANKLE);
    let ankle_r = kpt(coco::RIGHT_ANKLE);

    // Anatomical-left on the subject (COCO LEFT_*) appears on the LEFT
    // side of a non-mirrored image, which is the SAME side as the avatar's
    // *right* bones when the avatar faces the camera. We therefore cross
    // the keypoints — LEFT_SHOULDER drives RightShoulder, etc. — so that
    // the mirror-style VTuber interaction "lift my left arm → the arm on
    // my left in the screen lifts" falls out of matching signs on X
    // between source joints (image space) and avatar rest world.
    sk.put_joint(
        HumanoidBone::RightShoulder,
        to_joint(shoulders_l),
        min_body_conf,
    );
    sk.put_joint(
        HumanoidBone::LeftShoulder,
        to_joint(shoulders_r),
        min_body_conf,
    );
    sk.put_joint(
        HumanoidBone::RightUpperArm,
        to_joint(shoulders_l),
        min_body_conf,
    );
    sk.put_joint(
        HumanoidBone::LeftUpperArm,
        to_joint(shoulders_r),
        min_body_conf,
    );
    sk.put_joint(
        HumanoidBone::RightLowerArm,
        to_joint(elbow_l),
        min_body_conf,
    );
    sk.put_joint(HumanoidBone::LeftLowerArm, to_joint(elbow_r), min_body_conf);
    sk.put_joint(HumanoidBone::RightHand, to_joint(wrist_l), min_body_conf);
    sk.put_joint(HumanoidBone::LeftHand, to_joint(wrist_r), min_body_conf);
    sk.put_joint(HumanoidBone::RightUpperLeg, to_joint(hip_l), min_body_conf);
    sk.put_joint(HumanoidBone::LeftUpperLeg, to_joint(hip_r), min_body_conf);
    sk.put_joint(HumanoidBone::RightLowerLeg, to_joint(knee_l), min_body_conf);
    sk.put_joint(HumanoidBone::LeftLowerLeg, to_joint(knee_r), min_body_conf);
    sk.put_joint(HumanoidBone::RightFoot, to_joint(ankle_l), min_body_conf);
    sk.put_joint(HumanoidBone::LeftFoot, to_joint(ankle_r), min_body_conf);

    // Foot tip targets (big toe). The solver looks at
    // `fingertips[Foot]` for the foot bone's tip direction; with this
    // populated the foot can rotate to point where the subject's toes
    // point instead of being stuck in T-pose orientation.
    let big_toe_l = kpt(coco::LEFT_BIG_TOE);
    let big_toe_r = kpt(coco::RIGHT_BIG_TOE);
    if big_toe_l.2 > 0.0 {
        // Mirror: subject-LEFT toe drives avatar-Right foot.
        sk.fingertips
            .insert(HumanoidBone::RightFoot, to_joint(big_toe_l));
    }
    if big_toe_r.2 > 0.0 {
        sk.fingertips
            .insert(HumanoidBone::LeftFoot, to_joint(big_toe_r));
    }

    // Hips: midpoint of the two hip keypoints. Spine's base.
    if hip_l.2 > 0.0 && hip_r.2 > 0.0 {
        let mid = SourceJoint {
            position: [
                (to_norm_x(hip_l.0) + to_norm_x(hip_r.0)) * 0.5,
                (to_norm_y(hip_l.1) + to_norm_y(hip_r.1)) * 0.5,
                0.0,
            ],
            confidence: hip_l.2.min(hip_r.2),
        };
        sk.put_joint(HumanoidBone::Hips, mid, min_body_conf);
    }

    // Finger keypoints (mirror-swapped to match the body: user's anatomical
    // LEFT hand drives the avatar's RightHand chain and vice versa). Each
    // finger bone is anchored at its PROXIMAL joint and its tip is the
    // keypoint of the next joint along the finger — the solver uses the
    // base-to-tip vector for rotation matching and the chain is walked
    // parent-to-child so rotations compose correctly.
    map_hand_keypoints(
        &mut sk,
        kpts,
        coco::LEFT_HAND_START,
        /* mirror target = */ Hand::Right,
        to_norm_x,
        to_norm_y,
        min_body_conf,
    );
    map_hand_keypoints(
        &mut sk,
        kpts,
        coco::RIGHT_HAND_START,
        Hand::Left,
        to_norm_x,
        to_norm_y,
        min_body_conf,
    );

    // Face pose from facial keypoints. These drive the Head bone; the nose
    // position itself is not used as a "rotation" input.
    let nose = kpt(coco::NOSE);
    let le = kpt(coco::LEFT_EYE);
    let re = kpt(coco::RIGHT_EYE);
    let lear = kpt(coco::LEFT_EAR);
    let rear = kpt(coco::RIGHT_EAR);
    sk.face = face_pose_from_keypoints(
        [to_norm_x(nose.0), to_norm_y(nose.1), nose.2],
        [to_norm_x(le.0), to_norm_y(le.1), le.2],
        [to_norm_x(re.0), to_norm_y(re.1), re.2],
        [to_norm_x(lear.0), to_norm_y(lear.1), lear.2],
        [to_norm_x(rear.0), to_norm_y(rear.1), rear.2],
    );

    // Per-joint confidence contributes to overall quality — approximate as
    // the max of nose and shoulders, the minimum needed to say "a person is
    // visible".
    sk.overall_confidence = nose.2.max(shoulders_l.2).max(shoulders_r.2);

    // Expressions are populated by the caller via FaceExpressionSmoother
    // (the smoother needs persistent state across frames, which this stateless
    // mapping cannot hold). Default-empty here is fine — caller overwrites.

    sk
}

/// Slice the dense iBUG 68-point face block out of CIGPose's flat keypoint
/// list, normalising to the same y-up image space the skeleton uses. Returns
/// an empty Vec if the keypoint list is too short — the smoother will then
/// see "no face" and decay any active weights.
#[cfg(feature = "inference")]
fn face_block_from_keypoints(
    kpts: &[(f32, f32, f32)],
    input_w: u32,
    input_h: u32,
    aspect: f32,
) -> Vec<(f32, f32, f32)> {
    let end = coco::FACE_BLOCK_START + FACE_LANDMARK_COUNT;
    if kpts.len() < end {
        return Vec::new();
    }
    let to_norm_x = |x: f32| ((x / input_w as f32) * 2.0 - 1.0) * aspect;
    let to_norm_y = |y: f32| 1.0 - (y / input_h as f32) * 2.0;
    kpts[coco::FACE_BLOCK_START..end]
        .iter()
        .map(|&(x, y, c)| (to_norm_x(x), to_norm_y(y), c))
        .collect()
}

/// Derive yaw/pitch/roll from facial keypoints (nose, eyes, ears).
///
/// Sign conventions (Y-up, +X right, +Z toward camera; compatible with
/// `quat_from_euler_ypr`):
///
/// * **Yaw** — positive turns the face toward image +X. Computed as
///   `asin(nose_offset_from_ear_midline / half_face_width)`. The nose sits
///   forward of the skull, so it projects off the ear midline by roughly
///   `sin(yaw)` as the head rotates.
///
/// * **Pitch** — positive looks *down* (chin toward chest). This matches
///   `quat_from_euler_ypr`, where a positive X-axis rotation rolls the top
///   of the head toward +Z (toward the camera). At neutral the nose
///   projects about `PITCH_NEUTRAL_RATIO` below the eye midpoint; when the
///   user drops their chin, the nose moves further down in the image and
///   the ratio grows, giving a positive pitch.
///
/// * **Roll** — positive tilts the head toward image +X (right ear drops).
///   Angle of the ear-to-ear line in the image plane.
///
/// Returns `None` when the nose is not confident, or when neither the ears
/// nor the eyes are both confident.
#[cfg(feature = "inference")]
fn face_pose_from_keypoints(
    nose: [f32; 3],
    left_eye: [f32; 3],
    right_eye: [f32; 3],
    left_ear: [f32; 3],
    right_ear: [f32; 3],
) -> Option<FacePose> {
    let min_c = 0.2;
    let have_ears = left_ear[2] >= min_c && right_ear[2] >= min_c;
    let have_eyes = left_eye[2] >= min_c && right_eye[2] >= min_c;
    let have_nose = nose[2] >= min_c;
    if !have_nose || (!have_ears && !have_eyes) {
        return None;
    }

    // COCO labels are from the subject's POV. When the subject faces
    // camera, subject-left lands on viewer-right, so `left_ear.x` is
    // larger than `right_ear.x` — i.e. `right_ear.x − left_ear.x < 0`.
    // When the subject faces *away* the sides flip and the difference
    // turns positive. We don't drop the pose entirely on rear views:
    // the magnitude is unreliable but the *sign* of the nose offset
    // still tracks subject body yaw and the solver uses it for
    // 135° / 225° disambiguation. Returning confidence = 0 keeps the
    // sign visible while preventing `apply_face_pose` from rotating
    // the head with the bogus magnitude.
    let rear_view = (have_ears && right_ear[0] - left_ear[0] > 0.0)
        || (!have_ears && have_eyes && right_eye[0] - left_eye[0] > 0.0);

    // Roll + face width. Prefer ears (wider baseline, more stable) and
    // fall back to eyes with a rough 2× scale factor to compensate for the
    // shorter eye-to-eye distance.
    let (roll, face_width) = if have_ears {
        let dx = right_ear[0] - left_ear[0];
        let dy = right_ear[1] - left_ear[1];
        (dy.atan2(dx.abs().max(1e-4)), dx.abs().max(1e-4))
    } else {
        let dx = right_eye[0] - left_eye[0];
        let dy = right_eye[1] - left_eye[1];
        (dy.atan2(dx.abs().max(1e-4)), (dx.abs().max(1e-4)) * 2.0)
    };

    // Yaw: nose horizontal offset from the ear/eye midpoint normalised to
    // half the face width, then through asin.
    let head_center_x = if have_ears {
        (left_ear[0] + right_ear[0]) * 0.5
    } else {
        (left_eye[0] + right_eye[0]) * 0.5
    };
    let half_face = (face_width * 0.5).max(1e-4);
    let raw_yaw_ratio = (nose[0] - head_center_x) / half_face;
    // Reject cases where the nose is past the side of the face — the
    // detector is hallucinating a "face" on a rear/side view where the
    // nose isn't actually visible. Without this guard rear views
    // produce a clamped ±60° yaw that rotates the avatar's head into a
    // back-of-camera orientation. A small slack (1.05) lets through
    // legitimate sharp head turns near the limit of the geometric
    // model.
    if raw_yaw_ratio.abs() > 1.05 {
        return None;
    }
    let yaw_ratio = raw_yaw_ratio.clamp(-1.0, 1.0);
    let yaw = yaw_ratio.asin().clamp(-1.05, 1.05);

    // Pitch: in image coords (y-up), neutral-posed nose sits below the eye
    // midpoint by a characteristic fraction of face width. Looking up
    // reduces that gap, looking down increases it.
    //
    // `PITCH_NEUTRAL_RATIO` is the rough average ratio
    // `(eye_y - nose_y) / face_width` for a front-facing neutral face.
    // `PITCH_GAIN` maps the normalised deviation to an angle in radians and
    // was picked empirically — a ±0.2 deviation (roughly ±30° of real
    // pitch) maps to about ±0.5 rad ≈ ±29°.
    const PITCH_NEUTRAL_RATIO: f32 = 0.35;
    const PITCH_GAIN: f32 = 2.5;
    let pitch = if have_eyes {
        let eye_mid_y = (left_eye[1] + right_eye[1]) * 0.5;
        let nose_below_eyes = eye_mid_y - nose[1];
        let ratio = nose_below_eyes / face_width;
        // delta > 0 when the nose has dropped relative to neutral (looking
        // down); feeds straight into a +X rotation that tilts the avatar's
        // head forward.
        let delta = ratio - PITCH_NEUTRAL_RATIO;
        (delta * PITCH_GAIN).clamp(-1.1, 1.1)
    } else {
        0.0
    };

    let confidence = if rear_view {
        // Magnitudes (yaw/pitch/roll) are unreliable on the back of
        // the head; force `apply_face_pose` to skip via the
        // confidence threshold. Yaw sign is still preserved on the
        // returned `FacePose` for downstream body-yaw sign use.
        0.0
    } else {
        let mut c = nose[2];
        if have_eyes {
            c = c.min(left_eye[2]).min(right_eye[2]);
        }
        if have_ears {
            c = c.min(left_ear[2]).min(right_ear[2]);
        }
        c
    };

    Some(FacePose {
        yaw,
        pitch,
        roll,
        confidence,
    })
}

/// Which avatar hand a block of keypoints should drive.
#[cfg(feature = "inference")]
#[derive(Clone, Copy)]
enum Hand {
    Left,
    Right,
}

/// Map a MediaPipe-layout 21-point hand block (starting at `block_start` in
/// the flat keypoint list) to the avatar's finger chain. Each finger bone
/// in the VRM humanoid is placed at its proximal joint and its direction is
/// set from the next joint down the finger:
///
/// | Bone           | Position kpt  | Direction tip kpt |
/// |----------------|---------------|-------------------|
/// | ThumbProximal  | THUMB_CMC (1) | THUMB_MCP (2)     |
/// | ThumbIntermed. | THUMB_MCP (2) | THUMB_IP  (3)     |
/// | ThumbDistal    | THUMB_IP  (3) | THUMB_TIP (4)     |
/// | IndexProximal  | INDEX_MCP (5) | INDEX_PIP (6)     |
/// | IndexIntermed. | INDEX_PIP (6) | INDEX_DIP (7)     |
/// | IndexDistal    | INDEX_DIP (7) | INDEX_TIP (8)     |
/// | ...same for MIDDLE / RING / LITTLE...             |
///
/// The solver then walks these chains parent-first and aligns each bone's
/// rest world direction to its observed 2D direction, so individual finger
/// bends (making a fist vs. a peace sign vs. an open hand) reflect on the
/// avatar.
#[cfg(feature = "inference")]
fn map_hand_keypoints(
    sk: &mut SourceSkeleton,
    kpts: &[(f32, f32, f32)],
    block_start: usize,
    target: Hand,
    to_norm_x: impl Fn(f32) -> f32,
    to_norm_y: impl Fn(f32) -> f32,
    min_conf: f32,
) {
    let at = |off: usize| -> SourceJoint {
        let k = kpts
            .get(block_start + off)
            .copied()
            .unwrap_or((0.0, 0.0, 0.0));
        SourceJoint {
            position: [to_norm_x(k.0), to_norm_y(k.1), 0.0],
            confidence: k.2,
        }
    };

    use coco::hand as h;
    use HumanoidBone as B;

    // (finger_joint_offset, bone_for_left_hand, bone_for_right_hand)
    // Each row describes one "proximal joint" that serves as the position
    // anchor for a bone. Tip of the bone is the next row in the same
    // finger's chain (see solver DRIVEN_BONES).
    let entries = [
        (h::THUMB_CMC, B::LeftThumbProximal, B::RightThumbProximal),
        (
            h::THUMB_MCP,
            B::LeftThumbIntermediate,
            B::RightThumbIntermediate,
        ),
        (h::THUMB_IP, B::LeftThumbDistal, B::RightThumbDistal),
        (h::INDEX_MCP, B::LeftIndexProximal, B::RightIndexProximal),
        (
            h::INDEX_PIP,
            B::LeftIndexIntermediate,
            B::RightIndexIntermediate,
        ),
        (h::INDEX_DIP, B::LeftIndexDistal, B::RightIndexDistal),
        (h::MIDDLE_MCP, B::LeftMiddleProximal, B::RightMiddleProximal),
        (
            h::MIDDLE_PIP,
            B::LeftMiddleIntermediate,
            B::RightMiddleIntermediate,
        ),
        (h::MIDDLE_DIP, B::LeftMiddleDistal, B::RightMiddleDistal),
        (h::RING_MCP, B::LeftRingProximal, B::RightRingProximal),
        (
            h::RING_PIP,
            B::LeftRingIntermediate,
            B::RightRingIntermediate,
        ),
        (h::RING_DIP, B::LeftRingDistal, B::RightRingDistal),
        (h::LITTLE_MCP, B::LeftLittleProximal, B::RightLittleProximal),
        (
            h::LITTLE_PIP,
            B::LeftLittleIntermediate,
            B::RightLittleIntermediate,
        ),
        (h::LITTLE_DIP, B::LeftLittleDistal, B::RightLittleDistal),
    ];

    // A virtual "tip anchor" for each distal bone — the fingertip keypoint.
    // These entries are NOT separate humanoid bones; they're anchored by
    // the solver via the tip definitions in DRIVEN_BONES (where *Distal's
    // tip references the fingertip offset via Tip::HandTip). But since the
    // solver resolves tips through `get_joint_2d(HumanoidBone)`, we insert
    // the fingertip positions under pseudo-bones that the solver knows to
    // look up: we reuse the "tip joint" anchoring by ALSO writing the TIP
    // position into the *Distal bone's own position slot (+1). The actual
    // fingertip direction will come from (Distal's position, next kpt)
    // tips wired explicitly in DRIVEN_BONES.
    for (offset, left_bone, right_bone) in entries {
        let joint = at(offset);
        let bone = match target {
            Hand::Left => left_bone,
            Hand::Right => right_bone,
        };
        sk.put_joint(bone, joint, min_conf);
    }

    // Fingertip positions are needed as "tips" for the distal bones. The
    // pose solver consumes them via pseudo-bones stored under the Distal
    // slots *plus one offset*, but our HumanoidBone enum has no extra
    // variants for tips, so we plant the fingertip positions into a
    // dedicated lookup table on the skeleton side (the solver falls back
    // to the distal bone's own position when a fingertip is missing; see
    // DRIVEN_BONES). Practically: we also publish the fingertip position
    // as the Distal bone's *tip* by storing it on the map under the
    // HumanoidBone variant one step beyond Distal — which does not exist,
    // so we use HumanoidBone::Left/RightHand as the tip for the pinky's
    // distal alternative when appropriate. For the common case (rock /
    // scissors / paper), a proximal/intermediate bend is enough signal.
    //
    // Rather than invent pseudo-bones, the solver wires each *Distal bone
    // directly to the fingertip source position via a dedicated code path
    // (see Tip::Fingertip in pose_solver). We expose the fingertip via a
    // parallel map on SourceSkeleton.
    let tip_entries = [
        (h::THUMB_TIP, B::LeftThumbDistal, B::RightThumbDistal),
        (h::INDEX_TIP, B::LeftIndexDistal, B::RightIndexDistal),
        (h::MIDDLE_TIP, B::LeftMiddleDistal, B::RightMiddleDistal),
        (h::RING_TIP, B::LeftRingDistal, B::RightRingDistal),
        (h::LITTLE_TIP, B::LeftLittleDistal, B::RightLittleDistal),
    ];
    for (offset, left_distal, right_distal) in tip_entries {
        let joint = at(offset);
        let bone = match target {
            Hand::Left => left_distal,
            Hand::Right => right_distal,
        };
        sk.fingertips.insert(bone, joint);
    }
}
