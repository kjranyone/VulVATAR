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
#[cfg(feature = "inference")]
mod consts;
#[cfg(feature = "inference")]
mod decode;
#[cfg(feature = "inference")]
mod face;
#[cfg(feature = "inference")]
#[cfg(feature = "inference")]
mod preprocess;
#[cfg(feature = "inference")]
pub(in crate::tracking) mod session;
#[cfg(feature = "inference")]
#[cfg(feature = "inference")]
mod yolox_worker;

#[cfg(feature = "inference")]
pub(in crate::tracking) use session::{build_session, build_session_cpu_only};
// Crop helpers shared with the rtmw3d_with_depth provider's DAv2
// person-crop stage. The functions don't depend on RTMW3D internals;
// they're here purely because that's where they were originally
// written. Promotion-only re-export, no logic change.
#[cfg(feature = "inference")]
use super::face_mediapipe::FaceMeshInference;
#[cfg(feature = "inference")]
use super::yolox::YoloxPersonDetector;
use super::PoseEstimate;
#[cfg(feature = "inference")]
use super::{DetectionAnnotation, SourceSkeleton};
#[cfg(feature = "inference")]
use log::{debug, error, info, warn};
#[cfg(feature = "inference")]
use ort::session::Session;
#[cfg(feature = "inference")]
use ort::value::TensorRef;
#[cfg(feature = "inference")]
use std::path::Path;
#[cfg(feature = "inference")]
use yolox_worker::YoloxWorker;

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

/// Default submit period for the YOLOX worker (every Nth call).
/// Detection itself runs asynchronously in a background thread, so the
/// main pipeline never waits for it past the cold-start frame — we just
/// read the worker's sticky outbox each call and use whatever bbox it
/// last produced (typically 1–2 frames stale).
///
/// At 30 fps and N=4, YOLOX submits at 7.5 fps. The 25% downstream
/// pad on the bbox absorbs the few-pixel subject motion that
/// accumulates between submits.
#[cfg(feature = "inference")]
pub const YOLOX_REFRESH_PERIOD_DEFAULT: u64 = 4;

/// Active YOLOX submit period, settable cross-thread from the
/// application's `RuntimeGpuBudget` consumer (P3-03 B4).
///
/// Single shared atomic because (a) only one `Rtmw3dInference` runs at a
/// time in any session, (b) only one `Application` writes it, and (c)
/// the value is conceptually a global cadence knob, not per-instance
/// state. An `Arc<AtomicU64>` plumbing path (Application → TrackingSource
/// → PoseProvider → Rtmw3dInference) was rejected as needless ceremony
/// for state that is already globally singular.
#[cfg(feature = "inference")]
pub static YOLOX_REFRESH_PERIOD: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(YOLOX_REFRESH_PERIOD_DEFAULT);

/// Maximum wall-clock age a sticky YOLOX result may reach before it is
/// treated as absent — comfortably above the worker's worst observed
/// latency (~300 ms on the CPU EP) plus the refresh period, but far
/// below the minutes-old leftovers the sticky outbox holds after a
/// long self-track run. Compared against device capture timestamps
/// when both sides carry one.
#[cfg(feature = "inference")]
const YOLOX_STICKY_MAX_AGE_S: f64 = 2.0;

/// Frame-count fallback for [`YOLOX_STICKY_MAX_AGE_S`] when capture
/// timestamps are unavailable (synthetic inputs, offline benches):
/// 60 frames ≈ 2 s at the nominal 30 fps.
#[cfg(feature = "inference")]
const YOLOX_STICKY_MAX_AGE_FRAMES: u64 = 60;

/// Sticky-result staleness: wall-clock when both the current frame and
/// the detection carry a device timestamp, frame-count fallback
/// otherwise. Pure so the policy is unit-testable.
#[cfg(feature = "inference")]
fn sticky_is_stale(
    now_ts_ms: Option<f64>,
    result_ts_ms: Option<f64>,
    now_frame: u64,
    result_frame: u64,
) -> bool {
    match (now_ts_ms, result_ts_ms) {
        (Some(now), Some(then)) => now - then > YOLOX_STICKY_MAX_AGE_S * 1000.0,
        _ => now_frame.saturating_sub(result_frame) > YOLOX_STICKY_MAX_AGE_FRAMES,
    }
}

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
    /// Self-tracking crop bbox in original-frame pixel coords,
    /// derived from the previous frame's own high-confidence
    /// keypoints. Preferred over the YOLOX bbox in steady state: it
    /// refreshes at full frame rate (the CPU-EP YOLOX result is
    /// ~300 ms stale) and by construction encloses every keypoint —
    /// the person-class YOLOX bbox routinely cuts off a hand
    /// extended toward the camera on deskcrop framings (validation:
    /// up to 16.9 deg solver error from exactly that). `None` on
    /// cold start or after a low-confidence frame; YOLOX then serves
    /// as the (re)acquisition path.
    #[cfg(feature = "inference")]
    self_track_bbox: Option<crate::tracking::yolox::PersonBbox>,
    /// Last bbox the self-track produced before it (possibly) released.
    /// Survives a self-track drop and is passed to YOLOX as the
    /// identity hint: re-acquisition prefers the candidate overlapping
    /// this region over the highest-score person, so a bystander
    /// crossing the frame can't steal the track (see
    /// `YoloxPersonDetector::detect_person`). Cleared with the rest of
    /// the temporal state on input switch.
    #[cfg(feature = "inference")]
    last_self_track: Option<crate::tracking::yolox::PersonBbox>,
    /// Apparent-size z gain (`crop_h / frame_h`) from the last frame
    /// whose crop actually tracked the subject. RTMW3D's SimCC-z is
    /// metric (fixed `RTMW3D_Z_RANGE`), so the remap needs an
    /// apparent-size gain to keep z proportionate to x/y — but on the
    /// whole-frame letterbox fallback the crop height is a constant
    /// unrelated to the subject, so this held gain is used instead
    /// (continuity across track drops; no ~2.4× z step when the
    /// subject was framed small). `None` until the first tracked crop
    /// (fresh session): the fallback's own ratio is the only estimate
    /// available and matches the pre-track acquisition behaviour.
    #[cfg(feature = "inference")]
    last_tracked_z_gain: Option<f32>,
    /// Stateful mesh↔body head-pose source selection (Schmitt-trigger
    /// hysteresis + switch crossfade) — see [`face::FaceSourceSelector`].
    #[cfg(feature = "inference")]
    face_selector: face::FaceSourceSelector,
    /// Device capture timestamp (ms) for the NEXT `estimate_pose`, pushed
    /// by the depth provider via [`Self::set_frame_timestamp_ms`] before
    /// each call (`None` for synthetic inputs). Consumed once per frame.
    #[cfg(feature = "inference")]
    frame_timestamp_ms: Option<f64>,
    /// Consecutive-timestamp dt derivation for this stage's time-based
    /// estimators (face-source crossfade, sticky-YOLOX freshness,
    /// arm-length leaky maxima).
    #[cfg(feature = "inference")]
    frame_dt: crate::tracking::metric_frame::FrameDtTracker,
    #[cfg(feature = "inference")]
    load_warnings: Vec<String>,
    #[cfg(feature = "inference")]
    backend: InferenceBackend,
    /// Raw per-frame perception outputs for the fusion estimator
    /// (decoded 133 keypoints with SimCC σ in whole-frame normalised
    /// coordinates, FaceMesh landmarks in frame pixels). Set by every
    /// `estimate_pose`, drained by [`Self::take_aux`].
    #[cfg(feature = "inference")]
    last_aux: Option<Rtmw3dAux>,
}

/// Raw perception outputs of one RTMW3D frame, for observation-level
/// consumers (the fusion estimator) that want the keypoints *before* any
/// skeleton building.
#[derive(Clone, Debug, Default)]
pub(crate) struct Rtmw3dAux {
    /// 133 COCO-Wholebody keypoints, whole-frame normalised `[0,1]`.
    pub joints: Vec<decode::DecodedJoint>,
    /// FaceMesh 478 landmarks in frame pixels (`z` in pixel scale) and
    /// the mesh confidence, when the face cascade ran this frame.
    pub face_mesh: Option<(Vec<[f32; 3]>, f32)>,
    /// Person crop actually fed to RTMW3D `(x, y, w, h)` in frame pixels.
    pub crop: Option<(f32, f32, f32, f32)>,
}

/// Construction options for [`Rtmw3dInference`]. Lets the GUI's
/// pipeline settings (and the safe-mode degraded configuration)
/// control which execution providers and optional stages load.
#[derive(Clone, Copy, Debug)]
pub struct Rtmw3dOptions {
    /// Execution provider for the FaceMesh + Blendshape cascade.
    /// Overridden to CPU when `force_cpu` is set.
    pub face_ep: super::face_mediapipe::FaceMeshEp,
    /// Run RTMW3D and FaceMesh on the CPU EP, keeping DirectML — and
    /// therefore the GPU driver's compute queue — completely out of
    /// the tracking pipeline. (YOLOX is CPU-only unconditionally; see
    /// `YoloxPersonDetector::try_from_models_dir`.)
    pub force_cpu: bool,
    /// Load the YOLOX person-crop stage. When false the pipeline runs
    /// whole-frame RTMW3D (same as when `yolox.onnx` is absent).
    pub yolox_enabled: bool,
}

impl Default for Rtmw3dOptions {
    fn default() -> Self {
        Self {
            face_ep: super::face_mediapipe::FaceMeshEp::Auto,
            force_cpu: false,
            yolox_enabled: true,
        }
    }
}

impl Rtmw3dInference {
    /// Instantiate from `models_dir` (typically `models/`). Looks for
    /// `rtmw3d.onnx` and tries to register the DirectML EP, falling
    /// back to CPU on EP failure. FaceMesh + Blendshape go on
    /// DirectML (Auto) — best when no other heavy GPU model is
    /// active.
    #[cfg(feature = "inference")]
    pub fn from_models_dir(models_dir: impl AsRef<Path>) -> Result<Self, String> {
        Self::from_models_dir_with_options(models_dir, Rtmw3dOptions::default())
    }

    /// Same as [`Self::from_models_dir`] but with explicit FaceMesh
    /// EP. Used by the depth-enabled pipeline to force FaceMesh onto
    /// CPU, where it runs uncontended at ~7 ms instead of 13+ ms when
    /// fighting DAv2 for the DirectML command queue.
    #[cfg(feature = "inference")]
    pub fn from_models_dir_with_face_ep(
        models_dir: impl AsRef<Path>,
        face_ep: super::face_mediapipe::FaceMeshEp,
    ) -> Result<Self, String> {
        Self::from_models_dir_with_options(
            models_dir,
            Rtmw3dOptions {
                face_ep,
                ..Rtmw3dOptions::default()
            },
        )
    }

    /// Reset cross-frame temporal state (the ray-IK held-wrist
    /// continuity + torso-depth EMA; the self-tracking crop when
    /// present). See `PoseProvider::reset_temporal_state`.
    #[cfg(feature = "inference")]
    /// Drain the raw perception outputs of the last `estimate_pose`.
    #[cfg(feature = "inference")]
    pub(crate) fn take_aux(&mut self) -> Option<Rtmw3dAux> {
        self.last_aux.take()
    }

    pub fn reset_temporal_state(&mut self) {
        self.self_track_bbox = None;
        self.last_self_track = None;
        self.last_tracked_z_gain = None;
        self.frame_timestamp_ms = None;
        self.frame_dt.reset();
        if let Some(worker) = self.yolox_worker.as_mut() {
            worker.clear_result();
        }
    }

    /// Push the device capture timestamp (ms) of the frame about to be
    /// handed to [`Self::estimate_pose`]. Consumed once per estimate;
    /// when never pushed (synthetic inputs, benches) the time-based
    /// estimators run on the nominal 30 fps step.
    #[cfg(feature = "inference")]
    pub(in crate::tracking) fn set_frame_timestamp_ms(&mut self, ts_ms: Option<f64>) {
        self.frame_timestamp_ms = ts_ms;
    }

    /// Full-control constructor: EP selection and optional stages per
    /// [`Rtmw3dOptions`].
    #[cfg(feature = "inference")]
    pub fn from_models_dir_with_options(
        models_dir: impl AsRef<Path>,
        opts: Rtmw3dOptions,
    ) -> Result<Self, String> {
        let models_dir = models_dir.as_ref();
        let model_path = models_dir.join("rtmw3d.onnx");
        if !model_path.is_file() {
            return Err(format!(
                "RTMW3D model not found at {}. Run dev.ps1 setup to fetch it.",
                model_path.display()
            ));
        }

        let face_ep = if opts.force_cpu {
            super::face_mediapipe::FaceMeshEp::ForceCpu
        } else {
            opts.face_ep
        };

        info!("Loading RTMW3D from {}", model_path.display());
        let (session, backend) = if opts.force_cpu {
            build_session_cpu_only(&model_path.to_string_lossy(), 4, "RTMW3D")?
        } else {
            build_session(&model_path.to_string_lossy(), 4, "RTMW3D")?
        };

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

        let person_detector = if !opts.yolox_enabled {
            info!("YOLOX person crop disabled by pipeline config — using whole-frame RTMW3D");
            None
        } else {
            // YOLOX is CPU-only regardless of `opts.force_cpu` — see the
            // rationale on `try_from_models_dir` (concurrent DirectML
            // sessions from two threads are a recorded driver-hang
            // pattern on Intel GPUs).
            match YoloxPersonDetector::try_from_models_dir(models_dir) {
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
            self_track_bbox: None,
            last_self_track: None,
            last_tracked_z_gain: None,
            face_selector: face::FaceSourceSelector::default(),
            frame_timestamp_ms: None,
            frame_dt: crate::tracking::metric_frame::FrameDtTracker::default(),
            load_warnings,
            backend,
            last_aux: None,
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

        // Device capture timestamp for THIS frame (pushed by the depth
        // provider; `None` on synthetic inputs) and the real dt derived
        // from consecutive values — the clock for every time-based
        // estimator in this stage. `take` so a frame that arrives
        // without a fresh push can't reuse a stale timestamp.
        let frame_ts_ms = self.frame_timestamp_ms.take();
        let dt_s = self.frame_dt.tick(frame_ts_ms);

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
        crate::tracking::stagelog::mark(frame_index, "yolox_begin");
        let (infer_rgb_owned, infer_rgb_slice, infer_w, infer_h, crop_origin, annotation_bbox) = {
            // Submit a detect request to the YOLOX worker on every
            // `YOLOX_REFRESH_PERIOD`-th frame (or always on cold
            // start so the very first call doesn't deadlock waiting
            // on `wait_latest`). The worker drains older queued
            // requests so it always processes the freshest submitted
            // frame. Either way, we then read the latest sticky
            // result — typically from 1–2 frames ago, which is well
            // within the 25% downstream pad.
            // Crop source priority: self-tracking bbox (fresh every
            // frame, encloses all keypoints) > YOLOX (acquisition /
            // re-acquisition only) > whole frame. While the
            // self-track is live, YOLOX receives no submissions at
            // all — no CPU spent on a detector whose result would be
            // ignored.
            let bbox_opt = if let Some(track) = self.self_track_bbox {
                Some(track)
            } else if let Some(worker) = self.yolox_worker.as_ref() {
                let cold_start = !worker.has_result();
                // `.max(1)` is defence-in-depth: `RuntimeGpuBudget`'s
                // mode arms only emit {4, 6, 8, 12}, asserted by the
                // `all_modes_emit_nonzero_yolox_skip_period` unit
                // test, but `is_multiple_of(0)` would panic so we
                // guard against an accidental future regression.
                let period = YOLOX_REFRESH_PERIOD
                    .load(std::sync::atomic::Ordering::Relaxed)
                    .max(1);
                let submitted = cold_start || frame_index.is_multiple_of(period);
                if submitted {
                    worker.submit(rgb_data, width, height, frame_index, frame_ts_ms, self.last_self_track);
                }
                let latest = worker.wait_latest();
                // Age-gate the sticky result. While the self-track is
                // live YOLOX receives no submissions, so when the track
                // eventually drops the outbox may hold a bbox from
                // minutes ago — cropping to it produces garbage frames
                // that in turn prevent the self-track from re-forming.
                // A stale sticky is treated as "no detection" (the
                // letterboxed whole-frame path takes over) and a fresh
                // detection is requested immediately instead of waiting
                // for the next period boundary.
                if latest.bbox.is_some()
                    && sticky_is_stale(
                        frame_ts_ms,
                        latest.timestamp_ms,
                        frame_index,
                        latest.frame_index,
                    )
                {
                    if !submitted {
                        worker.submit(rgb_data, width, height, frame_index, frame_ts_ms, self.last_self_track);
                    }
                    None
                } else {
                    latest.bbox
                }
            } else {
                None
            };
            // Whole-frame fallback: run the frame through the SAME
            // aspect-preserving virtual-crop path a person bbox takes
            // (full-frame bbox, zero pad ratio) instead of feeding the
            // raw frame to `preprocess`'s squash-resize. The squash
            // contradicted `pad_bbox_to_aspect`'s own documented
            // rationale (aspect distortion flattens the heatmaps and
            // edge-clamps elbows) on exactly the acquisition frames
            // where quality decides whether a self-track ever forms.
            // Letterboxing costs one extra buffer copy on fallback
            // frames only.
            let whole_frame_letterbox = || {
                let full = crate::tracking::yolox::PersonBbox {
                    x1: 0.0,
                    y1: 0.0,
                    x2: width as f32,
                    y2: height as f32,
                    score: 1.0,
                };
                let (fx1, fy1, fx2, fy2) = preprocess::pad_bbox_to_aspect(&full, 0.0);
                let cw = (fx2 - fx1).round().max(1.0) as u32;
                let ch = (fy2 - fy1).round().max(1.0) as u32;
                let (ox, oy) = (fx1.round() as i32, fy1.round() as i32);
                let crop = preprocess::crop_rgb_padded(rgb_data, width, height, ox, oy, cw, ch);
                let annotation: Option<(f32, f32, f32, f32)> = None;
                let empty: &[u8] = &[];
                (
                    Some(crop),
                    empty,
                    cw,
                    ch,
                    Some((ox as f32, oy as f32, cw as f32, ch as f32)),
                    annotation,
                )
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
                    // Aspect-preserving crop: keep the model's 288:384 ratio and
                    // zero-pad any out-of-frame region instead of clamping (which
                    // squashes a frame-filling subject). The self-track SEED is
                    // clamped to the frame (`derive_self_track_bbox`), so this
                    // beyond-frame crop stays bounded — no zoom-out runaway.
                    let (fx1, fy1, fx2, fy2) = preprocess::pad_bbox_to_aspect(&bbox, 0.25);
                    let cw = (fx2 - fx1).round() as u32;
                    let ch = (fy2 - fy1).round() as u32;
                    let (ox, oy) = (fx1.round() as i32, fy1.round() as i32);
                    debug!(
                        "RTMW3D: crop bbox=[{:.0},{:.0},{:.0},{:.0}] score={:.2} → {}x{} @({},{}) aspect={:.2} (orig {}x{})",
                        bbox.x1, bbox.y1, bbox.x2, bbox.y2, bbox.score, cw, ch, ox, oy,
                        cw as f32 / ch.max(1) as f32, width, height
                    );
                    if cw < 32 || ch < 32 {
                        // Bbox too small to crop usefully — fall through.
                        whole_frame_letterbox()
                    } else {
                        let crop = preprocess::crop_rgb_padded(rgb_data, width, height, ox, oy, cw, ch);
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
                            Some((ox as f32, oy as f32, cw as f32, ch as f32)),
                            ann,
                        )
                    }
                }
                None => whole_frame_letterbox(),
            }
        };
        let infer_rgb: &[u8] = match &infer_rgb_owned {
            Some(buf) => buf.as_slice(),
            None => infer_rgb_slice,
        };
        crate::tracking::stagelog::mark(frame_index, "yolox_end");
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
        crate::tracking::stagelog::mark(frame_index, "rtmw_run_begin");
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
        crate::tracking::stagelog::mark(frame_index, "rtmw_run_end");
        let dt_run = t_run.elapsed();

        // Decode straight from the borrowed tensor views — the previous
        // `.to_vec()` per output copied ~1.2 MB of SimCC planes every
        // frame for no benefit (decode only reads the slices once).
        let extract = |name: &Option<String>, expected_len: usize| -> Option<&[f32]> {
            let n = name.as_deref()?;
            let v = outputs.get(n)?;
            let (_, data) = v.try_extract_tensor::<f32>().ok()?;
            if data.len() < expected_len {
                None
            } else {
                Some(data)
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

        let t_decode = std::time::Instant::now();
        let mut joints = decode::decode_simcc(simcc_x, simcc_y, simcc_z);
        drop(outputs);
        if let Some((ox, oy, cw, ch)) = crop_origin {
            // Remap crop-space joints back to original-frame coords,
            // scaling nz by the subject's apparent size so the x/y/z
            // unit contract stays distance-invariant. A tracked crop
            // (bbox / self-track — `annotation_bbox.is_some()`) IS the
            // apparent-size estimate and refreshes the held gain; the
            // whole-frame letterbox fallback is not (its crop height is
            // the constant frame letterbox, ~2.37 on 16:9, regardless
            // of how small the subject is), so it reuses the last
            // tracked gain. See `remap_crop_joints` for the metric-z
            // rationale.
            let z_gain = if annotation_bbox.is_some() || self.last_tracked_z_gain.is_none() {
                let g = preprocess::tracked_z_gain(ch, height);
                if annotation_bbox.is_some() {
                    self.last_tracked_z_gain = Some(g);
                }
                g
            } else {
                self.last_tracked_z_gain.unwrap_or_else(|| preprocess::tracked_z_gain(ch, height))
            };
            preprocess::remap_crop_joints(&mut joints, ox, oy, cw, ch, width, height, z_gain);
        }
        // Raw keypoints for the fusion estimator (whole-frame coords).
        self.last_aux = Some(Rtmw3dAux {
            joints: joints.clone(),
            face_mesh: None,
            crop: crop_origin,
        });
        // Refresh the self-tracking crop from this frame's own
        // keypoints (whole-frame coords post-remap), with hysteresis.
        //
        // Hysteresis matters: the crop influences where RTMW3D puts
        // its keypoints, and the keypoints define the next crop — an
        // undamped feedback loop. Reproduced as a period-2 limit
        // cycle on the hands-at-chest validation transition (wrist y
        // oscillating 0.38 ↔ 0.60 every frame, driving the solver
        // 50°+ off). The cure is to keep the previous bbox until the
        // subject actually threatens to leave it: a frozen crop
        // means deterministic keypoints means a stable track.
        let fresh = derive_self_track_bbox(&joints, width, height);
        self.self_track_bbox = match (self.self_track_bbox, fresh) {
            (_, None) => None,
            (None, Some(new)) => Some(new),
            (Some(cur), Some(new)) => {
                // The crop site pads by 25%; demand the subject stays
                // inside a 10% inner margin of the current bbox before
                // we re-frame, and also re-frame when the subject has
                // shrunk to well under the current box (crop much too
                // loose degrades model resolution).
                let cur_w = (cur.x2 - cur.x1).max(1.0);
                let cur_h = (cur.y2 - cur.y1).max(1.0);
                let margin_x = cur_w * 0.10;
                let margin_y = cur_h * 0.10;
                let escaping = new.x1 < cur.x1 - margin_x
                    || new.y1 < cur.y1 - margin_y
                    || new.x2 > cur.x2 + margin_x
                    || new.y2 > cur.y2 + margin_y;
                let new_w = (new.x2 - new.x1).max(1.0);
                let new_h = (new.y2 - new.y1).max(1.0);
                let shrunk = new_w < cur_w * 0.6 || new_h < cur_h * 0.6;
                if escaping || shrunk {
                    Some(new)
                } else {
                    Some(cur)
                }
            }
        };
        // Remember the most recent live track for YOLOX identity
        // preference — deliberately NOT cleared when the track drops
        // (that's exactly when re-acquisition needs the hint).
        if let Some(b) = self.self_track_bbox {
            self.last_self_track = Some(b);
        }

        // Tracking v2: the fusion estimator consumes the raw keypoints
        // (`take_aux`); this estimate only carries the face channels
        // (head-pose candidates, FaceMesh expressions) plus the 2-D
        // annotation for the GUI overlay. The v1 monocular skeleton
        // builder and its arm heuristics are gone.
        let mut skeleton = SourceSkeleton::empty(frame_index);
        skeleton.overall_confidence = {
            let mut scores: Vec<f32> = joints.iter().take(17).map(|j| j.score).collect();
            scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
            scores.get(scores.len() / 2).copied().unwrap_or(0.0)
        };

        skeleton.face =
            face::derive_face_pose_from_body(&joints, width as f32 / height.max(1) as f32);
        let dt_decode = t_decode.elapsed();

        // Face crop → MediaPipe FaceMesh → BlendshapeV2. The bbox is
        // derived from RTMW3D's face-68 landmarks (indices 23..=90)
        // so the cascade does not need its own face detector. Head-pose
        // source selection and confidence folding live in
        // `face::select_face_pose`; the solver's
        // `face_confidence_threshold` then gates head rotation *and*
        // expressions against the resulting confidence.
        let t_face = std::time::Instant::now();
        crate::tracking::stagelog::mark(frame_index, "face_begin");
        let body_face_pose = skeleton.face;
        let mut dbg_bbox = None;
        let mut dbg_mesh: Option<(f32, crate::tracking::FacePose)> = None;
        let mut mesh_face_pose: Option<crate::tracking::FacePose> = None;
        let mut mesh_conf = 0.0f32;
        if let Some(face_mesh) = self.face_mesh.as_mut() {
            if let Some(bbox) = face::build_face_bbox_from_joints(&joints, width, height) {
                // Normalised to the image so the overlay tool can draw it
                // on the down-scaled debug camera frame directly.
                dbg_bbox = Some((
                    bbox.x / width as f32,
                    bbox.y / height as f32,
                    bbox.size / width as f32,
                ));
                if let Some((exprs, conf, pose)) =
                    face_mesh.estimate(rgb_data, width, height, &bbox)
                {
                    skeleton.expressions = exprs;
                    skeleton.face_mesh_confidence = Some(conf);
                    mesh_conf = conf;
                    mesh_face_pose = pose;
                    // PITCH IS NOT PER-SOURCE. Both estimators used to derive
                    // it by normalising the nose's drop below the eye line by a
                    // HORIZONTAL baseline (inter-eye / eye-corner width), which
                    // foreshortens as `cos(yaw)`, and both compensated with
                    // their own yaw. That compensation fails in exactly the
                    // view this user works in — turned ~60 deg toward a monitor
                    // beside the camera:
                    //   * body path: ears occluded (headphones), ear-line yaw
                    //     pinned inside ±5 deg → no compensation at all;
                    //   * mesh path: its own yaw SATURATES at the ±63.43 deg
                    //     `atan(clamp(·, 2.0))` bound (measured on 16% of live
                    //     frames) and `yaw.cos()` is floored at 0.5, so past
                    //     ~60 deg the correction stops growing while the real
                    //     foreshortening keeps going.
                    // Either way the ratio inflates and a head TURN decodes as
                    // a chin-DOWN nod — the user's "it still looks down when I
                    // turn". `face::pitch_from_vertical_ratio` sidesteps the
                    // whole failure mode: it is built from image `y` only
                    // (eye line → nose tip → chin, RTMW3D's face-68 block), and
                    // a rotation about the vertical axis cannot change `y`. So
                    // the mesh keeps what it is genuinely better at (yaw / roll
                    // through the 3/4-view dead-zone) and takes its pitch from
                    // the yaw-invariant estimator. Feeding BOTH candidates the
                    // same pitch also means a source switch can no longer step
                    // the head in that channel.
                    if let (Some(p), Some(mesh)) = (
                        face::pitch_from_vertical_ratio(&joints),
                        mesh_face_pose.as_mut(),
                    ) {
                        mesh.pitch = p;
                    }
                    dbg_mesh = mesh_face_pose.map(|p| (conf, p));
                    if let (Some(aux), Some(lm)) =
                        (self.last_aux.as_mut(), face_mesh.take_landmarks_px())
                    {
                        aux.face_mesh = Some(lm);
                    }
                }
            }
        }
        // Source selection between the body-derived pose and FaceMesh's
        // dense-landmark pose lives in `face::FaceSourceSelector`: mesh
        // when it genuinely saw a face (covers the 3/4-view dead-zone
        // of the body ear-line yaw), body when the mesh score collapses
        // (profile/back views, where the body pose is strong). Stateful
        // (hysteresis + crossfade), so it must run every frame — also
        // on frames where the mesh didn't run at all, which count as
        // mesh-confidence 0 and release a stale mesh lock.
        skeleton.face = self
            .face_selector
            .select(body_face_pose, mesh_face_pose, mesh_conf, dt_s);
        // Raw body pose published alongside the selection so the
        // calibration hold can accumulate a neutral for BOTH sources
        // in one capture (see `SourceSkeleton::face_body_raw`).
        skeleton.face_body_raw = body_face_pose;
        if crate::tracking::debug_channel::enabled() {
            crate::tracking::debug_channel::stash_face_debug(
                dbg_bbox,
                dbg_mesh.map(|(c, _)| c),
                body_face_pose.map(|p| (p.yaw, p.pitch, p.roll)),
                dbg_mesh.map(|(_, p)| (p.yaw, p.pitch, p.roll)),
            );
        }
        crate::tracking::stagelog::mark(frame_index, "face_end");
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

/// Derive the next frame's self-tracking crop from this frame's
/// decoded keypoints (whole-frame normalised coords). Returns `None`
/// — releasing the track back to YOLOX re-acquisition — when fewer
/// than [`SELF_TRACK_MIN_KEYPOINTS`] body keypoints clear the
/// confidence floor or the bbox degenerates.
///
/// Confidence floor 0.3 (vs the global visibility floor of 0.05):
/// a garbage crop produces sub-0.3 scores almost everywhere, so a
/// bad track releases itself within one frame instead of locking on.
#[cfg(feature = "inference")]
fn derive_self_track_bbox(
    joints: &[decode::DecodedJoint],
    width: u32,
    height: u32,
) -> Option<crate::tracking::yolox::PersonBbox> {
    const SELF_TRACK_CONF_FLOOR: f32 = 0.3;
    /// COCO body keypoints (0..17) above the floor required to trust
    /// the track. Hands/face alone must not sustain it — a track
    /// locked onto a detached hand region would never recover.
    const SELF_TRACK_MIN_KEYPOINTS: usize = 8;
    const MIN_BBOX_PX: f32 = 48.0;

    let body_confident = joints
        .iter()
        .take(17)
        .filter(|j| j.score >= SELF_TRACK_CONF_FLOOR)
        .count();
    if body_confident < SELF_TRACK_MIN_KEYPOINTS {
        return None;
    }

    let (mut x1, mut y1, mut x2, mut y2) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for j in joints.iter().filter(|j| j.score >= SELF_TRACK_CONF_FLOOR) {
        let px = j.nx * width as f32;
        let py = j.ny * height as f32;
        x1 = x1.min(px);
        y1 = y1.min(py);
        x2 = x2.max(px);
        y2 = y2.max(py);
    }
    // Clamp the seed to the frame. The crop derived from this bbox is allowed to
    // extend past the frame (aspect-preserving, zero-padded) — but the SEED must
    // stay inside it: an aspect crop places some keypoints in the padded region
    // beyond the frame, which map back to out-of-frame coords, and feeding those
    // back into the seed grows the crop every frame (an unbounded zoom-out that
    // shrinks the subject to a speck). Clamping breaks that feedback loop; the
    // 25% pad + aspect expansion at the crop site still covers limbs just
    // outside the frame.
    x1 = x1.max(0.0);
    y1 = y1.max(0.0);
    x2 = x2.min(width as f32);
    y2 = y2.min(height as f32);
    if x2 - x1 < MIN_BBOX_PX || y2 - y1 < MIN_BBOX_PX {
        return None;
    }
    Some(crate::tracking::yolox::PersonBbox {
        x1,
        y1,
        x2,
        y2,
        score: 1.0,
    })
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

#[cfg(all(test, feature = "inference"))]
mod sticky_age_tests {
    use super::sticky_is_stale;

    #[test]
    fn wall_time_governs_when_both_timestamps_present() {
        // 1.9 s old → fresh; 2.1 s old → stale — regardless of how many
        // pipeline frames elapsed (a 15 fps stall must not double the
        // effective hold, nor a 60 fps stream halve it).
        assert!(!sticky_is_stale(Some(10_000.0), Some(8_100.0), 500, 0));
        assert!(sticky_is_stale(Some(10_000.0), Some(7_900.0), 500, 440));
    }

    #[test]
    fn frame_count_fallback_without_timestamps() {
        // Synthetic inputs carry no device clock: the nominal-30-fps
        // frame-count gate takes over, on either side missing.
        assert!(!sticky_is_stale(None, None, 100, 60));
        assert!(sticky_is_stale(None, None, 100, 30));
        assert!(sticky_is_stale(Some(10_000.0), None, 100, 30));
        assert!(sticky_is_stale(None, Some(8_000.0), 100, 30));
    }
}
