#![allow(dead_code)]
use log::{error, info, warn};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use crate::t;

#[cfg(feature = "webcam")]
mod webcam;

mod pose_estimation;

pub mod inference;
pub mod mediapipe;
pub mod source_skeleton;

mod face_expression;

pub use source_skeleton::{FacePose, SourceExpression, SourceJoint, SourceSkeleton};

/// Single source of truth for the per-keypoint confidence threshold default.
///
/// Tracker output exposes a `confidence ∈ [0, 1]` per joint and per face
/// pose; the solver discards anything below this floor. The value is a
/// trade-off:
///
/// * **Lower** (toward 0.1) admits more noisy keypoints, which can drive
///   the avatar from low-quality input but produces more twitchy motion.
/// * **Higher** (toward 0.5+) restricts to confident detections only —
///   smoother motion, but bones can freeze when the camera/lighting is
///   poor.
///
/// `0.3` is the project-wide canonical mid-point. The runtime default
/// ([`TrackingSmoothingParams::default`] / [`SolverParams::default`]),
/// the first-launch GUI value, the "Streaming" preset, and the
/// persistence fallback all reference this constant. Two presets
/// intentionally diverge — see their constructors for rationale.
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.3;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TrackingSourceId(pub u64);

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum TrackingImageFormat {
    Rgb8,
    Bgr8,
    #[default]
    Rgba8,
    Bgra8,
    Nv12,
    Yuyv,
}

#[derive(Clone, Debug)]
pub struct TrackingFrame {
    pub source_id: TrackingSourceId,
    pub capture_timestamp: u64,
    pub frame_size: [u32; 2],
    pub image_format: TrackingImageFormat,
    pub body_confidence: f32,
    pub face_confidence: f32,
}

/// Smoothing/threshold params that the GUI exposes as sliders. Consumed by
/// [`crate::avatar::pose_solver::solve_avatar_pose`] via
/// [`SolverParams`](crate::avatar::pose_solver::SolverParams).
#[derive(Clone, Debug)]
pub struct TrackingSmoothingParams {
    /// Per-frame blend factor toward the new rotation. Maps directly to
    /// `SolverParams::rotation_blend`.
    pub rotation_blend: f32,
    /// Per-frame blend factor toward new expression weights.
    pub expression_blend: f32,
    /// Minimum keypoint confidence for a joint to drive a bone.
    pub joint_confidence_threshold: f32,
    /// Minimum face-pose confidence for the head to react.
    pub face_confidence_threshold: f32,
    pub stale_timeout_nanos: u64,
}

impl Default for TrackingSmoothingParams {
    fn default() -> Self {
        Self {
            rotation_blend: 0.7,
            expression_blend: 0.8,
            joint_confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
            face_confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
            stale_timeout_nanos: 200_000_000,
        }
    }
}

/// Per-session calibration. The only field that survives the SourceSkeleton
/// rewrite is `neutral_face_pose`: operators calibrate by facing the camera
/// squarely for a moment, and every subsequent face pose gets this reading
/// subtracted so that "neutral" really is `yaw = pitch = roll = 0`.
#[derive(Clone, Debug, Default)]
pub struct TrackingCalibration {
    pub neutral_face_pose: FacePose,
}

impl TrackingCalibration {
    /// Subtract the stored neutral face pose from the live sample so the
    /// solver receives deltas relative to calibration, not absolute camera
    /// angles.
    pub fn apply_calibration(&self, sample: &mut SourceSkeleton) {
        if let Some(ref mut face) = sample.face {
            face.yaw -= self.neutral_face_pose.yaw;
            face.pitch -= self.neutral_face_pose.pitch;
            face.roll -= self.neutral_face_pose.roll;
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TrackingErrorLevel {
    Warning,
    Blocking,
}

// ---------------------------------------------------------------------------
// TrackingMailbox
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct TrackingMailbox {
    shared: Arc<Mutex<TrackingMailboxInner>>,
    stale_timeout_nanos: u64,
}

struct TrackingMailboxInner {
    latest_pose: Option<SourceSkeleton>,
    latest_frame: Option<WebcamFrame>,
    latest_annotation: Option<DetectionAnnotation>,
    sequence: u64,
    last_update_nanos: u64,
    pending_error: Option<(String, TrackingErrorLevel)>,
    /// One-line label of the inference backend in use (e.g. "DirectML",
    /// "CPU", "CPU (DirectML unavailable: ...)"). `None` when no
    /// inference engine is loaded — e.g. synthetic mode, or before the
    /// worker has finished init.
    inference_backend_label: Option<String>,
}

fn now_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Atomically captured snapshot of the tracking mailbox state.
#[derive(Clone, Debug, Default)]
pub struct MailboxSnapshot {
    pub pose: Option<SourceSkeleton>,
    pub frame: Option<WebcamFrame>,
    pub annotation: Option<DetectionAnnotation>,
    pub sequence: u64,
}

impl TrackingMailbox {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Mutex::new(TrackingMailboxInner {
                latest_pose: None,
                latest_frame: None,
                latest_annotation: None,
                sequence: 0,
                last_update_nanos: 0,
                pending_error: None,
                inference_backend_label: None,
            })),
            stale_timeout_nanos: TrackingSmoothingParams::default().stale_timeout_nanos,
        }
    }
}

impl Default for TrackingMailbox {
    fn default() -> Self {
        Self::new()
    }
}

impl TrackingMailbox {
    /// Thread-safe publish: takes &self, locks the mutex internally.
    pub fn publish(&self, pose: SourceSkeleton) {
        let mut inner = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        inner.latest_pose = Some(pose);
        inner.sequence += 1;
        inner.last_update_nanos = now_nanos();
    }

    /// Publish a full estimation result including frame and annotations.
    pub fn publish_estimate(&self, estimate: PoseEstimate, frame: Option<WebcamFrame>) {
        let mut inner = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        inner.latest_pose = Some(estimate.skeleton);
        inner.latest_annotation = Some(estimate.annotation);
        inner.latest_frame = frame;
        inner.sequence += 1;
        inner.last_update_nanos = now_nanos();
    }

    /// Atomically snapshot the entire mailbox state under a single lock.
    pub fn snapshot(&self) -> MailboxSnapshot {
        let inner = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        MailboxSnapshot {
            pose: inner.latest_pose.clone(),
            frame: inner.latest_frame.clone(),
            annotation: inner.latest_annotation.clone(),
            sequence: inner.sequence,
        }
    }

    /// Report a non-fatal error from the worker thread (shown as GUI toast).
    pub fn report_error(&self, msg: impl Into<String>, level: TrackingErrorLevel) {
        let mut inner = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        inner.pending_error = Some((msg.into(), level));
    }

    /// Drain the latest pending error (returns it only once).
    pub fn drain_error(&self) -> Option<(String, TrackingErrorLevel)> {
        let mut inner = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        inner.pending_error.take()
    }

    /// Set the inference-backend label. Called once by the worker after
    /// `CigPoseInference` finishes loading its model. `None` resets it
    /// (e.g. when tracking stops or the engine is destroyed).
    pub fn set_inference_backend_label(&self, label: Option<String>) {
        let mut inner = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        inner.inference_backend_label = label;
    }

    /// Read the inference-backend label. Cheap clone so the GUI thread
    /// can render without holding the lock.
    pub fn inference_backend_label(&self) -> Option<String> {
        let inner = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        inner.inference_backend_label.clone()
    }

    /// Thread-safe read: takes &self, locks the mutex, clones the pose.
    pub fn latest_pose(&self) -> Option<SourceSkeleton> {
        let inner = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        inner.latest_pose.clone()
    }

    /// Read the latest webcam frame (downscaled for GUI display).
    pub fn latest_frame(&self) -> Option<WebcamFrame> {
        let inner = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        inner.latest_frame.clone()
    }

    /// Read the latest 2D detection annotation.
    pub fn latest_annotation(&self) -> Option<DetectionAnnotation> {
        let inner = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        inner.latest_annotation.clone()
    }

    pub fn sequence(&self) -> u64 {
        let inner = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        inner.sequence
    }

    /// Returns true if the latest sample is older than `stale_timeout_nanos`,
    /// or if no sample has ever been published.
    pub fn is_stale(&self) -> bool {
        let inner = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        if inner.last_update_nanos == 0 {
            return true;
        }
        let elapsed = now_nanos().saturating_sub(inner.last_update_nanos);
        elapsed > self.stale_timeout_nanos
    }
}

// ---------------------------------------------------------------------------
// Webcam frame & detection annotation (for GUI PIP wipe display)
// ---------------------------------------------------------------------------

/// A downscaled webcam frame for the GUI camera-preview overlay.
#[derive(Clone, Debug)]
pub struct WebcamFrame {
    pub rgb_data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// 2D detection annotation overlaid on the camera preview.
#[derive(Clone, Debug, Default)]
pub struct DetectionAnnotation {
    /// Keypoints as (x, y, confidence) in normalised [0, 1] image coords.
    pub keypoints: Vec<(f32, f32, f32)>,
    /// Skeleton line connections as pairs of keypoint indices.
    pub skeleton: Vec<(usize, usize)>,
    /// Bounding box (min_x, min_y, max_x, max_y) in normalised coords.
    pub bounding_box: Option<(f32, f32, f32, f32)>,
}

/// Combined output of a pose estimation pass: source skeleton + 2D annotation.
pub struct PoseEstimate {
    pub skeleton: SourceSkeleton,
    pub annotation: DetectionAnnotation,
}

// ---------------------------------------------------------------------------
// CameraBackend
// ---------------------------------------------------------------------------

/// Selects the capture backend for the tracking source.
#[derive(Clone, Debug, Default)]
pub enum CameraBackend {
    /// Generate synthetic tracking data (no real camera needed).
    #[default]
    Synthetic,
    /// Capture from a real webcam at the given device index.
    /// Only available when compiled with the `webcam` cargo feature.
    #[cfg(feature = "webcam")]
    Webcam { camera_index: usize },
}

impl CameraBackend {
    /// Short label for display in the GUI status bar.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Synthetic => "Synthetic",
            #[cfg(feature = "webcam")]
            Self::Webcam { .. } => "Webcam",
        }
    }
}

// ---------------------------------------------------------------------------
// Camera information (available with or without the feature, but only
// populated when the webcam feature is enabled)
// ---------------------------------------------------------------------------

/// Describes a camera device discovered on the system.
#[derive(Clone, Debug)]
pub struct CameraInfo {
    pub index: usize,
    pub name: String,
}

/// Enumerate available cameras on the system.
///
/// Returns an empty list when compiled without the `webcam` feature.
pub fn list_cameras() -> Vec<CameraInfo> {
    #[cfg(feature = "webcam")]
    {
        webcam::list_cameras_impl()
    }
    #[cfg(not(feature = "webcam"))]
    {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// TrackingSource
// ---------------------------------------------------------------------------

pub struct TrackingSource {
    source_id: TrackingSourceId,
    frame_index: u64,
    mailbox: TrackingMailbox,
    backend: CameraBackend,
}

impl TrackingSource {
    pub fn new() -> Self {
        Self {
            source_id: TrackingSourceId(1),
            frame_index: 0,
            mailbox: TrackingMailbox::new(),
            backend: CameraBackend::default(),
        }
    }
}

impl Default for TrackingSource {
    fn default() -> Self {
        Self::new()
    }
}

impl TrackingSource {
    /// Create a tracking source with a specific backend.
    pub fn with_backend(backend: CameraBackend) -> Self {
        Self {
            backend,
            ..Self::new()
        }
    }

    /// Switch the active backend. Does not affect any running worker thread;
    /// stop and restart the worker after calling this.
    pub fn set_backend(&mut self, backend: CameraBackend) {
        self.backend = backend;
    }

    /// Select a webcam by device index. Convenience wrapper around `set_backend`.
    #[cfg(feature = "webcam")]
    pub fn select_camera(&mut self, index: usize) {
        self.set_backend(CameraBackend::Webcam {
            camera_index: index,
        });
    }

    pub fn mailbox(&self) -> &TrackingMailbox {
        &self.mailbox
    }

    /// Return a clone of the mailbox for sharing with a worker thread.
    pub fn shared_mailbox(&self) -> TrackingMailbox {
        self.mailbox.clone()
    }
}

// ---------------------------------------------------------------------------
// TrackingWorker
// ---------------------------------------------------------------------------

/// Background worker that continuously captures tracking frames and publishes
/// the latest `SourceSkeleton` to a shared `TrackingMailbox`.
///
/// The app thread reads from the same mailbox using latest-sample semantics.
pub struct TrackingWorker {
    handle: Option<JoinHandle<()>>,
    running: Arc<AtomicBool>,
    ready: Arc<AtomicBool>,
    mailbox: TrackingMailbox,
    backend: CameraBackend,
}

impl TrackingWorker {
    /// Create a new tracking worker bound to the given mailbox.
    ///
    /// The worker is not started until `start()` is called.
    pub fn new(mailbox: TrackingMailbox) -> Self {
        Self {
            handle: None,
            running: Arc::new(AtomicBool::new(false)),
            ready: Arc::new(AtomicBool::new(false)),
            mailbox,
            backend: CameraBackend::default(),
        }
    }

    /// Return a reference to the shared mailbox.
    pub fn mailbox(&self) -> &TrackingMailbox {
        &self.mailbox
    }

    /// Returns `true` if the worker thread is currently running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Returns `true` once the worker has finished initialisation (camera
    /// opened or fallback engaged) and is actively producing frames.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }

    /// Returns the backend this worker was started with.
    pub fn active_backend(&self) -> &CameraBackend {
        &self.backend
    }

    /// Spawn the tracking worker thread with the given backend.
    ///
    /// The thread loops at approximately `fps` frames per second, capturing
    /// frames from the selected backend and publishing poses to the shared
    /// mailbox. If the thread is already running this is a no-op.
    #[allow(dead_code)]
    pub fn start(&mut self, backend: CameraBackend) {
        self.start_with_params(backend, 640, 480, 30);
    }

    /// Like [`Self::start`] but allows specifying the webcam resolution and fps.
    pub fn start_with_params(&mut self, backend: CameraBackend, width: u32, height: u32, fps: u32) {
        self.start_with_params_full(backend, width, height, fps, false);
    }

    /// Same as [`Self::start_with_params`] but allows opting into a
    /// wholebody pose model (legs/feet keypoints) at the cost of upper-
    /// body fidelity. Only consulted at start time — toggling the flag
    /// while the worker is running is a no-op.
    pub fn start_with_params_full(
        &mut self,
        backend: CameraBackend,
        width: u32,
        height: u32,
        fps: u32,
        prefer_lower_body: bool,
    ) {
        if self.handle.is_some() {
            return;
        }

        self.backend = backend.clone();
        self.running = Arc::new(AtomicBool::new(true));
        self.ready = Arc::new(AtomicBool::new(false));
        let running = Arc::clone(&self.running);
        let ready = Arc::clone(&self.ready);
        let mailbox = self.mailbox.clone();
        let target_fps: u64 = fps.max(1) as u64;
        let frame_interval = Duration::from_micros(1_000_000 / target_fps);

        let handle = thread::Builder::new()
            .name("tracking-worker".into())
            .spawn(move || {
                Self::worker_loop(
                    backend,
                    mailbox,
                    running,
                    ready,
                    frame_interval,
                    width,
                    height,
                    fps,
                    prefer_lower_body,
                );
            })
            .expect("failed to spawn tracking-worker thread");

        self.handle = Some(handle);
    }

    /// Signal the worker to stop and join the thread.
    ///
    /// Waits up to 3 seconds for the worker thread to exit. If it hasn't
    /// exited by then (e.g. `grab_frame()` is blocking indefinitely), the
    /// thread is detached so the caller is not blocked. Safe to call
    /// multiple times.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let deadline = std::time::Instant::now() + Duration::from_secs(3);
            loop {
                if handle.is_finished() {
                    if let Err(e) = handle.join() {
                        error!("tracking-worker: thread panicked: {:?}", e);
                    }
                    return;
                }
                if std::time::Instant::now() >= deadline {
                    warn!("tracking-worker: thread did not exit within timeout, detaching");
                    std::mem::drop(handle);
                    return;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }

    // -- internal -----------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn worker_loop(
        backend: CameraBackend,
        mailbox: TrackingMailbox,
        running: Arc<AtomicBool>,
        ready: Arc<AtomicBool>,
        frame_interval: Duration,
        width: u32,
        height: u32,
        fps: u32,
        prefer_lower_body: bool,
    ) {
        match backend {
            CameraBackend::Synthetic => {
                let _ = (width, height, fps, prefer_lower_body);
                ready.store(true, Ordering::SeqCst);
                Self::run_synthetic(&mailbox, &running, frame_interval);
            }
            #[cfg(feature = "webcam")]
            CameraBackend::Webcam { camera_index } => {
                Self::run_webcam(
                    camera_index,
                    &mailbox,
                    &running,
                    &ready,
                    frame_interval,
                    width,
                    height,
                    fps,
                    prefer_lower_body,
                );
            }
        }
        // Ensure running is cleared when the thread exits for any reason.
        running.store(false, Ordering::SeqCst);
        // Clear the backend label so the inspector hides the row instead
        // of showing a stale value from the previous session.
        mailbox.set_inference_backend_label(None);
    }

    /// Generate synthetic tracking data in a loop.
    fn run_synthetic(mailbox: &TrackingMailbox, running: &AtomicBool, interval: Duration) {
        info!("tracking-worker: started (synthetic, ~30 fps)");
        let mut frame_index: u64 = 0;

        while running.load(Ordering::SeqCst) {
            let loop_start = std::time::Instant::now();

            let t = frame_index as f32 * 0.05;
            let pose = synthetic_pose(frame_index, t);
            mailbox.publish(pose);
            frame_index += 1;

            let elapsed = loop_start.elapsed();
            if elapsed < interval {
                thread::sleep(interval - elapsed);
            }
        }

        info!("tracking-worker: stopped");
    }

    /// Capture from a real webcam and run simplified pose estimation.
    #[cfg(feature = "webcam")]
    #[allow(clippy::too_many_arguments)]
    fn run_webcam(
        camera_index: usize,
        mailbox: &TrackingMailbox,
        running: &AtomicBool,
        ready: &AtomicBool,
        interval: Duration,
        width: u32,
        height: u32,
        fps: u32,
        prefer_lower_body: bool,
    ) {
        info!(
            "tracking-worker: opening webcam {} ({}x{} @ {} fps)",
            camera_index, width, height, fps
        );

        let mut capture = match webcam::WebcamCapture::open(camera_index, width, height, fps) {
            Ok(c) => c,
            Err(e) => {
                error!(
                    "tracking-worker: failed to open camera {}: {}",
                    camera_index, e
                );
                mailbox.report_error(t!("tracking.error_camera_open", index = camera_index, error = e.to_string()), TrackingErrorLevel::Blocking);
                warn!("tracking-worker: falling back to synthetic backend");
                ready.store(true, Ordering::SeqCst);
                Self::run_synthetic(mailbox, running, interval);
                return;
            }
        };

        // Initialize the MediaPipe Pose Landmarker. The legacy
        // `prefer_lower_body` flag is no longer meaningful because
        // BlazePose always tracks the full body — kept in the function
        // signature for now to avoid touching every caller, but the
        // value is ignored.
        let _ = prefer_lower_body;
        #[cfg(feature = "inference")]
        let mut pose_estimator = match mediapipe::MediaPipeInference::from_models_dir("models") {
            Ok(mut estimator) => {
                let warnings = estimator.take_load_warnings();
                if !warnings.is_empty() {
                    mailbox.report_error(t!("tracking.error_model_warning", warnings = warnings.join("; ")), TrackingErrorLevel::Warning);
                }
                mailbox.set_inference_backend_label(Some(estimator.backend().label()));
                Some(estimator)
            }
            Err(e) => {
                error!("tracking-worker: inference disabled: {}", e);
                mailbox.report_error(t!("tracking.error_model_unavailable", error = e.to_string()), TrackingErrorLevel::Blocking);
                None
            }
        };
        #[cfg(not(feature = "inference"))]
        let mut pose_estimator: Option<mediapipe::MediaPipeInference> = None;

        info!("tracking-worker: webcam opened successfully");
        ready.store(true, Ordering::SeqCst);
        let mut frame_index: u64 = 0;
        let mut consecutive_errors: u32 = 0;
        const MAX_CONSECUTIVE_ERRORS: u32 = 30;

        while running.load(Ordering::SeqCst) {
            let loop_start = std::time::Instant::now();

            match capture.grab_frame() {
                Ok(rgb_data) => {
                    consecutive_errors = 0;
                    let width = capture.width();
                    let height = capture.height();

                    let estimate = if let Some(ref mut estimator) = pose_estimator {
                        estimator.estimate_pose(&rgb_data, width, height, frame_index)
                    } else {
                        pose_estimation::estimate_pose(&rgb_data, width, height, frame_index)
                    };

                    let frame = Some(downscale_for_gui(&rgb_data, width, height, 320));
                    mailbox.publish_estimate(estimate, frame);
                }
                Err(e) => {
                    consecutive_errors += 1;
                    if consecutive_errors == 1 {
                        error!("tracking-worker: frame grab error: {}", e);
                    }
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                        error!(
                            "tracking-worker: {} consecutive grab failures, stopping worker",
                            consecutive_errors
                        );
                        mailbox.report_error(t!("tracking.error_camera_stopped", count = consecutive_errors), TrackingErrorLevel::Blocking);
                        return;
                    }
                }
            }

            frame_index += 1;

            let elapsed = loop_start.elapsed();
            if elapsed < interval {
                thread::sleep(interval - elapsed);
            }
        }

        info!("tracking-worker: stopped");
    }
}

impl Drop for TrackingWorker {
    fn drop(&mut self) {
        self.stop();
    }
}

// ---------------------------------------------------------------------------
// Synthetic pose generation
// ---------------------------------------------------------------------------

/// Build a synthetic tracking sample with gentle animation driven by `t`.
/// Used by `CameraBackend::Synthetic` when no webcam is available.
fn synthetic_pose(frame_index: u64, t: f32) -> SourceSkeleton {
    use crate::asset::HumanoidBone;
    let mut sk = SourceSkeleton::empty(frame_index);

    // Shoulders bob a few cm laterally so the pose solver produces a
    // visible spine lean on synthetic input.
    let sx = t.cos() * 0.01;
    sk.joints.insert(
        HumanoidBone::LeftShoulder,
        SourceJoint {
            position: [-0.15 + sx, 1.4, 0.0],
            confidence: 0.88,
        },
    );
    sk.joints.insert(
        HumanoidBone::RightShoulder,
        SourceJoint {
            position: [0.15 + sx, 1.4, 0.0],
            confidence: 0.88,
        },
    );

    sk.face = Some(FacePose {
        yaw: t.sin() * 0.1,
        pitch: (t * 0.7).cos() * 0.05,
        roll: 0.0,
        confidence: 0.95,
    });
    sk.expressions = vec![SourceExpression {
        name: "blink".to_string(),
        weight: 0.1,
    }];
    sk.overall_confidence = 0.9;
    sk
}

/// Downscale an RGB frame to a maximum width, preserving aspect ratio.
fn downscale_for_gui(rgb_data: &[u8], src_w: u32, src_h: u32, max_w: u32) -> WebcamFrame {
    if src_w <= max_w {
        return WebcamFrame {
            rgb_data: rgb_data.to_vec(),
            width: src_w,
            height: src_h,
        };
    }
    let scale = max_w as f32 / src_w as f32;
    let dst_w = max_w;
    let dst_h = ((src_h as f32 * scale).round() as u32).max(1);
    let mut out = vec![0u8; (dst_w * dst_h * 3) as usize];
    for y in 0..dst_h {
        let sy = ((y as f32 / scale) as u32).min(src_h - 1);
        for x in 0..dst_w {
            let sx = ((x as f32 / scale) as u32).min(src_w - 1);
            let si = ((sy * src_w + sx) * 3) as usize;
            let di = ((y * dst_w + x) * 3) as usize;
            out[di] = rgb_data[si];
            out[di + 1] = rgb_data[si + 1];
            out[di + 2] = rgb_data[si + 2];
        }
    }
    WebcamFrame {
        rgb_data: out,
        width: dst_w,
        height: dst_h,
    }
}
