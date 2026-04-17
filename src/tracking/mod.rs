#![allow(dead_code)]
use log::{error, info, warn};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(feature = "webcam")]
mod webcam;

mod pose_estimation;

pub mod inference;

use crate::asset::Quat;
use crate::math_utils::Vec3;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TrackingSourceId(pub u64);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TrackingImageFormat {
    Rgb8,
    Bgr8,
    Rgba8,
    Bgra8,
    Nv12,
    Yuyv,
}

impl Default for TrackingImageFormat {
    fn default() -> Self {
        Self::Rgba8
    }
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

#[derive(Clone, Debug)]
pub struct TrackingRigPose {
    pub source_timestamp: u64,
    pub head: Option<RigTarget>,
    pub neck: Option<RigTarget>,
    pub spine: Option<RigTarget>,
    pub shoulders: RigShoulderTargets,
    pub arms: RigArmTargets,
    pub hands: Option<RigHandTargets>,
    pub legs: Option<RigLegTargets>,
    pub feet: Option<RigFeetTargets>,
    pub fingers: Option<RigFingerTargets>,
    pub expressions: ExpressionWeightSet,
    pub confidence: TrackingConfidenceMap,
}

#[derive(Clone, Debug)]
pub struct RigTarget {
    pub position: Vec3,
    pub orientation: Quat,
    pub confidence: f32,
}

#[derive(Clone, Debug)]
pub struct RigShoulderTargets {
    pub left: Option<RigTarget>,
    pub right: Option<RigTarget>,
}

#[derive(Clone, Debug)]
pub struct RigArmTargets {
    pub left_upper: Option<RigTarget>,
    pub left_lower: Option<RigTarget>,
    pub right_upper: Option<RigTarget>,
    pub right_lower: Option<RigTarget>,
}

#[derive(Clone, Debug)]
pub struct RigHandTargets {
    pub left: Option<RigTarget>,
    pub right: Option<RigTarget>,
}

#[derive(Clone, Debug)]
pub struct RigLegTargets {
    pub left_upper: Option<RigTarget>,
    pub left_lower: Option<RigTarget>,
    pub right_upper: Option<RigTarget>,
    pub right_lower: Option<RigTarget>,
}

#[derive(Clone, Debug)]
pub struct RigFeetTargets {
    pub left: Option<RigTarget>,
    pub right: Option<RigTarget>,
}

#[derive(Clone, Debug)]
pub struct RigFingerTargets {
    pub left_thumb: Option<RigTarget>,
    pub left_index: Option<RigTarget>,
    pub left_middle: Option<RigTarget>,
    pub left_ring: Option<RigTarget>,
    pub left_pinky: Option<RigTarget>,
    pub right_thumb: Option<RigTarget>,
    pub right_index: Option<RigTarget>,
    pub right_middle: Option<RigTarget>,
    pub right_ring: Option<RigTarget>,
    pub right_pinky: Option<RigTarget>,
}

#[derive(Clone, Debug)]
pub struct ExpressionWeightSet {
    pub weights: Vec<ExpressionWeight>,
}

#[derive(Clone, Debug)]
pub struct ExpressionWeight {
    pub name: String,
    pub weight: f32,
}

#[derive(Clone, Debug)]
pub struct TrackingConfidenceMap {
    pub head_confidence: f32,
    pub torso_confidence: f32,
    pub left_arm_confidence: f32,
    pub right_arm_confidence: f32,
    pub face_confidence: f32,
    pub left_leg_confidence: f32,
    pub right_leg_confidence: f32,
    pub left_hand_confidence: f32,
    pub right_hand_confidence: f32,
}

impl Default for TrackingConfidenceMap {
    fn default() -> Self {
        Self {
            head_confidence: 0.0,
            torso_confidence: 0.0,
            left_arm_confidence: 0.0,
            right_arm_confidence: 0.0,
            face_confidence: 0.0,
            left_leg_confidence: 0.0,
            right_leg_confidence: 0.0,
            left_hand_confidence: 0.0,
            right_hand_confidence: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrackingSmoothingParams {
    pub position_smoothing: f32,
    pub orientation_smoothing: f32,
    pub expression_smoothing: f32,
    pub confidence_threshold: f32,
    pub stale_timeout_nanos: u64,
}

impl Default for TrackingSmoothingParams {
    fn default() -> Self {
        Self {
            position_smoothing: 0.3,
            orientation_smoothing: 0.3,
            expression_smoothing: 0.2,
            confidence_threshold: 0.5,
            stale_timeout_nanos: 200_000_000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrackingCalibration {
    pub neutral_head_offset: Quat,
    pub scale_factor: f32,
    pub shoulder_width_override: Option<f32>,
    pub head_position_offset: Vec3,
}

impl Default for TrackingCalibration {
    fn default() -> Self {
        Self {
            neutral_head_offset: [0.0, 0.0, 0.0, 1.0],
            scale_factor: 1.0,
            shoulder_width_override: None,
            head_position_offset: [0.0, 0.0, 0.0],
        }
    }
}

impl TrackingCalibration {
    /// Apply calibration adjustments to a raw tracking pose in-place.
    ///
    /// - Subtracts `neutral_head_offset` from head orientation (inverse multiply).
    /// - Scales all target positions by `scale_factor`.
    /// - Adds `head_position_offset` to the head target position.
    /// - If `shoulder_width_override` is set, adjusts shoulder target X positions
    ///   to match the desired width.
    pub fn apply_calibration(&self, pose: &mut TrackingRigPose) {
        // Helper: multiply quaternion a * b
        fn quat_mul(a: &Quat, b: &Quat) -> Quat {
            [
                a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
                a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
                a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
                a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
            ]
        }

        // Helper: conjugate (inverse for unit quaternion)
        fn quat_conjugate(q: &Quat) -> Quat {
            [-q[0], -q[1], -q[2], q[3]]
        }

        // Helper: scale a Vec3
        fn vec3_scale(v: &mut Vec3, s: f32) {
            v[0] *= s;
            v[1] *= s;
            v[2] *= s;
        }

        // Subtract neutral_head_offset from head orientation:
        // corrected = inverse(offset) * raw
        if let Some(ref mut head) = pose.head {
            let inv_offset = quat_conjugate(&self.neutral_head_offset);
            head.orientation = quat_mul(&inv_offset, &head.orientation);
        }

        // Scale all target positions by scale_factor
        let s = self.scale_factor;
        pose.for_each_target_mut(|t| vec3_scale(&mut t.position, s));

        // Apply head_position_offset AFTER scaling
        if let Some(ref mut head) = pose.head {
            head.position[0] += self.head_position_offset[0];
            head.position[1] += self.head_position_offset[1];
            head.position[2] += self.head_position_offset[2];
        }

        // Adjust shoulder width if override is set
        if let Some(desired_width) = self.shoulder_width_override {
            let half_width = desired_width / 2.0;
            if let Some(ref mut left) = pose.shoulders.left {
                left.position[0] = -half_width;
            }
            if let Some(ref mut right) = pose.shoulders.right {
                right.position[0] = half_width;
            }
        }
    }
}

impl TrackingRigPose {
    /// Iterate over every `RigTarget` in the pose, calling `f` on each one.
    /// Covers head, neck, spine, shoulders, arms, hands, legs, feet, fingers.
    pub fn for_each_target_mut(&mut self, mut f: impl FnMut(&mut RigTarget)) {
        if let Some(ref mut t) = self.head {
            f(t);
        }
        if let Some(ref mut t) = self.neck {
            f(t);
        }
        if let Some(ref mut t) = self.spine {
            f(t);
        }
        if let Some(ref mut t) = self.shoulders.left {
            f(t);
        }
        if let Some(ref mut t) = self.shoulders.right {
            f(t);
        }
        if let Some(ref mut t) = self.arms.left_upper {
            f(t);
        }
        if let Some(ref mut t) = self.arms.left_lower {
            f(t);
        }
        if let Some(ref mut t) = self.arms.right_upper {
            f(t);
        }
        if let Some(ref mut t) = self.arms.right_lower {
            f(t);
        }
        if let Some(ref mut hands) = self.hands {
            if let Some(ref mut t) = hands.left {
                f(t);
            }
            if let Some(ref mut t) = hands.right {
                f(t);
            }
        }
        if let Some(ref mut legs) = self.legs {
            if let Some(ref mut t) = legs.left_upper {
                f(t);
            }
            if let Some(ref mut t) = legs.left_lower {
                f(t);
            }
            if let Some(ref mut t) = legs.right_upper {
                f(t);
            }
            if let Some(ref mut t) = legs.right_lower {
                f(t);
            }
        }
        if let Some(ref mut feet) = self.feet {
            if let Some(ref mut t) = feet.left {
                f(t);
            }
            if let Some(ref mut t) = feet.right {
                f(t);
            }
        }
        if let Some(ref mut fingers) = self.fingers {
            for t in [
                &mut fingers.left_thumb,
                &mut fingers.left_index,
                &mut fingers.left_middle,
                &mut fingers.left_ring,
                &mut fingers.left_pinky,
                &mut fingers.right_thumb,
                &mut fingers.right_index,
                &mut fingers.right_middle,
                &mut fingers.right_ring,
                &mut fingers.right_pinky,
            ] {
                if let Some(ref mut t) = t {
                    f(t);
                }
            }
        }
    }
}

impl Default for TrackingRigPose {
    fn default() -> Self {
        Self {
            source_timestamp: 0,
            head: None,
            neck: None,
            spine: None,
            shoulders: RigShoulderTargets {
                left: None,
                right: None,
            },
            arms: RigArmTargets {
                left_upper: None,
                left_lower: None,
                right_upper: None,
                right_lower: None,
            },
            hands: None,
            legs: None,
            feet: None,
            fingers: None,
            expressions: ExpressionWeightSet { weights: vec![] },
            confidence: TrackingConfidenceMap::default(),
        }
    }
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
    latest_pose: Option<TrackingRigPose>,
    latest_frame: Option<WebcamFrame>,
    latest_annotation: Option<DetectionAnnotation>,
    sequence: u64,
    last_update_nanos: u64,
    pending_error: Option<String>,
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
    pub pose: Option<TrackingRigPose>,
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
            })),
            stale_timeout_nanos: TrackingSmoothingParams::default().stale_timeout_nanos,
        }
    }

    /// Thread-safe publish: takes &self, locks the mutex internally.
    pub fn publish(&self, pose: TrackingRigPose) {
        let mut inner = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        inner.latest_pose = Some(pose);
        inner.sequence += 1;
        inner.last_update_nanos = now_nanos();
    }

    /// Alias for `publish` to match the update_pose naming convention.
    pub fn update_pose(&self, pose: TrackingRigPose) {
        self.publish(pose);
    }

    /// Publish a full estimation result including frame and annotations.
    pub fn publish_estimate(&self, estimate: PoseEstimate, frame: Option<WebcamFrame>) {
        let mut inner = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        inner.latest_pose = Some(estimate.rig_pose);
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
    pub fn report_error(&self, msg: String) {
        let mut inner = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        inner.pending_error = Some(msg);
    }

    /// Drain the latest pending error (returns it only once).
    pub fn drain_error(&self) -> Option<String> {
        let mut inner = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        inner.pending_error.take()
    }

    /// Thread-safe read: takes &self, locks the mutex, clones the pose.
    pub fn latest_pose(&self) -> Option<TrackingRigPose> {
        let inner = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        inner.latest_pose.clone()
    }

    /// Legacy read alias.
    pub fn read(&self) -> Option<TrackingRigPose> {
        self.latest_pose()
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

/// Combined output of a pose estimation pass: rig pose + 2D annotation.
pub struct PoseEstimate {
    pub rig_pose: TrackingRigPose,
    pub annotation: DetectionAnnotation,
}

// ---------------------------------------------------------------------------
// CameraBackend
// ---------------------------------------------------------------------------

/// Selects the capture backend for the tracking source.
#[derive(Clone, Debug)]
pub enum CameraBackend {
    /// Generate synthetic tracking data (no real camera needed).
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

impl Default for CameraBackend {
    fn default() -> Self {
        Self::Synthetic
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
/// the latest `TrackingRigPose` to a shared `TrackingMailbox`.
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

    fn worker_loop(
        backend: CameraBackend,
        mailbox: TrackingMailbox,
        running: Arc<AtomicBool>,
        ready: Arc<AtomicBool>,
        frame_interval: Duration,
        width: u32,
        height: u32,
        fps: u32,
    ) {
        match backend {
            CameraBackend::Synthetic => {
                let _ = (width, height, fps);
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
                );
            }
        }
        // Ensure running is cleared when the thread exits for any reason.
        running.store(false, Ordering::SeqCst);
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
    fn run_webcam(
        camera_index: usize,
        mailbox: &TrackingMailbox,
        running: &AtomicBool,
        ready: &AtomicBool,
        interval: Duration,
        width: u32,
        height: u32,
        fps: u32,
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
                mailbox.report_error(format!(
                    "Camera {} open failed: {}. Check if another app is using the camera.",
                    camera_index, e
                ));
                warn!("tracking-worker: falling back to synthetic backend");
                ready.store(true, Ordering::SeqCst);
                Self::run_synthetic(mailbox, running, interval);
                return;
            }
        };

        // Initialize inference engine (use default wholebody model)
        #[cfg(feature = "inference")]
        let mut pose_estimator =
            inference::CigPoseInference::new("models/cigpose-m_coco-wholebody_256x192.onnx").ok();
        #[cfg(not(feature = "inference"))]
        let mut pose_estimator: Option<inference::CigPoseInference> = None;

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
                        mailbox.report_error(format!(
                            "Camera stopped after {} consecutive read errors. \
                             Another app may be using the camera, or the device was disconnected.",
                            consecutive_errors
                        ));
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

/// Build a synthetic tracking pose with gentle animation driven by `t`.
fn synthetic_pose(frame_index: u64, t: f32) -> TrackingRigPose {
    let head_yaw = t.sin() * 0.1;
    let head_pitch = (t * 0.7).cos() * 0.05;
    let head_orientation: Quat = pose_estimation::euler_to_quat(head_pitch, head_yaw, 0.0);

    TrackingRigPose {
        source_timestamp: frame_index,
        head: Some(RigTarget {
            position: [0.0, 1.6, 0.0],
            orientation: head_orientation,
            confidence: 0.95,
        }),
        neck: Some(RigTarget {
            position: [0.0, 1.5, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            confidence: 0.9,
        }),
        spine: Some(RigTarget {
            position: [0.0, 1.0, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            confidence: 0.85,
        }),
        shoulders: RigShoulderTargets {
            left: Some(RigTarget {
                position: [-0.15, 1.4, 0.0],
                orientation: [0.0, 0.0, 0.0, 1.0],
                confidence: 0.88,
            }),
            right: Some(RigTarget {
                position: [0.15, 1.4, 0.0],
                orientation: [0.0, 0.0, 0.0, 1.0],
                confidence: 0.88,
            }),
        },
        arms: RigArmTargets {
            left_upper: Some(RigTarget {
                position: [-0.2, 1.2, 0.0],
                orientation: [0.0, 0.0, 0.0, 1.0],
                confidence: 0.8,
            }),
            left_lower: None,
            right_upper: Some(RigTarget {
                position: [0.2, 1.2, 0.0],
                orientation: [0.0, 0.0, 0.0, 1.0],
                confidence: 0.8,
            }),
            right_lower: None,
        },
        hands: None,
        legs: None,
        feet: None,
        fingers: None,
        expressions: ExpressionWeightSet {
            weights: vec![ExpressionWeight {
                name: "blink".to_string(),
                weight: 0.1,
            }],
        },
        confidence: TrackingConfidenceMap {
            head_confidence: 0.95,
            torso_confidence: 0.85,
            left_arm_confidence: 0.8,
            right_arm_confidence: 0.8,
            face_confidence: 0.75,
            left_leg_confidence: 0.0,
            right_leg_confidence: 0.0,
            left_hand_confidence: 0.0,
            right_hand_confidence: 0.0,
        },
    }
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
