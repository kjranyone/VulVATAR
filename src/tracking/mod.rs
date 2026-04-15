#![allow(dead_code)]
#[cfg(feature = "webcam")]
use log::warn;
use log::{error, info};
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

        // Scale all positions by scale_factor
        let s = self.scale_factor;
        if let Some(ref mut t) = pose.head {
            vec3_scale(&mut t.position, s);
        }

        // Apply head_position_offset AFTER scaling
        if let Some(ref mut head) = pose.head {
            head.position[0] += self.head_position_offset[0];
            head.position[1] += self.head_position_offset[1];
            head.position[2] += self.head_position_offset[2];
        }
        if let Some(ref mut t) = pose.neck {
            vec3_scale(&mut t.position, s);
        }
        if let Some(ref mut t) = pose.spine {
            vec3_scale(&mut t.position, s);
        }
        if let Some(ref mut t) = pose.shoulders.left {
            vec3_scale(&mut t.position, s);
        }
        if let Some(ref mut t) = pose.shoulders.right {
            vec3_scale(&mut t.position, s);
        }
        if let Some(ref mut t) = pose.arms.left_upper {
            vec3_scale(&mut t.position, s);
        }
        if let Some(ref mut t) = pose.arms.left_lower {
            vec3_scale(&mut t.position, s);
        }
        if let Some(ref mut t) = pose.arms.right_upper {
            vec3_scale(&mut t.position, s);
        }
        if let Some(ref mut t) = pose.arms.right_lower {
            vec3_scale(&mut t.position, s);
        }

        if let Some(ref mut hands) = pose.hands {
            if let Some(ref mut t) = hands.left {
                vec3_scale(&mut t.position, s);
            }
            if let Some(ref mut t) = hands.right {
                vec3_scale(&mut t.position, s);
            }
        }

        if let Some(ref mut legs) = pose.legs {
            if let Some(ref mut t) = legs.left_upper {
                vec3_scale(&mut t.position, s);
            }
            if let Some(ref mut t) = legs.left_lower {
                vec3_scale(&mut t.position, s);
            }
            if let Some(ref mut t) = legs.right_upper {
                vec3_scale(&mut t.position, s);
            }
            if let Some(ref mut t) = legs.right_lower {
                vec3_scale(&mut t.position, s);
            }
        }

        if let Some(ref mut feet) = pose.feet {
            if let Some(ref mut t) = feet.left {
                vec3_scale(&mut t.position, s);
            }
            if let Some(ref mut t) = feet.right {
                vec3_scale(&mut t.position, s);
            }
        }

        if let Some(ref mut fingers) = pose.fingers {
            let scale_finger = |t: &mut Option<RigTarget>| {
                if let Some(ref mut t) = t {
                    vec3_scale(&mut t.position, s);
                }
            };
            scale_finger(&mut fingers.left_thumb);
            scale_finger(&mut fingers.left_index);
            scale_finger(&mut fingers.left_middle);
            scale_finger(&mut fingers.left_ring);
            scale_finger(&mut fingers.left_pinky);
            scale_finger(&mut fingers.right_thumb);
            scale_finger(&mut fingers.right_index);
            scale_finger(&mut fingers.right_middle);
            scale_finger(&mut fingers.right_ring);
            scale_finger(&mut fingers.right_pinky);
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
    sequence: u64,
    last_update_nanos: u64,
}

fn now_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

impl TrackingMailbox {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Mutex::new(TrackingMailboxInner {
                latest_pose: None,
                sequence: 0,
                last_update_nanos: 0,
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

    /// Thread-safe read: takes &self, locks the mutex, clones the pose.
    pub fn latest_pose(&self) -> Option<TrackingRigPose> {
        let inner = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        inner.latest_pose.clone()
    }

    /// Legacy read alias.
    pub fn read(&self) -> Option<TrackingRigPose> {
        self.latest_pose()
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

    /// Synchronous sample -- produces a `TrackingFrame` and publishes a
    /// `TrackingRigPose` to the internal mailbox.
    ///
    /// For the `Synthetic` backend this generates a synthetic pose inline.
    /// For the `Webcam` backend the actual capture runs on the worker thread;
    /// this method returns the latest data the worker has published.
    pub fn sample(&mut self) -> TrackingFrame {
        match &self.backend {
            CameraBackend::Synthetic => {
                let t = self.frame_index as f32 * 0.05;
                let pose = synthetic_pose(self.frame_index, t);
                self.mailbox.publish(pose);

                let frame = TrackingFrame {
                    source_id: self.source_id.clone(),
                    capture_timestamp: self.frame_index,
                    frame_size: [640, 480],
                    image_format: TrackingImageFormat::default(),
                    body_confidence: 0.9,
                    face_confidence: 0.8,
                };
                self.frame_index += 1;
                frame
            }
            #[cfg(feature = "webcam")]
            CameraBackend::Webcam { .. } => {
                // The worker thread does the real capture; here we just report
                // whatever the mailbox currently holds.
                let (body_conf, face_conf) = self
                    .mailbox
                    .latest_pose()
                    .as_ref()
                    .map(|p| (p.confidence.torso_confidence, p.confidence.face_confidence))
                    .unwrap_or((0.0, 0.0));

                let frame = TrackingFrame {
                    source_id: self.source_id.clone(),
                    capture_timestamp: self.frame_index,
                    frame_size: [640, 480],
                    image_format: TrackingImageFormat::Rgb8,
                    body_confidence: body_conf,
                    face_confidence: face_conf,
                };
                self.frame_index += 1;
                frame
            }
        }
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
    mailbox: TrackingMailbox,
}

impl TrackingWorker {
    /// Create a new tracking worker bound to the given mailbox.
    ///
    /// The worker is not started until `start()` is called.
    pub fn new(mailbox: TrackingMailbox) -> Self {
        Self {
            handle: None,
            running: Arc::new(AtomicBool::new(false)),
            mailbox,
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

        let running = Arc::clone(&self.running);
        let mailbox = self.mailbox.clone();
        let target_fps: u64 = fps.max(1) as u64;
        let frame_interval = Duration::from_micros(1_000_000 / target_fps);

        running.store(true, Ordering::SeqCst);

        let handle = thread::Builder::new()
            .name("tracking-worker".into())
            .spawn(move || {
                Self::worker_loop(
                    backend,
                    mailbox,
                    running,
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
    /// Blocks until the worker thread exits. Safe to call multiple times.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            if let Err(e) = handle.join() {
                error!("tracking-worker: thread panicked: {:?}", e);
            }
        }
    }

    // -- internal -----------------------------------------------------------

    fn worker_loop(
        backend: CameraBackend,
        mailbox: TrackingMailbox,
        running: Arc<AtomicBool>,
        frame_interval: Duration,
        width: u32,
        height: u32,
        fps: u32,
    ) {
        match backend {
            CameraBackend::Synthetic => {
                let _ = (width, height, fps);
                Self::run_synthetic(&mailbox, &running, frame_interval);
            }
            #[cfg(feature = "webcam")]
            CameraBackend::Webcam { camera_index } => {
                Self::run_webcam(
                    camera_index,
                    &mailbox,
                    &running,
                    frame_interval,
                    width,
                    height,
                    fps,
                );
            }
        }
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
                warn!("tracking-worker: falling back to synthetic backend");
                Self::run_synthetic(mailbox, running, interval);
                return;
            }
        };

        // Initialize inference engine (use default wholebody model)
        #[cfg(feature = "inference")]
        let pose_estimator =
            inference::CigPoseInference::new("models/cigpose-m_coco-wholebody_256x192.onnx").ok();
        #[cfg(not(feature = "inference"))]
        let pose_estimator: Option<inference::CigPoseInference> = None;

        info!("tracking-worker: webcam opened successfully");
        let mut frame_index: u64 = 0;

        while running.load(Ordering::SeqCst) {
            let loop_start = std::time::Instant::now();

            match capture.grab_frame() {
                Ok(rgb_data) => {
                    let width = capture.width();
                    let height = capture.height();

                    let pose = if let Some(ref mut estimator) = pose_estimator {
                        estimator.estimate_pose(&rgb_data, width, height, frame_index)
                    } else {
                        pose_estimation::estimate_pose(&rgb_data, width, height, frame_index)
                    };

                    mailbox.publish(pose);
                }
                Err(e) => {
                    error!("tracking-worker: frame grab error: {}", e);
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
