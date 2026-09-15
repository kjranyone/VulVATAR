//! Capture-backend worker: owns the tracking threads and publishes into
//! the mailbox.
//!
//! `TrackingWorker` spawns the `tracking-worker` thread, which (with the
//! `realsense` feature) drives a two-thread split: the `tracking-capture`
//! thread owns the librealsense pipeline and overwrites a
//! `latest_cell::LatestCell` with the freshest frame, while this thread
//! blocks on the cell and runs inference + mailbox publishing. The split
//! keeps latency bounded when an inference pass is slower than the frame
//! period — see `run_realsense` for the full rationale.
//!
//! Also carries the worker's public data shape: `PoseEstimate`
//! (estimation output), `TrackingSource` (app-side mailbox handle) and
//! `CAPTURE_BACKEND_LABEL`. Everything public is re-exported at the
//! `crate::tracking` root; the handoff types live in `mailbox.rs`.

#[cfg(feature = "realsense")]
use crate::t;
use log::{error, warn};
// `info!` only fires from the realsense capture loop; gate the import so a
// no-capture-backend build doesn't warn on it being unused.
#[cfg(feature = "realsense")]
use log::info;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use super::{debug_channel, provider, session_record, stagelog};
#[cfg(feature = "realsense")]
use super::{latest_cell, realsense, sequence_recorder};
#[cfg(feature = "realsense")]
use super::{pose_estimation, CameraIntrinsics, TrackingErrorLevel};
use super::{DetectionAnnotation, PreviewFrame, SourceSkeleton, TrackingMailbox};

// ---------------------------------------------------------------------------
// RuntimeGpuBudget cadence knobs (P3-03)
// ---------------------------------------------------------------------------
// Same shape as `rtmw3d::YOLOX_REFRESH_PERIOD`: one Application writes,
// the tracking-worker loop reads, so a plain static beats `Arc<Atomic>`
// plumbing (see that static's doc for the full rationale).

/// Pose estimate cadence target in Hz, published by
/// `Application::update_runtime_gpu_budget`. The estimate loop drops
/// frames that fail the pacing check (see [`pose_throttle_allows`]);
/// the mailbox keeps the last published estimate, so the app sees a
/// slower pose rate rather than a stall. Healthy default 30 = the
/// camera rate, i.e. no throttling in practice.
pub static POSE_HZ_TARGET: AtomicU32 = AtomicU32::new(30);

/// Metric-depth cloud rebuild period, published by
/// `Application::update_runtime_gpu_budget`: rebuild the full-frame
/// cloud every Nth frame and hand the provider a clone of the last
/// cloud in between. 1 (Healthy default) = the pre-budget per-frame
/// rebuild.
pub static DEPTH_REFRESH_PERIOD: AtomicU64 = AtomicU64::new(1);

/// Pacing decision for the pose-Hz throttle: publish this frame or
/// skip it. `next_pub_ms` / `published_once` form a device-clock
/// pacing state owned by the caller. The accumulator advances by a
/// fixed interval per publish (a virtual clock), so the long-run
/// publish rate converges to `hz` from any camera rate ≥ `hz` —
/// e.g. a 30 fps camera at hz=25 settles into a 5-publish-1-skip
/// pattern. Two simpler gates fail that case: a min-interval gate
/// collapses to 15 Hz (every 33 ms frame misses a 40 ms threshold),
/// and anchoring the next deadline at the publish timestamp (`ts +
/// interval`) never accrues credit and also collapses to 15 Hz. If
/// the loop stalls and falls behind, the deadline clamps to the
/// current timestamp so catch-up never bursts. `hz == 0` disables
/// throttling (defensive; the budget never emits 0).
pub fn pose_throttle_allows(
    frame_ts_ms: f64,
    next_pub_ms: &mut f64,
    published_once: &mut bool,
    hz: u32,
) -> bool {
    if hz == 0 {
        return true;
    }
    const EPS_MS: f64 = 1e-3;
    let interval_ms = 1000.0 / hz as f64;
    if !*published_once || frame_ts_ms + EPS_MS >= *next_pub_ms {
        *published_once = true;
        *next_pub_ms = (*next_pub_ms + interval_ms).max(frame_ts_ms);
        true
    } else {
        false
    }
}

/// Combined output of a pose estimation pass: source skeleton + 2D annotation.
pub struct PoseEstimate {
    pub skeleton: SourceSkeleton,
    pub annotation: DetectionAnnotation,
}

// ---------------------------------------------------------------------------
// Capture backend
// ---------------------------------------------------------------------------

/// Display label for the sole capture backend. This is a D435-exclusive
/// build: the Intel RealSense D435 (color-aligned metric depth) is the only
/// camera path, so the former "which backend" selector collapsed to a
/// constant. Shown in the status bar / tracking inspector while capture is
/// live.
pub const CAPTURE_BACKEND_LABEL: &str = "RealSense D435";

// ---------------------------------------------------------------------------
// TrackingSource
// ---------------------------------------------------------------------------

pub struct TrackingSource {
    mailbox: TrackingMailbox,
}

impl TrackingSource {
    pub fn new() -> Self {
        Self {
            mailbox: TrackingMailbox::new(),
        }
    }
}

impl Default for TrackingSource {
    fn default() -> Self {
        Self::new()
    }
}

impl TrackingSource {
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

    /// Spawn the tracking worker thread with the given capture parameters.
    /// The thread loops at approximately `fps` frames per second, capturing
    /// frames and publishing poses to the shared mailbox. If a worker is
    /// already running this is a no-op.
    ///
    /// `camera_serial` is the Tracking panel's device selection (see
    /// [`realsense::RealSenseCapture::open`]); `None` = first enumerated
    /// D400.
    #[allow(clippy::too_many_arguments)]
    pub fn start_with_params(
        &mut self,
        width: u32,
        height: u32,
        fps: u32,
        camera_serial: Option<String>,
        pipeline: provider::TrackingPipelineConfig,
    ) {
        // A handle can be left behind by a `stop()` that timed out while
        // the worker was wedged in a blocking camera call. If that thread
        // has since finished, reap it here and proceed with the restart —
        // the old behaviour (silent `return`) left the user with a Start
        // button that did nothing, with no path back short of an app
        // restart. Only a thread that is STILL alive blocks the restart
        // (overlapping GPU inference sessions must never coexist), and
        // that now gets an explicit error log instead of silence.
        if let Some(handle) = &self.handle {
            if handle.is_finished() {
                if let Some(handle) = self.handle.take() {
                    if let Err(e) = handle.join() {
                        error!("tracking-worker: previous thread panicked: {:?}", e);
                    }
                }
            } else {
                error!(
                    "tracking-worker: start requested while the previous worker thread is still \
                     shutting down; retry once it exits"
                );
                return;
            }
        }

        self.running = Arc::new(AtomicBool::new(true));
        self.ready = Arc::new(AtomicBool::new(false));
        let running = Arc::clone(&self.running);
        let ready = Arc::clone(&self.ready);
        let mailbox = self.mailbox.clone();

        let handle = thread::Builder::new()
            .name("tracking-worker".into())
            .spawn(move || {
                Self::worker_loop(
                    mailbox,
                    running,
                    ready,
                    width,
                    height,
                    fps,
                    camera_serial,
                    pipeline,
                );
            })
            .expect("failed to spawn tracking-worker thread");

        self.handle = Some(handle);
    }

    /// Signal the worker to stop and join the thread.
    ///
    /// Waits up to 3 seconds for the worker thread to exit. If it hasn't
    /// exited by then (e.g. `grab_frame()` is blocking indefinitely), the
    /// join handle is kept so a later call can reap it once the blocked
    /// operation unwinds. This prevents a restart from orphaning a still-live
    /// inference worker and creating overlapping GPU sessions.
    ///
    /// Returns `true` when the worker fully stopped and was joined, or when
    /// there was no worker to stop. Returns `false` when the stop request was
    /// issued but the worker is still alive after the timeout.
    pub fn stop(&mut self) -> bool {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let deadline = std::time::Instant::now() + Duration::from_secs(3);
            loop {
                if handle.is_finished() {
                    if let Err(e) = handle.join() {
                        error!("tracking-worker: thread panicked: {:?}", e);
                    }
                    return true;
                }
                if std::time::Instant::now() >= deadline {
                    warn!(
                        "tracking-worker: thread did not exit within timeout; keeping handle and refusing overlapping restart"
                    );
                    self.handle = Some(handle);
                    return false;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        true
    }

    // -- internal -----------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn worker_loop(
        mailbox: TrackingMailbox,
        running: Arc<AtomicBool>,
        ready: Arc<AtomicBool>,
        width: u32,
        height: u32,
        fps: u32,
        camera_serial: Option<String>,
        pipeline: provider::TrackingPipelineConfig,
    ) {
        // D435-exclusive: the RealSense depth camera is the sole capture
        // backend. When the `realsense` feature is compiled out there is no
        // camera to drive the pipeline, so the worker reports ready and
        // exits immediately — the app falls back to the avatar rest pose.
        #[cfg(feature = "realsense")]
        Self::run_realsense(
            &mailbox,
            &running,
            &ready,
            width,
            height,
            fps,
            camera_serial,
            pipeline,
        );
        #[cfg(not(feature = "realsense"))]
        {
            let _ = (width, height, fps, camera_serial, pipeline);
            warn!("tracking-worker: `realsense` feature disabled — no capture backend, idling");
            ready.store(true, Ordering::SeqCst);
        }
        // Ensure running is cleared when the thread exits for any reason.
        running.store(false, Ordering::SeqCst);
        // Clear the backend label so the inspector hides the row instead
        // of showing a stale value from the previous session.
        mailbox.set_inference_backend_label(None);
    }

    /// Capture color + aligned metric depth from a RealSense D435 and run
    /// the fusion pipeline, feeding the depth in via
    /// [`provider::PoseProvider::set_external_depth`]. Each per-frame step
    /// builds a `MetricDepthFrame` from the D435 depth and hands it to
    /// the provider before `estimate_pose`.
    ///
    /// Two threads:
    ///
    /// * **`tracking-capture`** (spawned here) owns the librealsense
    ///   pipeline: it opens the camera, grabs frames, and drops the
    ///   freshest one into a [`latest_cell::LatestCell`]. When grabs fail
    ///   persistently it reopens the device with backoff
    ///   ([`GrabRecovery`]).
    /// * **`tracking-worker`** (this thread) blocks on the cell and runs
    ///   inference + publish on whatever frame is freshest.
    ///
    /// The split is what keeps latency bounded: with grab and inference
    /// serialized on one thread, an inference pass slower than the frame
    /// period made `pipeline.wait` drain librealsense's internal queue
    /// oldest-first — the pipeline fell one-plus frames behind and STAYED
    /// behind. The cell's latest-only semantics discard frames that went
    /// stale while inference was busy, and the camera itself paces the
    /// loop (no `thread::sleep`: sleeping after a fast inference pass
    /// only made the next frame older by the slept amount).
    #[cfg(feature = "realsense")]
    #[allow(clippy::too_many_arguments)]
    fn run_realsense(
        mailbox: &TrackingMailbox,
        running: &Arc<AtomicBool>,
        ready: &AtomicBool,
        width: u32,
        height: u32,
        fps: u32,
        camera_serial: Option<String>,
        pipeline: provider::TrackingPipelineConfig,
    ) {
        info!(
            "tracking-worker: opening RealSense D435 ({}x{} @ {} fps, serial {:?})",
            width, height, fps, camera_serial
        );

        let _stage_session =
            stagelog::SessionGuard::begin(&format!("realsense {width}x{height}@{fps}"));
        stagelog::mark(0, "camera_open_begin");

        let cell = latest_cell::LatestCell::<CaptureItem>::new();
        let (open_tx, open_rx) =
            std::sync::mpsc::channel::<Result<(u32, u32), realsense::OpenFailure>>();
        let capture_handle = {
            let cell = Arc::clone(&cell);
            let running = Arc::clone(running);
            let mailbox = mailbox.clone();
            thread::Builder::new()
                .name("tracking-capture".into())
                .spawn(move || {
                    capture_loop(
                        cell,
                        running,
                        mailbox,
                        open_tx,
                        width,
                        height,
                        fps,
                        camera_serial,
                    );
                })
                .expect("failed to spawn tracking-capture thread")
        };

        // The capture thread reports the open outcome exactly once.
        let (cap_width, cap_height) = match open_rx.recv() {
            Ok(Ok(dims)) => dims,
            Ok(Err(e)) => {
                error!("tracking-worker: failed to open RealSense: {}", e);
                // Map the typed open failure to a root-cause-specific,
                // localized message so the GUI dialog names the real problem
                // (e.g. a USB-2 link speed) instead of an opaque driver string.
                let msg = match e {
                    realsense::OpenFailure::UsbLinkTooSlow { detected } => {
                        t!("tracking.error_usb_link_speed", detected = detected)
                    }
                    realsense::OpenFailure::NoDevice => t!("tracking.error_no_device"),
                    realsense::OpenFailure::Other(m) => {
                        t!("tracking.error_realsense_open", error = m)
                    }
                };
                mailbox.report_error(msg, TrackingErrorLevel::Blocking);
                // No synthetic fallback in the D435-exclusive build: surface
                // the blocking error and idle. `worker_loop` clears `running`
                // on return, so the app drops to the avatar rest pose.
                ready.store(true, Ordering::SeqCst);
                let _ = capture_handle.join();
                return;
            }
            Err(_) => {
                error!("tracking-worker: capture thread exited before reporting an open result");
                ready.store(true, Ordering::SeqCst);
                let _ = capture_handle.join();
                return;
            }
        };

        // Provider init under cooperative GPU exclusivity: the ONNX/EP
        // load bursts the GPU, so serialize it against other init work.
        stagelog::mark(0, "provider_load_begin");
        let mut pose_provider = {
            let _gpu_exclusive =
                crate::gpu_coordination::GpuExclusiveGuard::acquire("pose-provider-init");
            match provider::create_pose_provider_live("models", pipeline) {
                Ok(mut provider) => {
                    let warnings = provider.take_load_warnings();
                    if !warnings.is_empty() {
                        mailbox.report_error(
                            t!(
                                "tracking.error_model_warning",
                                warnings = warnings.join("; ")
                            ),
                            TrackingErrorLevel::Warning,
                        );
                    }
                    mailbox.set_inference_backend_label(Some(provider.label()));
                    stagelog::mark(0, "provider_warmup_begin");
                    let blank = vec![0u8; (cap_width as usize) * (cap_height as usize) * 3];
                    let _ = provider.estimate_pose(&blank, cap_width, cap_height, 0);
                    stagelog::mark(0, "provider_warmup_end");
                    provider.reset_temporal_state();
                    Some(provider)
                }
                Err(e) => {
                    error!("tracking-worker: inference disabled: {}", e);
                    mailbox.report_error(
                        t!("tracking.error_model_unavailable", error = e.to_string()),
                        TrackingErrorLevel::Blocking,
                    );
                    None
                }
            }
        };

        info!("tracking-worker: RealSense opened successfully");
        stagelog::mark(0, "provider_load_end");
        ready.store(true, Ordering::SeqCst);
        // P3-03 pose-Hz pacing state (device-clock accumulator).
        let mut pose_next_pub_ms: f64 = 0.0;
        let mut pose_published_once = false;
        // P3-03 depth-refresh cache: the last full-frame metric cloud,
        // cloned to the provider on frames between rebuilds.
        let mut cached_metric: Option<crate::tracking::metric_frame::MetricDepthFrame> = None;

        while running.load(Ordering::SeqCst) {
            // Blocks until the freshest capture is available; wakes with
            // `None` once the capture thread closed the cell (stop
            // request or unrecoverable camera loss).
            let Some(CaptureItem {
                frame: rs_frame,
                index: frame_index,
            }) = cell.take_blocking()
            else {
                break;
            };

            let width = rs_frame.width;
            let height = rs_frame.height;

            // Raw-input recorder (off unless the flag file exists). Placed
            // BEFORE any processing on purpose: the whole point is to keep a
            // record that no estimator has touched, so a replacement can be
            // scored on the same sensor data as the incumbent. Runs even on
            // pose-throttled frames — the capture stream must stay gapless.
            sequence_recorder::record(frame_index, &rs_frame);

            // P3-03 pose-Hz throttle: under GPU pressure the budget lowers
            // the estimate cadence below the camera rate. Frames that fail
            // the pacing check are dropped before the expensive inference
            // pass (mailbox keeps the last published estimate). Disabled
            // while the session recorder is active — its pose.jsonl must
            // stay gap-free relative to the raw captures, and diagnostics
            // sessions run with the budget effectively Healthy anyway.
            let pose_hz = POSE_HZ_TARGET.load(Ordering::Relaxed);
            if !session_record::active()
                && !pose_throttle_allows(
                    rs_frame.timestamp_ms,
                    &mut pose_next_pub_ms,
                    &mut pose_published_once,
                    pose_hz,
                )
            {
                continue;
            }

            stagelog::mark(frame_index, "estimate_begin");
            let mut estimate = if let Some(ref mut provider) = pose_provider {
                // Hand the D435's color-aligned metric depth to the
                // provider for THIS frame. Under pressure the budget
                // stretches the rebuild cadence (`DEPTH_REFRESH_PERIOD`)
                // and skipped frames reuse a clone of the last cloud —
                // stale by at most one refresh interval. Period 1
                // (Healthy) rebuilds every frame = pre-budget behaviour.
                let period = DEPTH_REFRESH_PERIOD.load(Ordering::Relaxed).max(1);
                let metric =
                    if period <= 1 || frame_index.is_multiple_of(period) || cached_metric.is_none()
                    {
                        let m = crate::tracking::metric_frame::build_metric_frame_from_d435(
                            &rs_frame,
                        );
                        if period > 1 {
                            cached_metric = Some(m.clone());
                        }
                        m
                    } else {
                        // Unwrap is infallible: the `cached_metric.is_none()`
                        // arm above populates it before we can get here.
                        cached_metric.clone().unwrap()
                    };
                provider.set_external_depth(metric);
                // Session recording pairs every pose with its own raw
                // capture (provenance analysis depends on it) — run the
                // synchronous protocol there, which consumes exactly this
                // frame's detector result. Otherwise use the pipelined
                // live entry point: may return None when the detector
                // thread has no fresh result yet — skip this capture
                // entirely (same shape as the pose-Hz throttle path); a
                // returned pose may describe the PREVIOUS capture,
                // carrying its own capture_timestamp_ms.
                if session_record::active() {
                    provider.estimate_pose(&rs_frame.rgb, width, height, frame_index)
                } else {
                    match provider.estimate_pose_latest(&rs_frame.rgb, width, height, frame_index)
                    {
                        Some(est) => est,
                        None => {
                            stagelog::mark(frame_index, "estimate_skipped");
                            continue;
                        }
                    }
                }
            } else {
                pose_estimation::estimate_pose(&rs_frame.rgb, width, height, frame_index)
            };
            // Stamp the device capture time onto the published sample —
            // the solver's measurement filters derive their dt from
            // consecutive capture timestamps (render/wall clocks say
            // nothing about when the subject actually moved). The fusion
            // provider already stamps the CONSUMED frame's device time
            // (which in pipelined mode is the previous capture's), so
            // only fill the gap for providers that don't know better.
            if estimate.skeleton.capture_timestamp_ms.is_none() {
                estimate.skeleton.capture_timestamp_ms = Some(rs_frame.timestamp_ms);
            }

            // Live debug channel: publish camera + 2D keypoints + source
            // arm joints for an external overlay (no-op unless the debug
            // flag file exists). Before the estimate is moved below.
            debug_channel::dump_observation(frame_index, &rs_frame.rgb, width, height, &estimate);
            // Full-res aligned depth snapshot once a second: lets an
            // external audit read the exact depth pixels under any
            // keypoint of the live session, without stealing the camera.
            if frame_index % 30 == 0 {
                debug_channel::dump_depth_snapshot(
                    frame_index,
                    &rs_frame.depth_raw,
                    width,
                    height,
                    rs_frame.depth_units,
                );
            }

            // Session recording: append this capture frame to the offline
            // time series (no-op unless VULVATAR_RECORD is set). Placed
            // beside the debug channel because both need the estimate
            // before it is moved into the mailbox — but unlike that
            // latest-only channel this one keeps every frame, so a
            // misbehaviour that lasted three frames is still there
            // afterwards. The raw colour/depth are passed by reference and
            // only cloned for frames that trip the jump trigger.
            session_record::record(
                frame_index,
                &estimate.skeleton,
                &rs_frame.rgb,
                width,
                height,
                &rs_frame.depth_raw,
                rs_frame.depth_units,
                CameraIntrinsics {
                    fx: rs_frame.intrinsics.fx,
                    fy: rs_frame.intrinsics.fy,
                    cx: rs_frame.intrinsics.cx,
                    cy: rs_frame.intrinsics.cy,
                    width: rs_frame.intrinsics.width,
                    height: rs_frame.intrinsics.height,
                },
                rs_frame.timestamp_ms,
            );

            let frame = Some(downscale_for_gui(&rs_frame.rgb, width, height, 320));
            mailbox.publish_estimate(estimate, frame);
            stagelog::mark(frame_index, "publish");
        }

        let _ = capture_handle.join();
        session_record::finish();
        info!("tracking-worker: stopped");
    }
}

/// One frame handed from the capture thread to the inference thread.
#[cfg(feature = "realsense")]
struct CaptureItem {
    frame: realsense::RealSenseFrame,
    index: u64,
}

/// Camera-owning loop of the `tracking-capture` thread: open the device
/// (reporting the outcome once through `open_tx`), then grab frames and
/// overwrite the shared [`latest_cell::LatestCell`] with the freshest one.
/// Persistent grab failures tear the pipeline down and reopen it with
/// backoff ([`GrabRecovery`]); desynced framesets are dropped without
/// counting toward that. Closes the cell on exit, which is the inference
/// thread's wake-up-and-quit signal.
#[cfg(feature = "realsense")]
#[allow(clippy::too_many_arguments)]
fn capture_loop(
    cell: Arc<latest_cell::LatestCell<CaptureItem>>,
    running: Arc<AtomicBool>,
    mailbox: TrackingMailbox,
    open_tx: std::sync::mpsc::Sender<Result<(u32, u32), realsense::OpenFailure>>,
    width: u32,
    height: u32,
    fps: u32,
    camera_serial: Option<String>,
) {
    let mut capture =
        match realsense::RealSenseCapture::open(width, height, fps, camera_serial.as_deref()) {
            Ok(c) => {
                let _ = open_tx.send(Ok((c.width(), c.height())));
                Some(c)
            }
            Err(e) => {
                let _ = open_tx.send(Err(e));
                cell.close();
                return;
            }
        };
    drop(open_tx);

    let mut frame_index: u64 = 0;
    let mut recovery = GrabRecovery::default();
    let mut sync_drops: u64 = 0;

    while running.load(Ordering::SeqCst) {
        let Some(cap) = capture.as_mut() else {
            // Reconnect path: the previous capture was torn down after a
            // persistent failure streak.
            match recovery.next_reopen_backoff() {
                None => {
                    error!(
                        "tracking-capture: camera did not come back after {} reconnect attempts, stopping",
                        GrabRecovery::MAX_REOPEN_ATTEMPTS
                    );
                    mailbox.report_error(
                        t!(
                            "tracking.error_camera_stopped",
                            count = recovery.total_errors()
                        ),
                        TrackingErrorLevel::Blocking,
                    );
                    break;
                }
                Some(backoff) => {
                    if !sleep_while_running(&running, backoff) {
                        break;
                    }
                    match realsense::RealSenseCapture::open(
                        width,
                        height,
                        fps,
                        camera_serial.as_deref(),
                    ) {
                        Ok(c) => {
                            info!("tracking-capture: camera reconnected");
                            capture = Some(c);
                            recovery.on_reopen_success();
                        }
                        Err(e) => warn!("tracking-capture: reconnect attempt failed: {e}"),
                    }
                }
            }
            continue;
        };

        match cap.grab_frame() {
            Ok(frame) => {
                recovery.on_success();
                stagelog::mark(frame_index, "capture_put");
                cell.put(CaptureItem {
                    frame,
                    index: frame_index,
                });
                frame_index += 1;
            }
            Err(realsense::GrabError::SyncMismatch { delta_ms }) => {
                // Healthy stream, unusable frameset. Log throttled —
                // in a dim room with a struggling exposure this can
                // recur for a while.
                sync_drops += 1;
                if sync_drops.is_power_of_two() {
                    warn!(
                        "tracking-capture: dropped depth/color-desynced frameset (#{sync_drops}, Δ {delta_ms:.1} ms)"
                    );
                }
            }
            Err(realsense::GrabError::Capture(msg)) => match recovery.on_capture_error(&msg) {
                GrabAction::Retry => {}
                GrabAction::LogAndRetry => {
                    error!("tracking-capture: frame grab error: {msg}");
                }
                GrabAction::Reopen => {
                    warn!(
                        "tracking-capture: {} consecutive grab failures — reopening the camera",
                        GrabRecovery::REOPEN_AFTER
                    );
                    capture = None;
                }
            },
        }
    }

    cell.close();
    info!("tracking-capture: stopped");
}

/// Sleep `total` in short slices, re-checking `running` between slices so
/// a stop request interrupts a reconnect backoff promptly. Returns `false`
/// when `running` flipped off during the wait.
#[cfg(feature = "realsense")]
fn sleep_while_running(running: &AtomicBool, total: Duration) -> bool {
    let mut remaining = total;
    while remaining > Duration::ZERO {
        if !running.load(Ordering::SeqCst) {
            return false;
        }
        let step = remaining.min(Duration::from_millis(100));
        thread::sleep(step);
        remaining = remaining.saturating_sub(step);
    }
    running.load(Ordering::SeqCst)
}

/// Grab-failure recovery ladder for the capture loop. Pure state machine
/// (no I/O, no clocks) so the retry → log-on-change → reopen → give-up
/// policy is unit-testable without a camera.
#[derive(Debug, Default)]
struct GrabRecovery {
    consecutive_errors: u32,
    total_errors: u64,
    last_logged: Option<String>,
    reopen_attempts: u32,
}

#[derive(Debug, PartialEq, Eq)]
enum GrabAction {
    /// Same error as already logged — retry silently.
    Retry,
    /// New (or changed) error message — log it, then retry. Logging on
    /// *change* rather than only on the first error of a streak keeps a
    /// mid-streak transition (timeout → device disconnected) visible.
    LogAndRetry,
    /// Failure streak exhausted the retry budget — tear the pipeline
    /// down and reopen the device.
    Reopen,
}

impl GrabRecovery {
    /// Consecutive capture errors before a reopen. At the 500 ms steady
    /// grab timeout this is ~5 s of a wedged / unplugged camera.
    const REOPEN_AFTER: u32 = 10;
    /// Reopen attempts (each preceded by [`Self::next_reopen_backoff`])
    /// before giving up for good.
    const MAX_REOPEN_ATTEMPTS: u32 = 5;

    fn on_success(&mut self) {
        self.consecutive_errors = 0;
        self.last_logged = None;
        self.reopen_attempts = 0;
    }

    fn on_capture_error(&mut self, msg: &str) -> GrabAction {
        self.consecutive_errors += 1;
        self.total_errors += 1;
        if self.consecutive_errors >= Self::REOPEN_AFTER {
            self.consecutive_errors = 0;
            return GrabAction::Reopen;
        }
        if self.last_logged.as_deref() != Some(msg) {
            self.last_logged = Some(msg.to_string());
            GrabAction::LogAndRetry
        } else {
            GrabAction::Retry
        }
    }

    /// Backoff before the next reopen attempt: 0.5 s doubling to an 8 s
    /// cap. `None` once the attempt budget is spent.
    fn next_reopen_backoff(&mut self) -> Option<Duration> {
        if self.reopen_attempts >= Self::MAX_REOPEN_ATTEMPTS {
            return None;
        }
        let backoff = Duration::from_millis(500u64 << self.reopen_attempts.min(4));
        self.reopen_attempts += 1;
        Some(backoff)
    }

    fn on_reopen_success(&mut self) {
        self.on_success();
    }

    fn total_errors(&self) -> u64 {
        self.total_errors
    }
}

#[cfg(test)]
mod grab_recovery_tests {
    use super::*;

    #[test]
    fn logs_on_error_message_change_not_just_first() {
        let mut r = GrabRecovery::default();
        assert_eq!(r.on_capture_error("timeout"), GrabAction::LogAndRetry);
        assert_eq!(r.on_capture_error("timeout"), GrabAction::Retry);
        // The mid-streak transition must be visible.
        assert_eq!(r.on_capture_error("disconnected"), GrabAction::LogAndRetry);
        assert_eq!(r.on_capture_error("disconnected"), GrabAction::Retry);
    }

    #[test]
    fn reopen_after_persistent_streak_and_success_resets() {
        let mut r = GrabRecovery::default();
        for _ in 0..GrabRecovery::REOPEN_AFTER - 1 {
            let a = r.on_capture_error("timeout");
            assert_ne!(a, GrabAction::Reopen);
        }
        assert_eq!(r.on_capture_error("timeout"), GrabAction::Reopen);
        // A successful grab resets the streak entirely.
        r.on_success();
        assert_eq!(r.on_capture_error("timeout"), GrabAction::LogAndRetry);
    }

    #[test]
    fn reopen_backoff_doubles_then_gives_up() {
        let mut r = GrabRecovery::default();
        let mut seen = Vec::new();
        while let Some(b) = r.next_reopen_backoff() {
            seen.push(b.as_millis() as u64);
        }
        assert_eq!(seen, vec![500, 1000, 2000, 4000, 8000]);
        assert!(r.next_reopen_backoff().is_none(), "budget spent");
        // A successful reopen restores the full budget.
        r.on_reopen_success();
        assert_eq!(r.next_reopen_backoff(), Some(Duration::from_millis(500)));
    }
}

impl Drop for TrackingWorker {
    fn drop(&mut self) {
        // During normal lifecycle management `stop()` should already have
        // joined the thread. If destruction happens while a camera backend is
        // still wedged inside a blocking call, there is no non-blocking way to
        // recover in `Drop`; requesting stop one last time is still the least
        // surprising behaviour.
        let _ = self.stop();
    }
}

/// Downscale an RGB frame to a maximum width, preserving aspect ratio.
fn downscale_for_gui(rgb_data: &[u8], src_w: u32, src_h: u32, max_w: u32) -> PreviewFrame {
    if src_w <= max_w {
        return PreviewFrame {
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
    PreviewFrame {
        rgb_data: out,
        width: dst_w,
        height: dst_h,
    }
}

#[cfg(test)]
mod pose_throttle_tests {
    use super::pose_throttle_allows;

    /// Camera-rate device timestamps for `fps` over `frames` frames.
    fn stamps(fps: f64, frames: u32) -> impl Iterator<Item = f64> {
        let dt = 1000.0 / fps;
        (0..frames).map(move |i| i as f64 * dt)
    }

    #[test]
    fn hz_zero_disables_throttling() {
        let (mut next, mut once) = (0.0, false);
        for ts in stamps(30.0, 100) {
            assert!(pose_throttle_allows(ts, &mut next, &mut once, 0));
        }
    }

    #[test]
    fn hz_at_camera_rate_publishes_every_frame() {
        let (mut next, mut once) = (0.0, false);
        for ts in stamps(30.0, 300) {
            assert!(pose_throttle_allows(ts, &mut next, &mut once, 30));
        }
    }

    #[test]
    fn camera_rate_converges_to_off_target_hz() {
        // 30 fps camera throttled to 25 Hz over 10 s: the accumulator
        // must land near 25 Hz — a naive min-interval gate collapses
        // this case to 15 Hz.
        let (mut next, mut once) = (0.0, false);
        let published = stamps(30.0, 300)
            .filter(|&ts| pose_throttle_allows(ts, &mut next, &mut once, 25))
            .count();
        assert!(
            (235..=265).contains(&published),
            "expected ~250 publishes over 300 frames, got {}",
            published
        );
    }

    #[test]
    fn half_rate_never_bursts() {
        // 30 fps camera throttled to 15 Hz: exactly every other frame,
        // and never two publishes on consecutive frames.
        let (mut next, mut once) = (0.0, false);
        let mut prev_published = false;
        let mut published = 0;
        for ts in stamps(30.0, 300) {
            let allow = pose_throttle_allows(ts, &mut next, &mut once, 15);
            if allow {
                published += 1;
                assert!(!prev_published, "two consecutive publishes at hz=15");
            }
            prev_published = allow;
        }
        assert_eq!(published, 150);
    }

    #[test]
    fn first_frame_always_publishes() {
        let (mut next, mut once) = (0.0, false);
        assert!(pose_throttle_allows(12_345.0, &mut next, &mut once, 20));
        // First publish anchors the virtual clock at the frame itself.
        assert_eq!(next, 12_345.0);
    }
}
