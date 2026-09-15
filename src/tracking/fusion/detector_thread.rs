//! Detector-side thread of the pipelined fusion provider.
//!
//! Live tracking is camera-paced at 30 fps while the detector stage
//! (YOLOX wait + crop/preprocess + the DirectML RTMW3D run + decode +
//! FaceMesh) costs ~29 ms and the solver stage (visibility, hand crops,
//! dense surface, the fusion estimator, output) another ~19 ms — run
//! serially on one thread they cap the published pose rate at
//! ~1000 / (29 + 19) ≈ 20 Hz. This module moves the detector stage onto
//! its own thread so the stages overlap:
//!
//! ```text
//! capture ──> LatestCell ──> tracking-worker (solver thread)
//!                             │  submit job(frame N)      ┌───────────────┐
//!                             │  take freshest result      │ tracking-detect│
//!                             │  ─────────────────────────>│ (RTMW3D +     │
//!                             │  <─────────────────────────│  FaceMesh)    │
//!                             ▼  result(frame M), M ≤ N    └───────────────┘
//!                        solve(M) → publish
//! ```
//!
//! The result consumed at frame N is usually frame N−1's: the published
//! pose lags its capture by one frame period, which is the same effective
//! latency the serial loop already had — there the whole ~48 ms inference
//! delayed the pose past the next capture anyway. Throughput becomes
//! `max(detector, solver)` instead of their sum.
//!
//! Handoff semantics mirror the capture side's latest-only cell:
//! * the **request cell is latest-wins** — when the detector runs behind,
//!   a queued older job is superseded (that capture was already stale;
//!   the estimator bridges gaps from device timestamps), and
//! * the **result cell holds only the freshest completed result** — a
//!   solver that falls behind skips frames rather than queueing them.
//!
//! Frame identity travels in every job and result so the solver can pair
//! a consumed result with the metric depth captured under the same index
//! (`FusionProvider::depth_ring`). Protocol invariants:
//!
//! * `estimate_pose` (synchronous — warm-up, mid-session recording) waits
//!   for exactly its frame's result, which makes a remote detector
//!   behave like the inline path;
//! * `estimate_pose_latest` (the live path) never blocks: it submits the
//!   current frame and returns whatever finished since its last consume,
//!   or `None` when nothing is ready yet;
//! * `reset_temporal_state` round-trips an in-band request so the reset
//!   is ordered before any subsequently submitted job. It is only called
//!   between sessions (never mid-stream), which is what makes replacing
//!   a pending job with the reset request safe.
//!
//! All cells are `Mutex<Option<_>>` + `Condvar`: one wake-up per frame,
//! no busy waiting, nothing allocated beyond the job itself.

use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crate::tracking::rtmw3d::{Rtmw3dAux, Rtmw3dInference, Rtmw3dOptions};
use crate::tracking::{PoseEstimate, SourceSkeleton, DetectionAnnotation};

/// Construction outcome reported by the detector thread once its ONNX
/// sessions are built (the constructing thread holds the GPU-exclusive
/// guard until this arrives).
pub(crate) struct DetectorReady {
    pub backend_label: String,
    pub warnings: Vec<String>,
}

/// One frame's detector work, submitted by the solver thread.
pub(crate) struct DetectorJob {
    pub frame_index: u64,
    /// Device capture timestamp (ms), forwarded to
    /// `Rtmw3dInference::set_frame_timestamp_ms`.
    pub ts_ms: Option<f64>,
    pub rgb: Arc<Vec<u8>>,
    pub width: u32,
    pub height: u32,
    /// State-driven person crop hint computed by the solver for THIS
    /// frame — the same value the inline path would have passed.
    pub crop_hint: Option<crate::tracking::yolox::PersonBbox>,
}

/// One finished detector pass. `Rtmw3dAux` carries the raw keypoints +
/// FaceMesh landmarks that the inline path drains via `take_aux`.
pub(crate) struct DetectorResult {
    pub frame_index: u64,
    pub base: PoseEstimate,
    pub aux: Option<Rtmw3dAux>,
    /// Wall time the detector thread spent on this frame (job dequeue →
    /// outputs ready). The solver-side `ph.rtmw_ms` carries this on the
    /// pipelined path, where the work happens off the solver thread.
    pub det_ms: f32,
}

enum Request {
    Job(DetectorJob),
    /// Apply `Rtmw3dInference::reset_temporal_state`; acknowledged with a
    /// `frame_index == u64::MAX` result.
    Reset,
}

/// Latest-wins slot with a wake-up condvar. `T` needs no `Default`: the
/// slot starts empty.
struct Cell<T> {
    slot: Mutex<Option<T>>,
    signal: Condvar,
}

impl<T> Default for Cell<T> {
    fn default() -> Self {
        Self {
            slot: Mutex::new(None),
            signal: Condvar::new(),
        }
    }
}

impl<T> Cell<T> {
    /// Overwrite the slot (latest-wins) and wake a waiter.
    fn put(&self, value: T) {
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        *slot = Some(value);
        self.signal.notify_all();
    }

    fn take(&self) -> Option<T> {
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        slot.take()
    }

    /// Block until a value is queued or `stop` is set (`None`).
    fn wait_take(&self, stop: &Mutex<bool>) -> Option<T> {
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(value) = slot.take() {
                return Some(value);
            }
            if *stop.lock().unwrap_or_else(|e| e.into_inner()) {
                return None;
            }
            slot = self.signal.wait(slot).unwrap_or_else(|e| e.into_inner());
        }
    }
}

fn empty_estimate() -> PoseEstimate {
    PoseEstimate {
        skeleton: SourceSkeleton::default(),
        annotation: DetectionAnnotation::default(),
    }
}

/// Solver-side handle to the live detector thread.
pub(crate) struct DetectorClient {
    requests: Arc<Cell<Request>>,
    results: Arc<Cell<DetectorResult>>,
    stop: Arc<Mutex<bool>>,
    handle: Option<JoinHandle<()>>,
}

impl DetectorClient {
    /// Spawn the detector thread, which builds the RTMW3D / FaceMesh /
    /// YOLOX sessions itself and reports the outcome on `ready`. Spawning
    /// must happen while the caller holds the GPU-exclusive guard — the
    /// thread does its DirectML setup under the caller's guard.
    pub(crate) fn spawn(
        models_dir: PathBuf,
        opts: Rtmw3dOptions,
    ) -> (Self, Receiver<Result<DetectorReady, String>>) {
        let (ready_tx, ready_rx) = mpsc::channel();
        let requests = Arc::new(Cell::<Request>::default());
        let results = Arc::new(Cell::<DetectorResult>::default());
        let stop = Arc::new(Mutex::new(false));
        let handle = {
            let requests = Arc::clone(&requests);
            let results = Arc::clone(&results);
            let stop = Arc::clone(&stop);
            std::thread::Builder::new()
                .name("tracking-detect".into())
                .spawn(move || {
                    match Rtmw3dInference::from_models_dir_with_options(models_dir, opts) {
                        Ok(mut rtmw3d) => {
                            let _ = ready_tx.send(Ok(DetectorReady {
                                backend_label: rtmw3d.backend().label(),
                                warnings: rtmw3d.take_load_warnings(),
                            }));
                            run_detector(rtmw3d, &requests, &results, &stop);
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                        }
                    }
                })
                .expect("failed to spawn tracking-detect thread")
        };
        (
            Self {
                requests,
                results,
                stop,
                handle: Some(handle),
            },
            ready_rx,
        )
    }

    /// Queue this frame's detector work, superseding any queued older
    /// job. Never blocks.
    pub(crate) fn submit(&self, job: DetectorJob) {
        self.requests.put(Request::Job(job));
    }

    /// Freshest completed result strictly newer than the solver's
    /// high-water mark, or `None` — the caller skips the frame and the
    /// camera paces the retry. The cell holds at most one result (a newer
    /// completion overwrites an unconsumed older one), so no draining is
    /// needed here.
    pub(crate) fn take_latest(&self, after: Option<u64>) -> Option<DetectorResult> {
        let res = self.results.take()?;
        after.is_none_or(|after| res.frame_index > after).then_some(res)
    }

    /// Blocking wait for exactly `frame_index`'s result — the synchronous
    /// `estimate_pose` path (warm-up, recording), where the call must
    /// behave like the inline pipeline. Returns `None` if the detector
    /// thread died before producing it. Must not race a concurrent
    /// `submit` for a different frame (single solver thread guarantees
    /// this).
    pub(crate) fn wait_for(&self, frame_index: u64) -> Option<DetectorResult> {
        loop {
            let res = self.results.wait_take(&self.stop)?;
            if res.frame_index == frame_index {
                return Some(res);
            }
            // u64::MAX = a reset ack nobody is waiting for; anything else
            // is a result for another frame — both are dropped (the cell
            // is latest-wins, so this can only be a superseded frame).
        }
    }

    /// Queue a temporal-state reset and block until acknowledged. Any
    /// result produced before the ack is dropped: the reset invalidates
    /// the detector's temporal state, so pairing it with the solver's
    /// old frame would be wrong — the next submission re-detects from
    /// scratch anyway.
    pub(crate) fn reset_temporal_state(&self) {
        self.requests.put(Request::Reset);
        loop {
            match self.results.wait_take(&self.stop) {
                Some(DetectorResult {
                    frame_index: u64::MAX,
                    ..
                }) => return,
                Some(_) => continue,
                None => return, // detector thread gone
            }
        }
    }

    /// Detector thread health: `false` after the thread exited (stop or
    /// panic) — a synchronous caller uses this to degrade instead of
    /// blocking forever.
    pub(crate) fn alive(&self) -> bool {
        self.handle
            .as_ref()
            .is_some_and(|h| !h.is_finished())
    }
}

impl Drop for DetectorClient {
    fn drop(&mut self) {
        *self.stop.lock().unwrap_or_else(|e| e.into_inner()) = true;
        self.requests.signal.notify_all();
        self.results.signal.notify_all();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Body of the detector thread: drain requests, run the RTMW3D pass on
/// the freshest job, publish results. Exits when `stop` is set and no
/// request is pending.
fn run_detector(
    mut rtmw3d: Rtmw3dInference,
    requests: &Cell<Request>,
    results: &Cell<DetectorResult>,
    stop: &Mutex<bool>,
) {
    loop {
        let Some(req) = requests.wait_take(stop) else {
            break;
        };
        match req {
            Request::Reset => {
                rtmw3d.reset_temporal_state();
                results.put(DetectorResult {
                    frame_index: u64::MAX,
                    base: empty_estimate(),
                    aux: None,
                    det_ms: 0.0,
                });
            }
            Request::Job(job) => {
                rtmw3d.set_frame_timestamp_ms(job.ts_ms);
                rtmw3d.set_crop_hint(job.crop_hint);
                let t_det = std::time::Instant::now();
                let base = rtmw3d.estimate_pose(&job.rgb, job.width, job.height, job.frame_index);
                let aux = rtmw3d.take_aux();
                results.put(DetectorResult {
                    frame_index: job.frame_index,
                    base,
                    aux,
                    det_ms: t_det.elapsed().as_secs_f32() * 1000.0,
                });
            }
        }
    }
}
