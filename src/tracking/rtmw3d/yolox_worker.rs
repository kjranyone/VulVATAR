//! Async YOLOX person-detection worker.
//!
//! Runs `YoloxPersonDetector` on a dedicated thread with an mpsc
//! inbox + sticky `Arc<DetectResult>` outbox. The main RTMW3D
//! pipeline submits a frame on every `YOLOX_REFRESH_PERIOD`-th call
//! and reads the latest available result on every call — never
//! blocking on detection past the cold-start frame. Without this,
//! every `YOLOX_REFRESH_PERIOD`-th frame had a 33 ms hitch from a
//! synchronous YOLOX run; here the hitch is hidden behind the
//! worker.
//!
//! Cold start blocks once on the first result; subsequent frames just
//! clone the Arc.

use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use super::super::latest_cell::LatestCell;
use super::super::yolox::{PersonBbox, YoloxPersonDetector};

struct DetectRequest {
    rgb: Vec<u8>,
    width: u32,
    height: u32,
    generation: u64,
    /// Pipeline frame index of the submitted frame — carried through to
    /// [`DetectResult::frame_index`] so the consumer can age the sticky
    /// result.
    frame_index: u64,
    /// Device capture timestamp (ms) of the submitted frame, when the
    /// pipeline carries one — carried through to
    /// [`DetectResult::timestamp_ms`] so staleness is judged in wall
    /// time rather than frame counts.
    timestamp_ms: Option<f64>,
    /// Last known subject bbox (typically the most recent self-track
    /// crop). When several people clear the detector threshold, the
    /// candidate overlapping this region wins over the highest-score
    /// one — re-acquisition must find the SAME person, not whoever is
    /// currently largest in frame.
    prefer: Option<PersonBbox>,
}

pub(super) struct DetectResult {
    pub bbox: Option<PersonBbox>,
    pub generation: u64,
    /// Frame index the detection ran on. The sticky-outbox design means
    /// this can be arbitrarily old when no submissions happened for a
    /// while (self-track live); consumers MUST age-gate on it instead of
    /// trusting the bbox unconditionally.
    pub frame_index: u64,
    /// Device capture timestamp (ms) of the frame the detection ran on;
    /// `None` when the pipeline carries no device clock (synthetic
    /// inputs). The wall-time counterpart of `frame_index` for the
    /// age gate.
    pub timestamp_ms: Option<f64>,
}

pub(super) struct DetectOutbox {
    pub slot: Mutex<Option<Arc<DetectResult>>>,
    pub cvar: Condvar,
}

pub(super) struct YoloxWorker {
    inbox: Arc<LatestCell<DetectRequest>>,
    outbox: Arc<DetectOutbox>,
    thread: Option<thread::JoinHandle<()>>,
    /// Temporal-state generation; results from an older generation
    /// are in-flight leftovers of a previous input and are treated
    /// as absent.
    generation: u64,
}

impl YoloxWorker {
    pub fn spawn(detector: YoloxPersonDetector) -> Result<Self, String> {
        let inbox = LatestCell::new();
        let outbox = Arc::new(DetectOutbox {
            slot: Mutex::new(None),
            cvar: Condvar::new(),
        });
        let inbox_for_thread = Arc::clone(&inbox);
        let outbox_for_thread = Arc::clone(&outbox);
        let thread = thread::Builder::new()
            .name("yolox-detect".into())
            .spawn(move || worker_loop(detector, inbox_for_thread, outbox_for_thread))
            .map_err(|e| format!("spawn yolox worker: {e}"))?;
        Ok(Self {
            inbox,
            outbox,
            thread: Some(thread),
            generation: 0,
        })
    }

    /// Submit a new frame for detection. The latest-only inbox drops any
    /// still-pending frame so the worker always processes the freshest
    /// submission. Cheap (one Vec clone of the RGB buffer).
    pub fn submit(
        &self,
        rgb: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
        timestamp_ms: Option<f64>,
        prefer: Option<PersonBbox>,
    ) {
        self.inbox.put(DetectRequest {
            rgb: rgb.to_vec(),
            width,
            height,
            generation: self.generation,
            frame_index,
            timestamp_ms,
            prefer,
        });
    }

    /// Block until the very first detection lands, then return it.
    /// On every subsequent call returns the latest sticky result
    /// without blocking. Returns the same `Arc<DetectResult>` across
    /// frames until the worker writes a newer one.
    pub fn wait_latest(&self) -> Arc<DetectResult> {
        let mut slot = self.outbox.slot.lock().unwrap();
        while !matches!(&*slot, Some(r) if r.generation == self.generation) {
            slot = self.outbox.cvar.wait(slot).unwrap();
        }
        Arc::clone(slot.as_ref().unwrap())
    }

    /// Non-blocking peek. `None` only on cold start before the worker
    /// has produced anything. Used to decide whether to force a
    /// submit even outside the refresh period (cold-start safeguard).
    pub fn has_result(&self) -> bool {
        matches!(
            &*self.outbox.slot.lock().unwrap(),
            Some(r) if r.generation == self.generation
        )
    }

    /// Drop the sticky result, restoring cold-start semantics: the
    /// next pipeline call re-submits and blocks for a fresh
    /// detection. Part of `reset_temporal_state` — without this the
    /// sticky bbox from one input leaks into the next unrelated one
    /// (measured at up to 7° of solver error on the deskcrop
    /// validation set when image N runs inside image N-1's bbox).
    pub fn clear_result(&mut self) {
        *self.outbox.slot.lock().unwrap() = None;
        self.generation = self.generation.wrapping_add(1);
    }
}

impl Drop for YoloxWorker {
    fn drop(&mut self) {
        // Close before join: the worker blocks in `take_blocking`, which
        // only wakes on `close` (dropping the handle would deadlock).
        self.inbox.close();
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

fn worker_loop(
    mut detector: YoloxPersonDetector,
    inbox: Arc<LatestCell<DetectRequest>>,
    outbox: Arc<DetectOutbox>,
) {
    // `take_blocking` already yields only the latest submitted frame, so
    // no drain loop is needed; it returns `None` when the worker is closed.
    while let Some(req) = inbox.take_blocking() {
        let bbox = detector.detect_person(&req.rgb, req.width, req.height, req.prefer.as_ref());
        let result = Arc::new(DetectResult {
            bbox,
            generation: req.generation,
            frame_index: req.frame_index,
            timestamp_ms: req.timestamp_ms,
        });
        {
            let mut slot = outbox.slot.lock().unwrap();
            *slot = Some(result);
        }
        outbox.cvar.notify_all();
    }
}
