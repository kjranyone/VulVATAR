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
//! Same drain-old / sticky-Arc design as the DAv2 worker in
//! `rtmw3d_with_depth.rs`. Cold start blocks once on the first
//! result; subsequent frames just clone the Arc.

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use super::super::yolox::{PersonBbox, YoloxPersonDetector};

struct DetectRequest {
    rgb: Vec<u8>,
    width: u32,
    height: u32,
}

pub(super) struct DetectResult {
    pub bbox: Option<PersonBbox>,
}

pub(super) struct DetectOutbox {
    pub slot: Mutex<Option<Arc<DetectResult>>>,
    pub cvar: Condvar,
}

pub(super) struct YoloxWorker {
    tx: Option<Sender<DetectRequest>>,
    outbox: Arc<DetectOutbox>,
    thread: Option<thread::JoinHandle<()>>,
}

impl YoloxWorker {
    pub fn spawn(detector: YoloxPersonDetector) -> Result<Self, String> {
        let (tx, rx) = channel::<DetectRequest>();
        let outbox = Arc::new(DetectOutbox {
            slot: Mutex::new(None),
            cvar: Condvar::new(),
        });
        let outbox_for_thread = Arc::clone(&outbox);
        let thread = thread::Builder::new()
            .name("yolox-detect".into())
            .spawn(move || worker_loop(detector, rx, outbox_for_thread))
            .map_err(|e| format!("spawn yolox worker: {e}"))?;
        Ok(Self {
            tx: Some(tx),
            outbox,
            thread: Some(thread),
        })
    }

    /// Submit a new frame for detection. Drops older queued requests
    /// in the worker so the worker always processes the latest. Cheap
    /// (one Vec clone of the RGB buffer + channel send).
    pub fn submit(&self, rgb: &[u8], width: u32, height: u32) {
        if let Some(tx) = self.tx.as_ref() {
            let _ = tx.send(DetectRequest {
                rgb: rgb.to_vec(),
                width,
                height,
            });
        }
    }

    /// Block until the very first detection lands, then return it.
    /// On every subsequent call returns the latest sticky result
    /// without blocking. Returns the same `Arc<DetectResult>` across
    /// frames until the worker writes a newer one.
    pub fn wait_latest(&self) -> Arc<DetectResult> {
        let mut slot = self.outbox.slot.lock().unwrap();
        while slot.is_none() {
            slot = self.outbox.cvar.wait(slot).unwrap();
        }
        Arc::clone(slot.as_ref().unwrap())
    }

    /// Non-blocking peek. `None` only on cold start before the worker
    /// has produced anything. Used to decide whether to force a
    /// submit even outside the refresh period (cold-start safeguard).
    pub fn has_result(&self) -> bool {
        self.outbox.slot.lock().unwrap().is_some()
    }
}

impl Drop for YoloxWorker {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

fn worker_loop(
    mut detector: YoloxPersonDetector,
    rx: Receiver<DetectRequest>,
    outbox: Arc<DetectOutbox>,
) {
    while let Ok(mut req) = rx.recv() {
        // Drain: skip any older queued requests. We only ever care
        // about the latest frame.
        while let Ok(newer) = rx.try_recv() {
            req = newer;
        }
        let bbox = detector.detect_largest_person(&req.rgb, req.width, req.height);
        let result = Arc::new(DetectResult { bbox });
        {
            let mut slot = outbox.slot.lock().unwrap();
            *slot = Some(result);
        }
        outbox.cvar.notify_all();
    }
}
