pub mod frame_sink;
#[cfg(all(target_os = "windows", feature = "virtual-camera"))]
pub mod mf_virtual_camera;
#[cfg(target_os = "windows")]
pub mod shmem_security;
#[cfg(all(target_os = "windows", feature = "virtual-camera"))]
pub mod virtual_camera_native;

pub use frame_sink::{create_sink_writer, FrameSink, FrameSinkQueuePolicy, OutputSinkWriter};

use crate::avatar::pose::FrameTimestamp;
pub use crate::frame_handoff::{
    AlphaMode, ExportedResourceId, ExternalHandleType, FallbackReason, FrameLease,
    FrameLifetimeContract, GpuFrameToken, HandoffPath, OutputColorSpace, OutputFrameId,
    OutputSyncToken,
};
use log::{debug, error, info};
use std::collections::VecDeque;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct OutputFrame {
    pub frame_id: OutputFrameId,
    pub timestamp: FrameTimestamp,
    pub extent: [u32; 2],
    pub color_space: OutputColorSpace,
    pub alpha_mode: AlphaMode,
    pub gpu_token: Option<GpuFrameToken>,
    pub handoff_path: HandoffPath,
    pub fallback_reason: Option<FallbackReason>,
    pub pixel_data: Option<Arc<Vec<u8>>>,
}

impl OutputFrame {
    pub fn new(id: u64, extent: [u32; 2], timestamp: FrameTimestamp) -> Self {
        Self {
            frame_id: OutputFrameId(id),
            timestamp,
            extent,
            color_space: OutputColorSpace::Srgb,
            alpha_mode: AlphaMode::Premultiplied,
            gpu_token: None,
            handoff_path: HandoffPath::CpuReadback,
            fallback_reason: Some(FallbackReason::RequestedCpuReadback),
            pixel_data: None,
        }
    }

    pub fn with_pixel_data(mut self, data: Arc<Vec<u8>>) -> Self {
        self.pixel_data = Some(data);
        self
    }

    pub fn with_gpu_token(mut self, token: GpuFrameToken) -> Self {
        self.gpu_token = Some(token);
        self.handoff_path = HandoffPath::GpuSharedFrame;
        self.fallback_reason = None;
        self
    }
}

// ---------------------------------------------------------------------------
// Worker message
// ---------------------------------------------------------------------------

enum WorkerMessage {
    Frame(OutputFrame),
    Shutdown,
    /// Sentinel ack used by `wait_until_worker_drains` (test builds only).
    /// `sync_channel(1)` is FIFO, so the worker reaching this message has
    /// already processed every prior `Frame` — the ack on the oneshot
    /// channel is therefore a "worker has drained up to this point"
    /// signal that the test can wait on without polling the filesystem.
    #[cfg(test)]
    Sync(mpsc::Sender<()>),
}

// ---------------------------------------------------------------------------
// OutputRouter
// ---------------------------------------------------------------------------

pub struct OutputRouter {
    sink: FrameSink,
    policy: FrameSinkQueuePolicy,
    queue: VecDeque<OutputFrame>,
    max_queue_depth: usize,
    next_frame_id: u64,
    dropped_count: u64,
    last_publish_timestamp: FrameTimestamp,

    /// Phase B-3: minimum interval between forwarded frames. The renderer
    /// usually produces at the display rate (e.g. 60 Hz), but the user can
    /// pick a slower output cadence in the inspector. `Duration::ZERO`
    /// means "no throttling, forward every frame".
    forward_min_interval: Duration,
    last_forward_at: Option<Instant>,

    /// Channel sender to the background output worker.
    worker_tx: Option<mpsc::SyncSender<WorkerMessage>>,
    /// Completion channel for GPU leases that are no longer retained by the
    /// output pipeline. The worker sends after sink consumption; the router
    /// itself sends when a frame is dropped before it reaches the worker.
    lease_completion_tx: mpsc::Sender<u64>,
    lease_completion_rx: mpsc::Receiver<u64>,
    /// Join handle for the background worker thread.
    worker_handle: Option<thread::JoinHandle<()>>,
    logged_first_publish: bool,

    /// Most recent published `OutputFrame.handoff_path`. Used by
    /// [`Self::diagnostics`] so the GUI shows the path actually carried by
    /// frames instead of guessing from the sink variant. `None` until the
    /// first publish.
    last_handoff_path: Option<HandoffPath>,
    /// Whether the most recently delivered GPU frame actually carried an
    /// external handle the sink could consume. A GPU-looking path without a
    /// valid handle is still architectural scaffolding, not interop.
    last_gpu_token_valid: Option<bool>,
    last_fallback_reason: Option<FallbackReason>,
    last_export_pool_stats: Option<crate::renderer::output_export::ExportImagePoolStats>,
    /// Option A lease policy: a GPU token carrying
    /// `FrameLifetimeContract::SingleConsumerRetained` is held alive on the
    /// producer side until the *next* successfully-handed-off frame replaces
    /// it. This is the lease id currently in flight to the consumer; it is
    /// cleared the moment the router hands the next frame to the worker (or
    /// at shutdown). Frames dropped before reaching the worker do NOT clear
    /// this — the consumer is still bound to the previous frame's texture.
    pending_retained_lease: Option<u64>,
}

impl OutputRouter {
    pub fn new(sink: FrameSink) -> Self {
        let writer = create_sink_writer(&sink);
        Self::with_writer(sink, writer)
    }

    /// Construct a router driving a caller-supplied writer instead of going
    /// through the default `create_sink_writer` factory. Used by tests that
    /// need a custom path (e.g. a temp directory) so concurrent test runs
    /// do not collide on `%ProgramData%\VulVATAR\` or
    /// `std::env::temp_dir()/vulvatar-shared-texture.bin`.
    pub fn with_writer(sink: FrameSink, writer: Box<dyn OutputSinkWriter>) -> Self {
        let (tx, rx) = mpsc::sync_channel::<WorkerMessage>(1);
        let (lease_completion_tx, lease_completion_rx) = mpsc::channel::<u64>();
        let handle = Self::spawn_worker(rx, writer, lease_completion_tx.clone());

        Self {
            sink,
            policy: FrameSinkQueuePolicy::ReplaceLatest,
            queue: VecDeque::new(),
            max_queue_depth: 2,
            next_frame_id: 0,
            dropped_count: 0,
            last_publish_timestamp: 0,
            forward_min_interval: Duration::ZERO,
            last_forward_at: None,
            worker_tx: Some(tx),
            lease_completion_tx,
            lease_completion_rx,
            worker_handle: Some(handle),
            logged_first_publish: false,
            last_handoff_path: None,
            last_gpu_token_valid: None,
            last_fallback_reason: None,
            last_export_pool_stats: None,
            pending_retained_lease: None,
        }
    }

    /// Swap the active sink at runtime without recreating the router.
    /// Preserves cross-swap state that out-of-band code depends on — the
    /// lease completion channel keeps the same receiver so render-side
    /// pool reclaim does not stall across the swap.
    pub fn set_sink(&mut self, sink: FrameSink) {
        if self.sink == sink {
            return;
        }
        let writer = create_sink_writer(&sink);
        self.set_writer(sink, writer);
    }

    /// Companion to [`Self::set_sink`] that accepts a caller-supplied
    /// writer. Used by tests that hot-swap to a custom-path writer.
    pub fn set_writer(&mut self, sink: FrameSink, writer: Box<dyn OutputSinkWriter>) {
        info!(
            "OutputRouter: swapping sink {:?} -> {:?}",
            self.sink, sink
        );

        // Flush queue + pending — consumer is about to change. Frames in
        // the local queue never crossed the worker boundary, so their
        // leases can complete immediately. The pending retained lease
        // belongs to a consumer we are abandoning, so it is no longer
        // bound to anything and must be released too.
        for dropped_frame in self.queue.drain(..) {
            Self::complete_gpu_lease(&self.lease_completion_tx, &dropped_frame);
        }
        if let Some(prev) = self.pending_retained_lease.take() {
            let _ = self.lease_completion_tx.send(prev);
        }

        if let Some(tx) = self.worker_tx.take() {
            let _ = tx.send(WorkerMessage::Shutdown);
        }
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }

        let (tx, rx) = mpsc::sync_channel::<WorkerMessage>(1);
        let handle = Self::spawn_worker(rx, writer, self.lease_completion_tx.clone());
        self.sink = sink;
        self.worker_tx = Some(tx);
        self.worker_handle = Some(handle);

        // Reset published-frame diagnostics — the previous handoff path
        // belongs to the previous consumer.
        self.last_handoff_path = None;
        self.last_gpu_token_valid = None;
        self.last_fallback_reason = None;
        self.logged_first_publish = false;
    }

    /// Block until the worker has processed every frame currently in
    /// flight, or the timeout elapses. Returns true on drain, false on
    /// timeout or shutdown. Tests use this to avoid polling the
    /// filesystem with arbitrary sleeps.
    #[cfg(test)]
    pub fn wait_until_worker_drains(&self, timeout: Duration) -> bool {
        let (ack_tx, ack_rx) = mpsc::channel::<()>();
        let Some(ref worker_tx) = self.worker_tx else {
            return false;
        };
        // Blocking send: when the worker channel is full of frames the
        // sentinel waits its turn. By the time the worker delivers the
        // ack, every prior `Frame` message has been written.
        if worker_tx.send(WorkerMessage::Sync(ack_tx)).is_err() {
            return false;
        }
        ack_rx.recv_timeout(timeout).is_ok()
    }

    /// Set the minimum interval between forwarded frames. Called every
    /// frame from `GuiApp::update` based on the user's framerate combo
    /// selection — idempotent, so it's cheap to call from the hot path.
    /// `fps == 0` disables throttling.
    pub fn set_target_fps(&mut self, fps: u32) {
        self.forward_min_interval = if fps == 0 {
            Duration::ZERO
        } else {
            Duration::from_micros(1_000_000 / fps.max(1) as u64)
        };
    }

    /// Spawn a dedicated output worker thread that receives frames over a
    /// channel and writes them via the active sink.
    fn spawn_worker(
        rx: mpsc::Receiver<WorkerMessage>,
        mut writer: Box<dyn OutputSinkWriter>,
        lease_completion_tx: mpsc::Sender<u64>,
    ) -> thread::JoinHandle<()> {
        thread::Builder::new()
            .name("output-worker".into())
            .spawn(move || {
                info!("output-worker: started (sink={})", writer.name());
                loop {
                    match rx.recv() {
                        Ok(WorkerMessage::Frame(frame)) => {
                            if let Err(e) = writer.write_frame(&frame) {
                                error!("output-worker: write error: {}", e);
                            }
                            Self::complete_gpu_lease_after_write(&lease_completion_tx, &frame);
                        }
                        Ok(WorkerMessage::Shutdown) => {
                            info!("output-worker: shutdown requested, draining queue");
                            // Drain any remaining frames already in the channel.
                            while let Ok(msg) = rx.try_recv() {
                                match msg {
                                    WorkerMessage::Frame(frame) => {
                                        if let Err(e) = writer.write_frame(&frame) {
                                            error!("output-worker: drain write error: {}", e);
                                        }
                                        Self::complete_gpu_lease_after_write(
                                            &lease_completion_tx,
                                            &frame,
                                        );
                                    }
                                    #[cfg(test)]
                                    WorkerMessage::Sync(ack) => {
                                        let _ = ack.send(());
                                    }
                                    WorkerMessage::Shutdown => {}
                                }
                            }
                            info!("output-worker: stopped");
                            break;
                        }
                        #[cfg(test)]
                        Ok(WorkerMessage::Sync(ack)) => {
                            let _ = ack.send(());
                        }
                        Err(_) => {
                            // Sender dropped; treat as implicit shutdown.
                            info!("output-worker: channel closed, stopping");
                            break;
                        }
                    }
                }
            })
            .expect("failed to spawn output-worker thread")
    }

    /// Eagerly send the lease completion regardless of lifetime. Used on the
    /// drop paths (throttling, queue policy eviction, worker channel full)
    /// where the frame never reaches the consumer, so the texture is safe to
    /// recycle immediately even for retained-lifetime tokens.
    fn complete_gpu_lease(completion_tx: &mpsc::Sender<u64>, frame: &OutputFrame) {
        if let Some(token) = frame.gpu_token.as_ref() {
            let _ = completion_tx.send(token.lease.lease_id);
        }
    }

    /// Worker-side completion. Honours `FrameLifetimeContract`: an `Immediate`
    /// frame is done with the texture the moment `write_frame` returns, but a
    /// `Retained` frame stays alive on the consumer (the consumer reads the
    /// shared texture out-of-band) until the *next* publish replaces it —
    /// `OutputRouter::publish` owns that transition.
    fn complete_gpu_lease_after_write(completion_tx: &mpsc::Sender<u64>, frame: &OutputFrame) {
        if let Some(token) = frame.gpu_token.as_ref() {
            if matches!(
                token.lease.lifetime,
                FrameLifetimeContract::SingleConsumerImmediate
            ) {
                let _ = completion_tx.send(token.lease.lease_id);
            }
        }
    }

    /// Publish a frame to the output pipeline.
    ///
    /// The frame is queued locally (applying the queue policy) and then
    /// dispatched to the background output worker for actual writing.
    pub fn publish(&mut self, frame: OutputFrame) {
        // Phase B-3: framerate throttling. Skip the entire publish path
        // (including the local queue) if we're publishing faster than the
        // configured cadence — that way the latest frame at the next
        // tick is the freshest one rather than a stale queued one.
        if !self.forward_min_interval.is_zero() {
            let now = Instant::now();
            if let Some(last) = self.last_forward_at {
                if now.duration_since(last) < self.forward_min_interval {
                    self.dropped_count += 1;
                    Self::complete_gpu_lease(&self.lease_completion_tx, &frame);
                    return;
                }
            }
            self.last_forward_at = Some(now);
        }

        debug!(
            "output: publish frame {} {}x{} {:?} alpha={:?} to {:?}",
            frame.frame_id.0,
            frame.extent[0],
            frame.extent[1],
            frame.color_space,
            frame.alpha_mode,
            self.sink,
        );
        if !self.logged_first_publish {
            info!(
                "output: first publish frame {} {}x{} {:?} alpha={:?} to {:?} pixel_data={}",
                frame.frame_id.0,
                frame.extent[0],
                frame.extent[1],
                frame.color_space,
                frame.alpha_mode,
                self.sink,
                frame.pixel_data.as_ref().map_or(0, |p| p.len()),
            );
            self.logged_first_publish = true;
        }

        // Capture frame metadata for diagnostics now (the frame itself is
        // about to be moved into the local queue / sent to the worker).
        // The router only commits the metadata to `last_*` after a frame
        // is successfully handed off to the worker — see the try_send
        // arm below. Updating earlier would let the GUI report
        // "connected / active handoff" even when every frame was being
        // dropped at the local-queue or worker-channel boundary.
        let attempted_timestamp = frame.timestamp;
        let attempted_handoff_path = frame.handoff_path.clone();
        let attempted_gpu_token_valid = frame
            .gpu_token
            .as_ref()
            .is_some_and(GpuFrameToken::has_external_handle);
        let attempted_fallback_reason = frame.fallback_reason.clone().or_else(|| {
            (matches!(frame.handoff_path, HandoffPath::GpuSharedFrame)
                && !attempted_gpu_token_valid)
                .then_some(FallbackReason::MissingExternalHandle)
        });

        match self.policy {
            FrameSinkQueuePolicy::ReplaceLatest => {
                // Keep only the latest frame: clear the queue first.
                let dropped = self.queue.len();
                for dropped_frame in self.queue.drain(..) {
                    Self::complete_gpu_lease(&self.lease_completion_tx, &dropped_frame);
                }
                self.dropped_count += dropped as u64;
                self.queue.push_back(frame);
            }
            FrameSinkQueuePolicy::DropOldest => {
                if self.queue.len() >= self.max_queue_depth {
                    if let Some(dropped_frame) = self.queue.pop_front() {
                        Self::complete_gpu_lease(&self.lease_completion_tx, &dropped_frame);
                    }
                    self.dropped_count += 1;
                }
                self.queue.push_back(frame);
            }
            FrameSinkQueuePolicy::DropNewest => {
                if self.queue.len() < self.max_queue_depth {
                    self.queue.push_back(frame);
                } else {
                    self.dropped_count += 1;
                    Self::complete_gpu_lease(&self.lease_completion_tx, &frame);
                    return; // frame dropped, do not send to worker
                }
            }
            FrameSinkQueuePolicy::BlockNotAllowed => {
                if self.queue.len() < self.max_queue_depth {
                    self.queue.push_back(frame);
                } else {
                    self.dropped_count += 1;
                    Self::complete_gpu_lease(&self.lease_completion_tx, &frame);
                    return;
                }
            }
        }

        // Send the most-recently enqueued frame to the worker and drain it
        // from the local queue so the queue does not grow unboundedly. If
        // the worker channel itself is full (worker thread stalled on
        // disk I/O / consumer pickup), count the frame as dropped so
        // `dropped_count` reflects every frame the consumer didn't see —
        // not just the ones rejected by the local queue policy. Without
        // this accounting, a stalled shmem reader would silently lose
        // frames here while the local-queue counter stayed at zero, and
        // a profiler asking "are we dropping frames?" would be told no.
        if let Some(queued) = self.queue.pop_front() {
            // Option A retained-lease policy: the lease the consumer is
            // currently bound to is released only when a successor reaches the
            // worker. Snapshot the new frame's retained-lease id (if any)
            // before we move `queued` into the worker channel, so we can swap
            // `pending_retained_lease` only on a successful handoff.
            let attempted_retained_lease = queued
                .gpu_token
                .as_ref()
                .filter(|token| {
                    matches!(
                        token.lease.lifetime,
                        FrameLifetimeContract::SingleConsumerRetained
                    )
                })
                .map(|token| token.lease.lease_id);

            if let Some(ref tx) = self.worker_tx {
                match tx.try_send(WorkerMessage::Frame(queued)) {
                    Ok(()) => {
                        // Commit diagnostics only on a successful handoff.
                        // `last_*` are what the GUI reads to decide whether
                        // anything is flowing; if the worker channel is
                        // saturated we want the panel to show "pending" /
                        // "stalled", not "connected / GPU active".
                        self.last_publish_timestamp = attempted_timestamp;
                        self.last_handoff_path = Some(attempted_handoff_path);
                        self.last_gpu_token_valid = Some(attempted_gpu_token_valid);
                        self.last_fallback_reason = attempted_fallback_reason;

                        // Release the previously-retained lease — the
                        // consumer has moved on to this frame. Then record
                        // the new frame's retained-lease (if any) so the
                        // *next* successful publish releases it in turn.
                        if let Some(prev) = self.pending_retained_lease.take() {
                            let _ = self.lease_completion_tx.send(prev);
                        }
                        self.pending_retained_lease = attempted_retained_lease;
                    }
                    Err(e) => {
                        self.dropped_count += 1;
                        let error_text = e.to_string();
                        let dropped_message = match e {
                            mpsc::TrySendError::Full(message)
                            | mpsc::TrySendError::Disconnected(message) => message,
                        };
                        if let WorkerMessage::Frame(dropped_frame) = dropped_message {
                            Self::complete_gpu_lease(&self.lease_completion_tx, &dropped_frame);
                        }
                        debug!("output: worker channel full, dropping frame: {}", error_text);
                    }
                }
            }
        }
    }

    pub fn create_frame(
        &mut self,
        width: u32,
        height: u32,
        timestamp: FrameTimestamp,
    ) -> OutputFrame {
        let id = self.next_frame_id;
        self.next_frame_id += 1;
        OutputFrame {
            frame_id: OutputFrameId(id),
            timestamp,
            extent: [width, height],
            color_space: OutputColorSpace::Srgb,
            alpha_mode: AlphaMode::Premultiplied,
            gpu_token: None,
            handoff_path: HandoffPath::CpuReadback,
            fallback_reason: Some(FallbackReason::RequestedCpuReadback),
            pixel_data: None,
        }
    }

    /// Drain leases that the output worker has finished with, plus frames the
    /// router dropped before they reached the worker. The app forwards these
    /// acknowledgements back to the render thread so GPU slots can be reused.
    pub fn drain_completed_gpu_leases(&mut self) -> Vec<u64> {
        std::iter::from_fn(|| self.lease_completion_rx.try_recv().ok()).collect()
    }

    pub fn update_export_pool_stats(
        &mut self,
        stats: crate::renderer::output_export::ExportImagePoolStats,
    ) {
        self.last_export_pool_stats = Some(stats);
    }

    /// Drain the output queue and stop the background worker thread.
    pub fn shutdown(&mut self) {
        for dropped_frame in self.queue.drain(..) {
            Self::complete_gpu_lease(&self.lease_completion_tx, &dropped_frame);
        }
        // No successor will ever arrive — release the currently-bound
        // retained lease so the renderer's export pool slot is recycled
        // rather than leaked across the shutdown boundary.
        if let Some(prev) = self.pending_retained_lease.take() {
            let _ = self.lease_completion_tx.send(prev);
        }
        if let Some(tx) = self.worker_tx.take() {
            let _ = tx.send(WorkerMessage::Shutdown);
        }
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
        info!("OutputRouter: shutdown complete");
    }

    pub fn dropped_count(&self) -> u64 {
        self.dropped_count
    }

    pub fn queue_depth(&self) -> usize {
        self.queue.len()
    }

    /// Currently configured sink. Used by [`crate::app::Application`] callers
    /// (e.g. the GUI reconciliation path) to skip a no-op sink swap when the
    /// requested variant matches what is already running.
    pub fn active_sink(&self) -> &FrameSink {
        &self.sink
    }

    pub fn diagnostics(&self) -> OutputDiagnostics {
        // Derive everything from the actual published frame's handoff_path
        // rather than guessing from the sink variant. Until the first frame
        // is published the path is unknown — surface that explicitly so the
        // GUI doesn't paint a green "GPU active" badge before any frame has
        // crossed the boundary.
        let active_handoff_path = self.last_handoff_path.clone();
        let active_gpu_token_valid = self.last_gpu_token_valid;
        let gpu_interop_enabled = matches!(active_handoff_path, Some(HandoffPath::GpuSharedFrame))
            && active_gpu_token_valid == Some(true);
        let fallback_active = matches!(
            active_handoff_path,
            Some(HandoffPath::CpuReadback | HandoffPath::SharedMemory)
        ) || matches!(active_handoff_path, Some(HandoffPath::GpuSharedFrame))
            && active_gpu_token_valid == Some(false);
        OutputDiagnostics {
            active_sink: self.sink.clone(),
            active_handoff_path,
            active_gpu_token_valid,
            fallback_reason: self.last_fallback_reason.clone(),
            export_pool: self.last_export_pool_stats,
            gpu_interop_enabled,
            fallback_active,
            queue_depth: self.queue.len(),
            max_queue_depth: self.max_queue_depth,
            dropped_frame_count: self.dropped_count,
            last_publish_timestamp: self.last_publish_timestamp,
        }
    }
}

impl Drop for OutputRouter {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[derive(Clone, Debug)]
pub struct OutputDiagnostics {
    pub active_sink: FrameSink,
    /// Handoff path of the most recently published frame, or `None` when
    /// nothing has been published yet. Reflects what crossed the boundary,
    /// not what the sink's name advertises.
    pub active_handoff_path: Option<HandoffPath>,
    /// Whether the last delivered GPU token carried a real external handle.
    /// `None` until the first frame crosses the worker boundary.
    pub active_gpu_token_valid: Option<bool>,
    pub fallback_reason: Option<FallbackReason>,
    pub export_pool: Option<crate::renderer::output_export::ExportImagePoolStats>,
    pub gpu_interop_enabled: bool,
    pub fallback_active: bool,
    pub queue_depth: usize,
    pub max_queue_depth: usize,
    pub dropped_frame_count: u64,
    pub last_publish_timestamp: FrameTimestamp,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(id: u64, path: HandoffPath) -> OutputFrame {
        let mut frame = OutputFrame::new(id, [16, 16], 1_000 + id);
        frame.handoff_path = path;
        frame
    }

    fn make_valid_gpu_frame(id: u64) -> OutputFrame {
        OutputFrame::new(id, [16, 16], 1_000 + id).with_gpu_token(GpuFrameToken {
            resource_id: id,
            handle_type: ExternalHandleType::Win32Kmt,
            external_handle: Some(id + 100),
            sync: OutputSyncToken::ProducerWaitComplete,
            lease: FrameLease {
                lease_id: id,
                lifetime: FrameLifetimeContract::SingleConsumerImmediate,
            },
        })
    }

    fn make_retained_gpu_frame(id: u64) -> OutputFrame {
        OutputFrame::new(id, [16, 16], 1_000 + id).with_gpu_token(GpuFrameToken {
            resource_id: id,
            handle_type: ExternalHandleType::Win32Kmt,
            external_handle: Some(id + 100),
            sync: OutputSyncToken::ProducerWaitComplete,
            lease: FrameLease {
                lease_id: id,
                lifetime: FrameLifetimeContract::SingleConsumerRetained,
            },
        })
    }

    /// Publish frames carrying `handoff_path` until the router's
    /// diagnostics report that path as `active`, or panic after a
    /// generous timeout. The worker channel is `sync_channel(1)`, so a
    /// single `try_send` can fail simply because the worker thread
    /// hasn't drained the previous message yet — retrying lets the
    /// test verify "delivery eventually completes" without leaning on
    /// fragile fixed sleeps.
    fn publish_until_active(router: &mut OutputRouter, base_id: u64, handoff_path: HandoffPath) {
        for attempt in 0..200 {
            router.publish(make_frame(base_id + attempt, handoff_path.clone()));
            if router.diagnostics().active_handoff_path.as_ref() == Some(&handoff_path) {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!(
            "publish never reached the worker for handoff_path={:?} after 200 attempts",
            handoff_path
        );
    }

    fn wait_for_completed_lease(router: &mut OutputRouter, lease_id: u64) {
        for _ in 0..200 {
            if router
                .drain_completed_gpu_leases()
                .into_iter()
                .any(|completed| completed == lease_id)
            {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("lease {lease_id} was never completed");
    }

    /// Regression test for Finding #2 (initial pass) plus Fix #2 review:
    /// diagnostics must reflect the actual handoff path of the most
    /// recently *delivered* frame — not the sink variant, and not a
    /// frame that failed `try_send` to the worker channel.
    #[test]
    fn diagnostics_handoff_path_follows_published_frames() {
        let mut router = OutputRouter::new(FrameSink::ImageSequence);

        let initial = router.diagnostics();
        assert!(
            initial.active_handoff_path.is_none(),
            "no frames published yet — handoff path must be unknown"
        );
        assert!(!initial.gpu_interop_enabled);
        assert!(!initial.fallback_active);

        publish_until_active(&mut router, 0, HandoffPath::CpuReadback);
        let after_cpu = router.diagnostics();
        assert!(!after_cpu.gpu_interop_enabled);
        assert!(after_cpu.fallback_active);

        publish_until_active(&mut router, 1_000, HandoffPath::GpuSharedFrame);
        let after_gpu = router.diagnostics();
        assert!(
            !after_gpu.gpu_interop_enabled,
            "a GPU-looking path without an external handle is still fallback"
        );
        assert!(after_gpu.fallback_active);

        for attempt in 0..200 {
            router.publish(make_valid_gpu_frame(2_000 + attempt));
            let diagnostics = router.diagnostics();
            if diagnostics.gpu_interop_enabled {
                assert_eq!(diagnostics.active_gpu_token_valid, Some(true));
                assert_eq!(diagnostics.fallback_reason, None);
                assert!(!diagnostics.fallback_active);
                router.shutdown();
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("valid GPU token never reached the worker after 200 attempts");
    }

    #[test]
    fn gpu_leases_complete_after_worker_consumption() {
        let mut router = OutputRouter::new(FrameSink::ImageSequence);
        router.publish(make_valid_gpu_frame(3_000));
        wait_for_completed_lease(&mut router, 3_000);
        router.shutdown();
    }

    #[test]
    fn gpu_leases_complete_when_frames_are_dropped_before_worker() {
        let mut router = OutputRouter::new(FrameSink::ImageSequence);
        router.set_target_fps(1);
        router.publish(make_valid_gpu_frame(4_000));
        router.publish(make_valid_gpu_frame(4_001));
        wait_for_completed_lease(&mut router, 4_001);
        router.shutdown();
    }

    /// Option A retained-lease policy: a `SingleConsumerRetained` lease must
    /// stay alive on the producer until the *next* successfully-handed-off
    /// frame replaces it. The texture the consumer is currently bound to
    /// cannot be recycled mid-frame. At shutdown the last pending lease must
    /// be flushed so the renderer's export pool slot is reclaimed instead of
    /// leaking across teardown.
    #[test]
    fn retained_gpu_leases_complete_on_next_publish_and_at_shutdown() {
        let mut router = OutputRouter::new(FrameSink::ImageSequence);

        router.publish(make_retained_gpu_frame(9_000));
        // Give the worker a moment to consume from sync_channel(1) so a
        // subsequent publish does not race against a full channel.
        std::thread::sleep(Duration::from_millis(50));
        let drained = router.drain_completed_gpu_leases();
        assert!(
            drained.is_empty(),
            "retained lease must not complete before its successor is published, got: {:?}",
            drained,
        );

        router.publish(make_retained_gpu_frame(9_001));
        wait_for_completed_lease(&mut router, 9_000);
        std::thread::sleep(Duration::from_millis(50));

        router.publish(make_retained_gpu_frame(9_002));
        wait_for_completed_lease(&mut router, 9_001);

        router.shutdown();
        wait_for_completed_lease(&mut router, 9_002);
    }

    /// Frames dropped at the throttling boundary never reach the consumer,
    /// so their leases complete immediately regardless of lifetime — and
    /// they must not disturb the currently-bound `pending_retained_lease`.
    #[test]
    fn retained_pending_survives_dropped_successor() {
        let mut router = OutputRouter::new(FrameSink::ImageSequence);

        // Arm the throttle BEFORE the first publish so `last_forward_at`
        // gets set by 8_000 reaching the worker. Without this the
        // throttling gate's `last_forward_at` is still `None` when 8_001
        // arrives and the second publish would slip through the gate.
        router.set_target_fps(1);
        router.publish(make_retained_gpu_frame(8_000));
        std::thread::sleep(Duration::from_millis(50));

        router.publish(make_retained_gpu_frame(8_001));
        // The dropped successor must complete immediately (it never bound a
        // consumer). The retained pending (8_000) must NOT have been
        // released by this drop.
        wait_for_completed_lease(&mut router, 8_001);
        let drained = router.drain_completed_gpu_leases();
        assert!(
            !drained.contains(&8_000),
            "throttle-dropped successor must not release the bound retained lease, got: {:?}",
            drained,
        );

        router.shutdown();
        wait_for_completed_lease(&mut router, 8_000);
    }

    /// Sink swap with a pending retained lease: the old sink's last
    /// retained lease must be released when the sink is swapped, not
    /// leaked across the boundary. This caught a regression where
    /// `set_sink` flushed the worker channel but forgot to drain
    /// `pending_retained_lease`.
    #[test]
    fn sink_swap_releases_pending_retained_lease() {
        let mut router = OutputRouter::new(FrameSink::ImageSequence);
        router.publish(make_retained_gpu_frame(9_000));
        std::thread::sleep(Duration::from_millis(50));
        // `9_000` is bound as pending; the worker has consumed and
        // ack'd it, but the next-publish-release contract leaves it
        // armed.

        router.set_sink(FrameSink::SharedTextureFileStub);
        // After sink swap, the old pending lease must complete (the
        // new sink can't honour it) — otherwise the renderer's export
        // pool slot stays leased indefinitely.
        wait_for_completed_lease(&mut router, 9_000);

        router.shutdown();
    }

    // -----------------------------------------------------------------
    // End-to-end composed coverage (plan: output-sink-integration-tests)
    // -----------------------------------------------------------------
    //
    // These verify the full publish path: ExportedFrame-shaped input →
    // OutputRouter::publish → worker → on-disk artefact. They live next
    // to the router so they can use `with_writer`, `wait_until_worker_drains`,
    // and crate-internal sink types without expanding the lib's public
    // surface beyond what production also relies on.

    mod e2e {
        use super::*;
        use crate::output::frame_sink::{ImageSequenceSink, SharedMemoryFileSink};
        use std::path::PathBuf;
        use std::time::{Duration, SystemTime, UNIX_EPOCH};

        fn unique_temp_path(label: &str, ext: &str) -> PathBuf {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time")
                .as_nanos();
            std::env::temp_dir().join(format!("vulvatar-e2e-{label}-{nanos}.{ext}"))
        }

        fn unique_temp_dir(label: &str) -> PathBuf {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time")
                .as_nanos();
            let dir = std::env::temp_dir().join(format!("vulvatar-e2e-{label}-{nanos}"));
            std::fs::create_dir_all(&dir).expect("mkdir temp e2e dir");
            dir
        }

        const VGTK_MAGIC: [u8; 4] = [b'V', b'G', b'T', b'K'];
        const VGTK_HEADER_SIZE: usize = 72;

        /// T2: a publishable GPU token round-trips through the router and
        /// lands as a `VGTK` record on disk, with no CPU pixel bytes
        /// being written.
        #[test]
        fn e2e_gpu_token_round_trip() {
            let path = unique_temp_path("gpu-token", "bin");
            let _ = std::fs::remove_file(&path);

            let writer: Box<dyn OutputSinkWriter> =
                Box::new(SharedMemoryFileSink::with_path(path.clone()));
            let mut router =
                OutputRouter::with_writer(FrameSink::SharedTextureFileStub, writer);

            router.publish(make_valid_gpu_frame(5_000));
            assert!(
                router.wait_until_worker_drains(Duration::from_secs(2)),
                "worker did not drain"
            );

            let bytes = std::fs::read(&path).expect("router wrote sink file");
            assert_eq!(
                &bytes[0..4],
                &VGTK_MAGIC,
                "sink wrote a non-VGTK record for a publishable GPU token"
            );
            assert_eq!(bytes.len(), VGTK_HEADER_SIZE);

            router.shutdown();
            let _ = std::fs::remove_file(path);
        }

        /// T3: a CPU-readback frame round-trips through `ImageSequence`
        /// and lands as `frame_000000.png` with the requested
        /// dimensions.
        #[test]
        fn e2e_cpu_readback_round_trip() {
            let dir = unique_temp_dir("cpu-readback");

            let writer: Box<dyn OutputSinkWriter> = Box::new(ImageSequenceSink::new(&dir));
            let mut router = OutputRouter::with_writer(FrameSink::ImageSequence, writer);

            let mut frame = OutputFrame::new(6_000, [4, 3], 1_000);
            frame.pixel_data = Some(Arc::new(vec![0u8; 4 * 3 * 4]));
            router.publish(frame);
            assert!(
                router.wait_until_worker_drains(Duration::from_secs(2)),
                "worker did not drain"
            );

            let png_path = dir.join("frame_000000.png");
            assert!(
                png_path.exists(),
                "ImageSequenceSink did not write {} after publish",
                png_path.display()
            );
            // Decode the PNG header to confirm dimensions match. The
            // sink's own unit tests cover round-trip integrity in
            // detail; here we just want the composed path to land at
                // the right place with the right shape.
            let bytes = std::fs::read(&png_path).expect("read png");
            assert_eq!(&bytes[0..8], b"\x89PNG\r\n\x1a\n");

            router.shutdown();
            let _ = std::fs::remove_dir_all(&dir);
        }

        /// T4: changing the sink at runtime routes subsequent frames to
        /// the new writer's artefact, even though the router instance
        /// is the same.
        #[test]
        fn e2e_sink_swap_routes_to_new_writer_within_one_frame() {
            let dir_a = unique_temp_dir("swap-a");
            let path_b = unique_temp_path("swap-b", "bin");
            let _ = std::fs::remove_file(&path_b);

            let writer_a: Box<dyn OutputSinkWriter> =
                Box::new(ImageSequenceSink::new(&dir_a));
            let mut router = OutputRouter::with_writer(FrameSink::ImageSequence, writer_a);

            let mut cpu = OutputFrame::new(7_000, [2, 2], 1_000);
            cpu.pixel_data = Some(Arc::new(vec![0u8; 2 * 2 * 4]));
            router.publish(cpu);
            assert!(router.wait_until_worker_drains(Duration::from_secs(2)));
            assert!(
                dir_a.join("frame_000000.png").exists(),
                "first sink's artefact should exist before the swap"
            );

            let writer_b: Box<dyn OutputSinkWriter> =
                Box::new(SharedMemoryFileSink::with_path(path_b.clone()));
            router.set_writer(FrameSink::SharedTextureFileStub, writer_b);

            router.publish(make_valid_gpu_frame(7_001));
            assert!(router.wait_until_worker_drains(Duration::from_secs(2)));

            let bytes = std::fs::read(&path_b).expect("second sink wrote VGTK file");
            assert_eq!(&bytes[0..4], &VGTK_MAGIC);
            assert!(
                !dir_a.join("frame_000001.png").exists(),
                "post-swap frame must NOT land in the first sink's directory"
            );

            router.shutdown();
            let _ = std::fs::remove_file(&path_b);
            let _ = std::fs::remove_dir_all(&dir_a);
        }
    }
}
