pub mod frame_sink;
#[cfg(all(target_os = "windows", feature = "virtual-camera"))]
pub mod mf_virtual_camera;
#[cfg(target_os = "windows")]
pub mod shmem_security;
#[cfg(all(target_os = "windows", feature = "virtual-camera"))]
pub mod virtual_camera_native;

pub use frame_sink::{create_sink_writer, FrameSink, FrameSinkQueuePolicy, OutputSinkWriter};

use crate::avatar::pose::FrameTimestamp;
use log::{debug, error, info};
use std::collections::VecDeque;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Unique identifier for a GPU-exported resource (shared texture handle, etc.).
pub type ExportedResourceId = u64;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OutputFrameId(pub u64);

#[derive(Clone, Debug)]
pub enum OutputColorSpace {
    Srgb,
    LinearSrgb,
}

#[derive(Clone, Debug)]
pub enum AlphaMode {
    Opaque,
    Premultiplied,
    Straight,
}

#[derive(Clone, Debug)]
pub enum ExternalHandleType {
    Win32Kmt,
    D3D12Fence,
    VkSemaphore,
    SharedMemoryHandle,
}

#[derive(Clone, Debug)]
pub enum GpuSyncToken {
    None,
    FenceValue(u64),
    SemaphoreHandle(u64),
}

#[derive(Clone, Debug)]
pub enum FrameLifetimeContract {
    SingleConsumerImmediate,
    SingleConsumerRetained,
}

#[derive(Clone, Debug)]
pub struct GpuFrameToken {
    pub resource_id: ExportedResourceId,
    pub handle_type: ExternalHandleType,
    pub sync: GpuSyncToken,
    pub lifetime: FrameLifetimeContract,
}

#[derive(Clone, Debug)]
pub struct OutputFrame {
    pub frame_id: OutputFrameId,
    pub timestamp: FrameTimestamp,
    pub extent: [u32; 2],
    pub color_space: OutputColorSpace,
    pub alpha_mode: AlphaMode,
    pub gpu_token: GpuFrameToken,
    pub handoff_path: HandoffPath,
    pub pixel_data: Option<Arc<Vec<u8>>>,
    pub gpu_image: Option<Arc<vulkano::image::Image>>,
}

impl OutputFrame {
    pub fn new(id: u64, extent: [u32; 2], timestamp: FrameTimestamp) -> Self {
        Self {
            frame_id: OutputFrameId(id),
            timestamp,
            extent,
            color_space: OutputColorSpace::Srgb,
            alpha_mode: AlphaMode::Premultiplied,
            gpu_token: GpuFrameToken {
                resource_id: id,
                handle_type: ExternalHandleType::SharedMemoryHandle,
                sync: GpuSyncToken::None,
                lifetime: FrameLifetimeContract::SingleConsumerImmediate,
            },
            handoff_path: HandoffPath::CpuReadback,
            pixel_data: None,
            gpu_image: None,
        }
    }

    pub fn with_pixel_data(mut self, data: Arc<Vec<u8>>) -> Self {
        self.pixel_data = Some(data);
        self
    }

    pub fn with_gpu_image(mut self, image: Arc<vulkano::image::Image>) -> Self {
        self.gpu_image = Some(image);
        self.handoff_path = HandoffPath::GpuSharedFrame;
        self.gpu_token.handle_type = ExternalHandleType::VkSemaphore;
        self
    }
}

// ---------------------------------------------------------------------------
// Worker message
// ---------------------------------------------------------------------------

enum WorkerMessage {
    Frame(OutputFrame),
    Shutdown,
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
    /// Join handle for the background worker thread.
    worker_handle: Option<thread::JoinHandle<()>>,
    logged_first_publish: bool,

    /// Most recent published `OutputFrame.handoff_path`. Used by
    /// [`Self::diagnostics`] so the GUI shows the path actually carried by
    /// frames instead of guessing from the sink variant. `None` until the
    /// first publish.
    last_handoff_path: Option<HandoffPath>,
}

impl OutputRouter {
    pub fn new(sink: FrameSink) -> Self {
        let writer = create_sink_writer(&sink);
        let (tx, rx) = mpsc::sync_channel::<WorkerMessage>(1);
        let handle = Self::spawn_worker(rx, writer);

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
            worker_handle: Some(handle),
            logged_first_publish: false,
            last_handoff_path: None,
        }
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
                        }
                        Ok(WorkerMessage::Shutdown) => {
                            info!("output-worker: shutdown requested, draining queue");
                            // Drain any remaining frames already in the channel.
                            while let Ok(msg) = rx.try_recv() {
                                if let WorkerMessage::Frame(frame) = msg {
                                    if let Err(e) = writer.write_frame(&frame) {
                                        error!("output-worker: drain write error: {}", e);
                                    }
                                }
                            }
                            info!("output-worker: stopped");
                            break;
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

        match self.policy {
            FrameSinkQueuePolicy::ReplaceLatest => {
                // Keep only the latest frame: clear the queue first.
                let dropped = self.queue.len();
                self.queue.clear();
                self.dropped_count += dropped as u64;
                self.queue.push_back(frame);
            }
            FrameSinkQueuePolicy::DropOldest => {
                if self.queue.len() >= self.max_queue_depth {
                    self.queue.pop_front();
                    self.dropped_count += 1;
                }
                self.queue.push_back(frame);
            }
            FrameSinkQueuePolicy::DropNewest => {
                if self.queue.len() < self.max_queue_depth {
                    self.queue.push_back(frame);
                } else {
                    self.dropped_count += 1;
                    return; // frame dropped, do not send to worker
                }
            }
            FrameSinkQueuePolicy::BlockNotAllowed => {
                if self.queue.len() < self.max_queue_depth {
                    self.queue.push_back(frame);
                } else {
                    self.dropped_count += 1;
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
                    }
                    Err(e) => {
                        self.dropped_count += 1;
                        debug!("output: worker channel full, dropping frame: {}", e);
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
            gpu_token: GpuFrameToken {
                resource_id: id,
                handle_type: ExternalHandleType::SharedMemoryHandle,
                sync: GpuSyncToken::None,
                lifetime: FrameLifetimeContract::SingleConsumerImmediate,
            },
            handoff_path: HandoffPath::CpuReadback,
            pixel_data: None,
            gpu_image: None,
        }
    }

    /// Drain the output queue and stop the background worker thread.
    pub fn shutdown(&mut self) {
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
        let gpu_interop_enabled = matches!(active_handoff_path, Some(HandoffPath::GpuSharedFrame));
        let fallback_active = matches!(
            active_handoff_path,
            Some(HandoffPath::CpuReadback | HandoffPath::SharedMemory)
        );
        OutputDiagnostics {
            active_sink: self.sink.clone(),
            active_handoff_path,
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
    pub gpu_interop_enabled: bool,
    pub fallback_active: bool,
    pub queue_depth: usize,
    pub max_queue_depth: usize,
    pub dropped_frame_count: u64,
    pub last_publish_timestamp: FrameTimestamp,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum HandoffPath {
    GpuSharedFrame,
    CpuReadback,
    SharedMemory,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(id: u64, path: HandoffPath) -> OutputFrame {
        let mut frame = OutputFrame::new(id, [16, 16], 1_000 + id);
        frame.handoff_path = path;
        frame
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
        assert!(after_gpu.gpu_interop_enabled);
        assert!(!after_gpu.fallback_active);

        router.shutdown();
    }
}
