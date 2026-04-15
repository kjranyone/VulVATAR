#![allow(dead_code)]
pub mod directshow_filter;
pub mod dshow_source;
pub mod frame_sink;
pub mod virtual_camera_native;

pub use frame_sink::{create_sink_writer, FrameSink, FrameSinkQueuePolicy, OutputSinkWriter};

use crate::avatar::pose::FrameTimestamp;
use log::{error, info};
use std::collections::VecDeque;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

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

    /// Channel sender to the background output worker.
    worker_tx: Option<mpsc::Sender<WorkerMessage>>,
    /// Join handle for the background worker thread.
    worker_handle: Option<thread::JoinHandle<()>>,
}

impl OutputRouter {
    pub fn new(sink: FrameSink) -> Self {
        let writer = create_sink_writer(&sink);
        let (tx, rx) = mpsc::channel::<WorkerMessage>();
        let handle = Self::spawn_worker(rx, writer);

        Self {
            sink,
            policy: FrameSinkQueuePolicy::ReplaceLatest,
            queue: VecDeque::new(),
            max_queue_depth: 2,
            next_frame_id: 0,
            dropped_count: 0,
            last_publish_timestamp: 0,
            worker_tx: Some(tx),
            worker_handle: Some(handle),
        }
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
        info!(
            "output: publish frame {} {}x{} {:?} alpha={:?} to {:?}",
            frame.frame_id.0,
            frame.extent[0],
            frame.extent[1],
            frame.color_space,
            frame.alpha_mode,
            self.sink,
        );

        self.last_publish_timestamp = frame.timestamp;

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
        // from the local queue so the queue does not grow unboundedly.
        if let Some(queued) = self.queue.pop_front() {
            if let Some(ref tx) = self.worker_tx {
                let _ = tx.send(WorkerMessage::Frame(queued));
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

    pub fn diagnostics(&self) -> OutputDiagnostics {
        let gpu_interop_enabled = matches!(self.sink, FrameSink::SharedTexture);
        let fallback_active = matches!(
            self.sink,
            FrameSink::SharedMemory | FrameSink::ImageSequence
        );
        OutputDiagnostics {
            active_sink: self.sink.clone(),
            active_handoff_path: if gpu_interop_enabled {
                HandoffPath::GpuSharedFrame
            } else {
                HandoffPath::CpuReadback
            },
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
    pub active_handoff_path: HandoffPath,
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
