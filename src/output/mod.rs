pub mod frame_sink;

pub use frame_sink::{FrameSink, FrameSinkQueuePolicy};

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
    pub resource_id: u64,
    pub handle_type: ExternalHandleType,
    pub sync: GpuSyncToken,
    pub lifetime: FrameLifetimeContract,
}

#[derive(Clone, Debug)]
pub struct OutputFrame {
    pub frame_id: OutputFrameId,
    pub timestamp: u64,
    pub extent: [u32; 2],
    pub color_space: OutputColorSpace,
    pub alpha_mode: AlphaMode,
    pub gpu_token: GpuFrameToken,
}

impl OutputFrame {
    pub fn new(id: u64, extent: [u32; 2], timestamp: u64) -> Self {
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
        }
    }
}

use std::collections::VecDeque;

pub struct OutputRouter {
    sink: FrameSink,
    policy: FrameSinkQueuePolicy,
    queue: VecDeque<OutputFrame>,
    max_queue_depth: usize,
    next_frame_id: u64,
    dropped_count: u64,
}

impl OutputRouter {
    pub fn new(sink: FrameSink) -> Self {
        Self {
            sink,
            policy: FrameSinkQueuePolicy::ReplaceLatest,
            queue: VecDeque::new(),
            max_queue_depth: 2,
            next_frame_id: 0,
            dropped_count: 0,
        }
    }

    pub fn publish(&mut self, frame: OutputFrame) {
        println!(
            "output: publish frame {} {}x{} {:?} alpha={:?} to {:?}",
            frame.frame_id.0,
            frame.extent[0],
            frame.extent[1],
            frame.color_space,
            frame.alpha_mode,
            self.sink,
        );

        match self.policy {
            FrameSinkQueuePolicy::ReplaceLatest => {
                if self.queue.len() >= self.max_queue_depth {
                    self.queue.pop_front();
                    self.dropped_count += 1;
                }
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
                }
            }
            FrameSinkQueuePolicy::BlockNotAllowed => {
                if self.queue.len() < self.max_queue_depth {
                    self.queue.push_back(frame);
                } else {
                    self.dropped_count += 1;
                }
            }
        }
    }

    pub fn create_frame(&mut self, width: u32, height: u32, timestamp: u64) -> OutputFrame {
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
        }
    }

    pub fn dropped_count(&self) -> u64 {
        self.dropped_count
    }

    pub fn queue_depth(&self) -> usize {
        self.queue.len()
    }

    pub fn diagnostics(&self) -> OutputDiagnostics {
        OutputDiagnostics {
            active_sink: self.sink.clone(),
            active_handoff_path: HandoffPath::CpuReadback,
            gpu_interop_enabled: false,
            fallback_active: true,
            queue_depth: self.queue.len(),
            max_queue_depth: self.max_queue_depth,
            dropped_frame_count: self.dropped_count,
            last_publish_timestamp: 0,
        }
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
    pub last_publish_timestamp: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum HandoffPath {
    GpuSharedFrame,
    CpuReadback,
    SharedMemory,
}

#[derive(Clone, Debug)]
pub enum SinkConnectionState {
    Connected,
    Disconnected,
    Error(String),
}

#[derive(Clone, Debug)]
pub enum OutputFormatConfig {
    Rgb,
    RgbaPremultiplied,
    RgbaStraight,
}
