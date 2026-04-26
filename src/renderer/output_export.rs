#![allow(dead_code)]
use crate::renderer::frame_input::{OutputTargetRequest, RenderColorSpace, RenderExportMode};
use crate::renderer::frame_pool::FramePool;
use crate::renderer::gpu_handle::{GpuHandleExporter, SharedHandle};
use std::collections::HashSet;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo,
};
use vulkano::device::Queue;
use vulkano::image::Image;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::sync::GpuFuture;

#[derive(Clone, Debug)]
pub struct ExportRequest {
    pub output_request: OutputTargetRequest,
    pub color_image_id: u64,
    pub alpha_image_id: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct ExportResult {
    pub exported_frame: Option<ExportedFrame>,
    pub export_succeeded: bool,
}

#[derive(Clone, Debug)]
pub enum ExportedPixelData {
    CpuReadback(Arc<Vec<u8>>),
    GpuOwned,
}

#[derive(Clone, Debug)]
pub struct ExportedFrame {
    pub pixel_data: ExportedPixelData,
    pub gpu_image: Option<Arc<Image>>,
    pub extent: [u32; 2],
    pub timestamp_nanos: u64,
    pub gpu_token_id: u64,
    pub export_metadata: ExportMetadata,
    pub handoff_path: crate::output::HandoffPath,
}

impl ExportedFrame {
    pub fn cpu_pixel_data(&self) -> Option<Arc<Vec<u8>>> {
        match &self.pixel_data {
            ExportedPixelData::CpuReadback(data) => Some(Arc::clone(data)),
            ExportedPixelData::GpuOwned => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        match &self.pixel_data {
            ExportedPixelData::CpuReadback(data) => data.is_empty(),
            ExportedPixelData::GpuOwned => false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExportMetadata {
    pub width: u32,
    pub height: u32,
    pub has_alpha: bool,
    /// Output colour space the renderer produced this frame in. Filled from
    /// the corresponding `OutputTargetRequest.color_space` rather than
    /// hardcoded — downstream sinks (MF virtual camera, image-sequence
    /// manifests) read this when tagging frames.
    pub color_space: RenderColorSpace,
    pub timestamp_nanos: u64,
}

pub struct OutputExporter {
    export_count: u64,
    gpu_handle_exporter: GpuHandleExporter,
    released_tokens: HashSet<u64>,
    frame_pool: FramePool,
}

impl OutputExporter {
    pub fn new() -> Self {
        Self {
            export_count: 0,
            gpu_handle_exporter: GpuHandleExporter::new(),
            released_tokens: HashSet::new(),
            frame_pool: FramePool::new(2),
        }
    }

    pub fn release_token(&mut self, token_id: u64) {
        log::info!("output-export: release_token({})", token_id);
        self.released_tokens.insert(token_id);
        self.frame_pool.release(token_id);
    }

    pub fn is_token_released(&self, token_id: u64) -> bool {
        self.released_tokens.contains(&token_id)
    }

    pub fn released_token_count(&self) -> usize {
        self.released_tokens.len()
    }

    pub fn frame_pool(&self) -> &FramePool {
        &self.frame_pool
    }

    pub fn frame_pool_mut(&mut self) -> &mut FramePool {
        &mut self.frame_pool
    }

    pub fn export(
        &mut self,
        request: &ExportRequest,
        color_image: Arc<Image>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        queue: Arc<Queue>,
        after_future: Box<dyn GpuFuture>,
        timestamp_nanos: u64,
    ) -> ExportResult {
        match request.output_request.export_mode {
            RenderExportMode::GpuExport => self.export_gpu(
                request,
                color_image,
                queue,
                after_future,
                timestamp_nanos,
                memory_allocator,
                command_buffer_allocator,
            ),
            RenderExportMode::CpuReadback => self.export_readback(
                request,
                color_image,
                memory_allocator,
                command_buffer_allocator,
                queue,
                after_future,
                timestamp_nanos,
            ),
            RenderExportMode::None => {
                after_future
                    .then_signal_fence_and_flush()
                    .ok()
                    .map(|f| f.wait(None).ok());
                ExportResult {
                    exported_frame: None,
                    export_succeeded: false,
                }
            }
        }
    }

    fn export_gpu(
        &mut self,
        request: &ExportRequest,
        color_image: Arc<Image>,
        queue: Arc<Queue>,
        after_future: Box<dyn GpuFuture>,
        timestamp_nanos: u64,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    ) -> ExportResult {
        self.gpu_handle_exporter.set_device(queue.device().clone());

        let width = request.output_request.extent[0];
        let height = request.output_request.extent[1];

        let shared_frame = self.gpu_handle_exporter.export_frame(color_image.clone());

        let fence = match after_future.then_signal_fence_and_flush() {
            Ok(f) => f,
            Err(e) => {
                log::error!("failed to flush gpu export: {e}");
                return ExportResult {
                    exported_frame: None,
                    export_succeeded: false,
                };
            }
        };
        if let Err(e) = fence.wait(None) {
            log::error!("failed to wait for gpu export: {e}");
            return ExportResult {
                exported_frame: None,
                export_succeeded: false,
            };
        }

        let token_id = self.export_count;
        self.export_count += 1;

        let color_space = request.output_request.color_space.clone();

        match &shared_frame.shared_handle {
            SharedHandle::Win32Kmt { handle: _ } => ExportResult {
                exported_frame: Some(ExportedFrame {
                    pixel_data: ExportedPixelData::GpuOwned,
                    gpu_image: Some(color_image),
                    extent: [width, height],
                    timestamp_nanos,
                    gpu_token_id: token_id,
                    export_metadata: ExportMetadata {
                        width,
                        height,
                        has_alpha: true,
                        color_space: color_space.clone(),
                        timestamp_nanos,
                    },
                    handoff_path: crate::output::HandoffPath::GpuSharedFrame,
                }),
                export_succeeded: true,
            },
            SharedHandle::NamedSharedMemory { name: _, size: _ } => {
                let pixel_data = Self::readback_image_pixels(
                    color_image,
                    memory_allocator,
                    command_buffer_allocator,
                    queue,
                    width,
                    height,
                );

                if let Err(ref e) = self.gpu_handle_exporter.export_frame_to_shared_memory(
                    &pixel_data,
                    width,
                    height,
                    shared_frame.frame_index,
                ) {
                    log::error!("gpu-handle: shared memory write failed: {}", e);
                }

                ExportResult {
                    exported_frame: Some(ExportedFrame {
                        pixel_data: ExportedPixelData::CpuReadback(Arc::new(pixel_data)),
                        gpu_image: None,
                        extent: [width, height],
                        timestamp_nanos,
                        gpu_token_id: token_id,
                        export_metadata: ExportMetadata {
                            width,
                            height,
                            has_alpha: true,
                            color_space: color_space.clone(),
                            timestamp_nanos,
                        },
                        handoff_path: crate::output::HandoffPath::SharedMemory,
                    }),
                    export_succeeded: true,
                }
            }
            SharedHandle::None => ExportResult {
                exported_frame: Some(ExportedFrame {
                    pixel_data: ExportedPixelData::GpuOwned,
                    gpu_image: Some(color_image),
                    extent: [width, height],
                    timestamp_nanos,
                    gpu_token_id: token_id,
                    export_metadata: ExportMetadata {
                        width,
                        height,
                        has_alpha: true,
                        color_space,
                        timestamp_nanos,
                    },
                    handoff_path: crate::output::HandoffPath::GpuSharedFrame,
                }),
                export_succeeded: true,
            },
        }
    }

    fn readback_image_pixels(
        color_image: Arc<Image>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        queue: Arc<Queue>,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let pixel_count = (width as u64) * (height as u64);
        let buffer_size = pixel_count * 4;

        let readback_buffer: Subbuffer<[u8]> = match Buffer::new_slice::<u8>(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            buffer_size,
        ) {
            Ok(b) => b,
            Err(e) => {
                log::error!("failed to create readback buffer: {}", e);
                return Vec::new();
            }
        };

        let mut builder = match AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ) {
            Ok(b) => b,
            Err(e) => {
                log::error!("failed to create readback command buffer: {}", e);
                return Vec::new();
            }
        };

        if let Err(e) = builder.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            color_image,
            readback_buffer.clone(),
        )) {
            log::error!("failed to record copy_image_to_buffer: {}", e);
            return Vec::new();
        }

        let command_buffer = match builder.build() {
            Ok(cb) => cb,
            Err(e) => {
                log::error!("failed to build readback command buffer: {}", e);
                return Vec::new();
            }
        };

        let device = queue.device().clone();
        let future = match vulkano::sync::now(device).then_execute(queue, command_buffer) {
            Ok(f) => f,
            Err(e) => {
                log::error!("failed to execute readback: {}", e);
                return Vec::new();
            }
        };

        let fence = match future.then_signal_fence_and_flush() {
            Ok(f) => f,
            Err(e) => {
                log::error!("failed to flush readback: {}", e);
                return Vec::new();
            }
        };

        if let Err(e) = fence.wait(None) {
            log::error!("failed to wait for readback: {}", e);
            return Vec::new();
        }

        let result = match readback_buffer.read() {
            Ok(data) => data.to_vec(),
            Err(e) => {
                log::error!("failed to read readback buffer: {}", e);
                Vec::new()
            }
        };
        result
    }

    fn export_readback(
        &mut self,
        request: &ExportRequest,
        color_image: Arc<Image>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        queue: Arc<Queue>,
        after_future: Box<dyn GpuFuture>,
        timestamp_nanos: u64,
    ) -> ExportResult {
        if !request.output_request.output_enabled {
            after_future
                .then_signal_fence_and_flush()
                .ok()
                .map(|f| f.wait(None).ok());
            return ExportResult {
                exported_frame: None,
                export_succeeded: false,
            };
        }

        let width = request.output_request.extent[0];
        let height = request.output_request.extent[1];
        let pixel_count = (width as u64) * (height as u64);
        let buffer_size = pixel_count * 4;

        let readback_buffer: Subbuffer<[u8]> = match Buffer::new_slice::<u8>(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            buffer_size,
        ) {
            Ok(b) => b,
            Err(e) => {
                log::error!("export_readback: failed to create readback buffer: {}", e);
                return ExportResult {
                    exported_frame: None,
                    export_succeeded: false,
                };
            }
        };

        let mut builder = match AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ) {
            Ok(b) => b,
            Err(e) => {
                log::error!("export_readback: failed to create command buffer: {}", e);
                return ExportResult {
                    exported_frame: None,
                    export_succeeded: false,
                };
            }
        };

        if let Err(e) = builder.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            color_image.clone(),
            readback_buffer.clone(),
        )) {
            log::error!("export_readback: failed to record copy: {}", e);
            return ExportResult {
                exported_frame: None,
                export_succeeded: false,
            };
        }

        let command_buffer = match builder.build() {
            Ok(cb) => cb,
            Err(e) => {
                log::error!("export_readback: failed to build command buffer: {}", e);
                return ExportResult {
                    exported_frame: None,
                    export_succeeded: false,
                };
            }
        };

        let fence = match after_future.then_execute(queue, command_buffer) {
            Ok(f) => match f.then_signal_fence_and_flush() {
                Ok(fence) => fence,
                Err(e) => {
                    log::error!("export_readback: failed to flush: {}", e);
                    return ExportResult {
                        exported_frame: None,
                        export_succeeded: false,
                    };
                }
            },
            Err(e) => {
                log::error!("export_readback: failed to execute: {}", e);
                return ExportResult {
                    exported_frame: None,
                    export_succeeded: false,
                };
            }
        };

        if let Err(e) = fence.wait(None) {
            log::error!("export_readback: failed to wait: {}", e);
            return ExportResult {
                exported_frame: None,
                export_succeeded: false,
            };
        }

        let pixel_data = match readback_buffer.read() {
            Ok(data) => Arc::new(data.to_vec()),
            Err(e) => {
                log::error!("export_readback: failed to read buffer: {}", e);
                return ExportResult {
                    exported_frame: None,
                    export_succeeded: false,
                };
            }
        };

        let token_id = self.export_count;
        self.export_count += 1;

        ExportResult {
            exported_frame: Some(ExportedFrame {
                pixel_data: ExportedPixelData::CpuReadback(pixel_data),
                gpu_image: None,
                extent: [width, height],
                timestamp_nanos,
                gpu_token_id: token_id,
                export_metadata: ExportMetadata {
                    width,
                    height,
                    has_alpha: true,
                    color_space: request.output_request.color_space.clone(),
                    timestamp_nanos,
                },
                handoff_path: crate::output::HandoffPath::CpuReadback,
            }),
            export_succeeded: true,
        }
    }
}
