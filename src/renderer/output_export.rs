use crate::renderer::frame_input::OutputTargetRequest;
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
pub struct ExportedFrame {
    /// Raw RGBA8 pixel data read back from GPU. Empty if readback was not
    /// requested or if the frame was produced via GPU-only export path.
    ///
    /// Wrapped in `Arc` so the buffer can be shared between the viewport
    /// display and the output pipeline without copying ~8 MB per frame.
    pub pixel_data: Arc<Vec<u8>>,
    /// Image extent as `[width, height]`.
    pub extent: [u32; 2],
    /// Frame timestamp in nanoseconds.
    pub timestamp_nanos: u64,
    pub gpu_token_id: u64,
    pub export_metadata: ExportMetadata,
}

#[derive(Clone, Debug)]
pub struct ExportMetadata {
    pub width: u32,
    pub height: u32,
    pub has_alpha: bool,
    pub color_space: String,
    pub timestamp_nanos: u64,
}

pub struct OutputExporter {
    export_count: u64,
}

impl OutputExporter {
    pub fn new() -> Self {
        Self { export_count: 0 }
    }

    /// Perform a GPU-to-CPU readback of the rendered color image and produce an
    /// `ExportResult` containing the raw pixel data.
    pub fn export_readback(
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
            // Even when output is disabled, we still need to flush the future so
            // that the GPU work is submitted.
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
        let buffer_size = pixel_count * 4; // RGBA8

        let readback_buffer: Subbuffer<[u8]> = Buffer::new_slice::<u8>(
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
        )
        .expect("failed to create readback buffer");

        let mut builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("failed to create readback command buffer");

        builder
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                color_image,
                readback_buffer.clone(),
            ))
            .expect("failed to record copy_image_to_buffer");

        let command_buffer = builder.build().unwrap();

        let future = after_future
            .then_execute(queue, command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .expect("failed to flush readback");

        future.wait(None).expect("failed to wait for readback");

        let data = readback_buffer.read().unwrap();
        let pixel_data = Arc::new(data.to_vec());

        let token_id = self.export_count;
        self.export_count += 1;

        ExportResult {
            exported_frame: Some(ExportedFrame {
                pixel_data,
                extent: [width, height],
                timestamp_nanos,
                gpu_token_id: token_id,
                export_metadata: ExportMetadata {
                    width,
                    height,
                    has_alpha: true,
                    color_space: "srgb".to_string(),
                    timestamp_nanos,
                },
            }),
            export_succeeded: true,
        }
    }
}
