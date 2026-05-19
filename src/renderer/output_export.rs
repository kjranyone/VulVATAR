use crate::frame_handoff::{
    ExternalHandleType, FallbackReason, FrameLease, FrameLifetimeContract, GpuFrameToken,
    HandoffPath, OutputSyncToken,
};
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

const DEFAULT_EXPORT_IMAGE_POOL_CAPACITY: usize = 2;

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
    GpuFrameToken(GpuFrameToken),
}

#[derive(Clone, Debug)]
pub struct ExportedFrame {
    pub pixel_data: ExportedPixelData,
    pub extent: [u32; 2],
    pub timestamp_nanos: u64,
    pub gpu_token_id: u64,
    pub export_metadata: ExportMetadata,
    pub handoff_path: HandoffPath,
    pub fallback_reason: Option<FallbackReason>,
}

impl ExportedFrame {
    pub fn cpu_pixel_data(&self) -> Option<Arc<Vec<u8>>> {
        match &self.pixel_data {
            ExportedPixelData::CpuReadback(data) => Some(Arc::clone(data)),
            ExportedPixelData::GpuFrameToken(_) => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        match &self.pixel_data {
            ExportedPixelData::CpuReadback(data) => data.is_empty(),
            ExportedPixelData::GpuFrameToken(_) => false,
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

/// Bounded ownership tracker for GPU-exported render targets.
///
/// The actual Vulkan image allocation stays renderer-owned. This pool owns the
/// externally visible resource identity and lease state so the export path can
/// refuse unsafe reuse instead of quietly handing the same image to multiple
/// consumers at once.
#[derive(Clone, Debug)]
pub struct ExportImagePool {
    slots: Vec<ExportImageSlot>,
    capacity: usize,
    next_resource_id: u64,
    next_lease_id: u64,
}

#[derive(Clone, Debug)]
struct ExportImageSlot {
    resource_id: u64,
    image_id: u64,
    state: ExportImageSlotState,
    last_used_frame: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExportImageSlotState {
    Available,
    Rendering,
    Leased { lease_id: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExportImageReservation {
    resource_id: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExportImageLease {
    resource_id: u64,
    lease_id: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExportImagePoolStats {
    pub capacity: usize,
    pub total_slots: usize,
    pub available_slots: usize,
    pub rendering_slots: usize,
    pub leased_slots: usize,
}

impl ExportImagePool {
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity.max(1)),
            capacity: capacity.max(1),
            next_resource_id: 0,
            next_lease_id: 0,
        }
    }

    /// Reserve a slot for a freshly rendered image without ever blocking the
    /// render thread. Policy: reuse the matching available slot first, then
    /// any available slot, then grow until capacity; when every slot is
    /// leased, return `None` and let the caller take an explicit fallback.
    fn begin_render(&mut self, image_id: u64, frame_index: u64) -> Option<ExportImageReservation> {
        let reusable_index = self
            .slots
            .iter()
            .position(|slot| {
                slot.image_id == image_id
                    && matches!(slot.state, ExportImageSlotState::Available)
            })
            .or_else(|| {
                self.slots
                    .iter()
                    .position(|slot| matches!(slot.state, ExportImageSlotState::Available))
            });

        if let Some(index) = reusable_index {
            let slot = &mut self.slots[index];
            slot.image_id = image_id;
            slot.state = ExportImageSlotState::Rendering;
            slot.last_used_frame = frame_index;
            return Some(ExportImageReservation {
                resource_id: slot.resource_id,
            });
        }

        if self.slots.len() >= self.capacity {
            return None;
        }

        let resource_id = self.next_resource_id;
        self.next_resource_id += 1;
        self.slots.push(ExportImageSlot {
            resource_id,
            image_id,
            state: ExportImageSlotState::Rendering,
            last_used_frame: frame_index,
        });
        Some(ExportImageReservation { resource_id })
    }

    fn lease_after_render(
        &mut self,
        reservation: ExportImageReservation,
    ) -> Option<ExportImageLease> {
        let slot = self
            .slots
            .iter_mut()
            .find(|slot| slot.resource_id == reservation.resource_id)?;
        if !matches!(slot.state, ExportImageSlotState::Rendering) {
            return None;
        }

        let lease_id = self.next_lease_id;
        self.next_lease_id += 1;
        slot.state = ExportImageSlotState::Leased { lease_id };
        Some(ExportImageLease {
            resource_id: slot.resource_id,
            lease_id,
        })
    }

    fn abandon_render(&mut self, reservation: ExportImageReservation) {
        if let Some(slot) = self
            .slots
            .iter_mut()
            .find(|slot| slot.resource_id == reservation.resource_id)
        {
            if matches!(slot.state, ExportImageSlotState::Rendering) {
                slot.state = ExportImageSlotState::Available;
            }
        }
    }

    fn release_lease(&mut self, lease_id: u64) -> bool {
        let Some(slot) = self.slots.iter_mut().find(|slot| {
            matches!(
                slot.state,
                ExportImageSlotState::Leased {
                    lease_id: active_lease_id
                } if active_lease_id == lease_id
            )
        }) else {
            return false;
        };
        slot.state = ExportImageSlotState::Available;
        true
    }

    pub fn stats(&self) -> ExportImagePoolStats {
        let mut stats = ExportImagePoolStats {
            capacity: self.capacity,
            total_slots: self.slots.len(),
            ..Default::default()
        };
        for slot in &self.slots {
            match slot.state {
                ExportImageSlotState::Available => stats.available_slots += 1,
                ExportImageSlotState::Rendering => stats.rendering_slots += 1,
                ExportImageSlotState::Leased { .. } => stats.leased_slots += 1,
            }
        }
        stats
    }
}

pub struct OutputExporter {
    export_count: u64,
    gpu_handle_exporter: GpuHandleExporter,
    released_tokens: HashSet<u64>,
    frame_pool: FramePool,
    export_image_pool: ExportImagePool,
}

impl Default for OutputExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputExporter {
    pub fn new() -> Self {
        Self {
            export_count: 0,
            gpu_handle_exporter: GpuHandleExporter::new(),
            released_tokens: HashSet::new(),
            frame_pool: FramePool::new(2),
            export_image_pool: ExportImagePool::new(DEFAULT_EXPORT_IMAGE_POOL_CAPACITY),
        }
    }

    pub fn release_token(&mut self, lease_id: u64) {
        log::info!("output-export: release_token({})", lease_id);
        self.released_tokens.insert(lease_id);
        self.frame_pool.release(lease_id);
        if !self.export_image_pool.release_lease(lease_id) {
            log::warn!(
                "output-export: release_token({}) did not match an active GPU lease",
                lease_id
            );
        }
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

    pub fn export_image_pool(&self) -> &ExportImagePool {
        &self.export_image_pool
    }

    #[allow(clippy::too_many_arguments)]
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
                Some(FallbackReason::RequestedCpuReadback),
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

    #[allow(clippy::too_many_arguments)]
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
        let reservation = match self
            .export_image_pool
            .begin_render(request.color_image_id, self.export_count)
        {
            Some(reservation) => reservation,
            None => {
                log::warn!(
                    "output-export: GPU export pool saturated; falling back to CPU readback"
                );
                return self.export_readback(
                    request,
                    color_image,
                    memory_allocator,
                    command_buffer_allocator,
                    queue,
                    after_future,
                    timestamp_nanos,
                    Some(FallbackReason::ExportPoolSaturated),
                );
            }
        };

        let shared_frame = self.gpu_handle_exporter.export_frame(color_image.clone());

        let fence = match after_future.then_signal_fence_and_flush() {
            Ok(f) => f,
            Err(e) => {
                self.export_image_pool.abandon_render(reservation);
                log::error!("failed to flush gpu export: {e}");
                return ExportResult {
                    exported_frame: None,
                    export_succeeded: false,
                };
            }
        };
        if let Err(e) = fence.wait(None) {
            self.export_image_pool.abandon_render(reservation);
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
            SharedHandle::Win32Kmt { handle } => {
                let Some(lease) = self.export_image_pool.lease_after_render(reservation) else {
                    log::error!("output-export: failed to lease GPU export slot after render");
                    return ExportResult {
                        exported_frame: None,
                        export_succeeded: false,
                    };
                };
                ExportResult {
                    exported_frame: Some(ExportedFrame {
                        pixel_data: ExportedPixelData::GpuFrameToken(GpuFrameToken {
                            resource_id: lease.resource_id,
                            handle_type: ExternalHandleType::Win32Kmt,
                            external_handle: Some(*handle),
                            // `export_gpu` waits the producer fence before
                            // publishing this token. That is a real readiness
                            // guarantee for the current path, but it is not
                            // yet an exported cross-process fence handle.
                            sync: OutputSyncToken::ProducerWaitComplete,
                            lease: FrameLease {
                                lease_id: lease.lease_id,
                                lifetime: FrameLifetimeContract::SingleConsumerImmediate,
                            },
                        }),
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
                        handoff_path: HandoffPath::GpuSharedFrame,
                        fallback_reason: None,
                    }),
                    export_succeeded: true,
                }
            }
            SharedHandle::NamedSharedMemory { name: _, size: _ } => {
                self.export_image_pool.abandon_render(reservation);
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
                        handoff_path: HandoffPath::SharedMemory,
                        fallback_reason: Some(FallbackReason::ExternalHandleUnavailable),
                    }),
                    export_succeeded: true,
                }
            }
            SharedHandle::None => {
                let Some(lease) = self.export_image_pool.lease_after_render(reservation) else {
                    log::error!("output-export: failed to lease GPU export slot after render");
                    return ExportResult {
                        exported_frame: None,
                        export_succeeded: false,
                    };
                };
                ExportResult {
                    exported_frame: Some(ExportedFrame {
                        pixel_data: ExportedPixelData::GpuFrameToken(GpuFrameToken {
                            resource_id: lease.resource_id,
                            handle_type: ExternalHandleType::Unavailable,
                            external_handle: None,
                            sync: OutputSyncToken::ProducerWaitComplete,
                            lease: FrameLease {
                                lease_id: lease.lease_id,
                                lifetime: FrameLifetimeContract::SingleConsumerImmediate,
                            },
                        }),
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
                        handoff_path: HandoffPath::GpuSharedFrame,
                        fallback_reason: Some(FallbackReason::ExternalHandleUnavailable),
                    }),
                    export_succeeded: true,
                }
            }
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

    #[allow(clippy::too_many_arguments)]
    fn export_readback(
        &mut self,
        request: &ExportRequest,
        color_image: Arc<Image>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        queue: Arc<Queue>,
        after_future: Box<dyn GpuFuture>,
        timestamp_nanos: u64,
        fallback_reason: Option<FallbackReason>,
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
                handoff_path: HandoffPath::CpuReadback,
                fallback_reason: fallback_reason
                    .or(Some(FallbackReason::RequestedCpuReadback)),
            }),
            export_succeeded: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_image_pool_reuses_available_slots_and_tracks_leases() {
        let mut pool = ExportImagePool::new(2);

        let first = pool.begin_render(10, 1).expect("first reservation");
        assert_eq!(
            pool.stats(),
            ExportImagePoolStats {
                capacity: 2,
                total_slots: 1,
                available_slots: 0,
                rendering_slots: 1,
                leased_slots: 0,
            }
        );
        let first_lease = pool.lease_after_render(first).expect("first lease");
        assert_eq!(pool.stats().leased_slots, 1);

        let second = pool.begin_render(20, 2).expect("second reservation");
        let second_lease = pool.lease_after_render(second).expect("second lease");
        assert_eq!(pool.stats().total_slots, 2);
        assert_eq!(pool.stats().leased_slots, 2);

        assert!(
            pool.begin_render(30, 3).is_none(),
            "pool must not grow or block when every slot is leased"
        );

        assert!(pool.release_lease(first_lease.lease_id));
        let reused = pool.begin_render(30, 4).expect("reused reservation");
        assert_eq!(
            reused.resource_id, first_lease.resource_id,
            "available slots should be recycled rather than allocated anew"
        );

        assert!(pool.release_lease(second_lease.lease_id));
    }

    #[test]
    fn export_image_pool_abandons_failed_renders() {
        let mut pool = ExportImagePool::new(1);
        let reservation = pool.begin_render(10, 1).expect("reservation");
        pool.abandon_render(reservation);
        assert_eq!(
            pool.stats(),
            ExportImagePoolStats {
                capacity: 1,
                total_slots: 1,
                available_slots: 1,
                rendering_slots: 0,
                leased_slots: 0,
            }
        );
    }
}
