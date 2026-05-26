//! Asynchronous CPU readback ring extracted from `VulkanRenderer` as
//! the second slice of the #12 renderer split. Owns the staging /
//! readback buffer pair, the per-frame fence wrapper, and the
//! `harvest` path that turns a finished readback into a
//! `RenderResult::CpuReadback`. The render-side enqueue still lives
//! in `mod.rs` next to the rest of the per-frame command-buffer
//! construction.

use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

use crate::renderer::{frame_input, output_export, RenderResult, RenderStats, VulkanRenderer};

/// The `wait_fn` closure captures the `FenceSignalFuture` by move so that we
/// can wait on it later without naming the complex concrete future type.
pub(super) struct PendingReadbackState {
    pub(super) wait_fn: Box<dyn FnOnce() -> Result<(), String> + Send>,
    pub(super) readback_buffer: Subbuffer<[u8]>,
    pub(super) extent: [u32; 2],
    pub(super) timestamp_nanos: u64,
    pub(super) stats: RenderStats,
    pub(super) color_space: frame_input::RenderColorSpace,
}

pub(super) const READBACK_RING_SIZE: usize = 2;

impl VulkanRenderer {
    /// Drain the pipelined CPU-readback queue, returning `Some(RenderResult)`
    /// only if a readback was already pending. Public to the crate so the
    /// render-thread driver can flush in-flight pixels before kicking off
    /// a synchronous thumbnail render. See `RenderCommand::RenderThumbnail`.
    pub(crate) fn harvest_pending(&mut self) -> Result<Option<RenderResult>, String> {
        self.harvest_pending_readback()
    }

    /// Wait on any pending readback and return its data as a `RenderResult`.
    pub(super) fn harvest_pending_readback(&mut self) -> Result<Option<RenderResult>, String> {
        let pending = match self.pending_readback.take() {
            Some(p) => p,
            None => return Ok(None),
        };

        (pending.wait_fn)()?;

        let pixel_data = pending
            .readback_buffer
            .read()
            .map_err(|e| format!("render: readback buffer read failed: {e}"))?
            .to_vec();

        let exported_frame = output_export::ExportedFrame {
            pixel_data: output_export::ExportedPixelData::CpuReadback(Arc::new(pixel_data)),
            extent: pending.extent,
            timestamp_nanos: pending.timestamp_nanos,
            gpu_token_id: self.frame_counter.saturating_sub(1),
            export_metadata: output_export::ExportMetadata {
                width: pending.extent[0],
                height: pending.extent[1],
                has_alpha: true,
                color_space: pending.color_space.clone(),
                timestamp_nanos: pending.timestamp_nanos,
            },
            handoff_path: crate::frame_handoff::HandoffPath::CpuReadback,
            fallback_reason: Some(crate::frame_handoff::FallbackReason::RequestedCpuReadback),
        };

        Ok(Some(RenderResult {
            extent: pending.extent,
            timestamp_nanos: pending.timestamp_nanos,
            has_alpha: true,
            stats: pending.stats,
            exported_frame: Some(exported_frame),
        }))
    }

    /// Ensure pre-allocated readback buffers exist for the given extent.
    ///
    /// Returns `(staging, readback)` where:
    /// - `staging` is a device-local + host-visible (BAR) buffer used as the
    ///   target for `copy_image_to_buffer` (fast GPU copy).
    /// - `readback` is a host-cached buffer for fast CPU reads; the render
    ///   command buffer also contains a `copy_buffer` from staging→readback
    ///   so that the CPU never reads uncached VRAM.
    #[allow(clippy::type_complexity)]
    pub(super) fn ensure_readback_buffers(
        &mut self,
        extent: [u32; 2],
    ) -> Result<(Subbuffer<[u8]>, Subbuffer<[u8]>), String> {
        let slot = self.readback_slot;
        let pixel_count = (extent[0] as u64) * (extent[1] as u64);
        let needed = pixel_count * 4;

        let needs_recreate = match &self.readback_buffers[slot] {
            Some(buf) => buf.len() != needed,
            None => true,
        };

        if needs_recreate {
            let ma = self
                .memory_allocator
                .as_ref()
                .ok_or("renderer: no memory allocator")?
                .clone();

            // Staging: device-local + host-visible (resizable BAR).
            // Falls back to plain host-visible if BAR unavailable.
            let staging = Buffer::new_slice::<u8>(
                ma.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                needed,
            )
            .or_else(|_| {
                Buffer::new_slice::<u8>(
                    ma.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                        ..Default::default()
                    },
                    needed,
                )
            })
            .map_err(|e| format!("render: staging buffer alloc failed: {e}"))?;

            // Readback: host-cached for fast CPU reads.
            let readback = Buffer::new_slice::<u8>(
                ma,
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                    ..Default::default()
                },
                needed,
            )
            .map_err(|e| format!("render: readback buffer alloc failed: {e}"))?;

            self.staging_buffers[slot] = Some(staging);
            self.readback_buffers[slot] = Some(readback);
        }

        // Advance slot for next frame.
        self.readback_slot = (slot + 1) % READBACK_RING_SIZE;

        Ok((
            self.staging_buffers[slot].as_ref().unwrap().clone(),
            self.readback_buffers[slot].as_ref().unwrap().clone(),
        ))
    }
}
