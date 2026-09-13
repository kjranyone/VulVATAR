//! Asynchronous CPU readback ring extracted from `VulkanRenderer` as
//! the second slice of the #12 renderer split. Owns the staging /
//! readback buffer pair, the per-frame fence wrapper, and the
//! `harvest` path that turns a finished readback into a
//! `RenderResult::CpuReadback`. The render-side enqueue still lives
//! in `mod.rs` next to the rest of the per-frame command-buffer
//! construction.
//!
//! ## Zero-allocation pixel harvest
//!
//! `harvest` used to do `readback_buffer.read().to_vec()` — a fresh
//! full-frame allocation every frame (8.3 MB at 1080p) that the Windows
//! heap services through large-block virtual allocs plus first-touch
//! page faults, only for the bytes to be handed to consumers as an
//! `Arc<Vec<u8>>` that dies one or two frames later. The ring now keeps
//! one owned `Arc` clone per slot: on harvest, `Arc::try_unwrap` reclaims
//! the previous frame's allocation when every consumer has already
//! dropped theirs (the common case — the preview replaces its reference
//! each frame and the output worker copies out within its queue depth)
//! and the GPU bytes are copied into the warm, pre-faulted pages. If a
//! consumer still holds the old Arc, the unwrap fails and we fall back
//! to a fresh allocation — correct, just slower. Ring safety is
//! unchanged: the slot's GPU buffer was fenced-complete before the CPU
//! looked at it.

use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

use crate::renderer::{frame_input, output_export, RenderResult, RenderStats, VulkanRenderer};

/// The `wait_fn` closure captures the `FenceSignalFuture` by move so that we
/// can wait on it later without naming the complex concrete future type.
pub(super) struct PendingReadbackState {
    pub(super) wait_fn: Box<dyn FnOnce() -> Result<(), String> + Send>,
    pub(super) readback_buffer: Subbuffer<[u8]>,
    /// Readback ring slot the buffer belongs to — identifies the CPU
    /// pixel-Vec pool entry the harvest refills.
    pub(super) pool_slot: usize,
    /// Host buffer holding the frame's depth aspect (D32_SFLOAT NDC) when the
    /// caller enabled depth readback; `None` on the live path.
    pub(super) depth_buffer: Option<Subbuffer<[u8]>>,
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

        let guard = pending
            .readback_buffer
            .read()
            .map_err(|e| format!("render: readback buffer read failed: {e}"))?;
        let pixel_data = {
            let reuse = if Self::pixel_pool_enabled() {
                self.readback_cpu_pool[pending.pool_slot]
                    .take()
                    .and_then(|arc| Arc::try_unwrap(arc).ok())
            } else {
                None
            };
            match reuse {
                Some(mut v) => {
                    v.clear();
                    v.extend_from_slice(&guard[..]);
                    self.gpu_runtime_counters.pixel_pool_reuses += 1;
                    Arc::new(v)
                }
                None => {
                    self.gpu_runtime_counters.pixel_pool_allocs += 1;
                    Arc::new(guard.to_vec())
                }
            }
        };
        if Self::pixel_pool_enabled() {
            self.readback_cpu_pool[pending.pool_slot] = Some(Arc::clone(&pixel_data));
        }
        drop(guard);

        // Depth aspect (D32_SFLOAT NDC), when the caller enabled depth readback.
        // The buffer is oversized to the D32S8 block (see `render`); the depth
        // occupies the leading `w*h` f32s, tightly packed — take exactly those.
        let depth_ndc = match pending.depth_buffer {
            Some(ref buf) => {
                let bytes = buf
                    .read()
                    .map_err(|e| format!("render: depth readback buffer read failed: {e}"))?;
                let count = (pending.extent[0] as usize) * (pending.extent[1] as usize);
                let floats: Vec<f32> = bytes
                    .chunks_exact(4)
                    .take(count)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Some(floats)
            }
            None => None,
        };

        let exported_frame = output_export::ExportedFrame {
            pixel_data: output_export::ExportedPixelData::CpuReadback(pixel_data),
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
            // CpuReadback `pixel_data` already feeds the preview directly.
            preview_pixels: None,
        };

        Ok(Some(RenderResult {
            extent: pending.extent,
            timestamp_nanos: pending.timestamp_nanos,
            has_alpha: true,
            stats: pending.stats,
            exported_frame: Some(exported_frame),
            depth_ndc,
            // Overwritten by `render` with the current-frame cloth
            // readback before the result leaves the renderer.
            cloth_readback: Vec::new(),
            // Same overwrite discipline as `cloth_readback` above.
            vbo_audit: Vec::new(),
            sdf_fields: Vec::new(),
        }))
    }

    /// A/B knob for the pixel-Vec pool benchmark (`VULVATAR_PIXEL_POOL=0`
    /// restores the per-frame allocation). Evaluated once per process.
    pub(super) fn pixel_pool_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("VULVATAR_PIXEL_POOL").map_or(true, |v| v != "0"))
    }

    /// Ensure pre-allocated readback buffers exist for the given extent.
    ///
    /// Returns `(slot, staging, readback)` where:
    /// - `slot` is the ring index the pair belongs to (identifies the CPU
    ///   pixel-Vec pool entry refilled at harvest),
    /// - `staging` is a device-local + host-visible (BAR) buffer used as the
    ///   target for `copy_image_to_buffer` (fast GPU copy),
    /// - `readback` is a host-cached buffer for fast CPU reads; the render
    ///   command buffer also contains a `copy_buffer` from staging→readback
    ///   so that the CPU never reads uncached VRAM.
    #[allow(clippy::type_complexity)]
    pub(super) fn ensure_readback_buffers(
        &mut self,
        extent: [u32; 2],
    ) -> Result<(usize, Subbuffer<[u8]>, Subbuffer<[u8]>), String> {
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
            // The GPU-side ring buffers were swapped: a cached command
            // buffer from the old extent would copy into dead buffers.
            self.readback_cpu_pool[slot] = None;
            self.cb_cache.clear();
        }

        // Advance slot for next frame.
        self.readback_slot = (slot + 1) % READBACK_RING_SIZE;

        Ok((
            slot,
            self.staging_buffers[slot].as_ref().unwrap().clone(),
            self.readback_buffers[slot].as_ref().unwrap().clone(),
        ))
    }

    /// Ensure the depth-aspect readback buffer exists for the given
    /// extent, reusing the previous allocation when the size still
    /// matches. (The live path never enables depth readback; this
    /// serves the metric-depth benches, which used to allocate a fresh
    /// full-resolution buffer every frame.)
    pub(super) fn ensure_depth_readback_buffer(
        &mut self,
        extent: [u32; 2],
    ) -> Result<Subbuffer<[u8]>, String> {
        // vulkano validates the buffer against the combined D32_SFLOAT_S8_UINT
        // block size (8 B), even though a DEPTH-aspect copy writes the depth
        // tightly packed at 4 B/texel (Vulkan spec) into the leading `w*h*4`
        // bytes. Size for the 8 B block so validation passes; the harvest
        // reads the tightly-packed depth prefix.
        let bytes = (extent[0] as u64) * (extent[1] as u64) * 8;
        let needs_recreate = match &self.depth_readback_buffer {
            Some(buf) => buf.len() as u64 != bytes,
            None => true,
        };
        if needs_recreate {
            let ma = self
                .memory_allocator
                .as_ref()
                .ok_or("renderer: no memory allocator")?
                .clone();
            let buf = Buffer::new_slice::<u8>(
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
                bytes,
            )
            .map_err(|e| format!("render: depth readback buffer alloc failed: {e}"))?;
            self.depth_readback_buffer = Some(buf);
            // The buffer handle a cached command buffer would copy into
            // changed — drop the cache.
            self.cb_cache.clear();
        }
        Ok(self.depth_readback_buffer.clone().unwrap())
    }
}
