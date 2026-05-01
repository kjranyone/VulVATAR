pub mod debug;
pub mod frame_input;
pub mod frame_pool;
pub mod gpu_handle;
pub mod material;
pub mod mtoon;
pub mod output_export;
pub mod pipeline;
#[allow(clippy::module_inception)]
mod pipeline_lint_tests;
pub mod targets;
pub mod thumbnail;
pub mod upload;

use crate::asset::{MeshId, MeshPrimitiveAsset, PrimitiveId, VertexData};
use frame_input::{ClothDeformSnapshot, RenderFrameInput};
use log::{info, warn};
use output_export::OutputExporter;
use pipeline::GpuVertex;
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo, CopyImageToBufferInfo,
    RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::sync::GpuFuture;
use vulkano::VulkanLibrary;

pub struct RenderResult {
    pub extent: [u32; 2],
    pub timestamp_nanos: u64,
    pub has_alpha: bool,
    pub stats: RenderStats,
    pub exported_frame: Option<output_export::ExportedFrame>,
}

/// Output of a one-shot thumbnail render. Carries decoded RGBA pixels
/// directly so the caller can hand them straight to a PNG encoder
/// without unwrapping the regular `RenderResult` / `ExportedFrame` chain.
#[derive(Clone, Debug)]
pub struct ThumbnailRenderResult {
    pub width: u32,
    pub height: u32,
    /// Tightly-packed RGBA8 (premultiplied alpha if the frame had a
    /// transparent background).
    pub rgba_pixels: Vec<u8>,
}

#[derive(Clone, Debug, Default)]
pub struct RenderStats {
    pub instance_count: u32,
    pub mesh_count: u32,
    pub material_count: u32,
    pub cloth_instances: u32,
}

/// Push constant layout for the outline pipeline (matches shader).
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct OutlinePushConstants {
    outline_width: f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

/// State for an in-flight readback whose GPU fence has not yet been waited on.
/// The `wait_fn` closure captures the `FenceSignalFuture` by move so that we
/// can wait on it later without naming the complex concrete future type.
struct PendingReadbackState {
    wait_fn: Box<dyn FnOnce() -> Result<(), String> + Send>,
    readback_buffer: Subbuffer<[u8]>,
    extent: [u32; 2],
    timestamp_nanos: u64,
    stats: RenderStats,
    color_space: frame_input::RenderColorSpace,
}

const READBACK_RING_SIZE: usize = 2;

#[allow(clippy::type_complexity)]
pub struct VulkanRenderer {
    initialized: bool,
    /// GPU export path (CpuReadback / GpuExport / SharedMemory). The
    /// instance is constructed and held here, but no render-loop call
    /// site delegates to it yet — kept so the wiring can be flipped on
    /// without re-creating the underlying frame pool / handle exporter.
    #[allow(dead_code)]
    output_exporter: OutputExporter,
    material_uploader: material::MaterialUploader,
    active_pipeline: Option<pipeline::PipelineState>,
    frame_counter: u64,
    /// MToon feature-coverage flags. Populated at construction with
    /// `initial_poc()` for diagnostics; no read-side yet.
    #[allow(dead_code)]
    mtoon_status: mtoon::MtoonCompatibilityStatus,
    mesh_cache: HashMap<(MeshId, PrimitiveId), (Subbuffer<[GpuVertex]>, Subbuffer<[u32]>, u32)>,
    texture_cache: HashMap<String, Arc<ImageView>>,
    device: Option<Arc<Device>>,
    queue: Option<Arc<Queue>>,
    memory_allocator: Option<Arc<StandardMemoryAllocator>>,
    command_buffer_allocator: Option<Arc<StandardCommandBufferAllocator>>,
    descriptor_set_allocator: Option<Arc<StandardDescriptorSetAllocator>>,
    render_pass: Option<Arc<RenderPass>>,
    graphics_pipeline: Option<Arc<GraphicsPipeline>>,
    graphics_pipeline_blend: Option<Arc<GraphicsPipeline>>,
    pipeline_no_cull: Option<Arc<GraphicsPipeline>>,
    pipeline_no_cull_blend: Option<Arc<GraphicsPipeline>>,
    pipeline_front_cull: Option<Arc<GraphicsPipeline>>,
    pipeline_front_cull_blend: Option<Arc<GraphicsPipeline>>,
    outline_pipeline: Option<Arc<GraphicsPipeline>>,
    sampler: Option<Arc<Sampler>>,
    default_texture_view: Option<Arc<ImageView>>,
    offscreen_color: Option<Arc<Image>>,
    offscreen_color_view: Option<Arc<ImageView>>,
    offscreen_depth: Option<Arc<Image>>,
    offscreen_depth_view: Option<Arc<ImageView>>,
    offscreen_framebuffer: Option<Arc<Framebuffer>>,
    current_extent: [u32; 2],
    /// Colour space the offscreen render target + render_pass are currently
    /// built for. Compared against `input.output_request.color_space` at the
    /// top of `render()` and triggers a rebuild of render_pass + pipelines +
    /// offscreen targets if the user changed it. Default `Srgb` matches the
    /// historical hardcoded format and keeps existing projects unaffected.
    current_color_space: frame_input::RenderColorSpace,
    camera_ring: Option<CameraRing>,
    thumb_cache: Option<ThumbnailCache>,

    // Async readback ring: two-stage (staging + readback) buffers and pending fence.
    staging_buffers: [Option<Subbuffer<[u8]>>; READBACK_RING_SIZE],
    readback_buffers: [Option<Subbuffer<[u8]>>; READBACK_RING_SIZE],
    readback_slot: usize,
    pending_readback: Option<PendingReadbackState>,
    // Cached skinning buffer + descriptor set per avatar instance slot.
    skinning_cache: Vec<SkinningCacheEntry>,
}

/// Reusable skinning buffer and its matching descriptor set.
struct SkinningCacheEntry {
    buffer: Subbuffer<[[[f32; 4]; 4]]>,
    descriptor_set: Arc<DescriptorSet>,
    capacity: usize,
}

struct ThumbnailCache {
    extent: [u32; 2],
    pipeline: Arc<GraphicsPipeline>,
    framebuffer: Arc<Framebuffer>,
    color_image: Arc<Image>,
    readback_buffer: Subbuffer<[u8]>,
}

impl Default for VulkanRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl VulkanRenderer {
    pub fn new() -> Self {
        Self {
            initialized: false,
            output_exporter: OutputExporter::new(),
            material_uploader: material::MaterialUploader::new(),
            active_pipeline: None,
            frame_counter: 0,
            mtoon_status: mtoon::MtoonCompatibilityStatus::initial_poc(),
            mesh_cache: HashMap::new(),
            texture_cache: HashMap::new(),
            device: None,
            queue: None,
            memory_allocator: None,
            command_buffer_allocator: None,
            descriptor_set_allocator: None,
            render_pass: None,
            graphics_pipeline: None,
            graphics_pipeline_blend: None,
            pipeline_no_cull: None,
            pipeline_no_cull_blend: None,
            pipeline_front_cull: None,
            pipeline_front_cull_blend: None,
            outline_pipeline: None,
            sampler: None,
            default_texture_view: None,
            offscreen_color: None,
            offscreen_color_view: None,
            offscreen_depth: None,
            offscreen_depth_view: None,
            offscreen_framebuffer: None,
            current_extent: [1920, 1080],
            current_color_space: frame_input::RenderColorSpace::Srgb,
            camera_ring: None,
            thumb_cache: None,

            staging_buffers: [None, None],
            readback_buffers: [None, None],
            readback_slot: 0,
            pending_readback: None,
            skinning_cache: Vec::new(),
        }
    }

    /// Crate-visible alias for [`Self::harvest_pending_readback`] so the
    /// render thread can drain the pipelined readback before slotting in
    /// a synchronous thumbnail render. See `RenderCommand::RenderThumbnail`.
    pub(crate) fn harvest_pending(&mut self) -> Result<Option<RenderResult>, String> {
        self.harvest_pending_readback()
    }

    /// Wait on any pending readback and return its data as a `RenderResult`.
    fn harvest_pending_readback(&mut self) -> Result<Option<RenderResult>, String> {
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
            gpu_image: None,
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
            handoff_path: crate::output::HandoffPath::CpuReadback,
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
    fn ensure_readback_buffers(
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

    /// Flush any in-flight readback (used during shutdown).
    pub fn flush_pending(&mut self) {
        if let Some(pending) = self.pending_readback.take() {
            let _ = (pending.wait_fn)();
        }
    }

    pub fn initialize(&mut self) {
        let library = VulkanLibrary::new().expect("failed to load Vulkan library");
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .expect("failed to create Vulkan instance");

        let minimal_extensions = DeviceExtensions {
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .expect("failed to enumerate physical devices")
            .filter(|p| p.supported_extensions().contains(&minimal_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(_i, q)| q.queue_flags.intersects(QueueFlags::GRAPHICS))
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            })
            .expect("no suitable physical device found");

        info!(
            "renderer: selected device: {} ({:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let mut device_extensions = DeviceExtensions {
            ..DeviceExtensions::empty()
        };

        let physical_device_exts = physical_device.supported_extensions();
        if physical_device_exts.khr_external_memory {
            device_extensions.khr_external_memory = true;
        }
        #[cfg(target_os = "windows")]
        {
            if physical_device_exts.khr_external_memory_win32 {
                device_extensions.khr_external_memory_win32 = true;
            }
        }

        if device_extensions.khr_external_memory {
            info!("renderer: enabled VK_KHR_external_memory");
        }
        #[cfg(target_os = "windows")]
        if device_extensions.khr_external_memory_win32 {
            info!("renderer: enabled VK_KHR_external_memory_win32");
        }

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create logical device");

        let queue = queues.next().expect("no queue available");

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        // Build the render pass for the renderer's currently configured
        // colour space. Defaults to sRGB; a user-driven switch to Linear
        // sRGB triggers a render-pass + pipeline rebuild later via
        // `apply_color_space`.
        let render_pass = Self::build_render_pass(device.clone(), &self.current_color_space);

        let extent = self.current_extent;
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: 0.0..=1.0,
        };

        use vulkano::pipeline::graphics::rasterization::CullMode;
        let gfx_pipeline = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Back,
            frame_input::RenderAlphaMode::Opaque,
        )
        .expect("failed to create graphics pipeline");

        let gfx_pipeline_blend = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Back,
            frame_input::RenderAlphaMode::Blend,
        )
        .expect("failed to create blend graphics pipeline");

        let no_cull_pipeline = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::None,
            frame_input::RenderAlphaMode::Opaque,
        )
        .expect("failed to create no-cull pipeline");

        let no_cull_pipeline_blend = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::None,
            frame_input::RenderAlphaMode::Blend,
        )
        .expect("failed to create blend no-cull pipeline");

        let front_cull_pipeline = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Front,
            frame_input::RenderAlphaMode::Opaque,
        )
        .expect("failed to create front-cull pipeline");

        let front_cull_pipeline_blend = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Front,
            frame_input::RenderAlphaMode::Blend,
        )
        .expect("failed to create blend front-cull pipeline");

        let outline_pipeline =
            pipeline::create_outline_pipeline(device.clone(), render_pass.clone(), viewport)
                .expect("failed to create outline pipeline");

        let (color_img, color_view, depth_img, depth_view, framebuffer) =
            Self::create_offscreen_targets(
                device.clone(),
                memory_allocator.clone(),
                render_pass.clone(),
                extent,
            );

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .expect("failed to create sampler");

        let default_texture_view = Self::create_default_white_texture(
            device.clone(),
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            queue.clone(),
        );

        self.device = Some(device);
        self.queue = Some(queue);
        self.memory_allocator = Some(memory_allocator.clone());
        self.command_buffer_allocator = Some(command_buffer_allocator);
        self.descriptor_set_allocator = Some(descriptor_set_allocator.clone());
        self.render_pass = Some(render_pass);
        self.graphics_pipeline = Some(gfx_pipeline.clone());
        self.graphics_pipeline_blend = Some(gfx_pipeline_blend);
        self.pipeline_no_cull = Some(no_cull_pipeline);
        self.pipeline_no_cull_blend = Some(no_cull_pipeline_blend);
        self.pipeline_front_cull = Some(front_cull_pipeline);
        self.pipeline_front_cull_blend = Some(front_cull_pipeline_blend);
        self.outline_pipeline = Some(outline_pipeline.clone());
        self.sampler = Some(sampler);
        self.default_texture_view = Some(default_texture_view);
        self.offscreen_color = Some(color_img);
        self.offscreen_color_view = Some(color_view);
        self.offscreen_depth = Some(depth_img);
        self.offscreen_depth_view = Some(depth_view);
        self.offscreen_framebuffer = Some(framebuffer);

        self.active_pipeline = Some(pipeline::PipelineState {
            active_pipeline: pipeline::RenderPipeline::SkinningUnlit,
            initialized: true,
            graphics_pipeline: Some(gfx_pipeline.clone()),
        });

        self.camera_ring = Some(CameraRing::new(
            &memory_allocator,
            &descriptor_set_allocator,
            &gfx_pipeline,
            &outline_pipeline,
        ));

        self.initialized = true;
        info!("renderer: initialized with Vulkano");
    }

    pub fn resize(&mut self, new_extent: [u32; 2]) -> Result<(), String> {
        if !self.initialized {
            return Ok(());
        }
        if new_extent == self.current_extent {
            return Ok(());
        }
        if new_extent[0] == 0 || new_extent[1] == 0 {
            return Ok(());
        }

        let render_pass = self
            .render_pass
            .as_ref()
            .ok_or("renderer: no render pass")?
            .clone();

        self.rebuild_pipelines_and_targets(render_pass, new_extent)
            .map_err(|e| format!("resize: {e}"))?;

        // Invalidate pre-allocated readback buffers (extent changed).
        self.staging_buffers = [None, None];
        self.readback_buffers = [None, None];

        info!("renderer: resized to {}x{}", new_extent[0], new_extent[1]);
        Ok(())
    }

    /// Recreate the seven graphics pipelines, the offscreen colour/depth
    /// images, the framebuffer, and the camera ring against the supplied
    /// render pass and extent. Used by both `resize` (extent change with
    /// the same render pass) and `apply_color_space` (render pass swapped
    /// to a different colour-attachment format). The thumbnail cache holds
    /// a pipeline keyed to the previous render pass, so it is invalidated
    /// here too.
    fn rebuild_pipelines_and_targets(
        &mut self,
        render_pass: Arc<RenderPass>,
        extent: [u32; 2],
    ) -> Result<(), String> {
        let device = self.device.as_ref().ok_or("renderer: no device")?.clone();
        let memory_allocator = self
            .memory_allocator
            .as_ref()
            .ok_or("renderer: no memory allocator")?
            .clone();

        let (color_img, color_view, depth_img, depth_view, framebuffer) =
            Self::create_offscreen_targets(
                device.clone(),
                memory_allocator,
                render_pass.clone(),
                extent,
            );

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: 0.0..=1.0,
        };
        use vulkano::pipeline::graphics::rasterization::CullMode;
        let gfx_pipeline = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Back,
            frame_input::RenderAlphaMode::Opaque,
        )
        .map_err(|e| format!("failed to create graphics pipeline: {e}"))?;

        let gfx_pipeline_blend = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Back,
            frame_input::RenderAlphaMode::Blend,
        )
        .map_err(|e| format!("failed to create blend graphics pipeline: {e}"))?;

        let no_cull_pipeline = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::None,
            frame_input::RenderAlphaMode::Opaque,
        )
        .map_err(|e| format!("failed to create no-cull pipeline: {e}"))?;

        let no_cull_pipeline_blend = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::None,
            frame_input::RenderAlphaMode::Blend,
        )
        .map_err(|e| format!("failed to create blend no-cull pipeline: {e}"))?;

        let front_cull_pipeline = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Front,
            frame_input::RenderAlphaMode::Opaque,
        )
        .map_err(|e| format!("failed to create front-cull pipeline: {e}"))?;

        let front_cull_pipeline_blend = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Front,
            frame_input::RenderAlphaMode::Blend,
        )
        .map_err(|e| format!("failed to create blend front-cull pipeline: {e}"))?;

        let outline_pipeline = pipeline::create_outline_pipeline(device, render_pass, viewport)
            .map_err(|e| format!("failed to create outline pipeline: {e}"))?;

        self.offscreen_color = Some(color_img);
        self.offscreen_color_view = Some(color_view);
        self.offscreen_depth = Some(depth_img);
        self.offscreen_depth_view = Some(depth_view);
        self.offscreen_framebuffer = Some(framebuffer);
        self.graphics_pipeline = Some(gfx_pipeline.clone());
        self.graphics_pipeline_blend = Some(gfx_pipeline_blend);
        self.pipeline_no_cull = Some(no_cull_pipeline);
        self.pipeline_no_cull_blend = Some(no_cull_pipeline_blend);
        self.pipeline_front_cull = Some(front_cull_pipeline);
        self.pipeline_front_cull_blend = Some(front_cull_pipeline_blend);
        self.outline_pipeline = Some(outline_pipeline.clone());
        self.current_extent = extent;

        if let (Some(ma), Some(dsa)) = (&self.memory_allocator, &self.descriptor_set_allocator) {
            self.camera_ring = Some(CameraRing::new(ma, dsa, &gfx_pipeline, &outline_pipeline));
        }

        if let Some(ref mut ps) = self.active_pipeline {
            ps.graphics_pipeline = Some(gfx_pipeline);
        }

        // The thumbnail pipeline is bound to the previous render pass; drop
        // the cache so the next thumbnail render rebuilds against the new
        // pass without a "incompatible render pass" panic.
        self.thumb_cache = None;

        Ok(())
    }

    /// Switch the renderer's output colour space at runtime. Rebuilds the
    /// render pass with the new colour-attachment format, then routes
    /// through `rebuild_pipelines_and_targets` so all dependent state
    /// (pipelines, offscreen targets, camera ring, thumb cache) ends up
    /// compatible. No-op when the renderer is already on `target_cs`.
    pub fn apply_color_space(
        &mut self,
        target_cs: frame_input::RenderColorSpace,
    ) -> Result<(), String> {
        if !self.initialized {
            return Ok(());
        }
        if target_cs == self.current_color_space {
            return Ok(());
        }

        let device = self.device.as_ref().ok_or("renderer: no device")?.clone();
        let new_render_pass = Self::build_render_pass(device, &target_cs);
        self.render_pass = Some(new_render_pass.clone());

        let extent = self.current_extent;
        self.rebuild_pipelines_and_targets(new_render_pass, extent)
            .map_err(|e| format!("apply_color_space: {e}"))?;

        info!(
            "renderer: switched output colour space {:?} -> {:?} (target {:?})",
            self.current_color_space,
            target_cs,
            Self::color_attachment_format(&target_cs),
        );
        self.current_color_space = target_cs;
        Ok(())
    }

    pub fn clear_caches(&mut self) {
        self.texture_cache.clear();
        self.mesh_cache.clear();
        self.material_uploader.clear_cache();
        self.skinning_cache.clear();
    }

    pub fn render(&mut self, input: &RenderFrameInput) -> Result<RenderResult, String> {
        if !self.initialized {
            warn!("renderer: skipped because Vulkan is not initialized");
            return Ok(RenderResult {
                extent: [0, 0],
                timestamp_nanos: 0,
                has_alpha: false,
                stats: RenderStats::default(),
                exported_frame: None,
            });
        }

        // Adopt the user's selected output colour space before any per-frame
        // GPU work. The path is a no-op when the renderer already matches,
        // so the cost only applies on actual user toggles.
        self.apply_color_space(input.output_request.color_space.clone())
            .map_err(|e| format!("render: color space switch failed: {e}"))?;

        let requested_extent = input.output_request.extent;
        if requested_extent != self.current_extent
            && requested_extent[0] > 0
            && requested_extent[1] > 0
        {
            self.resize(requested_extent)
                .map_err(|e| format!("render: resize failed: {e}"))?;
        }

        // ── Phase 1: Harvest the previous frame's readback ──────────────
        let harvested = self.harvest_pending_readback()?;

        // ── Phase 2: Build and submit the current frame ─────────────────

        let device = self.device.as_ref().ok_or("renderer: no device")?.clone();
        let queue = self.queue.as_ref().ok_or("renderer: no queue")?.clone();
        let memory_allocator = self
            .memory_allocator
            .as_ref()
            .ok_or("renderer: no memory allocator")?
            .clone();
        let cb_allocator = self
            .command_buffer_allocator
            .as_ref()
            .ok_or("renderer: no command buffer allocator")?
            .clone();
        let ds_allocator = self
            .descriptor_set_allocator
            .as_ref()
            .ok_or("renderer: no descriptor set allocator")?
            .clone();
        let gfx_pipeline = self
            .graphics_pipeline
            .as_ref()
            .ok_or("renderer: no graphics pipeline")?
            .clone();
        let outline_pipeline = self
            .outline_pipeline
            .as_ref()
            .ok_or("renderer: no outline pipeline")?
            .clone();
        let framebuffer = self
            .offscreen_framebuffer
            .as_ref()
            .ok_or("renderer: no framebuffer")?
            .clone();
        let color_image = self
            .offscreen_color
            .as_ref()
            .ok_or("renderer: no color image")?
            .clone();
        let sampler = self.sampler.as_ref().ok_or("renderer: no sampler")?.clone();
        let default_tex = self
            .default_texture_view
            .as_ref()
            .ok_or("renderer: no default texture")?
            .clone();

        let total_meshes: u32 = input
            .instances
            .iter()
            .map(|i| i.mesh_instances.len() as u32)
            .sum();
        let total_materials: u32 = input
            .instances
            .iter()
            .flat_map(|i| i.mesh_instances.iter())
            .count() as u32;
        let cloth_instances: u32 = input
            .instances
            .iter()
            .filter(|i| i.cloth_deform.is_some())
            .count() as u32;

        let mut builder = AutoCommandBufferBuilder::primary(
            cb_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| format!("render: failed to create command buffer: {e}"))?;

        // Use the user's Scene Background settings: transparent_background
        // clears to (0,0,0,0) so OBS chroma keys / RGBA-aware consumers can
        // composite the avatar over their own background. When false the
        // pass clears to the user's solid colour with full alpha so MF
        // virtual camera consumers (Meet, Zoom, NV12 conversion) get a
        // visible background instead of all-black.
        let bg_clear = if input.transparent_background {
            [0.0_f32, 0.0, 0.0, 0.0]
        } else {
            let [r, g, b] = input.background_color;
            [r, g, b, 1.0]
        };
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some(bg_clear.into()),
                        Some(vulkano::format::ClearValue::DepthStencil((1.0, 0))),
                    ],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .map_err(|e| format!("render: begin_render_pass failed: {e}"))?
            .bind_pipeline_graphics(gfx_pipeline.clone())
            .map_err(|e| format!("render: bind_pipeline_graphics failed: {e}"))?;

        // Camera uniform (set 0)
        let ld = input.lighting.main_light_dir_ws;
        let ld_len = (ld[0] * ld[0] + ld[1] * ld[1] + ld[2] * ld[2])
            .sqrt()
            .max(1e-6);
        let light_dir = [ld[0] / ld_len, ld[1] / ld_len, ld[2] / ld_len];

        let camera_data = CameraUniform {
            view: mat4_to_cols(input.camera.view),
            proj: mat4_to_cols(input.camera.projection),
            camera_pos: input.camera.position_ws,
            _pad0: 0.0,
            light_dir,
            light_intensity: input.lighting.main_light_intensity,
            light_color: input.lighting.main_light_color,
            _pad1: 0.0,
            ambient_term: input.lighting.ambient_term,
            _pad2: 0.0,
        };
        let slot = (self.frame_counter % FRAME_LAG as u64) as usize;
        let (camera_set, outline_camera_set) = {
            let ring = self.camera_ring.as_ref().ok_or("render: no camera ring")?;
            {
                let mut guard = ring.buffers[slot]
                    .write()
                    .map_err(|e| format!("render: camera buffer write failed: {e}"))?;
                *guard = camera_data;
            }
            (
                ring.main_sets[slot].clone(),
                ring.outline_sets[slot].clone(),
            )
        };

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                gfx_pipeline.layout().clone(),
                0,
                camera_set,
            )
            .map_err(|e| format!("render: bind camera descriptor sets failed: {e}"))?;

        // Collect outline draws for second pass
        struct OutlineDrawInfo {
            vertex_buffer: Subbuffer<[GpuVertex]>,
            index_buffer: Subbuffer<[u32]>,
            index_count: u32,
            skinning_set: Arc<DescriptorSet>,
            outline_width: f32,
            outline_color: [f32; 3],
        }
        let mut outline_draws: Vec<OutlineDrawInfo> = Vec::new();

        // Draw each avatar instance
        for (inst_idx, instance) in input.instances.iter().enumerate() {
            let skinning_mats: Vec<[[f32; 4]; 4]> = if instance.skinning_matrices.is_empty() {
                vec![mat4_cols_identity()]
            } else {
                instance.skinning_matrices.to_vec()
            };

            let mat_count = skinning_mats.len();
            let skinning_set = self.get_or_update_skinning(
                inst_idx,
                &skinning_mats,
                mat_count,
                memory_allocator.clone(),
                ds_allocator.clone(),
                &gfx_pipeline,
            )?;

            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    gfx_pipeline.layout().clone(),
                    1,
                    skinning_set.clone(),
                )
                .map_err(|e| format!("render: bind skinning descriptor sets failed: {e}"))?;

            for blend_pass in [false, true] {
                for mesh_inst in &instance.mesh_instances {
                    let is_blend =
                        matches!(mesh_inst.alpha_mode, frame_input::RenderAlphaMode::Blend);
                    if is_blend != blend_pass {
                        continue;
                    }

                    let active_pipeline = match (mesh_inst.cull_mode.clone(), is_blend) {
                        (frame_input::RenderCullMode::DoubleSided, true) => self
                            .pipeline_no_cull_blend
                            .as_ref()
                            .or(self.pipeline_no_cull.as_ref())
                            .unwrap_or(&gfx_pipeline)
                            .clone(),
                        (frame_input::RenderCullMode::FrontFace, true) => self
                            .pipeline_front_cull_blend
                            .as_ref()
                            .or(self.pipeline_front_cull.as_ref())
                            .unwrap_or(&gfx_pipeline)
                            .clone(),
                        (frame_input::RenderCullMode::BackFace, true) => self
                            .graphics_pipeline_blend
                            .as_ref()
                            .unwrap_or(&gfx_pipeline)
                            .clone(),
                        (frame_input::RenderCullMode::DoubleSided, false) => self
                            .pipeline_no_cull
                            .as_ref()
                            .unwrap_or(&gfx_pipeline)
                            .clone(),
                        (frame_input::RenderCullMode::FrontFace, false) => self
                            .pipeline_front_cull
                            .as_ref()
                            .unwrap_or(&gfx_pipeline)
                            .clone(),
                        (frame_input::RenderCullMode::BackFace, false) => gfx_pipeline.clone(),
                    };
                    builder
                        .bind_pipeline_graphics(active_pipeline.clone())
                        .map_err(|e| {
                            format!(
                                "render: bind_pipeline_graphics for cull/alpha mode failed: {e}"
                            )
                        })?;

                    let texture_view =
                        self.resolve_texture(&mesh_inst.material_binding.textures, &default_tex);
                    let shade_texture_view = self
                        .resolve_shade_texture(&mesh_inst.material_binding.textures, &default_tex);
                    let matcap_view = self
                        .resolve_matcap_texture(&mesh_inst.material_binding.textures, &default_tex);

                    let material_set = self.material_uploader.upload_to_gpu(
                        Some((mesh_inst.mesh_id, mesh_inst.primitive_id)),
                        &mesh_inst.material_binding,
                        memory_allocator.clone(),
                        ds_allocator.clone(),
                        &active_pipeline,
                        texture_view,
                        shade_texture_view,
                        sampler.clone(),
                        matcap_view,
                    )?;

                    builder
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            active_pipeline.layout().clone(),
                            2,
                            material_set,
                        )
                        .map_err(|e| {
                            format!("render: bind material descriptor sets failed: {e}")
                        })?;

                    let has_morph = !mesh_inst.morph_weights.is_empty()
                        && mesh_inst.morph_weights.iter().any(|w| w.abs() > 1e-6);

                    let (vertex_buffer, index_buffer, index_count) = if let Some(ref prim_asset) =
                        mesh_inst.primitive_data
                    {
                        if let Some(ref cloth) = instance.cloth_deform {
                            if let Some(ref vd) = prim_asset.vertices {
                                let verts = Self::build_cloth_vertices(vd, cloth);
                                let indices = prim_asset.indices.as_deref().unwrap_or(&[]);
                                let idx_count = indices.len() as u32;
                                let vb =
                                    Self::create_vertex_buffer(memory_allocator.clone(), &verts);
                                let ib =
                                    Self::create_index_buffer(memory_allocator.clone(), indices);
                                (vb, ib, idx_count)
                            } else {
                                Self::create_placeholder_mesh(memory_allocator.clone())
                            }
                        } else if has_morph {
                            if let Some(ref vd) = prim_asset.vertices {
                                let verts = Self::build_morphed_vertices(
                                    vd,
                                    &prim_asset.morph_targets,
                                    &mesh_inst.morph_weights,
                                );
                                let indices = prim_asset.indices.as_deref().unwrap_or(&[]);
                                let idx_count = indices.len() as u32;
                                let vb =
                                    Self::create_vertex_buffer(memory_allocator.clone(), &verts);
                                let ib =
                                    Self::create_index_buffer(memory_allocator.clone(), indices);
                                (vb, ib, idx_count)
                            } else {
                                Self::create_placeholder_mesh(memory_allocator.clone())
                            }
                        } else {
                            self.upload_mesh(
                                mesh_inst.mesh_id,
                                mesh_inst.primitive_id,
                                prim_asset,
                                memory_allocator.clone(),
                            )
                        }
                    } else {
                        Self::create_placeholder_mesh(memory_allocator.clone())
                    };

                    builder
                        .bind_vertex_buffers(0, vertex_buffer.clone())
                        .map_err(|e| format!("render: bind_vertex_buffers failed: {e}"))?
                        .bind_index_buffer(index_buffer.clone())
                        .map_err(|e| format!("render: bind_index_buffer failed: {e}"))?;
                    unsafe {
                        builder
                            .draw_indexed(index_count, 1, 0, 0, 0)
                            .map_err(|e| format!("render: draw_indexed failed: {e}"))?;
                    }

                    // Skip outlines for alpha-masked (Cutout) materials: the outline
                    // shader cannot do alpha testing, so it would draw in regions
                    // that the main pass discarded, creating dark jagged artifacts.
                    let is_cutout = matches!(mesh_inst.alpha_mode, frame_input::RenderAlphaMode::Cutout);
                    if !is_blend && !is_cutout && mesh_inst.outline.enabled && mesh_inst.outline.width > 0.0 {
                        outline_draws.push(OutlineDrawInfo {
                            vertex_buffer,
                            index_buffer,
                            index_count,
                            skinning_set: skinning_set.clone(),
                            outline_width: mesh_inst.outline.width,
                            outline_color: mesh_inst.outline.color,
                        });
                    }
                }
            }
        }

        // --- Outline pass ---
        if !outline_draws.is_empty() {
            builder
                .bind_pipeline_graphics(outline_pipeline.clone())
                .map_err(|e| format!("render: bind outline pipeline failed: {e}"))?;

            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    outline_pipeline.layout().clone(),
                    0,
                    outline_camera_set,
                )
                .map_err(|e| format!("render: bind outline camera descriptor sets failed: {e}"))?;

            for draw in &outline_draws {
                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        outline_pipeline.layout().clone(),
                        1,
                        draw.skinning_set.clone(),
                    )
                    .map_err(|e| {
                        format!("render: bind outline skinning descriptor sets failed: {e}")
                    })?;

                let push = OutlinePushConstants {
                    outline_width: draw.outline_width,
                    r: draw.outline_color[0],
                    g: draw.outline_color[1],
                    b: draw.outline_color[2],
                    a: 1.0,
                };
                builder
                    .push_constants(outline_pipeline.layout().clone(), 0, push)
                    .map_err(|e| format!("render: push_constants failed: {e}"))?;

                builder
                    .bind_vertex_buffers(0, draw.vertex_buffer.clone())
                    .map_err(|e| format!("render: outline bind_vertex_buffers failed: {e}"))?
                    .bind_index_buffer(draw.index_buffer.clone())
                    .map_err(|e| format!("render: outline bind_index_buffer failed: {e}"))?;
                unsafe {
                    builder
                        .draw_indexed(draw.index_count, 1, 0, 0, 0)
                        .map_err(|e| format!("render: outline draw_indexed failed: {e}"))?;
                }
            }
        }

        builder
            .end_render_pass(SubpassEndInfo::default())
            .map_err(|e| format!("render: end_render_pass failed: {e}"))?;

        // ── Readback: two-stage copy to avoid slow Intel DMA path ────
        // Stage 1 (GPU): image → device-local staging buffer (fast on-chip copy)
        // Stage 2 (GPU): staging → host-cached readback buffer (GPU memcpy)
        // The CPU then reads from the host-cached buffer without penalty.
        let (staging_buffer, readback_buffer) =
            self.ensure_readback_buffers(self.current_extent)?;

        builder
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                color_image,
                staging_buffer.clone(),
            ))
            .map_err(|e| format!("render: copy_image_to_buffer failed: {e}"))?;

        builder
            .copy_buffer(vulkano::command_buffer::CopyBufferInfo::buffers(
                staging_buffer,
                readback_buffer.clone(),
            ))
            .map_err(|e| format!("render: copy_buffer staging→readback failed: {e}"))?;

        let command_buffer = builder
            .build()
            .map_err(|e| format!("render: failed to build command buffer: {e}"))?;

        // ── Submit without waiting ──────────────────────────────────────
        let fence_future = vulkano::sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)
            .map_err(|e| format!("render: then_execute failed: {e}"))?
            .then_signal_fence_and_flush()
            .map_err(|e| format!("render: fence signal failed: {e}"))?;

        self.frame_counter += 1;
        let timestamp_nanos = self.frame_counter * 16_666_667u64;
        let extent = self.current_extent;

        let stats = RenderStats {
            instance_count: input.instances.len() as u32,
            mesh_count: total_meshes,
            material_count: total_materials,
            cloth_instances,
        };

        // Store the fence and readback buffer for harvest on the next frame.
        self.pending_readback = Some(PendingReadbackState {
            wait_fn: Box::new(move || {
                fence_future
                    .wait(None)
                    .map_err(|e| format!("render: fence wait failed: {e}"))
            }),
            readback_buffer,
            extent,
            timestamp_nanos,
            stats: stats.clone(),
            color_space: input.output_request.color_space.clone(),
        });

        // ── Return the *previous* frame's result ────────────────────────
        Ok(harvested.unwrap_or(RenderResult {
            extent,
            timestamp_nanos: 0,
            has_alpha: true,
            stats,
            exported_frame: None,
        }))
    }

    /// Render a single offscreen frame at thumbnail resolution and
    /// synchronously block on the GPU until its CPU readback is ready.
    /// Differs from [`Self::render`] in two ways:
    ///
    /// - The pipelined "harvest the previous frame" pattern is bypassed:
    ///   the caller (render thread) is expected to have drained any
    ///   in-flight regular-frame readback already, so the
    ///   `pending_readback` slot is empty on entry.
    /// - The fence wait + buffer copy happens before the call returns,
    ///   so the result reflects *this* submission, not the next one.
    ///
    /// Used by `RenderCommand::RenderThumbnail`.
    pub(crate) fn render_thumbnail(
        &mut self,
        input: &RenderFrameInput,
    ) -> Result<ThumbnailRenderResult, String> {
        // Submit. render() returns a RenderResult that's empty in the
        // typical case (we drained pending_readback before calling), so
        // the only side effect we care about is that pending_readback
        // is now populated with the thumbnail's submission.
        let _empty = self.render(input)?;

        let harvested = self
            .harvest_pending_readback()?
            .ok_or_else(|| "render_thumbnail: no pending readback after submit".to_string())?;
        let exported = harvested
            .exported_frame
            .ok_or_else(|| "render_thumbnail: no exported_frame on RenderResult".to_string())?;
        let pixels = match exported.pixel_data {
            output_export::ExportedPixelData::CpuReadback(arc) => match Arc::try_unwrap(arc) {
                Ok(v) => v,
                Err(arc) => (*arc).clone(),
            },
            _ => {
                return Err(
                    "render_thumbnail: exported pixel data isn't CPU readback".to_string()
                )
            }
        };

        Ok(ThumbnailRenderResult {
            width: exported.extent[0],
            height: exported.extent[1],
            rgba_pixels: pixels,
        })
    }

    /// Get or create a reusable skinning buffer + descriptor set for the given
    /// avatar instance slot, writing the current frame's matrices into it.
    fn get_or_update_skinning(
        &mut self,
        inst_idx: usize,
        skinning_mats: &[[[f32; 4]; 4]],
        mat_count: usize,
        memory_allocator: Arc<StandardMemoryAllocator>,
        ds_allocator: Arc<StandardDescriptorSetAllocator>,
        gfx_pipeline: &Arc<GraphicsPipeline>,
    ) -> Result<Arc<DescriptorSet>, String> {
        // Grow the cache vector if needed.
        while self.skinning_cache.len() <= inst_idx {
            // Placeholder; will be (re)created below.
            let buf = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                vec![mat4_cols_identity()],
            )
            .map_err(|e| format!("render: skinning buffer init failed: {e}"))?;

            let layout = gfx_pipeline
                .layout()
                .set_layouts()
                .get(1)
                .ok_or("render: no skinning set layout")?
                .clone();
            let ds = DescriptorSet::new(
                ds_allocator.clone(),
                layout,
                [WriteDescriptorSet::buffer(0, buf.clone())],
                [],
            )
            .map_err(|e| format!("render: skinning desc set init failed: {e}"))?;

            self.skinning_cache.push(SkinningCacheEntry {
                buffer: buf,
                descriptor_set: ds,
                capacity: 1,
            });
        }

        let entry = &mut self.skinning_cache[inst_idx];

        // If the buffer is too small, reallocate.
        if mat_count > entry.capacity {
            let buf = Buffer::from_iter(
                memory_allocator,
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                skinning_mats.iter().copied(),
            )
            .map_err(|e| format!("render: skinning buffer realloc failed: {e}"))?;

            let layout = gfx_pipeline
                .layout()
                .set_layouts()
                .get(1)
                .ok_or("render: no skinning set layout")?
                .clone();
            let ds = DescriptorSet::new(
                ds_allocator.clone(),
                layout,
                [WriteDescriptorSet::buffer(0, buf.clone())],
                [],
            )
            .map_err(|e| format!("render: skinning desc set realloc failed: {e}"))?;

            entry.buffer = buf;
            entry.descriptor_set = ds;
            entry.capacity = mat_count;

            return Ok(entry.descriptor_set.clone());
        }

        // Buffer is large enough — write in-place.
        {
            let mut guard = entry
                .buffer
                .write()
                .map_err(|e| format!("render: skinning buffer write failed: {e}"))?;
            guard[..mat_count].copy_from_slice(skinning_mats);
        }

        Ok(entry.descriptor_set.clone())
    }

    // -----------------------------------------------------------------------
    // Texture upload helpers
    // -----------------------------------------------------------------------

    fn resolve_texture(
        &mut self,
        slots: &material::MaterialTextureSlots,
        default_tex: &Arc<ImageView>,
    ) -> Arc<ImageView> {
        let binding = match slots.base_color.as_ref() {
            Some(b) => b,
            None => return default_tex.clone(),
        };
        let uri = &binding.uri;
        if let Some(cached) = self.texture_cache.get(uri) {
            return cached.clone();
        }
        let view = if binding.pixel_data.is_some() && binding.dimensions.0 > 0 {
            self.upload_texture_from_binding(binding)
                .unwrap_or_else(|| {
                    warn!(
                        "renderer: failed to upload embedded texture '{}', using default white",
                        uri
                    );
                    default_tex.clone()
                })
        } else {
            self.upload_texture(uri).unwrap_or_else(|| {
                warn!(
                    "renderer: failed to load texture '{}', using default white",
                    uri
                );
                default_tex.clone()
            })
        };
        self.texture_cache.insert(uri.clone(), view.clone());
        view
    }

    fn resolve_matcap_texture(
        &mut self,
        slots: &material::MaterialTextureSlots,
        default_tex: &Arc<ImageView>,
    ) -> Option<Arc<ImageView>> {
        let binding = slots.matcap.as_ref()?;
        let uri = &binding.uri;
        if let Some(cached) = self.texture_cache.get(uri) {
            return Some(cached.clone());
        }
        let view = if binding.pixel_data.is_some() && binding.dimensions.0 > 0 {
            self.upload_texture_from_binding(binding)
                .unwrap_or_else(|| {
                    warn!(
                        "renderer: failed to upload embedded matcap '{}', skipping",
                        uri
                    );
                    default_tex.clone()
                })
        } else {
            self.upload_texture(uri).unwrap_or_else(|| {
                warn!(
                    "renderer: failed to load matcap texture '{}', skipping",
                    uri
                );
                default_tex.clone()
            })
        };
        self.texture_cache.insert(uri.clone(), view.clone());
        Some(view)
    }

    fn resolve_shade_texture(
        &mut self,
        slots: &material::MaterialTextureSlots,
        default_tex: &Arc<ImageView>,
    ) -> Arc<ImageView> {
        let binding = match slots.shade_ramp.as_ref() {
            Some(b) => b,
            None => return default_tex.clone(),
        };
        let uri = &binding.uri;
        if let Some(cached) = self.texture_cache.get(uri) {
            return cached.clone();
        }
        let view = if binding.pixel_data.is_some() && binding.dimensions.0 > 0 {
            self.upload_texture_from_binding(binding)
                .unwrap_or_else(|| {
                    warn!(
                        "renderer: failed to upload embedded shade texture '{}', using default white",
                        uri
                    );
                    default_tex.clone()
                })
        } else {
            self.upload_texture(uri).unwrap_or_else(|| {
                warn!(
                    "renderer: failed to load shade texture '{}', using default white",
                    uri
                );
                default_tex.clone()
            })
        };
        self.texture_cache.insert(uri.clone(), view.clone());
        view
    }

    fn upload_texture_from_binding(
        &self,
        binding: &crate::renderer::material::MaterialTextureBinding,
    ) -> Option<Arc<ImageView>> {
        let pixels = binding.pixel_data.as_ref()?;
        let (width, height) = binding.dimensions;
        Self::upload_rgba_texture(
            self.device.as_ref()?.clone(),
            self.memory_allocator.as_ref()?.clone(),
            self.command_buffer_allocator.as_ref()?.clone(),
            self.queue.as_ref()?.clone(),
            width,
            height,
            pixels,
        )
    }

    fn upload_texture(&self, uri: &str) -> Option<Arc<ImageView>> {
        let path = std::path::Path::new(uri);
        let img = image::open(path).ok()?.into_rgba8();
        let (width, height) = img.dimensions();
        let pixels: Vec<u8> = img.into_raw();
        Self::upload_rgba_texture(
            self.device.as_ref()?.clone(),
            self.memory_allocator.as_ref()?.clone(),
            self.command_buffer_allocator.as_ref()?.clone(),
            self.queue.as_ref()?.clone(),
            width,
            height,
            &pixels,
        )
    }

    fn upload_rgba_texture(
        device: Arc<Device>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        cb_allocator: Arc<StandardCommandBufferAllocator>,
        queue: Arc<Queue>,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> Option<Arc<ImageView>> {
        let staging_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            pixels.iter().copied(),
        )
        .ok()?;

        let gpu_image = Image::new(
            memory_allocator,
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_SRGB,
                extent: [width, height, 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .ok()?;

        let mut cmd = AutoCommandBufferBuilder::primary(
            cb_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .ok()?;

        cmd.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            staging_buffer,
            gpu_image.clone(),
        ))
        .ok()?;

        let cb = cmd.build().ok()?;
        let future = vulkano::sync::now(device)
            .then_execute(queue, cb)
            .ok()?
            .then_signal_fence_and_flush()
            .ok()?;
        future.wait(None).ok()?;

        ImageView::new_default(gpu_image).ok()
    }

    fn create_default_white_texture(
        device: Arc<Device>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        cb_allocator: Arc<StandardCommandBufferAllocator>,
        queue: Arc<Queue>,
    ) -> Arc<ImageView> {
        let pixels: [u8; 4] = [255, 255, 255, 255];
        Self::upload_rgba_texture(device, memory_allocator, cb_allocator, queue, 1, 1, &pixels)
            .expect("failed to create default white texture")
    }

    // -----------------------------------------------------------------------
    // Mesh upload helpers
    // -----------------------------------------------------------------------

    fn vertex_data_to_gpu(vd: &VertexData) -> Vec<GpuVertex> {
        let count = vd.positions.len();
        (0..count)
            .map(|i| {
                let pos = vd.positions[i];
                let norm = if i < vd.normals.len() {
                    vd.normals[i]
                } else {
                    [0.0, 1.0, 0.0]
                };
                let uv = if i < vd.uvs.len() {
                    vd.uvs[i]
                } else {
                    [0.0, 0.0]
                };
                let ji = if i < vd.joint_indices.len() {
                    let j = vd.joint_indices[i];
                    [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32]
                } else {
                    [0, 0, 0, 0]
                };
                let jw = if i < vd.joint_weights.len() {
                    vd.joint_weights[i]
                } else {
                    [1.0, 0.0, 0.0, 0.0]
                };
                GpuVertex {
                    position: pos,
                    normal: norm,
                    uv,
                    joint_indices: ji,
                    joint_weights: jw,
                }
            })
            .collect()
    }

    #[allow(clippy::type_complexity)]
    fn upload_mesh(
        &mut self,
        mesh_id: MeshId,
        prim_id: PrimitiveId,
        prim: &MeshPrimitiveAsset,
        memory_allocator: Arc<StandardMemoryAllocator>,
    ) -> (Subbuffer<[GpuVertex]>, Subbuffer<[u32]>, u32) {
        let key = (mesh_id, prim_id);
        if let Some(cached) = self.mesh_cache.get(&key) {
            return cached.clone();
        }
        let vd = match prim.vertices.as_ref() {
            Some(vd) if !vd.positions.is_empty() => vd,
            _ => {
                let result = Self::create_placeholder_mesh(memory_allocator);
                self.mesh_cache.insert(key, result.clone());
                return result;
            }
        };
        let gpu_verts = Self::vertex_data_to_gpu(vd);
        let indices: &[u32] = prim.indices.as_deref().unwrap_or(&[]);
        let owned_indices;
        let idx_slice = if indices.is_empty() {
            owned_indices = (0..gpu_verts.len() as u32).collect::<Vec<_>>();
            &owned_indices
        } else {
            indices
        };
        let index_count = idx_slice.len() as u32;
        let vb = Self::create_vertex_buffer(memory_allocator.clone(), &gpu_verts);
        let ib = Self::create_index_buffer(memory_allocator, idx_slice);
        let result = (vb, ib, index_count);
        self.mesh_cache.insert(key, result.clone());
        result
    }

    /// Apply morph target (blend shape) deltas to base vertex data on the CPU.
    fn build_morphed_vertices(
        vd: &VertexData,
        morph_targets: &[crate::asset::MorphTargetDelta],
        morph_weights: &[f32],
    ) -> Vec<GpuVertex> {
        let count = vd.positions.len();
        (0..count)
            .map(|i| {
                let mut pos = vd.positions[i];
                let mut norm = vd.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]);

                for (ti, target) in morph_targets.iter().enumerate() {
                    let w = morph_weights.get(ti).copied().unwrap_or(0.0);
                    if w.abs() < 1e-6 {
                        continue;
                    }
                    if let Some(d) = target.position_deltas.get(i) {
                        pos[0] += w * d[0];
                        pos[1] += w * d[1];
                        pos[2] += w * d[2];
                    }
                    if let Some(d) = target.normal_deltas.get(i) {
                        norm[0] += w * d[0];
                        norm[1] += w * d[1];
                        norm[2] += w * d[2];
                    }
                }

                let uv = vd.uvs.get(i).copied().unwrap_or([0.0, 0.0]);
                let ji = if i < vd.joint_indices.len() {
                    let j = vd.joint_indices[i];
                    [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32]
                } else {
                    [0, 0, 0, 0]
                };
                let jw = vd.joint_weights.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 0.0]);

                GpuVertex {
                    position: pos,
                    normal: norm,
                    uv,
                    joint_indices: ji,
                    joint_weights: jw,
                }
            })
            .collect()
    }

    fn build_cloth_vertices(vd: &VertexData, cloth: &ClothDeformSnapshot) -> Vec<GpuVertex> {
        let count = vd.positions.len();
        (0..count)
            .map(|i| {
                let pos = if i < cloth.deformed_positions.len() {
                    cloth.deformed_positions[i]
                } else {
                    vd.positions[i]
                };
                let norm = if let Some(ref dn) = cloth.deformed_normals {
                    if i < dn.len() {
                        dn[i]
                    } else if i < vd.normals.len() {
                        vd.normals[i]
                    } else {
                        [0.0, 1.0, 0.0]
                    }
                } else if i < vd.normals.len() {
                    vd.normals[i]
                } else {
                    [0.0, 1.0, 0.0]
                };
                let uv = if i < vd.uvs.len() {
                    vd.uvs[i]
                } else {
                    [0.0, 0.0]
                };
                let ji = if i < vd.joint_indices.len() {
                    let j = vd.joint_indices[i];
                    [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32]
                } else {
                    [0, 0, 0, 0]
                };
                let jw = if i < vd.joint_weights.len() {
                    vd.joint_weights[i]
                } else {
                    [1.0, 0.0, 0.0, 0.0]
                };
                GpuVertex {
                    position: pos,
                    normal: norm,
                    uv,
                    joint_indices: ji,
                    joint_weights: jw,
                }
            })
            .collect()
    }

    fn create_vertex_buffer(
        memory_allocator: Arc<StandardMemoryAllocator>,
        vertices: &[GpuVertex],
    ) -> Subbuffer<[GpuVertex]> {
        Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices.iter().copied(),
        )
        .expect("failed to create vertex buffer")
    }

    fn create_index_buffer(
        memory_allocator: Arc<StandardMemoryAllocator>,
        indices: &[u32],
    ) -> Subbuffer<[u32]> {
        Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indices.iter().copied(),
        )
        .expect("failed to create index buffer")
    }

    /// Vulkan colour-attachment format that backs a given output colour
    /// space. Stage 2 picks between the GPU's automatic linear→sRGB encoder
    /// (`R8G8B8A8_SRGB`) and a no-encode UNORM target. Shaders write linear
    /// values in both cases — the difference is whether the GPU applies the
    /// transfer curve at store time.
    pub(crate) fn color_attachment_format(cs: &frame_input::RenderColorSpace) -> Format {
        match cs {
            frame_input::RenderColorSpace::Srgb => Format::R8G8B8A8_SRGB,
            frame_input::RenderColorSpace::LinearSrgb => Format::R8G8B8A8_UNORM,
        }
    }

    /// Build a render pass whose colour attachment matches the requested
    /// output colour space. Two arms because `single_pass_renderpass!` is a
    /// macro that doesn't accept a runtime `Format` value.
    fn build_render_pass(
        device: Arc<Device>,
        color_space: &frame_input::RenderColorSpace,
    ) -> Arc<RenderPass> {
        match color_space {
            frame_input::RenderColorSpace::Srgb => vulkano::single_pass_renderpass!(
                device,
                attachments: {
                    color: {
                        format: Format::R8G8B8A8_SRGB,
                        samples: 1,
                        load_op: Clear,
                        store_op: Store,
                    },
                    depth_stencil: {
                        format: Format::D32_SFLOAT_S8_UINT,
                        samples: 1,
                        load_op: Clear,
                        store_op: DontCare,
                    },
                },
                pass: {
                    color: [color],
                    depth_stencil: {depth_stencil},
                },
            )
            .expect("failed to create sRGB render pass"),
            frame_input::RenderColorSpace::LinearSrgb => vulkano::single_pass_renderpass!(
                device,
                attachments: {
                    color: {
                        format: Format::R8G8B8A8_UNORM,
                        samples: 1,
                        load_op: Clear,
                        store_op: Store,
                    },
                    depth_stencil: {
                        format: Format::D32_SFLOAT_S8_UINT,
                        samples: 1,
                        load_op: Clear,
                        store_op: DontCare,
                    },
                },
                pass: {
                    color: [color],
                    depth_stencil: {depth_stencil},
                },
            )
            .expect("failed to create linear render pass"),
        }
    }

    #[allow(clippy::type_complexity)]
    fn create_offscreen_targets(
        device: Arc<Device>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        render_pass: Arc<RenderPass>,
        extent: [u32; 2],
    ) -> (
        Arc<Image>,
        Arc<ImageView>,
        Arc<Image>,
        Arc<ImageView>,
        Arc<Framebuffer>,
    ) {
        // The first colour attachment of the render pass dictates the
        // output target format. Reading it back here keeps the offscreen
        // image and the render pass guaranteed-compatible after a colour
        // space switch.
        let color_format = render_pass
            .attachments()
            .first()
            .map(|a| a.format)
            .unwrap_or(Format::R8G8B8A8_SRGB);

        let mut color_create_info = ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: color_format,
            extent: [extent[0], extent[1], 1],
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC | ImageUsage::SAMPLED,
            ..Default::default()
        };

        let enabled_exts = device.enabled_extensions();
        if enabled_exts.khr_external_memory {
            use vulkano::image::ImageFormatInfo;
            use vulkano::memory::{ExternalMemoryHandleType, ExternalMemoryHandleTypes};
            let fmt_info = ImageFormatInfo {
                format: color_format,
                image_type: ImageType::Dim2d,
                usage: ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::TRANSFER_SRC
                    | ImageUsage::SAMPLED,
                external_memory_handle_type: Some(ExternalMemoryHandleType::OpaqueWin32Kmt),
                ..Default::default()
            };
            let external_ok = device
                .physical_device()
                .image_format_properties(fmt_info)
                .ok()
                .flatten()
                .is_some();
            if external_ok {
                color_create_info.external_memory_handle_types =
                    ExternalMemoryHandleTypes::OPAQUE_WIN32_KMT;
                info!(
                    "renderer: exportable color image ({:?}) with OPAQUE_WIN32_KMT",
                    color_format
                );
            } else {
                warn!(
                    "renderer: external memory unsupported for {:?} color target, skipping",
                    color_format
                );
            }
        }

        let color_image = Image::new(
            memory_allocator.clone(),
            color_create_info,
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .expect("failed to create offscreen color image");

        let depth_image = Image::new(
            memory_allocator,
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::D32_SFLOAT_S8_UINT,
                extent: [extent[0], extent[1], 1],
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .expect("failed to create offscreen depth image");

        let color_view = ImageView::new_default(color_image.clone()).unwrap();
        let depth_view = ImageView::new_default(depth_image.clone()).unwrap();

        let framebuffer = Framebuffer::new(
            render_pass,
            FramebufferCreateInfo {
                attachments: vec![color_view.clone(), depth_view.clone()],
                ..Default::default()
            },
        )
        .expect("failed to create offscreen framebuffer");

        (
            color_image,
            color_view,
            depth_image,
            depth_view,
            framebuffer,
        )
    }

    pub fn render_to_thumbnail(
        &mut self,
        input: &RenderFrameInput,
        thumb_width: u32,
        thumb_height: u32,
    ) -> Result<Vec<u8>, String> {
        if !self.initialized {
            return Err("renderer: not initialized".to_string());
        }

        // Thumbnails are written to disk as PNG and viewed by file browsers
        // that expect sRGB-encoded bytes. The thumbnail caller pins
        // `output_request.color_space = Srgb`; honour that here so a user
        // who has the main output on Linear sRGB still gets a correctly
        // encoded thumbnail. Costs one render-pass rebuild on the way in
        // and one on the next regular render, but thumbnails are infrequent.
        self.apply_color_space(input.output_request.color_space.clone())
            .map_err(|e| format!("thumbnail: color space switch failed: {e}"))?;

        let device = self.device.as_ref().ok_or("renderer: no device")?.clone();
        let queue = self.queue.as_ref().ok_or("renderer: no queue")?.clone();
        let memory_allocator = self
            .memory_allocator
            .as_ref()
            .ok_or("renderer: no memory allocator")?
            .clone();
        let cb_allocator = self
            .command_buffer_allocator
            .as_ref()
            .ok_or("renderer: no cb allocator")?
            .clone();
        let ds_allocator = self
            .descriptor_set_allocator
            .as_ref()
            .ok_or("renderer: no ds allocator")?
            .clone();
        let render_pass = self
            .render_pass
            .as_ref()
            .ok_or("renderer: no render pass")?
            .clone();
        let sampler = self.sampler.as_ref().ok_or("renderer: no sampler")?.clone();
        let default_tex = self
            .default_texture_view
            .as_ref()
            .ok_or("renderer: no default texture")?
            .clone();

        let extent = [thumb_width.max(1), thumb_height.max(1)];

        if self
            .thumb_cache
            .as_ref()
            .is_none_or(|c| c.extent != extent)
        {
            let viewport = Viewport {
                offset: [0.0, 0.0],
                extent: [extent[0] as f32, extent[1] as f32],
                depth_range: 0.0..=1.0,
            };

            let thumb_pipeline = pipeline::create_graphics_pipeline(
                device.clone(),
                render_pass.clone(),
                viewport,
                vulkano::pipeline::graphics::rasterization::CullMode::Back,
                frame_input::RenderAlphaMode::Opaque,
            )
            .map_err(|e| format!("thumbnail: pipeline creation failed: {e}"))?;

            let (color_img, _color_view, _depth_img, _depth_view, framebuffer) =
                Self::create_offscreen_targets(
                    device.clone(),
                    memory_allocator.clone(),
                    render_pass,
                    extent,
                );

            let byte_count = (extent[0] * extent[1] * 4) as u64;
            let readback_buffer: Subbuffer<[u8]> = Buffer::new_slice::<u8>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                    ..Default::default()
                },
                byte_count,
            )
            .map_err(|e| format!("thumbnail: readback buffer failed: {e}"))?;

            self.thumb_cache = Some(ThumbnailCache {
                extent,
                pipeline: thumb_pipeline,
                framebuffer,
                color_image: color_img,
                readback_buffer,
            });
        }

        let tc = self.thumb_cache.as_ref().unwrap();
        let thumb_pipeline = tc.pipeline.clone();
        let framebuffer = tc.framebuffer.clone();
        let color_img = tc.color_image.clone();
        let readback_buffer = tc.readback_buffer.clone();

        let camera_data = CameraUniform {
            view: mat4_to_cols(input.camera.view),
            proj: mat4_to_cols(input.camera.projection),
            camera_pos: input.camera.position_ws,
            _pad0: 0.0,
            light_dir: {
                let ld = input.lighting.main_light_dir_ws;
                let len = (ld[0] * ld[0] + ld[1] * ld[1] + ld[2] * ld[2])
                    .sqrt()
                    .max(1e-6);
                [ld[0] / len, ld[1] / len, ld[2] / len]
            },
            light_intensity: input.lighting.main_light_intensity,
            light_color: input.lighting.main_light_color,
            _pad1: 0.0,
            ambient_term: input.lighting.ambient_term,
            _pad2: 0.0,
        };

        let camera_buffer = Buffer::from_data(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            camera_data,
        )
        .map_err(|e| format!("thumbnail: camera buffer failed: {e}"))?;

        let camera_set_layout = thumb_pipeline
            .layout()
            .set_layouts()
            .first()
            .ok_or("thumbnail: no set 0")?
            .clone();
        let camera_set = DescriptorSet::new(
            ds_allocator.clone(),
            camera_set_layout,
            [WriteDescriptorSet::buffer(0, camera_buffer)],
            [],
        )
        .map_err(|e| format!("thumbnail: camera desc set failed: {e}"))?;

        let mut builder = AutoCommandBufferBuilder::primary(
            cb_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| format!("thumbnail: cmd buffer failed: {e}"))?;

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some([0.2, 0.2, 0.2, 1.0].into()),
                        Some(vulkano::format::ClearValue::DepthStencil((1.0, 0))),
                    ],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .map_err(|e| format!("thumbnail: begin_render_pass failed: {e}"))?
            .bind_pipeline_graphics(thumb_pipeline.clone())
            .map_err(|e| format!("thumbnail: bind pipeline failed: {e}"))?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                thumb_pipeline.layout().clone(),
                0,
                camera_set,
            )
            .map_err(|e| format!("thumbnail: bind camera set failed: {e}"))?;

        for instance in &input.instances {
            let skinning_mats: Vec<[[f32; 4]; 4]> = if instance.skinning_matrices.is_empty() {
                vec![mat4_cols_identity()]
            } else {
                instance.skinning_matrices.to_vec()
            };

            let skinning_buffer = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                skinning_mats,
            )
            .map_err(|e| format!("thumbnail: skinning buffer failed: {e}"))?;

            let skinning_set_layout = thumb_pipeline
                .layout()
                .set_layouts()
                .get(1)
                .ok_or("thumbnail: no set 1")?
                .clone();
            let skinning_set = DescriptorSet::new(
                ds_allocator.clone(),
                skinning_set_layout,
                [WriteDescriptorSet::buffer(0, skinning_buffer)],
                [],
            )
            .map_err(|e| format!("thumbnail: skinning desc set failed: {e}"))?;

            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    thumb_pipeline.layout().clone(),
                    1,
                    skinning_set,
                )
                .map_err(|e| format!("thumbnail: bind skinning set failed: {e}"))?;

            for mesh_inst in &instance.mesh_instances {
                let texture_view =
                    self.resolve_texture(&mesh_inst.material_binding.textures, &default_tex);

                let shade_texture_view =
                    self.resolve_shade_texture(&mesh_inst.material_binding.textures, &default_tex);

                let matcap_view =
                    self.resolve_matcap_texture(&mesh_inst.material_binding.textures, &default_tex);

                let material_set = self
                    .material_uploader
                    .upload_to_gpu(
                        None,
                        &mesh_inst.material_binding,
                        memory_allocator.clone(),
                        ds_allocator.clone(),
                        &thumb_pipeline,
                        texture_view,
                        shade_texture_view,
                        sampler.clone(),
                        matcap_view,
                    )
                    .map_err(|e| format!("thumbnail: material upload failed: {e}"))?;

                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        thumb_pipeline.layout().clone(),
                        2,
                        material_set,
                    )
                    .map_err(|e| format!("thumbnail: bind material set failed: {e}"))?;

                let prim_asset = match mesh_inst.primitive_data.as_ref() {
                    Some(p) => p.as_ref(),
                    None => {
                        let (vb, ib, idx_count) =
                            Self::create_placeholder_mesh(memory_allocator.clone());
                        builder
                            .bind_vertex_buffers(0, vb)
                            .map_err(|e| format!("thumbnail: bind vb failed: {e}"))?
                            .bind_index_buffer(ib)
                            .map_err(|e| format!("thumbnail: bind ib failed: {e}"))?;
                        unsafe {
                            builder
                                .draw_indexed(idx_count, 1, 0, 0, 0)
                                .map_err(|e| format!("thumbnail: draw failed: {e}"))?;
                        }
                        continue;
                    }
                };

                let (vb, ib, idx_count) = if let Some(ref cloth_snap) = instance.cloth_deform {
                    if let Some(ref vd) = prim_asset.vertices {
                        let cloth_verts = Self::build_cloth_vertices(vd, cloth_snap);
                        let cloth_vb = Buffer::from_iter(
                            memory_allocator.clone(),
                            BufferCreateInfo {
                                usage: BufferUsage::VERTEX_BUFFER,
                                ..Default::default()
                            },
                            AllocationCreateInfo {
                                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                                ..Default::default()
                            },
                            cloth_verts,
                        )
                        .map_err(|e| format!("thumbnail: cloth vb failed: {e}"))?;
                        let (_, cached_ib, cached_idx) = self.upload_mesh(
                            mesh_inst.mesh_id,
                            mesh_inst.primitive_id,
                            prim_asset,
                            memory_allocator.clone(),
                        );
                        (cloth_vb, cached_ib, cached_idx)
                    } else {
                        self.upload_mesh(
                            mesh_inst.mesh_id,
                            mesh_inst.primitive_id,
                            prim_asset,
                            memory_allocator.clone(),
                        )
                    }
                } else {
                    self.upload_mesh(
                        mesh_inst.mesh_id,
                        mesh_inst.primitive_id,
                        prim_asset,
                        memory_allocator.clone(),
                    )
                };

                builder
                    .bind_vertex_buffers(0, vb)
                    .map_err(|e| format!("thumbnail: bind vb failed: {e}"))?
                    .bind_index_buffer(ib)
                    .map_err(|e| format!("thumbnail: bind ib failed: {e}"))?;
                unsafe {
                    builder
                        .draw_indexed(idx_count, 1, 0, 0, 0)
                        .map_err(|e| format!("thumbnail: draw failed: {e}"))?;
                }
            }
        }

        builder
            .end_render_pass(SubpassEndInfo::default())
            .map_err(|e| format!("thumbnail: end_render_pass failed: {e}"))?;

        builder
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                color_img,
                readback_buffer.clone(),
            ))
            .map_err(|e| format!("thumbnail: copy_image_to_buffer failed: {e}"))?;

        let command_buffer = builder
            .build()
            .map_err(|e| format!("thumbnail: build cmd buffer failed: {e}"))?;

        let future = vulkano::sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)
            .map_err(|e| format!("thumbnail: execute failed: {e}"))?
            .then_signal_fence_and_flush()
            .map_err(|e| format!("thumbnail: fence failed: {e}"))?;

        future
            .wait(None)
            .map_err(|e| format!("thumbnail: wait failed: {e}"))?;

        let buffer_guard = readback_buffer
            .read()
            .map_err(|e| format!("thumbnail: read buffer failed: {e}"))?;
        let pixels: Vec<u8> = buffer_guard.to_vec();

        Ok(pixels)
    }

    #[allow(clippy::type_complexity)]
    fn create_placeholder_mesh(
        memory_allocator: Arc<StandardMemoryAllocator>,
    ) -> (Subbuffer<[GpuVertex]>, Subbuffer<[u32]>, u32) {
        let vertices = vec![
            GpuVertex {
                position: [0.0, -0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [0.5, 0.0],
                joint_indices: [0, 0, 0, 0],
                joint_weights: [1.0, 0.0, 0.0, 0.0],
            },
            GpuVertex {
                position: [-0.5, 0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [0.0, 1.0],
                joint_indices: [0, 0, 0, 0],
                joint_weights: [1.0, 0.0, 0.0, 0.0],
            },
            GpuVertex {
                position: [0.5, 0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [1.0, 1.0],
                joint_indices: [0, 0, 0, 0],
                joint_weights: [1.0, 0.0, 0.0, 0.0],
            },
        ];
        let indices: Vec<u32> = vec![0, 1, 2];

        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .expect("failed to create vertex buffer");

        let index_buffer = Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indices,
        )
        .expect("failed to create index buffer");

        (vertex_buffer, index_buffer, 3)
    }
}

// ---------------------------------------------------------------------------
// Camera uniform layout (matches shader `CameraData`)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct CameraUniform {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _pad0: f32,
    light_dir: [f32; 3],
    light_intensity: f32,
    light_color: [f32; 3],
    _pad1: f32,
    ambient_term: [f32; 3],
    _pad2: f32,
}

/// Convert row-major camera matrices into column-major for GLSL.
fn mat4_to_cols(m: crate::asset::Mat4) -> [[f32; 4]; 4] {
    [
        [m[0][0], m[1][0], m[2][0], m[3][0]],
        [m[0][1], m[1][1], m[2][1], m[3][1]],
        [m[0][2], m[1][2], m[2][2], m[3][2]],
        [m[0][3], m[1][3], m[2][3], m[3][3]],
    ]
}

/// Skinning and TRS matrices are already stored column-major in the asset/avatar path.
fn mat4_cols_identity() -> [[f32; 4]; 4] {
    crate::asset::identity_matrix()
}

const FRAME_LAG: usize = 3;

struct CameraRing {
    buffers: Vec<Subbuffer<CameraUniform>>,
    main_sets: Vec<Arc<DescriptorSet>>,
    outline_sets: Vec<Arc<DescriptorSet>>,
}

impl CameraRing {
    fn new(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        ds_allocator: &Arc<StandardDescriptorSetAllocator>,
        main_pipeline: &Arc<GraphicsPipeline>,
        outline_pipeline: &Arc<GraphicsPipeline>,
    ) -> Self {
        let main_layout = main_pipeline
            .layout()
            .set_layouts()
            .first()
            .expect("main pipeline has no set 0")
            .clone();
        let outline_layout = outline_pipeline
            .layout()
            .set_layouts()
            .first()
            .expect("outline pipeline has no set 0")
            .clone();

        let initial = CameraUniform {
            view: [[0.0; 4]; 4],
            proj: [[0.0; 4]; 4],
            camera_pos: [0.0; 3],
            _pad0: 0.0,
            light_dir: [0.0; 3],
            light_intensity: 0.0,
            light_color: [0.0; 3],
            _pad1: 0.0,
            ambient_term: [0.0; 3],
            _pad2: 0.0,
        };

        let mut buffers = Vec::with_capacity(FRAME_LAG);
        let mut main_sets = Vec::with_capacity(FRAME_LAG);
        let mut outline_sets = Vec::with_capacity(FRAME_LAG);

        for _ in 0..FRAME_LAG {
            let buffer = Buffer::from_data(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::UNIFORM_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                initial,
            )
            .expect("camera ring buffer alloc failed");

            let main_set = DescriptorSet::new(
                ds_allocator.clone(),
                main_layout.clone(),
                [WriteDescriptorSet::buffer(0, buffer.clone())],
                [],
            )
            .expect("camera main desc set alloc failed");

            let outline_set = DescriptorSet::new(
                ds_allocator.clone(),
                outline_layout.clone(),
                [WriteDescriptorSet::buffer(0, buffer.clone())],
                [],
            )
            .expect("camera outline desc set alloc failed");

            buffers.push(buffer);
            main_sets.push(main_set);
            outline_sets.push(outline_set);
        }

        CameraRing {
            buffers,
            main_sets,
            outline_sets,
        }
    }
}