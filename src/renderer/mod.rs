mod cloth_cache;
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
mod pipeline_targets;
mod readback;
mod texture_cache;
pub mod thumbnail;
mod transform_cache;

use crate::asset::{MeshId, PrimitiveId};
use frame_input::RenderFrameInput;
use log::{info, warn};
use output_export::OutputExporter;
use pipeline::{GpuVertex, GpuVertexBase, TransformControl};
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo,
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
use vulkano::image::Image;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, RenderPass};
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
    pub export_pool: output_export::ExportImagePoolStats,
}

/// Cumulative counters that prove the renderer has reached a steady state
/// after avatar warm-up rather than quietly recreating dynamic resources.
#[derive(Clone, Debug, Default)]
struct GpuRuntimeCounters {
    morph_gpu_resource_creations: u64,
    morph_ubo_writes: u64,
    cloth_cache_creations: u64,
    cloth_vbo_writes: u64,
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

/// Hard cap on morph targets per primitive the compute prepass supports.
/// VRoid Studio's default VRM ships ≲ 100 targets on the face primitive
/// (mouth / eye / brow shapes), so 256 leaves comfortable headroom.
/// Exceeding it logs a warning and clamps; this would only happen on
/// hand-authored avatars with unusually dense expression rigs. Mirrors
/// `pipeline::TransformControl::weights` length × 4.
const MORPH_MAX_TARGETS: usize = 256;

/// Workgroup size of `pipeline::transform_cs` (`local_size_x = 64`).
/// Used to compute the dispatch count `ceil(vertex_count / 64)`.
const TRANSFORM_LOCAL_SIZE: u32 = 64;

/// Persistent GPU resources backing the per-primitive compute prepass.
/// Allocated once on first encounter of a primitive and reused for the
/// lifetime of the avatar. The compute dispatch reads `base_ssbo` +
/// `morph_deltas` + `cloth_pos_ssbo` + `cloth_norm_ssbo` + `control_ubo`
/// and writes `transformed_vbo`, which the graphics pipelines bind as
/// their vertex buffer.
///
/// Stub-or-real selection at allocation time keeps the descriptor set
/// pinned for the slot's lifetime: a primitive without morph targets
/// points `morph_deltas` at the shared stub SSBO, and a primitive
/// without cloth points the cloth buffers at the same stub. The
/// `has_cloth` / `has_cloth_normals` flags in `control_ubo` make the
/// shader skip reads from stubbed bindings so it never observes
/// uninitialised data.
struct TransformGpuData {
    #[allow(dead_code)] // Held to keep the SSBO alive for the descriptor set's lifetime.
    base_ssbo: Subbuffer<[GpuVertexBase]>,
    index_buffer: Subbuffer<[u32]>,
    transformed_vbo: Subbuffer<[GpuVertex]>,
    control_ubo: Subbuffer<TransformControl>,
    #[allow(dead_code)]
    morph_deltas: Subbuffer<[[f32; 4]]>,
    cloth_pos_ssbo: Subbuffer<[[f32; 4]]>,
    cloth_norm_ssbo: Subbuffer<[[f32; 4]]>,
    transform_set: Arc<DescriptorSet>,
    index_count: u32,
    vertex_count: u32,
    target_count: u32,
    has_cloth_alloc: bool,
    has_cloth_normals_alloc: bool,
    last_cloth_version: Option<u64>,
    /// GPU cloth solver state, populated lazily on the first frame this
    /// primitive's snapshot reports `ClothSolverBackend::Gpu`. `None`
    /// for CPU-backed cloths (which take the snapshot-copy path) and
    /// for non-cloth primitives. Holding the SSBOs / descriptor sets
    /// here keeps them tied to the same lifecycle as the primitive's
    /// other compute resources.
    cloth_gpu: Option<ClothGpuSlot>,
}

/// Per-primitive GPU cloth solver resources. Built once on the first
/// frame the primitive's snapshot reports `ClothSolverBackend::Gpu`;
/// reused (with control UBO rewrites) every frame after that.
///
/// `cloth_pos_ssbo` is *not* held here — it lives on the parent
/// [`TransformGpuData`] because `transform_cs` also reads it. The
/// Verlet dispatch's descriptor set, however, binds the parent's
/// `cloth_pos_ssbo` as read-write alongside this slot's `prev_pos_ssbo`.
struct ClothGpuSlot {
    /// Counts + version mirror. `version` bumps each frame the GPU
    /// dispatch writes the position SSBO.
    state: crate::simulation::cloth_gpu_boundary::ClothGpuSimulationState,
    /// `vec4` per particle, xyz = previous position, w = pinned flag
    /// (>= 0.5 means pinned). Initialised on the first frame from the
    /// snapshot's `deformed_positions`; the integration shader rewrites
    /// it every dispatch. Held here to keep the buffer alive for the
    /// descriptor set's lifetime — Rust sees no direct read because
    /// only the GPU touches it after construction.
    #[allow(dead_code)]
    prev_pos_ssbo: Subbuffer<[[f32; 4]]>,
    /// Per-frame control block consumed by `cloth_verlet_cs`. CPU
    /// rewrites this each frame with `(dt, damping, particle_count,
    /// gravity, wind)`.
    verlet_control_ubo: Subbuffer<pipeline::ClothVerletControl>,
    /// Descriptor set 0 binding for the Verlet dispatch:
    ///   0 = parent's `cloth_pos_ssbo` (read-write)
    ///   1 = `prev_pos_ssbo` (read-write)
    ///   2 = `verlet_control_ubo` (uniform)
    verlet_set: Arc<DescriptorSet>,
    /// XPBD constraint projection resources. `None` when the cloth has
    /// no distance constraints (e.g. authoring only set up a triangulated
    /// mesh for normal recomputation).
    constraints: Option<ClothGpuConstraintResources>,
    /// Vertex normal recomputation resources. `None` when the cloth has
    /// no triangle index data.
    normals: Option<ClothGpuNormalResources>,
}

/// Per-primitive constraint-projection resources for the Jacobi
/// XPBD pipeline (3 passes per iteration: lambda update, Δx
/// accumulate, apply). All read-only buffers are written once at
/// attach time; `control_ubo`, `lambda_ssbo`, and `dlambda_ssbo`
/// mutate each frame.
struct ClothGpuConstraintResources {
    #[allow(dead_code)]
    constraint_ssbo: Subbuffer<[pipeline::ClothConstraintGpu]>,
    #[allow(dead_code)]
    delta_ssbo: Subbuffer<[[f32; 4]]>,
    #[allow(dead_code)]
    adj_offsets_ssbo: Subbuffer<[u32]>,
    #[allow(dead_code)]
    adj_constraints_ssbo: Subbuffer<[u32]>,
    control_ubo: Subbuffer<pipeline::ClothConstraintControl>,
    /// Persistent λ_j across the constraint projection iterations
    /// inside one substep. Reset to zero by the renderer (`fill_buffer`)
    /// at the start of every substep — XPBD's stiffness-independent
    /// behaviour depends on λ growing through the iterations of the
    /// CURRENT substep and being discarded at the next.
    lambda_ssbo: Subbuffer<[f32]>,
    /// Per-iteration Δλ_j written by the lambda-update pass and read
    /// by the accumulate pass. The two-buffer split (`lambda_ssbo`
    /// vs `dlambda_ssbo`) is what makes the accumulate pass safe
    /// without atomics: each constraint's Δλ is computed exactly
    /// once per iteration (by the per-constraint lambda-update
    /// dispatch) and then read by both endpoint invocations of the
    /// accumulate pass.
    #[allow(dead_code)]
    dlambda_ssbo: Subbuffer<[f32]>,
    /// Number of distance constraints — cached on the slot so the
    /// renderer can populate `ClothConstraintControl::constraint_count`
    /// each frame without re-reading the SSBO length.
    constraint_count: u32,
    /// Set 0 binding layout matches `cloth_constraint_lambda_update_cs`:
    /// `(0 positions, 1 constraints, 2 control, 3 lambda, 4 dlambda)`.
    lambda_update_set: Arc<DescriptorSet>,
    accumulate_set: Arc<DescriptorSet>,
    apply_set: Arc<DescriptorSet>,
}

/// Per-primitive normal-recomputation resources. The output normal SSBO
/// is the parent slot's `cloth_norm_ssbo` — reused so `transform_cs`
/// sees the GPU-written normals without an extra binding.
struct ClothGpuNormalResources {
    #[allow(dead_code)]
    triangle_idx_ssbo: Subbuffer<[u32]>,
    #[allow(dead_code)]
    adj_offsets_ssbo: Subbuffer<[u32]>,
    #[allow(dead_code)]
    adj_triangles_ssbo: Subbuffer<[u32]>,
    control_ubo: Subbuffer<pipeline::ClothNormalControl>,
    normal_set: Arc<DescriptorSet>,
}

/// State for an in-flight readback whose GPU fence has not yet been waited on.
use readback::{PendingReadbackState, READBACK_RING_SIZE};

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
    /// Per-primitive compute prepass resources. Built lazily on first
    /// encounter of a primitive and reused for the lifetime of the
    /// avatar; cleared by [`Self::clear_caches`] on avatar swap.
    transform_cache: HashMap<(MeshId, PrimitiveId), TransformGpuData>,
    /// Shared single-`vec4` SSBO bound at the morph / cloth slots of a
    /// primitive that has neither. The compute shader never reads past
    /// index 0 because the corresponding `has_cloth` / `target_count`
    /// flag in the control UBO is 0.
    stub_storage_ssbo: Option<Subbuffer<[[f32; 4]]>>,
    /// Compute pipeline that fuses skinning, morph-target blending, and
    /// cloth deformation into a single dispatch per (instance, primitive).
    /// Output lands in [`TransformGpuData::transformed_vbo`] which the
    /// graphics pipelines bind as their vertex buffer. Populated in
    /// [`Self::initialize`]; render paths unwrap with an error if
    /// initialization was skipped. See `pipeline::transform_cs` for
    /// the shader-side contract.
    transform_compute_pipeline: Option<Arc<ComputePipeline>>,
    /// Cloth Verlet integration compute pipeline. Dispatched per-primitive
    /// before `transform_compute_pipeline` for cloth-bearing primitives
    /// whose `ClothDeformSnapshot::solver_backend` is `Gpu`. Populated in
    /// [`Self::initialize`]. See `pipeline::cloth_verlet_cs` for the
    /// shader-side contract.
    cloth_verlet_pipeline: Option<Arc<ComputePipeline>>,
    /// Cloth XPBD lambda-update compute pipeline (pass 1 of 3 per
    /// iteration — one invocation per constraint, computes Δλ_j).
    cloth_constraint_lambda_update_pipeline: Option<Arc<ComputePipeline>>,
    /// Cloth XPBD distance-constraint Δx accumulate compute pipeline
    /// (pass 2 of 3 — per-particle, reads Δλ_j from the dlambda SSBO).
    cloth_constraint_accumulate_pipeline: Option<Arc<ComputePipeline>>,
    /// Cloth XPBD constraint apply compute pipeline (pass 3 of 3 —
    /// adds the accumulated Δx into positions, zeroes deltas).
    cloth_constraint_apply_pipeline: Option<Arc<ComputePipeline>>,
    /// Cloth vertex normal recomputation compute pipeline (S3.1).
    cloth_normal_pipeline: Option<Arc<ComputePipeline>>,
    gpu_runtime_counters: GpuRuntimeCounters,
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
            transform_cache: HashMap::new(),
            stub_storage_ssbo: None,
            transform_compute_pipeline: None,
            cloth_verlet_pipeline: None,
            cloth_constraint_lambda_update_pipeline: None,
            cloth_constraint_accumulate_pipeline: None,
            cloth_constraint_apply_pipeline: None,
            cloth_normal_pipeline: None,
            gpu_runtime_counters: GpuRuntimeCounters::default(),
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

            staging_buffers: [None, None],
            readback_buffers: [None, None],
            readback_slot: 0,
            pending_readback: None,
            skinning_cache: Vec::new(),
        }
    }

    // Readback helpers (`harvest_pending`, `harvest_pending_readback`,
    // `ensure_readback_buffers`) live in `src/renderer/readback.rs` as
    // part of the #12 module split. The render() function on this
    // struct still enqueues `PendingReadbackState` directly because it
    // is intertwined with command-buffer construction; only the
    // off-render-thread harvest + allocator helpers were moved.

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
        // Build the transform compute pipeline alongside the graphics
        // pipelines. The render loop dispatches it every frame to fuse
        // skinning + morph + cloth into world-space vertices, so failure
        // here means we cannot render at all — fail hard.
        let transform_compute_pipeline = pipeline::create_transform_compute_pipeline(
            self.device.as_ref().expect("device set above").clone(),
        )
        .expect("failed to create transform compute pipeline");
        self.transform_compute_pipeline = Some(transform_compute_pipeline);
        // Cloth Verlet integration pipeline. Only dispatched per-primitive
        // for `ClothSolverBackend::Gpu` cloths, but the pipeline itself
        // is cheap to construct upfront so attach-time GPU cloth slot
        // creation does not have to lazy-build it.
        let cloth_verlet_pipeline = pipeline::create_cloth_verlet_compute_pipeline(
            self.device.as_ref().expect("device set above").clone(),
        )
        .expect("failed to create cloth Verlet compute pipeline");
        self.cloth_verlet_pipeline = Some(cloth_verlet_pipeline);
        let cloth_constraint_lambda_update_pipeline =
            pipeline::create_cloth_constraint_lambda_update_compute_pipeline(
                self.device.as_ref().expect("device set above").clone(),
            )
            .expect("failed to create cloth constraint lambda update pipeline");
        self.cloth_constraint_lambda_update_pipeline =
            Some(cloth_constraint_lambda_update_pipeline);
        let cloth_constraint_accumulate_pipeline =
            pipeline::create_cloth_constraint_accumulate_compute_pipeline(
                self.device.as_ref().expect("device set above").clone(),
            )
            .expect("failed to create cloth constraint accumulate pipeline");
        self.cloth_constraint_accumulate_pipeline = Some(cloth_constraint_accumulate_pipeline);
        let cloth_constraint_apply_pipeline =
            pipeline::create_cloth_constraint_apply_compute_pipeline(
                self.device.as_ref().expect("device set above").clone(),
            )
            .expect("failed to create cloth constraint apply pipeline");
        self.cloth_constraint_apply_pipeline = Some(cloth_constraint_apply_pipeline);
        let cloth_normal_pipeline = pipeline::create_cloth_normal_compute_pipeline(
            self.device.as_ref().expect("device set above").clone(),
        )
        .expect("failed to create cloth normal compute pipeline");
        self.cloth_normal_pipeline = Some(cloth_normal_pipeline);
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
        self.transform_cache.clear();
        // `stub_storage_ssbo` is intentionally retained — it is a tiny
        // 1-element zero buffer with no per-avatar state, and reallocating
        // would burn one needless VRAM round-trip on every avatar swap.
        self.material_uploader.clear_cache();
        self.skinning_cache.clear();
    }

    pub(crate) fn release_export_lease(&mut self, lease_id: u64) {
        self.output_exporter.release_token(lease_id);
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
        let transform_pipeline = self
            .transform_compute_pipeline
            .as_ref()
            .ok_or("renderer: no transform compute pipeline")?
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
            .filter(|i| !i.cloth_deforms.is_empty())
            .count() as u32;

        let mut builder = AutoCommandBufferBuilder::primary(
            cb_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| format!("render: failed to create command buffer: {e}"))?;

        // ── Camera UBO update (ring slot for this frame) ────────────────
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
        let ring_slot = (self.frame_counter % FRAME_LAG as u64) as usize;
        let (camera_set, outline_camera_set) = {
            let ring = self.camera_ring.as_ref().ok_or("render: no camera ring")?;
            {
                let mut guard = ring.buffers[ring_slot]
                    .write()
                    .map_err(|e| format!("render: camera buffer write failed: {e}"))?;
                *guard = camera_data;
            }
            (
                ring.main_sets[ring_slot].clone(),
                ring.outline_sets[ring_slot].clone(),
            )
        };

        // Per-primitive draw record built during the compute prepass and
        // consumed by the graphics passes that follow. `vertex_buffer` is
        // the compute shader's output, so the graphics passes never see
        // base / morph / cloth data — only the world-space vertices.
        struct DrawInfo {
            pipeline: Arc<GraphicsPipeline>,
            alpha_mode: frame_input::RenderAlphaMode,
            vertex_buffer: Subbuffer<[GpuVertex]>,
            index_buffer: Subbuffer<[u32]>,
            index_count: u32,
            material_set: Arc<DescriptorSet>,
            outline: Option<(f32, [f32; 3])>,
        }
        let mut draws: Vec<DrawInfo> = Vec::new();

        // ── Compute prepass: fuse skinning + morph + cloth per primitive
        builder
            .bind_pipeline_compute(transform_pipeline.clone())
            .map_err(|e| format!("render: bind_pipeline_compute failed: {e}"))?;

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
                &transform_pipeline,
            )?;

            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    transform_pipeline.layout().clone(),
                    1,
                    skinning_set,
                )
                .map_err(|e| format!("render: bind compute skinning set failed: {e}"))?;

            for mesh_inst in &instance.mesh_instances {
                let prim_asset = match mesh_inst.primitive_data.as_ref() {
                    Some(p) => p.as_ref(),
                    None => continue,
                };
                let vd = match prim_asset.vertices.as_ref() {
                    Some(vd) if !vd.positions.is_empty() => vd,
                    _ => continue,
                };
                let key = (mesh_inst.mesh_id, mesh_inst.primitive_id);

                // Per-primitive cloth scope: find the first snapshot that
                // targets *this* primitive. Other primitives in the same
                // instance get `has_cloth = false` and reuse the shared
                // stub SSBO — no more "body collapses to origin because
                // the cloth solver shipped a shorter vector" footgun.
                let cloth_snap_opt = instance
                    .cloth_deforms
                    .iter()
                    .find(|c| c.target_primitive_id == mesh_inst.primitive_id);
                let has_cloth_prim = cloth_snap_opt.is_some();
                // For Gpu-backed cloth with a triangulated mesh, the
                // normal compute dispatch (S3.1) writes `cloth_norm_ssbo`
                // even though the CPU snapshot's `deformed_normals` is
                // `None` — flip the flag so `transform_cs` reads the
                // GPU-written normals.
                let has_cloth_normals_prim = cloth_snap_opt
                    .map(|c| {
                        c.deformed_normals.is_some()
                            || (c.solver_backend
                                == crate::simulation::cloth_gpu_boundary::ClothSolverBackend::Gpu
                                && c.gpu_attach
                                    .as_ref()
                                    .map(|a| !a.triangle_indices.is_empty())
                                    .unwrap_or(false))
                    })
                    .unwrap_or(false);

                self.ensure_transform_data(
                    mesh_inst.mesh_id,
                    mesh_inst.primitive_id,
                    prim_asset,
                    has_cloth_prim,
                    has_cloth_normals_prim,
                    &memory_allocator,
                    &ds_allocator,
                    &transform_pipeline,
                )?;

                // If the cloth snapshot reports GPU backend, lazily build
                // the per-primitive GPU cloth solver slot (seeding both
                // pos and prev_pos SSBOs from the rest pose). Dispatch
                // happens after the per-frame control UBO is rewritten
                // below.
                let cloth_is_gpu = cloth_snap_opt
                    .map(|c| {
                        c.solver_backend
                            == crate::simulation::cloth_gpu_boundary::ClothSolverBackend::Gpu
                    })
                    .unwrap_or(false);
                if let (true, Some(cloth_snap)) = (cloth_is_gpu, cloth_snap_opt) {
                    if let (
                        Some(attach),
                        Some(cloth_verlet_pipeline),
                        Some(cloth_constraint_lambda_update_pipeline),
                        Some(cloth_constraint_accumulate_pipeline),
                        Some(cloth_constraint_apply_pipeline),
                        Some(cloth_normal_pipeline),
                    ) = (
                        cloth_snap.gpu_attach.as_ref(),
                        self.cloth_verlet_pipeline.clone(),
                        self.cloth_constraint_lambda_update_pipeline.clone(),
                        self.cloth_constraint_accumulate_pipeline.clone(),
                        self.cloth_constraint_apply_pipeline.clone(),
                        self.cloth_normal_pipeline.clone(),
                    ) {
                        self.ensure_cloth_gpu_slot(
                            (mesh_inst.mesh_id, mesh_inst.primitive_id),
                            &cloth_snap.deformed_positions,
                            attach,
                            &memory_allocator,
                            &ds_allocator,
                            &cloth_verlet_pipeline,
                            &cloth_constraint_lambda_update_pipeline,
                            &cloth_constraint_accumulate_pipeline,
                            &cloth_constraint_apply_pipeline,
                            &cloth_normal_pipeline,
                        )?;
                    }
                }

                // Write the control UBO with this frame's morph weights and
                // cloth flags. Live-write is safe because the previous
                // frame's fence has already been waited on at the top of
                // `render` (via `harvest_pending_readback`).
                {
                    let slot = self
                        .transform_cache
                        .get(&key)
                        .expect("ensure_transform_data populated the slot");
                    let mut guard = slot
                        .control_ubo
                        .write()
                        .map_err(|e| format!("render: control UBO write failed: {e}"))?;
                    guard.vertex_count = slot.vertex_count;
                    guard.target_count = slot.target_count;
                    guard.has_cloth = if has_cloth_prim { 1 } else { 0 };
                    guard.has_cloth_normals = if has_cloth_normals_prim { 1 } else { 0 };
                    let copy_n = (slot.target_count as usize).min(MORPH_MAX_TARGETS);
                    for i in 0..copy_n {
                        let w = mesh_inst.morph_weights.get(i).copied().unwrap_or(0.0);
                        guard.weights[i / 4][i % 4] = w;
                    }
                    for i in copy_n..MORPH_MAX_TARGETS {
                        guard.weights[i / 4][i % 4] = 0.0;
                    }
                }
                self.gpu_runtime_counters.morph_ubo_writes += 1;

                // Cloth write: fill the SSBO with rest pose, then overlay
                // the snapshot's `[offset..offset+count)` slice. Vertices
                // outside the snapshot's vertex subset stay at rest pose
                // — they belong to the same primitive but are not part
                // of the cloth region.
                //
                // For `ClothSolverBackend::Gpu` cloths the GPU compute
                // dispatch (below) writes `cloth_pos_ssbo` in place; the
                // CPU snapshot copy is skipped so we don't clobber the
                // simulation state.
                let cloth_changed = if let (false, Some(cloth)) = (cloth_is_gpu, cloth_snap_opt) {
                    let slot = self
                        .transform_cache
                        .get_mut(&key)
                        .expect("slot present");
                    if slot.last_cloth_version != Some(cloth.version) {
                        let off = cloth.vertex_offset as usize;
                        let cnt = cloth.vertex_count as usize;
                        {
                            let mut pos_guard = slot
                                .cloth_pos_ssbo
                                .write()
                                .map_err(|e| format!("render: cloth pos write failed: {e}"))?;
                            for (i, dst) in pos_guard.iter_mut().enumerate() {
                                let p = if i >= off && i < off + cnt {
                                    cloth
                                        .deformed_positions
                                        .get(i - off)
                                        .copied()
                                        .unwrap_or_else(|| {
                                            vd.positions
                                                .get(i)
                                                .copied()
                                                .unwrap_or([0.0, 0.0, 0.0])
                                        })
                                } else {
                                    vd.positions.get(i).copied().unwrap_or([0.0, 0.0, 0.0])
                                };
                                *dst = [p[0], p[1], p[2], 1.0];
                            }
                        }
                        if let Some(ref normals) = cloth.deformed_normals {
                            let mut norm_guard = slot
                                .cloth_norm_ssbo
                                .write()
                                .map_err(|e| format!("render: cloth norm write failed: {e}"))?;
                            for (i, dst) in norm_guard.iter_mut().enumerate() {
                                let n = if i >= off && i < off + cnt {
                                    normals.get(i - off).copied().unwrap_or_else(|| {
                                        vd.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0])
                                    })
                                } else {
                                    vd.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0])
                                };
                                *dst = [n[0], n[1], n[2], 0.0];
                            }
                        }
                        slot.last_cloth_version = Some(cloth.version);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                if cloth_changed {
                    self.gpu_runtime_counters.cloth_vbo_writes += 1;
                }

                // GPU cloth Verlet integration dispatch. Runs before
                // `transform_cs` reads `cloth_pos_ssbo`. Inter-dispatch
                // synchronisation here relies on Vulkano 0.35's
                // `AutoCommandBufferBuilder` resource-access tracking:
                // when two dispatches in the same command buffer share
                // a buffer with conflicting access (write→read or
                // write→write), the builder inserts the appropriate
                // `VkMemoryBarrier` at the bind-point boundary. This
                // covers verlet → constraint accumulate → constraint
                // apply → normal → transform_cs because each pair
                // shares `cloth_pos_ssbo` and/or `delta_ssbo` /
                // `cloth_norm_ssbo`. If the Vulkano version is bumped
                // and auto-sync semantics change, this code needs to
                // gain explicit `synchronization_pipeline_barrier`
                // calls.
                if let (true, Some(cloth)) = (cloth_is_gpu, cloth_snap_opt) {
                    if cloth.gpu_control.is_none() {
                        // Silent skip → renderer reads stale positions.
                        // Warn loudly so the misconfiguration is visible
                        // (typically: snapshot was collected without
                        // `ClothSimState`, so the collector returned
                        // `gpu_control: None` despite backend == Gpu).
                        warn!(
                            "render: cloth backend == Gpu but gpu_control \
                             missing for primitive {:?}; cloth_pos_ssbo will \
                             not advance this frame",
                            mesh_inst.primitive_id
                        );
                    }
                    if let (Some(ctrl), Some(cloth_verlet_pipeline)) = (
                        cloth.gpu_control.as_ref(),
                        self.cloth_verlet_pipeline.clone(),
                    ) {
                        let (verlet_set, particle_count) = {
                            let slot = self
                                .transform_cache
                                .get(&key)
                                .expect("transform slot present");
                            match slot.cloth_gpu.as_ref() {
                                Some(gpu) => {
                                    (gpu.verlet_set.clone(), gpu.state.particle_count)
                                }
                                None => continue,
                            }
                        };
                        {
                            let slot = self
                                .transform_cache
                                .get_mut(&key)
                                .expect("transform slot present");
                            let gpu = slot
                                .cloth_gpu
                                .as_mut()
                                .expect("cloth_gpu present (just checked)");
                            let mut ctrl_guard = gpu
                                .verlet_control_ubo
                                .write()
                                .map_err(|e| format!("render: cloth verlet UBO write: {e}"))?;
                            *ctrl_guard = pipeline::ClothVerletControl {
                                dt: ctrl.dt,
                                damping: ctrl.damping,
                                particle_count,
                                _pad0: 0,
                                gravity: [
                                    ctrl.gravity[0],
                                    ctrl.gravity[1],
                                    ctrl.gravity[2],
                                    0.0,
                                ],
                                wind: [
                                    ctrl.wind_force[0],
                                    ctrl.wind_force[1],
                                    ctrl.wind_force[2],
                                    0.0,
                                ],
                            };
                            gpu.state.bump_version();
                        }
                        builder
                            .bind_pipeline_compute(cloth_verlet_pipeline.clone())
                            .map_err(|e| {
                                format!("render: bind cloth verlet pipeline: {e}")
                            })?;
                        builder
                            .bind_descriptor_sets(
                                PipelineBindPoint::Compute,
                                cloth_verlet_pipeline.layout().clone(),
                                0,
                                verlet_set,
                            )
                            .map_err(|e| format!("render: bind cloth verlet set: {e}"))?;
                        let groups = particle_count.div_ceil(TRANSFORM_LOCAL_SIZE);
                        unsafe {
                            builder
                                .dispatch([groups, 1, 1])
                                .map_err(|e| format!("render: cloth verlet dispatch: {e}"))?;
                        }

                        // S2.1 — XPBD constraint projection iterations.
                        let constraint_iters = ctrl.solver_iterations.max(1);
                        let constraint_resources = self
                            .transform_cache
                            .get(&key)
                            .and_then(|s| s.cloth_gpu.as_ref())
                            .and_then(|g| g.constraints.as_ref())
                            .map(|c| {
                                (
                                    c.lambda_update_set.clone(),
                                    c.accumulate_set.clone(),
                                    c.apply_set.clone(),
                                    c.control_ubo.clone(),
                                    c.lambda_ssbo.clone(),
                                    c.constraint_count,
                                )
                            });
                        if let (
                            Some((
                                lambda_update_set,
                                accumulate_set,
                                apply_set,
                                constraint_ctrl_ubo,
                                lambda_ssbo,
                                constraint_count,
                            )),
                            Some(lambda_update_pipeline),
                            Some(accumulate_pipeline),
                            Some(apply_pipeline),
                        ) = (
                            constraint_resources,
                            self.cloth_constraint_lambda_update_pipeline.clone(),
                            self.cloth_constraint_accumulate_pipeline.clone(),
                            self.cloth_constraint_apply_pipeline.clone(),
                        ) {
                            // Refresh the control UBO once per substep.
                            // particle_count + constraint_count are
                            // static for the slot; `dt` follows the
                            // substep duration so the XPBD lambda
                            // update can build `α̃ = α / dt²`.
                            {
                                let mut g = constraint_ctrl_ubo.write().map_err(|e| {
                                    format!("render: constraint UBO write: {e}")
                                })?;
                                *g = pipeline::ClothConstraintControl {
                                    particle_count,
                                    constraint_count,
                                    dt: ctrl.dt,
                                    _pad: 0,
                                };
                            }
                            // Reset lambda for this substep. XPBD's λ
                            // accumulates across the projection
                            // iterations *within* one substep, then
                            // starts fresh at the next substep — same
                            // lifecycle as `ClothSimTempBuffers::
                            // reset_lambda` on the CPU side.
                            builder
                                .fill_buffer(
                                    lambda_ssbo.clone().reinterpret::<[u32]>(),
                                    0u32,
                                )
                                .map_err(|e| {
                                    format!("render: lambda fill_buffer: {e}")
                                })?;
                            let constraint_groups =
                                constraint_count.div_ceil(64).max(1);
                            for _ in 0..constraint_iters {
                                // Pass 1: per-constraint XPBD lambda
                                // update writes Δλ_j to dlambda_ssbo
                                // and accumulates into lambda_ssbo.
                                builder
                                    .bind_pipeline_compute(lambda_update_pipeline.clone())
                                    .map_err(|e| {
                                        format!("render: bind constraint lambda update: {e}")
                                    })?;
                                builder
                                    .bind_descriptor_sets(
                                        PipelineBindPoint::Compute,
                                        lambda_update_pipeline.layout().clone(),
                                        0,
                                        lambda_update_set.clone(),
                                    )
                                    .map_err(|e| {
                                        format!("render: bind constraint lambda update set: {e}")
                                    })?;
                                unsafe {
                                    builder.dispatch([constraint_groups, 1, 1]).map_err(
                                        |e| {
                                            format!("render: constraint lambda update dispatch: {e}")
                                        },
                                    )?;
                                }
                                // Pass 2: per-particle Δx accumulate
                                // reads Δλ_j (immutable in this pass).
                                builder
                                    .bind_pipeline_compute(accumulate_pipeline.clone())
                                    .map_err(|e| {
                                        format!("render: bind constraint accumulate: {e}")
                                    })?;
                                builder
                                    .bind_descriptor_sets(
                                        PipelineBindPoint::Compute,
                                        accumulate_pipeline.layout().clone(),
                                        0,
                                        accumulate_set.clone(),
                                    )
                                    .map_err(|e| {
                                        format!("render: bind constraint accumulate set: {e}")
                                    })?;
                                unsafe {
                                    builder.dispatch([groups, 1, 1]).map_err(|e| {
                                        format!("render: constraint accumulate dispatch: {e}")
                                    })?;
                                }
                                // Pass 3: apply Δx to positions, zero
                                // deltas for next iteration.
                                builder
                                    .bind_pipeline_compute(apply_pipeline.clone())
                                    .map_err(|e| {
                                        format!("render: bind constraint apply: {e}")
                                    })?;
                                builder
                                    .bind_descriptor_sets(
                                        PipelineBindPoint::Compute,
                                        apply_pipeline.layout().clone(),
                                        0,
                                        apply_set.clone(),
                                    )
                                    .map_err(|e| {
                                        format!("render: bind constraint apply set: {e}")
                                    })?;
                                unsafe {
                                    builder.dispatch([groups, 1, 1]).map_err(|e| {
                                        format!("render: constraint apply dispatch: {e}")
                                    })?;
                                }
                            }
                        }

                        // S3.1 — vertex normal recomputation.
                        let normal_resources = self
                            .transform_cache
                            .get(&key)
                            .and_then(|s| s.cloth_gpu.as_ref())
                            .and_then(|g| g.normals.as_ref())
                            .map(|n| (n.normal_set.clone(), n.control_ubo.clone()));
                        if let (Some((normal_set, normal_ctrl_ubo)), Some(normal_pipeline)) = (
                            normal_resources,
                            self.cloth_normal_pipeline.clone(),
                        ) {
                            {
                                let mut g = normal_ctrl_ubo.write().map_err(|e| {
                                    format!("render: normal UBO write: {e}")
                                })?;
                                *g = pipeline::ClothNormalControl {
                                    vertex_count: particle_count,
                                    _pad0: 0,
                                    _pad1: 0,
                                    _pad2: 0,
                                };
                            }
                            builder
                                .bind_pipeline_compute(normal_pipeline.clone())
                                .map_err(|e| {
                                    format!("render: bind cloth normal pipeline: {e}")
                                })?;
                            builder
                                .bind_descriptor_sets(
                                    PipelineBindPoint::Compute,
                                    normal_pipeline.layout().clone(),
                                    0,
                                    normal_set,
                                )
                                .map_err(|e| {
                                    format!("render: bind cloth normal set: {e}")
                                })?;
                            unsafe {
                                builder.dispatch([groups, 1, 1]).map_err(|e| {
                                    format!("render: cloth normal dispatch: {e}")
                                })?;
                            }
                        }

                        // Switch the bound compute pipeline back to the
                        // transform pipeline so the dispatch below uses
                        // the right shader. (The descriptor set bound
                        // afterwards targets a different layout, so an
                        // explicit re-bind here is required.)
                        builder
                            .bind_pipeline_compute(transform_pipeline.clone())
                            .map_err(|e| {
                                format!("render: rebind transform pipeline: {e}")
                            })?;
                    }
                }

                // Bind set 0 + dispatch.
                let (transform_set, vertex_count, vbo, ibo, idx_count) = {
                    let slot = self.transform_cache.get(&key).expect("slot present");
                    (
                        slot.transform_set.clone(),
                        slot.vertex_count,
                        slot.transformed_vbo.clone(),
                        slot.index_buffer.clone(),
                        slot.index_count,
                    )
                };
                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        transform_pipeline.layout().clone(),
                        0,
                        transform_set,
                    )
                    .map_err(|e| format!("render: bind transform set 0 failed: {e}"))?;
                let groups = vertex_count.div_ceil(TRANSFORM_LOCAL_SIZE);
                unsafe {
                    builder
                        .dispatch([groups, 1, 1])
                        .map_err(|e| format!("render: dispatch failed: {e}"))?;
                }

                // Pick the graphics variant for the eventual draw.
                let is_blend = matches!(mesh_inst.alpha_mode, frame_input::RenderAlphaMode::Blend);
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

                // Material descriptor. Allocated against the canonical
                // `gfx_pipeline`; every variant has a structurally
                // identical set 1 layout (same shaders) so the binding
                // is valid under Vulkan descriptor set compatibility.
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
                    &gfx_pipeline,
                    texture_view,
                    shade_texture_view,
                    sampler.clone(),
                    matcap_view,
                )?;

                let is_cutout =
                    matches!(mesh_inst.alpha_mode, frame_input::RenderAlphaMode::Cutout);
                let outline_info = if !is_blend
                    && !is_cutout
                    && mesh_inst.outline.enabled
                    && mesh_inst.outline.width > 0.0
                {
                    Some((mesh_inst.outline.width, mesh_inst.outline.color))
                } else {
                    None
                };

                draws.push(DrawInfo {
                    pipeline: active_pipeline,
                    alpha_mode: mesh_inst.alpha_mode.clone(),
                    vertex_buffer: vbo,
                    index_buffer: ibo,
                    index_count: idx_count,
                    material_set,
                    outline: outline_info,
                });
            }
        }

        // ── Graphics pass: forward draws then outlines ──────────────────
        // Vulkano inserts the compute-write → vertex-input-read barrier on
        // each `transformed_vbo` automatically because the dispatch and the
        // draw share this single command buffer.
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
            .map_err(|e| format!("render: bind_pipeline_graphics failed: {e}"))?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                gfx_pipeline.layout().clone(),
                0,
                camera_set,
            )
            .map_err(|e| format!("render: bind camera descriptor set failed: {e}"))?;

        // Opaque pass then blend pass. The camera set 0 layout matches
        // across all graphics variants, so it stays bound when we swap
        // variant pipelines (Vulkan layout compatibility for set 0).
        for blend_pass in [false, true] {
            for draw in &draws {
                let is_blend = matches!(draw.alpha_mode, frame_input::RenderAlphaMode::Blend);
                if is_blend != blend_pass {
                    continue;
                }
                builder
                    .bind_pipeline_graphics(draw.pipeline.clone())
                    .map_err(|e| {
                        format!("render: bind_pipeline_graphics for variant failed: {e}")
                    })?
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        draw.pipeline.layout().clone(),
                        1,
                        draw.material_set.clone(),
                    )
                    .map_err(|e| format!("render: bind material descriptor set failed: {e}"))?
                    .bind_vertex_buffers(0, draw.vertex_buffer.clone())
                    .map_err(|e| format!("render: bind_vertex_buffers failed: {e}"))?
                    .bind_index_buffer(draw.index_buffer.clone())
                    .map_err(|e| format!("render: bind_index_buffer failed: {e}"))?;
                unsafe {
                    builder
                        .draw_indexed(draw.index_count, 1, 0, 0, 0)
                        .map_err(|e| format!("render: draw_indexed failed: {e}"))?;
                }
            }
        }

        // ── Outline pass ────────────────────────────────────────────────
        let outline_count = draws.iter().filter(|d| d.outline.is_some()).count();
        if outline_count > 0 {
            builder
                .bind_pipeline_graphics(outline_pipeline.clone())
                .map_err(|e| format!("render: bind outline pipeline failed: {e}"))?
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    outline_pipeline.layout().clone(),
                    0,
                    outline_camera_set,
                )
                .map_err(|e| format!("render: bind outline camera set failed: {e}"))?;

            for draw in &draws {
                let Some((width, color)) = draw.outline else {
                    continue;
                };
                let push = OutlinePushConstants {
                    outline_width: width,
                    r: color[0],
                    g: color[1],
                    b: color[2],
                    a: 1.0,
                };
                builder
                    .push_constants(outline_pipeline.layout().clone(), 0, push)
                    .map_err(|e| format!("render: push_constants failed: {e}"))?
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

        // ── Readback: two-stage copy to avoid slow Intel DMA path ───────
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

        let fence_future = vulkano::sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)
            .map_err(|e| format!("render: then_execute failed: {e}"))?
            .then_signal_fence_and_flush()
            .map_err(|e| format!("render: fence signal failed: {e}"))?;

        self.frame_counter += 1;
        let timestamp_nanos = self.frame_counter * 16_666_667u64;
        let extent = self.current_extent;

        if self.frame_counter.is_multiple_of(60) {
            let counters = &self.gpu_runtime_counters;
            info!(
                "GPU_RUNTIME frame={} transform_resources={} ubo_writes={} \
                 cloth_writes={} cloth_creations={} cache_slots={} readback_bytes={}",
                self.frame_counter,
                counters.morph_gpu_resource_creations,
                counters.morph_ubo_writes,
                counters.cloth_vbo_writes,
                counters.cloth_cache_creations,
                self.transform_cache.len(),
                (extent[0] as u64) * (extent[1] as u64) * 4,
            );
        }

        let stats = RenderStats {
            instance_count: input.instances.len() as u32,
            mesh_count: total_meshes,
            material_count: total_materials,
            cloth_instances,
            export_pool: self.output_exporter.export_image_pool().stats(),
        };

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

    /// Get or create a reusable skinning buffer + descriptor set for the
    /// given avatar instance slot, writing the current frame's matrices
    /// into it. The descriptor set is allocated against the compute
    /// prepass pipeline's set 1 layout because the live consumer is now
    /// `pipeline::transform_cs`, not the graphics vertex shaders.
    fn get_or_update_skinning(
        &mut self,
        inst_idx: usize,
        skinning_mats: &[[[f32; 4]; 4]],
        mat_count: usize,
        memory_allocator: Arc<StandardMemoryAllocator>,
        ds_allocator: Arc<StandardDescriptorSetAllocator>,
        transform_pipeline: &Arc<ComputePipeline>,
    ) -> Result<Arc<DescriptorSet>, String> {
        // Grow the cache vector if needed.
        while self.skinning_cache.len() <= inst_idx {
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

            let layout = transform_pipeline
                .layout()
                .set_layouts()
                .get(1)
                .ok_or("render: transform pipeline missing skinning set layout")?
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

            let layout = transform_pipeline
                .layout()
                .set_layouts()
                .get(1)
                .ok_or("render: transform pipeline missing skinning set layout")?
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


    // Cloth GPU allocator (`ensure_cloth_gpu_slot` +
    // `allocate_cloth_constraint_resources` +
    // `allocate_cloth_normal_resources`) lives in
    // `src/renderer/cloth_cache.rs` as part of the #12 module split.
    // The render-loop dispatch wiring still calls
    // `self.ensure_cloth_gpu_slot(...)` directly.

    // -----------------------------------------------------------------------
    // Texture upload helpers
    // -----------------------------------------------------------------------

    // Texture cache helpers (`resolve_texture`, `resolve_matcap_texture`,
    // `resolve_shade_texture`, `upload_rgba_texture`,
    // `create_default_white_texture`, etc.) live in
    // `src/renderer/texture_cache.rs` as part of the #12 module split.
    // The sibling submodule sees the private `VulkanRenderer` fields
    // directly, so the surface stays as `self.resolve_*(...)` at every
    // call site.

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
