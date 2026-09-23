mod background;
mod cloth_cache;
pub mod compute_prepass;
pub mod debug;
mod draw_pass;
pub mod frame_input;
mod frame_plan;
pub mod frame_pool;
mod gpu_alloc;
pub mod gpu_handle;
mod gpu_wait;
pub mod material;
pub mod mtoon;
pub mod offline;
pub mod output_export;
pub mod pipeline;
#[allow(clippy::module_inception)]
mod pipeline_lint_tests;
mod pipeline_targets;
mod post_effects;
mod readback;
pub mod sdf_field;
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
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo, PrimaryAutoCommandBuffer,
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
use vulkano::image::{Image, SampleCounts};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline, Pipeline};
use vulkano::render_pass::{Framebuffer, RenderPass};
use vulkano::sync::GpuFuture;
use vulkano::VulkanLibrary;

pub struct RenderResult {
    pub extent: [u32; 2],
    pub timestamp_nanos: u64,
    pub has_alpha: bool,
    pub stats: RenderStats,
    pub exported_frame: Option<output_export::ExportedFrame>,
    /// GPU cloth positions/normals read back from the previous frame's
    /// compute dispatches (empty unless a `Gpu`-backed cloth rendered).
    /// One frame stale by construction — read after the frame's fence
    /// — which is what makes the host-visible SSBO read coherent. The
    /// app thread folds these into `ClothState::deform_output` so
    /// CPU-side consumers (cloth inspector, backend flips) see the
    /// live solver state.
    pub cloth_readback: Vec<ClothReadback>,
    /// Final-VBO audit rows (R2) — populated only when
    /// `VULVATAR_VBO_AUDIT=1`. One row per anchor-bearing primitive:
    /// per-vertex clearance/containment correction telemetry from
    /// `transform_cs` (published in `GpuVertex.position.w`), read back
    /// from host-visible staging one frame stale (same fence semantics
    /// as `cloth_readback`).
    pub vbo_audit: Vec<VboAuditEntry>,
    /// Body-surface distance fields read back from the previous frame's
    /// splat dispatch, one row per avatar instance planned this frame.
    /// One frame stale by construction (same fence semantics as
    /// `cloth_readback`). The spring solver resolves hair against the
    /// matching avatar's field — see `simulation/sdf.rs`.
    pub sdf_fields: Vec<sdf_field::SdfFieldReadback>,
    /// Per-pixel non-linear NDC depth (`[0,1]`, `extent[0] × extent[1]`,
    /// row-major top-down, matching the colour readback) — populated only
    /// when [`VulkanRenderer::set_depth_readback`]`(true)` is active and the
    /// frame rendered at 1× (MSAA off). `None` on the live path. Consumers
    /// (`validate_gt`) linearise it against their projection to metres.
    pub depth_ndc: Option<Vec<f32>>,
}

/// Output of a one-shot thumbnail render. Carries decoded RGBA pixels
/// directly so the caller can hand them straight to a PNG encoder
/// without unwrapping the regular `RenderResult` / `ExportedFrame` chain.
/// One GPU cloth's solved state, read back to the CPU with the frame.
#[derive(Clone, Debug)]
pub struct ClothReadback {
    pub mesh_id: MeshId,
    pub primitive_id: PrimitiveId,
    /// Avatar instance that simulated this slot on the frame the
    /// readback belongs to (`AvatarInstanceId::0`, R4 delivery
    /// contract). `None` on a slot never stamped by a GPU-cloth
    /// dispatch — such rows must not be applied anywhere.
    pub instance_id: Option<u64>,
    /// Dispatch-count version mirror (`ClothGpuSimulationState::version`);
    /// monotonic per dispatched frame.
    pub version: u32,
    /// xyz per particle (the SSBO's `w` inv_mass channel dropped).
    pub positions: Vec<[f32; 3]>,
    /// Recomputed normals when the slot has the normal stage.
    pub normals: Option<Vec<[f32; 3]>>,
}

/// One anchor-bearing primitive's final-VBO audit stats (R2 detection
/// half). The correction telemetry is the total displacement the
/// render-side clearance/containment branches applied on top of the
/// physics state — the exact quantity the physics-side diagnostics
/// cannot see.
#[derive(Clone, Debug)]
pub struct VboAuditEntry {
    pub mesh_id: MeshId,
    pub primitive_id: PrimitiveId,
    /// Owning avatar instance when the slot is a GPU-cloth slot.
    pub instance_id: Option<u64>,
    pub vertex_count: usize,
    /// Vertices with a NaN position (or NaN telemetry) in the final VBO.
    pub nan_count: usize,
    /// Largest per-vertex render-side correction (metres). Spikes past
    /// `MAX_RENDER_CORRECTION_M` mean the clamp saturated on this
    /// primitive.
    pub max_correction_m: f32,
    /// 95th percentile of the per-vertex correction (metres).
    pub p95_correction_m: f32,
    /// Largest `|position|` in the final VBO (metres) — a runaway
    /// vertex shows up as a huge radius even when corrections are
    /// clamped.
    pub max_pos_len_m: f32,
}

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
    morph_weight_writes: u64,
    cloth_cache_creations: u64,
    cloth_vbo_writes: u64,
    cb_cache_hits: u64,
    cb_cache_misses: u64,
    pixel_pool_reuses: u64,
    pixel_pool_allocs: u64,
}

/// One cached command buffer plus its LRU stamp. See `frame_plan.rs` for
/// the key-safety argument (identities hashed; cached buffers pin their
/// resources so pointer keys cannot go stale through allocator reuse).
struct CachedFrameCb {
    cb: Arc<PrimaryAutoCommandBuffer>,
    last_used: u64,
}

/// LRU cap for the command-buffer cache. The steady-state frame cycle
/// needs at most (camera ring 3) × (readback ring 2) × (bg ring 2)
/// distinct shapes; the cap only bites on user-driven churn (slider
/// drags, toggles), which is exactly what should expire.
const CB_CACHE_CAP: usize = 16;

/// Workgroup size of `pipeline::transform_cs` (`local_size_x = 64`).
/// Used to compute the dispatch count `ceil(vertex_count / 64)`.
const TRANSFORM_LOCAL_SIZE: u32 = 64;

/// Persistent GPU resources backing the per-primitive compute prepass.
/// Allocated once on first encounter of a primitive and reused for the
/// lifetime of the avatar. The compute dispatch reads `base_ssbo` +
/// sparse morph resources (`morph_entries` / `morph_infos` /
/// `morph_weights_buf`) + `cloth_pos_ssbo` + `cloth_norm_ssbo` +
/// `control_ubo` and writes `transformed_vbo`, which the graphics
/// pipelines bind as their vertex buffer.
///
/// Stub-or-real selection at allocation time keeps the descriptor set
/// pinned for the slot's lifetime: a primitive without morph targets
/// points the morph bindings at shared stub SSBOs, and a primitive
/// without cloth points the cloth buffers at the same kind of stub. The
/// `has_cloth` / `has_cloth_normals` flags and `target_count == 0` in
/// `control_ubo` make the shader skip reads from stubbed bindings so it
/// never observes uninitialised data.
struct TransformGpuData {
    #[allow(dead_code)] // Held to keep the SSBO alive for the descriptor set's lifetime.
    base_ssbo: Subbuffer<[GpuVertexBase]>,
    index_buffer: Subbuffer<[u32]>,
    transformed_vbo: Subbuffer<[GpuVertex]>,
    /// R3: previous-frame FINAL vertices for slots that serve as
    /// containment parents. Written by a `copy_buffer` after every
    /// transform dispatch each frame; containment children bind THIS
    /// buffer (not `transformed_vbo`) so their clamp reads a
    /// deterministic last-frame surface instead of "whatever the
    /// parent's VBO happens to hold at dispatch time". `None` for slots
    /// that are not containment parents.
    containment_prev_vbo: Option<Subbuffer<[GpuVertex]>>,
    control_ubo: Subbuffer<TransformControl>,
    /// Sparse morph deltas: per-target runs of
    /// `(vertex_index, position_delta)` (+ normal delta at stride 2),
    /// located by `morph_infos`. Immutable after allocation.
    #[allow(dead_code)]
    morph_entries: Subbuffer<[[f32; 4]]>,
    /// `(begin, count, stride, _)` per morph target into `morph_entries`.
    /// Rewritten in-place each frame with the ACTIVE targets' rows
    /// (compacted morph gather).
    #[allow(dead_code)]
    morph_infos: Subbuffer<[[u32; 4]]>,
    /// Full per-target info rows (all authored targets) — the CPU-side
    /// source the per-frame compaction copies from.
    #[allow(dead_code)]
    morph_infos_full: Vec<[u32; 4]>,
    /// Per-frame morph weights, one float per target. Rewritten in-place
    /// by the render loop each frame (same live-write safety as
    /// `control_ubo`).
    morph_weights_buf: Subbuffer<[f32]>,
    cloth_pos_ssbo: Subbuffer<[[f32; 4]]>,
    cloth_norm_ssbo: Subbuffer<[[f32; 4]]>,
    #[allow(dead_code)]
    skin_anchors_ssbo: Subbuffer<[crate::asset::SkinAnchor]>,
    #[allow(dead_code)]
    containment_anchors_ssbo: Subbuffer<[crate::asset::SkinAnchor]>,
    transform_set: Arc<DescriptorSet>,
    index_count: u32,
    vertex_count: u32,
    target_count: u32,
    has_cloth_alloc: bool,
    has_cloth_normals_alloc: bool,
    has_skin_anchors_alloc: bool,
    has_containment_alloc: bool,
    last_cloth_version: Option<u64>,
    /// GPU cloth solver state, populated lazily on the first frame this
    /// primitive's snapshot reports `ClothSolverBackend::Gpu`. `None`
    /// for CPU-backed cloths (which take the snapshot-copy path) and
    /// for non-cloth primitives. Holding the SSBOs / descriptor sets
    /// here keeps them tied to the same lifecycle as the primitive's
    /// other compute resources.
    cloth_gpu: Option<ClothGpuSlot>,
    /// Which avatar instance simulated this slot's cloth on the most
    /// recent frame (`AvatarInstanceId::0`, R4). `transform_cache` is
    /// keyed by `(mesh_id, primitive_id)` — ids that are unique per
    /// avatar ASSET but shared across instances of the same asset — so
    /// the owner stamp is what keeps the per-frame cloth readback from
    /// being delivered to the wrong avatar. Rewritten every frame by
    /// `ensure_cloth_gpu_slot`.
    cloth_owner_instance: Option<u64>,
    /// R2 diagnostic: host-readable staging copy target for this
    /// primitive's final VBO, allocated only when `VULVATAR_VBO_AUDIT=1`
    /// and the primitive carries clearance/containment anchors (the only
    /// paths that write correction telemetry). Filled by a
    /// `copy_buffer` recorded after every transform dispatch.
    audit_staging: Option<Subbuffer<[GpuVertex]>>,
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
    /// Edge-angle bend stage. `None` when the garment has no bend
    /// constraints.
    bend: Option<ClothGpuBendResources>,
    /// Vertex normal recomputation resources. `None` when the cloth has
    /// no triangle index data.
    normals: Option<ClothGpuNormalResources>,
    collide: Option<ClothGpuCollideResources>,
    selfcol: Option<ClothGpuSelfColResources>,
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
/// Per-slot resources for the GPU collision stage (`cloth_collide_cs`).
/// The collider SSBO is world-space and moves with the bones, so the
/// prepare half rewrites it every frame and reallocates only when the
/// capsule count changes.
struct ClothGpuCollideResources {
    #[allow(dead_code)]
    collider_ssbo: Subbuffer<[pipeline::ClothGpuColliderGpu]>,
    control_ubo: Subbuffer<pipeline::ClothCollideControl>,
    collide_set: Arc<DescriptorSet>,
    collider_count: u32,
    /// Whether the descriptor set binds the avatar's real SDF field —
    /// part of the rebuild key (flipping presence reallocs the set).
    has_sdf: bool,
}

/// Per-slot resources for the GPU bend stage (T09 edge-angle hinge —
/// `cloth_bend_{update,accumulate,apply}_cs`). Attach-static topology;
/// allocated once at first plan, refreshed only when the bend table
/// changes.
pub(super) struct ClothGpuBendResources {
    #[allow(dead_code)]
    bend_ssbo: Subbuffer<[pipeline::ClothBendGpu]>,
    /// Not read on the host — held so the uniform buffer the descriptor
    /// sets point at outlives them (same keep-alive contract as
    /// `bend_ssbo`).
    #[allow(dead_code)]
    control_ubo: Subbuffer<pipeline::ClothBendControl>,
    update_set: Arc<DescriptorSet>,
    accumulate_set: Arc<DescriptorSet>,
    apply_set: Arc<DescriptorSet>,
    bend_count: u32,
}

struct ClothGpuSelfColResources {
    #[allow(dead_code)]
    adj_offsets_ssbo: Subbuffer<[u32]>,
    #[allow(dead_code)]
    adj_particles_ssbo: Subbuffer<[u32]>,
    /// Zeroed by the recording half (`fill_buffer`) before each build
    /// dispatch; TRANSFER_DST for exactly that.
    cell_counts_ssbo: Subbuffer<[u32]>,
    #[allow(dead_code)]
    cell_entries_ssbo: Subbuffer<[u32]>,
    control_ubo: Subbuffer<pipeline::ClothSelfColControl>,
    build_set: Arc<DescriptorSet>,
    resolve_set: Arc<DescriptorSet>,
}

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
    stub_skin_anchor_ssbo: Option<Subbuffer<[crate::asset::SkinAnchor]>>,
    stub_vertex_ssbo: Option<Subbuffer<[pipeline::GpuVertex]>>,
    /// Shared 1-element stubs for the sparse-morph bindings of a
    /// primitive with no morph targets (`morph_infos` / `morph_weights`
    /// shader blocks). Distinct buffer types from `stub_storage_ssbo`,
    /// so they cannot be shared with it.
    stub_uvec4_ssbo: Option<Subbuffer<[[u32; 4]]>>,
    stub_f32_ssbo: Option<Subbuffer<[f32]>>,
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
    cloth_collide_pipeline: Option<Arc<ComputePipeline>>,
    cloth_bend_update_pipeline: Option<Arc<ComputePipeline>>,
    cloth_bend_accumulate_pipeline: Option<Arc<ComputePipeline>>,
    cloth_bend_apply_pipeline: Option<Arc<ComputePipeline>>,
    cloth_selfcol_build_pipeline: Option<Arc<ComputePipeline>>,
    cloth_selfcol_resolve_pipeline: Option<Arc<ComputePipeline>>,
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
    // Cutout (alpha-tested) variants — same state as the opaque variants
    // (depth-write on, no blend) but with alpha-to-coverage enabled under
    // MSAA so alpha-tested edges (hair, foliage) antialias. One per cull mode.
    graphics_pipeline_cutout: Option<Arc<GraphicsPipeline>>,
    pipeline_no_cull_cutout: Option<Arc<GraphicsPipeline>>,
    pipeline_front_cull_cutout: Option<Arc<GraphicsPipeline>>,
    outline_pipeline: Option<Arc<GraphicsPipeline>>,
    sampler: Option<Arc<Sampler>>,
    default_texture_view: Option<Arc<ImageView>>,
    /// ClampToEdge linear sampler for the post-effect passes (the material
    /// `sampler` uses Repeat, which would wrap bloom taps across edges).
    post_sampler: Option<Arc<Sampler>>,
    /// 1x1 transparent-black texture bound as the composite's bloom input
    /// while bloom is disabled. Device-lifetime.
    black_texture_view: Option<Arc<ImageView>>,
    /// Post-effect layer (bloom chain + composite/encode pass + final 8-bit
    /// readback/export image). Rebuilt with the offscreen targets.
    post_effects: Option<post_effects::PostEffectResources>,
    /// Generative background pipeline (scene render pass, fixed viewport,
    /// baked MSAA sample count). Rebuilt with the scene pipelines.
    background_pipeline: Option<Arc<GraphicsPipeline>>,
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
    /// MSAA sample count the render pass + pipelines + offscreen targets are
    /// currently built for (1 = no MSAA). Compared against the requested
    /// level at the top of `render()`; a change rebuilds the render pass
    /// (the resolve attachment toggles on/off), the pipelines
    /// (`rasterization_samples`), and the targets. Default 1 keeps existing
    /// projects on the historical single-sample path.
    current_sample_count: u32,
    /// Multisampled colour target, present only when `current_sample_count > 1`.
    /// The subpass renders into this and resolves into `offscreen_color` (the
    /// single-sample HDR scene image the post passes sample). `None` on the
    /// no-MSAA path.
    msaa_color: Option<Arc<Image>>,
    msaa_color_view: Option<Arc<ImageView>>,
    camera_ring: Option<CameraRing>,
    /// Per-frame uniform ring for the generative background (the UBO
    /// replacement for its former push constants — `time` and the
    /// tracking anchors animate every frame and must stay out of the
    /// command buffer for the CB cache to hit). Rebuilt with the
    /// background pipeline.
    bg_uniform_ring: Option<background::BgUniformRing>,
    /// Built command buffers keyed by frame shape (see `frame_plan.rs`).
    /// Cleared at every site that swaps a command-buffer-visible
    /// resource; LRU-capped at [`CB_CACHE_CAP`].
    cb_cache: HashMap<u64, CachedFrameCb>,

    /// Per-avatar-instance body-SDF splat resources (field + params UBO
    /// + descriptor set). See `sdf_field.rs`.
    sdf_slots: sdf_field::SdfSlotMap,
    /// The splat compute pipeline (created with the other compute
    /// pipelines at device init).
    sdf_pipeline: Option<Arc<ComputePipeline>>,
    /// Instance ids whose splat dispatch was planned THIS frame — the
    /// readback window only maps fields the current frame will refresh.
    sdf_planned: Vec<u64>,

    // Async readback ring: two-stage (staging + readback) buffers and pending fence.
    staging_buffers: [Option<Subbuffer<[u8]>>; READBACK_RING_SIZE],
    readback_buffers: [Option<Subbuffer<[u8]>>; READBACK_RING_SIZE],
    /// One owned `Arc` per ring slot for the zero-allocation pixel
    /// harvest (see `readback.rs`). `Arc::try_unwrap` at harvest time
    /// reclaims the allocation once consumers dropped the frame's Arc.
    readback_cpu_pool: [Option<Arc<Vec<u8>>>; READBACK_RING_SIZE],
    readback_slot: usize,
    pending_readback: Option<PendingReadbackState>,
    /// GPU timestamp breakdown pool (`VULVATAR_RENDER_PROF=1`, on-demand
    /// instrumentation per docs/profiling.md — not a committed feature).
    prof_ts_pool: Option<(Arc<vulkano::query::QueryPool>, f32)>,
    // When set, the 1× render path copies the depth aspect to a CPU buffer and
    // surfaces it as `RenderResult::depth_ndc`. Off on the live path; enabled
    // only by the metric-depth benches (`validate_gt`) via `set_depth_readback`.
    depth_readback_enabled: bool,
    /// Cached depth-aspect readback buffer for the benches (recreated only
    // when the extent changes; was a fresh full-resolution alloc per frame).
    depth_readback_buffer: Option<Subbuffer<[u8]>>,
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
            stub_skin_anchor_ssbo: None,
            stub_vertex_ssbo: None,
            stub_uvec4_ssbo: None,
            stub_f32_ssbo: None,
            transform_compute_pipeline: None,
            cloth_verlet_pipeline: None,
            cloth_constraint_lambda_update_pipeline: None,
            cloth_constraint_accumulate_pipeline: None,
            cloth_constraint_apply_pipeline: None,
            cloth_normal_pipeline: None,
            cloth_collide_pipeline: None,
            cloth_bend_update_pipeline: None,
            cloth_bend_accumulate_pipeline: None,
            cloth_bend_apply_pipeline: None,
            cloth_selfcol_build_pipeline: None,
            cloth_selfcol_resolve_pipeline: None,
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
            graphics_pipeline_cutout: None,
            pipeline_no_cull_cutout: None,
            pipeline_front_cull_cutout: None,
            outline_pipeline: None,
            sampler: None,
            default_texture_view: None,
            post_sampler: None,
            black_texture_view: None,
            post_effects: None,
            background_pipeline: None,
            offscreen_color: None,
            offscreen_color_view: None,
            offscreen_depth: None,
            offscreen_depth_view: None,
            offscreen_framebuffer: None,
            current_extent: [1920, 1080],
            current_color_space: frame_input::RenderColorSpace::Srgb,
            current_sample_count: 1,
            msaa_color: None,
            msaa_color_view: None,
            camera_ring: None,
            bg_uniform_ring: None,
            cb_cache: HashMap::new(),
            sdf_slots: HashMap::new(),
            sdf_pipeline: None,
            sdf_planned: Vec::new(),

            staging_buffers: [None, None],
            readback_buffers: [None, None],
            readback_cpu_pool: [None, None],
            readback_slot: 0,
            pending_readback: None,
            prof_ts_pool: None,
            depth_readback_enabled: false,
            depth_readback_buffer: None,
            skinning_cache: Vec::new(),
        }
    }

    // Readback helpers (`harvest_pending`, `harvest_pending_readback`,
    // `ensure_readback_buffers`, `ensure_depth_readback_buffer`) live in
    // `src/renderer/readback.rs` as part of the #12 module split. The
    // render() function enqueues `PendingReadbackState` directly from the
    // prepared frame plan; only the off-render-thread harvest + allocator
    // helpers were moved.

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

        // The scene render pass always uses the HDR working format; the
        // user's output colour space only affects the post-effect composite
        // pass built below. Startup always builds the single-sample path;
        // the first frame's `apply_output_format` upgrades to the user's
        // requested MSAA level (clamped to device support) if it differs.
        let sample_count = self.current_sample_count;
        let render_pass = Self::build_render_pass(device.clone(), sample_count);

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
            sample_count,
        )
        .expect("failed to create graphics pipeline");

        let gfx_pipeline_blend = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Back,
            frame_input::RenderAlphaMode::Blend,
            sample_count,
        )
        .expect("failed to create blend graphics pipeline");

        let no_cull_pipeline = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::None,
            frame_input::RenderAlphaMode::Opaque,
            sample_count,
        )
        .expect("failed to create no-cull pipeline");

        let no_cull_pipeline_blend = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::None,
            frame_input::RenderAlphaMode::Blend,
            sample_count,
        )
        .expect("failed to create blend no-cull pipeline");

        let front_cull_pipeline = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Front,
            frame_input::RenderAlphaMode::Opaque,
            sample_count,
        )
        .expect("failed to create front-cull pipeline");

        let front_cull_pipeline_blend = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Front,
            frame_input::RenderAlphaMode::Blend,
            sample_count,
        )
        .expect("failed to create blend front-cull pipeline");

        let cutout_pipeline = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Back,
            frame_input::RenderAlphaMode::Cutout,
            sample_count,
        )
        .expect("failed to create cutout pipeline");

        let no_cull_pipeline_cutout = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::None,
            frame_input::RenderAlphaMode::Cutout,
            sample_count,
        )
        .expect("failed to create no-cull cutout pipeline");

        let front_cull_pipeline_cutout = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Front,
            frame_input::RenderAlphaMode::Cutout,
            sample_count,
        )
        .expect("failed to create front-cull cutout pipeline");

        let background_pipeline = background::create_background_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            sample_count,
        )
        .expect("failed to create background pipeline");

        let outline_pipeline = pipeline::create_outline_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport,
            sample_count,
        )
        .expect("failed to create outline pipeline");

        let targets = Self::create_offscreen_targets(
            memory_allocator.clone(),
            render_pass.clone(),
            extent,
            sample_count,
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

        let post_sampler = Self::create_post_sampler(device.clone());
        let black_texture_view = Self::create_transparent_black_texture(
            device.clone(),
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            queue.clone(),
        );
        let post_effects = Self::build_post_effect_resources(
            device.clone(),
            memory_allocator.clone(),
            descriptor_set_allocator.clone(),
            &self.current_color_space,
            extent,
            targets.color_view.clone(),
            post_sampler.clone(),
            black_texture_view.clone(),
        )
        .expect("failed to create post-effect resources");

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
        self.graphics_pipeline_cutout = Some(cutout_pipeline);
        self.pipeline_no_cull_cutout = Some(no_cull_pipeline_cutout);
        self.pipeline_front_cull_cutout = Some(front_cull_pipeline_cutout);
        self.outline_pipeline = Some(outline_pipeline.clone());
        self.background_pipeline = Some(background_pipeline);
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
        let cloth_collide_pipeline = pipeline::create_cloth_collide_compute_pipeline(
            self.device.as_ref().expect("device set above").clone(),
        )
        .expect("failed to create cloth collide compute pipeline");
        self.cloth_collide_pipeline = Some(cloth_collide_pipeline);
        let cloth_bend_update_pipeline = pipeline::create_cloth_bend_update_compute_pipeline(
            self.device.as_ref().expect("device set above").clone(),
        )
        .expect("failed to create cloth bend update compute pipeline");
        self.cloth_bend_update_pipeline = Some(cloth_bend_update_pipeline);
        let cloth_bend_accumulate_pipeline =
            pipeline::create_cloth_bend_accumulate_compute_pipeline(
                self.device.as_ref().expect("device set above").clone(),
            )
            .expect("failed to create cloth bend accumulate compute pipeline");
        self.cloth_bend_accumulate_pipeline = Some(cloth_bend_accumulate_pipeline);
        let cloth_bend_apply_pipeline = pipeline::create_cloth_bend_apply_compute_pipeline(
            self.device.as_ref().expect("device set above").clone(),
        )
        .expect("failed to create cloth bend apply compute pipeline");
        self.cloth_bend_apply_pipeline = Some(cloth_bend_apply_pipeline);
        let cloth_selfcol_build_pipeline =
            pipeline::create_cloth_selfcol_build_compute_pipeline(
                self.device.as_ref().expect("device set above").clone(),
            )
            .expect("failed to create cloth selfcol build pipeline");
        self.cloth_selfcol_build_pipeline = Some(cloth_selfcol_build_pipeline);
        let cloth_selfcol_resolve_pipeline =
            pipeline::create_cloth_selfcol_resolve_compute_pipeline(
                self.device.as_ref().expect("device set above").clone(),
            )
            .expect("failed to create cloth selfcol resolve pipeline");
        self.cloth_selfcol_resolve_pipeline = Some(cloth_selfcol_resolve_pipeline);
        // Body-SDF splat pipeline. Created upfront like the cloth
        // compute pipelines; the per-instance field slots allocate
        // lazily on first frame that requests one.
        let body_sdf_splat_pipeline = pipeline::create_body_sdf_splat_compute_pipeline(
            self.device.as_ref().expect("device set above").clone(),
        )
        .expect("failed to create body SDF splat compute pipeline");
        self.sdf_pipeline = Some(body_sdf_splat_pipeline);
        self.sampler = Some(sampler);
        self.default_texture_view = Some(default_texture_view);
        self.post_sampler = Some(post_sampler);
        self.black_texture_view = Some(black_texture_view);
        self.post_effects = Some(post_effects);
        self.offscreen_color = Some(targets.color);
        self.offscreen_color_view = Some(targets.color_view);
        self.offscreen_depth = Some(targets.depth);
        self.offscreen_depth_view = Some(targets.depth_view);
        self.offscreen_framebuffer = Some(targets.framebuffer);
        self.msaa_color = targets.msaa_color;
        self.msaa_color_view = targets.msaa_color_view;
        self.current_sample_count = sample_count;

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

        self.bg_uniform_ring = Some(background::BgUniformRing::new(
            &memory_allocator,
            &descriptor_set_allocator,
            &self.background_pipeline.as_ref().expect("set above"),
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

        self.rebuild_pipelines_and_targets(render_pass, new_extent, self.current_sample_count)
            .map_err(|e| format!("resize: {e}"))?;

        // Invalidate pre-allocated readback buffers (extent changed), the
        // CPU pixel pool (sized to the old extent), and any cached command
        // buffers (they reference the old pipelines / targets / buffers).
        self.staging_buffers = [None, None];
        self.readback_buffers = [None, None];
        self.readback_cpu_pool = [None, None];
        self.cb_cache.clear();

        info!("renderer: resized to {}x{}", new_extent[0], new_extent[1]);
        Ok(())
    }

    /// Clamp a requested MSAA sample count down to what the physical device
    /// can actually use as a framebuffer attachment. Intersects the colour
    /// and depth framebuffer sample-count limits (both attachments are
    /// multisampled together) and returns the highest supported power of two
    /// that does not exceed `requested`. `requested <= 1` always returns 1.
    ///
    /// Integrated GPUs are additionally capped at 4×: they share system
    /// memory and have far less bandwidth than discrete cards, so 8× MSAA at
    /// output resolution risks GPU timeouts (TDR) / stutter on them. The GUI
    /// still offers 8×; it just silently degrades to 4× on an iGPU.
    fn clamp_sample_count(device: &Arc<Device>, requested: u32) -> u32 {
        if requested <= 1 {
            return 1;
        }
        let props = device.physical_device().properties();
        let requested = if props.device_type == PhysicalDeviceType::IntegratedGpu {
            requested.min(4)
        } else {
            requested
        };
        let supported =
            props.framebuffer_color_sample_counts & props.framebuffer_depth_sample_counts;
        let mut best = 1u32;
        for (n, flag) in [
            (2u32, SampleCounts::SAMPLE_2),
            (4, SampleCounts::SAMPLE_4),
            (8, SampleCounts::SAMPLE_8),
        ] {
            if n <= requested && supported.intersects(flag) {
                best = n;
            }
        }
        best
    }

    /// Reconcile the renderer's output colour space and MSAA level with the
    /// user's request. Rebuilds the render pass (the colour-attachment format
    /// and the resolve-attachment structure both depend on these), then
    /// routes through `rebuild_pipelines_and_targets` so all dependent state
    /// (pipelines, offscreen targets, camera ring, thumb cache) ends up
    /// compatible. No-op when neither the colour space nor the (clamped)
    /// sample count changed.
    pub fn apply_output_format(
        &mut self,
        target_cs: frame_input::RenderColorSpace,
        requested_samples: u32,
    ) -> Result<(), String> {
        if !self.initialized {
            return Ok(());
        }

        let device = self.device.as_ref().ok_or("renderer: no device")?.clone();
        let target_samples = Self::clamp_sample_count(&device, requested_samples);
        if target_cs == self.current_color_space && target_samples == self.current_sample_count {
            return Ok(());
        }

        let prev_cs = self.current_color_space.clone();
        let prev_samples = self.current_sample_count;

        let new_render_pass = Self::build_render_pass(device, target_samples);
        self.render_pass = Some(new_render_pass.clone());

        let extent = self.current_extent;
        // The colour space must be set *before* the rebuild: the post-effect
        // composite pass reads `self.current_color_space` to pick the final
        // image format. (The scene render pass itself is colour-space
        // independent — always HDR.)
        self.current_color_space = target_cs.clone();
        // `rebuild_pipelines_and_targets` sets `current_extent` and
        // `current_sample_count`.
        self.rebuild_pipelines_and_targets(new_render_pass, extent, target_samples)
            .map_err(|e| format!("apply_output_format: {e}"))?;

        info!(
            "renderer: output format colour {:?} -> {:?} (composite target {:?}), MSAA {}x -> {}x",
            prev_cs,
            target_cs,
            Self::color_attachment_format(&target_cs),
            prev_samples,
            target_samples,
        );
        Ok(())
    }

    pub fn clear_caches(&mut self) {
        self.texture_cache.clear();
        self.transform_cache.clear();
        // Cached command buffers reference the old per-primitive /
        // skinning resources — they must not survive an avatar swap.
        self.cb_cache.clear();
        // SDF slots pin the old instances' field buffers and descriptor
        // sets; drop them with everything else.
        self.sdf_slots.clear();
        self.sdf_planned.clear();
        // `stub_storage_ssbo` is intentionally retained — it is a tiny
        // 1-element zero buffer with no per-avatar state, and reallocating
        // would burn one needless VRAM round-trip on every avatar swap.
        self.material_uploader.clear_cache();
        self.skinning_cache.clear();
    }

    pub(crate) fn release_export_lease(&mut self, lease_id: u64) {
        self.output_exporter.release_token(lease_id);
    }

    /// Enable/disable per-frame depth-aspect CPU readback. When on, `render`
    /// (1× only) copies the offscreen depth image to a host buffer and
    /// surfaces it as [`RenderResult::depth_ndc`] — pipelined like the colour
    /// readback, so it is harvested on the *next* `render` call. Off by
    /// default; used by the metric-depth benches (`validate_gt`) to
    /// reconstruct a `MetricDepthFrame` from a self-consistency render. No
    /// effect under MSAA (the depth attachment is multisampled + `DontCare`).
    pub fn set_depth_readback(&mut self, enabled: bool) {
        self.depth_readback_enabled = enabled;
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
                depth_ndc: None,
                cloth_readback: Vec::new(),
                vbo_audit: Vec::new(),
                sdf_fields: Vec::new(),
            });
        }

        // Adopt the user's selected output colour space + MSAA level before
        // any per-frame GPU work. The path is a no-op when the renderer
        // already matches, so the cost only applies on actual user toggles.
        self.apply_output_format(
            input.output_request.color_space.clone(),
            input.output_request.msaa.sample_count(),
        )
        .map_err(|e| format!("render: output format switch failed: {e}"))?;

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

        // ── Phase 2: Prepare the frame ───────────────────────────────────
        // Every per-frame CPU write (camera, background uniform, skinning,
        // control UBOs, morph weights, cloth controls + pins, material
        // uniforms) plus the dispatch structure / shape key. See
        // `frame_plan.rs`.
        let plan = self.prepare_frame(input)?;

        // ── Phase 3: Get or build the frame command buffer ──────────────
        let command_buffer = self.get_or_build_frame_cb(&plan)?;

        // GPU cloth readback: harvest (above) already waited the
        // previous frame's fence, so the previous frame's cloth compute
        // is complete and the host-visible SSBOs read coherently. This
        // MUST run before the current frame's submission — after
        // `then_execute` the new dispatch holds the SSBOs in flight
        // and vulkano's usage tracking refuses the CPU map ("resource
        // already in use"), which the live soak observed failing every
        // frame. Attached to whichever result this call returns.
        let cloth_readback = self.read_cloth_positions();
        // Same one-frame-stale discipline as the cloth readback above
        // (previous frame's fence already waited). Only the instances
        // planned THIS frame map their fields; rows for gated-off
        // instances would ship stale geometry.
        let sdf_planned = std::mem::take(&mut self.sdf_planned);
        let sdf_fields = sdf_field::read_sdf_fields(&self.sdf_slots, &sdf_planned);
        // Same overwrite discipline as `cloth_readback` above.
        let vbo_audit = self.read_vbo_audit();

        crate::tracking::stagelog::mark(self.frame_counter, "render_submit");
        let device = self.device.as_ref().ok_or("renderer: no device")?.clone();
        let queue = self.queue.as_ref().ok_or("renderer: no queue")?.clone();
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
                "GPU_RUNTIME frame={} transform_resources={} weight_writes={} \
                 cloth_writes={} cloth_creations={} cache_slots={} readback_bytes={} \
                 cb_hits={} cb_misses={} pix_reuse={} pix_alloc={}",
                self.frame_counter,
                counters.morph_gpu_resource_creations,
                counters.morph_weight_writes,
                counters.cloth_vbo_writes,
                counters.cloth_cache_creations,
                self.transform_cache.len(),
                (extent[0] as u64) * (extent[1] as u64) * 4,
                counters.cb_cache_hits,
                counters.cb_cache_misses,
                counters.pixel_pool_reuses,
                counters.pixel_pool_allocs,
            );
            if let Some((pool, period_ns)) = &self.prof_ts_pool {
                // Read ONLY the slots the CB writes (0..9): WAIT blocks on
                // any unwritten query and would stall the render thread
                // forever (observed as a driver device-lost on first use).
                let mut ticks = [0u64; 9];
                match pool.get_results(0..9, &mut ticks, vulkano::query::QueryResultFlags::WAIT) {
                    Ok(_) => {
                        // Timestamps carry the queue family's valid bits in
                        // the low word; mask to 32 bits (wraps ≈ 100 s at
                        // 24 ns — far apart relative to one frame).
                        const MASK: u64 = 0xFFFF_FFFF;
                        let ms = |a: u64, b: u64| {
                            (b.wrapping_sub(a) & MASK) as f32 * period_ns / 1_000_000.0
                        };
                        let total = ms(ticks[0], ticks[4]);
                        let cloth = ms(ticks[5], ticks[6]);
                        let sdf = ms(ticks[7], ticks[8]);
                        println!(
                            "RENDER_PROF compute_prepass={:.2}ms scene={:.2}ms post={:.2}ms readback_copy={:.2}ms total={:.2}ms | clothsim={:.2}ms sdf_splat={:.2}ms rest={:.2}ms (iters=env)",
                            ms(ticks[0], ticks[1]),
                            ms(ticks[1], ticks[2]),
                            ms(ticks[2], ticks[3]),
                            ms(ticks[3], ticks[4]),
                            total,
                            cloth,
                            sdf,
                            (total - cloth - sdf).max(0.0),
                        );
                    }
                    Err(e) => println!("RENDER_PROF get_results failed: {e:?}"),
                }
            }
        }

        let stats = RenderStats {
            instance_count: input.instances.len() as u32,
            mesh_count: total_meshes,
            material_count: total_materials,
            cloth_instances,
            export_pool: self.output_exporter.export_image_pool().stats(),
        };

        self.pending_readback = Some(PendingReadbackState {
            wait_fn: Box::new(move || gpu_wait::wait_fence_bounded(fence_future, "frame_readback")),
            readback_buffer: plan.readback_buffer.clone(),
            pool_slot: plan.readback_slot,
            depth_buffer: plan.depth_buffer.clone(),
            extent,
            timestamp_nanos,
            stats: stats.clone(),
            color_space: input.output_request.color_space.clone(),
        });

        match harvested {
            Some(mut h) => {
                h.cloth_readback = cloth_readback;
                h.vbo_audit = vbo_audit;
                h.sdf_fields = sdf_fields;
                Ok(h)
            }
            None => Ok(RenderResult {
                extent,
                timestamp_nanos: 0,
                has_alpha: true,
                stats,
                exported_frame: None,
                depth_ndc: None,
                cloth_readback,
                vbo_audit,
                sdf_fields,
            }),
        }
    }

    /// R2 diagnostic knob (`VULVATAR_VBO_AUDIT=1`): allocate host-
    /// readable staging for anchor-bearing primitives and copy their
    /// final VBOs into it every frame. Process-static on purpose — the
    /// audit copies are part of the cached command buffer, so flipping
    /// the flag mid-run would need a `cb_cache` invalidation.
    fn vbo_audit_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("VULVATAR_VBO_AUDIT").is_ok_and(|v| v == "1"))
    }

    /// A/B knob for the command-buffer cache (`VULVATAR_CB_CACHE=0`
    /// re-records + rebuilds every frame). Evaluated once per process.
    fn cb_cache_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("VULVATAR_CB_CACHE").is_ok_and(|v| v != "0"))
    }

    /// Profiling knob (`VULVATAR_RENDER_PROF=1`): GPU timestamp
    /// breakdown of the cached frame CB (compute prepass / scene /
    /// post / readback copies), logged as `RENDER_PROF` every 60
    /// frames. On-demand instrumentation per docs/profiling.md — not
    /// part of the committed feature set.
    fn render_prof_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("VULVATAR_RENDER_PROF").is_ok_and(|v| v == "1"))
    }

    /// Read the previous frame's final-VBO audit rows from the
    /// host-visible staging buffers. MUST be called after the previous
    /// frame's fence has been waited (same discipline as
    /// `read_cloth_positions`, which the call site honours) and before
    /// the current frame's submission. Empty unless
    /// `VULVATAR_VBO_AUDIT=1`.
    pub(crate) fn read_vbo_audit(&mut self) -> Vec<VboAuditEntry> {
        if !Self::vbo_audit_enabled() {
            return Vec::new();
        }
        let mut out = Vec::new();
        for ((mesh_id, primitive_id), slot) in self.transform_cache.iter() {
            let Some(staging) = slot.audit_staging.as_ref() else {
                continue;
            };
            let guard = match staging.read() {
                Ok(g) => g,
                Err(e) => {
                    log::warn!("render: vbo audit read failed for {:?}: {e}", primitive_id);
                    continue;
                }
            };
            let (nan_count, max_corr, p95, max_pos_len) = vbo_audit_stats(&guard);
            out.push(VboAuditEntry {
                mesh_id: *mesh_id,
                primitive_id: *primitive_id,
                instance_id: slot.cloth_owner_instance,
                vertex_count: guard.len(),
                nan_count,
                max_correction_m: max_corr,
                p95_correction_m: p95,
                max_pos_len_m: max_pos_len,
            });
        }
        out
    }

    /// Fetch the cached command buffer for `plan.shape_key`, or record +
    /// build + store one.
    fn get_or_build_frame_cb(
        &mut self,
        plan: &frame_plan::FramePlan,
    ) -> Result<Arc<PrimaryAutoCommandBuffer>, String> {
        if Self::cb_cache_enabled() {
            if let Some(entry) = self.cb_cache.get_mut(&plan.shape_key) {
                entry.last_used = self.frame_counter;
                self.gpu_runtime_counters.cb_cache_hits += 1;
                return Ok(entry.cb.clone());
            }
        }
        self.gpu_runtime_counters.cb_cache_misses += 1;
        let cb = self.build_frame_cb(plan)?;
        if Self::cb_cache_enabled() {
            if self.cb_cache.len() >= CB_CACHE_CAP {
                let evict = self
                    .cb_cache
                    .iter()
                    .min_by_key(|(_, e)| e.last_used)
                    .map(|(k, _)| *k);
                if let Some(k) = evict {
                    self.cb_cache.remove(&k);
                }
            }
            self.cb_cache.insert(
                plan.shape_key,
                CachedFrameCb {
                    cb: cb.clone(),
                    last_used: self.frame_counter,
                },
            );
        }
        Ok(cb)
    }

    /// Record + build the frame command buffer from a prepared plan.
    /// Mirrors the historical per-frame recording; the only intentional
    /// difference is `CommandBufferUsage::SimultaneousUse`, which is what
    /// makes the built buffer resubmittable from the cache. (Only one
    /// frame is ever in flight — `render` harvests the previous fence
    /// before submitting — so even simultaneous use is conservative.)
    fn build_frame_cb(
        &mut self,
        plan: &frame_plan::FramePlan,
    ) -> Result<Arc<PrimaryAutoCommandBuffer>, String> {
        let queue = self.queue.as_ref().ok_or("renderer: no queue")?.clone();
        let cb_allocator = self
            .command_buffer_allocator
            .as_ref()
            .ok_or("renderer: no command buffer allocator")?
            .clone();
        let framebuffer = self
            .offscreen_framebuffer
            .as_ref()
            .ok_or("renderer: no framebuffer")?
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
        let cloth_verlet_pipeline = self
            .cloth_verlet_pipeline
            .as_ref()
            .ok_or("renderer: no cloth verlet pipeline")?
            .clone();
        let cloth_lambda_pipeline = self
            .cloth_constraint_lambda_update_pipeline
            .as_ref()
            .ok_or("renderer: no cloth lambda pipeline")?
            .clone();
        let cloth_accumulate_pipeline = self
            .cloth_constraint_accumulate_pipeline
            .as_ref()
            .ok_or("renderer: no cloth accumulate pipeline")?
            .clone();
        let cloth_apply_pipeline = self
            .cloth_constraint_apply_pipeline
            .as_ref()
            .ok_or("renderer: no cloth apply pipeline")?
            .clone();
        let cloth_normal_pipeline = self
            .cloth_normal_pipeline
            .as_ref()
            .ok_or("renderer: no cloth normal pipeline")?
            .clone();
        let cloth_collide_pipeline = self
            .cloth_collide_pipeline
            .as_ref()
            .ok_or("renderer: no cloth collide pipeline")?
            .clone();
        let cloth_bend_update_pipeline = self
            .cloth_bend_update_pipeline
            .as_ref()
            .ok_or("renderer: no cloth bend update pipeline")?
            .clone();
        let cloth_bend_accumulate_pipeline = self
            .cloth_bend_accumulate_pipeline
            .as_ref()
            .ok_or("renderer: no cloth bend accumulate pipeline")?
            .clone();
        let cloth_bend_apply_pipeline = self
            .cloth_bend_apply_pipeline
            .as_ref()
            .ok_or("renderer: no cloth bend apply pipeline")?
            .clone();
        let cloth_selfcol_build_pipeline = self
            .cloth_selfcol_build_pipeline
            .as_ref()
            .ok_or("renderer: no cloth selfcol build pipeline")?
            .clone();
        let cloth_selfcol_resolve_pipeline = self
            .cloth_selfcol_resolve_pipeline
            .as_ref()
            .ok_or("renderer: no cloth selfcol resolve pipeline")?
            .clone();
        let body_sdf_splat_pipeline = self
            .sdf_pipeline
            .as_ref()
            .ok_or("renderer: no body SDF splat pipeline")?
            .clone();

        let mut builder = AutoCommandBufferBuilder::primary(
            cb_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::SimultaneousUse,
        )
        .map_err(|e| format!("render: failed to create command buffer: {e}"))?;

        // ── GPU timestamp breakdown (profiling only, RENDER_PROF) ───────
        let prof_pool = if Self::render_prof_enabled() {
            if self.prof_ts_pool.is_none() {
                let device = queue.device().clone();
                let mut info =
                    vulkano::query::QueryPoolCreateInfo::query_type(vulkano::query::QueryType::Timestamp);
                info.query_count = 9;
                let pool = vulkano::query::QueryPool::new(device.clone(), info)
                    .map_err(|e| format!("render: prof query pool failed: {e}"))?;
                let period = device.physical_device().properties().timestamp_period;
                self.prof_ts_pool = Some((pool, period));
            }
            Some(self.prof_ts_pool.as_ref().unwrap().clone())
        } else {
            None
        };
        if let Some((pool, _)) = &prof_pool {
            unsafe {
                builder
                    .reset_query_pool(pool.clone(), 0..9)
                    .map_err(|e| format!("render: prof reset failed: {e:?}"))?;
                builder
                    .write_timestamp(
                        pool.clone(),
                        0,
                        vulkano::sync::PipelineStage::BottomOfPipe,
                    )
                    .map_err(|e| format!("render: prof ts0 failed: {e:?}"))?;
            }
        }

        // ── Compute prepass: fuse skinning + morph + cloth per primitive
        Self::record_compute_prepass_planned(
            &mut builder,
            &transform_pipeline,
            &cloth_verlet_pipeline,
            &cloth_lambda_pipeline,
            &cloth_accumulate_pipeline,
            &cloth_apply_pipeline,
            &cloth_normal_pipeline,
            &cloth_collide_pipeline,
            &cloth_bend_update_pipeline,
            &cloth_bend_accumulate_pipeline,
            &cloth_bend_apply_pipeline,
            &cloth_selfcol_build_pipeline,
            &cloth_selfcol_resolve_pipeline,
            &body_sdf_splat_pipeline,
            &plan.instances,
            &plan.containment_copies,
            &plan.audit_copies,
        )?;
        if let Some((pool, _)) = &prof_pool {
            unsafe {
                builder
                    .write_timestamp(pool.clone(), 1, vulkano::sync::PipelineStage::BottomOfPipe)
                    .map_err(|e| format!("render: prof ts1 failed: {e:?}"))?;
            }
        }

        // ── Scene pass: forward draws + outlines ────────────────────────
        self.record_scene_pass(
            &mut builder,
            plan,
            &framebuffer,
            &gfx_pipeline,
            &outline_pipeline,
        )?;
        if let Some((pool, _)) = &prof_pool {
            unsafe {
                builder
                    .write_timestamp(pool.clone(), 2, vulkano::sync::PipelineStage::BottomOfPipe)
                    .map_err(|e| format!("render: prof ts2 failed: {e:?}"))?;
            }
        }

        // ── Post effects: bloom chain + composite/encode ────────────────
        // The composite always runs — it is the HDR→8-bit encode stage that
        // produces the readback / export image. The bloom chain only runs
        // when enabled; otherwise the composite samples the 1x1 transparent
        // black fallback with zero intensity (a passthrough).
        {
            let post = self
                .post_effects
                .as_ref()
                .ok_or("renderer: no post-effect resources")?;
            if plan.use_bloom {
                Self::record_bloom_chain(&mut builder, post, &plan.bloom)?;
            }
            Self::record_composite(&mut builder, post, plan.composite_intensity, plan.use_bloom)?;
        }
        if let Some((pool, _)) = &prof_pool {
            unsafe {
                builder
                    .write_timestamp(pool.clone(), 3, vulkano::sync::PipelineStage::BottomOfPipe)
                    .map_err(|e| format!("render: prof ts3 failed: {e:?}"))?;
            }
        }

        // ── Readback: two-stage copy to avoid slow Intel DMA path ───────
        builder
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                plan.final_color_image.clone(),
                plan.staging_buffer.clone(),
            ))
            .map_err(|e| format!("render: copy_image_to_buffer failed: {e}"))?;

        builder
            .copy_buffer(vulkano::command_buffer::CopyBufferInfo::buffers(
                plan.staging_buffer.clone(),
                plan.readback_buffer.clone(),
            ))
            .map_err(|e| format!("render: copy_buffer staging→readback failed: {e}"))?;
        if let Some((pool, _)) = &prof_pool {
            unsafe {
                builder
                    .write_timestamp(pool.clone(), 4, vulkano::sync::PipelineStage::BottomOfPipe)
                    .map_err(|e| format!("render: prof ts4 failed: {e:?}"))?;
            }
        }

        // Optional depth-aspect readback (metric-depth benches only). 1× only
        // — the MSAA depth attachment is multisampled + `DontCare`. The depth
        // image carries `TRANSFER_SRC` and is `Store`d (see `pipeline_targets`);
        // we copy only the DEPTH aspect of the combined D32S8 format to a
        // host-visible buffer harvested next frame alongside the colour.
        if let (Some(depth_image), Some(depth_buffer)) = (&plan.depth_image, &plan.depth_buffer) {
            let ext = self.current_extent;
            let region = vulkano::command_buffer::BufferImageCopy {
                image_subresource: vulkano::image::ImageSubresourceLayers {
                    aspects: vulkano::image::ImageAspects::DEPTH,
                    mip_level: 0,
                    array_layers: 0..1,
                },
                image_extent: [ext[0], ext[1], 1],
                ..Default::default()
            };
            builder
                .copy_image_to_buffer(CopyImageToBufferInfo {
                    regions: [region].into_iter().collect(),
                    ..CopyImageToBufferInfo::image_buffer(depth_image.clone(), depth_buffer.clone())
                })
                .map_err(|e| format!("render: depth copy_image_to_buffer failed: {e}"))?;
        }

        builder
            .build()
            .map_err(|e| format!("render: failed to build command buffer: {e}"))
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
            _ => return Err("render_thumbnail: exported pixel data isn't CPU readback".to_string()),
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
    /// Global avatar opacity multiplier (1.0 = opaque). Drives the
    /// fade-out-when-no-person-detected feature; reuses the former
    /// `_pad2` slot so the std140 layout is unchanged.
    fade_opacity: f32,
}

/// Project a world-space point through the **row-major** camera view +
/// projection matrices to a `[0,1]²` screen UV (matching the fullscreen
/// triangle's `frag_uv` orientation — the Vulkan Y flip is baked into the
/// projection matrix). Returns `None` when the point is at or behind the
/// near plane. Bone `global_transforms` are column-major, but only their
/// translation column is fed here, so no conversion is needed.
fn project_world_to_uv(
    view: &crate::asset::Mat4,
    proj: &crate::asset::Mat4,
    p: [f32; 3],
) -> Option<[f32; 2]> {
    fn mul_row_major(m: &crate::asset::Mat4, v: [f32; 4]) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for (row, out_v) in out.iter_mut().enumerate() {
            for (col, v_v) in v.iter().enumerate() {
                *out_v += m[row][col] * v_v;
            }
        }
        out
    }
    let clip = mul_row_major(proj, mul_row_major(view, [p[0], p[1], p[2], 1.0]));
    if clip[3] <= 1e-6 {
        return None;
    }
    Some([clip[0] / clip[3] * 0.5 + 0.5, clip[1] / clip[3] * 0.5 + 0.5])
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
            fade_opacity: 1.0,
        };

        let mut buffers = Vec::with_capacity(FRAME_LAG);
        let mut main_sets = Vec::with_capacity(FRAME_LAG);
        let mut outline_sets = Vec::with_capacity(FRAME_LAG);

        for _ in 0..FRAME_LAG {
            let buffer =
                gpu_alloc::host_ubo(&memory_allocator, initial, "camera ring buffer alloc")
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

/// Pure stats core of [`VulkanRenderer::read_vbo_audit`]: scan a final
/// VBO and return `(nan_count, max_correction, p95_correction,
/// max_pos_len)`. Position `.w` carries the render-side correction
/// total written by `transform_cs`; NaN positions are counted and
/// skipped.
pub(crate) fn vbo_audit_stats(vertices: &[GpuVertex]) -> (usize, f32, f32, f32) {
    let mut nan_count = 0usize;
    let mut max_corr = 0.0f32;
    let mut max_pos_len = 0.0f32;
    let mut corrs: Vec<f32> = Vec::with_capacity(vertices.len());
    for v in vertices {
        let p = v.position;
        if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
            nan_count += 1;
            continue;
        }
        let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if len.is_finite() {
            max_pos_len = max_pos_len.max(len);
        }
        // Non-finite telemetry means a corrupt vertex state too.
        let corr = if p[3].is_finite() {
            p[3]
        } else {
            nan_count += 1;
            0.0
        };
        corrs.push(corr);
        max_corr = max_corr.max(corr);
    }
    corrs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95 = corrs
        .get(((corrs.len() as f32) * 0.95) as usize)
        .copied()
        .unwrap_or(0.0);
    (nan_count, max_corr, p95, max_pos_len)
}

#[cfg(test)]
mod vbo_audit_tests {
    use super::{vbo_audit_stats, GpuVertex};

    fn v(x: f32, y: f32, z: f32, corr: f32) -> GpuVertex {
        GpuVertex {
            position: [x, y, z, corr],
            normal: [0.0, 1.0, 0.0, 0.0],
            uv: [0.0; 2],
            _pad: [0; 2],
        }
    }

    /// Healthy frame: finite positions, millimetre-scale corrections.
    #[test]
    fn audit_stats_healthy_frame() {
        let verts = vec![v(1.0, 0.0, 0.0, 0.004), v(0.0, 2.0, 0.0, 0.012)];
        let (nan, max, p95, max_len) = vbo_audit_stats(&verts);
        assert_eq!(nan, 0);
        assert!((max - 0.012).abs() < 1e-6);
        assert!((p95 - 0.012).abs() < 1e-6);
        assert!((max_len - 2.0).abs() < 1e-5);
    }

    /// NaN positions are counted and excluded from radius/correction
    /// stats; NaN telemetry counts as a NaN vertex too.
    #[test]
    fn audit_stats_counts_nan_and_saturating_corrections() {
        let verts = vec![
            v(0.0, 0.0, 0.0, 0.3),   // clamp-saturated correction
            v(f32::NAN, 0.0, 1.0, 0.0),
            v(3.0, 4.0, 0.0, f32::NAN), // NaN telemetry counts as NaN vertex
        ];
        let (nan, max, _p95, max_len) = vbo_audit_stats(&verts);
        assert_eq!(nan, 2, "NaN position and NaN telemetry both count");
        assert!((max - 0.3).abs() < 1e-6, "max correction sees the saturated push");
        assert!((max_len - 5.0).abs() < 1e-5, "radius ignores the NaN vertex");
    }
}
