//! Post-effect layer: full-screen passes that run after the scene render
//! pass and before readback / export.
//!
//! The scene renders into an HDR target (`R16G16B16A16_SFLOAT`); this module
//! owns everything downstream of it:
//!
//! - **Bloom downsample chain** (CoD: Advanced Warfare style): the HDR scene
//!   is downsampled into a half-resolution-per-level chain of independent
//!   16F images. The first downsample applies a soft-knee luminance
//!   threshold plus a per-group Karis average (firefly suppression).
//! - **Bloom upsample chain**: each level is tent-filtered and additively
//!   blended (`One`/`One`) back into the level above, accumulating the blur.
//! - **Composite/encode pass** (always runs, even with bloom disabled):
//!   samples the HDR scene + bloom level 0, adds them, and writes the final
//!   8-bit image whose format follows the user's colour-space selection —
//!   the same `R8G8B8A8_SRGB` / `R8G8B8A8_UNORM` contract the readback and
//!   export paths have always consumed. Shaders write linear values; the
//!   attachment format applies the transfer curve (gamma policy, see
//!   `pipeline.rs`).
//!
//! The chain uses independent images per level (not mips of one image) so
//! every dependency is a plain "pass A stores → pass B samples" transition
//! that vulkano's auto-sync handles across render pass boundaries.
//!
//! Alpha: the scene framebuffer content is effectively premultiplied over a
//! transparent background, so the chain blurs all four channels together and
//! the composite adds `intensity * bloom` into RGB *and* alpha — the glow
//! halo carries fractional alpha and survives downstream compositing.

use std::sync::Arc;

use vulkano::command_buffer::{
    AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo,
    SubpassContents, SubpassEndInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, BlendFactor, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};

use crate::renderer::frame_input;
use crate::renderer::VulkanRenderer;

/// HDR working format for the scene target and the bloom chain. 16F colour
/// attachment + blend + linear filtering are mandatory Vulkan format
/// features, so no device gating is needed.
pub(super) const HDR_COLOR_FORMAT: Format = Format::R16G16B16A16_SFLOAT;

/// Maximum number of bloom chain levels (each one is half the resolution of
/// the previous). 6 levels from 1080p reach ~17px — wide enough for a soft
/// emissive glow without visibly square artifacts.
const BLOOM_MAX_LEVELS: usize = 6;

/// Stop the chain before a level's smaller dimension would drop below this.
const BLOOM_MIN_DIM: u32 = 8;

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

/// Full-screen triangle vertex shader shared by every post pass. No vertex
/// buffer: `gl_VertexIndex` 0/1/2 expands to a triangle covering the screen.
/// Source and target share the same image-space orientation, so no Y flip.
pub mod post_fullscreen_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
#version 450

layout(location = 0) out vec2 frag_uv;

void main() {
    vec2 uv = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    frag_uv = uv;
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
"
    }
}

/// 13-tap Jimenez downsample (5 overlapping bilinear 4-tap boxes: centre
/// weighted 0.5, corners 0.125 each). On the first pass (`first_pass == 1`)
/// each box is Karis-weighted (`1 / (1 + max3(rgb))`) before combining to
/// suppress fireflies, then a soft-knee threshold scales the whole RGBA
/// contribution (premultiplied content scales coherently).
pub mod bloom_downsample_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D src_tex;

layout(push_constant) uniform DownsamplePush {
    vec2 src_texel;
    float threshold;
    float knee;
    uint first_pass;
} pc;

float max3(vec3 c) { return max(c.r, max(c.g, c.b)); }

vec4 box4(vec2 uv, vec2 o) {
    return 0.25 * (texture(src_tex, uv + vec2(-o.x, -o.y))
                 + texture(src_tex, uv + vec2( o.x, -o.y))
                 + texture(src_tex, uv + vec2(-o.x,  o.y))
                 + texture(src_tex, uv + vec2( o.x,  o.y)));
}

void main() {
    vec2 t = pc.src_texel;
    vec4 c0 = box4(frag_uv, t);
    vec4 c1 = box4(frag_uv + vec2(-t.x, -t.y), t);
    vec4 c2 = box4(frag_uv + vec2( t.x, -t.y), t);
    vec4 c3 = box4(frag_uv + vec2(-t.x,  t.y), t);
    vec4 c4 = box4(frag_uv + vec2( t.x,  t.y), t);

    if (pc.first_pass == 1u) {
        float w0 = 0.500 / (1.0 + max3(c0.rgb));
        float w1 = 0.125 / (1.0 + max3(c1.rgb));
        float w2 = 0.125 / (1.0 + max3(c2.rgb));
        float w3 = 0.125 / (1.0 + max3(c3.rgb));
        float w4 = 0.125 / (1.0 + max3(c4.rgb));
        vec4 c = (c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3 + c4 * w4)
               / (w0 + w1 + w2 + w3 + w4);

        float l = max3(c.rgb);
        float s = clamp(l - pc.threshold + pc.knee, 0.0, 2.0 * pc.knee);
        s = s * s / (4.0 * pc.knee + 1e-4);
        c *= max(s, l - pc.threshold) / max(l, 1e-4);
        out_color = c;
    } else {
        out_color = 0.5 * c0 + 0.125 * (c1 + c2 + c3 + c4);
    }
}
"
    }
}

/// 3x3 tent upsample. The shader only filters the lower-resolution level;
/// accumulation into the target level is done by the pipeline's additive
/// (`One`/`One`) hardware blend.
pub mod bloom_upsample_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D src_tex;

layout(push_constant) uniform UpsamplePush {
    vec2 src_texel;
} pc;

void main() {
    vec2 t = pc.src_texel;
    vec4 sum = texture(src_tex, frag_uv + vec2(-t.x, -t.y))
             + texture(src_tex, frag_uv + vec2( t.x, -t.y))
             + texture(src_tex, frag_uv + vec2(-t.x,  t.y))
             + texture(src_tex, frag_uv + vec2( t.x,  t.y))
             + 2.0 * (texture(src_tex, frag_uv + vec2(0.0, -t.y))
                    + texture(src_tex, frag_uv + vec2(-t.x, 0.0))
                    + texture(src_tex, frag_uv + vec2( t.x, 0.0))
                    + texture(src_tex, frag_uv + vec2(0.0,  t.y)))
             + 4.0 * texture(src_tex, frag_uv);
    out_color = sum / 16.0;
}
"
    }
}

/// Final composite/encode: HDR scene + bloom → 8-bit output. Writes linear
/// values; the attachment format (SRGB vs UNORM) decides whether the GPU
/// applies the transfer curve at store time. Values above 1.0 clamp at
/// store — deliberate, there is no tonemap (the avatar is composited over
/// live camera footage downstream and must not carry a baked-in look).
pub mod post_composite_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D scene_tex;
layout(set = 0, binding = 1) uniform sampler2D bloom_tex;

layout(push_constant) uniform CompositePush {
    float intensity;
} pc;

void main() {
    vec4 scene = texture(scene_tex, frag_uv);
    vec4 bloom = texture(bloom_tex, frag_uv);
    out_color = vec4(scene.rgb + pc.intensity * bloom.rgb,
                     clamp(scene.a + pc.intensity * bloom.a, 0.0, 1.0));
}
"
    }
}

/// Push constant layout for `bloom_downsample_fs` (matches shader).
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct DownsamplePushConstants {
    src_texel: [f32; 2],
    threshold: f32,
    knee: f32,
    first_pass: u32,
}

/// Push constant layout for `bloom_upsample_fs` (matches shader).
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct UpsamplePushConstants {
    src_texel: [f32; 2],
}

/// Push constant layout for `post_composite_fs` (matches shader).
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct CompositePushConstants {
    intensity: f32,
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// One level of the bloom chain. `framebuffer_down` targets it through the
/// DontCare/Store downsample render pass, `framebuffer_up` through the
/// Load/Store additive upsample render pass (unused on the last level, which
/// is never upsampled into).
pub(super) struct BloomLevel {
    pub view: Arc<ImageView>,
    pub framebuffer_down: Arc<Framebuffer>,
    pub framebuffer_up: Arc<Framebuffer>,
    pub extent: [u32; 2],
}

/// Everything the post-effect layer owns. Rebuilt as a unit whenever the
/// offscreen extent or the output colour space changes (the bloom chain and
/// descriptor sets are extent-dependent; the composite pipeline and final
/// image are colour-space-dependent). Bloom *parameters* are push constants
/// and never trigger a rebuild.
pub(super) struct PostEffectResources {
    pipe_down: Arc<GraphicsPipeline>,
    pipe_up: Arc<GraphicsPipeline>,
    pipe_composite: Arc<GraphicsPipeline>,
    bloom_levels: Vec<BloomLevel>,
    /// Final 8-bit output image — the readback / export source. Carries the
    /// external-memory (OPAQUE_WIN32_KMT) allocation when supported so the
    /// MF virtual-camera GPU export path keeps working against it.
    pub final_color: Arc<Image>,
    final_framebuffer: Arc<Framebuffer>,
    scene_extent: [u32; 2],
    /// `ds_down[i]` samples the scene HDR image (i == 0) or level i-1.
    ds_down: Vec<Arc<DescriptorSet>>,
    /// `ds_up[i]` samples level i+1 (valid for i < levels-1).
    ds_up: Vec<Arc<DescriptorSet>>,
    /// Composite inputs with the bloom chain bound (bloom enabled)…
    ds_composite_bloom: Option<Arc<DescriptorSet>>,
    /// …and with the 1x1 transparent-black texture bound (bloom disabled,
    /// or no chain levels because the output is tiny).
    ds_composite_black: Arc<DescriptorSet>,
}

impl PostEffectResources {
    /// Whether the bloom chain can run at the current extent.
    pub(super) fn has_bloom_chain(&self) -> bool {
        !self.bloom_levels.is_empty()
    }
}

impl VulkanRenderer {
    /// ClampToEdge linear sampler shared by every post pass (the material
    /// sampler uses Repeat, which would wrap bloom taps across screen edges).
    pub(super) fn create_post_sampler(device: Arc<Device>) -> Arc<Sampler> {
        Sampler::new(
            device,
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..Default::default()
            },
        )
        .expect("failed to create post-effect sampler")
    }

    /// 1x1 transparent-black texture bound as the composite's bloom input
    /// while bloom is disabled. Device-lifetime, created once at init.
    pub(super) fn create_transparent_black_texture(
        device: Arc<Device>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        cb_allocator: Arc<vulkano::command_buffer::allocator::StandardCommandBufferAllocator>,
        queue: Arc<vulkano::device::Queue>,
    ) -> Arc<ImageView> {
        let pixels: [u8; 4] = [0, 0, 0, 0];
        Self::upload_rgba_texture(device, memory_allocator, cb_allocator, queue, 1, 1, &pixels)
            .expect("failed to create transparent black texture")
    }

    /// Build the full post-effect resource set for the given scene extent
    /// and output colour space.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_post_effect_resources(
        device: Arc<Device>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        ds_allocator: Arc<StandardDescriptorSetAllocator>,
        color_space: &frame_input::RenderColorSpace,
        extent: [u32; 2],
        scene_hdr_view: Arc<ImageView>,
        sampler: Arc<Sampler>,
        black_view: Arc<ImageView>,
    ) -> Result<PostEffectResources, String> {
        // ── Render passes ───────────────────────────────────────────────
        // Load/store ops differ, so the down and up passes are distinct
        // render pass objects even though they are attachment-compatible.
        let rp_down = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: HDR_COLOR_FORMAT,
                    samples: 1,
                    load_op: DontCare,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .map_err(|e| format!("post: failed to create downsample render pass: {e}"))?;

        let rp_up = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: HDR_COLOR_FORMAT,
                    samples: 1,
                    load_op: Load,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .map_err(|e| format!("post: failed to create upsample render pass: {e}"))?;

        let rp_composite = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: Self::color_attachment_format(color_space),
                    samples: 1,
                    load_op: DontCare,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .map_err(|e| format!("post: failed to create composite render pass: {e}"))?;

        // ── Pipelines (dynamic viewport/scissor — extent-independent) ───
        let additive_blend = AttachmentBlend {
            src_color_blend_factor: BlendFactor::One,
            dst_color_blend_factor: BlendFactor::One,
            src_alpha_blend_factor: BlendFactor::One,
            dst_alpha_blend_factor: BlendFactor::One,
            ..Default::default()
        };

        let pipe_down = Self::create_post_pipeline(
            device.clone(),
            rp_down.clone(),
            bloom_downsample_fs::load(device.clone())
                .map_err(|e| format!("post: failed to load downsample shader: {e}"))?,
            None,
        )?;
        let pipe_up = Self::create_post_pipeline(
            device.clone(),
            rp_up.clone(),
            bloom_upsample_fs::load(device.clone())
                .map_err(|e| format!("post: failed to load upsample shader: {e}"))?,
            Some(additive_blend),
        )?;
        let pipe_composite = Self::create_post_pipeline(
            device.clone(),
            rp_composite.clone(),
            post_composite_fs::load(device.clone())
                .map_err(|e| format!("post: failed to load composite shader: {e}"))?,
            None,
        )?;

        // ── Bloom chain images ──────────────────────────────────────────
        let mut bloom_levels: Vec<BloomLevel> = Vec::new();
        let mut level_extent = extent;
        for _ in 0..BLOOM_MAX_LEVELS {
            let next = [(level_extent[0] / 2).max(1), (level_extent[1] / 2).max(1)];
            if next[0].min(next[1]) < BLOOM_MIN_DIM {
                break;
            }
            level_extent = next;

            let image = Image::new(
                memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: HDR_COLOR_FORMAT,
                    extent: [level_extent[0], level_extent[1], 1],
                    usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .map_err(|e| format!("post: failed to create bloom level image: {e}"))?;
            let view = ImageView::new_default(image)
                .map_err(|e| format!("post: failed to create bloom level view: {e}"))?;

            let framebuffer_down = Framebuffer::new(
                rp_down.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view.clone()],
                    ..Default::default()
                },
            )
            .map_err(|e| format!("post: failed to create bloom down framebuffer: {e}"))?;
            let framebuffer_up = Framebuffer::new(
                rp_up.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view.clone()],
                    ..Default::default()
                },
            )
            .map_err(|e| format!("post: failed to create bloom up framebuffer: {e}"))?;

            bloom_levels.push(BloomLevel {
                view,
                framebuffer_down,
                framebuffer_up,
                extent: level_extent,
            });
        }

        // ── Final 8-bit output image ────────────────────────────────────
        // This is the readback / export source, so it inherits the
        // external-memory (OPAQUE_WIN32_KMT) allocation contract the
        // offscreen colour target used to carry: the MF virtual-camera GPU
        // export shares this exact image cross-API.
        let final_format = Self::color_attachment_format(color_space);
        let mut final_create_info = ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: final_format,
            extent: [extent[0], extent[1], 1],
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC | ImageUsage::SAMPLED,
            ..Default::default()
        };
        let enabled_exts = device.enabled_extensions();
        if enabled_exts.khr_external_memory {
            use vulkano::image::ImageFormatInfo;
            use vulkano::memory::{ExternalMemoryHandleType, ExternalMemoryHandleTypes};
            let fmt_info = ImageFormatInfo {
                format: final_format,
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
                final_create_info.external_memory_handle_types =
                    ExternalMemoryHandleTypes::OPAQUE_WIN32_KMT;
                log::info!(
                    "renderer: exportable final color image ({:?}) with OPAQUE_WIN32_KMT",
                    final_format
                );
            } else {
                log::warn!(
                    "renderer: external memory unsupported for {:?} final color target, skipping",
                    final_format
                );
            }
        }
        let final_color = Image::new(
            memory_allocator,
            final_create_info,
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .map_err(|e| format!("post: failed to create final color image: {e}"))?;
        let final_color_view = ImageView::new_default(final_color.clone())
            .map_err(|e| format!("post: failed to create final color view: {e}"))?;
        let final_framebuffer = Framebuffer::new(
            rp_composite,
            FramebufferCreateInfo {
                attachments: vec![final_color_view],
                ..Default::default()
            },
        )
        .map_err(|e| format!("post: failed to create final framebuffer: {e}"))?;

        // ── Descriptor sets (cached until the next rebuild) ─────────────
        let make_single_input = |pipeline: &Arc<GraphicsPipeline>,
                                 input: Arc<ImageView>|
         -> Result<Arc<DescriptorSet>, String> {
            let layout = pipeline
                .layout()
                .set_layouts()
                .first()
                .ok_or("post: pipeline missing set 0 layout")?
                .clone();
            DescriptorSet::new(
                ds_allocator.clone(),
                layout,
                [WriteDescriptorSet::image_view_sampler(
                    0,
                    input,
                    sampler.clone(),
                )],
                [],
            )
            .map_err(|e| format!("post: descriptor set creation failed: {e}"))
        };

        let mut ds_down = Vec::with_capacity(bloom_levels.len());
        for i in 0..bloom_levels.len() {
            let input = if i == 0 {
                scene_hdr_view.clone()
            } else {
                bloom_levels[i - 1].view.clone()
            };
            ds_down.push(make_single_input(&pipe_down, input)?);
        }

        let mut ds_up = Vec::new();
        if bloom_levels.len() > 1 {
            for i in 0..bloom_levels.len() - 1 {
                ds_up.push(make_single_input(
                    &pipe_up,
                    bloom_levels[i + 1].view.clone(),
                )?);
            }
        }

        let make_composite_input =
            |bloom_input: Arc<ImageView>| -> Result<Arc<DescriptorSet>, String> {
                let layout = pipe_composite
                    .layout()
                    .set_layouts()
                    .first()
                    .ok_or("post: composite pipeline missing set 0 layout")?
                    .clone();
                DescriptorSet::new(
                    ds_allocator.clone(),
                    layout,
                    [
                        WriteDescriptorSet::image_view_sampler(
                            0,
                            scene_hdr_view.clone(),
                            sampler.clone(),
                        ),
                        WriteDescriptorSet::image_view_sampler(1, bloom_input, sampler.clone()),
                    ],
                    [],
                )
                .map_err(|e| format!("post: composite descriptor set creation failed: {e}"))
            };

        let ds_composite_bloom = match bloom_levels.first() {
            Some(level0) => Some(make_composite_input(level0.view.clone())?),
            None => None,
        };
        let ds_composite_black = make_composite_input(black_view)?;

        Ok(PostEffectResources {
            pipe_down,
            pipe_up,
            pipe_composite,
            bloom_levels,
            final_color,
            final_framebuffer,
            scene_extent: extent,
            ds_down,
            ds_up,
            ds_composite_bloom,
            ds_composite_black,
        })
    }

    /// Build one full-screen post pipeline: shared fullscreen-triangle VS,
    /// the given FS, no depth, dynamic viewport + scissor (one pipeline
    /// serves every bloom level extent and survives resizes).
    fn create_post_pipeline(
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        fs_module: Arc<vulkano::shader::ShaderModule>,
        blend: Option<AttachmentBlend>,
    ) -> Result<Arc<GraphicsPipeline>, String> {
        let vs_module = post_fullscreen_vs::load(device.clone())
            .map_err(|e| format!("post: failed to load fullscreen vertex shader: {e}"))?;
        let vs_entry = vs_module
            .entry_point("main")
            .ok_or_else(|| "post: fullscreen vertex shader entry point not found".to_string())?;
        let fs_entry = fs_module
            .entry_point("main")
            .ok_or_else(|| "post: fragment shader entry point not found".to_string())?;

        let stages = [
            PipelineShaderStageCreateInfo::new(vs_entry),
            PipelineShaderStageCreateInfo::new(fs_entry),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .map_err(|e| format!("post: failed to create pipeline layout info: {e}"))?,
        )
        .map_err(|e| format!("post: failed to create pipeline layout: {e}"))?;

        let subpass = Subpass::from(render_pass, 0)
            .ok_or_else(|| "post: failed to get subpass from render pass".to_string())?;

        GraphicsPipeline::new(
            device,
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(VertexInputState::default()),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState {
                    attachments: vec![ColorBlendAttachmentState {
                        blend,
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
                dynamic_state: [DynamicState::Viewport, DynamicState::Scissor]
                    .into_iter()
                    .collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .map_err(|e| format!("post: failed to create post pipeline: {e}"))
    }

    /// Record the bloom down/upsample chain. Caller must ensure
    /// `res.has_bloom_chain()` is true.
    pub(super) fn record_bloom_chain(
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        res: &PostEffectResources,
        settings: &frame_input::BloomSettings,
    ) -> Result<(), String> {
        let knee = settings.threshold * 0.5;

        // Downsample: scene → level 0 → level 1 → …
        for (i, level) in res.bloom_levels.iter().enumerate() {
            let src_extent = if i == 0 {
                res.scene_extent
            } else {
                res.bloom_levels[i - 1].extent
            };
            let push = DownsamplePushConstants {
                src_texel: [1.0 / src_extent[0] as f32, 1.0 / src_extent[1] as f32],
                threshold: settings.threshold,
                knee,
                first_pass: u32::from(i == 0),
            };
            Self::record_post_pass(
                builder,
                level.framebuffer_down.clone(),
                level.extent,
                &res.pipe_down,
                res.ds_down[i].clone(),
                |b, layout| {
                    b.push_constants(layout, 0, push)
                        .map_err(|e| format!("post: downsample push_constants failed: {e}"))?;
                    Ok(())
                },
            )?;
        }

        // Upsample: level n-1 → +level n-2 → … → +level 0 (additive blend).
        for i in (0..res.bloom_levels.len().saturating_sub(1)).rev() {
            let src_extent = res.bloom_levels[i + 1].extent;
            let push = UpsamplePushConstants {
                src_texel: [1.0 / src_extent[0] as f32, 1.0 / src_extent[1] as f32],
            };
            Self::record_post_pass(
                builder,
                res.bloom_levels[i].framebuffer_up.clone(),
                res.bloom_levels[i].extent,
                &res.pipe_up,
                res.ds_up[i].clone(),
                |b, layout| {
                    b.push_constants(layout, 0, push)
                        .map_err(|e| format!("post: upsample push_constants failed: {e}"))?;
                    Ok(())
                },
            )?;
        }

        Ok(())
    }

    /// Record the composite/encode pass into the final 8-bit image. Always
    /// runs; `use_bloom` selects between the bloom chain and the transparent
    /// black fallback input.
    pub(super) fn record_composite(
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        res: &PostEffectResources,
        intensity: f32,
        use_bloom: bool,
    ) -> Result<(), String> {
        let ds = if use_bloom {
            res.ds_composite_bloom
                .clone()
                .ok_or("post: composite requested bloom input but the chain is empty")?
        } else {
            res.ds_composite_black.clone()
        };
        let push = CompositePushConstants { intensity };
        Self::record_post_pass(
            builder,
            res.final_framebuffer.clone(),
            res.scene_extent,
            &res.pipe_composite,
            ds,
            |b, layout| {
                b.push_constants(layout, 0, push)
                    .map_err(|e| format!("post: composite push_constants failed: {e}"))?;
                Ok(())
            },
        )
    }

    /// Shared skeleton for one full-screen pass: begin render pass, set the
    /// dynamic viewport/scissor to the target extent, bind pipeline + set 0,
    /// run `extra` (push constants), draw the fullscreen triangle, end.
    fn record_post_pass(
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        framebuffer: Arc<Framebuffer>,
        target_extent: [u32; 2],
        pipeline: &Arc<GraphicsPipeline>,
        descriptor_set: Arc<DescriptorSet>,
        extra: impl FnOnce(
            &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
            Arc<PipelineLayout>,
        ) -> Result<(), String>,
    ) -> Result<(), String> {
        // Every post render pass uses DontCare or Load — never Clear — so
        // the single colour attachment takes a `None` clear value.
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![None],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .map_err(|e| format!("post: begin_render_pass failed: {e}"))?
            // The post pipeline must be bound before the dynamic viewport /
            // scissor are set: vulkano validates dynamic-state commands
            // against the *currently bound* pipeline, which at this point
            // may still be a scene pipeline with fixed viewport state.
            .bind_pipeline_graphics(pipeline.clone())
            .map_err(|e| format!("post: bind_pipeline_graphics failed: {e}"))?
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [target_extent[0] as f32, target_extent[1] as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .map_err(|e| format!("post: set_viewport failed: {e}"))?
            .set_scissor(
                0,
                [Scissor {
                    offset: [0, 0],
                    extent: target_extent,
                }]
                .into_iter()
                .collect(),
            )
            .map_err(|e| format!("post: set_scissor failed: {e}"))?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .map_err(|e| format!("post: bind_descriptor_sets failed: {e}"))?;
        extra(builder, pipeline.layout().clone())?;
        unsafe {
            builder
                .draw(3, 1, 0, 0)
                .map_err(|e| format!("post: draw failed: {e}"))?;
        }
        builder
            .end_render_pass(SubpassEndInfo::default())
            .map_err(|e| format!("post: end_render_pass failed: {e}"))?;
        Ok(())
    }
}
