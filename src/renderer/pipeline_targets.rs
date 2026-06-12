//! Render-pass / pipeline / offscreen-target construction extracted
//! from `VulkanRenderer` as a follow-up slice of the #12 renderer
//! split. Owns:
//!
//! - `build_render_pass` — the scene render pass. Its colour attachment
//!   is always the HDR working format (`R16G16B16A16_SFLOAT`); the
//!   user-selected output colour space only affects the post-effect
//!   composite pass (see `post_effects.rs`), which encodes into the
//!   final 8-bit image.
//! - `create_offscreen_targets` — colour + depth `Image`s plus the
//!   framebuffer wired to the active render pass. The colour image is
//!   GPU-internal (`COLOR_ATTACHMENT | SAMPLED`): the post-effect
//!   passes sample it, and readback / export run against the
//!   post-effect layer's final 8-bit image instead.
//! - `rebuild_pipelines_and_targets` — the orchestration that
//!   bundles `create_offscreen_targets` with the six graphics
//!   pipeline variants (cull mode × alpha mode) plus the outline
//!   pipeline and the post-effect resources, then re-seeds
//!   `CameraRing` against the new layouts and drops caches that
//!   pinned old `PipelineLayout` Arcs.

use std::sync::Arc;

use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage, SampleCount};
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};

use crate::renderer::{frame_input, pipeline, CameraRing, VulkanRenderer};

/// Bundle of offscreen render-target resources produced by
/// [`VulkanRenderer::create_offscreen_targets`]. `color` is always the
/// single-sample HDR scene image sampled by the post-effect passes; on the
/// MSAA path it doubles as the render pass' resolve target. `msaa_color` is
/// the multisampled image the subpass actually renders into, present only
/// when MSAA is active. The readback / export source is the post-effect
/// layer's final 8-bit image, not anything in this bundle.
pub(super) struct OffscreenTargets {
    pub color: Arc<Image>,
    pub color_view: Arc<ImageView>,
    pub depth: Arc<Image>,
    pub depth_view: Arc<ImageView>,
    pub framebuffer: Arc<Framebuffer>,
    pub msaa_color: Option<Arc<Image>>,
    pub msaa_color_view: Option<Arc<ImageView>>,
}

impl VulkanRenderer {
    /// Build the scene render pass for the requested MSAA level. The colour
    /// attachment is always the HDR working format
    /// ([`super::post_effects::HDR_COLOR_FORMAT`]); the user's output colour
    /// space is applied later by the post-effect composite pass.
    ///
    /// `single_pass_renderpass!` takes the attachment `samples` as an
    /// *expression* (the macro calls `SampleCount::try_from($samples)`
    /// internally), so it can be a runtime value here — only the
    /// resolve-vs-no-resolve attachment *structure* has to branch:
    ///
    /// - `sample_count == 1`: the historical two-attachment pass — a
    ///   single-sample colour target (stored, then sampled by the post
    ///   passes) plus depth.
    /// - `sample_count > 1`: a three-attachment pass — an N-sample colour
    ///   target (`msaa_color`) that the subpass renders into and *resolves*
    ///   into the single-sample `color` target, plus an N-sample depth
    ///   target. The resolve keeps the post-effect input single-sample.
    ///
    /// `sample_count` is expected to already be clamped to device support by
    /// [`VulkanRenderer::clamp_sample_count`].
    pub(super) fn build_render_pass(device: Arc<Device>, sample_count: u32) -> Arc<RenderPass> {
        let color_format = super::post_effects::HDR_COLOR_FORMAT;
        let depth_format = Format::D32_SFLOAT_S8_UINT;
        if sample_count <= 1 {
            vulkano::single_pass_renderpass!(
                device,
                attachments: {
                    color: {
                        format: color_format,
                        samples: 1,
                        load_op: Clear,
                        store_op: Store,
                    },
                    depth_stencil: {
                        format: depth_format,
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
            .expect("failed to create single-sample render pass")
        } else {
            vulkano::single_pass_renderpass!(
                device,
                attachments: {
                    msaa_color: {
                        format: color_format,
                        samples: sample_count,
                        load_op: Clear,
                        store_op: DontCare,
                    },
                    color: {
                        format: color_format,
                        samples: 1,
                        load_op: DontCare,
                        store_op: Store,
                    },
                    depth_stencil: {
                        format: depth_format,
                        samples: sample_count,
                        load_op: Clear,
                        store_op: DontCare,
                    },
                },
                pass: {
                    color: [msaa_color],
                    color_resolve: [color],
                    depth_stencil: {depth_stencil},
                },
            )
            .expect("failed to create MSAA render pass")
        }
    }

    pub(super) fn create_offscreen_targets(
        memory_allocator: Arc<StandardMemoryAllocator>,
        render_pass: Arc<RenderPass>,
        extent: [u32; 2],
        sample_count: u32,
    ) -> OffscreenTargets {
        // The first colour attachment of the render pass dictates the
        // output target format. Reading it back here keeps the offscreen
        // image and the render pass guaranteed-compatible after a colour
        // space switch. On the MSAA path the first attachment is `msaa_color`
        // and the single-sample resolve target shares its format, so this is
        // correct for both paths.
        let color_format = render_pass
            .attachments()
            .first()
            .map(|a| a.format)
            .unwrap_or(Format::R8G8B8A8_SRGB);
        let samples = SampleCount::try_from(sample_count).unwrap_or(SampleCount::Sample1);

        // GPU-internal HDR scene target: rendered into by the scene pass and
        // sampled by the post-effect passes. The external-memory export
        // allocation lives on the post-effect layer's final 8-bit image.
        // TRANSFER_DST is required by the MSAA path: vulkano's
        // `single_pass_renderpass!` macro hardwires the resolve attachment
        // reference to the `TransferDstOptimal` layout, and that layout is
        // only valid on images carrying this usage bit.
        let color_create_info = ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: color_format,
            extent: [extent[0], extent[1], 1],
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
            ..Default::default()
        };

        let color_image = Image::new(
            memory_allocator.clone(),
            color_create_info,
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .expect("failed to create offscreen color image");

        // Depth must carry the same sample count as the colour attachment it
        // is paired with in the subpass.
        let depth_image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::D32_SFLOAT_S8_UINT,
                extent: [extent[0], extent[1], 1],
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                samples,
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

        // On the MSAA path the subpass renders into a dedicated multisampled
        // colour image and resolves it into `color_image` (single-sample).
        // GPU-internal only: no readback, sampling, or external sharing, so
        // plain `COLOR_ATTACHMENT` usage is enough.
        let (msaa_color, msaa_color_view) = if sample_count > 1 {
            let msaa_image = Image::new(
                memory_allocator,
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: color_format,
                    extent: [extent[0], extent[1], 1],
                    usage: ImageUsage::COLOR_ATTACHMENT,
                    samples,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .expect("failed to create MSAA color image");
            let view = ImageView::new_default(msaa_image.clone()).unwrap();
            (Some(msaa_image), Some(view))
        } else {
            (None, None)
        };

        // Framebuffer attachment order must match the render pass declaration
        // order built in `build_render_pass`: `[color, depth]` for 1×, and
        // `[msaa_color, color (resolve), depth]` for MSAA.
        let fb_attachments = match &msaa_color_view {
            Some(mv) => vec![mv.clone(), color_view.clone(), depth_view.clone()],
            None => vec![color_view.clone(), depth_view.clone()],
        };
        let framebuffer = Framebuffer::new(
            render_pass,
            FramebufferCreateInfo {
                attachments: fb_attachments,
                ..Default::default()
            },
        )
        .expect("failed to create offscreen framebuffer");

        OffscreenTargets {
            color: color_image,
            color_view,
            depth: depth_image,
            depth_view,
            framebuffer,
            msaa_color,
            msaa_color_view,
        }
    }

    /// Rebuild every render-pass-dependent piece of state for the given
    /// render pass and extent. Used by both `resize` (extent change with
    /// the same render pass) and `apply_color_space` (render pass swapped
    /// to a different colour-attachment format). The thumbnail cache holds
    /// a pipeline keyed to the previous render pass, so it is invalidated
    /// here too.
    pub(super) fn rebuild_pipelines_and_targets(
        &mut self,
        render_pass: Arc<RenderPass>,
        extent: [u32; 2],
        sample_count: u32,
    ) -> Result<(), String> {
        let device = self.device.as_ref().ok_or("renderer: no device")?.clone();
        let memory_allocator = self
            .memory_allocator
            .as_ref()
            .ok_or("renderer: no memory allocator")?
            .clone();

        let targets = Self::create_offscreen_targets(
            memory_allocator,
            render_pass.clone(),
            extent,
            sample_count,
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
            sample_count,
        )
        .map_err(|e| format!("failed to create graphics pipeline: {e}"))?;

        let gfx_pipeline_blend = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Back,
            frame_input::RenderAlphaMode::Blend,
            sample_count,
        )
        .map_err(|e| format!("failed to create blend graphics pipeline: {e}"))?;

        let no_cull_pipeline = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::None,
            frame_input::RenderAlphaMode::Opaque,
            sample_count,
        )
        .map_err(|e| format!("failed to create no-cull pipeline: {e}"))?;

        let no_cull_pipeline_blend = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::None,
            frame_input::RenderAlphaMode::Blend,
            sample_count,
        )
        .map_err(|e| format!("failed to create blend no-cull pipeline: {e}"))?;

        let front_cull_pipeline = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Front,
            frame_input::RenderAlphaMode::Opaque,
            sample_count,
        )
        .map_err(|e| format!("failed to create front-cull pipeline: {e}"))?;

        let front_cull_pipeline_blend = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Front,
            frame_input::RenderAlphaMode::Blend,
            sample_count,
        )
        .map_err(|e| format!("failed to create blend front-cull pipeline: {e}"))?;

        let cutout_pipeline = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Back,
            frame_input::RenderAlphaMode::Cutout,
            sample_count,
        )
        .map_err(|e| format!("failed to create cutout pipeline: {e}"))?;

        let no_cull_pipeline_cutout = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::None,
            frame_input::RenderAlphaMode::Cutout,
            sample_count,
        )
        .map_err(|e| format!("failed to create no-cull cutout pipeline: {e}"))?;

        let front_cull_pipeline_cutout = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Front,
            frame_input::RenderAlphaMode::Cutout,
            sample_count,
        )
        .map_err(|e| format!("failed to create front-cull cutout pipeline: {e}"))?;

        let background_pipeline = super::background::create_background_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            sample_count,
        )
        .map_err(|e| format!("failed to create background pipeline: {e}"))?;

        let outline_pipeline =
            pipeline::create_outline_pipeline(device.clone(), render_pass, viewport, sample_count)
                .map_err(|e| format!("failed to create outline pipeline: {e}"))?;

        // Post-effect resources are extent- and colour-space-dependent
        // (bloom chain, final 8-bit image, cached descriptor sets), so they
        // rebuild together with the offscreen targets.
        let ds_allocator = self
            .descriptor_set_allocator
            .as_ref()
            .ok_or("renderer: no descriptor set allocator")?
            .clone();
        let post_sampler = self
            .post_sampler
            .as_ref()
            .ok_or("renderer: no post sampler")?
            .clone();
        let black_view = self
            .black_texture_view
            .as_ref()
            .ok_or("renderer: no black texture")?
            .clone();
        let post_effects = Self::build_post_effect_resources(
            device,
            self.memory_allocator
                .as_ref()
                .ok_or("renderer: no memory allocator")?
                .clone(),
            ds_allocator,
            &self.current_color_space,
            extent,
            targets.color_view.clone(),
            post_sampler,
            black_view,
        )?;

        self.offscreen_color = Some(targets.color);
        self.offscreen_color_view = Some(targets.color_view);
        self.post_effects = Some(post_effects);
        self.offscreen_depth = Some(targets.depth);
        self.offscreen_depth_view = Some(targets.depth_view);
        self.offscreen_framebuffer = Some(targets.framebuffer);
        self.msaa_color = targets.msaa_color;
        self.msaa_color_view = targets.msaa_color_view;
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
        self.current_extent = extent;
        self.current_sample_count = sample_count;

        if let (Some(ma), Some(dsa)) = (&self.memory_allocator, &self.descriptor_set_allocator) {
            self.camera_ring = Some(CameraRing::new(ma, dsa, &gfx_pipeline, &outline_pipeline));
        }

        if let Some(ref mut ps) = self.active_pipeline {
            ps.graphics_pipeline = Some(gfx_pipeline);
        }

        // Material descriptor sets were allocated against the *previous*
        // `gfx_pipeline.layout().set_layouts()[1]`. The new graphics
        // pipelines built above carry brand-new `PipelineLayout` /
        // `DescriptorSetLayout` Arcs. Even though they are structurally
        // identical to the old layouts (same shaders), vulkano's bind
        // validation hashes by Arc identity in places, so persisting old
        // descriptor sets across a rebuild risks an `incompatible set
        // layout` error the moment they are bound under the new pipelines.
        // Drop the cache; first frame after a resize / colour-space change
        // rebuilds them lazily. `transform_cache` is unaffected because
        // its descriptor sets are pinned to the (untouched) compute
        // pipeline's layout.
        self.material_uploader.clear_cache();

        Ok(())
    }
}
