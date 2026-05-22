//! Render-pass / pipeline / offscreen-target construction extracted
//! from `VulkanRenderer` as a follow-up slice of the #12 renderer
//! split. Owns:
//!
//! - `build_render_pass` — one of two `single_pass_renderpass!`
//!   instantiations based on the requested colour-attachment format
//!   (sRGB vs LinearSrgb).
//! - `create_offscreen_targets` — colour + depth `Image`s plus the
//!   framebuffer wired to the active render pass. The colour image
//!   tries to allocate with `OPAQUE_WIN32_KMT` external-memory
//!   handle types when the device supports them (for GPU sharing
//!   with the MF Virtual Camera DLL); falls back to a vanilla
//!   `COLOR_ATTACHMENT | TRANSFER_SRC | SAMPLED` image otherwise.
//! - `rebuild_pipelines_and_targets` — the orchestration that
//!   bundles `create_offscreen_targets` with the six graphics
//!   pipeline variants (cull mode × alpha mode) plus the outline
//!   pipeline, then re-seeds `CameraRing` against the new layouts
//!   and drops caches that pinned old `PipelineLayout` Arcs.

use std::sync::Arc;

use log::{info, warn};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};

use crate::renderer::{frame_input, pipeline, CameraRing, VulkanRenderer};

impl VulkanRenderer {
    /// Build a render pass whose colour attachment matches the requested
    /// output colour space. Two arms because `single_pass_renderpass!` is a
    /// macro that doesn't accept a runtime `Format` value.
    pub(super) fn build_render_pass(
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
    pub(super) fn create_offscreen_targets(
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

        (color_image, color_view, depth_image, depth_view, framebuffer)
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
        // pass without an "incompatible render pass" panic.
        self.thumb_cache = None;

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
