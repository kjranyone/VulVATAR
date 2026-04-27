use std::sync::Arc;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};

#[allow(clippy::type_complexity)]
pub fn create_offscreen_targets(
    memory_allocator: Arc<StandardMemoryAllocator>,
    render_pass: Arc<RenderPass>,
    extent: [u32; 2],
) -> Result<
    (
        Arc<Image>,
        Arc<ImageView>,
        Arc<Image>,
        Arc<ImageView>,
        Arc<Framebuffer>,
    ),
    String,
> {
    let color_image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_SRGB,
            extent: [extent[0], extent[1], 1],
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .map_err(|e| format!("failed to create offscreen color image: {e}"))?;

    let depth_image = Image::new(
        memory_allocator,
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::D32_SFLOAT,
            extent: [extent[0], extent[1], 1],
            usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .map_err(|e| format!("failed to create offscreen depth image: {e}"))?;

    let color_view = ImageView::new_default(color_image.clone())
        .map_err(|e| format!("failed to create color image view: {e}"))?;
    let depth_view = ImageView::new_default(depth_image.clone())
        .map_err(|e| format!("failed to create depth image view: {e}"))?;

    let framebuffer = Framebuffer::new(
        render_pass,
        FramebufferCreateInfo {
            attachments: vec![color_view.clone(), depth_view.clone()],
            ..Default::default()
        },
    )
    .map_err(|e| format!("failed to create offscreen framebuffer: {e}"))?;

    Ok((
        color_image,
        color_view,
        depth_image,
        depth_view,
        framebuffer,
    ))
}
