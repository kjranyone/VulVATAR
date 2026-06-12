//! Texture upload / cache layer extracted from `VulkanRenderer` as the
//! first slice of the #12 renderer split. All methods here own only
//! the texture cache HashMap and ad-hoc Vulkan handles passed in
//! through `self`; nothing here touches the graphics pipeline, the
//! render pass, or the readback ring.

use std::sync::Arc;

use log::warn;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::sync::GpuFuture;

use crate::renderer::material;
use crate::renderer::VulkanRenderer;

impl VulkanRenderer {
    pub(super) fn resolve_texture(
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

    pub(super) fn resolve_matcap_texture(
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

    pub(super) fn resolve_shade_texture(
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

    pub(super) fn upload_rgba_texture(
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
        super::gpu_wait::wait_fence_bounded(future, "texture_upload").ok()?;

        ImageView::new_default(gpu_image).ok()
    }

    pub(super) fn create_default_white_texture(
        device: Arc<Device>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        cb_allocator: Arc<StandardCommandBufferAllocator>,
        queue: Arc<Queue>,
    ) -> Arc<ImageView> {
        let pixels: [u8; 4] = [255, 255, 255, 255];
        Self::upload_rgba_texture(device, memory_allocator, cb_allocator, queue, 1, 1, &pixels)
            .expect("failed to create default white texture")
    }
}
