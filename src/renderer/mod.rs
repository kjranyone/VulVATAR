pub mod debug;
pub mod frame_input;
pub mod material;
pub mod mtoon;
pub mod output_export;
pub mod pipeline;

use crate::asset::{MeshId, MeshPrimitiveAsset, PrimitiveId, VertexData};
use frame_input::{ClothDeformSnapshot, RenderFrameInput};
use output_export::OutputExporter;
use pipeline::GpuVertex;
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents, SubpassEndInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
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

pub struct VulkanRenderer {
    initialized: bool,
    output_exporter: OutputExporter,
    material_uploader: material::MaterialUploader,
    active_pipeline: Option<pipeline::PipelineState>,
    frame_counter: u64,
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
    pipeline_no_cull: Option<Arc<GraphicsPipeline>>,
    pipeline_front_cull: Option<Arc<GraphicsPipeline>>,
    outline_pipeline: Option<Arc<GraphicsPipeline>>,
    sampler: Option<Arc<Sampler>>,
    default_texture_view: Option<Arc<ImageView>>,
    offscreen_color: Option<Arc<Image>>,
    offscreen_color_view: Option<Arc<ImageView>>,
    offscreen_depth: Option<Arc<Image>>,
    offscreen_depth_view: Option<Arc<ImageView>>,
    offscreen_framebuffer: Option<Arc<Framebuffer>>,
    current_extent: [u32; 2],
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
            pipeline_no_cull: None,
            pipeline_front_cull: None,
            outline_pipeline: None,
            sampler: None,
            default_texture_view: None,
            offscreen_color: None,
            offscreen_color_view: None,
            offscreen_depth: None,
            offscreen_depth_view: None,
            offscreen_framebuffer: None,
            current_extent: [1920, 1080],
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

        let device_extensions = DeviceExtensions {
            ..DeviceExtensions::empty()
        };
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .expect("failed to enumerate physical devices")
            .filter(|p| p.supported_extensions().contains(&device_extensions))
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

        println!(
            "renderer: selected device: {} ({:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

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

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: Format::R8G8B8A8_SRGB,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                depth_stencil: {
                    format: Format::D32_SFLOAT,
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
        .expect("failed to create render pass");

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
        );

        let no_cull_pipeline = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::None,
        );
        let front_cull_pipeline = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Front,
        );
        let outline_pipeline = pipeline::create_outline_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport,
        );

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
        self.memory_allocator = Some(memory_allocator);
        self.command_buffer_allocator = Some(command_buffer_allocator);
        self.descriptor_set_allocator = Some(descriptor_set_allocator);
        self.render_pass = Some(render_pass);
        self.graphics_pipeline = Some(gfx_pipeline.clone());
        self.pipeline_no_cull = Some(no_cull_pipeline);
        self.pipeline_front_cull = Some(front_cull_pipeline);
        self.outline_pipeline = Some(outline_pipeline);
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
            graphics_pipeline: Some(gfx_pipeline),
        });

        self.initialized = true;
        println!("renderer: initialized with Vulkano");
    }

    pub fn resize(&mut self, new_extent: [u32; 2]) {
        if !self.initialized {
            return;
        }
        if new_extent == self.current_extent {
            return;
        }
        if new_extent[0] == 0 || new_extent[1] == 0 {
            return;
        }

        let device = self.device.as_ref().unwrap().clone();
        let memory_allocator = self.memory_allocator.as_ref().unwrap().clone();
        let render_pass = self.render_pass.as_ref().unwrap().clone();

        let (color_img, color_view, depth_img, depth_view, framebuffer) =
            Self::create_offscreen_targets(
                device.clone(),
                memory_allocator,
                render_pass.clone(),
                new_extent,
            );

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [new_extent[0] as f32, new_extent[1] as f32],
            depth_range: 0.0..=1.0,
        };
        use vulkano::pipeline::graphics::rasterization::CullMode;
        let gfx_pipeline = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Back,
        );

        let no_cull_pipeline = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::None,
        );
        let front_cull_pipeline = pipeline::create_graphics_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
            CullMode::Front,
        );
        let outline_pipeline =
            pipeline::create_outline_pipeline(device, render_pass, viewport);

        self.offscreen_color = Some(color_img);
        self.offscreen_color_view = Some(color_view);
        self.offscreen_depth = Some(depth_img);
        self.offscreen_depth_view = Some(depth_view);
        self.offscreen_framebuffer = Some(framebuffer);
        self.graphics_pipeline = Some(gfx_pipeline.clone());
        self.pipeline_no_cull = Some(no_cull_pipeline);
        self.pipeline_front_cull = Some(front_cull_pipeline);
        self.outline_pipeline = Some(outline_pipeline);
        self.current_extent = new_extent;

        if let Some(ref mut ps) = self.active_pipeline {
            ps.graphics_pipeline = Some(gfx_pipeline);
        }

        println!(
            "renderer: resized to {}x{}",
            new_extent[0], new_extent[1]
        );
    }

    pub fn render(&mut self, input: &RenderFrameInput) -> RenderResult {
        if !self.initialized {
            eprintln!("renderer: skipped because Vulkan is not initialized");
            return RenderResult {
                extent: [0, 0],
                timestamp_nanos: 0,
                has_alpha: false,
                stats: RenderStats::default(),
                exported_frame: None,
            };
        }

        let requested_extent = input.output_request.extent;
        if requested_extent != self.current_extent
            && requested_extent[0] > 0
            && requested_extent[1] > 0
        {
            self.resize(requested_extent);
        }

        let device = self.device.as_ref().unwrap().clone();
        let queue = self.queue.as_ref().unwrap().clone();
        let memory_allocator = self.memory_allocator.as_ref().unwrap().clone();
        let cb_allocator = self.command_buffer_allocator.as_ref().unwrap().clone();
        let ds_allocator = self.descriptor_set_allocator.as_ref().unwrap().clone();
        let gfx_pipeline = self.graphics_pipeline.as_ref().unwrap().clone();
        let outline_pipeline = self.outline_pipeline.as_ref().unwrap().clone();
        let framebuffer = self.offscreen_framebuffer.as_ref().unwrap().clone();
        let color_image = self.offscreen_color.as_ref().unwrap().clone();
        let sampler = self.sampler.as_ref().unwrap().clone();
        let default_tex = self.default_texture_view.as_ref().unwrap().clone();

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
            &cb_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("failed to create command buffer");

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some([0.0, 0.0, 0.0, 0.0].into()),
                        Some(1.0f32.into()),
                    ],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap()
            .bind_pipeline_graphics(gfx_pipeline.clone())
            .unwrap();

        // Camera uniform (set 0)
        let ld = input.lighting.main_light_dir_ws;
        let ld_len =
            (ld[0] * ld[0] + ld[1] * ld[1] + ld[2] * ld[2]).sqrt().max(1e-6);
        let light_dir = [ld[0] / ld_len, ld[1] / ld_len, ld[2] / ld_len];

        let camera_data = CameraUniform {
            view: mat4_to_cols(input.camera.view),
            proj: mat4_to_cols(input.camera.projection),
            camera_pos: input.camera.position_ws,
            _pad0: 0.0,
            light_dir,
            light_intensity: input.lighting.main_light_intensity,
            ambient_term: input.lighting.ambient_term,
            _pad1: 0.0,
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
        .expect("failed to create camera buffer");

        let camera_set_layout = gfx_pipeline.layout().set_layouts().get(0).unwrap().clone();
        let camera_set = PersistentDescriptorSet::new(
            &ds_allocator,
            camera_set_layout,
            [WriteDescriptorSet::buffer(0, camera_buffer.clone())],
            [],
        )
        .expect("failed to create camera descriptor set");

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                gfx_pipeline.layout().clone(),
                0,
                camera_set,
            )
            .unwrap();

        // Collect outline draws for second pass
        struct OutlineDrawInfo {
            vertex_buffer: Subbuffer<[GpuVertex]>,
            index_buffer: Subbuffer<[u32]>,
            index_count: u32,
            skinning_set: Arc<PersistentDescriptorSet>,
            outline_width: f32,
            outline_color: [f32; 3],
        }
        let mut outline_draws: Vec<OutlineDrawInfo> = Vec::new();

        // Draw each avatar instance
        for instance in &input.instances {
            let skinning_mats: Vec<[[f32; 4]; 4]> =
                if instance.skinning_matrices.is_empty() {
                    vec![crate::asset::identity_matrix()]
                } else {
                    instance
                        .skinning_matrices
                        .iter()
                        .map(|m| mat4_to_cols(*m))
                        .collect()
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
            .expect("failed to create skinning buffer");

            let skinning_set_layout =
                gfx_pipeline.layout().set_layouts().get(1).unwrap().clone();
            let skinning_set = PersistentDescriptorSet::new(
                &ds_allocator,
                skinning_set_layout,
                [WriteDescriptorSet::buffer(0, skinning_buffer)],
                [],
            )
            .expect("failed to create skinning descriptor set");

            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    gfx_pipeline.layout().clone(),
                    1,
                    skinning_set.clone(),
                )
                .unwrap();

            for mesh_inst in &instance.mesh_instances {
                // Resolve texture
                let texture_view =
                    self.resolve_texture(&mesh_inst.material_binding.textures, &default_tex);

                // Upload material (set 2) with texture
                let material_set = self.material_uploader.upload_to_gpu(
                    &mesh_inst.material_binding,
                    memory_allocator.clone(),
                    ds_allocator.clone(),
                    &gfx_pipeline,
                    texture_view,
                    sampler.clone(),
                );

                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        gfx_pipeline.layout().clone(),
                        2,
                        material_set,
                    )
                    .unwrap();

                let (vertex_buffer, index_buffer, index_count) =
                    if let Some(ref prim_asset) = mesh_inst.primitive_data {
                        if let Some(ref cloth) = instance.cloth_deform {
                            if let Some(ref vd) = prim_asset.vertices {
                                let verts = Self::build_cloth_vertices(vd, cloth);
                                let indices =
                                    prim_asset.indices.as_deref().unwrap_or(&[]);
                                let idx_count = indices.len() as u32;
                                let vb = Self::create_vertex_buffer(
                                    memory_allocator.clone(),
                                    &verts,
                                );
                                let ib = Self::create_index_buffer(
                                    memory_allocator.clone(),
                                    indices,
                                );
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
                    .unwrap()
                    .bind_index_buffer(index_buffer.clone())
                    .unwrap()
                    .draw_indexed(index_count, 1, 0, 0, 0)
                    .unwrap();

                // Collect for outline pass
                if mesh_inst.outline.enabled && mesh_inst.outline.width > 0.0 {
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

        // --- Outline pass ---
        if !outline_draws.is_empty() {
            builder
                .bind_pipeline_graphics(outline_pipeline.clone())
                .unwrap();

            let outline_camera_set_layout = outline_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .unwrap()
                .clone();
            let outline_camera_set = PersistentDescriptorSet::new(
                &ds_allocator,
                outline_camera_set_layout,
                [WriteDescriptorSet::buffer(0, camera_buffer)],
                [],
            )
            .expect("failed to create outline camera descriptor set");

            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    outline_pipeline.layout().clone(),
                    0,
                    outline_camera_set,
                )
                .unwrap();

            for draw in &outline_draws {
                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        outline_pipeline.layout().clone(),
                        1,
                        draw.skinning_set.clone(),
                    )
                    .unwrap();

                let push = OutlinePushConstants {
                    outline_width: draw.outline_width,
                    r: draw.outline_color[0],
                    g: draw.outline_color[1],
                    b: draw.outline_color[2],
                    a: 1.0,
                };
                builder
                    .push_constants(outline_pipeline.layout().clone(), 0, push)
                    .unwrap();

                builder
                    .bind_vertex_buffers(0, draw.vertex_buffer.clone())
                    .unwrap()
                    .bind_index_buffer(draw.index_buffer.clone())
                    .unwrap()
                    .draw_indexed(draw.index_count, 1, 0, 0, 0)
                    .unwrap();
            }
        }

        builder
            .end_render_pass(SubpassEndInfo::default())
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = vulkano::sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)
            .unwrap();

        self.frame_counter += 1;
        let timestamp_nanos = self.frame_counter * 16_666_667u64;
        let extent = self.current_extent;

        let export_result = self.output_exporter.export_readback(
            &output_export::ExportRequest {
                output_request: input.output_request.clone(),
                color_image_id: 0,
                alpha_image_id: None,
            },
            color_image,
            memory_allocator,
            cb_allocator,
            queue,
            future.boxed(),
            timestamp_nanos,
        );

        RenderResult {
            extent,
            timestamp_nanos,
            has_alpha: true,
            stats: RenderStats {
                instance_count: input.instances.len() as u32,
                mesh_count: total_meshes,
                material_count: total_materials,
                cloth_instances,
            },
            exported_frame: export_result.exported_frame,
        }
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
        let view = self.upload_texture(uri).unwrap_or_else(|| {
            eprintln!(
                "renderer: failed to load texture '{}', using default white",
                uri
            );
            default_tex.clone()
        });
        self.texture_cache.insert(uri.clone(), view.clone());
        view
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
            &cb_allocator,
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
        Self::upload_rgba_texture(
            device,
            memory_allocator,
            cb_allocator,
            queue,
            1,
            1,
            &pixels,
        )
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

    fn build_cloth_vertices(
        vd: &VertexData,
        cloth: &ClothDeformSnapshot,
    ) -> Vec<GpuVertex> {
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

    fn create_offscreen_targets(
        _device: Arc<Device>,
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
        let color_image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_SRGB,
                extent: [extent[0], extent[1], 1],
                usage: ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::TRANSFER_SRC
                    | ImageUsage::SAMPLED,
                ..Default::default()
            },
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
    ambient_term: [f32; 3],
    _pad1: f32,
}

/// Convert the project's row-major `Mat4` into column-major for GLSL.
fn mat4_to_cols(m: crate::asset::Mat4) -> [[f32; 4]; 4] {
    [
        [m[0][0], m[1][0], m[2][0], m[3][0]],
        [m[0][1], m[1][1], m[2][1], m[3][1]],
        [m[0][2], m[1][2], m[2][2], m[3][2]],
        [m[0][3], m[1][3], m[2][3], m[3][3]],
    ]
}
