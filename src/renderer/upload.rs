use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
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

use crate::asset::VertexData;
use crate::renderer::frame_input::ClothDeformSnapshot;
use crate::renderer::pipeline::GpuVertex;

pub fn upload_rgba_texture(
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

pub fn create_default_white_texture(
    device: Arc<Device>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    cb_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
) -> Arc<ImageView> {
    let pixels: [u8; 4] = [255, 255, 255, 255];
    upload_rgba_texture(device, memory_allocator, cb_allocator, queue, 1, 1, &pixels)
        .expect("failed to create default white texture")
}

pub fn vertex_data_to_gpu(vd: &VertexData) -> Vec<GpuVertex> {
    build_vertices_impl(
        &vd.positions,
        |i| vd.positions[i],
        |i| {
            if i < vd.normals.len() {
                vd.normals[i]
            } else {
                [0.0, 1.0, 0.0]
            }
        },
        vd,
        vd.positions.len(),
    )
}

pub fn build_cloth_vertices(vd: &VertexData, cloth: &ClothDeformSnapshot) -> Vec<GpuVertex> {
    build_vertices_impl(
        &cloth.deformed_positions,
        |i| {
            if i < cloth.deformed_positions.len() {
                cloth.deformed_positions[i]
            } else {
                vd.positions[i]
            }
        },
        |i| {
            if let Some(ref dn) = cloth.deformed_normals {
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
            }
        },
        vd,
        vd.positions.len(),
    )
}

fn build_vertices_impl<FPos, FNorm>(
    _positions: &[[f32; 3]],
    pos_fn: FPos,
    norm_fn: FNorm,
    vd: &VertexData,
    count: usize,
) -> Vec<GpuVertex>
where
    FPos: Fn(usize) -> [f32; 3],
    FNorm: Fn(usize) -> [f32; 3],
{
    (0..count)
        .map(|i| {
            let pos = pos_fn(i);
            let norm = norm_fn(i);
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

pub fn create_vertex_buffer(
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

pub fn create_index_buffer(
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

pub fn create_placeholder_mesh(
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

    let vertex_buffer = create_vertex_buffer(memory_allocator.clone(), &vertices);
    let index_buffer = create_index_buffer(memory_allocator, &indices);

    (vertex_buffer, index_buffer, 3)
}
