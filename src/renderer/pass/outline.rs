use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};

use crate::renderer::pipeline::GpuVertex;

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct OutlinePushConstants {
    pub outline_width: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

pub struct OutlineDrawInfo {
    pub vertex_buffer: Subbuffer<[GpuVertex]>,
    pub index_buffer: Subbuffer<[u32]>,
    pub index_count: u32,
    pub skinning_set: Arc<PersistentDescriptorSet>,
    pub outline_width: f32,
    pub outline_color: [f32; 3],
}

pub fn record_outline_pass(
    builder: &mut AutoCommandBufferBuilder<
        PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>,
        Arc<StandardCommandBufferAllocator>,
    >,
    outline_pipeline: &Arc<GraphicsPipeline>,
    camera_buffer: Subbuffer<crate::renderer::CameraUniform>,
    ds_allocator: &Arc<StandardDescriptorSetAllocator>,
    draws: &[OutlineDrawInfo],
) -> Result<(), String> {
    if draws.is_empty() {
        return Ok(());
    }

    builder
        .bind_pipeline_graphics(outline_pipeline.clone())
        .map_err(|e| format!("outline: bind pipeline: {e}"))?;

    let outline_camera_set_layout = outline_pipeline
        .layout()
        .set_layouts()
        .get(0)
        .ok_or("outline: no camera set layout")?
        .clone();
    let outline_camera_set = PersistentDescriptorSet::new(
        ds_allocator,
        outline_camera_set_layout,
        [WriteDescriptorSet::buffer(0, camera_buffer)],
        [],
    )
    .map_err(|e| format!("outline: camera descriptor set: {e}"))?;

    builder
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            outline_pipeline.layout().clone(),
            0,
            outline_camera_set,
        )
        .map_err(|e| format!("outline: bind camera descriptor: {e}"))?;

    for draw in draws {
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                outline_pipeline.layout().clone(),
                1,
                draw.skinning_set.clone(),
            )
            .map_err(|e| format!("outline: bind skinning descriptor: {e}"))?;

        let push = OutlinePushConstants {
            outline_width: draw.outline_width,
            r: draw.outline_color[0],
            g: draw.outline_color[1],
            b: draw.outline_color[2],
            a: 1.0,
        };
        builder
            .push_constants(outline_pipeline.layout().clone(), 0, push)
            .map_err(|e| format!("outline: push constants: {e}"))?;

        builder
            .bind_vertex_buffers(0, draw.vertex_buffer.clone())
            .map_err(|e| format!("outline: bind vertex buffer: {e}"))?
            .bind_index_buffer(draw.index_buffer.clone())
            .map_err(|e| format!("outline: bind index buffer: {e}"))?
            .draw_indexed(draw.index_count, 1, 0, 0, 0)
            .map_err(|e| format!("outline: draw indexed: {e}"))?;
    }

    Ok(())
}
