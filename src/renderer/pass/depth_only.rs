use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};

use super::forward::ForwardPassDraw;

pub fn record_depth_prepass(
    builder: &mut AutoCommandBufferBuilder<
        PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>,
        Arc<StandardCommandBufferAllocator>,
    >,
    depth_pipeline: &Arc<GraphicsPipeline>,
    camera_buffer: Subbuffer<crate::renderer::CameraUniform>,
    ds_allocator: &Arc<StandardDescriptorSetAllocator>,
    draws: &[ForwardPassDraw],
) -> Result<(), String> {
    if draws.is_empty() {
        return Ok(());
    }

    builder
        .bind_pipeline_graphics(depth_pipeline.clone())
        .map_err(|e| format!("depth prepass: bind pipeline: {e}"))?;

    let camera_set_layout = depth_pipeline
        .layout()
        .set_layouts()
        .get(0)
        .ok_or("depth prepass: no camera set layout")?
        .clone();
    let camera_set = PersistentDescriptorSet::new(
        ds_allocator,
        camera_set_layout,
        [WriteDescriptorSet::buffer(0, camera_buffer)],
        [],
    )
    .map_err(|e| format!("depth prepass: camera descriptor set: {e}"))?;

    builder
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            depth_pipeline.layout().clone(),
            0,
            camera_set,
        )
        .map_err(|e| format!("depth prepass: bind camera descriptor: {e}"))?;

    for draw in draws {
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                depth_pipeline.layout().clone(),
                1,
                draw.skinning_set.clone(),
            )
            .map_err(|e| format!("depth prepass: bind skinning descriptor: {e}"))?;

        builder
            .bind_vertex_buffers(0, draw.vertex_buffer.clone())
            .map_err(|e| format!("depth prepass: bind vertex buffer: {e}"))?
            .bind_index_buffer(draw.index_buffer.clone())
            .map_err(|e| format!("depth prepass: bind index buffer: {e}"))?
            .draw_indexed(draw.index_count, 1, 0, 0, 0)
            .map_err(|e| format!("depth prepass: draw indexed: {e}"))?;
    }

    Ok(())
}
