//! Scene pass — slice 7 of the #12 renderer split. Records the offscreen
//! render pass that draws the generative background, the opaque/blend
//! avatar draws (consuming the frame plan's [`DrawInfo`] list),
//! and the outline pass. Camera / material descriptor sets are bound
//! here; the HDR scene target then feeds the post-effect layer.

use std::sync::Arc;

use vulkano::command_buffer::{
    AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo,
    SubpassContents, SubpassEndInfo,
};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::Framebuffer;

use crate::renderer::background;
use crate::renderer::frame_plan::FramePlan;
use crate::renderer::VulkanRenderer;

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

impl VulkanRenderer {
    /// Record the offscreen scene render pass: generative background,
    /// opaque-then-blend avatar draws, outline pass. All values come from
    /// the prepared [`FramePlan`] so a cached command buffer is
    /// byte-equivalent to a fresh recording of the same plan.
    pub(super) fn record_scene_pass(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        plan: &FramePlan,
        framebuffer: &Arc<Framebuffer>,
        gfx_pipeline: &Arc<GraphicsPipeline>,
        outline_pipeline: &Arc<GraphicsPipeline>,
    ) -> Result<(), String> {
        // ── Graphics pass: forward draws then outlines ──────────────────
        // Vulkano inserts the compute-write → vertex-input-read barrier on
        // each `transformed_vbo` automatically because the dispatch and the
        // draw share this single command buffer.
        // Clear values must line up with the render pass attachment order
        // built in `build_render_pass`:
        //   1×:   [color (Clear), depth (Clear)]
        //   MSAA: [msaa_color (Clear), color/resolve (DontCare → None), depth (Clear)]
        let clear_values = if self.current_sample_count > 1 {
            vec![
                Some(plan.clear.into()),
                None,
                Some(vulkano::format::ClearValue::DepthStencil((1.0, 0))),
            ]
        } else {
            vec![
                Some(plan.clear.into()),
                Some(vulkano::format::ClearValue::DepthStencil((1.0, 0))),
            ]
        };
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values,
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .map_err(|e| format!("render: begin_render_pass failed: {e}"))?;

        // Generative background: first draw inside the scene pass. Painter's
        // order alone keeps it behind the avatar (its pipeline neither tests
        // nor writes depth), and being inside the HDR scene pass means the
        // bloom chain picks up its highlights like any other scene content.
        if let (Some(bg_pipeline), Some(bg_set)) = (&plan.bg_pipeline, &plan.bg_set) {
            background::record_background(builder, bg_pipeline, bg_set.clone())?;
        }

        builder
            .bind_pipeline_graphics(gfx_pipeline.clone())
            .map_err(|e| format!("render: bind_pipeline_graphics failed: {e}"))?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                gfx_pipeline.layout().clone(),
                0,
                plan.camera_set.clone(),
            )
            .map_err(|e| format!("render: bind camera descriptor set failed: {e}"))?;

        // Opaque pass then blend pass. The camera set 0 layout matches
        // across all graphics variants, so it stays bound when we swap
        // variant pipelines (Vulkan layout compatibility for set 0).
        for blend_pass in [false, true] {
            for draw in &plan.draws {
                let is_blend = matches!(
                    draw.alpha_mode,
                    crate::renderer::frame_input::RenderAlphaMode::Blend
                );
                if is_blend != blend_pass {
                    continue;
                }
                builder
                    .bind_pipeline_graphics(draw.pipeline.clone())
                    .map_err(|e| format!("render: bind_pipeline_graphics for variant failed: {e}"))?
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        draw.pipeline.layout().clone(),
                        1,
                        draw.material_set.clone(),
                    )
                    .map_err(|e| format!("render: bind material descriptor set failed: {e}"))?
                    .bind_vertex_buffers(0, draw.vertex_buffer.clone())
                    .map_err(|e| format!("render: bind_vertex_buffers failed: {e}"))?
                    .bind_index_buffer(draw.index_buffer.clone())
                    .map_err(|e| format!("render: bind_index_buffer failed: {e}"))?;
                unsafe {
                    builder
                        .draw_indexed(draw.index_count, 1, 0, 0, 0)
                        .map_err(|e| format!("render: draw_indexed failed: {e}"))?;
                }
            }
        }

        // ── Outline pass ────────────────────────────────────────────────
        let outline_count = plan.draws.iter().filter(|d| d.outline.is_some()).count();
        if outline_count > 0 {
            builder
                .bind_pipeline_graphics(outline_pipeline.clone())
                .map_err(|e| format!("render: bind outline pipeline failed: {e}"))?
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    outline_pipeline.layout().clone(),
                    0,
                    plan.outline_camera_set.clone(),
                )
                .map_err(|e| format!("render: bind outline camera set failed: {e}"))?;

            for draw in &plan.draws {
                let Some((width, color)) = draw.outline else {
                    continue;
                };
                let push = OutlinePushConstants {
                    outline_width: width,
                    r: color[0],
                    g: color[1],
                    b: color[2],
                    a: 1.0,
                };
                builder
                    .push_constants(outline_pipeline.layout().clone(), 0, push)
                    .map_err(|e| format!("render: push_constants failed: {e}"))?
                    .bind_vertex_buffers(0, draw.vertex_buffer.clone())
                    .map_err(|e| format!("render: outline bind_vertex_buffers failed: {e}"))?
                    .bind_index_buffer(draw.index_buffer.clone())
                    .map_err(|e| format!("render: outline bind_index_buffer failed: {e}"))?;
                unsafe {
                    builder
                        .draw_indexed(draw.index_count, 1, 0, 0, 0)
                        .map_err(|e| format!("render: outline draw_indexed failed: {e}"))?;
                }
            }
        }

        builder
            .end_render_pass(SubpassEndInfo::default())
            .map_err(|e| format!("render: end_render_pass failed: {e}"))?;
        Ok(())
    }
}
