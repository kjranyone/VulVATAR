//! Generative (procedural) background: a full-screen flow-field shader
//! drawn as the *first* draw inside the HDR scene render pass, behind the
//! avatar, so the bloom post chain applies to its highlights like any other
//! scene content.
//!
//! - **Flow field**: curl noise (perpendicular of an fbm gradient) drives a
//!   short per-pixel march that accumulates ridged fbm into light streaks.
//! - **Tracking reactivity**: the avatar's hand positions act as swirl
//!   attractors on the field, the head position emits a radial wave, and
//!   the mouth-open viseme pulses the brightness. All reactive terms are
//!   pre-scaled on the CPU by `avatar_opacity` (tracking-loss fade) and
//!   zeroed when no avatar is loaded.
//!
//! Everything per-frame reaches the shader as push constants — changing any
//! setting never rebuilds the pipeline. The pipeline itself is rebuilt with
//! the scene pipelines (same render pass, baked viewport and MSAA sample
//! count). Depth test/write and stencil stay disabled: painter's order
//! alone puts the background behind the avatar, and the cleared depth
//! buffer is left untouched for the avatar draws.
//!
//! The shader writes linear HDR (streak peaks exceed 1.0 so the default
//! bloom threshold picks them up) with alpha = 1.0 — an enabled generative
//! background makes the output opaque regardless of the transparent
//! background setting.

use std::sync::Arc;

use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::device::Device;
use vulkano::image::SampleCount;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{RenderPass, Subpass};

use super::post_effects::post_fullscreen_vs;
use crate::renderer::frame_input::RenderFrameInput;

/// Flow-field fragment shader. Cost knobs are the march step count and the
/// fbm octave counts (direction field uses fewer octaves than the streak
/// lookup); tune those first if the background shows up in frame profiles.
mod background_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform Push {
    vec4 color_a;      // rgb used, w ignored
    vec4 color_b;      // rgb used, w ignored
    vec2 head_uv;
    vec2 left_uv;
    vec2 right_uv;
    vec2 resolution;
    float time;
    float intensity;
    float speed;
    float scale;
    float reactivity;  // pre-scaled by avatar_opacity, 0 when invalid
    float mouth_open;
    float _pad0;
    float _pad1;
} pc;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Cheap 3-octave fbm for the direction field (sampled 3x per march step).
float fbm3(vec2 p) {
    float v = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 3; i++) {
        v += amp * vnoise(p);
        p = p * 2.03 + vec2(17.3, 9.1);
        amp *= 0.5;
    }
    return v;
}

// 4-octave fbm for the streak/ridge lookup (sampled once per march step).
float fbm4(vec2 p) {
    float v = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 4; i++) {
        v += amp * vnoise(p);
        p = p * 2.03 + vec2(17.3, 9.1);
        amp *= 0.5;
    }
    return v;
}

// Curl-ish direction: perpendicular of the fbm gradient (forward
// differences — 3 fbm samples). Divergence-free enough for streaks.
vec2 flow_dir(vec2 q, float t) {
    const float e = 0.04;
    vec2 drift = vec2(0.0, t * 0.12);
    float n0 = fbm3(q + drift);
    float nx = fbm3(q + drift + vec2(e, 0.0));
    float ny = fbm3(q + drift + vec2(0.0, e));
    vec2 grad = vec2(nx - n0, ny - n0) / e;
    return vec2(grad.y, -grad.x);
}

// Tangential swirl around a hand anchor, Gaussian falloff.
vec2 swirl(vec2 p, vec2 c) {
    vec2 d = p - c;
    float fall = exp(-dot(d, d) * 14.0);
    return vec2(-d.y, d.x) * fall;
}

void main() {
    float aspect = pc.resolution.x / max(pc.resolution.y, 1.0);
    vec2 p = vec2(frag_uv.x * aspect, frag_uv.y);
    vec2 head = vec2(pc.head_uv.x * aspect, pc.head_uv.y);
    vec2 left = vec2(pc.left_uv.x * aspect, pc.left_uv.y);
    vec2 right = vec2(pc.right_uv.x * aspect, pc.right_uv.y);

    float t = pc.time * pc.speed;

    // March along the flow, accumulating ridged fbm into streaks. The
    // hand swirls bend the march direction before normalisation, so they
    // reshape the streaks rather than just brightening them.
    vec2 mp = p;
    float acc = 0.0;
    float total = 0.0;
    float w = 1.0;
    for (int i = 0; i < 8; i++) {
        vec2 dir = flow_dir(mp * pc.scale, t);
        dir += (swirl(mp, left) + swirl(mp, right)) * (pc.reactivity * 3.0);
        float len = max(length(dir), 1e-4);
        mp += (dir / len) * 0.035;
        float n = fbm4(mp * pc.scale + vec2(0.0, -t * 0.4));
        float ridge = 1.0 - abs(2.0 * n - 1.0);
        acc += ridge * ridge * w;
        total += w;
        w *= 0.9;
    }
    float field = acc / total;

    // Radial wave emanating from the head.
    float dh = length(p - head);
    float wave = sin(dh * 14.0 - t * 2.2);
    field *= 1.0 + wave * 0.3 * pc.reactivity;

    // Mouth-open pulse.
    field *= 1.0 + pc.mouth_open * 0.8 * pc.reactivity;

    // Palette + HDR ridge highlights. Squaring the field keeps most of
    // the frame near the dark base colour so the avatar stays readable,
    // with the flow showing up as brighter filaments. The highlight term
    // pushes peaks past 1.0 so the bloom chain catches them at its
    // default threshold. Linear output — the composite pass owns the
    // transfer curve.
    float shaped = field * field;
    vec3 col = mix(pc.color_a.rgb, pc.color_b.rgb, smoothstep(0.2, 0.8, shaped));
    float hi = smoothstep(0.55, 0.95, shaped);
    col += pc.color_b.rgb * hi * hi * 2.5;
    out_color = vec4(col * pc.intensity, 1.0);
}
"
    }
}

/// Push-constant mirror of the shader's `Push` block (96 bytes, under the
/// 128-byte guaranteed minimum).
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(super) struct BackgroundPushConstants {
    pub color_a: [f32; 4],
    pub color_b: [f32; 4],
    pub head_uv: [f32; 2],
    pub left_uv: [f32; 2],
    pub right_uv: [f32; 2],
    pub resolution: [f32; 2],
    pub time: f32,
    pub intensity: f32,
    pub speed: f32,
    pub scale: f32,
    pub reactivity: f32,
    pub mouth_open: f32,
    pub _pad: [f32; 2],
}

/// Assemble the per-frame push constants: project the tracking anchors to
/// screen UV and gate the reactive terms.
///
/// `reactivity` is scaled by `avatar_opacity` so a tracking loss fades the
/// field back to its neutral flow over the same 0.6 s ramp the avatar fades
/// with, and zeroed entirely when no avatar/humanoid is loaded. Anchors
/// that fail to project (behind the near plane — effectively impossible
/// with the orbital camera) fall back to screen centre rather than killing
/// the whole reactive layer.
pub(super) fn build_push_constants(input: &RenderFrameInput) -> BackgroundPushConstants {
    const CENTER: [f32; 2] = [0.5, 0.5];
    let settings = &input.generative_background;
    let tracking = &input.background_tracking;
    let extent = input.output_request.extent;
    let view = &input.camera.view;
    let proj = &input.camera.projection;

    let project =
        |p: [f32; 3]| super::project_world_to_uv(view, proj, p).unwrap_or(CENTER);

    let reactivity = if tracking.valid {
        settings.reactivity * input.avatar_opacity.clamp(0.0, 1.0)
    } else {
        0.0
    };

    BackgroundPushConstants {
        color_a: [
            settings.color_a[0],
            settings.color_a[1],
            settings.color_a[2],
            0.0,
        ],
        color_b: [
            settings.color_b[0],
            settings.color_b[1],
            settings.color_b[2],
            0.0,
        ],
        head_uv: project(tracking.head_ws),
        left_uv: project(tracking.left_hand_ws),
        right_uv: project(tracking.right_hand_ws),
        resolution: [extent[0] as f32, extent[1] as f32],
        time: input.time_seconds,
        intensity: settings.intensity.max(0.0),
        speed: settings.speed,
        scale: settings.scale.max(0.01),
        reactivity,
        mouth_open: tracking.mouth_open.clamp(0.0, 1.0),
        _pad: [0.0; 2],
    }
}

/// Build the background pipeline against the scene render pass (subpass 0).
/// Viewport and MSAA sample count are baked like the scene pipelines, so
/// this is rebuilt by `rebuild_pipelines_and_targets` alongside them.
pub(super) fn create_background_pipeline(
    device: Arc<Device>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
    sample_count: u32,
) -> Result<Arc<GraphicsPipeline>, String> {
    let vs_module = post_fullscreen_vs::load(device.clone())
        .map_err(|e| format!("background: failed to load fullscreen vertex shader: {e}"))?;
    let vs_entry = vs_module
        .entry_point("main")
        .ok_or_else(|| "background: fullscreen vertex shader entry point not found".to_string())?;
    let fs_module = background_fs::load(device.clone())
        .map_err(|e| format!("background: failed to load fragment shader: {e}"))?;
    let fs_entry = fs_module
        .entry_point("main")
        .ok_or_else(|| "background: fragment shader entry point not found".to_string())?;

    let stages = [
        PipelineShaderStageCreateInfo::new(vs_entry),
        PipelineShaderStageCreateInfo::new(fs_entry),
    ];

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| format!("background: failed to create pipeline layout info: {e}"))?,
    )
    .map_err(|e| format!("background: failed to create pipeline layout: {e}"))?;

    let subpass = Subpass::from(render_pass, 0)
        .ok_or_else(|| "background: failed to get subpass from render pass".to_string())?;

    GraphicsPipeline::new(
        device,
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(VertexInputState::default()),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState {
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState {
                rasterization_samples: SampleCount::try_from(sample_count)
                    .unwrap_or(SampleCount::Sample1),
                // No alpha-to-coverage: the background always writes
                // alpha = 1.0 and A2C would serve no purpose.
                ..MultisampleState::default()
            }),
            // The subpass has a depth/stencil attachment, so the state must
            // be present (VUID-06590) — but default() disables depth test,
            // depth write and stencil entirely, which is exactly what the
            // painter's-order background wants.
            depth_stencil_state: Some(DepthStencilState::default()),
            color_blend_state: Some(ColorBlendState {
                attachments: vec![ColorBlendAttachmentState::default()],
                ..Default::default()
            }),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .map_err(|e| format!("background: failed to create background pipeline: {e}"))
}

/// Record the background draw. Must be called inside the scene render pass,
/// before any avatar draw. The caller computes the push constants (UV
/// projection + reactivity scaling happen in `render()`).
pub(super) fn record_background(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    pipeline: &Arc<GraphicsPipeline>,
    push: BackgroundPushConstants,
) -> Result<(), String> {
    builder
        .bind_pipeline_graphics(pipeline.clone())
        .map_err(|e| format!("background: bind_pipeline failed: {e}"))?
        .push_constants(pipeline.layout().clone(), 0, push)
        .map_err(|e| format!("background: push_constants failed: {e}"))?;
    unsafe {
        builder
            .draw(3, 1, 0, 0)
            .map_err(|e| format!("background: draw failed: {e}"))?;
    }
    Ok(())
}
