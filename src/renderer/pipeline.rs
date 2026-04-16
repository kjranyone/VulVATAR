#![allow(dead_code)]
use crate::renderer::frame_input::{RenderAlphaMode, RenderCullMode};
use crate::renderer::material::MaterialShaderMode;
use std::sync::Arc;
use vulkano::device::Device;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::depth_stencil::{
    CompareOp, DepthState, DepthStencilState, StencilOp, StencilOpState, StencilState,
};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::{RenderPass, Subpass};

// ---------------------------------------------------------------------------
// Vertex type used by the GPU pipeline
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, Vertex, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
    #[format(R32G32B32A32_UINT)]
    pub joint_indices: [u32; 4],
    #[format(R32G32B32A32_SFLOAT)]
    pub joint_weights: [f32; 4],
}

// ---------------------------------------------------------------------------
// Shader modules
// ---------------------------------------------------------------------------

pub mod vs {
    vulkano_shaders::shader! {
                                                                                                                                                                                                                                                                        ty: "vertex",
                                                                                                                                                                                                                                                                        src: r"
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in uvec4 joint_indices;
layout(location = 4) in vec4 joint_weights;

layout(set = 0, binding = 0) uniform CameraData {
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
    float _pad0;
    vec3 light_dir;
    float light_intensity;
    vec3 light_color;
    float _pad1;
    vec3 ambient_term;
    float _pad2;
} camera;

layout(set = 1, binding = 0) readonly buffer SkinningData {
    mat4 matrices[];
} skinning;

layout(location = 0) out vec3 frag_normal;
layout(location = 1) out vec2 frag_uv;
layout(location = 2) out vec3 frag_world_pos;

void main() {
    // Apply skinning
    mat4 skin_mat = mat4(0.0);
    for (int i = 0; i < 4; i++) {
        skin_mat += joint_weights[i] * skinning.matrices[joint_indices[i]];
    }

    // If all weights are zero, use identity
    float total_weight = joint_weights.x + joint_weights.y + joint_weights.z + joint_weights.w;
    if (total_weight < 0.001) {
        skin_mat = mat4(1.0);
    }

    vec4 world_pos = skin_mat * vec4(position, 1.0);
    frag_world_pos = world_pos.xyz;
    frag_normal = mat3(skin_mat) * normal;
    frag_uv = uv;

    gl_Position = camera.proj * camera.view * world_pos;
}
"
                                                                                                                                                                                                                                                                    }
}

pub mod fs {
    vulkano_shaders::shader! {
                                                                                                                                                                                                                                                                        ty: "fragment",
                                                                                                                                                                                                                                                                        src: r"
#version 450

layout(location = 0) in vec3 frag_normal;
layout(location = 1) in vec2 frag_uv;
layout(location = 2) in vec3 frag_world_pos;

layout(set = 0, binding = 0) uniform CameraData {
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
    float _pad0;
    vec3 light_dir;
    float light_intensity;
    vec3 light_color;
    float _pad1;
    vec3 ambient_term;
    float _pad2;
} camera;

layout(set = 2, binding = 0) uniform MaterialData {
    vec4 base_color;
    float alpha_cutoff;
    int alpha_mode;
    float shade_shift;
    float shade_toony;
    int shading_mode;
    int debug_view;
    ivec2 _pad_debug0;
    vec4 shade_color;
    vec4 emissive_color;
    vec4 rim_color;
    float rim_fresnel_power;
    float rim_lift;
    float rim_lighting_mix;
    float uv_anim_scroll_x;
    float uv_anim_scroll_y;
    float uv_anim_rotation;
    float matcap_blend;
} material;

layout(set = 2, binding = 1) uniform sampler2D base_texture;
layout(set = 2, binding = 2) uniform sampler2D matcap_texture;
layout(set = 2, binding = 3) uniform sampler2D shade_texture;

layout(location = 0) out vec4 out_color;

float linearstep(float a, float b, float t) {
    float denom = b - a;
    if (abs(denom) < 0.00001) {
        return t >= b ? 1.0 : 0.0;
    }
    return clamp((t - a) / denom, 0.0, 1.0);
}

void main() {
    vec2 anim_uv = frag_uv;
    anim_uv.x += material.uv_anim_scroll_x;
    anim_uv.y += material.uv_anim_scroll_y;
    if (abs(material.uv_anim_rotation) > 0.0001) {
        float s = sin(material.uv_anim_rotation);
        float c = cos(material.uv_anim_rotation);
        anim_uv = vec2(anim_uv.x * c - anim_uv.y * s,
                       anim_uv.x * s + anim_uv.y * c);
    }

    vec4 tex_color = texture(base_texture, anim_uv);
    vec4 color = material.base_color * tex_color;
    vec3 base_color_term = color.rgb;
    vec3 shade_texture_term = texture(shade_texture, anim_uv).rgb;

    if (material.alpha_mode == 1 && color.a < material.alpha_cutoff) {
        discard;
    }

    if (material.debug_view == 1) {
        out_color = vec4(fract(frag_uv), 0.0, 1.0);
        return;
    }
    if (material.debug_view == 2) {
        out_color = vec4(tex_color.rgb, 1.0);
        return;
    }

    vec3 n = normalize(frag_normal);
    if (!gl_FrontFacing) n = -n;
    vec3 l = normalize(camera.light_dir);
    float ndotl = dot(n, l);

    if (material.shading_mode == 0) {
        // Unlit: no lighting at all
    } else if (material.shading_mode == 1) {
        // Approximate TinyMToon: mix base/shade color using linearstep over N.L + shift.
        // Use ambient to raise the floor of the N.L term so shadowed faces
        // don't go fully dark, then multiply by light color once.
        float ambient_avg = (camera.ambient_term.x + camera.ambient_term.y + camera.ambient_term.z) / 3.0;
        float lit = ndotl + material.shade_shift + ambient_avg;
        float shading = linearstep(
            -1.0 + clamp(material.shade_toony, 0.0, 1.0),
            1.0 - clamp(material.shade_toony, 0.0, 1.0),
            lit
        );
        vec3 shade_col = material.shade_color.rgb * shade_texture_term;
        vec3 toon_col = mix(shade_col, base_color_term, shading);
        color.rgb = toon_col * (camera.light_color * max(camera.light_intensity, 0.0));
    } else {
        // SimpleLit: standard diffuse
        float diffuse = max(ndotl, 0.0);
        vec3 lighting = camera.ambient_term + camera.light_color * (camera.light_intensity * diffuse);
        color.rgb = base_color_term * lighting;
    }

    color.rgb += material.emissive_color.rgb * material.emissive_color.a;

    vec3 view_dir = normalize(camera.camera_pos - frag_world_pos);
    float rim_dot = 1.0 - max(dot(view_dir, n), 0.0);
    float rim_factor = pow(rim_dot, max(material.rim_fresnel_power, 0.01));
    rim_factor = rim_factor + material.rim_lift;
    rim_factor = clamp(rim_factor, 0.0, 1.0);
    vec3 rim_light = material.rim_color.rgb;
    color.rgb += rim_light * rim_factor * material.rim_color.a;

    if (material.matcap_blend > 0.001) {
        vec3 view_normal = normalize(mat3(camera.view) * n);
        vec2 matcap_uv = view_normal.xy * 0.5 + 0.5;
        vec4 matcap_sample = texture(matcap_texture, matcap_uv);
        color.rgb = mix(color.rgb, color.rgb * matcap_sample.rgb, material.matcap_blend);
    }

    if (material.alpha_mode == 0) {
        color.a = 1.0;
    }

    out_color = color;
}
"
                                                                                                                                                                                                                                                                    }
}

// ---------------------------------------------------------------------------
// Outline shaders
// ---------------------------------------------------------------------------

pub mod outline_vs {
    vulkano_shaders::shader! {
                                                                                                                                                                                                                                                                        ty: "vertex",
                                                                                                                                                                                                                                                                        src: r"
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in uvec4 joint_indices;
layout(location = 4) in vec4 joint_weights;

layout(set = 0, binding = 0) uniform CameraData {
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
    float _pad0;
    vec3 light_dir;
    float light_intensity;
    vec3 light_color;
    float _pad1;
    vec3 ambient_term;
    float _pad2;
} camera;

layout(set = 1, binding = 0) readonly buffer SkinningData {
    mat4 matrices[];
} skinning;

layout(push_constant) uniform OutlinePush {
    float outline_width;
    float r;
    float g;
    float b;
    float a;
} outline;

void main() {
    // Apply skinning
    mat4 skin_mat = mat4(0.0);
    for (int i = 0; i < 4; i++) {
        skin_mat += joint_weights[i] * skinning.matrices[joint_indices[i]];
    }

    float total_weight = joint_weights.x + joint_weights.y + joint_weights.z + joint_weights.w;
    if (total_weight < 0.001) {
        skin_mat = mat4(1.0);
    }

    vec4 world_pos = skin_mat * vec4(position, 1.0);
    vec3 world_normal = normalize(mat3(skin_mat) * normal);

    vec4 clip_pos = camera.proj * camera.view * world_pos;

    // Offset along normal in clip space for screen-space-like outline width
    vec4 clip_normal = camera.proj * camera.view * vec4(world_normal, 0.0);
    vec2 clip_normal_xy = clip_normal.xy;
    float clip_normal_len = length(clip_normal_xy);
    vec2 screen_normal = clip_normal_len > 0.001 ? clip_normal_xy / clip_normal_len : vec2(0.0, 1.0);
    clip_pos.xy += screen_normal * outline.outline_width * clip_pos.w * 0.01;

    gl_Position = clip_pos;
}
"
                                                                                                                                                                                                                                                                    }
}

pub mod outline_fs {
    vulkano_shaders::shader! {
                                                                                                                                                                                                                                                                        ty: "fragment",
                                                                                                                                                                                                                                                                        src: r"
#version 450

layout(push_constant) uniform OutlinePush {
    float outline_width;
    float r;
    float g;
    float b;
    float a;
} outline;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(outline.r, outline.g, outline.b, outline.a);
}
"
                                                                                                                                                                                                                                                                    }
}

pub mod depth_only_vs {
    vulkano_shaders::shader! {
                                                                                                                                                                                                                                                                        ty: "vertex",
                                                                                                                                                                                                                                                                        src: r"
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in uvec4 joint_indices;
layout(location = 4) in vec4 joint_weights;

layout(set = 0, binding = 0) uniform CameraData {
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
    float _pad0;
    vec3 light_dir;
    float light_intensity;
    vec3 light_color;
    float _pad1;
    vec3 ambient_term;
    float _pad2;
} camera;

layout(set = 1, binding = 0) readonly buffer SkinningData {
    mat4 matrices[];
} skinning;

void main() {
    mat4 skin_mat = mat4(0.0);
    for (int i = 0; i < 4; i++) {
        skin_mat += joint_weights[i] * skinning.matrices[joint_indices[i]];
    }
    float total_weight = joint_weights.x + joint_weights.y + joint_weights.z + joint_weights.w;
    if (total_weight < 0.001) {
        skin_mat = mat4(1.0);
    }
    vec4 world_pos = skin_mat * vec4(position, 1.0);
    gl_Position = camera.proj * camera.view * world_pos;
}
"
                                                                                                                                                                                                                                                                    }
}

pub mod depth_only_fs {
    vulkano_shaders::shader! {
                                                                                                                                                                                                                                                                        ty: "fragment",
                                                                                                                                                                                                                                                                        src: r"
#version 450

void main() {
}
"
                                                                                                                                                                                                                                                                    }
}

// ---------------------------------------------------------------------------
// Pipeline enums (kept from original)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum RenderPipeline {
    SkinningUnlit,
    SkinningSimpleLit,
    SkinningToon,
    Outline,
    DepthOnly,
}

pub struct PipelineState {
    pub active_pipeline: RenderPipeline,
    pub initialized: bool,
    pub graphics_pipeline: Option<Arc<GraphicsPipeline>>,
}

impl PipelineState {
    pub fn new(pipeline: RenderPipeline) -> Self {
        Self {
            active_pipeline: pipeline,
            initialized: false,
            graphics_pipeline: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    pub material_mode: MaterialShaderMode,
    pub alpha_mode: RenderAlphaMode,
    pub outline_enabled: bool,
    pub cull_mode: RenderCullMode,
    pub vertex_layout: VertexLayout,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum VertexLayout {
    Skinned,
    Static,
}

impl Default for VertexLayout {
    fn default() -> Self {
        Self::Skinned
    }
}

impl PipelineKey {
    pub fn select_pipeline(&self) -> RenderPipeline {
        if self.outline_enabled {
            return RenderPipeline::Outline;
        }
        match self.material_mode {
            MaterialShaderMode::Unlit => RenderPipeline::SkinningUnlit,
            MaterialShaderMode::SimpleLit => RenderPipeline::SkinningSimpleLit,
            MaterialShaderMode::ToonLike => RenderPipeline::SkinningToon,
        }
    }

    pub fn from_mesh_instance(
        material_mode: MaterialShaderMode,
        alpha_mode: RenderAlphaMode,
        cull_mode: RenderCullMode,
        outline_enabled: bool,
        has_skin: bool,
    ) -> Self {
        Self {
            material_mode,
            alpha_mode,
            outline_enabled,
            cull_mode,
            vertex_layout: if has_skin {
                VertexLayout::Skinned
            } else {
                VertexLayout::Static
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

pub fn create_graphics_pipeline(
    device: Arc<Device>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
    cull_mode: CullMode,
    alpha_mode: RenderAlphaMode,
) -> Result<Arc<GraphicsPipeline>, String> {
    let vs_module = vs::load(device.clone())
        .map_err(|e| format!("failed to load vertex shader module: {e}"))?;
    let fs_module = fs::load(device.clone())
        .map_err(|e| format!("failed to load fragment shader module: {e}"))?;

    let vs_entry = vs_module
        .entry_point("main")
        .ok_or_else(|| "vertex shader entry point 'main' not found".to_string())?;
    let fs_entry = fs_module
        .entry_point("main")
        .ok_or_else(|| "fragment shader entry point 'main' not found".to_string())?;

    let vertex_input_state = GpuVertex::per_vertex()
        .definition(&vs_entry.info().input_interface)
        .map_err(|e| format!("failed to get vertex input state: {e}"))?;

    let stages = [
        PipelineShaderStageCreateInfo::new(vs_entry),
        PipelineShaderStageCreateInfo::new(fs_entry),
    ];

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| format!("failed to create pipeline layout info: {e}"))?,
    )
    .map_err(|e| format!("failed to create pipeline layout: {e}"))?;

    let subpass = Subpass::from(render_pass.clone(), 0)
        .ok_or_else(|| "failed to get subpass from render pass".to_string())?;

    let (depth_write_enable, blend_state) = match alpha_mode {
        RenderAlphaMode::Blend => (false, Some(AttachmentBlend::alpha())),
        RenderAlphaMode::Opaque | RenderAlphaMode::Cutout => (true, None),
    };

    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState {
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState {
                cull_mode,
                front_face: FrontFace::Clockwise,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState {
                    write_enable: depth_write_enable,
                    compare_op: CompareOp::LessOrEqual,
                }),
                stencil: Some(StencilState {
                    front: StencilOpState {
                        ops: vulkano::pipeline::graphics::depth_stencil::StencilOps {
                            pass_op: StencilOp::Replace,
                            fail_op: StencilOp::Keep,
                            depth_fail_op: StencilOp::Keep,
                            compare_op: CompareOp::Always,
                        },
                        compare_mask: 0xFF,
                        write_mask: 0xFF,
                        reference: 1,
                    },
                    back: StencilOpState {
                        ops: vulkano::pipeline::graphics::depth_stencil::StencilOps {
                            pass_op: StencilOp::Replace,
                            fail_op: StencilOp::Keep,
                            depth_fail_op: StencilOp::Keep,
                            compare_op: CompareOp::Always,
                        },
                        compare_mask: 0xFF,
                        write_mask: 0xFF,
                        reference: 1,
                    },
                }),
                ..Default::default()
            }),
            color_blend_state: Some(ColorBlendState {
                attachments: vec![ColorBlendAttachmentState {
                    blend: blend_state,
                    ..Default::default()
                }],
                ..Default::default()
            }),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .map_err(|e| format!("failed to create graphics pipeline with cull mode: {e}"))
}

/// Create the outline graphics pipeline (front-face culling, no depth write).
pub fn create_outline_pipeline(
    device: Arc<Device>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Result<Arc<GraphicsPipeline>, String> {
    let vs_module = outline_vs::load(device.clone())
        .map_err(|e| format!("failed to load outline vertex shader module: {e}"))?;
    let fs_module = outline_fs::load(device.clone())
        .map_err(|e| format!("failed to load outline fragment shader module: {e}"))?;

    let vs_entry = vs_module
        .entry_point("main")
        .ok_or_else(|| "outline vertex shader entry point 'main' not found".to_string())?;
    let fs_entry = fs_module
        .entry_point("main")
        .ok_or_else(|| "outline fragment shader entry point 'main' not found".to_string())?;

    let vertex_input_state = GpuVertex::per_vertex()
        .definition(&vs_entry.info().input_interface)
        .map_err(|e| format!("failed to get outline vertex input state: {e}"))?;

    let stages = [
        PipelineShaderStageCreateInfo::new(vs_entry),
        PipelineShaderStageCreateInfo::new(fs_entry),
    ];

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| format!("failed to create outline pipeline layout info: {e}"))?,
    )
    .map_err(|e| format!("failed to create outline pipeline layout: {e}"))?;

    let subpass = Subpass::from(render_pass.clone(), 0)
        .ok_or_else(|| "failed to get subpass from render pass for outline pipeline".to_string())?;

    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState {
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::Front,
                front_face: FrontFace::Clockwise,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState {
                    write_enable: false,
                    compare_op: CompareOp::LessOrEqual,
                }),
                stencil: Some(StencilState {
                    front: StencilOpState {
                        ops: vulkano::pipeline::graphics::depth_stencil::StencilOps {
                            pass_op: StencilOp::Keep,
                            fail_op: StencilOp::Keep,
                            depth_fail_op: StencilOp::Keep,
                            compare_op: CompareOp::NotEqual,
                        },
                        compare_mask: 0xFF,
                        write_mask: 0x00,
                        reference: 1,
                    },
                    back: StencilOpState {
                        ops: vulkano::pipeline::graphics::depth_stencil::StencilOps {
                            pass_op: StencilOp::Keep,
                            fail_op: StencilOp::Keep,
                            depth_fail_op: StencilOp::Keep,
                            compare_op: CompareOp::NotEqual,
                        },
                        compare_mask: 0xFF,
                        write_mask: 0x00,
                        reference: 1,
                    },
                }),
                ..Default::default()
            }),
            color_blend_state: Some(ColorBlendState {
                attachments: vec![ColorBlendAttachmentState {
                    blend: Some(AttachmentBlend::alpha()),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .map_err(|e| format!("failed to create outline graphics pipeline: {e}"))
}

pub fn create_depth_only_pipeline(
    device: Arc<Device>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Result<Arc<GraphicsPipeline>, String> {
    let vs_module = depth_only_vs::load(device.clone())
        .map_err(|e| format!("failed to load depth-only vertex shader: {e}"))?;
    let fs_module = depth_only_fs::load(device.clone())
        .map_err(|e| format!("failed to load depth-only fragment shader: {e}"))?;

    let vs_entry = vs_module
        .entry_point("main")
        .ok_or_else(|| "depth-only vertex shader entry point 'main' not found".to_string())?;
    let fs_entry = fs_module
        .entry_point("main")
        .ok_or_else(|| "depth-only fragment shader entry point 'main' not found".to_string())?;

    let vertex_input_state = GpuVertex::per_vertex()
        .definition(&vs_entry.info().input_interface)
        .map_err(|e| format!("failed to get depth-only vertex input state: {e}"))?;

    let stages = [
        PipelineShaderStageCreateInfo::new(vs_entry),
        PipelineShaderStageCreateInfo::new(fs_entry),
    ];

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| format!("failed to create depth-only pipeline layout info: {e}"))?,
    )
    .map_err(|e| format!("failed to create depth-only pipeline layout: {e}"))?;

    let subpass = Subpass::from(render_pass.clone(), 0).ok_or_else(|| {
        "failed to get subpass from render pass for depth-only pipeline".to_string()
    })?;

    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState {
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::Back,
                front_face: FrontFace::Clockwise,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState {
                    write_enable: true,
                    compare_op: CompareOp::Less,
                }),
                ..Default::default()
            }),
            color_blend_state: Some(ColorBlendState {
                attachments: vec![ColorBlendAttachmentState {
                    blend: None,
                    ..Default::default()
                }],
                ..Default::default()
            }),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .map_err(|e| format!("failed to create depth-only pipeline: {e}"))
}
