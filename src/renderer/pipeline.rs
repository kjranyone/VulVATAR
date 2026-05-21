use crate::renderer::frame_input::{RenderAlphaMode, RenderCullMode};
use crate::renderer::material::MaterialShaderMode;
use std::sync::Arc;
use vulkano::device::Device;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
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
use vulkano::pipeline::{
    ComputePipeline, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{RenderPass, Subpass};

// ---------------------------------------------------------------------------
// Vertex / compute resource types
// ---------------------------------------------------------------------------
//
// The renderer runs a single compute prepass per frame that fuses skinning,
// morph-target blending, and CPU-computed cloth deformation. The output is
// `GpuVertex` — already world-space and ready for graphics consumption. The
// forward / outline vertex shaders just transform that by view / projection.

/// Compute-stage *input* vertex layout, std430-friendly. Used inside the
/// per-primitive `BaseVertices` storage buffer that the transform compute
/// shader reads. `vec3` fields are padded to `vec4` because std430's array
/// layout treats them as 16-byte aligned. Joint indices / weights stay
/// because skinning happens inside the compute prepass, not in the
/// graphics vertex shader.
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuVertexBase {
    pub position: [f32; 4],
    pub normal: [f32; 4],
    pub uv: [f32; 2],
    pub _pad0: [u32; 2],
    pub joint_indices: [u32; 4],
    pub joint_weights: [f32; 4],
}

/// Graphics vertex layout. The compute prepass writes world-space
/// `(position.xyz, normal.xyz, uv)` per vertex into this buffer; the
/// vertex shaders only apply the camera view + projection. The trailing
/// `uvec2 _pad` keeps the layout 48 B and 16-byte aligned so std430 and
/// the vertex-input definition agree byte-for-byte with the compute
/// shader's `OutVertex` struct.
#[derive(Clone, Copy, Debug, Default, Vertex, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuVertex {
    #[format(R32G32B32A32_SFLOAT)]
    pub position: [f32; 4],
    #[format(R32G32B32A32_SFLOAT)]
    pub normal: [f32; 4],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
    #[format(R32G32_UINT)]
    pub _pad: [u32; 2],
}

/// Per-primitive control UBO consumed by the transform compute shader.
/// `weights` is packed as `vec4[64]` (up to 256 morph targets); excess
/// targets are clamped at upload time with a warning. The `has_cloth` /
/// `has_cloth_normals` flags select between the per-primitive cloth SSBO
/// (filled in-place by the CPU solver each frame) and the morph + base
/// fallback path.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct TransformControl {
    pub vertex_count: u32,
    pub target_count: u32,
    pub has_cloth: u32,
    pub has_cloth_normals: u32,
    pub weights: [[f32; 4]; 64],
}

impl TransformControl {
    pub fn zeroed() -> Self {
        Self {
            vertex_count: 0,
            target_count: 0,
            has_cloth: 0,
            has_cloth_normals: 0,
            weights: [[0.0; 4]; 64],
        }
    }
}

/// Pack `VertexData` into the std430-aligned `GpuVertexBase` form the
/// transform compute shader consumes. `vec3` lanes are padded to `vec4`
/// so the shader-side struct matches the Rust struct byte-for-byte.
pub fn vertex_data_to_base(vd: &crate::asset::VertexData) -> Vec<GpuVertexBase> {
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
            GpuVertexBase {
                position: [pos[0], pos[1], pos[2], 1.0],
                normal: [norm[0], norm[1], norm[2], 0.0],
                uv,
                _pad0: [0, 0],
                joint_indices: ji,
                joint_weights: jw,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Shader modules
// ---------------------------------------------------------------------------
//
// Gamma policy (Stage 3 audit, output color space):
//   All fragment shaders below write **linear** values to `out_color`.
//   - With an sRGB-format colour attachment (the default `Srgb` output mode)
//     the GPU applies the linear→sRGB transfer curve on store automatically.
//   - With a UNORM-format colour attachment (the `LinearSrgb` output mode)
//     the GPU stores the linear value verbatim — that is what users opting
//     into a linear pipeline want.
//
//   Texture sampling: VRM textures are uploaded with `R8G8B8A8_SRGB`, so
//   `texture(...)` reads return linear values regardless of the output
//   target format. No manual `pow(x, 2.2)` decode is required in any
//   fragment shader.
//
//   The outline fragment shader writes raw push-constant RGB; those values
//   originate from egui's RGB colour picker, which is a pre-existing
//   colour-management caveat unrelated to the output colour-space switch.

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
#version 450

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in uvec2 _pad;

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

layout(location = 0) out vec3 frag_normal;
layout(location = 1) out vec2 frag_uv;
layout(location = 2) out vec3 frag_world_pos;

void main() {
    frag_world_pos = position.xyz;
    frag_normal = normal.xyz;
    frag_uv = uv;
    gl_Position = camera.proj * camera.view * vec4(position.xyz, 1.0);
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

layout(set = 1, binding = 0) uniform MaterialData {
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

layout(set = 1, binding = 1) uniform sampler2D base_texture;
layout(set = 1, binding = 2) uniform sampler2D matcap_texture;
layout(set = 1, binding = 3) uniform sampler2D shade_texture;

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
        // MToon shade: shade_color modulates the base texture, not replaces it.
        // When no dedicated shade texture exists, shade_texture_term is white,
        // so shade_col = shade_color * base_texture_color.
        vec3 shade_col = material.shade_color.rgb * shade_texture_term * tex_color.rgb;
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

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in uvec2 _pad;

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

layout(push_constant) uniform OutlinePush {
    float outline_width;
    float r;
    float g;
    float b;
    float a;
} outline;

void main() {
    vec3 world_normal = normalize(normal.xyz);
    vec4 clip_pos = camera.proj * camera.view * vec4(position.xyz, 1.0);
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

// ---------------------------------------------------------------------------
// Transform compute shader (skinning + morph + cloth fuse)
//
// Dispatched once per (instance, primitive) per frame from the renderer's
// `render()` / `render_to_thumbnail()` paths. Reads per-vertex base data +
// per-frame morph weights + per-frame cloth snapshots and writes
// world-space `GpuVertex` records that the graphics pipelines consume as
// their vertex buffer. The graphics `vs` / `outline_vs` above only apply
// the camera view + projection on top.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cloth Verlet integration compute shader (P3-02 S1.2 — pipeline only)
// ---------------------------------------------------------------------------
//
// First stage of the GPU cloth solver migration: integrates particle
// positions under gravity / wind / damping using Verlet (pos + prev_pos).
// Mirrors `simulation::cloth_solver::integrator::verlet_integrate` formula
// bit-for-bit so the CPU PBD tests double as parity oracles for the eventual
// GPU side-by-side test (P3-02 S1.3).
//
// **Status**: shader + pipeline factory only. Not yet wired into the render
// loop; that lands when `ClothGpuSimulationState` grows actual Vulkano
// buffer handles (P3-02 S1.1 Vulkan side).
//
// SSBO + UBO layout matches the slot table documented in
// `simulation::cloth_gpu_boundary::ClothGpuSimulationState`.
pub mod cloth_verlet_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// xyz = position, w = inv_mass (0.0 marks an effectively immobile particle).
layout(set = 0, binding = 0) buffer Positions {
    vec4 p[];
} positions;

// xyz = previous position, w = pinned flag (>= 0.5 means pinned).
layout(set = 0, binding = 1) buffer PreviousPositions {
    vec4 p[];
} prev_positions;

// std140-friendly control block. Rust mirror: `ClothVerletControl`.
layout(set = 0, binding = 2) uniform Control {
    float dt;
    float damping;
    uint  particle_count;
    uint  _pad0;
    vec4  gravity;  // xyz = gravity vector
    vec4  wind;     // xyz = wind_direction * wind_response
} ctrl;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= ctrl.particle_count) return;

    vec4 cur = positions.p[idx];
    vec4 prv = prev_positions.p[idx];

    float inv_mass = cur.w;
    float pinned   = prv.w;

    // Pinned / immobile particles skip integration. Matches CPU
    // `if p.pinned { continue }` in `verlet_integrate`.
    if (pinned > 0.5 || inv_mass <= 0.0) {
        return;
    }

    vec3 pos      = cur.xyz;
    vec3 prev_pos = prv.xyz;

    // CPU reference (cloth_solver/integrator.rs verlet_integrate):
    //   vel        = pos - prev_pos
    //   damped_vel = vel * (1.0 - damping)
    //   accel      = (gravity + wind_force) * dt * dt
    //   new_pos    = pos + damped_vel + accel
    //   prev = pos; pos = new_pos
    vec3 vel        = pos - prev_pos;
    vec3 damped_vel = vel * (1.0 - ctrl.damping);
    float dt2       = ctrl.dt * ctrl.dt;
    vec3 accel      = (ctrl.gravity.xyz + ctrl.wind.xyz) * dt2;
    vec3 new_pos    = pos + damped_vel + accel;

    positions.p[idx]      = vec4(new_pos, inv_mass);
    prev_positions.p[idx] = vec4(pos, pinned);
}
"
    }
}

// ---------------------------------------------------------------------------
// Cloth PBD distance constraint projection — accumulate pass (P3-02 S2.1)
// ---------------------------------------------------------------------------
//
// Jacobi-style 2-pass constraint projection: this first shader runs one
// invocation per particle, walks the constraints touching that particle
// via a precomputed CSR adjacency (`build_particle_constraint_adjacency`),
// and writes per-particle position corrections into `delta_ssbo`. The
// apply pass below adds the deltas to `pos_ssbo` and zeroes
// `delta_ssbo` for the next iteration.
//
// Choosing Jacobi (2 passes per iteration) over Gauss-Seidel-on-GPU
// (atomics or partition coloring) keeps the shader simple — no
// atomicAdd, no `VK_EXT_shader_atomic_float`, no per-constraint colour
// dispatch. Convergence per iteration is slower than Gauss-Seidel, so
// real workloads typically run more iterations to compensate.
pub mod cloth_constraint_accumulate_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// xyz = position, w = inv_mass (0.0 = effectively pinned / immobile).
layout(set = 0, binding = 0) readonly buffer Positions {
    vec4 p[];
} positions;

// Per-particle position correction accumulated this iteration. The
// apply pass adds these to `positions.p[i].xyz` and zeroes the entry.
layout(set = 0, binding = 1) writeonly buffer Deltas {
    vec4 d[];
} deltas;

// Constraint table, matches Rust `ClothConstraintGpu`.
struct Constraint {
    uint  particle_a;
    uint  particle_b;
    float rest_length;
    float stiffness;
};
layout(set = 0, binding = 2) readonly buffer Constraints {
    Constraint c[];
} constraints;

// CSR adjacency: constraints touching vertex v live at
// `adj_constraints.c[adj_offsets.o[v] .. adj_offsets.o[v + 1]]`.
layout(set = 0, binding = 3) readonly buffer AdjacencyOffsets {
    uint o[];
} adj_offsets;
layout(set = 0, binding = 4) readonly buffer AdjacencyConstraints {
    uint c[];
} adj_constraints;

// Rust mirror: `ClothConstraintControl`.
layout(set = 0, binding = 5) uniform Control {
    uint particle_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
} ctrl;

void main() {
    uint pid = gl_GlobalInvocationID.x;
    if (pid >= ctrl.particle_count) return;

    vec4 self_p = positions.p[pid];
    float w_self = self_p.w;

    // Pinned (inv_mass == 0) particles take no correction.
    if (w_self <= 0.0) {
        deltas.d[pid] = vec4(0.0);
        return;
    }

    vec3 self_pos = self_p.xyz;
    vec3 delta = vec3(0.0);

    uint start = adj_offsets.o[pid];
    uint end   = adj_offsets.o[pid + 1u];
    for (uint k = start; k < end; ++k) {
        uint cidx = adj_constraints.c[k];
        Constraint cn = constraints.c[cidx];
        uint other = (cn.particle_a == pid) ? cn.particle_b : cn.particle_a;
        vec4 other_p = positions.p[other];
        vec3 dir = other_p.xyz - self_pos;
        float len = length(dir);
        if (len > 1e-6) {
            float err = len - cn.rest_length;
            float w_other = other_p.w;
            float w_sum = w_self + w_other;
            if (w_sum > 0.0) {
                // Jacobi PBD: pull self toward other proportional to mass
                // ratio, weighted by constraint stiffness. The other
                // particle's invocation does the mirrored correction.
                float scale = cn.stiffness * err / len * (w_self / w_sum);
                delta += dir * scale;
            }
        }
    }
    deltas.d[pid] = vec4(delta, 0.0);
}
"
    }
}

// ---------------------------------------------------------------------------
// Cloth PBD distance constraint projection — apply pass (P3-02 S2.1)
// ---------------------------------------------------------------------------
pub mod cloth_constraint_apply_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Positions {
    vec4 p[];
} positions;

layout(set = 0, binding = 1) buffer Deltas {
    vec4 d[];
} deltas;

layout(set = 0, binding = 2) uniform Control {
    uint particle_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
} ctrl;

void main() {
    uint pid = gl_GlobalInvocationID.x;
    if (pid >= ctrl.particle_count) return;

    vec4 pos = positions.p[pid];
    vec4 d = deltas.d[pid];
    pos.xyz += d.xyz;
    positions.p[pid] = pos;
    // Reset for the next constraint iteration's accumulate pass.
    deltas.d[pid] = vec4(0.0);
}
"
    }
}

// ---------------------------------------------------------------------------
// Cloth vertex normal recomputation compute shader (P3-02 S3.1 — pipeline only)
// ---------------------------------------------------------------------------
//
// Third stage of the GPU cloth solver migration. After integration (S1.2)
// and constraint projection (S2.1) produce updated positions, this shader
// recomputes per-vertex normals so the transform compute pass and the
// graphics pipelines see consistent shading.
//
// Strategy: one workgroup invocation per cloth vertex; each invocation
// walks its incident triangles via a precomputed CSR adjacency built once
// at attach time (`cloth_gpu_boundary::build_vertex_triangle_adjacency`).
// Face normals are accumulated **without** pre-normalising so the sum is
// area-weighted (same convention as `cloth_solver::output::compute_normals`),
// then the per-vertex result is normalised. Embarrassingly parallel — no
// atomics, no extensions, no cross-invocation synchronisation.
//
// **Status**: shader + pipeline factory only. Not yet wired; lands with
// the rest of the GPU cloth dispatch path in P3-02 stage 3.
pub mod cloth_normal_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// xyz = position, w = inv_mass (matches cloth_verlet_cs layout).
layout(set = 0, binding = 0) readonly buffer Positions {
    vec4 p[];
} positions;

// Flat triangle index buffer: 3 uints per triangle.
layout(set = 0, binding = 1) readonly buffer TriangleIndices {
    uint t[];
} indices;

// CSR adjacency: for vertex v, the incident triangle indices are
// `adj_triangles.t[adj_offsets.o[v] .. adj_offsets.o[v + 1]]`.
// Built once at attach time; never changes at runtime.
layout(set = 0, binding = 2) readonly buffer AdjacencyOffsets {
    uint o[];
} adj_offsets;

layout(set = 0, binding = 3) readonly buffer AdjacencyTriangles {
    uint t[];
} adj_triangles;

// Output: per-vertex normal, xyz used (matches cloth_solver::output convention).
layout(set = 0, binding = 4) writeonly buffer Normals {
    vec4 n[];
} normals;

// std140 control block. Rust mirror: `ClothNormalControl`.
layout(set = 0, binding = 5) uniform Control {
    uint vertex_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
} ctrl;

void main() {
    uint vid = gl_GlobalInvocationID.x;
    if (vid >= ctrl.vertex_count) return;

    uint start = adj_offsets.o[vid];
    uint end   = adj_offsets.o[vid + 1u];

    vec3 accum = vec3(0.0);
    for (uint k = start; k < end; ++k) {
        uint tri = adj_triangles.t[k];
        uint i0 = indices.t[tri * 3u + 0u];
        uint i1 = indices.t[tri * 3u + 1u];
        uint i2 = indices.t[tri * 3u + 2u];
        vec3 p0 = positions.p[i0].xyz;
        vec3 p1 = positions.p[i1].xyz;
        vec3 p2 = positions.p[i2].xyz;
        // Unnormalised cross product → area-weighted face normal.
        accum += cross(p1 - p0, p2 - p0);
    }

    float len = length(accum);
    vec3 nrm = (len > 1e-12) ? (accum / len) : vec3(0.0, 1.0, 0.0);
    normals.n[vid] = vec4(nrm, 0.0);
}
"
    }
}

pub mod transform_cs {
    // One workgroup invocation per output vertex. Reads the immutable
    // base SSBO, the per-frame morph weights / cloth deformed positions,
    // and the per-instance skinning matrices; writes the world-space
    // `GpuVertex` array that the graphics pipelines consume as their
    // vertex buffer.
    //
    // Layout invariants (must match the Rust `GpuVertexBase` / `GpuVertex`
    // / `TransformControl` types in this file). std430 places `vec3` on
    // 16-byte alignment, so we pad to `vec4` on both ends — see the Rust
    // struct comments for byte-for-byte breakdown.
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct VertexBase {
    vec4 position;
    vec4 normal;
    vec2 uv;
    uvec2 _pad0;
    uvec4 joint_indices;
    vec4 joint_weights;
};

struct OutVertex {
    vec4 position;
    vec4 normal;
    vec2 uv;
    uvec2 _pad;
};

layout(set = 0, binding = 0) readonly buffer BaseVertices {
    VertexBase v[];
} base;

layout(set = 0, binding = 1) readonly buffer MorphDeltas {
    vec4 data[];
} morph_deltas;

layout(set = 0, binding = 2) readonly buffer ClothPositions {
    vec4 p[];
} cloth_pos;

layout(set = 0, binding = 3) readonly buffer ClothNormals {
    vec4 n[];
} cloth_norm;

layout(set = 0, binding = 4) uniform TransformControl {
    uint vertex_count;
    uint target_count;
    uint has_cloth;
    uint has_cloth_normals;
    vec4 weights[64];
} ctrl;

layout(set = 0, binding = 5) writeonly buffer OutVertices {
    OutVertex v[];
} out_v;

layout(set = 1, binding = 0) readonly buffer SkinningData {
    mat4 matrices[];
} skinning;

void main() {
    uint vid = gl_GlobalInvocationID.x;
    if (vid >= ctrl.vertex_count) return;

    VertexBase b = base.v[vid];
    vec3 pos = b.position.xyz;
    vec3 nrm = b.normal.xyz;

    // Morph target blend (vertex pulling).
    for (uint t = 0u; t < ctrl.target_count; t++) {
        float w = ctrl.weights[t / 4u][t % 4u];
        if (abs(w) < 1e-6) continue;
        uint i = (t * ctrl.vertex_count + vid) * 2u;
        pos += w * morph_deltas.data[i + 0u].xyz;
        nrm += w * morph_deltas.data[i + 1u].xyz;
    }

    // Cloth override: the CPU physics solver wrote per-frame world-relative
    // positions into `cloth_pos` and (optionally) normals into `cloth_norm`.
    // Skinning still applies on top so cloth-bearing primitives can ride
    // along with the avatar's root transform.
    if (ctrl.has_cloth > 0u) {
        pos = cloth_pos.p[vid].xyz;
        if (ctrl.has_cloth_normals > 0u) {
            nrm = cloth_norm.n[vid].xyz;
        }
    }

    // Linear blend skinning. All-zero joint weights → identity (used for
    // primitives without skeletal binding, e.g. accessories).
    mat4 skin = mat4(0.0);
    for (uint i = 0u; i < 4u; i++) {
        skin += b.joint_weights[i] * skinning.matrices[b.joint_indices[i]];
    }
    float total_w = b.joint_weights.x + b.joint_weights.y +
                    b.joint_weights.z + b.joint_weights.w;
    if (total_w < 0.001) skin = mat4(1.0);

    vec4 world_pos = skin * vec4(pos, 1.0);
    vec3 world_nrm = mat3(skin) * nrm;

    out_v.v[vid].position = vec4(world_pos.xyz, 0.0);
    out_v.v[vid].normal   = vec4(world_nrm, 0.0);
    out_v.v[vid].uv       = b.uv;
    out_v.v[vid]._pad     = uvec2(0u, 0u);
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

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum VertexLayout {
    #[default]
    Skinned,
    Static,
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
        .definition(&vs_entry)
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
                front_face: FrontFace::CounterClockwise,
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
        .definition(&vs_entry)
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
                front_face: FrontFace::CounterClockwise,
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

/// Per-frame control block consumed by `cloth_verlet_cs` (P3-02 S1.2).
///
/// std140 layout — field offsets must match the `Control` UBO in
/// `cloth_verlet_cs`. `_pad0` exists so the following `vec4` lands on a
/// 16-byte boundary as std140 requires. `gravity` / `wind` are `[f32; 4]`
/// (xyz used, w unused) rather than `[f32; 3]` because std140 aligns
/// `vec3` to 16 anyway.
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ClothVerletControl {
    pub dt: f32,
    pub damping: f32,
    pub particle_count: u32,
    pub _pad0: u32,
    pub gravity: [f32; 4],
    pub wind: [f32; 4],
}

/// Build the cloth Verlet integration compute pipeline (P3-02 S1.2).
///
/// Not yet called at runtime: the per-primitive cloth particle SSBOs
/// (`pos_ssbo`, `prev_pos_ssbo`) land in P3-02 S1.1 Vulkan side. The
/// factory exists so that landing is a wiring change, not a new design.
pub fn create_cloth_verlet_compute_pipeline(
    device: Arc<Device>,
) -> Result<Arc<ComputePipeline>, String> {
    let cs_module = cloth_verlet_cs::load(device.clone())
        .map_err(|e| format!("failed to load cloth verlet compute shader: {e}"))?;
    let cs_entry = cs_module
        .entry_point("main")
        .ok_or_else(|| "cloth verlet compute shader entry point 'main' not found".to_string())?;
    let stages = [PipelineShaderStageCreateInfo::new(cs_entry)];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| format!("failed to build cloth verlet pipeline layout info: {e}"))?,
    )
    .map_err(|e| format!("failed to create cloth verlet pipeline layout: {e}"))?;
    let stage = stages.into_iter().next().expect("compute stage present");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .map_err(|e| format!("failed to create cloth verlet compute pipeline: {e}"))
}

/// Per-constraint SSBO entry consumed by `cloth_constraint_accumulate_cs`
/// (P3-02 S2.1). std430 layout — 16 bytes per constraint, packed tight.
/// `stiffness` is the [0,1] PBD relaxation factor copied from
/// `ClothDistanceConstraint::stiffness`.
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ClothConstraintGpu {
    pub particle_a: u32,
    pub particle_b: u32,
    pub rest_length: f32,
    pub stiffness: f32,
}

/// Per-frame control block consumed by both constraint shaders.
/// std140 layout — `particle_count` carries the data; three trailing
/// `uint` pad to a 16-byte boundary.
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ClothConstraintControl {
    pub particle_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Build the cloth PBD constraint accumulate compute pipeline
/// (P3-02 S2.1, first pass of the Jacobi-style iteration).
pub fn create_cloth_constraint_accumulate_compute_pipeline(
    device: Arc<Device>,
) -> Result<Arc<ComputePipeline>, String> {
    let cs_module = cloth_constraint_accumulate_cs::load(device.clone())
        .map_err(|e| format!("failed to load cloth constraint accumulate shader: {e}"))?;
    let cs_entry = cs_module.entry_point("main").ok_or_else(|| {
        "cloth constraint accumulate shader entry point 'main' not found".to_string()
    })?;
    let stages = [PipelineShaderStageCreateInfo::new(cs_entry)];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| {
                format!("failed to build cloth constraint accumulate layout info: {e}")
            })?,
    )
    .map_err(|e| format!("failed to create cloth constraint accumulate layout: {e}"))?;
    let stage = stages.into_iter().next().expect("compute stage present");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .map_err(|e| format!("failed to create cloth constraint accumulate pipeline: {e}"))
}

/// Build the cloth PBD constraint apply compute pipeline
/// (P3-02 S2.1, second pass of the Jacobi-style iteration).
pub fn create_cloth_constraint_apply_compute_pipeline(
    device: Arc<Device>,
) -> Result<Arc<ComputePipeline>, String> {
    let cs_module = cloth_constraint_apply_cs::load(device.clone())
        .map_err(|e| format!("failed to load cloth constraint apply shader: {e}"))?;
    let cs_entry = cs_module
        .entry_point("main")
        .ok_or_else(|| "cloth constraint apply shader entry point 'main' not found".to_string())?;
    let stages = [PipelineShaderStageCreateInfo::new(cs_entry)];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| format!("failed to build cloth constraint apply layout info: {e}"))?,
    )
    .map_err(|e| format!("failed to create cloth constraint apply layout: {e}"))?;
    let stage = stages.into_iter().next().expect("compute stage present");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .map_err(|e| format!("failed to create cloth constraint apply pipeline: {e}"))
}

/// Per-frame control block consumed by `cloth_normal_cs` (P3-02 S3.1).
///
/// std140 layout — only `vertex_count` carries data; three trailing `uint`
/// pad to a 16-byte boundary so the UBO meets std140's minimum size /
/// alignment expectations across drivers.
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ClothNormalControl {
    pub vertex_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Build the cloth vertex normal recomputation compute pipeline (P3-02 S3.1).
///
/// Not yet called at runtime. Lands with the rest of the GPU cloth dispatch
/// path once `ClothGpuSimulationState` carries actual Vulkano buffer
/// handles.
pub fn create_cloth_normal_compute_pipeline(
    device: Arc<Device>,
) -> Result<Arc<ComputePipeline>, String> {
    let cs_module = cloth_normal_cs::load(device.clone())
        .map_err(|e| format!("failed to load cloth normal compute shader: {e}"))?;
    let cs_entry = cs_module
        .entry_point("main")
        .ok_or_else(|| "cloth normal compute shader entry point 'main' not found".to_string())?;
    let stages = [PipelineShaderStageCreateInfo::new(cs_entry)];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| format!("failed to build cloth normal pipeline layout info: {e}"))?,
    )
    .map_err(|e| format!("failed to create cloth normal pipeline layout: {e}"))?;
    let stage = stages.into_iter().next().expect("compute stage present");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .map_err(|e| format!("failed to create cloth normal compute pipeline: {e}"))
}

/// Build the compute pipeline that fuses skinning, morph-target blend, and
/// CPU-fed cloth deformation into a single dispatch per (instance, primitive).
/// Output lands in the per-primitive `GpuVertex` SSBO that the graphics
/// pipelines bind as their vertex buffer. See `transform_cs` for the
/// shader-side contract.
pub fn create_transform_compute_pipeline(
    device: Arc<Device>,
) -> Result<Arc<ComputePipeline>, String> {
    let cs_module = transform_cs::load(device.clone())
        .map_err(|e| format!("failed to load transform compute shader: {e}"))?;
    let cs_entry = cs_module
        .entry_point("main")
        .ok_or_else(|| "transform compute shader entry point 'main' not found".to_string())?;
    let stages = [PipelineShaderStageCreateInfo::new(cs_entry)];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| format!("failed to build transform pipeline layout info: {e}"))?,
    )
    .map_err(|e| format!("failed to create transform pipeline layout: {e}"))?;
    let stage = stages.into_iter().next().expect("compute stage present");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .map_err(|e| format!("failed to create transform compute pipeline: {e}"))
}

#[cfg(test)]
mod tests {
    use super::ClothVerletControl;
    use std::mem::{offset_of, size_of};

    // GLSL std140 layout for `Control` UBO in cloth_verlet_cs:
    //   float dt;             // offset 0,  size 4
    //   float damping;        // offset 4,  size 4
    //   uint  particle_count; // offset 8,  size 4
    //   uint  _pad0;          // offset 12, size 4
    //   vec4  gravity;        // offset 16, size 16
    //   vec4  wind;           // offset 32, size 16
    //   total                 // 48 bytes
    #[test]
    fn cloth_verlet_control_matches_std140_layout() {
        assert_eq!(size_of::<ClothVerletControl>(), 48);
        assert_eq!(offset_of!(ClothVerletControl, dt), 0);
        assert_eq!(offset_of!(ClothVerletControl, damping), 4);
        assert_eq!(offset_of!(ClothVerletControl, particle_count), 8);
        assert_eq!(offset_of!(ClothVerletControl, _pad0), 12);
        assert_eq!(offset_of!(ClothVerletControl, gravity), 16);
        assert_eq!(offset_of!(ClothVerletControl, wind), 32);
    }

    // GLSL std140 layout for `Control` UBO in cloth_normal_cs:
    //   uint vertex_count; // offset 0,  size 4
    //   uint _pad0;        // offset 4,  size 4
    //   uint _pad1;        // offset 8,  size 4
    //   uint _pad2;        // offset 12, size 4
    //   total              // 16 bytes
    #[test]
    fn cloth_normal_control_matches_std140_layout() {
        use super::ClothNormalControl;
        assert_eq!(size_of::<ClothNormalControl>(), 16);
        assert_eq!(offset_of!(ClothNormalControl, vertex_count), 0);
        assert_eq!(offset_of!(ClothNormalControl, _pad0), 4);
        assert_eq!(offset_of!(ClothNormalControl, _pad1), 8);
        assert_eq!(offset_of!(ClothNormalControl, _pad2), 12);
    }

    // GLSL std430 layout for the `Constraint` struct in
    // `cloth_constraint_accumulate_cs`:
    //   uint  particle_a;  // offset 0, size 4
    //   uint  particle_b;  // offset 4, size 4
    //   float rest_length; // offset 8, size 4
    //   float stiffness;   // offset 12, size 4
    //   total              // 16 bytes
    #[test]
    fn cloth_constraint_gpu_matches_std430_layout() {
        use super::ClothConstraintGpu;
        assert_eq!(size_of::<ClothConstraintGpu>(), 16);
        assert_eq!(offset_of!(ClothConstraintGpu, particle_a), 0);
        assert_eq!(offset_of!(ClothConstraintGpu, particle_b), 4);
        assert_eq!(offset_of!(ClothConstraintGpu, rest_length), 8);
        assert_eq!(offset_of!(ClothConstraintGpu, stiffness), 12);
    }

    // GLSL std140 layout for the `Control` UBO in both constraint shaders.
    //   uint particle_count;  // offset 0, size 4
    //   uint _pad0;           // offset 4, size 4
    //   uint _pad1;           // offset 8, size 4
    //   uint _pad2;           // offset 12, size 4
    //   total                 // 16 bytes
    #[test]
    fn cloth_constraint_control_matches_std140_layout() {
        use super::ClothConstraintControl;
        assert_eq!(size_of::<ClothConstraintControl>(), 16);
        assert_eq!(offset_of!(ClothConstraintControl, particle_count), 0);
        assert_eq!(offset_of!(ClothConstraintControl, _pad0), 4);
        assert_eq!(offset_of!(ClothConstraintControl, _pad1), 8);
        assert_eq!(offset_of!(ClothConstraintControl, _pad2), 12);
    }
}

