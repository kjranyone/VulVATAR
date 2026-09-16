use crate::renderer::frame_input::{RenderAlphaMode, RenderCullMode};
use crate::renderer::material::MaterialShaderMode;
use std::sync::Arc;
use vulkano::device::Device;
use vulkano::image::SampleCount;
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
/// Morph weights live in a separate SSBO (`MorphWeights`, set 0 binding 9)
/// rather than a fixed-size UBO array, so the per-primitive target count
/// is unbounded — dense-rig FBX avatars carry 400+ blend shapes on the
/// face mesh alone. The `has_cloth` / `has_cloth_normals` flags select
/// between the per-primitive cloth SSBO (filled in-place by the CPU
/// solver each frame) and the morph + base fallback path.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct TransformControl {
    pub vertex_count: u32,
    pub target_count: u32,
    pub has_cloth: u32,
    pub has_cloth_normals: u32,
    pub has_skin_anchors: u32,
    pub has_containment: u32,
    pub _pad0: [u32; 2],
}

impl TransformControl {
    pub fn zeroed() -> Self {
        Self {
            vertex_count: 0,
            target_count: 0,
            has_cloth: 0,
            has_cloth_normals: 0,
            has_skin_anchors: 0,
            has_containment: 0,
            _pad0: [0; 2],
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
    float fade_opacity;
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
    float fade_opacity;
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
        vec3 direct_light = camera.light_color * max(camera.light_intensity, 0.0);
        // Ambient fill only on the shaded side: keeps the lit face identical
        // to before while unlit faces receive the ambient color instead of
        // collapsing to shade_color * light alone.
        color.rgb = toon_col * direct_light
                  + camera.ambient_term * base_color_term * (1.0 - shading);
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

    // Global avatar fade (e.g. fade-out when no person is detected). 1.0 =
    // fully opaque; applied after alpha_mode handling so it dims even opaque
    // materials. Composited against the (possibly transparent) background.
    out_color = vec4(color.rgb, color.a * camera.fade_opacity);
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
    float fade_opacity;
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
// `render()` path. Reads per-vertex base data +
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
// **Status**: wired. Dispatched from `record_compute_prepass` for cloths
// whose `ClothDeformSnapshot::solver_backend == Gpu` (opt-in at attach
// time via `VULVATAR_CLOTH_GPU=1`).
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

// =========================================================================
// Cloth XPBD distance constraint — lambda update pass (per-constraint)
// =========================================================================
//
// Three-pass XPBD constraint projection (Macklin/Müller/Chentanez 2016):
//   1. lambda update (per constraint): compute Δλ_j, update persistent λ
//   2. accumulate (per particle): walk adjacency CSR, sum w·d·Δλ into Δx
//   3. apply (per particle): add Δx to position, zero Δx for next iter
//
// Choosing Jacobi (2 passes per iteration after the lambda update) over
// Gauss-Seidel-on-GPU (atomics or partition coloring) keeps the shaders
// simple — no atomicAdd, no `VK_EXT_shader_atomic_float`, no
// per-constraint colour dispatch. Convergence per iteration is slower
// than Gauss-Seidel, so real workloads run more iterations to compensate.
// =========================================================================
//
// First of three passes per iteration of the GPU XPBD distance-constraint
// solver (Macklin, Müller, Chentanez 2016). One invocation per constraint:
// reads the two endpoint positions, computes Δλ_j via the XPBD formula,
// updates the persistent λ_j buffer, and writes Δλ_j to a transient
// per-constraint scratch buffer so the per-particle accumulate pass can
// translate Δλ_j into Δx without re-doing the math (and without racing
// the lambda update across endpoints).
//
// The math:
// ```
//   α̃   = α(stiffness) / dt²
//   Δλ_j = -(C_j + α̃ · λ_j) / (w_a + w_b + α̃)
//   λ_j ← λ_j + Δλ_j
//   dlambda_j ← Δλ_j
// ```
// `α(stiffness)` mirrors `compliance_from_pbd_stiffness` on the CPU
// (slack² × 1e-7); `stiffness <= 1e-6` short-circuits to "constraint
// disabled" exactly like the CPU path.
pub mod cloth_constraint_lambda_update_cs {
    vulkano_shaders::shader! {
                        ty: "compute",
                        src: r"
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Positions {
    vec4 p[];
} positions;

struct Constraint {
    uint  particle_a;
    uint  particle_b;
    float rest_length;
    float stiffness;
};
layout(set = 0, binding = 1) readonly buffer Constraints {
    Constraint c[];
} constraints;

layout(set = 0, binding = 2) uniform Control {
    uint  particle_count;
    uint  constraint_count;
    float dt;
    uint  _pad;
} ctrl;

// Persistent Lagrange multiplier per constraint. Reset to 0 by the
// host at the start of each substep; the lambda update pass mutates
// it in place, and the next iteration of the same substep reads the
// accumulated value.
layout(set = 0, binding = 3) buffer Lambda {
    float l[];
} lambda;

// Δλ_j for THIS iteration only. Written here, read by the accumulate
// pass to translate Δλ_j into Δx without re-running the formula and
// without racing the lambda update across constraint endpoints.
layout(set = 0, binding = 4) writeonly buffer DeltaLambda {
    float l[];
} dlambda;

float xpbd_compliance(float stiffness) {
    float s = clamp(stiffness, 0.0, 1.0);
    float slack = max(1.0 - s, 0.0);
    return slack * slack * 1.0e-7;
}

void main() {
    uint cidx = gl_GlobalInvocationID.x;
    if (cidx >= ctrl.constraint_count) return;

    Constraint cn = constraints.c[cidx];
    if (cn.stiffness <= 1.0e-6) {
        // Disabled constraint — mirror the CPU short-circuit so this
        // edge contributes nothing this iteration. Clear dlambda so a
        // previous iteration's value can't leak into the accumulate
        // pass's read.
        dlambda.l[cidx] = 0.0;
        return;
    }

    vec4 pa = positions.p[cn.particle_a];
    vec4 pb = positions.p[cn.particle_b];
    vec3 diff = pa.xyz - pb.xyz;
    float dist = length(diff);
    // 1e-9 m = 1 nm. Below this both endpoints are numerically
    // coincident; matches the CPU XPBD path's zero-length guard.
    if (dist < 1.0e-9) {
        dlambda.l[cidx] = 0.0;
        return;
    }

    float w_a = pa.w;
    float w_b = pb.w;
    float w_sum = w_a + w_b;
    if (w_sum < 1.0e-12) {
        dlambda.l[cidx] = 0.0;
        return;
    }

    float compliance = xpbd_compliance(cn.stiffness);
    float dt_sq = max(ctrl.dt * ctrl.dt, 1.0e-12);
    float alpha_tilde = compliance / dt_sq;

    float c = dist - cn.rest_length;
    float lambda_old = lambda.l[cidx];
    float denom = w_sum + alpha_tilde;
    float delta_lambda = (-c - alpha_tilde * lambda_old) / denom;

    lambda.l[cidx] = lambda_old + delta_lambda;
    dlambda.l[cidx] = delta_lambda;
}
"
                    }
}

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
    uint  particle_count;
    uint  constraint_count;
    float dt;
    uint  _pad;
} ctrl;

// Δλ_j computed by `cloth_constraint_lambda_update_cs` this iteration.
// Read-only here: each constraint's Δλ is consumed by both endpoint
// invocations, no inter-invocation write race.
layout(set = 0, binding = 6) readonly buffer DeltaLambda {
    float l[];
} dlambda;

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
    // Under-relaxed Jacobi: Δx = Σ corr / (n + 1). The plain sum
    // diverges once per-particle constraint degree grows past a
    // couple of edges (measured: welded skirt, degree ≈ 6, first-step
    // positions ±1e8). Mirrors the CPU apply pass exactly.
    float n_rel = 1.0;
    for (uint k = start; k < end; ++k) {
        uint cidx = adj_constraints.c[k];
        Constraint cn = constraints.c[cidx];
        uint other = (cn.particle_a == pid) ? cn.particle_b : cn.particle_a;
        vec4 other_p = positions.p[other];
        vec3 dir = other_p.xyz - self_pos;  // points self → other
        float len = length(dir);
        // Matches the lambda-update + CPU XPBD zero-length guard (1 nm).
        if (len > 1.0e-9) {
            // XPBD: Δx_self = -d_unit · w_self · Δλ_j
            //   where d_unit = (other - self) / len = +d_unit_GPU
            //   = w_self · ((self - other) / len) · Δλ_j on the CPU
            //     convention (d_CPU = (a - b) / |a - b|, so d_unit_GPU
            //     = -d_CPU and the sign flip cancels).
            float dl = dlambda.l[cidx];
            delta += -(dir / len) * w_self * dl;
            n_rel += 1.0;
        }
    }
    deltas.d[pid] = vec4(delta / n_rel, 0.0);
}
"
                    }
}

// ---------------------------------------------------------------------------
// Cloth XPBD distance constraint projection — apply pass (P3-02 S2.1)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Cloth bend constraints — three-point edge-angle hinge (T09 mirror).
// ---------------------------------------------------------------------------
//
// The GPU port of `cloth_solver::constraints::project_bend_constraints`
// (the T09 reference — correction direction, wing-only weights, and the
// never-moved hinge all mirror it exactly). Three passes, run inside the
// constraint iteration loop AFTER the distance apply pass:
//
//   bend_update     — per constraint: measure the angle at the hinge,
//                     write the two wing correction rows.
//   bend_accumulate — per particle: gather wing corrections over the
//                     bend CSR (wings only; the hinge never appears),
//                     under-relaxed Jacobi Δx / (n + 1) — same
//                     divergence guard as the distance accumulate pass.
//   bend_apply      — per particle: add the gathered delta, zero it.
//
// The CPU reference applies corrections sequentially (Gauss-Seidel);
// the GPU is Jacobi with the same per-constraint formula and the same
// 1/(n+1) under-relaxation the distance path uses for the same reason.
pub mod cloth_bend_update_cs {
    vulkano_shaders::shader! {
                        ty: "compute",
                        src: r"
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Positions {
    vec4 p[];
} positions;

// Rust mirror: `ClothBendGpu`.
struct BendConstraint {
    uint  p0;   // hinge — never moved
    uint  p1;   // wing
    uint  p2;   // wing
    uint  _pad;
    float rest_angle;
    float stiffness;
    uvec2 _pad2;
};
layout(set = 0, binding = 1) readonly buffer BendConstraints {
    BendConstraint c[];
} bend;

layout(set = 0, binding = 2) uniform Control {
    uint  particle_count;
    uint  bend_count;
    uvec2 _pad;
} ctrl;

// One correction row per constraint per wing. xyz = delta, w unused.
layout(set = 0, binding = 3) writeonly buffer DeltaRowsA {
    vec4 d[];
} rows_a;
layout(set = 0, binding = 4) writeonly buffer DeltaRowsB {
    vec4 d[];
} rows_b;

void main() {
    uint cidx = gl_GlobalInvocationID.x;
    if (cidx >= ctrl.bend_count) return;

    BendConstraint bc = bend.c[cidx];
    float stiffness = clamp(bc.stiffness, 0.0, 1.0);
    vec3 delta1 = vec3(0.0);
    vec3 delta2 = vec3(0.0);

    if (stiffness > 0.0) {
        vec4 x0 = positions.p[bc.p0];
        vec4 x1 = positions.p[bc.p1];
        vec4 x2 = positions.p[bc.p2];
        vec3 e1 = x1.xyz - x0.xyz;
        vec3 e2 = x2.xyz - x0.xyz;
        float e1_len = length(e1);
        float e2_len = length(e2);
        // Zero-length edge: no correction (CPU skips identically).
        if (e1_len >= 1.0e-12 && e2_len >= 1.0e-12) {
            float dot = dot(e1, e2);
            float cross_len = length(cross(e1, e2));
            // Near-collinear edges leave no stable correction direction.
            if (cross_len >= 1.0e-6 * e1_len * e2_len) {
                float current_angle = atan(cross_len, dot);
                float err = current_angle - bc.rest_angle;
                if (abs(err) >= 1.0e-6) {
                    vec3 e1_norm = e1 / e1_len;
                    vec3 e2_norm = e2 / e2_len;
                    float cos_over = dot / (e1_len * e2_len);
                    vec3 perp1 = e2_norm - e1_norm * cos_over;
                    vec3 perp2 = e1_norm - e2_norm * cos_over;
                    float perp1_len = length(perp1);
                    float perp2_len = length(perp2);
                    if (perp1_len >= 1.0e-6 && perp2_len >= 1.0e-6) {
                        perp1 /= perp1_len;
                        perp2 /= perp2_len;
                        // Inv-mass shares among FREE wings only; the
                        // hinge never moves. inv_mass 0 = pinned.
                        float w1 = x1.w;
                        float w2 = x2.w;
                        float w_sum = w1 + w2;
                        if (w_sum >= 1.0e-12) {
                            float scale = stiffness * err;
                            if (w1 > 0.0) {
                                delta1 = perp1 * ((w1 / w_sum) * scale * e1_len);
                            }
                            if (w2 > 0.0) {
                                delta2 = perp2 * ((w2 / w_sum) * scale * e2_len);
                            }
                        }
                    }
                }
            }
        }
    }
    rows_a.d[cidx] = vec4(delta1, 0.0);
    rows_b.d[cidx] = vec4(delta2, 0.0);
}
"
                    }
}

pub mod cloth_bend_accumulate_cs {
    vulkano_shaders::shader! {
                        ty: "compute",
                        src: r"
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Positions {
    vec4 p[];
} positions;

layout(set = 0, binding = 1) readonly buffer BendAdjOffsets {
    uint o[];
} adj_offsets;
layout(set = 0, binding = 2) readonly buffer BendAdjConstraints {
    uint c[];
} adj_constraints;

struct BendConstraint {
    uint  p0;
    uint  p1;
    uint  p2;
    uint  _pad;
    float rest_angle;
    float stiffness;
    uvec2 _pad2;
};
layout(set = 0, binding = 3) readonly buffer BendConstraints {
    BendConstraint c[];
} bend;

layout(set = 0, binding = 4) readonly buffer DeltaRowsA {
    vec4 d[];
} rows_a;
layout(set = 0, binding = 5) readonly buffer DeltaRowsB {
    vec4 d[];
} rows_b;

layout(set = 0, binding = 6) writeonly buffer BendDeltas {
    vec4 d[];
} out_deltas;

layout(set = 0, binding = 7) uniform Control {
    uint  particle_count;
    uint  bend_count;
    uvec2 _pad;
} ctrl;

void main() {
    uint pid = gl_GlobalInvocationID.x;
    if (pid >= ctrl.particle_count) return;

    vec4 pp = positions.p[pid];
    if (pp.w <= 0.0) {
        out_deltas.d[pid] = vec4(0.0);
        return;
    }

    vec3 delta = vec3(0.0);
    float n_rel = 1.0;
    uint start = adj_offsets.o[pid];
    uint end   = adj_offsets.o[pid + 1u];
    for (uint k = start; k < end; ++k) {
        uint cidx = adj_constraints.c[k];
        BendConstraint bc = bend.c[cidx];
        // The CSR holds wings only (the hinge is never in it), but the
        // guard keeps a degenerate entry harmless.
        if (bc.p1 == pid) {
            delta += rows_a.d[cidx].xyz;
            n_rel += 1.0;
        } else if (bc.p2 == pid) {
            delta += rows_b.d[cidx].xyz;
            n_rel += 1.0;
        }
    }
    // Under-relaxed Jacobi — mirrors the distance accumulate pass.
    out_deltas.d[pid] = vec4(delta / n_rel, 0.0);
}
"
                    }
}

pub mod cloth_bend_apply_cs {
    vulkano_shaders::shader! {
                        ty: "compute",
                        src: r"
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Positions {
    vec4 p[];
} positions;

layout(set = 0, binding = 1) buffer BendDeltas {
    vec4 d[];
} bend_deltas;

layout(set = 0, binding = 2) uniform Control {
    uint  particle_count;
    uint  bend_count;
    uvec2 _pad;
} ctrl;

void main() {
    uint pid = gl_GlobalInvocationID.x;
    if (pid >= ctrl.particle_count) return;

    vec4 pos = positions.p[pid];
    vec4 d = bend_deltas.d[pid];
    pos.xyz += d.xyz;
    positions.p[pid] = pos;
    bend_deltas.d[pid] = vec4(0.0);
}
"
                    }
}

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

// Layout matches the Rust `ClothConstraintControl` struct after the
// XPBD migration — apply_cs only reads `particle_count` so the other
// fields are inert here, but the binding must still match the layout
// shared with the lambda update / accumulate passes.
layout(set = 0, binding = 2) uniform Control {
    uint  particle_count;
    uint  constraint_count;
    float dt;
    uint  _pad;
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
// Cloth self-collision — grid build (P3-02 S2.3)
// ---------------------------------------------------------------------------
//
// Bucketed spatial hash: cell size == min interaction distance
// (2 * self_collision_radius), fixed-size table (16384 cells) with a
// bounded slot count per cell (16). Distant cells hashing to the same
// bucket are harmless (the resolve pass filters by exact distance);
// bucket overflow DROPS entries, which is the one approximation vs
// the CPU spatial hash — cloth needs >16 particles inside one
// 2*radius cell for that, i.e. a fold far denser than the authoring
// radius assumes. Counts are zeroed by the recording half
// (`fill_buffer`) before each build dispatch, one grid per substep.
pub mod cloth_selfcol_build_cs {
    vulkano_shaders::shader! {
                    ty: "compute",
                    src: r"
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Positions {
    vec4 p[];
} positions;
layout(set = 0, binding = 1) buffer CellCounts {
    uint count[];
} cell_counts;
layout(set = 0, binding = 2) buffer CellEntries {
    uint e[];
} cell_entries;
// Rust mirror: `ClothSelfColControl`.
layout(set = 0, binding = 3) uniform Control {
    uint  particle_count;
    float radius;
    uint  table_size;
    uint  _pad;
} ctrl;

// Shared with the resolve pass — must stay byte-identical.
uint cell_hash(ivec3 c) {
    uint h = uint(c.x) * 73856093u ^ uint(c.y) * 19349663u ^ uint(c.z) * 83492791u;
    return h % ctrl.table_size;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= ctrl.particle_count) return;
    vec4 pp = positions.p[i];
    // NaN guard mirrors the CPU `SpatialHashGrid::insert` skip.
    if (any(isnan(pp.xyz)) || any(isinf(pp.xyz))) return;
    float cell = 2.0 * ctrl.radius;
    ivec3 c = ivec3(floor(pp.xyz / max(cell, 1.0e-7)));
    uint ci = cell_hash(c);
    const uint K = 16u;
    uint slot = atomicAdd(cell_counts.count[ci], 1u);
    if (slot < K) {
        cell_entries.e[ci * K + slot] = i;
    }
}
"
                }
}

// ---------------------------------------------------------------------------
// Cloth self-collision — resolve (P3-02 S2.3)
// ---------------------------------------------------------------------------
//
// Per-particle pass over the 27-cell neighbourhood: exact-distance
// filter, pinned particles skipped (self and neighbour — CPU parity),
// constraint-connected pairs skipped via the particle-neighbour CSR
// built at attach time. Corrections are averaged per particle, which
// is order-independent and therefore matches the CPU accumulator
// despite a different iteration order. GPU twin of
// `cloth_solver::collision::resolve_self_collisions`.
pub mod cloth_selfcol_resolve_cs {
    vulkano_shaders::shader! {
                    ty: "compute",
                    src: r"
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Positions {
    vec4 p[];
} positions;
// Particle-neighbour CSR: particles sharing a distance constraint.
layout(set = 0, binding = 1) readonly buffer AdjOffsets {
    uint o[];
} adj_offsets;
layout(set = 0, binding = 2) readonly buffer AdjParticles {
    uint n[];
} adj_particles;
layout(set = 0, binding = 3) buffer CellCounts {
    uint count[];
} cell_counts;
layout(set = 0, binding = 4) buffer CellEntries {
    uint e[];
} cell_entries;
layout(set = 0, binding = 5) uniform Control {
    uint  particle_count;
    float radius;
    uint  table_size;
    uint  _pad;
} ctrl;

uint cell_hash(ivec3 c) {
    uint h = uint(c.x) * 73856093u ^ uint(c.y) * 19349663u ^ uint(c.z) * 83492791u;
    return h % ctrl.table_size;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= ctrl.particle_count) return;
    vec4 pp = positions.p[i];
    if (pp.w <= 0.0) return; // pinned — CPU `if particles[i].pinned continue`

    float min_dist = 2.0 * ctrl.radius;
    float cell = max(min_dist, 1.0e-7);
    ivec3 base = ivec3(floor(pp.xyz / cell));
    const uint K = 16u;

    vec3 acc = vec3(0.0);
    uint hits = 0u;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                uint ci = cell_hash(base + ivec3(dx, dy, dz));
                uint cnt = min(cell_counts.count[ci], K);
                for (uint s = 0u; s < cnt; ++s) {
                    uint j = cell_entries.e[ci * K + s];
                    if (j == i) continue;
                    vec4 q = positions.p[j];
                    if (q.w <= 0.0) continue; // pinned neighbour
                    // Constraint-connected pairs are expected to be
                    // close — CPU `connected_pairs` skip.
                    bool connected = false;
                    for (uint k = adj_offsets.o[i]; k < adj_offsets.o[i + 1u]; ++k) {
                        if (adj_particles.n[k] == j) {
                            connected = true;
                            break;
                        }
                    }
                    if (connected) continue;
                    vec3 diff = q.xyz - pp.xyz; // i -> j
                    float d2 = dot(diff, diff);
                    // Near-coincident pairs (< 0.5 mm) are weld copies /
                    // seam overlaps, not penetrations — must mirror the
                    // CPU's SELF_COL_COINCIDENT_EPS_M (5e-4 m; eps² =
                    // 2.5e-7). The old 1e-24 guard let a micron of
                    // divergence trigger the full 2·radius push and
                    // blew welded seams apart into spikes.
                    const float COINCIDENT_EPS_SQ = 2.5e-7;
                    if (d2 < min_dist * min_dist && d2 > COINCIDENT_EPS_SQ) {
                        float d = sqrt(d2);
                        float overlap = min_dist - d;
                        vec3 dir = diff / d;
                        // CPU: corrections[i] -= dir * overlap * 0.5
                        //      (averaged below) — i moves away from j.
                        acc -= dir * (overlap * 0.5);
                        hits += 1u;
                    }
                }
            }
        }
    }
    if (hits > 0u) {
        vec3 avg = acc / float(hits);
        positions.p[i] = vec4(pp.xyz + avg, pp.w);
    }
}
"
                }
}

// ---------------------------------------------------------------------------
// Cloth collision projection compute shader (P3-02 S2.2)
// ---------------------------------------------------------------------------
//
// Per-particle projection out of world-space capsule colliders — the GPU
// twin of `cloth_solver::collision::collide` (external collision only;
// self-collision stays CPU-side). Runs once per substep after the XPBD
// constraint iterations, matching the CPU step order. Pinned particles
// (inv_mass == 0) are skipped, so the pin rows authored by the prepare
// half survive untouched.
pub mod cloth_collide_cs {
    vulkano_shaders::shader! {
                    ty: "compute",
                    src: r"
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// xyz = position, w = inv_mass (0.0 = effectively pinned / immobile).
layout(set = 0, binding = 0) buffer Positions {
    vec4 p[];
} positions;

// Capsule: a.xyz / b.xyz = segment endpoints, a.w = radius.
struct Capsule {
    vec4 a;
    vec4 b;
};
layout(set = 0, binding = 1) readonly buffer Colliders {
    Capsule c[];
} colliders;

// Body-SDF distance field the splat pass filled (u32 cells holding f32
// bit patterns; u32(-1) = unsplatted sentinel). Avatar-root space —
// the same contract the spring solver's field uses.
layout(set = 0, binding = 3) readonly buffer SdfFieldBuf {
    uint v[];
} sdf;

// xyz = grid origin (metres), w = voxel size; dims in yzw via uvec4.
struct SdfParams {
    vec4 origin_voxel;
    uvec4 dims_pad;
};
layout(set = 0, binding = 4) uniform SdfCtl {
    SdfParams s;
} sdfctl;

// Rust mirror: `ClothCollideControl`.
layout(set = 0, binding = 2) uniform Control {
    uint  particle_count;
    uint  collider_count;
    float margin;
    uint  has_sdf;
    float sdf_contact;
    uvec2 _pad;
} ctrl;

// Matches `simulation::sdf::{SENTINEL, OUTSIDE_VALUE}` (f32::MAX).
const float SDF_SENTINEL_F = 3.402823466e38;
const float SDF_OUTSIDE = 0.12; // SHELL_METRES * 2.0

// Trilinear sample mirroring `SdfField::sample`, including the
// sentinel semantics: out-of-grid => SENTINEL, a `raw >= SENTINEL`
// corner interpolates as OUTSIDE (an unsplatted u32(-1) corner stays
// NaN through this compare — exactly like the CPU, and NaN fails the
// `d < contact` gate below, so both paths mean no contact).
float sdf_sample(vec3 p) {
    vec3 o = sdfctl.s.origin_voxel.xyz;
    float voxel = sdfctl.s.origin_voxel.w;
    vec3 f = (p - o) / voxel;
    if (f.x < 0.0 || f.y < 0.0 || f.z < 0.0) return SDF_SENTINEL_F;
    ivec3 c = ivec3(floor(f));
    uvec3 dims = sdfctl.s.dims_pad.xyz;
    if (c.x + 1 >= int(dims.x) || c.y + 1 >= int(dims.y) || c.z + 1 >= int(dims.z))
        return SDF_SENTINEL_F;
    vec3 t = f - vec3(c);
    float acc = 0.0;
    for (uint dz = 0u; dz < 2u; ++dz) {
        float wz = dz == 0u ? 1.0 - t.z : t.z;
        for (uint dy = 0u; dy < 2u; ++dy) {
            float wy = dy == 0u ? 1.0 - t.y : t.y;
            for (uint dx = 0u; dx < 2u; ++dx) {
                float wx = dx == 0u ? 1.0 - t.x : t.x;
                uint idx = uint(c.x + int(dx))
                         + dims.x * (uint(c.y + int(dy)) + dims.y * uint(c.z + int(dz)));
                float raw = uintBitsToFloat(sdf.v[idx]);
                acc += (raw >= SDF_SENTINEL_F ? SDF_OUTSIDE : raw) * wx * wy * wz;
            }
        }
    }
    return acc;
}

void main() {
    uint pid = gl_GlobalInvocationID.x;
    if (pid >= ctrl.particle_count) return;

    vec4 pp = positions.p[pid];
    // Matches the CPU `if p.pinned { continue }` skip.
    if (pp.w <= 0.0) return;

    vec3 pos = pp.xyz;
    for (uint k = 0u; k < ctrl.collider_count; ++k) {
        vec3 a = colliders.c[k].a.xyz;
        vec3 b = colliders.c[k].b.xyz;
        float radius = colliders.c[k].a.w + ctrl.margin;
        // Closest point on the segment — degenerate segment (sphere)
        // collapses to its endpoint, same as the CPU helper.
        vec3 ab = b - a;
        vec3 ap = pos - a;
        float ab_len_sq = dot(ab, ab);
        vec3 closest = a;
        if (ab_len_sq >= 1.0e-12) {
            float t = clamp(dot(ap, ab) / ab_len_sq, 0.0, 1.0);
            closest = a + ab * t;
        }
        vec3 diff = pos - closest;
        float dist = length(diff);
        if (dist < radius && dist > 1.0e-12) {
            vec3 n = diff / dist;
            pos = closest + n * radius;
        }
    }

    // Body-SDF stage (mirrors `SdfField::resolve`): project onto the
    // `sdf_contact` isosurface along the central-difference gradient.
    // The smooth direction field lets pleats fold instead of being
    // blasted apart by the capsule radial pushes above.
    if (ctrl.has_sdf != 0u && ctrl.sdf_contact > 0.0) {
        float d = sdf_sample(pos);
        if (d < ctrl.sdf_contact) {
            float h = sdfctl.s.origin_voxel.w;
            vec3 o = sdfctl.s.origin_voxel.xyz;
            vec3 g = vec3(
                sdf_sample(pos + vec3(h, 0.0, 0.0)) - sdf_sample(pos - vec3(h, 0.0, 0.0)),
                sdf_sample(pos + vec3(0.0, h, 0.0)) - sdf_sample(pos - vec3(0.0, h, 0.0)),
                sdf_sample(pos + vec3(0.0, 0.0, h)) - sdf_sample(pos - vec3(0.0, 0.0, h))
            );
            float gl = length(g);
            // `!(len > eps)` also catches NaN gradients (unsplatted
            // region) — same no-contact outcome as the CPU's
            // `gradient() -> None`.
            if (gl > 1.0e-6) {
                vec3 n = g / gl;
                float target = min(ctrl.sdf_contact, 0.12 * 0.9);
                pos += n * (target - d);
            }
        }
    }

    positions.p[pid] = vec4(pos, pp.w);
}
"
                }
}

// ---------------------------------------------------------------------------
// Cloth vertex normal recomputation compute shader (P3-02 S3.1)
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
// Each incident triangle contributes its **unit** face normal scaled by
// the incident angle at this vertex (angle-weighted accumulation, Max
// 1999 — matches `cloth_solver::output::compute_normals`). Bounded by
// [0, π] per triangle so the result follows the local 1-ring shape
// instead of the area of the largest incident triangle. Embarrassingly
// parallel — no atomics, no extensions, no cross-invocation
// synchronisation. Degenerate / isolated vertices fall back to
// `(0, 1, 0)`; CPU path uses the same fallback.
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

// Angle between two vectors, robust against degenerate inputs.
// Returns 0 when either side has near-zero length so a collinear /
// zero-length edge doesn't contribute a NaN to the accumulator.
float safe_angle(vec3 u, vec3 v) {
    float lu = length(u);
    float lv = length(v);
    if (lu < 1e-9 || lv < 1e-9) return 0.0;
    float d = dot(u, v) / (lu * lv);
    return acos(clamp(d, -1.0, 1.0));
}

void main() {
    uint vid = gl_GlobalInvocationID.x;
    if (vid >= ctrl.vertex_count) return;

    uint start = adj_offsets.o[vid];
    uint end   = adj_offsets.o[vid + 1u];

    // Angle-weighted (Max 1999) face-normal accumulation. Each
    // incident triangle contributes its **unit** face normal scaled
    // by the incident angle at THIS vertex, so the result depends
    // only on the local geometry around the 1-ring rather than on
    // which incident triangle happens to be largest.
    vec3 accum = vec3(0.0);
    for (uint k = start; k < end; ++k) {
        uint tri = adj_triangles.t[k];
        uint i0 = indices.t[tri * 3u + 0u];
        uint i1 = indices.t[tri * 3u + 1u];
        uint i2 = indices.t[tri * 3u + 2u];
        vec3 p0 = positions.p[i0].xyz;
        vec3 p1 = positions.p[i1].xyz;
        vec3 p2 = positions.p[i2].xyz;
        vec3 raw_normal = cross(p1 - p0, p2 - p0);
        float face_area2 = length(raw_normal);
        if (face_area2 < 1e-12) continue;  // degenerate triangle
        vec3 face_normal = raw_normal / face_area2;
        // Pick the two edges that meet at the current vertex `vid`
        // and weight by the angle between them.
        float angle;
        if (vid == i0) {
            angle = safe_angle(p1 - p0, p2 - p0);
        } else if (vid == i1) {
            angle = safe_angle(p0 - p1, p2 - p1);
        } else {
            angle = safe_angle(p0 - p2, p1 - p2);
        }
        accum += face_normal * angle;
    }

    float len = length(accum);
    vec3 nrm = (len > 1e-12) ? (accum / len) : vec3(0.0, 1.0, 0.0);
    normals.n[vid] = vec4(nrm, 0.0);
}
"
                    }
}

pub mod body_sdf_splat_cs {
    // One workgroup invocation per triangle. Splats the triangle's
    // exact point-to-triangle distance into an avatar-root-space voxel
    // field via `atomicMin` on the f32 bit pattern (valid for
    // non-negative floats, and splatted distances are non-negative).
    // The recording half fills the field with `u32::MAX` before this
    // dispatch; unsplatted cells stay at the sentinel, which the CPU
    // sampler (simulation/sdf.rs) reads as "outside the collision
    // band".
    //
    // The skinned vertices are in avatar-root space (see
    // frame_input.rs — the instance world transform is never applied),
    // which is exactly the space the grid and the spring solver live
    // in, so no vertex transform happens here.
    vulkano_shaders::shader! {
                    ty: "compute",
                    src: r"
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct OutVertex {
    vec4 position;
    vec4 normal;
    vec2 uv;
    uvec2 _pad;
};

layout(set = 0, binding = 0) readonly buffer Vertices {
    OutVertex v[];
} verts;

layout(set = 0, binding = 1) readonly buffer TriangleIndices {
    uint i[];
} idx;

layout(set = 0, binding = 2) buffer Field {
    uint d[];
} field;

// std140. Rust mirror: `BodySdfSplatParams`.
layout(set = 0, binding = 3) uniform Params {
    // xyz = grid dims, w = triangle count.
    uvec4 dims_tri;
    // xyz = grid origin (avatar-root), w = metres per cell.
    vec4 origin_voxel;
    // x = splat shell (metres). The rest pads to 16 B.
    vec4 shell_pad;
} prm;

// Cells never written past the shell stay at u32::MAX (filled by the
// recording half before this dispatch). The CPU side mirrors this
// contract as `simulation::sdf::SENTINEL`.

// Closest point on triangle (a,b,c) to p — Ericson, Real-Time
// Collision Detection §5.1.5.
vec3 closest_point_on_triangle(vec3 p, vec3 a, vec3 b, vec3 c) {
    vec3 ab = b - a;
    vec3 ac = c - a;
    vec3 ap = p - a;
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return a;
    vec3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return b;
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        float v = d1 / (d1 - d3);
        return a + v * ab;
    }
    vec3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return c;
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        float w = d2 / (d2 - d6);
        return a + w * ac;
    }
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }
    float denom = 1.0 / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return a + ab * v + ac * w;
}

void main() {
    uint tri = gl_GlobalInvocationID.x;
    if (tri >= prm.dims_tri.w) return;

    vec3 p0 = verts.v[idx.i[tri * 3u + 0u]].position.xyz;
    vec3 p1 = verts.v[idx.i[tri * 3u + 1u]].position.xyz;
    vec3 p2 = verts.v[idx.i[tri * 3u + 2u]].position.xyz;

    // Degenerate triangles contribute nothing (their point-to-triangle
    // distance is still exact below, but skipping saves the loop).
    vec3 cross_len = cross(p1 - p0, p2 - p0);
    if (dot(cross_len, cross_len) < 1e-20) return;

    float shell = prm.shell_pad.x;
    float voxel = prm.origin_voxel.w;
    vec3 lo = min(p0, min(p1, p2)) - vec3(shell);
    vec3 hi = max(p0, max(p1, p2)) + vec3(shell);

    ivec3 dims = ivec3(prm.dims_tri.xyz);
    ivec3 c0 = max(ivec3(floor((lo - prm.origin_voxel.xyz) / voxel)), ivec3(0));
    // ceil-1 == floor for the inclusive upper cell index.
    ivec3 c1 = min(ivec3(floor((hi - prm.origin_voxel.xyz) / voxel)), dims - 1);

    for (int z = c0.z; z <= c1.z; ++z) {
        for (int y = c0.y; y <= c1.y; ++y) {
            for (int x = c0.x; x <= c1.x; ++x) {
                // Node sampling: the value at node (x,y,z) is the
                // distance from that node's position — the CPU sampler
                // (SdfField::sample) trilinearly interpolates node
                // values, which is only coherent with this convention.
                vec3 node = prm.origin_voxel.xyz + vec3(x, y, z) * voxel;
                float dist = length(node - closest_point_on_triangle(node, p0, p1, p2));
                if (dist > shell) continue;
                uint cell_idx = uint(x) + uint(dims.x) * (uint(y) + uint(dims.y) * uint(z));
                atomicMin(field.d[cell_idx], floatBitsToUint(dist));
            }
        }
    }
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
    // Morph deltas are SPARSE: binding 1 holds per-target runs of
    // (vertex, delta) entries sorted by vertex index, binding 8 locates
    // each target's run, and binding 9 carries the per-frame weights as
    // an unbounded float array. This keeps GPU memory proportional to
    // non-zero deltas and removes any cap on targets per primitive.
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
    // Sparse per-target runs, located via `morph_info`. Each entry packs
    // the vertex index in .x (float — exact for indices < 2^24) and the
    // position delta in .yzw. Targets that also carry normal deltas use
    // stride 2: a second vec4 (nx, ny, nz, unused) follows each entry.
    vec4 e[];
} morph;

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
    uint has_skin_anchors;
    uint has_containment;
    uvec2 _pad0;
} ctrl;

layout(set = 0, binding = 8) readonly buffer MorphTargetInfo {
    // One uvec4 per morph target: x = first entry index into `morph.e`,
    // y = entry count, z = entry stride (1 = position only, 2 = +normal),
    // w = unused.
    uvec4 i[];
} morph_info;

layout(set = 0, binding = 9) readonly buffer MorphWeights {
    float w[];
} morph_w;

layout(set = 0, binding = 5) writeonly buffer OutVertices {
    OutVertex v[];
} out_v;

struct SkinAnchor {
    uint body_vertex_idx;
    float min_clearance;
    float weight;
    // 0 = clearance (push this vertex outward off the parent surface),
    // 1 = containment (clamp this vertex back inside the parent surface).
    // The two anchor sets travel in separate buffers (binding 6 vs 10).
    // Binding 6 applies clearance unconditionally; binding 10 branches
    // per anchor mode so the containment slot can also carry
    // cross-region clearance anchors (upper-outer vs bottom garments).
    uint mode;
};

layout(set = 0, binding = 6) readonly buffer SkinAnchors {
    SkinAnchor a[];
} skin_anchors;

layout(set = 0, binding = 7) readonly buffer BodyVertices {
    OutVertex v[];
} body_v;

layout(set = 0, binding = 10) readonly buffer ContainmentAnchors {
    SkinAnchor a[];
} containment_anchors;

layout(set = 0, binding = 11) readonly buffer ContainmentParent {
    OutVertex v[];
} containment_parent_v;

layout(set = 1, binding = 0) readonly buffer SkinningData {
    mat4 matrices[];
} skinning;

// =====================================================================
// Dual Quaternion Skinning (DQS, Kavan 2007) — derived in-shader from
// the per-bone skinning matrices that LBS would consume.
//
// Compared with the prior LBS path, DQS eliminates the candy-wrapper
// twist artefact at joints with non-trivial rotational delta (forearm
// pronation, hip yaw under skirts, neck twist). Cost: one
// matrix→quaternion extraction per blended joint (4 per vertex) plus
// a 4-way quaternion blend + normalise. Negligible against the
// per-frame morph and constraint solvers.
//
// Translation is recovered from the 4th column of the skinning matrix
// rather than re-derived from the dual, because the source matrices
// from `Application::build_skinning_matrices` are
// `global_transform · inverse_bind` products where the translation
// component is already in world space — re-encoding it into a dual
// quaternion just to decode it again would be lossy if the source
// matrix carries any non-rigid component.
// =====================================================================

// Mike Day's robust mat3→quat conversion. GLSL is column-major
// (`m[col][row]`), so the row-major literature's `M[i][j]` reads as
// `m[j][i]` here.
vec4 mat3_to_quat(mat3 m) {
    float trace = m[0][0] + m[1][1] + m[2][2];
    vec4 q;
    if (trace > 0.0) {
        float s = sqrt(trace + 1.0) * 2.0;
        q.w = 0.25 * s;
        q.x = (m[1][2] - m[2][1]) / s;
        q.y = (m[2][0] - m[0][2]) / s;
        q.z = (m[0][1] - m[1][0]) / s;
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        float s = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
        q.w = (m[1][2] - m[2][1]) / s;
        q.x = 0.25 * s;
        q.y = (m[1][0] + m[0][1]) / s;
        q.z = (m[2][0] + m[0][2]) / s;
    } else if (m[1][1] > m[2][2]) {
        float s = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
        q.w = (m[2][0] - m[0][2]) / s;
        q.x = (m[1][0] + m[0][1]) / s;
        q.y = 0.25 * s;
        q.z = (m[2][1] + m[1][2]) / s;
    } else {
        float s = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
        q.w = (m[0][1] - m[1][0]) / s;
        q.x = (m[2][0] + m[0][2]) / s;
        q.y = (m[2][1] + m[1][2]) / s;
        q.z = 0.25 * s;
    }
    return q;
}

// Rotate a vector by a unit quaternion. Standard
// `q ⊗ v ⊗ q⁻¹` derivation reduced to vector ops.
vec3 quat_rotate(vec4 q, vec3 v) {
    vec3 u = q.xyz;
    float w = q.w;
    return v + 2.0 * cross(u, cross(u, v) + w * v);
}

void main() {
    uint vid = gl_GlobalInvocationID.x;
    if (vid >= ctrl.vertex_count) return;

    VertexBase b = base.v[vid];
    vec3 pos = b.position.xyz;
    vec3 nrm = b.normal.xyz;

    // Morph target blend. Deltas are sparse: per target, a run of
    // (vertex, delta) entries sorted by vertex index. Each vertex
    // binary-searches its own index inside every *active* target's run,
    // so per-frame cost scales with the number of driven expressions,
    // and GPU memory scales with non-zero deltas instead of
    // targets × vertices.
    for (uint t = 0u; t < ctrl.target_count; t++) {
        float w = morph_w.w[t];
        if (abs(w) < 1e-6) continue;
        uvec4 info = morph_info.i[t];
        uint base = info.x;
        uint count = info.y;
        uint stride = info.z;
        uint lo = 0u;
        uint hi = count;
        while (lo < hi) {
            uint mid = lo + ((hi - lo) >> 1u);
            if (uint(morph.e[base + mid * stride].x) < vid) lo = mid + 1u;
            else hi = mid;
        }
        if (lo < count && uint(morph.e[base + lo * stride].x) == vid) {
            vec4 entry = morph.e[base + lo * stride];
            pos += w * entry.yzw;
            if (stride == 2u) {
                nrm += w * morph.e[base + lo * stride + 1u].xyz;
            }
        }
    }

    const float WEIGHT_EPS = 1.0e-4;
    vec3 world_pos;
    vec3 world_nrm;
    // R2 telemetry: total render-side correction applied by the
    // clearance / containment branches below (metres), published in
    // out_v.position.w — the vertex shaders only read .xyz, so the
    // channel is free. This is exactly the per-vertex gap between the
    // physics state (cloth SSBO / skinned position) and the drawn
    // position, which the physics-side diagnostics cannot see.
    float corr_len = 0.0;
    // R2: render-only corrections are NOT written back into the cloth /
    // physics state, so an unbounded push can stretch the drawn mesh
    // arbitrarily far from the state the solver validated. Bound each
    // anchor's applied displacement: healthy frames see millimetre-to-
    // centimetre corrections, so this cap only bites on runaway frames,
    // turning an unbounded tear into a bounded, telemetry-visible
    // defect. The persistent fix (folding the correction into the
    // physics constraints, or auditing per-stage vertex deltas) is
    // tracked in docs/quality-improvement-plan.md R2.
    const float MAX_RENDER_CORRECTION_M = 0.25;

    // Cloth override: the physics solver writes per-frame world-space
    // positions into `cloth_pos` and normals into `cloth_norm`. Pinned
    // particles are already attached to skeletal bones by the solver,
    // so skinning must NOT be applied on top of simulated cloth vertices.
    if (ctrl.has_cloth > 0u) {
        world_pos = cloth_pos.p[vid].xyz;
        world_nrm = (ctrl.has_cloth_normals > 0u) ? cloth_norm.n[vid].xyz : nrm;
    } else {
        float total_w = b.joint_weights.x + b.joint_weights.y +
                        b.joint_weights.z + b.joint_weights.w;
        if (total_w < WEIGHT_EPS) {
            // No skinning weights — emit the morph/cloth-blended position
            // as-is (used for accessories that aren't bone-bound).
            world_pos = pos;
            world_nrm = nrm;
        } else {
            // ----- Dual Quaternion Skinning (Kavan 2007) -----
            vec4 ref_real = vec4(0.0, 0.0, 0.0, 1.0);
            for (uint i = 0u; i < 4u; i++) {
                if (b.joint_weights[i] > WEIGHT_EPS) {
                    ref_real = mat3_to_quat(mat3(skinning.matrices[b.joint_indices[i]]));
                    break;
                }
            }
            vec4 acc_real = vec4(0.0);
            vec4 acc_dual = vec4(0.0);
            for (uint i = 0u; i < 4u; i++) {
                float wi = b.joint_weights[i];
                if (wi <= WEIGHT_EPS) continue;
                mat4 mi = skinning.matrices[b.joint_indices[i]];
                vec4 qr = mat3_to_quat(mat3(mi));
                float s = (dot(qr, ref_real) >= 0.0) ? 1.0 : -1.0;
                qr = s * qr;
                vec3 ti = mi[3].xyz;
                vec3 qrxyz = qr.xyz;
                float qrw = qr.w;
                vec3 dxyz = 0.5 * (qrw * ti + cross(ti, qrxyz));
                float dw = -0.5 * dot(ti, qrxyz);
                vec4 qd = vec4(dxyz, dw);
                acc_real += wi * qr;
                acc_dual += wi * qd;
            }
            float acc_len = length(acc_real);
            if (acc_len < 1.0e-6) {
                world_pos = pos;
                world_nrm = nrm;
            } else {
                float inv_len = 1.0 / acc_len;
                vec4 q_real = acc_real * inv_len;
                vec4 q_dual = acc_dual * inv_len;
                vec3 rqxyz = -q_real.xyz;
                float rqw = q_real.w;
                vec3 t_recovered = 2.0 * (q_dual.w * rqxyz
                                          + rqw * q_dual.xyz
                                          + cross(q_dual.xyz, rqxyz));
                world_pos = quat_rotate(q_real, pos) + t_recovered;
                world_nrm = quat_rotate(q_real, nrm);
            }
        }
    }

    // Skin anchor anti-penetration, two independent constraint sets:
    //
    // Clearance (binding 6, `body_v` = the INNER surface): this vertex
    //   must stay at least `min_clearance` OUTSIDE the parent surface,
    //   measured along the parent vertex's outward normal — pushes the
    //   outer surface away from the inner one (garment off garment, or
    //   garment off the body). The push direction is NEVER flipped to
    //   follow this vertex's own normal: for body parents the outward
    //   skin normal is the only valid away direction, and fold-region
    //   flips were measured pushing garments 200mm+ INTO the body.
    //
    // Containment (binding 10, `containment_parent_v` = the OUTER
    //   surface): this vertex must stay at most `min_clearance`
    //   (rest-derived) outside the parent surface — clamps the inner
    //   garment back inside the outer one when it pokes through at
    //   bent joints. The two sets are separate so a middle layer can
    //   carry both at once.
    //
    // R3 time contract for the containment parent surface:
    // `containment_parent_v` is the parent's PREVIOUS-frame FINAL VBO
    // (filled by a `copy_buffer` after every transform dispatch — see
    // `compute_prepass`), so the clamp reads a deterministic
    // last-frame state no matter where in the dispatch order the
    // parent ran. Both parent surfaces are read through a sanity band
    // on the parent position (avatar-local coords live well inside
    // it) plus NaN-rejecting comparisons; on the very first frame the
    // history buffer is uninitialised — garbage reads fail the band
    // or the NaN comparison and the clamp stays inert until the first
    // copy lands.
    if (ctrl.has_skin_anchors > 0u) {
        SkinAnchor anc = skin_anchors.a[vid];
        if (anc.body_vertex_idx != 0xFFFFFFFFu && anc.weight > 1e-4) {
            vec3 bp = body_v.v[anc.body_vertex_idx].position.xyz;
            vec3 bn = body_v.v[anc.body_vertex_idx].normal.xyz;
            float plen = length(bp);
            float nlen = length(bn);
            if (plen > 1e-3 && plen < 10.0 && nlen > 1e-4) {
                bn /= nlen;
                float clearance = dot(world_pos - bp, bn);
                // Anchor contract: anchors are derived at the
                // GENERATION pose and only certify small penetrations.
                // clearance < -5 mm means this vertex paired across a
                // fold / side of the torso and is now far behind the
                // anchor plane in THIS pose — 'correcting' it slams the
                // vertex ~8 cm outward every frame (measured: floating
                // chest shards). Skip deep negatives; only real, small
                // penetrations are corrected. Mirrors the generation
                // side's -5 mm rejection (clearance.rs, v20).
                if (clearance < anc.min_clearance && clearance > -0.005) {
                    float push = min((anc.min_clearance - clearance) * anc.weight,
                                     MAX_RENDER_CORRECTION_M);
                    world_pos += bn * push;
                    corr_len += push;
                }
            }
        }
    }

    if (ctrl.has_containment > 0u) {
        SkinAnchor anc = containment_anchors.a[vid];
        if (anc.body_vertex_idx != 0xFFFFFFFFu && anc.weight > 1e-4) {
            vec3 op = containment_parent_v.v[anc.body_vertex_idx].position.xyz;
            vec3 on = containment_parent_v.v[anc.body_vertex_idx].normal.xyz;
            float plen = length(op);
            float nlen = length(on);
            if (plen > 1e-3 && plen < 10.0 && nlen > 1e-4) {
                on /= nlen;
                float c = dot(world_pos - op, on);
                // The slot carries either semantic; the anchor's mode
                // decides. Containment (middle layers): clamp back
                // inside the outer surface. Clearance (cross-region,
                // e.g. a jacket hem over a skirt): push this vertex
                // out to at least `min_clearance` off the inner
                // surface — same math as the binding-6 branch, against
                // a garment parent the clearance slot cannot reference.
                // Same deep-negative skip as the clearance branch
                // above: cross-region pairs are pose-derived too.
                if (anc.mode == 1u) {
                    if (c > anc.min_clearance && c < anc.min_clearance + 0.005 + MAX_RENDER_CORRECTION_M) {
                        float pull = min((c - anc.min_clearance) * anc.weight,
                                         MAX_RENDER_CORRECTION_M);
                        world_pos -= on * pull;
                        corr_len += pull;
                    }
                } else {
                    if (c < anc.min_clearance && c > -0.005) {
                        float push = min((anc.min_clearance - c) * anc.weight,
                                         MAX_RENDER_CORRECTION_M);
                        world_pos += on * push;
                        corr_len += push;
                    }
                }
            }
        }
    }

    out_v.v[vid].position = vec4(world_pos, corr_len);
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
    sample_count: u32,
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
            multisample_state: Some(MultisampleState {
                // Must match the subpass' attachment sample count. MSAA off
                // (sample_count == 1) keeps the Vulkano default (Sample1).
                rasterization_samples: SampleCount::try_from(sample_count)
                    .unwrap_or(SampleCount::Sample1),
                // Alpha-to-coverage only on the Cutout variant, and only under
                // MSAA. It converts the fragment's alpha into a sample coverage
                // mask so alpha-tested edges (hair, foliage) antialias along
                // with geometry. Deliberately NOT enabled for Opaque/Blend:
                // opaque materials carry the global fade-out alpha
                // (`out_color.a = fade_opacity`), and A2C there would dither
                // the whole avatar into stipple while it fades.
                alpha_to_coverage_enable: matches!(alpha_mode, RenderAlphaMode::Cutout)
                    && sample_count > 1,
                ..MultisampleState::default()
            }),
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
    sample_count: u32,
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
            multisample_state: Some(MultisampleState {
                // Must match the subpass' attachment sample count. MSAA off
                // (sample_count == 1) keeps the Vulkano default (Sample1).
                rasterization_samples: SampleCount::try_from(sample_count)
                    .unwrap_or(SampleCount::Sample1),
                ..MultisampleState::default()
            }),
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
/// Dispatched from `record_compute_prepass` inside the GPU cloth
/// substep loop (verlet first, then the constraint passes).
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
/// `stiffness` is the [0,1] knob the author sets; the lambda-update
/// shader maps it to XPBD compliance via
/// `xpbd_compliance(stiffness) = (1 - stiffness)² * 1e-7` so
/// `stiffness = 1` yields the rigid (α = 0) limit and `stiffness = 0`
/// short-circuits to "constraint disabled".
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ClothConstraintGpu {
    pub particle_a: u32,
    pub particle_b: u32,
    pub rest_length: f32,
    pub stiffness: f32,
}

/// SSBO row for one edge-angle bend constraint (T09 model — see
/// `cloth_solver::constraints::project_bend_constraints`, the CPU
/// reference the bend kernels mirror). `p0` is the hinge (never
/// moved), `p1`/`p2` the wings; `rest_angle` is derived from the rest
/// positions by `ClothSimState::from_asset` on the CPU and shipped
/// here precomputed.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClothBendGpu {
    pub p0: u32,
    pub p1: u32,
    pub p2: u32,
    pub _pad: u32,
    pub rest_angle: f32,
    pub stiffness: f32,
    pub _pad2: [u32; 2],
}

/// Control block for the bend kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClothBendControl {
    pub particle_count: u32,
    pub bend_count: u32,
    pub _pad: [u32; 2],
}

/// Per-frame control block consumed by the constraint shaders.
/// std140 layout — 16 bytes packed.
///
/// `constraint_count` and `dt` are the two new fields added during the
/// GPU XPBD migration. `dt` is the substep duration; the lambda update
/// shader uses it to scale compliance into `α̃ = α / dt²`. At
/// `stiffness = 1` (the rigid limit), `α = 0` and `dt` falls out, so
/// the field is harmless for the legacy PBD-style usage.
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ClothConstraintControl {
    pub particle_count: u32,
    pub constraint_count: u32,
    pub dt: f32,
    pub _pad: u32,
}

/// Build the cloth XPBD lambda-update compute pipeline. One
/// invocation per constraint per iteration. Runs **before**
/// [`create_cloth_constraint_accumulate_compute_pipeline`] each
/// iteration so the accumulate pass has Δλ_j available without
/// reaching for the lambda buffer (and racing across endpoints).
pub fn create_cloth_constraint_lambda_update_compute_pipeline(
    device: Arc<Device>,
) -> Result<Arc<ComputePipeline>, String> {
    let cs_module = cloth_constraint_lambda_update_cs::load(device.clone())
        .map_err(|e| format!("failed to load cloth constraint lambda update shader: {e}"))?;
    let cs_entry = cs_module.entry_point("main").ok_or_else(|| {
        "cloth constraint lambda update shader entry point 'main' not found".to_string()
    })?;
    let stages = [PipelineShaderStageCreateInfo::new(cs_entry)];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| {
                format!("failed to build cloth constraint lambda update layout info: {e}")
            })?,
    )
    .map_err(|e| format!("failed to create cloth constraint lambda update layout: {e}"))?;
    let stage = stages.into_iter().next().expect("compute stage present");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .map_err(|e| format!("failed to create cloth constraint lambda update pipeline: {e}"))
}

/// Build the cloth XPBD constraint accumulate compute pipeline
/// (per-particle Δx accumulation; reads Δλ_j from the dlambda SSBO
/// written by the lambda-update pass).
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
            .map_err(|e| format!("failed to build cloth constraint accumulate layout info: {e}"))?,
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

/// Build the cloth XPBD constraint apply compute pipeline
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
/// Dispatched once per frame at the end of the GPU cloth substep loop,
/// before `transform_cs` reads the cloth normal SSBO.
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


/// Control block for `cloth_collide_cs`. Rust mirror of the GLSL
/// `Control` uniform.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClothCollideControl {
    pub particle_count: u32,
    pub collider_count: u32,
    pub margin: f32,
    /// 0 = SDF stage inert (dummy field bound), 1 = sample `sdf`.
    pub has_sdf: u32,
    /// Body-SDF contact radius in metres (mirrors
    /// `ClothSimState::sdf_contact`; 0 disables — belt and braces with
    /// `has_sdf`).
    pub sdf_contact: f32,
    pub _pad: [u32; 2],
}

/// Params UBO for the collide kernel's SDF stage — the grid the splat
/// pass filled (mirror of `SdfGrid` + `BodySdfSplatParams` layout
/// conventions). Avatar-root space, same as the spring solver's field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClothSdfParams {
    /// xyz = grid origin (avatar-root space, metres), w = voxel size.
    pub origin_voxel: [f32; 4],
    /// xyz = grid dims (cells), w = pad.
    pub dims_pad: [u32; 4],
}

/// SSBO row for one collision capsule: `a.xyz`/`b.xyz` = segment
/// endpoints, `a.w` = radius, `b.w` unused (padding).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClothGpuColliderGpu {
    pub a: [f32; 4],
    pub b: [f32; 4],
}

/// Build the cloth collision projection compute pipeline (P3-02 S2.2).
pub fn create_cloth_collide_compute_pipeline(
    device: Arc<Device>,
) -> Result<Arc<ComputePipeline>, String> {
    let cs_module = cloth_collide_cs::load(device.clone())
        .map_err(|e| format!("failed to load cloth collide compute shader: {e}"))?;
    let cs_entry = cs_module
        .entry_point("main")
        .ok_or_else(|| "cloth collide compute shader entry point 'main' not found".to_string())?;
    let stages = [PipelineShaderStageCreateInfo::new(cs_entry)];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| format!("failed to build cloth collide pipeline layout info: {e}"))?,
    )
    .map_err(|e| format!("failed to create cloth collide pipeline layout: {e}"))?;
    let stage = stages.into_iter().next().expect("compute stage present");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .map_err(|e| format!("failed to create cloth collide compute pipeline: {e}"))
}
pub fn create_cloth_bend_apply_compute_pipeline(
    device: Arc<Device>,
) -> Result<Arc<ComputePipeline>, String> {
    let cs_module = cloth_bend_apply_cs::load(device.clone())
        .map_err(|e| format!("failed to load cloth bend apply compute shader: {e}"))?;
    let cs_entry = cs_module
        .entry_point("main")
        .ok_or_else(|| "cloth bend apply compute shader entry point 'main' not found".to_string())?;
    let stages = [PipelineShaderStageCreateInfo::new(cs_entry)];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| format!("failed to build cloth bend apply pipeline layout info: {e}"))?,
    )
    .map_err(|e| format!("failed to create cloth bend apply pipeline layout: {e}"))?;
    let stage = stages.into_iter().next().expect("compute stage present");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .map_err(|e| format!("failed to create cloth bend apply compute pipeline: {e}"))
}

pub fn create_cloth_bend_accumulate_compute_pipeline(
    device: Arc<Device>,
) -> Result<Arc<ComputePipeline>, String> {
    let cs_module = cloth_bend_accumulate_cs::load(device.clone())
        .map_err(|e| format!("failed to load cloth bend accumulate compute shader: {e}"))?;
    let cs_entry = cs_module
        .entry_point("main")
        .ok_or_else(|| "cloth bend accumulate compute shader entry point 'main' not found".to_string())?;
    let stages = [PipelineShaderStageCreateInfo::new(cs_entry)];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| format!("failed to build cloth bend accumulate pipeline layout info: {e}"))?,
    )
    .map_err(|e| format!("failed to create cloth bend accumulate pipeline layout: {e}"))?;
    let stage = stages.into_iter().next().expect("compute stage present");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .map_err(|e| format!("failed to create cloth bend accumulate compute pipeline: {e}"))
}

pub fn create_cloth_bend_update_compute_pipeline(
    device: Arc<Device>,
) -> Result<Arc<ComputePipeline>, String> {
    let cs_module = cloth_bend_update_cs::load(device.clone())
        .map_err(|e| format!("failed to load cloth bend update compute shader: {e}"))?;
    let cs_entry = cs_module
        .entry_point("main")
        .ok_or_else(|| "cloth bend update compute shader entry point 'main' not found".to_string())?;
    let stages = [PipelineShaderStageCreateInfo::new(cs_entry)];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| format!("failed to build cloth bend update pipeline layout info: {e}"))?,
    )
    .map_err(|e| format!("failed to create cloth bend update pipeline layout: {e}"))?;
    let stage = stages.into_iter().next().expect("compute stage present");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .map_err(|e| format!("failed to create cloth bend update compute pipeline: {e}"))
}


/// Control block for the self-collision build/resolve passes. Rust
/// mirror of the GLSL `Control` uniform.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClothSelfColControl {
    pub particle_count: u32,
    pub radius: f32,
    pub table_size: u32,
    pub _pad: u32,
}

/// Cell-count entries in the self-collision hash table. Kept in sync
/// with the `K` constants inside both shaders.
pub const CLOTH_SELFCOL_BUCKET_SLOTS: u32 = 16;
/// Self-collision hash table size. 16384 cells × 16 slots × 4 B = 1 MiB.
pub const CLOTH_SELFCOL_TABLE_SIZE: u32 = 16384;

pub fn create_cloth_selfcol_build_compute_pipeline(
    device: Arc<Device>,
) -> Result<Arc<ComputePipeline>, String> {
    let cs_module = cloth_selfcol_build_cs::load(device.clone())
        .map_err(|e| format!("failed to load cloth selfcol build shader: {e}"))?;
    let cs_entry = cs_module
        .entry_point("main")
        .ok_or_else(|| "cloth selfcol build entry point 'main' not found".to_string())?;
    let stages = [PipelineShaderStageCreateInfo::new(cs_entry)];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| format!("failed to build cloth selfcol build layout info: {e}"))?,
    )
    .map_err(|e| format!("failed to create cloth selfcol build layout: {e}"))?;
    let stage = stages.into_iter().next().expect("compute stage present");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .map_err(|e| format!("failed to create cloth selfcol build pipeline: {e}"))
}

pub fn create_cloth_selfcol_resolve_compute_pipeline(
    device: Arc<Device>,
) -> Result<Arc<ComputePipeline>, String> {
    let cs_module = cloth_selfcol_resolve_cs::load(device.clone())
        .map_err(|e| format!("failed to load cloth selfcol resolve shader: {e}"))?;
    let cs_entry = cs_module
        .entry_point("main")
        .ok_or_else(|| "cloth selfcol resolve entry point 'main' not found".to_string())?;
    let stages = [PipelineShaderStageCreateInfo::new(cs_entry)];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| format!("failed to build cloth selfcol resolve layout info: {e}"))?,
    )
    .map_err(|e| format!("failed to create cloth selfcol resolve layout: {e}"))?;
    let stage = stages.into_iter().next().expect("compute stage present");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .map_err(|e| format!("failed to create cloth selfcol resolve pipeline: {e}"))
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
    //   uint  constraint_count; // offset 4, size 4
    //   float dt;               // offset 8, size 4
    //   uint  _pad;             // offset 12, size 4
    //   total                   // 16 bytes
    #[test]
    fn cloth_constraint_control_matches_std140_layout() {
        use super::ClothConstraintControl;
        assert_eq!(size_of::<ClothConstraintControl>(), 16);
        assert_eq!(offset_of!(ClothConstraintControl, particle_count), 0);
        assert_eq!(offset_of!(ClothConstraintControl, constraint_count), 4);
        assert_eq!(offset_of!(ClothConstraintControl, dt), 8);
        assert_eq!(offset_of!(ClothConstraintControl, _pad), 12);
    }

    // GLSL std140 layout for `TransformControl` UBO in transform_cs.
    // The morph weights moved out of this UBO into the `MorphWeights`
    // SSBO (set 0 binding 9) so the target count is unbounded; what
    // remains is scalars only.
    //   uint vertex_count;     // offset 0,  size 4
    //   uint target_count;     // offset 4,  size 4
    //   uint has_cloth;        // offset 8,  size 4
    //   uint has_cloth_normals;// offset 12, size 4
    //   uint has_skin_anchors; // offset 16, size 4
    //   uvec3 _pad0;           // offset 20, size 12
    //   total                  // 32 bytes
    #[test]
    fn transform_control_matches_std140_layout() {
        use super::TransformControl;
        assert_eq!(size_of::<TransformControl>(), 32);
        assert_eq!(offset_of!(TransformControl, vertex_count), 0);
        assert_eq!(offset_of!(TransformControl, target_count), 4);
        assert_eq!(offset_of!(TransformControl, has_cloth), 8);
        assert_eq!(offset_of!(TransformControl, has_cloth_normals), 12);
        assert_eq!(offset_of!(TransformControl, has_skin_anchors), 16);
        assert_eq!(offset_of!(TransformControl, has_containment), 20);
        assert_eq!(offset_of!(TransformControl, _pad0), 24);
    }
}

/// R8 — CPU mirror of `transform_cs`'s Dual Quaternion Skinning math,
/// used to pin the shader's input contract without a Vulkan device.
/// The GLSL reads matrices column-major (`m[col][row]`); the Rust
/// `Mat4` is row-major, so every index pair is transposed here. These
/// tests exist to separate the INTENDED DQS-vs-LBS differences from
/// INPUT-CONTRACT violations (non-rigid skinning matrices), per the
/// quality-improvement-plan R8 review item.
#[cfg(test)]
mod dqs_contract_tests {
    use crate::asset::Mat4;

    type Q = [f32; 4]; // (x, y, z, w)

    /// Mike Day's robust mat3→quat, transposed to row-major input.
    /// Direct mirror of the GLSL `mat3_to_quat` in `transform_cs`.
    fn mat3_to_quat(r: [[f32; 3]; 3]) -> Q {
        let trace = r[0][0] + r[1][1] + r[2][2];
        if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            [
                (r[2][1] - r[1][2]) / s,
                (r[0][2] - r[2][0]) / s,
                (r[1][0] - r[0][1]) / s,
                0.25 * s,
            ]
        } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
            let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
            [
                0.25 * s,
                (r[0][1] + r[1][0]) / s,
                (r[0][2] + r[2][0]) / s,
                (r[2][1] - r[1][2]) / s,
            ]
        } else if r[1][1] > r[2][2] {
            let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
            [
                (r[0][1] + r[1][0]) / s,
                0.25 * s,
                (r[1][2] + r[2][1]) / s,
                (r[1][0] - r[0][1]) / s,
            ]
        } else {
            let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
            [
                (r[0][2] + r[2][0]) / s,
                (r[1][2] + r[2][1]) / s,
                0.25 * s,
                (r[1][0] - r[0][1]) / s,
            ]
        }
    }

    fn quat_rotate(q: Q, v: [f32; 3]) -> [f32; 3] {
        let (ux, uy, uz) = (q[0], q[1], q[2]);
        let w = q[3];
        let cross_uv = [uy * v[2] - uz * v[1], uz * v[0] - ux * v[2], ux * v[1] - uy * v[0]];
        let t = [
            cross_uv[0] + w * v[0],
            cross_uv[1] + w * v[1],
            cross_uv[2] + w * v[2],
        ];
        let cross_ut = [uy * t[2] - uz * t[1], uz * t[0] - ux * t[2], ux * t[1] - uy * t[0]];
        [
            v[0] + 2.0 * cross_ut[0],
            v[1] + 2.0 * cross_ut[1],
            v[2] + 2.0 * cross_ut[2],
        ]
    }

    fn dot(a: Q, b: Q) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    }

    /// Full mirror of the shader's DQS branch for ONE vertex with the
    /// given joint indices / weights.
    fn dqs_skin(mats: &[Mat4], indices: [u32; 4], weights: [f32; 4], pos: [f32; 3]) -> [f32; 3] {
        const WEIGHT_EPS: f32 = 1.0e-4;
        let mut ref_real = [0.0f32, 0.0, 0.0, 1.0];
        for i in 0..4 {
            if weights[i] > WEIGHT_EPS {
                let m = &mats[indices[i] as usize];
                ref_real = mat3_to_quat([
                    // Transposed: the mirror expects the MATH 3x3
                    // (R[row][col] = m[col][row] in codebase storage).
                    [m[0][0], m[1][0], m[2][0]],
                    [m[0][1], m[1][1], m[2][1]],
                    [m[0][2], m[1][2], m[2][2]],
                ]);
                break;
            }
        }
        let mut acc_real = [0.0f32; 4];
        let mut acc_dual = [0.0f32; 4];
        for i in 0..4 {
            let wi = weights[i];
            if wi <= WEIGHT_EPS {
                continue;
            }
            let m = &mats[indices[i] as usize];
            let mut qr = mat3_to_quat([
                // Transposed: see the ref_real note above.
                [m[0][0], m[1][0], m[2][0]],
                [m[0][1], m[1][1], m[2][1]],
                [m[0][2], m[1][2], m[2][2]],
            ]);
            let s = if dot(qr, ref_real) >= 0.0 { 1.0 } else { -1.0 };
            qr = [qr[0] * s, qr[1] * s, qr[2] * s, qr[3] * s];
            let ti = [m[3][0], m[3][1], m[3][2]];
            let qrxyz = [qr[0], qr[1], qr[2]];
            let qrw = qr[3];
            // dxyz = 0.5*(qrw*ti + cross(ti, qrxyz)); dw = -0.5*dot(ti,qrxyz)
            let cross_t_q = [
                ti[1] * qrxyz[2] - ti[2] * qrxyz[1],
                ti[2] * qrxyz[0] - ti[0] * qrxyz[2],
                ti[0] * qrxyz[1] - ti[1] * qrxyz[0],
            ];
            let dxyz = [
                0.5 * (qrw * ti[0] + cross_t_q[0]),
                0.5 * (qrw * ti[1] + cross_t_q[1]),
                0.5 * (qrw * ti[2] + cross_t_q[2]),
            ];
            let dw = -0.5 * (ti[0] * qrxyz[0] + ti[1] * qrxyz[1] + ti[2] * qrxyz[2]);
            acc_real = [
                acc_real[0] + wi * qr[0],
                acc_real[1] + wi * qr[1],
                acc_real[2] + wi * qr[2],
                acc_real[3] + wi * qr[3],
            ];
            acc_dual = [
                acc_dual[0] + wi * dxyz[0],
                acc_dual[1] + wi * dxyz[1],
                acc_dual[2] + wi * dxyz[2],
                acc_dual[3] + wi * dw,
            ];
        }
        let acc_len = dot(acc_real, acc_real).sqrt();
        assert!(acc_len >= 1.0e-6, "degenerate DQS blend");
        let inv = 1.0 / acc_len;
        let q_real = [acc_real[0] * inv, acc_real[1] * inv, acc_real[2] * inv, acc_real[3] * inv];
        let q_dual = [acc_dual[0] * inv, acc_dual[1] * inv, acc_dual[2] * inv, acc_dual[3] * inv];
        let rqxyz = [-q_real[0], -q_real[1], -q_real[2]];
        let rqw = q_real[3];
        let c = [
            q_dual[1] * rqxyz[2] - q_dual[2] * rqxyz[1],
            q_dual[2] * rqxyz[0] - q_dual[0] * rqxyz[2],
            q_dual[0] * rqxyz[1] - q_dual[1] * rqxyz[0],
        ];
        let t_recovered = [
            2.0 * (q_dual[3] * rqxyz[0] + rqw * q_dual[0] + c[0]),
            2.0 * (q_dual[3] * rqxyz[1] + rqw * q_dual[1] + c[1]),
            2.0 * (q_dual[3] * rqxyz[2] + rqw * q_dual[2] + c[2]),
        ];
        let rotated = quat_rotate(q_real, pos);
        [
            rotated[0] + t_recovered[0],
            rotated[1] + t_recovered[1],
            rotated[2] + t_recovered[2],
        ]
    }

    /// Row-major rigid matrix from an axis-angle rotation about Y plus
    /// a translation, in the CODEBASE `Mat4` convention (established by
    /// `gpu_pin_targets` / `apply_pin_targets`: `m[col][row]` — the
    /// math matrix M is stored TRANSPOSED, translation in
    /// `m[3][0..3]`). This is exactly the memory layout the GLSL sees.
    fn rigid_y(deg: f32, t: [f32; 3]) -> Mat4 {
        let a = deg.to_radians();
        let (s, c) = (a.sin(), a.cos());
        // M (math, row-major) = [[c,0,s],[0,1,0],[-s,0,c]]; store m[i][j] = M[j][i].
        [
            [c, 0.0, -s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [s, 0.0, c, 0.0],
            [t[0], t[1], t[2], 1.0],
        ]
    }

    /// Apply a codebase-convention `Mat4` to a point (same arithmetic
    /// as `gpu_pin_targets`).
    fn mat4_apply(m: &Mat4, v: [f32; 3]) -> [f32; 3] {
        [
            m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0],
            m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1],
            m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2],
        ]
    }

    /// Contract 1: ONE rigid joint → DQS reproduces the matrix
    /// transform exactly (quaternion extraction is lossless for rigid
    /// input; translation comes from the 4th row).
    #[test]
    fn dqs_matches_matrix_for_single_rigid_joint() {
        let identity = crate::asset::identity_matrix();
        let mats = vec![rigid_y(30.0, [0.4, -0.2, 1.5]), identity];
        let got = dqs_skin(&mats, [0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0], [0.3, -0.7, 1.1]);
        let want = mat4_apply(&mats[0], [0.3, -0.7, 1.1]);
        for c in 0..3 {
            assert!(
                (got[c] - want[c]).abs() < 1.0e-4,
                "DQS != LBS on rigid input: got {got:?} want {want:?}"
            );
        }
    }

    /// Contract 2: blending rigid joints yields a RIGID result — the
    /// blended unit-quaternion transform preserves distances between
    /// two probe points (no shear, no scale leakage, antipodality fix
    /// keeps the blend well-defined).
    #[test]
    fn dqs_blend_of_rigid_joints_preserves_distance() {
        let mats = vec![
            rigid_y(40.0, [0.0, 0.0, 0.0]),
            rigid_y(-25.0, [0.05, 0.02, -0.03]),
        ];
        // Two points skin both joints with complementary weights, so
        // they see the SAME blended transform only if we keep the
        // weights fixed — instead skin ONE extra point through the same
        // weights by re-running the mirror on a second probe. The
        // mirror takes the point as a parameter.
        let p1 = [0.3f32, -0.7, 1.1];
        let p2 = [-0.4f32, 0.5, 0.2];
        let a1 = dqs_skin(&mats, [0, 1, 0, 0], [0.6, 0.4, 0.0, 0.0], p1);
        let a2 = dqs_skin(&mats, [0, 1, 0, 0], [0.6, 0.4, 0.0, 0.0], p2);
        let rest = (p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2);
        let deformed = (a1[0] - a2[0]).powi(2) + (a1[1] - a2[1]).powi(2) + (a1[2] - a2[2]).powi(2);
        assert!(
            (rest - deformed).abs() < 1.0e-2,
            "blended DQS must be rigid (distance preserved): rest²={rest} deformed²={deformed}"
        );
    }

    /// Contract 3 (the R8 input-violation probe): a NON-UNIFORM SCALE
    /// in the skinning matrix is silently dropped by the DQS rotation
    /// extraction — the shader rotates but does not scale, while the
    /// CPU LBS probes (and the clearance / auto-cloth rest-position
    /// recipes) apply the full matrix. Any asset whose import leaks a
    /// scaled node therefore renders DQS-displaced vertices against
    /// LBS-shaped collision/clearance surfaces. This test pins the
    /// divergence so the import-side fix (scale stripping at load)
    /// has a regression anchor.
    #[test]
    fn dqs_drops_nonuniform_scale_unlike_lbs() {
        // Codebase convention (m[col][row]): math S = x-scale 2 +
        // translation 0.1, stored transposed with translation row 3.
        let scale: Mat4 = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.1, 0.0, 0.0, 1.0],
        ];
        let probe = [0.3f32, -0.7, 1.1];
        let got = dqs_skin(&[scale.clone()], [0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0], probe);
        let lbs = mat4_apply(&scale, probe);
        // LBS doubles x; DQS must NOT.
        assert!(
            (lbs[0] - 0.7).abs() < 1.0e-5,
            "sanity: LBS applies the scale (x 0.3 → 0.7 incl. translation)"
        );
        assert!(
            (got[0] - lbs[0]).abs() > 1.0e-2,
            "DQS is expected to diverge from LBS under non-uniform scale; got {got:?} lbs {lbs:?}"
        );
        // And the divergence stays bounded (rotation-magnitude, not a
        // blow-up).
        assert!(got.iter().all(|c| c.is_finite() && c.abs() < 10.0));
    }

    /// Contract 4 (R8, real asset): Yumeka's per-frame skinning
    /// matrices must be RIGID (orthonormal 3×3, det +1) — the DQS
    /// shader's input contract. A failure here means the import path
    /// leaks scale/mirror into `global_transforms` or the inverse-binds
    /// and every DQS-vs-LBS mismatch downstream is an import bug, not
    /// a skinning bug.
    #[test]
    fn yumeka_skinning_matrices_are_rigid() {
        let pinned = "sample_data/YUMEKA_v1.0.1/FBX/Yumeka_v1.0.fbx";
        if !std::path::Path::new(pinned).exists() {
            return;
        }
        let loader = crate::asset::fbx::FbxAssetLoader::new();
        let asset = loader.load(pinned).expect("load Yumeka");
        let mut avatar = crate::avatar::AvatarInstance::new(
            crate::avatar::AvatarInstanceId(1),
            asset,
        );
        // The base pose is what every load path guarantees; the
        // rigidity contract must hold there (and is re-checked on real
        // poses by the replay benches).
        avatar.build_base_pose();
        avatar.compute_global_pose();
        avatar.build_skinning_matrices();

        assert!(avatar.pose.skinning_matrices.len() > 10);
        for (i, m) in avatar.pose.skinning_matrices.iter().enumerate() {
            // Row orthonormality: R·Rᵀ = I.
            for r in 0..3 {
                for c in 0..3 {
                    let dot = (0..3).map(|k| m[r][k] * m[c][k]).sum::<f32>();
                    let want = if r == c { 1.0 } else { 0.0 };
                    assert!(
                        (dot - want).abs() < 1.0e-3,
                        "skinning matrix {i} 3x3 not orthonormal: R·Rᵀ[{r}][{c}] = {dot}"
                    );
                }
            }
            // Rigid, not mirrored.
            let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
            assert!(
                (det - 1.0).abs() < 1.0e-3,
                "skinning matrix {i} determinant {det} (expected +1: rigid, unmirrored)"
            );
        }
    }
}
    

/// Control block for the body-SDF splat pass. Rust mirror of the GLSL
/// `Params` uniform in `body_sdf_splat_cs` (std140: every member on a
/// 16-byte boundary). Grid geometry must match
/// `simulation::sdf::SdfGrid` exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BodySdfSplatParams {
    /// xyz = grid dims (cells), w = triangle count.
    pub dims_tri: [u32; 4],
    /// xyz = grid origin (avatar-root space), w = metres per cell.
    pub origin_voxel: [f32; 4],
    /// x = splat shell in metres; yzw pad to 16 B.
    pub shell_pad: [f32; 4],
}

/// Build the compute pipeline that splats the freshly skinned body
/// primitive into the avatar-root-space distance field the spring
/// solver resolves hair against. See `body_sdf_splat_cs` and
/// `simulation/sdf.rs` for the field contract.
pub fn create_body_sdf_splat_compute_pipeline(
    device: Arc<Device>,
) -> Result<Arc<ComputePipeline>, String> {
    let cs_module = body_sdf_splat_cs::load(device.clone())
        .map_err(|e| format!("failed to load body SDF splat compute shader: {e}"))?;
    let cs_entry = cs_module
        .entry_point("main")
        .ok_or_else(|| "body SDF splat compute shader entry point 'main' not found".to_string())?;
    let stages = [PipelineShaderStageCreateInfo::new(cs_entry)];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| format!("failed to build body SDF splat pipeline layout info: {e}"))?,
    )
    .map_err(|e| format!("failed to create body SDF splat pipeline layout: {e}"))?;
    let stage = stages.into_iter().next().expect("compute stage present");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .map_err(|e| format!("failed to create body SDF splat compute pipeline: {e}"))
}
