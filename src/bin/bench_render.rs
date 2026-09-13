//! Render-pipeline benchmark.
//!
//! Renders the same avatar N times through the full `VulkanRenderer`
//! path (compute prepass + draw + post + CPU readback) with per-frame
//! expression weights and a small head motion so the morph gather and
//! skinning uploads do real work every frame. Reports per-frame wall
//! time (min / median / mean / max). Because `render()` submits
//! asynchronously and waits on the PREVIOUS frame's fence internally,
//! the per-iteration wall time approximates max(CPU submit, GPU exec)
//! — the effective frame cost.
//!
//! Attribution runs: set `VULVATAR_SKIP_COMPUTE=1` (skips compute
//! dispatch recording; draws reuse the warmed VBOs) or
//! `VULVATAR_SKIP_DRAW=1` (skips draw recording) and diff against the
//! full-run median.
//!
//! Usage:
//!   cargo run --bin bench_render -- <avatar.fbx|avatar.vrm> [frames] [width] [height]
//!
//! Results print to stdout; no files are written.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use vulvatar_lib::app::ViewportCamera;
use vulvatar_lib::asset::vrm::VrmAssetLoader;
use vulvatar_lib::avatar::expressions::ResolvedExpressionWeight;
use vulvatar_lib::avatar::{AvatarInstance, AvatarInstanceId};
use vulvatar_lib::renderer::frame_input::{
    CameraState, LightingState, OutputTargetRequest, RenderAvatarInstance, RenderColorSpace,
    RenderExportMode, RenderFrameInput, RenderMeshInstance, RenderOutputAlpha,
};
use vulvatar_lib::renderer::VulkanRenderer;

fn main() -> Result<(), String> {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let input_path = PathBuf::from(
        args.next()
            .ok_or("usage: bench_render <avatar.fbx|avatar.vrm> [frames] [width] [height]")?,
    );
    let frames: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(300);
    let extent = [
        args.next().and_then(|s| s.parse().ok()).unwrap_or(1920),
        args.next().and_then(|s| s.parse().ok()).unwrap_or(1080),
    ];

    let path_str = input_path
        .to_str()
        .ok_or_else(|| format!("invalid input path: {}", input_path.display()))?;
    let is_fbx = input_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("fbx"))
        .unwrap_or(false);
    let asset = if is_fbx {
        vulvatar_lib::asset::fbx::FbxAssetLoader::new()
            .load(path_str)
            .map_err(|e| format!("failed to load FBX '{}': {}", input_path.display(), e))?
    } else {
        VrmAssetLoader::new()
            .load(path_str)
            .map_err(|e| format!("failed to load VRM '{}': {}", input_path.display(), e))?
    };

    let mut avatar = AvatarInstance::new(AvatarInstanceId(1), Arc::clone(&asset));
    avatar.build_base_pose();
    avatar.compute_global_pose();
    avatar.build_skinning_matrices();

    // Animate a handful of expressions (first few presets) so the
    // sparse morph gather runs with realistic active-target counts.
    let expr_count: usize = std::env::var("VULVATAR_BENCH_EXPRS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let expr_names: Vec<String> = asset
        .default_expressions
        .expressions
        .iter()
        .take(expr_count)
        .map(|e| e.name.clone())
        .collect();
    println!(
        "animating {} expressions: {:?}",
        expr_names.len(),
        expr_names
    );
    for name in &expr_names {
        avatar.expression_weights.push(ResolvedExpressionWeight {
            name: name.clone(),
            weight: 0.0,
        });
    }

    let head_yaw_node = asset
        .skeleton
        .nodes
        .iter()
        .position(|n| {
            n.humanoid_bone == Some(vulvatar_lib::asset::HumanoidBone::Head)
                || n.name.eq_ignore_ascii_case("Head")
        })
        .or_else(|| {
            asset
                .skeleton
                .nodes
                .iter()
                .position(|n| n.name.to_lowercase().ends_with("_head"))
        });

    let mesh_instances: Vec<RenderMeshInstance> = asset
        .meshes
        .iter()
        .flat_map(|mesh| {
            mesh.primitives
                .iter()
                .map(|prim| RenderMeshInstance::from_primitive(&avatar, mesh.id, prim))
        })
        .collect();
    println!(
        "avatar: {} primitives / {} verts total / {} materials",
        mesh_instances.len(),
        mesh_instances
            .iter()
            .filter_map(|mi| mi.primitive_data.as_ref().map(|p| p.vertex_count as usize))
            .sum::<usize>(),
        asset.materials.len()
    );
    for (name, count, verts) in asset
        .meshes
        .iter()
        .flat_map(|m| {
            m.primitives
                .iter()
                .map(move |p| (m.name.as_str(), p.morph_targets.len(), p.vertex_count))
        })
        .filter(|(_, c, _)| *c > 0)
    {
        println!("  morph prim '{}' verts={} targets={}", name, verts, count);
    }

    let camera = ViewportCamera::default();
    let (view, eye_pos) = build_view_matrix(&camera);
    let projection = build_projection_matrix(
        camera.fov_deg,
        extent[0] as f32 / extent[1].max(1) as f32,
        0.1,
        1000.0,
    );

    let mut renderer = VulkanRenderer::new();
    renderer.initialize();

    let mut samples_ms: Vec<f32> = Vec::with_capacity(frames);
    let warmup = 30.min(frames / 3);
    for i in 0..(frames + warmup) {
        // Expression animation: out-of-phase sines, half amplitude.
        for (k, w) in avatar.expression_weights.iter_mut().enumerate() {
            w.weight = 0.5 * ((i as f32 * 0.13 + k as f32 * 1.7).sin() * 0.5 + 0.5);
        }
        // Head sway so skinning matrices change every frame.
        if let Some(node) = head_yaw_node {
            let a = (i as f32 * 0.05).sin() * 10.0_f32.to_radians();
            let (s, c) = ((a * 0.5).sin(), (a * 0.5).cos());
            let q = [0.0, s, 0.0, c];
            avatar.pose.local_transforms[node].rotation = vulvatar_lib::math_utils::quat_mul(
                &q,
                &asset.skeleton.nodes[node].rest_local.rotation,
            );
            avatar.compute_global_pose();
            avatar.build_skinning_matrices();
        }

        let mesh_instances: Vec<RenderMeshInstance> = asset
            .meshes
            .iter()
            .flat_map(|mesh| {
                mesh.primitives
                    .iter()
                    .map(|prim| RenderMeshInstance::from_primitive(&avatar, mesh.id, prim))
            })
            .collect();

        // bg-UBO validation knob: render with the generative background
        // enabled (`VULVATAR_BENCH_BG=1`) so the animated uniform-ring path
        // (and its interaction with the command-buffer cache) is exercised.
        let bench_bg = std::env::var("VULVATAR_BENCH_BG").map_or(false, |v| v == "1");
        let mut generative_background =
            vulvatar_lib::renderer::frame_input::GenerativeBackgroundSettings::default();
        if bench_bg {
            generative_background.enabled = true;
        }

        let frame_input = RenderFrameInput {
            camera: CameraState {
                view,
                projection,
                position_ws: eye_pos,
                viewport_extent: extent,
            },
            lighting: LightingState::default(),
            instances: vec![RenderAvatarInstance {
                instance_id: avatar.id,
                world_transform: avatar.world_transform.clone(),
                mesh_instances,
                skinning_matrices: avatar.pose.skinning_matrices.clone(),
                cloth_deforms: Vec::new(),
    body_sdf: None,
            debug_flags: Default::default(),
            }],
            output_request: OutputTargetRequest {
                preview_enabled: true,
                output_enabled: true,
                extent,
                color_space: RenderColorSpace::Srgb,
                alpha_mode: RenderOutputAlpha::Premultiplied,
                export_mode: RenderExportMode::CpuReadback,
                msaa: vulvatar_lib::renderer::frame_input::MsaaMode::Off,
            },
            background_image_path: None,
            show_ground_grid: false,
            background_color: [0.1, 0.1, 0.1],
            transparent_background: true,
            avatar_opacity: 1.0,
            bloom: Default::default(),
            generative_background,
            background_tracking: Default::default(),
            time_seconds: i as f32 / 60.0,
        };

        let t0 = Instant::now();
        let result = renderer
            .render(&frame_input)
            .map_err(|e| format!("render failed on frame {i}: {e}"))?;
        let dt = t0.elapsed().as_secs_f32() * 1000.0;
        // Equivalence-check hook: dump every harvested frame's pixels
        // (`VULVATAR_BENCH_DUMP=<dir>`) so runs with/without a renderer
        // optimization (e.g. `VULVATAR_CB_CACHE=0`, `VULVATAR_PIXEL_POOL=0`)
        // can be compared byte-for-byte. The animation is deterministic in
        // the frame index, so identical binaries+settings must produce
        // identical dumps.
        if let Ok(dir) = std::env::var("VULVATAR_BENCH_DUMP") {
            if let Some(exported) = result.exported_frame.as_ref() {
                if let Some(pixels) = exported.cpu_pixel_data() {
                    let _ = std::fs::create_dir_all(&dir);
                    let _ = std::fs::write(
                        std::path::Path::new(&dir).join(format!("f{:05}.rgba", i)),
                        &pixels[..],
                    );
                }
            }
        }
        if i >= warmup {
            samples_ms.push(dt);
        }
    }

    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples_ms.len();
    println!(
        "frames={} (warmup {}) extent={}x{} skip_compute={} skip_draw={}",
        n,
        warmup,
        extent[0],
        extent[1],
        std::env::var("VULVATAR_SKIP_COMPUTE").unwrap_or_default(),
        std::env::var("VULVATAR_SKIP_DRAW").unwrap_or_default(),
    );
    println!(
        "min={:.2}ms median={:.2}ms mean={:.2}ms max={:.2}ms (=> {:.1} fps median)",
        samples_ms[0],
        samples_ms[n / 2],
        samples_ms.iter().sum::<f32>() / n as f32,
        samples_ms[n - 1],
        1000.0 / samples_ms[n / 2]
    );
    Ok(())
}

fn build_view_matrix(cam: &ViewportCamera) -> (vulvatar_lib::asset::Mat4, [f32; 3]) {
    let yaw = cam.yaw_deg.to_radians();
    let pitch = cam.pitch_deg.to_radians();
    let (sy, cy) = (yaw.sin(), yaw.cos());
    let (sp, cp) = (pitch.sin(), pitch.cos());

    let right = [cy, 0.0, -sy];
    let up = [-sy * sp, cp, -cy * sp];

    let wx = cam.pan[0] * right[0] + cam.pan[1] * up[0];
    let wy = cam.pan[0] * right[1] + cam.pan[1] * up[1];
    let wz = cam.pan[0] * right[2] + cam.pan[1] * up[2];

    let eye_x = cam.distance * cp * sy + wx;
    let eye_y = cam.distance * sp + wy;
    let eye_z = cam.distance * cp * cy + wz;

    let target = [wx, wy, wz];

    let fwd = [target[0] - eye_x, target[1] - eye_y, target[2] - eye_z];
    let len = (fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2])
        .sqrt()
        .max(1e-6);
    let f = [fwd[0] / len, fwd[1] / len, fwd[2] / len];

    let world_up = [0.0f32, 1.0, 0.0];
    let r = [
        f[1] * world_up[2] - f[2] * world_up[1],
        f[2] * world_up[0] - f[0] * world_up[2],
        f[0] * world_up[1] - f[1] * world_up[0],
    ];
    let rlen = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt().max(1e-6);
    let r = [r[0] / rlen, r[1] / rlen, r[2] / rlen];

    let u = [
        r[1] * f[2] - r[2] * f[1],
        r[2] * f[0] - r[0] * f[2],
        r[0] * f[1] - r[1] * f[0],
    ];

    (
        [
            [
                r[0],
                r[1],
                r[2],
                -(r[0] * eye_x + r[1] * eye_y + r[2] * eye_z),
            ],
            [
                u[0],
                u[1],
                u[2],
                -(u[0] * eye_x + u[1] * eye_y + u[2] * eye_z),
            ],
            [
                -f[0],
                -f[1],
                -f[2],
                f[0] * eye_x + f[1] * eye_y + f[2] * eye_z,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [eye_x, eye_y, eye_z],
    )
}
fn build_projection_matrix(
    fov_deg: f32,
    aspect: f32,
    near: f32,
    far: f32,
) -> vulvatar_lib::asset::Mat4 {
    let mut m = vulvatar_lib::asset::identity_matrix();
    let f = 1.0 / (fov_deg.to_radians() * 0.5).tan();
    m[0][0] = f / aspect;
    m[1][1] = f;
    m[2][2] = far / (near - far);
    m[2][3] = -1.0;
    m[3][2] = far * near / (near - far);
    m[3][3] = 0.0;
    m
}
