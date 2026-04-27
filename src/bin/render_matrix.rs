//! Output color-space validation CLI.
//!
//! Stage 5 of the output color-space work: render the same VRM scene
//! across the 2x2 matrix of (color space) x (transparent background) and
//! save each result as PNG. Used as a manual smoke test before
//! shipping changes to the colour pipeline — looking at the four files
//! side-by-side catches double-encode / no-encode mismatches that the
//! type system can't.
//!
//! Usage:
//!   cargo run --bin render_matrix -- <input.vrm> [output_dir]
//!
//! Writes:
//!   <output_dir>/srgb_opaque.png
//!   <output_dir>/srgb_transparent.png
//!   <output_dir>/linear_opaque.png
//!   <output_dir>/linear_transparent.png
//!
//! Default output_dir is the input VRM's parent dir + `_color_matrix`.

use std::path::PathBuf;
use std::sync::Arc;

use image::ImageBuffer;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulvatar_lib::app::ViewportCamera;
use vulvatar_lib::asset::vrm::VrmAssetLoader;
use vulvatar_lib::avatar::{AvatarInstance, AvatarInstanceId};
use vulvatar_lib::renderer::frame_input::{
    CameraState, LightingState, OutputTargetRequest, RenderAlphaMode, RenderAvatarInstance,
    RenderColorSpace, RenderCullMode, RenderDebugFlags, RenderExportMode, RenderFrameInput,
    RenderMeshInstance, RenderOutputAlpha,
};
use vulvatar_lib::renderer::material::{MaterialShaderMode, MaterialUploadRequest};
use vulvatar_lib::renderer::pipeline::GpuVertex;
use vulvatar_lib::renderer::VulkanRenderer;

fn main() -> Result<(), String> {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let input_path = args
        .next()
        .ok_or_else(|| "usage: render_matrix <input.vrm> [output_dir]".to_string())?;
    let input_path = PathBuf::from(input_path);

    let output_dir = match args.next() {
        Some(d) => PathBuf::from(d),
        None => {
            let stem = input_path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "output".to_string());
            input_path
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."))
                .join(format!("{}_color_matrix", stem))
        }
    };

    std::fs::create_dir_all(&output_dir)
        .map_err(|e| format!("failed to create output dir '{}': {e}", output_dir.display()))?;

    let loader = VrmAssetLoader::new();
    eprintln!("loading {}", input_path.display());
    let asset = loader
        .load(
            input_path
                .to_str()
                .ok_or_else(|| format!("invalid input path: {}", input_path.display()))?,
        )
        .map_err(|e| format!("failed to load VRM '{}': {}", input_path.display(), e))?;

    let mut avatar = AvatarInstance::new(AvatarInstanceId(1), Arc::clone(&asset));
    avatar.build_base_pose();
    avatar.compute_global_pose();
    avatar.build_skinning_matrices();

    eprintln!("gpu_vertex={:?}", GpuVertex::per_vertex());

    let extent = [1024u32, 1024];
    let mut renderer = VulkanRenderer::new();
    renderer.initialize();

    // Two cells per axis × two axes → four renders. Order is important:
    // we use the same renderer instance, so a switch on either axis
    // exercises the apply_color_space rebuild path.
    let cells = [
        ("srgb_opaque", RenderColorSpace::Srgb, false),
        ("srgb_transparent", RenderColorSpace::Srgb, true),
        ("linear_opaque", RenderColorSpace::LinearSrgb, false),
        ("linear_transparent", RenderColorSpace::LinearSrgb, true),
    ];

    for (label, color_space, transparent) in cells {
        let frame_input =
            build_frame_input(&avatar, extent, color_space.clone(), transparent);
        // Two render() invocations per cell: the first kicks off a
        // pipelined CPU readback, the second harvests it. Without the
        // second call the first cell only produces an empty result.
        let _warm = renderer
            .render(&frame_input)
            .map_err(|e| format!("warm-up render '{}' failed: {e}", label))?;
        let result = renderer
            .render(&frame_input)
            .map_err(|e| format!("render '{}' failed: {e}", label))?;
        let pixels = result
            .exported_frame
            .as_ref()
            .and_then(|f| f.cpu_pixel_data())
            .ok_or_else(|| format!("render '{}' returned no CPU pixel data", label))?;

        let path = output_dir.join(format!("{}.png", label));
        save_png(&path, extent, pixels.as_slice())?;
        let stats = compute_stats(pixels.as_slice());
        println!(
            "{:<22} → {} avg_rgb=({:.1},{:.1},{:.1})",
            label,
            path.display(),
            stats.avg[0],
            stats.avg[1],
            stats.avg[2],
        );
    }

    println!("matrix written to {}", output_dir.display());
    Ok(())
}

fn build_frame_input(
    avatar: &AvatarInstance,
    extent: [u32; 2],
    color_space: RenderColorSpace,
    transparent_background: bool,
) -> RenderFrameInput {
    let mesh_instances: Vec<RenderMeshInstance> = avatar
        .asset
        .meshes
        .iter()
        .flat_map(|mesh| {
            mesh.primitives.iter().map(|prim| {
                let material_asset = avatar
                    .asset
                    .materials
                    .iter()
                    .find(|m| m.id == prim.material_id);
                let mut material_binding = material_asset
                    .map(MaterialUploadRequest::from_asset_material)
                    .unwrap_or_else(MaterialUploadRequest::default_material);
                material_binding.mode = MaterialShaderMode::ToonLike;

                let alpha_mode = match material_binding.alpha_mode {
                    vulvatar_lib::asset::AlphaMode::Opaque => RenderAlphaMode::Opaque,
                    vulvatar_lib::asset::AlphaMode::Mask(_) => RenderAlphaMode::Cutout,
                    vulvatar_lib::asset::AlphaMode::Blend => RenderAlphaMode::Blend,
                };
                let cull_mode = if material_binding.double_sided {
                    RenderCullMode::DoubleSided
                } else {
                    RenderCullMode::BackFace
                };

                RenderMeshInstance {
                    mesh_id: mesh.id,
                    primitive_id: prim.id,
                    material_binding,
                    bounds: prim.bounds,
                    alpha_mode,
                    cull_mode,
                    outline: Default::default(),
                    primitive_data: Some(Arc::new(prim.clone())),
                    morph_weights: Vec::new(),
                }
            })
        })
        .collect();

    let camera = ViewportCamera {
        distance: 1.5,
        pan: [0.0, -0.85],
        ..ViewportCamera::default()
    };
    let (view, eye_pos) = build_view_matrix(&camera);
    let projection = build_projection_matrix(
        camera.fov_deg,
        extent[0] as f32 / extent[1].max(1) as f32,
        0.1,
        1000.0,
    );

    let alpha_mode = if transparent_background {
        RenderOutputAlpha::Premultiplied
    } else {
        RenderOutputAlpha::Opaque
    };

    RenderFrameInput {
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
            cloth_deform: None,
            debug_flags: RenderDebugFlags::default(),
        }],
        output_request: OutputTargetRequest {
            preview_enabled: true,
            output_enabled: true,
            extent,
            color_space,
            alpha_mode,
            export_mode: RenderExportMode::CpuReadback,
        },
        background_image_path: None,
        show_ground_grid: false,
        background_color: [0.1, 0.1, 0.1],
        transparent_background,
    }
}

fn build_view_matrix(cam: &ViewportCamera) -> (vulvatar_lib::asset::Mat4, [f32; 3]) {
    let yaw = cam.yaw_deg.to_radians();
    let pitch = cam.pitch_deg.to_radians();
    let (sy, cy) = (yaw.sin(), yaw.cos());
    let (sp, cp) = (pitch.sin(), pitch.cos());

    let wx = cam.pan[0] * cy;
    let wy = cam.pan[1];
    let wz = cam.pan[0] * (-sy);

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
            [r[0], r[1], r[2], -(r[0] * eye_x + r[1] * eye_y + r[2] * eye_z)],
            [u[0], u[1], u[2], -(u[0] * eye_x + u[1] * eye_y + u[2] * eye_z)],
            [-f[0], -f[1], -f[2], (f[0] * eye_x + f[1] * eye_y + f[2] * eye_z)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [eye_x, eye_y, eye_z],
    )
}

fn build_projection_matrix(fov_deg: f32, aspect: f32, near: f32, far: f32) -> vulvatar_lib::asset::Mat4 {
    let f = 1.0 / (fov_deg.to_radians() * 0.5).tan();
    let a = far / (near - far);
    let b = far * near / (near - far);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0],
        [0.0, 0.0, a, b],
        [0.0, 0.0, -1.0, 0.0],
    ]
}

fn save_png(path: &PathBuf, extent: [u32; 2], pixels: &[u8]) -> Result<(), String> {
    let img: ImageBuffer<image::Rgba<u8>, _> =
        ImageBuffer::from_raw(extent[0], extent[1], pixels.to_vec())
            .ok_or_else(|| format!("PNG buffer construction failed for {}", path.display()))?;
    img.save(path)
        .map_err(|e| format!("failed to save '{}': {e}", path.display()))
}

struct Stats {
    avg: [f32; 3],
}

fn compute_stats(pixels: &[u8]) -> Stats {
    let pixel_count = (pixels.len() / 4).max(1) as f32;
    let mut sum = [0.0f32; 3];
    for px in pixels.chunks_exact(4) {
        sum[0] += px[0] as f32;
        sum[1] += px[1] as f32;
        sum[2] += px[2] as f32;
    }
    Stats {
        avg: [
            sum[0] / pixel_count,
            sum[1] / pixel_count,
            sum[2] / pixel_count,
        ],
    }
}
