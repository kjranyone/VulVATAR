use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::collections::HashSet;

use image::ImageBuffer;
use vulvatar::app::ViewportCamera;
use vulvatar::asset::vrm::VrmAssetLoader;
use vulvatar::avatar::{AvatarInstance, AvatarInstanceId};
use vulvatar::renderer::pipeline::GpuVertex;
use vulvatar::renderer::frame_input::{
    CameraState, LightingState, OutputTargetRequest, RenderAlphaMode, RenderAvatarInstance,
    RenderCullMode, RenderDebugFlags, RenderFrameInput, RenderMeshInstance, RenderOutputAlpha,
};
use vulvatar::renderer::material::{
    MaterialDebugView, MaterialShaderMode, MaterialUploadRequest,
};
use vulvatar::renderer::VulkanRenderer;
use vulkano::pipeline::graphics::vertex_input::Vertex;

fn main() -> Result<(), String> {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let input_path = args
        .next()
        .ok_or_else(|| "usage: diagnose_vrm <input.vrm> [output.png] [toon|simple|unlit]".to_string())?;
    let output_path = args.next();
    let mode_arg = args.next();
    let atlas_dump = matches!(mode_arg.as_deref(), Some("atlas"));
    let (material_mode, debug_view) = parse_mode(mode_arg.as_deref())?;
    let yaw_deg = args
        .next()
        .as_deref()
        .map(|s| {
            s.parse::<f32>()
                .map_err(|e| format!("invalid yaw '{}': {}", s, e))
        })
        .transpose()?
        .unwrap_or(0.0);
    let material_filter = args.next();
    let disable_skinning = args
        .next()
        .as_deref()
        .map(|s| matches!(s, "noskin" | "identity-skin"))
        .unwrap_or(false);

    let input_path = PathBuf::from(input_path);
    let output_path = output_path
        .map(PathBuf::from)
        .unwrap_or_else(|| default_output_path(&input_path, &material_mode, &debug_view));

    let loader = VrmAssetLoader::new();
    eprintln!("gpu_vertex={:?}", GpuVertex::per_vertex());
    let asset = loader
        .load(
            input_path
                .to_str()
                .ok_or_else(|| format!("invalid input path: {}", input_path.display()))?,
        )
        .map_err(|e| format!("failed to load VRM '{}': {}", input_path.display(), e))?;
    print_uv_stats(&asset, material_filter.as_deref());
    if atlas_dump {
        dump_first_matching_texture(&asset, &output_path, material_filter.as_deref())?;
        println!("saved_atlas={}", output_path.display());
        return Ok(());
    }

    let mut avatar = AvatarInstance::new(AvatarInstanceId(1), Arc::clone(&asset));
    avatar.build_base_pose();
    avatar.compute_global_pose();
    avatar.build_skinning_matrices();

    let extent = [1024, 1024];
    let frame_input = build_frame_input(
        &avatar,
        extent,
        material_mode.clone(),
        debug_view.clone(),
        yaw_deg,
        material_filter.as_deref(),
        disable_skinning,
    );

    let mut renderer = VulkanRenderer::new();
    renderer.initialize();
    let render_result = renderer
        .render(&frame_input)
        .map_err(|e| format!("failed to render frame: {e}"))?;
    let pixels = render_result
        .exported_frame
        .as_ref()
        .and_then(|frame| frame.cpu_pixel_data())
        .ok_or_else(|| "render returned no CPU pixel data".to_string())?;

    save_png(&output_path, extent, pixels.as_slice())?;

    let stats = compute_stats(pixels.as_slice());
    println!(
        "saved={} mode={:?} debug={:?} size={}x{} avg_rgb=({:.1},{:.1},{:.1}) std_rgb=({:.1},{:.1},{:.1})",
        output_path.display(),
        material_mode,
        debug_view,
        extent[0],
        extent[1],
        stats.avg[0],
        stats.avg[1],
        stats.avg[2],
        stats.stddev[0],
        stats.stddev[1],
        stats.stddev[2],
    );

    Ok(())
}

fn parse_mode(mode: Option<&str>) -> Result<(MaterialShaderMode, MaterialDebugView), String> {
    match mode.unwrap_or("toon") {
        "toon" | "toony" => Ok((MaterialShaderMode::ToonLike, MaterialDebugView::None)),
        "simple" | "lit" => Ok((MaterialShaderMode::SimpleLit, MaterialDebugView::None)),
        "unlit" => Ok((MaterialShaderMode::Unlit, MaterialDebugView::None)),
        "uv" => Ok((MaterialShaderMode::Unlit, MaterialDebugView::Uv)),
        "tex" | "texture" => Ok((MaterialShaderMode::Unlit, MaterialDebugView::BaseTexture)),
        other => Err(format!(
        "unsupported mode '{}', expected one of: toon, simple, unlit, uv, tex, atlas",
        other
    )),
    }
}

fn default_output_path(
    input_path: &Path,
    material_mode: &MaterialShaderMode,
    debug_view: &MaterialDebugView,
) -> PathBuf {
    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("avatar");
    let mode = match debug_view {
        MaterialDebugView::Uv => "uv",
        MaterialDebugView::BaseTexture => "tex",
        MaterialDebugView::None => match material_mode {
        MaterialShaderMode::Unlit => "unlit",
        MaterialShaderMode::SimpleLit => "simple",
        MaterialShaderMode::ToonLike => "toon",
        },
    };
    PathBuf::from("diagnostics").join(format!("{stem}_{mode}.png"))
}

fn dump_first_matching_texture(
    asset: &vulvatar::asset::AvatarAsset,
    output_path: &Path,
    material_filter: Option<&str>,
) -> Result<(), String> {
    let filter = material_filter.map(|s| s.to_ascii_lowercase());
    let material = asset
        .materials
        .iter()
        .find(|m| {
            if let Some(ref filter) = filter {
                m.name.to_ascii_lowercase().contains(filter)
            } else {
                true
            }
        })
        .ok_or_else(|| "no material matched filter for atlas dump".to_string())?;
    let texture = material
        .texture_bindings
        .base_color_texture
        .as_ref()
        .ok_or_else(|| format!("material '{}' has no base color texture", material.name))?;
    let pixels = texture
        .pixel_data
        .as_ref()
        .ok_or_else(|| format!("material '{}' texture has no embedded pixel data", material.name))?;
    save_png(output_path, [texture.dimensions.0, texture.dimensions.1], pixels.as_slice())
}

fn save_png(path: &Path, extent: [u32; 2], pixels: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create '{}': {}", parent.display(), e))?;
    }

    let image: ImageBuffer<image::Rgba<u8>, _> =
        ImageBuffer::from_raw(extent[0], extent[1], pixels.to_vec())
            .ok_or_else(|| "failed to construct output image".to_string())?;
    image
        .save(path)
        .map_err(|e| format!("failed to save '{}': {}", path.display(), e))?;
    Ok(())
}

fn build_frame_input(
    avatar: &AvatarInstance,
    extent: [u32; 2],
    material_mode: MaterialShaderMode,
    debug_view: MaterialDebugView,
    yaw_deg: f32,
    material_filter: Option<&str>,
    disable_skinning: bool,
) -> RenderFrameInput {
    let mesh_instances: Vec<RenderMeshInstance> = avatar
        .asset
        .meshes
        .iter()
        .flat_map(|mesh| {
            mesh.primitives.iter().filter_map(|prim| {
                let material_asset = avatar
                    .asset
                    .materials
                    .iter()
                    .find(|m| m.id == prim.material_id);

                if let Some(filter) = material_filter {
                    let material_name = material_asset.map(|m| m.name.as_str()).unwrap_or("");
                    if !material_name
                        .to_ascii_lowercase()
                        .contains(&filter.to_ascii_lowercase())
                    {
                        return None;
                    }
                }

                let mut material_binding = material_asset
                    .map(MaterialUploadRequest::from_asset_material)
                    .unwrap_or_else(MaterialUploadRequest::default_material);
                material_binding.mode = material_mode.clone();
                material_binding.debug_view = debug_view.clone();

                let outline = vulvatar::renderer::frame_input::OutlineSnapshot {
                    enabled: material_binding.outline_width > 0.0,
                    width: material_binding.outline_width,
                    color: material_binding.outline_color,
                };

                let alpha_mode = match material_binding.alpha_mode {
                    vulvatar::asset::AlphaMode::Opaque => RenderAlphaMode::Opaque,
                    vulvatar::asset::AlphaMode::Mask(_) => RenderAlphaMode::Cutout,
                    vulvatar::asset::AlphaMode::Blend => RenderAlphaMode::Blend,
                };

                let cull_mode = if material_binding.double_sided {
                    RenderCullMode::DoubleSided
                } else {
                    RenderCullMode::BackFace
                };

                Some(RenderMeshInstance {
                    mesh_id: mesh.id,
                    primitive_id: prim.id,
                    material_binding,
                    bounds: prim.bounds.clone(),
                    alpha_mode,
                    cull_mode,
                    outline,
                    primitive_data: Some(Arc::new(prim.clone())),
                })
            })
        })
        .collect();

    let cam_distance: f32 = std::env::var("CAM_DIST")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5.0);
    let cam_pan_y: f32 = std::env::var("CAM_PAN_Y")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let camera = ViewportCamera {
        yaw_deg,
        distance: cam_distance,
        pan: [0.0, cam_pan_y],
        ..ViewportCamera::default()
    };
    let (view, eye_pos) = build_view_matrix(&camera);
    let projection = build_projection_matrix(
        camera.fov_deg,
        extent[0] as f32 / extent[1].max(1) as f32,
        0.1,
        1000.0,
    );

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
            skinning_matrices: if disable_skinning {
                vec![vulvatar::asset::identity_matrix(); avatar.pose.skinning_matrices.len().max(1)]
            } else {
                avatar.pose.skinning_matrices.clone()
            },
            cloth_deform: None,
            debug_flags: RenderDebugFlags {
                material_mode_override: Some(material_mode),
                ..Default::default()
            },
        }],
        output_request: OutputTargetRequest {
            preview_enabled: true,
            output_enabled: true,
            extent,
            color_space: vulvatar::renderer::frame_input::RenderColorSpace::Srgb,
            alpha_mode: RenderOutputAlpha::Premultiplied,
            export_mode: vulvatar::renderer::frame_input::RenderExportMode::CpuReadback,
        },
        background_image_path: None,
        show_ground_grid: false,
    }
}

fn build_view_matrix(cam: &ViewportCamera) -> (vulvatar::asset::Mat4, [f32; 3]) {
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
) -> vulvatar::asset::Mat4 {
    let fov_rad = fov_deg.to_radians();
    let f = 1.0 / (fov_rad * 0.5).tan();
    let a = far / (near - far);
    let b = far * near / (near - far);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0],
        [0.0, 0.0, a, b],
        [0.0, 0.0, -1.0, 0.0],
    ]
}

struct ImageStats {
    avg: [f32; 3],
    stddev: [f32; 3],
}

fn compute_stats(pixels: &[u8]) -> ImageStats {
    let px_count = (pixels.len() / 4).max(1) as f32;
    let mut sum = [0.0f64; 3];
    let mut sum_sq = [0.0f64; 3];

    for px in pixels.chunks_exact(4) {
        for c in 0..3 {
            let v = px[c] as f64;
            sum[c] += v;
            sum_sq[c] += v * v;
        }
    }

    let mut avg = [0.0f32; 3];
    let mut stddev = [0.0f32; 3];
    for c in 0..3 {
        let mean = sum[c] / px_count as f64;
        let variance = (sum_sq[c] / px_count as f64) - mean * mean;
        avg[c] = mean as f32;
        stddev[c] = variance.max(0.0).sqrt() as f32;
    }

    ImageStats { avg, stddev }
}

fn print_uv_stats(asset: &vulvatar::asset::AvatarAsset, material_filter: Option<&str>) {
    let filter = material_filter.map(|s| s.to_ascii_lowercase());
    for mesh in &asset.meshes {
        for prim in &mesh.primitives {
            let Some(vd) = prim.vertices.as_ref() else {
                continue;
            };
            let Some(material) = asset.materials.iter().find(|m| m.id == prim.material_id) else {
                continue;
            };
            let material_name = material.name.as_str();
            if let Some(ref filter) = filter {
                if !material_name.to_ascii_lowercase().contains(filter) {
                    continue;
                }
            }
            if vd.uvs.is_empty() {
                eprintln!(
                    "uv_stats mesh={:?} prim={:?} material='{}' uvs=0",
                    mesh.id, prim.id, material_name
                );
                continue;
            }
            let used_indices: Vec<usize> = if let Some(indices) = prim.indices.as_ref() {
                let mut set = HashSet::new();
                for &idx in indices {
                    set.insert(idx as usize);
                }
                set.into_iter().collect()
            } else {
                (0..vd.uvs.len()).collect()
            };
            let mut min_u = f32::INFINITY;
            let mut min_v = f32::INFINITY;
            let mut max_u = f32::NEG_INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            let mut unique = HashSet::new();
            for &idx in &used_indices {
                let Some(uv) = vd.uvs.get(idx) else {
                    continue;
                };
                min_u = min_u.min(uv[0]);
                min_v = min_v.min(uv[1]);
                max_u = max_u.max(uv[0]);
                max_v = max_v.max(uv[1]);
                unique.insert(((uv[0] * 10000.0) as i32, (uv[1] * 10000.0) as i32));
            }
            eprintln!(
                "uv_stats mesh={:?} prim={:?} material='{}' used_vertices={} unique_q={} u=({:.4}..{:.4}) v=({:.4}..{:.4}) sample_avg_rgb={:?}",
                mesh.id,
                prim.id,
                material_name,
                used_indices.len(),
                unique.len(),
                min_u,
                max_u,
                min_v,
                max_v,
                sample_average_rgb(material, &vd.uvs, &used_indices)
            );
        }
    }
}

fn sample_average_rgb(
    material: &vulvatar::asset::MaterialAsset,
    uvs: &[[f32; 2]],
    used_indices: &[usize],
) -> Option<[u8; 3]> {
    let texture = material.texture_bindings.base_color_texture.as_ref()?;
    let pixels = texture.pixel_data.as_ref()?;
    let (width, height) = texture.dimensions;
    if width == 0 || height == 0 {
        return None;
    }
    let mut sum = [0u64; 3];
    let mut count = 0u64;
    for &idx in used_indices {
        let Some(uv) = uvs.get(idx) else {
            continue;
        };
        let u = uv[0].clamp(0.0, 0.999999);
        let v = uv[1].clamp(0.0, 0.999999);
        let x = (u * width as f32) as usize;
        let y = (v * height as f32) as usize;
        let offset = (y * width as usize + x) * 4;
        if offset + 3 >= pixels.len() {
            continue;
        }
        sum[0] += pixels[offset] as u64;
        sum[1] += pixels[offset + 1] as u64;
        sum[2] += pixels[offset + 2] as u64;
        count += 1;
    }
    if count == 0 {
        None
    } else {
        Some([
            (sum[0] / count) as u8,
            (sum[1] / count) as u8,
            (sum[2] / count) as u8,
        ])
    }
}
