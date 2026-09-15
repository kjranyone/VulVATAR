//! Z-fight flicker probe for the 2026-09-15 "nipple-area mesh flails at
//! idle" report. Live measurement established: all spring joints are
//! static, but ~2300 window pixels flicker violently, alternating
//! between a dark and a skin-tone surface — two nearly-coincident
//! surfaces swapping depth order every frame.
//!
//! This harness renders the SAME pipeline offline (toon + material
//! outlines via `RenderMeshInstance::from_primitive`) from a
//! chest-height front view, N identical frames, and reports the
//! temporal-std map: does the flicker exist with fully static inputs?
//!
//! Variants (positional args after the avatar path):
//!   `microns`  — per-frame white jitter (µm) injected on the breast
//!                chain locals (tests the "sub-mm jitter flips depth
//!                ties" mechanism; the live app's substep count varies
//!                1..6, so its spring output jitters at µm scale)
//!   `outline`  — 0 disables material outlines (A/B the outline's role)
//!
//! Usage: diagnose_zfight_flicker <avatar> <out_dir> [frames] [microns] [outline01]

use std::path::Path;

use vulvatar_lib::renderer::frame_input::{
    BodySdfPlan, CameraState, LightingState, OutputTargetRequest, RenderAvatarInstance,
    RenderFrameInput,
};
use vulvatar_lib::renderer::material::MaterialShaderMode;
use vulvatar_lib::renderer::VulkanRenderer;
use vulvatar_lib::simulation::sdf::SdfGrid;
use vulvatar_lib::simulation::spring::SpringTuning;

fn main() -> Result<(), String> {
    env_logger::init();
    let mut args = std::env::args().skip(1);
    let input_path = args
        .next()
        .unwrap_or_else(|| "sample_data/YUMEKA_v1.0.1/FBX/Yumeka_v1.0.fbx".to_string());
    let output_dir = args.next().unwrap_or_else(|| "diagnostics/zfight".to_string());
    if output_dir.contains("validation_images") {
        return Err("diagnostics outputs must not target validation_images/".into());
    }
    let total_frames: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(40);
    let microns: f32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let outline_on: bool = args.next().and_then(|v| v.parse::<u32>().ok()).map(|v| v != 0).unwrap_or(true);
    let springs_on: bool = args.next().and_then(|v| v.parse::<u32>().ok()).map(|v| v != 0).unwrap_or(true);
    std::fs::create_dir_all(&output_dir).map_err(|e| format!("create dir: {e}"))?;

    let loader = vulvatar_lib::asset::fbx::FbxAssetLoader::new();
    let asset = loader.load(&input_path).map_err(|e| format!("load: {e}"))?;
    let mut avatar =
        vulvatar_lib::avatar::AvatarInstance::new(vulvatar_lib::avatar::AvatarInstanceId(1), asset);

    // Rest pose (bind T-pose reference); the live idle display pose is
    // the A-pose overlay, but the z-fight condition (garment vs skin
    // gap at the breast apex) exists in both.
    for (i, node) in avatar.asset.skeleton.nodes.iter().enumerate() {
        avatar.pose.local_transforms[i] = node.rest_local.clone();
    }
    avatar.compute_global_pose();

    // Breast chain node ids for the jitter injection.
    let breast_nodes: Vec<usize> = avatar
        .asset
        .skeleton
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.name.to_lowercase().contains("breast"))
        .map(|(i, _)| i)
        .collect();
    println!(
        "breast nodes: {:?}",
        breast_nodes
            .iter()
            .map(|&i| avatar.asset.skeleton.nodes[i].name.clone())
            .collect::<Vec<_>>()
    );

    // Splat: body + face (garments OFF — isolate the garment-vs-skin
    // base geometry; springs still run against the body field).
    let mut splat = Vec::new();
    let mut union = vulvatar_lib::asset::Aabb::empty();
    if let Some((mesh_id, primitive_id)) =
        vulvatar_lib::asset::clearance::find_body_primitive(&avatar.asset)
    {
        for m in &avatar.asset.meshes {
            if m.id == mesh_id {
                for p in &m.primitives {
                    if p.id == primitive_id {
                        splat.push((mesh_id, primitive_id));
                        union.expand(&p.bounds);
                    }
                }
            }
        }
    }
    let grid = SdfGrid::for_aabb(&union);

    let mut renderer = VulkanRenderer::new();
    renderer.initialize();

    let width = 900u32;
    let height = 1200u32;
    let dt = 1.0 / 60.0;
    let tuning = SpringTuning::default();

    let mut rng: u32 = 0xC0FF_EE01;
    let mut white = || {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        ((rng >> 8) as f32 / 16777216.0) - 0.5
    };

    // Render the same input N times; collect CPU readbacks.
    let mut frames_pix: Vec<Vec<u8>> = Vec::new();
    for frame in 0..total_frames {
        if microns > 0.0 {
            let amt = microns * 1e-6;
            for &ni in &breast_nodes {
                let mut t = avatar.asset.skeleton.nodes[ni].rest_local.clone();
                t.translation[0] += amt * white();
                t.translation[1] += amt * white();
                t.translation[2] += amt * white();
                avatar.pose.local_transforms[ni] = t;
            }
        }
        avatar.compute_global_pose();
        avatar.build_skinning_matrices();
        let sdf = avatar.body_sdf.clone();
        if springs_on {
            vulvatar_lib::simulation::spring::step_spring_bones(
                dt,
                &mut avatar,
                &[],
                &tuning,
                [0.0, -1.0, 0.0],
                1.0,
                sdf.as_ref(),
            );
            avatar.compute_global_pose();
            avatar.build_skinning_matrices();
        }

        let fi = build_frame_input(&avatar, splat.clone(), grid, width, height, outline_on);
        let _ = renderer.render(&fi)?;
        let result = renderer.render(&fi)?;
        for entry in &result.sdf_fields {
            if entry.instance_id == avatar.id.0 {
                avatar.body_sdf = Some(vulvatar_lib::simulation::sdf::SdfField::new(
                    entry.grid,
                    entry.data.clone(),
                ));
            }
        }
        if let Some(exported) = result.exported_frame.as_ref() {
            if let Some(px) = exported.cpu_pixel_data() {
                frames_pix.push(px.as_ref().clone());
            }
        }
        if frame % 10 == 0 || frame == total_frames - 1 {
            let mut asleep = 0usize;
            let mut awake_sample: Vec<(String, f32)> = Vec::new();
            for (ci, st) in avatar.secondary_motion.spring_states.iter().enumerate() {
                if st.sleeping {
                    asleep += 1;
                } else if awake_sample.len() < 24 {
                    let m = st
                        .positions
                        .iter()
                        .zip(st.previous_positions.iter())
                        .map(|(p, q)| {
                            let d = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
                            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
                        })
                        .fold(0.0f32, f32::max);
                    let name = avatar.asset.skeleton.nodes[st.chain_root.0 as usize].name.clone();
                    awake_sample.push((name, m));
                }
            }
            println!(
                "frame {frame}: asleep {asleep}/{} | awake max |cur-prev| (mm): {:?}",
                avatar.secondary_motion.spring_states.len(),
                awake_sample
                    .iter()
                    .map(|(n, m)| (n.clone(), m * 1000.0))
                    .collect::<Vec<_>>()
            );
        }
        if frame == 0 || frame == total_frames - 1 {
            let img: image::ImageBuffer<image::Rgba<u8>, _> =
                image::ImageBuffer::from_raw(width, height, frames_pix[frame].clone())
                    .ok_or("buf")?;
            img.save(Path::new(&output_dir).join(format!(
                "frame_{:03}_m{}_o{}.png",
                frame,
                microns,
                outline_on as u8
            )))
            .map_err(|e| format!("save: {e}"))?;
        }
    }

    // Temporal std map over the collected frames.
    let n = frames_pix.len();
    if n < 3 {
        return Err("not enough frames".into());
    }
    let mut tstd = vec![0f32; (width * height) as usize];
    let mut acc = vec![[0f64; 3]; (width * height) as usize];
    let mut acc2 = vec![[0f64; 3]; (width * height) as usize];
    for px in &frames_pix {
        for i in 0..(width * height) as usize {
            for c in 0..3 {
                let v = px[i * 4 + c] as f64;
                acc[i][c] += v;
                acc2[i][c] += v * v;
            }
        }
    }
    let mut flicker = 0usize;
    let mut flicker_max = 0f32;
    for i in 0..(width * height) as usize {
        let mut s = 0f32;
        for c in 0..3 {
            let m = acc[i][c] / n as f64;
            let var = (acc2[i][c] / n as f64 - m * m).max(0.0);
            s += var.sqrt() as f32;
        }
        tstd[i] = s / 3.0;
        if tstd[i] > 20.0 {
            flicker += 1;
            flicker_max = flicker_max.max(tstd[i]);
        }
    }
    println!(
        "frames={n} microns={microns} outline={outline_on}: flicker px (std>20) = {flicker}, max std = {flicker_max:.1}"
    );
    // Where the remaining flicker lives: row-band distribution.
    let mut bands = [0usize; 6];
    for y in 0..height as usize {
        let b = y * 6 / height as usize;
        for x in 0..width as usize {
            if tstd[y * width as usize + x] > 20.0 {
                bands[b] += 1;
            }
        }
    }
    println!("flicker by vertical band (top→bottom, 6 bands): {bands:?}");
    // Save heat + top rows
    let mut heat = vec![0u8; (width * height) as usize];
    let mn = tstd.iter().cloned().fold(0f32, f32::max).max(1e-6);
    for (i, h) in heat.iter_mut().enumerate() {
        *h = (tstd[i] / mn * 255.0) as u8;
    }
    let img: image::ImageBuffer<image::Rgba<u8>, _> =
        image::ImageBuffer::from_raw(width, height, heat.iter().map(|&v| [v, v, v, 255]).flatten().collect::<Vec<u8>>())
            .ok_or("heat buf")?;
    img.save(Path::new(&output_dir).join(format!("heat_m{}_o{}.png", microns, outline_on as u8)))
        .map_err(|e| format!("save heat: {e}"))?;
    // Raw f32 temporal-std dump for offline cluster analysis.
    let bytes: Vec<u8> = tstd
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(
        Path::new(&output_dir).join(format!("tstd_m{}_o{}.f32", microns, outline_on as u8)),
        bytes,
    )
    .map_err(|e| format!("save tstd: {e}"))?;
    Ok(())
}

fn build_frame_input(
    avatar: &vulvatar_lib::avatar::AvatarInstance,
    prims: Vec<(vulvatar_lib::asset::MeshId, vulvatar_lib::asset::PrimitiveId)>,
    grid: SdfGrid,
    width: u32,
    height: u32,
    outline_on: bool,
) -> RenderFrameInput {
    let mut mesh_instances = Vec::new();
    for mesh in &avatar.asset.meshes {
        for prim in &mesh.primitives {
            let mut inst =
                vulvatar_lib::renderer::frame_input::RenderMeshInstance::from_primitive(
                    avatar, mesh.id, &std::sync::Arc::clone(prim),
                );
            inst.material_binding.mode = MaterialShaderMode::ToonLike;
            if !outline_on {
                inst.outline.enabled = false;
            }
            mesh_instances.push(inst);
        }
    }

    // Chest-height front view so the breast apex / garment gap region
    // is front and center.
    let camera = vulvatar_lib::app::ViewportCamera {
        distance: 1.0,
        pan: [0.0, 0.98],
        yaw_deg: 0.0,
        pitch_deg: 2.0,
        fov_deg: 30.0,
    };
    let (view, eye_pos) = build_view_matrix(&camera);
    let projection = build_projection_matrix(camera.fov_deg, width as f32 / height as f32, 0.05, 20.0);

    RenderFrameInput {
        camera: CameraState {
            view,
            projection,
            position_ws: eye_pos,
            viewport_extent: [width, height],
        },
        lighting: LightingState::default(),
        instances: vec![RenderAvatarInstance {
            instance_id: avatar.id,
            world_transform: avatar.world_transform.clone(),
            mesh_instances,
            skinning_matrices: avatar.pose.skinning_matrices.clone(),
            cloth_deforms: Vec::new(),
            body_sdf: (!prims.is_empty()).then(|| BodySdfPlan { prims, grid }),
            debug_flags: Default::default(),
        }],
        output_request: OutputTargetRequest {
            preview_enabled: false,
            output_enabled: true,
            extent: [width, height],
            color_space: vulvatar_lib::renderer::frame_input::RenderColorSpace::Srgb,
            alpha_mode: vulvatar_lib::renderer::frame_input::RenderOutputAlpha::Opaque,
            export_mode: vulvatar_lib::renderer::frame_input::RenderExportMode::CpuReadback,
            msaa: vulvatar_lib::renderer::frame_input::MsaaMode::Off,
        },
        background_image_path: None,
        show_ground_grid: false,
        background_color: [0.35, 0.35, 0.38],
        transparent_background: false,
        avatar_opacity: 1.0,
        bloom: Default::default(),
        generative_background: Default::default(),
        background_tracking: Default::default(),
        time_seconds: 0.0,
    }
}

fn build_view_matrix(cam: &vulvatar_lib::app::ViewportCamera) -> (vulvatar_lib::asset::Mat4, [f32; 3]) {
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
    let len = (fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2]).sqrt().max(1e-6);
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
