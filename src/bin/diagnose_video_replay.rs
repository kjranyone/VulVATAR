//! Video-sequence replay: feeds an extracted frame directory through
//! the production provider in order with CONTINUOUS temporal state —
//! the live pipeline, minus the camera. Used to validate temporal
//! behaviours that static benchmarks cannot represent (out-of-frame
//! slide-out / re-entry, holds, self-track stability).
//!
//! Usage:
//!   cargo run --bin diagnose_video_replay -- [frames_dir] [out.jsonl] [render_every]
//!
//! Emits one JSON line per frame to out.jsonl (default
//! diagnostics/video_replay.jsonl) with arm-chain observability.
//! With `render_every` = N > 0, additionally writes a camera|avatar
//! side-by-side composite PNG every N frames to `<out>_renders/` —
//! the ground truth for diagnosing what the user actually SEES,
//! which the scalar metrics cannot represent.

use std::io::Write;
use std::path::PathBuf;

use std::sync::Arc;

use vulvatar_lib::app::ViewportCamera;
use vulvatar_lib::asset::vrm::VrmAssetLoader;
use vulvatar_lib::asset::HumanoidBone;
use vulvatar_lib::avatar::pose_solver::{solve_avatar_pose, PoseSolverState, SolverParams};
use vulvatar_lib::avatar::{AvatarInstance, AvatarInstanceId};
use vulvatar_lib::renderer::frame_input::{
    CameraState, LightingState, OutputTargetRequest, RenderAlphaMode, RenderAvatarInstance,
    RenderColorSpace, RenderCullMode, RenderDebugFlags, RenderExportMode, RenderFrameInput,
    RenderMeshInstance, RenderOutputAlpha,
};
use vulvatar_lib::renderer::material::{MaterialShaderMode, MaterialUploadRequest};
use vulvatar_lib::renderer::VulkanRenderer;
use vulvatar_lib::tracking::provider::create_pose_provider;

fn render_extent() -> [u32; 2] {
    std::env::var("VULVATAR_REPLAY_RES")
        .ok()
        .and_then(|s| s.parse().ok())
        .map(|n: u32| [n, n])
        .unwrap_or([512, 512])
}

fn main() -> Result<(), String> {
    env_logger::init();
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "diagnostics/video_frames".to_string());
    let out_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "diagnostics/video_replay.jsonl".to_string());
    let render_every: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut renderer = if render_every > 0 {
        eprintln!("initializing Vulkan renderer…");
        let mut r = VulkanRenderer::new();
        r.initialize();
        Some(r)
    } else {
        None
    };
    let render_dir = PathBuf::from(format!("{}_renders", out_path.trim_end_matches(".jsonl")));
    if render_every > 0 {
        std::fs::create_dir_all(&render_dir).map_err(|e| format!("mkdir renders: {e}"))?;
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("read_dir {dir}: {e}"))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|x| x == "jpg" || x == "png")
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    eprintln!("{} frames", files.len());

    let mut provider = create_pose_provider("models", Default::default())?;
    let _ = provider.take_load_warnings();

    // Continuous solver, mirroring the live GUI configuration
    // (rotation_blend 1.0 per the user's project, lower body off).
    let vrm = std::env::var("VULVATAR_VRM")
        .unwrap_or_else(|_| "sample_data/AvatarSample_A.vrm".to_string());
    let asset = VrmAssetLoader::new()
        .load(&vrm)
        .map_err(|e| format!("load VRM: {e:?}"))?;
    let humanoid_map = asset.humanoid.as_ref().ok_or("no humanoid")?;
    let mut avatar = AvatarInstance::new(AvatarInstanceId(1), Arc::clone(&asset));
    let mut solver_state = PoseSolverState::default();
    let params = SolverParams {
        rotation_blend: 1.0,
        lower_body_tracking_enabled: false,
        ..Default::default()
    };

    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&out_path).map_err(|e| format!("create {out_path}: {e}"))?,
    );

    let sides = [
        ("L", HumanoidBone::LeftUpperArm, HumanoidBone::LeftLowerArm, HumanoidBone::LeftHand),
        ("R", HumanoidBone::RightUpperArm, HumanoidBone::RightLowerArm, HumanoidBone::RightHand),
    ];

    for (i, f) in files.iter().enumerate() {
        let img = image::open(f).map_err(|e| format!("open {}: {e}", f.display()))?.to_rgb8();
        let (w, h) = (img.width(), img.height());
        let est = provider.estimate_pose(img.as_raw(), w, h, i as u64);
        let sk = est.skeleton;

        avatar.build_base_pose();
        solve_avatar_pose(
            &sk,
            &asset.skeleton,
            asset.humanoid.as_ref(),
            &mut avatar.pose.local_transforms,
            &params,
            &mut solver_state,
        );
        avatar.compute_global_pose();

        if let Some(r) = renderer.as_mut() {
            if i % render_every == 0 {
                avatar.build_skinning_matrices();
                let ext = render_extent();
                if let Ok(rgba) = render_avatar(r, &avatar, ext) {
                    let _ = save_composite(&render_dir.join(format!("f{i:04}.png")), &img, &rgba, ext);
                }
            }
        }

        let av_pos = |b: HumanoidBone| -> Option<[f32; 3]> {
            let idx = humanoid_map.bone_map.get(&b).copied().map(|n| n.0 as usize)?;
            let m = avatar.pose.global_transforms.get(idx)?;
            Some([m[3][0], m[3][1], m[3][2]])
        };

        // Avatar reference heights (world y) so hand elevation can be
        // judged WITHOUT reading the render — disambiguates a raised
        // hand from long hanging hair that visually mimics a lowered
        // forearm on some rigs.
        let avy = |b: HumanoidBone| av_pos(b).map(|p| p[1]).unwrap_or(f32::NAN);
        let mut parts: Vec<String> = vec![format!(
            "\"frame\":{i},\"av_head_y\":{:.3},\"av_hip_y\":{:.3},\"av_sh_y\":{:.3}",
            avy(HumanoidBone::Head),
            avy(HumanoidBone::Hips),
            avy(HumanoidBone::LeftUpperArm)
        )];
        for (tag, sh, el, wr) in sides {
            let g = |b: HumanoidBone| sk.joints.get(&b);
            match (g(sh), g(el), g(wr)) {
                (Some(s), Some(e), Some(wj)) => {
                    let d3 = |a: [f32; 3], b: [f32; 3]| {
                        let v = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
                    };
                    parts.push(format!(
                        "\"{tag}\":{{\"wx\":{:.3},\"wy\":{:.3},\"wz\":{:.3},\"conf\":{:.2},\"fore\":{:.3},\"upper\":{:.3},\"sy\":{:.3},\"ey\":{:.3},\"econf\":{:.2}}}",
                        wj.position[0], wj.position[1], wj.position[2],
                        wj.confidence,
                        d3(e.position, wj.position),
                        d3(s.position, e.position),
                        s.position[1], e.position[1], e.confidence,
                    ));
                }
                _ => parts.push(format!("\"{tag}\":null")),
            }
            let hand_bone = if tag == "L" { HumanoidBone::LeftHand } else { HumanoidBone::RightHand };
            if let Some(p) = av_pos(hand_bone) {
                // Avatar hand height RELATIVE to its own shoulder —
                // the root-translation-invariant "is the arm raised?"
                // measure (absolute world y is confounded by the
                // pelvis being translated up/down by root tracking).
                let sh_y = av_pos(sh).map(|s| s[1]).unwrap_or(f32::NAN);
                parts.push(format!(
                    "\"{tag}av\":{{\"x\":{:.3},\"y\":{:.3},\"z\":{:.3},\"rel_sh\":{:.3}}}",
                    p[0], p[1], p[2], p[1] - sh_y
                ));
            }
        }
        parts.push(format!("\"overall\":{:.2}", sk.overall_confidence));
        writeln!(out, "{{{}}}", parts.join(",")).map_err(|e| e.to_string())?;
        if i % 100 == 0 {
            eprintln!("frame {i}");
        }
    }
    eprintln!("wrote {out_path}");
    Ok(())
}

fn save_composite(
    path: &std::path::Path,
    cam: &image::RgbImage,
    avatar_rgba: &[u8],
    ext: [u32; 2],
) -> Result<(), String> {
    let h = ext[1];
    let cam_w = (cam.width() * h / cam.height().max(1)).max(1);
    let cam_small =
        image::imageops::resize(cam, cam_w, h, image::imageops::FilterType::Triangle);
    let mut out = image::RgbImage::new(cam_w + ext[0], h);
    image::imageops::replace(&mut out, &cam_small, 0, 0);
    for y in 0..h {
        for x in 0..ext[0] {
            let idx = ((y * ext[0] + x) * 4) as usize;
            out.put_pixel(
                cam_w + x,
                y,
                image::Rgb([avatar_rgba[idx], avatar_rgba[idx + 1], avatar_rgba[idx + 2]]),
            );
        }
    }
    out.save(path).map_err(|e| format!("save composite: {e}"))
}

fn render_avatar(
    renderer: &mut VulkanRenderer,
    avatar: &AvatarInstance,
    extent: [u32; 2],
) -> Result<Vec<u8>, String> {
    let frame_input = build_frame_input(avatar, extent);
    let result = renderer
        .render(&frame_input)
        .map_err(|e| format!("render: {e}"))?;
    let pixels = result
        .exported_frame
        .as_ref()
        .and_then(|f| f.cpu_pixel_data())
        .ok_or_else(|| "no pixel data".to_string())?;
    Ok((*pixels).clone())
}

fn build_frame_input(avatar: &AvatarInstance, extent: [u32; 2]) -> RenderFrameInput {
    let mesh_instances: Vec<RenderMeshInstance> = avatar
        .asset
        .meshes
        .iter()
        .flat_map(|mesh| {
            mesh.primitives.iter().map(|prim| {
                let material_asset =
                    avatar.asset.materials.iter().find(|m| m.id == prim.material_id);
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
                    primitive_data: Some(Arc::clone(prim)),
                    morph_weights: Vec::new(),
                }
            })
        })
        .collect();

    // Upper-body webcam-style framing (matches the live use case).
    let camera = ViewportCamera {
        distance: 1.6,
        pan: [0.0, 0.95],
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
            skinning_matrices: avatar.pose.skinning_matrices.clone(),
            cloth_deforms: Vec::new(),
            debug_flags: RenderDebugFlags::default(),
        }],
        output_request: OutputTargetRequest {
            preview_enabled: true,
            output_enabled: true,
            extent,
            color_space: RenderColorSpace::Srgb,
            alpha_mode: RenderOutputAlpha::Opaque,
            export_mode: RenderExportMode::CpuReadback,
            msaa: vulvatar_lib::renderer::frame_input::MsaaMode::Off,
        },
        background_image_path: None,
        show_ground_grid: false,
        background_color: [0.04, 0.06, 0.10],
        transparent_background: false,
        avatar_opacity: 1.0,
        bloom: Default::default(),
        generative_background: Default::default(),
        background_tracking: Default::default(),
        time_seconds: 0.0,
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
            [-f[0], -f[1], -f[2], f[0] * eye_x + f[1] * eye_y + f[2] * eye_z],
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
