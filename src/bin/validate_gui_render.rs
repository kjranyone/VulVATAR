//! GUI-equivalent render validation.
//!
//! Mirrors `Application::run_frame` (the actual live-tracking path) on
//! static images from `validation_images/`. Lets us regress GPU
//! skinning + secondary-motion behaviour without a webcam.
//!
//! Differences from `validate_pipeline` (which is geometry-only with
//! provider 1 + no sim):
//!
//! * **Provider 2** (`Rtmw3dWithDepthProvider`) — the rtmw3d-with-depth
//!   path the user is actually running. Same construction parameters.
//! * **Spring bones** stepped once per frame (matches `toggle_spring`
//!   in `last_session.vvtproj`).
//! * **Render path** mirrors `Application::build_frame_input_multi`'s
//!   per-instance assembly so the renderer compute-prepass (morph +
//!   cloth + skinning) sees the same `RenderAvatarInstance` shape.
//!
//! Usage:
//!   cargo run --bin validate_gui_render -- [--limit N] [--out DIR]
//!                                          [--break-threshold-m M]
//!
//! Per-image outputs (under `diagnostics/validate_gui_render/`):
//!   <stem>.png            rendered avatar
//!   <stem>.bones.txt      per-bone position dump (world space)
//!   <stem>.broken         empty marker file when breakage detected
//!
//! Breakage criterion: any bone's world translation more than
//! `break_threshold_m` (default 3.0 m) from the avatar's Hips position.
//! At that distance the bone has clearly run off the rig and any
//! mesh vertex weighted to it will trail behind the rest of the
//! avatar in the render.
//!
//! Exit code 1 if any image broke, so the script can drive an iterate-
//! until-clean loop.
//!
//! NOTE: keep this binary in lockstep with the GUI path it mirrors.
//! When `run_frame` gains a new step (e.g. a new sim pass), this
//! binary needs the matching addition or its render diverges from
//! what the user actually sees.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use image::{ImageBuffer, Rgba};
use vulvatar_lib::app::ViewportCamera;
use vulvatar_lib::asset::vrm::VrmAssetLoader;
use vulvatar_lib::asset::{AvatarAsset, HumanoidBone};
use vulvatar_lib::avatar::pose_solver::{solve_avatar_pose, PoseSolverState, SolverParams};
use vulvatar_lib::avatar::{AvatarInstance, AvatarInstanceId};
use vulvatar_lib::renderer::frame_input::{
    CameraState, LightingState, OutputTargetRequest, RenderAlphaMode, RenderAvatarInstance,
    RenderColorSpace, RenderCullMode, RenderDebugFlags, RenderExportMode, RenderFrameInput,
    RenderMeshInstance, RenderOutputAlpha,
};
use vulvatar_lib::renderer::material::{MaterialShaderMode, MaterialUploadRequest};
use vulvatar_lib::renderer::VulkanRenderer;
use vulvatar_lib::simulation::PhysicsWorld;
use vulvatar_lib::tracking::provider::PoseProvider;
use vulvatar_lib::tracking::rtmw3d_with_depth::Rtmw3dWithDepthProvider;

const VRM_PATH: &str = "sample_data/AvatarSample_A.vrm";
const VALIDATION_ROOT: &str = "validation_images";
const OUT_DIR_DEFAULT: &str = "diagnostics/validate_gui_render";
const RENDER_EXTENT: [u32; 2] = [512, 512];
const FRAME_DT: f32 = 1.0 / 60.0;

fn main() -> Result<(), String> {
    env_logger::init();

    let mut limit: Option<usize> = None;
    let mut out_dir = PathBuf::from(OUT_DIR_DEFAULT);
    let mut break_threshold_m = 3.0f32;
    let mut args = std::env::args().skip(1).peekable();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--limit" => {
                let n: usize = args
                    .next()
                    .ok_or("--limit needs N")?
                    .parse()
                    .map_err(|e| format!("--limit parse: {e}"))?;
                limit = Some(n);
            }
            "--out" => {
                out_dir = PathBuf::from(args.next().ok_or("--out needs DIR")?);
            }
            "--break-threshold-m" => {
                break_threshold_m = args
                    .next()
                    .ok_or("--break-threshold-m needs M")?
                    .parse()
                    .map_err(|e| format!("--break-threshold-m parse: {e}"))?;
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }

    std::fs::create_dir_all(&out_dir).map_err(|e| format!("mkdir out: {e}"))?;

    eprintln!("loading provider 2 (rtmw3d-with-depth)…");
    let mut provider = Rtmw3dWithDepthProvider::from_models_dir("models")
        .map_err(|e| format!("provider 2 load: {e}"))?;
    let _ = provider.take_load_warnings();

    eprintln!("loading VRM {VRM_PATH}…");
    let asset = VrmAssetLoader::new()
        .load(VRM_PATH)
        .map_err(|e| format!("VRM load: {e}"))?;

    eprintln!("initialising Vulkan renderer…");
    let mut renderer = VulkanRenderer::new();
    renderer.initialize();

    let mut images = collect_images(Path::new(VALIDATION_ROOT))?;
    images.sort();
    if let Some(n) = limit {
        images.truncate(n);
    }
    eprintln!("rendering {} image(s)", images.len());

    let mut broke_count = 0usize;
    let mut physics = PhysicsWorld::new();
    for (idx, image_path) in images.iter().enumerate() {
        let stem = image_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| format!("img{idx:03}"));
        match process_one(
            image_path,
            &mut provider,
            &asset,
            &mut renderer,
            &mut physics,
            &out_dir,
            &stem,
            idx as u64,
            break_threshold_m,
        ) {
            Ok(verdict) => {
                eprintln!(
                    "[{:>3}/{}] {} {} bone={:.2}m vert={:.2}m",
                    idx + 1,
                    images.len(),
                    if verdict.broken { "BROKEN " } else { "ok     " },
                    image_path.display(),
                    verdict.max_bone_dist_m,
                    verdict.max_vertex_dist_m,
                );
                if verdict.broken {
                    broke_count += 1;
                }
            }
            Err(e) => {
                eprintln!(
                    "[{:>3}/{}] ERROR   {}: {}",
                    idx + 1,
                    images.len(),
                    image_path.display(),
                    e
                );
                broke_count += 1;
            }
        }
    }

    eprintln!();
    eprintln!(
        "summary: {} broken / {} total (threshold = {:.1} m)",
        broke_count,
        images.len(),
        break_threshold_m
    );
    if broke_count > 0 {
        std::process::exit(1);
    }
    Ok(())
}

struct Verdict {
    broken: bool,
    max_bone_dist_m: f32,
    max_vertex_dist_m: f32,
}

#[allow(clippy::too_many_arguments)]
fn process_one(
    image_path: &Path,
    provider: &mut Rtmw3dWithDepthProvider,
    asset: &Arc<AvatarAsset>,
    renderer: &mut VulkanRenderer,
    physics: &mut PhysicsWorld,
    out_dir: &Path,
    stem: &str,
    frame_index: u64,
    break_threshold_m: f32,
) -> Result<Verdict, String> {
    // 1. Inference (provider 2 = rtmw3d-with-depth).
    let img = image::open(image_path).map_err(|e| format!("open: {e}"))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let est = provider.estimate_pose(rgb.as_raw(), w, h, frame_index);

    // 2. Build avatar + solve pose (restored — we verified rest-pose
    // renders clean, so the breakage must come in via the solver).
    let mut state = PoseSolverState::default();
    let mut avatar = AvatarInstance::new(AvatarInstanceId(1), Arc::clone(asset));
    avatar.build_base_pose();
    let humanoid = asset.humanoid.as_ref();
    let params = SolverParams {
        rotation_blend: 1.0,
        joint_confidence_threshold: 0.1,
        face_confidence_threshold: 0.1,
        ..Default::default()
    };
    solve_avatar_pose(
        &est.skeleton,
        &asset.skeleton,
        humanoid,
        &mut avatar.pose.local_transforms,
        &params,
        &mut state,
    );
    avatar.compute_global_pose();
    // Match the live session (`toggle_spring: true`) — one substep at
    // 60 Hz mirrors what the GUI runs each frame.
    physics.step_springs(FRAME_DT, 1, &mut avatar);
    avatar.compute_global_pose();
    avatar.build_skinning_matrices();

    // Audit the upper 3x3 column lengths of every skinning matrix.
    // If any are off unity by more than 1%, the shader's
    // `mat3_to_quat(mat3(mi))` will fold that scale into the rotation
    // quaternion (the shader doesn't normalise columns before the
    // extraction — Mike Day's algorithm assumes the input is a pure
    // rotation). Applying the result to a vertex multiplies positions
    // by the embedded scale, producing the off-rig stretching seen
    // in the broken renders.
    audit_skinning_column_scales(asset, &avatar);

    // 5. Render (matches the live path's `build_frame_input_multi`
    // for the single-instance case).
    let rgba = render_avatar(renderer, &avatar, RENDER_EXTENT)?;
    let png_path = out_dir.join(format!("{stem}.png"));
    save_rgba_png(&png_path, RENDER_EXTENT, &rgba)?;

    // 6. Breakage detection on two axes:
    //    a. Bone world translation vs Hips — catches "bone ran off
    //       the rig" cases.
    //    b. CPU DQS-mirrored vertex world position vs Hips — catches
    //       "bones fine but skinning produces extreme vertices",
    //       which is what the user's `image.png` actually shows.
    //       Without (b) the bone-level check missed the screenshot
    //       case entirely (max bone dist was 0.84 m even though the
    //       sleeve flew across the screen).
    let bones_path = out_dir.join(format!("{stem}.bones.txt"));
    let (max_bone_dist, worst_name) =
        detect_bone_breakage(asset, &avatar, &bones_path, break_threshold_m)?;
    let (max_vertex_dist, top_verts) = detect_vertex_breakage_detailed(asset, &avatar);
    if !top_verts.is_empty() && top_verts[0].dist_from_hips_m > 0.5 {
        eprintln!("    CPU DQS top vertices (post-DQS world distance from Hips):");
        for v in top_verts.iter().take(5) {
            eprintln!(
                "      vid={} dist={:.3}m world=({:+.2},{:+.2},{:+.2}) acc_len={:.3}",
                v.vid, v.dist_from_hips_m, v.world_pos[0], v.world_pos[1], v.world_pos[2], v.acc_len_pre_normalize,
            );
        }
    }
    let max_dist = max_bone_dist.max(max_vertex_dist);
    let broken = max_dist > break_threshold_m;
    if broken {
        let _ = std::fs::write(
            out_dir.join(format!("{stem}.broken")),
            format!(
                "max_bone_dist_m={max_bone_dist:.3} worst_bone={worst_name}\n\
                 max_vertex_dist_m={max_vertex_dist:.3}\n"
            ),
        );
    } else {
        // Tidy up the .broken marker if the prior run flagged this stem.
        let _ = std::fs::remove_file(out_dir.join(format!("{stem}.broken")));
    }
    Ok(Verdict {
        broken,
        max_bone_dist_m: max_bone_dist,
        max_vertex_dist_m: max_vertex_dist,
    })
}

fn render_avatar(
    renderer: &mut VulkanRenderer,
    avatar: &AvatarInstance,
    extent: [u32; 2],
) -> Result<Vec<u8>, String> {
    let frame_input = build_frame_input(avatar, extent);
    // Warmup render — the first render after avatar/primitive changes
    // sets up GPU caches (transform_cache slot, descriptor sets) but
    // its readback may catch a partially-populated frame. The second
    // render hits the warm path and produces the steady-state output.
    let _ = renderer
        .render(&frame_input)
        .map_err(|e| format!("warm render: {e}"))?;
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
                    primitive_data: Some(Arc::clone(prim)),
                    morph_weights: Vec::new(),
                }
            })
        })
        .collect();

    let camera = ViewportCamera {
        distance: 3.5,
        pan: [0.0, 1.0],
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
        background_color: [0.10, 0.10, 0.10],
        transparent_background: false,
        avatar_opacity: 1.0,
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

/// Mirror of the `transform_cs` DQS skinning math, run for every
/// skinned vertex of the avatar. Returns the maximum distance any
/// world-space vertex sits from the Hips bone. The criterion is
/// "geometrically far from where the avatar lives" — if a sleeve
/// vertex flies off-screen, this metric jumps.
///
/// We use Hips as the reference because it's the rig's nominal
/// origin under root translation; comparing against per-vertex
/// rest position would also work but Hips is cheap and stable
/// across poses.
/// GLSL `mat3_to_quat` exact mirror — no column normalisation. Used
/// to verify whether the shader's non-normalising quaternion
/// extraction is responsible for the divergence between the CPU
/// `mat4_rotation_to_quat` (which DOES normalise) and the GPU output.
/// GLSL `mat3_to_quat` exact mirror, column-major (m[col][row]).
/// No column normalisation — Mike Day's formula expects a pure
/// rotation. The shader uses this as-is on `mat3(mi)` so any non-unit
/// scale or skew in the upper 3x3 propagates straight into the
/// "quaternion" and from there into the rotated vertex position.
#[allow(dead_code)]
fn mat3_to_quat_shader(m: &[[f32; 4]; 4]) -> [f32; 4] {
    let trace = m[0][0] + m[1][1] + m[2][2];
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        let inv = 1.0 / s;
        // q.x = (m[1][2] - m[2][1]) / s  (GLSL column-major access)
        [(m[1][2] - m[2][1]) * inv,
         (m[2][0] - m[0][2]) * inv,
         (m[0][1] - m[1][0]) * inv,
         0.25 * s]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        let inv = 1.0 / s;
        [0.25 * s,
         (m[1][0] + m[0][1]) * inv,
         (m[2][0] + m[0][2]) * inv,
         (m[1][2] - m[2][1]) * inv]
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        let inv = 1.0 / s;
        [(m[1][0] + m[0][1]) * inv,
         0.25 * s,
         (m[2][1] + m[1][2]) * inv,
         (m[2][0] - m[0][2]) * inv]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        let inv = 1.0 / s;
        [(m[2][0] + m[0][2]) * inv,
         (m[2][1] + m[1][2]) * inv,
         0.25 * s,
         (m[0][1] - m[1][0]) * inv]
    }
}

/// Per-vertex CPU mirror result + diagnostics.
#[derive(Clone, Copy, Debug)]
struct VertexResult {
    world_pos: [f32; 3],
    dist_from_hips_m: f32,
    acc_len_pre_normalize: f32,
    mesh_idx: usize,
    prim_idx: usize,
    vid: usize,
}

fn detect_vertex_breakage_detailed(
    asset: &Arc<AvatarAsset>,
    avatar: &AvatarInstance,
) -> (f32, Vec<VertexResult>) {
    const WEIGHT_EPS: f32 = 1.0e-4;
    const ACC_LEN_EPS: f32 = 1.0e-6;

    let Some(humanoid) = asset.humanoid.as_ref() else { return (0.0, Vec::new()) };
    let Some(hips_idx) = humanoid
        .bone_map
        .get(&HumanoidBone::Hips)
        .map(|n| n.0 as usize)
    else {
        return (0.0, Vec::new());
    };
    let Some(hips_m) = avatar.pose.global_transforms.get(hips_idx) else {
        return (0.0, Vec::new());
    };
    let hips_pos = [hips_m[3][0], hips_m[3][1], hips_m[3][2]];

    let skin_mats = &avatar.pose.skinning_matrices;
    let mut max_dist = 0.0f32;
    let mut top: Vec<VertexResult> = Vec::new();

    // Per-primitive iteration. Primitives may share VertexData via
    // Arc but mat3_to_quat is cheap and dedup'ing the work isn't
    // worth the complexity for a diagnostic.
    for (mesh_idx, mesh) in asset.meshes.iter().enumerate() {
        for (prim_idx, prim) in mesh.primitives.iter().enumerate() {
            let Some(vd) = prim.vertices.as_ref() else { continue };
            for (vid, (&jw, &ji)) in vd
                .joint_weights
                .iter()
                .zip(vd.joint_indices.iter())
                .enumerate()
            {
                let total_w = jw[0] + jw[1] + jw[2] + jw[3];
                if total_w < WEIGHT_EPS {
                    continue;
                }
                let pos = vd.positions[vid];

                // DQS rotation accumulation — must use the SAME
                // mat3_to_quat formulation as the shader for ref_real,
                // not the CPU normalising version. Mixing produces a
                // CPU/GPU divergence that masks the production bug.
                let mut ref_real: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
                for i in 0..4 {
                    if jw[i] > WEIGHT_EPS {
                        ref_real = mat3_to_quat_shader(&skin_mats[ji[i] as usize]);
                        break;
                    }
                }
                let mut acc_real = [0.0f32; 4];
                let mut acc_trans = [0.0f32; 3];
                for i in 0..4 {
                    let wi = jw[i];
                    if wi <= WEIGHT_EPS {
                        continue;
                    }
                    let mi = &skin_mats[ji[i] as usize];
                    let qi = mat3_to_quat_shader(mi);
                    let dot = qi[0] * ref_real[0]
                        + qi[1] * ref_real[1]
                        + qi[2] * ref_real[2]
                        + qi[3] * ref_real[3];
                    let s = if dot >= 0.0 { 1.0 } else { -1.0 };
                    for k in 0..4 {
                        acc_real[k] += wi * s * qi[k];
                    }
                    // mat4 column 3 = translation (column-major).
                    acc_trans[0] += wi * mi[3][0];
                    acc_trans[1] += wi * mi[3][1];
                    acc_trans[2] += wi * mi[3][2];
                }
                let acc_len = (acc_real[0] * acc_real[0]
                    + acc_real[1] * acc_real[1]
                    + acc_real[2] * acc_real[2]
                    + acc_real[3] * acc_real[3])
                    .sqrt();
                let world_pos = if acc_len < ACC_LEN_EPS {
                    // H-4 fallback: emit mesh-local pos as-is.
                    pos
                } else {
                    let inv = 1.0 / acc_len;
                    let q = [
                        acc_real[0] * inv,
                        acc_real[1] * inv,
                        acc_real[2] * inv,
                        acc_real[3] * inv,
                    ];
                    let r = vulvatar_lib::math_utils::quat_rotate_vec3(&q, &pos);
                    [r[0] + acc_trans[0], r[1] + acc_trans[1], r[2] + acc_trans[2]]
                };
                let d = ((world_pos[0] - hips_pos[0]).powi(2)
                    + (world_pos[1] - hips_pos[1]).powi(2)
                    + (world_pos[2] - hips_pos[2]).powi(2))
                .sqrt();
                if d > max_dist {
                    max_dist = d;
                }
                if d > 0.5 {
                    top.push(VertexResult {
                        world_pos,
                        dist_from_hips_m: d,
                        acc_len_pre_normalize: acc_len,
                        mesh_idx,
                        prim_idx,
                        vid,
                    });
                }
            }
        }
    }
    // Each body-mesh vertex appears 9 times via material-split prims
    // (they all share the same VertexData Arc), so dedup by vid +
    // mesh_idx before slicing the top entries.
    top.sort_by(|a, b| b.dist_from_hips_m.total_cmp(&a.dist_from_hips_m));
    let mut seen: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    top.retain(|v| seen.insert((v.mesh_idx, v.vid)));
    top.truncate(15);
    (max_dist, top)
}

/// Inspect every skinning matrix's upper 3x3 column lengths and
/// report the worst non-unit case. The shader's `mat3_to_quat` runs
/// on the upper 3x3 without first normalising columns; a non-unit
/// column folds the scale into the resulting quaternion and the
/// world position is multiplied by that scale, which produces the
/// "vertex flies off-rig" stretching observed in the broken renders.
fn audit_skinning_column_scales(asset: &Arc<AvatarAsset>, avatar: &AvatarInstance) {
    let mut worst_scale: f32 = 1.0;
    let mut worst_scale_idx = usize::MAX;
    let mut worst_scale_col = 0usize;
    // Track the bone whose translation column has the largest magnitude.
    // For a bone whose current pose ≈ bind pose, mi[3] ≈ 0. A big mi[3]
    // means "current pose differs a lot from bind" and feeds directly
    // into `acc_trans` in the DQS path. If this is unexpectedly large
    // for a bone the solver shouldn't have moved much, that's a red flag.
    let mut max_trans_mag: f32 = 0.0;
    let mut max_trans_idx = usize::MAX;
    let mut max_trans_xyz = [0.0f32; 3];
    for (idx, m) in avatar.pose.skinning_matrices.iter().enumerate() {
        for c in 0..3 {
            let len = (m[c][0] * m[c][0] + m[c][1] * m[c][1] + m[c][2] * m[c][2]).sqrt();
            let dev = (len - 1.0).abs();
            let worst_dev = (worst_scale - 1.0).abs();
            if dev > worst_dev {
                worst_scale = len;
                worst_scale_idx = idx;
                worst_scale_col = c;
            }
        }
        let tx = m[3][0];
        let ty = m[3][1];
        let tz = m[3][2];
        let tmag = (tx * tx + ty * ty + tz * tz).sqrt();
        if tmag > max_trans_mag {
            max_trans_mag = tmag;
            max_trans_idx = idx;
            max_trans_xyz = [tx, ty, tz];
        }
    }
    if worst_scale_idx != usize::MAX {
        let dev_pct = (worst_scale - 1.0).abs() * 100.0;
        if dev_pct > 1.0 {
            let name = asset
                .skeleton
                .nodes
                .get(worst_scale_idx)
                .map(|n| n.name.as_str())
                .unwrap_or("?");
            eprintln!(
                "    column-scale audit: worst |col|={:.4} (dev {:.1}%) at bone[{worst_scale_idx}]='{name}' col={worst_scale_col}",
                worst_scale, dev_pct
            );
        }
    }
    if max_trans_mag > 1.0 {
        let name = asset
            .skeleton
            .nodes
            .get(max_trans_idx)
            .map(|n| n.name.as_str())
            .unwrap_or("?");
        eprintln!(
            "    translation audit: max |mi[3]|={:.3} m at bone[{max_trans_idx}]='{name}' xyz=({:+.3},{:+.3},{:+.3})",
            max_trans_mag, max_trans_xyz[0], max_trans_xyz[1], max_trans_xyz[2]
        );
    }
}

fn detect_bone_breakage(
    asset: &Arc<AvatarAsset>,
    avatar: &AvatarInstance,
    dump_path: &Path,
    threshold_m: f32,
) -> Result<(f32, String), String> {
    use std::fmt::Write as _;
    let humanoid = asset.humanoid.as_ref().ok_or("no humanoid map")?;
    let hips_idx = humanoid
        .bone_map
        .get(&HumanoidBone::Hips)
        .map(|n| n.0 as usize)
        .ok_or("Hips bone missing from humanoid map")?;
    let hips_m = avatar
        .pose
        .global_transforms
        .get(hips_idx)
        .ok_or("Hips global transform missing")?;
    let hips_pos = [hips_m[3][0], hips_m[3][1], hips_m[3][2]];

    let mut s = String::new();
    let _ = writeln!(
        s,
        "hips world position: ({:+.3},{:+.3},{:+.3})\n",
        hips_pos[0], hips_pos[1], hips_pos[2]
    );
    let _ = writeln!(s, "                bone | world(x,y,z)              | dist_from_hips_m");
    let _ = writeln!(s, "-------------------- | ------------------------ | ----------------");

    let mut max_dist = 0.0f32;
    let mut worst = String::from("(none)");
    for node in &asset.skeleton.nodes {
        let idx = node.id.0 as usize;
        let Some(m) = avatar.pose.global_transforms.get(idx) else { continue };
        let p = [m[3][0], m[3][1], m[3][2]];
        let d = ((p[0] - hips_pos[0]).powi(2)
            + (p[1] - hips_pos[1]).powi(2)
            + (p[2] - hips_pos[2]).powi(2))
        .sqrt();
        let flag = if d > threshold_m { " *BROKEN*" } else { "" };
        let _ = writeln!(
            s,
            "{:>20} | ({:+.3},{:+.3},{:+.3}) | {:>10.3}{flag}",
            node.name, p[0], p[1], p[2], d,
        );
        if d > max_dist {
            max_dist = d;
            worst = node.name.clone();
        }
    }
    std::fs::write(dump_path, s).map_err(|e| format!("write dump: {e}"))?;
    Ok((max_dist, worst))
}

fn save_rgba_png(path: &Path, extent: [u32; 2], pixels: &[u8]) -> Result<(), String> {
    let img: ImageBuffer<Rgba<u8>, _> =
        ImageBuffer::from_raw(extent[0], extent[1], pixels.to_vec())
            .ok_or_else(|| "PNG buffer build failed".to_string())?;
    img.save(path).map_err(|e| format!("PNG save: {e}"))
}

fn collect_images(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    walk(root, &mut out).map_err(|e| format!("walk {}: {e}", root.display()))?;
    Ok(out)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let p = entry.path();
        if p.is_dir() {
            walk(&p, out)?;
        } else if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
            let ext = ext.to_ascii_lowercase();
            if ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "webp" {
                out.push(p);
            }
        }
    }
    Ok(())
}
