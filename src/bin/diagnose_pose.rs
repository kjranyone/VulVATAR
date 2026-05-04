//! Pose-tracking visual diagnostic CLI.
//!
//! Loads an input image, runs RTMW3D whole-body inference on it
//! (body + hands), applies the result to a VRM via
//! `pose_solver::solve_avatar_pose`, and dumps the rendered avatar to
//! PNG. Used to verify whether the solver's bone rotations match the
//! input pose visually — the fastest way to catch axis-handedness /
//! mirror / sign-flip bugs the unit tests don't cover.
//!
//! Usage:
//!   cargo run --release --bin diagnose_pose -- \
//!       [--prime <calib.png>]... <image.png> <model.vrm> [<output_dir>]
//!
//! `--prime` runs inference on a calibration image and feeds the result
//! through the solver to warm up `PoseSolverState` (per-bone running
//! max of 2D length). The pose solver needs a frame where each limb is
//! fully extended in the image plane to learn the subject's true bone
//! lengths so it can recover Z (depth) on subsequent foreshortened
//! frames. Pass a T-pose or A-pose photo as `--prime` for clean Z
//! recovery on the test image. Multiple `--prime` flags are
//! accumulated.
//!
//! Outputs in `<output_dir>` (default: same dir as `<image.png>` with
//! `_diagnose` suffix):
//!   keypoints_overlay.png  — input image + detected COCO keypoints drawn
//!   avatar_rest.png        — VRM in T-pose (reference)
//!   avatar_posed.png       — VRM after solve_avatar_pose
//!   source_skeleton.txt    — text dump of {bone, position, confidence}

use std::path::{Path, PathBuf};
use std::sync::Arc;

use image::{ImageBuffer, Rgb, Rgba};
use vulvatar_lib::app::ViewportCamera;
use vulvatar_lib::asset::vrm::VrmAssetLoader;
use vulvatar_lib::asset::AvatarAsset;
use vulvatar_lib::avatar::pose_solver::{solve_avatar_pose, PoseSolverState, SolverParams};
use vulvatar_lib::avatar::{AvatarInstance, AvatarInstanceId};
use vulvatar_lib::renderer::frame_input::{
    CameraState, LightingState, OutputTargetRequest, RenderAlphaMode, RenderAvatarInstance,
    RenderColorSpace, RenderCullMode, RenderDebugFlags, RenderExportMode, RenderFrameInput,
    RenderMeshInstance, RenderOutputAlpha,
};
use vulvatar_lib::renderer::material::{MaterialShaderMode, MaterialUploadRequest};
use vulvatar_lib::renderer::VulkanRenderer;
use vulvatar_lib::tracking::rtmw3d::Rtmw3dInference;
use vulvatar_lib::tracking::source_skeleton::SourceSkeleton;
use vulvatar_lib::tracking::DetectionAnnotation;

fn main() -> Result<(), String> {
    env_logger::init();

    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut prime_paths: Vec<PathBuf> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--prime" {
            let p = args.get(i + 1).ok_or_else(|| "--prime requires a path".to_string())?.clone();
            prime_paths.push(PathBuf::from(p));
            args.drain(i..i + 2);
        } else {
            i += 1;
        }
    }
    let mut positional = args.into_iter();
    let image_path = PathBuf::from(positional.next().ok_or_else(|| {
        "usage: diagnose_pose [--prime <calib.png>]... <image.png> <model.vrm> [<output_dir>]"
            .to_string()
    })?);
    let vrm_path = PathBuf::from(
        positional
            .next()
            .ok_or_else(|| "missing <model.vrm>".to_string())?,
    );
    let out_dir = match positional.next() {
        Some(d) => PathBuf::from(d),
        None => {
            let stem = image_path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "diagnose".to_string());
            image_path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."))
                .join(format!("{stem}_diagnose"))
        }
    };
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("create_dir_all '{}': {e}", out_dir.display()))?;

    // 1. Load image as RGB.
    eprintln!("loading image {}", image_path.display());
    let img = image::open(&image_path).map_err(|e| format!("open image: {e}"))?;
    let rgb = img.to_rgb8();
    let (iw, ih) = (rgb.width(), rgb.height());
    eprintln!("image: {iw}x{ih}");

    // 2. Run inference (and any prime frames).
    eprintln!("loading models from models/");
    let mut infer = Rtmw3dInference::from_models_dir("models")
        .map_err(|e| format!("load Rtmw3dInference: {e}"))?;
    let warnings = infer.take_load_warnings();
    for w in &warnings {
        eprintln!("model load warning: {w}");
    }

    // Load VRM up front so the prime pass can run the solver against it.
    eprintln!("loading VRM {}", vrm_path.display());
    let loader = VrmAssetLoader::new();
    let asset = loader
        .load(
            vrm_path
                .to_str()
                .ok_or_else(|| format!("invalid VRM path: {}", vrm_path.display()))?,
        )
        .map_err(|e| format!("load VRM '{}': {e:?}", vrm_path.display()))?;

    // Warm up the per-bone calibration using the prime frames. Each
    // prime image runs through the same inference + solver pass so the
    // running max of observed 2D bone length lives in `solver_state` by
    // the time we hit the test image.
    let mut solver_state = PoseSolverState::default();
    for prime in &prime_paths {
        eprintln!("priming with {}", prime.display());
        let pimg = image::open(prime).map_err(|e| format!("open prime '{}': {e}", prime.display()))?;
        let prgb = pimg.to_rgb8();
        let (pw, ph) = (prgb.width(), prgb.height());
        let pestimate = infer.estimate_pose(prgb.as_raw(), pw, ph, 0);
        eprintln!(
            "  inferred {} keypoints (prime)",
            pestimate.annotation.keypoints.len(),
        );
        let mut prime_avatar = AvatarInstance::new(AvatarInstanceId(99), Arc::clone(&asset));
        prime_avatar.build_base_pose();
        let humanoid = asset.humanoid.as_ref();
        let params = SolverParams {
            rotation_blend: 1.0,
            joint_confidence_threshold: 0.1,
            face_confidence_threshold: 0.1,
            ..Default::default()
        };
        solve_avatar_pose(
            &pestimate.skeleton,
            &asset.skeleton,
            humanoid,
            &mut prime_avatar.pose.local_transforms,
            &params,
            &mut solver_state,
        );
    }

    // The prime calls left filter / hysteresis state pointing at
    // those frames; treat the test image as a fresh observation so
    // the 1€ filter does not blend prime's pose into the result.
    // Per-bone running-max calibration is preserved.
    solver_state.reset_motion_smoothing();

    eprintln!("running inference…");
    let estimate = infer.estimate_pose(rgb.as_raw(), iw, ih, 0);
    eprintln!(
        "inferred {} keypoints, bbox={:?}",
        estimate.annotation.keypoints.len(),
        estimate.annotation.bounding_box
    );

    // 3. Save keypoint overlay.
    save_keypoint_overlay(
        &img.to_rgba8(),
        &estimate.annotation,
        &out_dir.join("keypoints_overlay.png"),
    )?;

    // 4. Dump source skeleton joints for inspection.
    dump_source_skeleton(&estimate.skeleton, &out_dir.join("source_skeleton.txt"))?;

    // 5. Render rest pose (reference) and posed pose (solver applied).
    let extent = [1024u32, 1024];
    let mut renderer = VulkanRenderer::new();
    renderer.initialize();

    eprintln!("rendering rest pose…");
    let rest_avatar = make_avatar(&asset, None, &mut PoseSolverState::default());
    save_avatar_render(&mut renderer, &rest_avatar, extent, &out_dir.join("avatar_rest.png"))?;

    eprintln!("rendering posed (solver applied)…");
    let posed_avatar = make_avatar(&asset, Some(&estimate.skeleton), &mut solver_state);
    dump_arm_world_positions(&posed_avatar, &out_dir.join("arm_world_positions.txt"))?;
    save_avatar_render(
        &mut renderer,
        &posed_avatar,
        extent,
        &out_dir.join("avatar_posed.png"),
    )?;

    println!("written: {}", out_dir.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Avatar building
// ---------------------------------------------------------------------------

fn make_avatar(
    asset: &Arc<AvatarAsset>,
    source: Option<&SourceSkeleton>,
    solver_state: &mut PoseSolverState,
) -> AvatarInstance {
    let mut avatar = AvatarInstance::new(AvatarInstanceId(1), Arc::clone(asset));
    avatar.build_base_pose();
    if let Some(src) = source {
        let humanoid = asset.humanoid.as_ref();
        let params = SolverParams {
            // Snap straight to the solver output — no smoothing, no
            // partial blend. We want to see exactly what the solver
            // produces from this single frame.
            rotation_blend: 1.0,
            joint_confidence_threshold: 0.1,
            face_confidence_threshold: 0.1,
            ..Default::default()
        };
        solve_avatar_pose(
            src,
            &asset.skeleton,
            humanoid,
            &mut avatar.pose.local_transforms,
            &params,
            solver_state,
        );
    }
    avatar.compute_global_pose();
    avatar.build_skinning_matrices();
    avatar
}

// ---------------------------------------------------------------------------
// Rendering (mostly cribbed from render_matrix.rs)
// ---------------------------------------------------------------------------

fn save_avatar_render(
    renderer: &mut VulkanRenderer,
    avatar: &AvatarInstance,
    extent: [u32; 2],
    out_path: &Path,
) -> Result<(), String> {
    let frame_input = build_frame_input(avatar, extent);
    // Two-render pattern: first kicks off pipelined readback, second
    // harvests it. Without the second render the result has no pixels.
    let _warm = renderer
        .render(&frame_input)
        .map_err(|e| format!("warm-up render '{}': {e}", out_path.display()))?;
    let result = renderer
        .render(&frame_input)
        .map_err(|e| format!("render '{}': {e}", out_path.display()))?;
    let pixels = result
        .exported_frame
        .as_ref()
        .and_then(|f| f.cpu_pixel_data())
        .ok_or_else(|| format!("render '{}' returned no pixel data", out_path.display()))?;
    save_png_rgba(out_path, extent, pixels.as_slice())
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
                    primitive_data: Some(Arc::new(prim.clone())),
                    morph_weights: Vec::new(),
                }
            })
        })
        .collect();

    // Frame the whole standing avatar so we can compare arm posture against
    // the input. render_matrix used a tight head-only crop for color
    // validation; that's useless for a pose diff.
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
            cloth_deform: None,
            debug_flags: RenderDebugFlags::default(),
        }],
        output_request: OutputTargetRequest {
            preview_enabled: true,
            output_enabled: true,
            extent,
            color_space: RenderColorSpace::Srgb,
            alpha_mode: RenderOutputAlpha::Opaque,
            export_mode: RenderExportMode::CpuReadback,
        },
        background_image_path: None,
        show_ground_grid: false,
        background_color: [0.10, 0.10, 0.10],
        transparent_background: false,
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

// ---------------------------------------------------------------------------
// PNG output / overlay
// ---------------------------------------------------------------------------

fn save_png_rgba(path: &Path, extent: [u32; 2], pixels: &[u8]) -> Result<(), String> {
    let img: ImageBuffer<Rgba<u8>, _> =
        ImageBuffer::from_raw(extent[0], extent[1], pixels.to_vec())
            .ok_or_else(|| format!("PNG buffer construction failed for {}", path.display()))?;
    img.save(path)
        .map_err(|e| format!("save '{}': {e}", path.display()))
}

/// Draw COCO keypoints + skeleton lines on top of the source image.
/// Keypoints are stored in `DetectionAnnotation` as (x, y, conf) in
/// normalised [0, 1] image coords (0=left/top, 1=right/bottom).
fn save_keypoint_overlay(
    src: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    ann: &DetectionAnnotation,
    out_path: &Path,
) -> Result<(), String> {
    let mut img = src.clone();
    let (w, h) = (img.width(), img.height());

    // Draw skeleton lines first (behind dots).
    for &(a, b) in &ann.skeleton {
        if a >= ann.keypoints.len() || b >= ann.keypoints.len() {
            continue;
        }
        let (ax, ay, ac) = ann.keypoints[a];
        let (bx, by, bc) = ann.keypoints[b];
        if ac <= 0.0 || bc <= 0.0 {
            continue;
        }
        let p0 = (ax * w as f32, ay * h as f32);
        let p1 = (bx * w as f32, by * h as f32);
        draw_line(&mut img, p0, p1, Rgba([0, 200, 255, 255]));
    }

    // Bounding box.
    if let Some((x0, y0, x1, y1)) = ann.bounding_box {
        let p00 = (x0 * w as f32, y0 * h as f32);
        let p10 = (x1 * w as f32, y0 * h as f32);
        let p11 = (x1 * w as f32, y1 * h as f32);
        let p01 = (x0 * w as f32, y1 * h as f32);
        draw_line(&mut img, p00, p10, Rgba([255, 200, 0, 255]));
        draw_line(&mut img, p10, p11, Rgba([255, 200, 0, 255]));
        draw_line(&mut img, p11, p01, Rgba([255, 200, 0, 255]));
        draw_line(&mut img, p01, p00, Rgba([255, 200, 0, 255]));
    }

    // Keypoint dots: green if confident, red if low.
    for (i, &(x, y, c)) in ann.keypoints.iter().enumerate() {
        if c <= 0.0 {
            continue;
        }
        let cx = (x * w as f32) as i32;
        let cy = (y * h as f32) as i32;
        let color = if c > 0.3 {
            Rgba([0u8, 255, 0, 255])
        } else {
            Rgba([255, 80, 80, 255])
        };
        draw_disk(&mut img, cx, cy, 4, color);
        // Tag the first 17 (COCO body) with their index for sanity.
        if i < 17 {
            // 6x6 numeric tag — too tiny to render glyphs, skip.
        }
    }

    img.save(out_path)
        .map_err(|e| format!("save overlay '{}': {e}", out_path.display()))
}

fn draw_line(
    img: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    p0: (f32, f32),
    p1: (f32, f32),
    color: Rgba<u8>,
) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let dx = p1.0 - p0.0;
    let dy = p1.1 - p0.1;
    let steps = dx.abs().max(dy.abs()).max(1.0) as i32;
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let x = (p0.0 + dx * t) as i32;
        let y = (p0.1 + dy * t) as i32;
        if x >= 0 && x < w && y >= 0 && y < h {
            img.put_pixel(x as u32, y as u32, color);
        }
    }
}

fn draw_disk(img: &mut ImageBuffer<Rgba<u8>, Vec<u8>>, cx: i32, cy: i32, radius: i32, color: Rgba<u8>) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let r2 = radius * radius;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy > r2 {
                continue;
            }
            let x = cx + dx;
            let y = cy + dy;
            if x >= 0 && x < w && y >= 0 && y < h {
                img.put_pixel(x as u32, y as u32, color);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Source-skeleton text dump
// ---------------------------------------------------------------------------

/// FK-derived world positions of the upper-body chain. Pulls global
/// transforms (already computed by `make_avatar`) and writes their
/// translation column for each humanoid bone of interest. Lets us
/// compare "where the solver/IK said the elbow should be in source
/// space" against "where the avatar's bone actually ends up after
/// FK", separating bone-rotation correctness from skinning effects.
fn dump_arm_world_positions(
    avatar: &AvatarInstance,
    out_path: &Path,
) -> Result<(), String> {
    use std::fmt::Write;
    use vulvatar_lib::asset::HumanoidBone;
    let humanoid = avatar
        .asset
        .humanoid
        .as_ref()
        .ok_or_else(|| "asset has no HumanoidMap".to_string())?;
    let bones: &[HumanoidBone] = &[
        HumanoidBone::Hips,
        HumanoidBone::Spine,
        HumanoidBone::Chest,
        HumanoidBone::UpperChest,
        HumanoidBone::Neck,
        HumanoidBone::Head,
        HumanoidBone::LeftShoulder,
        HumanoidBone::LeftUpperArm,
        HumanoidBone::LeftLowerArm,
        HumanoidBone::LeftHand,
        HumanoidBone::RightShoulder,
        HumanoidBone::RightUpperArm,
        HumanoidBone::RightLowerArm,
        HumanoidBone::RightHand,
        HumanoidBone::LeftUpperLeg,
        HumanoidBone::LeftLowerLeg,
        HumanoidBone::LeftFoot,
        HumanoidBone::RightUpperLeg,
        HumanoidBone::RightLowerLeg,
        HumanoidBone::RightFoot,
    ];
    let mut s = String::new();
    let _ = writeln!(
        s,
        "World positions (avatar-units; +X = avatar's left, +Y = up, +Z = forward toward camera)",
    );
    for bone in bones {
        let Some(node) = humanoid.bone_map.get(bone).copied() else {
            let _ = writeln!(s, "  {bone:?}  <missing>");
            continue;
        };
        let idx = node.0 as usize;
        if idx >= avatar.pose.global_transforms.len() {
            let _ = writeln!(s, "  {bone:?}  <out of range>");
            continue;
        }
        let g = &avatar.pose.global_transforms[idx];
        // Translation column of the global matrix.
        let pos = [g[3][0], g[3][1], g[3][2]];
        let _ = writeln!(
            s,
            "  {bone:?}  pos=({:+.3},{:+.3},{:+.3})",
            pos[0], pos[1], pos[2]
        );
    }
    std::fs::write(out_path, s)
        .map_err(|e| format!("write '{}': {e}", out_path.display()))
}

fn dump_source_skeleton(skeleton: &SourceSkeleton, out_path: &Path) -> Result<(), String> {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "source_ts={} overall_confidence={:.3}",
        skeleton.source_timestamp, skeleton.overall_confidence,
    );
    let _ = writeln!(s, "joints (image-space, x∈[-aspect,+aspect] Y-up, +z toward camera):");
    let mut entries: Vec<_> = skeleton.joints.iter().collect();
    entries.sort_by_key(|(b, _)| format!("{:?}", b));
    for (bone, joint) in entries {
        let _ = writeln!(
            s,
            "  {:>22?}  pos=({:>+7.3}, {:>+7.3}, {:>+7.3})  conf={:.3}",
            bone,
            joint.position[0],
            joint.position[1],
            joint.position[2],
            joint.confidence,
        );
    }
    if !skeleton.fingertips.is_empty() {
        let _ = writeln!(s, "fingertips (toe / finger tips, keyed by parent bone):");
        let mut tips: Vec<_> = skeleton.fingertips.iter().collect();
        tips.sort_by_key(|(b, _)| format!("{:?}", b));
        for (bone, joint) in tips {
            let _ = writeln!(
                s,
                "  {:>22?}  pos=({:>+7.3}, {:>+7.3}, {:>+7.3})  conf={:.3}",
                bone,
                joint.position[0],
                joint.position[1],
                joint.position[2],
                joint.confidence,
            );
        }
    }
    if let Some(face) = skeleton.face {
        let _ = writeln!(
            s,
            "face: yaw={:>+6.2}° pitch={:>+6.2}° roll={:>+6.2}° conf={:.3}",
            face.yaw.to_degrees(),
            face.pitch.to_degrees(),
            face.roll.to_degrees(),
            face.confidence,
        );
    }
    if !skeleton.expressions.is_empty() {
        let _ = writeln!(s, "expressions (52 ARKit blendshapes + VRM presets):");
        let mut exprs: Vec<_> = skeleton.expressions.iter().collect();
        exprs.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));
        for e in exprs.iter().take(20) {
            let _ = writeln!(s, "  {:>22}  weight={:.3}", e.name, e.weight);
        }
        if exprs.len() > 20 {
            let _ = writeln!(s, "  … ({} more, weights ≤ {:.3})", exprs.len() - 20, exprs[20].weight);
        }
    }
    std::fs::write(out_path, s).map_err(|e| format!("write '{}': {e}", out_path.display()))
}

// silence unused-import in the rgb path on non-default builds
#[allow(dead_code)]
fn _silence_rgb_unused(_: Rgb<u8>) {}
