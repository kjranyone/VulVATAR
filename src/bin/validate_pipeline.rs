//! Batch pose-pipeline validation: round-trip detection score.
//!
//! For each image in `validation_images/`, runs the full inference +
//! solver + render pipeline, then re-runs RTMW3D on the rendered avatar
//! and compares per-bone direction vectors against the input skeleton.
//! The score is the mean angular error (degrees) across major bones.
//! A perfectly-pose-preserving pipeline scores 0°; a totally broken one
//! scores ~90° (random directions).
//!
//! Outputs:
//!   diagnostics/validation_pipeline/results.jsonl  — per-image scores
//!   diagnostics/validation_pipeline/summary.md     — aggregated by category
//!
//! Usage:
//!   cargo run --bin validate_pipeline -- [--limit N] [--debug-renders DIR]
//!
//! Speed:
//!   * Models, VRM, and Vulkan renderer load once and are reused.
//!   * Render output is 512×512 (4× fewer pixels vs. 1024² → ~4× faster
//!     RTMW3D pass on the rendered side too).
//!   * No per-image PNG writes unless --debug-renders is passed.
//!   * 263 images currently take roughly 5–7 minutes on a warm cache.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use image::{ImageBuffer, Rgba};
use vulvatar_lib::app::ViewportCamera;
use vulvatar_lib::asset::vrm::VrmAssetLoader;
use vulvatar_lib::asset::AvatarAsset;
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
use vulvatar_lib::tracking::rtmw3d::Rtmw3dInference;
use vulvatar_lib::tracking::source_skeleton::SourceSkeleton;

const RENDER_EXTENT: [u32; 2] = [512, 512];
const VRM_PATH: &str = "sample_data/AvatarSample_A.vrm";
const VALIDATION_ROOT: &str = "validation_images";
const OUT_DIR: &str = "diagnostics/validation_pipeline";

/// Bones whose direction vector is meaningful for pose comparison.
/// Tuples of (bone, base, tip) where the bone's *direction* is the
/// vector from `base` joint to `tip` joint in source space. Skip
/// fingers (too noisy), Hips/Chest/Spine (synthesized at origin so
/// direction is degenerate).
const BONE_DIRS: &[(&str, HumanoidBone, HumanoidBone)] = &[
    ("L_upper_arm", HumanoidBone::LeftShoulder, HumanoidBone::LeftLowerArm),
    ("R_upper_arm", HumanoidBone::RightShoulder, HumanoidBone::RightLowerArm),
    ("L_forearm",   HumanoidBone::LeftLowerArm,  HumanoidBone::LeftHand),
    ("R_forearm",   HumanoidBone::RightLowerArm, HumanoidBone::RightHand),
    ("L_thigh",     HumanoidBone::LeftUpperLeg,  HumanoidBone::LeftLowerLeg),
    ("R_thigh",     HumanoidBone::RightUpperLeg, HumanoidBone::RightLowerLeg),
    ("L_shin",      HumanoidBone::LeftLowerLeg,  HumanoidBone::LeftFoot),
    ("R_shin",      HumanoidBone::RightLowerLeg, HumanoidBone::RightFoot),
    ("neck",        HumanoidBone::UpperChest,    HumanoidBone::Head),
    ("shoulder_line", HumanoidBone::RightShoulder, HumanoidBone::LeftShoulder),
    ("hip_line",    HumanoidBone::RightUpperLeg, HumanoidBone::LeftUpperLeg),
];

#[derive(Debug, Clone)]
struct PerBoneDelta {
    name: &'static str,
    angle_deg: Option<f32>,    // None when bone unavailable in either frame
    conf_in: f32,
    conf_out: f32,
}

#[derive(Debug, Clone)]
struct ImageResult {
    image: String,
    category: String,
    inferred_input: bool,
    inferred_output: bool,
    score_deg: Option<f32>, // mean angular error, None if no comparable bones
    bones: Vec<PerBoneDelta>,
    duration_ms: u128,
}

fn main() -> Result<(), String> {
    env_logger::init();

    // -------- arg parsing --------
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut limit: Option<usize> = None;
    let mut debug_renders: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--limit" => {
                limit = Some(args.get(i + 1)
                    .ok_or_else(|| "--limit requires N".to_string())?
                    .parse().map_err(|e| format!("--limit parse: {e}"))?);
                i += 2;
            }
            "--debug-renders" => {
                debug_renders = Some(PathBuf::from(args.get(i + 1)
                    .ok_or_else(|| "--debug-renders requires DIR".to_string())?));
                i += 2;
            }
            _ => return Err(format!("unknown arg: {}", args[i])),
        }
    }

    // -------- load models / renderer once --------
    eprintln!("loading RTMW3D…");
    let mut infer = Rtmw3dInference::from_models_dir("models")
        .map_err(|e| format!("load Rtmw3dInference: {e}"))?;
    for w in infer.take_load_warnings() {
        eprintln!("  warning: {w}");
    }

    eprintln!("loading VRM {VRM_PATH}…");
    let asset = VrmAssetLoader::new()
        .load(VRM_PATH)
        .map_err(|e| format!("load VRM: {e:?}"))?;

    eprintln!("initializing Vulkan renderer…");
    let mut renderer = VulkanRenderer::new();
    renderer.initialize();

    // -------- collect images --------
    let mut images = collect_images(Path::new(VALIDATION_ROOT))?;
    images.sort();
    if let Some(n) = limit {
        images.truncate(n);
    }
    eprintln!("validating {} image(s)", images.len());

    if let Some(dir) = &debug_renders {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir debug: {e}"))?;
    }
    let out_dir = Path::new(OUT_DIR);
    std::fs::create_dir_all(out_dir).map_err(|e| format!("mkdir out: {e}"))?;
    let jsonl_path = out_dir.join("results.jsonl");
    let mut jsonl = BufWriter::new(File::create(&jsonl_path)
        .map_err(|e| format!("create jsonl: {e}"))?);

    // -------- per-image loop --------
    let mut state = PoseSolverState::default();
    let mut results: Vec<ImageResult> = Vec::with_capacity(images.len());
    let total_start = Instant::now();
    for (idx, image_path) in images.iter().enumerate() {
        // Reset filter state between images so each is an independent
        // observation; running_shoulder_x_span_max persists by design
        // (anatomical, survives reset_motion_smoothing).
        state.reset_motion_smoothing();

        let res = process_image(image_path, &mut infer, &asset, &mut renderer, &mut state, debug_renders.as_deref());
        let summary = res.score_deg
            .map(|s| format!("{s:5.1}°"))
            .unwrap_or_else(|| "  N/A".to_string());
        eprintln!("[{:>3}/{}] {} {} ({}ms)",
            idx + 1, images.len(), summary, res.image, res.duration_ms);
        write_jsonl_row(&mut jsonl, &res)?;
        results.push(res);
    }
    jsonl.flush().map_err(|e| format!("flush jsonl: {e}"))?;

    eprintln!("\ntotal: {:.1}s", total_start.elapsed().as_secs_f32());
    write_summary(&results, &out_dir.join("summary.md"))?;
    eprintln!("wrote: {}", out_dir.display());
    Ok(())
}

// -----------------------------------------------------------------------
// Image processing
// -----------------------------------------------------------------------

fn process_image(
    image_path: &Path,
    infer: &mut Rtmw3dInference,
    asset: &Arc<AvatarAsset>,
    renderer: &mut VulkanRenderer,
    state: &mut PoseSolverState,
    debug_renders: Option<&Path>,
) -> ImageResult {
    let started = Instant::now();
    let image_str = image_path.to_string_lossy().into_owned();
    let category = categorize(image_path);

    let img = match image::open(image_path) {
        Ok(i) => i,
        Err(e) => return ImageResult {
            image: image_str, category, inferred_input: false, inferred_output: false,
            score_deg: None, bones: Vec::new(),
            duration_ms: started.elapsed().as_millis(),
        }.with_error(format!("open: {e}")),
    };
    let rgb_in = img.to_rgb8();
    let (iw, ih) = (rgb_in.width(), rgb_in.height());

    // Step 1: inference on input image.
    let est_input = infer.estimate_pose(rgb_in.as_raw(), iw, ih, 0);
    let sk_input = est_input.skeleton;

    // Step 2: solve avatar pose.
    let avatar = make_avatar(asset, Some(&sk_input), state);

    // Step 3: render avatar at small extent for speed.
    let rendered_rgba = match render_avatar(renderer, &avatar, RENDER_EXTENT) {
        Ok(p) => p,
        Err(e) => return ImageResult {
            image: image_str, category, inferred_input: true, inferred_output: false,
            score_deg: None, bones: Vec::new(),
            duration_ms: started.elapsed().as_millis(),
        }.with_error(format!("render: {e}")),
    };

    // Optional debug PNG (preview the rendered avatar for spot-check).
    if let Some(dir) = debug_renders {
        if let Some(stem) = image_path.file_stem() {
            let out = dir.join(format!("{}.png", stem.to_string_lossy()));
            let _ = save_rgba_png(&out, RENDER_EXTENT, &rendered_rgba);
        }
    }

    // Step 4: re-inference on rendered avatar.
    // Convert RGBA → RGB (RTMW3D wants RGB).
    let mut rgb_out: Vec<u8> = Vec::with_capacity(
        (RENDER_EXTENT[0] * RENDER_EXTENT[1] * 3) as usize,
    );
    for px in rendered_rgba.chunks_exact(4) {
        rgb_out.extend_from_slice(&px[..3]);
    }
    let est_output = infer.estimate_pose(&rgb_out, RENDER_EXTENT[0], RENDER_EXTENT[1], 1);
    let sk_output = est_output.skeleton;

    // Step 5: compare bone directions.
    let bones = compare_bones(&sk_input, &sk_output);
    let score = mean_angle(&bones);

    ImageResult {
        image: image_str,
        category,
        inferred_input: true,
        inferred_output: true,
        score_deg: score,
        bones,
        duration_ms: started.elapsed().as_millis(),
    }
}

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

// -----------------------------------------------------------------------
// Bone comparison
// -----------------------------------------------------------------------

fn compare_bones(a: &SourceSkeleton, b: &SourceSkeleton) -> Vec<PerBoneDelta> {
    let mut out = Vec::with_capacity(BONE_DIRS.len());
    for (name, base, tip) in BONE_DIRS {
        let a_base = a.joints.get(base).copied();
        let a_tip = a.joints.get(tip).copied();
        let b_base = b.joints.get(base).copied();
        let b_tip = b.joints.get(tip).copied();

        let conf_in = a_base.map(|j| j.confidence).unwrap_or(0.0)
            .min(a_tip.map(|j| j.confidence).unwrap_or(0.0));
        let conf_out = b_base.map(|j| j.confidence).unwrap_or(0.0)
            .min(b_tip.map(|j| j.confidence).unwrap_or(0.0));

        let angle = match (a_base, a_tip, b_base, b_tip) {
            (Some(ab), Some(at), Some(bb), Some(bt)) => {
                let da = unit_dir(ab.position, at.position);
                let db = unit_dir(bb.position, bt.position);
                match (da, db) {
                    (Some(da), Some(db)) => {
                        let dot = (da[0]*db[0] + da[1]*db[1] + da[2]*db[2]).clamp(-1.0, 1.0);
                        Some(dot.acos().to_degrees())
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        out.push(PerBoneDelta { name, angle_deg: angle, conf_in, conf_out });
    }
    out
}

fn unit_dir(a: [f32; 3], b: [f32; 3]) -> Option<[f32; 3]> {
    let v = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
    let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    if len < 1e-4 { None } else { Some([v[0]/len, v[1]/len, v[2]/len]) }
}

fn mean_angle(deltas: &[PerBoneDelta]) -> Option<f32> {
    let valid: Vec<f32> = deltas.iter().filter_map(|d| d.angle_deg).collect();
    if valid.is_empty() { None } else { Some(valid.iter().sum::<f32>() / valid.len() as f32) }
}

// -----------------------------------------------------------------------
// Categorization (top-level dir under validation_images/)
// -----------------------------------------------------------------------

fn categorize(p: &Path) -> String {
    let root = Path::new(VALIDATION_ROOT);
    p.strip_prefix(root)
        .ok()
        .and_then(|rel| rel.iter().next())
        .map(|c| c.to_string_lossy().into_owned())
        .unwrap_or_else(|| "uncategorized".to_string())
}

// -----------------------------------------------------------------------
// Image discovery
// -----------------------------------------------------------------------

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
            if ext == "png" || ext == "jpg" || ext == "jpeg" {
                out.push(p);
            }
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------
// Output writers
// -----------------------------------------------------------------------

fn write_jsonl_row(w: &mut impl Write, r: &ImageResult) -> Result<(), String> {
    let mut bones_json = String::from("{");
    for (i, b) in r.bones.iter().enumerate() {
        if i > 0 { bones_json.push(','); }
        let angle = b.angle_deg.map(|a| format!("{a:.2}"))
            .unwrap_or_else(|| "null".to_string());
        bones_json.push_str(&format!(
            "\"{}\":{{\"angle_deg\":{},\"conf_in\":{:.2},\"conf_out\":{:.2}}}",
            b.name, angle, b.conf_in, b.conf_out
        ));
    }
    bones_json.push('}');
    let score = r.score_deg.map(|s| format!("{s:.2}"))
        .unwrap_or_else(|| "null".to_string());
    let line = format!(
        "{{\"image\":\"{}\",\"category\":\"{}\",\"inferred_input\":{},\"inferred_output\":{},\"score_deg\":{},\"duration_ms\":{},\"bones\":{}}}\n",
        json_escape(&r.image), json_escape(&r.category),
        r.inferred_input, r.inferred_output, score, r.duration_ms, bones_json,
    );
    w.write_all(line.as_bytes()).map_err(|e| format!("write jsonl: {e}"))
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn write_summary(results: &[ImageResult], path: &Path) -> Result<(), String> {
    // Per-category aggregate.
    let mut by_cat: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut by_bone: BTreeMap<&'static str, Vec<f32>> = BTreeMap::new();
    let mut total: Vec<f32> = Vec::new();
    let mut failed = 0usize;
    for r in results {
        if let Some(s) = r.score_deg {
            by_cat.entry(r.category.clone()).or_default().push(s);
            total.push(s);
        } else {
            failed += 1;
        }
        for b in &r.bones {
            if let Some(a) = b.angle_deg {
                by_bone.entry(b.name).or_default().push(a);
            }
        }
    }

    let mut s = String::new();
    s.push_str("# Pose pipeline round-trip validation\n\n");
    s.push_str(&format!("Total images: **{}**, scored: **{}**, failed: **{}**\n\n",
        results.len(), total.len(), failed));
    if !total.is_empty() {
        let mean = total.iter().sum::<f32>() / total.len() as f32;
        let mut sorted = total.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = sorted[sorted.len()/2];
        let p90 = sorted[(sorted.len()*9/10).min(sorted.len()-1)];
        let max = *sorted.last().unwrap();
        s.push_str(&format!(
            "Overall: **mean {:.1}°**, p50 {:.1}°, p90 {:.1}°, max {:.1}°\n\n",
            mean, p50, p90, max
        ));
    }

    s.push_str("## Per category\n\n");
    s.push_str("| category | n | mean | p50 | p90 | max |\n");
    s.push_str("|---|---|---|---|---|---|\n");
    for (cat, scores) in &by_cat {
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = sorted.iter().sum::<f32>() / sorted.len() as f32;
        let p50 = sorted[sorted.len()/2];
        let p90 = sorted[(sorted.len()*9/10).min(sorted.len()-1)];
        let max = *sorted.last().unwrap();
        s.push_str(&format!("| {} | {} | {:.1}° | {:.1}° | {:.1}° | {:.1}° |\n",
            cat, sorted.len(), mean, p50, p90, max));
    }

    s.push_str("\n## Per bone (across all images)\n\n");
    s.push_str("| bone | n | mean | p50 | p90 | max |\n");
    s.push_str("|---|---|---|---|---|---|\n");
    for (bone, scores) in &by_bone {
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = sorted.iter().sum::<f32>() / sorted.len() as f32;
        let p50 = sorted[sorted.len()/2];
        let p90 = sorted[(sorted.len()*9/10).min(sorted.len()-1)];
        let max = *sorted.last().unwrap();
        s.push_str(&format!("| {} | {} | {:.1}° | {:.1}° | {:.1}° | {:.1}° |\n",
            bone, sorted.len(), mean, p50, p90, max));
    }

    s.push_str("\n## Worst 10 images\n\n");
    let mut by_score: Vec<&ImageResult> = results.iter().filter(|r| r.score_deg.is_some()).collect();
    by_score.sort_by(|a, b| b.score_deg.unwrap().partial_cmp(&a.score_deg.unwrap()).unwrap());
    s.push_str("| score | image |\n|---|---|\n");
    for r in by_score.iter().take(10) {
        s.push_str(&format!("| {:.1}° | `{}` |\n", r.score_deg.unwrap(), r.image));
    }

    s.push_str("\n## Best 10 images\n\n");
    by_score.sort_by(|a, b| a.score_deg.unwrap().partial_cmp(&b.score_deg.unwrap()).unwrap());
    s.push_str("| score | image |\n|---|---|\n");
    for r in by_score.iter().take(10) {
        s.push_str(&format!("| {:.1}° | `{}` |\n", r.score_deg.unwrap(), r.image));
    }

    std::fs::write(path, s).map_err(|e| format!("write summary: {e}"))
}

// -----------------------------------------------------------------------
// Rendering helpers (cribbed from diagnose_pose.rs)
// -----------------------------------------------------------------------

fn render_avatar(
    renderer: &mut VulkanRenderer,
    avatar: &AvatarInstance,
    extent: [u32; 2],
) -> Result<Vec<u8>, String> {
    let frame_input = build_frame_input(avatar, extent);
    let _warm = renderer.render(&frame_input).map_err(|e| format!("warm render: {e}"))?;
    let result = renderer.render(&frame_input).map_err(|e| format!("render: {e}"))?;
    let pixels = result.exported_frame
        .as_ref()
        .and_then(|f| f.cpu_pixel_data())
        .ok_or_else(|| "no pixel data".to_string())?;
    Ok((*pixels).clone())
}

fn build_frame_input(avatar: &AvatarInstance, extent: [u32; 2]) -> RenderFrameInput {
    let mesh_instances: Vec<RenderMeshInstance> = avatar.asset.meshes.iter()
        .flat_map(|mesh| mesh.primitives.iter().map(|prim| {
            let material_asset = avatar.asset.materials.iter().find(|m| m.id == prim.material_id);
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
        })).collect();

    let camera = ViewportCamera {
        distance: 3.5,
        pan: [0.0, 1.0],
        ..ViewportCamera::default()
    };
    let (view, eye_pos) = build_view_matrix(&camera);
    let projection = build_projection_matrix(camera.fov_deg,
        extent[0] as f32 / extent[1].max(1) as f32, 0.1, 1000.0);

    RenderFrameInput {
        camera: CameraState { view, projection, position_ws: eye_pos, viewport_extent: extent },
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
    let len = (fwd[0]*fwd[0] + fwd[1]*fwd[1] + fwd[2]*fwd[2]).sqrt().max(1e-6);
    let f = [fwd[0]/len, fwd[1]/len, fwd[2]/len];
    let world_up = [0.0f32, 1.0, 0.0];
    let r = [f[1]*world_up[2]-f[2]*world_up[1], f[2]*world_up[0]-f[0]*world_up[2], f[0]*world_up[1]-f[1]*world_up[0]];
    let rlen = (r[0]*r[0]+r[1]*r[1]+r[2]*r[2]).sqrt().max(1e-6);
    let r = [r[0]/rlen, r[1]/rlen, r[2]/rlen];
    let u = [r[1]*f[2]-r[2]*f[1], r[2]*f[0]-r[0]*f[2], r[0]*f[1]-r[1]*f[0]];
    (
        [
            [r[0], r[1], r[2], -(r[0]*eye_x + r[1]*eye_y + r[2]*eye_z)],
            [u[0], u[1], u[2], -(u[0]*eye_x + u[1]*eye_y + u[2]*eye_z)],
            [-f[0], -f[1], -f[2], (f[0]*eye_x + f[1]*eye_y + f[2]*eye_z)],
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
        [f/aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0],
        [0.0, 0.0, a, b],
        [0.0, 0.0, -1.0, 0.0],
    ]
}

fn save_rgba_png(path: &Path, extent: [u32; 2], pixels: &[u8]) -> Result<(), String> {
    let img: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_raw(extent[0], extent[1], pixels.to_vec())
        .ok_or_else(|| "PNG buffer construction failed".to_string())?;
    img.save(path).map_err(|e| format!("PNG save: {e}"))
}

// -----------------------------------------------------------------------
// ImageResult helper
// -----------------------------------------------------------------------

impl ImageResult {
    fn with_error(mut self, msg: String) -> Self {
        eprintln!("  error: {msg}");
        self.bones.clear();
        self.score_deg = None;
        self
    }
}
