//! Batch pose-pipeline validation: solver-quality score.
//!
//! For each image in `validation_images/`, runs RTMW3D on the input
//! image, drives the avatar via the pose solver, and compares the
//! avatar's actual bone world directions (geometric truth from
//! `pose.global_transforms`) against the input source skeleton's bone
//! directions. Both are in the same Y-up, +Z-toward-camera frame so
//! direction vectors compare cleanly without unit conversions.
//!
//! This isolates **solver quality** from the noise introduced by re-
//! detecting RTMW3D on a rendered cartoon avatar (the RTMW3D model is
//! trained on photos and gives unstable joint positions on stylized
//! renders, especially for shoulders / forearms when arms hang at
//! sides — the silhouette of a sleeve looks nothing like a human arm
//! to the network). A perfectly-pose-preserving solver scores 0°; a
//! solver that produces a random pose scores ~90°.
//!
//! Optional `--render` flag re-enables the round-trip render + re-
//! detect path for visual spot-checking; without it, we skip rendering
//! entirely (avatar.compute_global_pose() is enough for the metric).
//!
//! Outputs:
//!   diagnostics/validation_pipeline/results.jsonl  — per-image scores
//!   diagnostics/validation_pipeline/summary.md     — aggregated stats
//!
//! Usage:
//!   cargo run --bin validate_pipeline -- [--limit N] [--render]
//!                                         [--debug-renders DIR]
//!
//! Speed:
//!   * Models and VRM load once and are reused.
//!   * No render in default mode → ~30ms/image vs ~170ms with render.
//!   * 263 images take ~8s on a warm cache (no render),
//!     ~45s with --render.

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
/// vector from `base` joint to `tip` joint in source space.
///
/// Anatomical-alignment caveat: in VRM rigs the `LeftShoulder` bone's
/// world position is the **inner** clavicle pivot (sternum end), not
/// the outer shoulder joint. The COCO source skeleton's `LeftShoulder`
/// keypoint, by contrast, is at the **outer** shoulder joint (where
/// the deltoid sits). Comparing the two directly compares different
/// anatomical points and bakes a constant offset into every metric.
/// To get apples-to-apples positions we use `LeftUpperArm`/
/// `LeftUpperLeg` whose world position *is* the actual shoulder/hip
/// joint in both skeletons.
const BONE_DIRS: &[(&str, HumanoidBone, HumanoidBone)] = &[
    ("L_upper_arm", HumanoidBone::LeftUpperArm,  HumanoidBone::LeftLowerArm),
    ("R_upper_arm", HumanoidBone::RightUpperArm, HumanoidBone::RightLowerArm),
    ("L_forearm",   HumanoidBone::LeftLowerArm,  HumanoidBone::LeftHand),
    ("R_forearm",   HumanoidBone::RightLowerArm, HumanoidBone::RightHand),
    ("L_thigh",     HumanoidBone::LeftUpperLeg,  HumanoidBone::LeftLowerLeg),
    ("R_thigh",     HumanoidBone::RightUpperLeg, HumanoidBone::RightLowerLeg),
    ("L_shin",      HumanoidBone::LeftLowerLeg,  HumanoidBone::LeftFoot),
    ("R_shin",      HumanoidBone::RightLowerLeg, HumanoidBone::RightFoot),
    ("neck",        HumanoidBone::UpperChest,    HumanoidBone::Head),
    ("shoulder_line", HumanoidBone::RightUpperArm, HumanoidBone::LeftUpperArm),
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
    let mut do_render = false;
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
                do_render = true;
                i += 2;
            }
            "--render" => {
                do_render = true;
                i += 1;
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

    let mut renderer = if do_render {
        eprintln!("initializing Vulkan renderer…");
        let mut r = VulkanRenderer::new();
        r.initialize();
        Some(r)
    } else {
        None
    };

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
    let mut results: Vec<ImageResult> = Vec::with_capacity(images.len());
    let total_start = Instant::now();
    for (idx, image_path) in images.iter().enumerate() {
        // Each validation image is an INDEPENDENT subject (different
        // person / camera / scale). PoseSolverState carries
        // `running_shoulder_x_span_max` which persists by design across
        // `reset_motion_smoothing` calls (one user's anatomy doesn't
        // change between frames in live tracking). Across-subject reuse
        // poisons the body-yaw computation: a wide T-pose subject
        // sets running_max high, then a narrower frontal subject's
        // shoulder span looks foreshortened and gets reported as a
        // bogus 50°+ yaw. Fix is to use a fresh state per image.
        let mut state = PoseSolverState::default();

        let res = process_image(image_path, &mut infer, &asset, renderer.as_mut(), &mut state, debug_renders.as_deref());
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
    renderer: Option<&mut VulkanRenderer>,
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

    // Step 1: inference on input image — the source skeleton the
    // solver works against and our ground-truth for comparison.
    let est_input = infer.estimate_pose(rgb_in.as_raw(), iw, ih, 0);
    let sk_input = est_input.skeleton;

    // Step 2: solve avatar pose. After this, the avatar's
    // pose.global_transforms hold each bone's world-space transform —
    // that's the geometric truth of what the solver produced. We
    // compare those directly against the input source skeleton, no
    // RTMW3D round-trip needed.
    let avatar = make_avatar(asset, Some(&sk_input), state);

    // Optional render + re-detect path for visual spot-checking.
    // Doesn't affect the score; just dumps PNGs + a side-by-side
    // bone-position dump so failure cases can be inspected.
    if let Some(renderer) = renderer {
        if let Ok(rendered_rgba) = render_avatar(renderer, &avatar, RENDER_EXTENT) {
            if let Some(dir) = debug_renders {
                if let Some(stem) = image_path.file_stem() {
                    let out = dir.join(format!("{}.png", stem.to_string_lossy()));
                    let _ = save_rgba_png(&out, RENDER_EXTENT, &rendered_rgba);
                    let dump = dir.join(format!("{}.bones.txt", stem.to_string_lossy()));
                    // Build a rest-pose-only avatar (no solver) so the
                    // dump can compare avatar rest geometry vs the
                    // solved-and-driven bone positions side by side.
                    let rest_avatar = make_avatar(asset, None, state);
                    let _ = dump_bone_positions(&dump, &sk_input, asset, &avatar, &rest_avatar);
                }
            }
        }
    }

    // Step 3: compare bone directions (avatar geometric truth vs input
    // skeleton). Both are in Y-up, +Z-toward-camera frames, so unit
    // direction vectors compare without any unit conversion.
    let bones = compare_bones_avatar_vs_source(&sk_input, asset, &avatar);
    let score = mean_angle(&bones);

    ImageResult {
        image: image_str,
        category,
        inferred_input: true,
        inferred_output: bones.iter().any(|b| b.angle_deg.is_some()),
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

/// Compare per-bone direction vectors between the source skeleton
/// (RTMW3D detection on the input image) and the avatar's actual
/// world-space bones (geometric truth from the solver) in 3D.
///
/// Both frames are Y-up, +Z toward camera, +X image-right with 1:1
/// physical scale on x/y/z (source x is aspect-pre-applied; z is in
/// the same units). Positions differ in absolute scale (source is
/// image-normalised, avatar is metres) but unit direction vectors
/// compare cleanly without conversion.
///
/// 3D was chosen over 2D after iter-2 to remove a metric artefact:
/// at side-profile views the image-plane projection of `hip_line` /
/// `shoulder_line` collapses to near-zero magnitude and the angle
/// becomes pure noise (177° = sign flip of an ε-direction, not a
/// pipeline failure). 3D direction stays well-defined at every yaw.
fn compare_bones_avatar_vs_source(
    src: &SourceSkeleton,
    asset: &Arc<AvatarAsset>,
    avatar: &AvatarInstance,
) -> Vec<PerBoneDelta> {
    let humanoid = match asset.humanoid.as_ref() {
        Some(h) => h,
        None => return Vec::new(),
    };
    let bone_node = |b: HumanoidBone| -> Option<usize> {
        humanoid.bone_map.get(&b).copied().map(|n| n.0 as usize)
    };
    let avatar_pos = |b: HumanoidBone| -> Option<[f32; 3]> {
        let idx = bone_node(b)?;
        let m = avatar.pose.global_transforms.get(idx)?;
        // Mat4 is column-major [col][row]; translation is column 3.
        Some([m[3][0], m[3][1], m[3][2]])
    };

    let mut out = Vec::with_capacity(BONE_DIRS.len());
    for (name, base, tip) in BONE_DIRS {
        let src_base = src.joints.get(base).copied();
        let src_tip = src.joints.get(tip).copied();
        let conf_in = src_base.map(|j| j.confidence).unwrap_or(0.0)
            .min(src_tip.map(|j| j.confidence).unwrap_or(0.0));

        let av_base = avatar_pos(*base);
        let av_tip = avatar_pos(*tip);
        // "conf_out" doesn't really apply here (avatar bones are
        // geometric truth, no detection confidence). Repurpose to
        // signal "bone present in avatar rig" so downstream filters
        // still work; 1.0 = fully present.
        let conf_out = if av_base.is_some() && av_tip.is_some() { 1.0 } else { 0.0 };

        let angle = match (src_base, src_tip, av_base, av_tip) {
            (Some(sb), Some(st), Some(ab), Some(at)) => {
                let ds = unit_dir_3d(sb.position, st.position);
                let da = unit_dir_3d(ab, at);
                match (ds, da) {
                    (Some(ds), Some(da)) => {
                        let dot = (ds[0]*da[0] + ds[1]*da[1] + ds[2]*da[2])
                            .clamp(-1.0, 1.0);
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

fn unit_dir_3d(a: [f32; 3], b: [f32; 3]) -> Option<[f32; 3]> {
    let v = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
    let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    if len < 1e-4 { None } else { Some([v[0]/len, v[1]/len, v[2]/len]) }
}

/// Side-by-side dump of source skeleton positions vs avatar bone
/// world positions for every bone in `BONE_DIRS`. Written under
/// `--debug-renders DIR` for any image so failure cases can be
/// inspected without re-running the whole pipeline.
fn dump_bone_positions(
    path: &Path,
    src: &SourceSkeleton,
    asset: &Arc<AvatarAsset>,
    solved: &AvatarInstance,
    rest: &AvatarInstance,
) -> Result<(), String> {
    use std::fmt::Write as _;

    let humanoid = match asset.humanoid.as_ref() {
        Some(h) => h,
        None => return Err("avatar has no humanoid map".into()),
    };
    let pos_in = |av: &AvatarInstance, b: HumanoidBone| -> Option<[f32; 3]> {
        let idx = humanoid.bone_map.get(&b).copied().map(|n| n.0 as usize)?;
        let m = av.pose.global_transforms.get(idx)?;
        Some([m[3][0], m[3][1], m[3][2]])
    };

    // Stable, dedup'd order matching BONE_DIRS appearance.
    let mut bones: Vec<HumanoidBone> = Vec::new();
    for (_, b, t) in BONE_DIRS {
        if !bones.contains(b) { bones.push(*b); }
        if !bones.contains(t) { bones.push(*t); }
    }

    // Quaternion → Y-axis rotation angle (degrees, in [-180, 180]).
    // Approximate: extract yaw from a unit quaternion (w, x, y, z).
    let quat_yaw_deg = |q: &[f32; 4]| -> f32 {
        // Standard yaw from quaternion (Y-up): atan2(2*(w*y + x*z), 1 - 2*(y*y + z*z))
        let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
        let siny = 2.0 * (w * y + x * z);
        let cosy = 1.0 - 2.0 * (y * y + z * z);
        siny.atan2(cosy).to_degrees()
    };
    // Quaternion of an avatar bone's WORLD rotation (extracted from
    // its Mat4 via the upper 3x3 block). Returns identity if missing
    // or if the matrix isn't a clean rotation (fallback for safety).
    let world_quat = |av: &AvatarInstance, b: HumanoidBone| -> Option<[f32; 4]> {
        let idx = humanoid.bone_map.get(&b).copied().map(|n| n.0 as usize)?;
        let m = av.pose.global_transforms.get(idx)?;
        // m is column-major; rotation is the upper 3x3 in cols 0,1,2.
        let r = [
            [m[0][0], m[1][0], m[2][0]],
            [m[0][1], m[1][1], m[2][1]],
            [m[0][2], m[1][2], m[2][2]],
        ];
        // Convert 3x3 rotation matrix → quaternion (Shoemake).
        let trace = r[0][0] + r[1][1] + r[2][2];
        let q = if trace > 0.0 {
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
                (r[0][2] - r[2][0]) / s,
            ]
        } else {
            let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
            [
                (r[0][2] + r[2][0]) / s,
                (r[1][2] + r[2][1]) / s,
                0.25 * s,
                (r[1][0] - r[0][1]) / s,
            ]
        };
        Some(q)
    };

    let mut s = String::new();
    // Header: per-bone yaw of solved vs rest, useful for spotting
    // body_yaw distribution issues that the position-only view hides.
    s.push_str("## Bone Y-axis rotation (extracted from world transform)\n\n");
    s.push_str("              bone | rest yaw | solved yaw | delta\n");
    s.push_str("------------------ | -------- | ---------- | ------\n");
    for b in &[
        HumanoidBone::Hips,
        HumanoidBone::Spine,
        HumanoidBone::Chest,
        HumanoidBone::UpperChest,
        HumanoidBone::Neck,
        HumanoidBone::LeftUpperLeg,
        HumanoidBone::RightUpperLeg,
        HumanoidBone::LeftShoulder,
        HumanoidBone::RightShoulder,
        HumanoidBone::LeftUpperArm,
        HumanoidBone::RightUpperArm,
    ] {
        let r = world_quat(rest, *b).map(|q| quat_yaw_deg(&q));
        let a = world_quat(solved, *b).map(|q| quat_yaw_deg(&q));
        let d = match (r, a) {
            (Some(r), Some(a)) => format!("{:+6.1}°", a - r),
            _ => "—".into(),
        };
        let _ = writeln!(s, "{:>18} | {:>7} | {:>9} | {}",
            format!("{b:?}"),
            r.map(|v| format!("{:+.1}°", v)).unwrap_or_else(|| "—".into()),
            a.map(|v| format!("{:+.1}°", v)).unwrap_or_else(|| "—".into()),
            d);
    }

    s.push_str("\n# Source vs avatar bone positions (rest = no-solver pose)\n\n");
    s.push_str("                       bone  | src(x,y,z,conf)              | rest(x,y,z)              | solved(x,y,z)\n");
    s.push_str("---------------------------- | ---------------------------- | ------------------------ | ------------------------\n");
    for b in &bones {
        let src_str = match src.joints.get(b) {
            Some(j) => format!("({:+.3},{:+.3},{:+.3},c={:.2})",
                j.position[0], j.position[1], j.position[2], j.confidence),
            None => "—".to_string(),
        };
        let rest_str = match pos_in(rest, *b) {
            Some(p) => format!("({:+.3},{:+.3},{:+.3})", p[0], p[1], p[2]),
            None => "—".to_string(),
        };
        let solved_str = match pos_in(solved, *b) {
            Some(p) => format!("({:+.3},{:+.3},{:+.3})", p[0], p[1], p[2]),
            None => "—".to_string(),
        };
        let _ = writeln!(s, "{:>28} | {:<28} | {:<24} | {}",
            format!("{b:?}"), src_str, rest_str, solved_str);
    }

    s.push_str("\n## Per-bone direction (3D unit vectors)\n\n");
    s.push_str("              bone | src dir                  | rest dir                 | solved dir               | angle(src,solved)\n");
    s.push_str("------------------ | ------------------------ | ------------------------ | ------------------------ | -----------------\n");
    for (name, base, tip) in BONE_DIRS {
        let sb = src.joints.get(base).copied();
        let st = src.joints.get(tip).copied();
        let rb = pos_in(rest, *base);
        let rt = pos_in(rest, *tip);
        let ab = pos_in(solved, *base);
        let at = pos_in(solved, *tip);
        let fmt_dir = |d: Option<[f32; 3]>| d
            .map(|d| format!("({:+.3},{:+.3},{:+.3})", d[0], d[1], d[2]))
            .unwrap_or_else(|| "—".into());
        let ds = sb.zip(st).and_then(|(a, b)| unit_dir_3d(a.position, b.position));
        let dr = rb.zip(rt).and_then(|(a, b)| unit_dir_3d(a, b));
        let da = ab.zip(at).and_then(|(a, b)| unit_dir_3d(a, b));
        let ang = match (ds, da) {
            (Some(ds), Some(da)) => {
                let dot = (ds[0]*da[0] + ds[1]*da[1] + ds[2]*da[2]).clamp(-1.0, 1.0);
                format!("{:.1}°", dot.acos().to_degrees())
            }
            _ => "—".into(),
        };
        let _ = writeln!(s, "{:>18} | {:<24} | {:<24} | {:<24} | {}",
            name, fmt_dir(ds), fmt_dir(dr), fmt_dir(da), ang);
    }

    std::fs::write(path, s).map_err(|e| format!("write dump: {e}"))
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
