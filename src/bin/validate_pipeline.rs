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

use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba, RgbaImage};
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
use vulvatar_lib::tracking::provider::{create_pose_provider, PoseProvider, PoseProviderKind};
use vulvatar_lib::tracking::source_skeleton::SourceSkeleton;

const RENDER_EXTENT: [u32; 2] = [1536, 1536];
// Allow VRM swap via env so finger rigging can be A/B'd between the
// loose sample avatar (AvatarSample_A — has only mitten-style hands)
// and other test rigs.
const VRM_PATH_DEFAULT: &str = "sample_data/AvatarSample_A.vrm";
fn vrm_path() -> String {
    std::env::var("VULVATAR_VRM").unwrap_or_else(|_| VRM_PATH_DEFAULT.to_string())
}
const VALIDATION_ROOT: &str = "validation_images";
const OUT_DIR_DEFAULT: &str = "diagnostics/validation_pipeline";

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
    let mut provider_kind = "rtmw3d";
    let mut out_dir_override: Option<PathBuf> = None;
    let mut filter: Option<String> = None;
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
            "--provider" => {
                let kind = args.get(i + 1)
                    .ok_or_else(|| "--provider requires {rtmw3d|rtmw3d-with-depth|cigpose-metric-depth|vitpose-wholebody|vitpose-with-depth}".to_string())?;
                provider_kind = match kind.as_str() {
                    "rtmw3d" => "rtmw3d",
                    "rtmw3d-with-depth" => "rtmw3d-with-depth",
                    "cigpose-metric-depth" => "cigpose-metric-depth",
                    "vitpose" | "vitpose-wholebody" => "vitpose-wholebody",
                    "vitpose-with-depth" | "vitpose-dav2" => "vitpose-with-depth",
                    "hybrid"
                    | "vitpose-rtmw3d"
                    | "vitpose+rtmw3d"
                    | "vitpose-rtmw3d-hybrid" => "vitpose-rtmw3d-hybrid",
                    "hmr2" | "4d-humans" | "smpl" => "hmr2",
                    other => return Err(format!("unknown --provider value: {other}")),
                };
                i += 2;
            }
            "--out-dir" => {
                out_dir_override = Some(PathBuf::from(args.get(i + 1)
                    .ok_or_else(|| "--out-dir requires DIR".to_string())?));
                i += 2;
            }
            "--filter" => {
                filter = Some(args.get(i + 1)
                    .ok_or_else(|| "--filter requires PATTERN".to_string())?
                    .clone());
                i += 2;
            }
            _ => return Err(format!("unknown arg: {}", args[i])),
        }
    }

    // Force YOLOX to refresh every frame so each validation image
    // gets its own person bbox — the default period of 4 reuses the
    // previous image's bbox, which is fine for live tracking (same
    // subject moves slowly between frames) but catastrophic for an
    // image-by-image benchmark where every image is a different
    // subject in a different frame.
    vulvatar_lib::tracking::rtmw3d::YOLOX_REFRESH_PERIOD
        .store(1, std::sync::atomic::Ordering::Relaxed);

    // -------- load provider / renderer once --------
    let kind = match provider_kind {
        "rtmw3d-with-depth" => PoseProviderKind::Rtmw3dWithDepth,
        "cigpose-metric-depth" => PoseProviderKind::CigposeMetricDepth,
        "vitpose-wholebody" => PoseProviderKind::VitposeWholebody,
        "vitpose-with-depth" => PoseProviderKind::VitposeWithDepth,
        "vitpose-rtmw3d-hybrid" => PoseProviderKind::VitposeRtmw3dHybrid,
        "hmr2" => PoseProviderKind::Hmr2,
        _ => PoseProviderKind::Rtmw3d,
    };
    eprintln!("loading provider {provider_kind}…");
    let mut infer = create_pose_provider(kind, "models")
        .map_err(|e| format!("load provider: {e}"))?;
    for w in infer.take_load_warnings() {
        eprintln!("  warning: {w}");
    }

    let vrm = vrm_path();
    eprintln!("loading VRM {vrm}…");
    let asset = VrmAssetLoader::new()
        .load(&vrm)
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
    if let Some(pattern) = filter.as_deref() {
        images.retain(|p| p.to_string_lossy().contains(pattern));
    }
    if let Some(n) = limit {
        images.truncate(n);
    }
    eprintln!("validating {} image(s)", images.len());

    // Resolve `out_dir`. When `--out-dir` is given that wins; else
    // default to `OUT_DIR_DEFAULT`. When `--render` is on and no
    // explicit `--debug-renders` was passed, also point dbg/ at
    // `out_dir/dbg` so a per-provider run keeps its avatar PNGs next
    // to its own results.jsonl (otherwise two providers writing to
    // the same dbg/ would clobber each other's renders).
    let out_dir: PathBuf = out_dir_override
        .clone()
        .unwrap_or_else(|| PathBuf::from(OUT_DIR_DEFAULT));
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("mkdir out: {e}"))?;
    if do_render && debug_renders.is_none() {
        debug_renders = Some(out_dir.join("dbg"));
    }
    if let Some(dir) = &debug_renders {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir debug: {e}"))?;
    }
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

        // Per-image upper-body framing hint. "deskcrop" in the path
        // marks streamer / above-desk shots where the hips and legs
        // are entirely below the camera frame. Without this hint
        // 2D-only providers (ViTPose) hallucinate confident leg
        // keypoints from the desk surface; the hint routes them
        // through `force_shoulder_anchor` so the lower body is
        // dropped from the source skeleton. Without the hint
        // RTMW3D's well-supervised SimCC heads quietly tolerate the
        // hallucination (low confidence), but ViTPose's argmax
        // doesn't — see [[pose-pipeline-iter]] for the A/B that
        // motivated this gate. Apply to every provider so the
        // comparison stays apples-to-apples.
        let path_lower = image_path.to_string_lossy().to_ascii_lowercase();
        let upper_body_hint = if path_lower.contains("deskcrop") {
            Some(vulvatar_lib::tracking::CalibrationMode::UpperBody)
        } else {
            Some(vulvatar_lib::tracking::CalibrationMode::FullBody)
        };
        infer.set_calibration_mode_hint(upper_body_hint);

        let res = process_image(image_path, infer.as_mut(), &asset, renderer.as_mut(), &mut state, debug_renders.as_deref(), idx as u64);
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
    infer: &mut dyn PoseProvider,
    asset: &Arc<AvatarAsset>,
    renderer: Option<&mut VulkanRenderer>,
    state: &mut PoseSolverState,
    debug_renders: Option<&Path>,
    frame_index: u64,
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
    let est_input = infer.estimate_pose(rgb_in.as_raw(), iw, ih, frame_index);
    let sk_input = est_input.skeleton;

    // Step 2: solve avatar pose. After this, the avatar's
    // pose.global_transforms hold each bone's world-space transform —
    // that's the geometric truth of what the solver produced. We
    // compare those directly against the input source skeleton, no
    // RTMW3D round-trip needed.
    let avatar = make_avatar(asset, Some(&sk_input), state);

    // Step 3: compare bone directions (avatar geometric truth vs input
    // skeleton). Both are in Y-up, +Z-toward-camera frames, so unit
    // direction vectors compare without any unit conversion.
    let bones = compare_bones_avatar_vs_source(&sk_input, asset, &avatar);
    let score = mean_angle(&bones);

    // Optional render path for visual spot-checking. **Always emits
    // a side-by-side composite** (input photo | avatar) — never an
    // avatar-only PNG. Solo avatar renders make the triage workflow
    // ambiguous because you can't see which photo the pose belongs
    // to without cross-referencing the filename; the composite makes
    // visual A/B trivial. Score caption goes top-left so worst-case
    // outliers are easy to spot at a glance.
    if let Some(renderer) = renderer {
        if let Ok(rendered_rgba) = render_avatar(renderer, &avatar, RENDER_EXTENT) {
            if let Some(dir) = debug_renders {
                if let Some(stem) = image_path.file_stem() {
                    let out = dir.join(format!("{}.png", stem.to_string_lossy()));
                    let _ =
                        save_composite_render(&out, &img, RENDER_EXTENT, &rendered_rgba, score);
                    // Also dump the raw full-resolution avatar PNG so
                    // finger / thumb detail is visible at the model's
                    // native 1536×1536 — the composite resizes the
                    // avatar half to ~580 px and shrinks fingers below
                    // visual threshold.
                    if std::env::var("VULVATAR_DUMP_RAW_AVATAR").is_ok() {
                        let raw_out = dir.join(format!("{}_raw_avatar.png", stem.to_string_lossy()));
                        if let Some(img_buf) = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
                            RENDER_EXTENT[0], RENDER_EXTENT[1], rendered_rgba.clone(),
                        ) {
                            let _ = img_buf.save(&raw_out);
                        }
                    }
                    let dump = dir.join(format!("{}.bones.txt", stem.to_string_lossy()));
                    // Build a rest-pose-only avatar (no solver) so the
                    // dump can compare avatar rest geometry vs the
                    // solved-and-driven bone positions side by side.
                    let rest_avatar = make_avatar(asset, None, state);
                    let _ = dump_bone_positions(&dump, &sk_input, asset, &avatar, &rest_avatar);
                    if std::env::var("VULVATAR_DUMP_RAW_AVATAR").is_ok() {
                        if let Ok(rest_rgba) = render_avatar(renderer, &rest_avatar, RENDER_EXTENT) {
                            let rest_out = dir.join(format!("{}_rest_avatar.png", stem.to_string_lossy()));
                            if let Some(img_buf) = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
                                RENDER_EXTENT[0], RENDER_EXTENT[1], rest_rgba,
                            ) {
                                let _ = img_buf.save(&rest_out);
                            }
                        }
                    }
                }
            }
        }
    }

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

    // For the "neck" bone the solver's Tip::HeadFromShoulders uses
    // **shoulder midpoint** as the source base, not the UpperChest
    // position. Mirror that here for the avatar side so the score
    // metric measures the same geometric direction the solver
    // optimised toward — without this, a constant 5.4° residual
    // appears across every image from the rest-pose UpperChest →
    // Head offset (rest neck has -0.134 z).
    // Anatomical alignment: source.LeftShoulder (COCO kp 5) is the
    // deltoid area, while VRM HumanoidBone::LeftShoulder is the
    // clavicle base near the sternum. Using clavicle-mid introduces a
    // constant ~2.5° angular bias in the neck score. UpperArm bones
    // sit at the actual shoulder/deltoid area, matching the source's
    // anatomical reference.
    let avatar_shoulder_mid = || -> Option<[f32; 3]> {
        let l = avatar_pos(HumanoidBone::LeftUpperArm)?;
        let r = avatar_pos(HumanoidBone::RightUpperArm)?;
        Some([
            (l[0] + r[0]) * 0.5,
            (l[1] + r[1]) * 0.5,
            (l[2] + r[2]) * 0.5,
        ])
    };
    let source_shoulder_mid = || -> Option<[f32; 3]> {
        let l = src.joints.get(&HumanoidBone::LeftShoulder)?;
        let r = src.joints.get(&HumanoidBone::RightShoulder)?;
        Some([
            (l.position[0] + r.position[0]) * 0.5,
            (l.position[1] + r.position[1]) * 0.5,
            (l.position[2] + r.position[2]) * 0.5,
        ])
    };

    let mut out = Vec::with_capacity(BONE_DIRS.len());
    for (name, base, tip) in BONE_DIRS {
        // Override the "neck" base on both sides to match the solver's
        // Tip::HeadFromShoulders shoulder-midpoint anchor.
        let use_shoulder_mid_base = *name == "neck";
        let src_base = if use_shoulder_mid_base {
            source_shoulder_mid().map(|p| vulvatar_lib::tracking::source_skeleton::SourceJoint {
                position: p,
                confidence: 1.0,
                metric_depth_m: None,
            })
        } else {
            src.joints.get(base).copied()
        };
        let src_tip = src.joints.get(tip).copied();
        let conf_in = src_base.map(|j| j.confidence).unwrap_or(0.0)
            .min(src_tip.map(|j| j.confidence).unwrap_or(0.0));

        let av_base = if use_shoulder_mid_base {
            avatar_shoulder_mid()
        } else {
            avatar_pos(*base)
        };
        let av_tip = avatar_pos(*tip);
        // "conf_out" doesn't really apply here (avatar bones are
        // geometric truth, no detection confidence). Repurpose to
        // signal "bone present in avatar rig" so downstream filters
        // still work; 1.0 = fully present.
        let conf_out = if av_base.is_some() && av_tip.is_some() { 1.0 } else { 0.0 };

        // Skip bones where either source endpoint has unreliable
        // confidence — the source direction is dominated by detector
        // noise (e.g. occluded foot reads c=0.05) and any angular
        // delta we compute would penalise the avatar for tracking
        // garbage rather than tracking the actual pose. 0.30 chosen
        // empirically: visible joints sit at 0.7+, fully occluded
        // joints at 0.05–0.15, partially occluded around 0.3–0.5.
        const SOURCE_CONF_FLOOR: f32 = 0.30;
        // **2D image-plane projection metric**. Source positions come
        // from a 2D inference pipeline (ViTPose-L), so what's actually
        // observed in the photo is the xy projection. The previous
        // 3D-dot-product metric punished the avatar for having a
        // physically-correct z component (e.g. a body-yawed clavicle
        // has +z forward shoulder, but the source had no z signal),
        // turning real 3D pose accuracy into spurious residual. Drop
        // the z axis on both sides so the angle compared is the same
        // one a viewer perceives when looking at the photo + render
        // pair side-by-side.
        let project_xy = |a: [f32; 3], b: [f32; 3]| -> Option<[f32; 3]> {
            let v = [b[0] - a[0], b[1] - a[1], 0.0];
            let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
            if len < 1e-4 { None } else { Some([v[0] / len, v[1] / len, 0.0]) }
        };
        let angle = match (src_base, src_tip, av_base, av_tip) {
            (Some(sb), Some(st), Some(ab), Some(at))
                if sb.confidence >= SOURCE_CONF_FLOOR
                    && st.confidence >= SOURCE_CONF_FLOOR =>
            {
                let ds = project_xy(sb.position, st.position);
                let da = project_xy(ab, at);
                match (ds, da) {
                    (Some(ds), Some(da)) => {
                        let dot = (ds[0] * da[0] + ds[1] * da[1] + ds[2] * da[2])
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
    let v = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-4 {
        None
    } else {
        Some([v[0] / len, v[1] / len, v[2] / len])
    }
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
    // Include finger bones so HMR2+MediaPipe hand integration can be
    // visually verified — `BONE_DIRS` only covers the body skeleton.
    for b in [
        HumanoidBone::LeftHand, HumanoidBone::RightHand,
        HumanoidBone::LeftThumbProximal, HumanoidBone::LeftThumbIntermediate, HumanoidBone::LeftThumbDistal,
        HumanoidBone::LeftIndexProximal, HumanoidBone::LeftIndexIntermediate, HumanoidBone::LeftIndexDistal,
        HumanoidBone::LeftMiddleProximal, HumanoidBone::LeftMiddleIntermediate, HumanoidBone::LeftMiddleDistal,
        HumanoidBone::LeftRingProximal, HumanoidBone::LeftRingIntermediate, HumanoidBone::LeftRingDistal,
        HumanoidBone::LeftLittleProximal, HumanoidBone::LeftLittleIntermediate, HumanoidBone::LeftLittleDistal,
        HumanoidBone::RightThumbProximal, HumanoidBone::RightThumbIntermediate, HumanoidBone::RightThumbDistal,
        HumanoidBone::RightIndexProximal, HumanoidBone::RightIndexIntermediate, HumanoidBone::RightIndexDistal,
        HumanoidBone::RightMiddleProximal, HumanoidBone::RightMiddleIntermediate, HumanoidBone::RightMiddleDistal,
        HumanoidBone::RightRingProximal, HumanoidBone::RightRingIntermediate, HumanoidBone::RightRingDistal,
        HumanoidBone::RightLittleProximal, HumanoidBone::RightLittleIntermediate, HumanoidBone::RightLittleDistal,
    ] {
        if !bones.contains(&b) { bones.push(b); }
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
    // Quaternion of an avatar bone's WORLD rotation, extracted from
    // its Mat4 via the upper 3x3 block.
    let world_quat = |av: &AvatarInstance, b: HumanoidBone| -> Option<[f32; 4]> {
        let idx = humanoid.bone_map.get(&b).copied().map(|n| n.0 as usize)?;
        let m = av.pose.global_transforms.get(idx)?;
        Some(vulvatar_lib::math_utils::mat4_rotation_to_quat(m))
    };

    let mut s = String::new();

    // Rest local frame probe: for each finger bone, dump
    // rest_world_rotation applied to each canonical axis (+X/+Y/+Z).
    // Whichever resulting world direction matches the bone's tail
    // direction (next joint position − this joint position in rest
    // world) IS the avatar's "forward axis" convention for that bone.
    {
        use vulvatar_lib::math_utils::{mat4_rotation_to_quat, quat_rotate_vec3};
        let probe_axes = |b: HumanoidBone, tail: HumanoidBone| -> Option<String> {
            let i = humanoid.bone_map.get(&b).copied().map(|n| n.0 as usize)?;
            let ti = humanoid.bone_map.get(&tail).copied().map(|n| n.0 as usize)?;
            let m = rest.pose.global_transforms.get(i)?;
            let mt = rest.pose.global_transforms.get(ti)?;
            let q = mat4_rotation_to_quat(m);
            let p_head = [m[3][0], m[3][1], m[3][2]];
            let p_tail = [mt[3][0], mt[3][1], mt[3][2]];
            let tail_dir = {
                let d = [p_tail[0] - p_head[0], p_tail[1] - p_head[1], p_tail[2] - p_head[2]];
                let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-9);
                [d[0] / len, d[1] / len, d[2] / len]
            };
            let x = quat_rotate_vec3(&q, &[1.0, 0.0, 0.0]);
            let y = quat_rotate_vec3(&q, &[0.0, 1.0, 0.0]);
            let z = quat_rotate_vec3(&q, &[0.0, 0.0, 1.0]);
            Some(format!(
                "{:?} tail_dir=({:+.2},{:+.2},{:+.2}) +X=({:+.2},{:+.2},{:+.2}) +Y=({:+.2},{:+.2},{:+.2}) +Z=({:+.2},{:+.2},{:+.2})",
                b, tail_dir[0], tail_dir[1], tail_dir[2],
                x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2],
            ))
        };
        s.push_str("## Rest local frame probe (rest_world_rotation applied to canonical axes)\n\n");
        for (b, t) in [
            (HumanoidBone::LeftHand, HumanoidBone::LeftMiddleProximal),
            (HumanoidBone::LeftMiddleProximal, HumanoidBone::LeftMiddleIntermediate),
            (HumanoidBone::LeftIndexProximal, HumanoidBone::LeftIndexIntermediate),
            (HumanoidBone::LeftLittleProximal, HumanoidBone::LeftLittleIntermediate),
            (HumanoidBone::LeftThumbProximal, HumanoidBone::LeftThumbIntermediate),
            (HumanoidBone::LeftThumbIntermediate, HumanoidBone::LeftThumbDistal),
        ] {
            if let Some(line) = probe_axes(b, t) {
                s.push_str(&line);
                s.push('\n');
            }
        }
        s.push('\n');
    }

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
            if ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "webp" {
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
                primitive_data: Some(Arc::clone(prim)),
                morph_weights: Vec::new(),
            }
        })).collect();

    // 1.5 m AvatarSample_A is ~1.5 m tall; with the default 60° FOV
    // this puts the head at about 0.85× frame height with margin for
    // arm-extended poses, giving a snug auto-frame instead of the
    // tiny-figure-in-empty-canvas the previous 3.5 m distance gave.
    let camera = ViewportCamera {
        distance: 1.6,
        pan: [0.0, 0.85],
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

/// Side-by-side composite write: input photo on the left, rendered
/// avatar on the right, score caption in the top-left. The avatar-
/// only PNG path was retired (禁忌): triage requires the photo
/// alongside the avatar to evaluate pose match without filename
/// cross-referencing. Heights are normalised to `COMPOSITE_TARGET_H`
/// so the photo's native aspect (typically 16:9) and the avatar's
/// 1:1 don't stretch each other.
fn save_composite_render(
    path: &Path,
    photo: &DynamicImage,
    avatar_extent: [u32; 2],
    avatar_rgba: &[u8],
    score_deg: Option<f32>,
) -> Result<(), String> {
    const TARGET_HEIGHT: u32 = 768;
    const GUTTER: u32 = 16;

    let avatar_img: ImageBuffer<Rgba<u8>, _> =
        ImageBuffer::from_raw(avatar_extent[0], avatar_extent[1], avatar_rgba.to_vec())
            .ok_or_else(|| "avatar PNG buffer construction failed".to_string())?;
    let avatar_dyn = DynamicImage::ImageRgba8(avatar_img);

    let photo_resized = resize_to_height(photo, TARGET_HEIGHT);
    let avatar_resized = resize_to_height(&avatar_dyn, TARGET_HEIGHT);
    let (pw, ph) = photo_resized.dimensions();
    let (aw, ah) = avatar_resized.dimensions();
    let canvas_w = pw + GUTTER + aw;
    let canvas_h = ph.max(ah);

    let mut canvas = RgbaImage::from_pixel(canvas_w, canvas_h, Rgba([24, 24, 24, 255]));
    image::imageops::overlay(&mut canvas, &photo_resized.to_rgba8(), 0, 0);
    image::imageops::overlay(
        &mut canvas,
        &avatar_resized.to_rgba8(),
        (pw + GUTTER) as i64,
        0,
    );

    let caption = match score_deg {
        Some(s) => format!("score {:.2}", s),
        None => "score N/A".to_string(),
    };
    draw_caption(&mut canvas, &caption);

    canvas
        .save(path)
        .map_err(|e| format!("composite PNG save: {e}"))
}

fn resize_to_height(img: &DynamicImage, target_h: u32) -> DynamicImage {
    let (w, h) = img.dimensions();
    if h == target_h {
        return img.clone();
    }
    let scale = target_h as f32 / h as f32;
    let new_w = (w as f32 * scale).round().max(1.0) as u32;
    img.resize_exact(new_w, target_h, FilterType::Lanczos3)
}

/// 5×7 bitmap font caption rendered at 4× pixel scale over a
/// translucent dark stripe — readable against light or dark image
/// content without per-image background analysis.
fn draw_caption(canvas: &mut RgbaImage, text: &str) {
    let scale = 4u32;
    let pad = 6u32;
    let height = 7 * scale + 2 * pad;
    let width = (text.len() as u32 * (6 * scale)) + 2 * pad;
    for y in 0..height.min(canvas.height()) {
        for x in 0..width.min(canvas.width()) {
            canvas.put_pixel(x, y, Rgba([0, 0, 0, 200]));
        }
    }
    let mut cursor_x = pad;
    for ch in text.chars() {
        if cursor_x + 5 * scale + pad > canvas.width() {
            break;
        }
        if let Some(glyph) = glyph_5x7(ch) {
            for (gy, row) in glyph.iter().enumerate() {
                for gx in 0..5 {
                    let bit = (row >> (4 - gx)) & 0x1;
                    if bit == 0 {
                        continue;
                    }
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let px = cursor_x + gx as u32 * scale + dx;
                            let py = pad + gy as u32 * scale + dy;
                            if px < canvas.width() && py < canvas.height() {
                                canvas.put_pixel(px, py, Rgba([255, 255, 255, 255]));
                            }
                        }
                    }
                }
            }
        }
        cursor_x += 6 * scale;
    }
}

fn glyph_5x7(c: char) -> Option<[u8; 7]> {
    Some(match c {
        ' ' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x06],
        '/' => [0x01, 0x02, 0x04, 0x08, 0x10, 0x00, 0x00],
        '-' => [0x00, 0x00, 0x00, 0x0E, 0x00, 0x00, 0x00],
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x06, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        'N' => [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        's' => [0x00, 0x00, 0x0E, 0x10, 0x0E, 0x01, 0x1E],
        'c' => [0x00, 0x00, 0x0E, 0x10, 0x10, 0x10, 0x0E],
        'o' => [0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E],
        'r' => [0x00, 0x00, 0x16, 0x18, 0x10, 0x10, 0x10],
        'e' => [0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E],
        _ => return None,
    })
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
