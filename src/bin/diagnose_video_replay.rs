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

// D435 colour intrinsics (aligned depth stream), matching diagnose_depth_replay.
const D435_FX: f32 = 924.0;
const D435_FY: f32 = 924.0;
const D435_CX: f32 = 640.0;
const D435_CY: f32 = 360.0;

/// `<stem>_color.png` → `<stem>_depth_mm.npy`.
fn depth_sibling(color: &std::path::Path) -> std::path::PathBuf {
    std::path::PathBuf::from(color.to_string_lossy().replace("_color.png", "_depth_mm.npy"))
}

/// Minimal `.npy` reader for a 2-D little-endian `u16` array → (rows, cols, data).
fn parse_npy_u16(bytes: &[u8]) -> Result<(usize, usize, Vec<u16>), String> {
    if bytes.len() < 12 || &bytes[0..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let major = bytes[6];
    let (header_len, data_start) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
    };
    let data_start = data_start + header_len;
    let shape = std::str::from_utf8(&bytes[10.min(data_start)..data_start])
        .map_err(|e| e.to_string())?
        .split("'shape':").nth(1).and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next()).ok_or("no shape")?;
    let dims: Vec<usize> = shape.split(',').filter_map(|t| t.trim().parse().ok()).collect();
    let (rows, cols) = match dims.as_slice() { [r, c, ..] => (*r, *c), _ => return Err("not 2D".into()) };
    let count = rows * cols;
    let data = &bytes[data_start..];
    if data.len() < count * 2 { return Err("npy short".into()); }
    let out = (0..count).map(|i| u16::from_le_bytes([data[i * 2], data[i * 2 + 1]])).collect();
    Ok((rows, cols, out))
}

/// Deproject an aligned `*_depth_mm.npy` into a `MetricDepthFrame` (camera
/// metres, x-right/y-down/z-forward) — the exact input the D435 provider path
/// consumes via `set_external_depth`.
fn load_metric_depth(
    depth_path: &std::path::Path,
) -> Result<vulvatar_lib::tracking::skeleton_from_depth::MetricDepthFrame, String> {
    use vulvatar_lib::tracking::source_skeleton::CameraIntrinsics;
    let (rows, cols, depth_mm) = parse_npy_u16(&std::fs::read(depth_path).map_err(|e| e.to_string())?)?;
    let (dw, dh) = (cols as u32, rows as u32);
    let intr = CameraIntrinsics { fx: D435_FX, fy: D435_FY, cx: D435_CX, cy: D435_CY, width: dw, height: dh };
    let mut points_m = Vec::with_capacity(depth_mm.len());
    for v in 0..rows {
        for u in 0..cols {
            let z = depth_mm[v * cols + u] as f32 * 0.001;
            points_m.push([(u as f32 - intr.cx) / intr.fx * z, (v as f32 - intr.cy) / intr.fy * z, z]);
        }
    }
    Ok(vulvatar_lib::tracking::skeleton_from_depth::MetricDepthFrame {
        width: dw, height: dh, points_m, crop: None, intrinsics: Some(intr),
        // Offline replay at the recorded cadence — the nominal 30 fps
        // fallback matches the capture rate.
        timestamp_ms: None,
    })
}

// ---- registered-overlay math (avatar joints → camera image, torso-aligned) ----
// The user's acceptance test: "avatar rendering overlaps the 2D keypoints = OK".
// We register the avatar's 3D joints to camera space using ONLY the stable torso
// (both shoulders + nose) as a similarity anchor, then project every joint through
// the D435 intrinsics. This deliberately factors OUT root-translation error so the
// overlay isolates POSE fidelity: if the avatar's wrist/elbow dots land on the
// RTMW3D wrist/elbow keypoints, the lift+retarget faithfully mirrors the subject.
fn v_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn v_mid(a: [f32; 3], b: [f32; 3]) -> [f32; 3] { [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5] }
fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn v_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn v_len(a: [f32; 3]) -> f32 { v_dot(a, a).sqrt() }
fn v_norm(a: [f32; 3]) -> [f32; 3] { let l = v_len(a).max(1e-6); [a[0] / l, a[1] / l, a[2] / l] }

/// Orthonormal torso frame from two shoulders (x-axis) and an up DIRECTION:
/// returns (origin, [x, y, z]). Built identically in camera and avatar space so
/// `R = F_cam · F_avᵀ` follows from expressing a point in the avatar frame and
/// reconstructing it in the camera frame. The up direction is passed in (rather
/// than derived from a point) so the caller can cache the camera's static
/// up-axis and survive frames where the vertical reference (nose) is occluded.
fn torso_frame_dir(lsh: [f32; 3], rsh: [f32; 3], up_dir: [f32; 3]) -> ([f32; 3], [[f32; 3]; 3]) {
    let origin = v_mid(lsh, rsh);
    let x = v_norm(v_sub(rsh, lsh));
    let up = v_norm(up_dir);
    let z = v_norm(v_cross(x, up));
    let y = v_cross(z, x);
    (origin, [x, y, z])
}

/// Camera-space 3D at a normalised keypoint, as the valid-depth median of a
/// window in the deprojected metric grid (`None` if the window is all holes).
fn cam_point_at(pts: &[[f32; 3]], cols: usize, rows: usize, nx: f32, ny: f32, win: i32) -> Option<[f32; 3]> {
    let cx = (nx * cols as f32).round() as i32;
    let cy = (ny * rows as f32).round() as i32;
    let mut vals: Vec<[f32; 3]> = Vec::new();
    for dy in -win..=win {
        for dx in -win..=win {
            let (x, y) = (cx + dx, cy + dy);
            if x < 0 || y < 0 || x >= cols as i32 || y >= rows as i32 { continue; }
            let p = pts[y as usize * cols + x as usize];
            if p[2] > 0.05 { vals.push(p); }
        }
    }
    if vals.is_empty() { return None; }
    vals.sort_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());
    Some(vals[vals.len() / 2])
}

fn project(p: [f32; 3], fx: f32, fy: f32, cx: f32, cy: f32) -> Option<(f32, f32)> {
    if p[2] < 0.05 { return None; }
    Some((fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy))
}

fn draw_dot(img: &mut image::RgbImage, x: i32, y: i32, r: i32, col: [u8; 3]) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let (px, py) = (x + dx, y + dy);
                if px >= 0 && px < w && py >= 0 && py < h {
                    img.put_pixel(px as u32, py as u32, image::Rgb(col));
                }
            }
        }
    }
}

fn draw_line(img: &mut image::RgbImage, a: (i32, i32), b: (i32, i32), col: [u8; 3]) {
    let (mut x0, mut y0) = a;
    let (x1, y1) = b;
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let (w, h) = (img.width() as i32, img.height() as i32);
    loop {
        if x0 >= 0 && x0 < w && y0 >= 0 && y0 < h {
            img.put_pixel(x0 as u32, y0 as u32, image::Rgb(col));
        }
        if x0 == x1 && y0 == y1 { break; }
        let e2 = 2 * err;
        if e2 >= dy { err += dy; x0 += sx; }
        if e2 <= dx { err += dx; y0 += sy; }
    }
}

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

    let config = vulvatar_lib::tracking::provider::TrackingPipelineConfig::default();
    let mut provider = create_pose_provider("models", config)?;
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
    // Bisection toggles for the hands-together cross: set VULVATAR_REPLAY_ARM_REACH_IK=0
    // or VULVATAR_REPLAY_CONTACT_IK=0 to disable that solver stage and see which one
    // stops the avatar arms crossing. Default on (matches the shipping app).
    let env_bool = |k: &str| {
        std::env::var(k)
            .map(|v| !matches!(v.as_str(), "0" | "false" | "off"))
            .unwrap_or(true)
    };
    let params = SolverParams {
        rotation_blend: 1.0,
        lower_body_tracking_enabled: false,
        arm_reach_ik_enabled: env_bool("VULVATAR_REPLAY_ARM_REACH_IK"),
        contact_ik_enabled: env_bool("VULVATAR_REPLAY_CONTACT_IK"),
        ..Default::default()
    };
    eprintln!(
        "solver toggles: arm_reach_ik={} contact_ik={}",
        params.arm_reach_ik_enabled, params.contact_ik_enabled
    );

    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&out_path).map_err(|e| format!("create {out_path}: {e}"))?,
    );

    let sides = [
        ("L", HumanoidBone::LeftUpperArm, HumanoidBone::LeftLowerArm, HumanoidBone::LeftHand),
        ("R", HumanoidBone::RightUpperArm, HumanoidBone::RightLowerArm, HumanoidBone::RightHand),
    ];

    // The camera is static, so the world-up expressed in camera space is a fixed
    // extrinsic. Cache it from any frame where the nose is visible and reuse it on
    // frames where the face is occluded (e.g. a hand covering it).
    let mut last_cam_up: Option<[f32; 3]> = None;

    for (i, f) in files.iter().enumerate() {
        let mut img = image::open(f).map_err(|e| format!("open {}: {e}", f.display()))?.to_rgb8();
        let (w, h) = (img.width(), img.height());
        let rendering = render_every > 0 && renderer.is_some() && i % render_every == 0;
        // Feed the REAL D435 metric path when an aligned depth sibling exists
        // (set_external_depth → estimate_from_external_depth) — the exact
        // pipeline the live app runs — so the rendered avatar reflects the
        // shipping behaviour, not a monocular fallback. Only stamp a synthetic
        // metric frame when there is no depth (a plain colour-only frame dir).
        let depth_path = depth_sibling(f);
        let has_depth = depth_path.exists();
        // On rendered frames, keep the deprojected metric grid so the overlay can
        // back-project torso keypoints into camera-space 3D for the registration.
        let mut overlay_depth: Option<(usize, usize, Vec<[f32; 3]>)> = None;
        if has_depth {
            match load_metric_depth(&depth_path) {
                Ok(metric) => {
                    if rendering {
                        overlay_depth =
                            Some((metric.width as usize, metric.height as usize, metric.points_m.clone()));
                    }
                    provider.set_external_depth(metric);
                }
                Err(e) => eprintln!("depth {i}: {e}"),
            }
        }
        let est = provider.estimate_pose(img.as_raw(), w, h, i as u64);
        let keypoints: Vec<(f32, f32, f32)> = est.annotation.keypoints.clone();
        let mut sk = est.skeleton;
        if !has_depth {
            sk.stamp_synthetic_metric_frame();
        }

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

        let av_pos = |b: HumanoidBone| -> Option<[f32; 3]> {
            let idx = humanoid_map.bone_map.get(&b).copied().map(|n| n.0 as usize)?;
            let m = avatar.pose.global_transforms.get(idx)?;
            Some([m[3][0], m[3][1], m[3][2]])
        };

        // ---- Registered overlay (the user's acceptance test) ----
        // Draw the RTMW3D 2D arm keypoints (GREEN) and the avatar's projected
        // arm joints (MAGENTA) on the camera image, registered on the torso.
        // Overlap ⇒ the avatar faithfully mirrors the subject's pose.
        if rendering {
            if let Some((cols, rows, pts)) = overlay_depth.as_ref() {
                let (cols, rows) = (*cols, *rows);
                let kget = |idx: usize| -> Option<(f32, f32)> {
                    keypoints
                        .get(idx)
                        .and_then(|&(nx, ny, s)| if s > 0.2 { Some((nx, ny)) } else { None })
                };
                let cam_at = |idx: usize| -> Option<[f32; 3]> {
                    let (nx, ny) = kget(idx)?;
                    cam_point_at(pts, cols, rows, nx, ny, 4)
                };
                let kp_px = |idx: usize| -> Option<(i32, i32)> {
                    let (nx, ny) = kget(idx)?;
                    Some(((nx * w as f32).round() as i32, (ny * h as f32).round() as i32))
                };
                const GREEN: [u8; 3] = [0, 230, 0]; // detected 2D keypoints
                const MAGENTA: [u8; 3] = [235, 0, 235]; // avatar projected joints
                let chains = [
                    (5usize, 7usize, 9usize, HumanoidBone::LeftUpperArm, HumanoidBone::LeftLowerArm, HumanoidBone::LeftHand),
                    (6, 8, 10, HumanoidBone::RightUpperArm, HumanoidBone::RightLowerArm, HumanoidBone::RightHand),
                ];

                // Numeric dump of the raw arm keypoints (pixel + score) so the
                // 2D-correct-vs-lift-broken question is answered by data, not eyeball.
                {
                    let names = [(5, "Lsh"), (7, "Lel"), (9, "Lwr"), (6, "Rsh"), (8, "Rel"), (10, "Rwr")];
                    let mut s = format!("f{i:04} 2Dkp:");
                    for (idx, nm) in names {
                        if let Some(&(nx, ny, sc)) = keypoints.get(idx) {
                            s.push_str(&format!(" {nm}=({:.0},{:.0} s={:.2})", nx * w as f32, ny * h as f32, sc));
                        }
                    }
                    eprintln!("{s}");
                }

                // (1) Detected 2D arm skeleton (GREEN) — registration-free, always drawn.
                for (si, ei, wi, _, _, _) in chains {
                    let (ks, ke, kw) = (kp_px(si), kp_px(ei), kp_px(wi));
                    if let (Some(a), Some(b)) = (ks, ke) { draw_line(&mut img, a, b, GREEN); }
                    if let (Some(a), Some(b)) = (ke, kw) { draw_line(&mut img, a, b, GREEN); }
                    for p in [ks, ke, kw].into_iter().flatten() { draw_dot(&mut img, p.0, p.1, 7, GREEN); }
                }

                // (2) Avatar arm skeleton (MAGENTA) — needs the torso similarity.
                // Requires both shoulders in camera-space depth; the vertical axis
                // comes from the nose when visible, else the cached (static-camera)
                // up-axis so a hand over the face doesn't kill the registration.
                if let (Some(cl), Some(cr)) = (cam_at(5), cam_at(6)) {
                    if let Some(cn) = cam_at(0) {
                        last_cam_up = Some(v_norm(v_sub(cn, v_mid(cl, cr))));
                    }
                    let av = (av_pos(HumanoidBone::LeftUpperArm), av_pos(HumanoidBone::RightUpperArm), av_pos(HumanoidBone::Head));
                    if let (Some(cam_up), (Some(al), Some(ar), Some(an))) = (last_cam_up, av) {
                        let av_up = v_sub(an, v_mid(al, ar));
                        let (o_cam, f_cam) = torso_frame_dir(cl, cr, cam_up);
                        let (o_av, f_av) = torso_frame_dir(al, ar, av_up);
                        let scale = v_len(v_sub(cr, cl)) / v_len(v_sub(ar, al)).max(1e-6);
                        let to_cam = |p: [f32; 3]| -> [f32; 3] {
                            let d = v_sub(p, o_av);
                            let l = [v_dot(d, f_av[0]), v_dot(d, f_av[1]), v_dot(d, f_av[2])];
                            [
                                o_cam[0] + scale * (l[0] * f_cam[0][0] + l[1] * f_cam[1][0] + l[2] * f_cam[2][0]),
                                o_cam[1] + scale * (l[0] * f_cam[0][1] + l[1] * f_cam[1][1] + l[2] * f_cam[2][1]),
                                o_cam[2] + scale * (l[0] * f_cam[0][2] + l[1] * f_cam[1][2] + l[2] * f_cam[2][2]),
                            ]
                        };
                        let (sxp, syp) = (w as f32 / cols as f32, h as f32 / rows as f32);
                        let av_px = |b: HumanoidBone| -> Option<(i32, i32)> {
                            let (px, py) = project(to_cam(av_pos(b)?), D435_FX, D435_FY, D435_CX, D435_CY)?;
                            Some(((px * sxp).round() as i32, (py * syp).round() as i32))
                        };
                        for (_, _, _, sb, eb, wb) in chains {
                            let (asx, ae, aw) = (av_px(sb), av_px(eb), av_px(wb));
                            if let (Some(a), Some(b)) = (asx, ae) { draw_line(&mut img, a, b, MAGENTA); }
                            if let (Some(a), Some(b)) = (ae, aw) { draw_line(&mut img, a, b, MAGENTA); }
                            for p in [asx, ae, aw].into_iter().flatten() { draw_dot(&mut img, p.0, p.1, 5, MAGENTA); }
                        }
                    }
                } else {
                    eprintln!("f{i:04} overlay: shoulders missing in depth — no magenta");
                }
            }
        }

        // Avatar reference heights (world y) so hand elevation can be
        // judged WITHOUT reading the render — disambiguates a raised
        // hand from long hanging hair that visually mimics a lowered
        // forearm on some rigs.
        let avy = |b: HumanoidBone| av_pos(b).map(|p| p[1]).unwrap_or(f32::NAN);
        // Hips roll: x-component of the Hips' world up-axis (global transform
        // column 1). ~0 = pelvis grounded/upright; large = the whole body is
        // rotating about the root (the "zero-g spin" artefact).
        let hip_roll = humanoid_map
            .bone_map
            .get(&HumanoidBone::Hips)
            .and_then(|n| avatar.pose.global_transforms.get(n.0 as usize))
            .map(|m| m[1][0])
            .unwrap_or(f32::NAN);
        let mut parts: Vec<String> = vec![format!(
            "\"frame\":{i},\"av_head_y\":{:.3},\"av_hip_y\":{:.3},\"av_sh_y\":{:.3},\"hip_roll\":{:.4}",
            avy(HumanoidBone::Head),
            avy(HumanoidBone::Hips),
            avy(HumanoidBone::LeftUpperArm),
            hip_roll
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
        // Shoulder tilt (roll): source shoulder-height difference vs the avatar's.
        // Source y is image-DOWN (a raised L shoulder => smaller y), avatar y is
        // world-UP, so a faithful mirror makes these ANTI-correlated. If the avatar
        // tilt stays ~flat while the source varies, the shoulder roll is dropped.
        {
            let sp = |b: HumanoidBone| sk.joints.get(&b).map(|j| j.position[1]);
            let src_tilt = match (sp(HumanoidBone::LeftUpperArm), sp(HumanoidBone::RightUpperArm)) {
                (Some(l), Some(r)) => l - r,
                _ => f32::NAN,
            };
            let av_tilt = match (
                av_pos(HumanoidBone::LeftUpperArm),
                av_pos(HumanoidBone::RightUpperArm),
            ) {
                (Some(l), Some(r)) => l[1] - r[1],
                _ => f32::NAN,
            };
            parts.push(format!("\"tilt\":{{\"src\":{:.4},\"av\":{:.4}}}", src_tilt, av_tilt));
        }
        // Full avatar skeleton (world joint positions) so an offline drawer can
        // render the stick figure — GPU-free FK, to iterate torso tilt / elbow.
        {
            let names: [(&str, HumanoidBone); 14] = [
                ("Hips", HumanoidBone::Hips),
                ("Spine", HumanoidBone::Spine),
                ("Chest", HumanoidBone::Chest),
                ("UpperChest", HumanoidBone::UpperChest),
                ("Neck", HumanoidBone::Neck),
                ("Head", HumanoidBone::Head),
                ("LSh", HumanoidBone::LeftShoulder),
                ("RSh", HumanoidBone::RightShoulder),
                ("LUp", HumanoidBone::LeftUpperArm),
                ("RUp", HumanoidBone::RightUpperArm),
                ("LLo", HumanoidBone::LeftLowerArm),
                ("RLo", HumanoidBone::RightLowerArm),
                ("LHa", HumanoidBone::LeftHand),
                ("RHa", HumanoidBone::RightHand),
            ];
            let sk_parts: Vec<String> = names
                .iter()
                .filter_map(|(n, b)| {
                    av_pos(*b).map(|p| format!("\"{}\":[{:.3},{:.3},{:.3}]", n, p[0], p[1], p[2]))
                })
                .collect();
            parts.push(format!("\"skel\":{{{}}}", sk_parts.join(",")));
        }
        parts.push(format!("\"overall\":{:.2}", sk.overall_confidence));
        writeln!(out, "{{{}}}", parts.join(",")).map_err(|e| e.to_string())?;

        // Render + composite AFTER the av_pos closure's last use so the mutable
        // `build_skinning_matrices` borrow doesn't conflict with it. The overlay
        // was already drawn onto `img` above.
        if rendering {
            avatar.build_skinning_matrices();
            let ext = render_extent();
            // Full-resolution camera-with-overlay, un-shrunk, so the GREEN (2D
            // detected) vs MAGENTA (avatar projected) overlap is actually legible.
            let _ = img.save(render_dir.join(format!("f{i:04}_overlay.png")));
            if let Some(r) = renderer.as_mut() {
                if let Ok(rgba) = render_avatar(r, &avatar, ext) {
                    let _ = save_composite(&render_dir.join(format!("f{i:04}.png")), &img, &rgba, ext);
                }
            }
        }
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
