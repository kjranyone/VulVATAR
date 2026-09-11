//! Offline replay of the tracking-v2 fusion pipeline over a recorded
//! `frame_NNNN_color.png + frame_NNNN_depth_mm.npy` directory — no camera.
//!
//!   cargo run --features realsense --bin diagnose_fusion_replay -- <dir> [out_dir] [--render N]
//!
//! Emits a per-frame CSV (solve time, cost, observation counts, torso yaw,
//! head yaw/pitch/roll in the viewer frame, wrist camera positions, per-joint
//! σ) to `<out_dir>/frames.csv`, a temporal summary to stdout, and — with
//! `--render N` — an overlay PNG every N frames (colour image + projected
//! model skeleton + capsules, colour-coded by σ) into `<out_dir>/`.
//! `out_dir` defaults to `diagnostics/fusion/<dir name>/`.

use std::path::{Path, PathBuf};

use vulvatar_lib::tracking::fusion::estimator::Intrinsics;
use vulvatar_lib::tracking::fusion::math::*;
use vulvatar_lib::tracking::fusion::model::*;
use vulvatar_lib::tracking::fusion::provider::FusionProvider;
use vulvatar_lib::tracking::provider::{PoseProvider, TrackingPipelineConfig};
use vulvatar_lib::tracking::metric_frame::MetricDepthFrame;
use vulvatar_lib::tracking::CameraIntrinsics;

use vulvatar_lib::asset::Transform;
use vulvatar_lib::avatar::retarget::{apply_rig_pose, RetargetParams, RetargetState};
use vulvatar_lib::renderer::offline;
use vulvatar_lib::renderer::VulkanRenderer;

const D435_FX: f32 = 924.0;
const D435_FY: f32 = 924.0;
const D435_CX: f32 = 640.0;
const D435_CY: f32 = 360.0;

fn parse_npy_u16(bytes: &[u8]) -> Result<(usize, usize, Vec<u16>), String> {
    if bytes.len() < 12 || &bytes[0..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let major = bytes[6];
    let (header_len, data_start) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
    let data_start = data_start + header_len;
    let header = std::str::from_utf8(&bytes[10.min(data_start)..data_start])
        .map_err(|e| format!("npy header utf8: {e}"))?;
    if !header.contains("<u2") && !header.contains("|u2") {
        return Err(format!("expected little-endian u16 (<u2), header: {header}"));
    }
    let shape = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next())
        .ok_or("no shape in npy header")?;
    let dims: Vec<usize> = shape
        .split(',')
        .filter_map(|t| t.trim().parse::<usize>().ok())
        .collect();
    let (rows, cols) = match dims.as_slice() {
        [r, c, ..] => (*r, *c),
        _ => return Err(format!("expected 2-D shape, got {dims:?}")),
    };
    let count = rows * cols;
    let data = &bytes[data_start..];
    if data.len() < count * 2 {
        return Err(format!("npy data short: {} < {}", data.len(), count * 2));
    }
    let out = data[..count * 2]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    Ok((rows, cols, out))
}

fn load_metric_frame(
    color_path: &Path,
    depth_path: &Path,
) -> Result<(image::RgbImage, MetricDepthFrame), String> {
    let img = image::open(color_path).map_err(|e| format!("open colour: {e}"))?;
    let rgb = img.to_rgb8();
    let depth_bytes = std::fs::read(depth_path).map_err(|e| format!("read depth: {e}"))?;
    let (rows, cols, depth_mm) = parse_npy_u16(&depth_bytes)?;
    let (dw, dh) = (cols as u32, rows as u32);
    // Nominal D435 colour intrinsics per stream profile (the 4:3 profiles
    // crop the sensor, so they are not a pure scale of the 16:9 ones).
    let (fx, fy, cx, cy) = match (dw, dh) {
        (640, 480) => (616.0, 616.0, 320.0, 240.0),
        (848, 480) => (612.0, 612.0, 424.0, 240.0),
        _ => {
            let sx = dw as f32 / 1280.0;
            (D435_FX * sx, D435_FY * sx, D435_CX * sx, D435_CY * sx)
        }
    };
    let intr = CameraIntrinsics {
        fx,
        fy,
        cx,
        cy,
        width: dw,
        height: dh,
    };
    let mut points_m = Vec::with_capacity(depth_mm.len());
    for v in 0..rows {
        for u in 0..cols {
            let z = depth_mm[v * cols + u] as f32 * 0.001;
            if z > 0.0 {
                points_m.push([
                    (u as f32 - intr.cx) / intr.fx * z,
                    (v as f32 - intr.cy) / intr.fy * z,
                    z,
                ]);
            } else {
                points_m.push([f32::NAN, f32::NAN, f32::NAN]);
            }
        }
    }
    Ok((
        rgb,
        MetricDepthFrame {
            width: dw,
            height: dh,
            points_m,
            intrinsics: Some(intr),
            timestamp_ms: None,
        },
    ))
}

/// Median depth (m) of the valid pixels in a 3×3 window; NaN if none.
fn depth_median_3x3(pts: &[[f32; 3]], w: u32, h: u32, u: i64, v: i64) -> f32 {
    let mut zs: Vec<f32> = Vec::with_capacity(9);
    for dv in -1..=1 {
        for du in -1..=1 {
            let (x, y) = (u + du, v + dv);
            if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
                continue;
            }
            let z = pts[(y as u32 * w + x as u32) as usize][2];
            if z.is_finite() && z > 0.0 {
                zs.push(z);
            }
        }
    }
    if zs.is_empty() {
        return f32::NAN;
    }
    zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    zs[zs.len() / 2]
}

fn draw_line(img: &mut image::RgbImage, a: [f64; 2], b: [f64; 2], c: [u8; 3]) {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let (x0, y0, x1, y1) = (a[0] as i64, a[1] as i64, b[0] as i64, b[1] as i64);
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let (mut x, mut y) = (x0, y0);
    let mut guard = 0;
    loop {
        if x >= 0 && y >= 0 && x < w && y < h {
            for ox in -1..=1 {
                for oy in -1..=1 {
                    let (px, py) = (x + ox, y + oy);
                    if px >= 0 && py >= 0 && px < w && py < h {
                        img.put_pixel(px as u32, py as u32, image::Rgb(c));
                    }
                }
            }
        }
        if x == x1 && y == y1 {
            break;
        }
        guard += 1;
        if guard > 10000 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

fn draw_dot(img: &mut image::RgbImage, p: [f64; 2], r: i64, c: [u8; 3]) {
    let (w, h) = (img.width() as i64, img.height() as i64);
    for oy in -r..=r {
        for ox in -r..=r {
            if ox * ox + oy * oy > r * r {
                continue;
            }
            let (px, py) = (p[0] as i64 + ox, p[1] as i64 + oy);
            if px >= 0 && py >= 0 && px < w && py < h {
                img.put_pixel(px as u32, py as u32, image::Rgb(c));
            }
        }
    }
}

fn sigma_color(s: f64) -> [u8; 3] {
    // green (certain) → yellow → red (uncertain)
    let t = (s / 0.6).clamp(0.0, 1.0);
    let r = (255.0 * t.min(1.0)) as u8;
    let g = (255.0 * (1.0 - (t - 0.5).max(0.0) * 2.0)) as u8;
    [r, g, 40]
}

/// Yaw/pitch/roll (deg) of a viewer-frame rotation: yaw about +Y, pitch
/// about +X, roll about +Z (ZXY-ish decomposition good enough for a report).
fn ypr_deg(r: &M3) -> (f64, f64, f64) {
    // forward = R·(0,0,1), up = R·(0,1,0)
    let f = col(r, 2);
    let u = col(r, 1);
    let yaw = f[0].atan2(f[2]).to_degrees();
    let pitch = (-f[1]).asin().to_degrees();
    // roll: angle of up vector projected in the plane ⟂ forward
    let right = cross(u, f);
    let roll = right[1].atan2(u[1].abs().max(1e-6)).to_degrees();
    (yaw, pitch, roll)
}

fn stats(v: &[f64]) -> (f64, f64, f64, f64) {
    if v.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (mean, var.sqrt(), s[0], s[s.len() - 1])
}

fn main() -> Result<(), String> {
    env_logger::init();
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut render_every: usize = 0;
    if let Some(i) = args.iter().position(|a| a == "--render") {
        render_every = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(5);
        args.drain(i..(i + 2).min(args.len()));
    }
    // --avatar [path]: render the recovered rig pose on a VRM every
    // `--render N` frames and write camera|avatar composites. The path
    // defaults to the sample rig (or VULVATAR_VRM).
    let mut avatar_vrm: Option<String> = None;
    if let Some(i) = args.iter().position(|a| a == "--avatar") {
        let next = args.get(i + 1).cloned();
        if let Some(n) = next.as_ref().filter(|n| !n.starts_with("--")) {
            avatar_vrm = Some(n.clone());
            args.drain(i..(i + 2).min(args.len()));
        } else {
            avatar_vrm = Some(
                std::env::var("VULVATAR_VRM")
                    .unwrap_or_else(|_| "sample_data/AliciaSolid.vrm".to_string()),
            );
            args.drain(i..(i + 1));
        }
        if render_every == 0 {
            render_every = 5;
        }
    }
    let dir = PathBuf::from(args.first().ok_or("usage: diagnose_fusion_replay <dir> [out_dir] [--render N]")?);
    let out_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("diagnostics/fusion").join(dir.file_name().unwrap_or_default())
        });
    if out_dir.starts_with("validation_images") {
        return Err("refusing to write under validation_images/".into());
    }
    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;

    let mut pairs: Vec<(u64, PathBuf, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(&dir).map_err(|e| format!("read_dir: {e}"))? {
        let p = entry.map_err(|e| e.to_string())?.path();
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
        if let Some(stem) = name.strip_suffix("_color.png") {
            let depth = p.with_file_name(format!("{stem}_depth_mm.npy"));
            if depth.exists() {
                let idx = stem
                    .trim_start_matches(|c: char| !c.is_ascii_digit())
                    .rsplit('_')
                    .next()
                    .and_then(|s| s.trim_start_matches('f').parse::<u64>().ok())
                    .unwrap_or(0);
                pairs.push((idx, p.clone(), depth));
            }
        }
    }
    pairs.sort_by_key(|(i, _, _)| *i);
    if pairs.is_empty() {
        return Err(format!("no *_color.png + *_depth_mm.npy pairs in {}", dir.display()));
    }

    let mut cfg = TrackingPipelineConfig::default();
    if std::env::var_os("VULVATAR_REPLAY_NO_YOLOX").is_some() {
        cfg.yolox_enabled = false;
    }
    if std::env::var_os("VULVATAR_REPLAY_CPU").is_some() {
        cfg.force_cpu = true;
    }
    let mut provider = FusionProvider::from_models_dir_with_config("models", cfg)?;
    // Optional avatar rendering rig.
    let mut avatar_rig = if let Some(vrm) = avatar_vrm.as_ref() {
        eprintln!("loading VRM {vrm} + Vulkan renderer for composites…");
        let asset = vulvatar_lib::asset::vrm::VrmAssetLoader::new()
            .load(vrm)
            .map_err(|e| format!("load VRM: {e:?}"))?;
        let mut renderer = VulkanRenderer::new();
        renderer.initialize();
        let rest_locals: Vec<Transform> =
            asset.skeleton.nodes.iter().map(|n| n.rest_local.clone()).collect();
        Some((asset, renderer, rest_locals, RetargetState::default()))
    } else {
        None
    };
    eprintln!("provider: {} — {} frames → {}", provider.label(), pairs.len(), out_dir.display());

    // VULVATAR_REPLAY_VISDUMP=1: per-frame × per-keypoint dump of the
    // detector's raw output (SimCC peak stats), the score after each
    // gate of the provider, the crop and the raw depth under the pixel —
    // the input for the visibility calibration (`diagnostics/visibility`).
    let vis_dump = std::env::var_os("VULVATAR_REPLAY_VISDUMP").is_some();
    let mut vis_csv = String::new();
    if vis_dump {
        vis_csv.push_str("frame,j,nx,ny,nz,score,sx,sy,second_x,second_y,half_x,half_y,zscore,crop_x,crop_y,crop_w,crop_h,depth_z,p_vis,sil_dist,sil_zref,sil_bottom,sil_area");
    }
    let mut vis_header_done = false;

    let mut csv = String::new();
    csv.push_str("idx,t,solve_ms,cost0,cost1,iters,n2d,n3d,ncloud,quality,root_x,root_y,root_z,root_sig,torso_yaw,torso_pitch,torso_roll,head_yaw,head_pitch,head_roll,Lw_x,Lw_y,Lw_z,Rw_x,Rw_y,Rw_z,sig_spine,sig_neck,sig_head,sig_Lsh,sig_Lel,sig_Lwr,sig_Rsh,sig_Rel,sig_Rwr,scale,face68,mesh,len0,len1,len2,len3,len4,len5,len6,len7,rad0,rad1,rad2,cost_2d,cost_3d,cost_cloud,cost_prior,cost_temporal,med2d_px,mean3d_m,meancloud_m,dsig_Lhip,dsig_Lknee,dsig_Lankle,dsig_Rhip,dsig_Rknee,dsig_Rankle,Lk_x,Lk_y,Lk_z,Rk_x,Rk_y,Rk_z,La_x,La_y,La_z,Ra_x,Ra_y,Ra_z\n");

    let mut torso_yaws = Vec::new();
    let mut yaw_ref_pairs: Vec<(f64, f64)> = Vec::new();
    let mut kp3d_err: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut head_yaws = Vec::new();
    let mut head_pitches = Vec::new();
    let mut solve_ms = Vec::new();
    let mut est_ms = Vec::new();
    let mut hand_frames = [0usize; 2];
    let mut lw_prev: Option<V3> = None;
    let mut rw_prev: Option<V3> = None;
    let mut lw_jumps = Vec::new();
    let mut rw_jumps = Vec::new();
    let mut lk_prev: Option<V3> = None;
    let mut rk_prev: Option<V3> = None;
    let mut lk_jumps = Vec::new();
    let mut rk_jumps = Vec::new();
    let mut leg_sig: Vec<[f64; 6]> = Vec::new();
    let mut root_prev: Option<V3> = None;
    let mut root_jumps = Vec::new();
    let mut prev_seed = 0u64;
    let mut prev_lost = 0u64;
    let mut lwr_sig = Vec::new();
    let mut rwr_sig = Vec::new();

    for (n, (idx, cp, dp)) in pairs.iter().enumerate() {
        let (rgb, mut metric) = load_metric_frame(cp, dp)?;
        let (cw, ch) = (rgb.width(), rgb.height());
        metric.timestamp_ms = Some(*idx as f64 * (1000.0 / 30.0));
        let intr_cam = metric.intrinsics.unwrap();
        let depth_pts = metric.points_m.clone();
        provider.set_external_depth(metric);
        let est_out = provider.estimate_pose(rgb.as_raw(), cw, ch, n as u64);
        let rig = est_out.skeleton.rig.clone();
        if std::env::var_os("VULVATAR_REPLAY_KPDUMP").is_some() {
            let k = &est_out.annotation.keypoints;
            eprint!("frame {idx} kps:");
            for i in [0usize, 5, 6, 7, 8, 9, 10, 11, 12] {
                if let Some((nx, ny, sc)) = k.get(i) {
                    eprint!(" {i}:({:.0},{:.0},{:.2})", nx * cw as f32, ny * ch as f32, sc);
                }
            }
            eprintln!();
        }
        if vis_dump {
            if !vis_header_done {
                for (name, _) in &provider.last_kp_stages {
                    vis_csv.push_str(&format!(",s_{name}"));
                }
                vis_csv.push('\n');
                vis_header_done = true;
            }
            let (cx, cy, cw_, ch_) = provider.last_crop.unwrap_or((f32::NAN, f32::NAN, f32::NAN, f32::NAN));
            for (j, kp) in provider.last_raw_joints.iter().enumerate() {
                let u = (kp.nx * cw as f32).round() as i64;
                let v = (kp.ny * ch as f32).round() as i64;
                let dz = depth_median_3x3(&depth_pts, cw, ch, u, v);
                let (pv, sd) = provider.last_vis.get(j).copied().unwrap_or((f32::NAN, f32::NAN));
                let (sz, sb, sa) = provider
                    .last_silhouette
                    .map(|(z, b, a)| (z, b as u8, a))
                    .unwrap_or((f64::NAN, 0, f64::NAN));
                vis_csv.push_str(&format!(
                    "{idx},{j},{:.5},{:.5},{:.5},{:.4},{:.5},{:.5},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1},{:.1},{:.1},{:.1},{:.4},{:.4},{:.4},{:.3},{},{:.4}",
                    kp.nx, kp.ny, kp.nz, kp.score, kp.sx, kp.sy, kp.second_x, kp.second_y,
                    kp.half_x, kp.half_y, kp.zscore, cx, cy, cw_, ch_, dz, pv, sd, sz, sb, sa
                ));
                for (_, scores) in &provider.last_kp_stages {
                    vis_csv.push_str(&format!(",{:.4}", scores.get(j).copied().unwrap_or(f32::NAN)));
                }
                vis_csv.push('\n');
            }
        }
        let h = provider.humanoid();
        let est = provider.estimator();
        let m = &h.model;
        let fk = m.fk(&est.state);
        let cam_to_view = FACING_CAMERA;
        let (ty, tp, tr) = ypr_deg(&mat_mul(&cam_to_view, &fk.r[h.j.spine3]));
        let (hy, hp, hr) = ypr_deg(&mat_mul(&cam_to_view, &fk.r[h.j.head]));
        let sig = |j: usize| est.joint_world_sigma(j);
        let lw = fk.t[h.j.l_wrist];
        let rw = fk.t[h.j.r_wrist];
        let (lk, rk, la, ra) = (
            fk.t[h.j.l_knee],
            fk.t[h.j.r_knee],
            fk.t[h.j.l_ankle],
            fk.t[h.j.r_ankle],
        );
        let d = est.diag;
        let q = rig.as_ref().map(|r| r.quality).unwrap_or(0.0);
        csv.push_str(&format!(
            "{idx},{:.3},{:.2},{:.1},{:.1},{},{},{},{},{:.2},{:.3},{:.3},{:.3},{:.3},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{}\n",
            *idx as f64 / 30.0,
            provider.last_solve_ms,
            d.cost_initial,
            d.cost_final,
            d.iters,
            d.n_kp2d,
            d.n_kp3d,
            d.n_cloud,
            q,
            est.state.root_t[0],
            est.state.root_t[1],
            est.state.root_t[2],
            est.root_sigma_m(),
            ty, tp, tr, hy, hp, hr,
            lw[0], lw[1], lw[2], rw[0], rw[1], rw[2],
            sig(h.j.spine2), sig(h.j.neck), sig(h.j.head),
            sig(h.j.l_shoulder), sig(h.j.l_elbow), sig(h.j.l_wrist),
            sig(h.j.r_shoulder), sig(h.j.r_elbow), sig(h.j.r_wrist),
            est.state.scale,
            0, provider.mesh_learned(),
        ));
        {
            let l = &est.state.len;
            let r = &est.state.rad;
            csv.pop();
            csv.push_str(&format!(
                ",{l0:.3},{l1:.3},{l2:.3},{l3:.3},{l4:.3},{l5:.3},{l6:.3},{l7:.3},{r0:.3},{r1:.3},{r2:.3},\
{c2d:.1},{c3d:.1},{ccl:.1},{cpr:.1},{cte:.1},{m2d:.2},{m3d:.4},{mcl:.4},\
{ds0:.3},{ds1:.3},{ds2:.3},{ds3:.3},{ds4:.3},{ds5:.3},\
{lkx:.3},{lky:.3},{lkz:.3},{rkx:.3},{rky:.3},{rkz:.3},\
{lax:.3},{lay:.3},{laz:.3},{rax:.3},{ray:.3},{raz:.3}\n",
                l0 = l[0], l1 = l[1], l2 = l[2], l3 = l[3], l4 = l[4], l5 = l[5], l6 = l[6], l7 = l[7],
                r0 = r[0], r1 = r[1], r2 = r[2],
                c2d = d.cost_2d, c3d = d.cost_3d, ccl = d.cost_cloud, cpr = d.cost_prior, cte = d.cost_temporal,
                m2d = d.med_2d_px, m3d = d.mean_3d_m, mcl = d.mean_cloud_m,
                ds0 = est.joint_data_sigma(m, h.j.l_hip), ds1 = est.joint_data_sigma(m, h.j.l_knee), ds2 = est.joint_data_sigma(m, h.j.l_ankle),
                ds3 = est.joint_data_sigma(m, h.j.r_hip), ds4 = est.joint_data_sigma(m, h.j.r_knee), ds5 = est.joint_data_sigma(m, h.j.r_ankle),
                lkx = lk[0], lky = lk[1], lkz = lk[2], rkx = rk[0], rky = rk[1], rkz = rk[2],
                lax = la[0], lay = la[1], laz = la[2], rax = ra[0], ray = ra[1], raz = ra[2],
            ));
        }
        if std::env::var_os("VULVATAR_REPLAY_SHDUMP").is_some() {
            let lsh = fk.t[h.j.l_shoulder];
            let rsh = fk.t[h.j.r_shoulder];
            let near = |p: V3| -> Option<f32> {
                provider.last_surface.iter().map(|q| ([q[0] as f64, q[1] as f64, q[2] as f64], q[2]))
                    .filter(|(q, _)| ((q[0]-p[0]).powi(2)+(q[1]-p[1]).powi(2)).sqrt() < 0.12)
                    .map(|(_, z)| z).next()
            };
            eprintln!("idx {idx} yaw {ty:+.1} Lsh z {:.3} (surf {:?}) Rsh z {:.3} (surf {:?}) sig L/R {:.2}/{:.2} scale {:.3} len_sh {:.3}",
                lsh[2], near(lsh), rsh[2], near(rsh), sig(h.j.l_shoulder), sig(h.j.r_shoulder), est.state.scale, est.state.len[0]);
        }
        // Reference torso yaw straight from the detector shoulders + depth
        // (independent of the estimator): atan2(Δz, Δx) of the two shoulder
        // surface points, when both have valid depth and decent score.
        // Reference torso yaw, independent of the estimator: the depth
        // slope across the CHEST. Two-point shoulder sampling was not
        // robust — a forearm or clasped hands in front of a shoulder put
        // that patch 25 cm nearer and swung the reference by 60°+ — so
        // sample a strip between the shoulders, reduce each column to its
        // median depth, reject columns that sit clearly in front of the
        // chest plane (that is an arm), and fit z(x) over what is left.
        let yaw_ref = {
            let k = &est_out.annotation.keypoints;
            let px = |i: usize| -> Option<(f64, f64)> {
                let (nx, ny, sc) = *k.get(i)?;
                if sc < 0.5 {
                    return None;
                }
                Some((nx as f64 * cw as f64, ny as f64 * ch as f64))
            };
            match (px(5), px(6)) {
                (Some(lp), Some(rp)) => {
                    let span = ((lp.0 - rp.0).powi(2) + (lp.1 - rp.1).powi(2)).sqrt();
                    let cols = 15usize;
                    let mut med: Vec<Option<(f64, f64)>> = Vec::with_capacity(cols);
                    for c in 0..cols {
                        // Inset 12% at each end so the strip stays on the
                        // torso rather than straddling the silhouette.
                        let t = 0.12 + 0.76 * (c as f64 + 0.5) / cols as f64;
                        let bu = rp.0 + (lp.0 - rp.0) * t;
                        let bv = rp.1 + (lp.1 - rp.1) * t;
                        let (mut xs, mut zs): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
                        for row in 0..7 {
                            let v = bv + span * (0.02 + 0.16 * row as f64 / 6.0);
                            if let Some(q) = vulvatar_lib::tracking::fusion::observe::window_point(
                                &depth_pts, cw, ch, bu, v, 1, 0.2, 3.0,
                            ) {
                                xs.push(q[0]);
                                zs.push(q[2]);
                            }
                        }
                        if zs.len() < 4 {
                            med.push(None);
                            continue;
                        }
                        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        med.push(Some((xs[xs.len() / 2], zs[zs.len() / 2])));
                    }
                    let mut all: Vec<f64> = med.iter().flatten().map(|c| c.1).collect();
                    if all.len() < 8 {
                        None
                    } else {
                        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let mid = all[all.len() / 2];
                        let pts: Vec<(f64, f64)> = med
                            .iter()
                            .flatten()
                            .copied()
                            // In front of the chest plane by > 6 cm = arm
                            // or hand; behind by > 12 cm = background.
                            .filter(|(_, z)| *z > mid - 0.06 && *z < mid + 0.12)
                            .collect();
                        let n = pts.len() as f64;
                        let xspan = pts
                            .iter()
                            .map(|p| p.0)
                            .fold(f64::NEG_INFINITY, f64::max)
                            - pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
                        if pts.len() < 8 || xspan < 0.12 {
                            None
                        } else {
                            let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
                            let mz = pts.iter().map(|p| p.1).sum::<f64>() / n;
                            let (mut num, mut den) = (0.0, 0.0);
                            for (x, z) in &pts {
                                num += (x - mx) * (z - mz);
                                den += (x - mx) * (x - mx);
                            }
                            // θ = atan2(Δz_cam, Δx_cam): body yaw about
                            // viewer-up takes the shoulder line to
                            // (cos θ, −sin θ) in viewer x/z, z_v = −z_cam.
                            Some((num / den.max(1e-9)).atan().to_degrees())
                        }
                    }
                }
                _ => None,
            }
        };
        if std::env::var_os("VULVATAR_REPLAY_SHREF").is_some() {
            // Independent torso-yaw evidence: depth of each detector
            // shoulder (median of a patch pulled 15% toward the torso
            // centre so the window cannot straddle the silhouette), the
            // 2-D shoulder span (|θ| ≈ acos(span / span_frontal)) and the
            // estimator's own shoulders.
            let k = &est_out.annotation.keypoints;
            let sh_px = |i: usize| -> Option<(f64, f64)> {
                let (nx, ny, sc) = *k.get(i)?;
                if sc < 0.5 { return None; }
                Some((nx as f64 * cw as f64, ny as f64 * ch as f64))
            };
            if let (Some(lp), Some(rp)) = (sh_px(5), sh_px(6)) {
                let mid = (0.5 * (lp.0 + rp.0), 0.5 * (lp.1 + rp.1));
                let inset = |p: (f64, f64)| (p.0 + 0.15 * (mid.0 - p.0), p.1 + 0.15 * (mid.1 - p.1));
                let med_z = |p: (f64, f64)| -> Option<f64> {
                    let mut zs: Vec<f64> = Vec::new();
                    for du in -4i32..=4 {
                        for dv in -4i32..=4 {
                            let (u, v) = (p.0 + du as f64 * 2.0, p.1 + dv as f64 * 2.0);
                            if let Some(q) = vulvatar_lib::tracking::fusion::observe::window_point(
                                &depth_pts, cw, ch, u, v, 1, 0.2, 3.0) { zs.push(q[2]); }
                        }
                    }
                    if zs.len() < 12 { return None; }
                    zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    Some(zs[zs.len() / 2])
                };
                let span = ((lp.0 - rp.0).powi(2) + (lp.1 - rp.1).powi(2)).sqrt();
                let lz = med_z(inset(lp));
                let rz = med_z(inset(rp));
                let lsh = fk.t[h.j.l_shoulder];
                let rsh = fk.t[h.j.r_shoulder];
                eprintln!(
                    "SHREF idx {idx} est {ty:+.1} ref {:?} span_px {span:.0} lz {:?} rz {:?} est_lz {:.3} est_rz {:.3}",
                    yaw_ref.map(|v| (v * 10.0).round() / 10.0),
                    lz.map(|v| (v * 1000.0).round() / 1000.0),
                    rz.map(|v| (v * 1000.0).round() / 1000.0),
                    lsh[2], rsh[2]
                );
            }
        }
        if let Some(yr) = yaw_ref {
            yaw_ref_pairs.push((ty, yr));
        }
        if std::env::var_os("VULVATAR_REPLAY_VARDUMP").is_some() && n % 10 == 5 {
            let pj = m.joint_param[h.j.head];
            let pn = m.joint_param[h.j.neck];
            eprintln!("idx {idx} var head {:?} neck {:?} data_info head {:?} root_t var {:?}",
                &est.var[pj..pj + 3], &est.var[pn..pn + 3], &est.data_info_ema[pj..pj + 3], &est.var[3..6]);
        }
        // Metric-joint residuals for the key joints (model vs depth-lifted obs).
        for &(j, pobs, _) in &provider.last_kp3d {
            let e = norm(sub(fk.t[j], pobs));
            let slot = if j == h.j.l_shoulder || j == h.j.r_shoulder { 0 } else if j == h.j.l_elbow || j == h.j.r_elbow { 1 } else if j == h.j.l_wrist || j == h.j.r_wrist { 2 } else { 3 };
            kp3d_err[slot].push(e);
        }
        if std::env::var_os("VULVATAR_REPLAY_POSTURE").is_some() {
            // Whole-body posture split: root tilt (deg from upright in the
            // viewer frame) and each spine joint's rotation magnitude.
            let dv = mat_mul(&FACING_CAMERA, &est.state.root_r);
            let up = col(&dv, 1);
            let root_tilt = up[1].clamp(-1.0, 1.0).acos().to_degrees();
            let jr = |j: usize| {
                let w = est.state.joint_rotvec(m, j);
                (norm(w).to_degrees()) as i32
            };
            eprintln!(
                "POSTURE idx {idx} root_tilt {root_tilt:.0}° spine {}/{}/{}° neck {}° head {}° hipsL {}° kneeL {}°",
                jr(h.j.spine1), jr(h.j.spine2), jr(h.j.spine3), jr(h.j.neck), jr(h.j.head), jr(h.j.l_hip), jr(h.j.l_knee)
            );
        }
        if std::env::var_os("VULVATAR_REPLAY_HEADREF").is_some() {
            // Estimator head yaw vs the FaceMesh-selected face channel
            // (dense 478-landmark pose, trustworthy to ~±45°).
            if let (Some(f), Some(c)) = (
                est_out.skeleton.face,
                est_out.skeleton.face_mesh_confidence,
            ) {
                if c > 0.6 {
                    eprintln!("HEADREF idx {idx} est_yaw {hy:.1} est_pitch {hp:.1} sel_yaw {:.1} sel_pitch {:.1} mesh_c {c:.2}", f.yaw.to_degrees(), f.pitch.to_degrees());
                }
            }
        }
        torso_yaws.push(ty);
        head_yaws.push(hy);
        head_pitches.push(hp);
        solve_ms.push(provider.last_solve_ms as f64);
        est_ms.push(provider.last_est_ms as f64);
        for hand in 0..2 {
            if let Some(hr) = provider.last_hands[hand].as_ref() {
                hand_frames[hand] += 1;
                if std::env::var_os("VULVATAR_REPLAY_HANDDUMP").is_some() && n % 8 == 0 {
                    let w = &hr.world;
                    let tip = w[8];
                    let mcp = w[5];
                    eprintln!("idx {idx} hand {hand}: presence {:.2} handed {:.2} crop {:?} wrist px ({:.0},{:.0}) index_mcp world ({:+.3},{:+.3},{:+.3}) index_tip ({:+.3},{:+.3},{:+.3}) |mcp| {:.3}",
                        hr.presence, hr.handedness, hr.crop, hr.px[0][0], hr.px[0][1], mcp[0], mcp[1], mcp[2], tip[0], tip[1], tip[2],
                        (mcp[0]*mcp[0]+mcp[1]*mcp[1]+mcp[2]*mcp[2]).sqrt());
                }
            }
        }
        if let Some(p) = lw_prev {
            lw_jumps.push(norm(sub(lw, p)));
        }
        if let Some(p) = rw_prev {
            rw_jumps.push(norm(sub(rw, p)));
        }
        if let Some(p) = lk_prev.replace(lk) {
            lk_jumps.push(norm(sub(lk, p)));
        }
        if let Some(p) = rk_prev.replace(rk) {
            rk_jumps.push(norm(sub(rk, p)));
        }
        leg_sig.push([
            est.joint_data_sigma(m, h.j.l_hip),
            est.joint_data_sigma(m, h.j.l_knee),
            est.joint_data_sigma(m, h.j.l_ankle),
            est.joint_data_sigma(m, h.j.r_hip),
            est.joint_data_sigma(m, h.j.r_knee),
            est.joint_data_sigma(m, h.j.r_ankle),
        ]);
        // Event log for large wrist jumps: what state produced them.
        {
            let seed_now = est.diag.seed_wins;
            let lost_now = est.lost_events;
            for (name, cur, prev, sig_d) in [
                ("L", lw, lw_prev, est.joint_data_sigma(m, h.j.l_wrist)),
                ("R", rw, rw_prev, est.joint_data_sigma(m, h.j.r_wrist)),
            ] {
                if let Some(p) = prev {
                    let jump = norm(sub(cur, p));
                    if jump > 0.15 {
                        eprintln!(
                            "JUMP idx {idx} (frame {n}) {name} {:.2} m | seedΔ {} lostΔ {} med2d {:.1} n3d {} data_σ {:.2} iters {} cost {:.0}",
                            jump,
                            seed_now - prev_seed,
                            lost_now - prev_lost,
                            est.diag.med_2d_px,
                            est.diag.n_kp3d,
                            sig_d,
                            est.diag.iters,
                            est.diag.cost_final,
                        );
                    }
                }
            }
            prev_seed = seed_now;
            prev_lost = lost_now;
        }
        if let Some(p) = root_prev {
            root_jumps.push(norm(sub(est.state.root_t, p)));
        }
        lw_prev = Some(lw);
        rw_prev = Some(rw);
        root_prev = Some(est.state.root_t);
        lwr_sig.push(est.joint_data_sigma(m, h.j.l_wrist));
        rwr_sig.push(est.joint_data_sigma(m, h.j.r_wrist));

        // Avatar composite: retarget the recovered rig onto the VRM with the
        // same persistent state the app uses (display smoothing + anchor),
        // stepped every frame so the composite reflects live behaviour.
        if let Some((asset, renderer, rest_locals, rstate)) = avatar_rig.as_mut() {
            let mut locals = rest_locals.clone();
            if let (Some(rig), Some(hm)) = (est_out.skeleton.rig.as_ref(), asset.humanoid.as_ref())
            {
                apply_rig_pose(
                    rig,
                    &asset.skeleton,
                    hm,
                    &mut locals,
                    &RetargetParams::default(),
                    rstate,
                    est.diag.dt as f32,
                );
            }
            if std::env::var_os("VULVATAR_REPLAY_HEADCMP").is_some() && n % 8 == 0 {
                // Avatar head world orientation actually applied vs the rig delta.
                if let Some(hm) = asset.humanoid.as_ref() {
                    if let Some(node) = hm.bone_map.get(&vulvatar_lib::asset::HumanoidBone::Head) {
                        // world rot of head node under `locals`
                        let mut q = [0.0f32, 0.0, 0.0, 1.0];
                        let mut chain = vec![];
                        let mut i = node.0 as usize;
                        loop {
                            chain.push(i);
                            match asset.skeleton.nodes[i].parent {
                                Some(p) => i = p.0 as usize,
                                None => break,
                            }
                        }
                        for &k in chain.iter().rev() {
                            q = vulvatar_lib::math_utils::quat_mul(&q, &locals[k].rotation);
                        }
                        let fwd = vulvatar_lib::math_utils::quat_rotate_vec3(&q, &[0.0, 0.0, 1.0]);
                        let pitch_av = (-fwd[1]).asin().to_degrees();
                        let yaw_av = fwd[0].atan2(fwd[2]).to_degrees();
                        let rig_head = est_out.skeleton.rig.as_ref().and_then(|r| r.bones.get(&vulvatar_lib::asset::HumanoidBone::Head));
                        let (ry, rp) = rig_head
                            .map(|b| {
                                let m3 = vulvatar_lib::tracking::fusion::math::quat_to_mat(b.delta_world);
                                ypr_deg(&m3)
                            })
                            .map(|(y, p, _)| (y, p))
                            .unwrap_or((0.0, 0.0));
                        eprintln!("HEADCMP idx {idx} rig yaw/pitch {ry:.0}/{rp:.0} avatar yaw/pitch {yaw_av:.0}/{pitch_av:.0}");
                    }
                }
            }
            if render_every > 0 && n % render_every == 0 {
                let inst = offline::make_instance(asset, locals);
                let side = ch.min(720);
                let rgba =
                    offline::render_avatar(renderer, &inst, [side, side], &offline::bench_camera())?;
                let av = image::RgbaImage::from_raw(side, side, rgba)
                    .ok_or("avatar pixels")?;
                let mut comp = image::RgbImage::new(cw + side, ch);
                for (x, y, p) in rgb.enumerate_pixels() {
                    comp.put_pixel(x, y, *p);
                }
                for (x, y, p) in av.enumerate_pixels() {
                    let yy = y + (ch - side) / 2;
                    if yy < ch {
                        comp.put_pixel(cw + x, yy, image::Rgb([p[0], p[1], p[2]]));
                    }
                }
                let out = out_dir.join(format!("composite_{idx:05}.png"));
                comp.save(&out).map_err(|e| e.to_string())?;
            }
        }

        if render_every > 0 && n % render_every == 0 && avatar_rig.is_none() {
            let mut img = rgb.clone();
            let intr = Intrinsics {
                fx: intr_cam.fx as f64,
                fy: intr_cam.fy as f64,
                cx: intr_cam.cx as f64,
                cy: intr_cam.cy as f64,
                width: cw as f64,
                height: ch as f64,
            };
            // surface points
            for p in &provider.last_surface {
                if let Some(uv) = intr.project([p[0] as f64, p[1] as f64, p[2] as f64]) {
                    draw_dot(&mut img, uv, 4, [255, 200, 0]);
                }
            }
            // capsules
            for c in &m.capsules {
                let a = fk.point(c.a);
                let b = fk.point(c.b);
                if let (Some(pa), Some(pb)) = (intr.project(a), intr.project(b)) {
                    draw_line(&mut img, pa, pb, [60, 120, 255]);
                }
            }
            // bones (joint → parent), coloured by σ
            for (j, jd) in m.joints.iter().enumerate() {
                let Some(p) = jd.parent else { continue };
                if let (Some(pa), Some(pb)) = (intr.project(fk.t[j]), intr.project(fk.t[p])) {
                    let s = sig(j);
                    draw_line(&mut img, pa, pb, sigma_color(s));
                }
            }
            for j in 0..m.joints.len() {
                if let Some(p) = intr.project(fk.t[j]) {
                    draw_dot(&mut img, p, 3, sigma_color(sig(j)));
                }
            }
            for s in [h.s.nose, h.s.l_eye, h.s.r_eye, h.s.l_ear, h.s.r_ear] {
                if let Some(p) = intr.project(fk.site[s]) {
                    draw_dot(&mut img, p, 3, [255, 255, 255]);
                }
            }
            // hand crops (yellow box) + hand landmarks (orange)
            for hr in provider.last_hands.iter().flatten() {
                let (x, y, sz) = hr.crop;
                let (x, y, sz) = (x as f64, y as f64, sz as f64);
                draw_line(&mut img, [x, y], [x + sz, y], [255, 220, 0]);
                draw_line(&mut img, [x + sz, y], [x + sz, y + sz], [255, 220, 0]);
                draw_line(&mut img, [x + sz, y + sz], [x, y + sz], [255, 220, 0]);
                draw_line(&mut img, [x, y + sz], [x, y], [255, 220, 0]);
                for p in &hr.px {
                    draw_dot(&mut img, [p[0] as f64, p[1] as f64], 2, [255, 140, 0]);
                }
            }
            // detector keypoints (magenta = body 17, cyan = hands, grey = face)
            for (i, &(nx, ny, sc)) in est_out.annotation.keypoints.iter().enumerate() {
                if sc < 0.2 {
                    continue;
                }
                let c = if i < 17 {
                    [255, 0, 255]
                } else if i >= 91 {
                    [0, 255, 255]
                } else {
                    [160, 160, 160]
                };
                draw_dot(&mut img, [nx as f64 * cw as f64, ny as f64 * ch as f64], 2, c);
            }
            let out = out_dir.join(format!("overlay_{idx:05}.png"));
            img.save(&out).map_err(|e| e.to_string())?;
        }
    }
    std::fs::write(out_dir.join("frames.csv"), csv).map_err(|e| e.to_string())?;
    if vis_dump {
        std::fs::write(out_dir.join("kps.csv"), vis_csv).map_err(|e| e.to_string())?;
        println!("kps: {}", out_dir.join("kps.csv").display());
    }

    let (ym, ys, ymin, ymax) = stats(&torso_yaws);
    let (hm, hs, hmin, hmax) = stats(&head_yaws);
    let (pm, ps, pmin, pmax) = stats(&head_pitches);
    let (sm, _, _, smax) = stats(&solve_ms);
    let (_, _, _, lwmax) = stats(&lw_jumps);
    let (_, _, _, rwmax) = stats(&rw_jumps);
    let (_, _, _, rjmax) = stats(&root_jumps);
    let lw_snaps = lw_jumps.iter().filter(|&&j| j > 0.15).count();
    let rw_snaps = rw_jumps.iter().filter(|&&j| j > 0.15).count();
    let lw_duty = lwr_sig.iter().filter(|&&s| s < 0.4).count() as f64 / lwr_sig.len().max(1) as f64;
    let rw_duty = rwr_sig.iter().filter(|&&s| s < 0.4).count() as f64 / rwr_sig.len().max(1) as f64;
    let (_, _, _, lkmax) = stats(&lk_jumps);
    let (_, _, _, rkmax) = stats(&rk_jumps);
    let lk_snaps = lk_jumps.iter().filter(|&&j| j > 0.15).count();
    let rk_snaps = rk_jumps.iter().filter(|&&j| j > 0.15).count();
    let leg_names = ["Lhip", "Lknee", "Lankle", "Rhip", "Rknee", "Rankle"];
    let leg_duties: Vec<(usize, f64)> = (0..6)
        .map(|k| {
            (
                k,
                leg_sig.iter().filter(|s| s[k] < 0.4).count() as f64
                    / leg_sig.len().max(1) as f64,
            )
        })
        .collect();
    let leg_duty_s = leg_duties
        .iter()
        .map(|(k, d)| format!("{} {:.2}", leg_names[*k], d))
        .collect::<Vec<_>>()
        .join("  ");
    println!("=== fusion replay: {} frames ===", pairs.len());
    let (em, _, _, emax) = stats(&est_ms);
    println!("solve time     : mean {sm:.1} ms  max {smax:.1} ms (estimator only: mean {em:.1} ms max {emax:.1} ms)");
    println!("torso yaw (deg): mean {ym:+.1} std {ys:.1} range [{ymin:+.1}, {ymax:+.1}]");
    if !yaw_ref_pairs.is_empty() {
        let errs: Vec<f64> = yaw_ref_pairs.iter().map(|(a, b)| a - b).collect();
        let (em, es, emin, emax) = stats(&errs);
        let refs: Vec<f64> = yaw_ref_pairs.iter().map(|(_, b)| *b).collect();
        let (rm, rs, _, _) = stats(&refs);
        println!("torso yaw vs shoulder-depth reference ({} frames): ref mean {rm:+.1} std {rs:.1} | err mean {em:+.1} std {es:.1} range [{emin:+.1}, {emax:+.1}]", yaw_ref_pairs.len());
    }
    println!("head yaw (deg) : mean {hm:+.1} std {hs:.1} range [{hmin:+.1}, {hmax:+.1}]");
    println!("head pitch(deg): mean {pm:+.1} std {ps:.1} range [{pmin:+.1}, {pmax:+.1}]");
    println!("L wrist: max jump {lwmax:.3} m, snaps>0.15m {lw_snaps}, data-σ<0.4 duty {lw_duty:.2}");
    println!("R wrist: max jump {rwmax:.3} m, snaps>0.15m {rw_snaps}, data-σ<0.4 duty {rw_duty:.2}");
    println!("root: max jump {rjmax:.3} m");
    println!("L knee : max jump {lkmax:.3} m, snaps>0.15m {lk_snaps}");
    println!("R knee : max jump {rkmax:.3} m, snaps>0.15m {rk_snaps}");
    println!("leg data-σ<0.4 duty: {leg_duty_s}");
    for (name, v) in ["shoulders", "elbows", "wrists", "other"].iter().zip(kp3d_err.iter()) {
        if v.is_empty() { continue; }
        let mut s = v.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (m, _, _, mx) = stats(v);
        println!("metric-joint residual {name:>9}: n {} mean {:.3} med {:.3} p90 {:.3} max {:.3} m", v.len(), m, s[s.len()/2], s[(s.len()*9/10).min(s.len()-1)], mx);
    }
    println!("estimator: seed wins {}  re-acquisitions {}  cov failures {}", provider.estimator().diag.seed_wins, provider.estimator().lost_events, provider.estimator().diag.cov_failures);
    println!("hand crops: L {} R {} frames with presence≥0.5 (of {}); hand-block L/R re-labels {}; duplicate locks {}", hand_frames[0], hand_frames[1], pairs.len(), provider.hand_swaps, provider.hand_dupes);
    println!("csv: {}", out_dir.join("frames.csv").display());
    Ok(())
}
