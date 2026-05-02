//! Per-image deep diagnostic for the `rtmw3d-with-depth` provider's
//! elbow Z signal. Replicates the runtime pipeline (RTMW3D 2D + DAv2
//! depth + shoulder-span calibration) in a single synchronous pass and
//! dumps everything we need to pinpoint where the elbow's depth is
//! getting lost: window-pollution (7×7 median pulled to the
//! background), calibration degeneracy (anchor pair too close in
//! image), or just plain DAv2 grid coarseness.
//!
//! Two modes:
//!
//! ```text
//!   # Bulk JSONL across a directory tree
//!   diagnose_elbow_depth <dir> --out diagnostics/elbow_bulk.jsonl
//!
//!   # Per-image visualisation (depth heatmap + zoom + summary)
//!   diagnose_elbow_depth <image.png> --viz diagnostics/elbow_<stem>/
//! ```
//!
//! Bulk JSONL columns (one row per image):
//!
//!   image, width, height,
//!   calibration: { anchor, scale_c, anchor_2d_separation, d_raw_a, d_raw_b },
//!   <bone>: {
//!     nx, ny, score,             // RTMW3D output
//!     d_raw_r0, d_raw_r1, d_raw_r3,   // DAv2 inverse-depth median, radius 0/1/3
//!     z_metric_r0, z_metric_r1, z_metric_r3, // c / d_raw at each radius
//!     z_window_std,              // z stddev across the 7×7 window
//!     valid_pct                  // fraction of 7×7 pixels with finite depth
//!   }
//!
//! Where <bone> ∈ { l_shoulder, r_shoulder, l_elbow, r_elbow,
//!                  l_wrist, r_wrist, l_hip, r_hip }.
//!
//! Comparing `z_metric_r0` (single-pixel) against `z_metric_r3` (the
//! runtime median) reveals window pollution. Comparing `elbow.z` to
//! `shoulder.z` reveals limb-vs-torso depth contrast. Calibration
//! `scale_c` outside [1, 6] for indoor webcam framings flags the
//! anchor-degeneracy case described in `rtmw3d_with_depth.rs`.

use std::fs;
use std::path::{Path, PathBuf};

use image::{ImageBuffer, Rgb, RgbImage, Rgba, RgbaImage};
use vulvatar_lib::tracking::cigpose::{DecodedJoint2d, NUM_JOINTS};
use vulvatar_lib::tracking::depth_anything::{DepthAnythingFrame, DepthAnythingV2Inference};
use vulvatar_lib::tracking::rtmw3d::Rtmw3dInference;
use vulvatar_lib::tracking::rtmw3d_with_depth::{
    build_metric_frame_from_dav2, calibrate_scale, Calibration,
};
use vulvatar_lib::tracking::skeleton_from_depth::{
    sample_metric_point_with_radius, KEYPOINT_VISIBILITY_FLOOR, SAMPLE_RADIUS_PX,
};

/// COCO-Wholebody indices we track for the elbow-depth report. Order
/// matters — the JSONL columns appear in this order.
const INTEREST_JOINTS: &[(usize, &str)] = &[
    (5, "l_shoulder"),
    (6, "r_shoulder"),
    (7, "l_elbow"),
    (8, "r_elbow"),
    (9, "l_wrist"),
    (10, "r_wrist"),
    (11, "l_hip"),
    (12, "r_hip"),
];

fn main() -> Result<(), String> {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let target = args
        .next()
        .ok_or_else(|| "usage: diagnose_elbow_depth <image_or_dir> [--out file.jsonl] [--viz dir]".to_string())?;
    let target_path = PathBuf::from(&target);

    let mut out_jsonl: Option<PathBuf> = None;
    let mut viz_dir: Option<PathBuf> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out_jsonl = args.next().map(PathBuf::from),
            "--viz" => viz_dir = args.next().map(PathBuf::from),
            other => return Err(format!("unknown arg '{other}'")),
        }
    }

    if let Some(ref dir) = viz_dir {
        if dir.components().any(|c| c.as_os_str() == "validation_images") {
            return Err(format!(
                "refusing to write viz output under validation_images/ (reserved for input). got: {}",
                dir.display()
            ));
        }
        fs::create_dir_all(dir)
            .map_err(|e| format!("create_dir_all '{}': {e}", dir.display()))?;
    }
    if let Some(ref p) = out_jsonl {
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("create_dir_all '{}': {e}", parent.display()))?;
            }
        }
    }

    let images = collect_images(&target_path)?;
    eprintln!("diagnose_elbow_depth: {} image(s)", images.len());

    let mut rtmw3d = Rtmw3dInference::from_models_dir("models")
        .map_err(|e| format!("RTMW3D load: {e}"))?;
    let mut dav2 = DepthAnythingV2Inference::from_model_path("models/dav2_small.onnx")
        .map_err(|e| format!("DAv2 load: {e}"))?;
    eprintln!(
        "models loaded: RTMW3D {}, DAv2 {}",
        rtmw3d.backend().label(),
        dav2.backend().label()
    );
    let _ = rtmw3d.take_load_warnings();

    let mut jsonl = String::new();
    for (idx, image_path) in images.iter().enumerate() {
        let img = match image::open(image_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("[{idx:>3}] open failed: {e} — skipping {}", image_path.display());
                continue;
            }
        };
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());

        // Two warm-up RTMW3D passes: yolox_worker is async and
        // benefits from a steady-state bbox before we measure.
        let _ = rtmw3d.estimate_pose(rgb.as_raw(), w, h, 0);
        let _ = rtmw3d.estimate_pose(rgb.as_raw(), w, h, 1);
        let est = rtmw3d.estimate_pose(rgb.as_raw(), w, h, 2);

        let joints_2d: Vec<DecodedJoint2d> = est
            .annotation
            .keypoints
            .iter()
            .map(|&(nx, ny, score)| DecodedJoint2d { nx, ny, score })
            .collect();
        if joints_2d.len() < NUM_JOINTS {
            eprintln!("[{idx:>3}] RTMW3D returned {} keypoints (<{NUM_JOINTS}) — skipping", joints_2d.len());
            continue;
        }

        let depth = match dav2.estimate(rgb.as_raw(), w, h) {
            Some(d) => d,
            None => {
                eprintln!("[{idx:>3}] DAv2 returned None — skipping");
                continue;
            }
        };
        // Diagnostic always runs without an explicit pose calibration —
        // we want to see the raw, EMA-uncorrected calibration each
        // image would produce on a cold provider.
        let calib = match calibrate_scale(&joints_2d, &depth, w, h, None) {
            Some(c) => c,
            None => {
                eprintln!("[{idx:>3}] calibration failed — skipping");
                continue;
            }
        };
        let metric = build_metric_frame_from_dav2(&depth, &calib);

        let row = build_row(image_path, w, h, &joints_2d, &depth, &metric, &calib);
        eprintln!(
            "[{idx:>3}] {} → anchor={} scale_c={:.3} L_elbow.z(r0/r1/r3)={:.2}/{:.2}/{:.2}  R_elbow.z(r0/r1/r3)={:.2}/{:.2}/{:.2}",
            image_path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
            calib.anchor_label,
            calib.scale_c,
            row.bones[2].z_metric_r0,
            row.bones[2].z_metric_r1,
            row.bones[2].z_metric_r3,
            row.bones[3].z_metric_r0,
            row.bones[3].z_metric_r1,
            row.bones[3].z_metric_r3,
        );
        jsonl.push_str(&row.to_jsonl_line());
        jsonl.push('\n');

        if let Some(ref viz) = viz_dir {
            // viz mode is single-image-friendly — when given a dir
            // target, write each image's outputs into a subdir
            // named by stem.
            let dst_dir = if images.len() == 1 {
                viz.clone()
            } else {
                let stem = image_path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| format!("img_{idx}"));
                viz.join(stem)
            };
            fs::create_dir_all(&dst_dir)
                .map_err(|e| format!("create_dir_all '{}': {e}", dst_dir.display()))?;
            write_viz(&dst_dir, &img.to_rgba8(), &joints_2d, &depth, &calib, &row)?;
        }
    }

    if let Some(out_path) = out_jsonl {
        fs::write(&out_path, &jsonl).map_err(|e| format!("write {}: {e}", out_path.display()))?;
        eprintln!("wrote {}", out_path.display());
    } else if viz_dir.is_none() {
        // Default: print JSONL to stdout (so callers can pipe to jq).
        print!("{jsonl}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Image collection
// ---------------------------------------------------------------------------

fn collect_images(target: &Path) -> Result<Vec<PathBuf>, String> {
    if target.is_file() {
        return Ok(vec![target.to_path_buf()]);
    }
    if !target.is_dir() {
        return Err(format!("target not found: {}", target.display()));
    }
    fn walk(dir: &Path, acc: &mut Vec<PathBuf>) -> Result<(), String> {
        for entry in
            fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?
        {
            let entry = entry.map_err(|e| format!("entry: {e}"))?;
            let path = entry.path();
            if path.is_dir() {
                walk(&path, acc)?;
            } else if matches!(
                path.extension().and_then(|s| s.to_str()),
                Some("png" | "jpg" | "jpeg")
            ) {
                acc.push(path);
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(target, &mut out)?;
    out.sort();
    Ok(out)
}

// ---------------------------------------------------------------------------
// Per-image row
// ---------------------------------------------------------------------------

struct Row {
    image: String,
    width: u32,
    height: u32,
    calibration: CalibRow,
    bones: Vec<BoneRow>,
}

struct CalibRow {
    anchor: String,
    scale_c: f32,
    anchor_2d_separation: f32,
    d_raw_a: f32,
    d_raw_b: f32,
}

struct BoneRow {
    label: &'static str,
    nx: f32,
    ny: f32,
    score: f32,
    d_raw_r0: f32,
    d_raw_r1: f32,
    d_raw_r3: f32,
    z_metric_r0: f32,
    z_metric_r1: f32,
    z_metric_r3: f32,
    z_window_std: f32,
    valid_pct: f32,
}

fn build_row(
    image_path: &Path,
    width: u32,
    height: u32,
    joints: &[DecodedJoint2d],
    depth: &DepthAnythingFrame,
    metric: &vulvatar_lib::tracking::metric_depth::MoGeFrame,
    calib: &Calibration,
) -> Row {
    let anchor_2d_separation = match calib.anchor_label {
        "shoulder-span" => pair_2d_distance(joints, 5, 6),
        "hip-span" => pair_2d_distance(joints, 11, 12),
        _ => f32::NAN,
    };

    let bones = INTEREST_JOINTS
        .iter()
        .map(|&(idx, label)| build_bone_row(label, joints, idx, depth, metric))
        .collect();

    Row {
        image: image_path
            .strip_prefix(std::env::current_dir().unwrap_or_default())
            .unwrap_or(image_path)
            .display()
            .to_string()
            .replace('\\', "/"),
        width,
        height,
        calibration: CalibRow {
            anchor: calib.anchor_label.to_string(),
            scale_c: calib.scale_c,
            anchor_2d_separation,
            d_raw_a: calib.d_raw_a,
            d_raw_b: calib.d_raw_b,
        },
        bones,
    }
}

fn pair_2d_distance(joints: &[DecodedJoint2d], a: usize, b: usize) -> f32 {
    let ja = &joints[a];
    let jb = &joints[b];
    let dx = ja.nx - jb.nx;
    let dy = ja.ny - jb.ny;
    (dx * dx + dy * dy).sqrt()
}

fn build_bone_row(
    label: &'static str,
    joints: &[DecodedJoint2d],
    idx: usize,
    depth: &DepthAnythingFrame,
    metric: &vulvatar_lib::tracking::metric_depth::MoGeFrame,
) -> BoneRow {
    let j = &joints[idx];
    let d_raw_r0 = depth.sample_inverse_depth(j.nx, j.ny, 0).unwrap_or(f32::NAN);
    let d_raw_r1 = depth.sample_inverse_depth(j.nx, j.ny, 1).unwrap_or(f32::NAN);
    let d_raw_r3 = depth
        .sample_inverse_depth(j.nx, j.ny, SAMPLE_RADIUS_PX)
        .unwrap_or(f32::NAN);

    let z_r0 = sample_metric_point_with_radius(metric, j.nx, j.ny, 0)
        .map(|p| p[2])
        .unwrap_or(f32::NAN);
    let z_r1 = sample_metric_point_with_radius(metric, j.nx, j.ny, 1)
        .map(|p| p[2])
        .unwrap_or(f32::NAN);
    let z_r3 = sample_metric_point_with_radius(metric, j.nx, j.ny, SAMPLE_RADIUS_PX)
        .map(|p| p[2])
        .unwrap_or(f32::NAN);

    let (z_window_std, valid_pct) = window_z_stats(metric, j.nx, j.ny, SAMPLE_RADIUS_PX);
    BoneRow {
        label,
        nx: j.nx,
        ny: j.ny,
        score: j.score,
        d_raw_r0,
        d_raw_r1,
        d_raw_r3,
        z_metric_r0: z_r0,
        z_metric_r1: z_r1,
        z_metric_r3: z_r3,
        z_window_std,
        valid_pct,
    }
}

fn window_z_stats(
    frame: &vulvatar_lib::tracking::metric_depth::MoGeFrame,
    nx: f32,
    ny: f32,
    radius_px: i32,
) -> (f32, f32) {
    if !nx.is_finite() || !ny.is_finite() {
        return (f32::NAN, 0.0);
    }
    let w = frame.width as i32;
    let h = frame.height as i32;
    let cx = (nx * frame.width as f32).round() as i32;
    let cy = (ny * frame.height as f32).round() as i32;
    if cx < 0 || cy < 0 || cx >= w || cy >= h {
        return (f32::NAN, 0.0);
    }
    let total = ((radius_px * 2 + 1) * (radius_px * 2 + 1)) as f32;
    let mut zs: Vec<f32> = Vec::with_capacity(total as usize);
    for yy in (cy - radius_px).max(0)..=(cy + radius_px).min(h - 1) {
        for xx in (cx - radius_px).max(0)..=(cx + radius_px).min(w - 1) {
            let idx = yy as usize * frame.width as usize + xx as usize;
            let z = frame.points_m[idx][2];
            if z.is_finite() && z > 0.0 {
                zs.push(z);
            }
        }
    }
    let valid = zs.len() as f32;
    if valid < 2.0 {
        return (f32::NAN, valid / total);
    }
    let mean = zs.iter().copied().sum::<f32>() / valid;
    let var = zs.iter().map(|z| (z - mean).powi(2)).sum::<f32>() / valid;
    (var.sqrt(), valid / total)
}

// ---------------------------------------------------------------------------
// JSONL serialisation (manual to avoid pulling serde_json)
// ---------------------------------------------------------------------------

impl Row {
    fn to_jsonl_line(&self) -> String {
        let mut s = String::new();
        s.push('{');
        push_str(&mut s, "image", &self.image, true);
        push_num(&mut s, "width", self.width as f32, false);
        push_num(&mut s, "height", self.height as f32, false);
        s.push(',');
        s.push_str("\"calibration\":");
        s.push_str(&self.calibration.to_json());
        for bone in &self.bones {
            s.push(',');
            s.push('"');
            s.push_str(bone.label);
            s.push_str("\":");
            s.push_str(&bone.to_json());
        }
        s.push('}');
        s
    }
}

impl CalibRow {
    fn to_json(&self) -> String {
        format!(
            "{{\"anchor\":\"{}\",\"scale_c\":{:.4},\"anchor_2d_separation\":{:.4},\"d_raw_a\":{:.4},\"d_raw_b\":{:.4}}}",
            self.anchor, self.scale_c, self.anchor_2d_separation, self.d_raw_a, self.d_raw_b
        )
    }
}

impl BoneRow {
    fn to_json(&self) -> String {
        format!(
            "{{\"nx\":{},\"ny\":{},\"score\":{},\"d_raw_r0\":{},\"d_raw_r1\":{},\"d_raw_r3\":{},\"z_r0\":{},\"z_r1\":{},\"z_r3\":{},\"z_std\":{},\"valid_pct\":{}}}",
            f(self.nx), f(self.ny), f(self.score),
            f(self.d_raw_r0), f(self.d_raw_r1), f(self.d_raw_r3),
            f(self.z_metric_r0), f(self.z_metric_r1), f(self.z_metric_r3),
            f(self.z_window_std), f(self.valid_pct),
        )
    }
}

fn push_str(s: &mut String, key: &str, val: &str, first: bool) {
    if !first {
        s.push(',');
    }
    s.push('"');
    s.push_str(key);
    s.push_str("\":\"");
    for c in val.chars() {
        if c == '"' || c == '\\' {
            s.push('\\');
        }
        s.push(c);
    }
    s.push('"');
}
fn push_num(s: &mut String, key: &str, val: f32, first: bool) {
    if !first {
        s.push(',');
    }
    s.push('"');
    s.push_str(key);
    s.push_str("\":");
    s.push_str(&f(val));
}
fn f(v: f32) -> String {
    if v.is_finite() {
        format!("{:.4}", v)
    } else {
        "null".to_string()
    }
}

// ---------------------------------------------------------------------------
// Visualisation (depth heatmap + keypoint overlay + per-elbow zoom)
// ---------------------------------------------------------------------------

fn write_viz(
    dst_dir: &Path,
    rgba: &RgbaImage,
    joints: &[DecodedJoint2d],
    depth: &DepthAnythingFrame,
    calib: &Calibration,
    row: &Row,
) -> Result<(), String> {
    save_depth_heatmap(depth, &dst_dir.join("depth_heatmap.png"))?;
    save_depth_overlay(rgba, joints, depth, calib, &dst_dir.join("depth_overlay.png"))?;
    for bone in row.bones.iter() {
        // Only zoom on the elbows + wrists — that's where the action
        // is. Shoulders / hips already get summarised via calibration.
        if matches!(bone.label, "l_elbow" | "r_elbow" | "l_wrist" | "r_wrist") {
            let path = dst_dir.join(format!("zoom_{}.png", bone.label));
            save_zoom(rgba, depth, bone.nx, bone.ny, &path)?;
        }
    }
    let summary = render_summary(row, calib);
    fs::write(dst_dir.join("summary.txt"), summary)
        .map_err(|e| format!("write summary: {e}"))?;
    Ok(())
}

fn save_depth_heatmap(depth: &DepthAnythingFrame, out: &Path) -> Result<(), String> {
    let w = depth.width;
    let h = depth.height;
    // Use raw inverse-depth values directly (larger ≈ closer). 1–99
    // percentile clipping so background asymptote doesn't crush the
    // colour ramp.
    let mut vs: Vec<f32> = depth.relative.iter().copied().filter(|v| v.is_finite() && *v > 0.0).collect();
    if vs.is_empty() {
        return Err("no finite depth pixels".to_string());
    }
    vs.sort_by(|a, b| a.total_cmp(b));
    let lo = vs[vs.len() / 100];
    let hi = vs[vs.len() * 99 / 100];
    let mut img: RgbImage = ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let v = depth.relative[idx];
            let pixel = if v.is_finite() && v > 0.0 {
                let t = ((v - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0);
                turbo_like(t)
            } else {
                Rgb([0, 0, 0])
            };
            img.put_pixel(x, y, pixel);
        }
    }
    img.save(out).map_err(|e| format!("save depth heatmap: {e}"))?;
    Ok(())
}

/// Original RGB at full source resolution, with depth heatmap blended
/// at 60% on top, keypoint markers, and 7×7 sampling boxes around
/// shoulders / elbows / wrists / hips. The composite makes it obvious
/// when the elbow box straddles an arm-vs-background boundary.
fn save_depth_overlay(
    rgba: &RgbaImage,
    joints: &[DecodedJoint2d],
    depth: &DepthAnythingFrame,
    _calib: &Calibration,
    out: &Path,
) -> Result<(), String> {
    let (w, h) = (rgba.width(), rgba.height());
    let mut canvas: RgbaImage = rgba.clone();

    // Upscale depth to source res (nearest neighbour) and blend.
    let mut vs: Vec<f32> = depth.relative.iter().copied().filter(|v| v.is_finite() && *v > 0.0).collect();
    vs.sort_by(|a, b| a.total_cmp(b));
    let lo = vs[vs.len() / 100];
    let hi = vs[vs.len() * 99 / 100];
    for y in 0..h {
        for x in 0..w {
            let dx = ((x as f32 + 0.5) / w as f32 * depth.width as f32) as u32;
            let dy = ((y as f32 + 0.5) / h as f32 * depth.height as f32) as u32;
            let dx = dx.min(depth.width - 1);
            let dy = dy.min(depth.height - 1);
            let v = depth.relative[(dy * depth.width + dx) as usize];
            if !v.is_finite() || v <= 0.0 {
                continue;
            }
            let t = ((v - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0);
            let heat = turbo_like(t);
            let p = canvas.get_pixel_mut(x, y);
            // 60% heatmap over RGB.
            p[0] = blend(p[0], heat[0], 0.6);
            p[1] = blend(p[1], heat[1], 0.6);
            p[2] = blend(p[2], heat[2], 0.6);
        }
    }

    // Draw 7×7 sampling boxes (in DAv2 grid → mapped to source) and
    // labelled markers for each joint of interest.
    let half = SAMPLE_RADIUS_PX as f32;
    let grid_w = depth.width as f32;
    let grid_h = depth.height as f32;
    for &(idx, label) in INTEREST_JOINTS {
        let j = &joints[idx];
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            continue;
        }
        let cx = j.nx * w as f32;
        let cy = j.ny * h as f32;
        let box_w = (2.0 * half + 1.0) * (w as f32 / grid_w);
        let box_h = (2.0 * half + 1.0) * (h as f32 / grid_h);
        let x0 = (cx - box_w * 0.5).round() as i32;
        let y0 = (cy - box_h * 0.5).round() as i32;
        let x1 = x0 + box_w.round() as i32;
        let y1 = y0 + box_h.round() as i32;
        let color = bone_colour(label);
        draw_rect(&mut canvas, x0, y0, x1, y1, color);
        draw_disc(&mut canvas, cx as i32, cy as i32, 4, color);
    }

    canvas.save(out).map_err(|e| format!("save overlay: {e}"))?;
    Ok(())
}

/// 21×21 (in DAv2 grid units) source-pixel zoom around (nx,ny) with
/// the 7×7 sampling box drawn in green. Lets us eyeball whether the
/// median window straddles arm-vs-background.
fn save_zoom(
    rgba: &RgbaImage,
    depth: &DepthAnythingFrame,
    nx: f32,
    ny: f32,
    out: &Path,
) -> Result<(), String> {
    let (w, h) = (rgba.width(), rgba.height());
    // 21×21 DAv2-grid box → source-pixel box.
    let half = 10.0_f32 * (w as f32 / depth.width as f32);
    let half_y = 10.0_f32 * (h as f32 / depth.height as f32);
    let cx = (nx * w as f32).round();
    let cy = (ny * h as f32).round();
    let x0 = (cx - half).max(0.0) as u32;
    let y0 = (cy - half_y).max(0.0) as u32;
    let x1 = ((cx + half).round() as u32).min(w);
    let y1 = ((cy + half_y).round() as u32).min(h);
    if x1 <= x0 || y1 <= y0 {
        return Err("zoom box collapsed".to_string());
    }
    let crop = image::imageops::crop_imm(rgba, x0, y0, x1 - x0, y1 - y0).to_image();
    // Scale up 4× for visibility.
    let scaled = image::imageops::resize(
        &crop,
        crop.width() * 4,
        crop.height() * 4,
        image::imageops::FilterType::Nearest,
    );
    let mut out_img = scaled;
    // Draw the 7×7 sampling box on the zoom.
    let box_half_x = (SAMPLE_RADIUS_PX as f32 + 0.5) * (w as f32 / depth.width as f32) * 4.0;
    let box_half_y = (SAMPLE_RADIUS_PX as f32 + 0.5) * (h as f32 / depth.height as f32) * 4.0;
    let mid_x = ((cx - x0 as f32) * 4.0) as i32;
    let mid_y = ((cy - y0 as f32) * 4.0) as i32;
    draw_rect(
        &mut out_img,
        mid_x - box_half_x as i32,
        mid_y - box_half_y as i32,
        mid_x + box_half_x as i32,
        mid_y + box_half_y as i32,
        Rgba([0, 255, 80, 255]),
    );
    draw_disc(&mut out_img, mid_x, mid_y, 3, Rgba([255, 60, 60, 255]));
    out_img.save(out).map_err(|e| format!("save zoom: {e}"))?;
    Ok(())
}

fn render_summary(row: &Row, calib: &Calibration) -> String {
    let mut s = String::new();
    s.push_str(&format!("image: {}\n", row.image));
    s.push_str(&format!("size:  {}x{}\n\n", row.width, row.height));
    s.push_str("[calibration]\n");
    s.push_str(&format!("anchor:               {}\n", calib.anchor_label));
    s.push_str(&format!("scale_c:              {:.4}\n", calib.scale_c));
    s.push_str(&format!(
        "anchor 2D separation: {:.4} (frame-norm)\n",
        row.calibration.anchor_2d_separation
    ));
    s.push_str(&format!("d_raw_a / d_raw_b:    {:.4} / {:.4}\n\n", calib.d_raw_a, calib.d_raw_b));
    s.push_str("[per-bone depth window]\n");
    s.push_str("bone          score  nx     ny     d_r0    d_r1    d_r3    z_r0    z_r1    z_r3    z_std  valid%\n");
    for b in &row.bones {
        s.push_str(&format!(
            "{:<13} {:>5.2}  {:>5.3}  {:>5.3}  {:>6.3}  {:>6.3}  {:>6.3}  {:>6.2}  {:>6.2}  {:>6.2}  {:>5.2}  {:>5.0}%\n",
            b.label,
            b.score,
            b.nx,
            b.ny,
            b.d_raw_r0,
            b.d_raw_r1,
            b.d_raw_r3,
            b.z_metric_r0,
            b.z_metric_r1,
            b.z_metric_r3,
            b.z_window_std,
            b.valid_pct * 100.0,
        ));
    }
    s.push_str("\nz_r0 = single-pixel sample, z_r3 = 7x7 median (runtime default).\n");
    s.push_str("Large gap z_r0 vs z_r3 = window pollution (limb median pulled to background).\n");
    s
}

fn bone_colour(label: &str) -> Rgba<u8> {
    match label {
        "l_shoulder" | "r_shoulder" => Rgba([0, 200, 255, 255]),
        "l_elbow" | "r_elbow" => Rgba([255, 80, 80, 255]),
        "l_wrist" | "r_wrist" => Rgba([0, 255, 80, 255]),
        "l_hip" | "r_hip" => Rgba([255, 220, 0, 255]),
        _ => Rgba([240, 240, 240, 255]),
    }
}

fn draw_rect(img: &mut RgbaImage, x0: i32, y0: i32, x1: i32, y1: i32, c: Rgba<u8>) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let x0 = x0.clamp(0, w - 1);
    let x1 = x1.clamp(0, w - 1);
    let y0 = y0.clamp(0, h - 1);
    let y1 = y1.clamp(0, h - 1);
    for x in x0..=x1 {
        img.put_pixel(x as u32, y0 as u32, c);
        img.put_pixel(x as u32, y1 as u32, c);
    }
    for y in y0..=y1 {
        img.put_pixel(x0 as u32, y as u32, c);
        img.put_pixel(x1 as u32, y as u32, c);
    }
}

fn draw_disc(img: &mut RgbaImage, cx: i32, cy: i32, r: i32, c: Rgba<u8>) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let r2 = r * r;
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy > r2 {
                continue;
            }
            let x = cx + dx;
            let y = cy + dy;
            if x < 0 || y < 0 || x >= w || y >= h {
                continue;
            }
            img.put_pixel(x as u32, y as u32, c);
        }
    }
}

fn blend(a: u8, b: u8, t: f32) -> u8 {
    ((a as f32) * (1.0 - t) + (b as f32) * t).round().clamp(0.0, 255.0) as u8
}

fn turbo_like(t: f32) -> Rgb<u8> {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b);
    if t < 0.5 {
        let s = t * 2.0;
        r = (s * 255.0) as u8;
        g = ((1.0 - (s - 0.5).abs() * 2.0) * 255.0) as u8;
        b = ((1.0 - s) * 255.0) as u8;
    } else {
        let s = (t - 0.5) * 2.0;
        r = 255;
        g = ((1.0 - s) * 255.0) as u8;
        b = (s * 60.0) as u8;
    }
    Rgb([r, g, b])
}
