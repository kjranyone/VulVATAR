//! Per-pipeline diagnostic for the CIGPose + MoGe-2 provider.
//!
//! Loads an input image, runs each ONNX session in isolation, then
//! the combined provider, and prints the numeric signals that show
//! the inference is actually working (keypoint scores, metric scale,
//! depth ranges, source-skeleton bone count). Saves a keypoint
//! overlay PNG and a depth visualisation PNG so the result is also
//! visually checkable.
//!
//! Usage:
//!   cargo run --release --bin diagnose_cigpose_moge -- <image.png> [<output_dir>]
//!
//! Outputs in `<output_dir>` (default: `diagnostics/<stem>_cigpose_moge/`
//! under the repo root — `diagnostics/` is gitignored):
//!   keypoints_overlay.png   — input with CIGPose 2D keypoints + scores
//!   depth_vis.png           — MoGe-2 depth heatmap (turbo-ish ramp)
//!   summary.txt             — text dump of the numeric checks
//!
//! `validation_images/` is reserved for inference *input* data and must
//! not receive diagnostic output (see CLAUDE.md). Passing an `<output_dir>`
//! under `validation_images/` is rejected.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use image::{ImageBuffer, Rgb, RgbImage, Rgba, RgbaImage};
use vulvatar_lib::tracking::cigpose::{CigposeInference, NUM_JOINTS};
use vulvatar_lib::tracking::metric_depth::{MoGe2Inference, MoGeFrame, INPUT_H as MOGE_H, INPUT_W as MOGE_W};
use vulvatar_lib::tracking::provider::{create_pose_provider, PoseProviderKind};

fn main() -> Result<(), String> {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let image_path = PathBuf::from(args.next().ok_or_else(|| {
        "usage: diagnose_cigpose_moge <image.png> [<output_dir>]".to_string()
    })?);
    let out_dir = match args.next() {
        Some(d) => PathBuf::from(d),
        None => {
            let stem = image_path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "diagnose".to_string());
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("diagnostics")
                .join(format!("{stem}_cigpose_moge"))
        }
    };
    if out_dir
        .components()
        .any(|c| c.as_os_str() == "validation_images")
    {
        return Err(format!(
            "refusing to write diagnostic output under validation_images/ \
             (reserved for inference input). got: {}",
            out_dir.display()
        ));
    }
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("create_dir_all '{}': {e}", out_dir.display()))?;

    println!("== diagnose_cigpose_moge ==");
    println!("image:  {}", image_path.display());
    println!("output: {}", out_dir.display());

    let img = image::open(&image_path).map_err(|e| format!("open image: {e}"))?;
    let rgb = img.to_rgb8();
    let (iw, ih) = (rgb.width(), rgb.height());
    println!("loaded: {iw}x{ih}");

    let mut summary = String::new();
    summary.push_str(&format!("image: {}\n", image_path.display()));
    summary.push_str(&format!("size:  {iw}x{ih}\n\n"));

    // ---- Stage 1: CIGPose alone ------------------------------------------
    println!("\n--- CIGPose (2D pose) ---");
    let mut cig = CigposeInference::from_model_path("models/cigpose.onnx")
        .map_err(|e| format!("CIGPose load: {e}"))?;
    println!("backend: {}", cig.backend().label());
    summary.push_str("[CIGPose]\n");
    summary.push_str(&format!("backend: {}\n", cig.backend().label()));

    let t0 = std::time::Instant::now();
    let joints_2d = cig.estimate(rgb.as_raw(), iw, ih);
    let dt_cig = t0.elapsed();
    println!("inference: {} keypoints in {:.1}ms", joints_2d.len(), dt_cig.as_secs_f32() * 1000.0);
    summary.push_str(&format!("inference: {} keypoints in {:.1}ms\n", joints_2d.len(), dt_cig.as_secs_f32() * 1000.0));

    if joints_2d.len() < NUM_JOINTS {
        return Err(format!(
            "CIGPose returned {} joints, expected {}",
            joints_2d.len(),
            NUM_JOINTS
        ));
    }

    // Spot-check the body anchor keypoints — these are the joints
    // every downstream pipeline cares about.
    let body_labels: &[(usize, &str)] = &[
        (0, "nose"),
        (5, "L-shoulder"),
        (6, "R-shoulder"),
        (7, "L-elbow"),
        (8, "R-elbow"),
        (9, "L-wrist"),
        (10, "R-wrist"),
        (11, "L-hip"),
        (12, "R-hip"),
        (13, "L-knee"),
        (14, "R-knee"),
    ];
    println!("body keypoints (selfie-mirror NOT applied):");
    summary.push_str("body keypoints:\n");
    let mut nonzero_body = 0_u32;
    for &(idx, lbl) in body_labels {
        let j = &joints_2d[idx];
        let line = format!(
            "  [{:>2}] {:<11} nx={:.3} ny={:.3} score={:.3}",
            idx, lbl, j.nx, j.ny, j.score
        );
        println!("{line}");
        summary.push_str(&line);
        summary.push('\n');
        if j.score > 0.05 {
            nonzero_body += 1;
        }
    }
    let mean_body_score: f32 =
        body_labels.iter().map(|&(i, _)| joints_2d[i].score).sum::<f32>() / body_labels.len() as f32;
    println!(
        "body anchors above floor: {}/{}, mean score {:.3}",
        nonzero_body,
        body_labels.len(),
        mean_body_score
    );
    summary.push_str(&format!(
        "body anchors above floor: {}/{}, mean score {:.3}\n\n",
        nonzero_body,
        body_labels.len(),
        mean_body_score
    ));

    // Save 2D overlay.
    let overlay_path = out_dir.join("keypoints_overlay.png");
    save_keypoint_overlay(&img.to_rgba8(), &joints_2d, &overlay_path)?;
    println!("wrote {}", overlay_path.display());

    // ---- Stage 2: MoGe-2 alone -------------------------------------------
    println!("\n--- MoGe-2 (metric depth) ---");
    let mut moge = MoGe2Inference::from_model_path("models/moge_v2.onnx")
        .map_err(|e| format!("MoGe-2 load: {e}"))?;
    println!("backend: {}", moge.backend().label());
    summary.push_str("[MoGe-2]\n");
    summary.push_str(&format!("backend: {}\n", moge.backend().label()));

    let t1 = std::time::Instant::now();
    let depth = moge
        .estimate(rgb.as_raw(), iw, ih)
        .ok_or_else(|| "MoGe-2 returned None — postprocess rejected outputs".to_string())?;
    let dt_moge = t1.elapsed();
    println!(
        "inference: {}x{} point cloud in {:.1}ms",
        depth.width,
        depth.height,
        dt_moge.as_secs_f32() * 1000.0
    );
    summary.push_str(&format!(
        "inference: {}x{} point cloud in {:.1}ms\n",
        depth.width,
        depth.height,
        dt_moge.as_secs_f32() * 1000.0
    ));

    let stats = depth_stats(&depth);
    let stats_line = format!(
        "valid pixels {}/{} ({:.1}%); Z range {:.3}m / median {:.3}m / max {:.3}m; centre Z {:.3}m",
        stats.valid,
        stats.total,
        stats.valid_pct(),
        stats.z_min,
        stats.z_median,
        stats.z_max,
        stats.z_centre
    );
    println!("{stats_line}");
    summary.push_str(&stats_line);
    summary.push_str("\n\n");

    // Save depth visualisation.
    let depth_path = out_dir.join("depth_vis.png");
    save_depth_vis(&depth, &depth_path)?;
    println!("wrote {}", depth_path.display());

    // ---- Stage 3: Combined provider --------------------------------------
    println!("\n--- CigposeMetricDepthProvider (combined) ---");
    let mut provider = create_pose_provider(PoseProviderKind::CigposeMetricDepth, "models")
        .map_err(|e| format!("provider load: {e}"))?;
    println!("label: {}", provider.label());
    summary.push_str("[combined]\n");
    summary.push_str(&format!("label: {}\n", provider.label()));

    let t2 = std::time::Instant::now();
    let est = provider.estimate_pose(rgb.as_raw(), iw, ih, 0);
    let dt_combined = t2.elapsed();
    println!(
        "estimate_pose: {} 2D keypoints, {} skeleton joints, {:.1}ms",
        est.annotation.keypoints.len(),
        est.skeleton.joints.len(),
        dt_combined.as_secs_f32() * 1000.0
    );
    summary.push_str(&format!(
        "estimate_pose: {} 2D keypoints, {} skeleton joints, {:.1}ms\n",
        est.annotation.keypoints.len(),
        est.skeleton.joints.len(),
        dt_combined.as_secs_f32() * 1000.0
    ));

    let bones = bone_summary(&est.skeleton);
    println!("source skeleton:");
    summary.push_str("source skeleton:\n");
    for line in &bones {
        println!("  {line}");
        summary.push_str(&format!("  {line}\n"));
    }

    // YOLOX engagement is visible through annotation.bounding_box —
    // the provider only sets it when YOLOX was loaded *and* produced
    // a usable bbox.
    let yolox_line = match est.annotation.bounding_box {
        Some((x1, y1, x2, y2)) => format!(
            "YOLOX bbox (norm): [{:.2}, {:.2}] → [{:.2}, {:.2}]  (size {:.0}x{:.0}px)",
            x1, y1, x2, y2,
            (x2 - x1) * iw as f32,
            (y2 - y1) * ih as f32,
        ),
        None => "YOLOX bbox: none (whole-frame inference, or YOLOX off)".to_string(),
    };
    println!("{yolox_line}");
    summary.push_str(&yolox_line);
    summary.push('\n');

    // Face cascade visible through skeleton.face + expressions +
    // face_mesh_confidence. Each piece is independent: head pose can
    // be present without FaceMesh expressions, or vice versa.
    let face_line = match est.skeleton.face {
        Some(fp) => format!(
            "head pose: yaw={:>+5.1}° pitch={:>+5.1}° roll={:>+5.1}° conf={:.3}",
            fp.yaw.to_degrees(),
            fp.pitch.to_degrees(),
            fp.roll.to_degrees(),
            fp.confidence,
        ),
        None => "head pose: none (face landmarks below floor or origin missing)".to_string(),
    };
    println!("{face_line}");
    summary.push_str(&face_line);
    summary.push('\n');

    let mesh_line = format!(
        "FaceMesh expressions: {} (mesh_conf {})",
        est.skeleton.expressions.len(),
        est.skeleton
            .face_mesh_confidence
            .map(|c| format!("{:.3}", c))
            .unwrap_or_else(|| "n/a".to_string())
    );
    println!("{mesh_line}");
    summary.push_str(&mesh_line);
    summary.push('\n');
    if !est.skeleton.expressions.is_empty() {
        // Print the top-5 active blendshapes to confirm BlendshapeV2 is
        // actually contributing. Sort by absolute weight so neutral
        // baseline weights don't drown the signal.
        let mut top: Vec<&vulvatar_lib::tracking::SourceExpression> =
            est.skeleton.expressions.iter().collect();
        top.sort_by(|a, b| b.weight.abs().total_cmp(&a.weight.abs()));
        for ex in top.iter().take(5) {
            let line = format!("  {:<32} {:>6.3}", ex.name, ex.weight);
            println!("{line}");
            summary.push_str(&line);
            summary.push('\n');
        }
    }

    let summary_path = out_dir.join("summary.txt");
    std::fs::write(&summary_path, &summary)
        .map_err(|e| format!("write summary: {e}"))?;
    println!("wrote {}", summary_path.display());

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct DepthStats {
    valid: usize,
    total: usize,
    z_min: f32,
    z_median: f32,
    z_max: f32,
    z_centre: f32,
}

impl DepthStats {
    fn valid_pct(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            100.0 * self.valid as f32 / self.total as f32
        }
    }
}

fn depth_stats(frame: &MoGeFrame) -> DepthStats {
    let total = frame.points_m.len();
    let mut zs: Vec<f32> = frame
        .points_m
        .iter()
        .map(|p| p[2])
        .filter(|z| z.is_finite() && *z > 0.0)
        .collect();
    zs.sort_by(|a, b| a.total_cmp(b));
    let z_min = *zs.first().unwrap_or(&f32::NAN);
    let z_max = *zs.last().unwrap_or(&f32::NAN);
    let z_median = if zs.is_empty() {
        f32::NAN
    } else {
        zs[zs.len() / 2]
    };
    let cx = (MOGE_W / 2) as usize;
    let cy = (MOGE_H / 2) as usize;
    let centre_idx = cy * MOGE_W as usize + cx;
    let z_centre = frame
        .points_m
        .get(centre_idx)
        .map(|p| p[2])
        .unwrap_or(f32::NAN);
    DepthStats {
        valid: zs.len(),
        total,
        z_min,
        z_median,
        z_max,
        z_centre,
    }
}

fn bone_summary(sk: &vulvatar_lib::tracking::SourceSkeleton) -> Vec<String> {
    let mut lines = Vec::new();
    let bones: BTreeMap<_, _> = sk
        .joints
        .iter()
        .map(|(b, j)| (format!("{:?}", b), *j))
        .collect();
    let count = bones.len();
    lines.push(format!(
        "joints: {count}, fingertips: {}, overall_conf: {:.3}",
        sk.fingertips.len(),
        sk.overall_confidence
    ));
    // Highlight a few canonical bones the solver depends on.
    for name in [
        "Hips",
        "RightShoulder",
        "LeftShoulder",
        "RightUpperArm",
        "LeftUpperArm",
        "RightLowerArm",
        "LeftLowerArm",
        "RightUpperLeg",
        "LeftUpperLeg",
    ] {
        if let Some(j) = bones.get(name) {
            let p = j.position;
            lines.push(format!(
                "{:<14} pos=({:>+6.3}, {:>+6.3}, {:>+6.3})m  conf={:.3}",
                name, p[0], p[1], p[2], j.confidence
            ));
        } else {
            lines.push(format!("{:<14} (missing)", name));
        }
    }
    // Sanity: shoulder-to-shoulder distance, hip-to-shoulder distance.
    if let (Some(ls), Some(rs)) = (bones.get("LeftShoulder"), bones.get("RightShoulder")) {
        let d = dist(ls.position, rs.position);
        lines.push(format!("shoulder span:   {:.3} m", d));
    }
    if let (Some(rs), Some(re)) = (bones.get("RightShoulder"), bones.get("RightLowerArm")) {
        let d = dist(rs.position, re.position);
        lines.push(format!("R upper arm len: {:.3} m", d));
    }
    if let (Some(hips), Some(rs)) = (bones.get("Hips"), bones.get("RightShoulder")) {
        let d = dist(hips.position, rs.position);
        lines.push(format!("hip→R-shoulder:  {:.3} m", d));
    }
    lines
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn save_keypoint_overlay(
    src: &RgbaImage,
    joints: &[vulvatar_lib::tracking::cigpose::DecodedJoint2d],
    out: &Path,
) -> Result<(), String> {
    let (w, h) = (src.width(), src.height());
    let mut canvas: RgbaImage = src.clone();
    // Draw filled circles at every keypoint with score above the visibility floor.
    for (i, j) in joints.iter().enumerate() {
        if j.score < 0.05 {
            continue;
        }
        let x = (j.nx * w as f32).round() as i32;
        let y = (j.ny * h as f32).round() as i32;
        let radius = if i < 17 { 5 } else { 2 };
        let color = if i < 17 {
            // body: bright cyan
            Rgba([0, 255, 220, 255])
        } else if i < 23 {
            // foot: yellow
            Rgba([255, 220, 0, 255])
        } else if i < 91 {
            // face: magenta (small)
            Rgba([240, 0, 255, 255])
        } else if i < 112 {
            // left hand: green
            Rgba([0, 255, 80, 255])
        } else {
            // right hand: red
            Rgba([255, 60, 60, 255])
        };
        draw_disc(&mut canvas, x, y, radius, color);
    }
    canvas
        .save(out)
        .map_err(|e| format!("save overlay: {e}"))?;
    Ok(())
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

fn save_depth_vis(frame: &MoGeFrame, out: &Path) -> Result<(), String> {
    let w = frame.width;
    let h = frame.height;
    // Compute valid-Z range to span the colour ramp.
    let mut zs: Vec<f32> = frame
        .points_m
        .iter()
        .map(|p| p[2])
        .filter(|z| z.is_finite() && *z > 0.0)
        .collect();
    if zs.is_empty() {
        return Err("MoGe frame has no valid depth pixels".to_string());
    }
    zs.sort_by(|a, b| a.total_cmp(b));
    let zmin = zs[zs.len() / 100];
    let zmax = zs[zs.len() * 99 / 100];
    let mut img: RgbImage = ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let z = frame.points_m[idx][2];
            let pixel = if z.is_finite() && z > 0.0 {
                let t = ((z - zmin) / (zmax - zmin).max(1e-6)).clamp(0.0, 1.0);
                turbo_like(t)
            } else {
                Rgb([0, 0, 0])
            };
            img.put_pixel(x, y, pixel);
        }
    }
    img.save(out).map_err(|e| format!("save depth: {e}"))?;
    Ok(())
}

/// Cheap turbo-style ramp (3-stop interpolation). Good enough for a
/// quick depth visualisation; not the real Google Turbo LUT.
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
