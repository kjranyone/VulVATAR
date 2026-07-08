//! RTMW3D raw 2D keypoint detection probe. Runs the production provider
//! (RTMW3D, depth OFF) over an extracted frame directory, overlays the raw
//! COCO-Wholebody 2D keypoints on each frame (green=score>0.5, yellow>0.3,
//! red=low), and reports detection confidence. Purpose: measure 2D
//! detection quality at an unusual desk-looking-UP viewpoint — the
//! prerequisite risk for the D435 depth rebuild (depth-lift is only as
//! good as the 2D keypoints it samples at).
//!
//!   cargo run --bin diagnose_rtmw3d_2d -- <frames_dir> <out_dir>

use std::path::PathBuf;

// COCO-17 body indices within the 133-keypoint wholebody vector.
const NOSE: usize = 0;
const L_SH: usize = 5;
const R_SH: usize = 6;
const L_EL: usize = 7;
const R_EL: usize = 8;
const L_WR: usize = 9;
const R_WR: usize = 10;

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

fn main() -> Result<(), String> {
    env_logger::init();
    let dir = std::env::args()
        .nth(1)
        .ok_or("usage: diagnose_rtmw3d_2d <frames_dir> <out_dir>")?;
    let out_dir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "diagnostics/rtmw3d_2d".to_string());
    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;

    let config = vulvatar_lib::tracking::provider::TrackingPipelineConfig::default();
    let mut provider = vulvatar_lib::tracking::provider::create_pose_provider("models", config)?;
    let _ = provider.take_load_warnings();

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("read_dir {dir}: {e}"))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "png" || x == "jpg").unwrap_or(false))
        .collect();
    files.sort();
    eprintln!("{} frames from {dir}", files.len());

    let named = [
        ("nose", NOSE), ("Lsh", L_SH), ("Rsh", R_SH),
        ("Lel", L_EL), ("Rel", R_EL), ("Lwr", L_WR), ("Rwr", R_WR),
    ];
    let mut sum_overall = 0.0f32;
    let mut nframes = 0u32;
    let (mut wr_hits, mut wr_total) = (0u32, 0u32);
    let (mut body_hits, mut body_total) = (0u32, 0u32);

    for (i, f) in files.iter().enumerate() {
        let mut img = image::open(f).map_err(|e| format!("open {}: {e}", f.display()))?.to_rgb8();
        let (w, h) = (img.width(), img.height());
        let est = provider.estimate_pose(img.as_raw(), w, h, i as u64);
        let kp = &est.annotation.keypoints;

        for (idx, &(nx, ny, score)) in kp.iter().enumerate() {
            let is_body = idx <= 16;
            let is_hand = (91..=132).contains(&idx);
            if !(is_body || is_hand) {
                continue;
            }
            let x = (nx * w as f32) as i32;
            let y = (ny * h as f32) as i32;
            let col = if score > 0.5 {
                [0, 230, 0]
            } else if score > 0.3 {
                [235, 200, 0]
            } else {
                [235, 0, 0]
            };
            draw_dot(&mut img, x, y, if is_body { 6 } else { 3 }, col);
        }
        img.save(PathBuf::from(&out_dir).join(format!("f{i:04}.png")))
            .map_err(|e| format!("save: {e}"))?;

        sum_overall += est.skeleton.overall_confidence;
        nframes += 1;
        let get = |idx: usize| kp.get(idx).copied().unwrap_or((0.0, 0.0, 0.0));
        let mut parts = vec![format!("f{i:03} overall={:.2}", est.skeleton.overall_confidence)];
        for (nm, idx) in named {
            parts.push(format!("{nm}={:.2}", get(idx).2));
        }
        println!("{}", parts.join(" "));

        for idx in [L_WR, R_WR] {
            wr_total += 1;
            if get(idx).2 > 0.3 {
                wr_hits += 1;
            }
        }
        for idx in 0..=16usize {
            body_total += 1;
            if get(idx).2 > 0.3 {
                body_hits += 1;
            }
        }
    }
    eprintln!(
        "SUMMARY {dir}: frames={nframes} mean_overall={:.2} wrist_detect(score>0.3)={:.0}% body_detect={:.0}%",
        sum_overall / nframes.max(1) as f32,
        100.0 * wr_hits as f32 / wr_total.max(1) as f32,
        100.0 * body_hits as f32 / body_total.max(1) as f32,
    );
    Ok(())
}
