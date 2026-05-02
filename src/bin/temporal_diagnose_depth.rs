//! Temporal-sequence diagnostic for the rtmw3d-with-depth provider.
//!
//! Constructs a single `Rtmw3dWithDepthProvider` (so the EMA + sticky
//! depth outbox carry across frames the same way they would in a live
//! webcam session) and feeds a sorted directory of images as a
//! synthetic frame stream — frame_index increments monotonically. The
//! per-frame output captures both the EMA-smoothed scale `c` and the
//! resulting elbow Z, so we can verify that:
//!
//! 1. `scale_c_ema` stabilises (low frame-to-frame |Δc|) under the EMA.
//! 2. Elbow Z (in the `SourceSkeleton`) tracks smoothly, i.e. the
//!    breath we measured in static-image bulk runs (stdev 1.59) is
//!    actually getting flattened by the EMA.
//!
//! Usage:
//!   temporal_diagnose_depth <dir> [--out file.jsonl]
//!
//! Best run on rotation-style sequences (validation_images/*_rotation/)
//! where consecutive frames represent a smooth pose progression — any
//! per-frame Z jitter beyond the angular delta is genuine noise.
//!
//! The bin uses `Rtmw3dWithDepthProvider` directly (not the boxed
//! trait) so it can read `scale_c_ema` after each `estimate_pose`.

use std::fs;
use std::path::{Path, PathBuf};

use vulvatar_lib::asset::HumanoidBone;
use vulvatar_lib::tracking::provider::PoseProvider;
use vulvatar_lib::tracking::rtmw3d_with_depth::Rtmw3dWithDepthProvider;
use vulvatar_lib::tracking::source_skeleton::SourceSkeleton;

fn main() -> Result<(), String> {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let target = args
        .next()
        .ok_or_else(|| "usage: temporal_diagnose_depth <dir> [--out file.jsonl]".to_string())?;
    let target_path = PathBuf::from(&target);

    let mut out_path: Option<PathBuf> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out_path = args.next().map(PathBuf::from),
            other => return Err(format!("unknown arg '{other}'")),
        }
    }
    if let Some(ref p) = out_path {
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("create_dir_all '{}': {e}", parent.display()))?;
            }
        }
    }

    let images = collect_images(&target_path)?;
    eprintln!("temporal_diagnose_depth: {} frame(s)", images.len());

    let mut provider = Rtmw3dWithDepthProvider::from_models_dir("models")
        .map_err(|e| format!("provider load: {e}"))?;
    let _ = provider.take_load_warnings();

    let mut jsonl = String::new();
    let mut prev_scale: Option<f32> = None;
    let mut prev_l_elbow_z: Option<f32> = None;
    let mut prev_r_elbow_z: Option<f32> = None;
    let mut sum_abs_dscale = 0.0_f32;
    let mut sum_abs_dle = 0.0_f32;
    let mut sum_abs_dre = 0.0_f32;
    let mut n_dscale = 0_u32;
    let mut n_dle = 0_u32;
    let mut n_dre = 0_u32;

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

        // Single call per frame, monotonic frame_index. Mirrors what
        // the live tracking worker does at 30 fps.
        let est = provider.estimate_pose(rgb.as_raw(), w, h, idx as u64);
        let scale = provider.scale_c_ema();
        let row = build_row(image_path, idx as u64, scale, &est.skeleton);

        if let (Some(prev), Some(now)) = (prev_scale, scale) {
            sum_abs_dscale += (now - prev).abs();
            n_dscale += 1;
        }
        if let (Some(prev), Some(now)) = (prev_l_elbow_z, row.l_elbow_z) {
            sum_abs_dle += (now - prev).abs();
            n_dle += 1;
        }
        if let (Some(prev), Some(now)) = (prev_r_elbow_z, row.r_elbow_z) {
            sum_abs_dre += (now - prev).abs();
            n_dre += 1;
        }
        prev_scale = scale;
        prev_l_elbow_z = row.l_elbow_z;
        prev_r_elbow_z = row.r_elbow_z;

        eprintln!(
            "[{:>3}] c_ema={:>6}  L_sh.z={:>6}  L_el.z={:>6}  R_sh.z={:>6}  R_el.z={:>6}  {}",
            idx,
            opt(scale),
            opt(row.l_shoulder_z),
            opt(row.l_elbow_z),
            opt(row.r_shoulder_z),
            opt(row.r_elbow_z),
            image_path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
        );
        jsonl.push_str(&row.to_jsonl());
        jsonl.push('\n');
    }

    let avg_dscale = if n_dscale > 0 {
        sum_abs_dscale / n_dscale as f32
    } else {
        f32::NAN
    };
    let avg_dle = if n_dle > 0 {
        sum_abs_dle / n_dle as f32
    } else {
        f32::NAN
    };
    let avg_dre = if n_dre > 0 {
        sum_abs_dre / n_dre as f32
    } else {
        f32::NAN
    };
    eprintln!(
        "\n=== summary ({} frames) ===\n  mean |Δscale_c|       : {:.4}\n  mean |ΔL_elbow.z| (m): {:.4}\n  mean |ΔR_elbow.z| (m): {:.4}\n  (lower = smoother sequence — depth EMA flattening jitter)",
        images.len(),
        avg_dscale,
        avg_dle,
        avg_dre,
    );

    if let Some(p) = out_path {
        fs::write(&p, &jsonl).map_err(|e| format!("write {}: {e}", p.display()))?;
        eprintln!("wrote {}", p.display());
    } else {
        print!("{jsonl}");
    }
    Ok(())
}

fn opt(v: Option<f32>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{:.2}", x),
        _ => "  -- ".to_string(),
    }
}

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

struct Row {
    image: String,
    frame: u64,
    scale_c_ema: Option<f32>,
    l_shoulder_z: Option<f32>,
    r_shoulder_z: Option<f32>,
    l_elbow_z: Option<f32>,
    r_elbow_z: Option<f32>,
    l_hand_z: Option<f32>,
    r_hand_z: Option<f32>,
    overall_confidence: f32,
}

impl Row {
    fn to_jsonl(&self) -> String {
        format!(
            "{{\"image\":\"{}\",\"frame\":{},\"scale_c_ema\":{},\"l_shoulder_z\":{},\"r_shoulder_z\":{},\"l_elbow_z\":{},\"r_elbow_z\":{},\"l_hand_z\":{},\"r_hand_z\":{},\"overall_confidence\":{}}}",
            self.image.replace('\\', "/").replace('"', "\\\""),
            self.frame,
            f(self.scale_c_ema),
            f(self.l_shoulder_z),
            f(self.r_shoulder_z),
            f(self.l_elbow_z),
            f(self.r_elbow_z),
            f(self.l_hand_z),
            f(self.r_hand_z),
            self.overall_confidence,
        )
    }
}

fn build_row(
    image_path: &Path,
    frame: u64,
    scale_c_ema: Option<f32>,
    sk: &SourceSkeleton,
) -> Row {
    let z = |bone: HumanoidBone| -> Option<f32> { sk.joints.get(&bone).map(|j| j.position[2]) };
    Row {
        image: image_path
            .strip_prefix(std::env::current_dir().unwrap_or_default())
            .unwrap_or(image_path)
            .display()
            .to_string(),
        frame,
        scale_c_ema,
        l_shoulder_z: z(HumanoidBone::LeftShoulder),
        r_shoulder_z: z(HumanoidBone::RightShoulder),
        l_elbow_z: z(HumanoidBone::LeftLowerArm),
        r_elbow_z: z(HumanoidBone::RightLowerArm),
        l_hand_z: z(HumanoidBone::LeftHand),
        r_hand_z: z(HumanoidBone::RightHand),
        overall_confidence: sk.overall_confidence,
    }
}

fn f(v: Option<f32>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{:.4}", x),
        _ => "null".to_string(),
    }
}
