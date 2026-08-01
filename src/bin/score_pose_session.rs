//! Score a recorded session against objective, estimator-agnostic metrics.
//!
//! This is the measuring stick for replacing the pose estimator. It replays a
//! session recorded by `tracking::sequence_recorder` through the production
//! provider path (`set_external_depth` → `estimate_pose`, continuous temporal
//! state, real capture dt) and reduces the result to numbers that say whether
//! one estimator is better than another WITHOUT ground truth and WITHOUT
//! reference to how either one works internally:
//!
//! 1. **Depth fidelity** — each joint's own metric depth against the RAW
//!    sensor depth under its detected 2-D keypoint. The sensor is the
//!    reference; an estimator that "smooths" a joint off the surface it was
//!    measured on pays for it here.
//! 2. **Anatomical consistency** — a bone does not change length. Per-bone
//!    coefficient of variation over the session; the current pipeline has no
//!    skeleton, so its bones breathe.
//! 3. **Continuity** — per-frame joint displacement (median / p95 / max).
//!    This is where the teleports live.
//! 4. **Availability** — fraction of frames each joint exists, and how often
//!    that existence toggles. A joint that blinks at 15 Hz is unusable
//!    downstream no matter how accurate it is while present.
//!
//! Every metric is computed from the published `SourceSkeleton` plus the
//! recorded raw depth, so a future estimator that emits the same type is
//! scored by exactly this binary, unchanged.
//!
//!   cargo run --features realsense --bin score_pose_session -- <session_dir> [out.jsonl]
//!
//! Env: `VULVATAR_SCORE_LIMIT` caps the number of frames replayed.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

use vulvatar_lib::asset::HumanoidBone;
use vulvatar_lib::tracking::provider::create_pose_provider;
use vulvatar_lib::tracking::skeleton_from_depth::MetricDepthFrame;
use vulvatar_lib::tracking::{CameraIntrinsics, SourceSkeleton};

/// Joints scored, with the COCO-WholeBody keypoint whose pixel their depth is
/// checked against. Torso + arms: the chain the live artefacts live in.
const SCORED: &[(HumanoidBone, usize, &str)] = &[
    (HumanoidBone::Head, 0, "Head"),
    (HumanoidBone::LeftUpperArm, 6, "LUp"),
    (HumanoidBone::RightUpperArm, 5, "RUp"),
    (HumanoidBone::LeftLowerArm, 8, "LLo"),
    (HumanoidBone::RightLowerArm, 7, "RLo"),
    (HumanoidBone::LeftHand, 10, "LHa"),
    (HumanoidBone::RightHand, 9, "RHa"),
];

/// Bones whose length must not change: (name, parent, child).
const BONES: &[(&str, HumanoidBone, HumanoidBone)] = &[
    ("L upper arm", HumanoidBone::LeftUpperArm, HumanoidBone::LeftLowerArm),
    ("R upper arm", HumanoidBone::RightUpperArm, HumanoidBone::RightLowerArm),
    ("L forearm", HumanoidBone::LeftLowerArm, HumanoidBone::LeftHand),
    ("R forearm", HumanoidBone::RightLowerArm, HumanoidBone::RightHand),
    ("shoulders", HumanoidBone::LeftUpperArm, HumanoidBone::RightUpperArm),
];

#[derive(Clone, Copy)]
struct Meta {
    frame: u64,
    t_ms: f64,
    w: u32,
    h: u32,
    depth_units: f32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    dropped_before: u64,
}

fn parse_meta(line: &str) -> Option<Meta> {
    let num = |key: &str| -> Option<f64> {
        let at = line.find(&format!("\"{key}\":"))? + key.len() + 3;
        let rest = &line[at..];
        let end = rest.find(|c: char| c != '-' && c != '.' && !c.is_ascii_digit())?;
        rest[..end].parse().ok()
    };
    Some(Meta {
        frame: num("frame")? as u64,
        t_ms: num("t_ms")?,
        w: num("w")? as u32,
        h: num("h")? as u32,
        depth_units: num("depth_units")? as f32,
        fx: num("fx")? as f32,
        fy: num("fy")? as f32,
        cx: num("cx")? as f32,
        cy: num("cy")? as f32,
        dropped_before: num("dropped_before").unwrap_or(0.0) as u64,
    })
}

fn parse_npy_u16(bytes: &[u8]) -> Result<(usize, usize, Vec<u16>), String> {
    if bytes.len() < 12 || &bytes[0..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let start = 10 + hlen;
    let header = std::str::from_utf8(&bytes[10..start]).map_err(|e| e.to_string())?;
    let shape = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next())
        .ok_or("no shape")?;
    let dims: Vec<usize> = shape.split(',').filter_map(|t| t.trim().parse().ok()).collect();
    let (rows, cols) = match dims.as_slice() {
        [r, c, ..] => (*r, *c),
        _ => return Err("not 2D".into()),
    };
    let data = &bytes[start..];
    if data.len() < rows * cols * 2 {
        return Err("short".into());
    }
    Ok((
        rows,
        cols,
        (0..rows * cols)
            .map(|i| u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]))
            .collect(),
    ))
}

fn deproject(m: &Meta, rows: usize, cols: usize, raw: &[u16]) -> MetricDepthFrame {
    let intr = CameraIntrinsics {
        fx: m.fx,
        fy: m.fy,
        cx: m.cx,
        cy: m.cy,
        width: cols as u32,
        height: rows as u32,
    };
    let mut points = vec![[f32::NAN; 3]; rows * cols];
    for v in 0..rows {
        for u in 0..cols {
            let d = raw[v * cols + u];
            if d == 0 {
                continue;
            }
            let z = d as f32 * m.depth_units;
            points[v * cols + u] = [
                (u as f32 - intr.cx) * z / intr.fx,
                (v as f32 - intr.cy) * z / intr.fy,
                z,
            ];
        }
    }
    MetricDepthFrame {
        width: cols as u32,
        height: rows as u32,
        points_m: points,
        intrinsics: Some(intr),
        crop: None,
        timestamp_ms: Some(m.t_ms),
    }
}

fn median(v: &mut Vec<f32>) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn pct(v: &mut Vec<f32>, p: f32) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() as f32 - 1.0) * p) as usize]
}

fn main() -> Result<(), String> {
    env_logger::init();
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: score_pose_session <session_dir> [out.jsonl]")?,
    );
    let out_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| dir.join("score.jsonl").to_string_lossy().into_owned());
    let limit: usize = std::env::var("VULVATAR_SCORE_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let meta_txt = std::fs::read_to_string(dir.join("meta.jsonl"))
        .map_err(|e| format!("meta.jsonl: {e}"))?;
    let metas: Vec<Meta> = meta_txt.lines().filter_map(parse_meta).take(limit).collect();
    if metas.is_empty() {
        return Err("no frames in meta.jsonl".into());
    }
    eprintln!("{} frames from {}", metas.len(), dir.display());

    let mut provider = create_pose_provider("models", Default::default())?;
    let _ = provider.take_load_warnings();

    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&out_path).map_err(|e| format!("{out_path}: {e}"))?,
    );

    // Accumulators.
    let mut depth_res: HashMap<&str, Vec<f32>> = HashMap::new();
    let mut bone_len: HashMap<&str, Vec<f32>> = HashMap::new();
    let mut step: HashMap<&str, Vec<f32>> = HashMap::new();
    let mut present: HashMap<&str, u32> = HashMap::new();
    let mut toggles: HashMap<&str, u32> = HashMap::new();
    let mut prev_pos: HashMap<&str, [f32; 3]> = HashMap::new();
    let mut prev_present: HashMap<&str, bool> = HashMap::new();
    let mut dropped_total = 0u64;
    let mut span_s = 0.0f64;

    let file_for = |m: &Meta, suffix: &str| -> PathBuf {
        dir.join(format!("f{:05}{suffix}", m.frame))
    };

    let t0 = metas[0].t_ms;
    for (i, m) in metas.iter().enumerate() {
        dropped_total += m.dropped_before;
        span_s = (m.t_ms - t0) / 1000.0;
        let color = file_for(m, "_color.png");
        let depth = file_for(m, "_depth_mm.npy");
        let img = match image::open(&color) {
            Ok(im) => im.to_rgb8(),
            Err(e) => {
                eprintln!("skip {}: {e}", color.display());
                continue;
            }
        };
        let (rows, cols, raw) = match std::fs::read(&depth)
            .map_err(|e| e.to_string())
            .and_then(|b| parse_npy_u16(&b))
        {
            Ok(v) => v,
            Err(e) => {
                eprintln!("skip {}: {e}", depth.display());
                continue;
            }
        };

        provider.set_external_depth(deproject(m, rows, cols, &raw));
        let mut est = provider.estimate_pose(img.as_raw(), m.w, m.h, m.frame);
        est.skeleton.capture_timestamp_ms = Some(m.t_ms);
        let sk: &SourceSkeleton = &est.skeleton;

        // 1. Depth fidelity vs the raw sensor pixel under the keypoint.
        for (bone, kp_idx, name) in SCORED {
            let here = sk.joints.get(bone);
            let was = prev_present.insert(name, here.is_some()).unwrap_or(false);
            if here.is_some() != was {
                *toggles.entry(name).or_default() += 1;
            }
            let Some(j) = here else { continue };
            *present.entry(name).or_default() += 1;

            // 3. Continuity.
            if let Some(p) = prev_pos.insert(name, j.position) {
                let d = ((j.position[0] - p[0]).powi(2)
                    + (j.position[1] - p[1]).powi(2)
                    + (j.position[2] - p[2]).powi(2))
                .sqrt();
                step.entry(name).or_default().push(d);
            }

            let (Some(depth_m), Some(&(nx, ny, score))) =
                (j.metric_depth_m, est.annotation.keypoints.get(*kp_idx))
            else {
                continue;
            };
            if score < 0.3 || !(0.0..=1.0).contains(&nx) || !(0.0..=1.0).contains(&ny) {
                continue;
            }
            let u = (nx * cols as f32).round() as isize;
            let v = (ny * rows as f32).round() as isize;
            if u < 0 || v < 0 || u >= cols as isize || v >= rows as isize {
                continue;
            }
            let d = raw[v as usize * cols + u as usize];
            if d == 0 {
                continue;
            }
            let sensor_m = d as f32 * m.depth_units;
            depth_res
                .entry(name)
                .or_default()
                .push((depth_m - sensor_m).abs());
        }

        // 2. Anatomical consistency.
        for (name, a, b) in BONES {
            if let (Some(pa), Some(pb)) = (sk.joints.get(a), sk.joints.get(b)) {
                let d = ((pa.position[0] - pb.position[0]).powi(2)
                    + (pa.position[1] - pb.position[1]).powi(2)
                    + (pa.position[2] - pb.position[2]).powi(2))
                .sqrt();
                bone_len.entry(name).or_default().push(d);
            }
        }

        let line = format!(
            "{{\"i\":{},\"frame\":{},\"t_ms\":{},\"overall\":{}}}\n",
            i, m.frame, m.t_ms, sk.overall_confidence
        );
        let _ = out.write_all(line.as_bytes());
    }
    let _ = out.flush();

    let frames = metas.len() as f32;
    println!("\n=== session: {} ===", dir.display());
    println!(
        "frames {} over {:.1}s (recorder dropped {})",
        metas.len(),
        span_s,
        dropped_total
    );

    println!("\n-- depth fidelity: |joint depth - raw sensor depth under its keypoint| (m)");
    println!("   joint  n      median     p95");
    for (_, _, name) in SCORED {
        if let Some(v) = depth_res.get_mut(name) {
            let n = v.len();
            println!("   {name:5}  {n:5}  {:8.4}  {:8.4}", median(v), pct(v, 0.95));
        }
    }

    println!("\n-- anatomical consistency: bone length (source units), CV = sd/mean");
    for (name, _, _) in BONES {
        if let Some(v) = bone_len.get_mut(name) {
            let n = v.len() as f32;
            let mean = v.iter().sum::<f32>() / n;
            let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
            println!(
                "   {name:12} n={:5}  mean {:.3}  sd {:.3}  CV {:5.1}%",
                v.len(),
                mean,
                sd,
                100.0 * sd / mean.max(1e-6)
            );
        }
    }

    println!("\n-- continuity: per-frame joint displacement (source units)");
    println!("   joint  median      p95       max");
    for (_, _, name) in SCORED {
        if let Some(v) = step.get_mut(name) {
            println!(
                "   {name:5}  {:8.4} {:8.4} {:8.4}",
                median(v),
                pct(v, 0.95),
                pct(v, 1.0)
            );
        }
    }

    println!("\n-- availability");
    println!("   joint  present   toggles/s");
    for (_, _, name) in SCORED {
        let p = *present.get(name).unwrap_or(&0) as f32;
        let t = *toggles.get(name).unwrap_or(&0) as f32;
        println!(
            "   {name:5}  {:5.1}%   {:6.2}",
            100.0 * p / frames,
            t / span_s.max(1e-3) as f32
        );
    }
    eprintln!("\nper-frame log: {out_path}");
    Ok(())
}
