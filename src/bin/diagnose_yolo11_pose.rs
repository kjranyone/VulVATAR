//! YOLO11-pose detector benchmark: per-frame wall time (DirectML /
//! optional CPU EP) over recorded session frames, plus per-frame person
//! detection counts.
//!
//! The letterbox + decode mirrors Ultralytics' ONNX conventions
//! (114-grey padding, bilinear resize, `(1, 56, A)` output with box
//! cx/cy/w/h in input pixels at channel 0..4, person score at channel 4,
//! and 17×(x, y, conf) keypoints from channel 5) — the same contract the
//! production detector consumes.
//!
//! Output: `diagnostics/yolo11_bench/` — `summary.md` + one CSV per
//! model. The app may run concurrently; timing includes that contention.
//!
//! Usage:
//!   cargo run --release --no-default-features --features inference,inference-gpu //!     --bin diagnose_yolo11_pose -- [session_dir ...] [--model <path>]... //!     [--frames N] [--cpu] [--out <dir>]

use std::path::{Path, PathBuf};
use std::time::Instant;

use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;

const BOX_SCORE_MIN: f32 = 0.25;
const NMS_IOU: f32 = 0.65;

struct Args {
    sessions: Vec<PathBuf>,
    models: Vec<PathBuf>,
    frames_per_session: usize,
    also_cpu: bool,
    out_dir: PathBuf,
}

fn parse_args() -> Args {
    let mut sessions = Vec::new();
    let mut models = Vec::new();
    let mut frames = 150usize;
    let mut also_cpu = false;
    let mut out_dir = PathBuf::from("diagnostics/yolo11_bench");
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => models.push(PathBuf::from(it.next().expect("--model <path>"))),
            "--frames" => frames = it.next().expect("--frames <n>").parse().unwrap(),
            "--cpu" => also_cpu = true,
            "--out" => out_dir = PathBuf::from(it.next().expect("--out <dir>")),
            other => {
                if other.starts_with("--") {
                    panic!("unknown flag {other}");
                }
                sessions.push(PathBuf::from(other));
            }
        }
    }
    if sessions.is_empty() {
        let root = Path::new("diagnostics/sessions");
        let mut found: Vec<PathBuf> = std::fs::read_dir(root)
            .unwrap_or_else(|e| panic!("read_dir {}: {e}", root.display()))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("s17"))
                    .unwrap_or(false)
            })
            .collect();
        found.sort();
        sessions = found;
    }
    if models.is_empty() {
        models = [
            "models/yolo11n-pose_640.onnx",
            "models/yolo11n-pose_480.onnx",
            "models/yolo11s-pose_640.onnx",
        ]
        .iter()
        .map(PathBuf::from)
        .collect();
    }
    sessions.sort();
    Args {
        sessions,
        models,
        frames_per_session: frames,
        also_cpu,
        out_dir,
    }
}

/// One frame's decoded COCO-17 output: `[x_px, y_px, conf]`.
type FrameKps = Option<[[f32; 3]; 17]>;

struct FrameRun {
    ms_total: f32,
    ms_run: f32,
    persons: usize,
    kps: FrameKps,
}

fn main() -> Result<(), String> {
    env_logger::init();
    let args = parse_args();
    std::fs::create_dir_all(&args.out_dir).map_err(|e| format!("create out dir: {e}"))?;

    // Collect sampled frame paths per session (evenly strided).
    let mut sampled: Vec<(String, Vec<PathBuf>)> = Vec::new();
    for session in &args.sessions {
        let mut files: Vec<PathBuf> = std::fs::read_dir(session)
            .map_err(|e| format!("read_dir {}: {e}", session.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.ends_with("_color.png"))
                    .unwrap_or(false)
            })
            .collect();
        files.sort();
        if files.is_empty() {
            eprintln!("skip {}: no *_color.png", session.display());
            continue;
        }
        let stride = files.len().div_ceil(args.frames_per_session).max(1);
        let picked: Vec<PathBuf> = files
            .into_iter()
            .step_by(stride)
            // Session recorders can leave a truncated final frame behind;
            // validate decodability here so every downstream consumer (each
            // model + the RTMW3D reference) sees the identical frame list
            // and stays index-aligned.
            .filter(|p| match load_rgb(p) {
                Ok(_) => true,
                Err(e) => {
                    eprintln!("  skipping undecodable {}: {e}", p.display());
                    false
                }
            })
            .collect();
        eprintln!(
            "{}: {} frames total, sampling {}",
            session.display(),
            std::fs::read_dir(session).unwrap().count(),
            picked.len()
        );
        sampled.push((session.display().to_string(), picked));
    }
    let total_frames: usize = sampled.iter().map(|(_, f)| f.len()).sum();
    eprintln!("benchmark: {total_frames} frames across {} sessions", sampled.len());

    // ---- YOLO11-pose models, one session alive at a time.
    let mut per_model: Vec<(String, Vec<FrameRun>)> = Vec::new();
    let mut cpu_timings: Vec<(String, Vec<f32>)> = Vec::new();
    for model_path in &args.models {
        let name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string();
        // ImgSz is encoded in the export filename (`_640.onnx`).
        let size: u32 = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.rsplit('_').next())
            .and_then(|s| s.parse().ok())
            .unwrap_or(640);
        eprintln!("== {name} (imgsz {size}, DirectML)");
        let session = build_dml_session(model_path).map_err(|e| format!("{name}: {e}"))?;
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "images".to_string());
        let output_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .unwrap_or_else(|| "output0".to_string());
        let mut runner = YoloRunner {
            session,
            input_name,
            output_name,
            size,
        };

        let mut runs = Vec::with_capacity(total_frames);
        let mut warmed = 0usize;
        for (session_name, frames) in &sampled {
            for path in frames {
                let (rgb, w, h) = load_rgb(path)?;
                // Warm-up: first two frames run before the stopwatch so
                // DML kernel compilation / allocator warm-in don't pollute.
                if warmed < 2 {
                    warmed += 1;
                    let _ = runner.run(&rgb, w, h);
                    continue;
                }
                let t0 = Instant::now();
                let (kps, persons, ms_run) = runner.run(&rgb, w, h);
                runs.push(FrameRun {
                    ms_total: t0.elapsed().as_secs_f32() * 1000.0,
                    ms_run,
                    persons,
                    kps,
                });
                let _ = session_name;
            }
        }
        let label = name.clone();
        per_model.push((label, runs));
        drop(runner);

        if args.also_cpu {
            eprintln!("== {name} (CPU EP)");
            let session = build_cpu_session(model_path)?;
            let input_name = session
                .inputs()
                .first()
                .map(|i| i.name().to_string())
                .unwrap_or_else(|| "images".to_string());
            let output_name = session
                .outputs()
                .first()
                .map(|o| o.name().to_string())
                .unwrap_or_else(|| "output0".to_string());
            let mut runner = YoloRunner {
                session,
                input_name,
                output_name,
                size,
            };
            let mut times = Vec::with_capacity(30);
            for (_, frames) in sampled.iter().take(1) {
                for path in frames.iter().take(30) {
                    let (rgb, w, h) = load_rgb(path)?;
                    let t0 = Instant::now();
                    let _ = runner.run(&rgb, w, h);
                    times.push(t0.elapsed().as_secs_f32() * 1000.0);
                }
            }
            cpu_timings.push((format!("{name}"), times));
            drop(runner);
        }
    }

    // ---- Report.
    write_reports(&args, &sampled, &per_model, &cpu_timings, total_frames)
}

struct YoloRunner {
    session: Session,
    input_name: String,
    output_name: String,
    size: u32,
}

impl YoloRunner {
    /// Letterbox → run → decode. Returns (keypoints in frame pixels, person
    /// count after NMS, run-only ms).
    fn run(&mut self, rgb: &[u8], w: u32, h: u32) -> (FrameKps, usize, f32) {
        let r = (self.size as f32 / w as f32).min(self.size as f32 / h as f32);
        let pad_x = (self.size as f32 - w as f32 * r) / 2.0;
        let pad_y = (self.size as f32 - h as f32 * r) / 2.0;
        let tensor = letterbox(rgb, w, h, self.size);
        let input = TensorRef::from_array_view(&tensor).expect("tensor view");
        let t0 = Instant::now();
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
            .expect("yolo11 run");
        let ms_run = t0.elapsed().as_secs_f32() * 1000.0;
        let out = outputs.get(self.output_name.as_str()).expect("output");
        let (_, data) = out.try_extract_tensor::<f32>().expect("f32 tensor");
        assert!(
            data.len() % 56 == 0,
            "unexpected YOLO11-pose output size {} (expected 56×A)",
            data.len()
        );
        let anchors = data.len() / 56;

        // Channel-major (1, 56, A): channel c, anchor a → data[c * A + a].
        let at = |c: usize, a: usize| data[c * anchors + a];
        let mut candidates: Vec<(usize, f32)> = (0..anchors)
            .map(|a| (a, at(4, a)))
            .filter(|&(_, s)| s >= BOX_SCORE_MIN)
            .collect();
        candidates.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));

        // Greedy NMS on the box channels (cx, cy, w, h).
        let mut kept: Vec<usize> = Vec::new();
        for (a, _) in &candidates {
            let box_a = [
                at(0, *a) - at(2, *a) / 2.0,
                at(1, *a) - at(3, *a) / 2.0,
                at(0, *a) + at(2, *a) / 2.0,
                at(1, *a) + at(3, *a) / 2.0,
            ];
            if kept.iter().all(|k| {
                let bk = [
                    at(0, *k) - at(2, *k) / 2.0,
                    at(1, *k) - at(3, *k) / 2.0,
                    at(0, *k) + at(2, *k) / 2.0,
                    at(1, *k) + at(3, *k) / 2.0,
                ];
                iou(&box_a, &bk) <= NMS_IOU
            }) {
                kept.push(*a);
            }
        }

        let persons = kept.len();
        let kps = kept.first().map(|&best| {
            let mut arr = [[0f32; 3]; 17];
            for (k, slot) in arr.iter_mut().enumerate() {
                let base = 5 + k * 3;
                // Model-input pixels → frame pixels (undo letterbox).
                slot[0] = (at(base, best) - pad_x) / r;
                slot[1] = (at(base + 1, best) - pad_y) / r;
                slot[2] = at(base + 2, best);
            }
            arr
        });
        (kps, persons, ms_run)
    }
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

/// Ultralytics letterbox: bilinear resize so the long side fits `s`, pad
/// the short side with grey (114/255), NCHW RGB 0..1.
fn letterbox(rgb: &[u8], w: u32, h: u32, s: u32) -> Array4<f32> {
    let r = (s as f32 / w as f32).min(s as f32 / h as f32);
    let nw = (w as f32 * r).round() as i64;
    let nh = (h as f32 * r).round() as i64;
    let pad_x = ((s as i64 - nw) / 2).max(0);
    let pad_y = ((s as i64 - nh) / 2).max(0);
    let mut arr = Array4::<f32>::from_elem((1, 3, s as usize, s as usize), 114.0 / 255.0);
    for y in 0..nh {
        let sy = ((y as f32 + 0.5) / r - 0.5).max(0.0);
        let y0 = sy.floor() as u32;
        let y1 = (y0 + 1).min(h - 1);
        let fy = (sy - y0 as f32).clamp(0.0, 1.0);
        for x in 0..nw {
            let sx = ((x as f32 + 0.5) / r - 0.5).max(0.0);
            let x0 = sx.floor() as u32;
            let x1 = (x0 + 1).min(w - 1);
            let fx = (sx - x0 as f32).clamp(0.0, 1.0);
            for c in 0..3usize {
                let p00 = rgb[(y0 as usize * w as usize + x0 as usize) * 3 + c] as f32;
                let p01 = rgb[(y0 as usize * w as usize + x1 as usize) * 3 + c] as f32;
                let p10 = rgb[(y1 as usize * w as usize + x0 as usize) * 3 + c] as f32;
                let p11 = rgb[(y1 as usize * w as usize + x1 as usize) * 3 + c] as f32;
                let top = p00 * (1.0 - fx) + p01 * fx;
                let bot = p10 * (1.0 - fx) + p11 * fx;
                let v = (top * (1.0 - fy) + bot * fy) / 255.0;
                let dx = (x + pad_x) as usize;
                let dy = (y + pad_y) as usize;
                arr[[0, c, dy, dx]] = v;
            }
        }
    }
    arr
}

fn load_rgb(path: &Path) -> Result<(Vec<u8>, u32, u32), String> {
    let img = image::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Ok((rgb.into_raw(), w, h))
}

#[cfg(feature = "inference-gpu")]
fn build_dml_session(path: &Path) -> Result<Session, String> {
    let builder = Session::builder().map_err(|e| format!("builder: {e}"))?;
    let builder = builder
        .with_execution_providers([ort::ep::DirectML::default().build()])
        .map_err(|e| format!("with_execution_providers: {e}"))?;
    // DirectML contract: sequential execution, no memory patterns (same
    // as the production RTMW3D session).
    let builder = builder
        .with_parallel_execution(false)
        .map_err(|e| format!("with_parallel_execution: {e}"))?;
    let builder = builder
        .with_memory_pattern(false)
        .map_err(|e| format!("with_memory_pattern: {e}"))?;
    let builder = builder
        .with_intra_threads(4)
        .map_err(|e| format!("with_intra_threads: {e}"))?;
    let mut builder = builder;
    builder
        .commit_from_file(path)
        .map_err(|e| format!("commit_from_file: {e}"))
}

#[cfg(not(feature = "inference-gpu"))]
fn build_dml_session(_path: &Path) -> Result<Session, String> {
    Err("built without inference-gpu; DirectML unavailable".to_string())
}

fn build_cpu_session(path: &Path) -> Result<Session, String> {
    Session::builder()
        .map_err(|e| format!("builder: {e}"))?
        .with_intra_threads(4)
        .map_err(|e| format!("with_intra_threads: {e}"))?
        .commit_from_file(path)
        .map_err(|e| format!("commit_from_file: {e}"))
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

fn pct(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f32 * p) as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn stats(mut v: Vec<f32>) -> (f32, f32, f32, f32) {
    // (median, p95, mean, max)
    if v.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = v[v.len() / 2];
    let p95 = pct(&v, 0.95);
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    (median, p95, mean, *v.last().unwrap())
}

fn write_reports(
    args: &Args,
    sampled: &[(String, Vec<PathBuf>)],
    per_model: &[(String, Vec<FrameRun>)],
    cpu_timings: &[(String, Vec<f32>)],
    total_frames: usize,
) -> Result<(), String> {
    // CSV per model (timing row per frame).
    let mut csv_paths = Vec::new();
    for (name, runs) in per_model {
        let path = args.out_dir.join(format!("{name}.csv"));
        let mut csv = String::from("session,frame,ms_total,ms_run,persons\n");
        let mut i = 0usize;
        for (session, frames) in sampled {
            for path in frames {
                if i >= runs.len() {
                    break;
                }
                let r = &runs[i];
                csv.push_str(&format!(
                    "{session},{},{:.2},{:.2},{},\n",
                    path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
                    r.ms_total,
                    r.ms_run,
                    r.persons,
                ));
                i += 1;
            }
        }
        std::fs::write(&path, &csv).map_err(|e| format!("write {}: {e}", path.display()))?;
        csv_paths.push(path);
    }

    // Summary.
    let mut md = String::new();
    let unix_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    md.push_str(&format!(
        "# YOLO11-pose detector benchmark\n\n- date: unix {unix_secs}\n- frames: {total_frames} across {} sessions\n- note: measured with the live app running concurrently (GPU contention included)\n\n",
        sampled.len()
    ));

    md.push_str("## Timing (DirectML, per frame incl. preprocess)\n\n");
    md.push_str("| model | median | p95 | mean | max | fps(med) |\n|---|---|---|---|---|---|\n");
    for (name, runs) in per_model {
        let s = stats(runs.iter().map(|r| r.ms_total).collect());
        md.push_str(&format!(
            "| {name} | {:.1} ms | {:.1} ms | {:.1} ms | {:.1} ms | {:.1} |\n",
            s.0,
            s.1,
            s.2,
            s.3,
            1000.0 / s.0.max(1e-6)
        ));
    }
    md.push_str("\n");

    if !cpu_timings.is_empty() {
        md.push_str("## Timing (CPU EP, first-session sample)\n\n| model | median | p95 |\n|---|---|---|\n");
        for (name, times) in cpu_timings {
            let s = stats(times.clone());
            md.push_str(&format!("| {name} | {:.1} ms | {:.1} ms |\n", s.0, s.1));
        }
        md.push('\n');
    }

    md.push_str("## Persons detected per frame

");
    md.push_str("| model | median persons/frame |
|---|---|
");
    for (name, runs) in per_model {
        let mut persons_sum = 0usize;
        for r in runs {
            if let Some(p) = r.persons_checked() {
                persons_sum += p;
            }
        }
        md.push_str(&format!(
            "| {name} | {:.2} |
",
            if runs.is_empty() { 0.0 } else { persons_sum as f32 / runs.len() as f32 }
        ));
    }

    md.push_str("\n## CSVs\n\n");
    for p in &csv_paths {
        md.push_str(&format!("- {}\n", p.display()));
    }

    let summary = args.out_dir.join("summary.md");
    std::fs::write(&summary, &md).map_err(|e| format!("write {}: {e}", summary.display()))?;
    eprintln!("summary: {}", summary.display());
    println!("{md}");
    Ok(())
}


impl FrameRun {
    fn persons_checked(&self) -> Option<usize> {
        if self.persons == usize::MAX {
            None
        } else {
            Some(self.persons)
        }
    }
}
