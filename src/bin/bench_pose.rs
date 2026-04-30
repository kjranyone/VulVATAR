//! Pose-inference benchmark.
//!
//! Loads one image and runs `Rtmw3dInference::estimate_pose` N times,
//! reporting per-iteration wall time. Stage breakdown lands in stderr
//! via the existing `info!` calls inside `estimate_pose_internal`
//! (run with `RUST_LOG=vulvatar=info`).
//!
//! Usage:
//!   cargo run --release --bin bench_pose -- <image.png> [iters]
//!   cargo run --release --features inference-gpu --bin bench_pose -- <image.png> [iters]

use std::path::PathBuf;
use std::time::Instant;
use vulvatar_lib::tracking::rtmw3d::Rtmw3dInference;

fn main() -> Result<(), String> {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let image_path = PathBuf::from(
        args.next()
            .ok_or_else(|| "usage: bench_pose <image.png> [iters]".to_string())?,
    );
    let iters: usize = args
        .next()
        .map(|s| s.parse().unwrap_or(10))
        .unwrap_or(10);

    let img = image::open(&image_path).map_err(|e| format!("open image: {e}"))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    eprintln!("image: {}x{} from {}", w, h, image_path.display());

    let mut infer = Rtmw3dInference::from_models_dir("models")
        .map_err(|e| format!("load Rtmw3dInference: {e}"))?;
    eprintln!("backend: {}", infer.backend().label());

    // Warm-up so the ORT session, allocator and JIT are settled before
    // we start the stopwatch.
    eprintln!("warm-up…");
    let _ = infer.estimate_pose(rgb.as_raw(), w, h, 0);
    let _ = infer.estimate_pose(rgb.as_raw(), w, h, 1);

    eprintln!("running {} iterations…", iters);
    let mut samples_ms: Vec<f32> = Vec::with_capacity(iters);
    for i in 0..iters {
        let t0 = Instant::now();
        let _ = infer.estimate_pose(rgb.as_raw(), w, h, (i + 2) as u64);
        samples_ms.push(t0.elapsed().as_secs_f32() * 1000.0);
    }

    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples_ms.len();
    let median = samples_ms[n / 2];
    let min = samples_ms[0];
    let max = samples_ms[n - 1];
    let mean: f32 = samples_ms.iter().copied().sum::<f32>() / n as f32;

    println!(
        "iters={n} min={min:.1}ms median={median:.1}ms mean={mean:.1}ms max={max:.1}ms (=> {:.1} fps median)",
        1000.0 / median
    );
    Ok(())
}
