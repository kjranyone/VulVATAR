//! Per-provider pose-inference benchmark.
//!
//! Loads one image, builds the requested `PoseProviderKind`, runs
//! `estimate_pose` N times, and reports per-iteration wall time. Same
//! warm-up + steady-state convention as `bench_pose` but targeting
//! the provider abstraction so all three pipelines (`rtmw3d`,
//! `rtmw3d-with-depth`, `cigpose-metric-depth`) can be compared on
//! the same image with the same harness.
//!
//! Usage:
//!   cargo run --bin bench_provider -- <image.png> <provider> [iters]
//!     <provider> = rtmw3d | rtmw3d-with-depth | cigpose-metric-depth
//!     [iters]    = number of timed iterations (default 10)

use std::path::PathBuf;
use std::time::Instant;
use vulvatar_lib::tracking::provider::{create_pose_provider, PoseProviderKind};

fn main() -> Result<(), String> {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let image_path = PathBuf::from(args.next().ok_or_else(|| {
        "usage: bench_provider <image.png> <provider> [iters]".to_string()
    })?);
    let provider_arg = args
        .next()
        .ok_or_else(|| "missing <provider>".to_string())?;
    let iters: usize = args
        .next()
        .map(|s| s.parse().unwrap_or(10))
        .unwrap_or(10);

    let kind = match provider_arg.as_str() {
        "rtmw3d" => PoseProviderKind::Rtmw3d,
        "rtmw3d-with-depth" | "rtmw3d-da" => PoseProviderKind::Rtmw3dWithDepth,
        "cigpose-metric-depth" | "cigpose-metric" => PoseProviderKind::CigposeMetricDepth,
        other => {
            return Err(format!(
                "unknown provider '{}'. expected: rtmw3d | rtmw3d-with-depth | cigpose-metric-depth",
                other
            ));
        }
    };

    let img = image::open(&image_path).map_err(|e| format!("open image: {e}"))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    eprintln!("image: {}x{} from {}", w, h, image_path.display());
    eprintln!("provider: {:?}", kind);

    let mut provider =
        create_pose_provider(kind, "models").map_err(|e| format!("provider load: {e}"))?;
    eprintln!("label: {}", provider.label());
    let warnings = provider.take_load_warnings();
    for w in &warnings {
        eprintln!("warning: {}", w);
    }

    eprintln!("warm-up…");
    let _ = provider.estimate_pose(rgb.as_raw(), w, h, 0);
    let _ = provider.estimate_pose(rgb.as_raw(), w, h, 1);

    eprintln!("running {} iterations…", iters);
    let mut samples_ms: Vec<f32> = Vec::with_capacity(iters);
    for i in 0..iters {
        let t0 = Instant::now();
        let _ = provider.estimate_pose(rgb.as_raw(), w, h, (i + 2) as u64);
        samples_ms.push(t0.elapsed().as_secs_f32() * 1000.0);
    }

    samples_ms.sort_by(|a, b| a.total_cmp(b));
    let min = samples_ms.first().copied().unwrap_or(0.0);
    let max = samples_ms.last().copied().unwrap_or(0.0);
    let median = samples_ms[samples_ms.len() / 2];
    let mean = samples_ms.iter().sum::<f32>() / samples_ms.len() as f32;
    let fps = if median > 0.0 { 1000.0 / median } else { 0.0 };

    println!(
        "iters={} min={:.1}ms median={:.1}ms mean={:.1}ms max={:.1}ms (=> {:.1} fps median)",
        iters, min, median, mean, max, fps
    );

    Ok(())
}
