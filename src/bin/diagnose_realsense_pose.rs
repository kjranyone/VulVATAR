//! End-to-end Phase B smoke test for the RealSense D435 depth rebuild.
//!
//! Runs the FULL provider path on live frames — RTMW3D on the D435 color
//! image, metric depth built from the D435 depth via
//! `build_metric_frame_from_d435`, injected with `set_external_depth`, and
//! consumed by `estimate_from_external_depth` — then reports the resulting
//! `SourceSkeleton` (joint count, how many joints carry metric depth,
//! overall confidence, and a sample of joint positions). Proves the
//! external-depth wiring produces a body skeleton from real sensor depth,
//! without the GUI / renderer.
//!
//!   cargo run --bin diagnose_realsense_pose --features realsense
//!
//! Build/run env: see docs/realsense-build.md. Needs the RTMW3D models in
//! `models/` (the same set the app uses).

use vulvatar_lib::tracking::provider::{create_pose_provider, TrackingPipelineConfig};
use vulvatar_lib::tracking::realsense::RealSenseCapture;
use vulvatar_lib::tracking::rtmw3d_with_depth::build_metric_frame_from_d435;

fn main() -> Result<(), String> {
    env_logger::init();

    let mut cap = RealSenseCapture::open(1280, 720, 30)?;
    // depth_enabled: false — the D435 replaces the DAv2 stage entirely.
    let mut provider = create_pose_provider(
        "models",
        TrackingPipelineConfig {
            depth_enabled: false,
            ..Default::default()
        },
    )?;
    println!("provider: {}", provider.label());

    for i in 0..30u64 {
        let frame = cap.grab_frame()?;
        let metric = build_metric_frame_from_d435(&frame);

        let total = metric.points_m.len().max(1) as f32;
        let valid = metric
            .points_m
            .iter()
            .filter(|p| p[2].is_finite() && p[2] > 0.0)
            .count();

        provider.set_external_depth(metric);
        let est = provider.estimate_pose(&frame.rgb, frame.width, frame.height, i);

        if i % 5 != 0 {
            continue;
        }

        let sk = &est.skeleton;
        let with_depth = sk
            .joints
            .values()
            .filter(|j| j.metric_depth_m.is_some())
            .count();
        println!(
            "frame {:02}: depth_valid={:.0}%  joints={}  metric_joints={}  overall_conf={:.2}",
            i,
            100.0 * valid as f32 / total,
            sk.joints.len(),
            with_depth,
            sk.overall_confidence,
        );

        if i == 25 {
            let mut items: Vec<_> = sk.joints.iter().collect();
            items.sort_by_key(|(b, _)| format!("{:?}", b));
            for (bone, j) in items.iter().take(10) {
                println!(
                    "   {:<20?} pos=[{:+.3},{:+.3},{:+.3}] depth_m={:?} conf={:.2}",
                    bone, j.position[0], j.position[1], j.position[2], j.metric_depth_m, j.confidence
                );
            }
        }
    }

    println!("done — external-depth provider path verified.");
    Ok(())
}
