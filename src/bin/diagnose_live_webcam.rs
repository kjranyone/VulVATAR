//! Live-webcam smoke test: runs the REAL tracking worker (camera →
//! provider → mailbox) for a few seconds and reports what the
//! pipeline produces — person presence, arm-joint coverage, forward-z
//! dynamics — without the GUI. Verifies the webcam input path
//! end-to-end with the production pipeline configuration.
//!
//! Usage:
//!   cargo run --bin diagnose_live_webcam -- [camera_index] [seconds]
//!
//! Output: one summary line per second (stdout only).

use std::time::{Duration, Instant};

use vulvatar_lib::asset::HumanoidBone;
use vulvatar_lib::tracking::{CameraBackend, TrackingWorker};

fn main() -> Result<(), String> {
    env_logger::init();
    let mut args = std::env::args().skip(1);
    let camera_index: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let seconds: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(12);

    let mailbox = vulvatar_lib::tracking::TrackingMailbox::new();
    let mut worker = TrackingWorker::new(mailbox.clone());
    worker.start_with_params(
        CameraBackend::Webcam { camera_index },
        640,
        480,
        30,
        Default::default(),
    );

    // Wait for camera + provider init (DirectML compile takes ~2-3 s).
    let init_deadline = Instant::now() + Duration::from_secs(30);
    while !worker.is_ready() {
        if Instant::now() > init_deadline {
            return Err("worker never became ready (camera busy or model load failed)".into());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    println!(
        "ready. backend={:?} provider={}",
        worker.active_backend(),
        mailbox.inference_backend_label().unwrap_or_default()
    );

    let arm_bones = [
        HumanoidBone::LeftUpperArm,
        HumanoidBone::RightUpperArm,
        HumanoidBone::LeftLowerArm,
        HumanoidBone::RightLowerArm,
        HumanoidBone::LeftHand,
        HumanoidBone::RightHand,
    ];

    let start = Instant::now();
    let mut last_seq = 0u64;
    let mut sec_frames = 0u32;
    let mut sec_person = 0u32;
    let mut sec_arms_full = 0u32;
    let mut wrist_z_min = f32::MAX;
    let mut wrist_z_max = f32::MIN;
    let mut next_report = start + Duration::from_secs(1);

    while start.elapsed() < Duration::from_secs(seconds) {
        std::thread::sleep(Duration::from_millis(10));
        if Instant::now() >= next_report {
            let z_range = if wrist_z_min <= wrist_z_max {
                format!("[{wrist_z_min:+.2},{wrist_z_max:+.2}]")
            } else {
                "n/a".to_string()
            };
            if let Some((err, _)) = mailbox.drain_error() {
                println!("  worker error: {err}");
            }
            println!(
                "t={:>3}s fps={:>2} person={:>2} arms6/6={:>2} wrist_z={} stale={} running={}",
                start.elapsed().as_secs(),
                sec_frames,
                sec_person,
                sec_arms_full,
                z_range,
                mailbox.is_stale(),
                worker.is_running(),
            );
            sec_frames = 0;
            sec_person = 0;
            sec_arms_full = 0;
            wrist_z_min = f32::MAX;
            wrist_z_max = f32::MIN;
            next_report += Duration::from_secs(1);
        }
        let seq = mailbox.sequence();
        if seq == last_seq {
            continue;
        }
        last_seq = seq;
        let Some(pose) = mailbox.latest_pose() else {
            continue;
        };
        sec_frames += 1;
        let person = pose.overall_confidence > 0.2;
        if person {
            sec_person += 1;
            let arms = arm_bones
                .iter()
                .filter(|b| pose.joints.contains_key(b))
                .count();
            if arms == arm_bones.len() {
                sec_arms_full += 1;
            }
            for wb in [HumanoidBone::LeftHand, HumanoidBone::RightHand] {
                if let Some(j) = pose.joints.get(&wb) {
                    wrist_z_min = wrist_z_min.min(j.position[2]);
                    wrist_z_max = wrist_z_max.max(j.position[2]);
                }
            }
        }

    }

    worker.stop();
    println!("done.");
    Ok(())
}
