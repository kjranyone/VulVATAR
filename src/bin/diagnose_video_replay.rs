//! Video-sequence replay: feeds an extracted frame directory through
//! the production provider in order with CONTINUOUS temporal state —
//! the live pipeline, minus the camera. Used to validate temporal
//! behaviours that static benchmarks cannot represent (out-of-frame
//! slide-out / re-entry, holds, self-track stability).
//!
//! Usage:
//!   cargo run --bin diagnose_video_replay -- [frames_dir] [out.jsonl]
//!
//! Emits one JSON line per frame to out.jsonl (default
//! diagnostics/video_replay.jsonl) with arm-chain observability.

use std::io::Write;
use std::path::PathBuf;

use std::sync::Arc;

use vulvatar_lib::asset::vrm::VrmAssetLoader;
use vulvatar_lib::asset::HumanoidBone;
use vulvatar_lib::avatar::pose_solver::{solve_avatar_pose, PoseSolverState, SolverParams};
use vulvatar_lib::avatar::{AvatarInstance, AvatarInstanceId};
use vulvatar_lib::tracking::provider::create_pose_provider;

fn main() -> Result<(), String> {
    env_logger::init();
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "diagnostics/video_frames".to_string());
    let out_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "diagnostics/video_replay.jsonl".to_string());

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("read_dir {dir}: {e}"))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|x| x == "jpg" || x == "png")
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    eprintln!("{} frames", files.len());

    let mut provider = create_pose_provider("models", Default::default())?;
    let _ = provider.take_load_warnings();

    // Continuous solver, mirroring the live GUI configuration
    // (rotation_blend 1.0 per the user's project, lower body off).
    let vrm = std::env::var("VULVATAR_VRM")
        .unwrap_or_else(|_| "sample_data/AvatarSample_A.vrm".to_string());
    let asset = VrmAssetLoader::new()
        .load(&vrm)
        .map_err(|e| format!("load VRM: {e:?}"))?;
    let humanoid_map = asset.humanoid.as_ref().ok_or("no humanoid")?;
    let mut avatar = AvatarInstance::new(AvatarInstanceId(1), Arc::clone(&asset));
    let mut solver_state = PoseSolverState::default();
    let params = SolverParams {
        rotation_blend: 1.0,
        lower_body_tracking_enabled: false,
        ..Default::default()
    };

    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&out_path).map_err(|e| format!("create {out_path}: {e}"))?,
    );

    let sides = [
        ("L", HumanoidBone::LeftUpperArm, HumanoidBone::LeftLowerArm, HumanoidBone::LeftHand),
        ("R", HumanoidBone::RightUpperArm, HumanoidBone::RightLowerArm, HumanoidBone::RightHand),
    ];

    for (i, f) in files.iter().enumerate() {
        let img = image::open(f).map_err(|e| format!("open {}: {e}", f.display()))?.to_rgb8();
        let (w, h) = (img.width(), img.height());
        let est = provider.estimate_pose(img.as_raw(), w, h, i as u64);
        let sk = est.skeleton;

        avatar.build_base_pose();
        solve_avatar_pose(
            &sk,
            &asset.skeleton,
            asset.humanoid.as_ref(),
            &mut avatar.pose.local_transforms,
            &params,
            &mut solver_state,
        );
        avatar.compute_global_pose();
        let av_pos = |b: HumanoidBone| -> Option<[f32; 3]> {
            let idx = humanoid_map.bone_map.get(&b).copied().map(|n| n.0 as usize)?;
            let m = avatar.pose.global_transforms.get(idx)?;
            Some([m[3][0], m[3][1], m[3][2]])
        };

        let mut parts: Vec<String> = vec![format!("\"frame\":{i}")];
        for (tag, sh, el, wr) in sides {
            let g = |b: HumanoidBone| sk.joints.get(&b);
            match (g(sh), g(el), g(wr)) {
                (Some(s), Some(e), Some(wj)) => {
                    let d3 = |a: [f32; 3], b: [f32; 3]| {
                        let v = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
                    };
                    parts.push(format!(
                        "\"{tag}\":{{\"wx\":{:.3},\"wy\":{:.3},\"wz\":{:.3},\"conf\":{:.2},\"fore\":{:.3},\"upper\":{:.3}}}",
                        wj.position[0], wj.position[1], wj.position[2],
                        wj.confidence,
                        d3(e.position, wj.position),
                        d3(s.position, e.position),
                    ));
                }
                _ => parts.push(format!("\"{tag}\":null")),
            }
            let hand_bone = if tag == "L" { HumanoidBone::LeftHand } else { HumanoidBone::RightHand };
            if let Some(p) = av_pos(hand_bone) {
                parts.push(format!(
                    "\"{tag}av\":{{\"x\":{:.3},\"y\":{:.3},\"z\":{:.3}}}",
                    p[0], p[1], p[2]
                ));
            }
        }
        parts.push(format!("\"overall\":{:.2}", sk.overall_confidence));
        writeln!(out, "{{{}}}", parts.join(",")).map_err(|e| e.to_string())?;
        if i % 100 == 0 {
            eprintln!("frame {i}");
        }
    }
    eprintln!("wrote {out_path}");
    Ok(())
}
