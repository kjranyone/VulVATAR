//! Replay a RECORDED live source-joint sequence through the production
//! solver, headlessly — no camera, no inference, no GPU.
//!
//! Why: the live capture (2026-07-28) showed the avatar's hands teleporting
//! 0.45-0.60 m in a single frame, 34 times in 109 s, while the *source*
//! joints they came from were rock steady (±2 mm) and the raw depth under
//! the keypoints was 49/49 valid pixels. The jumps line up with the source
//! hand joint appearing / disappearing as the user's hands cross the frame
//! edge — which in their framing (hands off-screen 80-90 % of the time) is
//! constant. Diagnosing that needs the solver, not the camera, so this
//! binary feeds the recorded joints back in and reports what the avatar
//! does with them. The user's time is not a debugging resource.
//!
//! Input is the JSONL written by the live-debug poller: one object per
//! inference frame with `torso` / `arm` maps of `{p: [x,y,z], c, d}`,
//! plus `root` and `face`.
//!
//!   cargo run --bin diagnose_arm_replay -- <capture.jsonl> [max_frames]
//!
//! Env: `VULVATAR_VRM` (default `sample_data/AvatarSample_A.vrm`),
//! `VULVATAR_DIAG_BLEND` (default 0.7, the live rotation blend).

use std::sync::Arc;

use vulvatar_lib::asset::vrm::VrmAssetLoader;
use vulvatar_lib::asset::HumanoidBone;
use vulvatar_lib::avatar::pose_solver::{solve_avatar_pose, PoseSolverState, SolverParams};
use vulvatar_lib::avatar::{AvatarInstance, AvatarInstanceId};
use vulvatar_lib::tracking::hand_hold::HandHold;
use vulvatar_lib::tracking::source_skeleton::{FacePose, SourceJoint, SourceSkeleton};

/// Debug-dump key → humanoid bone, for both the `torso` and `arm` maps.
const DUMP_BONES: &[(&str, HumanoidBone)] = &[
    ("Head", HumanoidBone::Head),
    ("Neck", HumanoidBone::Neck),
    ("UpperChest", HumanoidBone::UpperChest),
    ("LSh", HumanoidBone::LeftShoulder),
    ("RSh", HumanoidBone::RightShoulder),
    ("Hips", HumanoidBone::Hips),
    ("LUp", HumanoidBone::LeftUpperArm),
    ("LLo", HumanoidBone::LeftLowerArm),
    ("LHa", HumanoidBone::LeftHand),
    ("RUp", HumanoidBone::RightUpperArm),
    ("RLo", HumanoidBone::RightLowerArm),
    ("RHa", HumanoidBone::RightHand),
];

fn joint_from(v: &serde_json::Value) -> Option<SourceJoint> {
    let p = v.get("p")?.as_array()?;
    let f = |i: usize| p.get(i).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
    Some(SourceJoint {
        position: [f(0), f(1), f(2)],
        confidence: v.get("c").and_then(|c| c.as_f64()).unwrap_or(0.0) as f32,
        metric_depth_m: v.get("d").and_then(|d| d.as_f64()).map(|d| d as f32),
    })
}

fn main() -> Result<(), String> {
    env_logger::init();
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: diagnose_arm_replay <capture.jsonl> [max_frames]")?;
    // Time window (recorded seconds) to replay, so a 200 s capture can be
    // narrowed to the few seconds around an artefact.
    let from_s: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let to_s: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(f64::MAX);
    // The solver derives its own `dt` from wall time and scales the
    // rotation blend by it, so a replay that runs flat out sees dt ~ 0 and
    // damps every transition into nothing — the exact artefact under
    // investigation would be filtered away by the harness. Sleep the
    // recorded inter-frame gap instead, so the solver's temporal filters
    // (1 euro, dt-aware blend, idle A-pose fade) see live timing.
    let pace = std::env::var("VULVATAR_REPLAY_PACE").map(|v| v != "0").unwrap_or(true);

    let vrm = std::env::var("VULVATAR_VRM")
        .unwrap_or_else(|_| "sample_data/AvatarSample_A.vrm".to_string());
    let asset = VrmAssetLoader::new()
        .load(&vrm)
        .map_err(|e| format!("load VRM {vrm}: {e:?}"))?;
    let mut avatar = AvatarInstance::new(AvatarInstanceId(1), Arc::clone(&asset));
    let humanoid = asset.humanoid.as_ref();
    let blend: f32 = std::env::var("VULVATAR_DIAG_BLEND")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.7);
    let params = SolverParams {
        rotation_blend: blend,
        joint_confidence_threshold: 0.1,
        face_confidence_threshold: 0.1,
        ..Default::default()
    };
    let mut state = PoseSolverState::default();
    // A/B the dropout bridge on identical recorded input:
    // `VULVATAR_REPLAY_HOLD=0` replays exactly what the live pipeline
    // published, `=1` (default) inserts the hold the provider now applies.
    let use_hold = std::env::var("VULVATAR_REPLAY_HOLD").map(|v| v != "0").unwrap_or(true);
    let mut hold = HandHold::default();

    let text = std::fs::read_to_string(&path).map_err(|e| format!("read {path}: {e}"))?;
    let mut prev: Option<([f32; 3], [f32; 3])> = None;
    let mut jumps: Vec<(f64, &'static str, f32, bool, bool)> = Vec::new();
    let mut frames = 0usize;

    println!("  t     src L/R hand | avatar LHa                  | avatar RHa                  | move L/R (m)");
    let mut prev_t: Option<f64> = None;
    for line in text.lines() {
        let Ok(rec) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let t = rec.get("t").and_then(|v| v.as_f64()).unwrap_or(0.0);
        if t < from_s || t > to_s {
            continue;
        }
        // Recorded inter-frame gap: paces the replay AND feeds the hold the
        // same dt the live provider would see. Captured before `prev_t` is
        // advanced (using it afterwards would hand every stage dt = 0).
        let gap = prev_t.map(|p| (t - p).clamp(0.0, 0.5)).unwrap_or(0.05);
        if pace {
            std::thread::sleep(std::time::Duration::from_secs_f64(gap));
        }
        prev_t = Some(t);
        let mut sk = SourceSkeleton::empty(frames as u64);
        for group in ["torso", "arm"] {
            let Some(map) = rec.get(group).and_then(|v| v.as_object()) else {
                continue;
            };
            for (key, bone) in DUMP_BONES {
                if let Some(v) = map.get(*key) {
                    if !v.is_null() {
                        if let Some(j) = joint_from(v) {
                            sk.joints.insert(*bone, j);
                        }
                    }
                }
            }
        }
        if let Some(r) = rec.get("root").and_then(|v| v.as_array()) {
            let f = |i: usize| r.get(i).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
            sk.root_offset = Some([f(0), f(1), f(2)]);
        }
        if let Some(fp) = rec.get("face").filter(|v| !v.is_null()) {
            let g = |k: &str| fp.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            sk.face = Some(FacePose {
                yaw: g("yaw"),
                pitch: g("pitch"),
                roll: g("roll"),
                confidence: g("c"),
                ..Default::default()
            });
        }
        sk.stamp_synthetic_metric_frame();
        if use_hold {
            hold.apply(&mut sk, gap as f32);
        }
        let l_src = sk.joints.contains_key(&HumanoidBone::LeftHand);
        let r_src = sk.joints.contains_key(&HumanoidBone::RightHand);

        avatar.build_base_pose();
        solve_avatar_pose(
            &sk,
            &asset.skeleton,
            humanoid,
            &mut avatar.pose.local_transforms,
            &params,
            &mut state,
        );
        avatar.compute_global_pose();
        let bone_pos = |b: HumanoidBone| -> [f32; 3] {
            humanoid
                .and_then(|h| h.bone_map.get(&b).copied())
                .map(|n| n.0 as usize)
                .and_then(|i| avatar.pose.global_transforms.get(i))
                .map(|m| [m[3][0], m[3][1], m[3][2]])
                .unwrap_or([f32::NAN; 3])
        };
        let lh = bone_pos(HumanoidBone::LeftHand);
        let rh = bone_pos(HumanoidBone::RightHand);
        let dist = |a: [f32; 3], b: [f32; 3]| {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        };
        let (dl, dr) = match prev {
            Some((pl, pr)) => (dist(pl, lh), dist(pr, rh)),
            None => (0.0, 0.0),
        };
        if dl > 0.10 {
            jumps.push((t, "LHa", dl, l_src, r_src));
        }
        if dr > 0.10 {
            jumps.push((t, "RHa", dr, l_src, r_src));
        }
        if frames % 10 == 0 || dl > 0.10 || dr > 0.10 {
            println!(
                "{t:6.2}   {}/{}      | {:6.3} {:6.3} {:6.3}       | {:6.3} {:6.3} {:6.3}       | {dl:.3} {dr:.3}",
                l_src as u8, r_src as u8, lh[0], lh[1], lh[2], rh[0], rh[1], rh[2]
            );
        }
        prev = Some((lh, rh));
        frames += 1;
    }

    eprintln!("\nreplayed {frames} frames from {path}");
    eprintln!("avatar hand jumps > 10 cm: {}", jumps.len());
    for (t, which, d, l, r) in jumps.iter().take(30) {
        eprintln!("  t={t:7.2} {which} {d:.3} m   src hands L={} R={}", *l as u8, *r as u8);
    }
    Ok(())
}
