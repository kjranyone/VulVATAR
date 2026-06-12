//! Temporal wrist-hold diagnostic: replays an A→B pose transition
//! through the production provider WITHOUT per-frame temporal resets,
//! reproducing live-session wrist-tracker / self-track / sticky-depth
//! behaviour that the per-image-reset `validate_pipeline` cannot see.
//!
//! Usage:
//!   cargo run --bin diagnose_wrist_hold -- <imageA> <imageB> [nA] [nB]
//!
//! Prints one line per frame: wrist presence, confidence, position,
//! and forearm/upper-arm lengths for both sides. Output is stdout
//! only (no files), so the validation-assets hygiene rule is moot.

use std::path::PathBuf;
use std::sync::Arc;

use vulvatar_lib::asset::vrm::VrmAssetLoader;
use vulvatar_lib::asset::HumanoidBone;
use vulvatar_lib::avatar::pose_solver::{solve_avatar_pose, PoseSolverState, SolverParams};
use vulvatar_lib::avatar::{AvatarInstance, AvatarInstanceId};
use vulvatar_lib::tracking::provider::create_pose_provider;
use vulvatar_lib::tracking::source_skeleton::SourceSkeleton;

fn main() -> Result<(), String> {
    env_logger::init();
    let mut args = std::env::args().skip(1);
    let img_a = PathBuf::from(args.next().ok_or("usage: diagnose_wrist_hold <imageA> <imageB> [nA] [nB]")?);
    let img_b = PathBuf::from(args.next().ok_or("missing <imageB>")?);
    let n_a: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);
    let n_b: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);

    let a = image::open(&img_a).map_err(|e| format!("open A: {e}"))?.to_rgb8();
    let b = image::open(&img_b).map_err(|e| format!("open B: {e}"))?.to_rgb8();

    let mut provider = create_pose_provider("models", Default::default())
        .map_err(|e| format!("provider: {e}"))?;
    let _ = provider.take_load_warnings();

    // Live-like solver: persistent state across frames, production
    // default blend (no benchmark snap).
    let vrm = std::env::var("VULVATAR_VRM")
        .unwrap_or_else(|_| "sample_data/AvatarSample_A.vrm".to_string());
    let asset = VrmAssetLoader::new()
        .load(&vrm)
        .map_err(|e| format!("load VRM: {e:?}"))?;
    let mut avatar = AvatarInstance::new(AvatarInstanceId(1), Arc::clone(&asset));
    let mut state = PoseSolverState::default();
    // VULVATAR_DIAG_BLEND: 1.0 = validate_pipeline's snap params,
    // otherwise the live default (0.7 dt-aware). Lets one binary A/B
    // the blend path against the snap path on identical input.
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
    let humanoid = asset.humanoid.as_ref();

    println!("frame img | L_wrist conf  pos                | R_wrist conf  pos                | fore_err L/R (deg)  upper_err L/R (deg)");
    for i in 0..(n_a + n_b) {
        let (img, tag) = if i < n_a { (&a, "A") } else { (&b, "B") };
        let est = provider.estimate_pose(img.as_raw(), img.width(), img.height(), i);
        let sk = est.skeleton;

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

        let bone_pos = |b: HumanoidBone| -> Option<[f32; 3]> {
            let idx = humanoid?.bone_map.get(&b).copied().map(|n| n.0 as usize)?;
            let m = avatar.pose.global_transforms.get(idx)?;
            Some([m[3][0], m[3][1], m[3][2]])
        };
        let src_pos = |b: HumanoidBone| sk.joints.get(&b).map(|j| j.position);
        let xy_angle = |base: HumanoidBone, tip: HumanoidBone| -> f32 {
            let (Some(ab), Some(at), Some(sb), Some(st)) =
                (bone_pos(base), bone_pos(tip), src_pos(base), src_pos(tip))
            else {
                return f32::NAN;
            };
            let av = norm2([at[0] - ab[0], at[1] - ab[1]]);
            let sv = norm2([st[0] - sb[0], st[1] - sb[1]]);
            let dot = (av[0] * sv[0] + av[1] * sv[1]).clamp(-1.0, 1.0);
            dot.acos().to_degrees()
        };

        let fe_l = xy_angle(HumanoidBone::LeftLowerArm, HumanoidBone::LeftHand);
        let fe_r = xy_angle(HumanoidBone::RightLowerArm, HumanoidBone::RightHand);
        let ue_l = xy_angle(HumanoidBone::LeftUpperArm, HumanoidBone::LeftLowerArm);
        let ue_r = xy_angle(HumanoidBone::RightUpperArm, HumanoidBone::RightLowerArm);
        let row = describe(&sk);
        println!("{i:>4}  {tag}  | {row} | {fe_l:5.1}/{fe_r:5.1}  {ue_l:5.1}/{ue_r:5.1}");
    }
    Ok(())
}

fn norm2(v: [f32; 2]) -> [f32; 2] {
    let l = (v[0] * v[0] + v[1] * v[1]).sqrt().max(1e-6);
    [v[0] / l, v[1] / l]
}

fn describe(sk: &SourceSkeleton) -> String {
    let side = |wrist: HumanoidBone, elbow: HumanoidBone, shoulder: HumanoidBone| -> String {
        let w = sk.joints.get(&wrist);
        let e = sk.joints.get(&elbow);
        let s = sk.joints.get(&shoulder);
        let fore = match (w, e) {
            (Some(w), Some(e)) => dist(w.position, e.position),
            _ => f32::NAN,
        };
        let upper = match (e, s) {
            (Some(e), Some(s)) => dist(e.position, s.position),
            _ => f32::NAN,
        };
        match w {
            Some(w) => format!(
                "{:.2} ({:+.2},{:+.2},{:+.2}) | {:.3}  {:.3}",
                w.confidence, w.position[0], w.position[1], w.position[2], fore, upper
            ),
            None => format!("MISS                       | {fore:.3}  {upper:.3}"),
        }
    };
    format!(
        "{} | {}",
        side(
            HumanoidBone::LeftHand,
            HumanoidBone::LeftLowerArm,
            HumanoidBone::LeftUpperArm
        ),
        side(
            HumanoidBone::RightHand,
            HumanoidBone::RightLowerArm,
            HumanoidBone::RightUpperArm
        )
    )
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}
