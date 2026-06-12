//! Out-of-frame benchmark: for each (uncropped, cropped) image pair,
//! run the production pipeline on both and compare the solved
//! avatar's arm-bone world directions. The uncropped run is ground
//! truth (the limb is fully visible); the cropped run sees the hand
//! exit the frame — the difference measures how badly edge-clamped
//! keypoints corrupt the pose.
//!
//! Usage:
//!   cargo run --bin diagnose_edge_exit -- [pairs.txt]
//!
//! pairs.txt lines: <uncropped>\t<cropped>\t<tag>
//! Output (stdout): per-pair worst arm-bone 3D angle + summary.

use std::path::PathBuf;
use std::sync::Arc;

use vulvatar_lib::asset::vrm::VrmAssetLoader;
use vulvatar_lib::asset::HumanoidBone;
use vulvatar_lib::avatar::pose_solver::{solve_avatar_pose, PoseSolverState, SolverParams};
use vulvatar_lib::avatar::{AvatarInstance, AvatarInstanceId};
use vulvatar_lib::tracking::provider::{create_pose_provider, PoseProvider};

const ARM_BONES: &[(&str, HumanoidBone, HumanoidBone)] = &[
    ("L_upper", HumanoidBone::LeftUpperArm, HumanoidBone::LeftLowerArm),
    ("R_upper", HumanoidBone::RightUpperArm, HumanoidBone::RightLowerArm),
    ("L_fore", HumanoidBone::LeftLowerArm, HumanoidBone::LeftHand),
    ("R_fore", HumanoidBone::RightLowerArm, HumanoidBone::RightHand),
];

fn main() -> Result<(), String> {
    env_logger::init();
    let pairs_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "diagnostics/edge_exit_inputs/pairs.txt".to_string());
    let pairs: Vec<(PathBuf, PathBuf, String)> = std::fs::read_to_string(&pairs_path)
        .map_err(|e| format!("read {pairs_path}: {e}"))?
        .lines()
        .filter_map(|l| {
            let mut it = l.split('\t');
            Some((
                PathBuf::from(it.next()?),
                PathBuf::from(it.next()?),
                it.next()?.to_string(),
            ))
        })
        .collect();

    let mut provider = create_pose_provider("models", Default::default())?;
    let _ = provider.take_load_warnings();
    let vrm = std::env::var("VULVATAR_VRM")
        .unwrap_or_else(|_| "sample_data/AvatarSample_A.vrm".to_string());
    let asset = VrmAssetLoader::new()
        .load(&vrm)
        .map_err(|e| format!("load VRM: {e:?}"))?;
    let humanoid = asset.humanoid.as_ref().ok_or("no humanoid")?;

    let solve = |provider: &mut Box<dyn PoseProvider>,
                     img_path: &PathBuf|
     -> Result<Vec<Option<[f32; 3]>>, String> {
        let img = image::open(img_path)
            .map_err(|e| format!("open {}: {e}", img_path.display()))?
            .to_rgb8();
        let (w, h) = (img.width(), img.height());
        // Reset policy is owned by the pair loop (warm mode resets
        // once per pair, before the truth run; cold mode resets
        // before every image).
        if std::env::var("VULVATAR_EDGE_WARM").is_err() {
            provider.reset_temporal_state();
        }
        // Acquisition pass + steady-state pass (matches the live
        // self-tracking crop behaviour).
        let _ = provider.estimate_pose(img.as_raw(), w, h, 0);
        let est = provider.estimate_pose(img.as_raw(), w, h, 1);

        let mut avatar = AvatarInstance::new(AvatarInstanceId(1), Arc::clone(&asset));
        avatar.build_base_pose();
        let mut state = PoseSolverState::default();
        let params = SolverParams {
            rotation_blend: 1.0,
            lower_body_tracking_enabled: false,
            ..Default::default()
        };
        solve_avatar_pose(
            &est.skeleton,
            &asset.skeleton,
            asset.humanoid.as_ref(),
            &mut avatar.pose.local_transforms,
            &params,
            &mut state,
        );
        avatar.compute_global_pose();

        let pos = |b: HumanoidBone| -> Option<[f32; 3]> {
            let idx = humanoid.bone_map.get(&b).copied().map(|n| n.0 as usize)?;
            let m = avatar.pose.global_transforms.get(idx)?;
            Some([m[3][0], m[3][1], m[3][2]])
        };
        Ok(ARM_BONES
            .iter()
            .map(|(_, base, tip)| {
                let (b, t) = (pos(*base)?, pos(*tip)?);
                let v = [t[0] - b[0], t[1] - b[1], t[2] - b[2]];
                let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                if l < 1e-4 {
                    None
                } else {
                    Some([v[0] / l, v[1] / l, v[2] / l])
                }
            })
            .collect())
    };

    let mut worst_overall = 0.0f32;
    let mut fail = 0;
    println!("{:>7} {:>7} {:>7} {:>7} | worst | pair", "L_up", "R_up", "L_fo", "R_fo");
    for (orig, crop, tag) in &pairs {
        // Warm mode: state flows truth -> crop within a pair (the
        // live arm-slides-out scenario) but never across pairs.
        if std::env::var("VULVATAR_EDGE_WARM").is_ok() {
            provider.reset_temporal_state();
        }
        let truth = solve(&mut provider, orig)?;
        let test = solve(&mut provider, crop)?;
        let mut angles = [f32::NAN; 4];
        let mut worst = 0.0f32;
        for (i, (t, c)) in truth.iter().zip(test.iter()).enumerate() {
            if let (Some(t), Some(c)) = (t, c) {
                let dot = (t[0] * c[0] + t[1] * c[1] + t[2] * c[2]).clamp(-1.0, 1.0);
                let a = dot.acos().to_degrees();
                angles[i] = a;
                worst = worst.max(a);
            }
        }
        worst_overall = worst_overall.max(worst);
        if worst > 25.0 {
            fail += 1;
        }
        let name = orig
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let short = name.split("streamer_").last().unwrap_or(&name);
        println!(
            "{:>7.1} {:>7.1} {:>7.1} {:>7.1} | {:>5.1} | {}__{}",
            angles[0], angles[1], angles[2], angles[3], worst, short, tag
        );
    }
    println!(
        "\npairs={} fail(worst>25deg)={} worst_overall={:.1}deg",
        pairs.len(),
        fail,
        worst_overall
    );
    Ok(())
}
