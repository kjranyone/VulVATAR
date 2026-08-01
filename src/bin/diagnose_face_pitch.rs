//! Face-pitch estimator probe. Runs the production RTMW3D provider over a
//! directory of frames and dumps the raw face keypoints (COCO-Wholebody
//! 0..=4 body-face + 23..=90 dlib-68) plus the published face pose to JSON,
//! so candidate pitch estimators can be evaluated OFFLINE against frames
//! whose true head pose is known (e.g. a yaw sweep held at constant
//! horizontal gaze).
//!
//! Motivation: the shipping body-path pitch normalises the nose-below-eye
//! distance by the INTER-EYE distance, which foreshortens by cos(yaw) —
//! and compensates with a `cos(yaw)` term taken from the ear-line yaw,
//! which collapses to ~0 whenever the ears are occluded. Live capture
//! showed body yaw pinned at ±5 deg through a real 45-70 deg head turn,
//! so the compensation never fires and a head TURN decodes as a 30-50 deg
//! chin-DOWN nod. Any replacement must be validated on real frames before
//! it goes in the pipeline; this binary produces that evidence.
//!
//!   cargo run --no-default-features --features inference \
//!       --bin diagnose_face_pitch -- <frames_dir> [out_json]
//!
//! Output defaults to `diagnostics/face_pitch/keypoints.json` (never
//! `validation_images/`, per the repo's validation-data rules).

use std::path::PathBuf;

fn main() -> Result<(), String> {
    env_logger::init();
    let dir = std::env::args()
        .nth(1)
        .ok_or("usage: diagnose_face_pitch <frames_dir> [out_json]")?;
    let out = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "diagnostics/face_pitch/keypoints.json".to_string());
    let out = PathBuf::from(out);
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let config = vulvatar_lib::tracking::provider::TrackingPipelineConfig::default();
    let mut provider = vulvatar_lib::tracking::provider::create_pose_provider("models", config)?;
    for w in provider.take_load_warnings() {
        eprintln!("load warning: {w}");
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("read_dir {dir}: {e}"))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|x| x == "png" || x == "jpg")
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    eprintln!("{} frames from {dir}", files.len());

    let mut frames = Vec::new();
    for (i, f) in files.iter().enumerate() {
        let img = image::open(f)
            .map_err(|e| format!("open {}: {e}", f.display()))?
            .to_rgb8();
        let (w, h) = (img.width(), img.height());
        let est = provider.estimate_pose(img.as_raw(), w, h, i as u64);
        let kp: Vec<serde_json::Value> = est
            .annotation
            .keypoints
            .iter()
            .map(|&(x, y, s)| serde_json::json!([x, y, s]))
            .collect();
        let face = est.skeleton.face.map(|p| {
            serde_json::json!({
                "yaw": p.yaw, "pitch": p.pitch, "roll": p.roll,
                "c": p.confidence, "source": format!("{:?}", p.source),
            })
        });
        let face_body_raw = est.skeleton.face_body_raw.map(|p| {
            serde_json::json!({ "yaw": p.yaw, "pitch": p.pitch, "roll": p.roll, "c": p.confidence })
        });
        eprintln!(
            "f{i:03} {} overall={:.2} mesh_c={:?} face={}",
            f.file_name().unwrap_or_default().to_string_lossy(),
            est.skeleton.overall_confidence,
            est.skeleton.face_mesh_confidence,
            face.as_ref()
                .map(|v| v.to_string())
                .unwrap_or_else(|| "none".into()),
        );
        frames.push(serde_json::json!({
            "file": f.file_name().unwrap_or_default().to_string_lossy(),
            "w": w, "h": h,
            "overall": est.skeleton.overall_confidence,
            "mesh_c": est.skeleton.face_mesh_confidence,
            "face": face,
            "face_body_raw": face_body_raw,
            "kp": kp,
        }));
    }

    std::fs::write(
        &out,
        serde_json::to_vec_pretty(&serde_json::json!({ "frames": frames }))
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    eprintln!("wrote {}", out.display());
    Ok(())
}
