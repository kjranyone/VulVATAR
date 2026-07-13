//! Live debug channel — lets an external tool (`scratchpad/live_debug.py`)
//! observe the running app's tracking pipeline AND tune the solver **without a
//! rebuild**, so a fix can be A/B-toggled or a threshold swept against the live
//! camera instead of re-launching for every experiment.
//!
//! Off by default: every entry point is a no-op unless the flag file
//! `%ProgramData%\VulVATAR\debug.on` exists (one `exists()` stat per frame).
//! When on:
//!   * [`dump_observation`] overwrites `debug_camera.bin` (down-scaled camera
//!     RGBA behind the same 32-byte header the virtual-camera tap already
//!     parses) and `debug_state.json` (2D keypoints + source arm joints), so the
//!     external overlay can line up camera ↔ 2D ↔ avatar-output in one view.
//!   * [`load_tuning`] reads `debug_tuning.json` (re-parsed only when its mtime
//!     changes) into a [`Tuning`]; the app applies it each frame.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::SystemTime;

use super::PoseEstimate;
use crate::asset::HumanoidBone;

fn base_dir() -> PathBuf {
    std::env::var_os("ProgramData")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\ProgramData"))
        .join("VulVATAR")
}

/// The master switch. Absent flag file → the whole channel is dead weight of one
/// `stat` per frame.
pub fn enabled() -> bool {
    base_dir().join("debug.on").exists()
}

fn atomic_write(path: &Path, bytes: &[u8]) {
    // Write-then-rename so a reader never sees a half-written buffer.
    let tmp = path.with_extension("writing");
    if std::fs::write(&tmp, bytes).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

/// Nearest-neighbour down-scale of the camera RGB to `dst_w` wide (aspect kept)
/// as an RGBA buffer with the 32-byte header the tap scripts read
/// (`magic 'VDBG', u32 w, u32 h, u32 fmt=0, u32 pad, u64 ts@16`).
fn write_camera(rgb: &[u8], w: u32, h: u32, dst_w: u32, frame_index: u64) {
    if w == 0 || h == 0 {
        return;
    }
    let dst_h = (dst_w * h / w).max(1);
    let mut out = vec![0u8; 32 + (dst_w as usize * dst_h as usize * 4)];
    out[0..4].copy_from_slice(b"VDBG");
    out[4..8].copy_from_slice(&dst_w.to_le_bytes());
    out[8..12].copy_from_slice(&dst_h.to_le_bytes());
    out[16..24].copy_from_slice(&frame_index.to_le_bytes());
    for dy in 0..dst_h {
        let sy = (dy * h / dst_h).min(h - 1);
        for dx in 0..dst_w {
            let sx = (dx * w / dst_w).min(w - 1);
            let si = ((sy * w + sx) * 3) as usize;
            let di = 32 + ((dy * dst_w + dx) * 4) as usize;
            if si + 2 < rgb.len() {
                out[di] = rgb[si];
                out[di + 1] = rgb[si + 1];
                out[di + 2] = rgb[si + 2];
                out[di + 3] = 255;
            }
        }
    }
    atomic_write(&base_dir().join("debug_camera.bin"), &out);
}

/// Publish the current frame's camera + 2D keypoints + source arm joints for an
/// external overlay. No-op unless the debug flag file is present.
pub fn dump_observation(frame_index: u64, rgb: &[u8], w: u32, h: u32, est: &PoseEstimate) {
    if !enabled() {
        return;
    }
    write_camera(rgb, w, h, 640, frame_index);

    let kps: Vec<serde_json::Value> = est
        .annotation
        .keypoints
        .iter()
        .take(17)
        .map(|&(x, y, s)| serde_json::json!([x, y, s]))
        .collect();
    let arm = |b: HumanoidBone| {
        est.skeleton
            .joints
            .get(&b)
            .map(|j| serde_json::json!({ "p": j.position, "c": j.confidence }))
    };
    let state = serde_json::json!({
        "frame": frame_index,
        "overall": est.skeleton.overall_confidence,
        "kp": kps,
        "arm": {
            "LUp": arm(HumanoidBone::LeftUpperArm),
            "LLo": arm(HumanoidBone::LeftLowerArm),
            "LHa": arm(HumanoidBone::LeftHand),
            "RUp": arm(HumanoidBone::RightUpperArm),
            "RLo": arm(HumanoidBone::RightLowerArm),
            "RHa": arm(HumanoidBone::RightHand),
        },
    });
    if let Ok(bytes) = serde_json::to_vec(&state) {
        atomic_write(&base_dir().join("debug_state.json"), &bytes);
    }
}

static AVATAR_DUMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Publish the SOLVED avatar's key joint world positions (after
/// `compute_global_pose`) so the external overlay can draw the avatar skeleton
/// WITHOUT a GPU render — enough to see torso tilt, elbow placement, whole-body
/// rotation, etc. Overwrites `debug_avatar.json` each frame (a `seq` counter
/// lets a reader detect fresh frames). No-op unless the debug flag file exists.
pub fn dump_avatar_pose<F: Fn(HumanoidBone) -> Option<[f32; 3]>>(pos: F) {
    if !enabled() {
        return;
    }
    use HumanoidBone::*;
    let bones: [(&str, HumanoidBone); 14] = [
        ("Hips", Hips),
        ("Spine", Spine),
        ("Chest", Chest),
        ("UpperChest", UpperChest),
        ("Neck", Neck),
        ("Head", Head),
        ("LSh", LeftShoulder),
        ("RSh", RightShoulder),
        ("LUp", LeftUpperArm),
        ("RUp", RightUpperArm),
        ("LLo", LeftLowerArm),
        ("RLo", RightLowerArm),
        ("LHa", LeftHand),
        ("RHa", RightHand),
    ];
    let map: serde_json::Map<String, serde_json::Value> = bones
        .iter()
        .filter_map(|(name, b)| pos(*b).map(|p| (name.to_string(), serde_json::json!(p))))
        .collect();
    let seq = AVATAR_DUMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let state = serde_json::json!({ "seq": seq, "joints": map });
    if let Ok(bytes) = serde_json::to_vec(&state) {
        atomic_write(&base_dir().join("debug_avatar.json"), &bytes);
    }
}

/// Live solver overrides. `None` = keep the app's normal value. Toggling
/// `arm_reach_ik` / `contact_ik` on the live app isolates which stage is
/// responsible for an arm artefact (e.g. the hands-together cross) without a
/// rebuild.
#[derive(Clone, Copy, Default)]
pub struct Tuning {
    pub arm_reach_ik: Option<bool>,
    pub contact_ik: Option<bool>,
    pub joint_confidence_threshold: Option<f32>,
}

static TUNING_CACHE: Mutex<(Option<SystemTime>, Tuning)> = Mutex::new((
    None,
    Tuning {
        arm_reach_ik: None,
        contact_ik: None,
        joint_confidence_threshold: None,
    },
));

/// Read `debug_tuning.json`, re-parsing only when its mtime changes. Returns the
/// default (all-`None`) tuning when the channel is off or the file is absent.
pub fn load_tuning() -> Tuning {
    if !enabled() {
        return Tuning::default();
    }
    let path = base_dir().join("debug_tuning.json");
    let mtime = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
    let mut cache = TUNING_CACHE.lock().unwrap();
    if mtime.is_some() && mtime == cache.0 {
        return cache.1;
    }
    cache.0 = mtime;
    cache.1 = std::fs::read(&path)
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .map(|v| Tuning {
            arm_reach_ik: v.get("arm_reach_ik").and_then(|x| x.as_bool()),
            contact_ik: v.get("contact_ik").and_then(|x| x.as_bool()),
            joint_confidence_threshold: v
                .get("joint_confidence_threshold")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32),
        })
        .unwrap_or_default();
    cache.1
}
