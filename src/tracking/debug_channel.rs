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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime};

use super::latest_cell::LatestCell;
use super::PoseEstimate;
use crate::asset::HumanoidBone;

fn base_dir() -> PathBuf {
    std::env::var_os("ProgramData")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\ProgramData"))
        .join("VulVATAR")
}

/// The master switch. The flag-file `stat` is cached and refreshed at
/// most every [`ENABLED_REFRESH_MS`] — the previous stat-per-call design
/// actually ran *several* filesystem stats per frame (`dump_observation`
/// + the face/mesh stashes + `load_tuning` each check independently) on
/// the tracking hot path. Toggling the flag file takes effect within the
/// refresh interval.
pub fn enabled() -> bool {
    const ENABLED_REFRESH_MS: u64 = 2000;
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    static LAST_CHECK_MS: AtomicU64 = AtomicU64::new(u64::MAX);
    static CACHED: AtomicBool = AtomicBool::new(false);

    let now_ms = EPOCH.get_or_init(Instant::now).elapsed().as_millis() as u64;
    let last = LAST_CHECK_MS.load(Ordering::Relaxed);
    if last == u64::MAX || now_ms.saturating_sub(last) >= ENABLED_REFRESH_MS {
        let value = base_dir().join("debug.on").exists();
        CACHED.store(value, Ordering::Relaxed);
        LAST_CHECK_MS.store(now_ms, Ordering::Relaxed);
        value
    } else {
        CACHED.load(Ordering::Relaxed)
    }
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

/// Face-stage diagnostics stashed by the RTMW3D face block (which has the
/// crop bbox + both pose candidates in scope) and merged into
/// `debug_state.json` by `dump_observation` (which does not). One slot,
/// overwritten per frame — the dump runs on the same worker right after.
static FACE_DEBUG: Mutex<Option<serde_json::Value>> = Mutex::new(None);

/// The five FaceMesh landmarks the pose derivation reads (nose tip,
/// eye outers, cheeks), in 256-px crop space — lets the overlay show
/// where the mesh model actually thinks those anatomical points are.
static MESH_LANDMARKS: Mutex<Option<[[f32; 2]; 5]>> = Mutex::new(None);

/// Stash the pose-relevant FaceMesh landmarks (crop space, 256 px).
/// Order: nose, right eye outer, left eye outer, right cheek, left cheek.
pub fn stash_mesh_landmarks(pts: [[f32; 2]; 5]) {
    if !enabled() {
        return;
    }
    if let Ok(mut slot) = MESH_LANDMARKS.lock() {
        *slot = Some(pts);
    }
}

/// Stash the face-crop bbox (image px), the FaceMesh score, and the two
/// head-pose candidates so the external overlay can show which estimator
/// won and whether the crop actually frames the face.
#[allow(clippy::too_many_arguments)]
pub fn stash_face_debug(
    bbox: Option<(f32, f32, f32)>,
    mesh_conf: Option<f32>,
    body_ypr: Option<(f32, f32, f32)>,
    mesh_ypr: Option<(f32, f32, f32)>,
) {
    if !enabled() {
        return;
    }
    let v = serde_json::json!({
        "bbox": bbox.map(|(x, y, s)| [x, y, s]),
        "mesh_c": mesh_conf,
        "body_ypr": body_ypr.map(|(y, p, r)| [y, p, r]),
        "mesh_ypr": mesh_ypr.map(|(y, p, r)| [y, p, r]),
        "mesh_lm": MESH_LANDMARKS.lock().ok().and_then(|mut s| s.take()),
    });
    if let Ok(mut slot) = FACE_DEBUG.lock() {
        *slot = Some(v);
    }
}

/// One observation queued for the background dump writer.
struct DumpJob {
    frame_index: u64,
    rgb: Vec<u8>,
    width: u32,
    height: u32,
    state: serde_json::Value,
}

/// Latest-only handoff to the `debug-dump` writer thread, started lazily
/// on the first enabled dump and left running for the process lifetime.
/// The camera down-scale, PNG-ish buffer packing and the two file writes
/// used to run synchronously on the tracking thread — a diagnostics
/// channel whose act of being enabled changed the jitter and latency it
/// was meant to observe. Latest-only semantics also mean a slow disk
/// drops intermediate debug frames instead of back-pressuring capture.
fn dump_cell() -> &'static Arc<LatestCell<DumpJob>> {
    static CELL: OnceLock<Arc<LatestCell<DumpJob>>> = OnceLock::new();
    CELL.get_or_init(|| {
        let cell = LatestCell::<DumpJob>::new();
        let worker = Arc::clone(&cell);
        let spawned = std::thread::Builder::new()
            .name("debug-dump".into())
            .spawn(move || {
                while let Some(job) = worker.take_blocking() {
                    write_camera(&job.rgb, job.width, job.height, 640, job.frame_index);
                    if let Ok(bytes) = serde_json::to_vec(&job.state) {
                        atomic_write(&base_dir().join("debug_state.json"), &bytes);
                    }
                }
            });
        if spawned.is_err() {
            log::warn!("debug_channel: could not spawn debug-dump writer thread");
        }
        cell
    })
}

/// Publish the current frame's camera + 2D keypoints + source arm joints for an
/// external overlay. No-op unless the debug flag file is present. The JSON
/// value is assembled here (it borrows the estimate); the pixel work and
/// file I/O happen on the background writer thread.
pub fn dump_observation(frame_index: u64, rgb: &[u8], w: u32, h: u32, est: &PoseEstimate) {
    if !enabled() {
        return;
    }

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
    // Face pose (head orientation) + the source torso/head 3D that drives the
    // neck/spine chain: lets an external tool tell whether an over-pitched
    // avatar head comes from the face track (pitch) or from a contaminated
    // shoulder→head geometry (the "face is forward but the head looks down"
    // case, which is coupled to the arms via the shoulder samples).
    let face = est.skeleton.face.map(|f| {
        serde_json::json!({ "yaw": f.yaw, "pitch": f.pitch, "roll": f.roll, "c": f.confidence })
    });
    let state = serde_json::json!({
        "frame": frame_index,
        "overall": est.skeleton.overall_confidence,
        "kp": kps,
        "face": face,
        // FaceMesh's own "is this a face" score, independent of the
        // published pose's confidence — tells an external tool which
        // head-pose source (mesh vs body ear-line) actually won.
        "mesh_c": est.skeleton.face_mesh_confidence,
        // Crop bbox + per-estimator pose candidates from the face stage
        // (stashed by the RTMW3D worker just before this dump).
        "face_dbg": FACE_DEBUG.lock().ok().and_then(|mut s| s.take()),
        "torso": {
            "Head": arm(HumanoidBone::Head),
            "Neck": arm(HumanoidBone::Neck),
            "UpperChest": arm(HumanoidBone::UpperChest),
            "LSh": arm(HumanoidBone::LeftShoulder),
            "RSh": arm(HumanoidBone::RightShoulder),
            "Hips": arm(HumanoidBone::Hips),
        },
        "arm": {
            "LUp": arm(HumanoidBone::LeftUpperArm),
            "LLo": arm(HumanoidBone::LeftLowerArm),
            "LHa": arm(HumanoidBone::LeftHand),
            "RUp": arm(HumanoidBone::RightUpperArm),
            "RLo": arm(HumanoidBone::RightLowerArm),
            "RHa": arm(HumanoidBone::RightHand),
        },
    });
    dump_cell().put(DumpJob {
        frame_index,
        rgb: rgb.to_vec(),
        width: w,
        height: h,
        state,
    });
}

static AVATAR_DUMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Publish the SOLVED avatar's key joint world positions (after
/// `compute_global_pose`) so the external overlay can draw the avatar skeleton
/// WITHOUT a GPU render — enough to see torso tilt, elbow placement, whole-body
/// rotation, etc. Overwrites `debug_avatar.json` each frame (a `seq` counter
/// lets a reader detect fresh frames). No-op unless the debug flag file exists.
/// `head_axes`, when present, are the Head bone's world-space X/Y/Z basis
/// vectors (normalised) — its facing. With the neck upright and the head
/// position UP, an avatar head that still looks DOWN reveals itself here as the
/// world Y (up) axis pitched forward / the Z axis pitched down, isolating a
/// face-track over-pitch from a neck-chain one.
pub fn dump_avatar_pose<F: Fn(HumanoidBone) -> Option<[f32; 3]>>(
    pos: F,
    head_axes: Option<[[f32; 3]; 3]>,
) {
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
    let state = serde_json::json!({ "seq": seq, "joints": map, "head_axes": head_axes });
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
    pub idle_arm_apose: Option<bool>,
    pub joint_confidence_threshold: Option<f32>,
}

static TUNING_CACHE: Mutex<(Option<SystemTime>, Tuning)> = Mutex::new((
    None,
    Tuning {
        arm_reach_ik: None,
        contact_ik: None,
        idle_arm_apose: None,
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
            idle_arm_apose: v.get("idle_arm_apose").and_then(|x| x.as_bool()),
            joint_confidence_threshold: v
                .get("joint_confidence_threshold")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32),
        })
        .unwrap_or_default();
    cache.1
}
