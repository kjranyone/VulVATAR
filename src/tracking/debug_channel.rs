//! Live debug channel — lets an external tool (`scratchpad/live_debug.py`)
//! observe the running app's tracking pipeline without stealing the camera.
//!
//! Off by default: every entry point is a no-op unless the flag file
//! `%ProgramData%\VulVATAR\debug.on` exists (one `exists()` stat per frame).
//! When on:
//!   * [`dump_observation`] overwrites `debug_camera.bin` (down-scaled camera
//!     RGBA behind the same 32-byte header the virtual-camera tap already
//!     parses) and `debug_state.json` (2D keypoints + source arm joints), so the
//!     external overlay can line up camera ↔ 2D ↔ avatar-output in one view.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

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
/// + the face/mesh stashes each check independently) on
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

/// Full-resolution aligned depth snapshot for offline pixel audits
/// ("is the depth under this keypoint actually right?") without stealing
/// the camera from the app. Layout: 32-byte header (`magic 'VDBD',
/// u32 w, u32 h, u32 fmt=1 (u16 mm), u32 pad, u64 frame_index@16`) then
/// `w*h` little-endian u16 millimetres, row-major top-down, 0 = no
/// return. ~1.8 MB at 1280×720 — callers throttle (once per second).
pub fn dump_depth_snapshot(frame_index: u64, depth_raw: &[u16], w: u32, h: u32, units_m: f32) {
    if !enabled() {
        return;
    }
    if depth_raw.len() < (w as usize * h as usize) || w == 0 || h == 0 {
        return;
    }
    let mm_per_unit = units_m * 1000.0;
    let mut out = vec![0u8; 32 + depth_raw.len() * 2];
    out[0..4].copy_from_slice(b"VDBD");
    out[4..8].copy_from_slice(&w.to_le_bytes());
    out[8..12].copy_from_slice(&h.to_le_bytes());
    out[12..16].copy_from_slice(&1u32.to_le_bytes());
    out[16..24].copy_from_slice(&frame_index.to_le_bytes());
    for (i, &raw) in depth_raw.iter().enumerate() {
        let mm = (raw as f32 * mm_per_unit)
            .round()
            .clamp(0.0, u16::MAX as f32) as u16;
        out[32 + i * 2..32 + i * 2 + 2].copy_from_slice(&mm.to_le_bytes());
    }
    atomic_write(&base_dir().join("debug_depth.bin"), &out);
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
/// Fusion estimator per-frame diagnostics, stashed by the provider just
/// before the dump (solve ms, cost, counts, re-acquisitions).
static RIG_DIAG: std::sync::Mutex<Option<serde_json::Value>> = std::sync::Mutex::new(None);

pub fn stash_rig_diag(v: serde_json::Value) {
    if let Ok(mut s) = RIG_DIAG.lock() {
        *s = Some(v);
    }
}

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
    // MCP knuckle keypoints of both hand landmark blocks (the joints
    // `attach_hand` needs 3-of-4 of) — lets an external audit correlate
    // the 2D hand detection with the depth snapshot without re-running
    // the detector. COCO-Wholebody: 91.. left block, 112.. right block.
    let mcp = |base: usize| -> Vec<serde_json::Value> {
        [5usize, 9, 13, 17]
            .iter()
            .filter_map(|&l| est.annotation.keypoints.get(base + l))
            .map(|&(x, y, s)| serde_json::json!([x, y, s]))
            .collect()
    };
    // `d` is the joint's raw camera-space depth in metres (the value the
    // window-median + person-band sampler actually returned) — the
    // ground truth for "did this keypoint get the right depth".
    let arm = |b: HumanoidBone| {
        est.skeleton.joints.get(&b).map(
            |j| serde_json::json!({ "p": j.position, "c": j.confidence, "d": j.metric_depth_m }),
        )
    };
    // Face pose (head orientation) + the source torso/head 3D that drives the
    // neck/spine chain: lets an external tool tell whether an over-pitched
    // avatar head comes from the face track (pitch) or from a contaminated
    // shoulder→head geometry (the "face is forward but the head looks down"
    // case, which is coupled to the arms via the shoulder samples).
    let face = est.skeleton.face.map(|f| {
        serde_json::json!({ "yaw": f.yaw, "pitch": f.pitch, "roll": f.roll, "c": f.confidence })
    });
    // Tracking-v2 rig summary (quality, σ of key bones, root) when the
    // fusion estimator produced this sample.
    let rig = est.skeleton.rig.as_ref().map(|r| {
        let b = |bone: HumanoidBone| {
            r.bones
                .get(&bone)
                .map(|x| serde_json::json!({ "sigma": x.sigma, "data_sigma": x.data_sigma }))
        };
        let db = |r: &crate::tracking::fusion::output::RigPose, bone: HumanoidBone| {
            r.bones.get(&bone).map(|x| {
                serde_json::json!({
                    "sigma": x.sigma,
                    "delta_world": x.delta_world,
                })
            })
        };
        serde_json::json!({
            "t": r.t,
            "quality": r.quality,
            "shape_confidence": r.shape_confidence,
            "root_cam_m": r.root_cam_m,
            "root_sigma_m": r.root_sigma_m,
            "hand_confidence": r.hand_confidence,
            "shoulder_span_m": r.shoulder_span_m,
            "head": r.bones.get(&HumanoidBone::Head).map(|x| {
                serde_json::json!({
                    "sigma": x.sigma,
                    "data_sigma": x.data_sigma,
                    // World-delta quaternion [x,y,z,w] the retarget applies
                    // (viewer frame): live rig↔avatar transfer audits read
                    // this against debug_avatar.json's head_axes.
                    "delta_world": x.delta_world,
                })
            }),
            // Spine-chain deltas (same contract as `head`) for the recline
            // localization: which trunk delta pitches the avatar's
            // Chest→Neck segment back when the person sits upright.
            "hips": db(r, HumanoidBone::Hips),
            "spine": db(r, HumanoidBone::Spine),
            "chest": db(r, HumanoidBone::Chest),
            "upper_chest": db(r, HumanoidBone::UpperChest),
            "neck": db(r, HumanoidBone::Neck),
            "upper_chest_sigma": b(HumanoidBone::UpperChest),
            "l_upper_arm": b(HumanoidBone::LeftUpperArm),
            "r_upper_arm": b(HumanoidBone::RightUpperArm),
            "l_hand": b(HumanoidBone::LeftHand),
            "r_hand": b(HumanoidBone::RightHand),
            "diag": RIG_DIAG.lock().ok().and_then(|mut s| s.take()),
        })
    });
    let state = serde_json::json!({
        "frame": frame_index,
        "overall": est.skeleton.overall_confidence,
        "rig": rig,
        "kp": kps,
        "kp_mcp": { "l_block": mcp(91), "r_block": mcp(112) },
        // Root / anchor channel (drives avatar translation + scale).
        "root": est.skeleton.root_offset,
        "root_is_hip": est.skeleton.root_anchor_is_hip,
        "metric": est.skeleton.metric_frame_info.as_ref().map(|m| serde_json::json!({
            "anchor_cam_m": m.anchor_cam_m,
            "anchor_is_hip": m.anchor_is_hip,
            "mpsu": m.mpsu,
            "ref_span_m": m.reference_span_m,
        })),
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

/// CPU-LBS bbox of one cloth-target primitive at ONE deformation stage
/// (R5). The pair of stages (pre/post physics) lets an external watcher
/// attribute a broken frame to a stage instead of inferring it.
#[derive(Clone, Debug)]
pub struct CostumePrimProbe {
    pub mesh: String,
    pub primitive: u64,
    /// Clearance anchor parent primitive (`body_primitive_id`), if the
    /// asset assigned one — the §10 "対応表" requirement: screen
    /// positions must NOT be mapped to ids by guesswork.
    pub clearance_parent: Option<u64>,
    /// Containment anchor parent primitive
    /// (`containment_primitive_id`).
    pub containment_parent: Option<u64>,
    /// `None` when the primitive has no CPU vertex payload or was not
    /// found on the avatar.
    pub pre_physics: Option<CostumeBBox>,
    pub post_physics: Option<CostumeBBox>,
}

/// Min/max world-space corner pair from a CPU LBS skinning pass.
#[derive(Clone, Copy, Debug)]
pub struct CostumeBBox {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

/// Costume-health probe: world positions of spring-driven garment bones
/// (skirt chains, tail), taken pre- and post-physics, plus CPU-LBS
/// bounding boxes of every cloth-target primitive at both stages.
/// Written to `debug_avatar_extra.json`.
///
/// R5 scope contract — what this probe measures and what it CANNOT:
/// the bboxes come from a CPU **linear blend skinning** pass over the
/// rest mesh. They do NOT include GPU dual-quaternion skinning
/// differences, the GPU cloth SSBO override, the clearance /
/// containment render-side corrections, or the final drawn VBO. A
/// healthy bbox here proves the pose + CPU skinning stage is sane; it
/// does NOT prove healthy drawn vertices (a GPU-side garment tear is
/// invisible to this probe). Use it to split
/// "solver/pose side is wrong" from "the numbers are healthy but the
/// pixels are not" — never to certify the render path.
///
/// No-op unless the debug flag file exists.
pub fn dump_costume_probe(
    bones: Vec<(String, [f32; 3])>,
    bones_post_physics: Vec<(String, [f32; 3])>,
    prim_probes: Vec<CostumePrimProbe>,
    cloth_deform_count: usize,
    cloth_targets: Vec<crate::asset::PrimitiveId>,
    sim_substeps: u32,
    fixed_dt: f32,
) {
    if !enabled() {
        return;
    }
    let seq = AVATAR_DUMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let bone_map = |bones: Vec<(String, [f32; 3])>| {
        serde_json::Value::Object(
            bones
                .into_iter()
                .map(|(name, p)| (name, serde_json::json!(p)))
                .collect(),
        )
    };
    let state = serde_json::json!({
        "seq": seq,
        // `sim_substeps == 0` marks a frozen-physics frame: the post
        // stage then equals the pre stage BY CONTRACT, not by accident.
        "sim_substeps": sim_substeps,
        "fixed_dt": fixed_dt,
        "bones": bone_map(bones),
        "bones_post_physics": bone_map(bones_post_physics),
        "cloth_deforms": cloth_deform_count,
        "cloth_targets": cloth_targets.iter().map(|t| t.0).collect::<Vec<_>>(),
        "cpu_lbs_prim_bbox": {
            "measured": "CPU LBS skinning only — no GPU DQS, no cloth SSBO override, no clearance/containment, no final VBO",
            "prims": prim_probes
                .iter()
                .map(|p| serde_json::json!({
                    "mesh": p.mesh,
                    "primitive": p.primitive,
                    "clearance_parent": p.clearance_parent,
                    "containment_parent": p.containment_parent,
                    "pre_physics": p.pre_physics.map(|b| serde_json::json!({"min": b.min, "max": b.max})),
                    "post_physics": p.post_physics.map(|b| serde_json::json!({"min": b.min, "max": b.max})),
                }))
                .collect::<Vec<_>>(),
        },
    });
    if let Ok(bytes) = serde_json::to_vec(&state) {
        atomic_write(&base_dir().join("debug_avatar_extra.json"), &bytes);
    }
}

/// One anchor-bearing primitive's final-VBO audit stats (R2), mirrored
/// from `renderer::VboAuditEntry` so the tracking channel does not
/// depend on the renderer module.
#[derive(Clone, Copy, Debug)]
pub struct VboAuditRow {
    pub mesh: u64,
    pub primitive: u64,
    pub instance: Option<u64>,
    pub vertex_count: usize,
    pub nan_count: usize,
    pub max_correction_m: f32,
    pub p95_correction_m: f32,
    pub max_pos_len_m: f32,
}

/// R2 final-VBO audit dump: per-primitive render-side correction
/// telemetry (the `position.w` channel `transform_cs` publishes),
/// written to `debug_vbo_audit.json`. Populated only when
/// `VULVATAR_VBO_AUDIT=1` — the readback copy does not exist otherwise.
/// Use it to detect clearance/containment runaway corrections and NaN
/// vertices that the CPU-side probes cannot see (they measure LBS
/// skinning, not the GPU path).
///
/// No-op unless the debug flag file exists.
pub fn dump_vbo_audit(rows: Vec<VboAuditRow>, timestamp_nanos: u64) {
    if !enabled() || rows.is_empty() {
        return;
    }
    let state = serde_json::json!({
        "timestamp_nanos": timestamp_nanos,
        "note": "corrections are render-only (not written back to physics); max = MAX_RENDER_CORRECTION_M (0.25 m) means the clamp saturated",
        "prims": rows
            .iter()
            .map(|r| serde_json::json!({
                "mesh": r.mesh,
                "primitive": r.primitive,
                "instance": r.instance,
                "vertex_count": r.vertex_count,
                "nan_count": r.nan_count,
                "max_correction_m": r.max_correction_m,
                "p95_correction_m": r.p95_correction_m,
                "max_pos_len_m": r.max_pos_len_m,
            }))
            .collect::<Vec<_>>(),
    });
    if let Ok(bytes) = serde_json::to_vec(&state) {
        atomic_write(&base_dir().join("debug_vbo_audit.json"), &bytes);
    }
}

/// GUI-thread heartbeat: the handful of raw flags that decide whether the
/// per-frame pipeline runs at all. Written from `GuiApp::update` *outside*
/// every gate, so its `seq` advances whenever the GUI is alive regardless
/// of what is switched off downstream.
///
/// Exists because "the avatar is frozen while tracking still runs" has at
/// least two indistinguishable causes from the outside — the frame loop
/// being paused, and there being no avatar to pose — and both leave the
/// same fingerprint on every other artefact the app writes. Reporting the
/// flags directly removes the guesswork: read `debug_gui.json` and the
/// answer is a value, not an inference.
///
/// No-op unless the debug flag file exists.
pub fn dump_gui_heartbeat(
    paused: bool,
    avatars_loaded: usize,
    tracking_enabled: bool,
    frame_count: u64,
    sim_substeps: u32,
    panel_hole: Option<[f32; 3]>,
    render_fps: Option<f32>,
    render_submit_drops: u64,
    render_cpu_ms: Option<f32>,
) {
    if !enabled() {
        return;
    }
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let state = serde_json::json!({
        "seq": SEQ.fetch_add(1, Ordering::Relaxed),
        // `false` here and a stalled `debug_avatar.json` means the frame
        // loop is running but produced no avatar — look at `avatars_loaded`.
        "paused": paused,
        // Zero means nothing is posed however healthy tracking looks.
        "avatars_loaded": avatars_loaded,
        "tracking_enabled": tracking_enabled,
        // Only advances on unpaused frames — the direct counterpart to
        // `seq`, which advances on every frame. seq climbing while
        // frame_count holds still IS the paused signature.
        "frame_count": frame_count,
        // Substeps the fixed-step sim clock yielded on the previous
        // unpaused frame. Zero means the spring solver did not run that
        // frame — correlate a one-frame hair clip with this before
        // suspecting the solver.
        "sim_substeps": sim_substeps,
        // Live egui-0.30 side-panel layout hole measurement (the
        // "black band beside the inspector" bug): `[frame_right,
        // cursor_left, width]` in points, null when the layout is
        // tight. A persistently non-null value means some inspector
        // widget still overflows the panel width.
        "panel_hole": panel_hole,
        // Render thread's own production rate (fps, active frames
        // only; null = no fresh measurement — paused / no avatar /
        // just resumed). Persistently below the output target while
        // `seq` climbs is the "GUI spins faster than the renderer can
        // draw" signature; the egui-side FPS display cannot see this
        // because backpressure makes it tick *faster*.
        "render_fps": render_fps,
        // Cumulative RenderFrame submits rejected because the render
        // thread's bounded command channel (capacity 2) was full.
        // Climbing ~1 per GUI frame = sustained backpressure; this is
        // the counter behind the (now rate-limited) "command queue
        // full" warning.
        "render_submit_drops": render_submit_drops,
        // CPU-side render() duration EMA (fence wait included). At the
        // frame budget while render_fps sags = GPU-bound; small while
        // fps sags = recording path itself is the cost.
        "render_cpu_ms": render_cpu_ms,
    });
    if let Ok(bytes) = serde_json::to_vec(&state) {
        atomic_write(&base_dir().join("debug_gui.json"), &bytes);
    }
}
