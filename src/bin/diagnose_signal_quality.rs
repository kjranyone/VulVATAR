//! Per-channel temporal signal-quality harness: replays a recorded frame
//! sequence through `vulvatar_lib::app::Application::run_frame` — the
//! SAME per-frame update function the live GUI calls, headless
//! (`render_thread: None`, `tracking_worker: None`; see the existing
//! `fade_keeps_avatar_opaque_until_tracking_worker_runs` unit test in
//! `src/app/mod.rs` for precedent that this is a supported, GPU-free
//! mode). Frames are pushed into `Application::tracking`'s mailbox the
//! same way the real tracking worker does
//! (`TrackingMailbox::publish_estimate`), so `run_frame` sees exactly
//! the same calibration application, `SolverParams` construction, and
//! solver-state plumbing a live session would — no hand-copied shadow
//! of that logic to drift out of sync. Derives, for every channel it
//! can see, the numeric proxies for "does this look twitchy / does it
//! hold-then-snap" that a human eye judges instantly but a
//! single-frame error metric cannot show:
//!
//!   - velocity / acceleration / jerk (vector, not speed-scalar, so an
//!     in-place back-and-forth oscillation shows up as high jerk even
//!     though its SPEED magnitude stays roughly constant)
//!   - high-frequency jitter (residual after a 3-frame EMA) and
//!     low-frequency wander (residual between a 3-frame and 30-frame
//!     EMA)
//!   - confidence-gated state flapping (tracked/lost hysteresis
//!     mirroring `arm_ray_ik`'s `oof_streak`: engage lost after 2
//!     consecutive sub-threshold frames, release after 3 consecutive
//!     clear frames — see docs/onnx-tracking-pipeline.md §"Detector
//!     hysteresis")
//!   - the position jump measured at the first tracked frame after a
//!     lost -> tracked transition ("recovery snap")
//!
//! Channels covered: every core `HumanoidBone` on BOTH the raw source
//! skeleton (SRC, confidence-gated) and the solved avatar bone (AV,
//! post ray-IK / solver, always present) — this split is what lets you
//! attribute a given wobble to the detector vs. the solver — plus
//! `root_offset`, head yaw/pitch/roll, and every expression weight seen.
//!
//! `FrameConfig` toggles: `hand_tracking_enabled` / `face_tracking_enabled`
//! are forced ON here even though the GUI's own fresh-session default is
//! hand OFF (`src/gui/mod.rs`'s `TrackingGuiState` initialiser) — with it
//! off the wrist bones are left at whatever the LowerArm chain's
//! shortest-arc settled on, not driven by tracked data, which would make
//! the LeftHand/RightHand channels this tool exists to inspect carry no
//! signal at all. This matches `SolverParams::default`'s own documented
//! intent ("non-GUI callers … get the full pipeline"). `lower_body_
//! tracking_enabled: false` matches the GUI default and the upper-body
//! webcam framing these diagnostics are recorded against.
//!
//! Per-frame pacing: real elapsed wall-clock time is floored at `1/fps`
//! (sleeping the remainder), mirroring the tracking worker's own
//! `frame_interval` sleep (`src/tracking/mod.rs`). `solve_avatar_pose`'s
//! temporal filters (1€ filter, `rotation_blend`) key off real
//! `Instant::now()` elapsed time, not any nominal fps value passed
//! around — an unpaced replay can produce a smaller dt than any real
//! camera could ever deliver, which the 1€ filter would treat as
//! genuine information. Note this only bounds dt from BELOW: it does
//! not reproduce the live render loop's decoupled cadence, which can
//! call `run_frame` several times against the same tracking sample
//! when rendering outpaces tracking — that multi-solve-per-sample
//! convergence effect is a known, accepted simplification here.
//!
//! This is a measurement tool, not a judgement: it does not label
//! anything "good" or "bad" on its own. Read it against the recorded
//! footage and the render composites from `diagnose_video_replay`.
//!
//! Usage:
//!   cargo run --bin diagnose_signal_quality -- [frames_dir] [out_dir] [fps]
//!
//! Emits:
//!   <out_dir>/frames.jsonl  — one JSON object per frame, all channels
//!   <out_dir>/summary.md    — per-channel metrics tables, worst-first

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

use vulvatar_lib::app::{Application, FrameConfig, RuntimeToggles};
use vulvatar_lib::asset::vrm::VrmAssetLoader;
use vulvatar_lib::asset::HumanoidBone;
use vulvatar_lib::avatar::{AvatarInstance, AvatarInstanceId};
use vulvatar_lib::tracking::provider::create_pose_provider;
use vulvatar_lib::tracking::{stagelog, MouthSource, TrackingSmoothingParams};

/// Bones tracked per-channel. Fingers are excluded to keep the report
/// legible; hand stability is covered by LeftHand/RightHand (the wrist).
const CORE_BONES: [HumanoidBone; 19] = [
    HumanoidBone::Hips,
    HumanoidBone::Spine,
    HumanoidBone::Chest,
    HumanoidBone::Neck,
    HumanoidBone::Head,
    HumanoidBone::LeftShoulder,
    HumanoidBone::LeftUpperArm,
    HumanoidBone::LeftLowerArm,
    HumanoidBone::LeftHand,
    HumanoidBone::RightShoulder,
    HumanoidBone::RightUpperArm,
    HumanoidBone::RightLowerArm,
    HumanoidBone::RightHand,
    HumanoidBone::LeftUpperLeg,
    HumanoidBone::LeftLowerLeg,
    HumanoidBone::LeftFoot,
    HumanoidBone::RightUpperLeg,
    HumanoidBone::RightLowerLeg,
    HumanoidBone::RightFoot,
];

/// Matches the MCP confidence floor used by `attach_hand` / ray-IK
/// (docs/onnx-tracking-pipeline.md §5) so "tracked" means the same
/// thing here as it does in the production gate.
const CONF_THRESHOLD: f32 = 0.3;
/// Frames below threshold before a channel flips to "lost" — matches
/// `oof_streak`'s engage count.
const LOST_ENGAGE: u32 = 2;
/// Frames at/above threshold before a channel flips back to "tracked".
const TRACKED_RELEASE: u32 = 3;

const EMA_SHORT_SPAN: f32 = 3.0;
const EMA_LONG_SPAN: f32 = 30.0;

#[derive(Clone, Copy)]
struct Sample {
    value: [f32; 3],
    dims: usize,
    confidence: f32,
}

struct Channel {
    name: String,
    has_confidence: bool,
    samples: Vec<Option<Sample>>,
}

#[derive(Default, Clone, Copy)]
struct Metrics {
    coverage: f32,
    mean_conf: f32,
    vel_rms: f32,
    vel_max: f32,
    acc_rms: f32,
    acc_max: f32,
    jerk_rms: f32,
    jerk_max: f32,
    jitter_hf_rms: f32,
    wander_lf_rms: f32,
    flap_count: u32,
    recovery_count: u32,
    recovery_jump_max: f32,
    recovery_jump_mean: f32,
}

fn main() -> Result<(), String> {
    env_logger::init();
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "diagnostics/video_frames".to_string());
    let out_dir = PathBuf::from(
        std::env::args()
            .nth(2)
            .unwrap_or_else(|| "diagnostics/signal_quality".to_string()),
    );
    let fps: f32 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30.0);
    let dt = 1.0 / fps.max(1.0);
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("mkdir {}: {e}", out_dir.display()))?;

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
    let total_frames = files.len();
    eprintln!("{total_frames} frames @ {fps} fps (dt={dt:.4}s)");

    // Crash forensics: if the *previous* run of this binary (or any
    // tracking session) died without a clean exit, its sentinel is
    // still here — surface it before we potentially do the same.
    // `begin_session` below overwrites the sentinel with our own, so
    // check first. See module docs on `stagelog` for why this exists:
    // a hard freeze on this GPU/driver combo leaves zero evidence
    // otherwise (2026-06-11/12 Arc B570 incident, docs in
    // src/gpu_coordination.rs).
    if let Some(stale) = stagelog::stale_sentinel() {
        eprintln!("!! previous session exited uncleanly, sentinel: {stale}");
    }
    let _session = stagelog::SessionGuard::begin("diagnose_signal_quality");

    let depth_enabled = std::env::var("VULVATAR_REPLAY_NO_DEPTH").is_err();
    let config = vulvatar_lib::tracking::provider::TrackingPipelineConfig {
        depth_enabled,
        ..Default::default()
    };
    eprintln!("creating pose provider (DirectML session init — the known-risky step)…");
    stagelog::mark(0, "provider_load_begin");
    let mut provider = create_pose_provider("models", config)?;
    stagelog::mark(0, "provider_load_end");
    eprintln!("pose provider created");
    let _ = provider.take_load_warnings();

    stagelog::mark(0, "vrm_load_begin");
    let vrm = std::env::var("VULVATAR_VRM")
        .unwrap_or_else(|_| "sample_data/AvatarSample_A.vrm".to_string());
    let asset = VrmAssetLoader::new()
        .load(&vrm)
        .map_err(|e| format!("load VRM: {e:?}"))?;
    stagelog::mark(0, "vrm_load_end");
    let humanoid_map = asset.humanoid.as_ref().ok_or("no humanoid")?;

    // Headless production app: no render thread, no camera worker.
    // `run_frame` gracefully no-ops the GPU-submission tail when
    // `render_thread` is `None` (guarded `if let Some(ref rt) =
    // self.render_thread` in src/app/render.rs) — the same pattern
    // `fade_keeps_avatar_opaque_until_tracking_worker_runs` in
    // src/app/mod.rs exercises in a unit test.
    let mut app = Application::new();
    app.set_avatar(AvatarInstance::new(AvatarInstanceId(1), Arc::clone(&asset)));
    // `run_frame` early-returns unless `running` is set (src/app/render.rs
    // `if !self.running { return; }`). `Application::new()` leaves it false
    // — the live app flips it true on start. Without this the solver never
    // runs and every avatar bone stays at the default zero transform (the
    // AV channels read all-`[0,0,0]`).
    app.running = true;

    // `run_frame` only reads the tracking mailbox when a `TrackingWorker`
    // is attached AND running (`step_tracking`, src/app/render.rs) — a
    // deliberate gate so a freshly loaded avatar can't fade out before
    // tracking starts. `new_external` satisfies that gate without
    // spawning a second (redundant, camera-less) capture thread; we
    // drive the same shared mailbox ourselves below.
    let external_worker =
        vulvatar_lib::tracking::TrackingWorker::new_external(app.tracking.shared_mailbox());
    app.tracking_worker = Some(external_worker);

    let frame_config = FrameConfig {
        toggles: RuntimeToggles::default(),
        smoothing: TrackingSmoothingParams::default(),
        material_mode_index: 0,
        hand_tracking_enabled: true,
        face_tracking_enabled: true,
        lower_body_tracking_enabled: false,
        root_translation_enabled: true,
        fade_on_tracking_loss: false,
        mouth_source: MouthSource::Image,
        spring_tuning: vulvatar_lib::simulation::spring::SpringTuning::default(),
        scene_gravity: vulvatar_lib::simulation::SceneGravity::default(),
        frame_dt: dt,
    };
    let frame_interval = Duration::from_secs_f32(dt);

    let mut channels: Vec<Channel> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut frames_out = std::io::BufWriter::new(
        std::fs::File::create(out_dir.join("frames.jsonl")).map_err(|e| e.to_string())?,
    );

    // Durable crash-forensic log. Distinct from `stagelog` (which only
    // `flush()`es to the OS page cache — lost on a hard GPU-TDR reboot,
    // as confirmed by the null-byte tail on logs/tracking_*.log after
    // the 2026-07-01 crashes). Every marker here is `sync_all()`'d to
    // physical disk, so after a hard reboot the last line names exactly
    // which frame's inference / solve was in flight when the machine
    // died. The confirmed culprit is the Intel Arc driver igdkmdnd64.sys
    // (dxgkrnl 0x116 TDR) — this log exists to VERIFY a driver-side fix,
    // by pinpointing the surviving frame count run-over-run.
    let mut forensic = std::fs::File::create(out_dir.join("forensic.log"))
        .map_err(|e| format!("create forensic.log: {e}"))?;
    forensic_mark(
        &mut forensic,
        "session_begin — GPU work is estimate_pose(); culprit driver igdkmdnd64.sys",
    );

    for (i, f) in files.iter().enumerate() {
        let loop_start = Instant::now();
        stagelog::mark(i as u64, "frame_begin");
        let img = image::open(f)
            .map_err(|e| format!("open {}: {e}", f.display()))?
            .to_rgb8();
        let (w, h) = (img.width(), img.height());
        // ENTER is fsync'd BEFORE the GPU submission, so a driver hang
        // during this call leaves "f{i} estimate_pose ENTER" as the
        // durable last line — the frame that killed the machine.
        forensic_mark(&mut forensic, &format!("f{i} estimate_pose ENTER"));
        let est = provider.estimate_pose(img.as_raw(), w, h, i as u64);
        let sk = est.skeleton.clone();
        forensic_mark(
            &mut forensic,
            &format!("f{i} estimate_pose EXIT overall={:.2}", sk.overall_confidence),
        );

        // Publish exactly as the real tracking worker does
        // (`TrackingMailbox::publish_estimate`, src/tracking/mod.rs)
        // so `run_frame` reads it back as a fresh sample.
        app.tracking.mailbox().publish_estimate(est, None);

        stagelog::mark(i as u64, "solve_begin");
        forensic_mark(&mut forensic, &format!("f{i} run_frame ENTER"));
        app.run_frame(&frame_config);
        forensic_mark(&mut forensic, &format!("f{i} run_frame EXIT"));
        stagelog::mark(i as u64, "solve_end");

        let av_pos = |b: HumanoidBone| -> Option<[f32; 3]> {
            let avatar = app.active_avatar()?;
            let idx = humanoid_map.bone_map.get(&b).copied().map(|n| n.0 as usize)?;
            let m = avatar.pose.global_transforms.get(idx)?;
            Some([m[3][0], m[3][1], m[3][2]])
        };

        for b in CORE_BONES {
            if let Some(j) = sk.joints.get(&b) {
                let sample = Sample { value: j.position, dims: 3, confidence: j.confidence };
                record(&mut channels, &mut index, total_frames, i, &format!("src.{b:?}"), true, sample);
            }
            if let Some(p) = av_pos(b) {
                let sample = Sample { value: p, dims: 3, confidence: 1.0 };
                record(&mut channels, &mut index, total_frames, i, &format!("av.{b:?}"), false, sample);
            }
        }
        if let Some(ro) = sk.root_offset {
            let sample = Sample { value: ro, dims: 3, confidence: 1.0 };
            record(&mut channels, &mut index, total_frames, i, "src.RootOffset", true, sample);
        }
        if let Some(face) = sk.face {
            for (suffix, val) in [("yaw", face.yaw), ("pitch", face.pitch), ("roll", face.roll)] {
                let sample = Sample { value: [val, 0.0, 0.0], dims: 1, confidence: face.confidence };
                record(&mut channels, &mut index, total_frames, i, &format!("src.Face.{suffix}"), true, sample);
            }
        }
        let expr_conf = sk.face_mesh_confidence.unwrap_or(sk.overall_confidence);
        for expr in &sk.expressions {
            let sample = Sample { value: [expr.weight, 0.0, 0.0], dims: 1, confidence: expr_conf };
            record(&mut channels, &mut index, total_frames, i, &format!("src.Expr.{}", expr.name), true, sample);
        }

        let mut obj = Map::new();
        obj.insert("frame".into(), json!(i));
        for ch in &channels {
            if let Some(s) = ch.samples[i] {
                obj.insert(ch.name.clone(), json!({ "v": s.value[0..s.dims].to_vec(), "c": s.confidence }));
            }
        }
        writeln!(frames_out, "{}", Value::Object(obj)).map_err(|e| e.to_string())?;
        if i % 100 == 0 {
            eprintln!("frame {i}");
        }

        // Floor real elapsed time at the nominal frame period so
        // `solve_avatar_pose`'s `Instant::now()`-based dt never sees a
        // smaller gap than a real camera could deliver (mirrors
        // `src/tracking/mod.rs`'s `frame_interval` sleep). When a
        // frame's own processing already exceeds `dt` this is a no-op,
        // exactly like the live pipeline falling behind under load.
        let elapsed = loop_start.elapsed();
        if elapsed < frame_interval {
            std::thread::sleep(frame_interval - elapsed);
        }
    }

    stagelog::mark(total_frames as u64, "all_frames_done");
    let mut rows: Vec<(String, bool, Metrics)> = channels
        .iter()
        .map(|ch| (ch.name.clone(), ch.has_confidence, compute_metrics(ch, dt)))
        .filter(|(_, _, m)| m.coverage > 0.0)
        .collect();
    write_summary(&out_dir.join("summary.md"), &dir, total_frames, fps, dt, &mut rows)?;
    eprintln!("wrote: {}", out_dir.display());
    Ok(())
}

/// Append one line to the durable forensic log and `sync_all()` it to
/// physical disk before returning. Unlike a `flush()` (which only hands
/// the bytes to the OS page cache, lost on a hard reboot), `sync_all`
/// guarantees the line survives a GPU-TDR bugcheck — so the on-disk tail
/// pinpoints the exact frame/phase in flight when the machine died.
/// Errors are swallowed: forensics must never abort the run.
fn forensic_mark(f: &mut std::fs::File, msg: &str) {
    let _ = writeln!(f, "{msg}");
    let _ = f.sync_all();
}

fn channel_idx(
    channels: &mut Vec<Channel>,
    index: &mut HashMap<String, usize>,
    name: &str,
    has_confidence: bool,
    total_frames: usize,
) -> usize {
    if let Some(&idx) = index.get(name) {
        return idx;
    }
    let idx = channels.len();
    channels.push(Channel {
        name: name.to_string(),
        has_confidence,
        samples: vec![None; total_frames],
    });
    index.insert(name.to_string(), idx);
    idx
}

/// Look up (or create) `name`'s channel and record `sample` at `frame`.
/// Every per-frame channel write in `main` is "get-or-create index, then
/// set `samples[frame]`" — this is the single place that idiom lives.
fn record(
    channels: &mut Vec<Channel>,
    index: &mut HashMap<String, usize>,
    total_frames: usize,
    frame: usize,
    name: &str,
    has_confidence: bool,
    sample: Sample,
) {
    let idx = channel_idx(channels, index, name, has_confidence, total_frames);
    channels[idx].samples[frame] = Some(sample);
}

fn norm(v: [f32; 3], dims: usize) -> f32 {
    (0..dims).map(|d| v[d] * v[d]).sum::<f32>().sqrt()
}

fn dist(a: [f32; 3], b: [f32; 3], dims: usize) -> f32 {
    let mut d = [0.0f32; 3];
    for k in 0..dims {
        d[k] = a[k] - b[k];
    }
    norm(d, dims)
}

fn rms(v: &[f32]) -> f32 {
    if v.is_empty() {
        0.0
    } else {
        (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
    }
}

fn maxf(v: &[f32]) -> f32 {
    v.iter().cloned().fold(0.0f32, f32::max)
}

/// Pooled raw magnitude samples across every tracked segment of a
/// channel. Kept as one struct (rather than five parallel `Vec`s) so
/// `accumulate_segment_stats` has a single out-param and the rms/max
/// reduction into `Metrics` lives in exactly one place
/// ([`Self::apply_to`]) instead of being copy-pasted at every call site.
#[derive(Default)]
struct MotionSamples {
    vel: Vec<f32>,
    acc: Vec<f32>,
    jerk: Vec<f32>,
    hf: Vec<f32>,
    lf: Vec<f32>,
}

impl MotionSamples {
    fn apply_to(&self, m: &mut Metrics) {
        m.vel_rms = rms(&self.vel);
        m.vel_max = maxf(&self.vel);
        m.acc_rms = rms(&self.acc);
        m.acc_max = maxf(&self.acc);
        m.jerk_rms = rms(&self.jerk);
        m.jerk_max = maxf(&self.jerk);
        m.jitter_hf_rms = rms(&self.hf);
        m.wander_lf_rms = rms(&self.lf);
    }
}

/// Accumulates vector velocity/acceleration/jerk magnitudes and the
/// EMA jitter/wander decomposition over one CONTINUOUS run of samples
/// (no gaps inside `seg` — gap-crossing deltas are handled separately
/// as "recovery jumps" so they don't poison the smoothness stats) into
/// `out`. Call once per tracked segment; `out` pools across segments.
///
/// Acceleration/jerk are derived from the velocity VECTOR, not the
/// speed scalar: an in-place back-and-forth oscillation keeps speed
/// roughly constant while the direction flips every frame, which a
/// speed-derivative would miss entirely but a vector derivative
/// correctly reports as a large acceleration spike.
fn accumulate_segment_stats(seg: &[([f32; 3], usize)], dt: f32, out: &mut MotionSamples) {
    let n = seg.len();
    if n < 2 {
        return;
    }
    let dims = seg[0].1;

    let mut vvec: Vec<[f32; 3]> = Vec::with_capacity(n - 1);
    for k in 1..n {
        let mut v = [0.0f32; 3];
        for d in 0..dims {
            v[d] = (seg[k].0[d] - seg[k - 1].0[d]) / dt;
        }
        out.vel.push(norm(v, dims));
        vvec.push(v);
    }
    if vvec.len() >= 2 {
        let mut avec: Vec<[f32; 3]> = Vec::with_capacity(vvec.len() - 1);
        for k in 1..vvec.len() {
            let mut a = [0.0f32; 3];
            for d in 0..dims {
                a[d] = (vvec[k][d] - vvec[k - 1][d]) / dt;
            }
            out.acc.push(norm(a, dims));
            avec.push(a);
        }
        for k in 1..avec.len() {
            let mut j = [0.0f32; 3];
            for d in 0..dims {
                j[d] = (avec[k][d] - avec[k - 1][d]) / dt;
            }
            out.jerk.push(norm(j, dims));
        }
    }

    let alpha_s = 2.0 / (EMA_SHORT_SPAN + 1.0);
    let alpha_l = 2.0 / (EMA_LONG_SPAN + 1.0);
    let mut ema_s = seg[0].0;
    let mut ema_l = seg[0].0;
    for k in 1..n {
        for d in 0..dims {
            ema_s[d] = alpha_s * seg[k].0[d] + (1.0 - alpha_s) * ema_s[d];
            ema_l[d] = alpha_l * seg[k].0[d] + (1.0 - alpha_l) * ema_l[d];
        }
        out.hf.push(dist(seg[k].0, ema_s, dims));
        out.lf.push(dist(ema_s, ema_l, dims));
    }
}

fn compute_metrics(ch: &Channel, dt: f32) -> Metrics {
    let n = ch.samples.len();
    let present = ch.samples.iter().filter(|s| s.is_some()).count();
    let coverage = present as f32 / n.max(1) as f32;
    let confs: Vec<f32> = ch.samples.iter().filter_map(|s| s.map(|s| s.confidence)).collect();
    let mean_conf = if confs.is_empty() { 0.0 } else { confs.iter().sum::<f32>() / confs.len() as f32 };
    let mut m = Metrics { coverage, mean_conf, ..Default::default() };

    if !ch.has_confidence {
        let vals: Vec<([f32; 3], usize)> = ch.samples.iter().filter_map(|s| s.map(|s| (s.value, s.dims))).collect();
        let mut samples = MotionSamples::default();
        accumulate_segment_stats(&vals, dt, &mut samples);
        samples.apply_to(&mut m);
        return m;
    }

    // Confidence-gated tracked/lost state machine, mirroring
    // `arm_ray_ik`'s `oof_streak` hysteresis (see module doc).
    let mut states = vec![false; n];
    let (mut below, mut above, mut tracked) = (0u32, 0u32, false);
    for i in 0..n {
        let conf = ch.samples[i].map(|s| s.confidence).unwrap_or(0.0);
        if conf >= CONF_THRESHOLD {
            above += 1;
            below = 0;
            if !tracked && above >= TRACKED_RELEASE {
                tracked = true;
            }
        } else {
            below += 1;
            above = 0;
            if tracked && below >= LOST_ENGAGE {
                tracked = false;
            }
        }
        states[i] = tracked;
    }

    let mut flap = 0u32;
    let mut prev_state = states[0];
    let mut last_tracked_val: Option<[f32; 3]> = None;
    let mut jumps: Vec<f32> = Vec::new();
    let mut segments: Vec<Vec<([f32; 3], usize)>> = Vec::new();
    let mut cur_seg: Vec<([f32; 3], usize)> = Vec::new();
    for i in 0..n {
        if states[i] != prev_state {
            flap += 1;
        }
        if states[i] {
            if let Some(s) = ch.samples[i] {
                if !prev_state {
                    if let Some(last) = last_tracked_val {
                        jumps.push(dist(s.value, last, s.dims));
                    }
                    if !cur_seg.is_empty() {
                        segments.push(std::mem::take(&mut cur_seg));
                    }
                }
                cur_seg.push((s.value, s.dims));
                last_tracked_val = Some(s.value);
            }
        } else if !cur_seg.is_empty() {
            segments.push(std::mem::take(&mut cur_seg));
        }
        prev_state = states[i];
    }
    if !cur_seg.is_empty() {
        segments.push(cur_seg);
    }

    m.flap_count = flap;
    m.recovery_count = jumps.len() as u32;
    if !jumps.is_empty() {
        m.recovery_jump_max = maxf(&jumps);
        m.recovery_jump_mean = jumps.iter().sum::<f32>() / jumps.len() as f32;
    }

    let mut samples = MotionSamples::default();
    for seg in &segments {
        accumulate_segment_stats(seg, dt, &mut samples);
    }
    samples.apply_to(&mut m);
    m
}

fn write_summary(
    path: &std::path::Path,
    frames_dir: &str,
    n_frames: usize,
    fps: f32,
    dt: f32,
    rows: &mut [(String, bool, Metrics)],
) -> Result<(), String> {
    let mut md = String::new();
    md.push_str("# Signal quality: temporal replay diagnostics\n\n");
    md.push_str(&format!(
        "`{frames_dir}`, {n_frames} frames @ {fps:.1} fps (dt={dt:.4}s).\n\n"
    ));
    md.push_str(&format!(
        "SRC = raw source-skeleton joints (confidence-gated tracked/lost \
         hysteresis: engage lost after {LOST_ENGAGE} consecutive frames below \
         {CONF_THRESHOLD:.1} confidence, release after {TRACKED_RELEASE} \
         consecutive clear frames — same thresholds as `arm_ray_ik`'s \
         `oof_streak`). AV = solved avatar bone world position (metres), \
         produced by `Application::run_frame` (the real per-frame update \
         function the GUI calls, driven headless — see module docs) — no \
         confidence gating, present every frame once the solver has run \
         once. Comparing SRC vs AV for the same bone attributes a wobble \
         to the detector or to the solver/calibration. `frame_dt` is \
         floored at 1/fps by sleeping out any faster-than-camera gap \
         (see module docs); it does not reproduce the live render loop's \
         decoupled cadence, which can re-solve the same tracking sample \
         multiple times.\n\n"
    ));
    md.push_str(
        "`jitter_hf` = RMS residual after a 3-frame EMA (single-frame \
         flutter). `wander_lf` = RMS residual between the 3-frame and \
         30-frame EMA (~1s-band drift). Neither is gated to \
         \"should-be-still\" periods — a channel with large intentional \
         motion shows large wander too, so compare within same-type \
         channels (L vs R, SRC vs AV) rather than reading the absolute \
         number. `acc`/`jerk` are derived from the VELOCITY VECTOR, not \
         speed, so an in-place back-and-forth oscillation (near-constant \
         speed, flipping direction) shows up as a large jerk spike \
         instead of being averaged away.\n\n\
         `recovery_jump` = position delta measured at the first tracked \
         frame after a lost→tracked transition, vs. the last known \
         tracked value (\"snap on reacquire\").\n\n"
    );

    md.push_str("## Coverage & state stability (SRC channels, sorted by flap_count desc)\n\n");
    md.push_str("| channel | coverage | mean_conf | flap_count | recoveries | jump_max | jump_mean |\n");
    md.push_str("|---|---|---|---|---|---|---|\n");
    let mut state_rows: Vec<&(String, bool, Metrics)> = rows.iter().filter(|(_, hc, _)| *hc).collect();
    state_rows.sort_by(|a, b| b.2.flap_count.cmp(&a.2.flap_count));
    for (name, _, m) in state_rows {
        md.push_str(&format!(
            "| {name} | {:.0}% | {:.2} | {} | {} | {:.4} | {:.4} |\n",
            m.coverage * 100.0,
            m.mean_conf,
            m.flap_count,
            m.recovery_count,
            m.recovery_jump_max,
            m.recovery_jump_mean,
        ));
    }

    for (title, prefix) in [("SRC", "src."), ("AV (post-solve)", "av.")] {
        md.push_str(&format!(
            "\n## Motion smoothness — {title} (sorted by jitter_hf desc)\n\n"
        ));
        md.push_str("| channel | vel_rms | vel_max | acc_rms | acc_max | jerk_rms | jerk_max | jitter_hf | wander_lf |\n");
        md.push_str("|---|---|---|---|---|---|---|---|---|\n");
        let mut section: Vec<&(String, bool, Metrics)> =
            rows.iter().filter(|(name, _, _)| name.starts_with(prefix)).collect();
        // NaN-safe: a diverged/NaN channel must not panic the final
        // write and destroy the whole report. `total_cmp` orders NaN
        // deterministically (sorts to the end here) instead of unwrapping
        // a `None`.
        section.sort_by(|a, b| b.2.jitter_hf_rms.total_cmp(&a.2.jitter_hf_rms));
        for (name, _, m) in section {
            md.push_str(&format!(
                "| {name} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
                m.vel_rms, m.vel_max, m.acc_rms, m.acc_max, m.jerk_rms, m.jerk_max, m.jitter_hf_rms, m.wander_lf_rms,
            ));
        }
    }

    std::fs::write(path, md).map_err(|e| e.to_string())
}
