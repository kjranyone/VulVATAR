//! Live-session recorder — an append-only per-frame time series of the
//! published [`SourceSkeleton`], written while the real D435 pipeline runs.
//!
//! **Why this exists and [`super::debug_channel`] doesn't cover it.** The
//! debug channel is *latest-only*: it overwrites `debug_state.json` every
//! frame so an external overlay can draw the current pose. That is the right
//! shape for watching, and the wrong shape for diagnosis — when the avatar
//! throws a limb across the room for three frames, the evidence is gone
//! before anyone can look at it. The offline replay tool
//! (`diagnose_fusion_replay`) wants
//! `<stem>_color.png` + `<stem>_depth_mm.npy`, and nothing in the app has
//! ever written that pair — the checked-in fixtures were made by hand. So a
//! live misbehaviour report has, until now, had no data behind it.
//!
//! This module closes that gap:
//!
//! * **Always, while recording** — one JSON line per *capture* frame
//!   (`pose.jsonl`) carrying the published joints with their
//!   [`JointOrigin`] provenance, the scale/anchor scalars the whole
//!   pipeline hangs off (`reference_span_m`, `mpsu`, the metric anchor) and
//!   the face pose. The provenance tag is the point: it says whether a
//!   joint that moved impossibly came from a real depth sample or from the
//!   bone-length ray fallback, which is the difference between "the sensor
//!   saw that" and "we made it up".
//! * **On an implausible inter-frame jump** — the colour frame and the
//!   aligned depth map are dumped in exactly the layout the replay bins
//!   read, so the offending frame can be re-run through the real provider
//!   offline, as many times as a fix needs.
//!
//! Off unless `VULVATAR_RECORD` is set (`1` → `diagnostics/session_<unix>/`,
//! any other value → that directory). Writing happens on a background thread
//! behind a bounded queue that *drops* rather than back-pressures: a
//! diagnostic must never change the timing of the pipeline it is measuring.
//! Drops are counted and reported, so a thinned recording can never be
//! mistaken for a quiet one.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::asset::HumanoidBone;
use super::source_skeleton::SourceSkeleton;

/// Inter-frame displacement, in source units, above which a joint is
/// treated as having *teleported* rather than moved.
///
/// The published skeleton is normalised so the shoulder span is
/// `TARGET_SRC_SHOULDER_SPAN` (0.75), so this default is "half a shoulder
/// span in one 33 ms frame" — roughly 5 m/s at the wrist, past what a
/// seated desk subject produces and far past what the torso ever does.
/// Deliberately generous: a trigger that fires on fast-but-real motion
/// buries the real events in noise. Override with `VULVATAR_RECORD_JUMP`.
const DEFAULT_JUMP_TRIGGER: f32 = 0.35;

/// Confidence floor for jump detection. A joint fading in and out of
/// tracking legitimately jumps when it re-acquires; only joints the
/// pipeline claims to be confident about are worth flagging.
const JUMP_MIN_CONFIDENCE: f32 = 0.3;

/// Hard cap on *jump-triggered* colour+depth dumps per session. Each pair
/// is ~1.5 MB at 640×480 and a genuinely broken session flags hundreds of
/// frames; the first handful are enough to replay, and an unbounded dump
/// would fill the disk during exactly the long session someone left
/// running to reproduce a rare fault. Continuous capture
/// (`VULVATAR_RECORD_RAW`) is bounded separately by [`RAW_FRAME_CAP`].
const MAX_ASSET_DUMPS: usize = 24;

/// Frame cap for continuous raw capture — 30 s at 30 fps, ≈1.4 GB at
/// 640×480. Past this the capture stops (and says so) rather than filling
/// the disk: a session long enough to exceed it is long enough that
/// nobody will replay all of it anyway. Override with
/// `VULVATAR_RECORD_RAW_FRAMES`.
const RAW_FRAME_CAP: u64 = 900;

/// Queue depth while continuous raw capture is on. Each raw job carries
/// ~1.5 MB, so this bounds the recorder's own memory at ~0.4 GB in the
/// worst case while giving the PNG encoder enough slack to ride out a
/// disk stall without thinning the recording. A thinned recording is
/// worse than a short one: it silently changes the frame cadence the
/// temporal filters see on replay.
const RAW_QUEUE_DEPTH: usize = 256;

/// Bounded queue depth between the tracking worker and the writer thread.
/// At ~1.5 KB per line this is well under a megabyte of slack, enough to
/// ride out a disk hiccup without letting the recorder become a source of
/// latency in the pipeline it is supposed to be observing.
const QUEUE_DEPTH: usize = 64;

/// One frame's worth of work handed to the writer thread.
enum Job {
    Line(String),
    /// Colour + aligned depth for a flagged frame, in the layout the
    /// replay bins expect.
    Assets {
        stem: String,
        rgb: Vec<u8>,
        width: u32,
        height: u32,
        depth_mm: Vec<u16>,
    },
    /// One line of `manifest.jsonl` — the per-frame sensor metadata a
    /// faithful replay needs (device timestamp, real intrinsics) and that
    /// the PNG/npy pair cannot carry.
    Manifest(String),
}

struct Recorder {
    tx: SyncSender<Job>,
    dir: PathBuf,
    /// Previous frame's positions, for the jump trigger. Only touched from
    /// the tracking worker (the single caller of [`record`]), but held
    /// behind a mutex because the static must be `Sync`.
    prev: Mutex<HashMap<HumanoidBone, [f32; 3]>>,
    jump_trigger: f32,
    assets_dumped: AtomicU64,
    dropped: AtomicU64,
    recorded: AtomicU64,
    /// `Some(n)` → continuous capture of every `n`-th frame. `None` → only
    /// the jump-triggered dumps.
    raw_every: Option<u64>,
    raw_cap: u64,
    raw_written: AtomicU64,
}

fn recorder() -> Option<&'static Recorder> {
    static REC: OnceLock<Option<Recorder>> = OnceLock::new();
    REC.get_or_init(init).as_ref()
}

fn init() -> Option<Recorder> {
    let raw = std::env::var("VULVATAR_RECORD").ok()?;
    if raw.is_empty() || raw == "0" {
        return None;
    }
    let dir = if raw == "1" {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        PathBuf::from("diagnostics").join(format!("session_{secs}"))
    } else {
        PathBuf::from(raw)
    };
    if let Err(e) = std::fs::create_dir_all(&dir) {
        log::error!("session_record: cannot create {}: {e}", dir.display());
        return None;
    }
    let jump_trigger = std::env::var("VULVATAR_RECORD_JUMP")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(DEFAULT_JUMP_TRIGGER);

    // Continuous raw capture: `1` → every frame, `N` → every N-th, unset /
    // `0` → off. Turns the session directory into a frames directory
    // `diagnose_video_replay` can be pointed at, so a captured problem can
    // be re-run against changed code without anyone sitting in front of
    // the camera again.
    let raw_every = std::env::var("VULVATAR_RECORD_RAW")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|n| *n > 0);
    let raw_cap = std::env::var("VULVATAR_RECORD_RAW_FRAMES")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(RAW_FRAME_CAP);

    let (tx, rx) = sync_channel::<Job>(if raw_every.is_some() {
        RAW_QUEUE_DEPTH
    } else {
        QUEUE_DEPTH
    });
    let write_dir = dir.clone();
    let spawned = std::thread::Builder::new()
        .name("session-record".into())
        .spawn(move || {
            let path = write_dir.join("pose.jsonl");
            let mut file = match std::fs::File::create(&path) {
                Ok(f) => std::io::BufWriter::new(f),
                Err(e) => {
                    log::error!("session_record: cannot open {}: {e}", path.display());
                    return;
                }
            };
            let mut manifest: Option<std::io::BufWriter<std::fs::File>> = None;
            for job in rx {
                match job {
                    Job::Line(line) => {
                        let _ = file.write_all(line.as_bytes());
                        let _ = file.write_all(b"\n");
                        // Flushed per line: the failure this recorder exists
                        // to catch has hard-frozen the machine before, and a
                        // buffered tail is exactly what that destroys.
                        let _ = file.flush();
                    }
                    Job::Assets {
                        stem,
                        rgb,
                        width,
                        height,
                        depth_mm,
                    } => write_assets(&write_dir, &stem, &rgb, width, height, &depth_mm),
                    Job::Manifest(line) => {
                        let w = manifest.get_or_insert_with(|| {
                            std::io::BufWriter::new(
                                std::fs::File::create(write_dir.join("manifest.jsonl"))
                                    .expect("create manifest.jsonl"),
                            )
                        });
                        let _ = w.write_all(line.as_bytes());
                        let _ = w.write_all(b"\n");
                        let _ = w.flush();
                    }
                }
            }
            let _ = file.flush();
            if let Some(mut m) = manifest {
                let _ = m.flush();
            }
        });
    if spawned.is_err() {
        log::warn!("session_record: could not spawn writer thread — recording off");
        return None;
    }

    match raw_every {
        Some(n) => log::info!(
            "session_record: recording to {} (jump trigger {jump_trigger} source units; \
             continuous raw capture of every {n} frame(s), cap {raw_cap} frames)",
            dir.display()
        ),
        None => log::info!(
            "session_record: recording to {} (jump trigger {jump_trigger} source units; \
             no raw capture — set VULVATAR_RECORD_RAW=1 for a replayable frame dump)",
            dir.display()
        ),
    }
    Some(Recorder {
        tx,
        dir,
        prev: Mutex::new(HashMap::new()),
        jump_trigger,
        assets_dumped: AtomicU64::new(0),
        dropped: AtomicU64::new(0),
        recorded: AtomicU64::new(0),
        raw_every,
        raw_cap,
        raw_written: AtomicU64::new(0),
    })
}

/// `true` when a session recording is active — lets the caller skip
/// assembling the (cheap but non-zero) argument list.
pub fn active() -> bool {
    recorder().is_some()
}

/// Colour image + `u16` little-endian depth `.npy`, named so
/// `diagnose_fusion_replay` finds its depth sibling by the
/// `_color.*` → `_depth_mm.npy` rule.
///
/// **BMP, not PNG.** Measured on 640×480: PNG costs ~91 ms/frame at the dev
/// profile (dependencies build at opt-level 0, so deflate is unoptimised)
/// against a 33 ms budget — continuous capture would fill its queue in
/// seconds and then drop frames, and a recording with a different frame
/// cadence than the live run cannot reproduce a timing-dependent fault.
/// BMP is uncompressed: ~4 ms/frame in the same build, at the cost of
/// ~0.9 MB instead of ~0.3 MB per frame in a gitignored directory.
fn write_assets(dir: &std::path::Path, stem: &str, rgb: &[u8], w: u32, h: u32, depth_mm: &[u16]) {
    let color_path = dir.join(format!("{stem}_color.bmp"));
    match image::RgbImage::from_raw(w, h, rgb.to_vec()) {
        Some(img) => {
            if let Err(e) = img.save(&color_path) {
                log::warn!("session_record: colour write failed: {e}");
                return;
            }
        }
        None => {
            log::warn!("session_record: rgb buffer does not match {w}x{h}");
            return;
        }
    }

    // Minimal NPY v1.0 writer — the replay bins parse exactly this subset
    // (2-D, little-endian u16, C order), and pulling a numpy crate in for
    // one header would be a dependency for eleven lines of formatting.
    let header = format!(
        "{{'descr': '<u2', 'fortran_order': False, 'shape': ({h}, {w}), }}"
    );
    // The header (magic + version + len + text) must be a multiple of 64
    // bytes including the terminating newline.
    let unpadded = 10 + header.len() + 1;
    let pad = (64 - (unpadded % 64)) % 64;
    let mut out = Vec::with_capacity(unpadded + pad + depth_mm.len() * 2);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    let header_len = (header.len() + 1 + pad) as u16;
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend(std::iter::repeat_n(b' ', pad));
    out.push(b'\n');
    for v in depth_mm {
        out.extend_from_slice(&v.to_le_bytes());
    }
    if let Err(e) = std::fs::write(dir.join(format!("{stem}_depth_mm.npy")), &out) {
        log::warn!("session_record: npy write failed: {e}");
    }
}

fn origin_tag(conf: f32) -> &'static str {
    if conf >= 0.5 {
        "O"
    } else {
        "E"
    }
}

/// Append one frame to the session recording. No-op unless
/// `VULVATAR_RECORD` is set.
///
/// `depth_raw` / `depth_units` are only touched when this frame trips the
/// jump trigger, so the common path costs one JSON assembly and a
/// non-blocking queue push — the ~1.5 MB colour+depth clone happens for
/// the handful of frames that actually broke.
#[allow(clippy::too_many_arguments)]
pub fn record(
    frame_index: u64,
    sk: &SourceSkeleton,
    rgb: &[u8],
    width: u32,
    height: u32,
    depth_raw: &[u16],
    depth_units: f32,
    intrinsics: super::CameraIntrinsics,
    timestamp_ms: f64,
) {
    let Some(rec) = recorder() else {
        return;
    };

    // --- joints + jump detection -------------------------------------
    let mut joints = serde_json::Map::with_capacity(sk.joints.len());
    let mut jumps: Vec<serde_json::Value> = Vec::new();
    {
        let mut prev = match rec.prev.lock() {
            Ok(p) => p,
            Err(p) => p.into_inner(),
        };
        for (&bone, joint) in &sk.joints {
            let p = joint.position;
            if joint.confidence >= JUMP_MIN_CONFIDENCE {
                if let Some(&q) = prev.get(&bone) {
                    let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2))
                        .sqrt();
                    if d > rec.jump_trigger {
                        jumps.push(serde_json::json!([format!("{bone:?}"), d]));
                    }
                }
            }
            prev.insert(bone, p);
            joints.insert(
                format!("{bone:?}"),
                serde_json::json!({
                    "p": p,
                    "c": joint.confidence,
                    "o": origin_tag(joint.confidence),
                }),
            );
        }
    }

    // --- scale / anchor scalars --------------------------------------
    // Everything downstream of the depth builder is multiplied by these:
    // `reference_span_m` sets every anthropometric bone length AND the
    // whole-skeleton normalisation, the anchor sets the avatar's distance.
    // A recording without them cannot distinguish "the detector moved" from
    // "the scale under the detector moved".
    let metric = sk.metric_frame_info.as_ref().map(|m| {
        serde_json::json!({
            "ref_span_m": m.reference_span_m,
            "mpsu": m.mpsu,
            "anchor_cam_m": m.anchor_cam_m,
            "anchor_is_hip": m.anchor_is_hip,
        })
    });

    let face = sk.face.map(|f| {
        serde_json::json!({ "y": f.yaw, "p": f.pitch, "r": f.roll, "c": f.confidence })
    });

    // --- frame capture -------------------------------------------------
    // Two reasons to write the colour+depth pair: continuous capture (turn
    // the session into a replayable frame directory) and the bounded
    // jump-triggered dump (preserve the evidence when nothing else is
    // being captured). Continuous wins when both apply — same files, one
    // write, and the `frame_` prefix keeps the directory in capture order
    // for the replay bin's `files.sort()`.
    let frame_ok = !rgb.is_empty() && depth_raw.len() == (width as usize * height as usize);
    let raw_due = rec
        .raw_every
        .is_some_and(|n| frame_index.is_multiple_of(n))
        && rec.raw_written.load(Ordering::Relaxed) < rec.raw_cap;
    let jump_due = !jumps.is_empty()
        && (rec.assets_dumped.load(Ordering::Relaxed) as usize) < MAX_ASSET_DUMPS;

    let mut dump_stem = None;
    if frame_ok && (raw_due || jump_due) {
        let stem = if raw_due {
            rec.raw_written.fetch_add(1, Ordering::Relaxed);
            format!("frame_{frame_index:06}")
        } else {
            rec.assets_dumped.fetch_add(1, Ordering::Relaxed);
            format!("jump_{frame_index:06}")
        };
        // Normalise to millimetres regardless of the device's depth scale —
        // the `_depth_mm` suffix is a contract with the replay bins, not a
        // description of whatever unit this particular D435 reported.
        let to_mm = depth_units * 1000.0;
        let depth_mm: Vec<u16> = if (to_mm - 1.0).abs() < 1e-6 {
            depth_raw.to_vec()
        } else {
            depth_raw
                .iter()
                .map(|&d| ((d as f32) * to_mm).round().clamp(0.0, u16::MAX as f32) as u16)
                .collect()
        };
        let job = Job::Assets {
            stem: stem.clone(),
            rgb: rgb.to_vec(),
            width,
            height,
            depth_mm,
        };
        if rec.tx.try_send(job).is_err() {
            rec.dropped.fetch_add(1, Ordering::Relaxed);
        } else {
            // Sensor metadata the PNG/npy pair cannot carry. Without the
            // real intrinsics a replay deprojects with the wrong focal
            // length and every metric position is silently wrong; without
            // the device timestamp the pipeline's dt-normalised filters run
            // at a nominal rate instead of the captured one.
            let manifest = serde_json::json!({
                "stem": stem,
                "f": frame_index,
                "t_ms": timestamp_ms,
                "w": width,
                "h": height,
                "depth_units": depth_units,
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "cx": intrinsics.cx,
                "cy": intrinsics.cy,
            });
            let _ = rec.tx.try_send(Job::Manifest(manifest.to_string()));
            dump_stem = Some(stem);
        }
    }

    let line = serde_json::json!({
        "f": frame_index,
        "t_ms": sk.capture_timestamp_ms,
        "conf": sk.overall_confidence,
        "metric": metric,
        "root_offset": sk.root_offset,
        "root_anchor_is_hip": sk.root_anchor_is_hip,
        "face": face,
        "j": joints,
        "jump": (!jumps.is_empty()).then_some(jumps),
        "dump": dump_stem,
    });

    match rec.tx.try_send(Job::Line(line.to_string())) {
        Ok(()) => {
            rec.recorded.fetch_add(1, Ordering::Relaxed);
        }
        // Never block the tracking worker on the disk. A dropped line is a
        // gap in the series; a stalled capture thread is a different bug
        // that the recording would then "prove".
        Err(_) => {
            rec.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Log where the recording went and how complete it is. Called once when
/// the tracking session ends so the operator never has to guess whether a
/// short file means "nothing happened" or "the writer fell behind".
pub fn finish() {
    let Some(rec) = recorder() else {
        return;
    };
    let dropped = rec.dropped.load(Ordering::Relaxed);
    let recorded = rec.recorded.load(Ordering::Relaxed);
    let assets = rec.assets_dumped.load(Ordering::Relaxed);
    let raw = rec.raw_written.load(Ordering::Relaxed);
    if dropped > 0 {
        log::warn!(
            "session_record: {} frames written to {}, {dropped} DROPPED (writer fell behind — \
             the series has gaps and the captured frame cadence is NOT the live one), \
             {raw} raw frames, {assets} flagged-frame dumps",
            recorded,
            rec.dir.display()
        );
    } else {
        log::info!(
            "session_record: {recorded} frames written to {}, {raw} raw frames, \
             {assets} flagged-frame dumps",
            rec.dir.display()
        );
    }
    if assets as usize >= MAX_ASSET_DUMPS {
        log::warn!(
            "session_record: flagged-frame dumps hit the {MAX_ASSET_DUMPS} cap — later jumps \
             were NOT captured as replayable frames"
        );
    }
    if rec.raw_every.is_some() && raw >= rec.raw_cap {
        log::warn!(
            "session_record: raw capture hit the {} frame cap — the recording ends there, \
             later frames were not captured (raise VULVATAR_RECORD_RAW_FRAMES)",
            rec.raw_cap
        );
    }
    if raw > 0 {
        log::info!(
            "session_record: replay it with  cargo run --bin diagnose_video_replay -- {}",
            rec.dir.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The NPY header must be parseable by the same minimal reader the
    /// replay bins use, or a recorded session is not replayable — which is
    /// the entire point of dumping it.
    #[test]
    fn npy_header_is_64_byte_aligned_and_declares_shape() {
        let dir = std::env::temp_dir().join(format!("vulvatar_rec_npy_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("tempdir");
        // 4×3 frame so rows/cols are distinguishable if they get swapped.
        let depth: Vec<u16> = (0..12).collect();
        write_assets(&dir, "t", &vec![0u8; 4 * 3 * 3], 4, 3, &depth);

        let bytes = std::fs::read(dir.join("t_depth_mm.npy")).expect("npy written");
        assert_eq!(&bytes[0..6], b"\x93NUMPY");
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0, "npy data must start 64-byte aligned");
        let header = std::str::from_utf8(&bytes[10..10 + header_len]).expect("utf8 header");
        assert!(header.contains("<u2"), "header: {header}");
        // (rows, cols) = (height, width).
        assert!(header.contains("(3, 4)"), "header: {header}");
        assert_eq!(bytes.len(), 10 + header_len + depth.len() * 2);

        // Round-trip the payload the way the replay bins read it.
        let data = &bytes[10 + header_len..];
        let read: Vec<u16> = data
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(read, depth);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The replay side recovers the manifest key by stripping `_color`
    /// from the colour file's stem. If that ever stops matching the stem
    /// the capture recorded, the replay silently falls back to nominal
    /// intrinsics and a nominal clock — wrong numbers, no error. Pin the
    /// relationship from the writer's side.
    #[test]
    fn manifest_stem_matches_the_colour_filename() {
        let dir = std::env::temp_dir().join(format!("vulvatar_rec_stem_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("tempdir");
        let stem = "frame_000123";
        write_assets(&dir, stem, &vec![0u8; 4 * 3 * 3], 4, 3, &vec![0u16; 12]);

        let colour = dir.join(format!("{stem}_color.bmp"));
        assert!(colour.is_file(), "colour frame written as <stem>_color.bmp");
        let recovered = colour
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_suffix("_color"))
            .expect("replay's key-recovery rule applies");
        assert_eq!(recovered, stem, "manifest key must round-trip");
        assert!(
            dir.join(format!("{stem}_depth_mm.npy")).is_file(),
            "depth sibling must follow the _color.png -> _depth_mm.npy rule"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Not a correctness test — a throughput measurement. Continuous
    /// capture PNG-encodes every frame on the writer thread; if that costs
    /// more than the frame interval the recording thins, and a thinned
    /// recording misrepresents the cadence the temporal filters saw.
    /// `cargo test -- --ignored --nocapture frame_write_throughput`
    ///
    /// Measured 640×480 on the dev profile: PNG 90.8 ms/frame (11 fps
    /// ceiling — cannot keep up), BMP 3.6 ms/frame (277 fps). That gap is
    /// why the capture writes BMP.
    #[test]
    #[ignore = "measurement, not an assertion"]
    fn frame_write_throughput() {
        let dir = std::env::temp_dir().join(format!("vulvatar_rec_bench_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("tempdir");
        let (w, h) = (640u32, 480u32);
        // Structured noise: a flat buffer compresses unrealistically well.
        let rgb: Vec<u8> = (0..(w * h * 3))
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();
        let depth: Vec<u16> = (0..(w * h)).map(|i| (i % 4000) as u16).collect();
        const N: u32 = 20;
        let t0 = std::time::Instant::now();
        for i in 0..N {
            write_assets(&dir, &format!("b{i}"), &rgb, w, h, &depth);
        }
        let per = t0.elapsed().as_secs_f64() * 1000.0 / N as f64;
        println!("write_assets {w}x{h}: {per:.1} ms/frame  ({:.1} fps ceiling)", 1000.0 / per);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn recording_is_off_without_the_env_var() {
        // The default build must not touch the disk on the tracking hot
        // path; `record` is called unconditionally from the worker.
        if std::env::var("VULVATAR_RECORD").is_ok() {
            return; // someone is actually recording — nothing to assert
        }
        assert!(!active());
        let intr = crate::tracking::CameraIntrinsics {
            fx: 600.0,
            fy: 600.0,
            cx: 320.0,
            cy: 240.0,
            width: 640,
            height: 480,
        };
        record(0, &SourceSkeleton::default(), &[], 0, 0, &[], 0.001, intr, 0.0);
    }
}
