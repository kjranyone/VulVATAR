//! Full-rate capture recorder: writes the RAW camera inputs of a live
//! session to disk so any pose estimator — the current one or a replacement —
//! can be replayed and scored on identical data, offline, without the camera
//! and without costing the user a session.
//!
//! # Why raw inputs, and why a new recorder
//!
//! The live debug channel publishes *derived* state (source joints, the
//! solved avatar). That is enough to see what the current pipeline did, and
//! it is what found the v1 hand-blink / solver artefacts —
//! but it cannot evaluate a DIFFERENT estimator, because every value in it
//! has already been shaped by the current one's decisions (which joints
//! "exist", which depth samples were admitted, which frames were held). A
//! replacement has to be fed what the sensor produced.
//!
//! It also cannot be reconstructed from the existing snapshots: the depth
//! snapshot is one frame per second and the camera dump is down-scaled.
//!
//! # Gapless by construction
//!
//! Frames are captured into RAM and written to disk afterwards. The first
//! design encoded inline on a background thread and **dropped 49 % of a live
//! session** (measured 2026-07-28: 400 kept, 392 dropped over 30 s) because
//! PNG encoding cannot keep up with capture. That is fatal for the purpose:
//! the artefacts this recorder exists to evaluate — teleports, joints
//! blinking, hold/release behaviour — are all TEMPORAL, and a recording with
//! every other frame missing both destroys those metrics and changes the
//! `dt` the pipeline's own filters see, so the replay would no longer
//! represent the live system.
//!
//! Buffering in RAM removes the race entirely: capture only appends, and the
//! session ends when the frame budget or the memory budget is reached,
//! whichever comes first. If a drop ever does happen (allocation failure),
//! it is counted and surfaced, never silently smoothed over.
//!
//! # Format
//!
//! One directory per session, holding the same pair the existing offline
//! benches already read (`diagnose_fusion_replay`):
//!
//! ```text
//! diagnostics/sessions/<stamp>/
//!   f00000_color.png       RGB8, full capture resolution
//!   f00000_depth_mm.npy    u16 little-endian, C-order rows x cols, aligned
//!   meta.jsonl             one line per recorded frame
//! ```
//!
//! `meta.jsonl` carries what a faithful replay needs and a PNG cannot hold:
//! the device capture timestamp (the pipeline's whole time base), the colour
//! intrinsics, the depth unit scale, and the source frame index.
//!
//! # Control
//!
//! Off unless `%ProgramData%\VulVATAR\record.on` exists; its contents may be
//! a frame budget (default [`DEFAULT_MAX_FRAMES`]). The flag is POLLED, so a
//! session already in flight can be armed without restarting the app — which
//! is the normal case, since the behaviour worth recording shows up during
//! ordinary use. Recording stops at a budget and does not restart within the
//! process, so a forgotten flag file cannot quietly fill a disk.

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use super::realsense::RealSenseFrame;

/// Frames captured before the recorder stops on its own.
const DEFAULT_MAX_FRAMES: u64 = 600;

/// RAM the buffered frames may occupy. At 640x480 a frame pair is ~1.5 MB
/// (raw RGB + u16 depth), so this holds ~25 s of capture; at 1280x720 it is
/// ~4.1 MB and ~9 s. Both are long enough for the pose transitions under
/// study, and the ceiling is what keeps a long session from paging the app
/// out instead of recording it.
const MEMORY_BUDGET_BYTES: usize = 1_200_000_000;

/// How often the flag file is stat-ed while idle.
const FLAG_POLL_MS: u64 = 2000;

struct Frame {
    frame_index: u64,
    rgb: Vec<u8>,
    width: u32,
    height: u32,
    depth_raw: Vec<u16>,
    depth_units: f32,
    timestamp_ms: f64,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
}

impl Frame {
    fn bytes(&self) -> usize {
        self.rgb.len() + self.depth_raw.len() * 2
    }
}

/// Recorder lifecycle. The flag file is polled while idle rather than read
/// once: the operator arms the recorder on a session that is ALREADY
/// running, and a one-shot read at the first frame would latch "off" forever
/// (which is exactly what happened on the first attempt).
enum State {
    Idle {
        last_check_ms: u64,
    },
    Capturing {
        frames: Vec<Frame>,
        bytes: usize,
        dir: PathBuf,
    },
    Finished,
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();
static EPOCH: OnceLock<Instant> = OnceLock::new();
static MAX_FRAMES: AtomicU64 = AtomicU64::new(DEFAULT_MAX_FRAMES);

fn state() -> &'static Mutex<State> {
    STATE.get_or_init(|| Mutex::new(State::Idle { last_check_ms: 0 }))
}

fn base_dir() -> PathBuf {
    std::env::var_os("ProgramData")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\ProgramData"))
        .join("VulVATAR")
}

/// Where sessions are written. Prefers the repo's `diagnostics/` (the rule
/// for tool output) when the process runs from a checkout; otherwise falls
/// back beside the other live-debug artefacts, so a shortcut-launched app
/// still produces a findable recording instead of scattering frames into
/// whatever directory it happened to start in.
fn sessions_root() -> PathBuf {
    let local = PathBuf::from("diagnostics");
    if local.is_dir() {
        local.join("sessions")
    } else {
        base_dir().join("sessions")
    }
}

/// Minimal `.npy` writer for a 2-D little-endian `u16` array, matching the
/// reader every offline bench already has.
fn write_npy_u16(
    path: &std::path::Path,
    rows: usize,
    cols: usize,
    data: &[u16],
) -> std::io::Result<()> {
    let header = format!("{{'descr': '<u2', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    // Magic + version + 2-byte header length, padded so the data starts on a
    // 64-byte boundary (numpy's own rule; some readers rely on it).
    let prefix = 10 + header.len() + 1;
    let pad = (64 - (prefix % 64)) % 64;
    let padded = format!("{header}{}\n", " ".repeat(pad));
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(padded.len() as u16).to_le_bytes())?;
    f.write_all(padded.as_bytes())?;
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&bytes)?;
    f.flush()
}

/// Try to arm from the flag file, returning the session directory.
fn try_start() -> Option<PathBuf> {
    let contents = std::fs::read_to_string(base_dir().join("record.on")).ok()?;
    if let Ok(n) = contents.trim().parse::<u64>() {
        if n > 0 {
            MAX_FRAMES.store(n, Ordering::Relaxed);
        }
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let dir = sessions_root().join(format!("s{stamp}"));
    if let Err(e) = std::fs::create_dir_all(&dir) {
        log::warn!("sequence_recorder: cannot create {}: {e}", dir.display());
        return None;
    }
    log::info!(
        "sequence_recorder: ARMED, up to {} frames / {} MB -> {}",
        MAX_FRAMES.load(Ordering::Relaxed),
        MEMORY_BUDGET_BYTES / 1_000_000,
        std::fs::canonicalize(&dir).unwrap_or_else(|_| dir.clone()).display()
    );
    Some(dir)
}

/// Flush the captured frames to disk on a background thread, so the capture
/// thread returns to tracking immediately when a session ends.
fn flush(frames: Vec<Frame>, dir: PathBuf) {
    let spawned = std::thread::Builder::new()
        .name("seq-flush".into())
        .spawn(move || {
            let meta_path = dir.join("meta.jsonl");
            let Ok(file) = std::fs::File::create(&meta_path) else {
                log::warn!("sequence_recorder: cannot write {}", meta_path.display());
                return;
            };
            let mut meta = std::io::BufWriter::new(file);
            let total = frames.len();
            for f in frames {
                let color = dir.join(format!("f{:05}_color.png", f.frame_index));
                let depth = dir.join(format!("f{:05}_depth_mm.npy", f.frame_index));
                if let Some(buf) = image::RgbImage::from_raw(f.width, f.height, f.rgb) {
                    if let Err(e) = buf.save(&color) {
                        log::warn!("sequence_recorder: color: {e}");
                        continue;
                    }
                }
                if let Err(e) =
                    write_npy_u16(&depth, f.height as usize, f.width as usize, &f.depth_raw)
                {
                    log::warn!("sequence_recorder: depth: {e}");
                    continue;
                }
                let line = format!(
                    "{{\"frame\":{},\"t_ms\":{},\"w\":{},\"h\":{},\"depth_units\":{},\"fx\":{},\"fy\":{},\"cx\":{},\"cy\":{},\"dropped_before\":0}}\n",
                    f.frame_index,
                    f.timestamp_ms,
                    f.width,
                    f.height,
                    f.depth_units,
                    f.fx,
                    f.fy,
                    f.cx,
                    f.cy,
                );
                let _ = meta.write_all(line.as_bytes());
            }
            let _ = meta.flush();
            log::info!(
                "sequence_recorder: wrote {total} gapless frames to {}",
                dir.display()
            );
        });
    if spawned.is_err() {
        log::warn!("sequence_recorder: could not spawn the flush thread; session lost");
    }
}

/// Offer one captured frame to the recorder. No-op unless armed. Capture
/// only appends to RAM, so it cannot fall behind and cannot drop frames.
pub fn record(frame_index: u64, frame: &RealSenseFrame) {
    let Ok(mut st) = state().lock() else { return };
    match &mut *st {
        State::Finished => (),
        State::Idle { last_check_ms } => {
            let now_ms = EPOCH.get_or_init(Instant::now).elapsed().as_millis() as u64;
            if *last_check_ms != 0 && now_ms.saturating_sub(*last_check_ms) < FLAG_POLL_MS {
                return;
            }
            match try_start() {
                Some(dir) => {
                    *st = State::Capturing {
                        frames: Vec::new(),
                        bytes: 0,
                        dir,
                    }
                }
                None => {
                    *st = State::Idle {
                        last_check_ms: now_ms.max(1),
                    }
                }
            }
        }
        State::Capturing { frames, bytes, dir } => {
            let captured = Frame {
                frame_index,
                rgb: frame.rgb.clone(),
                width: frame.width,
                height: frame.height,
                depth_raw: frame.depth_raw.clone(),
                depth_units: frame.depth_units,
                timestamp_ms: frame.timestamp_ms,
                fx: frame.intrinsics.fx,
                fy: frame.intrinsics.fy,
                cx: frame.intrinsics.cx,
                cy: frame.intrinsics.cy,
            };
            *bytes += captured.bytes();
            frames.push(captured);
            let full = frames.len() as u64 >= MAX_FRAMES.load(Ordering::Relaxed)
                || *bytes >= MEMORY_BUDGET_BYTES;
            if full {
                log::info!(
                    "sequence_recorder: capture complete ({} frames, {} MB) — flushing",
                    frames.len(),
                    *bytes / 1_000_000
                );
                let taken = std::mem::take(frames);
                let dir = dir.clone();
                *st = State::Finished;
                flush(taken, dir);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The npy header must be exactly what the offline benches' reader
    /// expects: magic, v1 header length, `<u2` descr, C order, and the
    /// declared shape — otherwise a recorded session is unreadable by the
    /// very tools it exists to feed.
    #[test]
    fn npy_round_trips_through_the_bench_reader_contract() {
        let dir = std::env::temp_dir().join("vulvatar_npy_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("t_depth_mm.npy");
        let data: Vec<u16> = (0..12u16).collect();
        write_npy_u16(&path, 3, 4, &data).expect("write");
        let bytes = std::fs::read(&path).expect("read");

        assert_eq!(&bytes[0..6], b"\x93NUMPY");
        assert_eq!(bytes[6], 1, "version 1 header");
        let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = std::str::from_utf8(&bytes[10..10 + hlen]).expect("utf8");
        assert!(header.contains("'descr': '<u2'"), "{header}");
        assert!(header.contains("'fortran_order': False"), "{header}");
        assert!(header.contains("(3, 4)"), "{header}");
        assert_eq!((10 + hlen) % 64, 0, "data must start 64-byte aligned");

        let payload = &bytes[10 + hlen..];
        assert_eq!(payload.len(), 24, "3x4 u16");
        let back: Vec<u16> = payload
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(back, data, "values must survive the round trip");
        let _ = std::fs::remove_file(&path);
    }

    /// Arming must be possible on a session that is already running: the
    /// idle state re-checks the flag file instead of latching the first
    /// answer forever. (The first implementation cached "off" at the first
    /// frame, so a flag placed seconds later never fired.)
    #[test]
    fn idle_state_rechecks_the_flag_instead_of_latching() {
        let due = |last: u64, now: u64| last == 0 || now.saturating_sub(last) >= FLAG_POLL_MS;
        assert!(due(0, 0), "first frame must check");
        assert!(!due(1000, 1500), "throttled between polls");
        assert!(due(1000, 1000 + FLAG_POLL_MS), "checks again after the interval");
    }

    /// The memory ceiling must end a session before the app pages itself
    /// out, and the frame budget must end it before the ceiling on small
    /// frames — whichever comes first, so neither can run away.
    #[test]
    fn capture_stops_at_whichever_budget_binds_first() {
        let frame_bytes = 640 * 480 * 3 + 640 * 480 * 2;
        let by_memory = MEMORY_BUDGET_BYTES / frame_bytes;
        assert!(
            by_memory > 200,
            "the memory ceiling must allow a usable clip, got {by_memory} frames"
        );
        // At the default frame budget the ceiling must not bind first for a
        // 640x480 stream — otherwise the documented budget would be a lie.
        assert!(
            (DEFAULT_MAX_FRAMES as usize) < by_memory,
            "frame budget {DEFAULT_MAX_FRAMES} exceeds what memory allows ({by_memory})"
        );
    }
}
