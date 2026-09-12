//! Consumer for the file-backed shared-texture bridge
//! (`FrameSink::SharedTextureFileStub` and the Win32 `.vgtk` sidecars).
//!
//! The producer overwrites a single record — metadata-only `VGTK` (72 B)
//! when the renderer published a GPU token, legacy `VSTX` (28 B header +
//! RGBA bytes) on the CPU fallback path. This tool is the read side of
//! that handoff: it validates and decodes records, can watch a live
//! bridge file and report the observed update cadence, and can dump
//! VSTX payloads as PNGs.
//!
//! What this consumer does NOT do yet: import the shared texture behind
//! a `Win32Kmt` handle (`OpenSharedResource1` in D3D, or a Vulkan
//! `OPAQUE_WIN32_KMT` import). The decode here stops at the record
//! boundary — enough to verify handoff metadata, colour-space / alpha
//! contracts and cadence; docs/output-interop.md §"Synchronization"
//! covers what a zero-copy consumer additionally needs.
//!
//! Usage:
//!   consume_shared_texture [path] [--watch secs] [--save-png dir] [--self-test]
//!
//!   path        bridge file to read (default: %TEMP%\vulvatar-shared-texture.bin;
//!               the Win32 sinks write %ProgramData%\VulVATAR\camera_frame_buffer.vgtk
//!               and shared_memory_buffer.vgtk — same VGTK shape)
//!   --watch N   poll for up to N seconds, printing each new record and a
//!               cadence summary (Ctrl-C to stop early)
//!   --save-png D  for VSTX records, write the RGBA payload as <D>/frame_<id>.png
//!   --self-test synthesise VGTK + VSTX records through the real sink writer,
//!               then parse them back (no producer needed)

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use vulvatar_lib::frame_handoff::{
    ExternalHandleType, FrameLease, FrameLifetimeContract, GpuFrameToken, OutputSyncToken,
};
use vulvatar_lib::output::frame_sink::SharedMemoryFileSink;
use vulvatar_lib::output::{AlphaMode, HandoffPath, OutputColorSpace, OutputFrame, OutputSinkWriter};

// The bridge format constants are private to `output::frame_sink`
// (producer-internal); the consumer re-declares them from the spec in
// docs/output-interop.md, same as the in-crate e2e tests do.
const VGTK_MAGIC: [u8; 4] = *b"VGTK";
const VGTK_HEADER_SIZE: usize = 72;
const VSTX_MAGIC: [u8; 4] = *b"VSTX";
const VSTX_HEADER_SIZE: usize = 28;

#[derive(Debug)]
enum BridgeRecord {
    /// 72-byte metadata-only record carrying an exported GPU texture token.
    GpuToken {
        version: u32,
        extent: [u32; 2],
        handle_type: u32,
        sync_type: u32,
        sync_value: u64,
        external_handle: u64,
        frame_id: u64,
        timestamp_ms: u64,
        color_space: u32,
        alpha_mode: u32,
        flags: u32,
    },
    /// Legacy CPU-readback record: 28-byte header + RGBA8 payload.
    CpuFrame {
        extent: [u32; 2],
        frame_index: u64,
        timestamp_ms: u64,
        pixels: Vec<u8>,
    },
}

fn le_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
fn le_u64(b: &[u8], off: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(a)
}

fn handle_type_name(code: u32) -> &'static str {
    match code {
        0 => "Unavailable",
        1 => "Win32Kmt",
        2 => "D3D12Fence",
        3 => "VkSemaphore",
        4 => "SharedMemoryHandle",
        _ => "unknown",
    }
}

fn sync_name(kind: u32, value: u64) -> String {
    match kind {
        0 => "None".to_string(),
        1 => "ProducerWaitComplete (texture already complete at publish)".to_string(),
        2 => format!("FenceValue({value})"),
        3 => format!("SemaphoreHandle({value})"),
        _ => format!("unknown({kind}, {value})"),
    }
}

fn parse_record(bytes: &[u8]) -> Result<BridgeRecord, String> {
    if bytes.len() < 4 {
        return Err(format!(
            "file too small to carry a record header ({} bytes) — producer down or mid-copy?",
            bytes.len()
        ));
    }
    if bytes[0..4] == VGTK_MAGIC {
        if bytes.len() != VGTK_HEADER_SIZE {
            return Err(format!(
                "VGTK record must be exactly {VGTK_HEADER_SIZE} bytes, file is {}",
                bytes.len()
            ));
        }
        let version = le_u32(bytes, 4);
        if version != 1 {
            return Err(format!(
                "unsupported VGTK version {version} (consumer knows version 1)"
            ));
        }
        return Ok(BridgeRecord::GpuToken {
            version,
            extent: [le_u32(bytes, 8), le_u32(bytes, 12)],
            handle_type: le_u32(bytes, 16),
            sync_type: le_u32(bytes, 20),
            sync_value: le_u64(bytes, 32),
            external_handle: le_u64(bytes, 24),
            frame_id: le_u64(bytes, 40),
            timestamp_ms: le_u64(bytes, 48),
            color_space: le_u32(bytes, 56),
            alpha_mode: le_u32(bytes, 60),
            flags: le_u32(bytes, 64),
        });
    }
    if bytes[0..4] == VSTX_MAGIC {
        if bytes.len() < VSTX_HEADER_SIZE {
            return Err(format!(
                "VSTX header truncated: {} bytes, need {VSTX_HEADER_SIZE}",
                bytes.len()
            ));
        }
        let extent = [le_u32(bytes, 4), le_u32(bytes, 8)];
        let expected = VSTX_HEADER_SIZE + extent[0] as usize * extent[1] as usize * 4;
        if bytes.len() != expected {
            return Err(format!(
                "VSTX payload size mismatch: file is {} bytes, header says {expected} ({}x{} RGBA)",
                bytes.len(),
                extent[0],
                extent[1]
            ));
        }
        return Ok(BridgeRecord::CpuFrame {
            extent,
            frame_index: le_u64(bytes, 12),
            timestamp_ms: le_u64(bytes, 20),
            pixels: bytes[VSTX_HEADER_SIZE..].to_vec(),
        });
    }
    Err(format!(
        "unrecognised magic {:?} — not a bridge record",
        &bytes[0..4]
    ))
}

fn describe(record: &BridgeRecord) -> String {
    match record {
        BridgeRecord::GpuToken {
            version,
            extent,
            handle_type,
            sync_type,
            sync_value,
            external_handle,
            frame_id,
            timestamp_ms,
            color_space,
            alpha_mode,
            flags,
        } => {
            let mut s = format!(
                "VGTK v{version} GPU token: {}x{}, handle_type={} (0x{external_handle:x}), frame_id={frame_id}, ts={timestamp_ms} ms\n",
                extent[0],
                extent[1],
                handle_type_name(*handle_type),
            );
            s.push_str(&format!("  sync: {}\n", sync_name(*sync_type, *sync_value)));
            s.push_str(&format!(
                "  color_space={} alpha_mode={} flags=[{}{}]\n",
                if *color_space == 1 { "LinearSrgb" } else { "Srgb" },
                match alpha_mode {
                    0 => "Opaque",
                    1 => "Premultiplied",
                    2 => "Straight",
                    _ => "unknown",
                },
                if flags & 1 != 0 { "preserve-alpha " } else { "" },
                if flags & 2 != 0 { "linear-color" } else { "" },
            ));
            s.push_str(
                "  lease: Option A (SingleConsumerRetained) — this record's texture stays\n\
                 \x20        valid until the NEXT record replaces it; no ack channel exists.\n\
                 \x20        import: D3D OpenSharedResource1 / Vulkan OPAQUE_WIN32_KMT import",
            );
            s
        }
        BridgeRecord::CpuFrame {
            extent,
            frame_index,
            timestamp_ms,
            pixels,
        } => format!(
            "VSTX CPU fallback: {}x{} RGBA ({} B payload), frame_index={frame_index}, ts={timestamp_ms} ms",
            extent[0],
            extent[1],
            pixels.len()
        ),
    }
}

fn save_png(record: &BridgeRecord, dir: &Path) -> Result<PathBuf, String> {
    let BridgeRecord::CpuFrame {
        extent,
        frame_index,
        pixels,
        ..
    } = record
    else {
        return Err("--save-png only applies to VSTX (CPU fallback) records".to_string());
    };
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let path = dir.join(format!("frame_{frame_index:06}.png"));
    image::RgbaImage::from_raw(extent[0], extent[1], pixels.clone())
        .ok_or_else(|| "payload does not fit the declared extent".to_string())?
        .save(&path)
        .map_err(|e| format!("save {}: {e}", path.display()))?;
    Ok(path)
}

struct Args {
    path: Option<PathBuf>,
    watch_secs: Option<f64>,
    save_png: Option<PathBuf>,
    self_test: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        path: None,
        watch_secs: None,
        save_png: None,
        self_test: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--watch" => {
                let v = it.next().ok_or("--watch needs a seconds argument")?;
                args.watch_secs = Some(v.parse().map_err(|_| format!("bad --watch value '{v}'"))?);
            }
            "--save-png" => {
                let v = it.next().ok_or("--save-png needs a directory argument")?;
                args.save_png = Some(PathBuf::from(v));
            }
            "--self-test" => args.self_test = true,
            other if other.starts_with("--") => {
                return Err(format!("unknown flag '{other}'"));
            }
            other => {
                if args.path.is_some() {
                    return Err("multiple positional paths given".to_string());
                }
                args.path = Some(PathBuf::from(other));
            }
        }
    }
    Ok(args)
}

fn usage() -> ! {
    eprintln!("usage: consume_shared_texture [path] [--watch secs] [--save-png dir] [--self-test]");
    std::process::exit(2);
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            usage();
        }
    };

    if args.self_test {
        match self_test() {
            Ok(()) => {
                println!("self-test: OK");
                return;
            }
            Err(e) => {
                eprintln!("self-test FAILED: {e}");
                std::process::exit(1);
            }
        }
    }

    let path = args
        .path
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join("vulvatar-shared-texture.bin"));

    match args.watch_secs {
        Some(secs) => {
            if let Err(e) = watch(&path, secs, args.save_png.as_deref()) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        None => match read_once(&path, args.save_png.as_deref()) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        },
    }
}

/// Read and describe the current record once.
fn read_once(path: &Path, save_png_dir: Option<&Path>) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let record = parse_record(&bytes)?;
    println!("{}", describe(&record));
    if let Some(dir) = save_png_dir {
        let out = save_png(&record, dir)?;
        println!("saved: {}", out.display());
    }
    Ok(())
}

/// Poll the bridge file, printing each distinct record and a cadence
/// summary. Polling uses (len, mtime) because the stub sink's publish
/// is a truncate+copy — a torn read manifests as a parse error, which
/// is reported and retried rather than fatal.
fn watch(path: &Path, secs: f64, save_png_dir: Option<&Path>) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs_f64(secs.max(0.0));
    let mut last_sig: Option<(u64, SystemTime)> = None;
    let mut records = 0u32;
    let mut gpu_tokens = 0u32;
    let mut cpu_frames = 0u32;
    let mut torn_reads = 0u32;
    let mut last_arrival: Option<Instant> = None;
    let mut min_gap = f64::INFINITY;
    let mut max_gap = 0.0f64;

    println!("watching {} for {:.1}s ...", path.display(), secs);
    while Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
        let Ok(meta) = std::fs::metadata(path) else {
            continue; // producer down / not created yet
        };
        let Ok(mtime) = meta.modified() else {
            continue;
        };
        let sig = (meta.len(), mtime);
        if last_sig.as_ref() == Some(&sig) {
            continue;
        }
        last_sig = Some(sig);
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        match parse_record(&bytes) {
            Ok(record) => {
                records += 1;
                match record {
                    BridgeRecord::GpuToken { .. } => gpu_tokens += 1,
                    BridgeRecord::CpuFrame { .. } => cpu_frames += 1,
                }
                if let Some(dir) = save_png_dir {
                    if let Err(e) = save_png(&record, dir) {
                        eprintln!("warn: {e}");
                    }
                }
                if let Some(prev) = last_arrival.replace(Instant::now()) {
                    let gap = prev.elapsed().as_secs_f64();
                    min_gap = min_gap.min(gap);
                    max_gap = max_gap.max(gap);
                }
                println!("{}", describe(&record));
            }
            Err(e) => {
                torn_reads += 1;
                eprintln!("warn: mid-copy / invalid record skipped: {e}");
            }
        }
    }

    println!(
        "summary: {} records ({} GPU tokens, {} CPU frames), {} torn reads",
        records, gpu_tokens, cpu_frames, torn_reads
    );
    if records >= 2 {
        println!(
            "arrival gap: min {:.1} ms, max {:.1} ms",
            min_gap * 1000.0,
            max_gap * 1000.0
        );
    } else {
        println!("no producer updates observed — is the app running with this sink selected?");
    }
    Ok(())
}

/// Produce a VGTK record (publishable token) and a VSTX record (CPU
/// fallback) through the real sink writer, then parse both back with
/// the consumer decoder. Mirrors the e2e fixtures in
/// `output::frame_sink` tests so the two sides can't drift silently.
fn self_test() -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!(
        "vulvatar_bridge_consumer_selftest_{}",
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let gpu_path = dir.join("gpu_token.bin");
    let cpu_path = dir.join("cpu_frame.bin");

    let mut sink = SharedMemoryFileSink::with_path(gpu_path.clone());
    let mut frame = OutputFrame::new(7, [1920, 1080], 123_456);
    frame.gpu_token = Some(GpuFrameToken {
        resource_id: 11,
        handle_type: ExternalHandleType::Win32Kmt,
        external_handle: Some(22),
        sync: OutputSyncToken::ProducerWaitComplete,
        lease: FrameLease {
            lease_id: 33,
            lifetime: FrameLifetimeContract::SingleConsumerRetained,
        },
    });
    frame.handoff_path = HandoffPath::GpuSharedFrame;
    frame.alpha_mode = AlphaMode::Premultiplied;
    frame.color_space = OutputColorSpace::Srgb;
    sink.write_frame(&frame).map_err(|e| e.to_string())?;

    let mut sink = SharedMemoryFileSink::with_path(cpu_path.clone());
    let mut frame = OutputFrame::new(8, [4, 4], 222_222);
    let pixels: std::sync::Arc<Vec<u8>> = std::sync::Arc::new(vec![128u8; 4 * 4 * 4]);
    frame.pixel_data = Some(pixels);
    frame.handoff_path = HandoffPath::CpuReadback;
    sink.write_frame(&frame).map_err(|e| e.to_string())?;

    let gpu_bytes = std::fs::read(&gpu_path).map_err(|e| e.to_string())?;
    let gpu = parse_record(&gpu_bytes)?;
    match &gpu {
        BridgeRecord::GpuToken {
            extent,
            handle_type,
            external_handle,
            frame_id,
            ..
        } => {
            if *extent != [1920, 1080] || *handle_type != 1 || *external_handle != 22 || *frame_id != 7
            {
                return Err(format!("VGTK round-trip mismatch: {gpu:?}"));
            }
        }
        _ => return Err("expected a GpuToken record".to_string()),
    }

    let cpu_bytes = std::fs::read(&cpu_path).map_err(|e| e.to_string())?;
    let cpu = parse_record(&cpu_bytes)?;
    match &cpu {
        BridgeRecord::CpuFrame {
            extent,
            frame_index,
            pixels,
            ..
        } => {
            if *extent != [4, 4] || *frame_index != 0 || pixels.len() != 64 {
                return Err(format!("VSTX round-trip mismatch: {cpu:?}"));
            }
        }
        _ => return Err("expected a CpuFrame record".to_string()),
    }

    println!("self-test: VGTK + VSTX round-trips decoded correctly ({})", dir.display());
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
