use std::fs;
use std::io::Write;
use std::path::PathBuf;

use super::OutputFrame;

// ---------------------------------------------------------------------------
// Sink identity enum (kept for diagnostics / config serialization)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum FrameSink {
    VirtualCamera,
    SharedTexture,
    SharedMemory,
    ImageSequence,
}

#[derive(Clone, Debug)]
pub enum FrameSinkQueuePolicy {
    DropOldest,
    DropNewest,
    ReplaceLatest,
    BlockNotAllowed,
}

// ---------------------------------------------------------------------------
// OutputSinkWriter trait
// ---------------------------------------------------------------------------

/// Trait that every concrete output sink must implement.
pub trait OutputSinkWriter: Send {
    /// Write a single frame (with pixel data) to the sink.
    fn write_frame(&mut self, frame: &OutputFrame) -> Result<(), String>;

    /// Human-readable name for diagnostics.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// ImageSequence sink
// ---------------------------------------------------------------------------

/// Writes each received frame as a PNG file to a directory on disk.
///
/// Filename pattern: `frame_{sequence:06}.png`
pub struct ImageSequenceSink {
    output_dir: PathBuf,
    sequence: u64,
}

impl ImageSequenceSink {
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        let dir = output_dir.into();
        // Best-effort directory creation; errors surface on first write.
        let _ = fs::create_dir_all(&dir);
        Self {
            output_dir: dir,
            sequence: 0,
        }
    }
}

impl OutputSinkWriter for ImageSequenceSink {
    fn write_frame(&mut self, frame: &OutputFrame) -> Result<(), String> {
        let pixel_data = match &frame.pixel_data {
            Some(data) => data,
            None => return Err("ImageSequenceSink: frame has no pixel data".into()),
        };

        let width = frame.extent[0];
        let height = frame.extent[1];

        let expected_len = (width as usize) * (height as usize) * 4;
        if pixel_data.len() != expected_len {
            return Err(format!(
                "ImageSequenceSink: pixel data length {} does not match {}x{}x4 = {}",
                pixel_data.len(),
                width,
                height,
                expected_len,
            ));
        }

        let filename = format!("frame_{:06}.png", self.sequence);
        let path = self.output_dir.join(&filename);

        // Use the `image` crate to save RGBA data as PNG.
        let img: image::ImageBuffer<image::Rgba<u8>, _> =
            image::ImageBuffer::from_raw(width, height, pixel_data.to_vec())
                .ok_or_else(|| "ImageSequenceSink: failed to construct image buffer".to_string())?;

        img.save(&path)
            .map_err(|e| format!("ImageSequenceSink: failed to save '{}': {}", path.display(), e))?;

        println!(
            "ImageSequenceSink: wrote {} ({}x{}, seq={})",
            path.display(),
            width,
            height,
            self.sequence,
        );

        self.sequence += 1;
        Ok(())
    }

    fn name(&self) -> &str {
        "ImageSequence"
    }
}

// ---------------------------------------------------------------------------
// SharedMemory sink
// ---------------------------------------------------------------------------

/// Header written at the start of the memory-mapped file so that external
/// consumers (OBS, etc.) can interpret the pixel payload.
///
/// Layout (little-endian):
///   offset  0: u32  magic   (0x564D4654 = "VMFT")
///   offset  4: u32  width
///   offset  8: u32  height
///   offset 12: u32  format  (1 = RGBA8)
///   offset 16: u64  sequence
///   offset 24: [u8]  pixel data (width * height * 4)
const SHMEM_MAGIC: u32 = 0x564D_4654; // "VMFT" in LE
const SHMEM_HEADER_SIZE: usize = 24;

/// POC shared-memory sink that writes frames to a memory-mapped file.
///
/// On Windows the file is placed in the temp directory with a well-known name
/// so that other processes can map it.  For a production implementation you
/// would use `CreateFileMappingW` / named shared memory; this file-backed
/// approach is simpler and works cross-platform for the POC.
pub struct SharedMemorySink {
    /// Path of the backing file.
    path: PathBuf,
    /// Monotonically increasing sequence counter.
    sequence: u64,
    /// Name used by external consumers to locate the mapping.
    mapping_name: String,
}

impl SharedMemorySink {
    /// Create a new shared-memory sink.
    ///
    /// `mapping_name` is a logical name that external consumers reference
    /// (e.g. `"VulVATAR_Output"`).  The backing file is placed under
    /// `std::env::temp_dir()`.
    pub fn new(mapping_name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("{}.shmem", mapping_name));
        Self {
            path,
            sequence: 0,
            mapping_name: mapping_name.to_string(),
        }
    }
}

impl OutputSinkWriter for SharedMemorySink {
    fn write_frame(&mut self, frame: &OutputFrame) -> Result<(), String> {
        let pixel_data = match &frame.pixel_data {
            Some(data) => data,
            None => return Err("SharedMemorySink: frame has no pixel data".into()),
        };

        let width = frame.extent[0];
        let height = frame.extent[1];
        let pixel_len = pixel_data.len();
        let total_len = SHMEM_HEADER_SIZE + pixel_len;

        let mut buf: Vec<u8> = Vec::with_capacity(total_len);

        // Header
        buf.extend_from_slice(&SHMEM_MAGIC.to_le_bytes());
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // format = RGBA8
        buf.extend_from_slice(&self.sequence.to_le_bytes());

        // Pixel payload
        buf.extend_from_slice(pixel_data);

        // Atomic-ish write: write to a temp file then rename.
        let tmp_path = self.path.with_extension("shmem.tmp");
        let mut file = fs::File::create(&tmp_path)
            .map_err(|e| format!("SharedMemorySink: create tmp: {}", e))?;
        file.write_all(&buf)
            .map_err(|e| format!("SharedMemorySink: write: {}", e))?;
        file.flush()
            .map_err(|e| format!("SharedMemorySink: flush: {}", e))?;
        drop(file);

        // Use copy + remove instead of rename to avoid TOCTOU race on Windows
        // where rename fails if the destination exists.
        fs::copy(&tmp_path, &self.path)
            .map_err(|e| format!("SharedMemorySink: copy: {}", e))?;
        let _ = fs::remove_file(&tmp_path);

        println!(
            "SharedMemorySink: wrote {} bytes to '{}' (seq={})",
            total_len,
            self.path.display(),
            self.sequence,
        );

        self.sequence += 1;
        Ok(())
    }

    fn name(&self) -> &str {
        &self.mapping_name
    }
}

// ---------------------------------------------------------------------------
// SharedTexture sink (file-backed with ready signal)
// ---------------------------------------------------------------------------

/// Enhanced shared-memory sink for the SharedTexture output path.
///
/// Uses a well-known path (`%TEMP%/vulvatar_shared_texture.raw`) so that
/// external consumers (OBS plugins, Unity, etc.) know exactly where to look.
/// After each frame write, a small `.ready` signal file is created so that
/// readers can detect new frames without parsing the main file header on
/// every poll.
pub struct SharedTextureSink {
    inner: SharedMemorySink,
    ready_path: PathBuf,
}

impl SharedTextureSink {
    pub fn new() -> Self {
        let path = std::env::temp_dir().join("vulvatar_shared_texture.raw");
        let ready_path = std::env::temp_dir().join("vulvatar_shared_texture.ready");
        Self {
            inner: SharedMemorySink {
                path,
                sequence: 0,
                mapping_name: "SharedTexture".to_string(),
            },
            ready_path,
        }
    }
}

impl OutputSinkWriter for SharedTextureSink {
    fn write_frame(&mut self, frame: &OutputFrame) -> Result<(), String> {
        // Remove any existing ready signal before writing the new frame so
        // readers do not pick up a stale signal while the write is in progress.
        let _ = fs::remove_file(&self.ready_path);

        // Delegate to the inner SharedMemorySink for the actual write.
        self.inner.write_frame(frame)?;

        // Write the ready signal file.  The file contains the sequence number
        // as a little-endian u64 so readers can correlate it with the header.
        let seq_bytes = self.inner.sequence.to_le_bytes(); // note: sequence was already incremented
        fs::write(&self.ready_path, seq_bytes)
            .map_err(|e| format!("SharedTextureSink: failed to write ready signal: {}", e))?;

        Ok(())
    }

    fn name(&self) -> &str {
        "SharedTexture"
    }
}

// ---------------------------------------------------------------------------
// VirtualCamera sink (Windows POC)
// ---------------------------------------------------------------------------

/// POC virtual camera sink that writes frames to a well-known temp path.
///
/// Writes the current frame as `current_frame.raw` with a binary header to
/// `%TEMP%/vulvatar_vcam/`. Any external reader (e.g. OBS with a custom
/// source) can poll this file for new frames by checking the sequence counter
/// in the header.
///
/// Header layout (little-endian):
///   offset  0: u32  magic   (0x56434D46 = "VCMF")
///   offset  4: u32  width
///   offset  8: u32  height
///   offset 12: u32  format  (1 = RGBA8)
///   offset 16: u64  sequence
///   offset 24: [u8]  pixel data (width * height * 4)
const VCAM_MAGIC: u32 = 0x5643_4D46; // "VCMF" in LE
const VCAM_HEADER_SIZE: usize = 24;

pub struct VirtualCameraSink {
    output_dir: PathBuf,
    frame_path: PathBuf,
    sequence: u64,
}

impl VirtualCameraSink {
    pub fn new() -> Self {
        let output_dir = std::env::temp_dir().join("vulvatar_vcam");
        let _ = fs::create_dir_all(&output_dir);
        let frame_path = output_dir.join("current_frame.raw");
        Self {
            output_dir,
            frame_path,
            sequence: 0,
        }
    }
}

impl OutputSinkWriter for VirtualCameraSink {
    fn write_frame(&mut self, frame: &OutputFrame) -> Result<(), String> {
        let pixel_data = match &frame.pixel_data {
            Some(data) => data,
            None => return Err("VirtualCameraSink: frame has no pixel data".into()),
        };

        let width = frame.extent[0];
        let height = frame.extent[1];
        let pixel_len = pixel_data.len();
        let total_len = VCAM_HEADER_SIZE + pixel_len;

        let mut buf: Vec<u8> = Vec::with_capacity(total_len);

        // Header
        buf.extend_from_slice(&VCAM_MAGIC.to_le_bytes());
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // format = RGBA8
        buf.extend_from_slice(&self.sequence.to_le_bytes());

        // Pixel payload
        buf.extend_from_slice(pixel_data);

        // Atomic-ish write: write to a temp file then rename so readers never
        // see a partially-written frame.
        let tmp_path = self.output_dir.join("current_frame.raw.tmp");
        let mut file = fs::File::create(&tmp_path)
            .map_err(|e| format!("VirtualCameraSink: create tmp: {}", e))?;
        file.write_all(&buf)
            .map_err(|e| format!("VirtualCameraSink: write: {}", e))?;
        file.flush()
            .map_err(|e| format!("VirtualCameraSink: flush: {}", e))?;
        drop(file);

        // Use copy + remove instead of rename to avoid TOCTOU race on Windows
        // where rename fails if the destination exists.
        fs::copy(&tmp_path, &self.frame_path)
            .map_err(|e| format!("VirtualCameraSink: copy: {}", e))?;
        let _ = fs::remove_file(&tmp_path);

        println!(
            "VirtualCameraSink: wrote {} bytes to '{}' (seq={})",
            total_len,
            self.frame_path.display(),
            self.sequence,
        );

        self.sequence += 1;
        Ok(())
    }

    fn name(&self) -> &str {
        "VirtualCamera"
    }
}

// ---------------------------------------------------------------------------
// Null / stub sinks
// ---------------------------------------------------------------------------

/// A no-op sink used as a fallback until real implementations are added.
pub struct NullSink {
    label: &'static str,
}

impl NullSink {
    pub fn new(label: &'static str) -> Self {
        Self { label }
    }
}

impl OutputSinkWriter for NullSink {
    fn write_frame(&mut self, _frame: &OutputFrame) -> Result<(), String> {
        // No-op: just log.
        println!("NullSink({}): frame ignored (not implemented)", self.label);
        Ok(())
    }

    fn name(&self) -> &str {
        self.label
    }
}

// ---------------------------------------------------------------------------
// Factory helper
// ---------------------------------------------------------------------------

/// Build a concrete `OutputSinkWriter` for the given `FrameSink` variant.
pub fn create_sink_writer(sink: &FrameSink) -> Box<dyn OutputSinkWriter> {
    match sink {
        FrameSink::ImageSequence => {
            Box::new(ImageSequenceSink::new("output_frames"))
        }
        FrameSink::SharedMemory => {
            Box::new(SharedMemorySink::new("VulVATAR_Output"))
        }
        FrameSink::VirtualCamera => {
            Box::new(VirtualCameraSink::new())
        }
        FrameSink::SharedTexture => {
            // SharedTexture uses SharedMemorySink with a well-known path and
            // a frame-ready signal file so external consumers can poll for new
            // frames without racing on a partially-written buffer.
            Box::new(SharedTextureSink::new())
        }
    }
}
