use log::debug;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use super::{
    AlphaMode, ExternalHandleType, GpuFrameToken, OutputColorSpace, OutputFrame, OutputSyncToken,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrameSink {
    VirtualCamera,
    /// File-backed bridge for the future GPU shared-texture path. Valid GPU
    /// tokens are serialized as metadata-only `VGTK` records; CPU fallback
    /// frames still use the legacy `VSTX` bytes-on-disk shape. The external
    /// consumer is still a stub, so the variant name remains intentionally
    /// conservative.
    SharedTextureFileStub,
    SharedMemory,
    ImageSequence,
}

impl FrameSink {
    /// Map a sink to the GUI inspector's combo box position. Kept on the
    /// type itself so the GUI and reconcile path agree without any party
    /// hard-coding integers.
    pub fn to_gui_index(&self) -> usize {
        match self {
            FrameSink::VirtualCamera => 0,
            FrameSink::SharedTextureFileStub => 1,
            FrameSink::SharedMemory => 2,
            FrameSink::ImageSequence => 3,
        }
    }

    /// Inverse of [`Self::to_gui_index`]. Out-of-range indices fall back to
    /// `ImageSequence` (matches the inspector's `_ =>` arm).
    pub fn from_gui_index(index: usize) -> Self {
        match index {
            0 => FrameSink::VirtualCamera,
            1 => FrameSink::SharedTextureFileStub,
            2 => FrameSink::SharedMemory,
            _ => FrameSink::ImageSequence,
        }
    }

    /// Whether the concrete writer produced by [`create_sink_writer`] for
    /// this variant accepts `OutputFrame::gpu_token` directly. The caller
    /// (`Application`) uses this to pick `RenderExportMode::GpuExport` vs
    /// `CpuReadback` without having to construct a writer just to ask. Must
    /// agree with the writer trait's `supports_gpu_tokens()` for the same
    /// variant — covered by `frame_sink_capability_matches_writer` below.
    ///
    /// On Windows the live `VirtualCamera` and `SharedMemory` sinks are
    /// backed by `Win32FileBackedSharedMemorySink`, which publishes a VGTK
    /// sidecar for GPU-token frames and falls back to RGBA bytes for CPU
    /// frames — so they advertise GPU-token support there. On non-Windows
    /// those variants fall back to other writers (RingBuffer /
    /// VirtualCameraFile) that only accept CPU pixels; the cfg-gated arms
    /// keep enum/writer agreement on every target.
    pub fn supports_gpu_tokens(&self) -> bool {
        match self {
            FrameSink::SharedTextureFileStub => true,
            #[cfg(target_os = "windows")]
            FrameSink::VirtualCamera => true,
            #[cfg(target_os = "windows")]
            FrameSink::SharedMemory => true,
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
pub enum FrameSinkQueuePolicy {
    DropOldest,
    DropNewest,
    ReplaceLatest,
    BlockNotAllowed,
}

pub trait OutputSinkWriter: Send {
    fn write_frame(&mut self, frame: &OutputFrame) -> Result<(), String>;
    fn name(&self) -> &str;

    fn supports_gpu_tokens(&self) -> bool {
        false
    }
}

pub struct ImageSequenceSink {
    output_dir: PathBuf,
    sequence: u64,
}

impl ImageSequenceSink {
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        let dir = output_dir.into();
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

        let img: image::ImageBuffer<image::Rgba<u8>, _> =
            image::ImageBuffer::from_raw(width, height, pixel_data.to_vec())
                .ok_or_else(|| "ImageSequenceSink: failed to construct image buffer".to_string())?;

        img.save(&path).map_err(|e| {
            format!(
                "ImageSequenceSink: failed to save '{}': {}",
                path.display(),
                e
            )
        })?;

        debug!(
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

const SHMEM_MAGIC: u32 = 0x564D_4654;
const SHMEM_HEADER_SIZE: usize = 24;

pub struct SharedMemorySink {
    path: PathBuf,
    sequence: u64,
    mapping_name: String,
}

impl SharedMemorySink {
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

        buf.extend_from_slice(&SHMEM_MAGIC.to_le_bytes());
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&self.sequence.to_le_bytes());

        buf.extend_from_slice(pixel_data);

        let tmp_path = self.path.with_extension("shmem.tmp");
        let mut file = fs::File::create(&tmp_path)
            .map_err(|e| format!("SharedMemorySink: create tmp: {}", e))?;
        file.write_all(&buf)
            .map_err(|e| format!("SharedMemorySink: write: {}", e))?;
        file.flush()
            .map_err(|e| format!("SharedMemorySink: flush: {}", e))?;
        drop(file);

        fs::copy(&tmp_path, &self.path).map_err(|e| format!("SharedMemorySink: copy: {}", e))?;
        let _ = fs::remove_file(&tmp_path);

        debug!(
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

const VCAM_MAGIC: u32 = 0x5643_4D46;
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

impl Default for VirtualCameraSink {
    fn default() -> Self {
        Self::new()
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

        buf.extend_from_slice(&VCAM_MAGIC.to_le_bytes());
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&self.sequence.to_le_bytes());

        buf.extend_from_slice(pixel_data);

        let tmp_path = self.output_dir.join("current_frame.raw.tmp");
        let mut file = fs::File::create(&tmp_path)
            .map_err(|e| format!("VirtualCameraSink: create tmp: {}", e))?;
        file.write_all(&buf)
            .map_err(|e| format!("VirtualCameraSink: write: {}", e))?;
        file.flush()
            .map_err(|e| format!("VirtualCameraSink: flush: {}", e))?;
        drop(file);

        fs::copy(&tmp_path, &self.frame_path)
            .map_err(|e| format!("VirtualCameraSink: copy: {}", e))?;
        let _ = fs::remove_file(&tmp_path);

        debug!(
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

const RING_MAGIC: u32 = 0x564D524E;
const RING_HEADER_SIZE: usize = 64;
const RING_MAX_SLOTS: usize = 4;

pub struct RingBufferSharedMemorySink {
    path: PathBuf,
    ready_path: PathBuf,
    sequence: u64,
    slot_count: usize,
    slot_size: usize,
}

impl RingBufferSharedMemorySink {
    pub fn new(mapping_name: &str, slot_count: usize) -> Self {
        let path = std::env::temp_dir().join(format!("{}.ringshmem", mapping_name));
        let ready_path = std::env::temp_dir().join(format!("{}.ringready", mapping_name));
        Self {
            path,
            ready_path,
            sequence: 0,
            slot_count: slot_count.clamp(1, RING_MAX_SLOTS),
            slot_size: 0,
        }
    }

    fn ensure_file(&mut self, total_size: usize) -> Result<fs::File, String> {
        let needs_recreate = self.slot_size != total_size;

        if needs_recreate {
            let header = RingBufferHeader {
                magic: RING_MAGIC,
                slot_count: self.slot_count as u32,
                slot_size: total_size as u64,
                write_sequence: 0,
                width: 0,
                height: 0,
            };
            let file_size = RING_HEADER_SIZE + self.slot_count * total_size;
            let mut buf = Vec::with_capacity(file_size);
            buf.extend_from_slice(&header.magic.to_le_bytes());
            buf.extend_from_slice(&header.slot_count.to_le_bytes());
            buf.extend_from_slice(&header.slot_size.to_le_bytes());
            buf.extend_from_slice(&header.write_sequence.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            while buf.len() < RING_HEADER_SIZE {
                buf.push(0);
            }
            buf.resize(file_size, 0);

            let tmp_path = self.path.with_extension("ringshmem.tmp");
            {
                let mut file = fs::File::create(&tmp_path)
                    .map_err(|e| format!("RingBuffer: create tmp: {}", e))?;
                file.write_all(&buf)
                    .map_err(|e| format!("RingBuffer: write: {}", e))?;
                file.flush()
                    .map_err(|e| format!("RingBuffer: flush: {}", e))?;
            }
            fs::rename(&tmp_path, &self.path).map_err(|e| format!("RingBuffer: rename: {}", e))?;
            self.slot_size = total_size;
        }

        fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)
            .map_err(|e| format!("RingBuffer: open: {}", e))
    }
}

#[repr(C)]
struct RingBufferHeader {
    magic: u32,
    slot_count: u32,
    slot_size: u64,
    write_sequence: u64,
    width: u32,
    height: u32,
}

impl OutputSinkWriter for RingBufferSharedMemorySink {
    fn write_frame(&mut self, frame: &OutputFrame) -> Result<(), String> {
        let pixel_data = match &frame.pixel_data {
            Some(data) => data,
            None => return Err("RingBuffer: frame has no pixel data".into()),
        };

        let width = frame.extent[0];
        let height = frame.extent[1];
        let slot_payload = 8 + pixel_data.len();
        let slot_total = slot_payload.div_ceil(16) * 16;

        let mut file = self.ensure_file(slot_total)?;

        let slot_index = (self.sequence % self.slot_count as u64) as usize;
        let offset = RING_HEADER_SIZE + slot_index * slot_total;

        let mut slot = Vec::with_capacity(slot_total);
        slot.extend_from_slice(&self.sequence.to_le_bytes());
        slot.extend_from_slice(pixel_data);
        slot.resize(slot_total, 0);

        {
            use std::io::{Seek, SeekFrom};
            file.seek(SeekFrom::Start(offset as u64))
                .map_err(|e| format!("RingBuffer: seek: {}", e))?;
            file.write_all(&slot)
                .map_err(|e| format!("RingBuffer: write slot: {}", e))?;
            file.flush()
                .map_err(|e| format!("RingBuffer: flush: {}", e))?;

            file.seek(SeekFrom::Start(16))
                .map_err(|e| format!("RingBuffer: seek header: {}", e))?;
            file.write_all(&self.sequence.to_le_bytes())
                .map_err(|e| format!("RingBuffer: write seq: {}", e))?;
            file.write_all(&width.to_le_bytes())
                .map_err(|e| format!("RingBuffer: write width: {}", e))?;
            file.write_all(&height.to_le_bytes())
                .map_err(|e| format!("RingBuffer: write height: {}", e))?;
        }

        let _ = fs::write(&self.ready_path, self.sequence.to_le_bytes());

        debug!(
            "RingBuffer: wrote slot {} (seq={}) to {}",
            slot_index,
            self.sequence,
            self.path.display()
        );

        self.sequence += 1;
        Ok(())
    }

    fn name(&self) -> &str {
        "ring-buffer-shared-memory"
    }
}

// ---------------------------------------------------------------------------
// VirtualCameraFileSink — OBS-friendly frame protocol
// ---------------------------------------------------------------------------

pub struct VirtualCameraFileSink {
    output_dir: PathBuf,
    sequence: u64,
    width: u32,
    height: u32,
}

impl VirtualCameraFileSink {
    pub fn new() -> Self {
        let output_dir = std::env::temp_dir().join("vulvatar-virtualcamera");
        let _ = fs::create_dir_all(&output_dir);
        Self {
            output_dir,
            sequence: 0,
            width: 0,
            height: 0,
        }
    }
}

impl Default for VirtualCameraFileSink {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputSinkWriter for VirtualCameraFileSink {
    fn write_frame(&mut self, frame: &OutputFrame) -> Result<(), String> {
        let pixel_data = match &frame.pixel_data {
            Some(data) => data,
            None => return Err("VirtualCameraFileSink: frame has no pixel data".into()),
        };

        let width = frame.extent[0];
        let height = frame.extent[1];

        let filename = format!("frame_{:06}.rgba", self.sequence);
        let frame_path = self.output_dir.join(&filename);

        let tmp_path = frame_path.with_extension("rgba.tmp");
        {
            let mut file = fs::File::create(&tmp_path)
                .map_err(|e| format!("VirtualCameraFileSink: create tmp: {}", e))?;
            file.write_all(pixel_data)
                .map_err(|e| format!("VirtualCameraFileSink: write: {}", e))?;
            file.flush()
                .map_err(|e| format!("VirtualCameraFileSink: flush: {}", e))?;
        }
        fs::copy(&tmp_path, &frame_path)
            .map_err(|e| format!("VirtualCameraFileSink: copy: {}", e))?;
        let _ = fs::remove_file(&tmp_path);

        self.width = width;
        self.height = height;

        let status = format!(
            "{{\"width\":{},\"height\":{},\"frame_rate\":30,\"frame_count\":{}}}",
            width,
            height,
            self.sequence + 1
        );
        let status_path = self.output_dir.join("status.json");
        let status_tmp = self.output_dir.join("status.json.tmp");
        {
            let mut file = fs::File::create(&status_tmp)
                .map_err(|e| format!("VirtualCameraFileSink: create status tmp: {}", e))?;
            file.write_all(status.as_bytes())
                .map_err(|e| format!("VirtualCameraFileSink: write status: {}", e))?;
            file.flush()
                .map_err(|e| format!("VirtualCameraFileSink: flush status: {}", e))?;
        }
        fs::copy(&status_tmp, &status_path)
            .map_err(|e| format!("VirtualCameraFileSink: copy status: {}", e))?;
        let _ = fs::remove_file(&status_tmp);

        if self.sequence > 0 {
            let old_frame = format!("frame_{:06}.rgba", self.sequence - 1);
            let _ = fs::remove_file(self.output_dir.join(&old_frame));
        }

        self.sequence += 1;
        Ok(())
    }

    fn name(&self) -> &str {
        "virtual-camera-file"
    }
}

// ---------------------------------------------------------------------------
// SharedMemoryFileSink ? shared-texture bridge stub
//
// This sink is intentionally still file-backed, but it now models the real
// handoff boundary:
//
// - `VGTK` records carry a valid exported GPU token and require no CPU pixels
// - `VSTX` records preserve the legacy CPU-readback fallback shape
//
// The consumer side is still a future piece of work, hence the public sink
// variant remains `SharedTextureFileStub`; the important change is that the
// output layer can now preserve whether a live GPU token or a CPU fallback
// reached the last hop.
// ---------------------------------------------------------------------------

const VSTX_MAGIC: [u8; 4] = [b'V', b'S', b'T', b'X'];
const VSTX_HEADER_SIZE: usize = 28;
const VGTK_MAGIC: [u8; 4] = [b'V', b'G', b'T', b'K'];
const VGTK_VERSION: u32 = 1;
const VGTK_HEADER_SIZE: usize = 72;

/// Bit 0 of the VGTK `flags` field tells the consumer to preserve the
/// renderer's RGBA alpha rather than forcing the output to opaque.
const VGTK_FLAG_PRESERVE_ALPHA: u32 = 1 << 0;
/// Bit 1 tells the consumer the renderer produced linear-RGB pixels rather
/// than sRGB-encoded ones. Identical semantics to the byte-buffer header's
/// `SHMEM_FLAG_LINEAR_COLOR_SPACE` so both sinks describe colour the same
/// way regardless of whether the consumer reads bytes or a shared texture.
const VGTK_FLAG_LINEAR_COLOR_SPACE: u32 = 1 << 1;

fn gpu_token_is_publishable(token: &GpuFrameToken) -> bool {
    token.has_external_handle() && !matches!(token.handle_type, ExternalHandleType::Unavailable)
}

fn vgtk_handle_type_code(handle_type: &ExternalHandleType) -> u32 {
    match handle_type {
        ExternalHandleType::Unavailable => 0,
        ExternalHandleType::Win32Kmt => 1,
        ExternalHandleType::D3D12Fence => 2,
        ExternalHandleType::VkSemaphore => 3,
        ExternalHandleType::SharedMemoryHandle => 4,
    }
}

fn vgtk_sync_parts(sync: &OutputSyncToken) -> (u32, u64) {
    match sync {
        OutputSyncToken::None => (0, 0),
        OutputSyncToken::ProducerWaitComplete => (1, 0),
        OutputSyncToken::FenceValue(value) => (2, *value),
        OutputSyncToken::SemaphoreHandle(handle) => (3, *handle),
    }
}

fn vgtk_color_space_code(c: &OutputColorSpace) -> u32 {
    match c {
        OutputColorSpace::Srgb => 0,
        OutputColorSpace::LinearSrgb => 1,
    }
}

fn vgtk_alpha_mode_code(a: &AlphaMode) -> u32 {
    match a {
        AlphaMode::Opaque => 0,
        AlphaMode::Premultiplied => 1,
        AlphaMode::Straight => 2,
    }
}

fn vgtk_flags_for_frame(frame: &OutputFrame) -> u32 {
    let mut flags = 0u32;
    if !matches!(frame.alpha_mode, AlphaMode::Opaque) {
        flags |= VGTK_FLAG_PRESERVE_ALPHA;
    }
    if matches!(frame.color_space, OutputColorSpace::LinearSrgb) {
        flags |= VGTK_FLAG_LINEAR_COLOR_SPACE;
    }
    flags
}

/// Encode a 72-byte VGTK record describing a published GPU texture token.
/// Shape is identical between `SharedMemoryFileSink` and the Win32
/// file-backed sinks so a future consumer can read one format regardless of
/// which producer wrote it. See `docs/output-interop.md` for the field map.
fn encode_vgtk_record(frame: &OutputFrame, token: &GpuFrameToken) -> Vec<u8> {
    let mut buf = Vec::with_capacity(VGTK_HEADER_SIZE);
    let (sync_type, sync_value) = vgtk_sync_parts(&token.sync);

    buf.extend_from_slice(&VGTK_MAGIC); // 0..4
    buf.extend_from_slice(&VGTK_VERSION.to_le_bytes()); // 4..8
    buf.extend_from_slice(&frame.extent[0].to_le_bytes()); // 8..12
    buf.extend_from_slice(&frame.extent[1].to_le_bytes()); // 12..16
    buf.extend_from_slice(&vgtk_handle_type_code(&token.handle_type).to_le_bytes()); // 16..20
    buf.extend_from_slice(&sync_type.to_le_bytes()); // 20..24
    buf.extend_from_slice(&token.external_handle.unwrap_or_default().to_le_bytes()); // 24..32
    buf.extend_from_slice(&sync_value.to_le_bytes()); // 32..40
    buf.extend_from_slice(&frame.frame_id.0.to_le_bytes()); // 40..48
    buf.extend_from_slice(&frame.timestamp.to_le_bytes()); // 48..56
    buf.extend_from_slice(&vgtk_color_space_code(&frame.color_space).to_le_bytes()); // 56..60
    buf.extend_from_slice(&vgtk_alpha_mode_code(&frame.alpha_mode).to_le_bytes()); // 60..64
    buf.extend_from_slice(&vgtk_flags_for_frame(frame).to_le_bytes()); // 64..68
    buf.extend_from_slice(&0u32.to_le_bytes()); // 68..72 reserved

    debug_assert_eq!(buf.len(), VGTK_HEADER_SIZE);
    buf
}

pub struct SharedMemoryFileSink {
    path: PathBuf,
    frame_index: u64,
}

impl SharedMemoryFileSink {
    pub fn new() -> Self {
        let path = std::env::temp_dir().join("vulvatar-shared-texture.bin");
        Self {
            path,
            frame_index: 0,
        }
    }

    /// Construct a sink writing to an explicit path instead of the
    /// platform's default `std::env::temp_dir()` location. Exposed for
    /// tests (per-test temp paths avoid collisions on concurrent runs)
    /// and any future caller wanting deterministic sidecar placement.
    pub fn with_path(path: PathBuf) -> Self {
        Self {
            path,
            frame_index: 0,
        }
    }

    fn write_atomically(&self, bytes: &[u8]) -> Result<(), String> {
        let tmp_path = self.path.with_extension("bin.tmp");
        {
            let mut file = fs::File::create(&tmp_path)
                .map_err(|e| format!("SharedMemoryFileSink: create tmp: {}", e))?;
            file.write_all(bytes)
                .map_err(|e| format!("SharedMemoryFileSink: write: {}", e))?;
            file.flush()
                .map_err(|e| format!("SharedMemoryFileSink: flush: {}", e))?;
        }
        fs::copy(&tmp_path, &self.path)
            .map_err(|e| format!("SharedMemoryFileSink: copy: {}", e))?;
        let _ = fs::remove_file(&tmp_path);
        Ok(())
    }

    fn encode_cpu_frame(&self, frame: &OutputFrame, pixel_data: &[u8]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(VSTX_HEADER_SIZE + pixel_data.len());
        buf.extend_from_slice(&VSTX_MAGIC);
        buf.extend_from_slice(&frame.extent[0].to_le_bytes());
        buf.extend_from_slice(&frame.extent[1].to_le_bytes());
        buf.extend_from_slice(&self.frame_index.to_le_bytes());
        buf.extend_from_slice(&frame.timestamp.to_le_bytes());
        buf.extend_from_slice(pixel_data);
        buf
    }
}

impl Default for SharedMemoryFileSink {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputSinkWriter for SharedMemoryFileSink {
    fn write_frame(&mut self, frame: &OutputFrame) -> Result<(), String> {
        if let Some(token) = frame
            .gpu_token
            .as_ref()
            .filter(|token| gpu_token_is_publishable(token))
        {
            let buf = encode_vgtk_record(frame, token);
            self.write_atomically(&buf)?;
            self.frame_index += 1;
            return Ok(());
        }

        let pixel_data = frame.pixel_data.as_deref().ok_or_else(|| {
            "SharedMemoryFileSink: frame has neither a valid GPU token nor pixel data".to_string()
        })?;
        let buf = self.encode_cpu_frame(frame, pixel_data);
        self.write_atomically(&buf)?;

        self.frame_index += 1;
        Ok(())
    }

    fn name(&self) -> &str {
        "shared-texture-file-bridge"
    }

    fn supports_gpu_tokens(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Null / stub sinks
// ---------------------------------------------------------------------------

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
        debug!("NullSink({}): frame ignored (not implemented)", self.label);
        Ok(())
    }

    fn name(&self) -> &str {
        self.label
    }
}

// ---------------------------------------------------------------------------
// Win32 Named Shared Memory Sink
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
mod win32_shmem {
    use super::{
        encode_vgtk_record, gpu_token_is_publishable, AlphaMode, GpuFrameToken, OutputColorSpace,
        OutputFrame, OutputSinkWriter,
    };
    use log::{debug, info};
    use std::ffi::c_void;
    use std::fs::File;
    use std::io::Write;
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::AsRawHandle;
    use std::path::{Path, PathBuf};
    use std::ptr;

    use crate::output::shmem_security::RawSecurityAttributes;

    const PAGE_READWRITE: u32 = 0x04;
    const FILE_MAP_ALL_ACCESS: u32 = 0xF001F;

    extern "system" {
        fn CreateFileMappingW(
            hFile: isize,
            lpFileMappingAttributes: *mut RawSecurityAttributes,
            flProtect: u32,
            dwMaximumSizeHigh: u32,
            dwMaximumSizeLow: u32,
            lpName: *const u16,
        ) -> isize;

        fn MapViewOfFile(
            hFileMappingObject: isize,
            dwDesiredAccess: u32,
            dwFileOffsetHigh: u32,
            dwFileOffsetLow: u32,
            dwNumberOfBytesToMap: usize,
        ) -> *mut c_void;

        fn UnmapViewOfFile(lpBaseAddress: *const c_void) -> i32;

        fn CloseHandle(hObject: isize) -> i32;

        fn GetLastError() -> u32;
    }

    const SHMEM_HEADER_SIZE: usize = 32;
    const SHMEM_MAGIC: u32 = 0x5643_4D46;

    /// Phase B-4: bit 0 in the shared-memory header `flags` u32 (offset 28)
    /// asks the consumer to preserve the source RGBA alpha channel rather
    /// than forcing the output to opaque. NV12 paths ignore the flag.
    const SHMEM_FLAG_PRESERVE_ALPHA: u32 = 1 << 0;
    /// Stage 4 (output color space): bit 1 of the same `flags` field tells
    /// the consumer that the producer wrote linear-RGB pixels rather than
    /// sRGB-encoded pixels. The MF DLL passes this through as
    /// `MF_MT_TRANSFER_FUNCTION = MFVideoTransFunc_10` on the IMFSample so
    /// downstream clients that honour sample-level colour attributes don't
    /// double-encode. Default (bit clear) means sRGB.
    const SHMEM_FLAG_LINEAR_COLOR_SPACE: u32 = 1 << 1;

    fn flags_for_frame(frame: &OutputFrame) -> u32 {
        let mut flags = 0u32;
        if !matches!(frame.alpha_mode, AlphaMode::Opaque) {
            flags |= SHMEM_FLAG_PRESERVE_ALPHA;
        }
        if matches!(frame.color_space, OutputColorSpace::LinearSrgb) {
            flags |= SHMEM_FLAG_LINEAR_COLOR_SPACE;
        }
        flags
    }

    // -----------------------------------------------------------------
    // Win32FileBackedSharedMemorySink
    //
    // A file-backed mapping bypasses the `Global\` namespace entirely:
    // both producer (this sink) and consumer (the MF virtual camera DLL
    // hosted in svchost / LocalService) open the same file by path and
    // create unnamed mappings of it. The kernel shares the underlying
    // physical pages between the two mappings — verified empirically:
    //   producer wrote 0xDEADBEEF → consumer read 0xDEADBEEF
    //   producer wrote 0xCAFEBABE → consumer read 0xCAFEBABE (live)
    //
    // Why this matters: a non-elevated interactive token under UAC
    // filtering does NOT carry SeCreateGlobalPrivilege, so any
    // `CreateFileMappingW(... "Global\\X")` from such a process fails
    // with ERROR_ACCESS_DENIED regardless of SECURITY_ATTRIBUTES or
    // name uniqueness. File-backed mappings do not consult that
    // privilege because they're not creating a named kernel object.
    //
    // The file lives at `%ProgramData%\VulVATAR\camera_frame_buffer.bin`,
    // which `dev.ps1` option 7 sets up with inherited
    // `NT AUTHORITY\LocalService:(RX)` so svchost can read it.
    //
    // Cache behaviour: the file is opened with FILE_ATTRIBUTE_TEMPORARY
    // so the cache manager keeps the dirty pages in RAM and doesn't
    // bother flushing to disk for transient frame data overwritten at
    // 60 Hz.
    // -----------------------------------------------------------------

    const FILE_SHARE_RW: u32 = 0x1 | 0x2; // FILE_SHARE_READ | FILE_SHARE_WRITE
    const FILE_ATTRIBUTE_TEMPORARY: u32 = 0x100;

    pub fn virtual_camera_file_path() -> PathBuf {
        let dir = std::env::var_os("ProgramData")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\ProgramData"));
        dir.join("VulVATAR").join("camera_frame_buffer.bin")
    }

    /// Distinct path for the SharedMemory sink so it doesn't collide
    /// with the VirtualCamera sink when the user toggles between them.
    /// Using a file-backed mapping here as well sidesteps the
    /// `Global\` named-section DACL race that plagued
    /// `Win32NamedSharedMemorySink` (see commit a4e2c66 — fix only
    /// covered the create-time race; existing-object handles created
    /// by other tokens still produced ERROR_ACCESS_DENIED).
    pub fn shared_memory_file_path() -> PathBuf {
        let dir = std::env::var_os("ProgramData")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\ProgramData"));
        dir.join("VulVATAR").join("shared_memory_buffer.bin")
    }

    pub struct Win32FileBackedSharedMemorySink {
        file: Option<File>,
        mapping_handle: isize,
        view: *mut c_void,
        file_path: PathBuf,
        /// Sidecar file carrying VGTK records for GPU-token frames. Lives
        /// next to `file_path` with a `.vgtk` extension swap so the consumer
        /// can locate it without an out-of-band agreement. Atomic
        /// tmp-rename keeps the sidecar internally consistent: a torn read
        /// is impossible because the consumer always opens a file that was
        /// fully written before its inode was published into the directory.
        sidecar_path: PathBuf,
        mapped_size: usize,
        sequence: u64,
        logged_first_write: bool,
        logged_first_sidecar_write: bool,
    }

    unsafe impl Send for Win32FileBackedSharedMemorySink {}

    impl Win32FileBackedSharedMemorySink {
        pub fn new(file_path: PathBuf) -> Self {
            let sidecar_path = file_path.with_extension("vgtk");
            Self {
                file: None,
                mapping_handle: 0,
                view: ptr::null_mut(),
                file_path,
                sidecar_path,
                mapped_size: 0,
                sequence: 0,
                logged_first_write: false,
                logged_first_sidecar_write: false,
            }
        }

        #[cfg(test)]
        pub fn sidecar_path(&self) -> &Path {
            &self.sidecar_path
        }

        fn write_sidecar_atomically(&self, bytes: &[u8]) -> Result<(), String> {
            if let Some(parent) = self.sidecar_path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    format!(
                        "Win32FileBackedSink: create_dir_all('{}') for sidecar failed: {}",
                        parent.display(),
                        e
                    )
                })?;
            }
            let tmp_path = self.sidecar_path.with_extension("vgtk.tmp");
            {
                let mut f = File::create(&tmp_path).map_err(|e| {
                    format!(
                        "Win32FileBackedSink: create sidecar tmp '{}' failed: {}",
                        tmp_path.display(),
                        e
                    )
                })?;
                f.write_all(bytes).map_err(|e| {
                    format!(
                        "Win32FileBackedSink: write sidecar tmp '{}' failed: {}",
                        tmp_path.display(),
                        e
                    )
                })?;
                f.flush().map_err(|e| {
                    format!(
                        "Win32FileBackedSink: flush sidecar tmp '{}' failed: {}",
                        tmp_path.display(),
                        e
                    )
                })?;
            }
            // `std::fs::rename` on Windows uses `MoveFileExW` with
            // `MOVEFILE_REPLACE_EXISTING`, so this overwrites any stale
            // sidecar from a slow consumer atomically. Failure modes that
            // remain: ERROR_SHARING_VIOLATION (consumer holds the file
            // without FILE_SHARE_DELETE), disk full, permission denied.
            // In any of those, the temp file is left behind without
            // cleanup, which a subsequent successful write would
            // overwrite via the same `File::create(&tmp_path)`. Best-
            // effort `remove_file` on rename failure keeps the disk
            // tidy across consumer hiccups; ignore the result because
            // even the cleanup may fail (locked file, etc.) and the
            // upstream error is the one the caller cares about.
            if let Err(e) = std::fs::rename(&tmp_path, &self.sidecar_path) {
                let _ = std::fs::remove_file(&tmp_path);
                return Err(format!(
                    "Win32FileBackedSink: rename sidecar '{}' -> '{}' failed: {}",
                    tmp_path.display(),
                    self.sidecar_path.display(),
                    e
                ));
            }
            Ok(())
        }

        fn write_gpu_token_sidecar(
            &mut self,
            frame: &OutputFrame,
            token: &GpuFrameToken,
        ) -> Result<(), String> {
            let buf = encode_vgtk_record(frame, token);
            self.write_sidecar_atomically(&buf)?;
            debug!(
                "Win32FileBackedSink: wrote VGTK sidecar to '{}' (seq={}, {}x{}, handle={:?})",
                self.sidecar_path.display(),
                self.sequence,
                frame.extent[0],
                frame.extent[1],
                token.external_handle,
            );
            if !self.logged_first_sidecar_write {
                info!(
                    "Win32FileBackedSink: first VGTK sidecar to '{}' (seq={}, {}x{})",
                    self.sidecar_path.display(),
                    self.sequence,
                    frame.extent[0],
                    frame.extent[1],
                );
                self.logged_first_sidecar_write = true;
            }
            Ok(())
        }

        fn open_file(path: &Path) -> std::io::Result<File> {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .share_mode(FILE_SHARE_RW)
                .attributes(FILE_ATTRIBUTE_TEMPORARY)
                .open(path)
        }

        fn ensure_mapping(&mut self, payload_size: usize) -> Result<(), String> {
            let total = SHMEM_HEADER_SIZE + payload_size;
            if total <= self.mapped_size && !self.view.is_null() {
                return Ok(());
            }

            unsafe { self.unmap_section() };

            if self.file.is_none() {
                let f = Self::open_file(&self.file_path).map_err(|e| {
                    format!(
                        "Win32FileBackedSink: open '{}' failed: {} \
                         (run dev.ps1 option 7 once to set up the directory + ACL)",
                        self.file_path.display(),
                        e
                    )
                })?;
                self.file = Some(f);
            }

            let file = self.file.as_ref().unwrap();
            file.set_len(total as u64).map_err(|e| {
                format!(
                    "Win32FileBackedSink: set_len({}) on '{}' failed: {}",
                    total,
                    self.file_path.display(),
                    e
                )
            })?;

            let file_handle = file.as_raw_handle() as isize;
            let handle = unsafe {
                CreateFileMappingW(
                    file_handle,
                    ptr::null_mut(),
                    PAGE_READWRITE,
                    (total >> 32) as u32,
                    total as u32,
                    ptr::null(),
                )
            };
            if handle == 0 {
                let err = unsafe { GetLastError() };
                return Err(format!(
                    "Win32FileBackedSink: CreateFileMappingW(file='{}', size={}) \
                     failed (GetLastError={})",
                    self.file_path.display(),
                    total,
                    err
                ));
            }

            let view = unsafe { MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, total) };
            if view.is_null() {
                let err = unsafe { GetLastError() };
                unsafe { CloseHandle(handle) };
                return Err(format!(
                    "Win32FileBackedSink: MapViewOfFile('{}') failed (GetLastError={})",
                    self.file_path.display(),
                    err
                ));
            }

            self.mapping_handle = handle;
            self.view = view;
            self.mapped_size = total;
            Ok(())
        }

        unsafe fn unmap_section(&mut self) {
            if !self.view.is_null() {
                unsafe { UnmapViewOfFile(self.view as *const c_void) };
                self.view = ptr::null_mut();
            }
            if self.mapping_handle != 0 {
                unsafe { CloseHandle(self.mapping_handle) };
                self.mapping_handle = 0;
            }
            self.mapped_size = 0;
            // Keep `self.file` open: closing it would force the
            // consumer to reopen on the next frame, which is a waste
            // when we're just resizing.
        }
    }

    impl OutputSinkWriter for Win32FileBackedSharedMemorySink {
        fn write_frame(&mut self, frame: &OutputFrame) -> Result<(), String> {
            // GPU shared-handle fast path. When the upstream renderer
            // produced a publishable GPU token, emit the VGTK sidecar and
            // skip the entire memory-mapped byte-buffer copy. The main
            // mapping is intentionally left at its previous (stale)
            // contents — a partial deploy (new producer, old DLL) sees the
            // old seq counter and reads stale bytes for one or two frames
            // until the DLL is restarted, rather than failing outright.
            // The sidecar's atomic tmp-rename + Option A lease policy
            // guarantee the shared texture is alive while the DLL holds it.
            if let Some(token) = frame
                .gpu_token
                .as_ref()
                .filter(|token| gpu_token_is_publishable(token))
            {
                self.write_gpu_token_sidecar(frame, token)?;
                self.sequence += 1;
                return Ok(());
            }

            let pixel_data = match &frame.pixel_data {
                Some(data) => data,
                None => return Err("Win32FileBackedSink: frame has no pixel data".into()),
            };

            let width = frame.extent[0];
            let height = frame.extent[1];
            let data_len = pixel_data.len();

            self.ensure_mapping(data_len)?;

            let flags = flags_for_frame(frame);

            unsafe {
                let base = self.view as *mut u8;

                let seq_in_progress = (self.sequence * 2 + 1).to_le_bytes();
                ptr::copy_nonoverlapping(seq_in_progress.as_ptr(), base.add(12), 8);

                let magic = SHMEM_MAGIC.to_le_bytes();
                ptr::copy_nonoverlapping(magic.as_ptr(), base, 4);

                let w_bytes = width.to_le_bytes();
                ptr::copy_nonoverlapping(w_bytes.as_ptr(), base.add(4), 4);

                let h_bytes = height.to_le_bytes();
                ptr::copy_nonoverlapping(h_bytes.as_ptr(), base.add(8), 4);

                let ts_bytes = frame.timestamp.to_le_bytes();
                ptr::copy_nonoverlapping(ts_bytes.as_ptr(), base.add(20), 8);

                let flag_bytes = flags.to_le_bytes();
                ptr::copy_nonoverlapping(flag_bytes.as_ptr(), base.add(28), 4);

                ptr::copy_nonoverlapping(
                    pixel_data.as_ptr(),
                    base.add(SHMEM_HEADER_SIZE),
                    data_len,
                );

                std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::Release);
                let seq_committed = (self.sequence * 2 + 2).to_le_bytes();
                ptr::copy_nonoverlapping(seq_committed.as_ptr(), base.add(12), 8);
            }

            debug!(
                "Win32FileBackedSink: wrote {} bytes to '{}' (seq={}, {}x{})",
                data_len,
                self.file_path.display(),
                self.sequence,
                width,
                height,
            );
            if !self.logged_first_write {
                info!(
                    "Win32FileBackedSink: first write {} bytes to '{}' (seq={}, {}x{})",
                    data_len,
                    self.file_path.display(),
                    self.sequence,
                    width,
                    height,
                );
                self.logged_first_write = true;
            }

            self.sequence += 1;
            Ok(())
        }

        fn name(&self) -> &str {
            "win32-file-backed-shared-memory"
        }

        fn supports_gpu_tokens(&self) -> bool {
            true
        }
    }

    impl Drop for Win32FileBackedSharedMemorySink {
        fn drop(&mut self) {
            unsafe { self.unmap_section() };
            // self.file drops here, closing the file handle.
        }
    }
}

#[cfg(target_os = "windows")]
use win32_shmem::{
    shared_memory_file_path, virtual_camera_file_path, Win32FileBackedSharedMemorySink,
};

// ---------------------------------------------------------------------------
// Factory helper
// ---------------------------------------------------------------------------

pub fn create_sink_writer(sink: &FrameSink) -> Box<dyn OutputSinkWriter> {
    match sink {
        FrameSink::ImageSequence => Box::new(ImageSequenceSink::new("output_frames")),
        FrameSink::SharedMemory => {
            // Was Win32NamedSharedMemorySink (Global\VulVATAR_Output);
            // file-backed sidesteps the DACL race that survived even
            // the permissive-SDDL fix when an existing object lingered
            // in the kernel namespace (see frame_sink.rs::win32_shmem
            // header). No in-tree consumer reads the named section, so
            // changing the transport breaks nothing.
            #[cfg(target_os = "windows")]
            {
                Box::new(Win32FileBackedSharedMemorySink::new(
                    shared_memory_file_path(),
                ))
            }
            #[cfg(not(target_os = "windows"))]
            {
                Box::new(RingBufferSharedMemorySink::new("VulVATAR_Output", 4))
            }
        }
        FrameSink::VirtualCamera => {
            #[cfg(target_os = "windows")]
            {
                Box::new(Win32FileBackedSharedMemorySink::new(
                    virtual_camera_file_path(),
                ))
            }
            #[cfg(not(target_os = "windows"))]
            {
                Box::new(VirtualCameraFileSink::new())
            }
        }
        FrameSink::SharedTextureFileStub => Box::new(SharedMemoryFileSink::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_handoff::{
        FrameLease, FrameLifetimeContract, HandoffPath, OutputFrameId,
    };
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_path(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!("vulvatar-{label}-{nanos}.bin"))
    }

    fn gpu_frame() -> OutputFrame {
        OutputFrame {
            frame_id: OutputFrameId(7),
            timestamp: 123_456,
            extent: [1920, 1080],
            color_space: OutputColorSpace::Srgb,
            alpha_mode: AlphaMode::Premultiplied,
            gpu_token: Some(GpuFrameToken {
                resource_id: 11,
                handle_type: ExternalHandleType::Win32Kmt,
                external_handle: Some(22),
                sync: OutputSyncToken::ProducerWaitComplete,
                lease: FrameLease {
                    lease_id: 33,
                    lifetime: FrameLifetimeContract::SingleConsumerImmediate,
                },
            }),
            handoff_path: HandoffPath::GpuSharedFrame,
            fallback_reason: None,
            pixel_data: None,
        }
    }

    #[test]
    fn shared_texture_bridge_writes_gpu_token_without_cpu_pixels() {
        let path = unique_temp_path("gpu-token");
        let mut sink = SharedMemoryFileSink::with_path(path.clone());

        sink.write_frame(&gpu_frame()).expect("write gpu token");

        let bytes = fs::read(&path).expect("read gpu token file");
        assert_eq!(&bytes[0..4], &VGTK_MAGIC);
        assert_eq!(bytes.len(), VGTK_HEADER_SIZE);
        assert!(sink.supports_gpu_tokens());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn frame_sink_capability_matches_writer() {
        // `Application` relies on `FrameSink::supports_gpu_tokens` to avoid
        // building a writer just to query GPU-token support. Drift between
        // the enum-level answer and the actual writer would silently route
        // GPU-token frames to a CPU-only sink and leave it returning
        // "frame has no pixel data" every tick.
        for sink in [
            FrameSink::VirtualCamera,
            FrameSink::SharedTextureFileStub,
            FrameSink::SharedMemory,
            FrameSink::ImageSequence,
        ] {
            let writer = create_sink_writer(&sink);
            assert_eq!(
                sink.supports_gpu_tokens(),
                writer.supports_gpu_tokens(),
                "{:?}: enum says {} but writer says {}",
                sink,
                sink.supports_gpu_tokens(),
                writer.supports_gpu_tokens(),
            );
        }
    }

    #[test]
    fn shared_texture_bridge_falls_back_to_cpu_bytes_when_token_is_not_publishable() {
        let path = unique_temp_path("cpu-fallback");
        let mut sink = SharedMemoryFileSink::with_path(path.clone());
        let mut frame = gpu_frame();
        frame.gpu_token.as_mut().expect("gpu token").external_handle = None;
        frame.pixel_data = Some(Arc::new(vec![1, 2, 3, 4]));

        sink.write_frame(&frame).expect("write cpu fallback");

        let bytes = fs::read(&path).expect("read cpu fallback file");
        assert_eq!(&bytes[0..4], &VSTX_MAGIC);
        assert_eq!(bytes.len(), VSTX_HEADER_SIZE + 4);
        let _ = fs::remove_file(path);
    }

    #[cfg(target_os = "windows")]
    mod win32_sink {
        use super::*;
        use crate::output::FallbackReason;

        fn small_cpu_frame() -> OutputFrame {
            OutputFrame {
                frame_id: OutputFrameId(1),
                timestamp: 0,
                extent: [2, 2],
                color_space: OutputColorSpace::Srgb,
                alpha_mode: AlphaMode::Premultiplied,
                gpu_token: None,
                handoff_path: HandoffPath::CpuReadback,
                fallback_reason: Some(FallbackReason::RequestedCpuReadback),
                pixel_data: Some(Arc::new(vec![0u8; 2 * 2 * 4])),
            }
        }

        /// A publishable GPU token must produce a `.vgtk` sidecar (atomic
        /// rename) and leave the main memory-mapped byte buffer untouched.
        /// Verifies W2 + W4 of the P2-05 phase 2 plan: capability flip plus
        /// real sidecar write.
        #[test]
        fn writes_vgtk_sidecar_for_publishable_gpu_token() {
            let path = unique_temp_path("win32-sink-gpu");
            let mut sink = Win32FileBackedSharedMemorySink::new(path.clone());

            assert!(sink.supports_gpu_tokens());

            // Make sure the main file does not exist before the test so the
            // "untouched" assertion below is meaningful.
            let _ = fs::remove_file(&path);
            let _ = fs::remove_file(sink.sidecar_path());

            sink.write_frame(&gpu_frame()).expect("write gpu token");

            let sidecar_bytes = fs::read(sink.sidecar_path()).expect("sidecar exists");
            assert_eq!(&sidecar_bytes[0..4], &VGTK_MAGIC);
            assert_eq!(sidecar_bytes.len(), VGTK_HEADER_SIZE);

            // The main mmap-backed file MUST NOT have been created — the
            // contract is "do not write pixel bytes on the GPU path".
            assert!(
                !path.exists(),
                "Win32 sink wrote main buffer despite publishable GPU token: '{}'",
                path.display()
            );

            let _ = fs::remove_file(sink.sidecar_path());
            let _ = fs::remove_file(&path);
        }

        /// A CPU-bytes frame must update the main mmap-backed file and
        /// leave any pre-existing sidecar in place (no stale clobbering,
        /// no spurious sidecar emit). Verifies W3 parity.
        #[test]
        fn cpu_fallback_leaves_sidecar_untouched_and_writes_bytes() {
            let path = unique_temp_path("win32-sink-cpu");
            let mut sink = Win32FileBackedSharedMemorySink::new(path.clone());

            // Seed a recognisable sidecar payload to prove the CPU path
            // does not touch it.
            let sentinel = b"SIDECAR-SENTINEL".to_vec();
            if let Some(parent) = sink.sidecar_path().parent() {
                std::fs::create_dir_all(parent).expect("mkdir sidecar parent");
            }
            std::fs::write(sink.sidecar_path(), &sentinel).expect("seed sidecar");

            sink.write_frame(&small_cpu_frame()).expect("write cpu");

            let sidecar_after = fs::read(sink.sidecar_path()).expect("sidecar still present");
            assert_eq!(
                sidecar_after, sentinel,
                "CPU-fallback path must not rewrite the sidecar"
            );

            // Main mmap file should now exist and carry the expected
            // header magic at offset 0.
            let main_bytes = fs::read(&path).expect("main buffer exists");
            assert!(main_bytes.len() >= 4, "main buffer must have a header");
            let magic = u32::from_le_bytes(main_bytes[0..4].try_into().unwrap());
            assert_eq!(magic, 0x5643_4D46, "main buffer header magic mismatch");

            let _ = fs::remove_file(sink.sidecar_path());
            let _ = fs::remove_file(&path);
        }
    }
}
