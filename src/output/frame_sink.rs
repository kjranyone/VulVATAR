use log::debug;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use super::{AlphaMode, OutputColorSpace, OutputFrame};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrameSink {
    VirtualCamera,
    SharedTexture,
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
            FrameSink::SharedTexture => 1,
            FrameSink::SharedMemory => 2,
            FrameSink::ImageSequence => 3,
        }
    }

    /// Inverse of [`Self::to_gui_index`]. Out-of-range indices fall back to
    /// `ImageSequence` (matches the inspector's `_ =>` arm).
    pub fn from_gui_index(index: usize) -> Self {
        match index {
            0 => FrameSink::VirtualCamera,
            1 => FrameSink::SharedTexture,
            2 => FrameSink::SharedMemory,
            _ => FrameSink::ImageSequence,
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
        let _ = fs::remove_file(&self.ready_path);
        let seq = self.inner.sequence;
        self.inner.write_frame(frame)?;
        let seq_bytes = seq.to_le_bytes();
        fs::write(&self.ready_path, seq_bytes)
            .map_err(|e| format!("SharedTextureSink: failed to write ready signal: {}", e))?;
        Ok(())
    }

    fn name(&self) -> &str {
        "SharedTexture"
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
            slot_count: slot_count.max(1).min(RING_MAX_SLOTS),
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
        let slot_total = ((slot_payload + 15) / 16) * 16;

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
// SharedMemoryFileSink — GPU shared texture file protocol (VSTX)
// ---------------------------------------------------------------------------

const VSTX_MAGIC: [u8; 4] = [b'V', b'S', b'T', b'X'];
const VSTX_HEADER_SIZE: usize = 28;

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
}

impl OutputSinkWriter for SharedMemoryFileSink {
    fn write_frame(&mut self, frame: &OutputFrame) -> Result<(), String> {
        let pixel_data = match &frame.pixel_data {
            Some(data) => data,
            None => return Err("SharedMemoryFileSink: frame has no pixel data".into()),
        };

        let width = frame.extent[0];
        let height = frame.extent[1];
        let timestamp_nanos = frame.timestamp;
        let total_len = VSTX_HEADER_SIZE + pixel_data.len();

        let mut buf: Vec<u8> = Vec::with_capacity(total_len);
        buf.extend_from_slice(&VSTX_MAGIC);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&self.frame_index.to_le_bytes());
        buf.extend_from_slice(&timestamp_nanos.to_le_bytes());
        buf.extend_from_slice(pixel_data);

        let tmp_path = self.path.with_extension("bin.tmp");
        {
            let mut file = fs::File::create(&tmp_path)
                .map_err(|e| format!("SharedMemoryFileSink: create tmp: {}", e))?;
            file.write_all(&buf)
                .map_err(|e| format!("SharedMemoryFileSink: write: {}", e))?;
            file.flush()
                .map_err(|e| format!("SharedMemoryFileSink: flush: {}", e))?;
        }
        fs::copy(&tmp_path, &self.path)
            .map_err(|e| format!("SharedMemoryFileSink: copy: {}", e))?;
        let _ = fs::remove_file(&tmp_path);

        self.frame_index += 1;
        Ok(())
    }

    fn name(&self) -> &str {
        "shared-memory-file-vstx"
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
    use super::{AlphaMode, OutputColorSpace, OutputFrame, OutputSinkWriter};
    use log::{debug, info};
    use std::ffi::{c_void, OsStr};
    use std::fs::File;
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::AsRawHandle;
    use std::path::{Path, PathBuf};
    use std::ptr;

    const PAGE_READWRITE: u32 = 0x04;
    const FILE_MAP_ALL_ACCESS: u32 = 0xF001F;
    const SDDL_REVISION_1: u32 = 1;

    // Win32 GetLastError codes used to diagnose CreateFileMappingW failures.
    // Documented under WinError.h.
    const ERROR_PRIVILEGE_NOT_HELD: u32 = 0x522; // 1314
    const ERROR_ACCESS_DENIED: u32 = 0x5; // 5

    extern "system" {
        fn CreateFileMappingW(
            hFile: isize,
            lpFileMappingAttributes: *mut SecurityAttributes,
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

        fn ConvertStringSecurityDescriptorToSecurityDescriptorW(
            string_security_descriptor: *const u16,
            string_sd_revision: u32,
            security_descriptor: *mut *mut c_void,
            security_descriptor_size: *mut u32,
        ) -> i32;

        fn LocalFree(mem: *mut c_void) -> *mut c_void;
    }

    #[repr(C)]
    struct SecurityAttributes {
        n_length: u32,
        security_descriptor: *mut c_void,
        inherit_handle: i32,
    }

    const SHMEM_HEADER_SIZE: usize = 32;
    const SHMEM_MAGIC: u32 = 0x5643_4D46;

    pub struct Win32NamedSharedMemorySink {
        handle: isize,
        view: *mut c_void,
        name: String,
        mapped_size: usize,
        sequence: u64,
        logged_first_write: bool,
    }

    unsafe impl Send for Win32NamedSharedMemorySink {}

    impl Win32NamedSharedMemorySink {
        pub fn new(name: &str) -> Self {
            Self {
                handle: 0,
                view: ptr::null_mut(),
                name: format!("Global\\{}", name),
                mapped_size: 0,
                sequence: 0,
                logged_first_write: false,
            }
        }

        fn ensure_mapping(&mut self, size: usize) -> Result<(), String> {
            let total = SHMEM_HEADER_SIZE + size;
            if total <= self.mapped_size && !self.view.is_null() {
                return Ok(());
            }

            unsafe { self.unmap() };

            let wide_name: Vec<u16> = OsStr::new(&self.name)
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();

            let mut security = MappingSecurity::new()?;
            let handle = unsafe {
                CreateFileMappingW(
                    -1isize,
                    security.as_mut_ptr(),
                    PAGE_READWRITE,
                    (total >> 32) as u32,
                    total as u32,
                    wide_name.as_ptr(),
                )
            };

            if handle == 0 {
                let err = unsafe { GetLastError() };
                let hint = match err {
                    ERROR_PRIVILEGE_NOT_HELD => " (SeCreateGlobalPrivilege missing — run as a normal interactive user, not from a non-Session-0 service or a stripped token)",
                    ERROR_ACCESS_DENIED => " (an existing object with the same name has a stricter DACL — check whether another process already opened it)",
                    _ => "",
                };
                return Err(format!(
                    "CreateFileMappingW failed for '{}' (GetLastError={}){}",
                    self.name, err, hint,
                ));
            }

            let view = unsafe { MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, total) };

            if view.is_null() {
                let err = unsafe { GetLastError() };
                unsafe { CloseHandle(handle) };
                return Err(format!(
                    "MapViewOfFile failed for '{}' (GetLastError={})",
                    self.name, err
                ));
            }

            self.handle = handle;
            self.view = view;
            self.mapped_size = total;
            Ok(())
        }

        unsafe fn unmap(&mut self) {
            if !self.view.is_null() {
                unsafe { UnmapViewOfFile(self.view as *const c_void) };
                self.view = ptr::null_mut();
            }
            if self.handle != 0 {
                unsafe { CloseHandle(self.handle) };
                self.handle = 0;
            }
            self.mapped_size = 0;
        }
    }

    struct MappingSecurity {
        descriptor: *mut c_void,
        attributes: SecurityAttributes,
    }

    impl MappingSecurity {
        fn new() -> Result<Self, String> {
            // Interactive users write the frames; FrameServer runs as
            // LocalService and only needs read access to consume them.
            // (Used by the SharedMemory sink only — VirtualCamera now
            // uses Win32FileBackedSharedMemorySink to sidestep the
            // SeCreateGlobalPrivilege gate on `Global\` named sections.)
            let sddl = "D:P(A;;GA;;;SY)(A;;GA;;;BA)(A;;GA;;;IU)(A;;GR;;;LS)";
            let wide: Vec<u16> = OsStr::new(sddl)
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();
            let mut descriptor: *mut c_void = ptr::null_mut();
            let ok = unsafe {
                ConvertStringSecurityDescriptorToSecurityDescriptorW(
                    wide.as_ptr(),
                    SDDL_REVISION_1,
                    &mut descriptor,
                    ptr::null_mut(),
                )
            };
            if ok == 0 || descriptor.is_null() {
                return Err("ConvertStringSecurityDescriptorToSecurityDescriptorW failed".into());
            }
            Ok(Self {
                descriptor,
                attributes: SecurityAttributes {
                    n_length: std::mem::size_of::<SecurityAttributes>() as u32,
                    security_descriptor: descriptor,
                    inherit_handle: 0,
                },
            })
        }

        fn as_mut_ptr(&mut self) -> *mut SecurityAttributes {
            &mut self.attributes
        }
    }

    impl Drop for MappingSecurity {
        fn drop(&mut self) {
            if !self.descriptor.is_null() {
                unsafe {
                    let _ = LocalFree(self.descriptor);
                }
            }
        }
    }

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

    impl OutputSinkWriter for Win32NamedSharedMemorySink {
        fn write_frame(&mut self, frame: &OutputFrame) -> Result<(), String> {
            let pixel_data = match &frame.pixel_data {
                Some(data) => data,
                None => return Err("Win32ShmemSink: frame has no pixel data".into()),
            };

            let width = frame.extent[0];
            let height = frame.extent[1];
            let data_len = pixel_data.len();

            self.ensure_mapping(data_len)?;

            // Pack alpha + colour-space intent into the header `flags` field
            // at offset 28. OutputFrame.{alpha_mode, color_space} are set by
            // the host (App::process_render_result) from the GUI toggles.
            let flags = flags_for_frame(frame);

            unsafe {
                // Cast to `*mut u8` so byte-offset arithmetic is well-defined.
                // (Historical note: this used to be `*mut ()` which made
                // `base.add(N)` a no-op since `size_of::<()>() == 0` —
                // every header field overwrote offset 0, the on-wire magic
                // was garbled, the DLL discarded every frame, and Frame
                // Server tore the camera down. The struct now stores
                // `*mut c_void` so the type system can't recreate that
                // bug, but the explicit `*mut u8` cast keeps the intent
                // visible.)
                let base = self.view as *mut u8;

                let magic = SHMEM_MAGIC.to_le_bytes();
                ptr::copy_nonoverlapping(magic.as_ptr(), base, 4);

                let w_bytes = width.to_le_bytes();
                ptr::copy_nonoverlapping(w_bytes.as_ptr(), base.add(4), 4);

                let h_bytes = height.to_le_bytes();
                ptr::copy_nonoverlapping(h_bytes.as_ptr(), base.add(8), 4);

                let seq_bytes = self.sequence.to_le_bytes();
                ptr::copy_nonoverlapping(seq_bytes.as_ptr(), base.add(12), 8);

                let ts_bytes = frame.timestamp.to_le_bytes();
                ptr::copy_nonoverlapping(ts_bytes.as_ptr(), base.add(20), 8);

                let flag_bytes = flags.to_le_bytes();
                ptr::copy_nonoverlapping(flag_bytes.as_ptr(), base.add(28), 4);

                ptr::copy_nonoverlapping(
                    pixel_data.as_ptr(),
                    base.add(SHMEM_HEADER_SIZE),
                    data_len,
                );
            }

            debug!(
                "Win32ShmemSink: wrote {} bytes to '{}' (seq={}, {}x{})",
                data_len, self.name, self.sequence, width, height,
            );
            if !self.logged_first_write {
                info!(
                    "Win32ShmemSink: first write {} bytes to '{}' (seq={}, {}x{})",
                    data_len, self.name, self.sequence, width, height,
                );
                self.logged_first_write = true;
            }

            self.sequence += 1;
            Ok(())
        }

        fn name(&self) -> &str {
            "win32-named-shared-memory"
        }
    }

    impl Drop for Win32NamedSharedMemorySink {
        fn drop(&mut self) {
            unsafe { self.unmap() };
        }
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

    pub struct Win32FileBackedSharedMemorySink {
        file: Option<File>,
        mapping_handle: isize,
        view: *mut c_void,
        file_path: PathBuf,
        mapped_size: usize,
        sequence: u64,
        logged_first_write: bool,
    }

    unsafe impl Send for Win32FileBackedSharedMemorySink {}

    impl Win32FileBackedSharedMemorySink {
        pub fn new(file_path: PathBuf) -> Self {
            Self {
                file: None,
                mapping_handle: 0,
                view: ptr::null_mut(),
                file_path,
                mapped_size: 0,
                sequence: 0,
                logged_first_write: false,
            }
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

                let magic = SHMEM_MAGIC.to_le_bytes();
                ptr::copy_nonoverlapping(magic.as_ptr(), base, 4);

                let w_bytes = width.to_le_bytes();
                ptr::copy_nonoverlapping(w_bytes.as_ptr(), base.add(4), 4);

                let h_bytes = height.to_le_bytes();
                ptr::copy_nonoverlapping(h_bytes.as_ptr(), base.add(8), 4);

                let seq_bytes = self.sequence.to_le_bytes();
                ptr::copy_nonoverlapping(seq_bytes.as_ptr(), base.add(12), 8);

                let ts_bytes = frame.timestamp.to_le_bytes();
                ptr::copy_nonoverlapping(ts_bytes.as_ptr(), base.add(20), 8);

                let flag_bytes = flags.to_le_bytes();
                ptr::copy_nonoverlapping(flag_bytes.as_ptr(), base.add(28), 4);

                ptr::copy_nonoverlapping(
                    pixel_data.as_ptr(),
                    base.add(SHMEM_HEADER_SIZE),
                    data_len,
                );
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
    virtual_camera_file_path, Win32FileBackedSharedMemorySink, Win32NamedSharedMemorySink,
};

// ---------------------------------------------------------------------------
// Factory helper
// ---------------------------------------------------------------------------

pub fn create_sink_writer(sink: &FrameSink) -> Box<dyn OutputSinkWriter> {
    match sink {
        FrameSink::ImageSequence => Box::new(ImageSequenceSink::new("output_frames")),
        FrameSink::SharedMemory => {
            #[cfg(target_os = "windows")]
            {
                Box::new(Win32NamedSharedMemorySink::new("VulVATAR_Output"))
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
        FrameSink::SharedTexture => Box::new(SharedMemoryFileSink::new()),
    }
}
