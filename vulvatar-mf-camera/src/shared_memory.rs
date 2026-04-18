//! Reader for the named shared-memory frame buffer that
//! `vulvatar::output::frame_sink::Win32NamedSharedMemorySink` writes into.
//!
//! Wire format (little-endian, 32-byte header followed by RGBA pixels):
//! ```text
//! offset 0   u32  magic     = 0x5643_4D46  ("FMCV")
//! offset 4   u32  width
//! offset 8   u32  height
//! offset 12  u64  sequence  (incremented on every frame)
//! offset 20  u64  timestamp (frame timestamp; nanoseconds since some epoch)
//! offset 28  u32  reserved  (padding to 32 bytes)
//! offset 32  ...  width * height * 4 RGBA bytes
//! ```
//!
//! Per the MS Frame Server docs the named object lives in the global
//! `Local\` namespace so a SYSTEM-hosted Frame Server process can see what
//! a normal user-mode `vulvatar.exe` writes.

use core::sync::atomic::{AtomicI64, Ordering};
use std::ffi::OsStr;
use std::os::windows::ffi::OsStrExt;
use std::sync::Mutex;

use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::Memory::{
    MapViewOfFile, OpenFileMappingW, UnmapViewOfFile, FILE_MAP_READ, MEMORY_MAPPED_VIEW_ADDRESS,
};

const SHMEM_NAME: &str = "Local\\VulVATAR_VirtualCamera";
const SHMEM_HEADER_SIZE: usize = 32;
const SHMEM_MAGIC: u32 = 0x5643_4D46;

/// Bit 0 of the producer-written `flags` u32 at header offset 28. When
/// set, the RGB32 conversion path keeps the source RGBA's alpha byte
/// instead of forcing 0xFF; NV12 paths ignore this (no alpha channel).
pub const FRAME_FLAG_PRESERVE_ALPHA: u32 = 1 << 0;

/// One frame view returned from the shared memory.
pub struct FrameView {
    pub width: u32,
    pub height: u32,
    pub sequence: u64,
    /// Frame timestamp the producer wrote. Currently unused on the
    /// consumer side — MF samples carry their own timestamp derived from
    /// the negotiated framerate — but kept around so the wire format
    /// matches the producer one-to-one.
    #[allow(dead_code)]
    pub timestamp_ns: u64,
    /// Producer-supplied flags (see `FRAME_FLAG_*` constants).
    pub flags: u32,
    /// RGBA pixels. `len() == width * height * 4`.
    pub pixels: Vec<u8>,
}

pub struct SharedMemoryReader {
    state: Mutex<Inner>,
    /// Last sequence handed back by [`Self::read_if_new`] — exposed as an
    /// atomic so the hot path doesn't need to lock just to see whether a
    /// new frame is available. Protected by `state` for writes.
    last_sequence: AtomicI64,
}

struct Inner {
    handle: HANDLE,
    view: Option<MEMORY_MAPPED_VIEW_ADDRESS>,
    mapped_size: usize,
}

unsafe impl Send for Inner {}

impl SharedMemoryReader {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(Inner {
                handle: HANDLE(core::ptr::null_mut()),
                view: None,
                mapped_size: 0,
            }),
            last_sequence: AtomicI64::new(-1),
        }
    }

    /// Read the latest frame from the shared memory. Returns `None` when
    /// the producer has not published any frame yet (mapping doesn't
    /// exist, or magic field still zero).
    pub fn read_latest(&self) -> Option<FrameView> {
        let mut state = self.state.lock().ok()?;
        if !ensure_mapping(&mut state) {
            return None;
        }

        let view = state.view.as_ref()?;
        let header_ptr = view.Value as *const u8;
        let header = unsafe { read_header(header_ptr)? };
        if header.magic != SHMEM_MAGIC || header.width == 0 || header.height == 0 {
            return None;
        }

        let pixel_bytes = (header.width as usize)
            .checked_mul(header.height as usize)?
            .checked_mul(4)?;
        let total_size = SHMEM_HEADER_SIZE + pixel_bytes;
        if total_size > state.mapped_size {
            // The producer enlarged the mapping; remap and retry next call.
            unsafe { unmap(&mut state) };
            return None;
        }

        let mut pixels = vec![0u8; pixel_bytes];
        unsafe {
            core::ptr::copy_nonoverlapping(
                header_ptr.add(SHMEM_HEADER_SIZE),
                pixels.as_mut_ptr(),
                pixel_bytes,
            );
        }

        self.last_sequence
            .store(header.sequence as i64, Ordering::Release);
        Some(FrameView {
            width: header.width,
            height: header.height,
            sequence: header.sequence,
            timestamp_ns: header.timestamp,
            flags: header.flags,
            pixels,
        })
    }
}

impl Drop for SharedMemoryReader {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            unsafe { unmap(&mut state) };
        }
    }
}

#[derive(Clone, Copy)]
struct Header {
    magic: u32,
    width: u32,
    height: u32,
    sequence: u64,
    timestamp: u64,
    flags: u32,
}

unsafe fn read_header(ptr: *const u8) -> Option<Header> {
    let read_u32 = |off: usize| -> u32 {
        let mut buf = [0u8; 4];
        core::ptr::copy_nonoverlapping(ptr.add(off), buf.as_mut_ptr(), 4);
        u32::from_le_bytes(buf)
    };
    let read_u64 = |off: usize| -> u64 {
        let mut buf = [0u8; 8];
        core::ptr::copy_nonoverlapping(ptr.add(off), buf.as_mut_ptr(), 8);
        u64::from_le_bytes(buf)
    };
    Some(Header {
        magic: read_u32(0),
        width: read_u32(4),
        height: read_u32(8),
        sequence: read_u64(12),
        timestamp: read_u64(20),
        flags: read_u32(28),
    })
}

/// Try to open or grow the mapping. Returns `false` if the producer is
/// not running (no mapping by that name exists).
fn ensure_mapping(state: &mut Inner) -> bool {
    if state.view.is_some() && state.mapped_size >= SHMEM_HEADER_SIZE {
        return true;
    }

    unsafe { unmap(state) };

    // Try a generous default-sized view (header + 1920×1080 RGBA). The
    // producer writes whatever real size the renderer is configured for;
    // if it's larger we'll re-map on the next call.
    let initial_size = SHMEM_HEADER_SIZE + 1920 * 1080 * 4;
    let wide_name: Vec<u16> = OsStr::new(SHMEM_NAME)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();

    let handle = unsafe { OpenFileMappingW(FILE_MAP_READ.0, false, windows::core::PCWSTR(wide_name.as_ptr())) };
    let handle = match handle {
        Ok(h) => h,
        Err(_) => return false,
    };
    if handle.is_invalid() {
        return false;
    }

    let view = unsafe { MapViewOfFile(handle, FILE_MAP_READ, 0, 0, initial_size) };
    if view.Value.is_null() {
        unsafe {
            let _ = CloseHandle(handle);
        }
        return false;
    }

    state.handle = handle;
    state.view = Some(view);
    state.mapped_size = initial_size;
    true
}

unsafe fn unmap(state: &mut Inner) {
    if let Some(view) = state.view.take() {
        let _ = UnmapViewOfFile(view);
    }
    if !state.handle.is_invalid() {
        let _ = CloseHandle(state.handle);
        state.handle = HANDLE(core::ptr::null_mut());
    }
    state.mapped_size = 0;
}
