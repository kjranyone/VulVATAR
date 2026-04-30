//! Reader for the file-backed frame buffer that
//! `vulvatar::output::frame_sink::Win32FileBackedSharedMemorySink` writes
//! into.
//!
//! Wire format (little-endian, 32-byte header followed by RGBA pixels):
//! ```text
//! offset 0   u32  magic     = 0x5643_4D46  ("FMCV")
//! offset 4   u32  width
//! offset 8   u32  height
//! offset 12  u64  sequence  (incremented on every frame)
//! offset 20  u64  timestamp (frame timestamp; nanoseconds since some epoch)
//! offset 28  u32  flags     (bit 0 = PRESERVE_ALPHA; NV12 ignores,
//!                            bit 1 = LINEAR_COLOR_SPACE)
//! offset 32  ...  width * height * 4 RGBA bytes
//! ```
//!
//! Both producer and consumer open the file at
//! `%ProgramData%\VulVATAR\camera_frame_buffer.bin` and create unnamed
//! file-backed mappings of it. The kernel shares the underlying physical
//! pages between the two mappings even though they're created by
//! processes in different sessions (interactive user vs LocalService
//! svchost), so cross-session data flow happens at the file system
//! level — not the kernel object namespace. This sidesteps the
//! `SeCreateGlobalPrivilege` gate that blocks `Global\` named sections
//! from non-elevated interactive tokens under UAC filtering.

use core::sync::atomic::{compiler_fence, AtomicI64, Ordering};
use std::fs::File;
use std::os::windows::fs::OpenOptionsExt;
use std::os::windows::io::AsRawHandle;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use windows::core::PCWSTR;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::Memory::{
    CreateFileMappingW, MapViewOfFile, UnmapViewOfFile, FILE_MAP_READ, MEMORY_MAPPED_VIEW_ADDRESS,
    PAGE_READONLY,
};

const SHMEM_HEADER_SIZE: usize = 32;
const SHMEM_MAGIC: u32 = 0x5643_4D46;

/// FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE so we never
/// lock the producer out of writing or rotating the file.
const FILE_SHARE_RWD: u32 = 0x7;

/// Bit 0 of the producer-written `flags` u32 at header offset 28. When
/// set, the RGB32 conversion path keeps the source RGBA's alpha byte
/// instead of forcing 0xFF; NV12 paths ignore this (no alpha channel).
pub const FRAME_FLAG_PRESERVE_ALPHA: u32 = 1 << 0;

/// Bit 1 of the producer-written `flags` u32. When set, the producer
/// rendered to a linear-RGB target (no sRGB encode on store) and clients
/// that respect colour metadata should treat the pixels as
/// `MFVideoTransFunc_10` (linear). When clear, pixels are sRGB-encoded.
/// Used to tag the IMFSample with `MF_MT_TRANSFER_FUNCTION` per frame so
/// downstream clients don't double-encode.
pub const FRAME_FLAG_LINEAR_COLOR_SPACE: u32 = 1 << 1;

/// One frame view returned from the shared memory.
#[derive(Clone)]
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
    /// RGBA pixels. `len() == width * height * 4`. Wrapped in `Arc` so
    /// the cached-last-frame path (see `SharedMemoryReader::read_latest`)
    /// can clone a `FrameView` without copying the pixel buffer — at
    /// 1080p that's an 8 MB memcpy we'd otherwise pay every time the
    /// seq-lock retry hits.
    pub pixels: Arc<Vec<u8>>,
}

/// Maximum number of consecutive `read_fresh` failures we hide by
/// returning the cached frame instead. Picked so a producer that has
/// stopped (VulVATAR closed, render thread hung) surfaces as black
/// within roughly a second at 30 fps, instead of silently freezing the
/// last visible frame forever.
const MAX_CACHE_REPEATS: u32 = 30;

struct CacheState {
    last_good: Option<FrameView>,
    misses_since_good: u32,
}

pub struct SharedMemoryReader {
    state: Mutex<Inner>,
    last_sequence: AtomicI64,
    /// Repeats the previous good frame when `read_fresh` returns `None`,
    /// up to [`MAX_CACHE_REPEATS`] consecutive failures. The MF stream
    /// otherwise inserts a fully-black sample on every miss, which
    /// shows up as visible black flicker whenever the writer commits
    /// during the reader's pixel-copy window — see `media_stream.rs`
    /// `RequestSample` paths.
    cache: Mutex<CacheState>,
}

struct Inner {
    /// Held open for the lifetime of the mapping. Reopened on each
    /// `ensure_mapping` if it was dropped (producer file disappeared).
    file: Option<File>,
    /// CreateFileMappingW handle. Cleared on remap.
    mapping: HANDLE,
    view: Option<MEMORY_MAPPED_VIEW_ADDRESS>,
    mapped_size: usize,
}

unsafe impl Send for Inner {}

impl SharedMemoryReader {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(Inner {
                file: None,
                mapping: HANDLE(core::ptr::null_mut()),
                view: None,
                mapped_size: 0,
            }),
            last_sequence: AtomicI64::new(-1),
            cache: Mutex::new(CacheState {
                last_good: None,
                misses_since_good: 0,
            }),
        }
    }

    /// Read the latest frame, falling back to a cached copy of the
    /// previous good frame when the underlying [`Self::read_fresh`]
    /// returns `None` because of a writer/reader seq-lock race.
    /// Returns `None` only when (a) no frame has ever been read, or
    /// (b) the producer has been silent for more than
    /// [`MAX_CACHE_REPEATS`] consecutive requests.
    pub fn read_latest(&self) -> Option<FrameView> {
        if let Some(view) = self.read_fresh() {
            if let Ok(mut cache) = self.cache.lock() {
                cache.last_good = Some(view.clone());
                cache.misses_since_good = 0;
            }
            return Some(view);
        }

        let mut cache = self.cache.lock().ok()?;
        if cache.misses_since_good >= MAX_CACHE_REPEATS {
            return None;
        }
        cache.misses_since_good += 1;
        cache.last_good.clone()
    }

    /// Read the latest frame from the shared memory. Returns `None` when
    /// the producer has not published any frame yet (file doesn't
    /// exist, or magic field still zero).
    fn read_fresh(&self) -> Option<FrameView> {
        let mut state = self.state.lock().ok()?;
        if !ensure_mapping(&mut state) {
            return None;
        }

        let view = state.view.as_ref()?;
        let header_ptr = view.Value as *const u8;

        let seq1 = unsafe { read_sequence(header_ptr) };
        if seq1 & 1 != 0 {
            return None;
        }

        let header = unsafe { read_header(header_ptr)? };
        if header.magic != SHMEM_MAGIC || header.width == 0 || header.height == 0 {
            return None;
        }

        if header.sequence != seq1 {
            return None;
        }

        let pixel_bytes = (header.width as usize)
            .checked_mul(header.height as usize)?
            .checked_mul(4)?;
        let total_size = SHMEM_HEADER_SIZE + pixel_bytes;
        if total_size > state.mapped_size {
            unsafe { unmap_section(&mut state) };
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

        compiler_fence(Ordering::Acquire);
        let seq2 = unsafe { read_sequence(header_ptr) };
        if seq1 != seq2 {
            return None;
        }

        let frame_sequence = seq1 / 2;

        self.last_sequence
            .store(frame_sequence as i64, Ordering::Release);
        Some(FrameView {
            width: header.width,
            height: header.height,
            sequence: frame_sequence,
            timestamp_ns: header.timestamp,
            flags: header.flags,
            pixels: Arc::new(pixels),
        })
    }
}

impl Drop for SharedMemoryReader {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            unsafe { unmap_all(&mut state) };
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

unsafe fn read_sequence(ptr: *const u8) -> u64 {
    let mut buf = [0u8; 8];
    core::ptr::copy_nonoverlapping(ptr.add(12), buf.as_mut_ptr(), 8);
    u64::from_le_bytes(buf)
}

fn camera_file_path() -> PathBuf {
    let dir = std::env::var_os("ProgramData")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\ProgramData"));
    dir.join("VulVATAR").join("camera_frame_buffer.bin")
}

/// Try to open or grow the mapping. Returns `false` if the producer is
/// not running (file doesn't exist) or if the file is too small to even
/// contain a header.
fn ensure_mapping(state: &mut Inner) -> bool {
    if state.file.is_none() {
        let path = camera_file_path();
        let f = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_RWD)
            .open(&path);
        match f {
            Ok(f) => state.file = Some(f),
            Err(_) => return false,
        }
    }

    // Scope the borrow tightly so we can call &mut state methods below.
    let file_size = match state.file.as_ref().unwrap().metadata() {
        Ok(m) => m.len() as usize,
        Err(_) => {
            // Lost access to the file (deleted underneath us, etc).
            // Drop everything so the next call starts clean.
            unsafe { unmap_all(state) };
            return false;
        }
    };
    if file_size < SHMEM_HEADER_SIZE {
        return false;
    }

    // Fast path: existing mapping still covers the current file size.
    if state.view.is_some() && state.mapped_size >= file_size {
        return true;
    }

    unsafe { unmap_section(state) };

    // Re-borrow the file just to grab its raw handle. The HANDLE we
    // build here doesn't extend the borrow past this statement — it's
    // an opaque pointer-sized value that the kernel resolves independently.
    let file_handle = HANDLE(
        state.file.as_ref().unwrap().as_raw_handle(),
    );
    let mapping = unsafe {
        // dwMaximumSizeHigh=0, dwMaximumSizeLow=0 → kernel uses current
        // file size, which is exactly what we want.
        CreateFileMappingW(file_handle, None, PAGE_READONLY, 0, 0, PCWSTR::null())
    };
    let mapping = match mapping {
        Ok(h) if !h.is_invalid() => h,
        _ => return false,
    };

    let view = unsafe { MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, file_size) };
    if view.Value.is_null() {
        unsafe {
            let _ = CloseHandle(mapping);
        }
        return false;
    }

    state.mapping = mapping;
    state.view = Some(view);
    state.mapped_size = file_size;
    true
}

/// Drop the section + view but keep the file handle open. Called when
/// the file has grown and we need to remap to the new size.
unsafe fn unmap_section(state: &mut Inner) {
    if let Some(view) = state.view.take() {
        let _ = UnmapViewOfFile(view);
    }
    if !state.mapping.is_invalid() {
        let _ = CloseHandle(state.mapping);
        state.mapping = HANDLE(core::ptr::null_mut());
    }
    state.mapped_size = 0;
}

/// Drop everything including the file handle. Called on Drop or when
/// the file becomes unreachable.
unsafe fn unmap_all(state: &mut Inner) {
    unsafe { unmap_section(state) };
    state.file = None;
}
