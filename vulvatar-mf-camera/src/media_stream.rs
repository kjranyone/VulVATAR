//! `IMFMediaStream` for the single video output of our virtual camera.
//!
//! Each MF media source exposes one or more media streams; the source
//! enqueues stream-scoped events through this object and `RequestSample`
//! is the hook the Frame Server uses to pull a frame. For the first cut
//! `RequestSample` returns `E_NOTIMPL`, so the camera registers and
//! enumerates but produces no frames yet.

#![allow(unused_variables)]

use std::sync::{Arc, Mutex};

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use windows::core::{implement, IUnknown, Interface, GUID, HRESULT, PROPVARIANT};
use windows::Win32::Foundation::E_FAIL;
use windows::Win32::Media::MediaFoundation::{
    IMFAsyncCallback, IMFAsyncResult, IMFAttributes, IMFMediaEvent, IMFMediaEventGenerator_Impl,
    IMFMediaEventQueue, IMFMediaSource, IMFMediaStream2, IMFMediaStream2_Impl, IMFMediaStream_Impl,
    IMFSample, IMFStreamDescriptor, IMFVideoSampleAllocatorEx, MEMediaSample, MFCreateMemoryBuffer,
    MFCreateSample, MFGetSystemTime, MFVideoFormat_NV12, MEDIA_EVENT_GENERATOR_GET_EVENT_FLAGS,
    MF_MT_FRAME_RATE, MF_MT_FRAME_SIZE, MF_MT_SUBTYPE, MF_STREAM_STATE, MF_STREAM_STATE_RUNNING,
};

use crate::media_source::DEFAULT_FPS;
use crate::shared_memory::{SharedMemoryReader, FRAME_FLAG_PRESERVE_ALPHA};
use crate::{dll_add_ref, dll_release};

const GUID_NULL_VALUE: GUID = GUID::from_u128(0);

/// Resolution + framerate the client has currently negotiated. If we can't
/// read it from the media-type handler (race during init etc.), fall back
/// to 1920×1080@30 NV12 so something sensible always ships.
#[derive(Clone, Copy, Debug)]
struct CurrentFormat {
    subtype: GUID,
    width: u32,
    height: u32,
    fps: u32,
}

impl CurrentFormat {
    fn fallback() -> Self {
        Self {
            subtype: MFVideoFormat_NV12,
            width: 1920,
            height: 1080,
            fps: DEFAULT_FPS,
        }
    }

    fn frame_duration_hns(&self) -> u64 {
        10_000_000u64 / self.fps.max(1) as u64
    }
}

#[implement(IMFMediaStream2, IMFAttributes)]
pub struct VulvatarMediaStream {
    inner: Mutex<StreamInner>,
    /// Stream-scoped attribute identity. `IMFMediaSourceEx::GetStreamAttributes`
    /// returns this same object through the stream's `IMFAttributes` view.
    attributes: IMFAttributes,
    /// Current MF_STREAM_STATE value. Stored as the discriminant so
    /// `IMFMediaStream2::Get/SetStreamState` can answer without holding
    /// the mutex. Initialised to MF_STREAM_STATE_RUNNING (the stream
    /// starts alive; Pause/Stop transitions are driven by the source).
    stream_state: AtomicI32,
    /// Reader for the named shared-memory frame buffer the host process
    /// fills via `vulvatar::output::frame_sink::Win32NamedSharedMemorySink`.
    shared_memory: SharedMemoryReader,
    /// Cached negotiated format. Resolved from the stream descriptor's
    /// media-type handler the first time `RequestSample` runs and reused
    /// thereafter — the handler call costs a COM clone + PROPVARIANT
    /// reads which would otherwise hit the hot path 30–60× per second.
    /// The cache lives for the stream's lifetime; the source drops the
    /// stream on Stop/Shutdown so a renegotiated format on the next Start
    /// gets a fresh stream with a fresh cache.
    cached_format: Mutex<Option<CurrentFormat>>,
    /// Frame Server-provided sample allocator. Shared with the source so
    /// a stream created during `GetStreamAttributes` still sees a later
    /// `SetDefaultAllocator` call before `Start`.
    allocator: Arc<Mutex<Option<IMFVideoSampleAllocatorEx>>>,
    /// Whether `IMFVideoSampleAllocatorEx::InitializeSampleAllocator` has
    /// been called for the currently-negotiated media type. The pool only
    /// needs to be initialised once unless the format changes mid-stream,
    /// which our SUPPORTED_FORMATS list doesn't currently allow.
    allocator_initialized: AtomicBool,
}

struct StreamInner {
    parent_source: IMFMediaSource,
    descriptor: IMFStreamDescriptor,
    event_queue: IMFMediaEventQueue,
}

impl VulvatarMediaStream {
    pub fn new(
        parent_source: IMFMediaSource,
        descriptor: IMFStreamDescriptor,
        event_queue: IMFMediaEventQueue,
        allocator: Arc<Mutex<Option<IMFVideoSampleAllocatorEx>>>,
    ) -> windows::core::Result<Self> {
        dll_add_ref();
        let attributes: IMFAttributes = descriptor.cast()?;
        Ok(Self {
            inner: Mutex::new(StreamInner {
                parent_source,
                descriptor,
                event_queue,
            }),
            attributes,
            stream_state: AtomicI32::new(MF_STREAM_STATE_RUNNING.0),
            shared_memory: SharedMemoryReader::new(),
            cached_format: Mutex::new(None),
            allocator,
            allocator_initialized: AtomicBool::new(false),
        })
    }
}

impl Drop for VulvatarMediaStream {
    fn drop(&mut self) {
        dll_release();
    }
}

impl IMFMediaEventGenerator_Impl for VulvatarMediaStream_Impl {
    fn GetEvent(
        &self,
        flags: MEDIA_EVENT_GENERATOR_GET_EVENT_FLAGS,
    ) -> windows::core::Result<IMFMediaEvent> {
        let queue = self.inner.lock().unwrap().event_queue.clone();
        unsafe { queue.GetEvent(flags.0 as u32) }
    }

    fn BeginGetEvent(
        &self,
        callback: Option<&IMFAsyncCallback>,
        state: Option<&IUnknown>,
    ) -> windows::core::Result<()> {
        let queue = self.inner.lock().unwrap().event_queue.clone();
        unsafe { queue.BeginGetEvent(callback, state) }
    }

    fn EndGetEvent(&self, result: Option<&IMFAsyncResult>) -> windows::core::Result<IMFMediaEvent> {
        let queue = self.inner.lock().unwrap().event_queue.clone();
        unsafe { queue.EndGetEvent(result) }
    }

    fn QueueEvent(
        &self,
        met: u32,
        guid_extended_type: *const GUID,
        hr_status: HRESULT,
        value: *const PROPVARIANT,
    ) -> windows::core::Result<()> {
        let queue = self.inner.lock().unwrap().event_queue.clone();
        let extended_type = unsafe {
            if guid_extended_type.is_null() {
                &GUID_NULL_VALUE
            } else {
                &*guid_extended_type
            }
        };
        unsafe { queue.QueueEventParamVar(met, extended_type, hr_status, value) }
    }
}

impl IMFMediaStream_Impl for VulvatarMediaStream_Impl {
    fn GetMediaSource(&self) -> windows::core::Result<IMFMediaSource> {
        crate::t!("Stream::GetMediaSource");
        Ok(self.inner.lock().unwrap().parent_source.clone())
    }

    fn GetStreamDescriptor(&self) -> windows::core::Result<IMFStreamDescriptor> {
        crate::t!("Stream::GetStreamDescriptor");
        Ok(self.inner.lock().unwrap().descriptor.clone())
    }

    fn RequestSample(&self, token: Option<&IUnknown>) -> windows::core::Result<()> {
        crate::t!("Stream::RequestSample");

        let format = self.current_format();
        let is_nv12 = format.subtype == MFVideoFormat_NV12;
        // Live source: sample timestamps follow MF's system clock (the
        // same clock SourceReader / capture engine uses to schedule
        // downstream consumers). Negative values aren't possible in
        // practice but `MFGetSystemTime` returns LONGLONG, so guard
        // before passing to `SetSampleTime`.
        let time: i64 = unsafe { MFGetSystemTime() }.max(0);
        let dur: i64 = format.frame_duration_hns() as i64;

        // Two paths:
        //   - allocator path (Frame Server-mediated client): snapshot the
        //     IMFVideoSampleAllocatorEx Frame Server gave us via
        //     SetDefaultAllocator, lazy-initialise the pool with the
        //     current media type, then `AllocateSample()` and fill its
        //     buffer in place. Frame Server requires this for cross-process
        //     sample marshaling; CPU MFCreateMemoryBuffer-backed samples
        //     can't reach the FrameServerPool and ReadSample returns
        //     MF_SOURCE_READERF_ENDOFSTREAM otherwise.
        //   - direct CoCreateInstance path (no allocator was set): build
        //     the sample with MFCreateMemoryBuffer + MFCreateSample.
        let allocator_opt = self.allocator.lock().unwrap().clone();

        let sample: IMFSample = if let Some(allocator) = allocator_opt {
            if !self.allocator_initialized.load(Ordering::Acquire) {
                let descriptor = self.inner.lock().unwrap().descriptor.clone();
                let mt = unsafe { descriptor.GetMediaTypeHandler()?.GetCurrentMediaType()? };
                crate::t!("Stream::RequestSample InitializeSampleAllocator(10, current_type)");
                unsafe { allocator.InitializeSampleAllocator(10, &mt)? };
                self.allocator_initialized.store(true, Ordering::Release);
            }
            let s = unsafe { allocator.AllocateSample()? };
            self.fill_allocated_sample(&s, format, is_nv12)?;
            unsafe {
                s.SetSampleTime(time)?;
                s.SetSampleDuration(dur)?;
            }
            s
        } else {
            // Direct path — keep the existing MFCreateMemoryBuffer flow.
            match self.shared_memory.read_latest() {
                Some(view) => {
                    let preserve_alpha = view.flags & FRAME_FLAG_PRESERVE_ALPHA != 0;
                    let resampled;
                    let (rgba, w, h) = if view.width == format.width && view.height == format.height
                    {
                        (view.pixels.as_slice(), view.width, view.height)
                    } else {
                        crate::t!(
                            "Stream::RequestSample resample {}x{} → {}x{}",
                            view.width,
                            view.height,
                            format.width,
                            format.height,
                        );
                        resampled = resample_rgba_nearest(
                            &view.pixels,
                            view.width as usize,
                            view.height as usize,
                            format.width as usize,
                            format.height as usize,
                        );
                        (resampled.as_slice(), format.width, format.height)
                    };
                    crate::t!(
                        "Stream::RequestSample direct frame seq={} {}x{}@{} subtype={} alpha={}",
                        view.sequence,
                        format.width,
                        format.height,
                        format.fps,
                        if is_nv12 { "NV12" } else { "RGB32" },
                        preserve_alpha,
                    );
                    if is_nv12 {
                        build_sample_nv12_from_rgba(rgba, w, h, time, dur)?
                    } else {
                        build_sample_rgb32_from_rgba(rgba, w, h, preserve_alpha, time, dur)?
                    }
                }
                None => {
                    crate::t!(
                        "Stream::RequestSample direct no producer ({}x{} {})",
                        format.width,
                        format.height,
                        if is_nv12 { "NV12" } else { "RGB32" },
                    );
                    if is_nv12 {
                        build_black_sample_nv12(format.width, format.height, time, dur)?
                    } else {
                        build_black_sample_rgb32(format.width, format.height, time, dur)?
                    }
                }
            }
        };

        if let Some(tok) = token {
            unsafe { sample.SetUnknown(&MFSampleExtension_Token, tok)? };
        }

        let queue = self.inner.lock().unwrap().event_queue.clone();
        let sample_unk: IUnknown = sample.cast()?;
        unsafe {
            queue.QueueEventParamUnk(
                MEMediaSample.0 as u32,
                &GUID_NULL_VALUE,
                windows::core::HRESULT(0),
                &sample_unk,
            )?;
        }
        Ok(())
    }
}

impl VulvatarMediaStream {
    /// Fill an already-allocated `IMFSample` (from
    /// `IMFVideoSampleAllocatorEx::AllocateSample`) with the latest pixel
    /// data from shared memory. Keeps the allocator's pre-allocated
    /// `IMFMediaBuffer` rather than swapping in a fresh `MFCreateMemoryBuffer`,
    /// which is what Frame Server's marshaller requires.
    fn fill_allocated_sample(
        &self,
        sample: &IMFSample,
        format: CurrentFormat,
        is_nv12: bool,
    ) -> windows::core::Result<()> {
        let buffer = unsafe { sample.GetBufferByIndex(0)? };
        let view = self.shared_memory.read_latest();
        let preserve_alpha = view
            .as_ref()
            .map(|v| v.flags & FRAME_FLAG_PRESERVE_ALPHA != 0)
            .unwrap_or(false);

        unsafe {
            let mut dst_ptr: *mut u8 = core::ptr::null_mut();
            let mut max_len: u32 = 0;
            let mut current_len: u32 = 0;
            buffer.Lock(&mut dst_ptr, Some(&mut max_len), Some(&mut current_len))?;
            if dst_ptr.is_null() {
                buffer.Unlock()?;
                return Err(E_FAIL.into());
            }

            let written = match view {
                Some(view) => {
                    let resampled;
                    let (rgba, w, h) = if view.width == format.width && view.height == format.height
                    {
                        (view.pixels.as_slice(), view.width, view.height)
                    } else {
                        resampled = resample_rgba_nearest(
                            &view.pixels,
                            view.width as usize,
                            view.height as usize,
                            format.width as usize,
                            format.height as usize,
                        );
                        (resampled.as_slice(), format.width, format.height)
                    };
                    crate::t!(
                        "Stream::RequestSample alloc frame seq={} {}x{}@{} subtype={} alpha={}",
                        view.sequence,
                        format.width,
                        format.height,
                        format.fps,
                        if is_nv12 { "NV12" } else { "RGB32" },
                        preserve_alpha,
                    );
                    if is_nv12 {
                        let need = nv12_size(w, h) as u32;
                        if max_len < need {
                            buffer.Unlock()?;
                            return Err(E_FAIL.into());
                        }
                        rgba_to_nv12(rgba, w as usize, h as usize, dst_ptr);
                        need
                    } else {
                        let need = (w as u32) * (h as u32) * 4;
                        if max_len < need {
                            buffer.Unlock()?;
                            return Err(E_FAIL.into());
                        }
                        let src = rgba.as_ptr();
                        for i in 0..(need as usize / 4) {
                            let s = src.add(i * 4);
                            let d = dst_ptr.add(i * 4);
                            *d = *s.add(2);
                            *d.add(1) = *s.add(1);
                            *d.add(2) = *s;
                            *d.add(3) = if preserve_alpha { *s.add(3) } else { 0xFF };
                        }
                        need
                    }
                }
                None => {
                    crate::t!(
                        "Stream::RequestSample alloc no producer ({}x{} {})",
                        format.width,
                        format.height,
                        if is_nv12 { "NV12" } else { "RGB32" },
                    );
                    if is_nv12 {
                        let need = nv12_size(format.width, format.height) as u32;
                        if max_len >= need {
                            // Limited-range black: Y=16, UV=128
                            let y_len = (format.width * format.height) as usize;
                            core::ptr::write_bytes(dst_ptr, 16, y_len);
                            core::ptr::write_bytes(dst_ptr.add(y_len), 128, y_len / 2);
                        }
                        need
                    } else {
                        let need = (format.width as u32) * (format.height as u32) * 4;
                        if max_len >= need {
                            for i in 0..(need as usize / 4) {
                                let d = dst_ptr.add(i * 4);
                                *d = 0;
                                *d.add(1) = 0;
                                *d.add(2) = 0;
                                *d.add(3) = 0xFF;
                            }
                        }
                        need
                    }
                }
            };

            buffer.SetCurrentLength(written)?;
            buffer.Unlock()?;
        }
        Ok(())
    }
}

impl VulvatarMediaStream {
    /// Resolve the negotiated subtype + frame size + frame rate. Cached
    /// after the first call so the hot path avoids the descriptor lock,
    /// `GetMediaTypeHandler` COM clone and three `GetUINT*` reads per
    /// frame. The cache is dropped from `clear_format_cache` whenever
    /// the source transitions through Stop/Shutdown — a renegotiated
    /// type is then picked up on the next Start.
    fn current_format(&self) -> CurrentFormat {
        if let Some(f) = *self.cached_format.lock().unwrap() {
            return f;
        }
        let format = self.read_format_from_descriptor();
        *self.cached_format.lock().unwrap() = Some(format);
        format
    }

    fn read_format_from_descriptor(&self) -> CurrentFormat {
        let descriptor = match self.inner.lock() {
            Ok(g) => g.descriptor.clone(),
            Err(_) => return CurrentFormat::fallback(),
        };
        unsafe {
            let Ok(handler) = descriptor.GetMediaTypeHandler() else {
                return CurrentFormat::fallback();
            };
            let Ok(mt) = handler.GetCurrentMediaType() else {
                return CurrentFormat::fallback();
            };
            let subtype = mt.GetGUID(&MF_MT_SUBTYPE).unwrap_or(MFVideoFormat_NV12);
            let size = mt.GetUINT64(&MF_MT_FRAME_SIZE).unwrap_or(0);
            let rate = mt.GetUINT64(&MF_MT_FRAME_RATE).unwrap_or(0);
            let width = (size >> 32) as u32;
            let height = (size & 0xFFFF_FFFF) as u32;
            // Frame rate is packed (numerator, denominator). For
            // integer framerates the denominator is 1; we ignore the
            // ratio and use the numerator directly.
            let fps = (rate >> 32) as u32;
            if width == 0 || height == 0 || fps == 0 {
                return CurrentFormat::fallback();
            }
            CurrentFormat {
                subtype,
                width,
                height,
                fps,
            }
        }
    }
}

// IMFAttributes forwarder — see `attributes.rs` for the macro definition
// and `media_source.rs` for the matching impl on the source side.
crate::forward_imfattributes!(VulvatarMediaStream_Impl);

/// Convert RGBA bytes (R first, length = w * h * 4) into an `IMFSample`
/// containing RGB32 (BGRA layout, B first) pixel data. When `preserve_alpha`
/// is true the source alpha byte is copied to the output's 4th byte;
/// otherwise it's forced to 0xFF (opaque, treating the format as BGRX).
/// The negotiated client size is used; caller is responsible for resampling
/// the renderer's source frame to that size before calling.
fn build_sample_rgb32_from_rgba(
    rgba: &[u8],
    width: u32,
    height: u32,
    preserve_alpha: bool,
    time_hns: i64,
    duration_hns: i64,
) -> windows::core::Result<IMFSample> {
    let buffer_size = (width as usize) * (height as usize) * 4;
    let buffer = unsafe { MFCreateMemoryBuffer(buffer_size as u32)? };
    unsafe {
        let mut dst_ptr: *mut u8 = core::ptr::null_mut();
        let mut max_len: u32 = 0;
        let mut current_len: u32 = 0;
        buffer.Lock(&mut dst_ptr, Some(&mut max_len), Some(&mut current_len))?;
        if dst_ptr.is_null() || (max_len as usize) < buffer_size {
            buffer.Unlock()?;
            return Err(E_FAIL.into());
        }

        let src = rgba.as_ptr();
        for i in 0..(buffer_size / 4) {
            let s = src.add(i * 4);
            let d = dst_ptr.add(i * 4);
            *d = *s.add(2); // B ← R
            *d.add(1) = *s.add(1); // G
            *d.add(2) = *s; // R ← B
            *d.add(3) = if preserve_alpha { *s.add(3) } else { 0xFF };
        }
        buffer.SetCurrentLength(buffer_size as u32)?;
        buffer.Unlock()?;
    }

    let sample = unsafe { MFCreateSample()? };
    unsafe {
        sample.AddBuffer(&buffer)?;
        sample.SetSampleTime(time_hns)?;
        sample.SetSampleDuration(duration_hns)?;
    }
    Ok(sample)
}

/// Build a fully-opaque black RGB32 sample of `width × height`.
fn build_black_sample_rgb32(
    width: u32,
    height: u32,
    time_hns: i64,
    duration_hns: i64,
) -> windows::core::Result<IMFSample> {
    let buffer_size = (width as usize) * (height as usize) * 4;
    let buffer = unsafe { MFCreateMemoryBuffer(buffer_size as u32)? };
    unsafe {
        let mut dst_ptr: *mut u8 = core::ptr::null_mut();
        let mut max_len: u32 = 0;
        let mut current_len: u32 = 0;
        buffer.Lock(&mut dst_ptr, Some(&mut max_len), Some(&mut current_len))?;
        if !dst_ptr.is_null() && (max_len as usize) >= buffer_size {
            for i in 0..(buffer_size / 4) {
                let d = dst_ptr.add(i * 4);
                *d = 0;
                *d.add(1) = 0;
                *d.add(2) = 0;
                *d.add(3) = 0xFF;
            }
        }
        buffer.SetCurrentLength(buffer_size as u32)?;
        buffer.Unlock()?;
    }
    let sample = unsafe { MFCreateSample()? };
    unsafe {
        sample.AddBuffer(&buffer)?;
        sample.SetSampleTime(time_hns)?;
        sample.SetSampleDuration(duration_hns)?;
    }
    Ok(sample)
}

/// Nearest-neighbour RGBA resampler. Cheap, GPU-free, fine for a virtual
/// camera path where the renderer is the upstream of truth and clients
/// only need plausible sub-frames.
fn resample_rgba_nearest(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    if src_w == dst_w && src_h == dst_h {
        return src.to_vec();
    }
    let mut out = vec![0u8; dst_w * dst_h * 4];
    let src_stride = src_w * 4;
    let dst_stride = dst_w * 4;
    for y in 0..dst_h {
        let sy = y * src_h / dst_h;
        for x in 0..dst_w {
            let sx = x * src_w / dst_w;
            let s_off = sy * src_stride + sx * 4;
            let d_off = y * dst_stride + x * 4;
            out[d_off..d_off + 4].copy_from_slice(&src[s_off..s_off + 4]);
        }
    }
    out
}

/// `MFSampleExtension_Token` (well-known constant) — when MF clients call
/// `RequestSample` with a token, they expect to find that same `IUnknown`
/// stashed on the returned sample under this attribute key.
#[allow(non_upper_case_globals)]
const MFSampleExtension_Token: GUID = GUID::from_u128(0x8294da66_f328_4805_b551_00deb4c57a61);

// ---------------------------------------------------------------------------
// NV12 path
// ---------------------------------------------------------------------------
//
// NV12 layout (per Microsoft's MFVideoFormat_NV12 spec):
//   plane 0  width × height          bytes of Y (luma)
//   plane 1  width × (height / 2)    bytes of UV interleaved (each pair
//                                    covers a 2x2 luma block, half-res in
//                                    both axes)
//
// Total payload = width × height × 3 / 2.
//
// Conversion uses the BT.601 Y'CbCr matrix in 8-bit limited range, which
// is what every "consumer webcam" pipeline assumes by default. Caller
// supplies RGBA bytes (R first); we down-sample chroma by averaging each
// 2×2 RGB block.

const fn nv12_size(w: u32, h: u32) -> usize {
    (w as usize) * (h as usize) * 3 / 2
}

fn build_sample_nv12_from_rgba(
    rgba: &[u8],
    width: u32,
    height: u32,
    time_hns: i64,
    duration_hns: i64,
) -> windows::core::Result<IMFSample> {
    let w = width as usize;
    let h = height as usize;
    let buffer_size = nv12_size(width, height);

    let buffer = unsafe { MFCreateMemoryBuffer(buffer_size as u32)? };
    unsafe {
        let mut dst_ptr: *mut u8 = core::ptr::null_mut();
        let mut max_len: u32 = 0;
        let mut current_len: u32 = 0;
        buffer.Lock(&mut dst_ptr, Some(&mut max_len), Some(&mut current_len))?;
        if dst_ptr.is_null() || (max_len as usize) < buffer_size {
            buffer.Unlock()?;
            return Err(E_FAIL.into());
        }
        rgba_to_nv12(rgba, w, h, dst_ptr);
        buffer.SetCurrentLength(buffer_size as u32)?;
        buffer.Unlock()?;
    }

    let sample = unsafe { MFCreateSample()? };
    unsafe {
        sample.AddBuffer(&buffer)?;
        sample.SetSampleTime(time_hns)?;
        sample.SetSampleDuration(duration_hns)?;
    }
    Ok(sample)
}

fn build_black_sample_nv12(
    width: u32,
    height: u32,
    time_hns: i64,
    duration_hns: i64,
) -> windows::core::Result<IMFSample> {
    let w = width as usize;
    let h = height as usize;
    let buffer_size = nv12_size(width, height);

    let buffer = unsafe { MFCreateMemoryBuffer(buffer_size as u32)? };
    unsafe {
        let mut dst_ptr: *mut u8 = core::ptr::null_mut();
        let mut max_len: u32 = 0;
        let mut current_len: u32 = 0;
        buffer.Lock(&mut dst_ptr, Some(&mut max_len), Some(&mut current_len))?;
        if !dst_ptr.is_null() && (max_len as usize) >= buffer_size {
            // Limited-range black: Y = 16, Cb = Cr = 128.
            core::ptr::write_bytes(dst_ptr, 16, w * h);
            core::ptr::write_bytes(dst_ptr.add(w * h), 128, w * h / 2);
        }
        buffer.SetCurrentLength(buffer_size as u32)?;
        buffer.Unlock()?;
    }
    let sample = unsafe { MFCreateSample()? };
    unsafe {
        sample.AddBuffer(&buffer)?;
        sample.SetSampleTime(time_hns)?;
        sample.SetSampleDuration(duration_hns)?;
    }
    Ok(sample)
}

/// BT.601 limited-range Y'CbCr conversion. Writes into the caller's
/// `dst` (must be at least `nv12_size(w, h)` bytes).
unsafe fn rgba_to_nv12(rgba: &[u8], w: usize, h: usize, dst: *mut u8) {
    let y_plane = dst;
    let uv_plane = dst.add(w * h);
    let stride = w * 4;

    // Walk in 2×2 blocks: every block contributes 4 Y samples and 1 UV pair.
    let y_idx = |x: usize, y: usize| y * w + x;

    let mut row = 0;
    while row < h {
        let mut col = 0;
        while col < w {
            // Read up to 4 source pixels (clamping at the right/bottom edge).
            let take = |dx: usize, dy: usize| -> (u8, u8, u8) {
                let xx = (col + dx).min(w - 1);
                let yy = (row + dy).min(h - 1);
                let off = yy * stride + xx * 4;
                (rgba[off], rgba[off + 1], rgba[off + 2])
            };
            let p00 = take(0, 0);
            let p10 = take(1, 0);
            let p01 = take(0, 1);
            let p11 = take(1, 1);

            for (dx, dy, p) in [(0, 0, p00), (1, 0, p10), (0, 1, p01), (1, 1, p11)] {
                let xx = col + dx;
                let yy = row + dy;
                if xx < w && yy < h {
                    let y = bt601_y(p.0, p.1, p.2);
                    *y_plane.add(y_idx(xx, yy)) = y;
                }
            }

            // Average the four source pixels for chroma.
            let r = (p00.0 as u16 + p10.0 as u16 + p01.0 as u16 + p11.0 as u16) / 4;
            let g = (p00.1 as u16 + p10.1 as u16 + p01.1 as u16 + p11.1 as u16) / 4;
            let b = (p00.2 as u16 + p10.2 as u16 + p01.2 as u16 + p11.2 as u16) / 4;
            let (cb, cr) = bt601_uv(r as u8, g as u8, b as u8);
            let uv_off = (row / 2) * w + col;
            *uv_plane.add(uv_off) = cb;
            *uv_plane.add(uv_off + 1) = cr;

            col += 2;
        }
        row += 2;
    }
}

/// BT.601 limited-range Y component (16..=235).
#[inline]
fn bt601_y(r: u8, g: u8, b: u8) -> u8 {
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;
    let y = (66 * r + 129 * g + 25 * b + 128) >> 8;
    (y + 16).clamp(0, 255) as u8
}

/// BT.601 limited-range Cb / Cr (16..=240).
#[inline]
fn bt601_uv(r: u8, g: u8, b: u8) -> (u8, u8) {
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;
    let cb = (-38 * r - 74 * g + 112 * b + 128) >> 8;
    let cr = (112 * r - 94 * g - 18 * b + 128) >> 8;
    (
        (cb + 128).clamp(0, 255) as u8,
        (cr + 128).clamp(0, 255) as u8,
    )
}

/// MF's Frame Server QI's the stream for `IMFMediaStream2` during
/// IMFVirtualCamera::Start; the plain IMFMediaStream vtable alone
/// triggers E_NOINTERFACE. Implementing the upgrade is cheap — just
/// remember/report the MF_STREAM_STATE atomically. The source drives
/// state transitions, but MF may still call Get/SetStreamState directly
/// on the stream.
impl IMFMediaStream2_Impl for VulvatarMediaStream_Impl {
    fn SetStreamState(&self, value: MF_STREAM_STATE) -> windows::core::Result<()> {
        self.stream_state.store(value.0, Ordering::SeqCst);
        Ok(())
    }

    fn GetStreamState(&self) -> windows::core::Result<MF_STREAM_STATE> {
        Ok(MF_STREAM_STATE(self.stream_state.load(Ordering::SeqCst)))
    }
}
