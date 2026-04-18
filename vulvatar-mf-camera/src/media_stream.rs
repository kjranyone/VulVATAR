//! `IMFMediaStream` for the single video output of our virtual camera.
//!
//! Each MF media source exposes one or more media streams; the source
//! enqueues stream-scoped events through this object and `RequestSample`
//! is the hook the Frame Server uses to pull a frame. For the first cut
//! `RequestSample` returns `E_NOTIMPL`, so the camera registers and
//! enumerates but produces no frames yet.

#![allow(unused_variables)]

use std::sync::Mutex;

use windows::core::{implement, IUnknown, Interface, GUID, HRESULT, PROPVARIANT};
use windows::Win32::Foundation::E_FAIL;
use windows::Win32::Media::MediaFoundation::{
    IMFAsyncCallback, IMFAsyncResult, IMFMediaEvent, IMFMediaEventGenerator_Impl,
    IMFMediaEventQueue, IMFMediaSource, IMFMediaStream, IMFMediaStream2, IMFMediaStream2_Impl,
    IMFMediaStream_Impl, IMFSample, IMFStreamDescriptor, MFCreateMemoryBuffer, MFCreateSample,
    MEDIA_EVENT_GENERATOR_GET_EVENT_FLAGS, MEMediaSample, MF_STREAM_STATE, MF_STREAM_STATE_RUNNING,
};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

use crate::media_source::{DEFAULT_FPS, DEFAULT_HEIGHT, DEFAULT_WIDTH};
use crate::shared_memory::{FrameView, SharedMemoryReader};
use crate::{dll_add_ref, dll_release};

#[implement(IMFMediaStream2, IMFMediaStream)]
pub struct VulvatarMediaStream {
    inner: Mutex<StreamInner>,
    /// Current MF_STREAM_STATE value. Stored as the discriminant so
    /// `IMFMediaStream2::Get/SetStreamState` can answer without holding
    /// the mutex. Initialised to MF_STREAM_STATE_RUNNING (the stream
    /// starts alive; Pause/Stop transitions are driven by the source).
    stream_state: AtomicI32,
    /// Reader for the named shared-memory frame buffer the host process
    /// fills via `vulvatar::output::frame_sink::Win32NamedSharedMemorySink`.
    shared_memory: SharedMemoryReader,
    /// Monotonic frame counter used for sample timestamps (in 100-ns
    /// units, MF's standard time base). Increments by `frame_duration_hns`
    /// per published sample.
    sample_time_hns: AtomicU64,
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
    ) -> Self {
        dll_add_ref();
        Self {
            inner: Mutex::new(StreamInner {
                parent_source,
                descriptor,
                event_queue,
            }),
            stream_state: AtomicI32::new(MF_STREAM_STATE_RUNNING.0),
            shared_memory: SharedMemoryReader::new(),
            sample_time_hns: AtomicU64::new(0),
        }
    }

    /// Frame interval in 100-nanosecond units. Used for both sample
    /// timestamps and durations.
    fn frame_duration_hns() -> u64 {
        10_000_000u64 / DEFAULT_FPS as u64
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

    fn EndGetEvent(
        &self,
        result: Option<&IMFAsyncResult>,
    ) -> windows::core::Result<IMFMediaEvent> {
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
        unsafe { queue.QueueEventParamVar(met, guid_extended_type, hr_status, value) }
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

        // Build the next sample (either from the latest published frame
        // or a black fallback) and queue it as MEMediaSample on the
        // stream's event queue. MF clients pull the sample by listening
        // for that event, not by waiting on this call.
        let sample = match self.shared_memory.read_latest() {
            Some(view) if view.width == DEFAULT_WIDTH && view.height == DEFAULT_HEIGHT => {
                crate::t!(
                    "Stream::RequestSample frame seq={} {}x{}",
                    view.sequence,
                    view.width,
                    view.height
                );
                build_sample_from_rgba(&view, self.next_timestamp(), VulvatarMediaStream::frame_duration_hns())?
            }
            other => {
                if let Some(view) = other.as_ref() {
                    crate::t!(
                        "Stream::RequestSample size mismatch (sm={}x{}, declared={}x{}); sending black",
                        view.width,
                        view.height,
                        DEFAULT_WIDTH,
                        DEFAULT_HEIGHT,
                    );
                } else {
                    crate::t!("Stream::RequestSample no producer yet, sending black");
                }
                build_black_sample(self.next_timestamp(), VulvatarMediaStream::frame_duration_hns())?
            }
        };

        if let Some(tok) = token {
            unsafe { sample.SetUnknown(&MFSampleExtension_Token, tok)? };
        }

        let queue = self.inner.lock().unwrap().event_queue.clone();
        let sample_unk: IUnknown = sample.cast()?;
        let value = PROPVARIANT::from(sample_unk);
        unsafe {
            queue.QueueEventParamVar(
                MEMediaSample.0 as u32,
                core::ptr::null(),
                windows::core::HRESULT(0),
                &value,
            )?;
        }
        Ok(())
    }
}

impl VulvatarMediaStream {
    fn next_timestamp(&self) -> u64 {
        self.sample_time_hns
            .fetch_add(VulvatarMediaStream::frame_duration_hns(), Ordering::Relaxed)
    }
}

/// Convert the renderer's RGBA output (one byte per channel, R first) into
/// an RGB32 (BGRX, B first, alpha-as-padding) `IMFSample`. Most MF clients
/// — including OBS, Teams and Windows Camera — consume RGB32 directly.
fn build_sample_from_rgba(
    view: &FrameView,
    time_hns: u64,
    duration_hns: u64,
) -> windows::core::Result<IMFSample> {
    let pixel_count = view.pixels.len() / 4;
    let buffer_size = pixel_count * 4;
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

        // RGBA → BGRX byte-swap. RGB32 stores pixels as B G R X (little-
        // endian D3DFMT_X8R8G8B8); the renderer writes R G B A.
        let src = view.pixels.as_ptr();
        for i in 0..pixel_count {
            let s = src.add(i * 4);
            let d = dst_ptr.add(i * 4);
            *d = *s.add(2); // B ← R
            *d.add(1) = *s.add(1); // G
            *d.add(2) = *s; // R ← B
            *d.add(3) = 0xFF; // X (padding) — opaque
        }
        buffer.SetCurrentLength(buffer_size as u32)?;
        buffer.Unlock()?;
    }

    let sample = unsafe { MFCreateSample()? };
    unsafe {
        sample.AddBuffer(&buffer)?;
        sample.SetSampleTime(time_hns as i64)?;
        sample.SetSampleDuration(duration_hns as i64)?;
    }
    Ok(sample)
}

/// Build a fully-opaque black frame of `DEFAULT_WIDTH × DEFAULT_HEIGHT`.
/// Used until the renderer has written its first frame.
fn build_black_sample(
    time_hns: u64,
    duration_hns: u64,
) -> windows::core::Result<IMFSample> {
    let buffer_size = DEFAULT_WIDTH as usize * DEFAULT_HEIGHT as usize * 4;
    let buffer = unsafe { MFCreateMemoryBuffer(buffer_size as u32)? };
    unsafe {
        let mut dst_ptr: *mut u8 = core::ptr::null_mut();
        let mut max_len: u32 = 0;
        let mut current_len: u32 = 0;
        buffer.Lock(&mut dst_ptr, Some(&mut max_len), Some(&mut current_len))?;
        if !dst_ptr.is_null() && (max_len as usize) >= buffer_size {
            // Black opaque: B=0, G=0, R=0, X=0xFF.
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
        sample.SetSampleTime(time_hns as i64)?;
        sample.SetSampleDuration(duration_hns as i64)?;
    }
    Ok(sample)
}

/// `MFSampleExtension_Token` (well-known constant) — when MF clients call
/// `RequestSample` with a token, they expect to find that same `IUnknown`
/// stashed on the returned sample under this attribute key.
#[allow(non_upper_case_globals)]
const MFSampleExtension_Token: GUID =
    GUID::from_u128(0x8294da66_f328_4805_b551_00deb4c57a61);

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
