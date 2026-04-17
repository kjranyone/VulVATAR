//! `IMFMediaSource` for the VulVATAR virtual camera.
//!
//! Owns the event queue, presentation descriptor and the single video
//! [`crate::media_stream::VulvatarMediaStream`]. Frame production proper
//! is task #10; this layer is responsible for the lifecycle dance
//! (Start → MENewStream + MESourceStarted + MEStreamStarted, etc.) that
//! `MFCreateVirtualCamera` and downstream Frame Server clients require.

#![allow(unused_variables)]

use std::sync::{Mutex, OnceLock};

use windows::core::{implement, IUnknown, Interface, GUID, HRESULT, PROPVARIANT};
use windows::Win32::Foundation::{E_INVALIDARG, E_NOTIMPL, E_POINTER};
use windows::Win32::Media::KernelStreaming::{
    IKsControl, IKsControl_Impl, KSIDENTIFIER, PINNAME_VIDEO_CAPTURE,
};
use windows::Win32::Media::MediaFoundation::{
    IMFAsyncCallback, IMFAsyncResult, IMFAttributes, IMFGetService, IMFGetService_Impl,
    IMFMediaEvent, IMFMediaEventGenerator_Impl, IMFMediaEventQueue, IMFMediaSource,
    IMFMediaSourceEx, IMFMediaSourceEx_Impl, IMFMediaSource_Impl, IMFMediaStream, IMFMediaType,
    IMFPresentationDescriptor, IMFSampleAllocatorControl, IMFSampleAllocatorControl_Impl,
    IMFStreamDescriptor, MFCreateAttributes, MFCreateEventQueue, MFCreateMediaType,
    MFCreatePresentationDescriptor, MFCreateStreamDescriptor, MFFrameSourceTypes_Color,
    MFMediaType_Video, MFSampleAllocatorUsage, MFVideoFormat_RGB32,
    MEDIA_EVENT_GENERATOR_GET_EVENT_FLAGS, MENewStream, MESourceStarted, MESourceStopped,
    MEStreamStarted, MEStreamStopped, MFMEDIASOURCE_CHARACTERISTICS, MFMEDIASOURCE_IS_LIVE,
    MF_DEVICESTREAM_ATTRIBUTE_FRAMESOURCE_TYPES, MF_DEVICESTREAM_FRAMESERVER_SHARED,
    MF_DEVICESTREAM_STREAM_CATEGORY, MF_DEVICESTREAM_STREAM_ID, MF_E_SHUTDOWN,
    MF_E_UNSUPPORTED_SERVICE, MF_MT_ALL_SAMPLES_INDEPENDENT, MF_MT_FRAME_RATE, MF_MT_FRAME_SIZE,
    MF_MT_INTERLACE_MODE, MF_MT_MAJOR_TYPE, MF_MT_PIXEL_ASPECT_RATIO, MF_MT_SUBTYPE,
    MFVideoInterlace_Progressive,
};

use crate::media_stream::VulvatarMediaStream;
use crate::{dll_add_ref, dll_release};

/// Default video output format. Matches the renderer's CPU readback.
pub const DEFAULT_WIDTH: u32 = 1920;
pub const DEFAULT_HEIGHT: u32 = 1080;
pub const DEFAULT_FPS: u32 = 30;
pub const DEFAULT_STREAM_ID: u32 = 0;

/// `HRESULT_FROM_WIN32(ERROR_SET_NOT_FOUND)` = 0x80070492. The value MS
/// SimpleMediaSource returns from unknown KS property / method / event
/// probes — MF distinguishes "this property set is not implemented" from
/// "this interface is unusable", and mapping to `E_NOTIMPL` causes the
/// latter at the IMFVirtualCamera::Start boundary.
const KS_PROPERTY_NOT_FOUND: HRESULT = HRESULT(0x80070492u32 as i32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourceState {
    Stopped,
    Started,
    Paused,
    Shutdown,
}

#[implement(
    IMFMediaSourceEx,
    IMFMediaSource,
    IKsControl,
    IMFGetService,
    IMFSampleAllocatorControl
)]
pub struct VulvatarMediaSource {
    inner: OnceLock<Inner>,
    mutable: Mutex<MutableState>,
}

struct Inner {
    event_queue: IMFMediaEventQueue,
    presentation_descriptor: IMFPresentationDescriptor,
    stream_descriptor: IMFStreamDescriptor,
    source_attributes: IMFAttributes,
    stream_attributes: IMFAttributes,
}

struct MutableState {
    state: SourceState,
    stream: Option<IMFMediaStream>,
}

impl VulvatarMediaSource {
    pub fn new() -> Self {
        dll_add_ref();
        Self {
            inner: OnceLock::new(),
            mutable: Mutex::new(MutableState {
                state: SourceState::Stopped,
                stream: None,
            }),
        }
    }

    /// Lazy-initialise the MF objects. Called the first time anyone asks
    /// for the presentation descriptor or starts the source. We can't do
    /// this in `new()` because building the IMFMediaType/etc. requires a
    /// running MediaFoundation; the platform is expected to have called
    /// `MFStartup` already by the time MFCreateVirtualCamera reaches us.
    fn initialised(&self) -> windows::core::Result<&Inner> {
        if let Some(inner) = self.inner.get() {
            return Ok(inner);
        }

        let media_type = build_default_media_type()?;
        let stream_descriptor =
            unsafe { MFCreateStreamDescriptor(DEFAULT_STREAM_ID, &[Some(media_type.clone())])? };
        let handler = unsafe { stream_descriptor.GetMediaTypeHandler()? };
        unsafe { handler.SetCurrentMediaType(&media_type)? };

        // Frame Server inspects these attributes to decide whether a
        // source is a Frame-Server-shareable camera. Omitting them (in
        // particular MF_DEVICESTREAM_FRAMESERVER_SHARED) causes
        // IMFVirtualCamera::Start to refuse with E_NOINTERFACE — not
        // because any interface is missing, but because Frame Server
        // won't dispatch the stream to the virtual-camera surface.
        let sd_attrs: IMFAttributes = stream_descriptor.cast()?;
        apply_stream_attributes(&sd_attrs)?;

        let presentation_descriptor = unsafe {
            MFCreatePresentationDescriptor(Some(&[Some(stream_descriptor.clone())]))?
        };
        // Mark our single stream as selected by default so the Frame Server
        // does not need to call SelectStream itself.
        unsafe { presentation_descriptor.SelectStream(0)? };

        let event_queue = unsafe { MFCreateEventQueue()? };

        let source_attributes = unsafe {
            let mut a: Option<IMFAttributes> = None;
            MFCreateAttributes(&mut a, 1)?;
            a.unwrap()
        };
        let stream_attributes = unsafe {
            let mut a: Option<IMFAttributes> = None;
            MFCreateAttributes(&mut a, 4)?;
            let attrs = a.unwrap();
            apply_stream_attributes(&attrs)?;
            attrs
        };

        let _ = self.inner.set(Inner {
            event_queue,
            presentation_descriptor,
            stream_descriptor,
            source_attributes,
            stream_attributes,
        });
        Ok(self.inner.get().unwrap())
    }
}

impl Drop for VulvatarMediaSource {
    fn drop(&mut self) {
        dll_release();
    }
}

impl IMFMediaEventGenerator_Impl for VulvatarMediaSource_Impl {
    fn GetEvent(
        &self,
        flags: MEDIA_EVENT_GENERATOR_GET_EVENT_FLAGS,
    ) -> windows::core::Result<IMFMediaEvent> {
        crate::t!("Source::GetEvent flags={:#x}", flags.0);
        let inner = self.initialised()?;
        unsafe { inner.event_queue.GetEvent(flags.0 as u32) }
    }

    fn BeginGetEvent(
        &self,
        callback: Option<&IMFAsyncCallback>,
        state: Option<&IUnknown>,
    ) -> windows::core::Result<()> {
        crate::t!("Source::BeginGetEvent");
        let inner = self.initialised()?;
        unsafe { inner.event_queue.BeginGetEvent(callback, state) }
    }

    fn EndGetEvent(
        &self,
        result: Option<&IMFAsyncResult>,
    ) -> windows::core::Result<IMFMediaEvent> {
        crate::t!("Source::EndGetEvent");
        let inner = self.initialised()?;
        unsafe { inner.event_queue.EndGetEvent(result) }
    }

    fn QueueEvent(
        &self,
        met: u32,
        guid_extended_type: *const GUID,
        hr_status: HRESULT,
        value: *const PROPVARIANT,
    ) -> windows::core::Result<()> {
        crate::t!("Source::QueueEvent met={}", met);
        let inner = self.initialised()?;
        unsafe { inner.event_queue.QueueEventParamVar(met, guid_extended_type, hr_status, value) }
    }
}

impl IMFMediaSource_Impl for VulvatarMediaSource_Impl {
    fn GetCharacteristics(&self) -> windows::core::Result<u32> {
        crate::t!("Source::GetCharacteristics");
        // Live software source, no seeking, no rate control.
        Ok(MFMEDIASOURCE_IS_LIVE.0 as u32)
    }

    fn CreatePresentationDescriptor(
        &self,
    ) -> windows::core::Result<IMFPresentationDescriptor> {
        crate::t!("Source::CreatePresentationDescriptor");
        let inner = self.initialised()?;
        unsafe { inner.presentation_descriptor.Clone() }
    }

    fn Start(
        &self,
        pd: Option<&IMFPresentationDescriptor>,
        _time_format: *const GUID,
        _start_position: *const PROPVARIANT,
    ) -> windows::core::Result<()> {
        crate::t!("Source::Start pd={}", pd.is_some());
        let inner = self.initialised()?;
        let pd = pd.ok_or_else(|| windows::core::Error::from(E_INVALIDARG))?;

        // Build (or rebuild) the media stream now that Start has been called.
        let stream: IMFMediaStream = {
            let mut state = self.mutable.lock().unwrap();
            if state.state == SourceState::Shutdown {
                return Err(MF_E_SHUTDOWN.into());
            }
            if let Some(ref s) = state.stream {
                s.clone()
            } else {
                // We need an IMFMediaSource handle for the stream to hold;
                // that's just the COM identity exposed by `_Impl`.
                let parent: IMFMediaSource = unsafe { self.cast()? };
                let event_queue = inner.event_queue.clone();
                let descriptor = inner.stream_descriptor.clone();
                let s: IMFMediaStream =
                    VulvatarMediaStream::new(parent, descriptor, event_queue).into();
                state.stream = Some(s.clone());
                s
            }
        };

        // Lifecycle events expected by MF: source must announce the new
        // stream before transitioning into Started, then the stream itself
        // must signal MEStreamStarted.
        let stream_unk: IUnknown = stream.cast()?;
        let stream_var = PROPVARIANT::from(stream_unk);
        unsafe {
            inner.event_queue.QueueEventParamVar(
                MENewStream.0 as u32,
                core::ptr::null(),
                windows::core::HRESULT(0),
                &stream_var,
            )?;
            inner.event_queue.QueueEventParamVar(
                MESourceStarted.0 as u32,
                core::ptr::null(),
                windows::core::HRESULT(0),
                core::ptr::null(),
            )?;
            stream.QueueEvent(
                MEStreamStarted.0 as u32,
                core::ptr::null(),
                windows::core::HRESULT(0),
                core::ptr::null(),
            )?;
        }

        let mut state = self.mutable.lock().unwrap();
        state.state = SourceState::Started;
        Ok(())
    }

    fn Stop(&self) -> windows::core::Result<()> {
        let inner = self.initialised()?;
        let stream = self.mutable.lock().unwrap().stream.clone();
        unsafe {
            if let Some(stream) = stream {
                stream.QueueEvent(
                    MEStreamStopped.0 as u32,
                    core::ptr::null(),
                    windows::core::HRESULT(0),
                    core::ptr::null(),
                )?;
            }
            inner.event_queue.QueueEventParamVar(
                MESourceStopped.0 as u32,
                core::ptr::null(),
                windows::core::HRESULT(0),
                core::ptr::null(),
            )?;
        }
        self.mutable.lock().unwrap().state = SourceState::Stopped;
        Ok(())
    }

    fn Pause(&self) -> windows::core::Result<()> {
        self.mutable.lock().unwrap().state = SourceState::Paused;
        Ok(())
    }

    fn Shutdown(&self) -> windows::core::Result<()> {
        if let Some(inner) = self.inner.get() {
            unsafe {
                let _ = inner.event_queue.Shutdown();
            }
        }
        self.mutable.lock().unwrap().state = SourceState::Shutdown;
        self.mutable.lock().unwrap().stream = None;
        Ok(())
    }
}

impl IMFMediaSourceEx_Impl for VulvatarMediaSource_Impl {
    fn GetSourceAttributes(&self) -> windows::core::Result<IMFAttributes> {
        crate::t!("SourceEx::GetSourceAttributes");
        let inner = self.initialised()?;
        Ok(inner.source_attributes.clone())
    }

    fn GetStreamAttributes(&self, stream_id: u32) -> windows::core::Result<IMFAttributes> {
        crate::t!("SourceEx::GetStreamAttributes id={}", stream_id);
        let inner = self.initialised()?;
        Ok(inner.stream_attributes.clone())
    }

    fn SetD3DManager(&self, _manager: Option<&IUnknown>) -> windows::core::Result<()> {
        crate::t!("SourceEx::SetD3DManager");
        Ok(())
    }
}

/// MFCreateVirtualCamera / Frame Server QI the source for `IKsControl`
/// at Start time, following the DirectShow-era property-set plumbing that
/// every camera pipeline eventually falls back on. We don't expose any
/// kernel-streaming properties, methods or events (nothing to configure
/// for a software virtual camera), but the interface has to be present
/// or Start returns `E_NOINTERFACE`.
impl IKsControl_Impl for VulvatarMediaSource_Impl {
    fn KsProperty(
        &self,
        _property: *const KSIDENTIFIER,
        _property_length: u32,
        _property_data: *mut core::ffi::c_void,
        _data_length: u32,
        _bytes_returned: *mut u32,
    ) -> windows::core::Result<()> {
        crate::t!("Source::KsProperty");
        // MS SimpleMediaSource pattern: unknown KS property/method/event
        // sets are reported as ERROR_SET_NOT_FOUND, not E_NOTIMPL. Frame
        // Server maps E_NOTIMPL from the KS probe back to E_NOINTERFACE
        // at the MFCreateVirtualCamera boundary.
        Err(KS_PROPERTY_NOT_FOUND.into())
    }

    fn KsMethod(
        &self,
        _method: *const KSIDENTIFIER,
        _method_length: u32,
        _method_data: *mut core::ffi::c_void,
        _data_length: u32,
        _bytes_returned: *mut u32,
    ) -> windows::core::Result<()> {
        crate::t!("Source::KsMethod");
        Err(KS_PROPERTY_NOT_FOUND.into())
    }

    fn KsEvent(
        &self,
        _event: *const KSIDENTIFIER,
        _event_length: u32,
        _event_data: *mut core::ffi::c_void,
        _data_length: u32,
        _bytes_returned: *mut u32,
    ) -> windows::core::Result<()> {
        crate::t!("Source::KsEvent");
        Err(KS_PROPERTY_NOT_FOUND.into())
    }
}

/// MF's Frame Server pipes several interop facilities (clock, rate control,
/// quality advise, DXGI device manager, …) through `IMFGetService`. A
/// software virtual camera provides none of them, but the interface itself
/// has to respond — clients query it before deciding whether to use a
/// service and cope fine with `E_NOINTERFACE`, whereas a missing
/// `IMFGetService` vtable kills `IMFVirtualCamera::Start`.
impl IMFGetService_Impl for VulvatarMediaSource_Impl {
    fn GetService(
        &self,
        guid_service: *const GUID,
        riid: *const GUID,
        ppv_object: *mut *mut core::ffi::c_void,
    ) -> windows::core::Result<()> {
        unsafe {
            crate::t!(
                "Source::GetService guid={:?} riid={:?}",
                if guid_service.is_null() { GUID::zeroed() } else { *guid_service },
                if riid.is_null() { GUID::zeroed() } else { *riid },
            );
            if ppv_object.is_null() {
                return Err(E_POINTER.into());
            }
            *ppv_object = core::ptr::null_mut();
        }
        // Match MS SimpleMediaSource: callers that probe a service we
        // don't offer expect MF_E_UNSUPPORTED_SERVICE, not E_NOINTERFACE.
        // E_NOINTERFACE would suggest IMFGetService itself is missing,
        // which would be a QI-level failure rather than a per-service
        // rejection — some Frame Server code paths treat it as fatal.
        Err(MF_E_UNSUPPORTED_SERVICE.into())
    }
}

/// Also on the "Frame Server QI probe" list — mirrored from the MS
/// SimpleMediaSource sample. Neither the default allocator nor the usage
/// query needs bespoke behaviour for a CPU-only software source; letting
/// MF plug its own allocator and reporting "uses default" covers it.
impl IMFSampleAllocatorControl_Impl for VulvatarMediaSource_Impl {
    fn SetDefaultAllocator(
        &self,
        _output_stream_id: u32,
        _allocator: Option<&IUnknown>,
    ) -> windows::core::Result<()> {
        crate::t!("Source::SetDefaultAllocator");
        Ok(())
    }

    fn GetAllocatorUsage(
        &self,
        _output_stream_id: u32,
        input_stream_id: *mut u32,
        usage: *mut MFSampleAllocatorUsage,
    ) -> windows::core::Result<()> {
        crate::t!("Source::GetAllocatorUsage");
        unsafe {
            if !input_stream_id.is_null() {
                *input_stream_id = 0;
            }
            if !usage.is_null() {
                // MFSampleAllocatorUsage_UsesProvidedAllocator = 0 → let MF
                // manage buffers for us.
                *usage = MFSampleAllocatorUsage(0);
            }
        }
        Ok(())
    }
}

/// Build the default RGB32 1920x1080 @ 30fps media type.
///
/// `MFSetAttributeSize` / `MFSetAttributeRatio` are inline helpers in the
/// SDK header and are not exposed by `windows-rs`; we pack the two `u32`
/// halves into a `u64` ourselves (high → numerator, low → denominator).
fn build_default_media_type() -> windows::core::Result<IMFMediaType> {
    unsafe {
        let mt = MFCreateMediaType()?;
        mt.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
        mt.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_RGB32)?;
        mt.SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)?;
        mt.SetUINT32(&MF_MT_ALL_SAMPLES_INDEPENDENT, 1)?;
        mt.SetUINT64(&MF_MT_FRAME_SIZE, pack_2x32(DEFAULT_WIDTH, DEFAULT_HEIGHT))?;
        mt.SetUINT64(&MF_MT_FRAME_RATE, pack_2x32(DEFAULT_FPS, 1))?;
        mt.SetUINT64(&MF_MT_PIXEL_ASPECT_RATIO, pack_2x32(1, 1))?;
        Ok(mt)
    }
}

#[inline]
fn pack_2x32(hi: u32, lo: u32) -> u64 {
    ((hi as u64) << 32) | lo as u64
}

/// Apply the four stream-scoped attributes the Frame Server inspects to
/// decide whether a stream is a colour-video Frame Server camera. Mirrors
/// MS's SimpleMediaStream::_SetStreamDescriptorAttributes.
fn apply_stream_attributes(attrs: &IMFAttributes) -> windows::core::Result<()> {
    unsafe {
        attrs.SetGUID(&MF_DEVICESTREAM_STREAM_CATEGORY, &PINNAME_VIDEO_CAPTURE)?;
        attrs.SetUINT32(&MF_DEVICESTREAM_STREAM_ID, DEFAULT_STREAM_ID)?;
        attrs.SetUINT32(&MF_DEVICESTREAM_FRAMESERVER_SHARED, 1)?;
        attrs.SetUINT32(
            &MF_DEVICESTREAM_ATTRIBUTE_FRAMESOURCE_TYPES,
            MFFrameSourceTypes_Color.0 as u32,
        )?;
    }
    Ok(())
}

// Compile-time guard: keep MFMEDIASOURCE_CHARACTERISTICS imported even when
// only its bitflags constants are referenced via the .0 conversion.
#[allow(dead_code)]
const _CHARACTERISTICS_TYPE: MFMEDIASOURCE_CHARACTERISTICS = MFMEDIASOURCE_IS_LIVE;
