//! `IMFMediaSource` for the VulVATAR virtual camera.
//!
//! Owns the event queue, presentation descriptor and the single video
//! [`crate::media_stream::VulvatarMediaStream`]. Frame production proper
//! is task #10; this layer is responsible for the lifecycle dance
//! (Start → MENewStream + MESourceStarted + MEStreamStarted, etc.) that
//! `MFCreateVirtualCamera` and downstream Frame Server clients require.

#![allow(unused_variables)]

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex, OnceLock,
};

use windows::core::{implement, IUnknown, Interface, GUID, HRESULT, PROPVARIANT};
use windows::Win32::Foundation::{E_INVALIDARG, E_POINTER};
use windows::Win32::Media::KernelStreaming::{
    IKsControl, IKsControl_Impl, KSCAMERAPROFILE_Legacy, KSIDENTIFIER, PINNAME_VIDEO_CAPTURE,
    PROPSETID_VIDCAP_CAMERACONTROL,
};
use windows::Win32::Media::MediaFoundation::{
    IMFActivate, IMFActivate_Impl, IMFAsyncCallback, IMFAsyncResult, IMFAttributes,
    IMFExtendedCameraControl, IMFExtendedCameraControl_Impl, IMFExtendedCameraController,
    IMFExtendedCameraController_Impl, IMFGetService, IMFGetService_Impl, IMFMediaEvent,
    IMFMediaEventGenerator_Impl, IMFMediaEventQueue, IMFMediaSource, IMFMediaSourceEx,
    IMFMediaSourceEx_Impl, IMFMediaSource_Impl, IMFMediaStream, IMFMediaStream2, IMFMediaType,
    IMFPresentationDescriptor, IMFRealTimeClient, IMFRealTimeClientEx, IMFRealTimeClientEx_Impl,
    IMFRealTimeClient_Impl, IMFSampleAllocatorControl, IMFSampleAllocatorControl_Impl,
    IMFStreamDescriptor, IMFVideoSampleAllocatorEx, MENewStream, MESourceStarted, MESourceStopped,
    MEStreamStarted, MEStreamStopped, MFCreateAttributes, MFCreateEventQueue, MFCreateMediaType,
    MFCreatePresentationDescriptor, MFCreateSensorProfile, MFCreateSensorProfileCollection,
    MFCreateStreamDescriptor, MFFrameSourceTypes_Color, MFMediaType_Video, MFSampleAllocatorUsage,
    MFT_TRANSFORM_CLSID_Attribute, MFVideoFormat_NV12, MFVideoFormat_RGB32,
    MFVideoInterlace_Progressive, MEDIA_EVENT_GENERATOR_GET_EVENT_FLAGS,
    MFMEDIASOURCE_CHARACTERISTICS, MFMEDIASOURCE_IS_LIVE, MF_CAPTURE_ENGINE_MEDIASOURCE,
    MF_DEVICEMFT_SENSORPROFILE_COLLECTION, MF_DEVICESTREAM_ATTRIBUTE_FRAMESOURCE_TYPES,
    MF_DEVICESTREAM_FRAMESERVER_SHARED, MF_DEVICESTREAM_STREAM_CATEGORY, MF_DEVICESTREAM_STREAM_ID,
    MF_E_INVALIDSTREAMNUMBER, MF_E_INVALID_STATE_TRANSITION, MF_E_SHUTDOWN,
    MFVideoPrimaries_BT709, MFVideoTransFunc_sRGB, MF_E_UNSUPPORTED_SERVICE,
    MF_MT_ALL_SAMPLES_INDEPENDENT, MF_MT_AVG_BITRATE, MF_MT_DEFAULT_STRIDE,
    MF_MT_FIXED_SIZE_SAMPLES, MF_MT_FRAME_RATE, MF_MT_FRAME_SIZE, MF_MT_INTERLACE_MODE,
    MF_MT_MAJOR_TYPE, MF_MT_PIXEL_ASPECT_RATIO, MF_MT_SAMPLE_SIZE, MF_MT_SUBTYPE,
    MF_MT_TRANSFER_FUNCTION, MF_MT_VIDEO_PRIMARIES,
};

use crate::media_stream::VulvatarMediaStream;
use crate::{dll_add_ref, dll_release, CLSID_VULVATAR_MEDIA_SOURCE};

/// Fallback framerate used when the negotiated media type can't be read
/// (e.g. during a transient race in `current_format`). The full set of
/// (subtype, width, height, fps) combinations the camera advertises lives
/// in [`SUPPORTED_FORMATS`].
pub const DEFAULT_FPS: u32 = 30;
pub const DEFAULT_STREAM_ID: u32 = 0;

/// `HRESULT_FROM_WIN32(ERROR_SET_NOT_FOUND)` = 0x80070492. The value MS
/// SimpleMediaSource returns from unknown KS property / method / event
/// probes — MF distinguishes "this property set is not implemented" from
/// "this interface is unusable", and mapping to `E_NOTIMPL` causes the
/// latter at the IMFVirtualCamera::Start boundary.
const KS_PROPERTY_NOT_FOUND: HRESULT = HRESULT(0x80070492u32 as i32);
const GUID_NULL_VALUE: GUID = GUID::from_u128(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourceState {
    Stopped,
    Started,
    Shutdown,
}

#[implement(
    IMFActivate,
    IMFMediaSourceEx,
    IMFAttributes,
    IKsControl,
    IMFGetService,
    IMFSampleAllocatorControl,
    IMFExtendedCameraController,
    IMFRealTimeClient,
    IMFRealTimeClientEx
)]
pub struct VulvatarMediaSource {
    /// Attribute store. `VulvatarMediaSource` implements `IMFAttributes`
    /// directly so `QueryInterface<IMFAttributes>(source)` returns self —
    /// smourier/VCamSample's pattern. Frame Server's FsProxy does that
    /// QI early in its setup; receiving `E_NOINTERFACE` causes it to
    /// abandon the bridge synchronously (microsecond FsProxy teardown
    /// with `ReadSample` → `MF_SOURCE_READERF_ENDOFSTREAM`). The same
    /// store backs `IMFMediaSourceEx::GetSourceAttributes` so both
    /// paths read/write the same values.
    attributes: IMFAttributes,
    inner: OnceLock<Inner>,
    mutable: Mutex<MutableState>,
    /// Frame Server-provided per-stream sample allocator. Set via
    /// `IMFSampleAllocatorControl::SetDefaultAllocator` BEFORE
    /// `IMFMediaSource::Start` is invoked. Snapshot into the
    /// `VulvatarMediaStream` at construction time so its
    /// `RequestSample` can `AllocateSample` from the Frame Server pool
    /// rather than `MFCreateMemoryBuffer` — only allocator-pool
    /// samples are cross-process marshalable.
    stream_allocator: Arc<Mutex<Option<IMFVideoSampleAllocatorEx>>>,
}

struct Inner {
    event_queue: IMFMediaEventQueue,
    presentation_descriptor: IMFPresentationDescriptor,
    stream_descriptor: IMFStreamDescriptor,
}

struct MutableState {
    state: SourceState,
    stream: Option<IMFMediaStream>,
}

impl VulvatarMediaSource {
    pub fn new() -> windows::core::Result<Self> {
        dll_add_ref();
        // Eagerly create the attribute store with the three attributes Frame
        // Server's FsProxy looks at during catalog matching / bridge setup.
        // Sensor profiles come later in `initialised()` (that one may need
        // MFStartup to have fully completed on the caller's side).
        let mut a: Option<IMFAttributes> = None;
        unsafe { MFCreateAttributes(&mut a, 6)? };
        let attributes = a.expect("MFCreateAttributes returned no interface");
        unsafe {
            attributes.SetGUID(&MFT_TRANSFORM_CLSID_Attribute, &CLSID_VULVATAR_MEDIA_SOURCE)?;
        }
        Ok(Self {
            attributes,
            inner: OnceLock::new(),
            mutable: Mutex::new(MutableState {
                state: SourceState::Stopped,
                stream: None,
            }),
            stream_allocator: Arc::new(Mutex::new(None)),
        })
    }

    fn get_or_create_stream(&self, inner: &Inner) -> windows::core::Result<IMFMediaStream> {
        let mut state = self.mutable.lock().unwrap();
        if state.state == SourceState::Shutdown {
            return Err(MF_E_SHUTDOWN.into());
        }
        if let Some(ref s) = state.stream {
            return Ok(s.clone());
        }

        // We need an IMFMediaSource handle for the stream to hold; that's
        // just the COM identity exposed by `_Impl`.
        let parent: IMFMediaSource = unsafe { self.cast()? };
        let event_queue = unsafe { MFCreateEventQueue()? };
        let descriptor = inner.stream_descriptor.clone();
        let allocator = self.stream_allocator.clone();
        let s2: IMFMediaStream2 =
            VulvatarMediaStream::new(parent, descriptor, event_queue, allocator)?.into();
        let s: IMFMediaStream = s2.cast()?;
        state.stream = Some(s.clone());
        Ok(s)
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

        // Offer the full SUPPORTED_FORMATS list so clients can pick
        // whichever resolution / framerate / pixel layout suits them.
        // RequestSample resamples the renderer's native frame into the
        // selected size on demand.
        let media_types: Vec<IMFMediaType> = SUPPORTED_FORMATS
            .iter()
            .map(|(subtype, w, h, fps)| build_media_type(subtype, *w, *h, *fps))
            .collect::<windows::core::Result<_>>()?;
        let descriptor_input: Vec<Option<IMFMediaType>> =
            media_types.iter().cloned().map(Some).collect();
        let stream_descriptor =
            unsafe { MFCreateStreamDescriptor(DEFAULT_STREAM_ID, &descriptor_input)? };
        let handler = unsafe { stream_descriptor.GetMediaTypeHandler()? };
        // Default to the first entry (1920×1080@30 NV12) so clients that
        // never call SetCurrentMediaType still get a sensible format.
        unsafe { handler.SetCurrentMediaType(&media_types[0])? };

        // Frame Server inspects these attributes to decide whether a
        // source is a Frame-Server-shareable camera. Omitting them (in
        // particular MF_DEVICESTREAM_FRAMESERVER_SHARED) causes
        // IMFVirtualCamera::Start to refuse with E_NOINTERFACE — not
        // because any interface is missing, but because Frame Server
        // won't dispatch the stream to the virtual-camera surface.
        let sd_attrs: IMFAttributes = stream_descriptor.cast()?;
        apply_stream_attributes(&sd_attrs)?;

        let presentation_descriptor =
            unsafe { MFCreatePresentationDescriptor(Some(&[Some(stream_descriptor.clone())]))? };
        // Eagerly select stream 0 in the canonical presentation descriptor
        // so the SourceReader (which gets a clone from
        // `CreatePresentationDescriptor`) can immediately enumerate at
        // least one selected stream. Without this, ReadSample fails with
        // MF_E_INVALIDSTREAMNUMBER. smourier's MediaSource gets away
        // without this because its `Start` reconciles selection from the
        // *client-supplied* PD; ours doesn't reach into a passed-in PD,
        // so it must publish a pre-selected default.
        unsafe { presentation_descriptor.SelectStream(0)? };

        let event_queue = unsafe { MFCreateEventQueue()? };

        // Sensor profile collection requires MFStartup; set it onto the
        // eagerly-created self.attributes now. Paired with the three
        // attrs set in `new()` this gives Frame Server everything it
        // looks at during catalog bridge matching.
        attach_sensor_profiles(&self.attributes)?;

        let _ = self.inner.set(Inner {
            event_queue,
            presentation_descriptor,
            stream_descriptor,
        });
        Ok(self.inner.get().unwrap())
    }
}

impl Drop for VulvatarMediaSource {
    fn drop(&mut self) {
        dll_release();
    }
}

impl IMFActivate_Impl for VulvatarMediaSource_Impl {
    fn ActivateObject(
        &self,
        riid: *const GUID,
        ppv: *mut *mut core::ffi::c_void,
    ) -> windows::core::Result<()> {
        unsafe {
            crate::t!(
                "SourceActivate::ActivateObject riid={:?}",
                if riid.is_null() {
                    GUID::zeroed()
                } else {
                    *riid
                }
            );
            if ppv.is_null() {
                return Err(E_POINTER.into());
            }
            *ppv = core::ptr::null_mut();

            let unk: IUnknown = self.cast()?;
            let hr = unk.query(riid, ppv);
            crate::t!("SourceActivate::ActivateObject: self.query -> {:?}", hr);
            if hr.is_err() {
                return Err(hr.into());
            }
            Ok(())
        }
    }

    fn ShutdownObject(&self) -> windows::core::Result<()> {
        crate::t!("SourceActivate::ShutdownObject");
        let source: IMFMediaSource = unsafe { self.cast()? };
        unsafe { source.Shutdown() }
    }

    fn DetachObject(&self) -> windows::core::Result<()> {
        crate::t!("SourceActivate::DetachObject");
        Ok(())
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

    fn EndGetEvent(&self, result: Option<&IMFAsyncResult>) -> windows::core::Result<IMFMediaEvent> {
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
        let extended_type = unsafe {
            if guid_extended_type.is_null() {
                &GUID_NULL_VALUE
            } else {
                &*guid_extended_type
            }
        };
        unsafe {
            inner
                .event_queue
                .QueueEventParamVar(met, extended_type, hr_status, value)
        }
    }
}

impl IMFMediaSource_Impl for VulvatarMediaSource_Impl {
    fn GetCharacteristics(&self) -> windows::core::Result<u32> {
        crate::t!("Source::GetCharacteristics");
        // Live software source, no seeking, no rate control.
        Ok(MFMEDIASOURCE_IS_LIVE.0 as u32)
    }

    fn CreatePresentationDescriptor(&self) -> windows::core::Result<IMFPresentationDescriptor> {
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
        let stream = self.get_or_create_stream(inner)?;

        // Lifecycle events expected by MF: source must announce the new
        // stream before transitioning into Started, then the stream itself
        // must signal MEStreamStarted.
        let stream_unk: IUnknown = stream.cast()?;
        unsafe {
            crate::t!("Source::Start queue MENewStream");
            inner.event_queue.QueueEventParamUnk(
                MENewStream.0 as u32,
                &GUID_NULL_VALUE,
                windows::core::HRESULT(0),
                &stream_unk,
            )?;
            crate::t!("Source::Start queue MESourceStarted");
            inner.event_queue.QueueEventParamVar(
                MESourceStarted.0 as u32,
                &GUID_NULL_VALUE,
                windows::core::HRESULT(0),
                core::ptr::null(),
            )?;
            crate::t!("Source::Start queue MEStreamStarted");
            stream.QueueEvent(
                MEStreamStarted.0 as u32,
                &GUID_NULL_VALUE,
                windows::core::HRESULT(0),
                core::ptr::null(),
            )?;
        }

        let mut state = self.mutable.lock().unwrap();
        state.state = SourceState::Started;
        Ok(())
    }

    fn Stop(&self) -> windows::core::Result<()> {
        crate::t!("Source::Stop");
        // Don't trigger lazy `initialised()` from Stop — if Start was never
        // called there is no event queue to drain, and constructing one
        // here just to immediately tear it down is wasteful.
        let Some(inner) = self.inner.get() else {
            self.mutable.lock().unwrap().state = SourceState::Stopped;
            return Ok(());
        };
        // Take the stream out of mutable state so the next Start builds
        // a fresh stream object — fresh format cache, fresh allocator-
        // initialised flag, fresh event queue. MF's source contract says
        // every Start sends `MENewStream`, so a new stream identity is
        // both legal and what clients expect after Stop.
        let stream = self.mutable.lock().unwrap().stream.take();
        unsafe {
            if let Some(stream) = stream {
                stream.QueueEvent(
                    MEStreamStopped.0 as u32,
                    &GUID_NULL_VALUE,
                    windows::core::HRESULT(0),
                    core::ptr::null(),
                )?;
            }
            inner.event_queue.QueueEventParamVar(
                MESourceStopped.0 as u32,
                &GUID_NULL_VALUE,
                windows::core::HRESULT(0),
                core::ptr::null(),
            )?;
        }
        self.mutable.lock().unwrap().state = SourceState::Stopped;
        Ok(())
    }

    fn Pause(&self) -> windows::core::Result<()> {
        // Live virtual cameras cannot meaningfully pause — the renderer
        // keeps producing frames whether or not Frame Server consumes them.
        // The MS Frame Server doc explicitly requires `MF_E_INVALID_STATE_
        // TRANSITION` for sources whose characteristics omit
        // `MFMEDIASOURCE_CAN_PAUSE` (which ours does).
        crate::t!("Source::Pause -> MF_E_INVALID_STATE_TRANSITION");
        Err(MF_E_INVALID_STATE_TRANSITION.into())
    }

    fn Shutdown(&self) -> windows::core::Result<()> {
        crate::t!("Source::Shutdown");
        if let Some(inner) = self.inner.get() {
            unsafe {
                let _ = inner.event_queue.Shutdown();
            }
        }
        // Single critical section: setting Shutdown state and dropping the
        // stream reference must be atomic, otherwise another thread can
        // observe Shutdown with a still-live stream and act on it.
        let mut state = self.mutable.lock().unwrap();
        state.state = SourceState::Shutdown;
        state.stream = None;
        Ok(())
    }
}

impl IMFMediaSourceEx_Impl for VulvatarMediaSource_Impl {
    fn GetSourceAttributes(&self) -> windows::core::Result<IMFAttributes> {
        crate::t!("SourceEx::GetSourceAttributes");
        // Return the same store that `IMFAttributes_Impl` forwards to —
        // Frame Server may read via either path and must see the same
        // values. Not tied to `initialised()` so early FsProxy probes
        // (which can happen before the first MFMediaType is built) see
        // a valid attribute store with the catalog-match attributes.
        Ok(self.attributes.clone())
    }

    fn GetStreamAttributes(&self, stream_id: u32) -> windows::core::Result<IMFAttributes> {
        crate::t!("SourceEx::GetStreamAttributes id={}", stream_id);
        let inner = self.initialised()?;
        if stream_id != DEFAULT_STREAM_ID {
            return Err(MF_E_INVALIDSTREAMNUMBER.into());
        }
        let stream = self.get_or_create_stream(inner)?;
        stream.cast()
    }

    fn SetD3DManager(&self, _manager: Option<&IUnknown>) -> windows::core::Result<()> {
        // Software-only source: every sample is built from a CPU buffer
        // (either MFCreateMemoryBuffer in the direct path or the
        // FrameServer-provided allocator), so a DXGI device manager has
        // nowhere to plug in. We accept the call and discard the manager —
        // returning E_NOTIMPL here makes some clients abandon the source.
        crate::t!("SourceEx::SetD3DManager (ignored, software-only)");
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
        property: *const KSIDENTIFIER,
        property_length: u32,
        property_data: *mut core::ffi::c_void,
        data_length: u32,
        bytes_returned: *mut u32,
    ) -> windows::core::Result<()> {
        log_ksid("KsProperty", property, property_length);

        if let Some((set, id, flags)) = parse_ksid(property, property_length) {
            // Windows 11's Frame Server makes registering a virtual camera
            // conditional on a successful GET of the privacy-shutter state
            // (PROPSETID_VIDCAP_CAMERACONTROL / KSPROPERTY_CAMERACONTROL_
            // PRIVACY). Older versions tolerated ERROR_SET_NOT_FOUND here,
            // newer Frame Server versions treat the refusal as
            // E_NOINTERFACE out of IMFVirtualCamera::Start. Respond as
            // "privacy shutter permanently off" so the registration path
            // succeeds; a software source has no physical shutter anyway.
            if set == PROPSETID_VIDCAP_CAMERACONTROL && id == KSPROPERTY_CAMERACONTROL_PRIVACY_ID {
                crate::t!(
                    "Source::KsProperty handling PRIVACY flags={:#x} data_ptr={:?} data_len={}",
                    flags,
                    property_data,
                    data_length,
                );
                const KSPROPERTY_TYPE_GET: u32 = 0x1;
                const KSPROPERTY_TYPE_SET: u32 = 0x2;
                if flags & KSPROPERTY_TYPE_GET != 0 {
                    unsafe {
                        write_extended_prop_privacy_off(property_data, data_length);
                        if !bytes_returned.is_null() {
                            *bytes_returned = data_length.min(EXTENDED_PROP_SIZE);
                        }
                    }
                    crate::t!("Source::KsProperty PRIVACY GET -> S_OK (EXTENDEDPROP)");
                    return Ok(());
                }
                if flags & KSPROPERTY_TYPE_SET != 0 {
                    crate::t!("Source::KsProperty PRIVACY SET -> S_OK (no-op)");
                    return Ok(());
                }
            }
        }

        Err(KS_PROPERTY_NOT_FOUND.into())
    }

    fn KsMethod(
        &self,
        method: *const KSIDENTIFIER,
        method_length: u32,
        _method_data: *mut core::ffi::c_void,
        _data_length: u32,
        _bytes_returned: *mut u32,
    ) -> windows::core::Result<()> {
        log_ksid("KsMethod", method, method_length);
        Err(KS_PROPERTY_NOT_FOUND.into())
    }

    fn KsEvent(
        &self,
        event: *const KSIDENTIFIER,
        event_length: u32,
        event_data: *mut core::ffi::c_void,
        data_length: u32,
        bytes_returned: *mut u32,
    ) -> windows::core::Result<()> {
        log_ksid("KsEvent", event, event_length);
        crate::t!(
            "Source::KsEvent data_ptr={:?} data_len={}",
            event_data,
            data_length
        );

        if let Some((set, id, flags)) = parse_ksid(event, event_length) {
            if set == PROPSETID_VIDCAP_CAMERACONTROL && id == KSPROPERTY_CAMERACONTROL_PRIVACY_ID {
                const KSEVENT_TYPE_ENABLE: u32 = 0x1;
                const KSEVENT_TYPE_BASICSUPPORT: u32 = 0x200;
                if flags & KSEVENT_TYPE_BASICSUPPORT != 0 {
                    // BASICSUPPORT reply: 32-byte KSCAMERA_EXTENDEDPROP_
                    // HEADER describing the event's capability set. All-
                    // zero (the previous response) left `Size=0`, which
                    // Frame Server treats as malformed and rejects.
                    unsafe {
                        write_extended_prop_header(
                            event_data,
                            data_length,
                            EXTENDED_PROP_HEADER_SIZE,
                        );
                        if !bytes_returned.is_null() {
                            *bytes_returned = data_length.min(EXTENDED_PROP_HEADER_SIZE);
                        }
                    }
                    crate::t!("Source::KsEvent PRIVACY BASICSUPPORT -> S_OK (HEADER)");
                    return Ok(());
                }
                if flags & KSEVENT_TYPE_ENABLE != 0 {
                    crate::t!("Source::KsEvent PRIVACY ENABLE -> S_OK (never fires)");
                    return Ok(());
                }
            }
        }

        Err(KS_PROPERTY_NOT_FOUND.into())
    }
}

/// ID of `KSPROPERTY_CAMERACONTROL_PRIVACY` within
/// `PROPSETID_VIDCAP_CAMERACONTROL`. windows-rs exposes the constant as
/// a strong enum type; the trace receives and compares raw u32 values,
/// so keep the numeric form handy.
const KSPROPERTY_CAMERACONTROL_PRIVACY_ID: u32 = 8;

/// `KSPROPERTY_CAMERACONTROL_EXTENDED_ROI_CONFIGCAPS` in ksmedia.h.
const KSPROPERTY_CAMERACONTROL_EXTENDED_ROI_CONFIGCAPS_ID: u32 = 21;

/// `KSPROPERTY_CAMERACONTROL_EXTENDED_ROI_ISPCONTROL` in ksmedia.h.
const KSPROPERTY_CAMERACONTROL_EXTENDED_ROI_ISPCONTROL_ID: u32 = 22;

/// `sizeof(KSCAMERA_EXTENDEDPROP_HEADER)` = 32. Just the header; used by
/// event BASICSUPPORT probes and some SET paths that carry no payload.
const EXTENDED_PROP_HEADER_SIZE: u32 = 32;

/// `sizeof(KSCAMERA_EXTENDEDPROP_HEADER)` (32) + `sizeof(ULONGLONG)` (8).
/// Windows 11 Frame Server probes privacy with this buffer size; writing
/// a legacy `KSPROPERTY_CAMERACONTROL_S` payload (12 bytes) leaves Frame
/// Server with garbage-looking data and Start fails.
const EXTENDED_PROP_SIZE: u32 = 40;

/// `KSCAMERA_EXTENDEDPROP_PRIVACY_ON/OFF` from `ksmedia.h`.
///
/// `Flags` reports the current state (privacy shutter is off), while
/// `Capability` is a mask of states the control understands. Reporting
/// only `OFF` as a capability makes the header look like a state value
/// rather than a capability mask to stricter Frame Server builds.
const PRIVACY_ON_FLAG: u64 = 1;
const PRIVACY_OFF_FLAG: u64 = 2;
const PRIVACY_CAPABILITY_MASK: u64 = PRIVACY_ON_FLAG | PRIVACY_OFF_FLAG;

/// Write just the 32-byte KSCAMERA_EXTENDEDPROP_HEADER into `data`.
///
/// The header is the common prefix for every extended-camera-control
/// payload. Callers that need an additional per-feature value after it
/// (like the privacy-state GET, which adds a 8-byte `ULONGLONG`) should
/// use [`write_extended_prop_privacy_off`] instead.
///
/// `size_field` is what we write into the `Size` field of the header —
/// typically `data_length` for BASICSUPPORT responses where the caller
/// told us the buffer is exactly as big as the header.
unsafe fn write_extended_prop_header(
    data: *mut core::ffi::c_void,
    data_length: u32,
    size_field: u32,
) {
    if data.is_null() {
        return;
    }
    let dst = data as *mut u8;
    let n = data_length.min(EXTENDED_PROP_HEADER_SIZE) as usize;
    core::ptr::write_bytes(dst, 0, n);
    write_header_fields(dst, n, size_field);
}

/// Write the 32-byte header followed by a 8-byte ULONGLONG Value field.
/// Total 40 bytes — the shape Windows 11 Frame Server uses for
/// `KSPROPERTY_CAMERACONTROL_PRIVACY` GET.
unsafe fn write_extended_prop_privacy_off(data: *mut core::ffi::c_void, data_length: u32) {
    if data.is_null() {
        return;
    }
    let dst = data as *mut u8;
    let n = data_length.min(EXTENDED_PROP_SIZE) as usize;
    core::ptr::write_bytes(dst, 0, n);
    write_header_fields(dst, n, EXTENDED_PROP_SIZE);

    // offset 32 u64 Value = 0 (privacy not currently engaged)
    if n >= 40 {
        let bytes = 0u64.to_le_bytes();
        for i in 0..8 {
            *dst.add(32 + i) = bytes[i];
        }
    }
}

/// Fill the common `KSCAMERA_EXTENDEDPROP_HEADER` fields into the first
/// 32 bytes at `dst` (clamped by `n`).
unsafe fn write_header_fields(dst: *mut u8, n: usize, size: u32) {
    let write_u32 = |offset: usize, value: u32| {
        if offset + 4 <= n {
            let bytes = value.to_le_bytes();
            for i in 0..4 {
                *dst.add(offset + i) = bytes[i];
            }
        }
    };
    let write_u64 = |offset: usize, value: u64| {
        if offset + 8 <= n {
            let bytes = value.to_le_bytes();
            for i in 0..8 {
                *dst.add(offset + i) = bytes[i];
            }
        }
    };

    write_u32(0, 1); // Version
    write_u32(4, 0xFFFF_FFFF); // PinId = KSCAMERA_EXTENDEDPROP_FILTERSCOPE
    write_u32(8, size); // Size
    write_u32(12, 0); // Result = S_OK
    write_u64(16, PRIVACY_OFF_FLAG); // Flags (current state)
    write_u64(24, PRIVACY_CAPABILITY_MASK); // Capability (mask)
}

/// Decompose a raw `KSIDENTIFIER` pointer into `(Set, Id, Flags)`. The
/// windows-rs definition uses a nameless union for the body; we treat
/// the first 24 bytes as the fixed `{ GUID, u32, u32 }` layout KS always
/// uses. Returns `None` for a null/short buffer.
fn parse_ksid(id: *const KSIDENTIFIER, len: u32) -> Option<(GUID, u32, u32)> {
    unsafe {
        if id.is_null() || len < 24 {
            return None;
        }
        let bytes = core::slice::from_raw_parts(id as *const u8, 24);
        let set = GUID::from_values(
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            u16::from_le_bytes([bytes[4], bytes[5]]),
            u16::from_le_bytes([bytes[6], bytes[7]]),
            [
                bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14],
                bytes[15],
            ],
        );
        let prop_id = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        let flags = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        Some((set, prop_id, flags))
    }
}

fn log_ksid(label: &str, id: *const KSIDENTIFIER, len: u32) {
    unsafe {
        if id.is_null() || len < std::mem::size_of::<KSIDENTIFIER>() as u32 {
            crate::t!("Source::{label} (null or short id, len={len})");
            return;
        }
        let ksid = &*id;
        // KSIDENTIFIER overlays: Set (GUID) / Id (u32) / Flags (u32). The
        // windows-rs definition uses a union; pull the fields out by
        // treating the struct as a fixed-layout { GUID, u32, u32 }.
        let bytes = core::slice::from_raw_parts(id as *const u8, 24);
        let mut set_bytes = [0u8; 16];
        set_bytes.copy_from_slice(&bytes[..16]);
        let set_guid = GUID::from_values(
            u32::from_le_bytes([set_bytes[0], set_bytes[1], set_bytes[2], set_bytes[3]]),
            u16::from_le_bytes([set_bytes[4], set_bytes[5]]),
            u16::from_le_bytes([set_bytes[6], set_bytes[7]]),
            [
                set_bytes[8],
                set_bytes[9],
                set_bytes[10],
                set_bytes[11],
                set_bytes[12],
                set_bytes[13],
                set_bytes[14],
                set_bytes[15],
            ],
        );
        let prop_id = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        let flags = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        let _ = ksid;
        crate::t!(
            "Source::{label} set={:?} id={} flags={:#x}",
            set_guid,
            prop_id,
            flags
        );
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
                if guid_service.is_null() {
                    GUID::zeroed()
                } else {
                    *guid_service
                },
                if riid.is_null() {
                    GUID::zeroed()
                } else {
                    *riid
                },
            );
            if ppv_object.is_null() {
                return Err(E_POINTER.into());
            }
            *ppv_object = core::ptr::null_mut();

            if !riid.is_null() && *riid == IMFExtendedCameraController::IID {
                let controller: IMFExtendedCameraController = self.cast()?;
                let unk: IUnknown = controller.cast()?;
                let hr = unk.query(riid, ppv_object);
                crate::t!("Source::GetService IMFExtendedCameraController -> {:?}", hr);
                if hr.is_err() {
                    return Err(hr.into());
                }
                return Ok(());
            }
        }
        // Match MS SimpleMediaSource: callers that probe a service we
        // don't offer expect MF_E_UNSUPPORTED_SERVICE, not E_NOINTERFACE.
        // E_NOINTERFACE would suggest IMFGetService itself is missing,
        // which would be a QI-level failure rather than a per-service
        // rejection — some Frame Server code paths treat it as fatal.
        Err(MF_E_UNSUPPORTED_SERVICE.into())
    }
}

/// Windows 11 Frame Server probes modern camera controls through
/// `IMFExtendedCameraController` before falling back to KS properties.
/// A software source has no hardware controls, but surfacing a valid
/// controller keeps this probe from looking like a missing COM interface.
impl IMFExtendedCameraController_Impl for VulvatarMediaSource_Impl {
    fn GetExtendedCameraControl(
        &self,
        stream_index: u32,
        property_id: u32,
    ) -> windows::core::Result<IMFExtendedCameraControl> {
        crate::t!(
            "Source::GetExtendedCameraControl stream={} property={}",
            stream_index,
            property_id
        );

        if stream_index != DEFAULT_STREAM_ID && stream_index != MF_CAPTURE_ENGINE_MEDIASOURCE.0 {
            return Err(MF_E_INVALIDSTREAMNUMBER.into());
        }

        let control: IMFExtendedCameraControl =
            VulvatarExtendedCameraControl::new(property_id).into();
        Ok(control)
    }
}

#[implement(IMFExtendedCameraControl)]
struct VulvatarExtendedCameraControl {
    property_id: u32,
    capabilities: u64,
    flags: AtomicU64,
    payload: Vec<u8>,
}

impl VulvatarExtendedCameraControl {
    fn new(property_id: u32) -> Self {
        dll_add_ref();

        // Frame Server already asks the legacy KS privacy-control path for
        // property id 8. If it asks the modern controller for the same id,
        // answer consistently: a virtual camera has no physical privacy
        // shutter and is currently unblocked.
        let (capabilities, flags, payload) = match property_id {
            KSPROPERTY_CAMERACONTROL_PRIVACY_ID => (
                PRIVACY_CAPABILITY_MASK,
                PRIVACY_OFF_FLAG,
                0u64.to_le_bytes().to_vec(),
            ),
            KSPROPERTY_CAMERACONTROL_EXTENDED_ROI_CONFIGCAPS_ID => {
                (0, 0, build_roi_configcaps_payload())
            }
            KSPROPERTY_CAMERACONTROL_EXTENDED_ROI_ISPCONTROL_ID => {
                (0, 0, build_roi_ispcontrol_payload())
            }
            _ => (0, 0, Vec::new()),
        };

        Self {
            property_id,
            capabilities,
            flags: AtomicU64::new(flags),
            payload,
        }
    }
}

impl Drop for VulvatarExtendedCameraControl {
    fn drop(&mut self) {
        dll_release();
    }
}

impl IMFExtendedCameraControl_Impl for VulvatarExtendedCameraControl_Impl {
    fn GetCapabilities(&self) -> u64 {
        crate::t!(
            "ExtendedControl::GetCapabilities property={} -> {:#x}",
            self.property_id,
            self.capabilities
        );
        self.capabilities
    }

    fn SetFlags(&self, flags: u64) -> windows::core::Result<()> {
        crate::t!(
            "ExtendedControl::SetFlags property={} flags={:#x}",
            self.property_id,
            flags
        );
        if self.capabilities == 0 {
            self.flags.store(0, Ordering::SeqCst);
        } else {
            self.flags
                .store(flags & self.capabilities, Ordering::SeqCst);
        }
        Ok(())
    }

    fn GetFlags(&self) -> u64 {
        let flags = self.flags.load(Ordering::SeqCst);
        crate::t!(
            "ExtendedControl::GetFlags property={} -> {:#x}",
            self.property_id,
            flags
        );
        flags
    }

    fn LockPayload(
        &self,
        payload: *mut *mut u8,
        payload_len: *mut u32,
    ) -> windows::core::Result<()> {
        crate::t!(
            "ExtendedControl::LockPayload property={} len={}",
            self.property_id,
            self.payload.len()
        );
        unsafe {
            if payload.is_null() || payload_len.is_null() {
                return Err(E_POINTER.into());
            }
            *payload = if self.payload.is_empty() {
                core::ptr::null_mut()
            } else {
                self.payload.as_ptr() as *mut u8
            };
            *payload_len = self.payload.len() as u32;
        }
        Ok(())
    }

    fn UnlockPayload(&self) -> windows::core::Result<()> {
        crate::t!(
            "ExtendedControl::UnlockPayload property={}",
            self.property_id
        );
        Ok(())
    }

    fn CommitSettings(&self) -> windows::core::Result<()> {
        crate::t!(
            "ExtendedControl::CommitSettings property={}",
            self.property_id
        );
        Ok(())
    }
}

fn build_roi_configcaps_payload() -> Vec<u8> {
    // KSCAMERA_EXTENDEDPROP_ROI_CONFIGCAPSHEADER with ConfigCapCount = 0.
    // This advertises that the software source supports no ISP ROI controls
    // while still returning the well-formed header Frame Server expects.
    let mut bytes = Vec::with_capacity(16);
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes
}

fn build_roi_ispcontrol_payload() -> Vec<u8> {
    // KSCAMERA_EXTENDEDPROP_ROI_ISPCONTROLHEADER with ControlCount = 0.
    // A zero count means there are no ROIs configured.
    let mut bytes = Vec::with_capacity(16);
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes
}

/// Frame Server's FsProxy initialisation dance:
///   1. QI `IMFSampleAllocatorControl` on the source
///   2. Call `GetAllocatorUsage` — we report `UsesProvidedAllocator (0)`
///   3. Call `SetDefaultAllocator(stream_id, IMFVideoSampleAllocatorEx)` —
///      we cache it for the upcoming `VulvatarMediaStream::new` to consume
///   4. Call `IMFMediaSource::Start` (where the stream picks up the
///      cached allocator and uses `AllocateSample` for every frame).
///
/// Returning `DoesNotAllocate (2)` previously was a contract break: Frame
/// Server expects the provided allocator to back the pixel buffers in
/// marshaled `IMFSample`s, and aborts FsProxy in microseconds when CPU
/// `MFCreateMemoryBuffer`-backed samples come back instead — the
/// `ReadSample → MF_SOURCE_READERF_ENDOFSTREAM` symptom we chased for
/// hours. smourier/VCamSample uses the provided allocator throughout.
impl IMFSampleAllocatorControl_Impl for VulvatarMediaSource_Impl {
    fn SetDefaultAllocator(
        &self,
        output_stream_id: u32,
        allocator: Option<&IUnknown>,
    ) -> windows::core::Result<()> {
        crate::t!(
            "Source::SetDefaultAllocator stream={} allocator={}",
            output_stream_id,
            allocator.is_some()
        );
        let alloc = allocator.ok_or_else(|| windows::core::Error::from(E_INVALIDARG))?;
        let video_alloc: IMFVideoSampleAllocatorEx = alloc.cast()?;
        *self.stream_allocator.lock().unwrap() = Some(video_alloc);
        // If the stream was constructed BEFORE Frame Server gave us the
        // allocator (unlikely but possible), the stream won't have the
        // current allocator. We don't currently push it after the fact —
        // SetDefaultAllocator is part of the FsProxy init contract that
        // happens before Start, so the snapshot in Source::Start picks it up.
        crate::t!("Source::SetDefaultAllocator stored IMFVideoSampleAllocatorEx");
        Ok(())
    }

    fn GetAllocatorUsage(
        &self,
        _output_stream_id: u32,
        input_stream_id: *mut u32,
        usage: *mut MFSampleAllocatorUsage,
    ) -> windows::core::Result<()> {
        crate::t!("Source::GetAllocatorUsage -> UsesProvidedAllocator");
        unsafe {
            if !input_stream_id.is_null() {
                *input_stream_id = 0;
            }
            if !usage.is_null() {
                // MFSampleAllocatorUsage_UsesProvidedAllocator = 0:
                // "Source uses the allocator that the consumer provides
                //  via SetDefaultAllocator." Required for Frame Server's
                //  cross-process sample marshaling path.
                *usage = MFSampleAllocatorUsage(0);
            }
        }
        Ok(())
    }
}

impl IMFRealTimeClient_Impl for VulvatarMediaSource_Impl {
    fn RegisterThreads(
        &self,
        dwtaskindex: u32,
        _wszclass: &windows::core::PCWSTR,
    ) -> windows::core::Result<()> {
        crate::t!("Source::RegisterThreads task={}", dwtaskindex);
        Ok(())
    }

    fn UnregisterThreads(&self) -> windows::core::Result<()> {
        crate::t!("Source::UnregisterThreads");
        Ok(())
    }

    fn SetWorkQueue(&self, dwworkqueueid: u32) -> windows::core::Result<()> {
        crate::t!("Source::SetWorkQueue id={}", dwworkqueueid);
        Ok(())
    }
}

impl IMFRealTimeClientEx_Impl for VulvatarMediaSource_Impl {
    fn RegisterThreadsEx(
        &self,
        pdwtaskindex: *mut u32,
        _wszclassname: &windows::core::PCWSTR,
        lbasepriority: i32,
    ) -> windows::core::Result<()> {
        crate::t!("Source::RegisterThreadsEx priority={}", lbasepriority);
        unsafe {
            if !pdwtaskindex.is_null() {
                *pdwtaskindex = 0;
            }
        }
        Ok(())
    }

    fn UnregisterThreads(&self) -> windows::core::Result<()> {
        crate::t!("Source::UnregisterThreadsEx");
        Ok(())
    }

    fn SetWorkQueueEx(
        &self,
        dwmultithreadedworkqueueid: u32,
        lworkitembasepriority: i32,
    ) -> windows::core::Result<()> {
        crate::t!(
            "Source::SetWorkQueueEx id={} priority={}",
            dwmultithreadedworkqueueid,
            lworkitembasepriority
        );
        Ok(())
    }
}

/// Build a `width × height @ fps` media type for the given subtype GUID.
///
/// `MFSetAttributeSize` / `MFSetAttributeRatio` are inline helpers in the
/// SDK header and are not exposed by `windows-rs`; we pack the two `u32`
/// halves into a `u64` ourselves (high → numerator, low → denominator).
fn build_media_type(
    subtype: &GUID,
    width: u32,
    height: u32,
    fps: u32,
) -> windows::core::Result<IMFMediaType> {
    unsafe {
        let mt = MFCreateMediaType()?;
        mt.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
        mt.SetGUID(&MF_MT_SUBTYPE, subtype)?;
        mt.SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)?;
        mt.SetUINT32(&MF_MT_ALL_SAMPLES_INDEPENDENT, 1)?;
        mt.SetUINT32(&MF_MT_FIXED_SIZE_SAMPLES, 1)?;
        mt.SetUINT64(&MF_MT_FRAME_SIZE, pack_2x32(width, height))?;
        mt.SetUINT64(&MF_MT_FRAME_RATE, pack_2x32(fps, 1))?;
        mt.SetUINT64(&MF_MT_PIXEL_ASPECT_RATIO, pack_2x32(1, 1))?;
        // Stage 4 (output color space): advertise BT.709 + sRGB transfer
        // as the negotiated baseline. The producer can override to linear
        // on a per-sample basis via `tag_sample_color_metadata` in
        // media_stream.rs, but having the media-type defaults explicit
        // makes the negotiation deterministic for clients that only
        // consult the descriptor.
        mt.SetUINT32(&MF_MT_VIDEO_PRIMARIES, MFVideoPrimaries_BT709.0 as u32)?;
        mt.SetUINT32(&MF_MT_TRANSFER_FUNCTION, MFVideoTransFunc_sRGB.0 as u32)?;
        if *subtype == MFVideoFormat_NV12 {
            let sample_size = width.saturating_mul(height).saturating_mul(3) / 2;
            mt.SetUINT32(&MF_MT_DEFAULT_STRIDE, width)?;
            mt.SetUINT32(&MF_MT_SAMPLE_SIZE, sample_size)?;
            mt.SetUINT32(
                &MF_MT_AVG_BITRATE,
                sample_size.saturating_mul(8).saturating_mul(fps),
            )?;
        } else {
            let sample_size = width.saturating_mul(height).saturating_mul(4);
            mt.SetUINT32(&MF_MT_DEFAULT_STRIDE, width.saturating_mul(4))?;
            mt.SetUINT32(&MF_MT_SAMPLE_SIZE, sample_size)?;
            mt.SetUINT32(
                &MF_MT_AVG_BITRATE,
                sample_size.saturating_mul(8).saturating_mul(fps),
            )?;
        }
        Ok(mt)
    }
}

/// Resolutions × framerates × pixel formats this source advertises.
///
/// Order matters: clients that just take "the first type" land on
/// `1920×1080@30 NV12` (the renderer's native output, no resampling on the
/// hot path). NV12 entries are listed before RGB32 because Chrome / WebRTC
/// reject the source when only RGB32 is offered.
pub const SUPPORTED_FORMATS: &[(GUID, u32, u32, u32)] = &[
    (MFVideoFormat_NV12, 1920, 1080, 30),
    (MFVideoFormat_NV12, 1920, 1080, 60),
    (MFVideoFormat_NV12, 1280, 720, 60),
    (MFVideoFormat_NV12, 1280, 720, 30),
    (MFVideoFormat_NV12, 960, 540, 30),
    (MFVideoFormat_NV12, 640, 480, 30),
    (MFVideoFormat_NV12, 320, 240, 30),
    (MFVideoFormat_RGB32, 1920, 1080, 30),
    (MFVideoFormat_RGB32, 1280, 720, 30),
    (MFVideoFormat_RGB32, 640, 480, 30),
];

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

/// Attach the minimal sensor-profile collection expected by Windows'
/// Frame Server for virtual cameras.
///
/// Microsoft's SimpleMediaSource sample explicitly marks the Legacy
/// profile as mandatory so non profile-aware camera clients still get a
/// valid stream configuration. Without this attribute the virtual camera
/// can enumerate and accept a media type, but the Frame Server may not
/// build the stream path far enough to call `IMFMediaSource::Start`.
fn attach_sensor_profiles(attrs: &IMFAttributes) -> windows::core::Result<()> {
    unsafe {
        let profiles = MFCreateSensorProfileCollection()?;
        let legacy = MFCreateSensorProfile(&KSCAMERAPROFILE_Legacy, 0, windows::core::w!(""))?;
        legacy.AddProfileFilter(
            DEFAULT_STREAM_ID,
            windows::core::w!("((RES==;FRT<=60,1;SUT==))"),
        )?;
        profiles.AddProfile(&legacy)?;
        attrs.SetUnknown(&MF_DEVICEMFT_SENSORPROFILE_COLLECTION, &profiles)?;
    }
    Ok(())
}

// Compile-time guard: keep MFMEDIASOURCE_CHARACTERISTICS imported even when
// only its bitflags constants are referenced via the .0 conversion.
#[allow(dead_code)]
const _CHARACTERISTICS_TYPE: MFMEDIASOURCE_CHARACTERISTICS = MFMEDIASOURCE_IS_LIVE;

// ---------------------------------------------------------------------------
// IMFAttributes — forward all 28 methods to `self.attributes`.
// ---------------------------------------------------------------------------
//
// smourier/VCamSample's MediaSource IS its IMFAttributes (via CBaseAttributes
// multiple-inheritance in C++). Frame Server's FsProxy does
// `QueryInterface<IMFAttributes>(source)` during catalog bridge setup;
// without this impl it gets E_NOINTERFACE and abandons the bridge
// synchronously. The same forwarder shape is needed in `media_stream.rs`,
// so the boilerplate lives in `attributes.rs` as a `forward_imfattributes!`
// macro that targets a `self.attributes: IMFAttributes` field.

crate::forward_imfattributes!(VulvatarMediaSource_Impl);
