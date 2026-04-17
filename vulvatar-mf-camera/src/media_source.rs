//! Stub `IMFMediaSource` implementation.
//!
//! First-cut skeleton — most methods return `E_NOTIMPL`. Just enough wired
//! up so the DLL can be loaded by `CoCreateInstance`. The presentation
//! descriptor, media stream and frame production come in follow-up tasks.

#![allow(unused_variables)]

use windows::core::{implement, IUnknown, GUID, HRESULT, PROPVARIANT};
use windows::Win32::Foundation::E_NOTIMPL;
use windows::Win32::Media::MediaFoundation::{
    IMFAsyncCallback, IMFAsyncResult, IMFMediaEvent, IMFMediaEventGenerator_Impl,
    IMFMediaSource, IMFMediaSource_Impl, IMFPresentationDescriptor,
    MEDIA_EVENT_GENERATOR_GET_EVENT_FLAGS,
};

use crate::{dll_add_ref, dll_release};

#[implement(IMFMediaSource)]
pub struct VulvatarMediaSource;

impl VulvatarMediaSource {
    pub fn new() -> Self {
        dll_add_ref();
        Self
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
        _flags: MEDIA_EVENT_GENERATOR_GET_EVENT_FLAGS,
    ) -> windows::core::Result<IMFMediaEvent> {
        Err(E_NOTIMPL.into())
    }

    fn BeginGetEvent(
        &self,
        _callback: Option<&IMFAsyncCallback>,
        _state: Option<&IUnknown>,
    ) -> windows::core::Result<()> {
        Err(E_NOTIMPL.into())
    }

    fn EndGetEvent(
        &self,
        _result: Option<&IMFAsyncResult>,
    ) -> windows::core::Result<IMFMediaEvent> {
        Err(E_NOTIMPL.into())
    }

    fn QueueEvent(
        &self,
        _met: u32,
        _guidextendedtype: *const GUID,
        _hrstatus: HRESULT,
        _pvvalue: *const PROPVARIANT,
    ) -> windows::core::Result<()> {
        Err(E_NOTIMPL.into())
    }
}

impl IMFMediaSource_Impl for VulvatarMediaSource_Impl {
    fn GetCharacteristics(&self) -> windows::core::Result<u32> {
        Err(E_NOTIMPL.into())
    }

    fn CreatePresentationDescriptor(
        &self,
    ) -> windows::core::Result<IMFPresentationDescriptor> {
        Err(E_NOTIMPL.into())
    }

    fn Start(
        &self,
        _pd: Option<&IMFPresentationDescriptor>,
        _time_format: *const GUID,
        _start_position: *const PROPVARIANT,
    ) -> windows::core::Result<()> {
        Err(E_NOTIMPL.into())
    }

    fn Stop(&self) -> windows::core::Result<()> {
        Err(E_NOTIMPL.into())
    }

    fn Pause(&self) -> windows::core::Result<()> {
        Err(E_NOTIMPL.into())
    }

    fn Shutdown(&self) -> windows::core::Result<()> {
        Ok(())
    }
}
