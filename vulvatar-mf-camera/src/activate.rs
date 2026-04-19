//! `IMFActivate` wrapper around [`crate::media_source::VulvatarMediaSource`].
//!
//! Why this exists: MFCreateVirtualCamera / Frame Server calls
//! `CoCreateInstance(CLSID, IID_IMFActivate)` on our DLL — not
//! `IMFMediaSource` directly. The MS VirtualCameraMediaSourceActivate
//! sample follows the same pattern: register an activate object, and
//! manufacture the real media source from `ActivateObject`.
//!
//! `IMFActivate` inherits `IMFAttributes`, so we also have to surface 28
//! `IMFAttributes_Impl` methods. They all forward to an internal
//! `IMFAttributes` built via `MFCreateAttributes` — no bespoke attribute
//! storage. The MS sample uses the same delegation pattern.

#![allow(non_snake_case)]

use core::ffi::c_void;

use windows::core::{implement, IUnknown, Interface as _, GUID, PROPVARIANT, PWSTR};
use windows::Win32::Foundation::{BOOL, E_POINTER};
use windows::Win32::Media::MediaFoundation::{
    IMFActivate, IMFActivate_Impl, IMFAttributes, IMFAttributes_Impl, MFCreateAttributes,
    MFT_TRANSFORM_CLSID_Attribute, MF_ATTRIBUTES_MATCH_TYPE, MF_ATTRIBUTE_TYPE,
    MF_VIRTUALCAMERA_PROVIDE_ASSOCIATED_CAMERA_SOURCES,
};

use std::sync::Mutex;

use windows::Win32::Media::MediaFoundation::{IMFMediaSource, IMFMediaSourceEx};

use crate::media_source::VulvatarMediaSource;
use crate::{dll_add_ref, dll_release, CLSID_VULVATAR_MEDIA_SOURCE};

#[implement(IMFActivate, IMFAttributes)]
pub struct VulvatarActivate {
    attributes: IMFAttributes,
    /// Cached media source. MS's VirtualCameraMediaSourceActivate keeps a
    /// strong reference so repeated `ActivateObject` calls and later QI's
    /// (e.g. for IMFGetService) observe the same instance. Creating a new
    /// source on every call is enough to confuse the Frame Server.
    cached_source: Mutex<Option<IMFMediaSource>>,
}

impl VulvatarActivate {
    pub fn new() -> windows::core::Result<Self> {
        dll_add_ref();
        let mut attrs: Option<IMFAttributes> = None;
        unsafe { MFCreateAttributes(&mut attrs, 1)? };
        let attributes = attrs.expect("MFCreateAttributes returned no interface");
        unsafe {
            // Mirrors Microsoft's VirtualCameraMediaSourceActivate sample.
            // Frame Server uses this activate-level hint while constructing
            // the software camera source graph.
            attributes.SetUINT32(&MF_VIRTUALCAMERA_PROVIDE_ASSOCIATED_CAMERA_SOURCES, 1)?;
            // Tell Frame Server *which* CLSID this activate object backs.
            // smourier/VCamSample sets this; without it Frame Server's
            // FsProxy can't resolve the source CLSID for the marshaled
            // path and short-circuits the stream with end-of-stream
            // (observed: ReadSample returns flags=0x100 immediately and
            // the DLL is never loaded into the Frame Server svchost,
            // even though direct CoCreateInstance + ActivateObject works).
            attributes.SetGUID(&MFT_TRANSFORM_CLSID_Attribute, &CLSID_VULVATAR_MEDIA_SOURCE)?;
        }
        Ok(Self {
            attributes,
            cached_source: Mutex::new(None),
        })
    }
}

impl Drop for VulvatarActivate {
    fn drop(&mut self) {
        dll_release();
    }
}

impl IMFActivate_Impl for VulvatarActivate_Impl {
    fn ActivateObject(
        &self,
        riid: *const GUID,
        ppv: *mut *mut c_void,
    ) -> windows::core::Result<()> {
        unsafe {
            crate::t!(
                "Activate::ActivateObject riid={:?}",
                if riid.is_null() { GUID::zeroed() } else { *riid }
            );
            if ppv.is_null() {
                return Err(E_POINTER.into());
            }
            *ppv = core::ptr::null_mut();

            let mut cache = self.cached_source.lock().unwrap();
            let source: IMFMediaSource = if let Some(ref s) = *cache {
                crate::t!("Activate::ActivateObject: returning cached source");
                s.clone()
            } else {
                crate::t!("Activate::ActivateObject: constructing new source");
                let source_ex: IMFMediaSourceEx = VulvatarMediaSource::new()?.into();

                // smourier/VCamSample's MediaSource::Initialize calls
                // `attributes->CopyAllItems(this)` on the activator's
                // attribute store. Frame Server stamps additional attributes
                // onto the activator BEFORE calling ActivateObject (the
                // device interface symbolic link, MF_FRAMESERVER_CLIENTCONTEXT
                // _CLIENTPID etc.); without copying them onto the source's
                // attribute store the source can't be matched against the
                // device interface that enumerated it, and FsProxy discards
                // it in microseconds — exactly the symptom we observe.
                if let Ok(source_attrs) = source_ex.GetSourceAttributes() {
                    if let Err(e) = self.attributes.CopyAllItems(&source_attrs) {
                        crate::t!(
                            "Activate::ActivateObject: CopyAllItems activator->source failed: {:?}",
                            e
                        );
                    } else {
                        crate::t!(
                            "Activate::ActivateObject: copied activator attrs into source"
                        );
                    }
                }

                let s: IMFMediaSource = source_ex.cast()?;
                *cache = Some(s.clone());
                s
            };
            drop(cache);

            let unk: IUnknown = source.cast()?;
            let hr = unk.query(riid, ppv);
            crate::t!("Activate::ActivateObject: source.query -> {:?}", hr);
            if hr.is_err() {
                return Err(hr.into());
            }
            Ok(())
        }
    }

    fn ShutdownObject(&self) -> windows::core::Result<()> {
        crate::t!("Activate::ShutdownObject");
        // Forward to the cached source so its event queue / streams wind
        // down cleanly. The cache stays populated — DetachObject is the
        // hook that drops it.
        let cache = self.cached_source.lock().unwrap();
        if let Some(ref s) = *cache {
            unsafe {
                let _ = s.Shutdown();
            }
        }
        Ok(())
    }

    fn DetachObject(&self) -> windows::core::Result<()> {
        crate::t!("Activate::DetachObject");
        let mut cache = self.cached_source.lock().unwrap();
        *cache = None;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// IMFAttributes — delegate every method to `self.attributes`.
// ---------------------------------------------------------------------------
//
// This boilerplate mirrors how the MS sample manually forwards each
// method to `m_spActivateAttributes`. A macro would compress it but
// windows-rs interface signatures vary enough (pointers vs references,
// Option<&T> params) that expanded code is easier to read and tweak.

impl IMFAttributes_Impl for VulvatarActivate_Impl {
    fn GetItem(
        &self,
        guidkey: *const GUID,
        pvalue: *mut PROPVARIANT,
    ) -> windows::core::Result<()> {
        unsafe { self.attributes.GetItem(guidkey, Some(pvalue)) }
    }

    fn GetItemType(&self, guidkey: *const GUID) -> windows::core::Result<MF_ATTRIBUTE_TYPE> {
        unsafe { self.attributes.GetItemType(guidkey) }
    }

    fn CompareItem(
        &self,
        guidkey: *const GUID,
        value: *const PROPVARIANT,
    ) -> windows::core::Result<BOOL> {
        unsafe { self.attributes.CompareItem(guidkey, value) }
    }

    fn Compare(
        &self,
        ptheirs: Option<&IMFAttributes>,
        matchtype: MF_ATTRIBUTES_MATCH_TYPE,
    ) -> windows::core::Result<BOOL> {
        unsafe {
            self.attributes
                .Compare(ptheirs.unwrap_or(&self.attributes), matchtype)
        }
    }

    fn GetUINT32(&self, guidkey: *const GUID) -> windows::core::Result<u32> {
        unsafe { self.attributes.GetUINT32(guidkey) }
    }

    fn GetUINT64(&self, guidkey: *const GUID) -> windows::core::Result<u64> {
        unsafe { self.attributes.GetUINT64(guidkey) }
    }

    fn GetDouble(&self, guidkey: *const GUID) -> windows::core::Result<f64> {
        unsafe { self.attributes.GetDouble(guidkey) }
    }

    fn GetGUID(&self, guidkey: *const GUID) -> windows::core::Result<GUID> {
        unsafe { self.attributes.GetGUID(guidkey) }
    }

    fn GetStringLength(&self, guidkey: *const GUID) -> windows::core::Result<u32> {
        unsafe { self.attributes.GetStringLength(guidkey) }
    }

    fn GetString(
        &self,
        guidkey: *const GUID,
        pwszvalue: PWSTR,
        cchbufsize: u32,
        pcchlength: *mut u32,
    ) -> windows::core::Result<()> {
        unsafe {
            let slice = core::slice::from_raw_parts_mut(pwszvalue.0, cchbufsize as usize);
            self.attributes
                .GetString(guidkey, slice, Some(pcchlength))
        }
    }

    fn GetAllocatedString(
        &self,
        guidkey: *const GUID,
        ppwszvalue: *mut PWSTR,
        pcchlength: *mut u32,
    ) -> windows::core::Result<()> {
        unsafe {
            self.attributes
                .GetAllocatedString(guidkey, ppwszvalue, pcchlength)
        }
    }

    fn GetBlobSize(&self, guidkey: *const GUID) -> windows::core::Result<u32> {
        unsafe { self.attributes.GetBlobSize(guidkey) }
    }

    fn GetBlob(
        &self,
        guidkey: *const GUID,
        pbuf: *mut u8,
        cbbufsize: u32,
        pcbblobsize: *mut u32,
    ) -> windows::core::Result<()> {
        unsafe {
            let slice = core::slice::from_raw_parts_mut(pbuf, cbbufsize as usize);
            self.attributes.GetBlob(guidkey, slice, Some(pcbblobsize))
        }
    }

    fn GetAllocatedBlob(
        &self,
        guidkey: *const GUID,
        ppbuf: *mut *mut u8,
        pcbsize: *mut u32,
    ) -> windows::core::Result<()> {
        unsafe { self.attributes.GetAllocatedBlob(guidkey, ppbuf, pcbsize) }
    }

    fn GetUnknown(
        &self,
        guidkey: *const GUID,
        riid: *const GUID,
        ppv: *mut *mut c_void,
    ) -> windows::core::Result<()> {
        // windows-rs's wrapper `GetUnknown<T: Interface>` hides the raw
        // ppv; use the vtable directly so we can forward the COM triple.
        unsafe {
            use windows::core::Interface as _;
            let vt = IMFAttributes::vtable(&self.attributes);
            (vt.GetUnknown)(IMFAttributes::as_raw(&self.attributes), guidkey, riid, ppv).ok()
        }
    }

    fn SetItem(
        &self,
        guidkey: *const GUID,
        value: *const PROPVARIANT,
    ) -> windows::core::Result<()> {
        unsafe { self.attributes.SetItem(guidkey, value) }
    }

    fn DeleteItem(&self, guidkey: *const GUID) -> windows::core::Result<()> {
        unsafe { self.attributes.DeleteItem(guidkey) }
    }

    fn DeleteAllItems(&self) -> windows::core::Result<()> {
        unsafe { self.attributes.DeleteAllItems() }
    }

    fn SetUINT32(&self, guidkey: *const GUID, unvalue: u32) -> windows::core::Result<()> {
        unsafe { self.attributes.SetUINT32(guidkey, unvalue) }
    }

    fn SetUINT64(&self, guidkey: *const GUID, unvalue: u64) -> windows::core::Result<()> {
        unsafe { self.attributes.SetUINT64(guidkey, unvalue) }
    }

    fn SetDouble(&self, guidkey: *const GUID, fvalue: f64) -> windows::core::Result<()> {
        unsafe { self.attributes.SetDouble(guidkey, fvalue) }
    }

    fn SetGUID(
        &self,
        guidkey: *const GUID,
        guidvalue: *const GUID,
    ) -> windows::core::Result<()> {
        unsafe { self.attributes.SetGUID(guidkey, guidvalue) }
    }

    fn SetString(
        &self,
        guidkey: *const GUID,
        wszvalue: &windows::core::PCWSTR,
    ) -> windows::core::Result<()> {
        unsafe { self.attributes.SetString(guidkey, *wszvalue) }
    }

    fn SetBlob(
        &self,
        guidkey: *const GUID,
        pbuf: *const u8,
        cbbufsize: u32,
    ) -> windows::core::Result<()> {
        unsafe {
            let slice = core::slice::from_raw_parts(pbuf, cbbufsize as usize);
            self.attributes.SetBlob(guidkey, slice)
        }
    }

    fn SetUnknown(
        &self,
        guidkey: *const GUID,
        punknown: Option<&IUnknown>,
    ) -> windows::core::Result<()> {
        unsafe { self.attributes.SetUnknown(guidkey, punknown) }
    }

    fn LockStore(&self) -> windows::core::Result<()> {
        unsafe { self.attributes.LockStore() }
    }

    fn UnlockStore(&self) -> windows::core::Result<()> {
        unsafe { self.attributes.UnlockStore() }
    }

    fn GetCount(&self) -> windows::core::Result<u32> {
        unsafe { self.attributes.GetCount() }
    }

    fn GetItemByIndex(
        &self,
        unindex: u32,
        pguidkey: *mut GUID,
        pvalue: *mut PROPVARIANT,
    ) -> windows::core::Result<()> {
        unsafe {
            self.attributes
                .GetItemByIndex(unindex, pguidkey, Some(pvalue))
        }
    }

    fn CopyAllItems(&self, pdest: Option<&IMFAttributes>) -> windows::core::Result<()> {
        unsafe { self.attributes.CopyAllItems(pdest) }
    }
}
