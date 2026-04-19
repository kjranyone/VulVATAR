//! `IClassFactory` implementation for our single media-source COM class.
//!
//! `DllGetClassObject` hands out one of these; Frame Server asks it for
//! `IMFActivate`, and the returned media source object implements that
//! interface directly.

use core::ffi::c_void;

use windows::core::{implement, IUnknown, Interface, GUID};
use windows::Win32::Foundation::{BOOL, CLASS_E_NOAGGREGATION, E_NOINTERFACE, E_POINTER};
use windows::Win32::System::Com::{IClassFactory, IClassFactory_Impl};

use crate::{dll_add_ref, dll_release, media_source::VulvatarMediaSource};

#[implement(IClassFactory)]
pub struct VulvatarClassFactory;

impl VulvatarClassFactory {
    pub fn new() -> Self {
        dll_add_ref();
        Self
    }
}

impl Drop for VulvatarClassFactory {
    fn drop(&mut self) {
        dll_release();
    }
}

impl IClassFactory_Impl for VulvatarClassFactory_Impl {
    fn CreateInstance(
        &self,
        outer: Option<&IUnknown>,
        iid: *const GUID,
        ppv: *mut *mut c_void,
    ) -> windows::core::Result<()> {
        unsafe {
            crate::t!("ClassFactory::CreateInstance iid={:?}", *iid);
            if ppv.is_null() {
                crate::t!("ClassFactory::CreateInstance: null ppv");
                return Err(E_POINTER.into());
            }
            *ppv = core::ptr::null_mut();
            if outer.is_some() {
                crate::t!("ClassFactory::CreateInstance: aggregation");
                return Err(CLASS_E_NOAGGREGATION.into());
            }

            // Keep IMFActivate, IMFMediaSourceEx, and IMFAttributes on one
            // COM identity. Frame Server writes attributes through the
            // activate view before calling ActivateObject, then reads them
            // through the source view during FsProxy setup.
            let source_unk: IUnknown = VulvatarMediaSource::new()
                .map_err(windows::core::Error::from)?
                .into();
            let hr = source_unk.query(iid, ppv);
            crate::t!("ClassFactory::CreateInstance: source.query -> {:?}", hr);
            if hr.is_err() {
                return Err(hr.into());
            }
            if (*ppv).is_null() {
                return Err(E_NOINTERFACE.into());
            }
            Ok(())
        }
    }

    fn LockServer(&self, lock: BOOL) -> windows::core::Result<()> {
        if lock.0 != 0 {
            dll_add_ref();
        } else {
            dll_release();
        }
        Ok(())
    }
}
