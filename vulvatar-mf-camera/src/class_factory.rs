//! `IClassFactory` implementation for our single media-source COM class.
//!
//! `DllGetClassObject` hands out one of these; `MFCreateVirtualCamera`
//! ultimately calls `CreateInstance(IID_IMFMediaSource)` through it to get
//! a fresh media source for the new virtual camera.

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
            if ppv.is_null() {
                return Err(E_POINTER.into());
            }
            *ppv = core::ptr::null_mut();
            if outer.is_some() {
                return Err(CLASS_E_NOAGGREGATION.into());
            }

            let source: IUnknown = VulvatarMediaSource::new().into();
            let hr = source.query(iid, ppv);
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
