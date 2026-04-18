//! VulVATAR MediaFoundation virtual camera DLL.
//!
//! Exposes a `IMFMediaSource` COM class that Windows' Frame Server can
//! surface as a real camera when `MFCreateVirtualCamera` is called with
//! this DLL's CLSID. Produces a test-pattern image until the shared-memory
//! frame pipe is wired up.
//!
//! Build:
//! ```text
//! cargo build -p vulvatar-mf-camera
//! ```
//! Output lands at `target/debug/vulvatar_mf_camera.dll`.
//!
//! Install (per-user, no admin):
//! ```text
//! regsvr32 /n /i:user vulvatar_mf_camera.dll
//! ```

#![cfg(target_os = "windows")]
#![allow(non_snake_case)]

mod activate;
mod class_factory;
mod media_source;
mod media_stream;
mod registry;
mod shared_memory;
#[macro_use]
pub mod trace;

use core::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

use windows::core::{IUnknown, Interface, GUID, HRESULT};
use windows::Win32::Foundation::{BOOL, CLASS_E_CLASSNOTAVAILABLE, E_POINTER, S_FALSE, S_OK};
use windows::Win32::System::Com::IClassFactory;

/// CLSID of the virtual camera media source. Keep in sync with the main
/// app's registration.
pub const CLSID_VULVATAR_MEDIA_SOURCE: GUID =
    GUID::from_u128(0xB5F1C320_2B8F_4A9C_9BDC_43B0E8E6B2E1);

/// Friendly name shown in camera pickers.
pub const VULVATAR_CAMERA_FRIENDLY_NAME: &str = "VulVATAR Virtual Camera";

/// Global DLL object counter. Kept for `DllCanUnloadNow`; the process can
/// unload us only when every live media source / class factory / stream
/// has been released.
pub(crate) static DLL_OBJECT_COUNT: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn dll_add_ref() {
    DLL_OBJECT_COUNT.fetch_add(1, Ordering::SeqCst);
}

pub(crate) fn dll_release() {
    DLL_OBJECT_COUNT.fetch_sub(1, Ordering::SeqCst);
}

/// `DllGetClassObject` — COM entry point. Returns an `IClassFactory` for
/// our single media source class.
/// Build-time identifier baked into the DLL. If the FrameServer reuses an
/// older scratch copy, the trace at the top of each run will show the
/// stale marker and we know not to trust subsequent diagnostics.
const BUILD_MARKER: &str = concat!("nv12-primary ", file!(), ":", line!());

#[no_mangle]
pub unsafe extern "system" fn DllGetClassObject(
    rclsid: *const GUID,
    riid: *const GUID,
    ppv: *mut *mut c_void,
) -> HRESULT {
    t!("DllGetClassObject enter marker={}", BUILD_MARKER);
    if rclsid.is_null() || riid.is_null() || ppv.is_null() {
        t!("DllGetClassObject: null pointer");
        return E_POINTER;
    }
    *ppv = core::ptr::null_mut();

    if *rclsid != CLSID_VULVATAR_MEDIA_SOURCE {
        t!("DllGetClassObject: CLSID mismatch");
        return CLASS_E_CLASSNOTAVAILABLE;
    }

    let factory: IClassFactory = class_factory::VulvatarClassFactory::new().into();
    let unk: IUnknown = factory.cast().unwrap_or_else(|_| factory.clone().into());
    let hr = unk.query(riid, ppv);
    t!("DllGetClassObject: query riid={:?} -> {:?}", *riid, hr);
    hr
}

/// `DllCanUnloadNow` — COM hook letting the loader release this DLL.
#[no_mangle]
pub extern "system" fn DllCanUnloadNow() -> HRESULT {
    if DLL_OBJECT_COUNT.load(Ordering::SeqCst) == 0 {
        S_OK
    } else {
        S_FALSE
    }
}

/// `DllRegisterServer` — called by `regsvr32`. Writes per-user COM class
/// registration so non-admin installs are possible.
#[no_mangle]
pub extern "system" fn DllRegisterServer() -> HRESULT {
    match registry::register(registry::Scope::PerUser) {
        Ok(()) => S_OK,
        Err(hr) => hr,
    }
}

/// `DllUnregisterServer` — called by `regsvr32 /u`.
#[no_mangle]
pub extern "system" fn DllUnregisterServer() -> HRESULT {
    match registry::unregister(registry::Scope::PerUser) {
        Ok(()) => S_OK,
        Err(hr) => hr,
    }
}

/// `DllInstall` — honours `regsvr32 /i:user` (per-user) vs `/i:"machine"`.
#[no_mangle]
pub unsafe extern "system" fn DllInstall(install: BOOL, cmd_line: *const u16) -> HRESULT {
    let scope = parse_scope(cmd_line);
    let result = if install.0 != 0 {
        registry::register(scope)
    } else {
        registry::unregister(scope)
    };
    match result {
        Ok(()) => S_OK,
        Err(hr) => hr,
    }
}

/// Parse the `pszCmdLine` argument passed by `regsvr32 /i:<string>`.
unsafe fn parse_scope(cmd_line: *const u16) -> registry::Scope {
    if cmd_line.is_null() {
        return registry::Scope::PerUser;
    }
    let mut len = 0usize;
    while *cmd_line.add(len) != 0 {
        len += 1;
        if len > 256 {
            break;
        }
    }
    let slice = core::slice::from_raw_parts(cmd_line, len);
    let s = String::from_utf16_lossy(slice).to_ascii_lowercase();
    if s.contains("machine") || s.contains("all") {
        registry::Scope::AllUsers
    } else {
        registry::Scope::PerUser
    }
}
