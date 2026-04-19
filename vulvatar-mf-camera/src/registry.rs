//! COM inproc-server registration.
//!
//! Writes the minimum registry shape that `CoCreateInstance` (and in turn
//! `MFCreateVirtualCamera`) needs to load our DLL:
//!
//! ```text
//! HKCU\Software\Classes\CLSID\{CLSID}                       = friendly name
//! HKCU\Software\Classes\CLSID\{CLSID}\InprocServer32        = <dll path>
//! HKCU\Software\Classes\CLSID\{CLSID}\InprocServer32\ThreadingModel = Both
//! ```
//!
//! Per-user scope (HKCU) keeps installs admin-free; the same DLL can be
//! registered for all users (HKLM) by passing [`Scope::AllUsers`] to
//! [`register`], which requires elevation.

use windows::core::{HRESULT, HSTRING, PCWSTR};
use windows::Win32::Foundation::{E_FAIL, E_UNEXPECTED};
use windows::Win32::System::LibraryLoader::GetModuleFileNameW;
use windows::Win32::System::Registry::{
    RegCloseKey, RegCreateKeyExW, RegDeleteTreeW, RegSetValueExW, HKEY, HKEY_CURRENT_USER,
    HKEY_LOCAL_MACHINE, KEY_WRITE, REG_OPTION_NON_VOLATILE, REG_SZ,
};

use crate::{CLSID_VULVATAR_MEDIA_SOURCE, VULVATAR_CAMERA_FRIENDLY_NAME};

#[derive(Clone, Copy, Debug)]
pub enum Scope {
    PerUser,
    AllUsers,
}

impl Scope {
    fn root(self) -> HKEY {
        match self {
            Self::PerUser => HKEY_CURRENT_USER,
            Self::AllUsers => HKEY_LOCAL_MACHINE,
        }
    }
}

pub fn register(scope: Scope) -> Result<(), HRESULT> {
    let dll_path = current_module_path()?;
    let clsid_key = format!(
        "Software\\Classes\\CLSID\\{}",
        format_clsid(&CLSID_VULVATAR_MEDIA_SOURCE)
    );
    let inproc_key = format!("{clsid_key}\\InprocServer32");

    write_string_value(scope, &clsid_key, None, VULVATAR_CAMERA_FRIENDLY_NAME)?;
    write_string_value(scope, &inproc_key, None, &dll_path)?;
    write_string_value(scope, &inproc_key, Some("ThreadingModel"), "Both")?;
    Ok(())
}

pub fn unregister(scope: Scope) -> Result<(), HRESULT> {
    let clsid_key = format!(
        "Software\\Classes\\CLSID\\{}",
        format_clsid(&CLSID_VULVATAR_MEDIA_SOURCE)
    );
    let w: HSTRING = clsid_key.into();
    let res = unsafe { RegDeleteTreeW(scope.root(), PCWSTR(w.as_ptr())) };
    if res.is_err() {
        // Treat "key not found" as success so regsvr32 /u is idempotent.
        let code = res.0 as i32;
        if code != 2 {
            return Err(HRESULT(res.0 as i32));
        }
    }
    Ok(())
}

fn write_string_value(
    scope: Scope,
    subkey: &str,
    value_name: Option<&str>,
    value: &str,
) -> Result<(), HRESULT> {
    let subkey_w: HSTRING = subkey.into();
    let mut key = HKEY::default();
    let res = unsafe {
        RegCreateKeyExW(
            scope.root(),
            PCWSTR(subkey_w.as_ptr()),
            0,
            None,
            REG_OPTION_NON_VOLATILE,
            KEY_WRITE,
            None,
            &mut key,
            None,
        )
    };
    if res.is_err() {
        return Err(HRESULT(res.0 as i32));
    }

    let value_w: HSTRING = value.into();
    let value_name_w: Option<HSTRING> = value_name.map(|n| n.into());
    // Include the trailing NUL in the byte length — Windows expects REG_SZ
    // strings to be null-terminated on disk.
    let value_bytes = unsafe {
        core::slice::from_raw_parts(value_w.as_ptr() as *const u8, (value_w.len() + 1) * 2)
    };
    let name_pcwstr = value_name_w
        .as_ref()
        .map(|s| PCWSTR(s.as_ptr()))
        .unwrap_or(PCWSTR::null());

    let res = unsafe { RegSetValueExW(key, name_pcwstr, 0, REG_SZ, Some(value_bytes)) };
    let _ = unsafe { RegCloseKey(key) };
    if res.is_err() {
        return Err(HRESULT(res.0 as i32));
    }
    Ok(())
}

fn current_module_path() -> Result<String, HRESULT> {
    let mut buf = vec![0u16; 1024];
    // Passing the module-handle of this DLL is the clean way; doing that
    // portably in pure Rust requires `__ImageBase` trickery. Easier: look
    // up our own DLL by module name via GetModuleHandleW, but to get that
    // handle dynamically we instead pass a pointer that lives inside the
    // DLL to GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS).
    unsafe {
        use windows::Win32::System::LibraryLoader::{
            GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        };
        let mut hmod = windows::Win32::Foundation::HMODULE(core::ptr::null_mut());
        let addr = current_module_path as *const () as *const u16;
        let got = GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            PCWSTR(addr),
            &mut hmod,
        );
        if got.is_err() || hmod.0.is_null() {
            return Err(E_FAIL);
        }
        let written = GetModuleFileNameW(hmod, &mut buf);
        if written == 0 {
            return Err(E_UNEXPECTED);
        }
        buf.truncate(written as usize);
        String::from_utf16(&buf).map_err(|_| E_UNEXPECTED)
    }
}

fn format_clsid(g: &windows::core::GUID) -> String {
    format!(
        "{{{:08X}-{:04X}-{:04X}-{:02X}{:02X}-{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}}}",
        g.data1,
        g.data2,
        g.data3,
        g.data4[0],
        g.data4[1],
        g.data4[2],
        g.data4[3],
        g.data4[4],
        g.data4[5],
        g.data4[6],
        g.data4[7],
    )
}
