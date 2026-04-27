#![cfg(all(target_os = "windows", feature = "virtual-camera"))]
#![allow(unused)]
#![allow(static_mut_refs)]
#![allow(clippy::missing_safety_doc)]

use windows::core::*;
use windows::Win32::Foundation::*;
use windows::Win32::System::Com::*;

static mut SOURCE_FILTER: Option<crate::output::dshow_source::DirectShowSourceFilter> = None;

fn get_filter() -> &'static mut crate::output::dshow_source::DirectShowSourceFilter {
    unsafe {
        if SOURCE_FILTER.is_none() {
            SOURCE_FILTER = Some(crate::output::dshow_source::DirectShowSourceFilter::new(
                crate::output::dshow_source::SourceFilterConfig::default(),
            ));
        }
        SOURCE_FILTER.as_mut().unwrap()
    }
}

fn parse_guid(s: &str) -> Option<GUID> {
    let s = s.trim_matches(|c| c == '{' || c == '}');
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 5 {
        return None;
    }
    let data1 = u32::from_str_radix(parts[0], 16).ok()?;
    let data2 = u16::from_str_radix(parts[1], 16).ok()?;
    let data3 = u16::from_str_radix(parts[2], 16).ok()?;
    let mut data4 = [0u8; 8];
    let combined1 = u16::from_str_radix(parts[3], 16).ok()?;
    data4[0] = (combined1 >> 8) as u8;
    data4[1] = combined1 as u8;
    let rest = parts[4];
    for i in 0..6 {
        let byte_str = &rest[i * 2..i * 2 + 2];
        data4[2 + i] = u8::from_str_radix(byte_str, 16).ok()?;
    }
    Some(GUID::from_values(data1, data2, data3, data4))
}

#[no_mangle]
pub unsafe extern "system" fn DllGetClassObject(
    rclsid: *const GUID,
    riid: *const GUID,
    ppv: *mut *mut core::ffi::c_void,
) -> HRESULT {
    let clsid = *rclsid;
    let expected = crate::output::directshow_filter::DirectShowFilterConfig::default().clsid;
    let expected_guid = match parse_guid(expected) {
        Some(g) => g,
        None => return CLASS_E_CLASSNOTAVAILABLE,
    };

    if clsid != expected_guid {
        return CLASS_E_CLASSNOTAVAILABLE;
    }

    let iid = *riid;
    if iid == IClassFactory::IID || iid == IUnknown::IID {
        let factory = Box::new(VulVatarClassFactory);
        *ppv = Box::into_raw(factory) as *mut _;
        return S_OK;
    }

    *ppv = core::ptr::null_mut();
    E_NOINTERFACE
}

#[no_mangle]
pub unsafe extern "system" fn DllCanUnloadNow() -> HRESULT {
    S_OK
}

#[no_mangle]
pub unsafe extern "system" fn DllRegisterServer() -> HRESULT {
    let mut registrar = crate::output::directshow_filter::DirectShowFilterRegistrar::new(
        crate::output::directshow_filter::DirectShowFilterConfig::default(),
    );
    match registrar.register() {
        Ok(()) => S_OK,
        Err(_) => E_FAIL,
    }
}

#[no_mangle]
pub unsafe extern "system" fn DllUnregisterServer() -> HRESULT {
    let mut registrar = crate::output::directshow_filter::DirectShowFilterRegistrar::new(
        crate::output::directshow_filter::DirectShowFilterConfig::default(),
    );
    match registrar.unregister() {
        Ok(()) => S_OK,
        Err(_) => E_FAIL,
    }
}

struct VulVatarClassFactory;

impl IClassFactory_Impl for VulVatarClassFactory {
    fn CreateInstance(
        &self,
        punkouter: Option<&IUnknown>,
        _riid: *const GUID,
        ppvobject: *mut *mut core::ffi::c_void,
    ) -> windows::core::Result<()> {
        if punkouter.is_some() {
            unsafe {
                *ppvobject = core::ptr::null_mut();
            }
            return Err(CLASS_E_NOAGGREGATION.into());
        }

        let filter = get_filter();
        match filter.start() {
            Ok(()) => unsafe {
                *ppvobject = filter as *mut _ as *mut core::ffi::c_void;
                Ok(())
            },
            Err(_) => Err(E_FAIL.into()),
        }
    }

    fn LockServer(&self, _flock: BOOL) -> windows::core::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_guid() {
        let guid = parse_guid("{B5F1C320-2B8F-4A9C-9BDC-43B0E8E6B2E1}").unwrap();
        assert_eq!(guid.data1, 0xB5F1C320);
    }

    #[test]
    fn parse_invalid_guid_returns_none() {
        assert!(parse_guid("not-a-guid").is_none());
    }

    #[test]
    fn parse_guid_wrong_parts() {
        assert!(parse_guid("{ABCD-1234}").is_none());
    }
}
