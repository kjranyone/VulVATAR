//! Diagnostic: enumerate MediaFoundation video capture devices the same
//! way Chrome / Meet does. Prints each device's friendly name, symbolic
//! link and source-type GUID. Used to verify whether the VulVATAR Virtual
//! Camera registration is visible to MF clients.
//!
//! Run with `cargo run --bin mfenum` (no extra features required; this
//! does not depend on the renderer).

#![cfg(target_os = "windows")]

use windows::core::*;
use windows::Win32::Media::MediaFoundation::*;
use windows::Win32::System::Com::{
    CoInitializeEx, CoUninitialize, IClassFactory, COINIT_MULTITHREADED,
};
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};

const CLSID_VULVATAR_MEDIA_SOURCE: GUID = GUID::from_u128(0xB5F1C320_2B8F_4A9C_9BDC_43B0E8E6B2E1);

fn main() -> Result<()> {
    unsafe {
        let coinit = CoInitializeEx(None, COINIT_MULTITHREADED);
        MFStartup(MF_VERSION, MFSTARTUP_FULL)?;

        direct_dll_probe();
        println!();
        let mut attrs: Option<IMFAttributes> = None;
        MFCreateAttributes(&mut attrs, 1)?;
        let attrs = attrs.unwrap();
        attrs.SetGUID(
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID,
        )?;

        let mut sources: *mut Option<IMFActivate> = std::ptr::null_mut();
        let mut count: u32 = 0;
        MFEnumDeviceSources(&attrs, &mut sources, &mut count)?;

        println!("Found {} video capture device(s):", count);
        let mut vulvatar_activate: Option<IMFActivate> = None;
        for i in 0..count {
            let src = (*sources.add(i as usize)).clone();
            if let Some(act) = src {
                let name = read_attribute_string(&act, &MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME);
                let symlink = read_attribute_string(
                    &act,
                    &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,
                );
                println!("  [{}] {}", i, name);
                println!("       symlink: {}", symlink);
                if name.contains("VulVATAR") {
                    vulvatar_activate = Some(act);
                }
            } else {
                println!("  [{}] <null activate>", i);
            }
        }

        // If we found our camera, ActivateObject + read a sample to drive
        // Source::Start / Stream::RequestSample end-to-end and verify the
        // pipeline.
        if let Some(act) = vulvatar_activate {
            println!("\n--- Attempting to open VulVATAR camera and read 1 sample ---");
            let source: IMFMediaSource = match act.ActivateObject() {
                Ok(s) => s,
                Err(e) => {
                    println!("ActivateObject failed: {:?}", e);
                    MFShutdown()?;
                    return Ok(());
                }
            };
            println!("ActivateObject ok, building source reader...");

            let reader: IMFSourceReader = match MFCreateSourceReaderFromMediaSource(&source, None) {
                Ok(r) => r,
                Err(e) => {
                    println!("MFCreateSourceReaderFromMediaSource failed: {:?}", e);
                    let _ = source.Shutdown();
                    MFShutdown()?;
                    return Ok(());
                }
            };
            // ~3 second budget — Frame-Server-mediated path can take
            // 100–300ms before the first allocator-backed sample arrives,
            // and we want a clean "no producer" diagnosis rather than
            // tripping out on an unrelated transient.
            const READ_ATTEMPTS: u32 = 30;
            const READ_INTERVAL_MS: u64 = 100;
            const FLAG_ENDOFSTREAM: u32 = 0x100;
            println!(
                "Source reader created, requesting up to {} samples (timeout ~{}s)...",
                READ_ATTEMPTS,
                (READ_ATTEMPTS as u64) * READ_INTERVAL_MS / 1000,
            );

            let stream_index = MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32;
            for i in 0..READ_ATTEMPTS {
                let mut actual_stream: u32 = 0;
                let mut flags: u32 = 0;
                let mut timestamp: i64 = 0;
                let mut sample: Option<IMFSample> = None;
                let read_ret = reader.ReadSample(
                    stream_index,
                    0,
                    Some(&mut actual_stream),
                    Some(&mut flags),
                    Some(&mut timestamp),
                    Some(&mut sample),
                );
                match read_ret {
                    Ok(_) => {
                        println!(
                            "ReadSample[{}] returned: actual_stream={} flags=0x{:X} ts={} sample={}",
                            i,
                            actual_stream,
                            flags,
                            timestamp,
                            sample.is_some()
                        );
                        if let Some(s) = sample {
                            let buf_count = s.GetBufferCount().unwrap_or(0);
                            let total_len = s.GetTotalLength().unwrap_or(0);
                            println!("  sample buffers={} total_bytes={}", buf_count, total_len);
                            break;
                        }
                        if flags & FLAG_ENDOFSTREAM != 0 {
                            // Frame Server / source signalled end of
                            // stream; further reads will keep returning
                            // the same flag with no sample. Bail so the
                            // failure surfaces fast.
                            println!("ENDOFSTREAM signalled — stopping retry loop");
                            break;
                        }
                    }
                    Err(e) => {
                        println!("ReadSample[{}] failed: {:?}", i, e);
                        break;
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(READ_INTERVAL_MS));
            }

            let _ = source.Shutdown();
        }

        MFShutdown()?;
        if coinit.is_ok() {
            CoUninitialize();
        }
    }
    Ok(())
}

unsafe fn direct_dll_probe() {
    println!("--- Direct DLL activation probe (target/debug) ---");
    let Ok(exe) = std::env::current_exe() else {
        println!("current_exe failed");
        return;
    };
    let Some(dir) = exe.parent() else {
        println!("current_exe has no parent");
        return;
    };
    let dll_path = dir.join("vulvatar_mf_camera.dll");
    let dll_path_w: HSTRING = dll_path.to_string_lossy().as_ref().into();
    let module = match LoadLibraryW(PCWSTR(dll_path_w.as_ptr())) {
        Ok(m) => m,
        Err(e) => {
            println!("LoadLibraryW({}) failed: {:?}", dll_path.display(), e);
            return;
        }
    };
    let proc = GetProcAddress(module, PCSTR(c"DllGetClassObject".as_ptr().cast()));
    let Some(proc) = proc else {
        println!("GetProcAddress(DllGetClassObject) failed");
        return;
    };
    type DllGetClassObjectFn =
        unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut core::ffi::c_void) -> HRESULT;
    let dll_get_class_object: DllGetClassObjectFn = core::mem::transmute(proc);

    let mut factory_ptr: *mut core::ffi::c_void = core::ptr::null_mut();
    let hr = dll_get_class_object(
        &CLSID_VULVATAR_MEDIA_SOURCE,
        &IClassFactory::IID,
        &mut factory_ptr,
    );
    if hr.is_err() || factory_ptr.is_null() {
        println!("DllGetClassObject(IClassFactory) failed: {:?}", hr);
        return;
    }
    let factory = IClassFactory::from_raw(factory_ptr as _);
    let activate: IMFActivate = match factory.CreateInstance(None::<&IUnknown>) {
        Ok(a) => a,
        Err(e) => {
            println!("IClassFactory::CreateInstance(IMFActivate) failed: {:?}", e);
            return;
        }
    };
    println!("DllGetClassObject/CreateInstance(IMFActivate) ok");
    read_samples_from_activate(&activate, "direct-dll");
}

unsafe fn read_samples_from_activate(activate: &IMFActivate, label: &str) {
    let source: IMFMediaSource = match activate.ActivateObject() {
        Ok(s) => s,
        Err(e) => {
            println!("{label} ActivateObject(IMFMediaSource) failed: {:?}", e);
            return;
        }
    };
    println!("{label} ActivateObject(IMFMediaSource) ok");

    let reader: IMFSourceReader = match MFCreateSourceReaderFromMediaSource(&source, None) {
        Ok(r) => r,
        Err(e) => {
            println!(
                "{label} MFCreateSourceReaderFromMediaSource failed: {:?}",
                e
            );
            let _ = source.Shutdown();
            return;
        }
    };
    println!("{label} SourceReader ok");

    for i in 0..3 {
        let stream_index = MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32;
        let mut actual_stream: u32 = 0;
        let mut flags: u32 = 0;
        let mut timestamp: i64 = 0;
        let mut sample: Option<IMFSample> = None;
        let read_ret = reader.ReadSample(
            stream_index,
            0,
            Some(&mut actual_stream),
            Some(&mut flags),
            Some(&mut timestamp),
            Some(&mut sample),
        );
        match read_ret {
            Ok(_) => println!(
                "{label} ReadSample[{}]: stream={} flags=0x{:X} ts={} sample={}",
                i,
                actual_stream,
                flags,
                timestamp,
                sample.is_some()
            ),
            Err(e) => println!("{label} ReadSample[{}] failed: {:?}", i, e),
        }
    }

    let _ = source.Shutdown();
}

unsafe fn read_attribute_string(act: &IMFActivate, key: &GUID) -> String {
    let len = match act.GetStringLength(key) {
        Ok(n) => n,
        Err(_) => return String::new(),
    };
    let mut buf = vec![0u16; (len + 1) as usize];
    let mut written: u32 = 0;
    if act.GetString(key, &mut buf, Some(&mut written)).is_err() {
        return String::new();
    }
    String::from_utf16_lossy(&buf[..written as usize])
}
