//! Wrapper around `MFCreateVirtualCamera`.
//!
//! Replaces the legacy DirectShow-only registrar (`directshow_filter`)
//! with the modern MediaFoundation virtual camera path. The companion DLL
//! `vulvatar_mf_camera.dll` (sub-crate `vulvatar-mf-camera`) hosts the
//! `IMFMediaSource` referenced by the CLSID below. Windows' Frame Server
//! loads that COM server out of HKLM registration.
//!
//! The virtual camera is created with `MFVirtualCameraLifetime_Session`
//! so it disappears the moment the main process exits — no cleanup needed
//! across runs, no stale registrations.

#![cfg(all(target_os = "windows", feature = "virtual-camera"))]

use log::{info, warn};
use windows::core::{HSTRING, PCWSTR};
use windows::Win32::Media::MediaFoundation::{
    IMFVirtualCamera, MFCreateVirtualCamera, MFStartup, MFVirtualCameraAccess_CurrentUser,
    MFVirtualCameraLifetime_Session, MFVirtualCameraType_SoftwareCameraSource, MFSTARTUP_FULL,
    MF_VERSION,
};
use windows::Win32::System::Registry::{
    RegCloseKey, RegCreateKeyExW, RegDeleteTreeW, RegOpenKeyExW, RegQueryValueExW, RegSetValueExW,
    HKEY, HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE, KEY_READ, KEY_WRITE, REG_OPTION_NON_VOLATILE,
    REG_SZ, REG_VALUE_TYPE,
};

/// Same CLSID as `vulvatar_mf_camera::CLSID_VULVATAR_MEDIA_SOURCE`. The
/// constant is duplicated here rather than imported because the DLL is a
/// `cdylib` (not an `rlib`) and is not in this crate's dependency graph.
pub const CLSID_VULVATAR_MEDIA_SOURCE: &str = "{B5F1C320-2B8F-4A9C-9BDC-43B0E8E6B2E1}";

pub const VULVATAR_CAMERA_FRIENDLY_NAME: &str = "VulVATAR Virtual Camera";

#[derive(Debug)]
pub enum MfVirtualCameraState {
    Stopped,
    Started,
    Failed(String),
}

pub struct MfVirtualCamera {
    state: MfVirtualCameraState,
    camera: Option<IMFVirtualCamera>,
    mf_started: bool,
}

impl MfVirtualCamera {
    pub fn new() -> Self {
        Self {
            state: MfVirtualCameraState::Stopped,
            camera: None,
            mf_started: false,
        }
    }

    pub fn state(&self) -> &MfVirtualCameraState {
        &self.state
    }

    pub fn is_started(&self) -> bool {
        matches!(self.state, MfVirtualCameraState::Started)
    }

    /// Register and start the virtual camera. The friendly name and the
    /// hosted media-source CLSID are baked in.
    pub fn start(&mut self) -> Result<(), String> {
        if self.is_started() {
            return Ok(());
        }

        ensure_dll_registered()?;

        unsafe {
            if !self.mf_started {
                MFStartup(MF_VERSION, MFSTARTUP_FULL)
                    .map_err(|e| format!("MFStartup failed: {e}"))?;
                self.mf_started = true;
            }

            let friendly_name: HSTRING = VULVATAR_CAMERA_FRIENDLY_NAME.into();
            let source_id: HSTRING = CLSID_VULVATAR_MEDIA_SOURCE.into();
            let camera = MFCreateVirtualCamera(
                MFVirtualCameraType_SoftwareCameraSource,
                MFVirtualCameraLifetime_Session,
                MFVirtualCameraAccess_CurrentUser,
                PCWSTR(friendly_name.as_ptr()),
                PCWSTR(source_id.as_ptr()),
                None,
            )
            .map_err(|e| format!("MFCreateVirtualCamera failed: {e}"))?;

            camera
                .Start(None)
                .map_err(|e| format!("IMFVirtualCamera::Start failed: {e}"))?;

            info!(
                "mf-virtual-camera: started (friendly='{}', source_id={})",
                VULVATAR_CAMERA_FRIENDLY_NAME, CLSID_VULVATAR_MEDIA_SOURCE
            );
            self.camera = Some(camera);
            self.state = MfVirtualCameraState::Started;
        }
        Ok(())
    }

    /// Stop the virtual camera. Safe to call multiple times.
    pub fn stop(&mut self) {
        if let Some(cam) = self.camera.take() {
            unsafe {
                if let Err(e) = cam.Stop() {
                    warn!("mf-virtual-camera: Stop failed: {e}");
                }
                if let Err(e) = cam.Remove() {
                    warn!("mf-virtual-camera: Remove failed: {e}");
                }
            }
        }
        self.state = MfVirtualCameraState::Stopped;
    }
}

impl Drop for MfVirtualCamera {
    fn drop(&mut self) {
        self.stop();
        if self.mf_started {
            unsafe {
                let _ = windows::Win32::Media::MediaFoundation::MFShutdown();
            }
            self.mf_started = false;
        }
    }
}

// ---------------------------------------------------------------------------
// Self-registration: keep HKLM's inproc-server entry pointing at a system
// install copy of the DLL.
// ---------------------------------------------------------------------------

/// Ensure HKLM has a usable registration for the media-source DLL.
///
/// **Why the copy?** Windows' Frame Server loads whatever path is in
/// `InprocServer32` and keeps it `LoadLibrary`-held long after our
/// process exits. If we registered `target/debug/vulvatar_mf_camera.dll`
/// directly, the next `cargo build` would hit "access denied" when
/// linking a rebuilt DLL. By copying the canonical build output to
/// `Program Files\VulVATAR\VirtualCamera\` under a timestamped filename,
/// the canonical build output is never loaded by any client and Frame
/// Server does not need to load code from the developer tree.
///
/// Writes three values under HKLM:
/// ```text
/// Software\Classes\CLSID\{CLSID}               = VulVATAR Virtual Camera
/// Software\Classes\CLSID\{CLSID}\InprocServer32 = <scratch dll path>
/// Software\Classes\CLSID\{CLSID}\InprocServer32\ThreadingModel = Both
/// ```
///
/// HKCU registration is not enough here: `MFCreateVirtualCamera` loads the
/// media source in FrameServerMonitor / FrameServer services. HKCU can pass
/// local probes and still fail later with HRESULT_FROM_WIN32(ERROR_PATH_NOT_FOUND).
/// It is also actively harmful once HKLM is installed because HKCR resolves
/// HKCU before HKLM for the current user.
fn ensure_dll_registered() -> Result<(), String> {
    let clsid_key = format!("Software\\Classes\\CLSID\\{}", CLSID_VULVATAR_MEDIA_SOURCE);
    let inproc_key = format!("{clsid_key}\\InprocServer32");

    remove_legacy_hkcu_registration(&clsid_key);

    if let Some(path) = read_registered_dll_path(&inproc_key).ok().flatten() {
        if std::path::Path::new(&path).exists() && is_system_install_path(&path) {
            info!("output: using existing HKLM COM class {}", path);
            return Ok(());
        }
        warn!("output: ignoring non-system HKLM COM class {}", path);
    }

    let dll_path = prepare_scratch_dll().map_err(|e| {
        format!(
            "{e}; run VulVATAR elevated once or use dev.ps1 -> install mf virtual camera (HKLM)"
        )
    })?;
    write_hklm_registration(&clsid_key, &inproc_key, &dll_path).map_err(|e| {
        format!(
            "{e}; run VulVATAR elevated once or use dev.ps1 -> install mf virtual camera (HKLM)"
        )
    })?;

    info!(
        "output: HKLM COM class {} -> {}",
        CLSID_VULVATAR_MEDIA_SOURCE, dll_path
    );
    Ok(())
}

fn remove_legacy_hkcu_registration(clsid_key: &str) {
    let subkey_w: HSTRING = clsid_key.into();
    let res = unsafe { RegDeleteTreeW(HKEY_CURRENT_USER, PCWSTR(subkey_w.as_ptr())) };
    if res.is_ok() {
        info!(
            "output: removed legacy HKCU COM class {}",
            CLSID_VULVATAR_MEDIA_SOURCE
        );
    }
}

fn write_hklm_registration(
    clsid_key: &str,
    inproc_key: &str,
    dll_path: &str,
) -> Result<(), String> {
    write_reg_string(
        HKEY_LOCAL_MACHINE,
        clsid_key,
        None,
        VULVATAR_CAMERA_FRIENDLY_NAME,
    )?;
    write_reg_string(HKEY_LOCAL_MACHINE, inproc_key, None, dll_path)?;
    write_reg_string(
        HKEY_LOCAL_MACHINE,
        inproc_key,
        Some("ThreadingModel"),
        "Both",
    )?;
    Ok(())
}

/// Copy the canonical `vulvatar_mf_camera.dll` (produced by cargo next
/// to the running exe) into a timestamped install file inside
/// `%ProgramFiles%\VulVATAR\VirtualCamera\`. Old install files left over
/// from previous runs are deleted best-effort; any that remain locked by a
/// still-loaded Frame Server stay on disk and are ignored.
fn prepare_scratch_dll() -> Result<String, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let exe_dir = exe
        .parent()
        .ok_or_else(|| "exe has no parent directory".to_string())?;

    let source = exe_dir.join("vulvatar_mf_camera.dll");
    if !source.exists() {
        return Err(format!(
            "{} not found — `cargo build` should produce it alongside the exe",
            source.display()
        ));
    }

    let scratch_dir = program_files_dir()?.join("VulVATAR").join("VirtualCamera");
    std::fs::create_dir_all(&scratch_dir)
        .map_err(|e| format!("create {}: {e}", scratch_dir.display()))?;

    // Best-effort cleanup of earlier scratch copies. Anything still held
    // open by Frame Server will fail to delete — harmless, they just sit
    // there until the next successful sweep. Cargo clean wipes the
    // parent target directory when the developer really wants a reset.
    if let Ok(rd) = std::fs::read_dir(&scratch_dir) {
        for entry in rd.flatten() {
            let path = entry.path();
            if path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|name| {
                    name.starts_with("vulvatar_mf_camera_") && name.ends_with(".dll")
                })
            {
                let _ = std::fs::remove_file(path);
            }
        }
    }

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let target = scratch_dir.join(format!("vulvatar_mf_camera_{ts}.dll"));
    std::fs::copy(&source, &target)
        .map_err(|e| format!("copy DLL to scratch {}: {e}", target.display()))?;

    target
        .to_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "scratch path contains non-UTF8 characters".to_string())
}

fn program_files_dir() -> Result<std::path::PathBuf, String> {
    std::env::var_os("ProgramFiles")
        .map(std::path::PathBuf::from)
        .ok_or_else(|| "%ProgramFiles% is not set".to_string())
}

fn is_system_install_path(path: &str) -> bool {
    let Ok(program_files) = program_files_dir() else {
        return false;
    };
    let Ok(path) = std::path::Path::new(path).canonicalize() else {
        return false;
    };
    let Ok(program_files) = program_files.canonicalize() else {
        return false;
    };
    path.starts_with(program_files)
}

fn write_reg_string(
    root: HKEY,
    subkey: &str,
    value_name: Option<&str>,
    value: &str,
) -> Result<(), String> {
    let subkey_w: HSTRING = subkey.into();
    let mut key = HKEY::default();
    let res = unsafe {
        RegCreateKeyExW(
            root,
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
        return Err(format!("RegCreateKeyExW({subkey}): {res:?}"));
    }

    let value_w: HSTRING = value.into();
    let value_name_w: Option<HSTRING> = value_name.map(|n| n.into());
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
        return Err(format!("RegSetValueExW({subkey}): {res:?}"));
    }
    Ok(())
}

fn read_registered_dll_path(subkey: &str) -> Result<Option<String>, String> {
    let subkey_w: HSTRING = subkey.into();
    let mut key = HKEY::default();
    let res = unsafe {
        RegOpenKeyExW(
            HKEY_LOCAL_MACHINE,
            PCWSTR(subkey_w.as_ptr()),
            0,
            KEY_READ,
            &mut key,
        )
    };
    if res.is_err() {
        return Ok(None);
    }

    let mut value_type = REG_VALUE_TYPE(0);
    let mut byte_len = 0u32;
    let res = unsafe {
        RegQueryValueExW(
            key,
            PCWSTR::null(),
            None,
            Some(&mut value_type),
            None,
            Some(&mut byte_len),
        )
    };
    if res.is_err() || value_type != REG_SZ || byte_len < 2 {
        let _ = unsafe { RegCloseKey(key) };
        return Ok(None);
    }

    let mut bytes = vec![0u8; byte_len as usize];
    let res = unsafe {
        RegQueryValueExW(
            key,
            PCWSTR::null(),
            None,
            Some(&mut value_type),
            Some(bytes.as_mut_ptr()),
            Some(&mut byte_len),
        )
    };
    let _ = unsafe { RegCloseKey(key) };
    if res.is_err() || value_type != REG_SZ {
        return Ok(None);
    }

    let words: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .take_while(|ch| *ch != 0)
        .collect();
    String::from_utf16(&words)
        .map(Some)
        .map_err(|e| format!("decode registered DLL path: {e}"))
}
