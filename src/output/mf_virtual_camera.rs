//! Wrapper around `MFCreateVirtualCamera`.
//!
//! Replaces the legacy DirectShow-only registrar (`directshow_filter`)
//! with the modern MediaFoundation virtual camera path. The companion DLL
//! `vulvatar_mf_camera.dll` (sub-crate `vulvatar-mf-camera`) hosts the
//! `IMFMediaSource` referenced by the CLSID below; it must be installed
//! once via `regsvr32 /n /i:user vulvatar_mf_camera.dll` before
//! [`MfVirtualCamera::start`] can succeed.
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
    RegCloseKey, RegCreateKeyExW, RegSetValueExW, HKEY, HKEY_CURRENT_USER, KEY_WRITE,
    REG_OPTION_NON_VOLATILE, REG_SZ,
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

        // Dev-friendly self-registration: point HKCU at our sibling DLL
        // every run so debug/release toggling and rebuilds to fresh output
        // paths "just work" without manual `regsvr32`. See
        // [`ensure_dll_registered`].
        if let Err(e) = ensure_dll_registered() {
            warn!(
                "output: could not auto-register vulvatar_mf_camera.dll ({e}); \
                 falling back to whatever regsvr32 has written previously"
            );
        }

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
// Self-registration: keep HKCU's inproc-server entry pointing at the DLL
// that sits next to the currently-running exe.
// ---------------------------------------------------------------------------

/// Ensure HKCU points at a fresh, unlocked copy of the DLL, then return.
///
/// **Why the copy?** Windows' Frame Server loads whatever path is in
/// `InprocServer32` and keeps it `LoadLibrary`-held long after our
/// process exits. If we registered `target/debug/vulvatar_mf_camera.dll`
/// directly, the next `cargo build` would hit "access denied" when
/// linking a rebuilt DLL. By copying the canonical build output to a
/// per-run scratch file and registering *that* path, the canonical build
/// output is never loaded by any client — `cargo build` is always free
/// to overwrite it.
///
/// Writes three values under HKCU:
/// ```text
/// Software\Classes\CLSID\{CLSID}               = VulVATAR Virtual Camera
/// Software\Classes\CLSID\{CLSID}\InprocServer32 = <scratch dll path>
/// Software\Classes\CLSID\{CLSID}\InprocServer32\ThreadingModel = Both
/// ```
fn ensure_dll_registered() -> Result<(), String> {
    let dll_path = prepare_scratch_dll()?;
    let clsid_key = format!("Software\\Classes\\CLSID\\{}", CLSID_VULVATAR_MEDIA_SOURCE);
    let inproc_key = format!("{clsid_key}\\InprocServer32");

    write_reg_string(&clsid_key, None, VULVATAR_CAMERA_FRIENDLY_NAME)?;
    write_reg_string(&inproc_key, None, &dll_path)?;
    write_reg_string(&inproc_key, Some("ThreadingModel"), "Both")?;
    info!(
        "output: HKCU COM class {} → {}",
        CLSID_VULVATAR_MEDIA_SOURCE, dll_path
    );
    Ok(())
}

/// Copy the canonical `vulvatar_mf_camera.dll` (produced by cargo next
/// to the running exe) into a timestamped scratch file inside
/// `<exe_dir>/mf-camera-runtime/`. Old scratch files left over from
/// previous runs are deleted best-effort; any that remain locked by a
/// still-loaded Frame Server stay on disk and are ignored.
fn prepare_scratch_dll() -> Result<String, String> {
    let exe = std::env::current_exe()
        .map_err(|e| format!("current_exe: {e}"))?;
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

    let scratch_dir = exe_dir.join("mf-camera-runtime");
    std::fs::create_dir_all(&scratch_dir)
        .map_err(|e| format!("create {}: {e}", scratch_dir.display()))?;

    // Best-effort cleanup of earlier scratch copies. Anything still held
    // open by Frame Server will fail to delete — harmless, they just sit
    // there until the next successful sweep. Cargo clean wipes the
    // parent target directory when the developer really wants a reset.
    if let Ok(rd) = std::fs::read_dir(&scratch_dir) {
        for entry in rd.flatten() {
            let _ = std::fs::remove_file(entry.path());
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

fn write_reg_string(subkey: &str, value_name: Option<&str>, value: &str) -> Result<(), String> {
    let subkey_w: HSTRING = subkey.into();
    let mut key = HKEY::default();
    let res = unsafe {
        RegCreateKeyExW(
            HKEY_CURRENT_USER,
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
