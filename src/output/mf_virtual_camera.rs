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
