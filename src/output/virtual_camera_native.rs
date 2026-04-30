#![allow(unused_imports)]
use log::{info, warn};

#[derive(Clone, Debug)]
pub struct VirtualCameraConfig {
    pub device_name: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

impl Default for VirtualCameraConfig {
    fn default() -> Self {
        Self {
            device_name: "VulVATAR Output".to_string(),
            width: 1920,
            height: 1080,
            fps: 60,
        }
    }
}

#[derive(Clone, Debug)]
pub enum VirtualCameraState {
    Unregistered,
    Registering,
    Registered { device_path: String },
    Failed(String),
}

pub struct VirtualCameraNative {
    config: VirtualCameraConfig,
    state: VirtualCameraState,
    #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
    shared_buffer: Option<windows::core::HSTRING>,
    #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
    frame_ready_event: Option<isize>,
    #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
    com_initialized: bool,
    #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
    shared_mapping: Option<isize>,
    #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
    shared_view: Option<*mut std::ffi::c_void>,
    #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
    shared_mapping_size: usize,
}

impl VirtualCameraNative {
    pub fn new(config: VirtualCameraConfig) -> Self {
        Self {
            config,
            state: VirtualCameraState::Unregistered,
            #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
            shared_buffer: None,
            #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
            frame_ready_event: None,
            #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
            com_initialized: false,
            #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
            shared_mapping: None,
            #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
            shared_view: None,
            #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
            shared_mapping_size: 0,
        }
    }

    pub fn state(&self) -> &VirtualCameraState {
        &self.state
    }

    pub fn config(&self) -> &VirtualCameraConfig {
        &self.config
    }

    pub fn register(&mut self) -> Result<(), String> {
        #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
        {
            self.state = VirtualCameraState::Registering;
            info!(
                "virtual-camera: registering virtual camera '{}'",
                self.config.device_name
            );
            match self.register_native() {
                Ok(path) => {
                    self.state = VirtualCameraState::Registered { device_path: path };
                    info!("virtual-camera: registered successfully");
                    Ok(())
                }
                Err(e) => {
                    self.state = VirtualCameraState::Failed(e.clone());
                    warn!("virtual-camera: registration failed: {}", e);
                    Err(e)
                }
            }
        }

        #[cfg(not(all(target_os = "windows", feature = "virtual-camera")))]
        {
            let msg =
                "virtual camera registration requires the `virtual-camera` feature on Windows"
                    .to_string();
            self.state = VirtualCameraState::Failed(msg.clone());
            Err(msg)
        }
    }

    #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
    fn register_native(&mut self) -> Result<String, String> {
        use windows::Win32::System::Com::CoInitializeEx;
        use windows::Win32::System::Com::COINIT_APARTMENTTHREADED;

        unsafe {
            let hr = CoInitializeEx(None, COINIT_APARTMENTTHREADED);
            if hr.is_err() && hr != windows::Win32::Foundation::RPC_E_CHANGED_MODE {
                warn!(
                    "virtual-camera: COM init returned {:?} (may already be initialized)",
                    hr
                );
            }
            if hr.is_ok() {
                self.com_initialized = true;
            }
        }

        self.register_media_foundation()
    }

    #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
    fn register_media_foundation(&mut self) -> Result<String, String> {
        use windows::Win32::Media::MediaFoundation::MFStartup;
        use windows::Win32::Media::MediaFoundation::MFSTARTUP_NOSOCKET;
        use windows::Win32::Media::MediaFoundation::MF_VERSION;

        unsafe {
            let hr = MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET);
            if hr.is_err() {
                return Err(format!("MFStartup failed: {:?}", hr));
            }
        }

        let width = self.config.width;
        let height = self.config.height;

        let map_name =
            windows::core::HSTRING::from(format!("VulVATAR_SharedFrame_{}_{}", width, height));

        self.shared_buffer = Some(map_name);

        let event_name =
            windows::core::HSTRING::from(format!("VulVATAR_FrameReady_{}_{}", width, height));
        self.frame_ready_event = unsafe {
            use windows::Win32::Foundation::*;
            use windows::Win32::System::Threading::*;

            let handle = CreateEventW(None, TRUE, FALSE, &event_name)
                .map_err(|e| format!("CreateEventW failed: {:?}", e))?;
            Some(handle.0 as isize)
        };

        let device_path = format!(
            "VulVATAR Virtual Camera ({}x{}@{}fps)",
            width, height, self.config.fps
        );

        info!(
            "virtual-camera: MF initialized, shared buffer + event ready for '{}'",
            device_path
        );

        Ok(device_path)
    }

    pub fn push_frame(&mut self, pixel_data: &[u8], width: u32, height: u32) -> Result<(), String> {
        match &self.state {
            VirtualCameraState::Registered { .. } => {}
            VirtualCameraState::Failed(e) => return Err(format!("camera in failed state: {}", e)),
            _ => return Err("camera not registered".to_string()),
        }

        let expected = (width as usize) * (height as usize) * 4;
        if pixel_data.len() < expected {
            return Err(format!(
                "frame data too small: {} < {} bytes",
                pixel_data.len(),
                expected
            ));
        }

        #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
        {
            self.write_shared_buffer(pixel_data, width, height)?;
        }

        let _ = (pixel_data, width, height);
        Ok(())
    }

    #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
    unsafe fn ensure_mapping(&mut self, total_size: usize) -> Result<*mut u8, String> {
        use windows::Win32::Foundation::*;
        use windows::Win32::System::Memory::*;

        if self.shared_mapping_size >= total_size {
            if let Some(view) = self.shared_view {
                return Ok(view as *mut u8);
            }
        }

        if let Some(view) = self.shared_view.take() {
            let _ = UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view });
        }
        if let Some(handle) = self.shared_mapping.take() {
            let _ = CloseHandle(HANDLE(handle as *mut _));
        }

        let map_name = match &self.shared_buffer {
            Some(n) => n.clone(),
            None => return Err("no shared buffer name".to_string()),
        };

        // Pass a permissive DACL so the interactive-user producer can
        // still write the section if a consumer (FrameServer-hosted MF
        // DLL under LocalService) wins the CreateFileMappingW race.
        // See shmem_security.rs.
        let security =
            super::shmem_security::PermissiveSharedMemorySecurity::new()?;

        let handle = CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            Some(security.as_win_ptr()),
            PAGE_READWRITE,
            (total_size >> 32) as u32,
            total_size as u32,
            &map_name,
        )
        .map_err(|e| format!("CreateFileMappingW failed: {:?}", e))?;

        let view = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, total_size);
        if view.Value.is_null() {
            let _ = CloseHandle(handle);
            return Err("MapViewOfFile returned null".to_string());
        }

        self.shared_mapping = Some(handle.0 as isize);
        self.shared_view = Some(view.Value);
        self.shared_mapping_size = total_size;

        Ok(view.Value as *mut u8)
    }

    #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
    fn write_shared_buffer(
        &mut self,
        pixel_data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        let frame_size = (width as usize) * (height as usize) * 4;
        let total_size = 64usize + frame_size;

        unsafe {
            let ptr = self.ensure_mapping(total_size)?;
            let buf = std::slice::from_raw_parts_mut(ptr, total_size);

            let width_bytes = width.to_le_bytes();
            let height_bytes = height.to_le_bytes();
            let frame_index = self.frame_index().to_le_bytes();
            buf[0..4].copy_from_slice(b"VVC0");
            buf[4..8].copy_from_slice(&width_bytes);
            buf[8..12].copy_from_slice(&height_bytes);
            buf[12..16].copy_from_slice(&[0u8; 4]);
            buf[16..24].copy_from_slice(&frame_index);
            buf[24..32].copy_from_slice(
                &(std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64)
                    .to_le_bytes(),
            );
            buf[64..64 + frame_size].copy_from_slice(&pixel_data[..frame_size]);
        }

        if let Some(event_handle) = self.frame_ready_event {
            unsafe {
                use windows::Win32::Foundation::HANDLE;
                use windows::Win32::System::Threading::SetEvent;
                let _ = SetEvent(HANDLE(event_handle as *mut _));
            }
        }

        Ok(())
    }

    #[allow(dead_code)]
    fn frame_index(&self) -> u64 {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    pub fn unregister(&mut self) {
        if matches!(self.state, VirtualCameraState::Registered { .. }) {
            info!("virtual-camera: unregistering device");

            #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
            {
                use windows::Win32::Foundation::{CloseHandle, HANDLE};
                use windows::Win32::Media::MediaFoundation::MFShutdown;
                use windows::Win32::System::Com::CoUninitialize;
                use windows::Win32::System::Memory::{UnmapViewOfFile, MEMORY_MAPPED_VIEW_ADDRESS};

                unsafe {
                    if let Some(view) = self.shared_view.take() {
                        let _ = UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view });
                    }
                    if let Some(handle) = self.shared_mapping.take() {
                        let _ = CloseHandle(HANDLE(handle as *mut _));
                    }
                    self.shared_mapping_size = 0;

                    let _ = MFShutdown();

                    if self.com_initialized {
                        CoUninitialize();
                        self.com_initialized = false;
                    }
                }
                self.shared_buffer = None;
                if let Some(event_handle) = self.frame_ready_event.take() {
                    unsafe {
                        let _ = CloseHandle(HANDLE(event_handle as *mut _));
                    }
                }
            }

            self.state = VirtualCameraState::Unregistered;
        }
    }

}

impl Drop for VirtualCameraNative {
    fn drop(&mut self) {
        self.unregister();

        #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
        {
            use windows::Win32::System::Com::CoUninitialize;
            if self.com_initialized {
                unsafe {
                    CoUninitialize();
                }
                self.com_initialized = false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn virtual_camera_config_default() {
        let config = VirtualCameraConfig::default();
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.fps, 60);
    }

    #[test]
    fn virtual_camera_initial_state() {
        let cam = VirtualCameraNative::new(VirtualCameraConfig::default());
        assert!(matches!(cam.state(), VirtualCameraState::Unregistered));
    }

    #[test]
    fn virtual_camera_push_without_register_fails() {
        let mut cam = VirtualCameraNative::new(VirtualCameraConfig::default());
        let result = cam.push_frame(&[], 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn virtual_camera_push_too_small_fails() {
        let mut cam = VirtualCameraNative::new(VirtualCameraConfig::default());
        cam.state = VirtualCameraState::Registered {
            device_path: "test".to_string(),
        };
        let result = cam.push_frame(&[0u8; 10], 1920, 1080);
        assert!(result.is_err());
    }

    #[test]
    fn virtual_camera_register_requires_feature() {
        let mut cam = VirtualCameraNative::new(VirtualCameraConfig::default());
        let result = cam.register();
        #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
        {
            let _ = result;
        }
        #[cfg(not(all(target_os = "windows", feature = "virtual-camera")))]
        assert!(result.is_err());
    }

}
