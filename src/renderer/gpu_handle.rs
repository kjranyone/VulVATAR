#![allow(dead_code)]
use ash::vk;
use log::{info, warn};
use std::sync::Arc;
use vulkano::device::Device;
use vulkano::image::{Image, ImageMemory};
use vulkano::memory::{DeviceMemory, ExternalMemoryHandleType};
use vulkano::VulkanObject;

#[derive(Clone, Debug)]
pub enum SharedHandle {
    None,
    Win32Kmt { handle: u64 },
    NamedSharedMemory { name: String, size: usize },
}

#[derive(Clone, Debug)]
pub struct GpuSharedFrame {
    pub gpu_image: Arc<Image>,
    pub shared_handle: SharedHandle,
    pub extent: [u32; 2],
    pub frame_index: u64,
    pub fence_value: u64,
}

pub struct GpuHandleExporter {
    supported: bool,
    external_memory_detected: bool,
    frame_index: u64,
    fence_counter: u64,
    device: Option<Arc<Device>>,
}

impl Default for GpuHandleExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuHandleExporter {
    pub fn new() -> Self {
        Self {
            supported: false,
            external_memory_detected: false,
            frame_index: 0,
            fence_counter: 0,
            device: None,
        }
    }

    pub fn set_device(&mut self, device: Arc<Device>) {
        if self.device.is_none() {
            self.device = Some(device);
        }
    }

    pub fn detect_support(&mut self) -> bool {
        if self.external_memory_detected {
            return self.supported;
        }
        self.external_memory_detected = true;

        if let Some(ref device) = self.device {
            let physical_device = device.physical_device();
            let supported_exts = physical_device.supported_extensions();
            let enabled_exts = device.enabled_extensions();

            #[cfg(target_os = "windows")]
            {
                let base_available = supported_exts.khr_external_memory;
                let base_enabled = enabled_exts.khr_external_memory;
                let win32_available = supported_exts.khr_external_memory_win32;
                let win32_enabled = enabled_exts.khr_external_memory_win32;

                if base_enabled && win32_enabled {
                    self.supported = true;
                    info!(
                        "gpu-handle: VK_KHR_external_memory_win32 enabled, \
                         true GPU sharing available"
                    );
                } else {
                    info!(
                        "gpu-handle: VK_KHR_external_memory_win32 not enabled \
                         (base: available={}, enabled={}, win32: available={}, enabled={}), \
                         falling back to shared memory",
                        base_available, base_enabled, win32_available, win32_enabled
                    );
                    self.supported = false;
                }
            }

            #[cfg(not(target_os = "windows"))]
            {
                let _ = supported_exts;
                let _ = enabled_exts;
                info!("gpu-handle: platform does not support Win32 external memory");
                self.supported = false;
            }
        } else {
            info!("gpu-handle: no device set, cannot detect external memory support");
            self.supported = false;
        }

        self.supported
    }

    pub fn is_supported(&self) -> bool {
        self.supported
    }

    pub fn export_frame(&mut self, gpu_image: Arc<Image>) -> GpuSharedFrame {
        if !self.external_memory_detected {
            self.detect_support();
        }

        let extent = gpu_image.extent();
        let frame_index = self.frame_index;
        self.frame_index += 1;

        let shared_handle = if self.supported {
            match self.export_win32_kmt_handle(&gpu_image) {
                Ok(handle) => SharedHandle::Win32Kmt { handle },
                Err(e) => {
                    warn!("gpu-handle: Win32 KMT export failed: {}, falling back", e);
                    self.create_fallback_handle(extent[0], extent[1], frame_index)
                }
            }
        } else {
            self.create_fallback_handle(extent[0], extent[1], frame_index)
        };

        let fence_value = self.fence_counter;
        self.fence_counter += 1;

        GpuSharedFrame {
            gpu_image,
            shared_handle,
            extent: [extent[0], extent[1]],
            frame_index,
            fence_value,
        }
    }

    #[cfg(target_os = "windows")]
    fn export_win32_kmt_handle(&self, gpu_image: &Arc<Image>) -> Result<u64, String> {
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| "no device set".to_string())?;

        let resource_memory = match gpu_image.memory() {
            ImageMemory::Normal(resources) => resources
                .first()
                .ok_or_else(|| "image has no memory bound".to_string())?,
            _ => return Err("image memory is not normal (sparse/swapchain)".to_string()),
        };

        let device_memory: &Arc<DeviceMemory> = resource_memory.device_memory();

        if !device_memory
            .export_handle_types()
            .contains_enum(ExternalMemoryHandleType::OpaqueWin32Kmt)
        {
            return Err(
                "image memory was not allocated with OPAQUE_WIN32_KMT export capability"
                    .to_string(),
            );
        }

        let raw_memory = device_memory.handle();

        let info = vk::MemoryGetWin32HandleInfoKHR::default()
            .memory(raw_memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KMT);

        let fns = device.fns();
        let mut handle: vk::HANDLE = 0;

        let result = unsafe {
            (fns.khr_external_memory_win32.get_memory_win32_handle_khr)(
                device.handle(),
                &info,
                &mut handle,
            )
        };

        match result {
            vk::Result::SUCCESS => Ok(handle as u64),
            err => Err(format!("vkGetMemoryWin32HandleKHR failed: {:?}", err)),
        }
    }

    #[cfg(not(target_os = "windows"))]
    fn export_win32_kmt_handle(&self, _gpu_image: &Arc<Image>) -> Result<u64, String> {
        Err("Win32 KMT handle export not supported on this platform".to_string())
    }

    fn create_fallback_handle(&self, width: u32, height: u32, frame_index: u64) -> SharedHandle {
        #[cfg(target_os = "windows")]
        {
            let name = format!("VulVATAR_Frame_{}", frame_index);
            let size = (width as usize) * (height as usize) * 4;
            SharedHandle::NamedSharedMemory { name, size }
        }

        #[cfg(not(target_os = "windows"))]
        {
            let _ = (width, height, frame_index);
            SharedHandle::None
        }
    }

    pub fn export_frame_to_shared_memory(
        &self,
        pixel_data: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> Result<String, String> {
        #[cfg(target_os = "windows")]
        {
            let name = format!("VulVATAR_Frame_{}", frame_index);
            let path = std::env::temp_dir().join(format!("{}.bin", name));

            use std::io::Write;
            let mut file = std::fs::File::create(&path)
                .map_err(|e| format!("failed to create shared memory file: {}", e))?;

            file.write_all(b"VVF\0")
                .map_err(|e| format!("failed to write magic: {}", e))?;
            file.write_all(&width.to_le_bytes())
                .map_err(|e| format!("failed to write width: {}", e))?;
            file.write_all(&height.to_le_bytes())
                .map_err(|e| format!("failed to write height: {}", e))?;
            file.write_all(&frame_index.to_le_bytes())
                .map_err(|e| format!("failed to write frame index: {}", e))?;
            file.write_all(pixel_data)
                .map_err(|e| format!("failed to write pixel data: {}", e))?;

            Ok(format!("{}", path.display()))
        }

        #[cfg(not(target_os = "windows"))]
        {
            let _ = (pixel_data, width, height, frame_index);
            Err("shared memory export not supported on this platform".to_string())
        }
    }
}
