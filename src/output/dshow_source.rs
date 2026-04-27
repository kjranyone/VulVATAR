#![allow(dead_code)]
use log::info;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceFilterState {
    Stopped,
    Running,
    Paused,
}

#[derive(Clone, Debug)]
pub struct SourceFilterConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub output_format: PixelFormat,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    RGB32,
    ARGB32,
}

impl Default for SourceFilterConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            fps: 60,
            output_format: PixelFormat::ARGB32,
        }
    }
}

impl SourceFilterConfig {
    pub fn bytes_per_frame(&self) -> usize {
        let bpp = match self.output_format {
            PixelFormat::RGB32 | PixelFormat::ARGB32 => 4,
        };
        (self.width as usize) * (self.height as usize) * bpp
    }

    pub fn media_subtype(&self) -> &str {
        match self.output_format {
            PixelFormat::RGB32 => "MEDIASUBTYPE_RGB32",
            PixelFormat::ARGB32 => "MEDIASUBTYPE_ARGB32",
        }
    }
}

pub struct DirectShowSourceFilter {
    config: SourceFilterConfig,
    state: SourceFilterState,
    frame_buffer: Vec<u8>,
    frame_index: u64,
    shared_memory_name: String,
}

impl DirectShowSourceFilter {
    pub fn new(config: SourceFilterConfig) -> Self {
        let buffer_size = config.bytes_per_frame();
        Self {
            config,
            state: SourceFilterState::Stopped,
            frame_buffer: vec![0u8; buffer_size],
            frame_index: 0,
            shared_memory_name: "VulVATAR_VirtualCamera".to_string(),
        }
    }

    pub fn state(&self) -> &SourceFilterState {
        &self.state
    }

    pub fn config(&self) -> &SourceFilterConfig {
        &self.config
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    pub fn shared_memory_name(&self) -> &str {
        &self.shared_memory_name
    }

    pub fn start(&mut self) -> Result<(), String> {
        if matches!(self.state, SourceFilterState::Running) {
            return Ok(());
        }

        #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
        {
            use windows::Win32::Media::MediaFoundation::*;
            unsafe {
                MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET)
                    .map_err(|e| format!("DirectShow source: MFStartup failed: {:?}", e))?;
            }
        }

        self.state = SourceFilterState::Running;
        info!("dshow_source: filter started");
        Ok(())
    }

    pub fn stop(&mut self) -> Result<(), String> {
        if matches!(self.state, SourceFilterState::Stopped) {
            return Ok(());
        }

        #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
        {
            use windows::Win32::Media::MediaFoundation::*;
            unsafe {
                let _ = MFShutdown();
            }
        }

        self.state = SourceFilterState::Stopped;
        self.frame_index = 0;
        info!("dshow_source: filter stopped");
        Ok(())
    }

    pub fn pause(&mut self) -> Result<(), String> {
        if !matches!(self.state, SourceFilterState::Running) {
            return Err("filter not running".to_string());
        }
        self.state = SourceFilterState::Paused;
        info!("dshow_source: filter paused");
        Ok(())
    }

    pub fn resume(&mut self) -> Result<(), String> {
        if !matches!(self.state, SourceFilterState::Paused) {
            return Err("filter not paused".to_string());
        }
        self.state = SourceFilterState::Running;
        info!("dshow_source: filter resumed");
        Ok(())
    }

    pub fn push_frame(&mut self, rgba_data: &[u8]) -> Result<(), String> {
        if !matches!(self.state, SourceFilterState::Running) {
            return Err("filter not running".to_string());
        }

        let expected = self.config.bytes_per_frame();
        if rgba_data.len() < expected {
            return Err(format!(
                "frame too small: got {} bytes, expected {}",
                rgba_data.len(),
                expected
            ));
        }

        self.frame_buffer.copy_from_slice(&rgba_data[..expected]);
        self.frame_index += 1;

        self.write_to_shared_memory()?;

        Ok(())
    }

    pub fn deliver_sample(&self) -> Result<DeliveredSample, String> {
        if self.frame_buffer.is_empty() {
            return Err("no frame available".to_string());
        }

        Ok(DeliveredSample {
            width: self.config.width,
            height: self.config.height,
            format: self.config.output_format.clone(),
            frame_index: self.frame_index,
            data_size: self.frame_buffer.len(),
        })
    }

    fn write_to_shared_memory(&self) -> Result<(), String> {
        #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
        {
            use windows::core::*;
            use windows::Win32::Foundation::*;
            use windows::Win32::System::Memory::*;

            let name: HSTRING = self.shared_memory_name.clone().into();
            let buf_size = self.frame_buffer.len() as u64 + 4096;

            // Pass a permissive DACL so the interactive-user producer
            // (vulvatar.exe) can still write the section if this DLL —
            // hosted in a consumer process under a different token — wins
            // the CreateFileMappingW race. See shmem_security.rs.
            let security =
                super::shmem_security::PermissiveSharedMemorySecurity::new()?;

            unsafe {
                let handle = CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    Some(security.as_win_ptr()),
                    PAGE_READWRITE,
                    (buf_size >> 32) as u32,
                    buf_size as u32,
                    &name,
                )
                .map_err(|e| format!("CreateFileMappingW failed: {:?}", e))?;

                let ptr = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
                if ptr.Value.is_null() {
                    let _ = CloseHandle(handle);
                    return Err("MapViewOfFile failed: null pointer".to_string());
                }

                let dst = ptr.Value as *mut u8;

                let header_size = 32usize;
                let magic = b"VVC0";
                std::ptr::copy_nonoverlapping(magic.as_ptr(), dst, 4);

                let w = self.config.width;
                let h = self.config.height;
                std::ptr::copy_nonoverlapping(&w as *const u32 as *const u8, dst.add(4), 4);
                std::ptr::copy_nonoverlapping(&h as *const u32 as *const u8, dst.add(8), 4);
                let ts = self.frame_index;
                std::ptr::copy_nonoverlapping(&ts as *const u64 as *const u8, dst.add(12), 8);
                let fi = self.frame_index;
                std::ptr::copy_nonoverlapping(&fi as *const u64 as *const u8, dst.add(20), 8);

                std::ptr::copy_nonoverlapping(
                    self.frame_buffer.as_ptr(),
                    dst.add(header_size),
                    self.frame_buffer.len(),
                );

                let _ = UnmapViewOfFile(ptr);
                let _ = CloseHandle(handle);
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct DeliveredSample {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub frame_index: u64,
    pub data_size: usize,
}

impl Drop for DirectShowSourceFilter {
    fn drop(&mut self) {
        if !matches!(self.state, SourceFilterState::Stopped) {
            let _ = self.stop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_filter_default_config() {
        let config = SourceFilterConfig::default();
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.fps, 60);
        assert_eq!(config.output_format, PixelFormat::ARGB32);
    }

    #[test]
    fn source_filter_bytes_per_frame() {
        let config = SourceFilterConfig::default();
        assert_eq!(config.bytes_per_frame(), 1920 * 1080 * 4);
    }

    #[test]
    fn source_filter_initial_state() {
        let filter = DirectShowSourceFilter::new(SourceFilterConfig::default());
        assert!(matches!(filter.state(), SourceFilterState::Stopped));
        assert_eq!(filter.frame_index(), 0);
    }

    #[test]
    fn source_filter_start_stop() {
        let mut filter = DirectShowSourceFilter::new(SourceFilterConfig::default());
        filter.start().unwrap();
        assert!(matches!(filter.state(), SourceFilterState::Running));
        filter.stop().unwrap();
        assert!(matches!(filter.state(), SourceFilterState::Stopped));
        assert_eq!(filter.frame_index(), 0);
    }

    #[test]
    fn source_filter_pause_resume() {
        let mut filter = DirectShowSourceFilter::new(SourceFilterConfig::default());
        filter.start().unwrap();
        filter.pause().unwrap();
        assert!(matches!(filter.state(), SourceFilterState::Paused));
        filter.resume().unwrap();
        assert!(matches!(filter.state(), SourceFilterState::Running));
        filter.stop().unwrap();
    }

    #[test]
    fn source_filter_push_when_stopped_fails() {
        let mut filter = DirectShowSourceFilter::new(SourceFilterConfig::default());
        let data = vec![0u8; filter.config().bytes_per_frame()];
        let result = filter.push_frame(&data);
        assert!(result.is_err());
    }

    #[test]
    fn source_filter_push_too_small_fails() {
        let mut filter = DirectShowSourceFilter::new(SourceFilterConfig::default());
        filter.start().unwrap();
        let result = filter.push_frame(&[0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn source_filter_media_subtype() {
        let config = SourceFilterConfig {
            output_format: PixelFormat::RGB32,
            ..Default::default()
        };
        assert_eq!(config.media_subtype(), "MEDIASUBTYPE_RGB32");
    }

    #[test]
    fn source_filter_custom_config() {
        let config = SourceFilterConfig {
            width: 1280,
            height: 720,
            fps: 30,
            output_format: PixelFormat::RGB32,
        };
        let filter = DirectShowSourceFilter::new(config);
        assert_eq!(filter.config().width, 1280);
        assert_eq!(filter.config().bytes_per_frame(), 1280 * 720 * 4);
    }

    #[test]
    fn source_filter_deliver_sample_initial() {
        let filter = DirectShowSourceFilter::new(SourceFilterConfig::default());
        let sample = filter.deliver_sample().unwrap();
        assert_eq!(sample.width, 1920);
        assert_eq!(sample.height, 1080);
        assert_eq!(sample.frame_index, 0);
        assert_eq!(sample.data_size, 1920 * 1080 * 4);
    }

    #[test]
    fn source_filter_double_start_ok() {
        let mut filter = DirectShowSourceFilter::new(SourceFilterConfig::default());
        filter.start().unwrap();
        filter.start().unwrap();
        assert!(matches!(filter.state(), SourceFilterState::Running));
        filter.stop().unwrap();
    }
}
