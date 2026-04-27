#![allow(dead_code, unused_imports)]
use log::info;
#[cfg(all(target_os = "windows", feature = "virtual-camera"))]
use log::info as _;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FilterRegistrationState {
    Unregistered,
    Registered {
        clsid_str: String,
        friendly_name: String,
    },
    Failed(String),
}

#[derive(Clone, Debug)]
pub struct DirectShowFilterConfig {
    pub clsid: &'static str,
    pub friendly_name: String,
    pub device_path: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

impl Default for DirectShowFilterConfig {
    fn default() -> Self {
        Self {
            clsid: "{D7A3C420-5E91-4F8A-ABCD-52C1E7D9A4F3}",
            friendly_name: "VulVATAR Virtual Camera".to_string(),
            device_path: "\\\\.\\VulVATAR_VirtualCamera".to_string(),
            width: 1920,
            height: 1080,
            fps: 60,
        }
    }
}

pub struct DirectShowFilterRegistrar {
    config: DirectShowFilterConfig,
    state: FilterRegistrationState,
}

impl DirectShowFilterRegistrar {
    pub fn new(config: DirectShowFilterConfig) -> Self {
        Self {
            config,
            state: FilterRegistrationState::Unregistered,
        }
    }

    pub fn state(&self) -> &FilterRegistrationState {
        &self.state
    }

    pub fn register(&mut self) -> Result<(), String> {
        #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
        {
            self.register_win32()
        }

        #[cfg(not(all(target_os = "windows", feature = "virtual-camera")))]
        {
            let msg =
                "DirectShow registration requires Windows + `virtual-camera` feature".to_string();
            self.state = FilterRegistrationState::Failed(msg.clone());
            Err(msg)
        }
    }

    pub fn unregister(&mut self) -> Result<(), String> {
        #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
        {
            self.unregister_win32()
        }

        #[cfg(not(all(target_os = "windows", feature = "virtual-camera")))]
        {
            Err("DirectShow unregistration requires Windows + `virtual-camera` feature".to_string())
        }
    }

    pub fn is_registered(&self) -> bool {
        matches!(self.state, FilterRegistrationState::Registered { .. })
    }

    pub fn config(&self) -> &DirectShowFilterConfig {
        &self.config
    }

    pub fn supported_media_types(&self) -> Vec<MediaTypeDesc> {
        vec![
            MediaTypeDesc {
                major_type: "MEDIATYPE_Video".to_string(),
                sub_type: "MEDIASUBTYPE_RGB32".to_string(),
                width: self.config.width,
                height: self.config.height,
                fps: self.config.fps,
            },
            MediaTypeDesc {
                major_type: "MEDIATYPE_Video".to_string(),
                sub_type: "MEDIASUBTYPE_ARGB32".to_string(),
                width: self.config.width,
                height: self.config.height,
                fps: self.config.fps,
            },
        ]
    }
}

#[derive(Clone, Debug)]
pub struct MediaTypeDesc {
    pub major_type: String,
    pub sub_type: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

impl std::fmt::Display for MediaTypeDesc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{} {}x{}@{}fps",
            self.major_type, self.sub_type, self.width, self.height, self.fps
        )
    }
}

#[cfg(all(target_os = "windows", feature = "virtual-camera"))]
impl DirectShowFilterRegistrar {
    fn register_win32(&mut self) -> Result<(), String> {
        use windows::core::*;
        use windows::Win32::System::Registry::*;

        info!("directshow: registering filter CLSID={}", self.config.clsid);

        let friendly_name_w: HSTRING = self.config.friendly_name.clone().into();
        let friendly_pcwstr = PCWSTR(friendly_name_w.as_ptr());
        let friendly_byte_len = ((friendly_name_w.len() + 1) * 2) as u32;

        unsafe {
            let mut clsid_key = Default::default();
            let clsid_path = format!("Software\\Classes\\CLSID\\{}", self.config.clsid);
            let clsid_path_w: HSTRING = clsid_path.into();

            let res = RegCreateKeyExW(
                HKEY_LOCAL_MACHINE,
                &clsid_path_w,
                0,
                None,
                REG_OPTION_NON_VOLATILE,
                KEY_WRITE,
                None,
                &mut clsid_key,
                None,
            );
            if res.is_err() {
                let msg = format!("Failed to create CLSID key: {:?}", res);
                self.state = FilterRegistrationState::Failed(msg.clone());
                return Err(msg);
            }

            let _ = RegSetValueW(
                clsid_key,
                PCWSTR::null(),
                REG_SZ,
                friendly_pcwstr,
                friendly_byte_len,
            );

            let mut inproc_key = Default::default();
            let inproc_path = format!(
                "Software\\Classes\\CLSID\\{}\\InprocServer32",
                self.config.clsid
            );
            let inproc_path_w: HSTRING = inproc_path.into();

            let res = RegCreateKeyExW(
                HKEY_LOCAL_MACHINE,
                &inproc_path_w,
                0,
                None,
                REG_OPTION_NON_VOLATILE,
                KEY_WRITE,
                None,
                &mut inproc_key,
                None,
            );
            if res.is_err() {
                let _ = RegCloseKey(clsid_key);
                let msg = format!("Failed to create InprocServer32 key: {:?}", res);
                self.state = FilterRegistrationState::Failed(msg.clone());
                return Err(msg);
            }

            // TODO: resolve full DLL path at runtime instead of hardcoding
            let dll_path_w: HSTRING = "vulvatar_dshow_filter.dll".into();
            let dll_pcwstr = PCWSTR(dll_path_w.as_ptr());
            let dll_byte_len = ((dll_path_w.len() + 1) * 2) as u32;
            let _ = RegSetValueW(inproc_key, PCWSTR::null(), REG_SZ, dll_pcwstr, dll_byte_len);

            let threading_w: HSTRING = "Both".into();
            let threading_name: HSTRING = "ThreadingModel".into();
            let threading_byte_len = ((threading_w.len() + 1) * 2) as u32;
            let threading_bytes = std::slice::from_raw_parts(
                threading_w.as_ptr() as *const u8,
                threading_byte_len as usize,
            );
            let _ = RegSetValueExW(
                inproc_key,
                PCWSTR(threading_name.as_ptr()),
                0,
                REG_SZ,
                Some(threading_bytes),
            );
            let _ = RegCloseKey(inproc_key);

            let mut filter_key = Default::default();
            let filter_path = format!("Software\\Classes\\Filter\\{}", self.config.clsid);
            let filter_path_w: HSTRING = filter_path.into();

            let filter_result = RegCreateKeyExW(
                HKEY_LOCAL_MACHINE,
                &filter_path_w,
                0,
                None,
                REG_OPTION_NON_VOLATILE,
                KEY_WRITE,
                None,
                &mut filter_key,
                None,
            );

            if filter_result.is_ok() {
                let _ = RegSetValueW(
                    filter_key,
                    PCWSTR::null(),
                    REG_SZ,
                    friendly_pcwstr,
                    friendly_byte_len,
                );
                let _ = RegCloseKey(filter_key);
            }

            let _ = RegCloseKey(clsid_key);
        }

        info!(
            "directshow: filter '{}' registered (CLSID={})",
            self.config.friendly_name, self.config.clsid
        );

        self.state = FilterRegistrationState::Registered {
            clsid_str: self.config.clsid.to_string(),
            friendly_name: self.config.friendly_name.clone(),
        };
        Ok(())
    }

    fn unregister_win32(&mut self) -> Result<(), String> {
        use windows::core::*;
        use windows::Win32::System::Registry::*;

        info!(
            "directshow: unregistering filter CLSID={}",
            self.config.clsid
        );

        unsafe {
            let clsid_path = format!("Software\\Classes\\CLSID\\{}", self.config.clsid);
            let clsid_path_w: HSTRING = clsid_path.into();
            let _ = RegDeleteTreeW(HKEY_LOCAL_MACHINE, &clsid_path_w);

            let filter_path = format!("Software\\Classes\\Filter\\{}", self.config.clsid);
            let filter_path_w: HSTRING = filter_path.into();
            let _ = RegDeleteTreeW(HKEY_LOCAL_MACHINE, &filter_path_w);
        }

        info!("directshow: filter unregistered");
        self.state = FilterRegistrationState::Unregistered;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ds_filter_default_config() {
        let config = DirectShowFilterConfig::default();
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.fps, 60);
        assert!(!config.clsid.is_empty());
    }

    #[test]
    fn ds_filter_initial_state() {
        let reg = DirectShowFilterRegistrar::new(DirectShowFilterConfig::default());
        assert!(matches!(reg.state(), FilterRegistrationState::Unregistered));
        assert!(!reg.is_registered());
    }

    #[test]
    fn ds_filter_supported_media_types() {
        let reg = DirectShowFilterRegistrar::new(DirectShowFilterConfig::default());
        let types = reg.supported_media_types();
        assert_eq!(types.len(), 2);
        assert_eq!(types[0].sub_type, "MEDIASUBTYPE_RGB32");
        assert_eq!(types[1].sub_type, "MEDIASUBTYPE_ARGB32");
    }

    #[test]
    fn ds_filter_register_requires_feature() {
        let mut reg = DirectShowFilterRegistrar::new(DirectShowFilterConfig::default());
        let result = reg.register();
        #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
        {
            let _ = result;
        }
        #[cfg(not(all(target_os = "windows", feature = "virtual-camera")))]
        assert!(result.is_err());
    }

    #[test]
    fn ds_filter_media_type_display() {
        let mt = MediaTypeDesc {
            major_type: "MEDIATYPE_Video".to_string(),
            sub_type: "MEDIASUBTYPE_RGB32".to_string(),
            width: 1280,
            height: 720,
            fps: 30,
        };
        let s = format!("{}", mt);
        assert!(s.contains("1280x720@30fps"));
    }
}
