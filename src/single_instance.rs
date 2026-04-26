use std::fs;
use std::io;
use std::path::PathBuf;

#[cfg(target_os = "windows")]
use windows::Win32::Foundation::BOOL;
#[cfg(target_os = "windows")]
use windows::Win32::System::Threading::{
    CreateMutexW, GetExitCodeProcess, OpenProcess, QueryFullProcessImageNameW, TerminateProcess,
    PROCESS_NAME_FORMAT, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_TERMINATE,
};

const MUTEX_NAME: &str = "VulVATAR_SingleInstance";
const PID_FILE: &str = "instance.pid";

#[cfg(target_os = "windows")]
fn mutex_name() -> windows::core::HSTRING {
    windows::core::HSTRING::from(MUTEX_NAME)
}

fn pid_path() -> PathBuf {
    let mut p = std::env::var("APPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    p.push("VulVATAR");
    let _ = fs::create_dir_all(&p);
    p.push(PID_FILE);
    p
}

fn write_pid() -> io::Result<()> {
    fs::write(pid_path(), std::process::id().to_string())
}

fn read_pid() -> Option<u32> {
    fs::read_to_string(pid_path())
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
}

#[cfg(target_os = "windows")]
fn is_process_alive(pid: u32) -> bool {
    unsafe {
        let Ok(h) = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, BOOL(0), pid) else {
            return false;
        };
        let mut exit_code: u32 = 0;
        let _ = GetExitCodeProcess(h, &mut exit_code);
        let _ = windows::Win32::Foundation::CloseHandle(h);
        exit_code == 259
    }
}

#[cfg(target_os = "windows")]
fn is_vulvatar_process(pid: u32) -> bool {
    unsafe {
        let Ok(h) = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, BOOL(0), pid) else {
            return false;
        };
        let mut buf = [0u16; 512];
        let mut size = buf.len() as u32;
        let ok = QueryFullProcessImageNameW(
            h,
            PROCESS_NAME_FORMAT(0),
            windows::core::PWSTR(buf.as_mut_ptr()),
            &mut size,
        );
        let _ = windows::Win32::Foundation::CloseHandle(h);
        if ok.is_err() || size == 0 {
            return false;
        }
        let image_name = String::from_utf16_lossy(&buf[..size as usize]);
        image_name.to_lowercase().contains("vulvatar")
    }
}

#[cfg(target_os = "windows")]
fn kill_process(pid: u32) -> bool {
    unsafe {
        let Ok(h) = OpenProcess(PROCESS_TERMINATE, BOOL(0), pid) else {
            return false;
        };
        let result = TerminateProcess(h, 1);
        let _ = windows::Win32::Foundation::CloseHandle(h);
        result.is_ok()
    }
}

#[cfg(target_os = "windows")]
fn try_create_mutex() -> Option<(windows::Win32::Foundation::HANDLE, bool)> {
    let mutex = mutex_name();
    unsafe {
        match CreateMutexW(None, BOOL(1), &mutex) {
            Ok(h) => {
                let err = windows::Win32::Foundation::GetLastError();
                Some((h, err.0 == 183u32))
            }
            Err(_) => None,
        }
    }
}

#[derive(Debug)]
pub struct SingleInstanceGuard {
    #[cfg(target_os = "windows")]
    _handle: windows::Win32::Foundation::HANDLE,
}

#[derive(Debug)]
pub enum SingleInstanceResult {
    First(SingleInstanceGuard),
    ExistingAlive { pid: u32 },
    ExistingDead { pid: u32 },
}

#[cfg(target_os = "windows")]
pub fn acquire() -> SingleInstanceResult {
    if let Some((handle, false)) = try_create_mutex() {
        let _ = write_pid();
        return SingleInstanceResult::First(SingleInstanceGuard { _handle: handle });
    }

    let pid = read_pid().unwrap_or(0);
    if pid != 0 && is_process_alive(pid) && is_vulvatar_process(pid) {
        return SingleInstanceResult::ExistingAlive { pid };
    }

    for _ in 0..10 {
        std::thread::sleep(std::time::Duration::from_millis(200));
        if let Some((handle, false)) = try_create_mutex() {
            let _ = write_pid();
            return SingleInstanceResult::First(SingleInstanceGuard { _handle: handle });
        }
    }

    SingleInstanceResult::ExistingDead { pid }
}

#[cfg(target_os = "windows")]
pub fn force_kill_and_acquire(pid: u32) -> Option<SingleInstanceGuard> {
    if !is_vulvatar_process(pid) {
        return None;
    }
    if !kill_process(pid) {
        return None;
    }

    for _ in 0..10 {
        std::thread::sleep(std::time::Duration::from_millis(200));
        if let Some((handle, false)) = try_create_mutex() {
            let _ = write_pid();
            return Some(SingleInstanceGuard { _handle: handle });
        }
    }
    None
}

#[cfg(not(target_os = "windows"))]
pub fn acquire() -> SingleInstanceResult {
    if let Some(pid) = read_pid() {
        if pid != std::process::id() {
            if fs::exists(format!("/proc/{}", pid)).unwrap_or(false) {
                return SingleInstanceResult::ExistingAlive { pid };
            }
        }
    }
    let _ = write_pid();
    SingleInstanceResult::First(SingleInstanceGuard {})
}

#[cfg(not(target_os = "windows"))]
pub fn force_kill_and_acquire(_pid: u32) -> Option<SingleInstanceGuard> {
    None
}
