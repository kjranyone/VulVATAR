//! File-based tracing so we can see what MF/Frame Server is calling when
//! `IMFVirtualCamera::Start` fails. `log` crate infrastructure is not set
//! up in the DLL (it runs inside the Frame Server process, which does not
//! configure `env_logger`), so we append lines directly to a known path.
//!
//! Log location: `C:\Users\Public\vulvatar_mf_camera.log`. Public is chosen
//! because both the user running the main app AND the `LocalService` svchost
//! that loads us inside Frame Server have write access there — the
//! per-user `%TEMP%` would only capture the user-side half.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

static LOG_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

fn log_path() -> PathBuf {
    let mut guard = LOG_PATH.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(p) = guard.as_ref() {
        return p.clone();
    }
    let public = std::env::var_os("PUBLIC")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Users\Public"));
    let mut p = public;
    p.push("vulvatar_mf_camera.log");
    *guard = Some(p.clone());
    p
}

pub fn trace(msg: &str) {
    let path = log_path();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let tid = unsafe { windows::Win32::System::Threading::GetCurrentThreadId() };
    let pid = std::process::id();
    let line = format!("{ts} pid={pid} tid={tid} {msg}\n");
    if let Ok(mut f) = OpenOptions::new().append(true).create(true).open(&path) {
        let _ = f.write_all(line.as_bytes());
        let _ = f.flush();
    }
}

#[macro_export]
macro_rules! t {
    ($($arg:tt)*) => {
        $crate::trace::trace(&format!($($arg)*))
    };
}
