//! Crash-forensics stage log + unclean-exit sentinel.
//!
//! Motivation: a 2026-06-11 incident hard-froze the whole machine
//! during live tracking (Kernel-Power 41, no TDR event flushed) and
//! left zero evidence of which pipeline stage was in flight. This
//! module makes the *next* freeze diagnosable without asking the user
//! to set up logging:
//!
//! * **Stage log** — while a tracking session is active, every stage
//!   boundary (camera grab, YOLOX submit, RTMW3D run, depth wait,
//!   publish) appends one line to `logs/tracking_<unix-secs>.log` and
//!   flushes immediately. After a hard freeze the last line names the
//!   stage that never completed.
//! * **Sentinel** — `logs/tracking.active` is created when a session
//!   begins and removed when it ends (including panic unwinds via
//!   [`SessionGuard`]). If it exists at startup, the previous session
//!   died uncleanly — the GUI uses this to offer a degraded safe-mode
//!   configuration.
//!
//! Cost when inactive: one relaxed atomic load per [`mark`] call.
//! Cost when active: one mutex lock + small `write` + `flush` per
//! stage (~6 per frame at 30 fps — microseconds against a 33 ms
//! budget, and worth it: an unflushed buffer is exactly what a hard
//! freeze destroys).

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use log::warn;

const LOG_DIR: &str = "logs";
const SENTINEL_NAME: &str = "tracking.active";
/// Keep at most this many session logs; oldest are deleted on session
/// start so the directory never grows unbounded.
const MAX_SESSION_LOGS: usize = 10;

static ACTIVE: AtomicBool = AtomicBool::new(false);
static STATE: Mutex<Option<SessionState>> = Mutex::new(None);

struct SessionState {
    file: File,
    started: Instant,
}

fn log_dir() -> PathBuf {
    PathBuf::from(LOG_DIR)
}

fn sentinel_path() -> PathBuf {
    log_dir().join(SENTINEL_NAME)
}

/// Begin a tracking-session stage log. Creates `logs/`, rotates old
/// session logs, writes the sentinel, and activates [`mark`]. Errors
/// are logged and swallowed — forensics must never block tracking.
pub fn begin_session(session_label: &str) {
    let dir = log_dir();
    if let Err(e) = fs::create_dir_all(&dir) {
        warn!("stagelog: cannot create {}: {e}", dir.display());
        return;
    }
    rotate_old_logs(&dir);

    let unix_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let log_path = dir.join(format!("tracking_{unix_secs}.log"));
    let mut file = match OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
    {
        Ok(f) => f,
        Err(e) => {
            warn!("stagelog: cannot open {}: {e}", log_path.display());
            return;
        }
    };
    let _ = writeln!(file, "session_begin {session_label}");
    let _ = file.flush();

    // Sentinel carries the session log filename so the safe-mode
    // banner can point the user (and a future bug report) at the
    // right file.
    if let Err(e) = fs::write(
        sentinel_path(),
        format!("{}\n{session_label}\n", log_path.display()),
    ) {
        warn!("stagelog: cannot write sentinel: {e}");
    }

    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    *state = Some(SessionState {
        file,
        started: Instant::now(),
    });
    drop(state);
    ACTIVE.store(true, Ordering::Release);
}

/// End the session cleanly: final marker, deactivate, remove the
/// sentinel. Safe to call when no session is active.
pub fn end_session() {
    ACTIVE.store(false, Ordering::Release);
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(mut s) = state.take() {
        let ms = s.started.elapsed().as_secs_f64() * 1000.0;
        let _ = writeln!(s.file, "{ms:>10.1} session_end");
        let _ = s.file.flush();
    }
    drop(state);
    if let Err(e) = fs::remove_file(sentinel_path()) {
        if e.kind() != std::io::ErrorKind::NotFound {
            warn!("stagelog: cannot remove sentinel: {e}");
        }
    }
}

/// Append one stage marker and flush. No-op (one atomic load) when no
/// session is active, so provider code can call this unconditionally.
pub fn mark(frame: u64, stage: &str) {
    if !ACTIVE.load(Ordering::Acquire) {
        return;
    }
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = state.as_mut() {
        let ms = s.started.elapsed().as_secs_f64() * 1000.0;
        let _ = writeln!(s.file, "{ms:>10.1} f={frame} {stage}");
        let _ = s.file.flush();
    }
}

/// RAII guard: ends the session when dropped, including on panic
/// unwind, so the sentinel only survives a *hard* death (process
/// kill, system freeze) — exactly the cases safe mode targets.
pub struct SessionGuard;

impl SessionGuard {
    pub fn begin(session_label: &str) -> Self {
        begin_session(session_label);
        Self
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        end_session();
    }
}

/// If the previous run died uncleanly, returns the sentinel's content
/// (session log path + label). Does NOT clear the sentinel — the next
/// `begin_session` overwrites it, and the GUI decides when the user
/// has acknowledged it via [`clear_stale_sentinel`].
pub fn stale_sentinel() -> Option<String> {
    fs::read_to_string(sentinel_path()).ok()
}

/// Acknowledge (remove) a stale sentinel without starting a session.
pub fn clear_stale_sentinel() {
    let _ = fs::remove_file(sentinel_path());
}

/// Delete the oldest `tracking_*.log` files beyond the retention cap.
/// Unix-seconds filenames sort chronologically as strings of equal
/// length, but sort by mtime anyway so manual renames don't confuse
/// retention.
fn rotate_old_logs(dir: &std::path::Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut logs: Vec<(SystemTime, PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if !(name.starts_with("tracking_") && name.ends_with(".log")) {
                return None;
            }
            let mtime = e.metadata().ok()?.modified().ok()?;
            Some((mtime, e.path()))
        })
        .collect();
    if logs.len() < MAX_SESSION_LOGS {
        return;
    }
    logs.sort_by_key(|(mtime, _)| *mtime);
    let excess = logs.len() + 1 - MAX_SESSION_LOGS;
    for (_, path) in logs.into_iter().take(excess) {
        if let Err(e) = fs::remove_file(&path) {
            warn!("stagelog: cannot rotate {}: {e}", path.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialise tests that touch the shared global session state.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn sentinel_lifecycle() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        end_session(); // ensure clean slate
        clear_stale_sentinel();
        assert!(stale_sentinel().is_none());

        begin_session("test-session");
        assert!(stale_sentinel().is_some(), "sentinel exists while active");
        mark(0, "unit_test_stage");
        end_session();
        assert!(
            stale_sentinel().is_none(),
            "clean end removes the sentinel"
        );
    }

    #[test]
    fn guard_ends_session_on_drop() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        end_session();
        clear_stale_sentinel();
        {
            let _session = SessionGuard::begin("guard-test");
            assert!(stale_sentinel().is_some());
        }
        assert!(stale_sentinel().is_none());
    }
}
