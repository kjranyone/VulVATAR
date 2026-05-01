//! Filesystem watcher for the avatar library auto-import path. Wraps
//! the `notify` crate's `RecommendedWatcher` with per-poll dedup of
//! `(kind, paths)` events and a VRM-extension filter so the GUI's
//! library inspector can react to new `.vrm` files without polling.
//!
//! ## Limitations
//!
//! `notify`'s recommended backend is **local-filesystem-only by
//! design**. The following cases are **best-effort** and should not
//! be relied on:
//!
//! * **UNC / SMB shares** (`\\server\share\...`) — file events from
//!   the server side may not be delivered at all on Windows. A copy
//!   landing on a network drive can sit there until a manual refresh
//!   pokes the library.
//! * **OneDrive / Google Drive / Dropbox folders** — cloud-sync
//!   clients perform a "place-holder swap" (the file appears in
//!   stages: tombstone → real bytes), and the events arrive in
//!   bursts that the dedup pass collapses into a single event when
//!   they share a `(kind, paths)` tuple. The library may end up with
//!   the file recorded but a thumbnail still pointing at zero bytes.
//! * **Symlinks** (NTFS junctions, POSIX symlinks) — whether events
//!   on the link target propagate through the link is platform- and
//!   filesystem-dependent. Don't assume both sides see changes.
//!
//! For these cases, the GUI exposes a manual **Refresh** action
//! (`GuiApp::refresh_watched_folders`) that walks every watched
//! directory and imports any `.vrm` files not yet in the library.
//! Operators on cloud-synced setups should prefer that as a routine,
//! not as a recovery step.
use log::warn;
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct FolderWatchEvent {
    pub kind: FolderWatchEventKind,
    pub paths: Vec<PathBuf>,
    pub timestamp: Instant,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FolderWatchEventKind {
    Created,
    Modified,
    Deleted,
    Other,
}

impl From<&EventKind> for FolderWatchEventKind {
    fn from(kind: &EventKind) -> Self {
        match kind {
            EventKind::Create(_) => FolderWatchEventKind::Created,
            EventKind::Modify(_) => FolderWatchEventKind::Modified,
            EventKind::Remove(_) => FolderWatchEventKind::Deleted,
            _ => FolderWatchEventKind::Other,
        }
    }
}

pub struct FolderWatcher {
    watcher: RecommendedWatcher,
    rx: Receiver<FolderWatchEvent>,
    watched_paths: Vec<PathBuf>,
    last_events: Vec<FolderWatchEvent>,
}

impl FolderWatcher {
    pub fn new() -> Self {
        Self::with_debounce(Duration::from_millis(100))
    }

    /// `_debounce` is currently a no-op — the watcher emits events as
    /// they arrive and only dedups by `(kind, paths)` in `poll()`. The
    /// parameter is preserved on the public API so callers (and the
    /// existing test) keep compiling; if a real time-window debounce
    /// is added later it should consume this argument.
    pub fn with_debounce(_debounce: Duration) -> Self {
        let (tx, rx): (SyncSender<FolderWatchEvent>, Receiver<FolderWatchEvent>) =
            sync_channel(256);

        let watcher = RecommendedWatcher::new(
            move |res: Result<Event, notify::Error>| match res {
                Ok(event) => {
                    let fw_event = FolderWatchEvent {
                        kind: FolderWatchEventKind::from(&event.kind),
                        paths: event.paths,
                        timestamp: Instant::now(),
                    };
                    let _ = tx.send(fw_event);
                }
                Err(e) => {
                    warn!("folder_watcher: watch error: {}", e);
                }
            },
            Config::default(),
        )
        .expect("folder_watcher: failed to create watcher");

        Self {
            watcher,
            rx,
            watched_paths: Vec::new(),
            last_events: Vec::new(),
        }
    }

    pub fn watch(&mut self, path: &Path, recursive: bool) -> Result<(), String> {
        if self.watched_paths.iter().any(|p| p == path) {
            return Ok(());
        }

        let mode = if recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };

        self.watcher
            .watch(path, mode)
            .map_err(|e| format!("failed to watch '{}': {}", path.display(), e))?;

        self.watched_paths.push(path.to_path_buf());
        Ok(())
    }

    pub fn unwatch(&mut self, path: &Path) -> Result<(), String> {
        self.watcher
            .unwatch(path)
            .map_err(|e| format!("failed to unwatch '{}': {}", path.display(), e))?;

        self.watched_paths.retain(|p| p != path);
        Ok(())
    }

    pub fn poll(&mut self) -> &[FolderWatchEvent] {
        self.last_events.clear();

        while let Ok(event) = self.rx.try_recv() {
            self.last_events.push(event);
        }

        let mut seen: HashSet<(FolderWatchEventKind, Vec<PathBuf>)> = HashSet::new();
        self.last_events.retain(|e| {
            let key = (e.kind.clone(), e.paths.clone());
            seen.insert(key)
        });

        &self.last_events
    }

    pub fn watched_paths(&self) -> &[PathBuf] {
        &self.watched_paths
    }

    pub fn filter_vrm(events: &[FolderWatchEvent]) -> Vec<&FolderWatchEvent> {
        events
            .iter()
            .filter(|e| {
                e.paths.iter().any(|p| {
                    p.extension()
                        .and_then(|ext| ext.to_str())
                        .map(|ext| ext.eq_ignore_ascii_case("vrm"))
                        .unwrap_or(false)
                })
            })
            .collect()
    }
}

impl Default for FolderWatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn folder_watcher_default_creates() {
        let _watcher = FolderWatcher::new();
    }

    #[test]
    fn folder_watcher_watch_invalid_path() {
        let mut watcher = FolderWatcher::new();
        let result = watcher.watch(Path::new("/nonexistent/path/that/does/not/exist"), false);
        assert!(result.is_err());
    }

    #[test]
    fn folder_watcher_poll_empty() {
        let mut watcher = FolderWatcher::new();
        let events = watcher.poll();
        assert!(events.is_empty());
    }

    #[test]
    fn folder_watcher_watch_valid_dir() {
        let dir = std::env::temp_dir().join("vulvatar_folder_watcher_test");
        let _ = fs::create_dir_all(&dir);

        let mut watcher = FolderWatcher::new();
        let result = watcher.watch(&dir, false);
        assert!(result.is_ok());
        assert!(watcher.watched_paths().contains(&dir));

        let _ = watcher.unwatch(&dir);
        assert!(!watcher.watched_paths().contains(&dir));

        let _ = fs::remove_dir(&dir);
    }

    #[test]
    #[ignore]
    fn folder_watcher_detects_creation() {
        let dir = std::env::temp_dir().join("vulvatar_folder_watcher_create");
        let _ = fs::create_dir_all(&dir);

        let mut watcher = FolderWatcher::with_debounce(Duration::from_millis(50));
        let result = watcher.watch(&dir, false);
        assert!(result.is_ok());

        std::thread::sleep(Duration::from_millis(100));

        let test_file = dir.join("test.txt");
        let _ = fs::write(&test_file, b"hello");

        std::thread::sleep(Duration::from_millis(500));

        let events = watcher.poll();
        let any_event = !events.is_empty();
        assert!(any_event, "expected at least one event after file creation");

        let _ = fs::remove_file(&test_file);
        let _ = watcher.unwatch(&dir);
        let _ = fs::remove_dir(&dir);
    }

    #[test]
    fn filter_vrm_selects_vrm_files() {
        let events = vec![FolderWatchEvent {
            kind: FolderWatchEventKind::Created,
            paths: vec![PathBuf::from("avatar.vrm"), PathBuf::from("data.json")],
            timestamp: Instant::now(),
        }];
        let filtered = FolderWatcher::filter_vrm(&events);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn filter_vrm_no_match() {
        let events = vec![FolderWatchEvent {
            kind: FolderWatchEventKind::Created,
            paths: vec![PathBuf::from("data.json")],
            timestamp: Instant::now(),
        }];
        let filtered = FolderWatcher::filter_vrm(&events);
        assert!(filtered.is_empty());
    }
}
