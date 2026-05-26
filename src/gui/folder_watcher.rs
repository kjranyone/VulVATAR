//! GUI-side wrappers around `crate::app::folder_watcher`. Manages
//! the lifecycle of the watcher, drains its events into avatar-
//! library imports, and exposes the manual-Refresh escape hatch
//! used when the notify backend silently misses events on cloud-
//! sync / network shares.
//!
//! See `app/folder_watcher.rs` for the underlying notify-based
//! watcher and the rationale for the manual Refresh path.

use log::warn;

use crate::t;

use super::GuiApp;

impl GuiApp {
    pub fn start_watching_folder(&mut self, path: std::path::PathBuf) -> Result<(), String> {
        if self.library.watched_avatar_dirs.iter().any(|p| p == &path) {
            return Ok(()); // idempotent
        }
        if self.library.folder_watcher.is_none() {
            self.library.folder_watcher = Some(crate::app::folder_watcher::FolderWatcher::new());
        }

        if let Some(ref mut fw) = self.library.folder_watcher {
            fw.watch(&path, true)?;
            self.library.watched_avatar_dirs.push(path.clone());
            self.push_notification(t!("toast.watched_folder", path = path.display().to_string()));
            let _ = crate::persistence::save_watched_folders(&self.library.watched_avatar_dirs);
        }
        Ok(())
    }

    pub fn stop_watching_folder(&mut self, path: &std::path::Path) -> Result<(), String> {
        if let Some(ref mut fw) = self.library.folder_watcher {
            fw.unwatch(path)?;
            self.library.watched_avatar_dirs.retain(|p| p != path);
            self.push_notification(t!("toast.stopped_watching", path = path.display().to_string()));
            let _ = crate::persistence::save_watched_folders(&self.library.watched_avatar_dirs);
            if self.library.watched_avatar_dirs.is_empty() {
                self.library.folder_watcher = None;
            }
        }
        Ok(())
    }

    pub(super) fn poll_folder_watcher(&mut self) {
        let Some(ref mut fw) = self.library.folder_watcher else {
            return;
        };
        let events = fw.poll().to_vec();
        let vrm_events = crate::app::folder_watcher::FolderWatcher::filter_vrm(&events);
        let mut newly_added: Vec<std::path::PathBuf> = Vec::new();
        for event in vrm_events {
            if event.kind != crate::app::folder_watcher::FolderWatchEventKind::Created {
                continue;
            }
            for vrm_path in &event.paths {
                if self.import_vrm_into_library_if_missing(vrm_path) {
                    newly_added.push(vrm_path.clone());
                }
            }
        }
        for vrm_path in &newly_added {
            self.push_notification(t!("toast.new_vrm_detected", path = vrm_path.display().to_string()));
        }
        if !newly_added.is_empty() && !self.test_no_persist {
            self.save_avatar_library_with_toast();
        }
    }

    /// Walk every watched directory and import any `.vrm` files that
    /// are not already in the library. The escape hatch for cases
    /// where the notify-based watcher misses an event — see
    /// `folder_watcher.rs` module docstring for which environments
    /// (UNC, OneDrive, symlinks) need this routinely.
    pub fn refresh_watched_folders(&mut self) {
        let dirs: Vec<std::path::PathBuf> = self.library.watched_avatar_dirs.clone();
        let mut imported = 0usize;
        for dir in &dirs {
            if !dir.exists() {
                continue;
            }
            let entries = match std::fs::read_dir(dir) {
                Ok(e) => e,
                Err(e) => {
                    warn!(
                        "refresh_watched_folders: read_dir '{}' failed: {}",
                        dir.display(),
                        e
                    );
                    continue;
                }
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let is_vrm = path
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("vrm"))
                    .unwrap_or(false);
                if !is_vrm {
                    continue;
                }
                if self.import_vrm_into_library_if_missing(&path) {
                    imported += 1;
                }
            }
        }
        if imported > 0 {
            if !self.test_no_persist {
                self.save_avatar_library_with_toast();
            }
            self.push_notification(t!("toast.refreshed_imported", count = imported));
        } else {
            self.push_notification(t!("toast.refreshed_none"));
        }
    }

    /// Add `vrm_path` to the avatar library if it isn't already there.
    /// Best-effort load: a panicking VRM loader, a malformed file, or
    /// a missing thumbnail all degrade gracefully — the entry still
    /// goes into the library with a placeholder thumbnail so the user
    /// sees *something*. Returns `true` iff the library grew by one.
    ///
    /// Shared by [`poll_folder_watcher`](Self::poll_folder_watcher)
    /// and [`refresh_watched_folders`](Self::refresh_watched_folders)
    /// so notify-driven and manual paths produce identical outcomes.
    pub(super) fn import_vrm_into_library_if_missing(
        &mut self,
        vrm_path: &std::path::Path,
    ) -> bool {
        if self.app.avatar_library.find_by_path(vrm_path).is_some() {
            return false;
        }
        let mut entry =
            crate::app::avatar_library::AvatarLibraryEntry::from_path(vrm_path);
        if let Ok(loader) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::asset::vrm::VrmAssetLoader::new()
        })) {
            if let Ok(asset) = loader.load(&vrm_path.to_string_lossy()) {
                entry.update_from_asset_with_thumbnail_dir(
                    &asset,
                    self.library.thumbnail_gen.output_dir(),
                );
            }
        }
        if entry
            .thumbnail_path
            .as_ref()
            .is_none_or(|p| !p.exists())
        {
            entry.thumbnail_path = self
                .library.thumbnail_gen
                .generate_and_save_placeholder(&entry.name);
        }
        self.app.avatar_library.add(entry);
        true
    }
}
