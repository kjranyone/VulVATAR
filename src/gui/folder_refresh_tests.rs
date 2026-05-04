//! Cover the manual-Refresh escape hatch added because the notify
//! backend can miss events on cloud-sync / network-share folders
//! (see `app/folder_watcher.rs` module docstring). The
//! `import_vrm_into_library_if_missing` helper is the shared
//! library-side action; both notify-poll and Refresh route
//! through it, so tests here cover both code paths' outcomes.
use super::*;

fn make_tempdir(suffix: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("vulvatar_folder_refresh_{}_{}", suffix, nanos));
    std::fs::create_dir_all(&dir).expect("create tempdir");
    dir
}

#[test]
fn import_skips_path_already_in_library() {
    let dir = make_tempdir("skip_existing");
    let vrm_path = dir.join("already.vrm");
    std::fs::write(&vrm_path, b"placeholder bytes").expect("seed file");

    let mut harness = GuiApp::for_test();
    harness
        .app
        .avatar_library
        .add(crate::app::avatar_library::AvatarLibraryEntry::from_path(&vrm_path));
    let before = harness.app.avatar_library.entries.len();

    let added = harness.import_vrm_into_library_if_missing(&vrm_path);
    assert!(!added, "an already-tracked path must not be re-added");
    assert_eq!(
        harness.app.avatar_library.entries.len(),
        before,
        "library size must not grow when path is already tracked"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn refresh_imports_new_vrm_files_in_watched_folder() {
    // Watcher's notify backend may have missed a Created event;
    // Refresh must walk the directory and import the VRM anyway.
    // The VRM body is intentionally invalid bytes — the loader
    // panic-catches and the entry still goes in with a placeholder
    // thumbnail, matching the auto-watch path's behavior.
    let dir = make_tempdir("refresh_new");
    let vrm_path = dir.join("missed.vrm");
    std::fs::write(&vrm_path, b"not a real vrm").expect("seed VRM");

    let mut harness = GuiApp::for_test();
    harness.watched_avatar_dirs.push(dir.clone());
    let before = harness.app.avatar_library.entries.len();

    harness.refresh_watched_folders();

    let after = harness.app.avatar_library.entries.len();
    assert_eq!(
        after,
        before + 1,
        "Refresh must import the watched-folder VRM that was missed"
    );
    assert!(
        harness.app.avatar_library.find_by_path(&vrm_path).is_some(),
        "library entry for the imported file must be locatable by path"
    );

    // A subsequent refresh is a no-op — the file is already tracked.
    harness.refresh_watched_folders();
    assert_eq!(
        harness.app.avatar_library.entries.len(),
        before + 1,
        "second Refresh must not duplicate the entry"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn refresh_with_no_watched_folders_is_noop() {
    let mut harness = GuiApp::for_test();
    let before = harness.app.avatar_library.entries.len();
    harness.refresh_watched_folders();
    assert_eq!(harness.app.avatar_library.entries.len(), before);
    // The "no new files" notification is the only side effect.
    assert!(
        harness
            .notifications
            .iter()
            .any(|n| n.message.contains("no new files")),
        "Refresh on an empty watch list must still post the no-new-files toast"
    );
}
