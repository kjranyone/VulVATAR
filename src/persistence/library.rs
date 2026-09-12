//! The %APPDATA% small-file family: avatar library, watched folders,
//! profiles, recent avatars, thumbnails/cache dirs, and the implicit
//! last-session project path. All live under the same app-data dir and
//! share the atomic-write + backup-fallback readers.

use log::{error, info};
use serde::{Deserialize, Serialize};

use super::{app_tag, atomic_write, backup_path_for, load_or_backup, LIBRARY_FORMAT_VERSION};

#[derive(Serialize, Deserialize, Debug)]
pub struct AvatarLibraryFile {
    pub format_version: u32,
    pub created_with: String,
    pub last_saved_with: String,
    #[serde(default)]
    pub library: crate::app::avatar_library::AvatarLibrary,
}

fn library_path() -> std::path::PathBuf {
    let mut path = app_data_dir();
    path.push("avatar_library.vvtlib");
    path
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct WatchedFoldersFile {
    #[serde(default)]
    pub format_version: u32,
    #[serde(default)]
    pub paths: Vec<std::path::PathBuf>,
}

fn watched_folders_path() -> std::path::PathBuf {
    let mut path = app_data_dir();
    path.push("watched_folders.json");
    path
}

/// Where avatar-library thumbnails (real VRM-embedded ones + generated
/// placeholders) are written. Co-located with the library file so a
/// user moving their config dir takes thumbnails along automatically.
pub fn thumbnails_dir() -> std::path::PathBuf {
    let mut path = app_data_dir();
    path.push("thumbnails");
    path
}

/// Where serialised `AvatarAsset` snapshots produced by
/// `src/asset/cache.rs` live. One `<source_hash_hex>.vvtcache` per VRM,
/// keyed by the SHA-256 of the source file. Lives alongside thumbnails
/// so the user can wipe / move both together.
pub fn cache_dir() -> std::path::PathBuf {
    let mut path = app_data_dir();
    path.push("cache");
    path
}

pub fn load_watched_folders() -> Vec<std::path::PathBuf> {
    let path = watched_folders_path();
    if !path.exists() {
        return Vec::new();
    }
    match std::fs::read_to_string(&path) {
        Ok(data) => match serde_json::from_str::<WatchedFoldersFile>(&data) {
            Ok(file) => file.paths,
            Err(e) => {
                error!(
                    "persistence: failed to parse watched folders at {}: {}",
                    path.display(),
                    e
                );
                Vec::new()
            }
        },
        Err(e) => {
            error!(
                "persistence: failed to read watched folders at {}: {}",
                path.display(),
                e
            );
            Vec::new()
        }
    }
}

pub fn save_watched_folders(paths: &[std::path::PathBuf]) -> Result<(), String> {
    let path = watched_folders_path();
    let file = WatchedFoldersFile {
        format_version: 1,
        paths: paths.to_vec(),
    };
    let data = serde_json::to_string_pretty(&file)
        .map_err(|e| format!("serialise watched folders: {}", e))?;
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    atomic_write(&path, &data).map_err(|e| format!("write watched folders: {}", e))
}

fn recent_avatars_path() -> std::path::PathBuf {
    let mut path = app_data_dir();
    path.push("recent_avatars.json");
    path
}

/// Where the user's `StreamProfile` library is stored. Lives next to
/// `last_session.vvtproj` under the OS app-data dir so the same set of
/// profiles is available across every project the user opens — the
/// "home desk" / "office desk" / "show stage" lighting + calibration
/// presets follow the *user*, not whichever scene file they happened
/// to open last.
pub fn profiles_path() -> std::path::PathBuf {
    let mut path = app_data_dir();
    path.push("profiles.json");
    path
}

/// Load the user's saved `ProfileLibrary`. Returns `None` when the file
/// does not yet exist (first run) or fails to parse — the GUI then
/// falls back to `ProfileLibrary::default()` (the built-in Streaming
/// / Recording / Performance presets) so a fresh install or a
/// corrupted file doesn't lock the user out of the profile dropdown.
/// A parse failure is logged but otherwise non-fatal.
pub fn load_profiles() -> Option<crate::gui::profile::ProfileLibrary> {
    let path = profiles_path();
    if !path.exists() {
        return None;
    }
    match std::fs::read_to_string(&path) {
        Ok(data) => match serde_json::from_str(&data) {
            Ok(lib) => Some(lib),
            Err(e) => {
                error!(
                    "persistence: failed to parse profiles at {}: {} \
                     (falling back to built-in presets)",
                    path.display(),
                    e
                );
                None
            }
        },
        Err(e) => {
            error!(
                "persistence: failed to read profiles at {}: {}",
                path.display(),
                e
            );
            None
        }
    }
}

/// Persist the user's `ProfileLibrary`. Called from
/// `GuiApp::save_profiles_if_dirty` whenever `profiles_dirty` flips —
/// triggered by `Calibrate Pose ▼` writing into the active profile,
/// by future profile-edit UI, etc. Atomic write via `atomic_write`
/// so a mid-write crash never leaves a half-truncated file behind.
pub fn save_profiles(library: &crate::gui::profile::ProfileLibrary) -> Result<(), String> {
    let path = profiles_path();
    let data =
        serde_json::to_string_pretty(library).map_err(|e| format!("serialise profiles: {}", e))?;
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    atomic_write(&path, &data).map_err(|e| format!("write profiles: {}", e))
}

/// Path of the implicit "last session" project file. The GUI writes here
/// every time `project_dirty` flips, even when the user has never run
/// File > Save As — so quitting after toggling a checkbox preserves the
/// change for the next launch. Lives next to `avatar_library.vvtlib` and
/// recovery snapshots under `%APPDATA%\VulVATAR`.
pub fn last_session_path() -> std::path::PathBuf {
    let mut path = app_data_dir();
    path.push("last_session.vvtproj");
    path
}

pub(super) fn app_data_dir() -> std::path::PathBuf {
    let mut path = dirs_data_dir();
    path.push("VulVATAR");
    let _ = std::fs::create_dir_all(&path);
    path
}

fn dirs_data_dir() -> std::path::PathBuf {
    if cfg!(target_os = "windows") {
        let appdata = std::env::var("APPDATA").unwrap_or_else(|_| ".".to_string());
        std::path::PathBuf::from(appdata)
    } else if cfg!(target_os = "macos") {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        std::path::PathBuf::from(home)
            .join("Library")
            .join("Application Support")
    } else {
        let xdg = std::env::var("XDG_DATA_HOME").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            format!("{}/.local/share", home)
        });
        std::path::PathBuf::from(xdg)
    }
}

pub fn load_avatar_library() -> crate::app::avatar_library::AvatarLibrary {
    let path = library_path();
    if !path.exists() {
        let backup = backup_path_for(&path);
        if !backup.exists() {
            info!(
                "persistence: no avatar library at {}, creating empty",
                path.display()
            );
            return crate::app::avatar_library::AvatarLibrary::new();
        }
    }
    match load_or_backup::<AvatarLibraryFile>(&path) {
        Ok(file) => {
            info!(
                "persistence: loaded avatar library with {} entries",
                file.library.entries.len()
            );
            file.library
        }
        Err(e) => {
            error!(
                "persistence: failed to load avatar library (primary and backup): {}",
                e
            );
            crate::app::avatar_library::AvatarLibrary::new()
        }
    }
}

pub fn save_avatar_library(
    library: &crate::app::avatar_library::AvatarLibrary,
) -> Result<(), String> {
    let path = library_path();
    let existing_created_with = if path.exists() {
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|data| serde_json::from_str::<AvatarLibraryFile>(&data).ok())
            .map(|f| f.created_with)
    } else {
        None
    };
    let mut file = AvatarLibraryFile {
        format_version: LIBRARY_FORMAT_VERSION,
        created_with: app_tag(),
        last_saved_with: app_tag(),
        library: library.clone(),
    };
    if let Some(cw) = existing_created_with {
        file.created_with = cw;
    }
    let json = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
    atomic_write(&path, &json)?;
    info!(
        "persistence: saved avatar library with {} entries",
        library.entries.len()
    );
    Ok(())
}

// ===========================================================================
// Recent avatars
// ===========================================================================

#[derive(Serialize, Deserialize, Debug, Default)]
struct RecentAvatarsFile {
    #[serde(default)]
    paths: Vec<String>,
}

pub fn load_recent_avatars() -> Vec<std::path::PathBuf> {
    let path = recent_avatars_path();
    if !path.exists() {
        let backup = backup_path_for(&path);
        if !backup.exists() {
            return Vec::new();
        }
    }
    match load_or_backup::<RecentAvatarsFile>(&path) {
        Ok(file) => file
            .paths
            .into_iter()
            .map(std::path::PathBuf::from)
            .collect(),
        Err(e) => {
            error!(
                "persistence: failed to load recent avatars (primary and backup): {}",
                e
            );
            Vec::new()
        }
    }
}

pub fn save_recent_avatars(paths: &[std::path::PathBuf]) -> Result<(), String> {
    let path = recent_avatars_path();
    let file = RecentAvatarsFile {
        paths: paths
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect(),
    };
    let json = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
    atomic_write(&path, &json)?;
    info!("persistence: saved {} recent avatars", paths.len());
    Ok(())
}
