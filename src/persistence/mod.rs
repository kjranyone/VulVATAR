//! Project persistence: the `.vvtproj` schema, save/load with format
//! migrations, cloth-overlay files, and the facade re-exports that keep
//! the historical `crate::persistence::*` call paths working.
//!
//! Module layout:
//! - `project`       — `ProjectState` / `ProjectFile`, save + load +
//!                     the version-migration chain, `.vvtcloth` overlays
//! - `project_dto`   — on-disk config DTOs
//! - `app_settings`  — app-level settings (independent of projects)
//! - `library`       — the %APPDATA% small-file family
//! - `scene_presets` — saved lighting/camera/rendering presets
//! - `recovery`      — crash-recovery snapshot
//! - `atomic_io`     — durable write + backup-fallback primitives

pub mod app_settings;
pub mod atomic_io;
pub mod library;
pub mod project;
pub mod project_dto;
pub mod recovery;
pub mod scene_presets;

// --- public facade: unchanged historical `persistence::*` paths ---
pub use app_settings::{
    app_settings_path, load_app_settings, migrate_legacy_app_settings, save_app_settings,
    AppSettings, SettingsConfig,
};
pub use atomic_io::atomic_write;
pub use library::{
    cache_dir, last_session_path, load_avatar_library, load_profiles, load_recent_avatars,
    load_watched_folders, profiles_path, save_avatar_library, save_profiles, save_recent_avatars,
    save_watched_folders, thumbnails_dir,
};
pub use project::{
    load_cloth_overlay, load_project, save_project, ClothOverlayFile, ProjectFile,
    ProjectLoadWarnings, ProjectState, LIBRARY_FORMAT_VERSION, OVERLAY_FORMAT_VERSION,
    PROJECT_FORMAT_VERSION, RECOVERY_FORMAT_VERSION,
};
pub use project_dto::{LipSyncConfig, OutputConfig, RenderingConfig, TrackingConfig, TransformState};
pub use recovery::{validate_recovery, RecoveryManager, RecoverySnapshot};
pub use scene_presets::{
    load_scene_presets, save_scene_presets, ScenePreset, ScenePresetCamera, ScenePresetLighting,
    ScenePresetRendering,
};

// --- names kept in this namespace for child modules (`super::` access) ---
use atomic_io::{backup_path_for, load_or_backup};
use library::app_data_dir;
use project::{app_tag, default_gravity_direction, default_spring_sway_scale, default_unit_scale};

// --- test-only helpers parameterized by explicit paths ---
#[cfg(test)]
use app_settings::{load_app_settings_from, migrate_legacy_app_settings_from, save_app_settings_to};
#[cfg(test)]
use project::{migrate_chain, migrate_cloth_overlay_json, read_format_version};

#[cfg(test)]
mod tests;
