//! App-level settings (independent of any project): locale and the
//! settings block persisted to `app_settings.vvtset` under the app-data
//! dir, plus the legacy migration that lifts settings out of an old
//! project file.

use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::path::Path;

use super::project_dto::{
    default_orbit_sensitivity, default_pan_sensitivity, default_zoom_sensitivity,
};
use super::{app_data_dir, atomic_write, last_session_path, ProjectFile};

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct SettingsConfig {
    #[serde(default = "default_locale")]
    pub locale: String,
    #[serde(default = "default_zoom_sensitivity")]
    pub zoom_sensitivity: f32,
    #[serde(default = "default_orbit_sensitivity")]
    pub orbit_sensitivity: f32,
    #[serde(default = "default_pan_sensitivity")]
    pub pan_sensitivity: f32,
    /// `None` on older project files = the consent dialog will fire
    /// once on first overlay attach in this session.
    #[serde(default)]
    pub cloth_autosave_consent: Option<bool>,
}

fn default_locale() -> String {
    "en".to_string()
}

/// App-level user preferences: UI language, viewport input
/// sensitivities, and cross-session consents. Lives in its own
/// `%APPDATA%\VulVATAR\settings.json`, deliberately OUTSIDE the
/// project file: these follow the user, not the scene. Saved by the
/// GUI's autosave tick whenever `app_settings_dirty` is set (any
/// Settings-pane change), loaded once at startup *before* font setup
/// so the locale drives the CJK fallback order.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct AppSettings {
    #[serde(default = "default_app_settings_version")]
    pub format_version: u32,
    #[serde(default = "default_locale")]
    pub locale: String,
    #[serde(default = "default_zoom_sensitivity")]
    pub zoom_sensitivity: f32,
    #[serde(default = "default_orbit_sensitivity")]
    pub orbit_sensitivity: f32,
    #[serde(default = "default_pan_sensitivity")]
    pub pan_sensitivity: f32,
    /// User's answer to the "include cloth overlay in recovery
    /// snapshots?" prompt. `Some(true)` = opt-in, `Some(false)` =
    /// opt-out, `None` = never asked (dialog fires on first overlay
    /// attach).
    #[serde(default)]
    pub cloth_autosave_consent: Option<bool>,
    /// Path of the last explicitly opened / saved `.vvtproj`, so the
    /// next launch re-opens the same project automatically (with the
    /// title bar showing the real file — unlike the implicit
    /// `last_session.vvtproj` slot, which stays anonymous). `None` when
    /// the user never opened or saved a named project, or when the file
    /// went missing at startup (the field is then cleared so a deleted
    /// project doesn't warn on every launch).
    #[serde(default)]
    pub last_project_path: Option<String>,
    /// Serial of the RealSense D400 the user picked in the Tracking
    /// panel's device list (radio selection). `None` = no explicit
    /// pick, i.e. "use the first connected". The capture backend
    /// falls back to the first device when this serial isn't
    /// connected, so a stale pick never blocks capture.
    #[serde(default)]
    pub camera_serial: Option<String>,
    /// Cloth solver backend for freshly attached cloth. `None` keeps
    /// the `VULVATAR_CLOTH_GPU` env-var decision (CPU default);
    /// `Some(true)` opts every future attach into the GPU compute
    /// backend for this install. Takes effect on the NEXT attach —
    /// the backend is a one-shot per-cloth decision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cloth_gpu_backend: Option<bool>,
}

fn default_app_settings_version() -> u32 {
    1
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            format_version: default_app_settings_version(),
            locale: default_locale(),
            zoom_sensitivity: default_zoom_sensitivity(),
            orbit_sensitivity: default_orbit_sensitivity(),
            pan_sensitivity: default_pan_sensitivity(),
            cloth_autosave_consent: None,
            last_project_path: None,
            camera_serial: None,
            cloth_gpu_backend: None,
        }
    }
}

pub fn app_settings_path() -> std::path::PathBuf {
    app_data_dir().join("settings.json")
}

/// Load `settings.json`. `None` when the file doesn't exist or fails
/// to parse — the caller falls back to [`migrate_legacy_app_settings`]
/// / defaults. Field-level `serde(default)`s keep partially-old files
/// loading as the file grows new fields.
pub fn load_app_settings() -> Option<AppSettings> {
    load_app_settings_from(&app_settings_path())
}

pub(super) fn load_app_settings_from(path: &Path) -> Option<AppSettings> {
    let data = std::fs::read_to_string(path).ok()?;
    // Tolerate a UTF-8 BOM: Windows editors (and PowerShell 5.1's
    // `-Encoding utf8`) prepend one, and serde_json rejects it. An
    // unreadable file falls back to migration/defaults — i.e. a BOM
    // would silently RESET the user's settings, the exact failure
    // this file exists to prevent.
    match serde_json::from_str(data.trim_start_matches('\u{feff}')) {
        Ok(settings) => Some(settings),
        Err(e) => {
            warn!("persistence: settings.json unreadable ({e}); using defaults");
            None
        }
    }
}

pub fn save_app_settings(settings: &AppSettings) -> Result<(), String> {
    save_app_settings_to(settings, &app_settings_path())
}

pub(super) fn save_app_settings_to(settings: &AppSettings, path: &Path) -> Result<(), String> {
    let data = serde_json::to_string_pretty(settings)
        .map_err(|e| format!("serialise app settings: {e}"))?;
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    atomic_write(path, &data).map_err(|e| format!("write app settings: {e}"))
}

/// One-time upgrade path for installs that predate `settings.json`:
/// pull the app-level settings out of the legacy slot in
/// `last_session.vvtproj` (where they used to ride along with the
/// project state). Returns `None` when there's nothing to migrate.
pub fn migrate_legacy_app_settings() -> Option<AppSettings> {
    let migrated = migrate_legacy_app_settings_from(&last_session_path());
    if migrated.is_some() {
        info!("persistence: migrated app settings from legacy last-session slot");
    }
    migrated
}

pub(super) fn migrate_legacy_app_settings_from(path: &Path) -> Option<AppSettings> {
    let data = std::fs::read_to_string(path).ok()?;
    let file: ProjectFile = serde_json::from_str(&data).ok()?;
    let legacy = file.settings?;
    Some(AppSettings {
        format_version: default_app_settings_version(),
        locale: legacy.locale,
        zoom_sensitivity: legacy.zoom_sensitivity,
        orbit_sensitivity: legacy.orbit_sensitivity,
        pan_sensitivity: legacy.pan_sensitivity,
        cloth_autosave_consent: legacy.cloth_autosave_consent,
        last_project_path: None,
        // Legacy projects predate the device pick — no selection yet.
        camera_serial: None,
        // And predate the cloth backend preference — env decides.
        cloth_gpu_backend: None,
    })
}
