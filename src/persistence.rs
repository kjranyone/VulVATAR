#![allow(dead_code)]
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Pure-data snapshot of the GUI/app state needed for project serialization.
/// This struct lives here so that `persistence` never imports `gui::GuiApp`,
/// breaking the former circular dependency.
#[derive(Clone, Debug)]
pub struct ProjectState {
    // Avatar info (extracted from Application, not GuiApp directly)
    pub avatar_source_path: Option<String>,
    pub avatar_source_hash: Option<Vec<u8>>,
    pub active_overlay_path: Option<String>,
    /// Filesystem paths of every cloth overlay attached to the active
    /// avatar via "Load Overlay File..." (i.e. each `ClothOverlaySlot`
    /// whose `source_path` is `Some`). On load the project re-attaches
    /// each one through `persistence::load_cloth_overlay`. Slots
    /// created procedurally (no file backing) are intentionally
    /// dropped — they have no canonical re-creation source.
    pub cloth_overlay_paths: Vec<String>,

    // Transform
    pub transform_position: [f32; 3],
    pub transform_rotation: [f32; 3],
    pub transform_scale: f32,

    pub camera_orbit_yaw: f32,
    pub camera_orbit_pitch: f32,
    pub camera_orbit_pan: [f32; 2],
    pub camera_orbit_distance: f32,

    // Tracking config
    pub tracking_enabled: bool,
    pub tracking_mirror: bool,
    pub camera_resolution_index: usize,
    pub camera_framerate_index: usize,
    pub smoothing_strength: f32,
    pub confidence_threshold: f32,
    pub hand_tracking_enabled: bool,
    pub face_tracking_enabled: bool,
    pub lower_body_tracking_enabled: bool,
    pub camera_index: usize,
    pub show_camera_wipe: bool,
    pub show_detection_annotations: bool,

    // Rendering config
    pub material_mode_index: usize,
    pub toon_ramp_threshold: f32,
    pub shadow_softness: f32,
    pub outline_enabled: bool,
    pub outline_width: f32,
    pub outline_color: [f32; 3],
    pub light_direction: [f32; 3],
    pub light_intensity: f32,
    pub ambient: [f32; 3],
    pub camera_fov: f32,
    pub background_color: [f32; 3],
    pub transparent_background: bool,
    pub toggle_spring: bool,
    pub toggle_cloth: bool,

    // Lip sync config
    pub lipsync_enabled: bool,
    pub lipsync_mic_device_index: usize,
    pub lipsync_volume_threshold: f32,
    pub lipsync_smoothing: f32,

    // Output config
    pub output_sink_index: usize,
    pub output_resolution_index: usize,
    pub output_framerate_index: usize,
    pub output_has_alpha: bool,
    pub output_color_space_index: usize,

    // Settings
    pub settings_locale: String,
    pub settings_zoom_sensitivity: f32,
    pub settings_orbit_sensitivity: f32,
    pub settings_pan_sensitivity: f32,
    pub settings_autosave_interval_secs: Option<u64>,
}

/// Serializable project state saved as `.vvtproj`.
#[derive(Serialize, Deserialize, Debug)]
pub struct ProjectFile {
    pub format_version: u32,
    pub created_with: String,
    pub last_saved_with: String,

    pub avatar_source_path: Option<String>,
    /// SHA-256 hash of the avatar source file at the time the project was saved.
    pub avatar_source_hash: Option<Vec<u8>>,
    pub avatar_transform: TransformState,
    #[serde(default)]
    pub camera_orbit_yaw: f32,
    #[serde(default)]
    pub camera_orbit_pitch: f32,
    #[serde(default)]
    pub camera_orbit_pan: [f32; 2],
    #[serde(default)]
    pub camera_orbit_distance: f32,
    pub active_overlay_path: Option<String>,
    /// See [`ProjectState::cloth_overlay_paths`]. `#[serde(default)]`
    /// keeps older projects (which lacked this field) loadable.
    #[serde(default)]
    pub cloth_overlay_paths: Vec<String>,

    pub tracking: TrackingConfig,
    pub rendering: RenderingConfig,
    #[serde(default)]
    pub lipsync: LipSyncConfig,
    pub output: OutputConfig,
    #[serde(default)]
    pub settings: SettingsConfig,
}

/// Warnings produced during project reload validation.
#[derive(Debug, Default)]
pub struct ProjectLoadWarnings {
    pub warnings: Vec<String>,
}

fn default_true() -> bool {
    true
}

/// Fallback for `TrackingConfig::confidence_threshold` when the field is
/// missing from a saved project (older serializations). Without this, serde's
/// `f32::default()` would return `0.0`, which silently disables confidence
/// filtering — every keypoint passes, including spurious ones, making
/// freshly-loaded projects feel twitchy until the user touches the slider.
fn default_confidence_threshold() -> f32 {
    crate::tracking::DEFAULT_CONFIDENCE_THRESHOLD
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct TransformState {
    #[serde(default)]
    pub position: [f32; 3],
    #[serde(default)]
    pub rotation: [f32; 3],
    #[serde(default)]
    pub scale: f32,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct TrackingConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub mirror: bool,
    #[serde(default)]
    pub camera_resolution_index: usize,
    #[serde(default)]
    pub camera_framerate_index: usize,
    #[serde(default)]
    pub smoothing_strength: f32,
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,
    #[serde(default)]
    pub hand_tracking_enabled: bool,
    #[serde(default)]
    pub face_tracking_enabled: bool,
    #[serde(default)]
    pub lower_body_tracking_enabled: bool,
    #[serde(default)]
    pub camera_index: usize,
    #[serde(default)]
    pub show_camera_wipe: bool,
    #[serde(default)]
    pub show_detection_annotations: bool,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct RenderingConfig {
    #[serde(default)]
    pub material_mode_index: usize,
    #[serde(default)]
    pub toon_ramp_threshold: f32,
    #[serde(default)]
    pub shadow_softness: f32,
    #[serde(default)]
    pub outline_enabled: bool,
    #[serde(default)]
    pub outline_width: f32,
    #[serde(default)]
    pub outline_color: [f32; 3],
    #[serde(default)]
    pub light_direction: [f32; 3],
    #[serde(default)]
    pub light_intensity: f32,
    #[serde(default)]
    pub ambient: [f32; 3],
    #[serde(default)]
    pub camera_fov: f32,
    #[serde(default)]
    pub background_color: [f32; 3],
    #[serde(default)]
    pub transparent_background: bool,
    #[serde(default = "default_true")]
    pub toggle_spring: bool,
    #[serde(default)]
    pub toggle_cloth: bool,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct LipSyncConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub mic_device_index: usize,
    #[serde(default = "default_volume_threshold")]
    pub volume_threshold: f32,
    #[serde(default = "default_lipsync_smoothing")]
    pub smoothing: f32,
}

fn default_volume_threshold() -> f32 {
    0.01
}

fn default_lipsync_smoothing() -> f32 {
    0.5
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct OutputConfig {
    #[serde(default)]
    pub sink_index: usize,
    #[serde(default)]
    pub resolution_index: usize,
    #[serde(default)]
    pub framerate_index: usize,
    #[serde(default)]
    pub has_alpha: bool,
    #[serde(default)]
    pub color_space_index: usize,
}

fn default_zoom_sensitivity() -> f32 {
    0.1
}

fn default_orbit_sensitivity() -> f32 {
    0.3
}

fn default_pan_sensitivity() -> f32 {
    1.0
}

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
    #[serde(default)]
    pub autosave_interval_secs: Option<u64>,
}

fn default_locale() -> String {
    "en".to_string()
}

/// Full cloth overlay file for `.vvtcloth` files, including the complete ClothAsset data.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ClothOverlayFile {
    pub format_version: u32,
    pub created_with: String,
    pub last_saved_with: String,
    pub overlay_name: String,
    pub target_avatar_path: Option<String>,
    /// Full cloth asset data for complete serialization round-trip.
    pub cloth_asset: Option<crate::asset::ClothAsset>,
    /// Audit marker recorded when `cloth_rebind::rebind_overlay`
    /// rewrote internal IDs against a re-imported avatar. `None` for
    /// overlays that have never been auto-rebound. `serde(default)`
    /// keeps older `.vvtcloth` files loadable without a format bump.
    #[serde(default)]
    pub last_rebound_with: Option<String>,
}

const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Currently-shipping schema version for `.vvtproj` project files. Bump
/// whenever a non-additive change lands and add a migrator entry to
/// [`migrate_project_json`].
pub const PROJECT_FORMAT_VERSION: u32 = 1;

/// Currently-shipping schema version for `.vvtcloth` overlay files.
/// Bump whenever a non-additive change lands and add a migrator entry
/// to [`migrate_cloth_overlay_json`].
pub const OVERLAY_FORMAT_VERSION: u32 = 1;

/// Currently-shipping schema version for `.vvtlib` avatar-library files.
pub const LIBRARY_FORMAT_VERSION: u32 = 1;

/// Currently-shipping schema version for the autosave recovery snapshot.
pub const RECOVERY_FORMAT_VERSION: u32 = 1;

fn app_tag() -> String {
    format!("VulVATAR {}", APP_VERSION)
}

/// Generic version-chain walker. Repeatedly applies `step(json, version)`
/// while incrementing the version counter, until `target_version` is
/// reached. Each `step` is responsible for the `version → version + 1`
/// transformation.
///
/// Extracted so it can be unit-tested with a fake `step` — the real
/// `step_overlay` / `step_project` match arms have nothing in them yet
/// (every released file is already at the current version), so without
/// a generic helper the loop logic itself never fires under test, and
/// would only meet the wild on the day a v2 ships.
fn migrate_chain<StepFn>(
    json: serde_json::Value,
    from_version: u32,
    target_version: u32,
    step: StepFn,
) -> Result<serde_json::Value, String>
where
    StepFn: Fn(serde_json::Value, u32) -> Result<serde_json::Value, String>,
{
    let mut current = json;
    let mut version = from_version;
    while version < target_version {
        current = step(current, version)?;
        version += 1;
    }
    Ok(current)
}

/// Walk a JSON document up the cloth-overlay version chain until it
/// matches [`OVERLAY_FORMAT_VERSION`]. Each iteration looks up the
/// migrator for `(version → version + 1)` and bumps the counter.
///
/// No migrators exist yet — every released file is already at v1. When
/// a v2 lands, register its migrator in [`step_overlay`]; the chain
/// then handles `v0 → v1 → v2` automatically.
fn migrate_cloth_overlay_json(
    json: serde_json::Value,
    from_version: u32,
) -> Result<serde_json::Value, String> {
    migrate_chain(json, from_version, OVERLAY_FORMAT_VERSION, step_overlay)
}

fn step_overlay(_json: serde_json::Value, version: u32) -> Result<serde_json::Value, String> {
    match version {
        // Future migrators land here:
        //   1 => v1_to_v2(json),
        v => Err(format!(
            "no migrator registered for cloth overlay v{} → v{}",
            v,
            v + 1
        )),
    }
}

/// See [`migrate_cloth_overlay_json`]. Same shape for project files.
fn migrate_project_json(
    json: serde_json::Value,
    from_version: u32,
) -> Result<serde_json::Value, String> {
    migrate_chain(json, from_version, PROJECT_FORMAT_VERSION, step_project)
}

fn step_project(_json: serde_json::Value, version: u32) -> Result<serde_json::Value, String> {
    match version {
        v => Err(format!(
            "no migrator registered for project v{} → v{}",
            v,
            v + 1
        )),
    }
}

fn read_format_version(json: &serde_json::Value) -> u32 {
    json.get("format_version")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .unwrap_or(0)
}

impl ProjectFile {
    /// Snapshot the current project state into a serializable project file.
    pub fn from_state(state: &ProjectState) -> Self {
        Self {
            format_version: PROJECT_FORMAT_VERSION,
            created_with: app_tag(),
            last_saved_with: app_tag(),

            avatar_source_path: state.avatar_source_path.clone(),
            avatar_source_hash: state.avatar_source_hash.clone(),
            avatar_transform: TransformState {
                position: state.transform_position,
                rotation: state.transform_rotation,
                scale: state.transform_scale,
            },
            camera_orbit_yaw: state.camera_orbit_yaw,
            camera_orbit_pitch: state.camera_orbit_pitch,
            camera_orbit_pan: state.camera_orbit_pan,
            camera_orbit_distance: state.camera_orbit_distance,
            active_overlay_path: state.active_overlay_path.clone(),
            cloth_overlay_paths: state.cloth_overlay_paths.clone(),

            tracking: TrackingConfig {
                enabled: state.tracking_enabled,
                mirror: state.tracking_mirror,
                camera_resolution_index: state.camera_resolution_index,
                camera_framerate_index: state.camera_framerate_index,
                smoothing_strength: state.smoothing_strength,
                confidence_threshold: state.confidence_threshold,
                hand_tracking_enabled: state.hand_tracking_enabled,
                face_tracking_enabled: state.face_tracking_enabled,
                lower_body_tracking_enabled: state.lower_body_tracking_enabled,
                camera_index: state.camera_index,
                show_camera_wipe: state.show_camera_wipe,
                show_detection_annotations: state.show_detection_annotations,
            },
            rendering: RenderingConfig {
                material_mode_index: state.material_mode_index,
                toon_ramp_threshold: state.toon_ramp_threshold,
                shadow_softness: state.shadow_softness,
                outline_enabled: state.outline_enabled,
                outline_width: state.outline_width,
                outline_color: state.outline_color,
                light_direction: state.light_direction,
                light_intensity: state.light_intensity,
                ambient: state.ambient,
                camera_fov: state.camera_fov,
                background_color: state.background_color,
                transparent_background: state.transparent_background,
                toggle_spring: state.toggle_spring,
                toggle_cloth: state.toggle_cloth,
            },
            lipsync: LipSyncConfig {
                enabled: state.lipsync_enabled,
                mic_device_index: state.lipsync_mic_device_index,
                volume_threshold: state.lipsync_volume_threshold,
                smoothing: state.lipsync_smoothing,
            },
            output: OutputConfig {
                sink_index: state.output_sink_index,
                resolution_index: state.output_resolution_index,
                framerate_index: state.output_framerate_index,
                has_alpha: state.output_has_alpha,
                color_space_index: state.output_color_space_index,
            },
            settings: SettingsConfig {
                locale: state.settings_locale.clone(),
                zoom_sensitivity: state.settings_zoom_sensitivity,
                orbit_sensitivity: state.settings_orbit_sensitivity,
                pan_sensitivity: state.settings_pan_sensitivity,
                autosave_interval_secs: state.settings_autosave_interval_secs,
            },
        }
    }

    /// Convert a loaded project file back into a `ProjectState`.
    pub fn to_state(&self) -> ProjectState {
        ProjectState {
            avatar_source_path: self.avatar_source_path.clone(),
            avatar_source_hash: self.avatar_source_hash.clone(),
            active_overlay_path: self.active_overlay_path.clone(),
            cloth_overlay_paths: self.cloth_overlay_paths.clone(),

            transform_position: self.avatar_transform.position,
            transform_rotation: self.avatar_transform.rotation,
            transform_scale: self.avatar_transform.scale,

            camera_orbit_yaw: self.camera_orbit_yaw,
            camera_orbit_pitch: self.camera_orbit_pitch,
            camera_orbit_pan: self.camera_orbit_pan,
            camera_orbit_distance: self.camera_orbit_distance,

            tracking_enabled: self.tracking.enabled,
            tracking_mirror: self.tracking.mirror,
            camera_resolution_index: self.tracking.camera_resolution_index,
            camera_framerate_index: self.tracking.camera_framerate_index,
            smoothing_strength: self.tracking.smoothing_strength,
            confidence_threshold: self.tracking.confidence_threshold,
            hand_tracking_enabled: self.tracking.hand_tracking_enabled,
            face_tracking_enabled: self.tracking.face_tracking_enabled,
            lower_body_tracking_enabled: self.tracking.lower_body_tracking_enabled,
            camera_index: self.tracking.camera_index,
            show_camera_wipe: self.tracking.show_camera_wipe,
            show_detection_annotations: self.tracking.show_detection_annotations,

            material_mode_index: self.rendering.material_mode_index,
            toon_ramp_threshold: self.rendering.toon_ramp_threshold,
            shadow_softness: self.rendering.shadow_softness,
            outline_enabled: self.rendering.outline_enabled,
            outline_width: self.rendering.outline_width,
            outline_color: self.rendering.outline_color,
            light_direction: self.rendering.light_direction,
            light_intensity: self.rendering.light_intensity,
            ambient: self.rendering.ambient,
            camera_fov: self.rendering.camera_fov,
            background_color: self.rendering.background_color,
            transparent_background: self.rendering.transparent_background,
            toggle_spring: self.rendering.toggle_spring,
            toggle_cloth: self.rendering.toggle_cloth,

            lipsync_enabled: self.lipsync.enabled,
            lipsync_mic_device_index: self.lipsync.mic_device_index,
            lipsync_volume_threshold: self.lipsync.volume_threshold,
            lipsync_smoothing: self.lipsync.smoothing,

            output_sink_index: self.output.sink_index,
            output_resolution_index: self.output.resolution_index,
            output_framerate_index: self.output.framerate_index,
            output_has_alpha: self.output.has_alpha,
            output_color_space_index: self.output.color_space_index,

            settings_locale: self.settings.locale.clone(),
            settings_zoom_sensitivity: self.settings.zoom_sensitivity,
            settings_orbit_sensitivity: self.settings.orbit_sensitivity,
            settings_pan_sensitivity: self.settings.pan_sensitivity,
            settings_autosave_interval_secs: self.settings.autosave_interval_secs,
        }
    }
}

/// Write a project file to disk as pretty-printed JSON using atomic write
/// (write to temp file, then rename).
pub fn save_project(state: &ProjectState, path: &Path) -> Result<(), String> {
    let existing_created_with = if path.exists() {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|data| serde_json::from_str::<ProjectFile>(&data).ok())
            .map(|f| f.created_with)
    } else {
        None
    };
    let mut project = ProjectFile::from_state(state);
    if let Some(cw) = existing_created_with {
        project.created_with = cw;
    }
    let json = serde_json::to_string_pretty(&project).map_err(|e| e.to_string())?;
    atomic_write(path, &json)?;
    info!("persistence: saved project to {}", path.display());
    Ok(())
}

/// Load a project file from disk and return the deserialized state plus
/// validation warnings. The caller is responsible for applying the returned
/// `ProjectState` to its own structures (e.g. `GuiApp`).
///
/// Two-stage parse mirrors [`load_cloth_overlay`]: inspect
/// `format_version` first, run migrations for older files, then
/// deserialize. Newer-than-supported files are rejected.
pub fn load_project(path: &Path) -> Result<(ProjectState, ProjectLoadWarnings), String> {
    let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let json: serde_json::Value =
        serde_json::from_str(&data).map_err(|e| format!("parse project JSON: {}", e))?;

    let from_version = read_format_version(&json);
    if from_version > PROJECT_FORMAT_VERSION {
        return Err(format!(
            "project '{}' was saved by a newer VulVATAR \
             (format_version {}); this build only supports up to v{}",
            path.display(),
            from_version,
            PROJECT_FORMAT_VERSION
        ));
    }

    let json = if from_version < PROJECT_FORMAT_VERSION {
        info!(
            "persistence: migrating project '{}' from v{} to v{}",
            path.display(),
            from_version,
            PROJECT_FORMAT_VERSION
        );
        migrate_project_json(json, from_version)?
    } else {
        json
    };

    let project: ProjectFile = serde_json::from_value(json)
        .map_err(|e| format!("decode project (post-migration): {}", e))?;

    let mut warnings = ProjectLoadWarnings::default();
    if let Some(ref p) = project.avatar_source_path {
        if p.trim().is_empty() {
            warnings.warnings.push(
                "avatar_source_path is set but empty; the avatar may not load correctly"
                    .to_string(),
            );
        }
    }

    let state = project.to_state();

    info!("persistence: loaded project from {}", path.display());
    for w in &warnings.warnings {
        warn!("persistence: WARNING: {}", w);
    }
    Ok((state, warnings))
}

/// Read a cloth overlay file from disk, deserializing the full ClothAsset data if present.
///
/// Two-stage parse: first as `serde_json::Value` so we can inspect
/// `format_version` and run migrations for older files; then into the
/// typed `ClothOverlayFile`. Files saved by a *newer* build of VulVATAR
/// (format_version > [`OVERLAY_FORMAT_VERSION`]) are rejected with a
/// clear error rather than silently dropping unknown fields.
pub fn load_cloth_overlay(path: &Path) -> Result<ClothOverlayFile, String> {
    let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let json: serde_json::Value =
        serde_json::from_str(&data).map_err(|e| format!("parse cloth overlay JSON: {}", e))?;

    let from_version = read_format_version(&json);
    if from_version > OVERLAY_FORMAT_VERSION {
        return Err(format!(
            "cloth overlay '{}' was saved by a newer VulVATAR \
             (format_version {}); this build only supports up to v{}",
            path.display(),
            from_version,
            OVERLAY_FORMAT_VERSION
        ));
    }

    let json = if from_version < OVERLAY_FORMAT_VERSION {
        info!(
            "persistence: migrating cloth overlay '{}' from v{} to v{}",
            path.display(),
            from_version,
            OVERLAY_FORMAT_VERSION
        );
        migrate_cloth_overlay_json(json, from_version)?
    } else {
        json
    };

    let overlay: ClothOverlayFile = serde_json::from_value(json)
        .map_err(|e| format!("decode cloth overlay (post-migration): {}", e))?;
    info!(
        "persistence: loaded cloth overlay '{}' from {} (has cloth_asset: {})",
        overlay.overlay_name,
        path.display(),
        overlay.cloth_asset.is_some(),
    );
    Ok(overlay)
}

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
    atomic_write(&path, &data)
        .map_err(|e| format!("write watched folders: {}", e))
}

fn recent_avatars_path() -> std::path::PathBuf {
    let mut path = app_data_dir();
    path.push("recent_avatars.json");
    path
}

fn app_data_dir() -> std::path::PathBuf {
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

fn backup_path_for(path: &std::path::Path) -> std::path::PathBuf {
    path.with_extension(
        path.extension()
            .map(|e| {
                let mut s = e.to_string_lossy().into_owned();
                s.push_str(".bak");
                s
            })
            .unwrap_or_else(|| "bak".to_string()),
    )
}

fn load_with_fallback<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> Result<T, String> {
    let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&data).map_err(|e| e.to_string())
}

fn load_or_backup<T: serde::de::DeserializeOwned>(primary: &std::path::Path) -> Result<T, String> {
    let backup = backup_path_for(primary);
    match load_with_fallback::<T>(primary) {
        Ok(v) => Ok(v),
        Err(primary_err) => {
            warn!(
                "persistence: primary file failed: {}, trying backup",
                primary_err
            );
            load_with_fallback::<T>(&backup)
                .map_err(|backup_err| format!("primary: {}; backup: {}", primary_err, backup_err))
        }
    }
}

pub fn atomic_write(path: &Path, data: &str) -> Result<(), String> {
    if path.exists() {
        let _ = std::fs::copy(path, &backup_path_for(path));
    }

    let parent = path.parent().unwrap_or(Path::new("."));
    let temp_name = format!(
        ".{}.tmp",
        path.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".to_string())
    );
    let temp_path = parent.join(&temp_name);

    std::fs::write(&temp_path, data).map_err(|e| {
        format!(
            "atomic_write: failed to write temp file {}: {}",
            temp_path.display(),
            e
        )
    })?;

    if cfg!(windows) {
        if std::fs::rename(&temp_path, path).is_err() {
            std::fs::copy(&temp_path, path).map_err(|e| {
                let _ = std::fs::remove_file(&temp_path);
                format!(
                    "atomic_write: failed to copy {} -> {}: {}",
                    temp_path.display(),
                    path.display(),
                    e
                )
            })?;
            let _ = std::fs::remove_file(&temp_path);
        }
    } else {
        std::fs::rename(&temp_path, path).map_err(|e| {
            let _ = std::fs::remove_file(&temp_path);
            format!(
                "atomic_write: failed to rename {} -> {}: {}",
                temp_path.display(),
                path.display(),
                e
            )
        })?;
    }

    Ok(())
}

// ===========================================================================
// Crash recovery
// ===========================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct RecoverySnapshot {
    pub format_version: u32,
    pub created_with: String,
    pub timestamp_secs: u64,
    #[serde(default)]
    pub project: Option<ProjectFile>,
    pub overlay_dirty: bool,
    #[serde(default)]
    pub overlay: Option<ClothOverlayFile>,
}

fn recovery_dir() -> std::path::PathBuf {
    let mut dir = app_data_dir();
    dir.push("recovery");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn recovery_snapshot_path() -> std::path::PathBuf {
    recovery_dir().join("recovery.vvtsnap")
}

pub struct RecoveryManager {
    interval_secs: u64,
    last_snapshot_secs: u64,
    enabled: bool,
}

impl RecoveryManager {
    pub fn new(interval_secs: u64) -> Self {
        Self {
            interval_secs,
            last_snapshot_secs: 0,
            enabled: true,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn set_interval(&mut self, secs: u64) {
        self.interval_secs = secs;
    }

    pub fn should_snapshot(&self, now_secs: u64) -> bool {
        if !self.enabled {
            return false;
        }
        now_secs >= self.last_snapshot_secs + self.interval_secs
    }

    pub fn write_snapshot(
        &mut self,
        project_state: Option<&ProjectState>,
        overlay_dirty: bool,
        overlay: Option<&ClothOverlayFile>,
    ) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let snapshot = RecoverySnapshot {
            format_version: RECOVERY_FORMAT_VERSION,
            created_with: app_tag(),
            timestamp_secs: now,
            project: project_state.map(ProjectFile::from_state),
            overlay_dirty,
            overlay: overlay.cloned(),
        };

        let json = serde_json::to_string_pretty(&snapshot).map_err(|e| e.to_string())?;
        let path = recovery_snapshot_path();
        atomic_write(&path, &json)?;

        self.last_snapshot_secs = now;
        Ok(())
    }

    pub fn detect_recovery() -> Option<RecoverySnapshot> {
        let path = recovery_snapshot_path();
        if !path.exists() {
            return None;
        }
        match std::fs::read_to_string(&path) {
            Ok(data) => match serde_json::from_str::<RecoverySnapshot>(&data) {
                Ok(snapshot) => {
                    info!(
                        "persistence: detected recovery snapshot from {} (has_project={}, has_overlay={})",
                        snapshot.timestamp_secs,
                        snapshot.project.is_some(),
                        snapshot.overlay.is_some(),
                    );
                    Some(snapshot)
                }
                Err(e) => {
                    error!("persistence: corrupt recovery snapshot, removing: {}", e);
                    let _ = std::fs::remove_file(&path);
                    None
                }
            },
            Err(e) => {
                error!("persistence: cannot read recovery snapshot: {}", e);
                None
            }
        }
    }

    pub fn clear_recovery() {
        let path = recovery_snapshot_path();
        if path.exists() {
            let _ = std::fs::remove_file(&path);
            info!("persistence: cleared recovery snapshot");
        }
    }
}

pub fn validate_recovery(snapshot: &RecoverySnapshot) -> Result<(), String> {
    if snapshot.format_version != RECOVERY_FORMAT_VERSION {
        return Err(format!(
            "unsupported recovery format_version {} (expected {})",
            snapshot.format_version, RECOVERY_FORMAT_VERSION
        ));
    }
    if let Some(ref project) = snapshot.project {
        if project.format_version != PROJECT_FORMAT_VERSION {
            return Err(format!(
                "unsupported project format_version {} in recovery (expected {})",
                project.format_version, PROJECT_FORMAT_VERSION
            ));
        }
    }
    if let Some(ref overlay) = snapshot.overlay {
        if overlay.format_version != OVERLAY_FORMAT_VERSION {
            return Err(format!(
                "unsupported overlay format_version {} in recovery (expected {})",
                overlay.format_version, OVERLAY_FORMAT_VERSION
            ));
        }
    }
    Ok(())
}

// ===========================================================================
// Scene presets
// ===========================================================================

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ScenePresetLighting {
    pub main_light_dir: [f32; 3],
    pub main_light_intensity: f32,
    pub ambient_intensity: [f32; 3],
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ScenePresetCamera {
    pub fov: f32,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ScenePresetRendering {
    pub material_mode_index: usize,
    pub toon_ramp_threshold: f32,
    pub shadow_softness: f32,
    pub outline_enabled: bool,
    pub outline_width: f32,
    pub outline_color: [f32; 3],
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ScenePreset {
    pub name: String,
    pub lighting: ScenePresetLighting,
    pub camera: ScenePresetCamera,
    pub rendering: ScenePresetRendering,
}

#[derive(Serialize, Deserialize, Debug)]
struct ScenePresetFile {
    pub format_version: u32,
    pub created_with: String,
    pub last_saved_with: String,
    #[serde(default)]
    pub presets: Vec<ScenePreset>,
}

fn scene_presets_path() -> std::path::PathBuf {
    let mut path = app_data_dir();
    path.push("scene_presets.vvtpresets");
    path
}

pub fn load_scene_presets() -> Vec<ScenePreset> {
    let path = scene_presets_path();
    if !path.exists() {
        let backup = backup_path_for(&path);
        if !backup.exists() {
            return Vec::new();
        }
    }
    match load_or_backup::<ScenePresetFile>(&path) {
        Ok(file) => {
            info!("persistence: loaded {} scene presets", file.presets.len());
            file.presets
        }
        Err(e) => {
            error!(
                "persistence: failed to load scene presets (primary and backup): {}",
                e
            );
            Vec::new()
        }
    }
}

pub fn save_scene_presets(presets: &[ScenePreset]) -> Result<(), String> {
    let path = scene_presets_path();
    let existing_created_with = if path.exists() {
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|data| serde_json::from_str::<ScenePresetFile>(&data).ok())
            .map(|f| f.created_with)
    } else {
        None
    };
    let mut file = ScenePresetFile {
        format_version: 1,
        created_with: app_tag(),
        last_saved_with: app_tag(),
        presets: presets.to_vec(),
    };
    if let Some(cw) = existing_created_with {
        file.created_with = cw;
    }
    let json = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
    atomic_write(&path, &json)?;
    info!("persistence: saved {} scene presets", presets.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tracking_config_supplies_canonical_confidence_when_field_missing() {
        // Older saved projects predate the confidence_threshold field.
        // Without an explicit serde default, missing-field deserialization
        // would silently produce 0.0 (no filtering) — masking noisy keypoints
        // until the user nudges the slider. Make sure the canonical default
        // wins instead.
        let bare = json!({});
        let cfg: TrackingConfig =
            serde_json::from_value(bare).expect("missing field should default, not error");
        assert_eq!(
            cfg.confidence_threshold,
            crate::tracking::DEFAULT_CONFIDENCE_THRESHOLD
        );
    }

    #[test]
    fn tracking_config_round_trips_explicit_confidence() {
        let saved = json!({"confidence_threshold": 0.42});
        let cfg: TrackingConfig = serde_json::from_value(saved).unwrap();
        assert!((cfg.confidence_threshold - 0.42).abs() < 1e-6);
    }

    #[test]
    fn read_format_version_defaults_to_zero_when_missing() {
        let no_field = json!({"foo": "bar"});
        assert_eq!(read_format_version(&no_field), 0);
    }

    #[test]
    fn read_format_version_picks_up_field() {
        let with_field = json!({"format_version": 7});
        assert_eq!(read_format_version(&with_field), 7);
    }

    #[test]
    fn migrate_chain_applies_each_step_with_correct_version_in_order() {
        // Real migrators don't exist yet — without this fake-stepper
        // exercise, the loop's "apply step → bump version → repeat" logic
        // would only fire for the first time on the day a v2 lands. Use a
        // stepper that records the (version → output) sequence so we can
        // verify each step sees the previous step's result with the right
        // version label.
        let stepper = |mut v: serde_json::Value, version: u32| -> Result<_, String> {
            let key = format!("step_{}", version);
            v.as_object_mut().unwrap().insert(key, json!(version));
            Ok(v)
        };
        let result = migrate_chain(json!({"start": true}), 0, 3, stepper).unwrap();
        let obj = result.as_object().unwrap();
        assert_eq!(obj.get("start"), Some(&json!(true)));
        assert_eq!(obj.get("step_0"), Some(&json!(0)));
        assert_eq!(obj.get("step_1"), Some(&json!(1)));
        assert_eq!(obj.get("step_2"), Some(&json!(2)));
        // Target version is 3 → step at version 3 must NOT be called.
        assert!(!obj.contains_key("step_3"));
    }

    #[test]
    fn migrate_chain_propagates_step_error_and_stops() {
        // If a stepper returns an error mid-chain, the loop must abort
        // immediately — no later steps fire and the original error
        // bubbles up unmodified.
        let calls = std::cell::RefCell::new(Vec::new());
        let stepper = |v: serde_json::Value, version: u32| -> Result<_, String> {
            calls.borrow_mut().push(version);
            if version == 1 {
                Err(format!("boom at v{}", version))
            } else {
                Ok(v)
            }
        };
        let err = migrate_chain(json!({}), 0, 4, stepper).unwrap_err();
        assert!(err.contains("v1"), "expected error to mention v1, got: {}", err);
        // Stepper called for v0 (success) and v1 (failure) only — v2/v3 must
        // not be reached after the error.
        assert_eq!(*calls.borrow(), vec![0, 1]);
    }

    #[test]
    fn migrate_chain_at_target_does_not_invoke_step() {
        // from == to is the "already current" case. The stepper must not
        // be called at all — invoking it would corrupt up-to-date files.
        let stepper = |_: serde_json::Value, _: u32| -> Result<_, String> {
            panic!("stepper should not be called when from_version == target_version");
        };
        let result = migrate_chain(json!({"x": 1}), 5, 5, stepper).unwrap();
        assert_eq!(result, json!({"x": 1}));
    }

    #[test]
    fn migrate_cloth_overlay_at_current_version_is_passthrough() {
        let original = json!({"format_version": OVERLAY_FORMAT_VERSION, "overlay_name": "x"});
        let migrated =
            migrate_cloth_overlay_json(original.clone(), OVERLAY_FORMAT_VERSION).unwrap();
        assert_eq!(migrated, original);
    }

    #[test]
    fn migrate_cloth_overlay_from_v0_surfaces_missing_migrator() {
        // No v0 → v1 migrator is currently registered. This test documents
        // the framework's failure mode: we want a clear error rather than
        // a silent accept, and we want the chain to fire on a real older
        // version. When a v2 lands, this test will need to be updated to
        // use whatever the *new* "below current" version is.
        let original = json!({"overlay_name": "x"});
        let err = migrate_cloth_overlay_json(original, 0)
            .expect_err("expected error from un-migratable v0");
        assert!(
            err.contains("no migrator"),
            "expected 'no migrator' message, got: {}",
            err
        );
    }

    #[test]
    fn load_cloth_overlay_rejects_future_version() {
        let dir = std::env::temp_dir().join("vulvatar_overlay_future_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("future.vvtcloth");
        let body = json!({
            "format_version": OVERLAY_FORMAT_VERSION + 99,
            "created_with": "test",
            "last_saved_with": "test",
            "overlay_name": "future",
            "target_avatar_path": null,
            "cloth_asset": null,
        });
        std::fs::write(&path, serde_json::to_string(&body).unwrap()).unwrap();
        let err = load_cloth_overlay(&path).expect_err("future version should be rejected");
        assert!(
            err.contains("newer VulVATAR"),
            "expected 'newer VulVATAR' message, got: {}",
            err
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_cloth_overlay_at_current_version_round_trips() {
        let dir = std::env::temp_dir().join("vulvatar_overlay_current_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("current.vvtcloth");
        let body = json!({
            "format_version": OVERLAY_FORMAT_VERSION,
            "created_with": "test",
            "last_saved_with": "test",
            "overlay_name": "current",
            "target_avatar_path": null,
            "cloth_asset": null,
        });
        std::fs::write(&path, serde_json::to_string(&body).unwrap()).unwrap();
        let overlay = load_cloth_overlay(&path).expect("v1 should load directly");
        assert_eq!(overlay.overlay_name, "current");
        assert_eq!(overlay.format_version, OVERLAY_FORMAT_VERSION);
        assert!(overlay.cloth_asset.is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_project_rejects_future_version() {
        let dir = std::env::temp_dir().join("vulvatar_project_future_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("future.vvtproj");
        // Minimal future-version body — we only need format_version high
        // enough to trigger the early reject before deserialisation.
        let body = json!({"format_version": PROJECT_FORMAT_VERSION + 99});
        std::fs::write(&path, serde_json::to_string(&body).unwrap()).unwrap();
        let err = load_project(&path).expect_err("future project should reject");
        assert!(
            err.contains("newer VulVATAR"),
            "expected 'newer VulVATAR' message, got: {}",
            err
        );
        let _ = std::fs::remove_file(&path);
    }
}
