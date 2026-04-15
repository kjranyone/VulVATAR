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

    // Transform
    pub transform_position: [f32; 3],
    pub transform_rotation: [f32; 3],
    pub transform_scale: f32,

    pub camera_orbit_yaw: f32,
    pub camera_orbit_pitch: f32,
    pub camera_orbit_pan: [f32; 2],
    pub camera_orbit_distance: f32,

    // Tracking config
    pub tracking_mirror: bool,
    pub smoothing_strength: f32,
    pub confidence_threshold: f32,
    pub hand_tracking_enabled: bool,
    pub face_tracking_enabled: bool,

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

    // Output config
    pub output_sink_index: usize,
    pub output_resolution_index: usize,
    pub output_framerate_index: usize,
    pub output_has_alpha: bool,
    pub output_color_space_index: usize,
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

    pub tracking: TrackingConfig,
    pub rendering: RenderingConfig,
    pub output: OutputConfig,
}

/// Warnings produced during project reload validation.
#[derive(Debug, Default)]
pub struct ProjectLoadWarnings {
    pub warnings: Vec<String>,
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
    #[serde(default)]
    pub mirror: bool,
    #[serde(default)]
    pub smoothing_strength: f32,
    #[serde(default)]
    pub confidence_threshold: f32,
    #[serde(default)]
    pub hand_tracking_enabled: bool,
    #[serde(default)]
    pub face_tracking_enabled: bool,
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
}

const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

fn app_tag() -> String {
    format!("VulVATAR {}", APP_VERSION)
}

impl ProjectFile {
    /// Snapshot the current project state into a serializable project file.
    pub fn from_state(state: &ProjectState) -> Self {
        Self {
            format_version: 1,
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

            tracking: TrackingConfig {
                mirror: state.tracking_mirror,
                smoothing_strength: state.smoothing_strength,
                confidence_threshold: state.confidence_threshold,
                hand_tracking_enabled: state.hand_tracking_enabled,
                face_tracking_enabled: state.face_tracking_enabled,
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
            },
            output: OutputConfig {
                sink_index: state.output_sink_index,
                resolution_index: state.output_resolution_index,
                framerate_index: state.output_framerate_index,
                has_alpha: state.output_has_alpha,
                color_space_index: state.output_color_space_index,
            },
        }
    }

    /// Convert a loaded project file back into a `ProjectState`.
    pub fn to_state(&self) -> ProjectState {
        ProjectState {
            avatar_source_path: self.avatar_source_path.clone(),
            avatar_source_hash: self.avatar_source_hash.clone(),
            active_overlay_path: self.active_overlay_path.clone(),

            transform_position: self.avatar_transform.position,
            transform_rotation: self.avatar_transform.rotation,
            transform_scale: self.avatar_transform.scale,

            camera_orbit_yaw: self.camera_orbit_yaw,
            camera_orbit_pitch: self.camera_orbit_pitch,
            camera_orbit_pan: self.camera_orbit_pan,
            camera_orbit_distance: self.camera_orbit_distance,

            tracking_mirror: self.tracking.mirror,
            smoothing_strength: self.tracking.smoothing_strength,
            confidence_threshold: self.tracking.confidence_threshold,
            hand_tracking_enabled: self.tracking.hand_tracking_enabled,
            face_tracking_enabled: self.tracking.face_tracking_enabled,

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

            output_sink_index: self.output.sink_index,
            output_resolution_index: self.output.resolution_index,
            output_framerate_index: self.output.framerate_index,
            output_has_alpha: self.output.has_alpha,
            output_color_space_index: self.output.color_space_index,
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
pub fn load_project(path: &Path) -> Result<(ProjectState, ProjectLoadWarnings), String> {
    let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let project: ProjectFile = serde_json::from_str(&data).map_err(|e| e.to_string())?;

    let mut warnings = ProjectLoadWarnings::default();

    if project.format_version != 1 {
        warnings.warnings.push(format!(
            "unsupported format_version {}, expected 1",
            project.format_version
        ));
    }
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
pub fn load_cloth_overlay(path: &Path) -> Result<ClothOverlayFile, String> {
    let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let overlay: ClothOverlayFile = serde_json::from_str(&data).map_err(|e| e.to_string())?;
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
        format_version: 1,
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
            format_version: 1,
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
    if snapshot.format_version != 1 {
        return Err(format!(
            "unsupported recovery format_version {}",
            snapshot.format_version
        ));
    }
    if let Some(ref project) = snapshot.project {
        if project.format_version != 1 {
            return Err(format!(
                "unsupported project format_version {} in recovery",
                project.format_version
            ));
        }
    }
    if let Some(ref overlay) = snapshot.overlay {
        if overlay.format_version != 1 {
            return Err(format!(
                "unsupported overlay format_version {} in recovery",
                overlay.format_version
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
