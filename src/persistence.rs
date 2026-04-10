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

#[derive(Serialize, Deserialize, Debug)]
pub struct TransformState {
    pub position: [f32; 3],
    pub rotation: [f32; 3],
    pub scale: f32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TrackingConfig {
    pub mirror: bool,
    pub smoothing_strength: f32,
    pub confidence_threshold: f32,
    pub hand_tracking_enabled: bool,
    pub face_tracking_enabled: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RenderingConfig {
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
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OutputConfig {
    pub sink_index: usize,
    pub resolution_index: usize,
    pub framerate_index: usize,
    pub has_alpha: bool,
    pub color_space_index: usize,
}

/// Full cloth overlay file for `.vvtcloth` files, including the complete ClothAsset data.
#[derive(Serialize, Deserialize, Debug)]
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

/// Write a project file to disk as pretty-printed JSON.
pub fn save_project(state: &ProjectState, path: &Path) -> Result<(), String> {
    let project = ProjectFile::from_state(state);
    let json = serde_json::to_string_pretty(&project).map_err(|e| e.to_string())?;
    std::fs::write(path, json).map_err(|e| e.to_string())?;
    println!("persistence: saved project to {}", path.display());
    Ok(())
}

/// Load a project file from disk and return the deserialized state plus
/// validation warnings. The caller is responsible for applying the returned
/// `ProjectState` to its own structures (e.g. `GuiApp`).
pub fn load_project(path: &Path) -> Result<(ProjectState, ProjectLoadWarnings), String> {
    let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let project: ProjectFile = serde_json::from_str(&data).map_err(|e| e.to_string())?;

    let warnings = ProjectLoadWarnings::default();
    let state = project.to_state();

    println!("persistence: loaded project from {}", path.display());
    for w in &warnings.warnings {
        eprintln!("persistence: WARNING: {}", w);
    }
    Ok((state, warnings))
}

/// Read a cloth overlay file from disk, deserializing the full ClothAsset data if present.
pub fn load_cloth_overlay(path: &Path) -> Result<ClothOverlayFile, String> {
    let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let overlay: ClothOverlayFile = serde_json::from_str(&data).map_err(|e| e.to_string())?;
    println!(
        "persistence: loaded cloth overlay '{}' from {} (has cloth_asset: {})",
        overlay.overlay_name,
        path.display(),
        overlay.cloth_asset.is_some(),
    );
    Ok(overlay)
}
