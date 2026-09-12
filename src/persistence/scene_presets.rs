//! Saved scene presets — lighting / camera / rendering snapshots
//! persisted as one JSON list under the app-data dir.

use log::{error, info};
use serde::{Deserialize, Serialize};

use super::{app_data_dir, app_tag, atomic_write, backup_path_for, load_or_backup};

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
