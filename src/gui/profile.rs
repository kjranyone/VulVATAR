#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::tracking::DEFAULT_CONFIDENCE_THRESHOLD;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamProfile {
    pub name: String,
    pub tracking_mirror: bool,
    pub smoothing_strength: f32,
    pub confidence_threshold: f32,
    pub light_direction: [f32; 3],
    pub light_intensity: f32,
    pub ambient: [f32; 3],
    pub camera_fov: f32,
    pub outline_enabled: bool,
    pub outline_width: f32,
    pub outline_color: [f32; 3],
    pub output_sink_index: usize,
    pub output_resolution_index: usize,
    pub output_framerate_index: usize,
}

impl StreamProfile {
    pub fn streaming_default() -> Self {
        Self {
            name: "Streaming".to_string(),
            tracking_mirror: true,
            smoothing_strength: 0.5,
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
            light_direction: [0.5, -0.7, 0.3],
            light_intensity: 1.0,
            ambient: [0.2, 0.2, 0.2],
            camera_fov: 45.0,
            outline_enabled: true,
            outline_width: 0.5,
            outline_color: [0.0, 0.0, 0.0],
            output_sink_index: 0,
            output_resolution_index: 0,
            output_framerate_index: 0,
        }
    }

    /// Loose-confidence preset for offline capture: a lower threshold
    /// admits more low-confidence keypoints, giving the post-edit pass
    /// extra raw signal to smooth or cherry-pick from. Live twitchiness
    /// is acceptable here because the recording will be re-cut.
    pub fn recording_default() -> Self {
        Self {
            name: "Recording".to_string(),
            tracking_mirror: false,
            smoothing_strength: 0.3,
            confidence_threshold: 0.2,
            light_direction: [0.4, -0.8, 0.5],
            light_intensity: 1.2,
            ambient: [0.3, 0.3, 0.3],
            camera_fov: 40.0,
            outline_enabled: false,
            outline_width: 0.0,
            outline_color: [0.0, 0.0, 0.0],
            output_sink_index: 3,
            output_resolution_index: 1,
            output_framerate_index: 1,
        }
    }

    /// Strict-confidence preset for live performance: a higher threshold
    /// drops noisy keypoints so the avatar stops moving rather than
    /// twitching when tracking degrades. Trades responsiveness for
    /// stability — a frozen bone reads better on stream than a wobbling
    /// one.
    pub fn performance_default() -> Self {
        Self {
            name: "Performance".to_string(),
            tracking_mirror: true,
            smoothing_strength: 0.7,
            confidence_threshold: 0.5,
            light_direction: [0.5, -0.7, 0.3],
            light_intensity: 0.8,
            ambient: [0.4, 0.4, 0.4],
            camera_fov: 50.0,
            outline_enabled: false,
            outline_width: 0.0,
            outline_color: [0.0, 0.0, 0.0],
            output_sink_index: 0,
            output_resolution_index: 0,
            output_framerate_index: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProfileLibrary {
    pub profiles: Vec<StreamProfile>,
    pub active_index: Option<usize>,
}

impl ProfileLibrary {
    pub fn new() -> Self {
        Self {
            profiles: vec![
                StreamProfile::streaming_default(),
                StreamProfile::recording_default(),
                StreamProfile::performance_default(),
            ],
            active_index: Some(0),
        }
    }

    pub fn active(&self) -> Option<&StreamProfile> {
        self.active_index.and_then(|i| self.profiles.get(i))
    }

    pub fn set_active(&mut self, index: usize) {
        if index < self.profiles.len() {
            self.active_index = Some(index);
        }
    }

    pub fn add(&mut self, profile: StreamProfile) {
        self.profiles.push(profile);
    }

    pub fn remove(&mut self, index: usize) {
        if index < self.profiles.len() {
            self.profiles.remove(index);
            if let Some(ai) = self.active_index {
                if index < ai {
                    self.active_index = Some(ai - 1);
                } else if ai >= self.profiles.len() {
                    self.active_index = if self.profiles.is_empty() {
                        None
                    } else {
                        Some(self.profiles.len() - 1)
                    };
                }
            }
        }
    }

    pub fn export_to_file(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("failed to serialize profiles: {}", e))?;
        std::fs::write(path, json).map_err(|e| format!("failed to write profiles file: {}", e))
    }

    pub fn import_from_file(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read profiles file: {}", e))?;
        serde_json::from_str(&content).map_err(|e| format!("failed to parse profiles JSON: {}", e))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WindPreset {
    pub name: String,
    pub direction: [f32; 3],
    pub strength: f32,
    pub gust_frequency: f32,
    pub gust_amplitude: f32,
}

impl WindPreset {
    pub fn calm() -> Self {
        Self {
            name: "Calm".to_string(),
            direction: [1.0, 0.0, 0.0],
            strength: 0.0,
            gust_frequency: 0.0,
            gust_amplitude: 0.0,
        }
    }

    pub fn gentle_breeze() -> Self {
        Self {
            name: "Gentle Breeze".to_string(),
            direction: [1.0, 0.0, 0.3],
            strength: 2.0,
            gust_frequency: 0.5,
            gust_amplitude: 1.0,
        }
    }

    pub fn strong_wind() -> Self {
        Self {
            name: "Strong Wind".to_string(),
            direction: [1.0, 0.0, 0.0],
            strength: 8.0,
            gust_frequency: 1.5,
            gust_amplitude: 4.0,
        }
    }

    pub fn storm() -> Self {
        Self {
            name: "Storm".to_string(),
            direction: [0.8, 0.2, 0.5],
            strength: 15.0,
            gust_frequency: 3.0,
            gust_amplitude: 8.0,
        }
    }

    pub fn directional(direction: [f32; 3], strength: f32) -> Self {
        Self {
            name: format!("Custom ({:.1} m/s)", strength),
            direction,
            strength,
            gust_frequency: 0.0,
            gust_amplitude: 0.0,
        }
    }

    pub fn sample_at(&self, time: f32) -> [f32; 3] {
        let gust = if self.gust_frequency > 0.0 {
            (time * self.gust_frequency * std::f32::consts::TAU).sin() * self.gust_amplitude
        } else {
            0.0
        };
        let total_strength = self.strength + gust;
        [
            self.direction[0] * total_strength,
            self.direction[1] * total_strength,
            self.direction[2] * total_strength,
        ]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WindPresetLibrary {
    pub presets: Vec<WindPreset>,
    pub active_index: Option<usize>,
}

impl WindPresetLibrary {
    pub fn new() -> Self {
        Self {
            presets: vec![
                WindPreset::calm(),
                WindPreset::gentle_breeze(),
                WindPreset::strong_wind(),
                WindPreset::storm(),
            ],
            active_index: Some(0),
        }
    }

    pub fn active(&self) -> Option<&WindPreset> {
        self.active_index.and_then(|i| self.presets.get(i))
    }

    pub fn set_active(&mut self, index: usize) {
        if index < self.presets.len() {
            self.active_index = Some(index);
        }
    }

    pub fn add(&mut self, preset: WindPreset) {
        self.presets.push(preset);
    }

    pub fn remove(&mut self, index: usize) {
        if index < self.presets.len() {
            self.presets.remove(index);
            if let Some(ai) = self.active_index {
                if index < ai {
                    self.active_index = Some(ai - 1);
                } else if ai >= self.presets.len() {
                    self.active_index = if self.presets.is_empty() {
                        None
                    } else {
                        Some(self.presets.len() - 1)
                    };
                }
            }
        }
    }
}
