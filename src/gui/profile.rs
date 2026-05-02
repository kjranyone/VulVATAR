use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamProfile {
    pub name: String,
    pub tracking_mirror: bool,
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

    /// Offline capture preset: mirror disabled, lighting tuned for
    /// post-edit colour grading.
    pub fn recording_default() -> Self {
        Self {
            name: "Recording".to_string(),
            tracking_mirror: false,
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

    /// Live performance preset: mirror enabled, warmer ambient and
    /// softer key for on-stream readability.
    pub fn performance_default() -> Self {
        Self {
            name: "Performance".to_string(),
            tracking_mirror: true,
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

impl Default for ProfileLibrary {
    fn default() -> Self {
        Self::new()
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

impl Default for WindPresetLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod profile_roundtrip_tests {
    //! Catch silent default insertion when a future field lands. Each
    //! `StreamProfile` field is set to a value that differs from every
    //! preset *and* from `f32::default()` / `bool::default()` / etc, so
    //! a missing-field deserialization that filled in zero/false/0.0
    //! would surface as a value mismatch on round-trip.
    //!
    //! Listing every field by name in the constructor (rather than
    //! `..Default::default()`) is deliberate: when a new field is
    //! added to `StreamProfile`, this test fails to compile until the
    //! author updates the literal — a compile-time tripwire that
    //! forces the round-trip coverage to keep up with the schema.
    use super::*;

    fn make_tempdir(suffix: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("vulvatar_profile_roundtrip_{}_{}", suffix, nanos));
        std::fs::create_dir_all(&dir).expect("create tempdir");
        dir
    }

    /// Compare two f32s with a small tolerance — `serde_json` round-trips
    /// f32s through decimal strings, so non-exact values can drift by an
    /// ulp on parse-back. ε of 1e-6 is comfortably above that and well
    /// below any value the GUI surfaces.
    fn approx_eq(a: f32, b: f32, label: &str) {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-6,
            "{} drifted across round-trip: {} → {} (Δ {})",
            label,
            a,
            b,
            diff
        );
    }

    fn approx_eq_arr3(a: [f32; 3], b: [f32; 3], label: &str) {
        for i in 0..3 {
            approx_eq(a[i], b[i], &format!("{}[{}]", label, i));
        }
    }

    /// Construct a profile whose every field differs from every preset
    /// and from any obvious default — so a missing-field deserialization
    /// that filled in zero/false/0.0 would surface as a mismatch.
    fn make_distinctive_profile() -> StreamProfile {
        StreamProfile {
            name: "RoundTripFixture".to_string(),
            tracking_mirror: false,
            light_direction: [0.125, -0.5, 0.875],
            light_intensity: 1.5,
            ambient: [0.0625, 0.125, 0.1875],
            camera_fov: 47.5,
            outline_enabled: true,
            outline_width: 0.0625,
            outline_color: [0.75, 0.25, 0.5],
            output_sink_index: 2,
            output_resolution_index: 4,
            output_framerate_index: 5,
        }
    }

    #[test]
    fn profile_library_round_trips_every_field_through_export_import() {
        let dir = make_tempdir("export_import");
        let path = dir.join("profiles.json");

        let original = ProfileLibrary {
            profiles: vec![make_distinctive_profile()],
            // Deliberately not 0 — exercises the active_index branch
            // even though it's logically out-of-bounds. The on-disk
            // bytes must round-trip verbatim; `set_active` clamps at
            // call sites, not in the persistence layer.
            active_index: Some(7),
        };

        original.export_to_file(&path).expect("export profiles");
        let restored = ProfileLibrary::import_from_file(&path).expect("import profiles");

        assert_eq!(restored.active_index, original.active_index);
        assert_eq!(
            restored.profiles.len(),
            1,
            "profile count must round-trip"
        );

        let r = &restored.profiles[0];
        let o = &original.profiles[0];
        assert_eq!(r.name, o.name);
        assert_eq!(r.tracking_mirror, o.tracking_mirror);
        approx_eq_arr3(r.light_direction, o.light_direction, "light_direction");
        approx_eq(r.light_intensity, o.light_intensity, "light_intensity");
        approx_eq_arr3(r.ambient, o.ambient, "ambient");
        approx_eq(r.camera_fov, o.camera_fov, "camera_fov");
        assert_eq!(r.outline_enabled, o.outline_enabled);
        approx_eq(r.outline_width, o.outline_width, "outline_width");
        approx_eq_arr3(r.outline_color, o.outline_color, "outline_color");
        assert_eq!(r.output_sink_index, o.output_sink_index);
        assert_eq!(r.output_resolution_index, o.output_resolution_index);
        assert_eq!(r.output_framerate_index, o.output_framerate_index);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn profile_library_round_trips_multi_profile_with_active_index() {
        // A library with several profiles + a meaningful active_index
        // catches Vec ordering or index bookkeeping bugs that a
        // single-profile case can't surface.
        let dir = make_tempdir("multi");
        let path = dir.join("profiles.json");

        let original = ProfileLibrary {
            profiles: vec![
                StreamProfile {
                    name: "First".to_string(),
                    ..make_distinctive_profile()
                },
                StreamProfile {
                    name: "Second".to_string(),
                    camera_fov: 12.5,
                    ..make_distinctive_profile()
                },
                StreamProfile {
                    name: "Third".to_string(),
                    output_sink_index: 9,
                    ..make_distinctive_profile()
                },
            ],
            active_index: Some(1),
        };

        original.export_to_file(&path).expect("export");
        let restored = ProfileLibrary::import_from_file(&path).expect("import");

        assert_eq!(restored.profiles.len(), 3);
        assert_eq!(restored.active_index, Some(1));
        assert_eq!(restored.profiles[0].name, "First");
        assert_eq!(restored.profiles[1].name, "Second");
        approx_eq(
            restored.profiles[1].camera_fov,
            12.5,
            "Second.camera_fov",
        );
        assert_eq!(restored.profiles[2].name, "Third");
        assert_eq!(restored.profiles[2].output_sink_index, 9);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn import_rejects_missing_required_fields() {
        // Today there are no `#[serde(default)]` attributes on
        // StreamProfile, so a partial JSON must error rather than
        // silently fill in zeros. If a future change adds serde
        // defaults for backwards compat, this test will start passing
        // unexpectedly — that's the moment to revisit whether the
        // chosen default is what the GUI actually wants and to expand
        // the round-trip coverage above to pin the new defaults.
        let dir = make_tempdir("partial");
        let path = dir.join("partial.json");
        // Missing every field except `name`.
        std::fs::write(&path, r#"{"profiles":[{"name":"only-name"}],"active_index":null}"#)
            .expect("seed partial JSON");

        let result = ProfileLibrary::import_from_file(&path);
        assert!(
            result.is_err(),
            "import must reject a profile JSON missing required fields; \
             got Ok({:?}) — if a serde(default) was added, decide whether \
             the chosen default is what users actually want and update this test",
            result.ok()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}

