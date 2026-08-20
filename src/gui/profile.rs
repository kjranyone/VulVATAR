use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::tracking::PoseCalibration;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamProfile {
    pub name: String,
    pub tracking_mirror: bool,
    pub light_direction: [f32; 3],
    pub light_intensity: f32,
    pub ambient: [f32; 3],
    pub camera_fov: f32,
    pub output_sink_index: usize,
    pub output_resolution_index: usize,
    pub output_framerate_index: usize,
    /// Output alpha / colour-space / MSAA travel with the profile so a
    /// profile switch can't leave the Output panel in a mixed state
    /// (sink + resolution + fps from the new profile, alpha / colour
    /// space / MSAA lingering from the previous one). `serde(default)`
    /// keeps pre-existing `profiles.json` files loading; the defaults
    /// match the GUI's initial values.
    #[serde(default = "default_output_has_alpha")]
    pub output_has_alpha: bool,
    #[serde(default)]
    pub output_color_space_index: usize,
    #[serde(default)]
    pub output_msaa_index: usize,
    /// Per-setup pose calibration. Lives on the profile (rather than
    /// the project file) so the user's "home desk" setup vs.
    /// "office desk" setup can each carry its own anchor / range
    /// values and switching profiles snaps the depth pipeline to
    /// the right baseline. `None` until the user runs `Calibrate
    /// Pose ▼` while this profile is active. See
    /// `docs/calibration-ux.md`.
    #[serde(default)]
    pub pose_calibration: Option<PoseCalibration>,
}

fn default_output_has_alpha() -> bool {
    true
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
            output_sink_index: 0,
            output_resolution_index: 0,
            output_framerate_index: 0,
            output_has_alpha: true,
            output_color_space_index: 0,
            output_msaa_index: 0,
            pose_calibration: None,
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
            output_sink_index: 3,
            output_resolution_index: 1,
            output_framerate_index: 1,
            output_has_alpha: true,
            output_color_space_index: 0,
            output_msaa_index: 0,
            pose_calibration: None,
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
            output_sink_index: 0,
            output_resolution_index: 0,
            output_framerate_index: 0,
            output_has_alpha: true,
            output_color_space_index: 0,
            output_msaa_index: 0,
            pose_calibration: None,
        }
    }
}

/// Why a [`ProfileLibrary::rename_at`] was rejected. Carried back to
/// the rename dialog so it can show a specific message instead of a
/// generic failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenameError {
    Empty,
    Duplicate,
    OutOfBounds,
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

    /// Whether `remove` may be called at all: the library must never
    /// become empty (calibration and output settings live on the
    /// active profile — an empty library would leave the app with no
    /// place to persist them).
    pub fn can_remove(&self) -> bool {
        self.profiles.len() > 1
    }

    /// `base` if unused, else `base (2)`, `base (3)`, … — the pattern
    /// every desktop app uses for duplicated documents, so a duplicate
    /// never silently collides with an existing profile name.
    pub fn unique_name(&self, base: &str) -> String {
        let taken = |name: &str| self.profiles.iter().any(|p| p.name == name);
        if !taken(base) {
            return base.to_string();
        }
        let mut n = 2usize;
        loop {
            let candidate = format!("{} ({})", base, n);
            if !taken(&candidate) {
                return candidate;
            }
            n += 1;
        }
    }

    /// Deep-copy the profile at `index` (calibration included — a
    /// duplicate exists to tweak a working setup, so it starts from
    /// the same baseline), append it under a uniquified name, and
    /// return the new profile's index.
    pub fn duplicate_at(&mut self, index: usize) -> Option<usize> {
        let source = self.profiles.get(index)?.clone();
        let mut copy = source;
        copy.name = self.unique_name(&copy.name);
        self.profiles.push(copy);
        Some(self.profiles.len() - 1)
    }

    /// Rename the profile at `index`. Rejects empty / whitespace-only
    /// names and names already used by *another* profile (renaming a
    /// profile to its own current name is a no-op success).
    pub fn rename_at(&mut self, index: usize, new_name: &str) -> Result<(), RenameError> {
        let trimmed = new_name.trim();
        if trimmed.is_empty() {
            return Err(RenameError::Empty);
        }
        if self
            .profiles
            .iter()
            .enumerate()
            .any(|(i, p)| i != index && p.name == trimmed)
        {
            return Err(RenameError::Duplicate);
        }
        match self.profiles.get_mut(index) {
            Some(p) => {
                p.name = trimmed.to_string();
                Ok(())
            }
            None => Err(RenameError::OutOfBounds),
        }
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

#[cfg(test)]
mod profile_management_tests {
    //! State-transition coverage for the profile-management UI's data
    //! layer: duplicate naming, rename validation, delete guards, and
    //! active-index bookkeeping — the operations the top-bar menu
    //! drives. Kept UI-free so they run without an egui context.
    use super::*;

    fn library_with(names: &[&str], active: usize) -> ProfileLibrary {
        ProfileLibrary {
            profiles: names
                .iter()
                .map(|n| StreamProfile {
                    name: n.to_string(),
                    ..StreamProfile::streaming_default()
                })
                .collect(),
            active_index: Some(active),
        }
    }

    #[test]
    fn duplicate_appends_with_uniquified_name_and_returns_new_index() {
        let mut lib = library_with(&["Desk", "Desk (2)"], 0);
        let idx = lib.duplicate_at(0).expect("duplicate");
        assert_eq!(idx, 2);
        assert_eq!(lib.profiles[2].name, "Desk (3)");
        // Duplicating the duplicate keeps walking the suffix.
        let idx2 = lib.duplicate_at(2).expect("duplicate again");
        assert_eq!(lib.profiles[idx2].name, "Desk (3) (2)");
    }

    #[test]
    fn duplicate_carries_calibration_with_it() {
        let mut lib = library_with(&["Desk"], 0);
        lib.profiles[0].pose_calibration = Some(crate::tracking::PoseCalibration {
            mode: crate::tracking::CalibrationMode::UpperBody,
            captured_at: "2026-01-15T12:34:56Z".to_string(),
            captured_at_unix: 1_768_654_496,
            frame_count: 12,
            anchor_x: 0.0,
            anchor_y: 0.0,
            anchor_depth_m: Some(1.2),
            confidence: 0.9,
            anchor_depth_jitter_m: None,
            shoulder_span_m: Some(0.4),
            x_range_observed: None,
            z_range_observed: None,
            neutral_expressions: Vec::new(),
            neutral_face_ypr_mesh: None,
            neutral_face_ypr_body: None,
            neutral_body_yaw: None,
        });
        let idx = lib.duplicate_at(0).expect("duplicate");
        assert!(
            lib.profiles[idx].pose_calibration.is_some(),
            "a duplicate exists to tweak a working setup — it must start \
             from the same calibration baseline"
        );
    }

    #[test]
    fn rename_rejects_empty_and_duplicate_but_allows_self() {
        let mut lib = library_with(&["Desk", "Sofa"], 0);
        assert_eq!(lib.rename_at(0, "   "), Err(RenameError::Empty));
        assert_eq!(lib.rename_at(0, "Sofa"), Err(RenameError::Duplicate));
        assert_eq!(lib.rename_at(0, "Desk"), Ok(()), "self-rename is a no-op success");
        assert_eq!(lib.rename_at(0, "  Studio "), Ok(()));
        assert_eq!(lib.profiles[0].name, "Studio", "rename trims whitespace");
        assert_eq!(lib.rename_at(9, "X"), Err(RenameError::OutOfBounds));
    }

    #[test]
    fn last_profile_cannot_be_removed() {
        let lib = library_with(&["Only"], 0);
        assert!(!lib.can_remove());
        // Even if a caller ignores the guard, the library must survive
        // with its single profile intact — remove() itself stays safe.
        assert_eq!(lib.profiles.len(), 1);
        let mut two = library_with(&["A", "B"], 1);
        assert!(two.can_remove());
        two.remove(0);
        assert_eq!(two.profiles.len(), 1);
        assert_eq!(
            two.active_index,
            Some(0),
            "removing below the active index shifts it down"
        );
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
            output_sink_index: 2,
            output_resolution_index: 4,
            output_framerate_index: 5,
            // All three differ from the serde defaults (true / 0 / 0) so
            // a deserialization that silently fell back to the defaults
            // is caught by the assertions below.
            output_has_alpha: false,
            output_color_space_index: 1,
            output_msaa_index: 3,
            // Distinctive non-default values so a missing-field
            // deserialization that filled in `None` would be caught
            // by the assertions below. The captured-at* / frame_count
            // / anchor / depth fields are all set to specific values
            // that don't match any preset constructor.
            pose_calibration: Some(crate::tracking::PoseCalibration {
                mode: crate::tracking::CalibrationMode::UpperBody,
                captured_at: "2026-01-15T12:34:56Z".to_string(),
                captured_at_unix: 1_768_654_496,
                frame_count: 47,
                anchor_x: 0.125,
                anchor_y: -0.0625,
                anchor_depth_m: Some(1.875),
                confidence: 0.875,
                anchor_depth_jitter_m: Some(0.025),
                shoulder_span_m: Some(0.412),
                x_range_observed: Some(0.6),
                z_range_observed: Some(0.35),
                // Tiny 2×2 template is enough to round-trip-check
                // the serde plumbing; the inference-time path uses
                // 32×32 grids in production.
                neutral_expressions: Vec::new(),
                neutral_face_ypr_mesh: Some([-0.61, 0.14, 0.02]),
                neutral_face_ypr_body: Some([-0.55, 0.31, 0.01]),
                neutral_body_yaw: Some(-0.42),
            }),
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
        assert_eq!(r.output_sink_index, o.output_sink_index);
        assert_eq!(r.output_resolution_index, o.output_resolution_index);
        assert_eq!(r.output_framerate_index, o.output_framerate_index);
        assert_eq!(r.output_has_alpha, o.output_has_alpha);
        assert_eq!(r.output_color_space_index, o.output_color_space_index);
        assert_eq!(r.output_msaa_index, o.output_msaa_index);

        // pose_calibration must round-trip every field — the per-axis
        // anchor + depth + jitter + range values all drive solver
        // behaviour, so a silent default insertion here would be a
        // calibration-loss bug across an app restart.
        match (&r.pose_calibration, &o.pose_calibration) {
            (Some(rc), Some(oc)) => {
                assert_eq!(rc.mode, oc.mode);
                assert_eq!(rc.captured_at, oc.captured_at);
                assert_eq!(rc.captured_at_unix, oc.captured_at_unix);
                assert_eq!(rc.frame_count, oc.frame_count);
                approx_eq(rc.anchor_x, oc.anchor_x, "pose.anchor_x");
                approx_eq(rc.anchor_y, oc.anchor_y, "pose.anchor_y");
                match (rc.anchor_depth_m, oc.anchor_depth_m) {
                    (Some(a), Some(b)) => approx_eq(a, b, "pose.anchor_depth_m"),
                    (None, None) => {}
                    _ => panic!("pose.anchor_depth_m presence drifted across round-trip"),
                }
                approx_eq(rc.confidence, oc.confidence, "pose.confidence");
                match (rc.anchor_depth_jitter_m, oc.anchor_depth_jitter_m) {
                    (Some(a), Some(b)) => approx_eq(a, b, "pose.anchor_depth_jitter_m"),
                    (None, None) => {}
                    _ => panic!("pose.anchor_depth_jitter_m presence drifted across round-trip"),
                }
                match (rc.x_range_observed, oc.x_range_observed) {
                    (Some(a), Some(b)) => approx_eq(a, b, "pose.x_range_observed"),
                    (None, None) => {}
                    _ => panic!("pose.x_range_observed presence drifted across round-trip"),
                }
                match (rc.z_range_observed, oc.z_range_observed) {
                    (Some(a), Some(b)) => approx_eq(a, b, "pose.z_range_observed"),
                    (None, None) => {}
                    _ => panic!("pose.z_range_observed presence drifted across round-trip"),
                }
                match (rc.neutral_face_ypr_mesh, oc.neutral_face_ypr_mesh) {
                    (Some(a), Some(b)) => approx_eq_arr3(a, b, "pose.neutral_face_ypr_mesh"),
                    (None, None) => {}
                    _ => panic!("pose.neutral_face_ypr_mesh presence drifted across round-trip"),
                }
                match (rc.neutral_face_ypr_body, oc.neutral_face_ypr_body) {
                    (Some(a), Some(b)) => approx_eq_arr3(a, b, "pose.neutral_face_ypr_body"),
                    (None, None) => {}
                    _ => panic!("pose.neutral_face_ypr_body presence drifted across round-trip"),
                }
                match (rc.neutral_body_yaw, oc.neutral_body_yaw) {
                    (Some(a), Some(b)) => approx_eq(a, b, "pose.neutral_body_yaw"),
                    (None, None) => {}
                    _ => panic!("pose.neutral_body_yaw presence drifted across round-trip"),
                }
            }
            (None, None) => {}
            _ => panic!("pose_calibration presence drifted across round-trip"),
        }

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
        // The core fields (tracking_mirror, lighting, fov, sink /
        // resolution / framerate indices) carry no `#[serde(default)]`,
        // so a partial JSON must error rather than silently fill in
        // zeros. Only later-added fields (pose_calibration, the output
        // alpha / colour-space / MSAA trio) default for backwards
        // compat. If a future change adds defaults to the core fields,
        // this test will start passing unexpectedly — that's the moment
        // to revisit whether the chosen default is what the GUI
        // actually wants and to expand the round-trip coverage above.
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

