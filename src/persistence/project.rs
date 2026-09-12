//! The `.vvtproj` project file: ProjectState (the pure-data snapshot the
//! GUI diffs for dirty-tracking), ProjectFile (the on-disk schema),
//! save/load with the version-migration chain, the `.vvtcloth` overlay
//! format, and the legacy camera table mappings for pre-resolution-index
//! saves. The config blocks' DTO forms live in `project_dto`.

use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::path::Path;

use super::atomic_io::atomic_write;
use super::project_dto::{
    dto_to_pose_calibration, pose_calibration_to_dto, LipSyncConfig, OutputConfig,
    RenderingConfig, TrackingConfig, TransformState,
};
use super::app_settings::SettingsConfig;

/// Pure-data snapshot of the GUI/app state needed for project serialization.
/// This struct lives here so that `persistence` never imports `gui::GuiApp`,
/// breaking the former circular dependency.
// `PartialEq` powers the derived-dirty probe (`GuiApp::refresh_project_dirty`):
// the GUI compares the current snapshot against the last persisted baseline
// instead of relying on 70+ hand-placed `project_dirty = true` sites.
#[derive(Clone, Debug, PartialEq)]
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
    /// Capture format as REAL values (not combo positions). Persisting
    /// the combo index meant reordering / inserting an entry in the
    /// resolution dropdown silently changed every saved project's
    /// capture format; values survive UI evolution. The GUI converts
    /// to/from its combo position at the snapshot/apply boundary.
    pub camera_capture_width: u32,
    pub camera_capture_height: u32,
    pub camera_capture_fps: u32,
    pub hand_tracking_enabled: bool,
    pub face_tracking_enabled: bool,
    pub lower_body_tracking_enabled: bool,
    pub root_translation_enabled: bool,
    pub fade_on_tracking_loss: bool,
    pub force_cpu_inference: bool,
    pub yolox_enabled: bool,
    pub show_camera_wipe: bool,
    pub show_detection_annotations: bool,
    // Pose-solver smoothing (Advanced smoothing inspector section). The
    // remaining `TrackingSmoothingParams` field (`stale_timeout_nanos`) is
    // not user-editable, so it is not persisted — it always reloads at its
    // default.
    pub smoothing_rotation_blend: f32,
    pub smoothing_expression_blend: f32,
    pub smoothing_face_confidence: f32,
    /// Captured pose-calibration reference. Wired through `Application`'s
    /// `tracking_calibration.pose` on load; the GUI reads it back through
    /// the same field so the inspector status line ("Calibrated 2 min
    /// ago") survives round-tripping. See `docs/calibration-ux.md`.
    pub pose_calibration: Option<crate::tracking::PoseCalibration>,

    // Rendering config
    pub material_mode_index: usize,
    pub light_direction: [f32; 3],
    pub light_intensity: f32,
    pub ambient: [f32; 3],
    pub camera_fov: f32,
    pub background_color: [f32; 3],
    pub transparent_background: bool,
    pub toggle_spring: bool,
    /// Spring-bone user tuning (see `simulation::spring::SpringTuning`).
    pub spring_sway_scale: f32,
    pub spring_gravity_offset: f32,
    pub spring_natural_gravity: bool,
    /// Scene gravity (see `simulation::SceneGravity`).
    pub scene_gravity_direction: [f32; 3],
    pub scene_gravity_strength: f32,
    pub toggle_cloth: bool,
    pub toggle_collision_debug: bool,
    pub toggle_skeleton_debug: bool,
    pub alpha_preview: bool,
    pub bloom_enabled: bool,
    pub bloom_intensity: f32,
    pub bloom_threshold: f32,
    pub bg_enabled: bool,
    pub bg_intensity: f32,
    pub bg_speed: f32,
    pub bg_scale: f32,
    pub bg_reactivity: f32,
    pub bg_color_a: [f32; 3],
    pub bg_color_b: [f32; 3],

    // Lip sync config
    pub lipsync_enabled: bool,
    pub lipsync_mic_device_index: usize,
    pub lipsync_volume_threshold: f32,
    pub lipsync_smoothing: f32,
    /// Mouth viseme driver: 0=Audio, 1=Image, 2=Both. See `MouthSource`.
    pub mouth_source_index: usize,

    // Output config
    pub output_sink_index: usize,
    pub output_resolution_index: usize,
    pub output_framerate_index: usize,
    pub output_has_alpha: bool,
    pub output_color_space_index: usize,
    /// Anti-aliasing (MSAA) level index: 0=Off, 1=2x, 2=4x, 3=8x.
    pub output_msaa_index: usize,
}

pub(super) fn default_spring_sway_scale() -> f32 {
    1.0
}

/// Neutral 1.0 multiplier for scale-type settings loaded from
/// pre-feature projects (kept separate from `default_spring_sway_scale`
/// so the two defaults can never move together by accident).
pub(super) fn default_unit_scale() -> f32 {
    1.0
}

pub(super) fn default_gravity_direction() -> [f32; 3] {
    [0.0, -1.0, 0.0]
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
    /// Legacy slot: app-level settings (UI language, viewport input
    /// sensitivities, consents) used to ride the project file, which
    /// made them follow the *scene* instead of the *user* — opening
    /// someone else's project switched your UI language, and a session
    /// that only touched the Settings pane never persisted at all.
    /// They now live in `settings.json` ([`AppSettings`]). Kept as an
    /// `Option` so pre-split project files still parse (and can be
    /// migrated once at startup); never written back.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub settings: Option<SettingsConfig>,
}

/// Warnings produced during project reload validation.
#[derive(Debug, Default)]
pub struct ProjectLoadWarnings {
    pub warnings: Vec<String>,
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

pub(super) fn app_tag() -> String {
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
pub(super) fn migrate_chain<StepFn>(
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
pub(super) fn migrate_cloth_overlay_json(
    json: serde_json::Value,
    from_version: u32,
) -> Result<serde_json::Value, String> {
    migrate_chain(json, from_version, OVERLAY_FORMAT_VERSION, step_overlay)
}

fn step_overlay(_json: serde_json::Value, version: u32) -> Result<serde_json::Value, String> {
    // Future migrators land here:
    //   1 => v1_to_v2(json),
    let v = version;
    Err(format!(
        "no migrator registered for cloth overlay v{} → v{}",
        v,
        v + 1
    ))
}

/// See [`migrate_cloth_overlay_json`]. Same shape for project files.
pub(super) fn migrate_project_json(
    json: serde_json::Value,
    from_version: u32,
) -> Result<serde_json::Value, String> {
    migrate_chain(json, from_version, PROJECT_FORMAT_VERSION, step_project)
}

fn step_project(_json: serde_json::Value, version: u32) -> Result<serde_json::Value, String> {
    let v = version;
    Err(format!(
        "no migrator registered for project v{} → v{}",
        v,
        v + 1
    ))
}

pub(super) fn read_format_version(json: &serde_json::Value) -> u32 {
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
                camera_resolution_index: legacy_camera_resolution_index(
                    state.camera_capture_width,
                    state.camera_capture_height,
                ),
                camera_framerate_index: legacy_camera_fps_index(state.camera_capture_fps),
                camera_capture_width: Some(state.camera_capture_width),
                camera_capture_height: Some(state.camera_capture_height),
                camera_capture_fps: Some(state.camera_capture_fps),
                hand_tracking_enabled: state.hand_tracking_enabled,
                face_tracking_enabled: state.face_tracking_enabled,
                lower_body_tracking_enabled: state.lower_body_tracking_enabled,
                root_translation_enabled: state.root_translation_enabled,
                fade_on_tracking_loss: state.fade_on_tracking_loss,
                force_cpu_inference: state.force_cpu_inference,
                yolox_enabled: state.yolox_enabled,
                show_camera_wipe: state.show_camera_wipe,
                show_detection_annotations: state.show_detection_annotations,
                smoothing_rotation_blend: state.smoothing_rotation_blend,
                smoothing_expression_blend: state.smoothing_expression_blend,
                smoothing_face_confidence: state.smoothing_face_confidence,
                pose_calibration: state.pose_calibration.as_ref().map(pose_calibration_to_dto),
            },
            rendering: RenderingConfig {
                material_mode_index: state.material_mode_index,
                light_direction: state.light_direction,
                light_intensity: state.light_intensity,
                ambient: state.ambient,
                camera_fov: state.camera_fov,
                background_color: state.background_color,
                transparent_background: state.transparent_background,
                toggle_spring: state.toggle_spring,
                spring_sway_scale: state.spring_sway_scale,
                spring_gravity_offset: state.spring_gravity_offset,
                spring_natural_gravity: state.spring_natural_gravity,
                scene_gravity_direction: state.scene_gravity_direction,
                scene_gravity_strength: state.scene_gravity_strength,
                toggle_cloth: state.toggle_cloth,
                toggle_collision_debug: state.toggle_collision_debug,
                toggle_skeleton_debug: state.toggle_skeleton_debug,
                alpha_preview: state.alpha_preview,
                bloom_enabled: state.bloom_enabled,
                bloom_intensity: state.bloom_intensity,
                bloom_threshold: state.bloom_threshold,
                bg_enabled: state.bg_enabled,
                bg_intensity: state.bg_intensity,
                bg_speed: state.bg_speed,
                bg_scale: state.bg_scale,
                bg_reactivity: state.bg_reactivity,
                bg_color_a: state.bg_color_a,
                bg_color_b: state.bg_color_b,
            },
            lipsync: LipSyncConfig {
                enabled: state.lipsync_enabled,
                mic_device_index: state.lipsync_mic_device_index,
                volume_threshold: state.lipsync_volume_threshold,
                smoothing: state.lipsync_smoothing,
                mouth_source_index: state.mouth_source_index,
            },
            output: OutputConfig {
                sink_index: state.output_sink_index,
                resolution_index: state.output_resolution_index,
                framerate_index: state.output_framerate_index,
                has_alpha: state.output_has_alpha,
                color_space_index: state.output_color_space_index,
                msaa_index: state.output_msaa_index,
            },
            // App-level settings are not part of the project any more —
            // see the field's doc comment.
            settings: None,
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
            camera_capture_width: self.tracking.camera_capture_width.unwrap_or_else(|| {
                legacy_camera_resolution(self.tracking.camera_resolution_index).0
            }),
            camera_capture_height: self.tracking.camera_capture_height.unwrap_or_else(|| {
                legacy_camera_resolution(self.tracking.camera_resolution_index).1
            }),
            camera_capture_fps: self
                .tracking
                .camera_capture_fps
                .unwrap_or_else(|| legacy_camera_fps(self.tracking.camera_framerate_index)),
            hand_tracking_enabled: self.tracking.hand_tracking_enabled,
            face_tracking_enabled: self.tracking.face_tracking_enabled,
            lower_body_tracking_enabled: self.tracking.lower_body_tracking_enabled,
            root_translation_enabled: self.tracking.root_translation_enabled,
            fade_on_tracking_loss: self.tracking.fade_on_tracking_loss,
            force_cpu_inference: self.tracking.force_cpu_inference,
            yolox_enabled: self.tracking.yolox_enabled,
            show_camera_wipe: self.tracking.show_camera_wipe,
            show_detection_annotations: self.tracking.show_detection_annotations,
            smoothing_rotation_blend: self.tracking.smoothing_rotation_blend,
            smoothing_expression_blend: self.tracking.smoothing_expression_blend,
            smoothing_face_confidence: self.tracking.smoothing_face_confidence,
            pose_calibration: self
                .tracking
                .pose_calibration
                .as_ref()
                .and_then(dto_to_pose_calibration),

            material_mode_index: self.rendering.material_mode_index,
            light_direction: self.rendering.light_direction,
            light_intensity: self.rendering.light_intensity,
            ambient: self.rendering.ambient,
            camera_fov: self.rendering.camera_fov,
            background_color: self.rendering.background_color,
            transparent_background: self.rendering.transparent_background,
            toggle_spring: self.rendering.toggle_spring,
            spring_sway_scale: self.rendering.spring_sway_scale,
            spring_gravity_offset: self.rendering.spring_gravity_offset,
            spring_natural_gravity: self.rendering.spring_natural_gravity,
            scene_gravity_direction: self.rendering.scene_gravity_direction,
            scene_gravity_strength: self.rendering.scene_gravity_strength,
            toggle_cloth: self.rendering.toggle_cloth,
            toggle_collision_debug: self.rendering.toggle_collision_debug,
            toggle_skeleton_debug: self.rendering.toggle_skeleton_debug,
            alpha_preview: self.rendering.alpha_preview,
            bloom_enabled: self.rendering.bloom_enabled,
            bloom_intensity: self.rendering.bloom_intensity,
            bloom_threshold: self.rendering.bloom_threshold,
            bg_enabled: self.rendering.bg_enabled,
            bg_intensity: self.rendering.bg_intensity,
            bg_speed: self.rendering.bg_speed,
            bg_scale: self.rendering.bg_scale,
            bg_reactivity: self.rendering.bg_reactivity,
            bg_color_a: self.rendering.bg_color_a,
            bg_color_b: self.rendering.bg_color_b,

            lipsync_enabled: self.lipsync.enabled,
            lipsync_mic_device_index: self.lipsync.mic_device_index,
            lipsync_volume_threshold: self.lipsync.volume_threshold,
            lipsync_smoothing: self.lipsync.smoothing,
            mouth_source_index: self.lipsync.mouth_source_index,

            output_sink_index: self.output.sink_index,
            output_resolution_index: self.output.resolution_index,
            output_framerate_index: self.output.framerate_index,
            output_has_alpha: self.output.has_alpha,
            output_color_space_index: self.output.color_space_index,
            output_msaa_index: self.output.msaa_index,
        }
    }
}

/// Write a project file to disk as pretty-printed JSON using atomic write
/// (write to temp file, then rename).
/// FROZEN mapping of the camera-resolution combo positions as they
/// were when projects persisted raw indices (0 = 640×480, 1 = 1280×720,
/// 2 = 1920×1080). Only for migrating old files and for writing the
/// best-effort legacy index alongside the value fields — the live GUI
/// combo mapping lives in `tracking::camera_resolution_for_index` and may
/// grow entries freely without touching this table.
fn legacy_camera_resolution(index: usize) -> (u32, u32) {
    match index {
        1 => (1280, 720),
        2 => (1920, 1080),
        _ => (640, 480),
    }
}

/// Reverse of [`legacy_camera_resolution`]; unknown values map to the
/// legacy default slot (0) so a downgraded build still opens the file.
fn legacy_camera_resolution_index(width: u32, height: u32) -> usize {
    match (width, height) {
        (1280, 720) => 1,
        (1920, 1080) => 2,
        _ => 0,
    }
}

/// FROZEN legacy fps combo mapping (0 = 30, 1 = 60).
fn legacy_camera_fps(index: usize) -> u32 {
    if index == 1 {
        60
    } else {
        30
    }
}

fn legacy_camera_fps_index(fps: u32) -> usize {
    if fps == 60 {
        1
    } else {
        0
    }
}

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
