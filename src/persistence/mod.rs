use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use std::path::Path;
pub mod scene_presets;

pub use scene_presets::{
    load_scene_presets, save_scene_presets, ScenePreset, ScenePresetCamera, ScenePresetLighting,
    ScenePresetRendering,
};

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

fn default_spring_sway_scale() -> f32 {
    1.0
}

/// Neutral 1.0 multiplier for scale-type settings loaded from
/// pre-feature projects (kept separate from `default_spring_sway_scale`
/// so the two defaults can never move together by accident).
fn default_unit_scale() -> f32 {
    1.0
}

fn default_gravity_direction() -> [f32; 3] {
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

fn default_true() -> bool {
    true
}

/// Convert the in-memory pose calibration to its on-disk DTO form.
/// Round-trips through `pose_calibration_from_dto`. Returns `None` when
/// the in-memory struct represents "no capture yet" so saved projects
/// don't accumulate empty fields.
fn pose_calibration_to_dto(
    cal: &crate::tracking::PoseCalibration,
) -> PoseCalibrationDto {
    PoseCalibrationDto {
        mode: cal.mode.as_str().to_string(),
        captured_at: cal.captured_at.clone(),
        captured_at_unix: cal.captured_at_unix,
        frame_count: cal.frame_count,
        anchor_x: cal.anchor_x,
        anchor_y: cal.anchor_y,
        anchor_depth_m: cal.anchor_depth_m,
        confidence: cal.confidence,
        anchor_depth_jitter_m: cal.anchor_depth_jitter_m,
        shoulder_span_m: cal.shoulder_span_m,
        x_range_observed: cal.x_range_observed,
        z_range_observed: cal.z_range_observed,
        neutral_expressions: cal.neutral_expressions.clone(),
        // Legacy mirror of the mesh-source neutral so a downgraded app
        // still subtracts something sensible.
        neutral_face_ypr: cal.neutral_face_ypr_mesh.unwrap_or([0.0; 3]),
        neutral_face_ypr_mesh: cal.neutral_face_ypr_mesh,
        neutral_face_ypr_body: cal.neutral_face_ypr_body,
        neutral_body_yaw: cal.neutral_body_yaw,
    }
}

/// Inverse of `pose_calibration_to_dto`. Returns `None` (and logs)
/// when the stored mode string is unrecognised — keeps a corrupted /
/// future-version file from blocking the project load.
fn dto_to_pose_calibration(
    dto: &PoseCalibrationDto,
) -> Option<crate::tracking::PoseCalibration> {
    let mode = match dto.mode.parse::<crate::tracking::CalibrationMode>() {
        Ok(m) => m,
        Err(_) => {
            warn!(
                "pose_calibration: unknown mode '{}' in project file, ignoring",
                dto.mode
            );
            return None;
        }
    };
    Some(crate::tracking::PoseCalibration {
        mode,
        captured_at: dto.captured_at.clone(),
        captured_at_unix: dto.captured_at_unix,
        frame_count: dto.frame_count,
        anchor_x: dto.anchor_x,
        anchor_y: dto.anchor_y,
        // Migration: pre-fix profiles stored `anchor_depth_m` as the
        // *source-space* z (negative for forward subjects). The field
        // is documented as camera-space metric depth (positive), and
        // the live consumers (calibrate_scale plausibility band,
        // the retired v1 root-reference seed) required that. Coerce on
        // load so stale profiles from before the bug fix don't break
        // tracking; new captures already store the positive value via
        // `aggregate()` so this is a no-op for them.
        anchor_depth_m: dto.anchor_depth_m.map(|d| d.abs()),
        confidence: dto.confidence,
        anchor_depth_jitter_m: dto.anchor_depth_jitter_m,
        // Migration: pre-fix captures measured the shoulder span on the
        // *published* skeleton, which `skeleton_from_depth` has already
        // normalised to `TARGET_SRC_SHOULDER_SPAN` — so the stored value
        // is a source-unit constant (≈0.75), not the subject's metres.
        // Consumed as metres it makes every anthropometric bone ~1.8×
        // too long (depth-hole joints fly off along their ray) and
        // shrinks the published skeleton to ~58% of the range the
        // solver's dead-zones / 1€ cutoffs are tuned for. Drop an
        // out-of-band value on load so the session falls back to the
        // provider's stabilised auto-span; the inspector then shows the
        // span as not captured, prompting a (now-correct) recapture.
        shoulder_span_m: dto.shoulder_span_m.filter(|s| {
            let ok = crate::tracking::shoulder_span_plausible(*s);
            if !ok {
                log::warn!(
                    "dropping implausible stored shoulder_span_m {s:.3} m (outside \
                     {:.2}–{:.2} m) — pre-fix calibration recorded in source units; \
                     re-run Calibrate Pose to restore it",
                    crate::tracking::SHOULDER_SPAN_MIN_M,
                    crate::tracking::SHOULDER_SPAN_MAX_M,
                );
            }
            ok
        }),
        x_range_observed: dto.x_range_observed,
        z_range_observed: dto.z_range_observed,
        neutral_expressions: dto.neutral_expressions.clone(),
        // One-time migration: pre-split saves carried a single neutral
        // that was, in practice, mesh-sourced (the mesh wins selection
        // during a frontal calibration hold), so a non-zero legacy
        // value loads as the mesh neutral when the new field is absent.
        neutral_face_ypr_mesh: dto.neutral_face_ypr_mesh.or_else(|| {
            if dto.neutral_face_ypr != [0.0; 3] {
                Some(dto.neutral_face_ypr)
            } else {
                None
            }
        }),
        neutral_face_ypr_body: dto.neutral_face_ypr_body,
        // Defensive re-clamp: the capture path already clamps, but a
        // hand-edited profile with a ±180° value would flip the whole
        // scene behind the camera every frame.
        neutral_body_yaw: dto
            .neutral_body_yaw
            .map(|y| y.clamp(-crate::tracking::BODY_YAW_MAX_RAD, crate::tracking::BODY_YAW_MAX_RAD)),
    })
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
    /// Legacy combo positions — still written (best-effort) for
    /// downgrade compatibility and still read as the fallback for
    /// files that predate the value fields below. The mapping is the
    /// FROZEN table the combos used when these files were written
    /// (`legacy_camera_*` helpers); do NOT retarget it when the GUI
    /// grows new combo entries.
    #[serde(default)]
    pub camera_resolution_index: usize,
    #[serde(default)]
    pub camera_framerate_index: usize,
    /// Real capture format values. `None` in files saved before the
    /// index→value migration; readers fall back to the legacy indices.
    #[serde(default)]
    pub camera_capture_width: Option<u32>,
    #[serde(default)]
    pub camera_capture_height: Option<u32>,
    #[serde(default)]
    pub camera_capture_fps: Option<u32>,
    #[serde(default)]
    pub hand_tracking_enabled: bool,
    /// `default_true`: matches the GUI's initial value
    /// (`TrackingGuiState`). Without this, projects that predate the
    /// retargeting block (or were saved before face tracking landed)
    /// load with face tracking silently disabled, which the user
    /// experiences as "the head no longer follows me after upgrading".
    #[serde(default = "default_true")]
    pub face_tracking_enabled: bool,
    #[serde(default)]
    pub lower_body_tracking_enabled: bool,
    #[serde(default = "default_true")]
    pub root_translation_enabled: bool,
    #[serde(default)]
    pub fade_on_tracking_loss: bool,
    /// Pipeline-stage toggles (bound at tracking start). Defaults
    /// mirror `TrackingPipelineConfig::default()` so projects predating
    /// the Pipeline inspector section load with the same pipeline the
    /// provider previously hardcoded.
    #[serde(default)]
    pub force_cpu_inference: bool,
    #[serde(default = "default_true")]
    pub yolox_enabled: bool,
    #[serde(default)]
    pub show_camera_wipe: bool,
    #[serde(default)]
    pub show_detection_annotations: bool,
    /// Pose-solver smoothing. Defaults mirror `TrackingSmoothingParams::
    /// default()` so projects predating the Advanced smoothing section load
    /// with the same behaviour the solver had when the values were hardcoded.
    #[serde(default = "default_rotation_blend")]
    pub smoothing_rotation_blend: f32,
    #[serde(default = "default_expression_blend")]
    pub smoothing_expression_blend: f32,
    #[serde(default)]
    pub smoothing_face_confidence: f32,
    /// **Legacy field — read-only for migration.** Pose calibration
    /// has moved to per-profile storage (`StreamProfile.pose_calibration`
    /// in `gui::profile`) so a user with multiple "home desk" /
    /// "office desk" profiles can carry a separate calibration per
    /// setup. This field is kept on the on-disk DTO purely so older
    /// project files don't lose their captured calibration on first
    /// open: `GuiApp::load_state` migrates a `Some(_)` here onto the
    /// currently-active profile (when that profile has no
    /// calibration yet) and marks the profile library dirty so the
    /// rescued value lands in `profiles.json` on the next autosave.
    /// New saves never emit this field thanks to the
    /// `skip_serializing_if` attribute, so a project saved by this
    /// version of the app and re-loaded by the same version simply
    /// won't see this branch take effect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pose_calibration: Option<PoseCalibrationDto>,
}

/// On-disk shape of [`crate::tracking::PoseCalibration`]. Kept as a
/// separate DTO so the persisted schema can evolve independently of the
/// in-memory struct (e.g. dropping a field, renaming `mode` enum
/// variants) without forcing a breaking change to live consumers.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PoseCalibrationDto {
    pub mode: String,
    pub captured_at: String,
    #[serde(default)]
    pub captured_at_unix: u64,
    pub frame_count: usize,
    pub anchor_x: f32,
    pub anchor_y: f32,
    pub anchor_depth_m: Option<f32>,
    pub confidence: f32,
    pub anchor_depth_jitter_m: Option<f32>,
    #[serde(default)]
    pub shoulder_span_m: Option<f32>,
    #[serde(default)]
    pub x_range_observed: Option<f32>,
    #[serde(default)]
    pub z_range_observed: Option<f32>,
    /// Per-person resting expression baseline — see
    /// `crate::tracking::PoseCalibration::neutral_expressions`. Stored
    /// as `(name, weight)` pairs; defaults to empty for older saves /
    /// captures taken with face tracking off.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub neutral_expressions: Vec<(String, f32)>,
    /// Legacy single-source resting head pose `[yaw, pitch, roll]`
    /// (radians). Written by app versions that predate the per-source
    /// split; on load it migrates to
    /// [`Self::neutral_face_ypr_mesh`] when the new field is absent
    /// (the legacy capture path accumulated the *published* face pose,
    /// which during a frontal hold was effectively always the
    /// mesh-sourced one). Still written on save (mirroring the mesh
    /// value) so a downgraded app keeps a usable neutral. Zeros mean
    /// "not captured".
    #[serde(default)]
    pub neutral_face_ypr: [f32; 3],
    /// Per-source resting head pose (FaceMesh estimator) — see
    /// `crate::tracking::PoseCalibration::neutral_face_ypr_mesh`.
    /// `None` for older saves (migrated from the legacy field above)
    /// and for captures where too few confident mesh frames were seen.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub neutral_face_ypr_mesh: Option<[f32; 3]>,
    /// Per-source resting head pose (RTMW3D body estimator) — see
    /// `crate::tracking::PoseCalibration::neutral_face_ypr_body`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub neutral_face_ypr_body: Option<[f32; 3]>,
    /// Neutral body yaw (radians) for oblique camera placement — see
    /// `crate::tracking::PoseCalibration::neutral_body_yaw`. `None`
    /// for older saves and non-metric captures (strict no-op on
    /// load).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub neutral_body_yaw: Option<f32>,
}


#[derive(Serialize, Deserialize, Debug, Default)]
pub struct RenderingConfig {
    #[serde(default)]
    pub material_mode_index: usize,
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
    /// Spring-bone user tuning. Defaults keep projects saved before the
    /// feature behaving as-authored (sway 1.0, gravity offset 0.0).
    #[serde(default = "default_spring_sway_scale")]
    pub spring_sway_scale: f32,
    #[serde(default)]
    pub spring_gravity_offset: f32,
    /// Natural (physically-scaled) gravity for spring bones. Defaults
    /// on: projects saved before the feature load with hair re-hanging
    /// naturally rather than the legacy unitless force mix.
    #[serde(default = "default_true")]
    pub spring_natural_gravity: bool,
    #[serde(default = "default_gravity_direction")]
    pub scene_gravity_direction: [f32; 3],
    #[serde(default = "default_unit_scale")]
    pub scene_gravity_strength: f32,
    #[serde(default)]
    pub toggle_cloth: bool,
    #[serde(default)]
    pub toggle_collision_debug: bool,
    #[serde(default)]
    pub toggle_skeleton_debug: bool,
    #[serde(default)]
    pub alpha_preview: bool,
    /// Bloom post-effect. Defaults keep pre-bloom project files on the
    /// historical no-post-processing output.
    #[serde(default)]
    pub bloom_enabled: bool,
    #[serde(default = "default_bloom_intensity")]
    pub bloom_intensity: f32,
    #[serde(default = "default_bloom_threshold")]
    pub bloom_threshold: f32,
    /// Generative background. Defaults keep older project files on the
    /// historical clear-color background; the parameter defaults mirror
    /// `GenerativeBackgroundSettings::default()`.
    #[serde(default)]
    pub bg_enabled: bool,
    #[serde(default = "default_bg_unit")]
    pub bg_intensity: f32,
    #[serde(default = "default_bg_unit")]
    pub bg_speed: f32,
    #[serde(default = "default_bg_scale")]
    pub bg_scale: f32,
    #[serde(default = "default_bg_unit")]
    pub bg_reactivity: f32,
    #[serde(default = "default_bg_color_a")]
    pub bg_color_a: [f32; 3],
    #[serde(default = "default_bg_color_b")]
    pub bg_color_b: [f32; 3],
}

fn default_bloom_intensity() -> f32 {
    0.6
}

fn default_bloom_threshold() -> f32 {
    1.0
}

fn default_bg_unit() -> f32 {
    1.0
}

fn default_bg_scale() -> f32 {
    2.0
}

fn default_bg_color_a() -> [f32; 3] {
    [0.02, 0.05, 0.18]
}

fn default_bg_color_b() -> [f32; 3] {
    [0.10, 0.85, 1.00]
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
    /// 0=Audio, 1=Image, 2=Both. Defaults to Both so the camera mouth works
    /// out of the box without silently overriding audio.
    #[serde(default = "default_mouth_source_index")]
    pub mouth_source_index: usize,
}

fn default_volume_threshold() -> f32 {
    0.01
}

fn default_mouth_source_index() -> usize {
    2
}

fn default_lipsync_smoothing() -> f32 {
    0.5
}

fn default_rotation_blend() -> f32 {
    1.0
}

fn default_expression_blend() -> f32 {
    0.8
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
    /// Anti-aliasing (MSAA) level index: 0=Off, 1=2x, 2=4x, 3=8x.
    /// `#[serde(default)]` keeps older projects (which lacked this) loadable.
    #[serde(default)]
    pub msaa_index: usize,
}

fn default_zoom_sensitivity() -> f32 {
    // Exponential-zoom scale (see viewport.rs): ~10%/notch. Projects saved
    // under the old linear scale (~0.1) are clamped to 0.01 at use-time.
    0.002
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

fn load_app_settings_from(path: &Path) -> Option<AppSettings> {
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

fn save_app_settings_to(settings: &AppSettings, path: &Path) -> Result<(), String> {
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

fn migrate_legacy_app_settings_from(path: &Path) -> Option<AppSettings> {
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
    })
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
fn migrate_project_json(
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
            camera_capture_width: self
                .tracking
                .camera_capture_width
                .unwrap_or_else(|| legacy_camera_resolution(self.tracking.camera_resolution_index).0),
            camera_capture_height: self
                .tracking
                .camera_capture_height
                .unwrap_or_else(|| legacy_camera_resolution(self.tracking.camera_resolution_index).1),
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
            pose_calibration: self.tracking.pose_calibration.as_ref().and_then(dto_to_pose_calibration),

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
/// combo mapping lives in `gui::camera_resolution_for_index` and may
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

/// Where the user's `StreamProfile` library is stored. Lives next to
/// `last_session.vvtproj` under the OS app-data dir so the same set of
/// profiles is available across every project the user opens — the
/// "home desk" / "office desk" / "show stage" lighting + calibration
/// presets follow the *user*, not whichever scene file they happened
/// to open last.
pub fn profiles_path() -> std::path::PathBuf {
    let mut path = app_data_dir();
    path.push("profiles.json");
    path
}

/// Load the user's saved `ProfileLibrary`. Returns `None` when the file
/// does not yet exist (first run) or fails to parse — the GUI then
/// falls back to `ProfileLibrary::default()` (the built-in Streaming
/// / Recording / Performance presets) so a fresh install or a
/// corrupted file doesn't lock the user out of the profile dropdown.
/// A parse failure is logged but otherwise non-fatal.
pub fn load_profiles() -> Option<crate::gui::profile::ProfileLibrary> {
    let path = profiles_path();
    if !path.exists() {
        return None;
    }
    match std::fs::read_to_string(&path) {
        Ok(data) => match serde_json::from_str(&data) {
            Ok(lib) => Some(lib),
            Err(e) => {
                error!(
                    "persistence: failed to parse profiles at {}: {} \
                     (falling back to built-in presets)",
                    path.display(),
                    e
                );
                None
            }
        },
        Err(e) => {
            error!(
                "persistence: failed to read profiles at {}: {}",
                path.display(),
                e
            );
            None
        }
    }
}

/// Persist the user's `ProfileLibrary`. Called from
/// `GuiApp::save_profiles_if_dirty` whenever `profiles_dirty` flips —
/// triggered by `Calibrate Pose ▼` writing into the active profile,
/// by future profile-edit UI, etc. Atomic write via `atomic_write`
/// so a mid-write crash never leaves a half-truncated file behind.
pub fn save_profiles(
    library: &crate::gui::profile::ProfileLibrary,
) -> Result<(), String> {
    let path = profiles_path();
    let data = serde_json::to_string_pretty(library)
        .map_err(|e| format!("serialise profiles: {}", e))?;
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    atomic_write(&path, &data).map_err(|e| format!("write profiles: {}", e))
}

/// Path of the implicit "last session" project file. The GUI writes here
/// every time `project_dirty` flips, even when the user has never run
/// File > Save As — so quitting after toggling a checkbox preserves the
/// change for the next launch. Lives next to `avatar_library.vvtlib` and
/// recovery snapshots under `%APPDATA%\VulVATAR`.
pub fn last_session_path() -> std::path::PathBuf {
    let mut path = app_data_dir();
    path.push("last_session.vvtproj");
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
        // Backup-copy is best-effort: if it fails (read-only target dir,
        // disk full, AV-quarantined source), proceed with the write
        // anyway — refusing to write because the backup couldn't be
        // taken would itself be a bigger UX failure. But log the
        // failure so a user diagnosing "where did my backup go?" can
        // trace it instead of finding a missing .bak silently.
        if let Err(e) = std::fs::copy(path, backup_path_for(path)) {
            warn!(
                "atomic_write: backup of {} failed (continuing without backup): {}",
                path.display(),
                e
            );
        }
    }

    let parent = path.parent().unwrap_or(Path::new("."));
    let temp_name = format!(
        ".{}.tmp",
        path.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".to_string())
    );
    let temp_path = parent.join(&temp_name);

    // Write + fsync the temp file BEFORE the rename. A plain
    // `fs::write` + rename is atomic against a process crash, but NOT
    // against power loss / a hard GPU-driver freeze: the rename can
    // commit in the directory while the temp file's data blocks are
    // still in the OS write cache, so after the freeze the file exists
    // at full length but reads back as all-zero. That is exactly how
    // the 2026-06-12 Arc freeze NUL-corrupted `last_session.vvtproj`
    // (and its `.bak`). `sync_all` forces the bytes to stable storage
    // first, so the rename only ever publishes durable data. Cost is
    // one fsync per write; every caller is throttled (autosave 4/s,
    // recovery on a timer) so the added latency is immaterial.
    {
        use std::io::Write as _;
        let mut f = std::fs::File::create(&temp_path).map_err(|e| {
            format!(
                "atomic_write: failed to create temp file {}: {}",
                temp_path.display(),
                e
            )
        })?;
        f.write_all(data.as_bytes()).map_err(|e| {
            let _ = std::fs::remove_file(&temp_path);
            format!(
                "atomic_write: failed to write temp file {}: {}",
                temp_path.display(),
                e
            )
        })?;
        f.sync_all().map_err(|e| {
            let _ = std::fs::remove_file(&temp_path);
            format!(
                "atomic_write: failed to fsync temp file {}: {}",
                temp_path.display(),
                e
            )
        })?;
    }

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
    /// Monotonic instant of the last successful snapshot write. The
    /// interval gate used to compare wall-clock unix seconds, so an
    /// NTP step backwards silently paused the crash-recovery net (and
    /// a step forward fired it early). Wall time is still *recorded*
    /// in the snapshot (`timestamp_secs`) — that's display data, not
    /// scheduling data.
    last_snapshot: Option<std::time::Instant>,
    enabled: bool,
}

impl RecoveryManager {
    pub fn new(interval_secs: u64) -> Self {
        Self {
            interval_secs,
            last_snapshot: None,
            enabled: true,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn set_interval(&mut self, secs: u64) {
        self.interval_secs = secs;
    }

    pub fn should_snapshot(&self, now: std::time::Instant) -> bool {
        if !self.enabled {
            return false;
        }
        match self.last_snapshot {
            None => true,
            Some(last) => now.duration_since(last).as_secs() >= self.interval_secs,
        }
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

        self.last_snapshot = Some(std::time::Instant::now());
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
#[cfg(test)]
mod tests;
