//! On-disk DTO forms of the project's configuration blocks plus the
// pose-calibration <-> DTO conversions (with their load-time
// migrations) and the serde default helpers those DTOs reference.

use log::warn;

use serde::{Deserialize, Serialize};

use super::{default_gravity_direction, default_spring_sway_scale, default_unit_scale};

pub(super) fn default_true() -> bool {
    true
}

/// Convert the in-memory pose calibration to its on-disk DTO form.
/// Round-trips through `pose_calibration_from_dto`. Returns `None` when
/// the in-memory struct represents "no capture yet" so saved projects
/// don't accumulate empty fields.
pub(super) fn pose_calibration_to_dto(
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
pub(super) fn dto_to_pose_calibration(
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
        neutral_body_yaw: dto.neutral_body_yaw.map(|y| {
            y.clamp(
                -crate::tracking::BODY_YAW_MAX_RAD,
                crate::tracking::BODY_YAW_MAX_RAD,
            )
        }),
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

pub(super) fn default_zoom_sensitivity() -> f32 {
    // Exponential-zoom scale (see viewport.rs): ~10%/notch. Projects saved
    // under the old linear scale (~0.1) are clamped to 0.01 at use-time.
    0.002
}

pub(super) fn default_orbit_sensitivity() -> f32 {
    0.3
}

pub(super) fn default_pan_sensitivity() -> f32 {
    1.0
}
