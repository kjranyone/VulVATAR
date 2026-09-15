//! On-disk DTO forms of the project's configuration blocks and the
//! serde default helpers those DTOs reference.

use serde::{Deserialize, Serialize};

use super::{default_gravity_direction, default_spring_sway_scale, default_unit_scale};

pub(super) fn default_true() -> bool {
    true
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
    #[serde(default = "default_true")]
    pub smoothing_pose_interp_enabled: bool,
    #[serde(default = "default_pose_interp_delay_frac")]
    pub smoothing_pose_interp_delay_frac: f32,
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

fn default_pose_interp_delay_frac() -> f32 {
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
