pub mod inspector;
pub mod mode_nav;
pub mod status_bar;
pub mod top_bar;
pub mod viewport;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use eframe::egui;

use crate::app::{Application, RuntimeToggles};
use crate::persistence::ProjectState;
use crate::tracking::TrackingSmoothingParams;

pub struct TransformState {
    pub position: [f32; 3],
    pub rotation: [f32; 3],
    pub scale: f32,
}

pub struct TrackingGuiState {
    pub toggle_tracking: bool,
    pub camera_running: bool,
    pub camera_resolution_index: usize,
    pub camera_framerate_index: usize,
    pub tracking_mirror: bool,
    pub hand_tracking_enabled: bool,
    pub face_tracking_enabled: bool,
    pub smoothing_strength: f32,
    pub confidence_threshold: f32,
}

pub struct RenderingGuiState {
    pub material_mode_index: usize,
    pub background_color: [f32; 3],
    pub transparent_background: bool,
    pub camera_fov: f32,
    pub main_light_dir: [f32; 3],
    pub main_light_intensity: f32,
    pub ambient_intensity: [f32; 3],
    pub toon_ramp_threshold: f32,
    pub shadow_softness: f32,
    pub outline_enabled: bool,
    pub outline_width: f32,
    pub outline_color: [f32; 3],
    pub alpha_preview: bool,
    pub toggle_spring: bool,
    pub toggle_cloth: bool,
    pub toggle_collision_debug: bool,
    pub toggle_skeleton_debug: bool,
}

pub struct OutputGuiState {
    pub output_sink_index: usize,
    pub output_resolution_index: usize,
    pub output_framerate_index: usize,
    pub output_has_alpha: bool,
    pub output_color_space_index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppMode {
    Preview,
    TrackingSetup,
    Rendering,
    Output,
    ClothAuthoring,
}

impl AppMode {
    pub const ALL: [AppMode; 5] = [
        AppMode::Preview,
        AppMode::TrackingSetup,
        AppMode::Rendering,
        AppMode::Output,
        AppMode::ClothAuthoring,
    ];

    pub fn label(&self) -> &str {
        match self {
            AppMode::Preview => "Preview",
            AppMode::TrackingSetup => "Tracking Setup",
            AppMode::Rendering => "Rendering",
            AppMode::Output => "Output",
            AppMode::ClothAuthoring => "Cloth Authoring",
        }
    }
}

pub struct GuiApp {
    pub mode: AppMode,
    pub app: Application,
    pub paused: bool,
    pub frame_count: u64,
    pub project_path: Option<PathBuf>,

    // E10: Frame timing
    pub last_frame_instant: Instant,
    pub frame_time_ms: f64,
    pub fps: f64,

    // E11: Recent avatars
    pub recent_avatars: Vec<PathBuf>,

    // E12: Project dirty state
    pub project_dirty: bool,

    // E14: Autosave
    /// If `Some`, autosave will trigger after this interval when the project is dirty.
    pub autosave_interval: Option<Duration>,
    /// Tracks the last time an autosave (or manual save) occurred.
    pub last_autosave: Instant,

    // M10: Notification/toast system
    pub notifications: Vec<(String, Instant)>,

    pub transform: TransformState,
    pub tracking: TrackingGuiState,
    pub rendering: RenderingGuiState,
    pub output: OutputGuiState,

    pub camera_index: usize,

    // Animation inspector state
    pub animation_playing: bool,

    // Cloth authoring inspector state
    pub cloth_sim_playing: bool,
    pub cloth_rename_buf: String,
    pub cloth_save_status: Option<String>,

    // Viewport texture integration
    pub viewport_texture: Option<egui::TextureHandle>,
    pub viewport_last_frame: u64,
}

impl GuiApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = Application::new();
        app.bootstrap();

        Self {
            mode: AppMode::Preview,
            app,
            paused: false,
            frame_count: 0,
            project_path: None,

            last_frame_instant: Instant::now(),
            frame_time_ms: 0.0,
            fps: 0.0,

            recent_avatars: Vec::new(),

            project_dirty: false,

            autosave_interval: Some(Duration::from_secs(300)), // 5 minutes
            last_autosave: Instant::now(),

            notifications: Vec::new(),

            transform: TransformState {
                position: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0],
                scale: 1.0,
            },
            tracking: TrackingGuiState {
                toggle_tracking: true,
                camera_running: false,
                camera_resolution_index: 0,
                camera_framerate_index: 0,
                tracking_mirror: true,
                hand_tracking_enabled: false,
                face_tracking_enabled: true,
                smoothing_strength: 0.5,
                confidence_threshold: 0.5,
            },
            rendering: RenderingGuiState {
                material_mode_index: 2,
                background_color: [0.1, 0.1, 0.1],
                transparent_background: true,
                camera_fov: 60.0,
                main_light_dir: [0.5, -1.0, 0.3],
                main_light_intensity: 1.0,
                ambient_intensity: [0.2, 0.2, 0.2],
                toon_ramp_threshold: 0.5,
                shadow_softness: 0.1,
                outline_enabled: true,
                outline_width: 0.01,
                outline_color: [0.0, 0.0, 0.0],
                alpha_preview: false,
                toggle_spring: true,
                toggle_cloth: false,
                toggle_collision_debug: false,
                toggle_skeleton_debug: false,
            },
            output: OutputGuiState {
                output_sink_index: 2,
                output_resolution_index: 0,
                output_framerate_index: 0,
                output_has_alpha: true,
                output_color_space_index: 0,
            },

            camera_index: 0,

            animation_playing: false,

            cloth_sim_playing: false,
            cloth_rename_buf: String::new(),
            cloth_save_status: None,

            viewport_texture: None,
            viewport_last_frame: 0,
        }
    }
}

impl GuiApp {
    /// Push a notification message that will auto-dismiss after 5 seconds.
    pub fn push_notification(&mut self, msg: String) {
        self.notifications.push((msg, Instant::now()));
    }

    /// Add a path to the recent avatars list (max 10, dedup, most recent first).
    pub fn add_recent_avatar(&mut self, path: PathBuf) {
        self.recent_avatars.retain(|p| p != &path);
        self.recent_avatars.insert(0, path);
        self.recent_avatars.truncate(10);
    }

    /// Build a `RuntimeToggles` from the current GUI state.
    fn runtime_toggles(&self) -> RuntimeToggles {
        RuntimeToggles {
            tracking_enabled: self.tracking.toggle_tracking,
            spring_enabled: self.rendering.toggle_spring,
            cloth_enabled: self.rendering.toggle_cloth,
            collision_debug: self.rendering.toggle_collision_debug,
            skeleton_debug: self.rendering.toggle_skeleton_debug,
        }
    }

    /// Snapshot the current GUI state into a `ProjectState` for persistence.
    pub fn to_project_state(&self) -> ProjectState {
        let avatar_source_path = self
            .app
            .avatar
            .as_ref()
            .map(|a| a.asset.source_path.to_string_lossy().into_owned());

        let avatar_source_hash = self
            .app
            .avatar
            .as_ref()
            .map(|a| a.asset.source_hash.0.to_vec());

        let active_overlay_path = self
            .app
            .editor
            .overlay_path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned());

        ProjectState {
            avatar_source_path,
            avatar_source_hash,
            active_overlay_path,

            transform_position: self.transform.position,
            transform_rotation: self.transform.rotation,
            transform_scale: self.transform.scale,

            tracking_mirror: self.tracking.tracking_mirror,
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
            light_direction: self.rendering.main_light_dir,
            light_intensity: self.rendering.main_light_intensity,
            ambient: self.rendering.ambient_intensity,
            camera_fov: self.rendering.camera_fov,

            output_sink_index: self.output.output_sink_index,
            output_resolution_index: self.output.output_resolution_index,
            output_framerate_index: self.output.output_framerate_index,
            output_has_alpha: self.output.output_has_alpha,
            output_color_space_index: self.output.output_color_space_index,
        }
    }

    /// Apply a loaded `ProjectState` to the GUI fields.
    /// Avatar and overlay loading is handled separately by the caller.
    pub fn apply_project_state(&mut self, state: &ProjectState) {
        self.transform.position = state.transform_position;
        self.transform.rotation = state.transform_rotation;
        self.transform.scale = state.transform_scale;

        self.tracking.tracking_mirror = state.tracking_mirror;
        self.tracking.smoothing_strength = state.smoothing_strength;
        self.tracking.confidence_threshold = state.confidence_threshold;
        self.tracking.hand_tracking_enabled = state.hand_tracking_enabled;
        self.tracking.face_tracking_enabled = state.face_tracking_enabled;

        self.rendering.material_mode_index = state.material_mode_index;
        self.rendering.toon_ramp_threshold = state.toon_ramp_threshold;
        self.rendering.shadow_softness = state.shadow_softness;
        self.rendering.outline_enabled = state.outline_enabled;
        self.rendering.outline_width = state.outline_width;
        self.rendering.outline_color = state.outline_color;
        self.rendering.main_light_dir = state.light_direction;
        self.rendering.main_light_intensity = state.light_intensity;
        self.rendering.ambient_intensity = state.ambient;
        self.rendering.camera_fov = state.camera_fov;

        self.output.output_sink_index = state.output_sink_index;
        self.output.output_resolution_index = state.output_resolution_index;
        self.output.output_framerate_index = state.output_framerate_index;
        self.output.output_has_alpha = state.output_has_alpha;
        self.output.output_color_space_index = state.output_color_space_index;
    }
}

impl eframe::App for GuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // E10: Measure frame timing.
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_frame_instant);
        self.last_frame_instant = now;
        let dt_secs = elapsed.as_secs_f64();
        // Exponential moving average for smoothing.
        self.frame_time_ms = self.frame_time_ms * 0.9 + (dt_secs * 1000.0) * 0.1;
        if dt_secs > 0.0 {
            self.fps = self.fps * 0.9 + (1.0 / dt_secs) * 0.1;
        }

        // Sync GUI-driven state into the application before running the frame.
        if let Some(avatar) = self.app.avatar.as_mut() {
            avatar.world_transform.translation = self.transform.position;
            // Convert Euler degrees to a quaternion (XYZ order).
            let [rx, ry, rz] = self.transform.rotation;
            let (rx, ry, rz) = (
                rx.to_radians() * 0.5,
                ry.to_radians() * 0.5,
                rz.to_radians() * 0.5,
            );
            let (sx, cx) = (rx.sin(), rx.cos());
            let (sy, cy) = (ry.sin(), ry.cos());
            let (sz, cz) = (rz.sin(), rz.cos());
            avatar.world_transform.rotation = [
                sx * cy * cz - cx * sy * sz,
                cx * sy * cz + sx * cy * sz,
                cx * cy * sz - sx * sy * cz,
                cx * cy * cz + sx * sy * sz,
            ];
            let s = self.transform.scale;
            avatar.world_transform.scale = [s, s, s];
        }

        // Sync GUI camera controls into the Application's viewport camera.
        self.app.viewport_camera.yaw_deg = self.transform.rotation[1];
        self.app.viewport_camera.pitch_deg = self.transform.rotation[0];
        // Map transform_position[2] as a camera distance offset (closer/further).
        self.app.viewport_camera.distance = 5.0 - self.transform.position[2] * 10.0;
        self.app.viewport_camera.pan = [self.transform.position[0], self.transform.position[1]];
        self.app.viewport_camera.fov_deg = self.rendering.camera_fov;

        // Sync lighting parameters from GUI inspector.
        self.app.viewport_lighting.main_light_dir_ws = self.rendering.main_light_dir;
        self.app.viewport_lighting.main_light_intensity = self.rendering.main_light_intensity;
        self.app.viewport_lighting.ambient_term = self.rendering.ambient_intensity;

        // Sync output resolution preference from GUI inspector.
        let output_res = match self.output.output_resolution_index {
            0 => [1920u32, 1080u32],
            1 => [1280, 720],
            _ => [640, 480],
        };
        self.app.output_extent = Some(output_res);

        if !self.paused {
            let real_dt = (self.frame_time_ms / 1000.0) as f32;
            // Clamp dt to avoid huge steps on first frame or after pauses.
            let frame_dt = real_dt.clamp(0.001, 0.1);
            let frame_config = crate::app::FrameConfig {
                toggles: self.runtime_toggles(),
                smoothing: TrackingSmoothingParams {
                    position_smoothing: self.tracking.smoothing_strength,
                    orientation_smoothing: self.tracking.smoothing_strength,
                    expression_smoothing: self.tracking.smoothing_strength * 0.6,
                    confidence_threshold: self.tracking.confidence_threshold,
                    ..TrackingSmoothingParams::default()
                },
                material_mode_index: self.rendering.material_mode_index,
                hand_tracking_enabled: self.tracking.hand_tracking_enabled,
                face_tracking_enabled: self.tracking.face_tracking_enabled,
                frame_dt,
            };
            self.app.run_frame(&frame_config);
            self.frame_count += 1;
        }

        // E14: Autosave - save if enabled, project is dirty, and enough time has passed.
        if let Some(interval) = self.autosave_interval {
            if self.project_dirty && self.last_autosave.elapsed() >= interval {
                if let Some(ref path) = self.project_path.clone() {
                    let project_state = self.to_project_state();
                    match crate::persistence::save_project(&project_state, path) {
                        Ok(()) => {
                            self.project_dirty = false;
                            self.last_autosave = Instant::now();
                            self.push_notification("Autosaved project.".to_string());
                        }
                        Err(e) => {
                            self.last_autosave = Instant::now();
                            self.push_notification(format!("Autosave failed: {}", e));
                        }
                    }
                }
            }
        }

        top_bar::draw(ctx, self);
        mode_nav::draw(ctx, self);
        status_bar::draw(ctx, self);
        inspector::draw(ctx, self);
        viewport::draw(ctx, self);

        // M10: Draw notification toasts and expire old ones.
        self.notifications
            .retain(|(_, created)| created.elapsed().as_secs_f32() < 5.0);
        if !self.notifications.is_empty() {
            egui::Area::new(egui::Id::new("notifications"))
                .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-10.0, -40.0))
                .show(ctx, |ui| {
                    for (msg, created) in &self.notifications {
                        let age = created.elapsed().as_secs_f32();
                        let alpha = if age > 4.0 {
                            ((5.0 - age) * 255.0) as u8
                        } else {
                            255u8
                        };
                        egui::Frame::group(ui.style())
                            .fill(egui::Color32::from_rgba_unmultiplied(40, 40, 40, alpha))
                            .show(ui, |ui| {
                                ui.label(egui::RichText::new(msg).color(
                                    egui::Color32::from_rgba_unmultiplied(255, 255, 255, alpha),
                                ));
                            });
                        ui.add_space(2.0);
                    }
                });
        }

        ctx.request_repaint();
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // Perform ordered shutdown of worker threads when the window closes.
        println!("gui: window closing, initiating application shutdown");
        self.app.shutdown();
    }
}
