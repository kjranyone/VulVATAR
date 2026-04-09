pub mod inspector;
pub mod mode_nav;
pub mod status_bar;
pub mod top_bar;
pub mod viewport;

use eframe::egui;

use crate::app::Application;

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

    pub transform_position: [f32; 3],
    pub transform_rotation: [f32; 3],
    pub transform_scale: f32,
    pub background_transparent: bool,
    pub background_color: [f32; 3],
    pub toggle_tracking: bool,
    pub toggle_spring: bool,
    pub toggle_cloth: bool,
    pub toggle_collision_debug: bool,
    pub toggle_skeleton_debug: bool,

    pub camera_index: usize,
    pub camera_running: bool,
    pub camera_resolution_index: usize,
    pub camera_framerate_index: usize,
    pub tracking_mirror: bool,
    pub smoothing_strength: f32,
    pub confidence_threshold: f32,
    pub hand_tracking_enabled: bool,
    pub face_tracking_enabled: bool,

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
    pub alpha_preview: bool,

    pub output_sink_index: usize,
    pub output_resolution_index: usize,
    pub output_framerate_index: usize,
    pub output_has_alpha: bool,
    pub output_color_space_index: usize,
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

            transform_position: [0.0, 0.0, 0.0],
            transform_rotation: [0.0, 0.0, 0.0],
            transform_scale: 1.0,
            background_transparent: true,
            background_color: [0.1, 0.1, 0.1],
            toggle_tracking: true,
            toggle_spring: true,
            toggle_cloth: false,
            toggle_collision_debug: false,
            toggle_skeleton_debug: false,

            camera_index: 0,
            camera_running: false,
            camera_resolution_index: 0,
            camera_framerate_index: 0,
            tracking_mirror: true,
            smoothing_strength: 0.5,
            confidence_threshold: 0.5,
            hand_tracking_enabled: false,
            face_tracking_enabled: true,

            material_mode_index: 2,
            toon_ramp_threshold: 0.5,
            shadow_softness: 0.1,
            outline_enabled: true,
            outline_width: 0.01,
            outline_color: [0.0, 0.0, 0.0],
            light_direction: [0.5, -1.0, 0.3],
            light_intensity: 1.0,
            ambient: [0.2, 0.2, 0.2],
            camera_fov: 60.0,
            alpha_preview: false,

            output_sink_index: 2,
            output_resolution_index: 0,
            output_framerate_index: 0,
            output_has_alpha: true,
            output_color_space_index: 0,
        }
    }
}

impl eframe::App for GuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.paused {
            self.app.run_frame();
            self.frame_count += 1;
        }

        top_bar::draw(ctx, self);
        mode_nav::draw(ctx, self);
        status_bar::draw(ctx, self);
        inspector::draw(ctx, self);
        viewport::draw(ctx, self);

        ctx.request_repaint();
    }
}
