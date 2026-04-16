pub mod hotkey;
pub mod inspector;
pub mod mesh_picking;
pub mod mode_nav;
pub mod profile;
pub mod status_bar;
pub mod top_bar;
pub mod viewport;
pub mod viewport_overlay;

use log::{info, warn};
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

pub struct CameraOrbitState {
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub pan: [f32; 2],
    pub distance: f32,
    pub target_distance: f32,
}

pub struct TrackingGuiState {
    pub toggle_tracking: bool,
    pub camera_resolution_index: usize,
    pub camera_framerate_index: usize,
    pub tracking_mirror: bool,
    pub hand_tracking_enabled: bool,
    pub face_tracking_enabled: bool,
    pub smoothing_strength: f32,
    pub confidence_threshold: f32,
}

pub struct LipSyncGuiState {
    pub enabled: bool,
    pub mic_device_index: usize,
    pub available_mics: Vec<crate::lipsync::AudioDeviceInfo>,
    pub volume_threshold: f32,
    pub smoothing: f32,
    /// Current RMS volume (for the live meter in the UI).
    pub current_volume: f32,
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
    pub app: Box<Application>,
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
    pub autosave_interval: Option<Duration>,
    pub last_autosave: Instant,

    // Crash recovery
    pub recovery_manager: crate::persistence::RecoveryManager,
    pub overlay_dirty: bool,

    // Hotkeys & profiles
    pub hotkeys: hotkey::HotkeyMap,
    pub profiles: profile::ProfileLibrary,
    pub wind_presets: profile::WindPresetLibrary,

    // M10: Notification/toast system
    pub notifications: Vec<(String, Instant)>,

    pub transform: TransformState,
    pub camera_orbit: CameraOrbitState,
    pub tracking: TrackingGuiState,
    pub rendering: RenderingGuiState,
    pub output: OutputGuiState,

    pub camera_index: usize,
    pub available_cameras: Vec<crate::tracking::CameraInfo>,
    pub show_camera_wipe: bool,
    pub show_detection_annotations: bool,
    pub camera_wipe_texture: Option<egui::TextureHandle>,
    pub camera_wipe_seq: u64,
    pub camera_wipe_rgba_buf: Vec<u8>,

    // Lip sync
    pub lipsync: LipSyncGuiState,
    #[cfg(feature = "lipsync")]
    pub lipsync_processor: Option<crate::lipsync::LipSyncProcessor>,

    // Animation inspector state
    pub animation_playing: bool,

    // Cloth authoring inspector state
    pub cloth_sim_playing: bool,
    pub cloth_rename_buf: String,
    pub cloth_save_status: Option<String>,

    // T04: Region selection and viewport overlay state
    pub region_selection: Option<crate::editor::cloth_authoring::RegionSelection>,
    pub viewport_overlay: viewport_overlay::ViewportOverlayState,
    pub cloth_material_pick_index: usize,
    pub cloth_distance_stiffness: f32,
    pub cloth_bend_enabled: bool,
    pub cloth_bend_stiffness: f32,
    pub cloth_collider_toggles: Vec<bool>,
    pub cloth_pin_node_index: usize,
    pub cloth_autosave_consent: Option<bool>,

    pub expression_weights: Vec<f32>,

    // Scene presets
    pub scene_presets: Vec<crate::persistence::ScenePreset>,
    pub scene_preset_index: Option<usize>,
    pub scene_preset_name: String,

    // Viewport texture integration
    pub viewport_texture: Option<egui::TextureHandle>,
    pub viewport_last_frame: u64,

    // Avatar library GUI state
    pub library_search_query: String,
    pub library_selected_index: Option<usize>,
    pub library_rename_buf: String,
    pub library_tag_buf: String,
    pub library_show_missing: bool,

    pub folder_watcher: Option<crate::app::folder_watcher::FolderWatcher>,
    pub watched_avatar_dirs: Vec<std::path::PathBuf>,
    pub thumbnail_gen: crate::renderer::thumbnail::ThumbnailGenerator,
}

impl GuiApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = Box::new(Application::new());
        app.bootstrap();
        app.avatar_library = crate::persistence::load_avatar_library();

        if let Some(snapshot) = crate::persistence::RecoveryManager::detect_recovery() {
            match crate::persistence::validate_recovery(&snapshot) {
                Ok(()) => {
                    warn!(
                        "persistence: recovery snapshot found (age={}s). \
                         It will be offered for restore in the GUI.",
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs()
                            .saturating_sub(snapshot.timestamp_secs)
                    );
                }
                Err(e) => {
                    warn!("persistence: invalid recovery snapshot: {}", e);
                    crate::persistence::RecoveryManager::clear_recovery();
                }
            }
        }

        Self {
            mode: AppMode::Preview,
            app,
            paused: false,
            frame_count: 0,
            project_path: None,

            last_frame_instant: Instant::now(),
            frame_time_ms: 0.0,
            fps: 0.0,

            recent_avatars: crate::persistence::load_recent_avatars(),

            project_dirty: false,

            autosave_interval: Some(Duration::from_secs(300)), // 5 minutes
            last_autosave: Instant::now(),

            recovery_manager: crate::persistence::RecoveryManager::new(120),
            overlay_dirty: false,

            hotkeys: hotkey::HotkeyMap::new(),
            profiles: profile::ProfileLibrary::new(),
            wind_presets: profile::WindPresetLibrary::new(),

            notifications: Vec::new(),

            transform: TransformState {
                position: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0],
                scale: 1.0,
            },
            camera_orbit: CameraOrbitState {
                yaw_deg: 0.0,
                pitch_deg: 0.0,
                pan: [0.0, 0.0],
                distance: 5.0,
                target_distance: 5.0,
            },
            tracking: TrackingGuiState {
                toggle_tracking: true,
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
            available_cameras: crate::tracking::list_cameras(),
            show_camera_wipe: false,
            show_detection_annotations: true,
            camera_wipe_texture: None,
            camera_wipe_seq: 0,
            camera_wipe_rgba_buf: Vec::new(),

            lipsync: LipSyncGuiState {
                enabled: false,
                mic_device_index: 0,
                available_mics: crate::lipsync::audio_capture::list_audio_devices(),
                volume_threshold: 0.01,
                smoothing: 0.5,
                current_volume: 0.0,
            },
            #[cfg(feature = "lipsync")]
            lipsync_processor: None,

            animation_playing: false,

            cloth_sim_playing: false,
            cloth_rename_buf: String::new(),
            cloth_save_status: None,

            region_selection: None,
            viewport_overlay: viewport_overlay::ViewportOverlayState::default(),
            cloth_material_pick_index: 0,
            cloth_distance_stiffness: 1.0,
            cloth_bend_enabled: true,
            cloth_bend_stiffness: 0.5,
            cloth_collider_toggles: Vec::new(),
            cloth_pin_node_index: 0,
            cloth_autosave_consent: None,
            expression_weights: Vec::new(),

            scene_presets: crate::persistence::load_scene_presets(),
            scene_preset_index: None,
            scene_preset_name: String::new(),

            viewport_texture: None,
            viewport_last_frame: 0,

            library_search_query: String::new(),
            library_selected_index: None,
            library_rename_buf: String::new(),
            library_tag_buf: String::new(),
            library_show_missing: false,

            folder_watcher: None,
            watched_avatar_dirs: Vec::new(),
            thumbnail_gen: crate::renderer::thumbnail::ThumbnailGenerator::default(),
        }
    }
}

impl GuiApp {
    /// Whether the tracking worker is actually running, derived from the
    /// real worker state instead of the GUI toggle flag.
    pub fn is_tracking_active(&self) -> bool {
        self.app
            .tracking_worker
            .as_ref()
            .map_or(false, |w| w.is_running())
    }

    /// Push a notification message that will auto-dismiss after 5 seconds.
    pub fn push_notification(&mut self, msg: String) {
        self.notifications.push((msg, Instant::now()));
    }

    /// Add a path to the recent avatars list (max 10, dedup, most recent first).
    pub fn add_recent_avatar(&mut self, path: PathBuf) {
        self.recent_avatars.retain(|p| p != &path);
        self.recent_avatars.insert(0, path);
        self.recent_avatars.truncate(10);
        let _ = crate::persistence::save_recent_avatars(&self.recent_avatars);
    }

    /// Build a `RuntimeToggles` from the current GUI state.
    fn runtime_toggles(&self) -> RuntimeToggles {
        RuntimeToggles {
            tracking_enabled: self.tracking.toggle_tracking,
            spring_enabled: self.rendering.toggle_spring,
            cloth_enabled: self.rendering.toggle_cloth && self.cloth_sim_playing,
            collision_debug: self.rendering.toggle_collision_debug,
            skeleton_debug: self.rendering.toggle_skeleton_debug,
        }
    }

    /// Snapshot the current GUI state into a `ProjectState` for persistence.
    pub fn to_project_state(&self) -> ProjectState {
        let avatar_source_path = self
            .app
            .active_avatar()
            .map(|a| a.asset.source_path.to_string_lossy().into_owned());

        let avatar_source_hash = self
            .app
            .active_avatar()
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

            camera_orbit_yaw: self.camera_orbit.yaw_deg,
            camera_orbit_pitch: self.camera_orbit.pitch_deg,
            camera_orbit_pan: self.camera_orbit.pan,
            camera_orbit_distance: self.camera_orbit.distance,

            tracking_enabled: self.tracking.toggle_tracking,
            tracking_mirror: self.tracking.tracking_mirror,
            camera_resolution_index: self.tracking.camera_resolution_index,
            camera_framerate_index: self.tracking.camera_framerate_index,
            smoothing_strength: self.tracking.smoothing_strength,
            confidence_threshold: self.tracking.confidence_threshold,
            hand_tracking_enabled: self.tracking.hand_tracking_enabled,
            face_tracking_enabled: self.tracking.face_tracking_enabled,
            camera_index: self.camera_index,
            show_camera_wipe: self.show_camera_wipe,
            show_detection_annotations: self.show_detection_annotations,

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
            background_color: self.rendering.background_color,
            transparent_background: self.rendering.transparent_background,
            toggle_spring: self.rendering.toggle_spring,
            toggle_cloth: self.rendering.toggle_cloth,

            lipsync_enabled: self.lipsync.enabled,
            lipsync_mic_device_index: self.lipsync.mic_device_index,
            lipsync_volume_threshold: self.lipsync.volume_threshold,
            lipsync_smoothing: self.lipsync.smoothing,

            output_sink_index: self.output.output_sink_index,
            output_resolution_index: self.output.output_resolution_index,
            output_framerate_index: self.output.output_framerate_index,
            output_has_alpha: self.output.output_has_alpha,
            output_color_space_index: self.output.output_color_space_index,
        }
    }

    pub fn apply_profile(&mut self, profile: &profile::StreamProfile) {
        self.tracking.tracking_mirror = profile.tracking_mirror;
        self.tracking.smoothing_strength = profile.smoothing_strength;
        self.tracking.confidence_threshold = profile.confidence_threshold;
        self.rendering.main_light_dir = profile.light_direction;
        self.rendering.main_light_intensity = profile.light_intensity;
        self.rendering.ambient_intensity = profile.ambient;
        self.rendering.camera_fov = profile.camera_fov;
        self.rendering.outline_enabled = profile.outline_enabled;
        self.rendering.outline_width = profile.outline_width;
        self.rendering.outline_color = profile.outline_color;
        self.output.output_sink_index = profile.output_sink_index;
        self.output.output_resolution_index = profile.output_resolution_index;
        self.output.output_framerate_index = profile.output_framerate_index;
        self.project_dirty = true;
    }

    fn process_hotkeys(&mut self, ctx: &egui::Context) {
        if self.hotkeys.check(hotkey::HotkeyAction::TogglePause, ctx) {
            self.paused = !self.paused;
            self.push_notification(if self.paused {
                "Paused".to_string()
            } else {
                "Resumed".to_string()
            });
        }
        if self
            .hotkeys
            .check(hotkey::HotkeyAction::ToggleTracking, ctx)
        {
            self.tracking.toggle_tracking = !self.tracking.toggle_tracking;
            self.project_dirty = true;
        }
        if self.hotkeys.check(hotkey::HotkeyAction::ToggleCloth, ctx) {
            self.rendering.toggle_cloth = !self.rendering.toggle_cloth;
            self.project_dirty = true;
        }
        if self.hotkeys.check(hotkey::HotkeyAction::ResetPose, ctx) {
            self.transform.position = [0.0, 0.0, 0.0];
            self.transform.rotation = [0.0, 0.0, 0.0];
            self.transform.scale = 1.0;
            self.project_dirty = true;
        }
        if self.hotkeys.check(hotkey::HotkeyAction::ResetCamera, ctx) {
            self.camera_orbit = CameraOrbitState {
                yaw_deg: 0.0,
                pitch_deg: 0.0,
                pan: [0.0, 0.0],
                distance: 5.0,
                target_distance: 5.0,
            };
            self.project_dirty = true;
        }
        if self
            .hotkeys
            .check(hotkey::HotkeyAction::SwitchModePreview, ctx)
        {
            self.mode = AppMode::Preview;
        }
        if self
            .hotkeys
            .check(hotkey::HotkeyAction::SwitchModeAuthoring, ctx)
        {
            self.mode = AppMode::ClothAuthoring;
        }
        if self.hotkeys.check(hotkey::HotkeyAction::SaveProject, ctx) {
            let ps = self.to_project_state();
            if let Some(ref path) = self.project_path.clone() {
                match crate::persistence::save_project(&ps, path) {
                    Ok(()) => {
                        self.project_dirty = false;
                        self.push_notification("Project saved.".to_string());
                    }
                    Err(e) => {
                        self.push_notification(format!("Save failed: {}", e));
                    }
                }
            } else {
                self.push_notification("No project path set. Use Save As.".to_string());
            }
        }
        if self.hotkeys.check(hotkey::HotkeyAction::LoadAvatar, ctx) {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("VRM 1.0", &["vrm"])
                .pick_file()
            {
                top_bar::load_avatar_from_path(self, &path);
            }
        }
    }

    /// Apply a loaded `ProjectState` to the GUI fields.
    /// Avatar and overlay loading is handled separately by the caller.
    pub fn apply_project_state(&mut self, state: &ProjectState) {
        self.transform.position = state.transform_position;
        self.transform.rotation = state.transform_rotation;
        self.transform.scale = state.transform_scale;

        self.camera_orbit.yaw_deg = state.camera_orbit_yaw;
        self.camera_orbit.pitch_deg = state.camera_orbit_pitch;
        self.camera_orbit.pan = state.camera_orbit_pan;
        self.camera_orbit.distance = state.camera_orbit_distance;
        self.camera_orbit.target_distance = state.camera_orbit_distance;

        self.tracking.toggle_tracking = state.tracking_enabled;
        self.tracking.tracking_mirror = state.tracking_mirror;
        self.tracking.camera_resolution_index = state.camera_resolution_index;
        self.tracking.camera_framerate_index = state.camera_framerate_index;
        self.tracking.smoothing_strength = state.smoothing_strength;
        self.tracking.confidence_threshold = state.confidence_threshold;
        self.tracking.hand_tracking_enabled = state.hand_tracking_enabled;
        self.tracking.face_tracking_enabled = state.face_tracking_enabled;
        self.camera_index = state.camera_index;
        self.show_camera_wipe = state.show_camera_wipe;
        self.show_detection_annotations = state.show_detection_annotations;

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
        self.rendering.background_color = state.background_color;
        self.rendering.transparent_background = state.transparent_background;
        self.rendering.toggle_spring = state.toggle_spring;
        self.rendering.toggle_cloth = state.toggle_cloth;

        self.lipsync.enabled = state.lipsync_enabled;
        self.lipsync.mic_device_index = state.lipsync_mic_device_index;
        self.lipsync.volume_threshold = state.lipsync_volume_threshold;
        self.lipsync.smoothing = state.lipsync_smoothing;

        self.output.output_sink_index = state.output_sink_index;
        self.output.output_resolution_index = state.output_resolution_index;
        self.output.output_framerate_index = state.output_framerate_index;
        self.output.output_has_alpha = state.output_has_alpha;
        self.output.output_color_space_index = state.output_color_space_index;
    }

    fn load_avatar_from_drop(&mut self, path: &std::path::Path) {
        top_bar::load_avatar_from_path(self, path);
    }

    pub fn start_watching_folder(&mut self, path: std::path::PathBuf) -> Result<(), String> {
        if self.folder_watcher.is_none() {
            self.folder_watcher = Some(crate::app::folder_watcher::FolderWatcher::new());
        }

        if let Some(ref mut fw) = self.folder_watcher {
            fw.watch(&path, true)?;
            self.watched_avatar_dirs.push(path.clone());
            self.push_notification(format!("Watching folder: {}", path.display()));
        }
        Ok(())
    }

    pub fn stop_watching_folder(&mut self, path: &std::path::Path) -> Result<(), String> {
        if let Some(ref mut fw) = self.folder_watcher {
            fw.unwatch(path)?;
            self.watched_avatar_dirs.retain(|p| p != path);
            self.push_notification(format!("Stopped watching: {}", path.display()));
            if self.watched_avatar_dirs.is_empty() {
                self.folder_watcher = None;
            }
        }
        Ok(())
    }

    fn poll_folder_watcher(&mut self) {
        if let Some(ref mut fw) = self.folder_watcher {
            let events = fw.poll().to_vec();
            let vrm_events = crate::app::folder_watcher::FolderWatcher::filter_vrm(&events);
            for event in vrm_events {
                if event.kind == crate::app::folder_watcher::FolderWatchEventKind::Created {
                    for vrm_path in &event.paths {
                        if !self.app.avatar_library.find_by_path(vrm_path).is_some() {
                            let mut entry =
                                crate::app::avatar_library::AvatarLibraryEntry::from_path(vrm_path);
                            if let Ok(loader) =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    crate::asset::vrm::VrmAssetLoader::new()
                                }))
                            {
                                let loader = loader;
                                if let Ok(asset) = loader.load(&vrm_path.to_string_lossy()) {
                                    entry.update_from_asset(&asset);
                                }
                            }
                            self.app.avatar_library.add(entry);
                            self.push_notification(format!(
                                "New VRM detected: {}",
                                vrm_path.display()
                            ));
                        }
                    }
                }
            }
            let _ = crate::persistence::save_avatar_library(&self.app.avatar_library);
        }
    }
}

impl eframe::App for GuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle drag-and-drop VRM files
        let dropped: Vec<_> = ctx.input(|i| i.raw.dropped_files.clone());
        for file in dropped {
            if let Some(path) = file.path {
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                if ext == "vrm" {
                    info!("gui: dropped VRM file: {:?}", path);
                    self.load_avatar_from_drop(&path);
                }
            }
        }

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

        self.process_hotkeys(ctx);

        self.poll_folder_watcher();

        // Sync GUI-driven state into the application before running the frame.
        if let Some(avatar) = self.app.active_avatar_mut() {
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

        // Smooth zoom: lerp distance toward target_distance each frame.
        let t = (1.0 - (-5.0 * self.frame_time_ms / 1000.0).exp()) as f32;
        self.camera_orbit.distance +=
            (self.camera_orbit.target_distance - self.camera_orbit.distance) * t;

        // Sync GUI camera controls into the Application's viewport camera.
        self.app.viewport_camera.yaw_deg = self.camera_orbit.yaw_deg;
        self.app.viewport_camera.pitch_deg = self.camera_orbit.pitch_deg;
        self.app.viewport_camera.distance = self.camera_orbit.distance;
        self.app.viewport_camera.pan = self.camera_orbit.pan;
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

        // Recovery snapshot on timer (separate from project autosave)
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if self.recovery_manager.should_snapshot(now_secs) {
            let project_state = if self.project_dirty {
                Some(self.to_project_state())
            } else {
                None
            };

            let has_overlay = self.app.editor.overlay_asset.is_some();
            let overlay_for_snapshot = if has_overlay && self.cloth_autosave_consent == Some(true) {
                let overlay_name = self
                    .app
                    .editor
                    .overlay_asset
                    .as_ref()
                    .map(|a| a.metadata.name.clone())
                    .unwrap_or_else(|| "Untitled".to_string());
                let target_avatar_path = self
                    .app
                    .active_avatar()
                    .map(|a| a.asset.source_path.to_string_lossy().into_owned());
                Some(crate::persistence::ClothOverlayFile {
                    format_version: 1,
                    created_with: format!("VulVATAR {}", env!("CARGO_PKG_VERSION")),
                    last_saved_with: format!("VulVATAR {}", env!("CARGO_PKG_VERSION")),
                    overlay_name,
                    target_avatar_path,
                    cloth_asset: self.app.editor.overlay_asset.clone(),
                })
            } else {
                None
            };

            let _ = self.recovery_manager.write_snapshot(
                project_state.as_ref(),
                self.overlay_dirty,
                overlay_for_snapshot.as_ref(),
            );
        }

        // Cloth overlay autosave consent dialog
        if self.app.editor.overlay_asset.is_some() && self.cloth_autosave_consent.is_none() {
            egui::Window::new("Cloth Autosave Consent")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    ui.label(
                        "A cloth overlay is active. Allow it to be included in recovery snapshots?",
                    );
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Yes").clicked() {
                            self.cloth_autosave_consent = Some(true);
                        }
                        if ui.button("No").clicked() {
                            self.cloth_autosave_consent = Some(false);
                        }
                    });
                });
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
        let _ = crate::persistence::save_avatar_library(&self.app.avatar_library);
        crate::persistence::RecoveryManager::clear_recovery();
        info!("gui: window closing, initiating application shutdown");
        self.app.shutdown();
    }
}
