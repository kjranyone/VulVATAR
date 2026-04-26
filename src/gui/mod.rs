pub mod avatar_load;
pub mod hotkey;
pub mod inspector;
pub mod mesh_picking;
pub mod mode_nav;
pub mod profile;
pub mod status_bar;
pub mod top_bar;
pub mod viewport;
pub mod viewport_overlay;

use log::{debug, info, warn};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use eframe::egui;

use crate::app::{Application, RuntimeToggles};
use crate::persistence::ProjectState;
use crate::tracking::{TrackingSmoothingParams, DEFAULT_CONFIDENCE_THRESHOLD};

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

/// Compute camera orbit settings that frame the given AABB with the avatar's
/// head/face near the centre of the viewport.
///
/// Returns `(pan_y, distance)`. The caller is expected to write these into
/// the camera orbit state alongside yaw=0, pitch=0 to get a conventional
/// front-facing view. Yaw/pitch are left untouched so the user's preferred
/// viewing angle (if any) is preserved when only the model changes.
///
/// `fov_deg` is the vertical FOV. `aspect` is the expected viewport aspect
/// — pass `1.0` to be conservative (guarantees the model fits even in a
/// square viewport).
pub fn autoframe_aabb(aabb: &crate::asset::Aabb, fov_deg: f32, aspect: f32) -> (f32, f32) {
    if aabb.is_empty() {
        return (0.0, 5.0);
    }
    let center = aabb.center();
    let size = aabb.size();

    let tan_v = (fov_deg.to_radians() * 0.5).tan().max(1e-4);
    let tan_h = tan_v * aspect.max(1e-4);

    let half_w = size[0].max(size[2]) * 0.5;
    let half_h = size[1] * 0.5;
    let half_d = size[2] * 0.5;

    let dist_v = half_h / tan_v;
    let dist_h = half_w / tan_h;
    // +half_d so the front of the bbox doesn't land on the near plane.
    // 1.15 is a visual margin so the avatar doesn't touch the viewport edges.
    let distance = (dist_v.max(dist_h) + half_d) * 1.15;

    (center[1], distance.max(0.5))
}

/// Map a `tracking.camera_resolution_index` (combo box position) to actual
/// width/height. Kept as a free function so both the inspector (combo
/// onchange and Start Camera button) and the GUI reconciliation path resolve
/// indices identically.
pub fn camera_resolution_for_index(index: usize) -> (u32, u32) {
    match index {
        1 => (1280, 720),
        2 => (1920, 1080),
        _ => (640, 480),
    }
}

/// Re-serialise a `.vvtcloth` after a Partial rebind so the on-disk
/// IDs match the new avatar going forward. Reads the existing file
/// first to preserve everything outside of `cloth_asset` and the
/// audit fields (`last_saved_with`, `last_rebound_with`).
fn save_rebound_overlay(
    path: &std::path::Path,
    cloth_asset: &crate::asset::ClothAsset,
) -> Result<(), String> {
    let mut file = crate::persistence::load_cloth_overlay(path)?;
    file.cloth_asset = Some(cloth_asset.clone());
    let app_tag = format!("VulVATAR {}", env!("CARGO_PKG_VERSION"));
    file.last_saved_with = app_tag.clone();
    file.last_rebound_with = Some(app_tag);
    let json = serde_json::to_string_pretty(&file)
        .map_err(|e| format!("serialise rebound overlay: {}", e))?;
    crate::persistence::atomic_write(path, &json)
        .map_err(|e| format!("write rebound overlay '{}': {}", path.display(), e))
}

/// Encode a [`ThumbnailRenderResult`] (raw RGBA8 + extent) as PNG at
/// `path`. Creates the parent directory if needed. Used by GuiApp's
/// `poll_thumbnail_jobs` to finalise a real-render thumbnail produced
/// on the render thread.
fn save_thumbnail_png(
    path: &std::path::Path,
    thumb: &crate::renderer::ThumbnailRenderResult,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create thumbnail dir '{}': {}", parent.display(), e))?;
    }
    let img = image::RgbaImage::from_raw(thumb.width, thumb.height, thumb.rgba_pixels.clone())
        .ok_or_else(|| "thumbnail RGBA buffer length didn't match width*height*4".to_string())?;
    img.save(path)
        .map_err(|e| format!("save thumbnail PNG '{}': {}", path.display(), e))?;
    Ok(())
}

/// Decide what to do with a single thumbnail render response, separated
/// from [`GuiApp::poll_thumbnail_jobs`] so the failure-path contract is
/// testable without an `egui::Context`.
///
/// Returns `Ok(())` iff a fresh PNG was successfully written — only
/// then should the caller invalidate egui's image cache for that path.
///
/// **Failure-path contract**: if the render thread reported an error,
/// or the encode step itself fails, **no on-disk file is touched**.
/// A pre-existing placeholder PNG (written eagerly on avatar import)
/// stays in place so the library inspector keeps showing *something*
/// instead of going blank on a transient GPU hiccup. The caller must
/// not, in either branch, replace or unlink the file outside this
/// helper.
fn handle_thumbnail_response(
    path: &std::path::Path,
    response: Result<crate::renderer::ThumbnailRenderResult, String>,
) -> Result<(), String> {
    match response {
        Ok(thumb) => match save_thumbnail_png(path, &thumb) {
            Ok(()) => {
                info!("thumbnail: wrote real-render PNG to {}", path.display());
                Ok(())
            }
            Err(e) => {
                warn!(
                    "thumbnail: failed to write '{}': {} (keeping any existing placeholder)",
                    path.display(),
                    e
                );
                Err(e)
            }
        },
        Err(e) => {
            warn!(
                "thumbnail: render failed for '{}': {} (keeping any existing placeholder)",
                path.display(),
                e
            );
            Err(e)
        }
    }
}

/// Map a `tracking.camera_framerate_index` to the actual fps value.
pub fn camera_fps_for_index(index: usize) -> u32 {
    if index == 1 {
        60
    } else {
        30
    }
}

/// Map an `output.output_resolution_index` (Output inspector combo position)
/// to the actual width/height the renderer should render at. Note the
/// inspector orders descending: 0 = 1920×1080, 1 = 1280×720, 2 = 640×480.
pub fn output_resolution_for_index(index: usize) -> [u32; 2] {
    match index {
        1 => [1280, 720],
        2 => [640, 480],
        _ => [1920, 1080],
    }
}

/// Map an `output.output_framerate_index` to the actual fps. Inspector
/// order: 0 = 60, 1 = 30.
pub fn output_fps_for_index(index: usize) -> u32 {
    if index == 1 {
        30
    } else {
        60
    }
}

pub struct TrackingGuiState {
    pub toggle_tracking: bool,
    pub camera_resolution_index: usize,
    pub camera_framerate_index: usize,
    pub tracking_mirror: bool,
    pub hand_tracking_enabled: bool,
    pub face_tracking_enabled: bool,
    /// When true, the inference layer's model picker prefers a CIGPose
    /// `wholebody` variant over `ubody` so hip/knee/ankle keypoints are
    /// emitted and the leg humanoid bones get driven. Toggle takes
    /// effect on the next start-tracking; mid-session it's a no-op.
    pub lower_body_tracking_enabled: bool,
    pub smoothing_strength: f32,
    pub confidence_threshold: f32,
}

pub struct LipSyncGuiState {
    /// List of mics enumerated for the dropdown. Pure UI artifact —
    /// runtime state (selected mic + enabled flag) lives on Application.
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

/// Output settings the inspector binds to. The pipeline-bound fields
/// (sink) live exclusively on `Application` after Phase C — the inspector
/// reads them via getters (`app.output.active_sink()`) and writes via
/// mutators (`app.ensure_output_sink_runtime()`).
///
/// The other fields below are not yet wired to any pipeline consumer
/// (Phase B). They remain GUI-only state for now and are persisted via
/// `to_project_state` so the user's selection survives across sessions.
pub struct OutputGuiState {
    pub output_resolution_index: usize,
    pub output_framerate_index: usize,
    pub output_has_alpha: bool,
    pub output_color_space_index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppMode {
    Avatar,
    Preview,
    TrackingSetup,
    Rendering,
    Output,
    ClothAuthoring,
}

impl AppMode {
    pub const ALL: [AppMode; 6] = [
        AppMode::Avatar,
        AppMode::Preview,
        AppMode::TrackingSetup,
        AppMode::Rendering,
        AppMode::Output,
        AppMode::ClothAuthoring,
    ];

    pub fn label(&self) -> &str {
        match self {
            AppMode::Avatar => "Avatar",
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

    pub inspector_open: bool,

    pub expression_weights: Vec<f32>,

    // Scene presets
    pub scene_presets: Vec<crate::persistence::ScenePreset>,
    pub scene_preset_index: Option<usize>,
    pub scene_preset_name: String,

    // Viewport texture integration
    pub viewport_texture: Option<egui::TextureHandle>,
    pub viewport_last_frame: u64,

    // Infinite-drag state (Blender-style cursor grab during orbit/pan)
    pub viewport_cursor_grabbed: bool,
    pub viewport_drag_origin: Option<egui::Pos2>,

    // Avatar library GUI state
    pub library_search_query: String,
    pub library_selected_index: Option<usize>,
    pub library_rename_buf: String,
    pub library_tag_buf: String,
    pub library_show_missing: bool,

    pub folder_watcher: Option<crate::app::folder_watcher::FolderWatcher>,
    pub watched_avatar_dirs: Vec<std::path::PathBuf>,
    pub thumbnail_gen: crate::renderer::thumbnail::ThumbnailGenerator,

    /// Pending real-render thumbnail jobs, keyed by the destination PNG
    /// path. Each entry's receiver completes when the render thread has
    /// finished its synchronous thumbnail render. `update` drains them
    /// once per frame, writes the PNG, and tells egui to forget the
    /// cached texture for that file:// URI so the new pixels show up.
    pub pending_thumbnail_jobs: Vec<(std::path::PathBuf, std::sync::mpsc::Receiver<Result<crate::renderer::ThumbnailRenderResult, String>>)>,

    /// Background avatar load in progress, if any. Set by `top_bar::load_*`
    /// helpers, polled and cleared by `update`.
    pub avatar_load_job: Option<avatar_load::AvatarLoadJob>,
}

impl GuiApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Lets `ui.image("file://...")` decode PNG/JPEG via the `image`
        // crate. Required for thumbnail rendering in the avatar library
        // inspector — without this, file:// URIs return UnknownLoader.
        egui_extras::install_image_loaders(&cc.egui_ctx);
        let mut app = Box::new(Application::new());
        app.bootstrap();
        app.avatar_library = crate::persistence::load_avatar_library();
        if std::env::var_os("VULVATAR_AUTOSTART_VIRTUAL_CAMERA").is_some() {
            if let Err(e) = app.set_requested_sink(crate::output::FrameSink::VirtualCamera) {
                warn!("virtual-camera autostart failed: {e}");
            } else {
                info!("virtual-camera autostart enabled");
            }
        }

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

        let mut state = Self {
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
                lower_body_tracking_enabled: false,
                smoothing_strength: 0.5,
                confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
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
                available_mics: crate::lipsync::audio_capture::list_audio_devices(),
                volume_threshold: 0.01,
                smoothing: 0.5,
                current_volume: 0.0,
            },

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
            inspector_open: true,
            expression_weights: Vec::new(),

            scene_presets: crate::persistence::load_scene_presets(),
            scene_preset_index: None,
            scene_preset_name: String::new(),

            viewport_texture: None,
            viewport_last_frame: 0,
            viewport_cursor_grabbed: false,
            viewport_drag_origin: None,

            library_search_query: String::new(),
            library_selected_index: None,
            library_rename_buf: String::new(),
            library_tag_buf: String::new(),
            library_show_missing: false,

            folder_watcher: None,
            watched_avatar_dirs: Vec::new(),
            thumbnail_gen: crate::renderer::thumbnail::ThumbnailGenerator::new(
                crate::persistence::thumbnails_dir(),
            ),
            pending_thumbnail_jobs: Vec::new(),

            avatar_load_job: None,
        };

        if let Some(path) = std::env::var_os("VULVATAR_AUTOSTART_AVATAR") {
            let path = PathBuf::from(path);
            if path.exists() {
                info!("avatar autostart loading {}", path.display());
                top_bar::load_avatar_from_path(&mut state, &path);
            } else {
                warn!("avatar autostart path does not exist: {}", path.display());
            }
        }

        // Backfill placeholder thumbnails for library entries that have
        // none on disk (legacy library data + entries whose VRM lacked an
        // embedded cover image). Cheap — a 128×128 procedural draw per
        // entry. Only writes new files.
        let mut library_dirty = false;
        for entry in state.app.avatar_library.entries.iter_mut() {
            let needs = entry
                .thumbnail_path
                .as_ref()
                .map_or(true, |p| !p.exists());
            if needs {
                if let Some(path) = state
                    .thumbnail_gen
                    .generate_and_save_placeholder(&entry.name)
                {
                    entry.thumbnail_path = Some(path);
                    library_dirty = true;
                }
            }
        }
        if library_dirty {
            let _ = crate::persistence::save_avatar_library(&state.app.avatar_library);
        }

        // Restore watched-folder subscriptions persisted from a previous
        // session. Each call may fail (path went missing, permissions
        // changed, etc); failures get a notification and the path is
        // dropped from the live list. The persisted file is rewritten
        // afterwards so a missing folder doesn't keep coming back.
        for path in crate::persistence::load_watched_folders() {
            if !path.exists() {
                state.push_notification(format!(
                    "Watched folder no longer exists, dropping: {}",
                    path.display()
                ));
                continue;
            }
            if let Err(e) = state.start_watching_folder(path.clone()) {
                state.push_notification(format!(
                    "Failed to restore watch on {}: {}",
                    path.display(),
                    e
                ));
            }
        }
        let _ = crate::persistence::save_watched_folders(&state.watched_avatar_dirs);

        state
    }

    /// Build a minimal `GuiApp` for unit/integration tests. Identical to
    /// [`GuiApp::new`] in shape, but skips:
    ///
    /// * `eframe::CreationContext` (no egui context available in tests)
    /// * disk reads (avatar library, scene presets, recent avatars,
    ///   recovery snapshot, watched folders) — start with empty state
    /// * system probes (`list_cameras`, `list_audio_devices`) — empty
    ///   vecs, since the test doesn't drive the GUI
    /// * environment-driven autostart (autoload avatar, virtual-camera)
    ///
    /// The test can then populate `app.avatars` directly and exercise
    /// methods on this harness without spinning up the full GUI.
    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        let app = Box::new(Application::new());
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

            autosave_interval: None,
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
                lower_body_tracking_enabled: false,
                smoothing_strength: 0.5,
                confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
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
                output_resolution_index: 0,
                output_framerate_index: 0,
                output_has_alpha: true,
                output_color_space_index: 0,
            },

            camera_index: 0,
            available_cameras: Vec::new(),
            show_camera_wipe: false,
            show_detection_annotations: true,
            camera_wipe_texture: None,
            camera_wipe_seq: 0,
            camera_wipe_rgba_buf: Vec::new(),

            lipsync: LipSyncGuiState {
                available_mics: Vec::new(),
                volume_threshold: 0.01,
                smoothing: 0.5,
                current_volume: 0.0,
            },

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
            inspector_open: true,
            expression_weights: Vec::new(),

            scene_presets: Vec::new(),
            scene_preset_index: None,
            scene_preset_name: String::new(),

            viewport_texture: None,
            viewport_last_frame: 0,
            viewport_cursor_grabbed: false,
            viewport_drag_origin: None,

            library_search_query: String::new(),
            library_selected_index: None,
            library_rename_buf: String::new(),
            library_tag_buf: String::new(),
            library_show_missing: false,

            folder_watcher: None,
            watched_avatar_dirs: Vec::new(),
            thumbnail_gen: crate::renderer::thumbnail::ThumbnailGenerator::new(
                std::env::temp_dir().join("vulvatar_test_thumbs"),
            ),
            pending_thumbnail_jobs: Vec::new(),

            avatar_load_job: None,
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

    /// Whether the tracking worker has finished initialisation and is
    /// actively producing frames.
    pub fn is_tracking_ready(&self) -> bool {
        self.app
            .tracking_worker
            .as_ref()
            .map_or(false, |w| w.is_ready())
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

        // File-backed cloth overlays attached to the active avatar via
        // "Load Overlay File...". Procedurally-added slots have None
        // here and intentionally don't round-trip through projects.
        let cloth_overlay_paths: Vec<String> = self
            .app
            .active_avatar()
            .map(|a| {
                a.cloth_overlays
                    .iter()
                    .filter_map(|s| s.source_path.as_ref())
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();

        ProjectState {
            avatar_source_path,
            avatar_source_hash,
            active_overlay_path,
            cloth_overlay_paths,

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
            lower_body_tracking_enabled: self.tracking.lower_body_tracking_enabled,
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

            // Save user **intent** (Phase D), not the currently-running
            // state. If MF init failed this session, runtime is on a
            // SharedMemory fallback but we don't want to overwrite the
            // user's saved VirtualCamera choice.
            lipsync_enabled: self.app.requested_lipsync_enabled(),
            lipsync_mic_device_index: self.app.lipsync_mic_device_index(),
            lipsync_volume_threshold: self.lipsync.volume_threshold,
            lipsync_smoothing: self.lipsync.smoothing,

            output_sink_index: self.app.requested_sink().to_gui_index(),
            output_resolution_index: self.output.output_resolution_index,
            output_framerate_index: self.output.output_framerate_index,
            output_has_alpha: self.output.output_has_alpha,
            output_color_space_index: self.output.output_color_space_index,
        }
    }

    /// Apply the pipeline-bound parts of a saved snapshot directly to
    /// `Application` mutators (the Phase D `set_requested_*` variants so
    /// runtime failures don't overwrite the user's saved intent). After
    /// Phase C the GUI no longer holds shadow values for these settings;
    /// `Application` is the single source of truth. Failures surface as
    /// user-visible notifications; the runtime stays at its fallback state
    /// while the requested state remembers the user's pick.
    fn apply_pipeline_bound_settings(
        &mut self,
        sink_index: usize,
        lipsync_enabled: bool,
        lipsync_mic: usize,
    ) {
        use crate::output::FrameSink;
        let want_sink = FrameSink::from_gui_index(sink_index);
        if let Err(e) = self.app.set_requested_sink(want_sink) {
            self.push_notification(format!("Virtual Camera unavailable: {e}"));
        }
        if let Err(e) = self.app.set_requested_lipsync(lipsync_enabled, lipsync_mic) {
            warn!("lipsync apply failed: {e}");
            self.push_notification(format!("Lip sync failed: {e}"));
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
        self.output.output_resolution_index = profile.output_resolution_index;
        self.output.output_framerate_index = profile.output_framerate_index;
        self.project_dirty = true;
        // Profiles don't carry lipsync settings; pass current App requested
        // state to keep them unchanged.
        self.apply_pipeline_bound_settings(
            profile.output_sink_index,
            self.app.requested_lipsync_enabled(),
            self.app.lipsync_mic_device_index(),
        );
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
        self.tracking.lower_body_tracking_enabled = state.lower_body_tracking_enabled;
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

        self.lipsync.volume_threshold = state.lipsync_volume_threshold;
        self.lipsync.smoothing = state.lipsync_smoothing;

        self.output.output_resolution_index = state.output_resolution_index;
        self.output.output_framerate_index = state.output_framerate_index;
        self.output.output_has_alpha = state.output_has_alpha;
        self.output.output_color_space_index = state.output_color_space_index;

        self.apply_pipeline_bound_settings(
            state.output_sink_index,
            state.lipsync_enabled,
            state.lipsync_mic_device_index,
        );

        self.restore_cloth_overlay_paths(&state.cloth_overlay_paths);
    }

    /// Re-attach each cloth overlay listed in the project file to the
    /// active avatar. Each path is loaded via
    /// [`crate::persistence::load_cloth_overlay`] (which handles version
    /// migration), routed through the rebinder so reimported avatars
    /// don't break the binding by-id, and then attached + initialised
    /// on a fresh slot. Failed rebinds surface as notifications and
    /// skip that single overlay rather than aborting the whole apply.
    fn restore_cloth_overlay_paths(&mut self, paths: &[String]) {
        if paths.is_empty() {
            return;
        }
        // Resolve the assets first so the avatar borrow doesn't overlap
        // with notification pushes.
        let resolved: Vec<(std::path::PathBuf, crate::asset::ClothAsset)> = paths
            .iter()
            .filter_map(|p| {
                let path = std::path::PathBuf::from(p);
                if !path.exists() {
                    self.push_notification(format!(
                        "Cloth overlay '{}' not found, skipping",
                        path.display()
                    ));
                    return None;
                }
                match crate::persistence::load_cloth_overlay(&path) {
                    Ok(file) => match file.cloth_asset {
                        Some(asset) => Some((path, asset)),
                        None => {
                            self.push_notification(format!(
                                "Cloth overlay '{}' has no payload, skipping",
                                path.display()
                            ));
                            None
                        }
                    },
                    Err(e) => {
                        self.push_notification(format!(
                            "Failed to load cloth overlay '{}': {}",
                            path.display(),
                            e
                        ));
                        None
                    }
                }
            })
            .collect();

        if resolved.is_empty() {
            return;
        }

        for (path, mut cloth_asset) in resolved {
            if !self.attempt_overlay_rebind(&path, &mut cloth_asset) {
                continue; // rebind reported Failed; skip attach
            }
            if let Some(avatar) = self.app.active_avatar_mut() {
                let overlay_id = crate::asset::ClothOverlayId(
                    (avatar.cloth_overlay_count() as u64) + 2,
                );
                let idx = avatar.attach_cloth_overlay(overlay_id);
                avatar.init_cloth_overlay(idx, &cloth_asset);
                if let Some(slot) = avatar.cloth_overlays.get_mut(idx) {
                    slot.source_path = Some(path);
                }
            }
        }
    }

    /// Run the cloth-rebind pass against the active avatar and route
    /// the report status: Clean → silent debug log; Partial → user
    /// notification + write-back to disk with `last_rebound_with`
    /// stamp; Failed → notification, return `false` so the caller
    /// skips attaching the overlay.
    pub(crate) fn attempt_overlay_rebind(
        &mut self,
        path: &std::path::Path,
        cloth_asset: &mut crate::asset::ClothAsset,
    ) -> bool {
        use crate::asset::cloth_rebind::{rebind_overlay, RebindStatus};

        let report = match self.app.active_avatar() {
            Some(avatar) => rebind_overlay(cloth_asset, &avatar.asset),
            None => return true, // no avatar to rebind against; nothing to do
        };

        match report.status {
            RebindStatus::Clean => {
                debug!(
                    "rebind: '{}' clean ({} node refs, {} primitive refs walked)",
                    path.display(),
                    report.node_remappings.len(),
                    report.primitive_remappings.len()
                );
                true
            }
            RebindStatus::Partial => {
                self.push_notification(format!(
                    "Rebound overlay '{}': {} nodes + {} primitives resolved by name",
                    path.display(),
                    report.node_remappings.len(),
                    report.primitive_remappings.len(),
                ));
                if let Err(e) = save_rebound_overlay(path, cloth_asset) {
                    warn!(
                        "rebind: failed to write back rebound overlay '{}': {}",
                        path.display(),
                        e
                    );
                }
                true
            }
            RebindStatus::Failed => {
                self.push_notification(format!(
                    "Could not auto-bind overlay '{}': {} unresolved + {} geometry mismatches. \
                     Overlay was not attached.",
                    path.display(),
                    report.unresolved.len(),
                    report.fatal_geometry_changes.len(),
                ));
                false
            }
        }
    }

    fn load_avatar_from_drop(&mut self, path: &std::path::Path) {
        top_bar::load_avatar_from_path(self, path);
    }

    pub fn start_watching_folder(&mut self, path: std::path::PathBuf) -> Result<(), String> {
        if self.watched_avatar_dirs.iter().any(|p| p == &path) {
            return Ok(()); // idempotent
        }
        if self.folder_watcher.is_none() {
            self.folder_watcher = Some(crate::app::folder_watcher::FolderWatcher::new());
        }

        if let Some(ref mut fw) = self.folder_watcher {
            fw.watch(&path, true)?;
            self.watched_avatar_dirs.push(path.clone());
            self.push_notification(format!("Watching folder: {}", path.display()));
            let _ = crate::persistence::save_watched_folders(&self.watched_avatar_dirs);
        }
        Ok(())
    }

    pub fn stop_watching_folder(&mut self, path: &std::path::Path) -> Result<(), String> {
        if let Some(ref mut fw) = self.folder_watcher {
            fw.unwatch(path)?;
            self.watched_avatar_dirs.retain(|p| p != path);
            self.push_notification(format!("Stopped watching: {}", path.display()));
            let _ = crate::persistence::save_watched_folders(&self.watched_avatar_dirs);
            if self.watched_avatar_dirs.is_empty() {
                self.folder_watcher = None;
            }
        }
        Ok(())
    }

    fn poll_avatar_load_job(&mut self) {
        // Drain progress messages; bail if no terminal outcome arrived yet.
        let outcome = match self.avatar_load_job.as_mut().and_then(|j| j.poll()) {
            Some(o) => o,
            None => return,
        };

        // Worker finished — take ownership so we can mutate other GuiApp fields.
        let job = self.avatar_load_job.take().expect("job still present");
        if let Some(handle) = job.worker {
            let _ = handle.join();
        }

        match outcome {
            avatar_load::LoadOutcome::Done(asset) => {
                top_bar::finalize_avatar_load(self, &job.path, asset);
                if let avatar_load::AfterLoad::ApplyProject {
                    project_state,
                    project_path,
                    warnings,
                } = job.after_load
                {
                    self.apply_project_state(&project_state);
                    self.project_path = Some(project_path.clone());
                    self.project_dirty = false;
                    for w in &warnings.warnings {
                        self.push_notification(format!("Warning: {}", w));
                    }
                    self.push_notification(format!("Opened project: {}", project_path.display()));
                }
            }
            avatar_load::LoadOutcome::Error(e) => {
                self.push_notification(format!(
                    "Failed to load avatar '{}': {}",
                    job.path.display(),
                    e
                ));
            }
        }
    }

    /// Build a `RenderFrameInput` for a one-shot thumbnail render of
    /// `avatar`. Camera is auto-framed to the avatar's bounding box,
    /// lighting is a fixed studio default (so all entries compare
    /// fairly), background is transparent (PNG with alpha), and the
    /// pose is whatever the avatar already has. Output extent matches
    /// `crate::renderer::thumbnail::THUMBNAIL_*`.
    fn build_thumbnail_frame_input(
        &self,
        avatar: &crate::avatar::AvatarInstance,
    ) -> crate::renderer::frame_input::RenderFrameInput {
        use crate::asset::AlphaMode;
        use crate::renderer::frame_input::*;
        use crate::renderer::material::MaterialUploadRequest;
        use std::sync::Arc;

        let extent = [
            crate::renderer::thumbnail::THUMBNAIL_WIDTH,
            crate::renderer::thumbnail::THUMBNAIL_HEIGHT,
        ];

        let mesh_instances: Vec<RenderMeshInstance> = avatar
            .asset
            .meshes
            .iter()
            .flat_map(|mesh| {
                mesh.primitives.iter().map(move |prim| {
                    let material_asset = avatar
                        .asset
                        .materials
                        .iter()
                        .find(|m| m.id == prim.material_id);
                    let material_binding = material_asset
                        .map(MaterialUploadRequest::from_asset_material)
                        .unwrap_or_else(MaterialUploadRequest::default_material);
                    let outline = OutlineSnapshot {
                        enabled: material_binding.outline_width > 0.0,
                        width: material_binding.outline_width,
                        color: material_binding.outline_color,
                    };
                    let alpha_mode = match material_binding.alpha_mode {
                        AlphaMode::Opaque => RenderAlphaMode::Opaque,
                        AlphaMode::Mask(_) => RenderAlphaMode::Cutout,
                        AlphaMode::Blend => RenderAlphaMode::Blend,
                    };
                    let cull_mode = if material_binding.double_sided {
                        RenderCullMode::DoubleSided
                    } else {
                        RenderCullMode::BackFace
                    };
                    RenderMeshInstance {
                        mesh_id: mesh.id,
                        primitive_id: prim.id,
                        material_binding,
                        bounds: prim.bounds.clone(),
                        alpha_mode,
                        cull_mode,
                        outline,
                        primitive_data: Some(Arc::new(prim.clone())),
                        morph_weights: Vec::new(),
                    }
                })
            })
            .collect();

        // Auto-frame the avatar from the front (yaw=0, pitch=0) so all
        // entries share a comparable view.
        let aspect = extent[0] as f32 / extent[1].max(1) as f32;
        let fov_deg: f32 = 30.0;
        let (pan_y, distance) = autoframe_aabb(&avatar.asset.root_aabb, fov_deg, aspect);
        let cam = crate::app::ViewportCamera {
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            pan: [0.0, pan_y],
            distance,
            fov_deg,
        };
        let (view, eye_pos) = crate::app::Application::build_view_matrix(&cam);
        let projection =
            crate::app::Application::build_projection_matrix(fov_deg, aspect, 0.1, 1000.0);

        RenderFrameInput {
            camera: CameraState {
                view,
                projection,
                position_ws: eye_pos,
                viewport_extent: extent,
            },
            // Studio lighting: hard-coded so thumbnails are visually
            // consistent across avatars regardless of the user's current
            // scene lighting (which may be tuned for streaming, not
            // library browsing).
            lighting: LightingState {
                main_light_dir_ws: [-0.3, -0.7, -0.5],
                main_light_color: [1.0, 1.0, 1.0],
                main_light_intensity: 1.0,
                ambient_term: [0.3, 0.3, 0.3],
            },
            instances: vec![RenderAvatarInstance {
                instance_id: avatar.id,
                world_transform: avatar.world_transform.clone(),
                mesh_instances,
                skinning_matrices: avatar.pose.skinning_matrices.clone(),
                cloth_deform: None,
                debug_flags: RenderDebugFlags::default(),
            }],
            output_request: OutputTargetRequest {
                preview_enabled: true,
                output_enabled: true, // force the readback path
                extent,
                color_space: RenderColorSpace::Srgb,
                alpha_mode: RenderOutputAlpha::Premultiplied,
                export_mode: RenderExportMode::CpuReadback,
            },
            background_image_path: None,
            show_ground_grid: false,
            background_color: [0.0, 0.0, 0.0],
            transparent_background: true,
        }
    }

    /// Submit a thumbnail render request to the render thread for the
    /// active avatar and queue the receiver for later polling. No-op
    /// when there's no render thread or no active avatar.
    pub fn kick_thumbnail_job(&mut self, output_path: std::path::PathBuf) {
        let Some(avatar) = self.app.active_avatar() else {
            return;
        };
        // Snapshot the frame input here while we still have an immutable
        // borrow of `self.app`.
        let input = self.build_thumbnail_frame_input(avatar);
        let Some(rt) = self.app.render_thread.as_ref() else {
            return;
        };
        let rx = rt.request_thumbnail(input);
        self.pending_thumbnail_jobs.push((output_path, rx));
    }

    /// Drain any thumbnail jobs whose render thread response has
    /// arrived. On success the destination PNG is overwritten and
    /// egui's `file://` cache for that path is invalidated so the new
    /// pixels appear in the library inspector on the next frame. On
    /// failure (render error, write error, or channel disconnect) the
    /// existing on-disk PNG — typically a placeholder written on
    /// avatar import — is left untouched. See
    /// [`handle_thumbnail_response`] for the exact contract.
    fn poll_thumbnail_jobs(&mut self, ctx: &egui::Context) {
        let mut still_pending = Vec::with_capacity(self.pending_thumbnail_jobs.len());
        for (path, rx) in std::mem::take(&mut self.pending_thumbnail_jobs) {
            match rx.try_recv() {
                Ok(response) => {
                    if handle_thumbnail_response(&path, response).is_ok() {
                        // Cache invalidation is gated on a successful
                        // write only — on failure we keep the existing
                        // placeholder texture so the inspector doesn't
                        // flash to nothing.
                        ctx.forget_image(&format!("file://{}", path.display()));
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    still_pending.push((path, rx));
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    warn!(
                        "thumbnail: response channel disconnected for '{}' (keeping any existing placeholder)",
                        path.display()
                    );
                }
            }
        }
        self.pending_thumbnail_jobs = still_pending;
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
                                    entry.update_from_asset_with_thumbnail_dir(
                                        &asset,
                                        self.thumbnail_gen.output_dir(),
                                    );
                                    if entry
                                        .thumbnail_path
                                        .as_ref()
                                        .map_or(true, |p| !p.exists())
                                    {
                                        entry.thumbnail_path = self
                                            .thumbnail_gen
                                            .generate_and_save_placeholder(&entry.name);
                                    }
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

        if let Some(err) = self.app.tracking.mailbox().drain_error() {
            self.push_notification(err);
        }

        self.poll_folder_watcher();
        self.poll_thumbnail_jobs(ctx);
        self.poll_avatar_load_job();

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

        // Sync Scene Background to the Vulkan renderer's clear color. Until
        // T11 found this gap the inspector toggle was a pure egui-side hint
        // and the renderer always cleared to (0,0,0,0), which made the MF
        // virtual camera output appear all-black to clients that don't
        // honour the alpha channel (Meet / Zoom).
        self.app.background_color = self.rendering.background_color;
        self.app.transparent_background = self.rendering.transparent_background;

        // Sync output resolution preference from GUI inspector. The renderer
        // reads this through OutputTargetRequest.extent, so changing the
        // combo immediately resizes the next render's output target.
        self.app.output_extent = Some(output_resolution_for_index(
            self.output.output_resolution_index,
        ));

        // Phase B-3: throttle the output cadence to the user's selection.
        // Idempotent (just sets a Duration), so no need for change-detect.
        self.app
            .output
            .set_target_fps(output_fps_for_index(self.output.output_framerate_index));

        // Phase B-4: forward the alpha preference. Read each frame in
        // process_render_result to tag OutputFrame.alpha_mode.
        self.app.output_preserve_alpha = self.output.output_has_alpha;

        // Forward the user's colour space pick so the renderer's
        // OutputTargetRequest carries it. Stage 1 plumbing: the value
        // reaches ExportMetadata + OutputFrame.color_space; format /
        // shader / MF media type changes land in later stages.
        self.app.output_color_space = match self.output.output_color_space_index {
            1 => crate::renderer::frame_input::RenderColorSpace::LinearSrgb,
            _ => crate::renderer::frame_input::RenderColorSpace::Srgb,
        };

        if !self.paused {
            let real_dt = (self.frame_time_ms / 1000.0) as f32;
            // Clamp dt to avoid huge steps on first frame or after pauses.
            let frame_dt = real_dt.clamp(0.001, 0.1);

            // Lipsync inference is per-frame (not per-panel-render) so the
            // viseme keeps moving even when the inspector lipsync section
            // is collapsed. `step_lipsync` returns None when not active.
            if let Some(rms) = self.app.step_lipsync(
                self.lipsync.smoothing,
                self.lipsync.volume_threshold,
                frame_dt,
            ) {
                self.lipsync.current_volume = rms;
            }

            let frame_config = crate::app::FrameConfig {
                toggles: self.runtime_toggles(),
                smoothing: TrackingSmoothingParams {
                    // "Smoothing strength" in the GUI is the inverse of the
                    // solver's blend toward the new value: 0 = instant snap,
                    // 1 = never move.
                    rotation_blend: (1.0 - self.tracking.smoothing_strength).clamp(0.05, 1.0),
                    expression_blend: (1.0 - self.tracking.smoothing_strength * 0.6)
                        .clamp(0.05, 1.0),
                    joint_confidence_threshold: self.tracking.confidence_threshold,
                    face_confidence_threshold: self.tracking.confidence_threshold,
                    ..TrackingSmoothingParams::default()
                },
                material_mode_index: self.rendering.material_mode_index,
                hand_tracking_enabled: self.tracking.hand_tracking_enabled,
                face_tracking_enabled: self.tracking.face_tracking_enabled,
                lower_body_tracking_enabled: self.tracking.lower_body_tracking_enabled,
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
                    format_version: crate::persistence::OVERLAY_FORMAT_VERSION,
                    created_with: format!("VulVATAR {}", env!("CARGO_PKG_VERSION")),
                    last_saved_with: format!("VulVATAR {}", env!("CARGO_PKG_VERSION")),
                    overlay_name,
                    target_avatar_path,
                    cloth_asset: self.app.editor.overlay_asset.clone(),
                    last_rebound_with: None,
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

        // Loading-spinner overlay while a background avatar load is in flight.
        if let Some(job) = self.avatar_load_job.as_ref() {
            let stage_label = job.current_stage.label();
            let file_label = job
                .path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| job.path.display().to_string());
            egui::Window::new("Loading avatar")
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.vertical(|ui| {
                            ui.label(file_label);
                            ui.label(stage_label);
                        });
                    });
                });
        }

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

#[cfg(test)]
mod rebind_integration_tests {
    //! End-to-end coverage for the partial-rebinding path on avatar
    //! reimport. The cloth_rebind algorithm is already unit-tested
    //! (`src/asset/cloth_rebind.rs`); these tests exercise the GUI-side
    //! dispatch + persistence layer through a real `.vvtcloth` file
    //! and a real `GuiApp`.
    //!
    //! Coverage matrix:
    //!
    //! * **Clean** (name match, primary tier) — verified via
    //!   `restore_cloth_overlay_paths`: overlay is attached, no
    //!   notification, on-disk file untouched.
    //! * **Failed** (unresolved name) — overlay is *not* attached,
    //!   "Could not auto-bind" notification fires, on-disk file
    //!   untouched.
    //! * **Partial** (secondary/tertiary tier resolution) — currently
    //!   unreachable through the resolver: see the TODO at
    //!   `cloth_rebind.rs:198-204`. The persistence leg
    //!   (`save_rebound_overlay` writing `last_rebound_with`) is
    //!   covered by a focused IO test below; once tier-2/3 resolution
    //!   lands, replace that with a true E2E Partial test.
    use super::*;
    use crate::asset::{
        Aabb, AssetSourceHash, AvatarAsset, AvatarAssetId, ClothAsset, ClothOverlayId, ClothPin,
        ExpressionAssetSet, NodeId, NodeRef, SkeletonAsset, SkeletonNode, Transform, VrmMeta,
    };
    use crate::avatar::{AvatarInstance, AvatarInstanceId};
    use std::sync::Arc;

    fn make_skeleton_node(id: u64, name: &str) -> SkeletonNode {
        SkeletonNode {
            id: NodeId(id),
            name: name.to_string(),
            parent: None,
            children: vec![],
            rest_local: Transform::default(),
            humanoid_bone: None,
        }
    }

    fn make_avatar_with_node(node_id: u64, node_name: &str) -> Arc<AvatarAsset> {
        let node = make_skeleton_node(node_id, node_name);
        Arc::new(AvatarAsset {
            id: AvatarAssetId(1),
            source_path: std::path::PathBuf::from("test.vrm"),
            source_hash: AssetSourceHash([0u8; 32]),
            skeleton: SkeletonAsset {
                root_nodes: vec![NodeId(node_id)],
                nodes: vec![node],
                inverse_bind_matrices: vec![],
            },
            meshes: vec![],
            materials: vec![],
            humanoid: None,
            spring_bones: vec![],
            colliders: vec![],
            default_expressions: ExpressionAssetSet { expressions: vec![] },
            animation_clips: vec![],
            node_to_mesh: std::collections::HashMap::new(),
            vrm_meta: VrmMeta::default(),
            root_aabb: Aabb::empty(),
            loaded_from_cache: false,
        })
    }

    fn install_avatar(harness: &mut GuiApp, asset: Arc<AvatarAsset>) {
        harness
            .app
            .add_avatar(AvatarInstance::new(AvatarInstanceId(1), asset));
    }

    fn make_overlay_pinning_to(name: &str, old_id: u64) -> ClothAsset {
        let mut overlay =
            ClothAsset::new_empty(ClothOverlayId(0), AvatarAssetId(1), AssetSourceHash([0u8; 32]));
        overlay.pins.push(ClothPin {
            sim_vertex_indices: vec![0],
            binding_node: NodeRef {
                id: NodeId(old_id),
                name: name.to_string(),
            },
            offset: [0.0, 0.0, 0.0],
        });
        overlay
    }

    /// Write an overlay to disk in the canonical `ClothOverlayFile`
    /// envelope so `load_cloth_overlay` can read it back.
    fn write_overlay_file(path: &std::path::Path, asset: &ClothAsset) {
        let file = crate::persistence::ClothOverlayFile {
            format_version: crate::persistence::OVERLAY_FORMAT_VERSION,
            created_with: "test".to_string(),
            last_saved_with: "test".to_string(),
            overlay_name: "test_overlay".to_string(),
            target_avatar_path: None,
            cloth_asset: Some(asset.clone()),
            last_rebound_with: None,
        };
        let json = serde_json::to_string_pretty(&file).expect("serialise overlay file");
        crate::persistence::atomic_write(path, &json).expect("write overlay file");
    }

    /// Allocate a unique tempdir per test so concurrent runs don't
    /// collide on Windows.
    fn make_tempdir(suffix: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("vulvatar_rebind_test_{}_{}", suffix, nanos));
        std::fs::create_dir_all(&dir).expect("create tempdir");
        dir
    }

    #[test]
    fn clean_rebind_attaches_silently_and_does_not_touch_disk() {
        // Avatar exposes a "Hips" node at id=10. The overlay was authored
        // when the same name lived at id=99 — id-only validation would
        // break, but name-based primary-tier resolution should rebind it
        // cleanly, attach the overlay, and leave the file alone.
        let mut harness = GuiApp::for_test();
        install_avatar(&mut harness, make_avatar_with_node(10, "Hips"));

        let dir = make_tempdir("clean");
        let overlay_path = dir.join("clean.vvtcloth");
        let overlay_asset = make_overlay_pinning_to("Hips", 99);
        write_overlay_file(&overlay_path, &overlay_asset);

        let paths = vec![overlay_path.to_string_lossy().to_string()];
        harness.restore_cloth_overlay_paths(&paths);

        // Attached on the active avatar.
        let avatar = harness.app.active_avatar().expect("active avatar");
        assert_eq!(
            avatar.cloth_overlay_count(),
            1,
            "Clean rebind should still attach the overlay"
        );

        // No notification — Clean is a silent path.
        assert!(
            harness.notifications.is_empty(),
            "Clean rebind should not push a notification, got: {:?}",
            harness.notifications
        );

        // On-disk file's last_rebound_with stays None — Clean does not
        // trigger save_rebound_overlay.
        let on_disk = crate::persistence::load_cloth_overlay(&overlay_path)
            .expect("reload overlay after Clean rebind");
        assert!(
            on_disk.last_rebound_with.is_none(),
            "Clean must not stamp last_rebound_with"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn failed_rebind_skips_attach_and_pushes_notification() {
        // Avatar lacks the named bone the overlay was authored against —
        // primary tier fails, no fallback is wired up yet, so the
        // dispatcher must skip attach, post the "Could not auto-bind"
        // toast, and leave the file alone.
        let mut harness = GuiApp::for_test();
        install_avatar(&mut harness, make_avatar_with_node(10, "DifferentBone"));

        let dir = make_tempdir("failed");
        let overlay_path = dir.join("failed.vvtcloth");
        let overlay_asset = make_overlay_pinning_to("MissingBone", 99);
        write_overlay_file(&overlay_path, &overlay_asset);

        let paths = vec![overlay_path.to_string_lossy().to_string()];
        harness.restore_cloth_overlay_paths(&paths);

        // Did NOT attach.
        let avatar = harness.app.active_avatar().expect("active avatar");
        assert_eq!(
            avatar.cloth_overlay_count(),
            0,
            "Failed rebind must not attach the overlay"
        );

        // Notification fired with the Failed-path message.
        assert_eq!(
            harness.notifications.len(),
            1,
            "Failed rebind should post exactly one notification"
        );
        let msg = &harness.notifications[0].0;
        assert!(
            msg.contains("Could not auto-bind"),
            "expected 'Could not auto-bind' in notification, got: {}",
            msg
        );

        // Disk file unchanged — Failed does not call save_rebound_overlay.
        let on_disk = crate::persistence::load_cloth_overlay(&overlay_path)
            .expect("reload overlay after Failed rebind");
        assert!(on_disk.last_rebound_with.is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_rebound_overlay_stamps_last_rebound_with_on_disk() {
        // Persistence-leg coverage for the (currently unreachable) Partial
        // dispatch path: when the dispatcher *would* call this on Partial,
        // it must produce a file whose last_rebound_with is set and whose
        // cloth_asset payload is the rebound (not the original) one. We
        // exercise the IO directly because the algorithm cannot yet
        // produce a Partial report — see the module docstring.
        let dir = make_tempdir("save");
        let overlay_path = dir.join("partial.vvtcloth");
        let original_asset = make_overlay_pinning_to("Hips", 99);
        write_overlay_file(&overlay_path, &original_asset);

        // Simulate the dispatcher having already rewritten the IDs in
        // memory before save (matching what the Partial branch does).
        let mut rebound_asset = original_asset.clone();
        rebound_asset.pins[0].binding_node.id = NodeId(7);

        save_rebound_overlay(&overlay_path, &rebound_asset).expect("save_rebound_overlay");

        let on_disk = crate::persistence::load_cloth_overlay(&overlay_path)
            .expect("reload after save_rebound_overlay");
        let stamp = on_disk
            .last_rebound_with
            .as_ref()
            .expect("save_rebound_overlay must stamp last_rebound_with");
        assert!(
            stamp.starts_with("VulVATAR "),
            "stamp should be 'VulVATAR <version>', got: {}",
            stamp
        );
        // last_saved_with is updated too — both should match.
        assert_eq!(&on_disk.last_saved_with, stamp);
        // Payload reflects the rebound IDs, not the originals.
        let saved_asset = on_disk
            .cloth_asset
            .expect("cloth_asset preserved through save");
        assert_eq!(saved_asset.pins[0].binding_node.id, NodeId(7));

        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod thumbnail_failure_tests {
    //! Pin the failure-path contract for [`super::handle_thumbnail_response`]:
    //! when the render thread reports a failure (or the encode itself
    //! errors), no on-disk PNG is touched. A placeholder written by the
    //! avatar-import path stays in place so the library inspector keeps
    //! showing *something* when the GPU briefly misbehaves.
    //!
    //! Without this guarantee, a transient thumbnail failure would
    //! delete the placeholder and the library would flash empty
    //! squares — that is the regression these tests prevent.

    use super::handle_thumbnail_response;
    use crate::renderer::ThumbnailRenderResult;

    fn make_tempdir(suffix: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("vulvatar_thumb_failure_{}_{}", suffix, nanos));
        std::fs::create_dir_all(&dir).expect("create tempdir");
        dir
    }

    /// 4x4 RGBA test image (64 bytes) with a deterministic gradient so a
    /// successful round-trip is identifiable against the bytes.
    fn make_thumbnail() -> ThumbnailRenderResult {
        let w = 4u32;
        let h = 4u32;
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                rgba.push(((x * 64) & 0xff) as u8);
                rgba.push(((y * 64) & 0xff) as u8);
                rgba.push(0x40);
                rgba.push(0xff);
            }
        }
        ThumbnailRenderResult {
            width: w,
            height: h,
            rgba_pixels: rgba,
        }
    }

    #[test]
    fn render_failure_does_not_create_a_file() {
        let dir = make_tempdir("noprior");
        let path = dir.join("thumb.png");
        assert!(!path.exists());

        let outcome = handle_thumbnail_response(&path, Err("simulated GPU OOM".to_string()));
        assert!(outcome.is_err(), "failure response must surface as Err");
        assert!(
            !path.exists(),
            "render failure must not materialise an empty PNG"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn render_failure_preserves_existing_placeholder() {
        // Avatar import already wrote a placeholder; a subsequent
        // real-render failure must not overwrite or unlink it.
        let dir = make_tempdir("preserve");
        let path = dir.join("thumb.png");
        let placeholder_bytes: &[u8] = b"placeholder-bytes";
        std::fs::write(&path, placeholder_bytes).expect("seed placeholder");

        let outcome = handle_thumbnail_response(
            &path,
            Err("simulated render thread panic".to_string()),
        );
        assert!(outcome.is_err());

        let after = std::fs::read(&path).expect("placeholder still readable");
        assert_eq!(
            after, placeholder_bytes,
            "render failure must leave the placeholder bytes verbatim"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn render_success_writes_png_and_overwrites_placeholder() {
        // Sanity: the success path actually does write — otherwise the
        // failure-path tests would pass for the wrong reason (helper
        // never writes anything).
        let dir = make_tempdir("success");
        let path = dir.join("thumb.png");
        std::fs::write(&path, b"placeholder").expect("seed placeholder");

        let outcome = handle_thumbnail_response(&path, Ok(make_thumbnail()));
        assert!(outcome.is_ok());

        let bytes = std::fs::read(&path).expect("PNG written");
        // PNG magic header — confirms the placeholder was replaced with
        // a real PNG, not just untouched.
        assert!(
            bytes.len() > 8 && bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
            "success path must write a real PNG; got first bytes {:?}",
            &bytes[..bytes.len().min(8)]
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
