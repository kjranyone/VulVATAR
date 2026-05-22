pub mod avatar_load;
pub mod calibration;
pub mod components;
mod folder_watcher;
pub mod hotkey;
pub mod inspector;
pub mod mesh_picking;
pub mod mode_nav;
pub mod notifications;
pub mod profile;
mod project;
mod snapshot;
pub mod status_bar;
pub mod theme;
pub mod top_bar;
pub mod viewport;

use log::{info, warn};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use eframe::egui;

use crate::app::{Application, RuntimeToggles};
use crate::t;
use crate::tracking::{TrackingErrorLevel, TrackingSmoothingParams};

/// Build an `egui::FontDefinitions` whose fallback chain matches the
/// active locale's preferred CJK font, falling back through the others.
///
/// Returns `None` when none of the three Noto Sans CJK subsets are
/// installed under `assets/` — egui then keeps its default Latin-only
/// fonts. Run `dev.ps1` `Install-Font` to download them.
///
/// Locale-aware ordering matters because most CJK Unified Ideographs
/// exist in all three fonts but render with different glyph shapes
/// (e.g. 飞 vs 飛, simplified vs traditional vs Japanese forms). The
/// first font in the chain wins per glyph, so the locale's "native"
/// font goes first and the others come after as tofu-prevention
/// fallbacks (Hangul → KR only; Korean-locale users see SC-only chars
/// via the SC fallback rather than missing glyphs).
pub(crate) fn build_font_definitions(locale: &str) -> Option<egui::FontDefinitions> {
    let cjk_candidates: &[(&str, &str)] = &[
        ("noto_sans_jp", "NotoSansJP-Regular.otf"),
        ("noto_sans_kr", "NotoSansKR-Regular.otf"),
        ("noto_sans_sc", "NotoSansSC-Regular.otf"),
    ];

    let mut fonts = egui::FontDefinitions::default();
    let mut available_cjk: Vec<&str> = Vec::new();

    for (name, file) in cjk_candidates {
        let path = std::path::Path::new("assets").join(file);
        if !path.exists() {
            warn!(
                "CJK font missing: {} — run dev.ps1 Install-Font",
                path.display()
            );
            continue;
        }
        match std::fs::read(&path) {
            Ok(data) => {
                fonts.font_data.insert(
                    (*name).into(),
                    Arc::new(egui::FontData::from_owned(data)),
                );
                available_cjk.push(name);
                info!("loaded CJK font: {}", path.display());
            }
            Err(e) => warn!("failed to read CJK font {}: {e}", path.display()),
        }
    }

    // Material Symbols Rounded — drives the icon glyphs surfaced via
    // `theme::typography::icon`. Optional: when the file is missing
    // the GUI degrades to tofu in icon slots but keeps working, so
    // dev workflows don't break before Install-Font has run.
    let symbols_path = std::path::Path::new("assets").join("MaterialSymbolsRounded.ttf");
    let icon_loaded = if symbols_path.exists() {
        match std::fs::read(&symbols_path) {
            Ok(data) => {
                fonts.font_data.insert(
                    "material_symbols".into(),
                    Arc::new(egui::FontData::from_owned(data)),
                );
                fonts.families.insert(
                    egui::FontFamily::Name(theme::ICON_FAMILY.into()),
                    vec!["material_symbols".to_string()],
                );
                info!("loaded icon font: {}", symbols_path.display());
                true
            }
            Err(e) => {
                warn!("failed to read icon font {}: {e}", symbols_path.display());
                false
            }
        }
    } else {
        warn!(
            "icon font missing: {} — run dev.ps1 Install-Font",
            symbols_path.display()
        );
        false
    };

    if available_cjk.is_empty() && !icon_loaded {
        return None;
    }

    let (primary, fallbacks): (&str, &[&str]) = match locale {
        "ko" => ("noto_sans_kr", &["noto_sans_jp", "noto_sans_sc"]),
        "zh" => ("noto_sans_sc", &["noto_sans_jp", "noto_sans_kr"]),
        _ => ("noto_sans_jp", &["noto_sans_kr", "noto_sans_sc"]),
    };

    let mut chain: Vec<&str> = Vec::new();
    for name in std::iter::once(primary).chain(fallbacks.iter().copied()) {
        if available_cjk.contains(&name) {
            chain.push(name);
        }
    }

    for name in chain {
        for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            fonts
                .families
                .entry(family)
                .or_default()
                .push(name.into());
        }
    }

    Some(fonts)
}

use notifications::Notification;

/// Which sort the user picked in the Model Library card. The library
/// itself doesn't store this — it just exposes mutating
/// `sort_by_name` / `sort_by_last_loaded` / `sort_favorites_first`
/// methods — so the GUI tracks the current visual state separately
/// to drive the chip "selected" rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LibrarySortMode {
    #[default]
    None,
    Name,
    Recent,
    Favorites,
}

/// Scene-preset UI state lifted out of `GuiApp` (architecture
/// finding #10). The preset library is loaded eagerly from disk;
/// the selected-index / name widget buffers track the rendering
/// inspector's preset combo and rename field.
#[derive(Default)]
pub struct ScenePresetUiState {
    pub presets: Vec<crate::persistence::ScenePreset>,
    pub selected_index: Option<usize>,
    pub name: String,
}

/// Cloth-authoring panel UI state lifted out of `GuiApp` (architecture
/// finding #10). Bundles the cloth-sim playback toggle, the
/// rename / save-status widget buffers, the per-vertex region
/// selection driven by mesh picking, and the sim-parameter widgets
/// (distance stiffness, bend stiffness, collider toggles, pin node).
pub struct ClothAuthoringUiState {
    pub sim_playing: bool,
    pub rename_buf: String,
    pub save_status: Option<String>,
    pub region_selection: Option<crate::editor::cloth_authoring::RegionSelection>,
    pub material_pick_index: usize,
    pub distance_stiffness: f32,
    pub bend_enabled: bool,
    pub bend_stiffness: f32,
    pub collider_toggles: Vec<bool>,
    pub pin_node_index: usize,
    pub autosave_consent: Option<bool>,
}

impl Default for ClothAuthoringUiState {
    fn default() -> Self {
        Self {
            sim_playing: false,
            rename_buf: String::new(),
            save_status: None,
            region_selection: None,
            material_pick_index: 0,
            // Cloth-sim defaults match the values the inspector
            // previously hard-coded into `GuiApp::new`.
            distance_stiffness: 1.0,
            bend_enabled: true,
            bend_stiffness: 0.5,
            collider_toggles: Vec::new(),
            pin_node_index: 0,
            autosave_consent: None,
        }
    }
}

/// Viewport-pane UI state lifted out of `GuiApp` (architecture
/// finding #10). Holds the texture egui binds the rendered scene to,
/// the cursor-grab state during orbit/pan drags, and the camera-wipe
/// PIP overlay (toggle + texture + dedup seq + rgba scratch).
#[derive(Default)]
pub struct ViewportUiState {
    /// egui texture handle the renderer's CPU readback uploads pixels
    /// into. The viewport draws this handle each frame; `None` until
    /// the first render result arrives.
    pub texture: Option<egui::TextureHandle>,
    /// Frame counter of the last render result we uploaded, used to
    /// avoid re-uploading the same pixels.
    pub last_frame: u64,
    /// Blender-style infinite-drag during orbit/pan: cursor is hidden +
    /// warped back to the drag origin every frame so the user can
    /// drag past screen edges. `true` while a drag is active.
    pub cursor_grabbed: bool,
    /// Cursor position at drag start; restored when the drag ends.
    /// `None` when no drag is in progress.
    pub drag_origin: Option<egui::Pos2>,
    /// Whether the camera-wipe PIP (live webcam preview) is shown.
    pub show_camera_wipe: bool,
    /// Whether the 2D detection annotation overlay is drawn over the
    /// camera-wipe PIP.
    pub show_detection_annotations: bool,
    /// egui texture for the camera-wipe PIP (separate from the
    /// calibration preview's texture so toggling either doesn't tear
    /// the other).
    pub camera_wipe_texture: Option<egui::TextureHandle>,
    /// Preview-mailbox sequence of the last frame we uploaded into
    /// `camera_wipe_texture`. Driven off `preview_sequence`, not
    /// pose `sequence` — the latter can advance mid-snapshot with
    /// the previous frame still in the mailbox and would permanently
    /// strand a frame.
    pub camera_wipe_seq: u64,
    /// RGBA scratch buffer reused across uploads so the wipe doesn't
    /// reallocate per frame.
    pub camera_wipe_rgba_buf: Vec<u8>,
}

/// Library-panel UI state lifted out of `GuiApp` (architecture
/// finding #10). Aggregates the search/selection/rename widget
/// buffers, the folder watcher + watched-dirs list, the thumbnail
/// generator, the pending real-render thumbnail-job queue, and the
/// background avatar-load job slot.
#[derive(Default)]
pub struct LibraryUiState {
    pub search_query: String,
    pub selected_index: Option<usize>,
    pub rename_buf: String,
    pub tag_buf: String,
    pub show_missing: bool,
    /// Which sort mode the user last selected via the chip row. Used
    /// purely for the chip's "selected" rendering — the actual entry
    /// order is mutated in place by `AvatarLibrary::sort_*`.
    pub sort_mode: LibrarySortMode,
    pub folder_watcher: Option<crate::app::folder_watcher::FolderWatcher>,
    pub watched_avatar_dirs: Vec<std::path::PathBuf>,
    pub thumbnail_gen: crate::renderer::thumbnail::ThumbnailGenerator,
    /// Pending real-render thumbnail jobs, keyed by the destination PNG
    /// path. Each entry's receiver completes when the render thread has
    /// finished its synchronous thumbnail render. `update` drains them
    /// once per frame, writes the PNG, and tells egui to forget the
    /// cached texture for that file:// URI so the new pixels show up.
    pub pending_thumbnail_jobs:
        Vec<(std::path::PathBuf, std::sync::mpsc::Receiver<Result<crate::renderer::ThumbnailRenderResult, String>>)>,
    /// Background avatar load in progress, if any. Set by `top_bar::load_*`
    /// helpers, polled and cleared by `update`.
    pub avatar_load_job: Option<avatar_load::AvatarLoadJob>,
}

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
    /// When true, the avatar's `Hips` follows the subject's side-step /
    /// lean / crouch (translation, on top of body-yaw rotation). When
    /// false the avatar pivots in place, keeping the framing stable.
    pub root_translation_enabled: bool,
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

pub struct SettingsGuiState {
    pub locale: String,
    pub zoom_sensitivity: f32,
    pub orbit_sensitivity: f32,
    pub pan_sensitivity: f32,
}

/// Per-frame timing + pause state lifted out of `GuiApp`
/// (architecture finding #10). Updated each frame's `update` entry by
/// the EMA smoothing block; consumed by the status bar's "frame N |
/// fps | frame_time_ms" readout and by the simulation step that uses
/// `frame_time_ms` as its `frame_dt` source.
pub struct RuntimeStatusUi {
    pub paused: bool,
    pub frame_count: u64,
    /// Wall-clock instant at the previous `update` entry. Subtracted
    /// from `now` each frame to derive the dt that drives the EMA
    /// smoothing of `frame_time_ms` and `fps`.
    pub last_frame_instant: Instant,
    /// Exponential-moving-average frame interval in milliseconds.
    /// Also doubles as the simulation step `frame_dt` source so the
    /// avatar's animation playback ticks at wall-clock pace rather
    /// than at the renderer's variable cadence.
    pub frame_time_ms: f64,
    /// Exponential-moving-average frame rate in Hz. Display-only —
    /// every consumer that needs a step duration reads
    /// `frame_time_ms` instead.
    pub fps: f64,
}

impl RuntimeStatusUi {
    pub fn new(now: Instant) -> Self {
        Self {
            paused: false,
            frame_count: 0,
            last_frame_instant: now,
            frame_time_ms: 16.0,
            fps: 60.0,
        }
    }
}

/// Project lifecycle + persistence status lifted out of `GuiApp`
/// (architecture finding #10). Owns the loaded project path, the
/// recent-avatar MRU list, the dirty flags that drive autosave, the
/// crash-recovery manager, and the throttle clock that decides when
/// a dirty mutation actually hits disk.
pub struct ProjectStatusUi {
    /// Filesystem path of the project file currently open, if any.
    /// `None` for the "untitled new project" mode.
    pub project_path: Option<PathBuf>,
    /// MRU list of recently loaded avatars; surfaced as the
    /// "Open Recent" submenu and persisted via
    /// `crate::persistence::save_recent_avatars`.
    pub recent_avatars: Vec<PathBuf>,
    /// Set whenever the in-memory project state diverges from disk.
    /// The per-frame autosave block compares
    /// `last_autosave.elapsed()` against `AUTOSAVE_THROTTLE`; once
    /// the throttle clears, a `save_project` call flips this back
    /// to `false`.
    pub project_dirty: bool,
    /// Immediate-save throttle clock. Updated each time the project
    /// is saved.
    pub last_autosave: Instant,
    /// Crash-recovery manager: writes a sidecar snapshot on every
    /// dirty mutation so an abnormal exit can be recovered from
    /// next session.
    pub recovery_manager: crate::persistence::RecoveryManager,
    /// Set whenever the active cloth overlay is in a dirty state
    /// (region edit, parameter change, rebind) and the user hasn't
    /// hit save yet. Drives the "Save Overlay" button's enabled
    /// state and the unload confirmation prompt.
    pub overlay_dirty: bool,
    /// Set whenever the in-memory `profiles` library diverges from
    /// disk (`%APPDATA%\VulVATAR\profiles.json`). Mirrors
    /// `project_dirty` but flushes via `save_profiles`.
    pub profiles_dirty: bool,
}

impl ProjectStatusUi {
    pub fn new(
        recent_avatars: Vec<PathBuf>,
        recovery_manager: crate::persistence::RecoveryManager,
        now: Instant,
    ) -> Self {
        Self {
            project_path: None,
            recent_avatars,
            project_dirty: false,
            last_autosave: now,
            recovery_manager,
            overlay_dirty: false,
            profiles_dirty: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppMode {
    Avatar,
    Preview,
    TrackingSetup,
    Rendering,
    Output,
    ClothAuthoring,
    Settings,
}

impl AppMode {
    pub const ALL: [AppMode; 7] = [
        AppMode::Avatar,
        AppMode::Preview,
        AppMode::TrackingSetup,
        AppMode::Rendering,
        AppMode::Output,
        AppMode::ClothAuthoring,
        AppMode::Settings,
    ];

    pub fn label(&self) -> String {
        match self {
            AppMode::Avatar => t!("app.modes.avatar"),
            AppMode::Preview => t!("app.modes.preview"),
            AppMode::TrackingSetup => t!("app.modes.tracking_setup"),
            AppMode::Rendering => t!("app.modes.rendering"),
            AppMode::Output => t!("app.modes.output"),
            AppMode::ClothAuthoring => t!("app.modes.cloth_authoring"),
            AppMode::Settings => t!("app.modes.settings"),
        }
    }
}

pub struct GuiApp {
    pub mode: AppMode,
    pub app: Box<Application>,

    /// Per-frame timing + pause flag, see `RuntimeStatusUi`.
    pub runtime_status: RuntimeStatusUi,

    /// Project lifecycle / persistence status, see `ProjectStatusUi`.
    pub project_status: ProjectStatusUi,

    // Hotkeys & profiles
    pub hotkeys: hotkey::HotkeyMap,
    pub profiles: profile::ProfileLibrary,

    // M10: Notification/toast system
    pub notifications: Vec<Notification>,

    pub transform: TransformState,
    pub camera_orbit: CameraOrbitState,
    pub tracking: TrackingGuiState,
    pub rendering: RenderingGuiState,
    pub output: OutputGuiState,
    pub settings: SettingsGuiState,

    pub camera_index: usize,
    pub available_cameras: Vec<crate::tracking::CameraInfo>,

    // Viewport-pane state: rendered-scene texture handle, the
    // Blender-style drag-grab state, and the camera-wipe PIP toggle +
    // its companion texture / scratch buffer. See `ViewportUiState`.
    pub viewport: ViewportUiState,

    // Pose-calibration modal (Phase B). Aggregated into
    // `calibration::CalibrationUiState` as part of the GuiApp state
    // split — see plan/architecture-pipeline-gui-critical-review.md #10.
    pub calibration: calibration::CalibrationUiState,

    // Lip sync
    pub lipsync: LipSyncGuiState,

    // Animation inspector state
    pub animation_playing: bool,

    // Cloth-authoring panel state (sim playback, rename / save
    // buffers, region selection, sim-parameter widgets). See
    // `ClothAuthoringUiState`.
    pub cloth_authoring: ClothAuthoringUiState,

    pub inspector_open: bool,

    pub expression_weights: Vec<f32>,

    // Scene preset library + inspector combo + rename buffer.
    pub scene_preset: ScenePresetUiState,

    // Avatar library GUI state
    // Avatar library panel state (search, selection, sort, thumbnail
    // pipeline, folder watching, background avatar-load job). See
    // `LibraryUiState` for the breakdown.
    pub library: LibraryUiState,

    /// When true, `save_avatar_library` and other persistence writes are
    /// suppressed so unit tests never clobber the user's real data.
    test_no_persist: bool,
}

impl GuiApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Lets `ui.image("file://...")` decode PNG/JPEG via the `image`
        // crate. Required for thumbnail rendering in the avatar library
        // inspector — without this, file:// URIs return UnknownLoader.
        egui_extras::install_image_loaders(&cc.egui_ctx);

        if let Some(fonts) = build_font_definitions(&crate::i18n::locale()) {
            cc.egui_ctx.set_fonts(fonts);
        }
        theme::apply(&cc.egui_ctx);
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

            runtime_status: RuntimeStatusUi::new(Instant::now()),

            project_status: ProjectStatusUi::new(
                crate::persistence::load_recent_avatars(),
                crate::persistence::RecoveryManager::new(120),
                Instant::now(),
            ),

            hotkeys: hotkey::HotkeyMap::new(),
            // Profiles persist across project loads — calibration
            // (which is per-room/setup, not per-scene) lives on
            // them, so a fresh-install fallback to the built-in
            // presets is fine but a parse failure of an existing
            // file should not silently wipe out the user's data
            // (load_profiles already logs and returns None then).
            profiles: crate::persistence::load_profiles().unwrap_or_default(),

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
                root_translation_enabled: true,
            },
            rendering: RenderingGuiState {
                material_mode_index: 2,
                background_color: [0.1, 0.1, 0.1],
                transparent_background: true,
                camera_fov: 60.0,
                main_light_dir: [0.5, -1.0, 0.3],
                main_light_intensity: 1.0,
                ambient_intensity: [0.2, 0.2, 0.2],
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

            settings: SettingsGuiState {
                locale: crate::i18n::locale(),
                zoom_sensitivity: 0.1,
                orbit_sensitivity: 0.3,
                pan_sensitivity: 1.0,
            },

            camera_index: 0,
            available_cameras: crate::tracking::list_cameras(),
            viewport: ViewportUiState {
                show_detection_annotations: true,
                ..ViewportUiState::default()
            },

            calibration: calibration::CalibrationUiState::default(),

            lipsync: LipSyncGuiState {
                available_mics: crate::lipsync::audio_capture::list_audio_devices(),
                volume_threshold: 0.01,
                smoothing: 0.5,
                current_volume: 0.0,
            },

            animation_playing: false,

            cloth_authoring: ClothAuthoringUiState::default(),
            inspector_open: true,
            expression_weights: Vec::new(),

            scene_preset: ScenePresetUiState {
                presets: crate::persistence::load_scene_presets(),
                ..ScenePresetUiState::default()
            },

            library: LibraryUiState {
                thumbnail_gen: crate::renderer::thumbnail::ThumbnailGenerator::new(
                    crate::persistence::thumbnails_dir(),
                ),
                ..LibraryUiState::default()
            },
            test_no_persist: false,
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

        // Implicit "last session" restore: if the user never saved a
        // project, we still want their checkbox / slider values to
        // survive a quit. The same path is written by the per-frame
        // autosave block whenever `project_dirty` is set and
        // `project_path` is `None`. Leave `project_path` unset so the
        // user doesn't see a phantom file in the title bar — they
        // never opened one.
        let last_session = crate::persistence::last_session_path();
        if state.project_status.project_path.is_none() && last_session.exists() {
            match crate::persistence::load_project(&last_session) {
                Ok((project_state, _warnings)) => {
                    state.apply_project_state(&project_state);
                    state.project_status.project_dirty = false;
                    info!(
                        "persistence: restored last session from {}",
                        last_session.display()
                    );
                }
                Err(e) => {
                    warn!(
                        "persistence: could not restore last session at {}: {}",
                        last_session.display(),
                        e
                    );
                }
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
                .is_none_or(|p| !p.exists());
            if needs {
                if let Some(path) = state
                    .library.thumbnail_gen
                    .generate_and_save_placeholder(&entry.name)
                {
                    entry.thumbnail_path = Some(path);
                    library_dirty = true;
                }
            }
        }
        if library_dirty {
            state.save_avatar_library_with_toast();
        }

        // Restore watched-folder subscriptions persisted from a previous
        // session. Each call may fail (path went missing, permissions
        // changed, etc); failures get a notification and the path is
        // dropped from the live list. The persisted file is rewritten
        // afterwards so a missing folder doesn't keep coming back.
        for path in crate::persistence::load_watched_folders() {
            if !path.exists() {
                state.push_notification(t!("toast.watched_folder_missing", path = path.display().to_string()));
                continue;
            }
            if let Err(e) = state.start_watching_folder(path.clone()) {
                state.push_notification(t!("toast.failed_restore_watch", path = path.display().to_string(), error = e.to_string()));
            }
        }
        let _ = crate::persistence::save_watched_folders(&state.library.watched_avatar_dirs);

        // Seed the active profile's pose calibration into Application
        // + tracking mailbox at startup. Without this, a user who
        // calibrated last session and re-launches sees the loaded
        // calibration sitting on the profile but the depth pipeline
        // operating on the auto-EMA fallback until the first manual
        // profile-switch. We deliberately don't call apply_profile()
        // here — only this *one* field needs forwarding at startup;
        // the rest of apply_profile would clobber render / output
        // settings that load_state will set from the project file
        // a moment later.
        if let Some(active) = state.profiles.active() {
            let cal = active.pose_calibration.clone();
            state.app.tracking_calibration.pose = cal.clone();
            state.app.tracking.mailbox().set_calibration(cal);
        }

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

            runtime_status: RuntimeStatusUi::new(Instant::now()),

            project_status: ProjectStatusUi::new(
                Vec::new(),
                crate::persistence::RecoveryManager::new(120),
                Instant::now(),
            ),

            hotkeys: hotkey::HotkeyMap::new(),
            profiles: profile::ProfileLibrary::new(),

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
                root_translation_enabled: true,
            },
            rendering: RenderingGuiState {
                material_mode_index: 2,
                background_color: [0.1, 0.1, 0.1],
                transparent_background: true,
                camera_fov: 60.0,
                main_light_dir: [0.5, -1.0, 0.3],
                main_light_intensity: 1.0,
                ambient_intensity: [0.2, 0.2, 0.2],
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

            settings: SettingsGuiState {
                locale: "en".to_string(),
                zoom_sensitivity: 0.1,
                orbit_sensitivity: 0.3,
                pan_sensitivity: 1.0,
            },

            camera_index: 0,
            available_cameras: Vec::new(),
            viewport: ViewportUiState {
                show_detection_annotations: true,
                ..ViewportUiState::default()
            },

            calibration: calibration::CalibrationUiState::default(),

            lipsync: LipSyncGuiState {
                available_mics: Vec::new(),
                volume_threshold: 0.01,
                smoothing: 0.5,
                current_volume: 0.0,
            },

            animation_playing: false,

            cloth_authoring: ClothAuthoringUiState::default(),
            inspector_open: true,
            expression_weights: Vec::new(),

            scene_preset: ScenePresetUiState::default(),

            library: LibraryUiState {
                thumbnail_gen: crate::renderer::thumbnail::ThumbnailGenerator::new(
                    std::env::temp_dir().join("vulvatar_test_thumbs"),
                ),
                ..LibraryUiState::default()
            },
            test_no_persist: true,
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
            .is_some_and(|w| w.is_running())
    }

    /// Whether the tracking worker has finished initialisation and is
    /// actively producing frames.
    pub fn is_tracking_ready(&self) -> bool {
        self.app
            .tracking_worker
            .as_ref()
            .is_some_and(|w| w.is_ready())
    }

    /// Save the avatar library to disk and surface failures as an error
    /// toast. Every former call site used `let _ = save_avatar_library(...)`
    /// — silently swallowing the error meant a full disk or read-only
    /// %APPDATA% would lose the user's library on next launch with no
    /// indication anything went wrong. Centralised here so the toast
    /// wording stays consistent and so future call sites can't regress
    /// to the silent pattern.
    pub fn save_avatar_library_with_toast(&mut self) {
        if let Err(e) = crate::persistence::save_avatar_library(&self.app.avatar_library) {
            warn!("persistence: save_avatar_library failed: {}", e);
            self.push_error_notification(t!("toast.library_save_failed", error = e.to_string()));
        }
    }

    /// Add a path to the recent avatars list (max 10, dedup, most recent first).
    pub fn add_recent_avatar(&mut self, path: PathBuf) {
        self.project_status.recent_avatars.retain(|p| p != &path);
        self.project_status.recent_avatars.insert(0, path);
        self.project_status.recent_avatars.truncate(10);
        let _ = crate::persistence::save_recent_avatars(&self.project_status.recent_avatars);
    }

    /// Build a `RuntimeToggles` from the current GUI state.
    fn runtime_toggles(&self) -> RuntimeToggles {
        RuntimeToggles {
            tracking_enabled: self.tracking.toggle_tracking,
            spring_enabled: self.rendering.toggle_spring,
            cloth_enabled: self.rendering.toggle_cloth && self.cloth_authoring.sim_playing,
            collision_debug: self.rendering.toggle_collision_debug,
            skeleton_debug: self.rendering.toggle_skeleton_debug,
        }
    }

    /// Push the current calibration-modal mode to the tracking mailbox
    /// when it differs from the last value pushed. Idempotent — safe to
    /// call multiple times per frame. The repaint loop calls this both
    /// before `run_frame` (catches mode-button switches in `Idle`) and
    /// after `draw_modal` (catches Cancel / Esc / Done auto-close
    /// transitions whose state change happens inside `draw_modal`,
    /// after the pre-`run_frame` push has already run).
    fn sync_calibration_mode_hint(&mut self) {
        let current = self.calibration.modal.active_mode();
        if current != self.calibration.last_pushed_mode_hint {
            self.app
                .tracking
                .mailbox()
                .set_calibration_mode_hint(current);
            self.calibration.last_pushed_mode_hint = current;
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
                    top_bar::load_avatar_from_path(self, &path);
                }
            }
        }

        // E10: Measure frame timing.
        let now = Instant::now();
        let elapsed = now.duration_since(self.runtime_status.last_frame_instant);
        self.runtime_status.last_frame_instant = now;
        let dt_secs = elapsed.as_secs_f64();
        // Exponential moving average for smoothing.
        self.runtime_status.frame_time_ms = self.runtime_status.frame_time_ms * 0.9 + (dt_secs * 1000.0) * 0.1;
        if dt_secs > 0.0 {
            self.runtime_status.fps = self.runtime_status.fps * 0.9 + (1.0 / dt_secs) * 0.1;
        }

        self.process_hotkeys(ctx);

        if let Some((err, level)) = self.app.tracking.mailbox().drain_error() {
            match level {
                TrackingErrorLevel::Blocking => self.push_error_notification(err),
                TrackingErrorLevel::Warning => self.push_notification(err),
            }
        }

        self.poll_folder_watcher();
        self.poll_thumbnail_jobs(ctx);
        self.poll_avatar_load_job();

        // Forward the calibration-mode hint on edges. Called twice per
        // update — once here (covers "user switched mode via the
        // segmented buttons in Idle"), once after `draw_modal` (covers
        // Cancel / Esc / auto-close transitions that happen *inside*
        // `draw_modal`, which would otherwise leave a stale
        // `Some(UpperBody|FullBody)` on the worker until the next
        // unrelated repaint reason fires).
        self.sync_calibration_mode_hint();

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
        let t = (1.0 - (-5.0 * self.runtime_status.frame_time_ms / 1000.0).exp()) as f32;
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
        // P3-03: routes through `RuntimeGpuBudget` so the user's intent is
        // stored as `user_render_fps` and the *effective* target may be
        // clamped below it under pressure. Idempotent so calling each
        // GUI update is cheap.
        self.app
            .set_user_render_fps(output_fps_for_index(self.output.output_framerate_index));

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

        if !self.runtime_status.paused {
            let real_dt = (self.runtime_status.frame_time_ms / 1000.0) as f32;
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
                smoothing: TrackingSmoothingParams::default(),
                material_mode_index: self.rendering.material_mode_index,
                hand_tracking_enabled: self.tracking.hand_tracking_enabled,
                face_tracking_enabled: self.tracking.face_tracking_enabled,
                lower_body_tracking_enabled: self.tracking.lower_body_tracking_enabled,
                root_translation_enabled: self.tracking.root_translation_enabled,
                frame_dt,
            };
            self.app.run_frame(&frame_config);
            self.runtime_status.frame_count += 1;
        } else {
            // Paused: drain any in-flight render result anyway so
            // `render_results_pending` doesn't pin the repaint gate
            // to full rate while we're idle. `run_frame` would have
            // drained as a side effect, but we skipped it.
            self.app.drain_render_results();
        }

        self.autosave_tick();
        self.write_recovery_snapshot_if_due();

        // Cloth overlay autosave consent dialog
        if self.app.editor.overlay_asset.is_some() && self.cloth_authoring.autosave_consent.is_none() {
        egui::Window::new(t!("dialog.cloth_autosave_title"))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.label(
                    t!("dialog.cloth_autosave_body"),
                );
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if components::filled_button(ui, None, &t!("dialog.yes"), true).clicked() {
                        self.cloth_authoring.autosave_consent = Some(true);
                    }
                    if components::outlined_button(
                        ui,
                        None,
                        &t!("dialog.no"),
                        theme::color::PRIMARY,
                        true,
                    )
                    .clicked()
                    {
                        self.cloth_authoring.autosave_consent = Some(false);
                    }
                });
            });
        }

        top_bar::draw(ctx, self);
        mode_nav::draw(ctx, self);
        status_bar::draw(ctx, self);
        inspector::draw(ctx, self);
        viewport::draw(ctx, self);

        // Pose-calibration modal. Rendered after viewport so the dim
        // overlay covers the entire viewport interior, not just the
        // inspector panel. No-op when the modal is closed.
        calibration::draw_modal(ctx, self);

        // Re-sync the calibration-mode hint *after* the modal has had a
        // chance to transition (Cancel / Esc / Done auto-close all
        // mutate `calibration_modal` from inside `draw_modal`). Without
        // this second pass the worker keeps the last-pushed
        // `Some(UpperBody|FullBody)` until some unrelated repaint fires,
        // which on a static avatar with no autosave can be never — the
        // provider would override the persisted calibration forever.
        self.sync_calibration_mode_hint();

        // Loading-spinner overlay while a background avatar load is in flight.
        if let Some(job) = self.library.avatar_load_job.as_ref() {
            let stage_label = job.current_stage.label();
            let file_label = job
                .path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| job.path.display().to_string());
            egui::Window::new(t!("dialog.loading_avatar"))
                .id(egui::Id::new("avatar_load_modal_window"))
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .collapsible(false)
                .resizable(false)
                // Match the calibration modal: live above any
                // `Order::Middle` scrim and any default-Order Window
                // so the load progress isn't visually buried by either.
                .order(egui::Order::Foreground)
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

        self.draw_toasts(ctx);

        // Repaint gating. The previous unconditional `request_repaint()`
        // pinned egui to monitor refresh (60–144 Hz) even when nothing
        // visible was changing — burning CPU + battery when the user
        // had the app open but idle. Continuous repaint is only needed
        // when frame *content* is actually changing tick to tick or the
        // frame loop has work it can't make progress on without another
        // tick. Egui's own event-driven repaint covers ordinary input
        // (clicks, key presses, mouse moves), so we don't have to.
        //
        // Conditions that require a follow-up tick:
        // - tracking active (live webcam → avatar pose)
        // - lipsync active (volume meter, viseme stream)
        // - avatar load in progress (progress bar)
        // - notification toast still visible (fade animation)
        // - clip animation playing (`active_clip` set on any avatar)
        // - calibration modal open (consumes tracking samples per-frame)
        // - a render result is in flight: `run_frame` submitted a frame
        //   to the render thread but hasn't drained the result yet, so
        //   `process_render_result` won't fire until we tick again. A
        //   slider drag on a static avatar produces input-driven repaint
        //   for the click but the *result* needs at least one more
        //   frame to land — without this the viewport texture stays on
        //   the old image until the user moves the mouse again.
        //
        // Note: `!self.runtime_status.paused` was previously included here, which
        // defeated the gate entirely because `paused` is false by
        // default and the gate then ran continuously from app start.
        let animation_playing = self
            .app
            .avatars
            .iter()
            .any(|a| a.animation_state.active_clip.is_some());
        let render_in_flight = self.app.has_pending_render_result();
        let calibration_modal_open = self.calibration.modal.is_open();
        let needs_animation_frame = self.tracking.toggle_tracking
            || self.app.is_lipsync_enabled()
            || self.library.avatar_load_job.is_some()
            || !self.notifications.is_empty()
            || animation_playing
            || render_in_flight
            || calibration_modal_open;
        if needs_animation_frame || self.project_status.project_dirty || self.project_status.profiles_dirty {
            // Dirty flags imply an unsaved (and almost always also
            // un-rendered) GUI mutation. Repaint immediately so the new
            // state reaches the screen on the next render-thread cycle
            // instead of waiting for the autosave throttle deadline —
            // `request_repaint_after(throttle_remaining)` would defer
            // the next render up to ~250 ms behind the user's click,
            // which is the bug the original gate fix introduced.
            //
            // The autosave throttle still lives in `autosave_tick` and
            // governs disk writes (≤ ~4/s); the worst case here is the
            // ~15 frames between a click and `project_dirty` clearing
            // when the save lands, which is well under any perceptual
            // threshold and well above any battery threshold worth
            // optimising. The narrow pathological case — autosave
            // permanently failing — leaves the gate burning frames at
            // the display rate, but that requires a read-only disk
            // condition the user will already be aware of.
            ctx.request_repaint();
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // Flush a pending dirty save synchronously. The per-frame
        // throttle waits 250ms before writing, so a user who flips a
        // checkbox and immediately Alt+F4s would otherwise lose the
        // change. Falls back to `last_session.vvtproj` when the user
        // never opened a project file.
        if self.project_status.project_dirty {
            let path = self
                .project_status
                .project_path
                .clone()
                .unwrap_or_else(crate::persistence::last_session_path);
            let project_state = self.to_project_state();
            if let Err(e) = crate::persistence::save_project(&project_state, &path) {
                warn!(
                    "persistence: final save on exit failed at {}: {}",
                    path.display(),
                    e
                );
            }
        }

        // No toast on shutdown — the window is already closing and no
        // notification could be displayed. Log the failure so a missing
        // library on the next launch can be traced back to a write failure.
        if let Err(e) = crate::persistence::save_avatar_library(&self.app.avatar_library) {
            warn!("persistence: shutdown save_avatar_library failed: {}", e);
        }
        crate::persistence::RecoveryManager::clear_recovery();
        info!("gui: window closing, initiating application shutdown");
        self.app.shutdown();
    }
}

#[cfg(test)]
mod folder_refresh_tests;
#[cfg(test)]
mod rebind_integration_tests;
#[cfg(test)]
mod thumbnail_failure_tests;
