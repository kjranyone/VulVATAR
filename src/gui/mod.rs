pub mod avatar_load;
pub mod components;
mod folder_watcher;
pub mod hotkey;
pub mod inspector;
pub mod mode_nav;
pub mod notifications;
pub mod profile;
mod project;
mod snapshot;
pub mod state;
pub mod status_bar;
pub mod theme;
pub mod top_bar;
pub mod viewport;
pub use state::{
    CameraOrbitState, ClothAuthoringUiState, LibrarySortMode, LibraryUiState, LipSyncGuiState,
    OutputGuiState, ProjectStatusUi, RenderingGuiState, RuntimeStatusUi, ScenePresetUiState,
    SettingsGuiState, TrackingGuiState, TransformState, ViewportUiState, BlockingError,
    BlockingErrorKind,
};



use log::{info, warn};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use eframe::egui;

use crate::app::{Application, RuntimeToggles};
use crate::t;
use crate::tracking::{
    camera_fps_for_index, camera_resolution_for_index, TrackingErrorLevel,
};

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
                fonts
                    .font_data
                    .insert((*name).into(), Arc::new(egui::FontData::from_owned(data)));
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
            fonts.families.entry(family).or_default().push(name.into());
        }
    }

    Some(fonts)
}

use notifications::Notification;

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
    /// Navigable modes. `Preview` is intentionally absent: its cards
    /// were redistributed (avatar info / expressions → Avatar,
    /// transforms / background / gravity → Scene, cloth attachment →
    /// Cloth Authoring) and the mode itself retired. The enum variant
    /// survives only because the hotkey layer still maps
    /// `SwitchModePreview` to it — [`AppMode::normalized`] folds it
    /// into `Rendering` (labelled "Scene") before any frame draws.
    pub const ALL: [AppMode; 6] = [
        AppMode::Avatar,
        AppMode::TrackingSetup,
        AppMode::Rendering,
        AppMode::Output,
        AppMode::ClothAuthoring,
        AppMode::Settings,
    ];

    /// Fold retired modes onto their successor. Call sites that accept
    /// a mode from outside the nav rail (hotkeys, persisted defaults)
    /// run their value through this so `Preview` can never reach the
    /// dispatch table as itself.
    pub fn normalized(self) -> AppMode {
        match self {
            AppMode::Preview => AppMode::Rendering,
            other => other,
        }
    }

    pub fn label(&self) -> String {
        match self {
            AppMode::Avatar => t!("app.modes.avatar"),
            // Retired — normalised to Rendering/Scene before display;
            // keep the arm exhaustive with the successor's label.
            AppMode::Preview => t!("app.modes.scene"),
            AppMode::TrackingSetup => t!("app.modes.tracking_setup"),
            AppMode::Rendering => t!("app.modes.scene"),
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

    /// Latest blocking tracking error, shown as a persistent modal dialog
    /// until the user dismisses it. Unlike a toast it never auto-expires, so
    /// a connection failure the user glanced away from stays on screen and
    /// actionable, stating its root cause + remedy.
    pub blocking_error: Option<BlockingError>,

    /// In-flight OS file-dialog worker (open/save pickers run off the
    /// UI thread so the viewport keeps rendering); drained by
    /// `poll_file_dialog` each frame.
    pub(crate) pending_file_dialog: Option<top_bar::PendingFileDialog>,

    /// A `.vrm` was dropped while another avatar is active — held here
    /// until the user confirms the replacement (drag-and-drop is the
    /// one load entrance where a single stray gesture can nuke the
    /// current avatar, so it alone gets a confirm step).
    pub(crate) pending_avatar_drop: Option<PathBuf>,

    /// Profile New / Rename / Delete dialog state (top-bar combo's
    /// management actions); drawn by `top_bar::draw_profile_dialogs`.
    pub(crate) profile_dialog: Option<top_bar::ProfileDialog>,

    /// Session-only Settings toggle: show the diagnostic status-bar
    /// fields (frame counter, output queue/dropped). Off by default —
    /// they are developer readouts, not streamer-facing signal.
    pub debug_status_bar: bool,

    /// Live measurement of the layout hole between a side panel's
    /// painted frame edge and the cursor it advanced to (egui 0.30
    /// SidePanel quirk; see `inspector::draw`). `[frame_right,
    /// cursor_left, width]` in points; `None` when the layout is
    /// tight. Published to `debug_gui.json` so the "black band" can
    /// be diagnosed from the outside.
    pub debug_panel_hole: Option<[f32; 3]>,

    pub transform: TransformState,
    pub camera_orbit: CameraOrbitState,
    pub tracking: TrackingGuiState,
    pub rendering: RenderingGuiState,
    pub output: OutputGuiState,
    pub settings: SettingsGuiState,

    /// 1:1 sensor-matched mirror render toggle. When on and the live pose is
    /// metric-native (D435 depth path), the render camera adopts the sensor's
    /// intrinsics + a front view so the avatar is framed like a mirror. A
    /// live view control, not persisted; defaults off each session.
    pub mirror_view: bool,

    // Viewport-pane state: rendered-scene texture handle, the
    // Blender-style drag-grab state, and the camera-wipe PIP toggle +
    // its companion texture / scratch buffer. See `ViewportUiState`.
    pub viewport: ViewportUiState,

    // Lip sync
    pub lipsync: LipSyncGuiState,

    // Cloth-authoring panel state (sim playback, rename / save
    // buffers, region selection, sim-parameter widgets). See
    // `ClothAuthoringUiState`.
    pub cloth_authoring: ClothAuthoringUiState,

    pub inspector_open: bool,

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

        // App-level settings (locale, input sensitivities, consents)
        // load FIRST — before fonts, because the locale drives the CJK
        // fallback order, and before any project/session restore,
        // because these are user preferences that no scene file may
        // override. Installs that predate `settings.json` migrate the
        // values out of the legacy last-session slot exactly once.
        let app_settings = crate::persistence::load_app_settings().unwrap_or_else(|| {
            let migrated = crate::persistence::migrate_legacy_app_settings().unwrap_or_default();
            if let Err(e) = crate::persistence::save_app_settings(&migrated) {
                warn!("persistence: could not write initial settings.json: {e}");
            }
            migrated
        });
        crate::i18n::set_locale(&app_settings.locale);
        // Publish the persisted cloth backend preference before any
        // avatar (and therefore any cloth attach) can happen.
        crate::simulation::cloth_gpu_boundary::set_cloth_backend_request(app_settings.cloth_gpu_backend);

        if let Some(fonts) = build_font_definitions(&crate::i18n::locale()) {
            cc.egui_ctx.set_fonts(fonts);
        }
        theme::apply(&cc.egui_ctx);
        let mut app = Box::new(Application::new());
        // Auto-cloth opt-out — published before bootstrap so the
        // default avatar's attach (inside bootstrap) already sees it.
        app.set_auto_cloth_enabled(app_settings.auto_cloth.unwrap_or(true));
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
            mode: AppMode::Avatar,
            app,

            runtime_status: RuntimeStatusUi::new(Instant::now()),

            project_status: ProjectStatusUi::new(
                crate::persistence::load_recent_avatars(),
                crate::persistence::RecoveryManager::new(120),
                Instant::now(),
            ),

            hotkeys: hotkey::HotkeyMap::new(),
            // Profiles persist across project loads (which is
            // per-room/setup, not per-scene), so a fresh-install
            // fallback to the built-in presets is fine but a parse
            // failure of an existing file should not silently wipe
            // out the user's data (load_profiles already logs and
            // returns None then).
            profiles: crate::persistence::load_profiles().unwrap_or_default(),

            notifications: Vec::new(),
            blocking_error: None,
            pending_file_dialog: None,
            pending_avatar_drop: None,
            profile_dialog: None,
            debug_status_bar: false,
            debug_panel_hole: None,

            transform: TransformState::default(),
            camera_orbit: CameraOrbitState::default(),
            tracking: TrackingGuiState {
                // Device pick rides settings.json (machine-level).
                camera_serial: app_settings.camera_serial.clone(),
                ..TrackingGuiState::default()
            },
            rendering: RenderingGuiState::default(),
            output: OutputGuiState::default(),

            settings: SettingsGuiState {
                locale: app_settings.locale.clone(),
                zoom_sensitivity: app_settings.zoom_sensitivity,
                orbit_sensitivity: app_settings.orbit_sensitivity,
                pan_sensitivity: app_settings.pan_sensitivity,
                last_project_path: app_settings.last_project_path.clone(),
                cloth_gpu_backend: app_settings.cloth_gpu_backend,
                auto_cloth: app_settings.auto_cloth,
            },

            mirror_view: false,
            viewport: ViewportUiState {
                show_detection_annotations: true,
                ..ViewportUiState::default()
            },

            lipsync: LipSyncGuiState {
                available_mics: crate::lipsync::audio_capture::list_audio_devices(),
                ..LipSyncGuiState::default()
            },

            cloth_authoring: ClothAuthoringUiState {
                // Consent is a user-level preference; it rides
                // settings.json, not the project file.
                autosave_consent: app_settings.cloth_autosave_consent,
                ..ClothAuthoringUiState::default()
            },
            inspector_open: true,

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

        // ── Startup restore ─────────────────────────────────────────
        // Source priority for the restored state:
        //   1. the last *explicitly* opened / saved project
        //      (`settings.json::last_project_path`) — reopened with its
        //      real path in the title bar;
        //   2. the implicit `last_session.vvtproj` slot — written by the
        //      per-frame autosave whenever `project_dirty` is set and
        //      `project_path` is `None`, so a user who never ran File >
        //      Save still gets their checkboxes / sliders / avatar back.
        //      `project_path` stays unset for this source so no phantom
        //      file shows in the title bar.
        // The restored state's avatar (and the VULVATAR_AUTOSTART_AVATAR
        // env override, which outranks it) loads asynchronously; the
        // state itself is applied *after* the avatar lands (AfterLoad::
        // ApplyProject) so `finalize_avatar_load`'s camera auto-frame
        // doesn't clobber the restored orbit.
        let mut restored: Option<(
            crate::persistence::ProjectState,
            crate::persistence::ProjectLoadWarnings,
            Option<PathBuf>,
        )> = None;
        // `true` when the restored state came from the project's
        // `.unsaved` sidecar (last session quit without an explicit
        // Save) — apply marks the explicit file stale + toasts.
        let mut restore_unsaved = false;
        if let Some(last_project) = state.settings.last_project_path.clone() {
            let last_project_path = PathBuf::from(&last_project);
            if last_project_path.exists() {
                // Unsaved changes from the previous session live in the
                // autosaved sidecar; prefer it when it's newer so a quit
                // without Save doesn't silently roll the scene back.
                let sidecar = project::unsaved_sidecar_path(&last_project_path);
                let load_path = if project::sidecar_is_newer(&last_project_path, &sidecar) {
                    restore_unsaved = true;
                    sidecar
                } else {
                    last_project_path.clone()
                };
                match crate::persistence::load_project(&load_path) {
                    Ok((ps, warnings)) => {
                        info!(
                            "persistence: reopening last project {} (unsaved sidecar: {})",
                            last_project_path.display(),
                            restore_unsaved
                        );
                        restored = Some((ps, warnings, Some(last_project_path)));
                    }
                    Err(e) => {
                        restore_unsaved = false;
                        warn!(
                            "persistence: could not reopen last project {}: {}",
                            last_project_path.display(),
                            e
                        );
                        state.push_error_notification(t!(
                            "top_bar.failed_load_project",
                            error = e.to_string()
                        ));
                    }
                }
            } else {
                warn!(
                    "persistence: last project no longer exists: {}",
                    last_project_path.display()
                );
                state.push_warning_notification(t!(
                    "toast.last_project_missing",
                    path = last_project.clone()
                ));
                // Clear so a deleted project doesn't warn on every launch.
                state.settings.last_project_path = None;
                state.project_status.app_settings_dirty = true;
            }
        }
        let last_session = crate::persistence::last_session_path();
        if restored.is_none() && last_session.exists() {
            match crate::persistence::load_project(&last_session) {
                Ok((ps, warnings)) => {
                    info!(
                        "persistence: restoring last session from {}",
                        last_session.display()
                    );
                    restored = Some((ps, warnings, None));
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

        let env_avatar = std::env::var_os("VULVATAR_AUTOSTART_AVATAR").map(PathBuf::from);
        let (startup_avatar, avatar_issues) = project::resolve_startup_avatar(
            env_avatar,
            restored
                .as_ref()
                .and_then(|(ps, _, _)| ps.avatar_source_path.as_deref()),
            &|p: &std::path::Path| p.exists(),
        );
        for issue in avatar_issues {
            match issue {
                project::StartupAvatarIssue::EnvPathMissing(p) => {
                    warn!("avatar autostart path does not exist: {}", p.display());
                }
                project::StartupAvatarIssue::ProjectAvatarMissing(p) => {
                    warn!("persistence: restored avatar no longer exists: {p}");
                    state.push_warning_notification(t!("top_bar.avatar_not_found", path = p));
                }
            }
        }
        let startup_avatar_path = match startup_avatar {
            project::StartupAvatar::Env(p) => {
                info!("avatar autostart loading {}", p.display());
                Some(p)
            }
            project::StartupAvatar::Project(p) => Some(p),
            project::StartupAvatar::None => None,
        };
        match (startup_avatar_path, restored) {
            (Some(avatar), Some((ps, warnings, project_path))) => {
                state.library.avatar_load_job =
                    Some(crate::gui::avatar_load::AvatarLoadJob::spawn(
                        avatar,
                        crate::gui::avatar_load::AfterLoad::ApplyProject {
                            project_state: Box::new(ps),
                            project_path,
                            warnings,
                            restore_unsaved,
                        },
                    ));
            }
            (Some(avatar), None) => {
                top_bar::load_avatar_from_path(&mut state, &avatar);
            }
            (None, Some((ps, warnings, project_path))) => {
                state.apply_project_state(&ps);
                state.mark_project_baseline();
                state.project_status.explicit_file_stale =
                    restore_unsaved && project_path.is_some();
                for w in &warnings.warnings {
                    state.push_warning_notification(t!("toast.warning", msg = w.to_string()));
                }
                if restore_unsaved {
                    state.push_notification(t!("toast.restored_unsaved_changes"));
                }
                if let Some(path) = project_path {
                    state.push_success_notification(t!(
                        "toast.opened_project",
                        path = path.display().to_string()
                    ));
                    state.project_status.project_path = Some(path);
                }
            }
            (None, None) => {}
        }

        // Environment-driven tracking autostart (automation / headless
        // validation of the live pipeline): starts the camera with the
        // restored panel parameters, exactly like the top-bar toggle.
        if std::env::var_os("VULVATAR_AUTOSTART_TRACKING").is_some() {
            info!("tracking autostart enabled");
            state.tracking.toggle_tracking = true;
            state.start_camera_with_current_params();
        }

        // Backfill placeholder thumbnails for library entries that have
        // none on disk (legacy library data + entries whose VRM lacked an
        // embedded cover image). Cheap — a 128×128 procedural draw per
        // entry. Only writes new files.
        let mut library_dirty = false;
        for entry in state.app.avatar_library.entries.iter_mut() {
            let needs = entry.thumbnail_path.as_ref().is_none_or(|p| !p.exists());
            if needs {
                if let Some(path) = state
                    .library
                    .thumbnail_gen
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
                state.push_notification(t!(
                    "toast.watched_folder_missing",
                    path = path.display().to_string()
                ));
                continue;
            }
            if let Err(e) = state.start_watching_folder(path.clone()) {
                state.push_notification(t!(
                    "toast.failed_restore_watch",
                    path = path.display().to_string(),
                    error = e.to_string()
                ));
            }
        }
        let _ = crate::persistence::save_watched_folders(&state.library.watched_avatar_dirs);

        state
    }

    /// Build a minimal `GuiApp` for unit/integration tests. Identical to
    /// [`GuiApp::new`] in shape, but skips:
    ///
    /// * `eframe::CreationContext` (no egui context available in tests)
    /// * disk reads (avatar library, scene presets, recent avatars,
    ///   recovery snapshot, watched folders) — start with empty state
    /// * system probes (`list_audio_devices`) — empty vecs, since the
    ///   test doesn't drive the GUI
    /// * environment-driven autostart (autoload avatar, virtual-camera)
    ///
    /// The test can then populate `app.avatars` directly and exercise
    /// methods on this harness without spinning up the full GUI.
    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        let app = Box::new(Application::new());
        Self {
            mode: AppMode::Avatar,
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
            blocking_error: None,
            pending_file_dialog: None,
            pending_avatar_drop: None,
            profile_dialog: None,
            debug_status_bar: false,
            debug_panel_hole: None,

            transform: TransformState::default(),
            camera_orbit: CameraOrbitState::default(),
            tracking: TrackingGuiState::default(),
            rendering: RenderingGuiState::default(),
            output: OutputGuiState::default(),

            settings: SettingsGuiState::default(),

            mirror_view: false,
            viewport: ViewportUiState {
                show_detection_annotations: true,
                ..ViewportUiState::default()
            },

            lipsync: LipSyncGuiState::default(),

            cloth_authoring: ClothAuthoringUiState::default(),
            inspector_open: true,

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

    /// Whether Start Camera can do anything right now: the last
    /// enumeration found a D400-series camera on a fast-enough USB link.
    /// While the camera is running this is moot (stop governs), and an
    /// exhausted scan reads as false so the button still works when the
    /// backend is a non-realsense build.
    pub fn camera_startable(&self) -> bool {
        self.tracking
            .available_cameras
            .as_ref()
            .is_some_and(|cams| crate::tracking::usable_capture_device(cams))
    }

    /// (Re-)enumerate connected RealSense cameras into
    /// [`TrackingGuiState::available_cameras`], pacing automatic rescans
    /// with `camera_scan_at`. `force` is the explicit Rescan button;
    /// otherwise an empty/never-scanned list rescans at most every
    /// `interval` so plugging the camera in self-heals without spamming
    /// librealsense (a scan loads/queries the context) — and never
    /// rescans while the camera is streaming, where it would contend
    /// with the capture thread. Returns the error toast text on a
    /// failed enumeration (context/driver trouble the user must see).
    pub fn rescan_cameras(&mut self, force: bool, interval: std::time::Duration) -> Option<String> {
        if self.is_tracking_active() {
            return None;
        }
        let due = self
            .tracking
            .camera_scan_at
            .is_none_or(|at| at.elapsed() >= interval);
        let list_empty = self
            .tracking
            .available_cameras
            .as_ref()
            .is_none_or(|cams| cams.is_empty());
        if !force && !due {
            return None;
        }
        // Only an empty (or never-scanned) list auto-retries; a
        // populated list stays put until an explicit Rescan so a camera
        // unplugged mid-session doesn't get silently re-added while the
        // user reads it.
        if !force && !list_empty {
            return None;
        }
        self.tracking.camera_scan_at = Some(std::time::Instant::now());
        match crate::tracking::enumerate_cameras() {
            Ok(cams) => {
                self.tracking.available_cameras = Some(cams);
                None
            }
            Err(e) => {
                // Keep whatever the last good scan produced; the error
                // is the actionable signal.
                Some(t!("tracking.camera_scan_failed", error = e))
            }
        }
    }

    /// Start the camera with the panel-configured resolution / fps /
    /// pipeline, applying the safe-mode clamp. Single entry point for
    /// the top-bar toggle; the Tracking panel's Start button and its
    /// restart-on-format-change block should migrate onto this helper
    /// too (left in place for now — the panel file is being reworked
    /// in parallel; see the Wave-2 shell-redesign handoff note).
    pub(crate) fn start_camera_with_current_params(&mut self) {
        let (w, h) = camera_resolution_for_index(self.tracking.camera_resolution_index);
        let fps = camera_fps_for_index(self.tracking.camera_framerate_index);
        // Safe mode also caps the capture format — less camera
        // bandwidth and per-frame CPU work while diagnosing an
        // unclean exit.
        let (w, h, fps) = if self.tracking.safe_mode_armed {
            (w.min(1280), h.min(720), fps.min(30))
        } else {
            (w, h, fps)
        };
        let pipeline = self.tracking.pipeline_config();
        let camera_serial = self.tracking.camera_serial.clone();
        self.app
            .start_tracking_with_params(w, h, fps, camera_serial, pipeline);
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
        if let Err(e) = crate::persistence::save_recent_avatars(&self.project_status.recent_avatars)
        {
            warn!("persistence: save_recent_avatars failed: {}", e);
        }
    }

    /// Build a `RuntimeToggles` from the current GUI state.
    fn runtime_toggles(&self) -> RuntimeToggles {
        RuntimeToggles {
            tracking_enabled: self.tracking.toggle_tracking,
            spring_enabled: self.rendering.toggle_spring,
            // The persisted Rendering-panel toggle is the gate; the Cloth
            // panel's transport only *pauses* an otherwise-enabled sim.
            // (The per-avatar `avatar.cloth_enabled` leg of the gate is
            // applied in `Application::run_frame`.)
            cloth_enabled: self.rendering.toggle_cloth && !self.cloth_authoring.sim_paused,
            collision_debug: self.rendering.toggle_collision_debug,
            skeleton_debug: self.rendering.toggle_skeleton_debug,
            mirror_view: self.mirror_view,
        }
    }

    // ---------------------------------------------------------------------
    // GUI → pipeline wiring. Every setting reaches the pipeline through
    // exactly one of four sanctioned paths (see docs/architecture.md,
    // "GUI → pipeline settings wiring"):
    //
    //   1. Reconciled — continuous scene/view/output parameters owned by
    //      the GUI and pushed idempotently into `Application` fields once
    //      per frame in [`Self::sync_app_settings`].
    //   2. FrameConfig — per-frame pipeline inputs (toggles, smoothing,
    //      tracking flags, mouth source) built in
    //      [`Self::build_frame_config`] and consumed inside `run_frame`.
    //   3. Requested (Phase D) — runtime resources that can fail to start
    //      (output sink, lipsync). Written via
    //      `Application::set_requested_*`; the GUI reads back the
    //      requested/active pair instead of holding its own copy.
    //   4. Instance-bound — per-avatar runtime state (expression weights,
    //      collider mask, cloth attachments) bound directly to
    //      `AvatarInstance`; resets with the avatar by design.
    //
    // When adding a new setting, pick the matching path — don't invent a
    // fifth.
    // ---------------------------------------------------------------------

    /// Path 1: reconcile GUI-owned scene / view / output parameters into
    /// their `Application` mirrors. Called exactly once per `update`,
    /// before `run_frame`, so the render thread always sees this frame's
    /// values. Every write here is idempotent — the GUI fields are the
    /// source of truth and `Application` holds per-frame mirrors.
    fn sync_app_settings(&mut self) {
        // Avatar world transform from the Preview panel's gizmo fields.
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

        // Viewport camera from the orbit controls + Rendering-panel FOV.
        self.app.viewport_camera.yaw_deg = self.camera_orbit.yaw_deg;
        self.app.viewport_camera.pitch_deg = self.camera_orbit.pitch_deg;
        self.app.viewport_camera.distance = self.camera_orbit.distance;
        self.app.viewport_camera.pan = self.camera_orbit.pan;
        self.app.viewport_camera.fov_deg = self.rendering.camera_fov;

        // Lighting parameters from the Rendering inspector.
        self.app.viewport_lighting.main_light_dir_ws = self.rendering.main_light_dir;
        self.app.viewport_lighting.main_light_intensity = self.rendering.main_light_intensity;
        self.app.viewport_lighting.ambient_term = self.rendering.ambient_intensity;

        // Scene Background → the Vulkan renderer's clear color. Until
        // T11 found this gap the inspector toggle was a pure egui-side hint
        // and the renderer always cleared to (0,0,0,0), which made the MF
        // virtual camera output appear all-black to clients that don't
        // honour the alpha channel (Meet / Zoom).
        self.app.background_color = self.rendering.background_color;
        self.app.transparent_background = self.rendering.transparent_background;

        // Output resolution preference. The renderer reads this through
        // OutputTargetRequest.extent, so changing the combo immediately
        // resizes the next render's output target.
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

        // Phase B-4: alpha preference. Read each frame in
        // process_render_result to tag OutputFrame.alpha_mode.
        self.app.output_preserve_alpha = self.output.output_has_alpha;

        // Colour space pick → the renderer's OutputTargetRequest. Stage 1
        // plumbing: the value reaches ExportMetadata + OutputFrame
        // .color_space; format / shader / MF media type changes land in
        // later stages.
        self.app.output_color_space = match self.output.output_color_space_index {
            1 => crate::renderer::frame_input::RenderColorSpace::LinearSrgb,
            _ => crate::renderer::frame_input::RenderColorSpace::Srgb,
        };

        // Anti-aliasing pick. The renderer rebuilds its render pass +
        // pipelines on change and clamps the level to device support.
        self.app.output_msaa =
            crate::renderer::frame_input::MsaaMode::from_index(self.output.msaa_index);

        // Bloom post-effect parameters. Push-constant driven on the
        // renderer side — no pipeline rebuild on change.
        self.app.bloom = crate::renderer::frame_input::BloomSettings {
            enabled: self.rendering.bloom_enabled,
            intensity: self.rendering.bloom_intensity,
            threshold: self.rendering.bloom_threshold,
        };

        // Generative background parameters. Push-constant driven on the
        // renderer side — no pipeline rebuild on change.
        self.app.generative_background = self.rendering.generative_background;
    }

    /// Path 2: bundle the per-frame pipeline inputs for `run_frame`.
    /// These are read inside the frame step (pose solver, simulation,
    /// material mode) and never need to outlive it, so they ride a value
    /// struct instead of `Application` fields.
    fn build_frame_config(&self, frame_dt: f32) -> crate::app::FrameConfig {
        crate::app::FrameConfig {
            toggles: self.runtime_toggles(),
            smoothing: self.tracking.smoothing.clone(),
            material_mode_index: self.rendering.material_mode_index,
            hand_tracking_enabled: self.tracking.hand_tracking_enabled,
            face_tracking_enabled: self.tracking.face_tracking_enabled,
            lower_body_tracking_enabled: self.tracking.lower_body_tracking_enabled,
            root_translation_enabled: self.tracking.root_translation_enabled,
            fade_on_tracking_loss: self.tracking.fade_on_tracking_loss,
            mouth_source: self.lipsync.mouth_source,
            spring_tuning: self.rendering.spring_tuning,
            scene_gravity: self.rendering.scene_gravity,
            frame_dt,
        }
    }
}

/// Localised label for an avatar-load progress stage. The asset-layer
/// `LoadStage::label()` is English-only; the GUI maps the enum through
/// the locale table so the loading spinner speaks the UI language.
fn load_stage_label(stage: &crate::asset::vrm::LoadStage) -> String {
    use crate::asset::vrm::LoadStage;
    match stage {
        LoadStage::Reading => t!("load_stage.reading"),
        LoadStage::Parsing => t!("load_stage.parsing"),
        LoadStage::Skeleton => t!("load_stage.skeleton"),
        LoadStage::Meshes => t!("load_stage.meshes"),
        LoadStage::Materials { current, total } => {
            t!("load_stage.materials", current = current, total = total)
        }
        LoadStage::SpringBones => t!("load_stage.spring_bones"),
        LoadStage::Finalizing => t!("load_stage.finalizing"),
    }
}

impl eframe::App for GuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle dropped files. Every supported project artefact is
        // accepted, and an unsupported extension gets a toast — a drop
        // that silently does nothing is indistinguishable from a hang.
        let dropped: Vec<_> = ctx.input(|i| i.raw.dropped_files.clone());
        for file in dropped {
            if let Some(path) = file.path {
                let resolved_path = if path.is_dir() {
                    crate::asset::find_avatar_file_in_dir(&path).unwrap_or(path.clone())
                } else {
                    path.clone()
                };

                let ext = resolved_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                match ext.as_str() {
                    "vrm" | "fbx" => {
                        info!(
                            "gui: dropped avatar file (original: {:?}): {:?}",
                            path, resolved_path
                        );
                        // Replacing a live avatar from a stray drag is
                        // the one accidental-destruction path — ask
                        // first. First load (no avatar yet) and a
                        // same-file reload go straight through.
                        let replaces_other = self
                            .app
                            .active_avatar()
                            .is_some_and(|a| a.asset.source_path != resolved_path);
                        if replaces_other {
                            self.pending_avatar_drop = Some(resolved_path);
                        } else {
                            top_bar::load_avatar_from_path(self, &resolved_path);
                        }
                    }
                    "vvtproj" => {
                        info!("gui: dropped project file: {:?}", resolved_path);
                        top_bar::open_project_from_path(self, &resolved_path);
                    }
                    "vvtcloth" => {
                        info!("gui: dropped cloth overlay: {:?}", resolved_path);
                        top_bar::open_overlay_from_path(self, &resolved_path);
                    }
                    _ => {
                        self.push_warning_notification(t!(
                            "toast.unsupported_drop",
                            path = path.display().to_string()
                        ));
                    }
                }
            }
        }

        // Confirm dialog for a dropped VRM that would replace the
        // active avatar.
        if let Some(pending) = self.pending_avatar_drop.clone() {
            let mut decision: Option<bool> = None;
            let file_label = pending
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| pending.display().to_string());
            egui::Window::new(t!("dialog.replace_avatar_title"))
                .id(egui::Id::new("avatar_replace_confirm"))
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .collapsible(false)
                .resizable(false)
                .order(egui::Order::Foreground)
                .show(ctx, |ui| {
                    ui.label(t!("dialog.replace_avatar_body", file = file_label));
                    ui.add_space(theme::space::SM);
                    ui.horizontal(|ui| {
                        if ui.button(t!("dialog.replace_avatar_confirm")).clicked() {
                            decision = Some(true);
                        }
                        if ui.button(t!("dialog.cancel")).clicked() {
                            decision = Some(false);
                        }
                    });
                });
            match decision {
                Some(true) => {
                    self.pending_avatar_drop = None;
                    top_bar::load_avatar_from_path(self, &pending);
                }
                Some(false) => {
                    self.pending_avatar_drop = None;
                }
                None => {}
            }
        }

        // Profile management dialogs (New / Rename / Delete) from the
        // top-bar combo.
        top_bar::draw_profile_dialogs(ctx, self);

        // Drain any finished OS file-dialog worker (open/save pickers
        // run off the UI thread — see `top_bar::request_file_dialog`).
        self.poll_file_dialog();

        // E10: Measure frame timing.
        let now = Instant::now();
        let elapsed = now.duration_since(self.runtime_status.last_frame_instant);
        self.runtime_status.last_frame_instant = now;
        let dt_secs = elapsed.as_secs_f64();
        // Exponential moving average for smoothing.
        self.runtime_status.frame_time_ms =
            self.runtime_status.frame_time_ms * 0.9 + (dt_secs * 1000.0) * 0.1;
        if dt_secs > 0.0 {
            self.runtime_status.fps = self.runtime_status.fps * 0.9 + (1.0 / dt_secs) * 0.1;
        }

        self.process_hotkeys(ctx);

        if let Some((err, level)) = self.app.tracking.mailbox().drain_error() {
            match level {
                // Blocking errors get a persistent modal (drawn below) so a
                // camera/connection failure can't be missed; warnings stay as
                // auto-expiring toasts.
                TrackingErrorLevel::Blocking => {
                    self.blocking_error = Some(BlockingError {
                        kind: BlockingErrorKind::Tracking,
                        message: err,
                    })
                }
                TrackingErrorLevel::Warning => self.push_warning_notification(err),
            }
        }

        self.poll_folder_watcher();
        self.poll_thumbnail_jobs(ctx);
        self.poll_avatar_load_job();

        // Path 1: reconcile GUI-owned settings into Application before
        // running the frame. See the wiring overview above
        // `sync_app_settings`.
        self.sync_app_settings();

        // Heartbeat BEFORE the pause gate: records the raw flags that
        // decide whether anything below runs. Its `seq` advances on every
        // GUI frame, so a stalled avatar can be attributed to the exact
        // switch responsible instead of inferred from which files stopped
        // being written. No-op unless the debug flag file exists.
        crate::tracking::debug_channel::dump_gui_heartbeat(
            self.runtime_status.paused,
            self.app.avatars.len(),
            self.tracking.toggle_tracking,
            self.runtime_status.frame_count,
            self.app.last_sim_substeps,
            self.debug_panel_hole,
            self.app.render_thread_fps(),
            self.app.render_submit_drops_total(),
            self.app.render_thread_cpu_ms(),
        );

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

            // Path 2: per-frame pipeline inputs ride the FrameConfig.
            let frame_config = self.build_frame_config(frame_dt);
            self.app.run_frame(&frame_config);
            self.runtime_status.frame_count += 1;
        } else {
            // Paused: drain any in-flight render result anyway so
            // `render_results_pending` doesn't pin the repaint gate
            // to full rate while we're idle. `run_frame` would have
            // drained as a side effect, but we skipped it.
            self.app.drain_render_results();
        }

        // Derive `project_dirty` from an actual state comparison before
        // the autosave reads it — see `refresh_project_dirty`.
        self.refresh_project_dirty(Instant::now(), false);
        self.autosave_tick();
        self.write_recovery_snapshot_if_due();

        // Cloth overlay autosave consent dialog
        if self.app.editor.overlay_asset.is_some()
            && self.cloth_authoring.autosave_consent.is_none()
        {
            egui::Window::new(t!("dialog.cloth_autosave_title"))
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    ui.label(t!("dialog.cloth_autosave_body"));
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if components::filled_button(ui, None, &t!("dialog.yes"), true).clicked() {
                            self.cloth_authoring.autosave_consent = Some(true);
                            self.project_status.app_settings_dirty = true;
                        }
                        if components::tonal_button(
                            ui,
                            None,
                            &t!("dialog.no"),
                            components::ButtonTone::Primary,
                            true,
                        )
                        .clicked()
                        {
                            self.cloth_authoring.autosave_consent = Some(false);
                            self.project_status.app_settings_dirty = true;
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
        if let Some(job) = self.library.avatar_load_job.as_ref() {
            let stage_label = load_stage_label(&job.current_stage);
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
                // Live above any `Order::Middle` scrim and any
                // default-Order Window so the load progress isn't
                // visually buried by either.
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

        // Blocking tracking error — persistent modal until dismissed. Drawn
        // above the viewport and other windows (Foreground) so a
        // camera/connection failure is impossible to miss and states its root
        // cause + remedy, rather than flashing past in a 15 s toast.
        let mut dismiss_blocking_error = false;
        if let Some(err) = self.blocking_error.as_ref() {
            egui::Window::new(t!("dialog.tracking_error_title"))
                .id(egui::Id::new("tracking_blocking_error_modal"))
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .collapsible(false)
                .resizable(false)
                .order(egui::Order::Foreground)
                .show(ctx, |ui| {
                    ui.set_max_width(440.0);
                    ui.label(egui::RichText::new(err.message.as_str()));
                    ui.add_space(12.0);
                    if components::filled_button(
                        ui,
                        None,
                        &t!("dialog.tracking_error_dismiss"),
                        true,
                    )
                    .clicked()
                    {
                        dismiss_blocking_error = true;
                    }
                });
        }
        if dismiss_blocking_error {
            self.blocking_error = None;
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
        // - tracking active (live camera → avatar pose)
        // - lipsync active (volume meter, viseme stream)
        // - avatar load in progress (progress bar)
        // - notification toast still visible (fade animation)
        // - clip animation playing (`active_clip` set on any avatar)
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
        let needs_animation_frame = self.tracking.toggle_tracking
            || self.app.is_lipsync_enabled()
            || self.library.avatar_load_job.is_some()
            || !self.notifications.is_empty()
            || animation_playing
            || render_in_flight;
        if needs_animation_frame
            || self.project_status.project_dirty
            || self.project_status.profiles_dirty
            || self.project_status.app_settings_dirty
        {
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
        // Final dirty probe, bypassing the 250 ms throttle: a change
        // made in the last quarter-second before quit must still be
        // detected before the flush below reads the flag.
        self.refresh_project_dirty(Instant::now(), true);
        // Flush a pending dirty save synchronously. The per-frame
        // throttle waits 250ms before writing, so a user who flips a
        // checkbox and immediately Alt+F4s would otherwise lose the
        // change. Like the autosave tick, this never touches the
        // explicit `.vvtproj` — unsaved changes go to the project's
        // `.unsaved` sidecar (restored + marked dirty next launch) or
        // to `last_session.vvtproj` when no project is open.
        if self.project_status.project_dirty {
            let path = match self.project_status.project_path.clone() {
                Some(project) => project::unsaved_sidecar_path(&project),
                None => crate::persistence::last_session_path(),
            };
            let project_state = self.to_project_state();
            if let Err(e) = crate::persistence::save_project(&project_state, &path) {
                warn!(
                    "persistence: final save on exit failed at {}: {}",
                    path.display(),
                    e
                );
            }
        }

        // Flush the other two dirty stores as well — settings carry
        // locale/sensitivities; both used to be silently dropped on a
        // quick quit.
        if self.project_status.profiles_dirty {
            if let Err(e) = crate::persistence::save_profiles(&self.profiles) {
                warn!("persistence: final save_profiles on exit failed: {}", e);
            }
        }
        if self.project_status.app_settings_dirty {
            if let Err(e) = crate::persistence::save_app_settings(&self.collect_app_settings()) {
                warn!("persistence: final save_app_settings on exit failed: {}", e);
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
