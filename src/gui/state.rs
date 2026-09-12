//! The per-pane UI state blocks lifted out of `GuiApp` (architecture
//! finding #10): library / scene-preset / cloth-authoring / viewport
//! state, the transform + camera-orbit PODs, the per-domain panel
//! states (tracking / lipsync / rendering / output / settings), and the
//! runtime + project-lifecycle status blocks. Everything public is
//! re-exported at the `crate::gui` root.

use std::path::PathBuf;
use std::time::Instant;

use eframe::egui;

use super::{avatar_load, project};
use crate::tracking::TrackingSmoothingParams;

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
/// finding #10). Bundles the cloth-sim pause toggle, the
/// rename / save-status widget buffers, the per-vertex region
/// selection driven by mesh picking, and the sim-parameter widgets
/// (distance stiffness, bend stiffness, collider toggles, pin node).
pub struct ClothAuthoringUiState {
    /// Session-scoped pause for the cloth simulation (Pause/Play in the
    /// Cloth panel's Preview section, used while authoring to inspect a
    /// static drape or single-step the solver). Deliberately **not**
    /// persisted and `false` by default: the runtime gate is
    /// `rendering.toggle_cloth` (which *is* persisted), so a saved
    /// project resumes simulating on restart without a visit to the
    /// Cloth panel. The previous polarity (`sim_playing`, default
    /// false) silently disabled cloth after every restart.
    pub sim_paused: bool,
    pub rename_buf: String,
    pub save_status: Option<String>,
    pub region_selection: Option<crate::editor::cloth_authoring::RegionSelection>,
    pub material_pick_index: usize,
    pub distance_stiffness: f32,
    pub bend_enabled: bool,
    pub bend_stiffness: f32,
    /// Identity of the overlay the Constraints sliders were last seeded
    /// from. When the active overlay changes, the constraint sliders are
    /// re-initialised from its actual stiffness so the displayed value
    /// matches the asset instead of the GUI default (otherwise an
    /// unwitting "Apply Constraints" would clobber the loaded value).
    pub constraints_seeded_for: Option<crate::asset::ClothOverlayId>,
    pub pin_node_index: usize,
    pub autosave_consent: Option<bool>,
}

impl Default for ClothAuthoringUiState {
    fn default() -> Self {
        Self {
            sim_paused: false,
            rename_buf: String::new(),
            save_status: None,
            region_selection: None,
            material_pick_index: 0,
            // Cloth-sim defaults match the values the inspector
            // previously hard-coded into `GuiApp::new`.
            distance_stiffness: 1.0,
            bend_enabled: true,
            bend_stiffness: 0.5,
            constraints_seeded_for: None,
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
    /// Whether the camera-wipe PIP (live camera preview) is shown.
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
    /// Detection annotation from the last consumed preview publish.
    /// Cached here so the frames *between* mailbox publishes (GUI
    /// refresh > camera fps) can redraw the overlay without pulling a
    /// fresh mailbox snapshot each tick.
    pub camera_wipe_annotation: Option<crate::tracking::DetectionAnnotation>,
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
    pub pending_thumbnail_jobs: Vec<(
        std::path::PathBuf,
        std::sync::mpsc::Receiver<Result<crate::renderer::ThumbnailRenderResult, String>>,
    )>,
    /// Background avatar load in progress, if any. Set by `top_bar::load_*`
    /// helpers, polled and cleared by `update`.
    pub avatar_load_job: Option<avatar_load::AvatarLoadJob>,
}

/// A failure that blocks the app's core function until acknowledged,
/// rendered as a persistent centre modal. Typed (rather than a bare
/// `String`) so the modal can grow per-kind affordances — a Retry
/// button for camera-open failures needs to know *what* to retry —
/// and so producers can't accidentally collapse distinct failures
/// into indistinguishable text.
#[derive(Clone, Debug)]
pub struct BlockingError {
    pub kind: BlockingErrorKind,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockingErrorKind {
    /// Tracking worker reported a fatal error (camera open failure,
    /// model missing, device disconnect past retry budget).
    Tracking,
}

pub struct TransformState {
    pub position: [f32; 3],
    pub rotation: [f32; 3],
    pub scale: f32,
}

// Production defaults for the GUI state blocks. `GuiApp::new` and
// `GuiApp::for_test` both build from these `Default` impls — the two
// constructors used to carry hand-copied literal duplicates, so adding
// a field meant updating both and a drifted default in tests still
// compiled (tests then exercised different defaults than production).
impl Default for TransformState {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0],
            scale: 1.0,
        }
    }
}

pub struct CameraOrbitState {
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub pan: [f32; 2],
    pub distance: f32,
    pub target_distance: f32,
}

impl Default for CameraOrbitState {
    fn default() -> Self {
        Self {
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            pan: [0.0, 0.0],
            distance: 5.0,
            target_distance: 5.0,
        }
    }
}

pub struct TrackingGuiState {
    pub toggle_tracking: bool,
    pub camera_resolution_index: usize,
    pub camera_framerate_index: usize,
    pub tracking_mirror: bool,
    pub hand_tracking_enabled: bool,
    pub face_tracking_enabled: bool,
    /// When true, hip/knee/ankle keypoints drive the leg humanoid
    /// bones; when false the retarget keeps the whole lower-body chain
    /// at rest pose (see `RetargetParams::lower_body_tracking_enabled`).
    pub lower_body_tracking_enabled: bool,
    /// When true, the avatar's `Hips` follows the subject's side-step /
    /// lean / crouch (translation, on top of body-yaw rotation). When
    /// false the avatar pivots in place, keeping the framing stable.
    pub root_translation_enabled: bool,
    /// When true, the avatar fades to transparent after person detection is
    /// lost (past the tracking hold window) and fades back in on re-detection.
    /// Passed through `FrameConfig::fade_on_tracking_loss`.
    pub fade_on_tracking_loss: bool,
    /// Display smoothing / face-confidence thresholds, surfaced in the
    /// Tracking inspector's *Advanced smoothing* section and passed
    /// straight through `FrameConfig::smoothing` each frame. Defaults are
    /// tuned for the common case; see [`TrackingSmoothingParams`]. Only the
    /// blend / confidence fields are user-editable — `stale_timeout_nanos`
    /// keeps its default.
    pub smoothing: TrackingSmoothingParams,
    /// Run every tracking ONNX session on the CPU EP, keeping DirectML
    /// off the GPU entirely. Slower but isolates tracking from GPU
    /// driver instability. Bound at tracking start.
    pub force_cpu_inference: bool,
    /// Run the YOLOX person-crop stage; off = whole-frame RTMW3D.
    /// Bound at tracking start.
    pub yolox_enabled: bool,
    /// Session-only safe-mode latch set from the unclean-exit banner.
    /// While true, tracking starts use the degraded
    /// `TrackingPipelineConfig::safe_mode()` regardless of the saved
    /// toggles above, and the camera is clamped to 720p/30. Never
    /// persisted.
    pub safe_mode_armed: bool,
    /// Sentinel content from a previous session that died uncleanly
    /// (`stagelog::stale_sentinel()`, read once at startup). `Some`
    /// shows the unclean-exit banner; cleared when the user picks
    /// safe mode or dismisses. Never persisted.
    pub unclean_exit_log: Option<String>,
    /// Cameras enumerated for the Tracking panel's device list. Pure UI
    /// artifact (like `LipSyncGuiState::available_mics`) — refreshed on
    /// panel draw / Rescan, never persisted. `None` = never scanned yet.
    pub available_cameras: Option<Vec<crate::tracking::CameraDeviceInfo>>,
    /// Instant of the last automatic rescan, pacing the every-few-seconds
    /// retry while the list is empty (so plugging the camera in flips the
    /// list — and the Start button — without a manual Rescan).
    pub camera_scan_at: Option<std::time::Instant>,
    /// Serial of the D400 the user selected with the device-list radio.
    /// `None` = first enumerated device. Persisted in `settings.json`
    /// (a machine-level choice, not scene state) and read at every
    /// camera start; while the camera runs, changing it stages for the
    /// next start (same policy as the capture-format combos).
    pub camera_serial: Option<String>,
}

impl TrackingGuiState {
    /// Pipeline configuration for the next tracking start: the saved
    /// toggles, unless the safe-mode latch is armed.
    pub fn pipeline_config(&self) -> crate::tracking::provider::TrackingPipelineConfig {
        if self.safe_mode_armed {
            crate::tracking::provider::TrackingPipelineConfig::safe_mode()
        } else {
            crate::tracking::provider::TrackingPipelineConfig {
                force_cpu: self.force_cpu_inference,
                yolox_enabled: self.yolox_enabled,
            }
        }
    }
}

pub struct LipSyncGuiState {
    /// List of mics enumerated for the dropdown. Pure UI artifact —
    /// runtime state (selected mic + enabled flag) lives on Application.
    pub available_mics: Vec<crate::lipsync::AudioDeviceInfo>,
    pub volume_threshold: f32,
    pub smoothing: f32,
    /// Current RMS volume (for the live meter in the UI).
    pub current_volume: f32,
    /// Which signal drives the mouth visemes (audio / camera / both).
    /// Passed through `FrameConfig::mouth_source` to the expression solver.
    pub mouth_source: crate::tracking::MouthSource,
}

impl Default for LipSyncGuiState {
    fn default() -> Self {
        Self {
            available_mics: Vec::new(),
            volume_threshold: 0.01,
            smoothing: 0.5,
            current_volume: 0.0,
            mouth_source: crate::tracking::MouthSource::Both,
        }
    }
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
    pub bloom_enabled: bool,
    pub bloom_intensity: f32,
    pub bloom_threshold: f32,
    /// Generative background parameters, edited in the Preview inspector's
    /// "Scene Background" section and synced to `Application` every frame.
    pub generative_background: crate::renderer::frame_input::GenerativeBackgroundSettings,
    pub toggle_spring: bool,
    /// User spring-bone tuning (sway strength / gravity adjust), edited
    /// next to the spring toggle and passed to the pipeline per frame via
    /// `FrameConfig` (path 2). Persisted with the project.
    pub spring_tuning: crate::simulation::spring::SpringTuning,
    /// Scene-wide gravity (direction + strength) for all physics solvers,
    /// edited in the Rendering inspector, passed via `FrameConfig`
    /// (path 2). Persisted with the project.
    pub scene_gravity: crate::simulation::SceneGravity,
    pub toggle_cloth: bool,
    pub toggle_collision_debug: bool,
    pub toggle_skeleton_debug: bool,
}

impl Default for RenderingGuiState {
    fn default() -> Self {
        Self {
            material_mode_index: 2,
            background_color: [0.1, 0.1, 0.1],
            transparent_background: true,
            camera_fov: 60.0,
            main_light_dir: [0.5, -1.0, 0.3],
            main_light_intensity: 1.0,
            ambient_intensity: [0.2, 0.2, 0.2],
            alpha_preview: false,
            bloom_enabled: false,
            bloom_intensity: 0.6,
            bloom_threshold: 1.0,
            generative_background:
                crate::renderer::frame_input::GenerativeBackgroundSettings::default(),
            toggle_spring: true,
            spring_tuning: crate::simulation::spring::SpringTuning::default(),
            scene_gravity: crate::simulation::SceneGravity::default(),
            toggle_cloth: false,
            toggle_collision_debug: false,
            toggle_skeleton_debug: false,
        }
    }
}

/// Output settings the inspector binds to. The pipeline-bound *sink* lives
/// exclusively on `Application` after Phase C — the inspector reads it via
/// getters (`app.output.active_sink()`) and writes via mutators
/// (`app.ensure_output_sink_runtime()`).
///
/// The index fields below are now fully wired (each is reconciled into
/// `Application` once per frame in `GuiApp::update`) and persisted via
/// `to_project_state`:
/// - `output_resolution_index` → `app.output_extent` → render target extent
///   (`OutputTargetRequest.extent`).
/// - `output_framerate_index` → `app.set_user_render_fps` → the output
///   router's forward-throttle interval.
/// - `output_has_alpha` → `app.output_preserve_alpha` → `OutputFrame.alpha_mode`
///   → VGTK / shared-memory alpha flag.
/// - `output_color_space_index` → `app.output_color_space` → Vulkan colour
///   attachment format (sRGB vs UNORM) + frame metadata.
/// - `msaa_index` → `app.output_msaa` → `OutputTargetRequest.msaa`.
pub struct OutputGuiState {
    pub output_resolution_index: usize,
    pub output_framerate_index: usize,
    pub output_has_alpha: bool,
    pub output_color_space_index: usize,
    /// Anti-aliasing (MSAA) level: 0=Off, 1=2x, 2=4x, 3=8x. Applies to the
    /// shared offscreen render target, so it antialiases both the viewport
    /// preview and the exported frame. The renderer clamps the level to the
    /// device's framebuffer sample-count support (and caps integrated GPUs
    /// at 4x), so an unsupported pick silently degrades.
    pub msaa_index: usize,
}

impl Default for OutputGuiState {
    fn default() -> Self {
        Self {
            output_resolution_index: 0,
            output_framerate_index: 0,
            output_has_alpha: true,
            output_color_space_index: 0,
            msaa_index: 0,
        }
    }
}

pub struct SettingsGuiState {
    pub locale: String,
    pub zoom_sensitivity: f32,
    pub orbit_sensitivity: f32,
    pub pan_sensitivity: f32,
    /// Mirror of [`crate::persistence::AppSettings::last_project_path`]
    /// (`collect_app_settings` rebuilds the DTO from these fields, so
    /// anything that must round-trip through `settings.json` needs a
    /// slot here). Updated by `remember_last_project` on every explicit
    /// project open / save; consumed once at startup for the auto-reopen.
    pub last_project_path: Option<String>,
}

impl Default for SettingsGuiState {
    fn default() -> Self {
        Self {
            locale: "en".to_string(),
            zoom_sensitivity: 0.002,
            orbit_sensitivity: 0.3,
            pan_sensitivity: 1.0,
            last_project_path: None,
        }
    }
}

/// Default for the tracking panel state — also the production default
/// (`GuiApp::new` starts from this and only startup-restore mutates it).
impl Default for TrackingGuiState {
    fn default() -> Self {
        Self {
            toggle_tracking: true,
            camera_resolution_index: 0,
            camera_framerate_index: 0,
            tracking_mirror: true,
            hand_tracking_enabled: false,
            face_tracking_enabled: true,
            lower_body_tracking_enabled: false,
            root_translation_enabled: true,
            fade_on_tracking_loss: false,
            smoothing: TrackingSmoothingParams::default(),
            force_cpu_inference: false,
            yolox_enabled: true,
            safe_mode_armed: false,
            unclean_exit_log: crate::tracking::stagelog::stale_sentinel(),
            available_cameras: None,
            camera_scan_at: None,
            camera_serial: None,
        }
    }
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
    /// Set whenever the in-memory project state diverges from its last
    /// autosaved snapshot. Cleared by the autosave tick once the state
    /// lands in `last_session.vvtproj` / the project's `.unsaved`
    /// sidecar, and by an explicit Save.
    pub project_dirty: bool,
    /// `true` when the explicitly opened `.vvtproj` on disk is behind
    /// the autosaved sidecar — i.e. there are changes only an explicit
    /// Save will persist to the user's real file. Drives the title-bar
    /// dot together with `project_dirty`; cleared by Save / Save As.
    pub explicit_file_stale: bool,
    /// Backoff clocks for the three autosave targets. Base delay is the
    /// 250 ms write throttle; consecutive failures back off to 10 s.
    pub(super) project_save_retry: project::SaveRetry,
    pub(super) profiles_save_retry: project::SaveRetry,
    pub(super) settings_save_retry: project::SaveRetry,
    /// Message text of the sticky failure toast currently shown per
    /// target (if any) so a later success can dismiss exactly it.
    pub(super) last_project_save_error: Option<String>,
    pub(super) last_profiles_save_error: Option<String>,
    pub(super) last_settings_save_error: Option<String>,
    /// One-shot latch for the recovery-write warning toast.
    pub(super) recovery_write_warned: bool,
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
    /// Set whenever an app-level setting (Settings pane: locale,
    /// viewport input sensitivities; cloth-autosave consent dialog)
    /// diverges from `%APPDATA%\VulVATAR\settings.json`. Mirrors
    /// `profiles_dirty` but flushes via `save_app_settings`. App
    /// settings deliberately do NOT ride `project_dirty`: they follow
    /// the user, not the scene.
    pub app_settings_dirty: bool,
    /// The `ProjectState` most recently persisted (autosave target) or
    /// loaded/applied. `GuiApp::refresh_project_dirty` derives
    /// `project_dirty` by comparing the live snapshot against this —
    /// the single mechanism that detects *every* project-visible
    /// change, replacing per-widget `project_dirty = true` calls
    /// (each of which was one forgotten line away from a setting
    /// that silently never saved). `None` only before the first
    /// probe/baseline of the session.
    pub(super) project_baseline: Option<crate::persistence::ProjectState>,
    /// Throttle clock for the dirty probe (compares a <2 KB struct;
    /// probing at the autosave cadence is plenty).
    pub(super) last_dirty_probe: Instant,
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
            explicit_file_stale: false,
            project_save_retry: project::SaveRetry::new(now),
            profiles_save_retry: project::SaveRetry::new(now),
            settings_save_retry: project::SaveRetry::new(now),
            last_project_save_error: None,
            last_profiles_save_error: None,
            last_settings_save_error: None,
            recovery_write_warned: false,
            recovery_manager,
            overlay_dirty: false,
            profiles_dirty: false,
            app_settings_dirty: false,
            project_baseline: None,
            last_dirty_probe: now,
        }
    }
}
