pub mod avatar_library;
pub mod bake_cache;
pub mod folder_watcher;
mod lipsync;
mod output_sink;
mod render;
pub mod render_thread;
pub mod runtime_gpu_budget;
mod tracking_lifecycle;

use log::{error, info, warn};
use std::collections::VecDeque;
use std::sync::Arc;

use crate::app::render_thread::RenderThread;
use crate::asset::vrm::VrmAssetLoader;
use crate::avatar::{AvatarInstance, AvatarInstanceId};
use crate::editor::EditorSession;
use crate::output::{FrameSink, OutputRouter};
use crate::renderer::frame_input::LightingState;
use crate::renderer::VulkanRenderer;
use crate::simulation::{PhysicsWorld, SimulationClock};
use crate::tracking::{TrackingSmoothingParams, TrackingSource, TrackingWorker};

/// All per-frame parameters passed from the GUI to `run_frame()`.
#[derive(Clone, Debug)]
pub struct FrameConfig {
    pub toggles: RuntimeToggles,
    pub smoothing: TrackingSmoothingParams,
    pub material_mode_index: usize,
    pub hand_tracking_enabled: bool,
    pub face_tracking_enabled: bool,
    pub lower_body_tracking_enabled: bool,
    pub root_translation_enabled: bool,
    /// When true, the avatar fades to transparent after person detection is
    /// lost past the tracking hold window, and fades back in on re-detection.
    pub fade_on_tracking_loss: bool,
    /// Which signal drives the mouth visemes (audio lip-sync / camera / both).
    pub mouth_source: crate::tracking::MouthSource,
    /// User spring-bone tuning from the Rendering inspector, layered on
    /// top of the VRM asset's authored values at simulation time.
    pub spring_tuning: crate::simulation::spring::SpringTuning,
    /// Scene-wide gravity (direction + strength) shared by spring / cloth
    /// / Rapier solvers.
    pub scene_gravity: crate::simulation::SceneGravity,
    pub frame_dt: f32,
}

/// Configuration for building the renderer-facing frame snapshot.
#[derive(Clone, Debug)]
pub struct FrameInputConfig {
    pub camera: ViewportCamera,
    pub lighting: LightingState,
    pub viewport_extent: [u32; 2],
    pub output_extent: [u32; 2],
    /// RGB clear color for the render pass when `transparent_background` is
    /// false. Synced from the GUI inspector. Ignored if transparent.
    pub background_color: [f32; 3],
    /// When true, the render target is cleared to `(0,0,0,0)` and only the
    /// avatar pixels are non-zero. When false, the render target is cleared
    /// to `(background_color, 1)` so the output is opaque.
    pub transparent_background: bool,
    /// Global avatar opacity (1.0 = opaque). Drives the fade-out-when-no-
    /// person-detected feature; folded into the per-frame camera uniform.
    pub avatar_opacity: f32,
    /// User-selected output colour space. Stage 1 only routes this through
    /// to `OutputTargetRequest.color_space` and metadata; render target
    /// format / shader gamma changes land in later stages.
    pub output_color_space: crate::renderer::frame_input::RenderColorSpace,
    /// P2-05: how the renderer should export the frame for the active sink.
    /// Populated each tick from `OutputRouter::active_sink().supports_gpu_tokens()`
    /// — sinks that consume `GpuFrameToken` get `GpuExport`, the file-backed
    /// Win32 sinks stay on `CpuReadback` so the existing MF DLL consumer
    /// keeps receiving raw RGBA bytes. Decoupled from any GUI setting so a
    /// sink swap is the single source of truth for the export path.
    pub export_mode: crate::renderer::frame_input::RenderExportMode,
    /// User-selected MSAA level. Forwarded to `OutputTargetRequest.msaa`;
    /// the renderer clamps it to device support and applies it to the
    /// offscreen render pass / pipelines (affects both the preview and the
    /// exported frame, which share the offscreen target).
    pub output_msaa: crate::renderer::frame_input::MsaaMode,
    /// Bloom post-effect parameters. Forwarded to `RenderFrameInput.bloom`
    /// each frame; the renderer applies them via push constants without any
    /// pipeline rebuild.
    pub bloom: crate::renderer::frame_input::BloomSettings,
    /// Generative background parameters. Forwarded to
    /// `RenderFrameInput.generative_background` each frame; push constants
    /// only, no pipeline rebuild.
    pub generative_background: crate::renderer::frame_input::GenerativeBackgroundSettings,
    /// Snapshot of `Application::background_time` for this frame, already
    /// wrapped to keep f32 precision.
    pub time_seconds: f32,
    /// When `Some`, the render camera is the depth sensor's own intrinsics +
    /// a front view (1:1 mirror), overriding the orbit `camera` field. Set
    /// only when the mirror toggle is on and the live pose is metric-native;
    /// `None` keeps the free orbit camera. See [`SensorCamera`].
    pub sensor_camera: Option<SensorCamera>,
}

/// Runtime toggles controlled by the GUI that gate pipeline steps in `run_frame()`.
#[derive(Clone, Debug)]
pub struct RuntimeToggles {
    pub tracking_enabled: bool,
    pub spring_enabled: bool,
    pub cloth_enabled: bool,
    pub collision_debug: bool,
    pub skeleton_debug: bool,
    /// 1:1 sensor-matched mirror render. When set AND the current tracking
    /// pose carries `metric_frame_info` (D435 metric path), the render camera
    /// switches from the free orbit camera to the depth sensor's own
    /// intrinsics + a front-facing view, so the avatar is framed with the
    /// real lens (FOV, principal point) — a mirror. Ignored when the
    /// pose has no intrinsics and while unset.
    pub mirror_view: bool,
}

impl Default for RuntimeToggles {
    fn default() -> Self {
        Self {
            tracking_enabled: true,
            spring_enabled: true,
            cloth_enabled: false,
            collision_debug: false,
            skeleton_debug: false,
            mirror_view: false,
        }
    }
}

/// Depth-sensor camera parameters for the 1:1 mirror render, plumbed from
/// [`crate::tracking::MetricFrameInfo`] into [`FrameInputConfig`] when the
/// mirror toggle is on. `intrinsics` gives the projection (FOV + principal
/// point); `anchor_depth_m` is the subject's neutral forward distance in
/// metres, used to place the eye so the avatar frames at the sensor's scale.
#[derive(Clone, Copy, Debug)]
pub struct SensorCamera {
    pub intrinsics: crate::tracking::CameraIntrinsics,
    pub anchor_depth_m: f32,
}

/// Camera parameters driven by the GUI viewport controls.
#[derive(Clone, Debug)]
pub struct ViewportCamera {
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub distance: f32,
    pub pan: [f32; 2],
    pub fov_deg: f32,
}

impl Default for ViewportCamera {
    fn default() -> Self {
        Self {
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            distance: 5.0,
            pan: [0.0, 0.0],
            fov_deg: 60.0,
        }
    }
}

pub struct Application {
    pub render_thread: Option<RenderThread>,
    pub physics: PhysicsWorld,
    pub tracking: TrackingSource,
    pub output: OutputRouter,
    /// P3-03: central GPU pressure / pacing policy. Driven per-frame by
    /// [`Self::update_runtime_gpu_budget`]; read by render/output/tracking
    /// consumers each tick so cadence decisions come from one place.
    pub runtime_gpu_budget: runtime_gpu_budget::RuntimeGpuBudget,
    /// Tracks frames dropped over the last second so the budget gets a
    /// drops-per-sec measurement rather than a raw counter.
    last_output_drop_count: u64,
    last_output_drop_sample: std::time::Instant,
    /// Smoothed render frame interval (EMA of `FrameConfig::frame_dt`).
    /// Initialised to the 60 fps target; updated each call to
    /// [`Self::run_frame`].
    render_dt_ema: std::time::Duration,
    pub sim_clock: SimulationClock,
    /// Substeps the sim clock yielded for the most recent `run_frame`.
    /// Reported through the GUI debug heartbeat (`debug_gui.json`) so a
    /// single-frame visual glitch can be checked against a zero-substep
    /// frame — the frame where the spring solver does not run.
    pub last_sim_substeps: u32,
    pub editor: EditorSession,
    pub avatars: Vec<AvatarInstance>,
    pub active_avatar_index: usize,
    pub next_avatar_instance_id: u64,
    /// Auto-cloth opt-out: derive and attach GPU cloth for
    /// skirt-classified primitives on every avatar load. Mirrors
    /// `AppSettings::auto_cloth` (published at startup, before any
    /// attach can race it).
    pub auto_cloth_enabled: bool,
    pub running: bool,
    pub avatar_library: avatar_library::AvatarLibrary,

    pub last_tracking_pose: Option<crate::tracking::SourceSkeleton>,

    /// Smoothed global avatar opacity for the fade-out-when-no-person feature.
    /// Lerps toward 1.0 while a person is detected (or within the hold window)
    /// and toward 0.0 once detection has been lost past the hold window. Fed
    /// into `RenderFrameInput::avatar_opacity` each frame. Always 1.0 when the
    /// feature is disabled or tracking is off.
    tracking_fade_opacity: f32,

    rendered_pixels: Option<Arc<Vec<u8>>>,
    rendered_extent: [u32; 2],
    rendered_frame_counter: u64,

    viewport_extent: [u32; 2],

    pub viewport_camera: ViewportCamera,
    pub viewport_lighting: LightingState,
    pub output_extent: Option<[u32; 2]>,
    /// Phase B-4: whether the output sink should preserve the renderer's
    /// alpha channel (true) or force opaque (false). Synced from the GUI
    /// `output.output_has_alpha` toggle every frame.
    pub output_preserve_alpha: bool,
    /// User's selected output colour space (sRGB / Linear sRGB). Synced from
    /// the GUI `output.output_color_space_index` each frame and forwarded
    /// to the renderer through `OutputTargetRequest.color_space`. Stage 1
    /// routes the value end-to-end into the export metadata; downstream
    /// effects on render target format and MF media type land in later
    /// stages.
    pub output_color_space: crate::renderer::frame_input::RenderColorSpace,
    /// User-selected MSAA level. Synced from the GUI Output panel
    /// (`output.msaa_index`) each frame and forwarded to the renderer
    /// through `OutputTargetRequest.msaa`. The renderer clamps it to device
    /// support (and caps integrated GPUs at 4x); the offscreen target is
    /// shared by preview + export, so the chosen level antialiases both.
    pub output_msaa: crate::renderer::frame_input::MsaaMode,
    /// Bloom post-effect parameters. Synced from the GUI Rendering
    /// inspector (Composition section) each frame and forwarded to the
    /// renderer through `RenderFrameInput.bloom`.
    pub bloom: crate::renderer::frame_input::BloomSettings,
    /// Generative background parameters. Synced from the Inspector's
    /// "Scene Background" panel each frame and forwarded to the renderer
    /// through `RenderFrameInput.generative_background`.
    pub generative_background: crate::renderer::frame_input::GenerativeBackgroundSettings,
    /// Background animation clock in seconds, advanced by `frame_dt` every
    /// `run_frame`. Accumulated in f64 and wrapped at 4096 s so the f32
    /// handed to the shader keeps sub-millisecond precision.
    pub background_time: f64,
    /// Solid background colour the Vulkan renderer clears to when
    /// `transparent_background` is false. Synced from the Inspector's
    /// "Scene Background" panel each frame.
    pub background_color: [f32; 3],
    /// When true the renderer clears to `(0,0,0,0)` so only avatar pixels
    /// are non-transparent — useful for OBS chroma keys but produces a
    /// black NV12 output for clients (Meet / Zoom) that don't honour
    /// alpha. Synced from the same inspector panel.
    pub transparent_background: bool,

    pub tracking_worker: Option<TrackingWorker>,
    /// Lifetime owner for the MediaFoundation virtual-camera registration.
    /// `None` until the user enables the VirtualCamera output sink; held
    /// for the rest of the process so the camera stays visible to clients.
    #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
    pub mf_virtual_camera: Option<crate::output::mf_virtual_camera::MfVirtualCamera>,
    /// Lifetime owner for the lipsync audio capture + inference worker. Used
    /// to be tracked by the GUI; moved here so [`Self::set_lipsync_enabled`]
    /// can manage it from any caller (project load reconciliation, GUI
    /// toggle, etc.) without GUI-specific plumbing.
    ///
    /// Deliberately **not `pub`** — the only allowed touch points are
    /// [`Self::set_lipsync_enabled`] (lifecycle) and [`Self::step_lipsync`]
    /// (per-frame inference). Direct access from the GUI would re-enable
    /// the kind of inline side-effect code that T11 was designed to
    /// eliminate.
    #[cfg(feature = "lipsync")]
    lipsync_processor: Option<crate::lipsync::LipSyncProcessor>,
    /// User's selected mic device. Doubles as:
    /// - The mic the running [`Self::lipsync_processor`] was started with
    ///   (so [`Self::set_lipsync_enabled`] can detect mic changes and
    ///   restart in place)
    /// - The mic to use the next time lipsync is enabled (so the user's
    ///   selection survives a disable / re-enable cycle without a
    ///   separate "preferred mic" field)
    ///
    /// Always updated by [`Self::set_lipsync_enabled`], including the
    /// `enabled=false` path.
    #[cfg(feature = "lipsync")]
    lipsync_mic_device_index: usize,
    /// Phase D: user's requested sink, kept separate from the actually-running
    /// `output.active_sink()`. If the runtime fails to switch (typical case:
    /// MFCreateVirtualCamera::Start fails), `requested_sink` keeps the user's
    /// intent so it survives autosave and a future retry. Persisted by
    /// `to_project_state`. Combo box display reads this, not active.
    requested_sink: FrameSink,
    /// Phase D companion to [`Self::lipsync_processor`]: the user's intent.
    /// `lipsync_processor.is_some()` reflects runtime success;
    /// `requested_lipsync_enabled` survives a failed start so the next
    /// mic refresh / app restart can retry, and `to_project_state` saves
    /// what the user picked rather than what currently happens to work.
    #[cfg(feature = "lipsync")]
    requested_lipsync_enabled: bool,
    shutdown_complete: bool,
    pub viewport_background: Option<std::path::PathBuf>,
    pub ground_grid_visible: bool,
    stale_warn_cooldown: std::time::Instant,
    logged_first_render_result: bool,
    /// Number of `RenderCommand::RenderFrame` submissions that have not
    /// yet produced a `RenderResult`. Bumped on a successful submit in
    /// `run_frame`, decremented for each result drained. The GUI repaint
    /// gate reads this so a viewport mutation on an otherwise-idle
    /// avatar still gets at least one follow-up frame to upload the
    /// re-rendered texture instead of stalling on the previous image.
    render_results_pending: u32,
    /// Last quantized pose key per avatar instance, used to skip the
    /// body-SDF splat on frames where the posed body is unchanged
    /// (see `app::render::splat_pose_key`). Only advanced when a splat
    /// actually runs (see the cadence gate in `run_frame`).
    sdf_splat_pose_keys: std::collections::HashMap<u64, u64>,
    /// Last instant each instance's body-SDF splat ran, gating the
    /// refresh cadence to 30 Hz while the pose is moving.
    sdf_splat_last: std::collections::HashMap<u64, std::time::Instant>,
    /// Lifetime tally of render results the render thread had to drop
    /// because the app hadn't drained the mailbox yet. Each entry is a
    /// frame that completed on the GPU but never reached
    /// `process_render_result`. Increasing values indicate the app
    /// thread is falling behind the renderer (modal stalls, long load
    /// jobs, etc.); plumbed for diagnostics surfacing in the inspector.
    render_results_dropped: u64,
    /// Lease completions waiting to be forwarded back to the render thread.
    /// The render command channel is bounded, so failed `try_send` attempts
    /// are retried next tick instead of dropping the release and pinning a
    /// pool slot forever.
    pending_export_lease_releases: VecDeque<u64>,
}

impl Default for Application {
    fn default() -> Self {
        Self::new()
    }
}

impl Application {
    pub fn active_avatar(&self) -> Option<&AvatarInstance> {
        self.avatars.get(self.active_avatar_index)
    }

    /// Get a mutable reference to the active avatar.
    pub fn active_avatar_mut(&mut self) -> Option<&mut AvatarInstance> {
        self.avatars.get_mut(self.active_avatar_index)
    }

    /// Load `instance` as **the** scene avatar, disposing whatever was
    /// loaded before: the previous physics character body is removed,
    /// the old `AvatarInstance`s are dropped, and the render thread is
    /// told to evict its GPU caches so the old textures/meshes don't
    /// stay pinned in VRAM.
    ///
    /// Loading is deliberately replace-only — the app drives a single
    /// avatar from a single tracked person. Multi-avatar scenes would
    /// need per-avatar inference routing that doesn't exist, so an
    /// additive load would only stack identically-posed copies at the
    /// origin while leaking the old ones' physics bodies and VRAM.
    pub fn set_avatar(&mut self, instance: AvatarInstance) {
        let had_existing = !self.avatars.is_empty();
        if had_existing {
            self.detach_avatar();
            self.avatars.clear();
        }
        self.physics.attach_avatar(&instance.asset);
        self.active_avatar_index = 0;
        self.avatars.push(instance);
        if self.auto_cloth_enabled {
            if let Some(avatar) = self.avatars.last_mut() {
                crate::simulation::auto_cloth::attach_auto_cloth(avatar);
            }
        }
        if had_existing {
            self.evict_render_caches();
        }
    }

    /// Publish the auto-cloth preference. Called at startup, before
    /// any avatar can attach (mirrors the cloth-backend request
    /// publication timing).
    pub fn set_auto_cloth_enabled(&mut self, enabled: bool) {
        self.auto_cloth_enabled = enabled;
    }

    pub fn remove_avatar_at(&mut self, index: usize) {
        if index < self.avatars.len() {
            self.detach_avatar();
            self.avatars.remove(index);
            if self.avatars.is_empty() {
                self.active_avatar_index = 0;
            } else if index < self.active_avatar_index {
                self.active_avatar_index -= 1;
            } else if index == self.active_avatar_index
                && self.active_avatar_index >= self.avatars.len()
            {
                self.active_avatar_index = self.avatars.len() - 1;
            }
            self.evict_render_caches();
        }
    }

    /// Tell the render thread to drop its texture and mesh caches.
    /// Call after replacing or removing an avatar so the previous
    /// avatar's GPU resources don't stay pinned in VRAM until process
    /// exit. The next render re-uploads the active avatar's data on
    /// demand — single-frame hitch in exchange for bounded VRAM use.
    pub fn evict_render_caches(&self) {
        if let Some(rt) = self.render_thread.as_ref() {
            rt.submit(crate::app::render_thread::RenderCommand::EvictCaches);
        }
    }

    pub fn detach_avatar(&mut self) {
        let path = self
            .active_avatar()
            .map(|a| a.asset.source_path.to_string_lossy().into_owned());
        if let Some(p) = path {
            self.physics.remove_character_bodies(&p);
        }
    }

    pub fn new() -> Self {
        // Avatar load cache eviction. Best-effort sweep on startup so the
        // cache directory under %APPDATA%\VulVATAR\cache doesn't grow
        // unbounded as the user iterates on different VRMs.
        crate::asset::cache::evict_to_count(crate::asset::cache::DEFAULT_MAX_CACHE_ENTRIES);

        let now = std::time::Instant::now();
        Self {
            render_thread: None,
            physics: PhysicsWorld::new(),
            tracking: TrackingSource::new(),
            output: OutputRouter::new(FrameSink::SharedMemory),
            runtime_gpu_budget: runtime_gpu_budget::RuntimeGpuBudget::new(now),
            last_output_drop_count: 0,
            last_output_drop_sample: now,
            render_dt_ema: std::time::Duration::from_secs_f32(1.0 / 60.0),
            sim_clock: SimulationClock::new(1.0 / 60.0, 8),
            last_sim_substeps: 0,
            editor: EditorSession::new(),
            avatars: Vec::new(),
            active_avatar_index: 0,
            next_avatar_instance_id: 1,
            auto_cloth_enabled: true,
            running: false,
            avatar_library: avatar_library::AvatarLibrary::new(),
            last_tracking_pose: None,
            tracking_fade_opacity: 1.0,
            rendered_pixels: None,
            rendered_extent: [0, 0],
            rendered_frame_counter: 0,

            viewport_extent: [1920, 1080],
            viewport_camera: ViewportCamera::default(),
            viewport_lighting: LightingState::default(),
            output_extent: None,
            output_preserve_alpha: false,
            output_color_space: crate::renderer::frame_input::RenderColorSpace::Srgb,
            output_msaa: crate::renderer::frame_input::MsaaMode::Off,
            bloom: crate::renderer::frame_input::BloomSettings::default(),
            generative_background:
                crate::renderer::frame_input::GenerativeBackgroundSettings::default(),
            background_time: 0.0,
            background_color: [0.1, 0.1, 0.1],
            transparent_background: true,
            tracking_worker: None,
            #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
            mf_virtual_camera: None,
            #[cfg(feature = "lipsync")]
            lipsync_processor: None,
            #[cfg(feature = "lipsync")]
            lipsync_mic_device_index: 0,
            requested_sink: FrameSink::SharedMemory,
            #[cfg(feature = "lipsync")]
            requested_lipsync_enabled: false,
            shutdown_complete: false,
            stale_warn_cooldown: std::time::Instant::now(),
            viewport_background: None,
            ground_grid_visible: false,
            logged_first_render_result: false,
            render_results_pending: 0,
            sdf_splat_pose_keys: std::collections::HashMap::new(),
            sdf_splat_last: std::collections::HashMap::new(),
            render_results_dropped: 0,
            pending_export_lease_releases: VecDeque::new(),
        }
    }

    /// True iff the renderer has been handed a frame that hasn't come
    /// back as a `RenderResult` yet. Drives the GUI repaint gate so a
    /// one-shot mutation (slider drag, sink swap) still produces a
    /// follow-up frame to drain the result, even when no animation,
    /// tracking, or lipsync is active.
    pub fn has_pending_render_result(&self) -> bool {
        self.render_results_pending > 0
    }

    /// Cumulative render-result drops since process start. Increases
    /// monotonically when the render thread had to replace an older
    /// mailbox entry with a newer one because the app hadn't drained
    /// the previous frame yet.
    pub fn render_results_dropped_count(&self) -> u64 {
        self.render_results_dropped
    }

    /// Effective production rate of the render thread (fps), or `None`
    /// while it has no fresh measurement (paused, no avatar, or fewer
    /// than two back-to-back frames since the last idle gap). Surfaced
    /// to the status bar debug readout, `debug_gui.json`, and the GPU
    /// budget — the GUI-side fps display only measures the egui tick,
    /// which spins *faster* than the renderer under backpressure and
    /// therefore cannot see this.
    pub fn render_thread_fps(&self) -> Option<f32> {
        self.render_thread.as_ref().and_then(|rt| rt.render_fps())
    }

    /// EMA of the render thread's CPU-side `render()` duration in ms
    /// (previous frame's fence wait included). Sits at the frame
    /// budget while production sags = GPU-bound; small while fps sags
    /// = the recording path itself is the cost.
    pub fn render_thread_cpu_ms(&self) -> Option<f32> {
        self.render_thread.as_ref().and_then(|rt| rt.render_cpu_ms())
    }

    /// Cumulative count of render commands dropped because the render
    /// thread's bounded command channel was full. Sustained growth is
    /// the direct counterpart of the "failed to send command" warning.
    pub fn render_submit_drops_total(&self) -> u64 {
        self.render_thread
            .as_ref()
            .map(|rt| rt.submit_drops_total())
            .unwrap_or(0)
    }

    /// Set the desired viewport resolution (called by the GUI when the viewport panel resizes).
    pub fn set_viewport_size(&mut self, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        self.viewport_extent = [w, h];
    }

    /// Extent whose aspect the render projection is built from this frame:
    /// the explicit output resolution when set, else the GUI viewport. The
    /// GUI's debug overlays must project through this same aspect (and then
    /// into the letterboxed image rect) to stay aligned with the character.
    pub fn render_extent(&self) -> [u32; 2] {
        self.output_extent.unwrap_or(self.viewport_extent)
    }

    /// Returns the rendered RGBA pixel data and its extent, if available.
    pub fn rendered_pixels(&self) -> Option<(&[u8], [u32; 2])> {
        self.rendered_pixels
            .as_ref()
            .map(|px| (px.as_slice(), self.rendered_extent))
    }

    /// Returns the monotonic frame counter for the last rendered frame.
    pub fn rendered_frame_counter(&self) -> u64 {
        self.rendered_frame_counter
    }

    pub fn bootstrap(&mut self) {
        let renderer = VulkanRenderer::new();
        let rt = RenderThread::new(renderer);
        self.render_thread = Some(rt);
        info!("bootstrap: render thread started");

        // Try to load a default avatar if present, but continue without one
        // if the file is missing or fails to load.
        let loader = VrmAssetLoader::new();
        let default_path = "assets/avatar.vrm";
        if std::path::Path::new(default_path).exists() {
            match loader.load(default_path) {
                Ok(asset) => {
                    let instance_id = AvatarInstanceId(self.next_avatar_instance_id);
                    self.next_avatar_instance_id += 1;
                    self.set_avatar(AvatarInstance::new(instance_id, asset));
                    info!("bootstrap: loaded default avatar from {}", default_path);
                }
                Err(e) => {
                    error!(
                        "bootstrap: failed to load default avatar '{}': {} (continuing without avatar)",
                        default_path, e
                    );
                }
            }
        } else {
            info!(
                "bootstrap: no default avatar at '{}', starting without avatar",
                default_path
            );
        }
        self.running = true;

        // Tracking worker is NOT started at bootstrap.
        // The user must explicitly start it from the Tracking Setup panel.
        info!("bootstrap: tracking worker idle (start from Tracking Setup)");

        // The output worker is already running inside OutputRouter (spawned in
        // OutputRouter::new()), so no additional start call is needed here.
        info!("bootstrap: output worker already running via OutputRouter");
    }

    /// Re-load the currently loaded avatar from its original source path on disk.
    pub fn reload_avatar(&mut self) {
        let source_path = match self.active_avatar() {
            Some(avatar) => avatar.asset.source_path.clone(),
            None => {
                warn!("reload_avatar: no avatar loaded");
                return;
            }
        };

        let loader = VrmAssetLoader::new();
        let path_str = source_path.to_string_lossy();
        let asset = match loader.load(&path_str) {
            Ok(a) => a,
            Err(e) => {
                error!("reload_avatar: failed to reload '{}': {}", path_str, e);
                return;
            }
        };

        let instance_id = AvatarInstanceId(self.next_avatar_instance_id);
        self.next_avatar_instance_id += 1;

        self.set_avatar(AvatarInstance::new(instance_id, asset));
    }

    // ---------------------------------------------------------------------
    // Phase D: requested vs active separation
    //
    // The `requested_*` fields track user intent (what the user picked in
    // the GUI / what the project file says). The `active_*` getters / fields
    // track runtime reality (what's currently running). They diverge when a
    // runtime change fails (e.g. MFCreateVirtualCamera::Start returns Err).
    // GUI display + project save read `requested_*` so the user's choice
    // survives transient failures and isn't silently overwritten by autosave
    // when a fallback kicks in.
    // ---------------------------------------------------------------------

    /// User's requested sink (the choice they made / the project file holds).
    /// May differ from `output.active_sink()` if the runtime failed to switch.
    pub fn requested_sink(&self) -> &FrameSink {
        &self.requested_sink
    }

    /// Store the user's render FPS intent in [`Self::runtime_gpu_budget`]
    /// and immediately forward the budget's (possibly-clamped) target to
    /// [`OutputRouter::set_target_fps`]. The per-frame
    /// [`Self::update_runtime_gpu_budget`] then re-applies the latest
    /// budget value each tick, so a transient pressure spike that
    /// happens *after* the GUI change is still reflected.
    pub fn set_user_render_fps(&mut self, fps: u32) {
        self.runtime_gpu_budget.set_user_render_fps(fps);
        self.output
            .set_target_fps(self.runtime_gpu_budget.render_fps_target());
    }

    /// Update the requested sink and try to make the runtime match. The
    /// requested value is updated unconditionally, even if the runtime
    /// transition fails — the user's intent persists for next time
    /// (autosave, restart, retry).
    pub fn set_requested_sink(&mut self, sink: FrameSink) -> Result<(), String> {
        self.requested_sink = sink.clone();
        self.ensure_output_sink_runtime(sink)
    }

    /// User's requested lipsync enabled state. May differ from
    /// [`Self::is_lipsync_enabled`] if the audio device couldn't open.
    pub fn requested_lipsync_enabled(&self) -> bool {
        #[cfg(feature = "lipsync")]
        return self.requested_lipsync_enabled;
        #[cfg(not(feature = "lipsync"))]
        return false;
    }

    /// Update the requested lipsync state and try to make the runtime match.
    /// As with [`Self::set_requested_sink`], the requested value is updated
    /// regardless of runtime success.
    pub fn set_requested_lipsync(
        &mut self,
        enabled: bool,
        mic_device_index: usize,
    ) -> Result<(), String> {
        #[cfg(feature = "lipsync")]
        {
            self.requested_lipsync_enabled = enabled;
        }
        // mic_device_index is always remembered by set_lipsync_enabled
        // (including the disable path) so we don't need a separate
        // requested_mic field.
        self.set_lipsync_enabled(enabled, mic_device_index)
    }

    /// Perform an ordered shutdown of all worker threads and subsystems.
    ///
    /// Shutdown order (per `docs/threading-model.md`):
    /// 1. Stop tracking worker (no new tracking data)
    /// 2. Drop MF virtual camera (Frame Server releases camera before
    ///    we tear down the producer it reads from)
    /// 3. Stop output worker (drain remaining frames + close producer)
    /// 4. Stop render thread (drains pending work, releases GPU resources)
    ///
    /// Safe to call multiple times; subsequent calls are no-ops.
    pub fn shutdown(&mut self) {
        if self.shutdown_complete {
            return;
        }

        info!("app: shutdown starting");

        // 1. Stop the tracking worker so no new tracking data arrives.
        if let Some(ref mut worker) = self.tracking_worker {
            info!("app: stopping tracking worker...");
            if worker.stop() {
                info!("app: tracking worker stopped");
            } else {
                warn!("app: tracking worker stop timed out during shutdown");
            }
        }

        // 2. Mark application as no longer running (prevents run_frame from
        //    doing further work).
        self.running = false;

        // 3. Drop the MF virtual camera *before* the output worker. The
        //    camera's media source is the Frame Server's reader of our
        //    producer-side shmem mapping; if we let `output.shutdown()`
        //    invalidate that mapping while the camera is still
        //    registered, Frame Server can issue one final `RequestSample`
        //    against torn-down memory. `MfVirtualCamera::Drop` calls
        //    `Stop()` + `Remove()`, after which Frame Server has released
        //    its handle and the producer is safe to retire.
        //
        //    Field-order Drop on Application would tear down `output`
        //    *before* `mf_virtual_camera` (declaration order: output
        //    line 104, mf_virtual_camera line 150), so we must do this
        //    explicitly here regardless of whether `shutdown` is called
        //    directly or via `Drop`.
        #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
        {
            if self.mf_virtual_camera.take().is_some() {
                info!("app: MediaFoundation virtual camera unregistered");
            }
        }

        // 4. Shut down the output router, which drains remaining frames and
        //    joins the output worker thread.
        info!("app: stopping output worker...");
        self.output.shutdown();
        info!("app: output worker stopped");

        // 5. Shut down the render thread, which drains pending work and
        //    releases GPU resources.
        if let Some(ref mut rt) = self.render_thread {
            info!("app: stopping render thread...");
            rt.shutdown();
            info!("app: render thread stopped");
        }

        self.shutdown_complete = true;
        info!("app: shutdown complete");
    }

    pub fn snap_avatar_to_ground(&mut self) {
        if let Some(avatar) = self.active_avatar_mut() {
            let current_y = avatar.world_transform.translation[1];
            avatar.world_transform.translation[1] = 0.0;
            info!("ground-align: snapped avatar from y={} to y=0.0", current_y);
        }
    }

    pub fn set_background_image(&mut self, path: Option<std::path::PathBuf>) {
        self.viewport_background = path;
    }

    pub fn set_ground_grid_visible(&mut self, visible: bool) {
        self.ground_grid_visible = visible;
    }
}

impl Drop for Application {
    fn drop(&mut self) {
        if !self.shutdown_complete {
            self.shutdown();
        }
    }
}

#[cfg(test)]
mod tests;
