pub mod avatar_library;
pub mod bake_cache;
pub mod folder_watcher;
mod lipsync;
mod output_sink;
mod render;
pub mod render_thread;
mod tracking_lifecycle;

use log::{error, info, warn};
use std::sync::Arc;

use crate::app::render_thread::RenderThread;
use crate::asset::vrm::VrmAssetLoader;
use crate::avatar::{AvatarInstance, AvatarInstanceId};
use crate::editor::EditorSession;
use crate::output::{FrameSink, OutputRouter};
use crate::renderer::frame_input::LightingState;
use crate::renderer::VulkanRenderer;
use crate::simulation::{PhysicsWorld, SimulationClock};
use crate::tracking::{TrackingCalibration, TrackingSmoothingParams, TrackingSource, TrackingWorker};

/// All per-frame parameters passed from the GUI to `run_frame()`.
#[derive(Clone, Debug)]
pub struct FrameConfig {
    pub toggles: RuntimeToggles,
    pub smoothing: TrackingSmoothingParams,
    pub material_mode_index: usize,
    pub hand_tracking_enabled: bool,
    pub face_tracking_enabled: bool,
    pub lower_body_tracking_enabled: bool,
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
    /// User-selected output colour space. Stage 1 only routes this through
    /// to `OutputTargetRequest.color_space` and metadata; render target
    /// format / shader gamma changes land in later stages.
    pub output_color_space: crate::renderer::frame_input::RenderColorSpace,
}

/// Runtime toggles controlled by the GUI that gate pipeline steps in `run_frame()`.
#[derive(Clone, Debug)]
pub struct RuntimeToggles {
    pub tracking_enabled: bool,
    pub spring_enabled: bool,
    pub cloth_enabled: bool,
    pub collision_debug: bool,
    pub skeleton_debug: bool,
}

impl Default for RuntimeToggles {
    fn default() -> Self {
        Self {
            tracking_enabled: true,
            spring_enabled: true,
            cloth_enabled: false,
            collision_debug: false,
            skeleton_debug: false,
        }
    }
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
    pub tracking_calibration: TrackingCalibration,
    pub output: OutputRouter,
    pub sim_clock: SimulationClock,
    pub editor: EditorSession,
    pub avatars: Vec<AvatarInstance>,
    pub active_avatar_index: usize,
    pub next_avatar_instance_id: u64,
    pub running: bool,
    pub avatar_library: avatar_library::AvatarLibrary,

    pub last_tracking_pose: Option<crate::tracking::SourceSkeleton>,

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

    pub fn add_avatar(&mut self, instance: AvatarInstance) {
        self.active_avatar_index = self.avatars.len();
        self.avatars.push(instance);
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

        Self {
            render_thread: None,
            physics: PhysicsWorld::new(),
            tracking: TrackingSource::new(),
            tracking_calibration: TrackingCalibration::default(),
            output: OutputRouter::new(FrameSink::SharedMemory),
            sim_clock: SimulationClock::new(1.0 / 60.0, 8),
            editor: EditorSession::new(),
            avatars: Vec::new(),
            active_avatar_index: 0,
            next_avatar_instance_id: 1,
            running: false,
            avatar_library: avatar_library::AvatarLibrary::new(),
            last_tracking_pose: None,
            rendered_pixels: None,
            rendered_extent: [0, 0],
            rendered_frame_counter: 0,

            viewport_extent: [1920, 1080],
            viewport_camera: ViewportCamera::default(),
            viewport_lighting: LightingState::default(),
            output_extent: None,
            output_preserve_alpha: false,
            output_color_space: crate::renderer::frame_input::RenderColorSpace::Srgb,
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
        }
    }

    /// Set the desired viewport resolution (called by the GUI when the viewport panel resizes).
    pub fn set_viewport_size(&mut self, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        self.viewport_extent = [w, h];
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
                    self.physics.attach_avatar(&asset);
                    self.add_avatar(AvatarInstance::new(instance_id, asset));
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

        self.detach_avatar();
        self.physics.attach_avatar(&asset);
        let new_instance = AvatarInstance::new(instance_id, asset);
        if self.active_avatar_index < self.avatars.len() {
            self.avatars[self.active_avatar_index] = new_instance;
        }
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
    /// 2. Render a final frame if needed
    /// 3. Stop output worker (drain remaining frames)
    /// 4. Drop renderer resources (handled by Rust drop order)
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
            worker.stop();
            info!("app: tracking worker stopped");
        }

        // 2. Mark application as no longer running (prevents run_frame from
        //    doing further work).
        self.running = false;

        // 3. Shut down the output router, which drains remaining frames and
        //    joins the output worker thread.
        info!("app: stopping output worker...");
        self.output.shutdown();
        info!("app: output worker stopped");

        // 4. Shut down the render thread, which drains pending work and
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
mod tests {
    use super::*;

    fn mat4_mul_vec4(m: &crate::asset::Mat4, v: [f32; 4]) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for row in 0..4 {
            for col in 0..4 {
                out[row] += m[row][col] * v[col];
            }
        }
        out
    }

    #[test]
    fn projection_maps_near_plane_to_zero() {
        let proj = Application::build_projection_matrix(60.0, 16.0 / 9.0, 0.1, 1000.0);
        let point = [0.0, 0.0, -0.1, 1.0];
        let clip = mat4_mul_vec4(&proj, point);
        assert!(
            clip[3].abs() > 0.001,
            "w_clip should be nonzero, got {}",
            clip[3]
        );
        let ndc_z = clip[2] / clip[3];
        assert!(
            (-0.01..=0.01).contains(&ndc_z),
            "near plane should map to z_ndc=0, got z_ndc={}",
            ndc_z
        );
    }

    #[test]
    fn projection_maps_far_plane_to_one() {
        let proj = Application::build_projection_matrix(60.0, 16.0 / 9.0, 0.1, 1000.0);
        let point = [0.0, 0.0, -1000.0, 1.0];
        let clip = mat4_mul_vec4(&proj, point);
        assert!(clip[3].abs() > 0.001, "w_clip should be nonzero");
        let ndc_z = clip[2] / clip[3];
        assert!(
            (0.99..=1.01).contains(&ndc_z),
            "far plane should map to z_ndc=1, got z_ndc={}",
            ndc_z
        );
    }

    #[test]
    fn projection_avatar_in_front_of_camera_is_visible() {
        let cam = ViewportCamera::default();
        let (view, eye) = Application::build_view_matrix(&cam);
        let proj = Application::build_projection_matrix(cam.fov_deg, 16.0 / 9.0, 0.1, 1000.0);

        let world_point = [0.0, 1.0, 0.0, 1.0];
        let view_point = mat4_mul_vec4(&view, world_point);
        let clip = mat4_mul_vec4(&proj, view_point);

        assert!(
            clip[3] > 0.0,
            "w_clip must be positive for visible geometry, got w={}",
            clip[3]
        );
        let ndc_z = clip[2] / clip[3];
        assert!(
            (0.0..=1.0).contains(&ndc_z),
            "avatar at origin should be in Vulkan clip range [0,1], got z_ndc={} (eye={:?}, view_pt={:?}, clip={:?})",
            ndc_z, eye, view_point, clip
        );
    }

    #[test]
    fn view_matrix_looks_toward_origin() {
        let cam = ViewportCamera {
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            distance: 5.0,
            pan: [0.0, 0.0],
            fov_deg: 60.0,
        };
        let (view, eye) = Application::build_view_matrix(&cam);
        assert!(
            eye[2] > 4.0,
            "camera at yaw=0 should be in +Z, got eye={:?}",
            eye
        );
        let origin_view = mat4_mul_vec4(&view, [0.0, 0.0, 0.0, 1.0]);
        assert!(
            origin_view[2] < 0.0,
            "origin should be at negative z in view space, got z={}",
            origin_view[2]
        );
    }

    fn make_test_avatar() -> AvatarInstance {
        use crate::asset::*;
        let skeleton = SkeletonAsset {
            nodes: vec![SkeletonNode {
                id: NodeId(0),
                name: "root".into(),
                parent: None,
                children: vec![],
                rest_local: Transform::default(),
                humanoid_bone: None,
            }],
            root_nodes: vec![NodeId(0)],
            inverse_bind_matrices: vec![identity_matrix()],
        };

        let verts = VertexData {
            positions: vec![[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0, 0.0]; 3],
            joint_indices: vec![[0, 0, 0, 0]; 3],
            joint_weights: vec![[1.0, 0.0, 0.0, 0.0]; 3],
        };

        let prim = MeshPrimitiveAsset {
            id: PrimitiveId(0),
            vertex_count: 3,
            index_count: 3,
            material_id: MaterialId(0),
            skin: Some(SkinBinding {
                joint_nodes: vec![NodeId(0)],
                inverse_bind_matrices: vec![identity_matrix()],
            }),
            bounds: Aabb {
                min: [-0.5, 0.0, 0.0],
                max: [0.5, 1.0, 0.0],
            },
            vertices: Some(verts),
            indices: Some(vec![0, 1, 2]),
            morph_targets: Vec::new(),
        };

        let mesh = MeshAsset {
            id: MeshId(0),
            name: "test_mesh".into(),
            primitives: vec![prim],
        };

        let asset = Arc::new(AvatarAsset {
            id: AvatarAssetId(0),
            source_path: std::path::PathBuf::from("test.vrm"),
            source_hash: AssetSourceHash([0u8; 32]),
            skeleton,
            meshes: vec![mesh],
            materials: vec![],
            humanoid: None,
            spring_bones: vec![],
            colliders: vec![],
            default_expressions: ExpressionAssetSet {
                expressions: vec![],
            },
            animation_clips: vec![],
            node_to_mesh: Default::default(),
            vrm_meta: Default::default(),
            root_aabb: crate::asset::Aabb::empty(),
            loaded_from_cache: false,
        });

        let mut inst = AvatarInstance::new(AvatarInstanceId(1), asset);
        inst.build_base_pose();
        inst.compute_global_pose();
        inst.build_skinning_matrices();
        inst
    }

    struct Ray {
        origin: [f32; 3],
        dir: [f32; 3],
    }

    struct RayHit {
        t: f32,
        point: [f32; 3],
    }

    fn ray_triangle_intersect(ray: &Ray, v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> Option<f32> {
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let h = [
            ray.dir[1] * e2[2] - ray.dir[2] * e2[1],
            ray.dir[2] * e2[0] - ray.dir[0] * e2[2],
            ray.dir[0] * e2[1] - ray.dir[1] * e2[0],
        ];
        let a = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
        if a.abs() < 1e-8 {
            return None;
        }
        let f = 1.0 / a;
        let s = [
            ray.origin[0] - v0[0],
            ray.origin[1] - v0[1],
            ray.origin[2] - v0[2],
        ];
        let u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = [
            s[1] * e1[2] - s[2] * e1[1],
            s[2] * e1[0] - s[0] * e1[2],
            s[0] * e1[1] - s[1] * e1[0],
        ];
        let v = f * (ray.dir[0] * q[0] + ray.dir[1] * q[1] + ray.dir[2] * q[2]);
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = f * (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]);
        if t > 1e-6 {
            Some(t)
        } else {
            None
        }
    }

    fn ray_cast_avatar(ray: &Ray, avatar: &AvatarInstance) -> Option<RayHit> {
        let mut best: Option<RayHit> = None;
        let skin_mats = &avatar.pose.skinning_matrices;
        for mesh in &avatar.asset.meshes {
            for prim in &mesh.primitives {
                if let Some(ref vd) = prim.vertices {
                    let indices = prim.indices.as_deref().unwrap_or(&[]);
                    for tri in indices.chunks(3) {
                        if tri.len() < 3 {
                            continue;
                        }
                        let apply_skin = |idx: usize| -> [f32; 3] {
                            let p = vd.positions[idx];
                            let ji = vd.joint_indices[idx];
                            let jw = vd.joint_weights[idx];
                            let total = jw[0] + jw[1] + jw[2] + jw[3];
                            if total < 0.001 {
                                return p;
                            }
                            let mut out = [0.0f32; 3];
                            for j in 0..4 {
                                if jw[j] < 0.001 {
                                    continue;
                                }
                                let mat = skin_mats
                                    .get(ji[j] as usize)
                                    .copied()
                                    .unwrap_or_else(crate::asset::identity_matrix);
                                let w = p[0] * mat[0][0]
                                    + p[1] * mat[1][0]
                                    + p[2] * mat[2][0]
                                    + mat[3][0];
                                let x = p[0] * mat[0][1]
                                    + p[1] * mat[1][1]
                                    + p[2] * mat[2][1]
                                    + mat[3][1];
                                let y = p[0] * mat[0][2]
                                    + p[1] * mat[1][2]
                                    + p[2] * mat[2][2]
                                    + mat[3][2];
                                out[0] += jw[j] * w;
                                out[1] += jw[j] * x;
                                out[2] += jw[j] * y;
                            }
                            out
                        };
                        let v0 = apply_skin(tri[0] as usize);
                        let v1 = apply_skin(tri[1] as usize);
                        let v2 = apply_skin(tri[2] as usize);
                        if let Some(t) = ray_triangle_intersect(ray, v0, v1, v2) {
                            if best.as_ref().is_none_or(|b| t < b.t) {
                                let pt = [
                                    ray.origin[0] + t * ray.dir[0],
                                    ray.origin[1] + t * ray.dir[1],
                                    ray.origin[2] + t * ray.dir[2],
                                ];
                                best = Some(RayHit { t, point: pt });
                            }
                        }
                    }
                }
            }
        }
        best
    }

    fn ndc_from_world(
        world: [f32; 3],
        view: &crate::asset::Mat4,
        proj: &crate::asset::Mat4,
    ) -> [f32; 3] {
        let v = mat4_mul_vec4(view, [world[0], world[1], world[2], 1.0]);
        let c = mat4_mul_vec4(proj, v);
        let w = c[3];
        if w.abs() < 1e-8 {
            return [f32::NAN; 3];
        }
        [c[0] / w, c[1] / w, c[2] / w]
    }

    #[test]
    fn ray_cast_center_hits_test_avatar() {
        let avatar = make_test_avatar();
        let cam = ViewportCamera::default();
        let (view, eye) = Application::build_view_matrix(&cam);

        let target = [0.0, 0.5, 0.0];
        let dx = target[0] - eye[0];
        let dy = target[1] - eye[1];
        let dz = target[2] - eye[2];
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        let ray = Ray {
            origin: eye,
            dir: [dx / len, dy / len, dz / len],
        };

        let hit = ray_cast_avatar(&ray, &avatar).expect("center ray should hit the test triangle");
        assert!(hit.t > 0.0, "hit t should be positive, got {}", hit.t);

        let ndc = ndc_from_world(
            hit.point,
            &view,
            &Application::build_projection_matrix(cam.fov_deg, 16.0 / 9.0, 0.1, 1000.0),
        );
        assert!(
            ndc[0] >= -1.0 && ndc[0] <= 1.0,
            "hit NDC x out of range: {}",
            ndc[0]
        );
        assert!(
            ndc[1] >= -1.0 && ndc[1] <= 1.0,
            "hit NDC y out of range: {}",
            ndc[1]
        );
        assert!(
            ndc[2] >= 0.0 && ndc[2] <= 1.0,
            "hit NDC z out of Vulkan range [0,1]: {}",
            ndc[2]
        );
    }

    #[test]
    fn ray_cast_sweep_viewport_corners() {
        let avatar = make_test_avatar();
        let cam = ViewportCamera {
            distance: 3.0,
            ..ViewportCamera::default()
        };
        let (view, eye) = Application::build_view_matrix(&cam);
        let proj = Application::build_projection_matrix(cam.fov_deg, 16.0 / 9.0, 0.1, 1000.0);

        let fov_rad = cam.fov_deg.to_radians();
        let half_fov = fov_rad * 0.5;
        let aspect = 16.0 / 9.0;
        let tan_h = half_fov.tan();
        let tan_w = tan_h * aspect;
        let fwd = [0.0 - eye[0], 0.0 - eye[1], 0.0 - eye[2]];
        let flen = (fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2]).sqrt();
        let fwd = [fwd[0] / flen, fwd[1] / flen, fwd[2] / flen];

        let right = [fwd[2], 0.0, -fwd[0]];
        let rlen = (right[0] * right[0] + right[1] * right[1] + right[2] * right[2]).sqrt();
        let right = [right[0] / rlen, right[1] / rlen, right[2] / rlen];
        let up = [
            right[1] * fwd[2] - right[2] * fwd[1],
            right[2] * fwd[0] - right[0] * fwd[2],
            right[0] * fwd[1] - right[1] * fwd[0],
        ];

        let mut hits = 0usize;
        let mut visible_ndc = 0usize;
        for &(sx, sy) in &[
            (0.0f32, 0.0),
            (0.3, 0.0),
            (-0.3, 0.0),
            (0.0, 0.3),
            (0.0, -0.3),
        ] {
            let dx = fwd[0] + sx * tan_w * right[0] + sy * tan_h * up[0];
            let dy = fwd[1] + sx * tan_w * right[1] + sy * tan_h * up[1];
            let dz = fwd[2] + sx * tan_w * right[2] + sy * tan_h * up[2];
            let dl = (dx * dx + dy * dy + dz * dz).sqrt();
            let ray = Ray {
                origin: eye,
                dir: [dx / dl, dy / dl, dz / dl],
            };
            if let Some(hit) = ray_cast_avatar(&ray, &avatar) {
                hits += 1;
                let ndc = ndc_from_world(hit.point, &view, &proj);
                if ndc[0] >= -1.0
                    && ndc[0] <= 1.0
                    && ndc[1] >= -1.0
                    && ndc[1] <= 1.0
                    && ndc[2] >= 0.0
                    && ndc[2] <= 1.0
                {
                    visible_ndc += 1;
                }
            }
        }
        assert!(hits > 0, "at least one sweep ray should hit the avatar");
        assert!(
            visible_ndc > 0,
            "at least one hit should be in Vulkan NDC range, got {}/{} hits visible",
            visible_ndc,
            hits
        );
    }

    #[test]
    fn ray_cast_aabb_frustum_culling() {
        let _avatar = make_test_avatar();
        let cam = ViewportCamera::default();
        let (view, _eye) = Application::build_view_matrix(&cam);
        let proj = Application::build_projection_matrix(cam.fov_deg, 16.0 / 9.0, 0.1, 1000.0);

        let mut inside_count = 0;
        let corners = [
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [-0.5, 1.0, 0.0],
            [0.5, 1.0, 0.0],
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [-0.5, 1.0, 0.0],
            [0.5, 1.0, 0.0],
        ];
        for &c in &corners {
            let ndc = ndc_from_world(c, &view, &proj);
            if ndc[0] >= -1.0
                && ndc[0] <= 1.0
                && ndc[1] >= -1.0
                && ndc[1] <= 1.0
                && ndc[2] >= 0.0
                && ndc[2] <= 1.0
            {
                inside_count += 1;
            }
        }
        assert!(
            inside_count >= 4,
            "AABB corners should be inside frustum: {}/8 corners visible",
            inside_count
        );
    }
}
