pub mod avatar_library;
pub mod bake_cache;
pub mod folder_watcher;
pub mod render_thread;

use log::{error, info, warn};
use std::sync::Arc;

use crate::app::render_thread::{RenderCommand, RenderThread};
use crate::asset::vrm::VrmAssetLoader;
use crate::avatar::retargeting;
use crate::avatar::{AvatarInstance, AvatarInstanceId};
use crate::editor::EditorSession;
use crate::output::{FrameSink, OutputFrame, OutputRouter};
use crate::renderer::frame_input::RenderDebugFlags;
use crate::renderer::frame_input::{
    CameraState, ClothDeformSnapshot, LightingState, OutlineSnapshot, OutputTargetRequest,
    RenderAlphaMode, RenderAvatarInstance, RenderCullMode, RenderExportMode, RenderFrameInput,
    RenderMeshInstance, RenderOutputAlpha,
};
use crate::renderer::material::MaterialShaderMode;
use crate::renderer::material::MaterialUploadRequest;
use crate::renderer::VulkanRenderer;
use crate::simulation::{PhysicsWorld, SimulationClock};
use crate::tracking::{
    CameraBackend, TrackingCalibration, TrackingSmoothingParams, TrackingSource, TrackingWorker,
};

/// All per-frame parameters passed from the GUI to `run_frame()`.
#[derive(Clone, Debug)]
pub struct FrameConfig {
    pub toggles: RuntimeToggles,
    pub smoothing: TrackingSmoothingParams,
    pub material_mode_index: usize,
    pub hand_tracking_enabled: bool,
    pub face_tracking_enabled: bool,
    pub frame_dt: f32,
}

/// Configuration for building the renderer-facing frame snapshot.
#[derive(Clone, Debug)]
pub struct FrameInputConfig {
    pub camera: ViewportCamera,
    pub lighting: LightingState,
    pub viewport_extent: [u32; 2],
    pub output_extent: [u32; 2],
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

    pub last_tracking_pose: Option<crate::tracking::TrackingRigPose>,

    rendered_pixels: Option<Arc<Vec<u8>>>,
    rendered_extent: [u32; 2],
    rendered_frame_counter: u64,

    viewport_extent: [u32; 2],

    pub viewport_camera: ViewportCamera,
    pub viewport_lighting: LightingState,
    pub output_extent: Option<[u32; 2]>,

    pub tracking_worker: Option<TrackingWorker>,
    shutdown_complete: bool,
    pub viewport_background: Option<std::path::PathBuf>,
    pub ground_grid_visible: bool,
    stale_warn_cooldown: std::time::Instant,
}

impl Application {
    /// Get a reference to the active avatar (the one at `active_avatar_index`).
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
            } else {
                if index < self.active_avatar_index {
                    self.active_avatar_index -= 1;
                } else if index == self.active_avatar_index
                    && self.active_avatar_index >= self.avatars.len()
                {
                    self.active_avatar_index = self.avatars.len() - 1;
                }
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
            tracking_worker: None,
            shutdown_complete: false,
            stale_warn_cooldown: std::time::Instant::now(),
            viewport_background: None,
            ground_grid_visible: false,
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

    pub fn run_frame(&mut self, config: &FrameConfig) {
        if !self.running {
            return;
        }

        let toggles = &config.toggles;
        let smoothing_params = &config.smoothing;
        let hand_tracking_enabled = config.hand_tracking_enabled;
        let face_tracking_enabled = config.face_tracking_enabled;
        let frame_dt = config.frame_dt;
        let material_mode_index = config.material_mode_index;

        // 1. input update
        self.input_update(frame_dt);

        // 2. read the latest completed tracking sample from the async tracking worker
        let tracking_sample = if toggles.tracking_enabled {
            if self.tracking.mailbox().is_stale() {
                if self.stale_warn_cooldown.elapsed() >= std::time::Duration::from_secs(5) {
                    warn!("tracking: stale sample detected, skipping tracking this frame");
                    self.stale_warn_cooldown = std::time::Instant::now();
                }
                None
            } else {
                self.step_tracking()
            }
        } else {
            None
        };

        // Run simulation on all avatars
        if let Some(ref tp) = tracking_sample {
            self.last_tracking_pose = Some(tp.clone());
        }

        for avatar in self.avatars.iter_mut() {
            avatar.build_base_pose();

            if let Some(ref mut tracking_pose) = tracking_sample.clone() {
                self.tracking_calibration.apply_calibration(tracking_pose);

                let humanoid = avatar.asset.humanoid.as_ref();
                retargeting::retarget_tracking_to_pose_filtered(
                    tracking_pose,
                    &avatar.asset.skeleton,
                    humanoid,
                    &mut avatar.pose.local_transforms,
                    smoothing_params,
                    hand_tracking_enabled,
                    face_tracking_enabled,
                );

                if face_tracking_enabled {
                    let new_weights = retargeting::retarget_expressions(
                        tracking_pose,
                        &avatar.asset.default_expressions,
                        Some(&avatar.expression_weights),
                    );
                    avatar.expression_weights = new_weights;
                }
            }

            avatar.compute_global_pose();

            if self.physics.rapier_initialized() {
                self.physics.step_all(frame_dt, avatar);
                avatar.compute_global_pose();
            } else {
                let substeps = self.sim_clock.advance(frame_dt);
                if toggles.spring_enabled {
                    self.physics
                        .step_springs(self.sim_clock.fixed_dt(), substeps, avatar);
                    avatar.compute_global_pose();
                }
                if toggles.cloth_enabled && avatar.cloth_enabled {
                    for _ in 0..substeps {
                        self.physics.step_cloth(self.sim_clock.fixed_dt(), avatar);
                    }
                    avatar.compute_global_pose();
                }
            }

            avatar.build_skinning_matrices();
        }

        if !self.avatars.is_empty() {
            let output_extent = self.output_extent.unwrap_or(self.viewport_extent);
            let fi_config = FrameInputConfig {
                camera: self.viewport_camera.clone(),
                lighting: self.viewport_lighting.clone(),
                viewport_extent: self.viewport_extent,
                output_extent,
            };
            let frame_input = Self::build_frame_input_multi(
                &self.avatars,
                &fi_config,
                toggles,
                material_mode_index,
                self.viewport_background.as_deref(),
                self.ground_grid_visible,
            );

            if let Some(ref rt) = self.render_thread {
                rt.submit(RenderCommand::RenderFrame(frame_input));
            }

            let pending_results: Vec<_> = if let Some(ref rt) = self.render_thread {
                std::iter::from_fn(|| rt.try_recv_result()).collect()
            } else {
                vec![]
            };

            for result in pending_results {
                self.process_render_result(result);
            }
        }
    }

    fn process_render_result(&mut self, render_result: crate::renderer::RenderResult) {
        let Some(exported) = render_result.exported_frame else {
            self.rendered_pixels = None;
            return;
        };

        let mut output_frame = OutputFrame::new(
            exported.gpu_token_id,
            exported.extent,
            exported.timestamp_nanos,
        );

        output_frame.handoff_path = exported.handoff_path.clone();

        match exported.pixel_data {
            crate::renderer::output_export::ExportedPixelData::CpuReadback(ref pixel_data)
                if !pixel_data.is_empty() =>
            {
                self.rendered_pixels = Some(Arc::clone(pixel_data));
                self.rendered_extent = render_result.extent;
                self.rendered_frame_counter += 1;
                output_frame.pixel_data = Some(Arc::clone(pixel_data));
            }
            crate::renderer::output_export::ExportedPixelData::GpuOwned => {
                self.rendered_extent = render_result.extent;
                self.rendered_frame_counter += 1;
                output_frame.gpu_image = exported.gpu_image.clone();
            }
            _ => {
                self.rendered_pixels = None;
            }
        }

        self.output.publish(output_frame);
    }

    /// Build a view matrix from orbital camera parameters (yaw, pitch, distance, pan).
    fn build_view_matrix(cam: &ViewportCamera) -> (crate::asset::Mat4, [f32; 3]) {
        let yaw = cam.yaw_deg.to_radians();
        let pitch = cam.pitch_deg.to_radians();
        let (sy, cy) = (yaw.sin(), yaw.cos());
        let (sp, cp) = (pitch.sin(), pitch.cos());

        let right = [cy, 0.0, -sy];
        let up = [-sy * sp, cp, -cy * sp];

        let wx = cam.pan[0] * right[0] + cam.pan[1] * up[0];
        let wy = cam.pan[0] * right[1] + cam.pan[1] * up[1];
        let wz = cam.pan[0] * right[2] + cam.pan[1] * up[2];

        let eye_x = cam.distance * cp * sy + wx;
        let eye_y = cam.distance * sp + wy;
        let eye_z = cam.distance * cp * cy + wz;

        let target = [wx, wy, wz];

        let fwd = [target[0] - eye_x, target[1] - eye_y, target[2] - eye_z];
        let len = (fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2])
            .sqrt()
            .max(1e-6);
        let f = [fwd[0] / len, fwd[1] / len, fwd[2] / len];

        let world_up = [0.0f32, 1.0, 0.0];
        let r = [
            f[1] * world_up[2] - f[2] * world_up[1],
            f[2] * world_up[0] - f[0] * world_up[2],
            f[0] * world_up[1] - f[1] * world_up[0],
        ];
        let rlen = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt().max(1e-6);
        let r = [r[0] / rlen, r[1] / rlen, r[2] / rlen];

        let u = [
            r[1] * f[2] - r[2] * f[1],
            r[2] * f[0] - r[0] * f[2],
            r[0] * f[1] - r[1] * f[0],
        ];

        let view_matrix = [
            [
                r[0],
                r[1],
                r[2],
                -(r[0] * eye_x + r[1] * eye_y + r[2] * eye_z),
            ],
            [
                u[0],
                u[1],
                u[2],
                -(u[0] * eye_x + u[1] * eye_y + u[2] * eye_z),
            ],
            [
                -f[0],
                -f[1],
                -f[2],
                f[0] * eye_x + f[1] * eye_y + f[2] * eye_z,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ];
        (view_matrix, [eye_x, eye_y, eye_z])
    }

    /// Build a perspective projection matrix from FOV (degrees) and aspect ratio.
    fn build_projection_matrix(
        fov_deg: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> crate::asset::Mat4 {
        let fov_rad = fov_deg.to_radians();
        let f = 1.0 / (fov_rad * 0.5).tan();
        let a = far / (near - far);
        let b = far * near / (near - far);
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, -f, 0.0, 0.0],
            [0.0, 0.0, a, b],
            [0.0, 0.0, -1.0, 0.0],
        ]
    }

    fn build_frame_input_multi(
        avatars: &[AvatarInstance],
        fi_config: &FrameInputConfig,
        toggles: &RuntimeToggles,
        material_mode_index: usize,
        background_image_path: Option<&std::path::Path>,
        show_ground_grid: bool,
    ) -> RenderFrameInput {
        let cam = &fi_config.camera;
        let lighting = &fi_config.lighting;
        let viewport_extent = fi_config.viewport_extent;
        let output_extent = fi_config.output_extent;

        let instances: Vec<RenderAvatarInstance> = avatars
            .iter()
            .map(|avatar| {
                let mesh_instances: Vec<RenderMeshInstance> = avatar
                    .asset
                    .meshes
                    .iter()
                    .flat_map(|mesh| {
                        mesh.primitives.iter().map(|prim| {
                            let material_asset = avatar
                                .asset
                                .materials
                                .iter()
                                .find(|m| m.id == prim.material_id);

                            let mut material_binding = material_asset
                                .map(MaterialUploadRequest::from_asset_material)
                                .unwrap_or_else(|| {
                                    if let Some(m) = avatar.asset.materials.first() {
                                        MaterialUploadRequest::from_asset_material(m)
                                    } else {
                                        MaterialUploadRequest::default_material()
                                    }
                                });

                            material_binding.mode = match material_mode_index {
                                0 => MaterialShaderMode::Unlit,
                                1 => MaterialShaderMode::SimpleLit,
                                _ => MaterialShaderMode::ToonLike,
                            };

                            let outline = OutlineSnapshot {
                                enabled: material_binding.outline_width > 0.0,
                                width: material_binding.outline_width,
                                color: material_binding.outline_color,
                            };

                            let alpha_mode = match material_binding.alpha_mode {
                                crate::asset::AlphaMode::Opaque => RenderAlphaMode::Opaque,
                                crate::asset::AlphaMode::Mask(_) => RenderAlphaMode::Cutout,
                                crate::asset::AlphaMode::Blend => RenderAlphaMode::Blend,
                            };

                            let cull_mode = if material_binding.double_sided {
                                RenderCullMode::DoubleSided
                            } else {
                                RenderCullMode::BackFace
                            };

                            // Resolve expression weights → per-primitive morph weights.
                            let morph_weights = if prim.morph_targets.is_empty() {
                                Vec::new()
                            } else {
                                let mut weights = vec![0.0f32; prim.morph_targets.len()];
                                for ew in &avatar.expression_weights {
                                    if let Some(expr_def) = avatar
                                        .asset
                                        .default_expressions
                                        .expressions
                                        .iter()
                                        .find(|e| e.name == ew.name)
                                    {
                                        for bind in &expr_def.morph_binds {
                                            // Resolve node → mesh index.
                                            if let Some(&mi) =
                                                avatar.asset.node_to_mesh.get(&bind.node_index)
                                            {
                                                if let Some(m) = avatar.asset.meshes.get(mi) {
                                                    if m.primitives.iter().any(|p| p.id == prim.id)
                                                        && bind.morph_target_index < weights.len()
                                                    {
                                                        weights[bind.morph_target_index] +=
                                                            ew.weight * bind.weight;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                for w in &mut weights {
                                    *w = w.clamp(0.0, 1.0);
                                }
                                weights
                            };

                            RenderMeshInstance {
                                mesh_id: mesh.id,
                                primitive_id: prim.id,
                                material_binding,
                                bounds: prim.bounds.clone(),
                                alpha_mode,
                                cull_mode,
                                outline,
                                primitive_data: Some(std::sync::Arc::new(prim.clone())),
                                morph_weights,
                            }
                        })
                    })
                    .collect();

                let cloth_deform = avatar
                    .cloth_state
                    .as_ref()
                    .or_else(|| {
                        avatar
                            .cloth_overlays
                            .iter()
                            .find(|s| s.enabled)
                            .map(|s| &s.state)
                    })
                    .map(|cs| ClothDeformSnapshot {
                        deformed_positions: cs.deform_output.deformed_positions.clone(),
                        deformed_normals: cs.deform_output.deformed_normals.clone(),
                        version: cs.deform_output.version,
                    });

                RenderAvatarInstance {
                    instance_id: avatar.id,
                    world_transform: avatar.world_transform.clone(),
                    mesh_instances,
                    skinning_matrices: avatar.pose.skinning_matrices.clone(),
                    cloth_deform,
                    debug_flags: RenderDebugFlags {
                        show_skeleton: toggles.skeleton_debug,
                        show_colliders: toggles.collision_debug,
                        show_cloth_mesh: toggles.cloth_enabled,
                        show_normals: false,
                        material_mode_override: Some(match material_mode_index {
                            0 => MaterialShaderMode::Unlit,
                            1 => MaterialShaderMode::SimpleLit,
                            _ => MaterialShaderMode::ToonLike,
                        }),
                    },
                }
            })
            .collect();

        let aspect = output_extent[0] as f32 / output_extent[1].max(1) as f32;
        let (view, eye_pos) = Self::build_view_matrix(cam);
        let projection = Self::build_projection_matrix(cam.fov_deg, aspect, 0.1, 1000.0);

        RenderFrameInput {
            camera: CameraState {
                view,
                projection,
                position_ws: eye_pos,
                viewport_extent,
            },
            lighting: lighting.clone(),
            instances,
            output_request: OutputTargetRequest {
                preview_enabled: true,
                output_enabled: true,
                extent: output_extent,
                color_space: crate::renderer::frame_input::RenderColorSpace::Srgb,
                alpha_mode: RenderOutputAlpha::Premultiplied,
                export_mode: RenderExportMode::CpuReadback,
            },
            background_image_path: background_image_path.map(|p| p.to_path_buf()),
            show_ground_grid,
        }
    }

    /// Per-frame input processing.
    ///
    /// Advances the animation playhead and applies any pending state that the
    /// GUI or external systems wrote into the `Application` between frames.
    fn input_update(&mut self, frame_dt: f32) {
        use crate::avatar::animation;

        for avatar in self.avatars.iter_mut() {
            if let Some(clip_id) = &avatar.animation_state.active_clip {
                let duration =
                    animation::clip_duration(clip_id, &avatar.asset.animation_clips).unwrap_or(1.0);
                avatar.animation_state.advance(frame_dt, duration);
            }
        }
    }

    fn step_tracking(&mut self) -> Option<crate::tracking::TrackingRigPose> {
        // When the tracking worker is running, read the latest pose from the
        // shared mailbox (non-blocking). The worker thread is producing poses
        // independently, so we just grab whatever is newest.
        let worker_running = self
            .tracking_worker
            .as_ref()
            .map_or(false, |w| w.is_running());

        if worker_running {
            // Non-blocking read from the shared mailbox populated by the worker.
            self.tracking.mailbox().read()
        } else {
            // No worker running — return None so tracking is skipped.
            // (The old synchronous fallback generated synthetic data on every
            // frame and spammed info logs at 60 fps.)
            None
        }
    }

    /// Replace the active output sink, shutting down the old OutputRouter and
    /// creating a new one with the selected sink.
    pub fn set_output_sink(&mut self, sink: FrameSink) {
        self.output.shutdown();
        self.output = OutputRouter::new(sink);
    }

    /// Start the tracking worker thread with default resolution/fps. No-op if already running.
    #[allow(dead_code)]
    pub fn start_tracking(&mut self, backend: CameraBackend) {
        self.start_tracking_with_params(backend, 640, 480, 30);
    }

    /// Like [`Self::start_tracking`] but allows specifying the webcam resolution and fps.
    ///
    /// If a worker is already running it is stopped and replaced with the new
    /// backend / parameters so the caller can switch backends at any time.
    pub fn start_tracking_with_params(
        &mut self,
        backend: CameraBackend,
        width: u32,
        height: u32,
        fps: u32,
    ) {
        if let Some(ref mut worker) = self.tracking_worker {
            if worker.is_running() {
                worker.stop();
                info!("app: stopped previous tracking worker for backend switch");
            }
        }
        let shared_mailbox = self.tracking.shared_mailbox();
        let mut worker = TrackingWorker::new(shared_mailbox);
        worker.start_with_params(backend, width, height, fps);
        if worker.is_running() {
            info!("app: tracking worker started");
        } else {
            warn!("app: tracking worker failed to start");
        }
        self.tracking_worker = Some(worker);
    }

    /// Stop the tracking worker thread. No-op if not running.
    pub fn stop_tracking(&mut self) {
        if let Some(ref mut worker) = self.tracking_worker {
            worker.stop();
            info!("app: tracking worker stopped");
        }
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
            ndc_z >= -0.01 && ndc_z <= 0.01,
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
            ndc_z >= 0.99 && ndc_z <= 1.01,
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
            ndc_z >= 0.0 && ndc_z <= 1.0,
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
        if u < 0.0 || u > 1.0 {
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
                                    .unwrap_or_else(|| crate::asset::identity_matrix());
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
                            if best.as_ref().map_or(true, |b| t < b.t) {
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
