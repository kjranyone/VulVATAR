use std::sync::Arc;

use crate::asset::vrm::VrmAssetLoader;
use crate::avatar::retargeting;
use crate::avatar::{AvatarInstance, AvatarInstanceId};
use crate::editor::EditorSession;
use crate::output::{FrameSink, OutputFrame, OutputRouter};
use crate::renderer::frame_input::{
    CameraState, ClothDeformSnapshot, LightingState, OutlineSnapshot, OutputTargetRequest,
    RenderAlphaMode, RenderAvatarInstance, RenderCullMode, RenderExportMode, RenderFrameInput,
    RenderMeshInstance, RenderOutputAlpha,
};
use crate::renderer::material::MaterialUploadRequest;
use crate::renderer::VulkanRenderer;
use crate::simulation::{PhysicsWorld, SimulationClock};
use crate::renderer::frame_input::RenderDebugFlags;
use crate::renderer::material::MaterialShaderMode;
use crate::tracking::{
    CameraBackend, TrackingCalibration, TrackingSmoothingParams, TrackingSource, TrackingWorker,
};

// ---------------------------------------------------------------------------
// Render Command (prepared for future render-thread separation)
// ---------------------------------------------------------------------------

/// Commands that can be submitted to the render subsystem.
///
/// These variants are not yet dispatched at runtime. They exist to define the
/// interface contract for the planned render-thread separation, where commands
/// will be sent to a dedicated render thread via a bounded channel. Until that
/// work lands, the renderer is invoked synchronously inside `run_frame()`.
#[allow(dead_code)]
pub enum RenderCommand {
    /// Render a complete frame from the given snapshot.
    RenderFrame(RenderFrameInput),
    /// Resize the swapchain / render targets.
    Resize(u32, u32),
    /// Shut down the renderer and release GPU resources.
    Shutdown,
}

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
    pub renderer: VulkanRenderer,
    pub physics: PhysicsWorld,
    pub tracking: TrackingSource,
    pub tracking_calibration: TrackingCalibration,
    pub output: OutputRouter,
    pub sim_clock: SimulationClock,
    pub editor: EditorSession,
    pub avatar: Option<AvatarInstance>,
    pub next_avatar_instance_id: u64,
    pub running: bool,

    /// The most recent tracking pose received from the tracking worker.
    /// Stored here for GUI inspector display (timestamp, confidence bars).
    pub last_tracking_pose: Option<crate::tracking::TrackingRigPose>,

    // Viewport integration: rendered pixel readback (Arc-wrapped to share
    // with the output pipeline without copying ~8 MB per frame at 1080p)
    rendered_pixels: Option<Arc<Vec<u8>>>,
    rendered_extent: [u32; 2],
    rendered_frame_counter: u64,

    // Viewport size requested by the GUI
    viewport_extent: [u32; 2],

    // Camera state driven by GUI viewport controls
    pub viewport_camera: ViewportCamera,

    // Lighting state driven by GUI
    pub viewport_lighting: LightingState,

    // Output resolution preference (may differ from viewport extent)
    pub output_extent: Option<[u32; 2]>,

    // Worker threads
    pub tracking_worker: Option<TrackingWorker>,
    /// Set to true after `shutdown()` has been called to prevent double-shutdown.
    shutdown_complete: bool,
}

impl Application {
    pub fn new() -> Self {
        Self {
            renderer: VulkanRenderer::new(),
            physics: PhysicsWorld::new(),
            tracking: TrackingSource::new(),
            tracking_calibration: TrackingCalibration::default(),
            output: OutputRouter::new(FrameSink::SharedMemory),
            sim_clock: SimulationClock::new(1.0 / 60.0, 8),
            editor: EditorSession::new(),
            avatar: None,
            next_avatar_instance_id: 1,
            running: false,
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
        self.renderer.initialize();

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
                    self.avatar = Some(AvatarInstance::new(instance_id, asset));
                    println!("bootstrap: loaded default avatar from {}", default_path);
                }
                Err(e) => {
                    eprintln!(
                        "bootstrap: failed to load default avatar '{}': {} (continuing without avatar)",
                        default_path, e
                    );
                }
            }
        } else {
            println!("bootstrap: no default avatar at '{}', starting without avatar", default_path);
        }
        self.running = true;

        // Start the tracking worker thread.
        // The worker publishes poses to the same mailbox that the app thread
        // reads from via `self.tracking.mailbox()`.
        let shared_mailbox = self.tracking.shared_mailbox();
        let mut worker = TrackingWorker::new(shared_mailbox);
        worker.start(crate::tracking::CameraBackend::default());
        if worker.is_running() {
            println!("bootstrap: tracking worker started");
        } else {
            eprintln!(
                "bootstrap: tracking worker failed to start, falling back to synchronous tracking"
            );
        }
        self.tracking_worker = Some(worker);

        // The output worker is already running inside OutputRouter (spawned in
        // OutputRouter::new()), so no additional start call is needed here.
        println!("bootstrap: output worker already running via OutputRouter");
    }

    /// Re-load the currently loaded avatar from its original source path on disk.
    pub fn reload_avatar(&mut self) {
        let source_path = match self.avatar.as_ref() {
            Some(avatar) => avatar.asset.source_path.clone(),
            None => {
                eprintln!("reload_avatar: no avatar loaded");
                return;
            }
        };

        let loader = VrmAssetLoader::new();
        let path_str = source_path.to_string_lossy();
        let asset = match loader.load(&path_str) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("reload_avatar: failed to reload '{}': {}", path_str, e);
                return;
            }
        };

        let instance_id = AvatarInstanceId(self.next_avatar_instance_id);
        self.next_avatar_instance_id += 1;

        self.physics.attach_avatar(&asset);
        self.avatar = Some(AvatarInstance::new(instance_id, asset));
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
                eprintln!("tracking: stale sample detected, skipping tracking this frame");
                None
            } else {
                self.step_tracking()
            }
        } else {
            None
        };

        if let Some(avatar) = self.avatar.as_mut() {
            // 3. locomotion or gameplay update
            // 4. animation sampling
            // 5. base local pose build
            avatar.build_base_pose();

            // 6. merge tracking inputs
            if let Some(mut tracking_pose) = tracking_sample {
                // Apply calibration before retargeting (M8)
                self.tracking_calibration
                    .apply_calibration(&mut tracking_pose);

                self.last_tracking_pose = Some(tracking_pose.clone());
                let humanoid = avatar.asset.humanoid.as_ref();
                retargeting::retarget_tracking_to_pose_filtered(
                    &tracking_pose,
                    &avatar.asset.skeleton,
                    humanoid,
                    &mut avatar.pose.local_transforms,
                    smoothing_params,
                    hand_tracking_enabled,
                    face_tracking_enabled,
                );

                // 6b. Retarget expression weights from tracking data.
                // Only run expression retargeting when face tracking is enabled.
                if face_tracking_enabled {
                    let new_weights = retargeting::retarget_expressions(
                        &tracking_pose,
                        &avatar.asset.default_expressions,
                        Some(&avatar.expression_weights),
                    );
                    avatar.expression_weights = new_weights;
                }
            }

            // 7. compute driven global pose for the body before secondary motion
            avatar.compute_global_pose();

            // 8. fixed-timestep spring bone simulation against that pose
            // 9. recompute global pose after spring outputs are applied
            let substeps = self.sim_clock.advance(frame_dt);
            if toggles.spring_enabled {
                for _ in 0..substeps {
                    self.physics.step_springs(self.sim_clock.fixed_dt(), avatar);
                }
                avatar.compute_global_pose();
            }

            // 10. fixed-timestep cloth simulation against the resolved body motion for this frame
            // 11. recompute final global pose
            if toggles.cloth_enabled && avatar.cloth_enabled {
                for _ in 0..substeps {
                    self.physics.step_cloth(self.sim_clock.fixed_dt(), avatar);
                }
                avatar.compute_global_pose();
            }

            // 12. build skinning matrices
            avatar.build_skinning_matrices();

            // 13. upload pose buffers
            // 14. upload optional cloth deformation buffers

            // 15. build renderer-facing frame snapshot
            let output_extent = self.output_extent.unwrap_or(self.viewport_extent);
            let fi_config = FrameInputConfig {
                camera: self.viewport_camera.clone(),
                lighting: self.viewport_lighting.clone(),
                viewport_extent: self.viewport_extent,
                output_extent,
            };
            let frame_input = Self::build_frame_input(
                avatar,
                &fi_config,
                toggles,
                material_mode_index,
            );

            // 16. render
            let render_result = self.renderer.render(&frame_input);

            // 17. Use GPU-rendered pixel data when available; fall back to a
            //     synthetic placeholder image when the renderer has not yet
            //     produced real pixels (e.g. during early init or without GPU).
            let has_gpu_pixels = render_result
                .exported_frame
                .as_ref()
                .map_or(false, |ef| !ef.pixel_data.is_empty());

            if has_gpu_pixels {
                let exported = render_result.exported_frame.unwrap();
                // Share the pixel buffer via Arc: one reference for the
                // viewport display, one for the output pipeline.  This avoids
                // cloning ~8 MB of RGBA data per frame at 1080p.
                let pixel_data = exported.pixel_data; // Arc<Vec<u8>>
                self.rendered_pixels = Some(Arc::clone(&pixel_data));
                self.rendered_extent = render_result.extent;
                self.rendered_frame_counter += 1;

                // 18. Construct OutputFrame from raw renderer data (app layer
                // bridges the renderer and output modules).
                let mut output_frame = OutputFrame::new(
                    exported.gpu_token_id,
                    exported.extent,
                    exported.timestamp_nanos,
                );
                output_frame.pixel_data = Some(pixel_data);
                self.output.publish(output_frame);
            } else {
                // No GPU pixels available – clear rendered output so the
                // viewport shows the placeholder "No GPU output" text instead
                // of a misleading synthetic gradient.
                self.rendered_pixels = None;

                // 18. Construct and publish output frame without pixel data.
                if let Some(exported) = render_result.exported_frame {
                    let output_frame = OutputFrame::new(
                        exported.gpu_token_id,
                        exported.extent,
                        exported.timestamp_nanos,
                    );
                    self.output.publish(output_frame);
                }
            }
        }
    }

    /// Build a view matrix from orbital camera parameters (yaw, pitch, distance, pan).
    fn build_view_matrix(cam: &ViewportCamera) -> crate::asset::Mat4 {
        let yaw = cam.yaw_deg.to_radians();
        let pitch = cam.pitch_deg.to_radians();
        let (sy, cy) = (yaw.sin(), yaw.cos());
        let (sp, cp) = (pitch.sin(), pitch.cos());

        let eye_x = cam.distance * cp * sy + cam.pan[0];
        let eye_y = cam.distance * sp + cam.pan[1];
        let eye_z = cam.distance * cp * cy;

        let target = [cam.pan[0], cam.pan[1], 0.0f32];

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

        [
            [r[0], u[0], -f[0], 0.0],
            [r[1], u[1], -f[1], 0.0],
            [r[2], u[2], -f[2], 0.0],
            [
                -(r[0] * eye_x + r[1] * eye_y + r[2] * eye_z),
                -(u[0] * eye_x + u[1] * eye_y + u[2] * eye_z),
                f[0] * eye_x + f[1] * eye_y + f[2] * eye_z,
                1.0,
            ],
        ]
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
        let range_inv = 1.0 / (near - far);
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) * range_inv, -1.0],
            [0.0, 0.0, 2.0 * far * near * range_inv, 0.0],
        ]
    }

    fn build_frame_input(
        avatar: &AvatarInstance,
        fi_config: &FrameInputConfig,
        toggles: &RuntimeToggles,
        material_mode_index: usize,
    ) -> RenderFrameInput {
        let cam = &fi_config.camera;
        let lighting = &fi_config.lighting;
        let viewport_extent = fi_config.viewport_extent;
        let output_extent = fi_config.output_extent;
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

                    let material_binding = material_asset
                        .map(MaterialUploadRequest::from_asset_material)
                        .unwrap_or_else(|| {
                            if let Some(m) = avatar.asset.materials.first() {
                                MaterialUploadRequest::from_asset_material(m)
                            } else {
                                MaterialUploadRequest::default_material()
                            }
                        });

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

                    RenderMeshInstance {
                        mesh_id: mesh.id,
                        primitive_id: prim.id,
                        material_binding,
                        bounds: prim.bounds.clone(),
                        alpha_mode,
                        cull_mode,
                        outline,
                        primitive_data: Some(std::sync::Arc::new(prim.clone())),
                    }
                })
            })
            .collect();

        let cloth_deform = avatar.cloth_state.as_ref().map(|cs| ClothDeformSnapshot {
            deformed_positions: cs.deform_output.deformed_positions.clone(),
            deformed_normals: cs.deform_output.deformed_normals.clone(),
            version: cs.deform_output.version,
        });

        let render_instance = RenderAvatarInstance {
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
        };

        // Build camera matrices from the viewport camera parameters.
        let aspect = viewport_extent[0] as f32 / viewport_extent[1].max(1) as f32;
        let view = Self::build_view_matrix(cam);
        let projection = Self::build_projection_matrix(cam.fov_deg, aspect, 0.1, 1000.0);

        let yaw = cam.yaw_deg.to_radians();
        let pitch = cam.pitch_deg.to_radians();
        let (sy, cy) = (yaw.sin(), yaw.cos());
        let (sp, cp) = (pitch.sin(), pitch.cos());
        let eye_pos = [
            cam.distance * cp * sy + cam.pan[0],
            cam.distance * sp + cam.pan[1],
            cam.distance * cp * cy,
        ];

        RenderFrameInput {
            camera: CameraState {
                view,
                projection,
                position_ws: eye_pos,
                viewport_extent,
            },
            lighting: lighting.clone(),
            instances: vec![render_instance],
            output_request: OutputTargetRequest {
                preview_enabled: true,
                output_enabled: true,
                extent: output_extent,
                color_space: crate::renderer::frame_input::RenderColorSpace::Srgb,
                alpha_mode: RenderOutputAlpha::Premultiplied,
                export_mode: RenderExportMode::CpuReadback,
            },
        }
    }

    /// Per-frame input processing.
    ///
    /// Advances the animation playhead and applies any pending state that the
    /// GUI or external systems wrote into the `Application` between frames.
    fn input_update(&mut self, frame_dt: f32) {
        use crate::avatar::animation;

        if let Some(avatar) = self.avatar.as_mut() {
            // Advance animation playhead using the real clip duration.
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
            // Fallback: synchronous capture on the app thread.
            let frame = self.tracking.sample();
            println!(
                "tracking: frame {} body_conf={:.2} face_conf={:.2}",
                frame.capture_timestamp, frame.body_confidence, frame.face_confidence
            );
            self.tracking.mailbox().read()
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

    /// Like [`start_tracking`] but allows specifying the webcam resolution and fps.
    pub fn start_tracking_with_params(
        &mut self,
        backend: CameraBackend,
        width: u32,
        height: u32,
        fps: u32,
    ) {
        // If there is an existing worker that is already running, do nothing.
        if let Some(ref worker) = self.tracking_worker {
            if worker.is_running() {
                return;
            }
        }
        let shared_mailbox = self.tracking.shared_mailbox();
        let mut worker = TrackingWorker::new(shared_mailbox);
        worker.start_with_params(backend, width, height, fps);
        if worker.is_running() {
            println!("app: tracking worker started");
        } else {
            eprintln!("app: tracking worker failed to start");
        }
        self.tracking_worker = Some(worker);
    }

    /// Stop the tracking worker thread. No-op if not running.
    pub fn stop_tracking(&mut self) {
        if let Some(ref mut worker) = self.tracking_worker {
            worker.stop();
            println!("app: tracking worker stopped");
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

        println!("app: shutdown starting");

        // 1. Stop the tracking worker so no new tracking data arrives.
        if let Some(ref mut worker) = self.tracking_worker {
            println!("app: stopping tracking worker...");
            worker.stop();
            println!("app: tracking worker stopped");
        }

        // 2. Mark application as no longer running (prevents run_frame from
        //    doing further work).
        self.running = false;

        // 3. Shut down the output router, which drains remaining frames and
        //    joins the output worker thread.
        println!("app: stopping output worker...");
        self.output.shutdown();
        println!("app: output worker stopped");

        // 4. Renderer resources are released when `self.renderer` is dropped
        //    (Rust's natural drop order handles this after shutdown returns).

        self.shutdown_complete = true;
        println!("app: shutdown complete");
    }
}

impl Drop for Application {
    fn drop(&mut self) {
        if !self.shutdown_complete {
            self.shutdown();
        }
    }
}
