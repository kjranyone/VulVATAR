//! Per-frame render orchestration: `run_frame` plus its direct helpers
//! (`process_render_result`, `build_frame_input_multi`, the camera /
//! projection math, `input_update`, `step_tracking`).

use log::{info, warn};
use std::sync::Arc;

use super::{Application, FrameConfig, FrameInputConfig, RuntimeToggles, ViewportCamera};
use crate::app::render_thread::RenderCommand;
use crate::avatar::pose_solver::{self, SolverParams};
use crate::avatar::AvatarInstance;
use crate::output::OutputFrame;
use crate::renderer::frame_input::RenderDebugFlags;
use crate::renderer::frame_input::{
    CameraState, ClothDeformSnapshot, OutlineSnapshot, OutputTargetRequest, RenderAlphaMode,
    RenderAvatarInstance, RenderCullMode, RenderExportMode, RenderFrameInput, RenderMeshInstance,
    RenderOutputAlpha,
};
use crate::renderer::material::MaterialShaderMode;
use crate::renderer::material::MaterialUploadRequest;

impl Application {
    pub fn run_frame(&mut self, config: &FrameConfig) {
        if !self.running {
            return;
        }

        let toggles = &config.toggles;
        let smoothing_params = &config.smoothing;
        let hand_tracking_enabled = config.hand_tracking_enabled;
        let face_tracking_enabled = config.face_tracking_enabled;
        let lower_body_tracking_enabled = config.lower_body_tracking_enabled;
        let root_translation_enabled = config.root_translation_enabled;
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

        let solver_params = SolverParams {
            rotation_blend: smoothing_params.rotation_blend,
            joint_confidence_threshold: smoothing_params.joint_confidence_threshold,
            face_confidence_threshold: smoothing_params.face_confidence_threshold,
            hand_tracking_enabled,
            face_tracking_enabled,
            lower_body_tracking_enabled,
            root_translation_enabled,
            // Pose calibration captured via the `Calibrate Pose ▼` modal
            // (`docs/calibration-ux.md`). The solver uses it to seed the
            // root-translation EMA reference so the avatar's neutral
            // position is the user's calibrated stance, not whatever the
            // first hip-visible frame happened to read.
            pose_calibration: self.tracking_calibration.pose.clone(),
            ..SolverParams::default()
        };

        for avatar in self.avatars.iter_mut() {
            avatar.build_base_pose();

            if let Some(ref mut source) = tracking_sample.clone() {
                self.tracking_calibration.apply_calibration(source);

                let humanoid = avatar.asset.humanoid.as_ref();
                pose_solver::solve_avatar_pose(
                    source,
                    &avatar.asset.skeleton,
                    humanoid,
                    &mut avatar.pose.local_transforms,
                    &solver_params,
                    &mut avatar.pose_solver_state,
                );

                if face_tracking_enabled {
                    let new_weights = pose_solver::solve_expressions(
                        source,
                        &avatar.asset.default_expressions,
                        Some(&avatar.expression_weights),
                        smoothing_params.expression_blend,
                        smoothing_params.face_confidence_threshold,
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
                background_color: self.background_color,
                transparent_background: self.transparent_background,
                output_color_space: self.output_color_space.clone(),
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
        // Phase B-4: tag the frame with the user's alpha preference so the
        // downstream sink (Win32FileBackedSharedMemorySink → DLL) knows
        // whether to preserve or clobber the alpha channel.
        output_frame.alpha_mode = if self.output_preserve_alpha {
            crate::output::AlphaMode::Premultiplied
        } else {
            crate::output::AlphaMode::Opaque
        };
        // Carry the renderer's output colour space through to the OutputFrame
        // so the output worker (and its sinks) see what the user picked
        // rather than the default Srgb that `OutputFrame::new` stamps in.
        output_frame.color_space = match exported.export_metadata.color_space {
            crate::renderer::frame_input::RenderColorSpace::Srgb => {
                crate::output::OutputColorSpace::Srgb
            }
            crate::renderer::frame_input::RenderColorSpace::LinearSrgb => {
                crate::output::OutputColorSpace::LinearSrgb
            }
        };

        match exported.pixel_data {
            crate::renderer::output_export::ExportedPixelData::CpuReadback(ref pixel_data)
                if !pixel_data.is_empty() =>
            {
                if !self.logged_first_render_result {
                    info!(
                        "render: first CPU readback result {}x{} bytes={}",
                        exported.extent[0],
                        exported.extent[1],
                        pixel_data.len(),
                    );
                    self.logged_first_render_result = true;
                }
                self.rendered_pixels = Some(Arc::clone(pixel_data));
                self.rendered_extent = render_result.extent;
                self.rendered_frame_counter += 1;
                output_frame.pixel_data = Some(Arc::clone(pixel_data));
            }
            crate::renderer::output_export::ExportedPixelData::GpuOwned => {
                if !self.logged_first_render_result {
                    info!(
                        "render: first result is GPU-owned {}x{}",
                        exported.extent[0], exported.extent[1],
                    );
                    self.logged_first_render_result = true;
                }
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
    pub(crate) fn build_view_matrix(cam: &ViewportCamera) -> (crate::asset::Mat4, [f32; 3]) {
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
    pub(crate) fn build_projection_matrix(
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
                    .enumerate()
                    .flat_map(|(mi, mesh)| {
                        mesh.primitives.iter().enumerate().map(move |(pi, prim)| {
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
                                            if let Some(&mesh_idx) =
                                                avatar.asset.node_to_mesh.get(&bind.node_index)
                                            {
                                                if let Some(m) = avatar.asset.meshes.get(mesh_idx) {
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

                            let prim_arc = avatar
                                .primitive_arcs
                                .get(mi)
                                .and_then(|v| v.get(pi))
                                .cloned()
                                .unwrap_or_else(|| std::sync::Arc::new(prim.clone()));

                            RenderMeshInstance {
                                mesh_id: mesh.id,
                                primitive_id: prim.id,
                                material_binding,
                                bounds: prim.bounds,
                                alpha_mode,
                                cull_mode,
                                outline,
                                primitive_data: Some(prim_arc),
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
                color_space: fi_config.output_color_space.clone(),
                alpha_mode: RenderOutputAlpha::Premultiplied,
                export_mode: RenderExportMode::CpuReadback,
            },
            background_image_path: background_image_path.map(|p| p.to_path_buf()),
            show_ground_grid,
            background_color: fi_config.background_color,
            transparent_background: fi_config.transparent_background,
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

    fn step_tracking(&mut self) -> Option<crate::tracking::SourceSkeleton> {
        // When the tracking worker is running, read the latest pose from the
        // shared mailbox (non-blocking). The worker thread is producing poses
        // independently, so we just grab whatever is newest.
        let worker_running = self
            .tracking_worker
            .as_ref()
            .is_some_and(|w| w.is_running());

        if worker_running {
            // Non-blocking read from the shared mailbox populated by the worker.
            self.tracking.mailbox().latest_pose()
        } else {
            // No worker running — return None so tracking is skipped.
            // (The old synchronous fallback generated synthetic data on every
            // frame and spammed info logs at 60 fps.)
            None
        }
    }
}
