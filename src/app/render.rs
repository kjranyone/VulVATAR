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
use crate::simulation::SimulationStepOptions;

/// Maximum age the tracking mailbox may reach before the hold/fade
/// policy gives up and lets the avatar return to its base / animation
/// pose. The mailbox starts reporting `is_stale() == true` at
/// `TrackingMailbox::stale_timeout()` (200 ms by default); inside the
/// window `[stale_timeout, TRACKING_HOLD_WINDOW]` the last good sample
/// is reused with its confidence decayed linearly so the avatar
/// freezes and then fades back instead of snapping. Tuned for typical
/// webcam tracking hiccups (occluded face, brief out-of-frame); larger
/// values risk visible "phantom pose" persistence after the user has
/// genuinely walked away.
const TRACKING_HOLD_WINDOW: std::time::Duration = std::time::Duration::from_millis(1000);

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

        self.update_render_dt_ema(frame_dt);
        self.update_runtime_gpu_budget(std::time::Instant::now());

        // Background animation clock. Wrapped at 4096 s (≈68 min) so the f32
        // handed to the shader keeps sub-millisecond precision; the wrap is a
        // one-off pattern jump in an abstract field, accepted trade-off.
        if frame_dt.is_finite() && frame_dt > 0.0 {
            self.background_time = (self.background_time + frame_dt as f64) % 4096.0;
        }

        // 1. input update
        self.input_update(frame_dt);

        // 2. read the latest completed tracking sample from the async
        //    tracking worker. Three age states:
        //    - **Fresh** (`age <= stale_timeout`): pull a new sample.
        //    - **Holding** (stale but age < `TRACKING_HOLD_WINDOW`):
        //      reuse `last_tracking_pose` with confidence decayed
        //      linearly to zero across the hold window. The avatar
        //      freezes in place and joints drop below the solver's
        //      confidence threshold gradually as the decay proceeds,
        //      so the pose drifts back to base rather than snapping.
        //    - **Expired** (age past the hold window, or never
        //      published): drop the sample. Solver bypass restores
        //      base / animation pose.
        let (tracking_sample, sample_is_fresh, tracking_present) = if toggles.tracking_enabled {
            let mailbox = self.tracking.mailbox();
            // Single mailbox lock per frame: age + stale_timeout
            // together describe the freshness state cheaper than
            // a separate `is_stale()` call.
            let age = mailbox.age();
            let stale_threshold = mailbox.stale_timeout();
            match age {
                Some(a) if a <= stale_threshold => (self.step_tracking(), true, true),
                Some(a) if a < TRACKING_HOLD_WINDOW => {
                    let held = self.last_tracking_pose.as_ref().map(|last| {
                        let span = (TRACKING_HOLD_WINDOW - stale_threshold)
                            .as_secs_f32()
                            .max(1e-6);
                        let into_hold = a.saturating_sub(stale_threshold).as_secs_f32();
                        let progress = (into_hold / span).clamp(0.0, 1.0);
                        let scale = (1.0 - progress).clamp(0.0, 1.0);
                        let mut held = last.clone();
                        held.scale_confidence(scale);
                        held
                    });
                    (held, false, true)
                }
                Some(_) => {
                    if self.stale_warn_cooldown.elapsed() >= std::time::Duration::from_secs(5) {
                        warn!(
                            "tracking: sample expired (age > {:?}), holding base pose",
                            TRACKING_HOLD_WINDOW
                        );
                        self.stale_warn_cooldown = std::time::Instant::now();
                    }
                    (None, false, false)
                }
                // Mailbox never published a sample — same end state
                // as Expired (no pose, no warn spam yet).
                None => (None, false, false),
            }
        } else {
            (None, false, false)
        };

        // Persist only fresh samples. A held sample is derived from
        // the previous `last_tracking_pose` with its confidences
        // decayed; persisting it would compound the decay multiplier
        // every frame and collapse the effective hold window to a
        // small fraction of TRACKING_HOLD_WINDOW.
        if sample_is_fresh {
            if let Some(ref tp) = tracking_sample {
                self.last_tracking_pose = Some(tp.clone());
            }
        }

        // Global avatar fade-out when person detection is lost. Target is full
        // opacity while a person is present (fresh sample or within the hold
        // window) and zero once detection has been lost past the hold window;
        // the feature off or tracking off pins it opaque. Linear ramp over
        // FADE_DURATION so a lost/recovered subject fades out/in smoothly.
        //
        // Gated on the tracking *worker* actually running, not just the
        // GUI toggle: both `tracking.enabled` and `fade_on_tracking_loss`
        // persist in the project / last-session file, so a restart after a
        // fade-enabled tracking session restores toggle=on with no camera
        // worker. Without the worker gate that state reads as "person
        // lost" and silently fades the freshly-loaded avatar to opacity 0
        // — a blank viewport with no hint why. No camera running means
        // "nothing to lose": stay opaque until tracking actually starts.
        {
            const FADE_DURATION_S: f32 = 0.6;
            let worker_running = self
                .tracking_worker
                .as_ref()
                .is_some_and(|w| w.is_running());
            let fade_target = if config.fade_on_tracking_loss
                && toggles.tracking_enabled
                && worker_running
                && !tracking_present
            {
                0.0
            } else {
                1.0
            };
            let max_step = (frame_dt / FADE_DURATION_S).clamp(0.0, 1.0);
            let delta = (fade_target - self.tracking_fade_opacity).clamp(-max_step, max_step);
            self.tracking_fade_opacity = (self.tracking_fade_opacity + delta).clamp(0.0, 1.0);
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

        // Advance the simulation clock once per frame so every avatar in the
        // scene observes the same `(fixed_dt, substeps)`. Previously this was
        // called inside the avatar loop and the first avatar would consume the
        // accumulator, leaving later avatars with zero substeps.
        let substeps = self.sim_clock.advance(frame_dt);
        let fixed_dt = self.sim_clock.fixed_dt();

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
                        config.mouth_source,
                        &mut avatar.pose_solver_state,
                    );
                    avatar.expression_weights = new_weights;
                }
            }

            avatar.compute_global_pose();

            let step_options = SimulationStepOptions {
                spring_enabled: toggles.spring_enabled,
                cloth_enabled: toggles.cloth_enabled && avatar.cloth_enabled,
            };

            if self.physics.rapier_initialized() {
                self.physics
                    .step_all(fixed_dt, substeps, avatar, step_options);
                avatar.compute_global_pose();
            } else {
                if step_options.spring_enabled {
                    self.physics.step_springs(fixed_dt, substeps, avatar);
                    avatar.compute_global_pose();
                }
                if step_options.cloth_enabled {
                    for _ in 0..substeps {
                        self.physics.step_cloth(fixed_dt, avatar);
                    }
                    avatar.compute_global_pose();
                }
            }

            avatar.build_skinning_matrices();
        }

        if !self.avatars.is_empty() {
            let output_extent = self.output_extent.unwrap_or(self.viewport_extent);
            // `EmergencyCpu` is the budget's "GPU export is failing —
            // stop trying" mode, escalated when the export pool reports
            // repeated GpuExport failures within FAILURE_WINDOW. Once
            // there, force CpuReadback regardless of sink capability so
            // we stop generating the same failures the budget escalated
            // on. The next ProducerWaitComplete that lands resets the
            // mode back to Healthy and GpuExport resumes naturally.
            let force_cpu_export = self.runtime_gpu_budget.degraded_mode()
                == crate::app::runtime_gpu_budget::DegradedMode::EmergencyCpu;
            let export_mode = if !force_cpu_export
                && self.output.active_sink().supports_gpu_tokens()
            {
                RenderExportMode::GpuExport
            } else {
                RenderExportMode::CpuReadback
            };
            let fi_config = FrameInputConfig {
                camera: self.viewport_camera.clone(),
                lighting: self.viewport_lighting.clone(),
                viewport_extent: self.viewport_extent,
                output_extent,
                background_color: self.background_color,
                transparent_background: self.transparent_background,
                avatar_opacity: self.tracking_fade_opacity,
                output_color_space: self.output_color_space.clone(),
                output_msaa: self.output_msaa,
                bloom: self.bloom,
                generative_background: self.generative_background,
                time_seconds: self.background_time as f32,
                export_mode,
            };
            let frame_input = Self::build_frame_input_multi(
                &self.avatars,
                &fi_config,
                toggles,
                material_mode_index,
                self.viewport_background.as_deref(),
                self.ground_grid_visible,
                frame_dt,
                fixed_dt,
                substeps as u32,
            );

            if let Some(ref rt) = self.render_thread {
                if rt.submit(RenderCommand::RenderFrame(frame_input)) {
                    self.render_results_pending = self.render_results_pending.saturating_add(1);
                }
            }
        }

        // Drain results unconditionally — outside the `!avatars.is_empty()`
        // gate above and meant to be called even when the GUI is paused
        // (`drain_render_results` below is wired into `GuiApp::update`).
        // Without this the in-flight counter would pin to >0 the moment
        // the user pauses or removes the last avatar with a frame still
        // in flight, and the GUI repaint gate would loop forever.
        self.drain_render_results();
    }

    fn update_render_dt_ema(&mut self, frame_dt: f32) {
        if frame_dt <= 0.0 || !frame_dt.is_finite() {
            return;
        }
        // 5-frame EMA: enough smoothing to ignore one-off hitches without
        // letting a permanent regression hide for too long.
        let alpha = 1.0 / 5.0;
        let new_dt = std::time::Duration::from_secs_f32(frame_dt);
        let old_secs = self.render_dt_ema.as_secs_f32();
        let blended = old_secs * (1.0 - alpha) + new_dt.as_secs_f32() * alpha;
        self.render_dt_ema = std::time::Duration::from_secs_f32(blended.max(1e-6));
    }

    /// Per-frame runtime measurement intake. Reads the latest render dt,
    /// output drops/sec, and export pool occupancy; pushes them through
    /// the budget; then propagates the (possibly-clamped) render target
    /// back to `OutputRouter::set_target_fps` so the throttling gate
    /// honours the budget without any other party knowing about it.
    pub fn update_runtime_gpu_budget(&mut self, now: std::time::Instant) {
        let user_fps = self.runtime_gpu_budget.user_render_fps().max(1) as f32;
        let render_target = std::time::Duration::from_secs_f32(1.0 / user_fps);

        let elapsed = now
            .saturating_duration_since(self.last_output_drop_sample)
            .as_secs_f32();
        let current_drops = self.output.dropped_count();
        let drops_per_sec = if elapsed >= 0.5 {
            let delta = current_drops.saturating_sub(self.last_output_drop_count) as f32;
            self.last_output_drop_count = current_drops;
            self.last_output_drop_sample = now;
            (delta / elapsed).max(0.0)
        } else {
            0.0
        };

        let diagnostics = self.output.diagnostics();
        let (export_pool_leased, export_pool_capacity) = diagnostics
            .export_pool
            .map(|p| (p.leased_slots, p.capacity))
            .unwrap_or((0, 0));

        let measurements = crate::app::runtime_gpu_budget::RuntimeMeasurements {
            render_dt: self.render_dt_ema,
            render_target,
            output_drops_per_sec: drops_per_sec,
            export_pool_leased,
            export_pool_capacity,
            // Per-tick delta from `OutputRouter`'s cumulative failure
            // counter (increments whenever a frame attempted the GPU
            // handoff but published with an invalid token — renderer
            // wanted GPU, sink fell back). The budget integrates these
            // deltas into a `FAILURE_WINDOW`-wide rolling history;
            // one failure per frame still accumulates to the
            // `EmergencyCpu` threshold within seconds.
            gpu_export_failures_this_tick: self.output.take_gpu_export_failure_count(),
        };
        self.runtime_gpu_budget.update(&measurements, now);

        // Forward the clamped target. `OutputRouter::set_target_fps` is
        // idempotent so calling it every frame is cheap.
        self.output
            .set_target_fps(self.runtime_gpu_budget.render_fps_target());

        // P3-03 B4 — publish the YOLOX submit period to the tracking
        // pipeline via the shared atomic. `Rtmw3dInference` reads this
        // each frame; the cadence flip takes effect on the next
        // `frame_index.is_multiple_of(period)` check (typically next
        // submit cycle).
        //
        // Ordering::Relaxed is intentional: the value is an advisory
        // cadence knob, no other state depends on this load's freshness,
        // and a one-frame stale read on the tracking thread is
        // semantically equivalent to the budget recomputing one frame
        // later (which can happen anyway). Worst-case lag: at Healthy
        // (period=4) the tracker only checks `frame_index.is_multiple_of(period)`
        // every 4 frames, so a transition to Emergency (period=12)
        // can take up to 3 frames before the new period takes effect.
        // Acceptable given the budget's own 5s/30s dwell times.
        #[cfg(feature = "inference")]
        {
            crate::tracking::rtmw3d::YOLOX_REFRESH_PERIOD.store(
                self.runtime_gpu_budget.yolox_skip_period() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
    }

    /// Drain the latest result from the render thread's mailbox and
    /// decrement the in-flight counter for both the drained frame and
    /// any frames the render thread had to replace before this call.
    /// Safe to invoke whether or not `run_frame` ran this tick — the
    /// counter is the boundary, not the avatar list.
    pub fn drain_render_results(&mut self) {
        if let Some(ref rt) = self.render_thread {
            // Compensate the in-flight counter for frames that completed
            // on the renderer but were replaced before they reached us
            // (latest-frame mailbox semantics). Without this the counter
            // would inflate by one per dropped frame and the GUI repaint
            // gate would pin to full rate forever after even one stall.
            let dropped = rt.take_dropped_results();
            if dropped > 0 {
                self.render_results_dropped =
                    self.render_results_dropped.saturating_add(dropped);
                let dropped_u32 = u32::try_from(dropped).unwrap_or(u32::MAX);
                self.render_results_pending =
                    self.render_results_pending.saturating_sub(dropped_u32);
            }

            if let Some(result) = rt.try_recv_result() {
                self.render_results_pending = self.render_results_pending.saturating_sub(1);
                self.process_render_result(result);
            }
        }
        self.forward_completed_output_leases();
    }

    fn forward_completed_output_leases(&mut self) {
        self.pending_export_lease_releases
            .extend(self.output.drain_completed_gpu_leases());

        let Some(ref rt) = self.render_thread else {
            return;
        };

        while let Some(&lease_id) = self.pending_export_lease_releases.front() {
            if rt.submit(RenderCommand::ReleaseExportLease(lease_id)) {
                self.pending_export_lease_releases.pop_front();
            } else {
                break;
            }
        }
    }

    fn process_render_result(&mut self, render_result: crate::renderer::RenderResult) {
        self.output
            .update_export_pool_stats(render_result.stats.export_pool);
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
        output_frame.fallback_reason = exported.fallback_reason.clone();
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
            crate::renderer::output_export::ExportedPixelData::GpuFrameToken(ref gpu_token) => {
                if !self.logged_first_render_result {
                    info!(
                        "render: first result is GPU-owned {}x{}",
                        exported.extent[0], exported.extent[1],
                    );
                    self.logged_first_render_result = true;
                }
                self.rendered_extent = render_result.extent;
                self.rendered_frame_counter += 1;
                output_frame.gpu_token = Some(gpu_token.clone());
                // Feed the in-app egui preview from the side-channel CPU
                // readback the export path attached. The output consumer
                // uses `gpu_token`; the viewport pane can only display host
                // RGBA. Without this the preview texture freezes on its last
                // CPU frame whenever output runs on the GPU-token path (which
                // is the default — `FrameSink::SharedMemory` reports
                // `supports_gpu_tokens()`), so camera / pose edits appear to
                // have no effect even though the *output* is updating.
                if let Some(ref preview) = exported.preview_pixels {
                    self.rendered_pixels = Some(Arc::clone(preview));
                }
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
        _frame_dt: f32,
        fixed_dt: f32,
        substeps: u32,
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

                            // `prim` is `&Arc<MeshPrimitiveAsset>` from
                            // the asset; the per-frame snapshot keeps a
                            // refcount bump rather than deep-cloning the
                            // vertex / index / morph payload. `mi` and
                            // `pi` are kept in scope for clarity but no
                            // longer index a separate arc table.
                            let _ = (mi, pi);
                            let prim_arc = std::sync::Arc::clone(prim);

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

                // GPU cloth substeps each at `fixed_dt`, matching the
                // CPU path's `for _ in 0..substeps { step_cloth(fixed_dt) }`
                // loop. The renderer's substep loop runs verlet +
                // constraint iters that many times per frame.
                let cloth_substep_dt = fixed_dt.max(1.0 / 1000.0);
                let cloth_deforms = collect_cloth_deforms(
                    avatar
                        .cloth_state
                        .as_ref()
                        .map(|cs| (cs, avatar.cloth_sim.as_ref()))
                        .into_iter()
                        .chain(
                            avatar
                                .cloth_overlays
                                .iter()
                                .filter(|s| s.enabled)
                                .map(|s| (&s.state, Some(&s.sim))),
                        ),
                    cloth_substep_dt,
                    substeps,
                );

                RenderAvatarInstance {
                    instance_id: avatar.id,
                    world_transform: avatar.world_transform.clone(),
                    mesh_instances,
                    skinning_matrices: avatar.pose.skinning_matrices.clone(),
                    cloth_deforms,
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

        // Tracking anchors for the generative background. Bone positions are
        // the translation column of the column-major `global_transforms`
        // (avatar-root space — the same space the renderer draws vertices
        // in). Mouth-open is the max of the five VRM mouth visemes; the
        // loader canonicalises expression names to VRM 1.0, so "jawOpen"
        // never appears here.
        let background_tracking = avatars
            .first()
            .and_then(|avatar| {
                let humanoid = avatar.asset.humanoid.as_ref()?;
                let bone_pos = |bone: crate::asset::HumanoidBone| {
                    humanoid
                        .bone_map
                        .get(&bone)
                        .and_then(|node| avatar.pose.global_transforms.get(node.0 as usize))
                        .map(crate::math_utils::mat4_translation)
                };
                let head_ws = bone_pos(crate::asset::HumanoidBone::Head)?;
                let left_hand_ws =
                    bone_pos(crate::asset::HumanoidBone::LeftHand).unwrap_or(head_ws);
                let right_hand_ws =
                    bone_pos(crate::asset::HumanoidBone::RightHand).unwrap_or(head_ws);
                let mouth_open = avatar
                    .expression_weights
                    .iter()
                    .filter(|ew| matches!(ew.name.as_str(), "aa" | "ih" | "ou" | "ee" | "oh"))
                    .map(|ew| ew.weight)
                    .fold(0.0f32, f32::max);
                Some(crate::renderer::frame_input::BackgroundTracking {
                    head_ws,
                    left_hand_ws,
                    right_hand_ws,
                    mouth_open: mouth_open.clamp(0.0, 1.0),
                    valid: true,
                })
            })
            .unwrap_or_default();

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
                export_mode: fi_config.export_mode.clone(),
                msaa: fi_config.output_msaa,
            },
            background_image_path: background_image_path.map(|p| p.to_path_buf()),
            show_ground_grid,
            background_color: fi_config.background_color,
            transparent_background: fi_config.transparent_background,
            avatar_opacity: fi_config.avatar_opacity,
            bloom: fi_config.bloom,
            generative_background: fi_config.generative_background,
            background_tracking,
            time_seconds: fi_config.time_seconds,
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

/// Collect per-primitive cloth snapshots from `(ClothState, Option<ClothSimState>)` pairs.
///
/// Cloth is scoped per primitive (`target_primitive_id`), so multiple cloths
/// can coexist on one avatar as long as they target distinct primitives. When
/// two sources target the same primitive, the first wins — the call site
/// iterates `avatar.cloth_state` before `cloth_overlays`, so the authoritative
/// cloth takes precedence over any overlay variant of the same garment.
///
/// `frame_dt` is the per-frame timestep used to populate `gpu_control.dt`
/// when the cloth's `solver_backend` is `Gpu`. The renderer uses it as the
/// `dt` input to the Verlet integration shader; for CPU-backed cloths it
/// is ignored.
fn collect_cloth_deforms<'a>(
    sources: impl IntoIterator<
        Item = (
            &'a crate::avatar::instance::ClothState,
            Option<&'a crate::simulation::cloth::ClothSimState>,
        ),
    >,
    fixed_dt: f32,
    substeps: u32,
) -> Vec<ClothDeformSnapshot> {
    use crate::renderer::frame_input::{ClothGpuAttachData, ClothGpuDispatchControl};
    use crate::simulation::cloth_gpu_boundary::ClothSolverBackend;
    use crate::math_utils::vec3_scale;

    let mut seen_targets: std::collections::HashSet<crate::asset::PrimitiveId> =
        std::collections::HashSet::new();
    sources
        .into_iter()
        .filter_map(|(cs, sim_opt)| {
            let target_primitive_id = cs.target_primitive_id?;
            if !seen_targets.insert(target_primitive_id) {
                return None;
            }
            let (gpu_control, gpu_attach) =
                if cs.solver_backend == ClothSolverBackend::Gpu {
                    match sim_opt {
                        Some(sim) => {
                            let wind_force =
                                vec3_scale(&sim.wind_direction, sim.wind_response);
                            let ctrl = ClothGpuDispatchControl {
                                dt: fixed_dt,
                                substeps,
                                damping: sim.damping,
                                gravity: sim.gravity,
                                wind_force,
                                solver_iterations: sim.solver_iterations as u32,
                            };
                            let attach = ClothGpuAttachData {
                                constraints: sim
                                    .distance_constraints
                                    .iter()
                                    .map(|c| {
                                        (
                                            c.a as u32,
                                            c.b as u32,
                                            c.rest_length,
                                            c.stiffness,
                                        )
                                    })
                                    .collect(),
                                triangle_indices: sim.triangle_indices.clone(),
                                inv_masses: sim
                                    .particles
                                    .iter()
                                    .map(|p| p.inv_mass)
                                    .collect(),
                                pinned: sim.particles.iter().map(|p| p.pinned).collect(),
                            };
                            (Some(ctrl), Some(attach))
                        }
                        None => (None, None),
                    }
                } else {
                    (None, None)
                };
            Some(ClothDeformSnapshot {
                target_primitive_id,
                target_mesh_id: cs.target_mesh_id,
                vertex_offset: cs.target_vertex_offset,
                vertex_count: cs.target_vertex_count,
                deformed_positions: cs.deform_output.deformed_positions.clone(),
                deformed_normals: cs.deform_output.deformed_normals.clone(),
                version: cs.deform_output.version,
                solver_backend: cs.solver_backend,
                gpu_control,
                gpu_attach,
            })
        })
        .collect()
}

#[cfg(test)]
mod cloth_collection_tests {
    use super::*;
    use crate::asset::{ClothOverlayId, MeshId, PrimitiveId};
    use crate::avatar::instance::{
        ClothCollisionRuntimeCache, ClothConstraintRuntimeCache, ClothDeformOutput, ClothState,
    };

    fn make_cloth_state(
        overlay_id: u64,
        target_primitive_id: Option<PrimitiveId>,
        vertex_count: u32,
        version: u64,
    ) -> ClothState {
        ClothState {
            overlay_id: ClothOverlayId(overlay_id),
            enabled: true,
            sim_positions: vec![],
            prev_sim_positions: vec![],
            sim_normals: vec![],
            constraint_cache: ClothConstraintRuntimeCache {
                active_constraint_count: 0,
            },
            collision_cache: ClothCollisionRuntimeCache {
                active_collision_count: 0,
            },
            deform_output: ClothDeformOutput {
                deformed_positions: vec![[0.0, 0.0, 0.0]; vertex_count as usize],
                deformed_normals: None,
                version,
            },
            target_primitive_id,
            target_mesh_id: target_primitive_id.map(|p| MeshId(p.0)),
            target_vertex_offset: 0,
            target_vertex_count: vertex_count,
            solver_backend:
                crate::simulation::cloth_gpu_boundary::ClothSolverBackend::Cpu,
        }
    }

    #[test]
    fn multi_cloth_targeting_distinct_primitives_all_survive() {
        let body = make_cloth_state(1, Some(PrimitiveId(10)), 32, 1);
        let skirt = make_cloth_state(2, Some(PrimitiveId(20)), 64, 1);
        let scarf = make_cloth_state(3, Some(PrimitiveId(30)), 16, 1);

        let result =
            collect_cloth_deforms([(&body, None), (&skirt, None), (&scarf, None)], 1.0 / 60.0, 1);

        assert_eq!(result.len(), 3, "all three distinct-target cloths must be kept");
        let target_ids: Vec<u64> =
            result.iter().map(|c| c.target_primitive_id.0).collect();
        assert_eq!(target_ids, vec![10, 20, 30]);
        let vertex_counts: Vec<u32> = result.iter().map(|c| c.vertex_count).collect();
        assert_eq!(vertex_counts, vec![32, 64, 16]);
    }

    #[test]
    fn duplicate_target_dedups_with_first_wins() {
        let authoritative = make_cloth_state(1, Some(PrimitiveId(10)), 32, 5);
        let overlay_duplicate = make_cloth_state(2, Some(PrimitiveId(10)), 64, 99);

        let result =
            collect_cloth_deforms([(&authoritative, None), (&overlay_duplicate, None)], 1.0 / 60.0, 1);

        assert_eq!(result.len(), 1, "duplicate target_primitive_id must dedup");
        assert_eq!(result[0].version, 5, "first (authoritative) cloth wins");
        assert_eq!(result[0].vertex_count, 32);
    }

    #[test]
    fn unbound_cloth_without_target_primitive_is_skipped() {
        let bound = make_cloth_state(1, Some(PrimitiveId(10)), 32, 1);
        let unbound = make_cloth_state(2, None, 64, 1);

        let result =
            collect_cloth_deforms([(&bound, None), (&unbound, None)], 1.0 / 60.0, 1);

        assert_eq!(result.len(), 1, "cloth with no render target is dropped");
        assert_eq!(result[0].target_primitive_id.0, 10);
    }
}
