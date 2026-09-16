//! Per-frame render orchestration: `run_frame` plus its direct helpers
//! (`process_render_result`, `build_frame_input_multi`, the camera /
//! projection math, `input_update`, `step_tracking`).

use log::{info, warn};
use std::sync::Arc;

use super::{
    Application, FrameConfig, FrameInputConfig, RuntimeToggles, SensorCamera, ViewportCamera,
};
use crate::app::render_thread::RenderCommand;
use crate::avatar::AvatarInstance;
use crate::output::OutputFrame;
use crate::renderer::frame_input::RenderDebugFlags;
use crate::renderer::frame_input::{
    BodySdfPlan, CameraState, ClothDeformSnapshot, OutputTargetRequest, RenderAvatarInstance,
    RenderExportMode, RenderFrameInput, RenderMeshInstance, RenderOutputAlpha,
};
use crate::renderer::material::MaterialShaderMode;
use crate::simulation::SimulationStepOptions;

/// Maximum age the tracking mailbox may reach before the hold/fade
/// policy gives up and lets the avatar return to its base / animation
/// pose. The mailbox starts reporting `is_stale() == true` at
/// `TrackingMailbox::stale_timeout()` (200 ms by default); inside the
/// window `[stale_timeout, TRACKING_HOLD_WINDOW]` the last good sample
/// is reused with its confidence decayed linearly so the avatar
/// freezes and then fades back instead of snapping. Tuned for typical
/// tracking hiccups (occluded face, brief out-of-frame); larger
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
        //      published): drop the sample. The avatar then eases into
        //      the procedural A-pose rest (avatar::relax) rather than
        //      snapping to the bind pose.
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

        // Display-rate pose interpolation. The estimator publishes at its
        // compute-bound cadence (~20-30 Hz); replaying the newest sample
        // as-is rendered the motion as a staircase — frozen for the full
        // inter-sample interval, then a multi-millimetre jump when the
        // next sample landed. Ingest the fresh sample, then hand the
        // avatar loop a slerped rig instead. `last_tracking_pose` above
        // keeps the raw sample: hold/fade, viewport overlay and status
        // bar read diagnostics, which must stay raw.
        let mut tracking_sample = tracking_sample;
        if sample_is_fresh {
            if let Some(s) = tracking_sample.as_mut() {
                if let Some(rig) = s.rig.clone() {
                    self.pose_timeline.ingest(&rig);
                    if smoothing_params.pose_interp_enabled {
                        s.rig = Some(Arc::new(self.pose_timeline.sample_at(
                            crate::avatar::pose_timeline::PoseInterpParams {
                                delay_frac: smoothing_params.pose_interp_delay_frac,
                                ..Default::default()
                            },
                            std::time::Instant::now(),
                        )));
                    }
                }
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

        let retarget_params = crate::avatar::retarget::RetargetParams {
            rotation_blend: smoothing_params.rotation_blend,
            root_translation_enabled,
            hand_tracking_enabled,
            lower_body_tracking_enabled,
            ..Default::default()
        };

        // Advance the simulation clock once per frame so every avatar in the
        // scene observes the same `(fixed_dt, substeps)`. Previously this was
        // called inside the avatar loop and the first avatar would consume the
        // accumulator, leaving later avatars with zero substeps.
        let substeps = self.sim_clock.advance(frame_dt);
        let fixed_dt = self.sim_clock.fixed_dt();
        // Surfaced for the live debug heartbeat (`debug_gui.json`) so a
        // single-frame visual glitch can be correlated with a zero-substep
        // frame instead of being inferred from frame pacing.
        self.last_sim_substeps = substeps;

        // Scene colliders are constant for the whole frame; resolve once
        // for the cloth settle-sleep gate and the snapshot collector
        // instead of once per use site.
        let scene_colliders = self.physics.resolved_scene_colliders();

        for (avatar_idx, avatar) in self.avatars.iter_mut().enumerate() {
            // Costume-health probe payload (R5): captured PRE-physics
            // inside the `avatar_idx == 0` block below, completed with
            // the POST-physics stage after the solver ran, and dumped
            // once at the end of this avatar iteration.
            let mut probe: Option<(
                Vec<(String, [f32; 3])>,
                Vec<crate::asset::PrimitiveId>,
                Vec<crate::tracking::debug_channel::CostumePrimProbe>,
            )> = None;

            // Rest relax: while a driver (tracking or an animation clip)
            // produces the pose, the relax state stays armed; the first
            // undriven frame captures the outgoing pose so later frames
            // ease into the procedural A-pose rest (avatar::relax)
            // instead of snapping to the T-pose bind in one frame.
            // Capture must run before build_base_pose resets the locals.
            let animation_active = avatar.animation_state.active_clip.is_some();
            let pose_driven = tracking_sample.as_ref().is_some_and(|s| s.rig.is_some())
                && avatar.asset.humanoid.is_some();
            if pose_driven || animation_active {
                avatar.relax_mark_driven();
            } else {
                avatar.relax_capture();
            }

            avatar.build_base_pose();

            if let Some(ref mut source) = tracking_sample.clone() {
                let humanoid = avatar.asset.humanoid.as_ref();
                if let (Some(rig), Some(hm)) = (source.rig.as_ref(), humanoid) {
                    // Tracking v2: joint rotations from the fusion estimator.
                    crate::avatar::retarget::apply_rig_pose(
                        rig,
                        &avatar.asset.skeleton,
                        hm,
                        &mut avatar.pose.local_transforms,
                        &retarget_params,
                        &mut avatar.retarget_state,
                        frame_dt,
                    );
                }

                if face_tracking_enabled {
                    let new_weights = crate::avatar::expressions::solve_expressions(
                        source,
                        &avatar.asset.default_expressions,
                        Some(&avatar.expression_weights),
                        smoothing_params.expression_blend,
                        smoothing_params.face_confidence_threshold,
                        config.mouth_source,
                        &mut avatar.expression_state,
                    );
                    avatar.expression_weights = new_weights;
                }
            } else if !animation_active {
                avatar.relax_apply(frame_dt);
            }

            avatar.compute_global_pose();

            // Live debug: publish the primary avatar's solved joint world
            // positions so the external overlay can draw the avatar skeleton
            // without a GPU render (torso tilt, elbow placement, whole-body
            // rotation). No-op unless %ProgramData%\VulVATAR\debug.on exists.
            if avatar_idx == 0 {
                let humanoid = avatar.asset.humanoid.as_ref();
                let gt = &avatar.pose.global_transforms;
                // Head bone's world basis (normalised X/Y/Z columns of its
                // column-major global matrix) = its facing, so the external tool
                // can see an over-pitched head directly.
                let head_axes = humanoid
                    .and_then(|h| h.bone_map.get(&crate::asset::HumanoidBone::Head))
                    .map(|n| n.0 as usize)
                    .and_then(|i| gt.get(i))
                    .map(|m| {
                        let norm = |v: [f32; 3]| {
                            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
                            [v[0] / l, v[1] / l, v[2] / l]
                        };
                        [
                            norm([m[0][0], m[0][1], m[0][2]]),
                            norm([m[1][0], m[1][1], m[1][2]]),
                            norm([m[2][0], m[2][1], m[2][2]]),
                        ]
                    });
                crate::tracking::debug_channel::dump_avatar_pose(
                    |b| {
                        humanoid
                            .and_then(|h| h.bone_map.get(&b))
                            .map(|n| n.0 as usize)
                            .and_then(|i| gt.get(i))
                            .map(|m| [m[3][0], m[3][1], m[3][2]])
                    },
                    head_axes,
                );
                // Costume-health probe, PRE-physics stage (R5). Every
                // cloth-target primitive is probed by id (not by a
                // hardcoded mesh name) so a chest/hem breakdown can be
                // attributed to a specific primitive + material. The
                // POST-physics stage is measured after the solver ran —
                // see the dump call at the end of this avatar iteration.
                if crate::tracking::debug_channel::enabled() {
                    let mut cloth_targets: Vec<crate::asset::PrimitiveId> = avatar
                        .cloth_overlays
                        .iter()
                        .filter(|s| s.enabled)
                        .filter_map(|s| s.state.target_primitive_id)
                        .collect();
                    if let Some(cs) = avatar.cloth_state.as_ref() {
                        if let Some(t) = cs.target_primitive_id {
                            cloth_targets.push(t);
                        }
                    }
                    // Refresh the skinning matrices against the
                    // pre-physics global pose before the CPU LBS probe.
                    avatar.build_skinning_matrices();
                    let prim_probes = cloth_targets
                        .iter()
                        .map(|t| {
                            let (clearance_parent, containment_parent) =
                                prim_anchor_parents(avatar, *t);
                            match cpu_lbs_prim_bbox(avatar, *t) {
                                Some((mesh, lo, hi)) => {
                                    crate::tracking::debug_channel::CostumePrimProbe {
                                        mesh,
                                        primitive: t.0,
                                        clearance_parent,
                                        containment_parent,
                                        pre_physics: Some(
                                            crate::tracking::debug_channel::CostumeBBox {
                                                min: lo,
                                                max: hi,
                                            },
                                        ),
                                        post_physics: None,
                                    }
                                }
                                None => crate::tracking::debug_channel::CostumePrimProbe {
                                    mesh: "<missing vertex data>".to_string(),
                                    primitive: t.0,
                                    clearance_parent,
                                    containment_parent,
                                    pre_physics: None,
                                    post_physics: None,
                                },
                            }
                        })
                        .collect();
                    probe = Some((costume_bone_positions(avatar), cloth_targets, prim_probes));
                }
            }

            let step_options = SimulationStepOptions {
                spring_enabled: toggles.spring_enabled,
                cloth_enabled: toggles.cloth_enabled && avatar.cloth_enabled,
            };

            // Zero-substep frames skip the spring solver entirely (both
            // the Rapier `step_all` early-return and the plain loop), but
            // `build_base_pose` above has already reset the spring-driven
            // joints to their rest rotations. Restore the last solved
            // rotations so the hair does not render one frame in the rest
            // pose — the single-frame clip into the head or body. On
            // stepped frames the solver's writeback overwrites these
            // anyway, so this path changes nothing there.
            if step_options.spring_enabled && substeps == 0 {
                avatar.reapply_spring_rotations();
                avatar.compute_global_pose();
            }

            // The body distance field is an Arc in the avatar — clone
            // the handle out so the solvers can borrow it while `avatar`
            // is mutably borrowed.
            let avatar_body_sdf = avatar.body_sdf.clone();
            if self.physics.rapier_initialized() {
                self.physics.step_all(
                    fixed_dt,
                    substeps,
                    avatar,
                    step_options,
                    &config.spring_tuning,
                    &config.scene_gravity,
                    avatar_body_sdf.as_ref(),
                );
                avatar.compute_global_pose();
            } else {
                if step_options.spring_enabled {
                    self.physics.step_springs(
                        fixed_dt,
                        substeps,
                        avatar,
                        &config.spring_tuning,
                        &config.scene_gravity,
                        avatar_body_sdf.as_ref(),
                    );
                    avatar.compute_global_pose();
                }
                if step_options.cloth_enabled {
                    for _ in 0..substeps {
                        self.physics
                            .step_cloth(fixed_dt, avatar, &config.scene_gravity);
                    }
                    avatar.compute_global_pose();
                }
            }

            avatar.build_skinning_matrices();

            // Settle-sleep gate for cloth (idle z-fight campaign,
            // 2026-09-15): fingerprint every solver input for each
            // GPU-backed cloth slot and, when the sim has gone quiet
            // (quiet streak fed from the position readback), suppress
            // this frame's dispatch — the snapshot's `substeps` is
            // forced to 0 and the renderer's frozen-frame contract
            // keeps `cloth_pos_ssbo` bit-stable. Must run before
            // `build_frame_input_multi` consumes `suppress_dispatch`.
            update_cloth_settle_gate(
                avatar,
                fixed_dt,
                step_options.cloth_enabled,
                &scene_colliders,
            );

            // Costume-health probe, POST-physics stage (R5): the same
            // cloth-target primitives re-probed against the post-solver
            // skinning matrices, then one dump carrying both stages.
            // On a frozen frame (substeps == 0) the two stages match BY
            // CONTRACT — the JSON carries `sim_substeps` so the reader
            // can tell "physics didn't run" from "physics ran and did
            // nothing".
            if let Some((bones_pre, cloth_targets, mut prim_probes)) = probe.take() {
                for p in prim_probes.iter_mut() {
                    if let Some((_, lo, hi)) =
                        cpu_lbs_prim_bbox(avatar, crate::asset::PrimitiveId(p.primitive))
                    {
                        p.post_physics = Some(crate::tracking::debug_channel::CostumeBBox {
                            min: lo,
                            max: hi,
                        });
                    }
                }
                crate::tracking::debug_channel::dump_costume_probe(
                    bones_pre,
                    costume_bone_positions(avatar),
                    prim_probes,
                    cloth_targets.len(),
                    cloth_targets,
                    substeps,
                    fixed_dt,
                );
            }
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
            let export_mode =
                if !force_cpu_export && self.output.active_sink().supports_gpu_tokens() {
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
                // 1:1 mirror camera: only when the toggle is on AND the live
                // pose is metric-native (carries the D435 intrinsics). When
                // `metric_frame_info` is `None` this stays `None` and the
                // free orbit camera is used — the toggle is a no-op there.
                sensor_camera: if toggles.mirror_view {
                    self.last_tracking_pose
                        .as_ref()
                        .and_then(|p| p.metric_frame_info.as_ref())
                        .map(|m| SensorCamera {
                            intrinsics: m.intrinsics,
                            // `anchor_cam_m[2]` is raw camera z (forward
                            // distance, positive). The eye sits this far in
                            // front of the avatar so it frames at sensor scale.
                            anchor_depth_m: m.anchor_cam_m[2].abs(),
                        })
                } else {
                    None
                },
            };
            // Body-SDF splat skip + cadence gate: the field is a pure
            // function of the posed body, so when the pose key is
            // unchanged (no tracking sample, no clip, hair settled) the
            // previous field is still exact — drop the plan and let the
            // renderer keep last frame's slot instead of re-running the
            // sentinel fill + splat dispatches (hundreds of ms of GPU
            // time on current models). While the pose *is* changing,
            // splat at most every `SDF_SPLAT_MIN_INTERVAL` — 30 Hz
            // matches the tracking cadence, so no field information is
            // ever lost; the intervening frames reuse the ~33 ms-stale
            // field, the same latency class the pipeline already has
            // (one frame stale by construction). The app-side
            // `AvatarInstance::body_sdf` keeps the last readback
            // (`apply_sdf_readback` never clears), so the spring solver
            // is unaffected.
            const SDF_SPLAT_MIN_INTERVAL: std::time::Duration =
                std::time::Duration::from_millis(33);
            let mut sdf_splat_changed = std::collections::HashSet::new();
            for avatar in &self.avatars {
                // Garment splat list is a pure function of the asset —
                // derive once per instance (the weight-mass scan is not
                // per-frame material).
                self.sdf_garment_prims
                    .entry(avatar.id.0)
                    .or_insert_with(|| {
                        crate::asset::clearance::sdf_garment_splat_prims(
                            &avatar.asset,
                            Self::sdf_garment_cap(),
                        )
                    });
                let spring_driven: std::collections::HashSet<usize> = avatar
                    .asset
                    .spring_bones
                    .iter()
                    .flat_map(|sb| {
                        std::iter::once(sb.chain_root.0 as usize)
                            .chain(sb.joints.iter().map(|&n| n.0 as usize))
                    })
                    .collect();
                let key = Self::splat_pose_key(
                    &avatar.pose.local_transforms,
                    &spring_driven,
                    &avatar.expression_weights,
                );
                let changed = self.sdf_splat_pose_keys.get(&avatar.id.0) != Some(&key);
                let due = self
                    .sdf_splat_last
                    .get(&avatar.id.0)
                    .map_or(true, |t| t.elapsed() >= SDF_SPLAT_MIN_INTERVAL);
                // Only fold the key in when the splat actually runs;
                // otherwise a skipped frame would swallow the change and
                // the field would never refresh.
                if changed && due {
                    sdf_splat_changed.insert(avatar.id.0);
                    self.sdf_splat_pose_keys.insert(avatar.id.0, key);
                    self.sdf_splat_last.insert(avatar.id.0, std::time::Instant::now());
                }
            }
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
                &scene_colliders,
                &sdf_splat_changed,
                &self.sdf_garment_prims,
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
            // Render thread's own production rate. Under render-thread
            // starvation the GUI keeps ticking (the repaint gate spins
            // on `render_results_pending`), so `render_dt` stays small
            // and only this measurement sees the shortfall.
            render_fps: self.render_thread_fps(),
        };
        self.runtime_gpu_budget.update(&measurements, now);

        // Forward the clamped target. `OutputRouter::set_target_fps` is
        // idempotent so calling it every frame is cheap.
        self.output
            .set_target_fps(self.runtime_gpu_budget.render_fps_target());

        // FaceMesh CPU-EP preference under pressure (takes effect on the
        // next tracking start — the ONNX session is built there).
        #[cfg(feature = "inference")]
        {
            crate::tracking::face_mediapipe::FACEMESH_EP_CPU.store(
                self.runtime_gpu_budget.facemesh_prefers_cpu_ep(),
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        // Pose Hz + depth refresh are consumed by the realsense
        // worker's estimate loop (worker::POSE_HZ_TARGET /
        // worker::DEPTH_REFRESH_PERIOD). Same Relaxed rationale.
#[cfg(feature = "realsense")]
        {
            crate::tracking::worker::POSE_HZ_TARGET.store(
                self.runtime_gpu_budget.pose_hz_target(),
                std::sync::atomic::Ordering::Relaxed,
            );
            crate::tracking::worker::DEPTH_REFRESH_PERIOD.store(
                self.runtime_gpu_budget.depth_refresh_period() as u64,
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
                self.render_results_dropped = self.render_results_dropped.saturating_add(dropped);
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

    /// Write GPU cloth readback entries into the matching
    /// `ClothState`s (primary first, then overlay slots — the same
    /// first-wins-by-primitive ordering the snapshot collector uses).
    ///
    /// Delivery contract (R4): an entry is applied ONLY where ALL of
    /// `instance_id`, `mesh_id`, and `primitive_id` match the
    /// avatar's own identity and the cloth's resolved render target.
    /// `primitive_id` is unique within one avatar, not across
    /// avatars, so the old primitive-only match mis-delivered one
    /// avatar's solved state into another's `ClothState`. Rows with
    /// no instance stamp (a slot that never saw a GPU dispatch) are
    /// never applied.
    fn apply_cloth_readback(&mut self, entries: &[crate::renderer::ClothReadback]) {
        for avatar in self.avatars.iter_mut() {
            for entry in entries {
                let apply = |cs: &mut crate::avatar::instance::ClothState| {
                    // Settle-sleep accounting first: this row is the
                    // delta baseline for the quiet streak. Runs even on
                    // suppressed (frozen) slots — identical bytes hold
                    // the streak; real motion breaks it and the gate
                    // stops suppressing on the next frame.
                    fold_cloth_readback_settle(cs, &entry.positions);
                    cs.sim_positions = entry.positions.clone();
                    cs.prev_sim_positions = entry.positions.clone();
                    if let Some(n) = entry.normals.as_ref() {
                        cs.sim_normals = n.clone();
                    }
                    cs.deform_output.deformed_positions = entry.positions.clone();
                    cs.deform_output.deformed_normals = entry.normals.clone();
                    cs.deform_output.version = entry.version as u64;
                };
                // Instance + mesh must match this avatar AND the cloth
                // must be GPU-backed and bound to this entry's primitive.
                let targets_entry =
                    |cs: &crate::avatar::instance::ClothState| {
                        cloth_readback_matches(entry, avatar.id, cs)
                    };
                if avatar
                    .cloth_state
                    .as_ref()
                    .is_some_and(|cs| targets_entry(cs))
                {
                    if let Some(cs) = avatar.cloth_state.as_mut() {
                        apply(cs);
                    }
                    continue;
                }
                for slot in avatar.cloth_overlays.iter_mut() {
                    if targets_entry(&slot.state) {
                        apply(&mut slot.state);
                    }
                }
            }
        }
    }

    /// Fold the body-SDF readback rows into the matching avatar
    /// instances so the spring solver picks the field up on the next
    /// `run_frame`. Delivery matches on `AvatarInstanceId` — an entry
    /// for an instance that no longer exists is dropped.
    fn apply_sdf_readback(&mut self, fields: &[crate::renderer::sdf_field::SdfFieldReadback]) {
        if fields.is_empty() {
            return;
        }
        for avatar in self.avatars.iter_mut() {
            for entry in fields {
                if entry.instance_id == avatar.id.0 {
                    avatar.body_sdf = Some(crate::simulation::sdf::SdfField::new(
                        entry.grid,
                        entry.data.clone(),
                    ));
                    break;
                }
            }
        }
    }

    fn process_render_result(&mut self, render_result: crate::renderer::RenderResult) {
        self.output
            .update_export_pool_stats(render_result.stats.export_pool);
        // Fold the GPU cloth readback into the avatar-side ClothState so
        // CPU-side consumers (cloth inspector live view, backend flips to
        // CPU) see the live solver state instead of the attach-time rest
        // pose. Must run before the exported-frame early-return: token
        // frames carry the readback too.
        if !render_result.cloth_readback.is_empty() {
            self.apply_cloth_readback(&render_result.cloth_readback);
            self.apply_sdf_readback(&render_result.sdf_fields);
        }
        // R2 final-VBO audit (only rows when `VULVATAR_VBO_AUDIT=1`):
        // publish the render-side correction telemetry for the external
        // watcher before the export early-return — token frames carry
        // the audit too.
        if !render_result.vbo_audit.is_empty() {
            crate::tracking::debug_channel::dump_vbo_audit(
                render_result
                    .vbo_audit
                    .iter()
                    .map(|e| crate::tracking::debug_channel::VboAuditRow {
                        mesh: e.mesh_id.0,
                        primitive: e.primitive_id.0,
                        instance: e.instance_id,
                        vertex_count: e.vertex_count,
                        nan_count: e.nan_count,
                        max_correction_m: e.max_correction_m,
                        p95_correction_m: e.p95_correction_m,
                        max_pos_len_m: e.max_pos_len_m,
                    })
                    .collect(),
                render_result.timestamp_nanos,
            );
        }
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

    /// Perspective projection from pinhole camera intrinsics (fx, fy, cx, cy,
    /// width, height). Same row-major layout, Vulkan `[0,1]` depth mapping and
    /// y-flip as [`Self::build_projection_matrix`], but the FOV comes from
    /// `fx`/`fy` and the principal point `(cx, cy)` makes the frustum
    /// asymmetric so the optical axis lands at `(cx, cy)` rather than the image
    /// centre. Reduces exactly to the symmetric form when `cx = w/2, cy = h/2`.
    ///
    /// Derivation: a view-space point `(x, y, z)` (z < 0 in front) projects to
    /// pixel `u = -fx·x/z + cx`, `v =  fy·y/z + cy` (camera y-up, image v-down).
    /// With `w_clip = -z`, matching `u = (ndc_x·0.5 + 0.5)·w` gives
    /// `P[0][0] = 2fx/w`, `P[0][2] = 1 − 2cx/w`; likewise for y.
    pub(crate) fn build_projection_from_intrinsics(
        intr: &crate::tracking::CameraIntrinsics,
        near: f32,
        far: f32,
    ) -> crate::asset::Mat4 {
        let w = intr.width.max(1) as f32;
        let h = intr.height.max(1) as f32;
        let a = far / (near - far);
        let b = far * near / (near - far);
        [
            [2.0 * intr.fx / w, 0.0, 1.0 - 2.0 * intr.cx / w, 0.0],
            [0.0, -2.0 * intr.fy / h, 1.0 - 2.0 * intr.cy / h, 0.0],
            [0.0, 0.0, a, b],
            [0.0, 0.0, -1.0, 0.0],
        ]
    }

    /// View matrix for the 1:1 sensor mirror: eye straight in front of
    /// `target` (`+Z`) at `depth` metres, looking back along `-Z` with `+Y`
    /// up — the same handedness as the orbit camera at yaw 0, so the source
    /// skeleton's selfie x-negation is NOT doubled into a re-mirror. `depth`
    /// is clamped to a small positive so a zero / garbage anchor can't drop
    /// the eye onto the subject. With no rotation the view is a pure
    /// translation by `-eye`.
    pub(crate) fn build_sensor_view_matrix(
        target: [f32; 3],
        depth: f32,
    ) -> (crate::asset::Mat4, [f32; 3]) {
        let eye = [target[0], target[1], target[2] + depth.max(0.2)];
        let view = [
            [1.0, 0.0, 0.0, -eye[0]],
            [0.0, 1.0, 0.0, -eye[1]],
            [0.0, 0.0, 1.0, -eye[2]],
            [0.0, 0.0, 0.0, 1.0],
        ];
        (view, eye)
    }

    /// Upper-body centre of an avatar in avatar-root space (the space the
    /// renderer draws in), used to aim the sensor-mirror camera. Midpoint of
    /// the two upper-arm (shoulder) bones, falling back to the head, then the
    /// hips. `None` when the avatar carries no humanoid rig.
    pub(crate) fn avatar_upper_body_center(avatar: &AvatarInstance) -> Option<[f32; 3]> {
        use crate::asset::HumanoidBone::*;
        let humanoid = avatar.asset.humanoid.as_ref()?;
        let bone_pos = |bone: crate::asset::HumanoidBone| {
            humanoid
                .bone_map
                .get(&bone)
                .and_then(|node| avatar.pose.global_transforms.get(node.0 as usize))
                .map(crate::math_utils::mat4_translation)
        };
        if let (Some(l), Some(r)) = (bone_pos(LeftUpperArm), bone_pos(RightUpperArm)) {
            return Some([
                (l[0] + r[0]) * 0.5,
                (l[1] + r[1]) * 0.5,
                (l[2] + r[2]) * 0.5,
            ]);
        }
        bone_pos(Head).or_else(|| bone_pos(Hips))
    }

    /// Diagnosis switch for the body-SDF splat cost: `VULVATAR_NO_BODY_SDF=1`
    /// drops the splat request from the frame input entirely, so the
    /// renderer never sentinel-fills, dispatches, or reads the field
    /// back (spring chains then sample "outside the band" everywhere =
    /// no hair collision). Used to attribute GPU time to the splat pass
    /// vs the rest of the frame.
    fn body_sdf_disabled_by_env() -> bool {
        static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *DISABLED.get_or_init(|| {
            std::env::var_os("VULVATAR_NO_BODY_SDF")
                .map(|v| v.to_string_lossy() != "0")
                .unwrap_or(false)
        })
    }

    /// How many garment primitives join the body-SDF splat (see
    /// `clearance::sdf_garment_splat_prims`). Env
    /// `VULVATAR_SDF_GARMENTS` (0 disables the garment layer — the
    /// skin+face-only behaviour before 2026-09-15). Defaults to 3.
    fn sdf_garment_cap() -> usize {
        static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *CAP.get_or_init(|| {
            std::env::var("VULVATAR_SDF_GARMENTS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(3)
        })
    }

    /// Quantized pose key covering everything the splatted vertices
    /// depend on: the skeleton pose (`local_transforms`, with
    /// spring-driven nodes excluded — hair chains jitter by
    /// construction and none of their bones skin the splatted
    /// body/face primitives) and the expression weights (the only
    /// morph driver). Quantizing to ~2^-11 relative means
    /// solver-settling jitter cannot keep the key moving; a boundary
    /// crossing costs one extra splat, never a stale field (the field
    /// is a pure function of the exact pose — the key only decides
    /// whether it is recomputed this frame).
    fn splat_pose_key(
        local_transforms: &[crate::asset::Transform],
        spring_driven: &std::collections::HashSet<usize>,
        expression_weights: &[crate::avatar::expressions::ResolvedExpressionWeight],
    ) -> u64 {
        #[inline]
        fn feed(h: &mut u64, x: f32) {
            *h ^= (x.to_bits() >> 10) as u64;
            *h = h.wrapping_mul(0x100000001b3);
        }
        let mut h = 0xcbf29ce484222325u64;
        for (i, t) in local_transforms.iter().enumerate() {
            if spring_driven.contains(&i) {
                continue;
            }
            for c in t
                .translation
                .iter()
                .chain(t.rotation.iter())
                .chain(t.scale.iter())
            {
                feed(&mut h, *c);
            }
        }
        for w in expression_weights {
            feed(&mut h, w.weight);
        }
        h
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
        scene_colliders: &[crate::simulation::cloth::ResolvedCollider],
        sdf_splat_changed: &std::collections::HashSet<u64>,
        garment_splat: &std::collections::HashMap<u64, Vec<(crate::asset::MeshId, crate::asset::PrimitiveId)>>,
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
                            // Shared material resolution + alpha/cull/outline
                            // mapping + morph weights live in
                            // `RenderMeshInstance::from_primitive`; the only
                            // live-path override is the user-selected shading
                            // mode.
                            let mut mesh_instance =
                                RenderMeshInstance::from_primitive(avatar, mesh.id, prim);
                            mesh_instance.material_binding.mode = match material_mode_index {
                                0 => MaterialShaderMode::Unlit,
                                1 => MaterialShaderMode::SimpleLit,
                                _ => MaterialShaderMode::ToonLike,
                            };
                            mesh_instance
                        })
                    })
                    .collect();

                // GPU cloth substeps each at `fixed_dt`, matching the
                // CPU path's `for _ in 0..substeps { step_cloth(fixed_dt) }`
                // loop. The renderer's substep loop runs verlet +
                // constraint iters that many times per frame.
                let cloth_substep_dt = fixed_dt.max(1.0 / 1000.0);
                // Avatar-node collision capsules in world space, resolved
                // once per frame and shared by every cloth snapshot. Scene
                // colliders are CPU-solver-only (see ClothGpuDispatchControl).
                let mut gpu_colliders = gpu_colliders_for(avatar);
                // Scene colliders ride the same capsule list (the CPU
                // solver gets them via PhysicsWorld::step_cloth).
                gpu_colliders.extend(
                    scene_colliders.iter().cloned().map(resolved_to_gpu_collider),
                );
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
                    &avatar.pose.global_transforms,
                    &gpu_colliders,
                );

                // Body-SDF splat request: only while the spring solver
                // runs, only when a body primitive exists to splat, and
                // only when the posed body actually changed since the
                // last splat (see the pose-key diff in `run_frame` —
                // the field is a pure function of the posed body, so an
                // unchanged pose reuses the previous field verbatim).
                // The grid is a pure function of the rest AABB, so
                // renderer-side slot reuse stays stable across frames.
                let body_sdf = if toggles.spring_enabled
                    && !Self::body_sdf_disabled_by_env()
                    && sdf_splat_changed.contains(&avatar.id.0)
                    && avatar
                        .asset
                        .spring_bones
                        .iter()
                        .any(|sb| sb.body_collision)
                {
                    // Splat list: the body surface plus any face/head
                    // surface — on many models the face is its own
                    // primitive, and hair bangs / twintails collide
                    // against it — plus the garment layer
                    // (`clearance::sdf_garment_splat_prims`) so hair
                    // resting on clothing resolves against the CLOTHED
                    // surface, not the skin beneath it (measured:
                    // Yumeka's jacket floats p50 20 mm and hair sank
                    // 9–16 mm into it, `diagnostics/sdf_hair_20260915`).
                    // Union their rest AABBs for the grid: the root AABB
                    // can be bloated by hair prims and a T-pose bind,
                    // which coarsens the voxels for everyone. Contacts
                    // only happen within the splat shell of these
                    // surfaces, so strands outside the grid correctly
                    // sample "no collision".
                    let garment_cap = Self::sdf_garment_cap();
                    let garment_prims: Vec<_> = (garment_cap > 0)
                        .then(|| {
                            // Cloth-simulated targets are excluded: their
                            // surface moves with the solver, and the
                            // solver-driven skirt rarely touches hair —
                            // splatting it would only add cells.
                            let cloth_targets: std::collections::HashSet<u64> = avatar
                                .cloth_overlays
                                .iter()
                                .filter_map(|s| s.state.target_primitive_id.map(|p| p.0))
                                .collect();
                            garment_splat
                                .get(&avatar.id.0)
                                .map(|v| v.as_slice())
                                .unwrap_or(&[])
                                .iter()
                                .filter(|(_, p)| !cloth_targets.contains(&p.0))
                                .cloned()
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    let mut splat: Vec<(crate::asset::MeshId, crate::asset::PrimitiveId)> =
                        Vec::new();
                    let mut union = crate::asset::Aabb::empty();
                    {
                        let splat_cap = 4 + garment_prims.len();
                        let mut push_prim = |mesh_id, primitive_id, bounds: &crate::asset::Aabb| {
                            if splat.len() >= splat_cap
                                || splat.iter().any(|&(m, p)| m == mesh_id && p == primitive_id)
                            {
                                return;
                            }
                            splat.push((mesh_id, primitive_id));
                            union.expand(bounds);
                        };
                        if let Some((mesh_id, primitive_id)) =
                            crate::asset::clearance::find_body_primitive(&avatar.asset)
                                .or_else(|| {
                                    avatar.asset.body_primitive_id.and_then(|pid| {
                                        avatar.asset.meshes.iter().find_map(|m| {
                                            m.primitives
                                                .iter()
                                                .find(|p| p.id == pid)
                                                .map(|_| (m.id, pid))
                                        })
                                    })
                                })
                        {
                            for m in &avatar.asset.meshes {
                                if m.id != mesh_id {
                                    continue;
                                }
                                for p in &m.primitives {
                                    if p.id == primitive_id {
                                        push_prim(mesh_id, primitive_id, &p.bounds);
                                    }
                                }
                            }
                        }
                        for m in &avatar.asset.meshes {
                            let mesh_hit =
                                m.name.to_lowercase().contains("face")
                                    || m.name.to_lowercase().contains("head");
                            for p in &m.primitives {
                                let mat_hit = avatar
                                    .asset
                                    .materials
                                    .iter()
                                    .find(|mat| mat.id == p.material_id)
                                    .map(|mat| {
                                        let n = mat.name.to_lowercase();
                                        n.contains("face") || n.contains("head")
                                    })
                                    .unwrap_or(false);
                                if (mesh_hit || mat_hit) && p.vertex_count > 100 {
                                    push_prim(m.id, p.id, &p.bounds);
                                }
                            }
                        }
                        for (mesh_id, primitive_id) in &garment_prims {
                            if let Some(bounds) = avatar
                                .asset
                                .meshes
                                .iter()
                                .find(|m| m.id == *mesh_id)
                                .and_then(|m| m.primitives.iter().find(|p| p.id == *primitive_id))
                                .map(|p| p.bounds)
                            {
                                push_prim(*mesh_id, *primitive_id, &bounds);
                            }
                        }
                    }
                    (!splat.is_empty()).then(|| BodySdfPlan {
                        prims: splat,
                        grid: crate::simulation::sdf::SdfGrid::for_aabb(&union),
                    })
                } else {
                    None
                };

                RenderAvatarInstance {
                    instance_id: avatar.id,
                    world_transform: avatar.world_transform.clone(),
                    mesh_instances,
                    skinning_matrices: avatar.pose.skinning_matrices.clone(),
                    cloth_deforms,
                    body_sdf,
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

        // Camera: free orbit by default, or the 1:1 sensor mirror when a
        // `sensor_camera` rode along (mirror toggle on + metric-native pose).
        // The mirror looks at the avatar's upper-body centre from straight in
        // front at the subject's real distance and projects through the
        // sensor's own intrinsics; the selfie flip is already in the source
        // skeleton, so the front view does not re-mirror.
        let (view, projection, eye_pos) = if let Some(sensor) = fi_config.sensor_camera.as_ref() {
            let target = avatars
                .first()
                .and_then(Self::avatar_upper_body_center)
                .unwrap_or([0.0, 1.0, 0.0]);
            let (v, eye) = Self::build_sensor_view_matrix(target, sensor.anchor_depth_m);
            let p = Self::build_projection_from_intrinsics(&sensor.intrinsics, 0.1, 10.0);
            (v, p, eye)
        } else {
            let aspect = output_extent[0] as f32 / output_extent[1].max(1) as f32;
            let (v, eye) = Self::build_view_matrix(cam);
            let p = Self::build_projection_matrix(cam.fov_deg, aspect, 0.1, 1000.0);
            (v, p, eye)
        };

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

/// Full-precision fingerprint of every input one GPU cloth dispatch
/// consumes besides particle state: the ctrl dt (post-clamp, exactly
/// what `collect_cloth_deforms` puts in the snapshot), sim parameters,
/// pin bindings + the transforms they resolve against, and the merged
/// capsule list. Full f32 bits — quantizing would hide driver motion
/// below ~1 mm, the band the settle-sleep gate must wake on.
/// Spatial quantisation grid of the cloth settle fingerprint, in metres
/// (~0.49 mm). Bit-precise hashing was the original contract, but live
/// tracking jitters every driven transform at f32 granularity, so the
/// fingerprint changed EVERY frame and the cloth never slept (measured
/// 2026-09-16: quiet_frames stuck at 0 with a still subject). The
/// solver cannot respond below ~0.5 mm anyway — collision margin is
/// 15 mm, SDF contact 4 mm — so motion inside one grid cell is noise
/// by contract, while any real ≥1 mm driver motion crosses cells and
/// wakes the slot.
fn settle_quant(v: f32) -> f32 {
    (v * 2048.0).round()
}

fn cloth_gpu_inputs_hash(
    ctrl_dt: f32,
    sim: &crate::simulation::cloth::ClothSimState,
    global_transforms: &[crate::asset::Mat4],
    colliders: &[crate::renderer::frame_input::ClothGpuCollider],
) -> u64 {
    use crate::math_utils::vec3_scale;
    use crate::simulation::settle::SettleHasher;
    let mut h = SettleHasher::new();
    h.write_f32(ctrl_dt);
    h.write_f32(sim.damping);
    h.write_f32(sim.sdf_contact);
    h.write_f32s(&sim.gravity);
    h.write_f32s(&vec3_scale(&sim.wind_direction, sim.wind_response));
    h.write_u32(sim.solver_iterations);
    h.write_f32(sim.collision_margin);
    h.write_bool(sim.self_collision);
    h.write_f32(sim.self_collision_radius);
    h.write_u32(sim.pin_targets.len() as u32);
    for pin in &sim.pin_targets {
        h.write_u32(pin.node_index as u32);
        // Pin offsets are spatial (metres) — quantise.
        h.write_f32s(&pin.offset.map(settle_quant));
    }
    h.write_u32(global_transforms.len() as u32);
    for m in global_transforms {
        for col in m {
            h.write_f32s(&col.map(settle_quant));
        }
    }
    h.write_u32(colliders.len() as u32);
    for c in colliders {
        h.write_f32s(&c.p0.map(settle_quant));
        h.write_f32s(&c.p1.map(settle_quant));
        h.write_f32(settle_quant(c.radius));
    }
    h.finish()
}

/// Per-frame settle-sleep gate for every cloth slot on one avatar
/// (app thread, before the snapshot collector — idle z-fight campaign,
/// 2026-09-15). Suppression piggybacks on the renderer's existing
/// frozen-frame contract: `substeps == 0` skips every cloth dispatch,
/// pin write and version bump, so the persistent `cloth_pos_ssbo`
/// keeps last frame's bits exactly.
///
/// Sleep requires BOTH halves: a quiet position-readback streak
/// (folded by [`fold_cloth_readback_settle`]) and an unchanged input
/// fingerprint. Any input change wakes the slot immediately; the quiet
/// streak then re-accumulates while the induced motion decays.
fn update_cloth_settle_gate(
    avatar: &mut crate::avatar::AvatarInstance,
    fixed_dt: f32,
    gate_active: bool,
    scene_colliders: &[crate::simulation::cloth::ResolvedCollider],
) {
    // Reset first — a suppression flag must never survive a frame the
    // gate did not evaluate (cloth toggle off, GUI pause, …).
    if let Some(cs) = avatar.cloth_state.as_mut() {
        cs.settle.suppress_dispatch = false;
    }
    for slot in &mut avatar.cloth_overlays {
        slot.state.settle.suppress_dispatch = false;
    }
    if !gate_active {
        return;
    }
    // Same merged list the snapshot collector ships: avatar-node
    // capsules + scene colliders, spheres as degenerate capsules.
    let mut gpu_colliders = gpu_colliders_for(avatar);
    gpu_colliders.extend(scene_colliders.iter().cloned().map(resolved_to_gpu_collider));
    // Same clamp `collect_cloth_deforms` applies to the ctrl dt — hash
    // what the dispatch will actually consume.
    let ctrl_dt = fixed_dt.max(1.0 / 1000.0);

    fn gate_slot(
        cs: &mut crate::avatar::instance::ClothState,
        sim: Option<&crate::simulation::cloth::ClothSimState>,
        ctrl_dt: f32,
        global_transforms: &[crate::asset::Mat4],
        colliders: &[crate::renderer::frame_input::ClothGpuCollider],
    ) {
        use crate::simulation::cloth_gpu_boundary::ClothSolverBackend;
        if cs.solver_backend != ClothSolverBackend::Gpu || !cs.enabled {
            return;
        }
        let Some(sim) = sim else { return };
        let hash = cloth_gpu_inputs_hash(ctrl_dt, sim, global_transforms, colliders);
        let settle = &mut cs.settle;
        if settle.last_inputs != Some(hash) {
            settle.sleeping = false;
            settle.quiet_frames = 0;
            settle.last_inputs = Some(hash);
        } else if settle.sleeping {
            settle.suppress_dispatch = true;
        }
    }

    if let Some(cs) = avatar.cloth_state.as_mut() {
        gate_slot(
            cs,
            avatar.cloth_sim.as_ref(),
            ctrl_dt,
            &avatar.pose.global_transforms,
            &gpu_colliders,
        );
    }
    for slot in &mut avatar.cloth_overlays {
        gate_slot(
            &mut slot.state,
            Some(&slot.sim),
            ctrl_dt,
            &avatar.pose.global_transforms,
            &gpu_colliders,
        );
    }
}

/// Fold one GPU position readback row into a cloth slot's settle
/// state: the largest per-particle motion since the previous row
/// advances the quiet streak (`settle_bump` re-evaluates `sleeping`).
/// While the slot is suppressed the renderer maps a frozen SSBO, so
/// the delta is exactly 0 and the streak keeps holding. A length
/// mismatch (first row after attach, re-attach, topology change) has
/// no baseline — delta 0, baseline re-seeded.
fn fold_cloth_readback_settle(
    cs: &mut crate::avatar::instance::ClothState,
    positions: &[crate::asset::Vec3],
) -> f32 {
    use crate::math_utils::{vec3_length, vec3_sub};
    use crate::simulation::settle::settle_bump;

    let settle = &mut cs.settle;
    let max_delta = if settle.last_readback.len() == positions.len() {
        settle
            .last_readback
            .iter()
            .zip(positions.iter())
            .map(|(a, b)| vec3_length(&vec3_sub(a, b)))
            .fold(0.0f32, f32::max)
    } else {
        0.0
    };
    settle.last_readback.clear();
    settle.last_readback.extend_from_slice(positions);
    settle.last_max_delta = max_delta;
    let (quiet_frames, sleeping) = settle_bump(settle.quiet_frames, max_delta);
    settle.quiet_frames = quiet_frames;
    settle.sleeping = sleeping;
    max_delta
}

/// World-space avatar collision capsules for the GPU cloth stage —
/// the sibling of the CPU solver's `resolve_colliders` output, with
/// spheres encoded as degenerate capsules. Enabled mask and node-index
/// bounds follow `simulation::cloth::resolve_colliders` exactly.
fn gpu_colliders_for(
    avatar: &crate::avatar::AvatarInstance,
) -> Vec<crate::renderer::frame_input::ClothGpuCollider> {
    crate::simulation::cloth::resolve_colliders(
        &avatar.asset.colliders,
        &avatar.pose.global_transforms,
        &avatar.collider_enabled,
    )
    .into_iter()
    .map(resolved_to_gpu_collider)
    .collect()
}

/// One resolved collider (sphere or capsule) → GPU capsule; spheres
/// become degenerate capsules, which the closest-point math treats
/// identically.
fn resolved_to_gpu_collider(
    rc: crate::simulation::cloth::ResolvedCollider,
) -> crate::renderer::frame_input::ClothGpuCollider {
    let (p0, p1, radius) = match rc {
        crate::simulation::cloth::ResolvedCollider::Sphere { center, radius } => {
            (center, center, radius)
        }
        crate::simulation::cloth::ResolvedCollider::Capsule {
            center,
            radius,
            half_height,
            axis,
        } => {
            let a = crate::math_utils::vec3_sub(
                &center,
                &crate::math_utils::vec3_scale(&axis, half_height),
            );
            let b = crate::math_utils::vec3_add(
                &center,
                &crate::math_utils::vec3_scale(&axis, half_height),
            );
            (a, b, radius)
        }
    };
    crate::renderer::frame_input::ClothGpuCollider { p0, p1, radius }
}

/// Per-particle pin world targets for the GPU solver, mirroring CPU
/// `cloth_solver::collision::apply_pin_targets`: `T(node) · offset`
/// per pin binding, expanded to particle index space. Entries for
/// unpinned particles are zero; the renderer skips them via
/// `gpu_attach.pinned`. Empty when the cloth has no pins.
fn gpu_pin_targets(
    sim: &crate::simulation::cloth::ClothSimState,
    global_transforms: &[crate::asset::Mat4],
) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0f32; 3]; sim.particles.len()];
    for pin in &sim.pin_targets {
        let Some(mat) = global_transforms.get(pin.node_index) else {
            continue;
        };
        let [ox, oy, oz] = pin.offset;
        let world = [
            mat[0][0] * ox + mat[1][0] * oy + mat[2][0] * oz + mat[3][0],
            mat[0][1] * ox + mat[1][1] * oy + mat[2][1] * oz + mat[3][1],
            mat[0][2] * ox + mat[1][2] * oy + mat[2][2] * oz + mat[3][2],
        ];
        for &pi in &pin.particle_indices {
            if pi < out.len() {
                out[pi] = world;
            }
        }
    }
    out
}

/// R5 probe: world positions of the spring-driven garment bones the
/// costume probe tracks (skirt chains, tail). Same name filter as the
/// original inline probe.
fn costume_bone_positions(avatar: &AvatarInstance) -> Vec<(String, [f32; 3])> {
    let gt = &avatar.pose.global_transforms;
    // All spring-chain joints (root + joints), not just skirt/tail: the
    // 2026-09-15 nipple-area flail report needs breast/hair chains in the
    // live dump. Named by node so chains stay identifiable.
    let mut seen = std::collections::HashSet::new();
    let mut idxs: Vec<usize> = Vec::new();
    for sb in &avatar.asset.spring_bones {
        for &nid in std::iter::once(&sb.chain_root).chain(sb.joints.iter()) {
            if seen.insert(nid.0) {
                idxs.push(nid.0 as usize);
            }
        }
    }
    idxs.sort_unstable();
    idxs.into_iter()
        .filter_map(|i| {
            let n = avatar.asset.skeleton.nodes.get(i)?;
            gt.get(i).map(|m| (n.name.clone(), [m[3][0], m[3][1], m[3][2]]))
        })
        .collect()
}

/// R5 probe: CPU linear-blend-skinning bbox of one primitive (4-weight
/// blend, weight-normalised when the weights sum above a threshold —
/// the same recipe as the original inline probe and the clearance
/// pass). Returns `(mesh_name, min, max)`, or `None` when the
/// primitive has no CPU vertex payload. Requires
/// `pose.skinning_matrices` to be current with `pose.global_transforms`
/// — callers refresh them first.
fn cpu_lbs_prim_bbox(
    avatar: &AvatarInstance,
    prim_id: crate::asset::PrimitiveId,
) -> Option<(String, [f32; 3], [f32; 3])> {
    let sm = &avatar.pose.skinning_matrices;
    let mesh = avatar
        .asset
        .meshes
        .iter()
        .find(|m| m.primitives.iter().any(|p| p.id == prim_id))?;
    let prim = mesh.primitives.iter().find(|p| p.id == prim_id)?;
    let vd = prim.vertices.as_ref()?;
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for (i, &pos) in vd.positions.iter().enumerate() {
        let mut world = [0.0f32; 3];
        let mut total_w = 0.0;
        if i < vd.joint_weights.len() && i < vd.joint_indices.len() {
            for k in 0..4 {
                let w = vd.joint_weights[i][k];
                if w > 0.0001 {
                    let j = vd.joint_indices[i][k] as usize;
                    if let Some(m) = sm.get(j) {
                        for c in 0..3 {
                            world[c] +=
                                w * (m[0][c] * pos[0] + m[1][c] * pos[1] + m[2][c] * pos[2] + m[3][c]);
                        }
                        total_w += w;
                    }
                }
            }
        }
        if total_w > 0.001 {
            for c in 0..3 {
                world[c] /= total_w;
            }
        }
        for c in 0..3 {
            lo[c] = lo[c].min(world[c]);
            hi[c] = hi[c].max(world[c]);
        }
    }
    Some((mesh.name.clone(), lo, hi))
}

/// R5 probe: the clearance (`body_primitive_id`) and containment
/// (`containment_primitive_id`) anchor parents of a primitive, straight
/// from the asset — the primitive-ID correspondence table the quality
/// plan's §10 asks for, instead of guessing ids from screen positions.
fn prim_anchor_parents(
    avatar: &AvatarInstance,
    prim_id: crate::asset::PrimitiveId,
) -> (Option<u64>, Option<u64>) {
    let prim = avatar
        .asset
        .meshes
        .iter()
        .flat_map(|m| m.primitives.iter())
        .find(|p| p.id == prim_id);
    (
        prim.and_then(|p| p.body_primitive_id).map(|id| id.0),
        prim.and_then(|p| p.containment_primitive_id).map(|id| id.0),
    )
}

/// R4 delivery predicate: does this GPU cloth readback row belong to
/// this avatar's cloth slot? All three identifiers must agree — the
/// avatar instance that simulated the slot, the mesh, and the
/// primitive the cloth is bound to — and the slot must be GPU-backed.
/// Free function so the contract stays unit-testable without an
/// `Application` or a Vulkan device.
fn cloth_readback_matches(
    entry: &crate::renderer::ClothReadback,
    avatar_id: crate::avatar::AvatarInstanceId,
    cs: &crate::avatar::instance::ClothState,
) -> bool {
    entry.instance_id == Some(avatar_id.0)
        && cs.solver_backend == crate::simulation::cloth_gpu_boundary::ClothSolverBackend::Gpu
        && cs.target_mesh_id == Some(entry.mesh_id)
        && cs.target_primitive_id == Some(entry.primitive_id)
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
///
/// `colliders` carries the frame's world-space collision capsules for the
/// GPU stage — the caller resolves the avatar-node colliders once per
/// frame (all cloths on one avatar collide against the same body).
fn collect_cloth_deforms<'a>(
    sources: impl IntoIterator<
        Item = (
            &'a crate::avatar::instance::ClothState,
            Option<&'a crate::simulation::cloth::ClothSimState>,
        ),
    >,
    fixed_dt: f32,
    substeps: u32,
    global_transforms: &[crate::asset::Mat4],
    colliders: &[crate::renderer::frame_input::ClothGpuCollider],
) -> Vec<ClothDeformSnapshot> {
    use crate::math_utils::vec3_scale;
    use crate::renderer::frame_input::{ClothGpuAttachData, ClothGpuDispatchControl};
    use crate::simulation::cloth_gpu_boundary::ClothSolverBackend;

    let mut seen_targets: std::collections::HashSet<crate::asset::PrimitiveId> =
        std::collections::HashSet::new();
    sources
        .into_iter()
        .filter_map(|(cs, sim_opt)| {
            let target_primitive_id = cs.target_primitive_id?;
            if !seen_targets.insert(target_primitive_id) {
                return None;
            }
            let (gpu_control, gpu_attach) = if cs.solver_backend == ClothSolverBackend::Gpu {
                match sim_opt {
                    Some(sim) => {
                        let wind_force = vec3_scale(&sim.wind_direction, sim.wind_response);
                        let mut ctrl = ClothGpuDispatchControl {
                            dt: fixed_dt,
                            substeps,
                            damping: sim.damping,
                            gravity: sim.gravity,
                            wind_force,
                            solver_iterations: sim.solver_iterations as u32,
                            pin_positions: gpu_pin_targets(sim, global_transforms),
                            collision_margin: sim.collision_margin,
                            colliders: colliders.to_vec(),
                            self_collision: sim.self_collision,
                            self_collision_radius: sim.self_collision_radius,
                            sdf_contact: sim.sdf_contact,
                        };
                        // Settle-sleep (idle z-fight campaign, 2026-09-15):
                        // the app-side gate suppressed this cloth — ship 0
                        // substeps so the renderer's frozen-frame contract
                        // skips every dispatch, pin write and version bump,
                        // keeping `cloth_pos_ssbo` bit-stable. The state
                        // stays where it was; `substeps` is hashed into the
                        // frame shape key, so the command buffer cache
                        // simply holds one variant per sleep state.
                        if cs.settle.suppress_dispatch {
                            ctrl.substeps = 0;
                        }
                        // Bend-wing CSR: constraint rows indexed per
                        // WING particle (the hinge never appears — the
                        // bend kernels never move it).
                        let mut bend_adj_offsets =
                            vec![0u32; sim.particles.len() + 1];
                        for bc in &sim.bend_constraints {
                            for wing in [bc.p1, bc.p2] {
                                if wing < sim.particles.len() {
                                    bend_adj_offsets[wing + 1] += 1;
                                }
                            }
                        }
                        for v in 1..bend_adj_offsets.len() {
                            bend_adj_offsets[v] += bend_adj_offsets[v - 1];
                        }
                        let mut cursor = bend_adj_offsets.clone();
                        let mut ordered: Vec<u32> =
                            vec![0; sim.bend_constraints.len() * 2];
                        for (ci, bc) in sim.bend_constraints.iter().enumerate() {
                            for wing in [bc.p1, bc.p2] {
                                if wing < sim.particles.len() {
                                    let slot = cursor[wing] as usize;
                                    ordered[slot] = ci as u32;
                                    cursor[wing] += 1;
                                }
                            }
                        }
                        let attach = ClothGpuAttachData {
                            constraints: sim
                                .distance_constraints
                                .iter()
                                .map(|c| (c.a as u32, c.b as u32, c.rest_length, c.stiffness))
                                .collect(),
                            triangle_indices: sim.triangle_indices.clone(),
                            inv_masses: sim.particles.iter().map(|p| p.inv_mass).collect(),
                            pinned: sim.particles.iter().map(|p| p.pinned).collect(),
                            bend: sim
                                .bend_constraints
                                .iter()
                                .map(|bc| crate::renderer::pipeline::ClothBendGpu {
                                    p0: bc.p0 as u32,
                                    p1: bc.p1 as u32,
                                    p2: bc.p2 as u32,
                                    _pad: 0,
                                    rest_angle: bc.rest_angle,
                                    stiffness: bc.stiffness,
                                    _pad2: [0; 2],
                                })
                                .collect(),
                            bend_adj_offsets,
                            bend_adj_constraints: ordered,
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
mod sensor_camera_tests {
    use super::*;

    /// Row-major Mat4 · column vec4.
    fn mul(m: &crate::asset::Mat4, v: &[f32; 4]) -> [f32; 4] {
        let mut o = [0.0f32; 4];
        for (r, row) in m.iter().enumerate() {
            o[r] = row[0] * v[0] + row[1] * v[1] + row[2] * v[2] + row[3] * v[3];
        }
        o
    }

    fn intrinsics(cx: f32, cy: f32) -> crate::tracking::CameraIntrinsics {
        crate::tracking::CameraIntrinsics {
            fx: 600.0,
            fy: 600.0,
            cx,
            cy,
            width: 1280,
            height: 720,
        }
    }

    /// Centred principal point: a view-space point straight down the optical
    /// axis projects to the NDC centre.
    #[test]
    fn axis_point_maps_to_ndc_centre() {
        let p = Application::build_projection_from_intrinsics(&intrinsics(640.0, 360.0), 0.1, 10.0);
        let clip = mul(&p, &[0.0, 0.0, -1.0, 1.0]);
        let ndc = [clip[0] / clip[3], clip[1] / clip[3]];
        assert!(
            ndc[0].abs() < 1e-5 && ndc[1].abs() < 1e-5,
            "axis → centre, got {ndc:?}"
        );
    }

    /// A pixel offset maps to the matching NDC offset. A point at x=+0.5,
    /// z=−1 projects to pixel u = −fx·x/z + cx = 940 → ndc_x = 0.46875.
    #[test]
    fn off_axis_point_maps_to_matching_ndc() {
        let p = Application::build_projection_from_intrinsics(&intrinsics(640.0, 360.0), 0.1, 10.0);
        let clip = mul(&p, &[0.5, 0.0, -1.0, 1.0]);
        let ndc_x = clip[0] / clip[3];
        assert!(
            (ndc_x - 0.46875).abs() < 1e-4,
            "expected 0.46875, got {ndc_x}"
        );
    }

    /// A principal point right-of-centre (cx > w/2) shifts the on-axis point
    /// to +NDC — the asymmetric frustum the sensor's real optics need.
    #[test]
    fn principal_point_offset_shifts_frustum() {
        let p = Application::build_projection_from_intrinsics(&intrinsics(700.0, 360.0), 0.1, 10.0);
        let clip = mul(&p, &[0.0, 0.0, -1.0, 1.0]);
        let ndc_x = clip[0] / clip[3];
        // ndc_x = 2·700/1280 − 1 = 0.09375.
        assert!(
            (ndc_x - 0.09375).abs() < 1e-4,
            "principal offset → 0.09375, got {ndc_x}"
        );
    }

    /// Near plane → NDC z 0, far plane → NDC z 1 (Vulkan depth range), same
    /// mapping as the symmetric projection.
    #[test]
    fn depth_maps_near_zero_far_one() {
        let p = Application::build_projection_from_intrinsics(&intrinsics(640.0, 360.0), 0.1, 10.0);
        let near = mul(&p, &[0.0, 0.0, -0.1, 1.0]);
        let far = mul(&p, &[0.0, 0.0, -10.0, 1.0]);
        assert!((near[2] / near[3]).abs() < 1e-4, "near → 0");
        assert!(((far[2] / far[3]) - 1.0).abs() < 1e-4, "far → 1");
    }

    /// The sensor view eye sits `depth` in front (+Z) of the target, and the
    /// target maps to (0,0,−depth) in view space — straight ahead, no rotation.
    #[test]
    fn sensor_view_places_eye_in_front_and_looks_back() {
        let target = [0.1, 1.0, 0.0];
        let (view, eye) = Application::build_sensor_view_matrix(target, 1.5);
        assert_eq!(eye, [0.1, 1.0, 1.5]);
        let vt = mul(&view, &[target[0], target[1], target[2], 1.0]);
        assert!(
            vt[0].abs() < 1e-6 && vt[1].abs() < 1e-6 && (vt[2] + 1.5).abs() < 1e-6,
            "target should sit at (0,0,-1.5) in view space, got {vt:?}"
        );
    }

    /// A garbage/zero anchor depth can't drop the eye onto the subject — it's
    /// clamped to a small positive standoff.
    #[test]
    fn sensor_view_clamps_zero_depth() {
        let (_v, eye) = Application::build_sensor_view_matrix([0.0, 1.0, 0.0], 0.0);
        assert!(
            eye[2] >= 0.2,
            "zero depth must clamp to a positive standoff, got {}",
            eye[2]
        );
    }
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
            solver_backend: crate::simulation::cloth_gpu_boundary::ClothSolverBackend::Cpu,
            settle: Default::default(),
        }
    }

    #[test]
    fn gpu_pin_targets_follow_bound_node_transform() {
        use crate::simulation::cloth::{ClothParticle, ClothSimState, PinTarget};

        let mut sim = ClothSimState::default();
        sim.particles = vec![
            ClothParticle::new([0.0, 0.0, 0.0], true),
            ClothParticle::new([1.0, 0.0, 0.0], false),
        ];
        sim.pin_targets.push(PinTarget {
            node_index: 1,
            offset: [0.5, 0.0, 0.0],
            particle_indices: vec![0],
        });

        // Node 0 identity, node 1 translated by (10, 20, 30) with a
        // +90° rotation about Y so the local +X offset maps to world
        // +Z (column-major Mat4, same convention as the CPU solver's
        // `apply_pin_targets`).
        let mut t = [[0.0f32; 4]; 4];
        t[0][2] = 1.0; // column 0 = (0,0,1)
        t[1][1] = 1.0;
        t[2][0] = -1.0; // column 2 = (-1,0,0)
        t[3][0] = 10.0;
        t[3][1] = 20.0;
        t[3][2] = 30.0;
        t[3][3] = 1.0;
        let mut identity = [[0.0f32; 4]; 4];
        identity[0][0] = 1.0;
        identity[1][1] = 1.0;
        identity[2][2] = 1.0;
        identity[3][3] = 1.0;
        let targets = gpu_pin_targets(&sim, &[identity, t]);

        assert_eq!(targets.len(), 2);
        // T * (0.5,0,0) = translate + R*(0.5,0,0) = (10, 20, 30.5).
        assert_eq!(targets[0], [10.0, 20.0, 30.5]);
        // Unpinned particle keeps the zero placeholder.
        assert_eq!(targets[1], [0.0, 0.0, 0.0]);

        // Out-of-range node index is skipped, not fatal.
        sim.pin_targets[0].node_index = 9;
        let targets = gpu_pin_targets(&sim, &[]);
        assert_eq!(targets[0], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn multi_cloth_targeting_distinct_primitives_all_survive() {
        let body = make_cloth_state(1, Some(PrimitiveId(10)), 32, 1);
        let skirt = make_cloth_state(2, Some(PrimitiveId(20)), 64, 1);
        let scarf = make_cloth_state(3, Some(PrimitiveId(30)), 16, 1);

        let result = collect_cloth_deforms(
            [(&body, None), (&skirt, None), (&scarf, None)],
            1.0 / 60.0,
            1,
            &[],
            &[],
        );

        assert_eq!(
            result.len(),
            3,
            "all three distinct-target cloths must be kept"
        );
        let target_ids: Vec<u64> = result.iter().map(|c| c.target_primitive_id.0).collect();
        assert_eq!(target_ids, vec![10, 20, 30]);
        let vertex_counts: Vec<u32> = result.iter().map(|c| c.vertex_count).collect();
        assert_eq!(vertex_counts, vec![32, 64, 16]);
    }

    #[test]
    fn duplicate_target_dedups_with_first_wins() {
        let authoritative = make_cloth_state(1, Some(PrimitiveId(10)), 32, 5);
        let overlay_duplicate = make_cloth_state(2, Some(PrimitiveId(10)), 64, 99);

        let result = collect_cloth_deforms(
            [(&authoritative, None), (&overlay_duplicate, None)],
            1.0 / 60.0,
            1,
            &[],
            &[],
        );

        assert_eq!(result.len(), 1, "duplicate target_primitive_id must dedup");
        assert_eq!(result[0].version, 5, "first (authoritative) cloth wins");
        assert_eq!(result[0].vertex_count, 32);
    }

    #[test]
    fn unbound_cloth_without_target_primitive_is_skipped() {
        let bound = make_cloth_state(1, Some(PrimitiveId(10)), 32, 1);
        let unbound = make_cloth_state(2, None, 64, 1);

        let result = collect_cloth_deforms([(&bound, None), (&unbound, None)], 1.0 / 60.0, 1, &[], &[]);

        assert_eq!(result.len(), 1, "cloth with no render target is dropped");
        assert_eq!(result[0].target_primitive_id.0, 10);
    }

    // ---- R4 readback delivery contract -------------------------------

    use crate::avatar::AvatarInstanceId;
    use crate::renderer::ClothReadback;

    fn gpu_cloth_state(
        overlay_id: u64,
        target_primitive_id: Option<PrimitiveId>,
    ) -> ClothState {
        let mut cs = make_cloth_state(overlay_id, target_primitive_id, 8, 1);
        cs.solver_backend = crate::simulation::cloth_gpu_boundary::ClothSolverBackend::Gpu;
        cs
    }

    fn readback(instance: Option<u64>, mesh: MeshId, prim: PrimitiveId) -> ClothReadback {
        ClothReadback {
            mesh_id: mesh,
            primitive_id: prim,
            instance_id: instance,
            version: 1,
            positions: vec![[0.0; 3]],
            normals: None,
        }
    }

    /// A row stamped with the simulating instance, mesh, and primitive
    /// is delivered to that cloth slot. (`make_cloth_state` derives
    /// `target_mesh_id = MeshId(primitive.0)`, so the agreeing mesh
    /// here is `MeshId(12)`.)
    #[test]
    fn readback_matches_when_all_identifiers_agree() {
        let cs = gpu_cloth_state(1, Some(PrimitiveId(12)));
        assert!(cloth_readback_matches(
            &readback(Some(7), MeshId(12), PrimitiveId(12)),
            AvatarInstanceId(7),
            &cs,
        ));
    }

    /// The old bug: primitive id alone is unique per avatar, not across
    /// avatars — a row from another instance must NOT be delivered.
    #[test]
    fn readback_rejected_on_instance_mismatch() {
        let cs = gpu_cloth_state(1, Some(PrimitiveId(12)));
        assert!(!cloth_readback_matches(
            &readback(Some(9), MeshId(3), PrimitiveId(12)),
            AvatarInstanceId(7),
            &cs,
        ));
        // Unstamped rows are never applied either.
        assert!(!cloth_readback_matches(
            &readback(None, MeshId(3), PrimitiveId(12)),
            AvatarInstanceId(7),
            &cs,
        ));
    }

    /// Same primitive id on a DIFFERENT mesh (two garments inside one
    /// avatar sharing a primitive index) must not cross-deliver.
    #[test]
    fn readback_rejected_on_mesh_mismatch() {
        let cs = gpu_cloth_state(1, Some(PrimitiveId(12)));
        assert!(!cloth_readback_matches(
            &readback(Some(7), MeshId(4), PrimitiveId(12)),
            AvatarInstanceId(7),
            &cs,
        ));
    }

    /// CPU-backed slots take the snapshot-copy path, not the readback.
    #[test]
    fn readback_rejected_on_cpu_backend() {
        let cs = make_cloth_state(1, Some(PrimitiveId(12)), 8, 1);
        assert!(!cloth_readback_matches(
            &readback(Some(7), MeshId(3), PrimitiveId(12)),
            AvatarInstanceId(7),
            &cs,
        ));
    }

    /// The quiet streak sleeps after five identical readbacks and any
    /// real motion resets it — this is the half of the gate fed from
    /// the one-frame-stale position readback.
    #[test]
    fn cloth_readback_settle_quiet_streak_sleeps_and_motion_wakes() {
        let mut cs = gpu_cloth_state(1, Some(PrimitiveId(12)));
        let rest = [[0.0f32; 3]; 4];
        // First row seeds the baseline (no delta yet).
        fold_cloth_readback_settle(&mut cs, &rest);
        for q in 1..5 {
            assert!(!cs.settle.sleeping, "quiet_frames={q}");
            fold_cloth_readback_settle(&mut cs, &rest);
        }
        assert!(cs.settle.sleeping, "five quiet rows must sleep");
        // Frozen stream holds the sleep…
        fold_cloth_readback_settle(&mut cs, &rest);
        assert!(cs.settle.sleeping);
        // …and 2 mm of real motion wakes.
        let mut moved = rest;
        moved[0][1] = 0.002;
        let delta = fold_cloth_readback_settle(&mut cs, &moved);
        assert!((delta - 0.002).abs() < 1e-6, "delta={delta}");
        assert!(!cs.settle.sleeping);
        assert_eq!(cs.settle.quiet_frames, 0);
    }

    /// The wake key is bit-precise: identical inputs hash identically,
    /// and a one-ULP change on ANY consumed input (transform, sim
    /// parameter, capsule, ctrl dt) must change the hash.
    #[test]
    fn cloth_gpu_inputs_hash_is_quantized() {
        let transforms = vec![crate::asset::identity_matrix()];
        let sim = crate::simulation::cloth::ClothSimState::default();
        let colliders = vec![crate::renderer::frame_input::ClothGpuCollider {
            p0: [0.0, 0.0, 0.0],
            p1: [0.0, 1.0, 0.0],
            radius: 0.1,
        }];
        let base = cloth_gpu_inputs_hash(1.0 / 60.0, &sim, &transforms, &colliders);
        assert_eq!(
            cloth_gpu_inputs_hash(1.0 / 60.0, &sim, &transforms, &colliders),
            base,
            "identical inputs must hash identically"
        );

        // Sub-grid jitter (~0.1 mm) on a transform element must NOT wake
        // the slot — live tracking jitters every driven transform at f32
        // granularity, and bit-precise hashing never let the cloth sleep.
        let mut jittered = transforms.clone();
        jittered[0][3][0] += 1e-4;
        assert_eq!(
            cloth_gpu_inputs_hash(1.0 / 60.0, &sim, &jittered, &colliders),
            base,
            "sub-grid tracking jitter must not wake the cloth"
        );

        // A real driver motion (2 mm) crosses grid cells and wakes it.
        let mut moved = transforms.clone();
        moved[0][3][0] += 0.002;
        assert_ne!(cloth_gpu_inputs_hash(1.0 / 60.0, &sim, &moved, &colliders), base);

        // Sim parameter (wind feeds the verlet acceleration) — exact.
        let mut blown = crate::simulation::cloth::ClothSimState::default();
        blown.wind_response = 0.5;
        assert_ne!(cloth_gpu_inputs_hash(1.0 / 60.0, &blown, &transforms, &colliders), base);

        // Capsule move past the grid (2 mm).
        let mut pushed = colliders.clone();
        pushed[0].radius = 0.102;
        assert_ne!(cloth_gpu_inputs_hash(1.0 / 60.0, &sim, &transforms, &pushed), base);

        // Ctrl dt.
        assert_ne!(cloth_gpu_inputs_hash(1.0 / 59.0, &sim, &transforms, &colliders), base);
    }
}

#[cfg(test)]
mod splat_pose_key_tests {
    use super::*;

    fn transform(t: [f32; 3], r: [f32; 4]) -> crate::asset::Transform {
        crate::asset::Transform {
            translation: t,
            rotation: r,
            scale: [1.0, 1.0, 1.0],
        }
    }

    fn weight(w: f32) -> crate::avatar::expressions::ResolvedExpressionWeight {
        crate::avatar::expressions::ResolvedExpressionWeight {
            name: "a".to_string(),
            weight: w,
        }
    }

    fn pose() -> Vec<crate::asset::Transform> {
        vec![
            transform([0.0, 0.8, 0.0], [0.0, 0.0, 0.0, 1.0]),
            transform([0.1, 0.4, 0.0], [0.1, 0.0, 0.0, 0.9]),
            transform([0.0, 0.2, 0.3], [0.0, 0.2, 0.0, 0.9]),
        ]
    }

    fn key(
        transforms: &[crate::asset::Transform],
        weights: &[crate::avatar::expressions::ResolvedExpressionWeight],
    ) -> u64 {
        Application::splat_pose_key(transforms, &std::collections::HashSet::new(), weights)
    }

    #[test]
    fn unchanged_pose_yields_unchanged_key() {
        let (t, w) = (pose(), vec![weight(0.5)]);
        assert_eq!(key(&t, &w), key(&t, &w));
    }

    /// Solver-settling jitter (below the ~2^-11 quantization) must not
    /// read as a pose change — otherwise the skip never fires and the
    /// splat runs at full cost every frame anyway.
    #[test]
    fn sub_quantization_jitter_is_ignored() {
        let w = vec![weight(0.5)];
        let t = pose();
        let mut jittered = pose();
        jittered[1].rotation[0] += 1.0e-7;
        jittered[2].translation[1] -= 1.0e-7;
        assert_ne!(t[1].rotation, jittered[1].rotation, "jitter must be real");
        assert_eq!(key(&t, &w), key(&jittered, &w));
    }

    #[test]
    fn body_bone_change_changes_key() {
        let w = vec![weight(0.5)];
        let mut moved = pose();
        moved[1].rotation[1] += 0.01;
        assert_ne!(key(&pose(), &w), key(&moved, &w));
    }

    #[test]
    fn spring_driven_bone_change_is_ignored() {
        let w = vec![weight(0.5)];
        let mut spring_set = std::collections::HashSet::new();
        spring_set.insert(2usize);
        let mut moved = pose();
        moved[2].rotation[2] += 0.03;
        let base = Application::splat_pose_key(&pose(), &spring_set, &w);
        assert_eq!(base, Application::splat_pose_key(&moved, &spring_set, &w));
        // A body bone still moves the key under the same mask.
        let mut moved_body = pose();
        moved_body[1].rotation[1] += 0.01;
        assert_ne!(base, Application::splat_pose_key(&moved_body, &spring_set, &w));
    }

    #[test]
    fn expression_weight_change_changes_key() {
        let t = pose();
        assert_ne!(key(&t, &[weight(0.5)]), key(&t, &[weight(0.6)]));
    }
}
