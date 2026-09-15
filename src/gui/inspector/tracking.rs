use eframe::egui;

use crate::gui::components::{collapsible_card, filled_button, tonal_button, ButtonTone};
use crate::gui::theme::{color, icon as ic};
use crate::gui::GuiApp;
use crate::t;

pub(super) fn draw_tracking(ui: &mut egui::Ui, state: &mut GuiApp) {
    // Unclean-exit banner: the previous session's stage-log sentinel
    // survived (system freeze, process kill). Offer the degraded
    // safe-mode pipeline for the next start; either choice clears the
    // sentinel so the banner shows once per incident.
    if let Some(sentinel) = state.tracking.unclean_exit_log.clone() {
        let log_path = sentinel.lines().next().unwrap_or("").to_string();
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.label(
                egui::RichText::new(t!("tracking.unclean_exit_title"))
                    .strong()
                    .color(ui.visuals().warn_fg_color),
            );
            ui.label(egui::RichText::new(t!("tracking.unclean_exit_body", log = log_path)).small());
            ui.horizontal(|ui| {
                if ui.button(t!("tracking.unclean_exit_safe_mode")).clicked() {
                    state.tracking.safe_mode_armed = true;
                    crate::tracking::stagelog::clear_stale_sentinel();
                    state.tracking.unclean_exit_log = None;
                }
                if ui.button(t!("tracking.unclean_exit_dismiss")).clicked() {
                    crate::tracking::stagelog::clear_stale_sentinel();
                    state.tracking.unclean_exit_log = None;
                }
            });
        });
        ui.add_space(4.0);
    }
    if state.tracking.safe_mode_armed {
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(t!("tracking.safe_mode_active"))
                    .color(ui.visuals().warn_fg_color),
            );
            if ui.button(t!("tracking.safe_mode_disarm")).clicked() {
                state.tracking.safe_mode_armed = false;
            }
        });
        ui.add_space(4.0);
    }

    // ① Camera & tracking control — the button every session starts
    // with, permanently visible at the top of the panel (it used to be
    // buried inside the collapsed "Input Device" section).
    draw_camera_control(ui, state);

    // ② Which body parts drive the avatar.
    draw_body_parts(ui, state);

    // ③ Capture format details — visited rarely, collapsed by default.
    draw_input_device(ui, state);

    // ④ Display / mirroring options, with the two distinct "mirror"
    // concepts finally side by side and explained.
    draw_display_options(ui, state);

    collapsible_card(
        ui,
        "tracking.pipeline",
        t!("tracking.pipeline"),
        false,
        |ui| {
            ui.checkbox(
                &mut state.tracking.yolox_enabled,
                t!("tracking.pipeline_yolox"),
            );
            ui.checkbox(
                &mut state.tracking.force_cpu_inference,
                t!("tracking.pipeline_force_cpu"),
            );
            ui.label(
                egui::RichText::new(t!("tracking.pipeline_hint"))
                    .small()
                    .weak(),
            );
        },
    );

    // Display smoothing / face-confidence thresholds. Collapsed by
    // default — most users should never need to touch them.
    collapsible_card(
        ui,
        "tracking.advanced_smoothing",
        t!("tracking.advanced_smoothing"),
        false,
        |ui| {
            ui.label(
                egui::RichText::new(t!("tracking.advanced_smoothing_hint"))
                    .color(color::ON_SURFACE_VARIANT),
            );
            ui.add_space(4.0);
            ui.checkbox(
                &mut state.tracking.smoothing.pose_interp_enabled,
                t!("tracking.pose_interp"),
            )
            .on_hover_text(t!("tracking.pose_interp_tooltip"));
            if state.tracking.smoothing.pose_interp_enabled {
                ui.add(
                    egui::Slider::new(
                        &mut state.tracking.smoothing.pose_interp_delay_frac,
                        0.25..=1.0,
                    )
                    .text(t!("tracking.pose_interp_delay")),
                )
                .on_hover_text(t!("tracking.pose_interp_delay_tooltip"));
            }
            ui.add(
                egui::Slider::new(&mut state.tracking.smoothing.rotation_blend, 0.0..=1.0)
                    .text(t!("tracking.rotation_blend")),
            )
            .on_hover_text(t!("tracking.rotation_blend_tooltip"));
            ui.add(
                egui::Slider::new(&mut state.tracking.smoothing.expression_blend, 0.0..=1.0)
                    .text(t!("tracking.expression_blend")),
            )
            .on_hover_text(t!("tracking.expression_blend_tooltip"));
            ui.add(
                egui::Slider::new(
                    &mut state.tracking.smoothing.face_confidence_threshold,
                    0.0..=1.0,
                )
                .text(t!("tracking.face_confidence")),
            )
            .on_hover_text(t!("tracking.face_confidence_tooltip"));
            ui.add_space(4.0);
            if tonal_button(
                ui,
                Some(ic::HISTORY),
                &t!("tracking.smoothing_reset"),
                ButtonTone::Primary,
                true,
            )
            .clicked()
            {
                let defaults = crate::tracking::TrackingSmoothingParams::default();
                state.tracking.smoothing.pose_interp_enabled = defaults.pose_interp_enabled;
                state.tracking.smoothing.pose_interp_delay_frac =
                    defaults.pose_interp_delay_frac;
                state.tracking.smoothing.rotation_blend = defaults.rotation_blend;
                state.tracking.smoothing.expression_blend = defaults.expression_blend;
                state.tracking.smoothing.face_confidence_threshold =
                    defaults.face_confidence_threshold;
            }
        },
    );

    // Detailed diagnostics — backend labels, raw timestamp, per-joint
    // confidence. Collapsed: the one-line status in ① is the everyday
    // read; this is for debugging sessions.
    collapsible_card(
        ui,
        "tracking.inference_status",
        t!("tracking.inference_status"),
        false,
        |ui| {
            if state.is_tracking_active() {
                ui.label(t!(
                    "tracking.camera_label",
                    label = crate::tracking::CAPTURE_BACKEND_LABEL.to_string()
                ));

                // Inference backend (ONNX execution provider) — surfaces a
                // silent CPU fallback when DirectML registration fails. Set
                // by the worker once the pose provider finishes loading;
                // absent before init completes.
                if let Some(label) = state
                    .app
                    .tracking_worker
                    .as_ref()
                    .and_then(|w| w.mailbox().inference_backend_label())
                {
                    ui.label(t!("tracking.inference_label", label = label.to_string()));
                }
            }

            if let Some(tracking) = &state.app.last_tracking_pose {
                ui.label(t!(
                    "tracking.timestamp",
                    value = tracking.source_timestamp.to_string()
                ));
                ui.separator();
                ui.label(t!("tracking.confidence"));
                if let Some(face) = tracking.face {
                    draw_confidence_bar(ui, &t!("tracking.confidence_face"), face.confidence);
                }
                use crate::asset::HumanoidBone;
                let joint_conf = |b: HumanoidBone| -> f32 {
                    tracking.joints.get(&b).map(|j| j.confidence).unwrap_or(0.0)
                };
                draw_confidence_bar(
                    ui,
                    &t!("tracking.confidence_left_shoulder"),
                    joint_conf(HumanoidBone::LeftShoulder),
                );
                draw_confidence_bar(
                    ui,
                    &t!("tracking.confidence_right_shoulder"),
                    joint_conf(HumanoidBone::RightShoulder),
                );
                draw_confidence_bar(
                    ui,
                    &t!("tracking.confidence_left_hand"),
                    joint_conf(HumanoidBone::LeftHand),
                );
                draw_confidence_bar(
                    ui,
                    &t!("tracking.confidence_right_hand"),
                    joint_conf(HumanoidBone::RightHand),
                );
            } else {
                ui.label(t!("tracking.no_tracking_data"));
            }
        },
    );

    collapsible_card(
        ui,
        "tracking.lip_sync",
        t!("tracking.lip_sync"),
        false,
        |ui| {
            draw_lipsync(ui, state);
        },
    );
}

/// Capture format after the safe-mode clamp. Single source for every
/// start/restart site — the clamp used to be copy-pasted between the
/// Start button and the change-while-running restart and had already
/// begun to drift risk.
fn effective_capture_params(state: &GuiApp) -> (u32, u32, u32) {
    let (w, h) = crate::tracking::camera_resolution_for_index(state.tracking.camera_resolution_index);
    let fps = crate::tracking::camera_fps_for_index(state.tracking.camera_framerate_index);
    clamp_for_safe_mode(w, h, fps, state.tracking.safe_mode_armed)
}

/// Safe mode caps the capture format — less camera bandwidth and
/// per-frame CPU work while diagnosing an unclean exit.
fn clamp_for_safe_mode(w: u32, h: u32, fps: u32, safe_mode: bool) -> (u32, u32, u32) {
    if safe_mode {
        (w.min(1280), h.min(720), fps.min(30))
    } else {
        (w, h, fps)
    }
}

/// Marker for the capture format the running camera was started with,
/// kept in egui temp memory. Lets the Input Device card offer an
/// explicit Apply instead of restarting the camera on every combo
/// click (crossing two entries used to restart twice).
fn applied_format_id() -> egui::Id {
    egui::Id::new("tracking_applied_capture_format")
}

fn start_camera(state: &mut GuiApp, ctx: &egui::Context) {
    // Starting the camera IS enabling tracking — the old standalone
    // "Tracking enabled" checkbox is gone. `toggle_tracking` stays the
    // single source of truth for the solve gate (the pause hotkey and
    // status displays still read/write it).
    if !state.tracking.toggle_tracking {
        state.tracking.toggle_tracking = true;
    }
    let (w, h, fps) = effective_capture_params(state);
    let pipeline = state.tracking.pipeline_config();
    let camera_serial = state.tracking.camera_serial.clone();
    state
        .app
        .start_tracking_with_params(w, h, fps, camera_serial, pipeline);
    ctx.data_mut(|d| {
        d.insert_temp(
            applied_format_id(),
            (
                state.tracking.camera_resolution_index,
                state.tracking.camera_framerate_index,
            ),
        )
    });
}

/// ① Always-visible camera / tracking control card.
fn draw_camera_control(ui: &mut egui::Ui, state: &mut GuiApp) {
    crate::gui::components::card(ui, t!("tracking.camera_section"), |ui| {
        let active = state.is_tracking_active();
        let ready = state.is_tracking_ready();
        let tracking_on = state.tracking.toggle_tracking;
        let startable = state.camera_startable();

        // Keep the device list below fresh: scans once on first draw,
        // then every few seconds while it's empty (so plugging the
        // camera in flips the list — and Start — without a manual
        // Rescan), and never while the camera is streaming.
        if let Some(err) = state.rescan_cameras(false, std::time::Duration::from_secs(3)) {
            state.push_warning_notification(err);
        }

        ui.horizontal(|ui| {
            if active && ready {
                if tonal_button(
                    ui,
                    Some(ic::PAUSE),
                    &t!("tracking.stop_camera"),
                    ButtonTone::Error,
                    true,
                )
                .clicked()
                {
                    state.app.stop_tracking();
                }
            } else if active {
                // Camera is initialising — disabled placeholder uses
                // the filled-button silhouette so the layout doesn't
                // jump when it flips to Stop.
                let _ = filled_button(ui, None, &t!("tracking.preparing"), false);
            } else {
                let start =
                    filled_button(ui, Some(ic::PLAY), &t!("tracking.start_camera"), startable);
                if startable {
                    if start.clicked() {
                        let ctx = ui.ctx().clone();
                        start_camera(state, &ctx);
                    }
                } else {
                    // Empty/unsupported device list — say why instead
                    // of letting a dead button read as a hang.
                    start.on_hover_text(t!("tracking.start_camera_no_device"));
                }
            }

            if !active
                && tonal_button(
                    ui,
                    Some(ic::REFRESH),
                    &t!("tracking.refresh"),
                    ButtonTone::Primary,
                    true,
                )
                .clicked()
            {
                if let Some(err) = state.rescan_cameras(true, std::time::Duration::from_secs(3)) {
                    state.push_warning_notification(err);
                }
            }
        });

        // One-line status; details only when something needs attention.
        let (status_color, status_text) = if active && !ready {
            (color::WARNING, t!("tracking.preparing_camera").to_string())
        } else if active && tracking_on {
            (color::SUCCESS, t!("tracking.running").to_string())
        } else if active {
            (
                color::WARNING,
                t!("tracking.camera_active_paused").to_string(),
            )
        } else {
            (color::ON_SURFACE_MUTED, t!("tracking.stopped").to_string())
        };
        ui.label(egui::RichText::new(status_text).color(status_color));

        draw_camera_devices(ui, state);

        // Camera on but solve paused (pause hotkey) — offer the way back
        // right where the amber status is shown.
        if active
            && !tracking_on
            && tonal_button(
                ui,
                Some(ic::PLAY),
                &t!("tracking.resume_tracking"),
                ButtonTone::Primary,
                true,
            )
            .clicked()
        {
            state.tracking.toggle_tracking = true;
        }

        if let Some(tracking) = &state.app.last_tracking_pose {
            draw_confidence_bar(
                ui,
                &t!("tracking.confidence_overall"),
                tracking.overall_confidence,
            );
        }
    });
}

/// Detected-camera list under the start button: which RealSense device
/// the OS/driver answers with, its USB link, and — when it can't drive
/// tracking — why. This is the "is my camera even seen?" answer that
/// used to arrive only as a blocking error dialog after pressing Start.
/// With several D400s connected, each row is a radio that records the
/// pick (serial, persisted in `settings.json`) used by the next start.
fn draw_camera_devices(ui: &mut egui::Ui, state: &mut GuiApp) {
    use crate::gui::components::status_dot_label;

    ui.add_space(4.0);
    match state.tracking.available_cameras.as_deref() {
        None => {
            status_dot_label(ui, color::ON_SURFACE_MUTED, &t!("tracking.camera_scanning"));
        }
        Some([]) => {
            status_dot_label(ui, color::ERROR, &t!("tracking.camera_detected_none"));
            ui.label(
                egui::RichText::new(t!("tracking.camera_detected_none_hint"))
                    .small()
                    .color(color::ON_SURFACE_VARIANT),
            );
        }
        Some(cameras) => {
            let selected_serial = state.tracking.camera_serial.clone();
            let first_usable = cameras
                .iter()
                .find(|c| c.supported && !crate::tracking::usb_link_too_slow(c.usb_type.as_deref()))
                .map(|c| c.serial.clone());
            let usable_count = cameras
                .iter()
                .filter(|c| {
                    c.supported && !crate::tracking::usb_link_too_slow(c.usb_type.as_deref())
                })
                .count();

            for cam in cameras {
                let usb_slow = crate::tracking::usb_link_too_slow(cam.usb_type.as_deref());
                let usable = cam.supported && !usb_slow;
                let usb = match cam.usb_type.as_deref() {
                    Some(u) => t!("tracking.camera_usb_link", usb = u).to_string(),
                    None => t!("tracking.camera_usb_unknown").to_string(),
                };
                let mut caption = format!(
                    "{}  ·  {}  ·  {}",
                    cam.name,
                    t!("tracking.camera_serial", serial = cam.serial),
                    usb
                );
                if usb_slow {
                    caption.push_str(&format!("  ·  {}", t!("tracking.camera_usb2_warning")));
                } else if !cam.supported {
                    caption.push_str(&format!("  ·  {}", t!("tracking.camera_not_d400")));
                }

                if cam.supported {
                    // Radio selects WHICH D400 to open; the pick rides
                    // along to `RealSenseCapture::open` at the next
                    // start. No explicit pick = first usable (implicit
                    // selection mirrors the backend's fallback).
                    // A USB-2 device can't stream, so its radio stays
                    // visible (the row explains why) but unselectable.
                    let checked = match selected_serial.as_deref() {
                        Some(s) => s == cam.serial,
                        None => first_usable.as_deref() == Some(cam.serial.as_str()),
                    };
                    let response =
                        ui.add_enabled(usable, egui::RadioButton::new(checked, &caption));
                    if usable && response.clicked() {
                        state.tracking.camera_serial = Some(cam.serial.clone());
                        state.project_status.app_settings_dirty = true;
                    }
                } else {
                    let dot = color::ON_SURFACE_MUTED;
                    status_dot_label(ui, dot, &caption);
                }
            }

            if usable_count > 1 && selected_serial.is_none() {
                ui.label(
                    egui::RichText::new(t!("tracking.camera_first_used"))
                        .small()
                        .color(color::ON_SURFACE_VARIANT),
                );
            }
            if state.is_tracking_active() && usable_count > 0 {
                ui.label(
                    egui::RichText::new(t!("tracking.camera_selection_pending"))
                        .small()
                        .color(color::ON_SURFACE_VARIANT),
                );
            }
        }
    }
}

/// ③ Which body parts drive the avatar.
fn draw_body_parts(ui: &mut egui::Ui, state: &mut GuiApp) {
    collapsible_card(
        ui,
        "tracking.body_parts",
        t!("tracking.body_parts"),
        true,
        |ui| {
            ui.checkbox(
                &mut state.tracking.hand_tracking_enabled,
                t!("tracking.hand_tracking"),
            )
            .changed();
            ui.checkbox(
                &mut state.tracking.face_tracking_enabled,
                t!("tracking.face_tracking"),
            )
            .changed();
            ui.checkbox(
                &mut state.tracking.lower_body_tracking_enabled,
                t!("tracking.lower_body"),
            )
            .on_hover_text(t!("tracking.lower_body_tooltip"))
            .changed();
            ui.checkbox(
                &mut state.tracking.root_translation_enabled,
                t!("tracking.root_translation"),
            )
            .on_hover_text(t!("tracking.root_translation_tooltip"))
            .changed();
            ui.checkbox(
                &mut state.tracking.fade_on_tracking_loss,
                t!("tracking.fade_on_loss"),
            )
            .on_hover_text(t!("tracking.fade_on_loss_tooltip"))
            .changed();
        },
    );
}

/// ④ Capture format — combo edits are staged and only take effect on
/// Apply (or the next Start), never by restarting mid-click-through.
fn draw_input_device(ui: &mut egui::Ui, state: &mut GuiApp) {
    collapsible_card(
        ui,
        "tracking.input_device",
        t!("tracking.input_device"),
        false,
        |ui| {
            #[cfg(feature = "realsense")]
            {
                // D435-exclusive capture: the sole input is the Intel
                // RealSense D435. It self-selects the first D400 device and
                // supplies its own color-aligned metric depth to the pose
                // pipeline — there is no device / backend to choose.
                ui.label(egui::RichText::new(t!("tracking.backend_realsense_hint")).small());
                ui.add_space(4.0);
            }
            let unknown = t!("tracking.option_unknown");
            egui::ComboBox::from_label(t!("tracking.resolution"))
                .selected_text(
                    ["640x480", "1280x720", "1920x1080"]
                        .get(state.tracking.camera_resolution_index)
                        .map(|s| s.to_string())
                        .unwrap_or(unknown),
                )
                .show_ui(ui, |ui| {
                    // Highest-first, matching the Output panel's ordering.
                    // Display order only — the GUI keeps combo positions
                    // 0=640, 1=1280, 2=1920; projects persist the real
                    // width/height values, not these indices.
                    ui.selectable_value(
                        &mut state.tracking.camera_resolution_index,
                        2,
                        "1920x1080",
                    );
                    ui.selectable_value(&mut state.tracking.camera_resolution_index, 1, "1280x720");
                    ui.selectable_value(&mut state.tracking.camera_resolution_index, 0, "640x480");
                });
            let unknown = t!("tracking.option_unknown");
            egui::ComboBox::from_label(t!("tracking.frame_rate"))
                .selected_text(
                    ["30 fps", "60 fps"]
                        .get(state.tracking.camera_framerate_index)
                        .map(|s| s.to_string())
                        .unwrap_or(unknown),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.tracking.camera_framerate_index, 0, "30 fps");
                    ui.selectable_value(&mut state.tracking.camera_framerate_index, 1, "60 fps");
                });

            // While running, combo edits are *pending* until Apply — the
            // camera restart takes seconds, so crossing two entries must
            // not restart twice. The format the camera actually started
            // with lives in egui temp memory (seeded on Start; if absent —
            // e.g. after an egui memory reset — the current selection is
            // treated as applied, which errs on not offering a stale Apply).
            if state.app.is_tracking_running() {
                let current = (
                    state.tracking.camera_resolution_index,
                    state.tracking.camera_framerate_index,
                );
                let applied = ui
                    .ctx()
                    .data_mut(|d| *d.get_temp_mut_or(applied_format_id(), current));
                if applied != current {
                    ui.label(
                        egui::RichText::new(t!("tracking.format_pending_hint"))
                            .small()
                            .color(color::ON_SURFACE_VARIANT),
                    );
                    if filled_button(ui, Some(ic::REFRESH), &t!("tracking.apply_format"), true)
                        .clicked()
                    {
                        let (w, h, fps) = effective_capture_params(state);
                        let pipeline = state.tracking.pipeline_config();
                        let camera_serial = state.tracking.camera_serial.clone();
                        state
                            .app
                            .start_tracking_with_params(w, h, fps, camera_serial, pipeline);
                        ui.ctx()
                            .data_mut(|d| d.insert_temp(applied_format_id(), current));
                        state.push_notification(t!(
                            "tracking.camera_restarted",
                            w = w,
                            h = h,
                            fps = fps
                        ));
                    }
                }
            }
        },
    );
}

/// ⑤ Display options — both mirror concepts in one place, each with a
/// caption spelling out what it flips (they used to live in different
/// sections with no explanation of the difference).
fn draw_display_options(ui: &mut egui::Ui, state: &mut GuiApp) {
    collapsible_card(
        ui,
        "tracking.display_options",
        t!("tracking.display_options"),
        false,
        |ui| {
            #[cfg(feature = "realsense")]
            {
                // 1:1 sensor-matched mirror render (needs the D435
                // intrinsics, so it is realsense-only).
                ui.checkbox(&mut state.mirror_view, t!("tracking.mirror_view"));
                ui.label(egui::RichText::new(t!("tracking.mirror_view_hint")).small());
                ui.add_space(4.0);
            }
            ui.checkbox(
                &mut state.tracking.tracking_mirror,
                t!("tracking.mirror_preview"),
            )
            .changed();
            ui.label(egui::RichText::new(t!("tracking.mirror_preview_hint")).small());
            ui.add_space(4.0);
            ui.checkbox(
                &mut state.viewport.show_camera_wipe,
                t!("tracking.camera_wipe"),
            );
            if state.viewport.show_camera_wipe {
                ui.checkbox(
                    &mut state.viewport.show_detection_annotations,
                    t!("tracking.show_annotations"),
                );
            }
        },
    );
}

fn draw_lipsync(ui: &mut egui::Ui, state: &mut GuiApp) {
    // Phase D: bind UI to user-requested state, not runtime state. If the
    // audio device fails to open the runtime stays disabled, but the
    // checkbox keeps the user's intent so a mic refresh can retry.
    let active_mic = state.app.lipsync_mic_device_index();
    let requested_enabled = state.app.requested_lipsync_enabled();
    let runtime_enabled = state.app.is_lipsync_enabled();

    let mut new_mic = active_mic;
    let mut refresh_clicked = false;

    ui.horizontal(|ui| {
        let selected_mic = state
            .lipsync
            .available_mics
            .iter()
            .find(|m| m.index == active_mic)
            .map(|m| m.name.clone())
            .unwrap_or_else(|| t!("tracking.device_n", index = active_mic).to_string());
        egui::ComboBox::from_label(t!("tracking.microphone"))
            .selected_text(selected_mic)
            .show_ui(ui, |ui| {
                if state.lipsync.available_mics.is_empty() {
                    ui.label(t!("tracking.no_mics"));
                } else {
                    for mic in &state.lipsync.available_mics {
                        ui.selectable_value(&mut new_mic, mic.index, &mic.name);
                    }
                }
            });
        if tonal_button(
            ui,
            Some(ic::REFRESH),
            &t!("tracking.refresh"),
            ButtonTone::Primary,
            true,
        )
        .clicked()
        {
            refresh_clicked = true;
        }
    });

    if refresh_clicked {
        state.lipsync.available_mics = crate::lipsync::audio_capture::list_audio_devices();
    }

    if new_mic != active_mic {
        match state.app.set_requested_lipsync(requested_enabled, new_mic) {
            Ok(()) => {}
            Err(e) => {
                // Mic device init failed — surface the error but
                // do *not* mark the project dirty; nothing in
                // persistable state actually changed.
                state.push_notification(t!("tracking.lip_sync_mic_failed", error = e.to_string()));
            }
        }
    }

    ui.add_space(4.0);

    let mut new_enabled = requested_enabled;
    if ui
        .checkbox(&mut new_enabled, t!("tracking.enable_lip_sync"))
        .changed()
    {
        match state.app.set_requested_lipsync(new_enabled, active_mic) {
            Ok(()) => {
                if new_enabled {
                    state.push_notification(t!("tracking.lip_sync_started").to_string());
                } else {
                    state.push_notification(t!("tracking.lip_sync_stopped").to_string());
                }
            }
            Err(e) => {
                // Same rationale as the mic-change Err arm above —
                // a failed enable/disable is not a saveable change.
                state.push_notification(t!("tracking.lip_sync_failed", error = e.to_string()));
            }
        }
    }

    if requested_enabled && !runtime_enabled {
        ui.colored_label(color::WARNING, t!("tracking.requested_but_failed"));
    }

    // Per-frame lipsync inference is owned by `Application::step_lipsync`
    // and called from `GuiApp::update` so it keeps running even when this
    // panel is collapsed. The volume meter value (`state.lipsync.current_volume`)
    // is also written by `GuiApp::update` from `step_lipsync`'s return value.

    #[cfg(not(feature = "lipsync"))]
    {
        if requested_enabled {
            // Without the feature, set_requested_lipsync is a no-op stub
            // that returns Ok, so just call it to reset App state.
            let _ = state.app.set_requested_lipsync(false, active_mic);
            state.push_notification(t!("tracking.lip_sync_unavailable").to_string());
        }
    }

    // Volume meter.
    let vol = state.lipsync.current_volume;
    let meter_rect = ui.available_rect_before_wrap();
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(meter_rect.width().min(200.0), 14.0),
        egui::Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, color::SURFACE_VARIANT);
    let fill_w = (vol * 10.0).clamp(0.0, 1.0) * rect.width();
    let fill_color = if vol > state.lipsync.volume_threshold {
        color::SUCCESS
    } else {
        color::ON_SURFACE_MUTED
    };
    painter.rect_filled(
        egui::Rect::from_min_size(rect.min, egui::vec2(fill_w, rect.height())),
        2.0,
        fill_color,
    );
    ui.label(t!("tracking.volume", vol = format!("{:.3}", vol)));

    ui.add_space(4.0);
    ui.add(
        egui::Slider::new(&mut state.lipsync.volume_threshold, 0.001..=0.1)
            .text(t!("tracking.threshold"))
            .logarithmic(true),
    )
    .changed();
    ui.add(
        egui::Slider::new(&mut state.lipsync.smoothing, 0.0..=1.0).text(t!("tracking.smoothing")),
    )
    .changed();

    // Mouth source: how the audio lip-sync and the camera (FaceMesh) mouth
    // visemes combine. Both = whichever is stronger, so the mouth opens for
    // speech or a visibly open mouth without one silently overriding the other.
    use crate::tracking::MouthSource;
    let mut ms = state.lipsync.mouth_source;
    let label = |m: MouthSource| match m {
        MouthSource::Audio => t!("tracking.mouth_source_audio"),
        MouthSource::Image => t!("tracking.mouth_source_image"),
        MouthSource::Both => t!("tracking.mouth_source_both"),
    };
    egui::ComboBox::from_label(t!("tracking.mouth_source"))
        .selected_text(label(ms))
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut ms, MouthSource::Both, label(MouthSource::Both));
            ui.selectable_value(&mut ms, MouthSource::Image, label(MouthSource::Image));
            ui.selectable_value(&mut ms, MouthSource::Audio, label(MouthSource::Audio));
        });
    if ms != state.lipsync.mouth_source {
        state.lipsync.mouth_source = ms;
    }
}

fn draw_confidence_bar(ui: &mut egui::Ui, label: &str, value: f32) {
    ui.horizontal(|ui| {
        ui.label(format!("{:>10}", label));
        let color = if value > 0.8 {
            color::SUCCESS
        } else if value > 0.5 {
            color::WARNING
        } else {
            color::ERROR
        };
        ui.label(egui::RichText::new(format!("{:.0}%", value * 100.0)).color(color));
    });
}

#[cfg(test)]
mod tests {
    use super::clamp_for_safe_mode;

    #[test]
    fn safe_mode_caps_resolution_and_fps() {
        assert_eq!(clamp_for_safe_mode(1920, 1080, 60, true), (1280, 720, 30));
        assert_eq!(clamp_for_safe_mode(640, 480, 30, true), (640, 480, 30));
    }

    #[test]
    fn normal_mode_passes_format_through() {
        assert_eq!(clamp_for_safe_mode(1920, 1080, 60, false), (1920, 1080, 60));
    }
}
