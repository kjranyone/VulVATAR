use eframe::egui;

use crate::gui::components::{filled_button, icon_button, outlined_button};
use crate::gui::theme::{color, icon as ic};
use crate::gui::GuiApp;
use crate::t;

pub(super) fn draw_tracking(ui: &mut egui::Ui, state: &mut GuiApp) {
    if ui
        .checkbox(&mut state.tracking.toggle_tracking, t!("tracking.enabled"))
        .changed()
    {
        state.project_dirty = true;
    }
    ui.add_space(4.0);

    egui::CollapsingHeader::new(t!("tracking.input_device"))
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let prev_camera = state.camera_index;
                let selected_text = state
                    .available_cameras
                    .iter()
                    .find(|c| c.index == state.camera_index)
                    .map(|c| format!("{}: {}", c.index, c.name))
                    .unwrap_or_else(|| t!("tracking.camera_n", index = state.camera_index).to_string());
                egui::ComboBox::from_label(t!("tracking.camera"))
                    .selected_text(selected_text)
                    .show_ui(ui, |ui| {
                        if state.available_cameras.is_empty() {
                            ui.label(t!("tracking.no_cameras"));
                        } else {
                            for cam in &state.available_cameras {
                                let label = format!("{}: {}", cam.index, cam.name);
                                ui.selectable_value(&mut state.camera_index, cam.index, label);
                            }
                        }
                    });
                if state.camera_index != prev_camera {
                    state.project_dirty = true;
                }
                if icon_button(ui, ic::REFRESH, &t!("tracking.refresh")).clicked() {
                    state.available_cameras = crate::tracking::list_cameras();
                    if !state
                        .available_cameras
                        .iter()
                        .any(|c| c.index == state.camera_index)
                    {
                        state.camera_index = state
                            .available_cameras
                            .first()
                            .map(|c| c.index)
                            .unwrap_or(0);
                    }
                }
            });
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let active = state.is_tracking_active();
                let ready = state.is_tracking_ready();
                if active && ready {
                    if outlined_button(
                        ui,
                        Some(ic::PAUSE),
                        &t!("tracking.stop_camera"),
                        color::ERROR,
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
                } else if filled_button(ui, Some(ic::PLAY), &t!("tracking.start_camera"), true)
                    .clicked()
                {
                    let (w, h) = crate::gui::camera_resolution_for_index(
                        state.tracking.camera_resolution_index,
                    );
                    let fps =
                        crate::gui::camera_fps_for_index(state.tracking.camera_framerate_index);
                    #[cfg(feature = "webcam")]
                    let backend = crate::tracking::CameraBackend::Webcam {
                        camera_index: state.camera_index,
                    };
                    #[cfg(not(feature = "webcam"))]
                    let backend = crate::tracking::CameraBackend::Synthetic;
                    state.app.start_tracking_with_params_full(
                        backend,
                        w,
                        h,
                        fps,
                        state.tracking.lower_body_tracking_enabled,
                    );
                }
            });
            if ui
                .checkbox(&mut state.show_camera_wipe, t!("tracking.camera_wipe"))
                .changed()
            {
                state.project_dirty = true;
            }
            if state.show_camera_wipe
                && ui
                    .checkbox(&mut state.show_detection_annotations, t!("tracking.show_annotations"))
                    .changed()
            {
                state.project_dirty = true;
            }
            let prev_res = state.tracking.camera_resolution_index;
            egui::ComboBox::from_label(t!("tracking.resolution"))
                .selected_text(
                    *["640x480", "1280x720", "1920x1080"]
                        .get(state.tracking.camera_resolution_index)
                        .unwrap_or(&"Unknown"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.tracking.camera_resolution_index, 0, "640x480");
                    ui.selectable_value(&mut state.tracking.camera_resolution_index, 1, "1280x720");
                    ui.selectable_value(
                        &mut state.tracking.camera_resolution_index,
                        2,
                        "1920x1080",
                    );
                });
            let res_changed = state.tracking.camera_resolution_index != prev_res;
            if res_changed {
                state.project_dirty = true;
            }
            let prev_fps = state.tracking.camera_framerate_index;
            egui::ComboBox::from_label(t!("tracking.frame_rate"))
                .selected_text(
                    *["30 fps", "60 fps"]
                        .get(state.tracking.camera_framerate_index)
                        .unwrap_or(&"Unknown"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.tracking.camera_framerate_index, 0, "30 fps");
                    ui.selectable_value(&mut state.tracking.camera_framerate_index, 1, "60 fps");
                });
            let fps_changed = state.tracking.camera_framerate_index != prev_fps;
            if fps_changed {
                state.project_dirty = true;
            }
            // Phase B-1: if the webcam is currently running, restart it with
            // the new params so the change takes effect immediately rather
            // than only at the next Stop / Start Camera cycle.
            if (res_changed || fps_changed) && state.app.is_tracking_running() {
                let (w, h) =
                    crate::gui::camera_resolution_for_index(state.tracking.camera_resolution_index);
                let fps = crate::gui::camera_fps_for_index(state.tracking.camera_framerate_index);
                #[cfg(feature = "webcam")]
                let backend = crate::tracking::CameraBackend::Webcam {
                    camera_index: state.camera_index,
                };
                #[cfg(not(feature = "webcam"))]
                let backend = crate::tracking::CameraBackend::Synthetic;
                state.app.start_tracking_with_params_full(
                    backend,
                    w,
                    h,
                    fps,
                    state.tracking.lower_body_tracking_enabled,
                );
                state.push_notification(t!("tracking.camera_restarted", w = w, h = h, fps = fps));
            }
        });

    egui::CollapsingHeader::new(t!("tracking.capture_format"))
        .default_open(false)
        .show(ui, |ui| {
            if ui
                .checkbox(&mut state.tracking.tracking_mirror, t!("tracking.mirror_preview"))
                .changed()
            {
                state.project_dirty = true;
            }
        });

    egui::CollapsingHeader::new(t!("tracking.inference_status"))
        .default_open(true)
        .show(ui, |ui| {
            let camera_running = state.is_tracking_active();
            let camera_ready = state.is_tracking_ready();
            let tracking_on = state.tracking.toggle_tracking;
            let (status_color, status_text) = if camera_running && !camera_ready {
                (egui::Color32::YELLOW, t!("tracking.preparing_camera").to_string())
            } else if camera_running && tracking_on {
                (egui::Color32::GREEN, t!("tracking.running").to_string())
            } else if camera_running {
                (egui::Color32::YELLOW, t!("tracking.camera_active_paused").to_string())
            } else {
                (egui::Color32::GRAY, t!("tracking.stopped").to_string())
            };
            ui.label(egui::RichText::new(status_text).color(status_color));
            if camera_running {
                let camera_label = state
                    .app
                    .tracking_worker
                    .as_ref()
                    .map(|w| w.active_backend().label())
                    .unwrap_or("Unknown");
                ui.label(t!("tracking.camera_label", label = camera_label.to_string()));

                // Inference backend (ONNX execution provider) — surfaces a
                // silent CPU fallback when DirectML registration fails. Set
                // by the worker once CIGPose finishes loading; absent during
                // synthetic mode or before init completes.
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
                ui.label(format!("Timestamp: {}", tracking.source_timestamp));
            } else {
                ui.label(t!("tracking.no_tracking_data"));
            }
        });

    egui::CollapsingHeader::new(t!("tracking.retargeting"))
        .default_open(true)
        .show(ui, |ui| {
            if ui
                .checkbox(&mut state.tracking.hand_tracking_enabled, t!("tracking.hand_tracking"))
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .checkbox(&mut state.tracking.face_tracking_enabled, t!("tracking.face_tracking"))
                .changed()
            {
                state.project_dirty = true;
            }
            let lower_body_resp = ui
                .checkbox(
                    &mut state.tracking.lower_body_tracking_enabled,
                    t!("tracking.lower_body"),
                )
                .on_hover_text(
                    t!("tracking.lower_body_tooltip"),
                );
            if lower_body_resp.changed() {
                state.project_dirty = true;
            }
            if ui
                .checkbox(
                    &mut state.tracking.root_translation_enabled,
                    t!("tracking.root_translation"),
                )
                .on_hover_text(t!("tracking.root_translation_tooltip"))
                .changed()
            {
                state.project_dirty = true;
            }
            ui.separator();
            ui.horizontal(|ui| {
                // Pose calibration entry — opens the fullscreen modal
                // directly. Mode selection (Full Body / Upper Body)
                // happens *inside* the modal via a Segmented Buttons
                // control, which fits Material Design better than a
                // menu_button → modal-launch chain (M3 considers the
                // latter an anti-pattern: too much depth, indirect
                // entry point, mode-without-context-of-modal).
                //
                // Default-open in FullBody mode; the user flips the
                // segment if they only have an upper-body crop. The
                // segment is editable up until Capture begins
                // (`Collecting` state); the modal disables the
                // toggle once a sample window has started.
                if filled_button(ui, None, &t!("calibration.button"), true).clicked() {
                    // Reopen at the mode the active profile was last
                    // calibrated with, falling back to FullBody for a
                    // first-time capture (or after a profile that
                    // never completed calibration). The Segmented
                    // Buttons inside the modal still let the user
                    // override before HoldStill expires — this just
                    // saves the round-trip for the common case where
                    // they're recalibrating in the same configuration.
                    let default_mode = state
                        .app
                        .tracking_calibration
                        .pose
                        .as_ref()
                        .map(|c| c.mode)
                        .unwrap_or(crate::tracking::CalibrationMode::FullBody);
                    state.calibration_modal.open(default_mode);
                }
            });
            draw_calibration_status(ui, state);
            if let Some(tracking) = &state.app.last_tracking_pose {
                ui.separator();
                ui.label(t!("tracking.confidence"));
                draw_confidence_bar(ui, &t!("tracking.confidence_overall"), tracking.overall_confidence);
                if let Some(face) = tracking.face {
                    draw_confidence_bar(ui, &t!("tracking.confidence_face"), face.confidence);
                }
                use crate::asset::HumanoidBone;
                let joint_conf = |b: HumanoidBone| -> f32 {
                    tracking.joints.get(&b).map(|j| j.confidence).unwrap_or(0.0)
                };
                draw_confidence_bar(ui, &t!("tracking.confidence_left_shoulder"), joint_conf(HumanoidBone::LeftShoulder));
                draw_confidence_bar(
                    ui,
                    &t!("tracking.confidence_right_shoulder"),
                    joint_conf(HumanoidBone::RightShoulder),
                );
                draw_confidence_bar(ui, &t!("tracking.confidence_left_hand"), joint_conf(HumanoidBone::LeftHand));
                draw_confidence_bar(ui, &t!("tracking.confidence_right_hand"), joint_conf(HumanoidBone::RightHand));
            }
        });

    egui::CollapsingHeader::new(t!("tracking.lip_sync"))
        .default_open(false)
        .show(ui, |ui| {
            draw_lipsync(ui, state);
        });
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
        if outlined_button(
            ui,
            Some(ic::REFRESH),
            &t!("tracking.refresh"),
            color::PRIMARY,
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
        if let Err(e) = state.app.set_requested_lipsync(requested_enabled, new_mic) {
            state.push_notification(t!("tracking.lip_sync_mic_failed", error = e.to_string()));
        }
        state.project_dirty = true;
    }

    ui.add_space(4.0);

    let mut new_enabled = requested_enabled;
    if ui.checkbox(&mut new_enabled, t!("tracking.enable_lip_sync")).changed() {
        match state.app.set_requested_lipsync(new_enabled, active_mic) {
            Ok(()) => {
                state.project_dirty = true;
                if new_enabled {
                    state.push_notification(t!("tracking.lip_sync_started").to_string());
                } else {
                    state.push_notification(t!("tracking.lip_sync_stopped").to_string());
                }
            }
            Err(e) => {
                state.push_notification(t!("tracking.lip_sync_failed", error = e.to_string()));
                state.project_dirty = true;
            }
        }
    }

    if requested_enabled && !runtime_enabled {
        ui.colored_label(
            egui::Color32::from_rgb(220, 160, 80),
            t!("tracking.requested_but_failed"),
        );
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
            state.push_notification(
                t!("tracking.lip_sync_unavailable").to_string(),
            );
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
    painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(30, 30, 35));
    let fill_w = (vol * 10.0).clamp(0.0, 1.0) * rect.width();
    let fill_color = if vol > state.lipsync.volume_threshold {
        egui::Color32::from_rgb(80, 200, 80)
    } else {
        egui::Color32::from_rgb(60, 60, 70)
    };
    painter.rect_filled(
        egui::Rect::from_min_size(rect.min, egui::vec2(fill_w, rect.height())),
        2.0,
        fill_color,
    );
    ui.label(t!("tracking.volume", vol = format!("{:.3}", vol)));

    ui.add_space(4.0);
    if ui
        .add(
            egui::Slider::new(&mut state.lipsync.volume_threshold, 0.001..=0.1)
                .text(t!("tracking.threshold"))
                .logarithmic(true),
        )
        .changed()
    {
        state.project_dirty = true;
    }
    if ui
        .add(egui::Slider::new(&mut state.lipsync.smoothing, 0.0..=1.0).text(t!("tracking.smoothing")))
        .changed()
    {
        state.project_dirty = true;
    }
}

/// Renders the pose-calibration status line(s) under the
/// Calibrate Pose ▼ button. Three rendering paths:
///
/// 1. **Calibrated + matching anchor** (the happy case): green
///    "Calibrated 5 min ago — Full Body" + a depth/jitter/sample
///    line so the user can sanity-check the captured values.
/// 2. **Calibrated + mismatched anchor**: amber warning when the
///    runtime tracker is currently using the *other* anchor (e.g.
///    calibrated in T-pose but the user is now seated and only the
///    shoulders are visible). Solver auto-falls back to EMA in this
///    case; the warning lets the user re-calibrate intentionally.
/// 3. **Uncalibrated**: grey "Pose not calibrated — auto-EMA active"
///    so first-time users know the feature exists.
fn draw_calibration_status(ui: &mut egui::Ui, state: &mut GuiApp) {
    let pose = match state.app.tracking_calibration.pose.clone() {
        Some(p) => p,
        None => {
            ui.label(
                egui::RichText::new(t!("calibration.status_uncalibrated"))
                    .size(11.0)
                    .color(egui::Color32::from_rgb(170, 170, 170)),
            );
            return;
        }
    };

    let mode_label = match pose.mode {
        crate::tracking::CalibrationMode::FullBody => t!("calibration.mode_full_body"),
        crate::tracking::CalibrationMode::UpperBody => t!("calibration.mode_upper_body"),
    };
    let age_label = format_age(pose.captured_at_unix);

    // Mismatch detection: if the live tracker is producing an anchor
    // type that doesn't match the calibration mode, show an amber
    // warning instead of the green "all good" line. Mismatch is the
    // user's signal to either re-calibrate or change their framing.
    let runtime_mismatch = state
        .app
        .last_tracking_pose
        .as_ref()
        .filter(|p| p.root_offset.is_some())
        .map(|p| match pose.mode {
            crate::tracking::CalibrationMode::FullBody => !p.root_anchor_is_hip,
            crate::tracking::CalibrationMode::UpperBody => p.root_anchor_is_hip,
        })
        .unwrap_or(false);

    if runtime_mismatch {
        ui.label(
            egui::RichText::new(t!(
                "calibration.status_mismatch",
                age = age_label,
                mode = mode_label
            ))
            .size(11.0)
            .color(egui::Color32::from_rgb(230, 180, 60)),
        );
    } else {
        ui.label(
            egui::RichText::new(t!(
                "calibration.status_calibrated",
                age = age_label,
                mode = mode_label
            ))
            .size(11.0)
            .color(egui::Color32::from_rgb(120, 220, 140)),
        );
    }

    // Detail line: depth + jitter + sample count + confidence. Skipped
    // when the calibration is from rtmw3d (no depth) — only the sample
    // / confidence parts are meaningful there.
    let detail = match (pose.anchor_depth_m, pose.anchor_depth_jitter_m) {
        (Some(d), Some(j)) => t!(
            "calibration.status_detail_depth",
            depth = format!("{:.2}", d),
            jitter = format!("{:.3}", j),
            samples = format!("{}", pose.frame_count),
            confidence = format!("{:.2}", pose.confidence)
        ),
        _ => t!(
            "calibration.status_detail_no_depth",
            samples = format!("{}", pose.frame_count),
            confidence = format!("{:.2}", pose.confidence)
        ),
    };
    ui.label(
        egui::RichText::new(detail)
            .size(10.0)
            .color(egui::Color32::from_rgb(150, 150, 160)),
    );
}

/// Format a unix timestamp as a relative-age label ("5 min ago",
/// "just now", "1 day ago"). Future timestamps (clock skew) render as
/// "just now" rather than "in N min" to keep the status line from
/// looking broken.
fn format_age(captured_unix: u64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let age_secs = now.saturating_sub(captured_unix);
    if age_secs < 60 {
        t!("calibration.age_just_now")
    } else if age_secs < 3600 {
        t!("calibration.age_minutes", n = format!("{}", age_secs / 60))
    } else if age_secs < 86_400 {
        t!("calibration.age_hours", n = format!("{}", age_secs / 3600))
    } else {
        t!("calibration.age_days", n = format!("{}", age_secs / 86_400))
    }
}

fn draw_confidence_bar(ui: &mut egui::Ui, label: &str, value: f32) {
    ui.horizontal(|ui| {
        ui.label(format!("{:>10}", label));
        let color = if value > 0.8 {
            egui::Color32::GREEN
        } else if value > 0.5 {
            egui::Color32::YELLOW
        } else {
            egui::Color32::from_rgb(255, 100, 100)
        };
        ui.label(egui::RichText::new(format!("{:.0}%", value * 100.0)).color(color));
    });
}
