use eframe::egui;

use crate::gui::components::collapsible_card;
use crate::gui::theme::color;
use crate::gui::GuiApp;
use crate::t;

pub(super) fn draw_output(ui: &mut egui::Ui, state: &mut GuiApp) {
    let sink_names: [String; 4] = [
        t!("status.sink_virtual_camera"),
        t!("status.sink_shared_texture"),
        t!("status.sink_shared_memory"),
        t!("status.sink_image_sequence"),
    ];

    // Combo binds to the user's REQUESTED sink (Phase D). If the runtime
    // can't honor it (MF init failed), the request still persists so
    // autosave doesn't overwrite the user's choice with a fallback.
    use crate::output::FrameSink;
    let requested_idx = state.app.requested_sink().to_gui_index();
    let active_idx = state.app.output.active_sink().to_gui_index();
    let mut new_idx = requested_idx;

    collapsible_card(
        ui,
        "inspector.output_sink",
        t!("inspector.output_sink"),
        true,
        |ui| {
            egui::ComboBox::from_label(t!("inspector.output_sink_label"))
                .selected_text(
                    sink_names
                        .get(requested_idx)
                        .map(|s: &String| s.as_str())
                        .unwrap_or("Unknown"),
                )
                .show_ui(ui, |ui| {
                    for (i, name) in sink_names.iter().enumerate() {
                        ui.selectable_value(&mut new_idx, i, name);
                    }
                });
            if active_idx != requested_idx {
                ui.colored_label(
                    color::WARNING,
                    t!(
                        "inspector.sink_mismatch",
                        active = sink_names
                            .get(active_idx)
                            .map(|s: &String| s.as_str())
                            .unwrap_or("?"),
                        requested = sink_names
                            .get(requested_idx)
                            .map(|s: &String| s.as_str())
                            .unwrap_or("?"),
                    ),
                );
            }
        },
    );

    if new_idx != requested_idx {
        let want_sink = FrameSink::from_gui_index(new_idx);
        match state.app.set_requested_sink(want_sink) {
            Ok(()) => {
                state.push_notification(t!(
                    "inspector.output_sink_changed",
                    name = sink_names[new_idx].to_string()
                ));
            }
            Err(e) => {
                state.push_notification(t!("inspector.output_sink_failed", error = e.to_string()));
            }
        }
    }

    collapsible_card(
        ui,
        "inspector.frame_format",
        t!("inspector.frame_format"),
        true,
        |ui| {
            egui::ComboBox::from_label(t!("tracking.resolution"))
                .selected_text(
                    *["1920x1080", "1280x720", "640x480"]
                        .get(state.output.output_resolution_index)
                        .unwrap_or(&"Unknown"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.output.output_resolution_index, 0, "1920x1080");
                    ui.selectable_value(&mut state.output.output_resolution_index, 1, "1280x720");
                    ui.selectable_value(&mut state.output.output_resolution_index, 2, "640x480");
                });
            egui::ComboBox::from_label(t!("tracking.frame_rate"))
                .selected_text(
                    *["60 fps", "30 fps"]
                        .get(state.output.output_framerate_index)
                        .unwrap_or(&"Unknown"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.output.output_framerate_index, 0, "60 fps");
                    ui.selectable_value(&mut state.output.output_framerate_index, 1, "30 fps");
                });
            ui.checkbox(
                &mut state.output.output_has_alpha,
                t!("inspector.rgba_alpha"),
            );
            let cs_names: [String; 2] = [t!("inspector.srgb"), t!("inspector.linear_srgb")];
            egui::ComboBox::from_label(t!("inspector.color_space"))
                .selected_text(
                    cs_names
                        .get(state.output.output_color_space_index)
                        .map(|s: &String| s.as_str())
                        .unwrap_or("Unknown"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut state.output.output_color_space_index,
                        0,
                        &*cs_names[0],
                    );
                    ui.selectable_value(
                        &mut state.output.output_color_space_index,
                        1,
                        &*cs_names[1],
                    );
                });
            // Anti-aliasing (MSAA). Applies to both the viewport preview and
            // the exported frame (they share the offscreen render target). The
            // renderer clamps the pick to device sample-count support (and
            // caps integrated GPUs at 4x), so an unsupported pick degrades.
            let msaa_names: [String; 4] = [
                t!("inspector.msaa_off"),
                t!("inspector.msaa_2x"),
                t!("inspector.msaa_4x"),
                t!("inspector.msaa_8x"),
            ];
            egui::ComboBox::from_label(t!("inspector.antialiasing"))
                .selected_text(
                    msaa_names
                        .get(state.output.msaa_index)
                        .map(|s: &String| s.as_str())
                        .unwrap_or("Unknown"),
                )
                .show_ui(ui, |ui| {
                    for (i, name) in msaa_names.iter().enumerate() {
                        ui.selectable_value(&mut state.output.msaa_index, i, name);
                    }
                });
        },
    );

    let diagnostics = state.app.output.diagnostics();

    collapsible_card(
        ui,
        "inspector.synchronization",
        t!("inspector.synchronization"),
        false,
        |ui| {
            use crate::output::HandoffPath;
            // Reflect the most recently published frame's handoff_path
            // rather than guessing from the sink variant. Until any frame
            // has crossed the boundary, show "pending" instead of a green
            // "GPU active" badge — the previous static labels claimed GPU
            // interop even when every frame was a CPU readback.
            let (label_key, color) = match &diagnostics.active_handoff_path {
                Some(HandoffPath::GpuSharedFrame) => {
                    ("inspector.handoff_active_gpu", color::SUCCESS)
                }
                Some(HandoffPath::CpuReadback) => {
                    ("inspector.handoff_active_cpu_readback", color::WARNING)
                }
                Some(HandoffPath::SharedMemory) => {
                    ("inspector.handoff_active_shared_memory", color::WARNING)
                }
                None => ("inspector.handoff_pending", color::ON_SURFACE_MUTED),
            };
            ui.label(egui::RichText::new(t!(label_key)).color(color));
            if diagnostics.fallback_active {
                ui.label(
                    egui::RichText::new(t!("inspector.handoff_fallback_warning"))
                        .color(color::WARNING),
                );
                if let Some(reason) = diagnostics.fallback_reason.as_ref() {
                    use crate::output::FallbackReason;
                    let reason_key = match reason {
                        FallbackReason::RequestedCpuReadback => {
                            "inspector.handoff_fallback_reason_requested_cpu"
                        }
                        FallbackReason::ExternalHandleUnavailable => {
                            "inspector.handoff_fallback_reason_external_handle"
                        }
                        FallbackReason::ExportPoolSaturated => {
                            "inspector.handoff_fallback_reason_pool_saturated"
                        }
                        FallbackReason::MissingExternalHandle => {
                            "inspector.handoff_fallback_reason_missing_handle"
                        }
                    };
                    ui.label(egui::RichText::new(t!(reason_key)).color(color::WARNING));
                }
            }
        },
    );

    collapsible_card(
        ui,
        "inspector.diagnostics",
        t!("inspector.diagnostics"),
        true,
        |ui| {
            // "Connected" was unconditional green before — only consider the
            // pipeline connected once a frame has actually been published.
            let (label_key, color) = if diagnostics.last_publish_timestamp == 0 {
                ("inspector.connection_pending", color::ON_SURFACE_MUTED)
            } else {
                ("inspector.connected", color::SUCCESS)
            };
            ui.label(egui::RichText::new(t!(label_key)).color(color));
            ui.label(t!("inspector.queue_depth", depth = diagnostics.queue_depth));
            if let Some(pool) = diagnostics.export_pool {
                ui.label(t!(
                    "inspector.export_pool",
                    total = pool.total_slots,
                    capacity = pool.capacity,
                    available = pool.available_slots,
                    leased = pool.leased_slots,
                ));
            }
            ui.label(t!(
                "inspector.dropped_frames",
                count = diagnostics.dropped_frame_count
            ));
        },
    );

    collapsible_card(
        ui,
        "inspector.runtime_budget",
        t!("inspector.runtime_budget"),
        false,
        |ui| {
            let budget = &state.app.runtime_gpu_budget;
            use crate::app::runtime_gpu_budget::DegradedMode;
            let mode_color = match budget.degraded_mode() {
                DegradedMode::Healthy => color::SUCCESS,
                DegradedMode::PressureLight => color::WARNING,
                // Deep-amber "strong warning" tier between WARNING and
                // ERROR — reuses the warning container's on-colour so the
                // ramp stays inside the warning hue family.
                DegradedMode::PressureHeavy => color::ON_WARNING_CONTAINER,
                DegradedMode::EmergencyCpu => color::ERROR,
            };
            ui.label(
                egui::RichText::new(t!(
                    "inspector.runtime_budget_mode",
                    mode = budget.degraded_mode().label().to_string()
                ))
                .color(mode_color),
            );
            ui.label(t!(
                "inspector.runtime_budget_render_fps",
                target = budget.render_fps_target(),
                user = budget.user_render_fps(),
            ));
            ui.label(t!(
                "inspector.runtime_budget_pose_hz",
                hz = budget.pose_hz_target()
            ));
            ui.label(t!(
                "inspector.runtime_budget_depth_refresh",
                period = budget.depth_refresh_period()
            ));
            ui.label(t!(
                "inspector.runtime_budget_face_ep",
                ep = if budget.facemesh_prefers_cpu_ep() {
                    "CPU (budget)"
                } else {
                    "Auto"
                }
            ));
            ui.label(t!(
                "inspector.runtime_budget_reason",
                reason = budget.last_transition_reason().label().to_string()
            ));
            ui.label(t!(
                "inspector.runtime_budget_failures",
                count = budget.gpu_export_failure_window_count()
            ));
            ui.label(t!(
                "inspector.runtime_budget_dropped_results",
                count = state.app.render_results_dropped_count()
            ));
        },
    );
}
