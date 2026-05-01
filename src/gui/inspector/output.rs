use eframe::egui;

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

    egui::CollapsingHeader::new(t!("inspector.output_sink"))
        .default_open(true)
        .show(ui, |ui| {
            egui::ComboBox::from_label(t!("inspector.output_sink_label"))
                .selected_text(sink_names.get(requested_idx).map(|s: &String| s.as_str()).unwrap_or("Unknown"))
                .show_ui(ui, |ui| {
                    for (i, name) in sink_names.iter().enumerate() {
                        ui.selectable_value(&mut new_idx, i, name);
                    }
                });
            if active_idx != requested_idx {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 160, 80),
                    t!("inspector.sink_mismatch",
                        active = sink_names.get(active_idx).map(|s: &String| s.as_str()).unwrap_or("?"),
                        requested = sink_names.get(requested_idx).map(|s: &String| s.as_str()).unwrap_or("?"),
                    ),
                );
            }
        });

    if new_idx != requested_idx {
        let want_sink = FrameSink::from_gui_index(new_idx);
        match state.app.set_requested_sink(want_sink) {
            Ok(()) => {
                state.push_notification(t!("inspector.output_sink_changed", name = sink_names[new_idx].to_string()));
                state.project_dirty = true;
            }
            Err(e) => {
                state.push_notification(t!("inspector.output_sink_failed", error = e.to_string()));
                state.project_dirty = true;
            }
        }
    }

    let prev_res = state.output.output_resolution_index;
    let prev_fps = state.output.output_framerate_index;
    let prev_alpha = state.output.output_has_alpha;
    let prev_cs = state.output.output_color_space_index;

    egui::CollapsingHeader::new(t!("inspector.frame_format"))
        .default_open(true)
        .show(ui, |ui| {
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
            ui.checkbox(&mut state.output.output_has_alpha, t!("inspector.rgba_alpha"));
            let cs_names: [String; 2] = [t!("inspector.srgb"), t!("inspector.linear_srgb")];
            egui::ComboBox::from_label(t!("inspector.color_space"))
                .selected_text(
                    cs_names
                        .get(state.output.output_color_space_index)
                        .map(|s: &String| s.as_str())
                        .unwrap_or("Unknown"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.output.output_color_space_index, 0, &*cs_names[0]);
                    ui.selectable_value(
                        &mut state.output.output_color_space_index,
                        1,
                        &*cs_names[1],
                    );
                });
        });

    if state.output.output_resolution_index != prev_res
        || state.output.output_framerate_index != prev_fps
        || state.output.output_has_alpha != prev_alpha
        || state.output.output_color_space_index != prev_cs
    {
        state.project_dirty = true;
    }

    egui::CollapsingHeader::new(t!("inspector.synchronization"))
        .default_open(false)
        .show(ui, |ui| {
            ui.label(t!("inspector.handoff_gpu"));
            ui.label(t!("inspector.fallback_cpu"));
            ui.label(egui::RichText::new(t!("inspector.gpu_active")).color(egui::Color32::GREEN));
        });

    egui::CollapsingHeader::new(t!("inspector.diagnostics"))
        .default_open(true)
        .show(ui, |ui| {
            ui.label(egui::RichText::new(t!("inspector.connected")).color(egui::Color32::GREEN));
            ui.label(t!("inspector.queue_depth", depth = state.app.output.queue_depth()));
            ui.label(t!("inspector.dropped_frames", count = state.app.output.dropped_count()));
        });
}
