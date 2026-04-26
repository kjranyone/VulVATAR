use eframe::egui;

use crate::gui::GuiApp;
use crate::t;

pub fn draw(ctx: &egui::Context, state: &mut GuiApp) {
    egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            let avatar_label = match state.app.active_avatar() {
                Some(avatar) => avatar.asset.source_path.to_string_lossy().to_string(),
                None => t!("app.no_avatar"),
            };
            ui.label(&avatar_label);

            ui.separator();

            let tracking_active = state.is_tracking_active();
            let tracking_enabled = state.tracking.toggle_tracking;
            let (tracking_color, tracking_text) = if tracking_active {
                let backend_label = state
                    .app
                    .tracking_worker
                    .as_ref()
                    .map(|w| w.active_backend().label())
                    .unwrap_or("Unknown");
                if tracking_enabled {
                    (
                        egui::Color32::GREEN,
                        t!("status.tracking_active", backend = backend_label.to_string()),
                    )
                } else {
                    (
                        egui::Color32::YELLOW,
                        t!("status.tracking_paused", backend = backend_label.to_string()),
                    )
                }
            } else {
                (egui::Color32::GRAY, t!("status.tracking_stopped"))
            };
            ui.label(egui::RichText::new(tracking_text).color(tracking_color));

            ui.separator();

            let sink_names: [String; 4] = [
                t!("status.sink_virtual_camera"),
                t!("status.sink_shared_texture"),
                t!("status.sink_shared_memory"),
                t!("status.sink_image_sequence"),
            ];
            let sink_label = sink_names
                .get(state.app.output.active_sink().to_gui_index())
                .cloned()
                .unwrap_or_else(|| t!("status.sink_unknown"));
            ui.label(t!("status.output_label", sink = sink_label.to_string()));

            ui.separator();

            ui.label(t!(
                "status.queue_dropped",
                queue = state.app.output.queue_depth(),
                dropped = state.app.output.dropped_count()
            ));

            ui.separator();

            ui.label(t!("status.frame", frame = state.frame_count));

            ui.separator();

            ui.label(t!(
                "status.fps",
                fps = format!("{:.0}", state.fps),
                ms = format!("{:.1}", state.frame_time_ms)
            ));

            // E12: Dirty indicator
            if state.project_dirty {
                ui.separator();
                ui.label(egui::RichText::new(t!("status.modified")).color(egui::Color32::YELLOW));
            }

            if state.paused {
                ui.label(
                    egui::RichText::new(t!("status.paused"))
                        .color(egui::Color32::YELLOW)
                        .strong(),
                );
            }
        });
    });
}
