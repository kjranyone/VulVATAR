use eframe::egui;

use crate::gui::GuiApp;

pub fn draw(ctx: &egui::Context, state: &mut GuiApp) {
    egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            let avatar_label = match state.app.active_avatar() {
                Some(avatar) => avatar.asset.source_path.to_string_lossy().to_string(),
                None => "No avatar".to_string(),
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
                        format!("Tracking: Active ({})", backend_label),
                    )
                } else {
                    (
                        egui::Color32::YELLOW,
                        format!("Tracking: Paused ({})", backend_label),
                    )
                }
            } else {
                (egui::Color32::GRAY, "Tracking: Stopped".to_string())
            };
            ui.label(egui::RichText::new(tracking_text).color(tracking_color));

            ui.separator();

            let sink_names = [
                "Virtual Camera",
                "Shared Texture",
                "Shared Memory",
                "Image Sequence",
            ];
            let sink_label = sink_names
                .get(state.app.output.active_sink().to_gui_index())
                .unwrap_or(&"Unknown");
            ui.label(format!("Output: {}", sink_label));

            ui.separator();

            ui.label(format!(
                "Queue: {}  Dropped: {}",
                state.app.output.queue_depth(),
                state.app.output.dropped_count()
            ));

            ui.separator();

            ui.label(format!("Frame: {}", state.frame_count));

            ui.separator();

            ui.label(format!(
                "FPS: {:.0}  ({:.1} ms)",
                state.fps, state.frame_time_ms
            ));

            // E12: Dirty indicator
            if state.project_dirty {
                ui.separator();
                ui.label(egui::RichText::new("* Modified").color(egui::Color32::YELLOW));
            }

            if state.paused {
                ui.label(
                    egui::RichText::new("PAUSED")
                        .color(egui::Color32::YELLOW)
                        .strong(),
                );
            }
        });
    });
}
