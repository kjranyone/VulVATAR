use eframe::egui;

use crate::gui::GuiApp;

pub fn draw(ctx: &egui::Context, state: &mut GuiApp) {
    egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            let avatar_label = match &state.app.avatar {
                Some(avatar) => avatar.asset.source_path.to_string_lossy().to_string(),
                None => "No avatar".to_string(),
            };
            ui.label(&avatar_label);

            ui.separator();

            let tracking_color = if state.camera_running {
                egui::Color32::GREEN
            } else {
                egui::Color32::GRAY
            };
            let tracking_text = if state.camera_running {
                "Tracking: Active"
            } else {
                "Tracking: Stopped"
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
                .get(state.output_sink_index)
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
