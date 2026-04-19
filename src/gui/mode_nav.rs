use eframe::egui;

use crate::gui::{AppMode, GuiApp};

pub fn draw(ctx: &egui::Context, state: &mut GuiApp) {
    egui::SidePanel::left("mode_nav")
        .resizable(false)
        .default_width(140.0)
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.heading("Modes");
            ui.separator();
            ui.add_space(4.0);

            for mode in AppMode::ALL {
                let is_active = state.mode == mode;
                if ui.selectable_label(is_active, mode.label()).clicked() {
                    if is_active {
                        state.inspector_open = !state.inspector_open;
                    } else {
                        state.mode = mode;
                        state.inspector_open = true;
                    }
                }
            }

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(4.0);
            if ui
                .selectable_label(!state.inspector_open, "Hide Panel")
                .clicked()
            {
                state.inspector_open = false;
            }
        });
}
