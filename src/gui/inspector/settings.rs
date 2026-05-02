use eframe::egui;

use crate::gui::GuiApp;
use crate::t;

pub(super) fn draw_settings(ui: &mut egui::Ui, state: &mut GuiApp) {
    let locales = crate::i18n::available_locales();
    let current_locale = state.settings.locale.clone();

    egui::CollapsingHeader::new(t!("settings.heading"))
        .default_open(true)
        .show(ui, |ui| {
            let selected_name = crate::i18n::locale_display_name(&current_locale);
            egui::ComboBox::from_label(t!("settings.language"))
                .selected_text(selected_name)
                .show_ui(ui, |ui| {
                    for code in &locales {
                        let name = crate::i18n::locale_display_name(code);
                        if ui.selectable_label(current_locale == *code, name).clicked() {
                            state.settings.locale = code.to_string();
                            crate::i18n::set_locale(code);
                            // Reorder the CJK font fallback chain so the
                            // newly-active locale's native shape wins per
                            // glyph (e.g. 飞 vs 飛). No-op if the user
                            // hasn't run dev.ps1 Install-Font yet — they
                            // see the warn line at startup either way.
                            if let Some(fonts) =
                                crate::gui::build_font_definitions(code)
                            {
                                ui.ctx().set_fonts(fonts);
                            }
                        }
                    }
                });
        });

    ui.add_space(4.0);

    egui::CollapsingHeader::new(t!("settings.viewport_controls"))
        .default_open(true)
        .show(ui, |ui| {
            ui.add(
                egui::Slider::new(&mut state.settings.zoom_sensitivity, 0.01..=0.5)
                    .text(t!("settings.zoom_sensitivity")),
            );
            ui.add(
                egui::Slider::new(&mut state.settings.orbit_sensitivity, 0.05..=1.0)
                    .text(t!("settings.orbit_sensitivity")),
            );
            ui.add(
                egui::Slider::new(&mut state.settings.pan_sensitivity, 0.1..=5.0)
                    .text(t!("settings.pan_sensitivity")),
            );
        });

    ui.add_space(4.0);
}
