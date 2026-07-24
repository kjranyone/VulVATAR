use eframe::egui;

use crate::gui::components::{collapsible_card, kv_row, scope_badge, SettingScope};
use crate::gui::theme::space;
use crate::gui::GuiApp;
use crate::t;

/// "App" storage-layer badge — every widget in this pane persists to
/// `settings.json`, and the badge makes that visible (the project /
/// app / profile split used to be indistinguishable in the UI).
fn app_badge(ui: &mut egui::Ui) {
    scope_badge(ui, SettingScope::App, &t!("settings.badge_app"));
}

/// Settings pane. Everything in here is an APP-level preference —
/// persisted to `%APPDATA%\VulVATAR\settings.json` via
/// `app_settings_dirty`, never to the project file. When adding a new
/// widget, set `app_settings_dirty` on change (not `project_dirty`),
/// and add the field to `persistence::AppSettings` +
/// `GuiApp::collect_app_settings`.
pub(super) fn draw_settings(ui: &mut egui::Ui, state: &mut GuiApp) {
    let locales = crate::i18n::available_locales();
    let current_locale = state.settings.locale.clone();
    let mut changed = false;

    collapsible_card(ui, "settings.heading", t!("settings.heading"), true, |ui| {
            app_badge(ui);
            let selected_name = crate::i18n::locale_display_name(&current_locale);
            egui::ComboBox::from_label(t!("settings.language"))
                .selected_text(selected_name)
                .show_ui(ui, |ui| {
                    for code in &locales {
                        let name = crate::i18n::locale_display_name(code);
                        if ui.selectable_label(current_locale == *code, name).clicked() {
                            state.settings.locale = code.to_string();
                            crate::i18n::set_locale(code);
                            changed = true;
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

    ui.add_space(space::SM);

    collapsible_card(ui, "settings.viewport_controls", t!("settings.viewport_controls"), true, |ui| {
            app_badge(ui);
            changed |= ui
                .add(
                    // Feeds `exp(-scroll * sens)` in the viewport: ~2.5%/notch at
                    // the low end to ~40%/notch at the high end. Logarithmic so the
                    // gentle end (where most users want to be) gets slider travel.
                    egui::Slider::new(&mut state.settings.zoom_sensitivity, 0.0005..=0.01)
                        .logarithmic(true)
                        .text(t!("settings.zoom_sensitivity")),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut state.settings.orbit_sensitivity, 0.05..=1.0)
                        .text(t!("settings.orbit_sensitivity")),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut state.settings.pan_sensitivity, 0.1..=5.0)
                        .text(t!("settings.pan_sensitivity")),
                )
                .changed();
        });

    if changed {
        state.project_status.app_settings_dirty = true;
    }

    ui.add_space(space::SM);

    // ── Diagnostics ──────────────────────────────────────────────
    // Session-only (deliberately not persisted): the status-bar debug
    // counters are a "look at something odd right now" tool, not a
    // preference worth carrying across launches.
    collapsible_card(
        ui,
        "settings.diagnostics",
        t!("settings.diagnostics"),
        false,
        |ui| {
            ui.checkbox(
                &mut state.debug_status_bar,
                t!("settings.debug_status_bar"),
            );
            ui.label(
                egui::RichText::new(t!("settings.debug_status_bar_hint")).small(),
            );
        },
    );

    ui.add_space(space::SM);

    // ── Keyboard shortcuts (read-only) ───────────────────────────
    // Generated from the live `HotkeyMap` bindings so this list can
    // never drift from what the keys actually do.
    collapsible_card(
        ui,
        "settings.shortcuts",
        t!("settings.shortcuts"),
        false,
        |ui| {
            for (action, key_label) in state.hotkeys.entries() {
                kv_row(ui, &t!(action.label_key()), &key_label);
            }
        },
    );

    ui.add_space(space::SM);
}
