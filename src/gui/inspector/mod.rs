use eframe::egui;

use crate::gui::theme::{color, space, typography};
use crate::gui::{AppMode, GuiApp};
use crate::t;

mod avatar;
mod cloth;
mod library;
mod output;
mod preview;
mod rendering;
mod settings;
mod tracking;

/// Render the standard "no avatar loaded" empty state — a primary
/// label plus a hint pointing to where the user can load one. Used
/// in every inspector card whose contents only make sense with an
/// active avatar (preview, cloth pin/binding, cloth simulation
/// controls, etc). Centralised so the wording / styling stays
/// consistent across cards instead of each module emitting a bare
/// `t!("inspector.no_avatar_loaded")` label with no follow-up.
pub(super) fn draw_no_avatar_state(ui: &mut egui::Ui) {
    ui.label(
        egui::RichText::new(t!("inspector.no_avatar_loaded"))
            .font(typography::body())
            .color(color::ON_SURFACE_VARIANT),
    );
    ui.label(
        egui::RichText::new(t!("inspector.no_avatar_hint"))
            .font(typography::caption())
            .color(color::ON_SURFACE_MUTED),
    );
}

/// Inspector panel — docked to the left of the viewport, right of the
/// mode-nav rail. Each mode owns the cards relevant to that mode; the
/// dispatch table here is the single point that maps `AppMode` to a
/// `draw_*` entry point.
///
/// Cross-mode helpers (e.g. Camera Transform) live in their owning
/// mode's module — Avatar for Camera Transform — rather than being
/// drawn here unconditionally. The mockup scopes panels per mode so a
/// shared trailing block would just clutter Tracking / Output / etc.
pub fn draw(ctx: &egui::Context, state: &mut GuiApp) {
    if !state.inspector_open {
        return;
    }

    egui::SidePanel::left("inspector_panel")
        .resizable(true)
        .default_width(400.0)
        .min_width(320.0)
        .max_width(560.0)
        .frame(egui::Frame {
            fill: color::SURFACE_DIM,
            inner_margin: egui::Margin::symmetric(space::MD, space::MD),
            stroke: egui::Stroke::new(1.0, color::OUTLINE_VARIANT),
            ..Default::default()
        })
        .show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = space::MD;
                    match state.mode {
                        AppMode::Avatar => avatar::draw_avatar(ui, state),
                        AppMode::Preview => preview::draw_preview(ui, state),
                        AppMode::TrackingSetup => tracking::draw_tracking(ui, state),
                        AppMode::Rendering => rendering::draw_rendering(ui, state),
                        AppMode::Output => output::draw_output(ui, state),
                        AppMode::ClothAuthoring => cloth::draw_cloth_authoring(ui, state),
                        AppMode::Settings => settings::draw_settings(ui, state),
                    }
                });
        });
}
