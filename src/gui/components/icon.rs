//! Material Symbols glyph helpers: every icon in the GUI flows
//! through these so we keep the icon font family + colour resolution
//! out of every call site.

use eframe::egui::{self, Response, Ui};

use crate::gui::theme::{color, typography};

/// `RichText` for a Material Symbols glyph at the given pixel size,
/// coloured `ON_SURFACE_VARIANT` by default. Caller chains `.color(...)`
/// to override (e.g. `icon_text(c, 20.0).color(color::PRIMARY)`).
pub fn icon_text(glyph: char, size: f32) -> egui::RichText {
    egui::RichText::new(glyph.to_string())
        .font(typography::icon(size))
        .color(color::ON_SURFACE_VARIANT)
}

/// Static label pairing an icon glyph and a text string with a small
/// gap. Use for inline "icon + label" composites that don't need
/// click handling (e.g. status-bar items).
pub fn icon_label(ui: &mut Ui, glyph: char, label: &str) -> Response {
    ui.horizontal(|ui| {
        ui.label(icon_text(glyph, 16.0));
        ui.label(label);
    })
    .response
}

/// Borderless icon-only button. Returns the click response.
pub fn icon_button(ui: &mut Ui, glyph: char, hover_text: &str) -> Response {
    let resp = ui.add(egui::Button::new(icon_text(glyph, 18.0)).frame(false));
    resp.on_hover_text(hover_text)
}
