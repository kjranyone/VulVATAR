//! MD3-style card surface: rounded rectangle on `SURFACE` with a
//! subtle outline, generous internal padding, and an optional title +
//! trailing action row at the top. Cards stack inside the inspector
//! and replace the previous `CollapsingHeader` chrome.
//!
//! Two entry points:
//!   * [`card`] — title + content. Use for static panels.
//!   * [`card_with_action`] — title + small trailing slot (e.g.
//!     refresh icon button) + content. Use when the card surface
//!     itself needs an action.

use eframe::egui::{self, Color32, Frame, Margin, Response, Rounding, Stroke, Ui};

use crate::gui::theme::{color, radius, space, typography};

/// Render a titled card. `add_contents` runs inside the card's
/// content area with `SURFACE` background and `space::MD` padding all
/// around. The title is rendered with `typography::title()` weight.
pub fn card<R>(
    ui: &mut Ui,
    title: impl Into<String>,
    add_contents: impl FnOnce(&mut Ui) -> R,
) -> R {
    card_with_action(ui, title, |_| {}, add_contents).0
}

/// Render a titled card with a trailing action slot in the title row.
/// `add_action` runs in a right-aligned ui inside the title row;
/// `add_contents` runs in the body. Returns `(body_value, action_value)`.
pub fn card_with_action<R, A>(
    ui: &mut Ui,
    title: impl Into<String>,
    add_action: impl FnOnce(&mut Ui) -> A,
    add_contents: impl FnOnce(&mut Ui) -> R,
) -> (R, A) {
    let frame = Frame {
        inner_margin: Margin::same(space::MD),
        outer_margin: Margin::ZERO,
        rounding: Rounding::same(radius::MD),
        shadow: egui::epaint::Shadow {
            offset: egui::Vec2::new(0.0, 1.0),
            blur: 3.0,
            spread: 0.0,
            color: Color32::from_rgba_unmultiplied(60, 50, 90, 18),
        },
        fill: color::SURFACE,
        stroke: Stroke::new(1.0, color::OUTLINE_VARIANT),
    };

    let mut action_value: Option<A> = None;
    let body = frame
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(title)
                        .font(typography::title())
                        .color(color::ON_SURFACE)
                        .strong(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    action_value = Some(add_action(ui));
                });
            });
            ui.add_space(space::SM);
            add_contents(ui)
        })
        .inner;

    // SAFETY: add_action runs unconditionally above, so action_value
    // is always Some at this point. unwrap is the standard pattern for
    // "moved out of an FnOnce closure" in egui's frame.show idiom.
    (body, action_value.expect("card action closure always runs"))
}

/// Convenience: a small `IconButton`-style trailing slot. Renders a
/// borderless icon button suitable for use as the `add_action` slot.
/// Returns the click response so callers can bind hover text and
/// click handling.
pub fn card_action_icon(ui: &mut Ui, glyph: char, hover_text: &str) -> Response {
    let txt = egui::RichText::new(glyph.to_string())
        .font(typography::icon(18.0))
        .color(color::ON_SURFACE_VARIANT);
    let resp = ui.add(egui::Button::new(txt).frame(false));
    resp.on_hover_text(hover_text)
}
