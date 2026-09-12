//! MD3-style elevated card surface: rounded rectangle on `SURFACE`
//! with a soft shadow, generous internal padding, and an optional
//! title + trailing action row at the top. Cards stack inside the
//! inspector and replace the previous `CollapsingHeader` chrome.
//! Elevation comes from the shadow + fill contrast against
//! `SURFACE_DIM` alone — no outline (egui can't render hairline
//! strokes at uniform thickness; see `theme::color::state_layer`).
//!
//! Three entry points:
//!   * [`card`] — title + content. Use for static panels.
//!   * [`card_with_action`] — title + small trailing slot (e.g.
//!     refresh icon button) + content. Use when the card surface
//!     itself needs an action.
//!   * [`collapsible_card`] — title row toggles the body open/closed.
//!     The direct replacement for raw `egui::CollapsingHeader` inside
//!     the inspector: same persistence semantics (open state keyed by
//!     a stable id), but rendered as a card so collapsed and expanded
//!     sections share one visual language.

use eframe::egui::{self, Color32, Frame, Margin, Response, Rounding, Sense, Stroke, Ui};

use crate::gui::theme::{color, icon, radius, space, typography};

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
    let frame = card_frame();

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

/// Render a collapsible card: the whole title row is the toggle
/// target. Open state persists in egui memory under
/// `ui.make_persistent_id(id_salt)` (same lifetime as the old
/// `CollapsingHeader` state it replaces). Returns `Some(body_value)`
/// while open, `None` while collapsed.
///
/// Hover feedback on the title row is a translucent primary fill
/// (MD3 state layer at ~8%) — no strokes, per the tonal-design rule.
pub fn collapsible_card<R>(
    ui: &mut Ui,
    id_salt: &str,
    title: impl Into<String>,
    default_open: bool,
    add_contents: impl FnOnce(&mut Ui) -> R,
) -> Option<R> {
    let title: String = title.into();
    let frame = card_frame();
    frame
        .show(ui, |ui| {
            let id = ui.make_persistent_id(id_salt);
            let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(
                ui.ctx(),
                id,
                default_open,
            );
            let open = state.is_open();

            // Title row — full width, single click target. The chevron
            // sits on the right so titles align with non-collapsible
            // cards.
            let row = ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(&title)
                        .font(typography::title())
                        .color(color::ON_SURFACE)
                        .strong(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let chevron = if open {
                        icon::EXPAND_MORE
                    } else {
                        icon::CHEVRON_RIGHT
                    };
                    ui.label(
                        egui::RichText::new(chevron.to_string())
                            .font(typography::icon(18.0))
                            .color(color::ON_SURFACE_VARIANT),
                    );
                });
            });
            // Stretch the click target across the card's inner width so
            // the whole header behaves like one control.
            let header_rect = row.response.rect;
            let target = egui::Rect::from_min_max(
                egui::pos2(ui.max_rect().left(), header_rect.top()),
                egui::pos2(ui.max_rect().right(), header_rect.bottom()),
            );
            let resp = ui.interact(target, id.with("header"), Sense::click());
            if resp.hovered() {
                ui.painter().rect_filled(
                    target.expand2(egui::vec2(space::XS, space::XS * 0.5)),
                    Rounding::same(radius::SM),
                    color::with_alpha(color::PRIMARY, 20),
                );
            }
            if resp.clicked() {
                state.toggle(ui);
            }

            let body = state
                .show_body_unindented(ui, |ui| {
                    ui.add_space(space::SM);
                    add_contents(ui)
                })
                .map(|inner| inner.inner);
            state.store(ui.ctx());
            body
        })
        .inner
}

/// Frameless variant of [`collapsible_card`] for sub-sections nested
/// *inside* an existing card, where a second layer of elevation would
/// read as clutter. Same header row + persistence semantics, label
/// weight one step down (`typography::label`), no card chrome.
pub fn collapsible_section<R>(
    ui: &mut Ui,
    id_salt: &str,
    title: impl Into<String>,
    default_open: bool,
    add_contents: impl FnOnce(&mut Ui) -> R,
) -> Option<R> {
    let title: String = title.into();
    let id = ui.make_persistent_id(id_salt);
    let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(
        ui.ctx(),
        id,
        default_open,
    );
    let open = state.is_open();

    let row = ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(&title)
                .font(typography::label())
                .color(color::ON_SURFACE_VARIANT)
                .strong(),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let chevron = if open {
                icon::EXPAND_MORE
            } else {
                icon::CHEVRON_RIGHT
            };
            ui.label(
                egui::RichText::new(chevron.to_string())
                    .font(typography::icon(16.0))
                    .color(color::ON_SURFACE_VARIANT),
            );
        });
    });
    let header_rect = row.response.rect;
    let target = egui::Rect::from_min_max(
        egui::pos2(ui.max_rect().left(), header_rect.top()),
        egui::pos2(ui.max_rect().right(), header_rect.bottom()),
    );
    let resp = ui.interact(target, id.with("header"), Sense::click());
    if resp.hovered() {
        ui.painter().rect_filled(
            target.expand2(egui::vec2(space::XS, space::XS * 0.5)),
            Rounding::same(radius::SM),
            color::with_alpha(color::PRIMARY, 20),
        );
    }
    if resp.clicked() {
        state.toggle(ui);
    }

    let body = state
        .show_body_unindented(ui, |ui| {
            ui.add_space(space::XS);
            add_contents(ui)
        })
        .map(|inner| inner.inner);
    state.store(ui.ctx());
    body
}

/// The shared card chrome (fill, rounding, shadow, padding) used by
/// every card variant so elevation stays identical across them.
fn card_frame() -> Frame {
    Frame {
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
        stroke: Stroke::NONE,
    }
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
