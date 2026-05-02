//! Material-style button variants. egui's default `Button` resolves
//! through `Visuals::widgets` (set up in [`super::super::theme::apply`]),
//! which is fine for most cases; these helpers cover the two slots
//! that need bespoke colour treatment per the mockup:
//!   * [`outlined_button`] — transparent fill, coloured 1 px border +
//!     coloured label. Used for card actions like Reload (primary
//!     tint) and Detach (error tint).
//!   * [`filled_button`] — solid coloured fill + on-colour label.
//!     Used for primary CTAs like the viewport's Fullscreen toggle.

use eframe::egui::{pos2, Align2, Color32, Response, Rounding, Sense, Stroke, Ui, Vec2};

use crate::gui::theme::{color, radius, space, typography};

const BUTTON_HEIGHT: f32 = 32.0;

pub fn outlined_button(ui: &mut Ui, glyph: Option<char>, label: &str, accent: Color32) -> Response {
    let icon_size = 16.0;
    let label_galley = ui
        .painter()
        .layout_no_wrap(label.to_string(), typography::body(), accent);
    let label_w = label_galley.size().x;
    let pad_x = space::MD;
    let icon_w = if glyph.is_some() {
        icon_size + space::SM
    } else {
        0.0
    };
    let w = pad_x + icon_w + label_w + pad_x;
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, BUTTON_HEIGHT), Sense::click());

    let bg = if resp.hovered() {
        Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 24)
    } else {
        Color32::TRANSPARENT
    };
    let painter = ui.painter_at(rect);
    painter.rect(
        rect,
        Rounding::same(radius::PILL),
        bg,
        Stroke::new(1.0, accent),
    );

    let mut text_x = rect.left() + pad_x;
    if let Some(g) = glyph {
        painter.text(
            pos2(text_x, rect.center().y),
            Align2::LEFT_CENTER,
            g.to_string(),
            typography::icon(icon_size),
            accent,
        );
        text_x += icon_size + space::SM;
    }
    let label_pos = pos2(text_x, rect.center().y - label_galley.size().y * 0.5);
    painter.galley(label_pos, label_galley, accent);

    resp
}

pub fn filled_button(ui: &mut Ui, glyph: Option<char>, label: &str) -> Response {
    let icon_size = 16.0;
    let label_galley =
        ui.painter()
            .layout_no_wrap(label.to_string(), typography::body(), color::ON_PRIMARY);
    let label_w = label_galley.size().x;
    let pad_x = space::MD;
    let icon_w = if glyph.is_some() {
        icon_size + space::SM
    } else {
        0.0
    };
    let w = pad_x + icon_w + label_w + pad_x;
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, BUTTON_HEIGHT), Sense::click());

    let bg = if resp.hovered() {
        color::PRIMARY_HOVER
    } else {
        color::PRIMARY
    };
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, Rounding::same(radius::PILL), bg);

    let mut text_x = rect.left() + pad_x;
    if let Some(g) = glyph {
        painter.text(
            pos2(text_x, rect.center().y),
            Align2::LEFT_CENTER,
            g.to_string(),
            typography::icon(icon_size),
            color::ON_PRIMARY,
        );
        text_x += icon_size + space::SM;
    }
    let label_pos = pos2(text_x, rect.center().y - label_galley.size().y * 0.5);
    painter.galley(label_pos, label_galley, color::ON_PRIMARY);

    resp
}
