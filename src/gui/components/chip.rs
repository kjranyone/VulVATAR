//! Pill-shaped chip used for sort options, filters, and tags.
//! Selected chips render with the primary container fill; unselected
//! chips render with the neutral input fill. State is expressed by
//! fill only — no outline (see `theme::color::state_layer` for why
//! hairline strokes are avoided).

use eframe::egui::{Align2, Response, Rounding, Sense, Ui, Vec2};

use crate::gui::theme::{color, radius, space, typography};

pub fn chip(ui: &mut Ui, label: &str, selected: bool) -> Response {
    let h = 28.0;
    let pad_x = space::MD;
    let label_galley =
        ui.painter()
            .layout_no_wrap(label.to_string(), typography::label(), color::ON_SURFACE);
    let w = pad_x + label_galley.size().x + pad_x;
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, h), Sense::click());

    let (base_bg, fg) = if selected {
        (color::PRIMARY_CONTAINER, color::ON_PRIMARY_CONTAINER)
    } else {
        (color::SURFACE_VARIANT, color::ON_SURFACE_VARIANT)
    };
    let bg = if resp.hovered() {
        color::state_layer(base_bg, color::PRIMARY)
    } else {
        base_bg
    };

    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, Rounding::same(radius::PILL), bg);
    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        typography::label(),
        fg,
    );
    resp
}
