//! Pill-shaped chip used for sort options, filters, and tags.
//! Selected chips render with the primary container fill; unselected
//! chips render with a subtle outline-only style.

use eframe::egui::{Align2, Color32, Response, Rounding, Sense, Stroke, Ui, Vec2};

use crate::gui::theme::{color, radius, space, typography};

pub fn chip(ui: &mut Ui, label: &str, selected: bool) -> Response {
    let h = 28.0;
    let pad_x = space::MD;
    let label_galley =
        ui.painter()
            .layout_no_wrap(label.to_string(), typography::label(), color::ON_SURFACE);
    let w = pad_x + label_galley.size().x + pad_x;
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, h), Sense::click());

    let (bg, fg, stroke) = if selected {
        (
            color::PRIMARY_CONTAINER,
            color::ON_PRIMARY_CONTAINER,
            Stroke::NONE,
        )
    } else if resp.hovered() {
        (
            Color32::from_rgba_unmultiplied(124, 92, 255, 16),
            color::ON_SURFACE,
            Stroke::new(1.0, color::PRIMARY),
        )
    } else {
        (
            Color32::TRANSPARENT,
            color::ON_SURFACE_VARIANT,
            Stroke::new(1.0, color::OUTLINE),
        )
    };

    let painter = ui.painter_at(rect);
    // Shrink by 0.5 px so the 1 px stroke (when present) lands fully
    // inside the painter's clip rect; otherwise the outer half of the
    // outline is clipped on the unselected / hover variants.
    painter.rect(rect.shrink(0.5), Rounding::same(radius::PILL), bg, stroke);
    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        typography::label(),
        fg,
    );
    let _ = label_galley;
    resp
}
