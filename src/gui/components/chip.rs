//! Pill-shaped chip used for sort options, filters, and tags.
//! Selected chips render with the primary container fill; unselected
//! chips render with the neutral input fill. State is expressed by
//! fill only — no outline (see `theme::color::state_layer` for why
//! hairline strokes are avoided).

use eframe::egui::{Align2, Response, Rounding, Sense, Ui, Vec2};

use crate::gui::theme::{color, radius, space, typography};

/// Where a setting is persisted. Drives [`scope_badge`]'s colouring so
/// the three storage layers (project file / app settings.json /
/// profiles.json) are visually distinguishable next to any widget.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SettingScope {
    /// `%APPDATA%\VulVATAR\settings.json` — follows the user.
    App,
    /// `profiles.json` — follows the active stream profile.
    Profile,
}

/// Tiny non-interactive badge marking which storage layer a setting
/// belongs to ("App" / "Profile"). The three save targets were
/// previously invisible in the UI — identical-looking sliders wrote to
/// three different files and only one of them lit the Modified dot.
pub fn scope_badge(ui: &mut Ui, scope: SettingScope, label: &str) -> Response {
    let h = 18.0;
    let pad_x = space::SM;
    let (bg, fg) = match scope {
        SettingScope::App => (color::SURFACE_VARIANT, color::ON_SURFACE_VARIANT),
        SettingScope::Profile => (color::PRIMARY_CONTAINER, color::ON_PRIMARY_CONTAINER),
    };
    let galley = ui
        .painter()
        .layout_no_wrap(label.to_string(), typography::caption(), fg);
    let w = pad_x + galley.size().x + pad_x;
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, h), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, Rounding::same(radius::PILL), bg);
    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        typography::caption(),
        fg,
    );
    resp
}

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
