//! Material-style button variants. egui's default `Button` resolves
//! through `Visuals::widgets` (set up in [`super::super::theme::apply`]),
//! which is fine for most cases; these helpers cover the two slots
//! that need bespoke colour treatment per the mockup:
//!   * [`tonal_button`] — container-coloured fill + on-container
//!     label, no border. Mid-emphasis: card actions like Reload
//!     ([`ButtonTone::Primary`]) and Detach ([`ButtonTone::Error`]).
//!   * [`filled_button`] — solid `PRIMARY` fill + on-colour label.
//!     High-emphasis CTAs like Start Camera or Save Overlay.
//!
//! Emphasis is expressed exclusively through fills — there is no
//! outlined variant. egui rasterises strokes as anti-aliased meshes
//! with no pixel snapping, so a 1 px outline renders at visibly
//! uneven thickness depending on sub-pixel phase, DPI scale and
//! corner curvature; a hairline border around a pill shape is the
//! worst case. Fills communicate the same hierarchy with none of
//! that fragility (see `theme::color::state_layer`).
//!
//! Both accept an `enabled` flag — disabled buttons render with a
//! neutral fill + muted label and do not react to hover/click. Use
//! this for "Preparing…" placeholders or refresh buttons gated on
//! list contents.

use eframe::egui::{pos2, Align2, Color32, Response, Rounding, Sense, Ui, Vec2};

use crate::gui::theme::{color, radius, space, typography};

const BUTTON_HEIGHT: f32 = 32.0;
const ICON_SIZE: f32 = 16.0;

/// Colour role for a [`tonal_button`]. Call sites pick a semantic
/// tone; the container / on-container pairing lives here so every
/// button resolves through the same theme tokens.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ButtonTone {
    /// Default actions (Reload, Step, Select…).
    Primary,
    /// Destructive actions (Detach, Reset, Clear…).
    Error,
}

impl ButtonTone {
    fn colors(self) -> (Color32, Color32) {
        match self {
            ButtonTone::Primary => (color::PRIMARY_CONTAINER, color::ON_PRIMARY_CONTAINER),
            ButtonTone::Error => (color::ERROR_CONTAINER, color::ON_ERROR_CONTAINER),
        }
    }
}

/// Mid-emphasis tonal button: container fill, on-container label,
/// hover/press via MD3 state layers.
pub fn tonal_button(
    ui: &mut Ui,
    glyph: Option<char>,
    label: &str,
    tone: ButtonTone,
    enabled: bool,
) -> Response {
    let (container, on_container) = tone.colors();
    let (bg, fg) = if enabled {
        (container, on_container)
    } else {
        (color::SURFACE_VARIANT, color::ON_SURFACE_MUTED)
    };
    button_impl(ui, glyph, label, bg, fg, enabled)
}

/// High-emphasis filled button: solid `PRIMARY` fill, `ON_PRIMARY`
/// label.
pub fn filled_button(ui: &mut Ui, glyph: Option<char>, label: &str, enabled: bool) -> Response {
    let (bg, fg) = if enabled {
        (color::PRIMARY, color::ON_PRIMARY)
    } else {
        (color::SURFACE_VARIANT, color::ON_SURFACE_MUTED)
    };
    button_impl(ui, glyph, label, bg, fg, enabled)
}

/// Shared layout + paint: pill-rounded fill, optional leading glyph,
/// label. Hover and press states blend the label colour over the fill
/// at the MD3 state-layer levels (8% / 12%).
fn button_impl(
    ui: &mut Ui,
    glyph: Option<char>,
    label: &str,
    bg: Color32,
    fg: Color32,
    enabled: bool,
) -> Response {
    let label_galley = ui
        .painter()
        .layout_no_wrap(label.to_string(), typography::body(), fg);
    let label_w = label_galley.size().x;
    let pad_x = space::MD;
    let icon_w = if glyph.is_some() {
        ICON_SIZE + space::SM
    } else {
        0.0
    };
    let w = pad_x + icon_w + label_w + pad_x;
    let sense = if enabled {
        Sense::click()
    } else {
        Sense::hover()
    };
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, BUTTON_HEIGHT), sense);

    let bg = if enabled && resp.is_pointer_button_down_on() {
        color::pressed_layer(bg, fg)
    } else if enabled && resp.hovered() {
        color::state_layer(bg, fg)
    } else {
        bg
    };
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, Rounding::same(radius::PILL), bg);

    let mut text_x = rect.left() + pad_x;
    if let Some(g) = glyph {
        painter.text(
            pos2(text_x, rect.center().y),
            Align2::LEFT_CENTER,
            g.to_string(),
            typography::icon(ICON_SIZE),
            fg,
        );
        text_x += ICON_SIZE + space::SM;
    }
    let label_pos = pos2(text_x, rect.center().y - label_galley.size().y * 0.5);
    painter.galley(label_pos, label_galley, fg);

    resp
}
