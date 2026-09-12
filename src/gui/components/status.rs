//! Status indicator row: a small coloured dot glyph followed by a
//! caption label. The single way this GUI expresses "state + colour"
//! (tracking state in the status bar, pass/fail telemetry in the
//! calibration modal) so OK/NG semantics always look the same.

use eframe::egui::{self, Ui};

use crate::gui::theme::{color, icon as ic, space, typography};

/// Dot + caption. `dot_color` carries the semantic (SUCCESS / WARNING /
/// ERROR / ON_SURFACE_MUTED); the text stays in the normal foreground
/// colour so colour-blind users still read the label.
///
/// The caption wraps at the available width. A plain `ui.label` inside
/// this horizontal row resolves to `TextWrapMode::Extend` (egui only
/// auto-wraps in vertical or wrapping layouts), so a long caption —
/// e.g. a camera row "name · S/N · USB link · why unusable" in the
/// Tracking panel — ran past the panel edge and got clipped.
pub fn status_dot_label(ui: &mut Ui, dot_color: egui::Color32, text: &str) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = space::XS;
        ui.label(
            egui::RichText::new(ic::STATUS_DOT.to_string())
                .font(typography::icon(10.0))
                .color(dot_color),
        );
        ui.add(
            egui::Label::new(
                egui::RichText::new(text)
                    .font(typography::caption())
                    .color(color::ON_SURFACE),
            )
            .wrap(),
        );
    });
}
