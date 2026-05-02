//! Label / value rows for "model information"-style cards. Two
//! variants:
//!   * [`kv_row`] — single-line label + value pair where the label
//!     gets a fixed-width column for visual alignment.
//!   * [`kv_grid`] — horizontal grid of label-on-top + value-below
//!     pairs, used for compact stat blocks (mesh count / material
//!     count / spring count …).

use eframe::egui::{self, Ui};

use crate::gui::theme::{color, space, typography};

const LABEL_COLUMN_WIDTH: f32 = 100.0;

/// Single-row label/value pair. Renders the label in
/// `ON_SURFACE_VARIANT` at fixed width on the left, then the value
/// in `ON_SURFACE` filling the rest of the row.
pub fn kv_row(ui: &mut Ui, label: impl Into<String>, value: impl Into<egui::WidgetText>) {
    ui.horizontal(|ui| {
        ui.add_sized(
            [LABEL_COLUMN_WIDTH, ui.spacing().interact_size.y],
            egui::Label::new(
                egui::RichText::new(label)
                    .font(typography::body())
                    .color(color::ON_SURFACE_VARIANT),
            ),
        );
        ui.label(value);
    });
}

/// Compact stat block: each entry is a small label-on-top + larger
/// value-below pair, laid out horizontally and wrapped onto multiple
/// lines as space requires. Use for the "Meshes / Materials / Springs"
/// row in Model Information.
pub fn kv_grid(ui: &mut Ui, items: &[(&str, String)]) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = space::LG;
        for (label, value) in items {
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 2.0;
                ui.label(
                    egui::RichText::new(*label)
                        .font(typography::caption())
                        .color(color::ON_SURFACE_VARIANT),
                );
                ui.label(
                    egui::RichText::new(value)
                        .font(typography::body())
                        .color(color::ON_SURFACE)
                        .strong(),
                );
            });
        }
    });
}
