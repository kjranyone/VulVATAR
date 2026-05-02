use eframe::egui::{self, pos2, Align2, Color32, Response, Rounding, Sense, Stroke, Ui, Vec2};

use crate::gui::theme::{color, icon as ic, radius, space, typography, ICON_FAMILY};
use crate::gui::{AppMode, GuiApp};
use crate::t;

const SIDEBAR_WIDTH: f32 = 200.0;
const ROW_HEIGHT: f32 = 40.0;

pub fn draw(ctx: &egui::Context, state: &mut GuiApp) {
    egui::SidePanel::left("mode_nav")
        .resizable(false)
        .exact_width(SIDEBAR_WIDTH)
        .frame(egui::Frame {
            fill: color::SURFACE_DIM,
            inner_margin: egui::Margin::symmetric(space::SM, space::MD),
            stroke: Stroke::new(1.0, color::OUTLINE_VARIANT),
            ..Default::default()
        })
        .show(ctx, |ui| {
            // ── Brand header ──────────────────────────────────────
            ui.horizontal(|ui| {
                ui.add_space(space::XS);
                ui.label(
                    egui::RichText::new("\u{f8d6}")
                        .font(typography::icon(22.0))
                        .color(color::PRIMARY),
                );
                ui.label(
                    egui::RichText::new("VulVATAR")
                        .font(typography::title())
                        .color(color::ON_SURFACE)
                        .strong(),
                );
            });
            ui.add_space(space::LG);

            // ── Section label ─────────────────────────────────────
            ui.label(
                egui::RichText::new(t!("app.modes_heading"))
                    .font(typography::caption())
                    .color(color::ON_SURFACE_MUTED),
            );
            ui.add_space(space::XS);

            // ── Mode rows ────────────────────────────────────────
            ui.spacing_mut().item_spacing.y = space::XS;
            for mode in AppMode::ALL {
                let is_active = state.mode == mode && state.inspector_open;
                let resp = mode_nav_item(ui, mode_icon(mode), &mode.label(), is_active);
                if resp.clicked() {
                    if state.mode == mode {
                        state.inspector_open = !state.inspector_open;
                    } else {
                        state.mode = mode;
                        state.inspector_open = true;
                    }
                }
            }

            // ── Footer: Hide Panel ───────────────────────────────
            ui.add_space(space::LG);
            ui.separator();
            ui.add_space(space::SM);
            let hide_resp =
                mode_nav_item(ui, ic::HIDE_PANEL, &t!("app.hide_panel"), !state.inspector_open);
            if hide_resp.clicked() {
                state.inspector_open = false;
            }
        });
}

/// Single sidebar row: icon + label inside a rounded pill that fills
/// the sidebar's content width. Active rows render with the primary
/// container fill; hovered rows get a subtle surface fill.
fn mode_nav_item(ui: &mut Ui, glyph: char, label: &str, active: bool) -> Response {
    let (rect, resp) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), ROW_HEIGHT),
        Sense::click(),
    );

    let bg = if active {
        color::PRIMARY_CONTAINER
    } else if resp.hovered() {
        Color32::from_rgba_unmultiplied(124, 92, 255, 18)
    } else {
        Color32::TRANSPARENT
    };
    let fg = if active {
        color::ON_PRIMARY_CONTAINER
    } else {
        color::ON_SURFACE_VARIANT
    };

    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, Rounding::same(radius::PILL), bg);

    let icon_x = rect.left() + space::MD;
    let icon_y = rect.center().y;
    painter.text(
        pos2(icon_x, icon_y),
        Align2::LEFT_CENTER,
        glyph.to_string(),
        egui::FontId::new(20.0, egui::FontFamily::Name(ICON_FAMILY.into())),
        fg,
    );

    let label_x = icon_x + 20.0 + space::SM;
    painter.text(
        pos2(label_x, icon_y),
        Align2::LEFT_CENTER,
        label,
        typography::body(),
        fg,
    );

    resp
}

fn mode_icon(mode: AppMode) -> char {
    match mode {
        AppMode::Avatar => ic::AVATAR,
        AppMode::Preview => ic::PREVIEW,
        AppMode::TrackingSetup => ic::TRACKING_SETUP,
        AppMode::Rendering => ic::RENDERING,
        AppMode::Output => ic::OUTPUT,
        AppMode::ClothAuthoring => ic::CLOTH_AUTHORING,
        AppMode::Settings => ic::SETTINGS,
    }
}
