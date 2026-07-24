use eframe::egui::{self, pos2, Align2, Color32, Response, Rounding, Sense, Stroke, Ui, Vec2};

use crate::gui::hotkey::HotkeyAction;
use crate::gui::theme::{color, icon as ic, radius, space, typography};
use crate::gui::{AppMode, GuiApp};
use crate::t;

const SIDEBAR_WIDTH: f32 = 200.0;
const ROW_HEIGHT: f32 = 40.0;

pub fn draw(ctx: &egui::Context, state: &mut GuiApp) {
    // Retired modes (Preview) can still be requested by the hotkey
    // layer — fold them onto their successor before anything reads
    // `state.mode` this frame so the rail highlight and inspector
    // dispatch agree.
    state.mode = state.mode.normalized();
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
                // Surface the F-key binding in the row's tooltip so
                // keyboard shortcuts aren't hidden behind reading the
                // source. Falls back to label-only when no binding is
                // mapped (defensive — every mode currently has one).
                let resp = if let Some(action) = mode_hotkey_action(mode) {
                    let key = state.hotkeys.label_for(action);
                    resp.on_hover_text(format!("{} ({})", mode.label(), key))
                } else {
                    resp.on_hover_text(mode.label())
                };
                if resp.clicked() {
                    if state.mode == mode {
                        state.inspector_open = !state.inspector_open;
                    } else {
                        state.mode = mode;
                        state.inspector_open = true;
                    }
                }
            }

            // ── Footer: panel visibility toggle ──────────────────
            // Rendered as a *utility* row (smaller, muted, square
            // rounding) so it can't be misread as an eighth mode —
            // with the shared pill chrome it used to light up like an
            // active mode whenever the inspector was hidden.
            ui.add_space(space::LG);
            ui.separator();
            ui.add_space(space::SM);
            let hidden = !state.inspector_open;
            let (glyph, label) = if hidden {
                (ic::CHEVRON_RIGHT, t!("app.show_panel"))
            } else {
                (ic::HIDE_PANEL, t!("app.hide_panel"))
            };
            if utility_nav_item(ui, glyph, &label).clicked() {
                state.inspector_open = hidden;
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
        color::with_alpha(color::PRIMARY, 18)
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
        typography::icon(20.0),
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

/// Footer utility row — visually distinct from the mode rows (shorter,
/// muted foreground, small rounding instead of the pill) so structural
/// actions don't read as navigation destinations.
fn utility_nav_item(ui: &mut Ui, glyph: char, label: &str) -> Response {
    let (rect, resp) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), 30.0),
        Sense::click(),
    );

    let bg = if resp.hovered() {
        color::with_alpha(color::PRIMARY, 14)
    } else {
        Color32::TRANSPARENT
    };
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, Rounding::same(radius::SM), bg);

    let icon_x = rect.left() + space::MD;
    let icon_y = rect.center().y;
    painter.text(
        pos2(icon_x, icon_y),
        Align2::LEFT_CENTER,
        glyph.to_string(),
        typography::icon(16.0),
        color::ON_SURFACE_MUTED,
    );
    painter.text(
        pos2(icon_x + 16.0 + space::SM, icon_y),
        Align2::LEFT_CENTER,
        label,
        typography::label(),
        color::ON_SURFACE_MUTED,
    );

    resp
}

fn mode_icon(mode: AppMode) -> char {
    match mode {
        AppMode::Avatar => ic::AVATAR,
        // Retired — normalised away before draw; successor's icon
        // keeps the match exhaustive.
        AppMode::Preview => ic::RENDERING,
        AppMode::TrackingSetup => ic::TRACKING_SETUP,
        AppMode::Rendering => ic::RENDERING,
        AppMode::Output => ic::OUTPUT,
        AppMode::ClothAuthoring => ic::CLOTH_AUTHORING,
        AppMode::Settings => ic::SETTINGS,
    }
}

fn mode_hotkey_action(mode: AppMode) -> Option<HotkeyAction> {
    Some(match mode {
        AppMode::Avatar => HotkeyAction::SwitchModeAvatar,
        // Retired — the action still exists in the hotkey layer and
        // lands on Scene via `AppMode::normalized`.
        AppMode::Preview => HotkeyAction::SwitchModeRendering,
        AppMode::TrackingSetup => HotkeyAction::SwitchModeTracking,
        AppMode::Rendering => HotkeyAction::SwitchModeRendering,
        AppMode::Output => HotkeyAction::SwitchModeOutput,
        AppMode::ClothAuthoring => HotkeyAction::SwitchModeAuthoring,
        AppMode::Settings => HotkeyAction::SwitchModeSettings,
    })
}

#[cfg(test)]
mod tests {
    use super::AppMode;

    #[test]
    fn nav_rail_has_six_modes_and_no_preview() {
        assert_eq!(AppMode::ALL.len(), 6);
        assert!(!AppMode::ALL.contains(&AppMode::Preview));
        // Scene (the Rendering variant) absorbed Preview's cards and
        // must stay navigable.
        assert!(AppMode::ALL.contains(&AppMode::Rendering));
    }

    #[test]
    fn normalized_folds_preview_onto_scene_and_keeps_the_rest() {
        assert_eq!(AppMode::Preview.normalized(), AppMode::Rendering);
        for mode in AppMode::ALL {
            assert_eq!(mode.normalized(), mode);
        }
    }
}
