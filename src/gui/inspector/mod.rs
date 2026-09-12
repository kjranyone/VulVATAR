use eframe::egui;

use crate::gui::theme::{color, space, typography};
use crate::gui::{AppMode, GuiApp};
use crate::t;

mod avatar;
mod cloth;
mod library;
mod output;
mod rendering;
mod settings;
mod tracking;

/// Render the standard "no avatar loaded" empty state — a primary
/// label plus a hint pointing to where the user can load one. Used
/// in every inspector card whose contents only make sense with an
/// active avatar (preview, cloth pin/binding, cloth simulation
/// controls, etc). Centralised so the wording / styling stays
/// consistent across cards instead of each module emitting a bare
/// `t!("inspector.no_avatar_loaded")` label with no follow-up.
pub(super) fn draw_no_avatar_state(ui: &mut egui::Ui) {
    ui.label(
        egui::RichText::new(t!("inspector.no_avatar_loaded"))
            .font(typography::body())
            .color(color::ON_SURFACE_VARIANT),
    );
    ui.label(
        egui::RichText::new(t!("inspector.no_avatar_hint"))
            .font(typography::caption())
            .color(color::ON_SURFACE_MUTED),
    );
}

/// Inspector panel — docked to the left of the viewport, right of the
/// mode-nav rail. Each mode owns the cards relevant to that mode; the
/// dispatch table here is the single point that maps `AppMode` to a
/// `draw_*` entry point.
///
/// Cross-mode helpers (e.g. Camera Transform) live in their owning
/// mode's module — Avatar for Camera Transform — rather than being
/// drawn here unconditionally. The mockup scopes panels per mode so a
/// shared trailing block would just clutter Tracking / Output / etc.
pub fn draw(ctx: &egui::Context, state: &mut GuiApp) {
    if !state.inspector_open {
        return;
    }

    egui::SidePanel::left("inspector_panel")
        .resizable(true)
        .default_width(400.0)
        .min_width(320.0)
        .max_width(560.0)
        .frame(egui::Frame {
            fill: color::SURFACE_DIM,
            inner_margin: egui::Margin::symmetric(space::MD, space::MD),
            stroke: egui::Stroke::new(1.0, color::OUTLINE_VARIANT),
            ..Default::default()
        })
        .show(ctx, |ui| {
            // egui 0.30 `SidePanel` guarantees only the frame HEIGHT via
            // `set_min_height`; the width floor is `width_range.min`. The
            // frame fill paints `content min_rect + margins`, so a claimed
            // panel width wider than the content's min width leaves the
            // surplus strip unpainted and the window clear colour (black)
            // shows through. Claim the full available width up front
            // (mirrors what TopBottomPanel does for its width). Labels
            // that could overflow (file paths, URLs) must additionally
            // truncate — overflowing content pushes the layout cursor
            // past the panel edge and re-opens the hole from the other
            // side; see `panel_fill_tests`.
            ui.set_min_width(ui.max_rect().width());
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = space::MD;
                    match state.mode.normalized() {
                        AppMode::Avatar => avatar::draw_avatar(ui, state),
                        // Preview is retired; `normalized()` folds it
                        // onto Rendering (Scene) so this arm is
                        // unreachable — kept for match exhaustiveness.
                        AppMode::Preview | AppMode::Rendering => rendering::draw_scene(ui, state),
                        AppMode::TrackingSetup => tracking::draw_tracking(ui, state),
                        AppMode::Output => output::draw_output(ui, state),
                        AppMode::ClothAuthoring => cloth::draw_cloth_authoring(ui, state),
                        AppMode::Settings => settings::draw_settings(ui, state),
                    }
                });
        });
}

#[cfg(test)]
mod panel_fill_tests {
    //! Regression tests for the "black band at the inspector's right
    //! edge" bug: egui 0.30's `SidePanel` only guarantees the frame
    //! fill covers the panel HEIGHT (`set_min_height`); for width the
    //! floor is `width_range.min`. The frame paints `content min_rect
    //! + margins`, so any gap between the claimed panel width and the
    //! content's min width is left unpainted and the window clear
    //! colour (black) bleeds through on screen.
    //!
    //! The tests rebuild the inspector's exact panel structure (resizable
    //! SidePanel → vertical ScrollArea with `auto_shrink([false, false])`
    //! → row with an unbreakable long path label), scan one horizontal
    //! line of the opaque fill coverage, and assert that truncating the
    //! overflowing label (plus the `set_min_width` claim from `draw`)
    //! closes the hole.

    use eframe::egui;
    use egui::{Pos2, Rect, Shape, Vec2};

    const LONG_PATH: &str =
        "C:\\lib\\github\\kjranyone\\VulVATAR\\sample_data\\YUMEKA_v1.0.3\\FBX\\Yumeka_v1.0.3.fbx";

    /// Build one frame of a window mimicking the app shell (left mode
    /// rail omitted — irrelevant to the hole) and return the widest
    /// horizontal run, inside the region between the two panels, that
    /// no opaque fill covers.
    fn widest_unpainted_gap(truncate_label: bool) -> f32 {
        let ctx = egui::Context::default();
        // Default (embedded) fonts: the long path must lay out at its
        // real galley width for the overflow to reproduce.
        let mut input = egui::RawInput::default();
        input.screen_rect = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1000.0, 600.0)));
        let output = ctx.run(input, |ctx| {
            egui::SidePanel::left("inspector_panel")
                .resizable(true)
                .default_width(300.0)
                .min_width(200.0)
                .max_width(400.0)
                .frame(egui::Frame {
                    fill: egui::Color32::from_rgb(247, 245, 250),
                    inner_margin: egui::Margin::symmetric(16.0, 16.0),
                    stroke: egui::Stroke::new(1.0, egui::Color32::from_rgb(200, 196, 208)),
                    ..Default::default()
                })
                .show(ctx, |ui| {
                    // Both production defenses under test: the width
                    // claim from `draw` plus the truncated row label.
                    ui.set_min_width(ui.max_rect().width());
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                if truncate_label {
                                    ui.add(egui::Label::new(LONG_PATH).truncate());
                                } else {
                                    ui.label(LONG_PATH);
                                }
                            });
                        });
                });

            // Central panel as in the app: no frame chrome, the app
            // paints its letterbox fill itself — model that with a
            // painted rect so the central area counts as covered.
            egui::CentralPanel::default()
                .frame(egui::Frame::none())
                .show(ctx, |ui| {
                    let rect = ui.max_rect();
                    ui.painter().rect_filled(rect, 0.0, egui::Color32::from_rgb(247, 245, 250));
                });
        });

        // Sample line at mid-height; collect x-ranges covered by opaque
        // rect fills (stroke/text never spans the panel boundary strip).
        let y = 300.0;
        let mut covered: Vec<(f32, f32)> = Vec::new();
        for clipped in &output.shapes {
            let Shape::Rect(rect_shape) = &clipped.shape else {
                continue;
            };
            if rect_shape.fill.a() == 0 {
                continue;
            }
            let r = rect_shape.rect.intersect(clipped.clip_rect);
            if r.min.y <= y && y < r.max.y && r.width() > 0.0 {
                covered.push((r.min.x, r.max.x));
            }
        }
        covered.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut gap = 0.0f32;
        let mut cursor = 0.0f32;
        for (min, max) in covered {
            if min > cursor {
                gap = gap.max(min - cursor);
            }
            cursor = cursor.max(max);
        }
        gap = gap.max((1000.0 - cursor).max(0.0));
        gap
    }

    #[test]
    fn overflowing_label_leaves_a_hole_without_truncation() {
        // egui 0.30 SidePanel advances the layout cursor past the
        // panel's own edge when content overflows, pushing the central
        // panel right and exposing an unpainted (window clear colour)
        // strip. This pins the mechanism the black band came from.
        let gap = widest_unpainted_gap(false);
        assert!(
            gap > 1.0,
            "expected the overflowing path label to leave an unpainted strip \
             (mechanism regression); gap = {gap:.1}px"
        );
    }

    #[test]
    fn truncated_label_covers_the_panel() {
        // Truncating the row label (plus the width claim from `draw`)
        // keeps the layout inside the panel: no unpainted strip.
        let gap = widest_unpainted_gap(true);
        assert!(
            gap < 0.5,
            "truncated label + set_min_width must keep the panel fully \
             painted; unpainted gap = {gap:.1}px"
        );
    }
}
