use eframe::egui::{self, Stroke};

use crate::gui::theme::{color, icon as ic, space, typography};
use crate::gui::GuiApp;
use crate::t;

pub fn draw(ctx: &egui::Context, state: &mut GuiApp) {
    egui::TopBottomPanel::bottom("status_bar")
        .exact_height(28.0)
        .frame(egui::Frame {
            fill: color::SURFACE_BRIGHT,
            inner_margin: egui::Margin::symmetric(space::MD, 0.0),
            stroke: Stroke::new(1.0, color::OUTLINE_VARIANT),
            ..Default::default()
        })
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.spacing_mut().item_spacing.x = space::MD;

                // ── Avatar source ─────────────────────────────────
                let avatar_label = match state.app.active_avatar() {
                    Some(avatar) => avatar.asset.source_path.to_string_lossy().to_string(),
                    None => t!("app.no_avatar"),
                };
                ui.label(
                    egui::RichText::new(avatar_label)
                        .font(typography::caption())
                        .color(color::ON_SURFACE_VARIANT),
                );

                ui.separator();

                // ── Tracking ─────────────────────────────────────
                let tracking_active = state.is_tracking_active();
                let tracking_enabled = state.tracking.toggle_tracking;
                let (dot_color, txt) = if tracking_active {
                    let backend_label = state
                        .app
                        .tracking_worker
                        .as_ref()
                        .map(|w| w.active_backend().label())
                        .unwrap_or("Unknown");
                    if tracking_enabled {
                        (
                            color::SUCCESS,
                            t!(
                                "status.tracking_active",
                                backend = backend_label.to_string()
                            ),
                        )
                    } else {
                        (
                            color::WARNING,
                            t!(
                                "status.tracking_paused",
                                backend = backend_label.to_string()
                            ),
                        )
                    }
                } else {
                    (color::ON_SURFACE_MUTED, t!("status.tracking_stopped"))
                };
                status_dot_label(ui, dot_color, &txt);

                ui.separator();

                // ── Output sink ───────────────────────────────────
                let sink_names: [String; 4] = [
                    t!("status.sink_virtual_camera"),
                    t!("status.sink_shared_texture"),
                    t!("status.sink_shared_memory"),
                    t!("status.sink_image_sequence"),
                ];
                let sink_label = sink_names
                    .get(state.app.output.active_sink().to_gui_index())
                    .cloned()
                    .unwrap_or_else(|| t!("status.sink_unknown"));
                ui.label(
                    egui::RichText::new(t!(
                        "status.output_label",
                        sink = sink_label.to_string()
                    ))
                    .font(typography::caption())
                    .color(color::ON_SURFACE_VARIANT),
                );

                ui.separator();

                ui.label(
                    egui::RichText::new(t!(
                        "status.queue_dropped",
                        queue = state.app.output.queue_depth(),
                        dropped = state.app.output.dropped_count()
                    ))
                    .font(typography::caption())
                    .color(color::ON_SURFACE_VARIANT),
                );

                ui.separator();

                ui.label(
                    egui::RichText::new(t!("status.frame", frame = state.frame_count))
                        .font(typography::caption())
                        .color(color::ON_SURFACE_VARIANT),
                );

                // ── Right-aligned tail: FPS + dirty/paused badges ──
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(t!(
                            "status.fps",
                            fps = format!("{:.0}", state.fps),
                            ms = format!("{:.1}", state.frame_time_ms)
                        ))
                        .font(typography::caption())
                        .color(color::ON_SURFACE),
                    );
                    if state.paused {
                        ui.label(
                            egui::RichText::new(t!("status.paused"))
                                .font(typography::caption())
                                .color(color::WARNING)
                                .strong(),
                        );
                    }
                    if state.project_dirty {
                        ui.label(
                            egui::RichText::new(t!("status.modified"))
                                .font(typography::caption())
                                .color(color::WARNING),
                        );
                    }
                });
            });
        });
}

fn status_dot_label(ui: &mut egui::Ui, dot_color: egui::Color32, text: &str) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = space::XS;
        ui.label(
            egui::RichText::new(ic::STATUS_DOT.to_string())
                .font(typography::icon(10.0))
                .color(dot_color),
        );
        ui.label(
            egui::RichText::new(text)
                .font(typography::caption())
                .color(color::ON_SURFACE),
        );
    });
}
