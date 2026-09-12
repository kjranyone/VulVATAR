use eframe::egui::{self, Stroke};

use crate::gui::components::status_dot_label;
use crate::gui::theme::{color, space, typography};
use crate::gui::GuiApp;
use crate::t;

/// Bottom status bar. Ordered by what a streamer needs at a glance
/// mid-session: tracking state + confidence first, then camera rate,
/// then where frames are going. Diagnostic counters (frame index,
/// output queue/dropped) are behind the Settings debug toggle — they
/// are developer readouts, not streaming signal. The avatar shows as
/// its file name only; the full path (which used to push everything
/// else off narrow windows) lives in the tooltip.
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

                // ── Tracking state + live confidence ─────────────
                let tracking_active = state.is_tracking_active();
                let tracking_enabled = state.tracking.toggle_tracking;
                let (dot_color, txt) = if tracking_active {
                    let backend_label = crate::tracking::CAPTURE_BACKEND_LABEL;
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
                if tracking_active {
                    if let Some(pose) = &state.app.last_tracking_pose {
                        ui.label(
                            egui::RichText::new(t!(
                                "status.confidence",
                                pct = format!("{:.0}", pose.overall_confidence * 100.0)
                            ))
                            .font(typography::caption())
                            .color(color::ON_SURFACE_VARIANT),
                        );
                    }
                }

                ui.separator();

                // ── Camera capture rate (configured) ─────────────
                if tracking_active {
                    let fps =
                        crate::tracking::camera_fps_for_index(state.tracking.camera_framerate_index);
                    ui.label(
                        egui::RichText::new(t!("status.camera_fps", fps = fps))
                            .font(typography::caption())
                            .color(color::ON_SURFACE_VARIANT),
                    );
                    ui.separator();
                }

                // ── Output sink + configured stream rate ─────────
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
                let out_fps = crate::gui::output_fps_for_index(state.output.output_framerate_index);
                ui.label(
                    egui::RichText::new(t!(
                        "status.output_with_fps",
                        sink = sink_label.to_string(),
                        fps = out_fps
                    ))
                    .font(typography::caption())
                    .color(color::ON_SURFACE_VARIANT),
                );

                // ── Avatar (file name; full path in tooltip) ──────
                ui.separator();
                match state.app.active_avatar() {
                    Some(avatar) => {
                        let full = avatar.asset.source_path.display().to_string();
                        let name = avatar
                            .asset
                            .source_path
                            .file_name()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_else(|| full.clone());
                        ui.label(
                            egui::RichText::new(name)
                                .font(typography::caption())
                                .color(color::ON_SURFACE_VARIANT),
                        )
                        .on_hover_text(full);
                    }
                    None => {
                        ui.label(
                            egui::RichText::new(t!("app.no_avatar"))
                                .font(typography::caption())
                                .color(color::ON_SURFACE_MUTED),
                        );
                    }
                }

                // ── Diagnostics (Settings › debug toggle only) ────
                if state.debug_status_bar {
                    ui.separator();
                    ui.label(
                        egui::RichText::new(t!(
                            "status.queue_dropped",
                            queue = state.app.output.queue_depth(),
                            dropped = state.app.output.dropped_count()
                        ))
                        .font(typography::caption())
                        .color(color::ON_SURFACE_MUTED),
                    );
                    ui.label(
                        egui::RichText::new(t!(
                            "status.frame",
                            frame = state.runtime_status.frame_count
                        ))
                        .font(typography::caption())
                        .color(color::ON_SURFACE_MUTED),
                    );
                }

                // ── Right-aligned tail: FPS + paused/unsaved badges ─
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(t!(
                            "status.fps",
                            fps = format!("{:.0}", state.runtime_status.fps),
                            ms = format!("{:.1}", state.runtime_status.frame_time_ms)
                        ))
                        .font(typography::caption())
                        .color(color::ON_SURFACE),
                    );
                    if state.runtime_status.paused {
                        ui.label(
                            egui::RichText::new(t!("status.paused"))
                                .font(typography::caption())
                                .color(color::WARNING)
                                .strong(),
                        );
                    }
                    if state.project_status.project_dirty
                        || state.project_status.explicit_file_stale
                    {
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
