use eframe::egui;

use crate::gui::{AppMode, GuiApp};
use crate::t;

mod avatar;
mod cloth;
mod library;
mod output;
mod preview;
mod rendering;
mod settings;
mod tracking;

pub fn draw(ctx: &egui::Context, state: &mut GuiApp) {
    if !state.inspector_open {
        return;
    }

    let mode_label = state.mode.label();

    egui::Window::new(&mode_label)
        .id(egui::Id::new("inspector_window"))
        .default_rect(egui::Rect::from_min_size(
            egui::pos2(150.0, 40.0),
            egui::vec2(300.0, 600.0),
        ))
        .resizable(true)
        .collapsible(true)
        .scroll([false, true])
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading(&mode_label);
            });
            ui.separator();

            ui.push_id("inspector_scroll", |ui| {
                egui::ScrollArea::vertical()
                    .max_height(f32::INFINITY)
                    .show(ui, |ui| {
                        match state.mode {
                            AppMode::Avatar => avatar::draw_avatar(ui, state),
                            AppMode::Preview => preview::draw_preview(ui, state),
                            AppMode::TrackingSetup => tracking::draw_tracking(ui, state),
                            AppMode::Rendering => rendering::draw_rendering(ui, state),
                            AppMode::Output => output::draw_output(ui, state),
                            AppMode::ClothAuthoring => cloth::draw_cloth_authoring(ui, state),
                            AppMode::Settings => settings::draw_settings(ui, state),
                        }

                        ui.separator();
                        egui::CollapsingHeader::new(t!("inspector.camera_transform"))
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(t!("inspector.yaw"));
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut state.camera_orbit.yaw_deg)
                                                .speed(0.5),
                                        )
                                        .changed()
                                    {
                                        state.project_dirty = true;
                                    }
                                    ui.label(t!("inspector.pitch"));
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut state.camera_orbit.pitch_deg)
                                                .speed(0.5),
                                        )
                                        .changed()
                                    {
                                        state.project_dirty = true;
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label(t!("inspector.distance"));
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut state.camera_orbit.distance)
                                                .speed(0.05)
                                                .range(0.1..=100.0),
                                        )
                                        .changed()
                                    {
                                        // The per-frame zoom path lerps `distance` toward
                                        // `target_distance` (mouse-wheel feel), so editing
                                        // `distance` alone gets pulled back to the old
                                        // target every frame. Snap both fields so the
                                        // typed value sticks.
                                        state.camera_orbit.target_distance =
                                            state.camera_orbit.distance;
                                        state.project_dirty = true;
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label(t!("inspector.pan_x"));
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut state.camera_orbit.pan[0])
                                                .speed(0.01),
                                        )
                                        .changed()
                                    {
                                        state.project_dirty = true;
                                    }
                                    ui.label(t!("inspector.pan_y"));
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut state.camera_orbit.pan[1])
                                                .speed(0.01),
                                        )
                                        .changed()
                                    {
                                        state.project_dirty = true;
                                    }
                                });
                                if ui.button(t!("inspector.reset_camera")).clicked() {
                                    state.camera_orbit.yaw_deg = 0.0;
                                    state.camera_orbit.pitch_deg = 0.0;
                                    state.camera_orbit.distance = 5.0;
                                    state.camera_orbit.target_distance = 5.0;
                                    state.camera_orbit.pan = [0.0, 0.0];
                                    state.project_dirty = true;
                                }
                            });
                    });
            });
        });
}
