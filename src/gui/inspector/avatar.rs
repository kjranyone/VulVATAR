use eframe::egui;

use crate::gui::components::{card, card_action_icon, kv_grid, kv_row, outlined_button};
use crate::gui::theme::{color, icon as ic, space, typography};
use crate::gui::GuiApp;
use crate::t;

use super::library;

pub(super) fn draw_avatar(ui: &mut egui::Ui, state: &mut GuiApp) {
    draw_model_information_card(ui, state);
    library::draw_model_library(ui, state);
    draw_camera_transform_card(ui, state);
}

fn draw_model_information_card(ui: &mut egui::Ui, state: &mut GuiApp) {
    let (_, refresh_clicked) = crate::gui::components::card_with_action(
        ui,
        t!("inspector.model_information"),
        |ui| card_action_icon(ui, ic::REFRESH, &t!("inspector.reload")).clicked(),
        |ui| {
            if let Some(avatar) = state.app.active_avatar() {
                let asset = &avatar.asset;
                let meta = &asset.vrm_meta;

                // Spec version pill — distinct from kv rows so the
                // version badge reads at a glance, like the
                // "musette" title in the mockup.
                let (chip_bg, chip_fg) = match meta.spec_version {
                    crate::asset::VrmSpecVersion::V1 => (
                        egui::Color32::from_rgb(220, 246, 226),
                        egui::Color32::from_rgb(28, 110, 50),
                    ),
                    crate::asset::VrmSpecVersion::V0 => (
                        egui::Color32::from_rgb(252, 240, 218),
                        egui::Color32::from_rgb(150, 95, 20),
                    ),
                    crate::asset::VrmSpecVersion::Unknown => (
                        color::ERROR_CONTAINER,
                        color::ON_ERROR_CONTAINER,
                    ),
                };
                ui.horizontal(|ui| {
                    let label_text = match &meta.spec_version_raw {
                        Some(raw) => format!("VRM {} ({})", meta.spec_version.label(), raw),
                        None => format!("VRM {}", meta.spec_version.label()),
                    };
                    egui::Frame {
                        fill: chip_bg,
                        rounding: egui::Rounding::same(crate::gui::theme::radius::PILL),
                        inner_margin: egui::Margin::symmetric(space::SM, 2.0),
                        ..Default::default()
                    }
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new(label_text)
                                .font(typography::label())
                                .color(chip_fg)
                                .strong(),
                        );
                    });
                });

                if meta.spec_version == crate::asset::VrmSpecVersion::V0 {
                    ui.add_space(space::XS);
                    ui.label(
                        egui::RichText::new(t!("inspector.vrm0x_note"))
                            .font(typography::caption())
                            .color(color::WARNING),
                    );
                }
                ui.add_space(space::SM);

                // Metadata key/value rows
                kv_row(
                    ui,
                    t!("inspector.title"),
                    egui::RichText::new(meta.title.as_deref().unwrap_or("—"))
                        .font(typography::body())
                        .color(color::ON_SURFACE),
                );
                let authors = if meta.authors.is_empty() {
                    "—".to_string()
                } else {
                    meta.authors.join(", ")
                };
                kv_row(
                    ui,
                    t!("inspector.authors"),
                    egui::RichText::new(authors)
                        .font(typography::body())
                        .color(color::ON_SURFACE),
                );
                if let Some(v) = &meta.model_version {
                    kv_row(ui, t!("inspector.model_version"), v.as_str());
                }
                if let Some(c) = &meta.contact_information {
                    kv_row(ui, t!("inspector.contact"), c.as_str());
                }
                if let Some(l) = &meta.license {
                    kv_row(
                        ui,
                        t!("inspector.license"),
                        egui::RichText::new(l)
                            .font(typography::body())
                            .color(color::PRIMARY),
                    );
                }
                if let Some(cr) = &meta.copyright_information {
                    kv_row(ui, t!("inspector.copyright"), cr.as_str());
                }
                kv_row(
                    ui,
                    t!("inspector.source_label"),
                    egui::RichText::new(asset.source_path.display().to_string())
                        .font(typography::caption())
                        .color(color::PRIMARY),
                );

                if !meta.references.is_empty() {
                    ui.add_space(space::XS);
                    ui.collapsing(t!("inspector.references"), |ui| {
                        for r in &meta.references {
                            ui.label(
                                egui::RichText::new(r)
                                    .font(typography::caption())
                                    .color(color::ON_SURFACE_VARIANT),
                            );
                        }
                    });
                }

                ui.add_space(space::MD);

                // Stats grid (matches the four-column block in mockup).
                kv_grid(
                    ui,
                    &[
                        (
                            t!("inspector.meshes_label").as_str(),
                            asset.meshes.len().to_string(),
                        ),
                        (
                            t!("inspector.materials_label").as_str(),
                            asset.materials.len().to_string(),
                        ),
                        (
                            t!("inspector.springs_label").as_str(),
                            asset.spring_bones.len().to_string(),
                        ),
                        (
                            t!("inspector.colliders_label").as_str(),
                            asset.colliders.len().to_string(),
                        ),
                    ],
                );
                ui.add_space(space::SM);
                kv_grid(
                    ui,
                    &[
                        (
                            t!("inspector.humanoids_label").as_str(),
                            if asset.humanoid.is_some() {
                                t!("inspector.humanoid_yes_short")
                            } else {
                                t!("inspector.humanoid_no_short")
                            },
                        ),
                        (
                            t!("inspector.expressions_label").as_str(),
                            asset.default_expressions.expressions.len().to_string(),
                        ),
                        (
                            t!("inspector.animations_label").as_str(),
                            asset.animation_clips.len().to_string(),
                        ),
                    ],
                );

                ui.add_space(space::MD);

                ui.horizontal(|ui| {
                    if outlined_button(
                        ui,
                        Some(ic::REFRESH),
                        &t!("inspector.reload"),
                        color::PRIMARY,
                        true,
                    )
                    .clicked()
                    {
                        state.app.reload_avatar();
                    }
                    if outlined_button(
                        ui,
                        Some(ic::REMOVE),
                        &t!("inspector.detach"),
                        color::ERROR,
                        true,
                    )
                    .clicked()
                        && !state.app.avatars.is_empty()
                    {
                        state.app.remove_avatar_at(state.app.active_avatar_index);
                    }
                });
            } else {
                super::draw_no_avatar_state(ui);
            }
        },
    );

    if refresh_clicked {
        state.app.reload_avatar();
    }
}

fn draw_camera_transform_card(ui: &mut egui::Ui, state: &mut GuiApp) {
    card(ui, t!("inspector.camera_transform"), |ui| {
        kv_grid(
            ui,
            &[
                (
                    t!("inspector.yaw").as_str(),
                    format!("{:.3}", state.camera_orbit.yaw_deg),
                ),
                (
                    t!("inspector.pitch").as_str(),
                    format!("{:.3}", state.camera_orbit.pitch_deg),
                ),
                (
                    t!("inspector.distance").as_str(),
                    format!("{:.4}", state.camera_orbit.distance),
                ),
            ],
        );
        ui.add_space(space::SM);
        ui.horizontal(|ui| {
            if ui
                .add(
                    egui::DragValue::new(&mut state.camera_orbit.yaw_deg)
                        .speed(0.5)
                        .prefix(format!("{}: ", t!("inspector.yaw"))),
                )
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .add(
                    egui::DragValue::new(&mut state.camera_orbit.pitch_deg)
                        .speed(0.5)
                        .prefix(format!("{}: ", t!("inspector.pitch"))),
                )
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .add(
                    egui::DragValue::new(&mut state.camera_orbit.distance)
                        .speed(0.05)
                        .range(0.1..=100.0)
                        .prefix(format!("{}: ", t!("inspector.distance"))),
                )
                .changed()
            {
                state.camera_orbit.target_distance = state.camera_orbit.distance;
                state.project_dirty = true;
            }
        });
        ui.horizontal(|ui| {
            if ui
                .add(
                    egui::DragValue::new(&mut state.camera_orbit.pan[0])
                        .speed(0.01)
                        .prefix(format!("{}: ", t!("inspector.pan_x"))),
                )
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .add(
                    egui::DragValue::new(&mut state.camera_orbit.pan[1])
                        .speed(0.01)
                        .prefix(format!("{}: ", t!("inspector.pan_y"))),
                )
                .changed()
            {
                state.project_dirty = true;
            }
        });
        ui.add_space(space::SM);
        if outlined_button(
            ui,
            Some(ic::HISTORY),
            &t!("inspector.reset_camera"),
            color::PRIMARY,
            true,
        )
        .clicked()
        {
            state.camera_orbit.yaw_deg = 0.0;
            state.camera_orbit.pitch_deg = 0.0;
            state.camera_orbit.distance = 5.0;
            state.camera_orbit.target_distance = 5.0;
            state.camera_orbit.pan = [0.0, 0.0];
            state.project_dirty = true;
        }
    });
}
