use eframe::egui;

use crate::gui::components::{
    card_action_icon, collapsible_card, kv_grid, kv_row, tonal_button, ButtonTone,
};
use crate::gui::theme::{color, icon as ic, space, typography};
use crate::gui::GuiApp;
use crate::t;

use super::library;

pub(super) fn draw_avatar(ui: &mut egui::Ui, state: &mut GuiApp) {
    draw_model_information_card(ui, state);
    library::draw_model_library(ui, state);
    draw_runtime_toggles(ui, state);
    draw_expression_control(ui, state);
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
                        color::SUCCESS_CONTAINER,
                        color::ON_SUCCESS_CONTAINER,
                    ),
                    crate::asset::VrmSpecVersion::V0 => (
                        color::WARNING_CONTAINER,
                        color::ON_WARNING_CONTAINER,
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
                    if tonal_button(
                        ui,
                        Some(ic::REFRESH),
                        &t!("inspector.reload"),
                        ButtonTone::Primary,
                        true,
                    )
                    .clicked()
                    {
                        state.app.reload_avatar();
                    }
                    if tonal_button(
                        ui,
                        Some(ic::REMOVE),
                        &t!("inspector.detach"),
                        ButtonTone::Error,
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

/// Runtime feature toggles for the loaded avatar (spring / cloth /
/// debug overlays). Moved here from the retired Preview mode: they
/// gate per-avatar runtime behaviour, so they belong with the avatar,
/// not with scene composition.
fn draw_runtime_toggles(ui: &mut egui::Ui, state: &mut GuiApp) {
    collapsible_card(ui, "inspector.runtime_toggles", t!("inspector.runtime_toggles"), false, |ui| {
        ui
            .checkbox(&mut state.rendering.toggle_spring, t!("inspector.spring_enabled"))
            .changed();
        ui.add_enabled_ui(state.rendering.toggle_spring, |ui| {
            use crate::simulation::spring::SpringTuning;
            ui.horizontal(|ui| {
                ui.label(t!("inspector.spring_sway"));
                ui
                    .add(
                        egui::Slider::new(
                            &mut state.rendering.spring_tuning.sway_scale,
                            SpringTuning::SWAY_RANGE,
                        )
                        .fixed_decimals(2),
                    )
                    .changed();
            });
            ui.horizontal(|ui| {
                ui.label(t!("inspector.spring_gravity"));
                ui
                    .add(
                        egui::Slider::new(
                            &mut state.rendering.spring_tuning.gravity_offset,
                            SpringTuning::GRAVITY_RANGE,
                        )
                        .fixed_decimals(2),
                    )
                    .changed();
            });
        });
        ui
            .checkbox(&mut state.rendering.toggle_cloth, t!("inspector.cloth_enabled"))
            .changed();
        ui
            .checkbox(
                &mut state.rendering.toggle_collision_debug,
                t!("inspector.collision_debug"),
            )
            .changed();
        ui
            .checkbox(
                &mut state.rendering.toggle_skeleton_debug,
                t!("inspector.skeleton_debug"),
            )
            .changed();
    });
}

/// Expression weight sliders. Bind directly to
/// `avatar.expression_weights[i].weight` so the slider widget IS the
/// avatar's value — no GUI shadow buffer, no per-frame reconcile.
/// Face tracking writes a fresh weights Vec to the avatar each frame
/// (`app/render.rs`); the slider sees that new value on the next
/// paint. When tracking is off, the slider directly drives the avatar.
fn draw_expression_control(ui: &mut egui::Ui, state: &mut GuiApp) {
    let Some(avatar) = state.app.active_avatar_mut() else {
        return;
    };
    if avatar.expression_weights.is_empty() {
        return;
    }

    collapsible_card(ui, "inspector.expression_control", t!("inspector.expression_control"), false, |ui| {
        ui.horizontal(|ui| {
            if tonal_button(
                ui,
                None,
                &t!("inspector.reset_all"),
                ButtonTone::Primary,
                true,
            )
            .clicked()
            {
                for ew in avatar.expression_weights.iter_mut() {
                    ew.weight = 0.0;
                }
            }
            if tonal_button(
                ui,
                None,
                &t!("inspector.set_all_50"),
                ButtonTone::Primary,
                true,
            )
            .clicked()
            {
                for ew in avatar.expression_weights.iter_mut() {
                    ew.weight = 0.5;
                }
            }
        });

        ui.separator();

        for ew in avatar.expression_weights.iter_mut() {
            ui.horizontal(|ui| {
                ui.label(&ew.name);
                ui.add(
                    egui::Slider::new(&mut ew.weight, 0.0..=1.0)
                        .text("")
                        .custom_formatter(|n, _| format!("{:.0}%", n * 100.0))
                        .custom_parser(|s| {
                            s.trim_end_matches('%')
                                .parse::<f64>()
                                .ok()
                                .map(|v| v / 100.0)
                        }),
                );
            });
        }

        ui.separator();
        ui.label(t!(
            "inspector.expressions_loaded",
            count = avatar.expression_weights.len()
        ));
    });
}

/// Camera orbit rig editor — drawn by the Scene panel
/// (`rendering::draw_scene`); defined here historically and re-used
/// via `pub(super)` so the move stayed a one-line call-site change.
pub(super) fn draw_camera_transform_card(ui: &mut egui::Ui, state: &mut GuiApp) {
    collapsible_card(ui, "inspector.camera_transform", t!("inspector.camera_transform"), true, |ui| {
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
            ui.add(
                egui::DragValue::new(&mut state.camera_orbit.yaw_deg)
                    .speed(0.5)
                    .prefix(format!("{}: ", t!("inspector.yaw"))),
            );
            ui.add(
                egui::DragValue::new(&mut state.camera_orbit.pitch_deg)
                    .speed(0.5)
                    .prefix(format!("{}: ", t!("inspector.pitch"))),
            );
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
            }
        });
        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut state.camera_orbit.pan[0])
                    .speed(0.01)
                    .prefix(format!("{}: ", t!("inspector.pan_x"))),
            );
            ui.add(
                egui::DragValue::new(&mut state.camera_orbit.pan[1])
                    .speed(0.01)
                    .prefix(format!("{}: ", t!("inspector.pan_y"))),
            );
        });
        ui.add_space(space::SM);
        // World-space eye position. The camera is an orbit rig, so X/Y/Z is a
        // derived quantity (eye = orbit of yaw/pitch/distance around the pan
        // target). Editing it repositions the eye about the world origin —
        // pan is cleared and yaw/pitch/distance re-derived, which is the only
        // *exact* inverse this rig supports. When not being edited the fields
        // track the live eye, matching the viewport overlay.
        let (sy, cy) = state.camera_orbit.yaw_deg.to_radians().sin_cos();
        let (sp, cp) = state.camera_orbit.pitch_deg.to_radians().sin_cos();
        let right = [cy, 0.0, -sy];
        let up = [-sy * sp, cp, -cy * sp];
        let pan = state.camera_orbit.pan;
        let dist = state.camera_orbit.distance;
        let mut eye = [
            dist * cp * sy + pan[0] * right[0] + pan[1] * up[0],
            dist * sp + pan[0] * right[1] + pan[1] * up[1],
            dist * cp * cy + pan[0] * right[2] + pan[1] * up[2],
        ];
        ui.label(t!("inspector.camera_world_pos"));
        ui.horizontal(|ui| {
            let mut changed = false;
            for (axis, prefix) in [(0usize, "X: "), (1, "Y: "), (2, "Z: ")] {
                changed |= ui
                    .add(egui::DragValue::new(&mut eye[axis]).speed(0.05).prefix(prefix))
                    .changed();
            }
            if changed {
                let d = (eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2])
                    .sqrt()
                    .max(0.1);
                state.camera_orbit.distance = d;
                state.camera_orbit.target_distance = d;
                state.camera_orbit.pitch_deg = (eye[1] / d).clamp(-1.0, 1.0).asin().to_degrees();
                state.camera_orbit.yaw_deg = eye[0].atan2(eye[2]).to_degrees();
                state.camera_orbit.pan = [0.0, 0.0];
            }
        });
        ui.add_space(space::SM);
        if tonal_button(
            ui,
            Some(ic::HISTORY),
            &t!("inspector.reset_camera"),
            ButtonTone::Primary,
            true,
        )
        .clicked()
        {
            state.camera_orbit.yaw_deg = 0.0;
            state.camera_orbit.pitch_deg = 0.0;
            state.camera_orbit.distance = 5.0;
            state.camera_orbit.target_distance = 5.0;
            state.camera_orbit.pan = [0.0, 0.0];
        }
    });
}
