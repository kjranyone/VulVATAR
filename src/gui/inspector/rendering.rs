use eframe::egui;

use crate::gui::components::{collapsible_card, filled_button, tonal_button, ButtonTone};
use crate::gui::theme::{color, icon as ic};
use crate::gui::GuiApp;
use crate::t;

/// Scene mode — everything that describes the stage the avatar stands
/// on: where the avatar and camera sit (both transforms live here,
/// side by side, ending the old Avatar/Preview criss-cross), what the
/// background looks like, lighting / materials / composition, and the
/// physics environment (global gravity, Rapier status, presets).
pub(super) fn draw_scene(ui: &mut egui::Ui, state: &mut GuiApp) {
    draw_avatar_transform(ui, state);
    super::avatar::draw_camera_transform_card(ui, state);
    draw_scene_background(ui, state);

    collapsible_card(
        ui,
        "inspector.material_mode",
        t!("inspector.material_mode"),
        false,
        |ui| {
            ui.radio_value(
                &mut state.rendering.material_mode_index,
                0,
                t!("inspector.unlit"),
            )
            .changed();
            ui.radio_value(
                &mut state.rendering.material_mode_index,
                1,
                t!("inspector.simple_lit"),
            )
            .changed();
            ui.radio_value(
                &mut state.rendering.material_mode_index,
                2,
                t!("inspector.toon_like"),
            )
            .changed();
        },
    );

    collapsible_card(
        ui,
        "inspector.lighting",
        t!("inspector.lighting"),
        true,
        |ui| {
            ui.label(t!("inspector.light_direction"));
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut state.rendering.main_light_dir[0])
                        .prefix("X: ")
                        .speed(0.01),
                )
                .changed();
                ui.add(
                    egui::DragValue::new(&mut state.rendering.main_light_dir[1])
                        .prefix("Y: ")
                        .speed(0.01),
                )
                .changed();
                ui.add(
                    egui::DragValue::new(&mut state.rendering.main_light_dir[2])
                        .prefix("Z: ")
                        .speed(0.01),
                )
                .changed();
            });
            ui.add(
                egui::Slider::new(&mut state.rendering.main_light_intensity, 0.0..=5.0)
                    .text(t!("inspector.intensity")),
            )
            .changed();
            ui.horizontal(|ui| {
                ui.label(t!("inspector.ambient"));
                ui.color_edit_button_rgb(&mut state.rendering.ambient_intensity);
            });
        },
    );

    collapsible_card(
        ui,
        "inspector.composition",
        t!("inspector.composition"),
        false,
        |ui| {
            ui.add(
                egui::Slider::new(&mut state.rendering.camera_fov, 10.0..=120.0)
                    .text(t!("inspector.camera_fov")),
            )
            .changed();
            ui.checkbox(
                &mut state.rendering.alpha_preview,
                t!("inspector.alpha_preview"),
            )
            .changed();

            ui.separator();

            ui.checkbox(&mut state.rendering.bloom_enabled, t!("inspector.bloom"))
                .changed();
            ui.add_enabled_ui(state.rendering.bloom_enabled, |ui| {
                ui.add(
                    egui::Slider::new(&mut state.rendering.bloom_intensity, 0.0..=2.0)
                        .text(t!("inspector.bloom_intensity")),
                )
                .changed();
                ui.add(
                    egui::Slider::new(&mut state.rendering.bloom_threshold, 0.0..=4.0)
                        .text(t!("inspector.bloom_threshold")),
                )
                .changed();
            });
        },
    );

    draw_global_gravity(ui, state);

    collapsible_card(
        ui,
        "inspector.physics",
        t!("inspector.physics"),
        false,
        |ui| {
            let rapier_active = state.app.physics.rapier_initialized();
            if rapier_active {
                ui.label(egui::RichText::new(t!("inspector.rapier_active")).color(color::SUCCESS));
            } else {
                ui.label(
                    egui::RichText::new(t!("inspector.rapier_not_init"))
                        .color(color::ON_SURFACE_MUTED),
                );
            }
            if let Some(pos) = state.app.physics.character_position() {
                ui.label(t!(
                    "inspector.character",
                    x = format!("{:.2}", pos[0]),
                    y = format!("{:.2}", pos[1]),
                    z = format!("{:.2}", pos[2])
                ));
            }
        },
    );

    draw_scene_presets(ui, state);
}

/// Avatar world transform (position / rotation / scale). Moved from
/// the retired Preview mode; labelled "Avatar Transform" so it can't
/// be confused with the camera rig card right below it.
fn draw_avatar_transform(ui: &mut egui::Ui, state: &mut GuiApp) {
    collapsible_card(
        ui,
        "inspector.transform",
        t!("inspector.avatar_transform"),
        true,
        |ui| {
            ui.label(t!("inspector.position"));
            ui.horizontal(|ui| {
                for axis in 0..3 {
                    ui.add(
                        egui::DragValue::new(&mut state.transform.position[axis])
                            .prefix(["X: ", "Y: ", "Z: "][axis])
                            .speed(0.01),
                    )
                    .changed();
                }
            });
            ui.label(t!("inspector.rotation"));
            ui.horizontal(|ui| {
                for axis in 0..3 {
                    ui.add(
                        egui::DragValue::new(&mut state.transform.rotation[axis])
                            .prefix(["X: ", "Y: ", "Z: "][axis])
                            .speed(0.1),
                    )
                    .changed();
                }
            });
            ui.horizontal(|ui| {
                ui.label(t!("inspector.scale"));
                ui.add(
                    egui::DragValue::new(&mut state.transform.scale)
                        .speed(0.01)
                        .range(0.01..=100.0),
                )
                .changed();
            });
            if tonal_button(
                ui,
                None,
                &t!("inspector.reset_transform"),
                ButtonTone::Primary,
                true,
            )
            .clicked()
            {
                state.transform.position = [0.0, 0.0, 0.0];
                state.transform.rotation = [0.0, 0.0, 0.0];
                state.transform.scale = 1.0;
            }
        },
    );
}

/// Scene background (solid / transparent / generative). Moved from
/// the retired Preview mode.
fn draw_scene_background(ui: &mut egui::Ui, state: &mut GuiApp) {
    collapsible_card(
        ui,
        "inspector.scene_background",
        t!("inspector.scene_background"),
        false,
        |ui| {
            // An enabled generative background fills the frame with
            // alpha = 1.0, overriding the transparent/solid clear — grey
            // those controls out to make the relationship visible.
            let generative_enabled = state.rendering.generative_background.enabled;
            ui.add_enabled_ui(!generative_enabled, |ui| {
                ui.checkbox(
                    &mut state.rendering.transparent_background,
                    t!("inspector.transparent_background"),
                )
                .changed();
                if !state.rendering.transparent_background {
                    ui.horizontal(|ui| {
                        ui.label(t!("inspector.color"));
                        ui.color_edit_button_rgb(&mut state.rendering.background_color)
                            .changed();
                    });
                }
            });

            ui.separator();

            ui.checkbox(
                &mut state.rendering.generative_background.enabled,
                t!("inspector.generative_background"),
            )
            .changed();
            ui.add_enabled_ui(state.rendering.generative_background.enabled, |ui| {
                ui.add(
                    egui::Slider::new(
                        &mut state.rendering.generative_background.intensity,
                        0.0..=3.0,
                    )
                    .text(t!("inspector.bg_intensity")),
                )
                .changed();
                ui.add(
                    egui::Slider::new(&mut state.rendering.generative_background.speed, 0.0..=2.0)
                        .text(t!("inspector.bg_speed")),
                )
                .changed();
                ui.add(
                    egui::Slider::new(&mut state.rendering.generative_background.scale, 0.5..=8.0)
                        .text(t!("inspector.bg_scale")),
                )
                .changed();
                ui.add(
                    egui::Slider::new(
                        &mut state.rendering.generative_background.reactivity,
                        0.0..=2.0,
                    )
                    .text(t!("inspector.bg_reactivity")),
                )
                .changed();
                ui.horizontal(|ui| {
                    ui.label(t!("inspector.bg_color_a"));
                    ui.color_edit_button_rgb(&mut state.rendering.generative_background.color_a)
                        .changed();
                    ui.label(t!("inspector.bg_color_b"));
                    ui.color_edit_button_rgb(&mut state.rendering.generative_background.color_b)
                        .changed();
                });
            });
        },
    );
}

/// Global gravity for spring / cloth / Rapier. Moved from the retired
/// Preview mode — it describes the scene's physics environment.
fn draw_global_gravity(ui: &mut egui::Ui, state: &mut GuiApp) {
    collapsible_card(
        ui,
        "inspector.gravity_global",
        t!("inspector.gravity_global"),
        false,
        |ui| {
            use crate::simulation::SceneGravity;
            ui.horizontal(|ui| {
                ui.label(t!("inspector.gravity_strength"));
                ui.add(
                    egui::Slider::new(
                        &mut state.rendering.scene_gravity.strength,
                        SceneGravity::STRENGTH_RANGE,
                    )
                    .fixed_decimals(2),
                );
            });
            // World-space direction (default straight down). Applied to
            // spring / cloth / Rapier uniformly; sims inverse-rotate it
            // into each avatar's local frame.
            ui.horizontal(|ui| {
                ui.label(t!("inspector.gravity_direction"));
                let d = &mut state.rendering.scene_gravity.direction;
                ui.add(egui::DragValue::new(&mut d[0]).speed(0.05).prefix("X "));
                ui.add(egui::DragValue::new(&mut d[1]).speed(0.05).prefix("Y "));
                ui.add(egui::DragValue::new(&mut d[2]).speed(0.05).prefix("Z "));
            });
        },
    );
}

fn draw_scene_presets(ui: &mut egui::Ui, state: &mut GuiApp) {
    collapsible_card(
        ui,
        "inspector.scene_preset.presets",
        t!("inspector.scene_preset.presets"),
        false,
        |ui| {
            ui.horizontal(|ui| {
                if filled_button(ui, Some(ic::SAVE), &t!("inspector.save_preset"), true).clicked() {
                    let preset = crate::persistence::ScenePreset {
                        name: state.scene_preset.name.clone(),
                        lighting: crate::persistence::ScenePresetLighting {
                            main_light_dir: state.rendering.main_light_dir,
                            main_light_intensity: state.rendering.main_light_intensity,
                            ambient_intensity: state.rendering.ambient_intensity,
                        },
                        camera: crate::persistence::ScenePresetCamera {
                            fov: state.rendering.camera_fov,
                        },
                        rendering: crate::persistence::ScenePresetRendering {
                            material_mode_index: state.rendering.material_mode_index,
                        },
                    };
                    let name = if state.scene_preset.name.is_empty() {
                        format!("Preset {}", state.scene_preset.presets.len() + 1)
                    } else {
                        state.scene_preset.name.clone()
                    };
                    state
                        .scene_preset
                        .presets
                        .push(crate::persistence::ScenePreset {
                            name: name.clone(),
                            ..preset
                        });
                    if let Err(e) =
                        crate::persistence::save_scene_presets(&state.scene_preset.presets)
                    {
                        log::warn!("persistence: save_scene_presets failed: {e}");
                        state.push_error_notification(t!("toast.preset_save_failed", error = e));
                    } else {
                        state.push_notification(t!("inspector.saved_preset", name = name));
                    }
                }
                if tonal_button(
                    ui,
                    Some(ic::DELETE),
                    &t!("inspector.delete_selected"),
                    ButtonTone::Error,
                    true,
                )
                .clicked()
                {
                    if let Some(idx) = state.scene_preset.selected_index {
                        if idx < state.scene_preset.presets.len() {
                            let name = state.scene_preset.presets[idx].name.clone();
                            state.scene_preset.presets.remove(idx);
                            state.scene_preset.selected_index = None;
                            if let Err(e) =
                                crate::persistence::save_scene_presets(&state.scene_preset.presets)
                            {
                                log::warn!("persistence: save_scene_presets failed: {e}");
                                state.push_error_notification(t!(
                                    "toast.preset_save_failed",
                                    error = e
                                ));
                            } else {
                                state
                                    .push_notification(t!("inspector.deleted_preset", name = name));
                            }
                        }
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label(t!("inspector.name"));
                ui.text_edit_singleline(&mut state.scene_preset.name);
            });

            ui.separator();

            let mut load_idx: Option<usize> = None;
            let preset_names: Vec<String> = state
                .scene_preset
                .presets
                .iter()
                .map(|p| p.name.clone())
                .collect();
            let selected_text = state
                .scene_preset
                .selected_index
                .and_then(|i| preset_names.get(i))
                .map(|s| s.as_str())
                .unwrap_or("None");
            egui::ComboBox::from_id_salt("scene_preset_selector")
                .selected_text(selected_text)
                .show_ui(ui, |ui| {
                    for (i, name) in preset_names.iter().enumerate() {
                        let is_selected = state.scene_preset.selected_index == Some(i);
                        if ui.selectable_label(is_selected, name.as_str()).clicked() {
                            load_idx = Some(i);
                        }
                    }
                });

            if let Some(idx) = load_idx {
                state.scene_preset.selected_index = Some(idx);
                if let Some(preset) = state.scene_preset.presets.get(idx).cloned() {
                    state.rendering.main_light_dir = preset.lighting.main_light_dir;
                    state.rendering.main_light_intensity = preset.lighting.main_light_intensity;
                    state.rendering.ambient_intensity = preset.lighting.ambient_intensity;
                    state.rendering.camera_fov = preset.camera.fov;
                    state.rendering.material_mode_index = preset.rendering.material_mode_index;
                    state.push_notification(t!("inspector.loaded_preset", name = preset.name));
                }
            }

            ui.label(t!(
                "inspector.presets_count",
                count = state.scene_preset.presets.len()
            ));
        },
    );
}
