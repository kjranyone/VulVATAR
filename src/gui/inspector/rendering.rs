use eframe::egui;

use crate::gui::components::{filled_button, outlined_button};
use crate::gui::theme::{color, icon as ic};
use crate::gui::GuiApp;
use crate::t;

use super::preview::draw_expression_control;

pub(super) fn draw_rendering(ui: &mut egui::Ui, state: &mut GuiApp) {
    egui::CollapsingHeader::new(t!("inspector.material_mode"))
        .default_open(true)
        .show(ui, |ui| {
            if ui
                .radio_value(&mut state.rendering.material_mode_index, 0, t!("inspector.unlit"))
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .radio_value(&mut state.rendering.material_mode_index, 1, t!("inspector.simple_lit"))
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .radio_value(&mut state.rendering.material_mode_index, 2, t!("inspector.toon_like"))
                .changed()
            {
                state.project_dirty = true;
            }
        });

    egui::CollapsingHeader::new(t!("inspector.toon_controls"))
        .default_open(true)
        .show(ui, |ui| {
            if ui
                .add_enabled(
                    state.rendering.material_mode_index == 2,
                    egui::Slider::new(&mut state.rendering.toon_ramp_threshold, 0.0..=1.0)
                        .text(t!("inspector.ramp_threshold")),
                )
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .add_enabled(
                    state.rendering.material_mode_index == 2,
                    egui::Slider::new(&mut state.rendering.shadow_softness, 0.0..=1.0)
                        .text(t!("inspector.shadow_softness")),
                )
                .changed()
            {
                state.project_dirty = true;
            }
        });

    egui::CollapsingHeader::new(t!("inspector.outline_controls"))
        .default_open(false)
        .show(ui, |ui| {
            if ui
                .checkbox(&mut state.rendering.outline_enabled, t!("inspector.enable_outline"))
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .add_enabled(
                    state.rendering.outline_enabled,
                    egui::Slider::new(&mut state.rendering.outline_width, 0.0..=0.1).text(t!("inspector.width")),
                )
                .changed()
            {
                state.project_dirty = true;
            }
            ui.add_enabled_ui(state.rendering.outline_enabled, |ui| {
                ui.horizontal(|ui| {
                    ui.label(t!("inspector.color"));
                    ui.color_edit_button_rgb(&mut state.rendering.outline_color);
                });
            });
        });

    egui::CollapsingHeader::new(t!("inspector.lighting"))
        .default_open(true)
        .show(ui, |ui| {
            ui.label(t!("inspector.light_direction"));
            ui.horizontal(|ui| {
                if ui
                    .add(
                        egui::DragValue::new(&mut state.rendering.main_light_dir[0])
                            .prefix("X: ")
                            .speed(0.01),
                    )
                    .changed()
                {
                    state.project_dirty = true;
                }
                if ui
                    .add(
                        egui::DragValue::new(&mut state.rendering.main_light_dir[1])
                            .prefix("Y: ")
                            .speed(0.01),
                    )
                    .changed()
                {
                    state.project_dirty = true;
                }
                if ui
                    .add(
                        egui::DragValue::new(&mut state.rendering.main_light_dir[2])
                            .prefix("Z: ")
                            .speed(0.01),
                    )
                    .changed()
                {
                    state.project_dirty = true;
                }
            });
            if ui
                .add(
                    egui::Slider::new(&mut state.rendering.main_light_intensity, 0.0..=5.0)
                        .text(t!("inspector.intensity")),
                )
                .changed()
            {
                state.project_dirty = true;
            }
            ui.horizontal(|ui| {
                ui.label(t!("inspector.ambient"));
                ui.color_edit_button_rgb(&mut state.rendering.ambient_intensity);
            });
        });

    egui::CollapsingHeader::new(t!("inspector.composition"))
        .default_open(false)
        .show(ui, |ui| {
            if ui
                .add(
                    egui::Slider::new(&mut state.rendering.camera_fov, 10.0..=120.0)
                        .text(t!("inspector.camera_fov")),
                )
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .checkbox(&mut state.rendering.alpha_preview, t!("inspector.alpha_preview"))
                .changed()
            {
                state.project_dirty = true;
            }
        });

    egui::CollapsingHeader::new(t!("inspector.physics"))
        .default_open(false)
        .show(ui, |ui| {
            let rapier_active = state.app.physics.rapier_initialized();
            if rapier_active {
                ui.label(egui::RichText::new(t!("inspector.rapier_active")).color(egui::Color32::GREEN));
            } else {
                ui.label(egui::RichText::new(t!("inspector.rapier_not_init")).color(egui::Color32::GRAY));
            }
            if let Some(pos) = state.app.physics.character_position() {
                ui.label(t!("inspector.character", x = format!("{:.2}", pos[0]), y = format!("{:.2}", pos[1]), z = format!("{:.2}", pos[2])));
            }
        });

    draw_scene_presets(ui, state);

    draw_expression_control(ui, state);
}

fn draw_scene_presets(ui: &mut egui::Ui, state: &mut GuiApp) {
    egui::CollapsingHeader::new(t!("inspector.scene_presets"))
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if filled_button(ui, Some(ic::SAVE), &t!("inspector.save_preset"), true).clicked() {
                    let preset = crate::persistence::ScenePreset {
                        name: state.scene_preset_name.clone(),
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
                            toon_ramp_threshold: state.rendering.toon_ramp_threshold,
                            shadow_softness: state.rendering.shadow_softness,
                            outline_enabled: state.rendering.outline_enabled,
                            outline_width: state.rendering.outline_width,
                            outline_color: state.rendering.outline_color,
                        },
                    };
                    let name = if state.scene_preset_name.is_empty() {
                        format!("Preset {}", state.scene_presets.len() + 1)
                    } else {
                        state.scene_preset_name.clone()
                    };
                    state.scene_presets.push(crate::persistence::ScenePreset {
                        name: name.clone(),
                        ..preset
                    });
                    let _ = crate::persistence::save_scene_presets(&state.scene_presets);
                    state.push_notification(t!("inspector.saved_preset", name = name));
                }
                if outlined_button(
                    ui,
                    Some(ic::DELETE),
                    &t!("inspector.delete_selected"),
                    color::ERROR,
                    true,
                )
                .clicked()
                {
                    if let Some(idx) = state.scene_preset_index {
                        if idx < state.scene_presets.len() {
                            let name = state.scene_presets[idx].name.clone();
                            state.scene_presets.remove(idx);
                            state.scene_preset_index = None;
                            let _ = crate::persistence::save_scene_presets(&state.scene_presets);
                            state.push_notification(t!("inspector.deleted_preset", name = name));
                        }
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label(t!("inspector.name"));
                ui.text_edit_singleline(&mut state.scene_preset_name);
            });

            ui.separator();

            let mut load_idx: Option<usize> = None;
            let preset_names: Vec<String> =
                state.scene_presets.iter().map(|p| p.name.clone()).collect();
            let selected_text = state
                .scene_preset_index
                .and_then(|i| preset_names.get(i))
                .map(|s| s.as_str())
                .unwrap_or("None");
            egui::ComboBox::from_id_salt("scene_preset_selector")
                .selected_text(selected_text)
                .show_ui(ui, |ui| {
                    for (i, name) in preset_names.iter().enumerate() {
                        let is_selected = state.scene_preset_index == Some(i);
                        if ui.selectable_label(is_selected, name.as_str()).clicked() {
                            load_idx = Some(i);
                        }
                    }
                });

            if let Some(idx) = load_idx {
                state.scene_preset_index = Some(idx);
                if let Some(preset) = state.scene_presets.get(idx).cloned() {
                    state.rendering.main_light_dir = preset.lighting.main_light_dir;
                    state.rendering.main_light_intensity = preset.lighting.main_light_intensity;
                    state.rendering.ambient_intensity = preset.lighting.ambient_intensity;
                    state.rendering.camera_fov = preset.camera.fov;
                    state.rendering.material_mode_index = preset.rendering.material_mode_index;
                    state.rendering.toon_ramp_threshold = preset.rendering.toon_ramp_threshold;
                    state.rendering.shadow_softness = preset.rendering.shadow_softness;
                    state.rendering.outline_enabled = preset.rendering.outline_enabled;
                    state.rendering.outline_width = preset.rendering.outline_width;
                    state.rendering.outline_color = preset.rendering.outline_color;
                    state.push_notification(t!("inspector.loaded_preset", name = preset.name));
                    state.project_dirty = true;
                }
            }

            ui.label(t!("inspector.presets_count", count = state.scene_presets.len()));
        });
}
