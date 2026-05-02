use eframe::egui;

use crate::gui::components::outlined_button;
use crate::gui::theme::{color, icon as ic};
use crate::gui::GuiApp;
use crate::t;

pub(super) fn draw_preview(ui: &mut egui::Ui, state: &mut GuiApp) {
    egui::CollapsingHeader::new(t!("inspector.avatar_asset"))
        .default_open(true)
        .show(ui, |ui| {
            if let Some(avatar) = state.app.active_avatar() {
                ui.horizontal(|ui| {
                    ui.label(t!("inspector.source", path = avatar.asset.source_path.display().to_string()));
                    if avatar.asset.loaded_from_cache {
                        ui.label(
                            egui::RichText::new(t!("inspector.cached"))
                                .small()
                                .color(egui::Color32::from_rgb(80, 200, 120))
                                .background_color(egui::Color32::from_rgb(20, 50, 30)),
                        )
                        .on_hover_text(
                            t!("inspector.cached_tooltip"),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new(t!("inspector.fresh_load"))
                                .small()
                                .color(egui::Color32::from_rgb(220, 220, 220)),
                        );
                    }
                });
                ui.label(t!("inspector.id", id = avatar.asset.id.0));
                ui.label(t!("inspector.meshes", count = avatar.asset.meshes.len()));
                ui.label(t!("inspector.materials", count = avatar.asset.materials.len()));
                ui.label(t!("inspector.spring_chains", count = avatar.asset.spring_bones.len()));
                ui.label(t!("inspector.colliders", count = avatar.asset.colliders.len()));
                if let Some(cloth_id) = &avatar.attached_cloth {
                    ui.label(t!("inspector.primary_cloth", id = cloth_id.0));
                } else {
                    ui.label(t!("inspector.primary_cloth_none"));
                }
                ui.label(t!("inspector.additional_cloth", count = avatar.cloth_overlay_count()));
            } else {
                ui.label(t!("inspector.no_avatar_loaded"));
            }
            ui.add_space(4.0);
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
                    None,
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
        });

    egui::CollapsingHeader::new(t!("inspector.transform"))
        .default_open(true)
        .show(ui, |ui| {
            ui.label(t!("inspector.position"));
            ui.horizontal(|ui| {
                if ui
                    .add(
                        egui::DragValue::new(&mut state.transform.position[0])
                            .prefix("X: ")
                            .speed(0.01),
                    )
                    .changed()
                {
                    state.project_dirty = true;
                }
                if ui
                    .add(
                        egui::DragValue::new(&mut state.transform.position[1])
                            .prefix("Y: ")
                            .speed(0.01),
                    )
                    .changed()
                {
                    state.project_dirty = true;
                }
                if ui
                    .add(
                        egui::DragValue::new(&mut state.transform.position[2])
                            .prefix("Z: ")
                            .speed(0.01),
                    )
                    .changed()
                {
                    state.project_dirty = true;
                }
            });
            ui.label(t!("inspector.rotation"));
            ui.horizontal(|ui| {
                if ui
                    .add(
                        egui::DragValue::new(&mut state.transform.rotation[0])
                            .prefix("X: ")
                            .speed(0.1),
                    )
                    .changed()
                {
                    state.project_dirty = true;
                }
                if ui
                    .add(
                        egui::DragValue::new(&mut state.transform.rotation[1])
                            .prefix("Y: ")
                            .speed(0.1),
                    )
                    .changed()
                {
                    state.project_dirty = true;
                }
                if ui
                    .add(
                        egui::DragValue::new(&mut state.transform.rotation[2])
                            .prefix("Z: ")
                            .speed(0.1),
                    )
                    .changed()
                {
                    state.project_dirty = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label(t!("inspector.scale"));
                if ui
                    .add(
                        egui::DragValue::new(&mut state.transform.scale)
                            .speed(0.01)
                            .range(0.01..=100.0),
                    )
                    .changed()
                {
                    state.project_dirty = true;
                }
            });
            if outlined_button(
                ui,
                None,
                &t!("inspector.reset_transform"),
                color::PRIMARY,
                true,
            )
            .clicked()
            {
                state.transform.position = [0.0, 0.0, 0.0];
                state.transform.rotation = [0.0, 0.0, 0.0];
                state.transform.scale = 1.0;
                state.project_dirty = true;
            }
        });

    egui::CollapsingHeader::new(t!("inspector.cloth_attachment"))
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if outlined_button(
                    ui,
                    None,
                    &t!("inspector.attach_primary"),
                    color::PRIMARY,
                    true,
                )
                .clicked()
                {
                    let overlay_id = state
                        .app
                        .editor
                        .active_overlay()
                        .unwrap_or(crate::asset::ClothOverlayId(1));
                    if let Some(avatar) = state.app.active_avatar_mut() {
                        avatar.attach_cloth(overlay_id);
                    }
                }
                if outlined_button(
                    ui,
                    None,
                    &t!("inspector.detach_primary"),
                    color::ERROR,
                    true,
                )
                .clicked()
                {
                    if let Some(avatar) = state.app.active_avatar_mut() {
                        avatar.detach_cloth();
                    }
                }
            });
            ui.horizontal(|ui| {
                if outlined_button(
                    ui,
                    Some(ic::ADD),
                    &t!("inspector.add_overlay_slot"),
                    color::PRIMARY,
                    true,
                )
                .clicked()
                {
                    let sim_mesh = state
                        .app
                        .editor
                        .overlay_asset
                        .as_ref()
                        .map(|o| o.simulation_mesh.clone());
                    if let Some(avatar) = state.app.active_avatar_mut() {
                        let overlay_id =
                            crate::asset::ClothOverlayId((avatar.cloth_overlay_count() as u64) + 2);
                        let idx = avatar.attach_cloth_overlay(overlay_id);
                        if let Some(sim_mesh) = sim_mesh {
                            let mut cloth_asset = crate::asset::ClothAsset::new_empty(
                                crate::asset::ClothOverlayId(0),
                                avatar.asset.id,
                                avatar.asset.source_hash.clone(),
                            );
                            cloth_asset.simulation_mesh = sim_mesh;
                            avatar.init_cloth_overlay(idx, &cloth_asset);
                        }
                    }
                }
                if outlined_button(
                    ui,
                    Some(ic::FOLDER_OPEN),
                    &t!("inspector.load_overlay_file"),
                    color::PRIMARY,
                    true,
                )
                .clicked()
                {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter(t!("top_bar.filter_cloth"), &["vvtcloth"])
                        .set_title("Load cloth overlay into a new slot")
                        .pick_file()
                    {
                        let load_result = crate::persistence::load_cloth_overlay(&path);
                        let asset_opt = match load_result {
                            Ok(file) => file.cloth_asset,
                            Err(e) => {
                                state.push_notification(t!("inspector.failed_load_overlay", path = path.display().to_string(), error = e.to_string()));
                                None
                            }
                        };
                        if let Some(mut cloth_asset) = asset_opt {
                            // Resolve the overlay's stored IDs against the
                            // currently-loaded avatar. If the avatar was
                            // re-exported and the IDs drifted, the rebinder
                            // rewrites them by name. A Failed status (refs
                            // we can't resolve at any tier) skips the attach.
                            if !state.attempt_overlay_rebind(&path, &mut cloth_asset) {
                                state.push_notification(t!("inspector.overlay_rebind_failed", path = path.display().to_string()));
                            } else if let Some(avatar) = state.app.active_avatar_mut() {
                                let overlay_id = crate::asset::ClothOverlayId(
                                    (avatar.cloth_overlay_count() as u64) + 2,
                                );
                                let idx = avatar.attach_cloth_overlay(overlay_id);
                                avatar.init_cloth_overlay(idx, &cloth_asset);
                                if let Some(slot) = avatar.cloth_overlays.get_mut(idx) {
                                    slot.source_path = Some(path.clone());
                                }
                                state.push_notification(t!("inspector.overlay_attached", path = path.display().to_string(), index = idx));
                            }
                        } else {
                            state.push_notification(t!("inspector.overlay_no_payload", path = path.display().to_string()));
                        }
                    }
                }
            });
            if let Some(avatar) = state.app.active_avatar_mut() {
                ui.checkbox(&mut avatar.cloth_enabled, t!("inspector.enable_cloth"));
                // Defer removal until after the iter_mut borrow ends —
                // can't mutate the Vec we're iterating.
                let mut remove_slot: Option<usize> = None;
                for (i, slot) in avatar.cloth_overlays.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.checkbox(
                            &mut slot.enabled,
                            format!(
                                "#{} (id={}, {} particles)",
                                i,
                                slot.overlay_id.0,
                                slot.sim.particle_count()
                            ),
                        );
                        if outlined_button(
                            ui,
                            None,
                            &t!("inspector.remove"),
                            color::ERROR,
                            true,
                        )
                        .clicked()
                        {
                            remove_slot = Some(i);
                        }
                    });
                }
                if let Some(idx) = remove_slot {
                    avatar.remove_cloth_overlay(idx);
                }
            }
        });

    egui::CollapsingHeader::new(t!("inspector.scene_background"))
        .default_open(false)
        .show(ui, |ui| {
            if ui
                .checkbox(
                    &mut state.rendering.transparent_background,
                    t!("inspector.transparent_background"),
                )
                .changed()
            {
                state.project_dirty = true;
            }
            if !state.rendering.transparent_background {
                ui.horizontal(|ui| {
                    ui.label(t!("inspector.color"));
                    if ui
                        .color_edit_button_rgb(&mut state.rendering.background_color)
                        .changed()
                    {
                        state.project_dirty = true;
                    }
                });
            }
        });

    egui::CollapsingHeader::new(t!("inspector.runtime_toggles"))
        .default_open(false)
        .show(ui, |ui| {
            if ui
                .checkbox(&mut state.rendering.toggle_spring, t!("inspector.spring_enabled"))
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .checkbox(&mut state.rendering.toggle_cloth, t!("inspector.cloth_enabled"))
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .checkbox(
                    &mut state.rendering.toggle_collision_debug,
                    t!("inspector.collision_debug"),
                )
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .checkbox(
                    &mut state.rendering.toggle_skeleton_debug,
                    t!("inspector.skeleton_debug"),
                )
                .changed()
            {
                state.project_dirty = true;
            }
        });

    draw_expression_control(ui, state);
}

pub(super) fn draw_expression_control(ui: &mut egui::Ui, state: &mut GuiApp) {
    let avatar_expressions = state
        .app
        .active_avatar()
        .map(|a| a.asset.default_expressions.expressions.clone());

    let expressions = match avatar_expressions {
        Some(e) if !e.is_empty() => e,
        _ => return,
    };

    if state.expression_weights.len() != expressions.len() {
        state.expression_weights = expressions.iter().map(|e| e.weight).collect();
    }

    egui::CollapsingHeader::new(t!("inspector.expression_control"))
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if outlined_button(
                    ui,
                    None,
                    &t!("inspector.reset_all"),
                    color::PRIMARY,
                    true,
                )
                .clicked()
                {
                    state.expression_weights.fill(0.0);
                }
                if outlined_button(
                    ui,
                    None,
                    &t!("inspector.set_all_50"),
                    color::PRIMARY,
                    true,
                )
                .clicked()
                {
                    state.expression_weights.fill(0.5);
                }
            });

            ui.separator();

            let mut changed = false;
            for (i, expr) in expressions.iter().enumerate() {
                let weight = state.expression_weights.get_mut(i).unwrap();
                ui.horizontal(|ui| {
                    ui.label(&expr.name);
                    let response = ui.add(
                        egui::Slider::new(weight, 0.0..=1.0)
                            .text("")
                            .custom_formatter(|n, _| format!("{:.0}%", n * 100.0))
                            .custom_parser(|s| {
                                s.trim_end_matches('%')
                                    .parse::<f64>()
                                    .ok()
                                    .map(|v| v / 100.0)
                            }),
                    );
                    if response.changed() {
                        changed = true;
                    }
                });
            }

            if changed {
                if let Some(avatar) = state.app.active_avatar_mut() {
                    avatar.expression_weights = expressions
                        .iter()
                        .zip(state.expression_weights.iter())
                        .map(
                            |(expr, &w)| crate::avatar::pose_solver::ResolvedExpressionWeight {
                                name: expr.name.clone(),
                                weight: w,
                            },
                        )
                        .collect();
                }
            }

            if !changed {
                if let Some(avatar) = state.app.active_avatar() {
                    let avatar_weights: Vec<f32> =
                        avatar.expression_weights.iter().map(|w| w.weight).collect();
                    if avatar_weights.len() == state.expression_weights.len() {
                        state.expression_weights = avatar_weights;
                    }
                }
            }

            ui.separator();
            ui.label(t!("inspector.expressions_loaded", count = expressions.len()));
        });
}
