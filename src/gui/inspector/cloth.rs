use eframe::egui;

use crate::gui::components::{filled_button, outlined_button};
use crate::gui::theme::{color, icon as ic};
use crate::gui::GuiApp;
use crate::t;

pub(super) fn draw_cloth_authoring(ui: &mut egui::Ui, state: &mut GuiApp) {
    egui::CollapsingHeader::new(t!("inspector.cloth_overlay"))
        .default_open(true)
        .show(ui, |ui| {
            if state.app.editor.is_dirty() {
                ui.label(egui::RichText::new(t!("inspector.unsaved_changes")).color(egui::Color32::YELLOW));
            } else {
                ui.label(t!("inspector.no_unsaved"));
            }
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if outlined_button(
                    ui,
                    Some(ic::ADD),
                    &t!("inspector.new_overlay"),
                    color::PRIMARY,
                    true,
                )
                .clicked()
                {
                    let asset = state.app.active_avatar().map(|a| a.asset.clone());
                    if let Some(asset) = asset {
                        let avatar_id = asset.id;
                        let avatar_hash = asset.source_hash.clone();
                        state.app.editor.open(asset);
                        state.app.editor.create_overlay(avatar_id, avatar_hash);
                    }
                }
                if filled_button(ui, Some(ic::SAVE), &t!("inspector.save_overlay"), true).clicked() {
                    let _ = state.app.editor.save_overlay(None);
                }
            });

            let mut autosave_enabled = state.cloth_autosave_consent.unwrap_or(false);
            if ui
                .checkbox(&mut autosave_enabled, t!("inspector.autosave_cloth"))
                .changed()
            {
                state.cloth_autosave_consent = Some(autosave_enabled);
            }
        });

    egui::CollapsingHeader::new(t!("inspector.sim_mesh"))
        .default_open(false)
        .show(ui, |ui| {
            if let Some(avatar) = state.app.active_avatar() {
                ui.label(t!("inspector.select_garment"));

                ui.horizontal(|ui| {
                    let mat_names: Vec<String> = avatar
                        .asset
                        .materials
                        .iter()
                        .map(|m| format!("{} (id {})", m.name, m.id.0))
                        .collect();
                    let selected_text = mat_names
                        .get(state.cloth_material_pick_index)
                        .cloned()
                        .unwrap_or_else(|| "None".to_string());
                    egui::ComboBox::from_id_salt("cloth_material_pick")
                        .selected_text(&selected_text)
                        .show_ui(ui, |ui| {
                            for (i, name) in mat_names.iter().enumerate() {
                                ui.selectable_value(&mut state.cloth_material_pick_index, i, name);
                            }
                        });
                    if outlined_button(
                        ui,
                        None,
                        &t!("inspector.select_by_material"),
                        color::PRIMARY,
                        true,
                    )
                    .clicked()
                    {
                        if let Some(mat) =
                            avatar.asset.materials.get(state.cloth_material_pick_index)
                        {
                            state.region_selection =
                                crate::editor::cloth_authoring::RegionSelection::select_by_material(
                                    &avatar.asset,
                                    mat.id,
                                );
                        }
                    }
                });

                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    if outlined_button(
                        ui,
                        None,
                        &t!("inspector.grow_selection"),
                        color::PRIMARY,
                        true,
                    )
                    .clicked()
                    {
                        if let Some(ref mut sel) = state.region_selection {
                            sel.grow_selection(&avatar.asset);
                        }
                    }
                    if outlined_button(
                        ui,
                        None,
                        &t!("inspector.shrink_selection"),
                        color::PRIMARY,
                        true,
                    )
                    .clicked()
                    {
                        if let Some(ref mut sel) = state.region_selection {
                            sel.shrink_selection(&avatar.asset);
                        }
                    }
                    if outlined_button(
                        ui,
                        None,
                        &t!("inspector.clear_selection"),
                        color::ERROR,
                        true,
                    )
                    .clicked()
                    {
                        state.region_selection = None;
                    }
                });

                ui.add_space(4.0);

                if let Some(ref sel) = state.region_selection {
                    ui.label(
                        egui::RichText::new(t!("inspector.selected_vertices", count = sel.selected_vertices.len()))
                        .color(egui::Color32::from_rgb(255, 220, 80)),
                    );
                    ui.label(t!("inspector.primitive", id = sel.target_primitive.0));
                } else {
                    ui.label(t!("inspector.no_region"));
                }

                ui.add_space(4.0);

                if filled_button(ui, None, &t!("inspector.generate_sim_mesh"), true).clicked() {
                    if let Some(ref sel) = state.region_selection {
                        let verts: Vec<usize> = sel.selected_vertices.iter().copied().collect();
                        if let Some(sim_mesh) = crate::editor::cloth_authoring::generate_sim_mesh(
                            &avatar.asset,
                            sel.target_primitive,
                            &verts,
                        ) {
                            if let Some(ref mut overlay) = state.app.editor.overlay_asset {
                                overlay.simulation_mesh = sim_mesh;
                                state.app.editor.set_dirty();
                                state.push_notification(t!("inspector.sim_mesh_generated").to_string());
                            }
                        }
                    }
                }
            } else {
                ui.label(t!("inspector.no_avatar_loaded"));
            }
        });

    egui::CollapsingHeader::new(t!("inspector.pin_binding"))
        .default_open(false)
        .show(ui, |ui| {
            if let Some(avatar) = state.app.active_avatar() {
                let nodes = &avatar.asset.skeleton.nodes;
                if nodes.is_empty() {
                    ui.label(t!("inspector.no_skeleton_nodes"));
                } else {
                    if state.cloth_pin_node_index >= nodes.len() {
                        state.cloth_pin_node_index = 0;
                    }
                    let node_names: Vec<String> = nodes
                        .iter()
                        .map(|n| format!("{} (id {})", n.name, n.id.0))
                        .collect();
                    let selected_text = node_names
                        .get(state.cloth_pin_node_index)
                        .cloned()
                        .unwrap_or_else(|| "None".to_string());
                    egui::ComboBox::from_id_salt("cloth_pin_node")
                        .selected_text(&selected_text)
                        .show_ui(ui, |ui| {
                            for (i, name) in node_names.iter().enumerate() {
                                ui.selectable_value(&mut state.cloth_pin_node_index, i, name);
                            }
                        });

                    ui.add_space(4.0);

                    let pin_node_id = nodes[state.cloth_pin_node_index].id;
                    let pin_node_name = nodes[state.cloth_pin_node_index].name.clone();

                    ui.horizontal(|ui| {
                        if filled_button(ui, Some(ic::ADD), &t!("inspector.create_pin_set"), true)
                            .clicked()
                        {
                            if state.region_selection.is_some() {
                                if let Some(ref mut overlay) = state.app.editor.overlay_asset {
                                    let sim_mesh = &overlay.simulation_mesh;
                                    if !sim_mesh.vertices.is_empty() {
                                        let pinned_verts: Vec<u32> = sim_mesh
                                            .vertices
                                            .iter()
                                            .enumerate()
                                            .filter(|(_, v)| v.pinned)
                                            .map(|(i, _)| i as u32)
                                            .collect();
                                        if !pinned_verts.is_empty() {
                                            let pin = crate::asset::ClothPin {
                                                sim_vertex_indices: pinned_verts,
                                                binding_node: crate::asset::NodeRef {
                                                    id: pin_node_id,
                                                    name: pin_node_name,
                                                },
                                                offset: [0.0, 0.0, 0.0],
                                            };
                                            overlay.pins.push(pin);
                                            state.app.editor.set_dirty();
                                            state.push_notification(t!("inspector.pin_set_created").to_string());
                                        } else {
                                            state.push_notification(
                                                t!("inspector.no_pinned_verts").to_string(),
                                            );
                                        }
                                    }
                                }
                            } else {
                                state.push_notification(t!("inspector.select_region_first").to_string());
                            }
                        }
                    });

                    ui.add_space(4.0);

                    if let Some(ref overlay) = state.app.editor.overlay_asset {
                        let total_pins: usize = overlay
                            .pins
                            .iter()
                            .map(|p| p.sim_vertex_indices.len())
                            .sum();
                        ui.label(format!(
                            "Pin sets: {} ({} vertices bound)",
                            overlay.pins.len(),
                            total_pins
                        ));
                        for (i, pin) in overlay.pins.iter().enumerate() {
                            ui.label(
                                egui::RichText::new(format!(
                                    "  Pin {}: {} verts -> {}",
                                    i,
                                    pin.sim_vertex_indices.len(),
                                    pin.binding_node.name
                                ))
                                .small()
                                .color(egui::Color32::from_rgb(160, 200, 255)),
                            );
                        }
                    }
                }
            } else {
                ui.label(t!("inspector.no_avatar_loaded"));
            }
        });

    egui::CollapsingHeader::new(t!("inspector.constraints"))
        .default_open(false)
        .show(ui, |ui| {
            ui.add(
                egui::Slider::new(&mut state.cloth_distance_stiffness, 0.0..=1.0)
                    .text(t!("inspector.distance_stiffness")),
            );
            ui.checkbox(&mut state.cloth_bend_enabled, t!("inspector.bend_constraints"));
            if state.cloth_bend_enabled {
                ui.add(
                    egui::Slider::new(&mut state.cloth_bend_stiffness, 0.0..=1.0)
                        .text(t!("inspector.bend_stiffness")),
                );
            }

            if let Some(ref mut overlay) = state.app.editor.overlay_asset {
                ui.add_space(4.0);
                ui.label(t!("inspector.distance_constraints", count = overlay.constraints.distance_constraints.len()));
                ui.label(t!("inspector.bend_constraints_count", count = overlay.constraints.bend_constraints.len()));
                if filled_button(ui, None, &t!("inspector.apply_constraints"), true).clicked() {
                    for dc in &mut overlay.constraints.distance_constraints {
                        dc.stiffness = state.cloth_distance_stiffness;
                    }
                    for bc in &mut overlay.constraints.bend_constraints {
                        bc.stiffness = state.cloth_bend_stiffness;
                    }
                    if !state.cloth_bend_enabled {
                        overlay.constraints.bend_constraints.clear();
                    }
                    state.app.editor.set_dirty();
                    state.push_notification(t!("inspector.constraints_applied").to_string());
                }
            } else {
                ui.label(t!("inspector.no_overlay_active"));
            }
        });

    egui::CollapsingHeader::new(t!("inspector.collision_proxies"))
        .default_open(false)
        .show(ui, |ui| {
            if let Some(avatar) = state.app.active_avatar() {
                let collider_count = avatar.asset.colliders.len();
                if state.cloth_collider_toggles.len() != collider_count {
                    state.cloth_collider_toggles = vec![true; collider_count];
                }
                if collider_count == 0 {
                    ui.label(t!("inspector.no_colliders"));
                } else {
                    for (i, collider) in avatar.asset.colliders.iter().enumerate() {
                        let node_name = avatar
                            .asset
                            .skeleton
                            .nodes
                            .iter()
                            .find(|n| n.id == collider.node)
                            .map(|n| n.name.as_str())
                            .unwrap_or("unknown");
                        let shape_label = match &collider.shape {
                            crate::asset::ColliderShape::Sphere { radius } => {
                                format!("Sphere(r={:.3})", radius)
                            }
                            crate::asset::ColliderShape::Capsule { radius, height } => {
                                format!("Capsule(r={:.3}, h={:.3})", radius, height)
                            }
                        };
                        ui.checkbox(
                            &mut state.cloth_collider_toggles[i],
                            format!("{}: {} ({})", collider.id.0, node_name, shape_label),
                        );
                    }
                }
            } else {
                ui.label(t!("inspector.no_avatar_loaded"));
            }
        });

    egui::CollapsingHeader::new(t!("inspector.solver_params"))
        .default_open(false)
        .show(ui, |ui| {
            if let Some(avatar) = state.app.active_avatar_mut() {
                if let Some(ref mut sim) = avatar.cloth_sim {
                    ui.horizontal(|ui| {
                        ui.label(t!("inspector.iterations"));
                        ui.add(egui::DragValue::new(&mut sim.solver_iterations).range(1..=32));
                    });
                    ui.horizontal(|ui| {
                        ui.label(t!("inspector.gravity"));
                        ui.add(
                            egui::DragValue::new(&mut sim.gravity[1])
                                .speed(0.1)
                                .range(-20.0..=0.0),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label(t!("inspector.damping"));
                        ui.add(
                            egui::DragValue::new(&mut sim.damping)
                                .speed(0.01)
                                .range(0.0..=1.0),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label(t!("inspector.collision_margin"));
                        ui.add(
                            egui::DragValue::new(&mut sim.collision_margin)
                                .speed(0.001)
                                .range(0.0..=0.1),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label(t!("inspector.wind_response"));
                        ui.add(
                            egui::DragValue::new(&mut sim.wind_response)
                                .speed(0.1)
                                .range(0.0..=10.0),
                        );
                    });
                    ui.checkbox(&mut sim.self_collision, t!("inspector.self_collision"));
                    if sim.self_collision {
                        ui.horizontal(|ui| {
                            ui.label(t!("inspector.self_collision_radius"));
                            ui.add(
                                egui::DragValue::new(&mut sim.self_collision_radius)
                                    .speed(0.001)
                                    .range(0.001..=0.1),
                            );
                        });
                    }
                    ui.add_space(4.0);
                    ui.label(t!("inspector.particles", count = sim.particle_count()));
                    ui.label(t!("inspector.distance_constraints", count = sim.distance_constraints.len()));
                    ui.label(t!("inspector.bend_constraints_count", count = sim.bend_constraints.len()));
                    ui.label(t!("inspector.pin_targets", count = sim.pin_targets.len()));
                } else {
                    ui.label(t!("inspector.no_cloth_sim"));
                }

                if let Some(avatar) = state.app.active_avatar() {
                    if !avatar.cloth_overlays.is_empty() {
                        ui.add_space(4.0);
                        ui.label(t!("inspector.overlay_slots"));
                        for (i, slot) in avatar.cloth_overlays.iter().enumerate() {
                            ui.label(format!(
                                "  #{}: {} particles, enabled={}",
                                i,
                                slot.sim.particle_count(),
                                slot.enabled
                            ));
                        }
                    }
                }
            } else {
                ui.label(t!("inspector.no_avatar_loaded"));
            }
        });

    egui::CollapsingHeader::new(t!("inspector.preview"))
        .default_open(false)
        .show(ui, |ui| {
            if let Some(avatar) = state.app.active_avatar_mut() {
                ui.checkbox(&mut avatar.cloth_enabled, t!("inspector.cloth_simulation"));

                ui.horizontal(|ui| {
                    if state.cloth_sim_playing {
                        if outlined_button(
                            ui,
                            Some(ic::PAUSE),
                            &t!("inspector.pause"),
                            color::PRIMARY,
                            true,
                        )
                        .clicked()
                        {
                            state.cloth_sim_playing = false;
                        }
                    } else if filled_button(ui, Some(ic::PLAY), &t!("inspector.play"), true)
                        .clicked()
                    {
                        state.cloth_sim_playing = true;
                    }
                    if outlined_button(ui, None, &t!("inspector.step"), color::PRIMARY, true)
                        .clicked()
                    {
                        crate::simulation::cloth_solver::step_cloth(1.0 / 60.0, avatar, &[]);
                    }
                    if outlined_button(ui, None, &t!("inspector.reset"), color::ERROR, true)
                        .clicked()
                    {
                        avatar.cloth_sim = None;
                        avatar.cloth_sim_buffers = None;
                    }
                });

                ui.add_space(4.0);

                if let Some(ref cs) = avatar.cloth_state {
                    ui.label(t!("inspector.primary_sim_positions", count = cs.sim_positions.len()));
                    ui.label(t!("inspector.primary_deform", version = cs.deform_output.version));
                } else if !avatar.cloth_overlays.is_empty() {
                    ui.label(t!("inspector.primary_state_none"));
                } else {
                    ui.label(t!("inspector.no_cloth_state"));
                }

                for (i, slot) in avatar.cloth_overlays.iter().enumerate() {
                    ui.label(format!(
                        "Overlay #{}: {} positions, v={}",
                        i,
                        slot.state.sim_positions.len(),
                        slot.state.deform_output.version
                    ));
                }
            } else {
                ui.label(t!("inspector.no_avatar_loaded"));
            }
        });
}
