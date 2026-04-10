use eframe::egui;

use crate::gui::{AppMode, GuiApp};

pub fn draw(ctx: &egui::Context, state: &mut GuiApp) {
    egui::SidePanel::right("inspector")
        .resizable(true)
        .default_width(300.0)
        .min_width(250.0)
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading(state.mode.label());
            });
            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| match state.mode {
                AppMode::Preview => draw_preview(ui, state),
                AppMode::TrackingSetup => draw_tracking(ui, state),
                AppMode::Rendering => draw_rendering(ui, state),
                AppMode::Output => draw_output(ui, state),
                AppMode::ClothAuthoring => draw_cloth_authoring(ui, state),
            });
        });
}

fn draw_preview(ui: &mut egui::Ui, state: &mut GuiApp) {
    egui::CollapsingHeader::new("Avatar Asset")
        .default_open(true)
        .show(ui, |ui| {
            if let Some(avatar) = &state.app.avatar {
                ui.label(format!("Source: {}", avatar.asset.source_path.display()));
                ui.label(format!("ID: {}", avatar.asset.id.0));
                ui.label(format!("Meshes: {}", avatar.asset.meshes.len()));
                ui.label(format!("Materials: {}", avatar.asset.materials.len()));
                ui.label(format!(
                    "Spring Chains: {}",
                    avatar.asset.spring_bones.len()
                ));
                ui.label(format!("Colliders: {}", avatar.asset.colliders.len()));
                if let Some(cloth_id) = &avatar.attached_cloth {
                    ui.label(format!("Cloth Overlay: {}", cloth_id.0));
                } else {
                    ui.label("Cloth Overlay: None");
                }
            } else {
                ui.label("No avatar loaded");
            }
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.button("Reload").clicked() {}
                if ui.button("Detach").clicked() {
                    state.app.avatar = None;
                }
            });
        });

    egui::CollapsingHeader::new("Transform")
        .default_open(true)
        .show(ui, |ui| {
            ui.label("Position");
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut state.transform.position[0])
                        .prefix("X: ")
                        .speed(0.01),
                );
                ui.add(
                    egui::DragValue::new(&mut state.transform.position[1])
                        .prefix("Y: ")
                        .speed(0.01),
                );
                ui.add(
                    egui::DragValue::new(&mut state.transform.position[2])
                        .prefix("Z: ")
                        .speed(0.01),
                );
            });
            ui.label("Rotation");
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut state.transform.rotation[0])
                        .prefix("X: ")
                        .speed(0.1),
                );
                ui.add(
                    egui::DragValue::new(&mut state.transform.rotation[1])
                        .prefix("Y: ")
                        .speed(0.1),
                );
                ui.add(
                    egui::DragValue::new(&mut state.transform.rotation[2])
                        .prefix("Z: ")
                        .speed(0.1),
                );
            });
            ui.horizontal(|ui| {
                ui.label("Scale");
                ui.add(
                    egui::DragValue::new(&mut state.transform.scale)
                        .speed(0.01)
                        .range(0.01..=100.0),
                );
            });
            if ui.button("Reset Transform").clicked() {
                state.transform.position = [0.0, 0.0, 0.0];
                state.transform.rotation = [0.0, 0.0, 0.0];
                state.transform.scale = 1.0;
            }
        });

    egui::CollapsingHeader::new("Cloth Attachment")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Attach Overlay").clicked() {
                    if let Some(avatar) = state.app.avatar.as_mut() {
                        avatar.attach_cloth(crate::asset::ClothOverlayId(1));
                    }
                }
                if ui.button("Detach Overlay").clicked() {
                    if let Some(avatar) = state.app.avatar.as_mut() {
                        avatar.detach_cloth();
                    }
                }
            });
            if let Some(avatar) = state.app.avatar.as_mut() {
                ui.checkbox(&mut avatar.cloth_enabled, "Enable Cloth");
            }
        });

    egui::CollapsingHeader::new("Scene Background")
        .default_open(false)
        .show(ui, |ui| {
            ui.checkbox(
                &mut state.rendering.transparent_background,
                "Transparent Background",
            );
            if !state.rendering.transparent_background {
                ui.horizontal(|ui| {
                    ui.label("Color");
                    ui.color_edit_button_rgb(&mut state.rendering.background_color);
                });
            }
        });

    egui::CollapsingHeader::new("Runtime Toggles")
        .default_open(false)
        .show(ui, |ui| {
            ui.checkbox(&mut state.tracking.toggle_tracking, "Tracking Enabled");
            ui.checkbox(&mut state.rendering.toggle_spring, "Spring Enabled");
            ui.checkbox(&mut state.rendering.toggle_cloth, "Cloth Enabled");
            ui.checkbox(
                &mut state.rendering.toggle_collision_debug,
                "Collision Debug",
            );
            ui.checkbox(
                &mut state.rendering.toggle_skeleton_debug,
                "Skeleton Debug",
            );
        });
}

fn draw_tracking(ui: &mut egui::Ui, state: &mut GuiApp) {
    egui::CollapsingHeader::new("Input Device")
        .default_open(true)
        .show(ui, |ui| {
            egui::ComboBox::from_label("Camera")
                .selected_text(format!("Camera {}", state.camera_index))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.camera_index, 0, "Camera 0");
                    ui.selectable_value(&mut state.camera_index, 1, "Camera 1");
                    ui.selectable_value(&mut state.camera_index, 2, "Camera 2");
                });
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if state.tracking.camera_running {
                    if ui.button("Stop Camera").clicked() {
                        state.tracking.camera_running = false;
                    }
                } else if ui.button("Start Camera").clicked() {
                    state.tracking.camera_running = true;
                }
            });
            egui::ComboBox::from_label("Resolution")
                .selected_text(
                    *["640x480", "1280x720", "1920x1080"]
                        .get(state.tracking.camera_resolution_index)
                        .unwrap_or(&"Unknown"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut state.tracking.camera_resolution_index,
                        0,
                        "640x480",
                    );
                    ui.selectable_value(
                        &mut state.tracking.camera_resolution_index,
                        1,
                        "1280x720",
                    );
                    ui.selectable_value(
                        &mut state.tracking.camera_resolution_index,
                        2,
                        "1920x1080",
                    );
                });
            egui::ComboBox::from_label("Frame Rate")
                .selected_text(
                    *["30 fps", "60 fps"]
                        .get(state.tracking.camera_framerate_index)
                        .unwrap_or(&"Unknown"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut state.tracking.camera_framerate_index,
                        0,
                        "30 fps",
                    );
                    ui.selectable_value(
                        &mut state.tracking.camera_framerate_index,
                        1,
                        "60 fps",
                    );
                });
        });

    egui::CollapsingHeader::new("Capture Format")
        .default_open(false)
        .show(ui, |ui| {
            ui.checkbox(&mut state.tracking.tracking_mirror, "Mirror Preview");
        });

    egui::CollapsingHeader::new("Inference Status")
        .default_open(true)
        .show(ui, |ui| {
            let (status_color, status_text) = if state.tracking.camera_running {
                (egui::Color32::GREEN, "Running")
            } else {
                (egui::Color32::GRAY, "Stopped")
            };
            ui.label(egui::RichText::new(status_text).color(status_color));

            if let Some(tracking) = &state.app.last_tracking_pose {
                ui.label(format!("Timestamp: {}", tracking.source_timestamp));
            } else {
                ui.label("No tracking data");
            }
        });

    egui::CollapsingHeader::new("Retargeting")
        .default_open(true)
        .show(ui, |ui| {
            ui.checkbox(&mut state.tracking.hand_tracking_enabled, "Hand Tracking");
            ui.checkbox(&mut state.tracking.face_tracking_enabled, "Face Tracking");
            if let Some(tracking) = &state.app.last_tracking_pose {
                ui.separator();
                ui.label("Confidence");
                let conf = &tracking.confidence;
                draw_confidence_bar(ui, "Head", conf.head_confidence);
                draw_confidence_bar(ui, "Torso", conf.torso_confidence);
                draw_confidence_bar(ui, "Left Arm", conf.left_arm_confidence);
                draw_confidence_bar(ui, "Right Arm", conf.right_arm_confidence);
                draw_confidence_bar(ui, "Face", conf.face_confidence);
            }
        });

    egui::CollapsingHeader::new("Smoothing and Confidence")
        .default_open(false)
        .show(ui, |ui| {
            ui.add(
                egui::Slider::new(&mut state.tracking.smoothing_strength, 0.0..=1.0)
                    .text("Smoothing"),
            );
            ui.add(
                egui::Slider::new(&mut state.tracking.confidence_threshold, 0.0..=1.0)
                    .text("Threshold"),
            );
        });
}

fn draw_confidence_bar(ui: &mut egui::Ui, label: &str, value: f32) {
    ui.horizontal(|ui| {
        ui.label(format!("{:>10}", label));
        let color = if value > 0.8 {
            egui::Color32::GREEN
        } else if value > 0.5 {
            egui::Color32::YELLOW
        } else {
            egui::Color32::from_rgb(255, 100, 100)
        };
        ui.label(egui::RichText::new(format!("{:.0}%", value * 100.0)).color(color));
    });
}

fn draw_rendering(ui: &mut egui::Ui, state: &mut GuiApp) {
    egui::CollapsingHeader::new("Material Mode")
        .default_open(true)
        .show(ui, |ui| {
            ui.radio_value(&mut state.rendering.material_mode_index, 0, "Unlit");
            ui.radio_value(&mut state.rendering.material_mode_index, 1, "Simple Lit");
            ui.radio_value(&mut state.rendering.material_mode_index, 2, "Toon-like");
        });

    egui::CollapsingHeader::new("Toon Controls")
        .default_open(true)
        .show(ui, |ui| {
            ui.add_enabled(
                state.rendering.material_mode_index == 2,
                egui::Slider::new(&mut state.rendering.toon_ramp_threshold, 0.0..=1.0)
                    .text("Ramp Threshold"),
            );
            ui.add_enabled(
                state.rendering.material_mode_index == 2,
                egui::Slider::new(&mut state.rendering.shadow_softness, 0.0..=1.0)
                    .text("Shadow Softness"),
            );
        });

    egui::CollapsingHeader::new("Outline Controls")
        .default_open(false)
        .show(ui, |ui| {
            ui.checkbox(&mut state.rendering.outline_enabled, "Enable Outline");
            ui.add_enabled(
                state.rendering.outline_enabled,
                egui::Slider::new(&mut state.rendering.outline_width, 0.0..=0.1).text("Width"),
            );
            ui.add_enabled_ui(state.rendering.outline_enabled, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Color");
                    ui.color_edit_button_rgb(&mut state.rendering.outline_color);
                });
            });
        });

    egui::CollapsingHeader::new("Lighting")
        .default_open(true)
        .show(ui, |ui| {
            ui.label("Light Direction");
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut state.rendering.main_light_dir[0])
                        .prefix("X: ")
                        .speed(0.01),
                );
                ui.add(
                    egui::DragValue::new(&mut state.rendering.main_light_dir[1])
                        .prefix("Y: ")
                        .speed(0.01),
                );
                ui.add(
                    egui::DragValue::new(&mut state.rendering.main_light_dir[2])
                        .prefix("Z: ")
                        .speed(0.01),
                );
            });
            ui.add(
                egui::Slider::new(&mut state.rendering.main_light_intensity, 0.0..=5.0)
                    .text("Intensity"),
            );
            ui.horizontal(|ui| {
                ui.label("Ambient");
                ui.color_edit_button_rgb(&mut state.rendering.ambient_intensity);
            });
        });

    egui::CollapsingHeader::new("Composition")
        .default_open(false)
        .show(ui, |ui| {
            ui.add(
                egui::Slider::new(&mut state.rendering.camera_fov, 10.0..=120.0)
                    .text("Camera FOV"),
            );
            ui.checkbox(&mut state.rendering.alpha_preview, "Alpha Preview");
        });
}

fn draw_output(ui: &mut egui::Ui, state: &mut GuiApp) {
    let sink_names = [
        "Virtual Camera",
        "Shared Texture",
        "Shared Memory",
        "Image Sequence",
    ];

    egui::CollapsingHeader::new("Sink Selection")
        .default_open(true)
        .show(ui, |ui| {
            egui::ComboBox::from_label("Output Sink")
                .selected_text(
                    *sink_names
                        .get(state.output.output_sink_index)
                        .unwrap_or(&"Unknown"),
                )
                .show_ui(ui, |ui| {
                    for (i, name) in sink_names.iter().enumerate() {
                        ui.selectable_value(&mut state.output.output_sink_index, i, *name);
                    }
                });
        });

    egui::CollapsingHeader::new("Frame Format")
        .default_open(true)
        .show(ui, |ui| {
            egui::ComboBox::from_label("Resolution")
                .selected_text(
                    *["1920x1080", "1280x720", "640x480"]
                        .get(state.output.output_resolution_index)
                        .unwrap_or(&"Unknown"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut state.output.output_resolution_index,
                        0,
                        "1920x1080",
                    );
                    ui.selectable_value(
                        &mut state.output.output_resolution_index,
                        1,
                        "1280x720",
                    );
                    ui.selectable_value(
                        &mut state.output.output_resolution_index,
                        2,
                        "640x480",
                    );
                });
            egui::ComboBox::from_label("Frame Rate")
                .selected_text(
                    *["60 fps", "30 fps"]
                        .get(state.output.output_framerate_index)
                        .unwrap_or(&"Unknown"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut state.output.output_framerate_index,
                        0,
                        "60 fps",
                    );
                    ui.selectable_value(
                        &mut state.output.output_framerate_index,
                        1,
                        "30 fps",
                    );
                });
            ui.checkbox(&mut state.output.output_has_alpha, "RGBA (with alpha)");
            egui::ComboBox::from_label("Color Space")
                .selected_text(
                    *["sRGB", "Linear sRGB"]
                        .get(state.output.output_color_space_index)
                        .unwrap_or(&"Unknown"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut state.output.output_color_space_index,
                        0,
                        "sRGB",
                    );
                    ui.selectable_value(
                        &mut state.output.output_color_space_index,
                        1,
                        "Linear sRGB",
                    );
                });
        });

    egui::CollapsingHeader::new("Synchronization")
        .default_open(false)
        .show(ui, |ui| {
            ui.label("Handoff: GPU Shared Frame");
            ui.label("Fallback: CPU Readback");
            ui.label(egui::RichText::new("GPU path active").color(egui::Color32::GREEN));
        });

    egui::CollapsingHeader::new("Diagnostics")
        .default_open(true)
        .show(ui, |ui| {
            ui.label(egui::RichText::new("Connected").color(egui::Color32::GREEN));
            ui.label(format!("Queue Depth: {}", state.app.output.queue_depth()));
            ui.label(format!(
                "Dropped Frames: {}",
                state.app.output.dropped_count()
            ));
        });
}

fn draw_cloth_authoring(ui: &mut egui::Ui, state: &mut GuiApp) {
    egui::CollapsingHeader::new("Cloth Overlay")
        .default_open(true)
        .show(ui, |ui| {
            if state.app.editor.is_dirty() {
                ui.label(egui::RichText::new("Unsaved changes").color(egui::Color32::YELLOW));
            } else {
                ui.label("No unsaved changes");
            }
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.button("New Overlay").clicked() {
                    let asset = state.app.avatar.as_ref().map(|a| a.asset.clone());
                    if let Some(asset) = asset {
                        let avatar_id = asset.id;
                        let avatar_hash = asset.source_hash.clone();
                        state.app.editor.open(asset);
                        state.app.editor.create_overlay(avatar_id, avatar_hash);
                    }
                }
                if ui.button("Save Overlay").clicked() {
                    let _ = state.app.editor.save_overlay(None);
                }
            });
        });

    egui::CollapsingHeader::new("Simulation Mesh")
        .default_open(false)
        .show(ui, |ui| {
            ui.label("Select garment region to author");
            ui.label("(Cloth mesh tools will appear here)");
        });

    egui::CollapsingHeader::new("Constraints")
        .default_open(false)
        .show(ui, |ui| {
            ui.label("Distance constraints");
            ui.label("Bend constraints");
            ui.label("(Constraint editing tools will appear here)");
        });

    egui::CollapsingHeader::new("Collision Proxies")
        .default_open(false)
        .show(ui, |ui| {
            ui.label("Body collision proxies for cloth");
            ui.label("(Proxy editing tools will appear here)");
        });

    egui::CollapsingHeader::new("Preview")
        .default_open(false)
        .show(ui, |ui| {
            if let Some(avatar) = state.app.avatar.as_mut() {
                ui.checkbox(&mut avatar.cloth_enabled, "Cloth Simulation");
                ui.label(format!(
                    "Sim positions: {}",
                    avatar
                        .cloth_state
                        .as_ref()
                        .map(|cs| cs.sim_positions.len())
                        .unwrap_or(0)
                ));
            } else {
                ui.label("No avatar loaded");
            }
        });
}
