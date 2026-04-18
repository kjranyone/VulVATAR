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

            egui::ScrollArea::vertical().show(ui, |ui| {
                match state.mode {
                    AppMode::Avatar => draw_avatar(ui, state),
                    AppMode::Preview => draw_preview(ui, state),
                    AppMode::TrackingSetup => draw_tracking(ui, state),
                    AppMode::Rendering => draw_rendering(ui, state),
                    AppMode::Output => draw_output(ui, state),
                    AppMode::ClothAuthoring => draw_cloth_authoring(ui, state),
                }

                ui.separator();
                egui::CollapsingHeader::new("Camera Transform")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Yaw:");
                            if ui
                                .add(
                                    egui::DragValue::new(&mut state.camera_orbit.yaw_deg)
                                        .speed(0.5),
                                )
                                .changed()
                            {
                                state.project_dirty = true;
                            }
                            ui.label("Pitch:");
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
                            ui.label("Distance:");
                            if ui
                                .add(
                                    egui::DragValue::new(&mut state.camera_orbit.distance)
                                        .speed(0.05)
                                        .range(0.1..=100.0),
                                )
                                .changed()
                            {
                                state.project_dirty = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Pan X:");
                            if ui
                                .add(
                                    egui::DragValue::new(&mut state.camera_orbit.pan[0])
                                        .speed(0.01),
                                )
                                .changed()
                            {
                                state.project_dirty = true;
                            }
                            ui.label("Y:");
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
                        if ui.button("Reset Camera").clicked() {
                            state.camera_orbit.yaw_deg = 0.0;
                            state.camera_orbit.pitch_deg = 0.0;
                            state.camera_orbit.distance = 5.0;
                            state.camera_orbit.pan = [0.0, 0.0];
                            state.project_dirty = true;
                        }
                    });
            });
        });
}

fn draw_avatar(ui: &mut egui::Ui, state: &mut GuiApp) {
    egui::CollapsingHeader::new("Current Avatar")
        .default_open(true)
        .show(ui, |ui| {
            if let Some(avatar) = state.app.active_avatar() {
                let asset = &avatar.asset;
                let meta = &asset.vrm_meta;

                let version_color = match meta.spec_version {
                    crate::asset::VrmSpecVersion::V1 => egui::Color32::from_rgb(120, 200, 140),
                    crate::asset::VrmSpecVersion::V0 => egui::Color32::from_rgb(220, 180, 100),
                    crate::asset::VrmSpecVersion::Unknown => egui::Color32::from_rgb(200, 100, 100),
                };
                ui.horizontal(|ui| {
                    ui.label("Spec:");
                    ui.label(
                        egui::RichText::new(meta.spec_version.label())
                            .strong()
                            .color(version_color),
                    );
                    if let Some(raw) = &meta.spec_version_raw {
                        ui.label(
                            egui::RichText::new(format!("({})", raw))
                                .small()
                                .color(egui::Color32::GRAY),
                        );
                    }
                });

                if meta.spec_version == crate::asset::VrmSpecVersion::V0 {
                    ui.label(
                        egui::RichText::new(
                            "Note: VRM 0.x — humanoid/expressions/springs not loaded by this build.",
                        )
                        .small()
                        .color(egui::Color32::from_rgb(220, 180, 100)),
                    );
                }

                let title = meta.title.as_deref().unwrap_or("—");
                ui.horizontal(|ui| {
                    ui.label("Title:");
                    ui.label(title);
                });
                let authors = if meta.authors.is_empty() {
                    "—".to_string()
                } else {
                    meta.authors.join(", ")
                };
                ui.horizontal(|ui| {
                    ui.label("Authors:");
                    ui.label(authors);
                });
                if let Some(v) = &meta.model_version {
                    ui.horizontal(|ui| {
                        ui.label("Model version:");
                        ui.label(v);
                    });
                }
                if let Some(c) = &meta.contact_information {
                    ui.horizontal(|ui| {
                        ui.label("Contact:");
                        ui.label(c);
                    });
                }
                if let Some(l) = &meta.license {
                    ui.horizontal(|ui| {
                        ui.label("License:");
                        ui.label(l);
                    });
                }
                if let Some(cr) = &meta.copyright_information {
                    ui.horizontal(|ui| {
                        ui.label("Copyright:");
                        ui.label(cr);
                    });
                }
                if !meta.references.is_empty() {
                    ui.collapsing("References", |ui| {
                        for r in &meta.references {
                            ui.label(
                                egui::RichText::new(r).small().color(egui::Color32::GRAY),
                            );
                        }
                    });
                }

                ui.separator();
                ui.label(
                    egui::RichText::new(format!("Source: {}", asset.source_path.display()))
                        .small()
                        .color(egui::Color32::GRAY),
                );
                ui.horizontal(|ui| {
                    ui.label(format!("Meshes: {}", asset.meshes.len()));
                    ui.label(format!("Materials: {}", asset.materials.len()));
                });
                ui.horizontal(|ui| {
                    ui.label(format!("Spring chains: {}", asset.spring_bones.len()));
                    ui.label(format!("Colliders: {}", asset.colliders.len()));
                });
                ui.horizontal(|ui| {
                    ui.label(format!(
                        "Humanoid: {}",
                        if asset.humanoid.is_some() { "yes" } else { "no" }
                    ));
                    ui.label(format!(
                        "Expressions: {}",
                        asset.default_expressions.expressions.len()
                    ));
                    ui.label(format!(
                        "Animations: {}",
                        asset.animation_clips.len()
                    ));
                });

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui.button("Reload").clicked() {
                        state.app.reload_avatar();
                    }
                    if ui.button("Detach").clicked() {
                        if !state.app.avatars.is_empty() {
                            state.app.remove_avatar_at(state.app.active_avatar_index);
                        }
                    }
                });
            } else {
                ui.label("No avatar loaded");
                ui.label(
                    egui::RichText::new("Use the library below or the top bar to load one.")
                        .small()
                        .color(egui::Color32::GRAY),
                );
            }
        });

    draw_model_library(ui, state);
}

fn draw_preview(ui: &mut egui::Ui, state: &mut GuiApp) {
    egui::CollapsingHeader::new("Avatar Asset")
        .default_open(true)
        .show(ui, |ui| {
            if let Some(avatar) = state.app.active_avatar() {
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
                    ui.label(format!("Primary Cloth Overlay: {}", cloth_id.0));
                } else {
                    ui.label("Primary Cloth Overlay: None");
                }
                ui.label(format!(
                    "Additional Cloth Overlays: {}",
                    avatar.cloth_overlay_count()
                ));
            } else {
                ui.label("No avatar loaded");
            }
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.button("Reload").clicked() {
                    state.app.reload_avatar();
                }
                if ui.button("Detach").clicked() {
                    if !state.app.avatars.is_empty() {
                        state.app.remove_avatar_at(state.app.active_avatar_index);
                    }
                }
            });
        });

    egui::CollapsingHeader::new("Transform")
        .default_open(true)
        .show(ui, |ui| {
            ui.label("Position");
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
            ui.label("Rotation");
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
                ui.label("Scale");
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
            if ui.button("Reset Transform").clicked() {
                state.transform.position = [0.0, 0.0, 0.0];
                state.transform.rotation = [0.0, 0.0, 0.0];
                state.transform.scale = 1.0;
                state.project_dirty = true;
            }
        });

    egui::CollapsingHeader::new("Cloth Attachment")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Attach Primary").clicked() {
                    let overlay_id = state
                        .app
                        .editor
                        .active_overlay()
                        .unwrap_or(crate::asset::ClothOverlayId(1));
                    if let Some(avatar) = state.app.active_avatar_mut() {
                        avatar.attach_cloth(overlay_id);
                    }
                }
                if ui.button("Detach Primary").clicked() {
                    if let Some(avatar) = state.app.active_avatar_mut() {
                        avatar.detach_cloth();
                    }
                }
            });
            ui.horizontal(|ui| {
                if ui.button("Add Overlay Slot").clicked() {
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
            });
            if let Some(avatar) = state.app.active_avatar_mut() {
                ui.checkbox(&mut avatar.cloth_enabled, "Enable Cloth");
                for (i, slot) in avatar.cloth_overlays.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut slot.enabled, format!("Overlay #{}", i));
                    });
                }
            }
        });

    egui::CollapsingHeader::new("Scene Background")
        .default_open(false)
        .show(ui, |ui| {
            if ui
                .checkbox(
                    &mut state.rendering.transparent_background,
                    "Transparent Background",
                )
                .changed()
            {
                state.project_dirty = true;
            }
            if !state.rendering.transparent_background {
                ui.horizontal(|ui| {
                    ui.label("Color");
                    if ui
                        .color_edit_button_rgb(&mut state.rendering.background_color)
                        .changed()
                    {
                        state.project_dirty = true;
                    }
                });
            }
        });

    egui::CollapsingHeader::new("Runtime Toggles")
        .default_open(false)
        .show(ui, |ui| {
            if ui
                .checkbox(&mut state.rendering.toggle_spring, "Spring Enabled")
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .checkbox(&mut state.rendering.toggle_cloth, "Cloth Enabled")
                .changed()
            {
                state.project_dirty = true;
            }
            ui.checkbox(
                &mut state.rendering.toggle_collision_debug,
                "Collision Debug",
            );
            ui.checkbox(&mut state.rendering.toggle_skeleton_debug, "Skeleton Debug");
        });

    draw_expression_control(ui, state);
}

fn draw_tracking(ui: &mut egui::Ui, state: &mut GuiApp) {
    if ui
        .checkbox(&mut state.tracking.toggle_tracking, "Tracking Enabled")
        .changed()
    {
        state.project_dirty = true;
    }
    ui.add_space(4.0);

    egui::CollapsingHeader::new("Input Device")
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let prev_camera = state.camera_index;
                let selected_text = state
                    .available_cameras
                    .iter()
                    .find(|c| c.index == state.camera_index)
                    .map(|c| format!("{}: {}", c.index, c.name))
                    .unwrap_or_else(|| format!("Camera {}", state.camera_index));
                egui::ComboBox::from_label("Camera")
                    .selected_text(selected_text)
                    .show_ui(ui, |ui| {
                        if state.available_cameras.is_empty() {
                            ui.label("No cameras detected. Click 🔄 to scan.");
                        } else {
                            for cam in &state.available_cameras {
                                let label = format!("{}: {}", cam.index, cam.name);
                                ui.selectable_value(&mut state.camera_index, cam.index, label);
                            }
                        }
                    });
                if state.camera_index != prev_camera {
                    state.project_dirty = true;
                }
                if ui.button("🔄").clicked() {
                    state.available_cameras = crate::tracking::list_cameras();
                    if !state
                        .available_cameras
                        .iter()
                        .any(|c| c.index == state.camera_index)
                    {
                        state.camera_index = state
                            .available_cameras
                            .first()
                            .map(|c| c.index)
                            .unwrap_or(0);
                    }
                }
            });
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let active = state.is_tracking_active();
                let ready = state.is_tracking_ready();
                if active && ready {
                    if ui.button("Stop Camera").clicked() {
                        state.app.stop_tracking();
                    }
                } else if active {
                    // Camera is initialising — show a disabled placeholder.
                    ui.add_enabled(false, egui::Button::new("Preparing..."));
                } else if ui.button("Start Camera").clicked() {
                    let (w, h) =
                        crate::gui::camera_resolution_for_index(state.tracking.camera_resolution_index);
                    let fps = crate::gui::camera_fps_for_index(state.tracking.camera_framerate_index);
                    #[cfg(feature = "webcam")]
                    let backend = crate::tracking::CameraBackend::Webcam {
                        camera_index: state.camera_index,
                    };
                    #[cfg(not(feature = "webcam"))]
                    let backend = crate::tracking::CameraBackend::Synthetic;
                    state.app.start_tracking_with_params(backend, w, h, fps);
                }
            });
            if ui
                .checkbox(&mut state.show_camera_wipe, "Camera Wipe (PIP)")
                .changed()
            {
                state.project_dirty = true;
            }
            if state.show_camera_wipe {
                if ui
                    .checkbox(&mut state.show_detection_annotations, "Show Annotations")
                    .changed()
                {
                    state.project_dirty = true;
                }
            }
            let prev_res = state.tracking.camera_resolution_index;
            egui::ComboBox::from_label("Resolution")
                .selected_text(
                    *["640x480", "1280x720", "1920x1080"]
                        .get(state.tracking.camera_resolution_index)
                        .unwrap_or(&"Unknown"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.tracking.camera_resolution_index, 0, "640x480");
                    ui.selectable_value(&mut state.tracking.camera_resolution_index, 1, "1280x720");
                    ui.selectable_value(
                        &mut state.tracking.camera_resolution_index,
                        2,
                        "1920x1080",
                    );
                });
            if state.tracking.camera_resolution_index != prev_res {
                state.project_dirty = true;
            }
            let prev_fps = state.tracking.camera_framerate_index;
            egui::ComboBox::from_label("Frame Rate")
                .selected_text(
                    *["30 fps", "60 fps"]
                        .get(state.tracking.camera_framerate_index)
                        .unwrap_or(&"Unknown"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.tracking.camera_framerate_index, 0, "30 fps");
                    ui.selectable_value(&mut state.tracking.camera_framerate_index, 1, "60 fps");
                });
            if state.tracking.camera_framerate_index != prev_fps {
                state.project_dirty = true;
            }
        });

    egui::CollapsingHeader::new("Capture Format")
        .default_open(false)
        .show(ui, |ui| {
            if ui
                .checkbox(&mut state.tracking.tracking_mirror, "Mirror Preview")
                .changed()
            {
                state.project_dirty = true;
            }
        });

    egui::CollapsingHeader::new("Inference Status")
        .default_open(true)
        .show(ui, |ui| {
            let camera_running = state.is_tracking_active();
            let camera_ready = state.is_tracking_ready();
            let tracking_on = state.tracking.toggle_tracking;
            let (status_color, status_text) = if camera_running && !camera_ready {
                (egui::Color32::YELLOW, "Preparing camera...")
            } else if camera_running && tracking_on {
                (egui::Color32::GREEN, "Running")
            } else if camera_running {
                (egui::Color32::YELLOW, "Camera active (tracking paused)")
            } else {
                (egui::Color32::GRAY, "Stopped")
            };
            ui.label(egui::RichText::new(status_text).color(status_color));
            if camera_running {
                let backend_label = state
                    .app
                    .tracking_worker
                    .as_ref()
                    .map(|w| w.active_backend().label())
                    .unwrap_or("Unknown");
                ui.label(format!("Backend: {}", backend_label));
            }

            if let Some(tracking) = &state.app.last_tracking_pose {
                ui.label(format!("Timestamp: {}", tracking.source_timestamp));
            } else {
                ui.label("No tracking data");
            }
        });

    egui::CollapsingHeader::new("Retargeting")
        .default_open(true)
        .show(ui, |ui| {
            if ui
                .checkbox(&mut state.tracking.hand_tracking_enabled, "Hand Tracking")
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .checkbox(&mut state.tracking.face_tracking_enabled, "Face Tracking")
                .changed()
            {
                state.project_dirty = true;
            }
            if let Some(tracking) = &state.app.last_tracking_pose {
                ui.separator();
                ui.label("Confidence");
                draw_confidence_bar(ui, "Overall", tracking.overall_confidence);
                if let Some(face) = tracking.face {
                    draw_confidence_bar(ui, "Face", face.confidence);
                }
                use crate::asset::HumanoidBone;
                let joint_conf = |b: HumanoidBone| -> f32 {
                    tracking.joints.get(&b).map(|j| j.confidence).unwrap_or(0.0)
                };
                draw_confidence_bar(
                    ui,
                    "Left Shoulder",
                    joint_conf(HumanoidBone::LeftShoulder),
                );
                draw_confidence_bar(
                    ui,
                    "Right Shoulder",
                    joint_conf(HumanoidBone::RightShoulder),
                );
                draw_confidence_bar(ui, "Left Hand", joint_conf(HumanoidBone::LeftHand));
                draw_confidence_bar(ui, "Right Hand", joint_conf(HumanoidBone::RightHand));
            }
        });

    egui::CollapsingHeader::new("Smoothing and Confidence")
        .default_open(false)
        .show(ui, |ui| {
            if ui
                .add(
                    egui::Slider::new(&mut state.tracking.smoothing_strength, 0.0..=1.0)
                        .text("Smoothing"),
                )
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .add(
                    egui::Slider::new(&mut state.tracking.confidence_threshold, 0.0..=1.0)
                        .text("Threshold"),
                )
                .changed()
            {
                state.project_dirty = true;
            }
        });

    egui::CollapsingHeader::new("Lip Sync")
        .default_open(false)
        .show(ui, |ui| {
            draw_lipsync(ui, state);
        });
}

fn draw_lipsync(ui: &mut egui::Ui, state: &mut GuiApp) {
    // Read App-derived state once for this frame. The combo / checkbox bind
    // to local copies; on change we route through Application mutators.
    let active_mic = state.app.lipsync_mic_device_index();
    let active_enabled = state.app.is_lipsync_enabled();

    let mut new_mic = active_mic;
    let mut refresh_clicked = false;

    ui.horizontal(|ui| {
        let selected_mic = state
            .lipsync
            .available_mics
            .iter()
            .find(|m| m.index == active_mic)
            .map(|m| m.name.clone())
            .unwrap_or_else(|| format!("Device {}", active_mic));
        egui::ComboBox::from_label("Microphone")
            .selected_text(selected_mic)
            .show_ui(ui, |ui| {
                if state.lipsync.available_mics.is_empty() {
                    ui.label("No devices found. Click Refresh to scan.");
                } else {
                    for mic in &state.lipsync.available_mics {
                        ui.selectable_value(&mut new_mic, mic.index, &mic.name);
                    }
                }
            });
        if ui.button("Refresh").clicked() {
            refresh_clicked = true;
        }
    });

    if refresh_clicked {
        state.lipsync.available_mics = crate::lipsync::audio_capture::list_audio_devices();
    }

    if new_mic != active_mic {
        // Always store the mic preference; restart only if currently enabled.
        if let Err(e) = state.app.set_lipsync_enabled(active_enabled, new_mic) {
            state.push_notification(format!("Lip sync mic switch failed: {e}"));
        }
        state.project_dirty = true;
    }

    ui.add_space(4.0);

    let mut new_enabled = active_enabled;
    if ui
        .checkbox(&mut new_enabled, "Enable Lip Sync")
        .changed()
    {
        match state.app.set_lipsync_enabled(new_enabled, active_mic) {
            Ok(()) => {
                state.project_dirty = true;
                if new_enabled {
                    state.push_notification("Lip sync started.".to_string());
                } else {
                    state.push_notification("Lip sync stopped.".to_string());
                }
            }
            Err(e) => {
                state.push_notification(format!("Lip sync failed: {e}"));
                // App stays disabled; checkbox will reflect that next frame.
            }
        }
    }

    // Per-frame lipsync inference is owned by `Application::step_lipsync`
    // and called from `GuiApp::update` so it keeps running even when this
    // panel is collapsed. The volume meter value (`state.lipsync.current_volume`)
    // is also written by `GuiApp::update` from `step_lipsync`'s return value.

    #[cfg(not(feature = "lipsync"))]
    {
        if active_enabled {
            // Without the feature, set_lipsync_enabled is a no-op stub that
            // returns Ok, so just call it to reset App state.
            let _ = state.app.set_lipsync_enabled(false, active_mic);
            state.push_notification(
                "Lip sync unavailable: build with --features lipsync".to_string(),
            );
        }
    }

    // Volume meter.
    let vol = state.lipsync.current_volume;
    let meter_rect = ui.available_rect_before_wrap();
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(meter_rect.width().min(200.0), 14.0),
        egui::Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(30, 30, 35));
    let fill_w = (vol * 10.0).clamp(0.0, 1.0) * rect.width();
    let fill_color = if vol > state.lipsync.volume_threshold {
        egui::Color32::from_rgb(80, 200, 80)
    } else {
        egui::Color32::from_rgb(60, 60, 70)
    };
    painter.rect_filled(
        egui::Rect::from_min_size(rect.min, egui::vec2(fill_w, rect.height())),
        2.0,
        fill_color,
    );
    ui.label(format!("Volume: {:.3}", vol));

    ui.add_space(4.0);
    if ui
        .add(
            egui::Slider::new(&mut state.lipsync.volume_threshold, 0.001..=0.1)
                .text("Threshold")
                .logarithmic(true),
        )
        .changed()
    {
        state.project_dirty = true;
    }
    if ui
        .add(egui::Slider::new(&mut state.lipsync.smoothing, 0.0..=1.0).text("Smoothing"))
        .changed()
    {
        state.project_dirty = true;
    }
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
            if ui
                .radio_value(&mut state.rendering.material_mode_index, 0, "Unlit")
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .radio_value(&mut state.rendering.material_mode_index, 1, "Simple Lit")
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .radio_value(&mut state.rendering.material_mode_index, 2, "Toon-like")
                .changed()
            {
                state.project_dirty = true;
            }
        });

    egui::CollapsingHeader::new("Toon Controls")
        .default_open(true)
        .show(ui, |ui| {
            if ui
                .add_enabled(
                    state.rendering.material_mode_index == 2,
                    egui::Slider::new(&mut state.rendering.toon_ramp_threshold, 0.0..=1.0)
                        .text("Ramp Threshold"),
                )
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .add_enabled(
                    state.rendering.material_mode_index == 2,
                    egui::Slider::new(&mut state.rendering.shadow_softness, 0.0..=1.0)
                        .text("Shadow Softness"),
                )
                .changed()
            {
                state.project_dirty = true;
            }
        });

    egui::CollapsingHeader::new("Outline Controls")
        .default_open(false)
        .show(ui, |ui| {
            if ui
                .checkbox(&mut state.rendering.outline_enabled, "Enable Outline")
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .add_enabled(
                    state.rendering.outline_enabled,
                    egui::Slider::new(&mut state.rendering.outline_width, 0.0..=0.1).text("Width"),
                )
                .changed()
            {
                state.project_dirty = true;
            }
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
                        .text("Intensity"),
                )
                .changed()
            {
                state.project_dirty = true;
            }
            ui.horizontal(|ui| {
                ui.label("Ambient");
                ui.color_edit_button_rgb(&mut state.rendering.ambient_intensity);
            });
        });

    egui::CollapsingHeader::new("Composition")
        .default_open(false)
        .show(ui, |ui| {
            if ui
                .add(
                    egui::Slider::new(&mut state.rendering.camera_fov, 10.0..=120.0)
                        .text("Camera FOV"),
                )
                .changed()
            {
                state.project_dirty = true;
            }
            ui.checkbox(&mut state.rendering.alpha_preview, "Alpha Preview");
        });

    egui::CollapsingHeader::new("Physics")
        .default_open(false)
        .show(ui, |ui| {
            let rapier_active = state.app.physics.rapier_initialized();
            if rapier_active {
                ui.label(egui::RichText::new("Rapier: Active").color(egui::Color32::GREEN));
            } else {
                ui.label(egui::RichText::new("Rapier: Not initialized").color(egui::Color32::GRAY));
            }
            if let Some(pos) = state.app.physics.character_position() {
                ui.label(format!(
                    "Character: ({:.2}, {:.2}, {:.2})",
                    pos[0], pos[1], pos[2]
                ));
            }
        });

    draw_scene_presets(ui, state);

    draw_expression_control(ui, state);
}

fn draw_output(ui: &mut egui::Ui, state: &mut GuiApp) {
    let sink_names = [
        "Virtual Camera",
        "Shared Texture",
        "Shared Memory",
        "Image Sequence",
    ];

    // After Phase C the active sink lives on Application. Bind the combo
    // to a local copy of the App-derived index, then on change push the
    // new selection through the mutator. No GuiState shadow.
    use crate::output::FrameSink;
    let active_idx = state.app.output.active_sink().to_gui_index();
    let mut new_idx = active_idx;

    egui::CollapsingHeader::new("Sink Selection")
        .default_open(true)
        .show(ui, |ui| {
            egui::ComboBox::from_label("Output Sink")
                .selected_text(*sink_names.get(active_idx).unwrap_or(&"Unknown"))
                .show_ui(ui, |ui| {
                    for (i, name) in sink_names.iter().enumerate() {
                        ui.selectable_value(&mut new_idx, i, *name);
                    }
                });
        });

    if new_idx != active_idx {
        let want_sink = FrameSink::from_gui_index(new_idx);
        match state.app.ensure_output_sink_runtime(want_sink) {
            Ok(()) => {
                state.push_notification(format!("Output sink changed to {}", sink_names[new_idx]));
                state.project_dirty = true;
            }
            Err(e) => {
                state.push_notification(format!("Output sink change failed: {e}"));
                // App stayed at the previous sink; the combo will display
                // that on the next frame because it reads from App.
            }
        }
    }

    let prev_res = state.output.output_resolution_index;
    let prev_fps = state.output.output_framerate_index;
    let prev_alpha = state.output.output_has_alpha;
    let prev_cs = state.output.output_color_space_index;

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
                    ui.selectable_value(&mut state.output.output_resolution_index, 0, "1920x1080");
                    ui.selectable_value(&mut state.output.output_resolution_index, 1, "1280x720");
                    ui.selectable_value(&mut state.output.output_resolution_index, 2, "640x480");
                });
            egui::ComboBox::from_label("Frame Rate")
                .selected_text(
                    *["60 fps", "30 fps"]
                        .get(state.output.output_framerate_index)
                        .unwrap_or(&"Unknown"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.output.output_framerate_index, 0, "60 fps");
                    ui.selectable_value(&mut state.output.output_framerate_index, 1, "30 fps");
                });
            ui.checkbox(&mut state.output.output_has_alpha, "RGBA (with alpha)");
            egui::ComboBox::from_label("Color Space")
                .selected_text(
                    *["sRGB", "Linear sRGB"]
                        .get(state.output.output_color_space_index)
                        .unwrap_or(&"Unknown"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.output.output_color_space_index, 0, "sRGB");
                    ui.selectable_value(
                        &mut state.output.output_color_space_index,
                        1,
                        "Linear sRGB",
                    );
                });
        });

    if state.output.output_resolution_index != prev_res
        || state.output.output_framerate_index != prev_fps
        || state.output.output_has_alpha != prev_alpha
        || state.output.output_color_space_index != prev_cs
    {
        state.project_dirty = true;
    }

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
                    let asset = state.app.active_avatar().map(|a| a.asset.clone());
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

            let mut autosave_enabled = state.cloth_autosave_consent.unwrap_or(false);
            if ui
                .checkbox(&mut autosave_enabled, "Auto-save cloth overlay")
                .changed()
            {
                state.cloth_autosave_consent = Some(autosave_enabled);
            }
        });

    egui::CollapsingHeader::new("Simulation Mesh")
        .default_open(false)
        .show(ui, |ui| {
            if let Some(avatar) = state.app.active_avatar() {
                ui.label("Select garment region to author");

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
                    if ui.button("Select by Material").clicked() {
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
                    if ui.button("Grow Selection").clicked() {
                        if let Some(ref mut sel) = state.region_selection {
                            sel.grow_selection(&avatar.asset);
                        }
                    }
                    if ui.button("Shrink Selection").clicked() {
                        if let Some(ref mut sel) = state.region_selection {
                            sel.shrink_selection(&avatar.asset);
                        }
                    }
                    if ui.button("Clear Selection").clicked() {
                        state.region_selection = None;
                    }
                });

                ui.add_space(4.0);

                if let Some(ref sel) = state.region_selection {
                    ui.label(
                        egui::RichText::new(format!(
                            "Selected: {} vertices",
                            sel.selected_vertices.len()
                        ))
                        .color(egui::Color32::from_rgb(255, 220, 80)),
                    );
                    ui.label(format!("Primitive: {}", sel.target_primitive.0));
                } else {
                    ui.label("No region selected");
                }

                ui.add_space(4.0);

                if ui.button("Generate Sim Mesh").clicked() {
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
                                state.push_notification("Sim mesh generated".to_string());
                            }
                        }
                    }
                }
            } else {
                ui.label("No avatar loaded");
            }
        });

    egui::CollapsingHeader::new("Pin Binding")
        .default_open(false)
        .show(ui, |ui| {
            if let Some(avatar) = state.app.active_avatar() {
                let nodes = &avatar.asset.skeleton.nodes;
                if nodes.is_empty() {
                    ui.label("No skeleton nodes");
                } else {
                    if state.cloth_pin_node_index >= nodes.len() {
                        state.cloth_pin_node_index = 0;
                    }
                    let node_names: Vec<String> = nodes
                        .iter()
                        .enumerate()
                        .map(|(_i, n)| format!("{} (id {})", n.name, n.id.0))
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
                        if ui.button("Create Pin Set").clicked() {
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
                                            state.push_notification("Pin set created".to_string());
                                        } else {
                                            state.push_notification(
                                                "No pinned vertices in sim mesh".to_string(),
                                            );
                                        }
                                    }
                                }
                            } else {
                                state.push_notification("Select a region first".to_string());
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
                ui.label("No avatar loaded");
            }
        });

    egui::CollapsingHeader::new("Constraints")
        .default_open(false)
        .show(ui, |ui| {
            ui.add(
                egui::Slider::new(&mut state.cloth_distance_stiffness, 0.0..=1.0)
                    .text("Distance Stiffness"),
            );
            ui.checkbox(&mut state.cloth_bend_enabled, "Bend Constraints");
            if state.cloth_bend_enabled {
                ui.add(
                    egui::Slider::new(&mut state.cloth_bend_stiffness, 0.0..=1.0)
                        .text("Bend Stiffness"),
                );
            }

            if let Some(ref mut overlay) = state.app.editor.overlay_asset {
                ui.add_space(4.0);
                ui.label(format!(
                    "Distance constraints: {}",
                    overlay.constraints.distance_constraints.len()
                ));
                ui.label(format!(
                    "Bend constraints: {}",
                    overlay.constraints.bend_constraints.len()
                ));
                if ui.button("Apply Constraints").clicked() {
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
                    state.push_notification("Constraints applied".to_string());
                }
            } else {
                ui.label("No overlay active");
            }
        });

    egui::CollapsingHeader::new("Collision Proxies")
        .default_open(false)
        .show(ui, |ui| {
            if let Some(avatar) = state.app.active_avatar() {
                let collider_count = avatar.asset.colliders.len();
                if state.cloth_collider_toggles.len() != collider_count {
                    state.cloth_collider_toggles = vec![true; collider_count];
                }
                if collider_count == 0 {
                    ui.label("No colliders in avatar");
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
                ui.label("No avatar loaded");
            }
        });

    egui::CollapsingHeader::new("Viewport")
        .default_open(false)
        .show(ui, |ui| {
            super::viewport_overlay::draw_viewport_controls(ui, &mut state.viewport_overlay);
        });

    egui::CollapsingHeader::new("Solver Parameters")
        .default_open(false)
        .show(ui, |ui| {
            if let Some(avatar) = state.app.active_avatar_mut() {
                if let Some(ref mut sim) = avatar.cloth_sim {
                    ui.horizontal(|ui| {
                        ui.label("Iterations:");
                        ui.add(egui::DragValue::new(&mut sim.solver_iterations).range(1..=32));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Gravity:");
                        ui.add(
                            egui::DragValue::new(&mut sim.gravity[1])
                                .speed(0.1)
                                .range(-20.0..=0.0),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Damping:");
                        ui.add(
                            egui::DragValue::new(&mut sim.damping)
                                .speed(0.01)
                                .range(0.0..=1.0),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Collision Margin:");
                        ui.add(
                            egui::DragValue::new(&mut sim.collision_margin)
                                .speed(0.001)
                                .range(0.0..=0.1),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Wind Response:");
                        ui.add(
                            egui::DragValue::new(&mut sim.wind_response)
                                .speed(0.1)
                                .range(0.0..=10.0),
                        );
                    });
                    ui.checkbox(&mut sim.self_collision, "Self-Collision");
                    if sim.self_collision {
                        ui.horizontal(|ui| {
                            ui.label("Self-Collision Radius:");
                            ui.add(
                                egui::DragValue::new(&mut sim.self_collision_radius)
                                    .speed(0.001)
                                    .range(0.001..=0.1),
                            );
                        });
                    }
                    ui.add_space(4.0);
                    ui.label(format!("Particles: {}", sim.particle_count()));
                    ui.label(format!(
                        "Distance constraints: {}",
                        sim.distance_constraints.len()
                    ));
                    ui.label(format!("Bend constraints: {}", sim.bend_constraints.len()));
                    ui.label(format!("Pin targets: {}", sim.pin_targets.len()));
                } else {
                    ui.label("No primary cloth simulation active");
                }

                if let Some(avatar) = state.app.active_avatar() {
                    if !avatar.cloth_overlays.is_empty() {
                        ui.add_space(4.0);
                        ui.label("Overlay Slots:");
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
                ui.label("No avatar loaded");
            }
        });

    egui::CollapsingHeader::new("Preview")
        .default_open(false)
        .show(ui, |ui| {
            if let Some(avatar) = state.app.active_avatar_mut() {
                ui.checkbox(&mut avatar.cloth_enabled, "Cloth Simulation");

                ui.horizontal(|ui| {
                    if state.cloth_sim_playing {
                        if ui.button("Pause").clicked() {
                            state.cloth_sim_playing = false;
                        }
                    } else if ui.button("Play").clicked() {
                        state.cloth_sim_playing = true;
                    }
                    if ui.button("Step").clicked() {
                        crate::simulation::cloth_solver::step_cloth(1.0 / 60.0, avatar, &[]);
                    }
                    if ui.button("Reset").clicked() {
                        avatar.cloth_sim = None;
                        avatar.cloth_sim_buffers = None;
                    }
                });

                ui.add_space(4.0);

                if let Some(ref cs) = avatar.cloth_state {
                    ui.label(format!("Primary sim positions: {}", cs.sim_positions.len()));
                    ui.label(format!(
                        "Primary deform version: {}",
                        cs.deform_output.version
                    ));
                } else if !avatar.cloth_overlays.is_empty() {
                    ui.label("Primary cloth state: None (overlays only)");
                } else {
                    ui.label("No cloth state");
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
                ui.label("No avatar loaded");
            }
        });
}

fn draw_model_library(ui: &mut egui::Ui, state: &mut GuiApp) {
    egui::CollapsingHeader::new("Model Library")
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut state.library_search_query);
                if state.library_search_query.is_empty() {
                    ui.label("Search");
                }
            });

            ui.horizontal(|ui| {
                if ui.button("Sort: Name").clicked() {
                    state.app.avatar_library.sort_by_name();
                }
                if ui.button("Sort: Recent").clicked() {
                    state.app.avatar_library.sort_by_last_loaded();
                }
                if ui.button("Favorites First").clicked() {
                    state.app.avatar_library.sort_favorites_first();
                }
            });

            ui.checkbox(&mut state.library_show_missing, "Show missing files");

            ui.separator();

            let current_avatar_path = state
                .app
                .active_avatar()
                .map(|a| a.asset.source_path.clone());

            let entries: Vec<(
                usize,
                String,
                std::path::PathBuf,
                bool,
                bool,
                Option<usize>,
                Option<usize>,
            )> = {
                let lib = &state.app.avatar_library;
                if state.library_search_query.is_empty() {
                    lib.entries
                        .iter()
                        .enumerate()
                        .filter(|(_, e)| state.library_show_missing || e.exists())
                        .map(|(i, e)| {
                            (
                                i,
                                e.name.clone(),
                                e.path.clone(),
                                e.favorite,
                                e.exists(),
                                e.mesh_count,
                                e.material_count,
                            )
                        })
                        .collect()
                } else {
                    lib.search(&state.library_search_query)
                        .into_iter()
                        .filter(|e| state.library_show_missing || e.exists())
                        .filter_map(|e| {
                            let idx = lib.entries.iter().position(|x| x.path == e.path)?;
                            Some((
                                idx,
                                e.name.clone(),
                                e.path.clone(),
                                e.favorite,
                                e.exists(),
                                e.mesh_count,
                                e.material_count,
                            ))
                        })
                        .collect()
                }
            };

            let total = entries.len();
            let available = entries
                .iter()
                .filter(|(_, _, _, _, exists, _, _)| *exists)
                .count();
            ui.label(format!("{} models ({} available)", total, available));

            let mut load_path: Option<std::path::PathBuf> = None;
            let mut remove_idx: Option<usize> = None;
            let mut toggle_fav_idx: Option<usize> = None;

            egui::ScrollArea::vertical()
                .max_height(300.0)
                .show(ui, |ui| {
                    for (idx, name, path, favorite, exists, mesh_count, mat_count) in &entries {
                        let is_current =
                            current_avatar_path.as_ref().map_or(false, |p| *p == *path);

                        let frame = if is_current {
                            egui::Frame::group(ui.style())
                                .fill(egui::Color32::from_rgba_unmultiplied(60, 80, 120, 200))
                                .stroke((1.0, egui::Color32::from_rgb(100, 150, 255)))
                        } else if !*exists {
                            egui::Frame::group(ui.style())
                                .fill(egui::Color32::from_rgba_unmultiplied(80, 40, 40, 200))
                                .stroke((1.0, egui::Color32::from_rgb(180, 60, 60)))
                        } else {
                            egui::Frame::group(ui.style())
                        };

                        let response = frame.show(ui, |ui| {
                            ui.horizontal(|ui| {
                                let star = if *favorite { "\u{2605}" } else { "\u{2606}" };
                                if ui.button(egui::RichText::new(star).size(14.0)).clicked() {
                                    toggle_fav_idx = Some(*idx);
                                }

                                ui.vertical(|ui| {
                                    ui.set_min_width(ui.available_width() - 60.0);
                                    let name_color = if is_current {
                                        egui::Color32::from_rgb(140, 190, 255)
                                    } else if !*exists {
                                        egui::Color32::from_rgb(180, 100, 100)
                                    } else {
                                        egui::Color32::WHITE
                                    };
                                    ui.label(egui::RichText::new(name).color(name_color));

                                    let mut info = String::new();
                                    if let Some(mc) = mesh_count {
                                        info.push_str(&format!("{} meshes", mc));
                                    }
                                    if let Some(mc) = mat_count {
                                        if !info.is_empty() {
                                            info.push_str(", ");
                                        }
                                        info.push_str(&format!("{} materials", mc));
                                    }
                                    if !info.is_empty() {
                                        ui.label(
                                            egui::RichText::new(info)
                                                .small()
                                                .color(egui::Color32::GRAY),
                                        );
                                    }

                                    ui.label(
                                        egui::RichText::new(path.to_string_lossy().as_ref())
                                            .small()
                                            .color(egui::Color32::DARK_GRAY),
                                    );
                                });

                                ui.vertical(|ui| {
                                    if *exists {
                                        if ui.button("Load").clicked() {
                                            load_path = Some(path.clone());
                                        }
                                    } else {
                                        ui.label(
                                            egui::RichText::new("MISSING")
                                                .small()
                                                .color(egui::Color32::RED),
                                        );
                                    }
                                    if ui.button("Remove").clicked() {
                                        remove_idx = Some(*idx);
                                    }
                                });
                            })
                        });

                        if response.response.clicked() {
                            state.library_selected_index = Some(*idx);
                        }
                    }
                });

            if let Some(path) = load_path {
                if path.exists() {
                    super::top_bar::load_avatar_from_path(state, &path);
                }
            }

            if let Some(idx) = remove_idx {
                if idx < state.app.avatar_library.entries.len() {
                    let path = state.app.avatar_library.entries[idx].path.clone();
                    state.app.avatar_library.remove(&path);
                    let _ = crate::persistence::save_avatar_library(&state.app.avatar_library);
                    state.push_notification("Removed from library".to_string());
                }
            }

            if let Some(idx) = toggle_fav_idx {
                if let Some(entry) = state.app.avatar_library.entries.get_mut(idx) {
                    entry.favorite = !entry.favorite;
                    let _ = crate::persistence::save_avatar_library(&state.app.avatar_library);
                }
            }

            ui.separator();

            let mut details_save_requested = false;
            let mut tag_to_add: Option<String> = None;
            let mut tag_to_remove: Option<usize> = None;

            if let Some(idx) = state.library_selected_index {
                if idx < state.app.avatar_library.entries.len() {
                    ui.label(egui::RichText::new("Details").strong());

                    let entry = &mut state.app.avatar_library.entries[idx];
                    ui.horizontal(|ui| {
                        ui.label("Name:");
                        ui.text_edit_singleline(&mut entry.name);
                    });
                    ui.horizontal(|ui| {
                        ui.label("Tags:");
                        ui.text_edit_singleline(&mut state.library_tag_buf);
                        if ui.button("+").clicked() && !state.library_tag_buf.is_empty() {
                            tag_to_add = Some(state.library_tag_buf.clone());
                        }
                    });

                    let tags_snapshot: Vec<String> =
                        state.app.avatar_library.entries[idx].tags.clone();
                    if !tags_snapshot.is_empty() {
                        ui.horizontal_wrapped(|ui| {
                            for tag_i in (0..tags_snapshot.len()).rev() {
                                let tag_label = format!("{} \u{00d7}", tags_snapshot[tag_i]);
                                if ui.button(egui::RichText::new(tag_label).small()).clicked() {
                                    tag_to_remove = Some(tag_i);
                                }
                            }
                        });
                    }

                    let entry = &mut state.app.avatar_library.entries[idx];
                    ui.horizontal(|ui| {
                        ui.label("Notes:");
                        ui.text_edit_multiline(&mut entry.notes);
                    });
                    if ui.button("Save Details").clicked() {
                        details_save_requested = true;
                    }
                }
            }

            if let Some(tag) = tag_to_add {
                if let Some(idx) = state.library_selected_index {
                    if let Some(entry) = state.app.avatar_library.entries.get_mut(idx) {
                        entry.tags.push(tag.clone());
                        state.library_tag_buf.clear();
                        details_save_requested = true;
                    }
                }
            }
            if let Some(tag_i) = tag_to_remove {
                if let Some(idx) = state.library_selected_index {
                    if let Some(entry) = state.app.avatar_library.entries.get_mut(idx) {
                        entry.tags.remove(tag_i);
                        details_save_requested = true;
                    }
                }
            }
            if details_save_requested {
                let _ = crate::persistence::save_avatar_library(&state.app.avatar_library);
                state.push_notification("Library entry updated".to_string());
            }

            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Purge Missing").clicked() {
                    let removed = state.app.avatar_library.purge_missing();
                    let _ = crate::persistence::save_avatar_library(&state.app.avatar_library);
                    state.push_notification(format!("Purged {} missing entries", removed));
                }
                if ui.button("Add File...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("VRM 1.0", &["vrm"])
                        .pick_file()
                    {
                        let entry =
                            crate::app::avatar_library::AvatarLibraryEntry::from_path(&path);
                        state.app.avatar_library.add(entry);
                        let _ = crate::persistence::save_avatar_library(&state.app.avatar_library);
                        state.push_notification(format!("Added to library: {}", path.display()));
                    }
                }
            });
        });
}

fn draw_scene_presets(ui: &mut egui::Ui, state: &mut GuiApp) {
    egui::CollapsingHeader::new("Scene Presets")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Save Preset").clicked() {
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
                    state.push_notification(format!("Saved scene preset: {}", name));
                }
                if ui.button("Delete Selected").clicked() {
                    if let Some(idx) = state.scene_preset_index {
                        if idx < state.scene_presets.len() {
                            let name = state.scene_presets[idx].name.clone();
                            state.scene_presets.remove(idx);
                            state.scene_preset_index = None;
                            let _ = crate::persistence::save_scene_presets(&state.scene_presets);
                            state.push_notification(format!("Deleted preset: {}", name));
                        }
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label("Name:");
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
                    state.push_notification(format!("Loaded scene preset: {}", preset.name));
                    state.project_dirty = true;
                }
            }

            ui.label(format!("{} preset(s) saved", state.scene_presets.len()));
        });
}

fn draw_expression_control(ui: &mut egui::Ui, state: &mut GuiApp) {
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

    egui::CollapsingHeader::new("Expression Control")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Reset All").clicked() {
                    state.expression_weights.fill(0.0);
                }
                if ui.button("Set All 50%").clicked() {
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
            ui.label(format!("{} expression(s) loaded", expressions.len()));
        });
}
