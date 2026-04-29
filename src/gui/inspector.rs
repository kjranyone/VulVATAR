use eframe::egui;

use crate::gui::{AppMode, GuiApp};
use crate::t;

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
                            AppMode::Avatar => draw_avatar(ui, state),
                            AppMode::Preview => draw_preview(ui, state),
                            AppMode::TrackingSetup => draw_tracking(ui, state),
                            AppMode::Rendering => draw_rendering(ui, state),
                            AppMode::Output => draw_output(ui, state),
                            AppMode::ClothAuthoring => draw_cloth_authoring(ui, state),
                            AppMode::Settings => draw_settings(ui, state),
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
                                    state.camera_orbit.pan = [0.0, 0.0];
                                    state.project_dirty = true;
                                }
                            });
                    });
            });
        });
}

fn draw_avatar(ui: &mut egui::Ui, state: &mut GuiApp) {
    egui::CollapsingHeader::new(t!("inspector.current_avatar"))
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
                    ui.label(t!("inspector.spec"));
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
                        egui::RichText::new(t!("inspector.vrm0x_note"))
                        .small()
                        .color(egui::Color32::from_rgb(220, 180, 100)),
                    );
                }

                let title = meta.title.as_deref().unwrap_or("—");
                ui.horizontal(|ui| {
                    ui.label(t!("inspector.title"));
                    ui.label(title);
                });
                let authors = if meta.authors.is_empty() {
                    "—".to_string()
                } else {
                    meta.authors.join(", ")
                };
                ui.horizontal(|ui| {
                    ui.label(t!("inspector.authors"));
                    ui.label(authors);
                });
                if let Some(v) = &meta.model_version {
                    ui.horizontal(|ui| {
                        ui.label(t!("inspector.model_version"));
                        ui.label(v);
                    });
                }
                if let Some(c) = &meta.contact_information {
                    ui.horizontal(|ui| {
                        ui.label(t!("inspector.contact"));
                        ui.label(c);
                    });
                }
                if let Some(l) = &meta.license {
                    ui.horizontal(|ui| {
                        ui.label(t!("inspector.license"));
                        ui.label(l);
                    });
                }
                if let Some(cr) = &meta.copyright_information {
                    ui.horizontal(|ui| {
                        ui.label(t!("inspector.copyright"));
                        ui.label(cr);
                    });
                }
                if !meta.references.is_empty() {
                    ui.collapsing(t!("inspector.references"), |ui| {
                        for r in &meta.references {
                            ui.label(
                                egui::RichText::new(r).small().color(egui::Color32::GRAY),
                            );
                        }
                    });
                }

                ui.separator();
                ui.label(
                    egui::RichText::new(t!("inspector.source", path = asset.source_path.display().to_string()))
                        .small()
                        .color(egui::Color32::GRAY),
                );
                ui.horizontal(|ui| {
                    ui.label(t!("inspector.meshes", count = asset.meshes.len()));
                    ui.label(t!("inspector.materials", count = asset.materials.len()));
                });
                ui.horizontal(|ui| {
                    ui.label(t!("inspector.spring_chains", count = asset.spring_bones.len()));
                    ui.label(t!("inspector.colliders", count = asset.colliders.len()));
                });
                ui.horizontal(|ui| {
                    ui.label(
                        if asset.humanoid.is_some() { t!("inspector.humanoid_yes") } else { t!("inspector.humanoid_no") }
                    );
                    ui.label(t!("inspector.expressions", count = asset.default_expressions.expressions.len()));
                    ui.label(t!("inspector.animations", count = asset.animation_clips.len()));
                });

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui.button(t!("inspector.reload")).clicked() {
                        state.app.reload_avatar();
                    }
                    if ui.button(t!("inspector.detach")).clicked()
                        && !state.app.avatars.is_empty()
                    {
                        state.app.remove_avatar_at(state.app.active_avatar_index);
                    }
                });
            } else {
                ui.label(t!("inspector.no_avatar_loaded"));
                ui.label(
                    egui::RichText::new(t!("inspector.no_avatar_hint"))
                        .small()
                        .color(egui::Color32::GRAY),
                );
            }
        });

    draw_model_library(ui, state);
}

fn draw_preview(ui: &mut egui::Ui, state: &mut GuiApp) {
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
                if ui.button(t!("inspector.reload")).clicked() {
                    state.app.reload_avatar();
                }
                if ui.button(t!("inspector.detach")).clicked()
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
            if ui.button(t!("inspector.reset_transform")).clicked() {
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
                if ui.button(t!("inspector.attach_primary")).clicked() {
                    let overlay_id = state
                        .app
                        .editor
                        .active_overlay()
                        .unwrap_or(crate::asset::ClothOverlayId(1));
                    if let Some(avatar) = state.app.active_avatar_mut() {
                        avatar.attach_cloth(overlay_id);
                    }
                }
                if ui.button(t!("inspector.detach_primary")).clicked() {
                    if let Some(avatar) = state.app.active_avatar_mut() {
                        avatar.detach_cloth();
                    }
                }
            });
            ui.horizontal(|ui| {
                if ui.button(t!("inspector.add_overlay_slot")).clicked() {
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
                if ui.button(t!("inspector.load_overlay_file")).clicked() {
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
                                    if ui.button(t!("inspector.remove")).clicked() {
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
            ui.checkbox(
                &mut state.rendering.toggle_collision_debug,
                t!("inspector.collision_debug"),
            );
            ui.checkbox(&mut state.rendering.toggle_skeleton_debug, t!("inspector.skeleton_debug"));
        });

    draw_expression_control(ui, state);
}

fn draw_tracking(ui: &mut egui::Ui, state: &mut GuiApp) {
    if ui
        .checkbox(&mut state.tracking.toggle_tracking, t!("tracking.enabled"))
        .changed()
    {
        state.project_dirty = true;
    }
    ui.add_space(4.0);

    egui::CollapsingHeader::new(t!("tracking.input_device"))
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let prev_camera = state.camera_index;
                let selected_text = state
                    .available_cameras
                    .iter()
                    .find(|c| c.index == state.camera_index)
                    .map(|c| format!("{}: {}", c.index, c.name))
                    .unwrap_or_else(|| t!("tracking.camera_n", index = state.camera_index).to_string());
                egui::ComboBox::from_label(t!("tracking.camera"))
                    .selected_text(selected_text)
                    .show_ui(ui, |ui| {
                        if state.available_cameras.is_empty() {
                            ui.label(t!("tracking.no_cameras"));
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
                    if ui.button(t!("tracking.stop_camera")).clicked() {
                        state.app.stop_tracking();
                    }
                } else if active {
                    // Camera is initialising — show a disabled placeholder.
                    ui.add_enabled(false, egui::Button::new(t!("tracking.preparing")));
                } else if ui.button(t!("tracking.start_camera")).clicked() {
                    let (w, h) = crate::gui::camera_resolution_for_index(
                        state.tracking.camera_resolution_index,
                    );
                    let fps =
                        crate::gui::camera_fps_for_index(state.tracking.camera_framerate_index);
                    #[cfg(feature = "webcam")]
                    let backend = crate::tracking::CameraBackend::Webcam {
                        camera_index: state.camera_index,
                    };
                    #[cfg(not(feature = "webcam"))]
                    let backend = crate::tracking::CameraBackend::Synthetic;
                    state.app.start_tracking_with_params_full(
                        backend,
                        w,
                        h,
                        fps,
                        state.tracking.lower_body_tracking_enabled,
                    );
                }
            });
            if ui
                .checkbox(&mut state.show_camera_wipe, t!("tracking.camera_wipe"))
                .changed()
            {
                state.project_dirty = true;
            }
            if state.show_camera_wipe
                && ui
                    .checkbox(&mut state.show_detection_annotations, t!("tracking.show_annotations"))
                    .changed()
            {
                state.project_dirty = true;
            }
            let prev_res = state.tracking.camera_resolution_index;
            egui::ComboBox::from_label(t!("tracking.resolution"))
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
            let res_changed = state.tracking.camera_resolution_index != prev_res;
            if res_changed {
                state.project_dirty = true;
            }
            let prev_fps = state.tracking.camera_framerate_index;
            egui::ComboBox::from_label(t!("tracking.frame_rate"))
                .selected_text(
                    *["30 fps", "60 fps"]
                        .get(state.tracking.camera_framerate_index)
                        .unwrap_or(&"Unknown"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.tracking.camera_framerate_index, 0, "30 fps");
                    ui.selectable_value(&mut state.tracking.camera_framerate_index, 1, "60 fps");
                });
            let fps_changed = state.tracking.camera_framerate_index != prev_fps;
            if fps_changed {
                state.project_dirty = true;
            }
            // Phase B-1: if the webcam is currently running, restart it with
            // the new params so the change takes effect immediately rather
            // than only at the next Stop / Start Camera cycle.
            if (res_changed || fps_changed) && state.app.is_tracking_running() {
                let (w, h) =
                    crate::gui::camera_resolution_for_index(state.tracking.camera_resolution_index);
                let fps = crate::gui::camera_fps_for_index(state.tracking.camera_framerate_index);
                #[cfg(feature = "webcam")]
                let backend = crate::tracking::CameraBackend::Webcam {
                    camera_index: state.camera_index,
                };
                #[cfg(not(feature = "webcam"))]
                let backend = crate::tracking::CameraBackend::Synthetic;
                state.app.start_tracking_with_params_full(
                    backend,
                    w,
                    h,
                    fps,
                    state.tracking.lower_body_tracking_enabled,
                );
                state.push_notification(t!("tracking.camera_restarted", w = w, h = h, fps = fps));
            }
        });

    egui::CollapsingHeader::new(t!("tracking.capture_format"))
        .default_open(false)
        .show(ui, |ui| {
            if ui
                .checkbox(&mut state.tracking.tracking_mirror, t!("tracking.mirror_preview"))
                .changed()
            {
                state.project_dirty = true;
            }
        });

    egui::CollapsingHeader::new(t!("tracking.inference_status"))
        .default_open(true)
        .show(ui, |ui| {
            let camera_running = state.is_tracking_active();
            let camera_ready = state.is_tracking_ready();
            let tracking_on = state.tracking.toggle_tracking;
            let (status_color, status_text) = if camera_running && !camera_ready {
                (egui::Color32::YELLOW, t!("tracking.preparing_camera").to_string())
            } else if camera_running && tracking_on {
                (egui::Color32::GREEN, t!("tracking.running").to_string())
            } else if camera_running {
                (egui::Color32::YELLOW, t!("tracking.camera_active_paused").to_string())
            } else {
                (egui::Color32::GRAY, t!("tracking.stopped").to_string())
            };
            ui.label(egui::RichText::new(status_text).color(status_color));
            if camera_running {
                let camera_label = state
                    .app
                    .tracking_worker
                    .as_ref()
                    .map(|w| w.active_backend().label())
                    .unwrap_or("Unknown");
                ui.label(t!("tracking.camera_label", label = camera_label.to_string()));

                // Inference backend (ONNX execution provider) — surfaces a
                // silent CPU fallback when DirectML registration fails. Set
                // by the worker once CIGPose finishes loading; absent during
                // synthetic mode or before init completes.
                if let Some(label) = state
                    .app
                    .tracking_worker
                    .as_ref()
                    .and_then(|w| w.mailbox().inference_backend_label())
                {
                    ui.label(t!("tracking.inference_label", label = label.to_string()));
                }
            }

            if let Some(tracking) = &state.app.last_tracking_pose {
                ui.label(format!("Timestamp: {}", tracking.source_timestamp));
            } else {
                ui.label(t!("tracking.no_tracking_data"));
            }
        });

    egui::CollapsingHeader::new(t!("tracking.retargeting"))
        .default_open(true)
        .show(ui, |ui| {
            if ui
                .checkbox(&mut state.tracking.hand_tracking_enabled, t!("tracking.hand_tracking"))
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .checkbox(&mut state.tracking.face_tracking_enabled, t!("tracking.face_tracking"))
                .changed()
            {
                state.project_dirty = true;
            }
            let lower_body_resp = ui
                .checkbox(
                    &mut state.tracking.lower_body_tracking_enabled,
                    t!("tracking.lower_body"),
                )
                .on_hover_text(
                    t!("tracking.lower_body_tooltip"),
                );
            if lower_body_resp.changed() {
                state.project_dirty = true;
            }
            ui.separator();
            ui.horizontal(|ui| {
                if ui
                    .button(t!("tracking.recalibrate"))
                    .on_hover_text(t!("tracking.recalibrate_tooltip"))
                    .clicked()
                {
                    state.app.recalibrate_pose_solver();
                }
            });
            if let Some(tracking) = &state.app.last_tracking_pose {
                ui.separator();
                ui.label(t!("tracking.confidence"));
                draw_confidence_bar(ui, &t!("tracking.confidence_overall"), tracking.overall_confidence);
                if let Some(face) = tracking.face {
                    draw_confidence_bar(ui, &t!("tracking.confidence_face"), face.confidence);
                }
                use crate::asset::HumanoidBone;
                let joint_conf = |b: HumanoidBone| -> f32 {
                    tracking.joints.get(&b).map(|j| j.confidence).unwrap_or(0.0)
                };
                draw_confidence_bar(ui, &t!("tracking.confidence_left_shoulder"), joint_conf(HumanoidBone::LeftShoulder));
                draw_confidence_bar(
                    ui,
                    &t!("tracking.confidence_right_shoulder"),
                    joint_conf(HumanoidBone::RightShoulder),
                );
                draw_confidence_bar(ui, &t!("tracking.confidence_left_hand"), joint_conf(HumanoidBone::LeftHand));
                draw_confidence_bar(ui, &t!("tracking.confidence_right_hand"), joint_conf(HumanoidBone::RightHand));
            }
        });

    egui::CollapsingHeader::new(t!("tracking.smoothing_confidence"))
        .default_open(false)
        .show(ui, |ui| {
            if ui
                .add(
                    egui::Slider::new(&mut state.tracking.smoothing_strength, 0.0..=1.0)
                        .text(t!("tracking.smoothing")),
                )
                .changed()
            {
                state.project_dirty = true;
            }
            if ui
                .add(
                    egui::Slider::new(&mut state.tracking.confidence_threshold, 0.0..=1.0)
                        .text(t!("tracking.threshold")),
                )
                .changed()
            {
                state.project_dirty = true;
            }
        });

    egui::CollapsingHeader::new(t!("tracking.lip_sync"))
        .default_open(false)
        .show(ui, |ui| {
            draw_lipsync(ui, state);
        });
}

fn draw_lipsync(ui: &mut egui::Ui, state: &mut GuiApp) {
    // Phase D: bind UI to user-requested state, not runtime state. If the
    // audio device fails to open the runtime stays disabled, but the
    // checkbox keeps the user's intent so a mic refresh can retry.
    let active_mic = state.app.lipsync_mic_device_index();
    let requested_enabled = state.app.requested_lipsync_enabled();
    let runtime_enabled = state.app.is_lipsync_enabled();

    let mut new_mic = active_mic;
    let mut refresh_clicked = false;

    ui.horizontal(|ui| {
        let selected_mic = state
            .lipsync
            .available_mics
            .iter()
            .find(|m| m.index == active_mic)
            .map(|m| m.name.clone())
            .unwrap_or_else(|| t!("tracking.device_n", index = active_mic).to_string());
        egui::ComboBox::from_label(t!("tracking.microphone"))
            .selected_text(selected_mic)
            .show_ui(ui, |ui| {
                if state.lipsync.available_mics.is_empty() {
                    ui.label(t!("tracking.no_mics"));
                } else {
                    for mic in &state.lipsync.available_mics {
                        ui.selectable_value(&mut new_mic, mic.index, &mic.name);
                    }
                }
            });
        if ui.button(t!("tracking.refresh")).clicked() {
            refresh_clicked = true;
        }
    });

    if refresh_clicked {
        state.lipsync.available_mics = crate::lipsync::audio_capture::list_audio_devices();
    }

    if new_mic != active_mic {
        if let Err(e) = state.app.set_requested_lipsync(requested_enabled, new_mic) {
            state.push_notification(t!("tracking.lip_sync_mic_failed", error = e.to_string()));
        }
        state.project_dirty = true;
    }

    ui.add_space(4.0);

    let mut new_enabled = requested_enabled;
    if ui.checkbox(&mut new_enabled, t!("tracking.enable_lip_sync")).changed() {
        match state.app.set_requested_lipsync(new_enabled, active_mic) {
            Ok(()) => {
                state.project_dirty = true;
                if new_enabled {
                    state.push_notification(t!("tracking.lip_sync_started").to_string());
                } else {
                    state.push_notification(t!("tracking.lip_sync_stopped").to_string());
                }
            }
            Err(e) => {
                state.push_notification(t!("tracking.lip_sync_failed", error = e.to_string()));
                state.project_dirty = true;
            }
        }
    }

    if requested_enabled && !runtime_enabled {
        ui.colored_label(
            egui::Color32::from_rgb(220, 160, 80),
            t!("tracking.requested_but_failed"),
        );
    }

    // Per-frame lipsync inference is owned by `Application::step_lipsync`
    // and called from `GuiApp::update` so it keeps running even when this
    // panel is collapsed. The volume meter value (`state.lipsync.current_volume`)
    // is also written by `GuiApp::update` from `step_lipsync`'s return value.

    #[cfg(not(feature = "lipsync"))]
    {
        if requested_enabled {
            // Without the feature, set_requested_lipsync is a no-op stub
            // that returns Ok, so just call it to reset App state.
            let _ = state.app.set_requested_lipsync(false, active_mic);
            state.push_notification(
                t!("tracking.lip_sync_unavailable").to_string(),
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
    ui.label(t!("tracking.volume", vol = format!("{:.3}", vol)));

    ui.add_space(4.0);
    if ui
        .add(
            egui::Slider::new(&mut state.lipsync.volume_threshold, 0.001..=0.1)
                .text(t!("tracking.threshold"))
                .logarithmic(true),
        )
        .changed()
    {
        state.project_dirty = true;
    }
    if ui
        .add(egui::Slider::new(&mut state.lipsync.smoothing, 0.0..=1.0).text(t!("tracking.smoothing")))
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
            ui.checkbox(&mut state.rendering.alpha_preview, t!("inspector.alpha_preview"));
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

fn draw_output(ui: &mut egui::Ui, state: &mut GuiApp) {
    let sink_names: [String; 4] = [
        t!("status.sink_virtual_camera"),
        t!("status.sink_shared_texture"),
        t!("status.sink_shared_memory"),
        t!("status.sink_image_sequence"),
    ];

    // Combo binds to the user's REQUESTED sink (Phase D). If the runtime
    // can't honor it (MF init failed), the request still persists so
    // autosave doesn't overwrite the user's choice with a fallback.
    use crate::output::FrameSink;
    let requested_idx = state.app.requested_sink().to_gui_index();
    let active_idx = state.app.output.active_sink().to_gui_index();
    let mut new_idx = requested_idx;

    egui::CollapsingHeader::new(t!("inspector.output_sink"))
        .default_open(true)
        .show(ui, |ui| {
            egui::ComboBox::from_label(t!("inspector.output_sink_label"))
                .selected_text(sink_names.get(requested_idx).map(|s: &String| s.as_str()).unwrap_or("Unknown"))
                .show_ui(ui, |ui| {
                    for (i, name) in sink_names.iter().enumerate() {
                        ui.selectable_value(&mut new_idx, i, name);
                    }
                });
            if active_idx != requested_idx {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 160, 80),
                    t!("inspector.sink_mismatch",
                        active = sink_names.get(active_idx).map(|s: &String| s.as_str()).unwrap_or("?"),
                        requested = sink_names.get(requested_idx).map(|s: &String| s.as_str()).unwrap_or("?"),
                    ),
                );
            }
        });

    if new_idx != requested_idx {
        let want_sink = FrameSink::from_gui_index(new_idx);
        match state.app.set_requested_sink(want_sink) {
            Ok(()) => {
                state.push_notification(t!("inspector.output_sink_changed", name = sink_names[new_idx].to_string()));
                state.project_dirty = true;
            }
            Err(e) => {
                state.push_notification(t!("inspector.output_sink_failed", error = e.to_string()));
                state.project_dirty = true;
            }
        }
    }

    let prev_res = state.output.output_resolution_index;
    let prev_fps = state.output.output_framerate_index;
    let prev_alpha = state.output.output_has_alpha;
    let prev_cs = state.output.output_color_space_index;

    egui::CollapsingHeader::new(t!("inspector.frame_format"))
        .default_open(true)
        .show(ui, |ui| {
            egui::ComboBox::from_label(t!("tracking.resolution"))
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
            egui::ComboBox::from_label(t!("tracking.frame_rate"))
                .selected_text(
                    *["60 fps", "30 fps"]
                        .get(state.output.output_framerate_index)
                        .unwrap_or(&"Unknown"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.output.output_framerate_index, 0, "60 fps");
                    ui.selectable_value(&mut state.output.output_framerate_index, 1, "30 fps");
                });
            ui.checkbox(&mut state.output.output_has_alpha, t!("inspector.rgba_alpha"));
            let cs_names: [String; 2] = [t!("inspector.srgb"), t!("inspector.linear_srgb")];
            egui::ComboBox::from_label(t!("inspector.color_space"))
                .selected_text(
                    cs_names
                        .get(state.output.output_color_space_index)
                        .map(|s: &String| s.as_str())
                        .unwrap_or("Unknown"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.output.output_color_space_index, 0, &*cs_names[0]);
                    ui.selectable_value(
                        &mut state.output.output_color_space_index,
                        1,
                        &*cs_names[1],
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

    egui::CollapsingHeader::new(t!("inspector.synchronization"))
        .default_open(false)
        .show(ui, |ui| {
            ui.label(t!("inspector.handoff_gpu"));
            ui.label(t!("inspector.fallback_cpu"));
            ui.label(egui::RichText::new(t!("inspector.gpu_active")).color(egui::Color32::GREEN));
        });

    egui::CollapsingHeader::new(t!("inspector.diagnostics"))
        .default_open(true)
        .show(ui, |ui| {
            ui.label(egui::RichText::new(t!("inspector.connected")).color(egui::Color32::GREEN));
            ui.label(t!("inspector.queue_depth", depth = state.app.output.queue_depth()));
            ui.label(t!("inspector.dropped_frames", count = state.app.output.dropped_count()));
        });
}

fn draw_cloth_authoring(ui: &mut egui::Ui, state: &mut GuiApp) {
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
                if ui.button(t!("inspector.new_overlay")).clicked() {
                    let asset = state.app.active_avatar().map(|a| a.asset.clone());
                    if let Some(asset) = asset {
                        let avatar_id = asset.id;
                        let avatar_hash = asset.source_hash.clone();
                        state.app.editor.open(asset);
                        state.app.editor.create_overlay(avatar_id, avatar_hash);
                    }
                }
                if ui.button(t!("inspector.save_overlay")).clicked() {
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
                    if ui.button(t!("inspector.select_by_material")).clicked() {
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
                    if ui.button(t!("inspector.grow_selection")).clicked() {
                        if let Some(ref mut sel) = state.region_selection {
                            sel.grow_selection(&avatar.asset);
                        }
                    }
                    if ui.button(t!("inspector.shrink_selection")).clicked() {
                        if let Some(ref mut sel) = state.region_selection {
                            sel.shrink_selection(&avatar.asset);
                        }
                    }
                    if ui.button(t!("inspector.clear_selection")).clicked() {
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

                if ui.button(t!("inspector.generate_sim_mesh")).clicked() {
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
                        if ui.button(t!("inspector.create_pin_set")).clicked() {
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
                if ui.button(t!("inspector.apply_constraints")).clicked() {
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

    egui::CollapsingHeader::new(t!("inspector.viewport"))
        .default_open(false)
        .show(ui, |ui| {
            super::viewport_overlay::draw_viewport_controls(ui, &mut state.viewport_overlay);
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
                        if ui.button(t!("inspector.pause")).clicked() {
                            state.cloth_sim_playing = false;
                        }
                    } else if ui.button(t!("inspector.play")).clicked() {
                        state.cloth_sim_playing = true;
                    }
                    if ui.button(t!("inspector.step")).clicked() {
                        crate::simulation::cloth_solver::step_cloth(1.0 / 60.0, avatar, &[]);
                    }
                    if ui.button(t!("inspector.reset")).clicked() {
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

fn draw_watched_folders(ui: &mut egui::Ui, state: &mut GuiApp) {
    egui::CollapsingHeader::new(t!("watched_folders.heading"))
        .default_open(false)
        .show(ui, |ui| {
            // The list is a snapshot — we collect paths up front so the
            // mutating Remove path doesn't trip the egui borrow on
            // state.watched_avatar_dirs while iterating.
            let watched: Vec<std::path::PathBuf> = state.watched_avatar_dirs.clone();

            if watched.is_empty() {
                ui.label(
                    egui::RichText::new(t!("watched_folders.empty_hint"))
                        .small()
                        .color(egui::Color32::GRAY),
                );
            }

            let mut to_remove: Option<std::path::PathBuf> = None;
            for path in &watched {
                ui.horizontal(|ui| {
                    let exists = path.exists();
                    let label = path.to_string_lossy().into_owned();
                    let color = if exists {
                        egui::Color32::WHITE
                    } else {
                        egui::Color32::from_rgb(180, 100, 100)
                    };
                    ui.label(egui::RichText::new(label).color(color));
                    if !exists {
                        ui.label(
                            egui::RichText::new(t!("watched_folders.missing"))
                                .small()
                                .color(egui::Color32::RED),
                        );
                    }
                    if ui.button(t!("watched_folders.stop")).clicked() {
                        to_remove = Some(path.clone());
                    }
                });
            }

            ui.horizontal(|ui| {
                if ui.button(t!("watched_folders.watch_folder")).clicked() {
                    if let Some(picked) = rfd::FileDialog::new()
                        .set_title(t!("dialog.watch_folder_dialog"))
                        .pick_folder()
                    {
                        if let Err(e) = state.start_watching_folder(picked.clone()) {
                            state.push_notification(t!("watched_folders.failed_watch", path = picked.display().to_string(), error = e.to_string()));
                        }
                    }
                }
                let refresh_resp = ui.add_enabled(
                    !watched.is_empty(),
                    egui::Button::new(t!("tracking.refresh")),
                );
                if refresh_resp
                    .on_hover_text(
                        t!("watched_folders.refresh_tooltip"),
                    )
                    .clicked()
                {
                    state.refresh_watched_folders();
                }
            });

            if !watched.is_empty() {
                ui.label(
                    egui::RichText::new(
                        t!("watched_folders.cloud_hint"),
                    )
                    .small()
                    .color(egui::Color32::GRAY),
                );
            }

            if let Some(path) = to_remove {
                if let Err(e) = state.stop_watching_folder(&path) {
                    state.push_notification(t!("watched_folders.failed_stop", path = path.display().to_string(), error = e.to_string()));
                }
            }
        });
}

/// `Avatar Load Cache` panel: surfaces the on-disk cache populated by
/// `src/asset/cache.rs` so users can see how big it has grown and wipe it
/// without having to open Explorer. Co-located with `Watched folders`
/// because both are admin-style controls for the avatar library data
/// stored under `%APPDATA%\VulVATAR`.
fn draw_avatar_cache(ui: &mut egui::Ui, state: &mut GuiApp) {
    egui::CollapsingHeader::new(t!("avatar_cache.heading"))
        .default_open(false)
        .show(ui, |ui| {
            let stats = crate::asset::cache::stats();
            let dir = crate::persistence::cache_dir();
            ui.label(t!("avatar_cache.location", path = dir.display().to_string()));
            ui.label(t!("avatar_cache.entries", count = stats.entry_count, size = format!("{:.1}", stats.total_bytes as f64 / 1024.0 / 1024.0)));
            ui.label(
                egui::RichText::new(
                    t!("avatar_cache.cache_description"),
                )
                .small()
                .color(egui::Color32::GRAY),
            );
            ui.horizontal(|ui| {
                if ui.button(t!("avatar_cache.clear_cache")).clicked() {
                    match crate::asset::cache::clear_all() {
                        Ok(n) => {
                            state.push_notification(t!("avatar_cache.cleared", count = n));
                        }
                        Err(e) => {
                            state.push_notification(t!("avatar_cache.clear_failed", error = e.to_string()));
                        }
                    }
                }
            });
        });
}

fn draw_model_library(ui: &mut egui::Ui, state: &mut GuiApp) {
    egui::CollapsingHeader::new(t!("inspector.model_library"))
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut state.library_search_query);
                if state.library_search_query.is_empty() {
                    ui.label(t!("inspector.search"));
                }
            });

            ui.horizontal(|ui| {
                if ui.button(t!("inspector.sort_name")).clicked() {
                    state.app.avatar_library.sort_by_name();
                }
                if ui.button(t!("inspector.sort_recent")).clicked() {
                    state.app.avatar_library.sort_by_last_loaded();
                }
                if ui.button(t!("inspector.favorites_first")).clicked() {
                    state.app.avatar_library.sort_favorites_first();
                }
            });

            ui.checkbox(&mut state.library_show_missing, t!("inspector.show_missing"));

            ui.separator();

            draw_watched_folders(ui, state);

            ui.separator();

            draw_avatar_cache(ui, state);

            ui.separator();

            let current_avatar_path = state
                .app
                .active_avatar()
                .map(|a| a.asset.source_path.clone());

            // Local row struct to keep the entries collector readable now
            // that we surface VRM metadata (title / author) alongside the
            // existing display fields. The library view runs once per
            // frame so the per-entry clones are cheap.
            struct LibraryRow {
                idx: usize,
                name: String,
                path: std::path::PathBuf,
                favorite: bool,
                exists: bool,
                mesh_count: Option<usize>,
                material_count: Option<usize>,
                vrm_title: Option<String>,
                vrm_author: Option<String>,
                thumbnail_path: Option<std::path::PathBuf>,
            }

            let entries: Vec<LibraryRow> = {
                let lib = &state.app.avatar_library;
                let to_row = |idx: usize,
                              e: &crate::app::avatar_library::AvatarLibraryEntry|
                 -> LibraryRow {
                    LibraryRow {
                        idx,
                        name: e.name.clone(),
                        path: e.path.clone(),
                        favorite: e.favorite,
                        exists: e.exists(),
                        mesh_count: e.mesh_count,
                        material_count: e.material_count,
                        vrm_title: e.vrm_title.clone(),
                        vrm_author: e.vrm_author.clone(),
                        thumbnail_path: e.thumbnail_path.clone(),
                    }
                };
                if state.library_search_query.is_empty() {
                    lib.entries
                        .iter()
                        .enumerate()
                        .filter(|(_, e)| state.library_show_missing || e.exists())
                        .map(|(i, e)| to_row(i, e))
                        .collect()
                } else {
                    lib.search(&state.library_search_query)
                        .into_iter()
                        .filter(|e| state.library_show_missing || e.exists())
                        .filter_map(|e| {
                            let idx = lib.entries.iter().position(|x| x.path == e.path)?;
                            Some(to_row(idx, e))
                        })
                        .collect()
                }
            };

            let total = entries.len();
            let available = entries.iter().filter(|r| r.exists).count();
            ui.label(t!("inspector.models_count", total = total, available = available));

            let mut load_path: Option<std::path::PathBuf> = None;
            let mut remove_idx: Option<usize> = None;
            let mut toggle_fav_idx: Option<usize> = None;

            egui::ScrollArea::vertical()
                .max_height(300.0)
                .show(ui, |ui| {
                    for row in &entries {
                        let is_current =
                            current_avatar_path.as_ref().is_some_and(|p| *p == row.path);

                        let frame = if is_current {
                            egui::Frame::group(ui.style())
                                .fill(egui::Color32::from_rgba_unmultiplied(60, 80, 120, 200))
                                .stroke((1.0, egui::Color32::from_rgb(100, 150, 255)))
                        } else if !row.exists {
                            egui::Frame::group(ui.style())
                                .fill(egui::Color32::from_rgba_unmultiplied(80, 40, 40, 200))
                                .stroke((1.0, egui::Color32::from_rgb(180, 60, 60)))
                        } else {
                            egui::Frame::group(ui.style())
                        };

                        let response = frame.show(ui, |ui| {
                            ui.horizontal(|ui| {
                                let star = if row.favorite { "\u{2605}" } else { "\u{2606}" };
                                if ui.button(egui::RichText::new(star).size(14.0)).clicked() {
                                    toggle_fav_idx = Some(row.idx);
                                }

                                // 64×64 thumbnail. egui_extras' file:// loader
                                // decodes PNG/JPEG via the `image` crate and
                                // caches the GPU texture across frames, so the
                                // per-frame cost is negligible after the first
                                // display.
                                if let Some(thumb_path) = row.thumbnail_path.as_ref() {
                                    if thumb_path.exists() {
                                        let uri = format!("file://{}", thumb_path.display());
                                        ui.add(
                                            egui::Image::new(uri)
                                                .max_size(egui::vec2(64.0, 64.0))
                                                .fit_to_exact_size(egui::vec2(64.0, 64.0)),
                                        );
                                    }
                                }

                                ui.vertical(|ui| {
                                    ui.set_min_width(ui.available_width() - 60.0);
                                    let name_color = if is_current {
                                        egui::Color32::from_rgb(140, 190, 255)
                                    } else if !row.exists {
                                        egui::Color32::from_rgb(180, 100, 100)
                                    } else {
                                        egui::Color32::WHITE
                                    };
                                    ui.label(egui::RichText::new(&row.name).color(name_color));

                                    // VRM-supplied title (only when distinct from the
                                    // file-stem name, so we don't echo).
                                    if let Some(title) = row.vrm_title.as_deref() {
                                        if !title.is_empty() && title != row.name {
                                            ui.label(
                                                egui::RichText::new(title)
                                                    .italics()
                                                    .small()
                                                    .color(egui::Color32::from_rgb(180, 200, 220)),
                                            );
                                        }
                                    }

                                    let mut info = String::new();
                                    if let Some(author) = row.vrm_author.as_deref() {
                                        if !author.is_empty() {
                                            info.push_str(&format!("by {}", author));
                                        }
                                    }
                                    if let Some(mc) = row.mesh_count {
                                        if !info.is_empty() {
                                            info.push_str(" \u{2022} ");
                                        }
                                        info.push_str(&format!("{} meshes", mc));
                                    }
                                    if let Some(mc) = row.material_count {
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
                                        egui::RichText::new(row.path.to_string_lossy().as_ref())
                                            .small()
                                            .color(egui::Color32::DARK_GRAY),
                                    );
                                });

                                ui.vertical(|ui| {
                                    if row.exists {
                                        if ui.button(t!("inspector.load")).clicked() {
                                            load_path = Some(row.path.clone());
                                        }
                                    } else {
                                        ui.label(
                                            egui::RichText::new(t!("inspector.missing"))
                                                .small()
                                                .color(egui::Color32::RED),
                                        );
                                    }
                        if ui.button(t!("inspector.remove")).clicked() {
                                        remove_idx = Some(row.idx);
                                    }
                                });
                            })
                        });

                        if response.response.clicked() {
                            state.library_selected_index = Some(row.idx);
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
                    state.push_notification(t!("inspector.removed_library").to_string());
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
                    ui.label(egui::RichText::new(t!("inspector.details")).strong());

                    let entry = &mut state.app.avatar_library.entries[idx];
                    ui.horizontal(|ui| {
                        ui.label(t!("inspector.name"));
                        ui.text_edit_singleline(&mut entry.name);
                    });
                    ui.horizontal(|ui| {
                        ui.label(t!("inspector.tags"));
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
                        ui.label(t!("inspector.notes"));
                        ui.text_edit_multiline(&mut entry.notes);
                    });
                    if ui.button(t!("inspector.save_details")).clicked() {
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
                state.push_notification(t!("inspector.library_updated").to_string());
            }

            ui.separator();
            ui.horizontal(|ui| {
                if ui.button(t!("inspector.purge_missing")).clicked() {
                    let removed = state.app.avatar_library.purge_missing();
                    let _ = crate::persistence::save_avatar_library(&state.app.avatar_library);
                    state.push_notification(t!("inspector.purged_entries", count = removed));
                }
                if ui.button(t!("inspector.add_file")).clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("VRM 1.0", &["vrm"])
                        .pick_file()
                    {
                        let mut entry =
                            crate::app::avatar_library::AvatarLibraryEntry::from_path(&path);
                        if entry.thumbnail_path.as_ref().is_none_or(|p| !p.exists()) {
                            entry.thumbnail_path = state
                                .thumbnail_gen
                                .generate_and_save_placeholder(&entry.name);
                        }
                        state.app.avatar_library.add(entry);
                        let _ = crate::persistence::save_avatar_library(&state.app.avatar_library);
                        state.push_notification(t!("inspector.added_library", path = path.display().to_string()));
                    }
                }
            });
        });
}

fn draw_scene_presets(ui: &mut egui::Ui, state: &mut GuiApp) {
    egui::CollapsingHeader::new(t!("inspector.scene_presets"))
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button(t!("inspector.save_preset")).clicked() {
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
                if ui.button(t!("inspector.delete_selected")).clicked() {
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

    egui::CollapsingHeader::new(t!("inspector.expression_control"))
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button(t!("inspector.reset_all")).clicked() {
                    state.expression_weights.fill(0.0);
                }
                if ui.button(t!("inspector.set_all_50")).clicked() {
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

fn draw_settings(ui: &mut egui::Ui, state: &mut GuiApp) {
    let locales = crate::i18n::available_locales();
    let current_locale = state.settings.locale.clone();

    egui::CollapsingHeader::new(t!("settings.heading"))
        .default_open(true)
        .show(ui, |ui| {
            let selected_name = crate::i18n::locale_display_name(&current_locale);
            egui::ComboBox::from_label(t!("settings.language"))
                .selected_text(selected_name)
                .show_ui(ui, |ui| {
                    for code in &locales {
                        let name = crate::i18n::locale_display_name(code);
                        if ui.selectable_label(current_locale == *code, name).clicked() {
                            state.settings.locale = code.to_string();
                            crate::i18n::set_locale(code);
                            // Reorder the CJK font fallback chain so the
                            // newly-active locale's native shape wins per
                            // glyph (e.g. 飞 vs 飛). No-op if the user
                            // hasn't run dev.ps1 Install-Font yet — they
                            // see the warn line at startup either way.
                            if let Some(fonts) =
                                crate::gui::build_cjk_font_definitions(code)
                            {
                                ui.ctx().set_fonts(fonts);
                            }
                        }
                    }
                });
        });

    ui.add_space(4.0);

    egui::CollapsingHeader::new(t!("settings.viewport_controls"))
        .default_open(true)
        .show(ui, |ui| {
            ui.add(
                egui::Slider::new(&mut state.settings.zoom_sensitivity, 0.01..=0.5)
                    .text(t!("settings.zoom_sensitivity")),
            );
            ui.add(
                egui::Slider::new(&mut state.settings.orbit_sensitivity, 0.05..=1.0)
                    .text(t!("settings.orbit_sensitivity")),
            );
            ui.add(
                egui::Slider::new(&mut state.settings.pan_sensitivity, 0.1..=5.0)
                    .text(t!("settings.pan_sensitivity")),
            );
        });

    ui.add_space(4.0);

    egui::CollapsingHeader::new(t!("settings.autosave"))
        .default_open(false)
        .show(ui, |ui| {
            let mut secs = state.settings.autosave_interval_secs.unwrap_or(0);
            let changed = ui
                .add(
                    egui::Slider::new(&mut secs, 0..=3600)
                        .text(t!("settings.autosave_interval")),
                )
                .changed();
            if changed {
                if secs == 0 {
                    state.settings.autosave_interval_secs = None;
                    state.autosave_interval = None;
                } else {
                    state.settings.autosave_interval_secs = Some(secs);
                    state.autosave_interval = Some(std::time::Duration::from_secs(secs));
                }
            }
            if state.settings.autosave_interval_secs.is_none() {
                ui.label(t!("settings.autosave_disabled"));
            }
        });
}
