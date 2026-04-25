use std::path::Path;
use std::sync::Arc;

use eframe::egui;
use log::info;

use crate::asset::AvatarAsset;
use crate::gui::avatar_load::{AfterLoad, AvatarLoadJob};
use crate::gui::GuiApp;
use crate::persistence;

/// Kick off a background avatar load. The heavy CPU work (file I/O,
/// glTF parse, texture decode) runs on a worker thread; finalisation
/// happens on the UI thread once the asset arrives via
/// [`finalize_avatar_load`].
pub fn load_avatar_from_path(state: &mut GuiApp, path: &Path) {
    if state.avatar_load_job.is_some() {
        state.push_notification("Avatar load already in progress.".to_string());
        return;
    }
    state.push_notification(format!("Loading avatar: {}", path.display()));
    state.avatar_load_job = Some(AvatarLoadJob::spawn(path.to_path_buf(), AfterLoad::None));
}

/// UI-thread post-load work: build an `AvatarInstance`, attach it to physics,
/// register it in the library, and update recent-avatar history.
pub fn finalize_avatar_load(state: &mut GuiApp, path: &Path, asset: Arc<AvatarAsset>) {
    let instance_id = crate::avatar::AvatarInstanceId(state.app.next_avatar_instance_id);
    state.app.next_avatar_instance_id += 1;
    state.app.physics.attach_avatar(&asset);

    let mut entry = crate::app::avatar_library::AvatarLibraryEntry::from_path(path);
    entry.update_from_asset_with_thumbnail_dir(&asset, state.thumbnail_gen.output_dir());
    // VRM didn't carry an embedded cover image — fall back to a generated
    // placeholder so the library row has *something* to display.
    if entry
        .thumbnail_path
        .as_ref()
        .map_or(true, |p| !p.exists())
    {
        entry.thumbnail_path = state.thumbnail_gen.generate_and_save_placeholder(&entry.name);
    }
    state.app.avatar_library.add(entry);
    let _ = crate::persistence::save_avatar_library(&state.app.avatar_library);

    let (pan_y, distance) =
        crate::gui::autoframe_aabb(&asset.root_aabb, state.rendering.camera_fov, 1.0);
    state.camera_orbit.pan = [0.0, pan_y];
    state.camera_orbit.distance = distance;
    state.camera_orbit.target_distance = distance;
    state.camera_orbit.yaw_deg = 0.0;
    state.camera_orbit.pitch_deg = 0.0;

    let instance = crate::avatar::AvatarInstance::new(instance_id, asset);
    state.app.add_avatar(instance);
    state.add_recent_avatar(path.to_path_buf());
    info!("avatar loaded: {}", path.display());
    state.push_notification(format!("Loaded avatar: {}", path.display()));
}

pub fn draw(ctx: &egui::Context, state: &mut GuiApp) {
    // E12: Show dirty indicator in the title heading.
    let title = if state.project_dirty {
        "VulVATAR *"
    } else {
        "VulVATAR"
    };

    egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.heading(title);
            ui.separator();

            if ui.button("Open Avatar").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("VRM 1.0", &["vrm"])
                    .pick_file()
                {
                    load_avatar_from_path(state, &path);
                }
            }

            // E11: Recent avatars menu.
            ui.menu_button("Recent", |ui| {
                if state.recent_avatars.is_empty() {
                    ui.label("No recent avatars");
                } else {
                    let mut load_path: Option<std::path::PathBuf> = None;
                    for path in &state.recent_avatars {
                        if ui.button(path.to_string_lossy().as_ref()).clicked() {
                            load_path = Some(path.clone());
                            ui.close_menu();
                        }
                    }
                    if let Some(path) = load_path {
                        load_avatar_from_path(state, &path);
                    }
                }
            });

            ui.separator();

            if ui.button("Open Project").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("VulVATAR Project", &["vvtproj"])
                    .pick_file()
                {
                    match persistence::load_project(&path) {
                        Ok((project_state, load_warnings)) => {
                            // Cloth overlay: small JSON, load synchronously.
                            if let Some(ref overlay_path_str) = project_state.active_overlay_path {
                                let overlay_file = std::path::Path::new(overlay_path_str);
                                if overlay_file.exists() {
                                    match persistence::load_cloth_overlay(overlay_file) {
                                        Ok(overlay_file_data) => {
                                            if let Some(cloth_asset) = overlay_file_data.cloth_asset
                                            {
                                                state.app.editor.overlay_asset = Some(cloth_asset);
                                                state.app.editor.overlay_path =
                                                    Some(overlay_file.to_path_buf());
                                            }
                                        }
                                        Err(e) => {
                                            state.push_notification(format!(
                                                "Failed to load cloth overlay '{}': {}",
                                                overlay_path_str, e
                                            ));
                                        }
                                    }
                                }
                            }

                            // Avatar: heavy. Either spawn a job that applies the
                            // project state once it finishes, or apply immediately
                            // if no avatar is referenced.
                            let avatar_to_load = project_state
                                .avatar_source_path
                                .as_ref()
                                .map(std::path::PathBuf::from)
                                .filter(|p| p.exists());
                            if let Some(missing) = project_state
                                .avatar_source_path
                                .as_ref()
                                .filter(|p| !std::path::Path::new(p).exists())
                            {
                                state.push_notification(format!(
                                    "Avatar source file not found: '{}'",
                                    missing
                                ));
                            }

                            if let Some(avatar_path) = avatar_to_load {
                                if state.avatar_load_job.is_some() {
                                    state.push_notification(
                                        "Avatar load already in progress.".to_string(),
                                    );
                                } else {
                                    state.push_notification(format!(
                                        "Loading project '{}'…",
                                        path.display()
                                    ));
                                    state.avatar_load_job = Some(AvatarLoadJob::spawn(
                                        avatar_path,
                                        AfterLoad::ApplyProject {
                                            project_state: Box::new(project_state),
                                            project_path: path.clone(),
                                            warnings: load_warnings,
                                        },
                                    ));
                                }
                            } else {
                                state.apply_project_state(&project_state);
                                state.project_path = Some(path.clone());
                                state.project_dirty = false;
                                for w in &load_warnings.warnings {
                                    state.push_notification(format!("Warning: {}", w));
                                }
                                state.push_notification(format!(
                                    "Opened project: {}",
                                    path.display()
                                ));
                            }
                        }
                        Err(e) => {
                            state.push_notification(format!("Failed to load project: {}", e));
                        }
                    }
                }
            }

            if ui.button("Save Project").clicked() {
                let ps = state.to_project_state();
                if let Some(ref path) = state.project_path.clone() {
                    // Re-save to the known path.
                    match persistence::save_project(&ps, path) {
                        Ok(()) => {
                            // E12: Clear dirty on save.
                            state.project_dirty = false;
                            state.push_notification("Project saved.".to_string());
                        }
                        Err(e) => {
                            state.push_notification(format!("Save project failed: {}", e));
                        }
                    }
                } else {
                    // No path yet -- behave like Save As.
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("VulVATAR Project", &["vvtproj"])
                        .set_file_name("project.vvtproj")
                        .save_file()
                    {
                        match persistence::save_project(&ps, &path) {
                            Ok(()) => {
                                state.project_path = Some(path);
                                state.project_dirty = false;
                                state.push_notification("Project saved.".to_string());
                            }
                            Err(e) => {
                                state.push_notification(format!("Save project failed: {}", e));
                            }
                        }
                    }
                }
            }
            if ui.button("Save As").clicked() {
                let ps = state.to_project_state();
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("VulVATAR Project", &["vvtproj"])
                    .set_file_name("project.vvtproj")
                    .save_file()
                {
                    match persistence::save_project(&ps, &path) {
                        Ok(()) => {
                            state.project_path = Some(path);
                            state.project_dirty = false;
                            state.push_notification("Project saved.".to_string());
                        }
                        Err(e) => {
                            state.push_notification(format!("Save as failed: {}", e));
                        }
                    }
                }
            }

            ui.separator();

            if ui.button("Open Overlay").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("VulVATAR Cloth Overlay", &["vvtcloth"])
                    .pick_file()
                {
                    match persistence::load_cloth_overlay(&path) {
                        Ok(overlay) => {
                            if let Some(cloth_asset) = overlay.cloth_asset {
                                state.app.editor.overlay_asset = Some(cloth_asset);
                                state.app.editor.dirty = false;
                            }
                            state.app.editor.overlay_path = Some(path.clone());
                            state.push_notification(format!(
                                "Opened overlay '{}' (format v{})",
                                overlay.overlay_name, overlay.format_version
                            ));
                        }
                        Err(e) => {
                            state.push_notification(format!("Failed to load cloth overlay: {}", e));
                        }
                    }
                }
            }
            if ui.button("Save Overlay").clicked() {
                match state.app.editor.save_overlay(None) {
                    Ok(()) => state.push_notification("Overlay saved.".to_string()),
                    Err(e) => state.push_notification(format!("Save overlay failed: {}", e)),
                }
            }

            ui.separator();

            if state.paused {
                if ui.button("Resume").clicked() {
                    state.paused = false;
                }
            } else {
                if ui.button("Pause").clicked() {
                    state.paused = true;
                }
            }

            ui.separator();

            ui.label("Profile:");
            let active_idx = state.profiles.active_index.unwrap_or(0);
            let profile_names: Vec<String> = state
                .profiles
                .profiles
                .iter()
                .map(|p| p.name.clone())
                .collect();
            let selected_name = profile_names
                .get(active_idx)
                .map(|s| s.as_str())
                .unwrap_or("None");
            let mut clicked_index: Option<usize> = None;
            egui::ComboBox::from_id_salt("profile_selector")
                .selected_text(selected_name)
                .show_ui(ui, |ui| {
                    for (i, name) in profile_names.iter().enumerate() {
                        if ui
                            .selectable_label(i == active_idx, name.as_str())
                            .clicked()
                        {
                            clicked_index = Some(i);
                        }
                    }
                });
            if let Some(i) = clicked_index {
                if let Some(profile) = state.profiles.profiles.get(i).cloned() {
                    let name = profile.name.clone();
                    state.profiles.set_active(i);
                    state.apply_profile(&profile);
                    state.push_notification(format!("Switched to profile: {}", name));
                }
            }
        });
    });
}
