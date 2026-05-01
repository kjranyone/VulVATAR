use eframe::egui;

use crate::gui::GuiApp;
use crate::t;

pub(super) fn draw_watched_folders(ui: &mut egui::Ui, state: &mut GuiApp) {
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
pub(super) fn draw_avatar_cache(ui: &mut egui::Ui, state: &mut GuiApp) {
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

pub(super) fn draw_model_library(ui: &mut egui::Ui, state: &mut GuiApp) {
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
                    crate::gui::top_bar::load_avatar_from_path(state, &path);
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
