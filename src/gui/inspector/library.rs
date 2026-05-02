use eframe::egui;

use crate::gui::components::{card, chip, icon_text, outlined_button};
use crate::gui::theme::{color, icon as ic, radius, space, typography};
use crate::gui::{GuiApp, LibrarySortMode};
use crate::t;

/// Watched-folder accordion. Stays as `CollapsingHeader` because it
/// is a sub-section of the Model Library card; the chrome already
/// reads correctly under the new theme.
pub(super) fn draw_watched_folders(ui: &mut egui::Ui, state: &mut GuiApp) {
    egui::CollapsingHeader::new(t!("watched_folders.heading"))
        .default_open(false)
        .show(ui, |ui| {
            let watched: Vec<std::path::PathBuf> = state.watched_avatar_dirs.clone();

            if watched.is_empty() {
                ui.label(
                    egui::RichText::new(t!("watched_folders.empty_hint"))
                        .font(typography::caption())
                        .color(color::ON_SURFACE_MUTED),
                );
            }

            let mut to_remove: Option<std::path::PathBuf> = None;
            for path in &watched {
                ui.horizontal(|ui| {
                    let exists = path.exists();
                    let label = path.to_string_lossy().into_owned();
                    let label_color = if exists {
                        color::ON_SURFACE
                    } else {
                        color::ON_ERROR_CONTAINER
                    };
                    ui.label(
                        egui::RichText::new(label)
                            .font(typography::caption())
                            .color(label_color),
                    );
                    if !exists {
                        ui.label(
                            egui::RichText::new(t!("watched_folders.missing"))
                                .font(typography::caption())
                                .color(color::ERROR),
                        );
                    }
                    if ui.small_button(t!("watched_folders.stop")).clicked() {
                        to_remove = Some(path.clone());
                    }
                });
            }

            ui.add_space(space::XS);
            ui.horizontal(|ui| {
                if outlined_button(
                    ui,
                    Some(ic::ADD),
                    &t!("watched_folders.watch_folder"),
                    color::PRIMARY,
                )
                .clicked()
                {
                    if let Some(picked) = rfd::FileDialog::new()
                        .set_title(t!("dialog.watch_folder_dialog"))
                        .pick_folder()
                    {
                        if let Err(e) = state.start_watching_folder(picked.clone()) {
                            state.push_notification(t!(
                                "watched_folders.failed_watch",
                                path = picked.display().to_string(),
                                error = e.to_string()
                            ));
                        }
                    }
                }
                let refresh_resp = ui.add_enabled(
                    !watched.is_empty(),
                    egui::Button::new(t!("tracking.refresh")),
                );
                if refresh_resp
                    .on_hover_text(t!("watched_folders.refresh_tooltip"))
                    .clicked()
                {
                    state.refresh_watched_folders();
                }
            });

            if !watched.is_empty() {
                ui.label(
                    egui::RichText::new(t!("watched_folders.cloud_hint"))
                        .font(typography::caption())
                        .color(color::ON_SURFACE_MUTED),
                );
            }

            if let Some(path) = to_remove {
                if let Err(e) = state.stop_watching_folder(&path) {
                    state.push_notification(t!(
                        "watched_folders.failed_stop",
                        path = path.display().to_string(),
                        error = e.to_string()
                    ));
                }
            }
        });
}

/// Avatar Load Cache accordion — surfaces cache stats so users can
/// see size and wipe without opening Explorer.
pub(super) fn draw_avatar_cache(ui: &mut egui::Ui, state: &mut GuiApp) {
    egui::CollapsingHeader::new(t!("avatar_cache.heading"))
        .default_open(false)
        .show(ui, |ui| {
            let stats = crate::asset::cache::stats();
            let dir = crate::persistence::cache_dir();
            ui.label(
                egui::RichText::new(t!(
                    "avatar_cache.location",
                    path = dir.display().to_string()
                ))
                .font(typography::caption())
                .color(color::ON_SURFACE_VARIANT),
            );
            ui.label(
                egui::RichText::new(t!(
                    "avatar_cache.entries",
                    count = stats.entry_count,
                    size = format!("{:.1}", stats.total_bytes as f64 / 1024.0 / 1024.0)
                ))
                .font(typography::body())
                .color(color::ON_SURFACE),
            );
            ui.label(
                egui::RichText::new(t!("avatar_cache.cache_description"))
                    .font(typography::caption())
                    .color(color::ON_SURFACE_MUTED),
            );
            ui.add_space(space::XS);
            if outlined_button(
                ui,
                Some(ic::DELETE),
                &t!("avatar_cache.clear_cache"),
                color::ERROR,
            )
            .clicked()
            {
                match crate::asset::cache::clear_all() {
                    Ok(n) => {
                        state.push_notification(t!("avatar_cache.cleared", count = n));
                    }
                    Err(e) => {
                        state.push_notification(t!(
                            "avatar_cache.clear_failed",
                            error = e.to_string()
                        ));
                    }
                }
            }
        });
}

pub(super) fn draw_model_library(ui: &mut egui::Ui, state: &mut GuiApp) {
    card(ui, t!("inspector.model_library"), |ui| {
        // ── Search ───────────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.label(icon_text(ic::SEARCH, 16.0));
            ui.add(
                egui::TextEdit::singleline(&mut state.library_search_query)
                    .hint_text(t!("inspector.search"))
                    .desired_width(f32::INFINITY),
            );
        });
        ui.add_space(space::SM);

        // ── Sort chips ───────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = space::SM;
            if chip(
                ui,
                &t!("inspector.sort_name"),
                state.library_sort_mode == LibrarySortMode::Name,
            )
            .clicked()
            {
                state.app.avatar_library.sort_by_name();
                state.library_sort_mode = LibrarySortMode::Name;
            }
            if chip(
                ui,
                &t!("inspector.sort_recent"),
                state.library_sort_mode == LibrarySortMode::Recent,
            )
            .clicked()
            {
                state.app.avatar_library.sort_by_last_loaded();
                state.library_sort_mode = LibrarySortMode::Recent;
            }
            if chip(
                ui,
                &t!("inspector.favorites_first"),
                state.library_sort_mode == LibrarySortMode::Favorites,
            )
            .clicked()
            {
                state.app.avatar_library.sort_favorites_first();
                state.library_sort_mode = LibrarySortMode::Favorites;
            }
        });
        ui.add_space(space::XS);

        // ── Show missing toggle ──────────────────────────────────
        ui.checkbox(&mut state.library_show_missing, t!("inspector.show_missing"));
        ui.add_space(space::SM);

        // ── Sub-accordions ───────────────────────────────────────
        draw_watched_folders(ui, state);
        draw_avatar_cache(ui, state);
        ui.add_space(space::SM);

        // ── Entry list ───────────────────────────────────────────
        let current_avatar_path = state
            .app
            .active_avatar()
            .map(|a| a.asset.source_path.clone());

        struct LibraryRow {
            idx: usize,
            name: String,
            path: std::path::PathBuf,
            favorite: bool,
            exists: bool,
            mesh_count: Option<usize>,
            material_count: Option<usize>,
            vrm_author: Option<String>,
            thumbnail_path: Option<std::path::PathBuf>,
        }

        let entries: Vec<LibraryRow> = {
            let lib = &state.app.avatar_library;
            let to_row = |idx: usize, e: &crate::app::avatar_library::AvatarLibraryEntry| -> LibraryRow {
                LibraryRow {
                    idx,
                    name: e.name.clone(),
                    path: e.path.clone(),
                    favorite: e.favorite,
                    exists: e.exists(),
                    mesh_count: e.mesh_count,
                    material_count: e.material_count,
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
        ui.label(
            egui::RichText::new(t!(
                "inspector.models_count",
                total = total,
                available = available
            ))
            .font(typography::caption())
            .color(color::ON_SURFACE_VARIANT),
        );

        let mut load_path: Option<std::path::PathBuf> = None;
        let mut remove_idx: Option<usize> = None;
        let mut toggle_fav_idx: Option<usize> = None;

        egui::ScrollArea::vertical()
            .max_height(360.0)
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = space::SM;
                for row in &entries {
                    let is_current = current_avatar_path
                        .as_ref()
                        .is_some_and(|p| *p == row.path);

                    let (row_fill, row_stroke) = if is_current {
                        (color::PRIMARY_CONTAINER, color::PRIMARY)
                    } else if !row.exists {
                        (color::ERROR_CONTAINER, color::ERROR)
                    } else {
                        (color::SURFACE, color::OUTLINE_VARIANT)
                    };

                    let frame = egui::Frame {
                        fill: row_fill,
                        rounding: egui::Rounding::same(radius::SM),
                        inner_margin: egui::Margin::symmetric(space::SM, space::SM),
                        stroke: egui::Stroke::new(1.0, row_stroke),
                        ..Default::default()
                    };

                    const ACTION_COL_WIDTH: f32 = 96.0;

                    let response = frame.show(ui, |ui| {
                        ui.horizontal(|ui| {
                            // Favourite toggle
                            let star = if row.favorite {
                                ic::FAVORITE_FILLED
                            } else {
                                ic::FAVORITE_BORDER
                            };
                            let star_color = if row.favorite {
                                color::WARNING
                            } else {
                                color::ON_SURFACE_VARIANT
                            };
                            let star_resp = ui.add(
                                egui::Button::new(
                                    egui::RichText::new(star.to_string())
                                        .font(typography::icon(18.0))
                                        .color(star_color),
                                )
                                .frame(false),
                            );
                            if star_resp.clicked() {
                                toggle_fav_idx = Some(row.idx);
                            }

                            // Thumbnail (64×64). file:// loader requires
                            // egui_extras "file" feature — already wired
                            // up in PR1.
                            if let Some(thumb_path) = row.thumbnail_path.as_ref() {
                                if thumb_path.exists() {
                                    let uri = format!("file://{}", thumb_path.display());
                                    ui.add(
                                        egui::Image::new(uri)
                                            .max_size(egui::vec2(56.0, 56.0))
                                            .fit_to_exact_size(egui::vec2(56.0, 56.0))
                                            .rounding(egui::Rounding::same(radius::SM)),
                                    );
                                }
                            }

                            let middle_width =
                                (ui.available_width() - ACTION_COL_WIDTH).max(80.0);

                            ui.allocate_ui_with_layout(
                                egui::vec2(middle_width, 0.0),
                                egui::Layout::top_down(egui::Align::LEFT),
                                |ui| {
                                    ui.set_min_width(middle_width);
                                    ui.spacing_mut().item_spacing.y = 2.0;
                                    let name_color = if is_current {
                                        color::PRIMARY
                                    } else if !row.exists {
                                        color::ON_ERROR_CONTAINER
                                    } else {
                                        color::ON_SURFACE
                                    };
                                    ui.label(
                                        egui::RichText::new(&row.name)
                                            .font(typography::body())
                                            .color(name_color)
                                            .strong(),
                                    );

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
                                                .font(typography::caption())
                                                .color(color::ON_SURFACE_VARIANT),
                                        );
                                    }

                                    if !row.exists {
                                        ui.label(
                                            egui::RichText::new(format!(
                                                "\u{26a0} {}",
                                                row.path.to_string_lossy()
                                            ))
                                            .font(typography::caption())
                                            .color(color::ERROR),
                                        );
                                    } else {
                                        ui.label(
                                            egui::RichText::new(
                                                row.path.to_string_lossy().as_ref(),
                                            )
                                            .font(typography::caption())
                                            .color(color::ON_SURFACE_MUTED),
                                        );
                                    }
                                },
                            );

                            ui.allocate_ui_with_layout(
                                egui::vec2(ACTION_COL_WIDTH, 0.0),
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    // More-vert opens a small menu
                                    // hosting Remove. The mockup shows
                                    // the dot trigger to keep the row
                                    // visually quiet.
                                    let menu_resp = ui
                                        .menu_button(
                                            egui::RichText::new(ic::MORE_VERT.to_string())
                                                .font(typography::icon(18.0))
                                                .color(color::ON_SURFACE_VARIANT),
                                            |ui| {
                                                if ui
                                                    .button(t!("inspector.remove"))
                                                    .clicked()
                                                {
                                                    remove_idx = Some(row.idx);
                                                    ui.close_menu();
                                                }
                                            },
                                        );
                                    let _ = menu_resp;

                                    if row.exists {
                                        if outlined_button(
                                            ui,
                                            None,
                                            &t!("inspector.load"),
                                            color::PRIMARY,
                                        )
                                        .clicked()
                                        {
                                            load_path = Some(row.path.clone());
                                        }
                                    } else {
                                        ui.label(
                                            egui::RichText::new(t!("inspector.missing"))
                                                .font(typography::caption())
                                                .color(color::ERROR),
                                        );
                                    }
                                },
                            );
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

        // ── Selected entry detail editor ────────────────────────
        let mut details_save_requested = false;
        let mut tag_to_add: Option<String> = None;
        let mut tag_to_remove: Option<usize> = None;

        if let Some(idx) = state.library_selected_index {
            if idx < state.app.avatar_library.entries.len() {
                ui.add_space(space::SM);
                ui.separator();
                ui.add_space(space::SM);
                ui.label(
                    egui::RichText::new(t!("inspector.details"))
                        .font(typography::title())
                        .color(color::ON_SURFACE)
                        .strong(),
                );
                let entry = &mut state.app.avatar_library.entries[idx];
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(t!("inspector.name"))
                            .font(typography::body())
                            .color(color::ON_SURFACE_VARIANT),
                    );
                    ui.text_edit_singleline(&mut entry.name);
                });
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(t!("inspector.tags"))
                            .font(typography::body())
                            .color(color::ON_SURFACE_VARIANT),
                    );
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
                            if ui.small_button(tag_label).clicked() {
                                tag_to_remove = Some(tag_i);
                            }
                        }
                    });
                }

                let entry = &mut state.app.avatar_library.entries[idx];
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(t!("inspector.notes"))
                            .font(typography::body())
                            .color(color::ON_SURFACE_VARIANT),
                    );
                    ui.text_edit_multiline(&mut entry.notes);
                });
                if outlined_button(
                    ui,
                    Some(ic::SAVE),
                    &t!("inspector.save_details"),
                    color::PRIMARY,
                )
                .clicked()
                {
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

        // ── Footer actions ───────────────────────────────────────
        ui.add_space(space::SM);
        ui.separator();
        ui.add_space(space::SM);
        ui.horizontal(|ui| {
            if outlined_button(
                ui,
                Some(ic::DELETE),
                &t!("inspector.purge_missing"),
                color::ERROR,
            )
            .clicked()
            {
                let removed = state.app.avatar_library.purge_missing();
                let _ = crate::persistence::save_avatar_library(&state.app.avatar_library);
                state.push_notification(t!("inspector.purged_entries", count = removed));
            }
            if outlined_button(
                ui,
                Some(ic::ADD),
                &t!("inspector.add_file"),
                color::PRIMARY,
            )
            .clicked()
            {
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
                    state.push_notification(t!(
                        "inspector.added_library",
                        path = path.display().to_string()
                    ));
                }
            }
        });
    });
}
