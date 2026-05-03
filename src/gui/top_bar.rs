use std::path::Path;
use std::sync::Arc;

use eframe::egui::{self, pos2, Align2, Color32, Response, Rounding, Sense, Stroke, Ui, Vec2};
use log::info;

use crate::asset::AvatarAsset;
use crate::gui::avatar_load::{AfterLoad, AvatarLoadJob};
use crate::gui::theme::{color, icon as ic, radius, space, typography};
use crate::gui::GuiApp;
use crate::persistence;
use crate::t;

const TOPBAR_HEIGHT: f32 = 56.0;

/// Kick off a background avatar load. The heavy CPU work (file I/O,
/// glTF parse, texture decode) runs on a worker thread; finalisation
/// happens on the UI thread once the asset arrives via
/// [`finalize_avatar_load`].
pub fn load_avatar_from_path(state: &mut GuiApp, path: &Path) {
    if state.avatar_load_job.is_some() {
        state.push_notification(t!("top_bar.avatar_load_in_progress"));
        return;
    }
    // Block while a calibration is mid-capture: the load Window
    // would otherwise pop on top of the calibration scrim, and the
    // user's HoldStill / Collecting timer would keep advancing
    // behind a UI they can't reach. Surfacing a notification gives
    // the user something to act on instead of silently swallowing
    // the click.
    if state.calibration_modal.is_open() {
        state.push_notification(t!("top_bar.avatar_load_blocked_by_calibration"));
        return;
    }
    // If a previous load is still in flight, refuse rather than
    // replacing the slot. Replacing detaches the prior worker — its
    // sends start failing silently and its decoded asset is discarded
    // when it finishes. Eventually harmless (the OS reaps the thread)
    // but wastes the multi-hundred-MB working set the worker built up.
    // The user almost certainly didn't mean to abandon the first load,
    // so a notification is more honest than a silent kick-off.
    if state.avatar_load_job.is_some() {
        state.push_notification(t!("top_bar.avatar_load_in_progress"));
        return;
    }
    state.push_notification(t!("top_bar.loading_avatar", path = path.display().to_string()));
    state.avatar_load_job = Some(AvatarLoadJob::spawn(path.to_path_buf(), AfterLoad::None));
}

/// UI-thread post-load work: build an `AvatarInstance`, attach it to physics,
/// register it in the library, and update recent-avatar history.
pub fn finalize_avatar_load(state: &mut GuiApp, path: &Path, asset: Arc<AvatarAsset>) {
    // If we're replacing an avatar that's already loaded at the same
    // library slot, the previous asset's GPU textures + meshes are
    // about to become unreferenced. The renderer caches them by URI
    // / MeshId and never evicts on its own, so without this nudge
    // they stay pinned in VRAM until process exit. `add_avatar`
    // below grows the avatar list rather than replacing in place,
    // but the active-avatar pointer moves — visual effect is the
    // same: the prior caches are stale.
    let had_existing_avatar = state.app.active_avatar().is_some();

    let instance_id = crate::avatar::AvatarInstanceId(state.app.next_avatar_instance_id);
    state.app.next_avatar_instance_id += 1;
    state.app.physics.attach_avatar(&asset);

    let mut entry = crate::app::avatar_library::AvatarLibraryEntry::from_path(path);
    entry.update_from_asset_with_thumbnail_dir(&asset, state.thumbnail_gen.output_dir());
    if entry
        .thumbnail_path
        .as_ref()
        .is_none_or(|p| !p.exists())
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
    if had_existing_avatar {
        state.app.evict_render_caches();
    }
    state.add_recent_avatar(path.to_path_buf());
    info!("avatar loaded: {}", path.display());
    state.push_notification(t!("top_bar.loaded_avatar", path = path.display().to_string()));

    let avatar_name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "avatar".to_string());
    let thumb_path = state.thumbnail_gen.thumbnail_path_for(&avatar_name);
    state.kick_thumbnail_job(thumb_path);
}

pub fn draw(ctx: &egui::Context, state: &mut GuiApp) {
    egui::TopBottomPanel::top("top_bar")
        .exact_height(TOPBAR_HEIGHT)
        .frame(egui::Frame {
            fill: color::SURFACE_BRIGHT,
            inner_margin: egui::Margin::symmetric(space::MD, space::SM),
            stroke: Stroke::new(1.0, color::OUTLINE_VARIANT),
            ..Default::default()
        })
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                draw_left(ui, state);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    draw_right(ui, state);
                    ui.add_space(space::MD);
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        draw_actions(ui, state);
                    });
                });
            });
        });
}

fn draw_left(ui: &mut Ui, state: &mut GuiApp) {
    if topbar_icon_button(ui, ic::MENU, &t!("app.toggle_inspector")).clicked() {
        state.inspector_open = !state.inspector_open;
    }
    ui.add_space(space::SM);
    let title_text = if state.project_dirty {
        format!("{} \u{2022}", state.mode.label())
    } else {
        state.mode.label()
    };
    ui.label(
        egui::RichText::new(title_text)
            .font(typography::heading())
            .color(color::ON_SURFACE)
            .strong(),
    );
}

fn draw_actions(ui: &mut Ui, state: &mut GuiApp) {
    ui.add_space(space::LG);
    if topbar_action(ui, ic::FOLDER_OPEN, &t!("top_bar.open_project")).clicked() {
        open_project(state);
    }
    if topbar_action(ui, ic::SAVE, &t!("top_bar.save_project")).clicked() {
        save_project(state);
    }
    ui.add_space(space::SM);
    if topbar_action(ui, ic::OPEN_OVERLAY, &t!("top_bar.open_overlay")).clicked() {
        open_overlay(state);
    }
    if topbar_action(ui, ic::SAVE_OVERLAY, &t!("top_bar.save_overlay_button")).clicked() {
        save_overlay(state);
    }
    ui.add_space(space::SM);
    let pause_glyph = if state.paused { ic::PLAY } else { ic::PAUSE };
    let pause_label = if state.paused {
        t!("top_bar.resume")
    } else {
        t!("top_bar.pause")
    };
    if topbar_action(ui, pause_glyph, &pause_label).clicked() {
        state.paused = !state.paused;
    }
}

fn draw_right(ui: &mut Ui, state: &mut GuiApp) {
    // Items lay out right-to-left: streaming combo first (rightmost),
    // then profile indicator. The overflow `more_vert` button was a
    // stub with no menu attached — re-add it once the Save As / Open
    // Avatar / Recent menu it was meant to host actually exists.

    // Profile combo
    let active_idx = state.profiles.active_index.unwrap_or(0);
    let profile_names: Vec<String> = state
        .profiles
        .profiles
        .iter()
        .map(|p| p.name.clone())
        .collect();
    let selected_name = profile_names
        .get(active_idx)
        .cloned()
        .unwrap_or_else(|| t!("top_bar.profile_none"));
    let mut clicked_index: Option<usize> = None;
    egui::ComboBox::from_id_salt("profile_selector")
        .selected_text(
            egui::RichText::new(selected_name)
                .font(typography::body())
                .color(color::ON_SURFACE),
        )
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
            state.push_notification(t!("top_bar.switched_profile", name = name));
        }
    }
    ui.add_space(space::SM);

    // Profile / connection indicator (green dot when tracking active).
    let active = state.is_tracking_active() && state.tracking.toggle_tracking;
    let dot = if active {
        color::SUCCESS
    } else {
        color::ON_SURFACE_MUTED
    };
    ui.label(
        egui::RichText::new(ic::STATUS_DOT.to_string())
            .font(typography::icon(10.0))
            .color(dot),
    );
}

// ─────────────────────────────────────────────────────────────────────
// Action wiring (extracted from previous monolithic draw to keep the
// layout block readable; behaviour preserved 1:1).
// ─────────────────────────────────────────────────────────────────────

fn open_project(state: &mut GuiApp) {
    let Some(path) = rfd::FileDialog::new()
        .add_filter(t!("top_bar.filter_project"), &["vvtproj"])
        .pick_file()
    else {
        return;
    };
    match persistence::load_project(&path) {
        Ok((project_state, load_warnings)) => {
            if let Some(ref overlay_path_str) = project_state.active_overlay_path {
                let overlay_file = std::path::Path::new(overlay_path_str);
                if overlay_file.exists() {
                    match persistence::load_cloth_overlay(overlay_file) {
                        Ok(overlay_file_data) => {
                            if let Some(cloth_asset) = overlay_file_data.cloth_asset {
                                state.app.editor.overlay_asset = Some(cloth_asset);
                                state.app.editor.overlay_path =
                                    Some(overlay_file.to_path_buf());
                            }
                        }
                        Err(e) => {
                            state.push_notification(t!(
                                "top_bar.failed_load_overlay",
                                path = overlay_path_str.to_string(),
                                error = e.to_string()
                            ));
                        }
                    }
                }
            }

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
                state.push_notification(t!(
                    "top_bar.avatar_not_found",
                    path = missing.to_string()
                ));
            }

            if let Some(avatar_path) = avatar_to_load {
                if state.avatar_load_job.is_some() {
                    state.push_notification(t!("top_bar.avatar_load_in_progress"));
                } else {
                    state.push_notification(t!(
                        "top_bar.loading_project",
                        path = path.display().to_string()
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
                    state.push_notification(t!("toast.warning", msg = w.to_string()));
                }
                state.push_notification(t!(
                    "top_bar.opened_project",
                    path = path.display().to_string()
                ));
            }
        }
        Err(e) => {
            state.push_notification(t!("top_bar.failed_load_project", error = e.to_string()));
        }
    }
}

fn save_project(state: &mut GuiApp) {
    let ps = state.to_project_state();
    if let Some(ref path) = state.project_path.clone() {
        match persistence::save_project(&ps, path) {
            Ok(()) => {
                state.project_dirty = false;
                state.push_notification(t!("top_bar.project_saved"));
            }
            Err(e) => {
                state.push_notification(t!("top_bar.save_failed", error = e.to_string()));
            }
        }
    } else if let Some(path) = rfd::FileDialog::new()
        .add_filter(t!("top_bar.filter_project"), &["vvtproj"])
        .set_file_name("project.vvtproj")
        .save_file()
    {
        match persistence::save_project(&ps, &path) {
            Ok(()) => {
                state.project_path = Some(path);
                state.project_dirty = false;
                state.push_notification(t!("top_bar.project_saved"));
            }
            Err(e) => {
                state.push_notification(t!("top_bar.save_failed", error = e.to_string()));
            }
        }
    }
}

fn open_overlay(state: &mut GuiApp) {
    let Some(path) = rfd::FileDialog::new()
        .add_filter(t!("top_bar.filter_cloth"), &["vvtcloth"])
        .pick_file()
    else {
        return;
    };
    match persistence::load_cloth_overlay(&path) {
        Ok(overlay) => {
            if let Some(cloth_asset) = overlay.cloth_asset {
                state.app.editor.overlay_asset = Some(cloth_asset);
                state.app.editor.dirty = false;
            }
            state.app.editor.overlay_path = Some(path.clone());
            state.push_notification(t!(
                "top_bar.opened_overlay",
                name = overlay.overlay_name.to_string(),
                version = overlay.format_version
            ));
        }
        Err(e) => {
            state.push_notification(t!("top_bar.failed_load_cloth", error = e.to_string()));
        }
    }
}

fn save_overlay(state: &mut GuiApp) {
    match state.app.editor.save_overlay(None) {
        Ok(()) => state.push_notification(t!("top_bar.overlay_saved")),
        Err(e) => state.push_notification(t!(
            "top_bar.save_overlay_failed",
            error = e.to_string()
        )),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Layout primitives (icon-only / icon+label top-bar buttons).
// Custom widgets so we can mix icon font + label font in one button
// surface — egui's `Button::new(WidgetText)` only takes a single font
// per text run, so the natural composition would render the glyph in
// the proportional fallback (tofu).
// ─────────────────────────────────────────────────────────────────────

fn topbar_icon_button(ui: &mut Ui, glyph: char, hover_text: &str) -> Response {
    let size = Vec2::splat(36.0);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    let bg = if resp.hovered() {
        Color32::from_rgba_unmultiplied(124, 92, 255, 18)
    } else {
        Color32::TRANSPARENT
    };
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, Rounding::same(radius::PILL), bg);
    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        glyph.to_string(),
        typography::icon(20.0),
        color::ON_SURFACE_VARIANT,
    );
    resp.on_hover_text(hover_text)
}

fn topbar_action(ui: &mut Ui, glyph: char, label: &str) -> Response {
    let icon_size = 16.0;
    let label_galley = ui.painter().layout_no_wrap(
        label.to_string(),
        typography::body(),
        color::ON_SURFACE,
    );
    let label_w = label_galley.size().x;
    let h = 32.0;
    let pad_x = space::MD;
    let icon_label_gap = space::SM;
    let w = pad_x + icon_size + icon_label_gap + label_w + pad_x;
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, h), Sense::click());
    let bg = if resp.hovered() {
        Color32::from_rgba_unmultiplied(124, 92, 255, 18)
    } else {
        Color32::TRANSPARENT
    };
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, Rounding::same(radius::PILL), bg);
    painter.text(
        pos2(rect.left() + pad_x, rect.center().y),
        Align2::LEFT_CENTER,
        glyph.to_string(),
        typography::icon(icon_size),
        color::ON_SURFACE_VARIANT,
    );
    let label_pos = pos2(
        rect.left() + pad_x + icon_size + icon_label_gap,
        rect.center().y - label_galley.size().y * 0.5,
    );
    painter.galley(label_pos, label_galley, color::ON_SURFACE);
    resp
}
