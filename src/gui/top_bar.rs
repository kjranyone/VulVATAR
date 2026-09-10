use std::path::{Path, PathBuf};
use std::sync::mpsc;
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
    if state.library.avatar_load_job.is_some() {
        state.push_notification(t!("top_bar.avatar_load_in_progress"));
        return;
    }
    // Block while a calibration is mid-capture: the load Window
    // would otherwise pop on top of the calibration scrim, and the
    // user's HoldStill / Collecting timer would keep advancing
    // behind a UI they can't reach. Surfacing a notification gives
    // the user something to act on instead of silently swallowing
    // the click.
    if state.calibration.modal.is_open() {
        state.push_notification(t!("top_bar.avatar_load_blocked_by_calibration"));
        return;
    }
    state.push_notification(t!("top_bar.loading_avatar", path = path.display().to_string()));
    state.library.avatar_load_job = Some(AvatarLoadJob::spawn(path.to_path_buf(), AfterLoad::None));
}

/// UI-thread post-load work: build an `AvatarInstance` and install it as
/// the scene avatar (replacing any previous one — see
/// `Application::set_avatar`), register it in the library, and update
/// recent-avatar history.
pub fn finalize_avatar_load(state: &mut GuiApp, path: &Path, asset: Arc<AvatarAsset>) {
    let instance_id = crate::avatar::AvatarInstanceId(state.app.next_avatar_instance_id);
    state.app.next_avatar_instance_id += 1;

    let mut entry = crate::app::avatar_library::AvatarLibraryEntry::from_path(path);
    entry.update_from_asset_with_thumbnail_dir(&asset, state.library.thumbnail_gen.output_dir());
    if entry
        .thumbnail_path
        .as_ref()
        .is_none_or(|p| !p.exists())
    {
        entry.thumbnail_path = state.library.thumbnail_gen.generate_and_save_placeholder(&entry.name);
    }
    state.app.avatar_library.add(entry);
    state.save_avatar_library_with_toast();

    // Auto-frame only the *first* avatar of the session. On a
    // replacement the user has usually adjusted the orbit to taste —
    // resetting yaw/pitch/distance on every reload threw that away.
    // (Project restore is unaffected either way: `AfterLoad::
    // ApplyProject` re-applies the saved orbit after this runs.)
    if state.app.active_avatar().is_none() {
        let (pan_y, distance) =
            crate::gui::autoframe_aabb(&asset.root_aabb, state.rendering.camera_fov, 1.0);
        state.camera_orbit.pan = [0.0, pan_y];
        state.camera_orbit.distance = distance;
        state.camera_orbit.target_distance = distance;
        state.camera_orbit.yaw_deg = 0.0;
        state.camera_orbit.pitch_deg = 0.0;
    }

    let instance = crate::avatar::AvatarInstance::new(instance_id, asset);
    state.app.set_avatar(instance);
    state.add_recent_avatar(path.to_path_buf());
    // The loaded avatar is part of the session state: without this an
    // "avatar-only" session (load VRM, quit) never triggers the
    // last-session autosave and the next launch starts empty. The
    // project-restore path (`AfterLoad::ApplyProject`) clears the flag
    // again right after applying, so restores don't loop a rewrite.
    info!("avatar loaded: {}", path.display());
    state.push_success_notification(t!("top_bar.loaded_avatar", path = path.display().to_string()));

    let avatar_name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "avatar".to_string());
    let thumb_path = state.library.thumbnail_gen.thumbnail_path_for(&avatar_name);
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
    // Title = the document, not the mode: "Tracking Setup •" read as
    // "this mode is unsaved". The mode identity lives on the nav rail;
    // what the dot qualifies is the project file.
    let unsaved =
        state.project_status.project_dirty || state.project_status.explicit_file_stale;
    let file_label = state
        .project_status
        .project_path
        .as_ref()
        .and_then(|p| p.file_name())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| t!("top_bar.untitled"));
    let title_text = if unsaved {
        format!("{file_label} \u{2022}")
    } else {
        file_label
    };
    let resp = ui.label(
        egui::RichText::new(title_text)
            .font(typography::heading())
            .color(color::ON_SURFACE)
            .strong(),
    );
    if let Some(p) = state.project_status.project_path.as_ref() {
        resp.on_hover_text(p.display().to_string());
    }
}

/// Display-ordered top-bar actions that may collapse into the File
/// menu when the bar is too narrow. Camera start/stop and the avatar
/// picker are session-critical and never collapse; pause goes first,
/// then calibrate.
const DROP_PRIORITY: [usize; 2] = [ACTION_PAUSE, ACTION_CALIBRATE];
const ACTION_AVATAR: usize = 0;
const ACTION_CAMERA: usize = 1;
const ACTION_CALIBRATE: usize = 2;
const ACTION_PAUSE: usize = 3;

/// Which display-ordered actions stay on the bar at `available` px.
/// Anything dropped is rendered inside the File menu instead — the
/// old bar let the layout clip actions silently at narrow widths.
pub(crate) fn fit_topbar_actions(available: f32, widths: &[f32]) -> Vec<bool> {
    let mut visible = vec![true; widths.len()];
    let used = |vis: &[bool]| -> f32 {
        vis.iter()
            .zip(widths)
            .filter(|(v, _)| **v)
            .map(|(_, w)| *w)
            .sum()
    };
    for &idx in DROP_PRIORITY.iter() {
        if used(&visible) <= available {
            break;
        }
        if idx < visible.len() {
            visible[idx] = false;
        }
    }
    visible
}

fn draw_actions(ui: &mut Ui, state: &mut GuiApp) {
    ui.add_space(space::LG);

    let avatar_label = state
        .app
        .active_avatar()
        .and_then(|a| {
            a.asset
                .source_path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
        })
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| t!("top_bar.avatar_menu_empty"));
    let pause_label = if state.runtime_status.paused {
        t!("top_bar.resume")
    } else {
        t!("top_bar.pause")
    };
    let camera_label = if state.is_tracking_active() {
        t!("top_bar.stop_camera")
    } else {
        t!("top_bar.start_camera")
    };
    let calibrate_label = t!("calibration.button");

    // Measure before laying out so narrow windows collapse the
    // low-priority actions into the File menu instead of clipping.
    let measure = |label: &str, icon: bool| -> f32 {
        let galley =
            ui.painter()
                .layout_no_wrap(label.to_string(), typography::body(), color::ON_SURFACE);
        let icon_w = if icon { 16.0 + space::SM } else { 0.0 };
        space::MD + icon_w + galley.size().x + space::MD + ui.spacing().item_spacing.x
    };
    let file_menu_w = measure(&t!("top_bar.file_menu"), false);
    let mut widths = [0.0f32; 4];
    widths[ACTION_AVATAR] = measure(&avatar_label, true) + 16.0; // combo arrow allowance
    widths[ACTION_CAMERA] = measure(&camera_label, true);
    widths[ACTION_CALIBRATE] = measure(&calibrate_label, false);
    widths[ACTION_PAUSE] = measure(&pause_label, true);
    let available = (ui.available_width() - file_menu_w).max(0.0);
    let visible = fit_topbar_actions(available, &widths);

    // ── Avatar picker (never collapses) ──────────────────────────
    draw_avatar_menu(ui, state, &avatar_label);

    // ── Camera start/stop (never collapses) ──────────────────────
    draw_camera_toggle(ui, state, &camera_label);

    // ── Calibrate ────────────────────────────────────────────────
    if visible[ACTION_CALIBRATE] {
        draw_calibrate_button(ui, state, &calibrate_label);
    }

    // ── Pause ────────────────────────────────────────────────────
    if visible[ACTION_PAUSE] {
        let pause_glyph = if state.runtime_status.paused { ic::PLAY } else { ic::PAUSE };
        if topbar_action(ui, pause_glyph, &pause_label).clicked() {
            state.runtime_status.paused = !state.runtime_status.paused;
        }
    }

    // ── File menu: project + overlay lifecycle, plus any actions the
    //    width squeeze pushed off the bar ─────────────────────────
    ui.menu_button(
        egui::RichText::new(t!("top_bar.file_menu")).font(typography::body()),
        |ui| {
            if ui.button(t!("top_bar.open_project")).clicked() {
                open_project(state);
                ui.close_menu();
            }
            if ui.button(t!("top_bar.save_project")).clicked() {
                save_project(state);
                ui.close_menu();
            }
            if ui.button(t!("top_bar.save_as")).clicked() {
                request_save_project_as_dialog(state);
                ui.close_menu();
            }
            ui.separator();
            if ui.button(t!("top_bar.open_overlay")).clicked() {
                open_overlay(state);
                ui.close_menu();
            }
            if ui.button(t!("top_bar.save_overlay_button")).clicked() {
                save_overlay(state);
                ui.close_menu();
            }
            let overflowed = visible.iter().any(|v| !v);
            if overflowed {
                ui.separator();
                if !visible[ACTION_CALIBRATE] && ui.button(calibrate_label.as_str()).clicked() {
                    state.open_calibration_modal();
                    ui.close_menu();
                }
                if !visible[ACTION_PAUSE] && ui.button(pause_label.as_str()).clicked() {
                    state.runtime_status.paused = !state.runtime_status.paused;
                    ui.close_menu();
                }
            }
        },
    );
}

/// The avatar picker — the one-click path to the app's mandatory
/// first step. Open file… / recent avatars / library entries, all
/// funnelling into the same async load as drag-and-drop.
fn draw_avatar_menu(ui: &mut Ui, state: &mut GuiApp, label: &str) {
    let mut chosen: Option<PathBuf> = None;
    // Plain proportional text — the Material Symbols glyphs only exist
    // in the icon font, and a menu_button label is a single text run
    // (see the layout-primitives note at the bottom of this file).
    ui.menu_button(
        egui::RichText::new(label.to_string()).font(typography::body()),
        |ui| {
            ui.set_min_width(220.0);
            if ui.button(format!("{}\u{2026}", t!("top_bar.open_avatar"))).clicked() {
                request_load_avatar_dialog(state);
                ui.close_menu();
            }

            // Recent — MRU list, existing files only.
            let recent: Vec<PathBuf> = state
                .project_status
                .recent_avatars
                .iter()
                .filter(|p| p.exists())
                .take(5)
                .cloned()
                .collect();
            ui.separator();
            ui.label(
                egui::RichText::new(t!("top_bar.recent"))
                    .font(typography::caption())
                    .color(color::ON_SURFACE_MUTED),
            );
            if recent.is_empty() {
                ui.add_enabled(false, egui::Button::new(t!("top_bar.no_recent")));
            }
            for path in recent {
                let name = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.display().to_string());
                if ui.button(name).on_hover_text(path.display().to_string()).clicked() {
                    chosen = Some(path);
                    ui.close_menu();
                }
            }

            // Library — favourites first, then the rest, scrollable.
            let mut entries: Vec<(String, PathBuf, bool)> = state
                .app
                .avatar_library
                .entries
                .iter()
                .filter(|e| e.path.exists())
                .map(|e| (e.name.clone(), e.path.clone(), e.favorite))
                .collect();
            if !entries.is_empty() {
                entries.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
                ui.separator();
                ui.label(
                    egui::RichText::new(t!("top_bar.library_section"))
                        .font(typography::caption())
                        .color(color::ON_SURFACE_MUTED),
                );
                egui::ScrollArea::vertical()
                    .max_height(240.0)
                    .show(ui, |ui| {
                        for (name, path, favorite) in entries {
                            // U+2605 from the text font, not the icon
                            // font — see the single-text-run note above.
                            let row = if favorite {
                                format!("\u{2605} {}", name)
                            } else {
                                name
                            };
                            if ui
                                .button(row)
                                .on_hover_text(path.display().to_string())
                                .clicked()
                            {
                                chosen = Some(path);
                                ui.close_menu();
                            }
                        }
                    });
            }
        },
    )
    .response
    .on_hover_text(t!("top_bar.open_avatar"));
    if let Some(path) = chosen {
        load_avatar_from_path(state, &path);
    }
}

/// Camera start/stop, promoted from the Tracking panel's collapsed
/// "input device" section — it's the every-session action. Wired to
/// the same Application entry points as the panel button.
fn draw_camera_toggle(ui: &mut Ui, state: &mut GuiApp, label: &str) {
    use crate::gui::components::{filled_button, tonal_button, ButtonTone};
    let active = state.is_tracking_active();
    let ready = state.is_tracking_ready();
    if active && ready {
        if tonal_button(ui, Some(ic::PAUSE), label, ButtonTone::Error, true).clicked() {
            state.app.stop_tracking();
        }
    } else if active {
        let _ = filled_button(ui, None, &t!("tracking.preparing"), false);
    } else if filled_button(ui, Some(ic::PLAY), label, true).clicked() {
        state.start_camera_with_current_params();
    }
}

fn draw_calibrate_button(ui: &mut Ui, state: &mut GuiApp, label: &str) {
    use crate::gui::components::{tonal_button, ButtonTone};
    let enabled = state.is_tracking_active() && state.library.avatar_load_job.is_none();
    let resp = tonal_button(ui, None, label, ButtonTone::Primary, enabled);
    if enabled {
        if resp.clicked() {
            state.open_calibration_modal();
        }
    } else {
        resp.on_hover_text(t!("top_bar.calibrate_needs_camera"));
    }
}

fn draw_right(ui: &mut Ui, state: &mut GuiApp) {
    // Items lay out right-to-left: profile combo first (rightmost),
    // then the tracking dot.

    // Profile combo + management. Rows show a calibration marker so
    // "which setups are calibrated" is visible before switching; the
    // management actions surface the create/duplicate/rename/delete
    // API that previously existed only in the data layer.
    let active_idx = state.profiles.active_index.unwrap_or(0);
    let rows: Vec<(String, bool)> = state
        .profiles
        .profiles
        .iter()
        .map(|p| (p.name.clone(), p.pose_calibration.is_some()))
        .collect();
    let selected_name = rows
        .get(active_idx)
        .map(|(n, _)| n.clone())
        .unwrap_or_else(|| t!("top_bar.profile_none"));
    let mut clicked_index: Option<usize> = None;
    let mut open_dialog: Option<ProfileDialog> = None;
    let mut duplicate_active = false;
    egui::ComboBox::from_id_salt("profile_selector")
        .selected_text(
            egui::RichText::new(selected_name)
                .font(typography::body())
                .color(color::ON_SURFACE),
        )
        .show_ui(ui, |ui| {
            for (i, (name, calibrated)) in rows.iter().enumerate() {
                // U+25C6 diamond from the text font marks a profile
                // that carries a pose calibration.
                let row_label = if *calibrated {
                    format!("{} \u{25C6}", name)
                } else {
                    name.clone()
                };
                let resp = ui.selectable_label(i == active_idx, row_label);
                let resp = if *calibrated {
                    resp.on_hover_text(t!("top_bar.profile_calibrated"))
                } else {
                    resp.on_hover_text(t!("top_bar.profile_uncalibrated"))
                };
                if resp.clicked() {
                    clicked_index = Some(i);
                }
            }
            ui.separator();
            if ui.button(t!("top_bar.profile_new")).clicked() {
                open_dialog = Some(ProfileDialog::New {
                    name: String::new(),
                });
                ui.close_menu();
            }
            if ui.button(t!("top_bar.profile_duplicate")).clicked() {
                duplicate_active = true;
                ui.close_menu();
            }
            if ui.button(t!("top_bar.profile_rename")).clicked() {
                let name = rows
                    .get(active_idx)
                    .map(|(n, _)| n.clone())
                    .unwrap_or_default();
                open_dialog = Some(ProfileDialog::Rename {
                    index: active_idx,
                    name,
                });
                ui.close_menu();
            }
            let can_remove = state.profiles.can_remove();
            let del = ui.add_enabled(
                can_remove,
                egui::Button::new(t!("top_bar.profile_delete")),
            );
            let del = if can_remove {
                del
            } else {
                del.on_disabled_hover_text(t!("top_bar.profile_delete_last"))
            };
            if del.clicked() {
                open_dialog = Some(ProfileDialog::Delete { index: active_idx });
                ui.close_menu();
            }
        });
    if let Some(dialog) = open_dialog {
        state.profile_dialog = Some(dialog);
    }
    if duplicate_active {
        if let Some(new_idx) = state.profiles.duplicate_at(active_idx) {
            let name = state.profiles.profiles[new_idx].name.clone();
            state.project_status.profiles_dirty = true;
            state.push_success_notification(t!("top_bar.profile_duplicated", name = name));
        }
    }
    if let Some(i) = clicked_index {
        if i != active_idx {
            // Switching from a calibrated setup to an uncalibrated one
            // clears the live calibration (apply_profile pushes `None`
            // through to the solver) — warn before doing that, because
            // a calibration capture costs the user a full hold-still
            // cycle to get back.
            let losing_calibration = state
                .profiles
                .active()
                .map(|p| p.pose_calibration.is_some())
                .unwrap_or(false)
                && state
                    .profiles
                    .profiles
                    .get(i)
                    .map(|p| p.pose_calibration.is_none())
                    .unwrap_or(false);
            if losing_calibration {
                state.pending_profile_switch = Some(i);
            } else {
                switch_to_profile(state, i);
            }
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

/// What a completed file-dialog thread was opened for. The dialog
/// itself runs on a worker thread (`request_file_dialog`) so the UI —
/// viewport, tracking preview, autosave — keeps running while the OS
/// picker is up; the chosen path is dispatched by
/// [`GuiApp::poll_file_dialog`] on the next frame.
pub(crate) enum FileDialogPurpose {
    OpenProject,
    SaveProjectAs,
    OpenOverlay,
    LoadAvatar,
}

pub(crate) struct PendingFileDialog {
    purpose: FileDialogPurpose,
    rx: mpsc::Receiver<Option<PathBuf>>,
}

fn request_file_dialog(
    state: &mut GuiApp,
    purpose: FileDialogPurpose,
    show: impl FnOnce() -> Option<PathBuf> + Send + 'static,
) {
    // One dialog at a time: a second click while the OS picker is
    // already up would spawn a competing picker behind it.
    if state.pending_file_dialog.is_some() {
        return;
    }
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(show());
    });
    state.pending_file_dialog = Some(PendingFileDialog { purpose, rx });
}

pub(super) fn request_open_project_dialog(state: &mut GuiApp) {
    let filter_label = t!("top_bar.filter_project");
    request_file_dialog(state, FileDialogPurpose::OpenProject, move || {
        rfd::FileDialog::new()
            .add_filter(filter_label, &["vvtproj"])
            .pick_file()
    });
}

pub(super) fn request_save_project_as_dialog(state: &mut GuiApp) {
    let filter_label = t!("top_bar.filter_project");
    request_file_dialog(state, FileDialogPurpose::SaveProjectAs, move || {
        rfd::FileDialog::new()
            .add_filter(filter_label, &["vvtproj"])
            .set_file_name("project.vvtproj")
            .save_file()
    });
}

pub(super) fn request_open_overlay_dialog(state: &mut GuiApp) {
    let filter_label = t!("top_bar.filter_cloth");
    request_file_dialog(state, FileDialogPurpose::OpenOverlay, move || {
        rfd::FileDialog::new()
            .add_filter(filter_label, &["vvtcloth"])
            .pick_file()
    });
}

pub(super) fn request_load_avatar_dialog(state: &mut GuiApp) {
    let filter_label = t!("top_bar.filter_vrm");
    request_file_dialog(state, FileDialogPurpose::LoadAvatar, move || {
        rfd::FileDialog::new()
            .add_filter(filter_label, &["vrm", "fbx"])
            .add_filter("VRM (*.vrm)", &["vrm"])
            .add_filter("FBX (*.fbx)", &["fbx"])
            .pick_file()
    });
}

impl GuiApp {
    /// Drain a finished file-dialog worker and dispatch its result.
    /// Called once per frame from `update()`.
    pub(super) fn poll_file_dialog(&mut self) {
        let Some(pending) = self.pending_file_dialog.as_ref() else {
            return;
        };
        let path = match pending.rx.try_recv() {
            Ok(p) => p,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => None,
        };
        let pending = self
            .pending_file_dialog
            .take()
            .expect("pending dialog present");
        let Some(path) = path else {
            return; // user cancelled
        };
        match pending.purpose {
            FileDialogPurpose::OpenProject => open_project_from_path(self, &path),
            FileDialogPurpose::SaveProjectAs => save_project_to_path(self, &path),
            FileDialogPurpose::OpenOverlay => open_overlay_from_path(self, &path),
            FileDialogPurpose::LoadAvatar => load_avatar_from_path(self, &path),
        }
    }
}

fn open_project(state: &mut GuiApp) {
    request_open_project_dialog(state);
}

pub(super) fn open_project_from_path(state: &mut GuiApp, path: &Path) {
    // Prefer the unsaved-changes sidecar when it is newer than the
    // project file — the previous session ended without an explicit
    // Save and its autosaved state would otherwise silently vanish.
    let sidecar = crate::gui::project::unsaved_sidecar_path(path);
    let (load_path, restore_unsaved) =
        if crate::gui::project::sidecar_is_newer(path, &sidecar) {
            (sidecar, true)
        } else {
            (path.to_path_buf(), false)
        };
    match persistence::load_project(&load_path) {
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
                            state.push_error_notification(t!(
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
                .filter(|p| p.exists())
                // Already showing this exact avatar (e.g. re-opening the
                // project that captured the current scene) — skip the
                // redundant multi-second reload and apply state directly.
                .filter(|p| {
                    state
                        .app
                        .active_avatar()
                        .is_none_or(|a| a.asset.source_path != *p)
                });
            if let Some(missing) = project_state
                .avatar_source_path
                .as_ref()
                .filter(|p| !std::path::Path::new(p).exists())
            {
                state.push_warning_notification(t!(
                    "top_bar.avatar_not_found",
                    path = missing.to_string()
                ));
            }

            state.remember_last_project(path);
            if let Some(avatar_path) = avatar_to_load {
                if state.library.avatar_load_job.is_some() {
                    state.push_warning_notification(t!("top_bar.avatar_load_in_progress"));
                } else {
                    state.push_notification(t!(
                        "top_bar.loading_project",
                        path = path.display().to_string()
                    ));
                    state.library.avatar_load_job = Some(AvatarLoadJob::spawn(
                        avatar_path,
                        AfterLoad::ApplyProject {
                            project_state: Box::new(project_state),
                            project_path: Some(path.to_path_buf()),
                            warnings: load_warnings,
                            restore_unsaved,
                        },
                    ));
                }
            } else {
                state.apply_project_state(&project_state);
                state.project_status.project_path = Some(path.to_path_buf());
                state.mark_project_baseline();
                state.project_status.explicit_file_stale = restore_unsaved;
                for w in &load_warnings.warnings {
                    state.push_warning_notification(t!("toast.warning", msg = w.to_string()));
                }
                if restore_unsaved {
                    state.push_notification(t!("toast.restored_unsaved_changes"));
                }
                state.push_success_notification(t!(
                    "top_bar.opened_project",
                    path = path.display().to_string()
                ));
            }
        }
        Err(e) => {
            state.push_error_notification(t!("top_bar.failed_load_project", error = e.to_string()));
        }
    }
}

/// Explicit save. With a project path, writes it directly; without
/// one, falls through to the Save As dialog. Ctrl+S and the toolbar
/// button both route here so the two entrances behave identically.
pub(super) fn save_project(state: &mut GuiApp) {
    if let Some(path) = state.project_status.project_path.clone() {
        save_project_to_path(state, &path);
    } else {
        request_save_project_as_dialog(state);
    }
}

fn save_project_to_path(state: &mut GuiApp, path: &Path) {
    let ps = state.to_project_state();
    match persistence::save_project(&ps, path) {
        Ok(()) => {
            state.remember_last_project(path);
            state.project_status.project_path = Some(path.to_path_buf());
            // Baseline = what we just wrote (== current state), so the
            // next probe keeps the project clean until a real edit.
            state.mark_project_baseline();
            state.project_status.explicit_file_stale = false;
            // The real file is current again — the unsaved sidecar has
            // nothing to add and must not resurrect on next launch.
            crate::gui::project::clear_unsaved_sidecar(path);
            state.push_success_notification(t!("top_bar.project_saved"));
        }
        Err(e) => {
            state.push_error_notification(t!("top_bar.save_failed", error = e.to_string()));
        }
    }
}

fn open_overlay(state: &mut GuiApp) {
    request_open_overlay_dialog(state);
}

pub(super) fn open_overlay_from_path(state: &mut GuiApp, path: &Path) {
    match persistence::load_cloth_overlay(path) {
        Ok(overlay) => {
            if let Some(cloth_asset) = overlay.cloth_asset {
                state.app.editor.overlay_asset = Some(cloth_asset);
                state.app.editor.dirty = false;
            }
            state.app.editor.overlay_path = Some(path.to_path_buf());
            state.push_success_notification(t!(
                "top_bar.opened_overlay",
                name = overlay.overlay_name.to_string(),
                version = overlay.format_version
            ));
        }
        Err(e) => {
            state.push_error_notification(t!("top_bar.failed_load_cloth", error = e.to_string()));
        }
    }
}

fn save_overlay(state: &mut GuiApp) {
    match state.app.editor.save_overlay(None) {
        Ok(()) => state.push_success_notification(t!("top_bar.overlay_saved")),
        Err(e) => state.push_error_notification(t!(
            "top_bar.save_overlay_failed",
            error = e.to_string()
        )),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Profile management (dialogs + switch)
// ─────────────────────────────────────────────────────────────────────

/// Modal state for the profile-management dialogs opened from the
/// top-bar combo. Held on `GuiApp::profile_dialog`; drawn each frame
/// by [`draw_profile_dialogs`].
pub(crate) enum ProfileDialog {
    New { name: String },
    Rename { index: usize, name: String },
    Delete { index: usize },
}

pub(crate) fn switch_to_profile(state: &mut GuiApp, index: usize) {
    if let Some(profile) = state.profiles.profiles.get(index).cloned() {
        let name = profile.name.clone();
        state.profiles.set_active(index);
        state.apply_profile(&profile);
        state.push_notification(t!("top_bar.switched_profile", name = name));
    }
}

/// Draw the profile New/Rename/Delete dialogs plus the
/// calibration-loss switch confirmation. Called once per frame from
/// `GuiApp::update` alongside the other modal dialogs.
pub(crate) fn draw_profile_dialogs(ctx: &egui::Context, state: &mut GuiApp) {
    // ── Switch-with-calibration-loss confirmation ─────────────────
    if let Some(target) = state.pending_profile_switch {
        let target_name = state
            .profiles
            .profiles
            .get(target)
            .map(|p| p.name.clone())
            .unwrap_or_default();
        let mut decision: Option<bool> = None;
        egui::Window::new(t!("dialog.profile_switch_title"))
            .id(egui::Id::new("profile_switch_confirm"))
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .collapsible(false)
            .resizable(false)
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                ui.label(t!("dialog.profile_switch_body", name = target_name.clone()));
                ui.add_space(space::SM);
                ui.horizontal(|ui| {
                    if ui.button(t!("dialog.profile_switch_confirm")).clicked() {
                        decision = Some(true);
                    }
                    if ui.button(t!("calibration.cancel")).clicked() {
                        decision = Some(false);
                    }
                });
            });
        match decision {
            Some(true) => {
                state.pending_profile_switch = None;
                switch_to_profile(state, target);
            }
            Some(false) => state.pending_profile_switch = None,
            None => {}
        }
    }

    let Some(dialog) = state.profile_dialog.take() else {
        return;
    };
    let mut keep: Option<ProfileDialog> = None;
    match dialog {
        ProfileDialog::New { mut name } => {
            let mut done = false;
            egui::Window::new(t!("top_bar.profile_new"))
                .id(egui::Id::new("profile_new_dialog"))
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .collapsible(false)
                .resizable(false)
                .order(egui::Order::Foreground)
                .show(ctx, |ui| {
                    ui.label(t!("top_bar.profile_name_label"));
                    ui.text_edit_singleline(&mut name);
                    ui.add_space(space::SM);
                    ui.horizontal(|ui| {
                        let trimmed = name.trim();
                        if ui
                            .add_enabled(
                                !trimmed.is_empty(),
                                egui::Button::new(t!("top_bar.profile_create")),
                            )
                            .clicked()
                        {
                            let unique = state.profiles.unique_name(trimmed);
                            let mut profile =
                                crate::gui::profile::StreamProfile::streaming_default();
                            profile.name = unique.clone();
                            profile.pose_calibration = None;
                            state.profiles.add(profile);
                            state.project_status.profiles_dirty = true;
                            state.push_success_notification(t!(
                                "top_bar.profile_created",
                                name = unique
                            ));
                            done = true;
                        }
                        if ui.button(t!("calibration.cancel")).clicked() {
                            done = true;
                        }
                    });
                });
            if !done {
                keep = Some(ProfileDialog::New { name });
            }
        }
        ProfileDialog::Rename { index, mut name } => {
            let mut done = false;
            egui::Window::new(t!("top_bar.profile_rename"))
                .id(egui::Id::new("profile_rename_dialog"))
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .collapsible(false)
                .resizable(false)
                .order(egui::Order::Foreground)
                .show(ctx, |ui| {
                    ui.label(t!("top_bar.profile_name_label"));
                    ui.text_edit_singleline(&mut name);
                    ui.add_space(space::SM);
                    ui.horizontal(|ui| {
                        if ui.button(t!("top_bar.profile_rename_apply")).clicked() {
                            match state.profiles.rename_at(index, &name) {
                                Ok(()) => {
                                    state.project_status.profiles_dirty = true;
                                    done = true;
                                }
                                Err(crate::gui::profile::RenameError::Empty) => {
                                    state.push_warning_notification(t!(
                                        "top_bar.profile_name_empty"
                                    ));
                                }
                                Err(crate::gui::profile::RenameError::Duplicate) => {
                                    state.push_warning_notification(t!(
                                        "top_bar.profile_name_taken"
                                    ));
                                }
                                Err(crate::gui::profile::RenameError::OutOfBounds) => {
                                    done = true;
                                }
                            }
                        }
                        if ui.button(t!("calibration.cancel")).clicked() {
                            done = true;
                        }
                    });
                });
            if !done {
                keep = Some(ProfileDialog::Rename { index, name });
            }
        }
        ProfileDialog::Delete { index } => {
            let name = state
                .profiles
                .profiles
                .get(index)
                .map(|p| p.name.clone())
                .unwrap_or_default();
            let mut done = false;
            egui::Window::new(t!("top_bar.profile_delete"))
                .id(egui::Id::new("profile_delete_dialog"))
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .collapsible(false)
                .resizable(false)
                .order(egui::Order::Foreground)
                .show(ctx, |ui| {
                    ui.label(t!("top_bar.profile_delete_body", name = name.clone()));
                    ui.add_space(space::SM);
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(
                                state.profiles.can_remove(),
                                egui::Button::new(t!("top_bar.profile_delete_confirm")),
                            )
                            .clicked()
                        {
                            let was_active =
                                state.profiles.active_index == Some(index);
                            state.profiles.remove(index);
                            state.project_status.profiles_dirty = true;
                            state.push_notification(t!(
                                "top_bar.profile_deleted",
                                name = name.clone()
                            ));
                            // Deleting the active profile lands on a
                            // neighbour — make its settings live so the
                            // UI and the solver agree on which profile
                            // is now driving.
                            if was_active {
                                if let Some(i) = state.profiles.active_index {
                                    switch_to_profile(state, i);
                                }
                            }
                            done = true;
                        }
                        if ui.button(t!("calibration.cancel")).clicked() {
                            done = true;
                        }
                    });
                });
            if !done {
                keep = Some(ProfileDialog::Delete { index });
            }
        }
    }
    state.profile_dialog = keep;
}

#[cfg(test)]
mod topbar_layout_tests {
    use super::*;

    // Display order: [avatar, camera, calibrate, pause] — see the
    // ACTION_* constants.
    const W: [f32; 4] = [120.0, 100.0, 90.0, 80.0];

    #[test]
    fn everything_visible_when_the_bar_is_wide_enough() {
        let vis = fit_topbar_actions(1000.0, &W);
        assert_eq!(vis, vec![true, true, true, true]);
    }

    #[test]
    fn pause_collapses_first_then_calibrate() {
        // Wide enough for all but pause.
        let vis = fit_topbar_actions(330.0, &W);
        assert_eq!(vis, vec![true, true, true, false], "pause drops first");
        // Only avatar + camera fit.
        let vis = fit_topbar_actions(230.0, &W);
        assert_eq!(
            vis,
            vec![true, true, false, false],
            "calibrate drops second"
        );
    }

    #[test]
    fn session_critical_actions_never_collapse() {
        let vis = fit_topbar_actions(0.0, &W);
        assert!(
            vis[ACTION_AVATAR] && vis[ACTION_CAMERA],
            "avatar picker and camera toggle must survive any width — \
             they are the app's mandatory first steps"
        );
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
        color::with_alpha(color::PRIMARY, 18)
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
        color::with_alpha(color::PRIMARY, 18)
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
