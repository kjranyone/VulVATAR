//! egui rendering for the calibration modal: the entry point
//! [`draw_modal`], the per-state status panes, the camera preview
//! pane, and the small view-formatting helpers (`step_text`,
//! `progress_for`, `telemetry`, `mode_mismatch_hint`).
//!
//! This is the only module under `calibration/` that depends on
//! `egui` types. The rest of the module (state, transitions,
//! finalize, refresh, pose_match) is pure logic and can be exercised
//! without bringing the GUI stack up.

use eframe::egui;

use crate::gui::components::status_dot_label;
use crate::gui::theme::{color, space, typography, viz};
use crate::gui::GuiApp;
use crate::t;
use crate::tracking::CalibrationMode;

use super::pose_match::REQUIRED_STABLE_FRAMES;
use super::refresh::refresh_anchor_telemetry;
use super::state::{relevant_mode, CalibrationModalState, DoneOutcome};
use super::transitions::{
    advance_state, begin_capture, begin_range_capture, finish_capture, retry_capture, skip_range,
};
use super::{
    COLLECTION_SECONDS, MIN_SAMPLES, NO_ANCHOR_HINT_SECONDS, RANGE_COLLECTION_SECONDS,
    RANGE_HOLD_STILL_SECONDS,
};

/// Per-frame entry point. No-op when the modal is closed; otherwise
/// advances the state machine, paints the dim overlay + centred
/// `Window`, and handles the Cancel / Capture Now buttons.
pub fn draw_modal(ctx: &egui::Context, state: &mut GuiApp) {
    if !state.calibration.modal.is_open() {
        return;
    }

    // Esc dismisses the modal at any stage. Without this the only
    // exit during WaitingForPose / Collecting / RangeHoldStill /
    // RangeCollecting was the Cancel button — which goes off-screen
    // if the user happens to collapse the inspector mid-capture, and
    // is generally an accessibility miss for keyboard-only users.
    // Done no longer auto-closes (the user reads the result summary
    // and clicks Close), so Esc works there too.
    if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
        state.calibration.modal.close();
        return;
    }

    // Tick the state machine forward.
    advance_state(state);
    // Drain any in-flight target-pose snapshot from the render thread
    // and upload to the modal's egui texture handle. No-op until the
    // render finishes (typically 1 frame after the kick request).
    state.poll_calibration_target_pose_snapshot(ctx);
    // Kick a fresh target-pose snapshot whenever the modal-relevant
    // mode changes (or the first time the modal opens). The kick is
    // idempotent (same-mode no-op) so it's safe to call every frame.
    if let Some(mode) = relevant_mode(&state.calibration.modal) {
        state.kick_calibration_target_pose_snapshot(mode);
    }
    // Done deliberately does NOT auto-close: it now carries the
    // result summary (shoulder span, range, face-neutral status) and
    // a 1.5 s linger gave the user no chance to read it — worse, the
    // only visible button was labelled "Cancel", which read as
    // "discard the calibration I just took".

    // Dim the entire viewport so background widgets read as inactive.
    // Placed on `Order::Middle` (the default for Windows) so the
    // calibration `Window` below — explicitly forced to
    // `Order::Foreground` — paints strictly *above* it. Painting both
    // on the same Order leaves the in-Order z-position dependent on
    // insertion / interaction history, which produced the modal
    // landing *behind* the scrim under some open paths.
    let screen_rect = ctx.screen_rect();
    let dim_layer = egui::LayerId::new(egui::Order::Middle, egui::Id::new("calibration_dim"));
    let dim_painter = ctx.layer_painter(dim_layer);
    dim_painter.rect_filled(
        screen_rect,
        0.0,
        egui::Color32::from_rgba_unmultiplied(0, 0, 0, 180),
    );

    // Centred modal window. Disabling collapse / resize / movement so
    // the user can only interact via the buttons we provide.
    // Pin the window's egui id explicitly: by default `Window::new(title)`
    // hashes the title into the id-source, so any future title that
    // mutates between frames (e.g. mid-modal mode swap) would reset
    // window state. The `modal_title` helper happens to return a
    // constant today — pin the id so a future title rewrite can't
    // silently regress that.
    egui::Window::new(modal_title(&state.calibration.modal))
        .id(egui::Id::new("calibration_modal_window"))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .collapsible(false)
        .resizable(false)
        .movable(false)
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            // Wider min-size now that the modal carries three columns
            // (camera preview + target-pose snapshot + status pane)
            // instead of the original two. Width = camera (480) +
            // target (240) + status (220) + paddings (≈ 32).
            ui.set_min_size(egui::vec2(972.0, 420.0));
            ui.horizontal_top(|ui| {
                draw_preview_pane(ui, state);
                ui.add_space(space::MD);
                draw_target_pose_pane(ui, state);
                ui.add_space(space::MD);
                draw_status_pane(ui, state);
            });
        });
}

/// Target-pose reference pane (centre of the modal, between the
/// camera preview on the left and the status pane on the right).
/// Renders the `calibration_target_pose_texture` snapshot if it's
/// arrived from the render thread, or a "rendering…" placeholder
/// while the kick is in flight. The snapshot is one-shot per mode —
/// the kick happens on modal open and on segmented-buttons mode
/// changes (see the `kick_calibration_target_pose_snapshot` call in
/// `draw_modal`).
fn draw_target_pose_pane(ui: &mut egui::Ui, state: &mut GuiApp) {
    let preview_size = egui::vec2(240.0, 270.0);
    let (rect, _) = ui.allocate_exact_size(preview_size, egui::Sense::hover());

    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 6.0, color::VIEWPORT_BG);
    painter.rect_stroke(
        rect.shrink(1.0),
        6.0,
        egui::Stroke::new(2.0, color::VIEWPORT_OVERLAY_OUTLINE),
    );

    if let Some(ref tex) = state.calibration.target_pose_texture {
        painter.image(
            tex.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
    } else {
        // Render-pending placeholder. The kick is idempotent so it
        // re-fires every frame until the texture arrives; the
        // surface render thread may take 1–2 frames after the kick
        // before the response lands.
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            t!("calibration.target_pose_loading"),
            typography::body(),
            viz::OVERLAY_TEXT,
        );
    }

    // Caption underneath the rect indicating this is the *target*
    // pose, not the user's live tracking.
    ui.label(
        egui::RichText::new(t!("calibration.target_pose_caption"))
            .font(typography::caption())
            .color(color::ON_SURFACE_MUTED),
    );
}

/// Modal window title. Stable across mode switches — the active mode
/// is conveyed by the Segmented Buttons control inside the status pane,
/// so duplicating it in the window header would just produce the
/// "every label says the same thing" Material Design anti-pattern.
///
/// Stable also keeps egui happy: the `Window::new(title)` value is
/// hashed into the window's id-source, so a title that mutates between
/// frames (e.g. on a mid-modal mode swap) would reset window state.
fn modal_title(state: &CalibrationModalState) -> String {
    if matches!(state, CalibrationModalState::Closed) {
        return String::new();
    }
    t!("calibration.title").to_string()
}

/// Webcam preview pane (left side of the modal). Reuses the same
/// camera-wipe texture-upload pattern from `gui::viewport` so the
/// frame data path doesn't fork; the difference is just a larger
/// rect and the absence of the PIP corner styling.
fn draw_preview_pane(ui: &mut egui::Ui, state: &mut GuiApp) {
    let preview_size = egui::vec2(480.0, 270.0);
    let (rect, _) = ui.allocate_exact_size(preview_size, egui::Sense::hover());

    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 6.0, color::VIEWPORT_BG);
    // Shrink by half the 2 px stroke width so the outline lands fully
    // inside the painter's clip rect (otherwise its outer 1 px is
    // clipped, leaving a hairline-thin / missing border).
    painter.rect_stroke(
        rect.shrink(1.0),
        6.0,
        egui::Stroke::new(2.0, color::VIEWPORT_OVERLAY_OUTLINE),
    );

    // Refresh texture on new preview sequence — same path the
    // camera-wipe uses (and same reason: pose `sequence` can advance
    // mid-snapshot while the frame is still the previous publish,
    // which would permanently strand a frame if we deduped on it).
    // Sharing the buffer / handle would couple the two previews;
    // keep them separate so toggling camera-wipe while calibrating
    // doesn't tear either preview. The cheap `preview_sequence()`
    // probe runs first so the full snapshot (frame refcount + pose
    // clone) is only pulled when a new publish actually landed.
    if state.app.tracking.mailbox().preview_sequence() != state.calibration.preview_seq {
        let snap = state.app.tracking.mailbox().snapshot();
        if let Some(ref frame) = snap.frame {
            let w = frame.width as usize;
            let h = frame.height as usize;
            if w > 0 && h > 0 && frame.rgb_data.len() == w * h * 3 {
                let needed = w * h * 4;
                if state.calibration.preview_rgba_buf.len() != needed {
                    state.calibration.preview_rgba_buf.resize(needed, 255);
                }
                let rgba = &mut state.calibration.preview_rgba_buf;
                for i in 0..w * h {
                    rgba[i * 4] = frame.rgb_data[i * 3];
                    rgba[i * 4 + 1] = frame.rgb_data[i * 3 + 1];
                    rgba[i * 4 + 2] = frame.rgb_data[i * 3 + 2];
                    rgba[i * 4 + 3] = 255;
                }
                let color_image = egui::ColorImage::from_rgba_unmultiplied([w, h], rgba);
                let options = egui::TextureOptions {
                    magnification: egui::TextureFilter::Linear,
                    minification: egui::TextureFilter::Linear,
                    ..Default::default()
                };
                if let Some(ref mut handle) = state.calibration.preview_texture {
                    handle.set(color_image, options);
                } else {
                    let handle = ui
                        .ctx()
                        .load_texture("calibration_preview", color_image, options);
                    state.calibration.preview_texture = Some(handle);
                }
            }
        }
        // Annotation + telemetry ride the same freshness gate; cached
        // so the GUI ticks between publishes redraw without another
        // snapshot.
        state.calibration.preview_annotation = snap.annotation.clone();
        refresh_anchor_telemetry(state, &snap);
        state.calibration.preview_seq = snap.preview_sequence;
    }

    if state.calibration.preview_texture.is_none() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            t!("calibration.no_frame"),
            egui::FontId::proportional(14.0),
            viz::OVERLAY_TEXT,
        );
        return;
    }

    let Some(ref tex) = state.calibration.preview_texture else {
        return;
    };

    // Letterbox/pillarbox the camera frame into the fixed preview rect
    // so 4:3 sources (the default 640×480) don't get stretched into
    // the 16:9 outer rect. Mirrors the fit-inside logic in
    // `viewport::draw` for the main viewport image.
    let tex_size = tex.size_vec2();
    let tex_aspect = tex_size.x / tex_size.y.max(1.0);
    let rect_aspect = rect.width() / rect.height().max(1.0);
    let (draw_w, draw_h) = if tex_aspect > rect_aspect {
        (rect.width(), rect.width() / tex_aspect)
    } else {
        (rect.height() * tex_aspect, rect.height())
    };
    let image_rect = egui::Rect::from_center_size(rect.center(), egui::vec2(draw_w, draw_h));

    let uv = if state.tracking.tracking_mirror {
        egui::Rect::from_min_max(egui::pos2(1.0, 0.0), egui::pos2(0.0, 1.0))
    } else {
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0))
    };
    painter.image(tex.id(), image_rect, uv, egui::Color32::WHITE);

    // Skeleton + keypoint overlay. Same style the camera-wipe paints —
    // the calibration UX is most useful when the user can see exactly
    // which keypoints the tracker is locking onto. Coordinates map
    // into the letterboxed image rect (not the outer pane rect) so
    // overlay points sit on top of the actual camera pixels even
    // when the image is pillarboxed.
    if let Some(ref ann) = state.calibration.preview_annotation {
        let kpt_color = viz::keypoint();
        let line_color = viz::bone();
        let mirror = state.tracking.tracking_mirror;
        let map_x = |nx: f32| {
            let x = if mirror { 1.0 - nx } else { nx };
            image_rect.left() + x * image_rect.width()
        };
        let map_y = |ny: f32| image_rect.top() + ny * image_rect.height();

        for &(kx, ky, conf) in &ann.keypoints {
            if conf < 0.1 {
                continue;
            }
            painter.circle_filled(egui::pos2(map_x(kx), map_y(ky)), 3.0, kpt_color);
        }
        for &(a, b) in &ann.skeleton {
            let ka = ann.keypoints.get(a);
            let kb = ann.keypoints.get(b);
            if let (Some(&(ax, ay, ac)), Some(&(bx, by, bc))) = (ka, kb) {
                if ac >= 0.1 && bc >= 0.1 {
                    painter.line_segment(
                        [
                            egui::pos2(map_x(ax), map_y(ay)),
                            egui::pos2(map_x(bx), map_y(by)),
                        ],
                        egui::Stroke::new(1.5, line_color),
                    );
                }
            }
        }
    }
}

/// Status pane (right side of the modal). Branches on the
/// state-machine variant and only renders the widgets that make
/// sense for that step — earlier the pane drew every widget
/// (mode picker, progress bar, telemetry, every action button) at
/// once and disabled the irrelevant ones, which made the modal
/// feel like a cluttered control panel rather than a guided flow.
fn draw_status_pane(ui: &mut egui::Ui, state: &mut GuiApp) {
    ui.vertical(|ui| {
        ui.set_min_width(220.0);
        match &state.calibration.modal {
            CalibrationModalState::Idle { .. } => draw_idle_pane(ui, state),
            CalibrationModalState::WaitingForPose { .. }
            | CalibrationModalState::Collecting { .. }
            | CalibrationModalState::RangeHoldStill { .. }
            | CalibrationModalState::RangeCollecting { .. } => draw_capturing_pane(ui, state),
            CalibrationModalState::AnchorDone { .. } => draw_anchor_done_pane(ui, state),
            CalibrationModalState::Done { .. } => draw_done_pane(ui, state),
            CalibrationModalState::Closed => {}
        }
    });
}

/// Idle pane: mode picker + instructions + framing telemetry +
/// Cancel/Start. No progress bar, no sample counter — capture has
/// not started yet.
fn draw_idle_pane(ui: &mut egui::Ui, state: &mut GuiApp) {
    if let Some(current_mode) = relevant_mode(&state.calibration.modal) {
        ui.label(egui::RichText::new(t!("calibration.mode_label")).font(typography::label()));
        ui.add_space(space::XS);
        ui.horizontal(|ui| {
            if ui
                .selectable_label(
                    current_mode == CalibrationMode::FullBody,
                    t!("calibration.mode_full_body"),
                )
                .clicked()
            {
                state.calibration.modal.set_mode(CalibrationMode::FullBody);
            }
            if ui
                .selectable_label(
                    current_mode == CalibrationMode::UpperBody,
                    t!("calibration.mode_upper_body"),
                )
                .clicked()
            {
                state.calibration.modal.set_mode(CalibrationMode::UpperBody);
            }
        });
        ui.add_space(space::MD);
    }

    let (step_label, instructions) = step_text(&state.calibration.modal);
    ui.label(
        egui::RichText::new(step_label)
            .font(typography::body())
            .strong(),
    );
    ui.add_space(space::XS);
    ui.label(egui::RichText::new(instructions).font(typography::label()));
    ui.add_space(space::MD);

    // Live framing check: lets the user verify the camera is seeing
    // the right anchor before they commit to the countdown.
    draw_live_telemetry(ui, &state.calibration.modal, false);

    ui.add_space(space::MD);

    ui.horizontal(|ui| {
        if ui.button(t!("calibration.cancel")).clicked() {
            state.calibration.modal.close();
        }
        if ui.button(t!("calibration.start")).clicked() {
            begin_capture(state);
        }
    });
}

/// Active-capture pane: covers WaitingForPose / Collecting /
/// RangeHoldStill / RangeCollecting. Each of these states wants the
/// same layout (instructions → progress → telemetry → buttons), and
/// `step_text` / `progress_for` / `telemetry` already specialise on
/// the variant. WaitingForPose adds an optional mode-mismatch hint
/// when the relevant anchor's been missing for `NO_ANCHOR_HINT_SECONDS`,
/// surfacing a "switch mode" button so a wrong-mode user can recover
/// without cancelling.
fn draw_capturing_pane(ui: &mut egui::Ui, state: &mut GuiApp) {
    let (step_label, instructions) = step_text(&state.calibration.modal);
    ui.label(
        egui::RichText::new(step_label)
            .font(typography::body())
            .strong(),
    );
    ui.add_space(space::XS);
    ui.label(egui::RichText::new(instructions).font(typography::label()));
    ui.add_space(space::MD);

    let (progress, time_left) = progress_for(&state.calibration.modal);
    ui.add(
        egui::ProgressBar::new(progress)
            .desired_width(220.0)
            .text(time_left),
    );
    ui.add_space(space::MD);

    draw_live_telemetry(ui, &state.calibration.modal, true);

    // Mode-mismatch hint. Only fires in WaitingForPose (= the user
    // is trying to take the pose but the required anchor isn't in
    // frame). Surfaces after NO_ANCHOR_HINT_SECONDS so a transient
    // dropout doesn't ping the user, but a sustained "wrong mode"
    // misconfiguration shows up and the user gets a one-click escape.
    let hint = mode_mismatch_hint(&state.calibration.modal);
    if let Some((message_key, switch_to)) = hint {
        ui.add_space(space::SM);
        ui.colored_label(
            color::WARNING,
            egui::RichText::new(t!(message_key)).font(typography::label()),
        );
        let label = match switch_to {
            CalibrationMode::FullBody => t!("calibration.switch_mode_full_body"),
            CalibrationMode::UpperBody => t!("calibration.switch_mode_upper_body"),
        };
        if ui.button(label).clicked() {
            // Drop back to Idle with the new mode pre-selected. The
            // user re-clicks Start to begin capture in the new mode.
            // Going back to Idle (rather than re-entering
            // WaitingForPose directly) gives the user a chance to
            // re-read the new mode's instructions before the gate
            // starts watching for a match.
            //
            state.calibration.modal = CalibrationModalState::Idle {
                mode: switch_to,
                last_anchor_seen: false,
                last_confidence: 0.0,
            };
        }
    } else if let Some(key) = framing_hint(&state.calibration.modal) {
        // Anchor is fine but the elbows have been cropped for a
        // while. In FullBody the fix is physical (step back so a
        // T-pose fits the frame) and the hint says so; in UpperBody
        // the stillness fallback has taken over by the time this
        // fires and the hint instead explains the gate switch.
        // Inline-only — no auxiliary button either way.
        ui.add_space(space::SM);
        ui.colored_label(
            color::WARNING,
            egui::RichText::new(t!(key)).font(typography::label()),
        );
    }

    ui.add_space(space::MD);

    ui.horizontal(|ui| {
        if ui.button(t!("calibration.cancel")).clicked() {
            // without this, a partial buffer would silently leak
            // (whose seq edge-detect would surface stale
            state.calibration.modal.close();
        }
        if ui.button(t!("calibration.capture_now")).clicked() {
            finish_capture(state);
        }
    });
}

/// Returns `(hint_message_key, mode_to_switch_to)` if WaitingForPose
/// has gone NO_ANCHOR_HINT_SECONDS without the relevant anchor
/// visible. `None` otherwise. Pure-read on the modal state so the
/// hint rendering can decide whether to draw the row without mutating
/// anything.
fn mode_mismatch_hint(state: &CalibrationModalState) -> Option<(&'static str, CalibrationMode)> {
    let CalibrationModalState::WaitingForPose {
        mode,
        no_anchor_since,
        ..
    } = state
    else {
        return None;
    };
    let since = (*no_anchor_since)?;
    if since.elapsed().as_secs_f32() < NO_ANCHOR_HINT_SECONDS {
        return None;
    }
    let (message, switch_to) = match mode {
        CalibrationMode::FullBody => (
            "calibration.no_anchor_hint_full_body",
            CalibrationMode::UpperBody,
        ),
        CalibrationMode::UpperBody => (
            "calibration.no_anchor_hint_upper_body",
            CalibrationMode::FullBody,
        ),
    };
    Some((message, switch_to))
}

/// Locale key for the elbow-framing hint, if it should show. Fires
/// once the user has been in `WaitingForPose` long enough with the
/// elbows cropped, reusing `NO_ANCHOR_HINT_SECONDS` so a transient
/// occlusion of one elbow (a hand passing over the lap, etc.) doesn't
/// immediately ping; the threshold is the same load-bearing "ignore
/// short blips, react to sustained problems" delay the anchor hint
/// uses.
///
/// Two variants: while the bust-up stillness fallback is engaged
/// (UpperBody, elbows permanently cropped — see
/// `docs/calibration-ux.md`, "Bust-up framing fallback") the "step back" advice is
/// unactionable, so the hint instead announces the gate switch. The
/// fallback latch implies the elapsed condition already passed, so
/// it's checked first.
fn framing_hint(state: &CalibrationModalState) -> Option<&'static str> {
    let CalibrationModalState::WaitingForPose {
        no_lower_arms_since,
        stillness_fallback,
        ..
    } = state
    else {
        return None;
    };
    if *stillness_fallback {
        return Some("calibration.stillness_fallback_hint");
    }
    no_lower_arms_since
        .map(|t| t.elapsed().as_secs_f32() >= NO_ANCHOR_HINT_SECONDS)
        .unwrap_or(false)
        .then_some("calibration.lower_arms_out_of_frame")
}

/// AnchorDone pane: result summary + the three branching choices
/// (Retry / Skip & Finish / Capture Range). No live telemetry —
/// capture is finished, the values shown via `step_text` come from
/// the aggregated calibration on the variant.
fn draw_anchor_done_pane(ui: &mut egui::Ui, state: &mut GuiApp) {
    let (step_label, instructions) = step_text(&state.calibration.modal);
    ui.label(
        egui::RichText::new(step_label)
            .font(typography::body())
            .strong(),
    );
    ui.add_space(space::XS);
    ui.label(egui::RichText::new(instructions).font(typography::label()));
    ui.add_space(space::MD);

    ui.horizontal(|ui| {
        if ui.button(t!("calibration.cancel")).clicked() {
            state.calibration.modal.close();
        }
        if ui.button(t!("calibration.retry")).clicked() {
            retry_capture(state);
        }
        if ui.button(t!("calibration.skip_and_finish")).clicked() {
            skip_range(state);
        }
        if ui.button(t!("calibration.capture_range")).clicked() {
            begin_range_capture(state);
        }
    });
}

/// Done pane: success/failure heading plus a summary of what the
/// capture actually produced (shoulder span, movement range,
/// face-neutral status), closed explicitly by the user. The old
/// 1.5 s auto-close + "Cancel" button pairing both hid the result
/// and mislabelled "dismiss a finished calibration" as a cancel.
fn draw_done_pane(ui: &mut egui::Ui, state: &mut GuiApp) {
    let (step_label, instructions) = step_text(&state.calibration.modal);
    ui.label(
        egui::RichText::new(step_label)
            .font(typography::body())
            .strong(),
    );
    ui.add_space(space::XS);
    ui.label(egui::RichText::new(instructions).font(typography::label()));
    ui.add_space(space::SM);

    if let CalibrationModalState::Done {
        outcome: DoneOutcome::Success { calibration, .. },
        ..
    } = &state.calibration.modal
    {
        let span_text = match calibration.shoulder_span_m {
            Some(s) => t!("calibration.summary_span_value", span = format!("{:.2}", s)),
            None => t!("calibration.summary_not_captured"),
        };
        summary_row(
            ui,
            &t!("calibration.summary_span"),
            &span_text,
            calibration.shoulder_span_m.is_some(),
        );

        let has_range =
            calibration.x_range_observed.is_some() || calibration.z_range_observed.is_some();
        let range_text = if has_range {
            t!("calibration.summary_captured")
        } else {
            t!("calibration.summary_skipped")
        };
        summary_row(ui, &t!("calibration.summary_range"), &range_text, has_range);

        let has_face = calibration.neutral_face_ypr_mesh.is_some()
            || calibration.neutral_face_ypr_body.is_some();
        let face_text = if has_face {
            t!("calibration.summary_captured")
        } else {
            t!("calibration.summary_skipped")
        };
        summary_row(ui, &t!("calibration.summary_face"), &face_text, has_face);
    }

    ui.add_space(space::MD);

    ui.horizontal(|ui| {
        if ui.button(t!("calibration.close")).clicked() {
            state.calibration.modal.close();
        }
    });
}

/// One result-summary row: caption label, then a status dot + value.
/// `captured = false` renders the dot muted (a skipped optional step
/// is not an error).
fn summary_row(ui: &mut egui::Ui, label: &str, value: &str, captured: bool) {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(label)
                .font(typography::caption())
                .color(color::ON_SURFACE_VARIANT),
        );
        let dot = if captured {
            color::SUCCESS
        } else {
            color::ON_SURFACE_MUTED
        };
        status_dot_label(ui, dot, value);
    });
}

/// Confidence + anchor-visible row, optionally followed by the
/// sample-count row. `show_samples = false` for Idle (nothing to
/// show yet); `true` for the active-capture states.
fn draw_live_telemetry(ui: &mut egui::Ui, state: &CalibrationModalState, show_samples: bool) {
    let (anchor_seen, conf, samples) = telemetry(state);
    let conf_color = if conf >= 0.5 {
        color::SUCCESS
    } else {
        color::ERROR
    };
    ui.horizontal(|ui| {
        ui.label(t!("calibration.confidence"));
        ui.colored_label(conf_color, format!("{:.2}", conf));
    });
    ui.horizontal(|ui| {
        ui.label(t!("calibration.anchor_visible"));
        if anchor_seen {
            status_dot_label(ui, color::SUCCESS, &t!("calibration.visible_yes"));
        } else {
            status_dot_label(ui, color::ERROR, &t!("calibration.visible_no"));
        }
    });
    if show_samples {
        if let Some(s) = samples {
            ui.horizontal(|ui| {
                ui.label(t!("calibration.samples"));
                ui.label(format!("{}", s));
            });
        }
    }
}

/// `(step heading, instructions body)` for the current state. View-
/// formatting helper used by every per-state pane to render the top
/// of the status column.
fn step_text(state: &CalibrationModalState) -> (String, String) {
    match state {
        CalibrationModalState::Idle { mode, .. } => {
            let pose = match mode {
                CalibrationMode::FullBody => t!("calibration.pose_full_body"),
                CalibrationMode::UpperBody => t!("calibration.pose_upper_body"),
            };
            (t!("calibration.step_idle"), pose)
        }
        CalibrationModalState::WaitingForPose {
            mode,
            frames_at_match,
            stillness_fallback,
            ..
        } => {
            // Once the bust-up fallback has engaged, the arm
            // instructions are unactionable (the arms aren't in
            // frame) — swap to the hold-still instruction the
            // stillness gate actually scores.
            let pose = if *stillness_fallback {
                t!("calibration.pose_upper_body_bust_up")
            } else {
                match mode {
                    CalibrationMode::FullBody => t!("calibration.pose_full_body"),
                    CalibrationMode::UpperBody => t!("calibration.pose_upper_body"),
                }
            };
            // Step text differs once the user has crossed the match
            // threshold: the system has "locked on" and is waiting
            // for stability before transitioning to Collecting. Gives
            // the user feedback that they nailed the pose. The
            // pre-lock heading also branches on the fallback —
            // "get into the pose" would contradict the body text's
            // "no pose needed, just hold still".
            let step = if *frames_at_match > 0 {
                t!("calibration.step_pose_locked")
            } else if *stillness_fallback {
                t!("calibration.step_hold_still")
            } else {
                t!("calibration.step_waiting_for_pose")
            };
            (step, pose)
        }
        CalibrationModalState::Collecting {
            mode,
            stillness_fallback,
            ..
        } => {
            // Same body swap as WaitingForPose: a bust-up capture
            // must keep saying "hold still" through the collection
            // window instead of reverting to arm instructions the
            // user can't see themselves follow.
            let pose = if *stillness_fallback {
                t!("calibration.pose_upper_body_bust_up")
            } else {
                match mode {
                    CalibrationMode::FullBody => t!("calibration.pose_full_body"),
                    CalibrationMode::UpperBody => t!("calibration.pose_upper_body"),
                }
            };
            (t!("calibration.step_collecting"), pose)
        }
        CalibrationModalState::AnchorDone { calibration, .. } => {
            // Surface what the anchor capture actually produced so
            // the user has context before deciding on the optional
            // range step. Depth gets the same with/without-depth
            // split as the final Done variant.
            let body = match calibration.anchor_depth_m {
                Some(d) => t!(
                    "calibration.anchor_done_body_with_depth",
                    frames = format!("{}", calibration.frame_count),
                    depth = format!("{:.2}", d)
                ),
                None => t!(
                    "calibration.anchor_done_body_no_depth",
                    frames = format!("{}", calibration.frame_count)
                ),
            };
            (t!("calibration.anchor_done_title"), body)
        }
        CalibrationModalState::RangeHoldStill { .. } => (
            t!("calibration.range_hold_still"),
            t!("calibration.range_pose"),
        ),
        CalibrationModalState::RangeCollecting { .. } => (
            t!("calibration.range_collecting"),
            t!("calibration.range_pose"),
        ),
        CalibrationModalState::Done { outcome, .. } => match outcome {
            DoneOutcome::Success {
                frame_count,
                calibration,
            } => {
                let body = match calibration.anchor_depth_m {
                    Some(d) => t!(
                        "calibration.step_done_body_with_depth",
                        frames = format!("{}", frame_count),
                        depth = format!("{:.2}", d)
                    ),
                    None => t!(
                        "calibration.step_done_body_no_depth",
                        frames = format!("{}", frame_count)
                    ),
                };
                (t!("calibration.step_done"), body)
            }
            DoneOutcome::Insufficient { samples_collected } => (
                t!("calibration.step_failed"),
                t!(
                    "calibration.step_failed_body",
                    samples = format!("{}", samples_collected),
                    minimum = format!("{}", MIN_SAMPLES)
                ),
            ),
        },
        CalibrationModalState::Closed => (String::new(), String::new()),
    }
}

/// `(progress 0..=1, status label)` for the progress bar. Each state
/// has its own progression semantics — pose-match score for
/// WaitingForPose, elapsed-vs-window for the timed states.
fn progress_for(state: &CalibrationModalState) -> (f32, String) {
    match state {
        // Idle has no time-driven progress — the user controls the
        // transition with the Start button. Returning (0, "") here
        // keeps the helper exhaustive without forcing every caller
        // to special-case Idle.
        CalibrationModalState::Idle { .. } => (0.0, String::new()),
        CalibrationModalState::WaitingForPose {
            last_score,
            frames_at_match,
            stillness_fallback,
            ..
        } => {
            // Two-tier progress bar:
            // - Below threshold: bar shows live match score so the user
            //   sees how close they are to matching the target pose.
            // - Above threshold: bar fills based on stable-frame count
            //   so the user sees the "lock-in" countdown before capture
            //   starts. This lets the bar signal both "you're getting
            //   warmer" and "almost there, hold it" without two
            //   separate UI elements.
            let progress = if *frames_at_match > 0 {
                (*frames_at_match as f32 / REQUIRED_STABLE_FRAMES as f32).clamp(0.0, 1.0)
            } else {
                last_score.clamp(0.0, 1.0)
            };
            // "Match N%" would misdescribe the stillness gate — the
            // score there is motion-derived, not pose similarity.
            let label = if *frames_at_match > 0 {
                t!("calibration.locking_in")
            } else if *stillness_fallback {
                t!(
                    "calibration.stillness_progress",
                    score = format!("{:.0}", last_score * 100.0)
                )
            } else {
                t!(
                    "calibration.match_progress",
                    score = format!("{:.0}", last_score * 100.0)
                )
            };
            (progress, label)
        }
        CalibrationModalState::Collecting { started_at, .. } => {
            let elapsed = started_at.elapsed().as_secs_f32();
            let remaining = (COLLECTION_SECONDS - elapsed).max(0.0);
            let progress = (elapsed / COLLECTION_SECONDS).clamp(0.0, 1.0);
            (
                progress,
                t!("calibration.time_left", time = format!("{:.1}", remaining)),
            )
        }
        // AnchorDone is user-input gated rather than time-gated, so
        // the progress bar is fully filled and the label tells the
        // user we're awaiting their choice.
        CalibrationModalState::AnchorDone { .. } => (1.0, t!("calibration.awaiting_choice")),
        CalibrationModalState::RangeHoldStill { started_at, .. } => {
            let elapsed = started_at.elapsed().as_secs_f32();
            let remaining = (RANGE_HOLD_STILL_SECONDS - elapsed).max(0.0);
            let progress = (elapsed / RANGE_HOLD_STILL_SECONDS).clamp(0.0, 1.0);
            (
                progress,
                t!("calibration.time_left", time = format!("{:.1}", remaining)),
            )
        }
        CalibrationModalState::RangeCollecting { started_at, .. } => {
            let elapsed = started_at.elapsed().as_secs_f32();
            let remaining = (RANGE_COLLECTION_SECONDS - elapsed).max(0.0);
            let progress = (elapsed / RANGE_COLLECTION_SECONDS).clamp(0.0, 1.0);
            (
                progress,
                t!("calibration.time_left", time = format!("{:.1}", remaining)),
            )
        }
        CalibrationModalState::Done { outcome, .. } => match outcome {
            DoneOutcome::Success { .. } => (1.0, t!("calibration.done")),
            DoneOutcome::Insufficient { .. } => (0.0, t!("calibration.failed")),
        },
        CalibrationModalState::Closed => (0.0, String::new()),
    }
}

/// `(anchor_seen, confidence, sample_count)` snapshot for the
/// telemetry rows in the status pane.
fn telemetry(state: &CalibrationModalState) -> (bool, f32, Option<usize>) {
    match state {
        CalibrationModalState::Idle {
            last_anchor_seen,
            last_confidence,
            ..
        } => (*last_anchor_seen, *last_confidence, None),
        CalibrationModalState::WaitingForPose {
            last_anchor_seen,
            last_confidence,
            ..
        } => (*last_anchor_seen, *last_confidence, None),
        CalibrationModalState::Collecting {
            last_anchor_seen,
            last_confidence,
            samples,
            ..
        } => (*last_anchor_seen, *last_confidence, Some(samples.len())),
        CalibrationModalState::AnchorDone { calibration, .. } => {
            // Anchor capture has already finished; the user just
            // needs to decide on the range step. Surface the
            // calibration's own confidence rather than the live
            // frame's (which they may have left to read the prompt).
            (true, calibration.confidence, Some(calibration.frame_count))
        }
        CalibrationModalState::RangeHoldStill {
            last_confidence, ..
        } => (true, *last_confidence, None),
        CalibrationModalState::RangeCollecting {
            last_confidence,
            samples_seen,
            ..
        } => (true, *last_confidence, Some(*samples_seen)),
        CalibrationModalState::Done { outcome, .. } => match outcome {
            DoneOutcome::Success { frame_count, .. } => (true, 1.0, Some(*frame_count)),
            DoneOutcome::Insufficient { samples_collected } => {
                (false, 0.0, Some(*samples_collected))
            }
        },
        CalibrationModalState::Closed => (false, 0.0, None),
    }
}
