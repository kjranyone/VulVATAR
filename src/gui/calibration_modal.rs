//! Pose-calibration fullscreen modal (Phase B scaffold).
//!
//! Phase B owns the *UI shell*: state machine, countdown timer, webcam
//! preview routing, and the Cancel / Capture Now controls. The actual
//! frame collection + median aggregation is stubbed (Phase C wires it
//! up to the source-skeleton mailbox), and the captured values are
//! discarded on close until Phase C hooks them into
//! `Application::tracking_calibration.pose`.
//!
//! The modal is a true overlay: a dimmed full-viewport rect underneath
//! a centred `egui::Window` so all background interaction (orbit
//! controls, inspector clicks) is blocked while calibration is active.

use eframe::egui;
use std::time::Instant;

use crate::gui::GuiApp;
use crate::t;
use crate::tracking::CalibrationMode;

/// Capture-window timing constants. Mirror the `docs/calibration-ux.md`
/// table; tuned so the user has time to settle into the pose without
/// the modal feeling sluggish.
///
/// `HOLD_STILL_SECONDS`: countdown before the actual capture begins.
/// `COLLECTION_SECONDS`: capture window duration. At 30 fps this
/// yields ~60 sample frames; the median over that window absorbs
/// per-frame keypoint jitter.
const HOLD_STILL_SECONDS: f32 = 1.0;
const COLLECTION_SECONDS: f32 = 2.0;

/// Internal state of the modal. `Closed` means it is not rendered at
/// all; the other variants drive the per-frame UI under
/// [`draw_modal`].
#[derive(Clone, Debug)]
pub enum CalibrationModalState {
    Closed,
    /// "Get into the T-pose / hands-at-sides pose" countdown. Counts
    /// down from `HOLD_STILL_SECONDS` to 0; transitions to
    /// `Collecting` automatically at zero.
    HoldStill {
        mode: CalibrationMode,
        started_at: Instant,
        last_anchor_seen: bool,
        last_confidence: f32,
    },
    /// Active sample collection. Phase C will populate `samples` from
    /// the source-skeleton mailbox; for Phase B we just count frames
    /// so the progress bar advances.
    Collecting {
        mode: CalibrationMode,
        started_at: Instant,
        samples_count: usize,
        last_anchor_seen: bool,
        last_confidence: f32,
    },
    /// Brief "Calibrated!" confirmation before the modal auto-closes.
    /// Phase B uses placeholder values; Phase C fills in the real
    /// median-aggregated `PoseCalibration`.
    Done {
        mode: CalibrationMode,
        shown_at: Instant,
        frame_count: usize,
    },
}

impl Default for CalibrationModalState {
    fn default() -> Self {
        CalibrationModalState::Closed
    }
}

impl CalibrationModalState {
    pub fn is_open(&self) -> bool {
        !matches!(self, CalibrationModalState::Closed)
    }

    pub fn open(&mut self, mode: CalibrationMode) {
        *self = CalibrationModalState::HoldStill {
            mode,
            started_at: Instant::now(),
            last_anchor_seen: false,
            last_confidence: 0.0,
        };
    }

    pub fn close(&mut self) {
        *self = CalibrationModalState::Closed;
    }
}

/// Per-frame entry point. No-op when the modal is closed; otherwise
/// advances the state machine, paints the dim overlay + centred
/// `Window`, and handles the Cancel / Capture Now buttons.
pub fn draw_modal(ctx: &egui::Context, state: &mut GuiApp) {
    if !state.calibration_modal.is_open() {
        return;
    }

    // Tick the state machine forward. Done after a `DONE_LINGER_SECONDS`
    // hold reverts to Closed.
    const DONE_LINGER_SECONDS: f32 = 1.5;
    advance_state(state);
    if let CalibrationModalState::Done { shown_at, .. } = &state.calibration_modal {
        if shown_at.elapsed().as_secs_f32() > DONE_LINGER_SECONDS {
            state.calibration_modal.close();
            return;
        }
    }

    // Dim the entire viewport so background widgets read as inactive.
    let screen_rect = ctx.screen_rect();
    let dim_layer = egui::LayerId::new(egui::Order::Foreground, egui::Id::new("calibration_dim"));
    let dim_painter = ctx.layer_painter(dim_layer);
    dim_painter.rect_filled(
        screen_rect,
        0.0,
        egui::Color32::from_rgba_unmultiplied(0, 0, 0, 180),
    );

    // Centred modal window. Disabling collapse / resize / movement so
    // the user can only interact via the buttons we provide.
    egui::Window::new(modal_title(&state.calibration_modal))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .collapsible(false)
        .resizable(false)
        .movable(false)
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            ui.set_min_size(egui::vec2(720.0, 420.0));
            ui.horizontal_top(|ui| {
                draw_preview_pane(ui, state);
                ui.add_space(16.0);
                draw_status_pane(ui, state);
            });
        });
}

fn modal_title(state: &CalibrationModalState) -> String {
    let mode = match state {
        CalibrationModalState::HoldStill { mode, .. }
        | CalibrationModalState::Collecting { mode, .. }
        | CalibrationModalState::Done { mode, .. } => *mode,
        CalibrationModalState::Closed => return String::new(),
    };
    let mode_label = match mode {
        CalibrationMode::FullBody => t!("calibration.mode_full_body"),
        CalibrationMode::UpperBody => t!("calibration.mode_upper_body"),
    };
    format!("{} — {}", t!("calibration.title"), mode_label)
}

/// Webcam preview pane (left side of the modal). Reuses the same
/// camera-wipe texture-upload pattern from `gui::viewport` so the
/// frame data path doesn't fork; the difference is just a larger
/// rect and the absence of the PIP corner styling.
fn draw_preview_pane(ui: &mut egui::Ui, state: &mut GuiApp) {
    let preview_size = egui::vec2(480.0, 270.0);
    let (rect, _) = ui.allocate_exact_size(preview_size, egui::Sense::hover());

    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 6.0, egui::Color32::from_rgb(18, 18, 22));
    painter.rect_stroke(
        rect,
        6.0,
        egui::Stroke::new(2.0, egui::Color32::from_rgb(60, 130, 200)),
    );

    let snap = state.app.tracking.mailbox().snapshot();
    let Some(ref frame) = snap.frame else {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            t!("calibration.no_frame"),
            egui::FontId::proportional(14.0),
            egui::Color32::from_rgb(180, 180, 180),
        );
        return;
    };

    // Refresh texture on new sequence — same path the camera-wipe
    // uses. Sharing the buffer / handle would couple the two; keep
    // them separate so toggling camera-wipe while calibrating doesn't
    // tear either preview.
    if snap.sequence != state.calibration_preview_seq {
        let w = frame.width as usize;
        let h = frame.height as usize;
        if w > 0 && h > 0 && frame.rgb_data.len() == w * h * 3 {
            let needed = w * h * 4;
            if state.calibration_preview_rgba_buf.len() != needed {
                state.calibration_preview_rgba_buf.resize(needed, 255);
            }
            let rgba = &mut state.calibration_preview_rgba_buf;
            for i in 0..w * h {
                rgba[i * 4] = frame.rgb_data[i * 3];
                rgba[i * 4 + 1] = frame.rgb_data[i * 3 + 1];
                rgba[i * 4 + 2] = frame.rgb_data[i * 3 + 2];
                rgba[i * 4 + 3] = 255;
            }
            let color_image =
                egui::ColorImage::from_rgba_unmultiplied([w, h], rgba);
            let options = egui::TextureOptions {
                magnification: egui::TextureFilter::Linear,
                minification: egui::TextureFilter::Linear,
                ..Default::default()
            };
            if let Some(ref mut handle) = state.calibration_preview_texture {
                handle.set(color_image, options);
            } else {
                let handle = ui
                    .ctx()
                    .load_texture("calibration_preview", color_image, options);
                state.calibration_preview_texture = Some(handle);
            }
        }
        state.calibration_preview_seq = snap.sequence;
    }

    let Some(ref tex) = state.calibration_preview_texture else {
        return;
    };

    let uv = if state.tracking.tracking_mirror {
        egui::Rect::from_min_max(egui::pos2(1.0, 0.0), egui::pos2(0.0, 1.0))
    } else {
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0))
    };
    painter.image(tex.id(), rect, uv, egui::Color32::WHITE);

    // Skeleton + keypoint overlay. Same style the camera-wipe paints —
    // the calibration UX is most useful when the user can see exactly
    // which keypoints the tracker is locking onto.
    if let Some(ref ann) = snap.annotation {
        let kpt_color = egui::Color32::from_rgba_unmultiplied(0, 255, 128, 220);
        let line_color = egui::Color32::from_rgba_unmultiplied(0, 200, 255, 180);
        let mirror = state.tracking.tracking_mirror;
        let map_x = |nx: f32| {
            let x = if mirror { 1.0 - nx } else { nx };
            rect.left() + x * rect.width()
        };
        let map_y = |ny: f32| rect.top() + ny * rect.height();

        for &(kx, ky, conf) in &ann.keypoints {
            if conf < 0.1 {
                continue;
            }
            painter.circle_filled(
                egui::pos2(map_x(kx), map_y(ky)),
                3.0,
                kpt_color,
            );
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

    // Latest tracking confidence drives the side panel's "Pose detected?"
    // indicator without re-querying the mailbox there.
    refresh_anchor_telemetry(state, &snap);
}

/// Status pane (right side of the modal): instructions, countdown bar,
/// confidence indicator, frame counter, action buttons.
fn draw_status_pane(ui: &mut egui::Ui, state: &mut GuiApp) {
    ui.vertical(|ui| {
        ui.set_min_width(220.0);

        // Step indicator + instructions.
        let (step_label, instructions) = step_text(&state.calibration_modal);
        ui.label(egui::RichText::new(step_label).size(13.0).strong());
        ui.add_space(4.0);
        ui.label(egui::RichText::new(instructions).size(12.0));
        ui.add_space(12.0);

        // Progress bar (HoldStill: 1s countdown; Collecting: 2s window).
        let (progress, time_left) = progress_for(&state.calibration_modal);
        ui.add(
            egui::ProgressBar::new(progress)
                .desired_width(220.0)
                .text(time_left),
        );
        ui.add_space(12.0);

        // Live telemetry: confidence + anchor visible flag + sample
        // count (Collecting only). Phase C will surface depth too.
        let (anchor_seen, conf, samples) = telemetry(&state.calibration_modal);
        let conf_color = if conf >= 0.5 {
            egui::Color32::from_rgb(120, 220, 140)
        } else {
            egui::Color32::from_rgb(220, 130, 120)
        };
        ui.horizontal(|ui| {
            ui.label(t!("calibration.confidence"));
            ui.colored_label(conf_color, format!("{:.2}", conf));
        });
        ui.horizontal(|ui| {
            ui.label(t!("calibration.anchor_visible"));
            let (label, color) = if anchor_seen {
                ("✓", egui::Color32::from_rgb(120, 220, 140))
            } else {
                ("✗", egui::Color32::from_rgb(220, 130, 120))
            };
            ui.colored_label(color, label);
        });
        if let Some(s) = samples {
            ui.horizontal(|ui| {
                ui.label(t!("calibration.samples"));
                ui.label(format!("{}", s));
            });
        }

        ui.add_space(16.0);

        // Action buttons. `Cancel` always available; `Capture Now` only
        // during HoldStill / Collecting. Phase C: Capture Now should
        // require ≥ 5 samples in Collecting state.
        ui.horizontal(|ui| {
            if ui.button(t!("calibration.cancel")).clicked() {
                state.calibration_modal.close();
            }
            let allow_capture_now = matches!(
                state.calibration_modal,
                CalibrationModalState::HoldStill { .. }
                    | CalibrationModalState::Collecting { .. }
            );
            ui.add_enabled_ui(allow_capture_now, |ui| {
                if ui.button(t!("calibration.capture_now")).clicked() {
                    finish_capture(state);
                }
            });
        });
    });
}

/// Drive the state machine forward by elapsed time.
fn advance_state(state: &mut GuiApp) {
    let now = Instant::now();
    let next = match state.calibration_modal {
        CalibrationModalState::HoldStill {
            mode, started_at, ..
        } if now.duration_since(started_at).as_secs_f32() >= HOLD_STILL_SECONDS => {
            Some(CalibrationModalState::Collecting {
                mode,
                started_at: now,
                samples_count: 0,
                last_anchor_seen: false,
                last_confidence: 0.0,
            })
        }
        CalibrationModalState::Collecting {
            mode, started_at, ..
        } if now.duration_since(started_at).as_secs_f32() >= COLLECTION_SECONDS => {
            // Phase C will return the median-aggregated PoseCalibration
            // here. For Phase B we just transition to Done with a
            // placeholder frame count.
            Some(CalibrationModalState::Done {
                mode,
                shown_at: now,
                frame_count: 0, // TODO(phase-c): real sample count
            })
        }
        _ => None,
    };
    if let Some(s) = next {
        state.calibration_modal = s;
    }
}

/// Force the state machine to the `Done` state regardless of timer.
/// Used by the `Capture Now` button so impatient users don't have to
/// wait for the full collection window.
fn finish_capture(state: &mut GuiApp) {
    let mode = match &state.calibration_modal {
        CalibrationModalState::HoldStill { mode, .. }
        | CalibrationModalState::Collecting { mode, .. } => *mode,
        _ => return,
    };
    state.calibration_modal = CalibrationModalState::Done {
        mode,
        shown_at: Instant::now(),
        frame_count: 0, // TODO(phase-c): real sample count
    };
}

/// Update the cached "anchor visible / confidence" telemetry from the
/// latest mailbox snapshot. Stored on the state-machine variant so the
/// status pane doesn't re-snapshot the mailbox.
fn refresh_anchor_telemetry(
    state: &mut GuiApp,
    snap: &crate::tracking::MailboxSnapshot,
) {
    let confidence = snap
        .pose
        .as_ref()
        .map(|p| p.overall_confidence)
        .unwrap_or(0.0);
    // Anchor visibility: hip pair for FullBody, shoulder pair for
    // UpperBody. Hits the joints map directly so we read the same
    // floors `skeleton_from_depth` uses.
    let anchor_seen = snap
        .pose
        .as_ref()
        .map(|p| {
            use crate::asset::HumanoidBone;
            match relevant_mode(&state.calibration_modal) {
                Some(CalibrationMode::FullBody) => {
                    p.joints.contains_key(&HumanoidBone::Hips)
                }
                Some(CalibrationMode::UpperBody) => {
                    p.joints.contains_key(&HumanoidBone::LeftShoulder)
                        && p.joints.contains_key(&HumanoidBone::RightShoulder)
                }
                None => false,
            }
        })
        .unwrap_or(false);

    match &mut state.calibration_modal {
        CalibrationModalState::HoldStill {
            last_anchor_seen,
            last_confidence,
            ..
        }
        | CalibrationModalState::Collecting {
            last_anchor_seen,
            last_confidence,
            ..
        } => {
            *last_anchor_seen = anchor_seen;
            *last_confidence = confidence;
        }
        _ => {}
    }

    // Bump samples_count once per new mailbox sequence so the side
    // panel reflects active capture progress. Phase C replaces this
    // with real frame accumulation into a Vec.
    if let CalibrationModalState::Collecting {
        ref mut samples_count,
        ..
    } = state.calibration_modal
    {
        if anchor_seen && confidence >= 0.5 {
            *samples_count = samples_count.saturating_add(1);
        }
    }
}

fn relevant_mode(state: &CalibrationModalState) -> Option<CalibrationMode> {
    match state {
        CalibrationModalState::HoldStill { mode, .. }
        | CalibrationModalState::Collecting { mode, .. }
        | CalibrationModalState::Done { mode, .. } => Some(*mode),
        CalibrationModalState::Closed => None,
    }
}

fn step_text(state: &CalibrationModalState) -> (String, String) {
    match state {
        CalibrationModalState::HoldStill { mode, .. } => {
            let pose = match mode {
                CalibrationMode::FullBody => t!("calibration.pose_full_body"),
                CalibrationMode::UpperBody => t!("calibration.pose_upper_body"),
            };
            (t!("calibration.step_hold_still"), pose)
        }
        CalibrationModalState::Collecting { mode, .. } => {
            let pose = match mode {
                CalibrationMode::FullBody => t!("calibration.pose_full_body"),
                CalibrationMode::UpperBody => t!("calibration.pose_upper_body"),
            };
            (t!("calibration.step_collecting"), pose)
        }
        CalibrationModalState::Done { .. } => (
            t!("calibration.step_done"),
            t!("calibration.step_done_body"),
        ),
        CalibrationModalState::Closed => (String::new(), String::new()),
    }
}

fn progress_for(state: &CalibrationModalState) -> (f32, String) {
    match state {
        CalibrationModalState::HoldStill { started_at, .. } => {
            let elapsed = started_at.elapsed().as_secs_f32();
            let remaining = (HOLD_STILL_SECONDS - elapsed).max(0.0);
            let progress = (elapsed / HOLD_STILL_SECONDS).clamp(0.0, 1.0);
            (
                progress,
                t!("calibration.time_left", time = format!("{:.1}", remaining)),
            )
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
        CalibrationModalState::Done { .. } => (1.0, t!("calibration.done")),
        CalibrationModalState::Closed => (0.0, String::new()),
    }
}

fn telemetry(state: &CalibrationModalState) -> (bool, f32, Option<usize>) {
    match state {
        CalibrationModalState::HoldStill {
            last_anchor_seen,
            last_confidence,
            ..
        } => (*last_anchor_seen, *last_confidence, None),
        CalibrationModalState::Collecting {
            last_anchor_seen,
            last_confidence,
            samples_count,
            ..
        } => (*last_anchor_seen, *last_confidence, Some(*samples_count)),
        CalibrationModalState::Done { frame_count, .. } => (true, 1.0, Some(*frame_count)),
        CalibrationModalState::Closed => (false, 0.0, None),
    }
}
