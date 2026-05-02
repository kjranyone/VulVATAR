//! Pose-calibration fullscreen modal (Phase C: live capture).
//!
//! Phase B owned the UI shell. Phase C wires the capture loop:
//! per-frame samples are accumulated from the source-skeleton mailbox
//! during the `Collecting` window, the per-axis **median** + per-frame
//! **stddev** are computed on transition to `Done`, and the resulting
//! [`PoseCalibration`] is written into
//! `Application::tracking_calibration.pose` so persistence (auto-save
//! + project file) picks it up on the next round-trip.
//!
//! The capture rejects low-quality frames at intake — anchor must be
//! detected, the anchor type must match the chosen mode, and the
//! source-skeleton overall confidence must clear `MIN_FRAME_CONF`.
//! "Capture Now" finishes early but still requires `MIN_SAMPLES`
//! gathered samples to avoid writing a 1-frame "calibration" that's
//! no better than the auto-EMA's first frame.
//!
//! The modal is a true overlay: a dimmed full-viewport rect underneath
//! a centred `egui::Window` so all background interaction (orbit
//! controls, inspector clicks) is blocked while calibration is active.

use eframe::egui;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::gui::GuiApp;
use crate::t;
use crate::tracking::{CalibrationMode, PoseCalibration};

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

/// Minimum source-skeleton overall confidence for a sample to be
/// admitted into the median pool. Below this, the keypoints are too
/// noisy to give a reliable anchor reading and would skew the
/// aggregate even if the median weights down individual outliers.
const MIN_FRAME_CONF: f32 = 0.5;

/// Smallest sample count the modal will accept as a valid capture.
/// Below this the auto-EMA's first-frame seed is no worse than what
/// the median can offer, so we reject and force the user to retry.
const MIN_SAMPLES: usize = 5;

/// One per-frame anchor reading collected during the capture window.
/// Stored as `Vec<AnchorSample>` on the `Collecting` variant; the
/// transition to `Done` walks the vec to compute medians + stddev.
#[derive(Clone, Copy, Debug)]
pub struct AnchorSample {
    /// Source-space anchor position (matches `SourceSkeleton::root_offset`).
    /// For depth-aware paths z is metric metres; for rtmw3d-only z is 0.
    pub position: [f32; 3],
    /// Per-frame `overall_confidence`. Aggregated as a separate median
    /// so the surfaced calibration confidence reflects the gathered
    /// samples, not just the first frame's reading.
    pub confidence: f32,
}

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
    /// Active sample collection. Per-frame `AnchorSample` rows are
    /// appended each time the mailbox sequence advances *and* the
    /// frame meets the quality bar (anchor visible, anchor type
    /// matches mode, confidence ≥ `MIN_FRAME_CONF`). The transition
    /// to `Done` walks `samples` to compute the median calibration
    /// values.
    Collecting {
        mode: CalibrationMode,
        started_at: Instant,
        samples: Vec<AnchorSample>,
        /// Tracks the last mailbox sequence we admitted a sample for,
        /// so we don't double-count when the GUI repaints faster than
        /// the tracking thread emits new frames.
        last_seq_consumed: u64,
        last_anchor_seen: bool,
        last_confidence: f32,
    },
    /// Calibration finished — either the timer ran out or the user
    /// hit Capture Now. Either succeeded (≥ `MIN_SAMPLES` admitted
    /// frames, calibration written to `Application`) or was rejected
    /// (`outcome == DoneOutcome::Insufficient`); status pane shows
    /// the matching message before the modal auto-closes.
    Done {
        mode: CalibrationMode,
        shown_at: Instant,
        outcome: DoneOutcome,
    },
}

/// Result of a `Collecting → Done` transition, surfaced on the status
/// pane so the user knows whether they need to retry.
#[derive(Clone, Debug)]
pub enum DoneOutcome {
    /// Capture succeeded. Carries the aggregated calibration that was
    /// just written to `Application::tracking_calibration.pose`, used
    /// only for the post-capture status display.
    Success { frame_count: usize, calibration: PoseCalibration },
    /// Capture failed. `samples_collected` is below `MIN_SAMPLES` —
    /// the modal shows "Capture failed — N samples (need M)" so the
    /// user can adjust framing / lighting and retry.
    Insufficient { samples_collected: usize },
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

/// Drive the state machine forward by elapsed time. When the
/// collection window expires, finalises the capture (median over
/// `samples`, write into `Application::tracking_calibration`) and
/// transitions to `Done` with the appropriate outcome.
fn advance_state(state: &mut GuiApp) {
    let now = Instant::now();
    let next = match &state.calibration_modal {
        CalibrationModalState::HoldStill {
            mode, started_at, ..
        } if now.duration_since(*started_at).as_secs_f32() >= HOLD_STILL_SECONDS => {
            Some(CalibrationModalState::Collecting {
                mode: *mode,
                started_at: now,
                samples: Vec::with_capacity(64),
                last_seq_consumed: 0,
                last_anchor_seen: false,
                last_confidence: 0.0,
            })
        }
        CalibrationModalState::Collecting {
            mode, started_at, ..
        } if now.duration_since(*started_at).as_secs_f32() >= COLLECTION_SECONDS => {
            // Capture window expired: finalise with whatever we
            // collected. `finalize_collection` handles median +
            // writing the calibration; the modal then lingers in
            // `Done` for `DONE_LINGER_SECONDS`.
            Some(finalize_collection(state, *mode, now))
        }
        _ => None,
    };
    if let Some(s) = next {
        state.calibration_modal = s;
    }
}

/// Force the state machine to the `Done` state regardless of timer.
/// Used by the `Capture Now` button so impatient users don't have to
/// wait for the full collection window — but the same `MIN_SAMPLES`
/// floor applies, so impatience can still produce an `Insufficient`
/// outcome.
fn finish_capture(state: &mut GuiApp) {
    let mode = match &state.calibration_modal {
        CalibrationModalState::HoldStill { mode, .. }
        | CalibrationModalState::Collecting { mode, .. } => *mode,
        _ => return,
    };
    let now = Instant::now();
    state.calibration_modal = finalize_collection(state, mode, now);
}

/// Take whatever's currently in the `Collecting` variant's `samples`
/// vec, compute the per-axis median + jitter, write the resulting
/// `PoseCalibration` to `Application::tracking_calibration.pose`, and
/// return the `Done` variant carrying the success/failure outcome.
///
/// Called from both the timer-expired path and the `Capture Now`
/// path, so the median / write logic stays in one place.
fn finalize_collection(
    state: &mut GuiApp,
    mode: CalibrationMode,
    now: Instant,
) -> CalibrationModalState {
    // Pull the current samples out of the state; HoldStill (Capture
    // Now hit before any frames were collected) yields an empty vec.
    let samples: Vec<AnchorSample> = match &mut state.calibration_modal {
        CalibrationModalState::Collecting { samples, .. } => std::mem::take(samples),
        _ => Vec::new(),
    };

    if samples.len() < MIN_SAMPLES {
        return CalibrationModalState::Done {
            mode,
            shown_at: now,
            outcome: DoneOutcome::Insufficient {
                samples_collected: samples.len(),
            },
        };
    }

    let calibration = aggregate(&samples, mode);

    // Write into Application so the solver / project file see the
    // new value next frame. Also push to the tracking mailbox so the
    // worker thread forwards it to the depth-pipeline provider via
    // `set_calibration` — that path enables Upper Body anchor
    // forcing and (future) jitter-derived c-clamp narrowing.
    // Marking the project dirty nudges the user to save (auto-save
    // also picks it up).
    state.app.tracking_calibration.pose = Some(calibration.clone());
    state.app.tracking.mailbox().set_calibration(Some(calibration.clone()));
    state.project_dirty = true;

    CalibrationModalState::Done {
        mode,
        shown_at: now,
        outcome: DoneOutcome::Success {
            frame_count: calibration.frame_count,
            calibration,
        },
    }
}

/// Compute the `PoseCalibration` from a non-empty sample vec.
/// Per-axis values use the **median** (robust to occasional outliers
/// like a finger drifting through the hip keypoint window); confidence
/// is the median per-frame confidence; depth jitter is the per-frame
/// stddev of the z component (used by Phase D as the dynamic c-clamp
/// range). For rtmw3d-only samples (z always 0), `anchor_depth_m`
/// and `anchor_depth_jitter_m` are emitted as `None`.
fn aggregate(samples: &[AnchorSample], mode: CalibrationMode) -> PoseCalibration {
    let mut xs: Vec<f32> = samples.iter().map(|s| s.position[0]).collect();
    let mut ys: Vec<f32> = samples.iter().map(|s| s.position[1]).collect();
    let mut zs: Vec<f32> = samples.iter().map(|s| s.position[2]).collect();
    let mut confs: Vec<f32> = samples.iter().map(|s| s.confidence).collect();

    let median_x = median_inplace(&mut xs);
    let median_y = median_inplace(&mut ys);
    let median_z = median_inplace(&mut zs);
    let median_conf = median_inplace(&mut confs);

    // rtmw3d-only paths emit z = 0 unconditionally (no metric depth).
    // Surface as `None` so consumers don't accidentally treat zero as
    // a meaningful metric distance.
    let z_is_metric = zs.iter().any(|&z| z.abs() > 1e-4);
    let anchor_depth_m = z_is_metric.then_some(median_z);
    let anchor_depth_jitter_m = if z_is_metric {
        let mean: f32 = zs.iter().sum::<f32>() / zs.len() as f32;
        let var: f32 = zs.iter().map(|&z| (z - mean).powi(2)).sum::<f32>() / zs.len() as f32;
        Some(var.sqrt())
    } else {
        None
    };

    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    PoseCalibration {
        mode,
        captured_at: iso_now_from_unix(now_unix),
        captured_at_unix: now_unix,
        frame_count: samples.len(),
        anchor_x: median_x,
        anchor_y: median_y,
        anchor_depth_m,
        confidence: median_conf,
        anchor_depth_jitter_m,
    }
}

/// In-place median: sorts `values` and returns the middle (or average
/// of the two middles for even length). Returns `0.0` for an empty
/// slice — callers gate on `MIN_SAMPLES` before reaching here, so
/// that branch should be unreachable in practice.
fn median_inplace(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

/// ISO-8601 UTC timestamp string from a unix-seconds value. The
/// `captured_at_unix` field on `PoseCalibration` is the canonical
/// timestamp source (used by the inspector for the relative-age
/// rendering); this string form exists for human-readable
/// project-file inspection.
fn iso_now_from_unix(secs: u64) -> String {
    // Hand-rolled formatter to avoid pulling in chrono just for this:
    // YYYY-MM-DDTHH:MM:SSZ. Date math via the standard "days since
    // epoch" Howard Hinnant algorithm.
    let days = (secs / 86_400) as i64;
    let secs_in_day = (secs % 86_400) as u32;
    let hour = secs_in_day / 3_600;
    let minute = (secs_in_day % 3_600) / 60;
    let second = secs_in_day % 60;

    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, m, d, hour, minute, second
    )
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

    // During the collection window, append the per-frame anchor +
    // confidence to the median pool. Three gates apply:
    //   1. The mailbox sequence must have advanced — otherwise we'd
    //      double-count the same frame each repaint.
    //   2. The anchor must match the chosen mode (hip for FullBody,
    //      shoulder for UpperBody) so a partially-visible subject
    //      doesn't mix anchor types in the median.
    //   3. The frame's overall confidence must clear MIN_FRAME_CONF
    //      so noisy keypoints don't drag the median around.
    if let CalibrationModalState::Collecting {
        mode,
        ref mut samples,
        ref mut last_seq_consumed,
        ..
    } = state.calibration_modal
    {
        if snap.sequence > *last_seq_consumed {
            *last_seq_consumed = snap.sequence;
            if let Some(pose) = snap.pose.as_ref() {
                let anchor_kind_matches = match mode {
                    CalibrationMode::FullBody => pose.root_anchor_is_hip,
                    CalibrationMode::UpperBody => !pose.root_anchor_is_hip,
                };
                if anchor_kind_matches
                    && pose.overall_confidence >= MIN_FRAME_CONF
                {
                    if let Some(pos) = pose.root_offset {
                        samples.push(AnchorSample {
                            position: pos,
                            confidence: pose.overall_confidence,
                        });
                    }
                }
            }
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
        CalibrationModalState::Done { outcome, .. } => match outcome {
            DoneOutcome::Success { frame_count, calibration } => {
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
        CalibrationModalState::Done { outcome, .. } => match outcome {
            DoneOutcome::Success { .. } => (1.0, t!("calibration.done")),
            DoneOutcome::Insufficient { .. } => (0.0, t!("calibration.failed")),
        },
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
            samples,
            ..
        } => (*last_anchor_seen, *last_confidence, Some(samples.len())),
        CalibrationModalState::Done { outcome, .. } => match outcome {
            DoneOutcome::Success { frame_count, .. } => (true, 1.0, Some(*frame_count)),
            DoneOutcome::Insufficient { samples_collected } => {
                (false, 0.0, Some(*samples_collected))
            }
        },
        CalibrationModalState::Closed => (false, 0.0, None),
    }
}
