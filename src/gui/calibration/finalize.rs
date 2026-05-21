//! Sample aggregation + persistence. Turns the per-frame
//! `Vec<AnchorSample>` collected during `Collecting` (and the per-axis
//! min/max tallied during `RangeCollecting`) into a
//! [`PoseCalibration`], then writes it everywhere the live solver
//! needs to see it: `Application::tracking_calibration`, the tracking
//! mailbox (so the worker thread forwards to the depth provider), and
//! the active profile (so the calibration survives a process restart).

use std::time::{Instant, SystemTime, UNIX_EPOCH};

use log::warn;

use crate::gui::GuiApp;
use crate::tracking::{CalibrationMode, PoseCalibration, SourceSkeleton};

use super::state::{AnchorSample, CalibrationModalState, DoneOutcome};
use super::transitions::transition_to_done_success;
use super::MIN_SAMPLES;

/// Finalize the *anchor* capture step. Pulls the samples out of the
/// `Collecting` variant, aggregates them, writes the resulting
/// `PoseCalibration` to `Application` + the tracking mailbox, and
/// returns either `AnchorDone` (success, awaiting the user's range
/// choice) or `Done { Insufficient }` (too few admitted frames).
///
/// Called from both the anchor-timer-expired path and `Capture Now`,
/// so the median / write logic stays in one place. Note that the
/// anchor calibration is *immediately* persisted at this point — the
/// optional range step only enriches the same record in place.
pub(super) fn finalize_collection(
    state: &mut GuiApp,
    mode: CalibrationMode,
    now: Instant,
) -> CalibrationModalState {
    // Pull the current samples out of the state; the WaitingForPose
    // escape-hatch path (Capture Now hit before any frames were
    // collected) yields an empty vec.
    let samples: Vec<AnchorSample> = match &mut state.calibration.modal {
        CalibrationModalState::Collecting { samples, .. } => std::mem::take(samples),
        _ => Vec::new(),
    };

    // Tell the worker to stop accumulating torso template samples
    // and to publish whatever's in its buffer. The publish lands on
    // the mailbox 1–2 frames later; `poll_torso_template` (called
    // every frame from `draw_modal`) picks it up and stitches it
    // onto the in-flight calibration.
    //
    // We send the toggle even on the Insufficient path so the
    // worker doesn't hold a stale buffer across a failed capture.
    state.app.tracking.mailbox().set_torso_capture(false);

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
    // forcing and jitter-derived c-clamp narrowing. Marking the
    // project dirty nudges the user to save (auto-save also picks
    // it up). Pause on `AnchorDone` for the user's range-step
    // decision; if they hit Skip we ship the same record unchanged,
    // and if they capture range we update *this same in-memory
    // record* in place.
    persist_calibration(state, calibration.clone());

    CalibrationModalState::AnchorDone {
        mode,
        shown_at: now,
        calibration,
    }
}

/// Write a freshly-aggregated `PoseCalibration` to all the places
/// that need to see it: Application (per-frame solver consumers),
/// the tracking mailbox (depth-pipeline provider's c-clamp /
/// anchor-mode), and the *active profile* (so the next app launch
/// or profile-switch picks it back up). Also flips the autosave
/// flag so the profile change hits disk on the next frame.
///
/// Lives next to `finalize_collection` and `finalize_range_collection`
/// because both finalisers do the same write — the only difference
/// between them is what's *in* the calibration, not where it lands.
/// When no profile is active (degenerate state — the GUI ships with
/// a default-active profile, but a corrupted profiles.json could
/// produce an empty library) we still set Application + mailbox so
/// the calibration applies to the current session, but the value is
/// lost on restart.
fn persist_calibration(state: &mut GuiApp, calibration: PoseCalibration) {
    state.app.tracking_calibration.pose = Some(calibration.clone());
    state
        .app
        .tracking
        .mailbox()
        .set_calibration(Some(calibration.clone()));
    if let Some(idx) = state.profiles.active_index {
        if let Some(profile) = state.profiles.profiles.get_mut(idx) {
            profile.pose_calibration = Some(calibration);
            state.profiles_dirty = true;
        }
    }
}

/// Poll the tracking mailbox for a worker-published torso depth
/// template and stitch it onto every place that holds the in-flight
/// `PoseCalibration`. No-op until the worker publishes (which
/// happens 1–2 frames after `Collecting → AnchorDone` because the
/// publish goes through the worker loop).
///
/// Stitching points (in order):
/// 1. The active profile's `pose_calibration` (so the persisted
///    calibration on disk includes the template once
///    `profiles_dirty` autosaves it).
/// 2. `Application::tracking_calibration.pose` (so the live solver
///    / depth pipeline see the enriched calibration starting next
///    frame).
/// 3. The mailbox's calibration channel (so the depth provider's
///    `set_calibration` sees the enriched value and can use the
///    template for inference-time bias correction).
/// 4. The modal-state's local calibration copy in AnchorDone /
///    RangeHoldStill / RangeCollecting / Done::Success — the
///    success-display body text can then mention "torso template
///    captured" and the user knows the rich version landed.
pub(super) fn poll_torso_template(state: &mut GuiApp) {
    let Some((template, seq)) = state
        .app
        .tracking
        .mailbox()
        .take_torso_template(state.calibration.torso_template_seq)
    else {
        return;
    };
    state.calibration.torso_template_seq = seq;

    // Stitch onto the active profile (and mark it dirty so the next
    // autosave persists the enriched calibration).
    let mut updated: Option<PoseCalibration> = None;
    if let Some(idx) = state.profiles.active_index {
        if let Some(profile) = state.profiles.profiles.get_mut(idx) {
            if let Some(cal) = profile.pose_calibration.as_mut() {
                cal.torso_depth_template = Some(template.clone());
                state.profiles_dirty = true;
                updated = Some(cal.clone());
            }
        }
    }
    // Mirror onto Application + mailbox so live consumers see the
    // template on the very next frame.
    if let Some(ref cal) = updated {
        state.app.tracking_calibration.pose = Some(cal.clone());
        state
            .app
            .tracking
            .mailbox()
            .set_calibration(Some(cal.clone()));
    }
    // Mirror onto the modal's own carried calibration so the
    // success-display reflects the template.
    match &mut state.calibration.modal {
        CalibrationModalState::AnchorDone { calibration, .. }
        | CalibrationModalState::RangeHoldStill { calibration, .. }
        | CalibrationModalState::RangeCollecting { calibration, .. } => {
            calibration.torso_depth_template = Some(template.clone());
        }
        CalibrationModalState::Done {
            outcome: DoneOutcome::Success { calibration, .. },
            ..
        } => {
            calibration.torso_depth_template = Some(template);
        }
        _ => {}
    }
}

/// Finalize the *range* capture step. Reads the per-axis min/max out
/// of the `RangeCollecting` variant, derives `x_range_observed` /
/// `z_range_observed`, updates the in-flight `PoseCalibration` (and
/// re-publishes it to `Application` + mailbox), and returns
/// `Done { Success }`.
///
/// "Insufficient" range data is not a failure here — the anchor
/// calibration was already persisted in `finalize_collection`, so
/// even a zero-sample range step just leaves the calibration as
/// anchor-only. Only the per-axis range fields are emitted as `None`
/// in that case, and the solver falls back to its static
/// `[0.6, 0.6, 0.3]` default for those axes.
///
/// `MIN_RANGE_OBSERVED` floors the recorded range so a user who
/// barely shifted during the window doesn't poison the solver with a
/// near-zero divisor on the per-axis sensitivity calc. Also requires
/// at least `MIN_RANGE_SAMPLES` to have been collected — fewer than
/// that and the min/max almost certainly reflects per-frame noise
/// rather than a deliberate sweep.
pub(super) fn finalize_range_collection(
    state: &mut GuiApp,
    mode: CalibrationMode,
    now: Instant,
) -> CalibrationModalState {
    const MIN_RANGE_OBSERVED: f32 = 0.05;
    const MIN_RANGE_SAMPLES: usize = 10;

    let (mut calibration, x_range, z_range, samples_seen, z_was_metric) =
        match &mut state.calibration.modal {
            CalibrationModalState::RangeCollecting {
                calibration,
                x_min,
                x_max,
                z_min,
                z_max,
                samples_seen,
                ..
            } => {
                let cal = calibration.clone();
                // INFINITY-seeded sentinels mean no admitted samples;
                // collapse to a 0 range so the gate below rejects.
                let x = if x_min.is_finite() && x_max.is_finite() {
                    (*x_max - *x_min).max(0.0)
                } else {
                    0.0
                };
                // Z is only meaningful on depth-aware providers — for
                // rtmw3d-only we never wrote a non-zero z into the
                // anchor sample, so `(z_max - z_min)` will be ~0 and
                // we'll discard via MIN_RANGE_OBSERVED below.
                let z = if z_min.is_finite() && z_max.is_finite() {
                    (*z_max - *z_min).max(0.0)
                } else {
                    0.0
                };
                let was_metric = cal.anchor_depth_m.is_some();
                (cal, x, z, *samples_seen, was_metric)
            }
            _ => {
                // finalize_range only runs out of RangeCollecting,
                // which is only entered after AnchorDone wrote a
                // Some(_) into Application. Reaching here means an
                // earlier transition was buggy or got stomped — log
                // loudly and bail to Closed instead of fabricating a
                // junk all-zero PoseCalibration that would silently
                // overwrite any real one the user already has.
                if let Some(existing) = state.app.tracking_calibration.pose.clone() {
                    return transition_to_done_success(mode, existing, now);
                }
                warn!(
                    "calibration: finalize_range entered without an existing pose \
                     (modal_state mismatch?); closing modal without writing"
                );
                return CalibrationModalState::Closed;
            }
        };

    if samples_seen >= MIN_RANGE_SAMPLES {
        if x_range >= MIN_RANGE_OBSERVED {
            calibration.x_range_observed = Some(x_range);
        }
        // Only emit z range when the anchor was metric to begin with;
        // otherwise the value is meaningless. The `MIN_RANGE_OBSERVED`
        // floor already rejects rtmw3d-only residuals, but the
        // explicit gate makes the contract obvious for downstream
        // readers.
        if z_was_metric && z_range >= MIN_RANGE_OBSERVED {
            calibration.z_range_observed = Some(z_range);
        }
    }

    persist_calibration(state, calibration.clone());

    transition_to_done_success(mode, calibration, now)
}

/// Compute the `PoseCalibration` from a non-empty sample vec.
/// Per-axis values use the **median** (robust to occasional outliers
/// like a finger drifting through the hip keypoint window); confidence
/// is the median per-frame confidence; depth jitter is the per-frame
/// stddev of the z component (used by the dynamic c-clamp range in
/// the depth provider). For rtmw3d-only samples (z always 0),
/// `anchor_depth_m` and `anchor_depth_jitter_m` are emitted as `None`.
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
    //
    // Sign convention: `pose.root_offset[2]` is in **source-space**
    // (z toward camera, so subjects in front of the camera produce a
    // negative value). The `anchor_depth_m` field is documented as
    // *Camera-space metric depth* (= the positive forward distance),
    // so we take the absolute value here. Without this, `cal_depth`
    // ends up negative and the plausibility check in
    // `rtmw3d_with_depth::calibrate_scale` rejects every live frame
    // because `predicted_depth` (always positive) can never match a
    // negative reference.
    let z_is_metric = zs.iter().any(|&z| z.abs() > 1e-4);
    let anchor_depth_m = z_is_metric.then_some(median_z.abs());
    let anchor_depth_jitter_m = if z_is_metric {
        let mean: f32 = zs.iter().sum::<f32>() / zs.len() as f32;
        let var: f32 = zs.iter().map(|&z| (z - mean).powi(2)).sum::<f32>() / zs.len() as f32;
        Some(var.sqrt())
    } else {
        None
    };

    // Median the per-frame measured shoulder spans. Only emit when at
    // least three frames produced a finite reading — single-frame
    // medians are trivially the value itself and don't earn the
    // robustness the median provides.
    let mut spans: Vec<f32> = samples.iter().filter_map(|s| s.shoulder_span_m).collect();
    let shoulder_span_m = if spans.len() >= 3 {
        Some(median_inplace(&mut spans))
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
        shoulder_span_m,
        // Range fields only get populated when the user opts into
        // the optional range-capture step from the AnchorDone prompt;
        // the anchor-only path leaves them None and the solver falls
        // back to its static per-axis sensitivity defaults.
        x_range_observed: None,
        z_range_observed: None,
        // Torso depth template is captured asynchronously by the
        // depth provider during the same collection window and
        // stitched back into the calibration *after* this aggregate
        // call by `poll_torso_template` — the anchor aggregate
        // happens on every provider (including rtmw3d-only, which
        // has no depth template), so we leave this `None` here and
        // let the depth-aware path overwrite it.
        torso_depth_template: None,
    }
}

/// Source-space 3D distance between LeftShoulder and RightShoulder
/// joints. Returns `None` when either shoulder is missing (off-frame,
/// low confidence, etc) or when the result is non-positive (rtmw3d-
/// only path emits z=0 across joints, so the 3D distance reduces to
/// the 2D shoulder span — still a reasonable body-scale proxy for
/// that path, but we gate on positive finite values for safety).
pub(super) fn measure_shoulder_span(pose: &SourceSkeleton) -> Option<f32> {
    use crate::asset::HumanoidBone;
    let l = pose.joints.get(&HumanoidBone::LeftShoulder)?.position;
    let r = pose.joints.get(&HumanoidBone::RightShoulder)?.position;
    let dx = l[0] - r[0];
    let dy = l[1] - r[1];
    let dz = l[2] - r[2];
    let span = (dx * dx + dy * dy + dz * dz).sqrt();
    if span.is_finite() && span > 0.0 {
        Some(span)
    } else {
        None
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
    if values.len().is_multiple_of(2) {
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
