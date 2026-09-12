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
    type ExprAccum = std::collections::HashMap<String, (f32, usize)>;
    type FaceAccums = (Vec<[f32; 3]>, Vec<[f32; 3]>);
    let (samples, expr_accum, (face_accum_mesh, face_accum_body), body_yaw_accum, q_accum): (
        Vec<AnchorSample>,
        ExprAccum,
        FaceAccums,
        Vec<f32>,
        Vec<Vec<[f32; 3]>>,
    ) = match &mut state.calibration.modal {
        CalibrationModalState::Collecting {
            samples,
            expr_accum,
            face_accum_mesh,
            face_accum_body,
            body_yaw_accum,
            q_accum,
            ..
        } => (
            std::mem::take(samples),
            std::mem::take(expr_accum),
            (
                std::mem::take(face_accum_mesh),
                std::mem::take(face_accum_body),
            ),
            std::mem::take(body_yaw_accum),
            std::mem::take(q_accum),
        ),
        _ => (
            Vec::new(),
            ExprAccum::new(),
            (Vec::new(), Vec::new()),
            Vec::new(),
            Vec::new(),
        ),
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

    let mut calibration = aggregate(&samples, mode);

    // A recalibration must not silently destroy face data the previous
    // calibration captured: the user re-running the *pose* capture
    // (e.g. after moving the camera) with the face momentarily
    // backlit / the FaceMesh model absent used to overwrite a good
    // neutral with zeros and an empty expression baseline — a silent
    // head-orientation regression. Carry the previous values forward
    // whenever this capture didn't gather enough of its own.
    let previous = state.app.tracking_calibration.pose.clone();

    // Average the resting expression weights into the per-person neutral
    // face baseline (sorted by name so the on-disk JSON is deterministic).
    // Empty when face tracking was off during the hold — `apply_calibration`
    // then leaves expressions untouched.
    let mut neutral: Vec<(String, f32)> = expr_accum
        .into_iter()
        .filter(|(_, (_, n))| *n > 0)
        .map(|(name, (sum, n))| (name, sum / n as f32))
        .collect();
    neutral.sort_by(|a, b| a.0.cmp(&b.0));
    if neutral.is_empty() {
        if let Some(prev) = previous.as_ref() {
            neutral = prev.neutral_expressions.clone();
        }
    }
    calibration.neutral_expressions = neutral;

    // Median resting head pose across the hold, independently per
    // estimator source → the person's neutral yaw/pitch/roll relative
    // to the camera as each estimator sees it. Median (not mean) so a
    // brief glance away during the hold doesn't skew the baseline.
    // Below 5 confident frames for a source the estimate is too thin —
    // keep the previous calibration's value for that source (or `None`)
    // rather than baking one noisy reading into every subsequent frame.
    calibration.neutral_face_ypr_mesh = neutral_from_accum(&face_accum_mesh)
        .or(previous.as_ref().and_then(|p| p.neutral_face_ypr_mesh));
    calibration.neutral_face_ypr_body = neutral_from_accum(&face_accum_body)
        .or(previous.as_ref().and_then(|p| p.neutral_face_ypr_body));

    // Median shoulder-line yaw → neutral body yaw (oblique camera
    // placement). Same carry-forward contract as the face
    // neutrals: a recapture that couldn't measure (non-metric frames,
    // torso wandering during the hold) keeps the previous value
    // instead of silently discarding a good one.
    calibration.neutral_body_yaw = neutral_body_yaw_from_accum(&body_yaw_accum)
        .or(previous.as_ref().and_then(|p| p.neutral_body_yaw));

    // Median solved joint state over the hold → the estimator posture
    // prior's q_neutral (‖q − q_neutral‖²). Same carry-forward contract
    // as the other neutrals: a recapture without a running fusion
    // provider (2D-only path) keeps the previous value instead of
    // silently discarding it.
    calibration.q_neutral = q_neutral_from_accum(&q_accum)
        .or_else(|| previous.as_ref().and_then(|p| p.q_neutral.clone()));

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
            state.project_status.profiles_dirty = true;
        }
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
                     (modal_state mismatch?); reporting failure without writing"
                );
                // Surface the failure through the normal `Done` pane
                // (1.5 s message, then auto-close) instead of vanishing
                // the modal with no feedback — the user needs to know
                // this attempt produced nothing so they retry.
                return CalibrationModalState::Done {
                    mode,
                    shown_at: now,
                    outcome: DoneOutcome::Insufficient {
                        samples_collected: 0,
                    },
                };
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
    // ends up negative and later consumers that treat the field as a
    // positive camera-forward distance reject the capture.
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
    // robustness the median provides. The plausibility gate is not
    // cosmetic: this scalar multiplies every anthropometric bone length
    // AND the whole-skeleton normalisation in `skeleton_from_depth`, so
    // storing garbage here is worse than storing nothing (the
    // uncalibrated path falls back to the provider's stabilised span).
    let mut spans: Vec<f32> = samples.iter().filter_map(|s| s.shoulder_span_m).collect();
    let shoulder_span_m = (spans.len() >= 3)
        .then(|| median_inplace(&mut spans))
        .filter(|s| {
            let ok = crate::tracking::shoulder_span_plausible(*s);
            if !ok {
                warn!(
                    "shoulder-span calibration rejected: {s:.3} m outside the plausible \
                     {:.2}–{:.2} m band — falling back to the auto-measured span",
                    crate::tracking::SHOULDER_SPAN_MIN_M,
                    crate::tracking::SHOULDER_SPAN_MAX_M,
                );
            }
            ok
        });

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
        // Set by the caller (`finalize_collection`) from the averaged
        // `expr_accum`; the aggregate over anchor samples has no view of
        // the expression accumulator, so it leaves the baseline empty.
        neutral_expressions: Vec::new(),
        // Likewise set by the caller from the medianed per-source
        // face accumulators.
        neutral_face_ypr_mesh: None,
        neutral_face_ypr_body: None,
        // Set by the caller from the medianed shoulder-line yaw
        // accumulator (metric captures only).
        neutral_body_yaw: None,
        q_neutral: None,
    }
}

/// The subject's 3D shoulder span (LeftShoulder ↔ RightShoulder) in
/// **real camera metres**.
///
/// The joint positions on a published `SourceSkeleton` are NOT metres:
/// `skeleton_from_depth` divides the metric skeleton by `mpsu` so every
/// subject's shoulder span lands on the fixed `TARGET_SRC_SHOULDER_SPAN`
/// constant (which is the point — it keeps the solver's
/// position-magnitude tunings in range). Measuring that span raw
/// therefore returns the same number for a child and a linebacker, and
/// storing it in `PoseCalibration::shoulder_span_m` — consumed as
/// metres, ~1.8× the anatomical mean — inflated every fallback bone
/// length and shrank the published skeleton out of the tuned range.
/// Multiplying by `mpsu` (`metres = source_units × mpsu`) undoes the
/// normalisation exactly and recovers the frame's own measurement.
///
/// Returns `None` when the frame has no metric backend (a monocular
/// span is in arbitrary units with no conversion available — there is
/// nothing honest to store), when either shoulder is missing
/// (off-frame, low confidence), or when the result is non-positive.
pub(super) fn measure_shoulder_span(pose: &SourceSkeleton) -> Option<f32> {
    use crate::asset::HumanoidBone;
    let mpsu = pose.metric_frame_info.as_ref()?.mpsu;
    if !mpsu.is_finite() || mpsu <= 1e-6 {
        return None;
    }
    let l = pose.joints.get(&HumanoidBone::LeftShoulder)?.position;
    let r = pose.joints.get(&HumanoidBone::RightShoulder)?.position;
    let dx = l[0] - r[0];
    let dy = l[1] - r[1];
    let dz = l[2] - r[2];
    let span_m = (dx * dx + dy * dy + dz * dz).sqrt() * mpsu;
    if span_m.is_finite() && span_m > 0.0 {
        Some(span_m)
    } else {
        None
    }
}

/// Minimum confident frames a face-pose source must contribute during
/// the hold before its neutral is trusted.
const FACE_NEUTRAL_MIN_SAMPLES: usize = 5;

/// Maximum per-channel median-absolute-deviation (radians, ≈ 8.6°)
/// across the hold before the head is considered "not actually held
/// still" for that source. The pose-match gate only checks the *body*,
/// so a user reading the on-screen instructions while glancing around
/// used to bake whatever direction they looked at into the neutral.
const FACE_NEUTRAL_MAX_MAD_RAD: f32 = 0.15;

/// Per-channel median of a face-pose accumulator, gated on sample
/// count and on hold stability (median absolute deviation of yaw and
/// pitch). Returns `None` when the source didn't gather enough
/// confident frames or the head visibly wandered during the hold —
/// callers then keep the previous calibration's value for that source.
/// Median solved joint state over the calibration hold — the posture
/// prior's `q_neutral`. Frames ride at the provider model's joint
/// count; if that count changed mid-hold (provider restart against a
/// different model version) the majority count wins and the minority
/// frames are dropped, so the median is never ragged. Below
/// `Q_NEUTRAL_MIN_SAMPLES` consistent frames the estimate is too thin
/// — `None` (the finalizer then carries the previous calibration's
/// value forward).
fn q_neutral_from_accum(accum: &[Vec<[f32; 3]>]) -> Option<Vec<[f32; 3]>> {
    const Q_NEUTRAL_MIN_SAMPLES: usize = 5;
    if accum.is_empty() {
        return None;
    }
    let mut counts: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for q in accum {
        *counts.entry(q.len()).or_default() += 1;
    }
    let (&n, _) = counts.iter().max_by_key(|(_, &c)| c)?;
    if n == 0 {
        return None;
    }
    let frames: Vec<&Vec<[f32; 3]>> = accum.iter().filter(|q| q.len() == n).collect();
    if frames.len() < Q_NEUTRAL_MIN_SAMPLES {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    for j in 0..n {
        let mut joint = [0.0f32; 3];
        for (k, comp) in joint.iter_mut().enumerate() {
            let mut vals: Vec<f32> = frames.iter().map(|q| q[j][k]).collect();
            *comp = median_inplace(&mut vals);
        }
        out.push(joint);
    }
    Some(out)
}

fn neutral_from_accum(accum: &[[f32; 3]]) -> Option<[f32; 3]> {
    if accum.len() < FACE_NEUTRAL_MIN_SAMPLES {
        return None;
    }
    let channel_median = |channel: usize| -> f32 {
        let mut v: Vec<f32> = accum.iter().map(|s| s[channel]).collect();
        median_inplace(&mut v)
    };
    let medians = [channel_median(0), channel_median(1), channel_median(2)];
    // Stability: MAD of yaw / pitch (roll wanders far less and is the
    // least damaging channel to get slightly wrong).
    for channel in 0..2 {
        let mut dev: Vec<f32> = accum
            .iter()
            .map(|s| (s[channel] - medians[channel]).abs())
            .collect();
        let mad = median_inplace(&mut dev);
        if mad > FACE_NEUTRAL_MAX_MAD_RAD {
            warn!(
                "face neutral capture rejected: channel {channel} MAD {mad:.3} rad exceeds {FACE_NEUTRAL_MAX_MAD_RAD:.3} — head not held still",
            );
            return None;
        }
    }
    Some(medians)
}

/// Maximum median-absolute-deviation (radians, ≈ 8.6°) of the
/// shoulder-line yaw across the hold. Same rationale as
/// [`FACE_NEUTRAL_MAX_MAD_RAD`]: the gate that started the capture
/// only proves the user WAS still — a torso that swivels mid-window
/// (turning to read the instructions) must not bake a transient
/// heading into every subsequent frame.
const BODY_YAW_MAX_MAD_RAD: f32 = 0.15;

/// Median shoulder-line yaw of the capture window, gated on sample
/// count, hold stability and the hard plausibility cap. `None` →
/// caller keeps the previous calibration's value (or stays
/// uncalibrated, which is a strict runtime no-op).
fn neutral_body_yaw_from_accum(accum: &[f32]) -> Option<f32> {
    if accum.len() < crate::tracking::BODY_YAW_MIN_SAMPLES {
        return None;
    }
    let mut v: Vec<f32> = accum.to_vec();
    let median = median_inplace(&mut v);
    let mut dev: Vec<f32> = accum.iter().map(|y| (y - median).abs()).collect();
    let mad = median_inplace(&mut dev);
    if mad > BODY_YAW_MAX_MAD_RAD {
        warn!(
            "body-yaw neutral capture rejected: MAD {mad:.3} rad exceeds {BODY_YAW_MAX_MAD_RAD:.3} — torso not held still",
        );
        return None;
    }
    if median.abs() > crate::tracking::BODY_YAW_MAX_RAD {
        warn!(
            "body-yaw neutral capture rejected: |{median:.3}| rad exceeds the {:.3} plausibility cap — likely an L/R swap, not camera geometry",
            crate::tracking::BODY_YAW_MAX_RAD,
        );
        return None;
    }
    Some(median)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn steady(n: usize, ypr: [f32; 3]) -> Vec<[f32; 3]> {
        vec![ypr; n]
    }

    /// A published metric skeleton whose shoulders sit `span_src` apart
    /// in normalised source units, carrying `mpsu` metres per unit —
    /// i.e. exactly what `skeleton_from_depth` emits.
    fn metric_pose(span_src: f32, mpsu: f32) -> SourceSkeleton {
        use crate::asset::HumanoidBone;
        use crate::tracking::source_skeleton::{CameraIntrinsics, MetricFrameInfo, SourceJoint};

        let mut sk = SourceSkeleton::default();
        for (bone, x) in [
            (HumanoidBone::LeftShoulder, span_src * 0.5),
            (HumanoidBone::RightShoulder, -span_src * 0.5),
        ] {
            sk.joints.insert(
                bone,
                SourceJoint {
                    position: [x, 0.0, 0.0],
                    confidence: 0.9,
                    metric_depth_m: None,
                },
            );
        }
        sk.metric_frame_info = Some(MetricFrameInfo {
            anchor_cam_m: [0.0, 0.0, 1.0],
            anchor_is_hip: true,
            mpsu,
            reference_span_m: span_src * mpsu,
            intrinsics: CameraIntrinsics {
                fx: 600.0,
                fy: 600.0,
                cx: 320.0,
                cy: 240.0,
                width: 640,
                height: 480,
            },
        });
        sk
    }

    #[cfg(feature = "inference")]
    #[test]
    fn shoulder_span_is_measured_in_metres_not_source_units() {
        // The published skeleton is normalised to the fixed
        // TARGET_SRC_SHOULDER_SPAN, so the raw distance carries no
        // per-subject information at all — only `× mpsu` recovers the
        // frame's actual metric measurement.
        const TARGET_SRC_SHOULDER_SPAN: f32 = 0.75;
        let true_span_m = 0.41;
        let mpsu = true_span_m / TARGET_SRC_SHOULDER_SPAN;
        let span = measure_shoulder_span(&metric_pose(TARGET_SRC_SHOULDER_SPAN, mpsu)).unwrap();
        assert!(
            (span - true_span_m).abs() < 1e-5,
            "expected {true_span_m} m, got {span} (source units leaked through?)"
        );
        assert!(
            crate::tracking::shoulder_span_plausible(span),
            "a real subject's span must survive the plausibility gate"
        );
    }

    #[cfg(feature = "inference")]
    #[test]
    fn shoulder_span_distinguishes_two_subjects() {
        // Regression pin for the fixed-point poison: both subjects
        // publish the SAME normalised span and differ only in mpsu, so
        // a measurement that ignores mpsu returns one constant for both.
        const TARGET_SRC_SHOULDER_SPAN: f32 = 0.75;
        let small = measure_shoulder_span(&metric_pose(
            TARGET_SRC_SHOULDER_SPAN,
            0.32 / TARGET_SRC_SHOULDER_SPAN,
        ))
        .unwrap();
        let large = measure_shoulder_span(&metric_pose(
            TARGET_SRC_SHOULDER_SPAN,
            0.48 / TARGET_SRC_SHOULDER_SPAN,
        ))
        .unwrap();
        assert!(
            large - small > 0.15,
            "spans collapsed to a constant ({small} vs {large})"
        );
    }

    #[test]
    fn shoulder_span_refuses_a_non_metric_frame() {
        // No mpsu → no honest conversion. Storing the raw source span
        // would hand `skeleton_from_depth` a ~0.75 "metre" reading.
        let mut sk = metric_pose(0.75, 0.5);
        sk.metric_frame_info = None;
        assert_eq!(measure_shoulder_span(&sk), None);
    }

    #[test]
    fn aggregate_refuses_to_store_an_implausible_span() {
        let s = |span: f32| AnchorSample {
            position: [0.0, 0.0, -0.8],
            confidence: 0.7,
            shoulder_span_m: Some(span),
        };
        // Pre-fix-shaped readings (the normalised source-space target)
        // must be dropped, not persisted: the uncalibrated fallback is
        // the provider's stabilised auto-span, which is strictly better
        // than a 1.8×-inflated constant.
        let bad = aggregate(&[s(0.75), s(0.74), s(0.76)], CalibrationMode::FullBody);
        assert_eq!(bad.shoulder_span_m, None);
        // A genuine metric measurement survives.
        let good = aggregate(&[s(0.40), s(0.41), s(0.42)], CalibrationMode::FullBody);
        assert_eq!(good.shoulder_span_m, Some(0.41));
    }

    #[test]
    fn plausibility_band_rejects_a_source_unit_span() {
        use crate::tracking::shoulder_span_plausible;
        // The exact value the pre-fix capture path wrote.
        assert!(!shoulder_span_plausible(0.75));
        // …and the one found in a real pre-fix profile.
        assert!(!shoulder_span_plausible(0.691_979_17));
        // Real human spans pass.
        for s in [0.30, 0.38, 0.45, 0.52] {
            assert!(shoulder_span_plausible(s), "{s} m should be accepted");
        }
        assert!(!shoulder_span_plausible(f32::NAN));
    }

    #[test]
    fn q_neutral_medians_a_steady_hold() {
        let frame = vec![[0.1, 0.2, 0.3], [-0.4, 0.0, 0.5]];
        let accum: Vec<Vec<[f32; 3]>> = (0..6).map(|_| frame.clone()).collect();
        assert_eq!(q_neutral_from_accum(&accum), Some(frame));
    }

    #[test]
    fn q_neutral_needs_minimum_samples() {
        let frame = vec![[0.1, 0.2, 0.3]];
        let accum: Vec<Vec<[f32; 3]>> = (0..4).map(|_| frame.clone()).collect();
        assert_eq!(q_neutral_from_accum(&accum), None);
    }

    #[test]
    fn q_neutral_drops_minority_joint_counts() {
        // 6 frames at 2 joints, 2 frames at 3 joints (provider restart
        // mid-hold): the majority count wins, minority frames are
        // excluded from the median rather than ragged-indexed.
        let majority = vec![[0.1, 0.2, 0.3], [-0.4, 0.0, 0.5]];
        let minority = vec![[9.0; 3], [9.0; 3], [9.0; 3]];
        let accum: Vec<Vec<[f32; 3]>> = (0..6)
            .map(|_| majority.clone())
            .chain((0..2).map(|_| minority.clone()))
            .collect();
        assert_eq!(q_neutral_from_accum(&accum), Some(majority));
    }

    #[test]
    fn q_neutral_rejects_empty_hold() {
        assert_eq!(q_neutral_from_accum(&[]), None);
    }

    #[test]
    fn neutral_from_accum_needs_minimum_samples() {
        assert_eq!(neutral_from_accum(&steady(4, [0.1, 0.2, 0.0])), None);
        assert_eq!(
            neutral_from_accum(&steady(5, [0.1, 0.2, 0.0])),
            Some([0.1, 0.2, 0.0])
        );
    }

    #[test]
    fn body_yaw_accum_medians_a_steady_hold() {
        let accum = [0.48, 0.50, 0.52, 0.49, 0.51];
        let yaw = neutral_body_yaw_from_accum(&accum).unwrap();
        assert!((yaw - 0.50).abs() < 1e-6);
    }

    #[test]
    fn body_yaw_accum_needs_minimum_samples() {
        assert_eq!(neutral_body_yaw_from_accum(&[0.5; 4]), None);
        assert!(neutral_body_yaw_from_accum(&[0.5; 5]).is_some());
    }

    #[test]
    fn body_yaw_accum_rejects_a_swiveling_torso() {
        // Median-stable but wildly spread readings (user turned to
        // read the instructions mid-window): MAD gate must refuse.
        let accum = [0.0, 0.5, -0.5, 0.6, -0.6, 0.1];
        assert_eq!(neutral_body_yaw_from_accum(&accum), None);
    }

    #[test]
    fn range_folding_rotates_samples_not_the_box() {
        // Spec (docs/calibration-ux.md, "Neutral body yaw" →
        // movement-range interaction): with a neutral body yaw active, each range
        // sample is rotated into the body frame BEFORE min/max
        // folding. Rotating the already-folded camera-frame box is
        // NOT equivalent (an axis-aligned box is not
        // rotation-equivariant) — this pins the difference so a
        // future refactor can't quietly swap the order.
        use crate::tracking::rotate_xz;
        let theta = 0.6_f32;
        let samples = [
            [0.30_f32, 0.0, -1.20],
            [-0.25, 0.0, -1.70],
            [0.10, 0.0, -1.95],
        ];
        let fold = |pts: &[[f32; 3]]| {
            let mut x = (f32::INFINITY, f32::NEG_INFINITY);
            let mut z = (f32::INFINITY, f32::NEG_INFINITY);
            for p in pts {
                x = (x.0.min(p[0]), x.1.max(p[0]));
                z = (z.0.min(p[2]), z.1.max(p[2]));
            }
            (x.1 - x.0, z.1 - z.0)
        };
        let rotated: Vec<[f32; 3]> = samples.iter().map(|p| rotate_xz(*p, theta)).collect();
        let (fold_rot_x, fold_rot_z) = fold(&rotated);
        // "Rotate the folded box" degenerates to the raw extents (a
        // peak-to-peak delta has no orientation to rotate).
        let (box_x, box_z) = fold(&samples);
        assert!(
            (fold_rot_x - box_x).abs() > 1e-3 || (fold_rot_z - box_z).abs() > 1e-3,
            "fold-of-rotated must differ from the raw-axis box for an oblique sweep"
        );
    }

    #[test]
    fn body_yaw_accum_rejects_implausible_magnitude() {
        // A steady ~86° reading is an L/R swap, not camera geometry —
        // reject outright rather than clamping to 60° (a clamped value
        // would apply a wrong-magnitude rotation every frame).
        let accum = [1.5; 6];
        assert_eq!(neutral_body_yaw_from_accum(&accum), None);
        // Negative side too.
        let accum = [-1.5; 6];
        assert_eq!(neutral_body_yaw_from_accum(&accum), None);
    }

    #[test]
    fn neutral_from_accum_median_averages_even_middle_pair() {
        // Even-length accumulator: the median must average the two
        // middle values, not grab the upper one (the old inline
        // `v[len / 2]` did the latter and was a second, divergent
        // median implementation next to `median_inplace`).
        let accum = vec![
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ];
        let n = neutral_from_accum(&accum).unwrap();
        assert!(
            (n[0] - 0.25).abs() < 1e-6,
            "expected mid-pair average, got {}",
            n[0]
        );
    }

    #[test]
    fn neutral_from_accum_rejects_wandering_head() {
        // Half the hold spent looking 0.5 rad away: MAD blows past the
        // stability gate, so no neutral must be produced -- the
        // previous calibration (or None) wins over a garbage capture.
        let mut accum = steady(6, [0.0, 0.0, 0.0]);
        accum.extend(steady(6, [0.5, 0.4, 0.0]));
        assert_eq!(neutral_from_accum(&accum), None);
    }

    #[test]
    fn neutral_from_accum_tolerates_brief_glance() {
        // A short glance away is exactly what the median is for: 2 of
        // 12 frames off-target must not reject nor skew the result.
        let mut accum = steady(10, [0.1, -0.05, 0.02]);
        accum.extend(steady(2, [0.8, 0.3, 0.0]));
        let n = neutral_from_accum(&accum).unwrap();
        assert!((n[0] - 0.1).abs() < 1e-6);
        assert!((n[1] - -0.05).abs() < 1e-6);
    }
}
