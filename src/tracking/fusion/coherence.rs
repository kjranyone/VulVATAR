//! Anthropometric & kinematic coherence sanity filters for 2-D detections.
//!
//! Deep learning 2-D pose detectors (e.g. COCO-17 / RTMPose) routinely hallucinate
//! keypoints when limbs are occluded, cropped at the frame border, or self-occluded.
//! Because perspective projection strictly compresses (never expands) physical bone lengths,
//! an apparent 2-D bone length at depth `z` that exceeds the human anatomical limit
//! is physically impossible and signifies a hallucination.
//!
//! This module provides filters to cull impossible leg, arm, and duplicate wrist detections
//! before they can enter the Levenberg–Marquardt optimization loop.

use super::observe::{window_point, RawKp};

/// Maximum permissible perspective bone reaches (m), including scaling margin.
pub const THIGH_REACH_M: f64 = 0.65;
pub const SHIN_REACH_M: f64 = 0.65;
pub const UPPER_ARM_REACH_M: f64 = 0.50;
pub const FOREARM_REACH_M: f64 = 0.45;
pub const FULL_ARM_REACH_M: f64 = 0.85;

/// Sample median depth (m) at normalized image coordinate (nx, ny).
#[inline]
pub fn sample_depth_at(
    points: &[[f32; 3]],
    width: u32,
    height: u32,
    nx: f64,
    ny: f64,
) -> Option<f64> {
    if !(0.0..=1.0).contains(&nx) || !(0.0..=1.0).contains(&ny) {
        return None;
    }
    window_point(
        points,
        width,
        height,
        nx * width as f64,
        ny * height as f64,
        2,
        0.15,
        8.0,
    )
    .map(|p| p[2])
}

/// Projective bone length (m) between two keypoints at depth `z` (focal length `fx`).
#[inline]
pub fn projective_bone_length(
    a: &RawKp,
    b: &RawKp,
    z: f64,
    width: u32,
    height: u32,
    fx: f64,
) -> f64 {
    let dx = (a.nx - b.nx) as f64 * width as f64;
    let dy = (a.ny - b.ny) as f64 * height as f64;
    (dx * dx + dy * dy).sqrt() * z / fx.max(1.0)
}

/// Zero out a leg detection chain (knee, ankle, and toes; optionally hip).
pub fn zero_leg_chain(raw: &mut [RawKp], side: usize, cull_hip: bool) {
    let mut idxs = vec![13 + side, 15 + side];
    if cull_hip {
        idxs.push(11 + side);
    }
    idxs.extend([17 + 3 * side, 18 + 3 * side, 19 + 3 * side]);
    for i in idxs {
        if i < raw.len() {
            raw[i].score = 0.0;
        }
    }
}

/// Zero out an arm detection chain (wrist and hand-block; elbow left for upper arm if needed).
pub fn zero_arm_chain(raw: &mut [RawKp], side: usize) {
    let mut idxs = vec![9 + side]; // wrist
    let base = if side == 0 { 91 } else { 112 };
    idxs.extend(base..base + 21);
    for i in idxs {
        if i < raw.len() {
            raw[i].score = 0.0;
        }
    }
}

/// Sanitize lower body detections (pelvis width, thigh reach, shin reach, height ordering).
pub fn filter_leg_coherence(
    raw: &mut [RawKp],
    points: &[[f32; 3]],
    width: u32,
    height: u32,
    fx: f64,
    z_person: f64,
    model_pelvis_w: f64,
) {
    let in_band = |z: f64| (z - z_person).abs() <= 0.7;
    let z_at = |nx: f64, ny: f64| sample_depth_at(points, width, height, nx, ny);
    let bone_m = |a: &RawKp, b: &RawKp, z: f64| projective_bone_length(a, b, z, width, height, fx);

    // Pelvis-width coherence: the two hip joints are a rigid ~0.2 m apart
    // and projective distance only foreshortens with pelvic yaw. A pair far narrower
    // than the model's pelvis is a fabrication pair (e.g. clasped palms mistaken for hips).
    if raw[11].score > 0.0 && raw[12].score > 0.0 {
        let w_obs = bone_m(&raw[11], &raw[12], z_person);
        if w_obs < 0.65 * model_pelvis_w {
            zero_leg_chain(raw, 0, true);
            zero_leg_chain(raw, 1, true);
        }
    }

    // Shoulder-line reference for height checks.
    let (sh_y, torso_ref): (Option<f32>, f32) = {
        let (l, r) = (raw[5], raw[6]);
        if l.score > 0.3 && r.score > 0.3 {
            let sh_y = 0.5 * (l.ny + r.ny);
            let torso_ref = (sh_y - raw[0].ny).abs().max(0.05);
            (Some(sh_y), torso_ref)
        } else {
            (None, 0.0)
        }
    };
    let below_shoulder = |kp: &RawKp| -> bool {
        match sh_y {
            Some(sy) => kp.ny > sy - 0.3 * torso_ref,
            None => true,
        }
    };

    for side in 0..2 {
        if raw[11 + side].score <= 0.0 {
            continue;
        }
        let hip_z = z_at(raw[11 + side].nx as f64, raw[11 + side].ny as f64);
        let hip_foreign = hip_z.map(|z| !in_band(z)).unwrap_or(false);
        let hip_clamped = raw[11 + side].ny > 0.82;

        if raw[13 + side].score > 0.0 {
            let knee_z = z_at(raw[13 + side].nx as f64, raw[13 + side].ny as f64);
            let knee_foreign = knee_z.map(|z| !in_band(z)).unwrap_or(false);
            let thigh_m = bone_m(
                &raw[13 + side],
                &raw[11 + side],
                knee_z.or(hip_z).unwrap_or(z_person),
            );
            // A knee ABOVE the hip is anatomically impossible in seated/standing poses —
            // especially when the hip is clamped at the frame bottom.
            let knee_below_hip = if hip_clamped {
                raw[13 + side].ny >= raw[11 + side].ny - 0.02
            } else {
                raw[13 + side].ny >= raw[11 + side].ny - 0.15 * torso_ref.max(0.1)
            };
            let knee_ok = !knee_foreign
                && thigh_m <= THIGH_REACH_M
                && below_shoulder(&raw[13 + side])
                && knee_below_hip
                && (!hip_clamped || knee_z.is_some());

            if hip_foreign || !knee_ok {
                let cull_hip = hip_foreign || (hip_clamped && (!knee_ok || knee_z.is_none()));
                zero_leg_chain(raw, side, cull_hip);
                continue;
            }

            if raw[15 + side].score > 0.0 {
                let ankle_z = z_at(raw[15 + side].nx as f64, raw[15 + side].ny as f64);
                let ankle_foreign = ankle_z.map(|z| !in_band(z)).unwrap_or(false);
                let shin_m = bone_m(
                    &raw[15 + side],
                    &raw[13 + side],
                    ankle_z.or(knee_z).unwrap_or(z_person),
                );
                let ankle_below_knee = raw[15 + side].ny >= raw[13 + side].ny - 0.05;
                if ankle_foreign
                    || shin_m > SHIN_REACH_M
                    || !below_shoulder(&raw[15 + side])
                    || !ankle_below_knee
                {
                    zero_leg_chain(raw, side, false);
                }
            }
        } else if hip_foreign || hip_clamped {
            // A bottom-band hip with no knee detection at all has nothing corroborating it.
            zero_leg_chain(raw, side, true);
        }
    }
}

/// Sanitize upper body arm detections (upper arm reach, forearm reach, border clamping).
pub fn filter_arm_coherence(
    raw: &mut [RawKp],
    points: &[[f32; 3]],
    width: u32,
    height: u32,
    fx: f64,
    z_person: f64,
) {
    let in_band = |z: f64| (z - z_person).abs() <= 0.7;
    let z_at = |nx: f64, ny: f64| sample_depth_at(points, width, height, nx, ny);
    let bone_m = |a: &RawKp, b: &RawKp, z: f64| projective_bone_length(a, b, z, width, height, fx);

    for side in 0..2 {
        let sh_i = 5 + side;
        let el_i = 7 + side;
        let wr_i = 9 + side;
        let base = if side == 0 { 91 } else { 112 };
        if raw[sh_i].score <= 0.0 {
            // An uncorroborated border-clamped arm when shoulder is not visible is a phantom.
            for &idx in &[el_i, wr_i] {
                if raw[idx].score > 0.0 {
                    let clamped = !(0.02..0.98).contains(&raw[idx].nx)
                        || !(0.02..0.98).contains(&raw[idx].ny);
                    if clamped {
                        raw[idx].score = 0.0;
                    }
                }
            }
            let has_wrist = raw[wr_i].score > 0.0 || (base < raw.len() && raw[base].score > 0.0);
            if !has_wrist {
                for i in base..base + 21 {
                    if i < raw.len() {
                        raw[i].score = 0.0;
                    }
                }
            }
            continue;
        }
        let sh_z = z_at(raw[sh_i].nx as f64, raw[sh_i].ny as f64).unwrap_or(z_person);
        let mut el_ok = false;
        let mut el_z = sh_z;

        if raw[el_i].score > 0.0 {
            let z = z_at(raw[el_i].nx as f64, raw[el_i].ny as f64);
            let foreign = z.map(|zv| !in_band(zv)).unwrap_or(false);
            el_z = z.unwrap_or(sh_z);
            let upper_m = bone_m(&raw[el_i], &raw[sh_i], el_z.max(sh_z));
            // A bottom-clamped elbow (ny >= 0.98) is exiting the frame (hands under desk).
            // A desk in foreground easily provides an in-band depth, but that is the desk,
            // not the elbow. Reject bottom-clamped elbows unconditionally.
            let el_bottom_clamped = raw[el_i].ny >= 0.98;
            let el_side_clamped = !(0.02..0.98).contains(&raw[el_i].nx) || raw[el_i].ny <= 0.02;
            el_ok = !foreign
                && upper_m <= UPPER_ARM_REACH_M
                && !el_bottom_clamped
                && (!el_side_clamped || z.is_some());
            if !el_ok {
                raw[el_i].score = 0.0;
            }
        }

        let validate_wrist = |idx: usize, raw: &[RawKp]| -> bool {
            if raw[idx].score <= 0.0 {
                return false;
            }
            let wr_z = z_at(raw[idx].nx as f64, raw[idx].ny as f64);
            let foreign = wr_z.map(|zv| !in_band(zv)).unwrap_or(false);
            let z = wr_z.unwrap_or(el_z);
            // Bottom-clamped wrists (ny >= 0.98) are hands resting below the desk line;
            // depth there belongs to the desk surface. Reject unconditionally.
            let wr_bottom_clamped = raw[idx].ny >= 0.98;
            let wr_side_clamped = !(0.02..0.98).contains(&raw[idx].nx) || raw[idx].ny <= 0.02;
            if foreign || wr_bottom_clamped || (wr_side_clamped && wr_z.is_none()) {
                return false;
            }
            if el_ok {
                let fore_m = bone_m(&raw[idx], &raw[el_i], z.max(el_z));
                if fore_m > FOREARM_REACH_M {
                    return false;
                }
            } else {
                let full_m = bone_m(&raw[idx], &raw[sh_i], z.max(sh_z));
                if full_m > FULL_ARM_REACH_M {
                    return false;
                }
            }
            let opp_side = 1 - side;
            let sh_opp_i = 5 + opp_side;
            if raw[sh_opp_i].score > 0.0 {
                let dx_sh = raw[sh_opp_i].nx - raw[sh_i].nx;
                let dy_sh = raw[sh_opp_i].ny - raw[sh_i].ny;
                let len_sq = dx_sh * dx_sh + dy_sh * dy_sh;
                if len_sq > 1e-4 {
                    let dx_wr = raw[idx].nx - raw[sh_i].nx;
                    let dy_wr = raw[idx].ny - raw[sh_i].ny;
                    let proj = (dx_wr * dx_sh + dy_wr * dy_sh) / len_sq;
                    if proj > 1.05 || (!el_ok && proj > 0.65) {
                        return false;
                    }
                }
            }
            true
        };

        if raw[wr_i].score > 0.0 && !validate_wrist(wr_i, raw) {
            raw[wr_i].score = 0.0;
        }
        if base < raw.len() && raw[base].score > 0.0 && !validate_wrist(base, raw) {
            raw[base].score = 0.0;
        }

        // Hand fingers validation: fingers require at least one valid wrist detection
        // (body wrist or hand-block wrist) to anchor them.
        let has_any_wrist = raw[wr_i].score > 0.0 || (base < raw.len() && raw[base].score > 0.0);
        if !has_any_wrist {
            // Completely unanchored fingers: without any wrist detection,
            // wholebody hand keypoints are phantom hallucinations.
            for i in base..base + 21 {
                if i < raw.len() {
                    raw[i].score = 0.0;
                }
            }
        }
    }
}

/// Deduplicate wrist detections when both land on the same physical hand.
pub fn filter_duplicate_wrists(raw: &mut [RawKp], width: u32, height: u32) {
    let (lw, rw) = (raw[9], raw[10]);
    let close = {
        let dx = (lw.nx - rw.nx) * width as f32;
        let dy = (lw.ny - rw.ny) * height as f32;
        (dx * dx + dy * dy).sqrt() < 0.06 * width as f32
    };
    if close && lw.score > 0.0 && rw.score > 0.0 {
        let drop_left = lw.score < rw.score;
        let (wrist_i, base) = if drop_left { (9, 91) } else { (10, 112) };
        raw[wrist_i].score = 0.0;
        for k in raw.iter_mut().skip(base).take(21) {
            k.score = 0.0;
        }
    }
}
