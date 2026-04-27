//! Heuristic face-expression estimator for the CIGPose face block.
//!
//! CIGPose emits a 68-point iBUG face landmark block alongside its body
//! keypoints, but does not output blendshape weights. This module turns those
//! 2D landmarks into a small subset of VRM 1.0 expression weights —
//! `blinkLeft`, `blinkRight`, `blink`, `aa`, `happy`, `surprised` — using
//! geometry-only heuristics (Eye Aspect Ratio, Mouth Aspect Ratio, mouth
//! corner elevation, brow lift).
//!
//! A pure per-frame heuristic is too jittery for a face on a screen, so the
//! module also owns smoothing state. Three layers of "human-likeness":
//!
//! 1. **Asymmetric per-channel EMA** — each expression has its own attack /
//!    release time constants. Blinks snap closed and ease open; smiles attack
//!    slowly and linger.
//! 2. **Schmitt gating on blink** (input-side) — debounces noisy EAR readings
//!    so a single low-confidence frame does not register as a blink.
//! 3. **Deadband on happy / surprised** (input-side) — smiles below an
//!    emit-threshold are clamped to zero, preventing micro-jitter from making
//!    the avatar twitch. Surprise additionally requires *both* brow-lift
//!    *and* an open mouth, so a raised brow alone reads as neutral.
//!
//! Subject ↔ avatar mirroring: VulVATAR drives the avatar mirror-style (the
//! body mapping in `inference.rs:990-1026` swaps subject-left ↔ avatar-right).
//! The same convention applies here: iBUG's "right eye" (block-local 36..=41)
//! is the **subject's** right eye, which drives the avatar's *left* eye, i.e.
//! VRM `blinkLeft`.

use std::time::Instant;

use super::SourceExpression;

/// iBUG 68-point face block size. Callers must pass exactly this many points.
pub const FACE_LANDMARK_COUNT: usize = 68;

// --- Heuristic thresholds ----------------------------------------------------

/// Per-keypoint confidence below which a landmark is considered missing.
const MIN_KEYPOINT_CONFIDENCE: f32 = 0.2;

/// Eye Aspect Ratio mapping. Open eyes score ~0.30, closed eyes score
/// ~0.10–0.15 in iBUG geometry. We map [`EAR_OPEN`, `EAR_CLOSED`] linearly
/// to a closed-ness weight in [0, 1].
const EAR_OPEN: f32 = 0.28;
const EAR_CLOSED: f32 = 0.12;

/// Schmitt thresholds for blink, in *post-mapping* weight units (so they
/// compose with EAR_OPEN/EAR_CLOSED). The eye must close past `BLINK_ENTER`
/// to register as blinking, and must reopen past `BLINK_EXIT` to release.
/// The asymmetric thresholds prevent a noisy EAR oscillating around 0.5
/// from flickering the blink state.
const BLINK_ENTER: f32 = 0.7;
const BLINK_EXIT: f32 = 0.3;

/// Mouth Aspect Ratio thresholds. MAR ≈ inner-mouth height / outer-mouth
/// width. Closed mouth ~0.05, fully open ~0.4.
const MAR_CLOSED: f32 = 0.05;
const MAR_OPEN: f32 = 0.40;

/// Mouth corner elevation, expressed as `(mouth_center_y - corner_y) / face_width`.
/// Positive when corners sit *higher* than the mouth's vertical midpoint
/// (image y-up). Neutral faces sit slightly above zero; a clear smile
/// reaches ~0.05.
const SMILE_NEUTRAL: f32 = 0.0;
const SMILE_FULL: f32 = 0.05;

/// Brow elevation, expressed as `(brow_y - eye_y) / face_width`. Neutral
/// faces sit around 0.05; a "surprise" raise reaches ~0.10.
const BROW_NEUTRAL: f32 = 0.05;
const BROW_FULL: f32 = 0.10;

/// Deadband for happy / surprised. Below the emit threshold, the gated raw
/// value is clamped to zero. Once active, the gate only releases when the
/// raw value drops below `EXPR_RELEASE_THRESHOLD`. This is an output-side
/// Schmitt that suppresses sub-threshold microflicker in the heuristic.
const EXPR_EMIT_THRESHOLD: f32 = 0.05;
const EXPR_RELEASE_THRESHOLD: f32 = 0.02;

/// Surprise requires the mouth to be at least this open in addition to a
/// raised brow. Without this AND-gate, "raised brows" alone (a "huh?" look)
/// would register as surprise.
const SURPRISED_MOUTH_GATE: f32 = 0.30;

// --- Time constants (seconds) ------------------------------------------------

/// Blink attack — fast, so the eye snaps shut.
const TAU_BLINK_ATTACK: f32 = 0.040;
/// Blink release — slower, so the eye eases open like real lids.
const TAU_BLINK_RELEASE: f32 = 0.120;
const TAU_AA_ATTACK: f32 = 0.080;
const TAU_AA_RELEASE: f32 = 0.120;
const TAU_HAPPY_ATTACK: f32 = 0.200;
const TAU_HAPPY_RELEASE: f32 = 0.300;
const TAU_SURPRISED_ATTACK: f32 = 0.150;
const TAU_SURPRISED_RELEASE: f32 = 0.300;

/// Maximum dt used in the EMA. Long pauses (tab backgrounded, debugger break)
/// would otherwise produce α≈1 and snap to the raw value, defeating the
/// smoother on the first frame after resume.
const DT_MAX: f32 = 0.5;

/// Final emitted weight below this is rounded to zero, to avoid sending
/// thousands of effectively-zero `SourceExpression` entries every frame.
const ZERO_EPSILON: f32 = 0.001;

// --- iBUG 68-point block-local indices --------------------------------------

const IBUG_RIGHT_BROW: [usize; 5] = [17, 18, 19, 20, 21];
const IBUG_LEFT_BROW: [usize; 5] = [22, 23, 24, 25, 26];
const IBUG_RIGHT_EYE: [usize; 6] = [36, 37, 38, 39, 40, 41];
const IBUG_LEFT_EYE: [usize; 6] = [42, 43, 44, 45, 46, 47];
// Outer mouth corners and centers used for smile + face_width.
const IBUG_MOUTH_LEFT_CORNER: usize = 48;
const IBUG_MOUTH_RIGHT_CORNER: usize = 54;
const IBUG_MOUTH_UPPER_CENTER: usize = 51;
const IBUG_MOUTH_LOWER_CENTER: usize = 57;
// Inner mouth, used for MAR.
const IBUG_INNER_MOUTH_UPPER: usize = 62;
const IBUG_INNER_MOUTH_LOWER: usize = 66;

/// Smoothed weight state per channel.
#[derive(Default, Clone, Copy, Debug)]
struct SmoothedWeights {
    blink_left: f32,
    blink_right: f32,
    aa: f32,
    happy: f32,
    surprised: f32,
}

/// Stateful expression smoother. One instance per inference engine; reset
/// when a new tracking session starts so prior facial state does not bleed
/// into the new session.
pub struct FaceExpressionSmoother {
    last_update: Option<Instant>,
    /// Per-eye Schmitt state (true = currently in a "blink" episode).
    blink_left_active: bool,
    blink_right_active: bool,
    /// Output-side deadband state for happy / surprised.
    happy_active: bool,
    surprised_active: bool,
    smoothed: SmoothedWeights,
}

impl Default for FaceExpressionSmoother {
    fn default() -> Self {
        Self::new()
    }
}

impl FaceExpressionSmoother {
    pub fn new() -> Self {
        Self {
            last_update: None,
            blink_left_active: false,
            blink_right_active: false,
            happy_active: false,
            surprised_active: false,
            smoothed: SmoothedWeights::default(),
        }
    }

    /// Drop all temporal state. Call when a new tracker session starts —
    /// otherwise a frozen smile or open mouth could carry across.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Update with the iBUG 68-point face block and return current
    /// per-channel weights. Wall-clock dt is measured against the previous
    /// call. Returns an empty vec on the first call (dt=0 → α=0 → no change
    /// from the default-zero state).
    pub fn update(&mut self, face: &[(f32, f32, f32)]) -> Vec<SourceExpression> {
        let now = Instant::now();
        let dt = match self.last_update {
            Some(prev) => (now - prev).as_secs_f32().min(DT_MAX),
            None => 0.0,
        };
        self.last_update = Some(now);
        self.update_inner(face, dt)
    }

    /// Test-only entry point that injects dt directly. Production code uses
    /// [`update`].
    #[cfg(test)]
    pub fn update_with_dt(
        &mut self,
        face: &[(f32, f32, f32)],
        dt: f32,
    ) -> Vec<SourceExpression> {
        self.update_inner(face, dt.clamp(0.0, DT_MAX))
    }

    fn update_inner(&mut self, face: &[(f32, f32, f32)], dt: f32) -> Vec<SourceExpression> {
        let raw = compute_raw_weights(face).unwrap_or_default();

        // Schmitt-gate the blink raws (input-side debounce per eye).
        let raw_blink_left =
            schmitt_gate(&mut self.blink_left_active, raw.blink_left, BLINK_ENTER, BLINK_EXIT);
        let raw_blink_right = schmitt_gate(
            &mut self.blink_right_active,
            raw.blink_right,
            BLINK_ENTER,
            BLINK_EXIT,
        );

        // Surprise AND-gate: needs both brow lift AND open mouth.
        let raw_surprised = if raw.aa >= SURPRISED_MOUTH_GATE {
            raw.surprised
        } else {
            0.0
        };

        // Output-side deadband for happy / surprised.
        let raw_happy = deadband_gate(
            &mut self.happy_active,
            raw.happy,
            EXPR_EMIT_THRESHOLD,
            EXPR_RELEASE_THRESHOLD,
        );
        let raw_surprised = deadband_gate(
            &mut self.surprised_active,
            raw_surprised,
            EXPR_EMIT_THRESHOLD,
            EXPR_RELEASE_THRESHOLD,
        );

        // Apply per-channel asymmetric EMAs.
        ema_step(
            &mut self.smoothed.blink_left,
            raw_blink_left,
            dt,
            TAU_BLINK_ATTACK,
            TAU_BLINK_RELEASE,
        );
        ema_step(
            &mut self.smoothed.blink_right,
            raw_blink_right,
            dt,
            TAU_BLINK_ATTACK,
            TAU_BLINK_RELEASE,
        );
        ema_step(
            &mut self.smoothed.aa,
            raw.aa,
            dt,
            TAU_AA_ATTACK,
            TAU_AA_RELEASE,
        );
        ema_step(
            &mut self.smoothed.happy,
            raw_happy,
            dt,
            TAU_HAPPY_ATTACK,
            TAU_HAPPY_RELEASE,
        );
        ema_step(
            &mut self.smoothed.surprised,
            raw_surprised,
            dt,
            TAU_SURPRISED_ATTACK,
            TAU_SURPRISED_RELEASE,
        );

        // Pack into a SourceExpression vec. Use VRM 1.0 preset names. `blink`
        // (both eyes) is min(blinkLeft, blinkRight) so a wink does not
        // register as a full blink on avatars that only declare `blink`.
        let mut out = Vec::with_capacity(6);
        push_if_nonzero(&mut out, "blinkLeft", self.smoothed.blink_left);
        push_if_nonzero(&mut out, "blinkRight", self.smoothed.blink_right);
        push_if_nonzero(
            &mut out,
            "blink",
            self.smoothed.blink_left.min(self.smoothed.blink_right),
        );
        push_if_nonzero(&mut out, "aa", self.smoothed.aa);
        push_if_nonzero(&mut out, "happy", self.smoothed.happy);
        push_if_nonzero(&mut out, "surprised", self.smoothed.surprised);
        out
    }

    #[cfg(test)]
    fn smoothed_for_test(&self) -> SmoothedWeights {
        self.smoothed
    }
}

fn push_if_nonzero(out: &mut Vec<SourceExpression>, name: &str, weight: f32) {
    let w = weight.clamp(0.0, 1.0);
    if w >= ZERO_EPSILON {
        out.push(SourceExpression {
            name: name.to_string(),
            weight: w,
        });
    }
}

/// Schmitt trigger on a scalar input. Returns `raw` while active, `0.0`
/// otherwise. Single-frame noise that crosses `enter` once but fails to stay
/// elevated will toggle on for one frame and immediately back off — but the
/// EMA downstream cannot accumulate appreciable weight from one frame, so the
/// avatar stays put.
fn schmitt_gate(active: &mut bool, raw: f32, enter: f32, exit: f32) -> f32 {
    if *active {
        if raw < exit {
            *active = false;
            0.0
        } else {
            raw
        }
    } else if raw > enter {
        *active = true;
        raw
    } else {
        0.0
    }
}

/// Deadband on a scalar input. Below `enter`, returns 0; above, passes
/// through. Releases when raw drops below `exit`.
fn deadband_gate(active: &mut bool, raw: f32, enter: f32, exit: f32) -> f32 {
    if *active {
        if raw < exit {
            *active = false;
            0.0
        } else {
            raw
        }
    } else if raw >= enter {
        *active = true;
        raw
    } else {
        0.0
    }
}

/// Asymmetric exponential moving average step. Picks the attack time
/// constant when raw exceeds the smoothed value, the release time constant
/// otherwise, then computes `α = 1 - exp(-dt/τ)` and applies a standard
/// lerp. dt=0 leaves the smoothed value untouched.
fn ema_step(smoothed: &mut f32, raw: f32, dt: f32, tau_attack: f32, tau_release: f32) {
    if dt <= 0.0 {
        return;
    }
    let tau = if raw > *smoothed { tau_attack } else { tau_release };
    let alpha = 1.0 - (-dt / tau.max(1e-6)).exp();
    *smoothed += alpha * (raw - *smoothed);
}

/// Per-frame raw weights computed from face landmarks alone. Returns None
/// when the face is too low-confidence to read; callers should treat this
/// as `SmoothedWeights::default()` so smoothed values decay via release.
fn compute_raw_weights(face: &[(f32, f32, f32)]) -> Option<SmoothedWeights> {
    if face.len() != FACE_LANDMARK_COUNT {
        return None;
    }

    let face_width = face_width_from_landmarks(face)?;

    let blink_left = blink_weight_from_eye(face, &IBUG_LEFT_EYE);
    let blink_right = blink_weight_from_eye(face, &IBUG_RIGHT_EYE);

    let aa = mouth_aspect_ratio_weight(face);
    let happy = smile_weight(face, face_width);
    let surprised = surprise_weight(face, face_width);

    // VRM/avatar mirror convention: subject-left → avatar-right (and vice
    // versa). iBUG's LEFT_EYE is the subject's left, so it drives `blinkRight`.
    Some(SmoothedWeights {
        blink_left: blink_right.unwrap_or(0.0),
        blink_right: blink_left.unwrap_or(0.0),
        aa: aa.unwrap_or(0.0),
        happy: happy.unwrap_or(0.0),
        surprised: surprised.unwrap_or(0.0),
    })
}

/// Use the inter-eye distance as a face-width proxy. Both eye centers must
/// be confident; otherwise we have no scale and bail.
fn face_width_from_landmarks(face: &[(f32, f32, f32)]) -> Option<f32> {
    let l_center = eye_center(face, &IBUG_LEFT_EYE)?;
    let r_center = eye_center(face, &IBUG_RIGHT_EYE)?;
    let dx = l_center.0 - r_center.0;
    let dy = l_center.1 - r_center.1;
    let d = (dx * dx + dy * dy).sqrt();
    if d > 1e-4 { Some(d) } else { None }
}

fn eye_center(face: &[(f32, f32, f32)], idx: &[usize]) -> Option<(f32, f32)> {
    let mut x = 0.0;
    let mut y = 0.0;
    let mut n = 0;
    for &i in idx {
        let p = face[i];
        if p.2 >= MIN_KEYPOINT_CONFIDENCE {
            x += p.0;
            y += p.1;
            n += 1;
        }
    }
    if n == idx.len() {
        Some((x / n as f32, y / n as f32))
    } else {
        None
    }
}

/// Soukupová & Čech (2016) Eye Aspect Ratio — mapped to a closed-ness
/// weight. Returns None when not all 6 eye keypoints are confident, so the
/// caller treats the eye as "unknown" rather than "open".
fn blink_weight_from_eye(face: &[(f32, f32, f32)], idx: &[usize; 6]) -> Option<f32> {
    let p: [(f32, f32, f32); 6] = [
        face[idx[0]],
        face[idx[1]],
        face[idx[2]],
        face[idx[3]],
        face[idx[4]],
        face[idx[5]],
    ];
    if p.iter().any(|kp| kp.2 < MIN_KEYPOINT_CONFIDENCE) {
        return None;
    }
    let v1 = dist((p[1].0, p[1].1), (p[5].0, p[5].1));
    let v2 = dist((p[2].0, p[2].1), (p[4].0, p[4].1));
    let h = dist((p[0].0, p[0].1), (p[3].0, p[3].1)).max(1e-4);
    let ear = (v1 + v2) / (2.0 * h);
    // Closer to EAR_CLOSED → higher closed-ness weight.
    let span = (EAR_OPEN - EAR_CLOSED).max(1e-4);
    let weight = ((EAR_OPEN - ear) / span).clamp(0.0, 1.0);
    Some(weight)
}

fn mouth_aspect_ratio_weight(face: &[(f32, f32, f32)]) -> Option<f32> {
    let upper = face[IBUG_INNER_MOUTH_UPPER];
    let lower = face[IBUG_INNER_MOUTH_LOWER];
    let left = face[IBUG_MOUTH_LEFT_CORNER];
    let right = face[IBUG_MOUTH_RIGHT_CORNER];
    if upper.2 < MIN_KEYPOINT_CONFIDENCE
        || lower.2 < MIN_KEYPOINT_CONFIDENCE
        || left.2 < MIN_KEYPOINT_CONFIDENCE
        || right.2 < MIN_KEYPOINT_CONFIDENCE
    {
        return None;
    }
    let height = dist((upper.0, upper.1), (lower.0, lower.1));
    let width = dist((left.0, left.1), (right.0, right.1)).max(1e-4);
    let mar = height / width;
    let span = (MAR_OPEN - MAR_CLOSED).max(1e-4);
    Some(((mar - MAR_CLOSED) / span).clamp(0.0, 1.0))
}

/// Smile weight from mouth-corner elevation relative to the mouth center.
/// The face block uses image coordinates *after* `to_norm_y` (y-up), so a
/// smile lifts the corners' y above the mouth's vertical midpoint, producing
/// a positive `corner_lift`.
fn smile_weight(face: &[(f32, f32, f32)], face_width: f32) -> Option<f32> {
    let left = face[IBUG_MOUTH_LEFT_CORNER];
    let right = face[IBUG_MOUTH_RIGHT_CORNER];
    let upper = face[IBUG_MOUTH_UPPER_CENTER];
    let lower = face[IBUG_MOUTH_LOWER_CENTER];
    if left.2 < MIN_KEYPOINT_CONFIDENCE
        || right.2 < MIN_KEYPOINT_CONFIDENCE
        || upper.2 < MIN_KEYPOINT_CONFIDENCE
        || lower.2 < MIN_KEYPOINT_CONFIDENCE
    {
        return None;
    }
    let corner_y = (left.1 + right.1) * 0.5;
    let mouth_mid_y = (upper.1 + lower.1) * 0.5;
    let lift_norm = (corner_y - mouth_mid_y) / face_width.max(1e-4);
    let span = (SMILE_FULL - SMILE_NEUTRAL).max(1e-4);
    Some(((lift_norm - SMILE_NEUTRAL) / span).clamp(0.0, 1.0))
}

/// Surprise weight from brow elevation above the eye line. Image coords
/// are y-up, so raised brows have a higher y than the eyes.
fn surprise_weight(face: &[(f32, f32, f32)], face_width: f32) -> Option<f32> {
    let left_brow_y = brow_average_y(face, &IBUG_LEFT_BROW)?;
    let right_brow_y = brow_average_y(face, &IBUG_RIGHT_BROW)?;
    let left_eye = eye_center(face, &IBUG_LEFT_EYE)?;
    let right_eye = eye_center(face, &IBUG_RIGHT_EYE)?;
    let brow_y = (left_brow_y + right_brow_y) * 0.5;
    let eye_y = (left_eye.1 + right_eye.1) * 0.5;
    let lift_norm = (brow_y - eye_y) / face_width.max(1e-4);
    let span = (BROW_FULL - BROW_NEUTRAL).max(1e-4);
    Some(((lift_norm - BROW_NEUTRAL) / span).clamp(0.0, 1.0))
}

fn brow_average_y(face: &[(f32, f32, f32)], idx: &[usize]) -> Option<f32> {
    let mut y = 0.0;
    let mut n = 0;
    for &i in idx {
        let p = face[i];
        if p.2 >= MIN_KEYPOINT_CONFIDENCE {
            y += p.1;
            n += 1;
        }
    }
    if n == idx.len() {
        Some(y / n as f32)
    } else {
        None
    }
}

#[inline]
fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Construct a synthetic 68-point face with a configurable eye openness
    /// and mouth openness. Image y-up coordinates: eyes near y=0.6, mouth
    /// around y=-0.2, brows at y=0.75.
    fn make_face(eye_open: f32, mouth_open: f32, smile: f32, brow_lift: f32) -> Vec<(f32, f32, f32)> {
        let mut face = vec![(0.0, 0.0, 1.0); FACE_LANDMARK_COUNT];

        // Right eye (subject-right, image-left side, x ≈ -0.15).
        let r_eye_cx = -0.15;
        let r_eye_cy = 0.60;
        let half_w = 0.05;
        let half_h = 0.5 * eye_open * 0.06; // EAR scales with eye_open
        face[36] = (r_eye_cx - half_w, r_eye_cy, 1.0);
        face[37] = (r_eye_cx - half_w * 0.4, r_eye_cy + half_h, 1.0);
        face[38] = (r_eye_cx + half_w * 0.4, r_eye_cy + half_h, 1.0);
        face[39] = (r_eye_cx + half_w, r_eye_cy, 1.0);
        face[40] = (r_eye_cx + half_w * 0.4, r_eye_cy - half_h, 1.0);
        face[41] = (r_eye_cx - half_w * 0.4, r_eye_cy - half_h, 1.0);

        // Left eye (subject-left, image-right side).
        let l_eye_cx = 0.15;
        let l_eye_cy = 0.60;
        face[42] = (l_eye_cx - half_w, l_eye_cy, 1.0);
        face[43] = (l_eye_cx - half_w * 0.4, l_eye_cy + half_h, 1.0);
        face[44] = (l_eye_cx + half_w * 0.4, l_eye_cy + half_h, 1.0);
        face[45] = (l_eye_cx + half_w, l_eye_cy, 1.0);
        face[46] = (l_eye_cx + half_w * 0.4, l_eye_cy - half_h, 1.0);
        face[47] = (l_eye_cx - half_w * 0.4, l_eye_cy - half_h, 1.0);

        // Mouth: outer corners at x=±0.1, mouth at y=-0.2.
        let mouth_y = -0.2;
        let corner_offset = smile * 0.025;
        face[48] = (-0.10, mouth_y + corner_offset, 1.0); // left corner
        face[54] = (0.10, mouth_y + corner_offset, 1.0); // right corner
        face[51] = (0.0, mouth_y + 0.02, 1.0); // upper center (outer)
        face[57] = (0.0, mouth_y - 0.02, 1.0); // lower center (outer)
        // Inner mouth — vertical extent governed by mouth_open.
        let mouth_half = mouth_open * 0.05; // MAR ≈ 0.05 + 2*mouth_half/0.2 * width
        face[62] = (0.0, mouth_y + mouth_half, 1.0); // inner upper
        face[66] = (0.0, mouth_y - mouth_half, 1.0); // inner lower

        // Brows: at eye_y + ~0.05, lifted by brow_lift.
        let r_brow_y = r_eye_cy + 0.05 + brow_lift * 0.05;
        let l_brow_y = l_eye_cy + 0.05 + brow_lift * 0.05;
        for (idx, i) in (17..=21).enumerate() {
            face[i] = (r_eye_cx + (idx as f32 - 2.0) * 0.02, r_brow_y, 1.0);
        }
        for (idx, i) in (22..=26).enumerate() {
            face[i] = (l_eye_cx + (idx as f32 - 2.0) * 0.02, l_brow_y, 1.0);
        }

        face
    }

    #[test]
    fn closed_eyes_ramp_blink_over_multiple_frames() {
        // Eyes fully closed (EAR collapses).
        let face = make_face(0.0, 0.0, 0.0, 0.0);
        let mut s = FaceExpressionSmoother::new();
        // Prime with first frame to set last_update.
        s.update_with_dt(&face, 0.033);
        let after_one_frame = s.smoothed_for_test();
        // After one 33ms frame at τ_attack=40ms, α≈0.56. So weight should be
        // partway up but well below 1.
        assert!(
            after_one_frame.blink_left > 0.3 && after_one_frame.blink_left < 0.8,
            "blink_left should be partway up after 1 frame, got {}",
            after_one_frame.blink_left
        );

        // Run several more frames and confirm it converges.
        for _ in 0..10 {
            s.update_with_dt(&face, 0.033);
        }
        let converged = s.smoothed_for_test();
        assert!(
            converged.blink_left > 0.95,
            "blink_left should converge to ~1, got {}",
            converged.blink_left
        );
        assert!(converged.blink_right > 0.95);
    }

    #[test]
    fn single_frame_closed_does_not_peak_blink() {
        let mut s = FaceExpressionSmoother::new();
        // Open eyes for several frames first.
        let open = make_face(1.0, 0.0, 0.0, 0.0);
        for _ in 0..5 {
            s.update_with_dt(&open, 0.033);
        }

        // One spurious closed frame.
        let closed = make_face(0.0, 0.0, 0.0, 0.0);
        s.update_with_dt(&closed, 0.033);
        // Then back to open.
        for _ in 0..5 {
            s.update_with_dt(&open, 0.033);
        }
        let final_state = s.smoothed_for_test();
        assert!(
            final_state.blink_left < 0.2,
            "blink_left should fall back to near zero after a single noisy frame; got {}",
            final_state.blink_left
        );
    }

    #[test]
    fn open_mouth_drives_aa() {
        let mut s = FaceExpressionSmoother::new();
        let face = make_face(1.0, 1.0, 0.0, 0.0);
        for _ in 0..15 {
            s.update_with_dt(&face, 0.033);
        }
        let st = s.smoothed_for_test();
        assert!(st.aa > 0.6, "aa should rise; got {}", st.aa);
    }

    #[test]
    fn deadband_swallows_microflicker_for_happy() {
        let mut s = FaceExpressionSmoother::new();
        // Geometry: smile param scales the corner offset by 0.025, then the
        // smile_weight calculation divides by face_width (~0.30) and then by
        // SMILE_FULL (0.05). So raw_weight ≈ smile * 1.667. With smile=0.02
        // the raw weight is ~0.033, which is below EXPR_EMIT_THRESHOLD (0.05)
        // — the deadband should clamp it to zero. Drive many frames to
        // confirm the gate stays closed even under sustained sub-threshold
        // input.
        let face = make_face(1.0, 0.0, 0.02, 0.0);
        for _ in 0..30 {
            s.update_with_dt(&face, 0.033);
        }
        let st = s.smoothed_for_test();
        assert!(
            st.happy < ZERO_EPSILON,
            "sub-threshold smile should be gated to zero; got {}",
            st.happy
        );
    }

    #[test]
    fn face_lost_decays_smoothly() {
        let mut s = FaceExpressionSmoother::new();
        let smiling = make_face(0.0, 1.0, 1.0, 0.0); // closed eyes + open mouth + smile
        for _ in 0..30 {
            s.update_with_dt(&smiling, 0.033);
        }
        let peaked = s.smoothed_for_test();
        assert!(peaked.aa > 0.6);
        assert!(peaked.blink_left > 0.6);

        // Face confidence drops to zero — pass an empty slice.
        let lost: Vec<(f32, f32, f32)> = Vec::new();
        // Run a few frames of "face lost" — values should decay, not snap.
        s.update_with_dt(&lost, 0.033);
        let after_one = s.smoothed_for_test();
        // After 33ms with τ_release=120ms, α≈0.24. Values should drop ~25%.
        assert!(after_one.aa < peaked.aa);
        assert!(after_one.aa > peaked.aa * 0.5, "decay should be gradual");

        // Many more frames → should approach 0 but via the release τ.
        for _ in 0..60 {
            s.update_with_dt(&lost, 0.033);
        }
        let decayed = s.smoothed_for_test();
        assert!(decayed.aa < 0.05);
        assert!(decayed.blink_left < 0.05);
    }

    #[test]
    fn reset_clears_state() {
        let mut s = FaceExpressionSmoother::new();
        let face = make_face(0.0, 1.0, 0.0, 0.0);
        for _ in 0..30 {
            s.update_with_dt(&face, 0.033);
        }
        assert!(s.smoothed_for_test().aa > 0.5);
        s.reset();
        let cleared = s.smoothed_for_test();
        assert!(cleared.aa < ZERO_EPSILON);
        assert!(cleared.blink_left < ZERO_EPSILON);
        assert_eq!(s.last_update, None);
        assert!(!s.blink_left_active);
    }

    #[test]
    fn first_frame_returns_no_change() {
        let mut s = FaceExpressionSmoother::new();
        let face = make_face(0.0, 1.0, 0.0, 0.0);
        // dt=0 on the first call (last_update is None).
        s.update_with_dt(&face, 0.0);
        assert_eq!(s.smoothed_for_test().aa, 0.0);
    }

    #[test]
    fn surprise_requires_open_mouth() {
        let mut s = FaceExpressionSmoother::new();
        // Brows raised but mouth closed → surprise gate blocks.
        let brow_only = make_face(1.0, 0.0, 0.0, 1.0);
        for _ in 0..30 {
            s.update_with_dt(&brow_only, 0.033);
        }
        assert!(
            s.smoothed_for_test().surprised < ZERO_EPSILON,
            "brow alone should not register surprise"
        );

        // Mouth open + brows raised → surprise activates.
        let both = make_face(1.0, 1.0, 0.0, 1.0);
        for _ in 0..30 {
            s.update_with_dt(&both, 0.033);
        }
        assert!(
            s.smoothed_for_test().surprised > 0.5,
            "open mouth + raised brow should register surprise"
        );
    }

    #[test]
    fn schmitt_blocks_oscillation() {
        let mut s = FaceExpressionSmoother::new();
        // Half-closed input — sits between EAR_CLOSED and EAR_OPEN, weight ≈ 0.5,
        // which is below BLINK_ENTER (0.7). Schmitt should keep blink at 0.
        let half = make_face(0.5, 0.0, 0.0, 0.0);
        for _ in 0..30 {
            s.update_with_dt(&half, 0.033);
        }
        assert!(s.smoothed_for_test().blink_left < ZERO_EPSILON);
    }

    #[test]
    fn vrm_mirror_left_right_eyes() {
        // Subject's RIGHT eye (iBUG 36-41) is closed; LEFT eye open. After
        // VRM mirroring, this should drive `blinkLeft` (the avatar's left).
        let mut face = make_face(1.0, 0.0, 0.0, 0.0);
        // Collapse iBUG right eye height.
        let r_cy = 0.60;
        let half_w = 0.05;
        face[36] = (-0.15 - half_w, r_cy, 1.0);
        face[37] = (-0.15 - half_w * 0.4, r_cy, 1.0);
        face[38] = (-0.15 + half_w * 0.4, r_cy, 1.0);
        face[39] = (-0.15 + half_w, r_cy, 1.0);
        face[40] = (-0.15 + half_w * 0.4, r_cy, 1.0);
        face[41] = (-0.15 - half_w * 0.4, r_cy, 1.0);

        let mut s = FaceExpressionSmoother::new();
        for _ in 0..30 {
            s.update_with_dt(&face, 0.033);
        }
        let st = s.smoothed_for_test();
        assert!(
            st.blink_left > 0.6,
            "subject-right closed → blinkLeft; got blink_left={}, blink_right={}",
            st.blink_left,
            st.blink_right
        );
        assert!(
            st.blink_right < 0.1,
            "subject-left open → blinkRight stays low; got {}",
            st.blink_right
        );
    }
}
