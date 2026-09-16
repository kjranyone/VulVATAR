//! State-driven hand crops → MediaPipe Hand Landmarker (21 keypoints,
//! 2.5-D screen + wrist-relative metric 3-D) → estimator observations.
//!
//! The body detector's hand block is coarse (≈ 20 px σ on a 1280-wide
//! frame); a dedicated crop around the *predicted* hand runs the hand
//! model at native resolution, giving finger keypoints good to a couple of
//! pixels plus a wrist-relative 3-D layout that resolves finger curl the
//! 2-D terms cannot. Crops are cut from the prediction, so no palm
//! detector is needed; a hand whose prediction leaves the frame is simply
//! not cropped (the estimator then relaxes it).
//!
//! Model: `models/mediapipe_hand_landmark.onnx` (OpenCV's conversion of
//! MediaPipe `hand_landmark_full`), input `(1,224,224,3)` NHWC `[0,1]`,
//! outputs `Identity` (63: screen x,y,z in crop px), `Identity_1`
//! (presence), `Identity_2` (handedness), `Identity_3` (63: world metres,
//! wrist-relative, camera-aligned axes).

use std::path::Path;

use log::{error, info};
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;

use crate::tracking::detector::session::build_session_cpu_only;

use super::estimator::{AngleObs, Intrinsics, Kp2d, Kp3d, ModelPoint};
use super::math::*;
use super::model::*;

pub const HAND_KP: usize = 21;
const INPUT: usize = 224;

#[derive(Clone, Debug, Default)]
pub struct HandResult {
    /// Frame-pixel x, y and crop-scale relative z per landmark.
    pub px: [[f32; 3]; HAND_KP],
    /// Wrist-relative metres (camera-aligned).
    pub world: [[f32; 3]; HAND_KP],
    pub presence: f32,
    pub handedness: f32,
    /// Crop used `(x, y, size)` in frame pixels.
    pub crop: (f32, f32, f32),
    /// Which crop candidate produced this result: 0 = previous frame's
    /// locked hand re-cropped, 1 = the body detector's hand block,
    /// 2 = the model prediction. Provenance for wrist-obs noise
    /// attribution (a prediction crop over desk pixels can hallucinate
    /// a coherent hand); the provider sets it.
    pub src: u8,
}

pub struct HandLandmarker {
    session: Session,
    input_name: String,
    output_names: Vec<String>,
    tensor: Array4<f32>,
}

impl HandLandmarker {
    pub fn try_from_models_dir(dir: impl AsRef<Path>) -> Result<Option<Self>, String> {
        let path = dir.as_ref().join("mediapipe_hand_landmark.onnx");
        if !path.is_file() {
            return Ok(None);
        }
        let (session, _backend) =
            build_session_cpu_only(&path.to_string_lossy(), 2, "MediaPipeHand-CPU")?;
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input_1".to_string());
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        if output_names.len() < 4 {
            return Err(format!(
                "hand model declared {} outputs, expected 4",
                output_names.len()
            ));
        }
        info!("Hand landmarker ready ({})", path.display());
        Ok(Some(Self {
            session,
            input_name,
            output_names,
            tensor: Array4::<f32>::zeros((1, INPUT, INPUT, 3)),
        }))
    }

    /// Run on a square crop `(x0, y0, size)` of the RGB frame (bilinear
    /// resample into 224×224). Returns landmarks in frame pixels.
    pub fn estimate(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
        crop: (f32, f32, f32),
    ) -> Option<HandResult> {
        let (x0, y0, size) = crop;
        if size < 8.0 || rgb.len() < (width as usize) * (height as usize) * 3 {
            return None;
        }
        let scale = size / INPUT as f32;
        let (w, h) = (width as i64, height as i64);
        for ty in 0..INPUT {
            let sy = y0 + (ty as f32 + 0.5) * scale - 0.5;
            for tx in 0..INPUT {
                let sx = x0 + (tx as f32 + 0.5) * scale - 0.5;
                let (fx, fy) = (sx.floor(), sy.floor());
                let (ax, ay) = (sx - fx, sy - fy);
                let (ix, iy) = (fx as i64, fy as i64);
                let mut acc = [0.0f32; 3];
                let mut wsum = 0.0f32;
                for (dy, wy) in [(0i64, 1.0 - ay), (1, ay)] {
                    for (dx, wx) in [(0i64, 1.0 - ax), (1, ax)] {
                        let (px, py) = (ix + dx, iy + dy);
                        if px < 0 || py < 0 || px >= w || py >= h {
                            continue;
                        }
                        let wgt = wx * wy;
                        let i = ((py * w + px) * 3) as usize;
                        acc[0] += rgb[i] as f32 * wgt;
                        acc[1] += rgb[i + 1] as f32 * wgt;
                        acc[2] += rgb[i + 2] as f32 * wgt;
                        wsum += wgt;
                    }
                }
                let inv = if wsum > 0.0 {
                    1.0 / (255.0 * wsum)
                } else {
                    0.0
                };
                self.tensor[(0, ty, tx, 0)] = acc[0] * inv;
                self.tensor[(0, ty, tx, 1)] = acc[1] * inv;
                self.tensor[(0, ty, tx, 2)] = acc[2] * inv;
            }
        }
        let input = match TensorRef::from_array_view(&self.tensor) {
            Ok(v) => v,
            Err(e) => {
                error!("hand: tensor ref failed: {e}");
                return None;
            }
        };
        let outputs = match self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
        {
            Ok(o) => o,
            Err(e) => {
                error!("hand: run failed: {e}");
                return None;
            }
        };
        let extract = |name: &str, expected: usize| -> Option<Vec<f32>> {
            let v = outputs.get(name)?;
            let (_, data) = v.try_extract_tensor::<f32>().ok()?;
            if data.len() < expected {
                None
            } else {
                Some(data[..expected].to_vec())
            }
        };
        let screen = extract(&self.output_names[0], 63)?;
        let presence = extract(&self.output_names[1], 1)
            .map(|v| v[0])
            .unwrap_or(0.0);
        let handedness = extract(&self.output_names[2], 1)
            .map(|v| v[0])
            .unwrap_or(0.0);
        let world = extract(&self.output_names[3], 63)?;
        drop(outputs);
        // The converted checkpoint emits probabilities (an empty crop reads
        // ≈ 0.0, a hand ≈ 0.7–1.0), not logits.
        let mut out = HandResult {
            presence: presence.clamp(0.0, 1.0),
            handedness: handedness.clamp(0.0, 1.0),
            crop,
            ..Default::default()
        };
        for i in 0..HAND_KP {
            out.px[i] = [
                x0 + screen[i * 3] * scale,
                y0 + screen[i * 3 + 1] * scale,
                screen[i * 3 + 2] * scale,
            ];
            out.world[i] = [world[i * 3], world[i * 3 + 1], world[i * 3 + 2]];
        }
        Some(out)
    }
}

/// Predicted square crop for one hand from the model prediction: centred
/// on the wrist/MCP centroid, sized to the projected hand span × 2.4 (min
/// `min_px`). `None` when the hand is outside the frame or too small.
pub fn predicted_hand_crop(
    h: &Humanoid,
    fk: &Fk,
    hand: usize,
    intr: &Intrinsics,
    min_px: f64,
) -> Option<(f32, f32, f32)> {
    let wrist = if hand == 0 { h.j.l_wrist } else { h.j.r_wrist };
    let mut pts = vec![fk.t[wrist]];
    for f in 0..5 {
        pts.push(fk.t[h.j.finger[hand][f][0]]);
        pts.push(fk.site[h.s.tip[hand][f]]);
    }
    let mut umin = f64::INFINITY;
    let mut umax = f64::NEG_INFINITY;
    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    for p in &pts {
        let uv = intr.project(*p)?;
        umin = umin.min(uv[0]);
        umax = umax.max(uv[0]);
        vmin = vmin.min(uv[1]);
        vmax = vmax.max(uv[1]);
    }
    let cx = 0.5 * (umin + umax);
    let cy = 0.5 * (vmin + vmax);
    let span = (umax - umin).max(vmax - vmin);
    let size = (span * 1.6).max(min_px);
    // Require the centre inside the frame with some margin.
    if cx < -size * 0.2
        || cy < -size * 0.2
        || cx > intr.width + size * 0.2
        || cy > intr.height + size * 0.2
    {
        return None;
    }
    Some((
        (cx - size * 0.5) as f32,
        (cy - size * 0.5) as f32,
        size as f32,
    ))
}

/// Crop from the body detector's hand block (wrist + 20 landmarks, COCO
/// 91.. / 112..): bbox of the confident landmarks, padded. `None` when too
/// few landmarks are confident or the box is tiny.
pub fn detector_hand_crop(
    kps: &[(f32, f32, f32)],
    hand: usize,
    width: u32,
    height: u32,
    min_score: f32,
    min_px: f64,
) -> Option<(f32, f32, f32)> {
    let base = if hand == 0 { 91 } else { 112 };
    let mut umin = f64::INFINITY;
    let mut umax = f64::NEG_INFINITY;
    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    let mut n = 0;
    for i in base..(base + HAND_KP).min(kps.len()) {
        let (nx, ny, sc) = kps[i];
        if sc < min_score || !(0.0..=1.0).contains(&nx) || !(0.0..=1.0).contains(&ny) {
            continue;
        }
        let u = nx as f64 * width as f64;
        let v = ny as f64 * height as f64;
        umin = umin.min(u);
        umax = umax.max(u);
        vmin = vmin.min(v);
        vmax = vmax.max(v);
        n += 1;
    }
    if n < 6 {
        return None;
    }
    let cx = 0.5 * (umin + umax);
    let cy = 0.5 * (vmin + vmax);
    let span = (umax - umin).max(vmax - vmin);
    if span < 12.0 {
        return None;
    }
    let size = (span * 1.8).max(min_px);
    Some((
        (cx - size * 0.5) as f32,
        (cy - size * 0.5) as f32,
        size as f32,
    ))
}

/// Crop seeded from the BODY wrist/elbow keypoints (COCO 7/9 left,
/// 8/10 right) — the wholebody hand blocks [`detector_hand_crop`]
/// relies on are gone with YOLO26, so this is the only detector-side
/// seed left for hand acquisition. The centre extrapolates past the
/// wrist along the elbow→wrist direction (the hand extends beyond the
/// wrist joint), sized from the projected forearm length; without a
/// confident elbow it falls back to a fixed-size crop centred on the
/// wrist. `None` when the wrist is unconfident / out of frame.
pub fn wrist_hand_crop(
    kps: &[(f32, f32, f32)],
    hand: usize,
    width: u32,
    height: u32,
    min_score: f32,
    min_px: f64,
) -> Option<(f32, f32, f32)> {
    let (el, wr) = if hand == 0 { (7usize, 9usize) } else { (8, 10) };
    let px = |i: usize| -> Option<(f64, f64)> {
        let (nx, ny, sc) = kps.get(i)?;
        if *sc < min_score || !(0.0..=1.0).contains(nx) || !(0.0..=1.0).contains(ny) {
            return None;
        }
        Some((*nx as f64 * width as f64, *ny as f64 * height as f64))
    };
    let (wu, wv) = px(wr)?;
    let (cu, cv, size) = match px(el) {
        Some((eu, ev)) => {
            let du = wu - eu;
            let dv = wv - ev;
            let fore = (du * du + dv * dv).sqrt();
            let size = (fore * 1.2).clamp(min_px, 320.0);
            if fore < 1.0 {
                (wu, wv, size)
            } else {
                // Centre a quarter crop-size past the wrist along the
                // forearm direction.
                (
                    wu + du / fore * size * 0.25,
                    wv + dv / fore * size * 0.25,
                    size,
                )
            }
        }
        // No confident elbow (bent arms hide it on desk/palms sessions):
        // enlarge generously instead of extrapolating — the hand may
        // extend in any direction from the wrist joint, and a wrist-tight
        // crop only ever sees the cuff.
        None => (wu, wv, min_px * 2.4),
    };
    if cu < -size * 0.2
        || cv < -size * 0.2
        || cu > width as f64 + size * 0.2
        || cv > height as f64 + size * 0.2
    {
        return None;
    }
    Some((
        (cu - size * 0.5) as f32,
        (cv - size * 0.5) as f32,
        size as f32,
    ))
}

/// Turn a hand result into estimator observations: 21 2-D keypoints
/// (σ from crop scale) and, when an absolute wrist position is known,
/// absolute 3-D points for the finger joints from the wrist-relative
/// world layout. `hand` = 0 left, 1 right (model side the crop was for).
#[allow(clippy::too_many_arguments)]
pub fn hand_observations(
    h: &Humanoid,
    hand: usize,
    res: &HandResult,
    wrist_abs: Option<V3>,
    width: u32,
    height: u32,
    // `sigma_scale` > 1 when the result is suspect (e.g. its handedness
    // contradicts the slot it was cropped for).
    sigma_scale: f64,
    out2d: &mut Vec<Kp2d>,
    out3d: &mut Vec<Kp3d>,
    out_ang: &mut Vec<AngleObs>,
) {
    let scale = res.crop.2 as f64 / INPUT as f64;
    // Landmark noise measured live (2026-09-14, MCP keypoints on a held
    // desk hand) is 15–19 px rms frame-to-frame at 640w — far above this
    // σ — but inflating it destabilises the torso basin (same replay:
    // ×4 flips torso yaw mean −23.7°→+59.5°; ×8 → −20.5°). The hand
    // terms are load-bearing beyond the hand, so the calibrated crop-space
    // base stays; `VULVATAR_HAND_SIGMA_SCALE` multiplies it for future
    // work (the right fix is per-finger angle observations, not global
    // inflation).
    let sigma_scale_env = std::env::var("VULVATAR_HAND_SIGMA_SCALE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(1.0);
    // Fingers-only multiplier (VULVATAR_FINGER_SIGMA_SCALE): inflating the
    // WRIST landmark σ destabilises the torso basin (it anchors the arm
    // chain → clavicle → spine yaw; measured −23.7°→+59.5° flip), so the
    // wrist keeps the calibrated σ while the finger landmarks — measured
    // 15–33 px rms in poor-visibility sessions vs ~1 px σ — loosen alone.
    let finger_scale_env = std::env::var("VULVATAR_FINGER_SIGMA_SCALE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(1.0);
    let sigma_px = (2.0 * scale).max(1.0)
        * (1.0 + 2.0 * (1.0 - res.presence as f64))
        * sigma_scale
        * sigma_scale_env;
    let wrist = if hand == 0 { h.j.l_wrist } else { h.j.r_wrist };
    let mut points: [Option<ModelPoint>; HAND_KP] = [None; HAND_KP];
    points[0] = Some(ModelPoint::Joint(wrist));
    for f in 0..5 {
        let fj = h.j.finger[hand][f];
        points[1 + f * 4] = Some(ModelPoint::Joint(fj[0]));
        points[2 + f * 4] = Some(ModelPoint::Joint(fj[2]));
        points[3 + f * 4] = Some(ModelPoint::Joint(fj[3]));
        points[4 + f * 4] = Some(ModelPoint::Site(h.s.tip[hand][f]));
    }
    for i in 0..HAND_KP {
        let Some(pt) = points[i] else { continue };
        let (u, v) = (res.px[i][0] as f64, res.px[i][1] as f64);
        if !u.is_finite()
            || !v.is_finite()
            || u < 0.0
            || v < 0.0
            || u >= width as f64
            || v >= height as f64
        {
            continue;
        }
        out2d.push(Kp2d {
            point: pt,
            u,
            v,
            sigma: if i == 0 { sigma_px } else { sigma_px * finger_scale_env },
        });
        if let Some(w) = wrist_abs {
            if i > 0 {
                // NOTE: currently unreachable from the provider — the
                // converted checkpoint's "world" output is not metric and
                // its proportions proved too noisy for Cartesian terms
                // even after per-frame re-scaling (measured wrist snap
                // 1→3–4, max jump 0.48 m). The intended revival is as
                // finger-joint ANGLE observations from landmark triples
                // (invariant to scale and to the wrist anchor), not as
                // these absolute points.
                let rel = [
                    res.world[i][0] as f64,
                    res.world[i][1] as f64,
                    res.world[i][2] as f64,
                ];
                out3d.push(Kp3d {
                    point: pt,
                    p: add(w, rel),
                    sigma: 0.05 * (1.0 + 2.0 * (1.0 - res.presence as f64)) * sigma_scale,
                    lat_scale: 1.0,
                });
            }
        }
    }
    // Inter-phalanx angle observations from landmark triples — the
    // scale/wrist-invariant core of finger curl. Per finger: the angle at
    // the MCP (wrist–MCP–PIP), at the PIP (MCP–PIP–DIP) and at the DIP
    // (PIP–DIP–tip). σ propagates the landmark σ through the angle
    // (dθ ≈ σ·(1/d₁ + 1/d₂)), floored so a near-degenerate triple can't
    // become an infinitely sharp constraint. The thumb's CMC angle is
    // skipped (its model triple is not co-planar with the landmark one).
    let push_angle = |out_ang: &mut Vec<AngleObs>,
                      (ia, iv, ib): (usize, usize, usize),
                      (pa, pv, pb): (ModelPoint, ModelPoint, ModelPoint)| {
        let valid = |p: [f32; 3]| {
            p[0].is_finite()
                && p[1].is_finite()
                && p[0] >= 0.0
                && p[1] >= 0.0
                && (p[0] as f64) < width as f64
                && (p[1] as f64) < height as f64
        };
        let (a, v, b) = (res.px[ia], res.px[iv], res.px[ib]);
        if !valid(a) || !valid(v) || !valid(b) {
            return;
        }
        let u = [a[0] as f64 - v[0] as f64, a[1] as f64 - v[1] as f64];
        let w = [b[0] as f64 - v[0] as f64, b[1] as f64 - v[1] as f64];
        let d1 = (u[0] * u[0] + u[1] * u[1]).sqrt();
        let d2 = (w[0] * w[0] + w[1] * w[1]).sqrt();
        if d1 < 4.0 || d2 < 4.0 {
            return; // foreshortened segment: the angle is pure noise
        }
        let cross = u[0] * w[1] - u[1] * w[0];
        let dot = u[0] * w[0] + u[1] * w[1];
        let sigma = (sigma_px * (1.0 / d1 + 1.0 / d2)).max(0.06);
        out_ang.push(AngleObs {
            a: pa,
            vertex: pv,
            b: pb,
            angle: cross.atan2(dot),
            sigma: sigma * sigma_scale,
        });
    };
    for f in 0..5 {
        let (mcp, pip, dip, tip) = (
            points[1 + f * 4].unwrap(),
            points[2 + f * 4].unwrap(),
            points[3 + f * 4].unwrap(),
            points[4 + f * 4].unwrap(),
        );
        let wrist_mp = ModelPoint::Joint(wrist);
        push_angle(out_ang, (0, 1 + f * 4, 2 + f * 4), (wrist_mp, mcp, pip));
        push_angle(out_ang, (1 + f * 4, 2 + f * 4, 3 + f * 4), (mcp, pip, dip));
        push_angle(out_ang, (2 + f * 4, 3 + f * 4, 4 + f * 4), (pip, dip, tip));
    }
}

#[cfg(test)]
mod wrist_crop_tests {
    use super::*;

    fn kps_133() -> Vec<(f32, f32, f32)> {
        vec![(0.0, 0.0, 0.0); 133]
    }

    #[test]
    fn wrist_crop_centres_past_the_wrist_along_the_forearm() {
        let mut kps = kps_133();
        // Left elbow at (0.3, 0.5), wrist at (0.5, 0.5) on a 640x480 frame:
        // forearm points +x, 128 px long → size = 153.6, centre past the
        // wrist by a quarter size.
        kps[7] = (0.30, 0.50, 0.9);
        kps[9] = (0.50, 0.50, 0.9);
        let (x, y, size) = wrist_hand_crop(&kps, 0, 640, 480, 0.35, 96.0).unwrap();
        assert!((size - 153.6).abs() < 0.1, "size {size}");
        let cx = x as f64 + size as f64 * 0.5;
        let cy = y as f64 + size as f64 * 0.5;
        assert!((cx - (320.0 + 153.6 * 0.25)).abs() < 0.1, "cx {cx}");
        assert!((cy - 240.0).abs() < 0.1, "cy {cy}");
    }

    #[test]
    fn wrist_crop_falls_back_to_the_wrist_without_an_elbow() {
        let mut kps = kps_133();
        kps[10] = (0.50, 0.60, 0.9);
        let (x, y, size) = wrist_hand_crop(&kps, 1, 640, 480, 0.35, 96.0).unwrap();
        assert!((size - 96.0 * 2.4).abs() < 0.01, "elbow-less fallback enlarges the crop");
        assert!((x as f64 + size as f64 * 0.5 - 320.0).abs() < 0.1);
        assert!((y as f64 + size as f64 * 0.5 - 288.0).abs() < 0.1);
    }

    #[test]
    fn wrist_crop_rejects_unconfident_or_out_of_frame_wrists() {
        let mut kps = kps_133();
        kps[9] = (0.50, 0.60, 0.2);
        assert!(wrist_hand_crop(&kps, 0, 640, 480, 0.35, 96.0).is_none(), "low score");
        let mut kps = kps_133();
        kps[9] = (1.20, 0.60, 0.9);
        assert!(wrist_hand_crop(&kps, 0, 640, 480, 0.35, 96.0).is_none(), "out of frame");
    }
}
