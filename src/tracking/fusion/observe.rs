//! Observation builders: turn perception outputs (RTMW3D 133 keypoints
//! with SimCC σ, FaceMesh landmarks, the D435 point cloud) into the
//! estimator's [`FrameObs`] terms. No gating logic lives here beyond
//! "is this measurement physically usable" (in frame, finite, inside the
//! person's depth band); everything else is the estimator's job.

use super::estimator::{closest_on_segment, Kp2d, Kp3d, ModelPoint};
use super::math::*;
use super::model::*;

/// COCO-Wholebody index → model point, for the 133-keypoint layout.
pub struct BodyMap {
    pub points: Vec<Option<ModelPoint>>,
}

impl BodyMap {
    pub fn new(h: &Humanoid) -> Self {
        let j = &h.j;
        let s = &h.s;
        let mut m: Vec<Option<ModelPoint>> = vec![None; 133];
        let mut set = |i: usize, p: ModelPoint| m[i] = Some(p);
        set(0, ModelPoint::Site(s.nose));
        set(1, ModelPoint::Site(s.l_eye));
        set(2, ModelPoint::Site(s.r_eye));
        set(3, ModelPoint::Site(s.l_ear));
        set(4, ModelPoint::Site(s.r_ear));
        set(5, ModelPoint::Joint(j.l_shoulder));
        set(6, ModelPoint::Joint(j.r_shoulder));
        set(7, ModelPoint::Joint(j.l_elbow));
        set(8, ModelPoint::Joint(j.r_elbow));
        set(9, ModelPoint::Joint(j.l_wrist));
        set(10, ModelPoint::Joint(j.r_wrist));
        set(11, ModelPoint::Joint(j.l_hip));
        set(12, ModelPoint::Joint(j.r_hip));
        set(13, ModelPoint::Joint(j.l_knee));
        set(14, ModelPoint::Joint(j.r_knee));
        set(15, ModelPoint::Joint(j.l_ankle));
        set(16, ModelPoint::Joint(j.r_ankle));
        // 17 l_big_toe, 18 l_small_toe, 19 l_heel, 20 r_big_toe, 21 r_small_toe, 22 r_heel
        set(17, ModelPoint::Site(s.toe[0]));
        set(20, ModelPoint::Site(s.toe[1]));
        // Hands: 91.. left, 112.. right (MediaPipe order)
        for hand in 0..2 {
            let base = if hand == 0 { 91 } else { 112 };
            let wrist = if hand == 0 { j.l_wrist } else { j.r_wrist };
            set(base, ModelPoint::Joint(wrist));
            for f in 0..5 {
                let fj = j.finger[hand][f];
                // per finger: mcp/cmc, pip/mcp, dip/ip, tip
                set(base + 1 + f * 4, ModelPoint::Joint(fj[0]));
                set(base + 2 + f * 4, ModelPoint::Joint(fj[2]));
                set(base + 3 + f * 4, ModelPoint::Joint(fj[3]));
                set(base + 4 + f * 4, ModelPoint::Site(s.tip[hand][f]));
            }
        }
        Self { points: m }
    }
}

/// Raw 2-D keypoint as produced by the detector, whole-frame normalised.
#[derive(Clone, Copy, Debug)]
pub struct RawKp {
    pub nx: f32,
    pub ny: f32,
    pub score: f32,
    /// σ in normalised units (x / y).
    pub sx: f32,
    pub sy: f32,
}

/// Zero the score of keypoints sitting on the border of the detector's
/// person crop `(x, y, w, h)` in frame pixels: a top-down detector clamps
/// joints it cannot see to the crop edge with a plausible-looking score,
/// so those carry no information (they are not "at the edge", they are
/// "somewhere outside"). `margin_frac` is relative to the crop size.
pub fn cull_crop_border(
    kps: &mut [RawKp],
    crop: (f32, f32, f32, f32),
    width: u32,
    height: u32,
    margin_frac: f32,
) {
    let (cx, cy, cw, ch) = crop;
    if cw <= 0.0 || ch <= 0.0 {
        return;
    }
    let mx = cw * margin_frac;
    let my = ch * margin_frac;
    for kp in kps.iter_mut() {
        let x = kp.nx * width as f32;
        let y = kp.ny * height as f32;
        if x <= cx + mx || x >= cx + cw - mx || y <= cy + my || y >= cy + ch - my {
            kp.score = 0.0;
        }
    }
}

/// σ policy for detector keypoints.
#[derive(Clone, Copy, Debug)]
pub struct KpSigma {
    /// Floor on the pixel σ.
    pub floor_px: f64,
    /// Multiplier on the SimCC σ.
    pub simcc_gain: f64,
    /// Below this score the keypoint is not used at all.
    pub min_score: f32,
    /// Extra inflation per unit of (1 − score).
    pub score_inflate: f64,
    /// Margin (fraction of frame) inside which a keypoint counts as in
    /// frame; border-clamped detections carry no information.
    pub border_frac: f32,
}

impl Default for KpSigma {
    fn default() -> Self {
        // `score` is the calibrated visibility probability (≈ 0.99 for a
        // sharp peak) since the visibility layer landed. Under the old
        // sigmoid peak height (≈ 0.71 for the same joints) the inflation
        // term contributed a constant ×1.87 that the whole 2-D/3-D/prior
        // balance was tuned against; that gain now lives in `simcc_gain`
        // and `floor_px` so a confident joint keeps the same σ, while
        // `score_inflate` acts on genuine uncertainty (p_vis 0.5–0.9).
        Self {
            floor_px: 2.8,
            simcc_gain: 1.9,
            min_score: 0.2,
            score_inflate: 3.0,
            border_frac: 0.02,
        }
    }
}

/// Constant part of the old score-driven σ inflation `1 + (1 − score)`
/// under the detector's sigmoid peak height (≈ 0.71 for a sharp peak):
/// the depth-lift and surface terms were tuned against ×1.3. `score` is
/// now the calibrated visibility probability (≈ 0.99 for the same joints),
/// so the constant is applied explicitly and the score term only acts on
/// genuine uncertainty. See `KpSigma::default` for the 2-D counterpart.
pub const SCORE_INFLATE_BASE_3D_DEFAULT: f64 = 1.3;

/// `SCORE_INFLATE_BASE_3D_DEFAULT`, or 1.0 under `VULVATAR_FUSION_OLDSIGMA`
/// (ablation bench only; read once).
#[allow(non_snake_case)]
pub fn SCORE_INFLATE_BASE_3D() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        if std::env::var_os("VULVATAR_FUSION_OLDSIGMA").is_some() {
            1.0
        } else {
            SCORE_INFLATE_BASE_3D_DEFAULT
        }
    })
}

/// Body 2-D observations from the 133 keypoints. `hand_scale` inflates
/// the σ of the body model's hand block (replaced by the dedicated hand
/// crop when that runs).
pub fn body_kp2d(
    map: &BodyMap,
    kps: &[RawKp],
    width: u32,
    height: u32,
    pol: KpSigma,
    hand_scale: f64,
    head_scale: f64,
    out: &mut Vec<Kp2d>,
) {
    let w = width as f64;
    let h = height as f64;
    let abl_no_headkp = std::env::var_os("VULVATAR_ABL_NOHEADKP").is_some();
    for (i, kp) in kps.iter().enumerate() {
        if abl_no_headkp && i < 5 {
            continue;
        }
        let Some(Some(point)) = map.points.get(i) else {
            continue;
        };
        if !(kp.score >= pol.min_score) || !kp.nx.is_finite() || !kp.ny.is_finite() {
            continue;
        }
        if kp.nx <= pol.border_frac
            || kp.nx >= 1.0 - pol.border_frac
            || kp.ny <= pol.border_frac
            || kp.ny >= 1.0 - pol.border_frac
        {
            continue;
        }
        let sx = kp.sx as f64 * w * pol.simcc_gain;
        let sy = kp.sy as f64 * h * pol.simcc_gain;
        let mut sigma = sx.max(sy).max(pol.floor_px);
        sigma *= 1.0 + pol.score_inflate * (1.0 - kp.score as f64);
        if i >= 91 {
            sigma *= hand_scale;
        }
        if i < 5 {
            // Face keypoints (nose/eyes/ears). SimCC frontalizes these
            // past ~30° head yaw just like the dense mesh — the caller
            // widens them whenever a better head-position anchor exists.
            sigma *= head_scale;
        }
        out.push(Kp2d {
            point: *point,
            u: kp.nx as f64 * w,
            v: kp.ny as f64 * h,
            sigma,
        });
    }
}

/// Depth-lifted 3-D observations for the body-17 keypoints (+ hand
/// wrists): median metric point under each keypoint, pushed along the
/// viewing ray by the expected skin→joint depth so the observation sits at
/// the joint centre. `z_ref` = person reference depth (m) for the band gate
/// (e.g. the median of the shoulder samples); pass `None` on bootstrap.
#[allow(clippy::too_many_arguments)]
pub fn body_kp3d(
    map: &BodyMap,
    kps: &[RawKp],
    points: &[[f32; 3]],
    width: u32,
    height: u32,
    z_ref: Option<f64>,
    min_score: f32,
    // `occluded(u, v)` = the pixel is inside an active hand crop. Face
    // sites are not depth-lifted there (see below); hand/wrist entries
    // ignore it, since their pixel is SUPPOSED to be inside a hand crop.
    occluded: &dyn Fn(f64, f64) -> bool,
    out: &mut Vec<Kp3d>,
) {
    // (coco index, skin→joint offset m, σ m). Only limb ends: their pixel is
    // rarely occluded by another body part, so the depth is the joint's own
    // surface. Torso / leg joints go through `body_surface_points`.
    // Ears were tried as an extra metric head anchor for face-cover
    // frames and made every replay worse (headphones / hair sit proud of
    // the anatomical point and the pixel often lands past the silhouette):
    // torso yaw rms +1.5° on palms and wave, and a 81° pitch spike on
    // namaste. The nose is the only face site worth lifting.
    const TABLE: [(usize, f64, f64); 7] = [
        (0, 0.0, 0.02),    // nose (site: on the surface)
        (7, 0.035, 0.045), // elbows (foreshortened arms put the pixel on the
        (8, 0.035, 0.045), //   limb surface, not the joint)
        (9, 0.02, 0.02),   // wrists
        (10, 0.02, 0.02),
        (91, 0.015, 0.02), // hand-block wrists
        (112, 0.015, 0.02),
    ];
    let (zlo, zhi) = match z_ref {
        Some(z) => ((z - 0.7) as f32, (z + 0.7) as f32),
        None => (0.15, 6.0),
    };
    for &(i, off, sigma) in &TABLE {
        let Some(Some(point)) = map.points.get(i) else {
            continue;
        };
        let Some(kp) = kps.get(i) else { continue };
        if !(kp.score >= min_score) || !kp.nx.is_finite() || !kp.ny.is_finite() {
            continue;
        }
        // Reject keypoints clamped at the frame border (within 2% margin): detector emits
        // limbs exiting the frame clamped to the border, and depth underneath is unrelated.
        if kp.nx <= 0.02 || kp.nx >= 0.98 || kp.ny <= 0.02 || kp.ny >= 0.98 {
            continue;
        }
        let u = kp.nx as f64 * width as f64;
        let v = kp.ny as f64 * height as f64;
        // A FACE keypoint whose pixel falls inside a hand crop is painted
        // onto the occluder: the depth there is the palm's, ~20 cm nearer
        // than the face, and lifting it at σ 2 cm drags the whole head —
        // and with it the torso — into a false bow (measured: namaste
        // head pitch +36°, torso pitch +21°, torso roll +27°). The 2-D
        // residual stays and keeps anchoring head position; only the
        // metric lift is dropped.
        let sigma = if i < 5 && occluded(u, v) {
            // Softened rather than dropped: dropping it costs the head its
            // only metric anchor through a full-face cover (palms torso
            // yaw swung ±25°), while keeping it at face σ bows the whole
            // body. The palm sits ~5–10 cm proud of the face, so widen to
            // that scale — the point still holds the head in place, and
            // the bias is small against its own σ.
            sigma * 5.0
        } else {
            sigma
        };
        let Some((p, z_spread)) = window_point_with_spread(points, width, height, u, v, 3, zlo, zhi)
        else {
            continue;
        };
        let n = norm(p);
        if n < 0.1 {
            continue;
        }
        let p_joint = scale(p, (n + off) / n);
        // Low detector score → the pixel may not be on the joint at all.
        let mut s = sigma * SCORE_INFLATE_BASE_3D() * (1.0 + 1.0 * (1.0 - kp.score as f64));
        // A two-surface window (arm against the desk plane, sleeve against
        // the background — common at the frame's bottom edge, where the
        // desk-entry elbows live) means the median depth is whichever
        // surface won this frame; it flips between frames while σ keeps
        // claiming confidence, and the slam lands downstream (measured:
        // the desk-edge left-elbow lift, and the R-wrist snaps on
        // s1789303569). ELBOW entries only: the wrist/hand-block lifts sit
        // inside an active hand crop whose window is legitimately mixed
        // (hand over desk) and demoting them starves the arm's trusted
        // metric anchor (benched: L-wrist snaps 2 → 3, max 0.19 → 0.30 m
        // on s1789311387 when the gate covered them too). Demote to a
        // kernel-neutering σ: small innovations still track, large ones
        // get outvoted by the robust loss. `VULVATAR_FUSION_LIFT_SPREAD`
        // overrides the gate; 0 disables.
        let spread_gate = std::env::var("VULVATAR_FUSION_LIFT_SPREAD")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| *v > 0.0)
            .unwrap_or(0.05);
        let is_elbow = i == 7 || i == 8;
        if is_elbow && z_spread > spread_gate {
            s = s.min(0.025);
        }
        out.push(Kp3d {
            point: *point,
            p: p_joint,
            sigma: s,
            lat_scale: 1.0,
        });
    }
}

/// Torso / leg joints (shoulders, hips, knees, ankles) from depth, with a
/// model z-buffer test: the depth under the keypoint is the joint's own
/// skin only if nothing else in the *predicted* model lies in front of
/// that joint along the ray. Then it becomes a metric joint observation
/// (surface + skin→joint offset, `kp3d`); otherwise (an arm across the
/// chest, a hand over a hip) it is an occluder surface point (`surface`).
/// `pred_fk`/`pred_st` = the predicted model for this frame.
#[allow(clippy::too_many_arguments)]
pub fn body_torso_leg_depth(
    map: &BodyMap,
    kps: &[RawKp],
    points: &[[f32; 3]],
    width: u32,
    height: u32,
    z_ref: Option<f64>,
    min_score: f32,
    model: &Model,
    pred_fk: &Fk,
    pred_st: &State,
    out3d: &mut Vec<Kp3d>,
    out_surface: &mut Vec<(V3, f64)>,
) {
    // (coco index, skin→joint offset m, σ m)
    const TABLE: [(usize, f64, f64); 8] = [
        (5, 0.045, 0.025),
        (6, 0.045, 0.025),
        (11, 0.09, 0.05),
        (12, 0.09, 0.05),
        (13, 0.045, 0.05),
        (14, 0.045, 0.05),
        (15, 0.035, 0.05),
        (16, 0.035, 0.05),
    ];
    let (zlo, zhi) = match z_ref {
        Some(z) => ((z - 0.7) as f32, (z + 0.7) as f32),
        None => (0.15, 6.0),
    };
    for &(i, off, sigma) in &TABLE {
        let Some(Some(point)) = map.points.get(i) else {
            continue;
        };
        let Some(kp) = kps.get(i) else { continue };
        if !(kp.score >= min_score) || kp.nx <= 0.0 || kp.nx >= 1.0 || kp.ny <= 0.0 || kp.ny >= 1.0
        {
            continue;
        }
        let u = kp.nx as f64 * width as f64;
        let v = kp.ny as f64 * height as f64;
        let Some(p) = window_point(points, width, height, u, v, 2, zlo, zhi) else {
            continue;
        };
        let n = norm(p);
        if n < 0.1 {
            continue;
        }
        // Predicted joint depth along this ray, and the nearest predicted
        // capsule that is clearly in front of it: an occluder.
        let (jw, jp) = match *point {
            ModelPoint::Joint(j) => (j, pred_fk.t[j]),
            _ => continue,
        };
        let joint_depth = norm(jp);
        let dir = scale(p, 1.0 / n);
        let mut occluded = false;
        for c in &model.capsules {
            // Skip the joint's own limb capsules (their surface IS the skin).
            // (Elliptic trunk: the depth semi-axis is the relevant extent.)
            let own = match (c.a, c.b) {
                (PointRef::Joint(a), _) if a == jw => true,
                (_, PointRef::Joint(b)) if b == jw => true,
                _ => false,
            };
            if own || matches!(c.part, Part::Torso) {
                continue;
            }
            let (q, _u) = closest_on_segment(
                pred_fk.point(c.a),
                pred_fk.point(c.b),
                scale(dir, joint_depth),
            );
            // Distance from the ray to the capsule axis, evaluated at the
            // segment point closest to the joint-depth sample.
            let along = dot(q, dir);
            let radial = norm(sub(q, scale(dir, along)));
            let r = model.capsule_radius(pred_st, c);
            if radial < r + 0.02 && along < joint_depth - 0.05 {
                occluded = true;
                break;
            }
        }
        if occluded || (p[2] as f64) < joint_depth - 0.20 {
            out_surface.push((
                p,
                0.015 * SCORE_INFLATE_BASE_3D() * (1.0 + (1.0 - kp.score as f64)),
            ));
        } else {
            let p_joint = scale(p, (n + off) / n);
            out3d.push(Kp3d {
                point: *point,
                p: p_joint,
                sigma: sigma * SCORE_INFLATE_BASE_3D() * (1.0 + (1.0 - kp.score as f64)),
                lat_scale: 1.0,
            });
        }
    }
}

/// Torso yaw from the chest's depth slope, in camera x/z (radians),
/// with the number of surviving columns.
///
/// A torso capsule is rotationally symmetric about its own axis, so the
/// surface term carries no yaw information at all and the 2-D shoulder
/// pixels are left to fix it alone — measured +17° of torso
/// over-rotation against the depth reference on a live desk session.
/// This samples the chest between the detected shoulders, reduces each
/// column to its median depth, rejects columns sitting clearly in front
/// of the chest plane (a forearm or clasped hands, which otherwise swing
/// the answer by 60°+) and fits z(x) over the rest.
pub fn chest_yaw_from_depth(
    kps: &[RawKp],
    points: &[[f32; 3]],
    width: u32,
    height: u32,
    z_ref: Option<f64>,
    occluded: &dyn Fn(f64, f64) -> bool,
) -> Option<(f64, usize)> {
    let (zlo, zhi) = match z_ref {
        Some(z) => ((z - 0.5) as f32, (z + 0.5) as f32),
        None => (0.15, 6.0),
    };
    let px = |i: usize| -> Option<(f64, f64)> {
        let kp = kps.get(i)?;
        if kp.score < 0.5 || !(0.0..=1.0).contains(&kp.nx) || !(0.0..=1.0).contains(&kp.ny) {
            return None;
        }
        Some((kp.nx as f64 * width as f64, kp.ny as f64 * height as f64))
    };
    let (lp, rp) = (px(5)?, px(6)?);
    let span = ((lp.0 - rp.0).powi(2) + (lp.1 - rp.1).powi(2)).sqrt();
    if span < 40.0 {
        if std::env::var_os("VULVATAR_CHEST_DUMP").is_some() {
            eprintln!("CHESTFAIL shspan {:.0}", span);
        }
        return None;
    }
    const COLS: usize = 15;
    let (mut n_occl, mut n_nodepth) = (0usize, 0usize);
    // Sampling bands, tried in order, as (row offset range, skip the
    // middle third). The chest is the best surface — wide, smooth,
    // rigidly tied to the trunk — but clasped or folded arms cover it
    // (measured on a live session: 95 of 105 chest samples occluded, so
    // the observation reached only 5% of frames). The shoulder tops stay
    // clear in those poses; their middle third is dropped because the
    // neck and chin sit there, not the trunk.
    let bands: [(f64, f64, bool); 2] = [(0.02, 0.18, false), (-0.16, -0.03, true)];
    for &(row_lo, row_hi, skip_middle) in &bands {
        let mut med: Vec<(f64, f64)> = Vec::with_capacity(COLS);
        for c in 0..COLS {
            // Inset 12% at each end: a column on the silhouette mixes the
            // background into its median.
            let t = 0.12 + 0.76 * (c as f64 + 0.5) / COLS as f64;
            if skip_middle && (0.33..0.67).contains(&t) {
                continue;
            }
            let bu = rp.0 + (lp.0 - rp.0) * t;
            let bv = rp.1 + (lp.1 - rp.1) * t;
            let (mut xs, mut zs): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
            for row in 0..7 {
                let v = bv + span * (row_lo + (row_hi - row_lo) * row as f64 / 6.0);
                if occluded(bu, v) {
                    n_occl += 1;
                    continue;
                }
                if let Some(q) = window_point(points, width, height, bu, v, 1, zlo, zhi) {
                    xs.push(q[0]);
                    zs.push(q[2]);
                } else {
                    n_nodepth += 1;
                }
            }
            if zs.len() < 4 {
                continue;
            }
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            med.push((xs[xs.len() / 2], zs[zs.len() / 2]));
        }
        // A band needs enough columns to fit a line; the shoulder-top band
        // has at most 10 (its middle third is skipped).
        if med.len() < if skip_middle { 6 } else { 8 } {
            if std::env::var_os("VULVATAR_CHEST_DUMP").is_some() {
                eprintln!(
                    "CHESTFAIL cols {} of {} occl {} nodepth {}",
                    med.len(),
                    COLS,
                    n_occl,
                    n_nodepth
                );
            }
            continue;
        }
        let mut zz: Vec<f64> = med.iter().map(|c| c.1).collect();
        zz.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let _ = zz.drain(..);
        // Slope-aware column filter (`VULVATAR_CHEST_LINEFIT=1`): reject by
        // residual to a first-pass line z(x) instead of the global depth
        // median. The median window (−6/+12 cm) cannot hold a genuinely
        // yawed trunk (0.4 m span at 40° spans ~26 cm of depth) and kills
        // the shoulder-top band (measured 576/600 frames by `zfilter`).
        // NOT the default: with the trunk's aspect now shape-fitted, the
        // chest observation at σ 0.12 FIGHTS the aspect fit and drags the
        // pathological session back to a leaned solution (measured:
        // aspect-only yaw −25°/tilt 2.6° → +obs yaw −59°/tilt 26°). Keep
        // env-gated until its σ is re-derived against the fitted trunk.
        let pts: Vec<(f64, f64)> = if std::env::var_os("VULVATAR_CHEST_LINEFIT").is_some() {
            let n = med.len() as f64;
            let mx = med.iter().map(|p| p.0).sum::<f64>() / n;
            let mz = med.iter().map(|p| p.1).sum::<f64>() / n;
            let (mut num, mut den) = (0.0, 0.0);
            for (x, z) in &med {
                num += (x - mx) * (z - mz);
                den += (x - mx) * (x - mx);
            }
            let a = num / den.max(1e-9);
            med.iter()
                .copied()
                .filter(|(x, z)| (z - (mz + a * (x - mx))).abs() < 0.05)
                .collect()
        } else {
            let mid = med[med.len() / 2].1;
            med.iter()
                .copied()
                .filter(|(_, z)| *z > mid - 0.06 && *z < mid + 0.12)
                .collect()
        };
        if pts.len() < if skip_middle { 6 } else { 8 } {
            if std::env::var_os("VULVATAR_CHEST_DUMP").is_some() {
                eprintln!("CHESTFAIL zfilter {} of {}", pts.len(), med.len());
            }
            continue;
        }
        let xmin = pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let xmax = pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        if xmax - xmin < 0.12 {
            if std::env::var_os("VULVATAR_CHEST_DUMP").is_some() {
                eprintln!("CHESTFAIL span {:.3}", xmax - xmin);
            }
            continue;
        }
        let n = pts.len() as f64;
        let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
        let mz = pts.iter().map(|p| p.1).sum::<f64>() / n;
        let (mut num, mut den) = (0.0, 0.0);
        for (x, z) in &pts {
            num += (x - mx) * (z - mz);
            den += (x - mx) * (x - mx);
        }
        return Some(((num / den.max(1e-9)).atan(), pts.len()));
    }
    None
}

/// Reachability filter on arm keypoints: an elbow / wrist / hand-block
/// landmark whose pixel has valid depth farther than the arm can reach
/// from the predicted shoulder (or much nearer than the shoulder plane
/// allows) is not on this person's arm — it is a detection on the chair,
/// the wall or a bystander — and its score is zeroed so neither the 2-D
/// nor the depth-lift terms see it. Pixels without depth are left alone
/// (near-range holes are common on real hands).
#[allow(clippy::too_many_arguments)]
pub fn reach_filter(
    kps: &mut [RawKp],
    points: &[[f32; 3]],
    width: u32,
    height: u32,
    shoulder_z: [f64; 2],
    reach_m: f64,
) {
    // (index range, side)
    let groups: [(usize, usize, usize); 6] = [
        (7, 8, 0),     // l_elbow
        (8, 9, 1),     // r_elbow
        (9, 10, 0),    // l_wrist
        (10, 11, 1),   // r_wrist
        (91, 112, 0),  // left hand block
        (112, 133, 1), // right hand block
    ];
    // Does this side's elbow have depth support? (Same window as below.)
    let elbow_has_depth: [bool; 2] = [7usize, 8].map(|i| {
        kps.get(i)
            .filter(|kp| kp.score > 0.2 && (0.0..=1.0).contains(&kp.nx))
            .and_then(|kp| {
                window_point(
                    points,
                    width,
                    height,
                    kp.nx as f64 * width as f64,
                    kp.ny as f64 * height as f64,
                    2,
                    0.15,
                    8.0,
                )
            })
            .is_some()
    });
    for &(lo, hi, side) in &groups {
        let zs = shoulder_z[side];
        if !zs.is_finite() || zs <= 0.0 {
            continue;
        }
        for i in lo..hi.min(kps.len()) {
            let kp = &mut kps[i];
            if kp.score <= 0.0 || !(0.0..=1.0).contains(&kp.nx) || !(0.0..=1.0).contains(&kp.ny) {
                continue;
            }
            let u = kp.nx as f64 * width as f64;
            let v = kp.ny as f64 * height as f64;
            let probe = window_point(points, width, height, u, v, 2, 0.15, 8.0);
            if std::env::var_os("VULVATAR_REACH_DUMP").is_some() && (i == 9 || i == 10 || i == 7) {
                eprintln!(
                    "REACH kp{i} px ({u:.0},{v:.0}) score {:.2} z {:?} shoulder_z {zs:.3} reach {reach_m:.2}",
                    kp.score,
                    probe.map(|p| (p[2] * 1000.0).round() / 1000.0)
                );
            }
            match probe {
                Some(p) => {
                    if p[2] > zs + reach_m || p[2] < zs - reach_m {
                        kp.score = 0.0;
                    }
                }
                // No depth AND at the frame edge: unverifiable. The depth
                // camera covers the person's working volume, so a limb
                // keypoint out at the border with nothing behind it is
                // either off-sensor or a detection on someone else — a
                // bystander's hand at x = 637/640 captured this user's
                // left arm for the first ~35 frames of a desk session.
                // Depth HOLES in frame are left alone: hands close to the
                // camera lose depth constantly and that is not evidence
                // of anything.
                None => {
                    // A depth-less keypoint out at the frame border is
                    // unverifiable — but the user's OWN hand loses depth
                    // there constantly (too close, or past the depth FOV),
                    // so it is only rejected when nothing else on that arm
                    // has depth either. A bystander's hand arrives alone;
                    // the user's hand arrives with an elbow.
                    // 5% of the frame. Swept: at 2% and 1% the phantom
                    // comes back (the bystander sat at 0.995 but the
                    // detector's estimate of it wanders inward).
                    let border = !(0.05..0.95).contains(&kp.nx) || !(0.05..0.95).contains(&kp.ny);
                    if border && !elbow_has_depth[side] {
                        kp.score = 0.0;
                    }
                }
            }
        }
    }
}

/// Shoulder-mid 3-D point from depth (camera metres) — the root seed hint.
pub fn shoulder_mid_hint(
    kps: &[RawKp],
    points: &[[f32; 3]],
    width: u32,
    height: u32,
    min_score: f32,
) -> Option<V3> {
    let mut pts = Vec::new();
    for i in [5usize, 6] {
        let Some(kp) = kps.get(i) else { continue };
        if kp.score < min_score || kp.nx <= 0.0 || kp.nx >= 1.0 || kp.ny <= 0.0 || kp.ny >= 1.0 {
            continue;
        }
        if let Some(p) = window_point(
            points,
            width,
            height,
            kp.nx as f64 * width as f64,
            kp.ny as f64 * height as f64,
            4,
            0.15,
            6.0,
        ) {
            pts.push(p);
        }
    }
    match pts.len() {
        0 => None,
        1 => Some(pts[0]),
        _ => Some(scale(add(pts[0], pts[1]), 0.5)),
    }
}

/// Median depth of the shoulder keypoints (person reference), if any.
pub fn shoulder_depth_ref(
    kps: &[RawKp],
    points: &[[f32; 3]],
    width: u32,
    height: u32,
    min_score: f32,
) -> Option<f64> {
    let mut zs = Vec::new();
    for i in [5usize, 6, 0] {
        let Some(kp) = kps.get(i) else { continue };
        if kp.score < min_score || kp.nx <= 0.0 || kp.nx >= 1.0 || kp.ny <= 0.0 || kp.ny >= 1.0 {
            continue;
        }
        if let Some(p) = window_point(
            points,
            width,
            height,
            kp.nx as f64 * width as f64,
            kp.ny as f64 * height as f64,
            3,
            0.15,
            6.0,
        ) {
            zs.push(p[2]);
        }
    }
    if zs.is_empty() {
        return None;
    }
    zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(zs[zs.len() / 2])
}

/// Median metric point in a small window of a full-frame point cloud,
/// restricted to a depth band. Returns `None` when fewer than 3 valid
/// samples exist.
pub fn window_point(
    points: &[[f32; 3]],
    width: u32,
    height: u32,
    u: f64,
    v: f64,
    radius: i32,
    z_lo: f32,
    z_hi: f32,
) -> Option<V3> {
    window_point_with_spread(points, width, height, u, v, radius, z_lo, z_hi).map(|(p, _)| p)
}

/// `window_point` plus the spread of the window's valid depths
/// (p90 − p10 of `zs`). A wide spread means the window straddles TWO
/// surfaces — an arm against the desk plane, a sleeve edge against the
/// background — and the median depth is whichever surface won the
/// majority this frame: it flips between them frame to frame while the
/// lift's σ keeps claiming millimetre confidence (measured live on the
/// desk rig: the left-elbow depth swung 0.45 ↔ 0.88 m at σ 0.03). The
/// spread is the sensor's own admission that the sample is ambiguous.
pub fn window_point_with_spread(
    points: &[[f32; 3]],
    width: u32,
    height: u32,
    u: f64,
    v: f64,
    radius: i32,
    z_lo: f32,
    z_hi: f32,
) -> Option<(V3, f32)> {
    let (w, h) = (width as i32, height as i32);
    let cu = u.round() as i32;
    let cv = v.round() as i32;
    let mut zs: Vec<f32> = Vec::with_capacity(((2 * radius + 1) * (2 * radius + 1)) as usize);
    let mut xs = Vec::with_capacity(zs.capacity());
    let mut ys = Vec::with_capacity(zs.capacity());
    for dy in -radius..=radius {
        let y = cv + dy;
        if y < 0 || y >= h {
            continue;
        }
        for dx in -radius..=radius {
            let x = cu + dx;
            if x < 0 || x >= w {
                continue;
            }
            let p = points[(y * w + x) as usize];
            if p[2].is_finite() && p[2] > z_lo && p[2] < z_hi {
                zs.push(p[2]);
                xs.push(p[0]);
                ys.push(p[1]);
            }
        }
    }
    if zs.len() < 3 {
        return None;
    }
    let med = |v: &mut Vec<f32>| -> f32 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let spread = {
        zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lo = zs[zs.len() / 10];
        let hi = zs[(zs.len() * 9 / 10).min(zs.len() - 1)];
        hi - lo
    };
    Some((
        [
            med(&mut xs) as f64,
            med(&mut ys) as f64,
            med(&mut zs) as f64,
        ],
        spread,
    ))
}
