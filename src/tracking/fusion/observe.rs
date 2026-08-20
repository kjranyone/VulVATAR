//! Observation builders: turn perception outputs (RTMW3D 133 keypoints
//! with SimCC σ, FaceMesh landmarks, the D435 point cloud) into the
//! estimator's [`FrameObs`] terms. No gating logic lives here beyond
//! "is this measurement physically usable" (in frame, finite, inside the
//! person's depth band); everything else is the estimator's job.

use super::estimator::{closest_on_segment, Cloud, Intrinsics, Kp2d, Kp3d, ModelPoint};
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
pub fn cull_crop_border(kps: &mut [RawKp], crop: (f32, f32, f32, f32), width: u32, height: u32, margin_frac: f32) {
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
        Self {
            floor_px: 1.5,
            simcc_gain: 1.0,
            min_score: 0.2,
            score_inflate: 3.0,
            border_frac: 0.004,
        }
    }
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
        let Some(Some(point)) = map.points.get(i) else { continue };
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
        let Some(Some(point)) = map.points.get(i) else { continue };
        let Some(kp) = kps.get(i) else { continue };
        if !(kp.score >= min_score) || !kp.nx.is_finite() || !kp.ny.is_finite() {
            continue;
        }
        if kp.nx <= 0.0 || kp.nx >= 1.0 || kp.ny <= 0.0 || kp.ny >= 1.0 {
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
        let Some(p) = window_point(points, width, height, u, v, 3, zlo, zhi) else { continue };
        let n = norm(p);
        if n < 0.1 {
            continue;
        }
        let p_joint = scale(p, (n + off) / n);
        // Low detector score → the pixel may not be on the joint at all.
        let s = sigma * (1.0 + 1.0 * (1.0 - kp.score as f64));
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
        let Some(Some(point)) = map.points.get(i) else { continue };
        let Some(kp) = kps.get(i) else { continue };
        if !(kp.score >= min_score) || kp.nx <= 0.0 || kp.nx >= 1.0 || kp.ny <= 0.0 || kp.ny >= 1.0 {
            continue;
        }
        let u = kp.nx as f64 * width as f64;
        let v = kp.ny as f64 * height as f64;
        let Some(p) = window_point(points, width, height, u, v, 2, zlo, zhi) else { continue };
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
            let own = match (c.a, c.b) {
                (PointRef::Joint(a), _) if a == jw => true,
                (_, PointRef::Joint(b)) if b == jw => true,
                _ => false,
            };
            if own || matches!(c.part, Part::Torso) {
                continue;
            }
            let (q, _u) = closest_on_segment(pred_fk.point(c.a), pred_fk.point(c.b), scale(dir, joint_depth));
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
            out_surface.push((p, 0.015 * (1.0 + (1.0 - kp.score as f64))));
        } else {
            let p_joint = scale(p, (n + off) / n);
            out3d.push(Kp3d {
                point: *point,
                p: p_joint,
                sigma: sigma * (1.0 + (1.0 - kp.score as f64)),
                lat_scale: 1.0,
            });
        }
    }
}

/// Surface points under the torso / leg keypoints (shoulders, hips, knees,
/// ankles): the depth there is *some* body surface — the joint's own skin,
/// or an occluding limb — so it is fed as a point→nearest-capsule term.
pub fn body_surface_points(
    kps: &[RawKp],
    points: &[[f32; 3]],
    width: u32,
    height: u32,
    z_ref: Option<f64>,
    min_score: f32,
    out: &mut Vec<(V3, f64)>,
) {
    let (zlo, zhi) = match z_ref {
        Some(z) => ((z - 0.7) as f32, (z + 0.7) as f32),
        None => (0.15, 6.0),
    };
    for i in [5usize, 6, 11, 12, 13, 14, 15, 16] {
        let Some(kp) = kps.get(i) else { continue };
        if !(kp.score >= min_score) || kp.nx <= 0.0 || kp.nx >= 1.0 || kp.ny <= 0.0 || kp.ny >= 1.0 {
            continue;
        }
        let u = kp.nx as f64 * width as f64;
        let v = kp.ny as f64 * height as f64;
        // A small window (radius 2 → 5×5) so the median stays on one surface.
        let Some(p) = window_point(points, width, height, u, v, 2, zlo, zhi) else { continue };
        let sigma = 0.015 * (1.0 + (1.0 - kp.score as f64));
        out.push((p, sigma));
    }
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
            if let Some(p) = window_point(points, width, height, u, v, 2, 0.15, 8.0) {
                if p[2] > zs + reach_m || p[2] < zs - reach_m {
                    kp.score = 0.0;
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

// ---------------------------------------------------------------------------
// Point cloud
// ---------------------------------------------------------------------------

/// Extract the person's neighbourhood from a full-frame point cloud
/// (`points[y*width+x]`, camera metres, NaN = invalid), using the
/// predicted model to define the region: image-space bounding box of the
/// projected capsules with a metric margin, and a depth band around the
/// model. Subsampled by stride to at most `max_points`.
#[allow(clippy::too_many_arguments)]
pub fn cloud_near_model(
    points: &[[f32; 3]],
    width: u32,
    height: u32,
    model: &Model,
    fk: &Fk,
    st: &State,
    intr: &Intrinsics,
    margin_m: f64,
    z_band_m: f64,
    max_points: usize,
) -> Cloud {
    let mut umin = f64::INFINITY;
    let mut umax = f64::NEG_INFINITY;
    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    let mut zmin = f64::INFINITY;
    let mut zmax = f64::NEG_INFINITY;
    for c in &model.capsules {
        let r = model.capsule_radius(st, c);
        for p in [fk.point(c.a), fk.point(c.b)] {
            zmin = zmin.min(p[2] - r);
            zmax = zmax.max(p[2] + r);
            if let Some(uv) = intr.project(p) {
                let pad = intr.fx * (r + margin_m) / p[2].max(0.2);
                umin = umin.min(uv[0] - pad);
                umax = umax.max(uv[0] + pad);
                vmin = vmin.min(uv[1] - pad);
                vmax = vmax.max(uv[1] + pad);
            }
        }
    }
    if !umin.is_finite() || !zmin.is_finite() {
        return Cloud::default();
    }
    let x0 = umin.floor().clamp(0.0, width as f64) as usize;
    let x1 = umax.ceil().clamp(0.0, width as f64) as usize;
    let y0 = vmin.floor().clamp(0.0, height as f64) as usize;
    let y1 = vmax.ceil().clamp(0.0, height as f64) as usize;
    if x1 <= x0 || y1 <= y0 {
        return Cloud::default();
    }
    let zlo = (zmin - z_band_m) as f32;
    let zhi = (zmax + z_band_m) as f32;
    let area = (x1 - x0) * (y1 - y0);
    // Choose a stride so the ROI yields ≈ max_points before validity culls.
    let stride = ((area as f64 / max_points.max(1) as f64).sqrt().floor() as usize).max(1);
    let mut out = Vec::with_capacity(max_points + 64);
    let w = width as usize;
    let mut y = y0;
    while y < y1 {
        let mut x = x0;
        while x < x1 {
            let p = points[y * w + x];
            if p[2].is_finite() && p[2] > zlo && p[2] < zhi && p[0].is_finite() && p[1].is_finite() {
                out.push(p);
            }
            x += stride;
        }
        y += stride;
    }
    Cloud {
        points: out,
        sigma: Vec::new(),
    }
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
    Some([med(&mut xs) as f64, med(&mut ys) as f64, med(&mut zs) as f64])
}
