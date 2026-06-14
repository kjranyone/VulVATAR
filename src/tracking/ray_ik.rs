//! Ray-IK: depth solve along 2D observation rays.
//!
//! Design: `docs/ray-ik-depth-solve.md`. Each tracked 2D keypoint
//! defines a camera ray; the joint's 3D position is `P_i = t_i · d_i`
//! with a single depth unknown `t_i`. Solving for the depth vector
//! makes 2D reprojection consistency an identity (not a residual), so
//! the avatar can never drift from the detection annotation in the
//! observable dimensions. All depth evidence — metric bone lengths,
//! DAv2 anchor depths, decisive-backward nz, temporal continuity, the
//! desk-streaming forward prior — fuses in one Gauss-Newton problem
//! instead of the sequential heuristic overwrites this replaces.
//!
//! Conventions: camera space, x = image right, y = image down,
//! z = away from the camera (all in metres). Rays are unit vectors
//! with positive z; depths `t` are strictly positive. Axis flips into
//! source space are the caller's job.

use std::collections::HashMap;

use crate::asset::HumanoidBone;

/// One observed joint: a unit ray through the camera centre plus the
/// optional per-joint depth evidence.
#[derive(Clone, Copy, Debug)]
pub struct RayObservation {
    pub bone: HumanoidBone,
    /// Unit ray direction in camera space (z > 0).
    pub ray: [f32; 3],
    /// Detector confidence [0, 1]; scales the residual weights of
    /// every term touching this joint.
    pub confidence: f32,
    /// Absolute camera depth ALONG THE RAY in metres (range, not the
    /// z-coordinate: callers convert `t = z / ray_z`), if available —
    /// a DAv2 sample or the span-derived torso anchor.
    pub metric_depth_m: Option<f32>,
    /// Multiplier on `cfg.w_depth` for this joint's depth-evidence
    /// residual. Lets a soft plane anchor (torso depth on a free
    /// shoulder, which must not crush the twist Δz the segments
    /// resolve) coexist with hard DAv2 samples (1.0).
    pub depth_weight: f32,
    /// Treat this joint as a constant: it participates in segments but
    /// is not a solve variable. Requires `metric_depth_m`; without it
    /// the joint falls back to being solved. Used to pin the shoulder
    /// anchors so the arm solve cannot drag the torso.
    pub fixed: bool,
}

/// One rigid segment between two observed joints. `a` is proximal,
/// `b` distal — the ordering drives the forward-bias initialisation
/// and the forward prior.
#[derive(Clone, Copy, Debug)]
pub struct SegmentSpec {
    pub a: HumanoidBone,
    pub b: HumanoidBone,
    /// Metric bone length (calibrated span × anthropometric ratio,
    /// band-clamped by the caller).
    pub length_m: f32,
    /// The model's own nz evidence: camera-z displacement `z_b − z_a`
    /// in metres, fed as a WEAK consistency term on every segment.
    /// Near-zero nz keeps ambiguous (projection-fits-flat) segments
    /// flat; a large backward value overcomes the forward prior. The
    /// decisive-backward deadband of the old heuristic emerges from
    /// the weight ratio: nz flips the sign only past
    /// `dz* ≈ w_fwd · L / (2 · w_nz)`.
    pub nz_dz_m: Option<f32>,
    /// Per-segment multiplier on `cfg.w_nz`. Lets a segment whose nz
    /// evidence is known-degraded (the shoulder line, whose source z
    /// is SYNTHESIZED by the legacy yaw prior rather than measured)
    /// keep nz as a sign tie-break only, without letting it drag the
    /// magnitude away from the length invariant.
    pub nz_weight: f32,
    /// One-sided length constraint: penalise only EXCEEDING
    /// `length_m`, never demand reaching it. For segments whose
    /// reference length is a noisy session estimate (the shoulder
    /// span), a two-sided residual actively fabricates depth out of
    /// estimation error — the sqrt amplifies a 5% over-reference into
    /// ~17° of fake twist. One-sided keeps the anatomy CAP without
    /// the fabrication.
    pub one_sided: bool,
    /// Apply the weak forward prior (penalise the distal end sitting
    /// behind the proximal one). True for arm segments in the webcam
    /// setting; false for legs/torso.
    pub forward_prior: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct RayIkConfig {
    /// Bone-length residual weight (strong — the geometric invariant).
    pub w_len: f32,
    /// Metric depth evidence weight (medium).
    pub w_depth: f32,
    /// nz consistency weight (weak — overridden by the length
    /// invariant, decides ties).
    pub w_nz: f32,
    /// Temporal continuity weight (light; dominates only when the
    /// frame's own evidence is weak).
    pub w_temp: f32,
    /// One-sided forward prior weight. The `w_fwd / w_nz` ratio sets
    /// the decisive-backward deadband (see [`SegmentSpec::nz_dz_m`]):
    /// 0.3 / 0.5 reproduces the old `0.15 × span` threshold at
    /// forearm scale.
    pub w_fwd: f32,
    pub iterations: usize,
    /// Depth floor in metres; keeps the solve away from the camera
    /// origin singularity.
    pub min_depth_m: f32,
}

impl Default for RayIkConfig {
    fn default() -> Self {
        Self {
            w_len: 10.0,
            w_depth: 3.0,
            w_nz: 0.5,
            w_temp: 0.5,
            w_fwd: 0.3,
            iterations: 10,
            min_depth_m: 0.05,
        }
    }
}

/// Frame-to-frame state: last solved depth per joint, feeding the
/// temporal term and the warm start.
#[derive(Default, Clone, Debug)]
pub struct RayIkState {
    prev_t: HashMap<HumanoidBone, f32>,
}

impl RayIkState {
    pub fn reset(&mut self) {
        self.prev_t.clear();
    }
}

/// Solve depths and return camera-space metric positions per joint.
///
/// Joints absent from `obs` are simply not solved (segments touching
/// them are dropped); the caller keeps whatever fallback it has for
/// those. The solve is a damped Gauss-Newton over the depth vector
/// with numeric Jacobians — the residual set is tiny (≈ joints +
/// segments), so forward differences cost microseconds and avoid
/// hand-derived gradient bugs.
pub fn solve(
    obs: &[RayObservation],
    segments: &[SegmentSpec],
    cfg: &RayIkConfig,
    state: &mut RayIkState,
) -> HashMap<HumanoidBone, [f32; 3]> {
    let n = obs.len();
    if n == 0 {
        return HashMap::new();
    }
    let is_fixed: Vec<bool> = obs
        .iter()
        .map(|o| o.fixed && o.metric_depth_m.is_some())
        .collect();
    let vars: Vec<usize> = (0..n).filter(|&i| !is_fixed[i]).collect();

    let index: HashMap<HumanoidBone, usize> =
        obs.iter().enumerate().map(|(i, o)| (o.bone, i)).collect();

    // Segments with both endpoints observed, as index pairs.
    let segs: Vec<(usize, usize, &SegmentSpec)> = segments
        .iter()
        .filter_map(|s| {
            let (Some(&a), Some(&b)) = (index.get(&s.a), index.get(&s.b)) else {
                return None;
            };
            Some((a, b, s))
        })
        .collect();

    // ---- Initialisation: closed-form basin selection ---------------------
    // The bone-length constraint admits exactly two depths for the
    // distal end given the proximal one (`t_b = p·d ± sqrt(L² − h²)`):
    // the toward-camera and behind basins. Gauss-Newton cannot be
    // trusted to pick between them from a symmetric start, so the
    // choice is made HERE, per segment, by scoring both candidates
    // against the side evidence (nz, forward prior, temporal). GN then
    // only polishes within the chosen basins. Metric depth evidence
    // pins a joint outright.
    let mut t = vec![f64::NAN; n];
    for (i, o) in obs.iter().enumerate() {
        if let Some(d) = o.metric_depth_m {
            t[i] = (d as f64).max(cfg.min_depth_m as f64);
        }
    }
    let prev_of = |i: usize| state.prev_t.get(&obs[i].bone).map(|v| *v as f64);
    // Roots without evidence seed from the previous frame.
    if t.iter().all(|v| !v.is_finite()) {
        for i in 0..n {
            if let Some(p) = prev_of(i) {
                t[i] = p;
                break;
            }
        }
    }
    let fallback = t
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0, f64::max)
        .max(1.0);

    // Score a candidate depth for joint `j` of segment `s` whose other
    // end sits at `t_other` — the same side terms the GN cost uses.
    let basin_score = |s: &SegmentSpec, za: f64, zb: f64, tb: f64, prev: Option<f64>| -> f64 {
        let mut c = 0.0;
        if let Some(dz) = s.nz_dz_m {
            c += cfg.w_nz as f64 * s.nz_weight as f64 * ((zb - za) - dz as f64).powi(2);
        }
        if s.forward_prior {
            c += cfg.w_fwd as f64 * (zb - za).max(0.0).powi(2);
        }
        if let Some(p) = prev {
            // Basin HYSTERESIS: stronger than the GN-time temporal
            // weight. The toward/behind basins sit ~2·sqrt(L²−h²)
            // apart, and per-frame nz noise routinely exceeds the
            // plain w_temp cost gap — real-footage replay showed a
            // raised-palms wrist flipping z = +0.84 ↔ −0.9 on
            // alternating frames, dragging the avatar's hand across
            // its face. The incumbent basin must be beaten decisively,
            // not by noise; once selected, GN refines within it at the
            // normal temporal weight.
            c += cfg.w_temp as f64 * BASIN_HYSTERESIS * (tb - p).powi(2);
        }
        c
    };
    // Two candidate depths for the joint whose ray is `d`, given the
    // other end's position `p` and the segment length. Falls back to
    // the closest-approach depth when the observation projects wider
    // than the bone (perspective magnification, handled by GN with
    // evidence).
    let candidates = |p: [f64; 3], d: [f32; 3], len: f64| -> (f64, f64) {
        let pd = p[0] * d[0] as f64 + p[1] * d[1] as f64 + p[2] * d[2] as f64;
        let disc = pd * pd - (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) + len * len;
        if disc <= 0.0 {
            (pd, pd)
        } else {
            let s = disc.sqrt();
            (pd - s, pd + s)
        }
    };
    // Proximal-first sweep (callers list segments proximal→distal);
    // two passes so late anchors still propagate backwards.
    for _ in 0..2 {
        for &(a, b, s) in &segs {
            if s.one_sided {
                // A cap does not pin the distal depth — nothing to
                // propagate from it.
                continue;
            }
            let (from, to) = if t[a].is_finite() && !t[b].is_finite() {
                (a, b)
            } else if t[b].is_finite() && !t[a].is_finite() {
                (b, a)
            } else {
                continue;
            };
            let df = obs[from].ray;
            let p = [
                df[0] as f64 * t[from],
                df[1] as f64 * t[from],
                df[2] as f64 * t[from],
            ];
            let dt = obs[to].ray;
            let (c1, c2) = candidates(p, dt, s.length_m as f64);
            let zf = p[2];
            let pick = |cand: f64| -> f64 {
                let cand = cand.max(cfg.min_depth_m as f64);
                let zc = dt[2] as f64 * cand;
                // `b` is distal in the spec; orient the (za, zb) pair
                // accordingly regardless of sweep direction.
                if to == b {
                    basin_score(s, zf, zc, cand, prev_of(to))
                } else {
                    basin_score(s, zc, zf, cand, prev_of(to))
                }
            };
            t[to] = if pick(c1) <= pick(c2) { c1 } else { c2 };
        }
    }
    for v in t.iter_mut() {
        if !v.is_finite() {
            *v = fallback;
        }
        *v = v.max(cfg.min_depth_m as f64);
    }

    // ---- Damped Gauss-Newton --------------------------------------------
    let residuals = |t: &[f64]| -> Vec<f64> {
        let mut r = Vec::with_capacity(segs.len() * 3 + n * 2);
        let pos = |i: usize, t: &[f64]| -> [f64; 3] {
            let d = obs[i].ray;
            [d[0] as f64 * t[i], d[1] as f64 * t[i], d[2] as f64 * t[i]]
        };
        for &(a, b, s) in &segs {
            let w_obs = (obs[a].confidence.min(obs[b].confidence) as f64).max(0.05);
            let pa = pos(a, t);
            let pb = pos(b, t);
            let len = ((pa[0] - pb[0]).powi(2)
                + (pa[1] - pb[1]).powi(2)
                + (pa[2] - pb[2]).powi(2))
            .sqrt();
            let len_err = if s.one_sided {
                (len - s.length_m as f64).max(0.0)
            } else {
                len - s.length_m as f64
            };
            r.push(len_err * (cfg.w_len as f64 * w_obs).sqrt());
            if let Some(dz) = s.nz_dz_m {
                if s.nz_weight > 0.0 {
                    r.push(
                        ((pb[2] - pa[2]) - dz as f64)
                            * (cfg.w_nz as f64 * s.nz_weight as f64 * w_obs).sqrt(),
                    );
                }
            }
            if s.forward_prior {
                // One-sided: only penalise the distal joint sitting
                // BEHIND the proximal one.
                let behind = (pb[2] - pa[2]).max(0.0);
                r.push(behind * (cfg.w_fwd as f64 * w_obs).sqrt());
            }
        }
        for (i, o) in obs.iter().enumerate() {
            if is_fixed[i] {
                continue; // constant — no evidence residuals needed
            }
            let w_obs = (o.confidence as f64).max(0.05);
            if let Some(d) = o.metric_depth_m {
                if o.depth_weight > 0.0 {
                    r.push(
                        (t[i] - d as f64)
                            * (cfg.w_depth as f64 * o.depth_weight as f64 * w_obs).sqrt(),
                    );
                }
            }
            if let Some(prev) = state.prev_t.get(&o.bone) {
                r.push((t[i] - *prev as f64) * (cfg.w_temp as f64).sqrt());
            }
        }
        r
    };

    let cost = |r: &[f64]| -> f64 { r.iter().map(|v| v * v).sum() };

    let nv = vars.len();
    let mut lambda = 1e-3;
    let mut r0 = residuals(&t);
    let mut c0 = cost(&r0);
    for _ in 0..cfg.iterations {
        if nv == 0 {
            break;
        }
        let m = r0.len();
        // Numeric Jacobian over the free variables, forward differences.
        const H: f64 = 1e-4;
        let mut jac = vec![vec![0.0f64; nv]; m];
        for (jv, &j) in vars.iter().enumerate() {
            let mut tp = t.clone();
            tp[j] += H;
            let rp = residuals(&tp);
            for k in 0..m {
                jac[k][jv] = (rp[k] - r0[k]) / H;
            }
        }
        // Normal equations A = JᵀJ + λ·diag, g = Jᵀr.
        let mut a = vec![vec![0.0f64; nv]; nv];
        let mut g = vec![0.0f64; nv];
        for k in 0..m {
            for i in 0..nv {
                let jki = jac[k][i];
                if jki == 0.0 {
                    continue;
                }
                g[i] += jki * r0[k];
                for j in i..nv {
                    a[i][j] += jki * jac[k][j];
                }
            }
        }
        for i in 0..nv {
            for j in 0..i {
                a[i][j] = a[j][i];
            }
        }

        // Levenberg damping with retry on cost increase.
        let mut accepted = false;
        for _ in 0..4 {
            let mut ad = a.clone();
            for i in 0..nv {
                ad[i][i] += lambda * (a[i][i].max(1e-9));
            }
            let Some(step) = solve_spd(&ad, &g) else {
                lambda *= 10.0;
                continue;
            };
            let mut tn = t.clone();
            for (iv, &i) in vars.iter().enumerate() {
                tn[i] = (tn[i] - step[iv]).max(cfg.min_depth_m as f64);
            }
            let rn = residuals(&tn);
            let cn = cost(&rn);
            if cn < c0 {
                t = tn;
                r0 = rn;
                c0 = cn;
                lambda = (lambda * 0.3).max(1e-6);
                accepted = true;
                break;
            }
            lambda *= 10.0;
        }
        if !accepted {
            break;
        }
    }

    // ---- Output ----------------------------------------------------------
    let mut out = HashMap::with_capacity(n);
    for (i, o) in obs.iter().enumerate() {
        let ti = t[i] as f32;
        state.prev_t.insert(o.bone, ti);
        out.insert(
            o.bone,
            [o.ray[0] * ti, o.ray[1] * ti, o.ray[2] * ti],
        );
    }
    // Joints that vanished this frame keep their stale prev_t for the
    // temporal term on re-entry; that is intentional (mirrors the
    // wrist temporal hold the design retires).
    out
}

/// Multiplier on `w_temp` used ONLY during basin-candidate scoring
/// (see `basin_score`); the GN refinement keeps the plain `w_temp`.
const BASIN_HYSTERESIS: f64 = 4.0;

/// Cholesky solve for a small symmetric positive-definite system.
/// Returns `None` if the factorisation hits a non-positive pivot.
fn solve_spd(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    // Forward + back substitution.
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ray_through(nx: f32, ny: f32, half_w: f32, half_h: f32) -> [f32; 3] {
        let x = (nx - 0.5) * 2.0 * half_w;
        let y = (ny - 0.5) * 2.0 * half_h;
        let len = (x * x + y * y + 1.0).sqrt();
        [x / len, y / len, 1.0 / len]
    }

    /// 60° vertical FOV, 16:9 — mirrors the pipeline's assumed camera.
    fn cam() -> (f32, f32) {
        let half_h = (30.0f32).to_radians().tan();
        (half_h * 16.0 / 9.0, half_h)
    }

    fn obs(bone: HumanoidBone, nx: f32, ny: f32, depth: Option<f32>) -> RayObservation {
        let (hw, hh) = cam();
        RayObservation {
            bone,
            ray: ray_through(nx, ny, hw, hh),
            confidence: 0.9,
            metric_depth_m: depth,
            depth_weight: 1.0,
            fixed: false,
        }
    }

    fn seg(a: HumanoidBone, b: HumanoidBone, len: f32) -> SegmentSpec {
        SegmentSpec {
            a,
            b,
            length_m: len,
            nz_dz_m: None,
            nz_weight: 1.0,
            one_sided: false,
            forward_prior: true,
        }
    }

    fn seg_nz(a: HumanoidBone, b: HumanoidBone, len: f32, nz: f32) -> SegmentSpec {
        SegmentSpec {
            nz_dz_m: Some(nz),
            ..seg(a, b, len)
        }
    }

    fn depth_of(out: &HashMap<HumanoidBone, [f32; 3]>, bone: HumanoidBone) -> f32 {
        out[&bone][2]
    }

    /// Forearm pointing straight at the camera: elbow and wrist rays
    /// nearly coincide, so the missing bone length must be resolved as
    /// forward (toward-camera) depth.
    #[test]
    fn fully_foreshortened_forearm_gains_forward_depth() {
        let o = vec![
            obs(HumanoidBone::LeftUpperArm, 0.62, 0.40, Some(1.0)),
            obs(HumanoidBone::LeftLowerArm, 0.66, 0.52, None),
            obs(HumanoidBone::LeftHand, 0.665, 0.525, None),
        ];
        let s = vec![
            seg(HumanoidBone::LeftUpperArm, HumanoidBone::LeftLowerArm, 0.28),
            seg(HumanoidBone::LeftLowerArm, HumanoidBone::LeftHand, 0.25),
        ];
        let mut state = RayIkState::default();
        let out = solve(&o, &s, &RayIkConfig::default(), &mut state);
        let elbow_z = depth_of(&out, HumanoidBone::LeftLowerArm);
        let wrist_z = depth_of(&out, HumanoidBone::LeftHand);
        assert!(
            elbow_z - wrist_z > 0.20,
            "wrist must resolve toward the camera, elbow_z={elbow_z} wrist_z={wrist_z}"
        );
        // Bone length must hold in 3D.
        let e = out[&HumanoidBone::LeftLowerArm];
        let w = out[&HumanoidBone::LeftHand];
        let len = ((e[0] - w[0]).powi(2) + (e[1] - w[1]).powi(2) + (e[2] - w[2]).powi(2)).sqrt();
        assert!((len - 0.25).abs() < 0.02, "forearm length {len} != 0.25");
    }

    /// In-plane arm at full extension: projected extent already
    /// explains the bone length at the anchor depth, so depths stay
    /// flat.
    #[test]
    fn in_plane_arm_stays_flat() {
        // At depth 1.0 the image half-height is tan(30°) ≈ 0.577 m, so
        // Δnx of 0.122 ≈ 0.25 m at 16:9.
        let o = vec![
            obs(HumanoidBone::LeftUpperArm, 0.60, 0.40, Some(1.0)),
            obs(HumanoidBone::LeftLowerArm, 0.7365, 0.40, None),
            obs(HumanoidBone::LeftHand, 0.8585, 0.40, None),
        ];
        let s = vec![
            seg_nz(HumanoidBone::LeftUpperArm, HumanoidBone::LeftLowerArm, 0.28, 0.0),
            seg_nz(HumanoidBone::LeftLowerArm, HumanoidBone::LeftHand, 0.25, 0.0),
        ];
        let mut state = RayIkState::default();
        let out = solve(&o, &s, &RayIkConfig::default(), &mut state);
        let shoulder_z = depth_of(&out, HumanoidBone::LeftUpperArm);
        let wrist_z = depth_of(&out, HumanoidBone::LeftHand);
        assert!(
            (shoulder_z - wrist_z).abs() < 0.08,
            "in-plane arm should stay near-flat, shoulder_z={shoulder_z} wrist_z={wrist_z}"
        );
    }

    /// Decisive backward nz evidence overrides the forward prior.
    #[test]
    fn decisive_backward_nz_keeps_backward_sign() {
        let o = vec![
            obs(HumanoidBone::LeftUpperArm, 0.62, 0.40, Some(1.0)),
            obs(HumanoidBone::LeftLowerArm, 0.66, 0.52, None),
            obs(HumanoidBone::LeftHand, 0.665, 0.525, None),
        ];
        let s = vec![
            seg_nz(HumanoidBone::LeftUpperArm, HumanoidBone::LeftLowerArm, 0.28, 0.0),
            // Wrist decisively BEHIND the elbow.
            seg_nz(HumanoidBone::LeftLowerArm, HumanoidBone::LeftHand, 0.25, 0.22),
        ];
        let mut state = RayIkState::default();
        let out = solve(&o, &s, &RayIkConfig::default(), &mut state);
        let elbow_z = depth_of(&out, HumanoidBone::LeftLowerArm);
        let wrist_z = depth_of(&out, HumanoidBone::LeftHand);
        assert!(
            wrist_z > elbow_z + 0.1,
            "backward evidence must win, elbow_z={elbow_z} wrist_z={wrist_z}"
        );
    }

    /// A forearm tilted toward the lens projects (via perspective
    /// magnification of the nearer wrist) at near-full bone length.
    /// With depth evidence on the wrist the solve must absorb the
    /// magnification as a smaller depth while the bone length holds —
    /// NOT inflate the length (the running-max failure mode of the
    /// heuristic this replaces).
    ///
    /// Geometry is exact: wrist = elbow + (0.105, 0, −0.227) under a
    /// 60°/16:9 camera, reprojected to (nx, ny) by hand.
    #[test]
    fn near_lens_depth_evidence_absorbs_projection_magnification() {
        let o = vec![
            // Elbow at (0.41056, −0.11547, 1.0): range 1.08715.
            obs(HumanoidBone::LeftLowerArm, 0.70, 0.40, Some(1.08715)),
            // True wrist (0.51556, −0.11547, 0.7732): range 0.93647.
            obs(HumanoidBone::LeftHand, 0.82482, 0.37067, Some(0.93647)),
        ];
        let s = vec![seg_nz(
            HumanoidBone::LeftLowerArm,
            HumanoidBone::LeftHand,
            0.25,
            0.0, // SimCC nz is blind to the toward-camera tilt
        )];
        let mut state = RayIkState::default();
        let out = solve(&o, &s, &RayIkConfig::default(), &mut state);
        let wrist = out[&HumanoidBone::LeftHand];
        let elbow = out[&HumanoidBone::LeftLowerArm];
        assert!(
            wrist[2] < elbow[2] - 0.15,
            "wrist must resolve toward the camera, wrist_z={} elbow_z={}",
            wrist[2],
            elbow[2]
        );
        let len = ((elbow[0] - wrist[0]).powi(2)
            + (elbow[1] - wrist[1]).powi(2)
            + (elbow[2] - wrist[2]).powi(2))
        .sqrt();
        assert!(
            (len - 0.25).abs() < 0.03,
            "bone length must hold instead of inflating, got {len}"
        );
    }

    /// Hands-contact coherence: an inter-wrist link segment pulls two
    /// arms whose own evidence disagrees about depth into a common
    /// depth (palms together must not render with a depth gap).
    #[test]
    fn contact_segment_equalizes_wrist_depths() {
        let o = vec![
            obs(HumanoidBone::LeftUpperArm, 0.40, 0.40, Some(1.0)),
            obs(HumanoidBone::RightUpperArm, 0.60, 0.40, Some(1.0)),
            obs(HumanoidBone::LeftLowerArm, 0.45, 0.52, None),
            obs(HumanoidBone::RightLowerArm, 0.55, 0.52, None),
            // Conflicting wrist depth evidence: 0.80 m vs 1.05 m.
            obs(HumanoidBone::LeftHand, 0.50, 0.55, Some(0.80)),
            obs(HumanoidBone::RightHand, 0.505, 0.555, Some(1.05)),
        ];
        let mut s = vec![
            seg_nz(HumanoidBone::LeftUpperArm, HumanoidBone::LeftLowerArm, 0.28, 0.0),
            seg_nz(HumanoidBone::RightUpperArm, HumanoidBone::RightLowerArm, 0.28, 0.0),
            seg_nz(HumanoidBone::LeftLowerArm, HumanoidBone::LeftHand, 0.25, 0.0),
            seg_nz(HumanoidBone::RightLowerArm, HumanoidBone::RightHand, 0.25, 0.0),
        ];
        let base = solve(&o, &s, &RayIkConfig::default(), &mut RayIkState::default());
        let gap_base =
            (depth_of(&base, HumanoidBone::LeftHand) - depth_of(&base, HumanoidBone::RightHand)).abs();

        s.push(SegmentSpec {
            a: HumanoidBone::LeftHand,
            b: HumanoidBone::RightHand,
            length_m: 0.03,
            nz_dz_m: Some(0.0),
            nz_weight: 1.0,
            one_sided: false,
            forward_prior: false,
        });
        let out = solve(&o, &s, &RayIkConfig::default(), &mut RayIkState::default());
        let gap =
            (depth_of(&out, HumanoidBone::LeftHand) - depth_of(&out, HumanoidBone::RightHand)).abs();
        assert!(
            gap < 0.10,
            "contact link must equalize wrist depths, gap={gap} (baseline {gap_base})"
        );
        assert!(
            gap < gap_base * 0.5,
            "contact link must at least halve the depth gap, gap={gap} baseline={gap_base}"
        );
    }

    /// A fixed shoulder anchor must come back exactly where it was
    /// pinned — the arm solve may not drag the torso.
    #[test]
    fn fixed_anchor_does_not_move() {
        let mut o = vec![
            obs(HumanoidBone::LeftUpperArm, 0.62, 0.40, Some(1.0)),
            obs(HumanoidBone::LeftLowerArm, 0.66, 0.52, None),
            obs(HumanoidBone::LeftHand, 0.665, 0.525, None),
        ];
        o[0].fixed = true;
        let s = vec![
            seg(HumanoidBone::LeftUpperArm, HumanoidBone::LeftLowerArm, 0.28),
            seg(HumanoidBone::LeftLowerArm, HumanoidBone::LeftHand, 0.25),
        ];
        let mut state = RayIkState::default();
        let out = solve(&o, &s, &RayIkConfig::default(), &mut state);
        let sh = out[&HumanoidBone::LeftUpperArm];
        let range = (sh[0] * sh[0] + sh[1] * sh[1] + sh[2] * sh[2]).sqrt();
        assert!(
            (range - 1.0).abs() < 1e-5,
            "fixed shoulder must stay at its pinned range, got {range}"
        );
        // The arm still resolves toward the camera off the fixed base.
        assert!(depth_of(&out, HumanoidBone::LeftHand) < depth_of(&out, HumanoidBone::LeftLowerArm));
    }

    /// With no fresh evidence beyond the rays, the temporal term keeps
    /// the solution near the previous frame instead of snapping.
    #[test]
    fn temporal_term_stabilises_ambiguous_frames() {
        let o = vec![
            obs(HumanoidBone::LeftUpperArm, 0.62, 0.40, Some(1.0)),
            obs(HumanoidBone::LeftLowerArm, 0.66, 0.52, None),
            obs(HumanoidBone::LeftHand, 0.665, 0.525, None),
        ];
        let s = vec![
            seg(HumanoidBone::LeftUpperArm, HumanoidBone::LeftLowerArm, 0.28),
            seg(HumanoidBone::LeftLowerArm, HumanoidBone::LeftHand, 0.25),
        ];
        let mut state = RayIkState::default();
        let first = solve(&o, &s, &RayIkConfig::default(), &mut state);
        let second = solve(&o, &s, &RayIkConfig::default(), &mut state);
        let d1 = depth_of(&first, HumanoidBone::LeftHand);
        let d2 = depth_of(&second, HumanoidBone::LeftHand);
        assert!(
            (d1 - d2).abs() < 0.02,
            "same observation must re-solve to the same depths, d1={d1} d2={d2}"
        );
    }
}
