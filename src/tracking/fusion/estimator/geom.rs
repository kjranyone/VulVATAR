//! Geometry and Jacobian helpers of the solve: parameter lock / trunk
//! masks, point Jacobians, the swivel-angle helper, ray-capsule entry,
//! and the capsule proximity geometry shared by the residual blocks.
//! Re-exported from the estimator root where public.

use super::ModelPoint;
use crate::tracking::fusion::{math::*, model::*};

/// Locked parameters: the pelvis joint (redundant with the root rotation).
pub(super) fn locked_params(model: &Model) -> Vec<bool> {
    let mut v = vec![false; model.num_params];
    for (j, jd) in model.joints.iter().enumerate() {
        if jd.name == "pelvis" {
            let p = model.joint_param[j];
            let n = match jd.kind {
                JointKind::Ball { .. } => 3,
                JointKind::Hinge { .. } => 1,
            };
            for k in 0..n {
                v[p + k] = true;
            }
        }
    }
    v
}

/// Append `sign ×` the point Jacobian of joint `j` to `out`.
pub(super) fn point_y_jac(
    model: &Model,
    st: &State,
    fk: &Fk,
    j: usize,
    sign: f64,
    out: &mut Vec<(usize, V3)>,
) {
    let mut tmp = Vec::with_capacity(48);
    model.point_jacobian(st, fk, PointRef::Joint(j), &mut tmp);
    for (i, v) in tmp {
        out.push((i, scale(v, sign)));
    }
}

/// Trunk parameters = joints with depth ≤ 4 in the tree that are not part
/// of an arm/leg chain (pelvis, spine1-3, neck, clavicles) — the shallow
/// torso joints. Determined by name prefix to stay model-generic.
pub(super) fn trunk_params(model: &Model) -> Vec<bool> {
    let mut v = vec![false; model.num_params];
    for (j, jd) in model.joints.iter().enumerate() {
        let trunk = matches!(
            jd.name,
            "pelvis" | "spine1" | "spine2" | "spine3" | "neck" | "head" | "l_clav" | "r_clav"
        );
        if trunk {
            let p = model.joint_param[j];
            let n = match jd.kind {
                JointKind::Ball { .. } => 3,
                JointKind::Hinge { .. } => 1,
            };
            for k in 0..n {
                v[p + k] = true;
            }
        }
    }
    v
}

/// Head-joint parameters — same shape as `trunk_params` but only the
/// "head" joint, so the estimator can give the head its own process
/// noise. Head yaw is weakly observed (circular capsule, profile-view
/// landmarks) and a too-loose random walk lets shallow-basin noise
/// oscillate it frame to frame.
pub(super) fn head_params(model: &Model) -> Vec<bool> {
    named_params(model, &["head"])
}

/// Finger-chain parameters: every joint that descends from a wrist
/// joint (MCP flex/abd + pip + dip, both hands). Finger curl is
/// 2-D-degenerate (a fist projects like an extended, rotated hand) so
/// the process noise is the main thing keeping unobserved curl still;
/// `Params::q_finger` lets it differ from the limb swing rate.
pub(super) fn finger_params(model: &Model) -> Vec<bool> {
    let wrist: Vec<usize> = model
        .joints
        .iter()
        .enumerate()
        .filter(|(_, jd)| jd.name == "l_wrist" || jd.name == "r_wrist")
        .map(|(j, _)| j)
        .collect();
    let mut in_chain = vec![false; model.joints.len()];
    // Walk children via parent links (few passes over a small tree).
    loop {
        let mut changed = false;
        for (j, jd) in model.joints.iter().enumerate() {
            if in_chain[j] {
                continue;
            }
            let touches = wrist.contains(&j)
                || jd.parent.map(|p| in_chain[p]).unwrap_or(false);
            if touches {
                in_chain[j] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut v = vec![false; model.num_params];
    for (j, jd) in model.joints.iter().enumerate() {
        if !in_chain[j] || jd.name == "l_wrist" || jd.name == "r_wrist" {
            continue; // the wrist itself stays a limb joint
        }
        let p = model.joint_param[j];
        let n = match jd.kind {
            JointKind::Ball { .. } => 3,
            JointKind::Hinge { .. } => 1,
        };
        for k in 0..n {
            v[p + k] = true;
        }
    }
    v
}

/// Parameters of joints whose name is in `names` (all DoF of each).
fn named_params(model: &Model, names: &[&str]) -> Vec<bool> {
    let mut v = vec![false; model.num_params];
    for (j, jd) in model.joints.iter().enumerate() {
        if !names.contains(&jd.name) {
            continue;
        }
        let p = model.joint_param[j];
        let n = match jd.kind {
            JointKind::Ball { .. } => 3,
            JointKind::Hinge { .. } => 1,
        };
        for k in 0..n {
            v[p + k] = true;
        }
    }
    v
}

/// Arm-chain parameters: the finger subtrees (see `finger_params`) plus
/// the wrist / elbow / shoulder joints of both sides. Used to mask
/// hand-side observation Jacobians away from the trunk (`arm_param`).
pub(super) fn arm_params(model: &Model) -> Vec<bool> {
    let mut v = finger_params(model);
    for j in 0..model.joints.len() {
        if !matches!(
            model.joints[j].name,
            "l_shoulder" | "l_elbow" | "l_wrist" | "r_shoulder" | "r_elbow" | "r_wrist"
        ) {
            continue;
        }
        let p = model.joint_param[j];
        let n = match model.joints[j].kind {
            JointKind::Ball { .. } => 3,
            JointKind::Hinge { .. } => 1,
        };
        for k in 0..n {
            v[p + k] = true;
        }
    }
    v
}

/// Parameter-space difference `a ⊖ b` (left perturbation for rotations).
pub fn param_difference(model: &Model, a: &State, b: &State) -> Vec<f64> {
    let n = model.num_params;
    let mut d = vec![0.0; n];
    let dr = so3_log(&mat_mul(&a.root_r, &transpose(&b.root_r)));
    d[ROOT_ROT..ROOT_ROT + 3].copy_from_slice(&dr);
    let dt = sub(a.root_t, b.root_t);
    d[ROOT_T..ROOT_T + 3].copy_from_slice(&dt);
    for (j, jd) in model.joints.iter().enumerate() {
        let p = model.joint_param[j];
        match jd.kind {
            JointKind::Ball { .. } => {
                let w = so3_log(&mat_mul(&a.rot[j], &transpose(&b.rot[j])));
                d[p..p + 3].copy_from_slice(&w);
            }
            JointKind::Hinge { .. } => d[p] = a.angle[j] - b.angle[j],
        }
    }
    d[model.beta_scale] = a.scale - b.scale;
    for g in 0..NUM_LEN_GROUPS {
        d[model.beta_len + g] = a.len[g] - b.len[g];
    }
    for g in 0..NUM_RAD_GROUPS {
        d[model.beta_rad + g] = a.rad[g] - b.rad[g];
    }
    d[model.beta_shear] = a.shear - b.shear;
    d
}

/// World position + owning joint of a model point.
#[inline]
pub fn resolve_point(model: &Model, fk: &Fk, mp: ModelPoint) -> (usize, V3) {
    match mp {
        ModelPoint::Joint(j) => (j, fk.t[j]),
        ModelPoint::Site(s) => (model.sites[s].joint, fk.site[s]),
        ModelPoint::Attached { joint, local } => {
            (joint, add(fk.t[joint], mat_vec(&fk.r[joint], local)))
        }
    }
}

/// Jacobian of a model point (dispatches attached points, which move with
/// their joint's rotation but carry no shape dependence of their own).
pub(super) fn point_jac(
    model: &Model,
    st: &State,
    fk: &Fk,
    mp: ModelPoint,
    joint: usize,
    out: &mut Vec<(usize, V3)>,
) {
    match mp {
        ModelPoint::Joint(j) => model.point_jacobian(st, fk, PointRef::Joint(j), out),
        ModelPoint::Site(s) => model.point_jacobian(st, fk, PointRef::Site(s), out),
        ModelPoint::Attached { local, .. } => {
            let pw = add(fk.t[joint], mat_vec(&fk.r[joint], local));
            model.attached_point_jacobian(st, fk, joint, pw, out);
        }
    }
}

/// Signed swivel angle of the forearm about the shoulder→elbow axis,
/// measured from the gravity meridian (world down projected ⊥ the axis).
/// Retained for diagnostics / future forearm-frame holds: measured on the
/// s1789219959 desk replay, pinning this angle alone does NOT stop the
/// unobserved-wrist snaps (13 of 15 remain — the flexion DOF of the
/// forearm direction carries them too), see `Params::wrist_hold_sigma`.
/// `None` when the geometry is degenerate (zero-length segment, or the
/// arm axis within ~3° of gravity so the meridian is undefined).
#[allow(dead_code)]
fn swivel_angle(s: V3, e: V3, w: V3) -> Option<f64> {
    let u = sub(e, s);
    let lu = norm(u);
    if lu < 1e-6 {
        return None;
    }
    let a = scale(u, 1.0 / lu);
    let v = sub(w, e);
    let lv = norm(v);
    if lv < 1e-6 {
        return None;
    }
    let vhat = scale(v, 1.0 / lv);
    // World down (camera y is down), projected ⊥ the arm axis.
    let d: V3 = [0.0, 1.0, 0.0];
    if norm(cross(a, d)) < 0.05 {
        return None;
    }
    let r = sub(d, scale(a, dot(d, a)));
    let rhat = scale(r, 1.0 / norm(r));
    let x = dot(rhat, vhat);
    let y = dot(cross(rhat, vhat), a);
    Some(y.atan2(x))
}

/// Entry depth `t` (along the unit ray `dir` from the origin) of a capsule
/// (segment `ab`, radius `r`), or `None` if the ray misses it. Uses the
/// closest approach between the ray and the segment: with approach
/// distance `d* ≤ r` the entry is `t* − sqrt(r² − d*²)`.
pub fn ray_capsule_entry(dir: V3, a: V3, b: V3, r: f64) -> Option<f64> {
    // Closest points between ray o + t·dir (t ≥ 0, o = 0) and segment a + u·(b−a).
    let ab = sub(b, a);
    let l2 = dot(ab, ab);
    let (t_star, d_star) = if l2 < 1e-12 {
        let t = dot(a, dir).max(0.0);
        (t, norm(sub(scale(dir, t), a)))
    } else {
        // Alternating projection (convex; converges in a few steps):
        // given u → t = max(0, q·dir); given t → u = clamp(((t·dir − a)·ab)/|ab|²).
        let mut u = (dot(sub(scale(dir, dot(a, dir).max(0.0)), a), ab) / l2).clamp(0.0, 1.0);
        let mut t = 0.0;
        for _ in 0..4 {
            let q = add(a, scale(ab, u));
            t = dot(q, dir).max(0.0);
            u = (dot(sub(scale(dir, t), a), ab) / l2).clamp(0.0, 1.0);
        }
        let q = add(a, scale(ab, u));
        (t, norm(sub(scale(dir, t), q)))
    };
    if d_star > r {
        return None;
    }
    let back = (r * r - d_star * d_star).max(0.0).sqrt();
    Some((t_star - back).max(0.0))
}

/// Closest point on segment `ab` to `p`, with the parameter `u ∈ [0,1]`.
/// Capsule geometry for the surface term: axis `a→b`, lateral radius `r`,
/// optional lateral reference point `c` with depth/lateral aspect `k`.
#[derive(Clone, Debug)]
pub struct CapGeom {
    pub a: V3,
    pub b: V3,
    pub c: Option<V3>,
    pub r: f64,
    pub k: f64,
}

impl CapGeom {
    /// Surface radius in the direction `dir` (unit, perpendicular to the
    /// axis) from axis point `q`: `r` for a round capsule, the ellipse
    /// radius for an elliptic one. Exact surface points for fixtures.
    pub fn radius_along(&self, dir: V3) -> f64 {
        match self.c {
            None => self.r,
            Some(c) => {
                let t = normalize(sub(self.b, self.a));
                let mut l = sub(c, self.a);
                l = sub(l, scale(t, dot(l, t)));
                let l = normalize(l);
                let n = normalize(cross(t, l));
                let (cs, sn) = (dot(dir, l), dot(dir, n));
                let rb = (self.r * self.k).max(1e-6);
                1.0 / ((cs / self.r.max(1e-6)).powi(2) + (sn / rb).powi(2)).sqrt()
            }
        }
    }

    /// Signed radial distance of `p` from the surface, the axis point, the
    /// axis parameter, and the local surface radius ρ in `p`'s direction.
    /// Round capsule: `|p − q| − r`. Elliptic capsule: the section's radius
    /// in the direction of `p` (polar angle from the lateral axis) replaces
    /// `r`, so a flat chest faces where the points say it does.
    pub fn dist(&self, p: V3) -> (f64, V3, f64, f64) {
        let (q, u) = closest_on_segment(self.a, self.b, p);
        let v = sub(p, q);
        let dv = norm(v);
        match self.c {
            None => (dv - self.r, q, u, self.r),
            Some(c) => {
                let t = normalize(sub(self.b, self.a));
                let mut l = sub(c, self.a);
                l = sub(l, scale(t, dot(l, t)));
                let l = normalize(l);
                let n = normalize(cross(t, l));
                let vl = dot(v, l);
                let vn = dot(v, n);
                let vt = dot(v, t);
                let rr = (vl * vl + vn * vn).sqrt().max(1e-9);
                let (cs, sn) = (vl / rr, vn / rr);
                let ra = self.r.max(1e-6);
                let rb = (self.r * self.k).max(1e-6);
                // Radial section radius in this direction (used for the
                // radius Jacobian and the end caps).
                let rho = 1.0 / ((cs / ra).powi(2) + (sn / rb).powi(2)).sqrt();
                if vt.abs() > 1e-6 && (u <= 0.0 || u >= 1.0) {
                    // End cap: treat as an ellipsoidal cap with the local
                    // radius ρ.
                    return (dv - rho, q, u, rho);
                }
                // Interior: true Euclidean distance to the ellipse
                // (a cos t, b sin t) by Newton on the tangency condition
                // f(t) = (x − a cos t) a sin t − (y − b sin t) b cos t = 0.
                let (x, y) = (vl.abs(), vn.abs());
                let mut tt = (y * ra).atan2(x * rb);
                for _ in 0..3 {
                    let (st, ct) = tt.sin_cos();
                    let f = (x - ra * ct) * ra * st - (y - rb * st) * rb * ct;
                    let fp = ra * ra * st * st
                        + (x - ra * ct) * ra * ct
                        + rb * rb * ct * ct
                        + (y - rb * st) * rb * st;
                    if fp.abs() < 1e-12 {
                        break;
                    }
                    tt -= f / fp;
                    tt = tt.clamp(0.0, std::f64::consts::FRAC_PI_2);
                }
                let (st, ct) = tt.sin_cos();
                let (ex, ey) = (ra * ct, rb * st);
                let de = ((x - ex).powi(2) + (y - ey).powi(2)).sqrt();
                let inside = (x / ra).powi(2) + (y / rb).powi(2) < 1.0;
                (if inside { -de } else { de }, q, u, rho)
            }
        }
    }
}

#[inline]
pub fn closest_on_segment(a: V3, b: V3, p: V3) -> (V3, f64) {
    let ab = sub(b, a);
    let l2 = dot(ab, ab);
    if l2 < 1e-12 {
        return (a, 0.0);
    }
    let u = (dot(sub(p, a), ab) / l2).clamp(0.0, 1.0);
    (add(a, scale(ab, u)), u)
}
