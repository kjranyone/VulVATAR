//! Analytic re-seeding of limb chains from metric keypoint observations.
//!
//! Gradient descent cannot cross the depth-mirror ambiguity of an arm (the
//! 2-D projection is identical for an elbow in front of or behind the
//! shoulder plane), so when depth-lifted joints exist the arm is also
//! solved *analytically* from them and offered to the estimator as an
//! alternative starting point; the LM run from each seed is compared by
//! total cost and the better basin wins.

use super::math::*;
use super::model::*;

/// Rotation taking the orthonormal pair `(a1, a2)` onto `(b1, b2)`
/// (`a2 ⟂ a1`, `b2 ⟂ b1` assumed after re-orthogonalisation).
fn rotation_from_pairs(a1: V3, a2: V3, b1: V3, b2: V3) -> M3 {
    let a1 = normalize(a1);
    let a2 = normalize(sub(a2, scale(a1, dot(a1, a2))));
    let a3 = cross(a1, a2);
    let b1 = normalize(b1);
    let b2 = normalize(sub(b2, scale(b1, dot(b1, b2))));
    let b3 = cross(b1, b2);
    // R = B Aᵀ with columns a_i / b_i.
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = b1[i] * a1[j] + b2[i] * a2[j] + b3[i] * a3[j];
        }
    }
    orthonormalize(&r)
}

/// Re-seed one arm so its elbow / wrist land on the observed camera-frame
/// points. `elbow` may be `None` (out of frame): the elbow is then placed by
/// two-bone IK with the current elbow as the swivel pole. Returns `None` if
/// the observation is unusable (degenerate direction).
pub fn seed_arm(h: &Humanoid, st: &State, left: bool, elbow: Option<V3>, wrist: V3) -> Option<State> {
    let m = &h.model;
    let (j_sh, j_el, j_wr, j_clav) = if left {
        (h.j.l_shoulder, h.j.l_elbow, h.j.l_wrist, h.j.l_clav)
    } else {
        (h.j.r_shoulder, h.j.r_elbow, h.j.r_wrist, h.j.r_clav)
    };
    let fk = m.fk(st);
    let sh = fk.t[j_sh];
    let l1 = norm(fk.off[j_el]);
    let l2 = norm(fk.off[j_wr]);
    if l1 < 1e-3 || l2 < 1e-3 {
        return None;
    }
    let el = match elbow {
        Some(e) => e,
        None => {
            // Two-bone IK toward the wrist with the current elbow as pole.
            let to = sub(wrist, sh);
            let d = norm(to);
            if d < 1e-3 {
                return None;
            }
            let reach = (l1 + l2) * 0.999;
            let d = d.min(reach);
            let dir = normalize(to);
            // Law of cosines: distance of elbow along `dir` and off-axis radius.
            let a = ((l1 * l1 - l2 * l2 + d * d) / (2.0 * d)).clamp(-l1, l1);
            let r = (l1 * l1 - a * a).max(0.0).sqrt();
            let pole = sub(fk.t[j_el], sh);
            let mut perp = sub(pole, scale(dir, dot(pole, dir)));
            if norm(perp) < 1e-4 {
                // Fallback pole: downward-ish (camera +y).
                perp = sub([0.0, 1.0, 0.0], scale(dir, dir[1]));
            }
            let perp = normalize(perp);
            add(sh, add(scale(dir, a), scale(perp, r)))
        }
    };
    let d1 = sub(el, sh);
    let d2 = sub(wrist, el);
    if norm(d1) < 1e-3 || norm(d2) < 1e-3 {
        return None;
    }
    let d1n = normalize(d1);
    let d2n = normalize(d2);
    let mut n = cross(d1n, d2n);
    let sin_th = norm(n);
    let cos_th = dot(d1n, d2n);
    let theta = sin_th.atan2(cos_th);
    if sin_th < 0.05 {
        // Nearly straight arm: the flexion plane is undetermined; keep the
        // current hinge axis orientation, only align the upper arm.
        let cur_axis = match m.joints[j_el].kind {
            JointKind::Hinge { axis, .. } => mat_vec(&fk.r[j_sh], axis),
            _ => return None,
        };
        n = sub(cur_axis, scale(d1n, dot(cur_axis, d1n)));
        if norm(n) < 1e-4 {
            return None;
        }
    }
    let n = normalize(n);
    let e1 = m.joints[j_el].offset_dir; // upper-arm direction in the shoulder frame
    let a = match m.joints[j_el].kind {
        JointKind::Hinge { axis, .. } => axis,
        _ => return None,
    };
    let r_world = rotation_from_pairs(e1, a, d1n, n);
    let r_parent = fk.r[j_clav];
    let r_local = mat_mul(&transpose(&r_parent), &r_world);
    let mut out = st.clone();
    out.rot[j_sh] = orthonormalize(&r_local);
    let (lo, hi) = match m.joints[j_el].kind {
        JointKind::Hinge { lo, hi, .. } => (lo, hi),
        _ => (0.0, 2.6),
    };
    out.set_hinge(m, j_el, theta.clamp(lo, hi));
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_arm_lands_on_targets() {
        let h = Humanoid::new();
        let m = &h.model;
        let mut st = State::rest(m);
        st.set_relaxed(m);
        st.root_r = FACING_CAMERA;
        st.root_t = [0.0, 0.3, 1.5];
        // Target: a bent arm from a random-ish pose.
        let mut gt = st.clone();
        gt.set_ball(h.j.l_shoulder, [0.4, -0.3, -0.7]);
        gt.set_hinge(m, h.j.l_elbow, 1.3);
        gt.set_ball(h.j.r_shoulder, [-0.2, 0.5, 0.9]);
        gt.set_hinge(m, h.j.r_elbow, 0.8);
        let fkg = m.fk(&gt);
        for left in [true, false] {
            let (el, wr) = if left {
                (fkg.t[h.j.l_elbow], fkg.t[h.j.l_wrist])
            } else {
                (fkg.t[h.j.r_elbow], fkg.t[h.j.r_wrist])
            };
            let seeded = seed_arm(&h, &st, left, Some(el), wr).unwrap();
            let fk = m.fk(&seeded);
            let (e2, w2) = if left {
                (fk.t[h.j.l_elbow], fk.t[h.j.l_wrist])
            } else {
                (fk.t[h.j.r_elbow], fk.t[h.j.r_wrist])
            };
            assert!(norm(sub(e2, el)) < 1e-6, "elbow {e2:?} vs {el:?}");
            assert!(norm(sub(w2, wr)) < 1e-6, "wrist {w2:?} vs {wr:?}");
            // Wrist-only IK reaches the wrist too.
            let seeded = seed_arm(&h, &st, left, None, wr).unwrap();
            let fk = m.fk(&seeded);
            let w3 = if left { fk.t[h.j.l_wrist] } else { fk.t[h.j.r_wrist] };
            assert!(norm(sub(w3, wr)) < 1e-3, "IK wrist {w3:?} vs {wr:?}");
        }
    }
}
