use super::*;

fn intr() -> Intrinsics {
    Intrinsics {
        fx: 600.0,
        fy: 600.0,
        cx: 320.0,
        cy: 240.0,
        width: 640.0,
        height: 480.0,
    }
}

/// Render a state to 2-D keypoints for every joint + template site.
fn observe(model: &Model, st: &State, intr: Intrinsics, sigma: f64) -> Vec<Kp2d> {
    let fk = model.fk(st);
    let mut v = Vec::new();
    for j in 0..model.joints.len() {
        if let Some(uv) = intr.project(fk.t[j]) {
            v.push(Kp2d {
                point: ModelPoint::Joint(j),
                u: uv[0],
                v: uv[1],
                sigma,
            });
        }
    }
    for s in 0..model.sites.len() {
        if let Some(uv) = intr.project(fk.site[s]) {
            v.push(Kp2d {
                point: ModelPoint::Site(s),
                u: uv[0],
                v: uv[1],
                sigma,
            });
        }
    }
    v
}

pub(super) fn surface_from_capsules(model: &Model, st: &State, n_per: usize) -> Vec<(V3, f64)> {
    surface_from_capsules_parts(model, st, n_per, None)
}

pub(super) fn surface_from_capsules_parts(
    model: &Model,
    st: &State,
    n_per: usize,
    parts: Option<&[super::super::model::Part]>,
) -> Vec<(V3, f64)> {
    let fk = model.fk(st);
    let mut pts = Vec::new();
    for c in &model.capsules {
        if let Some(ps) = parts {
            if !ps.contains(&c.part) {
                continue;
            }
        }
        let a = fk.point(c.a);
        let b = fk.point(c.b);
        let r = model.capsule_radius(st, c);
        let g = CapGeom {
            a,
            b,
            c: c.lateral.map(|l| fk.point(l)),
            r,
            k: c.aspect,
        };
        for i in 0..n_per {
            let u = i as f64 / (n_per.max(2) - 1) as f64;
            let q = add(a, scale(sub(b, a), u));
            let axis = normalize(sub(b, a));
            let mut nrm = [0.0, 0.0, -1.0];
            nrm = sub(nrm, scale(axis, dot(nrm, axis)));
            let nrm = normalize(nrm);
            // Radius in this direction (round: r; elliptic: ρ).
            let p = add(q, scale(nrm, g.radius_along(nrm)));
            pts.push((p, 0.015));
        }
    }
    pts
}

fn gt_state(h: &Humanoid) -> State {
    let m = &h.model;
    let mut st = State::rest(m);
    st.set_relaxed(m);
    st.root_r = mat_mul(&so3_exp([0.1, 0.4, 0.05]), &FACING_CAMERA);
    st.root_t = [0.1, 0.35, 1.6];
    st.set_hinge(m, h.j.l_elbow, 1.1);
    st.set_ball(h.j.l_shoulder, [0.3, -0.5, -0.9]);
    st.set_ball(h.j.spine2, [0.2, 0.15, 0.0]);
    st.set_ball(h.j.head, [0.1, 0.5, 0.0]);
    st.set_hinge(m, h.j.r_knee, 0.5);
    st.set_hinge(m, h.j.finger[0][1][0], 0.9);
    st
}

#[test]
fn recovers_pose_from_2d_and_core_cloud() {
    recover_with(Some(&[
        super::super::model::Part::Torso,
        super::super::model::Part::Head,
    ]));
}

#[test]
fn recovers_pose_from_2d_and_cloud() {
    recover_with(None);
}

#[test]
fn recovers_pose_from_2d_only() {
    recover_with(Some(&[]));
}

fn recover_with(parts: Option<&[super::super::model::Part]>) {
    let h = Humanoid::new();
    let m = &h.model;
    let gt = gt_state(&h);
    let intr = intr();
    let fk_gt0 = m.fk(&gt);
    let kp3d: Vec<Kp3d> = [h.j.l_wrist, h.j.r_wrist, h.j.l_elbow, h.j.r_elbow]
        .iter()
        .map(|&j| Kp3d {
            point: ModelPoint::Joint(j),
            p: fk_gt0.t[j],
            sigma: 0.02,
            lat_scale: 1.0,
        })
        .collect();
    let obs = FrameObs {
        t: 0.0,
        intr: Some(intr),
        kp2d: observe(m, &gt, intr, 1.0),
        kp3d,
        ori: Vec::new(),
        shoulder_yaw: None,
        torso_hint: None,
        surface: surface_from_capsules_parts(m, &gt, 12, parts),
        surf_allow: Vec::new(),
    };
    let mut est = Estimator::new(m, Params::default());
    // Warm start: relaxed pose facing camera at roughly the right place.
    est.state.root_t = [0.0, 0.3, 1.5];
    let fk_gt = m.fk(&gt);
    // The production path always solves with the analytic arm seeds
    // (metric elbow / wrist → shoulder + elbow angles): a raised arm
    // from a relaxed warm start is a textbook robust-kernel local
    // minimum otherwise (measured: wrist stuck 21 cm off at 3-D cost
    // 52 while every other joint sat within 2 cm).
    for i in 0..20 {
        let mut o = obs.clone();
        o.t = i as f64 / 30.0;
        super::super::seed::update_with_arm_seeds(&h, &mut est, &o);
    }
    let fk = m.fk(&est.state);
    let mut worst = 0.0f64;
    let mut worst_name = "";
    for j in 0..m.joints.len() {
        // Hand capsules never take surface points (the landmarker owns
        // the hands), so the fingers are unobserved in this fixture.
        let name = m.joints[j].name;
        if name.contains("mcp")
            || name.contains("pip")
            || name.contains("dip")
            || name.contains("tip")
            || name.contains("thumb")
        {
            continue;
        }
        let e = norm(sub(fk.t[j], fk_gt.t[j]));
        if e > 0.02 {
            eprintln!("joint {} err {:.3} m", m.joints[j].name, e);
        }
        if e > worst {
            worst = e;
            worst_name = m.joints[j].name;
        }
    }
    assert!(
        worst < 0.02,
        "worst joint error {worst:.4} m at {worst_name} (cost {:.2} → {:.2})",
        est.diag.cost_initial,
        est.diag.cost_final
    );
    // Head orientation recovered (world frame — the neck/head split of
    // a pure yaw is unobservable from head points alone and is decided
    // by the pose prior).
    let dr = so3_log(&mat_mul(&fk.r[h.j.head], &transpose(&fk_gt.r[h.j.head])));
    assert!(norm(dr) < 0.08, "head world rotation error {:?}", dr);
    // Observed joints have small σ; the never-observed right-hand
    // fingers keep a large one.
    assert!(est.joint_sigma(m, h.j.l_elbow) < 0.1);
}

#[test]
fn unobserved_joint_variance_grows_and_relaxes() {
    let h = Humanoid::new();
    let m = &h.model;
    let gt = gt_state(&h);
    let intr = intr();
    let full = observe(m, &gt, intr, 1.0);
    let mut est = Estimator::new(m, Params::default());
    est.state.root_t = [0.0, 0.3, 1.5];
    for i in 0..5 {
        est.update(
            m,
            &FrameObs {
                t: i as f64 / 30.0,
                intr: Some(intr),
                kp2d: full.clone(),
                kp3d: Vec::new(),
                ori: Vec::new(),
                shoulder_yaw: None,
                torso_hint: None,
                surface: Vec::new(),
                surf_allow: Vec::new(),
            },
        );
    }
    let sig_before = est.joint_sigma(m, h.j.l_elbow);
    // Now drop everything on the left arm.
    let l_arm_joints = [h.j.l_shoulder, h.j.l_elbow, h.j.l_elbow_twist, h.j.l_wrist];
    let partial: Vec<Kp2d> = full
        .iter()
        .cloned()
        .filter(|k| match k.point {
            ModelPoint::Joint(j) => {
                !l_arm_joints.contains(&j) && !(m.depth[j] > 6 && is_left_hand(&h, j))
            }
            ModelPoint::Site(s) => !is_left_hand(&h, m.sites[s].joint),
            _ => true,
        })
        .collect();
    for i in 5..60 {
        est.update(
            m,
            &FrameObs {
                t: i as f64 / 30.0,
                intr: Some(intr),
                kp2d: partial.clone(),
                kp3d: Vec::new(),
                ori: Vec::new(),
                shoulder_yaw: None,
                torso_hint: None,
                surface: Vec::new(),
                surf_allow: Vec::new(),
            },
        );
    }
    let sig_after = est.joint_sigma(m, h.j.l_elbow);
    assert!(sig_after > sig_before * 3.0, "{sig_before} → {sig_after}");
    // Elbow relaxed toward its prior mean.
    assert!((est.state.angle[h.j.l_elbow] - m.joints[h.j.l_elbow].prior_mean[0]).abs() < 0.3);
    // Torso still tracked.
    let fk_gt = m.fk(&gt);
    let fk = m.fk(&est.state);
    assert!(norm(sub(fk.t[h.j.neck], fk_gt.t[h.j.neck])) < 0.03);
}

fn is_left_hand(h: &Humanoid, mut j: usize) -> bool {
    loop {
        if j == h.j.l_wrist {
            return true;
        }
        match h.model.joints[j].parent {
            Some(p) => j = p,
            None => return false,
        }
    }
}

#[test]
fn predict_extrapolates_velocity() {
    let h = Humanoid::new();
    let m = &h.model;
    let intr = intr();
    let mut est = Estimator::new(m, Params::default());
    est.state.root_t = [0.0, 0.3, 1.5];
    let mut st = State::rest(m);
    st.set_relaxed(m);
    st.root_r = FACING_CAMERA;
    st.root_t = [0.0, 0.3, 1.5];
    for i in 0..20 {
        let mut s = st.clone();
        s.root_t[0] += 0.02 * i as f64; // 0.6 m/s to the right
        est.update(
            m,
            &FrameObs {
                t: i as f64 / 30.0,
                intr: Some(intr),
                kp2d: observe(m, &s, intr, 1.0),
                kp3d: Vec::new(),
                ori: Vec::new(),
                shoulder_yaw: None,
                torso_hint: None,
                surface: Vec::new(),
                surf_allow: Vec::new(),
            },
        );
    }
    let now = est.state.root_t[0];
    let ahead = est.predict(m, 19.0 / 30.0 + 0.05).root_t[0];
    assert!(ahead > now + 0.015, "now {now} ahead {ahead}");
}

/// Points placed on the surface by `radius_along` must measure zero
/// distance, inside points negative, outside positive — for both the
/// round and the elliptic section, interior and end caps.
#[test]
fn ellipse_distance_is_zero_on_its_own_surface() {
    let a = [0.0, 0.0, 1.0];
    let b = [0.0, 0.4, 1.0];
    for (c, k) in [(None, 1.0), (Some([0.3, 0.0, 1.0]), 0.6)] {
        let g = CapGeom {
            a,
            b,
            c,
            r: 0.17,
            k,
        };
        for i in 0..24 {
            let th = i as f64 * std::f64::consts::TAU / 24.0;
            let dir = [th.cos(), 0.0, th.sin()];
            for u in [0.1, 0.5, 0.9] {
                let q = add(a, scale(sub(b, a), u));
                let rho = g.radius_along(dir);
                let on = add(q, scale(dir, rho));
                let (d0, _, _, _) = g.dist(on);
                assert!(d0.abs() < 1e-6, "th {th:.2} u {u}: d {d0} rho {rho}");
                let (di, _, _, _) = g.dist(add(q, scale(dir, rho - 0.02)));
                let (dout, _, _, _) = g.dist(add(q, scale(dir, rho + 0.02)));
                assert!(di < -0.01 && di > -0.03, "inside {di}");
                assert!(dout > 0.01 && dout < 0.03, "outside {dout}");
            }
        }
        // End cap: 5 cm past the top along the axis, on the surface radius.
        let dir = [1.0, 0.0, 0.0];
        let rho = g.radius_along(dir);
        let p = add(add(b, [0.0, 0.05, 0.0]), scale(dir, rho));
        let (dcap, _, u, _) = g.dist(p);
        assert!(u >= 1.0 && dcap > 0.0 && dcap < 0.06, "cap d {dcap} u {u}");
    }
}

#[test]
fn surface_gradient_matches_finite_difference() {
    let h = Humanoid::new();
    let m = &h.model;
    let mut gt = State::rest(m);
    gt.set_relaxed(m);
    gt.root_r = FACING_CAMERA;
    gt.root_t = [0.1, 0.35, 1.6];
    gt.set_hinge(m, h.j.l_elbow, 1.1);
    let pts = super::tests::surface_from_capsules(m, &gt, 12);
    // Perturb slightly
    let mut st = gt.clone();
    let mut d = vec![0.0; m.num_params];
    d[ROOT_T] = 0.01;
    d[m.joint_param[h.j.l_elbow]] = 0.05;
    d[m.beta_scale] = 0.02;
    st.apply_delta(m, &d);
    let mut est = Estimator::new(m, Params::default());
    est.state = st.clone();
    let obs = FrameObs {
        t: 0.0,
        intr: None,
        kp2d: vec![],
        kp3d: vec![],
        ori: Vec::new(),
        shoulder_yaw: None,
        torso_hint: None,
        surface: pts.clone(),
        surf_allow: Vec::new(),
    };
    let prior_var = vec![1e9; m.num_params];
    est.surf_pts = pts;
    let fk = m.fk(&st);
    let c0 = est.accumulate(m, &obs, &fk, &prior_var, 0.033, true);
    let g: Vec<f64> = est.dense.g.clone();
    // finite difference of cost
    let sp2 = m.joint_param[h.j.spine2];
    let hd = m.joint_param[h.j.head];
    for k in [
        0,
        1,
        2,
        ROOT_T,
        ROOT_T + 1,
        ROOT_T + 2,
        m.joint_param[h.j.l_elbow],
        m.beta_scale,
        m.beta_rad,
        m.beta_shear,
        sp2,
        sp2 + 1,
        sp2 + 2,
        hd,
        hd + 1,
    ] {
        let eps = 1e-6;
        let mut sp = st.clone();
        let mut dd = vec![0.0; m.num_params];
        dd[k] = eps;
        sp.apply_delta(m, &dd);
        let fkp = m.fk(&sp);
        let cp = est.eval_cost(m, &obs, &fkp, &sp, &prior_var, 0.033);
        let mut sm = st.clone();
        dd[k] = -eps;
        sm.apply_delta(m, &dd);
        let fkm = m.fk(&sm);
        let cm = est.eval_cost(m, &obs, &fkm, &sm, &prior_var, 0.033);
        let num = (cp - cm) / (2.0 * eps);
        // ∂cost/∂x = 2 g (cost = Σ ρ, g = Σ w Jᵀ r)
        assert!(
            (num - 2.0 * g[k]).abs() < 1e-3 * (1.0 + num.abs()),
            "param {k}: analytic 2g {} vs numeric {}",
            2.0 * g[k],
            num
        );
    }
    assert!(c0.is_finite());
}
