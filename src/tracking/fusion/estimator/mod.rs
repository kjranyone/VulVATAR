//! MAP body-model fitter: fuses every observation of a frame into the
//! articulated [`Model`] state by damped Gauss–Newton (Levenberg–Marquardt)
//! and carries a diagonal covariance between frames (information-filter
//! prior with a constant-velocity process model).
//!
//! Residual blocks (all robust, all σ-weighted):
//! * 2-D reprojection of model points (joints, template sites, learned
//!   attached points such as face landmarks) into a pinhole camera;
//! * 3-D point observations of model points (depth-lifted landmarks);
//! * point cloud → capsule surface (dense torso / head / limb evidence);
//! * joint limits (soft hinge), pose prior (relaxed pose), shape prior;
//! * temporal prior `x ~ N(x̂, P + Q·dt)` from the previous posterior;
//! * wrist hold: a plain pseudo-observation pinning an unobserved wrist's
//!   root-frame position to the prediction (arm-twist null direction).
//!
//! Nothing here decides "is this joint visible": an unobserved joint simply
//! has no data terms, its variance grows by `Q·dt` per frame and the pose
//! prior pulls it toward the relaxed pose. Consumers read the marginal σ.

use super::math::*;
use super::model::*;

mod obs;
mod params;
mod geom;

pub use obs::{AngleObs, Intrinsics, ModelPoint, Kp2d, Kp3d, OriObs, ShoulderYawObs, FrameObs};
pub use params::Params;
pub use geom::{closest_on_segment, param_difference, ray_capsule_entry, resolve_point, CapGeom};
use geom::{
    arm_below_shoulder_params, arm_params, finger_params, head_params, locked_params, point_jac,
    point_y_jac, trunk_params,
};

/// Estimator with its carried temporal state.
pub struct Estimator {
    pub params: Params,
    pub state: State,
    /// Diagonal posterior variance per parameter (after the last solve).
    pub var: Vec<f64>,
    /// Parameter-space velocity (per second) for prediction.
    pub vel: Vec<f64>,
    pub last_t: Option<f64>,
    /// Number of solves performed.
    pub frames: u64,
    /// Whether the shape has been frozen (β prior tightened).
    pub shape_frozen: bool,
    dense: Dense,
    delta: Vec<f64>,
    jac: Vec<(usize, V3)>,
    jac2: Vec<(usize, V3)>,
    /// Predicted state used as the temporal-prior mean this frame.
    pred: State,
    /// Last solve diagnostics.
    pub diag: SolveDiag,
    /// Last solve wall-time breakdown (reset per `update_with_seeds`).
    pub timings: SolveTimings,
    /// Current GNC multiplier on the robust kernel scales.
    gnc: f64,
    /// Sparse-row scratch: dense accumulator + touched index list.
    row_acc: Vec<f64>,
    row_idx: Vec<usize>,
    row_out: Vec<(usize, f64)>,
    row_out2: Vec<(usize, f64)>,
    /// Per-capsule-end point Jacobians, rebuilt once per accumulate.
    cap_jac: Vec<Vec<(usize, V3)>>,
    /// Per-capsule association permission for the current frame (see
    /// `FrameObs::surf_allow`).
    surf_allow: Vec<bool>,
    /// Trunk axis sites (`torso_lo`, `torso_hi`) for the upright prior.
    trunk_sites: Option<(usize, usize)>,
    /// Diagnostics: capsule index chosen for each surface point in the
    /// last build pass (−1 = dropped).
    pub last_surf_assoc: Vec<i32>,
    lost_frames: u32,
    /// Number of re-acquisitions triggered by the health check.
    pub lost_events: u64,
    /// Sparse surface points of the current frame (set by `update`).
    surf_pts: Vec<([f64; 3], f64)>,
    idx_l_shoulder: usize,
    idx_r_shoulder: usize,
    idx_l_elbow: usize,
    idx_r_elbow: usize,
    /// Per-parameter flag: belongs to the trunk (slow process noise).
    trunk_param: Vec<bool>,
    /// Per-parameter flag: belongs to the head joint (own process noise,
    /// `Params::q_head`).
    head_param: Vec<bool>,
    /// Per-parameter flag: inside a finger chain (below the wrists) —
    /// `Params::q_finger`.
    finger_param: Vec<bool>,
    /// Per-parameter flag: the ARM chains (shoulder + elbow + wrist +
    /// finger subtree, both sides). With `VULVATAR_ARM_CHAIN_DEPTH=1`,
    /// observations resolving inside an arm chain have their Jacobian
    /// masked to these parameters: hand/wrist evidence may pose the arm
    /// but not drag the clavicle / spine / root — every hand-side σ
    /// adjustment measured flipping the profile-session torso basin
    /// (s1789349575: −23.7°→+59.5° hand σ, +4.3°→−64° finger σ).
    arm_param: Vec<bool>,
    /// Per-parameter flag: locked (never solved) — the pelvis ball, which is
    /// redundant with the root rotation.
    locked_param: Vec<bool>,
    /// Full posterior covariance (n×n, row-major) from the last solve.
    cov: Vec<f64>,
    /// Per-joint σ (rad) of the joint's *world* orientation, from the full
    /// covariance propagated along the kinematic chain (the local split
    /// between e.g. neck and head is prior-limited, the world orientation
    /// of the head is what the face landmarks pin).
    pub world_sigma: Vec<f64>,
    /// Data-only information (diag of H from measurement terms) at the last
    /// accumulate; smoothed into `data_info_ema` per frame.
    data_info: Vec<f64>,
    /// Leaky-integrated data information per parameter (tau ~ 0.3 s), so a
    /// joint that was measured a few frames ago still reads as observed.
    pub data_info_ema: Vec<f64>,
    /// Wrist joint indices (`l_wrist`, `r_wrist`; `(0, false)` if the
    /// model has none — holds disabled).
    wrist_joints: [(usize, bool); 2],
    /// Per-wrist hold target for THIS frame: `(active, wrist − root_t)`
    /// in the camera frame, taken from the prediction. `active` is false
    /// on bootstrap frames and on any frame where the wrist carries an
    /// observation (see `Params::wrist_hold_sigma`).
    hold_targets: [(bool, V3); 2],
    /// Trunk-stage observation policy (two-stage solve): keypoint / angle
    /// observations anchored in the arm chain and the wrist holds are
    /// excluded from the residual build — the arm chain is frozen there
    /// and its lagging prediction must not reach the trunk through the
    /// chain Jacobians. Set by `run_locked(LockSet::TrunkStage)`.
    arm_obs_masked: bool,
    /// Per-parameter flag: the arm chain BELOW the shoulder balls
    /// (elbow / twist / wrist / fingers). Unlike `arm_param` the shoulder
    /// itself is excluded: a shoulder-anchored keypoint is trunk-side
    /// evidence (clavicle / spine placement) and stays in the trunk
    /// stage, while the shoulder ball DoF stays frozen there.
    arm_below_shoulder: Vec<bool>,
}

/// Which parameter set a two-stage `run_locked` solve freezes on top of
/// the standing (pelvis) locks.
#[derive(Clone, Copy, Debug)]
enum LockSet {
    /// Trunk stage: arm chain frozen at the warm start.
    TrunkStage,
    /// Arm stage: root + trunk joints + shape frozen at the trunk stage.
    ArmStage,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SolveDiag {
    pub iters: usize,
    pub cost_initial: f64,
    pub cost_final: f64,
    pub n_kp2d: usize,
    pub n_kp3d: usize,
    pub n_cloud: usize,
    pub dt: f64,
    /// Final-state cost breakdown.
    pub cost_2d: f64,
    pub cost_3d: f64,
    pub cost_cloud: f64,
    pub cost_prior: f64,
    pub cost_temporal: f64,
    /// Cost of the wrist-hold pseudo-observations (active term only;
    /// 0 on frames where both wrists were observed). See
    /// `Params::wrist_hold_sigma`.
    pub cost_whold: f64,
    /// Number of 2-D observations that actually projected (model point in
    /// front of the camera). A collapsed state (body at/behind the camera
    /// plane) produces near-zero cost with zero projectable points — the
    /// health check treats that as lost, not as healthy.
    pub n_2d_proj: usize,
    /// Mean |residual| of the 2-D terms (px) and 3-D terms (m) at the
    /// final state, and mean signed cloud distance (m, + = point outside).
    pub rms_2d_px: f64,
    /// Median |residual| of the 2-D terms (px) — robust track-health signal.
    pub med_2d_px: f64,
    /// Median |residual| of sparse body/face terms (excluding hand crops / fingers).
    pub med_sparse_2d_px: f64,
    pub mean_3d_m: f64,
    pub mean_cloud_m: f64,
    /// Cumulative count of frames where an alternative seed won.
    pub seed_wins: u64,
    /// Cumulative count of frames whose covariance factorisation failed.
    pub cov_failures: u64,
}

/// Per-frame wall-time breakdown (ms) of one [`Estimator::update_with_seeds`].
/// `acc_ms` / `eval_ms` / `lin_ms` accumulate across every LM loop of the
/// frame (main + seed contests + re-accumulate), so together with the phase
/// totals they localise where the solve spends its budget.
#[derive(Clone, Copy, Debug, Default)]
pub struct SolveTimings {
    /// Residual + normal-equation accumulation passes.
    pub acc_ms: f64,
    /// Cost-only evaluation passes (the LM accept test).
    pub eval_ms: f64,
    /// Damped LDLᵀ linear solves.
    pub lin_ms: f64,
    /// The main LM loop (warm start from the prediction).
    pub main_ms: f64,
    /// Seed-contest LM loops.
    pub seeds_ms: f64,
    /// Winner re-accumulate + covariance restore.
    pub reacc_ms: f64,
    /// `finish()`: covariance inverse + σ propagation + bookkeeping.
    pub finish_ms: f64,
}

impl Estimator {
    pub fn new(model: &Model, params: Params) -> Self {
        let n = model.num_params;
        let mut state = State::rest(model);
        state.set_relaxed(model);
        state.root_r = FACING_CAMERA;
        state.root_t = [0.0, 0.3, 1.2];
        Self {
            params,
            pred: state.clone(),
            state,
            var: vec![1.0; n],
            vel: vec![0.0; n],
            last_t: None,
            frames: 0,
            shape_frozen: false,
            dense: Dense::new(n),
            delta: vec![0.0; n],
            jac: Vec::with_capacity(64),
            jac2: Vec::with_capacity(64),
            diag: SolveDiag::default(),
            timings: SolveTimings::default(),
            gnc: 1.0,
            row_acc: vec![0.0; n],
            row_idx: Vec::with_capacity(128),
            row_out: Vec::with_capacity(128),
            row_out2: Vec::with_capacity(128),
            cap_jac: Vec::new(),
            surf_allow: Vec::new(),
            last_surf_assoc: Vec::new(),
            trunk_sites: {
                // The prior must span the SKELETON axis (`axis_lo`/`axis_hi`,
                // unsheared). Spans that ride the trunk-shear sites would
                // constrain the capsule's shape-tilt instead of the pose,
                // freeing the skeleton to recline (measured on the
                // s1789219959 replay: pitch −11° → −23° once the shear was
                // introduced with the prior still on `torso_hi`).
                let find = |n: &str| model.sites.iter().position(|s| s.name == n);
                match (find("axis_lo"), find("axis_hi")) {
                    (Some(a), Some(b)) => Some((a, b)),
                    _ => None,
                }
            },
            lost_frames: 0,
            lost_events: 0,
            surf_pts: Vec::new(),
            idx_l_shoulder: model
                .joints
                .iter()
                .position(|j| j.name == "l_shoulder")
                .unwrap_or(0),
            idx_r_shoulder: model
                .joints
                .iter()
                .position(|j| j.name == "r_shoulder")
                .unwrap_or(0),
            idx_l_elbow: model
                .joints
                .iter()
                .position(|j| j.name == "l_elbow")
                .unwrap_or(0),
            idx_r_elbow: model
                .joints
                .iter()
                .position(|j| j.name == "r_elbow")
                .unwrap_or(0),
            trunk_param: trunk_params(model),
            head_param: head_params(model),
            finger_param: finger_params(model),
            arm_param: arm_params(model),
            locked_param: locked_params(model),
            data_info: vec![0.0; n],
            data_info_ema: vec![0.0; n],
            wrist_joints: [
                (
                    model
                        .joints
                        .iter()
                        .position(|j| j.name == "l_wrist")
                        .unwrap_or(0),
                    model.joints.iter().any(|j| j.name == "l_wrist"),
                ),
                (
                    model
                        .joints
                        .iter()
                        .position(|j| j.name == "r_wrist")
                        .unwrap_or(0),
                    model.joints.iter().any(|j| j.name == "r_wrist"),
                ),
            ],
            hold_targets: [(false, [0.0; 3]), (false, [0.0; 3])],
            arm_obs_masked: false,
            arm_below_shoulder: arm_below_shoulder_params(model),
            cov: Vec::new(),
            world_sigma: vec![1.0; model.joints.len()],
        }
    }

    /// Reset temporal state (new session): wide covariance, zero velocity,
    /// relaxed pose facing the camera.
    pub fn reset(&mut self, model: &Model) {
        let n = model.num_params;
        self.state = State::rest(model);
        self.state.set_relaxed(model);
        self.state.root_r = FACING_CAMERA;
        self.state.root_t = [0.0, 0.3, 1.2];
        self.pred = self.state.clone();
        self.var = vec![1.0; n];
        self.vel = vec![0.0; n];
        self.last_t = None;
        self.frames = 0;
        self.shape_frozen = false;
    }

    /// Predict the state at time `t` (seconds) from the last posterior:
    /// constant (damped) velocity in parameter space.
    pub fn predict(&self, model: &Model, t: f64) -> State {
        let dt = match self.last_t {
            Some(lt) => (t - lt).clamp(0.0, 0.25),
            None => 0.0,
        };
        let mut st = self.state.clone();
        if dt > 0.0 {
            let tau = self.params.velocity_tau.max(1e-3);
            // ∫₀^dt v e^{-s/τ} ds = v τ (1 − e^{−dt/τ})
            let f = tau * (1.0 - (-dt / tau).exp());
            let mut d: Vec<f64> = self.vel.iter().map(|v| v * f).collect();
            // Never extrapolate shape.
            for k in model.beta_scale..model.num_params {
                d[k] = 0.0;
            }
            st.apply_delta(model, &d);
        }
        st
    }

    /// Marginal σ (rad or m) of a joint (max over its components).
    pub fn joint_sigma(&self, model: &Model, j: usize) -> f64 {
        let p = model.joint_param[j];
        let n = match model.joints[j].kind {
            JointKind::Ball { .. } => 3,
            JointKind::Hinge { .. } => 1,
        };
        (0..n).map(|k| self.var[p + k]).fold(0.0, f64::max).sqrt()
    }

    /// World-orientation σ of a joint (rad), see `world_sigma`.
    pub fn joint_world_sigma(&self, j: usize) -> f64 {
        self.world_sigma.get(j).copied().unwrap_or(1.0)
    }

    /// Propagate the full covariance along each chain: to first order the
    /// world rotation perturbation of joint j is
    /// `Σ_k R_parent(k) δ_k + δ_root` over its ancestors (ball: 3 params in
    /// the parent frame; hinge: axis·δ), so
    /// `Cov_world = Σ_{k,l} A_k Σ_{kl} A_lᵀ` with `A_k` the 3×dof_k map.
    fn update_world_sigmas(&mut self, model: &Model) {
        let n = model.num_params;
        if self.cov.len() != n * n {
            return;
        }
        let fk = model.fk(&self.state);
        if self.world_sigma.len() != model.joints.len() {
            self.world_sigma = vec![1.0; model.joints.len()];
        }
        // Per joint: list of (param index, world-frame 3-vector column).
        let mut cols: Vec<(usize, V3)> = Vec::with_capacity(48);
        for j in 0..model.joints.len() {
            cols.clear();
            let mut k = j;
            loop {
                let jd = &model.joints[k];
                let pidx = model.joint_param[k];
                let pr = match jd.parent {
                    Some(pp) => fk.r[pp],
                    None => self.state.root_r,
                };
                match jd.kind {
                    JointKind::Ball { .. } => {
                        for c in 0..3 {
                            cols.push((pidx + c, col(&pr, c)));
                        }
                    }
                    JointKind::Hinge { axis, .. } => cols.push((pidx, mat_vec(&pr, axis))),
                }
                match jd.parent {
                    Some(pp) => k = pp,
                    None => break,
                }
            }
            for c in 0..3 {
                let mut e = [0.0; 3];
                e[c] = 1.0;
                cols.push((ROOT_ROT + c, e));
            }
            // Cov_world (3×3) = Σ a_i Σ_ij a_jᵀ
            let mut cw = [[0.0f64; 3]; 3];
            for &(i, ai) in &cols {
                for &(jj, aj) in &cols {
                    let s = self.cov[i * n + jj];
                    if s == 0.0 {
                        continue;
                    }
                    for r in 0..3 {
                        for q in 0..3 {
                            cw[r][q] += ai[r] * s * aj[q];
                        }
                    }
                }
            }
            let maxdiag = cw[0][0].max(cw[1][1]).max(cw[2][2]).max(0.0);
            self.world_sigma[j] = maxdiag.sqrt().min(5.0);
        }
    }

    /// Measurement-only σ of a joint (rad): 1/sqrt(leaky data information),
    /// ignoring the pose prior — large (capped at 10) when the joint has not
    /// been observed recently, regardless of how confident the prior makes
    /// the posterior look. Uses the best-observed component.
    pub fn joint_data_sigma(&self, model: &Model, j: usize) -> f64 {
        let p = model.joint_param[j];
        let n = match model.joints[j].kind {
            JointKind::Ball { .. } => 3,
            JointKind::Hinge { .. } => 1,
        };
        let info = (0..n)
            .map(|k| self.data_info_ema.get(p + k).copied().unwrap_or(0.0))
            .fold(0.0, f64::max);
        if info <= 1e-6 {
            10.0
        } else {
            (1.0 / info.sqrt()).min(10.0)
        }
    }

    /// σ of the root translation (max component, metres).
    pub fn root_sigma_m(&self) -> f64 {
        (0..3)
            .map(|k| self.var[ROOT_T + k])
            .fold(0.0, f64::max)
            .sqrt()
    }

    /// Ingest one frame's observations and update the posterior.
    pub fn update(&mut self, model: &Model, obs: &FrameObs) {
        self.update_with_seeds(model, obs, &[]);
    }

    /// Like [`Self::update`], but additionally tries each `seeds[i](&start)`
    /// (an alternative starting state derived from the warm start, e.g. an
    /// arm re-solved analytically from metric points) and keeps the LM
    /// result with the lowest total cost.
    pub fn update_with_seeds(
        &mut self,
        model: &Model,
        obs: &FrameObs,
        seeds: &[&dyn Fn(&State) -> Option<State>],
    ) {
        let n = model.num_params;
        self.timings = SolveTimings::default();
        let dt = match self.last_t {
            Some(lt) if obs.t > lt => (obs.t - lt).clamp(1e-3, 0.25),
            Some(_) => 1.0 / 30.0,
            None => 1.0 / 30.0,
        };
        // ---- predict --------------------------------------------------------
        let prev = self.state.clone();
        self.pred = self.predict(model, obs.t.max(self.last_t.unwrap_or(obs.t)));
        if self.last_t.is_none() {
            self.pred = self.state.clone();
        }
        // ---- wrist hold targets ---------------------------------------------
        // Root-frame wrist position from the prediction, active only when
        // the wrist carries no observation this frame (a returning
        // observation — including one a seed candidate relies on — must be
        // free to move the wrist; see `Params::wrist_hold_sigma`).
        {
            let joint_of = |mp: &ModelPoint| match mp {
                ModelPoint::Joint(j) => Some(*j),
                ModelPoint::Site(s) => model.sites.get(*s).map(|s| s.joint),
                ModelPoint::Attached { joint, .. } => Some(*joint),
            };
            let wrist_observed = [
                obs.kp2d.iter().any(|k| joint_of(&k.point) == Some(self.wrist_joints[0].0))
                    || obs.kp3d.iter().any(|k| joint_of(&k.point) == Some(self.wrist_joints[0].0)),
                obs.kp2d.iter().any(|k| joint_of(&k.point) == Some(self.wrist_joints[1].0))
                    || obs.kp3d.iter().any(|k| joint_of(&k.point) == Some(self.wrist_joints[1].0)),
            ];
            let pred_fk = model.fk(&self.pred);
            for k in 0..2 {
                let (wj, ok) = self.wrist_joints[k];
                // NOTE: activation hysteresis (2-frame delay) and a σ
                // ramp-in were both benched on the full recording set and
                // REJECTED (snaps 184→193 total, 10 sessions worse vs 7
                // better, 2026-09-14): delaying or softening the pin lets
                // brief hand-crop dropouts free-walk the wrist. Keep the
                // instant, fixed-σ hold.
                let active = ok && self.last_t.is_some() && !wrist_observed[k];
                self.hold_targets[k] = (active, sub(pred_fk.t[wj], self.pred.root_t));
            }
        }
        // Prior variance for this frame: P + Q dt (per parameter class).
        let mut prior_var = vec![0.0; n];
        for k in 0..n {
            let q = if k < 3 {
                self.params.q_root_rot
            } else if k < 6 {
                self.params.q_root_t
            } else if k < model.beta_scale {
                if self.head_param[k] {
                    self.params.q_head
                } else if self.trunk_param[k] {
                    self.params.q_trunk
                } else {
                    // Hold when unobserved: a limb nobody is looking at has
                    // no reason to move. The process noise (a random walk
                    // at human peak speed) is scaled by how much data the
                    // joint has been receiving, down to `q_hold_floor`,
                    // so an arm under the desk keeps its last pose instead
                    // of wandering toward every stray observation.
                    let info = self.data_info_ema.get(k).copied().unwrap_or(0.0);
                    let f =
                        (info / self.params.q_hold_info_ref).clamp(self.params.q_hold_floor, 1.0);
                    // Fingers share the hold scaling but have their own
                    // base rate: curl is a depth DOF the 2-D landmarks
                    // barely see, so the swing-scale q_joint is far too
                    // loose for them.
                    let base = if self.finger_param[k] {
                        self.params.q_finger
                    } else {
                        self.params.q_joint
                    };
                    base * f
                }
            } else {
                self.params.q_shape
            };
            prior_var[k] = (self.var[k] + q * dt).clamp(self.params.var_min, self.params.var_max);
        }
        // Warm start from the prediction.
        self.state = self.pred.clone();
        self.surf_allow = obs.surf_allow.clone();
        self.surf_pts = obs
            .surface
            .iter()
            .map(|(p, s)| {
                let s2 =
                    (s * s + self.params.surf_model_sigma * self.params.surf_model_sigma).sqrt();
                (*p, s2)
            })
            .collect();
        // Junk-burst guard: the dense cloud is sampled from the depth
        // person-mask, which is produced by the SAME detector pipeline
        // whose sparse keypoints we can watch. When the previous frame's
        // sparse residual was already at lost-level (> `lost_rms_px`),
        // the mask is not trustworthy either — during such bursts the
        // cloud has been measured covering room walls (1 657 points at
        // z≈3.2 m) and dragging the root 1.8→3.2 m into a permanent lock
        // (s1789279985 frames 150–209). Drop the cloud for that frame;
        // the temporal prior carries the pose through the burst instead
        // of a hard `mark_lost` reset. `VULVATAR_FUSION_NO_JUNKCLOUD=1`
        // disables.
        static JUNKCLOUD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let junkcloud = *JUNKCLOUD.get_or_init(|| {
            std::env::var_os("VULVATAR_FUSION_NO_JUNKCLOUD").is_none()
        });
        if junkcloud
            && self.last_t.is_some()
            && self.diag.med_sparse_2d_px > self.params.lost_rms_px
        {
            self.surf_pts.clear();
        }

        // ---- bootstrap: no temporal prior yet → seed the root from the torso
        // hint and fit the sparse terms with a wide (near-L2) kernel first, so
        // the dense cloud association starts from a sane pose ---------------------
        if self.last_t.is_none() {
            if let Some(hint) = obs.torso_hint {
                // Pelvis ≈ 0.45 m below the shoulder line for a subject facing
                // the camera (camera y is down).
                self.state.root_r = FACING_CAMERA;
                self.state.root_t = [hint[0], hint[1] + 0.45, hint[2] + 0.03];
                self.state.set_relaxed(model);
            }
            {
                let saved = (self.params.gnc_start, self.params.gnc_decay);
                self.params.gnc_start = 50.0;
                self.params.gnc_decay = 0.5;
                self.lm_loop(model, obs, &prior_var, dt);
                self.params.gnc_start = saved.0;
                self.params.gnc_decay = saved.1;
            }
        }
        // ---- LM loop --------------------------------------------------------
        // Two-stage solve (trunk → arms), `VULVATAR_FUSION_TWOSTAGE=1`:
        // stage 1 freezes the arm chain at the prediction and runs a
        // TRUNK-ONLY problem — observations anchored below the shoulder
        // balls and the wrist holds are excluded, limb capsules are
        // removed from surface association (a frozen arm at the real limb
        // position eats chest points as nearest-limb drops on one side
        // and the trunk fits the asymmetric remainder), so an arm in the
        // wrong basin cannot trade torso yaw / clavicle splay for its own
        // residuals. Stage 2 frees the arms with trunk + root + shape
        // frozen, so arm capsules may claim dense-surface points
        // (`VULVATAR_DENSE_ARMS`) without inventing the substitute that
        // blew |err| 19.0→32.7 in the single-stage bench. Analytic arm
        // seeds re-seed the STAGE-1 trunk and re-run stage 2 only.
        // (Intermediate measurements: keeping arm keypoints in the trunk
        // stage even under Cauchy → yaw sd 1–4° → 29–108°; core-only
        // association without the shoulder keypoints → yaw sd 11°.)
        static TWOSTAGE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        // Bootstrap / re-acquisition frames (no temporal prior yet) keep the
        // single-stage joint solve: a wrong torso hint from junk detections
        // is rejected there precisely by the arm observations the trunk
        // stage masks out (measured: dropout session s1789279985, the
        // re-entry seeded the root at a 2.9 m wall and every later frame
        // tracked it — 5 re-acquisitions and yaw err −16.5°).
        let twostage = self.last_t.is_some()
            && *TWOSTAGE.get_or_init(|| {
                std::env::var("VULVATAR_FUSION_TWOSTAGE")
                    .map(|v| v != "0")
                    .unwrap_or(false)
            });
        let mut cost;
        let start = self.state.clone();
        let t_phase = std::time::Instant::now();
        let mut stage1: Option<State> = None;
        if twostage {
            self.run_locked(model, obs, &prior_var, dt, LockSet::TrunkStage);
            stage1 = Some(self.state.clone());
            // Diagnostics: `VULVATAR_FUSION_TS_STAGE1_ONLY=1` publishes the
            // trunk-stage state (arms still at the prediction, seeds off).
            let stage1_only =
                std::env::var_os("VULVATAR_FUSION_TS_STAGE1_ONLY").is_some();
            cost = if stage1_only {
                self.diag.cost_final
            } else {
                self.run_locked(model, obs, &prior_var, dt, LockSet::ArmStage)
            };
        } else {
            cost = self.lm_loop(model, obs, &prior_var, dt);
        }
        self.timings.main_ms = t_phase.elapsed().as_secs_f64() * 1000.0;
        // Seed contest is judged WITHOUT the dense-surface cost: an analytic
        // arm seed is derived from the metric keypoint lifts, and letting the
        // surface term vote on its acceptance re-derives the association the
        // seed itself changed — with arm capsules claiming points, a candidate
        // that merely re-poses the arm to explain its own pixels won on
        // association gain, not landmark evidence (wave replay: seed wins
        // 3 → 15, wrists jumping 0.5–0.8 m). The surface term stays in the
        // SOLVE for every candidate; only the accept test excludes it.
        let mut cmp_cost = cost - self.diag.cost_cloud;
        let mut best_state = self.state.clone();
        let mut best_diag = self.diag;
        let mut best_gnc_final = true;
        let t_seeds = std::time::Instant::now();
        // Seeds grow from the prediction in single-stage mode, and from the
        // stage-1 (trunk-solved, arms-still-predicted) state in two-stage
        // mode — the analytic seed only re-poses the arm, so the trunk it
        // hangs from should be the solved one.
        let seed_base = stage1.as_ref().unwrap_or(&start);
        let skip_seeds = stage1.is_some()
            && std::env::var_os("VULVATAR_FUSION_TS_STAGE1_ONLY").is_some();
        for seed in seeds {
            if skip_seeds {
                break;
            }
            let Some(cand) = seed(seed_base) else { continue };
            self.state = cand;
            let c = if twostage {
                self.run_locked(model, obs, &prior_var, dt, LockSet::ArmStage)
            } else {
                self.lm_loop(model, obs, &prior_var, dt)
            };
            let c_cmp = c - self.diag.cost_cloud;
            if std::env::var_os("VULVATAR_SEED_DUMP").is_some() {
                eprintln!(
                    "SEEDCAND cost {cost:.1} cand {c:.1} ratio {:.3} (surf-excl {cmp_cost:.1} vs {c_cmp:.1} ratio {:.3})",
                    c / cost.max(1e-6),
                    c_cmp / cmp_cost.max(1e-6)
                );
            }
            if c_cmp < cmp_cost * self.params.seed_win_ratio {
                cmp_cost = c_cmp;
                cost = c;
                best_state = self.state.clone();
                best_diag = self.diag;
                best_gnc_final = false;
                self.diag.seed_wins += 1;
            }
        }
        self.timings.seeds_ms = t_seeds.elapsed().as_secs_f64() * 1000.0;
        let t_reacc = std::time::Instant::now();
        if !best_gnc_final || !seeds.is_empty() {
            // Re-accumulate at the winner so H (covariance) matches it.
            self.state = best_state;
            let fk = model.fk(&self.state);
            self.gnc = 1.0;
            let seed_wins = self.diag.seed_wins;
            self.diag = best_diag;
            self.diag.seed_wins = seed_wins;
            cost = self.accumulate(model, obs, &fk, &prior_var, dt, true);
        }
        self.timings.reacc_ms = t_reacc.elapsed().as_secs_f64() * 1000.0;
        self.diag.cost_final = cost;
        let t_finish = std::time::Instant::now();
        self.finish(model, &prev, dt, obs.t);
        self.timings.finish_ms = t_finish.elapsed().as_secs_f64() * 1000.0;
        // ---- track health: a fit whose sparse residuals stay far off is a
        // lost track (association collapse); re-acquire next frame ---------------
        // Track health is determined by sparse body/face sites. Finger residuals
        // from fast hand crops must not trigger a whole-body track collapse.
        // Furthermore, if metric 3D anchors (nose/shoulders) are closely tracking,
        // the body is undeniably in-track.
        let sparse_unhealthy = if self.diag.med_sparse_2d_px > 0.0 {
            self.diag.med_sparse_2d_px > self.params.lost_rms_px
        } else {
            self.diag.med_2d_px > self.params.lost_rms_px
        };
        let has_solid_3d = self.diag.n_kp3d >= 4 && self.diag.mean_3d_m < 0.12;
        // The dense surface is the primary observation: a trunk that sits
        // on hundreds of measured surface points is in track even when the
        // 2-D face points disagree (palms over the face: the detector
        // paints nose / eyes on the hands, their residual explodes, and
        // without this the track was reset onto the hands).
        let has_solid_surface = self.diag.n_cloud >= 150 && self.diag.mean_cloud_m.abs() < 0.03;
        let unhealthy = self.diag.n_kp2d >= 8
            && !has_solid_3d
            && !has_solid_surface
            && (sparse_unhealthy
                // Collapsed state: plenty of detections, almost nothing
                // projects — near-zero cost that must not read as healthy.
                || self.diag.n_2d_proj * 4 < self.diag.n_kp2d);
        if unhealthy {
            self.lost_frames += 1;
            if self.lost_frames >= 2 {
                self.mark_lost(model);
            }
        } else {
            self.lost_frames = 0;
        }
    }

    /// Zero the rows / columns of locked parameters in the normal equations
    /// (identity diagonal, zero gradient) so their increment is exactly 0.
    fn apply_locks(&mut self) {
        let n = self.dense.n;
        for k in 0..n {
            if !self.locked_param[k] {
                continue;
            }
            for j in 0..n {
                self.dense.h[k * n + j] = 0.0;
                self.dense.h[j * n + k] = 0.0;
            }
            self.dense.h[k * n + k] = 1.0;
            self.dense.g[k] = 0.0;
        }
    }

    /// Run one stage of the two-stage solve: `lm_loop` with a temporary
    /// parameter lock set applied on top of the standing locks (pelvis).
    /// See `update_with_seeds` for the rationale.
    fn run_locked(
        &mut self,
        model: &Model,
        obs: &FrameObs,
        prior_var: &[f64],
        dt: f64,
        locks: LockSet,
    ) -> f64 {
        let saved = self.locked_param.clone();
        self.arm_obs_masked = matches!(locks, LockSet::TrunkStage);
        for (k, l) in self.locked_param.iter_mut().enumerate() {
            let extra = match locks {
                // Trunk stage: the whole arm chain (incl. fingers and
                // pronation) stays at the warm start, and arm-anchored
                // observations / wrist holds are excluded from the build
                // (`arm_obs_masked`).
                LockSet::TrunkStage => self.arm_param[k],
                // Arm stage: root, trunk joints and shape stay at the
                // stage-1 solve; only limbs below the clavicles move.
                LockSet::ArmStage => k < 6 || self.trunk_param[k] || k >= model.beta_scale,
            };
            *l = saved[k] || extra;
        }
        let cost = self.lm_loop(model, obs, prior_var, dt);
        self.locked_param = saved;
        self.arm_obs_masked = false;
        cost
    }

    /// Does the observation resolving to `joint` land inside the arm chain
    /// below the shoulder balls (trunk-stage observation policy)? The
    /// shoulder keypoints themselves stay — they carry clavicle / spine
    /// placement, which the trunk stage owns.
    fn arm_anchored(&self, model: &Model, joint: usize) -> bool {
        self.arm_below_shoulder
            .get(model.joint_param.get(joint).copied().unwrap_or(0))
            .copied()
            .unwrap_or(false)
    }

    /// Drop the temporal state AND the pose itself so the next frame
    /// bootstraps from a sane state — a collapsed pose (body at/behind the
    /// camera plane) yields no observations at all, so re-optimising from
    /// it can never recover without this hard reset.
    pub fn mark_lost(&mut self, model: &Model) {
        let n = model.num_params;
        let prev_z = self.state.root_t[2].clamp(0.4, 2.5);
        self.state = State::rest(model);
        self.state.set_relaxed(model);
        self.state.root_r = FACING_CAMERA;
        self.state.root_t = [0.0, 0.3, prev_z];
        self.pred = self.state.clone();
        self.var = vec![1.0; n];
        self.vel = vec![0.0; n];
        self.last_t = None;
        self.lost_frames = 0;
        self.lost_events += 1;
    }

    /// Run the damped Gauss–Newton loop on the current state and leave the
    /// normal equations accumulated at the final state (GNC scale 1).
    fn lm_loop(&mut self, model: &Model, obs: &FrameObs, prior_var: &[f64], dt: f64) -> f64 {
        let mut lambda = 1e-3;
        // Graduated non-convexity only when there is no temporal prior to
        // warm-start from (bootstrap / re-acquisition); a tracked frame
        // starts close to the answer, and a wide kernel would let hallucinated
        // keypoints pull the model before the anneal rejects them.
        self.gnc = if self.last_t.is_none() {
            self.params.gnc_start.max(1.0)
        } else {
            self.params.gnc_tracked.max(1.0)
        };
        let mut fk = model.fk(&self.state);
        let t_acc = std::time::Instant::now();
        let mut cost = self.accumulate(model, obs, &fk, prior_var, dt, true);
        self.timings.acc_ms += t_acc.elapsed().as_secs_f64() * 1000.0;
        self.diag = SolveDiag {
            iters: 0,
            cost_initial: cost,
            cost_final: cost,
            n_kp2d: obs.kp2d.len(),
            n_kp3d: obs.kp3d.len(),
            n_cloud: 0,
            dt,
            ..self.diag
        };
        for it in 0..self.params.max_iters {
            let mut solved = false;
            for _attempt in 0..6 {
                let t_lin = std::time::Instant::now();
                self.apply_locks();
                let solved_lin = self
                    .dense
                    .solve_damped(lambda, 1e-9, &mut self.delta)
                    .is_some();
                self.timings.lin_ms += t_lin.elapsed().as_secs_f64() * 1000.0;
                if !solved_lin {
                    lambda *= 10.0;
                    continue;
                }
                let mut trial = self.state.clone();
                trial.apply_delta(model, &self.delta);
                let fk_trial = model.fk(&trial);
                let t_eval = std::time::Instant::now();
                let cost_trial = self.eval_cost(model, obs, &fk_trial, &trial, prior_var, dt);
                self.timings.eval_ms += t_eval.elapsed().as_secs_f64() * 1000.0;
                if cost_trial.is_finite() && cost_trial <= cost {
                    self.state = trial;
                    fk = fk_trial;
                    let improvement = (cost - cost_trial) / cost.max(1e-12);
                    cost = cost_trial;
                    lambda = (lambda * 0.3).max(1e-6);
                    solved = true;
                    self.diag.iters = it + 1;
                    if improvement < 1e-4 && self.gnc <= 1.0 {
                        // converged
                        let t_acc = std::time::Instant::now();
                        let c = self.accumulate(model, obs, &fk, prior_var, dt, true);
                        self.timings.acc_ms += t_acc.elapsed().as_secs_f64() * 1000.0;
                        return c;
                    }
                    break;
                } else {
                    lambda *= 8.0;
                }
            }
            if !solved {
                break;
            }
            // Anneal the cloud kernel; the cost is re-evaluated under the new
            // scale so the acceptance test stays consistent.
            self.gnc = (self.gnc * self.params.gnc_decay).max(1.0);
            let t_acc = std::time::Instant::now();
            cost = self.accumulate(model, obs, &fk, prior_var, dt, true);
            self.timings.acc_ms += t_acc.elapsed().as_secs_f64() * 1000.0;
        }
        let _ = cost;
        self.gnc = 1.0;
        let t_acc = std::time::Instant::now();
        let c = self.accumulate(model, obs, &fk, prior_var, dt, true);
        self.timings.acc_ms += t_acc.elapsed().as_secs_f64() * 1000.0;
        c
    }

    /// Posterior bookkeeping after the LM loop: covariance from the final
    /// normal matrix, velocity update, timestamps, shape freeze.
    fn finish(&mut self, model: &Model, prev: &State, dt: f64, t: f64) {
        let n = model.num_params;
        // Observed-ness: leaky integration of the measurement information.
        let decay = (-dt / 0.3).exp();
        if self.data_info_ema.len() != n {
            self.data_info_ema = vec![0.0; n];
        }
        for k in 0..n {
            let di = self.data_info.get(k).copied().unwrap_or(0.0);
            self.data_info_ema[k] = self.data_info_ema[k] * decay + di * (1.0 - decay);
        }
        // Marginals from the last accumulated H (state at convergence). The
        // normal matrix spans ~1e-2 (prior-only params) to ~1e7 (face
        // landmarks); retry with a scaled ridge if the factorisation hits a
        // non-positive pivot.
        let mut var = vec![0.0; n];
        let mut ok = false;
        for eps in [1e-9, 1e-6, 1e-3] {
            if self
                .dense
                .full_inverse(eps, &mut var, &mut self.cov)
                .is_some()
            {
                ok = true;
                break;
            }
        }
        if ok {
            for k in 0..n {
                self.var[k] = var[k].clamp(self.params.var_min, self.params.var_max);
            }
            self.update_world_sigmas(model);
        } else {
            self.diag.cov_failures += 1;
        }
        // Velocity: parameter-space difference to the previous posterior.
        // A frame with ZERO data terms carries no evidence of motion, and
        // the difference below would otherwise re-ingest the prediction's
        // own extrapolation (≈0.96× the old velocity per frame — the
        // decay inside `predict` is almost exactly cancelled), letting one
        // junk-detection spike keep the root drifting 0.1–0.2 m per frame
        // for the whole dropout (measured s1789279985: root_z 0.93 → 1.32 m
        // over 12 unobserved frames, 172 of the session's 203 wrist snaps).
        // Zero it: prediction = last state, the pose holds until data
        // returns (or `mark_lost` resets on junk).
        let no_data =
            self.diag.n_kp2d == 0 && self.diag.n_kp3d == 0 && self.diag.n_cloud == 0;
        if no_data {
            for v in self.vel.iter_mut() {
                *v = 0.0;
            }
        } else {
            let d = param_difference(model, &self.state, prev);
            let alpha = 0.6;
            for k in 0..n {
                let v = d[k] / dt;
                self.vel[k] = if self.last_t.is_some() {
                    alpha * v + (1.0 - alpha) * self.vel[k]
                } else {
                    0.0
                };
            }
        }
        // Clamp velocities to sane magnitudes.
        for k in 0..n {
            let cap = if k < 3 {
                6.0
            } else if k < 6 {
                3.0
            } else if k < model.beta_scale {
                12.0
            } else {
                0.0
            };
            self.vel[k] = self.vel[k].clamp(-cap, cap);
        }
        self.last_t = Some(t);
        self.frames += 1;
        // Freeze the shape once its variances have converged (a few seconds
        // of well-observed data).
        if std::env::var_os("VULVATAR_DENSE_FREEZE").is_some() && self.frames > 30 {
            self.shape_frozen = true;
        }
        if !self.shape_frozen && self.frames > 90 {
            let max_shape_var = (model.beta_scale..n)
                .map(|k| self.var[k])
                .fold(0.0, f64::max);
            if max_shape_var < 2e-4 {
                self.shape_frozen = true;
            }
        }
    }

    /// Accumulate normal equations (if `build`) and return the total robust
    /// cost at `fk`/`self.state`.
    fn accumulate(
        &mut self,
        model: &Model,
        obs: &FrameObs,
        fk: &Fk,
        prior_var: &[f64],
        dt: f64,
        build: bool,
    ) -> f64 {
        let st = self.state.clone();
        let surf = std::mem::take(&mut self.surf_pts);
        let c = self.accumulate_at(model, obs, fk, &st, &surf, prior_var, dt, build);
        self.surf_pts = surf;
        c
    }

    fn eval_cost(
        &mut self,
        model: &Model,
        obs: &FrameObs,
        fk: &Fk,
        st: &State,
        prior_var: &[f64],
        dt: f64,
    ) -> f64 {
        let surf = std::mem::take(&mut self.surf_pts);
        let c = self.accumulate_at(model, obs, fk, st, &surf, prior_var, dt, false);
        self.surf_pts = surf;
        c
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_at(
        &mut self,
        model: &Model,
        obs: &FrameObs,
        fk: &Fk,
        st: &State,
        surf: &[([f64; 3], f64)],
        prior_var: &[f64],
        _dt: f64,
        build: bool,
    ) -> f64 {
        let n = model.num_params;
        if build {
            self.dense.resize(n);
            self.dense.clear();
        }
        let mut cost = 0.0;
        let p = &self.params;
        let mut c2d = 0.0;
        let mut c3d = 0.0;
        let mut ccl = 0.0;
        let mut cwh = 0.0;
        let cpr;
        let ctm;
        let mut sum2d = 0.0;
        let mut n2d = 0usize;
        let mut res2d: Vec<f64> = Vec::with_capacity(obs.kp2d.len());
        let mut res2d_sparse: Vec<f64> = Vec::with_capacity(obs.kp2d.len());
        let mut sum3d = 0.0;
        let mut n3d = 0usize;
        let mut sumcl = 0.0;
        let mut ncl = 0usize;

        // ---- 2-D reprojection ------------------------------------------------
        if let Some(intr) = obs.intr {
            let c2 = p.c_2d * self.gnc;
            let kern = if p.cauchy_2d {
                Kernel::Cauchy(c2)
            } else {
                Kernel::GemanMcClure(c2)
            };
            // Default OFF (0): opt-in until the full bench proves it.
            let arm_mask =
                std::env::var("VULVATAR_ARM_CHAIN_DEPTH").map_or(false, |v| v != "0");
            for kp in &obs.kp2d {
                let (joint, pw) = resolve_point(model, fk, kp.point);
                // Trunk stage of the two-stage solve: keypoint observations
                // anchored in the (frozen) arm chain are dropped entirely —
                // a lagging predicted arm would otherwise pull the trunk
                // through its chain Jacobians even under Cauchy (measured:
                // 12-recording bench, torso yaw sd 1–4° → 29–108° with them
                // included). Stage 2 re-admits them with the arms free.
                if self.arm_obs_masked && self.arm_anchored(model, joint) {
                    continue;
                }
                let Some(uv) = intr.project(pw) else { continue };
                let inv_s = 1.0 / kp.sigma.max(0.25);
                let ru = (uv[0] - kp.u) * inv_s;
                let rv = (uv[1] - kp.v) * inv_s;
                let s = ru * ru + rv * rv;
                let (rho, w) = kern.eval(s);
                cost += rho;
                c2d += rho;
                let err_px = s.sqrt() * kp.sigma.max(0.25);
                sum2d += err_px;
                res2d.push(err_px);
                if matches!(kp.point, ModelPoint::Joint(_) | ModelPoint::Site(_)) {
                    res2d_sparse.push(err_px);
                }
                n2d += 1;
                if !build {
                    continue;
                }
                point_jac(model, st, fk, kp.point, joint, &mut self.jac);
                // Arm-chain masking (VULVATAR_ARM_CHAIN_DEPTH=1): an
                // observation resolving inside an arm chain may only move
                // that arm's parameters — hand/wrist evidence poses the
                // arm, the trunk (clavicle/spine/root) stays owned by the
                // torso observations. Measured motivation: every hand-side
                // σ adjustment flipped the profile-session torso basin.
                let mask_arm = arm_mask
                    && self
                        .arm_param
                        .get(model.joint_param.get(joint).copied().unwrap_or(0))
                        .copied()
                        .unwrap_or(false);
                if mask_arm {
                    self.jac.retain(|(i, _)| self.arm_param.get(*i).copied().unwrap_or(false));
                }
                // ∂u/∂p = fx/z (1, 0, -x/z) ; ∂v/∂p = fy/z (0, 1, -y/z)
                let iz = 1.0 / pw[2];
                let du = [intr.fx * iz, 0.0, -intr.fx * pw[0] * iz * iz];
                let dv = [0.0, intr.fy * iz, -intr.fy * pw[1] * iz * iz];
                self.row_out.clear();
                self.row_out2.clear();
                for &(i, v) in &self.jac {
                    self.row_out.push((i, dot(du, v) * inv_s));
                    self.row_out2.push((i, dot(dv, v) * inv_s));
                }
                self.dense.add_residual(&self.row_out, ru, w);
                self.dense.add_residual(&self.row_out2, rv, w);
            }
        }

        // ---- planar joint angles (finger curl) -------------------------------
        // Interior angle at the vertex of a projected model triple vs the
        // landmark triple's angle. Invariant to scale / translation /
        // wrist anchor — the property that makes finger curl (2-D
        // degenerate for point reprojection) observable. OFF by default:
        // on the desk replays the local finger churn it targets is
        // already small (mean 1.6°/frame, p95 8–9°) and the first
        // implementation neither won nor lost (mean 1.62 vs 1.66, p95
        // 9.1 vs 8.3 — σ uncalibrated, thumb geometry approximate).
        // Enable with VULVATAR_FUSION_ANG=1 while iterating.
        if let Some(intr) = obs.intr {
            if std::env::var_os("VULVATAR_FUSION_ANG").is_some() {
                let ca = p.c_2d * self.gnc;
                let kern = Kernel::Cauchy(ca);
                for ob in &obs.angles {
                    // Trunk stage: finger triples are arm-anchored.
                    let masked = self.arm_obs_masked && {
                        let (jv, _) = resolve_point(model, fk, ob.vertex);
                        self.arm_anchored(model, jv)
                    };
                    let Some(rho) = (|| {
                        let (ja, pa) = resolve_point(model, fk, ob.a);
                        let (jv, pv) = resolve_point(model, fk, ob.vertex);
                        if masked {
                            return None;
                        }
                        let (jb, pb) = resolve_point(model, fk, ob.b);
                        let ua = intr.project(pa)?;
                        let uv = intr.project(pv)?;
                        let ub = intr.project(pb)?;
                        let u = [ua[0] - uv[0], ua[1] - uv[1]];
                        let v = [ub[0] - uv[0], ub[1] - uv[1]];
                        let c = u[0] * v[1] - u[1] * v[0];
                        let d = u[0] * v[0] + u[1] * v[1];
                        let theta = c.atan2(d);
                        let inv_s = 1.0 / ob.sigma.max(0.02);
                        let r = (theta - ob.angle) * inv_s;
                        let s = r * r;
                        let (rho, w) = kern.eval(s);
                        if !build {
                            return Some(rho);
                        }
                        // dθ/d(each projected coordinate) = (d·dc − c·dd)/(c²+d²).
                        let den = c * c + d * d;
                        if den < 1e-9 {
                            return Some(rho);
                        }
                        let grad = |dc: [f64; 2], dd: [f64; 2]| -> [f64; 2] {
                            [
                                (d * dc[0] - c * dd[0]) / den,
                                (d * dc[1] - c * dd[1]) / den,
                            ]
                        };
                        let ga = grad([v[1], -v[0]], [v[0], v[1]]);
                        let gb = grad([-u[1], u[0]], [u[0], u[1]]);
                        let gv = grad(
                            [u[1] - v[1], v[0] - u[0]],
                            [-v[0] - u[0], -v[1] - u[1]],
                        );
                        self.row_out.clear();
                        for (mp, pw, g, joint) in [
                            (ob.a, pa, ga, ja),
                            (ob.b, pb, gb, jb),
                            (ob.vertex, pv, gv, jv),
                        ] {
                            let [gx, gy] = g;
                            if gx == 0.0 && gy == 0.0 {
                                continue;
                            }
                            let iz = 1.0 / pw[2];
                            let du = [intr.fx * iz, 0.0, -intr.fx * pw[0] * iz * iz];
                            let dv = [0.0, intr.fy * iz, -intr.fy * pw[1] * iz * iz];
                            point_jac(model, st, fk, mp, joint, &mut self.jac);
                            for &(i, vv) in &self.jac {
                                self.row_out.push((
                                    i,
                                    (dot(du, vv) * gx + dot(dv, vv) * gy) * inv_s,
                                ));
                            }
                        }
                        // Merge duplicates (three points share ancestor params).
                        self.row_out.sort_by(|x, y| x.0.cmp(&y.0));
                        let mut merged: Vec<(usize, f64)> = Vec::with_capacity(self.row_out.len());
                        for &(i, val) in &self.row_out {
                            match merged.last_mut() {
                                Some((li, lv)) if *li == i => *lv += val,
                                _ => merged.push((i, val)),
                            }
                        }
                        self.dense.add_residual(&merged, r, w);
                        Some(rho)
                    })() else {
                        continue;
                    };
                    cost += rho;
                    c2d += rho;
                }
            }
        }

        // ---- 3-D points -----------------------------------------------------
        {
            let c3 = p.c_3d * self.gnc;
            let kern = Kernel::Cauchy(c3);
            let arm_mask =
                std::env::var("VULVATAR_ARM_CHAIN_DEPTH").map_or(false, |v| v != "0");
            for kp in &obs.kp3d {
                let (joint, pw) = resolve_point(model, fk, kp.point);
                // Trunk stage: see the kp2d loop.
                if self.arm_obs_masked && self.arm_anchored(model, joint) {
                    continue;
                }
                let d = sub(pw, kp.p);
                let inv_s = 1.0 / kp.sigma.max(1e-4);
                let inv_lat = inv_s / kp.lat_scale.max(1.0);
                let r = [d[0] * inv_lat, d[1] * inv_lat, d[2] * inv_s];
                let s = dot(r, r);
                let (rho, w) = kern.eval(s);
                cost += rho;
                c3d += rho;
                sum3d += s.sqrt() * kp.sigma.max(1e-4);
                n3d += 1;
                if !build {
                    continue;
                }
                point_jac(model, st, fk, kp.point, joint, &mut self.jac);
                if arm_mask
                    && self.arm_param.get(model.joint_param.get(joint).copied().unwrap_or(0))
                        .copied()
                        .unwrap_or(false)
                {
                    self.jac.retain(|(i, _)| self.arm_param.get(*i).copied().unwrap_or(false));
                }
                self.jac2.clear();
                for &(i, v) in &self.jac {
                    self.jac2
                        .push((i, [v[0] * inv_lat, v[1] * inv_lat, v[2] * inv_s]));
                }
                self.dense.add_residual3(&self.jac2, r, w);
            }
        }

        // ---- capsule-end Jacobians (surface term) -------------------------------
        if build && !surf.is_empty() {
            let nc = model.capsules.len();
            if self.cap_jac.len() != 3 * nc {
                self.cap_jac = vec![Vec::with_capacity(48); 3 * nc];
            }
            for (ci, c) in model.capsules.iter().enumerate() {
                let mut ja = std::mem::take(&mut self.cap_jac[3 * ci]);
                model.point_jacobian(st, fk, c.a, &mut ja);
                self.cap_jac[3 * ci] = ja;
                let mut jb = std::mem::take(&mut self.cap_jac[3 * ci + 1]);
                model.point_jacobian(st, fk, c.b, &mut jb);
                self.cap_jac[3 * ci + 1] = jb;
                let mut jc = std::mem::take(&mut self.cap_jac[3 * ci + 2]);
                jc.clear();
                if let Some(l) = c.lateral {
                    model.point_jacobian(st, fk, l, &mut jc);
                }
                self.cap_jac[3 * ci + 2] = jc;
            }
        }
        // ---- sparse surface points (depth under body keypoints) -----------------
        {
            let (c, n, md) = self.surface_term(model, fk, st, surf, build);
            cost += c;
            ccl += c;
            sumcl += md * n as f64;
            ncl += n;
        }

        // Snapshot the data-only information (diagonal of H before any
        // prior) — the "how much did measurements say about this parameter
        // this frame" signal used for observed-vs-prior confidence.
        if build {
            let n = self.dense.n;
            if self.data_info.len() != n {
                self.data_info = vec![0.0; n];
            }
            for k in 0..n {
                self.data_info[k] = self.dense.h[k * n + k];
            }
        }
        // ---- joint limits + pose prior ----------------------------------------
        let p = &self.params;
        let cost_before_prior = cost;
        {
            let inv_lim = 1.0 / p.limit_sigma;
            for (j, jd) in model.joints.iter().enumerate() {
                let pidx = model.joint_param[j];
                match jd.kind {
                    JointKind::Hinge { lo, hi, .. } => {
                        let a = st.angle[j];
                        let viol = if a > hi {
                            a - hi
                        } else if a < lo {
                            a - lo
                        } else {
                            0.0
                        };
                        if viol != 0.0 {
                            let r = viol * inv_lim;
                            cost += r * r;
                            if build {
                                self.dense.add_residual(&[(pidx, inv_lim)], r, 1.0);
                            }
                        }
                        // prior — the model's relaxed pose. NOTE: since the
                        // pose-calibration neutral went out (058bfba) this
                        // mean is the model's hang-down pose, ~1+ rad from
                        // a desk user's reaching arms; benched on
                        // s1789311387, the prior (with the trunk terms)
                        // holds a reaching left elbow ~0.2 m behind its
                        // observation (L-wrist snaps 6, freed at
                        // PRIOR_SCALE 0.05 — but global relaxation wrecks
                        // trunk yaw, and shoulder-/elbow-only relaxation
                        // frees neither). The per-user neutral from the
                        // calibration rebuild is the real fix.
                        let sp = (jd.prior_sigma[0] * p.pose_prior_scale).max(1e-4);
                        let mean0 = jd.prior_mean[0];
                        let r = (a - mean0) / sp;
                        cost += r * r;
                        if build {
                            self.dense.add_residual(&[(pidx, 1.0 / sp)], r, 1.0);
                        }
                    }
                    JointKind::Ball { lo, hi } => {
                        let w = so3_log(&st.rot[j]);
                        let jl = so3_left_jacobian_inv(w);
                        for k in 0..3 {
                            let viol = if w[k] > hi[k] {
                                w[k] - hi[k]
                            } else if w[k] < lo[k] {
                                w[k] - lo[k]
                            } else {
                                0.0
                            };
                            let mean_k = jd.prior_mean[k];
                            let sp = (jd.prior_sigma[k] * p.pose_prior_scale).max(1e-4);
                            let rp = (w[k] - mean_k) / sp;
                            cost += rp * rp;
                            let rl = viol * inv_lim;
                            cost += rl * rl;
                            if build {
                                // ∂w_k/∂δ = row k of J_l⁻¹
                                let row: Vec<(usize, f64)> =
                                    (0..3).map(|c| (pidx + c, jl[k][c] / sp)).collect();
                                self.dense.add_residual(&row, rp, 1.0);
                                if viol != 0.0 {
                                    let row: Vec<(usize, f64)> =
                                        (0..3).map(|c| (pidx + c, jl[k][c] * inv_lim)).collect();
                                    self.dense.add_residual(&row, rl, 1.0);
                                }
                            }
                        }
                    }
                }
            }
        }

        // ---- torso yaw from the chest depth slope -----------------------------------
        if let Some(o) = &obs.shoulder_yaw {
            if o.left < model.joints.len() && o.right < model.joints.len() {
                let v = sub(fk.t[o.left], fk.t[o.right]);
                let d2 = v[0] * v[0] + v[2] * v[2];
                if d2 > 1e-6 {
                    let mut e = v[2].atan2(v[0]) - o.yaw;
                    while e > std::f64::consts::PI {
                        e -= 2.0 * std::f64::consts::PI;
                    }
                    while e < -std::f64::consts::PI {
                        e += 2.0 * std::f64::consts::PI;
                    }
                    let inv = 1.0 / o.sigma.max(1e-3);
                    let r = e * inv;
                    // Deliberately NOT robustified (despite the module
                    // contract): this obs is an anchor, not an
                    // outlier-prone measurement. Benched on s1789303569
                    // (2026-09-13): a Cauchy tail (c=3) let the trunk
                    // hover between basins — yaw err std 4.7 vs 4.0, and
                    // with the tail alone (no arm σ cap) 22.6 with -53°
                    // excursions, i.e. a weakened-but-abandoning anchor is
                    // worse than none. The committed L2 pull recovers from
                    // deep disagreement; its fight with arm transients is
                    // instead bounded at the source by the arm kp3d σ cap
                    // (provider) and the default σ scale 1.5 below the obs
                    // creation site.
                    cost += r * r;
                    if build {
                        // dθ/dv = (−v_z, 0, v_x) / (v_x² + v_z²)
                        let dth = [-v[2] / d2, 0.0, v[0] / d2];
                        self.jac.clear();
                        point_jac(
                            model,
                            st,
                            fk,
                            ModelPoint::Joint(o.left),
                            o.left,
                            &mut self.jac,
                        );
                        let jl = std::mem::take(&mut self.jac);
                        self.jac.clear();
                        point_jac(
                            model,
                            st,
                            fk,
                            ModelPoint::Joint(o.right),
                            o.right,
                            &mut self.jac,
                        );
                        let jr = std::mem::take(&mut self.jac);
                        self.row_out.clear();
                        for &(i, g) in &jl {
                            self.row_out.push((i, dot(dth, g) * inv));
                        }
                        for &(i, g) in &jr {
                            // Shared ancestor params appear in both chains;
                            // the solver's row builder sums duplicates.
                            self.row_out.push((i, -dot(dth, g) * inv));
                        }
                        self.dense.add_residual(&self.row_out, r, 1.0);
                        self.jac = jl;
                        let _ = jr;
                    }
                }
            }
        }

        // ---- direct orientation observations ---------------------------------------
        for o in &obs.ori {
            if o.joint >= model.joints.len() {
                continue;
            }
            let e = so3_log(&mat_mul(&fk.r[o.joint], &transpose(&o.target)));
            let inv = 1.0 / o.sigma.max(1e-3);
            let jl_inv = so3_left_jacobian_inv(e);
            // World-axis columns of every ancestor rotation param (the same
            // chain walk the covariance propagation uses).
            let mut cols: Vec<(usize, V3)> = Vec::with_capacity(48);
            let mut k = o.joint;
            let mut depth_left = o.chain_depth;
            loop {
                if depth_left == 0 {
                    break;
                }
                depth_left -= 1;
                let jd = &model.joints[k];
                let pidx = model.joint_param[k];
                let pr = match jd.parent {
                    Some(pp) => fk.r[pp],
                    None => st.root_r,
                };
                match jd.kind {
                    JointKind::Ball { .. } => {
                        for c in 0..3 {
                            cols.push((pidx + c, col(&pr, c)));
                        }
                    }
                    JointKind::Hinge { axis, .. } => cols.push((pidx, mat_vec(&pr, axis))),
                }
                match jd.parent {
                    Some(pp) => k = pp,
                    None => break,
                }
            }
            if depth_left > 0 {
                for c in 0..3 {
                    let mut ax = [0.0; 3];
                    ax[c] = 1.0;
                    cols.push((ROOT_ROT + c, ax));
                }
            }
            for r_idx in 0..3 {
                let r = e[r_idx] * inv;
                cost += r * r;
                if build {
                    self.row_out.clear();
                    for &(i, a) in &cols {
                        // d e / dδ_i = J_l⁻¹(e) · a
                        let v = mat_vec(&jl_inv, a);
                        self.row_out.push((i, v[r_idx] * inv));
                    }
                    self.dense.add_residual(&self.row_out, r, 1.0);
                }
            }
        }

        // ---- elbow-low prior --------------------------------------------------------
        {
            let inv = 1.0 / self.params.elbow_low_sigma;
            for (sh, el) in [
                (self.idx_l_shoulder, self.idx_l_elbow),
                (self.idx_r_shoulder, self.idx_r_elbow),
            ] {
                // camera y is down: elbow above shoulder ⇔ el.y < sh.y
                let viol = fk.t[sh][1] - fk.t[el][1];
                if viol <= 0.0 {
                    continue;
                }
                let r = viol * inv;
                cost += r * r;
                if build {
                    // d(viol) = d(sh.y) − d(el.y)
                    self.row_idx.clear();
                    let mut jacsh = std::mem::take(&mut self.jac);
                    jacsh.clear();
                    point_y_jac(model, st, fk, sh, 1.0, &mut jacsh);
                    point_y_jac(model, st, fk, el, -1.0, &mut jacsh);
                    self.row_out.clear();
                    for &(i, v) in &jacsh {
                        if self.row_acc[i] == 0.0 {
                            self.row_idx.push(i);
                        }
                        self.row_acc[i] += v[1] * inv;
                    }
                    for &i in &self.row_idx {
                        let v = self.row_acc[i];
                        if v != 0.0 {
                            self.row_out.push((i, v));
                        }
                        self.row_acc[i] = 0.0;
                    }
                    self.dense.add_residual(&self.row_out, r, 1.0);
                    self.jac = jacsh;
                }
            }
        }

        // ---- upright-root prior ---------------------------------------------------
        {
            // Body up (+Y) through the root rotation should be camera up
            // (−y in the camera frame). Two residuals: the x and z
            // components of the rotated up vector (0 when upright).
            let up = mat_vec(&st.root_r, [0.0, 1.0, 0.0]);
            // A correctly-fitted subject has up_cam ≈ (0, −1, 0). If the
            // solve is upside down (up_cam.y > 0) the prior would be blind
            // through x/z alone; the +y case is handled by the same two
            // residuals growing as the state escapes the basin.
            let inv = 1.0 / self.params.upright_sigma;
            for (k, comp) in [(0usize, up[0]), (2usize, up[2])] {
                let r = comp * inv;
                cost += r * r;
                if build {
                    // d(up)/dδ = δ × up  →  row_k over ROOT_ROT params.
                    let mut row: Vec<(usize, f64)> = Vec::with_capacity(3);
                    for c in 0..3 {
                        let mut e = [0.0; 3];
                        e[c] = 1.0;
                        let d = cross(e, up);
                        row.push((ROOT_ROT + c, d[k] * inv));
                    }
                    self.dense.add_residual(&row, r, 1.0);
                }
            }
        }

        // ---- shape prior ---------------------------------------------------------
        {
            let sig = if self.shape_frozen {
                0.005
            } else {
                p.shape_sigma
            };
            let inv = 1.0 / sig;
            let inv_scale = 1.0
                / if self.shape_frozen {
                    0.005
                } else {
                    p.scale_sigma
                };
            let r = st.scale * inv_scale;
            cost += r * r;
            if build {
                self.dense
                    .add_residual(&[(model.beta_scale, inv_scale)], r, 1.0);
            }
            for g in 0..NUM_LEN_GROUPS {
                let r = st.len[g] * inv;
                cost += r * r;
                if build {
                    self.dense
                        .add_residual(&[(model.beta_len + g, inv)], r, 1.0);
                }
            }
            let inv_r = 1.0
                / if self.shape_frozen {
                    0.005
                } else {
                    p.radius_sigma
                };
            for g in 0..NUM_RAD_GROUPS {
                let r = st.rad[g] * inv_r;
                cost += r * r;
                if build {
                    self.dense
                        .add_residual(&[(model.beta_rad + g, inv_r)], r, 1.0);
                }
            }
            // Freezing tightens toward 0.005 — but never LOOSENS a pin
            // (the disabled default 1e-9 must survive the freeze).
            let inv_sh = 1.0
                / if self.shape_frozen {
                    p.shear_sigma.min(0.005)
                } else {
                    p.shear_sigma
                };
            let r = st.shear * inv_sh;
            cost += r * r;
            if build {
                self.dense
                    .add_residual(&[(model.beta_shear, inv_sh)], r, 1.0);
            }
        }

        // ---- trunk axis prior (see `Params::trunk_axis_sigma`) --------------------
        if let (Some(sig), Some((lo, hi))) = (p.trunk_axis_sigma, self.trunk_sites) {
            let a = fk.site[lo];
            let b = fk.site[hi];
            let d = sub(b, a);
            let len = norm(d);
            if len > 1e-6 {
                let axis = scale(d, 1.0 / len);
                let r = axis[2] / sig;
                cost += r * r;
                if build {
                    // ∂axis_z/∂d = (e_z − axis·axis_z) / |d|
                    let gvec = scale(sub([0.0, 0.0, 1.0], scale(axis, axis[2])), 1.0 / len);
                    let mut jhi = Vec::with_capacity(48);
                    let mut jlo = Vec::with_capacity(48);
                    model.point_jacobian(st, fk, PointRef::Site(hi), &mut jhi);
                    model.point_jacobian(st, fk, PointRef::Site(lo), &mut jlo);
                    let mut jac1: Vec<(usize, f64)> = Vec::with_capacity(96);
                    for &(i, v) in &jhi {
                        jac1.push((i, dot(v, gvec) / sig));
                    }
                    for &(i, v) in &jlo {
                        jac1.push((i, -dot(v, gvec) / sig));
                    }
                    self.dense.add_residual(&jac1, r, 1.0);
                }
            }
        }
        cpr = cost - cost_before_prior;
        // ---- temporal prior (previous posterior propagated) --------------------
        let cost_before_temporal = cost;
        if self.last_t.is_some() {
            let d = param_difference(model, st, &self.pred);
            for k in 0..n {
                let inv = 1.0 / prior_var[k].sqrt();
                let r = d[k] * inv;
                cost += r * r;
                if build {
                    self.dense.add_residual(&[(k, inv)], r, 1.0);
                }
            }
        }
        ctm = cost - cost_before_temporal;
        // ---- wrist hold (unobserved end-effector) ----------------------------
        // Plain quadratic pseudo-observation of the wrist's root-frame
        // position toward the prediction — no robust kernel: the hold must
        // keep pulling exactly when the deviation is large. Gated per frame
        // in `update_with_seeds` (active only with zero wrist observations).
        let sig_wh = p.wrist_hold_sigma;
        if sig_wh > 1e-6 && !self.arm_obs_masked {
            // Trunk stage: the hold pins an unobserved wrist's root-frame
            // position; with the arm joints frozen that non-robust pull
            // would land on the spine / clavicle instead.
            let inv_s = 1.0 / sig_wh;
            for k in 0..2 {
                let (active, tgt) = self.hold_targets[k];
                if !active {
                    continue;
                }
                let wj = self.wrist_joints[k].0;
                let rel = sub(fk.t[wj], st.root_t);
                let r = [
                    (rel[0] - tgt[0]) * inv_s,
                    (rel[1] - tgt[1]) * inv_s,
                    (rel[2] - tgt[2]) * inv_s,
                ];
                let s = dot(r, r);
                cost += s;
                cwh += s;
                if build {
                    point_jac(model, st, fk, ModelPoint::Joint(wj), wj, &mut self.jac);
                    self.jac2.clear();
                    // ∂(wrist − root_t)/∂θ: the root-translation columns
                    // translate the wrist AND the anchor — subtract the
                    // identity from those three columns.
                    let mut root_seen = [false; 3];
                    for &(i, v) in &self.jac {
                        let mut v = v;
                        if i >= ROOT_T && i < ROOT_T + 3 {
                            let a = i - ROOT_T;
                            root_seen[a] = true;
                            v[a] -= 1.0;
                        }
                        self.jac2.push((i, [v[0] * inv_s, v[1] * inv_s, v[2] * inv_s]));
                    }
                    for a in 0..3 {
                        if !root_seen[a] {
                            let mut v = [0.0; 3];
                            v[a] = -inv_s;
                            self.jac2.push((ROOT_T + a, v));
                        }
                    }
                    self.dense.add_residual3(&self.jac2, r, 1.0);
                }
            }
        }
        if build {
            self.diag.cost_2d = c2d;
            self.diag.cost_3d = c3d;
            self.diag.cost_cloud = ccl;
            self.diag.cost_prior = cpr;
            self.diag.cost_temporal = ctm;
            self.diag.cost_whold = cwh;
            self.diag.n_2d_proj = n2d;
            self.diag.rms_2d_px = if n2d > 0 { sum2d / n2d as f64 } else { 0.0 };
            self.diag.med_2d_px = if n2d > 0 {
                res2d.sort_by(|a, b| a.partial_cmp(b).unwrap());
                res2d[res2d.len() / 2]
            } else {
                0.0
            };
            self.diag.med_sparse_2d_px = if !res2d_sparse.is_empty() {
                res2d_sparse.sort_by(|a, b| a.partial_cmp(b).unwrap());
                res2d_sparse[res2d_sparse.len() / 2]
            } else {
                self.diag.med_2d_px
            };
            self.diag.mean_3d_m = if n3d > 0 { sum3d / n3d as f64 } else { 0.0 };
            self.diag.mean_cloud_m = if ncl > 0 { sumcl / ncl as f64 } else { 0.0 };
            self.diag.n_cloud = ncl;
        }
        cost
    }
}

impl Estimator {
    /// Sparse point → capsule-surface term (lifted depth under keypoints).
    /// Returns `(cost, n_assoc, mean_d)`.
    fn surface_term(
        &mut self,
        model: &Model,
        fk: &Fk,
        st: &State,
        points: &[([f64; 3], f64)],
        build: bool,
    ) -> (f64, usize, f64) {
        let mut cost = 0.0;
        let mut sumcl = 0.0;
        let mut ncl = 0usize;
        if !points.is_empty() {
            let kern = Kernel::Cauchy(self.params.c_surf * self.gnc);
            let gate_m = self.params.surf_gate_m * self.gnc;
            // Association: nearest capsule surface within the gate. Hand
            // capsules are skipped: the hand landmarker owns the hands and
            // their pixels are excluded from the dense samples, so a hand
            // capsule can only steal torso / forearm points.
            let caps: Vec<CapGeom> = model
                .capsules
                .iter()
                .map(|c| CapGeom {
                    a: fk.point(c.a),
                    b: fk.point(c.b),
                    c: c.lateral.map(|l| fk.point(l)),
                    r: model.capsule_radius(st, c),
                    k: model.capsule_aspect(st, c),
                })
                .collect();
            let assoc_ok: Vec<bool> = model
                .capsules
                .iter()
                .map(|c| {
                    !matches!(
                        c.part,
                        super::model::Part::LeftHand | super::model::Part::RightHand
                    )
                })
                .collect();
            let allow = &self.surf_allow;
            let is_core: Vec<bool> = model
                .capsules
                .iter()
                .map(|c| matches!(c.part, super::model::Part::Torso | super::model::Part::Head))
                .collect();
            // Axis-aligned boxes (expanded by the gate) for cheap rejection.
            let boxes: Vec<([f64; 3], [f64; 3])> = caps
                .iter()
                .map(|g| {
                    let (a, b, r) = (g.a, g.b, g.r);
                    let e = r + gate_m;
                    let lo = [a[0].min(b[0]) - e, a[1].min(b[1]) - e, a[2].min(b[2]) - e];
                    let hi = [a[0].max(b[0]) + e, a[1].max(b[1]) + e, a[2].max(b[2]) + e];
                    (lo, hi)
                })
                .collect();
            // Trunk / head have priority: a point within `core_claim_m` of a
            // core capsule belongs to the core even if a limb capsule is
            // nearer. Limbs are free (often unobserved) and would otherwise
            // slide onto the chest and take its points, leaving the trunk
            // fitted to an asymmetric remainder (measured: 381 chest points
            // on the upper arm, trunk yaw 25° off the chest slope).
            let core_claim_m = 0.06;
            // (point, capsule, signed distance, axis point, axis param)
            let mut items: Vec<(usize, usize, f64, V3, f64)> = Vec::with_capacity(points.len());
            for (pi, (pt, _sig)) in points.iter().enumerate() {
                let mut best_core = (usize::MAX, f64::INFINITY, [0.0; 3], 0.0);
                let mut best_limb = (usize::MAX, f64::INFINITY, [0.0; 3], 0.0);
                let mut best_hand = f64::INFINITY;
                for (ci, g) in caps.iter().enumerate() {
                    // Trunk stage of the two-stage solve: limb capsules are
                    // excluded from association entirely — a frozen arm at
                    // the (solved, real) limb position eats chest points as
                    // nearest-limb drops on one side only, and the trunk
                    // fits the asymmetric remainder (measured: torso yaw
                    // climbs +4°/frame once a wave crosses the chest). The
                    // core front gate already rejects real-arm points
                    // (occluders) exactly as if the limbs were absent.
                    if self.arm_obs_masked && !is_core[ci] {
                        continue;
                    }
                    let (lo, hi) = boxes[ci];
                    if pt[0] < lo[0]
                        || pt[0] > hi[0]
                        || pt[1] < lo[1]
                        || pt[1] > hi[1]
                        || pt[2] < lo[2]
                        || pt[2] > hi[2]
                    {
                        continue;
                    }
                    let (d, q, u, _rho) = g.dist(*pt);
                    if !assoc_ok[ci] {
                        // Hand capsule: never fitted, but a point that is
                        // nearest to a hand belongs to the hand and must
                        // not be handed to the forearm.
                        best_hand = best_hand.min(d.abs());
                        continue;
                    }
                    let slot = if is_core[ci] {
                        &mut best_core
                    } else {
                        &mut best_limb
                    };
                    if d.abs() < slot.1.abs() {
                        *slot = (ci, d, q, u);
                    }
                }
                if best_hand < best_core.1.abs() && best_hand < best_limb.1.abs() {
                    continue;
                }
                // Depth order: the sensor only ever sees a surface from the
                // front, so a point well IN FRONT of a core capsule (d > 0)
                // is an occluder — a hand or forearm across the chest, the
                // desk edge — not the trunk surface a few centimetres out.
                // Behind / inside (d < 0, the model too fat or misplaced)
                // keeps the full gate so the fit can still converge.
                let front_m = self.params.surf_front_gate_m * self.gnc;
                let core_ok =
                    best_core.0 != usize::MAX && best_core.1 > -gate_m && best_core.1 < front_m;
                let limb_ok = best_limb.0 != usize::MAX && best_limb.1.abs() < gate_m;
                let limb_allowed = limb_ok
                    && (allow.is_empty() || allow.get(best_limb.0).copied().unwrap_or(true));
                // An OBSERVED limb competes fairly (nearest surface wins —
                // an arm held in front of the chest keeps its own points).
                // An unobserved limb never claims, but a point nearest to
                // it is dropped rather than handed to the trunk: the
                // predicted arm still occludes the chest there, and letting
                // a free arm explain chest points is how it slid onto the
                // chest at zero cost.
                let _ = core_claim_m;
                let pick = if limb_ok && best_limb.1.abs() < best_core.1.abs() {
                    if !limb_allowed {
                        continue;
                    }
                    best_limb
                } else if core_ok {
                    // A core capsule excluded this frame (e.g. the head while
                    // FaceMesh owns it) drops its points rather than passing
                    // them on to the neighbour.
                    if !allow.is_empty() && !allow.get(best_core.0).copied().unwrap_or(true) {
                        continue;
                    }
                    best_core
                } else {
                    continue;
                };
                items.push((pi, pick.0, pick.1, pick.2, pick.3));
            }
            let n_assoc = items.len();
            if build {
                self.last_surf_assoc.clear();
                self.last_surf_assoc.resize(points.len(), -1);
                for it in &items {
                    self.last_surf_assoc[it.0] = it.1 as i32;
                }
            }
            // Correlated-error normalisation per capsule (see `surf_n_eff`).
            let mut per_cap = vec![0usize; model.capsules.len()];
            for it in &items {
                per_cap[it.1] += 1;
            }
            // Trunk-facing obliquity: the shoulder-line direction (across
            // the torso capsules' axis midpoints) vs the camera x-axis.
            // sin² is sign-free; frontal view → 0. See
            // `Params::trunk_surf_obliq_k` for the measurement.
            let torso_obliq = {
                let k = self.params.trunk_surf_obliq_k;
                if k <= 0.0 {
                    1.0f64
                } else {
                    let mut min3 = [f64::INFINITY; 3];
                    let mut max3 = [f64::NEG_INFINITY; 3];
                    let mut any = false;
                    for (g, c) in caps.iter().zip(model.capsules.iter()) {
                        if !matches!(c.part, super::model::Part::Torso) {
                            continue;
                        }
                        for d in 0..3 {
                            let m = (g.a[d] + g.b[d]) * 0.5;
                            min3[d] = min3[d].min(m);
                            max3[d] = max3[d].max(m);
                        }
                        any = true;
                    }
                    if !any {
                        1.0
                    } else {
                        let (dx, dz) = (max3[0] - min3[0], max3[2] - min3[2]);
                        // sin⁴ of the shoulder-line tilt: a deadzone that
                        // keeps frontal view (|yaw| ≲ 20°) at (nearly) full
                        // weight — sin² alone let the frontal synthetic
                        // bench's trunk yaw wander into the down-weighting
                        // region and drift (streamer_da: yaw mean −1.3° →
                        // −15.9°), a feedback loop sin⁴ breaks at small
                        // angles (sin⁴ 10° = 0.0008 vs sin² 10° = 0.03).
                        let s2 = dz * dz / (dx * dx + dz * dz).max(1e-9);
                        1.0 / (1.0 + k * s2 * s2)
                    }
                }
            };
            let cap_w: Vec<f64> = per_cap
                .iter()
                .enumerate()
                .map(|(ci, &n)| {
                    if n == 0 {
                        return 1.0;
                    }
                    let n_eff = match model.capsules[ci].part {
                        super::model::Part::Head => self.params.surf_n_eff_head,
                        super::model::Part::Torso => self.params.surf_n_eff,
                        _ => self.params.surf_n_eff_limb,
                    };
                    let w = (n_eff / n as f64).min(1.0);
                    if matches!(model.capsules[ci].part, super::model::Part::Torso) {
                        w * torso_obliq
                    } else {
                        w
                    }
                })
                .collect();
            if n_assoc > 0 {
                // Per-capsule compressed normal equations: every point's
                // residual row is α·[Ja; Jb; radius] with a point-specific
                // 7-vector α, so Σ w αᵀα (7×7) and Σ w α r (7) per capsule
                // capture everything; the expansion into H is done once per
                // capsule instead of once per point.
                let nc = model.capsules.len();
                // α columns: [a(3), b(3), lateral(3), log-radius,
                // log-aspect]. The last is nonzero only for capsules with
                // an `aspect_group` (the trunk).
                const NA: usize = 11;
                let mut mm = vec![[[0.0f64; NA]; NA]; nc];
                let mut mv = vec![[0.0f64; NA]; nc];
                let mut used = vec![false; nc];
                for (pi, ci, d, q, u) in items {
                    let (pt, sig) = points[pi];
                    let inv_s = 1.0 / sig.max(1e-4);
                    let r = d * inv_s;
                    let (rho, w) = kern.eval(r * r);
                    let (rho, w) = (rho * cap_w[ci], w * cap_w[ci]);
                    cost += rho;
                    sumcl += d;
                    ncl += 1;
                    if !build {
                        continue;
                    }
                    let g = &caps[ci];
                    // α: ∂d/∂[a, b, c, log-radius] (× 1/σ). Round capsules
                    // analytically; elliptic ones by central differences on
                    // the closed-form distance (the lateral point c moves
                    // the section's orientation — that is the yaw gradient).
                    let mut alpha = [0.0f64; NA];
                    if g.c.is_none() {
                        let nhat = normalize(sub(pt, q));
                        for k in 0..3 {
                            alpha[k] = -(1.0 - u) * nhat[k] * inv_s;
                            alpha[3 + k] = -u * nhat[k] * inv_s;
                        }
                        alpha[9] = -g.r * inv_s;
                    } else {
                        // Forward differences (one extra distance per entry;
                        // the distance is smooth and the LM step tolerates
                        // O(ε) Jacobian error).
                        // 1e-5: the forward-difference bias is O(ε) and at
                        // 1e-4 it pushed the estimator's own gradient-vs-FD
                        // test past its 1e-3 tolerance on sheared-top
                        // operating points (measured 0.2–0.3 % mismatch).
                        const EPS: f64 = 1e-5;
                        let d0 = d;
                        for (blk, which) in [(0usize, 0usize), (3, 1), (6, 2)] {
                            for k in 0..3 {
                                let mut gp = g.clone();
                                match which {
                                    0 => gp.a[k] += EPS,
                                    1 => gp.b[k] += EPS,
                                    _ => {
                                        if let Some(cp) = gp.c.as_mut() {
                                            cp[k] += EPS;
                                        }
                                    }
                                }
                                alpha[blk + k] = (gp.dist(pt).0 - d0) / EPS * inv_s;
                            }
                        }
                        // ∂d/∂(log r): the Euclidean distance to an ellipse
                        // does not shrink exactly by the section radius.
                        let mut gp = g.clone();
                        gp.r *= (EPS * 10.0).exp();
                        alpha[9] = (gp.dist(pt).0 - d0) / (EPS * 10.0) * inv_s;
                        if model.capsules[ci].aspect_group.is_some() {
                            let mut gk = g.clone();
                            gk.k *= (EPS * 10.0).exp();
                            alpha[10] = (gk.dist(pt).0 - d0) / (EPS * 10.0) * inv_s;
                        }
                    }
                    let m = &mut mm[ci];
                    for k in 0..NA {
                        let ak = alpha[k] * w;
                        mv[ci][k] += ak * r;
                        for l in 0..NA {
                            m[k][l] += ak * alpha[l];
                        }
                    }
                    used[ci] = true;
                }
                if build {
                    let n = model.num_params;
                    for ci in 0..nc {
                        if !used[ci] {
                            continue;
                        }
                        let c = &model.capsules[ci];
                        // Local parameter set + B (L×NA).
                        self.row_idx.clear();
                        // row_acc doubles as "position in local list + 1".
                        let mut bloc: Vec<[f64; NA]> = Vec::with_capacity(64);
                        let touch =
                            |i: usize,
                             k: usize,
                             v: f64,
                             row_acc: &mut Vec<f64>,
                             row_idx: &mut Vec<usize>,
                             bloc: &mut Vec<[f64; NA]>| {
                                let pos = if row_acc[i] == 0.0 {
                                    row_idx.push(i);
                                    bloc.push([0.0; NA]);
                                    row_acc[i] = bloc.len() as f64;
                                    bloc.len() - 1
                                } else {
                                    row_acc[i] as usize - 1
                                };
                                bloc[pos][k] += v;
                            };
                        for &(i, v) in &self.cap_jac[3 * ci] {
                            for k in 0..3 {
                                touch(i, k, v[k], &mut self.row_acc, &mut self.row_idx, &mut bloc);
                            }
                        }
                        for &(i, v) in &self.cap_jac[3 * ci + 1] {
                            for k in 0..3 {
                                touch(
                                    i,
                                    3 + k,
                                    v[k],
                                    &mut self.row_acc,
                                    &mut self.row_idx,
                                    &mut bloc,
                                );
                            }
                        }
                        for &(i, v) in &self.cap_jac[3 * ci + 2] {
                            for k in 0..3 {
                                touch(
                                    i,
                                    6 + k,
                                    v[k],
                                    &mut self.row_acc,
                                    &mut self.row_idx,
                                    &mut bloc,
                                );
                            }
                        }
                        touch(
                            model.beta_rad + c.rad_group as usize,
                            9,
                            1.0,
                            &mut self.row_acc,
                            &mut self.row_idx,
                            &mut bloc,
                        );
                        if let Some(ag) = c.aspect_group {
                            touch(
                                model.beta_rad + ag as usize,
                                10,
                                1.0,
                                &mut self.row_acc,
                                &mut self.row_idx,
                                &mut bloc,
                            );
                        }
                        let l = self.row_idx.len();
                        // MB = B·M  (L×NA)
                        let m = &mm[ci];
                        let mut mb: Vec<[f64; NA]> = Vec::with_capacity(l);
                        for bi in &bloc {
                            let mut row = [0.0; NA];
                            for k in 0..NA {
                                let mut acc = 0.0;
                                for q in 0..NA {
                                    acc += bi[q] * m[q][k];
                                }
                                row[k] = acc;
                            }
                            mb.push(row);
                        }
                        let h = &mut self.dense.h;
                        let g = &mut self.dense.g;
                        for a in 0..l {
                            let ia = self.row_idx[a];
                            // gradient: g += Bᵀ mv
                            let mut ga = 0.0;
                            for k in 0..NA {
                                ga += bloc[a][k] * mv[ci][k];
                            }
                            g[ia] += ga;
                            let row = ia * n;
                            for bb in 0..l {
                                let ib = self.row_idx[bb];
                                let mut acc = 0.0;
                                for k in 0..NA {
                                    acc += mb[a][k] * bloc[bb][k];
                                }
                                h[row + ib] += acc;
                            }
                        }
                        for &i in &self.row_idx {
                            self.row_acc[i] = 0.0;
                        }
                    }
                }
            }
        }

        (cost, ncl, if ncl > 0 { sumcl / ncl as f64 } else { 0.0 })
    }
}

#[cfg(test)]
mod tests;
