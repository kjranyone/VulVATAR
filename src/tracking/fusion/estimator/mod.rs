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
//! * temporal prior `x ~ N(x̂, P + Q·dt)` from the previous posterior.
//!
//! Nothing here decides "is this joint visible": an unobserved joint simply
//! has no data terms, its variance grows by `Q·dt` per frame and the pose
//! prior pulls it toward the relaxed pose. Consumers read the marginal σ.

use super::math::*;
use super::model::*;

/// Pinhole intrinsics of the observing (colour) camera, pixels.
#[derive(Clone, Copy, Debug)]
pub struct Intrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub width: f64,
    pub height: f64,
}

impl Intrinsics {
    #[inline]
    pub fn project(&self, p: V3) -> Option<[f64; 2]> {
        if p[2] <= 0.05 {
            return None;
        }
        Some([
            self.fx * p[0] / p[2] + self.cx,
            self.fy * p[1] / p[2] + self.cy,
        ])
    }
    #[inline]
    pub fn deproject(&self, u: f64, v: f64, z: f64) -> V3 {
        [(u - self.cx) / self.fx * z, (v - self.cy) / self.fy * z, z]
    }
}

/// A model point an observation refers to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModelPoint {
    Joint(usize),
    Site(usize),
    /// A point rigidly attached to `joint` at `local` (metres, joint frame,
    /// already scaled — used for learned face landmarks).
    Attached {
        joint: usize,
        local: V3,
    },
}

/// One 2-D keypoint observation.
#[derive(Clone, Copy, Debug)]
pub struct Kp2d {
    pub point: ModelPoint,
    pub u: f64,
    pub v: f64,
    /// Isotropic pixel σ.
    pub sigma: f64,
}

/// One 3-D point observation (camera frame, metres).
#[derive(Clone, Copy, Debug)]
pub struct Kp3d {
    pub point: ModelPoint,
    pub p: V3,
    pub sigma: f64,
    /// Multiplier on the LATERAL (x/y, ≈ across-ray) σ. 1.0 = isotropic.
    /// Depth-lifted landmark points carry an honest z (the sensor) but
    /// lateral coordinates inherited from the 2-D landmark — when that
    /// landmark systematically frontalizes (dense face mesh), the lateral
    /// part must not outvote real orientation evidence.
    pub lat_scale: f64,
}

/// A direct orientation observation of one joint's WORLD (camera-frame)
/// rotation — e.g. the FaceMesh-derived head pose. Residual is the SO(3)
/// log of `R_world(joint) · R_targetᵀ`, whitened by `sigma` (rad).
#[derive(Clone, Copy, Debug)]
pub struct OriObs {
    pub joint: usize,
    pub target: M3,
    pub sigma: f64,
    /// How many joints up the kinematic chain (starting at `joint`) this
    /// observation is allowed to move. A head-pose obs with `depth: 2`
    /// adjusts head+neck only — a saturated / wrong face pose can then
    /// never twist the torso, whose orientation is owned by the body
    /// keypoints. `usize::MAX` = whole chain incl. root.
    pub chain_depth: usize,
}

/// Torso-yaw observation: the angle of the shoulder line in the camera
/// x/z plane, measured from the chest's depth slope. This is the only
/// channel that carries torso yaw from depth — a torso CAPSULE is
/// rotationally symmetric about its own axis, so the surface term is
/// blind to yaw by construction, leaving the 2-D shoulder pixels alone
/// to fix it (measured +17° of over-rotation against the depth
/// reference on a live desk session).
#[derive(Clone, Copy, Debug)]
pub struct ShoulderYawObs {
    /// Joint whose position is the LEFT end of the line.
    pub left: usize,
    /// … and the right end.
    pub right: usize,
    /// atan2(Δz_cam, Δx_cam) of the measured line, radians.
    pub yaw: f64,
    pub sigma: f64,
}

/// Everything observed at one capture time.
#[derive(Clone, Debug, Default)]
pub struct FrameObs {
    /// Capture time, seconds (device clock).
    pub t: f64,
    pub intr: Option<Intrinsics>,
    pub kp2d: Vec<Kp2d>,
    pub kp3d: Vec<Kp3d>,
    /// Direct world-orientation observations (see [`OriObs`]).
    pub ori: Vec<OriObs>,
    /// Torso yaw from the chest depth slope (see [`ShoulderYawObs`]).
    pub shoulder_yaw: Option<ShoulderYawObs>,
    /// Coarse torso reference (camera metres, e.g. shoulder-mid from
    /// depth) used only to seed the root on (re)acquisition.
    pub torso_hint: Option<V3>,
    /// Sparse surface points (camera metres, σ): depth samples under body
    /// keypoints, interpreted as "some body surface is here" (z-buffer
    /// semantics — an occluding arm in front of a shoulder is evidence for
    /// the arm, not a wrong shoulder). Associated to the nearest capsule of
    /// any part.
    pub surface: Vec<(V3, f64)>,
    /// Per-capsule association permission for `surface` (empty = all).
    /// Surface evidence may only be claimed by parts that are observed:
    /// the trunk and head always, a limb capsule only while its distal
    /// joint carries data. A point whose nearest capsule is a forbidden
    /// one is DROPPED, not re-assigned — an unobserved arm hanging at the
    /// side still occludes the trunk there. Without this, free
    /// (unobserved) arm capsules slide onto the chest and explain the
    /// torso surface at zero cost (measured: 1 400 of 4 800 points on the
    /// upper arms, trunk yaw 27° off).
    pub surf_allow: Vec<bool>,
}

/// Tunables. Units: pixels / metres / radians / seconds.
#[derive(Clone, Debug)]
pub struct Params {
    pub max_iters: usize,
    /// Robust kernel scale (whitened units) for 2-D / 3-D / cloud terms.
    pub c_2d: f64,
    pub c_3d: f64,
    /// Model-discrepancy σ (m) added in quadrature to surface points.
    pub surf_model_sigma: f64,
    pub gnc_start: f64,
    pub gnc_decay: f64,
    /// GNC start multiplier on tracked (warm-started) frames.
    pub gnc_tracked: f64,
    /// σ of the joint-limit hinge residual (rad) — small ⇒ hard limit.
    pub limit_sigma: f64,
    /// Process noise (variance per second) for joint rotations, root
    /// rotation, root translation, and shape.
    pub q_joint: f64,
    /// Process-noise scaling for unobserved limb joints: `q_joint` is
    /// multiplied by clamp(data_info_ema / q_hold_info_ref, q_hold_floor, 1).
    pub q_hold_floor: f64,
    pub q_hold_info_ref: f64,
    /// Trunk-axis prior (complements `upright_sigma`, which holds only the
    /// root): σ on the camera-z component of the unit pelvis→chest axis (0 = trunk perpendicular to the optical axis).
    /// In the desk envelope the pelvis is below the frame and nothing
    /// observes the trunk's lean; without an absolute prior the surface
    /// fit swings the unseen lower trunk toward the camera over a few
    /// hundred frames (measured: root depth 0.79 → 0.43 m while the chest
    /// stayed put). A camera pitched 20° up puts a truly upright trunk at
    /// 0.34; at σ 0.35 the prior was measured too weak to stop the drift
    /// (1.3σ against hundreds of cost units elsewhere), 0.15 holds it.
    /// `None` disables it.
    pub trunk_axis_sigma: Option<f64>,
    /// Process noise for the trunk (pelvis / spine / neck / clavicles):
    /// the torso turns far slower than a limb swings, so its yaw must keep
    /// memory across a frame where a shoulder is occluded.
    pub q_trunk: f64,
    pub q_root_rot: f64,
    pub q_root_t: f64,
    pub q_shape: f64,
    /// Prior σ of the per-group shape log-multipliers.
    pub shape_sigma: f64,
    /// Prior σ of the global scale (kept tight: per-group lengths absorb
    /// proportions, and a free global scale is what an unobserved chain
    /// pulls on when its detections are junk).
    pub scale_sigma: f64,
    /// Prior σ of the capsule radii (tight: a radius is only weakly
    /// observable from surface points and inflates to swallow outliers).
    pub radius_sigma: f64,
    /// Prior σ (m) of the trunk front-surface shear (`State::shear`).
    /// Identifiability: the visible front-surface depth slope measures
    /// only `axis_tilt − taper` (one equation, two unknowns); the taper is
    /// anatomical (session-constant) while posture fluctuates, so holding
    /// it in the slow shape state separates the pair over a session.
    /// Without it the solver books the whole chest slope (measured ≈ 11°
    /// on the s1789219959 desk replay) as trunk tilt and swings the
    /// out-of-frame pelvis to the desk plane. 0.04 m covers the
    /// chest-to-belly front offset spread of real torsos.
    /// `VULVATAR_TRUNK_SHEAR_SIGMA` overrides; a tiny value pins the
    /// shear at 0, reproducing the pre-shear behaviour for ablation.
    pub shear_sigma: f64,
    /// Velocity damping time constant (s) — the constant-velocity
    /// prediction decays toward zero over this horizon.
    pub velocity_tau: f64,
    /// Scale applied to the pose prior σ (1 = as defined in the model).
    pub pose_prior_scale: f64,
    /// Variance floor / cap of the carried covariance.
    pub var_min: f64,
    pub var_max: f64,
    /// σ (m) of the elbow-low prior: a hinge penalty on the elbow sitting
    /// ABOVE its shoulder (camera y is down, so elbow.y < shoulder.y).
    /// The elbow swivel is unobservable whenever the elbow keypoint is
    /// out of frame / culled (crop border, extreme close-up) and the LM
    /// otherwise parks it wherever the basin left it — humans rest
    /// elbows low. Real elbows-up poses out-pull this through the data.
    pub elbow_low_sigma: f64,
    /// σ (rad) of the upright-root prior: the pelvis "up" direction in the
    /// camera frame is pulled toward straight up (camera −y). Without it
    /// the pelvis/spine pitch split is unobservable in an upper-body
    /// framing and the solver parks the pelvis near-horizontal with the
    /// spine+neck curling to fit the face (measured 75–82° root tilt on
    /// the wave replay). Soft: a genuinely pitched camera pushes the
    /// residual tilt into gentle spine flexion instead.
    pub upright_sigma: f64,
    /// A re-seed candidate replaces the tracked pose only if it cuts the
    /// total cost to this fraction of it. Accepting ANY improvement makes
    /// the two hypotheses flap: on a live session 81 of 91 seed wins
    /// improved the cost by under 10% — statistical noise — while
    /// teleporting a wrist 0.15–0.56 m per event. Genuine re-acquisitions
    /// improve it several-fold (0.07–0.55), so a margin costs nothing.
    /// Swept on the replays: 0.98 and 0.95 both cut seed acceptances by
    /// ~87% and improve the palms torso error (std 7.8° → 6.4°); 0.90
    /// starts blocking real re-acquisitions and palms collapses
    /// (err std 31°).
    pub seed_win_ratio: f64,
    /// Median 2-D residual (px) above which the track counts as lost.
    pub lost_rms_px: f64,
    /// Use the heavy-tailed Cauchy kernel (vs Geman–McClure) on 2-D terms.
    pub cauchy_2d: bool,
    /// Front-side association gate for core capsules (m): a surface point
    /// farther out than this from the trunk / head surface is an occluder
    /// (hand across the chest measured at 0.10–0.20 m; a shirt fold at
    /// 0.02). Scaled by the GNC multiplier during bootstrap. Swept 0.06 →
    /// 0.03 on four recordings: torso yaw std sum 45.8 → 34.3, the palms
    /// replay 14.3 → 5.5.
    pub surf_front_gate_m: f64,
    /// Effective independent points per capsule for the dense surface
    /// term. The capsule model's shape error (±2–3 cm) is common to every
    /// point on a capsule, so N points do not carry N× the information;
    /// each capsule's point weights are scaled by min(1, n_eff / n).
    /// 30 points at σ ≈ 1.4 cm still pin a capsule's position to ~3 mm,
    /// while a direct orientation observation (FaceMesh OriObs) keeps
    /// its say over the round head capsule.
    pub surf_n_eff: f64,
    /// Same for the head (its round capsule is a crude face model; the
    /// FaceMesh centroid already fixes head position at pixel precision)
    /// and for limb capsules.
    pub surf_n_eff_head: f64,
    pub surf_n_eff_limb: f64,
    /// Kernel scale / gate for the sparse keypoint-lifted surface points.
    pub c_surf: f64,
    pub surf_gate_m: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            max_iters: 8,
            c_2d: 4.0,
            c_3d: 5.0,
            surf_model_sigma: 0.012,
            gnc_start: 16.0,
            gnc_decay: 0.5,
            gnc_tracked: std::env::var("VULVATAR_FUSION_GNC_TRACKED")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2.0),
            limit_sigma: 0.02,
            q_joint: 2.0, // rad²/s — a limb can swing ~250°/s
            // Measured worse at 0.25 (torso yaw std 6 → 21 on a gesturing
            // session: a held arm fights its returning observation and the
            // trunk pays); kept as an ablation knob.
            q_hold_floor: std::env::var("VULVATAR_HOLD_Q")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0),
            q_hold_info_ref: 50.0,
            trunk_axis_sigma: if std::env::var_os("VULVATAR_NO_UPRIGHT").is_some() {
                None
            } else {
                Some(
                    std::env::var("VULVATAR_TRUNK_AXIS_SIGMA")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0.15),
                )
            },
            q_trunk: 0.15, // rad²/s — the trunk turns ~120°/s at most
            q_root_rot: 0.15,
            q_root_t: std::env::var("VULVATAR_FUSION_Q_ROOT_T")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.005), // m²/s — ~0.35 m/s seated root motion
            q_shape: 1e-4,
            shape_sigma: 0.12,
            scale_sigma: 0.04,
            radius_sigma: 0.05,
            shear_sigma: std::env::var("VULVATAR_TRUNK_SHEAR_SIGMA")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| *v > 0.0)
                .unwrap_or(0.04),
            velocity_tau: 0.25,
            pose_prior_scale: 1.0,
            var_min: 1e-8,
            var_max: 25.0,
            elbow_low_sigma: 0.10,
            upright_sigma: 0.12,
            seed_win_ratio: 0.95,
            lost_rms_px: 40.0,
            cauchy_2d: true,
            surf_n_eff: std::env::var("VULVATAR_DENSE_NEFF")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(30.0),
            surf_n_eff_head: std::env::var("VULVATAR_DENSE_NEFF_HEAD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8.0),
            surf_n_eff_limb: 15.0,
            surf_front_gate_m: std::env::var("VULVATAR_DENSE_FRONT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.03),
            c_surf: 4.0,
            surf_gate_m: 0.20,
        }
    }
}

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
            gnc: 1.0,
            row_acc: vec![0.0; n],
            row_idx: Vec::with_capacity(128),
            row_out: Vec::with_capacity(128),
            row_out2: Vec::with_capacity(128),
            cap_jac: Vec::new(),
            surf_allow: Vec::new(),
            last_surf_assoc: Vec::new(),
            trunk_sites: {
                let find = |n: &str| model.sites.iter().position(|s| s.name == n);
                match (find("torso_lo"), find("torso_hi")) {
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
            locked_param: locked_params(model),
            data_info: vec![0.0; n],
            data_info_ema: vec![0.0; n],
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
        // Prior variance for this frame: P + Q dt (per parameter class).
        let mut prior_var = vec![0.0; n];
        for k in 0..n {
            let q = if k < 3 {
                self.params.q_root_rot
            } else if k < 6 {
                self.params.q_root_t
            } else if k < model.beta_scale {
                if self.trunk_param[k] {
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
                    self.params.q_joint * f
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
        let start = self.state.clone();
        let mut cost = self.lm_loop(model, obs, &prior_var, dt);
        let mut best_state = self.state.clone();
        let mut best_diag = self.diag;
        let mut best_gnc_final = true;
        for seed in seeds {
            let Some(cand) = seed(&start) else { continue };
            self.state = cand;
            let c = self.lm_loop(model, obs, &prior_var, dt);
            if std::env::var_os("VULVATAR_SEED_DUMP").is_some() {
                eprintln!(
                    "SEEDCAND cost {cost:.1} cand {c:.1} ratio {:.3}",
                    c / cost.max(1e-6)
                );
            }
            if c < cost * self.params.seed_win_ratio {
                cost = c;
                best_state = self.state.clone();
                best_diag = self.diag;
                best_gnc_final = false;
                self.diag.seed_wins += 1;
            }
        }
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
        self.diag.cost_final = cost;
        self.finish(model, &prev, dt, obs.t);
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
        let mut cost = self.accumulate(model, obs, &fk, prior_var, dt, true);
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
                self.apply_locks();
                if self
                    .dense
                    .solve_damped(lambda, 1e-9, &mut self.delta)
                    .is_none()
                {
                    lambda *= 10.0;
                    continue;
                }
                let mut trial = self.state.clone();
                trial.apply_delta(model, &self.delta);
                let fk_trial = model.fk(&trial);
                let cost_trial = self.eval_cost(model, obs, &fk_trial, &trial, prior_var, dt);
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
                        return self.accumulate(model, obs, &fk, prior_var, dt, true);
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
            cost = self.accumulate(model, obs, &fk, prior_var, dt, true);
        }
        let _ = cost;
        self.gnc = 1.0;
        self.accumulate(model, obs, &fk, prior_var, dt, true)
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
            for kp in &obs.kp2d {
                let (joint, pw) = resolve_point(model, fk, kp.point);
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

        // ---- 3-D points -----------------------------------------------------
        {
            let c3 = p.c_3d * self.gnc;
            let kern = Kernel::Cauchy(c3);
            for kp in &obs.kp3d {
                let (joint, pw) = resolve_point(model, fk, kp.point);
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
                        // prior
                        let sp = (jd.prior_sigma[0] * p.pose_prior_scale).max(1e-4);
                        let r = (a - jd.prior_mean[0]) / sp;
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
                            let sp = (jd.prior_sigma[k] * p.pose_prior_scale).max(1e-4);
                            let rp = (w[k] - jd.prior_mean[k]) / sp;
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
            let inv_sh = 1.0 / if self.shape_frozen { 0.005 } else { p.shear_sigma };
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
        if build {
            self.diag.cost_2d = c2d;
            self.diag.cost_3d = c3d;
            self.diag.cost_cloud = ccl;
            self.diag.cost_prior = cpr;
            self.diag.cost_temporal = ctm;
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
                    k: c.aspect,
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
                    (n_eff / n as f64).min(1.0)
                })
                .collect();
            if n_assoc > 0 {
                // Per-capsule compressed normal equations: every point's
                // residual row is α·[Ja; Jb; radius] with a point-specific
                // 7-vector α, so Σ w αᵀα (7×7) and Σ w α r (7) per capsule
                // capture everything; the expansion into H is done once per
                // capsule instead of once per point.
                let nc = model.capsules.len();
                const NA: usize = 10;
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

/// Locked parameters: the pelvis joint (redundant with the root rotation).
fn locked_params(model: &Model) -> Vec<bool> {
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
fn point_y_jac(
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
fn trunk_params(model: &Model) -> Vec<bool> {
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
fn point_jac(
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

#[cfg(test)]
mod tests;
