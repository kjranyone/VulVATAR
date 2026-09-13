//! Solver tunables: iteration bounds, robust-kernel scales, gates and
//! per-channel weights. Units: pixels / metres / radians / seconds.
//! Re-exported from the estimator root.

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
    /// out-of-frame pelvis to the desk plane.
    ///
    /// DEFAULT DISABLED (1e-9 pins the shear at 0 = the pre-shear
    /// behaviour). Benched on s1789219959, a FREE shear (0.03) made the
    /// equilibrium WORSE on the very lean it was meant to fix (torso
    /// pitch −11° → −23/−24°, pelvis still 0.46 m) while roughly halving
    /// wrist snaps — the recording's slope signal is not anatomy-dominated
    /// the way the confound model assumed, and separating "the subject is
    /// genuinely reclined" from "the fit is" needs independent ground
    /// truth (a deliberately-upright recording, or a synthetic slope
    /// fixture through `validate_gt`) before this can be enabled with a
    /// calibrated σ. `VULVATAR_TRUNK_SHEAR_SIGMA=0.03` re-enables.
    pub shear_sigma: f64,
    /// Obliquity gain for the torso dense-surface weight. Measured
    /// (s1789246071 ablation, 2026-09-13): at the habitual oblique desk
    /// pose (~40° trunk yaw) the dense torso-surface fit booked 35-40° of
    /// phantom spine/chest recline while the person sat upright (dense-off
    /// solves upright; the elliptical capsule section's approximation
    /// error projects into the trunk pitch DOF at oblique view). The
    /// weight of torso surface points is divided by
    /// `1 + k·sin⁴(trunk yaw)` — sin⁴ is a near-deadzone that keeps
    /// frontal view at full primary-observation weight (sin² let the
    /// frontal synthetic's yaw wander into the down-weighting region and
    /// drift), while at the habitual 40° pose it restores the upright
    /// solve (replay s1789246071: avatar Chest→Neck lean +57° → +21° ≈
    /// the asset rest profile; the shoulder-depth yaw reference on
    /// streamer_da also tracks better, err +9.4° → ~−5°). Flat optimum
    /// over k 20-30. `VULVATAR_TRUNK_SURF_OBLIQ` overrides; 0 disables.
    pub trunk_surf_obliq_k: f64,
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
    /// σ (m) of the wrist hold: when a wrist carries NO observation in
    /// the current frame, its root-frame position is pinned to the
    /// prediction by a plain (non-robust) pseudo-observation. With the
    /// elbow typically observed but the hand under the desk, the
    /// forearm's direction (flexion + twist, 2 DOF) is a null direction
    /// of every data term — only priors decide it, and the solver flips
    /// between their shallow basins frame to frame (measured on the
    /// s1789219959 desk replay: 29 of 31 wrist snaps > 8 cm with σ ≈ 0.6,
    /// no observation change, temporal cost 11.5× its median). Holding
    /// the wrist position (root-frame, so leaning root motion does not
    /// fight it) eliminates the snaps (replay: R wrist snaps > 15 cm
    /// 15 → 0, max jump 0.58 → 0.13 m, head yaw sd 16.1 → 15.1) at a
    /// small torso cost (yaw sd 2.8 → 3.5). Narrower holds were measured
    /// worse: the swivel angle alone (1 of the 2 DOF) leaves the snaps
    /// (13 of 15 remain); a shoulder-anchored position hold drags the
    /// torso (yaw err sd 2.7 → 4.5). The parameter-space hold
    /// (`q_hold_floor`) cannot treat this either: it was measured worse
    /// at 0.25 because a held arm fights its *returning* observation —
    /// hence this hold is gated off per frame whenever the wrist has any
    /// 2-D/3-D observation, so a real observation (or a re-seed, which
    /// needs one) always wins. 0 disables.
    pub wrist_hold_sigma: f64,
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
                // 1e-9 = pinned (disabled by default; see the field doc).
                .unwrap_or(1e-9),
            trunk_surf_obliq_k: std::env::var("VULVATAR_TRUNK_SURF_OBLIQ")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| *v >= 0.0)
                // 25 = middle of the flat optimum (k 20-30), see field doc.
                .unwrap_or(25.0),
            velocity_tau: 0.25,
            pose_prior_scale: std::env::var("VULVATAR_FUSION_PRIOR_SCALE")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| *v > 0.0)
                .unwrap_or(1.0),
            var_min: 1e-8,
            var_max: 25.0,
            elbow_low_sigma: 0.10,
            wrist_hold_sigma: if std::env::var_os("VULVATAR_FUSION_NO_WHOLD").is_some() {
                0.0
            } else {
                std::env::var("VULVATAR_FUSION_WHOLD_SIGMA")
                    .ok()
                    .and_then(|v| v.parse::<f64>().ok())
                    .filter(|v| *v > 0.0)
                    .unwrap_or(0.02)
            },
            // `VULVATAR_UPRIGHT_SIGMA` is a bench knob (desk-lean sweeps);
            // 0.12 is the calibrated default — a 40° tilt only pays ~2×(sin
            // 40°/σ)² ≈ 29 cost units per component here, which the
            // surface/2-D terms outvote by the hundreds (measured live).
            upright_sigma: std::env::var("VULVATAR_UPRIGHT_SIGMA")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| *v > 0.0)
                .unwrap_or(0.12),
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
