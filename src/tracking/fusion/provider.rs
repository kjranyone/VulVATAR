//! `FusionProvider` — the tracking-v2 production pipeline behind
//! [`PoseProvider`]: RTMW3D (2-D keypoints + SimCC σ, FaceMesh landmarks
//! and blendshapes) + the D435 point cloud → the fusion estimator →
//! [`RigPose`] (published on the compatibility `SourceSkeleton`).

use std::path::Path;
use std::sync::Arc;

use log::info;

use crate::tracking::provider::{PoseProvider, TrackingPipelineConfig};
use crate::tracking::rtmw3d::{Rtmw3dInference, Rtmw3dOptions};
use crate::tracking::metric_frame::MetricDepthFrame;
use crate::tracking::source_skeleton::CameraIntrinsics;
use crate::tracking::PoseEstimate;

use super::estimator::{Estimator, FrameObs, Intrinsics, Params};
use super::math::*;
use super::model::*;
use super::observe::*;
use super::output;

/// Number of RTMW3D face-68 landmarks (COCO-Wholebody 23..=90).
const FACE68: usize = 68;
const FACE68_BASE: usize = 23;
/// FaceMesh landmark count.
const MESH_N: usize = 478;

pub struct FusionProvider {
    rtmw3d: Rtmw3dInference,
    hands: Option<super::hands::HandLandmarker>,
    /// Last hand-crop results `[left, right]` (diagnostics).
    pub last_hands: [Option<super::hands::HandResult>; 2],
    /// Last LOCKED hand results (persist across a frame where every crop
    /// missed) — the seed for the temporal crop candidate.
    prev_hands: [Option<super::hands::HandResult>; 2],
    h: Humanoid,
    est: Estimator,
    body_map: BodyMap,
    face68: FaceShape,
    mesh: FaceShape,
    external_depth: Option<MetricDepthFrame>,
    load_warnings: Vec<String>,
    /// Last capture time (s) handed to the estimator.
    last_t: Option<f64>,
    frames: u64,
    pub kp_sigma: KpSigma,
    /// Cumulative count of frames where the detector's hand blocks were
    /// re-labelled L↔R against the predicted wrists.
    pub hand_swaps: u64,
    /// Index in `obs.kp2d` where the hand-crop keypoints of the current
    /// frame start (for the per-arm σ inflation pass).
    hand_kp_start: usize,
    /// Per-frame timing (ms) of the last estimate for diagnostics.
    pub last_solve_ms: f32,
    /// Estimator-only time (ms) of the last frame (excludes inference).
    pub last_est_ms: f32,
    /// Diagnostics: the cloud handed to the estimator this frame (subsampled
    /// exactly as the estimator saw it) and the sparse surface points.
    pub last_cloud: Vec<[f32; 3]>,
    pub last_surface: Vec<[f32; 3]>,
    /// Diagnostics: metric joint observations `(joint index, point, σ)`.
    pub last_kp3d: Vec<(usize, V3, f64)>,
}

impl FusionProvider {
    pub fn from_models_dir_with_config(
        models_dir: impl AsRef<Path>,
        config: TrackingPipelineConfig,
    ) -> Result<Self, String> {
        let dir = models_dir.as_ref();
        let mut rtmw3d = Rtmw3dInference::from_models_dir_with_options(
            dir,
            Rtmw3dOptions {
                face_ep: crate::tracking::face_mediapipe::FaceMeshEp::Auto,
                force_cpu: config.force_cpu,
                yolox_enabled: config.yolox_enabled,
            },
        )?;
        let mut warnings = rtmw3d.take_load_warnings();
        let hands = match super::hands::HandLandmarker::try_from_models_dir(dir) {
            Ok(h) => {
                if h.is_none() {
                    warnings.push("hand landmarker model missing (models/mediapipe_hand_landmark.onnx): fingers use the body detector only".to_string());
                }
                h
            }
            Err(e) => {
                warnings.push(format!("hand landmarker unavailable: {e}"));
                None
            }
        };
        info!("Fusion provider ready (RTMW3D {})", rtmw3d.backend().label());
        let h = Humanoid::new();
        let mut params = Params::default();
        // Bench override: dense-cloud information budget (0 = off).
        if let Some(b) = std::env::var("VULVATAR_FUSION_CLOUD_BUDGET").ok().and_then(|v| v.parse::<f64>().ok()) {
            params.cloud_budget = b;
        }
        let est = Estimator::new(&h.model, params);
        let body_map = BodyMap::new(&h);
        Ok(Self {
            rtmw3d,
            hands,
            last_hands: [None, None],
            prev_hands: [None, None],
            h,
            est,
            body_map,
            face68: FaceShape::new(FACE68, 8),
            mesh: FaceShape::new(MESH_N, 8),
            external_depth: None,
            load_warnings: warnings,
            last_t: None,
            frames: 0,
            kp_sigma: KpSigma::default(),
            hand_swaps: 0,
            hand_kp_start: 0,
            last_solve_ms: 0.0,
            last_est_ms: 0.0,
            last_cloud: Vec::new(),
            last_surface: Vec::new(),
            last_kp3d: Vec::new(),
        })
    }

    pub fn humanoid(&self) -> &Humanoid {
        &self.h
    }
    pub fn estimator(&self) -> &Estimator {
        &self.est
    }
    pub fn face68_learned(&self) -> usize {
        self.face68.learned_count()
    }
    pub fn mesh_learned(&self) -> usize {
        self.mesh.learned_count()
    }

    /// Which arm (0 = left, 1 = right) a model point belongs to, if any
    /// (shoulder joint excluded: it is torso-anchored).
    fn arm_side_of(&self, mp: super::estimator::ModelPoint) -> Option<usize> {
        use super::estimator::ModelPoint;
        let j = match mp {
            ModelPoint::Joint(j) => j,
            ModelPoint::Site(s) => self.h.model.sites[s].joint,
            ModelPoint::Attached { joint, .. } => joint,
        };
        let m = &self.h.model;
        let mut k = j;
        loop {
            if k == self.h.j.l_elbow {
                return Some(0);
            }
            if k == self.h.j.r_elbow {
                return Some(1);
            }
            match m.joints[k].parent {
                Some(p) => k = p,
                None => return None,
            }
        }
    }

    fn nominal_intrinsics(width: u32, height: u32) -> Intrinsics {
        // ~69° horizontal FOV (D435 colour) when no sensor intrinsics are
        // available (RGB-only replay).
        let f = width as f64 * 0.73;
        Intrinsics {
            fx: f,
            fy: f,
            cx: width as f64 * 0.5,
            cy: height as f64 * 0.5,
            width: width as f64,
            height: height as f64,
        }
    }
}

impl PoseProvider for FusionProvider {
    fn label(&self) -> String {
        format!("Fusion v2 / RTMW3D {}", self.rtmw3d.backend().label())
    }

    fn take_load_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.load_warnings)
    }

    fn reset_temporal_state(&mut self) {
        self.rtmw3d.reset_temporal_state();
        self.est.reset(&self.h.model);
        self.prev_hands = [None, None];
        self.face68.reset();
        self.mesh.reset();
        self.last_t = None;
        self.frames = 0;
    }

    fn set_external_depth(&mut self, depth: MetricDepthFrame) {
        self.external_depth = Some(depth);
    }

    fn estimate_pose(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> PoseEstimate {
        let t0 = std::time::Instant::now();
        let depth = self.external_depth.take();
        let ts_ms = depth.as_ref().and_then(|d| d.timestamp_ms);
        // Time base: device clock when available, else nominal 30 fps.
        let t = match ts_ms {
            Some(ms) => ms / 1000.0,
            None => frame_index as f64 / 30.0,
        };
        self.rtmw3d.set_frame_timestamp_ms(ts_ms);
        let base = self.rtmw3d.estimate_pose(rgb_data, width, height, frame_index);
        let aux = self.rtmw3d.take_aux();

        let intr_cam: Option<CameraIntrinsics> = depth.as_ref().and_then(|d| d.intrinsics);
        let intr = match intr_cam {
            Some(i) => Intrinsics {
                fx: i.fx as f64,
                fy: i.fy as f64,
                cx: i.cx as f64,
                cy: i.cy as f64,
                width: i.width as f64,
                height: i.height as f64,
            },
            None => Self::nominal_intrinsics(width, height),
        };

        let mut obs = FrameObs {
            t,
            intr: Some(intr),
            kp2d: Vec::with_capacity(700),
            kp3d: Vec::with_capacity(600),
            cloud: None,
            torso_hint: None,
            surface: Vec::new(),
        };

        // Prediction for this frame (drives ROI extraction + face-shape
        // observation geometry before the solve).
        let pred = self.est.predict(&self.h.model, t);
        let fk_pred = self.h.model.fk(&pred);
        let head_j = self.h.j.head;
        let head_center_pred = fk_pred.site[self.h.s.head_center];

        // Depth accessor over the aligned point cloud for FACE landmarks:
        // tight person band around the predicted head, plus a model
        // z-buffer test — a hand / forearm sweeping in front of the face
        // (the wave) sits within ~15 cm of the face plane, and without the
        // occlusion test its surface depth would be attributed to the face
        // landmarks, yanking the head (and with it the root) toward the
        // camera every pass.
        let head_depth_pred = norm(head_center_pred);
        let arm_capsules: Vec<(V3, V3, f64)> = self
            .h
            .model
            .capsules
            .iter()
            .filter(|c| {
                matches!(
                    c.part,
                    super::model::Part::LeftArm
                        | super::model::Part::RightArm
                        | super::model::Part::LeftHand
                        | super::model::Part::RightHand
                )
            })
            .map(|c| {
                (
                    fk_pred.point(c.a),
                    fk_pred.point(c.b),
                    self.h.model.capsule_radius(&pred, c) + 0.03,
                )
            })
            .collect();
        // Active hand regions (crop rectangles, slightly padded): a palm
        // in front of the face fills these, and any face landmark whose
        // pixel falls inside must not take depth from there — it would be
        // the palm's surface, not the face's.
        let hand_rects: Vec<(f32, f32, f32, f32)> = self
            .last_hands
            .iter()
            .flatten()
            .map(|hr| {
                let (x, y, sz) = hr.crop;
                let pad = sz * 0.05;
                (x + pad, y + pad, x + sz - pad, y + sz - pad)
            })
            .collect();
        let in_hand_rect = |u: f64, v: f64| -> bool {
            hand_rects.iter().any(|&(x0, y0, x1, y1)| {
                u >= x0 as f64 && u <= x1 as f64 && v >= y0 as f64 && v <= y1 as f64
            })
        };
        let depth_at = |u: f64, v: f64| -> Option<V3> {
            let d = depth.as_ref()?;
            if in_hand_rect(u, v) {
                return None;
            }
            let p = window_point(
                &d.points_m,
                d.width,
                d.height,
                u,
                v,
                1,
                (head_center_pred[2] - 0.12) as f32,
                (head_center_pred[2] + 0.12) as f32,
            )?;
            // Occlusion: does an arm/hand capsule cut this ray in front of
            // the head?
            let t_obs = norm(p);
            if t_obs > 1e-6 {
                let dir = scale(p, 1.0 / t_obs);
                for &(a, b, r) in &arm_capsules {
                    if let Some(t_hit) = super::estimator::ray_capsule_entry(dir, a, b, r) {
                        if t_hit < head_depth_pred - 0.05 {
                            return None;
                        }
                    }
                }
            }
            Some(p)
        };

        // ---- hand-block L/R assignment --------------------------------------
        // The detector's left/right hand blocks can be transposed when the
        // hands cross or touch. Decide the assignment against the predicted
        // wrists (only when both wrists are currently tracked): keep the
        // labelling unless swapping is clearly better.
        let mut det_kps: Vec<(f32, f32, f32)> = base.annotation.keypoints.clone();
        if let Some(d) = depth.as_ref() {
            if d.points_m.len() == (d.width * d.height) as usize && det_kps.len() >= 133 {
                let mut tmp: Vec<RawKp> = det_kps
                    .iter()
                    .map(|&(nx, ny, sc)| RawKp { nx, ny, score: sc, sx: 0.0, sy: 0.0 })
                    .collect();
                let zs = [fk_pred.t[self.h.j.l_shoulder][2], fk_pred.t[self.h.j.r_shoulder][2]];
                reach_filter(&mut tmp, &d.points_m, d.width, d.height, zs, 0.75);
                for (k, t) in det_kps.iter_mut().zip(tmp.iter()) {
                    k.2 = t.score;
                }
            }
        }
        let mut hands_swapped = false;
        if det_kps.len() >= 133 {
            let wl = det_kps[91];
            let wr = det_kps[112];
            let pl = intr.project(fk_pred.t[self.h.j.l_wrist]);
            let pr = intr.project(fk_pred.t[self.h.j.r_wrist]);
            let tracked = self.est.joint_data_sigma(&self.h.model, self.h.j.l_wrist) < 0.5
                && self.est.joint_data_sigma(&self.h.model, self.h.j.r_wrist) < 0.5;
            if let (Some(pl), Some(pr)) = (pl, pr) {
                if tracked && wl.2 >= 0.3 && wr.2 >= 0.3 {
                    let d = |a: (f32, f32, f32), b: [f64; 2]| {
                        ((a.0 as f64 * width as f64 - b[0]).powi(2)
                            + (a.1 as f64 * height as f64 - b[1]).powi(2))
                        .sqrt()
                    };
                    let same = d(wl, pl) + d(wr, pr);
                    let swap = d(wl, pr) + d(wr, pl);
                    if swap < 0.6 * same && same > 60.0 {
                        for k in 0..21 {
                            det_kps.swap(91 + k, 112 + k);
                        }
                        hands_swapped = true;
                        self.hand_swaps += 1;
                    }
                }
            }
        }
        // ---- hand crops (state-driven) --------------------------------------
        let mut hand_done = [false; 2];
        self.last_hands = [None, None];
        if let Some(hl) = self.hands.as_mut() {
            for hand in 0..2 {
                // Crop candidates, best first: (1) last frame's locked hand
                // re-cropped around its own landmarks (tightest, survives a
                // detector miss), (2) the body detector's hand block, (3)
                // the model prediction. Try until one locks (presence is
                // bimodal — a hit reads ~0.9+, a miss ~0).
                let mut candidates: Vec<(f32, f32, f32)> = Vec::with_capacity(3);
                if let Some(prev) = self.prev_hands[hand].as_ref() {
                    let (mut x0, mut y0, mut x1, mut y1) =
                        (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
                    for p in &prev.px {
                        x0 = x0.min(p[0]);
                        y0 = y0.min(p[1]);
                        x1 = x1.max(p[0]);
                        y1 = y1.max(p[1]);
                    }
                    let span = (x1 - x0).max(y1 - y0).max(48.0);
                    let size = span * 1.7;
                    candidates.push((
                        0.5 * (x0 + x1) - size * 0.5,
                        0.5 * (y0 + y1) - size * 0.5,
                        size,
                    ));
                }
                if let Some(c) =
                    super::hands::detector_hand_crop(&det_kps, hand, width, height, 0.35, 96.0)
                {
                    candidates.push(c);
                }
                if let Some(c) =
                    super::hands::predicted_hand_crop(&self.h, &fk_pred, hand, &intr, 96.0)
                {
                    candidates.push(c);
                }
                let mut best: Option<super::hands::HandResult> = None;
                for crop in candidates {
                    if let Some(res) = hl.estimate(rgb_data, width, height, crop) {
                        let better =
                            best.as_ref().map(|b| res.presence > b.presence).unwrap_or(true);
                        if better {
                            let lock = res.presence >= 0.6;
                            best = Some(res);
                            if lock {
                                break;
                            }
                        }
                    }
                }
                match best {
                    Some(res) if res.presence >= 0.5 => {
                        hand_done[hand] = true;
                        self.prev_hands[hand] = Some(res.clone());
                        self.last_hands[hand] = Some(res);
                    }
                    _ => self.prev_hands[hand] = None,
                }
            }
        }

        // ---- per-arm evidence burn-in -----------------------------------------
        // An arm that has not been observed for a while sits on its prior;
        // the first detections after that carry inflated σ (a hallucinated
        // hand at the desk edge and a real re-entering hand look identical
        // for one frame) and the inflation decays as the arm's measurement
        // information accumulates over ~0.3 s. Continuous, not a gate: the
        // arm still moves on frame one, just proportionally to the evidence.
        let arm_inflate = |wrist: usize, elbow: usize| -> f64 {
            let m = &self.h.model;
            let info = |j: usize| {
                let p = m.joint_param[j];
                let n = match m.joints[j].kind {
                    JointKind::Ball { .. } => 3,
                    JointKind::Hinge { .. } => 1,
                };
                (0..n)
                    .map(|k| self.est.data_info_ema.get(p + k).copied().unwrap_or(0.0))
                    .fold(0.0, f64::max)
            };
            let trust = (info(wrist).max(info(elbow)) / 200.0).clamp(0.0, 1.0);
            if std::env::var_os("VULVATAR_FUSION_NO_BURNIN").is_some() {
                return 1.0;
            }
            1.0 + 3.0 * (1.0 - trust)
        };
        let inflate = [
            arm_inflate(self.h.j.l_wrist, self.h.j.l_elbow),
            arm_inflate(self.h.j.r_wrist, self.h.j.r_elbow),
        ];

        // ---- body keypoints -------------------------------------------------
        let mut face68_px: Vec<[f32; 3]> = Vec::new();
        if let Some(aux) = aux.as_ref() {
            let mut raw: Vec<RawKp> = aux
                .joints
                .iter()
                .map(|j| RawKp {
                    nx: j.nx,
                    ny: j.ny,
                    score: j.score,
                    sx: j.sx,
                    sy: j.sy,
                })
                .collect();
            if let Some(crop) = aux.crop {
                cull_crop_border(&mut raw, crop, width, height, 0.02);
            }
            // Arm reachability against the predicted shoulders (depth-valid
            // pixels only) — see `reach_filter`.
            if let Some(d) = depth.as_ref() {
                if d.points_m.len() == (d.width * d.height) as usize {
                    let zs = [fk_pred.t[self.h.j.l_shoulder][2], fk_pred.t[self.h.j.r_shoulder][2]];
                    if std::env::var_os("VULVATAR_FUSION_NO_REACH").is_none() {
                        reach_filter(&mut raw, &d.points_m, d.width, d.height, zs, 0.75);
                    }
                }
            }
            if hands_swapped {
                for k in 0..21 {
                    raw.swap(91 + k, 112 + k);
                }
            }
            // Leg gating: in the desk envelope the legs are physically out
            // of frame, but the detector hallucinates hips/knees/ankles on
            // chair edges, desks and raised palms with mid confidence,
            // which flails the avatar's legs. A leg detection is only
            // meaningful when its hip is confidently in frame AND sits
            // anatomically below the shoulder line (the hallucinations
            // cluster at shoulder height on whatever object is in front).
            {
                let sh_y = 0.5 * (raw[5].ny + raw[6].ny);
                let nose_y = raw[0].ny;
                let torso_ref = (sh_y - nose_y).abs().max(0.05);
                for side in 0..2 {
                    let hip = &raw[11 + side];
                    let hip_ok = hip.score >= 0.5
                        && hip.nx > 0.02
                        && hip.nx < 0.98
                        && hip.ny > 0.02
                        && hip.ny < 0.98
                        && hip.ny > sh_y + 0.6 * torso_ref;
                    if !hip_ok {
                        raw[11 + side].score = 0.0;
                        for i in
                            [13 + side, 15 + side, 17 + 3 * side, 18 + 3 * side, 19 + 3 * side]
                        {
                            if i < raw.len() {
                                raw[i].score = 0.0;
                            }
                        }
                    }
                }
            }
            // Duplicate-wrist degeneracy: a single palm thrust at the camera
            // routinely captures BOTH wrist detections (and both hand
            // blocks). Two wrists on one physical hand pull both arms to
            // the same point and twist the torso; keep the better-scored
            // side and relax the other.
            {
                let (lw, rw) = (raw[9], raw[10]);
                let close = {
                    let dx = (lw.nx - rw.nx) * width as f32;
                    let dy = (lw.ny - rw.ny) * height as f32;
                    (dx * dx + dy * dy).sqrt() < 0.04 * width as f32
                };
                if close && lw.score > 0.0 && rw.score > 0.0 {
                    let drop_left = lw.score < rw.score;
                    let (wrist_i, base) = if drop_left { (9, 91) } else { (10, 112) };
                    raw[wrist_i].score = 0.0;
                    for k in raw.iter_mut().skip(base).take(21) {
                        k.score = 0.0;
                    }
                }
            }
            // The dedicated hand crop supersedes the body detector's hand
            // block for that hand (keep the wrist: it anchors the crop).
            for hand in 0..2 {
                if hand_done[hand] {
                    let base = if hand == 0 { 91 } else { 112 };
                    for k in raw.iter_mut().skip(base + 1).take(20) {
                        k.score = 0.0;
                    }
                }
            }
            let n_before = obs.kp2d.len();
            body_kp2d(
                &self.body_map,
                &raw,
                width,
                height,
                self.kp_sigma,
                1.5,
                &mut obs.kp2d,
            );
            for k in obs.kp2d[n_before..].iter_mut() {
                if let Some(side) = self.arm_side_of(k.point) {
                    k.sigma *= inflate[side];
                }
            }
            // Depth-lifted joints: the absolute-depth anchor that resolves
            // the projective scale/distance ambiguity of the 2-D terms.
            if let Some(d) = depth.as_ref() {
                if d.points_m.len() == (d.width * d.height) as usize {
                    let z_ref = if self.est.last_t.is_some() {
                        Some(head_center_pred[2])
                    } else {
                        shoulder_depth_ref(&raw, &d.points_m, d.width, d.height, 0.3)
                    };
                    obs.torso_hint = shoulder_mid_hint(&raw, &d.points_m, d.width, d.height, 0.3);
                    body_kp3d(
                        &self.body_map,
                        &raw,
                        &d.points_m,
                        d.width,
                        d.height,
                        z_ref,
                        0.3,
                        &mut obs.kp3d,
                    );
                    body_torso_leg_depth(
                        &self.body_map,
                        &raw,
                        &d.points_m,
                        d.width,
                        d.height,
                        z_ref,
                        0.3,
                        &self.h.model,
                        &fk_pred,
                        &pred,
                        &mut obs.kp3d,
                        &mut obs.surface,
                    );
                }
            }
            // Face-68 block in frame pixels (learned shape).
            for k in 0..FACE68 {
                let j = &aux.joints[FACE68_BASE + k];
                if j.score >= 0.3 {
                    face68_px.push([j.nx * width as f32, j.ny * height as f32, 0.0]);
                } else {
                    face68_px.push([f32::NAN, f32::NAN, 0.0]);
                }
            }
        }
        let face68_valid: Vec<[f32; 3]> = face68_px.clone();
        let mesh_px: Option<&Vec<[f32; 3]>> = aux
            .as_ref()
            .and_then(|a| a.face_mesh.as_ref())
            .filter(|(_, conf)| *conf >= 0.5)
            .map(|(lm, _)| lm);

        // Learned-face observations (skip NaN entries).
        {
            let head_sigma_pred = self.est.joint_sigma(&self.h.model, head_j);
            let sigma_scale = if head_sigma_pred < 0.2 { 1.0 } else { 2.0 };
            let clean68: Vec<[f32; 3]> = face68_valid.clone();
            self.face68.observe(
                head_j,
                &clean68,
                3.0 * sigma_scale,
                0.010,
                &depth_at,
                head_center_pred[2],
                &mut obs.kp2d,
                &mut obs.kp3d,
            );
            if let Some(lm) = mesh_px {
                self.mesh.observe(
                    head_j,
                    lm,
                    2.0 * sigma_scale,
                    0.008,
                    &depth_at,
                    head_center_pred[2],
                    &mut obs.kp2d,
                    &mut obs.kp3d,
                );
            }
        }
        // NaN landmarks produce NaN residuals: drop them defensively.
        obs.kp2d.retain(|k| k.u.is_finite() && k.v.is_finite());

        // Far-side head-landmark culling: at a 3/4 or profile view the
        // detector still emits the occluded eye/ear (and the occluded half
        // of the dense face landmarks) hallucinated near the visible
        // silhouette — those 2-D residuals fight the yaw and pin the head
        // ~50% short of the real turn. A landmark whose *predicted* surface
        // normal (radial from the head centre) faces away from the camera
        // is on the far side of the head and cannot be a real observation.
        {
            let head_c = fk_pred.site[self.h.s.head_center];
            let head_sites = [self.h.s.l_eye, self.h.s.r_eye, self.h.s.l_ear, self.h.s.r_ear];
            let facing_of = |pw: V3| -> f64 {
                let n = normalize(sub(pw, head_c));
                let to_cam = normalize(scale(pw, -1.0));
                dot(n, to_cam)
            };
            let h = &self.h;
            let pred = &pred;
            let fkp = &fk_pred;
            obs.kp2d.retain(|k| {
                use super::estimator::ModelPoint;
                let pw = match k.point {
                    ModelPoint::Site(sid) if head_sites.contains(&sid) => fkp.site[sid],
                    ModelPoint::Attached { joint, local } if joint == h.j.head => {
                        let _ = pred;
                        add(fkp.t[joint], mat_vec(&fkp.r[joint], local))
                    }
                    _ => return true,
                };
                facing_of(pw) > -0.15
            });
            obs.kp3d.retain(|k| {
                use super::estimator::ModelPoint;
                let pw = match k.point {
                    ModelPoint::Attached { joint, local } if joint == h.j.head => {
                        add(fkp.t[joint], mat_vec(&fkp.r[joint], local))
                    }
                    _ => return true,
                };
                facing_of(pw) > -0.15
            });
        }

        // ---- hand crop observations ---------------------------------------------
        self.hand_kp_start = obs.kp2d.len();
        for hand in 0..2 {
            let Some(res) = self.last_hands[hand].as_ref() else { continue };
            let wrist_j = if hand == 0 { self.h.j.l_wrist } else { self.h.j.r_wrist };
            // The converted model's "world" output is not metric on this
            // checkpoint (index MCP reads ~3 cm from the wrist) — 2-D only.
            let _ = wrist_j;
            super::hands::hand_observations(
                &self.h,
                hand,
                res,
                None,
                width,
                height,
                &mut obs.kp2d,
                &mut obs.kp3d,
            );
        }

        // ---- point cloud ------------------------------------------------------
        if let Some(d) = depth.as_ref() {
            if d.points_m.len() == (d.width * d.height) as usize {
                let cloud = cloud_near_model(
                    &d.points_m,
                    d.width,
                    d.height,
                    &self.h.model,
                    &fk_pred,
                    &pred,
                    &intr,
                    0.12,
                    0.35,
                    self.est.params.cloud_max_points,
                );
                if cloud.points.len() >= 50 {
                    obs.cloud = Some(cloud);
                }
            }
        }

        // Ablation switches for the replay bench.
        if std::env::var_os("VULVATAR_FUSION_NO_CLOUD").is_some() {
            obs.cloud = None;
        }
        if std::env::var_os("VULVATAR_FUSION_NO_SURF").is_some() {
            obs.surface.clear();
        }
        if std::env::var_os("VULVATAR_FUSION_NO_3D").is_some() {
            obs.kp3d.clear();
        }
        if std::env::var_os("VULVATAR_FUSION_KEEP_CLOUD").is_some() {
            self.last_cloud = obs.cloud.as_ref().map(|c| c.points.clone()).unwrap_or_default();
            self.last_surface = obs.surface.iter().map(|(p, _)| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
            self.last_kp3d = obs
                .kp3d
                .iter()
                .filter_map(|k| match k.point {
                    super::estimator::ModelPoint::Joint(j) => Some((j, k.p, k.sigma)),
                    _ => None,
                })
                .collect();
        }
        // Evidence burn-in on the metric arm terms and the hand-crop terms.
        for k in obs.kp3d.iter_mut() {
            if let Some(side) = self.arm_side_of(k.point) {
                k.sigma *= inflate[side];
            }
        }
        let n_hand_start = self.hand_kp_start.min(obs.kp2d.len());
        for k in obs.kp2d[n_hand_start..].iter_mut() {
            if let Some(side) = self.arm_side_of(k.point) {
                k.sigma *= inflate[side];
            }
        }

        // ---- solve (with analytic arm re-seeds from metric joints) ------------
        {
            use super::estimator::ModelPoint;
            let find3d = |j: usize| -> Option<V3> {
                obs.kp3d
                    .iter()
                    .find(|k| matches!(k.point, ModelPoint::Joint(jj) if jj == j))
                    .map(|k| k.p)
            };
            let l = (find3d(self.h.j.l_elbow), find3d(self.h.j.l_wrist));
            let r = (find3d(self.h.j.r_elbow), find3d(self.h.j.r_wrist));
            let h = &self.h;
            let seed_l = |st: &State| -> Option<State> {
                super::seed::seed_arm(h, st, true, l.0, l.1?)
            };
            let seed_r = |st: &State| -> Option<State> {
                super::seed::seed_arm(h, st, false, r.0, r.1?)
            };
            let seed_both = |st: &State| -> Option<State> {
                let a = super::seed::seed_arm(h, st, true, l.0, l.1?)?;
                super::seed::seed_arm(h, &a, false, r.0, r.1?)
            };
            let mut seeds: Vec<&dyn Fn(&State) -> Option<State>> = Vec::new();
            if l.1.is_some() {
                seeds.push(&seed_l);
            }
            if r.1.is_some() {
                seeds.push(&seed_r);
            }
            if l.1.is_some() && r.1.is_some() {
                seeds.push(&seed_both);
            }
            let t_est = std::time::Instant::now();
            self.est.update_with_seeds(&self.h.model, &obs, &seeds);
            self.last_est_ms = t_est.elapsed().as_secs_f32() * 1000.0;
        }
        let dump_frame = std::env::var("VULVATAR_FUSION_OBSDUMP").ok().and_then(|v| v.parse::<u64>().ok());
        if dump_frame == Some(frame_index) || (dump_frame == Some(999_999) && frame_index < 3) {
            let fk = self.h.model.fk(&self.est.state);
            eprintln!("--- frame {frame_index} t={t:.3} obs dump (post-solve) root_t={:?} lost={} ---", self.est.state.root_t, self.est.lost_events);
            for k in &obs.kp2d {
                let (j, pw) = super::estimator::resolve_point(&self.h.model, &fk, k.point);
                let name = match k.point {
                    super::estimator::ModelPoint::Joint(j) => self.h.model.joints[j].name.to_string(),
                    super::estimator::ModelPoint::Site(s) => format!("site:{}", self.h.model.sites[s].name),
                    super::estimator::ModelPoint::Attached { .. } => "attached".to_string(),
                };
                let _ = j;
                let proj = intr.project(pw).unwrap_or([f64::NAN, f64::NAN]);
                eprintln!("  2d {name:>14} obs=({:.0},{:.0}) σ={:.1}  model=({:.0},{:.0}) z={:.2}", k.u, k.v, k.sigma, proj[0], proj[1], pw[2]);
            }
            for k in &obs.kp3d {
                let (_, pw) = super::estimator::resolve_point(&self.h.model, &fk, k.point);
                let name = match k.point {
                    super::estimator::ModelPoint::Joint(j) => self.h.model.joints[j].name.to_string(),
                    super::estimator::ModelPoint::Site(s) => format!("site:{}", self.h.model.sites[s].name),
                    super::estimator::ModelPoint::Attached { .. } => "attached".to_string(),
                };
                eprintln!("  3d {name:>14} obs=({:.3},{:.3},{:.3}) σ={:.3} model=({:.3},{:.3},{:.3})", k.p[0], k.p[1], k.p[2], k.sigma, pw[0], pw[1], pw[2]);
            }
            eprintln!("  surface pts {}", obs.surface.len());
            {
                let m = &self.h.model;
                let fk = m.fk(&self.est.state);
                for (pt, sg) in &obs.surface {
                    let mut best = (usize::MAX, f64::INFINITY);
                    for (ci, c) in m.capsules.iter().enumerate() {
                        let (q, _u) = super::estimator::closest_on_segment(fk.point(c.a), fk.point(c.b), *pt);
                        let d = norm(sub(*pt, q)) - m.capsule_radius(&self.est.state, c);
                        if d.abs() < best.1.abs() {
                            best = (ci, d);
                        }
                    }
                    let name = if best.0 == usize::MAX { "none".to_string() } else {
                        let c = &m.capsules[best.0];
                        let pa = |p: PointRef| match p { PointRef::Joint(j) => m.joints[j].name.to_string(), PointRef::Site(s) => m.sites[s].name.to_string() };
                        format!("{:?}:{}-{}", c.part, pa(c.a), pa(c.b))
                    };
                    eprintln!("    surf ({:.3},{:.3},{:.3}) σ={:.3} → {name} d={:+.3}", pt[0], pt[1], pt[2], sg, best.1);
                }
            }
            eprintln!("  depth frame: {:?} valid@nose {:?}", depth.as_ref().map(|d| (d.width, d.height, d.points_m.len())),
                obs.kp2d.first().and_then(|k| depth.as_ref().and_then(|d| window_point(&d.points_m, d.width, d.height, k.u, k.v, 3, 0.1, 10.0))));
            eprintln!("  diag {:?}", self.est.diag);
        }
        self.last_t = Some(t);
        self.frames += 1;

        // ---- learn face shapes from the posterior ------------------------------
        {
            let fk = self.h.model.fk(&self.est.state);
            let head_sigma = self.est.joint_sigma(&self.h.model, head_j);
            let neck_sigma = self.est.joint_sigma(&self.h.model, self.h.j.neck);
            if head_sigma < 0.35 && neck_sigma < 0.4 && depth.is_some() {
                let head_center = fk.site[self.h.s.head_center];
                let z_ref = head_center[2];
                let depth_at2 = |u: f64, v: f64| -> Option<V3> {
                    let d = depth.as_ref()?;
                    if in_hand_rect(u, v) {
                        return None;
                    }
                    window_point(
                        &d.points_m,
                        d.width,
                        d.height,
                        u,
                        v,
                        1,
                        (z_ref - 0.25) as f32,
                        (z_ref + 0.25) as f32,
                    )
                };
                self.face68
                    .learn(&fk.r[head_j], fk.t[head_j], &face68_valid, &depth_at2, z_ref);
                if let Some(lm) = mesh_px {
                    self.mesh
                        .learn(&fk.r[head_j], fk.t[head_j], lm, &depth_at2, z_ref);
                }
            }
        }

        // ---- output ----------------------------------------------------------------
        let span_px = {
            let fk = self.h.model.fk(&self.est.state);
            match (intr.project(fk.t[self.h.j.l_shoulder]), intr.project(fk.t[self.h.j.r_shoulder])) {
                (Some(a), Some(b)) => Some(((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()),
                _ => None,
            }
        };
        let rig = output::rig_pose(&self.h, &self.est, t, span_px);
        let ref_span = {
            let fk = self.h.model.fk(&self.est.state);
            norm(sub(fk.t[self.h.j.l_shoulder], fk.t[self.h.j.r_shoulder])) as f32
        };
        let mut skeleton = output::source_skeleton(&self.h, &self.est, frame_index, intr_cam, ref_span);
        // Carry the perception-side channels the expression solver and GUI
        // still read from the base pipeline.
        skeleton.expressions = base.skeleton.expressions;
        skeleton.face_mesh_confidence = base.skeleton.face_mesh_confidence;
        skeleton.face = base.skeleton.face;
        skeleton.face_body_raw = base.skeleton.face_body_raw;
        skeleton.capture_timestamp_ms = ts_ms;
        skeleton.rig = Some(Arc::new(rig));
        self.last_solve_ms = t0.elapsed().as_secs_f32() * 1000.0;
        if crate::tracking::debug_channel::enabled() {
            let d = self.est.diag;
            crate::tracking::debug_channel::stash_rig_diag(serde_json::json!({
                "solve_ms": self.last_solve_ms,
                "est_ms": self.last_est_ms,
                "iters": d.iters,
                "cost": d.cost_final,
                "n2d": d.n_kp2d,
                "n3d": d.n_kp3d,
                "med_2d_px": d.med_2d_px,
                "lost_events": self.est.lost_events,
                "seed_wins": d.seed_wins,
                "cov_failures": d.cov_failures,
                "hand_crops": [self.last_hands[0].is_some(), self.last_hands[1].is_some()],
                "face68_learned": self.face68.learned_count(),
                "mesh_learned": self.mesh.learned_count(),
                "shape_frozen": self.est.shape_frozen,
            }));
        }

        PoseEstimate {
            skeleton,
            annotation: base.annotation,
        }
    }
}
