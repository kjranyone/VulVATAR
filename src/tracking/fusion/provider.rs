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
    /// Per-user canonical-face fit: uniform scale over the canonical
    /// template and its offset from the head joint, learned by EMA from
    /// depth-lifted mesh landmarks.
    face_fit: super::canonical_face::FaceFit,
    head_ori: super::head_ori::HeadOriTracker,
    /// Consecutive frames each locked hand crop has gone WITHOUT
    /// corroboration from the body detector (wrist / hand-block keypoints
    /// near the crop). The dedicated hand landmarker happily reports
    /// presence ≈ 0.9 for a hand-sized crop of a FACE, and the
    /// re-crop-around-last-landmarks candidate then feeds itself forever —
    /// a desk session with both hands under the desk showed a phantom hand
    /// parked on the face for minutes. The body model is immune to that
    /// (it sees the whole person), so a lock that the body cannot back up
    /// for a few frames is dropped.
    hand_unsupported: [u8; 2],
    /// This frame's crop result for each slot carries the OTHER hand's
    /// handedness (see the veto note at the crop loop). Dropping such a
    /// result outright makes the arm flicker between the data pose and
    /// the prior (measured: left-wrist snaps 14 → 40 on a live clasped
    /// hands session), so it is kept at inflated σ instead.
    hand_suspect: [bool; 2],
    /// Frames in which both crops locked onto one physical hand.
    pub hand_dupes: u32,
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
    /// Diagnostics: sparse surface points.
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
        let params = Params::default();
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
            face_fit: super::canonical_face::FaceFit::new(),
            head_ori: super::head_ori::HeadOriTracker::new(),
            hand_unsupported: [0, 0],
            hand_suspect: [false, false],
            hand_dupes: 0,
            external_depth: None,
            load_warnings: warnings,
            last_t: None,
            frames: 0,
            kp_sigma: KpSigma::default(),
            hand_swaps: 0,
            hand_kp_start: 0,
            last_solve_ms: 0.0,
            last_est_ms: 0.0,
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
    pub fn mesh_learned(&self) -> usize {
        self.face_fit.n as usize
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
        self.face_fit.reset();
        self.head_ori.reset();
        self.hand_unsupported = [0, 0];
        self.hand_suspect = [false, false];
        self.hand_dupes = 0;
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
            ori: Vec::new(),
            shoulder_yaw: None,
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
        let occl = self.head_ori.check_occlusion(
            &base.annotation.keypoints,
            width,
            height,
            &hand_rects,
            &self.last_hands,
        );
        let face_occluded = occl.face_occluded;
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
                (head_center_pred[2] - 0.20) as f32,
                (head_center_pred[2] + 0.20) as f32,
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
                        // Handedness veto: a slot must never lock onto the opposing hand
                        // when the classifier is decisive (left ≈0.15, right ≈0.85).
                        // Middle band [0.30, 0.70] is genuinely uncertain and passes.
                        let wrong_hand = if hand == 0 {
                            res.handedness > 0.70
                        } else {
                            res.handedness < 0.30
                        };
                        if wrong_hand {
                            continue;
                        }
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
                // Handedness veto: with the hands clasped or overlapping,
                // a crop aimed at one wrist routinely locks onto the OTHER
                // hand, and the landmarker says so — measured on a live
                // clasped-hands session, 46% of the left slot's results
                // carried right-hand handedness, and each flip moved the
                // wrist 0.15–0.35 m (the two hands are that far apart).
                // The convention is fixed by this pipeline's mirror: the
                // left slot reads ≈0.15, the right ≈0.85 (measured on a
                // clip with one raised, unambiguous hand). Scores inside
                // the middle band are genuinely uncertain and pass.

                // Body-detector corroboration (see `hand_unsupported`).
                if let Some(res) = best.as_ref().filter(|r| r.presence >= 0.5) {
                    let (cx, cy, sz) = res.crop;
                    let (ccx, ccy) = (cx + 0.5 * sz, cy + 0.5 * sz);
                    let reach = (1.5 * sz).max(120.0);
                    let near = |nx: f32, ny: f32| -> bool {
                        let (u, v) = (nx * width as f32, ny * height as f32);
                        (u - ccx).abs() < reach && (v - ccy).abs() < reach
                    };
                    let wrist = det_kps[9 + hand];
                    let mut supported = wrist.2 >= 0.3 && near(wrist.0, wrist.1);
                    if !supported {
                        let base = if hand == 0 { 91 } else { 112 };
                        let n_near = (base..(base + 21).min(det_kps.len()))
                            .filter(|&i| det_kps[i].2 >= 0.35 && near(det_kps[i].0, det_kps[i].1))
                            .count();
                        supported = n_near >= 6;
                    }
                    if supported {
                        self.hand_unsupported[hand] = 0;
                    } else {
                        self.hand_unsupported[hand] = self.hand_unsupported[hand].saturating_add(1);
                    }
                    if self.hand_unsupported[hand] >= 3 {
                        best = None;
                    }
                } else {
                    self.hand_unsupported[hand] = 0;
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
            // Duplicate lock: with the hands clasped or crossing, both
            // crops routinely land on the SAME physical hand, and the two
            // slots then flip between the user's real hands frame to
            // frame (measured: 0.15–0.35 m wrist steps on a live clasped
            // hands session). When both results sit on top of each other,
            // keep the one whose handedness matches its slot — this
            // pipeline's mirror puts the left slot at ≈0.15 and the right
            // at ≈0.85, measured on a clip with one unambiguous hand —
            // and let the other arm ride its prior for the frame rather
            // than be driven by its twin.
            if std::env::var_os("VULVATAR_NO_HANDEDNESS").is_none() {
                if let (Some(l), Some(r)) = (&self.last_hands[0], &self.last_hands[1]) {
                    let d = ((l.px[0][0] - r.px[0][0]).powi(2) + (l.px[0][1] - r.px[0][1]).powi(2))
                        .sqrt();
                    if d < 0.06 * width as f32 {
                        // Vote: distance of each result's handedness from
                        // its slot's expected pole.
                        let l_fit = 1.0 - l.handedness;
                        let r_fit = r.handedness;
                        self.hand_dupes += 1;
                        let loser = if l_fit >= r_fit { 1 } else { 0 };
                        self.hand_suspect = if loser == 1 {
                            [false, true]
                        } else {
                            [true, false]
                        };
                        // Clear prev_hands for the loser so it stops cropping the twin's position
                        self.prev_hands[loser] = None;
                    }
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
                    // A hip is only in-frame when well clear of borders and anatomically below the shoulders.
                    // If the hip is at or below ny 0.82, the thighs (≈0.4 m) physically extend beyond
                    // the bottom frame border, making any in-frame knee detection a 100% hallucination.
                    let hip_ok = hip.score >= 0.5
                        && hip.nx > 0.04
                        && hip.nx < 0.96
                        && hip.ny > 0.04
                        && hip.ny < 0.82
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
            // Leg-chain coherence: a hip that passes the gate can still be
            // a bottom-edge clamp while the detector paints its knee /
            // ankle / toe on the background or on the user's own raised
            // arms — measured on a chest-up session: l_hip in the last
            // frame rows (ny 0.97), l_knee at head height, a projective
            // 0.7–0.9 m "thigh" no human has, driving 11 knee snaps of
            // 0.15–0.5 m. The arms have `reach_filter` for exactly this;
            // Anthropometric & kinematic coherence sanity filters (see `super::coherence`):
            // Physical perspective projection cannot exceed true bone length. Detections
            // violating reach, pelvis width, height order, or edge borders are culled.
            if let Some(d) = depth.as_ref() {
                if d.points_m.len() == (d.width * d.height) as usize {
                    let z_person = head_center_pred[2];
                    let model_pelvis_w = norm(sub(
                        fk_pred.t[self.h.j.l_hip],
                        fk_pred.t[self.h.j.r_hip],
                    ));
                    super::coherence::filter_leg_coherence(
                        &mut raw,
                        &d.points_m,
                        d.width,
                        d.height,
                        intr.fx,
                        z_person,
                        model_pelvis_w,
                    );
                    super::coherence::filter_arm_coherence(
                        &mut raw,
                        &d.points_m,
                        d.width,
                        d.height,
                        intr.fx,
                        z_person,
                    );
                }
            }
            super::coherence::filter_duplicate_wrists(&mut raw, width, height);
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
            // With a dense-mesh centroid anchoring head position, the
            // SimCC face keypoints only ADD their frontalization bias —
            // widen them so they stop binding head yaw (~4° measured).
            let mesh_conf = aux
                .face_mesh
                .as_ref()
                .map(|(_, c)| *c)
                .unwrap_or(0.0);
            let head_scale = if !face_occluded && mesh_conf >= 0.35 {
                1.0 + 2.0 * ((mesh_conf - 0.35) / 0.25).clamp(0.0, 1.0) as f64
            } else {
                1.0
            };
            body_kp2d(
                &self.body_map,
                &raw,
                width,
                height,
                self.kp_sigma,
                1.5,
                head_scale,
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
                        &in_hand_rect,
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
                    // Torso yaw from the chest depth slope — the only
                    // depth channel that carries it (see
                    // `ShoulderYawObs`). Hand crops and predicted arm
                    // capsules are excluded so a forearm across the chest
                    // cannot tilt the fit.
                    if std::env::var_os("VULVATAR_FUSION_NO_CHESTYAW").is_none() {
                        // Only hand crops are vetoed per pixel. Testing
                        // the predicted ARM capsules here was measured and
                        // is wrong: the upper-arm capsule starts AT the
                        // shoulder joint, so every sample near a shoulder
                        // counts as "behind an arm" — 95 of 105 samples
                        // were rejected on a live session and the
                        // observation reached 5% of frames. Arm surfaces
                        // that really do cross the strip are removed by
                        // the per-column depth-outlier filter inside
                        // `chest_yaw_from_depth`, which needs no
                        // prediction to be right.
                        // Only hand crops are vetoed per pixel. Testing
                        // the predicted ARM capsules here was measured and
                        // is wrong: the upper-arm capsule starts AT the
                        // shoulder joint, so every sample near a shoulder
                        // counted as "behind an arm" — 95 of 105 samples
                        // rejected on a live session, and the observation
                        // reached only 5% of frames. Arm surfaces that
                        // really do cross the strip are removed by the
                        // per-column depth-outlier filter inside
                        // `chest_yaw_from_depth`, which needs no
                        // prediction to be right.
                        let occl = |u: f64, v: f64| -> bool { in_hand_rect(u, v) };
                        let cy = super::observe::chest_yaw_from_depth(
                            &raw,
                            &d.points_m,
                            d.width,
                            d.height,
                            z_ref,
                            &occl,
                        );
                        if cy.is_none() && std::env::var_os("VULVATAR_CHEST_DUMP").is_some() {
                            eprintln!("CHEST none");
                        }
                        if let Some((yaw, n)) = cy {
                            if std::env::var_os("VULVATAR_CHEST_DUMP").is_some() {
                                eprintln!("CHEST fired yaw {:.3} n {}", yaw, n);
                            }
                            obs.shoulder_yaw = Some(super::estimator::ShoulderYawObs {
                                left: self.h.j.l_shoulder,
                                right: self.h.j.r_shoulder,
                                yaw,
                                // ~7° at a full 15-column fit, widening as
                                // columns drop out.
                                sigma: 0.12 * (15.0 / n as f64).sqrt(),
                            });
                        }
                    }
                }
            }
        }
        let mesh_conf = aux
            .as_ref()
            .and_then(|a| a.face_mesh.as_ref())
            .map(|(_, conf)| *conf)
            .unwrap_or(0.0);
        let mesh_px: Option<&Vec<[f32; 3]>> = aux
            .as_ref()
            .and_then(|a| a.face_mesh.as_ref())
            .filter(|(_, conf)| *conf >= 0.35)
            .map(|(lm, _)| lm);

        // Canonical-face observations.
        {
            let head_sigma_pred = self.est.joint_sigma(&self.h.model, head_j);
            let sigma_scale = if head_sigma_pred < 0.2 { 1.0 } else { 2.0 };
            if let Some(lm) = mesh_px.filter(|_| !face_occluded) {
                // Canonical-face observations: every FaceMesh landmark is a
                // point rigidly attached to the head at
                // `scale · canonical + offset` — full-strength orientation
                // evidence with zero rotational gauge freedom.
                let s_fit = self.face_fit.scale;
                let t_fit = self.face_fit.offset;
                // Head POSITION anchor: ONE observation at the landmark
                // centroid. Individual landmark positions systematically
                // frontalize under yaw (the dense mesh compresses toward a
                // frontal arrangement past ~30°), so per-point 2-D
                // residuals out-vote every honest orientation source ~40:1
                // and pin the head near-frontal. The centroid keeps the
                // positional accuracy (the bias averages out to first
                // order) while carrying no orientation information —
                // rotation is owned by the mesh-pose OriObs and the depth
                // z-profile of the 3-D points below.
                {
                    let mut mu = [0.0f64; 2];
                    let mut mc = [0.0f64; 3];
                    let mut n = 0.0f64;
                    for (i, l) in lm.iter().enumerate().take(468) {
                        if !l[0].is_finite() || !l[1].is_finite() {
                            continue;
                        }
                        let c = super::canonical_face::CANONICAL_FACE_468[i];
                        mu[0] += l[0] as f64;
                        mu[1] += l[1] as f64;
                        mc = add(mc, [c[0] as f64, c[1] as f64, c[2] as f64]);
                        n += 1.0;
                    }
                    if n >= 100.0 {
                        let local = add(scale(scale(mc, 1.0 / n), s_fit), t_fit);
                        obs.kp2d.push(super::estimator::Kp2d {
                            point: super::estimator::ModelPoint::Attached { joint: head_j, local },
                            u: mu[0] / n,
                            v: mu[1] / n,
                            sigma: 1.5 * sigma_scale,
                        });
                    }
                }
                // Predicted facing of each landmark (canonical normal ≈
                // radial direction from the face centroid, rotated by the
                // predicted head): 3-D depth is only trusted on the front
                // hemisphere — at a 3/4 view the mesh imagines the far-side
                // landmarks near the visible silhouette, and the depth
                // there is the NEAR cheek's, which pulls the head frontal.
                let head_r_pred = fk_pred.r[head_j];
                let head_t_pred = fk_pred.t[head_j];
                for (i, l) in lm.iter().enumerate().take(468) {
                    if !l[0].is_finite() || !l[1].is_finite() {
                        continue;
                    }
                    let c = super::canonical_face::CANONICAL_FACE_468[i];
                    let local = add(
                        scale([c[0] as f64, c[1] as f64, c[2] as f64], s_fit),
                        t_fit,
                    );
                    let pw = add(head_t_pred, mat_vec(&head_r_pred, local));
                    let n_local = normalize([c[0] as f64, c[1] as f64, (c[2] as f64) - 0.02]);
                    let n_world = mat_vec(&head_r_pred, n_local);
                    let facing = dot(n_world, normalize(scale(pw, -1.0)));
                    if facing < -0.1 {
                        continue;
                    }
                    let point = super::estimator::ModelPoint::Attached { joint: head_j, local };
                    let oval = super::canonical_face::FACE_OVAL.contains(&i);
                    if i % 4 == 0 && facing > 0.15 && !oval {
                        if let Some(p) = depth_at(l[0] as f64, l[1] as f64) {
                            if (p[2] - head_center_pred[2]).abs() < 0.15 {
                                let sig = (0.015 / (mesh_conf as f64).clamp(0.35, 1.0))
                                    / (facing as f64).clamp(0.2, 1.0);
                                obs.kp3d.push(super::estimator::Kp3d {
                                    point,
                                    p,
                                    sigma: sig,
                                    // Honest z, frontalized lateral (see
                                    // Kp3d::lat_scale).
                                    lat_scale: 6.0,
                                });
                            }
                        }
                    }
                }
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
        if std::env::var_os("VULVATAR_ABL_NOCULL").is_none() {
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
                    // Site-based head points (nose / eyes / ears lifted by
                    // `body_kp3d`) need the same far-side cull as the 2-D
                    // ones: the detector places the hidden ear on the
                    // silhouette, where the depth belongs to the NEAR side
                    // of the head.
                    ModelPoint::Site(sid) if head_sites.contains(&sid) => fkp.site[sid],
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
                if self.hand_suspect[hand] { 4.0 } else { 1.0 },
                &mut obs.kp2d,
                &mut obs.kp3d,
            );
        }

        // ---- head orientation from the dense FaceMesh pose --------------------
        // The SimCC face-68 landmarks hallucinate a near-frontal arrangement
        // at 3/4+ turns (and the self-learned rigid face cannot anchor
        // absolute orientation), so landmark terms alone under-rotate the
        // head. FaceMesh's dense-landmark pose derivation tracks yaw
        // to ~±60° with high confidence — feed it as a direct world-
        if let Some(ori) = self.head_ori.estimate_orientation(
            &base,
            &occl,
            depth.as_ref(),
            head_center_pred,
            head_depth_pred,
            &fk_pred.r[self.h.j.head],
            self.h.j.head,
            t,
            self.last_t,
            width,
            height,
            &intr,
            &arm_capsules,
            &hand_rects,
        ) {
            obs.ori.push(ori);
        }

        // Ablation switches for the replay bench.
        if std::env::var_os("VULVATAR_FUSION_NO_SURF").is_some() {
            obs.surface.clear();
        }
        if std::env::var_os("VULVATAR_FUSION_NO_3D").is_some() {
            obs.kp3d.clear();
        }
        if std::env::var_os("VULVATAR_FUSION_KEEP_CLOUD").is_some() {
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
        let t_est = std::time::Instant::now();
        super::seed::update_with_arm_seeds(&self.h, &mut self.est, &obs);
        self.last_est_ms = t_est.elapsed().as_secs_f32() * 1000.0;

        dump_obs_post_solve(&self.h, &self.est, frame_index, t, &obs, &intr, depth.as_ref());
        self.last_t = Some(t);
        self.frames += 1;

        // ---- learn face shapes from the posterior ------------------------------
        {
            let fk = self.h.model.fk(&self.est.state);
            let head_sigma = self.est.joint_sigma(&self.h.model, head_j);
            let neck_sigma = self.est.joint_sigma(&self.h.model, self.h.j.neck);
            if head_sigma < 0.35 && neck_sigma < 0.4 {
                if let (Some(d), Some(lm)) = (depth.as_ref(), mesh_px) {
                    let z_ref = fk.site[self.h.s.head_center][2];
                    self.face_fit.update(
                        lm,
                        fk.t[head_j],
                        &fk.r[head_j],
                        z_ref,
                        d,
                        &in_hand_rect,
                    );
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
                "face68_learned": 0,
                "mesh_learned": self.face_fit.n,
                "shape_frozen": self.est.shape_frozen,
            }));
        }

        PoseEstimate {
            skeleton,
            annotation: base.annotation,
        }
    }
}

fn dump_obs_post_solve(
    h: &Humanoid,
    est: &Estimator,
    frame_index: u64,
    t: f64,
    obs: &super::estimator::FrameObs,
    intr: &Intrinsics,
    depth: Option<&MetricDepthFrame>,
) {
    let dump_frame = std::env::var("VULVATAR_FUSION_OBSDUMP")
        .ok()
        .and_then(|v| v.parse::<u64>().ok());
    if dump_frame == Some(frame_index) || (dump_frame == Some(999_999) && frame_index < 3) {
        let fk = h.model.fk(&est.state);
        eprintln!(
            "--- frame {frame_index} t={t:.3} obs dump (post-solve) root_t={:?} lost={} ---",
            est.state.root_t, est.lost_events
        );
        for k in &obs.kp2d {
            let (_j, pw) = super::estimator::resolve_point(&h.model, &fk, k.point);
            let name = match k.point {
                super::estimator::ModelPoint::Joint(j) => {
                    h.model.joints[j].name.to_string()
                }
                super::estimator::ModelPoint::Site(s) => {
                    format!("site:{}", h.model.sites[s].name)
                }
                super::estimator::ModelPoint::Attached { .. } => "attached".to_string(),
            };
            let proj = intr.project(pw).unwrap_or([f64::NAN, f64::NAN]);
            eprintln!(
                "  2d {name:>14} obs=({:.0},{:.0}) σ={:.1}  model=({:.0},{:.0}) z={:.2}",
                k.u, k.v, k.sigma, proj[0], proj[1], pw[2]
            );
        }
        for k in &obs.kp3d {
            let (_, pw) = super::estimator::resolve_point(&h.model, &fk, k.point);
            let name = match k.point {
                super::estimator::ModelPoint::Joint(j) => {
                    h.model.joints[j].name.to_string()
                }
                super::estimator::ModelPoint::Site(s) => {
                    format!("site:{}", h.model.sites[s].name)
                }
                super::estimator::ModelPoint::Attached { .. } => "attached".to_string(),
            };
            eprintln!(
                "  3d {name:>14} obs=({:.3},{:.3},{:.3}) σ={:.3} model=({:.3},{:.3},{:.3})",
                k.p[0], k.p[1], k.p[2], k.sigma, pw[0], pw[1], pw[2]
            );
        }
        eprintln!("  surface pts {}", obs.surface.len());
        {
            let m = &h.model;
            let fk = m.fk(&est.state);
            for (pt, sg) in &obs.surface {
                let mut best = (usize::MAX, f64::INFINITY);
                for (ci, c) in m.capsules.iter().enumerate() {
                    let (q, _u) = super::estimator::closest_on_segment(
                        fk.point(c.a),
                        fk.point(c.b),
                        *pt,
                    );
                    let d = norm(sub(*pt, q)) - m.capsule_radius(&est.state, c);
                    if d.abs() < best.1.abs() {
                        best = (ci, d);
                    }
                }
                let name = if best.0 == usize::MAX {
                    "none".to_string()
                } else {
                    let c = &m.capsules[best.0];
                    let pa = |p: PointRef| match p {
                        PointRef::Joint(j) => m.joints[j].name.to_string(),
                        PointRef::Site(s) => m.sites[s].name.to_string(),
                    };
                    format!("{:?}:{}-{}", c.part, pa(c.a), pa(c.b))
                };
                eprintln!(
                    "    surf ({:.3},{:.3},{:.3}) σ={:.3} → {name} d={:+.3}",
                    pt[0], pt[1], pt[2], sg, best.1
                );
            }
        }
        eprintln!(
            "  depth frame: {:?} valid@nose {:?}",
            depth.as_ref().map(|d| (d.width, d.height, d.points_m.len())),
            obs.kp2d.first().and_then(|k| depth.as_ref().and_then(|d| window_point(
                &d.points_m,
                d.width,
                d.height,
                k.u,
                k.v,
                3,
                0.1,
                10.0
            )))
        );
        eprintln!("  diag {:?}", est.diag);
    }
}
