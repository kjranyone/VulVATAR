//! `FusionProvider` — the tracking-v2 production pipeline behind
//! [`PoseProvider`]: RTMW3D (2-D keypoints + SimCC σ, FaceMesh landmarks
//! and blendshapes) + the D435 point cloud → the fusion estimator →
//! [`RigPose`] (published on the compatibility `SourceSkeleton`).

use std::path::Path;
use std::sync::Arc;

use log::info;

use crate::tracking::provider::{PoseProvider, TrackingPipelineConfig};
use crate::tracking::rtmw3d::{Rtmw3dInference, Rtmw3dOptions};
use crate::tracking::skeleton_from_depth::MetricDepthFrame;
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
        let est = Estimator::new(&h.model, Params::default());
        let body_map = BodyMap::new(&h);
        Ok(Self {
            rtmw3d,
            hands,
            last_hands: [None, None],
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

    fn set_calibration(&mut self, _calibration: Option<crate::tracking::PoseCalibration>) {}

    fn reset_temporal_state(&mut self) {
        self.rtmw3d.reset_temporal_state();
        self.est.reset(&self.h.model);
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

        // Depth accessor over the aligned point cloud (person band around
        // the predicted head for face landmarks).
        let depth_at = |u: f64, v: f64| -> Option<V3> {
            let d = depth.as_ref()?;
            if d.crop.is_some() {
                return None;
            }
            window_point(
                &d.points_m,
                d.width,
                d.height,
                u,
                v,
                1,
                (head_center_pred[2] - 0.25) as f32,
                (head_center_pred[2] + 0.25) as f32,
            )
        };

        // ---- hand-block L/R assignment --------------------------------------
        // The detector's left/right hand blocks can be transposed when the
        // hands cross or touch. Decide the assignment against the predicted
        // wrists (only when both wrists are currently tracked): keep the
        // labelling unless swapping is clearly better.
        let mut det_kps: Vec<(f32, f32, f32)> = base.annotation.keypoints.clone();
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
                // Crop source: the body detector's hand block when it is
                // confident (fresh every frame, independent of the estimator
                // state), else the model prediction (bridges detector misses
                // while the hand is tracked).
                let crop = super::hands::detector_hand_crop(&det_kps, hand, width, height, 0.35, 96.0)
                    .or_else(|| super::hands::predicted_hand_crop(&self.h, &fk_pred, hand, &intr, 96.0));
                let Some(crop) = crop else { continue };
                if let Some(res) = hl.estimate(rgb_data, width, height, crop) {
                    if res.presence >= 0.5 {
                        hand_done[hand] = true;
                        self.last_hands[hand] = Some(res);
                    }
                }
            }
        }

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
            if hands_swapped {
                for k in 0..21 {
                    raw.swap(91 + k, 112 + k);
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
            body_kp2d(
                &self.body_map,
                &raw,
                width,
                height,
                self.kp_sigma,
                1.5,
                &mut obs.kp2d,
            );
            // Depth-lifted joints: the absolute-depth anchor that resolves
            // the projective scale/distance ambiguity of the 2-D terms.
            if let Some(d) = depth.as_ref() {
                if d.crop.is_none() && d.points_m.len() == (d.width * d.height) as usize {
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

        // ---- hand crop observations ---------------------------------------------
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
            if d.crop.is_none() && d.points_m.len() == (d.width * d.height) as usize {
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
            eprintln!("  depth frame: {:?} valid@nose {:?}", depth.as_ref().map(|d| (d.width, d.height, d.points_m.len(), d.crop.is_some())),
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
        let rig = output::rig_pose(&self.h, &self.est, t);
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

        PoseEstimate {
            skeleton,
            annotation: base.annotation,
        }
    }
}
