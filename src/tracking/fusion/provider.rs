//! `FusionProvider` — the tracking-v2 production pipeline behind
//! [`PoseProvider`]: RTMW3D (2-D keypoints + SimCC σ, FaceMesh landmarks
//! and blendshapes) + the D435 point cloud → the fusion estimator →
//! [`RigPose`] (published on the compatibility `SourceSkeleton`).

use std::path::Path;
use std::sync::Arc;

use log::info;

use crate::tracking::metric_frame::MetricDepthFrame;
use crate::tracking::provider::{PoseProvider, TrackingPipelineConfig};
use crate::tracking::detector::yolo11::Yolo11PoseInference;
use crate::tracking::detector::DetectorOptions;
use crate::tracking::source_skeleton::CameraIntrinsics;
use crate::tracking::{DetectionAnnotation, PoseEstimate, SourceSkeleton};

use super::detector_thread;
use super::estimator::{Estimator, FrameObs, Intrinsics, Params};
use super::math::*;
use super::model::*;
use super::observe::*;
use super::output;
use super::visibility::{build_silhouette, window_median_z, SilhouetteParams, VisPolicy};

/// Per-phase wall-time breakdown (ms) of one `estimate_pose` call, from
/// `Instant` pairs around each section. `solve_ms` / `est_ms` are the
/// coarse totals; these localise the rest. Estimator-internal fields are
/// copies of [`super::estimator::SolveTimings`]. Exposed for the live
/// debug channel and the replay harness — the tracking-v2 counterpart
/// of the render pipeline's `PROFILE` log.
#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseTimings {
    /// Crop hint derivation from the predicted state.
    pub hint_ms: f32,
    /// RTMW3D detector pass: YOLOX wait + crop/preprocess + ONNX run +
    /// decode + FaceMesh/blendshapes (the `yolox`…`face` stage-log span).
    pub rtmw_ms: f32,
    /// Keypoint visibility policy (SimCC p_vis + silhouette test).
    pub vis_ms: f32,
    /// Hand-block assignment + hand crops + per-arm burn-in prep.
    pub hands_ms: f32,
    /// Hand-crop observations + head orientation from FaceMesh.
    pub head_ms: f32,
    /// Dense surface sampling + observation σ prep.
    pub dense_ms: f32,
    /// Canonical face-shape learning from the posterior.
    pub facefit_ms: f32,
    /// Rig pose + source skeleton construction.
    pub output_ms: f32,
    /// Estimator: residual/normal-equation accumulation passes.
    pub acc_ms: f32,
    /// Estimator: cost-only evaluations (LM accept test).
    pub eval_ms: f32,
    /// Estimator: damped LDLᵀ solves.
    pub lin_ms: f32,
    /// Estimator: main LM loop total.
    pub main_ms: f32,
    /// Estimator: seed-contest LM loops total.
    pub seeds_ms: f32,
    /// Estimator: winner re-accumulate.
    pub reacc_ms: f32,
    /// Estimator: `finish()` (covariance inverse + σ propagation).
    pub finish_ms: f32,
}

/// Max buffered metric depths awaiting their detector result (Remote
/// mode). Steady state holds 1–2; the cap only matters while the
/// detector is stalled or dead and nothing consumes entries.
const DEPTH_RING_CAP: usize = 8;

pub struct FusionProvider {
    det: DetectorSlot,
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
    /// Per-phase wall-time breakdown (ms) of the last `estimate_pose`.
    pub last_timings: PhaseTimings,
    /// Diagnostics: sparse surface points.
    pub last_surface: Vec<[f32; 3]>,
    /// Diagnostics: metric joint observations `(joint index, point, σ)`.
    pub last_kp3d: Vec<(usize, V3, f64)>,
    /// Diagnostics: the detector's raw 133 keypoints of the last frame
    /// (whole-frame normalised, before any gate) with SimCC peak stats.
    pub last_raw_joints: Vec<crate::tracking::detector::DecodedJoint>,
    /// Diagnostics: person crop `(x, y, w, h)` fed to the detector.
    pub last_crop: Option<(f32, f32, f32, f32)>,
    /// Diagnostics: per-gate score snapshots of the last frame —
    /// `(gate name, 133 scores after that gate)` in pipeline order.
    pub last_kp_stages: Vec<(&'static str, Vec<f32>)>,
    /// Keypoint visibility calibration (SimCC peak shape → probability).
    pub vis_policy: VisPolicy,
    /// Depth silhouette parameters.
    pub sil_params: SilhouetteParams,
    /// Diagnostics: per keypoint `(SimCC p_vis before the silhouette test,
    /// distance to silhouette m)` of the last frame (distance NaN without
    /// a silhouette). The final verdict is the keypoint's `score`.
    pub last_vis: Vec<(f32, f32)>,
    /// Diagnostics: last silhouette `(z_ref m, touches bottom, visible height m)`.
    pub last_silhouette: Option<(f64, bool, f64)>,
    /// Frames since the detector last showed ≥ 2 visible face keypoints
    /// (`u32::MAX` before the first).
    frames_since_face: u32,
    /// Diagnostics: crop hint pushed to the detector this frame (px).
    /// This frame's person silhouette (dense surface source, GUI overlay).
    pub last_sil: Option<super::visibility::Silhouette>,
    /// Dense surface points fed to the estimator this frame.
    pub last_dense_n: usize,
    /// Depth-confirmed wrist hold, `[left, right]`: when the 2-D detector
    /// drops a wrist it had been observing (frame border, desk clutter),
    /// the last observation is kept alive for as long as the RealSense
    /// depth still shows a surface at the predicted wrist pixel within
    /// `Z_GATE` of the last measured depth. The sensor outlives the
    /// detector's flicker, so a hand parked on the desk keeps its arm
    /// instead of falling to the relaxed-pose prior and snapping back on
    /// re-detection. Unlike the blanket `q_hold_floor` process-noise hold
    /// (measured worse: a held arm fights its returning observation and
    /// the trunk pays), this releases the moment the depth under the
    /// prediction changes — and its σ grows with time since the last real
    /// detection, so a re-acquired 2-D/3-D keypoint always outweighs it.
    wrist_hold: [WristHold; 2],
    /// In Remote (pipelined) mode, metric depths waiting for their
    /// detector result: the result consumed at call N is usually frame
    /// N−1's, so each frame's depth — handed in via `set_external_depth`
    /// before its `estimate_pose` call — is buffered under its capture
    /// index until the matching result is consumed. Cleared on temporal
    /// resets. (MetricDepthFrame is a few hundred KB; the ring stays at
    /// one or two entries in steady state.)
    depth_ring: Vec<(u64, MetricDepthFrame)>,
    /// Highest consumed detector frame index (Remote mode). `None` until
    /// the first consume so frame 0 passes the freshness test.
    last_consumed: Option<u64>,
    /// Logged once when the detector thread is found dead.
    det_dead_logged: bool,
}

/// Where the RTMW3D detector pass runs.
enum DetectorSlot {
    /// Inline inside `estimate_pose` — offline harnesses (replay /
    /// bench / validate_gt), safe mode, active session recording, and
    /// `VULVATAR_NO_PIPELINE=1`. Bit-identical to the pre-pipelining
    /// behaviour.
    Inline(Yolo11PoseInference),
    /// The detector runs on the `tracking-detect` thread
    /// ([`detector_thread::DetectorClient`]) so it overlaps the solver
    /// stage; `estimate_pose_latest` is the live entry point.
    Remote {
        client: detector_thread::DetectorClient,
        backend_label: String,
    },
}

/// State carried between frames for one wrist's depth-confirmed hold.
#[derive(Clone, Copy, Default)]
struct WristHold {
    /// Capture time (s) of the last real wrist observation (2-D or 3-D).
    last_t: Option<f64>,
    /// Last sensor-confirmed surface depth under the wrist (m, camera z).
    last_z: Option<f64>,
}

/// σ ceiling `[left, right]` for the depth-lifted ELBOW from forearm bone
/// rigidity: if the elbow lift and the same side's wrist lift disagree on
/// the obs-obs forearm length (rigid, shape-fitted — model state plays no
/// part), at least one of them is not on the arm, and the hand block is
/// the better-evidenced end (a crop with 25 corroborating keypoints vs a
/// single body keypoint). This is the one mis-detection that survives
/// every other filter: a desk-edge elbow passes the reach filter (it sits
/// a plausible 0.2 m from the shoulder) and its σ follows visibility, not
/// correctness — benched on s1789303569, such a lift pulled the wrist
/// 0.5 m for a frame and drove 5 of 6 R-wrist snaps. The 2-D elbow
/// residual stays (Cauchy already neuters it); only the metric lift is
/// demoted, mirroring the face-keypoint-under-a-hand-crop handling.
fn elbow_chain_sigma_cap(
    pred_fk: &Fk,
    kp3d: &[super::estimator::Kp3d],
    elbows: [usize; 2],
    wrists: [usize; 2],
) -> [f64; 2] {
    use super::estimator::ModelPoint;
    let mut cap = [f64::INFINITY, f64::INFINITY];
    // σ ceiling for a chain-contradicted elbow lift: ~2× its honest base
    // σ. At the benched 0.47 m innovation that whitens to ~16σ, where the
    // 3-D Cauchy weight drops to ≈0.1 — the hand block outvotes it.
    const CONTRADICTED_SIGMA: f64 = 0.03;
    // Tolerance beyond the model forearm length before the pair counts as
    // contradictory: skin→joint lifting offsets along two different rays
    // plus depth-median noise account for a few cm. 0.05 measured on
    // s1789311387: the desk-edge left-elbow lift reads the surface behind
    // the arm (0.728 m vs the true ~0.6 m), overshooting the wrist-to-elbow
    // obs span by 0.055 — at 0.06 it slipped through and the avatar's left
    // elbow stayed tucked behind the reach ("left elbow not tracked").
    // `VULVATAR_FUSION_CHAIN_TOL` overrides.
    const OVERSHOOT_TOL: f64 = 0.05;
    let lift = |j: usize| -> Option<&super::estimator::Kp3d> {
        kp3d.iter()
            .filter(|k| matches!(k.point, ModelPoint::Joint(jj) if jj == j))
            .min_by(|a, b| a.sigma.total_cmp(&b.sigma))
    };
    for (side, (&e_j, &w_j)) in elbows.iter().zip(wrists.iter()).enumerate() {
        let (Some(e), Some(w)) = (lift(e_j), lift(w_j)) else {
            continue;
        };
        let bone_obs = norm(sub(e.p, w.p));
        let bone_model = norm(sub(pred_fk.t[e_j], pred_fk.t[w_j]));
        if bone_obs - bone_model > OVERSHOOT_TOL {
            cap[side] = CONTRADICTED_SIGMA;
        }
    }
    cap
}

impl FusionProvider {
    /// Synchronous constructor: the detector pass runs inline inside
    /// `estimate_pose` (offline harnesses and the degraded paths).
    pub fn from_models_dir_with_config(
        models_dir: impl AsRef<Path>,
        config: TrackingPipelineConfig,
    ) -> Result<Self, String> {
        Self::construct(models_dir, config, false)
    }

    /// Live-tracking constructor: runs the detector stage on the
    /// `tracking-detect` thread so it overlaps the solver
    /// ([`PoseProvider::estimate_pose_latest`] is the matching entry
    /// point). Safe mode keeps everything inline and CPU-bound — the
    /// pipelining is a throughput optimisation, not a safety property.
    pub fn from_models_dir_live(
        models_dir: impl AsRef<Path>,
        config: TrackingPipelineConfig,
    ) -> Result<Self, String> {
        Self::construct(models_dir, config, !config.force_cpu)
    }

    fn construct(
        models_dir: impl AsRef<Path>,
        config: TrackingPipelineConfig,
        remote: bool,
    ) -> Result<Self, String> {
        let dir = models_dir.as_ref();
        let opts = DetectorOptions {
            face_ep: crate::tracking::face_mediapipe::FaceMeshEp::Auto,
            force_cpu: config.force_cpu,
        };
        let (det, backend_label, mut warnings) = if remote {
            // The detector thread builds its ONNX sessions while this
            // constructor blocks on `ready` — under the caller's
            // GPU-exclusive guard.
            let (client, ready_rx) = detector_thread::DetectorClient::spawn(dir.to_path_buf(), opts);
            match ready_rx.recv() {
                Ok(Ok(ready)) => {
                    let warnings = ready.warnings;
                    let label = ready.backend_label;
                    (
                        DetectorSlot::Remote {
                            client,
                            backend_label: label.clone(),
                        },
                        label,
                        warnings,
                    )
                }
                Ok(Err(e)) => return Err(e),
                Err(_) => return Err("detector thread died during model load".to_string()),
            }
        } else {
            let mut detector = Yolo11PoseInference::from_models_dir_with_options(dir, opts)?;
            let label = detector.backend().label();
            let warnings = detector.take_load_warnings();
            (
                DetectorSlot::Inline(detector),
                label,
                warnings,
            )
        };
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
        info!(
            "Fusion provider ready (YOLO11-pose {}{})",
            backend_label,
            if remote { ", pipelined" } else { "" }
        );
        let h = Humanoid::new();
        let params = Params::default();
        let est = Estimator::new(&h.model, params);
        let body_map = BodyMap::new(&h);
        Ok(Self {
            det,
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
            // VULVATAR_FUSION_OLDSIGMA=1: the pre-visibility σ policy
            // (ablation bench only).
            kp_sigma: if std::env::var_os("VULVATAR_FUSION_OLDSIGMA").is_some() {
                KpSigma {
                    floor_px: 1.5,
                    simcc_gain: 1.0,
                    ..KpSigma::default()
                }
            } else {
                KpSigma::default()
            },
            hand_swaps: 0,
            hand_kp_start: 0,
            last_solve_ms: 0.0,
            last_est_ms: 0.0,
            last_timings: PhaseTimings::default(),
            last_surface: Vec::new(),
            last_kp3d: Vec::new(),
            last_raw_joints: Vec::new(),
            last_crop: None,
            last_kp_stages: Vec::new(),
            vis_policy: VisPolicy::default(),
            sil_params: SilhouetteParams::default(),
            last_vis: Vec::new(),
            last_silhouette: None,
            frames_since_face: u32::MAX,
            last_sil: None,
            last_dense_n: 0,
            wrist_hold: Default::default(),
            depth_ring: Vec::new(),
            last_consumed: None,
            det_dead_logged: false,
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
        let backend = match &self.det {
            DetectorSlot::Inline(detector) => detector.backend().label(),
            DetectorSlot::Remote { backend_label, .. } => backend_label.clone(),
        };
        format!("Fusion v2 / YOLO11-pose {backend}")
    }

    fn take_load_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.load_warnings)
    }

    fn reset_temporal_state(&mut self) {
        match &mut self.det {
            DetectorSlot::Inline(detector) => detector.reset_temporal_state(),
            DetectorSlot::Remote { client, .. } => {
                // In-order on the detector thread: the reset is applied
                // before any job submitted after it; buffered depths
                // belong to discarded frames.
                client.reset_temporal_state();
                self.depth_ring.clear();
                self.last_consumed = None;
                self.det_dead_logged = false;
            }
        }
        self.est.reset(&self.h.model);
        self.prev_hands = [None, None];
        self.face_fit.reset();
        self.head_ori.reset();
        self.hand_unsupported = [0, 0];
        self.hand_suspect = [false, false];
        self.hand_dupes = 0;
        self.wrist_hold = Default::default();
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
        self.estimate_pose_inner(rgb_data, width, height, frame_index, false)
            .expect("synchronous estimate always yields an estimate")
    }

    fn estimate_pose_latest(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> Option<PoseEstimate> {
        self.estimate_pose_inner(rgb_data, width, height, frame_index, true)
    }

}

impl FusionProvider {
    /// One estimate pass. `latest` selects the pipelined live protocol:
    /// submit THIS frame's detector job, consume the freshest finished
    /// result, and return `None` when none is ready (the caller skips the
    /// frame; the camera paces the retry). With `latest = false` the call
    /// waits for exactly its own frame — the pre-pipelining behaviour.
    ///
    /// In Remote mode the frame whose solve runs here is the CONSUMED
    /// result's frame (`det_frame_index`), not the capture that triggered
    /// the call; everything downstream (observation build, output,
    /// diagnostics) is keyed off the consumed frame.
    fn estimate_pose_inner(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
        latest: bool,
    ) -> Option<PoseEstimate> {
        let t0 = std::time::Instant::now();
        let incoming_depth = self.external_depth.take();
        // This frame's device time — what the detector job (and the crop
        // hint derived from the predicted state) are keyed to.
        let job_ts_ms = incoming_depth.as_ref().and_then(|d| d.timestamp_ms);

        let intr_cam: Option<CameraIntrinsics> =
            incoming_depth.as_ref().and_then(|d| d.intrinsics);
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

        // ---- state-driven crop hint -----------------------------------------
        let mut ph = PhaseTimings::default();

        // ---- detector stage ---------------------------------------------------
        // (inline RTMW3D pass, or a job on the tracking-detect thread).
        let remote = matches!(self.det, DetectorSlot::Remote { .. });
        let mut det_depth: Option<MetricDepthFrame>;
        if remote {
            // Buffer this frame's metric depth until its result is
            // consumed (usually on the NEXT call). Capped: while the
            // detector is stalled or dead nothing consumes entries, and
            // each frame's cloud is a few hundred KB.
            if let Some(d) = incoming_depth {
                if self.depth_ring.len() >= DEPTH_RING_CAP {
                    self.depth_ring.remove(0);
                }
                self.depth_ring.push((frame_index, d));
            }
            det_depth = None;
        } else {
            det_depth = incoming_depth;
        }
        let mut det_frame_index = frame_index;
        let (mut base, mut aux) = 'det: {
            match &mut self.det {
                DetectorSlot::Inline(detector) => {
                    detector.set_frame_timestamp_ms(job_ts_ms);
                    let t_det = std::time::Instant::now();
                    let base = detector.estimate_pose(rgb_data, width, height, frame_index);
                    let aux = detector.take_aux();
                    ph.rtmw_ms = t_det.elapsed().as_secs_f32() * 1000.0;
                    break 'det (base, aux);
                }
                DetectorSlot::Remote { client, .. } => {
                    client.submit(detector_thread::DetectorJob {
                        frame_index,
                        ts_ms: job_ts_ms,
                        rgb: Arc::new(rgb_data.to_vec()),
                        width,
                        height,
                    });
                    let res = if latest {
                        match client.take_latest(self.last_consumed) {
                            Some(res) if res.frame_index != u64::MAX => res,
                            // Nothing fresh, or a stray reset ack: skip.
                            _ => return None,
                        }
                    } else if !client.alive() {
                        if !self.det_dead_logged {
                            self.det_dead_logged = true;
                            log::error!("fusion: detector thread is gone; publishing rest pose");
                        }
                        break 'det (
                            PoseEstimate {
                                skeleton: SourceSkeleton::default(),
                                annotation: DetectionAnnotation::default(),
                            },
                            None,
                        );
                    } else {
                        match client.wait_for(frame_index) {
                            Some(res) => res,
                            None => {
                                log::error!("fusion: detector thread lost frame {frame_index}");
                                break 'det (
                                    PoseEstimate {
                                        skeleton: SourceSkeleton::default(),
                                        annotation: DetectionAnnotation::default(),
                                    },
                                    None,
                                );
                            }
                        }
                    };
                    det_frame_index = res.frame_index;
                    // The detector work ran off-thread; surface its own
                    // elapsed time in the phase breakdown (0 would read
                    // as "stage missing").
                    ph.rtmw_ms = res.det_ms;
                    // Pair the consumed result with its own capture's
                    // depth; fall back to THIS call's depth (one frame
                    // stale, visually identical scene) rather than
                    // solving 2-D-only.
                    let pos = self
                        .depth_ring
                        .iter()
                        .position(|(i, _)| *i == det_frame_index);
                    det_depth = match pos.map(|p| self.depth_ring.remove(p).1) {
                        Some(d) => Some(d),
                        None => {
                            log::warn!(
                                "fusion: no buffered depth for consumed frame {det_frame_index}"
                            );
                            // Pathological (the matching depth is pushed
                            // before its job is submitted): fall back to
                            // the freshest buffered depth rather than a
                            // 2-D-only solve.
                            self.depth_ring.last().map(|(_, d)| d.clone())
                        }
                    };
                    self.depth_ring.retain(|(i, _)| *i > det_frame_index);
                    self.last_consumed = Some(det_frame_index);
                    break 'det (res.base, res.aux);
                }
            }
        };
        let depth = det_depth;
        let ts_ms = if det_frame_index == u64::MAX {
            None
        } else {
            depth.as_ref().and_then(|d| d.timestamp_ms)
        };
        // Time base of the SOLVED frame: device clock when available,
        // else nominal 30 fps.
        let t = match ts_ms {
            Some(ms) => ms / 1000.0,
            None => det_frame_index as f64 / 30.0,
        };

        let mut obs = FrameObs {
            t,
            intr: Some(intr),
            kp2d: Vec::with_capacity(700),
            kp3d: Vec::with_capacity(600),
            angles: Vec::with_capacity(30),
            ori: Vec::new(),
            shoulder_yaw: None,
            torso_hint: None,
            surface: Vec::new(),
            surf_allow: Vec::new(),
        };

        // Prediction for this frame (drives ROI extraction + face-shape
        // observation geometry before the solve).
        let pred = self.est.predict(&self.h.model, t);
        let fk_pred = self.h.model.fk(&pred);
        let head_j = self.h.j.head;
        let head_center_pred = fk_pred.site[self.h.s.head_center];

        // ---- keypoint visibility ---------------------------------------------
        // SimCC peak shape → p_vis, then the depth silhouette: a keypoint
        // not on the person's surface is dropped whatever its peak looks
        // like. The calibrated probability REPLACES the detector score from
        // here on (every downstream consumer reads `score`).
        let t_vis = std::time::Instant::now();
        self.last_vis.clear();
        self.last_silhouette = None;
        self.last_sil = None;
        self.last_dense_n = 0;
        if let Some(aux) = aux.as_mut() {
            if std::env::var_os("VULVATAR_FUSION_NO_VIS").is_none() && aux.joints.len() >= 133 {
                let pol = self.vis_policy;
                let sp = self.sil_params;
                let mut vis: Vec<f32> = aux.joints.iter().map(|j| pol.p_vis(j)).collect();
                // A keypoint outside the image is not an observation. The
                // person crop is zero-padded past the frame, and the
                // detector extrapolates arms into that black region with
                // SHARP peaks (measured: elbows / wrists at nx 1.12,
                // ny 1.29 with p_vis 0.98 on a hidden-hands desk session).
                // Likewise a keypoint pinned to the crop border is a joint
                // the crop cut off, not one at the border. A keypoint just
                // inside the FRAME edge stays: the 2-D / depth-lift terms
                // already skip the outer 2 % (`KpSigma::border_frac`), but
                // the hand-crop seeding needs it — a hand half cut by the
                // right edge (wrist at nx 0.98) still crops fine, and
                // without that crop the arm floated and the torso yawed
                // 30° to reach it (measured on a gesturing desk session).
                // Tolerance for a keypoint slightly past the edge: a face cut
                // by the frame has its nose tip 1–2 % outside with a sharp
                // peak (the visible face continues there) and dropping it
                // cost the FaceMesh crop its anchor (head yaw std 19 → 29°
                // on a profile session); the hallucinated arms sat 12–29 %
                // outside.
                const OUTSIDE_TOL: f32 = 0.05;
                let m = self.kp_sigma.border_frac;
                for (i, j) in aux.joints.iter().enumerate() {
                    let outside = !(-OUTSIDE_TOL..=1.0 + OUTSIDE_TOL).contains(&j.nx)
                        || !(-OUTSIDE_TOL..=1.0 + OUTSIDE_TOL).contains(&j.ny);
                    let on_crop_edge = aux.crop.is_some_and(|(cx, cy, cw, ch)| {
                        let (u, v) = (j.nx * width as f32, j.ny * height as f32);
                        let (mx, my) = (cw * m, ch * m);
                        u <= cx + mx || u >= cx + cw - mx || v <= cy + my || v >= cy + ch - my
                    });
                    if outside || on_crop_edge {
                        vis[i] = 0.0;
                    }
                }
                let p_simcc = vis.clone();
                let (dw, dh) = depth
                    .as_ref()
                    .map(|d| (d.width as f64, d.height as f64))
                    .unwrap_or((width as f64, height as f64));
                // Silhouette seeds: the visible face keypoints. At least two
                // are required — measured: every one of 7 359 frames with a
                // person had ≥ 3, the empty chair never had 2 (38 of 76
                // frames had exactly one). No fallback to the predicted
                // head: after the user leaves, the prediction would seed the
                // chair and keep a phantom track alive.
                let mut seeds: Vec<(f64, f64)> = (0..5)
                    .filter(|&i| vis[i] >= pol.min_p)
                    .map(|i| (aux.joints[i].nx as f64 * dw, aux.joints[i].ny as f64 * dh))
                    .filter(|&(u, v)| u >= 0.0 && v >= 0.0 && u < dw && v < dh)
                    .collect();
                if seeds.len() >= 2 {
                    self.frames_since_face = 0;
                } else {
                    seeds.clear();
                    self.frames_since_face = self.frames_since_face.saturating_add(1);
                    // Short bridge over a face-less frame (a hand sweep,
                    // a hard turn): seed from the predicted head while
                    // the face was seen within the last second. Longer
                    // than that the person may have left — a stale
                    // prediction would seed the empty chair.
                    if self.frames_since_face <= 30
                        && self.est.joint_data_sigma(&self.h.model, self.h.j.head) < 0.5
                    {
                        if let Some(p) = intr.project(head_center_pred) {
                            if p[0] >= 0.0 && p[1] >= 0.0 && p[0] < dw && p[1] < dh {
                                seeds.push((p[0], p[1]));
                            }
                        }
                    }
                }
                let mut sil = depth.as_ref().and_then(|d| {
                    build_silhouette(&d.points_m, d.width, d.height, intr.fx, &seeds, &sp)
                });
                let mut dist = vec![f32::NAN; aux.joints.len()];
                match sil.as_mut() {
                    Some(s) => {
                        let pts = &depth.as_ref().unwrap().points_m;
                        for (i, j) in aux.joints.iter().enumerate() {
                            let (u, v) = (j.nx as f64 * dw, j.ny as f64 * dh);
                            let dm = s.dist_m(u, v);
                            dist[i] = dm as f32;
                            let hole = window_median_z(
                                pts,
                                s.src_width as usize,
                                s.src_height as usize,
                                u.round() as i64,
                                v.round() as i64,
                                2,
                            )
                            .is_none();
                            // A hole AT the frame edge is unverifiable: the
                            // detector parks extrapolated arms there (a
                            // hidden-hands session had the wrist at nx
                            // 0.995–1.0 over a hole, 0.03 m from the
                            // shoulder's outline, in 11 % of frames). A real
                            // hand cut by the edge still seeds its crop from
                            // the hand-block points inside the frame.
                            let at_edge =
                                !(m..=1.0 - m).contains(&j.nx) || !(m..=1.0 - m).contains(&j.ny);
                            if hole && at_edge {
                                vis[i] = 0.0;
                                continue;
                            }
                            let tol = if hole {
                                sp.max_dist_hole_m
                            } else {
                                sp.max_dist_m
                            };
                            if dm > tol && vis[i] >= pol.min_p_obs {
                                // Not on the main surface. A hand / forearm
                                // held in front of the chest or entering from
                                // outside the frame is a small detached blob
                                // at the person's depth; the desk or a
                                // bystander is not.
                                let (lo, hi) = sp.limb_blob_area_m2;
                                let limb = s
                                    .blob_area_m2(pts, u, v, hi)
                                    .map(|a| a >= lo && a <= hi)
                                    .unwrap_or(false);
                                if limb {
                                    dist[i] = 0.0;
                                } else {
                                    vis[i] = 0.0;
                                }
                            }
                        }
                        // Truncated trunk: with less than a crown-to-hip
                        // height visible, the hips and everything below
                        // are outside the frame; a leg keypoint that still
                        // passes the peak test is painted on the torso /
                        // face (measured: knees at chest height, 31 snaps
                        // in one desk session).
                        if s.height_m < sp.min_height_for_legs_m {
                            for v in vis.iter_mut().take(23).skip(11) {
                                *v = 0.0;
                            }
                        }
                        self.last_silhouette = Some((s.z_ref, s.touches_bottom, s.height_m));
                    }
                    None => {
                        // Depth available but no person surface under a
                        // visible face: nobody to track (empty chair) —
                        // nothing is observed.
                        if depth.is_some() {
                            vis.iter_mut().for_each(|v| *v = 0.0);
                        }
                    }
                }
                for (i, j) in aux.joints.iter_mut().enumerate() {
                    j.score = if vis[i] >= pol.min_p_obs { vis[i] } else { 0.0 };
                }
                self.last_sil = sil.take();
                for (k, j) in base.annotation.keypoints.iter_mut().zip(aux.joints.iter()) {
                    k.2 = j.score;
                }
                self.last_vis = p_simcc
                    .iter()
                    .zip(dist.iter())
                    .map(|(&p, &d)| (p, d))
                    .collect();
            }
        }

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

        // Legacy per-joint geometric gates (border clamp, reach, leg /
        // arm coherence). Off by default now that the visibility layer
        // decides what is on the person; `VULVATAR_FUSION_OLDGATES=1`
        // re-enables them for the ablation bench.
        let old_gates = std::env::var_os("VULVATAR_FUSION_OLDGATES").is_some()
            || std::env::var_os("VULVATAR_FUSION_NO_VIS").is_some();

        // ---- hand-block L/R assignment --------------------------------------
        ph.vis_ms = t_vis.elapsed().as_secs_f32() * 1000.0;
        let t_hands = std::time::Instant::now();
        // The detector's left/right hand blocks can be transposed when the
        // hands cross or touch. Decide the assignment against the predicted
        // wrists (only when both wrists are currently tracked): keep the
        // labelling unless swapping is clearly better.
        let mut det_kps: Vec<(f32, f32, f32)> = base.annotation.keypoints.clone();
        if let Some(d) = depth.as_ref().filter(|_| old_gates) {
            if d.points_m.len() == (d.width * d.height) as usize && det_kps.len() >= 133 {
                let mut tmp: Vec<RawKp> = det_kps
                    .iter()
                    .map(|&(nx, ny, sc)| RawKp {
                        nx,
                        ny,
                        score: sc,
                        sx: 0.0,
                        sy: 0.0,
                    })
                    .collect();
                let zs = [
                    fk_pred.t[self.h.j.l_shoulder][2],
                    fk_pred.t[self.h.j.r_shoulder][2],
                ];
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
                    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
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
                for (ci, crop) in candidates.into_iter().enumerate() {
                    if let Some(mut res) = hl.estimate(rgb_data, width, height, crop) {
                        res.src = ci as u8;
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
                        let better = best
                            .as_ref()
                            .map(|b| res.presence > b.presence)
                            .unwrap_or(true);
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
        // Elbow σ ceilings from forearm bone-rigidity (filled after the
        // depth-lifts exist; see `elbow_chain_sigma_cap`).
        let mut chain_cap = [f64::INFINITY, f64::INFINITY];

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
            self.last_raw_joints = aux.joints.clone();
            self.last_crop = aux.crop;
            self.last_kp_stages.clear();
            let snap =
                |stages: &mut Vec<(&'static str, Vec<f32>)>, name: &'static str, raw: &[RawKp]| {
                    stages.push((name, raw.iter().map(|k| k.score).collect()));
                };
            snap(&mut self.last_kp_stages, "raw", &raw);
            if let Some(crop) = aux.crop.filter(|_| old_gates) {
                cull_crop_border(&mut raw, crop, width, height, 0.02);
            }
            snap(&mut self.last_kp_stages, "border", &raw);
            // Arm reachability against the predicted shoulders (depth-valid
            // pixels only) — see `reach_filter`.
            if let Some(d) = depth.as_ref().filter(|_| old_gates) {
                if d.points_m.len() == (d.width * d.height) as usize {
                    let zs = [
                        fk_pred.t[self.h.j.l_shoulder][2],
                        fk_pred.t[self.h.j.r_shoulder][2],
                    ];
                    if std::env::var_os("VULVATAR_FUSION_NO_REACH").is_none() {
                        reach_filter(&mut raw, &d.points_m, d.width, d.height, zs, 0.75);
                    }
                }
            }
            snap(&mut self.last_kp_stages, "reach", &raw);
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
            if old_gates {
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
                        for i in [
                            13 + side,
                            15 + side,
                            17 + 3 * side,
                            18 + 3 * side,
                            19 + 3 * side,
                        ] {
                            if i < raw.len() {
                                raw[i].score = 0.0;
                            }
                        }
                    }
                }
            }
            snap(&mut self.last_kp_stages, "leg", &raw);
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
            if let Some(d) = depth.as_ref().filter(|_| old_gates) {
                if d.points_m.len() == (d.width * d.height) as usize {
                    let z_person = head_center_pred[2];
                    let model_pelvis_w =
                        norm(sub(fk_pred.t[self.h.j.l_hip], fk_pred.t[self.h.j.r_hip]));
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
            snap(&mut self.last_kp_stages, "coherence", &raw);
            super::coherence::filter_duplicate_wrists(&mut raw, width, height);
            snap(&mut self.last_kp_stages, "dupwrist", &raw);
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
            snap(&mut self.last_kp_stages, "handcrop", &raw);
            let n_before = obs.kp2d.len();
            // With a dense-mesh centroid anchoring head position, the
            // SimCC face keypoints only ADD their frontalization bias —
            // widen them so they stop binding head yaw (~4° measured).
            let mesh_conf = aux.face_mesh.as_ref().map(|(_, c)| *c).unwrap_or(0.0);
            let head_scale = if !face_occluded && mesh_conf >= 0.35 {
                1.0 + 2.0 * ((mesh_conf - 0.35) / 0.25).clamp(0.0, 1.0) as f64
            } else {
                1.0
            };
            // Scale on the SimCC face-kp σ inflation. The conf-derived
            // ×3 inflation suppresses the eye-line signal so much that
            // head roll under-responds (GT round-trip gain 0.1–0.4,
            // validate_gt head_roll_*) and the profile-view torso lands
            // in the wrong basin. ×0.5 halves the inflation: full 29-
            // recording bench 2026-09-14 — torso yaw |err| sum 156°→74°,
            // big-error sessions collapse (−38.8→+4.3, −13.4→+2.9,
            // namaste −9.0→−1.1); jitter mixed (s1787200720 sd 18→30,
            // s1789349575 yaw jit 1.7→3.9). Env overrides for sweeps.
            let head_scale = head_scale
                * std::env::var("VULVATAR_HEAD_KP_SCALE")
                    .ok()
                    .and_then(|v| v.parse::<f64>().ok())
                    .unwrap_or(0.5);
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
                    // Corroboration gate, as the post-solve burn-in loop:
                    // an observation the prediction already agrees with is
                    // confirmation, not a suspect re-entry.
                    let agrees = std::env::var_os("VULVATAR_FUSION_NO_CORROB").is_none() && {
                        let (_, pw) =
                            super::estimator::resolve_point(&self.h.model, &fk_pred, k.point);
                        intr.project(pw).is_some_and(|pv| {
                            ((k.u - pv[0]) * (k.u - pv[0]) + (k.v - pv[1]) * (k.v - pv[1])).sqrt()
                                < 4.0 * k.sigma
                        })
                    };
                    if !agrees {
                        k.sigma *= inflate[side];
                    }
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
                    chain_cap = elbow_chain_sigma_cap(
                        &fk_pred,
                        &obs.kp3d,
                        [self.h.j.l_elbow, self.h.j.r_elbow],
                        [self.h.j.l_wrist, self.h.j.r_wrist],
                    );
                    // ---- depth-confirmed wrist hold (see `wrist_hold`) ----
                    if std::env::var("VULVATAR_WRIST_HOLD")
                        .map(|v| v == "1")
                        .unwrap_or(false)
                    {
                        let hold_max = std::env::var("VULVATAR_WRIST_HOLD_MAX")
                            .ok()
                            .and_then(|v| v.parse::<f64>().ok())
                            .unwrap_or(2.5);
                        for side in 0..2 {
                            let j_wr = if side == 0 {
                                self.h.j.l_wrist
                            } else {
                                self.h.j.r_wrist
                            };
                            let is_wr = |point: &super::estimator::ModelPoint| matches!(point, super::estimator::ModelPoint::Joint(j) if *j == j_wr);
                            let seen_3d = obs.kp3d.iter().rev().find(|k| is_wr(&k.point));
                            let seen_2d = obs.kp2d.iter().any(|k| is_wr(&k.point));
                            let trace = std::env::var_os("VULVATAR_WRIST_HOLD_TRACE").is_some();
                            if trace {
                                eprintln!(
                                    "WRISTHOLD f{frame_index} side {side} seen3d {} seen2d {} age {:.2}",
                                    seen_3d.is_some(),
                                    seen_2d,
                                    t - self.wrist_hold[side].last_t.unwrap_or(t)
                                );
                            }
                            if let Some(k) = seen_3d {
                                self.wrist_hold[side] = WristHold {
                                    last_t: Some(t),
                                    last_z: Some(k.p[2]),
                                };
                            } else if seen_2d {
                                // Fresh bearing without a lifted 3-D point
                                // (depth hole or hand-block-only frame): the
                                // sensor can still testify the depth under
                                // the pixel, which arms the hold for the
                                // frames the detector goes quiet.
                                let k2 = obs.kp2d.iter().rev().find(|k| is_wr(&k.point)).unwrap();
                                let band = 0.7_f64;
                                let (zlo, zhi) = match z_ref {
                                    Some(z) => ((z - band) as f32, (z + band) as f32),
                                    None => (0.15, 6.0),
                                };
                                let z = window_point(
                                    &d.points_m,
                                    d.width,
                                    d.height,
                                    k2.u,
                                    k2.v,
                                    3,
                                    zlo,
                                    zhi,
                                )
                                .filter(|p| p[2] > 0.1)
                                .map(|p| p[2]);
                                self.wrist_hold[side].last_t = Some(t);
                                self.wrist_hold[side].last_z = z;
                            } else if let (Some(t0), Some(z0)) =
                                (self.wrist_hold[side].last_t, self.wrist_hold[side].last_z)
                            {
                                let trace = std::env::var_os("VULVATAR_WRIST_HOLD_TRACE").is_some();
                                let age = t - t0;
                                let miss = |why: &str| {
                                    if trace {
                                        eprintln!("WRISTHOLD side {side} age {age:.2} MISS {why}");
                                    }
                                };
                                if !(0.0..=hold_max).contains(&age) {
                                    miss("age");
                                    continue;
                                }
                                let Some(uv) = intr.project(fk_pred.t[j_wr]) else {
                                    miss("project");
                                    continue;
                                };
                                let (mx, my) = (0.02 * intr.width, 0.02 * intr.height);
                                if uv[0] < mx
                                    || uv[0] > intr.width - mx
                                    || uv[1] < my
                                    || uv[1] > intr.height - my
                                {
                                    miss(&format!("border uv {:.0},{:.0}", uv[0], uv[1]));
                                    continue;
                                }
                                let Some(psurf) = window_point(
                                    &d.points_m,
                                    d.width,
                                    d.height,
                                    uv[0],
                                    uv[1],
                                    3,
                                    (z0 - 0.2) as f32,
                                    (z0 + 0.2) as f32,
                                ) else {
                                    miss("no-depth");
                                    continue;
                                };
                                const Z_GATE: f64 = 0.12;
                                if (psurf[2] - z0).abs() > Z_GATE {
                                    miss(&format!("zgate {:.2} vs {:.2}", psurf[2], z0));
                                    continue;
                                }
                                // Skin→joint push along the ray, as the
                                // wrist entries of `body_kp3d`.
                                let n = norm(psurf);
                                if n < 0.1 {
                                    continue;
                                }
                                let p_joint = scale(psurf, (n + 0.02) / n);
                                // σ grows with time since the last real
                                // detection, so a re-acquired keypoint
                                // (σ ≈ 0.05) always outvotes the hold.
                                let sigma = (0.05 + 0.08 * age).min(0.20);
                                if trace {
                                    eprintln!(
                                        "WRISTHOLD side {side} age {age:.2} HOLD uv {:.0},{:.0} z {:.2} (was {:.2}) sigma {sigma:.2}",
                                        uv[0], uv[1], psurf[2], z0
                                    );
                                }
                                obs.kp3d.push(super::estimator::Kp3d {
                                    point: super::estimator::ModelPoint::Joint(j_wr),
                                    p: p_joint,
                                    sigma,
                                    lat_scale: 1.0,
                                });
                                // Follow the sensor within the gate so a
                                // slow drift along the desk keeps tracking.
                                self.wrist_hold[side].last_z = Some(psurf[2]);
                            }
                        }
                    }
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
                            // ~7° at a full 15-column fit, widening as
                            // columns drop out. `VULVATAR_CHESTYAW_SIGMA`
                            // scales it (basin-selection bench: σ 0.12 keeps
                            // the trunk's cloud term from picking a
                            // self-consistent wrong-yaw basin).
                            // Default scale 1.5 (σ ≈ 0.18 rad): measured on
                            // the s1789303569 desk replay (2026-09-13) — at
                            // scale 1.0 the obs's hard L2 clamp fights the
                            // trunk terms through every arm transient
                            // (torso-yaw err std 4.0°, excursions to
                            // -28.5°, and the fight's damage surfaces as
                            // R-wrist snaps: max jump 0.71 m). A robust
                            // kernel tail is NOT the answer — benched 4×
                            // worse (a weakened-but-abandoning anchor lets
                            // the trunk hover between basins, yaw err std
                            // 22.6°); the committed quadratic pull must
                            // stay. Scale 1.5 keeps that pull while easing
                            // the clamp: on s1789303569 yaw err std 4.0 →
                            // 2.3°, range [-28.5°,+8.8°] → [-8.1°,+10.6°];
                            // on s1789246660 R-wrist snaps stay at 0
                            // (scale 2.0 re-introduced 4 there) while its
                            // yaw err std holds at 2.6°. The remaining
                            // s1789303569 wrist slam is the mis-detected
                            // elbow itself (see the arm σ cap above).
                            let base_sigma = 0.12 * (15.0 / n as f64).sqrt();
                            // 0.5 (2026-09-14, full 29-recording bench):
                            // torso yaw sd roughly halves on the noisy
                            // sessions (s1789349252 12.7→4.9, s1789088238
                            // 37.4→6.0), |err| sum ~103°→~60°, wrist snaps
                            // neutral (195→191) and the INDEPENDENT 2-D
                            // reprojection fit unchanged (11/11) — i.e. the
                            // gain is not just parroting the same-signal
                            // shoulder-depth reference. 0.25 was too tight
                            // (arm fights: L snaps 5→22 on s1789349252).
                            let scale = std::env::var("VULVATAR_CHESTYAW_SIGMA")
                                .ok()
                                .and_then(|v| v.parse::<f64>().ok())
                                .filter(|v| *v > 0.0)
                                .unwrap_or(0.5);
                            obs.shoulder_yaw = Some(super::estimator::ShoulderYawObs {
                                left: self.h.j.l_shoulder,
                                right: self.h.j.r_shoulder,
                                yaw,
                                sigma: base_sigma * scale,
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
            // ABL_NOMESH: bench ablation — cut the canonical-face attach
            // (centroid 2-D anchor + depth-lifted 3-D landmarks), keeping
            // the mesh-pose channel itself. Isolates how much head roll
            // the estimator draws from the dense template fit.
            .filter(|_| std::env::var_os("VULVATAR_ABL_NOMESH").is_none())
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
                            point: super::estimator::ModelPoint::Attached {
                                joint: head_j,
                                local,
                            },
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
                    let local = add(scale([c[0] as f64, c[1] as f64, c[2] as f64], s_fit), t_fit);
                    let pw = add(head_t_pred, mat_vec(&head_r_pred, local));
                    let n_local = normalize([c[0] as f64, c[1] as f64, (c[2] as f64) - 0.02]);
                    let n_world = mat_vec(&head_r_pred, n_local);
                    let facing = dot(n_world, normalize(scale(pw, -1.0)));
                    if facing < -0.1 {
                        continue;
                    }
                    let point = super::estimator::ModelPoint::Attached {
                        joint: head_j,
                        local,
                    };
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
            // Continuous far-side de-weighting (2026-09-16, replaces the
            // binary cull): a landmark still on the near hemisphere but
            // turning away gets its σ inflated smoothly up to ×4 at the
            // old cull edge, instead of a knife-edge drop at a single
            // threshold — sweeping the old binary threshold showed the
            // discontinuity (0.25 alone flipped one replay's head roll
            // +40°→+14° while 0.20/0.35 were unchanged). Below
            // `facing_min` (still `VULVATAR_HEAD_CULL_FACING`, −0.15) the
            // landmark remains dropped: it cannot be a real observation.
            let facing_min = std::env::var("VULVATAR_HEAD_CULL_FACING")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(-0.15);
            let head_c = fk_pred.site[self.h.s.head_center];
            let head_sites = [
                self.h.s.l_eye,
                self.h.s.r_eye,
                self.h.s.l_ear,
                self.h.s.r_ear,
            ];
            let facing_of = |pw: V3| -> f64 {
                let n = normalize(sub(pw, head_c));
                let to_cam = normalize(scale(pw, -1.0));
                dot(n, to_cam)
            };
            // σ multiplier for a still-visible landmark (None = drop).
            let facing_weight = |facing: f64| -> Option<f64> {
                if facing >= 0.0 {
                    Some(1.0)
                } else if facing > facing_min {
                    Some(1.0 + 3.0 * (facing / facing_min))
                } else {
                    None
                }
            };
            let h = &self.h;
            let pred = &pred;
            let fkp = &fk_pred;
            let head_pw = |k: &super::estimator::Kp2d| -> Option<V3> {
                use super::estimator::ModelPoint;
                match k.point {
                    ModelPoint::Site(sid) if head_sites.contains(&sid) => Some(fkp.site[sid]),
                    ModelPoint::Attached { joint, local } if joint == h.j.head => {
                        let _ = pred;
                        Some(add(fkp.t[joint], mat_vec(&fkp.r[joint], local)))
                    }
                    _ => None,
                }
            };
            let head_pw3 = |k: &super::estimator::Kp3d| -> Option<V3> {
                use super::estimator::ModelPoint;
                match k.point {
                    // Site-based head points (nose / eyes / ears lifted by
                    // `body_kp3d`) need the same far-side treatment as the
                    // 2-D ones: the detector places the hidden ear on the
                    // silhouette, where the depth belongs to the NEAR side
                    // of the head.
                    ModelPoint::Site(sid) if head_sites.contains(&sid) => Some(fkp.site[sid]),
                    ModelPoint::Attached { joint, local } if joint == h.j.head => {
                        Some(add(fkp.t[joint], mat_vec(&fkp.r[joint], local)))
                    }
                    _ => None,
                }
            };
            obs.kp2d.retain_mut(|k| match head_pw(k) {
                Some(pw) => match facing_weight(facing_of(pw)) {
                    Some(w) => {
                        k.sigma *= w;
                        true
                    }
                    None => false,
                },
                None => true,
            });
            obs.kp3d.retain_mut(|k| match head_pw3(k) {
                Some(pw) => match facing_weight(facing_of(pw)) {
                    Some(w) => {
                        k.sigma *= w;
                        true
                    }
                    None => false,
                },
                None => true,
            });
        }

        // ---- hand crop observations ---------------------------------------------
        ph.hands_ms = t_hands.elapsed().as_secs_f32() * 1000.0;
        let t_head = std::time::Instant::now();
        self.hand_kp_start = obs.kp2d.len();
        for hand in 0..2 {
            let Some(res) = self.last_hands[hand].as_ref() else {
                continue;
            };
            let wrist_j = if hand == 0 {
                self.h.j.l_wrist
            } else {
                self.h.j.r_wrist
            };
            // The converted model's "world" output is not metric on this
            // checkpoint (index MCP reads ~3 cm from the wrist). Its
            // PROPORTIONS are usable though, so it is re-scaled per frame
            // against the model's own wrist→index-MCP span and fed as 3-D
            // terms — finger curl is a depth DOF and the 2-D-only fallback
            // left it free to hyperextend through a clenched fist (the
            // projected silhouette of a fist is 2-D-degenerate with an
            // extended, rotated hand).
            // The converted model's "world" output is not metric on this
            // checkpoint (index MCP reads ~3 cm from the wrist); its
            // proportions are too noisy (±30–60 %) for Cartesian terms
            // even after per-frame re-scaling — measured: wrist snap
            // 1→3–4, max jump 0.48 m, med2d 3.6→9.6 on a desk session.
            // 2-D only until curl is observed as scale/wrist-invariant
            // finger-joint angles (see hands.rs).
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
                &mut obs.angles,
            );
        }

        // ---- head orientation from the dense FaceMesh pose --------------------
        // The SimCC face-68 landmarks hallucinate a near-frontal arrangement
        // at 3/4+ turns (and the self-learned rigid face cannot anchor
        // absolute orientation), so landmark terms alone under-rotate the
        // head. FaceMesh's dense-landmark pose derivation tracks yaw
        // to ~±60° with high confidence — feed it as a direct world-
        let mut have_full_ori = false;
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
            have_full_ori = true;
        }
        if !have_full_ori {
            // Roll-only fallback (opt-in VULVATAR_FUSION_ROLLORI=1, default
            // OFF): the mesh channel's eye-line tilt where the full
            // orientation obs is gated out (head_ori.rs docs).
            if let Some(rori) = self.head_ori.estimate_roll_ori(
                &base,
                &occl,
                &fk_pred.r[self.h.j.head],
                self.h.j.head,
            ) {
                obs.ori.push(rori);
            }
        }

        // ---- dense surface (primary metric observation) ----------------------
        ph.head_ms = t_head.elapsed().as_secs_f32() * 1000.0;
        let t_dense = std::time::Instant::now();
        // Every visible pixel of the person's silhouette is a measured point
        // on the body surface. Fitted against the capsule surfaces it fixes
        // root depth and trunk orientation frame by frame; the sparse 2-D
        // keypoints then only resolve what a surface cannot (left/right,
        // position along the surface, hands). Hand crops are excluded (the
        // landmarker owns them) and the estimator skips hand capsules in the
        // association.
        // `VULVATAR_FUSION_NO_DENSE=1` disables (ablation bench).
        if std::env::var_os("VULVATAR_FUSION_NO_DENSE").is_none() {
            if let (Some(sil), Some(d)) = (self.last_sil.as_ref(), depth.as_ref()) {
                if d.points_m.len() == (d.width * d.height) as usize {
                    // ~1 300 points at 640×480: the per-capsule count
                    // normalisation (`surf_n_eff`) makes denser sampling
                    // pure cost.
                    let stride = if d.width >= 1000 { 12 } else { 8 };
                    let dense = sil.sample_points(&d.points_m, stride, &in_hand_rect);
                    // Phantom-track guard: after the subject leaves, the
                    // ≤1 s predicted-head bridge (or a one-frame detection
                    // blip) can regrow a silhouette on the chair/wall and
                    // hand the trunk a few dozen "measurements" — enough
                    // to keep the data-info EMA, and with it rig quality,
                    // alive while the state random-walks (measured
                    // 2026-09-13: n2d=n3d=0 frames with 26-88 stray
                    // surface points, quality bouncing 0.4-0.7, root
                    // drifting 0.9→3.9 m over 45 s). A real subject at
                    // desk distance yields ~1000 points and even a full
                    // body at 3 m ~400; below this floor there is not
                    // enough surface to claim a person — drop the samples
                    // and let the estimator decay to unobserved.
                    const MIN_DENSE_SURFACE_POINTS: usize = 200;
                    self.last_dense_n = dense.len();
                    if dense.len() >= MIN_DENSE_SURFACE_POINTS {
                        obs.surface.extend(dense);
                    }
                    // Which capsules may claim surface points this frame.
                    let m = &self.h.model;
                    let tracked = |j: usize| self.est.joint_data_sigma(m, j) < 0.5;
                    let legs_in_frame = sil.height_m >= self.sil_params.min_height_for_legs_m;
                    let j = &self.h.j;
                    obs.surf_allow = m
                        .capsules
                        .iter()
                        .map(|c| {
                            let starts_at =
                                |jj: usize| matches!(c.a, PointRef::Joint(x) if x == jj);
                            let no_head = std::env::var_os("VULVATAR_DENSE_NOHEAD").is_some();
                            let no_neck = std::env::var_os("VULVATAR_DENSE_NONECK").is_some();
                            match c.part {
                                Part::Head => !no_head,
                                Part::Torso => !(no_neck && starts_at(j.spine3)),
                                Part::LeftHand | Part::RightHand => false,
                                // Arms never claim surface points (they still
                                // occlude: points nearest an arm are dropped).
                                // Letting a tracked arm compete for points was
                                // measured to couple the surface into the arm
                                // seed contest (seed wins 3 → 15, wrist jumps
                                // 0.5–0.8 m on the wave replay); the arms stay
                                // on 2-D + depth-lift + hand crops.
                                // `VULVATAR_DENSE_ARMS=1` re-enables for benches.
                                Part::LeftArm | Part::RightArm
                                    if std::env::var_os("VULVATAR_DENSE_ARMS").is_none() =>
                                {
                                    false
                                }
                                Part::LeftArm => {
                                    if starts_at(j.l_shoulder) {
                                        tracked(j.l_elbow)
                                    } else {
                                        tracked(j.l_wrist)
                                    }
                                }
                                Part::RightArm => {
                                    if starts_at(j.r_shoulder) {
                                        tracked(j.r_elbow)
                                    } else {
                                        tracked(j.r_wrist)
                                    }
                                }
                                Part::LeftLeg => legs_in_frame && tracked(j.l_knee),
                                Part::RightLeg => legs_in_frame && tracked(j.r_knee),
                            }
                        })
                        .collect();
                }
            }
        }

        // Ablation switches for the replay bench.
        if std::env::var_os("VULVATAR_FUSION_NO_SURF").is_some() {
            obs.surface.clear();
        }
        if std::env::var_os("VULVATAR_FUSION_NO_3D").is_some() {
            obs.kp3d.clear();
        }
        if std::env::var_os("VULVATAR_FUSION_KEEP_CLOUD").is_some() {
            self.last_surface = obs
                .surface
                .iter()
                .map(|(p, _)| [p[0] as f32, p[1] as f32, p[2] as f32])
                .collect();
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
        // Corroboration gate: an observation that already AGREES with the
        // predicted pose is not a suspect re-entry — it is the sensor
        // confirming where the arm is, and inflating it starves the arm's
        // information EMA (measured deadlock on a desk session: the hand
        // block fires in isolated single frames at the frame edge, every
        // burst arrives at σ×4, trust never accumulates, R-wrist duty
        // 0.08 and 32–39 wrist snaps / 600 frames). Only observations that
        // DISAGREE with the prediction keep the burn-in inflation.
        let corrob = std::env::var_os("VULVATAR_FUSION_NO_CORROB").is_none();
        // The inflation exists so a first re-acquisition still moves the arm
        // "proportionally to the evidence". Past ~2.7× the elbow's honest
        // σ (~3 cm) it breaks the 3-D Cauchy kernel instead: c_3d saturates
        // at 5σ, so a σ 0.23 m entry pulls with ~96% weight toward a
        // 0.47 m mis-detection (measured on s1789303569: the desk-edge
        // elbow slam behind every R-wrist snap; snap frames carried
        // σ 0.15–0.23 with 2-D innovation 278 px at claimed σ 9.9 px).
        // Cap the fiction: 8 cm keeps first-frame damping (3× base) while
        // a 0.4 m+ slam whitens back to ≥5σ where Cauchy actually
        // saturates. `VULVATAR_FUSION_ARM_SIG_CAP` overrides; 0 disables.
        let arm_sig_cap = std::env::var("VULVATAR_FUSION_ARM_SIG_CAP")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| *v > 0.0)
            .unwrap_or(0.08);
        for k in obs.kp3d.iter_mut() {
            if let Some(side) = self.arm_side_of(k.point) {
                let agrees = corrob && {
                    let (_, pw) = super::estimator::resolve_point(&self.h.model, &fk_pred, k.point);
                    norm(sub(k.p, pw)) < 0.10
                };
                if !agrees {
                    k.sigma *= inflate[side];
                }
                // The chain cap is elbow-only: the contradicting wrist /
                // hand-block lift must keep its full say.
                let is_elbow = matches!(
                    k.point,
                    super::estimator::ModelPoint::Joint(jj)
                        if jj == self.h.j.l_elbow || jj == self.h.j.r_elbow
                );
                k.sigma = k
                    .sigma
                    .min(arm_sig_cap)
                    .min(if is_elbow { chain_cap[side] } else { f64::INFINITY });
            }
        }
        let n_hand_start = self.hand_kp_start.min(obs.kp2d.len());
        for k in obs.kp2d[n_hand_start..].iter_mut() {
            if let Some(side) = self.arm_side_of(k.point) {
                let agrees = corrob && {
                    let (_, pw) = super::estimator::resolve_point(&self.h.model, &fk_pred, k.point);
                    intr.project(pw).is_some_and(|pv| {
                        ((k.u - pv[0]) * (k.u - pv[0]) + (k.v - pv[1]) * (k.v - pv[1])).sqrt()
                            < 4.0 * k.sigma
                    })
                };
                if !agrees {
                    k.sigma *= inflate[side];
                }
            }
        }

        // ---- solve (with analytic arm re-seeds from metric joints) ------------
        ph.dense_ms = t_dense.elapsed().as_secs_f32() * 1000.0;
        let t_est = std::time::Instant::now();
        super::seed::update_with_arm_seeds(&self.h, &mut self.est, &obs);
        self.last_est_ms = t_est.elapsed().as_secs_f32() * 1000.0;
        let et = self.est.timings;
        ph.acc_ms = et.acc_ms as f32;
        ph.eval_ms = et.eval_ms as f32;
        ph.lin_ms = et.lin_ms as f32;
        ph.main_ms = et.main_ms as f32;
        ph.seeds_ms = et.seeds_ms as f32;
        ph.reacc_ms = et.reacc_ms as f32;
        ph.finish_ms = et.finish_ms as f32;

        dump_obs_post_solve(
            &self.h,
            &self.est,
            frame_index,
            t,
            &obs,
            &intr,
            depth.as_ref(),
        );
        self.last_t = Some(t);
        self.frames += 1;

        // ---- learn face shapes from the posterior ------------------------------
        let t_facefit = std::time::Instant::now();
        {
            let fk = self.h.model.fk(&self.est.state);
            let head_sigma = self.est.joint_sigma(&self.h.model, head_j);
            let neck_sigma = self.est.joint_sigma(&self.h.model, self.h.j.neck);
            if head_sigma < 0.35 && neck_sigma < 0.4 {
                if let (Some(d), Some(lm)) = (depth.as_ref(), mesh_px) {
                    let z_ref = fk.site[self.h.s.head_center][2];
                    self.face_fit
                        .update(lm, fk.t[head_j], &fk.r[head_j], z_ref, d, &in_hand_rect);
                }
            }
        }
        ph.facefit_ms = t_facefit.elapsed().as_secs_f32() * 1000.0;

        // ---- output ----------------------------------------------------------------
        let t_output = std::time::Instant::now();
        let span_px = {
            let fk = self.h.model.fk(&self.est.state);
            match (
                intr.project(fk.t[self.h.j.l_shoulder]),
                intr.project(fk.t[self.h.j.r_shoulder]),
            ) {
                (Some(a), Some(b)) => Some(((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()),
                _ => None,
            }
        };
        let rig = output::rig_pose(&self.h, &self.est, t, span_px);
        let ref_span = {
            let fk = self.h.model.fk(&self.est.state);
            norm(sub(fk.t[self.h.j.l_shoulder], fk.t[self.h.j.r_shoulder])) as f32
        };
        let mut skeleton =
            output::source_skeleton(&self.h, &self.est, frame_index, intr_cam, ref_span);
        // Carry the perception-side channels the expression solver and GUI
        // still read from the base pipeline.
        skeleton.expressions = base.skeleton.expressions;
        skeleton.face_mesh_confidence = base.skeleton.face_mesh_confidence;
        skeleton.face = base.skeleton.face;
        skeleton.capture_timestamp_ms = ts_ms;
        skeleton.rig = Some(Arc::new(rig));
        self.last_solve_ms = t0.elapsed().as_secs_f32() * 1000.0;
        ph.output_ms = t_output.elapsed().as_secs_f32() * 1000.0;
        self.last_timings = ph;
        if crate::tracking::debug_channel::enabled() {
            let d = self.est.diag;
            let t = &self.last_timings;
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
                "phases": {
                    "hint": t.hint_ms, "rtmw": t.rtmw_ms, "vis": t.vis_ms,
                    "hands": t.hands_ms, "head": t.head_ms, "dense": t.dense_ms,
                    "facefit": t.facefit_ms, "output": t.output_ms,
                    "acc": t.acc_ms, "eval": t.eval_ms, "lin": t.lin_ms,
                    "main": t.main_ms, "seeds": t.seeds_ms,
                    "reacc": t.reacc_ms, "finish": t.finish_ms,
                },
            }));
        }

        Some(PoseEstimate {
            skeleton,
            annotation: base.annotation,
        })
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
                super::estimator::ModelPoint::Joint(j) => h.model.joints[j].name.to_string(),
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
                super::estimator::ModelPoint::Joint(j) => h.model.joints[j].name.to_string(),
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
            for (ci, c) in m.capsules.iter().enumerate() {
                let pa = |p: PointRef| match p {
                    PointRef::Joint(j) => m.joints[j].name.to_string(),
                    PointRef::Site(s) => m.sites[s].name.to_string(),
                };
                let (a, b) = (fk.point(c.a), fk.point(c.b));
                eprintln!(
                    "  cap{ci:2} {:?}:{}-{} a=({:.3},{:.3},{:.3}) b=({:.3},{:.3},{:.3}) r={:.3} allow={}",
                    c.part, pa(c.a), pa(c.b), a[0], a[1], a[2], b[0], b[1], b[2],
                    m.capsule_radius(&est.state, c),
                    obs.surf_allow.get(ci).copied().unwrap_or(true)
                );
            }
            let assoc = &est.last_surf_assoc;
            for (pi, (pt, sg)) in obs.surface.iter().enumerate() {
                let chosen = assoc.get(pi).copied().unwrap_or(-2);
                let mut best = (usize::MAX, f64::INFINITY);
                for (ci, c) in m.capsules.iter().enumerate() {
                    let (q, _u) =
                        super::estimator::closest_on_segment(fk.point(c.a), fk.point(c.b), *pt);
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
                    "    surf ({:.3},{:.3},{:.3}) σ={:.3} → {name} d={:+.3} est={chosen}",
                    pt[0], pt[1], pt[2], sg, best.1
                );
            }
        }
        eprintln!(
            "  depth frame: {:?} valid@nose {:?}",
            depth
                .as_ref()
                .map(|d| (d.width, d.height, d.points_m.len())),
            obs.kp2d
                .first()
                .and_then(|k| depth.as_ref().and_then(|d| window_point(
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

#[cfg(all(test, feature = "inference"))]
mod live_pipeline_tests {
    use super::*;

    /// Smoke test for the pipelined (Remote) detector path: thread spawn +
    /// model load, the latest-result protocol (Some / None), the
    /// synchronous protocol, the reset round trip, and clean shutdown.
    /// `#[ignore]`d — it loads the full RTMW3D model (~370 MB, DirectML
    /// init); run explicitly with
    /// `cargo test --features realsense live_pipeline -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn remote_detector_pipeline_produces_estimates() {
        let mut provider = FusionProvider::from_models_dir_live(
            "models",
            crate::tracking::provider::TrackingPipelineConfig::default(),
        )
        .expect("live provider construction");

        // A protocol walk paced like the camera (~30 fps): the detector
        // needs its ~30 ms budget per frame, and without pacing the loop
        // every submission would supersede the previous job before it
        // runs — that coalescing is by design.
        let rgb = vec![90u8; 640 * 480 * 3];
        let mut got = 0;
        let mut skipped = 0;
        for i in 0..24u64 {
            std::thread::sleep(std::time::Duration::from_millis(33));
            match provider.estimate_pose_latest(&rgb, 640, 480, i) {
                Some(est) => {
                    got += 1;
                    let _ = est.skeleton.capture_timestamp_ms;
                }
                None => skipped += 1,
            }
            if got >= 3 {
                break;
            }
        }
        assert!(got >= 3, "pipelined path produced only {got} estimates ({skipped} skips)");

        // Synchronous protocol must consume exactly its own frame.
        let est = provider.estimate_pose(&rgb, 640, 480, 900);
        let _ = est;

        // Reset round trip must not deadlock (ack via the result cell).
        provider.reset_temporal_state();

        // Post-reset the pipeline still produces estimates.
        let mut got_after = 0;
        for i in 1_000..1_024u64 {
            std::thread::sleep(std::time::Duration::from_millis(33));
            if provider.estimate_pose_latest(&rgb, 640, 480, i).is_some() {
                got_after += 1;
            }
            if got_after >= 2 {
                break;
            }
        }
        assert!(got_after >= 2, "pipeline dead after reset");

        // Drop joins the detector thread; if it deadlocked this hangs.
        drop(provider);
    }
}
