//! YOLO26-pose detector — the production single-stage whole-frame person
//! + COCO-17 keypoint model (replaced RTMW3D, 2026-09).
//!
//! One DirectML pass over the whole frame yields the person box and the
//! COCO-17 body keypoints; no YOLOX stage and no self-tracking crop (the
//! detector IS whole-frame, so there is no crop feedback loop to damp).
//! The output is shaped exactly like RTMW3D's downstream contract:
//!
//! * `DetectorAux.joints` — 133 COCO-Wholebody entries with the body 17
//!   filled from YOLO26 and every other block (foot / face-68 / both
//!   hand blocks) at score 0. The fusion estimator's gates already
//!   treat zero-score entries as absent, the depth-lift table only
//!   reads body indices (+ hand-block wrists, which are gone), and the
//!   hand-crop chain falls back to the model-predicted crops.
//! * `DetectorAux.face_mesh` — the same MediaPipe FaceMesh + Blendshape
//!   cascade as RTMW3D, fed a bbox derived from YOLO26's five face
//!   points (nose, eyes, ears) instead of the face-68 block.
//! * The σ the estimator consumes is synthesised (YOLO26 regresses
//!   directly — no SimCC posterior): a constant pixel σ scaled by the
//!   same score-inflation term RTMW3D's calibrated visibility uses.
//!
//! Comparison bench (speed + keypoint agreement vs RTMW3D) lives in
//! `diagnose_yolo26_pose`; the acceptance gate is the fusion replay
//! bench (`scratchpad/bench_p0.sh`).

use std::path::Path;

use ort::session::Session;
use ort::value::TensorRef;

use super::super::face_mediapipe::{derive_face_bbox, FaceBbox, FaceMeshInference};
use super::session::build_session;
use super::yolo_pose::{anchor_box, anchor_keypoints, best_pose_anchor, letterbox, pose_anchor_count, pose_channels};
use super::{
    annotation, decode::DecodedJoint, face, DetectorAux, DetectorOptions, InferenceBackend,
};
use crate::tracking::{PoseEstimate, SourceSkeleton};
use log::{debug, info, warn};

/// Synthesised localisation σ for YOLO26 keypoints, in frame pixels
/// BEFORE the estimator's `simcc_gain` (1.9) — the same point where a
/// sharp RTMW3D SimCC peak lands at ~1–2 px. Regression heads are
/// genuinely less precise than heatmaps, so the default sits above that;
/// `VULVATAR_YOLO_SIGMA_PX` sweeps it on the replay bench.
const DEFAULT_SIGMA_PX: f64 = 3.5;

/// Calibration factor applied to YOLO26's raw per-joint confidence before
/// it feeds the visibility logistic (as `score` AND `zscore`). 1.0 — the
/// calibration work is done by the zscore substitution (see below); this
/// knob stays for bench sweeps.
const DEFAULT_SCORE_GAIN: f64 = 1.0;

/// Extra inflation on the face bbox derived from YOLO26's five face
/// points. RTMW3D derives its crop from the face-68 spread (chin to
/// forehead ≈ face height); YOLO26's points span roughly the ear-to-ear
/// width only, so the box needs ~1.5× more pad to cover the full face.
const FACE_POINT_PAD_EXTRA: f32 = 1.55;

/// Minimum confident face points (of nose / 2 eyes / 2 ears) to build a
/// FaceMesh crop from.
const FACE_POINTS_MIN: usize = 4;

pub(crate) struct Yolo26PoseInference {
    session: Session,
    input_name: String,
    output_name: String,
    /// Square model input size (from the export filename convention).
    size: u32,
    face_mesh: Option<FaceMeshInference>,
    face_selector: face::FaceSourceSelector,
    frame_timestamp_ms: Option<f64>,
    frame_dt: crate::tracking::metric_frame::FrameDtTracker,
    /// Previous frame's smoothed keypoints (frame pixels) — a light EMA
    /// that substitutes for the temporal stability RTMW3D gets from its
    /// self-tracking crop. Single-frame regression jitters by several px
    /// per frame; at a frame border that flickers observations in/out of
    /// the 2 % border skip in bursts (measured: 0.5+ m wrist snaps on
    /// desk sessions where the hand sits at the bottom edge). A joint
    /// that moved more than `SMOOTH_RESET_PX` between frames takes the
    /// fresh value unblended, so real fast motion passes through.
    prev_kps: Option<[[f32; 3]; 17]>,
    load_warnings: Vec<String>,
    backend: InferenceBackend,
    last_aux: Option<DetectorAux>,
}

/// EMA weight of the PREVIOUS frame's keypoint (0 = no smoothing).
const SMOOTH_ALPHA: f32 = 0.75;
/// Per-frame displacement (px) above which the EMA resets for that joint.
const SMOOTH_RESET_PX: f32 = 64.0;

impl Yolo26PoseInference {
    pub(crate) fn from_models_dir_with_options(
        models_dir: impl AsRef<Path>,
        opts: DetectorOptions,
    ) -> Result<Self, String> {
        let model_path = models_dir.as_ref().join("yolo26-pose.onnx");
        let model_path = if model_path.exists() {
            model_path
        } else {
            // Fall back through the export-filename convention used by
            // Export-filename convention (prefer the fastest): the
            // trailing _480/_640 is the square input side the loader
            // derives its letterbox size from.
            let candidates = [
                "yolo26n-pose_480.onnx",
                "yolo26s-pose_480.onnx",
                "yolo26n-pose_640.onnx",
                "yolo26s-pose_640.onnx",
            ];
            let mut found = None;
            for c in candidates {
                let p = models_dir.as_ref().join(c);
                if p.exists() {
                    found = Some(p);
                    break;
                }
            }
            found.ok_or_else(|| {
                "YOLO26-pose model not found in models dir (expected yolo26-pose.onnx \
                 or yolo26n-pose_*.onnx)"
                    .to_string()
            })?
        };

        let (session, backend) = if opts.force_cpu {
            super::session::build_session_cpu_only(&model_path.to_string_lossy(), 4, "YOLO26-pose")?
        } else {
            build_session(&model_path.to_string_lossy(), 4, "YOLO26-pose")?
        };
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "images".to_string());
        let output_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .unwrap_or_else(|| "output0".to_string());
        // Square input side from the export filename (`_480.onnx` etc.).
        let size: u32 = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.rsplit('_').next())
            .and_then(|s| s.parse().ok())
            .unwrap_or(640);

        let mut load_warnings = Vec::new();
        let face_mesh = match FaceMeshInference::try_from_models_dir(&models_dir, opts.face_ep) {
            Ok(opt) => opt,
            Err(e) => {
                let msg = format!("Face inference failed to load: {e}. Expressions disabled.");
                warn!("{}", msg);
                load_warnings.push(msg);
                None
            }
        };

        info!(
            "YOLO26-pose loaded {} (input '{}x{}', output '{}'), backend {}",
            model_path.display(),
            input_name,
            size,
            output_name,
            backend.label()
        );

        Ok(Self {
            session,
            input_name,
            output_name,
            size,
            face_mesh,
            face_selector: face::FaceSourceSelector::default(),
            frame_timestamp_ms: None,
            frame_dt: crate::tracking::metric_frame::FrameDtTracker::default(),
            prev_kps: None,
            load_warnings,
            backend,
            last_aux: None,
        })
    }

    /// Execution provider the session landed on (GUI status surface).
    pub(crate) fn backend(&self) -> &InferenceBackend {
        &self.backend
    }

    pub(crate) fn take_load_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.load_warnings)
    }

    /// Device capture timestamp for the NEXT `estimate_pose` (drives the
    /// face-source crossfade dt via `frame_dt`).
    pub(crate) fn set_frame_timestamp_ms(&mut self, ts_ms: Option<f64>) {
        self.frame_timestamp_ms = ts_ms;
    }

    pub(crate) fn take_aux(&mut self) -> Option<DetectorAux> {
        self.last_aux.take()
    }

    pub(crate) fn reset_temporal_state(&mut self) {
        self.frame_timestamp_ms = None;
        self.frame_dt.reset();
        self.prev_kps = None;
    }

    pub(crate) fn estimate_pose(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> PoseEstimate {
        let t_total = std::time::Instant::now();
        let dt_s = self.frame_dt.tick(self.frame_timestamp_ms);

        let (person, mut kps17, det_ms) = self.detect(rgb_data, width, height);
        // Temporal EMA (see `prev_kps`): blend with the previous frame,
        // resetting per joint on a large displacement.
        if let Some(kps) = kps17.as_mut() {
            if let Some(prev) = self.prev_kps {
                for (i, slot) in kps.iter_mut().enumerate() {
                    let dx = slot[0] - prev[i][0];
                    let dy = slot[1] - prev[i][1];
                    if dx.hypot(dy) >= SMOOTH_RESET_PX {
                        continue;
                    }
                    slot[0] = SMOOTH_ALPHA * prev[i][0] + (1.0 - SMOOTH_ALPHA) * slot[0];
                    slot[1] = SMOOTH_ALPHA * prev[i][1] + (1.0 - SMOOTH_ALPHA) * slot[1];
                    slot[2] = SMOOTH_ALPHA * prev[i][2] + (1.0 - SMOOTH_ALPHA) * slot[2];
                }
            }
            self.prev_kps = Some(*kps);
        } else {
            self.prev_kps = None;
        }
        let aspect = width as f32 / height.max(1) as f32;

        // 133-entry COCO-Wholebody array with only the body block filled.
        let mut joints = vec![DecodedJoint::default(); 133];
        if let Some(ref kps) = kps17 {
            let w = width as f32;
            let h = height as f32;
            let sigma_px = std::env::var("VULVATAR_YOLO_SIGMA_PX")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(DEFAULT_SIGMA_PX);
            // YOLO26's per-joint confidence is a raw head sigmoid — badly
            // OVERconfident on occluded joints (a wrist below the desk
            // still reads 0.7–0.9, where RTMW3D's calibrated visibility
            // drops under the gates). The depth-lift 3D table then trusts
            // garbage pixels at a fixed 2 cm σ → 0.5+ m wrist snaps
            // (measured: bench sessions s1789246071/74/60). The gain
            // recalibrates onto RTMW3D's score scale; it feeds both the
            // score gates and the σ score-inflation term downstream.
            let score_gain = std::env::var("VULVATAR_YOLO_SCORE_GAIN")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(DEFAULT_SCORE_GAIN);
            // body_kp2d computes σ_px = sx * frame_w * simcc_gain(1.9);
            // invert that so the synthesised σ lands on `sigma_px`.
            let sx = (sigma_px / 1.9 / w as f64) as f32;
            for (i, slot) in joints.iter_mut().take(17).enumerate() {
                let [x, y, c] = kps[i];
                let score = ((c as f64 * score_gain) as f32).min(1.0);
                // RAW normalised coords, deliberately NOT clamped to [0,1]:
                // the provider's out-of-frame gate reads `nx`/`ny` against a
                // ±5% tolerance and zeroes keypoints the detector hallucinate
                // past the frame edge (measured RTMW3D-era: elbows/wrists at
                // nx 1.12 with sharp scores). Clamping here would pin those
                // exactly ON the border — inside the tolerance, invisible to
                // the gate — so they'd carry their overconfident score into
                // the solve and onto the preview wipe as border dots.
                *slot = DecodedJoint {
                    nx: x / w,
                    ny: y / h,
                    nz: 0.0,
                    score,
                    sx,
                    sy: sx,
                    // The provider's visibility calibration (`VisPolicy::
                    // p_vis`, a logistic over SimCC peak features) is the
                    // FINAL authority on scores — it replaces this score
                    // wholesale. Its dominant feature is `zscore` (weight
                    // −26.6): RTMW3D's visible z-peaks carry ~0.9 there,
                    // cancelling the raw-score term, and occluded ones sit
                    // near 0. YOLO26 has no z head; leaving zscore at 0
                    // made EVERY joint look maximally visible (p_vis ≈ 1.0
                    // regardless of confidence — measured as 0.5+ m wrist
                    // snaps on desk sessions where RTMW3D stays silent),
                    // so the kpt confidence stands in for it. p_vis then
                    // saturates around conf ≈ 0.55 and the gates/score
                    // inflation behave as on the calibrated RTMW scale.
                    zscore: score,
                    ..DecodedJoint::default()
                };
            }
        }

        self.last_aux = Some(DetectorAux {
            joints: joints.clone(),
            face_mesh: None,
            crop: person.map(|p| (p[0], p[1], p[2] - p[0], p[3] - p[1])),
        });

        let mut skeleton = SourceSkeleton::empty(frame_index);
        skeleton.overall_confidence = {
            let mut scores: Vec<f32> = joints.iter().take(17).map(|j| j.score).collect();
            scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
            scores.get(scores.len() / 2).copied().unwrap_or(0.0)
        };
        skeleton.face = face::derive_face_pose_from_body(&joints, aspect);
        let body_face_pose = skeleton.face;

        // Face cascade: bbox from the five body-frame face points.
        let mut mesh_conf = 0.0f32;
        let mut mesh_face_pose: Option<crate::tracking::FacePose> = None;
        if let Some(face_mesh) = self.face_mesh.as_mut() {
            if let Some(bbox) = build_face_bbox_from_body(&joints, width, height) {
                if let Some((exprs, conf, pose)) =
                    face_mesh.estimate(rgb_data, width, height, &bbox)
                {
                    skeleton.expressions = exprs;
                    skeleton.face_mesh_confidence = Some(conf);
                    mesh_conf = conf;
                    mesh_face_pose = pose;
                    if let Some(aux) = self.last_aux.as_mut() {
                        // take_landmarks_px drains; re-run only when the
                        // mesh produced a face this frame.
                        if let Some(lm) = face_mesh.take_landmarks_px() {
                            aux.face_mesh = Some(lm);
                        }
                    }
                }
            }
        }
        skeleton.face =
            self.face_selector
                .select(body_face_pose, mesh_face_pose, mesh_conf, dt_s);

        let mut detection = annotation::build_annotation(&joints);
        detection.bounding_box = person.map(|p| {
            (
                p[0] / width as f32,
                p[1] / height as f32,
                p[2] / width as f32,
                p[3] / height as f32,
            )
        });

        debug!(
            "YOLO26-pose timing: total={:>5.1}ms det={:>5.1}ms person={}",
            t_total.elapsed().as_secs_f32() * 1000.0,
            det_ms,
            person.is_some(),
        );

        PoseEstimate {
            annotation: detection,
            skeleton,
        }
    }

    /// Letterbox → run → decode → best person. Returns the person box in
    /// frame pixels `(x0, y0, x1, y1)` and the COCO-17 keypoints in frame
    /// pixels.
    fn detect(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> (Option<[f32; 4]>, Option<[[f32; 3]; 17]>, f32) {
        let t0 = std::time::Instant::now();
        let tensor = letterbox(rgb, width, height, self.size);
        let r = (self.size as f32 / width as f32).min(self.size as f32 / height as f32);
        let pad_x = (self.size as f32 - width as f32 * r) / 2.0;
        let pad_y = (self.size as f32 - height as f32 * r) / 2.0;
        let input = match TensorRef::from_array_view(&tensor) {
            Ok(v) => v,
            Err(e) => {
                warn!("YOLO26-pose: tensor view failed: {e}");
                return (None, None, t0.elapsed().as_secs_f32() * 1000.0);
            }
        };
        let outputs = match self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
        {
            Ok(o) => o,
            Err(e) => {
                warn!("YOLO26-pose: run failed: {e}");
                return (None, None, t0.elapsed().as_secs_f32() * 1000.0);
            }
        };
        let Some(out) = outputs.get(self.output_name.as_str()) else {
            warn!("YOLO26-pose: output '{}' missing", self.output_name);
            return (None, None, t0.elapsed().as_secs_f32() * 1000.0);
        };
        let Ok((_, data)) = out.try_extract_tensor::<f32>() else {
            warn!("YOLO26-pose: output not f32");
            return (None, None, t0.elapsed().as_secs_f32() * 1000.0);
        };
        let Some(anchors) = pose_anchor_count(data.len(), 17) else {
            warn!(
                "YOLO26-pose: unexpected output size {} (expected {}×A)",
                data.len(),
                pose_channels(17)
            );
            return (None, None, t0.elapsed().as_secs_f32() * 1000.0);
        };

        // Best-scoring anchor above the box threshold; a single-user desk
        // pipeline wants no NMS dance — take the argmax and require a
        // sane score.
        let det_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let Some((a, _)) = best_pose_anchor(&data, anchors, 0.25) else {
            return (None, None, det_ms);
        };
        let person = anchor_box(&data, anchors, a, pad_x, pad_y, r);
        let kps: Option<[[f32; 3]; 17]> =
            anchor_keypoints(&data, anchors, a, 17, pad_x, pad_y, r)
                .try_into()
                .ok();
        (Some(person), kps, det_ms)
    }
}

/// Face bbox from YOLO26's body-frame face points (nose / eyes / ears),
/// inflated past the point spread to cover chin + forehead the way the
/// face-68 spread does for the RTMW3D path.
fn build_face_bbox_from_body(
    joints: &[DecodedJoint],
    width: u32,
    height: u32,
) -> Option<FaceBbox> {
    let pts: Vec<(f32, f32)> = joints
        .iter()
        .take(5)
        .filter(|j| j.score >= 0.3)
        .map(|j| (j.nx * width as f32, j.ny * height as f32))
        .collect();
    if pts.len() < FACE_POINTS_MIN.min(5) {
        return None;
    }
    let mut bbox = derive_face_bbox(&pts, width, height)?;
    // Widen around the centre: the 5-point spread misses chin/forehead.
    let cx = bbox.x + bbox.size / 2.0;
    let cy = bbox.y + bbox.size / 2.0;
    bbox.size *= FACE_POINT_PAD_EXTRA;
    bbox.x = cx - bbox.size / 2.0;
    bbox.y = cy - bbox.size / 2.0;
    Some(bbox)
}
