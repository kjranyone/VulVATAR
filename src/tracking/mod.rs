#[cfg(feature = "realsense")]
use crate::t;
use log::{error, warn};
// `info!` only fires from the realsense capture loop; gate the import so a
// no-capture-backend build doesn't warn on it being unused.
#[cfg(feature = "realsense")]
use log::info;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[cfg(feature = "realsense")]
pub mod realsense;
/// Full-rate raw-capture recorder (colour + aligned depth + intrinsics) for
/// offline estimator evaluation. Needs the D435 frame type, hence the gate.
#[cfg(feature = "realsense")]
pub mod sequence_recorder;

pub(crate) mod latest_cell;
mod pose_estimation;
pub mod provider;
pub mod rtmw3d_with_depth;
pub mod stagelog;
#[cfg(feature = "inference")]
pub mod skeleton_from_depth;

pub mod auto_neutral;
pub mod calibration;
pub mod debug_channel;
pub mod face_mediapipe;
pub mod hand_hold;
pub mod rtmw3d;
pub mod session_record;
pub mod source_skeleton;
#[cfg(feature = "inference")]
pub mod yolox;

pub use calibration::{
    rotate_xz, shoulder_line_yaw, shoulder_span_plausible, CalibrationMode, PoseCalibration,
    TorsoDepthTemplate, BODY_YAW_MAX_RAD, BODY_YAW_MIN_SAMPLES, BODY_YAW_WARN_RAD,
    SHOULDER_SPAN_MAX_M, SHOULDER_SPAN_MIN_M,
};
pub use source_skeleton::{
    CameraIntrinsics, FacePose, FaceSource, MetricFrameInfo, SourceExpression, SourceJoint,
    SourceSkeleton,
};

/// Smoothing / threshold params consumed by
/// [`crate::avatar::pose_solver::solve_avatar_pose`] via
/// [`SolverParams`](crate::avatar::pose_solver::SolverParams).
///
/// The [`Default`] values below are tuned to lean on the upstream 1€ filter
/// and structural keypoint floors (see the `Default` impl), so most users
/// never need to touch these. They are now surfaced in the Tracking
/// inspector's *Advanced smoothing* section for the per-camera cases the
/// defaults don't cover (noisy confidence floors, jittery expression
/// rigs): the GUI holds the live values on `TrackingGuiState::smoothing`
/// and passes them through `FrameConfig::smoothing` each frame. The GUI
/// does not expose `stale_timeout_nanos` — it is a hold-policy timing knob,
/// not a smoothing control — so it always keeps its default.
#[derive(Clone, Debug)]
pub struct TrackingSmoothingParams {
    /// Per-frame blend factor toward the new rotation. Maps directly to
    /// `SolverParams::rotation_blend`.
    pub rotation_blend: f32,
    /// Per-frame blend factor toward new expression weights.
    pub expression_blend: f32,
    /// Minimum keypoint confidence for a joint to drive a bone.
    pub joint_confidence_threshold: f32,
    /// Minimum face-pose confidence for the head to react.
    pub face_confidence_threshold: f32,
    pub stale_timeout_nanos: u64,
}

impl Default for TrackingSmoothingParams {
    fn default() -> Self {
        // `rotation_blend = 1.0` snaps each frame straight to the
        // direction-matched output. The 1€ filter on joint positions
        // (`pose_solver::preprocess_source`) already smooths jitter
        // adaptively, so a separate per-frame rotation LPF on top
        // just adds blanket lag. `joint_confidence_threshold = 0.0`
        // delegates noise gating to the structural floors that already
        // run upstream (`KEYPOINT_VISIBILITY_FLOOR`, hip/shoulder
        // origin choice, hand-MCP quorum, wrist anatomy check).
        // `expression_blend` stays smoothed because there is no 1€ on
        // expression weights — without this LPF, ARKit blendshapes
        // chatter visibly.
        Self {
            rotation_blend: 1.0,
            expression_blend: 0.8,
            joint_confidence_threshold: 0.0,
            face_confidence_threshold: 0.0,
            stale_timeout_nanos: 200_000_000,
        }
    }
}

/// Which signal drives the avatar's mouth visemes (`aa`/`ih`/`ou`/`ee`/`oh`).
/// Audio lip-sync and the camera (FaceMesh) both produce mouth shapes; this
/// selects how they combine so the camera-based path doesn't silently
/// override audio (or vice-versa). Only the mouth visemes are affected —
/// eyes / brows / emotions always come from the camera. Consumed by
/// [`crate::avatar::pose_solver::solve_expressions`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MouthSource {
    /// Audio lip-sync only — the camera's mouth visemes are ignored.
    Audio,
    /// Camera (FaceMesh / "image lip-sync") only — audio is ignored.
    Image,
    /// Whichever is stronger per viseme (`max`) — mouth opens for speech
    /// *or* a visibly open mouth. The default.
    #[default]
    Both,
}

impl MouthSource {
    /// Stable index for GUI combo boxes / persistence. 0=Audio, 1=Image, 2=Both.
    pub fn to_index(self) -> usize {
        match self {
            MouthSource::Audio => 0,
            MouthSource::Image => 1,
            MouthSource::Both => 2,
        }
    }
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => MouthSource::Audio,
            1 => MouthSource::Image,
            _ => MouthSource::Both,
        }
    }
}

/// Per-session calibration.
///
/// `pose` — the user runs the `Calibrate Pose ▼` modal once per project
/// to capture a known-good pelvic / shoulder anchor reference; the
/// solver uses it to seed the root-translation EMA and
/// `skeleton_from_depth` uses it to clamp the metric calibration scale
/// against a desk-in-foreground bias. The same capture window also
/// records the per-person neutral expression baseline and the resting
/// head pose (`neutral_face_ypr`), both subtracted from live samples by
/// [`Self::apply_calibration`]. `None` until the user runs the modal —
/// consumers fall back to the existing auto-EMA / hardcoded clamp
/// behaviour. See `docs/calibration-ux.md`.
#[derive(Clone, Debug, Default)]
pub struct TrackingCalibration {
    pub pose: Option<PoseCalibration>,
}

impl TrackingCalibration {
    /// Apply the solve-time parts of the calibration to a clone of the
    /// published sample: the neutral-body-yaw scene de-rotation, the
    /// per-source neutral face-pose subtraction, and the neutral
    /// expression rescale. The solver then receives deltas relative to
    /// the user's calibrated setup, not absolute camera geometry.
    ///
    /// The *anchor* part of the pose calibration is intentionally not
    /// applied here — anchor forcing and EMA seeding are consumed by
    /// the depth-aware skeleton builder and the solver directly from
    /// the calibration record. The mailbox always carries the RAW
    /// sample; only the solve path sees the values transformed here,
    /// so the calibration modal's recapture loop measures true camera
    /// geometry (never a previously-corrected frame).
    pub fn apply_calibration(&self, sample: &mut SourceSkeleton) {
        // Neutral body yaw (oblique camera placement): rigidly
        // re-express the published sample "as if the camera had been
        // frontal" by rotating the horizontal plane by the calibrated
        // shoulder-line yaw. One uniform transform instead of per-consumer
        // subtraction: the solver's shoulder-yaw, direction-matched bones
        // and arm-IK wrist targets all read the same de-rotated scene, so
        // the "pelvis forward, arms chasing oblique targets" twist class
        // of bug can't exist. See docs/calibration-ux.md, "Neutral body yaw".
        //
        // Joints / fingertips are anchor-centred (the depth builder puts
        // the anchor at the joint-space origin), so their pivot is the
        // origin. `root_offset` is an absolute source-oriented metric
        // position, so it pivots about the calibrated anchor — deviations
        // from neutral rotate, the neutral point itself stays put.
        //
        // The face pose is deliberately NOT rotated: head neutrality is
        // owned by `neutral_face_ypr_*` below (per-estimator residuals);
        // rotating the face AND subtracting its neutral would double-count
        // the camera angle. Metric-only: rotating a monocular z≈0 offset
        // would fabricate depth from nothing, and no yaw is ever captured
        // on that path anyway (`shoulder_line_yaw` refuses).
        if let Some(pose) = self.pose.as_ref() {
            if let Some(theta) = pose.neutral_body_yaw {
                if sample.metric_frame_info.is_some() && theta != 0.0 {
                    for j in sample.joints.values_mut() {
                        j.position = calibration::rotate_xz(j.position, theta);
                    }
                    for t in sample.fingertips.values_mut() {
                        t.position = calibration::rotate_xz(t.position, theta);
                    }
                    for hand in [
                        sample.left_hand_orientation.as_mut(),
                        sample.right_hand_orientation.as_mut(),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        hand.forward = calibration::rotate_xz(hand.forward, theta);
                        hand.up = calibration::rotate_xz(hand.up, theta);
                    }
                    if let Some(offset) = sample.root_offset.as_mut() {
                        // Same [x, y, −depth] convention the solver's
                        // root-reference seed uses (`anchor_depth_m` is
                        // stored as positive camera-forward metres,
                        // source z is negative-forward).
                        let a0 = [
                            pose.anchor_x,
                            pose.anchor_y,
                            -pose.anchor_depth_m.unwrap_or(0.0),
                        ];
                        let d = calibration::rotate_xz(
                            [
                                offset[0] - a0[0],
                                offset[1] - a0[1],
                                offset[2] - a0[2],
                            ],
                            theta,
                        );
                        *offset = [a0[0] + d[0], a0[1] + d[1], a0[2] + d[2]];
                    }
                }
            }
        }
        // Resting head pose captured during the calibration hold: the
        // camera is rarely dead-ahead of where the user actually looks
        // (monitor offset, desk mount), so the absolute face angles
        // carry a constant turn/tilt. Subtracting the calibrated
        // neutral maps the user's habitual posture to avatar-forward.
        //
        // Keyed on the live pose's own source: the mesh and body
        // estimators have different systematic residuals, so the
        // neutral is captured per source and only the matching one is
        // subtracted. A source whose neutral wasn't captured (too few
        // confident frames during the hold) gets no extra subtraction —
        // its hardcoded anatomical neutral inside the estimator still
        // applies, which is exactly the pre-calibration behaviour.
        if let (Some(face), Some(pose)) = (sample.face.as_mut(), self.pose.as_ref()) {
            // An uncaptured neutral means "subtract nothing" — encoded
            // as zeros so the crossfade interpolation below stays a
            // plain lerp between the two sources' effective baselines.
            let neutral_of = |source: FaceSource| -> [f32; 3] {
                pose.neutral_face_ypr_for(source).unwrap_or([0.0; 3])
            };
            let target = neutral_of(face.source);
            let neutral = match face.blend {
                // Mid-crossfade: the raw angles are a lerp of the two
                // sources' raw spaces, so the subtracted neutral must
                // be the SAME lerp of the two neutrals. Lerp is linear,
                // so this equals blending in calibrated space —
                // subtracting the target's neutral outright would
                // re-introduce the inter-source neutral gap as a head
                // step on the exact frame the crossfade exists to hide.
                Some((from_source, t)) => {
                    let from = neutral_of(from_source);
                    [
                        from[0] + (target[0] - from[0]) * t,
                        from[1] + (target[1] - from[1]) * t,
                        from[2] + (target[2] - from[2]) * t,
                    ]
                }
                None => target,
            };
            face.yaw -= neutral[0];
            face.pitch -= neutral[1];
            face.roll -= neutral[2];
        }
        // Per-person neutral expression baseline (captured during pose
        // calibration when face tracking was on). Subtract the resting
        // weight and rescale the remaining headroom so a face that rests
        // with a slightly open mouth / narrowed eyes maps to neutral and
        // still reaches a full open/blink at the top of its range:
        //   w' = clamp((w − neutral) / (1 − neutral), 0, 1)
        // The neutral is floored away from 1 so the divisor stays sane
        // for an expression whose baseline is implausibly high.
        if let Some(ref pose) = self.pose {
            if !pose.neutral_expressions.is_empty() {
                for expr in sample.expressions.iter_mut() {
                    if let Some((_, neutral)) = pose
                        .neutral_expressions
                        .iter()
                        .find(|(name, _)| *name == expr.name)
                    {
                        let n = neutral.clamp(0.0, 0.95);
                        expr.weight = ((expr.weight - n) / (1.0 - n)).clamp(0.0, 1.0);
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TrackingErrorLevel {
    Warning,
    Blocking,
}

#[cfg(test)]
mod calibration_apply_tests {
    use super::*;
    use crate::tracking::source_skeleton::SourceExpression;

    fn cal_with_neutral(neutral: Vec<(String, f32)>) -> TrackingCalibration {
        TrackingCalibration {
            pose: Some(PoseCalibration {
                mode: crate::tracking::CalibrationMode::UpperBody,
                captured_at: String::new(),
                captured_at_unix: 0,
                frame_count: 1,
                anchor_x: 0.0,
                anchor_y: 0.0,
                anchor_depth_m: None,
                confidence: 1.0,
                anchor_depth_jitter_m: None,
                shoulder_span_m: None,
                x_range_observed: None,
                z_range_observed: None,
                torso_depth_template: None,
                neutral_expressions: neutral,
                neutral_face_ypr_mesh: None,
                neutral_face_ypr_body: None,
                neutral_body_yaw: None,
            }),
        }
    }

    fn skeleton_with(name: &str, weight: f32) -> SourceSkeleton {
        let mut sk = SourceSkeleton::default();
        sk.expressions = vec![SourceExpression {
            name: name.into(),
            weight,
        }];
        sk
    }

    #[test]
    fn neutral_baseline_subtracts_and_rescales() {
        let cal = cal_with_neutral(vec![("aa".into(), 0.2)]);

        // Resting mouth (raw == neutral) collapses to fully closed.
        let mut sk = skeleton_with("aa", 0.2);
        cal.apply_calibration(&mut sk);
        assert!(sk.expressions[0].weight.abs() < 1e-6, "rest → 0");

        // Full open survives the rescale unchanged.
        let mut sk = skeleton_with("aa", 1.0);
        cal.apply_calibration(&mut sk);
        assert!((sk.expressions[0].weight - 1.0).abs() < 1e-6, "full → 1");

        // Mid value rescales over the remaining headroom:
        // (0.6 - 0.2) / (1 - 0.2) = 0.5.
        let mut sk = skeleton_with("aa", 0.6);
        cal.apply_calibration(&mut sk);
        assert!((sk.expressions[0].weight - 0.5).abs() < 1e-6);

        // Below baseline clamps at 0 rather than going negative.
        let mut sk = skeleton_with("aa", 0.1);
        cal.apply_calibration(&mut sk);
        assert!(sk.expressions[0].weight.abs() < 1e-6, "below rest → 0");
    }

    #[test]
    fn expression_without_baseline_passes_through() {
        let cal = cal_with_neutral(vec![("aa".into(), 0.2)]);
        let mut sk = skeleton_with("blink", 0.7);
        cal.apply_calibration(&mut sk);
        assert!((sk.expressions[0].weight - 0.7).abs() < 1e-6);
    }

    #[test]
    fn neutral_face_pose_is_subtracted_for_matching_source() {
        let mut cal = cal_with_neutral(Vec::new());
        cal.pose.as_mut().unwrap().neutral_face_ypr_mesh = Some([-0.6, 0.4, 0.1]);
        let mut sk = SourceSkeleton::default();
        sk.face = Some(FacePose {
            yaw: -0.6,
            pitch: 0.4,
            roll: 0.1,
            confidence: 0.9,
            source: crate::tracking::FaceSource::Mesh,
            ..Default::default()
        });
        cal.apply_calibration(&mut sk);
        let f = sk.face.unwrap();
        // The habitual monitor-gaze pose maps to avatar-forward.
        assert!(f.yaw.abs() < 1e-6);
        assert!(f.pitch.abs() < 1e-6);
        assert!(f.roll.abs() < 1e-6);
        assert!((f.confidence - 0.9).abs() < 1e-6, "confidence untouched");
    }

    #[test]
    fn neutral_from_one_source_never_touches_the_other() {
        // The core of the per-source split: a mesh-measured neutral
        // subtracted from a body-sourced pose would inject the
        // inter-estimator residual as a head step on every runtime
        // source switch.
        let mut cal = cal_with_neutral(Vec::new());
        cal.pose.as_mut().unwrap().neutral_face_ypr_mesh = Some([-0.6, 0.4, 0.1]);
        let mut sk = SourceSkeleton::default();
        sk.face = Some(FacePose {
            yaw: 0.2,
            pitch: -0.1,
            roll: 0.05,
            confidence: 0.9,
            source: crate::tracking::FaceSource::Body,
            ..Default::default()
        });
        cal.apply_calibration(&mut sk);
        let f = sk.face.unwrap();
        // Body neutral is None → body pose passes through unchanged.
        assert!((f.yaw - 0.2).abs() < 1e-6);
        assert!((f.pitch - -0.1).abs() < 1e-6);
        assert!((f.roll - 0.05).abs() < 1e-6);

        // And with a body neutral present, the body pose uses IT, not
        // the mesh one.
        cal.pose.as_mut().unwrap().neutral_face_ypr_body = Some([0.2, -0.1, 0.05]);
        let mut sk = SourceSkeleton::default();
        sk.face = Some(FacePose {
            yaw: 0.2,
            pitch: -0.1,
            roll: 0.05,
            confidence: 0.9,
            source: crate::tracking::FaceSource::Body,
            ..Default::default()
        });
        cal.apply_calibration(&mut sk);
        let f = sk.face.unwrap();
        assert!(f.yaw.abs() < 1e-6);
        assert!(f.pitch.abs() < 1e-6);
        assert!(f.roll.abs() < 1e-6);
    }

    #[test]
    fn empty_calibration_is_a_noop() {
        let cal = TrackingCalibration::default();
        let mut sk = skeleton_with("aa", 0.42);
        cal.apply_calibration(&mut sk);
        assert!((sk.expressions[0].weight - 0.42).abs() < 1e-6);
    }

    // --- Neutral body yaw ---

    use crate::asset::HumanoidBone;
    use crate::tracking::source_skeleton::{HandOrientation, SourceJoint};

    /// A frontal reference sample with joints on both sides of the
    /// anchor, a fingertip, a palm frame and a root offset displaced
    /// from the calibrated anchor.
    fn frontal_metric_skeleton() -> SourceSkeleton {
        let mut sk = SourceSkeleton::default();
        let put = |sk: &mut SourceSkeleton, bone, position: [f32; 3]| {
            sk.joints.insert(
                bone,
                SourceJoint {
                    position,
                    confidence: 0.9,
                    metric_depth_m: None,
                },
            );
        };
        put(&mut sk, HumanoidBone::LeftUpperArm, [0.25, 0.1, 0.0]);
        put(&mut sk, HumanoidBone::RightUpperArm, [-0.25, 0.1, 0.0]);
        put(&mut sk, HumanoidBone::LeftHand, [0.4, -0.2, 0.15]);
        sk.fingertips.insert(
            HumanoidBone::LeftIndexDistal,
            SourceJoint {
                position: [0.45, -0.22, 0.18],
                confidence: 0.8,
                metric_depth_m: None,
            },
        );
        sk.left_hand_orientation = Some(HandOrientation {
            forward: [0.0, 0.0, 1.0],
            up: [0.0, 1.0, 0.0],
            confidence: 0.9,
        });
        sk.root_offset = Some([0.15, 0.02, -1.6]);
        sk.stamp_synthetic_metric_frame();
        sk
    }

    /// Rotate the frontal scene by −θ about the calibrated anchor —
    /// i.e. what the same subject looks like observed by a camera
    /// placed θ off to the side. `rotate_xz(·, θ)` must undo exactly
    /// this.
    fn obliquely_observed(frontal: &SourceSkeleton, theta: f32, a0: [f32; 3]) -> SourceSkeleton {
        let mut sk = frontal.clone();
        let fwd = |v: [f32; 3]| crate::tracking::calibration::rotate_xz(v, -theta);
        for j in sk.joints.values_mut() {
            j.position = fwd(j.position);
        }
        for t in sk.fingertips.values_mut() {
            t.position = fwd(t.position);
        }
        let h = sk.left_hand_orientation.as_mut().unwrap();
        h.forward = fwd(h.forward);
        h.up = fwd(h.up);
        let o = sk.root_offset.unwrap();
        let d = fwd([o[0] - a0[0], o[1] - a0[1], o[2] - a0[2]]);
        sk.root_offset = Some([a0[0] + d[0], a0[1] + d[1], a0[2] + d[2]]);
        sk
    }

    fn cal_with_yaw(theta: f32, a0: [f32; 3]) -> TrackingCalibration {
        let mut cal = cal_with_neutral(Vec::new());
        let pose = cal.pose.as_mut().unwrap();
        pose.neutral_body_yaw = Some(theta);
        pose.anchor_x = a0[0];
        pose.anchor_y = a0[1];
        pose.anchor_depth_m = Some(-a0[2]);
        cal
    }

    fn assert_vec3_eq(a: [f32; 3], b: [f32; 3], what: &str) {
        for i in 0..3 {
            assert!(
                (a[i] - b[i]).abs() < 1e-5,
                "{what}[{i}]: {} vs {}",
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn body_yaw_derotation_restores_the_frontal_scene() {
        let theta = 0.5_f32; // ~28.6° oblique camera
        let a0 = [0.1, 0.0, -1.5];
        let frontal = frontal_metric_skeleton();
        let mut observed = obliquely_observed(&frontal, theta, a0);

        cal_with_yaw(theta, a0).apply_calibration(&mut observed);

        for (bone, j) in &frontal.joints {
            assert_vec3_eq(
                observed.joints[bone].position,
                j.position,
                &format!("joint {bone:?}"),
            );
        }
        assert_vec3_eq(
            observed.fingertips[&HumanoidBone::LeftIndexDistal].position,
            frontal.fingertips[&HumanoidBone::LeftIndexDistal].position,
            "fingertip",
        );
        let (ho, hf) = (
            observed.left_hand_orientation.unwrap(),
            frontal.left_hand_orientation.unwrap(),
        );
        assert_vec3_eq(ho.forward, hf.forward, "hand forward");
        assert_vec3_eq(ho.up, hf.up, "hand up");
        assert_vec3_eq(
            observed.root_offset.unwrap(),
            frontal.root_offset.unwrap(),
            "root_offset",
        );
    }

    #[test]
    fn body_yaw_pivots_root_offset_about_the_calibrated_anchor() {
        // A subject AT the calibrated anchor must not move when the
        // de-rotation kicks in — only deviations from neutral rotate.
        let a0 = [0.1, 0.0, -1.5];
        let mut sk = frontal_metric_skeleton();
        sk.root_offset = Some(a0);
        cal_with_yaw(0.5, a0).apply_calibration(&mut sk);
        assert_vec3_eq(sk.root_offset.unwrap(), a0, "anchor stays put");
    }

    #[test]
    fn body_yaw_none_is_a_noop() {
        let frontal = frontal_metric_skeleton();
        let mut sk = frontal.clone();
        cal_with_neutral(Vec::new()).apply_calibration(&mut sk);
        for (bone, j) in &frontal.joints {
            assert_vec3_eq(sk.joints[bone].position, j.position, "joint");
        }
        assert_vec3_eq(sk.root_offset.unwrap(), frontal.root_offset.unwrap(), "offset");
    }

    #[test]
    fn body_yaw_never_rotates_non_metric_samples() {
        // Rotating a monocular z≈0 offset would fabricate depth from
        // nothing — the guard must hold even if a yaw somehow got
        // persisted against a non-metric session.
        let frontal = frontal_metric_skeleton();
        let mut sk = frontal.clone();
        sk.metric_frame_info = None;
        cal_with_yaw(0.5, [0.0, 0.0, 0.0]).apply_calibration(&mut sk);
        for (bone, j) in &frontal.joints {
            assert_vec3_eq(sk.joints[bone].position, j.position, "joint");
        }
    }

    #[test]
    fn body_yaw_leaves_the_face_pose_alone() {
        // Head neutrality is owned by neutral_face_ypr_* — rotating the
        // face AND subtracting its neutral would double-count θ.
        let mut sk = frontal_metric_skeleton();
        sk.face = Some(FacePose {
            yaw: 0.3,
            pitch: 0.1,
            roll: 0.0,
            confidence: 0.9,
            source: crate::tracking::FaceSource::Mesh,
            ..Default::default()
        });
        cal_with_yaw(0.5, [0.0, 0.0, 0.0]).apply_calibration(&mut sk);
        let f = sk.face.unwrap();
        assert!((f.yaw - 0.3).abs() < 1e-6, "face yaw untouched by body yaw");
    }
}

// ---------------------------------------------------------------------------
// TrackingMailbox
//
// The mailbox is split into four independent mutexes so the hot path
// (worker → app pose handoff) doesn't contend with GUI-only state. Each
// sub-mutex guards a self-contained slice of state and **no call site
// acquires more than one of them at a time**, so deadlock by lock-order
// inversion is impossible by construction.
//
// Atomicity trade-off: `snapshot` (pose + frame + annotation in one
// struct) and `publish_estimate` (pose + frame + annotation in one
// write) used to be one-lock atomic. They now span two locks (pose
// and preview), so a reader that races a writer can observe the new
// pose-side sequence with the previous frame for at most one
// publish_estimate. The GUI consumers (`viewport.rs` camera wipe,
// `calibration/refresh.rs` telemetry) dedup on `sequence`, so a
// torn read shows up as one extra "no-update" frame at worst — no
// rendering corruption.
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct TrackingMailbox {
    /// Hot path: worker writes per inference cycle, app reads every
    /// frame from `run_frame`. Keeping this isolated from the heavier
    /// `PreviewFrame` clone in the preview mailbox is the main reason
    /// for the split.
    pose: Arc<Mutex<PoseMailboxInner>>,
    /// GUI display: camera preview frame + detection annotation. Cloned per
    /// GUI tick. Larger payloads (RGB pixel buffers) so the lock is
    /// occasionally held a little longer, but never while the pose
    /// mutex is also held.
    preview: Arc<Mutex<PreviewMailboxInner>>,
    /// Low-frequency worker → GUI diagnostics: error toasts +
    /// inference backend label.
    diagnostics: Arc<Mutex<DiagnosticsMailboxInner>>,
    /// Bidirectional calibration channel: GUI → worker commands
    /// (calibration, torso-capture toggle, mode hint) plus worker → GUI
    /// results (captured torso template). All edge-detected via
    /// matching `_seq` counters.
    calibration: Arc<Mutex<CalibrationChannelInner>>,
    stale_timeout_nanos: u64,
}

struct PoseMailboxInner {
    latest_pose: Option<SourceSkeleton>,
    sequence: u64,
    /// Monotonic-clock instant of the last publish. `Instant`, not
    /// `SystemTime`: freshness is a process-local interval measurement,
    /// and a wall clock that gets NTP-stepped backwards would make a
    /// dead worker's sample report as eternally fresh (`saturating_sub`
    /// pinning the age at zero) — the avatar would freeze in the last
    /// pose instead of fading. `None` until the first publish.
    last_update: Option<Instant>,
}

struct PreviewMailboxInner {
    /// `Arc` so `snapshot()` hands the frame out by refcount instead of
    /// copying the RGB buffer (≈2.7 MB at 1280×720) under the lock on
    /// every GUI tick. GUI consumers should gate `snapshot()` behind
    /// [`TrackingMailbox::preview_sequence`] and only pull when it
    /// advanced — 60 Hz GUI × 30 fps camera used to clone-and-discard
    /// half the frames without ever reading them.
    latest_frame: Option<Arc<PreviewFrame>>,
    latest_annotation: Option<DetectionAnnotation>,
    /// Bumped on every preview write so GUI consumers can dedup
    /// texture uploads on a counter that strictly corresponds to
    /// frame freshness. The pose-mailbox `sequence` and this one
    /// advance together inside `publish_estimate` (under separate
    /// locks); a snapshot that races the writer can observe pose-
    /// `sequence == N` with preview `sequence == N-1`, in which case
    /// the GUI sees "old frame, new pose" — the right dedup key for
    /// preview consumers is *this* counter, not pose `sequence`.
    sequence: u64,
}

struct DiagnosticsMailboxInner {
    pending_error: Option<(String, TrackingErrorLevel)>,
    /// One-line label of the inference backend in use (e.g. "DirectML",
    /// "CPU", "CPU (DirectML unavailable: ...)"). `None` when no
    /// inference engine is loaded — e.g. synthetic mode, or before the
    /// worker has finished init.
    inference_backend_label: Option<String>,
}

struct CalibrationChannelInner {
    /// Latest pose-calibration capture pushed from the GUI. The
    /// tracking worker reads this each iteration and forwards it to
    /// the provider via `PoseProvider::set_calibration`. Replaces
    /// (rather than queues) on each write — only the most recent
    /// value matters; the worker doesn't care about intermediate
    /// captures the user dismissed.
    calibration: Option<PoseCalibration>,
    /// Bumped each time `set_calibration` writes. The worker compares
    /// against its last-seen value to skip the `set_calibration`
    /// call on iterations where nothing changed (avoids the per-frame
    /// `Option::clone` of the calibration).
    calibration_seq: u64,
    /// GUI-driven flag: `true` while the calibration modal is in its
    /// `Collecting` state and the depth provider should accumulate
    /// per-frame torso-bbox depth samples for `TorsoDepthTemplate`
    /// capture. The worker forwards transitions to the provider via
    /// `PoseProvider::set_torso_capture` and harvests the finished
    /// template via `take_torso_template` when the flag flips off.
    torso_capture_enabled: bool,
    /// Bumped each time `set_torso_capture` writes; same edge-detect
    /// rationale as `calibration_seq`.
    torso_capture_seq: u64,
    /// Worker → GUI: the most recently published torso template,
    /// posted by the worker after a `Collecting` window closes.
    /// Consumed by the calibration modal's `take_torso_template`
    /// poll which stitches it onto the in-flight `PoseCalibration`
    /// during `finalize_collection`.
    torso_template_collected: Option<TorsoDepthTemplate>,
    /// Bumped each time `publish_torso_template` writes. The GUI's
    /// poll uses the same `seq` pattern to consume each template
    /// exactly once and avoid double-stitching.
    torso_template_seq: u64,
    /// GUI-driven hint: the calibration mode the user is *currently
    /// collecting* (modal open, samples about to flow), independent of
    /// any confirmed `PoseCalibration`. The worker forwards this to the
    /// provider via [`PoseProvider::set_calibration_mode_hint`]; without
    /// it the very first `UpperBody` capture sees `force_shoulder_anchor=
    /// false` and the GUI rejects every collected sample.
    pending_calibration_mode_hint: Option<CalibrationMode>,
    /// Bumped each time `set_calibration_mode_hint` writes; same edge-
    /// detect rationale as `calibration_seq`.
    pending_calibration_mode_hint_seq: u64,
}

/// Captured snapshot of the tracking mailbox state. Not strictly
/// cross-lock-atomic — see `TrackingMailbox::snapshot` for the
/// torn-read trade-off. `sequence` is the pose-side counter (use for
/// pose-driven dedup like solver consumption); `preview_sequence` is
/// the preview-side counter (use for frame / annotation upload dedup
/// — using `sequence` for frame uploads can permanently skip a frame
/// whenever the snapshot races a writer mid-publish).
#[derive(Clone, Debug, Default)]
pub struct MailboxSnapshot {
    pub pose: Option<SourceSkeleton>,
    /// Shared, not copied — cloning the snapshot bumps a refcount
    /// instead of duplicating the RGB buffer. Consumers should avoid
    /// calling [`TrackingMailbox::snapshot`] at all unless
    /// [`TrackingMailbox::preview_sequence`] advanced.
    pub frame: Option<Arc<PreviewFrame>>,
    pub annotation: Option<DetectionAnnotation>,
    pub sequence: u64,
    pub preview_sequence: u64,
}

impl TrackingMailbox {
    pub fn new() -> Self {
        Self {
            pose: Arc::new(Mutex::new(PoseMailboxInner {
                latest_pose: None,
                sequence: 0,
                last_update: None,
            })),
            preview: Arc::new(Mutex::new(PreviewMailboxInner {
                latest_frame: None,
                latest_annotation: None,
                sequence: 0,
            })),
            diagnostics: Arc::new(Mutex::new(DiagnosticsMailboxInner {
                pending_error: None,
                inference_backend_label: None,
            })),
            calibration: Arc::new(Mutex::new(CalibrationChannelInner {
                calibration: None,
                calibration_seq: 0,
                torso_capture_enabled: false,
                torso_capture_seq: 0,
                torso_template_collected: None,
                torso_template_seq: 0,
                pending_calibration_mode_hint: None,
                pending_calibration_mode_hint_seq: 0,
            })),
            stale_timeout_nanos: TrackingSmoothingParams::default().stale_timeout_nanos,
        }
    }
}

impl Default for TrackingMailbox {
    fn default() -> Self {
        Self::new()
    }
}

impl TrackingMailbox {
    /// Thread-safe publish (pose-only path used by synthetic / test
    /// drivers that don't attach a preview frame).
    pub fn publish(&self, pose: SourceSkeleton) {
        let mut p = self.pose.lock().unwrap_or_else(|e| e.into_inner());
        p.latest_pose = Some(pose);
        p.sequence += 1;
        p.last_update = Some(Instant::now());
    }

    /// Publish a full estimation result including frame and annotations.
    /// Writes to two mutexes sequentially (pose, then preview) — see the
    /// module-level comment above `TrackingMailbox` for the cross-lock
    /// atomicity trade-off.
    pub fn publish_estimate(&self, estimate: PoseEstimate, frame: Option<PreviewFrame>) {
        {
            let mut p = self.pose.lock().unwrap_or_else(|e| e.into_inner());
            p.latest_pose = Some(estimate.skeleton);
            p.sequence += 1;
            p.last_update = Some(Instant::now());
        }
        // Wrap outside the lock: the one-time Arc allocation is the
        // publisher's cost; every reader clone afterwards is a refcount.
        let frame = frame.map(Arc::new);
        let mut v = self.preview.lock().unwrap_or_else(|e| e.into_inner());
        v.latest_annotation = Some(estimate.annotation);
        v.latest_frame = frame;
        v.sequence += 1;
    }

    /// Current preview-side sequence, read under the preview lock but
    /// without touching the payloads. GUI consumers poll this every
    /// tick and call [`Self::snapshot`] only when it advanced — the
    /// cheap gate that keeps a 60 Hz GUI from cloning a 30 fps
    /// camera's frames it will never upload.
    pub fn preview_sequence(&self) -> u64 {
        let v = self.preview.lock().unwrap_or_else(|e| e.into_inner());
        v.sequence
    }

    /// Snapshot the pose + preview slices. NOT cross-lock-atomic: a
    /// concurrent `publish_estimate` between the two lock acquisitions
    /// can produce a snapshot where `sequence` matches the new pose
    /// but `frame` / `annotation` are still from the previous publish.
    /// Both GUI consumers dedup on `sequence` and treat a momentary
    /// pose-newer-than-frame as "no update", so the worst-case is one
    /// skipped redraw rather than visible corruption.
    pub fn snapshot(&self) -> MailboxSnapshot {
        let (pose, sequence) = {
            let p = self.pose.lock().unwrap_or_else(|e| e.into_inner());
            (p.latest_pose.clone(), p.sequence)
        };
        let v = self.preview.lock().unwrap_or_else(|e| e.into_inner());
        MailboxSnapshot {
            pose,
            frame: v.latest_frame.clone(),
            annotation: v.latest_annotation.clone(),
            sequence,
            preview_sequence: v.sequence,
        }
    }

    /// Report a non-fatal error from the worker thread (shown as GUI toast).
    pub fn report_error(&self, msg: impl Into<String>, level: TrackingErrorLevel) {
        let mut d = self.diagnostics.lock().unwrap_or_else(|e| e.into_inner());
        d.pending_error = Some((msg.into(), level));
    }

    /// Drain the latest pending error (returns it only once).
    pub fn drain_error(&self) -> Option<(String, TrackingErrorLevel)> {
        let mut d = self.diagnostics.lock().unwrap_or_else(|e| e.into_inner());
        d.pending_error.take()
    }

    /// GUI-side push: stash the latest pose calibration so the worker
    /// thread can pick it up on its next iteration. Replaces (rather
    /// than queues) — only the most recent value matters; bumping the
    /// sequence lets the worker skip the per-frame
    /// `set_calibration` call when nothing changed.
    pub fn set_calibration(&self, calibration: Option<PoseCalibration>) {
        let mut c = self.calibration.lock().unwrap_or_else(|e| e.into_inner());
        c.calibration = calibration;
        c.calibration_seq += 1;
    }

    /// Worker-side poll: returns `Some((calibration, seq))` only when
    /// `seq` has advanced past `last_seen_seq`, so the worker
    /// processes a calibration update at most once per write.
    pub fn poll_calibration(
        &self,
        last_seen_seq: u64,
    ) -> Option<(Option<PoseCalibration>, u64)> {
        let c = self.calibration.lock().unwrap_or_else(|e| e.into_inner());
        if c.calibration_seq != last_seen_seq {
            Some((c.calibration.clone(), c.calibration_seq))
        } else {
            None
        }
    }

    /// GUI-side push: toggle the per-frame torso depth capture on or
    /// off. Same replace-don't-queue semantics as `set_calibration`;
    /// rapid toggles (e.g. modal open → cancel → re-open) collapse
    /// to whichever transition the worker observes first.
    pub fn set_torso_capture(&self, enabled: bool) {
        let mut c = self.calibration.lock().unwrap_or_else(|e| e.into_inner());
        c.torso_capture_enabled = enabled;
        c.torso_capture_seq += 1;
    }

    /// Worker-side poll for the torso-capture toggle. Same edge-detect
    /// pattern as `poll_calibration`. Returns `Some((enabled, seq))`
    /// only when the GUI flipped the flag since `last_seen_seq`.
    pub fn poll_torso_capture(&self, last_seen_seq: u64) -> Option<(bool, u64)> {
        let c = self.calibration.lock().unwrap_or_else(|e| e.into_inner());
        if c.torso_capture_seq != last_seen_seq {
            Some((c.torso_capture_enabled, c.torso_capture_seq))
        } else {
            None
        }
    }

    /// GUI-side push: the calibration mode the user has selected in the
    /// open modal, or `None` when the modal is closed. Forwarded to the
    /// provider as a transient anchor hint so the very first `UpperBody`
    /// capture isn't dropped by the model's hallucinated-hip output —
    /// see [`PoseProvider::set_calibration_mode_hint`] for the contract.
    pub fn set_calibration_mode_hint(&self, hint: Option<CalibrationMode>) {
        let mut c = self.calibration.lock().unwrap_or_else(|e| e.into_inner());
        c.pending_calibration_mode_hint = hint;
        c.pending_calibration_mode_hint_seq += 1;
    }

    /// Worker-side poll for the calibration-mode hint. Same edge-detect
    /// pattern as `poll_calibration`. Returns `Some((hint, seq))` only
    /// when the GUI changed the hint since `last_seen_seq`.
    pub fn poll_calibration_mode_hint(
        &self,
        last_seen_seq: u64,
    ) -> Option<(Option<CalibrationMode>, u64)> {
        let c = self.calibration.lock().unwrap_or_else(|e| e.into_inner());
        if c.pending_calibration_mode_hint_seq != last_seen_seq {
            Some((
                c.pending_calibration_mode_hint,
                c.pending_calibration_mode_hint_seq,
            ))
        } else {
            None
        }
    }

    /// Worker-side push: publish a finished torso template back to
    /// the GUI for stitching onto the in-flight `PoseCalibration`.
    /// Replaces the previous template (if any) — only the most
    /// recent capture matters; if a user runs two captures in rapid
    /// succession the second one wins.
    pub fn publish_torso_template(&self, template: TorsoDepthTemplate) {
        let mut c = self.calibration.lock().unwrap_or_else(|e| e.into_inner());
        c.torso_template_collected = Some(template);
        c.torso_template_seq += 1;
    }

    /// GUI-side peek: read the current torso-template publish seq
    /// without consuming the template. Used by the modal's `retry`
    /// path to fast-forward its `last_seen_seq` past any in-flight
    /// publish from the *previous* capture window — without this
    /// fast-forward, a publish that lands during the new HoldStill
    /// (1–2 frames after the user clicked Retry) would be picked up
    /// as if it belonged to the new capture and silently overwrite
    /// the wrong calibration.
    pub fn torso_template_seq(&self) -> u64 {
        let c = self.calibration.lock().unwrap_or_else(|e| e.into_inner());
        c.torso_template_seq
    }

    /// GUI-side consume: take the most recently published torso
    /// template if the worker published a new one since
    /// `last_seen_seq`. Returns `Some((template, seq))` exactly once
    /// per worker publish; subsequent calls with the new `seq`
    /// return `None` until the next publish. The template is
    /// consumed (cleared from the mailbox) so a stale capture from
    /// an earlier session can't leak into a later one.
    pub fn take_torso_template(
        &self,
        last_seen_seq: u64,
    ) -> Option<(TorsoDepthTemplate, u64)> {
        let mut c = self.calibration.lock().unwrap_or_else(|e| e.into_inner());
        if c.torso_template_seq != last_seen_seq {
            c.torso_template_collected
                .take()
                .map(|t| (t, c.torso_template_seq))
        } else {
            None
        }
    }

    /// Set the inference-backend label. Called once by the worker after
    /// `Rtmw3dInference` finishes loading its model. `None` resets it
    /// (e.g. when tracking stops or the engine is destroyed).
    pub fn set_inference_backend_label(&self, label: Option<String>) {
        let mut d = self.diagnostics.lock().unwrap_or_else(|e| e.into_inner());
        d.inference_backend_label = label;
    }

    /// Read the inference-backend label. Cheap clone so the GUI thread
    /// can render without holding the lock.
    pub fn inference_backend_label(&self) -> Option<String> {
        let d = self.diagnostics.lock().unwrap_or_else(|e| e.into_inner());
        d.inference_backend_label.clone()
    }

    /// Thread-safe read: takes &self, locks the pose mutex, clones the pose.
    pub fn latest_pose(&self) -> Option<SourceSkeleton> {
        let p = self.pose.lock().unwrap_or_else(|e| e.into_inner());
        p.latest_pose.clone()
    }

    /// Read the latest camera preview frame (downscaled for GUI display).
    pub fn latest_frame(&self) -> Option<Arc<PreviewFrame>> {
        let v = self.preview.lock().unwrap_or_else(|e| e.into_inner());
        v.latest_frame.clone()
    }

    /// Read the latest 2D detection annotation.
    pub fn latest_annotation(&self) -> Option<DetectionAnnotation> {
        let v = self.preview.lock().unwrap_or_else(|e| e.into_inner());
        v.latest_annotation.clone()
    }

    pub fn sequence(&self) -> u64 {
        let p = self.pose.lock().unwrap_or_else(|e| e.into_inner());
        p.sequence
    }

    /// Returns true if the latest sample is older than `stale_timeout_nanos`,
    /// or if no sample has ever been published.
    pub fn is_stale(&self) -> bool {
        let p = self.pose.lock().unwrap_or_else(|e| e.into_inner());
        match p.last_update {
            None => true,
            Some(at) => at.elapsed().as_nanos() as u64 > self.stale_timeout_nanos,
        }
    }

    /// Time elapsed since the last published sample. `None` when no
    /// sample has ever been published. The hold/fade policy in
    /// [`crate::app::Application::run_frame`] uses this to grade
    /// stale samples by age (fresh / holding / expired) instead of
    /// the binary stale flag.
    pub fn age(&self) -> Option<std::time::Duration> {
        let p = self.pose.lock().unwrap_or_else(|e| e.into_inner());
        p.last_update.map(|at| at.elapsed())
    }

    /// Stale-flip threshold the mailbox was constructed with. Surfaced
    /// so the hold/fade policy can ladder its windows from the same
    /// anchor instead of duplicating the constant.
    pub fn stale_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_nanos(self.stale_timeout_nanos)
    }
}

#[cfg(test)]
mod mailbox_tests {
    use super::*;

    fn empty_estimate() -> PoseEstimate {
        PoseEstimate {
            skeleton: SourceSkeleton::empty(0),
            annotation: DetectionAnnotation::default(),
        }
    }

    #[test]
    fn empty_mailbox_is_stale_and_has_no_age() {
        let mb = TrackingMailbox::new();
        assert!(mb.is_stale());
        assert_eq!(mb.age(), None);
        assert_eq!(mb.sequence(), 0);
        assert!(mb.latest_pose().is_none());
        assert!(mb.latest_frame().is_none());
    }

    #[test]
    fn publish_bumps_pose_sequence_only() {
        let mb = TrackingMailbox::new();
        let snap_before = mb.snapshot();
        mb.publish(SourceSkeleton::empty(0));
        let snap_after = mb.snapshot();
        assert_eq!(snap_after.sequence, snap_before.sequence + 1);
        assert_eq!(
            snap_after.preview_sequence, snap_before.preview_sequence,
            "publish (pose-only) must not touch preview_sequence"
        );
    }

    #[test]
    fn publish_estimate_bumps_both_sequences() {
        let mb = TrackingMailbox::new();
        mb.publish_estimate(empty_estimate(), None);
        let snap = mb.snapshot();
        assert_eq!(snap.sequence, 1);
        assert_eq!(snap.preview_sequence, 1);
    }

    #[test]
    fn calibration_command_and_template_use_independent_seqs() {
        let mb = TrackingMailbox::new();
        // Calibration command seq is independent of pose / preview seq.
        mb.set_calibration(None);
        let snap = mb.snapshot();
        assert_eq!(snap.sequence, 0, "calibration write must not bump pose seq");
        assert_eq!(
            snap.preview_sequence, 0,
            "calibration write must not bump preview seq"
        );

        // Calibration mailbox uses its own seqs for edge detection.
        assert!(mb.poll_calibration(0).is_some(), "first poll sees the write");
        let observed_seq = mb.poll_calibration(0).map(|(_, s)| s).unwrap();
        assert!(mb.poll_calibration(observed_seq).is_none(), "no advance, no work");
    }

    #[test]
    fn publish_makes_mailbox_fresh_with_monotonic_age() {
        let mb = TrackingMailbox::new();
        mb.publish(SourceSkeleton::empty(0));
        assert!(!mb.is_stale(), "just-published sample must be fresh");
        let age = mb.age().expect("age is Some after a publish");
        assert!(
            age < std::time::Duration::from_secs(1),
            "age of a just-published sample must be near zero, got {age:?}"
        );
    }

    #[test]
    fn error_drain_and_backend_label_round_trip() {
        let mb = TrackingMailbox::new();
        mb.report_error("test failure", TrackingErrorLevel::Warning);
        let drained = mb.drain_error();
        assert!(drained.is_some());
        assert!(
            mb.drain_error().is_none(),
            "errors drain exactly once"
        );
        mb.set_inference_backend_label(Some("CPU".to_string()));
        assert_eq!(mb.inference_backend_label(), Some("CPU".to_string()));
        mb.set_inference_backend_label(None);
        assert_eq!(mb.inference_backend_label(), None);
    }
}

// ---------------------------------------------------------------------------
// Camera preview frame & detection annotation (for GUI PIP wipe display)
// ---------------------------------------------------------------------------

/// A downscaled camera preview frame for the GUI camera-preview overlay.
#[derive(Clone, Debug)]
pub struct PreviewFrame {
    pub rgb_data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// 2D detection annotation overlaid on the camera preview.
#[derive(Clone, Debug, Default)]
pub struct DetectionAnnotation {
    /// Keypoints as (x, y, confidence) in normalised [0, 1] image coords.
    pub keypoints: Vec<(f32, f32, f32)>,
    /// Skeleton line connections as pairs of keypoint indices.
    pub skeleton: Vec<(usize, usize)>,
    /// Bounding box (min_x, min_y, max_x, max_y) in normalised coords.
    pub bounding_box: Option<(f32, f32, f32, f32)>,
}

/// Combined output of a pose estimation pass: source skeleton + 2D annotation.
pub struct PoseEstimate {
    pub skeleton: SourceSkeleton,
    pub annotation: DetectionAnnotation,
}

// ---------------------------------------------------------------------------
// Capture backend
// ---------------------------------------------------------------------------

/// Display label for the sole capture backend. This is a D435-exclusive
/// build: the Intel RealSense D435 (color-aligned metric depth) is the only
/// camera path, so the former "which backend" selector collapsed to a
/// constant. Shown in the status bar / tracking inspector while capture is
/// live.
pub const CAPTURE_BACKEND_LABEL: &str = "RealSense D435";

// ---------------------------------------------------------------------------
// TrackingSource
// ---------------------------------------------------------------------------

pub struct TrackingSource {
    mailbox: TrackingMailbox,
}

impl TrackingSource {
    pub fn new() -> Self {
        Self {
            mailbox: TrackingMailbox::new(),
        }
    }
}

impl Default for TrackingSource {
    fn default() -> Self {
        Self::new()
    }
}

impl TrackingSource {
    pub fn mailbox(&self) -> &TrackingMailbox {
        &self.mailbox
    }

    /// Return a clone of the mailbox for sharing with a worker thread.
    pub fn shared_mailbox(&self) -> TrackingMailbox {
        self.mailbox.clone()
    }
}

// ---------------------------------------------------------------------------
// TrackingWorker
// ---------------------------------------------------------------------------

/// Background worker that continuously captures tracking frames and publishes
/// the latest `SourceSkeleton` to a shared `TrackingMailbox`.
///
/// The app thread reads from the same mailbox using latest-sample semantics.
pub struct TrackingWorker {
    handle: Option<JoinHandle<()>>,
    running: Arc<AtomicBool>,
    ready: Arc<AtomicBool>,
    mailbox: TrackingMailbox,
}

impl TrackingWorker {
    /// Create a new tracking worker bound to the given mailbox.
    ///
    /// The worker is not started until `start()` is called.
    pub fn new(mailbox: TrackingMailbox) -> Self {
        Self {
            handle: None,
            running: Arc::new(AtomicBool::new(false)),
            ready: Arc::new(AtomicBool::new(false)),
            mailbox,
        }
    }

    /// Create a worker that reports `is_running()` / `is_ready()` as
    /// `true` immediately, without spawning a capture thread — for
    /// callers that drive the mailbox themselves (headless replay /
    /// diagnostics that publish `PoseEstimate`s via
    /// [`Self::mailbox`]`().publish_estimate(...)`, e.g.
    /// `src/bin/diagnose_signal_quality.rs`).
    ///
    /// `Application::run_frame` only reads the mailbox when
    /// `tracking_worker` is `Some` and running (`step_tracking`,
    /// `src/app/render.rs`) — this gate exists so a freshly loaded
    /// avatar doesn't fade out before tracking ever starts. Without
    /// this constructor an external driver has no sanctioned way to
    /// satisfy that gate short of spawning a real (and here,
    /// redundant) capture thread via `start_with_params`.
    pub fn new_external(mailbox: TrackingMailbox) -> Self {
        Self {
            handle: None,
            running: Arc::new(AtomicBool::new(true)),
            ready: Arc::new(AtomicBool::new(true)),
            mailbox,
        }
    }

    /// Return a reference to the shared mailbox.
    pub fn mailbox(&self) -> &TrackingMailbox {
        &self.mailbox
    }

    /// Returns `true` if the worker thread is currently running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Returns `true` once the worker has finished initialisation (camera
    /// opened or fallback engaged) and is actively producing frames.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }

    /// Spawn the tracking worker thread with the given capture parameters.
    /// The thread loops at approximately `fps` frames per second, capturing
    /// frames and publishing poses to the shared mailbox. If a worker is
    /// already running this is a no-op.
    pub fn start_with_params(
        &mut self,
        width: u32,
        height: u32,
        fps: u32,
        pipeline: provider::TrackingPipelineConfig,
    ) {
        // A handle can be left behind by a `stop()` that timed out while
        // the worker was wedged in a blocking camera call. If that thread
        // has since finished, reap it here and proceed with the restart —
        // the old behaviour (silent `return`) left the user with a Start
        // button that did nothing, with no path back short of an app
        // restart. Only a thread that is STILL alive blocks the restart
        // (overlapping GPU inference sessions must never coexist), and
        // that now gets an explicit error log instead of silence.
        if let Some(handle) = &self.handle {
            if handle.is_finished() {
                if let Some(handle) = self.handle.take() {
                    if let Err(e) = handle.join() {
                        error!("tracking-worker: previous thread panicked: {:?}", e);
                    }
                }
            } else {
                error!(
                    "tracking-worker: start requested while the previous worker thread is still \
                     shutting down; retry once it exits"
                );
                return;
            }
        }

        self.running = Arc::new(AtomicBool::new(true));
        self.ready = Arc::new(AtomicBool::new(false));
        let running = Arc::clone(&self.running);
        let ready = Arc::clone(&self.ready);
        let mailbox = self.mailbox.clone();

        let handle = thread::Builder::new()
            .name("tracking-worker".into())
            .spawn(move || {
                Self::worker_loop(mailbox, running, ready, width, height, fps, pipeline);
            })
            .expect("failed to spawn tracking-worker thread");

        self.handle = Some(handle);
    }

    /// Signal the worker to stop and join the thread.
    ///
    /// Waits up to 3 seconds for the worker thread to exit. If it hasn't
    /// exited by then (e.g. `grab_frame()` is blocking indefinitely), the
    /// join handle is kept so a later call can reap it once the blocked
    /// operation unwinds. This prevents a restart from orphaning a still-live
    /// inference worker and creating overlapping GPU sessions.
    ///
    /// Returns `true` when the worker fully stopped and was joined, or when
    /// there was no worker to stop. Returns `false` when the stop request was
    /// issued but the worker is still alive after the timeout.
    pub fn stop(&mut self) -> bool {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let deadline = std::time::Instant::now() + Duration::from_secs(3);
            loop {
                if handle.is_finished() {
                    if let Err(e) = handle.join() {
                        error!("tracking-worker: thread panicked: {:?}", e);
                    }
                    return true;
                }
                if std::time::Instant::now() >= deadline {
                    warn!(
                        "tracking-worker: thread did not exit within timeout; keeping handle and refusing overlapping restart"
                    );
                    self.handle = Some(handle);
                    return false;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        true
    }

    // -- internal -----------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn worker_loop(
        mailbox: TrackingMailbox,
        running: Arc<AtomicBool>,
        ready: Arc<AtomicBool>,
        width: u32,
        height: u32,
        fps: u32,
        pipeline: provider::TrackingPipelineConfig,
    ) {
        // D435-exclusive: the RealSense depth camera is the sole capture
        // backend. When the `realsense` feature is compiled out there is no
        // camera to drive the pipeline, so the worker reports ready and
        // exits immediately — the app falls back to the avatar rest pose.
        #[cfg(feature = "realsense")]
        Self::run_realsense(&mailbox, &running, &ready, width, height, fps, pipeline);
        #[cfg(not(feature = "realsense"))]
        {
            let _ = (width, height, fps, pipeline);
            warn!("tracking-worker: `realsense` feature disabled — no capture backend, idling");
            ready.store(true, Ordering::SeqCst);
        }
        // Ensure running is cleared when the thread exits for any reason.
        running.store(false, Ordering::SeqCst);
        // Clear the backend label so the inspector hides the row instead
        // of showing a stale value from the previous session.
        mailbox.set_inference_backend_label(None);
    }

    /// Capture color + aligned metric depth from a RealSense D435 and run
    /// the full RTMW3D pose pipeline, feeding the depth in via
    /// [`provider::PoseProvider::set_external_depth`] so the provider skips
    /// its internal DAv2 stage. Each per-frame step builds a
    /// `MetricDepthFrame` from the D435 depth and hands it to the provider
    /// before `estimate_pose`.
    ///
    /// Two threads:
    ///
    /// * **`tracking-capture`** (spawned here) owns the librealsense
    ///   pipeline: it opens the camera, grabs frames, and drops the
    ///   freshest one into a [`latest_cell::LatestCell`]. When grabs fail
    ///   persistently it reopens the device with backoff
    ///   ([`GrabRecovery`]).
    /// * **`tracking-worker`** (this thread) blocks on the cell and runs
    ///   inference + publish on whatever frame is freshest.
    ///
    /// The split is what keeps latency bounded: with grab and inference
    /// serialized on one thread, an inference pass slower than the frame
    /// period made `pipeline.wait` drain librealsense's internal queue
    /// oldest-first — the pipeline fell one-plus frames behind and STAYED
    /// behind. The cell's latest-only semantics discard frames that went
    /// stale while inference was busy, and the camera itself paces the
    /// loop (no `thread::sleep`: sleeping after a fast inference pass
    /// only made the next frame older by the slept amount).
    #[cfg(feature = "realsense")]
    #[allow(clippy::too_many_arguments)]
    fn run_realsense(
        mailbox: &TrackingMailbox,
        running: &Arc<AtomicBool>,
        ready: &AtomicBool,
        width: u32,
        height: u32,
        fps: u32,
        pipeline: provider::TrackingPipelineConfig,
    ) {
        info!(
            "tracking-worker: opening RealSense D435 ({}x{} @ {} fps)",
            width, height, fps
        );

        let _stage_session =
            stagelog::SessionGuard::begin(&format!("realsense {width}x{height}@{fps}"));
        stagelog::mark(0, "camera_open_begin");

        let cell = latest_cell::LatestCell::<CaptureItem>::new();
        let (open_tx, open_rx) =
            std::sync::mpsc::channel::<Result<(u32, u32), realsense::OpenFailure>>();
        let capture_handle = {
            let cell = Arc::clone(&cell);
            let running = Arc::clone(running);
            let mailbox = mailbox.clone();
            thread::Builder::new()
                .name("tracking-capture".into())
                .spawn(move || {
                    capture_loop(cell, running, mailbox, open_tx, width, height, fps);
                })
                .expect("failed to spawn tracking-capture thread")
        };

        // The capture thread reports the open outcome exactly once.
        let (cap_width, cap_height) = match open_rx.recv() {
            Ok(Ok(dims)) => dims,
            Ok(Err(e)) => {
                error!("tracking-worker: failed to open RealSense: {}", e);
                // Map the typed open failure to a root-cause-specific,
                // localized message so the GUI dialog names the real problem
                // (e.g. a USB-2 link speed) instead of an opaque driver string.
                let msg = match e {
                    realsense::OpenFailure::UsbLinkTooSlow { detected } => {
                        t!("tracking.error_usb_link_speed", detected = detected)
                    }
                    realsense::OpenFailure::NoDevice => t!("tracking.error_no_device"),
                    realsense::OpenFailure::Other(m) => {
                        t!("tracking.error_realsense_open", error = m)
                    }
                };
                mailbox.report_error(msg, TrackingErrorLevel::Blocking);
                // No synthetic fallback in the D435-exclusive build: surface
                // the blocking error and idle. `worker_loop` clears `running`
                // on return, so the app drops to the avatar rest pose.
                ready.store(true, Ordering::SeqCst);
                let _ = capture_handle.join();
                return;
            }
            Err(_) => {
                error!("tracking-worker: capture thread exited before reporting an open result");
                ready.store(true, Ordering::SeqCst);
                let _ = capture_handle.join();
                return;
            }
        };

        // Provider init under cooperative GPU exclusivity: the ONNX/EP
        // load bursts the GPU, so serialize it against other init work.
        stagelog::mark(0, "provider_load_begin");
        let mut pose_provider = {
            let _gpu_exclusive =
                crate::gpu_coordination::GpuExclusiveGuard::acquire("pose-provider-init");
            match provider::create_pose_provider("models", pipeline) {
                Ok(mut provider) => {
                    let warnings = provider.take_load_warnings();
                    if !warnings.is_empty() {
                        mailbox.report_error(
                            t!(
                                "tracking.error_model_warning",
                                warnings = warnings.join("; ")
                            ),
                            TrackingErrorLevel::Warning,
                        );
                    }
                    mailbox.set_inference_backend_label(Some(provider.label()));
                    stagelog::mark(0, "provider_warmup_begin");
                    let blank = vec![0u8; (cap_width as usize) * (cap_height as usize) * 3];
                    let _ = provider.estimate_pose(&blank, cap_width, cap_height, 0);
                    stagelog::mark(0, "provider_warmup_end");
                    provider.reset_temporal_state();
                    Some(provider)
                }
                Err(e) => {
                    error!("tracking-worker: inference disabled: {}", e);
                    mailbox.report_error(
                        t!("tracking.error_model_unavailable", error = e.to_string()),
                        TrackingErrorLevel::Blocking,
                    );
                    None
                }
            }
        };

        info!("tracking-worker: RealSense opened successfully");
        stagelog::mark(0, "provider_load_end");
        ready.store(true, Ordering::SeqCst);
        let mut last_calibration_seq: u64 = 0;
        let mut last_torso_capture_seq: u64 = 0;
        let mut last_calibration_mode_hint_seq: u64 = 0;

        while running.load(Ordering::SeqCst) {
            // Blocks until the freshest capture is available; wakes with
            // `None` once the capture thread closed the cell (stop
            // request or unrecoverable camera loss).
            let Some(CaptureItem {
                frame: rs_frame,
                index: frame_index,
            }) = cell.take_blocking()
            else {
                break;
            };

            // Forward calibration / torso-capture / mode-hint transitions
            // to the provider (edge-detected: forward only on change).
            if let Some(ref mut provider) = pose_provider {
                if let Some((cal, seq)) = mailbox.poll_calibration(last_calibration_seq) {
                    provider.set_calibration(cal);
                    last_calibration_seq = seq;
                }
                if let Some((enabled, seq)) = mailbox.poll_torso_capture(last_torso_capture_seq) {
                    if enabled {
                        provider.set_torso_capture(true);
                    } else {
                        if let Some(template) = provider.take_torso_template() {
                            mailbox.publish_torso_template(template);
                        }
                        provider.set_torso_capture(false);
                    }
                    last_torso_capture_seq = seq;
                }
                if let Some((hint, seq)) =
                    mailbox.poll_calibration_mode_hint(last_calibration_mode_hint_seq)
                {
                    provider.set_calibration_mode_hint(hint);
                    last_calibration_mode_hint_seq = seq;
                }
            }

            let width = rs_frame.width;
            let height = rs_frame.height;

            // Raw-input recorder (off unless the flag file exists). Placed
            // BEFORE any processing on purpose: the whole point is to keep a
            // record that no estimator has touched, so a replacement can be
            // scored on the same sensor data as the incumbent.
            sequence_recorder::record(frame_index, &rs_frame);

            stagelog::mark(frame_index, "estimate_begin");
            let mut estimate = if let Some(ref mut provider) = pose_provider {
                // Hand the D435's color-aligned metric depth to the
                // provider for THIS frame; it replaces the DAv2 stage.
                let metric =
                    crate::tracking::rtmw3d_with_depth::build_metric_frame_from_d435(&rs_frame);
                provider.set_external_depth(metric);
                provider.estimate_pose(&rs_frame.rgb, width, height, frame_index)
            } else {
                pose_estimation::estimate_pose(&rs_frame.rgb, width, height, frame_index)
            };
            // Stamp the device capture time onto the published sample —
            // the solver's measurement filters derive their dt from
            // consecutive capture timestamps (render/wall clocks say
            // nothing about when the subject actually moved).
            estimate.skeleton.capture_timestamp_ms = Some(rs_frame.timestamp_ms);

            // Live debug channel: publish camera + 2D keypoints + source
            // arm joints for an external overlay (no-op unless the debug
            // flag file exists). Before the estimate is moved below.
            debug_channel::dump_observation(frame_index, &rs_frame.rgb, width, height, &estimate);
            // Full-res aligned depth snapshot once a second: lets an
            // external audit read the exact depth pixels under any
            // keypoint of the live session, without stealing the camera.
            if frame_index % 30 == 0 {
                debug_channel::dump_depth_snapshot(
                    frame_index,
                    &rs_frame.depth_raw,
                    width,
                    height,
                    rs_frame.depth_units,
                );
            }

            // Session recording: append this capture frame to the offline
            // time series (no-op unless VULVATAR_RECORD is set). Placed
            // beside the debug channel because both need the estimate
            // before it is moved into the mailbox — but unlike that
            // latest-only channel this one keeps every frame, so a
            // misbehaviour that lasted three frames is still there
            // afterwards. The raw colour/depth are passed by reference and
            // only cloned for frames that trip the jump trigger.
            session_record::record(
                frame_index,
                &estimate.skeleton,
                &rs_frame.rgb,
                width,
                height,
                &rs_frame.depth_raw,
                rs_frame.depth_units,
                CameraIntrinsics {
                    fx: rs_frame.intrinsics.fx,
                    fy: rs_frame.intrinsics.fy,
                    cx: rs_frame.intrinsics.cx,
                    cy: rs_frame.intrinsics.cy,
                    width: rs_frame.intrinsics.width,
                    height: rs_frame.intrinsics.height,
                },
                rs_frame.timestamp_ms,
            );

            let frame = Some(downscale_for_gui(&rs_frame.rgb, width, height, 320));
            mailbox.publish_estimate(estimate, frame);
            stagelog::mark(frame_index, "publish");
        }

        let _ = capture_handle.join();
        session_record::finish();
        info!("tracking-worker: stopped");
    }
}

/// One frame handed from the capture thread to the inference thread.
#[cfg(feature = "realsense")]
struct CaptureItem {
    frame: realsense::RealSenseFrame,
    index: u64,
}

/// Camera-owning loop of the `tracking-capture` thread: open the device
/// (reporting the outcome once through `open_tx`), then grab frames and
/// overwrite the shared [`latest_cell::LatestCell`] with the freshest one.
/// Persistent grab failures tear the pipeline down and reopen it with
/// backoff ([`GrabRecovery`]); desynced framesets are dropped without
/// counting toward that. Closes the cell on exit, which is the inference
/// thread's wake-up-and-quit signal.
#[cfg(feature = "realsense")]
fn capture_loop(
    cell: Arc<latest_cell::LatestCell<CaptureItem>>,
    running: Arc<AtomicBool>,
    mailbox: TrackingMailbox,
    open_tx: std::sync::mpsc::Sender<Result<(u32, u32), realsense::OpenFailure>>,
    width: u32,
    height: u32,
    fps: u32,
) {
    let mut capture = match realsense::RealSenseCapture::open(width, height, fps) {
        Ok(c) => {
            let _ = open_tx.send(Ok((c.width(), c.height())));
            Some(c)
        }
        Err(e) => {
            let _ = open_tx.send(Err(e));
            cell.close();
            return;
        }
    };
    drop(open_tx);

    let mut frame_index: u64 = 0;
    let mut recovery = GrabRecovery::default();
    let mut sync_drops: u64 = 0;

    while running.load(Ordering::SeqCst) {
        let Some(cap) = capture.as_mut() else {
            // Reconnect path: the previous capture was torn down after a
            // persistent failure streak.
            match recovery.next_reopen_backoff() {
                None => {
                    error!(
                        "tracking-capture: camera did not come back after {} reconnect attempts, stopping",
                        GrabRecovery::MAX_REOPEN_ATTEMPTS
                    );
                    mailbox.report_error(
                        t!(
                            "tracking.error_camera_stopped",
                            count = recovery.total_errors()
                        ),
                        TrackingErrorLevel::Blocking,
                    );
                    break;
                }
                Some(backoff) => {
                    if !sleep_while_running(&running, backoff) {
                        break;
                    }
                    match realsense::RealSenseCapture::open(width, height, fps) {
                        Ok(c) => {
                            info!("tracking-capture: camera reconnected");
                            capture = Some(c);
                            recovery.on_reopen_success();
                        }
                        Err(e) => warn!("tracking-capture: reconnect attempt failed: {e}"),
                    }
                }
            }
            continue;
        };

        match cap.grab_frame() {
            Ok(frame) => {
                recovery.on_success();
                stagelog::mark(frame_index, "capture_put");
                cell.put(CaptureItem {
                    frame,
                    index: frame_index,
                });
                frame_index += 1;
            }
            Err(realsense::GrabError::SyncMismatch { delta_ms }) => {
                // Healthy stream, unusable frameset. Log throttled —
                // in a dim room with a struggling exposure this can
                // recur for a while.
                sync_drops += 1;
                if sync_drops.is_power_of_two() {
                    warn!(
                        "tracking-capture: dropped depth/color-desynced frameset (#{sync_drops}, Δ {delta_ms:.1} ms)"
                    );
                }
            }
            Err(realsense::GrabError::Capture(msg)) => match recovery.on_capture_error(&msg) {
                GrabAction::Retry => {}
                GrabAction::LogAndRetry => {
                    error!("tracking-capture: frame grab error: {msg}");
                }
                GrabAction::Reopen => {
                    warn!(
                        "tracking-capture: {} consecutive grab failures — reopening the camera",
                        GrabRecovery::REOPEN_AFTER
                    );
                    capture = None;
                }
            },
        }
    }

    cell.close();
    info!("tracking-capture: stopped");
}

/// Sleep `total` in short slices, re-checking `running` between slices so
/// a stop request interrupts a reconnect backoff promptly. Returns `false`
/// when `running` flipped off during the wait.
#[cfg(feature = "realsense")]
fn sleep_while_running(running: &AtomicBool, total: Duration) -> bool {
    let mut remaining = total;
    while remaining > Duration::ZERO {
        if !running.load(Ordering::SeqCst) {
            return false;
        }
        let step = remaining.min(Duration::from_millis(100));
        thread::sleep(step);
        remaining = remaining.saturating_sub(step);
    }
    running.load(Ordering::SeqCst)
}

/// Grab-failure recovery ladder for the capture loop. Pure state machine
/// (no I/O, no clocks) so the retry → log-on-change → reopen → give-up
/// policy is unit-testable without a camera.
#[derive(Debug, Default)]
struct GrabRecovery {
    consecutive_errors: u32,
    total_errors: u64,
    last_logged: Option<String>,
    reopen_attempts: u32,
}

#[derive(Debug, PartialEq, Eq)]
enum GrabAction {
    /// Same error as already logged — retry silently.
    Retry,
    /// New (or changed) error message — log it, then retry. Logging on
    /// *change* rather than only on the first error of a streak keeps a
    /// mid-streak transition (timeout → device disconnected) visible.
    LogAndRetry,
    /// Failure streak exhausted the retry budget — tear the pipeline
    /// down and reopen the device.
    Reopen,
}

impl GrabRecovery {
    /// Consecutive capture errors before a reopen. At the 500 ms steady
    /// grab timeout this is ~5 s of a wedged / unplugged camera.
    const REOPEN_AFTER: u32 = 10;
    /// Reopen attempts (each preceded by [`Self::next_reopen_backoff`])
    /// before giving up for good.
    const MAX_REOPEN_ATTEMPTS: u32 = 5;

    fn on_success(&mut self) {
        self.consecutive_errors = 0;
        self.last_logged = None;
        self.reopen_attempts = 0;
    }

    fn on_capture_error(&mut self, msg: &str) -> GrabAction {
        self.consecutive_errors += 1;
        self.total_errors += 1;
        if self.consecutive_errors >= Self::REOPEN_AFTER {
            self.consecutive_errors = 0;
            return GrabAction::Reopen;
        }
        if self.last_logged.as_deref() != Some(msg) {
            self.last_logged = Some(msg.to_string());
            GrabAction::LogAndRetry
        } else {
            GrabAction::Retry
        }
    }

    /// Backoff before the next reopen attempt: 0.5 s doubling to an 8 s
    /// cap. `None` once the attempt budget is spent.
    fn next_reopen_backoff(&mut self) -> Option<Duration> {
        if self.reopen_attempts >= Self::MAX_REOPEN_ATTEMPTS {
            return None;
        }
        let backoff = Duration::from_millis(500u64 << self.reopen_attempts.min(4));
        self.reopen_attempts += 1;
        Some(backoff)
    }

    fn on_reopen_success(&mut self) {
        self.on_success();
    }

    fn total_errors(&self) -> u64 {
        self.total_errors
    }
}

#[cfg(test)]
mod grab_recovery_tests {
    use super::*;

    #[test]
    fn logs_on_error_message_change_not_just_first() {
        let mut r = GrabRecovery::default();
        assert_eq!(r.on_capture_error("timeout"), GrabAction::LogAndRetry);
        assert_eq!(r.on_capture_error("timeout"), GrabAction::Retry);
        // The mid-streak transition must be visible.
        assert_eq!(r.on_capture_error("disconnected"), GrabAction::LogAndRetry);
        assert_eq!(r.on_capture_error("disconnected"), GrabAction::Retry);
    }

    #[test]
    fn reopen_after_persistent_streak_and_success_resets() {
        let mut r = GrabRecovery::default();
        for _ in 0..GrabRecovery::REOPEN_AFTER - 1 {
            let a = r.on_capture_error("timeout");
            assert_ne!(a, GrabAction::Reopen);
        }
        assert_eq!(r.on_capture_error("timeout"), GrabAction::Reopen);
        // A successful grab resets the streak entirely.
        r.on_success();
        assert_eq!(r.on_capture_error("timeout"), GrabAction::LogAndRetry);
    }

    #[test]
    fn reopen_backoff_doubles_then_gives_up() {
        let mut r = GrabRecovery::default();
        let mut seen = Vec::new();
        while let Some(b) = r.next_reopen_backoff() {
            seen.push(b.as_millis() as u64);
        }
        assert_eq!(seen, vec![500, 1000, 2000, 4000, 8000]);
        assert!(r.next_reopen_backoff().is_none(), "budget spent");
        // A successful reopen restores the full budget.
        r.on_reopen_success();
        assert_eq!(r.next_reopen_backoff(), Some(Duration::from_millis(500)));
    }
}

impl Drop for TrackingWorker {
    fn drop(&mut self) {
        // During normal lifecycle management `stop()` should already have
        // joined the thread. If destruction happens while a camera backend is
        // still wedged inside a blocking call, there is no non-blocking way to
        // recover in `Drop`; requesting stop one last time is still the least
        // surprising behaviour.
        let _ = self.stop();
    }
}

/// Downscale an RGB frame to a maximum width, preserving aspect ratio.
fn downscale_for_gui(rgb_data: &[u8], src_w: u32, src_h: u32, max_w: u32) -> PreviewFrame {
    if src_w <= max_w {
        return PreviewFrame {
            rgb_data: rgb_data.to_vec(),
            width: src_w,
            height: src_h,
        };
    }
    let scale = max_w as f32 / src_w as f32;
    let dst_w = max_w;
    let dst_h = ((src_h as f32 * scale).round() as u32).max(1);
    let mut out = vec![0u8; (dst_w * dst_h * 3) as usize];
    for y in 0..dst_h {
        let sy = ((y as f32 / scale) as u32).min(src_h - 1);
        for x in 0..dst_w {
            let sx = ((x as f32 / scale) as u32).min(src_w - 1);
            let si = ((sy * src_w + sx) * 3) as usize;
            let di = ((y * dst_w + x) * 3) as usize;
            out[di] = rgb_data[si];
            out[di + 1] = rgb_data[si + 1];
            out[di + 2] = rgb_data[si + 2];
        }
    }
    PreviewFrame {
        rgb_data: out,
        width: dst_w,
        height: dst_h,
    }
}
