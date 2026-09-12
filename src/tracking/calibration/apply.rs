//! Solve-time application of the captured calibration to live samples.
//!
//! `TrackingCalibration::apply_calibration` transforms a published
//! `SourceSkeleton` into the user's calibrated frame before the solver
//! sees it: the neutral-body-yaw scene de-rotation, the per-source
//! neutral face-pose subtraction, and the neutral-expression rescale.
//! The anchor / EMA-seeding half of the calibration record is consumed
//! elsewhere (depth builder, solver) and is deliberately NOT applied
//! here — see the method doc for the mailbox raw-sample contract.
//!
//! The capture record itself (`PoseCalibration`) lives in the parent
//! module; `TrackingCalibration` is re-exported from there and at the
//! `crate::tracking` root.

use super::{rotate_xz, PoseCalibration};
use crate::tracking::{FaceSource, SourceSkeleton};

/// Per-session calibration.
///
/// `pose` — the user runs the `Calibrate Pose ▼` modal once per project
/// to capture a known-good pelvic / shoulder anchor reference; the
/// solver uses it to seed the root-translation EMA and
/// the depth pipeline used it to clamp the metric calibration scale
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
                        j.position = rotate_xz(j.position, theta);
                    }
                    for t in sample.fingertips.values_mut() {
                        t.position = rotate_xz(t.position, theta);
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
                        let d = rotate_xz(
                            [offset[0] - a0[0], offset[1] - a0[1], offset[2] - a0[2]],
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

#[cfg(test)]
mod calibration_apply_tests {
    use super::*;
    use crate::tracking::source_skeleton::SourceExpression;
    use crate::tracking::FacePose;

    fn cal_with_neutral(neutral: Vec<(String, f32)>) -> TrackingCalibration {
        TrackingCalibration {
            pose: Some(PoseCalibration {
                mode: crate::tracking::CalibrationMode::UpperBody,
                captured_at: String::new(),
                captured_at_unix: 0,
                frame_count: 1,
                q_neutral: None,
                anchor_x: 0.0,
                anchor_y: 0.0,
                anchor_depth_m: None,
                confidence: 1.0,
                anchor_depth_jitter_m: None,
                shoulder_span_m: None,
                x_range_observed: None,
                z_range_observed: None,
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
    use crate::tracking::source_skeleton::SourceJoint;

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
        assert_vec3_eq(
            sk.root_offset.unwrap(),
            frontal.root_offset.unwrap(),
            "offset",
        );
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
