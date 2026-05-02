//! Shared "2D keypoints + per-pixel metric depth → SourceSkeleton"
//! pipeline used by both [`super::cigpose_metric_depth`] and
//! [`super::rtmw3d_with_depth`].
//!
//! The two providers feed differently-sourced 2D keypoints (CIGPose
//! and RTMW3D respectively) and differently-derived depth maps (MoGe
//! true-metric vs DAv2 calibrated-relative), but both produce a
//! [`super::metric_depth::MoGeFrame`]-shaped point cloud and consume
//! [`super::cigpose::DecodedJoint2d`]-shaped 2D landmarks. The
//! skeleton-building math (origin selection, axis flips, hand chain
//! attachment, head-pose derivation) is identical past that point —
//! keeping it single-sourced here avoids the two providers drifting.

use crate::asset::HumanoidBone;

use super::cigpose::DecodedJoint2d;
use super::face_mediapipe::{derive_face_bbox, FaceBbox};
use super::metric_depth::MoGeFrame;
use super::{FacePose, SourceJoint, SourceSkeleton};

/// Per-keypoint visibility floor — anything below is treated as "no
/// detection" rather than a low-confidence detection. The user-tunable
/// `joint_confidence_threshold` slider lives in the solver.
pub(super) const KEYPOINT_VISIBILITY_FLOOR: f32 = 0.05;

/// Half-size of the square depth-sampling window when reading metric
/// depth at a 2D keypoint. Small enough to track fingers without
/// dragging in neighbouring depth, large enough that single-pixel
/// model artefacts get out-voted by the median.
pub(super) const SAMPLE_RADIUS_PX: i32 = 3;

const NUM_JOINTS: usize = 133;

// ---------------------------------------------------------------------------
// COCO-Wholebody index → HumanoidBone tables. Same convention RTMW3D
// uses; selfie mirror = subject's anatomical-left → avatar `Right*`
// bone. Single-sourced here so any future tweak (e.g. excluding a
// noisy keypoint) lands in both providers.
// ---------------------------------------------------------------------------

pub(super) const COCO_BODY: &[(usize, HumanoidBone)] = &[
    (5, HumanoidBone::RightShoulder),
    (5, HumanoidBone::RightUpperArm),
    (6, HumanoidBone::LeftShoulder),
    (6, HumanoidBone::LeftUpperArm),
    (7, HumanoidBone::RightLowerArm),
    (8, HumanoidBone::LeftLowerArm),
    (11, HumanoidBone::RightUpperLeg),
    (12, HumanoidBone::LeftUpperLeg),
    (13, HumanoidBone::RightLowerLeg),
    (14, HumanoidBone::LeftLowerLeg),
    (15, HumanoidBone::RightFoot),
    (16, HumanoidBone::LeftFoot),
];

pub(super) const COCO_FOOT_TIPS: &[(usize, HumanoidBone)] = &[
    (17, HumanoidBone::RightFoot),
    (20, HumanoidBone::LeftFoot),
];

pub(super) const HAND_PHALANGES_RIGHT: &[(usize, HumanoidBone)] = &[
    (1, HumanoidBone::RightThumbProximal),
    (2, HumanoidBone::RightThumbIntermediate),
    (3, HumanoidBone::RightThumbDistal),
    (5, HumanoidBone::RightIndexProximal),
    (6, HumanoidBone::RightIndexIntermediate),
    (7, HumanoidBone::RightIndexDistal),
    (9, HumanoidBone::RightMiddleProximal),
    (10, HumanoidBone::RightMiddleIntermediate),
    (11, HumanoidBone::RightMiddleDistal),
    (13, HumanoidBone::RightRingProximal),
    (14, HumanoidBone::RightRingIntermediate),
    (15, HumanoidBone::RightRingDistal),
    (17, HumanoidBone::RightLittleProximal),
    (18, HumanoidBone::RightLittleIntermediate),
    (19, HumanoidBone::RightLittleDistal),
];

pub(super) const HAND_TIPS_RIGHT: &[(usize, HumanoidBone)] = &[
    (4, HumanoidBone::RightThumbDistal),
    (8, HumanoidBone::RightIndexDistal),
    (12, HumanoidBone::RightMiddleDistal),
    (16, HumanoidBone::RightRingDistal),
    (20, HumanoidBone::RightLittleDistal),
];

pub(super) const HAND_PHALANGES_LEFT: &[(usize, HumanoidBone)] = &[
    (1, HumanoidBone::LeftThumbProximal),
    (2, HumanoidBone::LeftThumbIntermediate),
    (3, HumanoidBone::LeftThumbDistal),
    (5, HumanoidBone::LeftIndexProximal),
    (6, HumanoidBone::LeftIndexIntermediate),
    (7, HumanoidBone::LeftIndexDistal),
    (9, HumanoidBone::LeftMiddleProximal),
    (10, HumanoidBone::LeftMiddleIntermediate),
    (11, HumanoidBone::LeftMiddleDistal),
    (13, HumanoidBone::LeftRingProximal),
    (14, HumanoidBone::LeftRingIntermediate),
    (15, HumanoidBone::LeftRingDistal),
    (17, HumanoidBone::LeftLittleProximal),
    (18, HumanoidBone::LeftLittleIntermediate),
    (19, HumanoidBone::LeftLittleDistal),
];

pub(super) const HAND_TIPS_LEFT: &[(usize, HumanoidBone)] = &[
    (4, HumanoidBone::LeftThumbDistal),
    (8, HumanoidBone::LeftIndexDistal),
    (12, HumanoidBone::LeftMiddleDistal),
    (16, HumanoidBone::LeftRingDistal),
    (20, HumanoidBone::LeftLittleDistal),
];

// ---------------------------------------------------------------------------
// Depth sampling
// ---------------------------------------------------------------------------

/// Sample a metric-depth frame's point cloud at a frame-relative
/// `(nx, ny)` keypoint with a local-window median. Returns `None` when
/// the keypoint falls outside the frame or every sample in the window
/// is masked / non-finite.
pub(super) fn sample_metric_point(frame: &MoGeFrame, nx: f32, ny: f32) -> Option<[f32; 3]> {
    if !nx.is_finite() || !ny.is_finite() {
        return None;
    }
    let w = frame.width as i32;
    let h = frame.height as i32;
    let cx = (nx * frame.width as f32).round() as i32;
    let cy = (ny * frame.height as f32).round() as i32;
    if cx < 0 || cy < 0 || cx >= w || cy >= h {
        return None;
    }
    let cap = ((SAMPLE_RADIUS_PX * 2 + 1) * (SAMPLE_RADIUS_PX * 2 + 1)) as usize;
    let mut xs = Vec::with_capacity(cap);
    let mut ys = Vec::with_capacity(cap);
    let mut zs = Vec::with_capacity(cap);
    for yy in (cy - SAMPLE_RADIUS_PX).max(0)..=(cy + SAMPLE_RADIUS_PX).min(h - 1) {
        for xx in (cx - SAMPLE_RADIUS_PX).max(0)..=(cx + SAMPLE_RADIUS_PX).min(w - 1) {
            let idx = yy as usize * frame.width as usize + xx as usize;
            let [px, py, pz] = frame.points_m[idx];
            if px.is_finite() && py.is_finite() && pz.is_finite() && pz > 0.0 {
                xs.push(px);
                ys.push(py);
                zs.push(pz);
            }
        }
    }
    Some([median(xs)?, median(ys)?, median(zs)?])
}

fn median(mut values: Vec<f32>) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    Some(values[values.len() / 2])
}

// ---------------------------------------------------------------------------
// Skeleton building
// ---------------------------------------------------------------------------

/// Resolve the body anchor origin in metric camera-space coords.
/// Hip mid is preferred (CIGPose 11/12); falls back to shoulder mid
/// (5/6) for upper-body crops. Returns `None` when neither pair is
/// usable — the frame produces no skeleton.
pub(super) fn resolve_origin_metric(
    joints: &[DecodedJoint2d],
    depth: &MoGeFrame,
) -> Option<([f32; 3], /*anchor_was_hip*/ bool, /*anchor_score*/ f32)> {
    if joints.len() < NUM_JOINTS {
        return None;
    }
    let try_pair = |a: &DecodedJoint2d, b: &DecodedJoint2d| -> Option<[f32; 3]> {
        let l = sample_metric_point(depth, a.nx, a.ny)?;
        let r = sample_metric_point(depth, b.nx, b.ny)?;
        Some([
            (l[0] + r[0]) * 0.5,
            (l[1] + r[1]) * 0.5,
            (l[2] + r[2]) * 0.5,
        ])
    };

    let lh = &joints[11];
    let rh = &joints[12];
    let hip_score = lh.score.min(rh.score);
    if hip_score >= KEYPOINT_VISIBILITY_FLOOR {
        if let Some(o) = try_pair(lh, rh) {
            return Some((o, true, hip_score));
        }
    }

    let ls = &joints[5];
    let rs = &joints[6];
    let shoulder_score = ls.score.min(rs.score);
    if shoulder_score >= KEYPOINT_VISIBILITY_FLOOR {
        if let Some(o) = try_pair(ls, rs) {
            return Some((o, false, shoulder_score));
        }
    }
    None
}

/// Build a [`SourceSkeleton`] from 2D keypoints + sampled metric
/// depth using a pre-resolved origin. Camera-space metric `(x-right,
/// y-down, z-forward)` is flipped on every axis to match the
/// source-space convention `(selfie-mirror x, y-up, z toward
/// camera)`. Selfie mirror: subject's anatomical-left landmarks
/// drive avatar `Right*` bones.
///
/// Wrist position = centroid of the four MCPs (rationale: the
/// hand-track wrist landmark is unreliable under occlusion). Face
/// pose / expressions are intentionally left empty here — the caller
/// populates them via [`derive_face_pose_metric`] and the optional
/// FaceMesh cascade.
pub(super) fn build_skeleton(
    frame_index: u64,
    joints: &[DecodedJoint2d],
    depth: &MoGeFrame,
    origin: [f32; 3],
    anchor_was_hip: bool,
    anchor_score: f32,
) -> SourceSkeleton {
    let mut sk = SourceSkeleton::empty(frame_index);
    if joints.len() < NUM_JOINTS {
        return sk;
    }

    let to_source = |p: [f32; 3]| -> [f32; 3] {
        [
            -(p[0] - origin[0]),
            -(p[1] - origin[1]),
            -(p[2] - origin[2]),
        ]
    };

    let resolve = |idx: usize| -> Option<SourceJoint> {
        if idx >= joints.len() {
            return None;
        }
        let j = &joints[idx];
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            return None;
        }
        let pos_cam = sample_metric_point(depth, j.nx, j.ny)?;
        Some(SourceJoint {
            position: to_source(pos_cam),
            confidence: j.score,
        })
    };

    if anchor_was_hip {
        sk.joints.insert(
            HumanoidBone::Hips,
            SourceJoint {
                position: [0.0, 0.0, 0.0],
                confidence: anchor_score,
            },
        );
    }

    const LOWER_BODY_FIRST_IDX: usize = 11;
    for &(idx, bone) in COCO_BODY {
        if !anchor_was_hip && idx >= LOWER_BODY_FIRST_IDX {
            continue;
        }
        if let Some(joint) = resolve(idx) {
            sk.joints.insert(bone, joint);
        }
    }

    if anchor_was_hip {
        for &(idx, bone) in COCO_FOOT_TIPS {
            if let Some(joint) = resolve(idx) {
                sk.fingertips.insert(bone, joint);
            }
        }
    }

    attach_hand(
        &mut sk,
        joints,
        depth,
        91,
        HumanoidBone::RightHand,
        HAND_PHALANGES_RIGHT,
        HAND_TIPS_RIGHT,
        &to_source,
    );
    attach_hand(
        &mut sk,
        joints,
        depth,
        112,
        HumanoidBone::LeftHand,
        HAND_PHALANGES_LEFT,
        HAND_TIPS_LEFT,
        &to_source,
    );

    let mut sum = 0.0_f32;
    let mut n = 0_u32;
    for j in joints.iter().take(17) {
        sum += j.score;
        n += 1;
    }
    sk.overall_confidence = if n > 0 { sum / n as f32 } else { 0.0 };

    sk.face = None;
    sk.expressions = Vec::new();
    sk
}

#[allow(clippy::too_many_arguments)]
fn attach_hand<F>(
    sk: &mut SourceSkeleton,
    joints: &[DecodedJoint2d],
    depth: &MoGeFrame,
    base_index: usize,
    wrist_bone: HumanoidBone,
    phalanges: &[(usize, HumanoidBone)],
    tips: &[(usize, HumanoidBone)],
    to_source: &F,
) where
    F: Fn([f32; 3]) -> [f32; 3],
{
    const MCP_LOCALS: [usize; 4] = [5, 9, 13, 17];
    let mut sum = [0.0_f32; 3];
    let mut min_conf = f32::INFINITY;
    let mut found = 0_u32;
    for &local in &MCP_LOCALS {
        let global = base_index + local;
        if global >= joints.len() {
            continue;
        }
        let j = &joints[global];
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            continue;
        }
        let p_cam = match sample_metric_point(depth, j.nx, j.ny) {
            Some(p) => p,
            None => continue,
        };
        let p = to_source(p_cam);
        sum[0] += p[0];
        sum[1] += p[1];
        sum[2] += p[2];
        min_conf = min_conf.min(j.score);
        found += 1;
    }
    if found < 3 {
        return;
    }
    let inv = 1.0 / found as f32;
    sk.joints.insert(
        wrist_bone,
        SourceJoint {
            position: [sum[0] * inv, sum[1] * inv, sum[2] * inv],
            confidence: min_conf,
        },
    );

    for &(local_idx, bone) in phalanges {
        let global = base_index + local_idx;
        if global >= joints.len() {
            continue;
        }
        let j = &joints[global];
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            continue;
        }
        if let Some(p_cam) = sample_metric_point(depth, j.nx, j.ny) {
            sk.joints.insert(
                bone,
                SourceJoint {
                    position: to_source(p_cam),
                    confidence: j.score,
                },
            );
        }
    }
    for &(local_idx, bone) in tips {
        let global = base_index + local_idx;
        if global >= joints.len() {
            continue;
        }
        let j = &joints[global];
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            continue;
        }
        if let Some(p_cam) = sample_metric_point(depth, j.nx, j.ny) {
            sk.fingertips.insert(
                bone,
                SourceJoint {
                    position: to_source(p_cam),
                    confidence: j.score,
                },
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Face cascade helpers
// ---------------------------------------------------------------------------

/// Derive head yaw / pitch / roll from face keypoints 0..=4 (nose /
/// left_eye / right_eye / left_ear / right_ear), each lifted into 3D
/// via metric-depth sampling and routed through the same axis flip +
/// hip-origin convention as the body skeleton.
///
/// Logic mirrors [`super::rtmw3d::face::derive_face_pose_from_body`]:
/// ear-line yaw, eye-line roll, eye-mid → nose pitch. Confidence is
/// the min of the 5 input scores. Includes the back-ear reflection
/// fallback for >45° head turns.
pub(super) fn derive_face_pose_metric(
    joints: &[DecodedJoint2d],
    depth: &MoGeFrame,
    origin: [f32; 3],
) -> Option<FacePose> {
    if joints.len() < 5 {
        return None;
    }
    for j in joints.iter().take(5) {
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            return None;
        }
    }

    let to_src = |idx: usize| -> Option<[f32; 3]> {
        let j = &joints[idx];
        let p_cam = sample_metric_point(depth, j.nx, j.ny)?;
        Some([
            -(p_cam[0] - origin[0]),
            -(p_cam[1] - origin[1]),
            -(p_cam[2] - origin[2]),
        ])
    };

    let nose = to_src(0)?;
    // Selfie mirror: subject's left maps to avatar's right.
    let avatar_right_eye = to_src(1)?;
    let avatar_left_eye = to_src(2)?;
    let avatar_right_ear = to_src(3)?;
    let avatar_left_ear = to_src(4)?;

    const EAR_OCCLUSION_RATIO: f32 = 0.5;
    let s_right_ear = joints[3].score;
    let s_left_ear = joints[4].score;
    let (right_ear_xz, left_ear_xz) = if s_right_ear < s_left_ear * EAR_OCCLUSION_RATIO {
        let recon = [
            2.0 * nose[0] - avatar_left_ear[0],
            avatar_left_ear[1],
            2.0 * nose[2] - avatar_left_ear[2],
        ];
        (recon, avatar_left_ear)
    } else if s_left_ear < s_right_ear * EAR_OCCLUSION_RATIO {
        let recon = [
            2.0 * nose[0] - avatar_right_ear[0],
            avatar_right_ear[1],
            2.0 * nose[2] - avatar_right_ear[2],
        ];
        (avatar_right_ear, recon)
    } else {
        (avatar_right_ear, avatar_left_ear)
    };
    let ear_dx = left_ear_xz[0] - right_ear_xz[0];
    let ear_dz = left_ear_xz[2] - right_ear_xz[2];
    let ear_dist = (ear_dx * ear_dx + ear_dz * ear_dz).sqrt();
    if ear_dist < 0.02 {
        return None;
    }
    let yaw = ear_dz.atan2(ear_dx);

    let eye_dx = avatar_left_eye[0] - avatar_right_eye[0];
    let eye_dy = avatar_left_eye[1] - avatar_right_eye[1];
    let roll = eye_dy.atan2(eye_dx);

    let mid_eye_y = (avatar_left_eye[1] + avatar_right_eye[1]) * 0.5;
    let face_width = (eye_dx * eye_dx + eye_dy * eye_dy).sqrt().max(0.02);
    let pitch_signal = (mid_eye_y - nose[1]) / face_width;
    let pitch = pitch_signal.clamp(-2.0, 2.0).atan();

    let conf = joints[..5].iter().map(|j| j.score).fold(1.0_f32, f32::min);
    Some(FacePose {
        yaw,
        pitch,
        roll,
        confidence: conf,
    })
}

/// Pixel-space face bbox derived from face-68 landmarks (indices
/// 23..=90). Mirrors [`super::rtmw3d::face::build_face_bbox_from_joints`]
/// but consumes the [`DecodedJoint2d`] type.
pub(super) fn build_face_bbox_from_joints_2d(
    joints: &[DecodedJoint2d],
    width: u32,
    height: u32,
) -> Option<FaceBbox> {
    let mut points_px: Vec<(f32, f32)> = Vec::with_capacity(68);
    for i in 23..=90 {
        let Some(j) = joints.get(i) else { continue };
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            continue;
        }
        points_px.push((j.nx * width as f32, j.ny * height as f32));
    }
    derive_face_bbox(&points_px, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_frame(width: u32, height: u32, point: [f32; 3]) -> MoGeFrame {
        let pixels = (width * height) as usize;
        MoGeFrame {
            width,
            height,
            points_m: vec![point; pixels],
        }
    }

    #[test]
    fn sample_metric_point_returns_pixel() {
        let frame = dummy_frame(644, 476, [0.5, 0.4, 1.2]);
        let p = sample_metric_point(&frame, 0.5, 0.5).unwrap();
        assert!((p[0] - 0.5).abs() < 1e-6);
        assert!((p[1] - 0.4).abs() < 1e-6);
        assert!((p[2] - 1.2).abs() < 1e-6);
    }

    #[test]
    fn sample_outside_frame_returns_none() {
        let frame = dummy_frame(644, 476, [0.0, 0.0, 1.0]);
        assert!(sample_metric_point(&frame, -0.1, 0.5).is_none());
        assert!(sample_metric_point(&frame, 1.1, 0.5).is_none());
    }
}
