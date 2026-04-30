//! Head pose (yaw / pitch / roll) derivation from RTMW3D's body face
//! keypoints, plus the dlib-68 face bbox we hand to the FaceMesh
//! cascade. Everything in this file consumes [`super::decode::DecodedJoint`]
//! and produces source-space outputs (`FacePose`, `FaceBbox`) — no
//! ONNX inference of its own.

use crate::asset::HumanoidBone;

use super::super::face_mediapipe::{derive_face_bbox, FaceBbox};
use super::super::{FacePose, SourceSkeleton};
use super::consts::KEYPOINT_VISIBILITY_FLOOR;
use super::decode::DecodedJoint;

/// Derive head yaw / pitch / roll from RTMW3D's body face keypoints
/// (0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear) in
/// source-skeleton 3D coords. Returns `None` if any keypoint is below
/// the confidence floor or the eye/ear spans are too small to give a
/// reliable angle.
///
/// All values are already mirrored at the source level (subject's
/// left ear ends up at source `+x`), so the sign conventions match
/// `quat_from_euler_ypr`.
pub(super) fn derive_face_pose_from_body(
    skeleton: &SourceSkeleton,
    joints: &[DecodedJoint],
) -> Option<FacePose> {
    if joints.len() < 5 {
        return None;
    }
    // Face pose math (ear-line yaw, eye-line roll, eye-to-nose pitch)
    // is well-defined as long as the keypoints actually exist and the
    // ear and eye baselines are non-degenerate (checked further down).
    // The aggregated confidence is the min over the 5 inputs, so the
    // solver can apply its own face threshold downstream — that's why
    // we only reject blatant non-detections here, not borderline ones.
    for j in joints.iter().take(5) {
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            return None;
        }
    }
    // We need positions in source-skeleton coords. The COCO body
    // keypoints 0..=4 are not stored in `skeleton.joints` (only
    // shoulders / arms / legs are), so reconstruct via the same
    // `to_source` math. We extract hip mid from the stored Hips
    // joint (which is `(0, 0, 0)` by definition of the origin),
    // then read the 5 face keypoints in normalised space and run
    // them through the same axis flip used elsewhere.
    let hip = (joints[11].nx + joints[12].nx) * 0.5;
    let aspect = skeleton
        .joints
        .get(&HumanoidBone::LeftShoulder)
        .or_else(|| skeleton.joints.get(&HumanoidBone::RightShoulder))
        .map(|s| s.position[0].abs())
        .unwrap_or(0.0)
        / ((joints[5].nx - joints[6].nx).abs() / 2.0).max(1e-3);
    // Conservative aspect estimate: if we can't read it from the
    // stored shoulders fall back to assuming square (1.0).
    let aspect = if aspect.is_finite() && aspect > 0.5 && aspect < 4.0 {
        aspect
    } else {
        1.0
    };
    let to_src = |j: &DecodedJoint| -> [f32; 3] {
        let sx = -(j.nx - hip) * aspect;
        let sy = -(j.ny - (joints[11].ny + joints[12].ny) * 0.5);
        let sz = -(j.nz - (joints[11].nz + joints[12].nz) * 0.5);
        [sx, sy, sz]
    };
    // Subject-POV indices map to avatar-POV after `to_src` (selfie
    // mirror): subject's left → avatar's right and vice versa. Bind
    // the avatar-frame names directly so the geometry below reads
    // without sign confusion.
    let nose = to_src(&joints[0]);
    let avatar_right_eye = to_src(&joints[1]); // subject left eye
    let avatar_left_eye = to_src(&joints[2]); // subject right eye
    let avatar_right_ear = to_src(&joints[3]); // subject left ear
    let avatar_left_ear = to_src(&joints[4]); // subject right ear

    // Ear line points avatar_left_ear → avatar_right_ear along +x at
    // rest, so dx = left.x - right.x is positive when the head faces
    // camera. As the head turns to camera-right (avatar yaw +) the
    // line tilts in XZ so dz becomes positive too. yaw = atan2(dz, dx).
    //
    // Back-ear robustness: when the head turns past ~45° one ear
    // becomes occluded and RTMW3D's SimCC head will still emit a
    // 3D position for it, but the Z component is essentially
    // hallucinated — typically pinned near the visible ear's depth,
    // collapsing `ear_dz` toward zero and stalling the avatar's yaw
    // tracking right when it should be most active. Detect that
    // case by score asymmetry and reconstruct the back ear as the
    // visible ear reflected through the nose's XZ position. Nose
    // is forward of the head's actual rotation axis so this introduces
    // a small constant bias in `ear_dz`; the per-session calibration
    // (`TrackingCalibration::apply_calibration`) absorbs the bias at
    // the neutral pose, leaving the *delta* — which is all the avatar
    // head bone consumes — close to correct.
    const EAR_OCCLUSION_RATIO: f32 = 0.5;
    let s_right_ear = joints[3].score;
    let s_left_ear = joints[4].score;
    let (right_ear_xz, left_ear_xz) = if s_right_ear < s_left_ear * EAR_OCCLUSION_RATIO {
        // Subject's left ear (joint 3 / avatar_right_ear) occluded.
        let recon = [
            2.0 * nose[0] - avatar_left_ear[0],
            avatar_left_ear[1],
            2.0 * nose[2] - avatar_left_ear[2],
        ];
        (recon, avatar_left_ear)
    } else if s_left_ear < s_right_ear * EAR_OCCLUSION_RATIO {
        // Subject's right ear (joint 4 / avatar_left_ear) occluded.
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
    if ear_dist < 0.01 {
        return None;
    }
    let yaw = ear_dz.atan2(ear_dx);

    // Eye line: same handedness. Roll positive = head tilts toward
    // avatar's left ear (avatar's left eye drops below the right).
    let eye_dx = avatar_left_eye[0] - avatar_right_eye[0];
    let eye_dy = avatar_left_eye[1] - avatar_right_eye[1];
    let roll = eye_dy.atan2(eye_dx);

    let mid_eye_y = (avatar_left_eye[1] + avatar_right_eye[1]) * 0.5;
    let face_width = (eye_dx * eye_dx + eye_dy * eye_dy).sqrt().max(0.01);
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

/// Pixel-space face bbox derived from RTMW3D's 68 face landmarks
/// (indices 23..=90, dlib 68-point convention with jawline / brows /
/// nose / eyes / mouth). Body keypoints 0..=4 (nose / eyes / ears)
/// give a centroid that's too high on the face — they cluster on the
/// upper half so 1.7× padding overshoots above the head and clips
/// the chin. The face-68 set spans the whole face including mouth
/// and jaw, so its centroid lands on the actual face centre. Returns
/// `None` when too few face keypoints clear the confidence floor.
pub(super) fn build_face_bbox_from_joints(
    joints: &[DecodedJoint],
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
