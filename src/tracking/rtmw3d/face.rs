//! Head pose (yaw / pitch / roll) derivation from RTMW3D's body face
//! keypoints, plus the dlib-68 face bbox we hand to the FaceMesh
//! cascade. Everything in this file consumes [`super::decode::DecodedJoint`]
//! and produces source-space outputs (`FacePose`, `FaceBbox`) — no
//! ONNX inference of its own.

use crate::asset::HumanoidBone;

use super::super::face_mediapipe::{derive_face_bbox, FaceBbox};
use super::super::{FacePose, SourceSkeleton};
use super::consts::{INPUT_H, INPUT_W, KEYPOINT_VISIBILITY_FLOOR};
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

    // PITCH from nose-below-eye-line, with an anatomical neutral
    // subtracted — mirroring the FaceMesh path
    // (`derive_face_pose_from_landmarks`, which subtracts its own
    // `PITCH_NEUTRAL_SIGNAL`). The body path never had this: at a frontal
    // neutral pose the nose tip sits a fixed fraction below the eye line,
    // so the raw ratio maps to a large spurious `+pitch` (chin-down). The
    // stored `neutral_face_pose` was meant to absorb it per-session, but
    // that channel is never populated (`TrackingCalibration` only ever
    // holds the default), so the bias reached the Head bone unmodified —
    // a forward-facing head decoded ~50° down.
    //
    // Computed in model-pixel space (`nx·INPUT_W, ny·INPUT_H`) rather than
    // source space so the ratio is independent of the shoulder-derived
    // `aspect` and metric depth, keeping the constant stable across
    // framings. Image y-down: nose below the eye line → positive → source
    // `+pitch` (chin drops). The selfie mirror is horizontal, so vertical
    // `ny` is untouched by it. The COCO 5-point set uses eye *centres*
    // (closer together than FaceMesh's outer eye corners), so the neutral
    // ratio is larger than that path's 0.49 — measured ≈ 1.25 on neutral
    // frames (live desk pose + the palms_front capture).
    const PITCH_NEUTRAL_SIGNAL: f32 = 1.25;
    let nose_py = joints[0].ny * INPUT_H as f32;
    let le_px = (joints[1].nx * INPUT_W as f32, joints[1].ny * INPUT_H as f32);
    let re_px = (joints[2].nx * INPUT_W as f32, joints[2].ny * INPUT_H as f32);
    let eye_mid_py = (le_px.1 + re_px.1) * 0.5;
    let inter_eye_px = ((le_px.0 - re_px.0).powi(2) + (le_px.1 - re_px.1).powi(2))
        .sqrt()
        .max(1.0);
    let pitch_signal = (nose_py - eye_mid_py) / inter_eye_px - PITCH_NEUTRAL_SIGNAL;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::source_skeleton::SourceJoint;

    fn dj(nx: f32, ny: f32, nz: f32) -> DecodedJoint {
        DecodedJoint { nx, ny, nz, score: 0.9 }
    }

    /// Front-facing synthetic face with the nose at `nose_py` pixels
    /// (model space is `INPUT_W×INPUT_H`). Eyes at py=150, inter-eye 32 px,
    /// ears symmetric at equal depth (yaw≈0), shoulders/hips set so
    /// `aspect` and the origin are well-defined. A frontal neutral head
    /// has the nose ~1.25·inter-eye (= 40 px) below the eye line.
    fn neutral_joints(nose_py: f32) -> Vec<DecodedJoint> {
        let w = INPUT_W as f32;
        let h = INPUT_H as f32;
        let px = |x: f32, y: f32, z: f32| dj(x / w, y / h, z);
        let mut j = vec![DecodedJoint::default(); 13];
        j[0] = px(144.0, nose_py, 0.5); // nose
        j[1] = px(160.0, 150.0, 0.5); // subject-left eye
        j[2] = px(128.0, 150.0, 0.5); // subject-right eye
        j[3] = px(174.0, 150.0, 0.5); // subject-left ear
        j[4] = px(114.0, 150.0, 0.5); // subject-right ear
        j[5] = px(200.0, 280.0, 0.5); // subject-left shoulder
        j[6] = px(88.0, 280.0, 0.5); // subject-right shoulder
        j[11] = px(168.0, 376.0, 0.5); // subject-left hip
        j[12] = px(120.0, 376.0, 0.5); // subject-right hip
        j
    }

    fn skeleton_with_shoulder() -> SourceSkeleton {
        let mut sk = SourceSkeleton::default();
        sk.joints.insert(
            HumanoidBone::LeftShoulder,
            SourceJoint { position: [0.18, 0.0, 0.0], confidence: 0.9, ..Default::default() },
        );
        sk
    }

    #[test]
    fn neutral_forward_face_has_near_zero_pitch() {
        // A front-facing head (nose the anatomical ~1.25·inter-eye below
        // the eye line) must decode to ~0 pitch — not the ~50°-down bias
        // the un-subtracted body path used to emit.
        let sk = skeleton_with_shoulder();
        let face = derive_face_pose_from_body(&sk, &neutral_joints(190.0)).unwrap();
        assert!(
            face.pitch.abs() < 0.12,
            "neutral face pitch should be ~0, got {} rad",
            face.pitch
        );
    }

    #[test]
    fn chin_down_is_positive_pitch() {
        // Nose dropped further below the eye line → chin-down → +pitch.
        let sk = skeleton_with_shoulder();
        let face = derive_face_pose_from_body(&sk, &neutral_joints(214.0)).unwrap();
        assert!(face.pitch > 0.3, "chin-down should be +pitch, got {}", face.pitch);
    }

    #[test]
    fn chin_up_is_negative_pitch() {
        // Nose lifted toward the eye line → chin-up → -pitch.
        let sk = skeleton_with_shoulder();
        let face = derive_face_pose_from_body(&sk, &neutral_joints(166.0)).unwrap();
        assert!(face.pitch < -0.3, "chin-up should be -pitch, got {}", face.pitch);
    }
}
