#![allow(dead_code)]
use super::PoseEstimate;
#[cfg(feature = "inference")]
use super::{DetectionAnnotation, FacePose, SourceExpression, SourceJoint, SourceSkeleton};
#[cfg(feature = "inference")]
use crate::asset::HumanoidBone;
#[cfg(feature = "inference")]
use log::{error, info};
#[cfg(feature = "inference")]
use ndarray::Array4;
#[cfg(feature = "inference")]
use ort::session::Session;
#[cfg(feature = "inference")]
use ort::value::{Shape, TensorRef};

pub struct CigPoseInference {
    #[cfg(feature = "inference")]
    session: Session,
    #[cfg(feature = "inference")]
    input_w: u32,
    #[cfg(feature = "inference")]
    input_h: u32,
}

impl CigPoseInference {
    #[cfg(feature = "inference")]
    pub fn new(model_path: &str) -> std::result::Result<Self, String> {
        info!("Loading CIGPose model from {}", model_path);

        let session = Session::builder()
            .map_err(|e| format!("ORT build error: {}", e))?
            .with_intra_threads(4)
            .map_err(|e| format!("ORT threads error: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("ORT load error: {}", e))?;

        Ok(Self {
            session,
            input_w: 192,
            input_h: 256,
        })
    }

    #[cfg(not(feature = "inference"))]
    pub fn new(_model_path: &str) -> Result<Self, String> {
        Err("Inference feature is not enabled. Build with --features inference".to_string())
    }

    pub fn estimate_pose(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> PoseEstimate {
        #[cfg(feature = "inference")]
        return self.estimate_pose_internal(rgb_data, width, height, frame_index);

        #[cfg(not(feature = "inference"))]
        {
            super::pose_estimation::estimate_pose(rgb_data, width, height, frame_index)
        }
    }

    #[cfg(feature = "inference")]
    fn estimate_pose_internal(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> PoseEstimate {
        let (tensor, aspect) =
            preprocess_frame(rgb_data, width, height, self.input_w, self.input_h);

        let input = match TensorRef::from_array_view(&tensor) {
            Ok(v) => v,
            Err(e) => {
                error!("TensorRef creation failed: {}", e);
                return PoseEstimate {
                    skeleton: SourceSkeleton::empty(frame_index),
                    annotation: DetectionAnnotation::default(),
                };
            }
        };

        let outputs = match self.session.run(ort::inputs![input]) {
            Ok(out) => out,
            Err(e) => {
                error!("ONNX inference failed: {}", e);
                return PoseEstimate {
                    skeleton: SourceSkeleton::empty(frame_index),
                    annotation: DetectionAnnotation::default(),
                };
            }
        };

        let (shape_x, data_x) = match outputs["simcc_x"].try_extract_tensor::<f32>() {
            Ok(v) => v,
            Err(e) => {
                error!("ONNX simcc_x extraction failed: {}", e);
                return PoseEstimate {
                    skeleton: SourceSkeleton::empty(frame_index),
                    annotation: DetectionAnnotation::default(),
                };
            }
        };
        let (_shape_y, data_y) = match outputs["simcc_y"].try_extract_tensor::<f32>() {
            Ok(v) => v,
            Err(e) => {
                error!("ONNX simcc_y extraction failed: {}", e);
                return PoseEstimate {
                    skeleton: SourceSkeleton::empty(frame_index),
                    annotation: DetectionAnnotation::default(),
                };
            }
        };

        let split_ratio = 2.0;
        let kpts = decode_simcc_from_slice(data_x, data_y, shape_x, split_ratio);

        let skeleton =
            map_to_source_skeleton(&kpts, self.input_w, self.input_h, aspect, frame_index);

        let annotation = build_annotation(kpts, self.input_w, self.input_h, width, height);

        PoseEstimate {
            skeleton,
            annotation,
        }
    }
}

#[cfg(feature = "inference")]
fn preprocess_frame(
    rgb_data: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> (Array4<f32>, f32) {
    let mut tensor = Array4::<f32>::zeros((1, 3, dst_h as usize, dst_w as usize));

    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;
    let aspect = src_w as f32 / src_h as f32;

    // Standard ImageNet normalization used by MMPose
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    for y in 0..dst_h {
        let src_y = ((y as f32 * scale_y) as u32).min(src_h - 1);
        for x in 0..dst_w {
            let src_x = ((x as f32 * scale_x) as u32).min(src_w - 1);
            let offset = ((src_y * src_w + src_x) * 3) as usize;

            if offset + 2 < rgb_data.len() {
                let r = (rgb_data[offset] as f32 / 255.0 - mean[0]) / std[0];
                let g = (rgb_data[offset + 1] as f32 / 255.0 - mean[1]) / std[1];
                let b = (rgb_data[offset + 2] as f32 / 255.0 - mean[2]) / std[2];

                tensor[[0, 0, y as usize, x as usize]] = r;
                tensor[[0, 1, y as usize, x as usize]] = g;
                tensor[[0, 2, y as usize, x as usize]] = b;
            }
        }
    }

    (tensor, aspect)
}

#[cfg(feature = "inference")]
fn decode_simcc_from_slice(
    data_x: &[f32],
    data_y: &[f32],
    shape_x: &Shape,
    split_ratio: f32,
) -> Vec<(f32, f32, f32)> {
    let num_keypoints = shape_x[1] as usize;
    let x_channels = shape_x[2] as usize;

    let y_channels = if data_y.len() >= num_keypoints * x_channels && num_keypoints > 0 {
        data_y.len() / num_keypoints
    } else {
        256 * 2
    };

    let mut kpts = Vec::with_capacity(num_keypoints);

    for k in 0..num_keypoints {
        let x_start = k * x_channels;
        let x_end = (k + 1) * x_channels;
        let x_slice = &data_x[x_start..x_end.min(data_x.len())];

        let mut max_x_idx = 0;
        let mut max_x_val = f32::MIN;
        for (i, &val) in x_slice.iter().enumerate() {
            if val > max_x_val {
                max_x_val = val;
                max_x_idx = i;
            }
        }

        let y_start = k * y_channels;
        let y_end = (k + 1) * y_channels;
        let y_slice = &data_y[y_start..y_end.min(data_y.len())];

        let mut max_y_idx = 0;
        let mut max_y_val = f32::MIN;
        for (i, &val) in y_slice.iter().enumerate() {
            if val > max_y_val {
                max_y_val = val;
                max_y_idx = i;
            }
        }

        let x = max_x_idx as f32 / split_ratio;
        let y = max_y_idx as f32 / split_ratio;
        let conf = max_x_val.max(max_y_val);
        kpts.push((x, y, conf));
    }
    kpts
}

/// COCO-wholebody body-only skeleton connections (for the GUI overlay).
const SKELETON_CONNECTIONS: [(usize, usize); 16] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
];

#[cfg(feature = "inference")]
fn build_annotation(
    kpts: Vec<(f32, f32, f32)>,
    model_w: u32,
    model_h: u32,
    _orig_w: u32,
    _orig_h: u32,
) -> DetectionAnnotation {
    let mw = model_w as f32;
    let mh = model_h as f32;
    let norm_kpts: Vec<(f32, f32, f32)> = kpts
        .into_iter()
        .map(|(x, y, c)| (x / mw, y / mh, c))
        .collect();
    DetectionAnnotation {
        keypoints: norm_kpts,
        skeleton: SKELETON_CONNECTIONS.to_vec(),
        bounding_box: None,
    }
}

// COCO-WholeBody keypoint indices.
//
// Layout (133 keypoints):
//   0..=16    body
//   17..=22   feet
//   23..=90   face (68)
//   91..=111  left hand (21, MediaPipe layout)
//   112..=132 right hand (21)
//
// The hand layout per 21-point block is the MediaPipe convention:
//   +0  WRIST
//   +1..+4   THUMB  (CMC, MCP, IP, TIP)
//   +5..+8   INDEX  (MCP, PIP, DIP, TIP)
//   +9..+12  MIDDLE (MCP, PIP, DIP, TIP)
//   +13..+16 RING   (MCP, PIP, DIP, TIP)
//   +17..+20 LITTLE (MCP, PIP, DIP, TIP)
#[cfg(feature = "inference")]
mod coco {
    pub const NOSE: usize = 0;
    pub const LEFT_EYE: usize = 1;
    pub const RIGHT_EYE: usize = 2;
    pub const LEFT_EAR: usize = 3;
    pub const RIGHT_EAR: usize = 4;
    pub const LEFT_SHOULDER: usize = 5;
    pub const RIGHT_SHOULDER: usize = 6;
    pub const LEFT_ELBOW: usize = 7;
    pub const RIGHT_ELBOW: usize = 8;
    pub const LEFT_WRIST: usize = 9;
    pub const RIGHT_WRIST: usize = 10;
    pub const LEFT_HIP: usize = 11;
    pub const RIGHT_HIP: usize = 12;
    pub const LEFT_KNEE: usize = 13;
    pub const RIGHT_KNEE: usize = 14;
    pub const LEFT_ANKLE: usize = 15;
    pub const RIGHT_ANKLE: usize = 16;

    pub const LEFT_HAND_START: usize = 91;
    pub const RIGHT_HAND_START: usize = 112;

    /// Offsets into a single 21-point hand block.
    pub mod hand {
        pub const THUMB_CMC: usize = 1;
        pub const THUMB_MCP: usize = 2;
        pub const THUMB_IP: usize = 3;
        pub const THUMB_TIP: usize = 4;
        pub const INDEX_MCP: usize = 5;
        pub const INDEX_PIP: usize = 6;
        pub const INDEX_DIP: usize = 7;
        pub const INDEX_TIP: usize = 8;
        pub const MIDDLE_MCP: usize = 9;
        pub const MIDDLE_PIP: usize = 10;
        pub const MIDDLE_DIP: usize = 11;
        pub const MIDDLE_TIP: usize = 12;
        pub const RING_MCP: usize = 13;
        pub const RING_PIP: usize = 14;
        pub const RING_DIP: usize = 15;
        pub const RING_TIP: usize = 16;
        pub const LITTLE_MCP: usize = 17;
        pub const LITTLE_PIP: usize = 18;
        pub const LITTLE_DIP: usize = 19;
        pub const LITTLE_TIP: usize = 20;
    }
}

/// Convert a flat keypoint list into a [`SourceSkeleton`].
///
/// Positions are normalised to the camera's image plane:
/// * x ∈ `[-aspect, +aspect]` (aspect = image_w / image_h)
/// * y ∈ `[-1, +1]`, **y-up** (the +Y axis points toward the top of the image)
///
/// Body joints come straight from the COCO-WholeBody body subset. The Head
/// rotation is *not* derived from the nose position (that only measures
/// where the face is in the frame, not where it is pointed); instead the
/// nose + eyes + ears geometry is fed into [`face_pose_from_keypoints`] to
/// yield yaw/pitch/roll.
#[cfg(feature = "inference")]
fn map_to_source_skeleton(
    kpts: &[(f32, f32, f32)],
    input_w: u32,
    input_h: u32,
    aspect: f32,
    frame_index: u64,
) -> SourceSkeleton {
    let kpt = |i: usize| -> (f32, f32, f32) { kpts.get(i).copied().unwrap_or((0.0, 0.0, 0.0)) };
    let to_norm_x = |x: f32| ((x / input_w as f32) * 2.0 - 1.0) * aspect;
    let to_norm_y = |y: f32| 1.0 - (y / input_h as f32) * 2.0;
    let to_joint = |k: (f32, f32, f32)| SourceJoint {
        position: [to_norm_x(k.0), to_norm_y(k.1)],
        confidence: k.2,
    };

    let mut sk = SourceSkeleton::empty(frame_index);

    // Body keypoints → humanoid joint positions. The solver uses parent
    // positions (shoulder/hip) plus child positions (elbow/knee/etc.) to
    // reconstruct each bone direction; we populate the hierarchy entries
    // the solver expects to see.
    let min_body_conf = 0.0; // store everything, solver filters by threshold
    let shoulders_l = kpt(coco::LEFT_SHOULDER);
    let shoulders_r = kpt(coco::RIGHT_SHOULDER);
    let elbow_l = kpt(coco::LEFT_ELBOW);
    let elbow_r = kpt(coco::RIGHT_ELBOW);
    let wrist_l = kpt(coco::LEFT_WRIST);
    let wrist_r = kpt(coco::RIGHT_WRIST);
    let hip_l = kpt(coco::LEFT_HIP);
    let hip_r = kpt(coco::RIGHT_HIP);
    let knee_l = kpt(coco::LEFT_KNEE);
    let knee_r = kpt(coco::RIGHT_KNEE);
    let ankle_l = kpt(coco::LEFT_ANKLE);
    let ankle_r = kpt(coco::RIGHT_ANKLE);

    // Anatomical-left on the subject (COCO LEFT_*) appears on the LEFT
    // side of a non-mirrored image, which is the SAME side as the avatar's
    // *right* bones when the avatar faces the camera. We therefore cross
    // the keypoints — LEFT_SHOULDER drives RightShoulder, etc. — so that
    // the mirror-style VTuber interaction "lift my left arm → the arm on
    // my left in the screen lifts" falls out of matching signs on X
    // between source joints (image space) and avatar rest world.
    sk.put_joint(HumanoidBone::RightShoulder, to_joint(shoulders_l), min_body_conf);
    sk.put_joint(HumanoidBone::LeftShoulder, to_joint(shoulders_r), min_body_conf);
    sk.put_joint(HumanoidBone::RightUpperArm, to_joint(shoulders_l), min_body_conf);
    sk.put_joint(HumanoidBone::LeftUpperArm, to_joint(shoulders_r), min_body_conf);
    sk.put_joint(HumanoidBone::RightLowerArm, to_joint(elbow_l), min_body_conf);
    sk.put_joint(HumanoidBone::LeftLowerArm, to_joint(elbow_r), min_body_conf);
    sk.put_joint(HumanoidBone::RightHand, to_joint(wrist_l), min_body_conf);
    sk.put_joint(HumanoidBone::LeftHand, to_joint(wrist_r), min_body_conf);
    sk.put_joint(HumanoidBone::RightUpperLeg, to_joint(hip_l), min_body_conf);
    sk.put_joint(HumanoidBone::LeftUpperLeg, to_joint(hip_r), min_body_conf);
    sk.put_joint(HumanoidBone::RightLowerLeg, to_joint(knee_l), min_body_conf);
    sk.put_joint(HumanoidBone::LeftLowerLeg, to_joint(knee_r), min_body_conf);
    sk.put_joint(HumanoidBone::RightFoot, to_joint(ankle_l), min_body_conf);
    sk.put_joint(HumanoidBone::LeftFoot, to_joint(ankle_r), min_body_conf);

    // Hips: midpoint of the two hip keypoints. Spine's base.
    if hip_l.2 > 0.0 && hip_r.2 > 0.0 {
        let mid = SourceJoint {
            position: [
                (to_norm_x(hip_l.0) + to_norm_x(hip_r.0)) * 0.5,
                (to_norm_y(hip_l.1) + to_norm_y(hip_r.1)) * 0.5,
            ],
            confidence: hip_l.2.min(hip_r.2),
        };
        sk.put_joint(HumanoidBone::Hips, mid, min_body_conf);
    }

    // Finger keypoints (mirror-swapped to match the body: user's anatomical
    // LEFT hand drives the avatar's RightHand chain and vice versa). Each
    // finger bone is anchored at its PROXIMAL joint and its tip is the
    // keypoint of the next joint along the finger — the solver uses the
    // base-to-tip vector for rotation matching and the chain is walked
    // parent-to-child so rotations compose correctly.
    map_hand_keypoints(
        &mut sk,
        kpts,
        coco::LEFT_HAND_START,
        /* mirror target = */ Hand::Right,
        to_norm_x,
        to_norm_y,
        min_body_conf,
    );
    map_hand_keypoints(
        &mut sk,
        kpts,
        coco::RIGHT_HAND_START,
        Hand::Left,
        to_norm_x,
        to_norm_y,
        min_body_conf,
    );

    // Face pose from facial keypoints. These drive the Head bone; the nose
    // position itself is not used as a "rotation" input.
    let nose = kpt(coco::NOSE);
    let le = kpt(coco::LEFT_EYE);
    let re = kpt(coco::RIGHT_EYE);
    let lear = kpt(coco::LEFT_EAR);
    let rear = kpt(coco::RIGHT_EAR);
    sk.face = face_pose_from_keypoints(
        [to_norm_x(nose.0), to_norm_y(nose.1), nose.2],
        [to_norm_x(le.0), to_norm_y(le.1), le.2],
        [to_norm_x(re.0), to_norm_y(re.1), re.2],
        [to_norm_x(lear.0), to_norm_y(lear.1), lear.2],
        [to_norm_x(rear.0), to_norm_y(rear.1), rear.2],
    );

    // Per-joint confidence contributes to overall quality — approximate as
    // the max of nose and shoulders, the minimum needed to say "a person is
    // visible".
    sk.overall_confidence = nose.2.max(shoulders_l.2).max(shoulders_r.2);

    // CIGPose does not currently emit expression weights; leave the vector
    // empty so the expression solver holds previous values.
    sk.expressions = Vec::<SourceExpression>::new();

    sk
}

/// Derive yaw/pitch/roll from facial keypoints (nose, eyes, ears).
///
/// Sign conventions (Y-up, +X right, +Z toward camera; compatible with
/// `quat_from_euler_ypr`):
///
/// * **Yaw** — positive turns the face toward image +X. Computed as
///   `asin(nose_offset_from_ear_midline / half_face_width)`. The nose sits
///   forward of the skull, so it projects off the ear midline by roughly
///   `sin(yaw)` as the head rotates.
///
/// * **Pitch** — positive looks *down* (chin toward chest). This matches
///   `quat_from_euler_ypr`, where a positive X-axis rotation rolls the top
///   of the head toward +Z (toward the camera). At neutral the nose
///   projects about `PITCH_NEUTRAL_RATIO` below the eye midpoint; when the
///   user drops their chin, the nose moves further down in the image and
///   the ratio grows, giving a positive pitch.
///
/// * **Roll** — positive tilts the head toward image +X (right ear drops).
///   Angle of the ear-to-ear line in the image plane.
///
/// Returns `None` when the nose is not confident, or when neither the ears
/// nor the eyes are both confident.
#[cfg(feature = "inference")]
fn face_pose_from_keypoints(
    nose: [f32; 3],
    left_eye: [f32; 3],
    right_eye: [f32; 3],
    left_ear: [f32; 3],
    right_ear: [f32; 3],
) -> Option<FacePose> {
    let min_c = 0.2;
    let have_ears = left_ear[2] >= min_c && right_ear[2] >= min_c;
    let have_eyes = left_eye[2] >= min_c && right_eye[2] >= min_c;
    let have_nose = nose[2] >= min_c;
    if !have_nose || (!have_ears && !have_eyes) {
        return None;
    }

    // Roll + face width. Prefer ears (wider baseline, more stable) and
    // fall back to eyes with a rough 2× scale factor to compensate for the
    // shorter eye-to-eye distance.
    let (roll, face_width) = if have_ears {
        let dx = right_ear[0] - left_ear[0];
        let dy = right_ear[1] - left_ear[1];
        (dy.atan2(dx.abs().max(1e-4)), dx.abs().max(1e-4))
    } else {
        let dx = right_eye[0] - left_eye[0];
        let dy = right_eye[1] - left_eye[1];
        (dy.atan2(dx.abs().max(1e-4)), (dx.abs().max(1e-4)) * 2.0)
    };

    // Yaw: nose horizontal offset from the ear/eye midpoint normalised to
    // half the face width, then through asin.
    let head_center_x = if have_ears {
        (left_ear[0] + right_ear[0]) * 0.5
    } else {
        (left_eye[0] + right_eye[0]) * 0.5
    };
    let half_face = (face_width * 0.5).max(1e-4);
    let yaw_ratio = ((nose[0] - head_center_x) / half_face).clamp(-1.0, 1.0);
    let yaw = yaw_ratio.asin().clamp(-1.05, 1.05);

    // Pitch: in image coords (y-up), neutral-posed nose sits below the eye
    // midpoint by a characteristic fraction of face width. Looking up
    // reduces that gap, looking down increases it.
    //
    // `PITCH_NEUTRAL_RATIO` is the rough average ratio
    // `(eye_y - nose_y) / face_width` for a front-facing neutral face.
    // `PITCH_GAIN` maps the normalised deviation to an angle in radians and
    // was picked empirically — a ±0.2 deviation (roughly ±30° of real
    // pitch) maps to about ±0.5 rad ≈ ±29°.
    const PITCH_NEUTRAL_RATIO: f32 = 0.35;
    const PITCH_GAIN: f32 = 2.5;
    let pitch = if have_eyes {
        let eye_mid_y = (left_eye[1] + right_eye[1]) * 0.5;
        let nose_below_eyes = eye_mid_y - nose[1];
        let ratio = nose_below_eyes / face_width;
        // delta > 0 when the nose has dropped relative to neutral (looking
        // down); feeds straight into a +X rotation that tilts the avatar's
        // head forward.
        let delta = ratio - PITCH_NEUTRAL_RATIO;
        (delta * PITCH_GAIN).clamp(-1.1, 1.1)
    } else {
        0.0
    };

    let confidence = {
        let mut c = nose[2];
        if have_eyes {
            c = c.min(left_eye[2]).min(right_eye[2]);
        }
        if have_ears {
            c = c.min(left_ear[2]).min(right_ear[2]);
        }
        c
    };

    Some(FacePose {
        yaw,
        pitch,
        roll,
        confidence,
    })
}

/// Which avatar hand a block of keypoints should drive.
#[cfg(feature = "inference")]
#[derive(Clone, Copy)]
enum Hand {
    Left,
    Right,
}

/// Map a MediaPipe-layout 21-point hand block (starting at `block_start` in
/// the flat keypoint list) to the avatar's finger chain. Each finger bone
/// in the VRM humanoid is placed at its proximal joint and its direction is
/// set from the next joint down the finger:
///
/// | Bone           | Position kpt  | Direction tip kpt |
/// |----------------|---------------|-------------------|
/// | ThumbProximal  | THUMB_CMC (1) | THUMB_MCP (2)     |
/// | ThumbIntermed. | THUMB_MCP (2) | THUMB_IP  (3)     |
/// | ThumbDistal    | THUMB_IP  (3) | THUMB_TIP (4)     |
/// | IndexProximal  | INDEX_MCP (5) | INDEX_PIP (6)     |
/// | IndexIntermed. | INDEX_PIP (6) | INDEX_DIP (7)     |
/// | IndexDistal    | INDEX_DIP (7) | INDEX_TIP (8)     |
/// | ...same for MIDDLE / RING / LITTLE...             |
///
/// The solver then walks these chains parent-first and aligns each bone's
/// rest world direction to its observed 2D direction, so individual finger
/// bends (making a fist vs. a peace sign vs. an open hand) reflect on the
/// avatar.
#[cfg(feature = "inference")]
fn map_hand_keypoints(
    sk: &mut SourceSkeleton,
    kpts: &[(f32, f32, f32)],
    block_start: usize,
    target: Hand,
    to_norm_x: impl Fn(f32) -> f32,
    to_norm_y: impl Fn(f32) -> f32,
    min_conf: f32,
) {
    let at = |off: usize| -> SourceJoint {
        let k = kpts
            .get(block_start + off)
            .copied()
            .unwrap_or((0.0, 0.0, 0.0));
        SourceJoint {
            position: [to_norm_x(k.0), to_norm_y(k.1)],
            confidence: k.2,
        }
    };

    use coco::hand as h;
    use HumanoidBone as B;

    // (finger_joint_offset, bone_for_left_hand, bone_for_right_hand)
    // Each row describes one "proximal joint" that serves as the position
    // anchor for a bone. Tip of the bone is the next row in the same
    // finger's chain (see solver DRIVEN_BONES).
    let entries = [
        (h::THUMB_CMC, B::LeftThumbProximal, B::RightThumbProximal),
        (
            h::THUMB_MCP,
            B::LeftThumbIntermediate,
            B::RightThumbIntermediate,
        ),
        (h::THUMB_IP, B::LeftThumbDistal, B::RightThumbDistal),
        (h::INDEX_MCP, B::LeftIndexProximal, B::RightIndexProximal),
        (
            h::INDEX_PIP,
            B::LeftIndexIntermediate,
            B::RightIndexIntermediate,
        ),
        (h::INDEX_DIP, B::LeftIndexDistal, B::RightIndexDistal),
        (h::MIDDLE_MCP, B::LeftMiddleProximal, B::RightMiddleProximal),
        (
            h::MIDDLE_PIP,
            B::LeftMiddleIntermediate,
            B::RightMiddleIntermediate,
        ),
        (h::MIDDLE_DIP, B::LeftMiddleDistal, B::RightMiddleDistal),
        (h::RING_MCP, B::LeftRingProximal, B::RightRingProximal),
        (
            h::RING_PIP,
            B::LeftRingIntermediate,
            B::RightRingIntermediate,
        ),
        (h::RING_DIP, B::LeftRingDistal, B::RightRingDistal),
        (h::LITTLE_MCP, B::LeftLittleProximal, B::RightLittleProximal),
        (
            h::LITTLE_PIP,
            B::LeftLittleIntermediate,
            B::RightLittleIntermediate,
        ),
        (h::LITTLE_DIP, B::LeftLittleDistal, B::RightLittleDistal),
    ];

    // A virtual "tip anchor" for each distal bone — the fingertip keypoint.
    // These entries are NOT separate humanoid bones; they're anchored by
    // the solver via the tip definitions in DRIVEN_BONES (where *Distal's
    // tip references the fingertip offset via Tip::HandTip). But since the
    // solver resolves tips through `get_joint_2d(HumanoidBone)`, we insert
    // the fingertip positions under pseudo-bones that the solver knows to
    // look up: we reuse the "tip joint" anchoring by ALSO writing the TIP
    // position into the *Distal bone's own position slot (+1). The actual
    // fingertip direction will come from (Distal's position, next kpt)
    // tips wired explicitly in DRIVEN_BONES.
    for (offset, left_bone, right_bone) in entries {
        let joint = at(offset);
        let bone = match target {
            Hand::Left => left_bone,
            Hand::Right => right_bone,
        };
        sk.put_joint(bone, joint, min_conf);
    }

    // Fingertip positions are needed as "tips" for the distal bones. The
    // pose solver consumes them via pseudo-bones stored under the Distal
    // slots *plus one offset*, but our HumanoidBone enum has no extra
    // variants for tips, so we plant the fingertip positions into a
    // dedicated lookup table on the skeleton side (the solver falls back
    // to the distal bone's own position when a fingertip is missing; see
    // DRIVEN_BONES). Practically: we also publish the fingertip position
    // as the Distal bone's *tip* by storing it on the map under the
    // HumanoidBone variant one step beyond Distal — which does not exist,
    // so we use HumanoidBone::Left/RightHand as the tip for the pinky's
    // distal alternative when appropriate. For the common case (rock /
    // scissors / paper), a proximal/intermediate bend is enough signal.
    //
    // Rather than invent pseudo-bones, the solver wires each *Distal bone
    // directly to the fingertip source position via a dedicated code path
    // (see Tip::Fingertip in pose_solver). We expose the fingertip via a
    // parallel map on SourceSkeleton.
    let tip_entries = [
        (h::THUMB_TIP, B::LeftThumbDistal, B::RightThumbDistal),
        (h::INDEX_TIP, B::LeftIndexDistal, B::RightIndexDistal),
        (h::MIDDLE_TIP, B::LeftMiddleDistal, B::RightMiddleDistal),
        (h::RING_TIP, B::LeftRingDistal, B::RightRingDistal),
        (h::LITTLE_TIP, B::LeftLittleDistal, B::RightLittleDistal),
    ];
    for (offset, left_distal, right_distal) in tip_entries {
        let joint = at(offset);
        let bone = match target {
            Hand::Left => left_distal,
            Hand::Right => right_distal,
        };
        sk.fingertips.insert(bone, joint);
    }
}
