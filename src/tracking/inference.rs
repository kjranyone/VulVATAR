#![allow(dead_code)]
use super::PoseEstimate;
#[cfg(feature = "inference")]
use super::{
    pose_estimation::euler_to_quat, DetectionAnnotation, ExpressionWeightSet, RigShoulderTargets,
    RigTarget, TrackingConfidenceMap, TrackingRigPose,
};
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
                    rig_pose: TrackingRigPose {
                        source_timestamp: frame_index,
                        ..Default::default()
                    },
                    annotation: DetectionAnnotation::default(),
                };
            }
        };

        let outputs = match self.session.run(ort::inputs![input]) {
            Ok(out) => out,
            Err(e) => {
                error!("ONNX inference failed: {}", e);
                return PoseEstimate {
                    rig_pose: TrackingRigPose {
                        source_timestamp: frame_index,
                        ..Default::default()
                    },
                    annotation: DetectionAnnotation::default(),
                };
            }
        };

        let (shape_x, data_x) = match outputs["simcc_x"].try_extract_tensor::<f32>() {
            Ok(v) => v,
            Err(e) => {
                error!("ONNX simcc_x extraction failed: {}", e);
                return PoseEstimate {
                    rig_pose: TrackingRigPose {
                        source_timestamp: frame_index,
                        ..Default::default()
                    },
                    annotation: DetectionAnnotation::default(),
                };
            }
        };
        let (_shape_y, data_y) = match outputs["simcc_y"].try_extract_tensor::<f32>() {
            Ok(v) => v,
            Err(e) => {
                error!("ONNX simcc_y extraction failed: {}", e);
                return PoseEstimate {
                    rig_pose: TrackingRigPose {
                        source_timestamp: frame_index,
                        ..Default::default()
                    },
                    annotation: DetectionAnnotation::default(),
                };
            }
        };

        let split_ratio = 2.0;
        let kpts = decode_simcc_from_slice(data_x, data_y, shape_x, split_ratio);

        let rig_pose = map_to_rig_pose(
            kpts.clone(),
            self.input_w,
            self.input_h,
            aspect,
            frame_index,
        );

        let annotation = build_annotation(kpts, self.input_w, self.input_h, width, height);

        PoseEstimate {
            rig_pose,
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
    // Basic scale/crop to fit the input maintaining aspect ratio constraints
    // Actually, simple nearest neighbor resize:
    let mut tensor = Array4::<f32>::zeros((1, 3, dst_h as usize, dst_w as usize));

    // Quick nearest-neighbor scaling
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

/// COCO-wholebody skeleton connections (body part only).
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
    orig_w: u32,
    orig_h: u32,
) -> DetectionAnnotation {
    let mw = model_w as f32;
    let mh = model_h as f32;
    let _ow = orig_w as f32;
    let _oh = orig_h as f32;
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

#[cfg(feature = "inference")]
fn map_to_rig_pose(
    kpts: Vec<(f32, f32, f32)>,
    input_w: u32,
    input_h: u32,
    aspect: f32,
    frame_index: u64,
) -> TrackingRigPose {
    use super::{RigArmTargets, RigFeetTargets, RigFingerTargets, RigHandTargets, RigLegTargets};

    let to_norm_x = |x: f32| (x / input_w as f32) * 2.0 - 1.0;
    let to_norm_y = |y: f32| 1.0 - (y / input_h as f32) * 2.0;

    let kpt = |i: usize| -> (f32, f32, f32) { kpts.get(i).copied().unwrap_or((0.0, 0.0, 0.0)) };

    let min_conf = 0.1f32;

    // Compute a bone orientation from the 2D vector (parent → child).
    // The angle of the vector in the image plane is mapped to a Z-axis
    // rotation (roll) so that the avatar's limbs visually follow the
    // performer's limb direction.  For upper arms the rest pose points
    // roughly downward, so we subtract π/2 (left) or add π/2 (right)
    // as the rest-pose offset.
    let orientation_from_limb =
        |parent: (f32, f32, f32), child: (f32, f32, f32), rest_angle: f32| -> [f32; 4] {
            let dx = child.0 - parent.0;
            let dy = child.1 - parent.1;
            let angle = dy.atan2(dx); // image-space angle (right=0, down=π/2)
            let roll = angle - rest_angle;
            euler_to_quat(0.0, 0.0, roll)
        };

    let make_target_with_orient =
        |k: (f32, f32, f32), orient: [f32; 4]| -> RigTarget {
            RigTarget {
                position: [to_norm_x(k.0) * aspect, to_norm_y(k.1) + 1.6, 0.0],
                orientation: orient,
                confidence: k.2,
            }
        };

    let make_target = |k: (f32, f32, f32)| -> RigTarget {
        RigTarget {
            position: [to_norm_x(k.0) * aspect, to_norm_y(k.1) + 1.6, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            confidence: k.2,
        }
    };

    let maybe_target = |k: (f32, f32, f32)| -> Option<RigTarget> {
        if k.2 < min_conf {
            None
        } else {
            Some(make_target(k))
        }
    };

    let nose = kpt(0);
    let left_shoulder = kpt(5);
    let right_shoulder = kpt(6);
    let left_elbow = kpt(7);
    let right_elbow = kpt(8);
    let left_wrist = kpt(9);
    let right_wrist = kpt(10);
    let left_hip = kpt(11);
    let right_hip = kpt(12);
    let left_knee = kpt(13);
    let right_knee = kpt(14);
    let left_ankle = kpt(15);
    let right_ankle = kpt(16);

    // --- Head: yaw from nose horizontal position, pitch from vertical ---
    let head_yaw = to_norm_x(nose.0) * -0.5;
    let head_pitch = to_norm_y(nose.1) * -0.15;
    let head_orientation = euler_to_quat(head_pitch, head_yaw, 0.0);

    // --- Spine: torso lean from shoulder midpoint ---
    let spine_orientation = if left_shoulder.2 >= min_conf && right_shoulder.2 >= min_conf {
        let mid_x = (left_shoulder.0 + right_shoulder.0) * 0.5;
        let mid_y = (left_shoulder.1 + right_shoulder.1) * 0.5;
        let hip_mid_x = (left_hip.0 + right_hip.0) * 0.5;
        let hip_mid_y = (left_hip.1 + right_hip.1) * 0.5;
        let dx = mid_x - hip_mid_x;
        let dy = mid_y - hip_mid_y;
        // Shoulder roll from shoulder-line angle
        let shoulder_dx = right_shoulder.0 - left_shoulder.0;
        let shoulder_dy = right_shoulder.1 - left_shoulder.1;
        let roll = shoulder_dy.atan2(shoulder_dx) * 0.5;
        // Forward lean from torso vertical deviation
        let lean = dx.atan2(dy.abs().max(1.0)) * 0.3;
        euler_to_quat(lean, 0.0, roll)
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };

    // --- Arms: compute orientations from limb direction vectors ---
    // Rest pose: upper arms point downward in image space.
    // Image atan2 convention: right=0, down=+π/2
    // Left upper arm rest angle ≈ π/2 (pointing down)
    // Right upper arm rest angle ≈ π/2 (pointing down)
    use std::f32::consts::FRAC_PI_2;

    let left_upper_orient = if left_shoulder.2 >= min_conf && left_elbow.2 >= min_conf {
        orientation_from_limb(left_shoulder, left_elbow, FRAC_PI_2)
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };
    let left_lower_orient = if left_elbow.2 >= min_conf && left_wrist.2 >= min_conf {
        orientation_from_limb(left_elbow, left_wrist, FRAC_PI_2)
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };
    let right_upper_orient = if right_shoulder.2 >= min_conf && right_elbow.2 >= min_conf {
        orientation_from_limb(right_shoulder, right_elbow, FRAC_PI_2)
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };
    let right_lower_orient = if right_elbow.2 >= min_conf && right_wrist.2 >= min_conf {
        orientation_from_limb(right_elbow, right_wrist, FRAC_PI_2)
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };

    let arms = RigArmTargets {
        left_upper: if left_shoulder.2 >= min_conf {
            Some(make_target_with_orient(left_shoulder, left_upper_orient))
        } else {
            None
        },
        left_lower: if left_elbow.2 >= min_conf {
            Some(make_target_with_orient(left_elbow, left_lower_orient))
        } else {
            None
        },
        right_upper: if right_shoulder.2 >= min_conf {
            Some(make_target_with_orient(right_shoulder, right_upper_orient))
        } else {
            None
        },
        right_lower: if right_elbow.2 >= min_conf {
            Some(make_target_with_orient(right_elbow, right_lower_orient))
        } else {
            None
        },
    };

    let hands = if left_wrist.2 >= min_conf || right_wrist.2 >= min_conf {
        Some(RigHandTargets {
            left: maybe_target(left_wrist),
            right: maybe_target(right_wrist),
        })
    } else {
        None
    };

    // --- Legs: compute orientations from limb direction vectors ---
    let left_upper_leg_orient = if left_hip.2 >= min_conf && left_knee.2 >= min_conf {
        orientation_from_limb(left_hip, left_knee, FRAC_PI_2)
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };
    let left_lower_leg_orient = if left_knee.2 >= min_conf && left_ankle.2 >= min_conf {
        orientation_from_limb(left_knee, left_ankle, FRAC_PI_2)
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };
    let right_upper_leg_orient = if right_hip.2 >= min_conf && right_knee.2 >= min_conf {
        orientation_from_limb(right_hip, right_knee, FRAC_PI_2)
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };
    let right_lower_leg_orient = if right_knee.2 >= min_conf && right_ankle.2 >= min_conf {
        orientation_from_limb(right_knee, right_ankle, FRAC_PI_2)
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };

    let legs = if left_hip.2 >= min_conf || right_hip.2 >= min_conf {
        Some(RigLegTargets {
            left_upper: if left_hip.2 >= min_conf {
                Some(make_target_with_orient(left_hip, left_upper_leg_orient))
            } else {
                None
            },
            left_lower: if left_knee.2 >= min_conf {
                Some(make_target_with_orient(left_knee, left_lower_leg_orient))
            } else {
                None
            },
            right_upper: if right_hip.2 >= min_conf {
                Some(make_target_with_orient(right_hip, right_upper_leg_orient))
            } else {
                None
            },
            right_lower: if right_knee.2 >= min_conf {
                Some(make_target_with_orient(right_knee, right_lower_leg_orient))
            } else {
                None
            },
        })
    } else {
        None
    };

    let feet = if left_ankle.2 >= min_conf || right_ankle.2 >= min_conf {
        Some(RigFeetTargets {
            left: maybe_target(left_ankle),
            right: maybe_target(right_ankle),
        })
    } else {
        None
    };

    let left_thumb_tip = kpt(94);
    let left_index_tip = kpt(98);
    let left_middle_tip = kpt(102);
    let left_ring_tip = kpt(106);
    let left_pinky_tip = kpt(110);
    let right_thumb_tip = kpt(115);
    let right_index_tip = kpt(119);
    let right_middle_tip = kpt(123);
    let right_ring_tip = kpt(127);
    let right_pinky_tip = kpt(131);

    let finger_conf = [
        left_thumb_tip.2,
        left_index_tip.2,
        left_middle_tip.2,
        right_thumb_tip.2,
        right_index_tip.2,
        right_middle_tip.2,
    ]
    .into_iter()
    .fold(0.0f32, f32::max);
    let fingers = if finger_conf >= min_conf {
        Some(RigFingerTargets {
            left_thumb: maybe_target(left_thumb_tip),
            left_index: maybe_target(left_index_tip),
            left_middle: maybe_target(left_middle_tip),
            left_ring: maybe_target(left_ring_tip),
            left_pinky: maybe_target(left_pinky_tip),
            right_thumb: maybe_target(right_thumb_tip),
            right_index: maybe_target(right_index_tip),
            right_middle: maybe_target(right_middle_tip),
            right_ring: maybe_target(right_ring_tip),
            right_pinky: maybe_target(right_pinky_tip),
        })
    } else {
        None
    };

    let spine_conf = if left_shoulder.2 >= min_conf && right_shoulder.2 >= min_conf {
        (left_shoulder.2 + right_shoulder.2) * 0.5
    } else {
        0.0
    };

    TrackingRigPose {
        source_timestamp: frame_index,
        head: Some(RigTarget {
            position: [to_norm_x(nose.0) * aspect, to_norm_y(nose.1) + 1.6, 0.0],
            orientation: head_orientation,
            confidence: nose.2,
        }),
        neck: maybe_target(kpt(1)),
        spine: if spine_conf >= min_conf {
            Some(RigTarget {
                position: [0.0, 1.0, 0.0],
                orientation: spine_orientation,
                confidence: spine_conf,
            })
        } else {
            None
        },
        shoulders: RigShoulderTargets {
            left: maybe_target(left_shoulder),
            right: maybe_target(right_shoulder),
        },
        arms,
        hands,
        legs,
        feet,
        fingers,
        expressions: ExpressionWeightSet { weights: vec![] },
        confidence: TrackingConfidenceMap {
            head_confidence: nose.2,
            torso_confidence: (left_shoulder.2 + right_shoulder.2) / 2.0,
            left_arm_confidence: left_elbow.2,
            right_arm_confidence: right_elbow.2,
            face_confidence: nose.2,
            left_leg_confidence: left_knee.2,
            right_leg_confidence: right_knee.2,
            left_hand_confidence: finger_conf.max(left_wrist.2),
            right_hand_confidence: finger_conf.max(right_wrist.2),
        },
    }
}
