#![allow(dead_code)]
//! Simplified pose estimation from webcam frames.
//!
//! This module provides a placeholder face/pose detector based on skin-color
//! filtering in HSV space.  It is intentionally simple -- the goal is to
//! produce plausible `TrackingRigPose` data from a real camera feed until a
//! proper ML inference backend (e.g. MediaPipe) is integrated.
//!
//! Algorithm outline:
//! 1. Convert RGB pixels to HSV.
//! 2. Threshold for skin-tone hue/saturation/value ranges.
//! 3. Find the bounding box of the largest skin-colored region.
//! 4. Estimate head position from the bounding-box center.
//! 5. Estimate head yaw from the horizontal offset of the face center.
//! 6. Estimate head pitch from the vertical offset.
//! 7. Derive a rough confidence from the proportion of skin pixels.

use super::{
    DetectionAnnotation, ExpressionWeight, ExpressionWeightSet, PoseEstimate, RigArmTargets,
    RigShoulderTargets, RigTarget, TrackingConfidenceMap, TrackingRigPose,
};

/// Produce a `PoseEstimate` from a raw RGB frame.
///
/// `rgb_data` must contain `width * height * 3` bytes in row-major RGB order.
pub fn estimate_pose(rgb_data: &[u8], width: u32, height: u32, frame_index: u64) -> PoseEstimate {
    let expected_len = (width * height * 3) as usize;
    if rgb_data.len() < expected_len {
        return PoseEstimate {
            rig_pose: TrackingRigPose {
                source_timestamp: frame_index,
                ..Default::default()
            },
            annotation: DetectionAnnotation::default(),
        };
    }

    // -----------------------------------------------------------------------
    // 1. Skin-color detection via HSV thresholding
    // -----------------------------------------------------------------------

    let mut skin_min_x: u32 = width;
    let mut skin_max_x: u32 = 0;
    let mut skin_min_y: u32 = height;
    let mut skin_max_y: u32 = 0;
    let mut skin_pixel_count: u32 = 0;
    let mut skin_sum_x: u64 = 0;
    let mut skin_sum_y: u64 = 0;

    for y in 0..height {
        for x in 0..width {
            let base = ((y * width + x) * 3) as usize;
            let r = rgb_data[base] as f32 / 255.0;
            let g = rgb_data[base + 1] as f32 / 255.0;
            let b = rgb_data[base + 2] as f32 / 255.0;

            let (h, s, v) = rgb_to_hsv(r, g, b);

            // Skin-tone HSV range (fairly permissive to handle varied lighting)
            // Hue: roughly 0-50 degrees (reds/oranges/yellows)
            // Saturation: 0.15 - 0.75
            // Value: 0.2 - 0.95
            let is_skin = (h <= 50.0 || h >= 340.0) && s >= 0.15 && s <= 0.75 && v >= 0.2;

            if is_skin {
                skin_pixel_count += 1;
                skin_sum_x += x as u64;
                skin_sum_y += y as u64;
                if x < skin_min_x {
                    skin_min_x = x;
                }
                if x > skin_max_x {
                    skin_max_x = x;
                }
                if y < skin_min_y {
                    skin_min_y = y;
                }
                if y > skin_max_y {
                    skin_max_y = y;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // 2. Detection quality / confidence
    // -----------------------------------------------------------------------

    let total_pixels = width * height;
    let skin_ratio = skin_pixel_count as f32 / total_pixels as f32;

    // We expect roughly 5-30% of the frame to be skin if a face/upper-body
    // is visible.  Below 2% we treat as "no detection".
    if skin_ratio < 0.02 || skin_pixel_count == 0 {
        return PoseEstimate {
            rig_pose: TrackingRigPose {
                source_timestamp: frame_index,
                confidence: TrackingConfidenceMap {
                    head_confidence: 0.0,
                    torso_confidence: 0.0,
                    left_arm_confidence: 0.0,
                    right_arm_confidence: 0.0,
                    face_confidence: 0.0,
                    left_leg_confidence: 0.0,
                    right_leg_confidence: 0.0,
                    left_hand_confidence: 0.0,
                    right_hand_confidence: 0.0,
                },
                ..Default::default()
            },
            annotation: DetectionAnnotation::default(),
        };
    }

    let confidence = (skin_ratio / 0.25).min(1.0).max(0.1);

    // -----------------------------------------------------------------------
    // 3. Face centroid in normalized image coordinates [-1, 1]
    // -----------------------------------------------------------------------

    let centroid_x = skin_sum_x as f32 / skin_pixel_count as f32;
    let centroid_y = skin_sum_y as f32 / skin_pixel_count as f32;

    // Normalize to [-1, 1] where (0,0) is image center.
    let norm_x = (centroid_x / width as f32) * 2.0 - 1.0;
    let norm_y = (centroid_y / height as f32) * 2.0 - 1.0;

    // -----------------------------------------------------------------------
    // 4. Head position from face centroid
    // -----------------------------------------------------------------------

    // Map normalized image position to a performer-space head position.
    // X: horizontal offset, Y: height (with 1.6 as neutral head height),
    // Z: fixed (single-camera, no depth).
    let head_x = norm_x * 0.3; // scale down lateral motion
    let head_y = 1.6 - norm_y * 0.2; // higher in frame = higher position
    let head_z = 0.0;

    // -----------------------------------------------------------------------
    // 5. Head orientation from face region aspect ratio and position
    // -----------------------------------------------------------------------

    let face_width = (skin_max_x as f32 - skin_min_x as f32).max(1.0);
    let face_height = (skin_max_y as f32 - skin_min_y as f32).max(1.0);
    let aspect = face_width / face_height;

    // Yaw: deviation from center -> head rotation
    let head_yaw = -norm_x * 0.4; // negative because looking right in camera = positive world yaw

    // Pitch: deviation from vertical center
    let head_pitch = norm_y * 0.3;

    // Roll: from aspect ratio asymmetry (very rough)
    // When aspect > 1 face appears wider -- could be slight tilt.
    let head_roll = (aspect - 0.75).clamp(-0.3, 0.3) * 0.2;

    // Convert small-angle Euler (pitch, yaw, roll) to quaternion (ZYX order).
    let head_orientation = euler_to_quat(head_pitch, head_yaw, head_roll);

    // -----------------------------------------------------------------------
    // 6. Build the pose
    // -----------------------------------------------------------------------

    let w = width as f32;
    let h = height as f32;
    let annotation = DetectionAnnotation {
        keypoints: vec![(centroid_x / w, centroid_y / h, confidence)],
        skeleton: vec![],
        bounding_box: Some((
            skin_min_x as f32 / w,
            skin_min_y as f32 / h,
            skin_max_x as f32 / w,
            skin_max_y as f32 / h,
        )),
    };

    PoseEstimate {
        rig_pose: TrackingRigPose {
            source_timestamp: frame_index,
            head: Some(RigTarget {
                position: [head_x, head_y, head_z],
                orientation: head_orientation,
                confidence,
            }),
            neck: Some(RigTarget {
                position: [head_x * 0.5, head_y - 0.1, 0.0],
                orientation: [0.0, 0.0, 0.0, 1.0],
                confidence: confidence * 0.8,
            }),
            spine: Some(RigTarget {
                position: [0.0, 1.0, 0.0],
                orientation: [0.0, 0.0, 0.0, 1.0],
                confidence: confidence * 0.5,
            }),
            shoulders: RigShoulderTargets {
                left: Some(RigTarget {
                    position: [-0.15, head_y - 0.2, 0.0],
                    orientation: [0.0, 0.0, 0.0, 1.0],
                    confidence: confidence * 0.4,
                }),
                right: Some(RigTarget {
                    position: [0.15, head_y - 0.2, 0.0],
                    orientation: [0.0, 0.0, 0.0, 1.0],
                    confidence: confidence * 0.4,
                }),
            },
            arms: RigArmTargets {
                left_upper: None,
                left_lower: None,
                right_upper: None,
                right_lower: None,
            },
            hands: None,
            legs: None,
            feet: None,
            fingers: None,
            expressions: ExpressionWeightSet {
                weights: vec![ExpressionWeight {
                    name: "blink".to_string(),
                    weight: 0.0,
                }],
            },
            confidence: TrackingConfidenceMap {
                head_confidence: confidence,
                torso_confidence: confidence * 0.5,
                left_arm_confidence: 0.0,
                right_arm_confidence: 0.0,
                face_confidence: confidence * 0.7,
                left_leg_confidence: 0.0,
                right_leg_confidence: 0.0,
                left_hand_confidence: 0.0,
                right_hand_confidence: 0.0,
            },
        },
        annotation,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert an RGB color (each component in [0, 1]) to HSV.
/// Returns (hue in [0, 360], saturation in [0, 1], value in [0, 1]).
fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let value = max;

    if delta < 1e-6 {
        return (0.0, 0.0, value);
    }

    let saturation = delta / max;

    let hue = if (r - max).abs() < 1e-6 {
        60.0 * (((g - b) / delta) % 6.0)
    } else if (g - max).abs() < 1e-6 {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };

    let hue = if hue < 0.0 { hue + 360.0 } else { hue };

    (hue, saturation, value)
}

/// Convert Euler angles (pitch, yaw, roll) in radians to a unit quaternion
/// using ZYX rotation order.
pub(super) fn euler_to_quat(pitch: f32, yaw: f32, roll: f32) -> [f32; 4] {
    let (sp, cp) = (pitch * 0.5).sin_cos();
    let (sy, cy) = (yaw * 0.5).sin_cos();
    let (sr, cr) = (roll * 0.5).sin_cos();

    [
        sr * cp * cy - cr * sp * sy, // x
        cr * sp * cy + sr * cp * sy, // y
        cr * cp * sy - sr * sp * cy, // z
        cr * cp * cy + sr * sp * sy, // w
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_to_hsv_red() {
        let (h, s, v) = rgb_to_hsv(1.0, 0.0, 0.0);
        assert!((h - 0.0).abs() < 1.0);
        assert!((s - 1.0).abs() < 0.01);
        assert!((v - 1.0).abs() < 0.01);
    }

    #[test]
    fn rgb_to_hsv_white() {
        let (_, s, v) = rgb_to_hsv(1.0, 1.0, 1.0);
        assert!(s < 0.01);
        assert!((v - 1.0).abs() < 0.01);
    }

    #[test]
    fn estimate_pose_empty_frame_returns_default() {
        let result = estimate_pose(&[], 640, 480, 0);
        assert!(result.rig_pose.head.is_none());
    }

    #[test]
    fn estimate_pose_uniform_skin_color() {
        let w: u32 = 32;
        let h: u32 = 32;
        let mut data = vec![0u8; (w * h * 3) as usize];
        for i in 0..(w * h) as usize {
            data[i * 3] = 200;
            data[i * 3 + 1] = 150;
            data[i * 3 + 2] = 120;
        }
        let result = estimate_pose(&data, w, h, 42);
        assert!(result.rig_pose.head.is_some());
        assert!(result.rig_pose.confidence.head_confidence > 0.0);
        assert_eq!(result.rig_pose.source_timestamp, 42);
        assert!(result.annotation.bounding_box.is_some());
    }

    #[test]
    fn euler_to_quat_identity() {
        let q = euler_to_quat(0.0, 0.0, 0.0);
        assert!((q[0]).abs() < 1e-6);
        assert!((q[1]).abs() < 1e-6);
        assert!((q[2]).abs() < 1e-6);
        assert!((q[3] - 1.0).abs() < 1e-6);
    }
}
