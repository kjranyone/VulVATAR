#![allow(dead_code)]
use crate::asset::{
    ExpressionAssetSet, HumanoidBone, HumanoidMap, Quat, SkeletonAsset, Transform, Vec3,
};
use crate::math_utils::quat_normalize;
use crate::tracking::{RigTarget, TrackingRigPose, TrackingSmoothingParams};

/// Default confidence threshold below which a tracking channel is ignored.
const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.5;

/// Quaternion linear interpolation (unnormalized). Used for simple smoothing.
fn quat_lerp(a: &Quat, b: &Quat, t: f32) -> Quat {
    // Ensure we interpolate along the shortest arc by checking the dot product.
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let sign = if dot < 0.0 { -1.0 } else { 1.0 };
    let inv_t = 1.0 - t;
    let out = [
        inv_t * a[0] + t * sign * b[0],
        inv_t * a[1] + t * sign * b[1],
        inv_t * a[2] + t * sign * b[2],
        inv_t * a[3] + t * sign * b[3],
    ];
    quat_normalize(&out)
}

/// Apply a single tracking target's orientation to the bone at the given node index.
///
/// If `smoothing_factor` is > 0, the rotation is blended with the bone's current (rest)
/// rotation to reduce jitter. A factor of 0.0 means no smoothing (instant snap),
/// while 1.0 would keep the current value entirely.
fn apply_target_rotation(
    target: &RigTarget,
    node_idx: usize,
    confidence_threshold: f32,
    smoothing_factor: f32,
    local_transforms: &mut [Transform],
) {
    // Skip low-confidence data to avoid noisy poses.
    if target.confidence < confidence_threshold {
        return;
    }
    if node_idx >= local_transforms.len() {
        return;
    }

    if smoothing_factor > 0.0 {
        // Blend between the current local rotation and the tracking orientation.
        // The blend parameter `t` is (1 - smoothing_factor) so that higher smoothing
        // means slower convergence toward the target.
        let t = 1.0 - smoothing_factor;
        local_transforms[node_idx].rotation =
            quat_lerp(&local_transforms[node_idx].rotation, &target.orientation, t);
    } else {
        local_transforms[node_idx].rotation = target.orientation;
    }
}

/// Vec3 linear interpolation.
fn vec3_lerp(a: &Vec3, b: &Vec3, t: f32) -> Vec3 {
    let inv_t = 1.0 - t;
    [
        inv_t * a[0] + t * b[0],
        inv_t * a[1] + t * b[1],
        inv_t * a[2] + t * b[2],
    ]
}

/// Apply a single tracking target's position to the bone at the given node index.
///
/// If `smoothing_factor` is > 0, the position is blended with the bone's current
/// translation to reduce jitter. A factor of 0.0 means no smoothing (instant snap),
/// while 1.0 would keep the current value entirely.
fn apply_target_position(
    target: &RigTarget,
    node_idx: usize,
    confidence_threshold: f32,
    smoothing_factor: f32,
    local_transforms: &mut [Transform],
) {
    if target.confidence < confidence_threshold {
        return;
    }
    if node_idx >= local_transforms.len() {
        return;
    }

    if smoothing_factor > 0.0 {
        let t = 1.0 - smoothing_factor;
        local_transforms[node_idx].translation =
            vec3_lerp(&local_transforms[node_idx].translation, &target.position, t);
    } else {
        local_transforms[node_idx].translation = target.position;
    }
}

/// Look up a humanoid bone in the bone map and return its node index, or `None`.
fn bone_index(bone_map: &HumanoidMap, bone: HumanoidBone) -> Option<usize> {
    bone_map.bone_map.get(&bone).map(|nid| nid.0 as usize)
}

/// Map tracking rig pose channels to avatar skeleton local transforms.
///
/// For each tracking channel that is present and has sufficient confidence,
/// the corresponding humanoid bone's local rotation is overwritten (or smoothed)
/// with the tracking orientation.
///
/// Lower body, hands, and finger channels are not mapped in this initial pass.
pub fn retarget_tracking_to_pose(
    tracking: &TrackingRigPose,
    skeleton: &SkeletonAsset,
    humanoid: Option<&HumanoidMap>,
    local_transforms: &mut [Transform],
) {
    retarget_tracking_to_pose_with_params(
        tracking,
        skeleton,
        humanoid,
        local_transforms,
        &TrackingSmoothingParams::default(),
    );
}

/// Like [`retarget_tracking_to_pose`] but accepts explicit smoothing parameters
/// so that the GUI sliders can drive the values instead of using defaults.
pub fn retarget_tracking_to_pose_with_params(
    tracking: &TrackingRigPose,
    skeleton: &SkeletonAsset,
    humanoid: Option<&HumanoidMap>,
    local_transforms: &mut [Transform],
    params: &TrackingSmoothingParams,
) {
    retarget_tracking_to_pose_filtered(
        tracking,
        skeleton,
        humanoid,
        local_transforms,
        params,
        true,
        true,
    );
}

/// Like [`retarget_tracking_to_pose_with_params`] but allows selectively
/// disabling hand and face retargeting via boolean flags.
pub fn retarget_tracking_to_pose_filtered(
    tracking: &TrackingRigPose,
    skeleton: &SkeletonAsset,
    humanoid: Option<&HumanoidMap>,
    local_transforms: &mut [Transform],
    params: &TrackingSmoothingParams,
    hand_tracking_enabled: bool,
    face_tracking_enabled: bool,
) {
    let _ = skeleton;
    let _ = face_tracking_enabled; // used by expression retargeting (separate call path)

    let Some(bone_map) = humanoid else {
        return;
    };

    let threshold = params.confidence_threshold;
    let orientation_smooth = params.orientation_smoothing;
    let position_smooth = params.position_smoothing;

    // Helper closure: apply both rotation and position for a target/bone pair.
    let mut apply_target = |target: &RigTarget, bone: HumanoidBone| {
        if let Some(idx) = bone_index(bone_map, bone) {
            apply_target_rotation(target, idx, threshold, orientation_smooth, local_transforms);
            apply_target_position(target, idx, threshold, position_smooth, local_transforms);
        }
    };

    // --- Head ---
    if let Some(ref target) = tracking.head {
        apply_target(target, HumanoidBone::Head);
    }

    // --- Neck ---
    if let Some(ref target) = tracking.neck {
        apply_target(target, HumanoidBone::Neck);
    }

    // --- Spine ---
    if let Some(ref target) = tracking.spine {
        apply_target(target, HumanoidBone::Spine);
    }

    // --- Left Shoulder ---
    if let Some(ref target) = tracking.shoulders.left {
        apply_target(target, HumanoidBone::LeftShoulder);
    }

    // --- Right Shoulder ---
    if let Some(ref target) = tracking.shoulders.right {
        apply_target(target, HumanoidBone::RightShoulder);
    }

    // --- Left Upper Arm ---
    if let Some(ref target) = tracking.arms.left_upper {
        apply_target(target, HumanoidBone::LeftUpperArm);
    }

    // --- Left Lower Arm ---
    if let Some(ref target) = tracking.arms.left_lower {
        apply_target(target, HumanoidBone::LeftLowerArm);
    }

    // --- Right Upper Arm ---
    if let Some(ref target) = tracking.arms.right_upper {
        apply_target(target, HumanoidBone::RightUpperArm);
    }

    // --- Right Lower Arm ---
    if let Some(ref target) = tracking.arms.right_lower {
        apply_target(target, HumanoidBone::RightLowerArm);
    }

    // --- Hands (only if hand tracking is enabled) ---
    if hand_tracking_enabled {
        if let Some(ref hands) = tracking.hands {
            if let Some(ref target) = hands.left {
                apply_target(target, HumanoidBone::LeftHand);
            }
            if let Some(ref target) = hands.right {
                apply_target(target, HumanoidBone::RightHand);
            }
        }
    }
}

/// Resolved expression weight for the avatar, matching a tracking expression
/// name to a valid avatar expression definition.
#[derive(Clone, Debug)]
pub struct ResolvedExpressionWeight {
    pub name: String,
    pub weight: f32,
}

/// Map tracking expression weights to avatar expression definitions.
///
/// Only expressions that appear in both the tracking data and the avatar's
/// `ExpressionAssetSet` are returned. Weights are clamped to [0, 1] and
/// optionally smoothed against previous values.
///
/// Returns a vec of resolved expression weights ready for blend-shape application.
pub fn retarget_expressions(
    tracking: &TrackingRigPose,
    avatar_expressions: &ExpressionAssetSet,
    previous_weights: Option<&[ResolvedExpressionWeight]>,
) -> Vec<ResolvedExpressionWeight> {
    let params = TrackingSmoothingParams::default();
    let face_confidence = tracking.confidence.face_confidence;

    // If face confidence is below threshold, return empty (keep previous values upstream).
    if face_confidence < params.confidence_threshold {
        return previous_weights.map(|p| p.to_vec()).unwrap_or_default();
    }

    let smoothing = params.expression_smoothing;

    avatar_expressions
        .expressions
        .iter()
        .filter_map(|expr_def| {
            // Find matching tracking weight by name.
            let tracking_weight = tracking
                .expressions
                .weights
                .iter()
                .find(|tw| tw.name == expr_def.name)?;

            let raw_weight = tracking_weight.weight.clamp(0.0, 1.0);

            // Apply smoothing against previous value if available.
            let final_weight = if smoothing > 0.0 {
                if let Some(prev) =
                    previous_weights.and_then(|pw| pw.iter().find(|p| p.name == expr_def.name))
                {
                    let t = 1.0 - smoothing;
                    prev.weight + t * (raw_weight - prev.weight)
                } else {
                    raw_weight
                }
            } else {
                raw_weight
            };

            Some(ResolvedExpressionWeight {
                name: expr_def.name.clone(),
                weight: final_weight,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::{ExpressionDef, NodeId, SkeletonNode};
    use crate::tracking::{
        ExpressionWeight, ExpressionWeightSet, RigArmTargets, RigShoulderTargets, RigTarget,
        TrackingConfidenceMap,
    };
    use std::collections::HashMap;

    fn identity_quat() -> Quat {
        [0.0, 0.0, 0.0, 1.0]
    }

    fn test_rotation() -> Quat {
        // ~30 degree rotation around Y axis
        let half_angle = std::f32::consts::FRAC_PI_6;
        [0.0, half_angle.sin(), 0.0, half_angle.cos()]
    }

    fn make_skeleton(count: usize) -> SkeletonAsset {
        let nodes: Vec<SkeletonNode> = (0..count)
            .map(|i| SkeletonNode {
                id: NodeId(i as u64),
                name: format!("node_{}", i),
                parent: None,
                children: vec![],
                rest_local: Transform::default(),
                humanoid_bone: None,
            })
            .collect();
        SkeletonAsset {
            nodes,
            root_nodes: vec![NodeId(0)],
            inverse_bind_matrices: vec![],
        }
    }

    fn make_bone_map(mappings: &[(HumanoidBone, usize)]) -> HumanoidMap {
        let mut bone_map = HashMap::new();
        for (bone, idx) in mappings {
            bone_map.insert(bone.clone(), NodeId(*idx as u64));
        }
        HumanoidMap { bone_map }
    }

    fn high_confidence_target(orientation: Quat) -> RigTarget {
        RigTarget {
            position: [0.0, 0.0, 0.0],
            orientation,
            confidence: 0.95,
        }
    }

    fn low_confidence_target(orientation: Quat) -> RigTarget {
        RigTarget {
            position: [0.0, 0.0, 0.0],
            orientation,
            confidence: 0.1,
        }
    }

    #[test]
    fn test_head_rotation_applied() {
        let skeleton = make_skeleton(5);
        let bone_map = make_bone_map(&[(HumanoidBone::Head, 2)]);
        let mut transforms = vec![Transform::default(); 5];
        let rot = test_rotation();

        let tracking = TrackingRigPose {
            head: Some(high_confidence_target(rot)),
            ..TrackingRigPose::default()
        };

        retarget_tracking_to_pose(&tracking, &skeleton, Some(&bone_map), &mut transforms);

        // With smoothing, the result should be between identity and the target.
        // It should NOT still be identity (i.e., the rotation was applied).
        let result = transforms[2].rotation;
        assert!(
            result[1].abs() > 0.01,
            "Head Y rotation should be non-zero after retargeting"
        );
    }

    #[test]
    fn test_low_confidence_skipped() {
        let skeleton = make_skeleton(5);
        let bone_map = make_bone_map(&[(HumanoidBone::Head, 2)]);
        let mut transforms = vec![Transform::default(); 5];
        let rot = test_rotation();

        let tracking = TrackingRigPose {
            head: Some(low_confidence_target(rot)),
            ..TrackingRigPose::default()
        };

        retarget_tracking_to_pose(&tracking, &skeleton, Some(&bone_map), &mut transforms);

        // Rotation should remain identity because confidence is too low.
        assert_eq!(transforms[2].rotation, identity_quat());
    }

    #[test]
    fn test_no_humanoid_map_is_noop() {
        let skeleton = make_skeleton(3);
        let mut transforms = vec![Transform::default(); 3];

        let tracking = TrackingRigPose {
            head: Some(high_confidence_target(test_rotation())),
            ..TrackingRigPose::default()
        };

        retarget_tracking_to_pose(&tracking, &skeleton, None, &mut transforms);

        // All transforms should remain identity.
        for t in &transforms {
            assert_eq!(t.rotation, identity_quat());
        }
    }

    #[test]
    fn test_arms_and_shoulders_applied() {
        let skeleton = make_skeleton(10);
        let bone_map = make_bone_map(&[
            (HumanoidBone::LeftShoulder, 3),
            (HumanoidBone::RightShoulder, 4),
            (HumanoidBone::LeftUpperArm, 5),
            (HumanoidBone::RightUpperArm, 6),
            (HumanoidBone::LeftLowerArm, 7),
            (HumanoidBone::RightLowerArm, 8),
        ]);
        let mut transforms = vec![Transform::default(); 10];
        let rot = test_rotation();

        let tracking = TrackingRigPose {
            shoulders: RigShoulderTargets {
                left: Some(high_confidence_target(rot)),
                right: Some(high_confidence_target(rot)),
            },
            arms: RigArmTargets {
                left_upper: Some(high_confidence_target(rot)),
                left_lower: Some(high_confidence_target(rot)),
                right_upper: Some(high_confidence_target(rot)),
                right_lower: Some(high_confidence_target(rot)),
            },
            ..TrackingRigPose::default()
        };

        retarget_tracking_to_pose(&tracking, &skeleton, Some(&bone_map), &mut transforms);

        // All mapped bones should have non-identity rotation.
        for idx in [3, 4, 5, 6, 7, 8] {
            assert!(
                transforms[idx].rotation[1].abs() > 0.01,
                "Bone at index {} should have non-identity rotation",
                idx
            );
        }
    }

    #[test]
    fn test_expression_retargeting() {
        let avatar_expressions = ExpressionAssetSet {
            expressions: vec![
                ExpressionDef {
                    name: "blink".to_string(),
                    weight: 0.0,
                },
                ExpressionDef {
                    name: "smile".to_string(),
                    weight: 0.0,
                },
            ],
        };

        let tracking = TrackingRigPose {
            expressions: ExpressionWeightSet {
                weights: vec![
                    ExpressionWeight {
                        name: "blink".to_string(),
                        weight: 0.8,
                    },
                    ExpressionWeight {
                        name: "unknown_expr".to_string(),
                        weight: 1.0,
                    },
                ],
            },
            confidence: TrackingConfidenceMap {
                head_confidence: 0.9,
                torso_confidence: 0.9,
                left_arm_confidence: 0.9,
                right_arm_confidence: 0.9,
                face_confidence: 0.9,
                ..TrackingConfidenceMap::default()
            },
            ..TrackingRigPose::default()
        };

        let result = retarget_expressions(&tracking, &avatar_expressions, None);

        // Only "blink" should match; "smile" has no tracking data, "unknown_expr"
        // has no avatar definition.
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "blink");
        assert!((result[0].weight - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_expression_low_face_confidence_preserves_previous() {
        let avatar_expressions = ExpressionAssetSet {
            expressions: vec![ExpressionDef {
                name: "blink".to_string(),
                weight: 0.0,
            }],
        };

        let previous = vec![ResolvedExpressionWeight {
            name: "blink".to_string(),
            weight: 0.5,
        }];

        let tracking = TrackingRigPose {
            expressions: ExpressionWeightSet {
                weights: vec![ExpressionWeight {
                    name: "blink".to_string(),
                    weight: 1.0,
                }],
            },
            confidence: TrackingConfidenceMap {
                head_confidence: 0.9,
                torso_confidence: 0.9,
                left_arm_confidence: 0.9,
                right_arm_confidence: 0.9,
                face_confidence: 0.1, // low
                ..TrackingConfidenceMap::default()
            },
            ..TrackingRigPose::default()
        };

        let result = retarget_expressions(&tracking, &avatar_expressions, Some(&previous));

        // Should preserve previous value since face confidence is too low.
        assert_eq!(result.len(), 1);
        assert!((result[0].weight - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_quat_normalize() {
        let q = quat_normalize(&[0.0, 0.0, 0.0, 2.0]);
        assert!((q[3] - 1.0).abs() < 1e-6);

        let zero = quat_normalize(&[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(zero, [0.0, 0.0, 0.0, 1.0]);
    }
}
