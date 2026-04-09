use crate::asset::{HumanoidBone, SkeletonAsset, Transform};
use crate::tracking::TrackingRigPose;

pub fn retarget_tracking_to_pose(
    tracking: &TrackingRigPose,
    skeleton: &SkeletonAsset,
    humanoid: Option<&crate::asset::HumanoidMap>,
    local_transforms: &mut [Transform],
) {
    let _ = skeleton;

    let Some(bone_map) = humanoid else {
        return;
    };

    if let Some(ref head_target) = tracking.head {
        if let Some(&head_node) = bone_map.bone_map.get(&HumanoidBone::Head) {
            let idx = head_node.0 as usize;
            if idx < local_transforms.len() {
                local_transforms[idx].rotation = head_target.orientation;
            }
        }
    }

    if let Some(ref spine_target) = tracking.spine {
        if let Some(&spine_node) = bone_map.bone_map.get(&HumanoidBone::Spine) {
            let idx = spine_node.0 as usize;
            if idx < local_transforms.len() {
                local_transforms[idx].rotation = spine_target.orientation;
            }
        }
    }
}
