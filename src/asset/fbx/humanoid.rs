use crate::asset::HumanoidBone;

/// Match a raw FBX bone/node name to a [`HumanoidBone`].
///
/// Handles common naming conventions from:
/// - Unity Humanoid (`Hips`, `Spine`, `UpperArm_L`, `ThumbProximal_L`, etc.)
/// - VRChat / BOOTH models (`Arm_L`, `Elbow_L`, `Wrist_L`, `Leg_L`, `Knee_L`, `Foot_L`)
/// - Blender / MMD / generic rigs (`left_arm`, `RightHand`, `Thumb.01.L`, etc.)
pub fn map_bone_name(raw_name: &str) -> Option<HumanoidBone> {
    // Strip namespaces or prefixes (e.g. "Armature|Hips" -> "Hips", "Root:Hips" -> "Hips")
    let name = raw_name
        .rsplit('|')
        .next()
        .unwrap_or(raw_name)
        .rsplit(':')
        .next()
        .unwrap_or(raw_name);

    // Normalize: lowercase, convert '.' and '-' to '_'
    let norm = name.to_lowercase().replace(['.', '-'], "_");

    // Exact or normalized lookups
    match norm.as_str() {
        // Spine & Head
        "hips" | "pelvis" | "root_hips" => Some(HumanoidBone::Hips),
        "spine" | "spine_01" | "spine1" => Some(HumanoidBone::Spine),
        "chest" | "spine_02" | "spine2" => Some(HumanoidBone::Chest),
        "upperchest" | "upper_chest" | "spine_03" | "spine3" => Some(HumanoidBone::UpperChest),
        "neck" | "neck_01" => Some(HumanoidBone::Neck),
        "head" => Some(HumanoidBone::Head),

        // Left arm
        "leftshoulder" | "shoulder_l" | "l_shoulder" | "left_shoulder" | "clavicle_l" => {
            Some(HumanoidBone::LeftShoulder)
        }
        "leftupperarm" | "upperarm_l" | "upper_arm_l" | "l_upperarm" | "left_upper_arm"
        | "arm_l" | "l_arm" | "leftarm" => Some(HumanoidBone::LeftUpperArm),
        "leftlowerarm" | "lowerarm_l" | "lower_arm_l" | "l_lowerarm" | "left_lower_arm"
        | "forearm_l" | "l_forearm" | "elbow_l" => Some(HumanoidBone::LeftLowerArm),
        "lefthand" | "hand_l" | "l_hand" | "left_hand" | "wrist_l" => {
            Some(HumanoidBone::LeftHand)
        }

        // Right arm
        "rightshoulder" | "shoulder_r" | "r_shoulder" | "right_shoulder" | "clavicle_r" => {
            Some(HumanoidBone::RightShoulder)
        }
        "rightupperarm" | "upperarm_r" | "upper_arm_r" | "r_upperarm" | "right_upper_arm"
        | "arm_r" | "r_arm" | "rightarm" => Some(HumanoidBone::RightUpperArm),
        "rightlowerarm" | "lowerarm_r" | "lower_arm_r" | "r_lowerarm" | "right_lower_arm"
        | "forearm_r" | "r_forearm" | "elbow_r" => Some(HumanoidBone::RightLowerArm),
        "righthand" | "hand_r" | "r_hand" | "right_hand" | "wrist_r" => {
            Some(HumanoidBone::RightHand)
        }

        // Left leg
        "leftupperleg" | "upperleg_l" | "upper_leg_l" | "l_upperleg" | "left_upper_leg"
        | "leg_l" | "l_leg" | "upleg_l" | "leftupleg" | "thigh_l" => {
            Some(HumanoidBone::LeftUpperLeg)
        }
        "leftlowerleg" | "lowerleg_l" | "lower_leg_l" | "l_lowerleg" | "left_lower_leg"
        | "knee_l" | "shin_l" | "calf_l" => Some(HumanoidBone::LeftLowerLeg),
        "leftfoot" | "foot_l" | "l_foot" | "left_foot" | "ankle_l" => {
            Some(HumanoidBone::LeftFoot)
        }

        // Right leg
        "rightupperleg" | "upperleg_r" | "upper_leg_r" | "r_upperleg" | "right_upper_leg"
        | "leg_r" | "r_leg" | "upleg_r" | "rightupleg" | "thigh_r" => {
            Some(HumanoidBone::RightUpperLeg)
        }
        "rightlowerleg" | "lowerleg_r" | "lower_leg_r" | "r_lowerleg" | "right_lower_leg"
        | "knee_r" | "shin_r" | "calf_r" => Some(HumanoidBone::RightLowerLeg),
        "rightfoot" | "foot_r" | "r_foot" | "right_foot" | "ankle_r" => {
            Some(HumanoidBone::RightFoot)
        }

        // Left fingers - Thumb
        "thumbproximal_l" | "thumb_proximal_l" | "thumb_01_l" | "thumb1_l" | "leftthumbproximal"
        | "lefthandthumb1" => Some(HumanoidBone::LeftThumbProximal),
        "thumbintermediate_l" | "thumb_intermediate_l" | "thumb_02_l" | "thumb2_l"
        | "leftthumbintermediate" | "lefthandthumb2" => Some(HumanoidBone::LeftThumbIntermediate),
        "thumbdistal_l" | "thumb_distal_l" | "thumb_03_l" | "thumb3_l" | "leftthumbdistal"
        | "lefthandthumb3" => Some(HumanoidBone::LeftThumbDistal),

        // Left fingers - Index
        "indexproximal_l" | "index_proximal_l" | "index_01_l" | "index1_l" | "leftindexproximal"
        | "lefthandindex1" => Some(HumanoidBone::LeftIndexProximal),
        "indexintermediate_l" | "index_intermediate_l" | "index_02_l" | "index2_l"
        | "leftindexintermediate" | "lefthandindex2" => Some(HumanoidBone::LeftIndexIntermediate),
        "indexdistal_l" | "index_distal_l" | "index_03_l" | "index3_l" | "leftindexdistal"
        | "lefthandindex3" => Some(HumanoidBone::LeftIndexDistal),

        // Left fingers - Middle
        "middleproximal_l" | "middle_proximal_l" | "middle_01_l" | "middle1_l"
        | "leftmiddleproximal" | "lefthandmiddle1" => Some(HumanoidBone::LeftMiddleProximal),
        "middleintermediate_l" | "middle_intermediate_l" | "middle_02_l" | "middle2_l"
        | "leftmiddleintermediate" | "lefthandmiddle2" => {
            Some(HumanoidBone::LeftMiddleIntermediate)
        }
        "middledistal_l" | "middle_distal_l" | "middle_03_l" | "middle3_l" | "leftmiddledistal"
        | "lefthandmiddle3" => Some(HumanoidBone::LeftMiddleDistal),

        // Left fingers - Ring
        "ringproximal_l" | "ring_proximal_l" | "ring_01_l" | "ring1_l" | "leftringproximal"
        | "lefthandring1" => Some(HumanoidBone::LeftRingProximal),
        "ringintermediate_l" | "ring_intermediate_l" | "ring_02_l" | "ring2_l"
        | "leftringintermediate" | "lefthandring2" => Some(HumanoidBone::LeftRingIntermediate),
        "ringdistal_l" | "ring_distal_l" | "ring_03_l" | "ring3_l" | "leftringdistal"
        | "lefthandring3" => Some(HumanoidBone::LeftRingDistal),

        // Left fingers - Little
        "littleproximal_l" | "little_proximal_l" | "little_01_l" | "little1_l"
        | "leftlittleproximal" | "pinkyproximal_l" | "pinky_01_l" | "lefthandpinky1" => {
            Some(HumanoidBone::LeftLittleProximal)
        }
        "littleintermediate_l" | "little_intermediate_l" | "little_02_l" | "little2_l"
        | "leftlittleintermediate" | "pinkyintermediate_l" | "pinky_02_l" | "lefthandpinky2" => {
            Some(HumanoidBone::LeftLittleIntermediate)
        }
        "littledistal_l" | "little_distal_l" | "little_03_l" | "little3_l" | "leftlittledistal"
        | "pinkydistal_l" | "pinky_03_l" | "lefthandpinky3" => Some(HumanoidBone::LeftLittleDistal),

        // Right fingers - Thumb
        "thumbproximal_r" | "thumb_proximal_r" | "thumb_01_r" | "thumb1_r" | "rightthumbproximal"
        | "righthandthumb1" => Some(HumanoidBone::RightThumbProximal),
        "thumbintermediate_r" | "thumb_intermediate_r" | "thumb_02_r" | "thumb2_r"
        | "rightthumbintermediate" | "righthandthumb2" => {
            Some(HumanoidBone::RightThumbIntermediate)
        }
        "thumbdistal_r" | "thumb_distal_r" | "thumb_03_r" | "thumb3_r" | "rightthumbdistal"
        | "righthandthumb3" => Some(HumanoidBone::RightThumbDistal),

        // Right fingers - Index
        "indexproximal_r" | "index_proximal_r" | "index_01_r" | "index1_r" | "rightindexproximal"
        | "righthandindex1" => Some(HumanoidBone::RightIndexProximal),
        "indexintermediate_r" | "index_intermediate_r" | "index_02_r" | "index2_r"
        | "rightindexintermediate" | "righthandindex2" => {
            Some(HumanoidBone::RightIndexIntermediate)
        }
        "indexdistal_r" | "index_distal_r" | "index_03_r" | "index3_r" | "rightindexdistal"
        | "righthandindex3" => Some(HumanoidBone::RightIndexDistal),

        // Right fingers - Middle
        "middleproximal_r" | "middle_proximal_r" | "middle_01_r" | "middle1_r"
        | "rightmiddleproximal" | "righthandmiddle1" => Some(HumanoidBone::RightMiddleProximal),
        "middleintermediate_r" | "middle_intermediate_r" | "middle_02_r" | "middle2_r"
        | "rightmiddleintermediate" | "righthandmiddle2" => {
            Some(HumanoidBone::RightMiddleIntermediate)
        }
        "middledistal_r" | "middle_distal_r" | "middle_03_r" | "middle3_r"
        | "rightmiddledistal" | "righthandmiddle3" => Some(HumanoidBone::RightMiddleDistal),

        // Right fingers - Ring
        "ringproximal_r" | "ring_proximal_r" | "ring_01_r" | "ring1_r" | "rightringproximal"
        | "righthandring1" => Some(HumanoidBone::RightRingProximal),
        "ringintermediate_r" | "ring_intermediate_r" | "ring_02_r" | "ring2_r"
        | "rightringintermediate" | "righthandring2" => {
            Some(HumanoidBone::RightRingIntermediate)
        }
        "ringdistal_r" | "ring_distal_r" | "ring_03_r" | "ring3_r" | "rightringdistal"
        | "righthandring3" => Some(HumanoidBone::RightRingDistal),

        // Right fingers - Little
        "littleproximal_r" | "little_proximal_r" | "little_01_r" | "little1_r"
        | "rightlittleproximal" | "pinkyproximal_r" | "pinky_01_r" | "righthandpinky1" => {
            Some(HumanoidBone::RightLittleProximal)
        }
        "littleintermediate_r" | "little_intermediate_r" | "little_02_r" | "little2_r"
        | "rightlittleintermediate" | "pinkyintermediate_r" | "pinky_02_r" | "righthandpinky2" => {
            Some(HumanoidBone::RightLittleIntermediate)
        }
        "littledistal_r" | "little_distal_r" | "little_03_r" | "little3_r"
        | "rightlittledistal" | "pinkydistal_r" | "pinky_03_r" | "righthandpinky3" => {
            Some(HumanoidBone::RightLittleDistal)
        }

        _ => None,
    }
}
