//! SMPL kinematic tree + forward kinematics for HMR2 output.
//!
//! HMR2 emits one global rotation (Pelvis) and 23 local joint rotations
//! relative to the SMPL kinematic-tree parents. To populate a
//! [`SourceSkeleton`] for the existing avatar solver we compute each
//! joint's WORLD POSITION by chaining rotations from Pelvis outward —
//! same FK that SMPL's PyTorch forward() does, just on the rest joint
//! positions (no shape-dependent blend, betas held at 0).
//!
//! Rest joints come from the neutral SMPL template (Loper et al. 2015
//! figures, hardcoded here so we don't bundle the license-gated
//! SMPL_NEUTRAL.pkl). The numbers are good to a few mm — close enough
//! for the solver, which only consumes RELATIVE bone direction.

use crate::asset::HumanoidBone;

/// SMPL 24-joint kinematic tree. `PARENTS[i]` is the parent index of
/// joint `i`, with -1 (encoded as -1_i8) for the Pelvis root.
pub const PARENTS: [i8; 24] = [
    -1, 0, 0, 0,  // Pelvis, L_Hip, R_Hip, Spine1
    1, 2, 3,      // L_Knee, R_Knee, Spine2
    4, 5, 6,      // L_Ankle, R_Ankle, Spine3
    7, 8, 9,      // L_Foot, R_Foot, Neck
    9, 9, 12,     // L_Collar, R_Collar, Head
    13, 14,       // L_Shoulder, R_Shoulder
    16, 17,       // L_Elbow, R_Elbow
    18, 19,       // L_Wrist, R_Wrist
    20, 21,       // L_Hand, R_Hand
];

/// SMPL joint names. Index matches the `PARENTS` array.
pub const NAMES: [&str; 24] = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
];

/// SMPL rest joint world positions for neutral betas (T-pose).
/// Approximate; sub-cm accuracy is sufficient because the solver
/// consumes only relative bone directions.
pub const REST_JOINTS: [[f32; 3]; 24] = [
    [ 0.0000,  0.0000,  0.0000], //  0 Pelvis
    [ 0.0580, -0.0820,  0.0140], //  1 L_Hip
    [-0.0600, -0.0900,  0.0140], //  2 R_Hip
    [ 0.0040,  0.1230, -0.0140], //  3 Spine1
    [ 0.0440, -0.3880, -0.0050], //  4 L_Knee
    [-0.0450, -0.3940, -0.0050], //  5 R_Knee
    [ 0.0050,  0.2750, -0.0190], //  6 Spine2
    [ 0.0340, -0.8210, -0.0390], //  7 L_Ankle
    [-0.0340, -0.8310, -0.0390], //  8 R_Ankle
    [ 0.0050,  0.3190,  0.0000], //  9 Spine3
    [ 0.0970, -0.8760,  0.1180], // 10 L_Foot
    [-0.0960, -0.8860,  0.1180], // 11 R_Foot
    [ 0.0000,  0.5120, -0.0300], // 12 Neck
    [ 0.0730,  0.4520, -0.0400], // 13 L_Collar
    [-0.0730,  0.4520, -0.0400], // 14 R_Collar
    [ 0.0000,  0.5830, -0.0110], // 15 Head
    [ 0.1810,  0.4340, -0.0620], // 16 L_Shoulder
    [-0.1810,  0.4340, -0.0620], // 17 R_Shoulder
    [ 0.4500,  0.4200, -0.0800], // 18 L_Elbow
    [-0.4500,  0.4200, -0.0800], // 19 R_Elbow
    [ 0.7170,  0.4200, -0.0800], // 20 L_Wrist
    [-0.7170,  0.4200, -0.0800], // 21 R_Wrist
    [ 0.8120,  0.4200, -0.0800], // 22 L_Hand
    [-0.8120,  0.4200, -0.0800], // 23 R_Hand
];

/// Map SMPL joint index → VRM HumanoidBone. SMPL 22/23 (hand-as-whole)
/// have no direct VRM analogue (VRM expects per-finger bones); the
/// existing fingertip path stays driven by ViTPose/RTMW3D for now.
pub fn smpl_index_to_vrm(idx: usize) -> Option<HumanoidBone> {
    Some(match idx {
        0 => HumanoidBone::Hips,
        // SourceSkeleton convention: HumanoidBone::Left* holds the
        // AVATAR's left, which under selfie-mirror is the SUBJECT's
        // right. SMPL's `L_*` joints are SUBJECT's anatomical left,
        // so swap L↔R when mapping to source-side keys.
        1 => HumanoidBone::RightUpperLeg,
        2 => HumanoidBone::LeftUpperLeg,
        3 => HumanoidBone::Spine,
        4 => HumanoidBone::RightLowerLeg,
        5 => HumanoidBone::LeftLowerLeg,
        6 => HumanoidBone::Chest,
        7 => HumanoidBone::RightFoot,
        8 => HumanoidBone::LeftFoot,
        9 => HumanoidBone::UpperChest,
        // SMPL 10/11 are foot/toe tips. VRM has no Toes bone in this
        // rig (LeftFoot is the ankle), so skip — SMPL 7/8 already
        // mapped to L/R Foot.
        10 | 11 => return None,
        12 => HumanoidBone::Neck,
        13 => HumanoidBone::RightShoulder,
        14 => HumanoidBone::LeftShoulder,
        15 => HumanoidBone::Head,
        16 => HumanoidBone::RightUpperArm,
        17 => HumanoidBone::LeftUpperArm,
        18 => HumanoidBone::RightLowerArm,
        19 => HumanoidBone::LeftLowerArm,
        20 => HumanoidBone::RightHand,
        21 => HumanoidBone::LeftHand,
        _ => return None, // 22 L_Hand / 23 R_Hand have no direct VRM equivalent
    })
}

/// Compute world joint positions in SMPL frame given the HMR2 output's
/// local joint rotations and a global root rotation.
///
/// SMPL convention (matched by 4D-Humans):
///   * +Y down (head is at negative Y in joint coordinates here we
///     flip the sign so the convention matches VRM Y-up)
///   * +X is anatomical left
///   * +Z is forward toward camera
///
/// We return positions in a **Y-up** frame to match the rest of the
/// source-skeleton pipeline; the caller still needs to negate X for
/// VRM's selfie-mirror semantics.
pub fn forward_kinematics(
    global_orient: &[[f32; 3]; 3],
    body_pose: &[[[f32; 3]; 3]; 23],
) -> [[f32; 3]; 24] {
    let mut world_rot = [[[0.0_f32; 3]; 3]; 24];
    let mut world_pos = [[0.0_f32; 3]; 24];

    // Pelvis: rotation is global_orient, position stays at origin.
    world_rot[0] = *global_orient;

    for i in 1..24 {
        let parent = PARENTS[i] as usize;
        let local_rot = &body_pose[i - 1]; // body_pose index = joint - 1
        // World rotation = parent_world_rot * local_rot
        world_rot[i] = mat3_mul(&world_rot[parent], local_rot);

        // Offset from parent in rest pose (local translation)
        let rest_offset = sub3(&REST_JOINTS[i], &REST_JOINTS[parent]);
        // World offset = parent_world_rot * rest_offset
        let world_offset = mat3_apply(&world_rot[parent], &rest_offset);
        world_pos[i] = add3(&world_pos[parent], &world_offset);
    }

    // Convert from SMPL frame (+Y down) to source-space frame (+Y up):
    // negate Y on every output position.
    for p in &mut world_pos {
        p[1] = -p[1];
    }
    world_pos
}

/// Same FK as [`forward_kinematics`] but returns positions in SMPL's
/// native frame (Y-down). Used for projecting joint world coords back
/// to image space via the weak-perspective camera, which is also in
/// Y-down convention.
pub fn forward_kinematics_camera_frame(
    global_orient: &[[f32; 3]; 3],
    body_pose: &[[[f32; 3]; 3]; 23],
) -> [[f32; 3]; 24] {
    let mut world_rot = [[[0.0_f32; 3]; 3]; 24];
    let mut world_pos = [[0.0_f32; 3]; 24];
    world_rot[0] = *global_orient;
    for i in 1..24 {
        let parent = PARENTS[i] as usize;
        let local_rot = &body_pose[i - 1];
        world_rot[i] = mat3_mul(&world_rot[parent], local_rot);
        let rest_offset = sub3(&REST_JOINTS[i], &REST_JOINTS[parent]);
        let world_offset = mat3_apply(&world_rot[parent], &rest_offset);
        world_pos[i] = add3(&world_pos[parent], &world_offset);
    }
    world_pos
}

#[inline]
fn mat3_mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0_f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

#[inline]
fn mat3_apply(m: &[[f32; 3]; 3], v: &[f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

#[inline]
fn sub3(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn add3(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
