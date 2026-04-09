use crate::asset::vrm::{
    ClothAsset, ColliderAsset, MaterialAsset, MeshAsset, SkeletonAsset, SpringBoneAsset, VrmAsset,
};
use crate::simulation::cloth::ClothState;
use crate::tracking::TrackingRigPose;

#[derive(Clone, Debug)]
pub struct AvatarAsset {
    pub name: String,
    pub skeleton: SkeletonAsset,
    pub meshes: Vec<MeshAsset>,
    pub materials: Vec<MaterialAsset>,
    pub spring_bones: Vec<SpringBoneAsset>,
    pub colliders: Vec<ColliderAsset>,
    pub cloth_regions: Vec<ClothAsset>,
}

#[derive(Clone, Debug)]
pub struct AvatarInstance {
    pub asset: AvatarAsset,
    pub pose: AvatarPose,
    pub tracking_pose: TrackingRigPose,
    pub spring_state: SecondaryMotionState,
    pub cloth_state: ClothState,
}

#[derive(Clone, Debug)]
pub struct AvatarPose {
    pub local_joints: Vec<JointTransform>,
    pub global_matrices: Vec<Mat4>,
    pub skinning_matrices: Vec<Mat4>,
}

#[derive(Clone, Debug, Default)]
pub struct JointTransform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

#[derive(Clone, Debug, Default)]
pub struct SecondaryMotionState {
    pub chain_states: Vec<SpringChainState>,
}

#[derive(Clone, Debug, Default)]
pub struct SpringChainState {
    pub chain_name: String,
    pub joint_indices: Vec<usize>,
    pub tip_position: Vec3,
    pub tip_velocity: Vec3,
}

pub type Vec3 = [f32; 3];
pub type Quat = [f32; 4];
pub type Mat4 = [[f32; 4]; 4];

impl AvatarAsset {
    pub fn from_vrm(asset: VrmAsset) -> Self {
        Self {
            name: asset.meta.name,
            skeleton: asset.skeleton,
            meshes: asset.meshes,
            materials: asset.materials,
            spring_bones: asset.spring_bones,
            colliders: asset.colliders,
            cloth_regions: asset.cloth_regions,
        }
    }

    pub fn joint_count(&self) -> usize {
        self.skeleton.joints.len()
    }
}

impl AvatarInstance {
    pub fn new(asset: AvatarAsset) -> Self {
        let joint_count = asset.joint_count();
        let cloth_state = ClothState::from_assets(&asset.cloth_regions);
        let spring_state = SecondaryMotionState {
            chain_states: asset
                .spring_bones
                .iter()
                .map(|spring| SpringChainState {
                    chain_name: spring.name.clone(),
                    joint_indices: spring.joint_indices.clone(),
                    tip_position: [0.0, 0.0, 0.0],
                    tip_velocity: [0.0, 0.0, 0.0],
                })
                .collect(),
        };

        Self {
            asset,
            pose: AvatarPose::identity(joint_count),
            tracking_pose: TrackingRigPose::default(),
            spring_state,
            cloth_state,
        }
    }

    pub fn apply_tracking(&mut self, tracking_pose: TrackingRigPose) {
        self.tracking_pose = tracking_pose;
    }
}

impl AvatarPose {
    pub fn identity(joint_count: usize) -> Self {
        let local = JointTransform {
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        };
        let matrix = identity_matrix();

        Self {
            local_joints: vec![local; joint_count],
            global_matrices: vec![matrix; joint_count],
            skinning_matrices: vec![matrix; joint_count],
        }
    }
}

fn identity_matrix() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}
