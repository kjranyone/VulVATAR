use crate::asset::{identity_matrix, Mat4, Transform};

pub type FrameTimestamp = u64;

#[derive(Clone, Debug)]
pub struct AvatarPose {
    pub local_transforms: Vec<Transform>,
    pub global_transforms: Vec<Mat4>,
    pub skinning_matrices: Vec<Mat4>,
    pub pose_timestamp: FrameTimestamp,
}

impl AvatarPose {
    pub fn identity(node_count: usize) -> Self {
        Self {
            local_transforms: vec![Transform::default(); node_count],
            global_transforms: vec![identity_matrix(); node_count],
            skinning_matrices: vec![identity_matrix(); node_count],
            pose_timestamp: 0,
        }
    }
}
