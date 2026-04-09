use std::sync::Arc;

use crate::asset::{AvatarAsset, AvatarAssetId, ClothOverlayId, NodeId, Transform, Vec3};
use crate::avatar::animation::AnimationState;
use crate::avatar::pose::AvatarPose;
use crate::tracking::TrackingRigPose;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AvatarInstanceId(pub u64);

#[derive(Clone, Debug)]
pub struct AvatarInstance {
    pub id: AvatarInstanceId,
    pub asset_id: AvatarAssetId,
    pub asset: Arc<AvatarAsset>,
    pub world_transform: Transform,
    pub pose: AvatarPose,
    pub animation_state: AnimationState,
    pub secondary_motion: SecondaryMotionState,
    pub tracking_input: Option<TrackingRigPose>,
    pub attached_cloth: Option<ClothOverlayId>,
    pub cloth_enabled: bool,
    pub cloth_state: Option<ClothState>,
}

#[derive(Clone, Debug)]
pub struct SecondaryMotionState {
    pub spring_states: Vec<SpringChainState>,
    pub collision_cache: SecondaryMotionCollisionCache,
}

#[derive(Clone, Debug)]
pub struct SpringChainState {
    pub chain_root: NodeId,
    pub joints: Vec<NodeId>,
    pub tip_position: Vec3,
    pub tip_previous_position: Vec3,
}

#[derive(Clone, Debug)]
pub struct SecondaryMotionCollisionCache {
    pub contact_count: usize,
}

#[derive(Clone, Debug)]
pub struct ClothState {
    pub overlay_id: ClothOverlayId,
    pub enabled: bool,
    pub sim_positions: Vec<Vec3>,
    pub prev_sim_positions: Vec<Vec3>,
    pub sim_normals: Vec<Vec3>,
    pub constraint_cache: ClothConstraintRuntimeCache,
    pub collision_cache: ClothCollisionRuntimeCache,
    pub deform_output: ClothDeformOutput,
}

#[derive(Clone, Debug)]
pub struct ClothConstraintRuntimeCache {
    pub active_constraint_count: usize,
}

#[derive(Clone, Debug)]
pub struct ClothCollisionRuntimeCache {
    pub active_collision_count: usize,
}

#[derive(Clone, Debug)]
pub struct ClothDeformOutput {
    pub deformed_positions: Vec<Vec3>,
    pub deformed_normals: Option<Vec<Vec3>>,
    pub version: u64,
}

impl AvatarInstance {
    pub fn new(id: AvatarInstanceId, asset: Arc<AvatarAsset>) -> Self {
        let node_count = asset.skeleton.nodes.len();
        let asset_id = asset.id.clone();

        let spring_states = asset
            .spring_bones
            .iter()
            .map(|spring| SpringChainState {
                chain_root: spring.chain_root.clone(),
                joints: spring.joints.clone(),
                tip_position: [0.0, 0.0, 0.0],
                tip_previous_position: [0.0, 0.0, 0.0],
            })
            .collect();

        Self {
            id,
            asset_id,
            asset,
            world_transform: Transform::default(),
            pose: AvatarPose::identity(node_count),
            animation_state: AnimationState::default(),
            secondary_motion: SecondaryMotionState {
                spring_states,
                collision_cache: SecondaryMotionCollisionCache { contact_count: 0 },
            },
            tracking_input: None,
            attached_cloth: None,
            cloth_enabled: false,
            cloth_state: None,
        }
    }

    pub fn attach_cloth(&mut self, overlay_id: ClothOverlayId) {
        self.attached_cloth = Some(overlay_id.clone());
        self.cloth_enabled = true;
        self.cloth_state = Some(ClothState {
            overlay_id,
            enabled: true,
            sim_positions: vec![],
            prev_sim_positions: vec![],
            sim_normals: vec![],
            constraint_cache: ClothConstraintRuntimeCache {
                active_constraint_count: 0,
            },
            collision_cache: ClothCollisionRuntimeCache {
                active_collision_count: 0,
            },
            deform_output: ClothDeformOutput {
                deformed_positions: vec![],
                deformed_normals: None,
                version: 0,
            },
        });
    }

    pub fn detach_cloth(&mut self) {
        self.attached_cloth = None;
        self.cloth_enabled = false;
        self.cloth_state = None;
    }

    pub fn apply_tracking(&mut self, tracking: TrackingRigPose) {
        self.tracking_input = Some(tracking);
    }

    pub fn build_base_pose(&mut self) {}

    pub fn compute_global_pose(&mut self) {}

    pub fn build_skinning_matrices(&mut self) {}
}

impl Default for SecondaryMotionState {
    fn default() -> Self {
        Self {
            spring_states: vec![],
            collision_cache: SecondaryMotionCollisionCache { contact_count: 0 },
        }
    }
}
