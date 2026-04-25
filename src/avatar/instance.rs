#![allow(dead_code)]
use std::sync::Arc;

use crate::asset::{
    AvatarAsset, AvatarAssetId, ClothAsset, ClothOverlayId, Mat4, MeshPrimitiveAsset, NodeId,
    Transform, Vec3,
};
use crate::avatar::animation::{self, AnimationState};
use crate::avatar::pose::AvatarPose;
use crate::avatar::pose_solver::ResolvedExpressionWeight;
use crate::simulation::cloth::{ClothSimState, ClothSimTempBuffers};

/// Multiply two column-major 4x4 matrices: result = a * b.
fn mul_mat4(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = a[0][row] * b[col][0]
                + a[1][row] * b[col][1]
                + a[2][row] * b[col][2]
                + a[3][row] * b[col][3];
        }
    }
    out
}

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
    pub attached_cloth: Option<ClothOverlayId>,
    pub cloth_enabled: bool,
    pub cloth_state: Option<ClothState>,
    pub cloth_sim: Option<ClothSimState>,
    pub cloth_sim_buffers: Option<ClothSimTempBuffers>,
    pub cloth_overlays: Vec<ClothOverlaySlot>,
    pub expression_weights: Vec<ResolvedExpressionWeight>,
    /// Pre-built `Arc<MeshPrimitiveAsset>` per (mesh, primitive) to avoid
    /// cloning vertex data every frame. Built once in `AvatarInstance::new`.
    pub primitive_arcs: Vec<Vec<Arc<MeshPrimitiveAsset>>>,
}

#[derive(Clone, Debug)]
pub struct ClothOverlaySlot {
    pub overlay_id: ClothOverlayId,
    pub enabled: bool,
    pub state: ClothState,
    pub sim: ClothSimState,
    pub buffers: ClothSimTempBuffers,
    /// Filesystem path the overlay was loaded from (if any). Saved with
    /// the project so load-time can re-attach the same overlays via
    /// `persistence::load_cloth_overlay`. `None` for slots created
    /// procedurally via "Add Overlay Slot" — those have no file backing.
    pub source_path: Option<std::path::PathBuf>,
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
    /// Per-joint positions for Verlet integration (one entry per joint).
    pub positions: Vec<Vec3>,
    /// Per-joint previous positions for Verlet integration (one entry per joint).
    pub previous_positions: Vec<Vec3>,
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
        let asset_id = asset.id;

        let spring_states = asset
            .spring_bones
            .iter()
            .map(|spring| {
                let joint_count = spring.joints.len();
                SpringChainState {
                    chain_root: spring.chain_root,
                    joints: spring.joints.clone(),
                    positions: vec![[0.0, 0.0, 0.0]; joint_count],
                    previous_positions: vec![[0.0, 0.0, 0.0]; joint_count],
                }
            })
            .collect();

        let primitive_arcs = asset
            .meshes
            .iter()
            .map(|mesh| {
                mesh.primitives
                    .iter()
                    .map(|prim| Arc::new(prim.clone()))
                    .collect()
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
            attached_cloth: None,
            cloth_enabled: false,
            cloth_state: None,
            cloth_sim: None,
            cloth_sim_buffers: None,
            cloth_overlays: Vec::new(),
            expression_weights: Vec::new(),
            primitive_arcs,
        }
    }

    pub fn attach_cloth(&mut self, overlay_id: ClothOverlayId) {
        self.attached_cloth = Some(overlay_id);
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

    /// Initialise the cloth simulation from a `ClothAsset`.
    /// Call this after `attach_cloth` to populate the PBD solver state.
    pub fn init_cloth_sim(&mut self, cloth_asset: &ClothAsset) {
        let sim = ClothSimState::from_asset(cloth_asset);
        let n = sim.particle_count();

        // Pre-fill ClothState positions from the sim mesh rest pose
        if let Some(cs) = self.cloth_state.as_mut() {
            cs.sim_positions = sim.particles.iter().map(|p| p.position).collect();
            cs.prev_sim_positions = cs.sim_positions.clone();
            cs.deform_output.deformed_positions = cs.sim_positions.clone();
        }

        self.cloth_sim_buffers = Some(ClothSimTempBuffers::new(n));
        self.cloth_sim = Some(sim);
    }

    pub fn detach_cloth(&mut self) {
        self.attached_cloth = None;
        self.cloth_enabled = false;
        self.cloth_state = None;
        self.cloth_sim = None;
        self.cloth_sim_buffers = None;
    }

    pub fn attach_cloth_overlay(&mut self, overlay_id: ClothOverlayId) -> usize {
        let slot_index = self.cloth_overlays.len();
        let state = ClothState {
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
        };
        let sim = ClothSimState::default();
        let n = sim.particle_count();
        let buffers = ClothSimTempBuffers::new(n);

        self.cloth_overlays.push(ClothOverlaySlot {
            overlay_id,
            enabled: true,
            state,
            sim,
            buffers,
            source_path: None,
        });
        self.cloth_enabled = true;
        if self.attached_cloth.is_none() {
            self.attached_cloth = Some(overlay_id);
        }
        slot_index
    }

    pub fn init_cloth_overlay(&mut self, slot_index: usize, cloth_asset: &ClothAsset) {
        if let Some(slot) = self.cloth_overlays.get_mut(slot_index) {
            slot.sim = ClothSimState::from_asset(cloth_asset);
            let n = slot.sim.particle_count();
            slot.state.sim_positions = slot.sim.particles.iter().map(|p| p.position).collect();
            slot.state.prev_sim_positions = slot.state.sim_positions.clone();
            slot.state.deform_output.deformed_positions = slot.state.sim_positions.clone();
            slot.buffers = ClothSimTempBuffers::new(n);
        }
    }

    pub fn remove_cloth_overlay(&mut self, slot_index: usize) {
        if slot_index < self.cloth_overlays.len() {
            let removed = self.cloth_overlays.remove(slot_index);
            if self.attached_cloth == Some(removed.overlay_id) {
                self.attached_cloth = self.cloth_overlays.first().map(|s| s.overlay_id);
            }
            self.cloth_enabled = self.cloth_state.is_some() || !self.cloth_overlays.is_empty();
        }
    }

    pub fn cloth_overlay_count(&self) -> usize {
        self.cloth_overlays.len()
    }

    pub fn get_cloth_overlay(&self, index: usize) -> Option<&ClothOverlaySlot> {
        self.cloth_overlays.get(index)
    }

    pub fn get_cloth_overlay_mut(&mut self, index: usize) -> Option<&mut ClothOverlaySlot> {
        self.cloth_overlays.get_mut(index)
    }

    pub fn build_base_pose(&mut self) {
        let skeleton = &self.asset.skeleton;
        for (i, node) in skeleton.nodes.iter().enumerate() {
            self.pose.local_transforms[i] = node.rest_local.clone();
        }

        // If an animation clip is active, sample it and overlay onto the rest pose.
        if let Some(clip_id) = &self.animation_state.active_clip {
            if let Some(clip) = animation::find_clip(clip_id, &self.asset.animation_clips) {
                let clip = clip.clone(); // clone to release the borrow on self.asset
                animation::sample_clip(
                    &clip,
                    self.animation_state.playhead_seconds,
                    &mut self.pose.local_transforms,
                );
            }
        }
    }

    pub fn compute_global_pose(&mut self) {
        let skeleton = &self.asset.skeleton;

        // Process root nodes first, then recurse into children.
        // We use an explicit stack to avoid borrowing issues with recursion.
        let mut stack: Vec<(usize, Option<usize>)> = Vec::new();

        // Push root nodes (no parent).
        for root_id in skeleton.root_nodes.iter().rev() {
            stack.push((root_id.0 as usize, None));
        }

        while let Some((node_idx, parent_idx)) = stack.pop() {
            let local_mat = self.pose.local_transforms[node_idx].to_matrix();

            let global_mat = match parent_idx {
                Some(pi) => mul_mat4(&self.pose.global_transforms[pi], &local_mat),
                None => local_mat,
            };

            self.pose.global_transforms[node_idx] = global_mat;

            // Push children in reverse order so they are processed in forward order.
            for child_id in skeleton.nodes[node_idx].children.iter().rev() {
                stack.push((child_id.0 as usize, Some(node_idx)));
            }
        }
    }

    pub fn build_skinning_matrices(&mut self) {
        let skeleton = &self.asset.skeleton;
        let node_count = skeleton.nodes.len();

        // Vertex joint indices are rewritten to be glTF node indices at load
        // time (see VrmAssetLoader::assign_skins_to_meshes). A single node-
        // indexed skinning array then serves every skin in the avatar,
        // regardless of whether the model ships one unified skin or separate
        // face/body/accessory skins.
        if self.pose.skinning_matrices.len() != node_count {
            self.pose.skinning_matrices = vec![crate::asset::identity_matrix(); node_count];
        }

        for i in 0..node_count {
            let ibm = if i < skeleton.inverse_bind_matrices.len() {
                &skeleton.inverse_bind_matrices[i]
            } else {
                continue;
            };
            let global = if i < self.pose.global_transforms.len() {
                &self.pose.global_transforms[i]
            } else {
                continue;
            };
            self.pose.skinning_matrices[i] = mul_mat4(global, ibm);
        }
    }
}

impl Default for SecondaryMotionState {
    fn default() -> Self {
        Self {
            spring_states: vec![],
            collision_cache: SecondaryMotionCollisionCache { contact_count: 0 },
        }
    }
}
