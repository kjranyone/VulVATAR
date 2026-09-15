use std::sync::Arc;

use crate::asset::{
    AvatarAsset, AvatarAssetId, ClothAsset, ClothOverlayId, NodeId, Transform, Vec3,
};
use crate::avatar::animation::{self, AnimationState};
use crate::avatar::expressions::{ExpressionState, ResolvedExpressionWeight};
use crate::avatar::pose::{self as pose_helpers, AvatarPose};
use crate::simulation::cloth::{ClothSimState, ClothSimTempBuffers};
use crate::simulation::sdf::SdfField;

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
    /// Per-collider enable mask for cloth collision, indexed parallel to
    /// `asset.colliders`. The cloth-authoring inspector's "Collision
    /// Proxies" checkboxes bind here directly (single source of truth — no
    /// GUI shadow). `resolve_colliders` skips entries set to `false`.
    /// Runtime-only (not persisted); resets to all-enabled on (re)load.
    pub collider_enabled: Vec<bool>,
    /// Body-surface distance field (avatar-root space) read back from
    /// the renderer's splat dispatch, refreshed whenever the posed body
    /// changes — unchanged-pose frames reuse the previous field (see
    /// `app::render::splat_pose_key`). The spring solver resolves
    /// `body_collision` chains against it — see `simulation/sdf.rs`.
    /// `None` until the first field arrives.
    pub body_sdf: Option<SdfField>,
    pub expression_weights: Vec<ResolvedExpressionWeight>,
    /// Temporal state for the expression (blend-shape) solve.
    pub expression_state: ExpressionState,
    /// Tracking-v2 retarget state (rig-pose path, see `avatar::retarget`).
    pub retarget_state: crate::avatar::retarget::RetargetState,
    /// Eased transition into the procedural A-pose rest (see
    /// `avatar::relax`). Armed while tracking / an animation drives the
    /// pose; the first undriven frame captures the outgoing pose.
    pub relax: crate::avatar::relax::RelaxState,
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
    /// Last local rotation the spring solver wrote back, one entry per
    /// joint. Index `k` is the rotation of node `joints[k]`: the solver
    /// writes each segment's rotation into the segment's head node, i.e.
    /// `joints[j-1]` for solved joint `j`, so indices `0..len-1` are live
    /// and the final index is a seeded rest rotation. Seeded with rest
    /// rotations overall so [`AvatarInstance::reapply_spring_rotations`]
    /// is a no-op before the first solve.
    pub solved_rotations: Vec<[f32; 4]>,
    /// Settle-sleep bookkeeping (idle z-fight fix, 2026-09-15). The
    /// solver's residual per-step jitter is far below every diagnostic's
    /// resolution yet enough to flip depth ties between nearly-coincident
    /// surfaces — the breast garment sits ~1 mm off the skin, so
    /// micro-jitter makes the two flicker violently at idle. A chain
    /// whose joints stay under `SPRING_SLEEP_EPS` motion for
    /// `SPRING_SLEEP_QUIET_FRAMES` consecutive steps AND whose driving
    /// inputs (root world position, scene gravity) are bit-identical is
    /// skipped entirely, freezing its vertex stream to bit-stable.
    pub quiet_frames: u32,
    pub sleeping: bool,
    pub last_root_pos: Vec3,
    pub last_gravity: ([f32; 3], f32),
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
    /// Render target resolved from the attached `ClothAsset`'s first
    /// `ClothRenderRegionBinding`. Populated by `init_cloth_sim` /
    /// `init_cloth_overlay`; `None` until then or when the asset has
    /// no render bindings. The renderer skips contributing this cloth
    /// to any primitive while `target_primitive_id` is `None`.
    pub target_primitive_id: Option<crate::asset::PrimitiveId>,
    pub target_mesh_id: Option<crate::asset::MeshId>,
    pub target_vertex_offset: u32,
    pub target_vertex_count: u32,
    /// Selected solver path for this cloth. `Cpu` (default) runs the
    /// existing PBD solver on the avatar instance and copies snapshots
    /// into the renderer's cloth SSBO each frame. `Gpu` skips the CPU
    /// integration / constraint loop and lets the renderer dispatch
    /// the cloth compute pipelines, which write the SSBO in place.
    /// Set once per attach from
    /// [`cloth_gpu_boundary::solver_backend_from_env`]
    /// (`VULVATAR_CLOTH_GPU=1`); a `RuntimeGpuBudget`-driven default
    /// flip is future work (needs a GPU collider stage first).
    ///
    /// **Mid-session flips**: the renderer reads the GPU cloth state
    /// back into `deform_output` every frame (one frame of latency,
    /// fence-synchronised — see `RenderResult::cloth_readback`), so a
    /// `Gpu` → `Cpu` switch resumes the CPU solver from the live
    /// positions rather than the rest pose. Selection is still made
    /// once per attach; flipping mid-garment remains untested and is
    /// not exposed in the UI.
    pub solver_backend: crate::simulation::cloth_gpu_boundary::ClothSolverBackend,
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
        let collider_count = asset.colliders.len();
        let asset_id = asset.id;
        // Seed one zero-weight slot per named expression (VRM expressions
        // and FBX blend shapes alike) so the avatar-tab sliders exist from
        // load time — without this the list stayed empty until face
        // tracking ran, and FBX-only shape keys (body-size morphs) never
        // appeared because tracking has no counterpart for them.
        let expression_weights = asset
            .default_expressions
            .expressions
            .iter()
            .map(|e| ResolvedExpressionWeight {
                name: e.name.clone(),
                weight: 0.0,
            })
            .collect();

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
                    solved_rotations: spring
                        .joints
                        .iter()
                        .map(|j| {
                            asset
                                .skeleton
                                .nodes
                                .get(j.0 as usize)
                                .map(|n| n.rest_local.rotation)
                                .unwrap_or([0.0, 0.0, 0.0, 1.0])
                        })
                        .collect(),
                    quiet_frames: 0,
                    sleeping: false,
                    last_root_pos: [f32::NAN; 3],
                    last_gravity: ([f32::NAN; 3], f32::NAN),
                }
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
            collider_enabled: vec![true; collider_count],
            // Body-surface distance field delivered by the renderer's
            // readback one frame after each splat; `None` until the
            // first field arrives (spring bones simply don't collide
            // with the body yet).
            body_sdf: None,
            expression_weights,
            expression_state: ExpressionState::default(),
            retarget_state: Default::default(),
            relax: Default::default(),
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
            target_primitive_id: None,
            target_mesh_id: None,
            target_vertex_offset: 0,
            target_vertex_count: 0,
            solver_backend:
                crate::simulation::cloth_gpu_boundary::solver_backend_from_env(),
        });
    }

    /// Initialise the cloth simulation from a `ClothAsset`.
    /// Call this after `attach_cloth` to populate the PBD solver state.
    /// Also resolves the render target from `cloth_asset.render_bindings[0]`
    /// so the renderer can scope the deform output to a single primitive
    /// instead of broadcasting it across every primitive in the instance.
    pub fn init_cloth_sim(&mut self, cloth_asset: &ClothAsset) {
        let sim = ClothSimState::from_asset(cloth_asset);
        let n = sim.particle_count();

        // Pre-fill ClothState positions from the sim mesh rest pose
        if let Some(cs) = self.cloth_state.as_mut() {
            cs.sim_positions = sim.particles.iter().map(|p| p.position).collect();
            cs.prev_sim_positions = cs.sim_positions.clone();
            cs.deform_output.deformed_positions = cs.sim_positions.clone();
            apply_render_target(cs, cloth_asset);
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
            target_primitive_id: None,
            target_mesh_id: None,
            target_vertex_offset: 0,
            target_vertex_count: 0,
            solver_backend:
                crate::simulation::cloth_gpu_boundary::solver_backend_from_env(),
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
            apply_render_target(&mut slot.state, cloth_asset);
        }
        // A simulated garment owns its body interaction (solver capsule
        // collision). Render-only clearance anchors on the same
        // primitive fight that state instead of helping: the anchor
        // branch displaces DRAWN vertices without ever feeding the
        // correction back, so the skirt renders p95 15–18 mm away from
        // its simulated positions every frame (measured 2026-09-15,
        // `diagnostics/cloth_vboaudit_20260915`). Strip the target
        // primitive's clearance anchors; containment (if any) keeps
        // running. Load-time anchors on the asset are untouched — this
        // only CoWs the instance's view of the one primitive.
        if let Some(pid) = self
            .cloth_overlays
            .get(slot_index)
            .and_then(|s| s.state.target_primitive_id)
        {
            let asset = std::sync::Arc::make_mut(&mut self.asset);
            for mesh in asset.meshes.iter_mut() {
                for prim in mesh.primitives.iter_mut() {
                    if prim.id == pid && prim.skin_anchors.is_some() {
                        std::sync::Arc::make_mut(prim).skin_anchors = None;
                    }
                }
            }
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
        pose_helpers::compute_global_transforms(
            &self.asset.skeleton,
            &self.pose.local_transforms,
            &mut self.pose.global_transforms,
        );
    }

    /// Arm the rest relax for the next driver stop — call on frames
    /// where tracking or an animation produced the pose.
    pub fn relax_mark_driven(&mut self) {
        self.relax.mark_driven();
    }

    /// Capture the outgoing pose for the rest relax — call BEFORE
    /// [`Self::build_base_pose`] on frames where no driver produced the
    /// pose (the locals still hold the last solved pose there).
    pub fn relax_capture(&mut self) {
        self.relax.capture(&self.pose.local_transforms);
    }

    /// Ease the avatar into the procedural A-pose rest (see
    /// `avatar::relax`). Call AFTER [`Self::build_base_pose`] on frames
    /// where neither tracking nor an animation clip is driving the pose.
    /// Also seeds the retarget's display smoothing from the on-screen
    /// pose so the next tracked frame blends from here rather than from
    /// the bind rest.
    pub fn relax_apply(&mut self, dt: f32) {
        let humanoid = self.asset.humanoid.as_ref();
        crate::avatar::retarget::ensure_rest_cache(
            &mut self.retarget_state,
            &self.asset.skeleton,
            humanoid,
        );
        let mut target: Vec<Transform> = self
            .asset
            .skeleton
            .nodes
            .iter()
            .map(|n| n.rest_local.clone())
            .collect();
        for (_, (node, rot)) in self.retarget_state.arelax_local.iter() {
            if let Some(t) = target.get_mut(node.0 as usize) {
                t.rotation = *rot;
            }
        }
        self.relax
            .step_blend(dt, &target, &mut self.pose.local_transforms);
        if let Some(hm) = humanoid {
            self.retarget_state
                .seed_display_smoothing(hm, &self.pose.local_transforms);
        }
    }

    /// Resolve the per-primitive morph-target weight vector driven by the
    /// avatar's current [`Self::expression_weights`].
    ///
    /// Expressions bind morph targets by `(skeleton node, target index)`;
    /// a primitive's weight for target *t* is the sum of every bound
    /// expression's `weight × bind.weight` whose node maps to a mesh
    /// containing this primitive, clamped to `[0, 1]`. Returns an empty
    /// `Vec` for primitives without morph targets. Pure function of
    /// `(asset, expression_weights, prim)` — the render frame-input
    /// builder, thumbnail snapshots, and offline renderers all go
    /// through here so the mapping lives exactly once.
    pub fn morph_weights_for_prim(&self, prim: &crate::asset::MeshPrimitiveAsset) -> Vec<f32> {
        if prim.morph_targets.is_empty() {
            return Vec::new();
        }
        let mut weights = vec![0.0f32; prim.morph_targets.len()];
        for ew in &self.expression_weights {
            let Some(expr_def) = self
                .asset
                .default_expressions
                .expressions
                .iter()
                .find(|e| e.name == ew.name)
            else {
                continue;
            };
            for bind in &expr_def.morph_binds {
                if let Some(&mesh_idx) = self.asset.node_to_mesh.get(&bind.node_index) {
                    if let Some(m) = self.asset.meshes.get(mesh_idx) {
                        if m.primitives.iter().any(|p| p.id == prim.id)
                            && bind.morph_target_index < weights.len()
                        {
                            weights[bind.morph_target_index] += ew.weight * bind.weight;
                        }
                    }
                }
            }
        }
        for w in &mut weights {
            *w = w.clamp(0.0, 1.0);
        }
        weights
    }

    /// Re-apply the spring solver's last written rotations over the local
    /// transforms that [`Self::build_base_pose`] just reset to rest.
    ///
    /// The solver's writeback is the only thing that puts the solved
    /// rotations into `local_transforms`, so on frames where the
    /// fixed-step simulation clock yields zero substeps (fast or jittery
    /// frame pacing) the spring-driven meshes would otherwise render one
    /// frame in their rest pose — hair clipping into the head or body for
    /// exactly one frame. No-op before the first solve because the store
    /// is seeded with the rest rotations.
    ///
    /// The solver writes each segment's rotation into the segment's head
    /// node — `joints[j-1]` for solved joint `j` — so the applied range is
    /// nodes `0..len-1` of each chain.
    pub fn reapply_spring_rotations(&mut self) {
        for chain in &self.secondary_motion.spring_states {
            let written_nodes = chain.joints.len().saturating_sub(1);
            for k in 0..written_nodes {
                let (node, rotation) = match (chain.joints.get(k), chain.solved_rotations.get(k)) {
                    (Some(&node), Some(&rotation)) => (node, rotation),
                    _ => continue,
                };
                if let Some(t) = self.pose.local_transforms.get_mut(node.0 as usize) {
                    t.rotation = rotation;
                }
            }
        }
    }

    pub fn build_skinning_matrices(&mut self) {
        let skeleton = &self.asset.skeleton;
        let node_count = skeleton.nodes.len();

        // Vertex joint indices are rewritten to be glTF node indices at
        // load time (see VrmAssetLoader::assign_skins_to_meshes). A
        // single node-indexed skinning array then serves every skin in
        // the avatar, regardless of whether the model ships one unified
        // skin or separate face / body / accessory skins.
        if self.pose.skinning_matrices.len() != node_count {
            self.pose.skinning_matrices = vec![crate::asset::identity_matrix(); node_count];
        }
        pose_helpers::build_skinning_matrices(
            skeleton,
            &self.pose.global_transforms,
            &mut self.pose.skinning_matrices,
        );
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

/// Resolve the render target (`target_primitive_id`, mesh id, vertex
/// subset) from a `ClothAsset`'s first `ClothRenderRegionBinding` and
/// store it on `state`. Leaves the fields at their pre-existing values
/// when the asset has no render bindings — that case represents a cloth
/// that simulates but hasn't been pinned to a render target yet, and
/// the renderer's per-primitive `has_cloth` gate will skip applying
/// the deform to any draw.
fn apply_render_target(state: &mut ClothState, cloth_asset: &ClothAsset) {
    let Some(binding) = cloth_asset.render_bindings.first() else {
        return;
    };
    state.target_primitive_id = Some(binding.primitive.id);
    state.target_mesh_id = binding.mesh.as_ref().map(|m| m.id);
    state.target_vertex_offset = binding.vertex_subset.offset;
    state.target_vertex_count = binding.vertex_subset.count;
}

#[cfg(test)]
mod morph_weight_tests {
    use super::*;
    use crate::asset::{
        Aabb, AssetSourceHash, AvatarAsset, AvatarAssetId, ExpressionAssetSet, ExpressionDef,
        ExpressionMorphBind, MaterialId, MeshAsset, MeshId, MeshPrimitiveAsset, MorphTargetDelta,
        PrimitiveId, SkeletonAsset, SkeletonNode, VrmMeta,
    };
    use std::collections::HashMap;

    fn prim(id: u64, target_count: usize) -> std::sync::Arc<MeshPrimitiveAsset> {
        std::sync::Arc::new(MeshPrimitiveAsset {
            id: PrimitiveId(id),
            vertex_count: 3,
            index_count: 3,
            material_id: MaterialId(1),
            skin: None,
            bounds: Aabb::empty(),
            vertices: None,
            indices: None,
            morph_targets: (0..target_count)
                .map(|t| MorphTargetDelta {
                    name: format!("t{t}"),
                    position_deltas: vec![],
                    normal_deltas: vec![],
                })
                .collect(),
            skin_anchors: None,
            body_primitive_id: None,
            containment_anchors: None,
            containment_primitive_id: None,
        })
    }

    fn expr(name: &str, binds: Vec<(usize, usize, f32)>) -> ExpressionDef {
        ExpressionDef {
            name: name.to_string(),
            weight: 0.0,
            morph_binds: binds
                .into_iter()
                .map(
                    |(node_index, morph_target_index, weight)| ExpressionMorphBind {
                        node_index,
                        morph_target_index,
                        weight,
                    },
                )
                .collect(),
        }
    }

    fn avatar_with_face_prim() -> (AvatarInstance, std::sync::Arc<MeshPrimitiveAsset>) {
        let face = prim(1, 3);
        let plain = prim(2, 0);
        let mut node_to_mesh = HashMap::new();
        node_to_mesh.insert(0usize, 0usize);
        let asset = std::sync::Arc::new(AvatarAsset {
            id: AvatarAssetId(1),
            source_path: std::path::PathBuf::from("test.fbx"),
            source_hash: AssetSourceHash([0u8; 32]),
            skeleton: SkeletonAsset {
                root_nodes: vec![NodeId(0)],
                nodes: vec![SkeletonNode {
                    id: NodeId(0),
                    name: "face".to_string(),
                    parent: None,
                    children: vec![],
                    rest_local: Transform::default(),
                    humanoid_bone: None,
                }],
                inverse_bind_matrices: vec![],
            },
            meshes: vec![MeshAsset {
                id: MeshId(1),
                name: "face".to_string(),
                primitives: vec![face.clone(), plain],
            }],
            materials: vec![],
            humanoid: None,
            spring_bones: vec![],
            colliders: vec![],
            default_expressions: ExpressionAssetSet {
                expressions: vec![
                    expr("smile", vec![(0, 1, 1.0)]),
                    expr("half", vec![(0, 2, 0.5)]),
                    expr("over", vec![(0, 0, 1.0)]),
                    expr("over2", vec![(0, 0, 1.0)]),
                    // Out-of-range target index must be ignored, not panic.
                    expr("oob", vec![(0, 99, 1.0)]),
                ],
            },
            animation_clips: vec![],
            node_to_mesh,
            vrm_meta: VrmMeta::default(),
            root_aabb: Aabb::empty(),
            body_primitive_id: None,
            loaded_from_cache: false,
        });
        let mut avatar = AvatarInstance::new(AvatarInstanceId(1), asset);
        avatar.expression_weights = vec![
            crate::avatar::expressions::ResolvedExpressionWeight {
                name: "smile".into(),
                weight: 0.8,
            },
            crate::avatar::expressions::ResolvedExpressionWeight {
                name: "half".into(),
                weight: 1.0,
            },
            crate::avatar::expressions::ResolvedExpressionWeight {
                name: "over".into(),
                weight: 1.0,
            },
            crate::avatar::expressions::ResolvedExpressionWeight {
                name: "over2".into(),
                weight: 1.0,
            },
            crate::avatar::expressions::ResolvedExpressionWeight {
                name: "oob".into(),
                weight: 1.0,
            },
        ];
        (avatar, face)
    }

    #[test]
    fn morph_weights_sum_bind_weights_and_clamp() {
        let (avatar, face) = avatar_with_face_prim();
        // target 0: over + over2 = 2.0 → clamped to 1.0
        // target 1: smile × 1.0 = 0.8
        // target 2: half × 0.5 = 0.5
        assert_eq!(avatar.morph_weights_for_prim(&face), vec![1.0, 0.8, 0.5]);
    }

    #[test]
    fn morph_weights_empty_for_primitive_without_targets() {
        let (avatar, _) = avatar_with_face_prim();
        let plain = avatar.asset.meshes[0].primitives[1].clone();
        assert!(avatar.morph_weights_for_prim(&plain).is_empty());
    }

    #[test]
    fn morph_weights_zero_when_no_expression_matches() {
        let (mut avatar, face) = avatar_with_face_prim();
        avatar.expression_weights.clear();
        assert_eq!(avatar.morph_weights_for_prim(&face), vec![0.0, 0.0, 0.0]);
    }
}
