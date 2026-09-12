//! Compute prepass — slice 6 of the #12 renderer split. Split into a
//! prepare half ([`VulkanRenderer::prepare_compute_prepass`]) that performs
//! the per-primitive CPU writes (control UBOs, compacted morph weights,
//! cloth snapshot copies, cloth controls + pin targets, material
//! uniforms) and captures the dispatch structure, and a recording half
//! ([`VulkanRenderer::record_compute_prepass_planned`]) that replays that
//! structure into a command buffer. The split is what lets `render` reuse
//! a cached command buffer on unchanged frame shapes (see `frame_plan.rs`).
//! All state lives on `VulkanRenderer` (`transform_cache`,
//! `skinning_cache`, `material_uploader`, texture caches, GPU runtime
//! counters).

use std::collections::HashMap;
use std::sync::Arc;

use log::warn;

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::image::sampler::Sampler;
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint};

use crate::asset::PrimitiveId;
use crate::renderer::frame_input::{self, RenderFrameInput};
use crate::renderer::frame_plan::{
    DrawInfo, PlannedCloth, PlannedClothCollide, PlannedClothConstraints, PlannedClothNormal,
    PlannedClothSelfCol, PlannedInstance, PlannedPrim,
};
use crate::renderer::pipeline::{self, GpuVertex};
use crate::renderer::{mat4_cols_identity, VulkanRenderer, TRANSFORM_LOCAL_SIZE};

impl VulkanRenderer {
    /// Fuse skinning + morph + cloth into one dispatch per primitive
    /// (see `pipeline::transform_cs` for the shader-side contract) and
    /// build the draw list. Kahn-orders primitives with hierarchical
    /// surface-clearance dependencies so a child's transform dispatch
    /// reads its parent's freshly skinned vertices. Containment children
    /// (inner garment layers) read the parent's previous-frame VBO
    /// instead — their parent's own clearance dispatch runs later in
    /// the same order, so same-frame input is impossible without a
    /// second pass.
    ///
    /// This is the prepare half: every GPU-bound value reaches the GPU
    /// through a persistent buffer written here, and the dispatch
    /// structure is captured into [`PlannedInstance`]s for
    /// [`Self::record_compute_prepass_planned`].
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_compute_prepass(
        &mut self,
        input: &RenderFrameInput,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        ds_allocator: &Arc<StandardDescriptorSetAllocator>,
        transform_pipeline: &Arc<ComputePipeline>,
        gfx_pipeline: &Arc<GraphicsPipeline>,
        default_tex: &Arc<ImageView>,
        sampler: &Arc<Sampler>,
    ) -> Result<(Vec<PlannedInstance>, Vec<DrawInfo>), String> {
        let mut planned_instances: Vec<PlannedInstance> = Vec::new();
        let mut draws: Vec<DrawInfo> = Vec::new();

        for (inst_idx, instance) in input.instances.iter().enumerate() {
            let skinning_mats: Vec<[[f32; 4]; 4]> = if instance.skinning_matrices.is_empty() {
                vec![mat4_cols_identity()]
            } else {
                instance.skinning_matrices.to_vec()
            };
            let mat_count = skinning_mats.len();

            let skinning_set = self.get_or_update_skinning(
                inst_idx,
                &skinning_mats,
                mat_count,
                memory_allocator.clone(),
                ds_allocator.clone(),
                &transform_pipeline,
            )?;

            // Topological dependency ordering for hierarchical surface clearances:
            // If primitive B specifies body_primitive_id = Some(A) (clearance
            // anchors), A must be dispatched before B so B reads A's freshly
            // skinned vertices. Using Kahn's algorithm with cycle detection to
            // guarantee a valid dispatch order.
            //
            // Containment parents (`containment_primitive_id`, the outer
            // layer) deliberately do NOT become graph edges: their child's
            // clamp reads the parent's previous-frame VBO because the
            // parent's own clearance dispatch runs later in this order.
            // Adding containment edges would form a 2-cycle with the
            // clearance edge and the cycle fallback below would disable
            // BOTH constraints.
            let n = instance.mesh_instances.len();
            let mut in_degree = vec![0usize; n];
            let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

            // Build unique primitive_id mapping; detect duplicate primitive IDs if any.
            let mut prim_id_to_idx = HashMap::new();
            let mut duplicate_ids = std::collections::HashSet::new();
            for (idx, mi) in instance.mesh_instances.iter().enumerate() {
                if prim_id_to_idx.insert(mi.primitive_id, idx).is_some() {
                    duplicate_ids.insert(mi.primitive_id);
                    warn!(
                        "render: duplicate primitive_id {:?} in mesh instances",
                        mi.primitive_id
                    );
                }
            }

            // Track which primitives have a strictly validated parent dependency.
            // Invalid dependencies (self-reference, missing parent, or duplicate/ambiguous IDs) are pruned immediately.
            // `validated_parent_ids` covers the clearance edge; the
            // containment equivalent lives in `validated_containment_ids`.
            let mut validated_parent_ids: HashMap<PrimitiveId, PrimitiveId> = HashMap::new();
            let mut validated_containment_ids: HashMap<PrimitiveId, PrimitiveId> = HashMap::new();

            for (idx, mi) in instance.mesh_instances.iter().enumerate() {
                // If this primitive itself has a duplicate ID, it cannot be safely ordered; prune
                if duplicate_ids.contains(&mi.primitive_id) {
                    continue;
                }

                if let Some(parent_id) =
                    mi.primitive_data.as_ref().and_then(|p| p.body_primitive_id)
                {
                    // If the parent ID is ambiguous due to duplicates, children cannot resolve it uniquely; prune
                    if duplicate_ids.contains(&parent_id) {
                        warn!(
                    "render: parent primitive_id {:?} is ambiguous (duplicate IDs); pruning clearance on child {:?}",
                    parent_id, mi.primitive_id
                );
                        continue;
                    }

                    if let Some(&parent_idx) = prim_id_to_idx.get(&parent_id) {
                        if parent_idx != idx {
                            validated_parent_ids.insert(mi.primitive_id, parent_id);
                            adj[parent_idx].push(idx);
                            in_degree[idx] += 1;
                        } else {
                            warn!(
                                "render: self-referencing body_primitive_id {:?} pruned",
                                parent_id
                            );
                        }
                    } else {
                        warn!("render: missing parent body_primitive_id {:?} for primitive {:?}; fallback to unconstrained skinning", parent_id, mi.primitive_id);
                    }
                }

                if let Some(outer_id) = mi
                    .primitive_data
                    .as_ref()
                    .and_then(|p| p.containment_primitive_id)
                {
                    if duplicate_ids.contains(&outer_id) || outer_id == mi.primitive_id {
                        warn!(
                            "render: invalid containment_primitive_id {:?} on primitive {:?}; pruning",
                            outer_id, mi.primitive_id
                        );
                        continue;
                    }
                    if prim_id_to_idx.contains_key(&outer_id) {
                        validated_containment_ids.insert(mi.primitive_id, outer_id);
                    } else {
                        warn!(
                            "render: missing containment parent {:?} for primitive {:?}; clamping disabled",
                            outer_id, mi.primitive_id
                        );
                    }
                }
            }

            let mut queue: std::collections::VecDeque<usize> = in_degree
                .iter()
                .enumerate()
                .filter_map(|(idx, &deg)| if deg == 0 { Some(idx) } else { None })
                .collect();

            let mut ordered_mesh_instances = Vec::with_capacity(n);
            let mut visited = vec![false; n];

            while let Some(u) = queue.pop_front() {
                visited[u] = true;
                ordered_mesh_instances.push(&instance.mesh_instances[u]);
                for &v in &adj[u] {
                    in_degree[v] -= 1;
                    if in_degree[v] == 0 {
                        queue.push_back(v);
                    }
                }
            }

            // Fallback for cyclic dependencies or unvisited nodes:
            // Prune their parent dependencies to completely disable clearance constraints,
            // falling back safely to standard skinning, then append remaining instances.
            if ordered_mesh_instances.len() < n {
                warn!(
            "render: cycle detected in clearance graph ({} of {} resolved); disabling clearance on cyclic nodes and falling back to standard skinning",
            ordered_mesh_instances.len(), n
        );
                for (idx, mi) in instance.mesh_instances.iter().enumerate() {
                    if !visited[idx] {
                        validated_parent_ids.remove(&mi.primitive_id);
                        ordered_mesh_instances.push(mi);
                    }
                }
            }

            let mut planned_prims: Vec<PlannedPrim> = Vec::new();
            for &mesh_inst in &ordered_mesh_instances {
                let prim_asset = match mesh_inst.primitive_data.as_ref() {
                    Some(p) => p.as_ref(),
                    None => continue,
                };
                let vd = match prim_asset.vertices.as_ref() {
                    Some(vd) if !vd.positions.is_empty() => vd,
                    _ => continue,
                };
                let key = (mesh_inst.mesh_id, mesh_inst.primitive_id);

                // Per-primitive cloth scope: find the first snapshot that
                // targets *this* primitive. Other primitives in the same
                // instance get `has_cloth = false` and reuse the shared
                // stub SSBO — no more "body collapses to origin because
                // the cloth solver shipped a shorter vector" footgun.
                let cloth_snap_opt = instance
                    .cloth_deforms
                    .iter()
                    .find(|c| c.target_primitive_id == mesh_inst.primitive_id);
                let has_cloth_prim = cloth_snap_opt.is_some();
                // For Gpu-backed cloth with a triangulated mesh, the
                // normal compute dispatch (S3.1) writes `cloth_norm_ssbo`
                // even though the CPU snapshot's `deformed_normals` is
                // `None` — flip the flag so `transform_cs` reads the
                // GPU-written normals.
                let has_cloth_normals_prim = cloth_snap_opt
                    .map(|c| {
                        c.deformed_normals.is_some()
                            || (c.solver_backend
                                == crate::simulation::cloth_gpu_boundary::ClothSolverBackend::Gpu
                                && c.gpu_attach
                                    .as_ref()
                                    .map(|a| !a.triangle_indices.is_empty())
                                    .unwrap_or(false))
                    })
                    .unwrap_or(false);

                // Look up parent surface transformed VBO ONLY if this primitive has a validated DAG parent.
                // The parent slot is materialised WITHOUT anchors just to obtain a stable VBO handle;
                // an existing slot is left untouched — re-ensuring it with a `None` parent VBO would
                // downgrade its anchor binding every frame (the parent's own iteration below would
                // then have to rebuild it right back).
                let prim_parent_vbo = self.materialize_parent_vbo(
                    instance,
                    validated_parent_ids.get(&mesh_inst.primitive_id).copied(),
                    memory_allocator,
                    ds_allocator,
                    &transform_pipeline,
                );

                // Containment parent (the OUTER garment). Same
                // materialisation rules; no dispatch-ordering requirement
                // because the containment clamp reads the parent's
                // previous-frame VBO.
                let containment_parent_vbo = self.materialize_parent_vbo(
                    instance,
                    validated_containment_ids
                        .get(&mesh_inst.primitive_id)
                        .copied(),
                    memory_allocator,
                    ds_allocator,
                    &transform_pipeline,
                );

                self.ensure_transform_data(
                    mesh_inst.mesh_id,
                    mesh_inst.primitive_id,
                    prim_asset,
                    has_cloth_prim,
                    has_cloth_normals_prim,
                    prim_parent_vbo.clone(),
                    containment_parent_vbo.clone(),
                    memory_allocator,
                    ds_allocator,
                    &transform_pipeline,
                )?;

                // If the cloth snapshot reports GPU backend, lazily build
                // the per-primitive GPU cloth solver slot (seeding both
                // pos and prev_pos SSBOs from the rest pose). Dispatch
                // happens after the per-frame control UBO is rewritten
                // below.
                let cloth_is_gpu = cloth_snap_opt
                    .map(|c| {
                        c.solver_backend
                            == crate::simulation::cloth_gpu_boundary::ClothSolverBackend::Gpu
                    })
                    .unwrap_or(false);
                if let (true, Some(cloth_snap)) = (cloth_is_gpu, cloth_snap_opt) {
                    if let (
                        Some(attach),
                        Some(cloth_verlet_pipeline),
                        Some(cloth_constraint_lambda_update_pipeline),
                        Some(cloth_constraint_accumulate_pipeline),
                        Some(cloth_constraint_apply_pipeline),
                        Some(cloth_normal_pipeline),
                    ) = (
                        cloth_snap.gpu_attach.as_ref(),
                        self.cloth_verlet_pipeline.clone(),
                        self.cloth_constraint_lambda_update_pipeline.clone(),
                        self.cloth_constraint_accumulate_pipeline.clone(),
                        self.cloth_constraint_apply_pipeline.clone(),
                        self.cloth_normal_pipeline.clone(),
                    ) {
                        self.ensure_cloth_gpu_slot(
                            (mesh_inst.mesh_id, mesh_inst.primitive_id),
                            &cloth_snap.deformed_positions,
                            attach,
                            memory_allocator,
                            ds_allocator,
                            &cloth_verlet_pipeline,
                            &cloth_constraint_lambda_update_pipeline,
                            &cloth_constraint_accumulate_pipeline,
                            &cloth_constraint_apply_pipeline,
                            &cloth_normal_pipeline,
                        )?;
                    }
                }

                // Write the control UBO and the morph weights SSBO with
                // this frame's data. Live-write is safe because the
                // previous frame's fence has already been waited on at
                // the top of `render` (via `harvest_pending_readback`).
                //
                // The morph gather is COMPACTED to the active targets:
                // the shader's per-vertex loop bound is `target_count`,
                // and driving it at the full authored count (446 on
                // Yumeka's face) costs ~1.4 ms of GPU for iterations
                // that only load a zero weight and continue. The
                // entries buffer stays untouched — the per-target info
                // rows are absolute indices into it, so writing the
                // active rows into the head of `morph_infos` and the
                // matching weights into the head of `morph_weights`
                // gives the shader the same binary searches over a
                // shorter loop.
                let mut active_target_count = 0u32;
                {
                    let slot = self
                        .transform_cache
                        .get(&key)
                        .expect("ensure_transform_data populated the slot");
                    if slot.target_count > 0 {
                        let mut infos = slot
                            .morph_infos
                            .write()
                            .map_err(|e| format!("render: morph infos write failed: {e}"))?;
                        let mut weights = slot
                            .morph_weights_buf
                            .write()
                            .map_err(|e| format!("render: morph weights write failed: {e}"))?;
                        for (t, &w) in mesh_inst.morph_weights.iter().enumerate() {
                            if w.abs() <= 1e-6 {
                                continue;
                            }
                            let (Some(info_dst), Some(weight_dst)) = (
                                infos.get_mut(active_target_count as usize),
                                weights.get_mut(active_target_count as usize),
                            ) else {
                                break;
                            };
                            *info_dst = slot.morph_infos_full[t];
                            *weight_dst = w;
                            active_target_count += 1;
                        }
                    }
                    {
                        let mut guard = slot
                            .control_ubo
                            .write()
                            .map_err(|e| format!("render: control UBO write failed: {e}"))?;
                        guard.vertex_count = slot.vertex_count;
                        guard.target_count = active_target_count;
                        guard.has_cloth = if has_cloth_prim { 1 } else { 0 };
                        guard.has_cloth_normals = if has_cloth_normals_prim { 1 } else { 0 };
                        guard.has_skin_anchors =
                            if prim_asset.skin_anchors.is_some() && prim_parent_vbo.is_some() {
                                1
                            } else {
                                0
                            };
                        guard.has_containment = if prim_asset.containment_anchors.is_some()
                            && containment_parent_vbo.is_some()
                        {
                            1
                        } else {
                            0
                        };
                        guard._pad0 = [0; 2];
                    }
                }
                self.gpu_runtime_counters.morph_weight_writes += 1;

                // Cloth write: fill the SSBO with rest pose, then overlay
                // the snapshot's `[offset..offset+count)` slice. Vertices
                // outside the snapshot's vertex subset stay at rest pose
                // — they belong to the same primitive but are not part
                // of the cloth region.
                //
                // For `ClothSolverBackend::Gpu` cloths the GPU compute
                // dispatch (below) writes `cloth_pos_ssbo` in place; the
                // CPU snapshot copy is skipped so we don't clobber the
                // simulation state.
                let cloth_changed = if let (false, Some(cloth)) = (cloth_is_gpu, cloth_snap_opt) {
                    let slot = self.transform_cache.get_mut(&key).expect("slot present");
                    if slot.last_cloth_version != Some(cloth.version) {
                        let off = cloth.vertex_offset as usize;
                        let cnt = cloth.vertex_count as usize;
                        {
                            let mut pos_guard = slot
                                .cloth_pos_ssbo
                                .write()
                                .map_err(|e| format!("render: cloth pos write failed: {e}"))?;
                            for (i, dst) in pos_guard.iter_mut().enumerate() {
                                let p = if i >= off && i < off + cnt {
                                    cloth
                                        .deformed_positions
                                        .get(i - off)
                                        .copied()
                                        .unwrap_or_else(|| {
                                            vd.positions.get(i).copied().unwrap_or([0.0, 0.0, 0.0])
                                        })
                                } else {
                                    vd.positions.get(i).copied().unwrap_or([0.0, 0.0, 0.0])
                                };
                                *dst = [p[0], p[1], p[2], 1.0];
                            }
                        }
                        if let Some(ref normals) = cloth.deformed_normals {
                            let mut norm_guard = slot
                                .cloth_norm_ssbo
                                .write()
                                .map_err(|e| format!("render: cloth norm write failed: {e}"))?;
                            for (i, dst) in norm_guard.iter_mut().enumerate() {
                                let n = if i >= off && i < off + cnt {
                                    normals.get(i - off).copied().unwrap_or_else(|| {
                                        vd.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0])
                                    })
                                } else {
                                    vd.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0])
                                };
                                *dst = [n[0], n[1], n[2], 0.0];
                            }
                        }
                        slot.last_cloth_version = Some(cloth.version);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                if cloth_changed {
                    self.gpu_runtime_counters.cloth_vbo_writes += 1;
                }

                // GPU cloth Verlet integration structure + per-frame
                // control writes. Inter-dispatch synchronisation in the
                // recorded command buffer relies on Vulkano 0.35's
                // `AutoCommandBufferBuilder` resource-access tracking:
                // when two dispatches in the same command buffer share
                // a buffer with conflicting access (write→read or
                // write→write), the builder inserts the appropriate
                // `VkMemoryBarrier` at the bind-point boundary. This
                // covers verlet → constraint accumulate → constraint
                // apply → normal → transform_cs because each pair
                // shares `cloth_pos_ssbo` and/or `delta_ssbo` /
                // `cloth_norm_ssbo`. If the Vulkano version is bumped
                // and auto-sync semantics change, the recording half
                // needs to gain explicit `synchronization_pipeline_barrier`
                // calls.
                let mut cloth_plan: Option<PlannedCloth> = None;
                if let (true, Some(cloth)) = (cloth_is_gpu, cloth_snap_opt) {
                    if cloth.gpu_control.is_none() {
                        // Silent skip → renderer reads stale positions.
                        // Warn loudly so the misconfiguration is visible
                        // (typically: snapshot was collected without
                        // `ClothSimState`, so the collector returned
                        // `gpu_control: None` despite backend == Gpu).
                        warn!(
                            "render: cloth backend == Gpu but gpu_control \
                     missing for primitive {:?}; cloth_pos_ssbo will \
                     not advance this frame",
                            mesh_inst.primitive_id
                        );
                    }
                    if let Some(ctrl) = cloth.gpu_control.as_ref() {
                        let (verlet_set, particle_count) = {
                            let slot = self
                                .transform_cache
                                .get(&key)
                                .expect("transform slot present");
                            match slot.cloth_gpu.as_ref() {
                                Some(gpu) => (gpu.verlet_set.clone(), gpu.state.particle_count),
                                None => continue,
                            }
                        };
                        {
                            let slot = self
                                .transform_cache
                                .get_mut(&key)
                                .expect("transform slot present");
                            let gpu = slot
                                .cloth_gpu
                                .as_mut()
                                .expect("cloth_gpu present (just checked)");
                            let mut ctrl_guard = gpu
                                .verlet_control_ubo
                                .write()
                                .map_err(|e| format!("render: cloth verlet UBO write: {e}"))?;
                            *ctrl_guard = pipeline::ClothVerletControl {
                                dt: ctrl.dt,
                                damping: ctrl.damping,
                                particle_count,
                                _pad0: 0,
                                gravity: [ctrl.gravity[0], ctrl.gravity[1], ctrl.gravity[2], 0.0],
                                wind: [
                                    ctrl.wind_force[0],
                                    ctrl.wind_force[1],
                                    ctrl.wind_force[2],
                                    0.0,
                                ],
                            };
                            gpu.state.bump_version();
                        }
                        // Dynamic pins: overwrite pinned particles'
                        // positions AND previous positions with this
                        // frame's bone-following targets — the GPU
                        // twin of the CPU solver's `apply_pin_targets`.
                        // prev_pos is written too so verlet derives no
                        // velocity from the pin move. Host-visible
                        // SSBO write from the render thread, the same
                        // live-write pattern as the control UBO
                        // rewrite above; the verlet shader skips
                        // pinned/immobile particles entirely, and the
                        // constraint passes treat inv_mass == 0 as
                        // immovable, so the pin rows are only ever
                        // authored here.
                        if let Some(attach) = cloth.gpu_attach.as_ref() {
                            if !ctrl.pin_positions.is_empty() {
                                let (pos_buf, prev_buf) = {
                                    let slot = self
                                        .transform_cache
                                        .get(&key)
                                        .expect("transform slot present");
                                    let gpu = slot
                                        .cloth_gpu
                                        .as_ref()
                                        .expect("cloth_gpu present (dispatch branch)");
                                    (slot.cloth_pos_ssbo.clone(), gpu.prev_pos_ssbo.clone())
                                };
                                if let Err(e) = write_cloth_pin_targets(
                                    &pos_buf,
                                    &prev_buf,
                                    &attach.pinned,
                                    &ctrl.pin_positions,
                                ) {
                                    warn!(
                                        "render: cloth pin target write failed for primitive {:?}: {}",
                                        mesh_inst.primitive_id, e
                                    );
                                }
                            }
                        }
                        let groups = particle_count.div_ceil(TRANSFORM_LOCAL_SIZE);
                        // S2.1 — XPBD constraint projection resources.
                        //
                        // Upper clamps are defence-in-depth at the GPU
                        // boundary: `substeps` is already capped at 8 by
                        // `SimulationClock` and `solver_iterations` at 32
                        // by the cloth inspector's DragValue, but a
                        // hand-edited project file bypasses both. The
                        // dispatch count below is
                        // `substeps × (1 + 3 × constraint_iters) + 1` in
                        // ONE command buffer — unbounded values turn a
                        // frame into a GPU burst long enough to trip the
                        // driver watchdog (Intel Arc TDR history).
                        let constraint_iters = ctrl.solver_iterations.clamp(1, 32);
                        let substeps = ctrl.substeps.clamp(1, 8);
                        if ctrl.solver_iterations > 32 || ctrl.substeps > 8 {
                            warn!(
                        "render: cloth dispatch params clamped (substeps {} → {}, iterations {} → {})",
                        ctrl.substeps, substeps, ctrl.solver_iterations, constraint_iters
                    );
                        }
                        let constraint_resources = self
                            .transform_cache
                            .get(&key)
                            .and_then(|s| s.cloth_gpu.as_ref())
                            .and_then(|g| g.constraints.as_ref())
                            .map(|c| {
                                (
                                    c.lambda_update_set.clone(),
                                    c.accumulate_set.clone(),
                                    c.apply_set.clone(),
                                    c.control_ubo.clone(),
                                    c.lambda_ssbo.clone(),
                                    c.constraint_count,
                                )
                            });
                        // Write the constraint UBO once per frame —
                        // particle_count + constraint_count are slot-
                        // static, dt is the per-substep duration, and
                        // we run every substep with the same value.
                        // λ is reset per-substep via fill_buffer in the
                        // recording half.
                        let constraints_plan = if let Some((
                            lambda_update_set,
                            accumulate_set,
                            apply_set,
                            constraint_ctrl_ubo,
                            lambda_ssbo,
                            constraint_count,
                        )) = constraint_resources
                        {
                            {
                                let mut g = constraint_ctrl_ubo
                                    .write()
                                    .map_err(|e| format!("render: constraint UBO write: {e}"))?;
                                *g = pipeline::ClothConstraintControl {
                                    particle_count,
                                    constraint_count,
                                    dt: ctrl.dt,
                                    _pad: 0,
                                };
                            }
                            let constraint_groups = constraint_count.div_ceil(64).max(1);
                            Some(PlannedClothConstraints {
                                lambda_update_set,
                                accumulate_set,
                                apply_set,
                                lambda_ssbo,
                                constraint_count,
                                constraint_groups,
                            })
                        } else {
                            None
                        };

                        // S3.1 — vertex normal recomputation control.
                        let normal_resources = self
                            .transform_cache
                            .get(&key)
                            .and_then(|s| s.cloth_gpu.as_ref())
                            .and_then(|g| g.normals.as_ref())
                            .map(|n| (n.normal_set.clone(), n.control_ubo.clone()));
                        let normal_plan =
                            if let Some((normal_set, normal_ctrl_ubo)) = normal_resources {
                                {
                                    let mut g = normal_ctrl_ubo
                                        .write()
                                        .map_err(|e| format!("render: normal UBO write: {e}"))?;
                                    *g = pipeline::ClothNormalControl {
                                        vertex_count: particle_count,
                                        _pad0: 0,
                                        _pad1: 0,
                                        _pad2: 0,
                                    };
                                }
                                Some(PlannedClothNormal { set: normal_set })
                            } else {
                                None
                            };

                        // S2.2 — collision projection resources. Built
                        // only when the frame actually carries capsules
                        // (avatar-node colliders resolved by the
                        // snapshot collector); rewritten every frame.
                        let collide_plan = if !ctrl.colliders.is_empty() {
                            match self.cloth_collide_pipeline.as_ref() {
                                Some(collide_pipeline) => {
                                    let slot = self
                                        .transform_cache
                                        .get_mut(&key)
                                        .expect("transform slot present");
                                    match super::cloth_cache::ensure_cloth_gpu_collide_resources(
                                        slot,
                                        &ctrl.colliders,
                                        particle_count,
                                        ctrl.collision_margin,
                                        memory_allocator,
                                        ds_allocator,
                                        collide_pipeline,
                                    ) {
                                        Ok(()) => slot
                                            .cloth_gpu
                                            .as_ref()
                                            .and_then(|g| g.collide.as_ref())
                                            .map(|c| PlannedClothCollide {
                                                set: c.collide_set.clone(),
                                                collider_count: c.collider_count,
                                            }),
                                        Err(e) => {
                                            warn!(
                                                "render: cloth collide resources failed for primitive {:?}: {}",
                                                mesh_inst.primitive_id, e
                                            );
                                            None
                                        }
                                    }
                                }
                                None => None,
                            }
                        } else {
                            None
                        };

                        // S2.3 — self-collision resources (opt-in,
                        // mirrors `ClothSimState::self_collision`). The
                        // particle-neighbour CSR is attach-static; the
                        // control UBO's radius is per-frame.
                        let selfcol_plan = if ctrl.self_collision && ctrl.self_collision_radius > 0.0
                        {
                            match (
                                self.cloth_selfcol_build_pipeline.as_ref(),
                                self.cloth_selfcol_resolve_pipeline.as_ref(),
                                cloth.gpu_attach.as_ref().map(|a| a.constraints.clone()),
                            ) {
                                (Some(build_pl), Some(resolve_pl), Some(constraints)) => {
                                    let slot = self
                                        .transform_cache
                                        .get_mut(&key)
                                        .expect("transform slot present");
                                    match super::cloth_cache::ensure_cloth_gpu_selfcol_resources(
                                        slot,
                                        &constraints,
                                        particle_count,
                                        ctrl.self_collision_radius,
                                        memory_allocator,
                                        ds_allocator,
                                        build_pl,
                                        resolve_pl,
                                    ) {
                                        Ok(()) => slot
                                            .cloth_gpu
                                            .as_ref()
                                            .and_then(|g| g.selfcol.as_ref())
                                            .map(|sc| PlannedClothSelfCol {
                                                build_set: sc.build_set.clone(),
                                                resolve_set: sc.resolve_set.clone(),
                                                counts_ssbo: sc.cell_counts_ssbo.clone(),
                                            }),
                                        Err(e) => {
                                            warn!(
                                                "render: cloth selfcol resources failed for primitive {:?}: {}",
                                                mesh_inst.primitive_id, e
                                            );
                                            None
                                        }
                                    }
                                }
                                _ => None,
                            }
                        } else {
                            None
                        };

                        cloth_plan = Some(PlannedCloth {
                            verlet_set,
                            groups: [groups, 1, 1],
                            substeps,
                            constraint_iters,
                            constraints: constraints_plan,
                            normal: normal_plan,
                            collide: collide_plan,
                            selfcol: selfcol_plan,
                        });
                    }
                }

                // Transform dispatch inputs.
                let (transform_set, vertex_count, vbo, ibo, idx_count) = {
                    let slot = self.transform_cache.get(&key).expect("slot present");
                    (
                        slot.transform_set.clone(),
                        slot.vertex_count,
                        slot.transformed_vbo.clone(),
                        slot.index_buffer.clone(),
                        slot.index_count,
                    )
                };
                let transform_groups = vertex_count.div_ceil(TRANSFORM_LOCAL_SIZE);

                // Pick the graphics variant for the eventual draw.
                let is_blend = matches!(mesh_inst.alpha_mode, frame_input::RenderAlphaMode::Blend);
                let is_cutout =
                    matches!(mesh_inst.alpha_mode, frame_input::RenderAlphaMode::Cutout);
                let active_pipeline = if is_cutout {
                    // Cutout variants carry alpha-to-coverage (under MSAA) so
                    // alpha-tested edges antialias. They share the opaque
                    // variants' depth/blend state, so fall back to the opaque
                    // variant of the same cull mode if a cutout pipeline is
                    // somehow missing.
                    match &mesh_inst.cull_mode {
                        frame_input::RenderCullMode::DoubleSided => self
                            .pipeline_no_cull_cutout
                            .as_ref()
                            .or(self.pipeline_no_cull.as_ref())
                            .unwrap_or(&gfx_pipeline)
                            .clone(),
                        frame_input::RenderCullMode::FrontFace => self
                            .pipeline_front_cull_cutout
                            .as_ref()
                            .or(self.pipeline_front_cull.as_ref())
                            .unwrap_or(&gfx_pipeline)
                            .clone(),
                        frame_input::RenderCullMode::BackFace => self
                            .graphics_pipeline_cutout
                            .as_ref()
                            .unwrap_or(&gfx_pipeline)
                            .clone(),
                    }
                } else {
                    match (mesh_inst.cull_mode.clone(), is_blend) {
                        (frame_input::RenderCullMode::DoubleSided, true) => self
                            .pipeline_no_cull_blend
                            .as_ref()
                            .or(self.pipeline_no_cull.as_ref())
                            .unwrap_or(&gfx_pipeline)
                            .clone(),
                        (frame_input::RenderCullMode::FrontFace, true) => self
                            .pipeline_front_cull_blend
                            .as_ref()
                            .or(self.pipeline_front_cull.as_ref())
                            .unwrap_or(&gfx_pipeline)
                            .clone(),
                        (frame_input::RenderCullMode::BackFace, true) => self
                            .graphics_pipeline_blend
                            .as_ref()
                            .unwrap_or(&gfx_pipeline)
                            .clone(),
                        (frame_input::RenderCullMode::DoubleSided, false) => self
                            .pipeline_no_cull
                            .as_ref()
                            .unwrap_or(&gfx_pipeline)
                            .clone(),
                        (frame_input::RenderCullMode::FrontFace, false) => self
                            .pipeline_front_cull
                            .as_ref()
                            .unwrap_or(&gfx_pipeline)
                            .clone(),
                        (frame_input::RenderCullMode::BackFace, false) => gfx_pipeline.clone(),
                    }
                };

                // Material descriptor. Allocated against the canonical
                // `gfx_pipeline`; every variant has a structurally
                // identical set 1 layout (same shaders) so the binding
                // is valid under Vulkan descriptor set compatibility.
                let texture_view =
                    self.resolve_texture(&mesh_inst.material_binding.textures, &default_tex);
                let shade_texture_view =
                    self.resolve_shade_texture(&mesh_inst.material_binding.textures, &default_tex);
                let matcap_view =
                    self.resolve_matcap_texture(&mesh_inst.material_binding.textures, &default_tex);

                let material_set = self.material_uploader.upload_to_gpu(
                    Some((mesh_inst.mesh_id, mesh_inst.primitive_id)),
                    &mesh_inst.material_binding,
                    memory_allocator.clone(),
                    ds_allocator.clone(),
                    &gfx_pipeline,
                    texture_view,
                    shade_texture_view,
                    sampler.clone(),
                    matcap_view,
                )?;

                let outline_info = if !is_blend
                    && !is_cutout
                    && mesh_inst.outline.enabled
                    && mesh_inst.outline.width > 0.0
                {
                    Some((mesh_inst.outline.width, mesh_inst.outline.color))
                } else {
                    None
                };

                planned_prims.push(PlannedPrim {
                    transform_set,
                    groups: [transform_groups, 1, 1],
                    cloth: cloth_plan,
                });
                draws.push(DrawInfo {
                    pipeline: active_pipeline,
                    alpha_mode: mesh_inst.alpha_mode.clone(),
                    vertex_buffer: vbo,
                    index_buffer: ibo,
                    index_count: idx_count,
                    material_set,
                    outline: outline_info,
                });
            }
            planned_instances.push(PlannedInstance {
                skinning_set,
                prims: planned_prims,
            });
        }
        Ok((planned_instances, draws))
    }

    /// Recording half of the compute prepass: replay a prepared dispatch
    /// structure into `builder`. Pure mechanical translation of the
    /// [`PlannedInstance`]s — no renderer state is touched, which is what
    /// makes the output identical between a fresh recording and the
    /// structure a cached command buffer was built from.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_compute_prepass_planned(
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        transform_pipeline: &Arc<ComputePipeline>,
        cloth_verlet_pipeline: &Arc<ComputePipeline>,
        cloth_constraint_lambda_update_pipeline: &Arc<ComputePipeline>,
        cloth_constraint_accumulate_pipeline: &Arc<ComputePipeline>,
        cloth_constraint_apply_pipeline: &Arc<ComputePipeline>,
        cloth_normal_pipeline: &Arc<ComputePipeline>,
        cloth_collide_pipeline: &Arc<ComputePipeline>,
        cloth_selfcol_build_pipeline: &Arc<ComputePipeline>,
        cloth_selfcol_resolve_pipeline: &Arc<ComputePipeline>,
        instances: &[PlannedInstance],
    ) -> Result<(), String> {
        builder
            .bind_pipeline_compute(transform_pipeline.clone())
            .map_err(|e| format!("render: bind_pipeline_compute failed: {e}"))?;

        for inst in instances {
            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    transform_pipeline.layout().clone(),
                    1,
                    inst.skinning_set.clone(),
                )
                .map_err(|e| format!("render: bind compute skinning set failed: {e}"))?;

            for prim in &inst.prims {
                if let Some(cloth) = &prim.cloth {
                    // Substep loop: each substep advances Verlet
                    // integration by the fixed `ctrl.dt`, then runs
                    // `constraint_iters` XPBD constraint iterations.
                    // Matches the CPU path's
                    // `for _ in 0..substeps { step_cloth(fixed_dt) }`
                    // loop in `simulation::step_cloth_overlays`.
                    for _ in 0..cloth.substeps {
                        builder
                            .bind_pipeline_compute(cloth_verlet_pipeline.clone())
                            .map_err(|e| format!("render: bind cloth verlet pipeline: {e}"))?;
                        builder
                            .bind_descriptor_sets(
                                PipelineBindPoint::Compute,
                                cloth_verlet_pipeline.layout().clone(),
                                0,
                                cloth.verlet_set.clone(),
                            )
                            .map_err(|e| format!("render: bind cloth verlet set: {e}"))?;
                        unsafe {
                            builder
                                .dispatch(cloth.groups)
                                .map_err(|e| format!("render: cloth verlet dispatch: {e}"))?;
                        }

                        if let Some(cs) = &cloth.constraints {
                            // Reset λ for this substep. XPBD's λ
                            // accumulates across the projection
                            // iterations *within* one substep, then
                            // starts fresh at the next substep — same
                            // lifecycle as `ClothSimTempBuffers::reset_lambda`
                            // on the CPU side.
                            builder
                                .fill_buffer(cs.lambda_ssbo.clone().reinterpret::<[u32]>(), 0u32)
                                .map_err(|e| format!("render: lambda fill_buffer: {e}"))?;
                            for _ in 0..cloth.constraint_iters {
                                // Pass 1: per-constraint XPBD λ
                                // update writes Δλ_j to dlambda_ssbo
                                // and accumulates into lambda_ssbo.
                                builder
                                    .bind_pipeline_compute(
                                        cloth_constraint_lambda_update_pipeline.clone(),
                                    )
                                    .map_err(|e| {
                                        format!("render: bind constraint lambda update: {e}")
                                    })?;
                                builder
                                    .bind_descriptor_sets(
                                        PipelineBindPoint::Compute,
                                        cloth_constraint_lambda_update_pipeline.layout().clone(),
                                        0,
                                        cs.lambda_update_set.clone(),
                                    )
                                    .map_err(|e| {
                                        format!("render: bind constraint lambda update set: {e}")
                                    })?;
                                unsafe {
                                    builder.dispatch([cs.constraint_groups, 1, 1]).map_err(
                                        |e| {
                                            format!(
                                                "render: constraint lambda update dispatch: {e}"
                                            )
                                        },
                                    )?;
                                }
                                // Pass 2: per-particle Δx accumulate
                                // reads Δλ_j.
                                builder
                                    .bind_pipeline_compute(
                                        cloth_constraint_accumulate_pipeline.clone(),
                                    )
                                    .map_err(|e| {
                                        format!("render: bind constraint accumulate: {e}")
                                    })?;
                                builder
                                    .bind_descriptor_sets(
                                        PipelineBindPoint::Compute,
                                        cloth_constraint_accumulate_pipeline.layout().clone(),
                                        0,
                                        cs.accumulate_set.clone(),
                                    )
                                    .map_err(|e| {
                                        format!("render: bind constraint accumulate set: {e}")
                                    })?;
                                unsafe {
                                    builder.dispatch(cloth.groups).map_err(|e| {
                                        format!("render: constraint accumulate dispatch: {e}")
                                    })?;
                                }
                                // Pass 3: apply Δx to positions, zero
                                // deltas for next iter.
                                builder
                                    .bind_pipeline_compute(cloth_constraint_apply_pipeline.clone())
                                    .map_err(|e| format!("render: bind constraint apply: {e}"))?;
                                builder
                                    .bind_descriptor_sets(
                                        PipelineBindPoint::Compute,
                                        cloth_constraint_apply_pipeline.layout().clone(),
                                        0,
                                        cs.apply_set.clone(),
                                    )
                                    .map_err(|e| {
                                        format!("render: bind constraint apply set: {e}")
                                    })?;
                                unsafe {
                                    builder.dispatch(cloth.groups).map_err(|e| {
                                        format!("render: constraint apply dispatch: {e}")
                                    })?;
                                }
                            }
                        }
                    }

                    // S2.3 — self-collision (opt-in): rebuild the grid
                    // (zero + atomic fill) then resolve, once per
                    // substep, between the constraint iterations and
                    // the capsule projection — the CPU step order.
                    if let Some(selfcol) = &cloth.selfcol {
                        builder
                            .fill_buffer(
                                selfcol.counts_ssbo.clone().reinterpret::<[u32]>(),
                                0u32,
                            )
                            .map_err(|e| format!("render: selfcol counts fill: {e}"))?;
                        builder
                            .bind_pipeline_compute(cloth_selfcol_build_pipeline.clone())
                            .map_err(|e| format!("render: bind selfcol build: {e}"))?;
                        builder
                            .bind_descriptor_sets(
                                PipelineBindPoint::Compute,
                                cloth_selfcol_build_pipeline.layout().clone(),
                                0,
                                selfcol.build_set.clone(),
                            )
                            .map_err(|e| format!("render: bind selfcol build set: {e}"))?;
                        unsafe {
                            builder.dispatch(cloth.groups).map_err(|e| {
                                format!("render: selfcol build dispatch: {e}")
                            })?;
                        }
                        builder
                            .bind_pipeline_compute(cloth_selfcol_resolve_pipeline.clone())
                            .map_err(|e| format!("render: bind selfcol resolve: {e}"))?;
                        builder
                            .bind_descriptor_sets(
                                PipelineBindPoint::Compute,
                                cloth_selfcol_resolve_pipeline.layout().clone(),
                                0,
                                selfcol.resolve_set.clone(),
                            )
                            .map_err(|e| format!("render: bind selfcol resolve set: {e}"))?;
                        unsafe {
                            builder.dispatch(cloth.groups).map_err(|e| {
                                format!("render: selfcol resolve dispatch: {e}")
                            })?;
                        }
                    }

                    // S2.2 — collision projection: push particles out of
                    // the world-space capsules, once per substep after
                    // the constraint iterations (CPU step order: XPBD →
                    // self-collision → colliders; self-collision stays
                    // CPU-side). Pinned rows are skipped in-shader, so
                    // the pin targets authored in prepare survive. The
                    // verlet pipeline is re-bound afterwards because the
                    // next substep iteration starts from it.
                    if let Some(collide) = &cloth.collide {
                        builder
                            .bind_pipeline_compute(cloth_collide_pipeline.clone())
                            .map_err(|e| format!("render: bind cloth collide pipeline: {e}"))?;
                        builder
                            .bind_descriptor_sets(
                                PipelineBindPoint::Compute,
                                cloth_collide_pipeline.layout().clone(),
                                0,
                                collide.set.clone(),
                            )
                            .map_err(|e| format!("render: bind cloth collide set: {e}"))?;
                        unsafe {
                            builder.dispatch(cloth.groups).map_err(|e| {
                                format!("render: cloth collide dispatch: {e}")
                            })?;
                        }
                        builder
                            .bind_pipeline_compute(cloth_verlet_pipeline.clone())
                            .map_err(|e| format!("render: rebind cloth verlet pipeline: {e}"))?;
                    }

                    // S3.1 — vertex normal recomputation.
                    if let Some(normal) = &cloth.normal {
                        builder
                            .bind_pipeline_compute(cloth_normal_pipeline.clone())
                            .map_err(|e| format!("render: bind cloth normal pipeline: {e}"))?;
                        builder
                            .bind_descriptor_sets(
                                PipelineBindPoint::Compute,
                                cloth_normal_pipeline.layout().clone(),
                                0,
                                normal.set.clone(),
                            )
                            .map_err(|e| format!("render: bind cloth normal set: {e}"))?;
                        unsafe {
                            builder
                                .dispatch(cloth.groups)
                                .map_err(|e| format!("render: cloth normal dispatch: {e}"))?;
                        }
                    }

                    // Switch the bound compute pipeline back to the
                    // transform pipeline so the dispatch below uses the
                    // right shader. Two things must be re-bound, not
                    // just the pipeline: the cloth passes bound set 0
                    // under their own (incompatible) layouts, which
                    // drops the transform layout's set-1 skinning set
                    // from the command buffer's tracked bindings —
                    // without this re-bind the transform dispatch
                    // fails validation ("pipeline accesses descriptor
                    // set 1, but no descriptor set was previously
                    // bound"). First exercised by the GPU-cloth smoke
                    // in diagnose_cloth; the CPU path never dispatches
                    // cloth, so the transform set-1 binding from the
                    // instance loop above stayed live.
                    builder
                        .bind_pipeline_compute(transform_pipeline.clone())
                        .map_err(|e| format!("render: rebind transform pipeline: {e}"))?;
                    builder
                        .bind_descriptor_sets(
                            PipelineBindPoint::Compute,
                            transform_pipeline.layout().clone(),
                            1,
                            inst.skinning_set.clone(),
                        )
                        .map_err(|e| format!("render: rebind skinning set: {e}"))?;
                }

                // Bind set 0 + dispatch.
                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        transform_pipeline.layout().clone(),
                        0,
                        prim.transform_set.clone(),
                    )
                    .map_err(|e| format!("render: bind transform set 0 failed: {e}"))?;
                unsafe {
                    builder
                        .dispatch(prim.groups)
                        .map_err(|e| format!("render: dispatch failed: {e}"))?;
                }
            }
        }
        Ok(())
    }

    /// Resolve the transformed-VBO handle of a primitive's hierarchical
    /// anchor parent (clearance or containment). When the parent slot is
    /// missing it is materialised WITHOUT anchors just to obtain a
    /// stable handle; an existing slot is left untouched — re-ensuring
    /// it with a `None` parent VBO would downgrade its own anchor
    /// binding every frame (the parent's own iteration below would then
    /// have to rebuild it right back).
    #[allow(clippy::too_many_arguments)]
    fn materialize_parent_vbo(
        &mut self,
        instance: &frame_input::RenderAvatarInstance,
        parent_pid: Option<PrimitiveId>,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        ds_allocator: &Arc<StandardDescriptorSetAllocator>,
        transform_pipeline: &Arc<ComputePipeline>,
    ) -> Option<Subbuffer<[GpuVertex]>> {
        let parent_pid = parent_pid?;
        instance
            .mesh_instances
            .iter()
            .find(|mi| mi.primitive_id == parent_pid)
            .and_then(|parent_mi| {
                let parent_key = (parent_mi.mesh_id, parent_mi.primitive_id);
                if self.transform_cache.get(&parent_key).is_none() {
                    if let Some(parent_asset) = parent_mi.primitive_data.as_ref() {
                        let _ = self.ensure_transform_data(
                            parent_mi.mesh_id,
                            parent_mi.primitive_id,
                            parent_asset,
                            false,
                            false,
                            None,
                            None,
                            memory_allocator,
                            ds_allocator,
                            transform_pipeline,
                        );
                    }
                }
                self.transform_cache
                    .get(&parent_key)
                    .map(|slot| slot.transformed_vbo.clone())
            })
    }
}

/// Write this frame's pin world targets into the pinned particles'
/// rows of `cloth_pos_ssbo` / `prev_pos_ssbo` — see the call site for
/// why both rows are written. Separate function so both write guards
/// live in one scope with a single error path.
fn write_cloth_pin_targets(
    pos_buf: &vulkano::buffer::Subbuffer<[[f32; 4]]>,
    prev_buf: &vulkano::buffer::Subbuffer<[[f32; 4]]>,
    pinned: &[bool],
    targets: &[[f32; 3]],
) -> Result<(), String> {
    let mut pos_guard = pos_buf.write().map_err(|e| e.to_string())?;
    let mut prev_guard = prev_buf.write().map_err(|e| e.to_string())?;
    for ((dst_pos, dst_prev), (&is_pinned, target)) in pos_guard
        .iter_mut()
        .zip(prev_guard.iter_mut())
        .zip(pinned.iter().zip(targets.iter()))
    {
        if is_pinned {
            *dst_pos = [target[0], target[1], target[2], dst_pos[3]];
            *dst_prev = [target[0], target[1], target[2], dst_prev[3]];
        }
    }
    Ok(())
}
