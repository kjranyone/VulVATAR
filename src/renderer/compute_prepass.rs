//! Compute prepass — slice 6 of the #12 renderer split. Records the
//! per-instance skinning + sparse-morph + cloth compute dispatches and
//! assembles the per-primitive [`DrawInfo`] list the scene pass
//! consumes. All state lives on `VulkanRenderer`
//! (`transform_cache`, `skinning_cache`, `material_uploader`,
//! texture caches, GPU runtime counters); `render` stays the
//! orchestrator that owns the command buffer.

use std::collections::HashMap;
use std::sync::Arc;

use log::warn;

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::DescriptorSet;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint};

use crate::asset::PrimitiveId;
use crate::renderer::frame_input::{self, RenderFrameInput};
use crate::renderer::pipeline::{self, GpuVertex};
use crate::renderer::{mat4_cols_identity, VulkanRenderer, TRANSFORM_LOCAL_SIZE};

// Per-primitive draw record built during the compute prepass and
// consumed by the graphics passes that follow. `vertex_buffer` is
// the compute shader's output, so the graphics passes never see
// base / morph / cloth data — only the world-space vertices.
pub(super) struct DrawInfo {
    pub(super) pipeline: Arc<GraphicsPipeline>,
    pub(super) alpha_mode: frame_input::RenderAlphaMode,
    pub(super) vertex_buffer: Subbuffer<[GpuVertex]>,
    pub(super) index_buffer: Subbuffer<[u32]>,
    pub(super) index_count: u32,
    pub(super) material_set: Arc<DescriptorSet>,
    pub(super) outline: Option<(f32, [f32; 3])>,
}

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
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_compute_prepass(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        input: &RenderFrameInput,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        ds_allocator: &Arc<StandardDescriptorSetAllocator>,
        transform_pipeline: &Arc<ComputePipeline>,
        gfx_pipeline: &Arc<GraphicsPipeline>,
        default_tex: &Arc<vulkano::image::view::ImageView>,
        sampler: &Arc<vulkano::image::sampler::Sampler>,
    ) -> Result<Vec<DrawInfo>, String> {
        let mut draws: Vec<DrawInfo> = Vec::new();
        // ── Compute prepass: fuse skinning + morph + cloth per primitive
        builder
            .bind_pipeline_compute(transform_pipeline.clone())
            .map_err(|e| format!("render: bind_pipeline_compute failed: {e}"))?;

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

            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    transform_pipeline.layout().clone(),
                    1,
                    skinning_set,
                )
                .map_err(|e| format!("render: bind compute skinning set failed: {e}"))?;

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
                    &memory_allocator,
                    &ds_allocator,
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
                    &memory_allocator,
                    &ds_allocator,
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
                    &memory_allocator,
                    &ds_allocator,
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
                            &memory_allocator,
                            &ds_allocator,
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
                {
                    let slot = self
                        .transform_cache
                        .get(&key)
                        .expect("ensure_transform_data populated the slot");
                    {
                        let mut guard = slot
                            .control_ubo
                            .write()
                            .map_err(|e| format!("render: control UBO write failed: {e}"))?;
                        guard.vertex_count = slot.vertex_count;
                        guard.target_count = slot.target_count;
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
                    if slot.target_count > 0 {
                        let mut weights = slot
                            .morph_weights_buf
                            .write()
                            .map_err(|e| format!("render: morph weights write failed: {e}"))?;
                        for (i, w) in weights.iter_mut().enumerate() {
                            *w = mesh_inst.morph_weights.get(i).copied().unwrap_or(0.0);
                        }
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

                // GPU cloth Verlet integration dispatch. Runs before
                // `transform_cs` reads `cloth_pos_ssbo`. Inter-dispatch
                // synchronisation here relies on Vulkano 0.35's
                // `AutoCommandBufferBuilder` resource-access tracking:
                // when two dispatches in the same command buffer share
                // a buffer with conflicting access (write→read or
                // write→write), the builder inserts the appropriate
                // `VkMemoryBarrier` at the bind-point boundary. This
                // covers verlet → constraint accumulate → constraint
                // apply → normal → transform_cs because each pair
                // shares `cloth_pos_ssbo` and/or `delta_ssbo` /
                // `cloth_norm_ssbo`. If the Vulkano version is bumped
                // and auto-sync semantics change, this code needs to
                // gain explicit `synchronization_pipeline_barrier`
                // calls.
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
                    if let (Some(ctrl), Some(cloth_verlet_pipeline)) = (
                        cloth.gpu_control.as_ref(),
                        self.cloth_verlet_pipeline.clone(),
                    ) {
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
                        // λ is reset per-substep via fill_buffer below.
                        let constraint_pack = if let (
                            Some((
                                lambda_update_set,
                                accumulate_set,
                                apply_set,
                                constraint_ctrl_ubo,
                                lambda_ssbo,
                                constraint_count,
                            )),
                            Some(lambda_update_pipeline),
                            Some(accumulate_pipeline),
                            Some(apply_pipeline),
                        ) = (
                            constraint_resources,
                            self.cloth_constraint_lambda_update_pipeline.clone(),
                            self.cloth_constraint_accumulate_pipeline.clone(),
                            self.cloth_constraint_apply_pipeline.clone(),
                        ) {
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
                            Some((
                                lambda_update_set,
                                accumulate_set,
                                apply_set,
                                lambda_ssbo,
                                constraint_count,
                                lambda_update_pipeline,
                                accumulate_pipeline,
                                apply_pipeline,
                            ))
                        } else {
                            None
                        };

                        // Substep loop: each substep advances Verlet
                        // integration by `ctrl.dt` (fixed_dt), then
                        // runs `constraint_iters` XPBD constraint
                        // iterations. Matches the CPU path's
                        // `for _ in 0..substeps { step_cloth(fixed_dt) }`
                        // loop in `simulation::step_cloth_overlays`.
                        // Before this loop the GPU dispatched the
                        // whole thing once at frame_dt, integrating
                        // gravity·dt² with `substeps²` more energy and
                        // making α̃ = α/dt² `substeps²` smaller — CPU
                        // and GPU produced qualitatively different
                        // cloth physics.
                        for _ in 0..substeps {
                            builder
                                .bind_pipeline_compute(cloth_verlet_pipeline.clone())
                                .map_err(|e| format!("render: bind cloth verlet pipeline: {e}"))?;
                            builder
                                .bind_descriptor_sets(
                                    PipelineBindPoint::Compute,
                                    cloth_verlet_pipeline.layout().clone(),
                                    0,
                                    verlet_set.clone(),
                                )
                                .map_err(|e| format!("render: bind cloth verlet set: {e}"))?;
                            unsafe {
                                builder
                                    .dispatch([groups, 1, 1])
                                    .map_err(|e| format!("render: cloth verlet dispatch: {e}"))?;
                            }

                            if let Some((
                                lambda_update_set,
                                accumulate_set,
                                apply_set,
                                lambda_ssbo,
                                constraint_count,
                                lambda_update_pipeline,
                                accumulate_pipeline,
                                apply_pipeline,
                            )) = constraint_pack.as_ref()
                            {
                                // Reset λ for this substep. XPBD's λ
                                // accumulates across the projection
                                // iterations *within* one substep,
                                // then starts fresh at the next
                                // substep — same lifecycle as
                                // `ClothSimTempBuffers::reset_lambda`
                                // on the CPU side.
                                builder
                                    .fill_buffer(lambda_ssbo.clone().reinterpret::<[u32]>(), 0u32)
                                    .map_err(|e| format!("render: lambda fill_buffer: {e}"))?;
                                let constraint_groups = constraint_count.div_ceil(64).max(1);
                                for _ in 0..constraint_iters {
                                    // Pass 1: per-constraint XPBD λ
                                    // update writes Δλ_j to
                                    // dlambda_ssbo and accumulates
                                    // into lambda_ssbo.
                                    builder
                                        .bind_pipeline_compute(lambda_update_pipeline.clone())
                                        .map_err(|e| {
                                            format!("render: bind constraint lambda update: {e}")
                                        })?;
                                    builder
                                        .bind_descriptor_sets(
                                            PipelineBindPoint::Compute,
                                            lambda_update_pipeline.layout().clone(),
                                            0,
                                            lambda_update_set.clone(),
                                        )
                                        .map_err(|e| {
                                            format!(
                                                "render: bind constraint lambda update set: {e}"
                                            )
                                        })?;
                                    unsafe {
                                        builder.dispatch([constraint_groups, 1, 1]).map_err(
                                    |e| {
                                        format!("render: constraint lambda update dispatch: {e}")
                                    },
                                )?;
                                    }
                                    // Pass 2: per-particle Δx
                                    // accumulate reads Δλ_j.
                                    builder
                                        .bind_pipeline_compute(accumulate_pipeline.clone())
                                        .map_err(|e| {
                                            format!("render: bind constraint accumulate: {e}")
                                        })?;
                                    builder
                                        .bind_descriptor_sets(
                                            PipelineBindPoint::Compute,
                                            accumulate_pipeline.layout().clone(),
                                            0,
                                            accumulate_set.clone(),
                                        )
                                        .map_err(|e| {
                                            format!("render: bind constraint accumulate set: {e}")
                                        })?;
                                    unsafe {
                                        builder.dispatch([groups, 1, 1]).map_err(|e| {
                                            format!("render: constraint accumulate dispatch: {e}")
                                        })?;
                                    }
                                    // Pass 3: apply Δx to positions,
                                    // zero deltas for next iter.
                                    builder
                                        .bind_pipeline_compute(apply_pipeline.clone())
                                        .map_err(|e| {
                                            format!("render: bind constraint apply: {e}")
                                        })?;
                                    builder
                                        .bind_descriptor_sets(
                                            PipelineBindPoint::Compute,
                                            apply_pipeline.layout().clone(),
                                            0,
                                            apply_set.clone(),
                                        )
                                        .map_err(|e| {
                                            format!("render: bind constraint apply set: {e}")
                                        })?;
                                    unsafe {
                                        builder.dispatch([groups, 1, 1]).map_err(|e| {
                                            format!("render: constraint apply dispatch: {e}")
                                        })?;
                                    }
                                }
                            }
                        }

                        // S3.1 — vertex normal recomputation.
                        let normal_resources = self
                            .transform_cache
                            .get(&key)
                            .and_then(|s| s.cloth_gpu.as_ref())
                            .and_then(|g| g.normals.as_ref())
                            .map(|n| (n.normal_set.clone(), n.control_ubo.clone()));
                        if let (Some((normal_set, normal_ctrl_ubo)), Some(normal_pipeline)) =
                            (normal_resources, self.cloth_normal_pipeline.clone())
                        {
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
                            builder
                                .bind_pipeline_compute(normal_pipeline.clone())
                                .map_err(|e| format!("render: bind cloth normal pipeline: {e}"))?;
                            builder
                                .bind_descriptor_sets(
                                    PipelineBindPoint::Compute,
                                    normal_pipeline.layout().clone(),
                                    0,
                                    normal_set,
                                )
                                .map_err(|e| format!("render: bind cloth normal set: {e}"))?;
                            unsafe {
                                builder
                                    .dispatch([groups, 1, 1])
                                    .map_err(|e| format!("render: cloth normal dispatch: {e}"))?;
                            }
                        }

                        // Switch the bound compute pipeline back to the
                        // transform pipeline so the dispatch below uses
                        // the right shader. (The descriptor set bound
                        // afterwards targets a different layout, so an
                        // explicit re-bind here is required.)
                        builder
                            .bind_pipeline_compute(transform_pipeline.clone())
                            .map_err(|e| format!("render: rebind transform pipeline: {e}"))?;
                    }
                }

                // Bind set 0 + dispatch.
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
                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        transform_pipeline.layout().clone(),
                        0,
                        transform_set,
                    )
                    .map_err(|e| format!("render: bind transform set 0 failed: {e}"))?;
                let groups = vertex_count.div_ceil(TRANSFORM_LOCAL_SIZE);
                unsafe {
                    builder
                        .dispatch([groups, 1, 1])
                        .map_err(|e| format!("render: dispatch failed: {e}"))?;
                }

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
        }
        Ok(draws)
    }

    /// Resolve the transformed-VBO handle of a primitive's hierarchical
    /// anchor parent (clearance or containment). When the parent slot is
    /// missing it is materialised WITHOUT anchors just to obtain a
    /// stable handle; an existing slot is left untouched — re-ensuring
    /// it with a `None` parent VBO would downgrade its own anchor
    /// binding every frame (the parent's iteration below would then
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
