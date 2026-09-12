//! Cloth GPU solver resource allocation — extracted from
//! `VulkanRenderer` as the third slice of the #12 renderer split.
//! Owns the per-primitive cloth slot construction (Verlet integrator
//! state, optional XPBD constraint resources, optional normal-
//! recomputation resources) used by the cloth compute dispatch
//! pipeline. The render-loop dispatch wiring still lives in `mod.rs`
//! next to the rest of the per-frame command-buffer construction;
//! only the slot-creation path moved.

use std::sync::Arc;

use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{ComputePipeline, Pipeline};

use crate::asset::{MeshId, PrimitiveId};
use crate::renderer::gpu_alloc;
use crate::renderer::{
    pipeline, ClothGpuCollideResources, ClothGpuConstraintResources, ClothGpuNormalResources,
    ClothGpuSelfColResources, ClothGpuSlot, ClothReadback, VulkanRenderer,
};

impl VulkanRenderer {
    /// Allocate the GPU cloth solver resources for a primitive whose
    /// snapshot reported `ClothSolverBackend::Gpu`. No-op if the slot
    /// already exists or if `initial_positions` is empty.
    ///
    /// Lifecycle:
    /// - Verlet resources (`prev_pos_ssbo`, `verlet_control_ubo`,
    ///   `verlet_set`) are always allocated.
    /// - Constraint resources are allocated only when
    ///   `attach.constraints` is non-empty (built once, read-only).
    /// - Normal-recomputation resources are allocated only when
    ///   `attach.triangle_indices` is non-empty (built once, read-only).
    ///
    /// On creation, `cloth_pos_ssbo` and `prev_pos_ssbo` are seeded
    /// from `initial_positions` × `attach.inv_masses` / `attach.pinned`
    /// so the very first dispatch reads valid data and pinned particles
    /// stay put.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ensure_cloth_gpu_slot(
        &mut self,
        key: (MeshId, PrimitiveId),
        initial_positions: &[crate::asset::Vec3],
        attach: &crate::renderer::frame_input::ClothGpuAttachData,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        ds_allocator: &Arc<StandardDescriptorSetAllocator>,
        cloth_verlet_pipeline: &Arc<ComputePipeline>,
        cloth_constraint_lambda_update_pipeline: &Arc<ComputePipeline>,
        cloth_constraint_accumulate_pipeline: &Arc<ComputePipeline>,
        cloth_constraint_apply_pipeline: &Arc<ComputePipeline>,
        cloth_normal_pipeline: &Arc<ComputePipeline>,
    ) -> Result<(), String> {
        let particle_count = initial_positions.len() as u32;
        let constraint_count = attach.constraints.len() as u32;

        // Compare against the cached slot's recorded counts. If either
        // the particle count or constraint count changed (e.g. the user
        // swapped to a different cloth asset on the same primitive
        // without toggling `has_cloth`), drop the stale slot so it
        // gets rebuilt at the new sizes; SSBOs allocated for the old
        // counts would silently mis-size or overflow.
        let needs_rebuild = self
            .transform_cache
            .get(&key)
            .and_then(|s| s.cloth_gpu.as_ref())
            .map(|gpu| {
                gpu.state.particle_count != particle_count
                    || gpu.state.constraint_count != constraint_count
            })
            .unwrap_or(false);
        if needs_rebuild {
            if let Some(slot) = self.transform_cache.get_mut(&key) {
                slot.cloth_gpu = None;
                // Cloth descriptor sets are about to be reallocated —
                // cached command buffers binding the old sets must go.
                self.cb_cache.clear();
            }
        }

        let already_alloc = self
            .transform_cache
            .get(&key)
            .map(|s| s.cloth_gpu.is_some())
            .unwrap_or(true);
        if already_alloc || initial_positions.is_empty() {
            return Ok(());
        }
        // First allocation of this slot's cloth descriptor sets — same
        // invalidation rule as the rebuild branch above.
        self.cb_cache.clear();
        let inv_mass_at = |i: usize| attach.inv_masses.get(i).copied().unwrap_or(1.0);
        let pinned_at = |i: usize| {
            if attach.pinned.get(i).copied().unwrap_or(false) {
                1.0_f32
            } else {
                0.0
            }
        };

        let prev_pos_ssbo = gpu_alloc::host_buffer(
            &memory_allocator,
            BufferUsage::STORAGE_BUFFER,
            // Seed prev_pos to same as initial positions ⇒ zero
            // velocity at start. `w` = pinned flag (1.0 = locked).
            initial_positions
                .iter()
                .enumerate()
                .map(|(i, p)| [p[0], p[1], p[2], pinned_at(i)]),
            "cloth prev_pos SSBO",
        )?;

        let verlet_control_ubo = gpu_alloc::host_ubo(
            &memory_allocator,
            pipeline::ClothVerletControl {
                dt: 0.0,
                damping: 0.0,
                particle_count,
                _pad0: 0,
                gravity: [0.0; 4],
                wind: [0.0; 4],
            },
            "cloth verlet control UBO",
        )?;

        let set0_layout = cloth_verlet_pipeline
            .layout()
            .set_layouts()
            .first()
            .ok_or("renderer: cloth verlet pipeline missing set 0 layout")?
            .clone();

        // Held only to build the descriptor set; the primitive's
        // `cloth_pos_ssbo` is owned by `TransformGpuData`.
        let cloth_pos_ssbo = self
            .transform_cache
            .get(&key)
            .ok_or("renderer: ensure_cloth_gpu_slot before transform slot exists")?
            .cloth_pos_ssbo
            .clone();

        // Seed cloth_pos_ssbo with the rest pose so the very first GPU
        // dispatch reads valid positions. `w` carries the inverse mass
        // (0.0 = effectively pinned / immobile per the shader).
        {
            let mut guard = cloth_pos_ssbo
                .write()
                .map_err(|e| format!("renderer: cloth pos initial seed failed: {e}"))?;
            for (i, (dst, p)) in guard.iter_mut().zip(initial_positions.iter()).enumerate() {
                *dst = [p[0], p[1], p[2], inv_mass_at(i)];
            }
        }

        let verlet_set = DescriptorSet::new(
            ds_allocator.clone(),
            set0_layout,
            [
                WriteDescriptorSet::buffer(0, cloth_pos_ssbo.clone()),
                WriteDescriptorSet::buffer(1, prev_pos_ssbo.clone()),
                WriteDescriptorSet::buffer(2, verlet_control_ubo.clone()),
            ],
            [],
        )
        .map_err(|e| format!("renderer: cloth verlet descriptor set: {e}"))?;

        // ---------- S2.1 — constraint resources (optional) ----------
        let constraints = if !attach.constraints.is_empty() {
            Some(allocate_cloth_constraint_resources(
                &attach.constraints,
                particle_count,
                &cloth_pos_ssbo,
                memory_allocator,
                ds_allocator,
                cloth_constraint_lambda_update_pipeline,
                cloth_constraint_accumulate_pipeline,
                cloth_constraint_apply_pipeline,
            )?)
        } else {
            None
        };

        // ---------- S3.1 — normal recomputation resources (optional) ----------
        let cloth_norm_ssbo = self
            .transform_cache
            .get(&key)
            .expect("transform slot exists")
            .cloth_norm_ssbo
            .clone();
        let normals = if !attach.triangle_indices.is_empty() {
            Some(allocate_cloth_normal_resources(
                &attach.triangle_indices,
                particle_count,
                &cloth_pos_ssbo,
                &cloth_norm_ssbo,
                memory_allocator,
                ds_allocator,
                cloth_normal_pipeline,
            )?)
        } else {
            None
        };

        if let Some(slot) = self.transform_cache.get_mut(&key) {
            slot.cloth_gpu = Some(ClothGpuSlot {
                state:
                    crate::simulation::cloth_gpu_boundary::ClothGpuSimulationState::from_authoring(
                        particle_count,
                        attach.constraints.len() as u32,
                        8,
                    ),
                prev_pos_ssbo,
                verlet_control_ubo,
                verlet_set,
                constraints,
                normals,
                collide: None,
                selfcol: None,
            });
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn allocate_cloth_constraint_resources(
    constraints: &[(u32, u32, f32, f32)],
    particle_count: u32,
    cloth_pos_ssbo: &Subbuffer<[[f32; 4]]>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    ds_allocator: &Arc<StandardDescriptorSetAllocator>,
    lambda_update_pipeline: &Arc<ComputePipeline>,
    accumulate_pipeline: &Arc<ComputePipeline>,
    apply_pipeline: &Arc<ComputePipeline>,
) -> Result<ClothGpuConstraintResources, String> {
    let constraint_ssbo = gpu_alloc::host_buffer(
        &memory_allocator,
        BufferUsage::STORAGE_BUFFER,
        constraints
            .iter()
            .map(|(a, b, r, s)| pipeline::ClothConstraintGpu {
                particle_a: *a,
                particle_b: *b,
                rest_length: *r,
                stiffness: *s,
            }),
        "cloth constraint SSBO",
    )?;

    let delta_ssbo = gpu_alloc::host_buffer(
        &memory_allocator,
        BufferUsage::STORAGE_BUFFER,
        (0..particle_count as usize).map(|_| [0.0_f32; 4]),
        "cloth delta SSBO",
    )?;

    let constraint_pairs: Vec<(u32, u32)> =
        constraints.iter().map(|(a, b, _, _)| (*a, *b)).collect();
    let adj = crate::simulation::cloth_gpu_boundary::build_particle_constraint_adjacency(
        &constraint_pairs,
        particle_count,
    );
    let adj_offsets_ssbo = gpu_alloc::host_buffer(
        &memory_allocator,
        BufferUsage::STORAGE_BUFFER,
        adj.offsets.iter().copied(),
        "constraint adj offsets SSBO",
    )?;
    // Vulkano refuses zero-sized buffers; supply a single u32 stub
    // when the CSR scatter array is empty (no particle touches any
    // constraint). The shader's adjacency loop won't iterate
    // because every `offsets[v+1] == offsets[v]` in that case.
    let adj_constraints_data: Vec<u32> = if adj.triangles.is_empty() {
        vec![0_u32]
    } else {
        adj.triangles.clone()
    };
    let adj_constraints_ssbo = gpu_alloc::host_buffer(
        &memory_allocator,
        BufferUsage::STORAGE_BUFFER,
        adj_constraints_data,
        "constraint adj scatter SSBO",
    )?;

    let control_ubo = gpu_alloc::host_ubo(
        &memory_allocator,
        pipeline::ClothConstraintControl {
            particle_count,
            // `constraint_count` and `dt` are populated by the render-
            // loop's per-frame UBO rewrite (constraint_count is stable
            // per slot; dt comes from the substep duration). Seed with
            // zero — the very first frame's lambda-update dispatch
            // sees an `α̃ = 0 / 0` early-return because dt² is guarded.
            constraint_count: constraints.len() as u32,
            dt: 0.0,
            _pad: 0,
        },
        "constraint control UBO",
    )?;

    // XPBD per-constraint Lagrange multiplier (accumulates across the
    // projection iterations within one substep; reset to zero at
    // substep start by the renderer's `fill_buffer`).
    let lambda_ssbo: Subbuffer<[f32]> = gpu_alloc::host_buffer(
        &memory_allocator,
        BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
        // Vulkano refuses zero-sized buffers — supply a single 0.0
        // stub when there are no constraints. Won't be read because
        // `constraint_count = 0` short-circuits the lambda-update
        // dispatch's per-invocation work.
        if constraints.is_empty() {
            vec![0.0_f32]
        } else {
            vec![0.0_f32; constraints.len()]
        },
        "cloth lambda SSBO",
    )?;

    // Per-iteration Δλ_j scratch buffer. Written by the
    // lambda-update pass each iteration, read by the accumulate pass;
    // contents are overwritten so no initial state matters.
    let dlambda_ssbo: Subbuffer<[f32]> = gpu_alloc::host_buffer(
        &memory_allocator,
        BufferUsage::STORAGE_BUFFER,
        if constraints.is_empty() {
            vec![0.0_f32]
        } else {
            vec![0.0_f32; constraints.len()]
        },
        "cloth dlambda SSBO",
    )?;

    let lambda_update_layout = lambda_update_pipeline
        .layout()
        .set_layouts()
        .first()
        .ok_or("renderer: constraint lambda-update pipeline missing set 0")?
        .clone();
    let lambda_update_set = DescriptorSet::new(
        ds_allocator.clone(),
        lambda_update_layout,
        [
            WriteDescriptorSet::buffer(0, cloth_pos_ssbo.clone()),
            WriteDescriptorSet::buffer(1, constraint_ssbo.clone()),
            WriteDescriptorSet::buffer(2, control_ubo.clone()),
            WriteDescriptorSet::buffer(3, lambda_ssbo.clone()),
            WriteDescriptorSet::buffer(4, dlambda_ssbo.clone()),
        ],
        [],
    )
    .map_err(|e| format!("renderer: constraint lambda-update descriptor set: {e}"))?;

    let accumulate_layout = accumulate_pipeline
        .layout()
        .set_layouts()
        .first()
        .ok_or("renderer: constraint accumulate pipeline missing set 0")?
        .clone();
    let accumulate_set = DescriptorSet::new(
        ds_allocator.clone(),
        accumulate_layout,
        [
            WriteDescriptorSet::buffer(0, cloth_pos_ssbo.clone()),
            WriteDescriptorSet::buffer(1, delta_ssbo.clone()),
            WriteDescriptorSet::buffer(2, constraint_ssbo.clone()),
            WriteDescriptorSet::buffer(3, adj_offsets_ssbo.clone()),
            WriteDescriptorSet::buffer(4, adj_constraints_ssbo.clone()),
            WriteDescriptorSet::buffer(5, control_ubo.clone()),
            WriteDescriptorSet::buffer(6, dlambda_ssbo.clone()),
        ],
        [],
    )
    .map_err(|e| format!("renderer: constraint accumulate descriptor set: {e}"))?;

    let apply_layout = apply_pipeline
        .layout()
        .set_layouts()
        .first()
        .ok_or("renderer: constraint apply pipeline missing set 0")?
        .clone();
    let apply_set = DescriptorSet::new(
        ds_allocator.clone(),
        apply_layout,
        [
            WriteDescriptorSet::buffer(0, cloth_pos_ssbo.clone()),
            WriteDescriptorSet::buffer(1, delta_ssbo.clone()),
            WriteDescriptorSet::buffer(2, control_ubo.clone()),
        ],
        [],
    )
    .map_err(|e| format!("renderer: constraint apply descriptor set: {e}"))?;

    Ok(ClothGpuConstraintResources {
        constraint_ssbo,
        delta_ssbo,
        adj_offsets_ssbo,
        adj_constraints_ssbo,
        control_ubo,
        lambda_ssbo,
        dlambda_ssbo,
        constraint_count: constraints.len() as u32,
        lambda_update_set,
        accumulate_set,
        apply_set,
    })
}

fn allocate_cloth_normal_resources(
    triangle_indices: &[u32],
    particle_count: u32,
    cloth_pos_ssbo: &Subbuffer<[[f32; 4]]>,
    cloth_norm_ssbo: &Subbuffer<[[f32; 4]]>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    ds_allocator: &Arc<StandardDescriptorSetAllocator>,
    normal_pipeline: &Arc<ComputePipeline>,
) -> Result<ClothGpuNormalResources, String> {
    let triangle_idx_ssbo = gpu_alloc::host_buffer(
        &memory_allocator,
        BufferUsage::STORAGE_BUFFER,
        triangle_indices.iter().copied(),
        "triangle index SSBO",
    )?;

    let adj = crate::simulation::cloth_gpu_boundary::build_vertex_triangle_adjacency(
        triangle_indices,
        particle_count,
    );
    let adj_offsets_ssbo = gpu_alloc::host_buffer(
        &memory_allocator,
        BufferUsage::STORAGE_BUFFER,
        adj.offsets.iter().copied(),
        "normal adj offsets SSBO",
    )?;
    let adj_triangles_data: Vec<u32> = if adj.triangles.is_empty() {
        vec![0_u32]
    } else {
        adj.triangles.clone()
    };
    let adj_triangles_ssbo = gpu_alloc::host_buffer(
        &memory_allocator,
        BufferUsage::STORAGE_BUFFER,
        adj_triangles_data,
        "normal adj scatter SSBO",
    )?;

    let control_ubo = gpu_alloc::host_ubo(
        &memory_allocator,
        pipeline::ClothNormalControl {
            vertex_count: particle_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        },
        "normal control UBO",
    )?;

    let normal_layout = normal_pipeline
        .layout()
        .set_layouts()
        .first()
        .ok_or("renderer: normal pipeline missing set 0")?
        .clone();
    let normal_set = DescriptorSet::new(
        ds_allocator.clone(),
        normal_layout,
        [
            WriteDescriptorSet::buffer(0, cloth_pos_ssbo.clone()),
            WriteDescriptorSet::buffer(1, triangle_idx_ssbo.clone()),
            WriteDescriptorSet::buffer(2, adj_offsets_ssbo.clone()),
            WriteDescriptorSet::buffer(3, adj_triangles_ssbo.clone()),
            WriteDescriptorSet::buffer(4, cloth_norm_ssbo.clone()),
            WriteDescriptorSet::buffer(5, control_ubo.clone()),
        ],
        [],
    )
    .map_err(|e| format!("renderer: normal descriptor set: {e}"))?;

    Ok(ClothGpuNormalResources {
        triangle_idx_ssbo,
        adj_offsets_ssbo,
        adj_triangles_ssbo,
        control_ubo,
        normal_set,
    })
}

impl VulkanRenderer {
    /// Read the previous frame's GPU cloth state (positions + normals)
    /// back to the CPU. Called from `render` after the frame's fence
    /// has been waited (see `RenderResult::cloth_readback`), so the
    /// host-visible SSBOs read coherently. Slots that never dispatched
    /// (`version == 0`) are skipped; cost is one mapped read per
    /// active GPU cloth (~30 KB + ~30 KB at 2.5k particles).
    pub(crate) fn read_cloth_positions(&mut self) -> Vec<ClothReadback> {
        let mut out = Vec::new();
        for ((mesh_id, primitive_id), slot) in self.transform_cache.iter() {
            let Some(gpu) = slot.cloth_gpu.as_ref() else {
                continue;
            };
            if gpu.state.version == 0 {
                continue; // allocated but never dispatched
            }
            let positions = match slot.cloth_pos_ssbo.read() {
                Ok(guard) => guard.iter().map(|p| [p[0], p[1], p[2]]).collect(),
                Err(e) => {
                    log::warn!("render: cloth readback (positions) failed: {e}");
                    continue;
                }
            };
            let normals = if gpu.normals.is_some() {
                slot.cloth_norm_ssbo
                    .read()
                    .ok()
                    .map(|guard| guard.iter().map(|n| [n[0], n[1], n[2]]).collect())
            } else {
                None
            };
            out.push(ClothReadback {
                mesh_id: *mesh_id,
                primitive_id: *primitive_id,
                version: gpu.state.version,
                positions,
                normals,
            });
        }
        out
    }
}

/// Allocate the per-slot self-collision resources (grid hash table +
/// particle-neighbour CSR from the attach constraints + control UBO).
/// Called lazily from the prepare half the first time a frame arrives
/// with `self_collision` enabled; the CSR is attach-static, the
/// control UBO's radius is rewritten per frame.
#[allow(clippy::too_many_arguments)]
pub(super) fn ensure_cloth_gpu_selfcol_resources(
    renderer_slot: &mut crate::renderer::TransformGpuData,
    constraints: &[(u32, u32, f32, f32)],
    particle_count: u32,
    radius: f32,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    ds_allocator: &Arc<StandardDescriptorSetAllocator>,
    cloth_selfcol_build_pipeline: &Arc<ComputePipeline>,
    cloth_selfcol_resolve_pipeline: &Arc<ComputePipeline>,
) -> Result<(), String> {
    use crate::simulation::cloth_gpu_boundary::build_particle_particle_adjacency;
    {
        let gpu = renderer_slot
            .cloth_gpu
            .as_mut()
            .ok_or("renderer: selfcol resources require an allocated cloth slot")?;
        if gpu.selfcol.is_some() {
            if let Some(res) = gpu.selfcol.as_mut() {
                let mut g = res
                    .control_ubo
                    .write()
                    .map_err(|e| format!("renderer: selfcol UBO write: {e}"))?;
                g.particle_count = particle_count;
                g.radius = radius;
            }
            return Ok(());
        }
    }
    let pairs: Vec<(u32, u32)> = constraints.iter().map(|c| (c.0, c.1)).collect();
    let adj = build_particle_particle_adjacency(&pairs, particle_count);
    let adj_offsets_ssbo = gpu_alloc::host_buffer(
        memory_allocator,
        BufferUsage::STORAGE_BUFFER,
        adj.offsets.iter().copied(),
        "selfcol adj offsets SSBO",
    )?;
    let adj_particles_ssbo = gpu_alloc::host_buffer(
        memory_allocator,
        BufferUsage::STORAGE_BUFFER,
        adj.triangles.iter().copied(),
        "selfcol adj particles SSBO",
    )?;
    let cell_counts_ssbo = gpu_alloc::host_buffer(
        memory_allocator,
        BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
        vec![0u32; pipeline::CLOTH_SELFCOL_TABLE_SIZE as usize],
        "selfcol cell counts SSBO",
    )?;
    let cell_entries_ssbo = gpu_alloc::host_buffer(
        memory_allocator,
        BufferUsage::STORAGE_BUFFER,
        vec![0u32; (pipeline::CLOTH_SELFCOL_TABLE_SIZE * pipeline::CLOTH_SELFCOL_BUCKET_SLOTS)
            as usize],
        "selfcol cell entries SSBO",
    )?;
    let control = pipeline::ClothSelfColControl {
        particle_count,
        radius,
        table_size: pipeline::CLOTH_SELFCOL_TABLE_SIZE,
        _pad: 0,
    };
    let control_ubo = gpu_alloc::host_ubo(memory_allocator, control, "selfcol control UBO")?;

    let build_layout = cloth_selfcol_build_pipeline
        .layout()
        .set_layouts()
        .first()
        .ok_or("renderer: selfcol build pipeline missing set 0")?
        .clone();
    let build_set = DescriptorSet::new(
        ds_allocator.clone(),
        build_layout,
        [
            WriteDescriptorSet::buffer(0, renderer_slot.cloth_pos_ssbo.clone()),
            WriteDescriptorSet::buffer(1, cell_counts_ssbo.clone()),
            WriteDescriptorSet::buffer(2, cell_entries_ssbo.clone()),
            WriteDescriptorSet::buffer(3, control_ubo.clone()),
        ],
        [],
    )
    .map_err(|e| format!("renderer: selfcol build descriptor set: {e}"))?;

    let resolve_layout = cloth_selfcol_resolve_pipeline
        .layout()
        .set_layouts()
        .first()
        .ok_or("renderer: selfcol resolve pipeline missing set 0")?
        .clone();
    let resolve_set = DescriptorSet::new(
        ds_allocator.clone(),
        resolve_layout,
        [
            WriteDescriptorSet::buffer(0, renderer_slot.cloth_pos_ssbo.clone()),
            WriteDescriptorSet::buffer(1, adj_offsets_ssbo.clone()),
            WriteDescriptorSet::buffer(2, adj_particles_ssbo.clone()),
            WriteDescriptorSet::buffer(3, cell_counts_ssbo.clone()),
            WriteDescriptorSet::buffer(4, cell_entries_ssbo.clone()),
            WriteDescriptorSet::buffer(5, control_ubo.clone()),
        ],
        [],
    )
    .map_err(|e| format!("renderer: selfcol resolve descriptor set: {e}"))?;

    let gpu = renderer_slot
        .cloth_gpu
        .as_mut()
        .expect("cloth slot present");
    gpu.selfcol = Some(ClothGpuSelfColResources {
        adj_offsets_ssbo,
        adj_particles_ssbo,
        cell_counts_ssbo,
        cell_entries_ssbo,
        control_ubo,
        build_set,
        resolve_set,
    });
    Ok(())
}

/// Build / refresh the per-slot GPU collision resources for this
/// frame's capsule list. The SSBO is rewritten every frame (capsules
/// follow the bones) and reallocated only when the count changes; the
/// control UBO carries the (slot-static) particle count plus the
/// frame's collider count and margin.
#[allow(clippy::too_many_arguments)]
pub(super) fn ensure_cloth_gpu_collide_resources(
    renderer_slot: &mut crate::renderer::TransformGpuData,
    colliders: &[crate::renderer::frame_input::ClothGpuCollider],
    particle_count: u32,
    margin: f32,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    ds_allocator: &Arc<StandardDescriptorSetAllocator>,
    cloth_collide_pipeline: &Arc<ComputePipeline>,
) -> Result<(), String> {
    let count = colliders.len() as u32;
    let cloth_pos_ssbo = renderer_slot.cloth_pos_ssbo.clone();
    let gpu = renderer_slot
        .cloth_gpu
        .as_mut()
        .ok_or("renderer: collide resources require an allocated cloth slot")?;
    let needs_rebuild = gpu
        .collide
        .as_ref()
        .map(|c| c.collider_count != count)
        .unwrap_or(true);
    if needs_rebuild {
        let collider_ssbo = gpu_alloc::host_buffer(
            memory_allocator,
            BufferUsage::STORAGE_BUFFER,
            colliders.iter().map(|c| pipeline::ClothGpuColliderGpu {
                a: [c.p0[0], c.p0[1], c.p0[2], c.radius],
                b: [c.p1[0], c.p1[1], c.p1[2], 0.0],
            }),
            "cloth collider SSBO",
        )?;
        let control_ubo = gpu_alloc::host_ubo(
            memory_allocator,
            pipeline::ClothCollideControl {
                particle_count,
                collider_count: count,
                margin,
                _pad: 0,
            },
            "cloth collide control UBO",
        )?;
        let set_layout = cloth_collide_pipeline
            .layout()
            .set_layouts()
            .first()
            .ok_or("renderer: cloth collide pipeline missing set 0 layout")?
            .clone();
        let collide_set = DescriptorSet::new(
            ds_allocator.clone(),
            set_layout,
            [
                // The positions buffer is the transform slot's
                // cloth_pos_ssbo, passed in by the caller through the
                // renderer slot below.
                WriteDescriptorSet::buffer(0, cloth_pos_ssbo),
                WriteDescriptorSet::buffer(1, collider_ssbo.clone()),
                WriteDescriptorSet::buffer(2, control_ubo.clone()),
            ],
            [],
        )
        .map_err(|e| format!("renderer: cloth collide descriptor set: {e}"))?;
        gpu.collide = Some(ClothGpuCollideResources {
            collider_ssbo,
            control_ubo,
            collide_set,
            collider_count: count,
        });
    } else if let Some(res) = gpu.collide.as_mut() {
        // Refresh the capsule rows in place (same live-write pattern as
        // the verlet control UBO).
        let mut guard = res
            .collider_ssbo
            .write()
            .map_err(|e| format!("renderer: collider SSBO write: {e}"))?;
        for (dst, c) in guard.iter_mut().zip(colliders.iter()) {
            *dst = pipeline::ClothGpuColliderGpu {
                a: [c.p0[0], c.p0[1], c.p0[2], c.radius],
                b: [c.p1[0], c.p1[1], c.p1[2], 0.0],
            };
        }
    }
    if let Some(res) = gpu.collide.as_mut() {
        let mut g = res
            .control_ubo
            .write()
            .map_err(|e| format!("renderer: collide control UBO write: {e}"))?;
        g.collider_count = count;
        g.margin = margin;
        g.particle_count = particle_count;
    }
    Ok(())
}
