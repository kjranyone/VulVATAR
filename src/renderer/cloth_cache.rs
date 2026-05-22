//! Cloth GPU solver resource allocation — extracted from
//! `VulkanRenderer` as the third slice of the #12 renderer split.
//! Owns the per-primitive cloth slot construction (Verlet integrator
//! state, optional PBD constraint resources, optional normal-
//! recomputation resources) used by the cloth compute dispatch
//! pipeline. The render-loop dispatch wiring still lives in `mod.rs`
//! next to the rest of the per-frame command-buffer construction;
//! only the slot-creation path moved.

use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::pipeline::{ComputePipeline, Pipeline};

use crate::asset::{MeshId, PrimitiveId};
use crate::renderer::{
    pipeline, ClothGpuConstraintResources, ClothGpuNormalResources, ClothGpuSlot, VulkanRenderer,
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
        let inv_mass_at = |i: usize| attach.inv_masses.get(i).copied().unwrap_or(1.0);
        let pinned_at = |i: usize| {
            if attach.pinned.get(i).copied().unwrap_or(false) {
                1.0_f32
            } else {
                0.0
            }
        };

        let prev_pos_ssbo = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            // Seed prev_pos to same as initial positions ⇒ zero
            // velocity at start. `w` = pinned flag (1.0 = locked).
            initial_positions
                .iter()
                .enumerate()
                .map(|(i, p)| [p[0], p[1], p[2], pinned_at(i)]),
        )
        .map_err(|e| format!("renderer: cloth prev_pos SSBO alloc failed: {e}"))?;

        let verlet_control_ubo = Buffer::from_data(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            pipeline::ClothVerletControl {
                dt: 0.0,
                damping: 0.0,
                particle_count,
                _pad0: 0,
                gravity: [0.0; 4],
                wind: [0.0; 4],
            },
        )
        .map_err(|e| format!("renderer: cloth verlet control UBO alloc failed: {e}"))?;

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
            for (i, (dst, p)) in
                guard.iter_mut().zip(initial_positions.iter()).enumerate()
            {
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
                state: crate::simulation::cloth_gpu_boundary::ClothGpuSimulationState::from_authoring(
                    particle_count,
                    attach.constraints.len() as u32,
                    8,
                ),
                prev_pos_ssbo,
                verlet_control_ubo,
                verlet_set,
                constraints,
                normals,
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
    let constraint_ssbo = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        constraints.iter().map(|(a, b, r, s)| pipeline::ClothConstraintGpu {
            particle_a: *a,
            particle_b: *b,
            rest_length: *r,
            stiffness: *s,
        }),
    )
    .map_err(|e| format!("renderer: cloth constraint SSBO alloc: {e}"))?;

    let delta_ssbo = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        (0..particle_count as usize).map(|_| [0.0_f32; 4]),
    )
    .map_err(|e| format!("renderer: cloth delta SSBO alloc: {e}"))?;

    let constraint_pairs: Vec<(u32, u32)> =
        constraints.iter().map(|(a, b, _, _)| (*a, *b)).collect();
    let adj = crate::simulation::cloth_gpu_boundary::build_particle_constraint_adjacency(
        &constraint_pairs,
        particle_count,
    );
    let adj_offsets_ssbo = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        adj.offsets.iter().copied(),
    )
    .map_err(|e| format!("renderer: constraint adj offsets SSBO alloc: {e}"))?;
    // Vulkano refuses zero-sized buffers; supply a single u32 stub
    // when the CSR scatter array is empty (no particle touches any
    // constraint). The shader's adjacency loop won't iterate
    // because every `offsets[v+1] == offsets[v]` in that case.
    let adj_constraints_data: Vec<u32> = if adj.triangles.is_empty() {
        vec![0_u32]
    } else {
        adj.triangles.clone()
    };
    let adj_constraints_ssbo = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        adj_constraints_data,
    )
    .map_err(|e| format!("renderer: constraint adj scatter SSBO alloc: {e}"))?;

    let control_ubo = Buffer::from_data(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
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
    )
    .map_err(|e| format!("renderer: constraint control UBO alloc: {e}"))?;

    // XPBD per-constraint Lagrange multiplier (accumulates across the
    // projection iterations within one substep; reset to zero at
    // substep start by the renderer's `fill_buffer`).
    let lambda_ssbo: Subbuffer<[f32]> = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        // Vulkano refuses zero-sized buffers — supply a single 0.0
        // stub when there are no constraints. Won't be read because
        // `constraint_count = 0` short-circuits the lambda-update
        // dispatch's per-invocation work.
        if constraints.is_empty() {
            vec![0.0_f32]
        } else {
            vec![0.0_f32; constraints.len()]
        },
    )
    .map_err(|e| format!("renderer: cloth lambda SSBO alloc: {e}"))?;

    // Per-iteration Δλ_j scratch buffer. Written by the
    // lambda-update pass each iteration, read by the accumulate pass;
    // contents are overwritten so no initial state matters.
    let dlambda_ssbo: Subbuffer<[f32]> = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        if constraints.is_empty() {
            vec![0.0_f32]
        } else {
            vec![0.0_f32; constraints.len()]
        },
    )
    .map_err(|e| format!("renderer: cloth dlambda SSBO alloc: {e}"))?;

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
    let triangle_idx_ssbo = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        triangle_indices.iter().copied(),
    )
    .map_err(|e| format!("renderer: triangle index SSBO alloc: {e}"))?;

    let adj = crate::simulation::cloth_gpu_boundary::build_vertex_triangle_adjacency(
        triangle_indices,
        particle_count,
    );
    let adj_offsets_ssbo = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        adj.offsets.iter().copied(),
    )
    .map_err(|e| format!("renderer: normal adj offsets SSBO alloc: {e}"))?;
    let adj_triangles_data: Vec<u32> = if adj.triangles.is_empty() {
        vec![0_u32]
    } else {
        adj.triangles.clone()
    };
    let adj_triangles_ssbo = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        adj_triangles_data,
    )
    .map_err(|e| format!("renderer: normal adj scatter SSBO alloc: {e}"))?;

    let control_ubo = Buffer::from_data(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        pipeline::ClothNormalControl {
            vertex_count: particle_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        },
    )
    .map_err(|e| format!("renderer: normal control UBO alloc: {e}"))?;

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
