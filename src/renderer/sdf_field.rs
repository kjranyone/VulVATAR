//! Body-surface distance field slots: per-avatar-instance GPU storage
//! for the splat pass that turns the skinned body primitives into the
//! distance field the spring solver resolves hair against.
//!
//! One slot per avatar instance, allocated on first use and reallocated
//! only when the grid geometry or the splatted-primitive list changes.
//! Each primitive gets its own descriptor set (its skinned VBO + index
//! buffer + the shared field + its own params UBO) and its own
//! dispatch; the atomicMin splats accumulate across dispatches. The
//! field buffer is host-visible so `read_sdf_fields` can map it
//! directly after the previous frame's fence — the same
//! one-frame-stale discipline as the cloth position readback
//! (`read_cloth_positions`). Field semantics live in
//! `simulation/sdf.rs`; the splat shader is `pipeline::body_sdf_splat_cs`.

use std::collections::HashMap;
use std::sync::Arc;

use vulkano::buffer::Subbuffer;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{ComputePipeline, Pipeline};

use crate::asset::{MeshId, PrimitiveId};
use crate::renderer::gpu_alloc;
use crate::renderer::pipeline::{BodySdfSplatParams, GpuVertex};
use crate::simulation::sdf::SdfGrid;

/// Per-primitive splat resources: its params UBO (tri count) and the
/// descriptor set binding its VBO/indices plus the shared field.
pub(super) struct BodySdfPrimGpu {
    pub(super) params: Subbuffer<BodySdfSplatParams>,
    pub(super) set: Arc<DescriptorSet>,
    pub(super) groups: [u32; 3],
}

/// Per-instance GPU resources for the splat dispatches.
pub(super) struct BodySdfSlot {
    /// The distance field. `u32` cells holding f32 bit patterns;
    /// `u32::MAX` (the `fill_buffer` value) = unsplatted sentinel.
    pub(super) field: Subbuffer<[u32]>,
    pub(super) prims: Vec<BodySdfPrimGpu>,
    /// Grid + primitive list the slot was built for — identifies stale
    /// slots after an avatar swap.
    pub(super) grid: SdfGrid,
    pub(super) prim_ids: Vec<(MeshId, PrimitiveId)>,
}

/// Readback payload handed to the application through
/// `RenderResult::sdf_fields`.
pub struct SdfFieldReadback {
    /// Avatar instance the field belongs to (`AvatarInstanceId::0`).
    pub instance_id: u64,
    pub grid: SdfGrid,
    pub data: Arc<Vec<f32>>,
}

pub(super) type SdfSlotMap = HashMap<u64, BodySdfSlot>;

/// Resolved inputs for one splatted primitive.
pub(super) struct SplatPrimInput {
    pub vertex_buffer: Subbuffer<[GpuVertex]>,
    pub index_buffer: Subbuffer<[u32]>,
    pub tri_count: u32,
}

/// Allocate or refresh the instance's splat resources for `grid` and
/// the primitive list (each entry with its resolved buffers), and
/// (re)build the per-primitive descriptor sets.
pub(super) fn ensure_sdf_slot(
    slots: &mut SdfSlotMap,
    instance_id: u64,
    grid: SdfGrid,
    prims: &[SplatPrimInput],
    prim_ids: Vec<(MeshId, PrimitiveId)>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    ds_allocator: &Arc<StandardDescriptorSetAllocator>,
    pipeline: &Arc<ComputePipeline>,
) -> Result<(), String> {
    let needs_rebuild = match slots.get(&instance_id) {
        Some(slot) => slot.grid != grid || slot.prim_ids != prim_ids,
        None => true,
    };
    if needs_rebuild {
        let cells = grid.cell_count() as u64;
        let field = gpu_alloc::host_read_slice(memory_allocator, cells, "body SDF field")?;
        let set_layout = pipeline
            .layout()
            .set_layouts()
            .first()
            .ok_or("renderer: body SDF splat pipeline missing set 0 layout")?
            .clone();
        let mut prim_gpu = Vec::with_capacity(prims.len());
        for prim in prims {
            let params = gpu_alloc::host_ubo(
                memory_allocator,
                BodySdfSplatParams {
                    dims_tri: [grid.dims[0], grid.dims[1], grid.dims[2], prim.tri_count],
                    origin_voxel: [
                        grid.origin[0],
                        grid.origin[1],
                        grid.origin[2],
                        grid.voxel,
                    ],
                    shell_pad: [crate::simulation::sdf::SHELL_METRES, 0.0, 0.0, 0.0],
                },
                "body SDF splat params",
            )?;
            let set = DescriptorSet::new(
                ds_allocator.clone(),
                set_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, prim.vertex_buffer.clone()),
                    WriteDescriptorSet::buffer(1, prim.index_buffer.clone()),
                    WriteDescriptorSet::buffer(2, field.clone()),
                    WriteDescriptorSet::buffer(3, params.clone()),
                ],
                [],
            )
            .map_err(|e| format!("renderer: body SDF splat descriptor set: {e}"))?;
            prim_gpu.push(BodySdfPrimGpu {
                params,
                set,
                groups: [(prim.tri_count + 63) / 64, 1, 1],
            });
        }
        slots.insert(
            instance_id,
            BodySdfSlot {
                field,
                prims: prim_gpu,
                grid,
                prim_ids,
            },
        );
    } else if let Some(slot) = slots.get_mut(&instance_id) {
        // Grid + prim list unchanged; refresh the triangle counts in
        // place (a LOD / subset switch could legally change them while
        // keeping the same primitive identities).
        for (gpu, prim) in slot.prims.iter_mut().zip(prims.iter()) {
            let mut g = gpu
                .params
                .write()
                .map_err(|e| format!("renderer: body SDF params write: {e}"))?;
            g.dims_tri[3] = prim.tri_count;
            gpu.groups = [(prim.tri_count + 63) / 64, 1, 1];
        }
    }
    Ok(())
}

/// Map the (previous frame's) field buffers into CPU memory, for the
/// instances planned this frame. Must be called after the previous
/// frame's fence has been waited and before this frame's submission —
/// the same window `read_cloth_positions` uses.
pub(super) fn read_sdf_fields(
    slots: &SdfSlotMap,
    planned_instances: &[u64],
) -> Vec<SdfFieldReadback> {
    let mut out = Vec::new();
    for instance_id in planned_instances {
        let Some(slot) = slots.get(instance_id) else {
            continue;
        };
        let guard = match slot.field.read() {
            Ok(g) => g,
            Err(e) => {
                log::warn!("render: body SDF readback failed: {e}");
                continue;
            }
        };
        let data: Vec<f32> = guard.iter().map(|bits| f32::from_bits(*bits)).collect();
        out.push(SdfFieldReadback {
            instance_id: *instance_id,
            grid: slot.grid,
            data: Arc::new(data),
        });
    }
    out
}
