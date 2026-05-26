//! Compute-prepass per-primitive resource cache extracted from
//! `VulkanRenderer` as a follow-up slice of the #12 renderer split.
//! Owns the lazy construction of [`TransformGpuData`] slots
//! (base SSBO + transformed VBO + morph delta packing + cloth SSBO
//! allocation + descriptor set), plus the shared 1-element stub SSBO
//! used by primitives that have no morph or cloth resources.
//!
//! The shared `stub_storage_ssbo` is owned by `VulkanRenderer` so the
//! same Arc-backed 1-element zero buffer can stand in for every
//! "this slot has neither morph nor cloth" descriptor binding.
//! Vulkano refuses zero-element storage buffers, so the stub is the
//! cheap way to keep set 0 binding consistent across morph / cloth
//! topology variants.

use std::sync::Arc;

use log::warn;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::pipeline::{ComputePipeline, Pipeline};

use crate::asset::{MeshId, MeshPrimitiveAsset, PrimitiveId};
use crate::renderer::pipeline::{self, GpuVertex, TransformControl};
use crate::renderer::{TransformGpuData, VulkanRenderer, MORPH_MAX_TARGETS};

impl VulkanRenderer {
    pub(super) fn ensure_stub_ssbo(
        &mut self,
        memory_allocator: &Arc<StandardMemoryAllocator>,
    ) -> Result<Subbuffer<[[f32; 4]]>, String> {
        if let Some(ref stub) = self.stub_storage_ssbo {
            return Ok(stub.clone());
        }
        let stub = Buffer::from_iter(
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
            [[0.0_f32; 4]].into_iter(),
        )
        .map_err(|e| format!("renderer: stub SSBO alloc failed: {e}"))?;
        self.stub_storage_ssbo = Some(stub.clone());
        Ok(stub)
    }

    /// Ensure a [`TransformGpuData`] slot exists for `(mesh_id, prim_id)`
    /// with cloth allocation matching this frame's requirements. If the
    /// existing slot's cloth allocation shape disagrees with `has_cloth` /
    /// `has_cloth_normals` (rare — user toggled a cloth-bearing accessory
    /// on / off at runtime), the slot is evicted and rebuilt so the
    /// descriptor set's pinned bindings stay consistent.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ensure_transform_data(
        &mut self,
        mesh_id: MeshId,
        prim_id: PrimitiveId,
        prim: &MeshPrimitiveAsset,
        has_cloth: bool,
        has_cloth_normals: bool,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        ds_allocator: &Arc<StandardDescriptorSetAllocator>,
        transform_pipeline: &Arc<ComputePipeline>,
    ) -> Result<(), String> {
        let key = (mesh_id, prim_id);

        if let Some(existing) = self.transform_cache.get(&key) {
            if existing.has_cloth_alloc == has_cloth
                && existing.has_cloth_normals_alloc == has_cloth_normals
            {
                return Ok(());
            }
            // Cloth allocation shape changed under us; drop the slot so
            // the fresh allocation lands below with the right SSBOs.
            self.transform_cache.remove(&key);
        }

        let vd = prim
            .vertices
            .as_ref()
            .ok_or("renderer: ensure_transform_data requires primitive vertex data")?;
        let vertex_count = vd.positions.len();
        if vertex_count == 0 {
            return Err("renderer: ensure_transform_data with empty vertex data".to_string());
        }
        let indices: &[u32] = prim.indices.as_deref().unwrap_or(&[]);
        let owned_indices;
        let idx_slice: &[u32] = if indices.is_empty() {
            owned_indices = (0..vertex_count as u32).collect::<Vec<_>>();
            &owned_indices
        } else {
            indices
        };
        let index_count = idx_slice.len() as u32;

        let base_data = pipeline::vertex_data_to_base(vd);
        let base_ssbo = Buffer::from_iter(
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
            base_data,
        )
        .map_err(|e| format!("renderer: base SSBO alloc failed: {e}"))?;

        let index_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            idx_slice.iter().copied(),
        )
        .map_err(|e| format!("renderer: index buffer alloc failed: {e}"))?;

        // Compute output: device-local storage + vertex buffer.
        let transformed_vbo: Subbuffer<[GpuVertex]> = Buffer::new_slice::<GpuVertex>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER | BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            vertex_count as u64,
        )
        .map_err(|e| format!("renderer: transformed VBO alloc failed: {e}"))?;

        // Morph deltas, or the shared stub when this primitive has none.
        let raw_target_count = prim.morph_targets.len();
        let target_count = raw_target_count.min(MORPH_MAX_TARGETS);
        if raw_target_count > MORPH_MAX_TARGETS {
            warn!(
                "renderer: primitive {:?} has {} morph targets, clamping to {}",
                key, raw_target_count, MORPH_MAX_TARGETS
            );
        }
        let morph_deltas = if target_count == 0 {
            self.ensure_stub_ssbo(memory_allocator)?
        } else {
            let mut deltas: Vec<[f32; 4]> =
                Vec::with_capacity(target_count * vertex_count * 2);
            for t in 0..target_count {
                let target = &prim.morph_targets[t];
                for v in 0..vertex_count {
                    let pd = target.position_deltas.get(v).copied().unwrap_or([0.0; 3]);
                    let nd = target.normal_deltas.get(v).copied().unwrap_or([0.0; 3]);
                    deltas.push([pd[0], pd[1], pd[2], 0.0]);
                    deltas.push([nd[0], nd[1], nd[2], 0.0]);
                }
            }
            Buffer::from_iter(
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
                deltas,
            )
            .map_err(|e| format!("renderer: morph deltas alloc failed: {e}"))?
        };

        // Cloth SSBOs, sized to `vertex_count` when the primitive is
        // cloth-bearing this frame; otherwise the shared stub. The render
        // loop rewrites `cloth_pos_ssbo` / `cloth_norm_ssbo` in place
        // whenever the cloth solver bumps its `version`.
        let cloth_pos_ssbo = if has_cloth {
            Buffer::from_iter(
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
                (0..vertex_count).map(|_| [0.0_f32; 4]),
            )
            .map_err(|e| format!("renderer: cloth pos SSBO alloc failed: {e}"))?
        } else {
            self.ensure_stub_ssbo(memory_allocator)?
        };
        let cloth_norm_ssbo = if has_cloth_normals {
            Buffer::from_iter(
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
                (0..vertex_count).map(|_| [0.0_f32; 4]),
            )
            .map_err(|e| format!("renderer: cloth norm SSBO alloc failed: {e}"))?
        } else {
            self.ensure_stub_ssbo(memory_allocator)?
        };

        let mut ctrl = TransformControl::zeroed();
        ctrl.vertex_count = vertex_count as u32;
        ctrl.target_count = target_count as u32;
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
            ctrl,
        )
        .map_err(|e| format!("renderer: control UBO alloc failed: {e}"))?;

        let set0_layout = transform_pipeline
            .layout()
            .set_layouts()
            .first()
            .ok_or("renderer: transform pipeline missing set 0 layout")?
            .clone();
        let transform_set = DescriptorSet::new(
            ds_allocator.clone(),
            set0_layout,
            [
                WriteDescriptorSet::buffer(0, base_ssbo.clone()),
                WriteDescriptorSet::buffer(1, morph_deltas.clone()),
                WriteDescriptorSet::buffer(2, cloth_pos_ssbo.clone()),
                WriteDescriptorSet::buffer(3, cloth_norm_ssbo.clone()),
                WriteDescriptorSet::buffer(4, control_ubo.clone()),
                WriteDescriptorSet::buffer(5, transformed_vbo.clone()),
            ],
            [],
        )
        .map_err(|e| format!("renderer: transform descriptor set: {e}"))?;

        self.transform_cache.insert(
            key,
            TransformGpuData {
                base_ssbo,
                index_buffer,
                transformed_vbo,
                control_ubo,
                morph_deltas,
                cloth_pos_ssbo,
                cloth_norm_ssbo,
                transform_set,
                index_count,
                vertex_count: vertex_count as u32,
                target_count: target_count as u32,
                has_cloth_alloc: has_cloth,
                has_cloth_normals_alloc: has_cloth_normals,
                last_cloth_version: None,
                cloth_gpu: None,
            },
        );
        self.gpu_runtime_counters.morph_gpu_resource_creations += 1;
        if has_cloth {
            self.gpu_runtime_counters.cloth_cache_creations += 1;
        }
        Ok(())
    }
}
