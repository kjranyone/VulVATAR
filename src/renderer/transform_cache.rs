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

use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{ComputePipeline, Pipeline};

use crate::asset::{MeshId, MeshPrimitiveAsset, PrimitiveId};
use crate::renderer::gpu_alloc;
use crate::renderer::pipeline::{self, GpuVertex, TransformControl};
use crate::renderer::{mat4_cols_identity, SkinningCacheEntry, TransformGpuData, VulkanRenderer};

impl VulkanRenderer {
    /// Ensure a [`TransformGpuData`] slot exists for `(mesh_id, prim_id)`
    /// with cloth and skin-anchor allocation matching this frame's requirements.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ensure_transform_data(
        &mut self,
        mesh_id: MeshId,
        prim_id: PrimitiveId,
        prim: &MeshPrimitiveAsset,
        has_cloth: bool,
        has_cloth_normals: bool,
        body_transformed_vbo: Option<Subbuffer<[GpuVertex]>>,
        containment_parent_vbo: Option<Subbuffer<[GpuVertex]>>,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        ds_allocator: &Arc<StandardDescriptorSetAllocator>,
        transform_pipeline: &Arc<ComputePipeline>,
    ) -> Result<(), String> {
        let key = (mesh_id, prim_id);
        let has_anchors = prim.skin_anchors.is_some() && body_transformed_vbo.is_some();
        let has_containment =
            prim.containment_anchors.is_some() && containment_parent_vbo.is_some();

        // When the slot exists with a matching allocation shape, keep it.
        // Otherwise rebuild — with buffer-handle reuse: every immutable
        // per-asset payload (base SSBO, index buffer, transformed VBO,
        // morph runs, anchor SSBOs, control UBO) is taken from the old
        // slot rather than reallocated. Keeping the `transformed_vbo`
        // handle stable for the lifetime of the (mesh, prim) key is what
        // lets OTHER primitives bind it as their hierarchical-clearance
        // parent without their descriptor sets going stale: frame 1
        // materialises a parent anchor-less before its own iteration
        // upgrades it, and cloth/anchor shape flips later would
        // otherwise hand every referencing child a dead VBO.
        let previous = if let Some(existing) = self.transform_cache.get(&key) {
            if existing.has_cloth_alloc == has_cloth
                && existing.has_cloth_normals_alloc == has_cloth_normals
                && existing.has_skin_anchors_alloc == has_anchors
                && existing.has_containment_alloc == has_containment
            {
                return Ok(());
            }
            // The slot's descriptor set (and possibly its cloth SSBOs) is
            // about to be swapped — cached command buffers referencing the
            // old set must not survive.
            self.cb_cache.clear();
            self.transform_cache.remove(&key)
        } else {
            None
        };
        // Cloth SSBOs are the one payload whose SIZE depends on the
        // allocation flags (full vertex count vs shared 1-element stub),
        // so they are only reusable when that flag did not flip.
        let (prev_had_cloth, prev_had_cloth_norm) = previous
            .as_ref()
            .map(|p| (p.has_cloth_alloc, p.has_cloth_normals_alloc))
            .unwrap_or((false, false));

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

        let base_ssbo = match previous.as_ref().map(|p| p.base_ssbo.clone()) {
            Some(b) => b,
            None => gpu_alloc::host_buffer(
                memory_allocator,
                BufferUsage::STORAGE_BUFFER,
                pipeline::vertex_data_to_base(vd),
                "base SSBO",
            )?,
        };

        let index_buffer = match previous.as_ref().map(|p| p.index_buffer.clone()) {
            Some(b) => b,
            None => gpu_alloc::host_buffer(
                memory_allocator,
                BufferUsage::INDEX_BUFFER,
                idx_slice.iter().copied(),
                "index buffer",
            )?,
        };

        // Compute output: device-local storage + vertex buffer.
        let transformed_vbo: Subbuffer<[GpuVertex]> =
            match previous.as_ref().map(|p| p.transformed_vbo.clone()) {
                Some(b) => b,
                None => gpu_alloc::device_slice(
                    memory_allocator,
                    BufferUsage::VERTEX_BUFFER | BufferUsage::STORAGE_BUFFER,
                    vertex_count as u64,
                    "transformed VBO",
                )?,
            };

        // Sparse morph deltas, or the shared stubs when this primitive
        // has none. Dense storage (targets × vertices × 2 vec4s) made a
        // 446-blendshape face mesh cost ~380 MB; real blend shapes move
        // a small subset of vertices, so storing only non-zero deltas
        // shrinks the same mesh by an order of magnitude and removes any
        // cap on the target count.
        let raw_target_count = prim.morph_targets.len();
        let target_count = raw_target_count as u32;
        let (morph_entries, morph_infos, morph_infos_full) = match previous.as_ref().map(|p| {
            (
                p.morph_entries.clone(),
                p.morph_infos.clone(),
                p.morph_infos_full.clone(),
            )
        }) {
            Some((e, i, f)) => (e, i, f),
            None => {
                let (entries, infos) = pack_sparse_morphs(&prim.morph_targets);
                if raw_target_count == 0 {
                    (
                        gpu_alloc::get_or_init_stub(
                            &mut self.stub_storage_ssbo,
                            memory_allocator,
                            BufferUsage::STORAGE_BUFFER,
                            [0.0_f32; 4],
                            "stub SSBO",
                        )?,
                        gpu_alloc::get_or_init_stub(
                            &mut self.stub_uvec4_ssbo,
                            memory_allocator,
                            BufferUsage::STORAGE_BUFFER,
                            [0u32; 4],
                            "stub uvec4 SSBO",
                        )?,
                        Vec::new(),
                    )
                } else {
                    (
                        gpu_alloc::host_buffer(
                            memory_allocator,
                            BufferUsage::STORAGE_BUFFER,
                            entries,
                            "morph deltas",
                        )?,
                        gpu_alloc::host_buffer(
                            memory_allocator,
                            BufferUsage::STORAGE_BUFFER,
                            infos.clone(),
                            "morph infos",
                        )?,
                        infos,
                    )
                }
            }
        };
        // Per-frame weight scratch, HOST_SEQUENTIAL so the render loop
        // can live-rewrite it through a mapped write each frame.
        let morph_weights_buf = match previous.as_ref().map(|p| p.morph_weights_buf.clone()) {
            Some(b) => b,
            None => {
                if raw_target_count == 0 {
                    gpu_alloc::get_or_init_stub(
                        &mut self.stub_f32_ssbo,
                        memory_allocator,
                        BufferUsage::STORAGE_BUFFER,
                        0.0_f32,
                        "stub f32 SSBO",
                    )?
                } else {
                    gpu_alloc::host_buffer(
                        memory_allocator,
                        BufferUsage::STORAGE_BUFFER,
                        (0..raw_target_count).map(|_| 0.0_f32),
                        "morph weights",
                    )?
                }
            }
        };

        // Cloth SSBOs, sized to `vertex_count` when the primitive is
        // cloth-bearing this frame; otherwise the shared stub. The render
        // loop rewrites `cloth_pos_ssbo` / `cloth_norm_ssbo` in place
        // whenever the cloth solver bumps its `version`.
        let cloth_pos_ssbo = if let (true, Some(p)) = (prev_had_cloth, previous.as_ref()) {
            p.cloth_pos_ssbo.clone()
        } else if has_cloth {
            gpu_alloc::host_buffer(
                memory_allocator,
                BufferUsage::STORAGE_BUFFER,
                (0..vertex_count).map(|_| [0.0_f32; 4]),
                "cloth pos SSBO",
            )?
        } else {
            gpu_alloc::get_or_init_stub(
                &mut self.stub_storage_ssbo,
                memory_allocator,
                BufferUsage::STORAGE_BUFFER,
                [0.0_f32; 4],
                "stub SSBO",
            )?
        };
        let cloth_norm_ssbo = if let (true, Some(p)) = (prev_had_cloth_norm, previous.as_ref()) {
            p.cloth_norm_ssbo.clone()
        } else if has_cloth_normals {
            gpu_alloc::host_buffer(
                memory_allocator,
                BufferUsage::STORAGE_BUFFER,
                (0..vertex_count).map(|_| [0.0_f32; 4]),
                "cloth norm SSBO",
            )?
        } else {
            gpu_alloc::get_or_init_stub(
                &mut self.stub_storage_ssbo,
                memory_allocator,
                BufferUsage::STORAGE_BUFFER,
                [0.0_f32; 4],
                "stub SSBO",
            )?
        };

        // Anchor payloads depend only on the asset, so an old slot's
        // SSBOs are always reusable — only which parent VBO the set binds
        // and the control flags change with `has_anchors` /
        // `has_containment`.
        let skin_anchors_ssbo = match previous.as_ref().map(|p| p.skin_anchors_ssbo.clone()) {
            Some(b) => b,
            None => {
                if let Some(ref anchors) = prim.skin_anchors {
                    gpu_alloc::host_buffer(
                        memory_allocator,
                        BufferUsage::STORAGE_BUFFER,
                        anchors.iter().copied(),
                        "skin anchors SSBO",
                    )?
                } else {
                    gpu_alloc::get_or_init_stub(
                        &mut self.stub_skin_anchor_ssbo,
                        memory_allocator,
                        BufferUsage::STORAGE_BUFFER,
                        crate::asset::SkinAnchor::default(),
                        "stub skin anchor SSBO",
                    )?
                }
            }
        };

        let containment_anchors_ssbo = match previous
            .as_ref()
            .map(|p| p.containment_anchors_ssbo.clone())
        {
            Some(b) => b,
            None => {
                if let Some(ref anchors) = prim.containment_anchors {
                    gpu_alloc::host_buffer(
                        memory_allocator,
                        BufferUsage::STORAGE_BUFFER,
                        anchors.iter().copied(),
                        "containment anchors SSBO",
                    )?
                } else {
                    // Same 1-element stub content as the clearance
                    // anchors, so the shared slot serves both.
                    gpu_alloc::get_or_init_stub(
                        &mut self.stub_skin_anchor_ssbo,
                        memory_allocator,
                        BufferUsage::STORAGE_BUFFER,
                        crate::asset::SkinAnchor::default(),
                        "stub skin anchor SSBO",
                    )?
                }
            }
        };

        let body_vbo = if let Some(ref vbo) = body_transformed_vbo {
            vbo.clone()
        } else {
            gpu_alloc::get_or_init_stub(
                &mut self.stub_vertex_ssbo,
                memory_allocator,
                BufferUsage::STORAGE_BUFFER | BufferUsage::VERTEX_BUFFER,
                GpuVertex::default(),
                "stub vertex SSBO",
            )?
        };

        let containment_body_vbo = if let Some(ref vbo) = containment_parent_vbo {
            vbo.clone()
        } else {
            gpu_alloc::get_or_init_stub(
                &mut self.stub_vertex_ssbo,
                memory_allocator,
                BufferUsage::STORAGE_BUFFER | BufferUsage::VERTEX_BUFFER,
                GpuVertex::default(),
                "stub vertex SSBO",
            )?
        };

        let mut ctrl = TransformControl::zeroed();
        ctrl.vertex_count = vertex_count as u32;
        ctrl.target_count = target_count;
        ctrl.has_cloth = if has_cloth { 1 } else { 0 };
        ctrl.has_cloth_normals = if has_cloth_normals { 1 } else { 0 };
        ctrl.has_skin_anchors = if has_anchors { 1 } else { 0 };
        ctrl.has_containment = if has_containment { 1 } else { 0 };
        // The render loop rewrites every field of the control UBO right
        // after this call, so the previous slot's buffer is reusable as-is.
        let control_ubo = match previous.as_ref().map(|p| p.control_ubo.clone()) {
            Some(b) => b,
            None => gpu_alloc::host_ubo(memory_allocator, ctrl, "control UBO")?,
        };

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
                WriteDescriptorSet::buffer(1, morph_entries.clone()),
                WriteDescriptorSet::buffer(2, cloth_pos_ssbo.clone()),
                WriteDescriptorSet::buffer(3, cloth_norm_ssbo.clone()),
                WriteDescriptorSet::buffer(4, control_ubo.clone()),
                WriteDescriptorSet::buffer(5, transformed_vbo.clone()),
                WriteDescriptorSet::buffer(6, skin_anchors_ssbo.clone()),
                WriteDescriptorSet::buffer(7, body_vbo),
                WriteDescriptorSet::buffer(8, morph_infos.clone()),
                WriteDescriptorSet::buffer(9, morph_weights_buf.clone()),
                WriteDescriptorSet::buffer(10, containment_anchors_ssbo.clone()),
                WriteDescriptorSet::buffer(11, containment_body_vbo),
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
                morph_entries,
                morph_infos,
                morph_infos_full,
                morph_weights_buf,
                cloth_pos_ssbo,
                cloth_norm_ssbo,
                skin_anchors_ssbo,
                containment_anchors_ssbo,
                transform_set,
                index_count,
                vertex_count: vertex_count as u32,
                target_count,
                has_cloth_alloc: has_cloth,
                has_cloth_normals_alloc: has_cloth_normals,
                has_skin_anchors_alloc: has_anchors,
                has_containment_alloc: has_containment,
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

    pub(super) fn get_or_update_skinning(
        &mut self,
        inst_idx: usize,
        skinning_mats: &[[[f32; 4]; 4]],
        mat_count: usize,
        memory_allocator: Arc<StandardMemoryAllocator>,
        ds_allocator: Arc<StandardDescriptorSetAllocator>,
        transform_pipeline: &Arc<ComputePipeline>,
    ) -> Result<Arc<DescriptorSet>, String> {
        // Grow the cache vector if needed.
        while self.skinning_cache.len() <= inst_idx {
            let buf = gpu_alloc::host_buffer(
                &memory_allocator,
                BufferUsage::STORAGE_BUFFER,
                vec![mat4_cols_identity()],
                "skinning buffer init",
            )?;

            let layout = transform_pipeline
                .layout()
                .set_layouts()
                .get(1)
                .ok_or("render: transform pipeline missing skinning set layout")?
                .clone();
            let ds = DescriptorSet::new(
                ds_allocator.clone(),
                layout,
                [WriteDescriptorSet::buffer(0, buf.clone())],
                [],
            )
            .map_err(|e| format!("render: skinning desc set init failed: {e}"))?;

            self.skinning_cache.push(SkinningCacheEntry {
                buffer: buf,
                descriptor_set: ds,
                capacity: 1,
            });
        }

        let entry = &mut self.skinning_cache[inst_idx];

        // If the buffer is too small, reallocate.
        if mat_count > entry.capacity {
            // The skinning descriptor set is about to be swapped — cached
            // command buffers binding the old set must not survive.
            self.cb_cache.clear();
            let buf = gpu_alloc::host_buffer(
                &memory_allocator,
                BufferUsage::STORAGE_BUFFER,
                skinning_mats.iter().copied(),
                "skinning buffer realloc",
            )?;

            let layout = transform_pipeline
                .layout()
                .set_layouts()
                .get(1)
                .ok_or("render: transform pipeline missing skinning set layout")?
                .clone();
            let ds = DescriptorSet::new(
                ds_allocator.clone(),
                layout,
                [WriteDescriptorSet::buffer(0, buf.clone())],
                [],
            )
            .map_err(|e| format!("render: skinning desc set realloc failed: {e}"))?;

            entry.buffer = buf;
            entry.descriptor_set = ds;
            entry.capacity = mat_count;

            return Ok(entry.descriptor_set.clone());
        }

        // Buffer is large enough — write in-place.
        {
            let mut guard = entry
                .buffer
                .write()
                .map_err(|e| format!("render: skinning buffer write failed: {e}"))?;
            guard[..mat_count].copy_from_slice(skinning_mats);
        }

        Ok(entry.descriptor_set.clone())
    }
}

/// Pack dense per-corner morph deltas into the sparse form consumed by
/// `transform_cs`. Pure function so the GPU layout invariants can be
/// unit-tested without a Vulkan device.
///
/// Format contract with `transform_cs`: per target, a run of entries
/// sorted by ascending vertex index (guaranteed by iterating the dense
/// arrays front-to-back). Entry stride 1 = one vec4
/// `(vertex, dx, dy, dz)`; targets that also carry normal deltas use
/// stride 2 with `(nx, ny, nz, _)` after each entry. `infos` records
/// `(begin, count, stride, _)` per target. The vertex index travels as
/// f32 — exact for values below 2^24, and mesh vertex counts are five
/// orders of magnitude under that.
pub(super) fn pack_sparse_morphs(
    targets: &[crate::asset::MorphTargetDelta],
) -> (Vec<[f32; 4]>, Vec<[u32; 4]>) {
    let mut entries: Vec<[f32; 4]> = Vec::new();
    let mut infos: Vec<[u32; 4]> = Vec::with_capacity(targets.len());
    for target in targets {
        let pos = &target.position_deltas;
        let nrm = &target.normal_deltas;
        let has_nrm = !nrm.is_empty();
        let stride: u32 = if has_nrm { 2 } else { 1 };
        let vertex_span = pos.len().max(if has_nrm { nrm.len() } else { 0 });
        let begin = entries.len() as u32;
        let mut record_count = 0u32;
        for vid in 0..vertex_span {
            let pd = pos.get(vid).copied().unwrap_or([0.0; 3]);
            let nd = if has_nrm { nrm.get(vid).copied() } else { None };
            let pos_nz = pd.iter().any(|&c| c.abs() > 1e-7);
            let nrm_nz = nd.is_some_and(|n| n.iter().any(|&c| c.abs() > 1e-7));
            if !pos_nz && !nrm_nz {
                continue;
            }
            entries.push([vid as f32, pd[0], pd[1], pd[2]]);
            if has_nrm {
                let n = nd.unwrap_or([0.0; 3]);
                entries.push([n[0], n[1], n[2], 0.0]);
            }
            record_count += 1;
        }
        // `count` is the number of vertex RECORDS (one record = `stride`
        // vec4s) — the shader's binary search bounds, not the vec4 total.
        infos.push([begin, record_count, stride, 0]);
    }
    (entries, infos)
}

#[cfg(test)]
mod tests {
    use super::pack_sparse_morphs;
    use crate::asset::MorphTargetDelta;

    /// Mirror of the transform_cs gather: binary-search `vid` inside the
    /// target's entry run and return `(position_delta, normal_delta)`.
    fn lookup(
        entries: &[[f32; 4]],
        info: &[u32; 4],
        vid: u32,
    ) -> Option<([f32; 3], Option<[f32; 3]>)> {
        let (base, count, stride) = (info[0] as usize, info[1] as usize, info[2] as usize);
        let (mut lo, mut hi) = (0usize, count);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if (entries[base + mid * stride][0] as u32) < vid {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo >= count || (entries[base + lo * stride][0] as u32) != vid {
            return None;
        }
        let e = entries[base + lo * stride];
        let nrm = if stride == 2 {
            Some(e_next(entries, base + lo * stride + 1))
        } else {
            None
        };
        Some(([e[1], e[2], e[3]], nrm))
    }

    fn e_next(entries: &[[f32; 4]], i: usize) -> [f32; 3] {
        [entries[i][0], entries[i][1], entries[i][2]]
    }

    fn target(name: &str, pos: Vec<[f32; 3]>, nrm: Vec<[f32; 3]>) -> MorphTargetDelta {
        MorphTargetDelta {
            name: name.to_string(),
            position_deltas: pos,
            normal_deltas: nrm,
        }
    }

    #[test]
    fn sparse_pack_skips_zero_and_keeps_ascending_order() {
        let targets = vec![target(
            "t0",
            vec![
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
            vec![],
        )];
        let (entries, infos) = pack_sparse_morphs(&targets);
        assert_eq!(infos, vec![[0, 3, 1, 0]]);
        // Only non-zero vertices appear, in ascending vertex order.
        let vids: Vec<u32> = entries.iter().map(|e| e[0] as u32).collect();
        assert_eq!(vids, vec![0, 2, 3]);
    }

    #[test]
    fn sparse_pack_round_trips_every_nonzero_delta() {
        let targets = vec![
            target(
                "pos_only",
                vec![[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -0.25, 0.0]],
                vec![],
            ),
            target(
                "pos_nrm",
                vec![[0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [0.0, 0.0, 0.9]],
                vec![[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            ),
            target("empty", vec![], vec![]),
            // normals-only target (positions accessor absent)
            target("nrm_only", vec![], vec![[0.0, 0.0, 0.0], [0.0, 7.0, 0.0]]),
        ];
        let (entries, infos) = pack_sparse_morphs(&targets);

        for (t, target) in targets.iter().enumerate() {
            let span = target.position_deltas.len().max(target.normal_deltas.len());
            for vid in 0..span {
                let pd = target.position_deltas.get(vid).copied().unwrap_or([0.0; 3]);
                let nd = target.normal_deltas.get(vid).copied();
                let pos_nz = pd.iter().any(|&c| c.abs() > 1e-7);
                let nrm_nz = nd.is_some_and(|n| n.iter().any(|&c| c.abs() > 1e-7));
                let hit = lookup(&entries, &infos[t], vid as u32);
                if !pos_nz && !nrm_nz {
                    assert!(
                        hit.is_none(),
                        "target {} vid {} should be skipped",
                        target.name,
                        vid
                    );
                } else {
                    let (got_pos, got_nrm) = hit.expect("non-zero delta must be findable");
                    assert_eq!(got_pos, pd, "target {} vid {} position", target.name, vid);
                    if !target.normal_deltas.is_empty() {
                        assert_eq!(got_nrm, Some(nd.unwrap_or([0.0; 3])));
                    } else {
                        assert_eq!(got_nrm, None);
                    }
                }
            }
        }
    }

    #[test]
    fn sparse_pack_reports_stride_per_target() {
        let targets = vec![
            target("a", vec![[1.0, 0.0, 0.0]], vec![]),
            target("b", vec![[1.0, 0.0, 0.0]], vec![[0.0, 1.0, 0.0]]),
        ];
        let (entries, infos) = pack_sparse_morphs(&targets);
        assert_eq!(infos[0][2], 1, "pos-only target uses stride 1");
        assert_eq!(infos[1][2], 2, "pos+normal target uses stride 2");
        assert_eq!(infos[1][1], 1, "one entry for one non-zero vertex");
        assert_eq!(entries.len(), 3, "1 entry + 1 entry×2 vec4s");
        assert_eq!(
            entries[2],
            [0.0, 1.0, 0.0, 0.0],
            "normal vec4 follows entry"
        );
    }
}
