//! Frame plan — the prepare/record split that backs the command-buffer
//! cache.
//!
//! `render` used to interleave two kinds of work while walking the instance
//! list: CPU-side writes into persistent UBOs/SSBOs (camera, skinning,
//! per-primitive control blocks, morph weights, cloth controls, material
//! uniforms) and Vulkan command recording. The interleaving forced a fresh
//! `AutoCommandBufferBuilder` + `build()` every frame — the single largest
//! CPU cost of a frame after the morph-gather compaction (~2 ms of
//! dependency-resolution work in `build` alone).
//!
//! [`VulkanRenderer::prepare_frame`] now performs every CPU write and
//! captures the frame's *dispatch structure* into a [`FramePlan`]: which
//! descriptor sets to bind, how many cloth substeps/iterations to run,
//! which draws to emit, which buffers to copy for readback. The shape is
//! summarised into `shape_key`, and `render` reuses the previously built
//! command buffer whenever the key matches — per-frame data reaches the
//! GPU exclusively through buffers the cached command buffer already
//! references (in-place UBO/SSBO writes completed before submit).
//!
//! Safety of the key rests on two invariants:
//!
//! 1. **Every command-buffer-visible identity is hashed.** Descriptor-set
//!    and pipeline `Arc` pointers, dispatch counts, push-constant bits
//!    (bloom intensity, outline width/colour), clear colour, readback ring
//!    slot. Buffer-only swaps (readback ring realloc on extent change)
//!    happen exclusively at sites that also clear the cache.
//! 2. **Cached command buffers pin their resources.** A built vulkano
//!    command buffer keeps every buffer/set/pipeline it references alive,
//!    so a resource's address can never be recycled while a key that
//!    contains it is still cached — pointer-identity keys cannot go stale
//!    through allocator reuse (no ABA).
//!
//! Animated per-frame values never enter the command buffer: the camera
//! lives in the `CameraRing` UBOs and the generative background's `time` /
//! tracking anchors moved from push constants to a small uniform ring
//! (see `background.rs`), which is precisely what lets a cached recording
//! keep animating.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use vulkano::buffer::Subbuffer;
use vulkano::descriptor_set::DescriptorSet;
use vulkano::image::Image;
use vulkano::pipeline::GraphicsPipeline;

use crate::renderer::frame_input::{self, RenderFrameInput};
use crate::renderer::pipeline::GpuVertex;
use crate::renderer::VulkanRenderer;

// Per-primitive draw record — the flat, dispatch-ordered list the scene
// pass consumes. `vertex_buffer` is the compute shader's output, so the
// graphics passes never see base / morph / cloth data — only the
// world-space vertices.
pub(super) struct DrawInfo {
    pub(super) pipeline: Arc<GraphicsPipeline>,
    pub(super) alpha_mode: frame_input::RenderAlphaMode,
    pub(super) vertex_buffer: Subbuffer<[GpuVertex]>,
    pub(super) index_buffer: Subbuffer<[u32]>,
    pub(super) index_count: u32,
    pub(super) material_set: Arc<DescriptorSet>,
    pub(super) outline: Option<(f32, [f32; 3])>,
}

/// GPU cloth dispatch structure for one primitive (Verlet integration +
/// XPBD substeps + normal recomputation), captured at prepare time so the
/// recording pass can replay it without touching renderer state. The
/// per-substep control UBO writes happen in prepare; every substep runs
/// with the same fixed-dt values, which is what makes a fixed recording
/// equivalent to today's interleaved write/dispatch loop.
pub(super) struct PlannedCloth {
    pub(super) verlet_set: Arc<DescriptorSet>,
    pub(super) groups: [u32; 3],
    pub(super) substeps: u32,
    pub(super) constraint_iters: u32,
    pub(super) constraints: Option<PlannedClothConstraints>,
    pub(super) normal: Option<PlannedClothNormal>,
    pub(super) collide: Option<PlannedClothCollide>,
    pub(super) selfcol: Option<PlannedClothSelfCol>,
}

pub(super) struct PlannedClothCollide {
    pub(super) set: Arc<DescriptorSet>,
    pub(super) collider_count: u32,
}

/// Self-collision pass pair: build (re-zero + atomic grid fill) then
/// resolve (neighbour push). `counts_ssbo` is the fill-buffer target
/// the recording half zeroes before every build dispatch.
pub(super) struct PlannedClothSelfCol {
    pub(super) build_set: Arc<DescriptorSet>,
    pub(super) resolve_set: Arc<DescriptorSet>,
    pub(super) counts_ssbo: Subbuffer<[u32]>,
}

pub(super) struct PlannedClothConstraints {
    pub(super) lambda_update_set: Arc<DescriptorSet>,
    pub(super) accumulate_set: Arc<DescriptorSet>,
    pub(super) apply_set: Arc<DescriptorSet>,
    pub(super) lambda_ssbo: Subbuffer<[f32]>,
    pub(super) constraint_count: u32,
    pub(super) constraint_groups: u32,
}

pub(super) struct PlannedClothNormal {
    pub(super) set: Arc<DescriptorSet>,
}

/// One instance's body-SDF splat dispatches, recorded after every
/// transform dispatch of that instance (they read the freshly skinned
/// vertices of each splatted primitive — body + face/head surfaces —
/// and accumulate into the shared field via atomicMin).
pub(super) struct PlannedSdf {
    /// One (set, groups) pair per splatted primitive.
    pub(super) dispatches: Vec<(Arc<DescriptorSet>, [u32; 3])>,
    /// The field buffer, sentinel-filled (`fill_buffer` u32::MAX) once
    /// before the dispatches — command-buffer-visible identity, hashed
    /// into the key.
    pub(super) field: Subbuffer<[u32]>,
}

/// One primitive's transform dispatch in Kahn order, with its optional
/// cloth prologue.
pub(super) struct PlannedPrim {
    pub(super) transform_set: Arc<DescriptorSet>,
    pub(super) groups: [u32; 3],
    pub(super) cloth: Option<PlannedCloth>,
}

pub(super) struct PlannedInstance {
    pub(super) skinning_set: Arc<DescriptorSet>,
    pub(super) prims: Vec<PlannedPrim>,
    pub(super) sdf: Option<PlannedSdf>,
}

/// Everything one frame's command buffer needs, minus the per-frame buffer
/// contents. Produced by [`VulkanRenderer::prepare_frame`] and consumed by
/// the recording path in `render`.
pub(super) struct FramePlan {
    pub(super) camera_set: Arc<DescriptorSet>,
    pub(super) outline_camera_set: Arc<DescriptorSet>,
    pub(super) bg_pipeline: Option<Arc<GraphicsPipeline>>,
    pub(super) bg_set: Option<Arc<DescriptorSet>>,
    /// Scene-pass clear colour (encodes `transparent_background`).
    pub(super) clear: [f32; 4],
    /// Per-instance compute structure in dispatch order.
    pub(super) instances: Vec<PlannedInstance>,
    /// Flat draw list in the same order the dispatches were planned.
    pub(super) draws: Vec<DrawInfo>,
    /// R3 containment history copies: (parent's transformed VBO →
    /// parent's history VBO), recorded after every transform dispatch.
    /// Command-buffer-visible identity, hence part of the shape key.
    pub(super) containment_copies:
        Vec<(Subbuffer<[GpuVertex]>, Subbuffer<[GpuVertex]>)>,
    /// R2 audit copies (final VBO -> host staging), only when
    /// `VULVATAR_VBO_AUDIT=1`. Also command-buffer-visible identity.
    pub(super) audit_copies: Vec<(Subbuffer<[GpuVertex]>, Subbuffer<[GpuVertex]>)>,
    pub(super) bloom: frame_input::BloomSettings,
    pub(super) use_bloom: bool,
    pub(super) composite_intensity: f32,
    pub(super) final_color_image: Arc<Image>,
    pub(super) staging_buffer: Subbuffer<[u8]>,
    pub(super) readback_buffer: Subbuffer<[u8]>,
    /// Ring slot the staging/readback pair came from — identifies the CPU
    /// pixel-Vec pool slot the harvest should refill.
    pub(super) readback_slot: usize,
    pub(super) depth_image: Option<Arc<Image>>,
    pub(super) depth_buffer: Option<Subbuffer<[u8]>>,
    /// Hash of every command-buffer-visible identity and value (see module
    /// docs). Equal keys ⇒ a cached recording is byte-equivalent to what
    /// re-recording would produce.
    pub(super) shape_key: u64,
}

fn hash_arc<T>(hasher: &mut DefaultHasher, arc: &Arc<T>) {
    hasher.write_usize(Arc::as_ptr(arc) as usize);
}

impl VulkanRenderer {
    /// Perform every per-frame CPU write (camera ring, background uniform
    /// ring, skinning, control UBOs, compacted morph weights, cloth
    /// controls, material uniforms) and capture the frame's dispatch
    /// structure into a [`FramePlan`] with its shape key.
    pub(super) fn prepare_frame(&mut self, input: &RenderFrameInput) -> Result<FramePlan, String> {
        let memory_allocator = self
            .memory_allocator
            .as_ref()
            .ok_or("renderer: no memory allocator")?
            .clone();
        let ds_allocator = self
            .descriptor_set_allocator
            .as_ref()
            .ok_or("renderer: no descriptor set allocator")?
            .clone();
        let transform_pipeline = self
            .transform_compute_pipeline
            .as_ref()
            .ok_or("renderer: no transform compute pipeline")?
            .clone();
        let gfx_pipeline = self
            .graphics_pipeline
            .as_ref()
            .ok_or("renderer: no graphics pipeline")?
            .clone();
        let default_tex = self
            .default_texture_view
            .as_ref()
            .ok_or("renderer: no default texture")?
            .clone();
        let sampler = self.sampler.as_ref().ok_or("renderer: no sampler")?.clone();

        // ── Camera UBO update (ring slot for this frame) ────────────────
        let ld = input.lighting.main_light_dir_ws;
        let ld_len = (ld[0] * ld[0] + ld[1] * ld[1] + ld[2] * ld[2])
            .sqrt()
            .max(1e-6);
        let light_dir = [ld[0] / ld_len, ld[1] / ld_len, ld[2] / ld_len];

        let camera_data = super::CameraUniform {
            view: super::mat4_to_cols(input.camera.view),
            proj: super::mat4_to_cols(input.camera.projection),
            camera_pos: input.camera.position_ws,
            _pad0: 0.0,
            light_dir,
            light_intensity: input.lighting.main_light_intensity,
            light_color: input.lighting.main_light_color,
            _pad1: 0.0,
            ambient_term: input.lighting.ambient_term,
            fade_opacity: input.avatar_opacity,
        };
        let ring_slot = (self.frame_counter % super::FRAME_LAG as u64) as usize;
        let (camera_set, outline_camera_set) = {
            let ring = self.camera_ring.as_ref().ok_or("render: no camera ring")?;
            {
                let mut guard = ring.buffers[ring_slot]
                    .write()
                    .map_err(|e| format!("render: camera buffer write failed: {e}"))?;
                *guard = camera_data;
            }
            (
                ring.main_sets[ring_slot].clone(),
                ring.outline_sets[ring_slot].clone(),
            )
        };

        // ── Generative background uniform (ring slot) ───────────────────
        let mut bg_pipeline = None;
        let mut bg_set = None;
        if input.generative_background.enabled {
            let pipeline = self
                .background_pipeline
                .as_ref()
                .ok_or("render: no background pipeline")?
                .clone();
            let push = super::background::build_push_constants(input);
            let ring = self
                .bg_uniform_ring
                .as_ref()
                .ok_or("render: no background uniform ring")?;
            let slot =
                (self.frame_counter % super::background::BG_UNIFORM_RING_SIZE as u64) as usize;
            bg_set = Some(ring.write_slot(slot, push)?);
            bg_pipeline = Some(pipeline);
        }

        // ── Compute prepass CPU writes + dispatch-structure capture ─────
        let (instances, draws, containment_copies, audit_copies) = self.prepare_compute_prepass(
            input,
            &memory_allocator,
            &ds_allocator,
            &transform_pipeline,
            &gfx_pipeline,
            &default_tex,
            &sampler,
        )?;

        // ── Scene clear colour ──────────────────────────────────────────
        let clear = if input.transparent_background {
            [0.0_f32, 0.0, 0.0, 0.0]
        } else {
            let [r, g, b] = input.background_color;
            [r, g, b, 1.0]
        };

        // ── Post-effect capture ─────────────────────────────────────────
        let use_bloom;
        let final_color_image;
        {
            let post = self
                .post_effects
                .as_ref()
                .ok_or("renderer: no post-effect resources")?;
            use_bloom = input.bloom.enabled && post.has_bloom_chain();
            final_color_image = post.final_color.clone();
        }
        let composite_intensity = if use_bloom {
            input.bloom.intensity
        } else {
            0.0
        };

        // ── Readback ring + optional depth aspect ───────────────────────
        let (readback_slot, staging_buffer, readback_buffer) =
            self.ensure_readback_buffers(self.current_extent)?;
        let depth = if self.depth_readback_enabled && self.current_sample_count == 1 {
            match self.offscreen_depth.clone() {
                Some(depth_image) => {
                    let buf = self.ensure_depth_readback_buffer(self.current_extent)?;
                    Some((depth_image, buf))
                }
                None => None,
            }
        } else {
            None
        };
        let (depth_image, depth_buffer) = match depth {
            Some((image, buf)) => (Some(image), Some(buf)),
            None => (None, None),
        };

        // ── Shape key ───────────────────────────────────────────────────
        let mut plan = FramePlan {
            camera_set,
            outline_camera_set,
            bg_pipeline,
            bg_set,
            clear,
            instances,
            draws,
            containment_copies,
            audit_copies,
            bloom: input.bloom.clone(),
            use_bloom,
            composite_intensity,
            final_color_image,
            staging_buffer,
            readback_buffer,
            readback_slot,
            depth_image,
            depth_buffer,
            shape_key: 0,
        };
        plan.shape_key = self.compute_shape_key(&plan);
        Ok(plan)
    }

    /// Hash every command-buffer-visible identity and value of the plan
    /// (see module docs for the safety argument).
    fn compute_shape_key(&self, plan: &FramePlan) -> u64 {
        let mut h = DefaultHasher::new();
        h.write_u32(self.current_extent[0]);
        h.write_u32(self.current_extent[1]);

        hash_arc(&mut h, &plan.camera_set);
        hash_arc(&mut h, &plan.outline_camera_set);
        match (&plan.bg_pipeline, &plan.bg_set) {
            (Some(p), Some(s)) => {
                h.write_u8(1);
                hash_arc(&mut h, p);
                hash_arc(&mut h, s);
            }
            _ => h.write_u8(0),
        }
        for c in plan.clear {
            h.write_u32(c.to_bits());
        }

        for inst in &plan.instances {
            hash_arc(&mut h, &inst.skinning_set);
            for prim in &inst.prims {
                hash_arc(&mut h, &prim.transform_set);
                h.write_u32(prim.groups[0]);
                // Body-SDF splat resources are bound by the recorded
                // command buffer too — same command-buffer-visible
                // identity rule as the cloth sets above.
                if let Some(sdf) = &inst.sdf {
                    h.write_u8(1);
                    h.write_usize(sdf.dispatches.len());
                    for (set, groups) in &sdf.dispatches {
                        hash_arc(&mut h, set);
                        h.write_u32(groups[0]);
                    }
                    sdf.field.hash(&mut h);
                } else {
                    h.write_u8(0);
                }
                if let Some(cloth) = &prim.cloth {
                    h.write_u8(1);
                    hash_arc(&mut h, &cloth.verlet_set);
                    h.write_u32(cloth.groups[0]);
                    h.write_u32(cloth.substeps);
                    h.write_u32(cloth.constraint_iters);
                    if let Some(cs) = &cloth.constraints {
                        h.write_u8(1);
                        hash_arc(&mut h, &cs.lambda_update_set);
                        hash_arc(&mut h, &cs.accumulate_set);
                        hash_arc(&mut h, &cs.apply_set);
                        h.write_u32(cs.constraint_count);
                        h.write_u32(cs.constraint_groups);
                    } else {
                        h.write_u8(0);
                    }
                    if let Some(n) = &cloth.normal {
                        h.write_u8(1);
                        hash_arc(&mut h, &n.set);
                    } else {
                        h.write_u8(0);
                    }
                    // S2.2/S2.3 collision resources are bound by the
                    // recorded command buffer too — their (re)allocation
                    // must invalidate the cached recording. (R1 moved
                    // collision INTO the substep loop; these sets are now
                    // dispatched substeps × per frame.)
                    if let Some(c) = &cloth.collide {
                        h.write_u8(1);
                        hash_arc(&mut h, &c.set);
                        h.write_u32(c.collider_count);
                    } else {
                        h.write_u8(0);
                    }
                    if let Some(sc) = &cloth.selfcol {
                        h.write_u8(1);
                        hash_arc(&mut h, &sc.build_set);
                        hash_arc(&mut h, &sc.resolve_set);
                        sc.counts_ssbo.hash(&mut h);
                    } else {
                        h.write_u8(0);
                    }
                    // R3 containment history copies are
                    // command-buffer-visible buffer identities.
                    h.write_usize(plan.containment_copies.len());
                    for (src, dst) in &plan.containment_copies {
                        src.hash(&mut h);
                        dst.hash(&mut h);
                    }
                    // R2 audit copies likewise.
                    h.write_usize(plan.audit_copies.len());
                    for (src, dst) in &plan.audit_copies {
                        src.hash(&mut h);
                        dst.hash(&mut h);
                    }
                } else {
                    h.write_u8(0);
                }
            }
        }

        for draw in &plan.draws {
            hash_arc(&mut h, &draw.pipeline);
            h.write_u8(match draw.alpha_mode {
                frame_input::RenderAlphaMode::Opaque => 0,
                frame_input::RenderAlphaMode::Blend => 1,
                frame_input::RenderAlphaMode::Cutout => 2,
            });
            h.write_u32(draw.index_count);
            hash_arc(&mut h, &draw.material_set);
            match draw.outline {
                Some((w, color)) => {
                    h.write_u8(1);
                    h.write_u32(w.to_bits());
                    for c in color {
                        h.write_u32(c.to_bits());
                    }
                }
                None => h.write_u8(0),
            }
        }

        h.write_u8(u8::from(plan.use_bloom));
        h.write_u32(plan.bloom.threshold.to_bits());
        h.write_u32(plan.composite_intensity.to_bits());
        if let Some(post) = self.post_effects.as_ref() {
            hash_arc(&mut h, &post.pipe_composite);
            if plan.use_bloom {
                hash_arc(&mut h, &post.pipe_down);
                hash_arc(&mut h, &post.pipe_up);
            }
        }
        h.write_usize(plan.readback_slot);
        h.write_u8(u8::from(plan.depth_buffer.is_some()));
        h.finish()
    }
}
