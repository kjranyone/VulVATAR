# Transform compute prepass migration

Date: 2026-05-19
Branch: `feature/compute-prepass-migration`

## Goal

Move skinning, morph-target blending, and cloth deformation out of the
graphics vertex shaders and into a single compute prepass per frame. The
forward / outline / thumbnail vertex shaders shrink to "apply camera view
+ projection" — every other transform is resolved compute-side.

This is the next step of the `gpu-runtime-roadmap` direction: large
frame-shaped data stays on the GPU, the CPU only writes compact control
data.

## Status

`pipeline.rs` ships the full compute scaffolding; `mod.rs` still pulls
morph + skinning inside the graphics vertex shaders. `initialize()`
already calls `pipeline::create_transform_compute_pipeline()` so the
shader compiles and the descriptor set layouts validate against the
live device — there will be no surprises at runtime when the dispatch
is wired up.

| Layer | Migration state |
|---|---|
| `pipeline.rs` shaders / types / pipeline-creation fn | DONE (scaffolding) |
| `pipeline::transform_cs` GLSL | DONE |
| `pipeline::create_transform_compute_pipeline` | DONE |
| `mod.rs` `transform_compute_pipeline` field + `initialize()` build | DONE |
| `GpuVertex` layout flip to 4-component | OPEN |
| `vs` / `outline_vs` thin-out | OPEN |
| `fs` material set renumber (set 2 → set 1) | OPEN |
| `material.rs` `set_layouts().get(2)` → `.get(1)` | OPEN |
| `mod.rs` cache restructure | OPEN |
| `mod.rs` `render()` compute prepass + thin graphics | OPEN |
| `mod.rs` `render_to_thumbnail()` parallel rewrite | OPEN |
| Helper deletions (morph_descriptor_for_draw, etc.) | OPEN |

Estimated size of the open work: ~500 lines of net change in
`src/renderer/mod.rs`, smaller deltas in `pipeline.rs`, `material.rs`,
and the three `src/bin/diagnose_*.rs` / `render_matrix.rs` that print
`GpuVertex::per_vertex()` for diagnostics.

## Architecture target

```text
Frame:
  harvest_pending_readback()                          [unchanged]
  update camera UBO ring                              [unchanged]

  bind transform_compute_pipeline                     [new]
  for each (instance, primitive):                     [new]
    ensure_transform_data(...)                        [new helper]
    write morph weights + has_cloth flags into        [new]
      control_ubo (host-mapped)
    if has_cloth_alloc and cloth.version changed:     [new]
      write cloth.deformed_positions into             [new]
        cloth_pos_ssbo (host-mapped)
      write cloth.deformed_normals into               [new]
        cloth_norm_ssbo (host-mapped, when present)
    bind transform_set (set 0)                        [new]
    bind skinning_set (set 1, allocated against       [new]
      transform_pipeline.set_layouts()[1])
    dispatch([ceil(vertex_count / 64), 1, 1])         [new]
    collect DrawInfo for graphics

  begin_render_pass                                   [moved later]
  forward graphics:
    bind camera (set 0)                               [unchanged]
    for each DrawInfo:
      bind variant pipeline                           [unchanged]
      bind material (set 1, was set 2)                [renumbered]
      bind transformed_vbo as vertex buffer
      draw_indexed
  outline graphics:
    bind outline pipeline + camera                    [unchanged]
    for each DrawInfo with outline:
      push_constants + draw                           [no more set 1/3]

  end_render_pass + readback + submit                 [unchanged]
```

Vulkano auto-tracks the storage-write → vertex-input-read barrier on
`transformed_vbo` when the compute dispatch and the graphics draw share
the same command buffer, so no explicit barriers are needed inside
`record_*` calls.

## Step-by-step plan

### Step 1 — `pipeline.rs` shader layout flip

Replace `GpuVertex` with the 4-component layout (16+16+8+8 = 48 B):

```rust
pub struct GpuVertex {
    #[format(R32G32B32A32_SFLOAT)] pub position: [f32; 4], // world-space, .xyz used
    #[format(R32G32B32A32_SFLOAT)] pub normal:   [f32; 4], // world-space, .xyz used
    #[format(R32G32_SFLOAT)]       pub uv:       [f32; 2],
    #[format(R32G32_UINT)]         pub _pad:     [u32; 2],
}
```

Rewrite `pub mod vs` source to drop set 1 (skinning) + set 3 (morph)
and read `vec4 position / vec4 normal / vec2 uv / uvec2 _pad`:

```glsl
void main() {
    frag_world_pos = position.xyz;
    frag_normal    = normal.xyz;
    frag_uv        = uv;
    gl_Position    = camera.proj * camera.view * vec4(position.xyz, 1.0);
}
```

Rewrite `pub mod outline_vs` similarly — keep the push constant, drop
the skinning + morph descriptors, work in world space throughout.

Renumber `MaterialData` / `base_texture` / `matcap_texture` /
`shade_texture` from `set = 2` to `set = 1` in `pub mod fs`.

### Step 2 — `material.rs`

```rust
let layout = pipeline.layout().set_layouts().get(1).ok_or(...)?;
```

Update the error message accordingly.

### Step 3 — `mod.rs` cache restructure

Delete:

- `MorphInfo` / `MORPH_WEIGHTS_VEC4_COUNT` (subsumed by
  `pipeline::TransformControl`)
- `MorphGpuData` + `morph_cache` + `stub_morph_data`
- `ClothMeshSlot` + `cloth_cache`
- `mesh_cache` (its role moves into `transform_cache`)

Add:

```rust
struct TransformGpuData {
    base_ssbo:        Subbuffer<[pipeline::GpuVertexBase]>,
    index_buffer:     Subbuffer<[u32]>,
    transformed_vbo:  Subbuffer<[GpuVertex]>,
    control_ubo:      Subbuffer<pipeline::TransformControl>,
    morph_deltas:     Subbuffer<[[f32; 4]]>,  // real or stub
    cloth_pos_ssbo:   Subbuffer<[[f32; 4]]>,  // real or stub
    cloth_norm_ssbo:  Subbuffer<[[f32; 4]]>,  // real or stub
    transform_set:    Arc<DescriptorSet>,
    index_count:      u32,
    vertex_count:     u32,
    target_count:     u32,
    has_cloth_alloc:  bool,
    has_cloth_normals_alloc: bool,
    last_cloth_version: Option<u64>,
}

transform_cache:     HashMap<(MeshId, PrimitiveId), TransformGpuData>,
stub_storage_ssbo:   Option<Subbuffer<[[f32; 4]]>>,
```

The `transform_compute_pipeline` field already exists.

### Step 4 — `mod.rs` new helpers

`ensure_stub_ssbo(memory_allocator)` — lazily allocate a shared
1-element zero SSBO for primitives that have neither morph deltas nor
cloth resources. Storage usage; host-sequential-write so it lives next
to other dynamic resources.

`ensure_transform_data(mesh_id, prim_id, prim, has_cloth, has_cloth_normals, memory_allocator, ds_allocator, transform_pipeline)`:

- Early-return if the cache slot exists.
- Allocate `base_ssbo` from `pipeline::vertex_data_to_base(vd)` (rename
  the existing `vertex_data_to_gpu` and switch its output type to
  `Vec<pipeline::GpuVertexBase>`).
- Allocate index buffer the same way `upload_mesh` does (fill with
  `(0..n)` when `prim.indices` is empty).
- Allocate `transformed_vbo` with `Buffer::new_slice::<GpuVertex>` and
  `VERTEX_BUFFER | STORAGE_BUFFER` usage, `PREFER_DEVICE` memory.
- Pack `prim.morph_targets` into `morph_deltas` (cap at
  `MORPH_MAX_TARGETS`, log + clamp on overflow, layout `[pos, norm,
  pos, norm, ...]` of `vec4` per `(t, v)`). When the primitive has no
  morph targets, use the stub SSBO.
- Allocate `cloth_pos_ssbo` / `cloth_norm_ssbo` sized to
  `vertex_count` when `has_cloth` / `has_cloth_normals`; otherwise use
  the stub.
- Allocate `control_ubo` initialised with `vertex_count`,
  `target_count`, zeroed flags + weights.
- Allocate descriptor set 0 against
  `transform_pipeline.layout().set_layouts()[0]` with bindings:

  | binding | resource |
  |---|---|
  | 0 | `base_ssbo` |
  | 1 | `morph_deltas` |
  | 2 | `cloth_pos_ssbo` |
  | 3 | `cloth_norm_ssbo` |
  | 4 | `control_ubo` |
  | 5 | `transformed_vbo` |

- Bump `gpu_runtime_counters.morph_gpu_resource_creations` (and
  `.cloth_cache_creations` when `has_cloth`).

### Step 5 — `mod.rs` skinning move

`get_or_update_skinning` already allocates against
`gfx_pipeline.layout().set_layouts().get(1)`. Move the allocation to
`transform_pipeline.layout().set_layouts().get(1)` — same shape, but
the live consumer is now the compute shader.

### Step 6 — `mod.rs` `render()` rewrite

Replace the entire per-instance graphics loop + outline pass with the
shape described in the *Architecture target* section above. Concrete
points worth flagging:

- `begin_render_pass` moves *after* the compute prepass loop.
- The per-draw `material_set` upload can run inside the compute loop
  (same `upload_to_gpu` call, but pass `&gfx_pipeline` since every
  graphics variant derives the same set 1 layout). Store the resulting
  `Arc<DescriptorSet>` in `DrawInfo` and bind it during graphics.
- The forward + outline passes no longer bind set 1 (skinning) or set
  3 (morph). Material moves to set 1.
- The 60-frame `GPU_RUNTIME` log line should report
  `transform_cache.len()` instead of the legacy
  `morph_cache.len() + cloth_cache.len()`. Field names on
  `GpuRuntimeCounters` can stay as-is — repurpose:
  - `morph_gpu_resource_creations` → `TransformGpuData` creations
  - `morph_ubo_writes`             → `control_ubo` writes
  - `cloth_vbo_writes`             → `cloth_pos_ssbo` writes
  - `cloth_cache_creations`        → cloth-bearing slot creations

### Step 7 — `mod.rs` `render_to_thumbnail()`

Same migration. The thumbnail pipeline auto-derives the same set 1
layout from the shared vs/fs, so the compute prepass and the graphics
pass can re-use the per-primitive descriptor sets allocated against
`transform_compute_pipeline.set_layouts()[0]` and
`gfx_pipeline.set_layouts()[1]`. If the thumbnail render needs its
own dispatch path, prefer that over duplicating the compute setup.

### Step 8 — Delete dead helpers

After `render()` and `render_to_thumbnail()` stop calling them:

- `morph_descriptor_for_draw`
- `ensure_stub_morph_data`
- `init_morph_gpu_data`
- `upload_or_update_cloth_mesh`
- `upload_mesh`
- `build_cloth_vertices`
- `create_placeholder_mesh` (primitives without vertex data simply
  skip the draw in the new path)
- `create_vertex_buffer` / `create_index_buffer` (only the cached path
  allocates buffers now)

### Step 9 — `src/bin/*` smoke

`render_matrix.rs`, `diagnose_vrm.rs`, `diagnose_target_pose.rs` print
`GpuVertex::per_vertex()` for layout introspection. The new layout
prints fine, but verify the binaries still build under
`cargo check --bin <name>`.

## Validation

- `cargo clippy -- -D warnings` clean.
- `cargo test --lib` 185+ pass.
- `cargo run` with a VRM + tracking webcam ON: avatar animates, face
  expression applies (morph), cloth-equipped overlays deform.
- `GPU_RUNTIME` log every 60 frames shows steady-state counters
  (`transform_resources` matches primitive count after warmup,
  `ubo_writes` ≈ frame_counter × visible primitives,
  `cloth_writes` only ticks when a cloth-bearing primitive's
  `version` changes).
- Thumbnail render in the avatar library matches the live preview.

## Risks / things to keep an eye on

- **std430 padding**: the `GpuVertexBase` Rust struct must match
  byte-for-byte against the GLSL `struct VertexBase` in
  `transform_cs`. Both are documented in `pipeline.rs`; the existing
  scaffolding is the authoritative reference.
- **Stub SSBO size**: vulkano refuses zero-element storage buffers, so
  the shared stub must hold at least one `vec4`. The compute shader
  never reads past index 0 when the relevant flag is 0.
- **Per-primitive descriptor set lifetime**: `transform_set` references
  `cloth_pos_ssbo` / `cloth_norm_ssbo`. If a primitive transitions
  from non-cloth to cloth-bearing at runtime (rare but possible via
  user toggle), the slot's descriptor set still points at the stub —
  evict the slot from `transform_cache` and recreate. Easiest place to
  detect: in `ensure_transform_data`, when `has_cloth_alloc !=
  has_cloth` for an existing slot.
- **Outline alpha gating**: keep the existing rule — primitives with
  `Blend` or `Cutout` alpha modes never appear in the outline pass.
- **Material descriptor set sharing**: every forward graphics variant
  derives the same set 1 layout from the shared `vs` / `fs`. Allocating
  the material descriptor set against the default `gfx_pipeline` and
  binding it during any variant is safe because set layouts are
  structurally identical (Vulkan descriptor set compatibility rules).

## Commit shape

One commit per logical step is fine if the build stays green at each
boundary. The atomic landing order that compiles at every step is:

1. Step 1 + Step 2 + Step 5 (vs/fs/outline_vs + material.rs +
   skinning allocation source) — graphics pipeline layout shape
   changes, mod.rs must already be aware.
2. Step 3 + Step 4 (cache restructure + helpers) — new infrastructure
   in place.
3. Step 6 + Step 7 (render + thumbnail rewrite) — switches over.
4. Step 8 (deletes) — cleanup, all references removed.
5. Step 9 (bin smoke) — verify.

If the migration lands as a single commit instead, the only constraint
is that all of Step 1 + Step 3 + Step 6 happen together — anything in
between leaves the descriptor set indices mismatched between shader
and render loop.
