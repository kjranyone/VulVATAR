# Vulkano Renderer Design

## Purpose

The renderer reference: design, module layout, API boundary, material
strategy, MToon compatibility status, and validation checklist.
(Absorbed the former `renderer-module-design.md`,
`renderer-api-contract.md`, `rendering-materials.md`,
`mtoon-compatibility.md`, and `mtoon-test-matrix.md`.)

Related documents:

- [architecture.md](/C:/lib/github/kjranyone/VulVATAR/docs/architecture.md)
- [output-interop.md](/C:/lib/github/kjranyone/VulVATAR/docs/output-interop.md)
- [shader-implementation-notes.md](/C:/lib/github/kjranyone/VulVATAR/docs/shader-implementation-notes.md)

## Core Rule

The renderer owns GPU objects and draw submission. It does not own
avatar truth, simulation truth, or tracking truth. It consumes resolved
frame snapshots and turns them into images.

## Module Layout and Dependency Direction

`src/renderer/` is organized by responsibility (no `types.rs` /
`utils.rs` / `common.rs` dumping grounds):

- `mod.rs` — `VulkanRenderer`, `RenderResult`, coordination
- `frame_input.rs` — snapshot types (`RenderFrameInput`,
  `RenderAvatarInstance`, `CameraState`, `LightingState`,
  `OutputTargetRequest`); enforces the snapshot boundary
- `material.rs` — `MaterialShaderMode`, `MaterialUploadRequest`,
  normalization of imported materials into renderer-ready payloads
- `pipeline.rs` — pipeline families, variant selection, cache keys,
  shader sources (including `transform_cs`)
- `output_export.rs` — export boundary (`OutputFrame` /
  `GpuFrameToken` production)
- later extractions: `cloth_cache.rs`, `transform_cache.rs`,
  `readback.rs`, `texture_cache.rs`, `thumbnail.rs`, `debug.rs`,
  `frame_pool.rs`, `gpu_handle.rs`

Dependency direction inside the domain:

- `mod.rs -> frame_input, material, pipeline, output_export`
- `material -> asset` is allowed (normalization input)
- avoid `frame_input -> output_export`, `pipeline -> app`,
  `output_export -> avatar`

## Public API Contract

```rust
pub struct VulkanRenderer { /* renderer-owned state */ }

impl VulkanRenderer {
    pub fn new() -> Self;
    pub fn initialize(&mut self);
    pub fn render(&mut self, input: &RenderFrameInput) -> RenderResult;
}
```

Snapshot in, result out. Rules:

- the renderer consumes immutable frame snapshots, never mutable
  `AvatarInstance` state — this keeps ownership clear, enables the
  render thread, and keeps the boundary testable
- `RenderFrameInput` carries renderer-facing data only: no editor tool
  state, no tracking mailbox state
- `RenderResult` (extent, timestamp, has_alpha, stats,
  `exported_frame`) never leaks raw GPU ownership without a token
  contract — output routing consumes `ExportedFrame` / `GpuFrameToken`,
  not renderer-owned images
- app code never constructs pipeline internals or descriptor layouts

Field-level shapes live in `src/renderer/frame_input.rs`; the code is
the authoritative reference.

## Pass Structure

1. transform compute prepass
2. main forward avatar pass
3. outline pass
4. output resolve / export step

### Transform Compute Prepass

Runs once per (instance, primitive) before the graphics passes start.
Fuses dual quaternion skinning, morph-target blending, and cloth
override into a single dispatch, writing world-space `GpuVertex`
records to a persistent SSBO that the graphics passes bind as their
vertex buffer.

The graphics vertex shaders are reduced to view × projection; they
never see the source `GpuVertexBase` or the morph / cloth buffers. See
`pipeline::transform_cs` for the shader contract (landed on
`feature/compute-prepass-migration`, 2026-05-19).

### Main Forward Avatar Pass

Draws opaque and alpha-tested avatar surfaces from the prepass output,
evaluating the selected material mode.

### Outline Pass

Renders material-driven outlines with its own raster state; stays
separate from generic material shading logic. See "Implementation
Lessons" and the stencil-masking section of
shader-implementation-notes.md.

### Output Resolve / Export Step

Prepares the final color target, preserves the alpha contract, and
exports the image when the selected output path needs it. The
renderer produces an `OutputFrame` / `GpuFrameToken` only after all
rendering work for the frame is complete; the export boundary defines
the image, format, alpha mode, synchronization primitive, and when the
image may be reused.

## Descriptor Layout

Two pipeline-bind-points, two independent layouts:

### Compute (`pipeline::transform_cs`)

- set 0 — per-primitive transform resources, allocated by
  `ensure_transform_data` and pinned for the slot's lifetime:
  - binding 0: `BaseVertices` (`GpuVertexBase[]` SSBO)
  - binding 1: `MorphDeltas` (`vec4[]` SSBO; shared stub when the
    primitive has no morph targets)
  - binding 2: `ClothPositions` (`vec4[]` SSBO; shared stub when the
    primitive isn't cloth-bearing)
  - binding 3: `ClothNormals` (`vec4[]` SSBO; shared stub when no cloth
    normals)
  - binding 4: `TransformControl` UBO (`vertex_count`, `target_count`,
    `has_cloth`, `has_cloth_normals`, packed morph `weights[64]`)
  - binding 5: `OutVertices` (`GpuVertex[]` SSBO — the prepass write
    target, also bound as the graphics vertex buffer)
- set 1 — per-instance skinning data, allocated by
  `get_or_update_skinning`:
  - binding 0: `SkinningData` (`mat4[]` SSBO)

### Graphics (forward + outline)

- set 0 — frame / scene data
  - binding 0: camera + lighting UBO
- set 1 — material data
  - binding 0: normalized material parameter UBO
  - binding 1: base color texture + sampler
  - binding 2: matcap texture + sampler
  - binding 3: shade ramp texture + sampler

The outline pipeline omits set 1; it uses only set 0 and a push
constant for outline width / colour.

## Skinning Strategy

Dual quaternion skinning (DQS) runs inside the transform compute
prepass, not the graphics vertex shader. Inputs: four weights and four
joint indices per vertex in `GpuVertexBase`, one skinning matrix SSBO
per `AvatarInstance` bound to compute set 1.

Reasons for skinning compute-side:

- a single compute dispatch produces world-space vertices that every
  graphics pass shares (forward + outline + thumbnail / debug),
  avoiding redundant skinning math per pass
- the graphics shaders shrink to view × projection
- skinning, morph blend, and cloth override fuse into a single
  dispatch instead of three independent shader paths

Algorithm details (Kavan 2007, antipodal handling, rigid-matrix
assumption) live in shader-implementation-notes.md.

## Cloth Deformation Strategy

The renderer treats cloth deformation as an input buffer it consumes,
not as simulation logic it owns.

- app thread resolves cloth deformation output and publishes a
  `Vec<ClothDeformSnapshot>` per `RenderAvatarInstance`. Each snapshot
  carries `target_primitive_id` + `target_mesh_id` (the primitive it
  applies to), `vertex_offset` / `vertex_count` (the subset of that
  primitive it covers), and monotonic `version`.
- renderer matches each snapshot against the primitive it targets and
  rewrites the primitive's `cloth_pos_ssbo` / `cloth_norm_ssbo` in
  place whenever `version` advances since the last frame. Vertices
  outside the subset range stay at the primitive's rest pose; vertices
  of primitives that are not the snapshot's target keep `has_cloth = 0`
  and bind the shared stub SSBO.
- the transform compute prepass picks the cloth values via the
  per-primitive `has_cloth` / `has_cloth_normals` flags in
  `TransformControl`, applying skinning on top so cloth-bearing
  primitives still ride the avatar's root transform.

Cloth targets are resolved at `init_cloth_sim` /
`init_cloth_overlay` time from `ClothAsset::render_bindings[0]` —
which carries `primitive: PrimitiveRef`, `mesh: Option<MeshRef>`, and
`vertex_subset` — and cached on `ClothState`. `app/render.rs` reads
those cached fields when building the snapshot list, so the simulation
boundary stays clean: the renderer never walks back into the asset
graph.

## Material Strategy

Three layers, normalized strictly in this order:

1. **imported material description** — raw glTF / VRM (MToon) data
2. **normalized renderer material data** — shader mode selection,
   normalized parameters, texture bindings, alpha / culling flags,
   outline flags. The normalization step (`material.rs`) absorbs unit
   differences, naming differences, and unsupported-field fallbacks.
3. **GPU-uploaded material data** — shaped by renderer constraints

The renderer never decodes raw VRM extension schema while drawing.

Normalized modes: `Unlit`, `SimpleLit`, `ToonLike` (the importer
chooses; the GUI exposes a debugging override). The staged roadmap
(Unlit → SimpleLit → ToonLike → Outline → broader compatibility) is
fully landed; new material work extends the normalized contract rather
than adding modes ad hoc.

Alpha policy is explicit per material: `Opaque`, `Cutout` (alpha
test), `Blend`. Culling intent (`Back`, `None`, `Front`) is surfaced
by normalization, never decided inside shaders. Texture slots are
explicit: base color, shade ramp, matcap, (emissive parameters in the
UBO).

Outline is a renderer feature layered onto materials — outline enable
/ width / color come from the material contract, but the
implementation is a dedicated pass, not an imported material family.

## MToon Compatibility

Target: stable MToon-like presentation first; selective compatibility
for parameters that affect avatar readability; closer parity only
where it materially improves real avatars. Judge against real
representative avatars, not parameter tables. Face and hair quality
are the primary compatibility checks, not edge cases.

### Implemented

- base color factor / texture
- shade color factor / texture, shade shift / shade toony
- emissive color / texture
- rim color, fresnel power, lift, lighting mix
- alpha modes (opaque, mask, blend), double-sided rendering
- outline width / color (stencil-masked)
- ToonLike ambient floor (`ambient_avg` added into the lit term —
  see `pipeline.rs` fragment source; the early "toon path lacks
  ambient" issue is fixed)

### Disabled / partial

- **Matcap**: the shader path exists (multiplicative mix gated on
  `matcap_blend > 0.001`) but normalization hard-codes
  `matcap_blend = 0.0` (`material.rs`), because VRoid placeholder
  `MatcapWarp` textures (8×8) over-brighten avatars when treated as
  direct color maps. Re-enable by mapping the imported matcap factor
  once a proper MToon matcap sampling implementation is in place.
- **UV animation**: scroll / rotation speeds are uploaded but not
  time-multiplied; rendering is static.

### Known visual notes

- VRoid Studio exports warm/brown `shade_color` for skin materials —
  intentional MToon behavior, reads as "too dark" only when the
  ambient term is mis-tuned.

## Transparency Strategy

Explicit buckets: opaque, alpha test, alpha blend. The renderer
defines draw-order rules for blend materials, whether outlines apply
to blended materials, and how alpha writes are preserved for output
(OBS compositing depends on correct alpha). Never leave transparency
as an accidental consequence of draw order.

## Resource Ownership and Frame Discipline

The renderer owns: swapchain images, offscreen color targets, depth
targets, exportable output images, descriptor allocators, pipeline
caches, GPU staging resources. External systems receive exported
handles or tokens only.

No formal frame-graph system, but: passes have explicit input/output
targets, image transitions are explicit, export happens after
rendering completes.

Preview and export share one offscreen target: render offscreen,
present that target to the preview path, export it (or a resolved
derivative) to the output path. This keeps preview and output visually
identical. Resize rebuilds swapchain-dependent targets and offscreen
targets; it never re-imports avatars or rebuilds CPU-side material
definitions.

Pipeline cache keys include material mode, alpha mode, cull mode, and
pass; avoid runtime pipeline creation after warmup. (MSAA changes are
the sanctioned exception: the renderer rebuilds the render pass +
pipelines on an `OutputTargetRequest.msaa` change, clamped to device
support.)

## Debug Features

Optional overlays, never special cases inside production material
logic: skeleton debug lines, collider debug shapes, cloth simulation
mesh overlay, material mode override, normal / UV visualization. See
shader-implementation-notes.md for the implemented `debug_view` modes
and the `diagnose_vrm` binary.

## Implementation Lessons

### Depth+Stencil Format

The render pass uses `D32_SFLOAT_S8_UINT` (not plain `D32_SFLOAT`).
The stencil bits are required for outline masking. All depth image
creation, clear values, and framebuffer attachments must match this
format. Clear values use `ClearValue::DepthStencil((1.0, 0))`, not
`1.0f32.into()`.

### Blend Pipeline Variants

Each cull mode (Back, Front, None) needs separate Opaque and Blend
pipeline variants. Blend pipelines differ in
`depth_write_enable: false` (prevents blend ordering artifacts) and
`blend: Some(AttachmentBlend::alpha())`. The render loop iterates
`blend_pass in [false, true]` and skips non-matching meshes.

### Front Face Winding

glTF/VRM uses counter-clockwise front face winding
(`FrontFace::CounterClockwise`), Vulkan's default. Do NOT change to
`Clockwise` — doing so causes back-face textures to render on the
front of the body (reversed legs, neck, etc.). All pipelines (main,
outline, depth-only) must use the same convention.

### Outline and Alpha-Masked Materials

The outline shader does not perform alpha testing. For alpha-masked
(Cutout) materials, the outline pass would draw in regions the main
pass discarded via alpha cutoff, creating dark jagged artifacts beyond
the visible mesh boundary. Outline collection must be skipped for
`RenderAlphaMode::Cutout` materials.

### Skinning Matrix Convention

Skinning matrices from the avatar/pose system are stored
**column-major** (matching GLSL `mat4`). The camera view/projection
matrices are stored **row-major** and require `mat4_to_cols()`
transposition before upload. Do not apply `mat4_to_cols()` to skinning
matrices — it double-transposes them.

## Validation Checklist

Evaluate against a small representative avatar set, per milestone:

1. face under front and slight side lighting — not excessively dark,
   shadow threshold stable under slight camera movement
2. hair and transparent regions — cutout/blend doesn't break, culling
   doesn't drop surfaces, outline doesn't artifact
3. outline readability — visible when intended, width changes
   perceptible and stable across camera angles
4. alpha output — cutout behaves like cutout, blend like blend, alpha
   stays usable for compositing
5. overall silhouette

Regression signals: face quality improves on one avatar and breaks on
another common one; hair transparency destabilizes after outline
changes; alpha output becomes incompatible with compositing; parameter
changes stop producing predictable visible differences.

## Failure Modes To Avoid

- renderer decodes VRM schema directly during draw submission
- render API borrows large mutable app state, or result types leak raw
  GPU ownership without a token
- output export owns renderer images without a lifetime token
- outline implementation forces all materials into one pipeline path
- transparency behavior left as an accidental consequence of draw order
- preview path and output path diverge into unrelated render code
- calling the renderer "MToon compatible" too early, or chasing
  low-impact parameters before face / hair quality
- validating against a single avatar
