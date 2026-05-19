# Vulkano Renderer Design

## Purpose

This document defines the renderer implementation strategy for VulVATAR on top of `Vulkano`.

It focuses on:

- renderer responsibilities
- pass structure
- pipeline structure
- GPU resource ownership
- data upload boundaries
- interaction with output handoff

Related documents:

- [architecture.md](/C:/lib/github/kjranyone/VulVATAR/docs/architecture.md)
- [rendering-materials.md](/C:/lib/github/kjranyone/VulVATAR/docs/rendering-materials.md)
- [mtoon-compatibility.md](/C:/lib/github/kjranyone/VulVATAR/docs/mtoon-compatibility.md)
- [output-interop.md](/C:/lib/github/kjranyone/VulVATAR/docs/output-interop.md)
- [shader-implementation-notes.md](/C:/lib/github/kjranyone/VulVATAR/docs/shader-implementation-notes.md)
- [renderer-module-design.md](/C:/lib/github/kjranyone/VulVATAR/docs/renderer-module-design.md)
- [renderer-api-contract.md](/C:/lib/github/kjranyone/VulVATAR/docs/renderer-api-contract.md)

## Core Rule

The renderer owns GPU objects and draw submission.

It does not own:

- avatar truth
- simulation truth
- tracking truth

The renderer consumes resolved frame data and turns it into images.

## First Renderer Scope

The first renderer should support:

- one avatar instance
- skinned mesh rendering
- staged material modes
- optional outline
- alpha-capable output
- exportable output frame creation

It does not need:

- a full generalized scene graph
- deferred rendering
- large-scale material permutation systems

## Renderer Responsibilities

The renderer should own:

- device and queue setup
- swapchain lifecycle
- render target allocation
- pipeline creation and caching
- descriptor set allocation strategy
- material upload layout
- pose buffer upload
- cloth deformation upload
- output frame export

## Renderer Inputs

Per-frame renderer input should be normalized and explicit.

Recommended top-level input:

- camera state
- lighting state
- one or more render instances
- per-instance pose data
- per-instance cloth deformation data
- frame output target request

## Recommended Frame Input Shape

The renderer should consume something conceptually like:

```rust
pub struct RenderFrameInput {
    pub camera: CameraState,
    pub lighting: LightingState,
    pub instances: Vec<RenderAvatarInstance>,
    pub output_request: OutputTargetRequest,
}
```

`RenderAvatarInstance` should contain renderer-facing data only.

It should not embed mutable gameplay or editor state.

## Pass Structure

Recommended early pass structure:

1. depth prepass if needed later
2. main forward avatar pass
3. outline pass
4. output resolve or export step

### Transform Compute Prepass

Runs once per (instance, primitive) before the graphics passes start.

Responsibilities:

- fuse linear blend skinning, morph-target blending, and cloth override
  into a single dispatch
- write world-space `GpuVertex` records to a persistent SSBO that the
  graphics passes bind as their vertex buffer

The graphics vertex shaders are reduced to view × projection; they
never see the source `GpuVertexBase` or the morph / cloth buffers. See
`pipeline::transform_cs` for the shader contract and
`plan/compute-prepass-migration.md` for the migration rationale.

### Main Forward Avatar Pass

Responsibilities:

- draw opaque and alpha-tested avatar surfaces using the compute
  prepass output
- evaluate selected material mode

### Outline Pass

Responsibilities:

- render material-driven outlines
- respect alpha and culling policy decisions

This should stay separate from generic material shading logic.

### Output Resolve or Export Step

Responsibilities:

- prepare the final color target
- preserve alpha contract if enabled
- export the image when the selected output path needs it

## Pipeline Strategy

The renderer should avoid one giant all-feature pipeline.

Recommended pipeline families:

- `SkinningUnlitPipeline`
- `SkinningSimpleLitPipeline`
- `SkinningToonPipeline`
- `OutlinePipeline`
- `DepthOnlyPipeline` if needed

These can share layout conventions, but feature differences should remain explicit.

## Shader Input Strategy

The shader input contract should be normalized before pipeline submission.

Recommended categories:

- scene uniforms
- per-instance transforms
- per-instance skinning data
- per-instance cloth deformation data
- material uniforms
- material textures

## Descriptor Layout Strategy

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

Linear blend skinning runs inside the transform compute prepass, not
the graphics vertex shader. Inputs are unchanged:

- four weights and four joint indices per vertex in `GpuVertexBase`
- one skinning matrix SSBO per `AvatarInstance`, bound to compute set 1

Reasons for skinning compute-side:

- a single compute dispatch produces world-space vertices that every
  graphics pass shares (forward + outline + future thumbnail / debug),
  avoiding redundant skinning math per pass
- the graphics shaders shrink to view × projection, reducing per-draw
  vertex work and making them trivial to extend with new passes
- skinning, morph blend, and cloth override fuse into a single
  dispatch instead of three independent shader paths

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

## Material Normalization Boundary

The importer and material normalization layer should translate VRM material input into renderer-ready data before draw time.

The renderer should consume:

- selected material mode
- normalized numeric parameters
- texture bindings
- alpha and culling flags
- outline settings

It should not decode raw VRM extension schema while drawing.

## Transparency Strategy

Transparency should be explicitly split by policy.

Recommended early buckets:

- opaque
- alpha test
- alpha blend

The renderer should define:

- draw order rules for blend materials
- whether outlines are allowed for blended materials in the first pass
- how alpha writes are preserved for output

## Outline Strategy

The outline pipeline should be an explicit pass with its own state.

The design should expose:

- width control
- color source
- culling choice

Do not bury outline logic inside the main surface shader if it materially changes raster state or geometry treatment.

## Resource Ownership

The renderer owns:

- swapchain images
- offscreen color targets
- depth targets
- exportable output images
- descriptor allocators
- pipeline caches
- GPU staging resources

External systems should receive only exported handles or tokens, not direct ownership of renderer internals.

## Frame Graph Discipline

The renderer does not need a full formal frame graph system in the first pass.

But it should still keep these rules:

- passes have explicit input and output targets
- image transitions are explicit in renderer code
- output export happens after rendering is complete

## Vulkano-Specific Guidance

The implementation should lean into `Vulkano` strengths:

- typed pipeline creation
- typed descriptor bindings
- explicit resource ownership

Avoid building an unnecessary engine-like abstraction layer too early.

Use thin wrappers around `Vulkano` concepts where they clarify project-specific intent:

- `Renderer`
- `RenderFrameInput`
- `MaterialUploader`
- `PoseBufferUploader`
- `OutputExporter`

## Swapchain and Offscreen Policy

The renderer should support both:

- on-screen preview through the swapchain
- offscreen rendering for export

Recommended first policy:

- render into an offscreen target
- present that target to the preview path
- export that target or a resolved derivative to the output path

This keeps preview and output closer to the same visual result.

## Output Export Boundary

The renderer should produce an `OutputFrame` or `GpuFrameToken` only after all rendering work for the frame is complete.

That export boundary should define:

- the exported image
- its format and alpha mode
- the synchronization primitive
- when the image may be reused

## Resizing Policy

Renderer resize handling should rebuild:

- swapchain-dependent targets
- preview framebuffers
- output-sized offscreen targets if coupled to viewport size

It should not require re-importing avatars or rebuilding CPU-side material definitions.

## Pipeline Cache Policy

The renderer should maintain a small explicit pipeline cache.

Keys should include:

- material mode
- alpha mode
- outline enabled state if relevant to pass selection
- vertex layout or skinning path if variants exist

Avoid runtime pipeline creation during normal rendering if possible after warmup.

## Debug Features

The renderer should leave room for:

- skeleton debug lines
- collider debug shapes
- cloth simulation mesh overlay
- material mode override
- normal visualization

These should be optional overlays, not special cases embedded into production material logic.

## Implementation Lessons

### Depth+Stencil Format

The render pass uses `D32_SFLOAT_S8_UINT` (not plain `D32_SFLOAT`). The stencil bits are required for outline masking. All depth image creation, clear values, and framebuffer attachments must match this format.

Clear values for depth+stencil attachments use `ClearValue::DepthStencil((1.0, 0))`, not `1.0f32.into()`.

### Blend Pipeline Variants

Each cull mode (Back, Front, None) needs separate Opaque and Blend pipeline variants. Blend pipelines differ in:

- `depth_write_enable: false` (prevents blend ordering artifacts)
- `blend: Some(AttachmentBlend::alpha())`

The render loop iterates `blend_pass in [false, true]` and skips non-matching meshes.

### Front Face Winding

glTF/VRM uses counter-clockwise front face winding (`FrontFace::CounterClockwise`). This is Vulkan's default. Do NOT change to `Clockwise` — doing so causes back-face textures to render on the front of the body (reversed legs, neck, etc.). All pipelines (main, outline, depth-only) must use the same `CounterClockwise` convention.

### Outline and Alpha-Masked Materials

The outline shader does not perform alpha testing. For alpha-masked (Cutout) materials, the outline pass would draw in regions that the main pass discarded via alpha cutoff, creating dark jagged artifacts that extend beyond the visible mesh boundary. Therefore, outline collection must be skipped for `RenderAlphaMode::Cutout` materials.

### Skinning Matrix Convention

Skinning matrices from the avatar/pose system are stored **column-major** (matching GLSL `mat4`). The camera view/projection matrices are stored **row-major** and require `mat4_to_cols()` transposition before upload.

Do not apply `mat4_to_cols()` to skinning matrices — it double-transposes them.

### Matcap Disabled

The matcap path (`matcap_blend`) is forced to 0.0 because VRoid placeholder matcap textures (`MatcapWarp`, 8x8) over-brighten avatars when treated as direct color maps. Re-enable once a proper MToon-compatible matcap implementation is added.

## Failure Modes To Avoid

- renderer decodes VRM schema directly during draw submission
- output export owns renderer images without a lifetime token
- outline implementation forces all materials into one pipeline path
- transparency behavior is left as an accidental consequence of draw order
- preview path and output path diverge into unrelated render code
