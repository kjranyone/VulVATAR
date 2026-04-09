# Renderer API Contract

## Purpose

This document defines the implementation-facing API boundary between app/runtime code and the renderer.

Related documents:

- [renderer-module-design.md](/C:/lib/github/kjranyone/VulVATAR/docs/renderer-module-design.md)
- [vulkano-renderer-design.md](/C:/lib/github/kjranyone/VulVATAR/docs/vulkano-renderer-design.md)
- [data-model.md](/C:/lib/github/kjranyone/VulVATAR/docs/data-model.md)

## Core Rule

The renderer consumes renderer-facing snapshots.

It does not consume raw app orchestration state or mutable avatar runtime structures as its primary contract.

## Top-Level API

Conceptually, the renderer boundary should look like:

```rust
pub struct VulkanRenderer { /* renderer-owned state */ }

impl VulkanRenderer {
    pub fn new() -> Self;
    pub fn initialize(&mut self);
    pub fn render(&mut self, input: &RenderFrameInput) -> RenderResult;
}
```

The exact signature can evolve, but this boundary should remain narrow.

## `RenderFrameInput`

Purpose:

- one immutable snapshot of everything the renderer needs for a frame

Recommended fields:

- `camera: CameraState`
- `lighting: LightingState`
- `instances: Vec<RenderAvatarInstance>`
- `output_request: OutputTargetRequest`

Rules:

- contains renderer-facing data only
- contains no editor tool state
- contains no tracking worker mailbox state

## `RenderAvatarInstance`

Purpose:

- one renderer-facing avatar draw package

Recommended fields:

- `instance_id`
- `world_transform`
- `mesh_instances`
- `skinning_matrices`
- `cloth_deform`
- `debug_flags`

Rules:

- material references must already be normalized enough for renderer use
- no mutable simulation ownership
- no retargeting logic

## `RenderMeshInstance`

Purpose:

- one draw-ready mesh primitive submission unit

Recommended fields:

- `mesh_id`
- `primitive_id`
- `material_binding`
- `bounds`
- `alpha_mode`
- `cull_mode`
- `outline`

## `CameraState`

Recommended fields:

- `view`
- `projection`
- `position_ws`
- `viewport_extent`

## `LightingState`

Recommended first fields:

- `main_light_dir_ws`
- `main_light_color`
- `main_light_intensity`
- `ambient_term`

## `OutputTargetRequest`

Purpose:

- describe what kind of output the frame should produce

Recommended fields:

- `preview_enabled`
- `output_enabled`
- `extent`
- `color_space`
- `alpha_mode`
- `export_mode`

## `RenderResult`

Purpose:

- summarize the renderer's completed frame result

Recommended fields:

- `extent`
- `timestamp`
- `has_alpha`
- `exported_frame: Option<ExportedFrame>`
- `stats`

The result should not expose raw renderer internals.

## `ExportedFrame`

Purpose:

- renderer-approved exported frame handoff for output routing

Recommended fields:

- `output_frame`
- `gpu_token`
- `export_metadata`

This may collapse into `OutputFrame` if the implementation stays simple.

## API Stability Rules

- app code can construct `RenderFrameInput`
- renderer code can mutate renderer internals only
- output routing consumes exported frame objects, not renderer-owned images

## Minimal First Version

The first implementation does not need the final rich types immediately.

But even a minimal version should preserve the shape:

- snapshot in
- result out

Do not start from:

- `render(&AvatarInstance)`

because that couples the renderer to avatar runtime ownership too early.

## Failure Modes To Avoid

- render API that borrows large mutable app state
- result types that leak raw GPU ownership without a token contract
- API signatures that force app code to understand pipeline internals
