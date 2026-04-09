# Renderer Module Design

## Purpose

This document defines how the renderer code should be split inside `src/renderer/`.

It is intended for implementation planning, not runtime architecture discussion.

Related documents:

- [architecture.md](/C:/lib/github/kjranyone/VulVATAR/docs/architecture.md)
- [vulkano-renderer-design.md](/C:/lib/github/kjranyone/VulVATAR/docs/vulkano-renderer-design.md)
- [renderer-api-contract.md](/C:/lib/github/kjranyone/VulVATAR/docs/renderer-api-contract.md)

## Core Rule

`src/renderer/` should be organized around renderer responsibilities, not around miscellaneous helpers.

Avoid these dumping-ground patterns:

- `renderer/types.rs`
- `renderer/utils.rs`
- `renderer/common.rs`

## Recommended Module Layout

```text
src/renderer/
  mod.rs
  frame_input.rs
  material.rs
  pipeline.rs
  output_export.rs
```

Later additions can include:

```text
src/renderer/
  upload.rs
  targets.rs
  debug_draw.rs
  pass/
    mod.rs
    forward.rs
    outline.rs
```

## Module Responsibilities

### `mod.rs`

Responsibilities:

- define the top-level renderer type
- expose public renderer-facing API
- coordinate calls into submodules

It should own:

- `VulkanRenderer`
- `RenderResult`

It should not contain:

- detailed material normalization logic
- pipeline variant tables
- output export details

### `frame_input.rs`

Responsibilities:

- define renderer-facing frame input structs
- define renderer-facing per-instance snapshot structs
- isolate renderer input from app or avatar mutable state

It should own:

- `RenderFrameInput`
- `RenderAvatarInstance`
- `CameraState`
- `LightingState`
- `OutputTargetRequest`

### `material.rs`

Responsibilities:

- define normalized renderer material upload data
- translate imported materials into renderer-facing material payloads
- own material upload staging strategy

It should own:

- `MaterialShaderMode`
- `MaterialUploadRequest`
- `MaterialUploader`

### `pipeline.rs`

Responsibilities:

- define pipeline family enums and selection logic
- map normalized material modes and alpha modes onto pipeline variants
- own pipeline cache keys

It should own:

- `RenderPipeline`
- `PipelineState`
- `PipelineKey`

### `output_export.rs`

Responsibilities:

- define renderer-side output export boundary
- produce `OutputFrame` or `GpuFrameToken`
- translate rendered targets into exportable output resources

It should own:

- `OutputExporter`
- `ExportRequest`
- `ExportResult`

## Dependency Direction

Inside the renderer domain:

- `mod.rs -> frame_input, material, pipeline, output_export`
- `output_export -> output` only through narrow public output types
- `material -> asset` is allowed for normalization input

Avoid:

- `frame_input -> output_export`
- `pipeline -> app`
- `material -> app`
- `output_export -> avatar`

## Public API Boundary

The rest of the app should interact with the renderer through a small surface:

- renderer construction
- renderer initialization
- render a `RenderFrameInput`
- receive a `RenderResult`

The app should not:

- choose internal pipeline objects directly
- allocate descriptor layouts directly
- inspect renderer-owned GPU image internals

## Snapshot Rule

`frame_input.rs` exists to enforce a snapshot boundary.

The renderer should consume immutable frame snapshots, not borrow mutable `AvatarInstance` state directly.

This is important for:

- ownership clarity
- future render threading
- testability

## Future Extension Guidance

When the renderer grows, split by responsibility:

- upload logic into `upload.rs`
- render target management into `targets.rs`
- per-pass logic into `pass/`

Do not split prematurely before each module has a clear responsibility.

## Failure Modes To Avoid

- a top-level renderer file that accumulates all logic
- material, pipeline, and export code intertwined in one module
- app code constructing renderer internals directly
- mutable avatar runtime state passed straight into renderer internals
