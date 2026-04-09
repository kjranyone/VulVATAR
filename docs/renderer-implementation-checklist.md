# Renderer Implementation Checklist

## Purpose

This document gives an implementation order for the renderer without redefining the architecture.

It is meant to reduce thrash for the implementation agent.

Related documents:

- [renderer-module-design.md](/C:/lib/github/kjranyone/VulVATAR/docs/renderer-module-design.md)
- [renderer-api-contract.md](/C:/lib/github/kjranyone/VulVATAR/docs/renderer-api-contract.md)
- [vulkano-renderer-design.md](/C:/lib/github/kjranyone/VulVATAR/docs/vulkano-renderer-design.md)

## Phase 0: Module Skeleton

Create:

- `src/renderer/frame_input.rs`
- `src/renderer/material.rs`
- `src/renderer/pipeline.rs`
- `src/renderer/output_export.rs`

Exit criteria:

- module tree compiles
- top-level renderer API compiles
- no real Vulkan work required yet

## Phase 1: Snapshot Boundary

Implement:

- `RenderFrameInput`
- `RenderAvatarInstance`
- `RenderResult`

Update app integration to build renderer-facing snapshots.

Exit criteria:

- renderer no longer takes raw `AvatarInstance` as its primary render API
- `cargo check` passes

## Phase 2: Material Normalization Boundary

Implement:

- normalized material upload request types
- mapping from imported `MaterialAsset` to renderer material payload

Exit criteria:

- renderer consumes normalized material payloads
- raw schema handling stays outside draw logic

## Phase 3: Pipeline Family Selection

Implement:

- pipeline family enums
- pipeline keys
- basic pipeline selection by material mode and alpha mode

Exit criteria:

- no giant single-pipeline assumption in type design
- pipeline cache key exists even if cache is still stubbed

## Phase 4: Output Export Boundary

Implement:

- renderer-side export request type
- `OutputFrame` or token construction path
- explicit boundary from rendered target to exported frame

Exit criteria:

- output routing no longer depends on ad hoc render result fields alone

## Phase 5: Real Rendering Work

Implement in order:

1. unlit skinned path
2. simple lit path
3. toon-like path
4. outline path

Exit criteria:

- each stage compiles and can be demonstrated independently

## Integration Checks

At each phase:

- `cargo check` passes
- public module boundaries still match the docs
- new types do not pull app or tracking ownership into renderer modules

## Red Flags

Stop and re-evaluate if implementation starts doing any of these:

- renderer API still takes whole `AvatarInstance`
- material interpretation spills into app orchestration
- output export logic spreads across app and renderer without a single boundary
- one renderer file becomes the home for every new idea
