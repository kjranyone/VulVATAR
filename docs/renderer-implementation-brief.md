# Renderer Implementation Brief

## Purpose

This document is the handoff brief for the implementation agent working on the renderer.

It does not redefine the architecture.

It tells the implementation agent:

- what to build first
- what documents are authoritative
- what boundaries must not be broken
- what counts as done

## Primary Goal

Restructure the renderer code so it matches the documented design direction before attempting deep Vulkan or shader completeness work.

The immediate target is not full rendering fidelity.

The immediate target is a clean renderer boundary with:

- renderer-facing frame snapshots
- explicit module responsibilities
- normalized material input
- explicit output export boundary

## Required Reading

The implementation agent should treat these as the main sources of truth:

1. [docs/architecture.md](/C:/lib/github/kjranyone/VulVATAR/docs/architecture.md)
2. [docs/vulkano-renderer-design.md](/C:/lib/github/kjranyone/VulVATAR/docs/vulkano-renderer-design.md)
3. [docs/renderer-module-design.md](/C:/lib/github/kjranyone/VulVATAR/docs/renderer-module-design.md)
4. [docs/renderer-api-contract.md](/C:/lib/github/kjranyone/VulVATAR/docs/renderer-api-contract.md)
5. [docs/renderer-implementation-checklist.md](/C:/lib/github/kjranyone/VulVATAR/docs/renderer-implementation-checklist.md)
6. [docs/rendering-materials.md](/C:/lib/github/kjranyone/VulVATAR/docs/rendering-materials.md)
7. [docs/mtoon-compatibility.md](/C:/lib/github/kjranyone/VulVATAR/docs/mtoon-compatibility.md)
8. [docs/output-interop.md](/C:/lib/github/kjranyone/VulVATAR/docs/output-interop.md)

For shader-facing detail:

- [docs/shader-implementation-notes.md](/C:/lib/github/kjranyone/VulVATAR/docs/shader-implementation-notes.md)
- [docs/mtoon-test-matrix.md](/C:/lib/github/kjranyone/VulVATAR/docs/mtoon-test-matrix.md)

## Immediate Scope

The first implementation pass should focus on structure, not completeness.

Implement these steps:

1. create the intended `src/renderer/` module shape
2. introduce renderer-facing frame snapshot types
3. change the renderer API so it consumes snapshot input instead of raw avatar runtime ownership
4. introduce normalized material upload types
5. introduce a renderer-side output export boundary

## Expected File-Level Work

The implementation agent is expected to work primarily in:

- `src/renderer/mod.rs`
- `src/renderer/frame_input.rs`
- `src/renderer/material.rs`
- `src/renderer/pipeline.rs`
- `src/renderer/output_export.rs`
- `src/app/mod.rs`

Minimal supporting changes in other files are acceptable only when needed to preserve compile correctness.

## Hard Constraints

The implementation must not:

- make the renderer API depend directly on full mutable `AvatarInstance` ownership as its long-term contract
- decode raw VRM material schema during draw submission
- mix output export logic ad hoc into app orchestration
- create a single giant renderer module that absorbs every concern
- claim full MToon compatibility

## Recommended First API Shape

The implementation should move toward this boundary:

```rust
pub struct VulkanRenderer { /* renderer-owned state */ }

impl VulkanRenderer {
    pub fn new() -> Self;
    pub fn initialize(&mut self);
    pub fn render(&mut self, input: &RenderFrameInput) -> RenderResult;
}
```

The exact types can be thinner at first, but the shape should be preserved.

## Implementation Priorities

### Priority 1

- module split
- snapshot boundary
- compile-safe public API

### Priority 2

- material normalization boundary
- pipeline family selection types
- output export types

### Priority 3

- richer per-instance data
- richer result stats
- more exact future-ready pipeline keys

## What "Done" Means For This Pass

This pass is complete when:

- renderer module boundaries match the design docs
- app code builds a renderer-facing frame snapshot
- renderer no longer conceptually renders "the app" or "the avatar runtime" directly
- output export has a named boundary in code
- `cargo check` passes

It is still acceptable if:

- Vulkan work remains stubbed
- shader work remains partial
- MToon remains staged and incomplete

## Review Checklist For The Implementation Agent

Before finishing, the implementation agent should verify:

- `render(&AvatarInstance)` is no longer the main renderer boundary
- new renderer modules have clear responsibility and low overlap
- app orchestration does not absorb renderer internals
- material upload types are renderer-normalized
- output handoff is represented by an explicit type boundary

## Common Failure Cases

These are signs the implementation is drifting:

- renderer snapshots are just thin wrappers around borrowing all app state
- material normalization still lives in random draw-call code
- output export is represented only by width or height or timestamp fields
- pipeline logic is spread across unrelated files with no clear selector type

## Follow-Up Work After This Pass

Once this pass lands cleanly, the next implementation pass should move to:

1. unlit skinned rendering through the new API
2. simple lit rendering
3. toon-like path
4. outline path
5. tighter output export integration
