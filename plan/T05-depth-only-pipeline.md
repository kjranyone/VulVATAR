# T05: Depth-Only Pipeline + Pass Modules

## Priority: P2

## Source

- `docs/architecture.md` — MToon-like shader strategy, pipeline split
- `docs/renderer-module-design.md` — future module layout
- `docs/vulkano-renderer-design.md` — pass structure

## Status

**IMPLEMENTED**

- Depth-only vertex + fragment shaders in `pipeline.rs`
- `create_depth_only_pipeline()` function
- `src/renderer/pass/` module with `forward.rs`, `outline.rs`, `depth_only.rs`
- Depth-only pipeline stored in `VulkanRenderer`, recreated on resize
- `depth_prepass_enabled` flag controls pre-pass execution
- Depth pre-pass issues actual draw calls sharing vertex/index buffers with forward pass
- Forward pass depth compare changed to `LessOrEqual` for pre-pass compatibility
- Inline outline pass migrated to `pass::outline::record_outline_pass()`
- Upload helpers extracted to `src/renderer/upload.rs`
- Offscreen target management extracted to `src/renderer/targets.rs`
- `render()` refactored into phases: collect → depth pre-pass → forward → outline

## Current Code

- `src/renderer/pipeline.rs` — pipeline creation, depth-only shaders
- `src/renderer/pass/forward.rs` — ForwardPassDraw data struct
- `src/renderer/pass/outline.rs` — record_outline_pass() function
- `src/renderer/pass/depth_only.rs` — record_depth_prepass() function
- `src/renderer/upload.rs` — shared upload helpers
- `src/renderer/targets.rs` — offscreen target management
- `src/renderer/mod.rs` — orchestrates passes, no inline draw logic

## Requirements

### Depth-Only Pipeline

- [x] Create `DepthOnlyPipeline` shader (vertex-only, no fragment output)
- [x] Use depth pre-pass for occlusion culling before main pass
- [x] Share vertex buffer and skinning data with forward pass

### Pass Module Split

- [x] Create `src/renderer/pass/mod.rs`
- [x] Create `src/renderer/pass/forward.rs` — main forward pass logic
- [x] Create `src/renderer/pass/outline.rs` — outline pass logic (migrated from inline)
- [x] Migrate inline render code from `mod.rs` into pass modules

### Upload Module

- [x] Create `src/renderer/upload.rs` — shared upload helpers (vertex, index, texture, uniform)
- [x] Consolidate `create_vertex_buffer`, `create_index_buffer`, etc.

### Targets Module

- [x] Create `src/renderer/targets.rs` — offscreen target management
- [x] Consolidate `create_offscreen_targets`, resize logic

## Acceptance Criteria

- [x] `src/renderer/mod.rs` coordinates passes but does not contain inline draw logic
- [x] Each pass module owns its pipeline creation and draw submission
- [x] Depth pre-pass can be enabled/disabled without breaking forward pass
