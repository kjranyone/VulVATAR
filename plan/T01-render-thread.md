# T01: Independent Render Thread

## Priority: P1

## Source

- `docs/threading-model.md` — render thread role
- `docs/architecture.md` — frame update order

## Status

**IMPLEMENTED**

Dedicated render thread spawned in `src/app/render_thread.rs`.
`RenderCommand` enum dispatched via `mpsc::channel`.
`VulkanRenderer` owned exclusively by the render thread.
App thread submits commands non-blocking; receives `RenderResult` via separate channel.

## Current Code

- `src/app/render_thread.rs` — `RenderThread` struct, `RenderCommand` enum, worker loop
- `src/app/mod.rs` — `Application::render_thread` field, `process_render_result()`
- `src/app/mod.rs` — `run_frame()` submits `RenderCommand::RenderFrame` and polls results
- `src/app/mod.rs` — `shutdown()` sends `RenderCommand::Shutdown` and joins render thread

## Requirements

- [x] Spawn a dedicated render thread at bootstrap
- [x] App thread submits `RenderCommand::RenderFrame(RenderFrameInput)` via bounded channel
- [x] Render thread owns GPU resources and executes draw calls
- [x] Render thread publishes completed `RenderResult` back to app thread
- [x] `RenderCommand::Resize` handled asynchronously
- [x] `RenderCommand::Shutdown` drains pending work and joins
- [x] App thread never blocks on render submission (bounded queue + drop policy)

## Acceptance Criteria

- Render thread runs independently from app thread
- Frame rate of app loop is not coupled to render latency
- Shutdown order from `docs/threading-model.md` is respected
- `cargo check` passes with no warnings about dead `RenderCommand`
