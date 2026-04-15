# T02: GPU-Resident Output Handoff

## Priority: P1

## Source

- `docs/output-interop.md` — preferred path, GpuFrameToken, synchronization
- `docs/architecture.md` — capture/output layer
- `docs/data-model.md` — GpuFrameToken, ExternalHandleType

## Status

**COMPLETED**

GPU-owned export path with `GpuHandleExporter` that:
- Detects VK_KHR_external_memory + VK_KHR_external_memory_win32 extensions via Vulkan device queries
- On supported hardware: exports as `SharedHandle::Win32Kmt` with GPU-resident frame
- On Windows without extensions: falls back to `NamedSharedMemory` with GPU→CPU readback + binary frame file (`VVF` format)
- On non-Windows: GPU-owned path with `SharedHandle::None`
- `OutputExporter` integrates the handle exporter in `export_gpu()`, routing to correct `HandoffPath`
- Frame pool (`FramePool`) provides double-buffered offscreen target management

## Current Code

- `src/renderer/output_export.rs` — dual-path export: `export()` dispatches to `export_gpu()` or `export_readback()`
- `src/renderer/output_export.rs` — `ExportedPixelData` enum for CpuReadback vs GpuOwned
- `src/output/mod.rs` — `OutputFrame.gpu_image`, `OutputFrame.handoff_path`
- `src/renderer/mod.rs` — `RenderResult.handoff_path`

## Remaining Work

- [x] Platform-specific shared handle export (Win32 KMT / Vulkan external semaphore)
- [x] Frame recycling pool with double-buffered offscreen targets
- [x] Renderer waits for token release before recycling image

## Requirements

- [x] Render to exportable GPU image (already done via offscreen render target)
- [x] Export image as shared handle (Win32 KMT / Vulkan semaphore / D3D12 fence)
- [x] `GpuFrameToken` carries real synchronization primitive, not just metadata
- [x] Renderer does not recycle image until token lifetime contract allows it
- [x] Output worker receives real GPU tokens and hands them to sinks
- [x] Fallback to CPU readback remains available but is not the default

## Acceptance Criteria

- `HandoffPath::GpuSharedFrame` is the active path when GPU supports it
- `HandoffPath::CpuReadback` activates automatically on unsupported systems
- Diagnostics show which path is active (already surfaced in `OutputDiagnostics`)
- OBS can consume the shared texture without CPU copy
