# T03: VirtualCamera / SharedTexture Sinks

## Priority: P2

## Source

- `docs/architecture.md` — capture/output layer, FrameSink variants
- `docs/output-interop.md` — sink abstraction
- `docs/application-design.md` — Output mode, sink selection

## Status

**COMPLETED**

All four sink variants have full `OutputSinkWriter` implementations:
- `ImageSequenceSink` — writes PNG files
- `RingBufferSharedMemorySink` — lock-free ring buffer with N slots, magic header, slot-level writes (now wired into `create_sink_writer()`)
- `SharedMemoryFileSink` — binary shared memory file with `VSTX` header + RGBA pixel data
- `VirtualCameraFileSink` — writes individual `frame_NNNNNN.rgba` files with `status.json` metadata to temp directory

Output inspector sink switching now recreates the `OutputRouter` when sink type changes.

## Current Code

- `src/output/frame_sink.rs` — all four sink variants have working writers
- `src/output/mod.rs` — `OutputRouter` dispatches to correct writer via `create_sink_writer()`
- `src/output/mod.rs` — `OutputDiagnostics` reports active sink and handoff path

## Remaining Work

- [x] Register a real virtual camera device (MediaFoundation init + shared memory via `CreateFileMappingW`, `virtual-camera` feature gate)
- [x] GPU image handle sharing with real synchronization (T02 GPU path now functional)
- [x] Named shared memory with ring buffer for lock-free consumption
- [x] Win32 event synchronization for frame-ready signaling
- [x] GPU fence counter for consumer synchronization

## Requirements

### VirtualCamera

- [x] Register a virtual camera device (platform-specific)
- [x] Push RGBA frames to the virtual camera at target framerate
- [x] Handle device creation failure gracefully
- [x] Win32 event for frame-ready notification

### SharedTexture

- [x] Share GPU image handle with external consumers (KMT handle export)
- [x] GPU fence counter for synchronization
- [x] Depends on T02 (GPU-resident handoff)

### SharedMemory (real implementation)

- [x] Write RGBA frames to named shared memory region
- [x] Ring buffer or double-buffer for lock-free consumption
- [x] Metadata header with timestamp, extent, format

## Acceptance Criteria

- Switching sink type in Output mode inspector activates the correct writer
- `OutputDiagnostics` reflects the active sink accurately
- Failure to initialize a sink falls back gracefully and surfaces a diagnostic
