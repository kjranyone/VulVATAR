# Output Interop

## Purpose

This document defines how rendered frames leave VulVATAR and reach OBS or other consumers.

It focuses on:

- output sink abstraction
- GPU-resident handoff
- synchronization
- queue policy
- fallback behavior

Related documents:

- [architecture.md](/C:/lib/github/kjranyone/VulVATAR/docs/architecture.md)
- [data-model.md](/C:/lib/github/kjranyone/VulVATAR/docs/data-model.md)
- [threading-model.md](/C:/lib/github/kjranyone/VulVATAR/docs/threading-model.md)

## Core Rule

The preferred output path is GPU-resident handoff with explicit synchronization.

CPU readback is a fallback path, not the architectural center.

Current bridge note: `SharedTextureFileStub` now preserves that distinction at
the last hop. It writes metadata-only `VGTK` records when a valid
`GpuFrameToken` arrives and legacy `VSTX` records when the frame is on the CPU
fallback path. The downstream consumer is still a stub; this is the first
token-capable sink boundary, not yet the finished live interop path.

P2-05 phase 2 extends this contract to the live `VirtualCamera` and
`SharedMemory` sinks on Windows. Both sinks are backed by
`Win32FileBackedSharedMemorySink`, which now emits the same `VGTK` record
into a sidecar file alongside the existing byte buffer when the renderer
publishes a valid GPU token, and leaves the byte buffer to the existing
CPU-fallback path. See [`Win32 GPU-handle sidecar`](#win32-gpu-handle-sidecar).

## Win32 GPU-handle sidecar

`Win32FileBackedSharedMemorySink` (used by `FrameSink::VirtualCamera` and
`FrameSink::SharedMemory` on Windows) now supports two on-disk shapes that
the consumer (MF virtual camera DLL) must distinguish:

| Path | File | Contents | When |
|---|---|---|---|
| GPU shared handle | `<main>.vgtk` (sidecar) | 72-byte `VGTK` record | renderer published a publishable `GpuFrameToken` |
| CPU readback | `<main>.bin` (memory-mapped) | existing 32-byte header + RGBA bytes | no token / handle unavailable |

The sidecar path is the main file's path with the extension swapped to
`.vgtk`. For the two production deployments:

- `%ProgramData%\VulVATAR\camera_frame_buffer.bin` →
  `%ProgramData%\VulVATAR\camera_frame_buffer.vgtk`
- `%ProgramData%\VulVATAR\shared_memory_buffer.bin` →
  `%ProgramData%\VulVATAR\shared_memory_buffer.vgtk`

### VGTK record layout (v1)

```
offset  size  field
  0       4   magic "VGTK"
  4       4   version (u32, currently 1)
  8       4   width
 12       4   height
 16       4   handle_type (0 Unavailable, 1 Win32Kmt, 2 D3D12Fence,
                          3 VkSemaphore, 4 SharedMemoryHandle)
 20       4   sync_type   (0 None, 1 ProducerWaitComplete,
                          2 FenceValue, 3 SemaphoreHandle)
 24       8   external_handle (u64)
 32       8   sync_value      (u64)
 40       8   frame_id
 48       8   timestamp_nanos
 56       4   color_space (0 srgb, 1 linear)
 60       4   alpha_mode  (0 opaque, 1 premultiplied, 2 straight)
 64       4   flags       (bit 0: preserve alpha; bit 1: linear colour)
 68       4   reserved
 72             — end of header
```

The shape is shared with `SharedMemoryFileSink` so a future consumer
factoring can read one format regardless of which producer wrote it.

### Atomic publish

Each VGTK sidecar write goes through a `<sidecar>.tmp` file then a
`std::fs::rename` to publish the new inode atomically. A consumer that
`OpenSharedResource1`s the handle from a fully-written sidecar will never
observe a torn record.

### Lease policy (Option A — short-lived)

The renderer's export-image pool reserves a slot for each published GPU
frame. That slot must not be recycled while the consumer is bound to the
shared texture; otherwise the renderer would write into the texture the
consumer is sampling.

`Win32FileBackedSharedMemorySink` cooperates with `OutputRouter` to enforce
the following lifetime:

1. renderer publishes frame N with `FrameLifetimeContract::SingleConsumerRetained`
2. router writes the VGTK sidecar via the worker; does NOT release the
   lease yet
3. router records frame N's lease id as the "currently bound" lease
4. on the next successful publish (frame N+1), router sends frame N's
   lease completion before recording frame N+1 as the new bound lease
5. on shutdown, router flushes the bound lease so the slot is reclaimed

This Option A protocol assumes the consumer reads the latest published
sidecar within one producer frame interval. A consumer that is slower than
the producer sees the second-latest GPU frame for one tick — the same
behaviour as the existing byte-buffer's `seq_in_progress` /
`seq_committed` window provides on the CPU path. Adopting the
back-channelled "Option B / C" protocol becomes possible once the DLL
gains a producer-readable acknowledgement field.

### Partial-deploy safety

A new producer paired with an old DLL keeps working: the DLL keeps reading
the byte buffer's stale contents, which the producer no longer overwrites
during GPU frames. The DLL sees the same RGBA pixels indefinitely until
the producer falls back to the CPU path (handle export failure / GPU
contention) or the user restarts the DLL.

An old producer paired with a new DLL also works: the sidecar simply does
not exist, and the DLL reads the byte buffer as before.

## Sink Abstraction

Recommended sink families:

- `VirtualCamera`
- `SharedTexture`
- `SharedMemory`
- `ImageSequence`

The sink API should consume `OutputFrame` or `GpuFrameToken`, not renderer-internal image objects.

## Handoff Contract

`handoff` means:

- what resource is exported
- who owns it while the sink consumes it
- what synchronization primitive proves it is ready
- when the renderer is allowed to recycle it

## Preferred Path

The preferred path is:

1. render thread finishes a frame into an exportable GPU image
2. render thread creates a `GpuFrameToken`
3. token carries external handle metadata and sync metadata
4. output worker passes that token to the selected sink
5. sink signals completion or the token lifetime ends by policy
6. renderer recycles the image only after the contract allows it

## `GpuFrameToken` Requirements

The token should encode:

- exported resource identifier
- handle type
- frame extent
- color format
- alpha mode
- synchronization token
- lifetime contract

## Synchronization

The design should support explicit sync primitives.

Examples:

- exported semaphore
- exported fence
- keyed mutex style synchronization if a target API requires it

Do not assume "submission finished" implies safe external consumption without explicit sync.

Current implementation note: the first token path uses
`OutputSyncToken::ProducerWaitComplete`, meaning the producer waits for render
completion before publishing the token. That is an honest readiness contract
for the present code, but it is not yet an exported cross-process wait object.
True zero-copy GPU interop still needs an exported fence or semaphore handle
before a live sink can consume frames asynchronously without CPU-side waiting.

## Queue Policy

Output must use a bounded queue.

Recommended first policy:

- size 1 or 2
- prefer `ReplaceLatest` or `DropOldest`
- never block the app thread

If OBS or another sink falls behind, stale frames should be discarded before they stall live preview.

## Format Policy

The output contract must declare:

- resolution
- frame rate target
- color space
- alpha mode
- handle type

If OBS compositing is a goal, alpha behavior must be explicit.

## Fallback Paths

Fallbacks are allowed for unsupported systems.

Examples:

- CPU readback to shared memory
- CPU readback to image sequence

Fallback activation should be visible in diagnostics.

## Failure Policy

If the preferred interop path fails:

- keep local preview running
- disable only the affected sink
- surface the fallback or failure clearly

Output failure should not destroy scene state.

## Diagnostics

The GUI should expose:

- active sink type
- active handoff path
- GPU interop enabled or disabled
- queue depth
- dropped frame count
- last successful publish timestamp

## Failure Modes To Avoid

- passing raw renderer-owned image pointers across threads without lifetime tracking
- blocking rendering on sink-side latency
- making CPU readback the default path by accident
- hiding fallback activation from the user

## Manual QA — Output Color Space

These checks are not automated (no Vulkan device or sample VRM in CI).
Run them before shipping any change that touches `pipeline.rs` shader
sources, `color_attachment_format`, or the output sink path.

### Visual Matrix

1. `cargo run --bin render_matrix -- <some.vrm>` produces four PNGs:
   `srgb_opaque`, `srgb_transparent`, `linear_opaque`, `linear_transparent`.
2. Open them side-by-side. The two sRGB renders should look like the
   avatar in the GUI viewport. The two linear renders should look
   noticeably darker (un-encoded linear values).
3. Watch for tell-tale regressions:
   - **Double encode** (sRGB cell looks washed-out / pale): a manual
     `pow(c, 1/2.2)` slipped into a fragment shader on top of the
     sRGB attachment format.
   - **No encode** (sRGB cell looks dark, like the linear cell): the
     attachment format is wrong (UNORM where SRGB was expected) or the
     shader stopped writing linear.
   - **Alpha bleed** in the transparent cells: pre-multiplied vs.
     straight alpha got crossed.

### OBS Handoff

The Windows virtual-camera path tags frames with
`MF_MT_TRANSFER_FUNCTION`; downstream consumers (OBS, Discord,
Teams) read it to decide how to interpret the bytes. The CI lint
catches *shader-side* gamma regressions but cannot verify that the
metadata reaches the consumer correctly.

1. Start VulVATAR with the virtual-camera output sink active.
2. Open OBS Studio, add a Video Capture Device source, pick the
   VulVATAR virtual camera.
3. Compare the OBS preview against the VulVATAR GUI viewport. They
   should match. A consistent hue / brightness shift means OBS is
   re-applying a transfer curve (or skipping one we expected it to
   apply) — check `MF_MT_TRANSFER_FUNCTION` in `mfenum`'s output and
   the format VulVATAR is publishing.
4. Repeat with the linear output mode active. OBS preview should
   look noticeably darker; if it looks identical to the sRGB run,
   the metadata isn't being honoured downstream.
