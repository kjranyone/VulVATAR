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
