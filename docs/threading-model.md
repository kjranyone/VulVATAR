# Threading Model

## Purpose

This document defines concurrency boundaries for VulVATAR.

The goal is to make async tracking and async output explicit without making the rest of the system accidentally multithreaded.

Related documents:

- [architecture.md](architecture.md)
- [tracking-v2-design.md](tracking-v2-design.md)
- [output-interop.md](output-interop.md)

## Thread Roles

Runtime threads:

1. `app thread` (main / UI loop)
2. `render thread` (Vulkano rendering & GPU management)
3. `tracking worker` (RealSense acquisition & fusion estimator)
4. `yolox worker` (person detection inference)
5. `output worker` (OBS / Spout / virtual camera sink)
6. `audio capture worker` (CPAL lipsync stream)
7. `avatar load worker` (async VRM / FBX asset loading)
8. `folder watcher` (avatar directory monitoring)
9. optional `debug channel worker` (tracking telemetry stream)

## `app thread`

Responsibilities:

- own application state and UI event loop (`eframe` / `egui`)
- run mode transitions (`AppMode::Avatar`, `TrackingSetup`, `Rendering`, `Output`, `ClothAuthoring`, `Settings`)
- update avatar runtime (`AvatarInstance`, retargeting, expression solving)
- step animation and simulation scheduling (spring bones, cloth XPBD)
- read latest tracking sample from `TrackingMailbox`
- submit render commands and receive results from `ResultMailbox`

Rules:

- this is the owner of `AvatarInstance`
- this is the owner of project state
- simulation state should not be mutated concurrently from other threads

## `render thread`

Responsibilities:

- own renderer-facing GPU state and Vulkan resources (`VulkanRenderer`)
- execute frame rendering (HDR scene pass, bloom down/upsample, tone-mapping composite)
- publish completed output frames to the active output sink
- execute offscreen cloth baking passes when requested

Rules:

- GPU resource lifetime is decided here
- render thread does not own scene logic
- render thread accepts immutable frame snapshots (`RenderFrameInput`), not mutable scene ownership
- results and errors are sent back to the app thread via `ResultMailbox`

## `tracking worker`

Responsibilities:

- acquire RealSense D435 depth + colour frames
- run inference pipeline (via `FusionProvider`)
- normalize landmarks and estimate rig rotations
- publish latest `SourceSkeleton` and `RigPose` into `TrackingMailbox`

Rules:

- never mutate avatar runtime state directly
- use latest-sample mailbox semantics (`TrackingMailbox`)
- dropping stale tracking samples is acceptable

## `yolox worker`

Responsibilities:

- run YOLOX person detector asynchronously on camera color frames
- publish bounding boxes back to the tracking estimator

## `output worker`

Responsibilities:

- consume `OutputFrame` tokens
- communicate with sink-specific APIs (Direct3D11 shared texture, Spout, MediaFoundation virtual camera, shared memory, PNG sequence)
- enforce output queue policy

Rules:

- never block the app thread waiting for sink completion
- never access renderer-owned GPU images without a valid exported token
- drop or replace frames according to explicit policy when overloaded

## `audio capture worker`

Responsibilities:

- capture audio input via CPAL
- compute volume and phoneme / viseme weights in real time
- provide lip-sync weights to the app thread

## `avatar load worker`

Responsibilities:

- parse `.vrm` and `.fbx` files asynchronously on a background thread
- parse optional `.unitypackage` files for secondary motion
- build immutable `AvatarAsset` and send to app thread via channel

## Communication Channels

Live channels:

- `tracking worker -> app thread`: `TrackingMailbox` (`Arc<Mutex<Option<TrackingPayload>>>`) with `SourceSkeleton` and `RigPose`
- `app thread -> render thread`: `sync_channel(2)` for `RenderCommand`, with results returned via `ResultMailbox` (`Arc<Mutex<Option<RenderResult>>>`)
- `render thread -> output worker`: bounded frame token queue / mailbox
- `audio capture -> app thread`: lipsync weight mailbox
- `avatar load worker -> app thread`: `mpsc::channel` delivering loaded `AvatarAsset`
- `folder watcher -> app thread`: directory notification channel

## Ownership Rules

- `AvatarAsset` may be shared immutably via `Arc`
- `AvatarInstance` is app-thread-owned
- `ClothState` is app-thread-owned
- `GpuFrameToken` transfers temporary ownership from render to output
- GUI session state is app-thread-owned

## Locking Rules

Keep locking narrow:

- immutable shared assets by `Arc`
- latest tracking sample by single-writer `TrackingMailbox`
- render results by `ResultMailbox` (`Arc<Mutex<Option<RenderResult>>>`)
- bounded output queue by explicit backpressure policy

Avoid broad shared mutable state guarded by coarse global mutexes.

## Shutdown Order

Recommended shutdown:

1. stop new GUI actions
2. stop tracking worker and YOLOX worker capture
3. stop audio capture stream
4. stop output worker intake
5. drain or discard pending output frames
6. stop render thread
7. join background watcher/load threads
8. destroy shared runtime state

## Failure Policy

If one worker fails:

- tracking worker failure should disable tracking, not kill rendering
- output worker failure should disable that sink, not kill preview
- render thread failure is fatal to live preview
