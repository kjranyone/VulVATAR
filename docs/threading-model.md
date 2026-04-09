# Threading Model

## Purpose

This document defines concurrency boundaries for VulVATAR.

The goal is to make async tracking and async output explicit without making the rest of the system accidentally multithreaded.

Related documents:

- [architecture.md](/C:/lib/github/kjranyone/VulVATAR/docs/architecture.md)
- [tracking-retargeting.md](/C:/lib/github/kjranyone/VulVATAR/docs/tracking-retargeting.md)
- [output-interop.md](/C:/lib/github/kjranyone/VulVATAR/docs/output-interop.md)

## Thread Roles

Recommended first runtime threads:

1. app thread
2. render thread
3. tracking worker
4. output worker

## `app thread`

Responsibilities:

- own application state
- run mode transitions
- update avatar runtime
- step animation and simulation scheduling
- read latest tracking sample
- submit render work

Rules:

- this is the owner of `AvatarInstance`
- this is the owner of project state
- simulation state should not be mutated concurrently from other threads

## `render thread`

Responsibilities:

- own renderer-facing GPU state
- execute frame rendering
- publish completed output frames

Rules:

- GPU resource lifetime is decided here
- render thread does not own scene logic
- render thread should accept snapshots or commands, not mutable scene ownership

## `tracking worker`

Responsibilities:

- acquire webcam frames
- run inference
- normalize landmarks
- publish latest `TrackingRigPose`

Rules:

- never mutate avatar runtime state directly
- use latest-sample mailbox semantics
- dropping stale tracking samples is acceptable

## `output worker`

Responsibilities:

- consume `OutputFrame` tokens
- communicate with sink-specific APIs
- enforce output queue policy

Rules:

- never block the app thread waiting for sink completion
- never access renderer-owned GPU images without a valid exported token
- drop or replace frames according to explicit policy when overloaded

## Communication Channels

Recommended first channels:

- `tracking worker -> app thread`: latest-sample mailbox
- `app thread -> render thread`: frame submission queue or immediate command boundary
- `render thread -> output worker`: bounded frame token queue

## Ownership Rules

- `AvatarAsset` may be shared immutably
- `AvatarInstance` is app-thread-owned
- `ClothState` is app-thread-owned unless cloth solve is explicitly moved later
- `GpuFrameToken` transfers temporary ownership from render to output
- GUI session state is app-thread-owned

## Locking Rules

Keep locking narrow.

Recommended first approach:

- immutable shared assets by `Arc`
- latest tracking sample by single-writer mailbox
- bounded output queue by explicit backpressure policy

Avoid broad shared mutable state guarded by coarse global mutexes.

## Shutdown Order

Recommended shutdown:

1. stop new GUI actions
2. stop tracking worker capture
3. stop output worker intake
4. drain or discard pending output frames
5. stop render thread
6. destroy shared runtime state

## Failure Policy

If one worker fails:

- tracking worker failure should disable tracking, not kill rendering
- output worker failure should disable that sink, not kill preview
- render thread failure is fatal to live preview

## Future Expansion

Possible later changes:

- dedicated simulation thread
- async asset import worker
- background cloth bake worker

These should be added only with explicit ownership updates to this document.
