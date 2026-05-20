# GPU runtime roadmap tasks

Date: 2026-05-18

## Goal

Turn the GPU runtime direction in [`docs/gpu-runtime-roadmap.md`](../docs/gpu-runtime-roadmap.md)
into an executable sequence of work.

The renderer has already crossed the first important line:

- morph targets are now GPU-pulled
- cloth draw buffers are persistent and rewritten in place
- DirectML session setup and tracking-worker restart behavior have been hardened

The remaining work is no longer "fix one crash path." It is to make the whole
runtime obey one architectural rule:

> Large frame-shaped data stays on the GPU; the CPU moves compact control data
> and lifetime tokens.

## Status at a glance

| Workstream | Status | Notes |
|---|---|---|
| Morph GPU pulling | DONE | landed in `src/renderer/{mod,pipeline}.rs` |
| Cloth draw-buffer reuse | DONE | landed in `src/renderer/mod.rs` |
| DirectML EP contract | DONE | explicit sequential mode + memory-pattern disable |
| Tracking worker overlap guard | DONE | restart is refused while old worker is still alive |
| GPU-first output | OPEN | next critical path |
| Preview GPU residency | OPEN | follows output-token boundary |
| Cloth compute | OPEN | should wait until output path is no longer CPU-centred |
| Shared GPU budget policy | OPEN | should become explicit after output path is visible |

## Critical path

```text
1. stabilize + instrument
2. introduce GpuFrameToken boundary
3. add export-image pool + explicit sync
4. switch live output sinks to GPU-first handoff
5. move dynamic simulation deeper onto GPU
6. unify GPU pacing policy
```

The next indispensable task is **not** further readback tuning. It is:

> Build the `GpuFrameToken` / export-image-pool boundary while keeping the
> current CPU readback path as an explicit fallback.

## Phase 1 — Stabilize the new baseline

### P1-01 — Confirm no steady-state dynamic-buffer allocation remains

**Why**

The recent crash fix removed the suspected allocator churn. Before building on
top of it, verify the invariant explicitly so later regressions are obvious.

**Scope**

- `src/renderer/mod.rs`
- profiling / diagnostics hooks

**Tasks**

- [x] Add counters for:
  - [x] morph GPU resource creations
  - [x] morph UBO writes
  - [x] cloth cache creations
  - [x] cloth VBO writes
- [x] Expose the counters in periodic profile logs.
- [ ] Verify steady-state:
  - [ ] morph resource creations stop after avatar warm-up
  - [ ] cloth cache creations stop after first use
  - [ ] only small writes continue per frame

**Done when**

- a 60-second tracked run proves resources are reused rather than recreated
- a future regression would be visible in logs without a debugger

### P1-02 — Add a GPU-runtime soak profile

**Why**

This project just learned that "works for a few seconds" and "stable as a live
avatar runtime" are not the same thing.

**Scope**

- `docs/profiling.md`
- optional helper under `scripts/` if useful

**Tasks**

- [x] Document a repeatable 30-minute soak scenario:
  - [x] face tracking on
  - [x] cloth on
  - [x] 1080p output
  - [x] virtual camera / output sink active
- [x] Record the metrics to inspect:
  - [x] frame time
  - [x] readback bytes/frame
  - [x] output queue depth
  - [x] dropped frames
  - [x] GPU-resource creation counters
- [x] Record failure signatures:
  - [x] device loss
  - [x] worker restart refusal
  - [x] output fallback activation

**Done when**

- there is one canonical soak recipe the project can rerun after GPU-path changes

### P1-03 — Trim formatting-only tracking diffs before merge

**Why**

The current worktree contains many unrelated rustfmt-style changes in tracking
files. They obscure review signal and make archaeology harder.

**Scope**

- `src/tracking/**`
- `src/app/render.rs`

**Tasks**

- [ ] Separate semantic changes from formatting-only changes.
- [ ] Keep the real DirectML / lifecycle changes.
- [ ] Avoid bundling unrelated formatting churn into the GPU-runtime merge.

**Done when**

- the review diff says what changed, not what rustfmt would prefer today

## Phase 2 — GPU-first output

### P2-01 — Define the exported-frame boundary

**Why**

Today the docs describe GPU-resident handoff, but the main runtime still
centres CPU readback. The code needs a type boundary that makes the intended
future hard to violate.

**Scope**

- `src/renderer/output_export.rs`
- `src/output/mod.rs`
- `src/renderer/frame_input.rs` only if result typing needs it

**Tasks**

- [ ] Introduce:
  - [x] `GpuFrameToken`
  - [x] `OutputSyncToken`
  - [x] `FrameLease`
  - [x] `ExportedPixelData::{GpuFrameToken, CpuReadback}`
- [ ] Make lifetime ownership explicit:
  - [x] who owns the image
  - [x] when the sink may consume it
  - [x] when the renderer may recycle it
- [x] Keep CPU readback as a first-class fallback, not a hidden side path.

**Done when**

- output code can talk about a frame without receiving renderer-owned Vulkan
  internals directly

### P2-02 — Introduce an export-image pool

**Why**

GPU handoff needs reusable image identity and backpressure discipline. Ad hoc
allocation would simply recreate the old problem one layer later.

**Scope**

- `src/renderer/output_export.rs`
- possibly `src/renderer/frame_pool.rs`

**Tasks**

- [x] Add `ExportImagePool` / slot states:
  - [x] available
  - [x] rendering
  - [x] leased to sink
- [x] Bound the pool size.
- [x] Choose and document queue policy:
  - [ ] replace latest
  - [ ] drop oldest
  - [x] never block app thread
- [x] Recycle only after lease completion.

**Done when**

- the renderer can hand out GPU frame leases without leaking ownership or
  growing memory unboundedly

### P2-03 — Add explicit synchronization to the token path

**Why**

"Submitted" is not the same as "safe to consume." A GPU-first path without
sync is just a quieter bug.

**Scope**

- `src/renderer/output_export.rs`
- sink-specific code in `src/output/`

**Tasks**

- [ ] Choose the first supported sync primitive for the Windows path.
- [x] Store readiness metadata in `OutputSyncToken`.
- [x] Define sink-completion acknowledgement.
- [ ] Add failure handling:
  - [ ] if sync export fails, fall back visibly
  - [ ] local preview remains alive

**Done when**

- exported frames have an explicit readiness/lifetime story end to end

### P2-04 — Make diagnostics tell the truth

**Why**

This project has already paid for misleading labels once. The UI must say
whether it is actually on GPU handoff or CPU fallback.

**Scope**

- `src/output/`
- `src/gui/inspector/output.rs`

**Tasks**

- [ ] Surface:
  - [x] active handoff mode
  - [x] pool occupancy
  - [x] queue depth
  - [x] dropped frame count
  - [x] fallback reason
- [x] Keep diagnostics tied to published frames, not sink names.

**Done when**

- a user can tell within seconds whether the fast path is genuinely active

### P2-05 — Switch live output sinks one by one

**Why**

This is where architecture becomes runtime truth.

**Scope**

- `src/output/`
- `src/renderer/output_export.rs`

**Tasks**

- [ ] Keep thumbnails / image-sequence export on CPU readback.
- [x] Add the first GPU-token-capable bridge sink (`SharedTextureFileStub`)
      while preserving CPU fallback bytes.
- [ ] Move the first live sink to GPU token consumption.
- [ ] Preserve fallback parity.
- [ ] Repeat for the next live sink only after the first one is stable.

**Done when**

- a supported live-output path no longer requires per-frame CPU readback

## Phase 3 — GPU simulation + pacing

### P3-01 — Define the cloth GPU-state boundary

**Why**

Cloth draw upload is fixed, but cloth simulation is still CPU-owned. Moving it
later will be easier if the boundary is named now.

**Scope**

- `src/simulation/`
- `src/renderer/`
- design docs

**Tasks**

- [ ] Define a future-facing `ClothGpuState` shape.
- [ ] Distinguish:
  - [ ] authoring data
  - [ ] CPU fallback simulation state
  - [ ] GPU simulation state
  - [ ] render-consumable deform buffer
- [ ] Avoid changing avatar ownership rules just to accommodate compute.

**Done when**

- compute cloth can be added without redrawing the app/renderer boundary

### P3-02 — Move cloth solver stages to compute incrementally

**Why**

The safe path is staged, not heroic.

**Tasks**

- [ ] particle integration compute pass
- [ ] constraint projection pass
- [ ] normal/deform output pass
- [ ] direct draw consumption of GPU-produced deformation
- [ ] CPU fallback retained for diagnostics / unsupported devices

**Done when**

- the main cloth path no longer ships full deformed geometry through CPU each
  frame

### P3-03 — Add one runtime GPU budget policy

**Why**

DirectML and Vulkan share one physical machine. Their cadence decisions should
stop being scattered folklore.

**Scope**

- `src/tracking/`
- `src/app/`
- possibly a new runtime policy module

**Tasks**

- [ ] Introduce a compact `RuntimeGpuBudget` policy.
- [ ] Centralize:
  - [ ] render FPS target
  - [ ] pose target Hz
  - [ ] YOLOX refresh period
  - [ ] depth refresh period
  - [ ] FaceMesh EP choice
- [ ] Feed it live measurements:
  - [ ] render timings
  - [ ] provider timings
  - [ ] output pressure
- [ ] Define intentional degraded modes.

**Done when**

- GPU contention can be explained and tuned from one place

## Verification matrix

| Scenario | Must prove |
|---|---|
| face tracking off → on | morph path no longer creates per-frame buffers |
| cloth off → on | cloth draw path reuses persistent buffers |
| 640p / 720p / 1080p | readback cost becomes visible and scales as expected |
| GPU token vs CPU fallback | output parity and diagnostics correctness |
| 30-minute soak | no allocator leak / no device loss |
| provider variants | DirectML/Vulkan contention is observable |
| OBS preview | color metadata and handoff path remain correct |

## Suggested implementation order for the next agent

1. P1-01 resource counters
2. P2-01 exported-frame boundary
3. P2-02 export-image pool
4. P2-04 diagnostics
5. P2-03 sync primitive
6. P2-05 first live sink migration

This order keeps the system inspectable while moving it toward the real
architecture.

## Explicit non-goals for the next pass

- full cloth compute rewrite
- GUI preview rewrite
- broad renderer module cleanup unrelated to output handoff
- more tuning of the default CPU readback path beyond correctness/fallback

## Per-task plans

Each open workstream below has its own plan document:

- P2-05 phase 2 (live Win32 sink GPU token consumption) →
  [`win32-sink-gpu-token-consumption.md`](./win32-sink-gpu-token-consumption.md)
- Composed output sink coverage →
  [`output-sink-integration-tests.md`](./output-sink-integration-tests.md)
- P3-02 (cloth solver to compute) →
  [`gpu-cloth-solver-compute-migration.md`](./gpu-cloth-solver-compute-migration.md)
- P3-03 (runtime GPU budget) →
  [`runtime-gpu-budget.md`](./runtime-gpu-budget.md)

Per-task plans are the source of truth for scope, tasks, and done-when.
This roadmap stays the high-level critical path.

## Related files

- `docs/gpu-runtime-roadmap.md`
- `docs/output-interop.md`
- `docs/profiling.md`
- `src/renderer/mod.rs`
- `src/renderer/output_export.rs`
- `src/output/`
- `src/simulation/`
- `src/tracking/`
