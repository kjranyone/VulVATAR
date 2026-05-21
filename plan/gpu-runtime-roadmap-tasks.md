# GPU runtime roadmap tasks

Date: 2026-05-18  
Last stocktake: 2026-05-21

## Goal

Turn the GPU runtime direction in [`docs/gpu-runtime-roadmap.md`](../docs/gpu-runtime-roadmap.md)
into an executable sequence of work. One architectural rule:

> Large frame-shaped data stays on the GPU; the CPU moves compact control data
> and lifetime tokens.

## Status snapshot (2026-05-21)

| Phase | Workstream | State |
|---|---|---|
| 1 | P1-01 resource counters + steady-state instrumentation | DONE |
| 1 | P1-02 GPU-runtime soak profile | DONE |
| 1 | P1-03 trim formatting-only tracking diffs | DONE (rolled into the merged tracking work) |
| 2 | P2-01 `GpuFrameToken` / `FrameLease` / `ExportedPixelData` boundary | DONE |
| 2 | P2-02 export image pool (`ExportImagePool`) | DONE |
| 2 | P2-03 explicit sync on token path (`OutputSyncToken`, sink ack) | DONE — `ProducerWaitComplete` is the first supported primitive; fence / keyed-mutex variants land alongside the out-of-tree DLL work |
| 2 | P2-04 diagnostics tell the truth (handoff path, pool occupancy) | DONE |
| 2 | P2-05 first live sink (`Win32FileBackedSharedMemorySink`) emits VGTK | DONE on the producer side. MF Virtual Camera DLL consumer is tracked **out-of-tree** in the separate `vulvatar-mf-camera` repository |
| 2 | P2-05 second live sink migration | OPEN — waiting for the DLL slice to ship before touching another sink |
| 3 | P3-01 cloth GPU-state boundary | DONE |
| 3 | P3-02 cloth solver to compute (Verlet + PBD + normals) | DONE |
| 3 | P3-03 `RuntimeGpuBudget` policy + consumers | DONE |
| — | compute prepass migration (skinning + morph + cloth fused) | DONE — see [`docs/vulkano-renderer-design.md`](../docs/vulkano-renderer-design.md) §"Compute prepass" |

## What is actually still open

### Producer-side roadmap

1. **Second live sink GPU-token migration** — once the MF DLL consumer ships,
   the second live sink (currently still on CPU readback by capability flag) is
   the next P2-05 slice. Pick the next sink, audit its `supports_gpu_tokens()`,
   land its sidecar parity tests.
2. **Cross-process sync beyond `ProducerWaitComplete`** — keyed-mutex via
   `Win32Kmt`, D3D12 fence, and external semaphore variants are stubbed in
   `OutputSyncToken` but the producer does not emit them yet. Each variant
   needs (a) a renderer-side export path and (b) the matching DLL contract.

### Out-of-tree

- **MF DLL update to open shared D3D textures by exported handle** —
  tracked in the `vulvatar-mf-camera` repository. Producer-side is complete
  and the wire protocol is documented in
  [`docs/output-interop.md`](../docs/output-interop.md) §"Win32 GPU-handle sidecar".

## Hardware verification (separate plan)

Once the DLL slice lands, vendor-difference verification (NVIDIA / AMD /
Intel Arc) is its own pass — not a sub-task of this roadmap. Capture the
soak metrics from P1-02 on each adapter family.

## Verification matrix (kept as a regression checklist)

| Scenario | Must prove |
|---|---|
| face tracking off → on | morph path no longer creates per-frame buffers |
| cloth off → on | cloth draw path reuses persistent buffers |
| 640p / 720p / 1080p | readback cost scales as expected on the CPU fallback path |
| GPU token vs CPU fallback | output parity and diagnostics correctness |
| 30-minute soak | no allocator leak / no device loss |
| provider variants | DirectML/Vulkan contention is observable via `RuntimeGpuBudget` |
| OBS preview | color metadata and handoff path remain correct |

## Explicit non-goals

- broad renderer module cleanup unrelated to output handoff (#12 of
  [`architecture-pipeline-gui-critical-review.md`](./architecture-pipeline-gui-critical-review.md))
- more tuning of the default CPU readback path beyond correctness/fallback
- per-avatar `RuntimeGpuBudget` policies — single global budget for now

## Related files

- `docs/gpu-runtime-roadmap.md`
- `docs/output-interop.md`
- `docs/profiling.md`
- `src/renderer/mod.rs`
- `src/renderer/output_export.rs`
- `src/output/`
- `src/simulation/`
- `src/tracking/`
- `src/app/runtime_gpu_budget.rs`
