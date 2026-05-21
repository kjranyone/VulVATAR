# VulVATAR architecture / pipeline / GUI — open findings

Original review date: 2026-05-06  
Last stocktake: 2026-05-22

This file used to be the full 1095-line review with the work history of every
fix. The history is in `git log`; this file now only lists the **open**
findings and the recommended order for picking them up. For the closed ones
see commits between `2026-05-06` and `2026-05-22` (architecture review pass +
output refactor + simulation step correctness + GUI repaint gate fix +
P3-03 runtime budget + #1 producer-side capability routing + #4 render-thread
mailbox + #7 tracking hold/fade + #8 mailbox split + #13 primitive Arc lift).

## Open findings at a glance

| # | Finding | Severity |
|---|---------|----------|
| #10 | `GuiApp` has been partially split into sub-states but is still a broad mutable coordinator | High |
| #12 | `src/renderer/mod.rs` is ~3900 lines and has too many reasons to change | Medium |

#1 (live output GPU handoff) is closed on the producer side: every
non-`ImageSequence` sink on Windows
(`VirtualCamera` / `SharedMemory` / `SharedTextureFileStub`) advertises
GPU-token support via `Win32FileBackedSharedMemorySink`, and the export
path picks `RenderExportMode::GpuExport` whenever the active sink agrees.
The remaining unfinished work is the MF Virtual Camera **DLL consumer**
(tracked out-of-tree in the `vulvatar-mf-camera` repository) and the
additional cross-process sync primitives — both items live in
[`gpu-runtime-roadmap-tasks.md`](./gpu-runtime-roadmap-tasks.md) §"What
is actually still open" rather than here.

## #10 — GUI state split is partial

Current state: `GuiApp` has had a few sub-states extracted (`TrackingGuiState`,
`RenderingGuiState`, `OutputGuiState` per the verification pass), but the
broader pattern — most panels mutating `GuiApp` directly and `GuiApp::update`
reconciling fields into `Application` every frame — is unchanged.

Recommendation: continue the slice, prioritising panels with
requested-vs-active mismatches:

1. Output panel state and reconciliation (partly done already)
2. Tracking panel state next
3. Keep `GuiApp` as a coordinator and owner of top-level sub-states only

Target shape (kept here as the reference contract):

- `ProjectUiModel`
- `RuntimeStatus`
- `TrackingPanelState`
- `RenderingPanelState`
- `OutputPanelState`
- `CalibrationUiState`
- `LibraryUiState`

Files: `src/gui/mod.rs` and panel modules under `src/gui/`.

## #12 — Renderer module size

Current state: `src/renderer/mod.rs` is approximately 3900 lines. It owns
initialisation, pipeline selection, render pass, resource caches, readback,
thumbnails, debug-ish paths, texture resolution, transform compute prepass,
cloth GPU dispatch wiring, and placeholder geometry. Touching colour space,
thumbnail rendering, output export, or mesh upload all share one file.

Recommendation: split by execution path:

- `device.rs` / `context.rs`
- `frame.rs`
- `readback.rs`
- `draw.rs`
- `thumbnail.rs` (already exists but still shares substantial code in `mod.rs`)
- `mesh_cache.rs`
- `texture_cache.rs`

Keep `VulkanRenderer` as the facade.

## Suggested order

1. **#10** — biggest open item, large mechanical refactor; pick up
   when behaviour churn has settled enough that the sub-state split
   doesn't immediately need to be revisited.
2. **#12** — the renderer split is comparably large; defer until the
   compute prepass and GPU cloth paths stop churning.

## What this file is NOT

- A history of the closed findings (those are in commits / `git log`).
- A roadmap of GPU runtime work (see
  [`gpu-runtime-roadmap-tasks.md`](./gpu-runtime-roadmap-tasks.md)).
- A description of design intent (see `docs/architecture.md` and
  `docs/threading-model.md`).
