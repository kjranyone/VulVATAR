# VulVATAR architecture / pipeline / GUI — open findings

Original review date: 2026-05-06  
Last stocktake: 2026-05-21

This file used to be the full 1095-line review with the work history of every
fix. The history is in `git log`; this file now only lists the **open**
findings and the recommended order for picking them up. For the closed ones
see commits between `2026-05-06` and `2026-05-21` (architecture review pass +
output refactor + simulation step correctness + GUI repaint gate fix).

## Open findings at a glance

| # | Finding | Severity |
|---|---------|----------|
| #1  | Live output is CPU-readback-centred (producer-side capability routing landed; second live sink still on CPU) | High |
| #4  | Render thread result delivery still uses blocking `result_tx.send`; result side has no latest-frame semantics | High |
| #7  | Stale tracking samples cause a hard pose reset rather than hold/fade | High |
| #8  | Tracking mailbox clones relatively heavy GUI data (pose + webcam preview + diagnostics) under one mutex | Medium |
| #10 | `GuiApp` has been partially split into sub-states but is still a broad mutable coordinator | High |
| #12 | `src/renderer/mod.rs` is ~3900 lines and has too many reasons to change | Medium |
| #13 | `AvatarInstance` stores `primitive_arcs`, blurring asset / runtime / render boundaries | Medium |

## #1 — Second live sink not yet on GPU handoff

Status: **PARTIAL.** Capability-based routing landed (P2-05 phase 1); the
first live sink (`Win32FileBackedSharedMemorySink` for `VirtualCamera` /
`SharedMemory`) now publishes VGTK sidecars when the consumer can read
them. The second live sink is still on CPU readback until the MF DLL slice
lands and a follow-up sink is chosen.

Recommendation: wait for the out-of-tree DLL ship, then pick the next sink
in [`gpu-runtime-roadmap-tasks.md`](./gpu-runtime-roadmap-tasks.md) §"What
is actually still open".

## #4 — Result-side render thread backpressure

Current state: `src/app/render_thread.rs` uses `sync_channel(2)` + `try_send`
on the command side (good), but result delivery still uses blocking
`result_tx.send(result)`. A modal loop, long UI work, or an asset-load
hiccup that delays draining results can stall the render thread.

Recommendation:

- Make result delivery use `sync_channel(1)` + `try_send` with
  replace-latest semantics, OR a mailbox (`Arc<Mutex<Option<RenderResult>>>`).
- Track dropped render results in diagnostics so the regression is visible.

Files: `src/app/render_thread.rs`, `src/app/render.rs`.

## #7 — Stale tracking causes hard pose reset

Current state: `src/app/render.rs` treats `tracking.is_stale()` as `None`
and rebuilds base/rest pose every frame. `Application::last_tracking_pose`
exists (set when a fresh sample arrives) but is never *consumed* on a
stale frame.

Recommendation: add a latched tracking sample with age metadata and a
three-state policy:

- fresh: solve normally
- stale but within hold window: reuse `last_tracking_pose` with decaying
  confidence
- expired: fade from last solved pose toward animation / rest

One stale frame must not invalidate all pose channels at once.

Files: `src/app/render.rs`, possibly a small new policy module.

## #8 — Tracking mailbox does too much under one mutex

Current state: `src/tracking/mod.rs` `TrackingMailboxInner` holds pose,
webcam preview frame, annotation, calibration, and torso template under
one `Arc<Mutex<…>>`. The webcam preview clone happens on every inference
frame.

Recommendation: split into separate mailboxes (pose / camera preview /
diagnostics / calibration command). Use `Arc<Vec<u8>>` for preview frame
payloads if snapshots continue to clone them.

Files: `src/tracking/mod.rs` (and call sites in
`src/app/tracking_lifecycle.rs`, `src/gui/inspector/tracking.rs`).

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

## #13 — Avatar instance carries render-facing primitives

Current state: `src/avatar/instance.rs` stores `primitive_arcs:
Vec<Vec<Arc<MeshPrimitiveAsset>>>` so that the render snapshot does not
clone vertex data every frame. It works, but avatar runtime now carries
renderer-snapshot optimisation data — not a GPU type leak, but it blurs
the asset / runtime / render boundary.

Recommendation:

- Move primitive arc / cache ownership into immutable asset construction.
- Or build a `RenderAssetView` once per loaded `AvatarAsset` and share it
  through render snapshot construction.

Files: `src/avatar/instance.rs`, `src/app/render.rs` (snapshot build sites).

## Suggested order

1. **#7** — most user-visible runtime issue still open; isolated to
   `app::run_frame` + a small bit of state.
2. **#4** — small change with high robustness payoff; unblocks longer UI
   work without render thread stalls.
3. **#8** — independent of the others; smaller blast radius.
4. **#10** — large mechanical refactor; pick up after the runtime issues
   so the split lands on a stable contract.
5. **#13** — incremental cleanup; pair with whichever asset path is being
   touched next.
6. **#12** — the renderer split is the biggest chunk; defer until the
   compute prepass and GPU cloth paths stop churning.

## What this file is NOT

- A history of the closed findings (those are in commits / `git log`).
- A roadmap of GPU runtime work (see
  [`gpu-runtime-roadmap-tasks.md`](./gpu-runtime-roadmap-tasks.md)).
- A description of design intent (see `docs/architecture.md` and
  `docs/threading-model.md`).
