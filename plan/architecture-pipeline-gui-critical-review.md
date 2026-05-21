# VulVATAR architecture / pipeline / GUI — open findings

Original review date: 2026-05-06  
Last stocktake: 2026-05-22

This file used to be the full 1095-line review with the work history of every
fix. The history is in `git log`; this file now only lists the **open**
findings and the recommended order for picking them up. For the closed ones
see commits between `2026-05-06` and `2026-05-22` (architecture review pass +
output refactor + simulation step correctness + GUI repaint gate fix +
P3-03 runtime budget + #1 producer-side capability routing + #4 render-thread
mailbox + #7 tracking hold/fade + #8 mailbox split + #10 partial GuiApp
state split + #12 partial renderer module split + #13 primitive Arc lift).

## Open findings at a glance

| # | Finding | Severity |
|---|---------|----------|
| #10 | `GuiApp` state split is mostly landed; one panel cluster + the runtime/project status group are still top-level fields | Medium |
| #12 | `src/renderer/mod.rs` is down ≈810 lines from the baseline; the `render()` body and pipeline-targets builders still dominate the file | Medium |

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

## #10 — GUI state split (mostly landed)

Status: **mostly landed**. The 2026-05-22 series extracted five new
sub-states on top of the four pre-existing ones:

Pre-existing:
- `TransformState`, `CameraOrbitState`, `TrackingGuiState`,
  `RenderingGuiState`, `OutputGuiState`, `SettingsGuiState`,
  `LipSyncGuiState`

Added in this series:
- `CalibrationUiState` — modal state + preview texture + torso/template
  + target-pose snapshot pipeline
- `LibraryUiState` — search/selection/rename buffers, folder watcher,
  thumbnail generator, pending-thumbnail-jobs queue, avatar-load job
- `ViewportUiState` — rendered-scene texture handle, Blender-style drag
  state, camera-wipe PIP (toggle + texture + dedup seq + RGBA scratch)
- `ClothAuthoringUiState` — sim playback, rename/save buffers, region
  selection, sim-parameter widget values
- `ScenePresetUiState` — preset library + inspector combo + rename buffer

Total: 12 sub-states. The plan's original 7-state target is comfortably
exceeded.

Still top-level on `GuiApp`:

- single-field roles (`mode`, `paused`, `app`, `inspector_open`,
  `animation_playing`, `test_no_persist`, `expression_weights`,
  `camera_index`, `available_cameras`, `hotkeys`, `profiles`,
  `notifications`)
- "RuntimeStatus" cluster: `frame_count`, `last_frame_instant`,
  `frame_time_ms`, `fps`
- "ProjectStatus" cluster: `project_path`, `project_dirty`,
  `last_autosave`, `recovery_manager`, `overlay_dirty`,
  `recent_avatars`, `profiles_dirty`

The remaining clusters are smaller and live closer to the coordinator
role of `GuiApp::update`. Extracting them is incremental cleanup
rather than a structural change; the brittleness pattern the plan
flagged (panels mutating widget state through `GuiApp` directly) is
already broken up.

Files: `src/gui/mod.rs` and the per-panel modules under `src/gui/`.

## #12 — Renderer module size

Status: **partial split landed**. Three sibling submodules now host
distinct concerns of `VulkanRenderer`:

- `renderer::texture_cache` — texture upload + cache (`resolve_*`,
  `upload_*`, `create_default_white_texture`)
- `renderer::readback` — async CPU readback ring
  (`harvest_pending*`, `ensure_readback_buffers`,
  `PendingReadbackState`, `READBACK_RING_SIZE`)
- `renderer::cloth_cache` — GPU cloth solver slot allocator
  (`ensure_cloth_gpu_slot`, `allocate_cloth_constraint_resources`,
  `allocate_cloth_normal_resources`)

Each submodule reuses the same `impl VulkanRenderer` pattern, so
call sites in `mod.rs` are unchanged. `renderer/mod.rs` shrank from
3883 to roughly 3074 lines (≈ 810 lines extracted; numbers will
keep moving as the remaining slices land).

Recommendation for follow-up slices:

- `transform_cache` — `ensure_transform_data` + `ensure_stub_ssbo`
  + the `TransformGpuData` struct definitions
- `pipeline_targets` — `rebuild_pipelines_and_targets` /
  `build_render_pass` / `create_offscreen_targets`
- `draw` — the ~900-line `render()` body. Most tightly coupled to
  command-buffer state; needs its own analysis pass before
  extraction.

Keep `VulkanRenderer` as the facade.

## Suggested order

Both remaining items are now medium-severity incremental work
rather than structural blockers. Pick them up when nearby code is
already being touched.

## What this file is NOT

- A history of the closed findings (those are in commits / `git log`).
- A roadmap of GPU runtime work (see
  [`gpu-runtime-roadmap-tasks.md`](./gpu-runtime-roadmap-tasks.md)).
- A description of design intent (see `docs/architecture.md` and
  `docs/threading-model.md`).
