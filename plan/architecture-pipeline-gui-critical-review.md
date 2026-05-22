# VulVATAR architecture / pipeline / GUI — open findings

Original review date: 2026-05-06  
Last stocktake: 2026-05-22

This file used to be the full 1095-line review with the work history of every
fix. The history is in `git log`; this file now only lists the **open**
findings and the recommended order for picking them up. For the closed ones
see commits between `2026-05-06` and `2026-05-22` (architecture review pass +
output refactor + simulation step correctness + GUI repaint gate fix +
P3-03 runtime budget + #1 producer-side capability routing + #4 render-thread
mailbox + #7 tracking hold/fade + #8 mailbox split + #10 GuiApp state split +
#12 renderer module split + #13 primitive Arc lift).

## Open findings at a glance

| # | Finding | Severity |
|---|---------|----------|
| #12 | `renderer/mod.rs` ≈ 2600 lines; the ~900-line `render()` body still dominates | Low |

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

#10 (GuiApp state split) is closed: 14 sub-state structs now own the
widget buffers, with `GuiApp` reduced to a coordinator that holds the
sub-states plus a small set of genuinely top-level fields. See "Closed"
section below for the breakdown.

## Closed: #10 — GUI state split

14 sub-state structs split out of `GuiApp` (commit series
`4e50e33 … 8729897`). Final shape:

Pre-existing (kept):
- `TransformState`, `CameraOrbitState`,
  `TrackingGuiState`, `RenderingGuiState`, `OutputGuiState`,
  `SettingsGuiState`, `LipSyncGuiState`

Added in this split:
- `CalibrationUiState` — modal state, preview texture, torso template
  shadow, calibration target-pose snapshot pipeline
- `LibraryUiState` — search/selection/rename buffers, folder watcher,
  thumbnail generator, pending-thumbnail-jobs queue, avatar-load job
- `ViewportUiState` — rendered-scene texture handle, Blender-style drag
  state, camera-wipe PIP (toggle + texture + dedup seq + RGBA scratch)
- `ClothAuthoringUiState` — sim playback, rename/save buffers, region
  selection, sim-parameter widget values
- `ScenePresetUiState` — preset library + inspector combo + rename buffer
- `RuntimeStatusUi` — per-frame timing + pause state (`paused`,
  `frame_count`, `last_frame_instant`, `frame_time_ms`, `fps`)
- `ProjectStatusUi` — project lifecycle (`project_path`, `recent_avatars`,
  `project_dirty`, `last_autosave`, `recovery_manager`, `overlay_dirty`,
  `profiles_dirty`)

Still top-level on `GuiApp` (intentional — these are single-field roles
or "coordinator owns" data that doesn't fit a panel cluster):
`mode`, `app`, `inspector_open`, `animation_playing`,
`test_no_persist`, `expression_weights`, `camera_index`,
`available_cameras`, `hotkeys`, `profiles`, `notifications`.

The brittleness pattern the original review flagged (panels mutating
widget state through `GuiApp` directly, `GuiApp::update` reconciling
fields into `Application` every frame) is now contained — each panel
mutates only its own sub-state struct.

## #12 — Renderer module size (residual)

Status: **substantial split landed**. Five sibling submodules now host
distinct concerns of `VulkanRenderer`:

- `renderer::texture_cache` — texture upload + cache
  (`resolve_*`, `upload_*`, `create_default_white_texture`)
- `renderer::readback` — async CPU readback ring
  (`harvest_pending*`, `ensure_readback_buffers`,
  `PendingReadbackState`, `READBACK_RING_SIZE`)
- `renderer::cloth_cache` — GPU cloth solver slot allocator
  (`ensure_cloth_gpu_slot`, `allocate_cloth_constraint_resources`,
  `allocate_cloth_normal_resources`)
- `renderer::transform_cache` — compute-prepass per-primitive resource
  builder (`ensure_transform_data`, `ensure_stub_ssbo`)
- `renderer::pipeline_targets` — render-pass / pipeline / offscreen-
  target construction (`build_render_pass`,
  `create_offscreen_targets`, `rebuild_pipelines_and_targets`)

Each submodule reuses the same `impl VulkanRenderer` pattern, so call
sites in `mod.rs` are unchanged. `renderer/mod.rs` shrank from 3883
to roughly 2599 lines (≈ 1280 lines extracted, 33% reduction).

What's left in `mod.rs` at this point:

- The `render()` body itself (~900 lines) — tightly coupled to
  command-buffer state, would need its own analysis pass before
  extraction.
- `render_thumbnail` / `render_to_thumbnail` — share infrastructure
  with `render()`; split alongside it.
- The `VulkanRenderer` struct definition.
- Small helpers: `color_attachment_format`, `apply_color_space`,
  `resize`, `clear_caches`, `release_export_lease`,
  `get_or_update_skinning`.
- `CameraRing` + small `mat4_to_cols` / `mat4_cols_identity` helpers.

Severity downgraded High → Low: the remaining content is either
genuinely coordination-shaped (the render loop) or small helpers that
don't justify their own modules. Further splitting is incremental
cleanup if the render loop ever needs major restructuring.

## What this file is NOT

- A history of the closed findings (those are in commits / `git log`).
- A roadmap of GPU runtime work (see
  [`gpu-runtime-roadmap-tasks.md`](./gpu-runtime-roadmap-tasks.md)).
- A SOTA-direction tracker (see
  [`sota-algorithm-upgrades.md`](./sota-algorithm-upgrades.md)).
- A description of design intent (see `docs/architecture.md` and
  `docs/threading-model.md`).
