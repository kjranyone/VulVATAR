# T10: Future Extensions

## Priority: P3

## Source

- `docs/application-design.md` — future extensions section
- `docs/editor-cloth-authoring.md` — future extensions

## Status

This is the project's standing backlog of "nice to have" extensions.
Most application-level and library items have landed; remaining work
clusters around webcam tracking, renderer color management, and a few
cloth-authoring polish items that each need their own design pass
before implementation. Mark items completed as they ship, with a
short note pointing at the relevant code.

Legend: `[x]` done, `[~]` partial, `[ ]` not started.

## Items

### Application-Level

- [x] **Multiple avatars** — support more than one avatar instance in the scene (`run_frame` iterates `self.avatars`, `build_frame_input_multi` renders all)
- [x] **Scene presets** — save and load scene configurations
- [x] **Expression control panels** — direct blendshape/expression editing UI
- [x] **Hotkey support** — configurable keyboard shortcuts
- [x] **Preset export/import** — share rendering/tracking presets between setups (ProfileLibrary::export_to_file / import_from_file implemented)
- [x] **Profile switching** — quick-switch between streaming configurations

### Cloth Authoring

- [x] **Multiple cloth overlays per avatar** — data model already plural; UI now shows per-slot id/particle-count with Remove button, "Load Overlay File..." attaches a .vvtcloth directly into a new slot, and the attached set is persisted to ProjectFile.cloth_overlay_paths so save/load round-trips
- [x] **Cloth LoD authoring** — ClothLoDLevel + ClothLoDConfig + distance-based auto-select + simulation integration
- [x] **Wind authoring tools** — directional/gust wind presets (WindPresetLibrary with 4 presets + WindPreset::sample_at)
- [x] **Overlay version migration** — load_cloth_overlay / load_project now run a two-stage parse (Value → version check → optional migration → typed deserialize). Future-version files reject with a clear "newer VulVATAR" error; older files walk a `step_*` migration chain. Plumbing is in place + tested; no actual migrators registered yet because every released file is at v1.
- [ ] **Partial rebinding after avatar reimport** — smart re-link when avatar changes
- [ ] **Offline bake / cache generation** — pre-compute simulation data for testing

### Renderer

- [x] **Image background** — load image as viewport background (RenderFrameInput.background_image_path)
- [x] **Ground alignment helper** — snap avatar to ground plane (Application::snap_avatar_to_ground)
- [ ] **Output color space (sRGB / Linear sRGB)** — GUI → app wiring is trivial,
  but `src/renderer/output_export.rs` hardcodes `color_space: "srgb"`. A real
  switch needs render-target format selection, MToon shader output gamma
  branching, and MF sample `MF_MT_VIDEO_PRIMARIES` / `MF_MT_TRANSFER_FUNCTION`
  attributes — i.e. an end-to-end renderer color-management pass, not a single
  combo wire-up. Originally tracked as T11 phase B-5 and deferred there as
  out-of-scope for that task.

### Webcam Tracking

- [x] **Hand tracking** — finger-level tracking from webcam. CIGPose ubody/wholebody emits 21 keypoints per hand; `map_hand_keypoints` in `src/tracking/inference.rs` writes the proximal/intermediate/distal of all five fingers per hand into `SourceSkeleton.joints` and the fingertips into `SourceSkeleton.fingertips`, with the mirror swap (user's anatomical left → avatar's right). The pose solver's `DRIVEN_BONES` enumerates each finger bone and uses `Tip::Fingertip` for the distals so the chain solves base→tip. End-to-end: open hand / fist / peace sign all visibly drive the avatar.
- [x] **Lower-body tracking** — opt-in flag (`TrackingGuiState.lower_body_tracking_enabled`) flips `pose_model_score` so the model picker prefers a `*_coco-wholebody` CIGPose variant over `*_coco-ubody`. Hip/knee/ankle keypoints were already wired through inference + the pose solver, so flipping the model choice is enough. Toggle is start-time only (apply by restarting tracking) and surfaced as a checkbox under Retargeting with a tradeoff tooltip.
- [ ] **GPU inference backend** — `ort` builds against CPU EP today. Switching to DirectML / CUDA needs an `ExecutionProvider` choice in `Session::builder()`, a runtime fallback when the GPU EP fails to create, and a build flag to gate the backend so headless CI keeps passing without GPU drivers.

### Model Library

- [~] **Thumbnail generation** — placeholder thumbnails (procedural circle + 5x7 pixel-font label) generated for every library entry on load and on legacy backfill, displayed at 64×64 in the library inspector via egui_extras file:// loader. **Real rendered thumbnails** (actual VRM avatar pose rendered offscreen) remain TODO — needs a `RenderCommand::Thumbnail` variant on the render thread + readback wiring.
- [x] **Drag-and-drop import** — drop VRM files onto window (egui raw.dropped_files)
- [x] **Folder watching** — auto-update library when files change (FolderWatcher with notify backend, inspector UI to add/remove watched dirs, persisted to %APPDATA%\\VulVATAR\\watched_folders.json)
- [x] **VRM metadata import** — title, author, version, thumbnail from VRM extensions (fields on AvatarLibraryEntry, surfaced in library inspector; thumbnail bytes saved to `thumbnail_dir`)
- [x] **Library categories/folders** — organize entries into groups (AvatarLibrary::categories / by_category)
- [x] **Library catalog export/import** — share library databases (export_catalog / import_catalog)

## Notes

This file is the project's only active backlog (older T01–T11 plans
were retired once their work shipped — see git history for that
context). Promote an item to its own dedicated plan file only when
the work needs cross-session design discussion (e.g. partial
rebinding, offline bake), or when the implementation is large enough
that a single git commit message wouldn't capture the design.
