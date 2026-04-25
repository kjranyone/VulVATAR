# T10: Future Extensions

## Priority: P3

## Source

- `docs/application-design.md` — future extensions section
- `docs/editor-cloth-authoring.md` — future extensions

## Status

**IMPLEMENTED**

Hotkey system (`src/gui/hotkey.rs`) wired into the update loop with 9 default bindings:
TogglePause (Space), ToggleTracking (Ctrl+T), ToggleCloth (Ctrl+C), ResetPose (Ctrl+Shift+R),
ResetCamera (Home), SwitchModePreview (F5), SwitchModeAuthoring (F6), SaveProject (Ctrl+S),
LoadAvatar (Ctrl+O).

Profile system (`src/gui/profile.rs`) with 3 presets (Streaming, Recording, Performance)
and a profile selector dropdown in the top bar that live-applies tracking/rendering/output
settings to the GUI state.
Remaining items below are still deferred.

## Items

### Application-Level

- [x] **Multiple avatars** — support more than one avatar instance in the scene (`run_frame` iterates `self.avatars`, `build_frame_input_multi` renders all)
- [x] **Scene presets** — save and load scene configurations
- [x] **Expression control panels** — direct blendshape/expression editing UI
- [x] **Hotkey support** — configurable keyboard shortcuts
- [x] **Preset export/import** — share rendering/tracking presets between setups (ProfileLibrary::export_to_file / import_from_file implemented)
- [x] **Profile switching** — quick-switch between streaming configurations

### Cloth Authoring

- [ ] **Multiple cloth overlays per avatar** — attach more than one overlay
- [x] **Cloth LoD authoring** — ClothLoDLevel + ClothLoDConfig + distance-based auto-select + simulation integration
- [x] **Wind authoring tools** — directional/gust wind presets (WindPresetLibrary with 4 presets + WindPreset::sample_at)
- [ ] **Overlay version migration** — upgrade old overlay formats
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

- [ ] **Hand tracking** — finger-level tracking from webcam
- [ ] **Lower-body tracking** — legs and feet from webcam (if supported by CV stack)
- [ ] **GPU inference backend** — run inference on GPU instead of CPU

### Model Library

- [ ] **Thumbnail generation** — render avatar thumbnails for library browser
- [x] **Drag-and-drop import** — drop VRM files onto window (egui raw.dropped_files)
- [ ] **Folder watching** — auto-update library when files change
- [x] **VRM metadata import** — title, author, version, thumbnail from VRM extensions (fields on AvatarLibraryEntry, surfaced in library inspector; thumbnail bytes saved to `thumbnail_dir`)
- [x] **Library categories/folders** — organize entries into groups (AvatarLibrary::categories / by_category)
- [x] **Library catalog export/import** — share library databases (export_catalog / import_catalog)

## Notes

These items should each get their own TODO file when promoted to active work.
Priority and phase should be re-evaluated at that time.
