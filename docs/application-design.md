# Application Design

## Purpose

This document describes the user-facing application structure for VulVATAR.

It focuses on:

- primary application modes
- major user workflows
- top-level screen and panel layout
- feature ownership at the GUI level

It does not replace `architecture.md`.

`architecture.md` defines runtime boundaries, data ownership, and update order.

Detailed cloth authoring workflow lives in [editor-cloth-authoring.md](editor-cloth-authoring.md).

Related detailed contracts:

- [project-persistence.md](project-persistence.md)
- [tracking-v2-design.md](tracking-v2-design.md)
- [output-interop.md](output-interop.md)
- [vulkano-renderer-design.md](vulkano-renderer-design.md)

## Product Goal

The application should let a user:

- load a generic `VRM 1.0` avatar
- preview and position that avatar in a live scene
- drive the avatar from RealSense D435 depth-camera tracking
- tune rendering and output behavior
- optionally attach authored cloth overlays
- route the result to OBS or other output sinks

The first version should be usable as a live avatar application, not only as a renderer testbed.

## Primary User Scenarios

The application should support these core scenarios:

1. load an avatar and preview it locally
2. connect and tune RealSense D435 depth tracking
3. adjust avatar placement and visual presentation
4. configure output for OBS ingestion
5. open cloth authoring for a selected avatar and save an overlay
6. switch back to live preview and use the authored overlay

## Application Modes

The application exposes 6 explicit modes on the navigation rail (`AppMode::ALL`):

1. `Avatar`
2. `Tracking Setup`
3. `Rendering` (Scene)
4. `Output`
5. `Cloth Authoring`
6. `Settings`

(`Preview` was an earlier prototype mode that has been retired; its features were redistributed to `Avatar`, `Rendering`, and `Cloth Authoring`.)

### `Avatar`

Purpose:

- inspect loaded avatar identity and statistics
- browse and manage the Model Library
- test and inspect facial expressions and blendshapes
- manage avatar attachment and reload

### `Tracking Setup`

Purpose:

- start or stop the RealSense D435 capture
- calibrate pose reference and anchors
- configure driven body parts (head, face, lower body, root translation, hands)
- inspect tracking latency, confidence, and solver diagnostics
- adjust smoothing parameters and safety fallbacks

### `Rendering` (Scene)

Purpose:

- adjust avatar world transform (position, rotation, scale)
- configure material modes (Unlit, SimpleLit, ToonLike) and toon ramp parameters
- tune scene lighting, ambient floor, and background
- configure post-processing effects (Bloom downsample/upsample/composite)

### `Output`

Purpose:

- select frame sink (Virtual Camera, Shared Memory, Shared Texture)
- configure resolution, target frame rate, color space, alpha mode, MSAA
- monitor queue latency, dropped frames, and GPU handoff lease tokens

### `Cloth Authoring`

Purpose:

- create and edit `ClothAsset` overlay slots
- select garment regions with viewport ray picking or material filters
- generate simulation meshes and configure pin sets / collision proxies
- tune XPBD solver parameters in interactive preview

### `Settings`

Purpose:

- choose UI language (English, Japanese, Simplified Chinese, Korean)
- adjust viewport navigation sensitivities (orbit, pan, zoom)
- toggle session diagnostics
- inspect keyboard shortcuts

## Top-Level Layout

The application has a stable shell with:

- top bar
- left mode navigation
- central viewport
- left inspector (docked beside mode navigation)
- bottom status strip

### Top Bar

The top bar provides global session actions:

- inspector toggle
- project title & dirty status
- avatar picker / quick switch
- camera start / stop
- pose calibration launcher
- runtime pause toggle
- File dropdown menu (New / Open / Save project, import avatar, load/save overlay)

### Left Mode Navigation

Switches between the 6 modes:

- Avatar (F1)
- Tracking Setup (F2)
- Rendering / Scene (F3)
- Output (F4)
- Cloth Authoring (F6)
- Settings (F7)
(F5 folds into Rendering for legacy muscle-memory compatibility)

Switching modes preserves loaded assets and current session state.

### Central Viewport

The viewport is the main shared visual surface across most modes.

It should support:

- orbit, pan, zoom
- reset camera
- framed view of the active avatar
- transparent or composited background preview
- debug overlays when enabled

Different modes may add overlays, but the core camera interaction should stay consistent.

### Right Inspector

The inspector should change by mode.

It should hold structured controls rather than floating dialogs whenever possible.

### Bottom Status Strip

The status area should show:

- current avatar name
- tracking source state
- output sink state
- frame timing
- warnings and validation messages

## Avatar Mode

This mode manages the active avatar asset, model catalog, and facial blendshapes.

Recommended inspector sections:

- avatar asset info
- model library
- expression testing
- camera transform / autoframe

### Avatar Asset Info

Controls:

- loaded avatar identity, source format (.vrm, .fbx), hash, and mesh/material statistics
- reload avatar
- detach avatar

### Model Library

Controls:

- persistent catalog of imported VRM/FBX models (`avatar_library.vvtlib`)
- search by name, path, or tags
- sorting (name, recent, favorites)
- add file / purge missing files
- watched folders for auto-import

### Expression Testing

Controls:

- test sliders for facial blendshape weights (vowels, eyes, brows, emotions)
- reset expressions

### Camera Transform / Autoframe

Controls:

- auto-frame camera to avatar bounding box
- orbit, pan, zoom controls and reset view

## Tracking Setup Mode

This mode should make camera and retargeting setup explicit instead of burying it in a small preferences panel.

Recommended inspector sections:

- input device
- capture format
- inference status
- retargeting
- smoothing and confidence

### Input Device

Controls:

- camera selection
- camera start or stop
- resolution
- frame rate

### Capture Format

Controls:

- pixel format if exposed
- mirror preview toggle
- crop or framing helpers if added later

### Inference Status

Controls or indicators:

- current tracking state
- last frame timestamp
- dropped frame warnings
- CPU or GPU inference backend if selectable later

### Retargeting

Controls:

- head mapping
- torso mapping
- arm mapping
- hand tracking enable if available
- facial tracking enable if available

### Smoothing and Confidence

Controls:

- smoothing strength
- confidence threshold
- missing landmark fallback behavior

## Rendering Mode

This mode owns visual presentation, not runtime tracking behavior.

Recommended inspector sections:

- material mode
- toon controls
- outline controls
- lighting
- composition

### Material Mode

Controls:

- unlit
- simple lit
- toon-like

### Toon Controls

Controls:

- ramp threshold
- shadow softness if supported
- specular or highlight behavior if supported

### Outline Controls

Controls:

- outline enable
- width
- color

### Lighting

Controls:

- key light direction
- intensity
- ambient term

### Composition

Controls:

- camera field of view
- framing presets
- alpha preview

## Output Mode

This mode should make OBS and output sink behavior visible and debuggable.

Recommended inspector sections:

- sink selection
- frame format
- synchronization
- diagnostics

### Sink Selection

Controls:

- virtual camera
- shared texture
- shared memory
- image sequence for debugging

### Frame Format

Controls:

- output resolution
- output frame rate
- RGB or RGBA
- alpha mode
- color space

### Synchronization

Controls or indicators:

- current handoff path
- GPU shared-frame enabled state
- fallback path in use
- backpressure policy

### Diagnostics

Indicators:

- output connected or disconnected
- queue depth
- dropped frame count
- last handoff timestamp

## Cloth Authoring Mode

This mode is a specialized editor entry point, not a separate application.

Its detailed structure is defined in [editor-cloth-authoring.md](editor-cloth-authoring.md).

At the application level, the important rule is:

- entering cloth authoring should keep the current avatar and session context
- leaving cloth authoring should return to the active avatar with the saved overlay available

## Settings Mode

This mode manages application-level preferences persisted to `%APPDATA%\VulVATAR\settings.json`.

Recommended inspector sections:

- language selection (English, Japanese, Simplified Chinese, Korean)
- viewport sensitivities (orbit, pan, zoom with logarithmic slider)
- session-only diagnostics toggle (status-bar debug telemetry)
- keyboard shortcuts (read-only list generated from live bindings)

## Project Model

The GUI should expose a project concept early, even if the first version is minimal.

A project should own:

- selected avatar source
- active avatar transform
- tracking configuration
- rendering configuration
- output configuration
- optional cloth overlay references

This avoids scattering state across unrelated config files and unsaved runtime panels.

## State Boundaries

The GUI should separate:

- persistent project state
- persistent avatar overlay state
- transient session state
- pure runtime diagnostics

Examples:

- avatar transform belongs to project state
- cloth overlay data belongs to overlay state
- live camera connection state is session state
- current frame time is runtime diagnostic state

## First-Run Workflow

The first-run experience should be short and practical:

1. open application
2. load VRM avatar
3. see avatar in preview
4. choose the RealSense D435 depth device
5. enable tracking
6. adjust avatar transform
7. pick output sink
8. validate OBS routing

Cloth authoring should not block first-run success.

## Failure Cases The GUI Must Surface

- avatar load failure
- unsupported or invalid VRM data
- RealSense D435 depth camera unavailable
- low-confidence tracking
- output sink unavailable
- cloth overlay validation failure
- output fallback from GPU handoff to slower path

These should appear in visible diagnostics, not only in logs.

## Hotkeys

Default bindings live in `src/gui/hotkey.rs::HotkeyMap::set_defaults`
— that's the source of truth. Tooltips on the relevant buttons surface
the chord (`mode_nav.rs`, `top_bar.rs`, `inspector/library.rs`).

| Chord          | Action                                                              |
|----------------|---------------------------------------------------------------------|
| `Space`        | Toggle pause                                                        |
| `Ctrl+T`       | Toggle tracking enabled                                             |
| `Ctrl+Shift+C` | Toggle cloth simulation                                             |
| `Ctrl+Shift+R` | Reset pose                                                          |
| `Home`         | Reset camera                                                        |
| `Ctrl+S`       | Save project                                                        |
| `Ctrl+O`       | Load avatar                                                         |
| `F1`           | Switch to Avatar mode                                               |
| `F2`           | Switch to Tracking Setup mode                                       |
| `F3`           | Switch to Rendering (Scene) mode                                    |
| `F4`           | Switch to Output mode                                               |
| `F5`           | Switch to Rendering (Scene) mode (legacy Preview key)                |
| `F6`           | Switch to Cloth Authoring mode                                      |
| `F7`           | Switch to Settings mode                                             |

Hotkeys are suppressed while a text input owns keyboard focus —
typing "Save the day" in a rename field doesn't trigger save.

## Future Extensions

Later versions can add:

- multiple avatars
- scene presets
- expression control panels
- user-rebindable hotkeys (defaults exist; rebinding UI does not)
- preset export and import
- profile switching for different streaming setups
