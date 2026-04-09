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

Detailed cloth authoring workflow lives in [editor-cloth-authoring.md](/C:/lib/github/kjranyone/VulVATAR/docs/editor-cloth-authoring.md).

Related detailed contracts:

- [project-persistence.md](/C:/lib/github/kjranyone/VulVATAR/docs/project-persistence.md)
- [tracking-retargeting.md](/C:/lib/github/kjranyone/VulVATAR/docs/tracking-retargeting.md)
- [output-interop.md](/C:/lib/github/kjranyone/VulVATAR/docs/output-interop.md)
- [rendering-materials.md](/C:/lib/github/kjranyone/VulVATAR/docs/rendering-materials.md)

## Product Goal

The application should let a user:

- load a generic `VRM 1.0` avatar
- preview and position that avatar in a live scene
- drive the avatar from webcam-based tracking
- tune rendering and output behavior
- optionally attach authored cloth overlays
- route the result to OBS or other output sinks

The first version should be usable as a live avatar application, not only as a renderer testbed.

## Primary User Scenarios

The application should support these core scenarios:

1. load an avatar and preview it locally
2. connect and tune webcam tracking
3. adjust avatar placement and visual presentation
4. configure output for OBS ingestion
5. open cloth authoring for a selected avatar and save an overlay
6. switch back to live preview and use the authored overlay

## Application Modes

The application should expose explicit modes rather than mixing all tools into one overloaded screen.

Recommended first modes:

1. `Preview`
2. `Tracking Setup`
3. `Rendering`
4. `Output`
5. `Cloth Authoring`

### `Preview`

Purpose:

- load an avatar
- inspect the current scene
- position the avatar
- toggle runtime features on or off

### `Tracking Setup`

Purpose:

- choose webcam input
- inspect tracking confidence
- tune retargeting behavior
- validate latency and stability

### `Rendering`

Purpose:

- adjust material and shading controls
- inspect lighting and background composition
- validate alpha and scene framing

### `Output`

Purpose:

- configure frame sink selection
- inspect output state
- validate OBS-facing behavior

### `Cloth Authoring`

Purpose:

- create and edit `ClothAsset` overlays
- preview cloth simulation
- save overlay data

Detailed flow is defined in [editor-cloth-authoring.md](/C:/lib/github/kjranyone/VulVATAR/docs/editor-cloth-authoring.md).

## Top-Level Layout

The application should have a stable shell with:

- top bar
- left mode navigation
- central viewport
- right inspector
- bottom status and diagnostics strip

### Top Bar

The top bar should provide global actions:

- open avatar
- recent avatars
- save project
- save project as
- open cloth overlay
- save cloth overlay
- runtime play or pause if needed later

### Left Mode Navigation

This should switch between:

- Preview
- Tracking Setup
- Rendering
- Output
- Cloth Authoring

Switching modes should preserve loaded assets and current session state.

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

## Preview Mode

This is the default mode after loading an avatar.

Recommended inspector sections:

- avatar asset
- transform
- cloth attachment
- scene background
- runtime toggles

### Avatar Asset

Controls:

- loaded avatar identity
- reload avatar
- detach avatar
- active cloth overlay

### Transform

Controls:

- position
- rotation
- scale
- reset transform
- ground alignment helper if added later

### Cloth Attachment

Controls:

- attach overlay
- detach overlay
- enable cloth
- disable cloth

### Scene Background

Controls:

- transparent background
- solid color background
- image background if added later

### Runtime Toggles

Controls:

- tracking enabled
- spring enabled
- cloth enabled
- collision debug
- skeleton debug

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

Its detailed structure is defined in [editor-cloth-authoring.md](/C:/lib/github/kjranyone/VulVATAR/docs/editor-cloth-authoring.md).

At the application level, the important rule is:

- entering cloth authoring should keep the current avatar and session context
- leaving cloth authoring should return to preview with the saved overlay available

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
4. choose webcam
5. enable tracking
6. adjust avatar transform
7. pick output sink
8. validate OBS routing

Cloth authoring should not block first-run success.

## Failure Cases The GUI Must Surface

- avatar load failure
- unsupported or invalid VRM data
- webcam unavailable
- low-confidence tracking
- output sink unavailable
- cloth overlay validation failure
- output fallback from GPU handoff to slower path

These should appear in visible diagnostics, not only in logs.

## Future Extensions

Later versions can add:

- multiple avatars
- scene presets
- expression control panels
- hotkey support
- preset export and import
- profile switching for different streaming setups
