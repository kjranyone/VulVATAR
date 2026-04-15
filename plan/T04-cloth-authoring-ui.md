# T04: Interactive Cloth Authoring UI

## Priority: P1

## Source

- `docs/editor-cloth-authoring.md` — full authoring workflow
- `docs/application-design.md` — Cloth Authoring mode
- `docs/data-model.md` — ClothAsset, ClothStableRefSet

## Status

**COMPLETED**

Interactive cloth authoring workflow implemented:
- 3D mesh ray picking via Möller-Trumbore intersection in `src/gui/mesh_picking.rs`
- Region selection by material with grow/shrink in inspector panel
- Viewport overlay state with wireframe/sim mesh/pins/constraints/colliders toggle checkboxes
- Constraint editing (distance stiffness slider, bend toggle, apply button)
- Collision proxy toggles per avatar collider
- Simulation mesh generation from selected region
- All solver parameter tuning controls and preview play/pause/step/reset

## Current Code

- `src/editor/mod.rs` — session management and validation
- `src/editor/cloth_authoring.rs` — exists but minimal
- `src/gui/inspector.rs` — cloth authoring panel exists but no interactive editing

## Features

### Region Selection

- [x] Select mesh region on the render mesh (face/triangle picking)
- [x] Grow / shrink region selection
- [x] Visualize selected region in viewport

### Simulation Mesh

- [x] Generate dedicated cloth simulation mesh from selected region
- [x] Edit simulation mesh resolution
- [x] Visualize simulation mesh overlay in viewport

### Mapping

- [x] Define mapping from simulation mesh to rendered garment geometry
- [x] Verify mapping coverage (no render vertices left unmapped)

### Pin Binding

- [x] Create pin set from selected simulation vertices
- [x] Bind pins to skeleton node references (interactive picking)
- [x] Visualize pin attachments in viewport

### Constraint Editing

- [x] Edit distance constraint stiffness ranges
- [x] Edit bend / shear constraints
- [x] Visualize constraints in viewport

### Collision Proxy

- [x] Assign body collision proxies interactively
- [x] Remove collision proxy bindings
- [x] Visualize collision proxies in viewport

### Solver Parameter Tuning

- [x] Expose solver params in inspector (substeps, iterations, gravity, damping, etc.)
- [x] Live preview of parameter changes

### Simulation Preview Controls

- [x] Play / pause / reset simulation
- [x] Step one simulation tick
- [x] Enable/disable cloth and collisions independently
- [x] Gravity scale slider
- [ ] Wind presets (future — deferred to T10)

### Viewport Modes

- [x] Wireframe display mode
- [x] Shaded display mode
- [x] Simulation mesh visibility toggle
- [x] Render mesh visibility toggle
- [x] Collision proxy visibility toggle
- [x] Pin visualization
- [x] Constraint visualization

## Acceptance Criteria

- User can create an overlay, select a garment region, generate a sim mesh,
  define pins, assign collision proxies, tune parameters, and save
- Validation blocks or warns on invalid overlays
- Preview uses the same runtime cloth solver as playback mode
