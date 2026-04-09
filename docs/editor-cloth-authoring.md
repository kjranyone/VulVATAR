# Editor Cloth Authoring

## Purpose

This document describes the cloth authoring workflow and GUI responsibilities for the in-app editor.

It is intentionally separate from `architecture.md`.

`architecture.md` defines runtime ownership, data boundaries, and frame update rules.

Application-level mode structure and non-cloth GUI design live in [application-design.md](/C:/lib/github/kjranyone/VulVATAR/docs/application-design.md).

Persistence and data contracts live in:

- [data-model.md](/C:/lib/github/kjranyone/VulVATAR/docs/data-model.md)
- [project-persistence.md](/C:/lib/github/kjranyone/VulVATAR/docs/project-persistence.md)

This document defines:

- how a user creates cloth data
- how the editor presents and edits that data
- how authored data is saved as a project-local overlay

## Design Goal

The editor should let a user start from a generic imported `VRM 1.0` avatar and author an optional `ClothAsset` overlay without mutating the base VRM.

The target is not a toy "secondary jiggle" tool.

The target is a path toward higher-fidelity garment authoring with:

- a dedicated cloth simulation mesh
- mapping from simulation results back to rendered geometry
- explicit pin, constraint, and collision proxy editing

## Non-Goals

- editing the original VRM file in place
- requiring cloth data for every avatar
- making the editor a runtime dependency for playback
- collapsing cloth authoring into attachment-point-only motion

## Authoring Model

The editor works on:

- one imported base avatar
- zero or one active cloth overlay in the first pass

The saved result is a project-local overlay asset that can later be attached to an `AvatarInstance`.

The overlay should contain:

- stable references into the imported avatar mesh and skeleton
- cloth simulation mesh data
- render-region binding data
- simulation-to-render mapping data
- pin definitions
- constraint definitions
- collision proxy bindings
- solver tuning parameters

## GUI Scope

The first editor version should focus on one narrow garment region at a time.

Recommended first targets:

- skirt hem
- coat flap
- cape edge
- ribbon sheet

Avoid full-garment workflows in the first pass.

## Screen Structure

The editor should have four primary areas:

1. viewport
2. asset and overlay browser
3. cloth authoring inspector
4. simulation preview controls

### Viewport

The viewport is the main authoring surface.

It should support:

- avatar orbit, pan, zoom
- wireframe and shaded display modes
- simulation mesh visibility toggle
- render mesh visibility toggle
- collision proxy visibility toggle
- pin visualization
- constraint visualization
- region selection feedback

### Asset and Overlay Browser

This area should show:

- imported base avatar
- existing cloth overlays for that avatar
- current active overlay
- save status

Primary actions:

- create overlay
- duplicate overlay
- rename overlay
- delete overlay
- attach overlay to preview instance

### Cloth Authoring Inspector

This area should edit structured cloth data rather than ad hoc scene state.

Recommended sections:

- selected garment region
- simulation mesh generation or edit settings
- mapping settings
- pin groups
- constraint groups
- collision proxy bindings
- solver parameters
- LoD settings if enabled

### Simulation Preview Controls

This area should control preview execution and debugging.

Recommended controls:

- play
- pause
- reset
- step one simulation tick
- enable or disable cloth
- enable or disable collisions
- gravity scale
- wind presets if added later

## Authoring Workflow

The intended workflow is:

1. import or open a base avatar
2. create a new cloth overlay
3. select a garment region on the render mesh
4. generate or edit a dedicated simulation mesh for that region
5. define mapping from simulation mesh to rendered garment geometry
6. define pins and attachment behavior
7. assign body collision proxies
8. tune solver parameters in preview
9. save the overlay
10. attach or detach the overlay at runtime for validation

## Editing Operations

The first authoring version should support these operations explicitly:

- select mesh region
- grow or shrink region selection
- create pin set from selected vertices
- bind pins to skeleton references
- assign or remove collision proxies
- regenerate simulation mesh
- adjust constraint stiffness ranges
- reset overlay to last saved state

If an operation cannot be represented in `ClothAsset`, it should not exist only as hidden editor state.

## Persistence Rules

The editor must save cloth as project-local overlay data, not as a mutated VRM.

The persistence contract should guarantee:

- the base imported avatar stays reusable
- the overlay survives editor restart
- the overlay can be reattached to the same imported avatar deterministically
- validation errors are surfaced when stable references no longer resolve

If the base avatar is reimported or changes shape, the editor should detect overlay invalidation rather than silently rebinding by index.

## Preview Rules

Preview should exercise the same broad runtime path as gameplay playback whenever practical.

That means:

- preview uses the runtime cloth solver
- preview uses the same body collision proxy definitions
- preview can run with cloth enabled or disabled
- preview can show final rendered deformation and simulation mesh at the same time

The editor may add debug rendering, but it should not maintain a separate fake cloth execution path.

## Validation

The editor should block or warn on:

- overlays with unresolved stable references
- pinned sets with no valid skeleton binding
- collision proxies that do not map to the avatar
- mappings that leave the render region without a valid driver
- parameter values outside supported solver ranges

## Runtime Boundary

The editor produces `ClothAsset`.

The runtime consumes `ClothAsset`.

The runtime does not depend on editor UI state, viewport state, or unsaved tool selections.

## Future Extensions

Later versions can add:

- multiple cloth overlays per avatar
- cloth LoD authoring
- wind authoring tools
- overlay version migration
- partial rebinding tools after avatar reimport
- offline bake or cache generation for testing
