# Project Persistence

## Purpose

This document defines what VulVATAR saves, where it saves it, and how saved state is validated on reload.

Related documents:

- [application-design.md](/C:/lib/github/kjranyone/VulVATAR/docs/application-design.md)
- [data-model.md](/C:/lib/github/kjranyone/VulVATAR/docs/data-model.md)
- [editor-cloth-authoring.md](/C:/lib/github/kjranyone/VulVATAR/docs/editor-cloth-authoring.md)

## Persistence Classes

The application should separate saved state into four classes:

1. project state
2. imported avatar state
3. cloth overlay state
4. session-only state

### Project State

Project state is user intent for a working setup.

It should include:

- selected avatar source
- active avatar transform
- active cloth overlay reference
- tracking configuration
- rendering configuration
- output configuration
- last selected application mode if useful

### Imported Avatar State

Imported avatar state is derived from the source VRM and import pipeline.

It should not be edited by the GUI as user-authored content.

It includes:

- imported asset cache metadata
- source hash
- import-time derived IDs

### Cloth Overlay State

Cloth overlay state is authored content.

It should live outside the VRM file and be saved independently.

It includes:

- `ClothAsset`
- overlay metadata
- validation metadata against the target avatar

### Session-Only State

Session-only state should not be persisted by default.

It includes:

- current camera connection state
- current tracking frame
- output queue depth
- transient diagnostics
- unsaved viewport selection state

## File Model

Recommended first file model:

- project file: `*.vvtproj`
- cloth overlay file: `*.vvtcloth`

The project file should reference external source assets and overlay files rather than embedding everything.

## Recommended Project Shape

```text
project/
  avatar.vvtproj
  overlays/
    outfit_a.vvtcloth
  cache/
    imported-avatar.cache
```

The exact layout can change, but these boundaries should remain:

- project config is separate from imported caches
- cloth overlays are separate from the project root file
- source VRM remains external or explicitly copied by policy

## Save Ownership Rules

- the project owns references to avatars and overlays
- the overlay owns cloth authoring data
- the imported cache owns derived import artifacts
- runtime diagnostics own nothing persisted

## Project File Contents

The project file should contain:

- project version
- selected avatar source path or URI
- optional cached avatar identity
- avatar transform
- active overlay path or ID
- tracking settings
- rendering settings
- output settings

It should not contain:

- raw VRM binary data by default
- editor-only transient selection state
- runtime output queues
- raw tracking frames

## Overlay File Contents

The overlay file should contain:

- overlay version
- target avatar identity
- target avatar source hash
- stable references into imported avatar data
- cloth simulation mesh
- render-region bindings
- mesh mapping
- pins and constraints
- collision bindings
- solver parameters

## Versioning Rules

Every persisted file type should include:

- `format_version`
- `created_with`
- `last_saved_with`

Do not rely on implicit schema compatibility.

## Reload Validation

On project load, the application should validate:

- avatar source path exists
- avatar source hash matches if available
- overlay target avatar matches the selected avatar
- stable references still resolve
- output configuration is still supported on this machine

## Overlay Rebinding Policy

If an overlay does not match its target avatar cleanly:

- do not silently rebind by index
- mark the overlay invalid
- surface the failure in the GUI
- offer an explicit repair workflow later if implemented

## Save Semantics

The application should distinguish:

- saved
- dirty
- invalid

Examples:

- edited cloth parameters with no save means dirty
- unresolved primitive reference means invalid
- valid and written to disk means saved

## Autosave Policy

Recommended first policy:

- autosave project file optionally
- do not autosave cloth overlay without explicit user consent
- preserve recovery snapshots for crash recovery if implemented later

## Path Policy

The project system should decide this early:

- either allow absolute external source paths
- or copy/import assets into a managed project folder

For the first pass, external source references are simpler, but the project loader must clearly show missing-path failures.

## Serialization Guidance

- use explicit field names
- avoid positional encodings
- keep stable IDs visible in serialized data
- make unsupported extra fields ignorable where practical

## Failure Modes To Avoid

- mutating the base VRM to store cloth data
- storing runtime-only diagnostics in project files
- silently dropping unknown fields during migration
- rebinding overlays to the wrong mesh by import order
- treating cache files as authoritative user data
