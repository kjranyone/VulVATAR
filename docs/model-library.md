# Model Library

## Purpose

This document describes the avatar model library feature, which provides persistent cataloging of VRM 1.0 files for quick access, switching, and metadata management.

## Data Model

### `AvatarLibraryEntry`

One entry per known VRM file:

- `path`: source file path
- `name`: display name (defaults to file stem, editable)
- `source_hash`: SHA-256 hash of the VRM file at import time
- `last_loaded`: timestamp of last load (epoch seconds)
- `tags`: user-defined tags for filtering
- `notes`: free-form notes
- `favorite`: boolean flag for favorites sorting
- `mesh_count`, `material_count`, `spring_chain_count`, `collider_count`: cached asset stats
- `has_humanoid`: whether the VRM includes a humanoid rig

### `AvatarLibrary`

Holds a `Vec<AvatarLibraryEntry>` with operations:

- `add` / `remove` by path
- `search` by name, path, or tag substring
- `favorites` / `existing` filters
- `purge_missing`: remove entries whose source files no longer exist
- Sort by name, last loaded, or favorites-first

## Persistence

The library is stored as a JSON file (`avatar_library.vvtlib`) in the platform-specific application data directory:

- Windows: `%APPDATA%/VulVATAR/avatar_library.vvtlib`
- macOS: `~/Library/Application Support/VulVATAR/avatar_library.vvtlib`
- Linux: `$XDG_DATA_HOME/VulVATAR/avatar_library.vvtlib`

The file uses the standard `AvatarLibraryFile` wrapper with `format_version`, `created_with`, and `last_saved_with` fields.

The library is:

- Loaded at application startup (`GuiApp::new`)
- Saved automatically when avatars are loaded, entries are modified, or the application exits

## GUI

The library browser appears as a collapsible section ("Model Library") in the Preview mode inspector panel.

### Features

- **Search**: text filter by name, path, or tag
- **Sort**: by name, by recent usage, or favorites-first
- **Show missing**: toggle to display entries whose source files no longer exist
- **Entry list**: scrollable list with color-coded entries:
  - Blue highlight: currently loaded avatar
  - Red highlight: source file missing
  - Default: available avatar
- **Per-entry actions**:
  - Star/unstar favorite
  - Load button (loads avatar into the scene)
  - Remove button (removes from library only)
- **Detail editor** (for selected entry):
  - Editable name
  - Tag management (add/remove)
  - Notes field
  - Save button
- **Bulk actions**:
  - Purge Missing: remove all entries with missing source files
  - Add File: file picker to add a VRM to the library

## Integration Points

- `top_bar::load_avatar_from_path`: automatically registers loaded avatars in the library and saves metadata from the parsed asset
- `Application::avatar_library`: the library instance lives on the `Application` struct
- `GuiApp` holds search state, selected index, and tag buffer for the library UI
- Library persistence is handled via `persistence::load_avatar_library` / `save_avatar_library`

## Future Extensions

- Thumbnail generation and display
- Drag-and-drop import
- Folder watching for automatic library updates
- Import metadata from VRM extension data (title, author, version, thumbnail)
- Library categories or folders
- Export/import of library catalogs
