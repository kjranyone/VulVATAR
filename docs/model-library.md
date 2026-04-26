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

## Folder Watching

Every directory added via *Watched folders* is registered with a
`notify`-backed watcher in the avatar library inspector; new `.vrm`
files appearing under a watched directory get auto-imported with a
notification toast. The watcher's debounce is fixed at 100 ms and
only handles flat directories (recursion is off).

### Limitations

`notify`'s recommended backend is local-filesystem-only by design:

- **UNC / SMB shares** — server-side change events typically don't
  reach the client on Windows; new files can sit on the share until
  the user manually pokes the library.
- **OneDrive / Google Drive / Dropbox** — cloud-sync clients land
  files in stages (placeholder → real bytes) and the events
  collapse under the debounce as `Modified` rather than `Created`,
  so the VRM-Created filter rejects them.
- **Symlinks / NTFS junctions** — propagation across the link is
  platform-dependent.

For these cases, the inspector exposes a manual **Refresh** button
that walks every watched directory, calls the same import path the
notify-driven poll uses (`GuiApp::import_vrm_into_library_if_missing`),
and reports the count of newly-imported files. Operators on
cloud-synced setups should treat Refresh as routine, not as a
recovery step.

## Avatar Load Cache

The library is independent from the on-disk parse cache at
`%APPDATA%\VulVATAR\cache\*.vvtcache` — see `src/asset/cache.rs`.
On `Application::new`, `evict_to_count(DEFAULT_MAX_CACHE_ENTRIES = 20)`
drops the oldest-mtime entries past the cap. When eviction fires it
logs `evicted N entries — freed X.X MB (cache now Y.Y MB, cap M)` so
the cap's real meaning ("how much disk does 20 entries cost?") is
visible per the operator's own avatars rather than guessed at from a
docstring estimate.

## Future Extensions

- Drag-and-drop import
- Import metadata from VRM extension data (title, author, version,
  thumbnail) — partially landed; tags and category remain manual.
- Library categories or folders
- Export/import of library catalogs
