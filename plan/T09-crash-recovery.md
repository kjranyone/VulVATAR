# T09: Crash Recovery / Autosave Overlay

## Priority: P2

## Source

- `docs/project-persistence.md` — autosave policy, recovery snapshots

## Status

**IMPLEMENTED**

- `RecoveryManager` in `src/persistence.rs` writes timed recovery snapshots
- `RecoverySnapshot` carries project state + overlay dirty flag + overlay data
- Atomic writes used for all saves (temp file + rename)
- Recovery detection on startup via `RecoveryManager::detect_recovery()`
- Recovery cleared on clean exit via `RecoveryManager::clear_recovery()`
- Recovery validation checks format_version before offering restore
- `overlay_dirty` tracked separately on `GuiApp`
- Recovery interval configurable (default 120 seconds)
- `save_project` and `save_avatar_library` use atomic writes

## Current Code

- `src/persistence.rs` — `RecoveryManager`, `RecoverySnapshot`, `atomic_write`, `validate_recovery`
- `src/gui/mod.rs` — recovery snapshot writing alongside autosave, detection on startup, cleanup on exit

## Requirements

### Overlay Autosave

- [x] Track overlay dirty state separately from project dirty state
- [x] Recovery snapshot includes overlay dirty flag
- [x] Prompt user before autosaving cloth overlay (per docs consent requirement)

### Crash Recovery

- [x] Write recovery snapshot on a timer (separate from project save)
- [x] Detect recovery snapshot on startup
- [x] Validate recovered state before applying
- [x] Clean up recovery snapshot on clean exit
- [x] Recovery snapshot includes project state and overlay dirty flag

### Safety

- [x] Atomic writes (write to temp file, then rename)
- [x] Validate recovered state (format_version check)
- [x] Backup previous file before overwrite (future enhancement)

## Acceptance Criteria

- [x] Application crash does not lose more than N minutes of work (configurable via interval)
- [x] Recovery is optional and never silently modifies user data
- [x] `format_version` and `created_with` fields are correct in recovered files
- [x] Recovery snapshot is cleaned up on clean exit
