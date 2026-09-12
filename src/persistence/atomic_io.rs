//! Durable JSON write primitives shared by every persistence store:
//! `atomic_write` (backup-copy -> temp-file -> fsync -> rename — the
//! hardening added after the 2026-06-12 Arc-freeze NUL corruption) and
//! the primary/backup fallback readers.

use log::warn;
use std::path::Path;

pub(super) fn backup_path_for(path: &std::path::Path) -> std::path::PathBuf {
    path.with_extension(
        path.extension()
            .map(|e| {
                let mut s = e.to_string_lossy().into_owned();
                s.push_str(".bak");
                s
            })
            .unwrap_or_else(|| "bak".to_string()),
    )
}

pub(super) fn load_with_fallback<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> Result<T, String> {
    let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&data).map_err(|e| e.to_string())
}

pub(super) fn load_or_backup<T: serde::de::DeserializeOwned>(primary: &std::path::Path) -> Result<T, String> {
    let backup = backup_path_for(primary);
    match load_with_fallback::<T>(primary) {
        Ok(v) => Ok(v),
        Err(primary_err) => {
            warn!(
                "persistence: primary file failed: {}, trying backup",
                primary_err
            );
            load_with_fallback::<T>(&backup)
                .map_err(|backup_err| format!("primary: {}; backup: {}", primary_err, backup_err))
        }
    }
}

pub fn atomic_write(path: &Path, data: &str) -> Result<(), String> {
    if path.exists() {
        // Backup-copy is best-effort: if it fails (read-only target dir,
        // disk full, AV-quarantined source), proceed with the write
        // anyway — refusing to write because the backup couldn't be
        // taken would itself be a bigger UX failure. But log the
        // failure so a user diagnosing "where did my backup go?" can
        // trace it instead of finding a missing .bak silently.
        if let Err(e) = std::fs::copy(path, backup_path_for(path)) {
            warn!(
                "atomic_write: backup of {} failed (continuing without backup): {}",
                path.display(),
                e
            );
        }
    }

    let parent = path.parent().unwrap_or(Path::new("."));
    let temp_name = format!(
        ".{}.tmp",
        path.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".to_string())
    );
    let temp_path = parent.join(&temp_name);

    // Write + fsync the temp file BEFORE the rename. A plain
    // `fs::write` + rename is atomic against a process crash, but NOT
    // against power loss / a hard GPU-driver freeze: the rename can
    // commit in the directory while the temp file's data blocks are
    // still in the OS write cache, so after the freeze the file exists
    // at full length but reads back as all-zero. That is exactly how
    // the 2026-06-12 Arc freeze NUL-corrupted `last_session.vvtproj`
    // (and its `.bak`). `sync_all` forces the bytes to stable storage
    // first, so the rename only ever publishes durable data. Cost is
    // one fsync per write; every caller is throttled (autosave 4/s,
    // recovery on a timer) so the added latency is immaterial.
    {
        use std::io::Write as _;
        let mut f = std::fs::File::create(&temp_path).map_err(|e| {
            format!(
                "atomic_write: failed to create temp file {}: {}",
                temp_path.display(),
                e
            )
        })?;
        f.write_all(data.as_bytes()).map_err(|e| {
            let _ = std::fs::remove_file(&temp_path);
            format!(
                "atomic_write: failed to write temp file {}: {}",
                temp_path.display(),
                e
            )
        })?;
        f.sync_all().map_err(|e| {
            let _ = std::fs::remove_file(&temp_path);
            format!(
                "atomic_write: failed to fsync temp file {}: {}",
                temp_path.display(),
                e
            )
        })?;
    }

    if cfg!(windows) {
        if std::fs::rename(&temp_path, path).is_err() {
            std::fs::copy(&temp_path, path).map_err(|e| {
                let _ = std::fs::remove_file(&temp_path);
                format!(
                    "atomic_write: failed to copy {} -> {}: {}",
                    temp_path.display(),
                    path.display(),
                    e
                )
            })?;
            let _ = std::fs::remove_file(&temp_path);
        }
    } else {
        std::fs::rename(&temp_path, path).map_err(|e| {
            let _ = std::fs::remove_file(&temp_path);
            format!(
                "atomic_write: failed to rename {} -> {}: {}",
                temp_path.display(),
                path.display(),
                e
            )
        })?;
    }

    Ok(())
}
