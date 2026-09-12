//! Crash-recovery snapshot: a periodic dump of the in-memory project
//! (plus a dirty cloth overlay) under the app-data recovery dir, so a
//! crash loses at most one interval of edits.

use log::{error, info};
use serde::{Deserialize, Serialize};

use super::{
    app_data_dir, app_tag, atomic_write, ClothOverlayFile, ProjectFile, ProjectState,
    OVERLAY_FORMAT_VERSION, PROJECT_FORMAT_VERSION, RECOVERY_FORMAT_VERSION,
};

#[derive(Serialize, Deserialize, Debug)]
pub struct RecoverySnapshot {
    pub format_version: u32,
    pub created_with: String,
    pub timestamp_secs: u64,
    #[serde(default)]
    pub project: Option<ProjectFile>,
    pub overlay_dirty: bool,
    #[serde(default)]
    pub overlay: Option<ClothOverlayFile>,
}

fn recovery_dir() -> std::path::PathBuf {
    let mut dir = app_data_dir();
    dir.push("recovery");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn recovery_snapshot_path() -> std::path::PathBuf {
    recovery_dir().join("recovery.vvtsnap")
}

pub struct RecoveryManager {
    interval_secs: u64,
    /// Monotonic instant of the last successful snapshot write. The
    /// interval gate used to compare wall-clock unix seconds, so an
    /// NTP step backwards silently paused the crash-recovery net (and
    /// a step forward fired it early). Wall time is still *recorded*
    /// in the snapshot (`timestamp_secs`) — that's display data, not
    /// scheduling data.
    last_snapshot: Option<std::time::Instant>,
    enabled: bool,
}

impl RecoveryManager {
    pub fn new(interval_secs: u64) -> Self {
        Self {
            interval_secs,
            last_snapshot: None,
            enabled: true,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn set_interval(&mut self, secs: u64) {
        self.interval_secs = secs;
    }

    pub fn should_snapshot(&self, now: std::time::Instant) -> bool {
        if !self.enabled {
            return false;
        }
        match self.last_snapshot {
            None => true,
            Some(last) => now.duration_since(last).as_secs() >= self.interval_secs,
        }
    }

    pub fn write_snapshot(
        &mut self,
        project_state: Option<&ProjectState>,
        overlay_dirty: bool,
        overlay: Option<&ClothOverlayFile>,
    ) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let snapshot = RecoverySnapshot {
            format_version: RECOVERY_FORMAT_VERSION,
            created_with: app_tag(),
            timestamp_secs: now,
            project: project_state.map(ProjectFile::from_state),
            overlay_dirty,
            overlay: overlay.cloned(),
        };

        let json = serde_json::to_string_pretty(&snapshot).map_err(|e| e.to_string())?;
        let path = recovery_snapshot_path();
        atomic_write(&path, &json)?;

        self.last_snapshot = Some(std::time::Instant::now());
        Ok(())
    }

    pub fn detect_recovery() -> Option<RecoverySnapshot> {
        let path = recovery_snapshot_path();
        if !path.exists() {
            return None;
        }
        match std::fs::read_to_string(&path) {
            Ok(data) => match serde_json::from_str::<RecoverySnapshot>(&data) {
                Ok(snapshot) => {
                    info!(
                        "persistence: detected recovery snapshot from {} (has_project={}, has_overlay={})",
                        snapshot.timestamp_secs,
                        snapshot.project.is_some(),
                        snapshot.overlay.is_some(),
                    );
                    Some(snapshot)
                }
                Err(e) => {
                    error!("persistence: corrupt recovery snapshot, removing: {}", e);
                    let _ = std::fs::remove_file(&path);
                    None
                }
            },
            Err(e) => {
                error!("persistence: cannot read recovery snapshot: {}", e);
                None
            }
        }
    }

    pub fn clear_recovery() {
        let path = recovery_snapshot_path();
        if path.exists() {
            let _ = std::fs::remove_file(&path);
            info!("persistence: cleared recovery snapshot");
        }
    }
}

pub fn validate_recovery(snapshot: &RecoverySnapshot) -> Result<(), String> {
    if snapshot.format_version != RECOVERY_FORMAT_VERSION {
        return Err(format!(
            "unsupported recovery format_version {} (expected {})",
            snapshot.format_version, RECOVERY_FORMAT_VERSION
        ));
    }
    if let Some(ref project) = snapshot.project {
        if project.format_version != PROJECT_FORMAT_VERSION {
            return Err(format!(
                "unsupported project format_version {} in recovery (expected {})",
                project.format_version, PROJECT_FORMAT_VERSION
            ));
        }
    }
    if let Some(ref overlay) = snapshot.overlay {
        if overlay.format_version != OVERLAY_FORMAT_VERSION {
            return Err(format!(
                "unsupported overlay format_version {} in recovery (expected {})",
                overlay.format_version, OVERLAY_FORMAT_VERSION
            ));
        }
    }
    Ok(())
}
