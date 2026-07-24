//! Background avatar loading: spawns `VrmAssetLoader::load_with_progress` on
//! a worker thread and reports progress back to the GUI via an mpsc channel.
//!
//! The GUI keeps an `Option<AvatarLoadJob>` and polls it each frame in
//! `GuiApp::update`. Heavy CPU work (file I/O, glTF parse, texture decode)
//! runs off the UI thread; finalisation (instance construction, physics
//! attachment, library bookkeeping) happens back on the UI thread once the
//! asset arrives.

use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crate::asset::{
    vrm::{LoadStage, VrmAssetLoader},
    AvatarAsset,
};
use crate::persistence::{ProjectLoadWarnings, ProjectState};
use crate::t;

use super::{top_bar, GuiApp};

/// Optional follow-up work to perform on the UI thread once the avatar
/// finishes loading. `Open Project` uses this to apply the project state
/// after the referenced avatar has been brought into the scene (the
/// state must apply *after* `finalize_avatar_load`, which auto-frames
/// the camera — applying first would let the auto-frame clobber the
/// restored orbit). The startup last-session restore reuses the same
/// mechanism with `project_path: None` so the title bar doesn't show a
/// phantom file the user never opened.
pub enum AfterLoad {
    None,
    ApplyProject {
        project_state: Box<ProjectState>,
        /// `Some` for an explicitly opened `.vvtproj` (sets
        /// `project_status.project_path` + "opened" toast); `None` for
        /// the implicit last-session restore (stays anonymous, no toast).
        project_path: Option<PathBuf>,
        warnings: ProjectLoadWarnings,
        /// `true` when `project_state` came from the project's
        /// `.unsaved` sidecar rather than the file itself — the apply
        /// marks the explicit file stale (title-bar dot) and toasts
        /// that unsaved changes were restored.
        restore_unsaved: bool,
    },
}

pub enum LoadMessage {
    Progress(LoadStage),
    Done(Arc<AvatarAsset>),
    Error(String),
}

/// Terminal outcome of a load job — what `poll` returns once the worker
/// finishes. Progress messages are absorbed internally by `poll`.
pub enum LoadOutcome {
    Done(Arc<AvatarAsset>),
    Error(String),
}

pub struct AvatarLoadJob {
    pub path: PathBuf,
    pub current_stage: LoadStage,
    pub receiver: mpsc::Receiver<LoadMessage>,
    pub after_load: AfterLoad,
    /// Held so the worker is not detached; on completion the GUI joins it.
    pub worker: Option<JoinHandle<()>>,
}

impl AvatarLoadJob {
    /// Spawn a worker thread that loads `path` and streams progress through
    /// the returned job's receiver.
    pub fn spawn(path: PathBuf, after_load: AfterLoad) -> Self {
        let (tx, rx) = mpsc::channel();
        let path_for_worker = path.clone();
        let worker = thread::spawn(move || {
            let loader = VrmAssetLoader::new();
            let tx_progress = tx.clone();
            let result = loader.load_with_progress(
                path_for_worker.to_string_lossy().as_ref(),
                move |stage| {
                    let _ = tx_progress.send(LoadMessage::Progress(stage));
                },
            );
            let final_msg = match result {
                Ok(asset) => LoadMessage::Done(asset),
                Err(e) => LoadMessage::Error(e.to_string()),
            };
            let _ = tx.send(final_msg);
        });

        Self {
            path,
            current_stage: LoadStage::Reading,
            receiver: rx,
            after_load,
            worker: Some(worker),
        }
    }

    /// Drain queued progress messages without blocking. Returns the terminal
    /// outcome (Done/Error) if one arrived, otherwise `None`.
    pub fn poll(&mut self) -> Option<LoadOutcome> {
        loop {
            match self.receiver.try_recv() {
                Ok(LoadMessage::Progress(stage)) => {
                    self.current_stage = stage;
                }
                Ok(LoadMessage::Done(asset)) => return Some(LoadOutcome::Done(asset)),
                Ok(LoadMessage::Error(e)) => return Some(LoadOutcome::Error(e)),
                Err(mpsc::TryRecvError::Empty) => return None,
                Err(mpsc::TryRecvError::Disconnected) => {
                    // Worker died without sending a terminal message.
                    return Some(LoadOutcome::Error("loader worker disconnected".to_string()));
                }
            }
        }
    }
}

impl GuiApp {
    /// Drain the active background load job (if any). On terminal
    /// outcome, finalise the avatar onto the scene; if the job was
    /// chained behind an `Open Project` action, also apply the
    /// pending `ProjectState` and emit any deferred load warnings.
    /// Called once per frame from `update()`.
    pub(super) fn poll_avatar_load_job(&mut self) {
        // Drain progress messages; bail if no terminal outcome arrived yet.
        let outcome = match self.library.avatar_load_job.as_mut().and_then(|j| j.poll()) {
            Some(o) => o,
            None => return,
        };

        // Worker finished — take ownership so we can mutate other GuiApp fields.
        let job = self.library.avatar_load_job.take().expect("job still present");
        if let Some(handle) = job.worker {
            let _ = handle.join();
        }

        match outcome {
            LoadOutcome::Done(asset) => {
                top_bar::finalize_avatar_load(self, &job.path, asset);
                if let AfterLoad::ApplyProject {
                    project_state,
                    project_path,
                    warnings,
                    restore_unsaved,
                } = job.after_load
                {
                    self.apply_project_state(&project_state);
                    // The state we just applied IS the on-disk state —
                    // re-seed the dirty baseline so the derived probe
                    // reads clean and a startup restore doesn't rewrite
                    // last_session.vvtproj with identical content.
                    self.mark_project_baseline();
                    self.project_status.explicit_file_stale =
                        restore_unsaved && project_path.is_some();
                    for w in &warnings.warnings {
                        self.push_warning_notification(t!("toast.warning", msg = w.to_string()));
                    }
                    if restore_unsaved {
                        self.push_notification(t!("toast.restored_unsaved_changes"));
                    }
                    if let Some(project_path) = project_path {
                        self.project_status.project_path = Some(project_path.clone());
                        self.push_success_notification(t!("toast.opened_project", path = project_path.display().to_string()));
                    }
                }
            }
            LoadOutcome::Error(e) => {
                self.push_error_notification(t!("toast.failed_load_avatar", path = job.path.display().to_string(), error = e.to_string()));
            }
        }
    }
}
