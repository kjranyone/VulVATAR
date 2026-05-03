//! Tracking worker lifecycle: start / stop / status.

use log::{info, warn};

use super::Application;
use crate::tracking::{CameraBackend, TrackingWorker};

impl Application {
    /// Start the tracking worker thread with default resolution/fps. No-op if already running.
    #[allow(dead_code)]
    pub fn start_tracking(&mut self, backend: CameraBackend) {
        self.start_tracking_with_params(backend, 640, 480, 30);
    }

    /// Like [`Self::start_tracking`] but allows specifying the webcam resolution and fps.
    ///
    /// If a worker is already running it is stopped and replaced with the new
    /// backend / parameters so the caller can switch backends at any time.
    pub fn start_tracking_with_params(
        &mut self,
        backend: CameraBackend,
        width: u32,
        height: u32,
        fps: u32,
    ) {
        self.start_tracking_with_params_full(backend, width, height, fps, false);
    }

    /// Same as [`Self::start_tracking_with_params`] but plumbs through the
    /// `prefer_lower_body` flag to the pose-model picker.
    pub fn start_tracking_with_params_full(
        &mut self,
        backend: CameraBackend,
        width: u32,
        height: u32,
        fps: u32,
        prefer_lower_body: bool,
    ) {
        if let Some(ref mut worker) = self.tracking_worker {
            if worker.is_running() {
                worker.stop();
                info!("app: stopped previous tracking worker for backend switch");
            }
        }
        // Tracking restart most likely means a different camera /
        // session / subject. Clear every avatar's motion-smoothing
        // state (1€ filter, hysteresis, root_offset EMA) so the
        // first frame after restart isn't blended against stale
        // history from the previous session.
        for avatar in self.avatars.iter_mut() {
            avatar.pose_solver_state.reset();
        }
        let shared_mailbox = self.tracking.shared_mailbox();
        let mut worker = TrackingWorker::new(shared_mailbox);
        worker.start_with_params_full(backend, width, height, fps, prefer_lower_body);
        if worker.is_running() {
            info!("app: tracking worker started");
        } else {
            warn!("app: tracking worker failed to start");
        }
        self.tracking_worker = Some(worker);
    }

    /// Stop the tracking worker thread. No-op if not running.
    pub fn stop_tracking(&mut self) {
        if let Some(ref mut worker) = self.tracking_worker {
            worker.stop();
            info!("app: tracking worker stopped");
        }
    }

    /// True if the webcam tracking worker is currently running. Used by
    /// the inspector to decide whether a resolution / framerate combo
    /// change should restart the worker in place.
    pub fn is_tracking_running(&self) -> bool {
        self.tracking_worker
            .as_ref()
            .is_some_and(|w| w.is_running())
    }
}
