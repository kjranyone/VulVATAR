//! Output sink runtime reconciliation: bring the OutputRouter and (on
//! Windows) the MediaFoundation virtual camera in line with the
//! requested sink.

use log::info;
#[cfg(all(target_os = "windows", feature = "virtual-camera"))]
use log::warn;

use super::Application;
use crate::output::{FrameSink, OutputRouter};

impl Application {
    /// Bring the OutputRouter and (on Windows) the MF virtual camera in line
    /// with the requested sink. Idempotent.
    ///
    /// Critically, the OutputRouter swap and the MF camera lifetime are
    /// managed as **independent steps**. If a previous call swapped the
    /// router to `VirtualCamera` but `MfVirtualCamera::start` failed, a
    /// subsequent call still retries the MF start (the router swap is
    /// no-op the second time). Without this, a transient MF failure left
    /// the system in a permanently-broken `output.sink == VirtualCamera &&
    /// mf_virtual_camera == None` state.
    ///
    /// Returns `Err` only when the requested sink is `VirtualCamera` and
    /// MF camera registration fails. Callers should propagate to the user
    /// (notification) and roll their requested-sink shadow back to a
    /// known-good fallback like `SharedMemory`.
    pub fn ensure_output_sink_runtime(&mut self, sink: FrameSink) -> Result<(), String> {
        #[cfg(all(target_os = "windows", feature = "virtual-camera"))]
        {
            use crate::output::mf_virtual_camera::MfVirtualCamera;
            match sink {
                FrameSink::VirtualCamera => {
                    if self.mf_virtual_camera.is_none() {
                        let mut cam = MfVirtualCamera::new();
                        cam.start().map_err(|e| {
                            warn!("output: MFCreateVirtualCamera failed ({e})");
                            e
                        })?;
                        info!("output: MediaFoundation virtual camera registered");
                        self.mf_virtual_camera = Some(cam);
                    }
                }
                FrameSink::SharedTexture | FrameSink::SharedMemory | FrameSink::ImageSequence => {
                    if self.mf_virtual_camera.take().is_some() {
                        info!("output: MediaFoundation virtual camera unregistered");
                    }
                }
            }
        }

        if self.output.active_sink() != &sink {
            info!(
                "output: swapping sink {:?} -> {:?}",
                self.output.active_sink(),
                sink
            );
            self.output.shutdown();
            self.output = OutputRouter::new(sink.clone());
        }

        Ok(())
    }
}
