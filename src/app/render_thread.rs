use log::{error, info, warn};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::renderer::frame_input::RenderFrameInput;
use crate::renderer::output_export::ExportedPixelData;
use crate::renderer::{RenderResult, ThumbnailRenderResult, VulkanRenderer};

pub enum RenderCommand {
    RenderFrame(RenderFrameInput),
    /// One-shot offscreen render at thumbnail resolution. The render
    /// thread harvests the response synchronously (waits on the GPU
    /// fence rather than deferring to the next frame's pipelined
    /// readback) and posts `Result` to `respond_to`. Any pending
    /// regular-frame readback is drained first so the main viewport
    /// channel doesn't lose a frame's data.
    RenderThumbnail {
        request: RenderFrameInput,
        respond_to: SyncSender<Result<ThumbnailRenderResult, String>>,
    },
    /// Drop every entry from the renderer's texture and mesh caches.
    /// Called by the GUI when an avatar is removed or replaced —
    /// without this, the previous avatar's GPU textures (typically
    /// 100–500 MB per VRM) and mesh buffers (50–200 MB) stay
    /// resident in VRAM until process exit. After the next render
    /// the active avatar's data re-uploads on demand; the briefly
    /// re-paid upload cost (~10–30 ms hitch on a fresh swap) is
    /// strictly cheaper than monotonic VRAM growth across a session.
    EvictCaches,
    /// The output worker has finished consuming a GPU-exported frame.
    /// Return its lease to the renderer-owned export pool so the slot may be
    /// reused by a future frame.
    ReleaseExportLease(u64),
    #[allow(dead_code)]
    Resize(u32, u32),
    Shutdown,
}

/// Latest-frame mailbox for `RenderResult` handoff (render thread →
/// app). Replaces the original `sync_channel(2) + blocking send` model
/// so that a momentarily slow app drain (modal, asset load, GUI
/// hitch) cannot stall the render thread on `send`. When a previous
/// result is still pending and a newer one is produced, the older
/// result is dropped and accounted via `dropped` so the
/// in-flight counter on the app side stays accurate; if the dropped
/// result was carrying a GPU export lease, the lease is released back
/// to the export pool so the slot doesn't leak.
type ResultMailbox = Arc<Mutex<Option<RenderResult>>>;

struct RenderThreadInner {
    renderer: VulkanRenderer,
    cmd_rx: Receiver<RenderCommand>,
    result_mailbox: ResultMailbox,
    result_dropped: Arc<AtomicU64>,
}

pub struct RenderThread {
    cmd_tx: SyncSender<RenderCommand>,
    result_mailbox: ResultMailbox,
    result_dropped: Arc<AtomicU64>,
    handle: Option<thread::JoinHandle<()>>,
}

impl RenderThread {
    pub fn new(renderer: VulkanRenderer) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::sync_channel::<RenderCommand>(2);
        let result_mailbox: ResultMailbox = Arc::new(Mutex::new(None));
        let result_dropped = Arc::new(AtomicU64::new(0));

        let init_barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let barrier_clone = init_barrier.clone();

        let inner = RenderThreadInner {
            renderer,
            cmd_rx,
            result_mailbox: Arc::clone(&result_mailbox),
            result_dropped: Arc::clone(&result_dropped),
        };

        let handle = thread::Builder::new()
            .name("vulvatar-render".into())
            .stack_size(4 * 1024 * 1024)
            .spawn(move || {
                let mut inner = inner;
                inner.renderer.initialize();
                barrier_clone.wait();
                inner.run()
            })
            .expect("failed to spawn render thread");

        init_barrier.wait();

        Self {
            cmd_tx,
            result_mailbox,
            result_dropped,
            handle: Some(handle),
        }
    }

    /// Try to enqueue a command for the render thread. Returns `true`
    /// when the command was accepted, `false` when the bounded command
    /// channel was already full (in which case the command is dropped
    /// and the caller should *not* count it as in-flight). Callers
    /// driving the GUI repaint gate use the return value to decide
    /// whether to expect a result back next frame.
    pub fn submit(&self, cmd: RenderCommand) -> bool {
        match self.cmd_tx.try_send(cmd) {
            Ok(()) => true,
            Err(e) => {
                warn!("render_thread: failed to send command: {}", e);
                false
            }
        }
    }

    /// Submit a thumbnail render and return the receiving end of the
    /// per-job response channel. Caller polls the receiver in its own
    /// loop; the render thread drains its main pending readback (if
    /// any), then renders the thumbnail and posts the result.
    pub fn request_thumbnail(
        &self,
        request: RenderFrameInput,
    ) -> Receiver<Result<ThumbnailRenderResult, String>> {
        let (tx, rx) = mpsc::sync_channel::<Result<ThumbnailRenderResult, String>>(1);
        let cmd = RenderCommand::RenderThumbnail {
            request,
            respond_to: tx.clone(),
        };
        if let Err(e) = self.cmd_tx.try_send(cmd) {
            warn!("render_thread: failed to send thumbnail command: {}", e);
            // Drop tx → rx will surface a Disconnected when the caller polls.
            let _ = tx.try_send(Err(format!("queue full: {e}")));
        }
        rx
    }

    /// Take the latest result if one has been published. Returns
    /// `None` when the render thread hasn't produced anything new
    /// since the last call.
    pub fn try_recv_result(&self) -> Option<RenderResult> {
        self.result_mailbox
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }

    /// Read-and-reset the count of results the render thread had to
    /// drop because the app hadn't drained the mailbox yet. Each entry
    /// corresponds to a render that completed on the GPU but never
    /// reached `process_render_result`. The app uses this to keep its
    /// `render_results_pending` counter accurate.
    pub fn take_dropped_results(&self) -> u64 {
        self.result_dropped.swap(0, Ordering::Relaxed)
    }

    pub fn shutdown(&mut self) {
        let _ = self.cmd_tx.send(RenderCommand::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
            info!("render_thread: joined");
        }
    }
}

impl Drop for RenderThread {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl RenderThreadInner {
    /// Publish a freshly produced result via the mailbox. If a previous
    /// result was still pending, drop it and release its GPU lease (if
    /// any) so the export pool slot doesn't leak — only the renderer
    /// (i.e. this thread) is allowed to release, so the cleanup
    /// happens here rather than across the boundary.
    fn publish_result(&mut self, result: RenderResult) {
        let prev = {
            let mut mb = self
                .result_mailbox
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            mb.replace(result)
        };
        if let Some(prev) = prev {
            self.result_dropped.fetch_add(1, Ordering::Relaxed);
            if let Some(lease_id) = lease_id_for_release(&prev) {
                self.renderer.release_export_lease(lease_id);
            }
        }
    }

    fn run(mut self) {
        info!("render_thread: started");
        loop {
            let cmd = match self.cmd_rx.recv() {
                Ok(cmd) => cmd,
                Err(_) => {
                    info!("render_thread: command channel disconnected, exiting");
                    return;
                }
            };

            match cmd {
                RenderCommand::RenderFrame(input) => {
                    // On render error we still publish an empty result back
                    // to the app — `Application::render_results_pending`
                    // was bumped on submit and would otherwise leak
                    // forever, pinning the GUI's repaint gate at full
                    // rate. The empty `exported_frame` makes
                    // `process_render_result` a no-op (clears
                    // `rendered_pixels`) without surfacing a fake
                    // successful frame to the output worker.
                    let result = match self.renderer.render(&input) {
                        Ok(r) => r,
                        Err(e) => {
                            error!("render_thread: render error: {}", e);
                            crate::renderer::RenderResult {
                                extent: [0, 0],
                                timestamp_nanos: 0,
                                has_alpha: false,
                                stats: crate::renderer::RenderStats::default(),
                                exported_frame: None,
                            }
                        }
                    };
                    self.publish_result(result);
                }
                RenderCommand::RenderThumbnail {
                    request,
                    respond_to,
                } => {
                    // Drain any in-flight regular-frame readback first,
                    // so the main viewport doesn't lose a frame's data
                    // to the upcoming synchronous thumbnail wait.
                    if let Ok(Some(prev)) = self.renderer.harvest_pending() {
                        self.publish_result(prev);
                    }
                    let result = self.renderer.render_thumbnail(&request);
                    let _ = respond_to.send(result);
                }
                RenderCommand::Resize(w, h) => {
                    let _ = self.renderer.resize([w, h]);
                }
                RenderCommand::EvictCaches => {
                    self.renderer.flush_pending();
                    self.renderer.clear_caches();
                    info!("render_thread: caches evicted (avatar swap)");
                }
                RenderCommand::ReleaseExportLease(lease_id) => {
                    self.renderer.release_export_lease(lease_id);
                }
                RenderCommand::Shutdown => {
                    info!("render_thread: shutdown received");
                    self.renderer.flush_pending();
                    // On shutdown, any still-pending mailbox entry would
                    // leak its lease. Drain explicitly so the export pool
                    // doesn't keep a slot marked `Leased` past process
                    // teardown.
                    let leftover = self
                        .result_mailbox
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .take();
                    if let Some(prev) = leftover {
                        if let Some(lease_id) = lease_id_for_release(&prev) {
                            self.renderer.release_export_lease(lease_id);
                        }
                    }
                    return;
                }
            }
        }
    }
}

/// Extract the export pool lease id from a render result, if it
/// carries a GPU-token frame. CPU-readback results have no leased
/// slot and return `None`.
fn lease_id_for_release(result: &RenderResult) -> Option<u64> {
    let exported = result.exported_frame.as_ref()?;
    match &exported.pixel_data {
        ExportedPixelData::GpuFrameToken(_) => Some(exported.gpu_token_id),
        ExportedPixelData::CpuReadback(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_handoff::{
        ExternalHandleType, FrameLease, FrameLifetimeContract, GpuFrameToken, OutputSyncToken,
    };
    use crate::renderer::frame_input::RenderColorSpace;
    use crate::renderer::output_export::{ExportMetadata, ExportedFrame};
    use crate::renderer::RenderStats;
    use crate::output::{FallbackReason, HandoffPath};
    use std::sync::Arc;

    fn empty_result() -> RenderResult {
        RenderResult {
            extent: [0, 0],
            timestamp_nanos: 0,
            has_alpha: false,
            stats: RenderStats::default(),
            exported_frame: None,
        }
    }

    fn cpu_result() -> RenderResult {
        RenderResult {
            extent: [1, 1],
            timestamp_nanos: 0,
            has_alpha: false,
            stats: RenderStats::default(),
            exported_frame: Some(ExportedFrame {
                pixel_data: ExportedPixelData::CpuReadback(Arc::new(vec![0, 0, 0, 255])),
                extent: [1, 1],
                timestamp_nanos: 0,
                gpu_token_id: 0,
                export_metadata: ExportMetadata {
                    width: 1,
                    height: 1,
                    has_alpha: false,
                    color_space: RenderColorSpace::Srgb,
                    timestamp_nanos: 0,
                },
                handoff_path: HandoffPath::CpuReadback,
                fallback_reason: Some(FallbackReason::RequestedCpuReadback),
                preview_pixels: None,
            }),
        }
    }

    fn gpu_result(lease_id: u64) -> RenderResult {
        let token = GpuFrameToken {
            resource_id: 1,
            handle_type: ExternalHandleType::Win32Kmt,
            external_handle: None,
            sync: OutputSyncToken::ProducerWaitComplete,
            lease: FrameLease {
                lease_id,
                lifetime: FrameLifetimeContract::SingleConsumerImmediate,
            },
        };
        RenderResult {
            extent: [1, 1],
            timestamp_nanos: 0,
            has_alpha: false,
            stats: RenderStats::default(),
            exported_frame: Some(ExportedFrame {
                pixel_data: ExportedPixelData::GpuFrameToken(token),
                extent: [1, 1],
                timestamp_nanos: 0,
                gpu_token_id: lease_id,
                export_metadata: ExportMetadata {
                    width: 1,
                    height: 1,
                    has_alpha: false,
                    color_space: RenderColorSpace::Srgb,
                    timestamp_nanos: 0,
                },
                handoff_path: HandoffPath::GpuSharedFrame,
                fallback_reason: None,
                preview_pixels: None,
            }),
        }
    }

    #[test]
    fn lease_id_extraction_skips_results_without_a_lease() {
        assert_eq!(lease_id_for_release(&empty_result()), None);
        assert_eq!(
            lease_id_for_release(&cpu_result()),
            None,
            "CPU readback frames carry no export pool slot"
        );
    }

    #[test]
    fn lease_id_extraction_returns_gpu_token_id_for_gpu_path() {
        assert_eq!(lease_id_for_release(&gpu_result(42)), Some(42));
    }
}
