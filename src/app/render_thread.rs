use log::{error, info, warn};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::thread;

use crate::renderer::frame_input::RenderFrameInput;
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
    #[allow(dead_code)]
    Resize(u32, u32),
    Shutdown,
}

struct RenderThreadInner {
    renderer: VulkanRenderer,
    cmd_rx: Receiver<RenderCommand>,
    result_tx: SyncSender<RenderResult>,
}

pub struct RenderThread {
    cmd_tx: SyncSender<RenderCommand>,
    result_rx: Receiver<RenderResult>,
    handle: Option<thread::JoinHandle<()>>,
}

impl RenderThread {
    pub fn new(renderer: VulkanRenderer) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::sync_channel::<RenderCommand>(2);
        let (result_tx, result_rx) = mpsc::sync_channel::<RenderResult>(2);

        let init_barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let barrier_clone = init_barrier.clone();

        let inner = RenderThreadInner {
            renderer,
            cmd_rx,
            result_tx,
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
            result_rx,
            handle: Some(handle),
        }
    }

    pub fn submit(&self, cmd: RenderCommand) {
        if let Err(e) = self.cmd_tx.try_send(cmd) {
            warn!("render_thread: failed to send command: {}", e);
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

    pub fn try_recv_result(&self) -> Option<RenderResult> {
        match self.result_rx.try_recv() {
            Ok(result) => Some(result),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                warn!("render_thread: result channel disconnected");
                None
            }
        }
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
                    let result = match self.renderer.render(&input) {
                        Ok(r) => r,
                        Err(e) => {
                            error!("render_thread: render error: {}", e);
                            continue;
                        }
                    };
                    if let Err(e) = self.result_tx.send(result) {
                        error!("render_thread: failed to send result: {}", e);
                        return;
                    }
                }
                RenderCommand::RenderThumbnail {
                    request,
                    respond_to,
                } => {
                    // Drain any in-flight regular-frame readback first,
                    // so the main viewport doesn't lose a frame's data
                    // to the upcoming synchronous thumbnail wait.
                    if let Ok(Some(prev)) = self.renderer.harvest_pending() {
                        let _ = self.result_tx.try_send(prev);
                    }
                    let result = self.renderer.render_thumbnail(&request);
                    let _ = respond_to.send(result);
                }
                RenderCommand::Resize(w, h) => {
                    let _ = self.renderer.resize([w, h]);
                }
                RenderCommand::Shutdown => {
                    info!("render_thread: shutdown received");
                    self.renderer.flush_pending();
                    return;
                }
            }
        }
    }
}
