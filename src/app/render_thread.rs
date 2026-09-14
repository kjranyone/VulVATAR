use log::{error, info, warn};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

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

/// GUI ticks can outrun the render thread (heavy scene, GPU shared with
/// tracking inference), so `submit` rejections are *normal backpressure*
/// under latest-frame-wins semantics — not errors. A per-drop `warn!`
/// therefore floods the log at GUI tick rate (~60-144 lines/s) the
/// moment the render thread falls behind. This aggregator keeps the
/// signal but bounds the spam: at most one line per
/// [`SUBMIT_DROP_WARN_COOLDOWN`], carrying how many drops accumulated
/// since the previous line.
struct SubmitDropWarnAggregator {
    last_warn: Option<Instant>,
    suppressed_since_warn: u64,
}

/// Minimum interval between two "command queue full" warnings.
const SUBMIT_DROP_WARN_COOLDOWN: Duration = Duration::from_secs(5);

/// Gaps between consecutive rendered frames larger than this are an
/// idle period (paused, no avatar, GUI not submitting) rather than a
/// production interval, so they reset the render-fps sampler instead
/// of diluting its EMA toward zero.
const ACTIVE_PRODUCTION_GAP: Duration = Duration::from_millis(500);
/// EMA weight of the newest frame interval in the render-fps sampler.
const RENDER_FPS_EMA_ALPHA: f32 = 1.0 / 8.0;

struct RenderThreadInner {
    renderer: VulkanRenderer,
    cmd_rx: Receiver<RenderCommand>,
    result_mailbox: ResultMailbox,
    result_dropped: Arc<AtomicU64>,
    /// Shared sink for the render-fps sampler (EMA frame interval in
    /// nanos; 0 = no measurement yet). Written by the render thread,
    /// read by the GUI via `RenderThread::render_fps`.
    render_frame_period_nanos: Arc<AtomicU64>,
    /// EMA of the CPU-side duration of `VulkanRenderer::render` (the
    /// previous frame's fence wait included). When this sits at the
    /// frame budget, the thread is blocked on GPU fences = GPU-bound;
    /// when it is small while fps sags, recording itself is the cost.
    render_cpu_nanos: Arc<AtomicU64>,
}

pub struct RenderThread {
    cmd_tx: SyncSender<RenderCommand>,
    result_mailbox: ResultMailbox,
    result_dropped: Arc<AtomicU64>,
    /// Cumulative count of `submit` calls rejected because the command
    /// channel was full. Monotonic display/diagnostic counter.
    submit_drops: Arc<AtomicU64>,
    /// GUI-side clone of the render-fps sampler sink (see
    /// `RenderThreadInner::render_frame_period_nanos`).
    render_frame_period_nanos: Arc<AtomicU64>,
    /// GUI-side clone of the render-CPU-time sampler sink.
    render_cpu_nanos: Arc<AtomicU64>,
    warn_state: Mutex<SubmitDropWarnAggregator>,
    handle: Option<thread::JoinHandle<()>>,
}

impl RenderThread {
    pub fn new(renderer: VulkanRenderer) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::sync_channel::<RenderCommand>(2);
        let result_mailbox: ResultMailbox = Arc::new(Mutex::new(None));
        let result_dropped = Arc::new(AtomicU64::new(0));
        let submit_drops = Arc::new(AtomicU64::new(0));
        let render_frame_period_nanos = Arc::new(AtomicU64::new(0));
        let render_cpu_nanos = Arc::new(AtomicU64::new(0));

        let init_barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let barrier_clone = init_barrier.clone();

        let inner = RenderThreadInner {
            renderer,
            cmd_rx,
            result_mailbox: Arc::clone(&result_mailbox),
            result_dropped: Arc::clone(&result_dropped),
            render_frame_period_nanos: Arc::clone(&render_frame_period_nanos),
            render_cpu_nanos: Arc::clone(&render_cpu_nanos),
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
            submit_drops,
            render_frame_period_nanos,
            render_cpu_nanos,
            warn_state: Mutex::new(SubmitDropWarnAggregator {
                last_warn: None,
                suppressed_since_warn: 0,
            }),
            handle: Some(handle),
        }
    }

    /// Try to enqueue a command for the render thread. Returns `true`
    /// when the command was accepted, `false` when the bounded command
    /// channel was already full (in which case the command is dropped
    /// and the caller should *not* count it as in-flight). Callers
    /// driving the GUI repaint gate use the return value to decide
    /// whether to expect a result back next frame.
    ///
    /// A `false` return is expected backpressure whenever the render
    /// thread's production rate falls below the GUI tick rate, so the
    /// warning is aggregated: at most one log line per
    /// `SUBMIT_DROP_WARN_COOLDOWN`, with the suppressed-drop count.
    pub fn submit(&self, cmd: RenderCommand) -> bool {
        match self.cmd_tx.try_send(cmd) {
            Ok(()) => true,
            Err(e) => {
                self.submit_drops.fetch_add(1, Ordering::Relaxed);
                let mut agg = self
                    .warn_state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                agg.suppressed_since_warn += 1;
                let due = agg
                    .last_warn
                    .map_or(true, |t| t.elapsed() >= SUBMIT_DROP_WARN_COOLDOWN);
                if due {
                    let suppressed = agg.suppressed_since_warn;
                    agg.suppressed_since_warn = 0;
                    agg.last_warn = Some(Instant::now());
                    drop(agg);
                    warn!(
                        "render_thread: command queue full ({}), dropping frame — \
                         render thread slower than GUI tick rate \
                         ({} dropped since last report, {} total)",
                        e,
                        suppressed,
                        self.submit_drops.load(Ordering::Relaxed)
                    );
                }
                false
            }
        }
    }

    /// Cumulative count of commands dropped because the channel was
    /// full. Read-and-*keep* — this is a display/diagnostic counter
    /// (status bar, `debug_gui.json`), unlike `take_dropped_results`
    /// which the app must drain to keep its in-flight math honest.
    pub fn submit_drops_total(&self) -> u64 {
        self.submit_drops.load(Ordering::Relaxed)
    }

    /// Effective production rate of the render thread in fps, from an
    /// EMA over back-to-back `RenderFrame` intervals only. Gaps longer
    /// than `ACTIVE_PRODUCTION_GAP` (pause, no avatar) reset the
    /// sampler instead of diluting the rate, so a resumed session
    /// reads `None` until two fresh frames have landed.
    pub fn render_fps(&self) -> Option<f32> {
        let nanos = self.render_frame_period_nanos.load(Ordering::Relaxed);
        (nanos > 0).then(|| 1_000_000_000.0 / nanos as f32)
    }

    /// EMA of the CPU-side `render()` duration in milliseconds
    /// (including the previous frame's fence wait), or `None` before
    /// the first frame. `render_fps()` below the target while this sits
    /// at the frame budget = GPU-bound; this small while fps sags =
    /// the recording path itself is the cost.
    pub fn render_cpu_ms(&self) -> Option<f32> {
        let nanos = self.render_cpu_nanos.load(Ordering::Relaxed);
        (nanos > 0).then(|| nanos as f32 / 1_000_000.0)
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
        // Render-fps sampler state. `last_frame_at` + `ema_period` are
        // local because only this thread writes them; the result is
        // published through the shared atomic in `inner`.
        let mut last_frame_at: Option<Instant> = None;
        let mut ema_period: Option<Duration> = None;
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
                    // Cooperative GPU exclusivity: while heavyweight
                    // DirectML initialisation is in flight (tracking
                    // start), skip Vulkan work entirely instead of
                    // racing the driver on one device — see
                    // `gpu_coordination` for the incident evidence.
                    // Publish an empty result so the app's
                    // `render_results_pending` gate doesn't leak.
                    if crate::gpu_coordination::is_exclusive_active() {
                        self.publish_result(crate::renderer::RenderResult {
                            extent: [0, 0],
                            timestamp_nanos: 0,
                            has_alpha: false,
                            stats: crate::renderer::RenderStats::default(),
                            exported_frame: None,
                            depth_ndc: None,

                            cloth_readback: Vec::new(),
                            vbo_audit: Vec::new(),
                            sdf_fields: Vec::new(),
                        });
                        continue;
                    }
                    // On render error we still publish an empty result back
                    // to the app — `Application::render_results_pending`
                    // was bumped on submit and would otherwise leak
                    // forever, pinning the GUI's repaint gate at full
                    // rate. The empty `exported_frame` makes
                    // `process_render_result` a no-op (clears
                    // `rendered_pixels`) without surfacing a fake
                    // successful frame to the output worker.
                    let render_started = Instant::now();
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
                                depth_ndc: None,

                                cloth_readback: Vec::new(),
                            vbo_audit: Vec::new(),
                            sdf_fields: Vec::new(),                            }
                        }
                    };
                    // CPU-side render duration (fence wait included):
                    // EMA published alongside the fps sampler so the
                    // heartbeat can separate GPU-bound from
                    // record-bound frames.
                    {
                        let cpu = render_started.elapsed();
                        let prev = self.render_cpu_nanos.load(Ordering::Relaxed);
                        let blended = if prev == 0 {
                            cpu.as_nanos() as u64
                        } else {
                            (prev as f32 * (1.0 - RENDER_FPS_EMA_ALPHA)
                                + cpu.as_nanos() as f32 * RENDER_FPS_EMA_ALPHA)
                                as u64
                        };
                        self.render_cpu_nanos.store(blended, Ordering::Relaxed);
                    }
                    // Feed the fps sampler on both the success and error
                    // path — both consumed a frame slot at production
                    // pace. The exclusive-active skip above intentionally
                    // does NOT reach here: no rendering happened.
                    let now = Instant::now();
                    if let Some(last) = last_frame_at {
                        let gap = now.duration_since(last);
                        if gap < ACTIVE_PRODUCTION_GAP {
                            ema_period = Some(match ema_period {
                                Some(ema) => ema.mul_f32(1.0 - RENDER_FPS_EMA_ALPHA)
                                    + gap.mul_f32(RENDER_FPS_EMA_ALPHA),
                                None => gap,
                            });
                            if let Some(ema) = ema_period {
                                self.render_frame_period_nanos.store(
                                    ema.as_nanos() as u64,
                                    Ordering::Relaxed,
                                );
                            }
                        } else {
                            // Idle gap (pause / no submits): resync the
                            // sampler without feeding it, and clear any
                            // pre-pause rate so callers read `None`
                            // rather than a stale number.
                            ema_period = None;
                            self.render_frame_period_nanos.store(0, Ordering::Relaxed);
                        }
                    }
                    last_frame_at = Some(now);
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
    use crate::output::{FallbackReason, HandoffPath};
    use crate::renderer::frame_input::RenderColorSpace;
    use crate::renderer::output_export::{ExportMetadata, ExportedFrame};
    use crate::renderer::RenderStats;
    use std::sync::Arc;

    fn empty_result() -> RenderResult {
        RenderResult {
            extent: [0, 0],
            timestamp_nanos: 0,
            has_alpha: false,
            stats: RenderStats::default(),
            exported_frame: None,
            depth_ndc: None,

            cloth_readback: Vec::new(),
                            vbo_audit: Vec::new(),
                            sdf_fields: Vec::new(),        }
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
            depth_ndc: None,

            cloth_readback: Vec::new(),
                            vbo_audit: Vec::new(),
                            sdf_fields: Vec::new(),        }
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
            depth_ndc: None,

            cloth_readback: Vec::new(),
                            vbo_audit: Vec::new(),
                            sdf_fields: Vec::new(),        }
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
