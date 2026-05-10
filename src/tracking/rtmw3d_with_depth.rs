//! RTMW3D + Depth Anything V2 Small streaming pipeline.
//!
//! The third pose provider, sized for 30-fps virtual-camera output.
//! RTMW3D's 2D body / hand / face landmarks are excellent and run in
//! ~38 ms; its baked-in z is a body-prior fiction that doesn't
//! capture forward / backward limb motion. DAv2-Small's relative
//! depth (~42 ms at 392×294) provides a measured per-pixel signal
//! that the provider calibrates to approximate metric scale via a
//! single body-anchor fit (shoulder span ≈ 0.40 m).
//!
//! ## Async architecture
//!
//! DAv2 runs on a dedicated worker thread so it overlaps with RTMW3D
//! on the calling thread. The worker reads a single-slot mailbox
//! (`mpsc::channel` with drain-old semantics) and writes to a
//! `Mutex<Option<Arc<DepthResult>>>` outbox. The provider always
//! reads the *latest available* depth result regardless of frame age,
//! so steady-state wall time on the calling thread is just RTMW3D
//! (~38 ms) and depth lags by 1 frame. Cold start blocks on the
//! first depth result.
//!
//! ```text
//!  estimate_pose(N) on caller thread:
//!    ├─ submit RGB frame N to depth_tx (drains older queued)
//!    ├─ run RTMW3D inline                            ┐
//!    │                                                │ in parallel
//!    │  depth-worker thread:                          │ with caller
//!    │    ├─ recv frame N (or drain to latest)        │
//!    │    └─ run DAv2 → write outbox (frame N depth) ─┘
//!    ├─ read latest outbox (could be N or N-1)
//!    ├─ calibrate, build metric, build skeleton
//!    └─ return PoseEstimate
//! ```
//!
//! Depth lag of 1 frame (33 ms at 30 fps) is invisible for typical
//! streaming motion: subject distance changes slowly compared to
//! limb articulation. Fast motion may briefly cause depth-sample
//! drift on a moving wrist; the small `SAMPLE_RADIUS_PX` median
//! window absorbs most of it.
//!
//! ## Trade-off vs the other depth-aware provider
//!
//! `cigpose-metric-depth` runs MoGe-2 ViT for true metric
//! coordinates (~175 ms) and is the right choice when accuracy
//! matters more than fps — offline analysis, ground-truth
//! comparisons. This provider trades MoGe's true metric for DAv2's
//! relative-then-calibrated metric so the inference budget fits
//! inside the streaming frame budget.
//!
//! ## Calibration math
//!
//! DAv2 emits inverse-depth-style values (`d_raw`, larger ≈ closer).
//! Camera-space metric Z is `z = c / d_raw` for a single unknown
//! scale `c`. With assumed vertical FOV (default 60°), camera
//! intrinsics are determined and the back-projection is:
//!
//! ```text
//!   x = (nx - 0.5) · 2·tan(h_fov/2) · z
//!   y = (ny - 0.5) · 2·tan(v_fov/2) · z
//! ```
//!
//! All three components scale linearly with `c`, so any single known
//! body distance pins it: `c = D / sqrt(geom)` where `D` is the
//! anatomical reference (shoulder span 0.40 m by default) and `geom`
//! is the squared bracketed term computed from the two anchor
//! pixels. One closed-form solve, no iteration.

use std::path::Path;

use super::provider::PoseProvider;
use super::{DetectionAnnotation, PoseEstimate, SourceSkeleton};
#[cfg(feature = "inference")]
use crate::asset::HumanoidBone;

#[cfg(feature = "inference")]
use std::sync::mpsc::{channel, Receiver, Sender};
#[cfg(feature = "inference")]
use std::sync::{Arc, Condvar, Mutex};
#[cfg(feature = "inference")]
use std::thread;

#[cfg(feature = "inference")]
use super::cigpose::{DecodedJoint2d, NUM_JOINTS};
#[cfg(feature = "inference")]
use super::depth_anything::{DepthAnythingFrame, DepthAnythingV2Inference};
#[cfg(feature = "inference")]
use super::metric_depth::MoGeFrame;
#[cfg(feature = "inference")]
use super::rtmw3d::Rtmw3dInference;
#[cfg(feature = "inference")]
use super::skeleton_from_depth::{
    build_options_from_calibration, build_skeleton, resolve_origin_metric, SAMPLE_RADIUS_PX,
};
#[cfg(feature = "inference")]
use log::{debug, error, info, warn};

/// Anatomical anchor used to calibrate DAv2's relative depth to
/// metres. Shoulder span (subject's left shoulder ↔ right shoulder)
/// is the most reliable anchor: present in nearly every pose, well-
/// localised by RTMW3D, and tighter inter-subject variance than
/// hip span (clothing / body-fat dependent).
#[cfg(feature = "inference")]
const ASSUMED_SHOULDER_SPAN_M: f32 = 0.40;

/// Fallback anchor when the shoulder pair fails. Hip span is
/// noisier (clothing, body shape) but at least anatomically present
/// when the subject is in the lower-body crop case.
#[cfg(feature = "inference")]
const ASSUMED_HIP_SPAN_M: f32 = 0.32;

/// Assumed camera vertical FOV. Webcams cluster between 50° and 65°;
/// 60° is a defensible default. The avatar solver consumes only
/// relative bone directions so a ±10° error in this assumption does
/// not break the output — it just rescales the absolute metric
/// numbers, which the solver quietly absorbs.
#[cfg(feature = "inference")]
const ASSUMED_VERTICAL_FOV_DEG: f32 = 60.0;

/// Submit a depth request once every N frames. Pose runs every frame
/// (RTMW3D + skeleton), but DAv2 runs at `1 / DEPTH_REFRESH_PERIOD` of
/// pose rate. Off-tick frames reuse the sticky outbox result.
///
/// Why: under DirectML EP, DAv2 and RTMW3D contend for the GPU
/// command queue (see docs/pose-provider-benchmark.md). Skipping
/// every other frame frees the GPU on off-ticks so RTMW3D runs at
/// its uncontested speed (~38 ms vs ~43 ms contended). Combined with
/// the DAv2 worker's cold-start sticky-outbox semantics, depth
/// updates land at ~half pose rate without producing visible motion
/// artefacts: subject↔camera distance is largely static within the
/// 33–66 ms refresh window.
///
/// Set to 1 to refresh every frame (no skip); 2 to halve depth rate;
/// 3 to third it. 2 is the sweet spot for streaming.
#[cfg(feature = "inference")]
const DEPTH_REFRESH_PERIOD: u64 = 2;

/// Rate-limit window for the "DAv2 scale calibration failed" WARN.
/// At pose rate (~30 fps) a sustained failure dumps 30 lines/sec —
/// the user can't read that and it crowds out other diagnostics. One
/// line every 5 seconds preserves the "something is wrong" signal
/// without burying the log.
#[cfg(feature = "inference")]
const CALIB_WARN_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

/// EMA blend factor for the calibration scalar `c`. The subject↔camera
/// distance is a slow physical quantity (changes on the order of
/// seconds), but the per-frame `c` solver is sensitive to ±1px
/// shoulder-keypoint jitter — DAv2's d_raw at the shoulder pixel
/// changes by ~1% per pixel of motion, which translates 1:1 into
/// `c = D/√geom`. Bulk diagnostic over 135 validation images measured
/// `c` stdev = 1.59 over a 1.1–8.4 range, i.e. frame-to-frame the
/// scale can swing 5–10% even when the subject is still. That swing
/// multiplies every metric Z (shoulder, elbow, wrist, hip) uniformly,
/// which the avatar solver perceives as the entire skeleton "breathing"
/// in/out — and the elbow Z signal we care about (~5–10cm at typical
/// hands-near-body framing) drowns in that breath.
///
/// 0.15 = ~6-frame time constant at the depth-worker rate (15 Hz with
/// REFRESH_PERIOD=2). Aggressive enough to track real subject motion
/// (subjects don't reposition meaningfully in <400 ms), gentle enough
/// to flatten per-frame jitter.
#[cfg(feature = "inference")]
const SCALE_C_EMA_ALPHA: f32 = 0.15;

/// Single mailbox slot wrapped in `Mutex<Option<...>>`. Used for both
/// the inbox (from caller → worker, drain-old via try_recv loop) and
/// outbox (from worker → caller, sticky latest result via Arc).
#[cfg(feature = "inference")]
struct DepthRequest {
    rgb: Vec<u8>,
    width: u32,
    height: u32,
    frame_index: u64,
}

#[cfg(feature = "inference")]
struct DepthResult {
    frame_index: u64,
    depth: DepthAnythingFrame,
    inference_ms: f32,
}

/// Sticky outbox: the latest completed depth result. `Arc<DepthResult>`
/// so the caller's read is a refcount bump (~ns) instead of cloning a
/// 100K+ pixel buffer (~ms). `Condvar` wakes cold-start waiters when
/// the very first result lands.
#[cfg(feature = "inference")]
struct DepthOutbox {
    slot: Mutex<Option<Arc<DepthResult>>>,
    cvar: Condvar,
}

pub struct Rtmw3dWithDepthProvider {
    #[cfg(feature = "inference")]
    rtmw3d: Rtmw3dInference,
    /// Inbox sender. `Option` so `Drop` can take and drop the
    /// sender first to terminate the worker's `recv()` loop.
    #[cfg(feature = "inference")]
    depth_tx: Option<Sender<DepthRequest>>,
    /// Sticky outbox holding the latest finished depth result. The
    /// worker writes here; the caller reads each frame.
    #[cfg(feature = "inference")]
    depth_outbox: Arc<DepthOutbox>,
    /// Worker thread handle, joined on `Drop`.
    #[cfg(feature = "inference")]
    depth_thread: Option<thread::JoinHandle<()>>,
    /// One-shot guard so a dead depth thread logs once instead of
    /// flooding stderr at 30 fps. Flips to true the first time a
    /// `tx.send` returns SendError. Without this we silently degraded
    /// to stale-outbox depth forever with no visible signal.
    #[cfg(feature = "inference")]
    depth_thread_dead_logged: bool,
    /// Cached for `label()` — backend label is read from the worker-
    /// owned `DepthAnythingV2Inference`, but we don't keep that
    /// instance on the main thread, so cache the string at startup.
    #[cfg(feature = "inference")]
    depth_backend_label: String,
    load_warnings: Vec<String>,
    /// Latest pose calibration pushed from the GUI via the tracking
    /// mailbox. `None` means no calibration is active — the depth
    /// pipeline keeps its default hip-preferred / shoulder-fallback
    /// anchor selection. When `Some` and the mode is `UpperBody`,
    /// `BuildOptions::force_shoulder_anchor` is set so a hallucinated
    /// hip detection (e.g. desk surface in foreground) can't override
    /// the user-acknowledged "I'm only showing my upper body"
    /// declaration.
    pose_calibration: Option<crate::tracking::PoseCalibration>,
    /// Transient mode hint pushed by the GUI while the calibration
    /// modal is open. **Overrides** `pose_calibration.mode` for the
    /// `force_shoulder_anchor` decision so re-calibration in either
    /// direction works (see
    /// [`super::skeleton_from_depth::build_options_from_calibration`]
    /// for the full contract). `None` whenever the modal is closed.
    calibration_mode_hint: Option<crate::tracking::CalibrationMode>,
    /// EMA-smoothed calibration scalar from the previous frame. Used
    /// to dampen per-frame `c` jitter that otherwise breathes the
    /// entire metric skeleton in/out (see [`SCALE_C_EMA_ALPHA`]).
    /// `None` until the first successful calibration. Independent of
    /// the user-driven `pose_calibration` above — that one seeds the
    /// solver / anchor selection, this one smooths the per-frame
    /// `calibrate_scale` output regardless of whether the user has
    /// captured an explicit pose calibration.
    #[cfg(feature = "inference")]
    scale_c_ema: Option<f32>,
    /// Active torso-template capture buffer, set by the GUI while
    /// the calibration modal is in `Collecting`. `Some` means each
    /// successful frame contributes one sample per cell to the
    /// running median; `None` means the per-frame capture path is
    /// skipped (no overhead in normal streaming).
    #[cfg(feature = "inference")]
    torso_capture: Option<super::skeleton_from_depth::TorsoCaptureBuffer>,
    /// Rate-limit the per-frame "DAv2 scale calibration failed" WARN.
    /// `calibrate_scale` runs at pose rate (~30 fps); a sustained
    /// failure (e.g. plausibility band rejecting every frame because
    /// the user is at a different distance than when they calibrated)
    /// would otherwise dump 30 lines/sec to stderr. Holds the
    /// `Instant` of the last emitted warning; the next emission waits
    /// at least `CALIB_WARN_INTERVAL` past that.
    #[cfg(feature = "inference")]
    last_calib_warn_at: Option<std::time::Instant>,
    /// Tracks whether the previous frame's `calibrate_scale` failed.
    /// Used to emit a one-shot INFO when calibration recovers, so the
    /// user sees "ok, working again" rather than just silently stopping
    /// the WARN spam.
    #[cfg(feature = "inference")]
    calib_was_failing: bool,
}

impl Rtmw3dWithDepthProvider {
    /// Snapshot the EMA-smoothed scale `c` for diagnostics. Returns
    /// `None` until the first successful calibration. Called by
    /// temporal-sequence diagnostics that want to plot scale stability
    /// across a sorted run of frames.
    #[cfg(feature = "inference")]
    pub fn scale_c_ema(&self) -> Option<f32> {
        self.scale_c_ema
    }

    #[cfg(feature = "inference")]
    pub fn from_models_dir(models_dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = models_dir.as_ref();

        // Force FaceMesh on CPU EP: with DAv2 on DirectML running
        // concurrently in the worker thread, sharing the GPU command
        // queue inflates FaceMesh from ~3 ms to ~13 ms. CPU runs the
        // small face cascade in ~7 ms uncontested, net win of ~6 ms
        // per frame.
        let mut rtmw3d = Rtmw3dInference::from_models_dir_with_face_ep(
            dir,
            super::face_mediapipe::FaceMeshEp::ForceCpu,
        )?;
        let warnings = rtmw3d.take_load_warnings();
        let dav2 = DepthAnythingV2Inference::from_model_path(dir.join("dav2_small.onnx"))?;
        let depth_backend_label = dav2.backend().label().to_string();

        let (depth_tx, depth_rx) = channel::<DepthRequest>();
        let depth_outbox = Arc::new(DepthOutbox {
            slot: Mutex::new(None),
            cvar: Condvar::new(),
        });
        let depth_outbox_for_thread = Arc::clone(&depth_outbox);
        let depth_thread = thread::Builder::new()
            .name("dav2-depth".into())
            .spawn(move || depth_worker_loop(dav2, depth_rx, depth_outbox_for_thread))
            .map_err(|e| format!("spawn depth worker: {e}"))?;

        info!(
            "RTMW3D+DAv2 provider ready (RTMW3D: {}, DAv2: {} async)",
            rtmw3d.backend().label(),
            depth_backend_label,
        );

        Ok(Self {
            rtmw3d,
            depth_tx: Some(depth_tx),
            depth_outbox,
            depth_thread: Some(depth_thread),
            depth_thread_dead_logged: false,
            depth_backend_label,
            load_warnings: warnings,
            pose_calibration: None,
            calibration_mode_hint: None,
            scale_c_ema: None,
            torso_capture: None,
            last_calib_warn_at: None,
            calib_was_failing: false,
        })
    }

    #[cfg(not(feature = "inference"))]
    pub fn from_models_dir(_: impl AsRef<Path>) -> Result<Self, String> {
        Err("RTMW3D+DAv2 provider requires the `inference` cargo feature".to_string())
    }
}

#[cfg(feature = "inference")]
impl Drop for Rtmw3dWithDepthProvider {
    fn drop(&mut self) {
        // Drop the sender first — the worker's `rx.recv()` then
        // returns Err(RecvError) and the loop exits cleanly. Joining
        // afterwards waits for any in-flight DAv2 inference to
        // complete (~50 ms worst case).
        self.depth_tx.take();
        if let Some(handle) = self.depth_thread.take() {
            let _ = handle.join();
        }
    }
}

/// DAv2 worker thread loop. Owns the inference instance, reads
/// requests via `rx`, drains the queue down to the latest entry
/// (so a slow worker never falls behind on stale frames), runs
/// inference, and publishes to the sticky outbox. Exits when `rx`
/// returns `Err` (provider dropped its `Sender`).
#[cfg(feature = "inference")]
fn depth_worker_loop(
    mut dav2: DepthAnythingV2Inference,
    rx: Receiver<DepthRequest>,
    outbox: Arc<DepthOutbox>,
) {
    while let Ok(mut req) = rx.recv() {
        // Drain: consume any newer requests that piled up while we
        // were processing the previous frame. Keeps depth always on
        // the latest available frame.
        while let Ok(newer) = rx.try_recv() {
            req = newer;
        }
        let t0 = std::time::Instant::now();
        let depth_opt = dav2.estimate(&req.rgb, req.width, req.height);
        let inference_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let depth = match depth_opt {
            Some(d) => d,
            None => continue,
        };
        let result = Arc::new(DepthResult {
            frame_index: req.frame_index,
            depth,
            inference_ms,
        });
        {
            let mut slot = outbox.slot.lock().unwrap();
            *slot = Some(result);
        }
        outbox.cvar.notify_all();
    }
}

impl PoseProvider for Rtmw3dWithDepthProvider {
    fn label(&self) -> String {
        #[cfg(feature = "inference")]
        {
            format!(
                "RTMW3D+DAv2 / {} + {} async",
                self.rtmw3d.backend().label(),
                self.depth_backend_label
            )
        }
        #[cfg(not(feature = "inference"))]
        {
            "RTMW3D+DAv2 (inference disabled)".to_string()
        }
    }

    fn take_load_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.load_warnings)
    }

    fn set_calibration(&mut self, calibration: Option<crate::tracking::PoseCalibration>) {
        // Mirror the persisted `force_shoulder_anchor` onto the inner
        // `Rtmw3dInference` so the fallback paths in
        // `estimate_pose_internal` (RTMW3D keypoint shortage at the
        // top, DAv2 cold-start scale-calibration failure later) see
        // the same anchor contract as the depth-built skeleton. Both
        // fallbacks return `rtmw_est` straight through from the inner
        // RTMW3D, which builds its source skeleton via the override
        // logic in `Rtmw3dInference::process_pose`
        // (`hint Some(UpperBody) → true`,
        // `hint Some(FullBody) → false`,
        // `hint None → persisted force_shoulder_anchor`). Without this
        // mirror the persisted side of that match would always read
        // `false`, so a fallback frame outside the modal would build
        // a hip-anchored skeleton even when the user calibrated
        // `UpperBody`.
        #[cfg(feature = "inference")]
        {
            self.rtmw3d.force_shoulder_anchor = calibration
                .as_ref()
                .map(|c| matches!(c.mode, crate::tracking::CalibrationMode::UpperBody))
                .unwrap_or(false);
        }
        self.pose_calibration = calibration;
    }

    fn set_calibration_mode_hint(&mut self, hint: Option<crate::tracking::CalibrationMode>) {
        // Override the persisted calibration's mode for the
        // `force_shoulder_anchor` decision while the modal is open —
        // see `build_options_from_calibration` for the contract.
        self.calibration_mode_hint = hint;
        // Also mirror onto the inner Rtmw3dInference so fallback frames
        // (see `set_calibration` above for the rationale) honour the
        // same override the depth-built path does.
        #[cfg(feature = "inference")]
        {
            self.rtmw3d.force_shoulder_anchor_hint = hint;
        }
    }

    fn set_torso_capture(&mut self, enabled: bool) {
        #[cfg(feature = "inference")]
        {
            // Toggling on starts a fresh buffer; toggling off
            // discards any in-flight buffer (the canonical exit is
            // through `take_torso_template`, which moves the buffer
            // out and finalises it). This drop-on-off path covers
            // user-initiated cancel of the calibration modal mid-
            // capture, where we don't want stale partial data to
            // leak into the next capture.
            self.torso_capture = if enabled {
                Some(super::skeleton_from_depth::TorsoCaptureBuffer::new())
            } else {
                None
            };
        }
        #[cfg(not(feature = "inference"))]
        {
            let _ = enabled;
        }
    }

    fn take_torso_template(&mut self) -> Option<crate::tracking::TorsoDepthTemplate> {
        #[cfg(feature = "inference")]
        {
            self.torso_capture.take().and_then(|buf| buf.finalize())
        }
        #[cfg(not(feature = "inference"))]
        {
            None
        }
    }

    fn estimate_pose(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> PoseEstimate {
        #[cfg(feature = "inference")]
        return self.estimate_pose_internal(rgb_data, width, height, frame_index);

        #[cfg(not(feature = "inference"))]
        {
            let _ = (rgb_data, width, height);
            empty_estimate(frame_index)
        }
    }
}

#[cfg(feature = "inference")]
impl Rtmw3dWithDepthProvider {
    fn estimate_pose_internal(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> PoseEstimate {
        let expected_len = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(3);
        if rgb_data.len() < expected_len || width == 0 || height == 0 {
            return empty_estimate(frame_index);
        }

        let t_total = std::time::Instant::now();

        // Phase 0: dispatch DAv2 work for this frame to the worker
        // thread. We submit only on every `DEPTH_REFRESH_PERIOD`-th
        // frame to halve GPU contention between RTMW3D + DAv2 — but
        // we ALWAYS submit on cold start (outbox empty) so the very
        // first call doesn't deadlock waiting for a never-submitted
        // depth. The worker drains older queued requests so depth is
        // always being computed for the most recent submission.
        let outbox_empty = self.depth_outbox.slot.lock().unwrap().is_none();
        let submit_depth_this_frame =
            outbox_empty || frame_index.is_multiple_of(DEPTH_REFRESH_PERIOD);
        if submit_depth_this_frame {
            if let Some(tx) = self.depth_tx.as_ref() {
                let req = DepthRequest {
                    rgb: rgb_data.to_vec(),
                    width,
                    height,
                    frame_index,
                };
                // SendError only happens if the worker thread died —
                // at which point we'll fall back to whatever's left
                // in the outbox (sticky last result). Log loudly the
                // first time so a power user diagnosing "tracking
                // feels stale" sees the cause in the log; gate further
                // sends behind the same flag so the log isn't
                // hammered at 30 fps.
                if let Err(e) = tx.send(req) {
                    if !self.depth_thread_dead_logged {
                        error!(
                            "DAv2 depth worker died (frame {}); falling back to sticky outbox forever: {}",
                            frame_index, e
                        );
                        self.depth_thread_dead_logged = true;
                    }
                }
            }
        }

        // Phase 1: RTMW3D — full pipeline (YOLOX + RTMW3D + FaceMesh).
        // Runs in parallel with the depth worker. We keep its
        // `annotation.keypoints` (whole-frame normalised 2D + score)
        // and `skeleton.expressions` / `face_mesh_confidence` (the
        // FaceMesh cascade output). Everything else from
        // `rtmw_est.skeleton` gets thrown away and rebuilt with
        // metric depth below.
        let t_rtmw = std::time::Instant::now();
        let mut rtmw_est = self.rtmw3d.estimate_pose(rgb_data, width, height, frame_index);
        let dt_rtmw = t_rtmw.elapsed();

        let joints_2d: Vec<DecodedJoint2d> = rtmw_est
            .annotation
            .keypoints
            .iter()
            .map(|&(nx, ny, score)| DecodedJoint2d { nx, ny, score })
            .collect();
        if joints_2d.len() < NUM_JOINTS {
            // RTMW3D failed to emit a full keypoint set — pass its
            // empty estimate through unchanged.
            return rtmw_est;
        }

        // Phase 2: read latest depth from outbox. Block on first
        // frame (cold start) until the worker produces something;
        // after that the outbox always has *some* result so we read
        // non-blocking, possibly using a 1–2-frame-old depth.
        let t_wait = std::time::Instant::now();
        let depth_result = {
            let mut slot = self.depth_outbox.slot.lock().unwrap();
            while slot.is_none() {
                slot = self.depth_outbox.cvar.wait(slot).unwrap();
            }
            // Sticky: don't take, just clone the Arc — subsequent
            // frames can keep using this result if no newer one has
            // arrived yet.
            Arc::clone(slot.as_ref().unwrap())
        };
        let dt_wait = t_wait.elapsed();
        let depth_age = frame_index.saturating_sub(depth_result.frame_index);
        let dav2_frame = &depth_result.depth;

        // Phase 3: calibrate scale from a body anchor. We use the
        // CURRENT frame's keypoints with the (possibly older) depth
        // map. For typical streaming motion the depth-vs-keypoint
        // misalignment is sub-keypoint and absorbed by the
        // sample-window median.
        //
        // After getting the raw `c`, blend it into the EMA so the
        // metric frame uses a temporally-smoothed scale. Without this,
        // shoulder-keypoint jitter swings `c` ±5–10% per frame and the
        // entire metric skeleton breathes uniformly, drowning the
        // ~5–10cm elbow-Z signal we depend on at hands-near-body
        // framings.
        let t_calib = std::time::Instant::now();
        // While the calibration modal is open the user is most likely
        // re-capturing from a different distance than the persisted
        // calibration was taken at. Suppressing the persisted value
        // here disables the depth-plausibility gate inside
        // `calibrate_scale` (`anchor_depth_m ± max_deviation`), which
        // would otherwise `PlausibilityRejected` every recapture frame
        // and starve the EMA — leaving the recapture session running
        // on the RTMW3D synthetic-z fallback so the user can never
        // produce a fresh depth-aware calibration. Once the modal
        // closes and `persist_calibration` writes the new value into
        // `self.pose_calibration`, the gate snaps back on with the
        // updated band. (`resolve_origin_metric` / `build_skeleton`
        // below still receive the persisted calibration because they
        // use orthogonal channels — torso depth template, shoulder
        // span — that are not affected by the recapture distance.)
        let plausibility_calibration = if self.calibration_mode_hint.is_some() {
            None
        } else {
            self.pose_calibration.as_ref()
        };
        let calib_result = calibrate_scale(
            &joints_2d,
            dav2_frame,
            width,
            height,
            plausibility_calibration,
        );
        let mut calib = match calib_result {
            Ok(c) => {
                // Recovery path: if the previous frame failed, surface
                // a one-shot INFO so the user can correlate the WARN
                // burst with "tracking started working again".
                if self.calib_was_failing {
                    info!(
                        "RTMW3D+DAv2: per-frame DAv2 scale calibration recovered (anchor={}, c={:.3})",
                        c.anchor_label, c.scale_c
                    );
                    self.calib_was_failing = false;
                }
                c
            }
            Err(failure) => {
                // Per-frame DAv2 scale calibration failed this frame
                // (anchor occluded, degenerate side-profile, plausibility
                // band rejected, etc — see `failure` for the per-anchor
                // breakdown). This is *separate* from the user's saved
                // `PoseCalibration` profile, which the GUI surfaces as
                // "calibrated"; the saved profile only narrows the
                // plausibility band inside `calibrate_scale`, it does
                // not itself produce a per-frame scale.
                //
                // Fall back to the EMA value if we have one — keeps
                // tracking alive instead of dropping all metric Z to
                // RTMW3D synthetic. Cold start (no EMA yet) still
                // falls through to the RTMW3D-only estimate.
                self.calib_was_failing = true;
                let now = std::time::Instant::now();
                let should_warn = self
                    .last_calib_warn_at
                    .map(|t| now.duration_since(t) >= CALIB_WARN_INTERVAL)
                    .unwrap_or(true);
                if should_warn {
                    self.last_calib_warn_at = Some(now);
                    if let Some(prev) = self.scale_c_ema {
                        warn!(
                            "RTMW3D+DAv2: per-frame DAv2 scale calibration failed — reusing EMA scale_c={:.4} ({})",
                            prev, failure
                        );
                    } else {
                        warn!(
                            "RTMW3D+DAv2: per-frame DAv2 scale calibration failed (no EMA seed yet) — falling back to RTMW3D synthetic z ({})",
                            failure
                        );
                    }
                }
                if let Some(prev) = self.scale_c_ema {
                    Calibration {
                        scale_c: prev,
                        h_half: (ASSUMED_VERTICAL_FOV_DEG.to_radians() * 0.5).tan()
                            * (width as f32 / height.max(1) as f32),
                        v_half: (ASSUMED_VERTICAL_FOV_DEG.to_radians() * 0.5).tan(),
                        anchor_label: "ema-fallback",
                        d_raw_a: f32::NAN,
                        d_raw_b: f32::NAN,
                    }
                } else {
                    return rtmw_est;
                }
            }
        };
        let raw_c = calib.scale_c;
        let smoothed_c = match self.scale_c_ema {
            Some(prev) => prev * (1.0 - SCALE_C_EMA_ALPHA) + raw_c * SCALE_C_EMA_ALPHA,
            None => raw_c, // Cold start: snap to first measurement.
        };
        self.scale_c_ema = Some(smoothed_c);
        calib.scale_c = smoothed_c;
        let dt_calib = t_calib.elapsed();
        debug!(
            "RTMW3D+DAv2 calibration: anchor={} d_raw=({:.4}, {:.4}) → c_raw={:.4} c_ema={:.4}",
            calib.anchor_label, calib.d_raw_a, calib.d_raw_b, raw_c, smoothed_c
        );

        // Phase 4: build a MoGeFrame-shaped point cloud out of DAv2 +
        // calibration so the shared `skeleton_from_depth` helpers
        // apply unchanged. The DAv2 grid (392×294 ≈ 115 K pixels)
        // back-projects in well under a millisecond.
        let t_metric = std::time::Instant::now();
        let metric_frame = build_metric_frame_from_dav2(dav2_frame, &calib);
        let dt_metric = t_metric.elapsed();

        // Phase 4.5 (active only during calibration capture): contribute
        // one frame of torso-template data. The buffer's `add_frame`
        // gates on shoulder + hip visibility internally — frames
        // where the four torso anchors aren't all above the floor
        // are silently skipped. Cost when no capture is active is
        // a single Option::is_some branch.
        if let Some(buf) = self.torso_capture.as_mut() {
            let _ = buf.add_frame(&joints_2d, &metric_frame);
        }

        // Phase 5: skeleton from depth.
        let t_skel = std::time::Instant::now();
        // BuildOptions derived from the active pose calibration plus
        // the GUI's transient mode hint. The hint **overrides** the
        // persisted calibration's mode for the `force_shoulder_anchor`
        // decision (in either direction): `Some(UpperBody)` forces
        // shoulder anchor even before any calibration is persisted —
        // makes the first `UpperBody` capture possible — and
        // `Some(FullBody)` clears the flag even when the persisted
        // calibration is `UpperBody`, which is what makes
        // re-calibrating from `UpperBody` back to `FullBody`
        // possible at all. See
        // `build_options_from_calibration` for the full contract.
        let opts = build_options_from_calibration(
            self.pose_calibration.as_ref(),
            self.calibration_mode_hint,
        );
        let mut skeleton = match resolve_origin_metric(
            &joints_2d,
            &metric_frame,
            opts,
            self.pose_calibration.as_ref(),
        ) {
            Some(anchor) => build_skeleton(
                frame_index,
                &joints_2d,
                &metric_frame,
                anchor,
                opts,
                self.pose_calibration.as_ref(),
            ),
            None => {
                warn!("RTMW3D+DAv2: no body anchor — emitting empty skeleton");
                SourceSkeleton::empty(frame_index)
            }
        };
        let dt_skel = t_skel.elapsed();

        // Phase 6: inherit RTMW3D's face pose + FaceMesh cascade
        // output. The face track in particular is much better when
        // taken from RTMW3D's own body-derived `derive_face_pose_from_body`
        // (which uses the model's nz heatmap trained on face geometry)
        // than from `derive_face_pose_metric` over DAv2 — the head is
        // small in DAv2's 392×294 grid so per-pixel depth at the 5
        // face landmarks is near-uniform noise. Expressions + mesh
        // confidence have nothing to do with the body depth pipeline,
        // so reusing them also avoids running FaceMesh twice.
        skeleton.face = rtmw_est.skeleton.face;
        skeleton.expressions = std::mem::take(&mut rtmw_est.skeleton.expressions);
        skeleton.face_mesh_confidence = rtmw_est.skeleton.face_mesh_confidence;
        if let (Some(ref mut fp), Some(mesh_conf)) =
            (skeleton.face.as_mut(), skeleton.face_mesh_confidence)
        {
            fp.confidence = fp.confidence.max(mesh_conf);
        }

        // Phase 7: replace per-joint position[xyz] with RTMW3D's own
        // source-space coordinates while preserving the metric depth
        // sampled by Phase 5 in `metric_depth_m`.
        //
        // Why: the dav2 path back-projects every pixel through the
        // depth-model's per-pixel Z to recover (X, Y) in metric metres.
        // That back-projection inherits DAv2's lateral depth-map bias
        // — measured at ~22cm L/R asymmetry on photoreal frontal-pose
        // images even when the actual subject is square-on — and the
        // bias *propagates into XY*, not just Z, because the
        // back-projection geometry is `X = (px - cx) * Z / fx`.
        // Downstream rotation calculations (`compute_body_yaw_3d` and
        // every direction-vector bone solve in `pose_solver`) read this
        // skewed XYZ and rotate the avatar to match it, producing
        // ~50° body-yaw on a frontal pose. The validation set in
        // `diagnostics/calib_eff_validation/body_yaw_report.md` makes
        // this concrete: dav2 path mean |yaw error| 51.5° vs RTMW3D
        // nz-only path 19.6° on the same 23 evaluation images.
        //
        // RTMW3D's own SimCC-decoded (nx, ny, nz) is unbiased (its
        // training disentangles 2D-detection from depth bin), so
        // overwriting position[xyz] from `rtmw_est.skeleton` reverts
        // the rotation path to the unbiased signal. The metric depth
        // (which the back-percentile / torso-template logic in
        // skeleton_from_depth still produces correctly) survives on
        // `metric_depth_m` for consumers that legitimately need
        // real-world distance — root translation Z, bone-length
        // reconstruction, the torso template itself.
        //
        // Joints that exist on `skeleton` but not on
        // `rtmw_est.skeleton` (e.g. spine-chain proxies built with a
        // different fallback when one of the source paths failed)
        // keep their dav2-derived position so we don't regress on
        // partial-coverage cases.
        // Capture the shoulders' DAv2-metric X *before* Phase 7 so we
        // can recover the per-frame metres-per-source-unit conversion
        // factor used by the elbow-Z injection below. After the
        // overwrite loop the metric XY is gone — only `metric_depth_m`
        // (the Z scalar) survives on the joint. We only read X here
        // because the conversion factor is derived from the X-axis
        // shoulder span; Y/Z are not needed.
        let shoulder_metric_x_pre_overwrite = {
            let l = skeleton
                .joints
                .get(&HumanoidBone::LeftUpperArm)
                .map(|j| j.position[0]);
            let r = skeleton
                .joints
                .get(&HumanoidBone::RightUpperArm)
                .map(|j| j.position[0]);
            l.zip(r)
        };

        for (bone, joint) in skeleton.joints.iter_mut() {
            if let Some(rtmw_joint) = rtmw_est.skeleton.joints.get(bone) {
                joint.position = rtmw_joint.position;
            }
        }
        for (bone, joint) in skeleton.fingertips.iter_mut() {
            if let Some(rtmw_joint) = rtmw_est.skeleton.fingertips.get(bone) {
                joint.position = rtmw_joint.position;
            }
        }

        // Phase 7.5: re-inject Z components from DAv2 metric depth.
        // Phase 7 above replaced every joint's `position` with RTMW3D's
        // own (nx, ny, nz) to dodge the lateral-bias-into-XY problem in
        // the dav2 back-projection — but RTMW3D's nz is a single
        // softmax over 576 depth bins and does a poor job of resolving
        // small forward/backward displacements at the elbow / inter-
        // shoulder Δz at 30°–60° body yaw.
        //
        // For each joint we re-inject DAv2's signal as a *relative*
        // delta against the next-anchored joint up the chain (shoulder
        // Δ vs the shoulder midpoint; elbow Δ vs the shoulder), which
        // cancels DAv2's near-uniform lateral bias (the bias is roughly
        // constant across nearby pixels in the same body region). The
        // delta is converted to source-Z units via the shoulder span
        // (metres-per-source-unit), keeping the result dimensionally
        // compatible with the IK and `compute_body_yaw_3d` that consume
        // the source skeleton.
        //
        // Order matters: shoulders first so the elbow delta is taken
        // against the corrected shoulder Z, not the RTMW3D-nz one.
        // `pose_calibration` flows into `metric_depth_band` so the
        // outlier-rejection range tightens around the user's habitual
        // seat distance — the validation set has 9/16 mocap rotation
        // frames where DAv2 samples background depth on the occluded
        // shoulder, and a wider band would let those bogus samples
        // through.
        //
        // Wrist Z is *not* re-injected: the MCP-centroid that drives
        // the wrist's pixel sample lands on the back of the hand, which
        // routinely (arms-forward / arms-back poses on the
        // calibration_effect_validation/arm_reach set) reads as either
        // background depth (3–5 m) or a body region behind the actual
        // wrist. RTMW3D's nz captures wrist forward/backward extension
        // more reliably than DAv2 at the MCP centroid does.
        if let Some((sl_metric_x, sr_metric_x)) = shoulder_metric_x_pre_overwrite {
            let cal = self.pose_calibration.as_ref();
            inject_shoulder_bz_from_dav2(&mut skeleton, sl_metric_x, sr_metric_x, cal);
            inject_elbow_z_from_dav2(&mut skeleton, sl_metric_x, sr_metric_x);
        }
        // Hand orientation is also re-derived from RTMW3D's own
        // (unbiased) hand-keypoint xyz — same rationale as the body
        // joints. Keep the dav2-built one only as a fallback for
        // when RTMW3D failed to produce one (low hand-keypoint
        // confidence on RTMW3D, depth path got something usable).
        if let Some(rt) = rtmw_est.skeleton.left_hand_orientation {
            skeleton.left_hand_orientation = Some(rt);
        }
        if let Some(rt) = rtmw_est.skeleton.right_hand_orientation {
            skeleton.right_hand_orientation = Some(rt);
        }

        let dt_total = t_total.elapsed();
        debug!(
            "RTMW3D+DAv2 timing: total={:>5.1}ms rtmw={:>5.1}ms wait={:>4.1}ms calib={:>4.2}ms metric={:>4.1}ms skel={:>4.1}ms (depth_age={} frames, dav2_async={:.1}ms)",
            dt_total.as_secs_f32() * 1000.0,
            dt_rtmw.as_secs_f32() * 1000.0,
            dt_wait.as_secs_f32() * 1000.0,
            dt_calib.as_secs_f32() * 1000.0,
            dt_metric.as_secs_f32() * 1000.0,
            dt_skel.as_secs_f32() * 1000.0,
            depth_age,
            depth_result.inference_ms,
        );

        // Annotation reuses RTMW3D's bbox + 2D keypoints. Skeleton is
        // the new metric one.
        PoseEstimate {
            annotation: rtmw_est.annotation,
            skeleton,
        }
    }
}

fn empty_estimate(frame_index: u64) -> PoseEstimate {
    PoseEstimate {
        annotation: DetectionAnnotation {
            keypoints: Vec::new(),
            skeleton: Vec::new(),
            bounding_box: None,
        },
        skeleton: SourceSkeleton::empty(frame_index),
    }
}

/// Replace the elbow joints' source-space Z with one derived from
/// DAv2 metric depth, taken relative to the shoulder. See the call
/// site for why we do this only for the elbow (and not e.g. the
/// wrist or shoulder itself).
///
/// `shoulder_l_metric_x` and `shoulder_r_metric_x` are the shoulders'
/// metric-space X coordinates in metres, captured before Phase 7
/// overwrote them with RTMW3D's source-space coords.
///
/// Bails on:
/// - shoulder span too short to anchor a stable scale (< 1 mm metric
///   or < 1e-3 source units — both indicate a degenerate detection
///   where the conversion factor would be unstable);
/// - either shoulder or elbow with confidence below
///   `MIN_INJECT_CONFIDENCE` — RTMW3D is uncertain about the 2D
///   pixel, so the DAv2 sample is centred on a possibly-wrong
///   location and propagating it would amplify the error;
/// - elbow or shoulder lacking a `metric_depth_m` sample (DAv2
///   returned no finite pixels in the depth window);
/// - implied Z displacement exceeding the same anatomical band that
///   `apply_anatomical_z_clamps` enforces during the metric-space
///   build (see [`super::skeleton_from_depth::MAX_UPPER_ARM_DZ_M`]).
///   Without this guard we'd let DAv2 mis-samples (e.g. a hand from
///   the other arm crossing in front of the target elbow's pixel)
///   produce an elbow Z larger than what Phase 5 would have clamped,
///   contradicting the anatomical floor we already enforce upstream.
#[cfg(feature = "inference")]
fn inject_elbow_z_from_dav2(
    skeleton: &mut SourceSkeleton,
    shoulder_l_metric_x: f32,
    shoulder_r_metric_x: f32,
) {
    /// Confidence floor below which we don't trust the joint's 2D
    /// localisation enough to inject a depth sample for it. Above the
    /// codebase-wide `KEYPOINT_VISIBILITY_FLOOR` of 0.05 (which is the
    /// "model produced any output here" floor) but below the typical
    /// solver slider value, since the DAv2-derived Z is more
    /// fragile than direct 2D-only consumers — a few jittered pixels
    /// in the depth window can pull the median to a background depth.
    const MIN_INJECT_CONFIDENCE: f32 = 0.4;

    let metric_x_span = (shoulder_l_metric_x - shoulder_r_metric_x).abs();
    if metric_x_span < 1e-3 {
        return;
    }
    let source_x_span = match (
        skeleton.joints.get(&HumanoidBone::LeftUpperArm),
        skeleton.joints.get(&HumanoidBone::RightUpperArm),
    ) {
        (Some(l), Some(r)) => (l.position[0] - r.position[0]).abs(),
        _ => return,
    };
    if source_x_span < 1e-3 {
        return;
    }
    let metres_per_source_unit = metric_x_span / source_x_span;

    for (shoulder_bone, elbow_bone) in [
        (HumanoidBone::LeftUpperArm, HumanoidBone::LeftLowerArm),
        (HumanoidBone::RightUpperArm, HumanoidBone::RightLowerArm),
    ] {
        let (shoulder_z_m, shoulder_z_src) = match skeleton.joints.get(&shoulder_bone) {
            Some(s) if s.confidence >= MIN_INJECT_CONFIDENCE => match s.metric_depth_m {
                Some(z_m) => (z_m, s.position[2]),
                None => continue,
            },
            _ => continue,
        };
        let Some(elbow_joint_mut) = skeleton.joints.get_mut(&elbow_bone) else {
            continue;
        };
        if elbow_joint_mut.confidence < MIN_INJECT_CONFIDENCE {
            continue;
        }
        let Some(elbow_z_m) = elbow_joint_mut.metric_depth_m else {
            continue;
        };
        // Δ in metric: positive when the elbow is forward (closer to
        // the camera) than the shoulder. Source-z convention matches
        // `rtmw3d::skeleton::to_source` — +z is toward the camera —
        // so the metric Δ maps directly without a sign flip.
        let delta_metric = shoulder_z_m - elbow_z_m;
        if delta_metric.abs() > super::skeleton_from_depth::MAX_UPPER_ARM_DZ_M {
            continue;
        }
        let delta_source = delta_metric / metres_per_source_unit;
        elbow_joint_mut.position[2] = shoulder_z_src + delta_source;
    }
}

/// Plausibility band for an individual joint's metric depth, in
/// metres. When `pose_calibration.anchor_depth_m` is set the band is
/// narrower (subject's habitual distance ± `CALIBRATED_DEPTH_TOL_M`);
/// otherwise the static webcam-friendly range covers anywhere from
/// "right against the camera" to "across a small room".
///
/// Why the gate exists: at intermediate body yaws (~30°–60°) the rear
/// shoulder is partially occluded by the torso, and DAv2 frequently
/// samples background depth (3–10 m) at that pixel instead of the
/// shoulder surface. Without this guard `inject_shoulder_bz_from_dav2`
/// happily writes the resulting bogus `bz` into the source skeleton
/// and `compute_body_yaw_3d` reads it as a wild rotation. Bulk
/// validation across `mocap_suit_rotation` (16 frames, 22.5° steps)
/// shows 9/16 frames have one shoulder DAv2 sample landing on
/// background — the gate catches those without affecting the clean
/// frontal / cardinal poses where DAv2 succeeds.
#[cfg(feature = "inference")]
fn metric_depth_band(pose_calibration: Option<&crate::tracking::PoseCalibration>) -> (f32, f32) {
    /// Half-width of the plausibility band when the user has captured a
    /// `PoseCalibration` profile with `anchor_depth_m`. Allows the
    /// subject to lean ±60 cm toward / away from the calibrated seat
    /// without rejecting frames; beyond that we'd rather drop the
    /// metric Z than propagate a sample taken from the wall behind
    /// them.
    const CALIBRATED_DEPTH_TOL_M: f32 = 0.6;
    /// Static fallback when no calibration is active. 0.3 m is close
    /// enough to the camera that a webcam subject's body would already
    /// be partially out of frame; 5.0 m is past where DAv2 + a typical
    /// indoor scene retain useful depth contrast against the
    /// background. Anything outside this is by construction not the
    /// subject.
    const STATIC_MIN_M: f32 = 0.3;
    const STATIC_MAX_M: f32 = 5.0;

    if let Some(d) = pose_calibration.and_then(|c| c.anchor_depth_m) {
        if d.is_finite() && d > 0.0 {
            let lo = (d - CALIBRATED_DEPTH_TOL_M).max(STATIC_MIN_M);
            let hi = (d + CALIBRATED_DEPTH_TOL_M).min(STATIC_MAX_M);
            return (lo, hi);
        }
    }
    (STATIC_MIN_M, STATIC_MAX_M)
}

/// Replace the shoulder pair's source-space Z with one derived from
/// DAv2's inter-shoulder Δz. Phase 7's overwrite replaced both
/// shoulders' Z with RTMW3D's nz, which suffers from the same
/// coarse-softmax limitation as the elbow nz: at intermediate body
/// yaws (~30°–60°) the inter-shoulder Δz is small in nz units, and
/// `compute_body_yaw_3d`'s X-foreshortening recovery under-estimates
/// the rotation (validation set: 45° label → 26° recovered).
///
/// Mirror of `inject_elbow_z_from_dav2`: we use DAv2's bz signal as a
/// *relative* delta `Δ = R_metric − L_metric`, which cancels DAv2's
/// near-uniform lateral bias. Only the L↔R asymmetry is corrected;
/// the average source Z (= shoulder midpoint) is preserved so the
/// downstream elbow injection still anchors against the same midpoint
/// the spine-chain proxy logic uses.
///
/// Bails on the same conditions as the elbow injection, plus:
/// - either shoulder's `metric_depth_m` outside [`metric_depth_band`]
///   (DAv2 sampled background instead of the shoulder pixel);
/// - `|Δ| > MAX_BZ_METRIC_M` (anatomical L↔R Z spread bound);
/// - `|Δ| < MIN_BZ_METRIC_M` (dead-zone for DAv2's residual lateral-
///   bias DC offset — see the constant's docstring).
#[cfg(feature = "inference")]
fn inject_shoulder_bz_from_dav2(
    skeleton: &mut SourceSkeleton,
    shoulder_l_metric_x: f32,
    shoulder_r_metric_x: f32,
    pose_calibration: Option<&crate::tracking::PoseCalibration>,
) {
    /// See `inject_elbow_z_from_dav2`. Same threshold, same rationale —
    /// DAv2 windows on a low-confidence keypoint pixel are dominated by
    /// surrounding-region depth rather than the body part.
    const MIN_INJECT_CONFIDENCE: f32 = 0.4;

    /// Anatomical bound on `|R_z_m − L_z_m|`. At 90° body yaw the
    /// shoulder pair is fully separated along the camera Z axis so the
    /// max anatomical spread ≈ `ASSUMED_SHOULDER_SPAN_M` (0.40 m).
    /// Allow 25% margin for DAv2 sample noise / non-square anatomy
    /// before rejecting — beyond that the Δ is almost certainly a
    /// DAv2 outlier sampling something other than the shoulder
    /// (background, hair, the other shoulder occluding).
    const MAX_BZ_METRIC_M: f32 = 0.50;

    /// Lower bound below which DAv2's `bz_metric` is treated as
    /// indistinguishable from its residual lateral-bias DC offset.
    /// The bias-into-Δz is small (the elbow comment notes the bias is
    /// "roughly constant across nearby pixels") but not zero — bulk
    /// `mocap_suit_rotation` data shows ~0.10–0.16 m of phantom L↔R
    /// asymmetry on visually frontal poses. Without this dead-zone
    /// the injection writes that bias into the source bz, which
    /// inflates `span_xz` in `compute_body_yaw_3d`'s
    /// `target_span = running_max.max(span_xz)` formula, dragging
    /// `cos_yaw` below 1 and producing a phantom 15-25° rotation
    /// when the subject is square-on.
    ///
    /// 0.20 m chosen so a genuine ~12° rotation
    /// (`bz_metric ≈ sin(12°)·shoulder_span ≈ 0.083 m`) doesn't fire,
    /// but a ~22° rotation (`≈ 0.15 m`) — still under-shooting since
    /// the synthetic validation set's 22.5° label measured at 0.24 m —
    /// does. Below 12° the foreshortening recovery from `bx` alone
    /// already gives correct magnitude (the running max captures
    /// frontal width), and the small-magnitude sign jitter from the
    /// fall-back RTMW3D nz path is dominated by `rotation_blend` /
    /// 1€ smoothing downstream.
    const MIN_BZ_METRIC_M: f32 = 0.20;

    let metric_x_span = (shoulder_l_metric_x - shoulder_r_metric_x).abs();
    if metric_x_span < 1e-3 {
        return;
    }
    let source_x_span = match (
        skeleton.joints.get(&HumanoidBone::LeftUpperArm),
        skeleton.joints.get(&HumanoidBone::RightUpperArm),
    ) {
        (Some(l), Some(r)) => (l.position[0] - r.position[0]).abs(),
        _ => return,
    };
    if source_x_span < 1e-3 {
        return;
    }
    let metres_per_source_unit = metric_x_span / source_x_span;

    let l = skeleton.joints.get(&HumanoidBone::LeftUpperArm);
    let r = skeleton.joints.get(&HumanoidBone::RightUpperArm);
    let (Some(l), Some(r)) = (l, r) else {
        return;
    };
    if l.confidence < MIN_INJECT_CONFIDENCE || r.confidence < MIN_INJECT_CONFIDENCE {
        return;
    }
    let (Some(l_z_m), Some(r_z_m)) = (l.metric_depth_m, r.metric_depth_m) else {
        return;
    };
    let (lo, hi) = metric_depth_band(pose_calibration);
    if !(lo..=hi).contains(&l_z_m) || !(lo..=hi).contains(&r_z_m) {
        return;
    }
    let mid_z_src = (l.position[2] + r.position[2]) * 0.5;

    // Sign convention: source-z is `+z` toward the camera, DAv2
    // metric_depth_m is distance-from-camera. So "L closer to camera"
    // means `L_metric < R_metric` (smaller distance) and `L_src >
    // R_src` (larger source-z). The bz signal `compute_body_yaw_3d`
    // reads is `bz = L_src − R_src`, which has the same sign as
    // `(R_metric − L_metric)` — both positive when L is forward.
    let bz_metric = r_z_m - l_z_m;
    let abs_bz = bz_metric.abs();
    if !(MIN_BZ_METRIC_M..=MAX_BZ_METRIC_M).contains(&abs_bz) {
        return;
    }
    let bz_source_target = bz_metric / metres_per_source_unit;
    let half = bz_source_target * 0.5;
    let l_new = mid_z_src + half;
    let r_new = mid_z_src - half;

    // RTMW3D shares the same source-space sample for `LeftShoulder` /
    // `LeftUpperArm` (both come from COCO index 5, likewise 6 for the
    // right side). Update both so consumers reading either bone see
    // the corrected Z.
    for bone in [HumanoidBone::LeftShoulder, HumanoidBone::LeftUpperArm] {
        if let Some(j) = skeleton.joints.get_mut(&bone) {
            j.position[2] = l_new;
        }
    }
    for bone in [HumanoidBone::RightShoulder, HumanoidBone::RightUpperArm] {
        if let Some(j) = skeleton.joints.get_mut(&bone) {
            j.position[2] = r_new;
        }
    }
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

#[cfg(feature = "inference")]
pub struct Calibration {
    /// Scalar `c` such that `z_metric = c / d_raw` for any DAv2
    /// inverse-depth sample. Determined by the anchor pair.
    pub scale_c: f32,
    /// Half-tangent of horizontal FOV, used to convert pixel
    /// position to camera-space x: `x = (nx − 0.5) · 2·h_half · z`.
    pub h_half: f32,
    /// Half-tangent of vertical FOV (counterpart of h_half).
    pub v_half: f32,
    /// Diagnostic-only: which body anchor was used and its raw
    /// inverse-depth values. Lets the diagnostic binary print
    /// "shoulder span / hip span / fallback" without re-running the
    /// chooser logic.
    pub anchor_label: &'static str,
    pub d_raw_a: f32,
    pub d_raw_b: f32,
}

/// Per-anchor reason `calibrate_scale` rejected a candidate, surfaced
/// through [`CalibrationFailure`]. The function tries shoulders first,
/// then hips; if shoulders succeed, the hip slot stays `NotChecked`.
/// When both fail, the caller logs both reasons so the user can see
/// which gate is biting (low keypoint score vs depth-sample failure
/// vs plausibility rejection vs …) without enabling debug logs.
#[cfg(feature = "inference")]
#[derive(Debug, Clone, Copy)]
pub enum AnchorOutcome {
    /// Candidate not examined (the previous candidate already
    /// succeeded, or the joints buffer didn't include this index).
    NotChecked,
    /// One or both keypoints sat below `KEYPOINT_VISIBILITY_FLOOR`.
    LowKeypointScore { score_a: f32, score_b: f32 },
    /// 2D separation in normalised image coords below
    /// `MIN_ANCHOR_2D_SEPARATION` — too close together for the
    /// `c = D / √geom` solve to be numerically stable.
    PairTooClose2D { separation: f32 },
    /// DAv2 returned no usable inverse-depth at one of the keypoints
    /// (off-frame, or value at/below 1e-3 = "background asymptote").
    DepthSampleFailed { which: char },
    /// `geom` collapsed to ~0, usually because both samples landed on
    /// the same depth surface (desk in foreground covering the anchor).
    DegenerateGeometry { geom: f32 },
    /// Computed scale_c non-finite, ≤ 0, or outside `[MIN_C, MAX_C]`.
    ScaleOutOfRange { scale_c: f32 },
    /// Plausibility band derived from the user's saved
    /// `PoseCalibration` rejected this frame's predicted anchor depth.
    /// Most likely cause: subject is at a different distance than when
    /// they calibrated. Fix is to re-calibrate at the current seat /
    /// stance.
    PlausibilityRejected {
        predicted_depth: f32,
        cal_depth: f32,
        max_deviation: f32,
    },
}

#[cfg(feature = "inference")]
impl std::fmt::Display for AnchorOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnchorOutcome::NotChecked => write!(f, "not-checked"),
            AnchorOutcome::LowKeypointScore { score_a, score_b } => {
                write!(f, "low-score(a={:.2}, b={:.2})", score_a, score_b)
            }
            AnchorOutcome::PairTooClose2D { separation } => {
                write!(f, "pair-too-close-2d(sep={:.3})", separation)
            }
            AnchorOutcome::DepthSampleFailed { which } => {
                write!(f, "depth-sample-failed({})", which)
            }
            AnchorOutcome::DegenerateGeometry { geom } => {
                write!(f, "degenerate-geometry(geom={:.2e})", geom)
            }
            AnchorOutcome::ScaleOutOfRange { scale_c } => {
                write!(f, "scale-out-of-range(c={:.3})", scale_c)
            }
            AnchorOutcome::PlausibilityRejected {
                predicted_depth,
                cal_depth,
                max_deviation,
            } => write!(
                f,
                "plausibility-rejected(predicted={:.2}m vs cal={:.2}m, max-dev={:.2}m)",
                predicted_depth, cal_depth, max_deviation
            ),
        }
    }
}

/// All-anchor failure breakdown returned by [`calibrate_scale`] when
/// no candidate yielded a usable scale this frame. Constructed by the
/// loop body itself; outer callers should `Display` it directly into
/// their warning text.
#[cfg(feature = "inference")]
#[derive(Debug, Clone, Copy)]
pub struct CalibrationFailure {
    pub shoulder: AnchorOutcome,
    pub hip: AnchorOutcome,
}

#[cfg(feature = "inference")]
impl std::fmt::Display for CalibrationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "shoulder={}, hip={}", self.shoulder, self.hip)
    }
}

#[cfg(feature = "inference")]
pub fn calibrate_scale(
    joints: &[DecodedJoint2d],
    dav2: &DepthAnythingFrame,
    src_w: u32,
    src_h: u32,
    pose_calibration: Option<&crate::tracking::PoseCalibration>,
) -> Result<Calibration, CalibrationFailure> {
    let v_half = (ASSUMED_VERTICAL_FOV_DEG.to_radians() * 0.5).tan();
    let h_half = v_half * (src_w as f32 / src_h.max(1) as f32);

    // Anchor preference: shoulder pair (5, 6) → hip pair (11, 12).
    // Shoulders are visible in every framing CIGPose / RTMW3D handle
    // well and have tight inter-subject variance.
    let candidates: &[(usize, usize, &'static str, f32)] = &[
        (5, 6, "shoulder-span", ASSUMED_SHOULDER_SPAN_M),
        (11, 12, "hip-span", ASSUMED_HIP_SPAN_M),
    ];

    // 2D anchor-pair separation must clear this fraction of frame
    // diagonal before we trust the calibration. Side-profile views
    // collapse the shoulder pair (and often hip pair too) onto
    // near-overlapping pixels, where DAv2 returns the same depth for
    // both points and `geom` shrinks toward zero — `c = D / √geom`
    // then explodes, multiplying every metric coordinate by the
    // resulting huge scalar (we observed shoulder-Y ≈ 3.5 m for a
    // standing subject in a side-profile sweep). Requiring a minimum
    // 2D separation rejects these degenerate configurations and falls
    // through to the next anchor (or to "no calibration this frame",
    // which the provider handles by passing through RTMW3D's synthetic
    // z).
    //
    // 0.05 was too permissive: bulk diagnostic over 135 validation
    // images showed scale_c spans 1.1–8.4 (8× variation), with the
    // outliers all sitting at separations 0.05–0.07 (3/4-rotation
    // poses). 0.08 catches them and falls through to hip-span (or
    // none-this-frame, which the EMA below preserves).
    const MIN_ANCHOR_2D_SEPARATION: f32 = 0.08;

    // Calibrated `c` plausibility window. Webcam subjects sit
    // 0.5–4 m from camera; with shoulder span ≈ 0.4 m, `c = D/√geom`
    // typically lands 1–6. Outside [0.3, 15] is almost certainly the
    // degenerate-anchor pathology described above leaking through.
    const MIN_C: f32 = 0.3;
    const MAX_C: f32 = 15.0;

    // Outcome slots — one per candidate. We keep these so the caller
    // can see *which* gate killed each anchor instead of just "no
    // calibration this frame". The two indices are hardcoded against
    // `candidates` above for readability — adding a third candidate
    // would need another slot.
    let mut shoulder_outcome = AnchorOutcome::NotChecked;
    let mut hip_outcome = AnchorOutcome::NotChecked;

    for &(idx_a, idx_b, label, distance_m) in candidates {
        // Borrow the right slot for this candidate. The shoulder
        // candidate is `(5, 6, ...)`, hip is `(11, 12, ...)`; this
        // discriminator stays valid as long as the table above keeps
        // those COCO indices. A future "e.g. ear-pair fallback" anchor
        // would need a third slot.
        let outcome_slot = if idx_a == 5 {
            &mut shoulder_outcome
        } else {
            &mut hip_outcome
        };

        // Joints buffer is gated to `NUM_JOINTS` upstream so missing
        // is a logic bug, not a recoverable failure — but we keep a
        // belt-and-braces guard rather than panicking so a corrupted
        // upstream just shows up as `NotChecked` in the warning.
        let (a, b) = match (joints.get(idx_a), joints.get(idx_b)) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        if a.score < super::skeleton_from_depth::KEYPOINT_VISIBILITY_FLOOR
            || b.score < super::skeleton_from_depth::KEYPOINT_VISIBILITY_FLOOR
        {
            *outcome_slot = AnchorOutcome::LowKeypointScore {
                score_a: a.score,
                score_b: b.score,
            };
            continue;
        }

        // 2D separation guard. `(nx, ny)` are in normalised image
        // coords so the comparison is resolution-independent.
        let dnx = a.nx - b.nx;
        let dny = a.ny - b.ny;
        let pair_dist_2d = (dnx * dnx + dny * dny).sqrt();
        if pair_dist_2d < MIN_ANCHOR_2D_SEPARATION {
            *outcome_slot = AnchorOutcome::PairTooClose2D {
                separation: pair_dist_2d,
            };
            continue;
        }

        let d_a = match dav2.sample_inverse_depth(a.nx, a.ny, SAMPLE_RADIUS_PX) {
            Some(v) if v > 1e-3 => v,
            _ => {
                *outcome_slot = AnchorOutcome::DepthSampleFailed { which: 'a' };
                continue;
            }
        };
        let d_b = match dav2.sample_inverse_depth(b.nx, b.ny, SAMPLE_RADIUS_PX) {
            Some(v) if v > 1e-3 => v,
            _ => {
                *outcome_slot = AnchorOutcome::DepthSampleFailed { which: 'b' };
                continue;
            }
        };

        let inv_a = 1.0 / d_a;
        let inv_b = 1.0 / d_b;
        let ax = (a.nx - 0.5) * inv_a - (b.nx - 0.5) * inv_b;
        let ay = (a.ny - 0.5) * inv_a - (b.ny - 0.5) * inv_b;
        let bz = inv_a - inv_b;
        let geom = (2.0 * h_half * ax).powi(2) + (2.0 * v_half * ay).powi(2) + bz.powi(2);
        if !geom.is_finite() || geom < 1e-9 {
            *outcome_slot = AnchorOutcome::DegenerateGeometry { geom };
            continue;
        }
        let scale_c = distance_m / geom.sqrt();
        if !scale_c.is_finite() || scale_c <= 0.0 || !(MIN_C..=MAX_C).contains(&scale_c) {
            *outcome_slot = AnchorOutcome::ScaleOutOfRange { scale_c };
            continue;
        }

        // Dynamic plausibility narrowing from pose calibration. The
        // hardcoded MIN_C / MAX_C above only catches degenerate-anchor
        // failure (geom near zero); a desk in foreground typically
        // yields a `scale_c` that's *plausible in absolute terms* but
        // far off the user's actual standing distance. We compute the
        // anchor depth this `scale_c` would produce and reject
        // configurations whose predicted depth deviates more than
        // `max_deviation` from the captured calibration depth.
        //
        // `max_deviation` is the larger of:
        //   - 30 % of calibration depth (subject genuinely walks closer
        //     / further by up to a third without re-calibrating), or
        //   - 3 σ of measured per-frame depth jitter (noise floor for
        //     a still subject).
        // — whichever is bigger, so the band is generous on either
        // axis but doesn't open all the way back up to MIN_C / MAX_C.
        if let Some(cal) = pose_calibration {
            if let Some(cal_depth) = cal.anchor_depth_m {
                let avg_d_raw = (d_a + d_b) * 0.5;
                let predicted_depth = scale_c / avg_d_raw;
                let jitter = cal.anchor_depth_jitter_m.unwrap_or(0.05).max(0.05);
                let max_deviation = (jitter * 3.0).max(cal_depth * 0.30);
                if (predicted_depth - cal_depth).abs() > max_deviation {
                    *outcome_slot = AnchorOutcome::PlausibilityRejected {
                        predicted_depth,
                        cal_depth,
                        max_deviation,
                    };
                    continue;
                }
            }
        }

        return Ok(Calibration {
            scale_c,
            h_half,
            v_half,
            anchor_label: label,
            d_raw_a: d_a,
            d_raw_b: d_b,
        });
    }
    Err(CalibrationFailure {
        shoulder: shoulder_outcome,
        hip: hip_outcome,
    })
}

/// Back-project every DAv2 pixel into a metric `(x, y, z)` triple in
/// camera-space (x-right, y-down, z-forward) and pack into a
/// MoGeFrame-shaped point cloud so the shared skeleton helpers can
/// consume it. Pixels with non-positive raw depth (background /
/// out-of-domain) are stored as NaN — `sample_metric_point`'s window
/// median ignores them naturally.
#[cfg(feature = "inference")]
pub fn build_metric_frame_from_dav2(dav2: &DepthAnythingFrame, calib: &Calibration) -> MoGeFrame {
    let w = dav2.width;
    let h = dav2.height;
    let pixel_count = (w as usize) * (h as usize);
    let mut points_m = Vec::with_capacity(pixel_count);
    let inv_w = 1.0 / w.max(1) as f32;
    let inv_h = 1.0 / h.max(1) as f32;
    let two_h = 2.0 * calib.h_half;
    let two_v = 2.0 * calib.v_half;

    for py in 0..h {
        for px in 0..w {
            let idx = (py * w + px) as usize;
            let d_raw = dav2.relative[idx];
            if !d_raw.is_finite() || d_raw <= 1e-3 {
                points_m.push([f32::NAN, f32::NAN, f32::NAN]);
                continue;
            }
            let z = calib.scale_c / d_raw;
            // Pixel-centre normalisation matches the convention used
            // when sampling: the keypoint's `(nx, ny)` lands on a
            // pixel whose centre is at `(px + 0.5, py + 0.5)`.
            let nx = (px as f32 + 0.5) * inv_w;
            let ny = (py as f32 + 0.5) * inv_h;
            let x = (nx - 0.5) * two_h * z;
            let y = (ny - 0.5) * two_v * z;
            points_m.push([x, y, z]);
        }
    }

    MoGeFrame {
        width: w,
        height: h,
        points_m,
    }
}

#[cfg(all(test, feature = "inference"))]
mod elbow_z_injection_tests {
    use super::*;
    use crate::tracking::source_skeleton::SourceJoint;

    /// Build a minimal skeleton with both shoulders + the requested
    /// elbows populated. Source-X span between shoulders = 0.50 units;
    /// metric-X span = 0.40 m; conversion factor = 0.80 m/unit.
    fn skeleton_with_elbows(
        elbow_z_m_l: Option<f32>,
        elbow_z_m_r: Option<f32>,
        elbow_initial_src_z: f32,
    ) -> (SourceSkeleton, f32, f32) {
        skeleton_with_elbows_and_confidence(elbow_z_m_l, elbow_z_m_r, elbow_initial_src_z, 1.0)
    }

    fn skeleton_with_elbows_and_confidence(
        elbow_z_m_l: Option<f32>,
        elbow_z_m_r: Option<f32>,
        elbow_initial_src_z: f32,
        elbow_confidence: f32,
    ) -> (SourceSkeleton, f32, f32) {
        let sl_metric_x = 0.20;
        let sr_metric_x = -0.20;
        let mk_shoulder = |src_x: f32| SourceJoint {
            position: [src_x, 0.0, 0.0],
            confidence: 1.0,
            metric_depth_m: Some(1.00),
        };
        let mk_elbow = |z_m: f32| SourceJoint {
            position: [0.0, 0.0, elbow_initial_src_z],
            confidence: elbow_confidence,
            metric_depth_m: Some(z_m),
        };
        let mut sk = SourceSkeleton::empty(0);
        sk.joints.insert(HumanoidBone::LeftUpperArm, mk_shoulder(0.25));
        sk.joints
            .insert(HumanoidBone::RightUpperArm, mk_shoulder(-0.25));
        if let Some(z) = elbow_z_m_l {
            sk.joints.insert(HumanoidBone::LeftLowerArm, mk_elbow(z));
        }
        if let Some(z) = elbow_z_m_r {
            sk.joints.insert(HumanoidBone::RightLowerArm, mk_elbow(z));
        }
        (sk, sl_metric_x, sr_metric_x)
    }

    #[test]
    fn forward_elbow_gets_positive_source_z_delta_vs_shoulder() {
        // Left elbow 0.16 m forward of shoulder → expected source delta
        // 0.16 / 0.80 = 0.20.
        let (mut sk, sl, sr) = skeleton_with_elbows(Some(0.84), None, 0.0);
        inject_elbow_z_from_dav2(&mut sk, sl, sr);
        let z = sk.joints[&HumanoidBone::LeftLowerArm].position[2];
        assert!((z - 0.20).abs() < 1e-4, "got {z}");
    }

    #[test]
    fn backward_elbow_gets_negative_source_z_delta() {
        // Right elbow 0.10 m behind shoulder → expected delta -0.125.
        let (mut sk, sl, sr) = skeleton_with_elbows(None, Some(1.10), 0.0);
        inject_elbow_z_from_dav2(&mut sk, sl, sr);
        let z = sk.joints[&HumanoidBone::RightLowerArm].position[2];
        assert!((z + 0.125).abs() < 1e-4, "got {z}");
    }

    #[test]
    fn implausible_forward_displacement_is_rejected_and_keeps_source_z() {
        // Elbow 0.50 m forward of shoulder — past anatomical
        // MAX_UPPER_ARM_DZ_M of 0.45.
        let initial = 0.42;
        let (mut sk, sl, sr) = skeleton_with_elbows(Some(0.50), None, initial);
        inject_elbow_z_from_dav2(&mut sk, sl, sr);
        let z = sk.joints[&HumanoidBone::LeftLowerArm].position[2];
        assert!((z - initial).abs() < 1e-6, "got {z}");
    }

    #[test]
    fn missing_elbow_metric_depth_skips_silently() {
        let initial = 0.07;
        let (mut sk, sl, sr) = skeleton_with_elbows(None, None, initial);
        sk.joints.insert(
            HumanoidBone::LeftLowerArm,
            SourceJoint {
                position: [0.0, 0.0, initial],
                confidence: 1.0,
                metric_depth_m: None,
            },
        );
        inject_elbow_z_from_dav2(&mut sk, sl, sr);
        let z = sk.joints[&HumanoidBone::LeftLowerArm].position[2];
        assert!((z - initial).abs() < 1e-6);
    }

    #[test]
    fn degenerate_metric_shoulder_span_skips_silently() {
        let initial = 0.07;
        let (mut sk, _, _) = skeleton_with_elbows(Some(0.84), None, initial);
        inject_elbow_z_from_dav2(&mut sk, 0.0001, -0.0001);
        let z = sk.joints[&HumanoidBone::LeftLowerArm].position[2];
        assert!((z - initial).abs() < 1e-6, "got {z}");
    }

    #[test]
    fn degenerate_source_shoulder_span_skips_silently() {
        // Source-X span < 1mm — both shoulders coincide post-Phase-7.
        // Function must skip even though the metric span we pass in
        // looks fine.
        let initial = 0.07;
        let (mut sk, sl, sr) = skeleton_with_elbows(Some(0.84), None, initial);
        // Collapse the source-X span by overwriting both shoulders to
        // the same X.
        for bone in [HumanoidBone::LeftUpperArm, HumanoidBone::RightUpperArm] {
            if let Some(j) = sk.joints.get_mut(&bone) {
                j.position[0] = 0.10;
            }
        }
        inject_elbow_z_from_dav2(&mut sk, sl, sr);
        let z = sk.joints[&HumanoidBone::LeftLowerArm].position[2];
        assert!((z - initial).abs() < 1e-6, "got {z}");
    }

    #[test]
    fn low_elbow_confidence_skips_injection() {
        // Elbow has plausible metric depth but RTMW3D was uncertain
        // about its 2D pixel — DAv2 sample is centred on a maybe-wrong
        // location, so we should fall back to the RTMW3D nz value
        // instead of trusting the depth sample.
        let initial = 0.07;
        let (mut sk, sl, sr) =
            skeleton_with_elbows_and_confidence(Some(0.84), None, initial, 0.30);
        inject_elbow_z_from_dav2(&mut sk, sl, sr);
        let z = sk.joints[&HumanoidBone::LeftLowerArm].position[2];
        assert!((z - initial).abs() < 1e-6, "got {z}");
    }

    #[test]
    fn high_confidence_at_threshold_does_inject() {
        // Sanity boundary: confidence exactly at MIN_INJECT_CONFIDENCE
        // should pass (>=, not >). Otherwise small floor changes break
        // poses near the threshold.
        let (mut sk, sl, sr) =
            skeleton_with_elbows_and_confidence(Some(0.84), None, 0.0, 0.40);
        inject_elbow_z_from_dav2(&mut sk, sl, sr);
        let z = sk.joints[&HumanoidBone::LeftLowerArm].position[2];
        assert!((z - 0.20).abs() < 1e-4, "got {z}");
    }
}

#[cfg(all(test, feature = "inference"))]
mod shoulder_bz_injection_tests {
    use super::*;
    use crate::tracking::source_skeleton::SourceJoint;

    /// Source-X span = 0.50 units, metric-X span = 0.40 m,
    /// metres_per_source_unit = 0.80. Average shoulder source-Z = 0.04
    /// (a small RTMW3D-nz baseline) so tests can assert the mid is
    /// preserved while the L-R asymmetry shifts.
    fn skeleton_with_shoulders(
        l_metric_z: Option<f32>,
        r_metric_z: Option<f32>,
        confidence: f32,
    ) -> (SourceSkeleton, f32, f32) {
        let sl_metric_x = 0.20;
        let sr_metric_x = -0.20;
        let mk = |src_x: f32, z_m: Option<f32>| SourceJoint {
            position: [src_x, 0.0, 0.04],
            confidence,
            metric_depth_m: z_m,
        };
        let mut sk = SourceSkeleton::empty(0);
        sk.joints
            .insert(HumanoidBone::LeftShoulder, mk(0.25, l_metric_z));
        sk.joints
            .insert(HumanoidBone::LeftUpperArm, mk(0.25, l_metric_z));
        sk.joints
            .insert(HumanoidBone::RightShoulder, mk(-0.25, r_metric_z));
        sk.joints
            .insert(HumanoidBone::RightUpperArm, mk(-0.25, r_metric_z));
        (sk, sl_metric_x, sr_metric_x)
    }

    #[test]
    fn left_forward_yields_positive_l_minus_r_bz() {
        // L closer to camera by 0.16 m → bz_metric = R - L = +0.16 m,
        // bz_source_target = +0.20 source units, half = 0.10. Mid
        // (= 0.04) preserved, so L = 0.29, R = -0.21.
        let (mut sk, sl, sr) = skeleton_with_shoulders(Some(0.80), Some(1.20), 1.0);
        inject_shoulder_bz_from_dav2(&mut sk, sl, sr, None);
        let l_z = sk.joints[&HumanoidBone::LeftShoulder].position[2];
        let r_z = sk.joints[&HumanoidBone::RightShoulder].position[2];
        assert!((l_z - 0.29).abs() < 1e-4, "L got {l_z}");
        assert!((r_z + 0.21).abs() < 1e-4, "R got {r_z}");
        // bz must be positive (subject turned to her left).
        assert!(l_z - r_z > 0.0);
    }

    #[test]
    fn right_forward_yields_negative_l_minus_r_bz() {
        // R closer (= subject turned to her right) by 0.40 m → bz_src
        // = -0.50, half = -0.25, L = -0.21, R = 0.29.
        let (mut sk, sl, sr) = skeleton_with_shoulders(Some(1.20), Some(0.80), 1.0);
        inject_shoulder_bz_from_dav2(&mut sk, sl, sr, None);
        let l_z = sk.joints[&HumanoidBone::LeftShoulder].position[2];
        let r_z = sk.joints[&HumanoidBone::RightShoulder].position[2];
        assert!((l_z - (-0.21)).abs() < 1e-4, "L got {l_z}");
        assert!((r_z - 0.29).abs() < 1e-4, "R got {r_z}");
        assert!(l_z - r_z < 0.0);
    }

    #[test]
    fn upper_arm_alias_receives_same_z_as_shoulder() {
        // RTMW3D shares COCO 5/6 between LeftShoulder/LeftUpperArm —
        // both must end up with identical source-Z post-injection.
        let (mut sk, sl, sr) = skeleton_with_shoulders(Some(0.80), Some(1.20), 1.0);
        inject_shoulder_bz_from_dav2(&mut sk, sl, sr, None);
        let l_sh = sk.joints[&HumanoidBone::LeftShoulder].position[2];
        let l_ua = sk.joints[&HumanoidBone::LeftUpperArm].position[2];
        let r_sh = sk.joints[&HumanoidBone::RightShoulder].position[2];
        let r_ua = sk.joints[&HumanoidBone::RightUpperArm].position[2];
        assert!((l_sh - l_ua).abs() < 1e-9);
        assert!((r_sh - r_ua).abs() < 1e-9);
    }

    #[test]
    fn average_shoulder_z_is_preserved() {
        // The pre-existing mid (0.04) should survive — the elbow
        // injection that runs immediately after relies on it.
        let (mut sk, sl, sr) = skeleton_with_shoulders(Some(0.80), Some(1.20), 1.0);
        inject_shoulder_bz_from_dav2(&mut sk, sl, sr, None);
        let l_z = sk.joints[&HumanoidBone::LeftShoulder].position[2];
        let r_z = sk.joints[&HumanoidBone::RightShoulder].position[2];
        assert!(((l_z + r_z) * 0.5 - 0.04).abs() < 1e-6);
    }

    #[test]
    fn small_bz_metric_below_deadzone_is_not_injected() {
        // 0.16 m of inter-shoulder Δ is in the residual lateral-bias
        // band — DAv2 routinely emits this much "phantom asymmetry"
        // even on visually frontal poses. Without the dead-zone the
        // injection writes 0.16 m of source-Z bias, which inflates
        // `span_xz` in `compute_body_yaw_3d` and produces a phantom
        // ~20° rotation. With the dead-zone we keep the RTMW3D nz
        // value (= the pre-existing 0.04 source-Z) unchanged.
        let (mut sk, sl, sr) = skeleton_with_shoulders(Some(0.92), Some(1.08), 1.0);
        inject_shoulder_bz_from_dav2(&mut sk, sl, sr, None);
        let l_z = sk.joints[&HumanoidBone::LeftShoulder].position[2];
        let r_z = sk.joints[&HumanoidBone::RightShoulder].position[2];
        assert!((l_z - 0.04).abs() < 1e-6, "L got {l_z}");
        assert!((r_z - 0.04).abs() < 1e-6, "R got {r_z}");
    }

    #[test]
    fn implausible_spread_is_rejected_and_keeps_source_z() {
        // 0.60 m spread > MAX_BZ_METRIC_M (0.50) — almost certainly a
        // DAv2 sample landing on the wrong surface. Bail and let the
        // RTMW3D nz value through.
        let (mut sk, sl, sr) = skeleton_with_shoulders(Some(0.70), Some(1.30), 1.0);
        inject_shoulder_bz_from_dav2(&mut sk, sl, sr, None);
        let l_z = sk.joints[&HumanoidBone::LeftShoulder].position[2];
        let r_z = sk.joints[&HumanoidBone::RightShoulder].position[2];
        assert!((l_z - 0.04).abs() < 1e-6);
        assert!((r_z - 0.04).abs() < 1e-6);
    }

    #[test]
    fn missing_metric_depth_skips_silently() {
        let (mut sk, sl, sr) = skeleton_with_shoulders(None, Some(1.00), 1.0);
        inject_shoulder_bz_from_dav2(&mut sk, sl, sr, None);
        let l_z = sk.joints[&HumanoidBone::LeftShoulder].position[2];
        assert!((l_z - 0.04).abs() < 1e-6);
    }

    #[test]
    fn low_confidence_skips_injection() {
        // RTMW3D uncertain about the shoulder 2D pixel → DAv2 sample is
        // centred on the wrong location and propagating the bz would
        // amplify the error.
        let (mut sk, sl, sr) = skeleton_with_shoulders(Some(0.92), Some(1.08), 0.30);
        inject_shoulder_bz_from_dav2(&mut sk, sl, sr, None);
        let l_z = sk.joints[&HumanoidBone::LeftShoulder].position[2];
        assert!((l_z - 0.04).abs() < 1e-6);
    }

    #[test]
    fn confidence_at_threshold_does_inject() {
        let (mut sk, sl, sr) = skeleton_with_shoulders(Some(0.80), Some(1.20), 0.40);
        inject_shoulder_bz_from_dav2(&mut sk, sl, sr, None);
        let l_z = sk.joints[&HumanoidBone::LeftShoulder].position[2];
        assert!((l_z - 0.29).abs() < 1e-4, "L got {l_z}");
    }

    #[test]
    fn elbow_injection_anchors_against_corrected_shoulder() {
        // End-to-end ordering check: shoulder injection runs first, so
        // the elbow injection's `shoulder_z_src + delta_source` line
        // sees the corrected Z. With L_shoulder advanced from 0.04 →
        // 0.29 and L_elbow's metric depth identical to L_shoulder's
        // metric depth (= 0.80), the elbow Δ vs shoulder is 0, so the
        // elbow's final source-Z should equal the new shoulder Z
        // (0.29), not the original 0.04.
        let (mut sk, sl, sr) = skeleton_with_shoulders(Some(0.80), Some(1.20), 1.0);
        sk.joints.insert(
            HumanoidBone::LeftLowerArm,
            SourceJoint {
                position: [0.0, 0.0, 0.0],
                confidence: 1.0,
                metric_depth_m: Some(0.80),
            },
        );
        inject_shoulder_bz_from_dav2(&mut sk, sl, sr, None);
        inject_elbow_z_from_dav2(&mut sk, sl, sr);
        let elbow_z = sk.joints[&HumanoidBone::LeftLowerArm].position[2];
        assert!((elbow_z - 0.29).abs() < 1e-4, "elbow got {elbow_z}");
    }

    #[test]
    fn out_of_band_metric_depth_skips_silently() {
        // 8.0 m on the left shoulder is exactly the failure mode
        // observed on `mocap_suit_rotation` 22.5° / 67.5° frames where
        // DAv2 sampled the wall behind the subject instead of the
        // shoulder pixel. Without the band guard the bz delta still
        // looks small (both shoulders on the same wall) and the
        // injection writes a garbage offset onto the source-Z. With
        // the guard we bail and let the RTMW3D nz fallback through.
        let (mut sk, sl, sr) = skeleton_with_shoulders(Some(8.0), Some(1.08), 1.0);
        inject_shoulder_bz_from_dav2(&mut sk, sl, sr, None);
        let l_z = sk.joints[&HumanoidBone::LeftShoulder].position[2];
        let r_z = sk.joints[&HumanoidBone::RightShoulder].position[2];
        assert!((l_z - 0.04).abs() < 1e-6);
        assert!((r_z - 0.04).abs() < 1e-6);
    }

    #[test]
    fn calibrated_band_rejects_off_distance_subject() {
        // Calibrated to 1.8 m ± 0.6 m (band [1.2, 2.4]). Both
        // shoulders at 1.0 m — subject is much closer than calibrated
        // and the DAv2 sample probably hit the user's hand/face crossing
        // the shoulder ROI. Inject must bail.
        let cal = crate::tracking::PoseCalibration {
            mode: crate::tracking::CalibrationMode::FullBody,
            captured_at: String::new(),
            captured_at_unix: 0,
            frame_count: 1,
            anchor_x: 0.0,
            anchor_y: 0.0,
            anchor_depth_m: Some(1.8),
            confidence: 1.0,
            anchor_depth_jitter_m: None,
            shoulder_span_m: None,
            x_range_observed: None,
            z_range_observed: None,
            torso_depth_template: None,
        };
        let (mut sk, sl, sr) = skeleton_with_shoulders(Some(1.0), Some(1.0), 1.0);
        inject_shoulder_bz_from_dav2(&mut sk, sl, sr, Some(&cal));
        let l_z = sk.joints[&HumanoidBone::LeftShoulder].position[2];
        assert!((l_z - 0.04).abs() < 1e-6);
    }
}
