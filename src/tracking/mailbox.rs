//! Worker → GUI handoff mailbox and its payload types.
//!
//! The mailbox is split into four independent mutexes so the hot path
//! (worker → app pose handoff) doesn't contend with GUI-only state. Each
//! sub-mutex guards a self-contained slice of state and **no call site
//! acquires more than one of them at a time**, so deadlock by lock-order
//! inversion is impossible by construction.
//!
//! Atomicity trade-off: `snapshot` (pose + frame + annotation in one
//! struct) and `publish_estimate` (pose + frame + annotation in one
//! write) used to be one-lock atomic. They now span two locks (pose
//! and preview), so a reader that races a writer can observe the new
//! pose-side sequence with the previous frame for at most one
//! publish_estimate. The GUI consumers (`viewport.rs` camera wipe)
//! dedup on `sequence`, so a
//! torn read shows up as one extra "no-update" frame at worst — no
//! rendering corruption.
//!
//! Also carries the camera-preview payload types (`PreviewFrame`,
//! `DetectionAnnotation`) transported through the preview channel.
//! Everything public here is re-exported at the `crate::tracking` root;
//! the producer side lives in `worker.rs`.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::{PoseEstimate, SourceSkeleton, TrackingErrorLevel, TrackingSmoothingParams};

#[derive(Clone)]
pub struct TrackingMailbox {
    /// Hot path: worker writes per inference cycle, app reads every
    /// frame from `run_frame`. Keeping this isolated from the heavier
    /// `PreviewFrame` clone in the preview mailbox is the main reason
    /// for the split.
    pose: Arc<Mutex<PoseMailboxInner>>,
    /// GUI display: camera preview frame + detection annotation. Cloned per
    /// GUI tick. Larger payloads (RGB pixel buffers) so the lock is
    /// occasionally held a little longer, but never while the pose
    /// mutex is also held.
    preview: Arc<Mutex<PreviewMailboxInner>>,
    /// Low-frequency worker → GUI diagnostics: error toasts +
    /// inference backend label.
    diagnostics: Arc<Mutex<DiagnosticsMailboxInner>>,
    stale_timeout_nanos: u64,
}

struct PoseMailboxInner {
    latest_pose: Option<SourceSkeleton>,
    sequence: u64,
    /// Monotonic-clock instant of the last publish. `Instant`, not
    /// `SystemTime`: freshness is a process-local interval measurement,
    /// and a wall clock that gets NTP-stepped backwards would make a
    /// dead worker's sample report as eternally fresh (`saturating_sub`
    /// pinning the age at zero) — the avatar would freeze in the last
    /// pose instead of fading. `None` until the first publish.
    last_update: Option<Instant>,
}

struct PreviewMailboxInner {
    /// `Arc` so `snapshot()` hands the frame out by refcount instead of
    /// copying the RGB buffer (≈2.7 MB at 1280×720) under the lock on
    /// every GUI tick. GUI consumers should gate `snapshot()` behind
    /// [`TrackingMailbox::preview_sequence`] and only pull when it
    /// advanced — 60 Hz GUI × 30 fps camera used to clone-and-discard
    /// half the frames without ever reading them.
    latest_frame: Option<Arc<PreviewFrame>>,
    latest_annotation: Option<DetectionAnnotation>,
    /// Bumped on every preview write so GUI consumers can dedup
    /// texture uploads on a counter that strictly corresponds to
    /// frame freshness. The pose-mailbox `sequence` and this one
    /// advance together inside `publish_estimate` (under separate
    /// locks); a snapshot that races the writer can observe pose-
    /// `sequence == N` with preview `sequence == N-1`, in which case
    /// the GUI sees "old frame, new pose" — the right dedup key for
    /// preview consumers is *this* counter, not pose `sequence`.
    sequence: u64,
}

struct DiagnosticsMailboxInner {
    pending_error: Option<(String, TrackingErrorLevel)>,
    /// One-line label of the inference backend in use (e.g. "DirectML",
    /// "CPU", "CPU (DirectML unavailable: ...)"). `None` when no
    /// inference engine is loaded — e.g. synthetic mode, or before the
    /// worker has finished init.
    inference_backend_label: Option<String>,
}

/// Captured snapshot of the tracking mailbox state. Not strictly
/// cross-lock-atomic — see `TrackingMailbox::snapshot` for the
/// torn-read trade-off. `sequence` is the pose-side counter (use for
/// pose-driven dedup like solver consumption); `preview_sequence` is
/// the preview-side counter (use for frame / annotation upload dedup
/// — using `sequence` for frame uploads can permanently skip a frame
/// whenever the snapshot races a writer mid-publish).
#[derive(Clone, Debug, Default)]
pub struct MailboxSnapshot {
    pub pose: Option<SourceSkeleton>,
    /// Shared, not copied — cloning the snapshot bumps a refcount
    /// instead of duplicating the RGB buffer. Consumers should avoid
    /// calling [`TrackingMailbox::snapshot`] at all unless
    /// [`TrackingMailbox::preview_sequence`] advanced.
    pub frame: Option<Arc<PreviewFrame>>,
    pub annotation: Option<DetectionAnnotation>,
    pub sequence: u64,
    pub preview_sequence: u64,
}

impl TrackingMailbox {
    pub fn new() -> Self {
        Self {
            pose: Arc::new(Mutex::new(PoseMailboxInner {
                latest_pose: None,
                sequence: 0,
                last_update: None,
            })),
            preview: Arc::new(Mutex::new(PreviewMailboxInner {
                latest_frame: None,
                latest_annotation: None,
                sequence: 0,
            })),
            diagnostics: Arc::new(Mutex::new(DiagnosticsMailboxInner {
                pending_error: None,
                inference_backend_label: None,
            })),
            stale_timeout_nanos: TrackingSmoothingParams::default().stale_timeout_nanos,
        }
    }
}

impl Default for TrackingMailbox {
    fn default() -> Self {
        Self::new()
    }
}

impl TrackingMailbox {
    /// Thread-safe publish (pose-only path used by synthetic / test
    /// drivers that don't attach a preview frame).
    pub fn publish(&self, pose: SourceSkeleton) {
        let mut p = self.pose.lock().unwrap_or_else(|e| e.into_inner());
        p.latest_pose = Some(pose);
        p.sequence += 1;
        p.last_update = Some(Instant::now());
    }

    /// Publish a full estimation result including frame and annotations.
    /// Writes to two mutexes sequentially (pose, then preview) — see the
    /// module-level comment above `TrackingMailbox` for the cross-lock
    /// atomicity trade-off.
    pub fn publish_estimate(&self, estimate: PoseEstimate, frame: Option<PreviewFrame>) {
        {
            let mut p = self.pose.lock().unwrap_or_else(|e| e.into_inner());
            p.latest_pose = Some(estimate.skeleton);
            p.sequence += 1;
            p.last_update = Some(Instant::now());
        }
        // Wrap outside the lock: the one-time Arc allocation is the
        // publisher's cost; every reader clone afterwards is a refcount.
        let frame = frame.map(Arc::new);
        let mut v = self.preview.lock().unwrap_or_else(|e| e.into_inner());
        v.latest_annotation = Some(estimate.annotation);
        v.latest_frame = frame;
        v.sequence += 1;
    }

    /// Current preview-side sequence, read under the preview lock but
    /// without touching the payloads. GUI consumers poll this every
    /// tick and call [`Self::snapshot`] only when it advanced — the
    /// cheap gate that keeps a 60 Hz GUI from cloning a 30 fps
    /// camera's frames it will never upload.
    pub fn preview_sequence(&self) -> u64 {
        let v = self.preview.lock().unwrap_or_else(|e| e.into_inner());
        v.sequence
    }

    /// Snapshot the pose + preview slices. NOT cross-lock-atomic: a
    /// concurrent `publish_estimate` between the two lock acquisitions
    /// can produce a snapshot where `sequence` matches the new pose
    /// but `frame` / `annotation` are still from the previous publish.
    /// Both GUI consumers dedup on `sequence` and treat a momentary
    /// pose-newer-than-frame as "no update", so the worst-case is one
    /// skipped redraw rather than visible corruption.
    pub fn snapshot(&self) -> MailboxSnapshot {
        let (pose, sequence) = {
            let p = self.pose.lock().unwrap_or_else(|e| e.into_inner());
            (p.latest_pose.clone(), p.sequence)
        };
        let v = self.preview.lock().unwrap_or_else(|e| e.into_inner());
        MailboxSnapshot {
            pose,
            frame: v.latest_frame.clone(),
            annotation: v.latest_annotation.clone(),
            sequence,
            preview_sequence: v.sequence,
        }
    }

    /// Report a non-fatal error from the worker thread (shown as GUI toast).
    pub fn report_error(&self, msg: impl Into<String>, level: TrackingErrorLevel) {
        let mut d = self.diagnostics.lock().unwrap_or_else(|e| e.into_inner());
        d.pending_error = Some((msg.into(), level));
    }

    /// Drain the latest pending error (returns it only once).
    pub fn drain_error(&self) -> Option<(String, TrackingErrorLevel)> {
        let mut d = self.diagnostics.lock().unwrap_or_else(|e| e.into_inner());
        d.pending_error.take()
    }

    /// Set the inference-backend label. Called once by the worker after
    /// the pose provider finishes loading its model. `None` resets it
    /// (e.g. when tracking stops or the engine is destroyed).
    pub fn set_inference_backend_label(&self, label: Option<String>) {
        let mut d = self.diagnostics.lock().unwrap_or_else(|e| e.into_inner());
        d.inference_backend_label = label;
    }

    /// Read the inference-backend label. Cheap clone so the GUI thread
    /// can render without holding the lock.
    pub fn inference_backend_label(&self) -> Option<String> {
        let d = self.diagnostics.lock().unwrap_or_else(|e| e.into_inner());
        d.inference_backend_label.clone()
    }

    /// Thread-safe read: takes &self, locks the pose mutex, clones the pose.
    pub fn latest_pose(&self) -> Option<SourceSkeleton> {
        let p = self.pose.lock().unwrap_or_else(|e| e.into_inner());
        p.latest_pose.clone()
    }

    /// Read the latest camera preview frame (downscaled for GUI display).
    pub fn latest_frame(&self) -> Option<Arc<PreviewFrame>> {
        let v = self.preview.lock().unwrap_or_else(|e| e.into_inner());
        v.latest_frame.clone()
    }

    pub fn sequence(&self) -> u64 {
        let p = self.pose.lock().unwrap_or_else(|e| e.into_inner());
        p.sequence
    }

    /// Returns true if the latest sample is older than `stale_timeout_nanos`,
    /// or if no sample has ever been published.
    pub fn is_stale(&self) -> bool {
        let p = self.pose.lock().unwrap_or_else(|e| e.into_inner());
        match p.last_update {
            None => true,
            Some(at) => at.elapsed().as_nanos() as u64 > self.stale_timeout_nanos,
        }
    }

    /// Time elapsed since the last published sample. `None` when no
    /// sample has ever been published. The hold/fade policy in
    /// [`crate::app::Application::run_frame`] uses this to grade
    /// stale samples by age (fresh / holding / expired) instead of
    /// the binary stale flag.
    pub fn age(&self) -> Option<std::time::Duration> {
        let p = self.pose.lock().unwrap_or_else(|e| e.into_inner());
        p.last_update.map(|at| at.elapsed())
    }

    /// Stale-flip threshold the mailbox was constructed with. Surfaced
    /// so the hold/fade policy can ladder its windows from the same
    /// anchor instead of duplicating the constant.
    pub fn stale_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_nanos(self.stale_timeout_nanos)
    }
}

#[cfg(test)]
mod mailbox_tests {
    use super::*;

    fn empty_estimate() -> PoseEstimate {
        PoseEstimate {
            skeleton: SourceSkeleton::empty(0),
            annotation: DetectionAnnotation::default(),
        }
    }

    #[test]
    fn empty_mailbox_is_stale_and_has_no_age() {
        let mb = TrackingMailbox::new();
        assert!(mb.is_stale());
        assert_eq!(mb.age(), None);
        assert_eq!(mb.sequence(), 0);
        assert!(mb.latest_pose().is_none());
        assert!(mb.latest_frame().is_none());
    }

    #[test]
    fn publish_bumps_pose_sequence_only() {
        let mb = TrackingMailbox::new();
        let snap_before = mb.snapshot();
        mb.publish(SourceSkeleton::empty(0));
        let snap_after = mb.snapshot();
        assert_eq!(snap_after.sequence, snap_before.sequence + 1);
        assert_eq!(
            snap_after.preview_sequence, snap_before.preview_sequence,
            "publish (pose-only) must not touch preview_sequence"
        );
    }

    #[test]
    fn publish_estimate_bumps_both_sequences() {
        let mb = TrackingMailbox::new();
        mb.publish_estimate(empty_estimate(), None);
        let snap = mb.snapshot();
        assert_eq!(snap.sequence, 1);
        assert_eq!(snap.preview_sequence, 1);
    }

    #[test]
    fn publish_makes_mailbox_fresh_with_monotonic_age() {
        let mb = TrackingMailbox::new();
        mb.publish(SourceSkeleton::empty(0));
        assert!(!mb.is_stale(), "just-published sample must be fresh");
        let age = mb.age().expect("age is Some after a publish");
        assert!(
            age < std::time::Duration::from_secs(1),
            "age of a just-published sample must be near zero, got {age:?}"
        );
    }

    #[test]
    fn error_drain_and_backend_label_round_trip() {
        let mb = TrackingMailbox::new();
        mb.report_error("test failure", TrackingErrorLevel::Warning);
        let drained = mb.drain_error();
        assert!(drained.is_some());
        assert!(mb.drain_error().is_none(), "errors drain exactly once");
        mb.set_inference_backend_label(Some("CPU".to_string()));
        assert_eq!(mb.inference_backend_label(), Some("CPU".to_string()));
        mb.set_inference_backend_label(None);
        assert_eq!(mb.inference_backend_label(), None);
    }
}

// ---------------------------------------------------------------------------
// Camera preview frame & detection annotation (for GUI PIP wipe display)
// ---------------------------------------------------------------------------

/// A downscaled camera preview frame for the GUI camera-preview overlay.
#[derive(Clone, Debug)]
pub struct PreviewFrame {
    pub rgb_data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// 2D detection annotation overlaid on the camera preview.
#[derive(Clone, Debug, Default)]
pub struct DetectionAnnotation {
    /// Keypoints as (x, y, confidence) in normalised [0, 1] image coords.
    /// COCO-Wholebody layout: body 0..17, feet 17..23, face 23..91,
    /// left hand 91..112, right hand 112..133.
    pub keypoints: Vec<(f32, f32, f32)>,
    /// Skeleton line connections as pairs of keypoint indices.
    pub skeleton: Vec<(usize, usize)>,
    /// Bounding box (min_x, min_y, max_x, max_y) in normalised coords.
    pub bounding_box: Option<(f32, f32, f32, f32)>,
    /// Per-hand observability: the crop each hand landmarker ran on
    /// (normalised x, y, w, h) and the lock presence, `[left, right]`.
    /// `None` = no crop attempted / no lock this frame.
    pub hand_crops: [Option<HandCropDiag>; 2],
}

/// One hand-landmarker attempt, as surfaced on the preview wipe.
#[derive(Clone, Copy, Debug, Default)]
pub struct HandCropDiag {
    /// Crop rect in normalised frame coords (x, y, w, h).
    pub rect: (f32, f32, f32, f32),
    /// Landmarker presence for the best attempt on this crop (0..1).
    pub presence: f32,
}
