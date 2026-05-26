//! Runtime GPU budget policy (P3-03).
//!
//! Centralises the render / output / tracking cadence decisions that were
//! previously scattered as constants across `OutputRouter::set_target_fps`
//! and `tracking/rtmw3d/mod.rs::YOLOX_REFRESH_PERIOD`. Consumers read the
//! latest target from the budget; the budget reads live measurements
//! (render dt, output queue depth, export pool occupancy, recent GPU
//! export failures) and decides one global [`DegradedMode`] level that
//! drives the wired targets together.
//!
//! Currently wired:
//! - `render_fps_target` → `OutputRouter::set_target_fps` via
//!   `Application::set_user_render_fps`
//! - `yolox_skip_period` → `tracking::rtmw3d::YOLOX_REFRESH_PERIOD`
//!   (`pub static AtomicU64`)
//! - `degraded_mode == EmergencyCpu` → forces `RenderExportMode::CpuReadback`
//!   in `app/render.rs`, breaking the GPU-export → failure loop
//!
//! Not yet wired (would be follow-up work, with a real consumer in the
//! same commit so the budget doesn't accumulate dead policy outputs):
//! - pose worker Hz throttle
//! - depth refresh period
//! - FaceMesh ONNX EP preference (today the EP is hard-coded per
//!   provider: `Auto` for standalone rtmw3d, `ForceCpu` for the depth-
//!   colocated paths)
//!
//! See `plan/runtime-gpu-budget.md` for the design and threshold rationale.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Coarse GPU pressure level. Levels are intentionally few — fine-grained
/// auto-tuning produces oscillation more often than it produces wins.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DegradedMode {
    /// All targets follow user intent.
    #[default]
    Healthy,
    /// One signal is persistently elevated; cadence cuts are conservative.
    PressureLight,
    /// GPU saturation; aggressive cadence cuts (and inference EP swap).
    PressureHeavy,
    /// GPU export failing; fall back to CPU-only paths until recovery.
    EmergencyCpu,
}

impl DegradedMode {
    pub fn label(self) -> &'static str {
        match self {
            DegradedMode::Healthy => "Healthy",
            DegradedMode::PressureLight => "Pressure (light)",
            DegradedMode::PressureHeavy => "Pressure (heavy)",
            DegradedMode::EmergencyCpu => "Emergency CPU",
        }
    }
}

/// Snapshot of the inputs the budget uses for its decision. Constructed
/// each frame by `Application::update_runtime_gpu_budget`.
#[derive(Clone, Copy, Debug)]
pub struct RuntimeMeasurements {
    /// Smoothed render frame interval. EMA of `FrameConfig::frame_dt`.
    pub render_dt: Duration,
    /// Render target the user picked. Reconstructed from
    /// `OutputRouter::forward_min_interval` source of truth, or set
    /// directly via `RuntimeGpuBudget::set_user_render_fps`.
    pub render_target: Duration,
    /// Output drops per second over the most recent measurement window.
    /// `OutputRouter::dropped_count` delta divided by the window.
    pub output_drops_per_sec: f32,
    /// How many export pool slots are currently leased to sinks. If
    /// this equals capacity for any meaningful length of time, the
    /// renderer is starving on slot reclaim.
    pub export_pool_leased: usize,
    /// Export pool capacity (constant in production today, but plumbed
    /// from `OutputDiagnostics::export_pool` so the budget never holds
    /// a stale copy).
    pub export_pool_capacity: usize,
    /// Number of GPU export failures observed since the previous
    /// [`RuntimeGpuBudget::update`] call. Producer (`OutputRouter`)
    /// counts handle-export / fence-wait misses cumulatively; the
    /// caller drains via `take_gpu_export_failure_count` each tick and
    /// passes the delta here. The budget accumulates these deltas
    /// into its own time-windowed counter (see `FAILURE_WINDOW`) and
    /// compares the window total against `EMERGENCY_FAILURE_COUNT` —
    /// per-tick counts of 0 or 1 would otherwise never reach the
    /// threshold in normal operation.
    pub gpu_export_failures_this_tick: u32,
}

impl Default for RuntimeMeasurements {
    fn default() -> Self {
        Self {
            render_dt: Duration::from_secs_f32(1.0 / 60.0),
            render_target: Duration::from_secs_f32(1.0 / 60.0),
            output_drops_per_sec: 0.0,
            export_pool_leased: 0,
            export_pool_capacity: 2,
            gpu_export_failures_this_tick: 0,
        }
    }
}

/// Knobs published by the budget. Consumers read these each tick.
///
/// Not `Copy` — the failure-history ring is a `VecDeque`. `Application`
/// owns one instance by value; nothing in the codebase needs to copy it.
#[derive(Clone, Debug)]
pub struct RuntimeGpuBudget {
    /// User's intent for render cadence (combo box). The budget may
    /// clamp the *exposed* target below this under pressure.
    user_render_fps: u32,

    render_fps_target: u32,
    yolox_skip_period: u32,
    degraded_mode: DegradedMode,

    pressure_since: Option<Instant>,
    clean_streak_started: Option<Instant>,
    last_transition_at: Instant,
    last_transition_reason: TransitionReason,

    /// Timestamps of recent GPU export failures, pruned to the
    /// `FAILURE_WINDOW` rolling window on every `update`. Counted in
    /// the `PressureHeavy → EmergencyCpu` decision so a per-tick
    /// count of 0 or 1 (the normal case — at most one publish per
    /// frame) can still accumulate to the threshold.
    failure_history: VecDeque<Instant>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransitionReason {
    #[default]
    InitialState,
    RenderOverrun,
    OutputDrops,
    ExportPoolSaturated,
    Sustained,
    GpuExportFailures,
    Recovery,
}

impl TransitionReason {
    pub fn label(self) -> &'static str {
        match self {
            TransitionReason::InitialState => "initial state",
            TransitionReason::RenderOverrun => "render time exceeded target",
            TransitionReason::OutputDrops => "output dropping frames",
            TransitionReason::ExportPoolSaturated => "export pool saturated",
            TransitionReason::Sustained => "sustained pressure",
            TransitionReason::GpuExportFailures => "GPU export failures",
            TransitionReason::Recovery => "signals clean for 30s",
        }
    }
}

/// Hysteresis windows. Tuned after live data lands; values are the
/// round-numbered first cut.
const LIGHT_TO_HEAVY_DWELL: Duration = Duration::from_secs(5);
const RECOVERY_DWELL: Duration = Duration::from_secs(30);
const HEAVY_DROP_RATE_PER_SEC: f32 = 20.0;
const LIGHT_DROP_RATE_PER_SEC: f32 = 5.0;
const RENDER_OVERRUN_MULTIPLE: f32 = 1.2;
/// Failures within this rolling window count toward the
/// `PressureHeavy → EmergencyCpu` escalation.
const FAILURE_WINDOW: Duration = Duration::from_secs(10);
const EMERGENCY_FAILURE_COUNT: usize = 2;
/// Cap on history entries to bound memory; at 60 fps with one failure
/// per frame, `FAILURE_WINDOW * 60 = 600` is the theoretical maximum.
/// A higher cap protects against a runaway producer; pruning happens
/// each tick regardless.
const MAX_FAILURE_HISTORY: usize = 1024;

impl RuntimeGpuBudget {
    pub fn new(now: Instant) -> Self {
        Self {
            user_render_fps: 60,
            render_fps_target: 60,
            yolox_skip_period: 4,
            degraded_mode: DegradedMode::Healthy,
            pressure_since: None,
            clean_streak_started: Some(now),
            last_transition_at: now,
            last_transition_reason: TransitionReason::InitialState,
            failure_history: VecDeque::new(),
        }
    }

    /// Currently-windowed GPU export failure count. Exposed for
    /// inspector diagnostics; the escalation logic uses it directly
    /// inside `update`.
    pub fn gpu_export_failure_window_count(&self) -> usize {
        self.failure_history.len()
    }

    pub fn set_user_render_fps(&mut self, fps: u32) {
        self.user_render_fps = fps.max(1);
        self.recompute_targets();
    }

    pub fn render_fps_target(&self) -> u32 {
        self.render_fps_target
    }
    pub fn yolox_skip_period(&self) -> u32 {
        self.yolox_skip_period
    }
    pub fn degraded_mode(&self) -> DegradedMode {
        self.degraded_mode
    }
    pub fn user_render_fps(&self) -> u32 {
        self.user_render_fps
    }
    pub fn last_transition_reason(&self) -> TransitionReason {
        self.last_transition_reason
    }

    pub fn update(&mut self, m: &RuntimeMeasurements, now: Instant) {
        // Clock-backward defence: `Instant` is monotonic on most
        // platforms but some embed Instant in a wrapper that can be
        // adjusted (or `now` might be supplied from a test). If either
        // dwell-tracking timestamp lies in the future relative to `now`,
        // reset it — `duration_since` saturates to zero in that case,
        // which would otherwise freeze the state machine indefinitely
        // because no dwell ever appears to elapse.
        if let Some(s) = self.pressure_since {
            if s > now {
                self.pressure_since = Some(now);
            }
        }
        if let Some(s) = self.clean_streak_started {
            if s > now {
                self.clean_streak_started = Some(now);
            }
        }

        // Drop history entries that are either older than the window
        // (normal pruning) or in the future relative to `now` (clock
        // adjustment / test injection). Both cases would otherwise
        // produce stale "recent failures" counts.
        self.failure_history
            .retain(|t| *t <= now && now.duration_since(*t) <= FAILURE_WINDOW);
        // Record the failures this tick at `now`. Saturate the per-tick
        // count so a pathological producer can't exhaust the bound in
        // one call, then trim from the front to keep `MAX_FAILURE_HISTORY`.
        let to_push = (m.gpu_export_failures_this_tick as usize).min(MAX_FAILURE_HISTORY);
        for _ in 0..to_push {
            self.failure_history.push_back(now);
        }
        while self.failure_history.len() > MAX_FAILURE_HISTORY {
            self.failure_history.pop_front();
        }

        let pressure = self.detect_pressure(m);

        match (self.degraded_mode, pressure) {
            (DegradedMode::Healthy, Some(reason)) => {
                self.transition_to(DegradedMode::PressureLight, now, reason);
            }
            (DegradedMode::PressureLight, Some(reason)) => {
                let dwelled = self
                    .pressure_since
                    .map(|s| now.duration_since(s) > LIGHT_TO_HEAVY_DWELL)
                    .unwrap_or(false);
                if dwelled || m.output_drops_per_sec > HEAVY_DROP_RATE_PER_SEC {
                    self.transition_to(
                        DegradedMode::PressureHeavy,
                        now,
                        if m.output_drops_per_sec > HEAVY_DROP_RATE_PER_SEC {
                            TransitionReason::OutputDrops
                        } else {
                            TransitionReason::Sustained
                        },
                    );
                } else {
                    self.last_transition_reason = reason;
                }
            }
            (DegradedMode::PressureHeavy, p) => {
                if self.failure_history.len() >= EMERGENCY_FAILURE_COUNT {
                    self.transition_to(
                        DegradedMode::EmergencyCpu,
                        now,
                        TransitionReason::GpuExportFailures,
                    );
                } else if p.is_none() {
                    self.maybe_recover_one_level(now);
                }
            }
            (DegradedMode::EmergencyCpu, p) => {
                // Emergency recovery also requires the failure window
                // to drop below the trigger threshold. Without this,
                // pressure-clear + persistent GPU export failures would
                // oscillate Emergency → Heavy → Emergency: recovery
                // fires on `p.is_none()`, then the next update sees
                // `failure_history.len() >= EMERGENCY_FAILURE_COUNT`
                // and re-escalates. The 30 s dwell already exceeds the
                // 10 s `FAILURE_WINDOW`, so by the time recovery is
                // legitimately due, expired failures have been pruned;
                // anything still in the window is "new failures since
                // recovery started" and is the right reason to stay.
                if p.is_none() && self.failure_history.len() < EMERGENCY_FAILURE_COUNT {
                    self.maybe_recover_one_level(now);
                }
            }
            (_, None) => {
                // Healthy + None → step_down is a no-op; PressureLight
                // + None → step down to Healthy after the dwell.
                self.maybe_recover_one_level(now);
            }
        }

        if pressure.is_some() {
            if self.pressure_since.is_none() {
                self.pressure_since = Some(now);
            }
            self.clean_streak_started = None;
        } else {
            self.pressure_since = None;
            if self.clean_streak_started.is_none() {
                self.clean_streak_started = Some(now);
            }
        }

        self.recompute_targets();
    }

    fn detect_pressure(&self, m: &RuntimeMeasurements) -> Option<TransitionReason> {
        let target = m.render_target.as_secs_f32().max(1e-6);
        let render_overrun = m.render_dt.as_secs_f32() > RENDER_OVERRUN_MULTIPLE * target;
        let high_drops = m.output_drops_per_sec > LIGHT_DROP_RATE_PER_SEC;
        let pool_saturated =
            m.export_pool_capacity > 0 && m.export_pool_leased >= m.export_pool_capacity;
        if pool_saturated {
            Some(TransitionReason::ExportPoolSaturated)
        } else if render_overrun {
            Some(TransitionReason::RenderOverrun)
        } else if high_drops {
            Some(TransitionReason::OutputDrops)
        } else {
            None
        }
    }

    fn transition_to(&mut self, mode: DegradedMode, now: Instant, reason: TransitionReason) {
        log::info!(
            "RuntimeGpuBudget: {:?} -> {:?} ({})",
            self.degraded_mode,
            mode,
            reason.label()
        );
        self.degraded_mode = mode;
        self.last_transition_at = now;
        self.last_transition_reason = reason;
        // Force the recovery streak to restart after any transition so
        // each step-down (Heavy → Light → Healthy) requires its own
        // RECOVERY_DWELL of clean signals rather than cascading in two
        // consecutive updates. The post-match streak-state block at
        // the bottom of `update` re-arms it when pressure is clear.
        self.clean_streak_started = None;
    }

    /// Recovery transition: drop one level toward Healthy when the
    /// clean streak has run for at least `RECOVERY_DWELL`. Called
    /// from the `update` match arms that handle `pressure.is_none()`;
    /// a no-op when no streak is active, the dwell hasn't elapsed,
    /// or we are already at Healthy.
    fn maybe_recover_one_level(&mut self, now: Instant) {
        let Some(streak) = self.clean_streak_started else {
            return;
        };
        if now.duration_since(streak) <= RECOVERY_DWELL {
            return;
        }
        let next = match self.degraded_mode {
            DegradedMode::EmergencyCpu => DegradedMode::PressureHeavy,
            DegradedMode::PressureHeavy => DegradedMode::PressureLight,
            DegradedMode::PressureLight => DegradedMode::Healthy,
            DegradedMode::Healthy => return,
        };
        self.transition_to(next, now, TransitionReason::Recovery);
    }

    fn recompute_targets(&mut self) {
        match self.degraded_mode {
            DegradedMode::Healthy => {
                self.render_fps_target = self.user_render_fps;
                self.yolox_skip_period = 4;
            }
            DegradedMode::PressureLight => {
                self.render_fps_target = self.user_render_fps.min(45);
                self.yolox_skip_period = 6;
            }
            DegradedMode::PressureHeavy => {
                self.render_fps_target = self.user_render_fps.min(30);
                self.yolox_skip_period = 8;
            }
            DegradedMode::EmergencyCpu => {
                self.render_fps_target = self.user_render_fps.min(30);
                self.yolox_skip_period = 12;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy_measurements() -> RuntimeMeasurements {
        RuntimeMeasurements::default()
    }

    fn render_overrun_measurements() -> RuntimeMeasurements {
        let mut m = RuntimeMeasurements::default();
        m.render_dt = Duration::from_secs_f32(1.0 / 60.0 * 1.5);
        m
    }

    fn pool_saturated_measurements() -> RuntimeMeasurements {
        let mut m = RuntimeMeasurements::default();
        m.export_pool_leased = m.export_pool_capacity;
        m
    }

    fn heavy_drop_measurements() -> RuntimeMeasurements {
        let mut m = RuntimeMeasurements::default();
        m.output_drops_per_sec = HEAVY_DROP_RATE_PER_SEC + 1.0;
        m
    }

    #[test]
    fn healthy_stays_healthy_under_no_pressure() {
        let t0 = Instant::now();
        let mut budget = RuntimeGpuBudget::new(t0);
        budget.update(&healthy_measurements(), t0 + Duration::from_millis(16));
        assert_eq!(budget.degraded_mode(), DegradedMode::Healthy);
        assert_eq!(budget.render_fps_target(), 60);
    }

    #[test]
    fn first_pressure_signal_triggers_light_pressure() {
        let t0 = Instant::now();
        let mut budget = RuntimeGpuBudget::new(t0);
        budget.update(&render_overrun_measurements(), t0 + Duration::from_millis(16));
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureLight);
        assert_eq!(
            budget.render_fps_target(),
            45,
            "PressureLight clamps render target to 45 fps"
        );
    }

    #[test]
    fn export_pool_saturation_is_a_pressure_signal() {
        let t0 = Instant::now();
        let mut budget = RuntimeGpuBudget::new(t0);
        budget.update(&pool_saturated_measurements(), t0 + Duration::from_millis(16));
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureLight);
        assert_eq!(
            budget.last_transition_reason(),
            TransitionReason::ExportPoolSaturated
        );
    }

    #[test]
    fn sustained_pressure_escalates_to_heavy_after_dwell() {
        let t0 = Instant::now();
        let mut budget = RuntimeGpuBudget::new(t0);
        budget.update(&render_overrun_measurements(), t0);
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureLight);
        // Just under the dwell window: still PressureLight.
        budget.update(
            &render_overrun_measurements(),
            t0 + LIGHT_TO_HEAVY_DWELL - Duration::from_millis(1),
        );
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureLight);
        // Past the dwell window: escalates.
        budget.update(
            &render_overrun_measurements(),
            t0 + LIGHT_TO_HEAVY_DWELL + Duration::from_millis(1),
        );
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureHeavy);
        assert_eq!(budget.render_fps_target(), 30);
    }

    #[test]
    fn heavy_drop_rate_bypasses_dwell_into_heavy() {
        let t0 = Instant::now();
        let mut budget = RuntimeGpuBudget::new(t0);
        budget.update(&render_overrun_measurements(), t0);
        // Immediately spike drops above the heavy threshold — no dwell needed.
        budget.update(&heavy_drop_measurements(), t0 + Duration::from_millis(16));
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureHeavy);
        assert_eq!(budget.last_transition_reason(), TransitionReason::OutputDrops);
    }

    #[test]
    fn repeated_gpu_export_failures_escalate_to_emergency() {
        let t0 = Instant::now();
        let mut budget = RuntimeGpuBudget::new(t0);
        // First clean transition Healthy → PressureLight.
        budget.update(&heavy_drop_measurements(), t0);
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureLight);
        // Heavy drop rate from PressureLight escalates immediately (no dwell).
        budget.update(&heavy_drop_measurements(), t0 + Duration::from_millis(16));
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureHeavy);
        // Two GPU export failures in a single tick bumps to Emergency.
        let mut m = heavy_drop_measurements();
        m.gpu_export_failures_this_tick = EMERGENCY_FAILURE_COUNT as u32;
        budget.update(&m, t0 + Duration::from_millis(32));
        assert_eq!(budget.degraded_mode(), DegradedMode::EmergencyCpu);
        assert_eq!(budget.yolox_skip_period(), 12);
    }

    /// Failures arriving one-per-tick still accumulate to the
    /// threshold because the budget keeps a sliding `FAILURE_WINDOW`
    /// history. This is the real-world path: `OutputRouter` increments
    /// its counter by at most 1 per publish (one publish per frame),
    /// so the previous per-tick threshold of 2 was effectively
    /// unreachable.
    #[test]
    fn failures_one_per_tick_still_reach_emergency_within_window() {
        let t0 = Instant::now();
        let mut budget = RuntimeGpuBudget::new(t0);
        budget.update(&heavy_drop_measurements(), t0);
        budget.update(&heavy_drop_measurements(), t0 + Duration::from_millis(16));
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureHeavy);

        let mut m = heavy_drop_measurements();
        m.gpu_export_failures_this_tick = 1;
        budget.update(&m, t0 + Duration::from_millis(100));
        assert_eq!(
            budget.degraded_mode(),
            DegradedMode::PressureHeavy,
            "one failure within the window is below threshold"
        );
        budget.update(&m, t0 + Duration::from_millis(200));
        assert_eq!(
            budget.degraded_mode(),
            DegradedMode::EmergencyCpu,
            "second failure inside the window escalates"
        );
    }

    /// EmergencyCpu does not recover while the failure window is
    /// still above the trigger threshold, even after the dwell
    /// elapses. Prevents Emergency → Heavy → Emergency oscillation
    /// when GPU export keeps failing under otherwise-clean pressure.
    #[test]
    fn emergency_cpu_does_not_recover_while_failures_persist() {
        let t0 = Instant::now();
        let mut budget = RuntimeGpuBudget::new(t0);
        budget.update(&heavy_drop_measurements(), t0);
        budget.update(&heavy_drop_measurements(), t0 + Duration::from_millis(16));
        let mut m = heavy_drop_measurements();
        m.gpu_export_failures_this_tick = EMERGENCY_FAILURE_COUNT as u32;
        budget.update(&m, t0 + Duration::from_millis(32));
        assert_eq!(budget.degraded_mode(), DegradedMode::EmergencyCpu);

        // Pressure clears. Most of the dwell is clean, but right at
        // the recovery boundary a fresh burst of failures arrives —
        // enough to keep `failure_history` at or above the trigger
        // threshold. Recovery must hold off until that burst expires.
        let streak_start = t0 + Duration::from_millis(100);
        budget.update(&healthy_measurements(), streak_start);
        let mut fresh_burst = healthy_measurements();
        fresh_burst.gpu_export_failures_this_tick = EMERGENCY_FAILURE_COUNT as u32;
        budget.update(
            &fresh_burst,
            streak_start + RECOVERY_DWELL + Duration::from_millis(1),
        );
        assert_eq!(
            budget.degraded_mode(),
            DegradedMode::EmergencyCpu,
            "Emergency must not recover while failure_history >= threshold"
        );

        // Once the burst ages out of the window, recovery proceeds —
        // proves the gate is conditional on the failure window, not
        // a permanent block.
        budget.update(
            &healthy_measurements(),
            streak_start + RECOVERY_DWELL + FAILURE_WINDOW + Duration::from_millis(2),
        );
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureHeavy);
    }

    /// PressureHeavy steps down to PressureLight after a full
    /// RECOVERY_DWELL of clean signals. Before the recovery fix the
    /// match arm `(PressureHeavy, _)` swallowed the pressure-clear
    /// case, leaving Heavy permanently stuck.
    #[test]
    fn pressure_heavy_recovers_to_light_after_clean_dwell() {
        let t0 = Instant::now();
        let mut budget = RuntimeGpuBudget::new(t0);
        budget.update(&heavy_drop_measurements(), t0);
        budget.update(&heavy_drop_measurements(), t0 + Duration::from_millis(16));
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureHeavy);

        let streak_started = t0 + Duration::from_millis(100);
        budget.update(&healthy_measurements(), streak_started);
        budget.update(
            &healthy_measurements(),
            streak_started + RECOVERY_DWELL + Duration::from_millis(1),
        );
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureLight);
    }

    /// EmergencyCpu steps down to PressureHeavy after a full
    /// RECOVERY_DWELL of clean signals. Symmetric to the Heavy → Light
    /// case; without the recovery fix EmergencyCpu was a permanent
    /// trap once entered.
    #[test]
    fn emergency_cpu_recovers_to_heavy_after_clean_dwell() {
        let t0 = Instant::now();
        let mut budget = RuntimeGpuBudget::new(t0);
        // Force into EmergencyCpu the same way the unit-tested
        // escalation path does.
        budget.update(&heavy_drop_measurements(), t0);
        budget.update(&heavy_drop_measurements(), t0 + Duration::from_millis(16));
        let mut m = heavy_drop_measurements();
        m.gpu_export_failures_this_tick = EMERGENCY_FAILURE_COUNT as u32;
        budget.update(&m, t0 + Duration::from_millis(32));
        assert_eq!(budget.degraded_mode(), DegradedMode::EmergencyCpu);

        let streak_started = t0 + Duration::from_millis(100);
        budget.update(&healthy_measurements(), streak_started);
        budget.update(
            &healthy_measurements(),
            streak_started + RECOVERY_DWELL + Duration::from_millis(1),
        );
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureHeavy);
    }

    /// Recovery is one level per dwell, not a cascade. After Heavy
    /// steps down to Light the streak must restart so a second
    /// `RECOVERY_DWELL` is required before Light → Healthy. Without
    /// this, two consecutive updates a few ms apart could carry the
    /// budget from Heavy all the way to Healthy.
    #[test]
    fn recovery_cascades_one_level_per_dwell_not_instant() {
        let t0 = Instant::now();
        let mut budget = RuntimeGpuBudget::new(t0);
        budget.update(&heavy_drop_measurements(), t0);
        budget.update(&heavy_drop_measurements(), t0 + Duration::from_millis(16));
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureHeavy);

        let streak_start = t0 + Duration::from_millis(100);
        budget.update(&healthy_measurements(), streak_start);
        // First recovery: Heavy → Light at streak + RECOVERY_DWELL + ε.
        let first_recovery = streak_start + RECOVERY_DWELL + Duration::from_millis(1);
        budget.update(&healthy_measurements(), first_recovery);
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureLight);

        // Immediately afterwards, the streak must have been reset:
        // a clean update a few ms later should not skip straight to
        // Healthy.
        budget.update(
            &healthy_measurements(),
            first_recovery + Duration::from_millis(50),
        );
        assert_eq!(
            budget.degraded_mode(),
            DegradedMode::PressureLight,
            "second step-down must wait for its own dwell, not cascade"
        );

        // Wait a full RECOVERY_DWELL from the first recovery — second
        // step Light → Healthy fires.
        budget.update(
            &healthy_measurements(),
            first_recovery + RECOVERY_DWELL + Duration::from_millis(2),
        );
        assert_eq!(budget.degraded_mode(), DegradedMode::Healthy);
    }

    /// Failures separated by more than the window do not escalate —
    /// the older entry is pruned before the newer one lands.
    #[test]
    fn failures_outside_window_do_not_accumulate() {
        let t0 = Instant::now();
        let mut budget = RuntimeGpuBudget::new(t0);
        budget.update(&heavy_drop_measurements(), t0);
        budget.update(&heavy_drop_measurements(), t0 + Duration::from_millis(16));
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureHeavy);

        let mut m = heavy_drop_measurements();
        m.gpu_export_failures_this_tick = 1;
        budget.update(&m, t0 + Duration::from_secs(1));
        assert_eq!(budget.gpu_export_failure_window_count(), 1);
        // 11 seconds after the first failure — the first entry is now
        // outside FAILURE_WINDOW and gets pruned before the second is
        // pushed. The window count never reaches 2.
        budget.update(&m, t0 + Duration::from_secs(12));
        assert_eq!(budget.gpu_export_failure_window_count(), 1);
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureHeavy);
    }

    #[test]
    fn clean_signals_for_30s_step_down_one_level() {
        let t0 = Instant::now();
        let mut budget = RuntimeGpuBudget::new(t0);
        budget.update(&render_overrun_measurements(), t0);
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureLight);

        // First clean call starts the recovery streak; nothing changes yet
        // (recovery is measured from when pressure cleared, not from when
        // it began).
        let streak_started = t0 + Duration::from_millis(100);
        budget.update(&healthy_measurements(), streak_started);
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureLight);
        // Just under RECOVERY_DWELL elapsed since the streak started:
        // still PressureLight.
        budget.update(
            &healthy_measurements(),
            streak_started + RECOVERY_DWELL - Duration::from_millis(1),
        );
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureLight);
        // Past the dwell window: one level down.
        budget.update(
            &healthy_measurements(),
            streak_started + RECOVERY_DWELL + Duration::from_millis(1),
        );
        assert_eq!(budget.degraded_mode(), DegradedMode::Healthy);
    }

    #[test]
    fn recovery_resets_clean_streak_when_pressure_returns() {
        let t0 = Instant::now();
        let mut budget = RuntimeGpuBudget::new(t0);
        budget.update(&render_overrun_measurements(), t0);
        // Halfway through recovery, pressure returns.
        budget.update(
            &healthy_measurements(),
            t0 + Duration::from_secs(15),
        );
        budget.update(
            &render_overrun_measurements(),
            t0 + Duration::from_secs(20),
        );
        // Another 30s pass — but the clean streak restarted at t+20s so
        // recovery hasn't fully completed yet.
        budget.update(
            &healthy_measurements(),
            t0 + Duration::from_secs(30),
        );
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureLight);
    }

    #[test]
    fn user_render_fps_change_is_visible_immediately() {
        let t0 = Instant::now();
        let mut budget = RuntimeGpuBudget::new(t0);
        budget.set_user_render_fps(120);
        assert_eq!(budget.render_fps_target(), 120);
        // Under pressure the clamp wins.
        budget.update(&render_overrun_measurements(), t0);
        assert_eq!(budget.render_fps_target(), 45);
        // User can still raise their intent; the clamp tracks it.
        budget.set_user_render_fps(144);
        assert_eq!(
            budget.render_fps_target(),
            45,
            "user intent above PressureLight ceiling is clamped"
        );
    }

    /// Contract: no mode emits `yolox_skip_period = 0`. The tracking
    /// thread reads the period inside `frame_index.is_multiple_of(period)`
    /// which would panic on zero. Defence-in-depth `.max(1)` on the
    /// read site backs this, but the invariant should hold at the
    /// source too.
    #[test]
    fn all_modes_emit_nonzero_yolox_skip_period() {
        let t0 = Instant::now();
        let mut budget = RuntimeGpuBudget::new(t0);
        for mode in [
            DegradedMode::Healthy,
            DegradedMode::PressureLight,
            DegradedMode::PressureHeavy,
            DegradedMode::EmergencyCpu,
        ] {
            // Force the mode + recompute targets through the public surface.
            budget.degraded_mode = mode;
            budget.recompute_targets();
            assert!(
                budget.yolox_skip_period() >= 1,
                "mode {:?} emitted yolox_skip_period = 0",
                mode
            );
        }
    }

    /// Defence: if the dwell timer's anchor is in the future relative
    /// to `now` (clock adjustment / test injection), it must be clamped
    /// down to `now` so subsequent `duration_since(s)` doesn't saturate
    /// to zero forever.
    #[test]
    fn clock_going_backward_does_not_freeze_state_machine() {
        let t0 = Instant::now();
        let mut budget = RuntimeGpuBudget::new(t0);
        // Move into PressureLight and arm pressure_since at t0+5s.
        budget.update(&render_overrun_measurements(), t0);
        budget.update(&render_overrun_measurements(), t0 + LIGHT_TO_HEAVY_DWELL / 2);
        // Now travel "backward" by calling update with `now = t0` again
        // (simulates a clock adjustment). pressure_since was in the
        // future; the guard must reset it to `now` so dwell can advance.
        budget.update(&render_overrun_measurements(), t0);
        // From this point, advancing forward by the dwell time must
        // reach PressureHeavy as if the backward jump never happened.
        budget.update(
            &render_overrun_measurements(),
            t0 + LIGHT_TO_HEAVY_DWELL + std::time::Duration::from_millis(1),
        );
        assert_eq!(budget.degraded_mode(), DegradedMode::PressureHeavy);
    }
}
