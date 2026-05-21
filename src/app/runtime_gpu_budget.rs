//! Runtime GPU budget policy (P3-03).
//!
//! Centralises the render / output / tracking cadence decisions that were
//! previously scattered as constants across `OutputRouter::set_target_fps`,
//! `tracking/rtmw3d/mod.rs::YOLOX_REFRESH_PERIOD`, depth refresh periods
//! and FaceMesh EP selection. Consumers read the latest target from the
//! budget; the budget reads live measurements (render dt, output queue
//! depth, export pool occupancy, recent GPU export failures) and decides
//! one global [`DegradedMode`] level that drives all targets together.
//!
//! See `plan/runtime-gpu-budget.md` for the design and threshold rationale.

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

/// FaceMesh ONNX Runtime execution-provider preference. The actual EP
/// negotiation happens in `tracking::face_mediapipe`; the budget just
/// publishes the *intent* so the tracking worker can act on it next time
/// it negotiates a session.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FaceMeshEpPreference {
    /// Let the EP factory decide (current behaviour).
    #[default]
    Auto,
    /// Prefer the GPU EP (DirectML on Windows).
    Gpu,
    /// Force the CPU EP. Used under heavy pressure to free up the
    /// DirectML command queue for other workloads.
    Cpu,
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
    /// Number of GPU export failures observed in the last 10 seconds.
    /// Drives the `PressureHeavy → EmergencyCpu` step.
    pub gpu_export_failures_recent: u32,
}

impl Default for RuntimeMeasurements {
    fn default() -> Self {
        Self {
            render_dt: Duration::from_secs_f32(1.0 / 60.0),
            render_target: Duration::from_secs_f32(1.0 / 60.0),
            output_drops_per_sec: 0.0,
            export_pool_leased: 0,
            export_pool_capacity: 2,
            gpu_export_failures_recent: 0,
        }
    }
}

/// Knobs published by the budget. Consumers read these each tick.
#[derive(Clone, Copy, Debug)]
pub struct RuntimeGpuBudget {
    /// User's intent for render cadence (combo box). The budget may
    /// clamp the *exposed* target below this under pressure.
    user_render_fps: u32,

    render_fps_target: u32,
    pose_hz_target: u32,
    yolox_skip_period: u32,
    depth_skip_period: u32,
    facemesh_ep_preference: FaceMeshEpPreference,
    degraded_mode: DegradedMode,

    pressure_since: Option<Instant>,
    clean_streak_started: Option<Instant>,
    last_transition_at: Instant,
    last_transition_reason: TransitionReason,
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

/// Hysteresis windows. Mirror the values documented in
/// `plan/runtime-gpu-budget.md`; tuned after live data lands.
const LIGHT_TO_HEAVY_DWELL: Duration = Duration::from_secs(5);
const RECOVERY_DWELL: Duration = Duration::from_secs(30);
const HEAVY_DROP_RATE_PER_SEC: f32 = 20.0;
const LIGHT_DROP_RATE_PER_SEC: f32 = 5.0;
const RENDER_OVERRUN_MULTIPLE: f32 = 1.2;
const EMERGENCY_FAILURE_COUNT: u32 = 2;

impl RuntimeGpuBudget {
    pub fn new(now: Instant) -> Self {
        Self {
            user_render_fps: 60,
            render_fps_target: 60,
            pose_hz_target: 30,
            yolox_skip_period: 4,
            depth_skip_period: 4,
            facemesh_ep_preference: FaceMeshEpPreference::Auto,
            degraded_mode: DegradedMode::Healthy,
            pressure_since: None,
            clean_streak_started: Some(now),
            last_transition_at: now,
            last_transition_reason: TransitionReason::InitialState,
        }
    }

    pub fn set_user_render_fps(&mut self, fps: u32) {
        self.user_render_fps = fps.max(1);
        self.recompute_targets();
    }

    pub fn render_fps_target(&self) -> u32 {
        self.render_fps_target
    }
    pub fn pose_hz_target(&self) -> u32 {
        self.pose_hz_target
    }
    pub fn yolox_skip_period(&self) -> u32 {
        self.yolox_skip_period
    }
    pub fn depth_skip_period(&self) -> u32 {
        self.depth_skip_period
    }
    pub fn facemesh_ep_preference(&self) -> FaceMeshEpPreference {
        self.facemesh_ep_preference
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
            (DegradedMode::PressureHeavy, _) => {
                if m.gpu_export_failures_recent >= EMERGENCY_FAILURE_COUNT {
                    self.transition_to(
                        DegradedMode::EmergencyCpu,
                        now,
                        TransitionReason::GpuExportFailures,
                    );
                }
            }
            (DegradedMode::EmergencyCpu, _) => {
                // Recovery only — no escalation past Emergency.
            }
            (_, None) => {
                if let Some(streak) = self.clean_streak_started {
                    if now.duration_since(streak) > RECOVERY_DWELL {
                        let next = match self.degraded_mode {
                            DegradedMode::EmergencyCpu => DegradedMode::PressureHeavy,
                            DegradedMode::PressureHeavy => DegradedMode::PressureLight,
                            DegradedMode::PressureLight => DegradedMode::Healthy,
                            DegradedMode::Healthy => DegradedMode::Healthy,
                        };
                        if next != self.degraded_mode {
                            self.transition_to(next, now, TransitionReason::Recovery);
                        }
                    }
                }
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
    }

    fn recompute_targets(&mut self) {
        match self.degraded_mode {
            DegradedMode::Healthy => {
                self.render_fps_target = self.user_render_fps;
                self.pose_hz_target = 30;
                self.yolox_skip_period = 4;
                self.depth_skip_period = 4;
                self.facemesh_ep_preference = FaceMeshEpPreference::Auto;
            }
            DegradedMode::PressureLight => {
                self.render_fps_target = self.user_render_fps.min(45);
                self.pose_hz_target = 25;
                self.yolox_skip_period = 6;
                self.depth_skip_period = 6;
                self.facemesh_ep_preference = FaceMeshEpPreference::Auto;
            }
            DegradedMode::PressureHeavy => {
                self.render_fps_target = self.user_render_fps.min(30);
                self.pose_hz_target = 20;
                self.yolox_skip_period = 8;
                self.depth_skip_period = 8;
                self.facemesh_ep_preference = FaceMeshEpPreference::Cpu;
            }
            DegradedMode::EmergencyCpu => {
                self.render_fps_target = self.user_render_fps.min(30);
                self.pose_hz_target = 15;
                self.yolox_skip_period = 12;
                self.depth_skip_period = 12;
                self.facemesh_ep_preference = FaceMeshEpPreference::Cpu;
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
        // Two GPU export failures bumps to Emergency.
        let mut m = heavy_drop_measurements();
        m.gpu_export_failures_recent = EMERGENCY_FAILURE_COUNT;
        budget.update(&m, t0 + Duration::from_millis(32));
        assert_eq!(budget.degraded_mode(), DegradedMode::EmergencyCpu);
        assert_eq!(
            budget.facemesh_ep_preference(),
            FaceMeshEpPreference::Cpu
        );
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
