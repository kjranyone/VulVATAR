//! Lipsync lifecycle: enable / disable + per-frame inference, with
//! feature-gated stubs that keep call sites identical when the
//! `lipsync` feature is disabled.

#[cfg(feature = "lipsync")]
use log::info;

use super::Application;

impl Application {
    /// Bring the lipsync processor in line with the requested state. Idempotent
    /// over `(enabled, mic_device_index)`. The mic preference is always
    /// stored, even when `enabled=false`, so the user's selection survives
    /// a disable / re-enable cycle.
    ///
    /// - `enabled=true`, not running: start with `mic_device_index`
    /// - `enabled=true`, running, mic changed: restart with new mic
    /// - `enabled=true`, running, mic unchanged: no-op
    /// - `enabled=false`, running: stop, remember mic for next enable
    /// - `enabled=false`, not running: just remember mic
    ///
    /// Errors propagate from the underlying audio capture init; callers
    /// should surface them to the user (push notification).
    #[cfg(feature = "lipsync")]
    pub fn set_lipsync_enabled(
        &mut self,
        enabled: bool,
        mic_device_index: usize,
    ) -> Result<(), String> {
        let active = self.lipsync_processor.is_some();
        let mic_changed = self.lipsync_mic_device_index != mic_device_index;
        // Remember the mic preference up-front so the disable path also
        // updates it (used as the start mic for the next enable).
        self.lipsync_mic_device_index = mic_device_index;

        if !enabled {
            if active {
                if let Some(ref mut proc) = self.lipsync_processor {
                    proc.stop();
                }
                self.lipsync_processor = None;
                info!("app: lipsync stopped");
            }
            return Ok(());
        }

        if !active || mic_changed {
            if let Some(ref mut proc) = self.lipsync_processor {
                proc.stop();
                self.lipsync_processor = None;
            }
            let proc = crate::lipsync::LipSyncProcessor::start(Some(mic_device_index))?;
            self.lipsync_processor = Some(proc);
            info!(
                "app: lipsync {} (mic={})",
                if mic_changed && active {
                    "restarted"
                } else {
                    "started"
                },
                mic_device_index
            );
        }
        Ok(())
    }

    /// No-op stub when the `lipsync` feature is disabled — keeps GUI call
    /// sites identical regardless of build features.
    #[cfg(not(feature = "lipsync"))]
    pub fn set_lipsync_enabled(
        &mut self,
        _enabled: bool,
        _mic_device_index: usize,
    ) -> Result<(), String> {
        Ok(())
    }

    /// True if the lipsync processor is currently running. Reflects **runtime
    /// reality**, not user intent — e.g. if the user requested `enabled=true`
    /// but the audio device couldn't open, this stays `false`. Use
    /// [`Self::requested_lipsync_enabled`] for user intent (project save,
    /// combo display).
    pub fn is_lipsync_enabled(&self) -> bool {
        #[cfg(feature = "lipsync")]
        return self.lipsync_processor.is_some();
        #[cfg(not(feature = "lipsync"))]
        return false;
    }

    /// The mic device index the running processor uses, or the preferred
    /// mic for the next enable. Always defined (defaults to 0).
    pub fn lipsync_mic_device_index(&self) -> usize {
        #[cfg(feature = "lipsync")]
        return self.lipsync_mic_device_index;
        #[cfg(not(feature = "lipsync"))]
        return 0;
    }

    /// Run lipsync inference for one frame and apply viseme weights to the
    /// active avatar's expression weights. Returns the current rms volume
    /// for the GUI's volume meter, or `None` when lipsync is not active.
    ///
    /// Called every frame from `GuiApp::update` so lipsync output continues
    /// even when the lip sync inspector panel is collapsed (the previous
    /// design ran inference inside `draw_lipsync` and froze when the panel
    /// was hidden).
    #[cfg(feature = "lipsync")]
    pub fn step_lipsync(&mut self, smoothing: f32, volume_threshold: f32, dt: f32) -> Option<f32> {
        let viseme = self
            .lipsync_processor
            .as_mut()?
            .process_frame(smoothing, dt.max(0.001));
        let rms = self.lipsync_processor.as_ref()?.rms_volume();

        if let Some(avatar) = self.active_avatar_mut() {
            for (name, weight) in viseme.as_expression_pairs() {
                let effective = if weight < volume_threshold && name == "aa" {
                    // Below threshold: close mouth.
                    0.0
                } else {
                    weight
                };
                if let Some(ew) = avatar
                    .expression_weights
                    .iter_mut()
                    .find(|w| w.name == name)
                {
                    ew.weight = effective;
                }
            }
        }
        Some(rms)
    }

    #[cfg(not(feature = "lipsync"))]
    pub fn step_lipsync(
        &mut self,
        _smoothing: f32,
        _volume_threshold: f32,
        _dt: f32,
    ) -> Option<f32> {
        None
    }
}
