//! Project & profile lifecycle for `GuiApp`. Snapshots GUI state into a
//! `ProjectState` for persistence, applies a loaded `ProjectState` (or
//! `StreamProfile`) back onto the GUI fields, and handles cloth-overlay
//! rebinding/save when the active avatar's bone IDs have shifted since
//! the overlay was authored.
//!
//! Pipeline-bound settings (sink, lipsync) round-trip through
//! `Application::set_requested_*` rather than the GUI's shadow fields —
//! see `apply_pipeline_bound_settings` for the rationale.

use std::time::{Duration, Instant};

use log::{debug, warn};

use crate::persistence::ProjectState;
use crate::t;

use super::{profile, GuiApp};

/// Re-serialise a `.vvtcloth` after a Partial rebind so the on-disk
/// IDs match the new avatar going forward. Reads the existing file
/// first to preserve everything outside of `cloth_asset` and the
/// audit fields (`last_saved_with`, `last_rebound_with`).
pub(super) fn save_rebound_overlay(
    path: &std::path::Path,
    cloth_asset: &crate::asset::ClothAsset,
) -> Result<(), String> {
    let mut file = crate::persistence::load_cloth_overlay(path)?;
    file.cloth_asset = Some(cloth_asset.clone());
    let app_tag = format!("VulVATAR {}", env!("CARGO_PKG_VERSION"));
    file.last_saved_with = app_tag.clone();
    file.last_rebound_with = Some(app_tag);
    let json = serde_json::to_string_pretty(&file)
        .map_err(|e| format!("serialise rebound overlay: {}", e))?;
    crate::persistence::atomic_write(path, &json)
        .map_err(|e| format!("write rebound overlay '{}': {}", path.display(), e))
}

impl GuiApp {
    /// Snapshot the current GUI state into a `ProjectState` for persistence.
    pub fn to_project_state(&self) -> ProjectState {
        let avatar_source_path = self
            .app
            .active_avatar()
            .map(|a| a.asset.source_path.to_string_lossy().into_owned());

        let avatar_source_hash = self
            .app
            .active_avatar()
            .map(|a| a.asset.source_hash.0.to_vec());

        let active_overlay_path = self
            .app
            .editor
            .overlay_path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned());

        // File-backed cloth overlays attached to the active avatar via
        // "Load Overlay File...". Procedurally-added slots have None
        // here and intentionally don't round-trip through projects.
        let cloth_overlay_paths: Vec<String> = self
            .app
            .active_avatar()
            .map(|a| {
                a.cloth_overlays
                    .iter()
                    .filter_map(|s| s.source_path.as_ref())
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();

        ProjectState {
            avatar_source_path,
            avatar_source_hash,
            active_overlay_path,
            cloth_overlay_paths,

            transform_position: self.transform.position,
            transform_rotation: self.transform.rotation,
            transform_scale: self.transform.scale,

            camera_orbit_yaw: self.camera_orbit.yaw_deg,
            camera_orbit_pitch: self.camera_orbit.pitch_deg,
            camera_orbit_pan: self.camera_orbit.pan,
            camera_orbit_distance: self.camera_orbit.distance,

            tracking_enabled: self.tracking.toggle_tracking,
            tracking_mirror: self.tracking.tracking_mirror,
            camera_resolution_index: self.tracking.camera_resolution_index,
            camera_framerate_index: self.tracking.camera_framerate_index,
            hand_tracking_enabled: self.tracking.hand_tracking_enabled,
            face_tracking_enabled: self.tracking.face_tracking_enabled,
            lower_body_tracking_enabled: self.tracking.lower_body_tracking_enabled,
            root_translation_enabled: self.tracking.root_translation_enabled,
            fade_on_tracking_loss: self.tracking.fade_on_tracking_loss,
            depth_enabled: self.tracking.depth_enabled,
            force_cpu_inference: self.tracking.force_cpu_inference,
            yolox_enabled: self.tracking.yolox_enabled,
            camera_index: self.camera_index,
            show_camera_wipe: self.viewport.show_camera_wipe,
            show_detection_annotations: self.viewport.show_detection_annotations,
            smoothing_rotation_blend: self.tracking.smoothing.rotation_blend,
            smoothing_expression_blend: self.tracking.smoothing.expression_blend,
            smoothing_joint_confidence: self.tracking.smoothing.joint_confidence_threshold,
            smoothing_face_confidence: self.tracking.smoothing.face_confidence_threshold,
            // Calibration now lives on `StreamProfile.pose_calibration`
            // (per-room/setup storage), not the project file. New saves
            // never emit this field — the on-disk DTO has
            // `#[serde(skip_serializing_if = "Option::is_none")]` so
            // older project files that *did* carry a calibration get
            // their value migrated onto the active profile by
            // `load_state`, then this `None` ensures the field is
            // dropped from the next save.
            pose_calibration: None,

            material_mode_index: self.rendering.material_mode_index,
            light_direction: self.rendering.main_light_dir,
            light_intensity: self.rendering.main_light_intensity,
            ambient: self.rendering.ambient_intensity,
            camera_fov: self.rendering.camera_fov,
            background_color: self.rendering.background_color,
            transparent_background: self.rendering.transparent_background,
            toggle_spring: self.rendering.toggle_spring,
            toggle_cloth: self.rendering.toggle_cloth,
            toggle_collision_debug: self.rendering.toggle_collision_debug,
            toggle_skeleton_debug: self.rendering.toggle_skeleton_debug,
            alpha_preview: self.rendering.alpha_preview,
            bloom_enabled: self.rendering.bloom_enabled,
            bloom_intensity: self.rendering.bloom_intensity,
            bloom_threshold: self.rendering.bloom_threshold,
            bg_enabled: self.rendering.generative_background.enabled,
            bg_intensity: self.rendering.generative_background.intensity,
            bg_speed: self.rendering.generative_background.speed,
            bg_scale: self.rendering.generative_background.scale,
            bg_reactivity: self.rendering.generative_background.reactivity,
            bg_color_a: self.rendering.generative_background.color_a,
            bg_color_b: self.rendering.generative_background.color_b,

            // Save user **intent** (Phase D), not the currently-running
            // state. If MF init failed this session, runtime is on a
            // SharedMemory fallback but we don't want to overwrite the
            // user's saved VirtualCamera choice.
            lipsync_enabled: self.app.requested_lipsync_enabled(),
            lipsync_mic_device_index: self.app.lipsync_mic_device_index(),
            lipsync_volume_threshold: self.lipsync.volume_threshold,
            lipsync_smoothing: self.lipsync.smoothing,
            mouth_source_index: self.lipsync.mouth_source.to_index(),

            output_sink_index: self.app.requested_sink().to_gui_index(),
            output_resolution_index: self.output.output_resolution_index,
            output_framerate_index: self.output.output_framerate_index,
            output_has_alpha: self.output.output_has_alpha,
            output_color_space_index: self.output.output_color_space_index,
            output_msaa_index: self.output.msaa_index,
        }
    }

    /// Snapshot the app-level settings for `settings.json`. Kept next
    /// to `to_project_state` so the "which file owns which setting"
    /// split stays visible in one place: locale + input sensitivities +
    /// consents follow the user (settings.json); everything above
    /// follows the scene (.vvtproj).
    fn collect_app_settings(&self) -> crate::persistence::AppSettings {
        crate::persistence::AppSettings {
            locale: self.settings.locale.clone(),
            zoom_sensitivity: self.settings.zoom_sensitivity,
            orbit_sensitivity: self.settings.orbit_sensitivity,
            pan_sensitivity: self.settings.pan_sensitivity,
            cloth_autosave_consent: self.cloth_authoring.autosave_consent,
            ..crate::persistence::AppSettings::default()
        }
    }

    /// Apply the pipeline-bound parts of a saved snapshot directly to
    /// `Application` mutators (the Phase D `set_requested_*` variants so
    /// runtime failures don't overwrite the user's saved intent). After
    /// Phase C the GUI no longer holds shadow values for these settings;
    /// `Application` is the single source of truth. Failures surface as
    /// user-visible notifications; the runtime stays at its fallback state
    /// while the requested state remembers the user's pick.
    fn apply_pipeline_bound_settings(
        &mut self,
        sink_index: usize,
        lipsync_enabled: bool,
        lipsync_mic: usize,
    ) {
        use crate::output::FrameSink;
        let want_sink = FrameSink::from_gui_index(sink_index);
        if let Err(e) = self.app.set_requested_sink(want_sink) {
            self.push_notification(t!("toast.virtual_camera_unavailable", error = e.to_string()));
        }
        if let Err(e) = self.app.set_requested_lipsync(lipsync_enabled, lipsync_mic) {
            warn!("lipsync apply failed: {e}");
            self.push_notification(t!("toast.lipsync_failed", error = e.to_string()));
        }
    }

    pub fn apply_profile(&mut self, profile: &profile::StreamProfile) {
        self.tracking.tracking_mirror = profile.tracking_mirror;
        self.rendering.main_light_dir = profile.light_direction;
        self.rendering.main_light_intensity = profile.light_intensity;
        self.rendering.ambient_intensity = profile.ambient;
        self.rendering.camera_fov = profile.camera_fov;
        self.output.output_resolution_index = profile.output_resolution_index;
        self.output.output_framerate_index = profile.output_framerate_index;
        self.output.output_has_alpha = profile.output_has_alpha;
        self.output.output_color_space_index = profile.output_color_space_index;
        self.output.msaa_index = profile.output_msaa_index;
        // Pose calibration follows the profile — switching to "Office
        // desk" pulls in the office calibration; switching to "Home
        // setup" pulls in the home one. Push to both Application
        // (so the solver's per-frame `apply_calibration` sees it
        // next tick) and the tracking mailbox (so the depth-pipeline
        // provider re-applies the c-clamp / anchor-mode for the new
        // setup). `None` here is meaningful: it deliberately clears
        // any previous setup's calibration so the auto-EMA falls
        // back to its first-frame seed instead of carrying over a
        // baseline from the wrong room.
        self.app.tracking_calibration.pose = profile.pose_calibration.clone();
        self.app
            .tracking
            .mailbox()
            .set_calibration(profile.pose_calibration.clone());
        self.project_status.project_dirty = true;
        // Profiles don't carry lipsync settings; pass current App requested
        // state to keep them unchanged.
        self.apply_pipeline_bound_settings(
            profile.output_sink_index,
            self.app.requested_lipsync_enabled(),
            self.app.lipsync_mic_device_index(),
        );
    }

    /// Apply a loaded `ProjectState` to the GUI fields.
    /// Avatar and overlay loading is handled separately by the caller.
    pub fn apply_project_state(&mut self, state: &ProjectState) {
        self.transform.position = state.transform_position;
        self.transform.rotation = state.transform_rotation;
        self.transform.scale = state.transform_scale;

        self.camera_orbit.yaw_deg = state.camera_orbit_yaw;
        self.camera_orbit.pitch_deg = state.camera_orbit_pitch;
        self.camera_orbit.pan = state.camera_orbit_pan;
        self.camera_orbit.distance = state.camera_orbit_distance;
        self.camera_orbit.target_distance = state.camera_orbit_distance;

        self.tracking.toggle_tracking = state.tracking_enabled;
        self.tracking.tracking_mirror = state.tracking_mirror;
        self.tracking.camera_resolution_index = state.camera_resolution_index;
        self.tracking.camera_framerate_index = state.camera_framerate_index;
        self.tracking.hand_tracking_enabled = state.hand_tracking_enabled;
        self.tracking.face_tracking_enabled = state.face_tracking_enabled;
        self.tracking.lower_body_tracking_enabled = state.lower_body_tracking_enabled;
        self.tracking.root_translation_enabled = state.root_translation_enabled;
        self.tracking.fade_on_tracking_loss = state.fade_on_tracking_loss;
        self.tracking.depth_enabled = state.depth_enabled;
        self.tracking.force_cpu_inference = state.force_cpu_inference;
        self.tracking.yolox_enabled = state.yolox_enabled;
        self.camera_index = state.camera_index;
        self.viewport.show_camera_wipe = state.show_camera_wipe;
        self.viewport.show_detection_annotations = state.show_detection_annotations;
        self.tracking.smoothing.rotation_blend = state.smoothing_rotation_blend;
        self.tracking.smoothing.expression_blend = state.smoothing_expression_blend;
        self.tracking.smoothing.joint_confidence_threshold = state.smoothing_joint_confidence;
        self.tracking.smoothing.face_confidence_threshold = state.smoothing_face_confidence;
        // Pose calibration migration: per-profile storage replaced
        // per-project storage in this version. When the loaded
        // project file carries a legacy `pose_calibration` value
        // *and* the currently-active profile has none yet, treat it
        // as a one-time rescue and copy it onto the profile (marking
        // the profile library dirty so it lands in `profiles.json`
        // on the next autosave). When the profile already has a
        // calibration, the legacy value is discarded — the user has
        // since calibrated under the per-profile model and that
        // takes precedence.
        //
        // Either way, push the *resulting* calibration (active
        // profile's, post-rescue) to Application + tracking mailbox
        // so the solver and depth pipeline see the right value
        // immediately. Calling with `None` is meaningful — it means
        // "no calibration for this profile; fall back to auto-EMA".
        if let Some(legacy) = state.pose_calibration.as_ref() {
            if let Some(idx) = self.profiles.active_index {
                if let Some(active_profile) = self.profiles.profiles.get_mut(idx) {
                    if active_profile.pose_calibration.is_none() {
                        active_profile.pose_calibration = Some(legacy.clone());
                        self.project_status.profiles_dirty = true;
                    }
                }
            }
        }
        let active_calibration = self
            .profiles
            .active()
            .and_then(|p| p.pose_calibration.clone());
        self.app.tracking_calibration.pose = active_calibration.clone();
        self.app
            .tracking
            .mailbox()
            .set_calibration(active_calibration);

        self.rendering.material_mode_index = state.material_mode_index;
        self.rendering.main_light_dir = state.light_direction;
        self.rendering.main_light_intensity = state.light_intensity;
        self.rendering.ambient_intensity = state.ambient;
        self.rendering.camera_fov = state.camera_fov;
        self.rendering.background_color = state.background_color;
        self.rendering.transparent_background = state.transparent_background;
        self.rendering.toggle_spring = state.toggle_spring;
        self.rendering.toggle_cloth = state.toggle_cloth;
        self.rendering.toggle_collision_debug = state.toggle_collision_debug;
        self.rendering.toggle_skeleton_debug = state.toggle_skeleton_debug;
        self.rendering.alpha_preview = state.alpha_preview;
        self.rendering.bloom_enabled = state.bloom_enabled;
        self.rendering.bloom_intensity = state.bloom_intensity;
        self.rendering.bloom_threshold = state.bloom_threshold;
        self.rendering.generative_background =
            crate::renderer::frame_input::GenerativeBackgroundSettings {
                enabled: state.bg_enabled,
                intensity: state.bg_intensity,
                speed: state.bg_speed,
                scale: state.bg_scale,
                reactivity: state.bg_reactivity,
                color_a: state.bg_color_a,
                color_b: state.bg_color_b,
            };

        self.lipsync.volume_threshold = state.lipsync_volume_threshold;
        self.lipsync.smoothing = state.lipsync_smoothing;
        self.lipsync.mouth_source =
            crate::tracking::MouthSource::from_index(state.mouth_source_index);

        self.output.output_resolution_index = state.output_resolution_index;
        self.output.output_framerate_index = state.output_framerate_index;
        self.output.output_has_alpha = state.output_has_alpha;
        self.output.output_color_space_index = state.output_color_space_index;
        self.output.msaa_index = state.output_msaa_index;

        // App-level settings (locale, sensitivities, consents) are
        // deliberately NOT applied from project state — they live in
        // settings.json and follow the user, not the scene. A project
        // file must never switch the UI language.

        self.apply_pipeline_bound_settings(
            state.output_sink_index,
            state.lipsync_enabled,
            state.lipsync_mic_device_index,
        );

        self.restore_cloth_overlay_paths(&state.cloth_overlay_paths);
    }

    /// Re-attach each cloth overlay listed in the project file to the
    /// active avatar. Each path is loaded via
    /// [`crate::persistence::load_cloth_overlay`] (which handles version
    /// migration), routed through the rebinder so reimported avatars
    /// don't break the binding by-id, and then attached + initialised
    /// on a fresh slot. Failed rebinds surface as notifications and
    /// skip that single overlay rather than aborting the whole apply.
    pub(super) fn restore_cloth_overlay_paths(&mut self, paths: &[String]) {
        if paths.is_empty() {
            return;
        }
        // Resolve the assets first so the avatar borrow doesn't overlap
        // with notification pushes.
        let resolved: Vec<(std::path::PathBuf, crate::asset::ClothAsset)> = paths
            .iter()
            .filter_map(|p| {
                let path = std::path::PathBuf::from(p);
                if !path.exists() {
                    self.push_notification(t!("toast.cloth_overlay_not_found", path = path.display().to_string()));
                    return None;
                }
                match crate::persistence::load_cloth_overlay(&path) {
                    Ok(file) => match file.cloth_asset {
                        Some(asset) => Some((path, asset)),
                        None => {
                            self.push_notification(t!("toast.cloth_overlay_no_payload", path = path.display().to_string()));
                            None
                        }
                    },
                    Err(e) => {
                        self.push_notification(t!("toast.failed_load_cloth_overlay", path = path.display().to_string(), error = e.to_string()));
                        None
                    }
                }
            })
            .collect();

        if resolved.is_empty() {
            return;
        }

        for (path, mut cloth_asset) in resolved {
            if !self.attempt_overlay_rebind(&path, &mut cloth_asset) {
                continue; // rebind reported Failed; skip attach
            }
            if let Some(avatar) = self.app.active_avatar_mut() {
                let overlay_id = crate::asset::ClothOverlayId(
                    (avatar.cloth_overlay_count() as u64) + 2,
                );
                let idx = avatar.attach_cloth_overlay(overlay_id);
                avatar.init_cloth_overlay(idx, &cloth_asset);
                if let Some(slot) = avatar.cloth_overlays.get_mut(idx) {
                    slot.source_path = Some(path);
                }
            }
        }
    }

    /// Run the cloth-rebind pass against the active avatar and route
    /// the report status: Clean → silent debug log; Partial → user
    /// notification + write-back to disk with `last_rebound_with`
    /// stamp; Failed → notification, return `false` so the caller
    /// skips attaching the overlay.
    pub(crate) fn attempt_overlay_rebind(
        &mut self,
        path: &std::path::Path,
        cloth_asset: &mut crate::asset::ClothAsset,
    ) -> bool {
        use crate::asset::cloth_rebind::{rebind_overlay, RebindStatus};

        let report = match self.app.active_avatar() {
            Some(avatar) => rebind_overlay(cloth_asset, &avatar.asset),
            None => return true, // no avatar to rebind against; nothing to do
        };

        match report.status {
            RebindStatus::Clean => {
                debug!(
                    "rebind: '{}' clean ({} node refs, {} primitive refs walked)",
                    path.display(),
                    report.node_remappings.len(),
                    report.primitive_remappings.len()
                );
                true
            }
            RebindStatus::Partial => {
                self.push_notification(t!("toast.rebound_overlay", path = path.display().to_string(), nodes = report.node_remappings.len(), primitives = report.primitive_remappings.len()));
                if let Err(e) = save_rebound_overlay(path, cloth_asset) {
                    warn!(
                        "rebind: failed to write back rebound overlay '{}': {}",
                        path.display(),
                        e
                    );
                }
                true
            }
            RebindStatus::Failed => {
                self.push_notification(t!("toast.could_not_rebind", path = path.display().to_string(), unresolved = report.unresolved.len(), geometry = report.fatal_geometry_changes.len()));
                false
            }
        }
    }

    /// Per-frame autosave tick. Flushes the project file if dirty (subject
    /// to a 250 ms throttle so slider drags don't hammer the disk) and
    /// flushes the profile library if dirty (no throttle: writes are rare
    /// and target a different file). Both failures are non-fatal — the
    /// next dirty mutation retries.
    ///
    /// The throttle clock is bumped on success **and** failure so a
    /// permanently-failing target (read-only disk, missing directory)
    /// can't pin the GUI in a tight save-fail loop.
    pub(super) fn autosave_tick(&mut self) {
        // Immediate-save throttle: write project state to disk within
        // `AUTOSAVE_THROTTLE` of the last mutation that flipped
        // `project_dirty`. The user toggles a checkbox; ~250 ms later it
        // hits disk; quitting the app a second after that does not lose
        // the change. Slider drags re-trigger every frame but the
        // throttle caps writes at ~4/s — at <2 KB per write the I/O is
        // negligible. When `project_path` is `None` (the common case —
        // user never ran File > Save As) we fall back to the implicit
        // `last_session.vvtproj` so non-power-users get persistence too.
        // No toast: a "saved" notification firing 4× a second during a
        // slider drag would be UI spam.
        const AUTOSAVE_THROTTLE: Duration = Duration::from_millis(250);
        if self.project_status.project_dirty && self.project_status.last_autosave.elapsed() >= AUTOSAVE_THROTTLE {
            let path = self
                .project_status
                .project_path
                .clone()
                .unwrap_or_else(crate::persistence::last_session_path);
            let project_state = self.to_project_state();
            match crate::persistence::save_project(&project_state, &path) {
                Ok(()) => {
                    self.project_status.project_dirty = false;
                    self.project_status.last_autosave = Instant::now();
                }
                Err(e) => {
                    // Bump the timestamp so a persistent failure (read-only
                    // disk, missing directory) doesn't pin the GUI in a
                    // tight save-fail loop. The next mutation will retry.
                    self.project_status.last_autosave = Instant::now();
                    self.push_notification(t!("toast.autosave_failed", error = e.to_string()));
                }
            }
        }

        // Profile-library autosave. Triggered by the Calibrate Pose
        // modal writing into the active profile's `pose_calibration`,
        // and by any future profile-edit UI. Unlike `project_dirty`
        // we don't share the AUTOSAVE_THROTTLE clock with the
        // project save — the two writes target different files and
        // it's fine for them to interleave. Failure is non-fatal:
        // the next dirty mutation retries on the following frame.
        if self.project_status.profiles_dirty {
            match crate::persistence::save_profiles(&self.profiles) {
                Ok(()) => {
                    self.project_status.profiles_dirty = false;
                }
                Err(e) => {
                    // Don't toast — calibration capture already
                    // surfaces a "Done" notification, and pinning
                    // a save-failed toast on top would be noise.
                    // Log only; the next mutation retries.
                    warn!("persistence: save_profiles failed: {}", e);
                    // Clear the flag so a permanently-read-only
                    // config dir doesn't trigger a save attempt
                    // on every frame for the rest of the session.
                    self.project_status.profiles_dirty = false;
                }
            }
        }

        // App-settings autosave (`settings.json`): locale / input
        // sensitivities / consents. Own dirty flag + own file, but
        // throttled like the project save above — the sensitivity
        // sliders re-set the flag every frame of a drag, and each write
        // is a backup-copy + temp + rename, so an unthrottled flush
        // would hammer the disk (~180 fs ops/s) for the duration of a
        // drag.
        if self.project_status.app_settings_dirty
            && self.project_status.last_app_settings_save.elapsed() >= AUTOSAVE_THROTTLE
        {
            match crate::persistence::save_app_settings(&self.collect_app_settings()) {
                Ok(()) => {
                    self.project_status.app_settings_dirty = false;
                    self.project_status.last_app_settings_save = Instant::now();
                }
                Err(e) => {
                    warn!("persistence: save_app_settings failed: {}", e);
                    // Bump the clock and clear the flag so a
                    // permanently-read-only config dir doesn't retry
                    // every frame; the next Settings change re-arms it.
                    self.project_status.last_app_settings_save = Instant::now();
                    self.project_status.app_settings_dirty = false;
                }
            }
        }
    }

    /// Write a recovery snapshot (project + optional cloth overlay) when
    /// the recovery manager's timer fires. The overlay is included only
    /// if the user opted in via the cloth-autosave consent dialog.
    /// Called once per frame; the manager's internal timer gates writes.
    pub(super) fn write_recovery_snapshot_if_due(&mut self) {
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if !self.project_status.recovery_manager.should_snapshot(now_secs) {
            return;
        }

        let project_state = if self.project_status.project_dirty {
            Some(self.to_project_state())
        } else {
            None
        };

        let has_overlay = self.app.editor.overlay_asset.is_some();
        let overlay_for_snapshot = if has_overlay && self.cloth_authoring.autosave_consent == Some(true) {
            let overlay_name = self
                .app
                .editor
                .overlay_asset
                .as_ref()
                .map(|a| a.metadata.name.clone())
                .unwrap_or_else(|| "Untitled".to_string());
            let target_avatar_path = self
                .app
                .active_avatar()
                .map(|a| a.asset.source_path.to_string_lossy().into_owned());
            Some(crate::persistence::ClothOverlayFile {
                format_version: crate::persistence::OVERLAY_FORMAT_VERSION,
                created_with: format!("VulVATAR {}", env!("CARGO_PKG_VERSION")),
                last_saved_with: format!("VulVATAR {}", env!("CARGO_PKG_VERSION")),
                overlay_name,
                target_avatar_path,
                cloth_asset: self.app.editor.overlay_asset.clone(),
                last_rebound_with: None,
            })
        } else {
            None
        };

        let _ = self.project_status.recovery_manager.write_snapshot(
            project_state.as_ref(),
            self.project_status.overlay_dirty,
            overlay_for_snapshot.as_ref(),
        );
    }
}
