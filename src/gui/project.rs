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

/// How often [`GuiApp::refresh_project_dirty`] actually compares the
/// live snapshot against the baseline. Matches the autosave write
/// throttle — probing faster than the writer can act is wasted work.
const DIRTY_PROBE_INTERVAL: Duration = Duration::from_millis(250);

impl GuiApp {
    /// Derive `project_dirty` by snapshotting the live state and
    /// comparing it against [`ProjectStatusUi::project_baseline`] (the
    /// last persisted or applied `ProjectState`). This is the *sole
    /// authority* on `project_dirty`: widgets no longer set the flag
    /// by hand, so a new setting added to `to_project_state` is
    /// automatically autosave-covered — forgetting a manual
    /// `project_dirty = true` used to mean that setting silently never
    /// persisted (the avatar-load case shipped exactly that bug).
    ///
    /// A side benefit of comparing states rather than accumulating
    /// events: toggling a value and toggling it back reads as clean
    /// again (until an autosave has landed in between, after which the
    /// baseline is the autosaved state — still correct).
    ///
    /// `force` skips the 250 ms probe throttle — used by `on_exit`,
    /// where the final flush must not race the throttle window.
    pub(super) fn refresh_project_dirty(&mut self, now: Instant, force: bool) {
        if !force && now.duration_since(self.project_status.last_dirty_probe) < DIRTY_PROBE_INTERVAL
        {
            return;
        }
        self.project_status.last_dirty_probe = now;
        let current = self.to_project_state();
        match self.project_status.project_baseline.as_ref() {
            Some(baseline) => self.project_status.project_dirty = *baseline != current,
            // First probe of the session (no load/apply seeded a
            // baseline): whatever exists now is the reference point.
            None => {
                self.project_status.project_baseline = Some(current);
                self.project_status.project_dirty = false;
            }
        }
    }

    /// Re-seed the dirty baseline to the current state and mark the
    /// project clean. Call after applying a loaded `ProjectState` or
    /// completing an explicit Save — any path where "what's on screen"
    /// and "what the flag should treat as persisted" re-converge.
    pub(super) fn mark_project_baseline(&mut self) {
        self.project_status.project_baseline = Some(self.to_project_state());
        self.project_status.project_dirty = false;
    }

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
            camera_capture_width: super::camera_resolution_for_index(
                self.tracking.camera_resolution_index,
            )
            .0,
            camera_capture_height: super::camera_resolution_for_index(
                self.tracking.camera_resolution_index,
            )
            .1,
            camera_capture_fps: super::camera_fps_for_index(self.tracking.camera_framerate_index),
            hand_tracking_enabled: self.tracking.hand_tracking_enabled,
            face_tracking_enabled: self.tracking.face_tracking_enabled,
            lower_body_tracking_enabled: self.tracking.lower_body_tracking_enabled,
            root_translation_enabled: self.tracking.root_translation_enabled,
            fade_on_tracking_loss: self.tracking.fade_on_tracking_loss,
            force_cpu_inference: self.tracking.force_cpu_inference,
            yolox_enabled: self.tracking.yolox_enabled,
            show_camera_wipe: self.viewport.show_camera_wipe,
            show_detection_annotations: self.viewport.show_detection_annotations,
            smoothing_rotation_blend: self.tracking.smoothing.rotation_blend,
            smoothing_expression_blend: self.tracking.smoothing.expression_blend,
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
            spring_sway_scale: self.rendering.spring_tuning.sway_scale,
            spring_gravity_offset: self.rendering.spring_tuning.gravity_offset,
            spring_natural_gravity: self.rendering.spring_tuning.natural_gravity,
            scene_gravity_direction: self.rendering.scene_gravity.direction,
            scene_gravity_strength: self.rendering.scene_gravity.strength,
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
    pub(super) fn collect_app_settings(&self) -> crate::persistence::AppSettings {
        crate::persistence::AppSettings {
            locale: self.settings.locale.clone(),
            zoom_sensitivity: self.settings.zoom_sensitivity,
            orbit_sensitivity: self.settings.orbit_sensitivity,
            pan_sensitivity: self.settings.pan_sensitivity,
            cloth_autosave_consent: self.cloth_authoring.autosave_consent,
            last_project_path: self.settings.last_project_path.clone(),
            camera_serial: self.tracking.camera_serial.clone(),
            ..crate::persistence::AppSettings::default()
        }
    }

    /// Record `path` as the last explicitly opened / saved project so
    /// the next launch re-opens it. Rides `settings.json` (a user-level
    /// preference, not scene state) via the app-settings autosave tick.
    pub(super) fn remember_last_project(&mut self, path: &std::path::Path) {
        let as_string = path.to_string_lossy().into_owned();
        if self.settings.last_project_path.as_deref() != Some(as_string.as_str()) {
            self.settings.last_project_path = Some(as_string);
            self.project_status.app_settings_dirty = true;
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
            self.push_warning_notification(t!(
                "toast.virtual_camera_unavailable",
                error = e.to_string()
            ));
        }
        if let Err(e) = self.app.set_requested_lipsync(lipsync_enabled, lipsync_mic) {
            warn!("lipsync apply failed: {e}");
            self.push_warning_notification(t!("toast.lipsync_failed", error = e.to_string()));
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
        self.tracking.camera_resolution_index = super::camera_resolution_index_for(
            state.camera_capture_width,
            state.camera_capture_height,
        );
        self.tracking.camera_framerate_index =
            super::camera_fps_index_for(state.camera_capture_fps);
        self.tracking.hand_tracking_enabled = state.hand_tracking_enabled;
        self.tracking.face_tracking_enabled = state.face_tracking_enabled;
        self.tracking.lower_body_tracking_enabled = state.lower_body_tracking_enabled;
        self.tracking.root_translation_enabled = state.root_translation_enabled;
        self.tracking.fade_on_tracking_loss = state.fade_on_tracking_loss;
        self.tracking.force_cpu_inference = state.force_cpu_inference;
        self.tracking.yolox_enabled = state.yolox_enabled;
        self.viewport.show_camera_wipe = state.show_camera_wipe;
        self.viewport.show_detection_annotations = state.show_detection_annotations;
        self.tracking.smoothing.rotation_blend = state.smoothing_rotation_blend;
        self.tracking.smoothing.expression_blend = state.smoothing_expression_blend;
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
        self.rendering.spring_tuning.sway_scale = state.spring_sway_scale;
        self.rendering.spring_tuning.gravity_offset = state.spring_gravity_offset;
        self.rendering.spring_tuning.natural_gravity = state.spring_natural_gravity;
        self.rendering.scene_gravity.direction = state.scene_gravity_direction;
        self.rendering.scene_gravity.strength = state.scene_gravity_strength;
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
                    self.push_warning_notification(t!(
                        "toast.cloth_overlay_not_found",
                        path = path.display().to_string()
                    ));
                    return None;
                }
                match crate::persistence::load_cloth_overlay(&path) {
                    Ok(file) => match file.cloth_asset {
                        Some(asset) => Some((path, asset)),
                        None => {
                            self.push_warning_notification(t!(
                                "toast.cloth_overlay_no_payload",
                                path = path.display().to_string()
                            ));
                            None
                        }
                    },
                    Err(e) => {
                        self.push_error_notification(t!(
                            "toast.failed_load_cloth_overlay",
                            path = path.display().to_string(),
                            error = e.to_string()
                        ));
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
                let overlay_id =
                    crate::asset::ClothOverlayId((avatar.cloth_overlay_count() as u64) + 2);
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
                self.push_notification(t!(
                    "toast.rebound_overlay",
                    path = path.display().to_string(),
                    nodes = report.node_remappings.len(),
                    primitives = report.primitive_remappings.len()
                ));
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
                self.push_error_notification(t!(
                    "toast.could_not_rebind",
                    path = path.display().to_string(),
                    unresolved = report.unresolved.len(),
                    geometry = report.fatal_geometry_changes.len()
                ));
                false
            }
        }
    }

    /// Per-frame autosave tick.
    ///
    /// **Autosave never writes the user's explicit `.vvtproj`.** Opening
    /// a project and experimenting used to overwrite the original file
    /// within 250 ms of the first slider drag, making the Save button
    /// meaningless and "just trying something" destructive. Unsaved
    /// changes now flow to:
    /// * `last_session.vvtproj` when no project is open (unchanged), or
    /// * a `<project>.vvtproj.unsaved` sidecar next to the open project
    ///   — restored (with a toast + dirty marker) on the next launch and
    ///   deleted by an explicit Save.
    ///
    /// Every save target (session/sidecar, profiles.json, settings.json)
    /// shares one failure policy via [`SaveRetry`]: the dirty flag stays
    /// set, retries back off exponentially (250 ms → 10 s), and the user
    /// sees exactly one sticky error toast per failing target, cleared
    /// when a later attempt succeeds. Profiles and settings previously
    /// cleared their dirty flag on failure, silently discarding pose
    /// calibrations and locale changes.
    pub(super) fn autosave_tick(&mut self) {
        // Unsaved-state autosave (session slot or project sidecar). No
        // toast on success: a "saved" notification firing 4× a second
        // during a slider drag would be UI spam.
        if self.project_status.project_dirty && self.project_status.project_save_retry.due() {
            let path = match self.project_status.project_path.clone() {
                Some(project) => unsaved_sidecar_path(&project),
                None => crate::persistence::last_session_path(),
            };
            let is_sidecar = self.project_status.project_path.is_some();
            let project_state = self.to_project_state();
            self.project_status.project_save_retry.record_attempt();
            match crate::persistence::save_project(&project_state, &path) {
                Ok(()) => {
                    self.project_status.project_save_retry.reset();
                    if let Some(msg) = self.project_status.last_project_save_error.take() {
                        self.dismiss_notifications_matching(&msg);
                    }
                    if is_sidecar {
                        // The explicit file is now behind the sidecar;
                        // keep the title-bar dot on until an explicit
                        // Save. `project_dirty` clears so the throttle
                        // logic (and the repaint gate keyed on it)
                        // doesn't spin while the user is idle.
                        self.project_status.explicit_file_stale = true;
                    }
                    // The written state becomes the dirty baseline: the
                    // next probe compares against what's actually on
                    // disk, so further edits re-flag and an untouched
                    // session stays clean.
                    self.project_status.project_baseline = Some(project_state);
                    self.project_status.project_dirty = false;
                }
                Err(e) => {
                    self.project_status.project_save_retry.record_failure();
                    let msg = t!("toast.autosave_failed", error = e.to_string());
                    self.push_sticky_error_notification(msg.clone());
                    self.project_status.last_project_save_error = Some(msg);
                }
            }
        }

        // Profile-library autosave. Triggered by the Calibrate Pose
        // modal writing into the active profile's `pose_calibration`,
        // and by any future profile-edit UI. Calibration data is
        // expensive to recapture, so a failure keeps the dirty flag,
        // backs off, and surfaces a sticky error toast.
        if self.project_status.profiles_dirty && self.project_status.profiles_save_retry.due() {
            self.project_status.profiles_save_retry.record_attempt();
            match crate::persistence::save_profiles(&self.profiles) {
                Ok(()) => {
                    self.project_status.profiles_dirty = false;
                    self.project_status.profiles_save_retry.reset();
                    if let Some(msg) = self.project_status.last_profiles_save_error.take() {
                        self.dismiss_notifications_matching(&msg);
                    }
                }
                Err(e) => {
                    warn!("persistence: save_profiles failed: {}", e);
                    self.project_status.profiles_save_retry.record_failure();
                    let msg = t!("toast.profiles_save_failed", error = e.to_string());
                    self.push_sticky_error_notification(msg.clone());
                    self.project_status.last_profiles_save_error = Some(msg);
                }
            }
        }

        // App-settings autosave (`settings.json`): locale / input
        // sensitivities / consents. The sensitivity sliders re-set the
        // flag every frame of a drag; the retry's base delay doubles as
        // the 250 ms write throttle.
        if self.project_status.app_settings_dirty && self.project_status.settings_save_retry.due() {
            self.project_status.settings_save_retry.record_attempt();
            match crate::persistence::save_app_settings(&self.collect_app_settings()) {
                Ok(()) => {
                    self.project_status.app_settings_dirty = false;
                    self.project_status.settings_save_retry.reset();
                    if let Some(msg) = self.project_status.last_settings_save_error.take() {
                        self.dismiss_notifications_matching(&msg);
                    }
                }
                Err(e) => {
                    warn!("persistence: save_app_settings failed: {}", e);
                    self.project_status.settings_save_retry.record_failure();
                    let msg = t!("toast.settings_save_failed", error = e.to_string());
                    self.push_sticky_error_notification(msg.clone());
                    self.project_status.last_settings_save_error = Some(msg);
                }
            }
        }
    }

    /// Write a recovery snapshot (project + optional cloth overlay) when
    /// the recovery manager's timer fires. The overlay is included only
    /// if the user opted in via the cloth-autosave consent dialog.
    /// Called once per frame; the manager's internal timer gates writes.
    pub(super) fn write_recovery_snapshot_if_due(&mut self) {
        if !self
            .project_status
            .recovery_manager
            .should_snapshot(Instant::now())
        {
            return;
        }

        // Always include the project state. It used to be gated on
        // `project_dirty`, but the 250 ms autosave clears that flag long
        // before the recovery timer fires, so the snapshot carried
        // `None` in practice and crash recovery restored nothing.
        // Serialising <2 KB every interval is free.
        let project_state = Some(self.to_project_state());

        let has_overlay = self.app.editor.overlay_asset.is_some();
        let overlay_for_snapshot =
            if has_overlay && self.cloth_authoring.autosave_consent == Some(true) {
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

        if let Err(e) = self.project_status.recovery_manager.write_snapshot(
            project_state.as_ref(),
            self.project_status.overlay_dirty,
            overlay_for_snapshot.as_ref(),
        ) {
            // A dead recovery channel is worth exactly one warning —
            // repeating it every interval would be noise, but total
            // silence means nobody learns the safety net is gone until
            // they need it.
            warn!("persistence: recovery snapshot write failed: {}", e);
            if !self.project_status.recovery_write_warned {
                self.project_status.recovery_write_warned = true;
                self.push_warning_notification(t!(
                    "toast.recovery_write_failed",
                    error = e.to_string()
                ));
            }
        }
    }
}

/// Exponential-backoff clock for a repeatedly failing save target.
/// `due()` gates attempts: the base delay doubles per consecutive
/// failure (250 ms, 500 ms, … capped at 10 s) and resets on success,
/// so a read-only disk degrades to one attempt every 10 s instead of
/// a 4 Hz fail-toast loop, while the dirty flag stays set and the data
/// is written the moment the target recovers.
pub(crate) struct SaveRetry {
    failures: u32,
    last_attempt: Instant,
}

impl SaveRetry {
    const BASE: Duration = Duration::from_millis(250);
    const MAX: Duration = Duration::from_secs(10);

    pub(super) fn new(now: Instant) -> Self {
        Self {
            failures: 0,
            // Backdate so the first mutation saves after one BASE delay.
            last_attempt: now.checked_sub(Self::MAX).unwrap_or(now),
        }
    }

    fn delay(&self) -> Duration {
        Self::BASE
            .saturating_mul(1u32 << self.failures.min(6))
            .min(Self::MAX)
    }

    pub(super) fn due(&self) -> bool {
        self.last_attempt.elapsed() >= self.delay()
    }

    pub(super) fn record_attempt(&mut self) {
        self.last_attempt = Instant::now();
    }

    pub(super) fn record_failure(&mut self) {
        self.failures = self.failures.saturating_add(1);
    }

    pub(super) fn reset(&mut self) {
        self.failures = 0;
    }
}

/// Sidecar slot for unsaved changes to an explicitly opened project:
/// `foo.vvtproj` → `foo.vvtproj.unsaved`. Autosave writes here so the
/// user's real file is only ever touched by an explicit Save; startup
/// restores it (marked dirty) when it is newer than the project file.
pub(super) fn unsaved_sidecar_path(project: &std::path::Path) -> std::path::PathBuf {
    let mut os = project.as_os_str().to_owned();
    os.push(".unsaved");
    std::path::PathBuf::from(os)
}

/// `true` when `sidecar` exists and its mtime is strictly newer than
/// `project`'s — i.e. the last session ended with unsaved changes on
/// top of the saved file. Unreadable metadata counts as "not newer" so
/// a broken sidecar can never shadow the real file.
pub(super) fn sidecar_is_newer(project: &std::path::Path, sidecar: &std::path::Path) -> bool {
    let mtime = |p: &std::path::Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    match (mtime(project), mtime(sidecar)) {
        (Some(proj), Some(side)) => side > proj,
        _ => false,
    }
}

/// Remove the unsaved-changes sidecar after an explicit Save has made
/// the real file current. Best-effort: a leftover sidecar older than
/// the project file is ignored by [`sidecar_is_newer`] anyway.
pub(super) fn clear_unsaved_sidecar(project: &std::path::Path) {
    let sidecar = unsaved_sidecar_path(project);
    if sidecar.exists() {
        if let Err(e) = std::fs::remove_file(&sidecar) {
            warn!(
                "persistence: could not remove unsaved sidecar '{}': {}",
                sidecar.display(),
                e
            );
        }
    }
}

/// Which avatar the startup flow should load, decided by
/// [`resolve_startup_avatar`].
#[derive(Debug, PartialEq, Eq)]
pub(super) enum StartupAvatar {
    /// `VULVATAR_AUTOSTART_AVATAR` — highest priority (explicit
    /// per-launch override, e.g. kiosk / streaming scripts).
    Env(std::path::PathBuf),
    /// The avatar recorded in the restored project / last-session state.
    Project(std::path::PathBuf),
    None,
}

/// Non-fatal problems found while resolving the startup avatar; the
/// caller turns these into warn logs / toasts.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum StartupAvatarIssue {
    EnvPathMissing(std::path::PathBuf),
    ProjectAvatarMissing(String),
}

/// Pure priority logic for the startup avatar: env override first,
/// then the restored state's `avatar_source_path`. A set-but-missing
/// path at either level is reported as an issue and the next source is
/// considered. Extracted from the GUI init flow so the priority /
/// missing-file matrix is unit-testable without a filesystem.
pub(super) fn resolve_startup_avatar(
    env_path: Option<std::path::PathBuf>,
    project_avatar: Option<&str>,
    exists: &dyn Fn(&std::path::Path) -> bool,
) -> (StartupAvatar, Vec<StartupAvatarIssue>) {
    let mut issues = Vec::new();
    if let Some(p) = env_path {
        if exists(&p) {
            return (StartupAvatar::Env(p), issues);
        }
        issues.push(StartupAvatarIssue::EnvPathMissing(p));
    }
    if let Some(p) = project_avatar {
        let path = std::path::PathBuf::from(p);
        if exists(&path) {
            return (StartupAvatar::Project(path), issues);
        }
        issues.push(StartupAvatarIssue::ProjectAvatarMissing(p.to_string()));
    }
    (StartupAvatar::None, issues)
}

#[cfg(test)]
mod save_policy_tests {
    use super::*;

    fn unique_tmp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "vulvatar_save_policy_{tag}_{}_{:?}",
            std::process::id(),
            std::thread::current().id(),
        ));
        std::fs::create_dir_all(&dir).expect("create tmp dir");
        dir
    }

    #[test]
    fn save_retry_backoff_doubles_and_caps() {
        let mut retry = SaveRetry::new(Instant::now());
        assert_eq!(retry.delay(), Duration::from_millis(250));
        retry.record_failure();
        assert_eq!(retry.delay(), Duration::from_millis(500));
        retry.record_failure();
        assert_eq!(retry.delay(), Duration::from_millis(1000));
        for _ in 0..20 {
            retry.record_failure();
        }
        assert_eq!(retry.delay(), Duration::from_secs(10), "capped at 10 s");
        retry.reset();
        assert_eq!(retry.delay(), Duration::from_millis(250));
    }

    #[test]
    fn new_save_retry_is_immediately_due() {
        // A fresh mutation must be able to save within one base delay,
        // not wait out a max-backoff window.
        let retry = SaveRetry::new(Instant::now());
        assert!(retry.due());
    }

    #[test]
    fn unsaved_sidecar_appends_suffix_without_replacing_extension() {
        let p = std::path::Path::new("C:/scenes/my scene.vvtproj");
        assert_eq!(
            unsaved_sidecar_path(p),
            std::path::PathBuf::from("C:/scenes/my scene.vvtproj.unsaved"),
        );
    }

    #[test]
    fn sidecar_is_newer_requires_existing_strictly_newer_file() {
        let dir = unique_tmp_dir("mtime");
        let project = dir.join("p.vvtproj");
        let sidecar = unsaved_sidecar_path(&project);
        std::fs::write(&project, "project").unwrap();
        // Missing sidecar → false.
        assert!(!sidecar_is_newer(&project, &sidecar));
        // Strictly newer sidecar → true. Push the mtime forward
        // explicitly instead of sleeping across filesystem resolution.
        std::fs::write(&sidecar, "sidecar").unwrap();
        let future = std::time::SystemTime::now() + Duration::from_secs(5);
        let f = std::fs::File::options().write(true).open(&sidecar).unwrap();
        f.set_modified(future).unwrap();
        drop(f);
        assert!(sidecar_is_newer(&project, &sidecar));
        // Older sidecar → false.
        let past = std::time::SystemTime::now() - Duration::from_secs(3600);
        let f = std::fs::File::options().write(true).open(&sidecar).unwrap();
        f.set_modified(past).unwrap();
        drop(f);
        assert!(!sidecar_is_newer(&project, &sidecar));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn autosave_never_touches_the_explicit_project_file() {
        let dir = unique_tmp_dir("autosave");
        let project = dir.join("scene.vvtproj");
        std::fs::write(&project, "ORIGINAL USER FILE").unwrap();

        let mut app = crate::gui::GuiApp::for_test();
        app.project_status.project_path = Some(project.clone());
        // No manual dirty flag anywhere: seed the baseline, make a
        // project-visible change, and let the derived probe detect it.
        app.mark_project_baseline();
        app.transform.position[0] += 1.0;
        app.refresh_project_dirty(Instant::now(), true);
        assert!(
            app.project_status.project_dirty,
            "state change must derive project_dirty without a manual flag",
        );
        app.autosave_tick();

        assert_eq!(
            std::fs::read_to_string(&project).unwrap(),
            "ORIGINAL USER FILE",
            "autosave must not overwrite the explicitly saved project",
        );
        let sidecar = unsaved_sidecar_path(&project);
        assert!(sidecar.exists(), "unsaved changes go to the sidecar");
        assert!(!app.project_status.project_dirty, "autosave landed");
        assert!(
            app.project_status.explicit_file_stale,
            "title-bar dot stays on until an explicit Save",
        );
        // The autosaved state is the new baseline: an unchanged app
        // stays clean on the next probe...
        app.refresh_project_dirty(Instant::now(), true);
        assert!(
            !app.project_status.project_dirty,
            "baseline tracks the write"
        );
        // ...and a further edit re-flags.
        app.transform.position[1] += 0.5;
        app.refresh_project_dirty(Instant::now(), true);
        assert!(
            app.project_status.project_dirty,
            "post-save edits re-derive dirty"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every widget used to pair `.changed()` with a manual
    /// `project_dirty = true`; forgetting one meant that setting never
    /// autosaved. The derived probe kills the class: ANY field of
    /// `to_project_state` flags on change, and reverting before a
    /// write reads clean again.
    #[test]
    fn dirty_is_derived_from_state_comparison_not_manual_flags() {
        let mut app = crate::gui::GuiApp::for_test();
        app.mark_project_baseline();
        app.refresh_project_dirty(Instant::now(), true);
        assert!(
            !app.project_status.project_dirty,
            "untouched state is clean"
        );

        // A sample across unrelated subsystems — transform, tracking
        // toggle, rendering value — all detected by the same probe.
        app.tracking.hand_tracking_enabled = !app.tracking.hand_tracking_enabled;
        app.refresh_project_dirty(Instant::now(), true);
        assert!(app.project_status.project_dirty, "tracking toggle detected");

        // Toggle back before any autosave: clean again (state equals
        // baseline; dirty is a comparison, not an event accumulator).
        app.tracking.hand_tracking_enabled = !app.tracking.hand_tracking_enabled;
        app.refresh_project_dirty(Instant::now(), true);
        assert!(!app.project_status.project_dirty, "revert reads clean");

        app.rendering.bloom_intensity += 0.25;
        app.refresh_project_dirty(Instant::now(), true);
        assert!(
            app.project_status.project_dirty,
            "rendering change detected"
        );
    }

    /// The capture format persists as VALUES; combo positions are
    /// derived at the boundary. Adding a 4K entry at combo slot 0
    /// tomorrow must not retarget existing projects.
    #[test]
    fn capture_format_round_trips_as_values_with_legacy_indices_alongside() {
        let mut app = crate::gui::GuiApp::for_test();
        app.tracking.camera_resolution_index = 2; // combo: 1920×1080
        app.tracking.camera_framerate_index = 1; // combo: 60 fps
        let state = app.to_project_state();
        assert_eq!(
            (
                state.camera_capture_width,
                state.camera_capture_height,
                state.camera_capture_fps
            ),
            (1920, 1080, 60),
            "snapshot carries real values, not combo slots",
        );

        let file = crate::persistence::ProjectFile::from_state(&state);
        assert_eq!(file.tracking.camera_capture_width, Some(1920));
        assert_eq!(
            file.tracking.camera_resolution_index, 2,
            "legacy index still written for downgrade compatibility",
        );

        let back = file.to_state();
        app.apply_project_state(&back);
        assert_eq!(app.tracking.camera_resolution_index, 2);
        assert_eq!(app.tracking.camera_framerate_index, 1);
    }

    /// Files saved before the value fields existed migrate from their
    /// frozen legacy indices exactly once, at load.
    #[test]
    fn legacy_project_without_value_fields_migrates_from_indices() {
        let base = crate::gui::GuiApp::for_test().to_project_state();
        let mut file = crate::persistence::ProjectFile::from_state(&base);
        file.tracking.camera_capture_width = None;
        file.tracking.camera_capture_height = None;
        file.tracking.camera_capture_fps = None;
        file.tracking.camera_resolution_index = 1;
        file.tracking.camera_framerate_index = 1;

        let state = file.to_state();
        assert_eq!(
            (
                state.camera_capture_width,
                state.camera_capture_height,
                state.camera_capture_fps
            ),
            (1280, 720, 60),
        );
    }

    /// Unknown persisted values (a future build's 4K project opened
    /// here) snap to the nearest existing combo entry instead of
    /// panicking or silently landing on slot 0.
    #[test]
    fn unknown_capture_values_snap_to_nearest_combo_entry() {
        assert_eq!(crate::gui::camera_resolution_index_for(3840, 2160), 2);
        assert_eq!(crate::gui::camera_resolution_index_for(1280, 720), 1);
        assert_eq!(crate::gui::camera_resolution_index_for(320, 240), 0);
        assert_eq!(crate::gui::camera_fps_index_for(90), 1);
        assert_eq!(crate::gui::camera_fps_index_for(24), 0);
    }

    #[test]
    fn dirty_probe_is_throttled_but_force_bypasses() {
        let mut app = crate::gui::GuiApp::for_test();
        app.mark_project_baseline();
        let t0 = Instant::now();
        app.project_status.last_dirty_probe = t0;
        app.transform.scale += 0.5;
        // Within the throttle window a non-forced probe is a no-op...
        app.refresh_project_dirty(t0 + Duration::from_millis(10), false);
        assert!(!app.project_status.project_dirty, "throttled probe skips");
        // ...a forced probe (on_exit path) sees it immediately.
        app.refresh_project_dirty(t0 + Duration::from_millis(10), true);
        assert!(app.project_status.project_dirty, "forced probe detects");
    }
}

#[cfg(test)]
mod startup_avatar_tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn always(_: &Path) -> bool {
        true
    }
    fn never(_: &Path) -> bool {
        false
    }

    #[test]
    fn env_override_wins_over_project_avatar() {
        let (choice, issues) =
            resolve_startup_avatar(Some(PathBuf::from("env.vrm")), Some("project.vrm"), &always);
        assert_eq!(choice, StartupAvatar::Env(PathBuf::from("env.vrm")));
        assert!(issues.is_empty());
    }

    #[test]
    fn missing_env_falls_back_to_project_avatar_with_issue() {
        let exists = |p: &Path| p == Path::new("project.vrm");
        let (choice, issues) =
            resolve_startup_avatar(Some(PathBuf::from("env.vrm")), Some("project.vrm"), &exists);
        assert_eq!(choice, StartupAvatar::Project(PathBuf::from("project.vrm")));
        assert_eq!(
            issues,
            vec![StartupAvatarIssue::EnvPathMissing(PathBuf::from("env.vrm"))]
        );
    }

    #[test]
    fn project_avatar_alone_loads_when_present() {
        let (choice, issues) = resolve_startup_avatar(None, Some("project.vrm"), &always);
        assert_eq!(choice, StartupAvatar::Project(PathBuf::from("project.vrm")));
        assert!(issues.is_empty());
    }

    #[test]
    fn all_sources_missing_reports_every_issue() {
        let (choice, issues) =
            resolve_startup_avatar(Some(PathBuf::from("env.vrm")), Some("project.vrm"), &never);
        assert_eq!(choice, StartupAvatar::None);
        assert_eq!(
            issues,
            vec![
                StartupAvatarIssue::EnvPathMissing(PathBuf::from("env.vrm")),
                StartupAvatarIssue::ProjectAvatarMissing("project.vrm".to_string()),
            ]
        );
    }

    #[test]
    fn no_sources_is_quiet() {
        let (choice, issues) = resolve_startup_avatar(None, None, &always);
        assert_eq!(choice, StartupAvatar::None);
        assert!(issues.is_empty());
    }
}
