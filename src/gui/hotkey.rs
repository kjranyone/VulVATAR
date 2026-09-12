use std::collections::HashMap;

use eframe::egui;

use crate::t;

use super::{AppMode, CameraOrbitState, GuiApp};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HotkeyAction {
    TogglePause,
    /// Ctrl+T flips the `tracking_enabled` checkbox in the Tracking
    /// inspector. Renamed from `ToggleTracking` so the name doesn't
    /// imply it switches *to* the Tracking mode (use `SwitchModeTracking`
    /// — F3 — for that).
    ToggleTrackingEnabled,
    /// Ctrl+C toggles cloth simulation rendering. Renamed from
    /// `ToggleCloth` for the same reason as above.
    ToggleClothSimulation,
    ResetPose,
    ResetCamera,
    /// Per-mode F1..F7 nav, ordered to match `AppMode::ALL`. Older
    /// builds had only F5/F6 (Preview / ClothAuthoring); the rest
    /// of the modes were unreachable by keyboard, which the
    /// topology audit flagged as a discoverability gap. F5/F6 are
    /// preserved for muscle-memory continuity.
    SwitchModeAvatar,
    SwitchModePreview,
    SwitchModeTracking,
    SwitchModeRendering,
    SwitchModeOutput,
    SwitchModeAuthoring,
    SwitchModeSettings,
    SaveProject,
    LoadAvatar,
}

impl HotkeyAction {
    /// Presentation order for the read-only shortcut list in Settings.
    /// This is the single enumeration point: `label_key`'s exhaustive
    /// match means a new action fails to compile until it gets a
    /// locale key, and this list is what the Settings pane iterates —
    /// keep it in sync when adding an action (the `entries_cover_
    /// every_action` test counts variants via `label_key`).
    pub const ALL: [HotkeyAction; 14] = [
        HotkeyAction::TogglePause,
        HotkeyAction::ToggleTrackingEnabled,
        HotkeyAction::ToggleClothSimulation,
        HotkeyAction::ResetPose,
        HotkeyAction::ResetCamera,
        HotkeyAction::SaveProject,
        HotkeyAction::LoadAvatar,
        HotkeyAction::SwitchModeAvatar,
        HotkeyAction::SwitchModeTracking,
        HotkeyAction::SwitchModeRendering,
        HotkeyAction::SwitchModeOutput,
        HotkeyAction::SwitchModePreview,
        HotkeyAction::SwitchModeAuthoring,
        HotkeyAction::SwitchModeSettings,
    ];

    /// Locale key describing what the action does — rendered next to
    /// `HotkeyMap::label_for` in the Settings shortcut list.
    pub fn label_key(self) -> &'static str {
        match self {
            HotkeyAction::TogglePause => "hotkey.toggle_pause",
            HotkeyAction::ToggleTrackingEnabled => "hotkey.toggle_tracking",
            HotkeyAction::ToggleClothSimulation => "hotkey.toggle_cloth",
            HotkeyAction::ResetPose => "hotkey.reset_pose",
            HotkeyAction::ResetCamera => "hotkey.reset_camera",
            HotkeyAction::SwitchModeAvatar => "hotkey.mode_avatar",
            HotkeyAction::SwitchModePreview => "hotkey.mode_preview",
            HotkeyAction::SwitchModeTracking => "hotkey.mode_tracking",
            HotkeyAction::SwitchModeRendering => "hotkey.mode_rendering",
            HotkeyAction::SwitchModeOutput => "hotkey.mode_output",
            HotkeyAction::SwitchModeAuthoring => "hotkey.mode_authoring",
            HotkeyAction::SwitchModeSettings => "hotkey.mode_settings",
            HotkeyAction::SaveProject => "hotkey.save_project",
            HotkeyAction::LoadAvatar => "hotkey.load_avatar",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeyBinding {
    pub key: egui::Key,
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
}

impl KeyBinding {
    pub fn key(key: egui::Key) -> Self {
        Self {
            key,
            ctrl: false,
            shift: false,
            alt: false,
        }
    }

    pub fn ctrl(key: egui::Key) -> Self {
        Self {
            key,
            ctrl: true,
            shift: false,
            alt: false,
        }
    }

    pub fn ctrl_shift(key: egui::Key) -> Self {
        Self {
            key,
            ctrl: true,
            shift: true,
            alt: false,
        }
    }

    pub fn just_pressed(&self, ctx: &egui::Context) -> bool {
        ctx.input(|i| {
            if !i.key_pressed(self.key) {
                return false;
            }
            let mods = &i.modifiers;
            mods.ctrl == self.ctrl && mods.shift == self.shift && mods.alt == self.alt
        })
    }
}

pub struct HotkeyMap {
    bindings: HashMap<HotkeyAction, KeyBinding>,
}

impl Default for HotkeyMap {
    fn default() -> Self {
        Self::new()
    }
}

impl HotkeyMap {
    pub fn new() -> Self {
        let mut map = Self {
            bindings: HashMap::new(),
        };
        map.set_defaults();
        map
    }

    fn set_defaults(&mut self) {
        self.bindings
            .insert(HotkeyAction::TogglePause, KeyBinding::key(egui::Key::Space));
        self.bindings.insert(
            HotkeyAction::ToggleTrackingEnabled,
            KeyBinding::ctrl(egui::Key::T),
        );
        // Ctrl+Shift+C, NOT Ctrl+C: plain Ctrl+C is universal muscle
        // memory for "copy" — binding it here silently toggled the
        // cloth sim whenever the user reflexively pressed copy.
        self.bindings.insert(
            HotkeyAction::ToggleClothSimulation,
            KeyBinding::ctrl_shift(egui::Key::C),
        );
        self.bindings.insert(
            HotkeyAction::ResetPose,
            KeyBinding::ctrl_shift(egui::Key::R),
        );
        self.bindings
            .insert(HotkeyAction::ResetCamera, KeyBinding::key(egui::Key::Home));
        // Per-mode nav — F1..F7 paralleling AppMode::ALL. F5
        // (Preview) and F6 (Authoring) were the original two
        // bindings; keep them at the same key so existing muscle
        // memory survives. The remaining five fill in around them.
        self.bindings.insert(
            HotkeyAction::SwitchModeAvatar,
            KeyBinding::key(egui::Key::F1),
        );
        self.bindings.insert(
            HotkeyAction::SwitchModePreview,
            KeyBinding::key(egui::Key::F5),
        );
        self.bindings.insert(
            HotkeyAction::SwitchModeTracking,
            KeyBinding::key(egui::Key::F2),
        );
        self.bindings.insert(
            HotkeyAction::SwitchModeRendering,
            KeyBinding::key(egui::Key::F3),
        );
        self.bindings.insert(
            HotkeyAction::SwitchModeOutput,
            KeyBinding::key(egui::Key::F4),
        );
        self.bindings.insert(
            HotkeyAction::SwitchModeAuthoring,
            KeyBinding::key(egui::Key::F6),
        );
        self.bindings.insert(
            HotkeyAction::SwitchModeSettings,
            KeyBinding::key(egui::Key::F7),
        );
        self.bindings
            .insert(HotkeyAction::SaveProject, KeyBinding::ctrl(egui::Key::S));
        self.bindings
            .insert(HotkeyAction::LoadAvatar, KeyBinding::ctrl(egui::Key::O));
    }

    pub fn bind(&mut self, action: HotkeyAction, binding: KeyBinding) {
        self.bindings.insert(action, binding);
    }

    pub fn check(&self, action: HotkeyAction, ctx: &egui::Context) -> bool {
        // Suppress every binding while a text widget owns keyboard focus.
        // Without this, Ctrl+S in the rename / preset-name / scene preset
        // text inputs saves the *project* instead of accepting the
        // user's text — and an F-key press in a search box jumps the
        // mode nav out from under the typist. egui's
        // `wants_keyboard_input` returns true exactly when a focused
        // widget (TextEdit, slider drag, color picker text entry) is
        // claiming the next key event, which is the right scope.
        if ctx.wants_keyboard_input() {
            return false;
        }
        self.bindings
            .get(&action)
            .map(|b| b.just_pressed(ctx))
            .unwrap_or(false)
    }

    /// Every action paired with its human-readable key label, in the
    /// stable [`HotkeyAction::ALL`] presentation order. Drives the
    /// read-only shortcut list in the Settings pane — generated from
    /// the live bindings so the list can never drift from reality.
    pub fn entries(&self) -> Vec<(HotkeyAction, String)> {
        HotkeyAction::ALL
            .iter()
            .map(|a| (*a, self.label_for(*a)))
            .collect()
    }

    pub fn label_for(&self, action: HotkeyAction) -> String {
        if let Some(binding) = self.bindings.get(&action) {
            let mut parts = Vec::new();
            if binding.ctrl {
                parts.push("Ctrl");
            }
            if binding.shift {
                parts.push("Shift");
            }
            if binding.alt {
                parts.push("Alt");
            }
            parts.push(key_name(binding.key));
            parts.join("+")
        } else {
            "unbound".to_string()
        }
    }
}

fn key_name(key: egui::Key) -> &'static str {
    match key {
        egui::Key::Space => "Space",
        egui::Key::Home => "Home",
        egui::Key::F1 => "F1",
        egui::Key::F2 => "F2",
        egui::Key::F3 => "F3",
        egui::Key::F4 => "F4",
        egui::Key::F5 => "F5",
        egui::Key::F6 => "F6",
        egui::Key::F7 => "F7",
        egui::Key::S => "S",
        egui::Key::T => "T",
        egui::Key::C => "C",
        egui::Key::R => "R",
        egui::Key::O => "O",
        _ => "?",
    }
}

#[cfg(test)]
mod shortcut_list_tests {
    use super::*;

    #[test]
    fn entries_cover_every_action_with_a_real_key_label() {
        let map = HotkeyMap::new();
        let entries = map.entries();
        assert_eq!(
            entries.len(),
            HotkeyAction::ALL.len(),
            "entries() must enumerate the full presentation list"
        );
        for (action, label) in &entries {
            assert_ne!(
                label, "unbound",
                "{action:?} has no default binding — the Settings list \
                 would show a dead row"
            );
            assert!(
                !label.contains('?'),
                "{action:?} renders as '{label}' — its key is missing \
                 from key_name()"
            );
            // Exhaustive-match guarantee: every listed action resolves
            // to a locale key (compile-time), and the key is non-empty.
            assert!(!action.label_key().is_empty());
        }
    }
}

impl GuiApp {
    /// Per-frame hotkey dispatcher: read the egui input, fire the
    /// matching `HotkeyAction`, and apply its GUI-side effect (toggle a
    /// flag, switch mode, save project, open the avatar picker, etc.).
    /// Called once near the top of `update()` before any UI is drawn.
    pub(super) fn process_hotkeys(&mut self, ctx: &egui::Context) {
        // Space doubles as "activate the focused button" in egui.
        // When any widget holds focus, let the widget have it —
        // otherwise pressing a focused button also flips pause.
        let widget_focused = ctx.memory(|m| m.focused().is_some());
        if !widget_focused && self.hotkeys.check(HotkeyAction::TogglePause, ctx) {
            self.runtime_status.paused = !self.runtime_status.paused;
            self.push_notification(if self.runtime_status.paused {
                t!("toast.paused")
            } else {
                t!("toast.resumed")
            });
        }
        // Every state-changing hotkey below announces itself with a
        // toast: an invisible global shortcut that silently flips
        // tracking or cloth physics reads as "the app broke".
        if self.hotkeys.check(HotkeyAction::ToggleTrackingEnabled, ctx) {
            self.tracking.toggle_tracking = !self.tracking.toggle_tracking;
            self.push_notification(if self.tracking.toggle_tracking {
                t!("toast.tracking_on")
            } else {
                t!("toast.tracking_off")
            });
        }
        if self.hotkeys.check(HotkeyAction::ToggleClothSimulation, ctx) {
            self.rendering.toggle_cloth = !self.rendering.toggle_cloth;
            self.push_notification(if self.rendering.toggle_cloth {
                t!("toast.cloth_sim_on")
            } else {
                t!("toast.cloth_sim_off")
            });
        }
        if self.hotkeys.check(HotkeyAction::ResetPose, ctx) {
            self.transform.position = [0.0, 0.0, 0.0];
            self.transform.rotation = [0.0, 0.0, 0.0];
            self.transform.scale = 1.0;
            self.push_notification(t!("toast.pose_reset"));
        }
        if self.hotkeys.check(HotkeyAction::ResetCamera, ctx) {
            self.camera_orbit = CameraOrbitState {
                yaw_deg: 0.0,
                pitch_deg: 0.0,
                pan: [0.0, 0.0],
                distance: 5.0,
                target_distance: 5.0,
            };
            self.push_notification(t!("toast.camera_reset"));
        }
        // Per-mode F-key nav. Listed in the same order as
        // `AppMode::ALL` so the binding-to-mode mapping reads
        // straight off the enum without surprises.
        for (action, mode) in [
            (HotkeyAction::SwitchModeAvatar, AppMode::Avatar),
            (HotkeyAction::SwitchModePreview, AppMode::Preview),
            (HotkeyAction::SwitchModeTracking, AppMode::TrackingSetup),
            (HotkeyAction::SwitchModeRendering, AppMode::Rendering),
            (HotkeyAction::SwitchModeOutput, AppMode::Output),
            (HotkeyAction::SwitchModeAuthoring, AppMode::ClothAuthoring),
            (HotkeyAction::SwitchModeSettings, AppMode::Settings),
        ] {
            if self.hotkeys.check(action, ctx) {
                self.mode = mode;
                self.inspector_open = true;
            }
        }
        // Ctrl+S routes through the exact same function as the toolbar
        // Save button — including the Save As fallback when no project
        // path is set. The two entrances used to diverge (the hotkey
        // dead-ended on a "no project path" toast and never recorded
        // `last_project_path` for the next-launch reopen).
        if self.hotkeys.check(HotkeyAction::SaveProject, ctx) {
            super::top_bar::save_project(self);
        }
        if self.hotkeys.check(HotkeyAction::LoadAvatar, ctx) {
            super::top_bar::request_load_avatar_dialog(self);
        }
    }
}
