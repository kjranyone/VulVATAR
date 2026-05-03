use std::collections::HashMap;

use eframe::egui;

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
        self.bindings.insert(
            HotkeyAction::ToggleClothSimulation,
            KeyBinding::ctrl(egui::Key::C),
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
        self.bindings
            .get(&action)
            .map(|b| b.just_pressed(ctx))
            .unwrap_or(false)
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
