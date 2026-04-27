#![allow(dead_code)]
use std::collections::HashMap;

use eframe::egui;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HotkeyAction {
    TogglePause,
    ToggleTracking,
    ToggleCloth,
    ResetPose,
    ResetCamera,
    SwitchModePreview,
    SwitchModeAuthoring,
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
        self.bindings
            .insert(HotkeyAction::ToggleTracking, KeyBinding::ctrl(egui::Key::T));
        self.bindings
            .insert(HotkeyAction::ToggleCloth, KeyBinding::ctrl(egui::Key::C));
        self.bindings.insert(
            HotkeyAction::ResetPose,
            KeyBinding::ctrl_shift(egui::Key::R),
        );
        self.bindings
            .insert(HotkeyAction::ResetCamera, KeyBinding::key(egui::Key::Home));
        self.bindings.insert(
            HotkeyAction::SwitchModePreview,
            KeyBinding::key(egui::Key::F5),
        );
        self.bindings.insert(
            HotkeyAction::SwitchModeAuthoring,
            KeyBinding::key(egui::Key::F6),
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
        egui::Key::F5 => "F5",
        egui::Key::F6 => "F6",
        egui::Key::S => "S",
        egui::Key::T => "T",
        egui::Key::C => "C",
        egui::Key::R => "R",
        egui::Key::O => "O",
        _ => "?",
    }
}
