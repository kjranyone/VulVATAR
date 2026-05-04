//! Toast/notification system. Owns the `Notification` value type, the
//! `push_*` mutators on `GuiApp`, and the per-frame toast renderer.
//!
//! Toasts are drawn on `egui::Order::Tooltip` so they sit above every
//! modal (`Foreground`), the calibration scrim (`Middle`), and ordinary
//! panels — they're read-only status reports and should be visible
//! regardless of what dialog is open.

use std::time::Instant;

use eframe::egui;

use super::GuiApp;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotificationLevel {
    Info,
    Error,
}

impl NotificationLevel {
    /// Lifetime of a toast at this level, in seconds. Errors stay up
    /// 3× longer so the user has time to read them before they fade.
    fn ttl_seconds(self) -> f32 {
        match self {
            NotificationLevel::Info => 5.0,
            NotificationLevel::Error => 15.0,
        }
    }
}

#[derive(Debug)]
pub struct Notification {
    pub message: String,
    pub created: Instant,
    pub level: NotificationLevel,
}

impl GuiApp {
    /// Push an info notification that will auto-dismiss after 5 seconds.
    pub fn push_notification(&mut self, msg: impl Into<String>) {
        self.notifications.push(Notification {
            message: msg.into(),
            created: Instant::now(),
            level: NotificationLevel::Info,
        });
    }

    /// Push an error notification that will auto-dismiss after 15 seconds
    /// with a red-tinted background.
    pub fn push_error_notification(&mut self, msg: impl Into<String>) {
        self.notifications.push(Notification {
            message: msg.into(),
            created: Instant::now(),
            level: NotificationLevel::Error,
        });
    }

    /// Per-frame: expire stale toasts and draw the survivors anchored to
    /// the bottom-right corner.
    pub(super) fn draw_toasts(&mut self, ctx: &egui::Context) {
        self.notifications
            .retain(|n| n.created.elapsed().as_secs_f32() < n.level.ttl_seconds());
        if self.notifications.is_empty() {
            return;
        }
        egui::Area::new(egui::Id::new("notifications"))
            .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-10.0, -40.0))
            // Tooltip ordering puts toasts above every modal
            // (`Order::Foreground`), the calibration scrim
            // (`Order::Middle`), and ordinary panels (default
            // `Order::Middle`). Notifications are read-only
            // status reports — they should always be visible
            // regardless of what dialog is open.
            .order(egui::Order::Tooltip)
            .show(ctx, |ui| {
                for n in &self.notifications {
                    let ttl = n.level.ttl_seconds();
                    let fade_start = ttl - 1.0;
                    let age = n.created.elapsed().as_secs_f32();
                    let alpha = if age > fade_start {
                        ((ttl - age) * 255.0) as u8
                    } else {
                        255u8
                    };
                    let bg = match n.level {
                        NotificationLevel::Info => {
                            egui::Color32::from_rgba_unmultiplied(40, 40, 40, alpha)
                        }
                        NotificationLevel::Error => {
                            egui::Color32::from_rgba_unmultiplied(120, 30, 30, alpha)
                        }
                    };
                    egui::Frame::group(ui.style())
                        .fill(bg)
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new(&n.message).color(
                                egui::Color32::from_rgba_unmultiplied(255, 255, 255, alpha),
                            ));
                        });
                    ui.add_space(2.0);
                }
            });
    }
}
