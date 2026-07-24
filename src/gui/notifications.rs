//! Toast/notification system. Owns the `Notification` value type, the
//! `push_*` mutators on `GuiApp`, and the per-frame toast renderer.
//!
//! Behavioural contract (each point exists because its absence was a
//! reported defect):
//! * Four levels — Info / Success / Warning / Error — coloured from the
//!   theme's container tokens so toasts sit inside the light theme
//!   instead of floating as dark chips.
//! * Duplicate messages merge into one toast with a ×N counter instead
//!   of stacking (a failing autosave used to cover the screen).
//! * At most [`MAX_VISIBLE`] toasts draw at once; the backlog is capped
//!   at [`MAX_QUEUED`], dropping the oldest non-sticky entries.
//! * Click dismisses. Hover freezes the countdown so a toast can't
//!   fade away mid-read.
//! * `sticky` toasts never expire on their own; the producer clears
//!   them via [`GuiApp::dismiss_notifications_matching`] when the
//!   underlying condition resolves (e.g. a save finally succeeding).
//!
//! Toasts are drawn on `egui::Order::Tooltip` so they sit above every
//! modal (`Foreground`), the calibration scrim (`Middle`), and ordinary
//! panels — they're read-only status reports and should be visible
//! regardless of what dialog is open.

use std::time::Instant;

use eframe::egui;

use super::theme::{color, icon, radius, space, typography};
use super::GuiApp;

/// Maximum number of toasts drawn per frame (most recent first).
const MAX_VISIBLE: usize = 5;
/// Hard cap on the queue; oldest non-sticky entries are evicted past this.
const MAX_QUEUED: usize = 32;
/// Toast body width cap — long localized messages wrap instead of
/// stretching across the whole window.
const MAX_WIDTH: f32 = 360.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotificationLevel {
    Info,
    Success,
    Warning,
    Error,
}

impl NotificationLevel {
    /// Lifetime of a toast at this level, in seconds. Errors stay up
    /// 3× longer so the user has time to read them before they fade.
    fn ttl_seconds(self) -> f32 {
        match self {
            NotificationLevel::Info => 5.0,
            NotificationLevel::Success => 5.0,
            NotificationLevel::Warning => 8.0,
            NotificationLevel::Error => 15.0,
        }
    }

    fn palette(self) -> (egui::Color32, egui::Color32) {
        match self {
            NotificationLevel::Info => (color::SURFACE_VARIANT, color::ON_SURFACE),
            NotificationLevel::Success => (color::SUCCESS_CONTAINER, color::ON_SUCCESS_CONTAINER),
            NotificationLevel::Warning => (color::WARNING_CONTAINER, color::ON_WARNING_CONTAINER),
            NotificationLevel::Error => (color::ERROR_CONTAINER, color::ON_ERROR_CONTAINER),
        }
    }
}

#[derive(Debug)]
pub struct Notification {
    pub message: String,
    /// Creation / last-refresh time; hovering refreshes it so the
    /// fade countdown restarts while the pointer is over the toast.
    pub created: Instant,
    pub level: NotificationLevel,
    /// Number of identical pushes merged into this toast (≥ 1).
    pub count: u32,
    /// Sticky toasts never expire; they're dismissed by click or by
    /// [`GuiApp::dismiss_notifications_matching`].
    pub sticky: bool,
}

impl GuiApp {
    fn push_toast(&mut self, msg: String, level: NotificationLevel, sticky: bool) {
        // Merge with an existing toast carrying the same message +
        // level: bump the counter and refresh the countdown instead of
        // stacking a duplicate.
        if let Some(existing) = self
            .notifications
            .iter_mut()
            .find(|n| n.level == level && n.message == msg)
        {
            existing.count = existing.count.saturating_add(1);
            existing.created = Instant::now();
            existing.sticky |= sticky;
            return;
        }
        self.notifications.push(Notification {
            message: msg,
            created: Instant::now(),
            level,
            count: 1,
            sticky,
        });
        // Backlog cap: evict the oldest non-sticky entry first so a
        // burst can't grow the queue without bound, while sticky
        // failures survive the eviction.
        while self.notifications.len() > MAX_QUEUED {
            if let Some(idx) = self.notifications.iter().position(|n| !n.sticky) {
                self.notifications.remove(idx);
            } else {
                self.notifications.remove(0);
            }
        }
    }

    /// Push an info notification that auto-dismisses.
    pub fn push_notification(&mut self, msg: impl Into<String>) {
        self.push_toast(msg.into(), NotificationLevel::Info, false);
    }

    /// Push a success notification (green container) that auto-dismisses.
    pub fn push_success_notification(&mut self, msg: impl Into<String>) {
        self.push_toast(msg.into(), NotificationLevel::Success, false);
    }

    /// Push a warning notification (amber container) that auto-dismisses.
    pub fn push_warning_notification(&mut self, msg: impl Into<String>) {
        self.push_toast(msg.into(), NotificationLevel::Warning, false);
    }

    /// Push an error notification (red container) with a longer TTL.
    pub fn push_error_notification(&mut self, msg: impl Into<String>) {
        self.push_toast(msg.into(), NotificationLevel::Error, false);
    }

    /// Push an error that stays until clicked or explicitly cleared
    /// via [`Self::dismiss_notifications_matching`] — for conditions
    /// that persist (e.g. a save that keeps failing) where a fading
    /// toast would let the user miss that the problem is ongoing.
    pub fn push_sticky_error_notification(&mut self, msg: impl Into<String>) {
        self.push_toast(msg.into(), NotificationLevel::Error, true);
    }

    /// Remove every toast whose message equals `msg`. Producers of
    /// sticky errors call this when the underlying condition clears.
    pub fn dismiss_notifications_matching(&mut self, msg: &str) {
        self.notifications.retain(|n| n.message != msg);
    }

    /// Per-frame: expire stale toasts and draw the survivors anchored to
    /// the bottom-right corner. Click dismisses; hover refreshes the
    /// countdown.
    pub(super) fn draw_toasts(&mut self, ctx: &egui::Context) {
        self.notifications
            .retain(|n| n.sticky || n.created.elapsed().as_secs_f32() < n.level.ttl_seconds());
        if self.notifications.is_empty() {
            return;
        }
        let mut dismissed: Vec<usize> = Vec::new();
        let visible_from = self.notifications.len().saturating_sub(MAX_VISIBLE);
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
                ui.set_max_width(MAX_WIDTH);
                for (idx, n) in self.notifications.iter_mut().enumerate().skip(visible_from) {
                    let ttl = n.level.ttl_seconds();
                    let fade_start = ttl - 1.0;
                    let age = n.created.elapsed().as_secs_f32();
                    let alpha = if !n.sticky && age > fade_start {
                        ((ttl - age).max(0.0) * 255.0).min(255.0) as u8
                    } else {
                        255u8
                    };
                    let (bg, fg) = n.level.palette();
                    let bg = color::with_alpha(bg, alpha);
                    let fg = color::with_alpha(fg, alpha);
                    let text = if n.count > 1 {
                        format!("{} (×{})", n.message, n.count)
                    } else {
                        n.message.clone()
                    };
                    let inner = egui::Frame::none()
                        .fill(bg)
                        .rounding(egui::Rounding::same(radius::SM))
                        .inner_margin(egui::Margin::symmetric(space::MD * 0.75, space::SM))
                        .show(ui, |ui| {
                            ui.set_max_width(MAX_WIDTH - space::MD * 1.5);
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(icon::STATUS_DOT.to_string())
                                        .font(typography::icon(10.0))
                                        .color(fg),
                                );
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(text)
                                            .font(typography::body())
                                            .color(fg),
                                    )
                                    .wrap(),
                                );
                            });
                        });
                    let resp = ui.interact(
                        inner.response.rect,
                        egui::Id::new(("toast", idx)),
                        egui::Sense::click(),
                    );
                    if resp.hovered() {
                        // Freeze the countdown while the user reads.
                        n.created = Instant::now();
                    }
                    if resp.clicked() {
                        dismissed.push(idx);
                    }
                    ui.add_space(2.0);
                }
            });
        for idx in dismissed.into_iter().rev() {
            self.notifications.remove(idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_pushes_merge_into_one_toast_with_a_counter() {
        let mut app = GuiApp::for_test();
        app.push_error_notification("disk full");
        app.push_error_notification("disk full");
        app.push_error_notification("disk full");
        assert_eq!(app.notifications.len(), 1);
        assert_eq!(app.notifications[0].count, 3);
        // A different level with the same text stays separate.
        app.push_notification("disk full");
        assert_eq!(app.notifications.len(), 2);
    }

    #[test]
    fn queue_is_capped_evicting_oldest_non_sticky_first() {
        let mut app = GuiApp::for_test();
        app.push_sticky_error_notification("sticky failure");
        for i in 0..(MAX_QUEUED + 10) {
            app.push_notification(format!("info {i}"));
        }
        assert!(app.notifications.len() <= MAX_QUEUED);
        assert!(
            app.notifications.iter().any(|n| n.sticky),
            "sticky toast must survive eviction",
        );
    }

    #[test]
    fn dismiss_matching_clears_sticky_toasts() {
        let mut app = GuiApp::for_test();
        app.push_sticky_error_notification("save failed: E");
        app.push_notification("unrelated");
        app.dismiss_notifications_matching("save failed: E");
        assert_eq!(app.notifications.len(), 1);
        assert_eq!(app.notifications[0].message, "unrelated");
    }
}
