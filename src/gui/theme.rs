//! Material 3-flavoured design tokens for the GUI.
//!
//! All hand-picked colors that used to litter inspector / library /
//! status code should resolve through these tokens so the app has one
//! point of control. Calibrated against the light theme mockup
//! (purple accent on near-white surfaces) — switching to a dark
//! variant later means swapping the constants here, not chasing every
//! `Color32::from_rgb(...)` call site.
//!
//! Token vocabulary follows MD3 nomenclature so designers can map
//! straight from the spec:
//!   * `primary` / `on_primary` / `primary_container` / `on_primary_container`
//!   * `surface` / `surface_dim` / `surface_bright` / `surface_variant`
//!   * `on_surface` / `on_surface_variant` / `on_surface_muted`
//!   * `outline` / `outline_variant`
//!
//! Spacing follows MD3's 4 dp grid; radii match the mockup's rounded
//! cards (12 px) and pill chips (999 px).

use eframe::egui::{self, Color32, FontFamily, FontId, Margin, Rounding, Shadow, Stroke, Vec2};

pub mod color {
    use super::Color32;

    // ── Brand / accent ────────────────────────────────────────────
    pub const PRIMARY: Color32 = Color32::from_rgb(124, 92, 255);
    pub const PRIMARY_HOVER: Color32 = Color32::from_rgb(108, 76, 240);
    pub const ON_PRIMARY: Color32 = Color32::WHITE;
    /// Light-purple chip used for active mode-nav items, selected sort
    /// chips, and any "you are here" affordance.
    pub const PRIMARY_CONTAINER: Color32 = Color32::from_rgb(232, 222, 248);
    pub const ON_PRIMARY_CONTAINER: Color32 = Color32::from_rgb(60, 35, 130);

    // ── Surfaces (page backgrounds, cards, inputs) ────────────────
    /// Card background — the brightest surface, sits on top of
    /// `SURFACE_DIM` and provides the "elevated" reading area.
    pub const SURFACE: Color32 = Color32::from_rgb(255, 255, 255);
    /// Sidebar / inspector chrome background — a half-step darker than
    /// `SURFACE` so cards visually pop without needing real shadows.
    pub const SURFACE_DIM: Color32 = Color32::from_rgb(247, 245, 250);
    /// App window background (between sidebar and viewport).
    pub const SURFACE_BRIGHT: Color32 = Color32::from_rgb(252, 251, 254);
    /// Form-input background (text fields, drop-downs).
    pub const SURFACE_VARIANT: Color32 = Color32::from_rgb(240, 238, 244);

    // ── Text on surfaces ─────────────────────────────────────────
    pub const ON_SURFACE: Color32 = Color32::from_rgb(28, 27, 31);
    pub const ON_SURFACE_VARIANT: Color32 = Color32::from_rgb(99, 91, 111);
    pub const ON_SURFACE_MUTED: Color32 = Color32::from_rgb(140, 134, 152);

    // ── Status (tracking / errors / warnings) ─────────────────────
    pub const ERROR: Color32 = Color32::from_rgb(220, 60, 60);
    pub const ERROR_CONTAINER: Color32 = Color32::from_rgb(252, 232, 232);
    pub const ON_ERROR_CONTAINER: Color32 = Color32::from_rgb(180, 40, 40);
    pub const SUCCESS: Color32 = Color32::from_rgb(46, 160, 67);
    pub const WARNING: Color32 = Color32::from_rgb(220, 140, 30);

    // ── Outlines / dividers ───────────────────────────────────────
    pub const OUTLINE: Color32 = Color32::from_rgb(218, 213, 226);
    pub const OUTLINE_VARIANT: Color32 = Color32::from_rgb(232, 228, 240);

    // ── Viewport background (always dark for avatar contrast) ─────
    /// 3D viewport stays dark even in light theme — avatars and
    /// streaming preview read better against a deep background.
    pub const VIEWPORT_BG: Color32 = Color32::from_rgb(20, 20, 26);
    /// Cool-blue accent used to outline overlay panels inside the
    /// viewport (camera PIP, calibration preview pane). Distinct from
    /// the brand purple so the viewport's own affordances don't read
    /// as just-another-button.
    pub const VIEWPORT_OVERLAY_OUTLINE: Color32 = Color32::from_rgb(60, 130, 200);
}

pub mod space {
    pub const XS: f32 = 4.0;
    pub const SM: f32 = 8.0;
    pub const MD: f32 = 16.0;
    pub const LG: f32 = 24.0;
    pub const XL: f32 = 32.0;
}

pub mod radius {
    pub const SM: f32 = 6.0;
    pub const MD: f32 = 12.0;
    pub const LG: f32 = 16.0;
    pub const PILL: f32 = 24.0;
}

/// Font family name registered in [`super::build_font_definitions`] for
/// Material Symbols glyph lookup. Use [`typography::icon`] to construct
/// a `FontId` rather than referencing this directly.
pub const ICON_FAMILY: &str = "material_symbols";

pub mod typography {
    use super::{FontFamily, FontId, ICON_FAMILY};

    pub fn heading() -> FontId {
        FontId::new(20.0, FontFamily::Proportional)
    }
    pub fn title() -> FontId {
        FontId::new(15.0, FontFamily::Proportional)
    }
    pub fn body() -> FontId {
        FontId::new(13.0, FontFamily::Proportional)
    }
    pub fn label() -> FontId {
        FontId::new(12.0, FontFamily::Proportional)
    }
    pub fn caption() -> FontId {
        FontId::new(11.0, FontFamily::Proportional)
    }
    pub fn icon(size: f32) -> FontId {
        FontId::new(size, FontFamily::Name(ICON_FAMILY.into()))
    }
}

/// Material Symbols Rounded code points used by the GUI. New icons go
/// here so the lookup table stays in one place; reference by constant
/// at call sites (e.g. `t!("inspector.load")` paired with `icon::PLAY_ARROW`).
///
/// Code points come from the official metadata at
/// `https://fonts.google.com/icons` — search the icon name there to
/// confirm the codepoint before adding.
pub mod icon {
    // Mode nav
    pub const AVATAR: char = '\u{f8d6}'; // person_4
    pub const PREVIEW: char = '\u{e8f4}'; // visibility
    pub const TRACKING_SETUP: char = '\u{e8aa}'; // track_changes
    pub const RENDERING: char = '\u{e40a}'; // palette
    pub const OUTPUT: char = '\u{ebbe}'; // output
    pub const CLOTH_AUTHORING: char = '\u{f19e}'; // checkroom
    pub const SETTINGS: char = '\u{e8b8}'; // settings
    pub const HIDE_PANEL: char = '\u{e408}'; // chevron_left

    // Top-bar actions
    pub const MENU: char = '\u{e5d2}'; // menu (hamburger)
    pub const FOLDER_OPEN: char = '\u{e2c8}'; // folder_open
    pub const SAVE: char = '\u{e161}'; // save
    pub const OPEN_OVERLAY: char = '\u{e89e}'; // layers
    pub const SAVE_OVERLAY: char = '\u{eb4f}'; // save_alt
    pub const PAUSE: char = '\u{e034}'; // pause
    pub const PLAY: char = '\u{e037}'; // play_arrow
    pub const PROFILE: char = '\u{e7fd}'; // person
    pub const MORE_VERT: char = '\u{e5d4}'; // more_vert

    // Common actions
    pub const REFRESH: char = '\u{e5d5}'; // refresh
    pub const SEARCH: char = '\u{e8b6}'; // search
    pub const ADD: char = '\u{e145}'; // add
    pub const REMOVE: char = '\u{e15b}'; // remove
    pub const DELETE: char = '\u{e872}'; // delete
    pub const CLOSE: char = '\u{e5cd}'; // close
    pub const FILTER: char = '\u{ef4f}'; // filter_alt
    pub const FAVORITE: char = '\u{e838}'; // star (filled outline)
    pub const FAVORITE_FILLED: char = '\u{e838}'; // star
    pub const FAVORITE_BORDER: char = '\u{e83a}'; // star_border

    // Status indicators
    pub const STATUS_DOT: char = '\u{e061}'; // fiber_manual_record (small filled circle)
    pub const FULLSCREEN: char = '\u{e5d0}'; // fullscreen
    pub const HOME: char = '\u{e88a}'; // home
    pub const ZOOM_IN: char = '\u{e8ff}'; // zoom_in
    pub const ZOOM_OUT: char = '\u{e900}'; // zoom_out
    pub const HISTORY: char = '\u{e889}'; // history
}

/// Apply the theme-wide [`egui::Visuals`] + spacing overrides. Call
/// once at GUI startup after fonts are installed; egui re-uses the
/// stored visuals for every subsequent paint.
pub fn apply(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::light();

    // ── Surfaces ──────────────────────────────────────────────────
    visuals.window_fill = color::SURFACE;
    visuals.panel_fill = color::SURFACE_DIM;
    visuals.faint_bg_color = color::SURFACE_VARIANT;
    visuals.extreme_bg_color = color::SURFACE_VARIANT;
    visuals.code_bg_color = color::SURFACE_VARIANT;

    // ── Text ──────────────────────────────────────────────────────
    visuals.override_text_color = Some(color::ON_SURFACE);
    visuals.hyperlink_color = color::PRIMARY;

    // ── Selection (text edits, list rows) ────────────────────────
    visuals.selection.bg_fill = color::PRIMARY_CONTAINER;
    visuals.selection.stroke = Stroke::new(1.0, color::ON_PRIMARY_CONTAINER);

    // ── Window decorations (popups / context menus) ──────────────
    visuals.window_rounding = Rounding::same(radius::MD);
    visuals.window_stroke = Stroke::new(1.0, color::OUTLINE);
    visuals.window_shadow = Shadow {
        offset: Vec2::new(0.0, 4.0),
        blur: 16.0,
        spread: 0.0,
        color: Color32::from_rgba_unmultiplied(60, 50, 90, 30),
    };
    visuals.popup_shadow = Shadow {
        offset: Vec2::new(0.0, 2.0),
        blur: 8.0,
        spread: 0.0,
        color: Color32::from_rgba_unmultiplied(60, 50, 90, 24),
    };
    visuals.menu_rounding = Rounding::same(radius::SM);

    // ── Widget states ────────────────────────────────────────────
    let widgets = &mut visuals.widgets;
    let widget_round = Rounding::same(radius::SM);

    widgets.noninteractive.bg_fill = color::SURFACE;
    widgets.noninteractive.weak_bg_fill = color::SURFACE;
    widgets.noninteractive.bg_stroke = Stroke::new(1.0, color::OUTLINE_VARIANT);
    widgets.noninteractive.fg_stroke = Stroke::new(1.0, color::ON_SURFACE);
    widgets.noninteractive.rounding = widget_round;

    widgets.inactive.bg_fill = color::SURFACE_VARIANT;
    widgets.inactive.weak_bg_fill = color::SURFACE_VARIANT;
    widgets.inactive.bg_stroke = Stroke::new(1.0, color::OUTLINE);
    widgets.inactive.fg_stroke = Stroke::new(1.0, color::ON_SURFACE);
    widgets.inactive.rounding = widget_round;

    widgets.hovered.bg_fill = color::SURFACE;
    widgets.hovered.weak_bg_fill = color::SURFACE;
    widgets.hovered.bg_stroke = Stroke::new(1.0, color::PRIMARY);
    widgets.hovered.fg_stroke = Stroke::new(1.5, color::PRIMARY);
    widgets.hovered.rounding = widget_round;

    widgets.active.bg_fill = color::PRIMARY_CONTAINER;
    widgets.active.weak_bg_fill = color::PRIMARY_CONTAINER;
    widgets.active.bg_stroke = Stroke::new(1.0, color::PRIMARY);
    widgets.active.fg_stroke = Stroke::new(1.5, color::ON_PRIMARY_CONTAINER);
    widgets.active.rounding = widget_round;

    widgets.open.bg_fill = color::PRIMARY_CONTAINER;
    widgets.open.weak_bg_fill = color::PRIMARY_CONTAINER;
    widgets.open.bg_stroke = Stroke::new(1.0, color::PRIMARY);
    widgets.open.fg_stroke = Stroke::new(1.0, color::ON_PRIMARY_CONTAINER);
    widgets.open.rounding = widget_round;

    ctx.set_visuals(visuals);

    // ── Spacing ──────────────────────────────────────────────────
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(space::SM, space::SM);
    style.spacing.button_padding = Vec2::new(space::MD * 0.75, space::SM * 0.75);
    style.spacing.menu_margin = Margin::same(space::SM);
    style.spacing.window_margin = Margin::same(space::MD);
    style.spacing.indent = space::MD;
    style.spacing.interact_size = Vec2::new(40.0, 28.0);
    ctx.set_style(style);
}
