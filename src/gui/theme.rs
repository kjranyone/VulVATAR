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
    pub const SUCCESS_CONTAINER: Color32 = Color32::from_rgb(226, 244, 231);
    pub const ON_SUCCESS_CONTAINER: Color32 = Color32::from_rgb(25, 105, 50);
    pub const WARNING: Color32 = Color32::from_rgb(220, 140, 30);
    pub const WARNING_CONTAINER: Color32 = Color32::from_rgb(252, 238, 216);
    pub const ON_WARNING_CONTAINER: Color32 = Color32::from_rgb(150, 90, 10);

    // ── Outlines / dividers ───────────────────────────────────────
    pub const OUTLINE_VARIANT: Color32 = Color32::from_rgb(232, 228, 240);

    // ── Viewport background (always dark for avatar contrast) ─────
    /// 3D viewport stays dark even in light theme — avatars and
    /// streaming preview read better against a deep background.
    pub const VIEWPORT_BG: Color32 = Color32::from_rgb(20, 20, 26);

    /// MD3 state layer: blend `on` over `base` at 8% — the hover
    /// treatment for every tonal surface (buttons, chips, rows).
    /// Interactive emphasis in this GUI is expressed through fills,
    /// not hairline strokes: egui rasterises strokes as anti-aliased
    /// meshes with no pixel snapping, so a 1 px outline renders with
    /// visibly uneven thickness depending on sub-pixel phase and DPI
    /// scale. Fills have no such failure mode.
    pub fn state_layer(base: Color32, on: Color32) -> Color32 {
        mix(base, on, 0.08)
    }

    /// Like [`state_layer`] but at the 12% pressed level.
    pub fn pressed_layer(base: Color32, on: Color32) -> Color32 {
        mix(base, on, 0.12)
    }

    /// `c` at opacity `a` — for translucent hover overlays on
    /// surfaces whose base colour isn't known at the call site
    /// (e.g. icon buttons floating on the top bar).
    pub fn with_alpha(c: Color32, a: u8) -> Color32 {
        Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a)
    }

    /// Per-channel gamma-space blend of `b` over `a`.
    pub fn mix(a: Color32, b: Color32, t: f32) -> Color32 {
        let ch = |x: u8, y: u8| -> u8 {
            (x as f32 + (y as f32 - x as f32) * t)
                .round()
                .clamp(0.0, 255.0) as u8
        };
        Color32::from_rgb(ch(a.r(), b.r()), ch(a.g(), b.g()), ch(a.b(), b.b()))
    }
    /// Cool-blue accent used to outline overlay panels inside the
    /// viewport (camera PIP). Distinct from
    /// the brand purple so the viewport's own affordances don't read
    /// as just-another-button.
    pub const VIEWPORT_OVERLAY_OUTLINE: Color32 = Color32::from_rgb(60, 130, 200);
}

/// Colours for content painted *inside* the always-dark viewport and
/// the camera preview pane: skeleton annotations, grid,
/// crosshair, overlay badges. Kept separate from [`color`] because the
/// viewport is its own dark context regardless of app theme — but
/// still tokenised so the same annotation reads identically in every
/// pane that draws it (panes used to
/// carry copy-pasted literals that could drift apart).
pub mod viz {
    use super::Color32;

    // ── Detection annotations (keypoints / skeleton / bbox) ───────
    // Alpha-carrying tokens are functions: `from_rgba_unmultiplied`
    // is not const, and hand-premultiplying would hide the authored
    // channel values.
    pub fn keypoint() -> Color32 {
        Color32::from_rgba_unmultiplied(0, 255, 128, 220)
    }
    pub fn bone() -> Color32 {
        Color32::from_rgba_unmultiplied(0, 200, 255, 180)
    }
    pub fn bbox() -> Color32 {
        Color32::from_rgba_unmultiplied(255, 220, 80, 200)
    }
    /// Hand-landmarker keypoints / edges — a distinct hue so the hand
    /// stage reads separately from the body skeleton on the wipe.
    pub fn hand_keypoint() -> Color32 {
        Color32::from_rgba_unmultiplied(255, 120, 60, 220)
    }
    pub fn hand_bone() -> Color32 {
        Color32::from_rgba_unmultiplied(255, 120, 60, 130)
    }
    /// Hand-crop diagnostic rect (what the hand stage looked at).
    pub fn hand_crop() -> Color32 {
        Color32::from_rgba_unmultiplied(255, 120, 60, 90)
    }
    /// Wipe status badge text (drawn on a dark plate).
    pub fn wipe_badge() -> Color32 {
        Color32::from_rgba_unmultiplied(220, 228, 235, 230)
    }
    pub fn wipe_badge_plate() -> Color32 {
        Color32::from_rgba_unmultiplied(10, 12, 16, 170)
    }

    /// Keypoint colour with the confidence baked into the alpha, so a dot
    /// visually encodes how sure the inference was — a 0.2 detection
    /// reads fainter than a 0.9 one on the wipe.
    pub fn keypoint_graded(conf: f32) -> Color32 {
        let c = conf.clamp(0.0, 1.0);
        Color32::from_rgba_unmultiplied(0, 255, 128, (90.0 + 140.0 * c) as u8)
    }
    pub fn hand_keypoint_graded(conf: f32) -> Color32 {
        let c = conf.clamp(0.0, 1.0);
        Color32::from_rgba_unmultiplied(255, 120, 60, (90.0 + 140.0 * c) as u8)
    }

    // ── Empty-viewport furniture ──────────────────────────────────
    pub const GRID: Color32 = Color32::from_rgb(45, 45, 55);
    pub const CROSSHAIR: Color32 = Color32::from_rgb(70, 70, 85);
    pub const LABEL_PRIMARY: Color32 = Color32::from_rgb(100, 100, 115);

    // ── Alpha-preview checkerboards ───────────────────────────────
    /// Light checker pair drawn *behind the rendered image* when alpha
    /// preview is on (light so transparency reads as "photo editor").
    pub const CHECKER_LIGHT_A: Color32 = Color32::from_rgb(180, 180, 180);
    pub const CHECKER_LIGHT_B: Color32 = Color32::from_rgb(220, 220, 220);
    /// Dark checker pair for the placeholder (no rendered image yet).
    pub const CHECKER_DARK_A: Color32 = Color32::from_rgb(30, 30, 35);
    pub const CHECKER_DARK_B: Color32 = Color32::from_rgb(40, 40, 45);

    // ── Overlay text / badges on dark content ─────────────────────
    pub const OVERLAY_TEXT: Color32 = Color32::from_rgb(205, 212, 230);
    pub fn overlay_badge_bg() -> Color32 {
        Color32::from_rgba_unmultiplied(18, 20, 26, 190)
    }
    pub fn selection_text() -> Color32 {
        Color32::from_rgba_unmultiplied(255, 220, 80, 220)
    }
    pub fn muted_overlay_text() -> Color32 {
        Color32::from_rgba_unmultiplied(180, 180, 180, 160)
    }
}

pub mod space {
    pub const XS: f32 = 4.0;
    pub const SM: f32 = 8.0;
    pub const MD: f32 = 16.0;
    pub const LG: f32 = 24.0;
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
    pub const TRACKING_SETUP: char = '\u{e8aa}'; // track_changes
    pub const RENDERING: char = '\u{e40a}'; // palette
    pub const OUTPUT: char = '\u{ebbe}'; // output
    pub const CLOTH_AUTHORING: char = '\u{f19e}'; // checkroom
    pub const SETTINGS: char = '\u{e8b8}'; // settings
    pub const HIDE_PANEL: char = '\u{e408}'; // chevron_left
    pub const CHEVRON_RIGHT: char = '\u{e5cc}'; // chevron_right (collapsed section)
    pub const EXPAND_MORE: char = '\u{e5cf}'; // expand_more (expanded section)

    // Top-bar actions
    pub const MENU: char = '\u{e5d2}'; // menu (hamburger)
    pub const FOLDER_OPEN: char = '\u{e2c8}'; // folder_open
    pub const SAVE: char = '\u{e161}'; // save
    pub const PAUSE: char = '\u{e034}'; // pause
    pub const PLAY: char = '\u{e037}'; // play_arrow
    pub const MORE_VERT: char = '\u{e5d4}'; // more_vert

    // Common actions
    pub const REFRESH: char = '\u{e5d5}'; // refresh
    pub const SEARCH: char = '\u{e8b6}'; // search
    pub const ADD: char = '\u{e145}'; // add
    pub const REMOVE: char = '\u{e15b}'; // remove
    pub const DELETE: char = '\u{e872}'; // delete
    pub const FAVORITE_FILLED: char = '\u{e838}'; // star
    pub const FAVORITE_BORDER: char = '\u{e83a}'; // star_border

    // Status indicators
    pub const STATUS_DOT: char = '\u{e061}'; // fiber_manual_record (small filled circle)
    pub const HOME: char = '\u{e88a}'; // home
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
    // Elevation is communicated by shadow alone — see the note on
    // `color::state_layer` for why hairline strokes are avoided.
    visuals.window_rounding = Rounding::same(radius::MD);
    visuals.window_stroke = Stroke::NONE;
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
    // Interactive states are differentiated purely by fill (MD3 state
    // layers): rest = SURFACE_VARIANT, hover = +8% primary, active /
    // open = PRIMARY_CONTAINER. No bg_stroke hairlines — egui can't
    // render them at uniform thickness (see `color::state_layer`).
    // The only stroked state left is `noninteractive.bg_stroke`,
    // which egui uses for separators — a structural divider, and
    // axis-aligned lines don't suffer the rounded-corner artefacts.
    //
    // fg_stroke carries text/glyph colour plus checkmark / arrow
    // line width; the width is kept identical across states so marks
    // don't fatten on hover.
    let widgets = &mut visuals.widgets;
    let widget_round = Rounding::same(radius::SM);
    const FG_STROKE_W: f32 = 1.5;

    widgets.noninteractive.bg_fill = color::SURFACE;
    widgets.noninteractive.weak_bg_fill = color::SURFACE;
    widgets.noninteractive.bg_stroke = Stroke::new(1.0, color::OUTLINE_VARIANT);
    widgets.noninteractive.fg_stroke = Stroke::new(FG_STROKE_W, color::ON_SURFACE);
    widgets.noninteractive.rounding = widget_round;

    widgets.inactive.bg_fill = color::SURFACE_VARIANT;
    widgets.inactive.weak_bg_fill = color::SURFACE_VARIANT;
    widgets.inactive.bg_stroke = Stroke::NONE;
    widgets.inactive.fg_stroke = Stroke::new(FG_STROKE_W, color::ON_SURFACE);
    widgets.inactive.rounding = widget_round;

    widgets.hovered.bg_fill = color::state_layer(color::SURFACE_VARIANT, color::PRIMARY);
    widgets.hovered.weak_bg_fill = color::state_layer(color::SURFACE_VARIANT, color::PRIMARY);
    widgets.hovered.bg_stroke = Stroke::NONE;
    widgets.hovered.fg_stroke = Stroke::new(FG_STROKE_W, color::PRIMARY);
    widgets.hovered.rounding = widget_round;

    widgets.active.bg_fill = color::PRIMARY_CONTAINER;
    widgets.active.weak_bg_fill = color::PRIMARY_CONTAINER;
    widgets.active.bg_stroke = Stroke::NONE;
    widgets.active.fg_stroke = Stroke::new(FG_STROKE_W, color::ON_PRIMARY_CONTAINER);
    widgets.active.rounding = widget_round;

    widgets.open.bg_fill = color::PRIMARY_CONTAINER;
    widgets.open.weak_bg_fill = color::PRIMARY_CONTAINER;
    widgets.open.bg_stroke = Stroke::NONE;
    widgets.open.fg_stroke = Stroke::new(FG_STROKE_W, color::ON_PRIMARY_CONTAINER);
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
