pub mod app;
pub mod asset;
pub mod avatar;
pub mod editor;
pub mod frame_handoff;
pub mod gpu_coordination;
pub mod gui;
pub mod i18n;
pub mod lipsync;
pub mod math_utils;
pub mod output;
pub mod persistence;
pub mod renderer;
pub mod simulation;
pub mod single_instance;
pub mod tracking;

rust_i18n::i18n!("locales", fallback = "en");

// `into_owned`, not `to_string`: rust_i18n hands back a `Cow<str>`
// that is `Owned` for every parameterised key (the format! result).
// `to_string` goes through `&self` and *re-copies* that already-owned
// String on every call; `into_owned` moves it and only copies the
// `Borrowed` (static-key) case, where a copy is unavoidable for a
// `String` return. Same signature, so the ~600 call sites are
// untouched. A full `Cow` return was evaluated and rejected: egui's
// `WidgetText`/`RichText` take `Into<String>`/`impl Into<WidgetText>`
// and own their text, so exactly one allocation happens per label
// either way — `Cow` would only relocate it while breaking call sites.
#[macro_export]
macro_rules! t {
    ($key:expr) => {
        rust_i18n::t!($key).into_owned()
    };
    ($key:expr, $($args:tt)*) => {
        rust_i18n::t!($key, $($args)*).into_owned()
    };
}
