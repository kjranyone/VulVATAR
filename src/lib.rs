pub mod app;
pub mod asset;
pub mod avatar;
#[cfg(all(target_os = "windows", feature = "virtual-camera"))]
pub mod dshow_dll;
pub mod editor;
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

#[macro_export]
macro_rules! t {
    ($key:expr) => {
        rust_i18n::t!($key).to_string()
    };
    ($key:expr, $($args:tt)*) => {
        rust_i18n::t!($key, $($args)*).to_string()
    };
}
