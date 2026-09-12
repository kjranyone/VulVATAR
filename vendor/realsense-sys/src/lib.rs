#![doc = include_str!("../README.md")]
// Allow all warnings here -- Bindgen generates this file, we really don't care about individual
// warnings since we can't really do much about them, we'd have to fix bindgen upstream or
// librealsense2 itself.
#![allow(warnings)]
#![allow(missing_docs)]
#![allow(clippy::missing_docs_in_private_items)]
// With buildtime-bindgen the build script writes bindings.rs into
// OUT_DIR; every other configuration (docs-only / prebuilt) falls back
// to the bindings shipped in this package.
#[cfg(feature = "buildtime-bindgen")]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
#[cfg(not(feature = "buildtime-bindgen"))]
include!("../bindings/bindings.rs");
