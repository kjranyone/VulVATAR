//! Pin the failure-path contract for [`super::handle_thumbnail_response`]:
//! when the render thread reports a failure (or the encode itself
//! errors), no on-disk PNG is touched. A placeholder written by the
//! avatar-import path stays in place so the library inspector keeps
//! showing *something* when the GPU briefly misbehaves.
//!
//! Without this guarantee, a transient thumbnail failure would
//! delete the placeholder and the library would flash empty
//! squares — that is the regression these tests prevent.

use super::handle_thumbnail_response;
use crate::renderer::ThumbnailRenderResult;

fn make_tempdir(suffix: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("vulvatar_thumb_failure_{}_{}", suffix, nanos));
    std::fs::create_dir_all(&dir).expect("create tempdir");
    dir
}

/// 4x4 RGBA test image (64 bytes) with a deterministic gradient so a
/// successful round-trip is identifiable against the bytes.
fn make_thumbnail() -> ThumbnailRenderResult {
    let w = 4u32;
    let h = 4u32;
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            rgba.push(((x * 64) & 0xff) as u8);
            rgba.push(((y * 64) & 0xff) as u8);
            rgba.push(0x40);
            rgba.push(0xff);
        }
    }
    ThumbnailRenderResult {
        width: w,
        height: h,
        rgba_pixels: rgba,
    }
}

#[test]
fn render_failure_does_not_create_a_file() {
    let dir = make_tempdir("noprior");
    let path = dir.join("thumb.png");
    assert!(!path.exists());

    let outcome = handle_thumbnail_response(&path, Err("simulated GPU OOM".to_string()));
    assert!(outcome.is_err(), "failure response must surface as Err");
    assert!(
        !path.exists(),
        "render failure must not materialise an empty PNG"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn render_failure_preserves_existing_placeholder() {
    // Avatar import already wrote a placeholder; a subsequent
    // real-render failure must not overwrite or unlink it.
    let dir = make_tempdir("preserve");
    let path = dir.join("thumb.png");
    let placeholder_bytes: &[u8] = b"placeholder-bytes";
    std::fs::write(&path, placeholder_bytes).expect("seed placeholder");

    let outcome = handle_thumbnail_response(
        &path,
        Err("simulated render thread panic".to_string()),
    );
    assert!(outcome.is_err());

    let after = std::fs::read(&path).expect("placeholder still readable");
    assert_eq!(
        after, placeholder_bytes,
        "render failure must leave the placeholder bytes verbatim"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn render_success_writes_png_and_overwrites_placeholder() {
    // Sanity: the success path actually does write — otherwise the
    // failure-path tests would pass for the wrong reason (helper
    // never writes anything).
    let dir = make_tempdir("success");
    let path = dir.join("thumb.png");
    std::fs::write(&path, b"placeholder").expect("seed placeholder");

    let outcome = handle_thumbnail_response(&path, Ok(make_thumbnail()));
    assert!(outcome.is_ok());

    let bytes = std::fs::read(&path).expect("PNG written");
    // PNG magic header — confirms the placeholder was replaced with
    // a real PNG, not just untouched.
    assert!(
        bytes.len() > 8 && bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
        "success path must write a real PNG; got first bytes {:?}",
        &bytes[..bytes.len().min(8)]
    );

    let _ = std::fs::remove_dir_all(&dir);
}
