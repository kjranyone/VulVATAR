//! Pixel-space face bbox geometry derived from body-detector face
//! points. Pure image geometry — no inference — so it survives the
//! face-backend swaps (it was pulled out of the deleted MediaPipe
//! module; the RTMPose-face sidecar path is its only consumer today).

/// Pixel-space face bbox (image-relative).
#[derive(Clone, Copy, Debug)]
pub struct FaceBbox {
    pub x: f32,
    pub y: f32,
    pub size: f32,
}

const FACE_BBOX_PAD: f32 = 1.3;

/// Derive a face bbox from a list of pixel-space face-region points
/// (typically the five body keypoints 0..=4: nose, left/right eye,
/// left/right ear). Returns `None` if too few points were available
/// or the bbox falls outside the image.
#[cfg(feature = "inference")]
pub fn derive_face_bbox(
    points_px: &[(f32, f32)],
    source_w: u32,
    source_h: u32,
) -> Option<FaceBbox> {
    if points_px.len() < 4 {
        return None;
    }
    let cx = points_px.iter().map(|p| p.0).sum::<f32>() / points_px.len() as f32;
    let cy = points_px.iter().map(|p| p.1).sum::<f32>() / points_px.len() as f32;
    let max_d = points_px
        .iter()
        .map(|(x, y)| ((x - cx).powi(2) + (y - cy).powi(2)).sqrt())
        .fold(0.0_f32, f32::max);
    let half = (max_d * FACE_BBOX_PAD).max(16.0);
    let size = half * 2.0;
    if cx + half < 0.0 || cy + half < 0.0 {
        return None;
    }
    if cx - half > source_w as f32 || cy - half > source_h as f32 {
        return None;
    }
    Some(FaceBbox {
        x: cx - half,
        y: cy - half,
        size,
    })
}
