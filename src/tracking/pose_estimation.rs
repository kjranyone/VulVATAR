#![allow(dead_code)]
//! HSV skin-tone pose estimation fallback.
//!
//! Used only when the `inference` feature is disabled — a stand-in detector
//! that locates a single skin-coloured region in the frame and produces a
//! bare-minimum [`SourceSkeleton`] (no body joints, just a face pose
//! derived from the centroid) so the downstream solver has something to
//! work with. Proper pose estimation comes from
//! [`super::inference::CigPoseInference`].

use super::{DetectionAnnotation, FacePose, PoseEstimate, SourceSkeleton};

/// Produce a `PoseEstimate` from a raw RGB frame by skin-colour centroiding.
pub fn estimate_pose(rgb_data: &[u8], width: u32, height: u32, frame_index: u64) -> PoseEstimate {
    let expected_len = (width * height * 3) as usize;
    if rgb_data.len() < expected_len {
        return PoseEstimate {
            skeleton: SourceSkeleton::empty(frame_index),
            annotation: DetectionAnnotation::default(),
        };
    }

    let mut skin_pixel_count: u32 = 0;
    let mut skin_sum_x: u64 = 0;
    let mut skin_sum_y: u64 = 0;
    let mut skin_min_x: u32 = width;
    let mut skin_max_x: u32 = 0;
    let mut skin_min_y: u32 = height;
    let mut skin_max_y: u32 = 0;

    let stride: u32 = 2;
    for y in (0..height).step_by(stride as usize) {
        for x in (0..width).step_by(stride as usize) {
            let base = ((y * width + x) * 3) as usize;
            let r = rgb_data[base] as f32 / 255.0;
            let g = rgb_data[base + 1] as f32 / 255.0;
            let b = rgb_data[base + 2] as f32 / 255.0;

            let (h, s, v) = rgb_to_hsv(r, g, b);
            let is_skin =
                !(50.0..340.0).contains(&h) && (0.15..=0.75).contains(&s) && (0.2..=0.95).contains(&v);

            if is_skin {
                skin_pixel_count += 1;
                skin_sum_x += x as u64;
                skin_sum_y += y as u64;
                if x < skin_min_x {
                    skin_min_x = x;
                }
                if x > skin_max_x {
                    skin_max_x = x;
                }
                if y < skin_min_y {
                    skin_min_y = y;
                }
                if y > skin_max_y {
                    skin_max_y = y;
                }
            }
        }
    }

    let sampled_pixels = (width / stride) * (height / stride);
    let skin_ratio = skin_pixel_count as f32 / sampled_pixels.max(1) as f32;

    let mut skeleton = SourceSkeleton::empty(frame_index);
    if skin_ratio < 0.02 || skin_pixel_count == 0 {
        return PoseEstimate {
            skeleton,
            annotation: DetectionAnnotation::default(),
        };
    }

    let centroid_x = skin_sum_x as f32 / skin_pixel_count as f32;
    let centroid_y = skin_sum_y as f32 / skin_pixel_count as f32;

    // Very rough face pose from centroid offset. No keypoints → no yaw/pitch
    // derivable from geometry; we fall back to "how far off-centre the face
    // is in the frame" for a subtle sway.
    let norm_x = (centroid_x / width as f32) * 2.0 - 1.0;
    let norm_y = 1.0 - (centroid_y / height as f32) * 2.0;

    let confidence = (skin_ratio / 0.25).clamp(0.1, 1.0);
    skeleton.face = Some(FacePose {
        yaw: -norm_x * 0.3,
        pitch: norm_y * 0.2,
        roll: 0.0,
        confidence,
    });
    skeleton.overall_confidence = confidence;

    let w = width as f32;
    let h = height as f32;
    let annotation = DetectionAnnotation {
        keypoints: vec![(centroid_x / w, centroid_y / h, confidence)],
        skeleton: vec![],
        bounding_box: Some((
            skin_min_x as f32 / w,
            skin_min_y as f32 / h,
            skin_max_x as f32 / w,
            skin_max_y as f32 / h,
        )),
    };

    PoseEstimate {
        skeleton,
        annotation,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert an RGB color (each component in [0, 1]) to HSV.
/// Returns (hue in [0, 360], saturation in [0, 1], value in [0, 1]).
fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let value = max;

    if delta < 1e-6 {
        return (0.0, 0.0, value);
    }

    let saturation = delta / max;

    let hue = if (r - max).abs() < 1e-6 {
        60.0 * (((g - b) / delta).rem_euclid(6.0))
    } else if (g - max).abs() < 1e-6 {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };

    let hue = if hue < 0.0 { hue + 360.0 } else { hue };

    (hue, saturation, value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_to_hsv_red() {
        let (h, s, v) = rgb_to_hsv(1.0, 0.0, 0.0);
        assert!((h - 0.0).abs() < 1.0);
        assert!((s - 1.0).abs() < 0.01);
        assert!((v - 1.0).abs() < 0.01);
    }

    #[test]
    fn rgb_to_hsv_white() {
        let (_h, s, v) = rgb_to_hsv(1.0, 1.0, 1.0);
        assert!((s - 0.0).abs() < 0.01);
        assert!((v - 1.0).abs() < 0.01);
    }

    #[test]
    fn estimate_pose_empty_frame_returns_default() {
        let buf = vec![0u8; 10 * 10 * 3];
        let est = estimate_pose(&buf, 10, 10, 0);
        assert!(est.skeleton.face.is_none());
    }

    #[test]
    fn estimate_pose_uniform_skin_color() {
        // Fill the frame with a salmon skin tone; should trigger skin
        // detection and emit a face pose with non-zero confidence.
        let (w, h) = (40u32, 40u32);
        let mut buf = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                buf[i] = 220;
                buf[i + 1] = 170;
                buf[i + 2] = 140;
            }
        }
        let est = estimate_pose(&buf, w, h, 0);
        let face = est.skeleton.face.expect("face pose produced");
        assert!(face.confidence > 0.0);
    }
}
