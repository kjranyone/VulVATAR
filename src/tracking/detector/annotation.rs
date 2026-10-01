//! 2D detection annotation for the GUI camera-preview overlay and
//! downstream consumers. Carries the full COCO-Wholebody 133 keypoints
//! in image-normalised coords; the skeleton edge list still describes
//! only the body-17 topology so the GUI renderer keeps working without
//! changes. Writers per block: the detector fills body 0..17 (everything
//! else score 0), the fusion provider owns the hand blocks 91..133
//! (`fill_hand_annotation` there), and the face sidecar's measured
//! WFLW-98 points ride in `face_points` — the COCO-Wholebody face block
//! 23..91 is never written (no 98→68 mapping exists).

use super::super::DetectionAnnotation;
use super::decode::DecodedJoint;

pub(super) fn build_annotation(joints: &[DecodedJoint]) -> DetectionAnnotation {
    let keypoints = joints.iter().map(|j| (j.nx, j.ny, j.score)).collect();
    DetectionAnnotation {
        keypoints,
        skeleton: coco_body_edges(),
        bounding_box: None,
        hand_crops: [None, None],
        face_points: Vec::new(),
    }
}

/// Normalise the face sidecar's frame-pixel WFLW-98 landmarks into the
/// annotation, carrying the single Procrustes confidence on every point
/// (the sidecar has no per-point score). Mirrors what
/// `fill_hand_annotation` does for the hand blocks.
pub(super) fn fill_face_annotation(
    annotation: &mut DetectionAnnotation,
    pts_frame_px: &[[f32; 2]],
    confidence: f32,
    width: u32,
    height: u32,
) {
    let (w, h) = (width as f32, height as f32);
    annotation.face_points = pts_frame_px
        .iter()
        .map(|p| (p[0] / w, p[1] / h, confidence))
        .collect();
}

fn coco_body_edges() -> Vec<(usize, usize)> {
    vec![
        // Face
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        // Torso
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        // Arms
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        // Legs
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]
}

#[cfg(test)]
mod face_annotation_tests {
    use super::*;

    #[test]
    fn face_points_normalise_frame_px_and_carry_confidence() {
        let mut ann = build_annotation(&[]);
        assert!(ann.face_points.is_empty(), "no face stage result yet");

        fill_face_annotation(
            &mut ann,
            &[[64.0, 48.0], [320.0, 240.0], [0.0, 0.0]],
            0.43,
            640,
            480,
        );
        assert_eq!(ann.face_points.len(), 3);
        assert_eq!(ann.face_points[0], (0.1, 0.1, 0.43));
        assert_eq!(ann.face_points[1], (0.5, 0.5, 0.43));
        assert_eq!(ann.face_points[2], (0.0, 0.0, 0.43));

        // A fresher result replaces the whole set (98-point payload is
        // constant-length, but the wipe must never see a stale mix).
        fill_face_annotation(&mut ann, &[[320.0, 240.0]], 0.9, 640, 480);
        assert_eq!(ann.face_points.len(), 1);
        assert_eq!(ann.face_points[0], (0.5, 0.5, 0.9));
    }
}
