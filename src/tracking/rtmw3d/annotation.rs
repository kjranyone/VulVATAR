//! 2D detection annotation for the GUI camera-preview overlay. Just
//! the body-17 keypoints in image-normalised coords plus the COCO
//! body skeleton edge list — the renderer reads it without further
//! transforms.

use super::super::DetectionAnnotation;
use super::decode::DecodedJoint;

pub(super) fn build_annotation(joints: &[DecodedJoint]) -> DetectionAnnotation {
    // Emit the body 17 in image-normalised coords for the GUI overlay.
    // Edges follow the COCO body topology so the existing renderer can
    // draw it without changes.
    let mut keypoints = Vec::with_capacity(17);
    for j in joints.iter().take(17) {
        keypoints.push((j.nx, j.ny, j.score));
    }
    DetectionAnnotation {
        keypoints,
        skeleton: coco_body_edges(),
        bounding_box: None,
    }
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
