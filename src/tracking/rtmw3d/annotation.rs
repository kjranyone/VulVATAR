//! 2D detection annotation for the GUI camera-preview overlay and
//! downstream consumers. Carries the full COCO-Wholebody 133 keypoints
//! in image-normalised coords; the skeleton edge list still describes
//! only the body-17 topology so the GUI renderer keeps working without
//! changes. Depth-aware providers (`rtmw3d-with-depth`) read the full
//! 133 to back-project hands and face.

use super::super::DetectionAnnotation;
use super::decode::DecodedJoint;

pub(super) fn build_annotation(joints: &[DecodedJoint]) -> DetectionAnnotation {
    let keypoints = joints
        .iter()
        .map(|j| (j.nx, j.ny, j.score))
        .collect();
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
