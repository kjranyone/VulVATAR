# Streamer Display Webcam Capture Composition Plan

This dataset contains synthetic upper-body pose-validation frames for a
display-mounted webcam view of a seated streamer.

## Dataset Goal

Validate tracking behavior in a realistic desk-streaming crop where the pelvis
is hidden by the tabletop and upper-body inference must remain stable across
head, shoulder, torso, arm, wrist, hand-depth, and self-occlusion changes.

The visual identity should remain constant:

- Fictional non-identifiable adult woman streamer.
- Same jirai-kei inspired sleeveless home-streaming outfit.
- Same head-height display webcam, desk crop, room, and seated scale.
- 16:9 WebP output (q90 lossy) normalized to 1600x900.

## Camera Annotation

Use `displaycam_headlevel_deskcrop`.

Meaning:

- Webcam clipped to the top center of a desktop display.
- Camera height approximately level with the seated subject's head.
- Optical axis nearly horizontal and straight toward the subject.
- Desk tabletop hides the pelvis/hip-bone region.
- Subject remains seated behind the desk.

## Directory Layout

```text
validation_images/streamer_display_webcam_capture/
  subject_prompt.md
  composition_plan.md
  annotations.csv
  photorealistic/
    calibration_reference/
    head_face/
    shoulders/
    torso_posture/
    arm_reach/
    elbow_depth/
    wrists_hands/
    self_occlusion/
    depth_gestures/
```

## Filename Pattern

```text
pose_inference_displaycam_headlevel_deskcrop_jirai_streamer_<group>_<case_id>_<short_pose>_16x9.webp
```

Required CSV columns:

```text
file,title,subject_id,camera_annotation,group,case_id,pose_annotation,calibration_role,expected_stress
```

## Composition Matrix

The 64-frame matrix is intentionally upper-body biased because the desk crop
hides the pelvis and legs.

| Group | Cases | Intent |
| --- | --- | --- |
| calibration_reference | 4 | Neutral, A-pose-like, desk T-pose-like, and right-profile calibration crops |
| head_face | 6 | Yaw, pitch, roll, and chin changes while chest stays readable |
| shoulders | 6 | Bilateral and asymmetric shoulder elevation and shoulder-forward posture |
| torso_posture | 6 | Lean, twist, slouch, and chest rotation under seated desk framing |
| arm_reach | 10 | Overhead, side, diagonal, forward, behind-head, and on-desk arm reach |
| elbow_depth | 6 | Elbow depth ordering and elbow overlap against the torso |
| wrists_hands | 8 | Palm orientation, fist/open-hand contrast, crossed wrists, and hand signs |
| self_occlusion | 8 | Shoulder, chest, neck, and face-adjacent hand occlusion |
| depth_gestures | 10 | Near-camera hands, asymmetric reach, keyboard/mouse-rest style desk gestures |

## Evaluation Notes

Compare each frame with the same fixed webcam crop and subject identity. Track
at least:

- Shoulder width stability under head and arm motion.
- Torso center and spine orientation when hips are hidden by the desk.
- Left/right wrist swaps and palm depth ambiguity.
- Chest and ribcage silhouette behavior in a close-fitting sleeveless top.
- Landmark loss when hands overlap shoulders, chest, neck, and near-camera
  perspective.
