# Validation Images

This directory contains generated validation images for pose-estimation and retargeting tests.

All images are 16:9 PNG files normalized to 1600x900. Filenames include the scenario, pose or rotation label, and subject constraints so they can be used as lightweight annotations.

## Sets

- `basic_pose_samples/photorealistic/`
  - 5 basic full-body pose samples.
  - Includes T-pose, single-leg balance, squat, side lunge, and walking stride.
  - Useful for quick pose-estimation smoke tests without props.

- `gothic_lolita_rotation/photorealistic/`
  - 8 frames of an adult woman in gothic lolita fashion.
  - Covers a 360 degree turn at 45 degree increments.
  - Useful for checking pose stability with layered clothing and silhouette complexity.

- `mocap_suit_rotation/photorealistic/`
  - 16 frames of an adult woman in a motion-capture style suit with joint markers.
  - Covers a 360 degree turn at 22.5 degree increments.
  - Useful as the clearest synthetic rotation reference for body-joint annotation.

- `mocap_suit_rotation/interpolated_photorealistic/`
  - 8 additional 22.5 degree offset frames.
  - Kept separately because these were generated as interpolation-angle supplements.

- `reiwa_idol_rotation/anime/`
  - 16 frames of an adult woman in Reiwa-era Japanese idol stage fashion.
  - Covers a 360 degree turn at 22.5 degree increments.
  - Useful for validating pose estimation under layered idol-style clothing.

- `guitar_woman_rotation/anime/`
  - 16 frames of an adult woman holding an electric guitar.
  - Covers a 360 degree turn at 22.5 degree increments.
  - Useful for checking side, rear, and three-quarter body orientation with a large occluding prop.

- `guitar_idol_playing/anime/`
  - 20 varied anime-style performance poses.
  - Includes front, side, rear, crouch, lunge, jump, kneeling, torso twist, and overhead guitar poses.
  - Useful for stressing keypoint stability under self-occlusion and guitar occlusion.

- `guitar_idol_playing/photorealistic/`
  - 20 photorealistic counterparts to the anime-style performance set.
  - Uses the same pose IDs and labels where possible.
  - Useful for checking domain shift between illustration-like inputs and realistic studio-photo inputs.

- `reiwa_idol_webcam_capture/photorealistic/`
  - 15 photorealistic webcam-style frames of a fictional adult woman in Reiwa-era Japanese idol-inspired rehearsal fashion.
  - Split into `face_head/`, `shoulders/`, `wrists_hands/`, `torso_hips/`, and `orientation/`.
  - Filenames annotate head yaw/pitch/roll, shoulder elevation, wrist orientation, torso/hip twist, and full-body facing direction.
  - Useful for validating webcam-like pose tracking where facial motion, shoulder motion, hand rotation, waist twist, and body orientation are unstable.

- `reiwa_idol_webcam_capture/photorealistic_fixed_camera/`
  - 15 photorealistic webcam-style frames with a fixed capture setup.
  - Uses filename-title annotations such as `cam_fov70_dist280_h130_fullbody75` to mark the simulated camera geometry:
    70 degree horizontal FOV, 2.80 m subject distance, 1.30 m camera height, and full-body framing at about 75% image height.
  - Split into the same `face_head/`, `shoulders/`, `wrists_hands/`, `torso_hips/`, and `orientation/` categories.
  - Prefer this set when comparing pose changes without camera framing changes.

- `reiwa_idol_chest_silhouette_variation/photorealistic_fixed_camera/`
  - 9 photorealistic fixed-camera webcam-style frames for checking whether chest silhouette changes affect pose inference.
  - Split into `small_chest/`, `medium_chest/`, and `large_chest/`, with the same 3 pose labels in each group:
    `front_neutral`, `front_arms_crossed`, and `three_quarter_torso_twist`.
  - Uses the same `cam_fov70_dist280_h130_fullbody75` filename-title camera annotation as the fixed-camera webcam set.
  - All images are fully clothed, non-sexual synthetic validation assets.

- `reiwa_idol_chest_silhouette_variation/photorealistic_fixed_camera_frilly_idol/`
  - 9 photorealistic fixed-camera webcam-style frames using a more frilly idol-stage costume.
  - Uses the same chest-size and pose matrix as `photorealistic_fixed_camera/`, but adds layered ruffles, puff sleeves, bows, shoulder frills, and a tiered skirt.
  - Useful for checking whether idol-style costume volume around the chest, shoulders, and waist changes pose inference.
  - All images are fully clothed, non-sexual synthetic validation assets.

- `reiwa_idol_chest_silhouette_variation/photorealistic_fixed_camera_high_skin/`
  - 9 photorealistic fixed-camera webcam-style frames with increased visible skin area for RGB pose inference.
  - Uses the same chest-size and pose matrix as `photorealistic_fixed_camera/`, but switches to a modest sleeveless idol dance outfit with visible shoulders, arms, and legs.
  - Useful for checking whether RGB skin-region visibility improves or destabilizes pose inference.
  - All images are adult, fully clothed, non-sexual synthetic validation assets.

- `gothic_lolita_chest_silhouette_variation/photorealistic_fixed_camera_high_skin/`
  - 9 photorealistic fixed-camera studio frames loosely based on the gothic lolita validation character.
  - Uses `gothic_lolita_rotation/photorealistic/pose_inference_gothic_lolita_rotation_frame_01_front_000deg_full_body_adult_woman_plain_studio_16x9.png`
    as a character/style reference for hair, headwear, and black-and-white gothic styling.
  - Split into `small_chest/`, `medium_chest/`, and `large_chest/`, with the same 3 pose labels:
    `front_neutral`, `front_arms_crossed`, and `three_quarter_torso_twist`.
  - Uses a modest sleeveless gothic idol outfit with visible shoulders, arms, and legs for RGB skin-region validation.
  - All images are adult, fully clothed, non-sexual synthetic validation assets.

## Notes

- These images are synthetic validation assets, not training data.
- The subject is an adult, fully clothed, non-identifiable generated person.
- The guitar is intentionally included as an occluder because it stresses torso, arm, wrist, and hand keypoint detection.
