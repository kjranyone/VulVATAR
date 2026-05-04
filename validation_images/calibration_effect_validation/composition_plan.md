# Calibration Effect Validation Dataset Composition Plan

This directory is reserved for a single-subject validation dataset used to
compare tracking behavior before and after calibration.

This document defines the composition matrix, filename annotations, and
evaluation intent. Some rows may already have rendered images, while later rows
can remain planned entries in `annotations.csv` until generated.

## Dataset Goal

Validate whether calibration improves pose stability for one consistent
fictional adult subject across camera framing, body orientation, clothing
silhouette, RGB skin visibility, and self-occlusion cases.

The subject should remain visually consistent across the full set:

- Adult fictional woman, non-identifiable, no celebrity likeness.
- Same face, hairstyle, height, build, skin tone, and baseline proportions.
- Fully clothed and non-sexual in every frame.
- 16:9 PNG output normalized to 1600x900.

## Fixed Camera Annotation

Use this camera annotation unless a row explicitly tests camera variation:

`cam_fov70_dist280_h130_fullbody75`

Meaning:

- Horizontal FOV: 70 deg
- Camera distance: 2.80 m
- Camera height: 1.30 m
- Optical axis: level and straight ahead
- Subject framing: full body at about 75% image height

## Directory Layout

Planned layout:

```text
validation_images/calibration_effect_validation/
  composition_plan.md
  annotations.csv
  calibration_reference/
  fixed_camera/
    baseline_pose/
    orientation/
    head_face/
    shoulders/
    wrists_hands/
    torso_hips/
    lower_body/
    arm_reach/
    elbow_motion/
    depth_occlusion/
    balance_motion/
    hand_shoulder_occlusion/
    chest_silhouette/
    clothing_skin_visibility/
  camera_variation/
    distance/
    height/
    crop/
```

## Filename Pattern

Use compact filenames and keep the full labels in `annotations.csv`.

```text
pose_inference_<camera>_<subject>_<group>_<case_id>_<short_pose>_16x9.png
```

Example:

```text
pose_inference_cam_fov70_dist280_h130_fullbody75_single_subject_shoulders_003_r_up_l_down_16x9.png
```

Required CSV columns:

```text
file,title,subject_id,camera_annotation,group,case_id,pose_annotation,calibration_role,expected_stress
```

## Composition Matrix

| Group | Case ID | Short Pose | Calibration Role | Composition | Expected Stress |
| --- | --- | --- | --- | --- | --- |
| calibration_reference | 001 | front_neutral | calibration_input | Front-facing neutral stance, arms slightly away from torso, feet shoulder width | Baseline scale, shoulder width, hip width |
| calibration_reference | 002 | front_t_pose | calibration_input | Front-facing T-pose, arms horizontal, palms down | Arm span, shoulder center, bone length |
| calibration_reference | 003 | front_a_pose | calibration_input | Front-facing A-pose, arms 30 deg down from horizontal | Relaxed calibration posture |
| calibration_reference | 004 | side_profile | calibration_input | Right side profile, arms relaxed, feet visible | Depth prior and body thickness |
| baseline_pose | 001 | front_neutral | evaluation | Same as calibration neutral but not used as calibration input | Calibration repeatability |
| baseline_pose | 002 | front_wide_stance | evaluation | Front-facing, feet wider than shoulders, arms relaxed | Hip and leg scale stability |
| orientation | 001 | front_000 | evaluation | Front-facing, chest and pelvis to camera | Facing direction baseline |
| orientation | 002 | front_right_045 | evaluation | Body front-right 45 deg, head to camera | Three-quarter torso orientation |
| orientation | 003 | right_profile_090 | evaluation | Exact right side profile | Side-view shoulder and hip alignment |
| orientation | 004 | rear_right_135 | evaluation | Rear-right three-quarter, head slightly turned | Back-facing ambiguity |
| orientation | 005 | back_180 | evaluation | Back-facing, hands visible away from torso | Left/right landmark swap risk |
| head_face | 001 | head_yaw_l030 | evaluation | Chest front, head yaw left 30 deg | Head vs chest decoupling |
| head_face | 002 | head_yaw_r030 | evaluation | Chest front, head yaw right 30 deg | Head vs chest decoupling |
| head_face | 003 | head_pitch_up020 | evaluation | Chest front, head pitched up 20 deg | Neck and head anchor stability |
| head_face | 004 | head_roll_r018 | evaluation | Chest front, head rolled right 18 deg | Face orientation robustness |
| shoulders | 001 | l_up_r_down | evaluation | Left shoulder raised, right shoulder lowered | Shoulder elevation calibration |
| shoulders | 002 | both_up_shrug | evaluation | Both shoulders raised toward ears | Neck and shoulder separation |
| shoulders | 003 | r_up_l_down | evaluation | Right shoulder raised, left shoulder lowered | Asymmetric shoulder tracking |
| wrists_hands | 001 | r_palm_camera | evaluation | Right palm faces camera near chest height | Wrist orientation, hand scale |
| wrists_hands | 002 | r_down_l_up | evaluation | Right palm down, left palm up | Forearm rotation ambiguity |
| wrists_hands | 003 | crossed_x | evaluation | Wrists crossed in front of chest | Hand and wrist crossing |
| torso_hips | 001 | pelvis_r030_chest_front | evaluation | Pelvis angled right, chest front | Waist twist decoupling |
| torso_hips | 002 | hips_l_chest_r | evaluation | Hips shifted left, chest leaning right | Hip center and spine compensation |
| torso_hips | 003 | pelvis_fl_chest_fr | evaluation | Pelvis front-left, chest front-right | Opposing torso rotation |
| hand_shoulder_occlusion | 001 | l_hand_l_shoulder | evaluation | Left hand covers left shoulder landmark | Hand/shoulder coordinate overlap |
| hand_shoulder_occlusion | 002 | r_hand_r_shoulder | evaluation | Right hand covers right shoulder landmark | Hand/shoulder coordinate overlap |
| hand_shoulder_occlusion | 003 | l_hand_r_shoulder | evaluation | Left hand crosses to right shoulder | Cross-body occlusion |
| hand_shoulder_occlusion | 004 | r_hand_l_shoulder | evaluation | Right hand crosses to left shoulder | Cross-body occlusion |
| hand_shoulder_occlusion | 005 | both_hands_shoulders | evaluation | Both hands cover both shoulders | Bilateral shoulder occlusion |
| hand_shoulder_occlusion | 006 | palms_front_shoulders | evaluation | Open palms in front of shoulder landmarks | Near-camera hand distraction |
| chest_silhouette | 001 | chest_small_front | evaluation | Small chest silhouette, front neutral | Chest size effect |
| chest_silhouette | 002 | chest_medium_front | evaluation | Medium chest silhouette, front neutral | Chest size effect |
| chest_silhouette | 003 | chest_large_front | evaluation | Large chest silhouette, front neutral | Chest size effect |
| chest_silhouette | 004 | chest_small_arms_crossed | evaluation | Small chest silhouette, arms crossed | Chest plus arm occlusion |
| chest_silhouette | 005 | chest_medium_arms_crossed | evaluation | Medium chest silhouette, arms crossed | Chest plus arm occlusion |
| chest_silhouette | 006 | chest_large_arms_crossed | evaluation | Large chest silhouette, arms crossed | Chest plus arm occlusion |
| clothing_skin_visibility | 001 | modest_idol_front | evaluation | Sleeved idol outfit, lower skin visibility | Clothing baseline |
| clothing_skin_visibility | 002 | frilly_idol_front | evaluation | Frilly idol outfit, shoulder and waist volume | Costume silhouette volume |
| clothing_skin_visibility | 003 | high_skin_front | evaluation | Sleeveless outfit, visible arms and legs | RGB skin-region tracking |
| clothing_skin_visibility | 004 | gothic_high_skin_front | evaluation | Gothic sleeveless outfit, visible arms and legs | Dark clothing plus skin contrast |
| camera_variation_distance | 001 | dist220_fullbody | evaluation | Same subject, closer camera distance 2.20 m | Scale calibration under distance shift |
| camera_variation_distance | 002 | dist340_fullbody | evaluation | Same subject, farther camera distance 3.40 m | Scale calibration under distance shift |
| camera_variation_height | 001 | h100_low_camera | evaluation | Camera height 1.00 m, full body visible | Low camera perspective |
| camera_variation_height | 002 | h160_high_camera | evaluation | Camera height 1.60 m, full body visible | High camera perspective |
| camera_variation_crop | 001 | upper_body_crop | evaluation | Knees or lower legs cropped, upper body centered | Partial-body calibration failure |
| camera_variation_crop | 002 | hands_near_edge | evaluation | Hands near image edge, body centered | Edge keypoint stability |

## Additional Pose Expansion

These cases expand the fixed-camera matrix when the first 46 cases are not
enough to stress calibration behavior.

| Group | Case ID | Short Pose | Calibration Role | Composition | Expected Stress |
| --- | --- | --- | --- | --- | --- |
| lower_body | 001 | deep_squat | evaluation | Front-facing deep squat, knees apart, arms forward for balance | Hip height, knee bend, leg shortening |
| lower_body | 002 | half_squat | evaluation | Front-facing half squat, feet shoulder width, arms relaxed | Hip and knee scale under compression |
| lower_body | 003 | l_side_lunge | evaluation | Left side lunge, left knee bent, right leg extended | Asymmetric hip/knee placement |
| lower_body | 004 | r_side_lunge | evaluation | Right side lunge, right knee bent, left leg extended | Asymmetric hip/knee placement |
| lower_body | 005 | l_forward_lunge | evaluation | Left foot forward lunge, right leg back | Forward/back leg ordering |
| lower_body | 006 | r_forward_lunge | evaluation | Right foot forward lunge, left leg back | Forward/back leg ordering |
| lower_body | 007 | l_knee_raise | evaluation | Standing on right leg, left knee raised to hip height | Single-leg balance and hip center |
| lower_body | 008 | r_knee_raise | evaluation | Standing on left leg, right knee raised to hip height | Single-leg balance and hip center |
| lower_body | 009 | cross_step_l_over_r | evaluation | Left foot crosses in front of right foot | Ankle/leg crossing |
| lower_body | 010 | cross_step_r_over_l | evaluation | Right foot crosses in front of left foot | Ankle/leg crossing |
| arm_reach | 001 | both_arms_overhead | evaluation | Both arms straight overhead, palms forward | Arm length and shoulder elevation |
| arm_reach | 002 | l_arm_overhead | evaluation | Left arm overhead, right arm down | Asymmetric arm extension |
| arm_reach | 003 | r_arm_overhead | evaluation | Right arm overhead, left arm down | Asymmetric arm extension |
| arm_reach | 004 | both_arms_forward | evaluation | Both arms extended forward at shoulder height | Foreshortened arms and wrist depth |
| arm_reach | 005 | both_arms_back | evaluation | Both arms swept backward, chest front | Hidden wrists and shoulder extension |
| arm_reach | 006 | diagonal_l_up_r_down | evaluation | Left arm diagonal up, right arm diagonal down | Diagonal limb scale |
| arm_reach | 007 | diagonal_r_up_l_down | evaluation | Right arm diagonal up, left arm diagonal down | Diagonal limb scale |
| arm_reach | 008 | hands_behind_head | evaluation | Both hands behind head, elbows wide | Wrist/ear/shoulder ambiguity |
| arm_reach | 009 | hands_on_hips | evaluation | Both hands on hips, elbows out | Wrist near pelvis landmarks |
| arm_reach | 010 | elbows_forward | evaluation | Elbows lifted forward, fists near collarbone | Elbow/shoulder overlap |
| elbow_motion | 001 | l_elbow_forward_r_back | evaluation | Left elbow projects forward, right elbow sits back | Elbow depth ordering |
| elbow_motion | 002 | r_elbow_forward_l_back | evaluation | Right elbow projects forward, left elbow sits back | Elbow depth ordering |
| elbow_motion | 003 | both_elbows_forward | evaluation | Both elbows project forward toward camera | Elbow and shoulder overlap |
| elbow_motion | 004 | both_elbows_back | evaluation | Both elbows sit behind torso line | Rear elbow visibility |
| depth_occlusion | 001 | l_hand_near_camera | evaluation | Left hand reaches toward camera, palm large, shoulder visible behind | Near-hand scale distraction |
| depth_occlusion | 002 | r_hand_near_camera | evaluation | Right hand reaches toward camera, palm large, shoulder visible behind | Near-hand scale distraction |
| depth_occlusion | 003 | both_hands_near_camera | evaluation | Both hands extended toward camera, palms visible | Bilateral near-hand scale distraction |
| depth_occlusion | 004 | l_knee_near_camera | evaluation | Left knee lifted and slightly forward toward camera | Knee depth ambiguity |
| depth_occlusion | 005 | r_knee_near_camera | evaluation | Right knee lifted and slightly forward toward camera | Knee depth ambiguity |
| depth_occlusion | 006 | forward_lean_hands_low | evaluation | Torso leans toward camera, hands low near knees | Spine depth and wrist/knee confusion |
| balance_motion | 001 | forward_lean | evaluation | Straight legs, torso leans forward from hips, arms back | Spine angle and hip hinge |
| balance_motion | 002 | backward_lean | evaluation | Body leans backward, knees soft, arms forward | Balance compensation |
| balance_motion | 003 | lean_left | evaluation | Whole torso leans left, hips counter-shift right | Center-of-mass lateral shift |
| balance_motion | 004 | lean_right | evaluation | Whole torso leans right, hips counter-shift left | Center-of-mass lateral shift |
| balance_motion | 005 | small_jump_airborne | evaluation | Small vertical jump, both feet slightly off floor, arms out | Foot contact and body scale |
| balance_motion | 006 | twist_step | evaluation | Feet planted, knees bent, torso twisting opposite hips | Combined lower-body and torso stress |

## Suggested Generation Order

1. Generate `calibration_reference/` first and keep subject identity locked.
2. Generate `fixed_camera/baseline_pose/` from the same visual identity.
3. Generate pose stress groups under the fixed camera setup.
4. Generate camera variation groups last, because they intentionally break the
   fixed-camera assumption.

## Evaluation Notes

For each frame, compare tracking output with calibration disabled and enabled.
Record at least:

- Shoulder width stability
- Hip width stability
- Spine length stability
- Left/right shoulder swap frequency
- Left/right wrist swap frequency
- Z-depth plausibility if using depth-aware providers
- Temporal jitter if the frame is later extended into a short clip

The most important comparisons are rows that keep camera and subject constant
while changing only one stress factor, such as shoulder elevation, hand
occlusion, chest silhouette, or clothing volume.
