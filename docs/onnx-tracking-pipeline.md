# ONNX Full-Body Tracking Pipeline

This document describes the design and implementation of the ONNX
Runtime-based full-body tracking pipeline used in VulVATAR.

## Architecture Overview

The tracking system uses **MediaPipe Holistic** ONNX exports — a
3D-native cascaded pipeline (Pose → Hand) running on a background
`TrackingWorker` thread so it never blocks the main rendering loop.

```mermaid
graph TD;
    Camera[Webcam / Video Source] --> |RGB Frame| Letterbox[Letterbox to 256×256]
    Letterbox --> |NHWC [1,256,256,3]| Pose[Pose Landmarker ONNX]
    Pose --> |39 × screen + 39 × world| Skeleton[Source Skeleton]
    Pose --> |Body bbox: 15/17/19/21| BodyL[Hand bbox - subject left]
    Pose --> |Body bbox: 16/18/20/22| BodyR[Hand bbox - subject right]
    BodyL --> |224×224 crop| HandL[Hand Landmarker ONNX]
    BodyR --> |224×224 crop| HandR[Hand Landmarker ONNX]
    HandL --> |21 × world| Skeleton
    HandR --> |21 × world| Skeleton
    Skeleton --> |3D joints| Solver[pose_solver]
    Solver --> Mailbox[Tracking Mailbox]
```

## Models

All three ONNX files live in `models/` and are fetched by `dev.ps1`'s
`Install-Models` step (curl from Hugging Face, ~13 MB total):

| File                  | Source                                    | Size  | Role                            |
|-----------------------|-------------------------------------------|-------|---------------------------------|
| `pose_landmark.onnx`  | `opencv/pose_estimation_mediapipe`        | 5.5MB | 33 body + 6 aux landmarks (3D)  |
| `palm_detection.onnx` | `opencv/palm_detection_mediapipe`         | 3.9MB | Hand bbox detector (unused MVP) |
| `hand_landmark.onnx`  | `opencv/handpose_estimation_mediapipe`    | 3.9MB | 21 hand landmarks per side (3D) |

The palm detector ships with the bundle but is not yet wired in: the
hand-bbox derivation uses BlazePose's auxiliary hand markers (wrist +
pinky/index/thumb knuckles, indices 15/17/19/21 for subject's left and
16/18/20/22 for right) padded by 1.6×. This is enough for non-overlap
hand poses; the palm detector is held in reserve for occluded hands.

## Pipeline

### 1. Pose Landmarker

`tracking::mediapipe::MediaPipeInference::estimate_pose` letterboxes
the source frame to 256×256 (preserve aspect, zero-pad shorter side)
and runs the BlazePose Heavy export. Five outputs (positional, set by
the OpenCV ONNX export):

1. `landmarks`      `(1, 195)`    — 39 × (x, y, z, vis, pres) screen coords
2. `conf`           `(1, 1)`      — overall pose confidence (sigmoid-pre)
3. `mask`           `(1, 256, 256, 1)` — body segmentation (unused)
4. `heatmap`        `(1, 256, 256, 39)` — per-joint heatmaps (unused)
5. `landmarks_word` `(1, 117)`    — 39 × (x, y, z) metric, hip-relative

Visibility/presence are sigmoid-pre on disk; the loader applies σ
before thresholding. Joint confidence = `min(visibility, presence)`.

### 2. Mirror & Retarget

MediaPipe labels are subject-POV (their `LEFT_SHOULDER` is on the
subject's left). VulVATAR uses the selfie-style mirror — the avatar's
RIGHT side is on the subject's LEFT side — so MediaPipe's index 11
(LEFT_SHOULDER) maps to `HumanoidBone::RightShoulder`. See
`MEDIAPIPE_TO_HUMANOID` in `tracking::mediapipe`.

The world output is in metric metres with the hip midpoint as origin,
X=subject-right, Y=down, Z=away-from-camera. We axis-flip all three
components to land in the source-skeleton convention
(x ∈ [-aspect, +aspect] camera-space-right, y ∈ [-1, +1] Y-up,
+z toward camera). The Hips bone takes the midpoint of the two upper-leg
landmarks; toe tips (indices 31, 32) feed the `fingertips` map keyed
by `HumanoidBone::*Foot` so the foot heel-to-toe direction is solvable.

### 3. Hand Landmarker

For each side that has a tracked body wrist (`RightHand` for subject
left, `LeftHand` for subject right), the bbox is built from the four
auxiliary body landmarks for that hand (centroid + max distance × 1.6
padding), then cropped & resized to 224×224 NHWC.

Hand model outputs four tensors:

1. `landmarks`      `(1, 63)` — 21 × (x, y, z) screen
2. `conf`           `(1, 1)`  — hand confidence (sigmoid-pre)
3. `handedness`     `(1, 1)`  — left/right (ignored: bbox tells us)
4. `landmarks_word` `(1, 63)` — 21 × (x, y, z) metric, hand-center origin

We use the world output for finger geometry. Same axis flip as the
body, then we translate so landmark 0 (wrist) coincides with the body
wrist position. Indices 1..=20 fan out to the 15 finger phalanges
(`{Thumb,Index,Middle,Ring,Little} × {Proximal,Intermediate,Distal}`)
plus 5 distal-tip entries that go into `SourceSkeleton::fingertips`
keyed by the distal bone.

Hand confidence threshold is 0.4 (sigmoid). On low confidence, missing
bbox, or any ORT error, that side's fingers stay at rest pose for
the frame — the avatar's body still tracks.

## Feature Flags

- `inference` — enables ONNX Runtime + the `MediaPipeInference` path.
  Without it, the worker falls back to a simple skin-colour HSV
  centroid in `tracking::pose_estimation` (no body joints, just a
  rough face pose) so the GUI still has a heartbeat.
- `inference-gpu` — additionally registers the DirectML execution
  provider before the CPU fallback. The Inspector's *Inference Status*
  panel surfaces which EP the live session is running on
  (`DirectML`, `CPU`, or `CPU (DirectML unavailable: …)`), so a silent
  CPU fallback is visible.

```bash
cargo build --features "webcam inference inference-gpu"
```

## Limitations & Future Work

1. **Face blendshapes.** No ARKit-style 52-blendshape head is wired
   yet. The Pose Landmarker emits a few face landmarks (nose, eyes,
   ears, mouth corners) which could feed a basic head pose, but full
   expression coverage needs the dedicated MediaPipe Face Landmarker
   ONNX export. Until that lands, head rotation defaults to whatever
   the spine direction implies and `expressions` is empty.
2. **Single-person only.** BlazePose Heavy does not emit per-person
   bboxes — there is one implicit subject per frame. Multi-person
   support would require wiring in the `palm_detection` model (for
   bbox proposals) plus a separate person detector.
3. **Letterbox resampling.** Preprocessing still uses nearest-neighbour
   downscaling on the critical frame path to avoid pulling in a
   heavyweight imaging crate. Can be upgraded to bilinear via shaders
   or `imageops` for a small accuracy bump on tiny subjects.
