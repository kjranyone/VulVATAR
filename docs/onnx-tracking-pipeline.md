# ONNX Whole-Body Tracking Pipeline

This document describes the design and implementation of the ONNX
Runtime-based whole-body tracking pipeline used in VulVATAR.

## Architecture Overview

The tracking system uses **RTMW3D** (mmpose / OpenMMLab) — a
3D-native single-model whole-body pose estimator running on a
background `TrackingWorker` thread so it never blocks the main
rendering loop.

```mermaid
graph TD;
    Camera[Webcam / Video Source] --> |RGB Frame| Resize[Resize to 288×384]
    Resize --> |NCHW [1,3,384,288]| RTMW3D[RTMW3D ONNX]
    RTMW3D --> |SimCC X/Y/Z heatmaps| Decode[argmax decode]
    Decode --> |133 × (nx, ny, nz)| Skeleton[Source Skeleton]
    Skeleton --> |body 17 + foot 6 + hand 21+21| Solver[pose_solver]
    Solver --> Mailbox[Tracking Mailbox]
```

## Model

A single ONNX file under `models/` (~370 MB), fetched by `dev.ps1`'s
`Install-Models` step:

| File          | Source                | Size    | Role                                |
|---------------|-----------------------|---------|-------------------------------------|
| `rtmw3d.onnx` | `Soykaf/RTMW3D-x` HF  | 370 MB  | 133 COCO-Wholebody keypoints (3D)   |

Soykaf's HuggingFace mirror hosts the official mmpose export
(`rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx`,
Apache-2). The model is the heavy "x" tier — the lighter "l" tier
exists but RTMW3D-x's higher MPJPE on COCO-Wholebody is what gets us
the front-facing yaw stability we need.

VulVATAR does **not** ship a person-bbox detector. RTMW3D is a
top-down model and ordinarily expects a tightly cropped person, but
for single-subject VTubing the camera is already framed on the user
— Kinemotion (UE5.5) demonstrated that direct frame input gives
adequate quality, and that's what this implementation does. If
multi-person or extreme-distance support becomes a requirement, a
detector (YOLOX-m or RTMDet) can be wired in front of RTMW3D without
schema changes.

## Pipeline

### 1. Preprocessing

`tracking::rtmw3d::preprocess` resizes the source frame to 288×384
(W×H) NCHW with ImageNet mean/std normalization. The aspect is
squashed (no letterbox); the model is robust to it for centred-subject
input. We use nearest-neighbour resampling on the critical frame path
to avoid pulling in a heavyweight imaging crate; it can be upgraded
to bilinear later for a small accuracy bump on tiny subjects.

### 2. SimCC decode

RTMW3D emits three SimCC (Simulated Coordinate Classification)
heatmaps in positional order:

1. `simcc_x` `(1, 133, 576)`  — argmax bin → x ∈ [0, 1]
2. `simcc_y` `(1, 133, 768)`  — argmax bin → y ∈ [0, 1]
3. `simcc_z` `(1, 133, 576)`  — argmax bin → z ∈ [0, 1]

We argmax each axis per joint and take the smaller of the X / Y peak
values (post sigmoid) as the per-joint confidence. The Z peak is
ignored for confidence — its position localises depth, not the
detection itself.

Note: the rtmlib reference implementation converts `nz` to metric
metres via `(z_index / 144 - 1) * 2.1745`. We deliberately skip this
step. The pose solver consumes bone *directions*, not absolute
positions, so the metric scale is never load-bearing — keeping z in
normalised `[0, 1]` units sidesteps having to characterise the
model's true z-range and matches the Kinemotion (UE5.5) reference
implementation's approach.

### 3. SourceSkeleton mapping

The 133 keypoints map to humanoid bones with COCO-Wholebody indices:

| Index range | Group        | Used                                     |
|-------------|--------------|------------------------------------------|
| 0–16        | Body 17      | Shoulders / arms / hips / legs           |
| 17–22       | Foot 6       | Big-toe tips → `fingertips[*Foot]`       |
| 23–90       | Face 68      | **Currently unused** (Phase C: SMIRK)    |
| 91–111      | Left hand    | 15 phalanges + 5 tip slots (avatar Right)|
| 112–132     | Right hand   | 15 phalanges + 5 tip slots (avatar Left) |

Selfie mirror is applied at mapping: subject's anatomical-left limbs
(COCO indices 5, 7, 9, 11, 13, 15 and the 91-111 hand) drive the
avatar's `Right*` bones, so the on-screen avatar mirrors the user.
See `COCO_BODY` and `attach_hand` in `tracking::rtmw3d`.

The hip mid-point (average of indices 11 and 12) is the depth origin.
All emitted positions are hip-relative, in source-space `[-aspect,
+aspect] × [-1, 1] × z` coords with `+z` toward the camera.

### 4. Coordinate-system conversion

Image-space → source-space:

| Source axis | Formula                                 |
|-------------|-----------------------------------------|
| `src.x`     | `-(nx - hip_nx) * 2 * aspect`           |
| `src.y`     | `-(ny - hip_ny) * 2`  (Y-down → Y-up)   |
| `src.z`     | `-(nz - hip_nz) * 2`                    |

The x-axis is *negated* on top of the aspect scaling so that COCO's
anatomical-left keypoints (which appear at high image-x because the
subject faces the camera) end up at source `-x` — that's where the
avatar's `Right*` bones rest after the VRM 0.x Y180 root flip.

## Feature Flags

- `inference` — enables ONNX Runtime + the `Rtmw3dInference` path.
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

1. **Face / blendshapes are empty.** RTMW3D emits 68 face landmarks
   (indices 23-90) that this implementation does not yet decode.
   The head bone falls back to spine direction, and ARKit blendshape
   coefficients (smile / blink / jaw / brow) are not produced. The
   Phase C plan is to add a SMIRK FLAME-regression head that emits
   FLAME params, plus a FLAME-50 → ARKit-52 retarget matrix.
2. **Occluded-hand depth.** Monocular 3D pose places hands behind
   the body when an instrument or object occludes the wrist
   (e.g. guitar strumming hand). RTMW3D handles this better than
   the previous BlazePose pipeline but not perfectly. A planned
   follow-up: temporal hold + arm-length sanity clamp on
   low-confidence wrist keypoints.
3. **Single-person only.** RTMW3D is a top-down model and we drive
   it with the full camera frame. With more than one person in
   frame the keypoint outputs will average over both subjects.
   Multi-person support would require a YOLOX or RTMDet person
   detector running in front of RTMW3D per detected bbox.
4. **Aspect distortion.** Preprocessing squashes the camera frame
   (typically 16:9) into the model's 4:3 input without letterbox.
   At wide aspect ratios this slightly compresses limbs in y; in
   practice the model is robust enough that the rendered avatar
   doesn't show the distortion, but a letterbox path is easy to
   add if it becomes a problem.
