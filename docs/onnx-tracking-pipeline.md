# ONNX Full-Body Tracking Pipeline

This document describes the design and implementation of the ONNX Runtime-based full-body tracking pipeline used in VulVATAR.

## Architecture Overview

The tracking system has been upgraded from a basic skin-color HSV thresholding to an ML-based inference pipeline using **CIGPose**. The implementation operates independently in a background `TrackingWorker` thread to prevent blocking the main rendering loop.

```mermaid
graph TD;
    Camera[Webcam / Video Source] --> |RGB Frame| Preprocess[Preprocessing]
    Preprocess --> |NCHW Tensor [1,3,H,W]| ORT[ONNX Runtime Session]
    ORT --> |simcc_x, simcc_y| Decode[SimCC Decoding]
    Decode --> |[x,y,conf] * 133| Map[Rig Retargeting]
    Map --> |TrackingRigPose| Mailbox[Tracking Mailbox]
```

## Current Baseline Implementation

The pipeline has been initially implemented as a "baseline" utilizing `cigpose-onnx`.

### 1. Requirements & Dependencies
- **Engine:** `ort` (Rust wrapper for ONNX Runtime v2.x).
- **Tensor Operations:** `ndarray`.
- **Target OS:** Windows (Execution parallelized natively via ORT; the
  DirectML execution provider is registered first when built with the
  `inference-gpu` feature, falling back to the CPU EP if registration
  fails. The Inspector's *Inference Status* panel surfaces which EP
  the session ended up on — `DirectML`, `CPU`, or
  `CPU (DirectML unavailable: …)` — so a silent CPU fallback is
  visible to the operator.
- **Model:** `cigpose-m_coco-wholebody_256x192.onnx` (133 keypoints covering body, face, and hands).

### 2. Pipeline Steps

#### Preprocessing
- Currently, the entire webcam frame is used (assuming the user is decently centered).
- A custom high-performance Nearest-Neighbor downscaling algorithm translates the input to the model's required `192x256` dimension.
- The tensor is normalized using ImageNet configurations (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

#### Inference
- Executed on a dedicated thread, outputting `simcc_x` and `simcc_y` logs.

#### SimCC Decoding
- Calculates the `argmax` over the X and Y coordinate bins.
- Scales the resultant coordinates down by the model's `split_ratio` (default `2.0`) to retrieve pixel-space keypoints.
- Calculates point confidence using the logarithmic response.

#### TrackingRigPose Mapping
- Extracts specific keypoints from the 133-point COCO-Wholebody mapping:
  - `0`: Nose.
  - `5, 6`: Left / Right Shoulder.
  - `7, 8`: Elbows, etc.
- Maps the predicted points into the normalized Screen-Space `[-1, 1]` with aspect ratio corrections.
- Approximates Head Orientation (yaw/pitch/roll) using offset locations (e.g., nose).

#### Expression Weights from Face Landmarks
CIGPose does not emit ARKit-style blendshape weights. The face block
(iBUG 68-point, indices 23..=90 of the 133-point output) is fed
through `tracking::face_expression::FaceExpressionSmoother` to
derive a small VRM 1.0 preset subset:

- `blinkLeft` / `blinkRight` / `blink` — Eye Aspect Ratio (EAR) per
  eye, with an input-side Schmitt trigger to debounce noisy frames.
- `aa` — Mouth Aspect Ratio (MAR), inner-mouth height over outer
  width.
- `happy` — mouth-corner elevation above the mouth-center y axis,
  normalised by face width.
- `surprised` — brow lift above the eye line, AND-gated against
  `aa` so a raised brow alone does not register as surprise.

The smoother holds per-channel state (asymmetric attack/release time
constants, `Instant`-driven dt) so live output eases rather than
jitters. ARKit 52-blendshape coverage is intentionally out of scope
for v1 — adding it would mean a separate face-only ONNX model in
the pipeline.

Subject ↔ avatar mirroring follows the body convention (subject-left
keypoints drive avatar-right bones), so iBUG's "left eye" — the
subject's left — drives `blinkRight` on the avatar.

## Feature Flag

The full ML-baseline requires the `inference` feature to compile in order to prevent forcing large AI dependencies on lightweight builds.

```bash
cargo build --features "webcam inference"
```

If the `inference` feature is explicitly omitted, the pipeline safely falls back to the previous simple skin-color heuristic.

## Future Work & Limitations

1. **Resampling quality.** Preprocessing still uses nearest-neighbor
   downscaling on the critical frame path to avoid pulling in a
   heavyweight imaging crate. Can be upgraded to bilinear via shaders
   or `imageops` for a small accuracy bump.
2. **ARKit-grade expression coverage.** The face_expression heuristic
   covers ~6 VRM presets — enough for "the avatar is alive" but not a
   replacement for a dedicated face blendshape model. A separate
   face-only ONNX (e.g. MediaPipe FaceLandmarker with blendshapes)
   would be a second pipeline stage, not a swap-in.
3. **Lower-body fidelity trade-off.** The `wholebody` CIGPose variant
   emits hip/knee/ankle keypoints but at lower input resolution, so
   the *Lower-body Tracking* toggle costs hand fidelity on the same
   tier of model. Resolution-matched whole-body models would close
   that gap.

YOLOX person-crop integration and DirectML EP registration both
shipped with T9 / T10 and are no longer aspirational. The Inspector's
*Inference Status* panel reports which EP the live session is
running on.
