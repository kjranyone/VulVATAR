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
- **Target OS:** Windows (Execution parallelized natively via ORT, utilizing GPU acceleration like DirectML when the pipeline expands).
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

## Feature Flag

The full ML-baseline requires the `inference` feature to compile in order to prevent forcing large AI dependencies on lightweight builds.

```bash
cargo build --features "webcam inference"
```

If the `inference` feature is explicitly omitted, the pipeline safely falls back to the previous simple skin-color heuristic.

## Future Work & Limitations

The current baseline skips a few capabilities to streamline the setup process:
1. **YOLOX Integration:** Currently skips the YOLOX bounding box object detection in favor of squashing the entire frame. For better hand/facial tracking, adding the YOLOX detector to crop specifically to the actor's body bounds is recommended.
2. **Quality of Resampling:** Uses nearest-neighbor to avoid bloated imaging package constraints on the critical frame path. Can be upgraded to Bilinear using shaders or `imageops` for slight accuracy bumps.
3. **Hardware Execution Provider (DirectML):** To utilize GPUs natively on Windows, the `directml` execution provider flag should be integrated firmly into the SessionBuilder once native build paths (like CMake dependencies) are fully stabilized in the target environment.
