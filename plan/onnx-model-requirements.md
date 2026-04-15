# ONNX Tracking Model Investigation Requirements

## Overview

We need to investigate and select appropriate ONNX models for **Full-Body Tracking (Pose, Face, and Hands)** to be used in **VulVATAR**, a Rust-based VTuber application.

The models will run via ONNX Runtime (`ort` crate) on Windows using **DirectML** for GPU acceleration.

## 1. Tracking Requirements (The "Full Tracking" Scope)

The chosen model(s) must be able to estimate the following from a standard RGB webcam feed:

### A. Body Pose Estimation

- **Required:** Head position, Neck base, Shoulders, Spine/Torso orientation, Upper Arms, Lower Arms (Elbow to Wrist).
- **Optional/Bonus:** Lower body (hips, legs, feet).
- **Metadata:** Needs to provide confidence scores per keypoint.

### B. Face Tracking & Expressions

- **Required:** Face landmarks (e.g., 468/478 points like MediaPipe, or sufficient points to derive head orientation: pitch/yaw/roll).
- **Highly Desired:** Blendshape weights (e.g., Apple ARKit 52 blendshapes) for driving expressions like blinks, jaw opening, and mouth movements.

### C. Hand Tracking

- **Required:** Wrist, finger joints, and fingertips for both left and right hands.
- **Robustness:** Needs to handle occlusions and fast movements reasonably well.

---

## 2. Technical Requirements

### A. Format

- **Target Format:** `.onnx` (ONNX format).
- **Alternative:** Models in `.tflite` (TensorFlow Lite), PyTorch (`.pt`), or ONNX format inside `.task` files are acceptable **IF AND ONLY IF** a reliable, documented conversion pipeline to plain ONNX exists (e.g., using `tf2onnx` or `torch.onnx.export`).

### B. Input/Output Specifications

- **Inputs:** Standard image tensors (e.g., RGB, `1x3xHxW` or `1xHxWx3`). Typical resolutions like 192x192, 256x256, or 640x480.
- **Outputs:** The output tensor shapes and semantics must be clearly documented. We need to know exactly which index corresponds to which joint/landmark to map it to our internal `TrackingRigPose` structure.

### C. Performance

- **Target FPS:** 30 FPS minimum, 60 FPS preferred on consumer-grade Windows GPUs (via DirectML).
- **Latency:** Single-frame inference latency should allow for real-time interaction (ideally <15-20ms per frame for the combined pipeline).
- **Resource Footprint:** Lightweight "edge" models are preferred over massive server-grade models.

### D. Architecture Pattern

We are open to two approaches:

1. **Unified Model:** A single model that outputs Pose, Face, and Hands simultaneously (e.g., MediaPipe Holistic or RTMPose-WholeBody).
2. **Cascaded Pipeline:**
   - Person/Face Detector → Crop
   - Pose Estimator
   - Hand Estimator (using wrist crops)
   - Face Estimator (using face crops)

_Note: Unified models are strongly preferred due to simplicity in integration and often lower latency._

### E. Licensing

- Must be permissible for commercial use or open-source distribution without heavily restrictive copyleft constraints. (e.g., Apache 2.0, MIT, or similar are ideal. Research-only licenses are unacceptable).

---

## 3. Investigation Deliverables

The investigating agent should provide:

1. **Model Recommendations:** Name and source (URL/Repository) of the best candidate models for Pose, Face, and Hands (or a unified model).
2. **Conversion Steps (if any):** If the model is not natively in `.onnx`, provide the script/commands to convert it.
3. **Input/Output Math:** Describe the input tensor shape/type. Describe the output tensors (e.g., "Output 0 is [1, 33, 3] representing x, y, confidence for 33 body landmarks").
4. **Pros/Cons:** A brief summary of why to use this model versus alternatives (accuracy vs. speed trade-offs).
5. **Pre-processing / Post-processing needs:** e.g., Does it need letterbox padding? Does it output normalized `[0, 1]` coordinates or absolute pixel coordinates?
