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
    Camera[Webcam / Video Source] --> |RGB Frame| Yolox[YOLOX-m person bbox]
    Yolox --> |bbox + 25% pad| Crop[Crop to subject]
    Crop --> |RGB crop| Resize[Resize to 288×384]
    Resize --> |NCHW [1,3,384,288]| RTMW3D[RTMW3D ONNX]
    RTMW3D --> |SimCC X/Y/Z heatmaps| Decode[argmax decode]
    Decode --> |133 × (nx, ny, nz)| Remap[Remap crop → orig frame]
    Remap --> |133 × (nx, ny, nz)| Skeleton[Source Skeleton]
    Skeleton --> |body 17 + foot 6 + hand 21+21| Solver[pose_solver]
    Solver --> Mailbox[Tracking Mailbox]
```

## Models

Four ONNX files under `models/` (~478 MB total), fetched by
`dev.ps1`'s `Install-Models` step:

| File                    | Source                       | Size    | Role                                                  |
|-------------------------|------------------------------|---------|-------------------------------------------------------|
| `rtmw3d.onnx`           | `Soykaf/RTMW3D-x` HF         | 370 MB  | 133 COCO-Wholebody keypoints (3D)                     |
| `yolox.onnx`            | mmpose `rtmposev1` onnx_sdk  | 101 MB  | Human-Art tuned YOLOX-m person bbox detector          |
| `face_landmark.onnx`    | PINTO 410 FaceMeshV2         | 4.8 MB  | 478 dense face landmarks                              |
| `face_blendshapes.onnx` | PINTO 390 BlendshapeV2       | 1.8 MB  | 146 face landmarks → 52 ARKit blendshape coefficients |

Soykaf's HuggingFace mirror hosts the official mmpose RTMW3D export
(`rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx`,
Apache-2). The model is the heavy "x" tier — the lighter "l" tier
exists but RTMW3D-x's higher MPJPE on COCO-Wholebody is what gets us
the front-facing yaw stability we need.

YOLOX-m is the `end2end.onnx` shipped in mmpose's `rtmposev1`
ONNX SDK bundle (`yolox_m_8xb8-300e_humanart-c2c7a14a.zip`,
Apache-2 / OpenMMLab). The export bakes NMS into the graph so
post-processing on our side is just a max-score gather over the
per-class detections.

The 101 MB YOLOX-m is optional — if the file is missing, the
pipeline falls through to whole-frame RTMW3D inference
(Kinemotion-style). With YOLOX present, average overall_confidence
across the 8-case sample sweep moves from ~0.652 (whole-frame
bilinear) to ~0.683 (bbox-cropped, +25% pad), and weak categories
(small / off-centre / cross-step subjects) gain +0.030 to +0.040.

## Pipeline

### 0. Person bbox crop (YOLOX-m)

When `models/yolox.onnx` is present, `tracking::yolox` runs first on
the full camera frame. The Human-Art tuned YOLOX-m emits post-NMS
detections (`dets [N, 5]`, `labels [N]`); we pick the highest-score
person bbox, expand it by 25% on each side, clamp to image bounds,
and crop the RGB to that rectangle. RTMW3D then runs on the crop and
its `(nx, ny)` outputs are remapped back to original-frame
normalised coords before the source-skeleton step.

The 25% pad matters: 10% padding regressed the cross-step / walking
poses (-0.018 confidence) where a leading limb extended beyond the
YOLOX bbox. 25% gives the model enough slack while still keeping
the subject filling most of the model input.

YOLOX is preprocessed with letterbox to 640×640, BGR raw `[0, 255]`,
no normalization — matching mmdetection's default YOLOX preprocessing
config.

### 1. RTMW3D preprocessing

`tracking::rtmw3d::preprocess` resizes the (cropped) source frame to
288×384 (W×H) NCHW with ImageNet mean/std normalization. The aspect
is squashed (no letterbox); the model is robust to it for
centred-subject input. We use OpenCV-style bilinear resampling so
small subjects don't lose keypoint detail to nearest-neighbour
quantisation — this matches RTMW3D's training-time preprocessing
distribution and lifts overall_confidence on small / distant subjects
by +0.005 to +0.020 in our sample sweep.

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
| 23–90       | Face 68      | Face bbox derivation for FaceMesh cascade|
| 91–111      | Left hand    | 15 phalanges + 5 tip slots (avatar Right)|
| 112–132     | Right hand   | 15 phalanges + 5 tip slots (avatar Left) |

Selfie mirror is applied at mapping: subject's anatomical-left limbs
(COCO indices 5, 7, 9, 11, 13, 15 and the 91-111 hand) drive the
avatar's `Right*` bones, so the on-screen avatar mirrors the user.
See `COCO_BODY` and `attach_hand` in `tracking::rtmw3d`.

The hip mid-point (average of indices 11 and 12) is the depth origin.
All emitted positions are hip-relative, in source-space `[-aspect,
+aspect] × [-1, 1] × z` coords with `+z` toward the camera.

Wrist positions are *not* read from the COCO-Wholebody body wrists
(indices 9 / 10) or hand-track wrists (91 / 112) — see §5 for the
MCP-centroid derivation and the resilience layer that wraps it.

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

### 5. Wrist resilience layer

Monocular 3D wrist keypoints are the brittlest part of any
single-camera pose pipeline — they routinely fall behind the torso
when occluded by an instrument, drift across the body when arms
cross, or evaporate to low confidence on extreme yaw. A four-stage
fallback in `tracking::rtmw3d` keeps the avatar's hands stable
through these failure modes. Stages 1–2 run inline in
`build_source_skeleton`'s `attach_hand`; stages 3–4 run in
`apply_wrist_temporal_hold` per side.

**Stage 1 — MCP-centroid wrist derivation.** The wrist position
emitted to the source skeleton is the centroid of the four finger
MCP joints (local indices 5 / 9 / 13 / 17 — index, middle, ring,
little knuckles), *not* the COCO body wrist (idx 9 / 10) or the
hand-track wrist (idx 91 / 112). Both wrist landmarks become wildly
unreliable when the actual wrist is occluded — e.g. a guitar
strumming hand whose wrist is hidden behind the instrument lands
~0.3 units behind the torso in normalised z. The MCP knuckles, by
contrast, anchor on the visible part of the hand. Averaging four of
them produces a stable hand position that respects the actual
monocular geometry. The centroid sits ~half a hand-length forward
of the true wrist; the solver re-anchors the chain at the same
point regardless, so this offset is visually invisible.

A hand is considered tracked only when ≥3 of the 4 MCPs clear the
0.3 confidence floor. Below that, the entire hand chain is dropped
for the frame so the solver leaves fingers at rest (preferable to
emitting a noisy wrist that flicks around).

**Stage 2 — Forearm length sanity check.** `apply_wrist_temporal_hold`
runs the live MCP-centroid wrist through a length plausibility
check: `||wrist - elbow|| ≤ 1.4 × max(running_max_upper_arm, 0.18)`.
The bound rides on a per-side running max of clean upper-arm
lengths, with a 0.18 floor so the very first frame never rejects
the wrist outright. The 1.4× ratio limit is loose enough for normal
arm posture (real arms have upper-arm ≈ forearm) but tight enough
to catch the "wrist hallucinated across the body" pathology where
RTMW3D snaps the wrist past the opposite shoulder.

**Stage 3 — Temporal hold (≤6 frames).** When the live wrist is
rejected (stage 2) or absent (stage 1 returned no hand), reuse the
previous frame's accepted wrist position for up to
`TEMPORAL_HOLD_MAX_FRAMES = 6` frames, decaying confidence by 15%
per frame (`TEMPORAL_HOLD_DECAY = 0.85`). At 30 fps that's a 200 ms
window — long enough to bridge a strum stroke or a hand pass behind
the body, short enough that a genuinely-stationary hand stays at
its real position.

When live fingers exist alongside a rejected wrist (the wrist was
implausible but the MCPs were OK), the entire hand chain is
translated by `held_wrist - bogus_wrist` so finger geometry around
the held wrist stays self-consistent — without this, the held
wrist + live fingers would form a tear in the chain.

**Stage 4 — Chain-extension synthesis.** When the temporal hold
window expires *and* both shoulder and elbow are still tracked, the
wrist is synthesised by extending forward from the elbow along the
upper-arm direction by the upper-arm length, with a small +0.2 z
bias toward the camera so the synthesised hand reads as "extended
in front" rather than "stuck inside the torso". The finger chain is
cleared (its geometry was suspect anyway) and the wrist is emitted
with confidence 0.35 — just above the solver's default 0.3 threshold
so the chain still participates, but visibly damped relative to a
real measurement so the value is distinguishable in confidence
histograms.

If neither the held position nor the synth path is available
(no shoulder, no elbow, no recent good wrist), the wrist + finger
chain are dropped and the solver leaves the hand at rest.

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

## Face cascade

If `face_landmark.onnx` and `face_blendshapes.onnx` are present, a
face inference cascade runs after the body track:

1. **Face bbox derivation.** RTMW3D's 68 face landmarks (indices
   23..=90, dlib 68-point convention with jawline / brows / nose /
   eyes / mouth) define a centroid and `max_d` to the farthest
   landmark; the bbox is centred on the centroid with `1.3× max_d`
   half-size. The face-68 set spans the whole face including chin,
   so unlike the body-track 5-point face landmarks (nose / eyes /
   ears) the bbox does not clip the lower face.
2. **FaceMeshV2** runs on a 256×256 NCHW crop, emitting 478 dense
   face landmarks plus a per-frame face-presence confidence (sigmoid
   threshold 0.4).
3. **BlendshapeV2** receives a 146-landmark subset of the FaceMesh
   output (the canonical `kLandmarksSubsetIdxs` from MediaPipe's
   `face_blendshapes_graph.cc`) and emits 52 ARKit-style blendshape
   coefficients.
4. The 52 weights are passed through verbatim under their ARKit
   names (`mouthSmileLeft`, `eyeBlinkLeft`, `jawOpen`, …) and *also*
   aggregated into VRM 1.0 preset names (`blink` / `aa` / `happy` /
   `sad` / `surprised`) so a stock rig with no custom expressions
   still gets live facial motion. Selfie mirror applies — the
   subject's left eye drives `blinkRight`.

Head pose (yaw / pitch / roll) is computed independently from
RTMW3D's body face keypoints (0=nose, 1=left_eye, 2=right_eye,
3=left_ear, 4=right_ear) which already sit in source-skeleton 3D
coords. Using FaceMeshV2's z output for head pose is avoided
because that z is in arbitrary face-relative pixel units —
re-projecting it into the body frame would need camera intrinsics
we don't have.

## Limitations & Future Work

1. **Occluded-hand depth.** Monocular 3D pose places hands behind
   the body when an instrument or object occludes the wrist
   (e.g. guitar strumming hand). RTMW3D handles this better than
   the previous BlazePose pipeline but not perfectly. The four-stage
   wrist resilience layer (§5) keeps the avatar's hands stable
   through the failure modes that remain.
2. **Single-person only.** YOLOX-m returns multiple detections but
   we always pick the single highest-score person. With more than
   one person in frame, RTMW3D will track whichever the detector
   ranks first per frame — potentially flickering between subjects.
   For multi-subject support, the picker would need a stable
   identity (size + previous-frame proximity) rather than picking
   per-frame top score.
3. **Aspect distortion.** Even with the YOLOX crop, the cropped
   region is squashed to 288×384 without letterbox. With a 25%-padded
   bbox the residual aspect mismatch is small (typical subject
   crops are close to the model's 3:4) but still present. The
   model is robust enough in practice; a letterbox path inside the
   crop preprocessing is easy to add if it becomes a problem.
