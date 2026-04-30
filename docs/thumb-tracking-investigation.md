# Thumb tracking — open investigation

Tracked symptom: when the user opens the thumb wide (e.g. thumbs-up), the
avatar's thumb does **not** move as much. The other four fingers track
correctly. This file captures the working hypothesis and the diagnostic
hooks that were added to narrow it down — once the relevant logs are
available, this doc should be updated with the conclusion (or replaced
by the eventual fix's commit message).

## What we already know is correct

* COCO-WholeBody hand keypoint indices (1=CMC, 2=MCP, 3=IP, 4=tip) map
  cleanly to `LeftThumbProximal / Intermediate / Distal` and
  `fingertips[LeftThumbDistal]` — see `HAND_PHALANGES_LEFT` /
  `HAND_TIPS_LEFT` in `src/tracking/rtmw3d.rs`. Naming follows VRM 0.x
  where `ThumbProximal` is the thumb metacarpal in anatomy.
* The wrist orientation pass (`solve_wrist_orientation` in
  `pose_solver.rs`) uses index/middle/pinky MCPs to derive the palm
  plane — the thumb is intentionally excluded so it cannot bias the
  wrist frame, but the chain it parents to the wrist still inherits a
  3-DoF wrist rotation.
* Confidence floor is 0.05 (`KEYPOINT_VISIBILITY_FLOOR`); thumb keypoints
  rarely fall below this even when noisy.

## Why the thumb is structurally fragile

* The thumb's "open wide" motion is concentrated in **one** joint
  (CMC abduction). Other fingers spread the curl over three joints.
  A single noisy direction-match step has nowhere to hide.
* The metacarpal segment (CMC → MCP) is the **shortest** thumb
  segment. With selfie-distance webcams the source-space length is on
  the order of 0.02-0.04 normalized units — close to detection noise.
* `vec3_normalize` of a short, noisy vector amplifies angular error.

## Three working hypotheses

| | Hypothesis | What logs would show |
|-|-|-|
| **(A)** | RTMW3D simply doesn't detect thumb keypoints reliably during the gesture. | `src CMC=none` / `MCP=none` lines; or wildly fluctuating positions frame-to-frame. |
| **(B)** | Keypoints are present but the **metacarpal segment is too short** for the direction-match to recover abduction without amplifying noise. | `seg meta=` < 0.005, or seg present but `rot prox` near 0° while user is doing thumbs-up. |
| **(C)** | Math is fine and rotation magnitude is correct, but VRM rest-pose orientation makes the abduction look smaller than expected. | `rot prox` is reasonably large (≥ 30°) but the visible avatar thumb still doesn't move much. |

## Diagnostic hook (already landed)

`src/avatar/pose_solver.rs::log_thumb_diagnostics` emits one info-level
line per hand per frame, gated by the `VULVATAR_DEBUG_THUMB`
environment variable. Run as:

```bash
VULVATAR_DEBUG_THUMB=1 RUST_LOG=vulvatar=info cargo run \
  2> profile/thumb_$(date +%Y%m%d_%H%M%S).log
```

Each line carries:

```
thumb[L] src CMC=(±x,±y,±z) MCP=… IP=… tip=…
         | seg meta=… prox=… dist=…
         | rot prox=…° inter=…° dist=…°
```

`src` = the source-skeleton position the solver actually consumed for
each thumb keypoint (post 1€-filter), `seg` = consecutive-segment
length in source units, `rot` = angle between the bone's *current*
local rotation and its rest local rotation.

## Action plan

1. **Capture a sample log.** Hold a relaxed hand for ~3 s, open the
   thumb wide for ~3 s, return to relaxed for ~3 s. Do this for both
   hands.
2. **Classify against the table above.** The seg / rot columns make
   (A) / (B) / (C) distinguishable in seconds.
3. **Pick a fix.**
   * (A) → tune the model picker / drop a stricter person-bbox
     pad ratio so the thumb gets more pixels.
   * (B) → drive the thumb chain via tip-driven IK using the highest-SNR
     keypoint (COCO 4, the thumb tip). The metacarpal abduction falls
     out of solving for the tip. Other fingers stay on the current
     direction-match.
   * (C) → bake a thumb-specific rest-pose alignment into the avatar
     load path. Less likely to be the right answer for arbitrary VRMs;
     prefer (B) unless the logs really point this way.
4. **Update this doc** with the chosen route and remove the diagnostic
   block once the fix is verified.
