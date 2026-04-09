# Tracking Retargeting

## Purpose

This document defines how webcam-driven tracking data becomes avatar-driving input.

It covers:

- source acquisition expectations
- normalization
- confidence handling
- retargeting boundaries
- timing policy

Related documents:

- [architecture.md](/C:/lib/github/kjranyone/VulVATAR/docs/architecture.md)
- [data-model.md](/C:/lib/github/kjranyone/VulVATAR/docs/data-model.md)
- [threading-model.md](/C:/lib/github/kjranyone/VulVATAR/docs/threading-model.md)

## Core Rule

Tracking does not write final bone transforms.

Tracking produces normalized performer intent in `TrackingRigPose`.

Avatar runtime resolves that intent into the final pose.

## Pipeline

Recommended pipeline:

1. capture webcam frame
2. run landmark inference
3. build normalized body and face targets
4. attach confidence values
5. smooth temporal noise
6. publish `TrackingRigPose`
7. retarget into avatar runtime during frame update

## Coordinate Strategy

The system should distinguish:

- source image coordinates
- source camera-relative coordinates
- normalized performer coordinates
- avatar rig coordinates

Retargeting should happen from normalized performer coordinates into avatar rig coordinates, not directly from image pixels.

## Initial Scope

The first pass should prioritize:

- head orientation
- neck hint
- upper torso orientation
- upper arms
- facial weights if practical

Lower body can remain out of scope for the first tracking implementation.

## `TrackingRigPose` Contract

`TrackingRigPose` should contain:

- timestamped targets
- optional channels when tracking is absent
- confidence per channel
- expression weights

It should not contain:

- raw image buffers
- renderer commands
- final avatar bone matrices

## Confidence Policy

Each channel should have confidence.

Examples:

- head orientation confidence
- left arm confidence
- right arm confidence
- blink confidence

When confidence is low:

- keep the last stable value for a short window
- fade toward a neutral or animation-driven fallback
- avoid hard snapping to zero unless tracking is explicitly lost

## Smoothing Policy

The first implementation should support:

- position smoothing
- orientation smoothing
- expression smoothing

The smoothing layer must preserve responsiveness for head motion. Over-smoothing head rotation will feel worse than mild jitter.

## Timing Policy

Tracking is asynchronous.

The app thread should:

- read the latest completed sample once per frame
- latch that sample for the frame
- use it consistently for all simulation work in that frame

Recommended first policy:

- no prediction in the first pass
- reuse the latest valid sample if no fresh sample is available
- mark samples stale after a configurable timeout

## Retargeting Policy

Retargeting should be avatar-aware but source-agnostic.

That means:

- source-specific inference adapters produce normalized targets
- retargeting maps normalized targets into the avatar skeleton
- avatar runtime blends retargeted results with animation and secondary motion

## Blending Order

Recommended order:

1. sample animation
2. apply tracking retargeting
3. run spring simulation
4. run cloth simulation
5. build final pose

## Missing Data Policy

Examples:

- if hands are unavailable, keep head and torso active
- if face tracking is missing, keep body tracking active
- if all tracking is lost, fade toward animation or idle pose

Do not make one missing channel invalidate the whole tracking frame.

## Calibration

The design should leave room for:

- neutral pose calibration
- scale normalization
- shoulder width adjustment
- avatar-specific head offset tuning

These can be basic in the first pass, but the data path should not prevent them.

## Failure Modes To Avoid

- direct bone writes from the CV stack
- pixel-space retargeting logic mixed into avatar runtime
- blocking the frame loop waiting for fresh inference
- low-confidence frames causing hard pose snapping
