# Tracking Retargeting

## Purpose

This document defines how RealSense D435 depth-driven tracking data becomes avatar-driving input.

It covers:

- source acquisition expectations
- normalization
- confidence handling
- retargeting boundaries
- timing policy

Related documents:

- [architecture.md](architecture.md)
- [data-model.md](data-model.md)
- [threading-model.md](threading-model.md)

## Core Rule

Tracking does not write final bone transforms.

Tracking produces normalized performer intent in `TrackingRigPose`.

Avatar runtime resolves that intent into the final pose.

## Pipeline

Recommended pipeline:

1. capture RealSense D435 depth + colour frame
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

## Observability Contract

**Confidence and observability are different axes, and detector scores lie
on both** (2026-07-27 live audit: RTMW3D put score ≈ 0.6 on fully
hallucinated off-frame hands; FaceMesh put score ≈ 0 on a correctly
tracking profile face). Every producer therefore answers a geometric
question before publishing: *can this quantity be measured from this
camera at this framing at all?*

The rule: **an unobservable quantity is never published** (or is
published explicitly marked), regardless of its detector score. A score
low-pass or threshold is NOT a substitute — unobservable values are not
noisy, they are fiction.

Per-quantity observability tests currently in force:

| Quantity | Test | Where |
| --- | --- | --- |
| Depth-path joints | keypoint in frame + person-band depth sample; ray fallback only for in-frame holes | `skeleton_from_depth` (`keypoint_in_frame`, `sample_metric_point_person`) |
| Joint provenance | `JointOrigin::{Observed, Extrapolated, Synthesized}` on every depth joint; statistical consumers refuse non-`Observed` | `source_skeleton::JointOrigin` |
| Hand blocks | in-frame MCP knuckles (3-of-4) + visibility-duty engage gate | `rtmw3d::skeleton::attach_hand`, `ArmEngageGate` |
| Hand 3D (depth path) | depth is a REFINEMENT, not a requirement: own/centroid person-band sample when it exists, else the 2D wrist ray on the sampled-knuckle plane, else a forearm-length ray–sphere solve from the elbow — depth under a near-limit hand is a hole in half the frames (2026-07-28 bisect) while its 2D stays valid, so requiring depth made the hand blink at frame rate. Depth-free paths demand the full block strictly inside the frame (border-clamped desk hands must not be pinned) and are marked `Extrapolated`. | `skeleton_from_depth::attach_hand` |
| Palm orientation | all four basis landmarks in frame (monocular path); metric MCP samples, plane-resolved through depth holes (depth path) | `rtmw3d::skeleton`, `skeleton_from_depth::attach_hand` |
| Ear-line head yaw / eye-line roll | horizontal baseline ≥ 15 % / 10 % of the head's keypoint spread — a collapsed baseline divides by hallucinated z | `rtmw3d::face::derive_face_pose_from_body` |
| Mesh head pose | selector trusts the pose stream's continuity over the mesh's own flag score during dips | `rtmw3d::face::FaceSourceSelector` |
| Root / anchor | person-band gating + velocity spike gate + ring-consistency reseed | `TorsoScaleStabilizer` |

Known accepted gap: the RTMW3D-only (no external depth) legacy path
publishes off-frame body joints gated by score alone. The production
D435 path rebuilds the skeleton and never consumes them; only synthetic
benches without depth siblings can observe this.

## Degradation Policy

When an observable channel LOSES observability, every channel follows
the same shape — **hold → ease → re-engage** — implemented per channel
where the required state lives:

1. **Hold**: republish the last real measurement for a short window with
   linearly decaying confidence, marked `Extrapolated`
   (`hand_hold::HandHold` for hands/forearms; the mailbox hold window
   for whole-sample loss; the solver's rotation hold for end-effector
   twist).
2. **Ease**: as the decayed confidence crosses the solver gates, the
   affected bones blend toward their idle pose (`apply_idle_arm_pose`,
   face crossfade) instead of snapping. A per-frame blend alone does NOT
   achieve this: the arm's target is discontinuous at the switch, so a
   lerp toward "the current target" just yields a blend-fraction-sized
   step (measured 0.4–0.7 m of avatar hand travel in one frame). The arm
   chain therefore also carries a rotation-rate bound
   (`ARM_MAX_STEP_PER_FRAME`), which converts any target discontinuity
   into a short traversal while leaving real motion untouched.
3. **Re-engage**: re-acquisition must prove itself before driving the
   avatar again where flicker is possible (`ArmEngageGate` visibility
   duty for hands; `FaceSourceSelector` Schmitt + crossfade for the head
   pose source), and eases back in through the same blends.

Nothing in the hold path fabricates data: a held value is always the
last real measurement, and its provenance says so.

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

### Head orientation is absolute, not stacked

The face pose (yaw / pitch / roll) is measured against the image and
neutral-corrected by calibration, so it is an **absolute** head
orientation — not an offset from the neck. `apply_face_pose` therefore
sets the Head bone's *world* rotation from it and divides the parent
chain's rotation out; it does not compose the two.

The rule this encodes: exactly one channel owns each degree of freedom.
The neck / upper-spine chain owns torso posture (chin thrust, forward-head
lean); the face pose owns gaze. Composing them double-counts every
upstream rotation into where the avatar looks. Measured cost of the old
composed form (2026-07-27, 1123 live frames): the source Head sits a
steady +0.20 shoulder-spans in front of the shoulder midpoint — part real
desk posture, part depth-surface sampling geometry — which bows the chain
~21°, and that bow landed on gaze as a permanent look-down
(`head_elev = −0.79·pitch − 0.156·yaw − 21.5` degrees).

Consequence to keep in mind: when the face channel is dead the head no
longer inherits torso yaw. That is the correct trade — a stale
face-derived gaze is the smaller error — but it means a *dead* face
channel must publish nothing rather than publish zeros.

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

See [calibration-ux.md](calibration-ux.md) for the capture UX and
data model: neutral pose calibration (anchor + face neutrals +
neutral body yaw), scale normalization (`shoulder_span_m` →
`reference_span_m`/`mpsu`), and the per-profile persistence model.
Avatar-specific head offset tuning remains future work.

These can be basic in the first pass, but the data path should not prevent them.

## Failure Modes To Avoid

- direct bone writes from the CV stack
- pixel-space retargeting logic mixed into avatar runtime
- blocking the frame loop waiting for fresh inference
- low-confidence frames causing hard pose snapping
