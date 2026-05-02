//! Map decoded RTMW3D keypoints into a [`SourceSkeleton`] with hip-/
//! shoulder-anchored 3D bone positions, plus the per-hand chain
//! attachment that includes the palm orientation derivation.
//!
//! The wrist *position* emitted here is the MCP centroid (see
//! [`attach_hand`] for why we don't trust either of RTMW3D's own
//! wrist landmarks); the wrist resilience layer in [`super::wrist`]
//! validates that position against the upper-arm chain afterwards.

use crate::asset::HumanoidBone;

use super::super::source_skeleton::HandOrientation;
use super::super::{SourceJoint, SourceSkeleton};
use super::consts::{
    COCO_BODY, COCO_FOOT_TIPS, HAND_PHALANGES_LEFT, HAND_PHALANGES_RIGHT, HAND_TIPS_LEFT,
    HAND_TIPS_RIGHT, KEYPOINT_VISIBILITY_FLOOR,
};
use super::decode::{DecodedJoint, NUM_JOINTS, RTMW3D_SOURCE_Z_SCALE};
use super::math::{length3, sub3};

pub(super) fn build_source_skeleton(
    frame_index: u64,
    joints: &[DecodedJoint],
    width: u32,
    height: u32,
) -> SourceSkeleton {
    let mut sk = SourceSkeleton::empty(frame_index);
    if joints.len() < NUM_JOINTS {
        return sk;
    }

    // Origin selection. Hips (COCO 11/12) are the canonical choice
    // because every body bone direction is most stable when measured
    // from the pelvis. But VTuber webcam framing very commonly cuts
    // the lower body — so when hips are not visible we fall back to
    // shoulder mid (COCO 5/6), which is essentially always in frame
    // for upper-body shots. The solver consumes *relative* bone
    // directions so the absolute origin location does not matter as
    // long as it is the same point each frame.
    //
    // If neither pair is visible the subject is either out of frame
    // or so poorly detected that any rotation we infer would be
    // garbage; only then do we bail.
    let lh = &joints[11];
    let rh = &joints[12];
    let hip_score = lh.score.min(rh.score);
    let hip_visible = hip_score >= KEYPOINT_VISIBILITY_FLOOR;

    let (origin_nx, origin_ny, origin_nz) = if hip_visible {
        (
            (lh.nx + rh.nx) * 0.5,
            (lh.ny + rh.ny) * 0.5,
            (lh.nz + rh.nz) * 0.5,
        )
    } else {
        let ls = &joints[5];
        let rs = &joints[6];
        let shoulder_score = ls.score.min(rs.score);
        if shoulder_score < KEYPOINT_VISIBILITY_FLOOR {
            // No reliable hips *and* no reliable shoulders — there is
            // no anatomical anchor we trust. Leave the skeleton empty
            // so the solver leaves the avatar at rest pose this
            // frame instead of latching onto noise.
            return sk;
        }
        (
            (ls.nx + rs.nx) * 0.5,
            (ls.ny + rs.ny) * 0.5,
            (ls.nz + rs.nz) * 0.5,
        )
    };

    // Source-space coord conversion. Aspect is recovered by scaling
    // the x axis: image x in [0, 1] maps to source x in
    // `[-aspect, +aspect]` where aspect = width/height. y is flipped
    // (image y-down → source y-up). z is centred on hip mid, sign-chosen
    // so `+z` is toward the camera, and scaled like rtmlib's
    // `(z / (input_h / 2) - 1) * z_range` conversion. Because we subtract
    // the same origin for every joint, only the relative part matters:
    // `Δz_source = Δz_bin / input_h * z_range`.
    //
    // The x axis is *negated* on top of that: COCO's anatomical-left
    // keypoints (e.g. index 5 LEFT_SHOULDER) sit at high image-x
    // because the subject faces the camera mirrored. Selfie mirror
    // routes those into avatar `Right*` bones, whose rest position
    // (post-VRM-Y180 root flip) is at source `-x`. So subject-left
    // (high norm_x) must end up at source `-x`. The negation makes
    // that hold.
    let aspect = width as f32 / height.max(1) as f32;
    let to_source = |j: &DecodedJoint| -> [f32; 3] {
        let sx = -(j.nx - origin_nx) * (2.0 * aspect);
        let sy = -(j.ny - origin_ny) * 2.0;
        let sz = -(j.nz - origin_nz) * RTMW3D_SOURCE_Z_SCALE;
        [sx, sy, sz]
    };

    // Mark width/height used so the unused-warning is silenced
    // when the code path skips downstream consumers of those values.
    let _ = (width, height);

    // Hips: only emit when the hip pair was actually detected. When
    // the origin came from the shoulder fallback above, the avatar's
    // pelvis bone is left at rest pose — the solver decides what
    // (if anything) to do with the absent Hips entry.
    //
    // `Spine` shares the same source position as `Hips` (the pelvic
    // anchor): COCO has no spine keypoint, but the solver drives the
    // Spine bone via `Hips → ShoulderMidpoint`, so it needs an entry
    // in `source.joints` for the *base* lookup or it will silently
    // skip the bone (the solver returns early when `source.joints
    // .get(&bone)` is None). Without this insertion the avatar never
    // bends at the waist no matter how far the subject leans forward,
    // because the spine direction signal never reaches the solver.
    if hip_visible {
        let hip_joint = SourceJoint {
            position: [0.0, 0.0, 0.0],
            confidence: hip_score,
        };
        sk.joints.insert(HumanoidBone::Hips, hip_joint);
        sk.joints.insert(HumanoidBone::Spine, hip_joint);
    }

    // Body bones. Emit every keypoint with its actual confidence and
    // let the downstream solver apply its (user-tunable) confidence
    // threshold. Filtering here would pre-empt the GUI slider, which
    // is exactly what hid the upper-body framing case from the user
    // before this rewrite.
    //
    // Exception: when the hip pair was not detected (= we fell back
    // to shoulder origin, meaning the lower body is structurally out
    // of frame), drop the lower-body indices (hips through ankles,
    // COCO 11..=16). The model still emits hallucinated guesses for
    // those joints — usually clustered near the bottom of the crop —
    // and they pass the visibility floor at low scores. Without this
    // skip the avatar's legs flail as the model's hallucinations
    // wander frame to frame.
    const LOWER_BODY_FIRST_IDX: usize = 11;
    for &(idx, bone) in COCO_BODY {
        if idx >= joints.len() {
            continue;
        }
        if !hip_visible && idx >= LOWER_BODY_FIRST_IDX {
            continue;
        }
        let j = &joints[idx];
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            continue;
        }
        sk.joints.insert(
            bone,
            SourceJoint {
                position: to_source(j),
                confidence: j.score,
            },
        );
    }

    // Foot tips. Same upper-body-only suppression: if hips weren't
    // visible the toes definitely aren't — anything the model emits
    // for COCO 17..=22 (toe / heel landmarks) is a hallucination.
    if hip_visible {
        for &(idx, bone) in COCO_FOOT_TIPS {
            if idx >= joints.len() {
                continue;
            }
            let j = &joints[idx];
            if j.score < KEYPOINT_VISIBILITY_FLOOR {
                continue;
            }
            sk.fingertips.insert(
                bone,
                SourceJoint {
                    position: to_source(j),
                    confidence: j.score,
                },
            );
        }
    }

    // Hands. COCO Wholebody indices 91..=111 are the subject's left
    // hand, and 112..=132 are the subject's right hand. Within each
    // group the local index runs 0..=20 (0 = wrist). Selfie mirror:
    // subject's left → avatar Right* fingers, and the wrist landmark
    // becomes the avatar's `RightHand` / `LeftHand` bone position.
    attach_hand(
        &mut sk,
        joints,
        91,
        HumanoidBone::RightHand,
        HAND_PHALANGES_RIGHT,
        HAND_TIPS_RIGHT,
        &to_source,
    );
    attach_hand(
        &mut sk,
        joints,
        112,
        HumanoidBone::LeftHand,
        HAND_PHALANGES_LEFT,
        HAND_TIPS_LEFT,
        &to_source,
    );

    // Overall confidence: average of the body 17 keypoint scores —
    // gives the GUI status bar a coarse quality signal.
    let mut sum = 0.0_f32;
    let mut n = 0_u32;
    for j in joints.iter().take(17) {
        sum += j.score;
        n += 1;
    }
    sk.overall_confidence = if n > 0 { sum / n as f32 } else { 0.0 };

    // Face pose / blendshapes are intentionally left at None / empty
    // for v1 — the head bone falls back to spine direction. Blendshape
    // expressions need the dedicated face model (Phase C: SMIRK).
    sk.face = None;
    sk.expressions = Vec::new();
    sk
}

fn attach_hand<F>(
    sk: &mut SourceSkeleton,
    joints: &[DecodedJoint],
    base_index: usize,
    wrist_bone: HumanoidBone,
    phalanges: &[(usize, HumanoidBone)],
    tips: &[(usize, HumanoidBone)],
    to_source: &F,
) where
    F: Fn(&DecodedJoint) -> [f32; 3],
{
    // Wrist position = centroid of the four finger MCP joints (index,
    // middle, ring, little — local indices 5, 9, 13, 17). The thumb
    // CMC sits anatomically away from the MCP line so it's excluded.
    //
    // We *do not* use either the body track's wrist keypoint
    // (index 9 / 10) or the hand-track's own wrist (index 91 / 112)
    // because both are unreliable when the wrist is occluded — e.g.
    // a guitar strumming hand whose wrist is hidden behind the
    // instrument lands ~0.3 units behind the torso in normalised z.
    // The MCPs anchor on the visible knuckles, so averaging four of
    // them produces a stable hand position that respects the actual
    // monocular geometry. The centroid sits ~half a hand-length
    // forward of the true wrist; that's biomechanically slightly
    // wrong but visually invisible and the avatar's solver re-anchors
    // the chain at the same point regardless.
    const MCP_LOCALS: [usize; 4] = [5, 9, 13, 17];
    let mut sum = [0.0_f32; 3];
    let mut min_conf = f32::INFINITY;
    let mut found = 0_u32;
    for &local in &MCP_LOCALS {
        let global = base_index + local;
        if global >= joints.len() {
            continue;
        }
        let j = &joints[global];
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            continue;
        }
        let p = to_source(j);
        sum[0] += p[0];
        sum[1] += p[1];
        sum[2] += p[2];
        min_conf = min_conf.min(j.score);
        found += 1;
    }
    if found < 3 {
        // Need at least 3 of 4 MCPs to call this hand tracked.
        // Fewer than that → drop the whole hand chain for the frame
        // so the solver leaves fingers at rest.
        return;
    }
    let inv = 1.0 / found as f32;
    sk.joints.insert(
        wrist_bone,
        SourceJoint {
            position: [sum[0] * inv, sum[1] * inv, sum[2] * inv],
            confidence: min_conf,
        },
    );

    for &(local_idx, bone) in phalanges {
        let global = base_index + local_idx;
        if global >= joints.len() {
            continue;
        }
        let j = &joints[global];
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            continue;
        }
        sk.joints.insert(
            bone,
            SourceJoint {
                position: to_source(j),
                confidence: j.score,
            },
        );
    }
    for &(local_idx, bone) in tips {
        let global = base_index + local_idx;
        if global >= joints.len() {
            continue;
        }
        let j = &joints[global];
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            continue;
        }
        sk.fingertips.insert(
            bone,
            SourceJoint {
                position: to_source(j),
                confidence: j.score,
            },
        );
    }

    // Palm orientation. Without this, the wrist bone's twist around
    // its length axis is unconstrained — the LowerArm direction-match
    // sets a wrist position but no rotation, so finger MCPs end up
    // shortest-arc'd from a random starting twist and the avatar's
    // fingers appear rotated 90° at the base. Solve by deriving a
    // full forward+up basis from the four MCP knuckles plus the
    // hand-track wrist:
    //
    //   raw_forward = middle_mcp - wrist           (palm-direction)
    //   across      = index_mcp  - pinky_mcp       (palm-width)
    //   normal      = across × raw_forward         (palm plane normal)
    //   forward     = normal × across              (re-orthogonalised)
    //
    // Cross-orthogonalising the forward axis against the normal
    // means a noisy hand-track wrist Z (the upstream comment warns
    // it can land 0.3 units behind the torso under occlusion) only
    // perturbs the normal slightly without tilting the in-palm
    // forward direction.
    let l_wrist = base_index;
    let l_middle = base_index + 9;
    let l_index = base_index + 5;
    let l_pinky = base_index + 17;
    if l_wrist < joints.len()
        && l_middle < joints.len()
        && l_index < joints.len()
        && l_pinky < joints.len()
        && joints[l_wrist].score >= KEYPOINT_VISIBILITY_FLOOR
        && joints[l_middle].score >= KEYPOINT_VISIBILITY_FLOOR
        && joints[l_index].score >= KEYPOINT_VISIBILITY_FLOOR
        && joints[l_pinky].score >= KEYPOINT_VISIBILITY_FLOOR
    {
        let wrist_pos = to_source(&joints[l_wrist]);
        let middle_pos = to_source(&joints[l_middle]);
        let index_pos = to_source(&joints[l_index]);
        let pinky_pos = to_source(&joints[l_pinky]);
        let raw_forward = sub3(middle_pos, wrist_pos);
        let across = sub3(index_pos, pinky_pos);
        let normal_raw = [
            across[1] * raw_forward[2] - across[2] * raw_forward[1],
            across[2] * raw_forward[0] - across[0] * raw_forward[2],
            across[0] * raw_forward[1] - across[1] * raw_forward[0],
        ];
        let normal_len = length3(normal_raw);
        if normal_len > 1e-6 {
            let normal = [
                normal_raw[0] / normal_len,
                normal_raw[1] / normal_len,
                normal_raw[2] / normal_len,
            ];
            let forward_raw = [
                normal[1] * across[2] - normal[2] * across[1],
                normal[2] * across[0] - normal[0] * across[2],
                normal[0] * across[1] - normal[1] * across[0],
            ];
            let forward_len = length3(forward_raw);
            if forward_len > 1e-6 {
                let forward = [
                    forward_raw[0] / forward_len,
                    forward_raw[1] / forward_len,
                    forward_raw[2] / forward_len,
                ];
                let orientation = HandOrientation {
                    forward,
                    up: normal,
                    confidence: min_conf,
                };
                match wrist_bone {
                    HumanoidBone::LeftHand => sk.left_hand_orientation = Some(orientation),
                    HumanoidBone::RightHand => sk.right_hand_orientation = Some(orientation),
                    _ => {}
                }
            }
        }
    }
}
