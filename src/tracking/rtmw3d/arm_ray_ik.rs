//! Arm-depth solve via ray-IK — the live replacement for the
//! bone-length z-reconstruction heuristic (`docs/ray-ik-depth-solve.md`).
//!
//! The old `reconstruct_arm_z` injected `sqrt(L² − xy²)` of forward
//! depth using a session running-max length reference, which
//! perspective magnification (a hand swept near the lens) inflated
//! permanently — after that every in-plane arm read as foreshortened
//! and the avatar warped while the 2D annotation stayed perfect. Here
//! the shoulder→elbow→wrist chain is instead re-solved as depths
//! along the observation rays with *anatomical metric* lengths:
//! near-lens magnification resolves into a smaller depth by
//! construction and can never corrupt the length reference, and the
//! solved joints reproject exactly onto the observed 2D keypoints.
//!
//! The shoulders are pinned — freeing them was tried and reverted:
//! the arm segments' anatomical reference lengths clash with
//! non-human rig proportions and DRAG free shoulders off the torso,
//! and a span-foreshortening Δz demand fabricates twist out of
//! span-estimation jitter (sqrt amplification — the legacy ±45°
//! yaw-lock physics). Shoulder twist Δz is a separate estimation
//! problem owned by the DAv2 bz injection; fusing it fully into this
//! solve is migration step 3 in the design doc. Elbow and wrist
//! depths are free; RTMW3D's nz enters as the weak consistency term,
//! DAv2 samples (when the skeleton carries them — the with-depth
//! pipeline's post-merge re-solve) as per-joint depth evidence, the
//! desk-streaming forward prior breaks the remaining toward/behind
//! tie, and the temporal term carries occlusion gaps.

use std::collections::HashMap;

use crate::asset::HumanoidBone;
use log::{debug, info};

use super::super::ray_ik::{self, RayIkConfig, RayIkState, RayObservation, SegmentSpec};
use super::super::source_skeleton::{SourceJoint, SourceSkeleton};
use super::arm_z;

/// Assumed biacromial span — same constant the depth pipeline uses.
/// The absolute value cancels out of the source-space output (a wrong
/// span rescales the metric problem uniformly); it only tunes how
/// strong the perspective correction is via the torso-depth estimate.
const ASSUMED_SHOULDER_SPAN_M: f32 = 0.40;
/// Anthropometric segment priors as fractions of the biacromial span
/// (same values `arm_z` documents).
const UPPER_ARM_SPAN_RATIO: f32 = 0.76;
const FOREARM_SPAN_RATIO: f32 = 0.62;
/// Assumed webcam optics — shared with the depth pipeline.
const ASSUMED_VERTICAL_FOV_DEG: f32 = 60.0;
/// Minimum keypoint confidence for a side to participate.
const MIN_CONF: f32 = 0.3;
/// EMA factor for the torso-depth estimate; smooths per-frame
/// jitter out of the perspective scale.
const Z0_EMA_ALPHA: f32 = 0.2;
/// Assumed shoulder-mid → hip-mid trunk length. The torso-depth
/// estimate anchors on this (not the shoulder span) because the
/// trunk's projected length is invariant under SHOULDER TWIST — a
/// span-based Z₀ silently re-absorbs exactly the foreshortening any
/// twist solve needs to see.
const ASSUMED_TORSO_LEN_M: f32 = 0.45;

/// Max frames a dropped wrist is reconstructed from its last observed
/// ray before the hand is released. Matches the legacy wrist temporal
/// hold (`wrist.rs` Stage 3) this folds in.
const MAX_WRIST_HOLD_FRAMES: u32 = 6;

/// Confidence assigned to a wrist reconstructed from a held ray. Low
/// enough that the length invariant + temporal term own its placement
/// (the held 2D direction is trusted, but the detector did not vouch
/// for it this frame), high enough to clear the solver's default join
/// threshold so the hand still participates.
const HELD_WRIST_CONF: f32 = 0.3;

/// Cross-frame state: the IK solver's temporal depths plus the
/// torso-depth EMA. Reset with the rest of the temporal state.
#[derive(Default)]
pub(in crate::tracking) struct ArmRayIk {
    ik: RayIkState,
    z0_ema: Option<f32>,
    /// Session max of the projected shoulder span in metres —
    /// captured when the subject faces front (twist only SHRINKS the
    /// projection, and unlike a hand the torso cannot sweep near the
    /// lens, so the max is a legitimate frontal-span estimate here).
    /// Read with an anatomical ceiling only.
    span_m_max: f32,
    /// Last observed wrist ray per side, held across frames where the
    /// detector drops the wrist. The 2D ray is a pure function of the
    /// image point, so holding it == holding the last good 2D wrist;
    /// the depth then re-solves every frame against current evidence
    /// instead of freezing a stale 3D position. This is the ray-IK
    /// replacement for `wrist.rs` Stage 3 (temporal hold) and Stage 4
    /// (forward-extension synthesis) — the depth solve's forward prior
    /// + length invariant place the held wrist instead of a hardcoded
    /// "+0.2 toward camera" nudge. `u32` is the frames-since-observed
    /// age, capped by `MAX_WRIST_HOLD_FRAMES`.
    held_wrist: HashMap<HumanoidBone, ([f32; 3], u32)>,
    frame: u64,
}

impl ArmRayIk {
    pub(in crate::tracking) fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Re-solve elbow + wrist depths for both arms and write the
/// perspective-true positions back into the source skeleton. The
/// finger chains translate with their wrist (same policy as the wrist
/// temporal hold) so hand orientation and finger solving stay
/// coherent.
pub(in crate::tracking) fn solve_arm_depth(
    sk: &mut SourceSkeleton,
    aspect: f32,
    stage: &mut ArmRayIk,
) {
    solve_arm_depth_impl(sk, aspect, stage, false)
}

/// Post-merge variant for the with-depth pipeline: the Phase-7.6
/// metric chain replacement already places the arms from REAL
/// measured vectors, so a wholesale prior-based re-solve only
/// degrades them (photo benchmark: mean 2.4°→3.6°). The one thing
/// 7.6 cannot do is keep the two wrists depth-coherent when the
/// hands touch — so this variant solves and writes back ONLY when
/// the hands-contact link engages, and is a no-op otherwise.
pub(in crate::tracking) fn solve_arm_depth_contact_only(
    sk: &mut SourceSkeleton,
    aspect: f32,
    stage: &mut ArmRayIk,
) {
    solve_arm_depth_impl(sk, aspect, stage, true)
}

fn solve_arm_depth_impl(
    sk: &mut SourceSkeleton,
    aspect: f32,
    stage: &mut ArmRayIk,
    contact_only: bool,
) {
    stage.frame += 1;
    let Some(root) = sk.root_offset else {
        return;
    };
    // Invert the root offset back to the anchor's normalized image
    // coords (exact inverse of `build_source_skeleton`'s mapping).
    let origin_nx = 0.5 - root[0] / (2.0 * aspect);
    let origin_ny = 0.5 - root[1] / 2.0;

    let (Some(ls), Some(rs)) = (
        sk.joints.get(&HumanoidBone::LeftUpperArm).copied(),
        sk.joints.get(&HumanoidBone::RightUpperArm).copied(),
    ) else {
        return;
    };
    if ls.confidence < MIN_CONF || rs.confidence < MIN_CONF {
        return;
    }
    let span_src = {
        let dx = ls.position[0] - rs.position[0];
        let dy = ls.position[1] - rs.position[1];
        (dx * dx + dy * dy).sqrt()
    };
    if span_src < 1e-3 {
        return;
    }

    // Torso depth from the perspective relation: 1 source unit equals
    // `tan(vfov/2) · Z₀` metres. Anchor on the TRUNK length when the
    // hip anchor is available — the trunk's projection is invariant
    // under shoulder twist, while a span-anchored Z₀ re-absorbs the
    // very foreshortening a twist solve must observe. Shoulder-span
    // fallback (upper-body framing without hips) keeps perspective
    // correction working.
    let tan_v = (ASSUMED_VERTICAL_FOV_DEG.to_radians() * 0.5).tan();
    let torso_src = if sk.root_anchor_is_hip {
        sk.joints.get(&HumanoidBone::Hips).and_then(|h| {
            let mx = (ls.position[0] + rs.position[0]) * 0.5 - h.position[0];
            let my = (ls.position[1] + rs.position[1]) * 0.5 - h.position[1];
            let len = (mx * mx + my * my).sqrt();
            (len > 1e-3).then_some(len)
        })
    } else {
        None
    };
    let z0_raw = match torso_src {
        Some(t) => (ASSUMED_TORSO_LEN_M / (t * tan_v)).clamp(0.25, 4.0),
        None => (ASSUMED_SHOULDER_SPAN_M / (span_src * tan_v)).clamp(0.25, 4.0),
    };
    let z0 = match stage.z0_ema {
        Some(prev) => prev + (z0_raw - prev) * Z0_EMA_ALPHA,
        None => z0_raw,
    };
    stage.z0_ema = Some(z0);
    let m_per_src = tan_v * z0;

    // Session frontal-span estimate. NO absolute lower bound: any
    // floor above the subject's true frontal span (in OUR z0-scaled
    // units — proportion priors don't hold for stylized rigs, and z0
    // unit error scales the measurement) fabricates geometry. The
    // anatomical ceiling stays as the anti-runaway bound (sleeve-edge
    // detections inflate the span, and unlike the torso they wander).
    let span_m_proj = span_src * m_per_src;
    if span_m_proj > stage.span_m_max {
        stage.span_m_max = span_m_proj;
    }
    let span_m = stage
        .span_m_max
        .min(ASSUMED_SHOULDER_SPAN_M * 1.35)
        .max(0.05);

    // source xy → normalized image coords (inverse of `to_source`).
    let to_n = |p: [f32; 3]| -> (f32, f32) {
        (
            origin_nx - p[0] / (2.0 * aspect),
            origin_ny - p[1] / 2.0,
        )
    };
    // Unnormalized camera ray (x = image right, y = image down,
    // z = away from camera) through a normalized image point.
    let ray_u = |nx: f32, ny: f32| -> [f32; 3] {
        [
            (nx - 0.5) * 2.0 * tan_v * aspect,
            (ny - 0.5) * 2.0 * tan_v,
            1.0,
        ]
    };
    // Unit observation ray through a source-space joint, plus the
    // unnormalized ray length (= the per-unit-depth range scale a
    // camera-z metric converts through).
    let ray_of = |pos: [f32; 3]| -> ([f32; 3], f32) {
        let (nx, ny) = to_n(pos);
        let u = ray_u(nx, ny);
        let ulen = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        ([u[0] / ulen, u[1] / ulen, u[2] / ulen], ulen)
    };
    // Build one observation from a source-space joint. `z_cam`
    // attaches absolute depth (camera-z metres, converted to range
    // along the ray); `fixed` additionally pins the joint there
    // (shoulders).
    let make_obs = |bone: HumanoidBone,
                    pos: [f32; 3],
                    conf: f32,
                    z_cam: Option<f32>,
                    depth_weight: f32,
                    fixed: bool|
     -> RayObservation {
        let (ray, ulen) = ray_of(pos);
        // Range along the ray = camera z · |u|.
        let metric = z_cam.map(|z| z.max(0.15) * ulen);
        RayObservation {
            bone,
            ray,
            confidence: conf,
            metric_depth_m: metric,
            depth_weight,
            fixed: fixed && metric.is_some(),
        }
    };
    // Build a wrist observation from a held ray (the detector dropped
    // the wrist this frame): no metric depth, low confidence so the
    // length invariant + temporal term own the placement.
    let make_held_obs = |bone: HumanoidBone, ray: [f32; 3]| -> RayObservation {
        RayObservation {
            bone,
            ray,
            confidence: HELD_WRIST_CONF,
            metric_depth_m: None,
            depth_weight: 0.0,
            fixed: false,
        }
    };
    // Pin depth at the joint's current source z (shoulders). Source
    // +z is toward the camera, so camera z shrinks as source z grows.
    let pin_z = |pos: [f32; 3]| -> Option<f32> { Some(z0 - pos[2] * m_per_src) };
    // DAv2 evidence for a free joint, RELATIVE to the shoulders' own
    // DAv2 depth: the depth pipeline's metric scale (span-calibrated)
    // and this solve's Z₀ scale (torso prior) are different absolute
    // frames, but their per-joint DELTAS agree — same rationale as
    // the Phase-7.5 relative injections. Gated against background /
    // occlusion samples by a plausibility bound.
    let dav2_ref = match (ls.metric_depth_m, rs.metric_depth_m) {
        (Some(a), Some(b)) => Some((a + b) * 0.5),
        _ => None,
    };
    let dav2_ev = |m: Option<f32>| -> Option<f32> {
        let r = dav2_ref?;
        let m = m?;
        ((m - r).abs() < 0.8).then(|| z0 + (m - r))
    };

    let mut obs: Vec<RayObservation> = Vec::with_capacity(6);
    let mut segs: Vec<SegmentSpec> = Vec::with_capacity(5);
    let l_ua = UPPER_ARM_SPAN_RATIO * span_m;
    let l_fa = FOREARM_SPAN_RATIO * span_m;
    // nz consistency evidence: source Δz (+toward camera) → camera Δz
    // (+away), in metres.
    let nz_dz = |a: [f32; 3], b: [f32; 3]| -> f32 { -(b[2] - a[2]) * m_per_src };

    // Shoulders: FIXED anchors at their incoming source position
    // (synthetic body-prior z included) — see the module doc for why
    // freeing them was reverted.
    obs.push(make_obs(
        HumanoidBone::LeftUpperArm,
        ls.position,
        ls.confidence,
        pin_z(ls.position),
        1.0,
        true,
    ));
    obs.push(make_obs(
        HumanoidBone::RightUpperArm,
        rs.position,
        rs.confidence,
        pin_z(rs.position),
        1.0,
        true,
    ));

    // Wrist source positions per side, recorded for the hands-contact
    // coherence pass below.
    let mut wrist_pos: [Option<[f32; 3]>; 2] = [None, None];
    // Wrists reconstructed from a held ray this frame (absent from the
    // incoming skeleton); inserted back after the solve.
    let mut synth_wrist: Vec<HumanoidBone> = Vec::new();
    for (side, (shoulder_bone, shoulder, elbow_bone, wrist_bone)) in [
        (
            HumanoidBone::LeftUpperArm,
            ls,
            HumanoidBone::LeftLowerArm,
            HumanoidBone::LeftHand,
        ),
        (
            HumanoidBone::RightUpperArm,
            rs,
            HumanoidBone::RightLowerArm,
            HumanoidBone::RightHand,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let Some(el) = sk.joints.get(&elbow_bone).copied() else {
            continue;
        };
        if el.confidence < MIN_CONF {
            continue;
        }
        obs.push(make_obs(
            elbow_bone,
            el.position,
            el.confidence,
            dav2_ev(el.metric_depth_m),
            1.0,
            false,
        ));
        segs.push(SegmentSpec {
            a: shoulder_bone,
            b: elbow_bone,
            length_m: l_ua,
            nz_dz_m: Some(nz_dz(shoulder.position, el.position)),
            nz_weight: 1.0,
            one_sided: false,
            forward_prior: true,
        });
        // Wrist: trust the live keypoint when the detector produced one
        // (any confidence — its weight scales the residuals) and record
        // its ray for the temporal hold. When the wrist was dropped
        // upstream (the plausibility filter) or never detected,
        // reconstruct it from the last held ray for up to
        // MAX_WRIST_HOLD_FRAMES — the ray-IK fold-in of the legacy
        // wrist Stage 3 (temporal hold) and Stage 4 (forward-extension
        // synthesis). The forward prior + forearm-length invariant
        // place the held wrist; the depth re-solves against current
        // evidence instead of freezing a stale 3D position or nudging a
        // hardcoded "+0.2 toward camera".
        //
        // The wrist's own DAv2 sample is deliberately NOT consumed
        // either way: its MCP-centroid pixel lands on the back of the
        // hand and routinely reads background depth (the documented
        // reason Phase 7.5 never re-injects wrist z).
        if let Some(wr) = sk.joints.get(&wrist_bone).copied() {
            obs.push(make_obs(wrist_bone, wr.position, wr.confidence, None, 0.0, false));
            stage.held_wrist.insert(wrist_bone, (ray_of(wr.position).0, 0));
            wrist_pos[side] = Some(wr.position);
            segs.push(SegmentSpec {
                a: elbow_bone,
                b: wrist_bone,
                length_m: l_fa,
                nz_dz_m: Some(nz_dz(el.position, wr.position)),
                nz_weight: 1.0,
                one_sided: false,
                forward_prior: true,
            });
        } else if let Some((held_ray, age)) = stage.held_wrist.get(&wrist_bone).copied() {
            if age < MAX_WRIST_HOLD_FRAMES {
                obs.push(make_held_obs(wrist_bone, held_ray));
                stage.held_wrist.insert(wrist_bone, (held_ray, age + 1));
                synth_wrist.push(wrist_bone);
                // No nz evidence for a held wrist (its source z is
                // stale): the forward prior + length invariant own the
                // depth, the temporal term carries it from last frame.
                segs.push(SegmentSpec {
                    a: elbow_bone,
                    b: wrist_bone,
                    length_m: l_fa,
                    nz_dz_m: None,
                    nz_weight: 0.0,
                    one_sided: false,
                    forward_prior: true,
                });
                debug!(
                    "ray-ik f{} {:?}: wrist held from last ray (age {})",
                    stage.frame, wrist_bone, age + 1
                );
            } else {
                stage.held_wrist.remove(&wrist_bone);
            }
        }
    }
    if segs.is_empty() {
        return;
    }

    // Hands-contact depth coherence: each arm otherwise solves its
    // depth independently, so palms-together can come back with the
    // two wrists at different depths — a gap the 2D annotation cannot
    // show. When the wrists are close in 2D AND the two hand keypoint
    // blocks project at similar size (similar apparent size ⇒ similar
    // depth; dissimilar sizes mean one hand is merely passing in front
    // of the other and must NOT be glued), link the pair at the
    // lateral distance the 2D separation implies at equal depth.
    let mut contact_engaged = false;
    if let (Some(wl), Some(wr)) = (wrist_pos[0], wrist_pos[1]) {
        let spread = |left: bool| -> f32 {
            let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            let mut n = 0u32;
            for (bone, j) in sk.joints.iter() {
                if arm_z::is_finger_bone_side(*bone, left) {
                    x0 = x0.min(j.position[0]);
                    y0 = y0.min(j.position[1]);
                    x1 = x1.max(j.position[0]);
                    y1 = y1.max(j.position[1]);
                    n += 1;
                }
            }
            if n >= 4 {
                (x1 - x0).max(y1 - y0)
            } else {
                0.0
            }
        };
        let sp_l = spread(true);
        let sp_r = spread(false);
        let d2d = {
            let dx = wl[0] - wr[0];
            let dy = wl[1] - wr[1];
            (dx * dx + dy * dy).sqrt()
        };
        let sp_max = sp_l.max(sp_r);
        let sp_min = sp_l.min(sp_r);
        // Spread floor 0.04 source units ≈ the detector's 0.02-frame
        // "hand really in view" floor.
        if sp_min > 0.04 && sp_max < sp_min * 1.4 && d2d < 1.2 * sp_max {
            segs.push(SegmentSpec {
                a: HumanoidBone::LeftHand,
                b: HumanoidBone::RightHand,
                // Floor keeps the length residual differentiable when
                // the wrists project coincident.
                length_m: (d2d * m_per_src).max(0.02),
                nz_dz_m: Some(nz_dz(wl, wr)),
                nz_weight: 1.0,
                one_sided: false,
                forward_prior: false,
            });
            debug!(
                "ray-ik f{}: hands-contact link engaged (d2d={:.3} src, spreads {:.3}/{:.3})",
                stage.frame, d2d, sp_l, sp_r
            );
            contact_engaged = true;
        }
    }
    if contact_only && !contact_engaged {
        return;
    }

    let solved = ray_ik::solve(&obs, &segs, &RayIkConfig::default(), &mut stage.ik);

    // Camera-space anchor (the hip/shoulder mid at torso depth) for
    // the back-conversion into source space.
    let ou = ray_u(origin_nx, origin_ny);
    let origin_cam = [ou[0] * z0, ou[1] * z0, z0];
    let to_source = |p: [f32; 3]| -> [f32; 3] {
        [
            -(p[0] - origin_cam[0]) / m_per_src,
            -(p[1] - origin_cam[1]) / m_per_src,
            -(p[2] - origin_cam[2]) / m_per_src,
        ]
    };

    for bone in [
        HumanoidBone::LeftLowerArm,
        HumanoidBone::RightLowerArm,
        HumanoidBone::LeftHand,
        HumanoidBone::RightHand,
    ] {
        let Some(p_cam) = solved.get(&bone) else {
            continue;
        };
        let new_pos = to_source(*p_cam);
        match sk.joints.get_mut(&bone) {
            Some(j) => {
                let delta = [
                    new_pos[0] - j.position[0],
                    new_pos[1] - j.position[1],
                    new_pos[2] - j.position[2],
                ];
                j.position = new_pos;
                debug!(
                    "ray-ik f{} {:?}: dz_src={:+.3} (z {:+.3}, cam z {:.3} m)",
                    stage.frame, bone, delta[2], new_pos[2], p_cam[2]
                );
                if matches!(bone, HumanoidBone::LeftHand | HumanoidBone::RightHand) {
                    arm_z::shift_hand_chain(sk, bone, delta);
                }
            }
            None => {
                // Wrist reconstructed from a held ray: the joint was
                // absent from the incoming skeleton, so insert it. No
                // finger chain to shift — a dropped wrist drops its
                // fingers too (the plausibility filter clears them).
                if synth_wrist.contains(&bone) {
                    sk.joints.insert(
                        bone,
                        SourceJoint {
                            position: new_pos,
                            confidence: HELD_WRIST_CONF,
                            metric_depth_m: None,
                        },
                    );
                    debug!(
                        "ray-ik f{} {:?}: held wrist inserted (z {:+.3}, cam z {:.3} m)",
                        stage.frame, bone, new_pos[2], p_cam[2]
                    );
                }
            }
        }
    }

    if stage.frame % 60 == 0 {
        let z = |b: HumanoidBone| {
            sk.joints
                .get(&b)
                .map(|j| j.position[2])
                .unwrap_or(f32::NAN)
        };
        info!(
            "ray-ik diag f{}: z0={:.2}m span_src={:.3} elbow_z=[{:+.3},{:+.3}] wrist_z=[{:+.3},{:+.3}]",
            stage.frame,
            z0,
            span_src,
            z(HumanoidBone::LeftLowerArm),
            z(HumanoidBone::RightLowerArm),
            z(HumanoidBone::LeftHand),
            z(HumanoidBone::RightHand),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASPECT: f32 = 1.0;

    /// Unit camera ray through a source-space joint, replicating the
    /// solver's internal `ray_of` so the test can assert the
    /// reconstructed wrist reprojects onto the held 2D direction.
    fn ray_of_pos(root: [f32; 3], pos: [f32; 3]) -> [f32; 3] {
        let origin_nx = 0.5 - root[0] / (2.0 * ASPECT);
        let origin_ny = 0.5 - root[1] / 2.0;
        let nx = origin_nx - pos[0] / (2.0 * ASPECT);
        let ny = origin_ny - pos[1] / 2.0;
        let tan_v = (ASSUMED_VERTICAL_FOV_DEG.to_radians() * 0.5).tan();
        let u = [
            (nx - 0.5) * 2.0 * tan_v * ASPECT,
            (ny - 0.5) * 2.0 * tan_v,
            1.0,
        ];
        let l = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        [u[0] / l, u[1] / l, u[2] / l]
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn joint(pos: [f32; 3]) -> SourceJoint {
        SourceJoint {
            position: pos,
            confidence: 0.9,
            metric_depth_m: None,
        }
    }

    /// A plausible both-arms skeleton anchored on the shoulder pair at
    /// the image centre (root_offset = 0 → origin at 0.5, 0.5).
    fn make_skeleton() -> SourceSkeleton {
        let mut sk = SourceSkeleton::empty(0);
        sk.root_offset = Some([0.0, 0.0, 0.0]);
        sk.root_anchor_is_hip = false;
        sk.joints.insert(HumanoidBone::LeftUpperArm, joint([0.2, 0.0, 0.0]));
        sk.joints.insert(HumanoidBone::RightUpperArm, joint([-0.2, 0.0, 0.0]));
        sk.joints.insert(HumanoidBone::LeftLowerArm, joint([0.4, -0.2, 0.0]));
        sk.joints.insert(HumanoidBone::RightLowerArm, joint([-0.4, -0.2, 0.0]));
        sk.joints.insert(HumanoidBone::LeftHand, joint([0.5, -0.4, 0.0]));
        sk.joints.insert(HumanoidBone::RightHand, joint([-0.5, -0.4, 0.0]));
        sk
    }

    /// A dropped wrist is reconstructed from its last observed ray and
    /// reprojects onto the same 2D direction (Stage 3 fold-in).
    #[test]
    fn dropped_wrist_reconstructed_from_held_ray() {
        let mut stage = ArmRayIk::default();
        let mut sk = make_skeleton();
        // Frame 1: wrist present → records the held ray.
        solve_arm_depth(&mut sk, ASPECT, &mut stage);
        let held_ray = ray_of_pos([0.0, 0.0, 0.0], [0.5, -0.4, 0.0]);
        assert!(stage.held_wrist.contains_key(&HumanoidBone::LeftHand));

        // Frame 2: wrist dropped (plausibility filter / occlusion).
        sk.joints.remove(&HumanoidBone::LeftHand);
        solve_arm_depth(&mut sk, ASPECT, &mut stage);

        let rec = sk
            .joints
            .get(&HumanoidBone::LeftHand)
            .expect("wrist reconstructed from held ray");
        assert!(
            (rec.confidence - HELD_WRIST_CONF).abs() < 1e-5,
            "held wrist carries the held confidence"
        );
        let rec_ray = ray_of_pos([0.0, 0.0, 0.0], rec.position);
        assert!(
            dot(rec_ray, held_ray) > 0.999,
            "reconstructed wrist reprojects onto the held 2D ray (dot {})",
            dot(rec_ray, held_ray)
        );
    }

    /// After the hold age cap is exceeded the hand is released rather
    /// than frozen forever (the legacy Stage 3 bound, preserved).
    #[test]
    fn held_wrist_released_after_max_age() {
        let mut stage = ArmRayIk::default();
        let mut sk = make_skeleton();
        solve_arm_depth(&mut sk, ASPECT, &mut stage);
        sk.joints.remove(&HumanoidBone::LeftHand);

        // Hold persists up to MAX_WRIST_HOLD_FRAMES reconstructions.
        for _ in 0..MAX_WRIST_HOLD_FRAMES {
            sk.joints.remove(&HumanoidBone::LeftHand);
            solve_arm_depth(&mut sk, ASPECT, &mut stage);
            assert!(sk.joints.contains_key(&HumanoidBone::LeftHand));
        }
        // The next frame past the cap releases the hand.
        sk.joints.remove(&HumanoidBone::LeftHand);
        solve_arm_depth(&mut sk, ASPECT, &mut stage);
        assert!(
            !sk.joints.contains_key(&HumanoidBone::LeftHand),
            "wrist released once the hold age cap is exceeded"
        );
    }

    /// A re-observed wrist refreshes the hold (age resets), so a brief
    /// occlusion does not consume the budget for a later one.
    #[test]
    fn live_wrist_refreshes_hold() {
        let mut stage = ArmRayIk::default();
        let mut sk = make_skeleton();
        solve_arm_depth(&mut sk, ASPECT, &mut stage);
        // Drop for a few frames (under the cap).
        for _ in 0..3 {
            sk.joints.remove(&HumanoidBone::LeftHand);
            solve_arm_depth(&mut sk, ASPECT, &mut stage);
        }
        // Re-observe: a fresh wrist resets the age to 0.
        sk.joints.insert(HumanoidBone::LeftHand, joint([0.5, -0.4, 0.0]));
        solve_arm_depth(&mut sk, ASPECT, &mut stage);
        let (_, age) = stage.held_wrist[&HumanoidBone::LeftHand];
        assert_eq!(age, 0, "a live wrist resets the hold age");
    }
}
