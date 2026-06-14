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

use crate::asset::HumanoidBone;
use log::{debug, info};

use super::super::ray_ik::{self, RayIkConfig, RayIkState, RayObservation, SegmentSpec};
use super::super::source_skeleton::SourceSkeleton;
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
        let (nx, ny) = to_n(pos);
        let u = ray_u(nx, ny);
        let ulen = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        let ray = [u[0] / ulen, u[1] / ulen, u[2] / ulen];
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
        if let Some(wr) = sk.joints.get(&wrist_bone).copied() {
            // The wrist's DAv2 sample is deliberately NOT consumed:
            // its MCP-centroid pixel lands on the back of the hand
            // and routinely reads background depth (the documented
            // reason Phase 7.5 never re-injects wrist z either).
            obs.push(make_obs(wrist_bone, wr.position, wr.confidence, None, 0.0, false));
            segs.push(SegmentSpec {
                a: elbow_bone,
                b: wrist_bone,
                length_m: l_fa,
                nz_dz_m: Some(nz_dz(el.position, wr.position)),
                nz_weight: 1.0,
                one_sided: false,
                forward_prior: true,
            });
            wrist_pos[side] = Some(wr.position);
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
        let Some(j) = sk.joints.get_mut(&bone) else {
            continue;
        };
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
