//! Arm keypoint repair on the projected (xy) plane: edge-exit
//! detection + restoration of clamp-shortened limbs, plus the
//! session length bookkeeping the repair's band clamp reads.
//!
//! The z-depth reconstruction that used to live here (`|dz| =
//! sqrt(L² − xy²)` against a session running-max length) moved to the
//! ray-IK stage (`super::arm_ray_ik`, `docs/ray-ik-depth-solve.md`):
//! the running max was structurally corruptible — one near-lens hand
//! sweep inflated it past anatomy (measured 2×) and a pure max never
//! recovers, after which every in-plane arm read as foreshortened and
//! gained fake forward depth every frame. The ray-IK solve uses
//! anatomical metric lengths instead, so near-lens magnification
//! resolves into depth by construction.
//!
//! What remains here:
//!
//! * [`observe_lengths`] — running maxima of projected segment
//!   lengths, used (band-clamped!) by the edge repair.
//! * [`wrist_out_of_frame`] / [`repair_edge_clamped_arms`] — keypoints
//!   clamped at the frame border are observation clamps, not
//!   positions; restore the limb along the last visible direction.
//! * [`shift_hand_chain`] — translate a wrist's finger chain
//!   coherently (shared with the ray-IK stage).

use crate::asset::HumanoidBone;
use log::debug;

use super::super::source_skeleton::SourceSkeleton;

/// Anthropometric segment-length priors as fractions of the
/// biacromial (shoulder) span. Upper arm 0.186·H vs biacromial
/// 0.245·H → 0.76; forearm (elbow→wrist) 0.146·H → 0.60, nudged to
/// 0.62 because the COCO wrist keypoint sits slightly past the
/// anatomical wrist crease.
const UPPER_ARM_SPAN_RATIO: f32 = 0.76;
const FOREARM_SPAN_RATIO: f32 = 0.62;

/// Minimum keypoint confidence for a segment to participate.
const MIN_CONF: f32 = 0.3;

/// Per-session running maxima of projected segment lengths.
/// `[left, right]` per segment; reset with the rest of the temporal
/// state on session / benchmark-image boundaries.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ArmLengthState {
    upper_xy_max: [f32; 2],
    forearm_xy_max: [f32; 2],
    span_max: f32,
    /// Last-good unit directions (source space) per side, captured
    /// while the hand was genuinely visible. An arm SLIDES out of
    /// frame, so the direction held from the last visible frame is a
    /// far better continuation than anything derivable from the
    /// clamped stub (cold-start geometric extension stays as the
    /// no-history fallback).
    last_fore_dir: [Option<[f32; 3]>; 2],
    last_upper_dir: [Option<[f32; 3]>; 2],
    /// Detector hysteresis: consecutive fired / clear frame counts.
    oof_streak: [i32; 2],
    /// Extension engagement 0..1, ramped over a few frames so the
    /// wrist cross-fades between the observed stub and the extended
    /// position instead of teleporting on detector flips (real-video
    /// replay measured 44–60 per-frame jumps >0.6 source units at
    /// exit/re-entry transitions before this).
    engagement: [f32; 2],
    /// Diagnostics: frame counter plus last logged maxima — growth
    /// logs gate behind a 5% threshold so warmup creep stays quiet.
    /// Per-event lines need
    /// `RUST_LOG=vulvatar_lib::tracking::rtmw3d::arm_z=debug`.
    frame: u64,
    logged_span: f32,
    logged_upper_max: [f32; 2],
    logged_forearm_max: [f32; 2],
}

/// Fold the current frame's observable segment lengths into the
/// running maxima (idempotent: pure max()). Called by the edge-exit
/// repair at the top of the frame; the repair's band clamp is the
/// only consumer of the maxima.
pub(super) fn observe_lengths(sk: &SourceSkeleton, state: &mut ArmLengthState) {
    if let (Some(l), Some(r)) = (
        sk.joints.get(&HumanoidBone::LeftUpperArm),
        sk.joints.get(&HumanoidBone::RightUpperArm),
    ) {
        if l.confidence >= MIN_CONF && r.confidence >= MIN_CONF {
            let span = xy_len(l.position, r.position);
            if span > state.span_max {
                state.span_max = span;
                if span > state.logged_span * 1.05 {
                    debug!(
                        "arm-z diag f{}: span_max grew {:.3} -> {:.3}",
                        state.frame, state.logged_span, span
                    );
                    state.logged_span = span;
                }
            }
        }
    }

    let span_ref = state.span_max.max(1e-3);
    for (side, shoulder, elbow, wrist) in [
        (
            0usize,
            HumanoidBone::LeftUpperArm,
            HumanoidBone::LeftLowerArm,
            HumanoidBone::LeftHand,
        ),
        (
            1usize,
            HumanoidBone::RightUpperArm,
            HumanoidBone::RightLowerArm,
            HumanoidBone::RightHand,
        ),
    ] {
        let Some(sh) = sk.joints.get(&shoulder) else {
            continue;
        };
        let Some(el) = sk.joints.get(&elbow) else {
            continue;
        };
        if sh.confidence < MIN_CONF || el.confidence < MIN_CONF {
            continue;
        }
        let ua_xy = xy_len(sh.position, el.position);
        if ua_xy > state.upper_xy_max[side] {
            state.upper_xy_max[side] = ua_xy;
            if ua_xy > state.logged_upper_max[side] * 1.05 {
                let prior = UPPER_ARM_SPAN_RATIO * span_ref;
                debug!(
                    "arm-z diag f{} {}: upper_xy_max grew to {:.3} ({:.2}x prior {:.3})",
                    state.frame,
                    side_tag(side),
                    ua_xy,
                    ua_xy / prior.max(1e-6),
                    prior
                );
                state.logged_upper_max[side] = ua_xy;
            }
        }
        let Some(wr) = sk.joints.get(&wrist) else {
            continue;
        };
        if wr.confidence < MIN_CONF {
            continue;
        }
        let fa_xy = xy_len(el.position, wr.position);
        if fa_xy > state.forearm_xy_max[side] {
            state.forearm_xy_max[side] = fa_xy;
            if fa_xy > state.logged_forearm_max[side] * 1.05 {
                let prior = FOREARM_SPAN_RATIO * span_ref;
                debug!(
                    "arm-z diag f{} {}: forearm_xy_max grew to {:.3} ({:.2}x prior {:.3})",
                    state.frame,
                    side_tag(side),
                    fa_xy,
                    fa_xy / prior.max(1e-6),
                    prior
                );
                state.logged_forearm_max[side] = fa_xy;
            }
        }
    }
}

/// Translate a wrist's whole finger chain (phalanges + fingertips) by
/// `d` so the hand stays coherent when the wrist is relocated. Shared
/// with the ray-IK stage.
pub(super) fn shift_hand_chain(sk: &mut SourceSkeleton, wrist: HumanoidBone, d: [f32; 3]) {
    let left = wrist == HumanoidBone::LeftHand;
    for (bone, joint) in sk.joints.iter_mut() {
        if is_finger_bone_side(*bone, left) {
            joint.position[0] += d[0];
            joint.position[1] += d[1];
            joint.position[2] += d[2];
        }
    }
    for (bone, joint) in sk.fingertips.iter_mut() {
        if is_finger_bone_side(*bone, left) {
            joint.position[0] += d[0];
            joint.position[1] += d[1];
            joint.position[2] += d[2];
        }
    }
}

pub(super) fn is_finger_bone_side(bone: HumanoidBone, left: bool) -> bool {
    let name = format!("{bone:?}");
    let side_ok = if left {
        name.starts_with("Left")
    } else {
        name.starts_with("Right")
    };
    side_ok
        && (name.contains("Thumb")
            || name.contains("Index")
            || name.contains("Middle")
            || name.contains("Ring")
            || name.contains("Little"))
}

fn xy_len(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    (dx * dx + dy * dy).sqrt()
}

/// COCO body keypoint indices for the arm chain, paired with the
/// (selfie-mirrored) humanoid bones they feed, plus the COCO hand
/// block belonging to the same anatomical arm. Layout: 0–16 body,
/// 17–22 feet, 23–90 face, 91–111 anatomical-LEFT hand, 112–132
/// anatomical-RIGHT hand. Selfie mirror: anatomical right → `Left*`
/// bones.
const ARM_KP: [(usize, usize, std::ops::Range<usize>, HumanoidBone, HumanoidBone, HumanoidBone, usize); 2] = [
    // (elbow_kp, wrist_kp, hand_kp_block, shoulder, elbow, wrist, side)
    (8, 10, 112..133, HumanoidBone::LeftUpperArm, HumanoidBone::LeftLowerArm, HumanoidBone::LeftHand, 0),
    (7, 9, 91..112, HumanoidBone::RightUpperArm, HumanoidBone::RightLowerArm, HumanoidBone::RightHand, 1),
];

/// Hand-keypoint SPREAD floor (frame-normalised bbox extent of the
/// 21-point block) below which the hand is treated as not actually
/// in view. Confidence does NOT discriminate — RTMW3D hallucinates
/// an out-of-frame hand at conf ≈ 0.55 — but the hallucination
/// collapses all 21 points onto the wrist stump (measured spread
/// 0.000 vs 0.106–0.139 for a real hand, including the
/// hand-toward-camera pose this gate must not misfire on).
const HAND_SPREAD_FLOOR: f32 = 0.02;

/// Foreshortening ratio that arms the out-of-frame test at all.
const SHORT_RATIO: f32 = 0.8;

/// Out-of-frame wrist detector. RTMW3D does NOT clamp a vanished
/// wrist at the frame border — it hallucinates it well inside, at
/// the visible forearm stump (measured 22% inboard on the crop
/// benchmark), so a positional edge test never fires. The reliable
/// signature is geometric + visibility:
///   1. the observed forearm is meaningfully shorter than the bone
///      length (`SHORT_RATIO`),
///   2. extending the observed elbow→wrist ray to full bone length
///      exits the frame (the missing length has somewhere to go), and
///   3. the hand keypoint block is near-invisible (rules out the
///      hand-toward-camera reading, which keeps fingers strong).
/// `get(i)` returns whole-frame-normalised `(nx, ny, score)`.
pub(in crate::tracking) fn wrist_out_of_frame(
    get: &dyn Fn(usize) -> Option<(f32, f32, f32)>,
    side: usize,
    forearm_len_frame: f32,
) -> bool {
    let (elbow_kp, wrist_kp, hand_block) = if side == 0 {
        (8usize, 10usize, 112..133usize)
    } else {
        (7, 9, 91..112)
    };
    let Some((ex, ey, _)) = get(elbow_kp) else {
        return false;
    };
    let Some((wx, wy, _)) = get(wrist_kp) else {
        return false;
    };
    // Decisive gate first: a hand actually in view (toward-camera
    // pose included) keeps its 21-keypoint block strong; a hand that
    // left the frame leaves it near the floor.
    let (mut hx0, mut hy0, mut hx1, mut hy1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for i in hand_block {
        if let Some((x, y, _)) = get(i) {
            hx0 = hx0.min(x);
            hy0 = hy0.min(y);
            hx1 = hx1.max(x);
            hy1 = hy1.max(y);
        }
    }
    let spread = if hx1 > hx0 { (hx1 - hx0).max(hy1 - hy0) } else { 0.0 };
    if spread >= HAND_SPREAD_FLOOR {
        return false;
    }

    // Mode A: clamp at the border. SimCC sometimes piles the wrist
    // mass directly on the boundary bins.
    const BORDER_EPS: f32 = 0.02;
    if wx < BORDER_EPS || wx > 1.0 - BORDER_EPS || wy < BORDER_EPS || wy > 1.0 - BORDER_EPS {
        return true;
    }

    // Mode B: hallucinated inboard at the visible forearm stump —
    // detectable because the observed forearm is short of bone
    // length AND extending the observed ray to full length exits
    // the frame.
    let dx = wx - ex;
    let dy = wy - ey;
    let obs = (dx * dx + dy * dy).sqrt();
    if obs < 1e-4 || obs >= forearm_len_frame * SHORT_RATIO {
        return false;
    }
    let k = forearm_len_frame / obs;
    let fx = ex + dx * k;
    let fy = ey + dy * k;
    fx < 0.0 || fx > 1.0 || fy < 0.0 || fy > 1.0
}

/// Repair edge-clamped arm keypoints: out-of-frame is missing data,
/// but the *direction* from the last visible parent joint toward the
/// clamp point is approximately right — only the length collapsed.
/// Restore the segment to its known bone length along the observed
/// direction, placing the joint at a legitimate out-of-frame source
/// coordinate. This also neutralises the false forward-z that the
/// bone-length reconstruction would otherwise fabricate from the
/// clamp-shortened projection (after extension the segment satisfies
/// the invariant, so `reconstruct_arm_z` is a no-op for it).
///
/// Runs BEFORE the wrist resilience layer (so the temporal hold sees
/// a plausible wrist instead of memorising the clamped one) and
/// before `reconstruct_arm_z`.
pub(super) fn repair_edge_clamped_arms(
    sk: &mut SourceSkeleton,
    joints: &[super::decode::DecodedJoint],
    frame_aspect: f32,
    state: &mut ArmLengthState,
) {
    state.frame += 1;
    observe_lengths(sk, state);
    if state.span_max <= 0.0 {
        return;
    }
    /// Confidence multiplier for an extrapolated joint — real enough
    /// for the solver to keep driving the limb, visibly lower so
    /// downstream consumers can tell it was not observed.
    const EXTRAPOLATED_CONF: f32 = 0.7;

    let span_ref = state.span_max.max(1e-3);
    let get = |i: usize| -> Option<(f32, f32, f32)> {
        joints.get(i).map(|j| (j.nx, j.ny, j.score))
    };
    // Frame-normalised forearm length: source-space lengths divide by
    // (2 × half-height = 2.0) in y and the aspect in x; for the ray
    // test we approximate with the y scale, which is exact for
    // vertical exits and conservative for horizontal ones.
    let _ = frame_aspect;

    for (_, _, _, sh_bone, el_bone, wr_bone, side) in ARM_KP {
        // Anthropometric BAND clamp: the running max fine-tunes the
        // segment length only within plausible anatomy. A raw max is
        // unusable live — an arm sweeping near the lens inflates the
        // projected length far past anatomy (measured 2x) and a pure
        // max never recovers.
        let l_fa_src = state.forearm_xy_max[side].clamp(
            FOREARM_SPAN_RATIO * span_ref * 0.8,
            FOREARM_SPAN_RATIO * span_ref * 1.35,
        );
        let l_ua_src = state.upper_xy_max[side].clamp(
            UPPER_ARM_SPAN_RATIO * span_ref * 0.8,
            UPPER_ARM_SPAN_RATIO * span_ref * 1.35,
        );
        // source units -> frame-normalised: y spans [-1, 1] over the
        // frame height, so 1 source unit = 0.5 frame units.
        let l_fa_frame = l_fa_src * 0.5;
        let raw_fired = wrist_out_of_frame(&get, side, l_fa_frame);

        // Hysteresis: engage after 2 consecutive detections, release
        // after 3 consecutive clears; engagement cross-fades the
        // extension in/out (~3 frames each way).
        let streak = &mut state.oof_streak[side];
        *streak = if raw_fired {
            (*streak).max(0) + 1
        } else {
            (*streak).min(0) - 1
        };
        let engaged_now = if *streak >= 2 {
            true
        } else if *streak <= -3 {
            false
        } else {
            state.engagement[side] > 0.5
        };
        let eng = &mut state.engagement[side];
        let prev_eng = *eng;
        *eng = (*eng + if engaged_now { 0.34 } else { -0.34 }).clamp(0.0, 1.0);
        if prev_eng == 0.0 && *eng > 0.0 {
            debug!(
                "arm-z diag f{} {}: edge-exit extension engaged",
                state.frame,
                side_tag(side)
            );
        } else if prev_eng > 0.0 && *eng == 0.0 {
            debug!(
                "arm-z diag f{} {}: edge-exit extension released",
                state.frame,
                side_tag(side)
            );
        }
        let fired = *eng > 0.0;
        let blend = *eng;

        let sh = sk.joints.get(&sh_bone).map(|j| j.position);
        let el = sk.joints.get(&el_bone).map(|j| j.position);
        let wr = sk.joints.get(&wr_bone).copied();

        if !fired {
            // Hand genuinely visible: refresh the held directions for
            // the moment it leaves the frame.
            if let (Some(sh), Some(el), Some(wr)) = (sh, el, wr.map(|j| j.position)) {
                if let Some(d) = unit3(sub3v(el, sh)) {
                    state.last_upper_dir[side] = Some(d);
                }
                if let Some(d) = unit3(sub3v(wr, el)) {
                    state.last_fore_dir[side] = Some(d);
                }
            }
            continue;
        }

        let (Some(sh), Some(el), Some(wr)) = (sh, el, wr) else {
            continue;
        };

        // Elbow first: when a held upper-arm direction exists,
        // re-place the elbow from the shoulder - the stub elbow is
        // itself displaced once most of the forearm is gone.
        let elbow_target = match state.last_upper_dir[side] {
            Some(d) => [
                sh[0] + d[0] * l_ua_src,
                sh[1] + d[1] * l_ua_src,
                sh[2] + d[2] * l_ua_src,
            ],
            None => el,
        };
        let elbow_new = lerp3(el, elbow_target, blend);
        if let Some(j) = sk.joints.get_mut(&el_bone) {
            j.position = elbow_new;
            j.confidence = (j.confidence * EXTRAPOLATED_CONF).max(0.05);
        }

        // Wrist continuation: held direction when available (the arm
        // slid out - the last visible direction is the best
        // estimate); otherwise geometric fallback (observed stub if
        // long enough, straight-arm continuation if not).
        const STUB_TRUST_RATIO: f32 = 0.35;
        let dir = if let Some(d) = state.last_fore_dir[side] {
            d
        } else {
            let dx = wr.position[0] - elbow_new[0];
            let dy = wr.position[1] - elbow_new[1];
            let obs = (dx * dx + dy * dy).sqrt();
            if obs >= l_fa_src * STUB_TRUST_RATIO {
                [dx / obs, dy / obs, 0.0]
            } else if let Some(d) = unit3(sub3v(elbow_new, sh)) {
                d
            } else {
                continue;
            }
        };
        let p_target = [
            elbow_new[0] + dir[0] * l_fa_src,
            elbow_new[1] + dir[1] * l_fa_src,
            elbow_new[2] + dir[2] * l_fa_src,
        ];
        let p = lerp3(wr.position, p_target, blend);
        let d = [
            p[0] - wr.position[0],
            p[1] - wr.position[1],
            p[2] - wr.position[2],
        ];
        if let Some(j) = sk.joints.get_mut(&wr_bone) {
            j.position = p;
            j.confidence = (j.confidence * EXTRAPOLATED_CONF).max(0.05);
        }
        shift_hand_chain(sk, wr_bone, d);
    }
}

/// Log tag for a side index: 0 → `Left*` bones, 1 → `Right*` bones
/// (selfie-mirrored, so "L" is the subject's anatomical right).
fn side_tag(side: usize) -> &'static str {
    if side == 0 {
        "L"
    } else {
        "R"
    }
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn sub3v(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn unit3(v: [f32; 3]) -> Option<[f32; 3]> {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l < 1e-4 {
        None
    } else {
        Some([v[0] / l, v[1] / l, v[2] / l])
    }
}
