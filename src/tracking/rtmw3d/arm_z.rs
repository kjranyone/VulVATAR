//! Arm keypoint repair on the projected (xy) plane: edge-exit
//! detection + restoration of clamp-shortened limbs, plus the
//! session length bookkeeping the repair's band clamp reads.
//!
//! The z-depth reconstruction that used to live here (`|dz| =
//! sqrt(L² − xy²)` against a session running-max length) is gone: the
//! running max was structurally corruptible — one near-lens hand sweep
//! inflated it past anatomy (measured 2×) and a pure max never recovers.
//! On the D435-exclusive pipeline the metric depth path
//! (`skeleton_from_depth`) supplies limb z directly, so this module now
//! only repairs 2D-plane keypoints.
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

/// Leak time constant (s) of the running maxima toward the current
/// measurement (only applied while the measurement is BELOW the held
/// max — an observation above it still snaps up instantly;
/// α = 1 − exp(−dt/τ), 0.02/frame at 30 fps). A pure running max is
/// structurally corruptible: one L/R mis-assignment or near-lens sweep
/// inflates it past anatomy and it never recovers, permanently
/// shifting the band clamp upward. With τ ≈ 1.65 s a corrupted max
/// decays back to within 5% of the true value in a few seconds while
/// genuine maxima (re-confirmed every extension) hold steady — in wall
/// time, regardless of the capture frame rate.
const LEN_MAX_TAU_S: f32 = 1.65;

/// Fold `sample` into a leaky running max (see [`LEN_MAX_TAU_S`]).
/// `dt_s` is the capture-timestamp frame step.
fn leaky_max(held: f32, sample: f32, dt_s: f32) -> f32 {
    if sample > held {
        sample
    } else {
        held + (1.0 - (-dt_s / LEN_MAX_TAU_S).exp()) * (sample - held)
    }
}

/// Per-session running maxima of projected segment lengths.
/// `[left, right]` per segment; reset with the rest of the temporal
/// state on session / benchmark-image boundaries.
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::tracking) struct ArmLengthState {
    upper_xy_max: [f32; 2],
    forearm_xy_max: [f32; 2],
    span_max: f32,
    /// Hand L/R swap-detector hysteresis (see [`correct_hand_lr_swap`]):
    /// consecutive frames the engage signature has held (pre-latch) and
    /// the engaged latch itself. Near the cost-ratio boundary a per-frame
    /// independent decision flips all 16 hand-chain bone pairs at frame
    /// rate — visually far worse than the transposition it fixes — so
    /// engagement requires [`SWAP_ENGAGE_FRAMES`] consecutive signatures
    /// and release uses a wider ratio ([`SWAP_RELEASE_RATIO`]).
    swap_streak: i32,
    swap_engaged: bool,
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
/// leaky running maxima. Called by the edge-exit repair at the top of
/// the frame; the repair's band clamp is the only consumer of the
/// maxima. `dt_s` drives the leak in wall time.
pub(super) fn observe_lengths(sk: &SourceSkeleton, state: &mut ArmLengthState, dt_s: f32) {
    if let (Some(l), Some(r)) = (
        sk.joints.get(&HumanoidBone::LeftUpperArm),
        sk.joints.get(&HumanoidBone::RightUpperArm),
    ) {
        if l.confidence >= MIN_CONF && r.confidence >= MIN_CONF {
            let span = xy_len(l.position, r.position);
            let grew = span > state.span_max;
            state.span_max = leaky_max(state.span_max, span, dt_s);
            if grew && span > state.logged_span * 1.05 {
                debug!(
                    "arm-z diag f{}: span_max grew {:.3} -> {:.3}",
                    state.frame, state.logged_span, span
                );
                state.logged_span = span;
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
        let ua_grew = ua_xy > state.upper_xy_max[side];
        state.upper_xy_max[side] = leaky_max(state.upper_xy_max[side], ua_xy, dt_s);
        if ua_grew {
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
        let fa_grew = fa_xy > state.forearm_xy_max[side];
        state.forearm_xy_max[side] = leaky_max(state.forearm_xy_max[side], fa_xy, dt_s);
        if fa_grew {
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

/// Left/right hand-chain bone pairs swapped together when the detector
/// transposes the two hand keypoint blocks (see [`correct_hand_lr_swap`]).
const HAND_LR_PAIRS: [(HumanoidBone, HumanoidBone); 16] = [
    (HumanoidBone::LeftHand, HumanoidBone::RightHand),
    (HumanoidBone::LeftThumbProximal, HumanoidBone::RightThumbProximal),
    (HumanoidBone::LeftThumbIntermediate, HumanoidBone::RightThumbIntermediate),
    (HumanoidBone::LeftThumbDistal, HumanoidBone::RightThumbDistal),
    (HumanoidBone::LeftIndexProximal, HumanoidBone::RightIndexProximal),
    (HumanoidBone::LeftIndexIntermediate, HumanoidBone::RightIndexIntermediate),
    (HumanoidBone::LeftIndexDistal, HumanoidBone::RightIndexDistal),
    (HumanoidBone::LeftMiddleProximal, HumanoidBone::RightMiddleProximal),
    (HumanoidBone::LeftMiddleIntermediate, HumanoidBone::RightMiddleIntermediate),
    (HumanoidBone::LeftMiddleDistal, HumanoidBone::RightMiddleDistal),
    (HumanoidBone::LeftRingProximal, HumanoidBone::RightRingProximal),
    (HumanoidBone::LeftRingIntermediate, HumanoidBone::RightRingIntermediate),
    (HumanoidBone::LeftRingDistal, HumanoidBone::RightRingDistal),
    (HumanoidBone::LeftLittleProximal, HumanoidBone::RightLittleProximal),
    (HumanoidBone::LeftLittleIntermediate, HumanoidBone::RightLittleIntermediate),
    (HumanoidBone::LeftLittleDistal, HumanoidBone::RightLittleDistal),
];

/// Minimum elbow x-separation (as a fraction of shoulder span) for the
/// swap test to arm. A genuine arms-crossed pose tucks the elbows
/// toward the centre — requiring the elbows to stay clearly L/R
/// separated restricts the correction to the "hands meet at the
/// midline, blocks transposed" case and leaves a deliberate cross
/// alone.
const SWAP_ELBOW_SEP_RATIO: f32 = 0.3;

/// Correct a left/right HAND-BLOCK transposition from the detector.
/// When two hands meet at the midline, RTMW3D routinely assigns the
/// anatomical-left hand keypoints to the right and vice versa: the
/// elbows stay correctly placed but the MCP-centroid wrists (and their
/// finger chains) land on the wrong sides, so the avatar's forearms
/// cross. The signature is unambiguous and measurable — swapping the
/// two hands sharply SHORTENS both forearms, because each hand belongs
/// to its nearest elbow. Gated so it only fires when the elbows are
/// clearly L/R separated (not a tucked genuine cross), the hands are
/// inverted in x, sit close together (the detector-confusion regime),
/// and the swap shortens the forearms by a clear margin. Runs before
/// the edge repair / ray-IK so all downstream stages see correct hands.
/// Cost-ratio below which the swap signature ENGAGES (swap clearly
/// shortens the forearms) and the wider ratio below which an already
/// engaged latch keeps applying. The gap is the deadband that stops
/// boundary oscillation; release also requires the ratio to leave the
/// deadband, not just dip out of the engage window.
const SWAP_ENGAGE_RATIO: f32 = 0.8;
const SWAP_RELEASE_RATIO: f32 = 0.95;
/// Consecutive signature frames required before the latch engages
/// (one frame of detector noise must not flip 16 bone pairs).
const SWAP_ENGAGE_FRAMES: i32 = 2;

pub(in crate::tracking) fn correct_hand_lr_swap(sk: &mut SourceSkeleton, state: &mut ArmLengthState) {
    let apply = |sk: &mut SourceSkeleton, cur: f32, swapped: f32| {
        for (l, r) in HAND_LR_PAIRS {
            swap_joint_pair(&mut sk.joints, l, r);
            swap_joint_pair(&mut sk.fingertips, l, r);
        }
        std::mem::swap(&mut sk.left_hand_orientation, &mut sk.right_hand_orientation);
        debug!(
            "hand-swap: corrected L/R hand-block transposition (forearm cost {:.3} -> {:.3} swapped)",
            cur, swapped
        );
    };

    let gj = |sk: &SourceSkeleton, b: HumanoidBone| sk.joints.get(&b).copied();
    let (Some(wlj), Some(wrj)) = (
        gj(sk, HumanoidBone::LeftHand),
        gj(sk, HumanoidBone::RightHand),
    ) else {
        // No hands at all — the confusion regime is over.
        state.swap_streak = 0;
        state.swap_engaged = false;
        return;
    };
    let elbows = (
        gj(sk, HumanoidBone::LeftLowerArm),
        gj(sk, HumanoidBone::RightLowerArm),
    );
    let shoulders = (
        gj(sk, HumanoidBone::LeftUpperArm),
        gj(sk, HumanoidBone::RightUpperArm),
    );
    let ((Some(elj), Some(erj)), (Some(slj), Some(srj))) = (elbows, shoulders) else {
        // A depth hole dropped an elbow/shoulder — exactly the frames
        // where the confusion is most likely, and where a per-frame
        // decision used to go silent. The latch carries the last
        // confident decision across the hole instead of flipping.
        if state.swap_engaged {
            apply(sk, f32::NAN, f32::NAN);
        } else {
            state.swap_streak = 0;
        }
        return;
    };
    let (sl, sr, el, er) = (slj.position, srj.position, elj.position, erj.position);
    let (wl, wr) = (wlj.position, wrj.position);

    let dist3 = |a: [f32; 3], b: [f32; 3]| {
        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    };
    let span = dist3(sl, sr);
    if span < 1e-3 {
        state.swap_streak = 0;
        state.swap_engaged = false;
        return;
    }

    // Cost on the projection plane only. This correction runs on the
    // RTMW3D skeleton BEFORE external-depth injection (`rtmw3d/mod.rs`),
    // so `metric_depth_m` is always `None` here and z is monocular
    // noise — a 3D cost would be judging garbage. Depth IS the
    // strongest hand↔elbow discriminator, but exploiting it belongs to
    // the post-injection layer (`skeleton_from_depth`), where metric z
    // actually exists; moving/duplicating the swap test there is a
    // known future improvement, not this stage's job.
    let cur = xy_len(el, wl) + xy_len(er, wr);
    let swapped = xy_len(el, wr) + xy_len(er, wl);

    // Structural arming gates — they define when a NEW engagement may
    // start; an engaged latch is governed by the cost ratio alone so
    // momentary gate noise cannot flip the hands mid-hold.
    //  * elbows clearly L/R separated (source +x is anatomical-left):
    //    a tucked genuine cross must not arm the detector,
    //  * hands inverted in x (Left hand on the −x side of Right),
    //  * hands close together (the detector-confusion regime).
    let armed = el[0] - er[0] >= SWAP_ELBOW_SEP_RATIO * span
        && wl[0] < wr[0]
        && dist3(wl, wr) <= 1.5 * span;

    if state.swap_engaged {
        if swapped < SWAP_RELEASE_RATIO * cur {
            apply(sk, cur, swapped);
        } else {
            state.swap_engaged = false;
            state.swap_streak = 0;
        }
        return;
    }
    if armed && swapped < SWAP_ENGAGE_RATIO * cur {
        state.swap_streak += 1;
        if state.swap_streak >= SWAP_ENGAGE_FRAMES {
            state.swap_engaged = true;
            apply(sk, cur, swapped);
        }
    } else {
        state.swap_streak = 0;
    }
}

fn swap_joint_pair(
    map: &mut std::collections::HashMap<HumanoidBone, super::super::source_skeleton::SourceJoint>,
    l: HumanoidBone,
    r: HumanoidBone,
) {
    let lv = map.remove(&l);
    let rv = map.remove(&r);
    if let Some(v) = rv {
        map.insert(l, v);
    }
    if let Some(v) = lv {
        map.insert(r, v);
    }
}

pub(super) fn is_finger_bone_side(bone: HumanoidBone, left: bool) -> bool {
    // Direct enum match — this runs per-bone inside shift_hand_chain's
    // full-map scans, where the previous Debug-format string allocation
    // was pure hot-path waste.
    use HumanoidBone::*;
    let is_left = match bone {
        LeftThumbProximal | LeftThumbIntermediate | LeftThumbDistal | LeftIndexProximal
        | LeftIndexIntermediate | LeftIndexDistal | LeftMiddleProximal
        | LeftMiddleIntermediate | LeftMiddleDistal | LeftRingProximal
        | LeftRingIntermediate | LeftRingDistal | LeftLittleProximal
        | LeftLittleIntermediate | LeftLittleDistal => true,
        RightThumbProximal | RightThumbIntermediate | RightThumbDistal | RightIndexProximal
        | RightIndexIntermediate | RightIndexDistal | RightMiddleProximal
        | RightMiddleIntermediate | RightMiddleDistal | RightRingProximal
        | RightRingIntermediate | RightRingDistal | RightLittleProximal
        | RightLittleIntermediate | RightLittleDistal => false,
        _ => return false,
    };
    is_left == left
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
/// `forearm_len_src` is the bone length in SOURCE units and
/// `frame_aspect` the frame's width/height: the observed stub length is
/// measured isotropically (1 frame unit of x = `2·aspect` source units,
/// 1 of y = `2`) so a horizontal exit — the overwhelmingly common case —
/// is judged against the true length instead of a y-scale approximation
/// that over-fires by up to the aspect ratio (1.78× at 16:9), each
/// over-fire stretching a real wrist out to full bone length.
pub(in crate::tracking) fn wrist_out_of_frame(
    get: &dyn Fn(usize) -> Option<(f32, f32, f32)>,
    side: usize,
    forearm_len_src: f32,
    frame_aspect: f32,
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
    // Lengths compare in isotropic source units (1 frame unit of x =
    // 2·aspect source units, 1 of y = 2); the extension itself is
    // affine along the ray, so scaling the frame coordinates by the
    // same factor k stays geometrically exact.
    let dx = wx - ex;
    let dy = wy - ey;
    let dx_src = dx * 2.0 * frame_aspect.max(1e-3);
    let dy_src = dy * 2.0;
    let obs = (dx_src * dx_src + dy_src * dy_src).sqrt();
    if obs < 1e-4 || obs >= forearm_len_src * SHORT_RATIO {
        return false;
    }
    let k = forearm_len_src / obs;
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
    dt_s: f32,
) {
    state.frame += 1;
    observe_lengths(sk, state, dt_s);
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
        let raw_fired = wrist_out_of_frame(&get, side, l_fa_src, frame_aspect);

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

#[cfg(test)]
mod swap_tests {
    use super::*;
    use super::super::super::source_skeleton::{SourceJoint, SourceSkeleton};

    fn j(p: [f32; 3]) -> SourceJoint {
        SourceJoint { position: p, confidence: 0.9, metric_depth_m: None }
    }

    fn arms(sk: &mut SourceSkeleton, sl: [f32; 3], sr: [f32; 3], el: [f32; 3], er: [f32; 3], wl: [f32; 3], wr: [f32; 3]) {
        sk.joints.insert(HumanoidBone::LeftUpperArm, j(sl));
        sk.joints.insert(HumanoidBone::RightUpperArm, j(sr));
        sk.joints.insert(HumanoidBone::LeftLowerArm, j(el));
        sk.joints.insert(HumanoidBone::RightLowerArm, j(er));
        sk.joints.insert(HumanoidBone::LeftHand, j(wl));
        sk.joints.insert(HumanoidBone::RightHand, j(wr));
    }

    /// Measured fingertips_touch probe: elbows correct, hands swapped.
    /// The fix relabels them so LeftHand is back on +x, RightHand −x.
    #[test]
    fn corrects_measured_hand_block_swap() {
        let mut sk = SourceSkeleton::empty(0);
        arms(
            &mut sk,
            [0.097, 0.352, 0.025],
            [-0.062, 0.349, -0.088],
            [0.168, 0.195, 0.015],
            [-0.141, 0.213, -0.098],
            [-0.080, 0.225, 0.034], // LeftHand on the wrong (−x) side
            [0.129, 0.244, -0.078], // RightHand on the wrong (+x) side
        );
        // Tag a finger on each side to confirm the whole chain swaps.
        sk.joints.insert(HumanoidBone::LeftIndexProximal, j([-0.10, 0.22, 0.03]));
        sk.joints.insert(HumanoidBone::RightIndexProximal, j([0.11, 0.24, -0.08]));

        // The latch engages on the SWAP_ENGAGE_FRAMES-th consecutive
        // signature frame; the first call only counts.
        let mut st = ArmLengthState::default();
        correct_hand_lr_swap(&mut sk, &mut st);
        assert!(
            sk.joints[&HumanoidBone::LeftHand].position[0] < 0.0,
            "one signature frame must not flip the hands yet"
        );
        correct_hand_lr_swap(&mut sk, &mut st);

        assert!(sk.joints[&HumanoidBone::LeftHand].position[0] > 0.0, "LeftHand back on +x");
        assert!(sk.joints[&HumanoidBone::RightHand].position[0] < 0.0, "RightHand back on −x");
        assert!(sk.joints[&HumanoidBone::LeftIndexProximal].position[0] > 0.0, "L finger chain swapped");
        assert!(sk.joints[&HumanoidBone::RightIndexProximal].position[0] < 0.0, "R finger chain swapped");
    }

    /// A clean hands-together pose (hands correctly sided, just close)
    /// must be left untouched.
    #[test]
    fn leaves_clean_hands_together_alone() {
        let mut sk = SourceSkeleton::empty(0);
        arms(
            &mut sk,
            [0.20, 0.40, 0.0],
            [-0.20, 0.40, 0.0],
            [0.28, 0.15, 0.25],
            [-0.28, 0.15, 0.25],
            [0.03, 0.0, 0.5],   // LeftHand correctly on +x
            [-0.03, 0.0, 0.5],  // RightHand correctly on −x
        );
        let mut st = ArmLengthState::default();
        correct_hand_lr_swap(&mut sk, &mut st);
        correct_hand_lr_swap(&mut sk, &mut st);
        assert!(sk.joints[&HumanoidBone::LeftHand].position[0] > 0.0);
        assert!(sk.joints[&HumanoidBone::RightHand].position[0] < 0.0);
    }

    /// Genuine arms-crossed pose with the elbows tucked toward centre
    /// must NOT be "corrected" — the elbow-separation gate blocks it.
    #[test]
    fn leaves_genuine_cross_with_tucked_elbows() {
        let mut sk = SourceSkeleton::empty(0);
        // Elbows near the midline (tucked), hands at opposite sides:
        // a deliberate chest cross, not a detector transposition.
        arms(
            &mut sk,
            [0.20, 0.40, 0.0],
            [-0.20, 0.40, 0.0],
            [0.05, 0.10, 0.2],   // L elbow just left of centre
            [-0.05, 0.10, 0.2],  // R elbow just right of centre
            [-0.18, 0.05, 0.25], // L hand reaches across to −x
            [0.18, 0.05, 0.25],  // R hand reaches across to +x
        );
        let before = sk.joints[&HumanoidBone::LeftHand].position[0];
        let mut st = ArmLengthState::default();
        correct_hand_lr_swap(&mut sk, &mut st);
        correct_hand_lr_swap(&mut sk, &mut st);
        assert_eq!(
            sk.joints[&HumanoidBone::LeftHand].position[0], before,
            "tucked-elbow genuine cross must be left alone"
        );
    }

    /// Build the raw (detector-output) frame for the hysteresis test:
    /// elbows fixed L/R at ±0.3, hands at ∓a (inverted), so the
    /// swapped/current forearm-cost ratio is (0.6−2a)/(0.6+2a).
    fn raw_frame(a: f32) -> SourceSkeleton {
        let mut sk = SourceSkeleton::empty(0);
        arms(
            &mut sk,
            [0.20, 0.40, 0.0],
            [-0.20, 0.40, 0.0],
            [0.30, 0.0, 0.0],
            [-0.30, 0.0, 0.0],
            [-a, 0.0, 0.0], // LeftHand on −x: inverted
            [a, 0.0, 0.0],
        );
        sk
    }

    /// Boundary oscillation: once engaged, a frame whose cost ratio sits
    /// between the engage (0.8) and release (0.95) thresholds must KEEP
    /// the swap applied (a per-frame decision would flip 16 bone pairs
    /// back), and only a ratio past the release threshold lets go.
    #[test]
    fn swap_latch_holds_through_deadband_and_releases_past_it() {
        let mut st = ArmLengthState::default();

        // Two strong-signature frames (a=0.05 → ratio ≈ 0.73): engage.
        let mut f = raw_frame(0.05);
        correct_hand_lr_swap(&mut f, &mut st);
        let mut f = raw_frame(0.05);
        correct_hand_lr_swap(&mut f, &mut st);
        assert!(f.joints[&HumanoidBone::LeftHand].position[0] > 0.0, "engaged");

        // Deadband frame (a=0.016 → ratio ≈ 0.90): would NOT engage on
        // its own, but the latch must hold.
        let mut f = raw_frame(0.016);
        correct_hand_lr_swap(&mut f, &mut st);
        assert!(
            f.joints[&HumanoidBone::LeftHand].position[0] > 0.0,
            "deadband frame must keep the engaged swap applied"
        );

        // Hands back on their correct sides (swap would LENGTHEN):
        // release, and the frame passes through unswapped.
        let mut f = raw_frame(0.05);
        let (wl, wr) = (
            f.joints[&HumanoidBone::LeftHand].position,
            f.joints[&HumanoidBone::RightHand].position,
        );
        f.joints.get_mut(&HumanoidBone::LeftHand).unwrap().position = wr;
        f.joints.get_mut(&HumanoidBone::RightHand).unwrap().position = wl;
        correct_hand_lr_swap(&mut f, &mut st);
        assert!(
            f.joints[&HumanoidBone::LeftHand].position[0] > 0.0,
            "correctly-sided hands stay put after release"
        );

        // And re-engagement needs the full streak again.
        let mut f = raw_frame(0.05);
        correct_hand_lr_swap(&mut f, &mut st);
        assert!(
            f.joints[&HumanoidBone::LeftHand].position[0] < 0.0,
            "post-release, one signature frame must not re-engage"
        );
    }

    /// A depth hole dropping an elbow mid-hold must not flip the hands:
    /// the engaged latch carries the decision across the hole.
    #[test]
    fn swap_latch_carries_across_missing_elbow() {
        let mut st = ArmLengthState::default();
        let mut f = raw_frame(0.05);
        correct_hand_lr_swap(&mut f, &mut st);
        let mut f = raw_frame(0.05);
        correct_hand_lr_swap(&mut f, &mut st);
        assert!(f.joints[&HumanoidBone::LeftHand].position[0] > 0.0, "engaged");

        let mut f = raw_frame(0.05);
        f.joints.remove(&HumanoidBone::LeftLowerArm);
        correct_hand_lr_swap(&mut f, &mut st);
        assert!(
            f.joints[&HumanoidBone::LeftHand].position[0] > 0.0,
            "hole frame must keep applying the engaged swap"
        );
    }

    /// The session length maxima must recover from a one-off corruption
    /// (L/R mis-assignment inflating the shoulder span): a pure running
    /// max never comes back down, permanently shifting the band clamp.
    #[test]
    fn span_max_decays_back_after_corruption() {
        const DT: f32 = 1.0 / 30.0;
        let mut st = ArmLengthState::default();
        let frame = |span: f32| {
            let mut sk = SourceSkeleton::empty(0);
            sk.joints.insert(HumanoidBone::LeftUpperArm, j([span / 2.0, 0.4, 0.0]));
            sk.joints.insert(HumanoidBone::RightUpperArm, j([-span / 2.0, 0.4, 0.0]));
            sk
        };
        // One corrupted frame at 2× the true span…
        observe_lengths(&frame(2.0), &mut st, DT);
        assert!(st.span_max >= 2.0 - 1e-6);
        // …then a few seconds of honest frames.
        for _ in 0..200 {
            observe_lengths(&frame(1.0), &mut st, DT);
        }
        assert!(
            st.span_max < 1.1,
            "corrupted span_max must decay back toward the true span, got {}",
            st.span_max
        );
        assert!(st.span_max >= 1.0 - 1e-6, "never decays below the live observation");
    }

    /// The leak is a wall-time constant, not a per-frame one: the same
    /// seconds of honest observation must shed the same fraction of a
    /// corruption at any capture rate.
    #[test]
    fn span_max_decay_is_framerate_invariant() {
        const DT: f32 = 1.0 / 30.0;
        let frame = |span: f32| {
            let mut sk = SourceSkeleton::empty(0);
            sk.joints.insert(HumanoidBone::LeftUpperArm, j([span / 2.0, 0.4, 0.0]));
            sk.joints.insert(HumanoidBone::RightUpperArm, j([-span / 2.0, 0.4, 0.0]));
            sk
        };
        // 2 s of honest frames after the same corruption, at 30 vs 15 fps.
        let mut st30 = ArmLengthState::default();
        observe_lengths(&frame(2.0), &mut st30, DT);
        for _ in 0..60 {
            observe_lengths(&frame(1.0), &mut st30, DT);
        }
        let mut st15 = ArmLengthState::default();
        observe_lengths(&frame(2.0), &mut st15, 2.0 * DT);
        for _ in 0..30 {
            observe_lengths(&frame(1.0), &mut st15, 2.0 * DT);
        }
        assert!(
            (st30.span_max - st15.span_max).abs() < 0.01,
            "2 s of decay must match across frame rates (30 fps → {}, 15 fps → {})",
            st30.span_max,
            st15.span_max
        );
    }

    /// Horizontal-exit judgement must use the true (aspect-corrected)
    /// stub length: at 16:9 the old y-scale approximation read a fully
    /// in-frame horizontal stub as "short + exits" and stretched a real
    /// wrist out to bone length.
    #[test]
    fn wrist_out_of_frame_respects_aspect_on_horizontal_stubs() {
        let aspect = 16.0 / 9.0;
        let mk = |ex: f32, ey: f32, wx: f32, wy: f32| {
            move |i: usize| -> Option<(f32, f32, f32)> {
                match i {
                    8 => Some((ex, ey, 1.0)),
                    10 => Some((wx, wy, 1.0)),
                    _ => None, // hand block absent → spread 0, gate passes
                }
            }
        };
        // In-frame horizontal stub (obs ≈ 0.36 src of l=0.5): extending
        // to full length stays inside → must NOT fire. (The y-scale
        // approximation called this k=2.5 → exit at fx=1.05.)
        let g = mk(0.8, 0.5, 0.9, 0.5);
        assert!(
            !wrist_out_of_frame(&g, 0, 0.5, aspect),
            "in-frame horizontal stub must not fire at 16:9"
        );
        // Genuine horizontal exit: extension leaves the frame → fires.
        let g = mk(0.88, 0.5, 0.96, 0.5);
        assert!(
            wrist_out_of_frame(&g, 0, 0.5, aspect),
            "true horizontal exit must still fire"
        );
        // Vertical exit unaffected by aspect handling.
        let g = mk(0.5, 0.8, 0.5, 0.93);
        assert!(
            wrist_out_of_frame(&g, 0, 0.5, aspect),
            "vertical exit must still fire"
        );
    }
}
