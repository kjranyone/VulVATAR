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

pub(in crate::tracking) fn build_source_skeleton(
    frame_index: u64,
    joints: &[DecodedJoint],
    width: u32,
    height: u32,
    force_shoulder_anchor: bool,
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
    //
    // `force_shoulder_anchor = true` (UpperBody calibration mode)
    // overrides the confidence-based gate: the hip keypoint detector
    // hallucinates a plausible-looking hip position even when the
    // subject's hips are entirely below the frame, so confidence
    // alone isn't enough to reject it. With this flag set, lower-
    // body keypoints (Hips + UpperLeg/LowerLeg/Foot/Toes) are
    // dropped entirely via the existing `!hip_visible` gates below
    // and the origin always comes from the shoulder pair.
    let lh = &joints[11];
    let rh = &joints[12];
    let hip_score = lh.score.min(rh.score);
    let hip_visible = !force_shoulder_anchor && hip_score >= KEYPOINT_VISIBILITY_FLOOR;

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

    // Root offset: anchor mid relative to image centre, in the same
    // source-space units the joints use. This is what lets the solver
    // *translate* the avatar's Hips bone (instead of just rotating it
    // for body yaw) so the avatar follows the subject side-stepping or
    // leaning into the camera, and it's the per-frame sample the pose-
    // calibration modal accumulates.
    //
    // Emitted whenever any anchor was detected — hip preferred, with
    // shoulder fallback for upper-body framing. `root_anchor_is_hip`
    // tells downstream consumers which anchor produced the value so
    // calibration mode-matching and EMA seed selection can gate
    // appropriately. RTMW3D has no metric depth, so Z is always 0.
    let root_x = -(origin_nx - 0.5) * (2.0 * aspect);
    let root_y = -(origin_ny - 0.5) * 2.0;
    sk.root_offset = Some([root_x, root_y, 0.0]);
    sk.root_anchor_is_hip = hip_visible;

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
            metric_depth_m: None,
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
                metric_depth_m: None,
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
                    metric_depth_m: None,
                },
            );
        }
    }

    // Spine-chain proxies + Head. The solver drives Chest, UpperChest,
    // and Neck via derived directions (Spine/Chest = hip->shoulder mid,
    // UpperChest/Neck = shoulder mid -> Head proxy). None of those
    // bones have a dedicated COCO keypoint, so without dual-inserts
    // here `pose_solver`'s `source.joints.get(&bone)` lookup fails on
    // each and the whole spine chain stays at rest pose — including
    // for chin-thrust, where Neck is the bone that should bend.
    inject_spine_chain_proxies(&mut sk, joints, &to_source, hip_visible, hip_score);

    // Hands. COCO Wholebody indices 91..=111 are the subject's left

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

    // Anatomical sanity check for leg chain (knee → ankle). ViTPose
    // occasionally misdetects the ankle as a point well off the
    // anatomical "downward" axis from the knee — observed on
    // shoulders_003 where the LeftFoot landed at x=+0.775 vs y=-0.632
    // from the knee, giving a 51° shin angle from vertical that the
    // avatar's solver can't follow without rig damage. Cap the shin
    // x/y ratio so a wildly tilted source ankle gets pulled back to
    // a straight-down position from the knee; this preserves the
    // knee position (which is the primary leg joint signal) and
    // gives the foot a plausible default.
    sanity_check_shin(&mut sk, HumanoidBone::LeftLowerLeg, HumanoidBone::LeftFoot);
    sanity_check_shin(&mut sk, HumanoidBone::RightLowerLeg, HumanoidBone::RightFoot);

    // 2D-only foreshortening → spine forward-pitch inference. Fires
    // only when the upper-body source-z is essentially flat (= caller
    // is a 2D-only provider like ViTPose with `nz = 0.5`). RTMW3D's
    // own 3D output produces non-trivial source z so the gate inside
    // skips the inference. See the function docstring for the
    // anatomy reasoning. `pitch_sign` is derived from head /
    // face landmarks (chin-down vs chin-up) so backward-leaning
    // poses get -z injection instead of the default forward bias.
    // Spine pitch: gated and scaled by the 2D head-pose signal
    // (nose-vs-ear-midpoint y ratio). Face-pose-derived pitch was
    // attempted (v3p) but produced near-zero pitch on deep bow
    // cases where the face landmarks are noisy, undershooting
    // depth_006. The nose-vs-ear heuristic in detect_pitch_scale
    // is more robust because it uses only the COCO 0/3/4 body
    // keypoints which stay reliable when the face cascade does
    // not.
    if let Some(pitch_scale) = detect_pitch_scale(joints) {
        // Estimate head pitch angle in radians from the nose-vs-
        // ear-midpoint signal. Use it as a cap on the inferred
        // body pitch angle: head can pitch *further* than body
        // (independent neck tilt) but not *less* in normal poses
        // (rigid neck-spine coupling). For balance_001 this caps
        // the foreshortening-derived 45° down to ~20°, matching
        // the photo's mild lean; for depth_006 the head pitch
        // is large so cap doesn't bind and full pitch fires.
        let head_pitch_cap = estimate_head_pitch_angle(joints);
        infer_spine_pitch_from_foreshortening(&mut sk, pitch_scale, head_pitch_cap);
    }
    // Same idea, orthogonal axis: shoulder / hip pair X-span
    // compression → body yaw around vertical (twist poses). Sign
    // disambiguation uses the raw COCO nose + ear-pair keypoints:
    // when the body yaws, the face yaws with it, and the nose's
    // image-x shifts relative to the ear-midpoint in the direction
    // that side of the face is exposed.
    let yaw_sign = detect_yaw_sign(joints);
    infer_body_yaw_from_foreshortening(&mut sk, yaw_sign);

    // Face pose: derive yaw / pitch / roll from the 5 body-face
    // keypoints (nose + eyes + ears, COCO 0..=4) when they clear
    // confidence. Drives the Head bone independently of the body
    // direction and gives the solver a face-yaw signal that's
    // orthogonal to my spine-yaw inference. (v3x probe confirmed
    // face pose does NOT contribute to the 5.4° neck residual —
    // that's a pose_solver-level rest-pose offset artefact.)
    sk.face = super::face::derive_face_pose_from_body(&sk, joints);
    sk.expressions = Vec::new();
    sk
}

/// Recover a spine forward-pitch (around X axis) signal from 2D
/// foreshortening of the shoulder-hip y span. ViTPose and any other
/// 2D-only provider emits `position[2] = 0` for every joint, so the
/// solver's direction-matching path sees the spine direction as
/// pure `+y` regardless of whether the subject is upright or
/// deeply bowed. This routine bridges the gap by injecting a
/// per-frame `+z` offset into the spine-chain tip joints derived
/// from the anatomical relation
///
/// ```text
/// shoulder-hip y distance ≈ 1.15 × shoulder x span (upright frontal adult)
/// ```
///
/// When the observed y span is shorter than that prediction, the
/// subject is foreshortened — either leaning forward (toward
/// camera) or backward (away). Without a depth signal we cannot
/// distinguish the two, so we **default to forward**: subjects
/// performing for a camera are vastly more likely to lean toward
/// it than away, and the rest-pose neutral covers the rare
/// backward-lean case with a 5° error budget.
///
/// Gating:
/// - Skips when `UpperChest.z` already carries a real depth signal
///   (> `MAX_EXISTING_Z_FOR_INFERENCE` in source units), so depth-
///   aware providers and RTMW3D's native 3D output don't get this
///   bias added on top.
/// - Skips when the subject is side-profile (`shoulder_span_x` too
///   small) — direction is ambiguous and the anatomy ratio doesn't
///   hold.
/// - Skips when compression is < 5 % (subject is upright; jitter
///   should not push the avatar around).
/// Constrain the shin direction (knee → ankle) to a plausibly
/// downward orientation. ViTPose's ankle detection misfires on
/// some images, placing the foot at a wildly off-axis position
/// (51°+ from vertical) that the avatar's leg chain can't follow.
/// We cap the horizontal offset to half the vertical offset
/// (≈ 27° from vertical max). Outside that, snap the ankle x to
/// be inline with the knee while preserving the y/z components —
/// the shin then points roughly downward like the actual anatomy.
fn sanity_check_shin(sk: &mut SourceSkeleton, knee: HumanoidBone, ankle: HumanoidBone) {
    /// Max |dx| / |dy| ratio for shin direction. ≈ 27° from
    /// vertical. Beyond this is a detection error, not a real
    /// anatomical pose (legs splaying outward extremely is rare
    /// in standing / standard poses; a kick or sit-cross pose
    /// would also bend the knee, which we don't model here).
    const MAX_SHIN_LATERAL_RATIO: f32 = 0.5;

    let Some(knee_j) = sk.joints.get(&knee).copied() else {
        return;
    };
    let Some(ankle_j) = sk.joints.get(&ankle).copied() else {
        return;
    };
    let dx = ankle_j.position[0] - knee_j.position[0];
    let dy = ankle_j.position[1] - knee_j.position[1];
    // shin must point downward: dy < 0
    if dy >= 0.0 {
        // Ankle above knee — anatomically impossible; pull it
        // directly below the knee at expected shin length.
        if let Some(j) = sk.joints.get_mut(&ankle) {
            j.position[0] = knee_j.position[0];
            j.position[1] = knee_j.position[1] - 0.30;
        }
        return;
    }
    if dx.abs() > MAX_SHIN_LATERAL_RATIO * dy.abs() {
        // Lateral offset too large; snap x directly to knee
        // (= vertical shin). Earlier "clamp to max ratio"
        // approach left a 20° residual error because the source
        // detection was unreliable in this regime — better to
        // default to the anatomically common vertical posture
        // than partially honour a bad detection.
        if let Some(j) = sk.joints.get_mut(&ankle) {
            j.position[0] = knee_j.position[0];
        }
    }
}

fn infer_spine_pitch_from_foreshortening(
    sk: &mut SourceSkeleton,
    pitch_sign: f32,
    head_pitch_cap: Option<f32>,
) {
    /// Anatomical ratio for an upright frontal adult: shoulder-to-
    /// hip y distance divided by shoulder x span. Drives the
    /// expected y span the per-frame measurement is compared
    /// against.
    const EXPECTED_SHOULDER_HIP_RATIO: f32 = 1.15;
    /// Above this absolute source-z value on the UpperChest, the
    /// caller has already produced a depth signal — assume it's the
    /// truth and don't overwrite.
    const MAX_EXISTING_Z_FOR_INFERENCE: f32 = 0.05;
    /// Below this shoulder x span, the subject is in side-profile
    /// and the foreshortening signal is ambiguous (the shoulders
    /// themselves are now z-separated, not x-separated).
    const MIN_SHOULDER_SPAN_FOR_INFERENCE: f32 = 0.10;
    /// Trigger threshold. compression ≥ this is "essentially
    /// upright"; below is treated as forward lean.
    const COMPRESSION_THRESHOLD: f32 = 0.95;

    // Require an explicit Hips entry. Without it the subject is in
    // upper-body framing (`force_shoulder_anchor` mode) and origin is
    // shoulder-mid — `chest.y` collapses to ~0 and the foreshortening
    // ratio is meaningless (we'd divide by ~0 and inject a giant z).
    let Some(hips) = sk.joints.get(&HumanoidBone::Hips).copied() else {
        return;
    };
    let Some(chest) = sk.joints.get(&HumanoidBone::UpperChest).copied() else {
        return;
    };
    if chest.position[2].abs() > MAX_EXISTING_Z_FOR_INFERENCE {
        return;
    }
    let (lsh, rsh) = match (
        sk.joints.get(&HumanoidBone::LeftShoulder),
        sk.joints.get(&HumanoidBone::RightShoulder),
    ) {
        (Some(l), Some(r)) => (*l, *r),
        _ => return,
    };
    let shoulder_span = (lsh.position[0] - rsh.position[0]).abs();
    if shoulder_span < MIN_SHOULDER_SPAN_FOR_INFERENCE {
        return;
    }
    // Skip the forward-pitch interpretation when the shoulder line
    // is significantly tilted in y: that's a roll (side lean), not
    // a pitch (forward lean). Treating roll-induced y compression
    // as forward pitch produces visible contortion artefacts —
    // tested on the `three_quarter_left_side_lunge` validation
    // image, where pitch inference fired at ~50° and bent the
    // avatar forward when the subject was actually leaning side-
    // ways. 0.20 source units of y tilt across the shoulder pair
    // corresponds to ~30° roll, well outside the noise floor.
    const MAX_SHOULDER_LINE_Y_TILT: f32 = 0.20;
    let shoulder_line_y_tilt = (lsh.position[1] - rsh.position[1]).abs();
    if shoulder_line_y_tilt > MAX_SHOULDER_LINE_Y_TILT {
        return;
    }

    let hip_y = hips.position[1];
    let actual_y = chest.position[1] - hip_y;
    // Minimum sane shoulder-hip y distance to attempt the ratio
    // inference. Below this the geometry is degenerate (shoulder
    // overlaps hip in image y, e.g. extreme rear-view torso) and the
    // division below would amplify floating-point noise into a
    // catastrophic z injection. Tuned against the validation set:
    // legitimate forward-lean cases sit at ≥ 0.08 source units.
    const MIN_ACTUAL_Y: f32 = 0.05;
    if actual_y < MIN_ACTUAL_Y {
        return;
    }

    let expected_y = shoulder_span * EXPECTED_SHOULDER_HIP_RATIO;
    let compression = (actual_y / expected_y).clamp(-1.0, 1.0);
    if compression >= COMPRESSION_THRESHOLD {
        return;
    }

    // Reconstruct the missing z component so the 3-D shoulder-hip
    // vector still has length `expected_y` but is rotated by
    // `acos(compression)` toward the camera. cos_pitch = compression,
    // sin_pitch = √(1 − cos²). Sign comes from the caller's face-
    // pitch signal (chin-down = +, chin-up = −); default + on no-
    // signal still picks the more common performer direction.
    let cos_pitch = compression;
    let pitch_angle = cos_pitch.acos();
    // Cap the body pitch by the head pitch estimate when
    // available — body normally pitches ≤ head, so a small
    // head pitch on a "compressed-looking" image means the
    // compression is per-subject anatomy, not pose.
    let effective_pitch = match head_pitch_cap {
        Some(head_pitch) => pitch_angle.min(head_pitch),
        None => pitch_angle,
    };
    let sin_pitch = effective_pitch.sin();
    let z_offset = sin_pitch * expected_y * pitch_sign;

    // Propagate the z into the spine-chain tip joints. UpperChest
    // and Neck share a position (shoulder mid); Head's source
    // entry (ear / nose midpoint) follows the spine so we scale
    // its offset by the head/chest height ratio. Clamp the ratio
    // to anatomical plausibility (1.0–2.0) so a head/chest with an
    // unexpected y compression can't blow up the z offset.
    let head_scale = sk
        .joints
        .get(&HumanoidBone::Head)
        .map(|h| ((h.position[1] - hip_y) / actual_y).clamp(1.0, 2.0))
        .unwrap_or(1.5);
    for (bone, scale) in [
        (HumanoidBone::UpperChest, 1.0),
        (HumanoidBone::Neck, 1.0),
        (HumanoidBone::Head, head_scale),
    ] {
        if let Some(j) = sk.joints.get_mut(&bone) {
            j.position[2] = z_offset * scale;
        }
    }
}

/// Recover a body yaw (rotation around vertical) signal from 2D
/// foreshortening of the shoulder / hip X-span. Counterpart to
/// [`infer_spine_pitch_from_foreshortening`]: same anatomy-ratio
/// approach, orthogonal axis. Independent inference for upper body
/// (shoulder pair drives shoulder + spine + arm chain) and lower
/// body (hip pair drives leg chain) lets a torso-twist pose come
/// through without forcing both halves to rotate together.
///
/// Implementation: detect yaw magnitude from x-span compression,
/// then **rotate the entire body half** by `yaw` around the
/// vertical axis through hip mid. This is structurally correct —
/// in a real twist, the elbow / wrist / knee / foot all rotate
/// with their parent (shoulder / hip). Per-pair single-joint z
/// injection (the obvious first attempt) leaves the chain children
/// in the 2D plane and produces an "arm pointing forward into the
/// torso" artefact when the body actually twisted.
///
/// Without a depth signal we cannot tell which way the subject is
/// turned (+yaw vs −yaw produce identical x-span compression).
/// Default to subject-right-toward-camera (= positive yaw per
/// [`super::super::super::avatar::pose_solver::compute_body_yaw_3d`]'s
/// convention) — performers facing the camera most often demo this
/// direction, and a wrong-direction case is angularly identical so
/// the rest-pose neutral keeps the bound tight.
fn infer_body_yaw_from_foreshortening(sk: &mut SourceSkeleton, yaw_sign: f32) {
    /// Anatomical ratio: shoulder vertical span (shoulder-to-hip y)
    /// divided by frontal shoulder x-span. 1.50 covers most adult
    /// subjects (true 1.15-1.20) without over-estimating yaw on
    /// slim / stylised builds where the shoulder/y ratio runs
    /// closer to 2.5. Earlier 1.15 produced 55°+ inferred yaw on
    /// gothic_twist where the actual yaw was ~35°.
    const SHOULDER_RATIO: f32 = 1.50;
    /// Hip x-span expected = `HIP_FROM_SHOULDER_RATIO × observed
    /// shoulder span`. Tying hip anchor to the same frame's
    /// observed shoulder span (rather than an independent anatomy
    /// ratio against torso_y) cancels per-subject build variance —
    /// a slim subject's narrow shoulders also have narrow hips, so
    /// the ratio between the two is more stable than either's
    /// ratio to torso_y. 0.80 is the population mean for adult hip/
    /// shoulder.
    const HIP_FROM_SHOULDER_RATIO: f32 = 0.80;
    /// Skip when any joint already carries a real depth signal.
    const MAX_EXISTING_Z_FOR_INFERENCE: f32 = 0.05;
    /// `compression ≥ this` is treated as upright; below is yawed.
    const COMPRESSION_THRESHOLD: f32 = 0.95;
    const MIN_TORSO_Y: f32 = 0.05;

    let Some(hips) = sk.joints.get(&HumanoidBone::Hips).copied() else {
        return;
    };
    let Some(chest) = sk.joints.get(&HumanoidBone::UpperChest).copied() else {
        return;
    };
    let torso_y = chest.position[1] - hips.position[1];
    if torso_y < MIN_TORSO_Y {
        return;
    }
    let pivot = hips.position;

    // Observed shoulder span: feeds both the shoulder-yaw expected
    // (anatomical fallback) and the hip-yaw expected (derived
    // ratio). Compute before rotating so the rotation doesn't
    // shrink it further.
    let observed_shoulder_x_span = match (
        sk.joints.get(&HumanoidBone::LeftShoulder),
        sk.joints.get(&HumanoidBone::RightShoulder),
    ) {
        (Some(l), Some(r)) => (l.position[0] - r.position[0]).abs(),
        _ => 0.0,
    };

    // Step 1: shoulder yaw. Gate the entire inference on the
    // shoulder pair clearing the compression threshold — shoulders
    // are the most reliable anatomical anchor, and hip-only yaw
    // signals are dominated by per-subject build variance (narrow-
    // hipped anime / idol photoshoots produced 41° false-positive
    // hip yaw on forward-lean poses where the shoulder pair clearly
    // showed no rotation). If shoulders say "frontal", hips do too.
    let shoulder_expected = torso_y / SHOULDER_RATIO;

    // 180° back-view bias: when L/R shoulder x positions are
    // swapped (L < R in source space, opposite of the frontal
    // anatomy), the body is yawed ~180°. The `quat_from_vectors`
    // shortest-arc rotation in `compute_hip_align_rotation` /
    // `compute_body_yaw_3d` becomes degenerate at exactly 180° —
    // any axis perpendicular to both vectors is equally valid, so
    // numerical noise picks an arbitrary one (typically Z), which
    // rotates the avatar around an unintended axis. Inject a tiny
    // +z bias into the L side and -z into R so the cross-product
    // rotation axis tilts toward Y (body yaw). Apply this BEFORE
    // the yaw-magnitude gate so 180° back-view cases (where shoulder
    // x-span is full so no yaw is detected) still get the bias.
    let back_view_for_bias = match (
        sk.joints.get(&HumanoidBone::LeftShoulder),
        sk.joints.get(&HumanoidBone::RightShoulder),
    ) {
        (Some(l), Some(r)) => l.position[0] < r.position[0],
        _ => false,
    };
    if back_view_for_bias && observed_shoulder_x_span > 0.05 {
        let bias = observed_shoulder_x_span * 0.03;
        let left_bones = [
            HumanoidBone::LeftShoulder,
            HumanoidBone::LeftUpperArm,
            HumanoidBone::LeftLowerArm,
            HumanoidBone::LeftHand,
            HumanoidBone::LeftUpperLeg,
            HumanoidBone::LeftLowerLeg,
            HumanoidBone::LeftFoot,
        ];
        let right_bones = [
            HumanoidBone::RightShoulder,
            HumanoidBone::RightUpperArm,
            HumanoidBone::RightLowerArm,
            HumanoidBone::RightHand,
            HumanoidBone::RightUpperLeg,
            HumanoidBone::RightLowerLeg,
            HumanoidBone::RightFoot,
        ];
        for bone in &left_bones {
            if let Some(j) = sk.joints.get_mut(bone) {
                j.position[2] += bias;
            }
        }
        for bone in &right_bones {
            if let Some(j) = sk.joints.get_mut(bone) {
                j.position[2] -= bias;
            }
        }
    }

    let shoulder_yaw = detect_yaw_magnitude(
        sk,
        HumanoidBone::LeftShoulder,
        HumanoidBone::RightShoulder,
        shoulder_expected,
        MAX_EXISTING_Z_FOR_INFERENCE,
        COMPRESSION_THRESHOLD,
    );
    let Some(shoulder_yaw) = shoulder_yaw else {
        return;
    };

    // Anatomical recovery for the 90°-profile collapse case. When
    // the shoulder pair projects to (nearly) the same image x —
    // i.e. the subject is in deep side profile and the L/R
    // shoulders overlap in 2D — the bone-label distinguishes
    // anatomical L from R while the pixel coordinate doesn't.
    // Separate the L/R chain joints by ±half_span in the source x
    // axis *before* applying the rotation: rotation around vertical
    // then projects them to opposite z, which is the correct 3D
    // configuration for the 90° profile. Without this recovery,
    // both shoulders stay at x = 0 and rotation produces no
    // separation in z (cos·0 + sin·0 = 0), so the avatar fails to
    // turn even though yaw was correctly inferred as 90°.
    const COLLAPSE_X_SPAN: f32 = 0.05;
    /// Subject-adaptive collapse threshold via shoulder/torso_y
    /// ratio. For an upright frontal adult, shoulder_span_x ≈
    /// 0.7-1.3 × torso_y. When the observed ratio drops below
    /// 0.50, the subject is in deep profile (~60°+ yaw) — apply
    /// the anatomical-separation recovery so the chain rotation
    /// produces a proper z separation. orient_001 walking case
    /// had ratio 0.45 (shoulder_span 0.135, torso 0.30) and was
    /// missing the recovery despite being near-90°. The absolute
    /// COLLAPSE_X_SPAN gate above stays for the literal 0-span
    /// case (90° + zero detection noise).
    const COLLAPSE_RATIO_THRESHOLD: f32 = 0.30;
    let upper_left_chain: &[HumanoidBone] = &[
        HumanoidBone::LeftShoulder,
        HumanoidBone::LeftUpperArm,
        HumanoidBone::LeftLowerArm,
        HumanoidBone::LeftHand,
    ];
    let upper_right_chain: &[HumanoidBone] = &[
        HumanoidBone::RightShoulder,
        HumanoidBone::RightUpperArm,
        HumanoidBone::RightLowerArm,
        HumanoidBone::RightHand,
    ];
    // Soft collapse recovery: instead of binary on/off, scale the
    // anatomical separation by how close the observed shoulder span
    // is to the expected. compression < 0.5 → progressive recovery;
    // ≥ 0.5 → no recovery (frontal-enough that 2D detection alone
    // suffices). Tested against orient_001 walking 90° profile
    // (compression 0.32) where binary firing helped but the
    // resulting rotation overshoots; soft scaling produces an
    // intermediate amount that matches the actual yaw better.
    let upper_compression = observed_shoulder_x_span / shoulder_expected.max(1e-3);
    if upper_compression < 1.0 || observed_shoulder_x_span < COLLAPSE_X_SPAN {
        let recovery_strength = if observed_shoulder_x_span < COLLAPSE_X_SPAN {
            1.0  // total collapse — full anatomical recovery
        } else {
            // Quadratic ramp: 1.0 at compression 0 → 0.0 at 1.0,
            // emphasising the recovery for deep-profile cases
            // while leaving near-frontal cases lightly touched.
            let r = (1.0 - upper_compression).clamp(0.0, 1.0);
            r * r
        };
        anatomically_separate(
            sk,
            upper_left_chain,
            upper_right_chain,
            shoulder_expected * 0.5 * recovery_strength,
        );
    }

    // Upper body chain (shoulders → arms → spine chain tips).
    let upper_chain: &[HumanoidBone] = &[
        HumanoidBone::UpperChest,
        HumanoidBone::Neck,
        HumanoidBone::Head,
        HumanoidBone::LeftShoulder,
        HumanoidBone::RightShoulder,
        HumanoidBone::LeftUpperArm,
        HumanoidBone::RightUpperArm,
        HumanoidBone::LeftLowerArm,
        HumanoidBone::RightLowerArm,
        HumanoidBone::LeftHand,
        HumanoidBone::RightHand,
    ];
    rotate_chain_by_yaw_magnitude(sk, upper_chain, shoulder_yaw * yaw_sign, pivot);

    // Step 2: hip yaw, gated on shoulder yaw firing. Use observed
    // shoulder x-span (× anatomical ratio 0.80) as the hip anchor
    // — ties hip yaw to the same frame's body width and cancels
    // per-subject build variance. Additionally require the inferred
    // hip yaw magnitude to be within a 2× band of the shoulder yaw:
    // a real body yaw rotates both pairs by similar amounts (full
    // rotation) or hip-yaw slightly less (torso twist), but hip-only
    // rotations beyond shoulder yaw are anatomically implausible
    // and almost always anatomy / detection artefacts.
    let hip_yaw = detect_yaw_magnitude(
        sk,
        HumanoidBone::LeftUpperLeg,
        HumanoidBone::RightUpperLeg,
        observed_shoulder_x_span * HIP_FROM_SHOULDER_RATIO,
        MAX_EXISTING_Z_FOR_INFERENCE,
        COMPRESSION_THRESHOLD,
    );
    let lower_chain: &[HumanoidBone] = &[
        HumanoidBone::LeftUpperLeg,
        HumanoidBone::RightUpperLeg,
        HumanoidBone::LeftLowerLeg,
        HumanoidBone::RightLowerLeg,
        HumanoidBone::LeftFoot,
        HumanoidBone::RightFoot,
    ];
    // Same anatomical recovery for the leg chain when the hip pair
    // has collapsed in 2D.
    let observed_hip_x_span = match (
        sk.joints.get(&HumanoidBone::LeftUpperLeg),
        sk.joints.get(&HumanoidBone::RightUpperLeg),
    ) {
        (Some(l), Some(r)) => (l.position[0] - r.position[0]).abs(),
        _ => 0.0,
    };
    let lower_left_chain: &[HumanoidBone] = &[
        HumanoidBone::LeftUpperLeg,
        HumanoidBone::LeftLowerLeg,
        HumanoidBone::LeftFoot,
    ];
    let lower_right_chain: &[HumanoidBone] = &[
        HumanoidBone::RightUpperLeg,
        HumanoidBone::RightLowerLeg,
        HumanoidBone::RightFoot,
    ];
    // Hip expected: observed shoulder span × 0.80 (subject-anchor).
    // When the shoulder pair has fully collapsed in 2D (observed
    // shoulder x-span < COLLAPSE_X_SPAN, i.e. 90° profile) the
    // subject-anchor is useless (= 0) so we fall back to an
    // anatomical estimate from torso_y. Outside the collapse case
    // we stick with the subject-anchor to avoid perturbing
    // shoulders_003-style non-rotated poses where the anatomical
    // fallback would over-estimate hip_expected and fire hip_yaw
    // when it shouldn't.
    let hip_expected_span = if observed_shoulder_x_span < COLLAPSE_X_SPAN {
        torso_y * 0.40
    } else {
        observed_shoulder_x_span * HIP_FROM_SHOULDER_RATIO
    };
    let lower_compression = observed_hip_x_span / hip_expected_span.max(1e-3);
    if lower_compression < 1.0 || observed_hip_x_span < COLLAPSE_X_SPAN {
        let recovery_strength = if observed_hip_x_span < COLLAPSE_X_SPAN {
            1.0
        } else {
            let r = (1.0 - lower_compression).clamp(0.0, 1.0);
            r * r
        };
        let half_span = hip_expected_span * 0.5 * recovery_strength;
        anatomically_separate(
            sk,
            lower_left_chain,
            lower_right_chain,
            half_span,
        );
    }
    if let Some(hip_yaw_raw) = hip_yaw {
        // Clamp hip yaw to at most the shoulder yaw — torso twist
        // never has hips yawed *more* than shoulders.
        let hip_yaw_clamped = hip_yaw_raw.min(shoulder_yaw.abs());
        rotate_chain_by_yaw_magnitude(sk, lower_chain, hip_yaw_clamped * yaw_sign, pivot);
    } else {
        // Hip span suggested frontal — but shoulders are yawed. Most
        // likely this is a full body yaw (not torso twist) and the
        // hip detection noise is hiding it. Apply the shoulder yaw
        // to the legs too so the full body rotates coherently.
        rotate_chain_by_yaw_magnitude(sk, lower_chain, shoulder_yaw * yaw_sign, pivot);
    }
}

/// Disambiguate the yaw direction (+ or -) from the raw COCO
/// keypoints. When the body yaws, the face yaws with it, and the
/// nose's image-x shifts relative to the ear-midpoint in the
/// direction whose side of the face becomes exposed to the camera.
///
/// Convention: positive yaw = subject's right side toward camera
/// (per `compute_body_yaw_3d`). After selfie mirror and `to_source`,
/// subject's right is image-LEFT (small nx). For +yaw, the face
/// has rotated such that the nose moves toward subject's right →
/// nose.nx is SMALLER than ear_mid.nx.
///
/// Returns +1 (default) or -1. When nose/ear keypoints fail
/// visibility, returns +1 — preserving the "performer turning
/// toward camera" default.
/// Combined sign + magnitude scalar for spine pitch inference,
/// derived from the head pose. Returns a signed value in `[-1, 1]`:
/// 0 = head essentially neutral (skip pitch), ±0.x = mild chin-
/// down / chin-up (scale pitch down), ±1 = strong chin angle
/// (full pitch). Without a confirmed head pitch direction, raw
/// shoulder-hip foreshortening is ambiguous between real body
/// pitch and per-subject anatomy variance — letting the head
/// magnitude scale the inferred z_offset cancels both the over-
/// estimate on slim-torsoed subjects and the under-estimate on
/// wide-shouldered ones.
/// Estimate the head pitch angle in radians from the COCO 0/3/4
/// keypoints (nose, left/right ear). Returns `None` when the
/// face triangle is unavailable. Used as a cap on the spine
/// pitch inference.
fn estimate_head_pitch_angle(joints: &[DecodedJoint]) -> Option<f32> {
    const NOSE_IDX: usize = 0;
    const LEFT_EAR_IDX: usize = 3;
    const RIGHT_EAR_IDX: usize = 4;
    if joints.len() <= RIGHT_EAR_IDX {
        return None;
    }
    let nose = &joints[NOSE_IDX];
    let l_ear = &joints[LEFT_EAR_IDX];
    let r_ear = &joints[RIGHT_EAR_IDX];
    let visibility = super::consts::KEYPOINT_VISIBILITY_FLOOR;
    if nose.score < visibility
        || l_ear.score < visibility
        || r_ear.score < visibility
    {
        return None;
    }
    let ear_span = (l_ear.nx - r_ear.nx)
        .hypot(l_ear.ny - r_ear.ny)
        .max(1.0e-4);
    let ear_mid_ny = (l_ear.ny + r_ear.ny) * 0.5;
    let nose_dy = nose.ny - ear_mid_ny;
    // Normalised dy / ear_span ≈ tan(pitch_angle). Atan recovers
    // a face-pitch-equivalent angle in radians.
    Some((nose_dy.abs() / ear_span).atan())
}

fn detect_pitch_scale(joints: &[DecodedJoint]) -> Option<f32> {
    const NOSE_IDX: usize = 0;
    const LEFT_EAR_IDX: usize = 3;
    const RIGHT_EAR_IDX: usize = 4;
    /// Below this nose-vs-ear-midpoint offset (as a fraction of
    /// face width) the head is treated as neutral. Roughly
    /// corresponds to ±10° head pitch.
    const MIN_HEAD_PITCH_RATIO: f32 = 0.10;
    /// At and beyond this ratio, the chin is clearly down /
    /// up — corresponds to ±30° head pitch. Saturates the scale
    /// at 1 so a deep bow still injects the full inferred pitch.
    /// Lower threshold (vs 0.50) keeps deep bows at full strength
    /// while mild head tilts get a proportional reduction —
    /// tested against balance_001 (mild) vs depth_006 (deep).
    const MAX_HEAD_PITCH_RATIO: f32 = 0.30;

    if joints.len() <= RIGHT_EAR_IDX {
        return None;
    }
    let nose = &joints[NOSE_IDX];
    let l_ear = &joints[LEFT_EAR_IDX];
    let r_ear = &joints[RIGHT_EAR_IDX];
    let visibility = super::consts::KEYPOINT_VISIBILITY_FLOOR;
    if nose.score < visibility
        || l_ear.score < visibility
        || r_ear.score < visibility
    {
        return None;
    }
    // Normalise the nose-vs-ear dy by the ear span — gives a
    // resolution- and subject-size-independent head pitch ratio.
    let ear_span = (l_ear.nx - r_ear.nx)
        .hypot(l_ear.ny - r_ear.ny)
        .max(1.0e-4);
    let ear_mid_ny = (l_ear.ny + r_ear.ny) * 0.5;
    let nose_dy_ratio = (nose.ny - ear_mid_ny) / ear_span;
    if nose_dy_ratio.abs() < MIN_HEAD_PITCH_RATIO {
        return None;
    }
    // Linear scale between MIN..MAX, clamped at ±1.
    let signed = nose_dy_ratio.clamp(-MAX_HEAD_PITCH_RATIO, MAX_HEAD_PITCH_RATIO)
        / MAX_HEAD_PITCH_RATIO;
    Some(signed)
}

fn detect_yaw_sign(joints: &[DecodedJoint]) -> f32 {
    const NOSE_IDX: usize = 0;
    const LEFT_EAR_IDX: usize = 3;
    const RIGHT_EAR_IDX: usize = 4;
    const LEFT_SHOULDER_IDX: usize = 5;
    const RIGHT_SHOULDER_IDX: usize = 6;
    /// Below this normalised-image-x offset the nose-vs-ears signal
    /// is dominated by detection noise. Drop to shoulder-asymmetry
    /// fallback below this threshold.
    const MIN_NX_OFFSET_FOR_SIGN: f32 = 0.01;
    /// Shoulder confidence asymmetry threshold for the fallback —
    /// the front shoulder (toward camera) detects with high score
    /// in deep profile, the back one drops well below. 0.2 was
    /// chosen empirically: frontal poses sit at ~0.85 / ~0.85, deep
    /// profile sits at ~0.85 / ~0.45.
    const MIN_SHOULDER_CONF_ASYM: f32 = 0.20;

    if joints.len() <= RIGHT_SHOULDER_IDX {
        return 1.0;
    }
    let nose = &joints[NOSE_IDX];
    let l_ear = &joints[LEFT_EAR_IDX];
    let r_ear = &joints[RIGHT_EAR_IDX];
    let visibility = super::consts::KEYPOINT_VISIBILITY_FLOOR;

    // Primary signal: nose x position relative to ear midpoint.
    // Most reliable for 3/4 / mild profile poses where both ears
    // are still visible.
    if nose.score >= visibility && l_ear.score >= visibility && r_ear.score >= visibility {
        let ear_mid_nx = (l_ear.nx + r_ear.nx) * 0.5;
        let nose_offset = nose.nx - ear_mid_nx;
        if nose_offset.abs() >= MIN_NX_OFFSET_FOR_SIGN {
            // Nose shifted toward image-LEFT (smaller nx) =
            // subject's right side exposed = +yaw.
            return if nose_offset < 0.0 { 1.0 } else { -1.0 };
        }
    }

    // Fallback signal: shoulder pair confidence asymmetry. At
    // deep profile (~90°) the back shoulder is occluded and its
    // detection confidence drops sharply; the front shoulder
    // stays high. COCO 5 = subject's left shoulder, COCO 6 =
    // subject's right shoulder.
    let l_sh = &joints[LEFT_SHOULDER_IDX];
    let r_sh = &joints[RIGHT_SHOULDER_IDX];
    if l_sh.score >= visibility && r_sh.score >= visibility {
        let asym = r_sh.score - l_sh.score;
        if asym.abs() >= MIN_SHOULDER_CONF_ASYM {
            // R shoulder more visible (asym > 0) = subject's right
            // side toward camera = +yaw.
            return if asym > 0.0 { 1.0 } else { -1.0 };
        }
    }

    // No reliable signal — default to + (subject's right toward
    // camera, statistically the more common performer-facing
    // direction).
    1.0
}

/// Shift the L/R sides of a collapsed anatomical pair apart by
/// `half_span` along source x. Recovers the body-frame L/R
/// distinction lost in 90°-profile 2D projection where both
/// shoulders (or both hips) project to the same pixel — the bone
/// label still says "this is the left", so we move it back to the
/// L side of the body. Downstream rotation then maps that body-
/// frame x to its proper image-frame (x, z).
fn anatomically_separate(
    sk: &mut SourceSkeleton,
    left_chain: &[HumanoidBone],
    right_chain: &[HumanoidBone],
    half_span: f32,
) {
    if half_span < 0.025 {
        return;
    }
    for bone in left_chain {
        if let Some(j) = sk.joints.get_mut(bone) {
            j.position[0] += half_span;
        }
    }
    for bone in right_chain {
        if let Some(j) = sk.joints.get_mut(bone) {
            j.position[0] -= half_span;
        }
    }
}

/// Detect yaw magnitude (always positive) from x-span compression.
/// Returns `None` when the pair is degenerate, depth-occupied, or
/// not foreshortened.
fn detect_yaw_magnitude(
    sk: &SourceSkeleton,
    left_bone: HumanoidBone,
    right_bone: HumanoidBone,
    expected_x_span: f32,
    max_existing_z: f32,
    compression_threshold: f32,
) -> Option<f32> {
    let lj = sk.joints.get(&left_bone)?;
    let rj = sk.joints.get(&right_bone)?;
    if lj.position[2].abs() > max_existing_z || rj.position[2].abs() > max_existing_z {
        return None;
    }
    if expected_x_span < 0.05 {
        return None;
    }
    let x_span = (lj.position[0] - rj.position[0]).abs();
    let compression = (x_span / expected_x_span).clamp(0.0, 1.0);
    if compression >= compression_threshold {
        return None;
    }
    Some(compression.acos())
}

/// Rotate every joint in `chain` by `yaw_magnitude` radians around
/// the vertical axis through `pivot`. Sign convention: positive yaw
/// rotates the avatar's `x > 0` side (= subject's right via selfie-
/// mirror) toward +z (camera). See [`infer_body_yaw_from_foreshortening`]
/// for why this is the default.
fn rotate_chain_by_yaw_magnitude(
    sk: &mut SourceSkeleton,
    chain: &[HumanoidBone],
    yaw_magnitude: f32,
    pivot: [f32; 3],
) {
    if yaw_magnitude.abs() < 1e-3 {
        return;
    }
    let cos_yaw = yaw_magnitude.cos();
    let sin_yaw = yaw_magnitude.sin();
    for bone in chain {
        if let Some(j) = sk.joints.get_mut(bone) {
            let lx = j.position[0] - pivot[0];
            let lz = j.position[2] - pivot[2];
            let nx = lx * cos_yaw + lz * sin_yaw;
            let nz = -lx * sin_yaw + lz * cos_yaw;
            j.position[0] = pivot[0] + nx;
            j.position[2] = pivot[2] + nz;
        }
    }
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
            metric_depth_m: None,
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
                metric_depth_m: None,
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
                metric_depth_m: None,
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

/// Inserts the spine-chain proxy joints + the Head proxy that
/// `pose_solver` needs in order to drive Chest, UpperChest, Neck and
/// (indirectly) the Head bone's pivot.
///
/// * `Chest` is dual-inserted at the hip-mid origin (same convention
///   as `Spine`). Its driving signal is `Tip::ShoulderMidpoint`, so
///   the source-side base lookup just needs *some* entry — FK rebases
///   the rest direction so Chest's actual rotation lands near identity
///   once Spine has absorbed the lower-spine bend.
/// * `UpperChest` and `Neck` are dual-inserted at the L/R-shoulder
///   midpoint. Their driving signal (`Tip::HeadFromShoulders`)
///   overrides the source-side base to the same midpoint, but the
///   solver still requires the entry-with-confidence check to clear
///   before it'll touch the bone.
/// * `Head` is the actual chin-thrust signal: ear midpoint when both
///   ears clear the visibility floor, with a nose fallback for
///   three-quarter / profile shots where one ear is occluded. Ear
///   midpoint is preferred because it sits at roughly the head's
///   pivot (the atlanto-occipital joint), whereas the nose tip leads
///   it by ~5 cm and would over-report forward motion.
fn inject_spine_chain_proxies<F>(
    sk: &mut SourceSkeleton,
    joints: &[DecodedJoint],
    to_source: &F,
    hip_visible: bool,
    hip_score: f32,
) where
    F: Fn(&DecodedJoint) -> [f32; 3],
{
    if hip_visible {
        sk.joints.insert(
            HumanoidBone::Chest,
            SourceJoint {
                position: [0.0, 0.0, 0.0],
                confidence: hip_score,
                metric_depth_m: None,
            },
        );
    }

    if let (Some(l), Some(r)) = (
        sk.joints.get(&HumanoidBone::LeftShoulder).copied(),
        sk.joints.get(&HumanoidBone::RightShoulder).copied(),
    ) {
        let mid = SourceJoint {
            position: [
                (l.position[0] + r.position[0]) * 0.5,
                (l.position[1] + r.position[1]) * 0.5,
                (l.position[2] + r.position[2]) * 0.5,
            ],
            confidence: l.confidence.min(r.confidence),
            metric_depth_m: None,
        };
        sk.joints.insert(HumanoidBone::UpperChest, mid);
        sk.joints.insert(HumanoidBone::Neck, mid);
    }

    const NOSE_IDX: usize = 0;
    const LEFT_EAR_IDX: usize = 3;
    const RIGHT_EAR_IDX: usize = 4;

    let head = match (joints.get(LEFT_EAR_IDX), joints.get(RIGHT_EAR_IDX)) {
        (Some(l), Some(r))
            if l.score >= KEYPOINT_VISIBILITY_FLOOR && r.score >= KEYPOINT_VISIBILITY_FLOOR =>
        {
            let lp = to_source(l);
            let rp = to_source(r);
            Some(SourceJoint {
                position: [
                    (lp[0] + rp[0]) * 0.5,
                    (lp[1] + rp[1]) * 0.5,
                    (lp[2] + rp[2]) * 0.5,
                ],
                confidence: l.score.min(r.score),
                metric_depth_m: None,
            })
        }
        _ => joints.get(NOSE_IDX).and_then(|n| {
            (n.score >= KEYPOINT_VISIBILITY_FLOOR).then(|| SourceJoint {
                position: to_source(n),
                confidence: n.score,
                metric_depth_m: None,
            })
        }),
    };

    if let Some(h) = head {
        sk.joints.insert(HumanoidBone::Head, h);
    }
}

#[cfg(test)]
mod tests {
    use super::super::decode::{DecodedJoint, NUM_JOINTS};
    use super::*;

    /// 133 evenly-confident COCO-Wholebody joints arranged so the hip
    /// pair (idx 11/12) is plausible — the regression we're guarding
    /// against is the *upstream* keypoint head emitting confident
    /// hips even when the user's pelvis is below the camera frame.
    fn fake_joints_with_visible_hips() -> Vec<DecodedJoint> {
        let mut joints = vec![DecodedJoint::default(); NUM_JOINTS];
        // Shoulders (5/6) — used as the fallback anchor.
        joints[5] = DecodedJoint {
            nx: 0.40,
            ny: 0.30,
            nz: 0.5,
            score: 0.9,
        };
        joints[6] = DecodedJoint {
            nx: 0.60,
            ny: 0.30,
            nz: 0.5,
            score: 0.9,
        };
        // Hips (11/12) — the hallucinated pair we want to suppress.
        joints[11] = DecodedJoint {
            nx: 0.42,
            ny: 0.70,
            nz: 0.5,
            score: 0.8,
        };
        joints[12] = DecodedJoint {
            nx: 0.58,
            ny: 0.70,
            nz: 0.5,
            score: 0.8,
        };
        // Knees (13/14) and ankles (15/16) — same hallucination
        // pattern; downstream consumers must not see any of these.
        joints[13] = DecodedJoint {
            nx: 0.42,
            ny: 0.85,
            nz: 0.5,
            score: 0.7,
        };
        joints[14] = DecodedJoint {
            nx: 0.58,
            ny: 0.85,
            nz: 0.5,
            score: 0.7,
        };
        joints[15] = DecodedJoint {
            nx: 0.42,
            ny: 0.95,
            nz: 0.5,
            score: 0.7,
        };
        joints[16] = DecodedJoint {
            nx: 0.58,
            ny: 0.95,
            nz: 0.5,
            score: 0.7,
        };
        joints
    }

    #[test]
    fn upper_body_mode_drops_lower_body_even_when_hips_are_high_confidence() {
        let joints = fake_joints_with_visible_hips();
        let sk = build_source_skeleton(0, &joints, 640, 480, true);
        for bone in [
            HumanoidBone::Hips,
            HumanoidBone::LeftUpperLeg,
            HumanoidBone::RightUpperLeg,
            HumanoidBone::LeftLowerLeg,
            HumanoidBone::RightLowerLeg,
            HumanoidBone::LeftFoot,
            HumanoidBone::RightFoot,
        ] {
            assert!(
                !sk.joints.contains_key(&bone),
                "UpperBody mode must suppress {:?} regardless of keypoint confidence",
                bone
            );
        }
        assert!(
            !sk.root_anchor_is_hip,
            "root anchor must come from shoulders, not the suppressed hip pair"
        );
    }

    #[test]
    fn full_body_mode_emits_hips_when_visible() {
        // Sanity counterpart: with the suppression flag off, the
        // existing hip-driven path still works — confirms the guard
        // is mode-gated, not a regression of the FullBody contract.
        let joints = fake_joints_with_visible_hips();
        let sk = build_source_skeleton(0, &joints, 640, 480, false);
        assert!(sk.joints.contains_key(&HumanoidBone::Hips));
        assert!(sk.root_anchor_is_hip);
    }
}
