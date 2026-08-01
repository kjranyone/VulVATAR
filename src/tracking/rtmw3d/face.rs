//! Head pose (yaw / pitch / roll) derivation from RTMW3D's body face
//! keypoints, plus the dlib-68 face bbox we hand to the FaceMesh
//! cascade. Everything in this file consumes [`super::decode::DecodedJoint`]
//! and produces source-space outputs (`FacePose`, `FaceBbox`) — no
//! ONNX inference of its own.

use crate::asset::HumanoidBone;

use super::super::face_mediapipe::{derive_face_bbox, FaceBbox};
use super::super::{FacePose, FaceSource, SourceSkeleton};
use super::consts::{INPUT_H, INPUT_W, KEYPOINT_VISIBILITY_FLOOR};
use super::decode::DecodedJoint;

/// Derive head yaw / pitch / roll from RTMW3D's body face keypoints
/// (0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear) in
/// source-skeleton 3D coords. Returns `None` if any keypoint is below
/// the confidence floor or the eye/ear spans are too small to give a
/// reliable angle.
///
/// All values are already mirrored at the source level (subject's
/// left ear ends up at source `+x`), so the sign conventions match
/// `quat_from_euler_ypr`.
pub(super) fn derive_face_pose_from_body(
    skeleton: &SourceSkeleton,
    joints: &[DecodedJoint],
) -> Option<FacePose> {
    if joints.len() < 5 {
        return None;
    }
    // Face pose math (ear-line yaw, eye-line roll, eye-to-nose pitch)
    // is well-defined as long as the keypoints actually exist and the
    // ear and eye baselines are non-degenerate (checked further down).
    // The aggregated confidence is the min over the 5 inputs, so the
    // solver can apply its own face threshold downstream — that's why
    // we only reject blatant non-detections here, not borderline ones.
    for j in joints.iter().take(5) {
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            return None;
        }
    }
    // Observability of the two HORIZONTAL baselines, in the same fixed
    // pixel space the roll/pitch math uses. In an oblique / profile view
    // both ears (and both eyes) project nearly onto the same image
    // point; their z separation is SimCC-hallucinated, so
    // `yaw = atan2(dz, dx≈0)` and `roll = atan2(dy, dx≈0)` amplify pure
    // noise into whole-hemisphere thrash (live desk capture 2026-07-27:
    // ear separation ≈ 3% of the head's keypoint spread → published yaw
    // ranged −154°…+97° with sd 66° while the subject held still — the
    // avatar's "broken neck"). A collapsed baseline is UNOBSERVABLE,
    // not low-confidence (the scores stayed ~0.65): refuse the pose so
    // the selector holds the previous / mesh pose or the head eases to
    // neutral, instead of publishing an unbounded division.
    {
        let px = |j: &DecodedJoint| (j.nx * INPUT_W as f32, j.ny * INPUT_H as f32);
        let dist = |a: (f32, f32), b: (f32, f32)| (a.0 - b.0).hypot(a.1 - b.1);
        let pts = [px(&joints[0]), px(&joints[1]), px(&joints[2]), px(&joints[3]), px(&joints[4])];
        let mut head_size = 0.0_f32;
        for i in 0..pts.len() {
            for k in i + 1..pts.len() {
                head_size = head_size.max(dist(pts[i], pts[k]));
            }
        }
        /// Ear baseline below this fraction of the head's keypoint
        /// spread → the ear-line yaw divides by ~nothing. Frontal sits
        /// at ~1.0, a 60° turn at ~0.6; the live broken-neck frames at
        /// 0.03.
        const MIN_EAR_SEP_FRAC: f32 = 0.15;
        /// Same for the eye baseline (roll). Frontal ~0.5.
        const MIN_EYE_SEP_FRAC: f32 = 0.10;
        if head_size < 1.0
            || dist(pts[3], pts[4]) < MIN_EAR_SEP_FRAC * head_size
            || dist(pts[1], pts[2]) < MIN_EYE_SEP_FRAC * head_size
        {
            return None;
        }
    }
    // We need positions in source-skeleton coords. The COCO body
    // keypoints 0..=4 are not stored in `skeleton.joints` (only
    // shoulders / arms / legs are), so reconstruct via the same
    // `to_source` math. We extract hip mid from the stored Hips
    // joint (which is `(0, 0, 0)` by definition of the origin),
    // then read the 5 face keypoints in normalised space and run
    // them through the same axis flip used elsewhere.
    let hip = (joints[11].nx + joints[12].nx) * 0.5;
    let aspect = skeleton
        .joints
        .get(&HumanoidBone::LeftShoulder)
        .or_else(|| skeleton.joints.get(&HumanoidBone::RightShoulder))
        .map(|s| s.position[0].abs())
        .unwrap_or(0.0)
        / ((joints[5].nx - joints[6].nx).abs() / 2.0).max(1e-3);
    // Conservative aspect estimate: if we can't read it from the
    // stored shoulders fall back to assuming square (1.0).
    let aspect = if aspect.is_finite() && aspect > 0.5 && aspect < 4.0 {
        aspect
    } else {
        1.0
    };
    let to_src = |j: &DecodedJoint| -> [f32; 3] {
        let sx = -(j.nx - hip) * aspect;
        let sy = -(j.ny - (joints[11].ny + joints[12].ny) * 0.5);
        let sz = -(j.nz - (joints[11].nz + joints[12].nz) * 0.5);
        [sx, sy, sz]
    };
    // Subject-POV indices map to avatar-POV after `to_src` (selfie
    // mirror): subject's left → avatar's right and vice versa. Bind
    // the avatar-frame names directly so the geometry below reads
    // without sign confusion.
    let nose = to_src(&joints[0]);
    // Eyes are consumed in model-pixel space (roll / pitch below) so
    // they never go through `to_src` — only the ear line (yaw) and the
    // nose (back-ear reconstruction) need source coords.
    let avatar_right_ear = to_src(&joints[3]); // subject left ear
    let avatar_left_ear = to_src(&joints[4]); // subject right ear

    // Ear line points avatar_left_ear → avatar_right_ear along +x at
    // rest, so dx = left.x - right.x is positive when the head faces
    // camera. As the head turns to camera-right (avatar yaw +) the
    // line tilts in XZ so dz becomes positive too. yaw = atan2(dz, dx).
    //
    // Back-ear robustness: when the head turns past ~45° one ear
    // becomes occluded and RTMW3D's SimCC head will still emit a
    // 3D position for it, but the Z component is essentially
    // hallucinated — typically pinned near the visible ear's depth,
    // collapsing `ear_dz` toward zero and stalling the avatar's yaw
    // tracking right when it should be most active. Detect that
    // case by score asymmetry and reconstruct the back ear as the
    // visible ear reflected through the nose's XZ position. Nose
    // is forward of the head's actual rotation axis, so the
    // reflection over-rotates the reconstructed ear and `ear_dz`
    // carries a systematic yaw-magnitude bias in these views. Note
    // this bias is NOT absorbed by the per-session calibration: the
    // reconstruction only fires on occluded-ear (turned-head) frames,
    // and the calibration neutral is captured during a *frontal* hold
    // where this branch never runs. The bias is accepted as-is —
    // it overstates how far the head is turned but preserves turn
    // direction and motion continuity, which is what the avatar's
    // head bone visibly needs in profile views.
    const EAR_OCCLUSION_RATIO: f32 = 0.5;
    let s_right_ear = joints[3].score;
    let s_left_ear = joints[4].score;
    // Whether the ear line is evidence a *pitch* estimator may lean on.
    // Both ears must be genuinely seen (no reconstruction) — a
    // reconstructed ear carries the documented yaw-magnitude bias, and a
    // yaw that is merely plausible is not good enough to divide a
    // horizontal baseline by. See `legacy_pitch_from_inter_eye`.
    let ears_trustworthy = s_right_ear >= s_left_ear * EAR_OCCLUSION_RATIO
        && s_left_ear >= s_right_ear * EAR_OCCLUSION_RATIO;
    let (right_ear_xz, left_ear_xz) = if s_right_ear < s_left_ear * EAR_OCCLUSION_RATIO {
        // Subject's left ear (joint 3 / avatar_right_ear) occluded.
        let recon = [
            2.0 * nose[0] - avatar_left_ear[0],
            avatar_left_ear[1],
            2.0 * nose[2] - avatar_left_ear[2],
        ];
        (recon, avatar_left_ear)
    } else if s_left_ear < s_right_ear * EAR_OCCLUSION_RATIO {
        // Subject's right ear (joint 4 / avatar_left_ear) occluded.
        let recon = [
            2.0 * nose[0] - avatar_right_ear[0],
            avatar_right_ear[1],
            2.0 * nose[2] - avatar_right_ear[2],
        ];
        (avatar_right_ear, recon)
    } else {
        (avatar_right_ear, avatar_left_ear)
    };
    let ear_dx = left_ear_xz[0] - right_ear_xz[0];
    let ear_dz = left_ear_xz[2] - right_ear_xz[2];
    let ear_dist = (ear_dx * ear_dx + ear_dz * ear_dz).sqrt();
    if ear_dist < 0.01 {
        return None;
    }
    let yaw = ear_dz.atan2(ear_dx);

    // Eye line: same handedness. Roll positive = head tilts toward
    // avatar's left ear (avatar's left eye drops below the right).
    //
    // Computed in model-pixel space (like pitch below) rather than
    // source space: the source-space x carries the shoulder-derived
    // `aspect` estimate, whose per-frame noise would inject directly
    // into the roll angle as jitter, and whose 0.5–4.0 gate makes the
    // roll gain framing-dependent. Pixel space is the geometry the
    // model actually saw (the crop is letterboxed, not squashed), so
    // the angle is anatomically correct and aspect-free. Sign check:
    // avatar-left eye = subject-right eye = joints[2]; source
    // eye_dy = y_src(left) − y_src(right) = j1.ny − j2.ny (y_src
    // negates image-y), and source eye_dx = aspect·(j1.nx − j2.nx) —
    // both preserved below with the fixed pixel scale instead of
    // `aspect`.
    let roll_dy_px = (joints[1].ny - joints[2].ny) * INPUT_H as f32;
    let roll_dx_px = (joints[1].nx - joints[2].nx) * INPUT_W as f32;
    let roll = roll_dy_px.atan2(roll_dx_px);

    // PITCH from the face's own VERTICAL landmark geometry — see
    // `pitch_from_vertical_ratio`. Falls back to the legacy inter-eye
    // normalisation only while the ear line can vouch for the yaw, and
    // refuses to publish a pose at all when neither holds: a pitch derived
    // from a horizontal baseline with an unknown yaw is not a weak reading,
    // it is a wrong one (live capture decoded a 60 deg head TURN as a 50 deg
    // chin-DOWN nod).
    let pitch = match pitch_from_vertical_ratio(joints) {
        Some(p) => p,
        None => legacy_pitch_from_inter_eye(joints, yaw, ears_trustworthy)?,
    };

    let conf = joints[..5].iter().map(|j| j.score).fold(1.0_f32, f32::min);
    Some(FacePose {
        yaw,
        pitch,
        roll,
        confidence: conf,
        source: FaceSource::Body,
        ..Default::default()
    })
}

/// dlib-68 face landmarks start at this COCO-WholeBody index (0..=4 are the
/// body-face keypoints, 5..=22 the body, 23..=90 the face-68 set — the same
/// block [`build_face_bbox_from_joints`] consumes).
const FACE68_BASE: usize = 23;
/// dlib-68 indices used by the pitch estimator.
const DLIB_CHIN: usize = 8;
const DLIB_NOSE_TIP: usize = 30;
const DLIB_EYE_CORNERS: [usize; 4] = [36, 39, 42, 45];

/// Frontal-neutral value of the vertical ratio below, i.e. how far down the
/// eye-line→chin span the nose tip sits on a level head. Measured 0.26 over
/// the live desk capture's level-gaze frames (2026-07-27); the per-person
/// residual is absorbed by `PoseCalibration::neutral_face_ypr_body`.
const VERTICAL_RATIO_NEUTRAL: f32 = 0.26;
/// Converts the ratio's deviation from neutral into `tan(pitch)`. The ratio
/// moves as `a/b + (d/b)·tan(pitch)` where `b` is the eye-line→chin distance
/// and `d` the nose tip's protrusion in front of that plane, so the gain is
/// `b/d`. Live check: a 30 deg chin-down frame read 0.465 against a 0.26
/// neutral → 2.81. Anthropometry agrees (b ≈ 0.11 m, d ≈ 0.039 m → 2.8).
const VERTICAL_RATIO_GAIN: f32 = 2.8;
/// Minimum eye-line→chin separation (normalised image units) for the ratio to
/// be meaningful. Below this the face is a few pixels tall or the landmarks
/// have collapsed, and the division amplifies noise without bound.
const MIN_EYE_CHIN_SPAN: f32 = 1e-3;

/// Head pitch from landmarks that share a vertical line: the nose tip's
/// height between the eye line and the chin.
///
/// Yaw-invariance is the whole point. A head turn is a rotation about the
/// vertical axis, which leaves every landmark's image `y` untouched, so a
/// ratio built only from `y` cannot move when the subject looks sideways.
/// The shipping formula normalised the nose's drop by the INTER-EYE
/// distance, which foreshortens by `cos(yaw)`; it compensated with a
/// `cos(yaw)` term read off the ear line — and live capture (2026-07-27)
/// showed that ear-line yaw pinned inside ±5 deg through a real 45-70 deg
/// turn (hair and glasses temples hide the ears, so `ear_dz` never
/// separates). With the compensation dead, the raw 1/cos inflation went
/// straight to the avatar: measured body pitch +29 deg at a 45 deg turn and
/// +50 deg at 60 deg, against a true level gaze. The vertical ratio measured
/// on the same frames held 0.25-0.26 from frontal through profile.
///
/// Pitch response comes from the nose tip standing in FRONT of the
/// eye-line→chin plane: under a nod the plane's projected height shrinks as
/// `cos(pitch)` while the nose gains `d·sin(pitch)`, so the ratio is
/// `a/b + (d/b)·tan(pitch)` — monotonic across the usable range and linear
/// in `tan`, which is exactly what `atan` inverts.
///
/// `None` when the face-68 block is absent or any input landmark is below
/// the visibility floor; the caller then decides whether a fallback estimate
/// is defensible.
pub(super) fn pitch_from_vertical_ratio(joints: &[DecodedJoint]) -> Option<f32> {
    let lm = |dlib: usize| -> Option<&DecodedJoint> {
        let j = joints.get(FACE68_BASE + dlib)?;
        (j.score >= KEYPOINT_VISIBILITY_FLOOR).then_some(j)
    };
    let chin = lm(DLIB_CHIN)?;
    let nose = lm(DLIB_NOSE_TIP)?;
    // Mean of all four eye corners rather than two: the outer corners alone
    // put the baseline on the two points most likely to be occluded in a
    // profile view, and averaging halves the per-landmark noise.
    let mut eye_y = 0.0;
    for &c in &DLIB_EYE_CORNERS {
        eye_y += lm(c)?.ny;
    }
    let eye_y = eye_y / DLIB_EYE_CORNERS.len() as f32;
    // Image y grows downward, so both spans are positive on an upright head.
    let span = chin.ny - eye_y;
    if span < MIN_EYE_CHIN_SPAN {
        return None;
    }
    let ratio = (nose.ny - eye_y) / span;
    Some(((ratio - VERTICAL_RATIO_NEUTRAL) * VERTICAL_RATIO_GAIN).atan())
}

/// Legacy nose-below-eye-line pitch, normalised by the inter-eye distance
/// and yaw-decoupled by `cos(yaw)`. Only valid while `yaw` itself is
/// trustworthy — hence `ears_trustworthy`, which the caller sets from the
/// ear-line evidence that produced the yaw. Kept as the fallback for frames
/// where the face-68 block is missing or unscored, and returns `None`
/// (no face pose at all) when the yaw cannot be vouched for.
fn legacy_pitch_from_inter_eye(
    joints: &[DecodedJoint],
    yaw: f32,
    ears_trustworthy: bool,
) -> Option<f32> {
    if !ears_trustworthy {
        return None;
    }
    // An anatomical neutral subtracted — mirroring the FaceMesh path
    // (`derive_face_pose_from_landmarks`, which subtracts its own
    // `PITCH_NEUTRAL_SIGNAL`). The body path never had this: at a frontal
    // neutral pose the nose tip sits a fixed fraction below the eye line,
    // so the raw ratio maps to a large spurious `+pitch` (chin-down) —
    // before this subtraction a forward-facing head decoded ~50° down.
    // The per-person residual (camera height, face proportions) is
    // absorbed by `PoseCalibration::neutral_face_ypr_body`, captured
    // during the Calibrate Pose hold (from `face_body_raw`, so the
    // body path is measured even while the mesh wins selection) and
    // subtracted per-source in `apply_calibration`.
    //
    // Computed in model-pixel space (`nx·INPUT_W, ny·INPUT_H`) rather than
    // source space so the ratio is independent of the shoulder-derived
    // `aspect` and metric depth, keeping the constant stable across
    // framings. Image y-down: nose below the eye line → positive → source
    // `+pitch` (chin drops). The selfie mirror is horizontal, so vertical
    // `ny` is untouched by it. The COCO 5-point set uses eye *centres*
    // (closer together than FaceMesh's outer eye corners), so the neutral
    // ratio is larger than that path's 0.49 — measured ≈ 1.25 on neutral
    // frames (live desk pose + the palms_front capture).
    const PITCH_NEUTRAL_SIGNAL: f32 = 1.25;
    let nose_py = joints[0].ny * INPUT_H as f32;
    let le_px = (joints[1].nx * INPUT_W as f32, joints[1].ny * INPUT_H as f32);
    let re_px = (joints[2].nx * INPUT_W as f32, joints[2].ny * INPUT_H as f32);
    let eye_mid_py = (le_px.1 + re_px.1) * 0.5;
    let inter_eye_px = ((le_px.0 - re_px.0).powi(2) + (le_px.1 - re_px.1).powi(2))
        .sqrt()
        .max(1.0);
    // Yaw decoupling: under head yaw the inter-eye baseline
    // foreshortens by ~cos(yaw) while the nose-below-eye-line vertical
    // distance stays roughly constant, so the raw ratio inflates by
    // 1/cos(yaw) — a 45° head turn read as a spurious extra nod.
    // Multiply the measured ratio back by cos(yaw) to recover the
    // frontal-equivalent signal before subtracting the frontal-space
    // anatomical neutral. The 0.5 floor stops the correction from
    // collapsing the signal in profile views where the ear-line yaw
    // itself saturates and cos would over-shrink a noisy ratio.
    let yaw_foreshorten = yaw.cos().clamp(0.5, 1.0);
    let pitch_signal =
        (nose_py - eye_mid_py) / inter_eye_px * yaw_foreshorten - PITCH_NEUTRAL_SIGNAL;
    Some(pitch_signal.clamp(-2.0, 2.0).atan())
}

/// FaceMesh confidence above which the selector *switches to* the mesh
/// pose, and the lower floor below which it *abandons* it. Live
/// measurements (2026-07-17, headphones + glasses subject): frontal
/// ≈ 0.95–1.0, 3/4 view ≈ 0.4 (where the mesh yaw is still correct and
/// the body ear-line yaw reads ~0 because the ears are occluded by the
/// headphone cups), true collapse ≈ 0.003–0.19 at close-up profile.
/// The two sources disagree by tens of degrees in exactly the views
/// where the mesh confidence hovers near the boundary (the module's own
/// 3/4-view test shows a 59° yaw gap), so a single threshold made the
/// head snap between them every frame the confidence crossed it. The
/// enter/exit pair is a Schmitt trigger: the confidence must climb to
/// 0.3 to adopt the mesh and fall below 0.15 to drop it — chatter
/// inside [0.15, 0.3) keeps whichever source is currently active.
const MESH_ENTER_CONFIDENCE: f32 = 0.3;
const MESH_EXIT_CONFIDENCE: f32 = 0.15;

/// How long an ACTIVE mesh source survives a confidence dip while its
/// pose stream stays continuous. The mesh's "is this a face" flag
/// collapses to ~0 for a few frames during oblique up-looks even
/// though the landmark pose keeps tracking correctly (live desk
/// capture 2026-07-27: mesh_c flapped 1.0 → 0.00 → 0.93 → 0.01 → 1.0
/// over ~2.4 s while mesh yaw stayed a coherent −45…−57°; every dip
/// kicked the selector to the body path, whose ear-line yaw reads ~0
/// in profile — the avatar's head snapped to front instead of
/// following the visible mesh). Score dip ≠ pose invalid: ride it,
/// bounded by this window so a genuinely dead mesh (sustained
/// collapse, the frozen-profile case) still releases.
const MESH_DIP_RIDE_S: f32 = 1.0;
/// A mesh pose stepping farther than this (radians, any axis, per
/// frame) from the last published pose during a dip is NOT a
/// continuous track — it is the mesh reading garbage. Release
/// immediately instead of riding.
const MESH_DIP_MAX_STEP_RAD: f32 = 0.8;

/// Wrapped absolute angular difference (radians).
fn angle_gap(a: f32, b: f32) -> f32 {
    use std::f32::consts::PI;
    let mut d = a - b;
    while d > PI {
        d -= 2.0 * PI;
    }
    while d < -PI {
        d += 2.0 * PI;
    }
    d.abs()
}

/// Wall-clock duration of a source-switch crossfade. ~233 ms (the old
/// 6-blended-frames-at-30-fps schedule reached the target on the 7th
/// frame) — fast enough to not read as lag, slow enough that the worst
/// measured inter-source disagreement (~1 rad yaw at 3/4 view) moves
/// the head smoothly instead of teleporting it. Advanced by the real
/// capture dt, so the ease takes the same wall time at any frame rate.
const SOURCE_SWITCH_BLEND_S: f32 = 7.0 / 30.0;

/// Shortest-arc angle interpolation (radians). The inter-source gap is
/// well under π in practice, but wrap anyway so a pathological pair
/// can't blend the long way around.
fn lerp_angle(from: f32, to: f32, t: f32) -> f32 {
    use std::f32::consts::PI;
    let mut d = to - from;
    while d > PI {
        d -= 2.0 * PI;
    }
    while d < -PI {
        d += 2.0 * PI;
    }
    from + d * t
}

/// Stateful head-pose source selection: FaceMesh's dense-landmark pose
/// when the mesh actually saw a face (it has real Z separation through
/// the 22.5°–67.5° "3/4 view" dead-zone where the body-derived ear-line
/// yaw collapses to ~0), otherwise the body-derived pose. Adds two
/// pieces of temporal state over a pure per-frame pick:
///
/// * **Hysteresis** (`MESH_ENTER_CONFIDENCE` / `MESH_EXIT_CONFIDENCE`)
///   so a mesh confidence flickering around a single threshold can't
///   flip the source — and with it, each source's tens-of-degrees
///   systematic offset — every frame.
/// * **Crossfade** (`SOURCE_SWITCH_BLEND_S`) so the residual
///   inter-source disagreement at a legitimate switch plays out as a
///   ~230 ms ease instead of a head snap.
///
/// The published pose keeps the *target* source's tag and confidence
/// during the blend — only the angles are eased.
#[derive(Debug, Default)]
pub(super) struct FaceSourceSelector {
    using_mesh: bool,
    /// Wall time elapsed since the active crossfade began; the blend is
    /// live while `blend_from` is `Some` and this is under
    /// [`SOURCE_SWITCH_BLEND_S`].
    blend_elapsed_s: f32,
    blend_from: Option<FacePose>,
    last_output: Option<FacePose>,
    /// Wall time spent in the current below-EXIT confidence dip while
    /// the mesh source stays active (see [`MESH_DIP_RIDE_S`]).
    dip_elapsed_s: f32,
}

impl FaceSourceSelector {
    /// `dt_s`: capture-timestamp frame step (nominal 1/30 when the input
    /// carries no device timestamps) — advances the crossfade in wall
    /// time so the ease duration is frame-rate independent.
    pub(super) fn select(
        &mut self,
        body: Option<FacePose>,
        mesh: Option<FacePose>,
        mesh_conf: f32,
        dt_s: f32,
    ) -> Option<FacePose> {
        // Schmitt-trigger source state. The mesh must exist this frame
        // to be (or stay) the active source. An ACTIVE mesh additionally
        // rides out brief confidence dips while its pose stream stays
        // continuous (see [`MESH_DIP_RIDE_S`]) — the flag score lies
        // during oblique up-looks while the landmarks keep tracking.
        let mesh_wanted = if self.using_mesh {
            match &mesh {
                None => false,
                Some(_) if mesh_conf >= MESH_EXIT_CONFIDENCE => {
                    self.dip_elapsed_s = 0.0;
                    true
                }
                Some(m) => {
                    let continuous = self.last_output.as_ref().is_some_and(|prev| {
                        angle_gap(m.yaw, prev.yaw)
                            .max(angle_gap(m.pitch, prev.pitch))
                            .max(angle_gap(m.roll, prev.roll))
                            <= MESH_DIP_MAX_STEP_RAD
                    });
                    self.dip_elapsed_s += dt_s;
                    continuous && self.dip_elapsed_s < MESH_DIP_RIDE_S
                }
            }
        } else {
            let wanted = mesh.is_some() && mesh_conf >= MESH_ENTER_CONFIDENCE;
            if wanted {
                self.dip_elapsed_s = 0.0;
            }
            wanted
        };
        if mesh_wanted != self.using_mesh {
            self.using_mesh = mesh_wanted;
            // Ease from whatever we last published (which may itself be
            // mid-blend) toward the new source. First-ever frame has no
            // previous output — no blend, publish the new source as-is.
            self.blend_from = self.last_output;
            self.blend_elapsed_s = 0.0;
            if self.blend_from.is_none() {
                self.blend_elapsed_s = SOURCE_SWITCH_BLEND_S;
            }
        }

        let raw = match (self.using_mesh, body, mesh) {
            (true, body, Some(mut mesh_pose)) => {
                // The two confidences are independent "is the face
                // visible" signals, so the higher one is the better
                // evidence for the solver's gate — symmetric with the
                // body-retained arm below.
                let body_conf = body.map(|b| b.confidence).unwrap_or(0.0);
                mesh_pose.confidence = mesh_conf.max(body_conf);
                Some(mesh_pose)
            }
            (false, Some(mut body_pose), _) => {
                body_pose.confidence = body_pose.confidence.max(mesh_conf);
                Some(body_pose)
            }
            (false, None, Some(mut mesh_pose)) => {
                // No body pose to fall back on — surface the weak mesh
                // pose with its honest confidence and let the solver
                // gate decide. Deliberately does NOT flip `using_mesh`:
                // this is a last resort, not evidence the mesh is
                // healthy.
                mesh_pose.confidence = mesh_conf;
                Some(mesh_pose)
            }
            (true, _, None) => unreachable!("using_mesh requires mesh.is_some()"),
            (false, None, None) => None,
        };

        let out = match raw {
            Some(target) => {
                let eased = if self.blend_elapsed_s < SOURCE_SWITCH_BLEND_S {
                    if let Some(from) = self.blend_from {
                        self.blend_elapsed_s += dt_s;
                        if self.blend_elapsed_s >= SOURCE_SWITCH_BLEND_S {
                            // The ease's wall time has fully elapsed —
                            // publish the target outright.
                            self.blend_from = None;
                            target
                        } else {
                            let t = self.blend_elapsed_s / SOURCE_SWITCH_BLEND_S;
                            FacePose {
                                yaw: lerp_angle(from.yaw, target.yaw, t),
                                pitch: lerp_angle(from.pitch, target.pitch, t),
                                roll: lerp_angle(from.roll, target.roll, t),
                                // Calibration must interpolate the per-source
                                // neutrals with this same t (see
                                // `FacePose::blend`). `from.source` is exact
                                // for a switch out of steady state; a switch
                                // landing mid-blend anchors on the previous
                                // target's source, whose neutral only
                                // approximates the mixed anchor pose — the
                                // residual is bounded by the neutral gap
                                // times the interrupted blend's remaining
                                // fraction and decays over this blend.
                                blend: Some((from.source, t)),
                                ..target
                            }
                        }
                    } else {
                        self.blend_elapsed_s = SOURCE_SWITCH_BLEND_S;
                        target
                    }
                } else {
                    target
                };
                Some(eased)
            }
            None => {
                // Full face dropout: forget the blend anchor so a pose
                // reappearing seconds later doesn't ease from stale
                // angles.
                self.blend_elapsed_s = SOURCE_SWITCH_BLEND_S;
                self.blend_from = None;
                None
            }
        };
        if out.is_some() {
            self.last_output = out;
        } else {
            self.last_output = None;
        }
        out
    }
}

/// Pixel-space face bbox derived from RTMW3D's 68 face landmarks
/// (indices 23..=90, dlib 68-point convention with jawline / brows /
/// nose / eyes / mouth). Body keypoints 0..=4 (nose / eyes / ears)
/// give a centroid that's too high on the face — they cluster on the
/// upper half so 1.7× padding overshoots above the head and clips
/// the chin. The face-68 set spans the whole face including mouth
/// and jaw, so its centroid lands on the actual face centre. Returns
/// `None` when too few face keypoints clear the confidence floor.
pub(super) fn build_face_bbox_from_joints(
    joints: &[DecodedJoint],
    width: u32,
    height: u32,
) -> Option<FaceBbox> {
    let mut points_px: Vec<(f32, f32)> = Vec::with_capacity(68);
    for i in 23..=90 {
        let Some(j) = joints.get(i) else { continue };
        if j.score < KEYPOINT_VISIBILITY_FLOOR {
            continue;
        }
        points_px.push((j.nx * width as f32, j.ny * height as f32));
    }
    derive_face_bbox(&points_px, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::source_skeleton::SourceJoint;

    /// Nominal 30 fps step — the baseline the crossfade schedule was
    /// calibrated at (6 blended frames, target on the 7th).
    const DT: f32 = 1.0 / 30.0;
    /// Ideal per-frame calibrated-space increment divisor at 30 fps.
    const BLEND_STEPS_AT_30: u32 = 7;

    fn dj(nx: f32, ny: f32, nz: f32) -> DecodedJoint {
        DecodedJoint {
            nx,
            ny,
            nz,
            score: 0.9,
            z_score: 0.9,
        }
    }

    /// Eye-line→chin span used by the face-68 fixtures, in the same
    /// `INPUT_H` pixel space as the 5-point fixtures below. Eyes sit at
    /// py=150, so the chin lands at py=270.
    const CHIN_SPAN_PY: f32 = 120.0;

    /// Add the face-68 block (indices 23..=90) to a 5-point fixture so the
    /// primary estimator has landmarks to read. The nose tip is placed at
    /// `ratio` of the eye-line→chin span, i.e. exactly the quantity
    /// [`pitch_from_vertical_ratio`] measures; eye corners share the eye
    /// line and the chin closes the span. Landmarks the estimator does not
    /// read are left at the fixture's default (score 0) — it must not
    /// depend on them.
    fn with_face68(mut j: Vec<DecodedJoint>, ratio: f32) -> Vec<DecodedJoint> {
        let h = INPUT_H as f32;
        let w = INPUT_W as f32;
        j.resize(FACE68_BASE + 68, DecodedJoint::default());
        let eye_py = 150.0;
        let put = |j: &mut Vec<DecodedJoint>, dlib: usize, px: f32, py: f32| {
            j[FACE68_BASE + dlib] = dj(px / w, py / h, 0.5);
        };
        for (dlib, px) in [(36, 128.0), (39, 140.0), (42, 148.0), (45, 160.0)] {
            put(&mut j, dlib, px, eye_py);
        }
        put(&mut j, DLIB_CHIN, 144.0, eye_py + CHIN_SPAN_PY);
        put(&mut j, DLIB_NOSE_TIP, 144.0, eye_py + ratio * CHIN_SPAN_PY);
        j
    }

    /// Front-facing synthetic face with the nose at `nose_py` pixels
    /// (model space is `INPUT_W×INPUT_H`). Eyes at py=150, inter-eye 32 px,
    /// ears symmetric at equal depth (yaw≈0), shoulders/hips set so
    /// `aspect` and the origin are well-defined. A frontal neutral head
    /// has the nose ~1.25·inter-eye (= 40 px) below the eye line.
    fn neutral_joints(nose_py: f32) -> Vec<DecodedJoint> {
        let w = INPUT_W as f32;
        let h = INPUT_H as f32;
        let px = |x: f32, y: f32, z: f32| dj(x / w, y / h, z);
        let mut j = vec![DecodedJoint::default(); 13];
        j[0] = px(144.0, nose_py, 0.5); // nose
        j[1] = px(160.0, 150.0, 0.5); // subject-left eye
        j[2] = px(128.0, 150.0, 0.5); // subject-right eye
        j[3] = px(174.0, 150.0, 0.5); // subject-left ear
        j[4] = px(114.0, 150.0, 0.5); // subject-right ear
        j[5] = px(200.0, 280.0, 0.5); // subject-left shoulder
        j[6] = px(88.0, 280.0, 0.5); // subject-right shoulder
        j[11] = px(168.0, 376.0, 0.5); // subject-left hip
        j[12] = px(120.0, 376.0, 0.5); // subject-right hip
        j
    }

    fn pose(yaw: f32, confidence: f32) -> FacePose {
        FacePose {
            yaw,
            pitch: 0.0,
            roll: 0.0,
            confidence,
            ..Default::default()
        }
    }

    fn mesh_pose(yaw: f32, confidence: f32) -> FacePose {
        FacePose {
            source: FaceSource::Mesh,
            ..pose(yaw, confidence)
        }
    }

    #[test]
    fn collapsed_mesh_confidence_keeps_body_pose() {
        // Live-measured profile view: body ear-line yaw −0.93 at conf
        // 0.66, mesh score collapsed to 1.9e-8. The mesh pose must NOT
        // replace the body pose (the old unconditional preference froze
        // the avatar's head on every sideways turn).
        let mut sel = FaceSourceSelector::default();
        let out = sel
            .select(Some(pose(-0.93, 0.66)), Some(mesh_pose(-0.2, 1.0)), 1.9e-8, DT)
            .unwrap();
        assert!((out.yaw - -0.93).abs() < 1e-6);
        assert!(out.confidence >= 0.66);
        assert_eq!(out.source, FaceSource::Body);
    }

    #[test]
    fn confident_mesh_pose_wins() {
        // Front / 3/4 views: the mesh genuinely sees a face, and its
        // dense-landmark pose covers the body ear-line dead-zone.
        let mut sel = FaceSourceSelector::default();
        let out = sel
            .select(Some(pose(0.05, 0.7)), Some(mesh_pose(0.6, 1.0)), 0.92, DT)
            .unwrap();
        assert!((out.yaw - 0.6).abs() < 1e-6);
        assert!((out.confidence - 0.92).abs() < 1e-6);
        assert_eq!(out.source, FaceSource::Mesh);
    }

    #[test]
    fn three_quarter_view_degraded_mesh_still_wins() {
        // Live-measured 3/4 view: mesh_c ≈ 0.42 with a correct yaw
        // while the body ear-line reads ~0 (ears occluded by headphone
        // cups). The degraded-but-alive mesh pose must win. Confidence
        // folding is symmetric (max of the two independent visibility
        // signals), so the healthy body confidence carries through.
        let mut sel = FaceSourceSelector::default();
        let out = sel
            .select(Some(pose(-0.04, 0.7)), Some(mesh_pose(-1.07, 1.0)), 0.42, DT)
            .unwrap();
        assert!((out.yaw - -1.07).abs() < 1e-6);
        assert!((out.confidence - 0.7).abs() < 1e-6);
        assert_eq!(out.source, FaceSource::Mesh);
    }

    #[test]
    fn weak_mesh_pose_is_last_resort_when_body_missing() {
        // No body pose at all: surface the mesh pose with its honest
        // (low) confidence so the solver's GUI threshold decides.
        let mut sel = FaceSourceSelector::default();
        let out = sel.select(None, Some(mesh_pose(0.3, 1.0)), 0.1, DT).unwrap();
        assert!((out.yaw - 0.3).abs() < 1e-6);
        assert!((out.confidence - 0.1).abs() < 1e-6);
    }

    #[test]
    fn mesh_conf_folds_into_body_confidence_when_mesh_pose_absent() {
        let mut sel = FaceSourceSelector::default();
        let out = sel.select(Some(pose(-0.4, 0.3)), None, 0.45, DT).unwrap();
        assert!((out.yaw - -0.4).abs() < 1e-6);
        assert!((out.confidence - 0.45).abs() < 1e-6);
    }

    #[test]
    fn confidence_chatter_between_thresholds_does_not_flip_source() {
        // Regression for the head-snap bug: the 3/4-view sources
        // disagree by ~59° while mesh confidence hovers around the old
        // single 0.2 threshold. Inside the Schmitt band [0.15, 0.3)
        // the active source must not change in either direction.
        let body = pose(-0.04, 0.7);
        let mesh = mesh_pose(-1.07, 1.0);

        // Starting on body: 0.2 / 0.29 are below ENTER → stays body.
        let mut sel = FaceSourceSelector::default();
        for conf in [0.2, 0.29, 0.18, 0.25] {
            let out = sel.select(Some(body), Some(mesh), conf, DT).unwrap();
            assert_eq!(out.source, FaceSource::Body, "conf {conf} flipped to mesh");
        }

        // Starting on mesh (one confident frame), then chatter in the
        // band: 0.2 / 0.16 are above EXIT → stays mesh.
        let mut sel = FaceSourceSelector::default();
        let out = sel.select(Some(body), Some(mesh), 0.9, DT).unwrap();
        assert_eq!(out.source, FaceSource::Mesh);
        for conf in [0.2, 0.16, 0.28, 0.19] {
            let out = sel.select(Some(body), Some(mesh), conf, DT).unwrap();
            assert_eq!(out.source, FaceSource::Mesh, "conf {conf} dropped to body");
        }

        // Genuine collapse below EXIT: the dip-ride keeps the mesh for
        // up to MESH_DIP_RIDE_S while its pose is continuous, then
        // releases to body.
        for _ in 0..((MESH_DIP_RIDE_S / DT).ceil() as u32 + 2) {
            sel.select(Some(body), Some(mesh), 0.05, DT).unwrap();
        }
        let out = sel.select(Some(body), Some(mesh), 0.05, DT).unwrap();
        assert_eq!(out.source, FaceSource::Body, "sustained collapse must release the mesh");
    }

    #[test]
    fn mesh_confidence_dip_with_continuous_pose_rides_through() {
        // THE live up-look failure (2026-07-27): mesh_c flaps
        // 1.0 → 0.00 → 0.93 → 0.01 → 1.0 while the mesh pose stays a
        // coherent −45…−57° track. Every dip used to flip the selector
        // to the body path (ear-line yaw ≈ 0 in profile) and the head
        // snapped to front. The dip must ride on the mesh pose.
        let body = pose(0.0, 0.7); // body claims "facing forward" — the wrong pose
        let mut sel = FaceSourceSelector::default();
        let mut yaw = -0.80;
        sel.select(Some(body), Some(mesh_pose(yaw, 1.0)), 1.0, DT).unwrap();
        for i in 0..20 {
            // Pose keeps drifting slightly (a real head mid-motion)
            // while the confidence flaps hard every other frame.
            yaw -= 0.01;
            let conf = if i % 2 == 0 { 0.005 } else { 0.95 };
            let out = sel.select(Some(body), Some(mesh_pose(yaw, 1.0)), conf, DT).unwrap();
            assert_eq!(out.source, FaceSource::Mesh, "frame {i}: dip must not flip to body");
            assert!(
                (out.yaw - yaw).abs() < 1e-5,
                "frame {i}: published yaw must follow the mesh, got {} want {yaw}",
                out.yaw
            );
        }
    }

    #[test]
    fn mesh_dip_with_discontinuous_pose_releases_immediately() {
        // Safety valve: a score collapse WITH a wild pose jump is a mesh
        // reading garbage (not a lying flag) — release on that frame
        // instead of riding on nonsense.
        let body = pose(0.0, 0.7);
        let mut sel = FaceSourceSelector::default();
        sel.select(Some(body), Some(mesh_pose(-0.8, 1.0)), 1.0, DT).unwrap();
        let out = sel.select(Some(body), Some(mesh_pose(1.4, 1.0)), 0.01, DT).unwrap();
        assert_eq!(
            out.source,
            FaceSource::Body,
            "dip + 2.2 rad pose jump must release the mesh immediately"
        );
    }

    #[test]
    fn source_switch_crossfades_instead_of_snapping() {
        // 59° inter-source yaw gap (live-measured 3/4 view). On the
        // switch frame the published yaw must land strictly between
        // the two sources, and successive frames must approach the new
        // source monotonically until the blend runs out.
        let body = pose(-0.04, 0.7);
        let mesh = mesh_pose(-1.07, 1.0);
        let mut sel = FaceSourceSelector::default();
        // Establish body as the active source.
        let out = sel.select(Some(body), Some(mesh), 0.05, DT).unwrap();
        assert!((out.yaw - -0.04).abs() < 1e-6);

        // Mesh becomes confident → switch begins.
        let mut prev_yaw = out.yaw;
        let mut reached = false;
        for frame in 0..12 {
            let out = sel.select(Some(body), Some(mesh), 0.9, DT).unwrap();
            assert_eq!(out.source, FaceSource::Mesh, "tag is the target source");
            assert!(
                out.yaw <= prev_yaw + 1e-6,
                "frame {frame}: yaw must move monotonically toward the mesh pose"
            );
            if frame == 0 {
                assert!(
                    out.yaw < -0.04 - 0.05 && out.yaw > -1.07 + 0.05,
                    "switch frame must be strictly between sources, got {}",
                    out.yaw
                );
            }
            prev_yaw = out.yaw;
            if (out.yaw - -1.07).abs() < 1e-6 {
                reached = true;
                break;
            }
        }
        assert!(reached, "blend must converge to the mesh pose, ended at {prev_yaw}");
    }

    #[test]
    fn crossfade_is_continuous_in_calibrated_space() {
        // C1 regression: the crossfade blends RAW angles, but the
        // per-source neutral subtraction keys on the (target) source
        // tag. Without interpolating the neutrals with the same t, the
        // switch frame subtracts the full target neutral from angles
        // that are still ~6/7 from-source — the inter-source neutral
        // gap appears as a calibrated-space head step on the exact
        // frame the crossfade exists to smooth.
        use crate::tracking::{
            CalibrationMode, PoseCalibration, SourceSkeleton, TrackingCalibration,
        };
        let n_body = [0.3, 0.0, 0.0];
        let n_mesh = [-0.5, 0.0, 0.0];
        let cal = TrackingCalibration {
            pose: Some(PoseCalibration {
                mode: CalibrationMode::UpperBody,
                captured_at: String::new(),
                captured_at_unix: 0,
                frame_count: 1,
                anchor_x: 0.0,
                anchor_y: 0.0,
                anchor_depth_m: None,
                confidence: 1.0,
                anchor_depth_jitter_m: None,
                shoulder_span_m: None,
                x_range_observed: None,
                z_range_observed: None,
                torso_depth_template: None,
                neutral_expressions: Vec::new(),
                neutral_face_ypr_mesh: Some(n_mesh),
                neutral_face_ypr_body: Some(n_body),
                neutral_body_yaw: None,
            }),
        };
        let calibrated = |p: FacePose| -> f32 {
            let mut sk = SourceSkeleton::default();
            sk.face = Some(p);
            cal.apply_calibration(&mut sk);
            sk.face.unwrap().yaw
        };

        // Body at its own neutral (calibrated 0), mesh well off it.
        let body = pose(n_body[0], 0.7);
        let mesh = mesh_pose(-1.07, 1.0);
        let mesh_cal_target = -1.07 - n_mesh[0]; // −0.57

        let mut sel = FaceSourceSelector::default();
        let mut prev = calibrated(sel.select(Some(body), Some(mesh), 0.05, DT).unwrap());
        assert!(prev.abs() < 1e-6, "body at its neutral must calibrate to 0");

        // Switch to mesh: every calibrated-space step must stay near
        // the ideal per-frame increment (gap / (BLEND+1) ≈ 0.081) —
        // in particular NO step anywhere near the raw neutral gap
        // (0.8) or the pre-fix switch-frame jump (~0.6).
        let gap = (mesh_cal_target - prev).abs();
        let max_step = gap / BLEND_STEPS_AT_30 as f32 + 0.01;
        for frame in 0..12 {
            let now = calibrated(sel.select(Some(body), Some(mesh), 0.9, DT).unwrap());
            assert!(
                (now - prev).abs() <= max_step,
                "frame {frame}: calibrated yaw stepped {:.3} (limit {:.3})",
                (now - prev).abs(),
                max_step
            );
            prev = now;
            if (now - mesh_cal_target).abs() < 1e-6 {
                break;
            }
        }
        assert!(
            (prev - mesh_cal_target).abs() < 1e-6,
            "must converge to the mesh calibrated pose, ended at {prev}"
        );
    }

    #[test]
    fn blend_metadata_marks_only_crossfade_frames() {
        // P4 regression: the calibration hold keys on `FacePose::blend`
        // to reject mid-crossfade frames (target tag over mixed
        // angles). Steady frames must NOT carry it, every crossfade
        // frame must, and it must clear once the blend completes.
        let body = pose(-0.04, 0.7);
        let mesh = mesh_pose(-1.07, 1.0);
        let mut sel = FaceSourceSelector::default();
        let out = sel.select(Some(body), Some(mesh), 0.05, DT).unwrap();
        assert!(out.blend.is_none(), "steady body frame must not be marked");

        for frame in 0..(BLEND_STEPS_AT_30 - 1) {
            let out = sel.select(Some(body), Some(mesh), 0.9, DT).unwrap();
            let (from, t) = out.blend.expect("crossfade frame must be marked");
            assert_eq!(from, FaceSource::Body, "frame {frame}: blend anchors on the from-source");
            assert!(t > 0.0 && t < 1.0, "frame {frame}: t={t} out of (0,1)");
        }
        let out = sel.select(Some(body), Some(mesh), 0.9, DT).unwrap();
        assert!(out.blend.is_none(), "post-blend steady mesh frame must clear the mark");
        assert_eq!(out.source, FaceSource::Mesh);
    }

    #[test]
    fn crossfade_duration_is_wall_time_not_frames() {
        // At 15 fps (dt = 2/30) the same ~233 ms ease must complete in
        // about half the frames — the old frame-count schedule stretched
        // it to ~400 ms whenever the capture rate halved.
        let body = pose(-0.04, 0.7);
        let mesh = mesh_pose(-1.07, 1.0);
        let count_blend_frames = |dt: f32| -> u32 {
            let mut sel = FaceSourceSelector::default();
            sel.select(Some(body), Some(mesh), 0.05, dt).unwrap();
            let mut frames = 0;
            for _ in 0..20 {
                let out = sel.select(Some(body), Some(mesh), 0.9, dt).unwrap();
                if out.blend.is_none() {
                    return frames;
                }
                frames += 1;
            }
            frames
        };
        let at_30 = count_blend_frames(DT);
        let at_15 = count_blend_frames(2.0 * DT);
        assert_eq!(at_30, 6, "30 fps keeps the calibrated 6-blended-frame schedule");
        assert_eq!(at_15, 3, "15 fps covers the same wall time in half the frames");
    }

    #[test]
    fn face_dropout_resets_blend_anchor() {
        // A pose reappearing after a full dropout must not ease from
        // stale pre-dropout angles.
        let mut sel = FaceSourceSelector::default();
        sel.select(Some(pose(0.8, 0.7)), None, 0.0, DT).unwrap();
        assert!(sel.select(None, None, 0.0, DT).is_none());
        let out = sel.select(Some(pose(-0.5, 0.7)), None, 0.0, DT).unwrap();
        assert!((out.yaw - -0.5).abs() < 1e-6, "no blend from stale pose");
    }

    fn skeleton_with_shoulder() -> SourceSkeleton {
        let mut sk = SourceSkeleton::default();
        sk.joints.insert(
            HumanoidBone::LeftShoulder,
            SourceJoint { position: [0.18, 0.0, 0.0], confidence: 0.9, ..Default::default() },
        );
        sk
    }

    #[test]
    fn neutral_face68_has_zero_pitch() {
        // Primary estimator: nose tip at the neutral fraction of the
        // eye-line→chin span decodes to ~0.
        let sk = skeleton_with_shoulder();
        let joints = with_face68(neutral_joints(190.0), VERTICAL_RATIO_NEUTRAL);
        let face = derive_face_pose_from_body(&sk, &joints).unwrap();
        assert!(face.pitch.abs() < 1e-5, "neutral must be 0, got {}", face.pitch);
    }

    #[test]
    fn face68_pitch_follows_the_nose_between_eye_line_and_chin() {
        // Monotonic and signed the documented way: the nose dropping
        // toward the chin is chin-DOWN (+), rising toward the eye line is
        // chin-UP (−). Magnitudes follow `atan(Δratio · gain)`.
        let sk = skeleton_with_shoulder();
        let pitch_at = |ratio: f32| {
            derive_face_pose_from_body(&sk, &with_face68(neutral_joints(190.0), ratio))
                .unwrap()
                .pitch
        };
        let down = pitch_at(VERTICAL_RATIO_NEUTRAL + 0.20);
        let up = pitch_at(VERTICAL_RATIO_NEUTRAL - 0.20);
        assert!(
            (down - (0.20 * VERTICAL_RATIO_GAIN).atan()).abs() < 1e-5,
            "chin-down magnitude off: {down}"
        );
        assert!((up + down).abs() < 1e-5, "must be antisymmetric: {up} vs {down}");
        assert!(down > 0.5 && up < -0.5, "expected ±29°, got {down} / {up}");
    }

    #[test]
    fn dead_ear_yaw_does_not_leak_into_pitch() {
        // THE live failure (2026-07-27): hair / glasses temples hide both
        // ears, so the ear-line yaw reads ~0 through a real 60° turn while
        // the inter-eye baseline foreshortens by cos(60°) = 0.5. The old
        // inter-eye estimator inflated its ratio 2× and published +50°
        // chin-down for a level gaze — the avatar slammed its head down
        // every time the subject looked sideways. The vertical ratio is
        // built from image `y` alone, which a yaw cannot change.
        let w = INPUT_W as f32;
        let h = INPUT_H as f32;
        let px = |x: f32, y: f32, z: f32| dj(x / w, y / h, z);
        let c = 0.5_f32; // cos 60°
        let mut j = vec![DecodedJoint::default(); 13];
        j[0] = px(144.0, 190.0, 0.5);
        j[1] = px(144.0 + 16.0 * c, 150.0, 0.5);
        j[2] = px(144.0 - 16.0 * c, 150.0, 0.5);
        // Ears at EQUAL depth and equal score: the yaw signal is dead, but
        // nothing about the scores says so.
        j[3] = px(144.0 + 30.0 * c, 150.0, 0.5);
        j[4] = px(144.0 - 30.0 * c, 150.0, 0.5);
        j[5] = px(200.0, 280.0, 0.5);
        j[6] = px(88.0, 280.0, 0.5);
        j[11] = px(168.0, 376.0, 0.5);
        j[12] = px(120.0, 376.0, 0.5);

        let sk = skeleton_with_shoulder();
        let turned = with_face68(j, VERTICAL_RATIO_NEUTRAL);
        let face = derive_face_pose_from_body(&sk, &turned).unwrap();
        assert!(
            face.yaw.abs() < 0.1,
            "fixture must reproduce the DEAD ear-line yaw, got {}",
            face.yaw
        );
        assert!(
            face.pitch.abs() < 0.02,
            "a turned head must not decode as a nod, got {} rad ({}°)",
            face.pitch,
            face.pitch.to_degrees()
        );
    }

    #[test]
    fn profile_view_collapsed_ear_baseline_refuses_pose() {
        // THE live "broken neck" (2026-07-27, oblique desk camera): both
        // ears project onto nearly the same image point (separation ≈ 3%
        // of the head's keypoint spread) while their SimCC z values are
        // hallucinated and well-separated. The old `ear_dist` gate mixed
        // z into its magnitude, so the hallucinated dz let the frame
        // through — and yaw = atan2(dz, dx≈0) published ±90…±150° noise
        // every frame. A collapsed 2D baseline must refuse the pose.
        let w = INPUT_W as f32;
        let h = INPUT_H as f32;
        let px = |x: f32, y: f32, z: f32| dj(x / w, y / h, z);
        let mut j = vec![DecodedJoint::default(); 13];
        j[0] = px(234.0, 170.0, 0.50); // nose far to the side (profile)
        j[1] = px(228.0, 150.0, 0.52); // eyes nearly stacked
        j[2] = px(222.0, 147.0, 0.48);
        j[3] = px(183.0, 170.0, 0.62); // both ears at ~the same point...
        j[4] = px(181.0, 166.0, 0.38); // ...but hallucinated z separates
        j[5] = px(200.0, 280.0, 0.5);
        j[6] = px(88.0, 280.0, 0.5);
        j[11] = px(168.0, 376.0, 0.5);
        j[12] = px(120.0, 376.0, 0.5);
        let sk = skeleton_with_shoulder();
        assert!(
            derive_face_pose_from_body(&sk, &j).is_none(),
            "collapsed ear baseline must refuse the pose, not divide by it"
        );
        // Even with the face-68 pitch block present the yaw/roll are
        // still unobservable — the refusal must hold.
        let with68 = with_face68(j, VERTICAL_RATIO_NEUTRAL);
        assert!(derive_face_pose_from_body(&sk, &with68).is_none());
    }

    #[test]
    fn no_face68_and_untrustworthy_ears_publishes_no_pose() {
        // Fallback refusal: without the face-68 block the only estimator
        // left divides by a yaw-foreshortened baseline, and with one ear
        // reconstructed the yaw that would correct it is itself a guess.
        // Publishing nothing lets the selector hold the mesh / previous
        // pose instead of nodding the avatar on a bad frame.
        let sk = skeleton_with_shoulder();
        let mut j = neutral_joints(190.0);
        j[3].score = 0.05; // subject-left ear occluded → reconstruction fires
        assert!(derive_face_pose_from_body(&sk, &j).is_none());
        // Same frame WITH the face-68 block: the primary estimator does not
        // need the ears at all, so the pose still publishes.
        let with68 = with_face68(j, VERTICAL_RATIO_NEUTRAL);
        assert!(derive_face_pose_from_body(&sk, &with68).is_some());
    }

    #[test]
    fn legacy_fallback_neutral_forward_face_has_near_zero_pitch() {
        // Fallback estimator (no face-68 block, ears trustworthy): a
        // front-facing head (nose the anatomical ~1.25·inter-eye below the
        // eye line) must decode to ~0 pitch — not the ~50°-down bias the
        // un-subtracted body path used to emit.
        let sk = skeleton_with_shoulder();
        let face = derive_face_pose_from_body(&sk, &neutral_joints(190.0)).unwrap();
        assert!(
            face.pitch.abs() < 0.12,
            "neutral face pitch should be ~0, got {} rad",
            face.pitch
        );
    }

    #[test]
    fn legacy_fallback_chin_down_is_positive_pitch() {
        // Fallback estimator. Nose dropped further below the eye line →
        // chin-down → +pitch.
        let sk = skeleton_with_shoulder();
        let face = derive_face_pose_from_body(&sk, &neutral_joints(214.0)).unwrap();
        assert!(face.pitch > 0.3, "chin-down should be +pitch, got {}", face.pitch);
    }

    #[test]
    fn legacy_fallback_chin_up_is_negative_pitch() {
        // Fallback estimator. Nose lifted toward the eye line → chin-up →
        // -pitch.
        let sk = skeleton_with_shoulder();
        let face = derive_face_pose_from_body(&sk, &neutral_joints(166.0)).unwrap();
        assert!(face.pitch < -0.3, "chin-up should be -pitch, got {}", face.pitch);
    }

    #[test]
    fn legacy_fallback_yawed_head_does_not_leak_into_pitch() {
        // Fallback estimator, ears trustworthy so its cos(yaw) decoupling
        // actually has a yaw to work with.
        // Head turned ~45° with NO pitch: the inter-eye baseline
        // foreshortens by cos(45°) while the nose-below-eye vertical
        // distance stays put, so the raw ratio inflates 1.41× — which
        // used to decode as a spurious ~27° nod every time the subject
        // shook their head. With the cos(yaw) decoupling the pitch must
        // stay near zero.
        let w = INPUT_W as f32;
        let h = INPUT_H as f32;
        let px = |x: f32, y: f32, z: f32| dj(x / w, y / h, z);
        let c = std::f32::consts::FRAC_1_SQRT_2; // cos 45°
        let mut j = vec![DecodedJoint::default(); 13];
        // Horizontal spans compressed by cos(45°); vertical untouched.
        j[0] = px(144.0, 190.0, 0.5); // nose, 40 px below eye line
        j[1] = px(144.0 + 16.0 * c, 150.0, 0.5);
        j[2] = px(144.0 - 16.0 * c, 150.0, 0.5);
        // Ears: depth-separated so the ear-line reads yaw ≈ +45°
        // (|ear_dz| chosen equal to the compressed source-space
        // |ear_dx| = 2 · (30·c/288) · aspect with aspect ≈ 0.9257).
        let ear_half_nx = 30.0 * c / w;
        let aspect = 0.18 / ((200.0 - 88.0) / w / 2.0);
        let d = ear_half_nx * aspect;
        j[3] = px(144.0 + 30.0 * c, 150.0, 0.5 + d);
        j[4] = px(144.0 - 30.0 * c, 150.0, 0.5 - d);
        j[5] = px(200.0, 280.0, 0.5);
        j[6] = px(88.0, 280.0, 0.5);
        j[11] = px(168.0, 376.0, 0.5);
        j[12] = px(120.0, 376.0, 0.5);

        let sk = skeleton_with_shoulder();
        let face = derive_face_pose_from_body(&sk, &j).unwrap();
        assert!(
            face.yaw.abs() > 0.6,
            "test geometry must actually read as a yawed head, got {} rad",
            face.yaw
        );
        assert!(
            face.pitch.abs() < 0.12,
            "yaw must not leak into pitch, got {} rad ({}°)",
            face.pitch,
            face.pitch.to_degrees()
        );
    }
}
