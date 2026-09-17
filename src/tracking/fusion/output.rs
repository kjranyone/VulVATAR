//! Estimator posterior → [`RigPose`] (what the avatar retarget consumes)
//! and a compatibility [`SourceSkeleton`] (what the GUI overlay,
//! debug channel and expression solver still read).

use std::collections::HashMap;

use crate::asset::HumanoidBone;
use crate::tracking::source_skeleton::{
    CameraIntrinsics, MetricFrameInfo, SourceJoint, SourceSkeleton,
};

use super::estimator::Estimator;
use super::math::*;
use super::model::*;

/// Rotation from the camera frame (x-right, y-down, z-forward) to the
/// avatar/viewer frame (x-right, y-up, z-toward-camera): Rx(180°).
pub const CAM_TO_VIEW: M3 = FACING_CAMERA;

/// One driven bone: its world-space rotation in the *viewer* frame
/// expressed as a delta from the model rest (T-pose facing the viewer),
/// plus the marginal σ of the joint that produced it.
#[derive(Clone, Copy, Debug)]
pub struct RigBone {
    /// World-space delta rotation Δ_V (viewer frame), quaternion `[x,y,z,w]`.
    /// Avatar world rotation = Δ_V · avatar_rest_world_rotation.
    pub delta_world: [f32; 4],
    /// Marginal σ (rad) of the joint's own rotation parameters (posterior:
    /// data + priors).
    pub sigma: f32,
    /// Measurement-only σ (rad): how well the *data* alone pins this joint
    /// (leaky over ~0.3 s). Large ⇒ the pose is the prior's relaxed guess.
    pub data_sigma: f32,
}

/// The tracking output contract for the retarget: joint rotations as
/// world deltas, the pelvis position, and per-channel uncertainty.
#[derive(Clone, Debug, Default)]
pub struct RigPose {
    /// Capture time (s, device clock).
    pub t: f64,
    pub bones: HashMap<HumanoidBone, RigBone>,
    /// Pelvis (root) position in camera metres.
    pub root_cam_m: [f32; 3],
    /// σ of the root position (m).
    pub root_sigma_m: f32,
    /// Overall track quality in `[0,1]`: posterior torso σ (scaled by
    /// projected shoulder span) times a measurement-support factor from
    /// the trunk's data-only σ — collapses to 0 a few seconds after the
    /// subject stops being observed, even though the prior keeps the
    /// posterior covariance tight.
    pub quality: f32,
    /// Fraction of solved shape (0 = template, 1 = frozen personal shape).
    pub shape_confidence: f32,
    /// Ready-to-use hand orientation confidence per hand `[left, right]`
    /// (from wrist σ) — for consumers that gate finger driving.
    pub hand_confidence: [f32; 2],
    /// Subject shoulder span (m) under the current shape — the metric
    /// scale reference for 1:1 root translation.
    pub shoulder_span_m: f32,
    /// Wrist position in the subject's own head-local frame, camera axes,
    /// per hand `[left, right]`. Rotation-only retarget preserves a
    /// reach's direction, not its endpoint, so a finger-to-lips pose
    /// lands short on a different-proportioned avatar; consumers
    /// conjugate this by the avatar head node's rest world rotation and
    /// anchor the avatar's wrist at the same head-relative spot
    /// (self-contact preservation). `[0;3]` when unavailable.
    pub head_local_wrists: [[f32; 3]; 2],
    /// Face-proximity weight per hand `[left, right]` in `[0,1]`: 1 with
    /// the wrist within `FACE_CONTACT_M` of the head, fading to 0 at
    /// `FACE_FADE_M`, scaled by that hand's confidence. The retarget
    /// scales the head-local anchoring by this, so far-from-face poses
    /// keep the pure rotation path untouched.
    pub face_proximity: [f32; 2],
}

/// Wrist-to-head distance at which face-contact anchoring is fully on.
pub const FACE_CONTACT_M: f64 = 0.30;
/// Wrist-to-head distance at which face-contact anchoring has faded to 0.
pub const FACE_FADE_M: f64 = 0.45;

fn face_proximity_weight(dist_m: f64) -> f32 {
    ((FACE_FADE_M - dist_m) / (FACE_FADE_M - FACE_CONTACT_M)).clamp(0.0, 1.0) as f32
}

/// Build the rig pose from the estimator posterior. `shoulder_span_px` is
/// the projected shoulder span (pixels) — a subject that is tiny in the
/// image (a poster, a bystander across the room) cannot be tracked at
/// avatar-driving fidelity and is reported with a low `quality` so the
/// retarget rests the avatar instead of flailing.
pub fn rig_pose(h: &Humanoid, est: &Estimator, t: f64, shoulder_span_px: Option<f64>) -> RigPose {
    let m = &h.model;
    let fk = m.fk(&est.state);
    let mut bones = HashMap::with_capacity(64);
    for (j, jd) in m.joints.iter().enumerate() {
        let Some(bone) = jd.bone else { continue };
        // Δ_V = Rx180 · R_w(j)   (see design: rest world = Rx180 · I)
        let dv = mat_mul(&CAM_TO_VIEW, &fk.r[j]);
        // σ of the bone's WORLD orientation (what the retarget applies):
        // propagated along the chain from the full posterior covariance.
        let sigma = est.joint_world_sigma(j);
        let mut data_sigma = est.joint_data_sigma(m, j);
        if j == h.j.l_elbow_twist {
            data_sigma = data_sigma.min(est.joint_data_sigma(m, h.j.l_elbow));
        }
        if j == h.j.r_elbow_twist {
            data_sigma = data_sigma.min(est.joint_data_sigma(m, h.j.r_elbow));
        }
        bones.insert(
            bone,
            RigBone {
                delta_world: mat_to_quat(&dv),
                sigma: sigma as f32,
                data_sigma: data_sigma as f32,
            },
        );
    }
    let torso_sigma = [h.j.spine1, h.j.spine2, h.j.spine3]
        .iter()
        .map(|&j| est.joint_sigma(m, j))
        .fold(0.0, f64::max)
        .max(est.var[..3].iter().cloned().fold(0.0, f64::max).sqrt());
    // Posterior σ alone cannot report "subject lost": with no
    // measurements the pose prior keeps the covariance tight while the
    // state random-walks (measured 2026-09-13: root drifting 0.9→3.9 m
    // over 45 s of n2d=n3d=ncloud=0 with quality steady at 0.5-0.7, so
    // the retarget kept driving the avatar on a runaway prior). Gate
    // quality on the trunk's measurement-only σ as well — the leaky
    // data-info EMA (~0.3 s) collapses it a few seconds after the last
    // real constraint, and well-observed frames (cloud + keypoints feed
    // the trunk) sit near 1.0, so normal tracking is unchanged.
    let spine_data_sigma = [h.j.spine1, h.j.spine2, h.j.spine3]
        .iter()
        .map(|&j| est.joint_data_sigma(m, j))
        .fold(0.0, f64::max);
    let data_support = confidence_from_sigma(spine_data_sigma, 0.08, 0.6);
    let mut quality = (1.0 - torso_sigma / 0.5).clamp(0.0, 1.0) as f32 * data_support;
    if let Some(px) = shoulder_span_px {
        // Full trust above ~60 px of shoulder span, none below ~25 px.
        quality *= ((px - 25.0) / 35.0).clamp(0.0, 1.0) as f32;
    }
    let shape_var = (m.beta_scale..m.num_params)
        .map(|k| est.var[k])
        .fold(0.0, f64::max);
    let shape_confidence = if est.shape_frozen {
        1.0
    } else {
        (1.0 - (shape_var / 0.0144)).clamp(0.0, 1.0) as f32
    };
    let hand_confidence = [
        (1.0 - est.joint_data_sigma(m, h.j.l_wrist) / 0.6).clamp(0.0, 1.0) as f32,
        (1.0 - est.joint_data_sigma(m, h.j.r_wrist) / 0.6).clamp(0.0, 1.0) as f32,
    ];
    let shoulder_span_m = norm(sub(fk.t[h.j.l_shoulder], fk.t[h.j.r_shoulder])) as f32;
    // Head-local wrist anchors for self-contact poses. The offset is
    // reported in the subject's own head-local frame on CAMERA axes:
    // o = R_cam_headᵀ · (wrist_cam − head_cam). The retarget conjugates
    // it by the avatar head node's rest world rotation before applying it
    // through the solved avatar head transform (see
    // `anchor_face_local_hands`), which reproduces the subject's
    // head-relative wrist placement on the avatar exactly, regardless of
    // the avatar's rest-pose or bone-axis conventions or the hips
    // tilt-clamp rebase.
    let mut head_local_wrists = [[0.0f32; 3]; 2];
    let mut face_proximity = [0.0f32; 2];
    let head_r = &fk.r[h.j.head];
    let head_t = fk.t[h.j.head];
    for (side, wrist) in [(0usize, h.j.l_wrist), (1usize, h.j.r_wrist)] {
        let d = sub(fk.t[wrist], head_t);
        let dist = norm(d);
        // Rᵀ · d, kept in camera axes.
        let o = mat_vec(&transpose(head_r), d);
        head_local_wrists[side] = [o[0] as f32, o[1] as f32, o[2] as f32];
        face_proximity[side] = face_proximity_weight(dist) * hand_confidence[side];
    }
    RigPose {
        t,
        bones,
        shoulder_span_m,
        head_local_wrists,
        face_proximity,
        root_cam_m: [
            est.state.root_t[0] as f32,
            est.state.root_t[1] as f32,
            est.state.root_t[2] as f32,
        ],
        root_sigma_m: est.root_sigma_m() as f32,
        quality,
        shape_confidence,
        hand_confidence,
    }
}

/// Confidence from a marginal σ (rad): 1 at σ≤σ_lo, 0 at σ≥σ_hi.
pub fn confidence_from_sigma(sigma: f64, lo: f64, hi: f64) -> f32 {
    ((hi - sigma) / (hi - lo)).clamp(0.0, 1.0) as f32
}

/// Compatibility skeleton: FK joint positions in the legacy source frame
/// (`−(cam − anchor)`, i.e. selfie-mirror x / y-up / z-toward-camera),
/// confidences from σ, metric frame info for the GUI. `intr` = colour
/// intrinsics of the frame.
pub fn source_skeleton(
    h: &Humanoid,
    est: &Estimator,
    frame_index: u64,
    intr: Option<CameraIntrinsics>,
    reference_span_m: f32,
) -> SourceSkeleton {
    let m = &h.model;
    let fk = m.fk(&est.state);
    let mut sk = SourceSkeleton::empty(frame_index);
    let anchor = est.state.root_t;
    let to_src = |p: V3| -> [f32; 3] {
        [
            -(p[0] - anchor[0]) as f32,
            -(p[1] - anchor[1]) as f32,
            -(p[2] - anchor[2]) as f32,
        ]
    };
    for (j, jd) in m.joints.iter().enumerate() {
        let Some(bone) = jd.bone else { continue };
        let sigma = est.joint_data_sigma(m, j).max(est.joint_sigma(m, j));
        let conf = confidence_from_sigma(sigma, 0.08, 0.6);
        let p = fk.t[j];
        sk.joints.insert(
            bone,
            SourceJoint {
                position: to_src(p),
                confidence: conf,
                metric_depth_m: Some(p[2] as f32),
            },
        );
    }
    // Fingertips (keyed by the distal bone).
    for hand in 0..2 {
        for f in 0..5 {
            let dip = h.j.finger[hand][f][3];
            let Some(bone) = m.joints[dip].bone else {
                continue;
            };
            let p = fk.site[h.s.tip[hand][f]];
            let sigma = est.joint_sigma(m, dip);
            sk.fingertips.insert(
                bone,
                SourceJoint {
                    position: to_src(p),
                    confidence: confidence_from_sigma(sigma, 0.08, 0.6),
                    metric_depth_m: Some(p[2] as f32),
                },
            );
        }
    }
    sk.overall_confidence = confidence_from_sigma(
        est.var[..6].iter().cloned().fold(0.0, f64::max).sqrt(),
        0.05,
        0.5,
    );
    sk.root_offset = Some([
        -(anchor[0] as f32),
        -(anchor[1] as f32),
        -(anchor[2] as f32),
    ]);
    sk.root_anchor_is_hip = true;
    if let Some(intr) = intr {
        sk.metric_frame_info = Some(MetricFrameInfo {
            anchor_cam_m: [anchor[0] as f32, anchor[1] as f32, anchor[2] as f32],
            anchor_is_hip: true,
            mpsu: 1.0,
            reference_span_m,
            intrinsics: intr,
        });
    }
    sk
}

/// Per-update published-rotation cap (degrees, world delta).
/// `VULVATAR_TRANSITION_MAX_STEP_DEG` overrides; 0 disables the limiter.
pub const TRANSITION_MAX_STEP_DEG_DEFAULT: f32 = 25.0;
/// Per-update cap (m) for the head-local wrist anchors. Desk hand motion
/// tops out ≈ 0.10 m/update at 20 Hz; 0.12 m leaves it untouched while
/// the measured 0.2-0.35 m yanks become 2-3 update glides.
const TRANSITION_WRIST_STEP_M: f32 = 0.12;
/// Publication gap (s) that clears the limiter's memory (person left /
/// re-seeded — the pose itself resets there, nothing to cap from).
const TRANSITION_MEMORY_GAP_S: f64 = 0.5;

/// Last published state of one bone (post-cap): the value the NEXT
/// frame's rotation step is measured against.
#[derive(Clone, Copy, Debug)]
struct PublishedBone {
    delta: [f32; 4],
    wrist: [f32; 3],
}

/// Rate-limits published bone rotations to a physical per-update ceiling.
///
/// Measured live (desk streaming, 2026-09-16): when a hand leaves the
/// frame or its detection drops, the solved wrist teleports 0.2-0.35 m
/// and the bone world-deltas jump 90-155° in a single ~50 ms update —
/// >1000°/s, physically impossible. `data_sigma` leaks out over ~0.3 s,
/// so those yanks mostly happen when BOTH frames already read
/// unobserved, and single mis-detection frames produce them while BOTH
/// read observed — no observation-state gate can catch the whole class
/// (measured: a flip-gated blender engaged zero times on replay456).
/// The limiter is therefore observation-agnostic: any per-update change
/// over `max_step` publishes only `max_step` of rotation toward the new
/// pose. Real motion under the cap passes untouched; motion over the
/// cap catches up within a couple of updates, reading as a fast glide.
/// The wrist anchors get the same treatment in position space.
pub struct PoseTransitionBlender {
    max_step: f32,
    /// Last published state per bone.
    last: HashMap<HumanoidBone, PublishedBone>,
    last_t: f64,
}

impl PoseTransitionBlender {
    pub fn new() -> Self {
        let deg = std::env::var("VULVATAR_TRANSITION_MAX_STEP_DEG")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(TRANSITION_MAX_STEP_DEG_DEFAULT);
        Self {
            max_step: deg.to_radians(),
            last: HashMap::new(),
            last_t: f64::NAN,
        }
    }

    /// Apply to the rig right before publication. With the step knob at
    /// 0 this is a pass-through that still maintains the per-bone memory,
    /// so toggling the knob at runtime is continuous.
    pub fn apply(&mut self, rig: &mut RigPose) {
        if !self.last_t.is_nan() && rig.t - self.last_t > TRANSITION_MEMORY_GAP_S {
            self.last.clear();
        }
        self.last_t = rig.t;
        for (bone, rb) in rig.bones.iter_mut() {
            let new = rb.delta_world;
            let side = hand_side(bone);
            let Some(prev) = self.last.get(bone).copied() else {
                let wrist = side.map(|i| rig.head_local_wrists[i]).unwrap_or([0.0; 3]);
                self.last.insert(
                    bone.clone(),
                    PublishedBone {
                        delta: new,
                        wrist,
                    },
                );
                continue;
            };
            if self.max_step > 0.0 {
                let angle = quat_angle_between(prev.delta, new);
                if angle > self.max_step {
                    rb.delta_world = quat_slerp_short2(prev.delta, new, self.max_step / angle);
                }
                if let Some(i) = side {
                    let target = rig.head_local_wrists[i];
                    let d = ((target[0] - prev.wrist[0]).powi(2)
                        + (target[1] - prev.wrist[1]).powi(2)
                        + (target[2] - prev.wrist[2]).powi(2))
                    .sqrt();
                    if d > TRANSITION_WRIST_STEP_M {
                        rig.head_local_wrists[i] =
                            lerp3(prev.wrist, target, TRANSITION_WRIST_STEP_M / d);
                    }
                }
            }
            let wrist = match side {
                Some(i) => rig.head_local_wrists[i],
                None => prev.wrist,
            };
            self.last.insert(
                bone.clone(),
                PublishedBone {
                    delta: rb.delta_world,
                    wrist,
                },
            );
        }
    }
}

/// 0 = left hand, 1 = right hand, None = not a hand (wrist) bone.
fn hand_side(bone: &HumanoidBone) -> Option<usize> {
    match bone {
        HumanoidBone::LeftHand => Some(0),
        HumanoidBone::RightHand => Some(1),
        _ => None,
    }
}

fn smoothstep01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Rotation angle (rad) between two quaternions, hemisphere-agnostic.
fn quat_angle_between(a: [f32; 4], b: [f32; 4]) -> f32 {
    let mut d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    if d < 0.0 {
        d = -d;
    }
    2.0 * d.clamp(-1.0, 1.0).acos()
}

/// Short-arc slerp for `[x, y, z, w]` quaternions (nearly-parallel safe).
fn quat_slerp_short2(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let b = if dot < 0.0 {
        dot = -dot;
        [-b[0], -b[1], -b[2], -b[3]]
    } else {
        b
    };
    if dot > 0.9995 {
        return [
            a[0] + t * (b[0] - a[0]),
            a[1] + t * (b[1] - a[1]),
            a[2] + t * (b[2] - a[2]),
            a[3] + t * (b[3] - a[3]),
        ];
    }
    let theta = dot.clamp(-1.0, 1.0).acos();
    let sin_t = theta.sin();
    let wa = (theta * (1.0 - t)).sin() / sin_t;
    let wb = (theta * t).sin() / sin_t;
    let q = [
        a[0] * wa + b[0] * wb,
        a[1] * wa + b[1] * wb,
        a[2] * wa + b[2] * wb,
        a[3] * wa + b[3] * wb,
    ];
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n > 1e-8 {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    } else {
        q
    }
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

#[cfg(test)]
mod transition_blend_tests {
    use super::*;

    fn rig(t: f64, delta: [f32; 4], wrist: [f32; 3]) -> RigPose {
        let mut r = RigPose::default();
        r.t = t;
        r.bones.insert(
            HumanoidBone::LeftHand,
            RigBone {
                delta_world: delta,
                sigma: 0.1,
                data_sigma: 10.0,
            },
        );
        r.head_local_wrists = [wrist, [0.0; 3]];
        r
    }

    const IDENTITY: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
    const QUARTER_TURN: [f32; 4] = [0.0, 0.3826834, 0.0, 0.9238795];
    /// 10° — a real rotation change too small to hit the cap.
    const SMALL_TURN: [f32; 4] = [0.0, 0.0871557, 0.0, 0.9961947];

    fn limiter() -> PoseTransitionBlender {
        let mut b = PoseTransitionBlender::new();
        b.max_step = 25.0f32.to_radians();
        b
    }

    #[test]
    fn jump_over_cap_spreads_over_updates_and_converges() {
        let mut b = limiter();
        b.apply(&mut rig(1.0, IDENTITY, [0.1, 0.2, 0.3]));
        // 45° yank in one update: first frame publishes exactly the cap.
        let mut r = rig(1.05, QUARTER_TURN, [0.5, 0.2, 0.3]);
        b.apply(&mut r);
        let a1 = quat_angle_between(r.bones[&HumanoidBone::LeftHand].delta_world, IDENTITY);
        assert!((a1 - 25.0f32.to_radians()).abs() < 1e-3, "first frame {a1} rad");
        // Remaining 20° is under the cap: next frame reaches the target.
        let mut r = rig(1.1, QUARTER_TURN, [0.5, 0.2, 0.3]);
        b.apply(&mut r);
        assert!(
            quat_angle_between(r.bones[&HumanoidBone::LeftHand].delta_world, QUARTER_TURN) < 1e-3,
            "under-cap remainder should pass through to the target"
        );
    }

    #[test]
    fn change_under_cap_passes_through() {
        let mut b = limiter();
        b.apply(&mut rig(1.0, IDENTITY, [0.1, 0.2, 0.3]));
        let mut r = rig(1.05, SMALL_TURN, [0.1, 0.2, 0.3]);
        b.apply(&mut r);
        assert_eq!(r.bones[&HumanoidBone::LeftHand].delta_world, SMALL_TURN);
    }

    #[test]
    fn wrist_anchor_is_capped_in_position_space() {
        let mut b = limiter();
        b.apply(&mut rig(1.0, IDENTITY, [0.0, 0.0, 0.0]));
        // Small rotation (passes the rotation cap) but a 1 m wrist jump.
        let mut r = rig(1.05, SMALL_TURN, [1.0, 0.0, 0.0]);
        b.apply(&mut r);
        let w = r.head_local_wrists[0];
        let d = (w[0].powi(2) + w[1].powi(2) + w[2].powi(2)).sqrt();
        assert!(
            (d - TRANSITION_WRIST_STEP_M).abs() < 1e-3,
            "wrist should advance exactly the cap, got {d}"
        );
        // Catch-up moves one cap per update until it lands on the target.
        let mut x = TRANSITION_WRIST_STEP_M;
        for i in 2.. {
            let mut r = rig(1.0 + 0.05 * i as f64, SMALL_TURN, [1.0, 0.0, 0.0]);
            b.apply(&mut r);
            x = (x + TRANSITION_WRIST_STEP_M).min(1.0);
            assert!((r.head_local_wrists[0][0] - x).abs() < 1e-4, "frame {i}");
            if x >= 1.0 {
                break;
            }
        }
    }

    #[test]
    fn publication_gap_clears_memory() {
        let mut b = limiter();
        b.apply(&mut rig(1.0, IDENTITY, [0.1, 0.2, 0.3]));
        // Person left and came back: first new frame publishes as-is.
        let mut r = rig(2.5, QUARTER_TURN, [0.5, 0.2, 0.3]);
        b.apply(&mut r);
        assert_eq!(r.bones[&HumanoidBone::LeftHand].delta_world, QUARTER_TURN);
    }

    #[test]
    fn zero_step_disables_limiting() {
        let mut b = PoseTransitionBlender::new();
        b.max_step = 0.0;
        b.apply(&mut rig(1.0, IDENTITY, [0.1, 0.2, 0.3]));
        let mut r = rig(1.05, QUARTER_TURN, [0.5, 0.2, 0.3]);
        b.apply(&mut r);
        assert_eq!(r.bones[&HumanoidBone::LeftHand].delta_world, QUARTER_TURN);
    }

    #[test]
    fn first_frame_establishes_baseline_without_limiting() {
        let mut b = limiter();
        // Even a huge first delta passes through: nothing to cap from.
        let mut r = rig(1.0, QUARTER_TURN, [0.5, 0.2, 0.3]);
        b.apply(&mut r);
        assert_eq!(r.bones[&HumanoidBone::LeftHand].delta_world, QUARTER_TURN);
    }
}
