//! Estimator posterior → [`RigPose`] (what the avatar retarget consumes)
//! and a compatibility [`SourceSkeleton`] (what the GUI overlay,
//! calibration modal, debug channel and expression solver still read).

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
    /// Overall track quality in `[0,1]` (1 = torso well constrained).
    pub quality: f32,
    /// Fraction of solved shape (0 = template, 1 = frozen personal shape).
    pub shape_confidence: f32,
    /// Ready-to-use hand orientation confidence per hand `[left, right]`
    /// (from wrist σ) — for consumers that gate finger driving.
    pub hand_confidence: [f32; 2],
    /// Subject shoulder span (m) under the current shape — the metric
    /// scale reference for 1:1 root translation.
    pub shoulder_span_m: f32,
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
    let mut quality = (1.0 - torso_sigma / 0.5).clamp(0.0, 1.0) as f32;
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
    RigPose {
        t,
        bones,
        shoulder_span_m,
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
            let Some(bone) = m.joints[dip].bone else { continue };
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
    sk.root_offset = Some([-(anchor[0] as f32), -(anchor[1] as f32), -(anchor[2] as f32)]);
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
