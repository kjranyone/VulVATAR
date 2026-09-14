//! Tracking-v2 retarget: [`RigPose`] (world-delta joint rotations in the
//! viewer frame + pelvis position + per-joint σ) → avatar local bone
//! transforms.
//!
//! Every driven bone's *world* rotation is set to `Δ_V · rest_world`, i.e.
//! the same rotation the subject's corresponding segment made from its own
//! T-pose rest, so body-proportion differences never bias a reach's
//! direction (there is no global position matching). Two bounded
//! positional corrections run after the rotation pass, each fading to the
//! pure rotation result outside its zone: hand-cross prevention, and
//! face-contact anchoring (the wrist is placed at the subject's
//! head-relative position when the hand is near the face, so
//! self-contact poses like finger-to-lips survive proportion
//! differences — measured on the composite bench, rotation-only
//! transfer landed the fingertip a quarter head-height short).
//! Local rotations are recovered top-down against the already-solved
//! parents.
//!
//! Display smoothing is a dt-aware slerp toward the target (the user's
//! `rotation_blend` setting) from the previous frame's solved local
//! rotation; the estimator's own dynamics already produce a smooth 30 Hz
//! signal, so this only bridges render frames between tracking samples.

use std::collections::HashMap;

use crate::asset::HumanoidBone as HB;
use crate::asset::{HumanoidBone, HumanoidMap, NodeId, SkeletonAsset, Transform};
use crate::math_utils::{
    quat_conjugate, quat_mul, quat_normalize, quat_rotate_vec3, vec3_length, vec3_sub, Quat, Vec3,
};
use crate::tracking::fusion::output::RigPose;

/// Persistent retarget state per avatar instance.
#[derive(Default, Clone, Debug)]
pub struct RetargetState {
    /// Previous frame's solved local rotation per bone (display smoothing).
    prev_local: HashMap<HumanoidBone, Quat>,
    prev_hips_translation: Option<Vec3>,
    /// Session anchor: pelvis camera position the avatar's rest position
    /// corresponds to. Seeded from the first well-tracked second, then
    /// frozen (1:1 metric mirror — every later move persists).
    anchor_cam: Option<[f32; 3]>,
    anchor_seed: Vec<[f32; 3]>,
    anchor_seed_start: Option<f64>,
    /// Cached rest world rotations / positions of every node (keyed by node
    /// count to detect a swapped avatar).
    pub(crate) rest_world_rot: Vec<Quat>,
    pub(crate) rest_world_pos: Vec<Vec3>,
    rest_cache_len: usize,
    /// A-pose display-rest locals for the arm bones (see `avatar::relax`),
    /// rebuilt together with the rest cache. Bones relaxing through the
    /// quality / σ gates target these instead of the T-pose bind.
    pub(crate) arelax_local: crate::avatar::relax::APoseOverlay,
    /// 1€ adaptive filter state per bone rotation.
    one_euro_bones: HashMap<HumanoidBone, OneEuroQuat>,
    /// 1€ adaptive filter state for hips root translation.
    one_euro_hips_t: OneEuroVec3,
}

impl RetargetState {
    pub fn reset(&mut self) {
        self.prev_local.clear();
        self.prev_hips_translation = None;
        self.anchor_cam = None;
        self.anchor_seed.clear();
        self.anchor_seed_start = None;
        self.one_euro_bones.clear();
        self.one_euro_hips_t.reset();
    }
    /// Forget only the display-smoothing state (avatar swap).
    pub fn reset_smoothing(&mut self) {
        self.prev_local.clear();
        self.prev_hips_translation = None;
        self.rest_cache_len = 0;
        self.arelax_local.clear();
        self.one_euro_bones.clear();
        self.one_euro_hips_t.reset();
    }

    /// Seed the display smoothing from an externally-produced pose (the
    /// idle A-pose relax) so the next tracked frame's slerp starts from
    /// what is on screen instead of the bind rest.
    pub fn seed_display_smoothing(&mut self, humanoid: &HumanoidMap, locals: &[Transform]) {
        for bone in ORDER {
            let Some(&NodeId(node)) = humanoid.bone_map.get(&bone) else {
                continue;
            };
            if let Some(t) = locals.get(node as usize) {
                self.prev_local.insert(bone, t.rotation);
            }
        }
        if let Some(NodeId(hips)) = humanoid.bone_map.get(&HumanoidBone::Hips) {
            if let Some(t) = locals.get(*hips as usize) {
                self.prev_hips_translation = Some(t.translation);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RetargetParams {
    /// Display blend per reference frame (same semantics as the pose
    /// solver's `rotation_blend`).
    pub rotation_blend: f32,
    pub root_translation_enabled: bool,
    pub hand_tracking_enabled: bool,
    pub lower_body_tracking_enabled: bool,
    /// Bones whose joint σ exceeds this are left at rest (rad).
    pub sigma_rest: f32,
    /// Bones whose measurement-only σ (`RigBone::data_sigma`) exceeds
    /// this are left at rest even when the posterior σ looks confident:
    /// with no recent measurements the posterior stays tight (the pose
    /// prior dominates an information-starved solve) while the state
    /// random-walks — measured 2026-09-13 on a face-only stretch (body
    /// detector lost, FaceMesh alive): arm data-σ pinned at 10 rad while
    /// the posterior read 0.3-0.9, and the wrists teleported ±1.5 m for
    /// minutes. Observed joints sit at ~0.001-0.05 rad; the leaky
    /// data-info EMA (~0.3 s) crosses this bound a couple of seconds
    /// after a joint stops being measured.
    pub sigma_data_rest: f32,
    /// Rig quality below which nothing is driven (subject lost / tiny).
    pub min_quality: f32,
    /// Seconds of well-tracked data used to seed the root anchor.
    pub anchor_seed_s: f64,
    /// Maximum whole-body lean (rad) the hips may carry; the rest of the
    /// estimator's pelvis tilt is treated as pelvis/spine-split wander (or
    /// camera mounting pitch) and absorbed. Body-internal articulation is
    /// unaffected — bones are re-expressed relative to the hips.
    pub max_root_tilt: f32,
    /// Prevent hands from crossing each other in front of the body (Two-Bone IK anti-cross).
    pub hand_cross_prevention: bool,
    /// Anchor the wrist at the subject's head-relative position when the
    /// hand is near the face (`rig.face_proximity`), so self-contact
    /// poses (finger to lips) survive avatar proportion differences.
    pub face_contact_anchoring: bool,
    /// Minimum lateral distance (m) between left and right hand centers when touching.
    pub min_hand_distance: f32,
    /// Enable 1€ (One Euro) adaptive smoothing filter for rotation and root translation.
    pub one_euro_enabled: bool,
    /// Minimum cutoff frequency in Hz for 1€ filter (stillness jitter suppression). Default: 1.0
    pub one_euro_min_cutoff: f32,
    /// Speed sensitivity coefficient β for 1€ filter (higher = less lag during fast motion). Default: 1.0
    pub one_euro_beta: f32,
    /// Cutoff frequency in Hz for derivative filtering in 1€ filter. Default: 1.0
    pub one_euro_d_cutoff: f32,
}

impl Default for RetargetParams {
    fn default() -> Self {
        Self {
            // Tracking-lag calibrated (ARMCMP audit, s1789311387, 2026-09-13):
            // at the old 0.35 a fast left reach landed up to 15.2° behind
            // its source with 6.7% of audited frames >13°; β 2 + blend 0.6
            // removes the >13° tail entirely (max 11.4°) while the still
            // arm's frame-to-frame jitter DROPS (med 4.69 → 3.94°) — less
            // lag is less noise there too.
            rotation_blend: std::env::var("VULVATAR_RETARGET_BLEND")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|v| *v > 0.0)
                .unwrap_or(0.6),
            root_translation_enabled: true,
            hand_tracking_enabled: true,
            lower_body_tracking_enabled: true,
            sigma_rest: 1.2,
            sigma_data_rest: 0.6,
            min_quality: 0.2,
            anchor_seed_s: 1.0,
            max_root_tilt: 0.35,
            hand_cross_prevention: true,
            face_contact_anchoring: true,
            min_hand_distance: 0.08,
            one_euro_enabled: true,
            one_euro_min_cutoff: 1.0,
            // Speed coefficient: higher tracks fast motion with less lag
            // (calibrated with rotation_blend above).
            one_euro_beta: std::env::var("VULVATAR_RETARGET_BETA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2.0),
            one_euro_d_cutoff: 1.0,
        }
    }
}

const REFERENCE_DT: f32 = 1.0 / 30.0;

fn dt_aware_blend(slider_blend: f32, dt: f32) -> f32 {
    if dt <= 0.0 {
        return 0.0;
    }
    let b = slider_blend.clamp(0.0, 1.0);
    if b >= 1.0 {
        return 1.0;
    }
    if b <= 0.0 {
        return 0.0;
    }
    let tau = -REFERENCE_DT / (1.0 - b).ln();
    1.0 - (-dt / tau).exp()
}

pub(crate) fn slerp_short(a: &Quat, b: &Quat, t: f32) -> Quat {
    let mut d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let mut b2 = *b;
    if d < 0.0 {
        d = -d;
        b2 = [-b[0], -b[1], -b[2], -b[3]];
    }
    if d > 0.9995 {
        return quat_normalize(&[
            a[0] + t * (b2[0] - a[0]),
            a[1] + t * (b2[1] - a[1]),
            a[2] + t * (b2[2] - a[2]),
            a[3] + t * (b2[3] - a[3]),
        ]);
    }
    let th0 = d.clamp(-1.0, 1.0).acos();
    let th = th0 * t;
    let s0 = (th0 - th).sin() / th0.sin();
    let s1 = th.sin() / th0.sin();
    quat_normalize(&[
        s0 * a[0] + s1 * b2[0],
        s0 * a[1] + s1 * b2[1],
        s0 * a[2] + s1 * b2[2],
        s0 * a[3] + s1 * b2[3],
    ])
}

fn compute_alpha(fc: f32, dt: f32) -> f32 {
    let r = 2.0 * std::f32::consts::PI * fc * dt;
    (r / (1.0 + r)).clamp(0.0, 1.0)
}

/// 1D Low-Pass Filter for derivative estimation in 1€ filter.
#[derive(Clone, Copy, Debug, Default)]
pub struct LowPassFilter1D {
    hat: Option<f32>,
}

impl LowPassFilter1D {
    pub fn reset(&mut self) {
        self.hat = None;
    }

    pub fn filter(&mut self, val: f32, alpha: f32) -> f32 {
        let next = match self.hat {
            Some(prev) => prev + alpha * (val - prev),
            None => val,
        };
        self.hat = Some(next);
        next
    }
}

/// 1€ adaptive filter for 3D position vectors (e.g. Hips root translation).
#[derive(Clone, Copy, Debug, Default)]
pub struct OneEuroVec3 {
    prev_filtered: Option<Vec3>,
    d_hat: LowPassFilter1D,
}

impl OneEuroVec3 {
    pub fn reset(&mut self) {
        self.prev_filtered = None;
        self.d_hat.reset();
    }

    pub fn set_filtered(&mut self, p: Vec3) {
        self.prev_filtered = Some(p);
    }

    pub fn filter(
        &mut self,
        target: Vec3,
        dt: f32,
        min_cutoff: f32,
        beta: f32,
        d_cutoff: f32,
    ) -> Vec3 {
        let prev = match self.prev_filtered {
            Some(p) => p,
            None => {
                self.prev_filtered = Some(target);
                return target;
            }
        };

        if dt <= 1e-5 {
            return prev;
        }

        let speed = vec3_length(&vec3_sub(&target, &prev)) / dt;
        let alpha_d = compute_alpha(d_cutoff, dt);
        let speed_hat = self.d_hat.filter(speed, alpha_d);

        let fc = (min_cutoff + beta * speed_hat).max(1e-3);
        let alpha = compute_alpha(fc, dt);

        let filtered = [
            prev[0] + alpha * (target[0] - prev[0]),
            prev[1] + alpha * (target[1] - prev[1]),
            prev[2] + alpha * (target[2] - prev[2]),
        ];
        self.prev_filtered = Some(filtered);
        filtered
    }
}

/// 1€ adaptive filter for Unit Quaternions (SO(3) rotations).
#[derive(Clone, Copy, Debug, Default)]
pub struct OneEuroQuat {
    prev_filtered: Option<Quat>,
    d_hat: LowPassFilter1D,
}

impl OneEuroQuat {
    pub fn reset(&mut self) {
        self.prev_filtered = None;
        self.d_hat.reset();
    }

    pub fn set_filtered(&mut self, q: Quat) {
        self.prev_filtered = Some(q);
    }

    pub fn filter(
        &mut self,
        target: Quat,
        dt: f32,
        min_cutoff: f32,
        beta: f32,
        d_cutoff: f32,
    ) -> Quat {
        let prev = match self.prev_filtered {
            Some(q) => q,
            None => {
                let norm = quat_normalize(&target);
                self.prev_filtered = Some(norm);
                return norm;
            }
        };

        if dt <= 1e-5 {
            return prev;
        }

        // Angular distance theta = 2 * acos(|prev · target|)
        let dot =
            (prev[0] * target[0] + prev[1] * target[1] + prev[2] * target[2] + prev[3] * target[3])
                .abs()
                .clamp(0.0, 1.0);
        let theta = 2.0 * dot.acos();
        let omega = theta / dt; // rad/s

        let alpha_d = compute_alpha(d_cutoff, dt);
        let omega_hat = self.d_hat.filter(omega, alpha_d);

        let fc = (min_cutoff + beta * omega_hat).max(1e-3);
        let alpha = compute_alpha(fc, dt);

        let filtered = slerp_short(&prev, &target, alpha);
        self.prev_filtered = Some(filtered);
        filtered
    }
}

/// Hierarchical driving order (parents before children).
const ORDER: [HumanoidBone; 50] = [
    HumanoidBone::Hips,
    HumanoidBone::Spine,
    HumanoidBone::Chest,
    HumanoidBone::UpperChest,
    HumanoidBone::Neck,
    HumanoidBone::Head,
    HumanoidBone::LeftShoulder,
    HumanoidBone::LeftUpperArm,
    HumanoidBone::LeftLowerArm,
    HumanoidBone::LeftHand,
    HumanoidBone::RightShoulder,
    HumanoidBone::RightUpperArm,
    HumanoidBone::RightLowerArm,
    HumanoidBone::RightHand,
    HumanoidBone::LeftUpperLeg,
    HumanoidBone::LeftLowerLeg,
    HumanoidBone::LeftFoot,
    HumanoidBone::RightUpperLeg,
    HumanoidBone::RightLowerLeg,
    HumanoidBone::RightFoot,
    HumanoidBone::LeftThumbProximal,
    HumanoidBone::LeftThumbIntermediate,
    HumanoidBone::LeftThumbDistal,
    HumanoidBone::LeftIndexProximal,
    HumanoidBone::LeftIndexIntermediate,
    HumanoidBone::LeftIndexDistal,
    HumanoidBone::LeftMiddleProximal,
    HumanoidBone::LeftMiddleIntermediate,
    HumanoidBone::LeftMiddleDistal,
    HumanoidBone::LeftRingProximal,
    HumanoidBone::LeftRingIntermediate,
    HumanoidBone::LeftRingDistal,
    HumanoidBone::LeftLittleProximal,
    HumanoidBone::LeftLittleIntermediate,
    HumanoidBone::LeftLittleDistal,
    HumanoidBone::RightThumbProximal,
    HumanoidBone::RightThumbIntermediate,
    HumanoidBone::RightThumbDistal,
    HumanoidBone::RightIndexProximal,
    HumanoidBone::RightIndexIntermediate,
    HumanoidBone::RightIndexDistal,
    HumanoidBone::RightMiddleProximal,
    HumanoidBone::RightMiddleIntermediate,
    HumanoidBone::RightMiddleDistal,
    HumanoidBone::RightRingProximal,
    HumanoidBone::RightRingIntermediate,
    HumanoidBone::RightRingDistal,
    HumanoidBone::RightLittleProximal,
    HumanoidBone::RightLittleIntermediate,
    HumanoidBone::RightLittleDistal,
];

fn is_finger(b: HumanoidBone) -> bool {
    !matches!(
        b,
        HumanoidBone::Hips
            | HumanoidBone::Spine
            | HumanoidBone::Chest
            | HumanoidBone::UpperChest
            | HumanoidBone::Neck
            | HumanoidBone::Head
            | HumanoidBone::LeftShoulder
            | HumanoidBone::LeftUpperArm
            | HumanoidBone::LeftLowerArm
            | HumanoidBone::LeftHand
            | HumanoidBone::RightShoulder
            | HumanoidBone::RightUpperArm
            | HumanoidBone::RightLowerArm
            | HumanoidBone::RightHand
            | HumanoidBone::LeftUpperLeg
            | HumanoidBone::LeftLowerLeg
            | HumanoidBone::LeftFoot
            | HumanoidBone::RightUpperLeg
            | HumanoidBone::RightLowerLeg
            | HumanoidBone::RightFoot
    )
}
fn is_leg(b: HumanoidBone) -> bool {
    matches!(
        b,
        HumanoidBone::LeftUpperLeg
            | HumanoidBone::LeftLowerLeg
            | HumanoidBone::LeftFoot
            | HumanoidBone::RightUpperLeg
            | HumanoidBone::RightLowerLeg
            | HumanoidBone::RightFoot
    )
}

/// Fill the cached rest world rotations / positions (rebuilding when the
/// node count changes = swapped avatar) and derive the A-pose display
/// rest overlay from them. `humanoid` is `None` for non-humanoid rigs,
/// which simply get no overlay.
pub(crate) fn ensure_rest_cache(
    state: &mut RetargetState,
    skeleton: &SkeletonAsset,
    humanoid: Option<&HumanoidMap>,
) {
    if state.rest_cache_len == skeleton.nodes.len() && !state.rest_world_rot.is_empty() {
        return;
    }
    let n = skeleton.nodes.len();
    let mut rot = vec![[0.0, 0.0, 0.0, 1.0]; n];
    let mut pos = vec![[0.0, 0.0, 0.0]; n];
    // Nodes are not guaranteed to be parent-first; iterate from roots.
    let mut stack: Vec<usize> = skeleton.root_nodes.iter().map(|r| r.0 as usize).collect();
    while let Some(i) = stack.pop() {
        if i >= n {
            continue;
        }
        let (pr, pp) = match skeleton.nodes[i].parent {
            Some(NodeId(p)) => (rot[p as usize], pos[p as usize]),
            None => ([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
        };
        let l = &skeleton.nodes[i].rest_local;
        rot[i] = quat_normalize(&quat_mul(&pr, &l.rotation));
        let off = quat_rotate_vec3(&pr, &l.translation);
        pos[i] = [pp[0] + off[0], pp[1] + off[1], pp[2] + off[2]];
        for c in &skeleton.nodes[i].children {
            stack.push(c.0 as usize);
        }
    }
    state.arelax_local = humanoid
        .map(|hm| crate::avatar::relax::a_pose_overlay(skeleton, hm, &rot, &pos))
        .unwrap_or_default();
    state.rest_world_rot = rot;
    state.rest_world_pos = pos;
    state.rest_cache_len = n;
}

/// World rotation of node `i` under the current `locals` (walks up the
/// parent chain; the chain is short).
fn world_rot(skeleton: &SkeletonAsset, locals: &[Transform], mut i: usize) -> Quat {
    let mut chain: Vec<usize> = Vec::with_capacity(16);
    loop {
        chain.push(i);
        match skeleton.nodes[i].parent {
            Some(NodeId(p)) => i = p as usize,
            None => break,
        }
    }
    let mut q = [0.0, 0.0, 0.0, 1.0];
    for &k in chain.iter().rev() {
        q = quat_mul(&q, &locals[k].rotation);
    }
    quat_normalize(&q)
}

/// Shortest-arc rotation taking unit vector `a` onto unit vector `b`.
fn quat_from_to(a: Vec3, b: Vec3) -> Quat {
    let c = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let w = (1.0 + d).max(1e-6);
    quat_normalize(&[c[0], c[1], c[2], w])
}

/// Apply the rig pose. `dt` = render-side seconds since the last call.
pub fn apply_rig_pose(
    rig: &RigPose,
    skeleton: &SkeletonAsset,
    humanoid: &HumanoidMap,
    local_transforms: &mut [Transform],
    params: &RetargetParams,
    state: &mut RetargetState,
    dt: f32,
) {
    ensure_rest_cache(state, skeleton, Some(humanoid));
    let blend = dt_aware_blend(params.rotation_blend, dt);

    // ---- upright root: yaw-only hips with a clamped lean ----------------------
    // With the legs unobserved (the desk envelope) the estimator's
    // pelvis-vs-spine pitch split is prior-determined and free to wander,
    // and any camera mounting pitch lands on the whole body. A VTuber
    // avatar must stay upright: split the hips world delta into
    // tilt ∘ yaw, clamp the tilt to `max_root_tilt`, and re-express every
    // other bone RELATIVE to the hips so body-internal articulation
    // (head vs torso, arms vs chest) is preserved exactly.
    let (hips_fix, hips_full): (Quat, Quat) = {
        match rig.bones.get(&HB::Hips) {
            Some(rb) if rig.quality >= params.min_quality => {
                let dh = quat_normalize(&rb.delta_world);
                let up = quat_rotate_vec3(&dh, &[0.0, 1.0, 0.0]);
                let tilt = quat_from_to([0.0, 1.0, 0.0], up);
                let yaw = quat_normalize(&quat_mul(&quat_conjugate(&tilt), &dh));
                // Clamp the tilt angle.
                let ang = 2.0 * tilt[3].clamp(-1.0, 1.0).acos();
                let ang = if ang > std::f32::consts::PI {
                    2.0 * std::f32::consts::PI - ang
                } else {
                    ang
                };
                let clamped = if ang > params.max_root_tilt && ang > 1e-4 {
                    let t = params.max_root_tilt / ang;
                    slerp_short(&[0.0, 0.0, 0.0, 1.0], &tilt, t)
                } else {
                    tilt
                };
                (quat_normalize(&quat_mul(&clamped, &yaw)), dh)
            }
            _ => ([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]),
        }
    };
    // Δ' = ΔH_clamped ∘ ΔH⁻¹ ∘ Δ_bone
    let rebase = quat_mul(&hips_fix, &quat_conjugate(&hips_full));

    for bone in ORDER {
        let Some(NodeId(node)) = humanoid.bone_map.get(&bone).copied() else {
            continue;
        };
        let node = node as usize;
        if node >= local_transforms.len() {
            continue;
        }
        let rest_local_rot = skeleton.nodes[node].rest_local.rotation;
        let is_tracked = rig.quality >= params.min_quality
            && (params.hand_tracking_enabled || !is_finger(bone))
            && (params.lower_body_tracking_enabled || !is_leg(bone));

        // Target local rotation. A low-quality rig (subject lost / tiny)
        // drives nothing: every bone relaxes toward the display rest
        // (A-pose for the arms, bind otherwise). A bone with no recent
        // measurements (data-σ grown past `sigma_data_rest`) also rests
        // — its posterior σ is prior-dominated and cannot be trusted
        // however confident it looks.
        let (target, tracking_valid) = match rig.bones.get(&bone) {
            Some(rb)
                if is_tracked
                    && rb.sigma <= params.sigma_rest
                    && rb.data_sigma <= params.sigma_data_rest =>
            {
                let corrected = quat_mul(&rebase, &rb.delta_world);
                let desired_world = quat_mul(&corrected, &state.rest_world_rot[node]);
                let parent_world = match skeleton.nodes[node].parent {
                    Some(NodeId(p)) => world_rot(skeleton, local_transforms, p as usize),
                    None => [0.0, 0.0, 0.0, 1.0],
                };
                (
                    quat_normalize(&quat_mul(&quat_conjugate(&parent_world), &desired_world)),
                    true,
                )
            }
            _ => (
                state
                    .arelax_local
                    .get(&bone)
                    .map(|(_, q)| *q)
                    .unwrap_or(rest_local_rot),
                false,
            ),
        };

        // 1€ filter smooths tracking jitter adaptively when tracked;
        // when tracking is lost or gated by σ, reset filter state so
        // re-acquisition never drags from past positions.
        let filtered_target = if tracking_valid && params.one_euro_enabled {
            state.one_euro_bones.entry(bone).or_default().filter(
                target,
                dt,
                params.one_euro_min_cutoff,
                params.one_euro_beta,
                params.one_euro_d_cutoff,
            )
        } else {
            state.one_euro_bones.remove(&bone);
            target
        };

        let prev = state
            .prev_local
            .get(&bone)
            .copied()
            .unwrap_or(rest_local_rot);
        let out = if blend >= 1.0 {
            filtered_target
        } else {
            slerp_short(&prev, &filtered_target, blend)
        };
        state.prev_local.insert(bone, out);
        local_transforms[node].rotation = out;
    }

    // ---- root translation -----------------------------------------------------
    if let Some(NodeId(hips)) = humanoid.bone_map.get(&HumanoidBone::Hips).copied() {
        let hips = hips as usize;
        if hips < local_transforms.len() {
            let rest_pos = skeleton.nodes[hips].rest_local.translation;
            // Anchor seeding: median of the first well-tracked second.
            if state.anchor_cam.is_none() && rig.quality > 0.5 && rig.root_sigma_m < 0.08 {
                let start = *state.anchor_seed_start.get_or_insert(rig.t);
                state.anchor_seed.push(rig.root_cam_m);
                if rig.t - start >= params.anchor_seed_s && state.anchor_seed.len() >= 5 {
                    let mut med = [0.0f32; 3];
                    for (k, m) in med.iter_mut().enumerate() {
                        let mut v: Vec<f32> = state.anchor_seed.iter().map(|p| p[k]).collect();
                        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        *m = v[v.len() / 2];
                    }
                    state.anchor_cam = Some(med);
                    state.anchor_seed.clear();
                }
            }
            let hips_tracked = params.root_translation_enabled && rig.quality >= params.min_quality;
            let (target, tracking_valid) = match (state.anchor_cam, hips_tracked) {
                (Some(anchor), true) => {
                    // camera → viewer: Rx(180°) = (x, −y, −z); scaled to the
                    // avatar's size so a 10 cm real step is a proportional
                    // avatar step.
                    let scale = avatar_shoulder_span(skeleton, humanoid, state)
                        .map(|av| {
                            if rig.shoulder_span_m > 0.05 {
                                av / rig.shoulder_span_m
                            } else {
                                1.0
                            }
                        })
                        .unwrap_or(1.0);
                    let d_view = [
                        (rig.root_cam_m[0] - anchor[0]) * scale,
                        -(rig.root_cam_m[1] - anchor[1]) * scale,
                        -(rig.root_cam_m[2] - anchor[2]) * scale,
                    ];
                    let d = d_view;
                    (
                        [rest_pos[0] + d[0], rest_pos[1] + d[1], rest_pos[2] + d[2]],
                        true,
                    )
                }
                _ => (rest_pos, false),
            };

            let filtered_target = if tracking_valid && params.one_euro_enabled {
                state.one_euro_hips_t.filter(
                    target,
                    dt,
                    params.one_euro_min_cutoff,
                    params.one_euro_beta,
                    params.one_euro_d_cutoff,
                )
            } else {
                state.one_euro_hips_t.reset();
                target
            };

            let prev = state.prev_hips_translation.unwrap_or(rest_pos);
            let out = [
                prev[0] + blend * (filtered_target[0] - prev[0]),
                prev[1] + blend * (filtered_target[1] - prev[1]),
                prev[2] + blend * (filtered_target[2] - prev[2]),
            ];
            state.prev_hips_translation = Some(out);
            local_transforms[hips].translation = out;
        }
    }

    // ---- self-contact anchoring (head-local wrist, Two-Bone IK) ----------
    anchor_face_local_hands(skeleton, humanoid, local_transforms, rig, params, state);

    // ---- hand cross prevention (Two-Bone IK) ---------------------------------
    prevent_hand_crossing(skeleton, humanoid, local_transforms, state, params);
}

/// Place the avatar's wrist at the subject's head-relative wrist position
/// when the subject's hand is near their own face. Rotation-only transfer
/// preserves a reach's direction, not its endpoint, so on a
/// different-proportioned avatar a finger-to-lips pose lands short
/// (measured on the composite bench: fingertip at the chin, a quarter
/// head-height low). The rig reports the wrist offset in the subject's
/// head-local frame (camera axes); conjugating it by the avatar head
/// node's rest world rotation expresses it in the same head-local frame
/// the solved `h_rot` lives in, because the rotation path realises
/// `h_rot ≈ rebase · Rx180 · R_cam_head · rest_world_head`, so
/// `h_rot · (rest_world_head⁻¹ · o) = rebase · Rx180 · R_cam_head · o` —
/// the subject's offset mapped into this avatar's solved frame regardless
/// of the avatar's rest-pose or bone-axis conventions (an FBX head bone
/// with a pitched rest or the hips tilt-clamp rebase are both absorbed).
/// The existing two-bone IK moves the wrist there while preserving the
/// hand's world rotation. The correction is scaled by
/// `rig.face_proximity` and is exactly zero away from the face, so every
/// other pose keeps the pure rotation path.
fn anchor_face_local_hands(
    skeleton: &SkeletonAsset,
    humanoid: &HumanoidMap,
    local_transforms: &mut [Transform],
    rig: &RigPose,
    params: &RetargetParams,
    state: &RetargetState,
) {
    if !params.face_contact_anchoring || rig.quality < params.min_quality {
        return;
    }
    let Some(NodeId(head_node)) = humanoid.bone_map.get(&HB::Head).copied() else {
        return;
    };
    let head_node = head_node as usize;
    if head_node >= local_transforms.len() {
        return;
    }
    let (h_pos, h_rot) = node_world_transform(skeleton, local_transforms, head_node);
    let rest_head_world = state
        .rest_world_rot
        .get(head_node)
        .copied()
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let rest_head_world_inv = quat_conjugate(&rest_head_world);
    for (side, bones) in [
        (
            0usize,
            (HB::LeftUpperArm, HB::LeftLowerArm, HB::LeftHand),
        ),
        (
            1usize,
            (HB::RightUpperArm, HB::RightLowerArm, HB::RightHand),
        ),
    ] {
        let w = rig.face_proximity[side];
        if w <= 0.01 {
            continue;
        }
        let (upper_b, lower_b, hand_b) = bones;
        let (Some(NodeId(un)), Some(NodeId(ln)), Some(NodeId(hn))) = (
            humanoid.bone_map.get(&upper_b).copied(),
            humanoid.bone_map.get(&lower_b).copied(),
            humanoid.bone_map.get(&hand_b).copied(),
        ) else {
            continue;
        };
        let (un, ln, hn) = (un as usize, ln as usize, hn as usize);
        let n = local_transforms.len();
        if un >= n || ln >= n || hn >= n {
            continue;
        }
        let (s_pos, s_rot) = node_world_transform(skeleton, local_transforms, un);
        let (e_pos, e_rot) = node_world_transform(skeleton, local_transforms, ln);
        let (w_pos, w_rot) = node_world_transform(skeleton, local_transforms, hn);
        // Camera-axes head-local offset → this avatar's head-node-local
        // offset (see the derivation in the doc comment above).
        let off = quat_rotate_vec3(&rest_head_world_inv, &rig.head_local_wrists[side]);
        let rot_off = quat_rotate_vec3(&h_rot, &off);
        let target = [
            h_pos[0] + rot_off[0],
            h_pos[1] + rot_off[1],
            h_pos[2] + rot_off[2],
        ];
        let err = vec3_length(&vec3_sub(&target, &w_pos));
        if err < 0.005 {
            continue;
        }
        let parent_rot = match skeleton.nodes[un].parent {
            Some(NodeId(p)) => world_rot(skeleton, local_transforms, p as usize),
            None => [0.0, 0.0, 0.0, 1.0],
        };
        let (u_new, l_new, h_new) = solve_two_bone_ik(
            s_pos, e_pos, w_pos, target, s_rot, e_rot, w_rot, parent_rot,
        );
        local_transforms[un].rotation = slerp_short(&local_transforms[un].rotation, &u_new, w);
        local_transforms[ln].rotation = slerp_short(&local_transforms[ln].rotation, &l_new, w);
        local_transforms[hn].rotation = slerp_short(&local_transforms[hn].rotation, &h_new, w);
    }
}

/// Compute world position and world rotation of node `node` given `locals`.
fn node_world_transform(
    skeleton: &SkeletonAsset,
    locals: &[Transform],
    mut node: usize,
) -> (Vec3, Quat) {
    let mut chain = Vec::with_capacity(16);
    loop {
        chain.push(node);
        match skeleton.nodes[node].parent {
            Some(NodeId(p)) => node = p as usize,
            None => break,
        }
    }
    let mut p = [0.0f32; 3];
    let mut r = [0.0f32, 0.0, 0.0, 1.0];
    for &k in chain.iter().rev() {
        let l_t = locals[k].translation;
        let l_r = locals[k].rotation;
        let off = quat_rotate_vec3(&r, &l_t);
        p = [p[0] + off[0], p[1] + off[1], p[2] + off[2]];
        r = quat_normalize(&quat_mul(&r, &l_r));
    }
    (p, r)
}

/// Analytical Two-Bone Inverse Kinematics for a 3-joint chain (Shoulder -> Elbow -> Wrist).
///
/// Modifies the upper arm and lower arm rotations so that the wrist reaches `target_pos`
/// while preserving the existing elbow bend direction and maintaining the hand's
/// orientation in world space.
///
/// Returns `(new_upper_local_rot, new_lower_local_rot, new_hand_local_rot)`.
fn solve_two_bone_ik(
    shoulder_pos: Vec3,
    elbow_pos: Vec3,
    wrist_pos: Vec3,
    target_pos: Vec3,
    upper_world_rot: Quat,
    lower_world_rot: Quat,
    hand_world_rot: Quat,
    parent_world_rot: Quat,
) -> (Quat, Quat, Quat) {
    let v_se = vec3_sub(&elbow_pos, &shoulder_pos);
    let v_ew = vec3_sub(&wrist_pos, &elbow_pos);
    let l1 = vec3_length(&v_se);
    let l2 = vec3_length(&v_ew);
    if l1 < 1e-3 || l2 < 1e-3 {
        let u_loc = quat_normalize(&quat_mul(
            &quat_conjugate(&parent_world_rot),
            &upper_world_rot,
        ));
        let l_loc = quat_normalize(&quat_mul(
            &quat_conjugate(&upper_world_rot),
            &lower_world_rot,
        ));
        let h_loc = quat_normalize(&quat_mul(
            &quat_conjugate(&lower_world_rot),
            &hand_world_rot,
        ));
        return (u_loc, l_loc, h_loc);
    }

    let d_vec = vec3_sub(&target_pos, &shoulder_pos);
    let d_len = vec3_length(&d_vec);
    let min_reach = (l1 - l2).abs() + 1e-4;
    let max_reach = (l1 + l2) * 0.9999;
    let d_clamped = d_len.clamp(min_reach, max_reach);

    let u_d = if d_len > 1e-6 {
        [d_vec[0] / d_len, d_vec[1] / d_len, d_vec[2] / d_len]
    } else {
        [0.0, 0.0, 1.0]
    };

    // Law of cosines: angle at shoulder
    let cos_alpha =
        ((l1 * l1 + d_clamped * d_clamped - l2 * l2) / (2.0 * l1 * d_clamped)).clamp(-1.0, 1.0);
    let sin_alpha = (1.0 - cos_alpha * cos_alpha).max(0.0).sqrt();

    // Pole vector from current elbow
    let dot_e_d = v_se[0] * u_d[0] + v_se[1] * u_d[1] + v_se[2] * u_d[2];
    let perp = [
        v_se[0] - dot_e_d * u_d[0],
        v_se[1] - dot_e_d * u_d[1],
        v_se[2] - dot_e_d * u_d[2],
    ];
    let perp_len = vec3_length(&perp);
    let u_p = if perp_len > 1e-4 {
        [perp[0] / perp_len, perp[1] / perp_len, perp[2] / perp_len]
    } else {
        let cross = [
            u_d[1] * 0.0 - u_d[2] * 1.0,
            u_d[2] * 0.0 - u_d[0] * 0.0,
            u_d[0] * 1.0 - u_d[1] * 0.0,
        ];
        let clen = vec3_length(&cross);
        if clen > 1e-4 {
            [cross[0] / clen, cross[1] / clen, cross[2] / clen]
        } else {
            [1.0, 0.0, 0.0]
        }
    };

    // Solved elbow position
    let e_new = [
        shoulder_pos[0] + l1 * (cos_alpha * u_d[0] + sin_alpha * u_p[0]),
        shoulder_pos[1] + l1 * (cos_alpha * u_d[1] + sin_alpha * u_p[1]),
        shoulder_pos[2] + l1 * (cos_alpha * u_d[2] + sin_alpha * u_p[2]),
    ];

    // Upper arm rotation delta: rotates v_se onto (e_new - shoulder_pos)
    let v_se_norm = [v_se[0] / l1, v_se[1] / l1, v_se[2] / l1];
    let v_se_new = vec3_sub(&e_new, &shoulder_pos);
    let v_se_new_norm = [v_se_new[0] / l1, v_se_new[1] / l1, v_se_new[2] / l1];
    let delta_upper = quat_from_to(v_se_norm, v_se_new_norm);

    let upper_world_new = quat_normalize(&quat_mul(&delta_upper, &upper_world_rot));
    let upper_local_new = quat_normalize(&quat_mul(
        &quat_conjugate(&parent_world_rot),
        &upper_world_new,
    ));

    // Lower arm rotation delta: rotates v_ew carried by delta_upper onto (target_pos - e_new)
    let v_ew_norm = [v_ew[0] / l2, v_ew[1] / l2, v_ew[2] / l2];
    let v_ew_carried = quat_rotate_vec3(&delta_upper, &v_ew_norm);

    let v_target = vec3_sub(&target_pos, &e_new);
    let target_dist = vec3_length(&v_target).max(1e-6);
    let v_target_norm = [
        v_target[0] / target_dist,
        v_target[1] / target_dist,
        v_target[2] / target_dist,
    ];
    let delta_lower = quat_from_to(v_ew_carried, v_target_norm);

    let lower_world_new = quat_normalize(&quat_mul(
        &delta_lower,
        &quat_mul(&delta_upper, &lower_world_rot),
    ));
    let lower_local_new = quat_normalize(&quat_mul(
        &quat_conjugate(&upper_world_new),
        &lower_world_new,
    ));

    // Preserve hand world rotation
    let hand_local_new = quat_normalize(&quat_mul(
        &quat_conjugate(&lower_world_new),
        &hand_world_rot,
    ));

    (upper_local_new, lower_local_new, hand_local_new)
}

/// Detect and prevent hands from crossing or penetrating each other in front of the body.
fn prevent_hand_crossing(
    skeleton: &SkeletonAsset,
    humanoid: &HumanoidMap,
    local_transforms: &mut [Transform],
    state: &mut RetargetState,
    params: &RetargetParams,
) {
    if !params.hand_cross_prevention {
        return;
    }

    let l_upper_node = humanoid
        .bone_map
        .get(&HB::LeftUpperArm)
        .map(|n| n.0 as usize);
    let l_lower_node = humanoid
        .bone_map
        .get(&HB::LeftLowerArm)
        .map(|n| n.0 as usize);
    let l_hand_node = humanoid.bone_map.get(&HB::LeftHand).map(|n| n.0 as usize);
    let r_upper_node = humanoid
        .bone_map
        .get(&HB::RightUpperArm)
        .map(|n| n.0 as usize);
    let r_lower_node = humanoid
        .bone_map
        .get(&HB::RightLowerArm)
        .map(|n| n.0 as usize);
    let r_hand_node = humanoid.bone_map.get(&HB::RightHand).map(|n| n.0 as usize);

    let (Some(lu), Some(ll), Some(lh), Some(ru), Some(rl), Some(rh)) = (
        l_upper_node,
        l_lower_node,
        l_hand_node,
        r_upper_node,
        r_lower_node,
        r_hand_node,
    ) else {
        return;
    };

    let n = local_transforms.len();
    if lu >= n || ll >= n || lh >= n || ru >= n || rl >= n || rh >= n {
        return;
    }

    // World positions and rotations of arm joints
    let (s_l_pos, s_l_rot) = node_world_transform(skeleton, local_transforms, lu);
    let (e_l_pos, e_l_rot) = node_world_transform(skeleton, local_transforms, ll);
    let (w_l_pos, w_l_rot) = node_world_transform(skeleton, local_transforms, lh);

    let (s_r_pos, s_r_rot) = node_world_transform(skeleton, local_transforms, ru);
    let (e_r_pos, e_r_rot) = node_world_transform(skeleton, local_transforms, rl);
    let (w_r_pos, w_r_rot) = node_world_transform(skeleton, local_transforms, rh);

    // Lateral axis: points from Right Shoulder to Left Shoulder
    let v_shoulders = vec3_sub(&s_l_pos, &s_r_pos);
    let d_shoulders = vec3_length(&v_shoulders);
    if d_shoulders < 1e-3 {
        return;
    }
    let u_lat = [
        v_shoulders[0] / d_shoulders,
        v_shoulders[1] / d_shoulders,
        v_shoulders[2] / d_shoulders,
    ];

    // Signed lateral separation along u_lat:
    // When left hand is to the left of right hand, delta_x > 0.
    // When crossed, delta_x < 0.
    let v_wrists = vec3_sub(&w_l_pos, &w_r_pos);
    let delta_x = v_wrists[0] * u_lat[0] + v_wrists[1] * u_lat[1] + v_wrists[2] * u_lat[2];
    let d_wrists = vec3_length(&v_wrists);

    // Only apply when hands are in the interaction zone (near each other)
    const INTERACTION_ZONE_M: f32 = 0.28;
    let min_dist = params.min_hand_distance.max(0.04);

    if d_wrists >= INTERACTION_ZONE_M && delta_x >= min_dist {
        return;
    }

    // Check if hands have crossed or penetrated
    if delta_x < min_dist || d_wrists < min_dist {
        let mid_wrists = [
            0.5 * (w_l_pos[0] + w_r_pos[0]),
            0.5 * (w_l_pos[1] + w_r_pos[1]),
            0.5 * (w_l_pos[2] + w_r_pos[2]),
        ];

        // Desired lateral separation
        let delta_x_target = delta_x.max(min_dist);

        // Perpendicular offset (height & depth)
        let v_perp = [
            v_wrists[0] - delta_x * u_lat[0],
            v_wrists[1] - delta_x * u_lat[1],
            v_wrists[2] - delta_x * u_lat[2],
        ];

        let mut v_target = [
            delta_x_target * u_lat[0] + v_perp[0],
            delta_x_target * u_lat[1] + v_perp[1],
            delta_x_target * u_lat[2] + v_perp[2],
        ];

        let target_dist = vec3_length(&v_target);
        if target_dist < min_dist {
            let scale = min_dist / target_dist.max(1e-6);
            v_target = [
                v_target[0] * scale,
                v_target[1] * scale,
                v_target[2] * scale,
            ];
        }

        let w_l_target = [
            mid_wrists[0] + 0.5 * v_target[0],
            mid_wrists[1] + 0.5 * v_target[1],
            mid_wrists[2] + 0.5 * v_target[2],
        ];
        let w_r_target = [
            mid_wrists[0] - 0.5 * v_target[0],
            mid_wrists[1] - 0.5 * v_target[1],
            mid_wrists[2] - 0.5 * v_target[2],
        ];

        // Parent world rotations for upper arms
        let parent_l_rot = match skeleton.nodes[lu].parent {
            Some(NodeId(p)) => world_rot(skeleton, local_transforms, p as usize),
            None => [0.0, 0.0, 0.0, 1.0],
        };
        let parent_r_rot = match skeleton.nodes[ru].parent {
            Some(NodeId(p)) => world_rot(skeleton, local_transforms, p as usize),
            None => [0.0, 0.0, 0.0, 1.0],
        };

        let (lu_new, ll_new, lh_new) = solve_two_bone_ik(
            s_l_pos,
            e_l_pos,
            w_l_pos,
            w_l_target,
            s_l_rot,
            e_l_rot,
            w_l_rot,
            parent_l_rot,
        );
        let (ru_new, rl_new, rh_new) = solve_two_bone_ik(
            s_r_pos,
            e_r_pos,
            w_r_pos,
            w_r_target,
            s_r_rot,
            e_r_rot,
            w_r_rot,
            parent_r_rot,
        );

        local_transforms[lu].rotation = lu_new;
        local_transforms[ll].rotation = ll_new;
        local_transforms[lh].rotation = lh_new;

        local_transforms[ru].rotation = ru_new;
        local_transforms[rl].rotation = rl_new;
        local_transforms[rh].rotation = rh_new;

        state.prev_local.insert(HB::LeftUpperArm, lu_new);
        state.prev_local.insert(HB::LeftLowerArm, ll_new);
        state.prev_local.insert(HB::LeftHand, lh_new);

        state.prev_local.insert(HB::RightUpperArm, ru_new);
        state.prev_local.insert(HB::RightLowerArm, rl_new);
        state.prev_local.insert(HB::RightHand, rh_new);

        if let Some(f) = state.one_euro_bones.get_mut(&HB::LeftUpperArm) {
            f.set_filtered(lu_new);
        }
        if let Some(f) = state.one_euro_bones.get_mut(&HB::LeftLowerArm) {
            f.set_filtered(ll_new);
        }
        if let Some(f) = state.one_euro_bones.get_mut(&HB::LeftHand) {
            f.set_filtered(lh_new);
        }

        if let Some(f) = state.one_euro_bones.get_mut(&HB::RightUpperArm) {
            f.set_filtered(ru_new);
        }
        if let Some(f) = state.one_euro_bones.get_mut(&HB::RightLowerArm) {
            f.set_filtered(rl_new);
        }
        if let Some(f) = state.one_euro_bones.get_mut(&HB::RightHand) {
            f.set_filtered(rh_new);
        }
    }
}

fn avatar_shoulder_span(
    skeleton: &SkeletonAsset,
    humanoid: &HumanoidMap,
    state: &RetargetState,
) -> Option<f32> {
    let l = humanoid.bone_map.get(&HumanoidBone::LeftUpperArm)?.0 as usize;
    let r = humanoid.bone_map.get(&HumanoidBone::RightUpperArm)?.0 as usize;
    if l >= skeleton.nodes.len() || r >= skeleton.nodes.len() {
        return None;
    }
    let d = vec3_length(&vec3_sub(
        &state.rest_world_pos[l],
        &state.rest_world_pos[r],
    ));
    (d > 0.05).then_some(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::SkeletonNode;
    use crate::tracking::fusion::output::RigBone;

    /// Minimal 3-node chain: root → Hips → Spine, identity rest.
    fn skeleton() -> (SkeletonAsset, HumanoidMap) {
        let mk =
            |id: u64, name: &str, parent: Option<u64>, children: Vec<u64>, bone| SkeletonNode {
                id: NodeId(id),
                name: name.to_string(),
                parent: parent.map(NodeId),
                children: children.into_iter().map(NodeId).collect(),
                rest_local: Transform {
                    translation: [0.0, 0.5, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                },
                humanoid_bone: bone,
            };
        let nodes = vec![
            mk(0, "root", None, vec![1], None),
            mk(1, "hips", Some(0), vec![2], Some(HumanoidBone::Hips)),
            mk(2, "spine", Some(1), vec![], Some(HumanoidBone::Spine)),
        ];
        let sk = SkeletonAsset {
            nodes,
            root_nodes: vec![NodeId(0)],
            inverse_bind_matrices: vec![],
        };
        let mut bone_map = HashMap::new();
        bone_map.insert(HumanoidBone::Hips, NodeId(1));
        bone_map.insert(HumanoidBone::Spine, NodeId(2));
        (sk, HumanoidMap { bone_map })
    }

    #[test]
    fn upright_rebase_clamps_whole_body_tilt_but_keeps_articulation() {
        let (sk, hm) = skeleton();
        let mut locals: Vec<Transform> = sk.nodes.iter().map(|n| n.rest_local.clone()).collect();
        // Estimator says: hips pitched forward 60° (pelvis/spine-split
        // wander), spine pitched forward 80° in world (i.e. 20° relative
        // flexion which is REAL).
        let qx = |a: f32| [(a / 2.0).sin(), 0.0, 0.0, (a / 2.0).cos()];
        let mut rig = RigPose {
            quality: 1.0,
            ..Default::default()
        };
        rig.bones.insert(
            HumanoidBone::Hips,
            RigBone {
                delta_world: qx(1.0),
                sigma: 0.05,
                data_sigma: 0.05,
            },
        );
        rig.bones.insert(
            HumanoidBone::Spine,
            RigBone {
                delta_world: qx(1.2),
                sigma: 0.05,
                data_sigma: 0.05,
            },
        );
        let mut st = RetargetState::default();
        let params = RetargetParams {
            rotation_blend: 1.0,
            max_root_tilt: 0.2,
            ..Default::default()
        };
        apply_rig_pose(&rig, &sk, &hm, &mut locals, &params, &mut st, 1.0 / 30.0);
        // Hips world tilt clamped to 0.2 rad.
        let hips_w = world_rot(&sk, &locals, 1);
        let up = quat_rotate_vec3(&hips_w, &[0.0, 1.0, 0.0]);
        let tilt = up[1].clamp(-1.0, 1.0).acos();
        assert!((tilt - 0.2).abs() < 1e-3, "hips tilt {tilt} (want 0.2)");
        // Spine local flexion preserved at 0.2 rad relative to hips.
        let q = locals[2].rotation;
        let ang = 2.0 * q[3].clamp(-1.0, 1.0).acos();
        let ang = if ang > std::f32::consts::PI {
            2.0 * std::f32::consts::PI - ang
        } else {
            ang
        };
        assert!(
            (ang - 0.2).abs() < 1e-3,
            "spine relative flexion {ang} (want 0.2)"
        );
    }

    #[test]
    fn world_delta_lands_on_child_local() {
        let (sk, hm) = skeleton();
        let mut locals: Vec<Transform> = sk.nodes.iter().map(|n| n.rest_local.clone()).collect();
        // Hips rotated 30° about Y, spine rotated 50° about Y in world.
        let qy = |a: f32| [0.0, (a / 2.0).sin(), 0.0, (a / 2.0).cos()];
        let mut rig = RigPose {
            quality: 1.0,
            ..Default::default()
        };
        rig.bones.insert(
            HumanoidBone::Hips,
            RigBone {
                delta_world: qy(0.5),
                sigma: 0.05,
                data_sigma: 0.05,
            },
        );
        rig.bones.insert(
            HumanoidBone::Spine,
            RigBone {
                delta_world: qy(0.9),
                sigma: 0.05,
                data_sigma: 0.05,
            },
        );
        let mut st = RetargetState::default();
        let params = RetargetParams {
            rotation_blend: 1.0,
            ..Default::default()
        };
        apply_rig_pose(&rig, &sk, &hm, &mut locals, &params, &mut st, 1.0 / 60.0);
        // Spine local should be the 0.4 rad difference.
        let q = locals[2].rotation;
        let ang = 2.0 * q[3].clamp(-1.0, 1.0).acos();
        assert!((ang - 0.4).abs() < 1e-4, "spine local angle {ang}");
        assert!(q[1] > 0.0);
        // Uncertain bones stay at rest.
        rig.bones.get_mut(&HumanoidBone::Spine).unwrap().sigma = 5.0;
        apply_rig_pose(&rig, &sk, &hm, &mut locals, &params, &mut st, 1.0 / 60.0);
        assert!((locals[2].rotation[3] - 1.0).abs() < 1e-5);
    }

    /// T-pose arms: Hips → Spine → Chest → (±UpperArm → LowerArm → Hand),
    /// arms along ±X (the hand-cross test's skeleton, identity rest).
    fn arm_skeleton() -> (SkeletonAsset, HumanoidMap) {
        let mk =
            |id: u64, parent: Option<u64>, children: Vec<u64>, t: [f32; 3], bone| SkeletonNode {
                id: NodeId(id),
                name: format!("n{id}"),
                parent: parent.map(NodeId),
                children: children.into_iter().map(NodeId).collect(),
                rest_local: Transform {
                    translation: t,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                },
                humanoid_bone: bone,
            };
        let nodes = vec![
            mk(0, None, vec![1], [0.0, 0.0, 0.0], None),
            mk(
                1,
                Some(0),
                vec![2],
                [0.0, 1.0, 0.0],
                Some(HumanoidBone::Hips),
            ),
            mk(
                2,
                Some(1),
                vec![3],
                [0.0, 0.2, 0.0],
                Some(HumanoidBone::Spine),
            ),
            mk(
                3,
                Some(2),
                vec![4, 7],
                [0.0, 0.2, 0.0],
                Some(HumanoidBone::Chest),
            ),
            mk(
                4,
                Some(3),
                vec![5],
                [0.2, 0.0, 0.0],
                Some(HumanoidBone::LeftUpperArm),
            ),
            mk(
                5,
                Some(4),
                vec![6],
                [0.25, 0.0, 0.0],
                Some(HumanoidBone::LeftLowerArm),
            ),
            mk(
                6,
                Some(5),
                vec![],
                [0.25, 0.0, 0.0],
                Some(HumanoidBone::LeftHand),
            ),
            mk(
                7,
                Some(3),
                vec![8],
                [-0.2, 0.0, 0.0],
                Some(HumanoidBone::RightUpperArm),
            ),
            mk(
                8,
                Some(7),
                vec![9],
                [-0.25, 0.0, 0.0],
                Some(HumanoidBone::RightLowerArm),
            ),
            mk(
                9,
                Some(8),
                vec![],
                [-0.25, 0.0, 0.0],
                Some(HumanoidBone::RightHand),
            ),
        ];
        let sk = SkeletonAsset {
            nodes,
            root_nodes: vec![NodeId(0)],
            inverse_bind_matrices: vec![],
        };
        let mut bone_map = HashMap::new();
        for (b, id) in [
            (HumanoidBone::Hips, 1),
            (HumanoidBone::Spine, 2),
            (HumanoidBone::Chest, 3),
            (HumanoidBone::LeftUpperArm, 4),
            (HumanoidBone::LeftLowerArm, 5),
            (HumanoidBone::LeftHand, 6),
            (HumanoidBone::RightUpperArm, 7),
            (HumanoidBone::RightLowerArm, 8),
            (HumanoidBone::RightHand, 9),
        ] {
            bone_map.insert(b, NodeId(id));
        }
        (sk, HumanoidMap { bone_map })
    }

    #[test]
    fn sigma_gated_arm_relaxes_to_a_pose_not_bind() {
        let (sk, hm) = arm_skeleton();
        let mut locals: Vec<Transform> = sk.nodes.iter().map(|n| n.rest_local.clone()).collect();
        let qy = |a: f32| [0.0, (a / 2.0).sin(), 0.0, (a / 2.0).cos()];
        let mut rig = RigPose {
            quality: 1.0,
            ..Default::default()
        };
        rig.bones.insert(
            HumanoidBone::LeftUpperArm,
            RigBone {
                delta_world: qy(0.4),
                sigma: 0.05,
                data_sigma: 0.05,
            },
        );
        let mut st = RetargetState::default();
        let params = RetargetParams {
            rotation_blend: 1.0,
            ..Default::default()
        };
        apply_rig_pose(&rig, &sk, &hm, &mut locals, &params, &mut st, 1.0 / 30.0);
        // Driven: the yaw delta is on the bone, not the A-pose overlay.
        assert!((locals[4].rotation[1] - (0.2f32).sin()).abs() < 1e-4);

        // σ blows past the gate: the arm must relax toward the A-pose
        // overlay, not the T-pose bind (identity here).
        rig.bones
            .get_mut(&HumanoidBone::LeftUpperArm)
            .unwrap()
            .sigma = 5.0;
        apply_rig_pose(&rig, &sk, &hm, &mut locals, &params, &mut st, 1.0 / 30.0);
        let (_, a_local) = st.arelax_local[&HumanoidBone::LeftUpperArm];
        let dot = locals[4].rotation[0] * a_local[0]
            + locals[4].rotation[1] * a_local[1]
            + locals[4].rotation[2] * a_local[2]
            + locals[4].rotation[3] * a_local[3];
        assert!(dot.abs() > 0.9999, "arm relax target is not the A-pose");
        // The A-pose is a non-identity rotation on this T-bind rig.
        assert!(a_local[3] < 0.9999, "A-pose overlay must rotate the arm");
    }

    #[test]
    fn test_hand_cross_prevention() {
        let mk =
            |id: u64, name: &str, parent: Option<u64>, children: Vec<u64>, t: [f32; 3], bone| {
                SkeletonNode {
                    id: NodeId(id),
                    name: name.to_string(),
                    parent: parent.map(NodeId),
                    children: children.into_iter().map(NodeId).collect(),
                    rest_local: Transform {
                        translation: t,
                        rotation: [0.0, 0.0, 0.0, 1.0],
                        scale: [1.0, 1.0, 1.0],
                    },
                    humanoid_bone: bone,
                }
            };

        // Rigs: Hips -> Spine -> Chest
        // Chest has LeftUpperArm (+0.2m) and RightUpperArm (-0.2m)
        // Arms extend outward along X:
        // LeftUpperArm -> LeftLowerArm (+0.25m) -> LeftHand (+0.25m)
        // RightUpperArm -> RightLowerArm (-0.25m) -> RightHand (-0.25m)
        let nodes = vec![
            mk(0, "root", None, vec![1], [0.0, 0.0, 0.0], None),
            mk(
                1,
                "hips",
                Some(0),
                vec![2],
                [0.0, 1.0, 0.0],
                Some(HumanoidBone::Hips),
            ),
            mk(
                2,
                "spine",
                Some(1),
                vec![3],
                [0.0, 0.2, 0.0],
                Some(HumanoidBone::Spine),
            ),
            mk(
                3,
                "chest",
                Some(2),
                vec![4, 7],
                [0.0, 0.2, 0.0],
                Some(HumanoidBone::Chest),
            ),
            mk(
                4,
                "l_upper",
                Some(3),
                vec![5],
                [0.2, 0.0, 0.0],
                Some(HumanoidBone::LeftUpperArm),
            ),
            mk(
                5,
                "l_lower",
                Some(4),
                vec![6],
                [0.25, 0.0, 0.0],
                Some(HumanoidBone::LeftLowerArm),
            ),
            mk(
                6,
                "l_hand",
                Some(5),
                vec![],
                [0.25, 0.0, 0.0],
                Some(HumanoidBone::LeftHand),
            ),
            mk(
                7,
                "r_upper",
                Some(3),
                vec![8],
                [-0.2, 0.0, 0.0],
                Some(HumanoidBone::RightUpperArm),
            ),
            mk(
                8,
                "r_lower",
                Some(7),
                vec![9],
                [-0.25, 0.0, 0.0],
                Some(HumanoidBone::RightLowerArm),
            ),
            mk(
                9,
                "r_hand",
                Some(8),
                vec![],
                [-0.25, 0.0, 0.0],
                Some(HumanoidBone::RightHand),
            ),
        ];
        let sk = SkeletonAsset {
            nodes,
            root_nodes: vec![NodeId(0)],
            inverse_bind_matrices: vec![],
        };
        let mut bone_map = HashMap::new();
        bone_map.insert(HumanoidBone::Hips, NodeId(1));
        bone_map.insert(HumanoidBone::Spine, NodeId(2));
        bone_map.insert(HumanoidBone::Chest, NodeId(3));
        bone_map.insert(HumanoidBone::LeftUpperArm, NodeId(4));
        bone_map.insert(HumanoidBone::LeftLowerArm, NodeId(5));
        bone_map.insert(HumanoidBone::LeftHand, NodeId(6));
        bone_map.insert(HumanoidBone::RightUpperArm, NodeId(7));
        bone_map.insert(HumanoidBone::RightLowerArm, NodeId(8));
        bone_map.insert(HumanoidBone::RightHand, NodeId(9));
        let hm = HumanoidMap { bone_map };

        let mut locals: Vec<Transform> = sk.nodes.iter().map(|n| n.rest_local.clone()).collect();
        let mut st = RetargetState::default();

        // Deliberately rotate arms inwards so hands cross past each other:
        // Left arm rotates inward (yaw ~+145°)
        // Right arm rotates inward (yaw ~-145°)
        let q_yaw = |rad: f32| [0.0, (rad / 2.0).sin(), 0.0, (rad / 2.0).cos()];
        locals[4].rotation = q_yaw(2.5);
        locals[5].rotation = q_yaw(0.2);
        locals[7].rotation = q_yaw(-2.5);
        locals[8].rotation = q_yaw(-0.2);

        let (w_l_before, _) = node_world_transform(&sk, &locals, 6);
        let (w_r_before, _) = node_world_transform(&sk, &locals, 9);
        println!(
            "Before IK: Left wrist X = {:.4}, Right wrist X = {:.4}",
            w_l_before[0], w_r_before[0]
        );
        // Confirm that without anti-cross, Left wrist is to the right of Right wrist (crossed!)
        assert!(
            w_l_before[0] < w_r_before[0],
            "Hands must be crossed initially for this test"
        );

        let params = RetargetParams {
            hand_cross_prevention: true,
            min_hand_distance: 0.08,
            ..Default::default()
        };

        prevent_hand_crossing(&sk, &hm, &mut locals, &mut st, &params);

        let (w_l_after, _) = node_world_transform(&sk, &locals, 6);
        let (w_r_after, _) = node_world_transform(&sk, &locals, 9);
        println!(
            "After IK: Left wrist X = {:.4}, Right wrist X = {:.4}",
            w_l_after[0], w_r_after[0]
        );

        // 1. Left wrist must now be on the left side of Right wrist (no crossing!)
        assert!(
            w_l_after[0] > w_r_after[0],
            "Left wrist ({}) must be to the left of Right wrist ({})",
            w_l_after[0],
            w_r_after[0]
        );

        // 2. Separation must be at least min_hand_distance
        let sep = w_l_after[0] - w_r_after[0];
        assert!(sep >= 0.079, "Lateral separation {sep} should be >= 0.08m");

        // 3. Bone lengths must remain rigid
        let (s_l, _) = node_world_transform(&sk, &locals, 4);
        let (e_l, _) = node_world_transform(&sk, &locals, 5);
        let (w_l, _) = node_world_transform(&sk, &locals, 6);
        let l1 = vec3_length(&vec3_sub(&e_l, &s_l));
        let l2 = vec3_length(&vec3_sub(&w_l, &e_l));
        assert!(
            (l1 - 0.25).abs() < 1e-4,
            "Upper arm length must be 0.25 (got {l1})"
        );
        assert!(
            (l2 - 0.25).abs() < 1e-4,
            "Forearm length must be 0.25 (got {l2})"
        );
    }

    #[test]
    fn test_one_euro_filter_reduces_jitter_when_still_and_tracks_fast_movement() {
        let (sk, hm) = skeleton();
        let qy = |a: f32| [0.0, (a / 2.0).sin(), 0.0, (a / 2.0).cos()];

        // 1. Stillness jitter test:
        // Input oscillates at 30Hz: base angle 0.5 rad with noise +-0.05 rad (peak-to-peak 0.1 rad).
        let mut st_filter = RetargetState::default();
        let params_filter = RetargetParams {
            rotation_blend: 1.0, // pure 1€ filter behavior
            one_euro_enabled: true,
            one_euro_min_cutoff: 1.0,
            one_euro_beta: 1.0,
            one_euro_d_cutoff: 1.0,
            ..Default::default()
        };

        let mut spine_angles = Vec::new();
        let mut locals: Vec<Transform> = sk.nodes.iter().map(|n| n.rest_local.clone()).collect();
        let dt = 1.0 / 30.0;

        for frame in 0..60 {
            let noise = if frame % 2 == 0 { 0.05 } else { -0.05 };
            let raw_angle = 0.5 + noise;
            let mut rig = RigPose {
                quality: 1.0,
                ..Default::default()
            };
            rig.bones.insert(
                HumanoidBone::Hips,
                RigBone {
                    delta_world: [0.0, 0.0, 0.0, 1.0],
                    sigma: 0.05,
                    data_sigma: 0.05,
                },
            );
            rig.bones.insert(
                HumanoidBone::Spine,
                RigBone {
                    delta_world: qy(raw_angle),
                    sigma: 0.05,
                    data_sigma: 0.05,
                },
            );
            apply_rig_pose(
                &rig,
                &sk,
                &hm,
                &mut locals,
                &params_filter,
                &mut st_filter,
                dt,
            );
            let q = locals[2].rotation;
            let ang = 2.0 * q[3].clamp(-1.0, 1.0).acos();
            if frame >= 10 {
                spine_angles.push(ang);
            }
        }

        let max_ang = spine_angles
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_ang = spine_angles.iter().copied().fold(f32::INFINITY, f32::min);
        let jitter_range = max_ang - min_ang;
        println!(
            "Jitter range with 1€ filter: {:.4} rad (raw: 0.1000 rad)",
            jitter_range
        );
        assert!(
            jitter_range < 0.025,
            "1€ filter failed to suppress stillness jitter: got range {:.4}, expected < 0.025",
            jitter_range
        );

        // 2. Fast movement tracking test:
        // Sudden jump to 1.5 rad. High angular velocity -> dynamic cutoff rises -> responds rapidly.
        let mut rig = RigPose {
            quality: 1.0,
            ..Default::default()
        };
        rig.bones.insert(
            HumanoidBone::Hips,
            RigBone {
                delta_world: [0.0, 0.0, 0.0, 1.0],
                sigma: 0.05,
                data_sigma: 0.05,
            },
        );
        rig.bones.insert(
            HumanoidBone::Spine,
            RigBone {
                delta_world: qy(1.5),
                sigma: 0.05,
                data_sigma: 0.05,
            },
        );
        apply_rig_pose(
            &rig,
            &sk,
            &hm,
            &mut locals,
            &params_filter,
            &mut st_filter,
            dt,
        );
        let q_fast = locals[2].rotation;
        let ang_fast = 2.0 * q_fast[3].clamp(-1.0, 1.0).acos();
        println!(
            "Angle after sudden fast motion: {:.4} rad (target: 1.5 rad)",
            ang_fast
        );
        assert!(
            ang_fast > 1.1,
            "1€ filter responded too sluggishly to fast motion: got {:.4}, expected > 1.1",
            ang_fast
        );

        // 3. Test disabling 1€ filter:
        let params_disabled = RetargetParams {
            rotation_blend: 1.0,
            one_euro_enabled: false,
            ..Default::default()
        };
        let mut st_disabled = RetargetState::default();
        apply_rig_pose(
            &rig,
            &sk,
            &hm,
            &mut locals,
            &params_disabled,
            &mut st_disabled,
            dt,
        );
        let q_raw = locals[2].rotation;
        let ang_raw = 2.0 * q_raw[3].clamp(-1.0, 1.0).acos();
        assert!(
            (ang_raw - 1.5).abs() < 1e-4,
            "With 1€ disabled, angle should match target exactly"
        );
    }
}
