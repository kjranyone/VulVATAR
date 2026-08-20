//! Tracking-v2 retarget: [`RigPose`] (world-delta joint rotations in the
//! viewer frame + pelvis position + per-joint σ) → avatar local bone
//! transforms.
//!
//! Every driven bone's *world* rotation is set to `Δ_V · rest_world`, i.e.
//! the same rotation the subject's corresponding segment made from its own
//! T-pose rest, so body-proportion differences never move a hand short of
//! or past its target (there is no position matching and no IK). Local
//! rotations are recovered top-down against the already-solved parents.
//!
//! Display smoothing is a dt-aware slerp toward the target (the user's
//! `rotation_blend` setting) from the previous frame's solved local
//! rotation; the estimator's own dynamics already produce a smooth 30 Hz
//! signal, so this only bridges render frames between tracking samples.

use std::collections::HashMap;

use crate::asset::{HumanoidBone, HumanoidMap, NodeId, SkeletonAsset, Transform};
use crate::math_utils::{
    quat_conjugate, quat_mul, quat_normalize, quat_rotate_vec3, vec3_length, vec3_sub, Quat, Vec3,
};
use crate::asset::HumanoidBone as HB;
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
    rest_world_rot: Vec<Quat>,
    rest_world_pos: Vec<Vec3>,
    rest_cache_len: usize,
}

impl RetargetState {
    pub fn reset(&mut self) {
        self.prev_local.clear();
        self.prev_hips_translation = None;
        self.anchor_cam = None;
        self.anchor_seed.clear();
        self.anchor_seed_start = None;
    }
    /// Forget only the display-smoothing state (avatar swap).
    pub fn reset_smoothing(&mut self) {
        self.prev_local.clear();
        self.prev_hips_translation = None;
        self.rest_cache_len = 0;
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
    /// Rig quality below which nothing is driven (subject lost / tiny).
    pub min_quality: f32,
    /// Seconds of well-tracked data used to seed the root anchor.
    pub anchor_seed_s: f64,
    /// Maximum whole-body lean (rad) the hips may carry; the rest of the
    /// estimator's pelvis tilt is treated as pelvis/spine-split wander (or
    /// camera mounting pitch) and absorbed. Body-internal articulation is
    /// unaffected — bones are re-expressed relative to the hips.
    pub max_root_tilt: f32,
}

impl Default for RetargetParams {
    fn default() -> Self {
        Self {
            rotation_blend: 0.35,
            root_translation_enabled: true,
            hand_tracking_enabled: true,
            lower_body_tracking_enabled: true,
            sigma_rest: 1.2,
            min_quality: 0.2,
            anchor_seed_s: 1.0,
            max_root_tilt: 0.20,
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

fn slerp_short(a: &Quat, b: &Quat, t: f32) -> Quat {
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

fn ensure_rest_cache(state: &mut RetargetState, skeleton: &SkeletonAsset) {
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
    ensure_rest_cache(state, skeleton);
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
                let ang = if ang > std::f32::consts::PI { 2.0 * std::f32::consts::PI - ang } else { ang };
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
        // Target local rotation. A low-quality rig (subject lost / tiny)
        // drives nothing: every bone relaxes to rest.
        let target = match rig.bones.get(&bone) {
            Some(rb)
                if rig.quality >= params.min_quality
                    && rb.sigma <= params.sigma_rest
                    && (params.hand_tracking_enabled || !is_finger(bone))
                    && (params.lower_body_tracking_enabled || !is_leg(bone)) =>
            {
                let corrected = quat_mul(&rebase, &rb.delta_world);
                let desired_world = quat_mul(&corrected, &state.rest_world_rot[node]);
                let parent_world = match skeleton.nodes[node].parent {
                    Some(NodeId(p)) => world_rot(skeleton, local_transforms, p as usize),
                    None => [0.0, 0.0, 0.0, 1.0],
                };
                quat_normalize(&quat_mul(&quat_conjugate(&parent_world), &desired_world))
            }
            _ => rest_local_rot,
        };
        let prev = state.prev_local.get(&bone).copied().unwrap_or(rest_local_rot);
        let out = if blend >= 1.0 {
            target
        } else {
            slerp_short(&prev, &target, blend)
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
            let target = match (state.anchor_cam, params.root_translation_enabled && rig.quality >= params.min_quality) {
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
                    [rest_pos[0] + d[0], rest_pos[1] + d[1], rest_pos[2] + d[2]]
                }
                _ => rest_pos,
            };
            let prev = state.prev_hips_translation.unwrap_or(rest_pos);
            let out = [
                prev[0] + blend * (target[0] - prev[0]),
                prev[1] + blend * (target[1] - prev[1]),
                prev[2] + blend * (target[2] - prev[2]),
            ];
            state.prev_hips_translation = Some(out);
            local_transforms[hips].translation = out;
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
    let d = vec3_length(&vec3_sub(&state.rest_world_pos[l], &state.rest_world_pos[r]));
    (d > 0.05).then_some(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::SkeletonNode;
    use crate::tracking::fusion::output::RigBone;

    /// Minimal 3-node chain: root → Hips → Spine, identity rest.
    fn skeleton() -> (SkeletonAsset, HumanoidMap) {
        let mk = |id: u64, name: &str, parent: Option<u64>, children: Vec<u64>, bone| SkeletonNode {
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
        rig.bones.insert(HumanoidBone::Hips, RigBone { delta_world: qx(1.0), sigma: 0.05, data_sigma: 0.05 });
        rig.bones.insert(HumanoidBone::Spine, RigBone { delta_world: qx(1.2), sigma: 0.05, data_sigma: 0.05 });
        let mut st = RetargetState::default();
        let params = RetargetParams { rotation_blend: 1.0, max_root_tilt: 0.2, ..Default::default() };
        apply_rig_pose(&rig, &sk, &hm, &mut locals, &params, &mut st, 1.0 / 30.0);
        // Hips world tilt clamped to 0.2 rad.
        let hips_w = world_rot(&sk, &locals, 1);
        let up = quat_rotate_vec3(&hips_w, &[0.0, 1.0, 0.0]);
        let tilt = up[1].clamp(-1.0, 1.0).acos();
        assert!((tilt - 0.2).abs() < 1e-3, "hips tilt {tilt} (want 0.2)");
        // Spine local flexion preserved at 0.2 rad relative to hips.
        let q = locals[2].rotation;
        let ang = 2.0 * q[3].clamp(-1.0, 1.0).acos();
        let ang = if ang > std::f32::consts::PI { 2.0 * std::f32::consts::PI - ang } else { ang };
        assert!((ang - 0.2).abs() < 1e-3, "spine relative flexion {ang} (want 0.2)");
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
}
