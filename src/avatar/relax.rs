//! Display-level rest pose: procedural A-pose plus the eased transition
//! into it ("relax").
//!
//! The rig's bind pose (`SkeletonNode::rest_local`, a T-pose on most
//! VRM / FBX avatars) is a mathematical reference — skinning, the
//! retarget's world-delta math, cloth bind state and the body SDF are
//! all derived against it — so it must not be rewritten. The *displayed*
//! rest is this overlay instead: the upper arms rotated down to
//! [`a_pose_drop_rad`] below horizontal, with the rotation derived from
//! the rig's own rest geometry (upper-arm → elbow direction), so a
//! T-bound rig lands on the same A-pose regardless of bone-axis
//! conventions, and a rig already bound at/below the target elevation
//! is left alone.
//!
//! When a pose driver (tracking, an animation clip) stops,
//! [`RelaxState`] captures the last solved pose and eases into the
//! overlay rest over [`relax_duration_s`] instead of snapping to the
//! bind pose in one frame. Tunables: `VULVATAR_APOSE_DEG` (arm drop in
//! degrees, default 45) and `VULVATAR_RELAX_S` (transition seconds,
//! default 0.9; 0 = instant, the old behaviour).

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::asset::{HumanoidBone as HB, HumanoidMap, NodeId, SkeletonAsset, Transform};
use crate::math_utils::{
    quat_conjugate, quat_from_vectors, quat_mul, quat_normalize, vec3_dot, vec3_length,
    vec3_normalize, vec3_sub, Quat, Vec3,
};

const UP: Vec3 = [0.0, 1.0, 0.0];

/// Target arm elevation below horizontal (rad). Env `VULVATAR_APOSE_DEG`
/// (degrees, clamped to 0..=90).
pub fn a_pose_drop_rad() -> f32 {
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("VULVATAR_APOSE_DEG")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| (0.0..=90.0).contains(v))
            .unwrap_or(45.0)
            .to_radians()
    })
}

/// Seconds the captured pose takes to ease into the rest. Env
/// `VULVATAR_RELAX_S` (0 = snap).
pub fn relax_duration_s() -> f32 {
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("VULVATAR_RELAX_S")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| *v >= 0.0)
            .unwrap_or(0.9)
    })
}

/// A-pose overlay: driven bone → (node, local rotation replacing the
/// bind rotation).
pub type APoseOverlay = HashMap<HB, (NodeId, Quat)>;

/// Derive the A-pose arm rotations from the rig's rest geometry. For
/// each upper-arm bone, the bind-pose upper-arm→elbow segment direction
/// is rotated down (in the segment's own vertical plane) to
/// [`a_pose_drop_rad`] below horizontal, then re-expressed as a local
/// rotation against the untouched bind parent — the same local-recovery
/// math `apply_rig_pose` uses, so the result is convention-free. Elbows
/// and hands follow as child bones.
pub fn a_pose_overlay(
    skeleton: &SkeletonAsset,
    humanoid: &HumanoidMap,
    rest_world_rot: &[Quat],
    rest_world_pos: &[Vec3],
) -> APoseOverlay {
    let drop = a_pose_drop_rad();
    let mut out = APoseOverlay::new();
    for (arm, elbow) in [
        (HB::LeftUpperArm, HB::LeftLowerArm),
        (HB::RightUpperArm, HB::RightLowerArm),
    ] {
        let (Some(&NodeId(u)), Some(&NodeId(l))) =
            (humanoid.bone_map.get(&arm), humanoid.bone_map.get(&elbow))
        else {
            continue;
        };
        let (u, l) = (u as usize, l as usize);
        if u >= rest_world_pos.len() || l >= rest_world_pos.len() || u >= rest_world_rot.len() {
            continue;
        }
        let seg = vec3_sub(&rest_world_pos[l], &rest_world_pos[u]);
        if vec3_length(&seg) < 1e-6 {
            continue;
        }
        let dir = vec3_normalize(&seg);
        // Already at or below the target elevation: leave the bind pose.
        if vec3_dot(&dir, &UP) <= -drop.sin() + 1e-3 {
            continue;
        }
        // Near-vertical arms have no horizontal component to splay from;
        // aiming them straight down is not a sensible A-pose — skip.
        let horiz_sq = dir[0] * dir[0] + dir[2] * dir[2];
        if horiz_sq < 0.25 {
            continue;
        }
        let h_scale = drop.cos() / horiz_sq.sqrt();
        let target = vec3_normalize(&[dir[0] * h_scale, -drop.sin(), dir[2] * h_scale]);
        let delta = quat_from_vectors(&dir, &target);
        let desired_world = quat_mul(&delta, &rest_world_rot[u]);
        let parent_world = match skeleton.nodes[u].parent {
            Some(NodeId(p)) if (p as usize) < rest_world_rot.len() => rest_world_rot[p as usize],
            _ => [0.0, 0.0, 0.0, 1.0],
        };
        let local = quat_normalize(&quat_mul(&quat_conjugate(&parent_world), &desired_world));
        out.insert(arm, (NodeId(u as u64), local));
    }
    out
}

/// Per-avatar state for the eased transition into the rest pose.
#[derive(Default, Clone, Debug)]
pub struct RelaxState {
    /// The previous frame's pose was produced by a driver (tracking or
    /// an animation clip). Consumed by [`RelaxState::capture`].
    was_driven: bool,
    /// Pose captured when the driver stopped; `None` once settled (and
    /// on a fresh load, where the avatar simply appears in the rest).
    captured: Option<Vec<Transform>>,
    elapsed_s: f32,
}

impl RelaxState {
    /// Arm the next transition. Call on frames where a driver produced
    /// the pose.
    pub fn mark_driven(&mut self) {
        self.was_driven = true;
        self.captured = None;
    }

    /// Capture the outgoing pose if a driver just stopped. Call BEFORE
    /// the base pose is rebuilt on undriven frames — `locals` still
    /// hold the last solved pose there. Without a preceding driven
    /// frame (fresh load) nothing is captured, so the avatar appears
    /// directly in the rest rather than easing from the identity pose.
    pub fn capture(&mut self, locals: &[Transform]) {
        if self.was_driven && self.captured.is_none() {
            self.captured = Some(locals.to_vec());
            self.elapsed_s = 0.0;
        }
        self.was_driven = false;
    }

    /// One relax step: blend the captured pose (if any) toward `target`
    /// with a smoothstep over [`relax_duration_s`] and write the result
    /// into `locals`; without a capture this writes `target` directly.
    /// Rotations slerp, translations lerp, scale takes the target's
    /// (rest) value.
    pub fn step_blend(&mut self, dt: f32, target: &[Transform], locals: &mut [Transform]) {
        let Some(captured) = self.captured.as_ref() else {
            let n = target.len().min(locals.len());
            locals[..n].clone_from_slice(&target[..n]);
            return;
        };
        let n = target.len().min(locals.len()).min(captured.len());
        let duration = relax_duration_s();
        self.elapsed_s += dt.max(0.0);
        let t = if duration <= 0.0 {
            1.0
        } else {
            (self.elapsed_s / duration).clamp(0.0, 1.0)
        };
        let alpha = t * t * (3.0 - 2.0 * t);
        for i in 0..n {
            let c = &captured[i];
            let tg = &target[i];
            let out = &mut locals[i];
            out.rotation = crate::avatar::retarget::slerp_short(&c.rotation, &tg.rotation, alpha);
            for k in 0..3 {
                out.translation[k] =
                    c.translation[k] + alpha * (tg.translation[k] - c.translation[k]);
            }
            out.scale = tg.scale;
        }
        if t >= 1.0 {
            self.captured = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::{HumanoidMap, SkeletonNode};
    use crate::avatar::pose::compute_global_transforms;
    use crate::avatar::retarget::{ensure_rest_cache, RetargetState};

    fn node(
        id: u64,
        parent: Option<u64>,
        children: Vec<u64>,
        t: [f32; 3],
        bone: Option<HB>,
    ) -> SkeletonNode {
        SkeletonNode {
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
        }
    }

    /// Hips → Spine → Chest with T-pose arms along ±X (`lower_dir`
    /// overrides the elbow offset to bind arms in another pose).
    fn arm_skeleton(lower_dir: [f32; 3]) -> (SkeletonAsset, HumanoidMap) {
        let nodes = vec![
            node(0, None, vec![1], [0.0, 0.0, 0.0], None),
            node(1, Some(0), vec![2], [0.0, 1.0, 0.0], Some(HB::Hips)),
            node(2, Some(1), vec![3], [0.0, 0.2, 0.0], Some(HB::Spine)),
            node(3, Some(2), vec![4, 7], [0.0, 0.2, 0.0], Some(HB::Chest)),
            node(4, Some(3), vec![5], [0.2, 0.0, 0.0], Some(HB::LeftUpperArm)),
            node(5, Some(4), vec![6], lower_dir, Some(HB::LeftLowerArm)),
            node(6, Some(5), vec![], [0.25, 0.0, 0.0], Some(HB::LeftHand)),
            node(
                7,
                Some(3),
                vec![8],
                [-0.2, 0.0, 0.0],
                Some(HB::RightUpperArm),
            ),
            node(
                8,
                Some(7),
                vec![9],
                [-lower_dir[0], lower_dir[1], lower_dir[2]],
                Some(HB::RightLowerArm),
            ),
            node(9, Some(8), vec![], [-0.25, 0.0, 0.0], Some(HB::RightHand)),
        ];
        let sk = SkeletonAsset {
            nodes,
            root_nodes: vec![NodeId(0)],
            inverse_bind_matrices: vec![],
        };
        let mut bone_map = HashMap::new();
        for (b, id) in [
            (HB::Hips, 1),
            (HB::Spine, 2),
            (HB::Chest, 3),
            (HB::LeftUpperArm, 4),
            (HB::LeftLowerArm, 5),
            (HB::LeftHand, 6),
            (HB::RightUpperArm, 7),
            (HB::RightLowerArm, 8),
            (HB::RightHand, 9),
        ] {
            bone_map.insert(b, NodeId(id));
        }
        (sk, HumanoidMap { bone_map })
    }

    /// World elevation of the left upper-arm→elbow segment after
    /// applying the overlay to the bind locals (rad below horizontal).
    fn overlay_arm_elevation(sk: &SkeletonAsset, overlay: &APoseOverlay) -> (f32, f32) {
        let mut locals: Vec<Transform> = sk.nodes.iter().map(|n| n.rest_local.clone()).collect();
        for (_, (node, rot)) in overlay {
            locals[node.0 as usize].rotation = *rot;
        }
        let mut globals = vec![crate::asset::identity_matrix(); sk.nodes.len()];
        compute_global_transforms(sk, &locals, &mut globals);
        let arm_dir = |upper: usize, lower: usize| {
            let d = [
                globals[lower][3][0] - globals[upper][3][0],
                globals[lower][3][1] - globals[upper][3][1],
                globals[lower][3][2] - globals[upper][3][2],
            ];
            let len = vec3_length(&d);
            -d[1] / len
        };
        (arm_dir(4, 5).asin(), arm_dir(7, 8).asin())
    }

    #[test]
    fn a_pose_overlay_drops_t_pose_arms_to_target_elevation() {
        let (sk, hm) = arm_skeleton([0.25, 0.0, 0.0]);
        let mut st = RetargetState::default();
        ensure_rest_cache(&mut st, &sk, Some(&hm));
        assert_eq!(st.arelax_local.len(), 2, "both arms get an overlay");
        let (l, r) = overlay_arm_elevation(&sk, &st.arelax_local);
        let want = a_pose_drop_rad();
        assert!((l - want).abs() < 1e-3, "left elevation {l} (want {want})");
        assert!((r - want).abs() < 1e-3, "right elevation {r} (want {want})");
    }

    #[test]
    fn a_pose_overlay_leaves_already_low_arms_at_bind() {
        // Bind arms 45° below horizontal — already at the target.
        let d = [
            0.25 * (45.0f32).to_radians().cos(),
            -0.25 * (45.0f32).to_radians().sin(),
            0.0,
        ];
        let (sk, hm) = arm_skeleton(d);
        let mut st = RetargetState::default();
        ensure_rest_cache(&mut st, &sk, Some(&hm));
        assert!(
            st.arelax_local.is_empty(),
            "rig bound at the target elevation must not be rotated further"
        );
    }

    fn pose_at(y: f32) -> Vec<Transform> {
        vec![Transform {
            translation: [0.0, y, 0.0],
            rotation: [0.0, 0.38, 0.0, 0.92],
            scale: [1.0, 1.0, 1.0],
        }]
    }

    #[test]
    fn relax_holds_capture_then_settles_on_target() {
        let target = pose_at(0.0);
        let captured = pose_at(0.5);
        let mut relax = RelaxState::default();

        // Fresh load: no driven frame before, nothing to capture.
        relax.capture(&captured);
        let mut locals = pose_at(99.0);
        relax.step_blend(0.0, &target, &mut locals);
        assert!((locals[0].translation[1] - 0.0).abs() < 1e-6);

        // Driven frame, then the driver stops: the outgoing pose is
        // held exactly on the first relax frame (alpha 0).
        relax.mark_driven();
        relax.capture(&captured);
        relax.step_blend(0.0, &target, &mut locals);
        assert!((locals[0].translation[1] - 0.5).abs() < 1e-6);

        // Well past the duration: settled on the target.
        for _ in 0..100 {
            relax.step_blend(0.1, &target, &mut locals);
        }
        assert!((locals[0].translation[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn relax_midpoint_blends_both_ends() {
        let target = pose_at(0.0);
        let captured = pose_at(1.0);
        let mut relax = RelaxState::default();
        relax.mark_driven();
        relax.capture(&captured);
        let mut locals = captured.clone();
        relax.step_blend(0.45, &target, &mut locals);
        let t = (0.45f32 / relax_duration_s()).clamp(0.0, 1.0);
        let want = t * t * (3.0 - 2.0 * t);
        assert!((locals[0].translation[1] - (1.0 - want)).abs() < 1e-5);
        // Rotation moved off both ends.
        let d_target = locals[0].rotation[1] - target[0].rotation[1];
        let d_captured = locals[0].rotation[1] - captured[0].rotation[1];
        assert!(d_target.abs() > 1e-3 && d_captured.abs() > 1e-3);
    }
}
