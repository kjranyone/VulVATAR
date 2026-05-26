//! DQS antipodal-cancellation reproducer.
//!
//! Loads `sample_data/AvatarSample_A.vrm`, applies a synthetic
//! "raise right arm above head" pose, computes per-bone skinning
//! matrices, and runs an exact CPU mirror of the `transform_cs`
//! DQS math (`src/renderer/pipeline.rs:cloth_constraint_apply_cs`
//! → the skinning shader at the bottom of `pipeline.rs`).
//!
//! Reports, for every vertex weighted to a right-arm bone:
//!
//! * Whether the H-4 fallback (`length(acc_real) < 1e-6`) triggers
//!   — i.e. the antipodal flip cancelled the contributing
//!   quaternions and `normalize(acc_real)` would have returned NaN.
//! * The world-space distance between the pass-7 fallback (`pos`
//!   emitted raw, in mesh-local) and a Linear Blend Skinning (LBS)
//!   fallback that respects the skinning matrices. A large delta
//!   means the pass-7 fallback parks the vertex far from where the
//!   rest of the skinned mesh lives — exactly the "sleeve stretches
//!   across the screen" failure mode visible in `sample_data/image.png`.
//!
//! Usage:
//!   cargo run --bin diagnose_dqs
//!
//! Output:
//!   stdout summary; non-zero exit if any vertex hits the fallback.

use std::sync::Arc;

use vulvatar_lib::asset::vrm::VrmAssetLoader;
use vulvatar_lib::asset::{HumanoidBone, NodeId, Transform};
use vulvatar_lib::avatar::{AvatarInstance, AvatarInstanceId};
use vulvatar_lib::math_utils::{mat4_rotation_to_quat, quat_normalize, Mat4, Quat};

const VRM_PATH: &str = "sample_data/AvatarSample_A.vrm";
const WEIGHT_EPS: f32 = 1.0e-4; // matches transform_cs WEIGHT_EPS
const ACC_LEN_EPS: f32 = 1.0e-6; // matches transform_cs H-4 guard

fn main() -> Result<(), String> {
    let loader = VrmAssetLoader::new();
    let asset = loader
        .load(VRM_PATH)
        .map_err(|e| format!("load {VRM_PATH}: {e}"))?;
    let humanoid = asset
        .humanoid
        .as_ref()
        .ok_or_else(|| "VRM has no humanoid map".to_string())?;

    let mut avatar = AvatarInstance::new(AvatarInstanceId(1), Arc::clone(&asset));
    avatar.build_base_pose();

    // Synthetic "raise right arm over head" pose: rotate the
    // right shoulder + upper arm so the arm points roughly straight
    // up. The bones rotated here are chosen to be the same ones the
    // pose solver targets when the user lifts their arm in front of
    // the webcam (matching `sample_data/image.png`).
    //
    // VRM "right arm" is on the avatar's right (screen-left for a
    // front-facing avatar). In rest pose the arm hangs down along
    // -Y; we rotate around +Z (out of screen) by roughly +π so it
    // points up along +Y. A 180° rotation maximises the chance the
    // antipodal-flip pairing in DQS cancels — that's exactly the
    // pathology pass-7 H-4 was added to guard against.
    let r_shoulder = lookup_bone_idx(humanoid, HumanoidBone::RightShoulder)?;
    let r_upper = lookup_bone_idx(humanoid, HumanoidBone::RightUpperArm)?;
    let r_lower = lookup_bone_idx(humanoid, HumanoidBone::RightLowerArm)?;
    let r_hand = lookup_bone_idx(humanoid, HumanoidBone::RightHand)?;

    // A realistic "right hand to face" pose has the shoulder
    // moderately rotated, the upper arm raised and twisted, the
    // elbow bent ~90°, and the hand rotated to point inward toward
    // the head. Mixing rotation axes across bones is what makes the
    // DQS antipodal cancellation reachable — uniform rotations on
    // a single axis cannot cancel after the hemisphere flip.
    set_local_rotation(
        &mut avatar,
        r_shoulder,
        axis_angle([0.0, 0.0, 1.0], 0.4),
    );
    set_local_rotation(
        &mut avatar,
        r_upper,
        // Compose: raise the arm (+Z) + twist the humerus (X).
        quat_mul_local(
            axis_angle([0.0, 0.0, 1.0], std::f32::consts::FRAC_PI_2 * 1.6),
            axis_angle([1.0, 0.0, 0.0], std::f32::consts::FRAC_PI_2),
        ),
    );
    set_local_rotation(
        &mut avatar,
        r_lower,
        // Elbow bend (around the forearm's local Y).
        axis_angle([0.0, 1.0, 0.0], -std::f32::consts::FRAC_PI_2),
    );
    set_local_rotation(
        &mut avatar,
        r_hand,
        // Wrist rotates inward / palm faces face.
        quat_mul_local(
            axis_angle([1.0, 0.0, 0.0], std::f32::consts::FRAC_PI_4),
            axis_angle([0.0, 0.0, 1.0], std::f32::consts::FRAC_PI_4),
        ),
    );

    avatar.compute_global_pose();
    avatar.build_skinning_matrices();

    let arm_node_indices = [r_shoulder, r_upper, r_lower, r_hand];

    let skin_mats = &avatar.pose.skinning_matrices;
    println!(
        "loaded {} (nodes={}, skinning_matrices={})",
        VRM_PATH,
        asset.skeleton.nodes.len(),
        skin_mats.len()
    );
    println!(
        "right-arm bone node indices: shoulder={r_shoulder} upper={r_upper} lower={r_lower} hand={r_hand}",
    );
    // Build the set of all spring-bone joint node indices so we can
    // cross-reference vertex skinning weights against them. Spring
    // bones that no mesh vertex weights to are dead-weight metadata —
    // they consume sim cycles but contribute nothing visible. The ones
    // that DO have vertices weighted to them are the candidates for
    // "rotation writeback puts the joint in a bad place and the
    // weighted mesh follows".
    let mut spring_joint_idx_to_chain: std::collections::HashMap<usize, (usize, usize)> =
        std::collections::HashMap::new();
    for (ci, chain) in asset.spring_bones.iter().enumerate() {
        for (ji, joint_node) in chain.joints.iter().enumerate() {
            spring_joint_idx_to_chain.insert(joint_node.0 as usize, (ci, ji));
        }
    }

    println!("spring_bone chains: {}", asset.spring_bones.len());
    // Inventory which spring chains actually drive mesh vertices.
    // For each chain, count how many vertices have any non-trivial
    // weight on any of the chain's joints; report the cumulative
    // weight too because a chain with hundreds of low-weight verts
    // could still drag a noticeable patch of mesh.
    let mut per_chain_vert_count = vec![0usize; asset.spring_bones.len()];
    let mut per_chain_weight = vec![0.0f32; asset.spring_bones.len()];
    for mesh in asset.meshes.iter() {
        for prim in mesh.primitives.iter() {
            let Some(vd) = prim.vertices.as_ref() else { continue };
            for (jw, ji) in vd.joint_weights.iter().zip(vd.joint_indices.iter()) {
                let mut hits_per_chain: std::collections::HashMap<usize, f32> =
                    std::collections::HashMap::new();
                for i in 0..4 {
                    if jw[i] <= WEIGHT_EPS { continue; }
                    if let Some((ci, _)) = spring_joint_idx_to_chain.get(&(ji[i] as usize)) {
                        *hits_per_chain.entry(*ci).or_insert(0.0) += jw[i];
                    }
                }
                for (ci, w) in hits_per_chain {
                    per_chain_vert_count[ci] += 1;
                    per_chain_weight[ci] += w;
                }
            }
        }
    }
    for (i, chain) in asset.spring_bones.iter().enumerate() {
        if per_chain_vert_count[i] == 0 {
            continue; // ignore chains that nothing weights to
        }
        let root_name = chain
            .joints
            .first()
            .and_then(|n| asset.skeleton.nodes.get(n.0 as usize))
            .map(|n| n.name.as_str())
            .unwrap_or("?");
        println!(
            "  chain {i}: root='{root_name}' joints={} weighted_verts={} cumul_w={:.1}",
            chain.joints.len(),
            per_chain_vert_count[i],
            per_chain_weight[i]
        );
    }

    // Catalogue the mesh primitives so we can see whether any of them
    // are bound to right-arm bones with a dense weighting (the sleeve
    // typically has hundreds of vertices weighted to the lower arm).
    println!();
    println!("--- mesh primitives ---");
    for (mi, mesh) in asset.meshes.iter().enumerate() {
        for (pi, prim) in mesh.primitives.iter().enumerate() {
            let Some(vd) = prim.vertices.as_ref() else { continue };
            let mut arm_weight_sum = 0.0f32;
            let mut arm_vertex_count = 0usize;
            for (jw, ji) in vd.joint_weights.iter().zip(vd.joint_indices.iter()) {
                let mut v_arm = 0.0;
                for i in 0..4 {
                    if arm_node_indices.contains(&(ji[i] as usize)) {
                        v_arm += jw[i];
                    }
                }
                if v_arm > WEIGHT_EPS {
                    arm_weight_sum += v_arm;
                    arm_vertex_count += 1;
                }
            }
            let max_ji = vd
                .joint_indices
                .iter()
                .map(|ji| *ji.iter().max().unwrap_or(&0))
                .max()
                .unwrap_or(0);
            let min_ji = vd
                .joint_indices
                .iter()
                .map(|ji| *ji.iter().min().unwrap_or(&0))
                .min()
                .unwrap_or(0);
            println!(
                "  mesh={mi} prim={pi} ('{}') verts={} arm-weighted={} cumulative-arm-w={:.1} ji=[{min_ji}..={max_ji}] morphs={}",
                mesh.name,
                vd.positions.len(),
                arm_vertex_count,
                arm_weight_sum,
                prim.morph_targets.len(),
            );
        }
    }
    println!();

    let mut total_vertices_touched_by_arm = 0usize;
    let mut fallback_triggered = 0usize;
    let mut max_lbs_delta_m = 0.0f32;
    let mut worst_vertex: Option<(usize, usize, usize, f32)> = None;
    let mut min_acc_len = f32::INFINITY;
    let mut min_acc_vertex: Option<(usize, usize, usize, [f32; 4], [u16; 4])> = None;
    // Histogram buckets: log10(acc_len) → count.
    let mut buckets: [usize; 8] = [0; 8]; // <1e-6, 1e-6..1e-5, ..., >=1e-1

    for (mesh_idx, mesh) in asset.meshes.iter().enumerate() {
        for (prim_idx, prim) in mesh.primitives.iter().enumerate() {
            let Some(vd) = prim.vertices.as_ref() else { continue };
            for (vid, (&jw, &ji)) in vd
                .joint_weights
                .iter()
                .zip(vd.joint_indices.iter())
                .enumerate()
            {
                let touches_arm = (0..4).any(|i| {
                    jw[i] > WEIGHT_EPS
                        && arm_node_indices.contains(&(ji[i] as usize))
                });
                if !touches_arm {
                    continue;
                }
                total_vertices_touched_by_arm += 1;

                let pos_local = vd.positions[vid];
                let nrm_local = vd.normals[vid];

                let dqs = dqs_mirror(pos_local, nrm_local, jw, ji, skin_mats);
                let lbs = lbs_mirror(pos_local, jw, ji, skin_mats);

                if dqs.acc_len < min_acc_len {
                    min_acc_len = dqs.acc_len;
                    min_acc_vertex = Some((mesh_idx, prim_idx, vid, jw, ji));
                }
                let b = match dqs.acc_len {
                    x if x < 1e-6 => 0,
                    x if x < 1e-5 => 1,
                    x if x < 1e-4 => 2,
                    x if x < 1e-3 => 3,
                    x if x < 1e-2 => 4,
                    x if x < 1e-1 => 5,
                    x if x < 1.0 => 6,
                    _ => 7,
                };
                buckets[b] += 1;

                if dqs.fallback_triggered {
                    fallback_triggered += 1;
                    // The pass-7 fallback emits pos_local (mesh-local).
                    // LBS emits a properly-skinned world-space position.
                    // Delta = how far the visible mesh stretches.
                    let delta = sub3(lbs, pos_local);
                    let d = len3(delta);
                    if d > max_lbs_delta_m {
                        max_lbs_delta_m = d;
                        worst_vertex = Some((mesh_idx, prim_idx, vid, d));
                    }
                }
            }
        }
    }

    println!();
    println!("--- DQS antipodal-cancellation reproduction ---");
    println!("right-arm vertices examined: {}", total_vertices_touched_by_arm);
    println!("H-4 fallback triggered:      {}", fallback_triggered);
    println!("worst pass-7 fallback gap:   {:.3} m", max_lbs_delta_m);
    if let Some((m, p, v, d)) = worst_vertex {
        println!("worst vertex:                mesh={m} prim={p} vid={v} delta={d:.3} m");
    }
    println!("min |acc_real| across set:   {:.6e}", min_acc_len);
    if let Some((m, p, v, jw, ji)) = min_acc_vertex {
        println!(
            "min-acc vertex:              mesh={m} prim={p} vid={v} jw={:?} ji={:?}",
            jw, ji
        );
    }
    let labels = ["<1e-6", "1e-6..1e-5", "1e-5..1e-4", "1e-4..1e-3", "1e-3..1e-2", "1e-2..1e-1", "1e-1..1", ">=1"];
    println!("|acc_real| histogram:");
    for (i, c) in buckets.iter().enumerate() {
        println!("  {:>11}: {}", labels[i], c);
    }
    println!();

    if fallback_triggered > 0 {
        println!(
            "HYPOTHESIS CONFIRMED — DQS antipodal flip cancels for some right-arm vertices.\n\
             The pass-7 H-4 guard emits mesh-local `pos` raw, leaving the affected\n\
             vertices behind while the rest of the sleeve skins to the raised-arm\n\
             world position. That is the visible stretch in sample_data/image.png."
        );
        std::process::exit(1);
    } else {
        println!(
            "no vertex hit the H-4 fallback on this synthetic pose. The bug may be\n\
             reachable only via a different bone configuration (e.g. with twist or\n\
             with face/finger bones contributing), or the cause may lie elsewhere."
        );
        Ok(())
    }
}

struct DqsResult {
    /// Whether `length(acc_real) < ACC_LEN_EPS`. True ⇒ pass-7
    /// fallback would emit `pos` (mesh-local) instead of a
    /// world-space skinned position.
    fallback_triggered: bool,
    /// Magnitude of the antipodal-flipped quaternion sum, before
    /// normalisation. Smaller ⇒ closer to the H-4 trigger.
    acc_len: f32,
}

/// Exact CPU mirror of the DQS math in `transform_cs` (skipping the
/// `total_w < WEIGHT_EPS` early-return — the caller already filters
/// to vertices with non-trivial right-arm weight).
fn dqs_mirror(
    _pos: [f32; 3],
    _nrm: [f32; 3],
    jw: [f32; 4],
    ji: [u16; 4],
    skin_mats: &[Mat4],
) -> DqsResult {
    // Pick the antipodal reference: first contributing joint.
    let mut ref_real: Quat = [0.0, 0.0, 0.0, 1.0];
    for i in 0..4 {
        if jw[i] > WEIGHT_EPS {
            ref_real = mat4_rotation_to_quat(&skin_mats[ji[i] as usize]);
            break;
        }
    }

    // Accumulate sign-flipped quaternions weighted by joint weight.
    let mut acc_real: Quat = [0.0, 0.0, 0.0, 0.0];
    for i in 0..4 {
        let wi = jw[i];
        if wi <= WEIGHT_EPS {
            continue;
        }
        let qi = mat4_rotation_to_quat(&skin_mats[ji[i] as usize]);
        let dot = qi[0] * ref_real[0]
            + qi[1] * ref_real[1]
            + qi[2] * ref_real[2]
            + qi[3] * ref_real[3];
        let s = if dot >= 0.0 { 1.0 } else { -1.0 };
        for k in 0..4 {
            acc_real[k] += wi * s * qi[k];
        }
    }
    let acc_len = (acc_real[0] * acc_real[0]
        + acc_real[1] * acc_real[1]
        + acc_real[2] * acc_real[2]
        + acc_real[3] * acc_real[3])
        .sqrt();
    let _ = quat_normalize; // imported for downstream parity tests
    DqsResult {
        fallback_triggered: acc_len < ACC_LEN_EPS,
        acc_len,
    }
}

/// LBS reference. Output is world-space position.
fn lbs_mirror(pos: [f32; 3], jw: [f32; 4], ji: [u16; 4], skin_mats: &[Mat4]) -> [f32; 3] {
    let mut blended = [[0.0f32; 4]; 4];
    for i in 0..4 {
        let wi = jw[i];
        if wi <= WEIGHT_EPS {
            continue;
        }
        let m = &skin_mats[ji[i] as usize];
        for r in 0..4 {
            for c in 0..4 {
                blended[r][c] += wi * m[r][c];
            }
        }
    }
    // Matrices in this codebase are stored as column-major
    // (`m[col][row]`), matching GLSL and `mat4_rotation_to_quat`'s
    // accessors. Apply as a column-major mat4 × vec4(pos, 1.0).
    let v = [pos[0], pos[1], pos[2], 1.0];
    let mut out = [0.0f32; 3];
    for r in 0..3 {
        let mut acc = 0.0;
        for c in 0..4 {
            acc += blended[c][r] * v[c];
        }
        out[r] = acc;
    }
    out
}

fn lookup_bone_idx(
    humanoid: &vulvatar_lib::asset::HumanoidMap,
    bone: HumanoidBone,
) -> Result<usize, String> {
    humanoid
        .bone_map
        .get(&bone)
        .copied()
        .map(|n| n.0 as usize)
        .ok_or_else(|| format!("VRM humanoid map missing {bone:?}"))
}

fn set_local_rotation(avatar: &mut AvatarInstance, node_idx: usize, rot: Quat) {
    let t: &mut Transform = &mut avatar.pose.local_transforms[node_idx];
    t.rotation = rot;
}

fn axis_angle(axis: [f32; 3], angle: f32) -> Quat {
    let half = angle * 0.5;
    let s = half.sin();
    quat_normalize(&[axis[0] * s, axis[1] * s, axis[2] * s, half.cos()])
}

/// Local-multiply two quaternions: returns `lhs ∘ rhs` (apply `rhs`
/// first, then `lhs`). Same convention as `math_utils::quat_mul`.
fn quat_mul_local(lhs: Quat, rhs: Quat) -> Quat {
    vulvatar_lib::math_utils::quat_mul(&lhs, &rhs)
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn len3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[allow(dead_code)]
fn _node_id_type_hint(_: NodeId) {}
