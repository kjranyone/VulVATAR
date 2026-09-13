use std::path::Path;
use std::sync::Arc;

use vulvatar_lib::asset::fbx::FbxAssetLoader;
use vulvatar_lib::asset::{ColliderShape, HumanoidBone};
use vulvatar_lib::avatar::pose::compute_global_transforms;
use vulvatar_lib::avatar::{AvatarInstance, AvatarInstanceId};
use vulvatar_lib::simulation::spring::SpringTuning;

fn main() -> Result<(), String> {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let input_path = args
        .next()
        .unwrap_or_else(|| "sample_data/YUMEKA_v1.0.3/FBX/Yumeka_v1.0.3.fbx".to_string());
    let output_dir = args
        .next()
        .unwrap_or_else(|| "diagnostics/hair_collider".to_string());
    if output_dir.contains("validation_images") {
        return Err(
            "Diagnostics outputs must not be written to validation_images/ (see AGENTS.md)".into(),
        );
    }

    let loader = FbxAssetLoader::new();
    let asset = loader
        .load(&input_path)
        .map_err(|e| format!("load failed: {e}"))?;
    println!(
        "loaded {} ({} nodes, {} spring chains, {} colliders, from_cache={})",
        input_path,
        asset.skeleton.nodes.len(),
        asset.spring_bones.len(),
        asset.colliders.len(),
        asset.loaded_from_cache
    );

    // Rest-pose global transforms (what the solver sees before tracking).
    let skeleton = &asset.skeleton;
    let rest_locals: Vec<_> = skeleton
        .nodes
        .iter()
        .map(|n| n.rest_local.clone())
        .collect();    let mut globals = vec![vulvatar_lib::asset::identity_matrix(); skeleton.nodes.len()];
    compute_global_transforms(skeleton, &rest_locals, &mut globals);

    let node_name = |idx: usize| -> String { skeleton.nodes[idx].name.clone() };
    let node_world = |idx: usize| -> [f32; 3] {
        let m = &globals[idx];
        [m[3][0], m[3][1], m[3][2]]
    };

    // Reference heights for "above the shoulders".
    let mut report = String::new();
    let mut shoulder_y = f32::MIN;
    for bone in [
        HumanoidBone::LeftUpperArm,
        HumanoidBone::RightUpperArm,
        HumanoidBone::LeftLowerArm,
        HumanoidBone::RightLowerArm,
        HumanoidBone::Hips,
        HumanoidBone::Spine,
        HumanoidBone::Chest,
        HumanoidBone::UpperChest,
        HumanoidBone::Neck,
        HumanoidBone::Head,
    ] {
        if let Some(idx) = skeleton
            .nodes
            .iter()
            .position(|n| n.humanoid_bone == Some(bone))
        {
            let p = node_world(idx);
            let line = format!("{:?} '{}' y={:.4}\n", bone, node_name(idx), p[1]);
            println!("{}", line.trim_end());
            report.push_str(&line);
            if matches!(
                bone,
                HumanoidBone::LeftUpperArm | HumanoidBone::RightUpperArm
            ) {
                shoulder_y = shoulder_y.max(p[1]);
            }
        }
    }
    println!("shoulder_y = {:.4}", shoulder_y);
    report.push_str(&format!("shoulder_y = {:.4}\n\n", shoulder_y));

    // Collider envelopes in world space.
    report.push_str("## Colliders (world, rest pose)\n");
    let mut collider_info: Vec<(u64, String, f32)> = Vec::new();
    for c in &asset.colliders {
        let idx = c.node.0 as usize;
        if idx >= globals.len() {
            continue;
        }
        let m = &globals[idx];
        let node_pos = [m[3][0], m[3][1], m[3][2]];
        // Same direction transform as the spring solver (upper-left 3x3,
        // includes scale).
        let off = [
            m[0][0] * c.offset[0] + m[1][0] * c.offset[1] + m[2][0] * c.offset[2],
            m[0][1] * c.offset[0] + m[1][1] * c.offset[1] + m[2][1] * c.offset[2],
            m[0][2] * c.offset[0] + m[1][2] * c.offset[1] + m[2][2] * c.offset[2],
        ];
        let center = [
            node_pos[0] + off[0],
            node_pos[1] + off[1],
            node_pos[2] + off[2],
        ];
        let (desc, top_y) = match c.shape {
            ColliderShape::Sphere { radius } => {
                (format!("sphere r={:.3}", radius), center[1] + radius)
            }
            ColliderShape::Capsule { radius, height } => {
                let up = [
                    m[0][1], m[1][1], m[2][1],
                ];
                let len = (up[0] * up[0] + up[1] * up[1] + up[2] * up[2]).sqrt();
                let up = if len > 1e-8 {
                    [up[0] / len, up[1] / len, up[2] / len]
                } else {
                    [0.0, 1.0, 0.0]
                };
                let hh = height * 0.5;
                let a = [
                    center[0] - up[0] * hh,
                    center[1] - up[1] * hh,
                    center[2] - up[2] * hh,
                ];
                let b = [
                    center[0] + up[0] * hh,
                    center[1] + up[1] * hh,
                    center[2] + up[2] * hh,
                ];
                (
                    format!(
                        "capsule r={:.3} h={:.3} a=({:.3},{:.3},{:.3}) b=({:.3},{:.3},{:.3})",
                        radius, height, a[0], a[1], a[2], b[0], b[1], b[2]
                    ),
                    center[1] + hh + radius,
                )
            }
        };
        let above = top_y > shoulder_y;
        let line = format!(
            "collider id={} on '{}' center=({:.3},{:.3},{:.3}) {} top_y={:.4}{}\n",
            c.id.0,
            node_name(idx),
            center[0],
            center[1],
            center[2],
            desc,
            top_y,
            if above {
                format!("  <-- ABOVE SHOULDER by {:+.3} m", top_y - shoulder_y)
            } else {
                String::new()
            },
        );
        println!("{}", line.trim_end());
        report.push_str(&line);
        collider_info.push((c.id.0, node_name(idx), top_y));
    }

    // Per spring chain: collider references.
    report.push_str("\n## Spring chains -> collider refs\n");
    for sb in &asset.spring_bones {
        if sb.collider_refs.is_empty() {
            continue;
        }
        let root = node_name(sb.chain_root.0 as usize);
        let refs: Vec<String> = sb
            .collider_refs
            .iter()
            .map(|r| {
                asset
                    .colliders
                    .iter()
                    .find(|c| c.id == r.id)
                    .map(|c| format!("#{}({})", c.id.0, node_name(c.node.0 as usize)))
                    .unwrap_or_else(|| format!("#{}(missing)", r.id.0))
            })
            .collect();
        let line = format!(
            "chain '{}' joints={} r={:.3} colliders: {}\n",
            root,
            sb.joints.len(),
            sb.radius,
            refs.join(", ")
        );
        println!("{}", line.trim_end());
        report.push_str(&line);
    }

    // Settle the solver at rest and measure where each chain's tail ends
    // up relative to every referenced collider envelope.
    let mut instance = AvatarInstance::new(AvatarInstanceId(1), Arc::clone(&asset));
    instance.pose.local_transforms = rest_locals.clone();
    instance.pose.global_transforms = globals.clone();
    let tuning = SpringTuning::default();
    for _ in 0..240 {
        vulvatar_lib::simulation::spring::step_spring_bones(
            1.0 / 60.0,
            &mut instance,
            &[],
            &tuning,
            [0.0, -1.0, 0.0],
            1.0,
            None,
        );
    }

    report.push_str("\n## After 4 s settle: tail position vs referenced colliders\n");
    for (ci, sb) in asset.spring_bones.iter().enumerate() {
        if sb.joints.len() < 2 || sb.collider_refs.is_empty() {
            continue;
        }
        let state = &instance.secondary_motion.spring_states[ci];
        let tail = state.positions[sb.joints.len() - 1];
        let root = node_name(sb.chain_root.0 as usize);
        let mut contacts = Vec::new();
        for r in &sb.collider_refs {
            if let Some(c) = asset.colliders.iter().find(|c| c.id == r.id) {
                let idx = c.node.0 as usize;
                if idx >= globals.len() {
                    continue;
                }
                let m = &globals[idx];
                let node_pos = [m[3][0], m[3][1], m[3][2]];
                let off = [
                    m[0][0] * c.offset[0] + m[1][0] * c.offset[1] + m[2][0] * c.offset[2],
                    m[0][1] * c.offset[0] + m[1][1] * c.offset[1] + m[2][1] * c.offset[2],
                    m[0][2] * c.offset[0] + m[1][2] * c.offset[1] + m[2][2] * c.offset[2],
                ];
                let center = [
                    node_pos[0] + off[0],
                    node_pos[1] + off[1],
                    node_pos[2] + off[2],
                ];
                let (radius, hh, axis) = match c.shape {
                    ColliderShape::Sphere { radius } => (radius, 0.0, [0.0, 1.0, 0.0]),
                    ColliderShape::Capsule { radius, height } => {
                        let up = [m[0][1], m[1][1], m[2][1]];
                        let len = (up[0] * up[0] + up[1] * up[1] + up[2] * up[2]).sqrt();
                        let up = if len > 1e-8 {
                            [up[0] / len, up[1] / len, up[2] / len]
                        } else {
                            [0.0, 1.0, 0.0]
                        };
                        (radius, height * 0.5, up)
                    }
                };
                // Distance from tail to capsule segment (degenerate -> sphere).
                let a = [
                    center[0] - axis[0] * hh,
                    center[1] - axis[1] * hh,
                    center[2] - axis[2] * hh,
                ];
                let b = [
                    center[0] + axis[0] * hh,
                    center[1] + axis[1] * hh,
                    center[2] + axis[2] * hh,
                ];
                let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let ab2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
                let t = if ab2 > 1e-12 {
                    (((tail[0] - a[0]) * ab[0]
                        + (tail[1] - a[1]) * ab[1]
                        + (tail[2] - a[2]) * ab[2])
                        / ab2)
                        .clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let cp = [
                    a[0] + ab[0] * t,
                    a[1] + ab[1] * t,
                    a[2] + ab[2] * t,
                ];
                let d = [
                    tail[0] - cp[0],
                    tail[1] - cp[1],
                    tail[2] - cp[2],
                ];
                let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let gap = dist - (radius + sb.radius);
                if gap < 0.005 {
                    contacts.push(format!(
                        "#{}({}) gap={:+.4} m tail_y={:.4}",
                        c.id.0,
                        node_name(idx),
                        gap,
                        tail[1]
                    ));
                }
            }
        }
        if !contacts.is_empty() {
            let line = format!(
                "chain '{}' tail=({:.3},{:.3},{:.3}) TOUCHING: {}\n",
                root,
                tail[0],
                tail[1],
                tail[2],
                contacts.join("; ")
            );
            println!("{}", line.trim_end());
            report.push_str(&line);
        }
    }

    // Also show what the current fallback would build from scratch (no
    // unitypackage): the on-disk cache may predate the current synthesis.
    let (fresh_chains, fresh_colliders) = vulvatar_lib::asset::vrc::build_spring_bones_and_colliders(
        skeleton,
        &vulvatar_lib::asset::vrc::ParsedVrcData::default(),
    );
    println!(
        "\n== current-code fallback (no cache): {} chains, {} colliders ==",
        fresh_chains.len(),
        fresh_colliders.len()
    );
    report.push_str(&format!(
        "\n## Current-code fallback (would replace the cache on invalidation): {} colliders\n",
        fresh_colliders.len()
    ));
    for c in &fresh_colliders {
        let idx = c.node.0 as usize;
        let (shape_desc, top_y) = match c.shape {
            ColliderShape::Sphere { radius } => (format!("sphere r={:.3}", radius), node_world(idx)[1] + c.offset[1] + radius),
            ColliderShape::Capsule { radius, height } => (
                format!("capsule r={:.3} h={:.3} offset=({:.3},{:.3},{:.3})", radius, height, c.offset[0], c.offset[1], c.offset[2]),
                node_world(idx)[1] + c.offset[1] + height * 0.5 + radius,
            ),
        };
        let line = format!(
            "id={} on '{}' y={:.4} {} top_y={:.4}{}\n",
            c.id.0,
            node_name(idx),
            node_world(idx)[1],
            shape_desc,
            top_y,
            if top_y > shoulder_y {
                format!("  <-- ABOVE SHOULDER by {:+.3} m", top_y - shoulder_y)
            } else {
                String::new()
            },
        );
        println!("{}", line.trim_end());
        report.push_str(&line);
    }

    // Mesh budget: what a mesh-faithful collision layer would have to
    // consume per frame.
    let total_verts: usize = asset
        .meshes
        .iter()
        .flat_map(|m| m.primitives.iter())
        .map(|p| p.vertex_count as usize)
        .sum();
    println!(
        "\n== mesh budget: {} meshes, {} primitives, {} verts total ==",
        asset.meshes.len(),
        asset.meshes.iter().map(|m| m.primitives.len()).sum::<usize>(),
        total_verts
    );
    let mut mesh_lines = Vec::new();
    // Top 8 largest primitives only.
    let mut prim_sizes: Vec<(String, usize)> = Vec::new();
    for (mi, m) in asset.meshes.iter().enumerate() {
        for (pi, p) in m.primitives.iter().enumerate() {
            prim_sizes.push((format!("mesh[{}].prim[{}]", mi, pi), p.vertex_count as usize));
        }
    }
    prim_sizes.sort_by(|a, b| b.1.cmp(&a.1));
    for (name, n) in prim_sizes.iter().take(8) {
        println!("  {} verts={}", name, n);
        mesh_lines.push(format!("{} verts={}\n", name, n));
    }
    let hair_joints: usize = asset
        .spring_bones
        .iter()
        .filter(|sb| {
            let root = node_name(sb.chain_root.0 as usize).to_lowercase();
            root.contains("hair")
        })
        .map(|sb| sb.joints.len())
        .sum();
    println!(
        "hair chains: {} joints total (spring-bone query count per step)",
        hair_joints
    );
    report.push_str("\n## Mesh budget\n");
    for l in mesh_lines {
        report.push_str(&l);
    }
    report.push_str(&format!(
        "total verts = {}\nhair joints = {}\n",
        total_verts, hair_joints
    ));

    std::fs::create_dir_all(&output_dir)
        .map_err(|e| format!("create dir {}: {e}", output_dir))?;
    let out_path = Path::new(&output_dir).join("yumeka_v1.0.3_report.md");
    std::fs::write(&out_path, &report).map_err(|e| format!("write: {e}"))?;
    println!("\nreport: {}", out_path.display());
    Ok(())
}
