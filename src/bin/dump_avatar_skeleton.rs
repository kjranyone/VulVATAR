//! Dump an avatar's rest skeleton + humanoid mapping as JSON: per node
//! (name, parent, humanoid bone, rest local translation/rotation, world
//! rest position) plus the spine-chain geometry (Hips→Chest→Neck→Head
//! segment lean angles). Purpose: localize whether a visibly reclining
//! avatar comes from the asset's rest pose, a humanoid mis-mapping, or the
//! retarget — before touching any of them.
//!
//! Usage: dump_avatar_skeleton <avatar.(fbx|vrm)> [out.json]

use std::sync::Arc;

use vulvatar_lib::asset::{AvatarAsset, HumanoidBone, NodeId};

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: dump_avatar_skeleton <avatar.(fbx|vrm)> [out.json]");
        std::process::exit(2);
    };
    let out = args
        .next()
        .unwrap_or_else(|| "diagnostics/avatar_skeleton.json".to_string());

    let asset: Arc<AvatarAsset> = if path.to_ascii_lowercase().ends_with(".fbx") {
        vulvatar_lib::asset::fbx::FbxAssetLoader::new()
            .load(&path)
            .expect("load FBX")
    } else {
        vulvatar_lib::asset::vrm::VrmAssetLoader::new()
            .load(&path)
            .expect("load VRM")
    };

    let sk = &asset.skeleton;
    // World rest transforms (parent-first walk, same recipe as retarget).
    let mut wpos = vec![[0.0f32; 3]; sk.nodes.len()];
    let mut wrot = vec![[0.0f32, 0.0, 0.0, 1.0]; sk.nodes.len()];
    let mut stack: Vec<usize> = sk.root_nodes.iter().map(|r| r.0 as usize).collect();
    while let Some(i) = stack.pop() {
        if i >= sk.nodes.len() {
            continue;
        }
        let (pr, pp) = match sk.nodes[i].parent {
            Some(NodeId(p)) => (wrot[p as usize], wpos[p as usize]),
            None => ([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
        };
        let l = &sk.nodes[i].rest_local;
        wrot[i] = vulvatar_lib::math_utils::quat_normalize(&vulvatar_lib::math_utils::quat_mul(
            &pr,
            &l.rotation,
        ));
        let off = vulvatar_lib::math_utils::quat_rotate_vec3(&pr, &l.translation);
        for k in 0..3 {
            wpos[i][k] = pp[k] + off[k];
        }
        for c in &sk.nodes[i].children {
            stack.push(c.0 as usize);
        }
    }

    let mut nodes = Vec::new();
    for (i, n) in sk.nodes.iter().enumerate() {
        let l = &n.rest_local;
        nodes.push(serde_json::json!({
            "i": i,
            "name": n.name,
            "parent": n.parent.map(|NodeId(p)| p as u32),
            "humanoid": n.humanoid_bone.map(format_hb),
            "t_local": l.translation,
            "r_local": l.rotation,
            "p_world": wpos[i],
        }));
    }

    // Spine chain lean profile + humanoid map summary.
    let bone_pos = |b: HumanoidBone| -> Option<[f32; 3]> {
        asset
            .humanoid
            .as_ref()
            .and_then(|h| h.bone_map.get(&b))
            .map(|n| wpos[n.0 as usize])
    };
    let seg = |a: HumanoidBone, b: HumanoidBone| -> Option<serde_json::Value> {
        let (pa, pb) = (bone_pos(a)?, bone_pos(b)?);
        let (dy, dz) = (pb[1] - pa[1], pb[2] - pa[2]);
        let lean = dz.atan2(dy).to_degrees(); // + = leaning away from camera (recline)
        Some(serde_json::json!({
            "from": format_hb(a), "to": format_hb(b),
            "dy": dy, "dz": dz, "lean_deg": lean,
        }))
    };
    let mut chain = Vec::new();
    for (a, b) in [
        (HumanoidBone::Hips, HumanoidBone::Spine),
        (HumanoidBone::Spine, HumanoidBone::Chest),
        (HumanoidBone::Chest, HumanoidBone::UpperChest),
        (HumanoidBone::UpperChest, HumanoidBone::Neck),
        (HumanoidBone::Chest, HumanoidBone::Neck),
        (HumanoidBone::Neck, HumanoidBone::Head),
    ] {
        if let Some(v) = seg(a, b) {
            chain.push(v);
        }
    }
    let map: Vec<String> = asset
        .humanoid
        .as_ref()
        .map(|h| {
            h.bone_map
                .iter()
                .map(|(b, n)| {
                    let name = &sk.nodes[n.0 as usize].name;
                    format!("{} -> node {} {:?}", format_hb(*b), n.0, name)
                })
                .collect()
        })
        .unwrap_or_default();

    let doc = serde_json::json!({
        "path": path,
        "nodes": nodes,
        "spine_chain": chain,
        "humanoid_map": map,
    });
    let p = std::path::Path::new(&out);
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    std::fs::write(p, serde_json::to_string_pretty(&doc).unwrap())
        .expect("write json");
    println!("wrote {} ({} nodes)", out, sk.nodes.len());
    for s in &chain {
        println!(
            "  {} -> {}: dy {:+.3} dz {:+.3} lean {:+.1} deg",
            s["from"].as_str().unwrap(),
            s["to"].as_str().unwrap(),
            s["dy"].as_f64().unwrap() as f32,
            s["dz"].as_f64().unwrap() as f32,
            s["lean_deg"].as_f64().unwrap() as f32,
        );
    }
}

fn format_hb(b: HumanoidBone) -> String {
    format!("{b:?}")
}
