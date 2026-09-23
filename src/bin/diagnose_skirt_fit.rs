//! Skirt constraint-fit audit for auto-cloth (Yumeka default).
//!
//! Quantifies the two constraint families that hold the skirt around the
//! body, as groundwork for the 2026-09 constraint redesign (hip dent +
//! front-top jut reported on the desk envelope):
//!
//! * M1 pin binding skew — per pinned particle, the distance between its
//!   bone-frame pin target and the vertex's OWN LBS-skinned position
//!   under the posed skeleton. This is the band-vs-body mismatch behind
//!   both symptoms: the pin band is bound 100% to one node (skirt_root /
//!   hips), so any pose where the skinned surface moves differently from
//!   that single bone's rigid frame leaves the band sheared against the
//!   body/jacket. Reported over a static pose sweep, split by band
//!   height and by the front half (the reported jut region).
//! * M2 free-hem collider coverage — which body colliders exist, and
//!   during a CPU-sim pose sequence (rest → lean back → sit), how deep
//!   free particles penetrate the resolved colliders. Pins hard-snap
//!   after collision in the solver, so M2 measures only the free hem.
//! * Pin-band weight audit — dominant-joint histogram of the pinned
//!   vertices (what a multi-bone pin redistribution would actually
//!   distribute over).
//!
//! Runs the app path (`attach_auto_cloth`) with the backend forced to
//! CPU so the sim steps headless; the GPU solver shares the same
//! formulas (formula-parity tests).
//!
//! Output: `<out_dir>/metrics.csv` + `summary.md` (default
//! `diagnostics/skirt_fit/`).

use std::path::PathBuf;

use vulvatar_lib::asset::{ColliderShape, HumanoidBone, SkeletonAsset, Transform, Vec3};
use vulvatar_lib::math_utils::{
    quat_conjugate, quat_mul, quat_normalize, quat_rotate_vec3, vec3_add, vec3_sub,
};
use vulvatar_lib::simulation::auto_cloth::attach_auto_cloth;
use vulvatar_lib::simulation::cloth_gpu_boundary::ClothSolverBackend;

type Quat = [f32; 4];

fn main() -> Result<(), String> {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let input_path = args
        .next()
        .unwrap_or_else(|| "sample_data/YUMEKA_v1.0.1/FBX/Yumeka_v1.0.fbx".to_string());
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| "diagnostics/skirt_fit".to_string()));
    if out_dir.to_string_lossy().contains("validation_images") {
        return Err("diagnostics must not be written to validation_images/".into());
    }
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("mkdir {}: {e}", out_dir.display()))?;

    println!("Loading {input_path}");
    let loader = vulvatar_lib::asset::fbx::FbxAssetLoader::new();
    let asset = loader.load(&input_path).map_err(|e| format!("load: {e}"))?;
    let mut avatar =
        vulvatar_lib::avatar::AvatarInstance::new(vulvatar_lib::avatar::AvatarInstanceId(1), asset);
    avatar.build_base_pose();
    avatar.compute_global_pose();
    avatar.build_skinning_matrices();

    // ---- collider inventory (rest pose) --------------------------------
    println!(
        "\n=== collider inventory: {} colliders ===",
        avatar.asset.colliders.len()
    );
    for c in &avatar.asset.colliders {
        let node = avatar
            .asset
            .skeleton
            .nodes
            .get(c.node.0 as usize)
            .map(|n| n.name.as_str())
            .unwrap_or("?");
        let (shape, r) = match c.shape {
            ColliderShape::Sphere { radius } => (format!("sphere  r={radius:.3}"), radius),
            ColliderShape::Capsule { radius, height } => {
                (format!("capsule r={radius:.3} h={height:.3}"), radius)
            }
        };
        let _ = r;
        println!("  #{:3} node {:<22} {shape}", c.id.0, node);
    }
    if avatar.asset.colliders.is_empty() {
        println!("  (none — the FBX carries no VRChat collider data)");
    }

    // ---- attach the APP-path auto cloth, force CPU, zero wind ----------
    let attached = attach_auto_cloth(&mut avatar);
    println!("\nauto-cloth attached: {attached} garment(s)");
    if attached == 0 {
        return Err("no skirt classified for auto-cloth".into());
    }
    let slot_idx = avatar
        .cloth_overlays
        .iter()
        .enumerate()
        .max_by_key(|(_, s)| s.sim.particles.len())
        .map(|(i, _)| i)
        .ok_or("no cloth slots after attach")?;
    {
        let slot = &mut avatar.cloth_overlays[slot_idx];
        slot.state.solver_backend = ClothSolverBackend::Cpu;
        slot.sim.wind_response = 0.0;
        let sim = &slot.sim;
        println!(
            "slot {}: {} particles, {} pins, {} constraints, backend {:?}",
            slot_idx,
            sim.particles.len(),
            sim.pin_targets.len(),
            sim.distance_constraints.len(),
            slot.state.solver_backend
        );
    }

    // Skirt prim vertex data for LBS (particles are 1:1 with vertices).
    let skirt_pid = avatar.cloth_overlays[slot_idx]
        .state
        .target_primitive_id
        .ok_or("cloth slot has no target primitive")?;
    let prim = avatar
        .asset
        .meshes
        .iter()
        .flat_map(|m| m.primitives.iter())
        .find(|p| p.id == skirt_pid)
        .ok_or("skirt prim not found")?
        .clone();
    let vd = prim
        .vertices
        .as_ref()
        .ok_or("skirt prim has no CPU vertex data")?
        .clone();

    // Body prim vertex data for the cone-poke metric (M3).
    let body_vd = avatar
        .asset
        .body_primitive_id
        .and_then(|pid| {
            avatar
                .asset
                .meshes
                .iter()
                .flat_map(|m| m.primitives.iter())
                .find(|p| p.id == pid)
                .and_then(|p| p.vertices.clone())
        })
        .ok_or("body prim / vertices not found")?;
    println!(
        "\nbody prim: {} vertices (M3 cone-poke surface)",
        body_vd.positions.len()
    );

    // ---- pin-band weight audit -----------------------------------------
    println!("\n=== pin-band dominant-joint histogram ===");
    {
        let sim = &avatar.cloth_overlays[slot_idx].sim;
        // Which node do the pins actually bind to (and where does it sit
        // in the hierarchy)?
        let mut bind_nodes: std::collections::BTreeMap<usize, usize> =
            std::collections::BTreeMap::new();
        for t in &sim.pin_targets {
            *bind_nodes.entry(t.node_index).or_default() += 1;
        }
        for (node, count) in &bind_nodes {
            let mut chain = String::new();
            let mut cur = Some(*node);
            while let Some(i) = cur {
                let n = &avatar.asset.skeleton.nodes[i];
                if !chain.is_empty() {
                    chain.push_str(" <- ");
                }
                chain.push_str(&n.name);
                cur = n.parent.map(|p| p.0 as usize);
            }
            println!("  bind node '{}' ({count} pins), chain: {chain}", avatar.asset.skeleton.nodes[*node].name);
        }
    }
    let mut joint_hist: std::collections::BTreeMap<String, (usize, Vec<f32>)> =
        std::collections::BTreeMap::new();
    let mut pinned_rows: Vec<(usize, f32)> = Vec::new(); // (particle, rest y)
    for (vi, p) in avatar.cloth_overlays[slot_idx]
        .sim
        .particles
        .iter()
        .enumerate()
    {
        if !p.pinned {
            continue;
        }
        pinned_rows.push((vi, p.position[1]));
        let mut best = (0usize, 0.0f32);
        if let Some(w) = vd.joint_weights.get(vi) {
            for k in 0..4 {
                if w[k] > best.1 {
                    best = (vd.joint_indices[vi][k] as usize, w[k]);
                }
            }
        }
        let name = avatar
            .asset
            .skeleton
            .nodes
            .get(best.0)
            .map(|n| n.name.clone())
            .unwrap_or_else(|| format!("node#{})", best.0));
        let e = joint_hist.entry(name).or_default();
        e.0 += 1;
        e.1.push(best.1);
    }
    pinned_rows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let band_y_min = pinned_rows.first().map(|r| r.1).unwrap_or(0.0);
    let band_y_max = pinned_rows.last().map(|r| r.1).unwrap_or(0.0);
    let band_split_y = (band_y_min + band_y_max) * 0.5;
    for (name, (count, ws)) in &joint_hist {
        let mut ws = ws.clone();
        ws.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = ws[ws.len() / 2];
        println!(
            "  {:<22} {:4} pins  dominant-weight p50 {:.2}",
            name, count, p50
        );
    }
    println!(
        "  band rest-y span {:.3}..{:.3} (split at {:.3})",
        band_y_min, band_y_max, band_split_y
    );

    // Front half split (model forward = hips +Z at rest). Borrow through
    // a private Arc clone so `avatar` stays free for `set_pose` later.
    let asset_arc = std::sync::Arc::clone(&avatar.asset);
    let humanoid = &asset_arc.humanoid.as_ref().expect("humanoid map").bone_map;
    let skeleton = &asset_arc.skeleton;
    let rest_locals = avatar.pose.local_transforms.clone();
    let rest_fk = fk(skeleton, &rest_locals);
    let hips_rot = humanoid
        .get(&HumanoidBone::Hips)
        .map(|id| rest_fk[id.0 as usize].0)
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let forward = quat_rotate_vec3(&hips_rot, &[0.0, 0.0, 1.0]);
    println!("  model forward (hips basis +Z): [{:.2} {:.2} {:.2}]", forward[0], forward[1], forward[2]);
    let fwd_of = |p: &[f32; 3]| p[0] * forward[0] + p[1] * forward[1] + p[2] * forward[2];

    // ---- pose sweep: M1 pin binding skew --------------------------------
    println!("\n=== M1 pin binding skew (pin target vs own-LBS skinned position) ===");
    println!(
        "  pin target = T(node)·offset from slot.sim.pin_targets; LBS = the vertex's own");
    println!("  4-weight blend under the posed skinning matrices (the render surface).");

    let skew_for_pose = |avatar: &vulvatar_lib::avatar::AvatarInstance| -> PoseSkew {
        compute_skew(avatar, &vd, &sim_pin_snapshot(&avatar.cloth_overlays[slot_idx].sim), band_split_y, fwd_of)
    };

    // Sit anchor: thighs + knees; signs auto-picked so knees move along
    // the model forward (desk envelope = seated).
    let node_of = |b: HumanoidBone| humanoid.get(&b).map(|v| v.0 as usize);
    let sit_ops = {
        let probe = |sign: f32| {
            let mut locals = rest_locals.clone();
            apply_world_delta(
                skeleton,
                humanoid,
                &mut locals,
                HumanoidBone::LeftUpperLeg,
                &axis_angle(&[1.0, 0.0, 0.0], sign * 70.0),
            );
            apply_world_delta(
                skeleton,
                humanoid,
                &mut locals,
                HumanoidBone::RightUpperLeg,
                &axis_angle(&[1.0, 0.0, 0.0], sign * 70.0),
            );
            let world = fk(skeleton, &locals);
            node_of(HumanoidBone::LeftLowerLeg)
                .and_then(|i| world.get(i))
                .map(|w| w.1)
                .unwrap_or([0.0; 3])
        };
        let thigh_i = node_of(HumanoidBone::LeftUpperLeg).unwrap_or(0);
        let hip_pos = rest_fk.get(thigh_i).map(|w| w.1).unwrap_or([0.0; 3]);
        let knee_a = probe(1.0);
        let knee_b = probe(-1.0);
        let dot_a = dot3(&vec3_sub(&knee_a, &hip_pos), &forward);
        let dot_b = dot3(&vec3_sub(&knee_b, &hip_pos), &forward);
        let sign = if dot_a > dot_b { 1.0 } else { -1.0 };
        println!("  sit: thigh sign {:+} (knee fwd dot {:+.3} vs {:+.3})", sign, dot_a, dot_b);
        vec![
            (HumanoidBone::LeftUpperLeg, [1.0, 0.0, 0.0], sign * 70.0),
            (HumanoidBone::RightUpperLeg, [1.0, 0.0, 0.0], sign * 70.0),
            (HumanoidBone::LeftLowerLeg, [1.0, 0.0, 0.0], -sign * 80.0),
            (HumanoidBone::RightLowerLeg, [1.0, 0.0, 0.0], -sign * 80.0),
        ]
    };
    // Lean back: rotate Spine (fallback Chest/Spine in apply_world_delta)
    // so the chest moves against model forward.
    let lean_ops = {
        let probe = |sign: f32| {
            let mut locals = rest_locals.clone();
            apply_world_delta(
                skeleton,
                humanoid,
                &mut locals,
                HumanoidBone::Spine,
                &axis_angle(&[1.0, 0.0, 0.0], sign * 15.0),
            );
            let world = fk(skeleton, &locals);
            node_of(HumanoidBone::Head)
                .and_then(|i| world.get(i))
                .map(|w| w.1)
                .unwrap_or([0.0; 3])
        };
        let head = node_of(HumanoidBone::Head)
            .and_then(|i| rest_fk.get(i))
            .map(|w| w.1)
            .unwrap_or([0.0; 3]);
        let d_a = dot3(&vec3_sub(&probe(1.0), &head), &forward);
        let d_b = dot3(&vec3_sub(&probe(-1.0), &head), &forward);
        // Lean BACK = head moves AGAINST forward.
        let sign = if d_a < d_b { 1.0 } else { -1.0 };
        println!("  lean back: spine sign {:+} (head fwd dot {:+.3} vs {:+.3})", sign, d_a, d_b);
        vec![(HumanoidBone::Spine, [1.0, 0.0, 0.0], sign * 15.0)]
    };

    let mut sweep = Vec::new();
    for (name, ops) in [
        ("rest", vec![]),
        ("hips_yaw15", vec![(HumanoidBone::Hips, [0.0, 1.0, 0.0], 15.0)]),
        ("hips_yaw-15", vec![(HumanoidBone::Hips, [0.0, 1.0, 0.0], -15.0)]),
        ("lean_back15", lean_ops.clone()),
        ("sit", sit_ops.clone()),
        ("sit_lean_back", {
            let mut v = lean_ops.clone();
            v.extend(sit_ops.clone());
            v
        }),
    ] {
        set_pose(&mut avatar, &rest_locals, &ops);
        let skew = skew_for_pose(&avatar);
        println!(
            "  {:<14} skew max {:6.1} mm  p95 {:6.1} mm  mean {:5.1} mm | upper-half max {:6.1} mm  front-half max {:6.1} mm",
            name,
            skew.max * 1e3,
            skew.p95 * 1e3,
            skew.mean * 1e3,
            skew.upper_max * 1e3,
            skew.front_max * 1e3
        );
        sweep.push((name.to_string(), skew));
    }

    // ---- M2: CPU sim, rest → lean back → sit → yaw oscillation ----------
    println!("\n=== M2 sim: free-particle penetration into body colliders ===");
    // Rest snapshot for the displacement sanity metric (is the sim
    // actually moving the free particles?).
    let rest_positions: Vec<Vec3> = avatar.cloth_overlays[slot_idx]
        .sim
        .particles
        .iter()
        .map(|p| p.position)
        .collect();

    // Per-collider coverage at rest: how close does the free hem get?
    {
        let sim = &avatar.cloth_overlays[slot_idx].sim;
        println!("  collider coverage at rest (free particles):");
        let cols = resolve_named(&avatar);
        for (a, b, r, node) in &cols {
            let mut min_d = f32::MAX;
            let mut inside = 0usize;
            for p in &sim.particles {
                if p.pinned {
                    continue;
                }
                let d = point_segment_dist(&p.position, a, b);
                min_d = min_d.min(d);
                if d < r + sim.collision_margin {
                    inside += 1;
                }
            }
            println!(
                "    {:<22} r={:.3}  min hem dist {:7.1} mm  within r+margin: {inside}",
                node,
                r,
                min_d * 1e3
            );
        }
    }

    let frame_dt = 1.0 / 30.0;
    let substeps = 4u32;
    let step_dt = frame_dt / substeps as f32;
    let total_frames = 300usize;

    // Pose anchors: 0 = rest, 1 = lean, 2 = sit; locals lerped between.
    let anchor_locals = {
        let mut v = vec![rest_locals.clone()];
        let mut l = rest_locals.clone();
        apply_ops(skeleton, humanoid, &mut l, &lean_ops);
        v.push(l);
        let mut l2 = v[1].clone();
        apply_ops(skeleton, humanoid, &mut l2, &sit_ops);
        v.push(l2);
        v
    };
    let anchor_of = |f: usize| -> (usize, usize, f32) {
        // ramp 30..60 rest→lean, hold, ramp 90..120 lean→sit, hold.
        if f < 30 {
            (0, 0, 0.0)
        } else if f < 60 {
            (0, 1, (f - 30) as f32 / 30.0)
        } else if f < 90 {
            (1, 1, 0.0)
        } else if f < 120 {
            (1, 2, (f - 90) as f32 / 30.0)
        } else {
            (2, 2, 0.0)
        }
    };

    let mut csv = String::from(
        "frame,phase,skew_max_mm,skew_p95_mm,pen_max_mm,pen_p95_mm,pen_count,disp_max_mm,pen_node,poke_max_mm,poke_count\n",
    );
    let mut pen_overall_max = 0.0f32;
    let mut pen_sit_max = 0.0f32;
    let mut pin_node_names: std::collections::HashSet<String> = std::collections::HashSet::new();

    for f in 0..total_frames {
        // Frames 210-300: hips yaw oscillation ±15° at 1 Hz on top of the
        // sit anchor — transient inertia vs the hard-snapped band.
        let extra_yaw = if f >= 210 {
            15.0 * (2.0 * std::f32::consts::PI * ((f - 210) as f32) / 30.0).sin()
        } else {
            0.0
        };
        let (a, b, t) = anchor_of(f.min(210));
        let mut locals = lerp_locals(&anchor_locals[a], &anchor_locals[b], t);
        if extra_yaw != 0.0 {
            apply_world_delta(
                skeleton,
                humanoid,
                &mut locals,
                HumanoidBone::Hips,
                &axis_angle(&[0.0, 1.0, 0.0], extra_yaw),
            );
        }
        avatar.pose.local_transforms = locals;
        avatar.compute_global_pose();
        avatar.build_skinning_matrices();

        for _ in 0..substeps {
            vulvatar_lib::simulation::cloth_solver::step_cloth(step_dt, &mut avatar, &[], None);
        }

        let sim = &avatar.cloth_overlays[slot_idx].sim;
        let disp_max = sim
            .particles
            .iter()
            .zip(&rest_positions)
            .filter(|(p, _)| !p.pinned)
            .map(|(p, r0)| {
                ((p.position[0] - r0[0]).powi(2)
                    + (p.position[1] - r0[1]).powi(2)
                    + (p.position[2] - r0[2]).powi(2))
                    .sqrt()
            })
            .fold(0.0f32, f32::max);
        let skew = compute_skew(&avatar, &vd, &sim_pin_snapshot(sim), band_split_y, fwd_of);
        let pens = penetration(&avatar, sim);

        // M3: how far the BODY surface pokes outside the skirt ring at
        // band heights (the rendered dent = body outside skirt). Two
        // variants: vs the pinned band alone (what collision can never
        // correct — pins hard-snap after collision) and vs ALL skirt
        // particles (the full silhouette). Sampled every 15 frames —
        // the LBS over the full body prim is the cost.
        let cone = if f % 15 == 0 {
            let sim2 = &avatar.cloth_overlays[slot_idx].sim;
            let all: Vec<[f32; 3]> = sim2.particles.iter().map(|p| p.position).collect();
            let (max, count, sectors) = cone_poke(&avatar, &body_vd, &all);
            Some((max, count, sectors))
        } else {
            None
        };
        let phase = if f < 30 {
            "rest"
        } else if f < 60 {
            "lean_ramp"
        } else if f < 90 {
            "lean_hold"
        } else if f < 120 {
            "sit_ramp"
        } else if f < 210 {
            "sit_hold"
        } else {
            "yaw_osc"
        };
        pen_overall_max = pen_overall_max.max(pens.max);
        if (120..210).contains(&f) {
            pen_sit_max = pen_sit_max.max(pens.max);
            for n in &pens.nodes {
                pin_node_names.insert(n.clone());
            }
        }
        csv.push_str(&format!(
            "{f},{phase},{:.2},{:.2},{:.2},{:.2},{},{:.2},\"{}\",{:.2},{}",
            skew.max * 1e3,
            skew.p95 * 1e3,
            pens.max * 1e3,
            pens.p95 * 1e3,
            pens.count,
            disp_max * 1e3,
            pens.nodes.iter().cloned().collect::<Vec<_>>().join("|"),
            cone.as_ref().map(|c| c.0).unwrap_or(0.0) * 1e3,
            cone.as_ref().map(|c| c.1).unwrap_or(0),
        ));
        csv.push('\n');

        if f % 30 == 0 {
            println!(
                "  f{:3} {:<9} disp max {:6.1} mm | pen max {:6.1} mm n {:3} | poke max {:6.1} mm n {:3} [{}] {}",
                f,
                phase,
                disp_max * 1e3,
                pens.max * 1e3,
                pens.count,
                cone.as_ref().map(|c| c.0).unwrap_or(0.0) * 1e3,
                cone.as_ref().map(|c| c.1).unwrap_or(0),
                cone.as_ref().map(|c| c.2.clone()).unwrap_or_default(),
                pens.nodes.iter().cloned().collect::<Vec<_>>().join(",")
            );
        }
    }
    println!(
        "  overall pen max {:.1} mm | sit-hold pen max {:.1} mm | penetrating colliders: {:?}",
        pen_overall_max * 1e3,
        pen_sit_max * 1e3,
        pin_node_names
    );

    // ---- write outputs ---------------------------------------------------
    let csv_path = out_dir.join("metrics.csv");
    std::fs::write(&csv_path, &csv).map_err(|e| format!("write csv: {e}"))?;

    let mut summary = String::new();
    summary.push_str("# skirt_fit audit\n\n");
    summary.push_str(&format!("asset: `{input_path}`\n\n"));
    summary.push_str("## collider inventory\n\n");
    for c in &avatar.asset.colliders {
        let node = avatar
            .asset
            .skeleton
            .nodes
            .get(c.node.0 as usize)
            .map(|n| n.name.as_str())
            .unwrap_or("?");
        let shape = match c.shape {
            ColliderShape::Sphere { radius } => format!("sphere r={radius:.3}"),
            ColliderShape::Capsule { radius, height } => {
                format!("capsule r={radius:.3} h={height:.3}")
            }
        };
        summary.push_str(&format!("- `{node}`: {shape}\n"));
    }
    summary.push_str("\n## pin-band dominant joints\n\n");
    for (name, (count, ws)) in &joint_hist {
        summary.push_str(&format!("- `{name}`: {count} pins\n"));
        let _ = ws;
    }
    summary.push_str("\n## M1 pin binding skew (static pose sweep)\n\n");
    summary.push_str("| pose | max mm | p95 mm | mean mm | upper-half max mm | front-half max mm |\n|---|---|---|---|---|---|\n");
    for (name, s) in &sweep {
        summary.push_str(&format!(
            "| {name} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} |\n",
            s.max * 1e3,
            s.p95 * 1e3,
            s.mean * 1e3,
            s.upper_max * 1e3,
            s.front_max * 1e3
        ));
    }
    summary.push_str(&format!(
        "\n## M2 sim penetration\n\n- overall max: {:.1} mm\n- sit-hold max: {:.1} mm\n- colliders penetrated: {pin_node_names:?}\n\nmetrics: `metrics.csv`\n",
        pen_overall_max * 1e3,
        pen_sit_max * 1e3
    ));
    let md_path = out_dir.join("summary.md");
    std::fs::write(&md_path, summary).map_err(|e| format!("write summary: {e}"))?;
    println!(
        "\nwrote {} and {}",
        csv_path.display(),
        md_path.display()
    );
    Ok(())
}

// =========================================================================
// pose plumbing (validate_gt's apply_world_delta + fk, trimmed)
// =========================================================================

fn fk(skeleton: &SkeletonAsset, locals: &[Transform]) -> Vec<(Quat, Vec3)> {
    let mut world: Vec<(Quat, Vec3)> = vec![([0.0, 0.0, 0.0, 1.0], [0.0; 3]); skeleton.nodes.len()];
    let mut stack: Vec<vulvatar_lib::asset::NodeId> = skeleton.root_nodes.clone();
    while let Some(idx) = stack.pop() {
        let i = idx.0 as usize;
        if i >= skeleton.nodes.len() || i >= locals.len() {
            continue;
        }
        let (p_rot, p_pos) = skeleton.nodes[i]
            .parent
            .map(|p| world[p.0 as usize])
            .unwrap_or(([0.0, 0.0, 0.0, 1.0], [0.0; 3]));
        let rot = quat_normalize(&quat_mul(&p_rot, &locals[i].rotation));
        let pos = vec3_add(&p_pos, &quat_rotate_vec3(&p_rot, &locals[i].translation));
        world[i] = (rot, pos);
        stack.extend(skeleton.nodes[i].children.iter().copied());
    }
    world
}

/// Pre-multiply `bone`'s world rotation by `delta_world` as a local-rotation
/// update (validate_gt recipe, incl. the Yumeka chest fallback).
fn apply_world_delta(
    skeleton: &SkeletonAsset,
    humanoid: &std::collections::HashMap<HumanoidBone, vulvatar_lib::asset::NodeId>,
    locals: &mut [Transform],
    bone: HumanoidBone,
    delta_world: &Quat,
) {
    let bone = match humanoid.get(&bone) {
        Some(_) => bone,
        None => {
            let candidates: &[HumanoidBone] = match bone {
                HumanoidBone::UpperChest => &[HumanoidBone::Chest, HumanoidBone::Spine],
                HumanoidBone::Chest => &[HumanoidBone::Spine],
                _ => &[],
            };
            match candidates.iter().copied().find(|b| humanoid.contains_key(b)) {
                Some(b) => b,
                None => return,
            }
        }
    };
    let Some(vulvatar_lib::asset::NodeId(idx)) = humanoid.get(&bone).copied() else {
        return;
    };
    let i = idx as usize;
    let world = fk(skeleton, locals);
    let p_rot = skeleton
        .nodes[i]
        .parent
        .map(|p| world[p.0 as usize].0)
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let local_delta = quat_mul(&quat_mul(&quat_conjugate(&p_rot), delta_world), &p_rot);
    locals[i].rotation = quat_normalize(&quat_mul(&local_delta, &locals[i].rotation));
}

fn apply_ops(
    skeleton: &SkeletonAsset,
    humanoid: &std::collections::HashMap<HumanoidBone, vulvatar_lib::asset::NodeId>,
    locals: &mut [Transform],
    ops: &[(HumanoidBone, [f32; 3], f32)],
) {
    for (bone, axis, deg) in ops {
        apply_world_delta(skeleton, humanoid, locals, *bone, &axis_angle(axis, *deg));
    }
}

fn set_pose(
    avatar: &mut vulvatar_lib::avatar::AvatarInstance,
    rest_locals: &[Transform],
    ops: &[(HumanoidBone, [f32; 3], f32)],
) {
    let mut locals = rest_locals.to_vec();
    let skeleton = avatar.asset.skeleton.clone();
    let humanoid = avatar.asset.humanoid.as_ref().expect("humanoid map").bone_map.clone();
    apply_ops(&skeleton, &humanoid, &mut locals, ops);
    avatar.pose.local_transforms = locals;
    avatar.compute_global_pose();
    avatar.build_skinning_matrices();
}

fn axis_angle(axis: &[f32; 3], deg: f32) -> Quat {
    let rad = deg.to_radians() * 0.5;
    let s = rad.sin();
    [axis[0] * s, axis[1] * s, axis[2] * s, rad.cos()]
}

fn lerp_locals(a: &[Transform], b: &[Transform], t: f32) -> Vec<Transform> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let q = [
                x.rotation[0] + (y.rotation[0] - x.rotation[0]) * t,
                x.rotation[1] + (y.rotation[1] - x.rotation[1]) * t,
                x.rotation[2] + (y.rotation[2] - x.rotation[2]) * t,
                x.rotation[3] + (y.rotation[3] - x.rotation[3]) * t,
            ];
            Transform {
                translation: x.translation,
                rotation: quat_normalize(&q),
                scale: x.scale,
            }
        })
        .collect()
}

fn dot3(a: &[f32], b: &[f32]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// M3: body-vs-skirt poke. Measures how far the posed BODY mesh sticks
/// outside the skirt silhouette at band heights: bucket skirt particles
/// into 20 mm-height bins and record, per bin, the largest horizontal
/// radius (from the vertical axis through the hips joint); a body
/// vertex in that y-range whose radius exceeds the bin's radius by more
/// than 5 mm pokes out of the skirt. Pokes are also bucketed into eight
/// 45° sectors (0 = model forward) so the report can say WHERE the body
/// escapes the skirt. Returns (max poke depth, poke count, sector text).
fn cone_poke(
    avatar: &vulvatar_lib::avatar::AvatarInstance,
    body_vd: &vulvatar_lib::asset::VertexData,
    skirt: &[[f32; 3]],
) -> (f32, usize, String) {
    if skirt.is_empty() {
        return (0.0, 0, String::new());
    }
    let skinning = &avatar.pose.skinning_matrices;
    // Vertical axis through the hips joint (posed).
    let humanoid = avatar.asset.humanoid.as_ref().expect("humanoid map");
    let hips_mat = humanoid
        .bone_map
        .get(&HumanoidBone::Hips)
        .and_then(|id| avatar.pose.global_transforms.get(id.0 as usize));
    let axis = hips_mat
        .map(|m| [m[3][0], m[3][1], m[3][2]])
        .unwrap_or([0.0; 3]);
    // Model forward (hips basis +Z, third column).
    let forward = hips_mat
        .map(|m| [m[0][2], m[1][2], m[2][2]])
        .unwrap_or([0.0, 0.0, 1.0]);

    let skirt_y_min = skirt.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
    let band_y_min = skirt_y_min - 0.02;
    let band_y_max = skirt.iter().map(|p| p[1]).fold(f32::MIN, f32::max) + 0.02;
    const BIN: f32 = 0.02;
    let mut bins: std::collections::HashMap<i64, f32> = std::collections::HashMap::new();
    for p in skirt {
        let r = ((p[0] - axis[0]).powi(2) + (p[2] - axis[2]).powi(2)).sqrt();
        let b = ((p[1] - band_y_min) / BIN).floor() as i64;
        let e = bins.entry(b).or_default();
        if r > *e {
            *e = r;
        }
    }

    let mut max_poke = 0.0f32;
    let mut count = 0usize;
    let mut sectors = [0.0f32; 8];
    for (vi, &pos) in body_vd.positions.iter().enumerate() {
        // LBS (posed).
        let mut w = [0.0f32; 3];
        let mut tw = 0.0f32;
        if let Some(w4) = body_vd.joint_weights.get(vi) {
            for k in 0..4 {
                let wk = w4[k];
                if wk > 0.0001 {
                    let j = body_vd.joint_indices[vi][k] as usize;
                    if let Some(sm) = skinning.get(j) {
                        for c in 0..3 {
                            w[c] += wk
                                * (sm[0][c] * pos[0]
                                    + sm[1][c] * pos[1]
                                    + sm[2][c] * pos[2]
                                    + sm[3][c]);
                        }
                        tw += wk;
                    }
                }
            }
        }
        if tw <= 0.001 {
            continue;
        }
        for c in 0..3 {
            w[c] /= tw;
        }
        if w[1] < band_y_min || w[1] > band_y_max {
            continue;
        }
        let dx = w[0] - axis[0];
        let dz = w[2] - axis[2];
        let r = (dx * dx + dz * dz).sqrt();
        let b = ((w[1] - band_y_min) / BIN).floor() as i64;
        let Some(&skirt_r) = bins.get(&b) else {
            continue;
        };
        let poke = r - skirt_r - 0.005;
        if poke > 0.0 {
            max_poke = max_poke.max(poke);
            count += 1;
            // Sector 0 = forward, growing clockwise when viewed from
            // above (+Y): angle = atan2(right, forward).
            let right = [-forward[2], 0.0, forward[0]];
            let ang = (dx * right[0] + dz * right[2])
                .atan2(dx * forward[0] + dz * forward[2]);
            let sector = ((ang.to_degrees() + 360.0 + 22.5).rem_euclid(360.0) / 45.0) as usize % 8;
            if poke > sectors[sector] {
                sectors[sector] = poke;
            }
        }
    }
    let names = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"];
    let mut text = String::new();
    for (i, s) in sectors.iter().enumerate() {
        if *s > 0.0 {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(&format!("{}:{:.0}mm", names[i], s * 1e3));
        }
    }
    (max_poke, count, text)
}

// =========================================================================
// M1: pin binding skew
// =========================================================================

struct PoseSkew {
    max: f32,
    p95: f32,
    mean: f32,
    upper_max: f32,
    front_max: f32,
}

/// Copy the pin (particle → node, offset) pairs out so the closure can
/// borrow the avatar immutably.
fn sim_pin_snapshot(
    sim: &vulvatar_lib::simulation::cloth::ClothSimState,
) -> Vec<(usize, usize, Vec3)> {
    let mut out = Vec::new();
    for (vi, p) in sim.particles.iter().enumerate() {
        if let (true, Some(pt)) = (p.pinned, p.pin_target_index) {
            let t = &sim.pin_targets[pt];
            out.push((vi, t.node_index, t.offset));
        }
    }
    out
}

fn compute_skew(
    avatar: &vulvatar_lib::avatar::AvatarInstance,
    vd: &vulvatar_lib::asset::VertexData,
    pins: &[(usize, usize, Vec3)],
    band_split_y: f32,
    fwd_of: impl Fn(&[f32; 3]) -> f32,
) -> PoseSkew {
    let transforms = &avatar.pose.global_transforms;
    let skinning = &avatar.pose.skinning_matrices;
    let mut skews: Vec<(f32, [f32; 3])> = Vec::with_capacity(pins.len());
    for &(vi, node, offset) in pins {
        let Some(m) = transforms.get(node) else {
            continue;
        };
        let w = [
            m[0][0] * offset[0] + m[1][0] * offset[1] + m[2][0] * offset[2] + m[3][0],
            m[0][1] * offset[0] + m[1][1] * offset[1] + m[2][1] * offset[2] + m[3][1],
            m[0][2] * offset[0] + m[1][2] * offset[1] + m[2][2] * offset[2] + m[3][2],
        ];
        // Own-weight LBS of the vertex's rest local position.
        let rest = vd.positions[vi];
        let mut lbs = [0.0f32; 3];
        let mut tw = 0.0f32;
        if let Some(w4) = vd.joint_weights.get(vi) {
            for k in 0..4 {
                let wk = w4[k];
                if wk > 0.0001 {
                    let j = vd.joint_indices[vi][k] as usize;
                    if let Some(sm) = skinning.get(j) {
                        for c in 0..3 {
                            lbs[c] += wk
                                * (sm[0][c] * rest[0]
                                    + sm[1][c] * rest[1]
                                    + sm[2][c] * rest[2]
                                    + sm[3][c]);
                        }
                        tw += wk;
                    }
                }
            }
        }
        if tw > 0.001 {
            for c in 0..3 {
                lbs[c] /= tw;
            }
        } else {
            lbs = rest;
        }
        let d = ((w[0] - lbs[0]).powi(2) + (w[1] - lbs[1]).powi(2) + (w[2] - lbs[2]).powi(2)).sqrt();
        skews.push((d, lbs));
    }
    let mut vals: Vec<f32> = skews.iter().map(|s| s.0).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let max = vals.last().copied().unwrap_or(0.0);
    let p95 = vals
        .get(((vals.len() as f32) * 0.95) as usize)
        .copied()
        .unwrap_or(max);
    let mean = if vals.is_empty() {
        0.0
    } else {
        vals.iter().sum::<f32>() / vals.len() as f32
    };
    let upper_max = skews
        .iter()
        .filter(|(_, p)| p[1] >= band_split_y)
        .map(|(d, _)| *d)
        .fold(0.0f32, f32::max);
    let front_max = skews
        .iter()
        .filter(|(_, p)| fwd_of(p) > 0.0)
        .map(|(d, _)| *d)
        .fold(0.0f32, f32::max);
    PoseSkew {
        max,
        p95,
        mean,
        upper_max,
        front_max,
    }
}

// =========================================================================
// M2: free-particle penetration vs resolved colliders (with node names)
// =========================================================================

struct Pens {
    max: f32,
    p95: f32,
    count: usize,
    nodes: std::collections::BTreeSet<String>,
}

/// Mirror cloth::resolve_colliders but keep the node name. Returns
/// `(seg_a, seg_b, radius, node_name)` capsules (spheres degenerate).
fn resolve_named(avatar: &vulvatar_lib::avatar::AvatarInstance) -> Vec<(Vec3, Vec3, f32, String)> {
    let mut colliders: Vec<(Vec3, Vec3, f32, String)> = Vec::new();
    for (i, c) in avatar.asset.colliders.iter().enumerate() {
        if !avatar.collider_enabled.get(i).copied().unwrap_or(true) {
            continue;
        }
        let Some(mat) = avatar.pose.global_transforms.get(c.node.0 as usize) else {
            continue;
        };
        let [ox, oy, oz] = c.offset;
        let center = [
            mat[0][0] * ox + mat[1][0] * oy + mat[2][0] * oz + mat[3][0],
            mat[0][1] * ox + mat[1][1] * oy + mat[2][1] * oz + mat[3][1],
            mat[0][2] * ox + mat[1][2] * oy + mat[2][2] * oz + mat[3][2],
        ];
        let node = avatar
            .asset
            .skeleton
            .nodes
            .get(c.node.0 as usize)
            .map(|n| n.name.clone())
            .unwrap_or_default();
        match c.shape {
            ColliderShape::Sphere { radius } => colliders.push((center, center, radius, node)),
            ColliderShape::Capsule { radius, height } => {
                let up = [mat[1][0], mat[1][1], mat[1][2]];
                let l = (up[0] * up[0] + up[1] * up[1] + up[2] * up[2]).sqrt();
                let axis = if l > 1e-8 {
                    [up[0] / l, up[1] / l, up[2] / l]
                } else {
                    [0.0, 1.0, 0.0]
                };
                let hh = height * 0.5;
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
                colliders.push((a, b, radius, node));
            }
        }
    }
    colliders
}

fn point_segment_dist(p: &Vec3, a: &Vec3, b: &Vec3) -> f32 {
    let ab = vec3_sub(b, a);
    let ap = vec3_sub(p, a);
    let ab2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    let t = if ab2 > 1e-12 {
        (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / ab2
    } else {
        0.0
    }
    .clamp(0.0, 1.0);
    let cx = a[0] + ab[0] * t;
    let cy = a[1] + ab[1] * t;
    let cz = a[2] + ab[2] * t;
    ((p[0] - cx).powi(2) + (p[1] - cy).powi(2) + (p[2] - cz).powi(2)).sqrt()
}

fn penetration(
    avatar: &vulvatar_lib::avatar::AvatarInstance,
    sim: &vulvatar_lib::simulation::cloth::ClothSimState,
) -> Pens {
    let colliders = resolve_named(avatar);
    let mut depths: Vec<f32> = Vec::new();
    let mut nodes = std::collections::BTreeSet::new();
    for p in &sim.particles {
        for (a, b, r, node) in &colliders {
            let depth = r - point_segment_dist(&p.position, a, b);
            if depth > 0.0 {
                depths.push(depth);
                nodes.insert(node.clone());
            }
        }
    }
    depths.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let max = depths.last().copied().unwrap_or(0.0);
    let p95 = depths
        .get(((depths.len() as f32) * 0.95) as usize)
        .copied()
        .unwrap_or(0.0);
    Pens {
        max,
        p95,
        count: depths.len(),
        nodes,
    }
}
