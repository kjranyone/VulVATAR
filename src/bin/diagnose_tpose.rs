//! T-pose render repro: load an avatar exactly like the live app
//! (FBX via the shared asset cache), build the base pose, and render
//! with NO cloth overlay, NO expressions, NO springs — the avatar's
//! load-state look. Saves a full-body PNG plus a CPU-side skinning
//! sanity report for the skirt mesh (Circle.056) to catch asset-level
//! breakage independent of the renderer.
//!
//! Usage:
//!   cargo run --bin diagnose_tpose -- <avatar.fbx|avatar.vrm> [out_dir]

use std::path::PathBuf;
use std::sync::Arc;

use image::ImageBuffer;
use vulvatar_lib::app::ViewportCamera;
use vulvatar_lib::asset::fbx::FbxAssetLoader;
use vulvatar_lib::asset::vrm::VrmAssetLoader;
use vulvatar_lib::asset::AvatarAsset;
use vulvatar_lib::avatar::{AvatarInstance, AvatarInstanceId};
use vulvatar_lib::renderer::frame_input::{
    CameraState, LightingState, MsaaMode, OutputTargetRequest, RenderAlphaMode,
    RenderAvatarInstance, RenderColorSpace, RenderCullMode, RenderExportMode, RenderFrameInput,
    RenderMeshInstance,
};
use vulvatar_lib::renderer::material::{MaterialShaderMode, MaterialUploadRequest};
use vulvatar_lib::renderer::VulkanRenderer;

fn main() -> Result<(), String> {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let input_path = args
        .next()
        .unwrap_or_else(|| "sample_data/YUMEKA_v1.0.3/FBX/Yumeka_v1.0.3.fbx".to_string());
    let output_dir = args.next().unwrap_or_else(|| "diagnostics/tpose_repro".to_string());
    if output_dir.contains("validation_images") {
        return Err("diagnostics outputs must not target validation_images/".into());
    }
    let out = PathBuf::from(&output_dir);
    std::fs::create_dir_all(&out).map_err(|e| format!("mkdir {}: {e}", out.display()))?;

    let is_fbx = PathBuf::from(&input_path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("fbx"))
        .unwrap_or(false);
    println!("Loading avatar: {} (cache-enabled loader, same as app)", input_path);
    let asset: Arc<AvatarAsset> = if is_fbx {
        FbxAssetLoader::new()
            .load(&input_path)
            .map_err(|e| format!("FBX load failed: {e}"))?
    } else {
        VrmAssetLoader::new()
            .load(&input_path)
            .map_err(|e| format!("VRM load failed: {e}"))?
    };
    println!(
        "loaded: {} nodes, {} meshes",
        asset.skeleton.nodes.len(),
        asset.meshes.len(),
    );

    let mut avatar = AvatarInstance::new(AvatarInstanceId(1), Arc::clone(&asset));
    avatar.build_base_pose();
    avatar.compute_global_pose();
    avatar.build_skinning_matrices();

    // CPU-side skinning sanity for the skirt mesh: at the bind pose every
    // skinning matrix must be ~identity, so skinned world positions must
    // stay inside the raw bounds. Report the bbox and any flyers.
    if let Some(skirt) = asset
        .meshes
        .iter()
        .find(|m| m.name.eq_ignore_ascii_case("circle.056"))
        .and_then(|m| m.primitives.iter().find(|p| p.vertices.is_some()))
    {
        let vd = skirt.vertices.as_ref().unwrap();
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        let mut flyers = 0usize;
        let mut max_disp: f32 = 0.0;
        for (i, &pos) in vd.positions.iter().enumerate() {
            let mut world = [0.0f32; 3];
            let mut total_w = 0.0;
            if i < vd.joint_weights.len() && i < vd.joint_indices.len() {
                for k in 0..4 {
                    let w = vd.joint_weights[i][k];
                    if w > 0.0001 {
                        let j = vd.joint_indices[i][k] as usize;
                        if j < avatar.pose.skinning_matrices.len() {
                            let sm = &avatar.pose.skinning_matrices[j];
                            world[0] += w
                                * (sm[0][0] * pos[0] + sm[1][0] * pos[1] + sm[2][0] * pos[2]
                                    + sm[3][0]);
                            world[1] += w
                                * (sm[0][1] * pos[0] + sm[1][1] * pos[1] + sm[2][1] * pos[2]
                                    + sm[3][1]);
                            world[2] += w
                                * (sm[0][2] * pos[0] + sm[1][2] * pos[1] + sm[2][2] * pos[2]
                                    + sm[3][2]);
                            total_w += w;
                        }
                    }
                }
            }
            if total_w > 0.001 {
                for c in 0..3 {
                    world[c] /= total_w;
                }
            } else {
                world = pos;
            }
            let disp = (world[0] - pos[0]).powi(2)
                + (world[1] - pos[1]).powi(2)
                + (world[2] - pos[2]).powi(2);
            max_disp = max_disp.max(disp.sqrt());
            if disp.sqrt() > 0.02 {
                flyers += 1;
            }
            for c in 0..3 {
                lo[c] = lo[c].min(world[c]);
                hi[c] = hi[c].max(world[c]);
            }
        }
        println!(
            "Circle.056 CPU-skinned bbox: [{:.3},{:.3},{:.3}] .. [{:.3},{:.3},{:.3}]  verts={} max_disp={:.4}m",
            lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], vd.positions.len(), max_disp
        );

        // Per-joint weight census with bone names — arm/hand influence on
        // the skirt is the smoking gun for cross-mesh weight bleed that
        // only shows once the pose leaves bind.
        let mut wsum: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
        let mut dom: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for i in 0..vd.positions.len() {
            let mut best = (0usize, 0.0f32);
            for k in 0..4 {
                let w = vd.joint_weights[i][k];
                if w > 0.0001 {
                    let j = vd.joint_indices[i][k] as usize;
                    *wsum.entry(j).or_insert(0.0) += w;
                    if w > best.1 {
                        best = (j, w);
                    }
                }
            }
            if best.1 > 0.0 {
                *dom.entry(best.0).or_insert(0) += 1;
            }
        }
        let name = |j: usize| {
            asset
                .skeleton
                .nodes
                .get(j)
                .map(|n| n.name.as_str())
                .unwrap_or("<out-of-range>")
        };
        let mut by_w: Vec<_> = wsum.iter().collect();
        by_w.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        println!("Circle.056 weight census (joint: total_weight, dominant_vertex_count):");
        for (j, w) in by_w.iter().take(20) {
            println!("  {:>4} {:<28} w={:8.2}  dom={}", j, name(**j), w, dom.get(*j).copied().unwrap_or(0));
        }
        let arm_joints: Vec<_> = by_w
            .iter()
            .filter(|(j, _)| {
                let n = name(**j).to_lowercase();
                n.contains("arm") || n.contains("hand") || n.contains("wrist") || n.contains("elbow")
            })
            .map(|(j, w)| format!("{}(w={:.1})", name(**j), w))
            .collect();
        let arm_report = if arm_joints.is_empty() {
            "NONE".to_string()
        } else {
            arm_joints.join(", ")
        };
        println!("ARM/HAND influence on skirt: {}", arm_report);
    } else {
        println!("(no Circle.056 mesh found — skipping CPU skinning check)");
    }

    // ---- ALL-mesh joint census: flag any garment mesh whose weights
    // span unrelated body regions (e.g. arm bones in a skirt/hem mesh).
    println!("--- spring bones ({}), colliders ({}) ---", asset.spring_bones.len(), asset.colliders.len());
    for sb in &asset.spring_bones {
        let names: Vec<String> = sb
            .joints
            .iter()
            .filter_map(|n| asset.skeleton.nodes.get(n.0 as usize).map(|x| x.name.clone()))
            .collect();
        println!("  spring ({} joints): {:?}", sb.joints.len(), names);
    }
    println!("--- clearance anchor graph ---");
    for mesh in &asset.meshes {
        for prim in &mesh.primitives {
            let body_id = prim.body_primitive_id.map(|p| p.0);
            let cont_id = prim.containment_primitive_id.map(|p| p.0);
            let (n_clear, n_cont) = (
                prim.skin_anchors.as_ref().map(|a| a.len()).unwrap_or(0),
                prim.containment_anchors.as_ref().map(|a| a.len()).unwrap_or(0),
            );
            if body_id.is_some() || cont_id.is_some() || n_clear > 0 || n_cont > 0 {
                let clear_stats = prim.skin_anchors.as_ref().map(|a| {
                    let mut sorted: Vec<f32> = a.iter().map(|x| x.min_clearance).collect();
                    sorted.sort_by(|p, q| p.partial_cmp(q).unwrap());
                    format!(
                        "n={} min={:.4} med={:.4} max={:.4}",
                        sorted.len(),
                        sorted.first().copied().unwrap_or(0.0),
                        sorted[sorted.len() / 2],
                        sorted.last().copied().unwrap_or(0.0)
                    )
                });
                println!(
                    "  {:<16} prim{:>3?} clearance_parent={:?} containment_parent={:?} skin=[{}] cont_n={}",
                    mesh.name,
                    prim.id.0,
                    body_id,
                    cont_id,
                    clear_stats.unwrap_or_default(),
                    n_cont,
                );
            }
        }
    }
    println!(
        "body_primitive_id of asset: {:?}",
        asset.body_primitive_id.map(|p| p.0)
    );

    // ---- jacket <-> skirt rest-penetration census ---------------------
    // For jacket verts inside the skirt's vertical band, signed
    // clearance along the nearest skirt vertex normal (negative =
    // inside the skirt volume). Same reversed. Decides which mesh
    // penetrates which at bind and by how much.
    {
        let find_prim = |name: &str| {
            asset
                .meshes
                .iter()
                .find(|m| m.name.eq_ignore_ascii_case(name))
                .and_then(|m| m.primitives.iter().find(|p| p.vertices.is_some()))
        };
        if let (Some(jacket), Some(skirt)) = (find_prim("circle.051"), find_prim("circle.056")) {
            avatar.build_base_pose();
            avatar.compute_global_pose();
            avatar.build_skinning_matrices();
            let sm = avatar.pose.skinning_matrices.clone();
            let skin_world = |prim: &vulvatar_lib::asset::MeshPrimitiveAsset| {
                let vd = prim.vertices.as_ref().unwrap();
                (0..vd.positions.len())
                    .map(|i| {
                        let pos = vd.positions[i];
                        let mut w = [0.0f32; 3];
                        let mut tw = 0.0f32;
                        for k in 0..4 {
                            let wt = vd.joint_weights[i][k];
                            if wt > 0.0001 {
                                let j = vd.joint_indices[i][k] as usize;
                                if let Some(m) = sm.get(j) {
                                    for c in 0..3 {
                                        w[c] += wt
                                            * (m[0][c] * pos[0] + m[1][c] * pos[1]
                                                + m[2][c] * pos[2] + m[3][c]);
                                    }
                                    tw += wt;
                                }
                            }
                        }
                        if tw > 0.001 {
                            for c in 0..3 {
                                w[c] /= tw;
                            }
                        }
                        let n = vd.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]);
                        (w, n)
                    })
                    .collect::<Vec<_>>()
            };
            let jacket_w = skin_world(jacket);
            let skirt_w = skin_world(skirt);
            let skirt_lo_y = skirt_w.iter().map(|(p, _)| p[1]).fold(f32::MAX, f32::min);
            let skirt_hi_y = skirt_w.iter().map(|(p, _)| p[1]).fold(f32::MIN, f32::max);
            let mut inside = 0usize;
            let mut close = 0usize;
            let mut outside = 0usize;
            let mut worst = 0.0f32;
            let mut worst_at = [0.0f32; 3];
            for (jp, _) in &jacket_w {
                if jp[1] < skirt_lo_y || jp[1] > skirt_hi_y {
                    continue;
                }
                let mut best = (f32::INFINITY, 0usize);
                for (si, (sp, _)) in skirt_w.iter().enumerate() {
                    let d2 = (jp[0] - sp[0]).powi(2)
                        + (jp[1] - sp[1]).powi(2)
                        + (jp[2] - sp[2]).powi(2);
                    if d2 < best.0 {
                        best = (d2, si);
                    }
                }
                let (sp, sn) = skirt_w[best.1];
                let raw = (jp[0] - sp[0]) * sn[0] + (jp[1] - sp[1]) * sn[1] + (jp[2] - sp[2]) * sn[2];
                if raw < -0.001 {
                    inside += 1;
                    if raw < worst {
                        worst = raw;
                        worst_at = *jp;
                    }
                } else if raw < 0.006 {
                    close += 1;
                } else {
                    outside += 1;
                }
            }
            println!(
                "jacket-vs-skirt @bind: band y [{:.2},{:.2}] jacket verts: inside={} close={} outside={} worst={:.1}mm at ({:.2},{:.2},{:.2})",
                skirt_lo_y, skirt_hi_y, inside, close, outside, worst * 1000.0, worst_at[0], worst_at[1], worst_at[2]
            );
            // reversed: skirt verts vs jacket surface (jacket's band)
            let j_lo = jacket_w.iter().map(|(p, _)| p[1]).fold(f32::MAX, f32::min);
            let j_hi = jacket_w.iter().map(|(p, _)| p[1]).fold(f32::MIN, f32::max);
            let mut s_in = 0usize;
            let mut s_worst = 0.0f32;
            for (sp, _) in &skirt_w {
                if sp[1] < j_lo || sp[1] > j_hi {
                    continue;
                }
                let mut best = (f32::INFINITY, 0usize);
                for (ji, (jp, _)) in jacket_w.iter().enumerate() {
                    let d2 = (sp[0] - jp[0]).powi(2)
                        + (sp[1] - jp[1]).powi(2)
                        + (sp[2] - jp[2]).powi(2);
                    if d2 < best.0 {
                        best = (d2, ji);
                    }
                }
                let (jp, jn) = jacket_w[best.1];
                let raw = (sp[0] - jp[0]) * jn[0] + (sp[1] - jp[1]) * jn[1] + (sp[2] - jp[2]) * jn[2];
                if raw < -0.001 {
                    s_in += 1;
                    if raw < s_worst {
                        s_worst = raw;
                    }
                }
            }
            println!(
                "skirt-vs-jacket @bind: jacket band y [{:.2},{:.2}] skirt verts inside jacket={} worst={:.1}mm",
                j_lo, j_hi, s_in, s_worst * 1000.0
            );

            // Apply the jacket's cross-region anchors on CPU (same math
            // as the shader's containment-slot clearance branch) and
            // re-measure: the fixed state must leave (almost) no jacket
            // vertex inside the skirt.
            if let Some(anchors) = jacket.containment_anchors.as_ref() {
                if anchors.len() == jacket_w.len() {
                    let mut fixed = jacket_w.clone();
                    let mut applied = 0usize;
                    for (i, anc) in anchors.iter().enumerate() {
                        if anc.body_vertex_idx == u32::MAX || anc.weight <= 1e-4 {
                            continue;
                        }
                        let (sp, sn) = skirt_w[anc.body_vertex_idx as usize];
                        let jp = &mut fixed[i].0;
                        let c = (jp[0] - sp[0]) * sn[0]
                            + (jp[1] - sp[1]) * sn[1]
                            + (jp[2] - sp[2]) * sn[2];
                        if c < anc.min_clearance {
                            let push = (anc.min_clearance - c) * anc.weight;
                            for comp in 0..3 {
                                jp[comp] += sn[comp] * push;
                            }
                            applied += 1;
                        }
                    }
                    let mut in2 = 0usize;
                    let mut worst2 = 0.0f32;
                    for (jp, _) in &fixed {
                        if jp[1] < skirt_lo_y || jp[1] > skirt_hi_y {
                            continue;
                        }
                        let mut best = (f32::INFINITY, 0usize);
                        for (si, (sp, _)) in skirt_w.iter().enumerate() {
                            let d2 = (jp[0] - sp[0]).powi(2)
                                + (jp[1] - sp[1]).powi(2)
                                + (jp[2] - sp[2]).powi(2);
                            if d2 < best.0 {
                                best = (d2, si);
                            }
                        }
                        let (sp, sn) = skirt_w[best.1];
                        let raw = (jp[0] - sp[0]) * sn[0]
                            + (jp[1] - sp[1]) * sn[1]
                            + (jp[2] - sp[2]) * sn[2];
                        if raw < -0.001 {
                            in2 += 1;
                            if raw < worst2 {
                                worst2 = raw;
                            }
                        }
                    }
                    println!(
                        "jacket-vs-skirt AFTER cross-region anchors: pushed={} inside={} worst={:.1}mm (parent prim {:?})",
                        applied,
                        in2,
                        worst2 * 1000.0,
                        jacket.containment_primitive_id.map(|p| p.0)
                    );
                } else {
                    println!(
                        "jacket containment_anchors len {} != verts {} (slot empty or mismatched)",
                        anchors.len(),
                        jacket_w.len()
                    );
                }
            } else {
                println!("jacket has NO containment anchors (Phase 3 did not bind)");
            }
        }
    }
    // ---- who pokes THROUGH the skirt? --------------------------------
    // For every prim whose rest AABB overlaps the skirt's band, count
    // verts sitting OUTSIDE the skirt surface (dot > 2mm along the
    // nearest skirt vertex normal, within 6 cm). The offender list
    // finds the underwear/slip layer that clips through the skirt.
    {
        let find_prim = |name: &str| {
            asset
                .meshes
                .iter()
                .find(|m| m.name.eq_ignore_ascii_case(name))
                .and_then(|m| m.primitives.iter().find(|p| p.vertices.is_some()))
        };
        if let Some(skirt) = find_prim("circle.056") {
            avatar.build_base_pose();
            avatar.compute_global_pose();
            avatar.build_skinning_matrices();
            let sm = avatar.pose.skinning_matrices.clone();
            let skin_world = |prim: &vulvatar_lib::asset::MeshPrimitiveAsset| {
                let vd = prim.vertices.as_ref().unwrap();
                (0..vd.positions.len())
                    .map(|i| {
                        let pos = vd.positions[i];
                        let mut w = [0.0f32; 3];
                        let mut tw = 0.0f32;
                        for k in 0..4 {
                            let wt = vd.joint_weights[i][k];
                            if wt > 0.0001 {
                                let j = vd.joint_indices[i][k] as usize;
                                if let Some(m) = sm.get(j) {
                                    for c in 0..3 {
                                        w[c] += wt
                                            * (m[0][c] * pos[0] + m[1][c] * pos[1]
                                                + m[2][c] * pos[2] + m[3][c]);
                                    }
                                    tw += wt;
                                }
                            }
                        }
                        if tw > 0.001 {
                            for c in 0..3 {
                                w[c] /= tw;
                            }
                        }
                        w
                    })
                    .collect::<Vec<_>>()
            };
            let skirt_w: Vec<([f32; 3], [f32; 3])> = {
                let vd = skirt.vertices.as_ref().unwrap();
                (0..vd.positions.len())
                    .map(|i| {
                        let pos = vd.positions[i];
                        let mut w = [0.0f32; 3];
                        let mut tw = 0.0f32;
                        for k in 0..4 {
                            let wt = vd.joint_weights[i][k];
                            if wt > 0.0001 {
                                let j = vd.joint_indices[i][k] as usize;
                                if let Some(m) = sm.get(j) {
                                    for c in 0..3 {
                                        w[c] += wt
                                            * (m[0][c] * pos[0] + m[1][c] * pos[1]
                                                + m[2][c] * pos[2] + m[3][c]);
                                    }
                                    tw += wt;
                                }
                            }
                        }
                        if tw > 0.001 {
                            for c in 0..3 {
                                w[c] /= tw;
                            }
                        }
                        let n = vd.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]);
                        (w, n)
                    })
                    .collect()
            };
            let s_lo = skirt_w.iter().map(|(p, _)| p[1]).fold(f32::MAX, f32::min);
            let s_hi = skirt_w.iter().map(|(p, _)| p[1]).fold(f32::MIN, f32::max);
            {
                let stats = |anchors: &Option<Vec<vulvatar_lib::asset::SkinAnchor>>, label: &str| {
                    if let Some(a) = anchors {
                        let bound: Vec<f32> =
                            a.iter().filter(|x| x.body_vertex_idx != u32::MAX).map(|x| x.min_clearance).collect();
                        let n_at_floor = bound.iter().filter(|v| **v <= 0.0021).count();
                        println!(
                            "skirt {label}: n={} sum={:.3} max={:.4} at2mmfloor={}",
                            bound.len(),
                            bound.iter().sum::<f32>(),
                            bound.iter().cloned().fold(0.0f32, f32::max),
                            n_at_floor
                        );
                    }
                };
                stats(&skirt.skin_anchors, "skin(body)");
                stats(&skirt.containment_anchors, "containment(underwear)");
            }

            // The GPU applies the skirt's own anchors (Phase-1 body +
            // containment-slot cross-region) before anything reads it,
            // so replicate both pushes for a true post-fix skirt
            // surface, then census pokes against it.
            let skin_world_n = |prim: &vulvatar_lib::asset::MeshPrimitiveAsset| {
                let vd = prim.vertices.as_ref().unwrap();
                (0..vd.positions.len())
                    .map(|i| {
                        let pos = vd.positions[i];
                        let mut w = [0.0f32; 3];
                        let mut tw = 0.0f32;
                        for k in 0..4 {
                            let wt = vd.joint_weights[i][k];
                            if wt > 0.0001 {
                                let j = vd.joint_indices[i][k] as usize;
                                if let Some(m) = sm.get(j) {
                                    for c in 0..3 {
                                        w[c] += wt
                                            * (m[0][c] * pos[0] + m[1][c] * pos[1]
                                                + m[2][c] * pos[2] + m[3][c]);
                                    }
                                    tw += wt;
                                }
                            }
                        }
                        if tw > 0.001 {
                            for c in 0..3 {
                                w[c] /= tw;
                            }
                        }
                        let n = vd.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]);
                        (w, n)
                    })
                    .collect::<Vec<_>>()
            };
            let body_verts: Vec<([f32; 3], [f32; 3])> = asset
                .meshes
                .iter()
                .flat_map(|m| m.primitives.iter())
                .find(|p| Some(p.id) == asset.body_primitive_id)
                .map(|p| skin_world_n(p))
                .unwrap_or_default();
            let mut skirt_fixed = skirt_w.clone();
            let apply_clearance = |w: &mut Vec<([f32; 3], [f32; 3])>,
                                   anchors: &Option<Vec<vulvatar_lib::asset::SkinAnchor>>,
                                   parent: &[([f32; 3], [f32; 3])]| {
                let Some(anchors) = anchors else { return };
                if anchors.len() != w.len() || parent.is_empty() {
                    return;
                }
                for (i, anc) in anchors.iter().enumerate() {
                    if anc.body_vertex_idx == u32::MAX || anc.weight <= 1e-4 {
                        continue;
                    }
                    let &(pp, pn) = parent.get(anc.body_vertex_idx as usize).unwrap_or(&(w[i].0, [0.0, 1.0, 0.0]));
                    let v = &mut w[i].0;
                    let c = (v[0] - pp[0]) * pn[0] + (v[1] - pp[1]) * pn[1] + (v[2] - pp[2]) * pn[2];
                    if c < anc.min_clearance {
                        let push = (anc.min_clearance - c) * anc.weight;
                        for comp in 0..3 {
                            v[comp] += pn[comp] * push;
                        }
                    }
                }
            };
            apply_clearance(&mut skirt_fixed, &skirt.skin_anchors, &body_verts);
            let under_parent_verts: Vec<([f32; 3], [f32; 3])> = skirt
                .containment_primitive_id
                .and_then(|pid| {
                    asset
                        .meshes
                        .iter()
                        .flat_map(|m| m.primitives.iter())
                        .find(|p| p.id == pid)
                        .map(|p| skin_world_n(p))
                })
                .unwrap_or_default();
            apply_clearance(&mut skirt_fixed, &skirt.containment_anchors, &under_parent_verts);

            println!("--- poke-through census vs skirt (band y [{:.2},{:.2}]) ---", s_lo, s_hi);
            for mesh in &asset.meshes {
                for prim in &mesh.primitives {
                    if prim.id == skirt.id || prim.vertices.is_none() {
                        continue;
                    }
                    let mat_name = asset
                        .materials
                        .iter()
                        .find(|m| m.id == prim.material_id)
                        .map(|m| m.name.clone())
                        .unwrap_or_default();
                    let w = skin_world(prim);
                    let mut pokes = 0usize;
                    let mut pokes_fixed = 0usize;
                    let mut near = 0usize;
                    let mut worst = 0.0f32;
                    let mut worst_fixed = 0.0f32;
                    let mut pmin = [f32::MAX; 3];
                    let mut pmax = [f32::MIN; 3];
                    for p in &w {
                        for c in 0..3 {
                            pmin[c] = pmin[c].min(p[c]);
                            pmax[c] = pmax[c].max(p[c]);
                        }
                        if p[1] < s_lo || p[1] > s_hi {
                            continue;
                        }
                        let mut best = (f32::INFINITY, 0usize);
                        let mut best_f = (f32::INFINITY, 0usize);
                        for (si, (sp, _)) in skirt_w.iter().enumerate() {
                            let d2 = (p[0] - sp[0]).powi(2)
                                + (p[1] - sp[1]).powi(2)
                                + (p[2] - sp[2]).powi(2);
                            if d2 < best.0 {
                                best = (d2, si);
                            }
                        }
                        for (si, (sp, _)) in skirt_fixed.iter().enumerate() {
                            let d2 = (p[0] - sp[0]).powi(2)
                                + (p[1] - sp[1]).powi(2)
                                + (p[2] - sp[2]).powi(2);
                            if d2 < best_f.0 {
                                best_f = (d2, si);
                            }
                        }
                        if best.0 > 0.06 * 0.06 {
                            continue;
                        }
                        near += 1;
                        let (sp, sn) = skirt_w[best.1];
                        let d = (p[0] - sp[0]) * sn[0] + (p[1] - sp[1]) * sn[1] + (p[2] - sp[2]) * sn[2];
                        if d > 0.002 {
                            pokes += 1;
                            if d > worst {
                                worst = d;
                            }
                        }
                        if best_f.0 <= 0.06 * 0.06 {
                            let (fp, fnn) = skirt_fixed[best_f.1];
                            let df = (p[0] - fp[0]) * fnn[0]
                                + (p[1] - fp[1]) * fnn[1]
                                + (p[2] - fp[2]) * fnn[2];
                            if df > 0.002 {
                                pokes_fixed += 1;
                                if df > worst_fixed {
                                    worst_fixed = df;
                                }
                            }
                        }
                    }
                    if near > 20 {
                        println!(
                            "  {:<16} prim{:>3?} mat='{:<24}' bbox_y=[{:.2},{:.2}] near={} pokes={} worst=+{:.1}mm | after-skirt-anchors pokes={} worst=+{:.1}mm",
                            mesh.name,
                            prim.id.0,
                            mat_name,
                            pmin[1],
                            pmax[1],
                            near,
                            pokes,
                            worst * 1000.0,
                            pokes_fixed,
                            worst_fixed * 1000.0
                        );
                        // Spatial distribution of residual pokes
                        // (quadrants: front/back x left/right).
                        let mut quad = [0usize; 4];
                        for p in &w {
                            if p[1] < s_lo || p[1] > s_hi {
                                continue;
                            }
                            let mut best_f = (f32::INFINITY, 0usize);
                            for (si, (sp, _)) in skirt_fixed.iter().enumerate() {
                                let d2 = (p[0] - sp[0]).powi(2)
                                    + (p[1] - sp[1]).powi(2)
                                    + (p[2] - sp[2]).powi(2);
                                if d2 < best_f.0 {
                                    best_f = (d2, si);
                                }
                            }
                            if best_f.0 > 0.06 * 0.06 {
                                continue;
                            }
                            let (fp, fnn) = skirt_fixed[best_f.1];
                            let df = (p[0] - fp[0]) * fnn[0]
                                + (p[1] - fp[1]) * fnn[1]
                                + (p[2] - fp[2]) * fnn[2];
                            if df > 0.002 {
                                let qi = if p[2] >= 0.0 { 0 } else { 2 } + if p[0] >= 0.0 { 0 } else { 1 };
                                quad[qi] += 1;
                            }
                        }
                        let mut w_pos = [0.0f32; 3];
                        let mut w_val = 0.0f32;
                        let mut w_by_y = std::collections::BTreeMap::new();
                        for p in &w {
                            if p[1] < s_lo || p[1] > s_hi {
                                continue;
                            }
                            let mut best_f = (f32::INFINITY, 0usize);
                            for (si, (sp, _)) in skirt_fixed.iter().enumerate() {
                                let d2 = (p[0] - sp[0]).powi(2)
                                    + (p[1] - sp[1]).powi(2)
                                    + (p[2] - sp[2]).powi(2);
                                if d2 < best_f.0 {
                                    best_f = (d2, si);
                                }
                            }
                            if best_f.0 > 0.06 * 0.06 {
                                continue;
                            }
                            let (fp, fnn) = skirt_fixed[best_f.1];
                            let df = (p[0] - fp[0]) * fnn[0]
                                + (p[1] - fp[1]) * fnn[1]
                                + (p[2] - fp[2]) * fnn[2];
                            if df > 0.002 {
                                let ybin = (p[1] * 50.0).round() as i32;
                                *w_by_y.entry(ybin).or_insert(0usize) += 1;
                                if df > w_val {
                                    w_val = df;
                                    w_pos = *p;
                                }
                            }
                        }
                        println!(
                            "      residual poke quadrants [front-R, front-L, back-R, back-L]: {:?}",
                            quad
                        );
                        println!(
                            "      residual by y(2cm bins): {:?}  worst=+{:.1}mm at ({:.2},{:.2},{:.2})",
                            w_by_y, w_val * 1000.0, w_pos[0], w_pos[1], w_pos[2]
                        );
                    }
                }
            }
        }
    }
    println!("--- per-mesh joint census (arm/hand influence) ---");
    for mesh in &asset.meshes {
        for prim in &mesh.primitives {
            let Some(vd) = prim.vertices.as_ref() else { continue };
            if vd.joint_indices.is_empty() || vd.joint_weights.is_empty() {
                continue;
            }
            let mut wsum: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
            for i in 0..vd.positions.len().min(vd.joint_indices.len()) {
                for k in 0..4 {
                    let w = vd.joint_weights[i][k];
                    if w > 0.0001 {
                        *wsum.entry(vd.joint_indices[i][k] as usize).or_insert(0.0) += w;
                    }
                }
            }
            let total: f32 = wsum.values().sum();
            let arm_w: f32 = wsum
                .iter()
                .filter(|(j, _)| {
                    let n = asset
                        .skeleton
                        .nodes
                        .get(**j)
                        .map(|n| n.name.to_lowercase())
                        .unwrap_or_default();
                    n.contains("arm") || n.contains("hand") || n.contains("wrist")
                })
                .map(|(_, w)| *w)
                .sum();
            let n_joints = wsum.len();
            println!(
                "  {:<16} prim{:?} verts={:>6} joints={:>3} arm_w={:>7.2} ({:.1}%)",
                mesh.name,
                prim.id.0,
                vd.positions.len(),
                n_joints,
                arm_w,
                100.0 * arm_w / total.max(1e-6)
            );
        }
    }
    if false {
        println!("(no Circle.056 mesh found — skipping CPU skinning check)");
    }

    // Full-body render, no cloth / expressions / springs.
    let width = 1024u32;
    let height = 1024u32;
    let mut renderer = VulkanRenderer::new();
    renderer.initialize();

    let camera = ViewportCamera {
        distance: 3.4,
        pan: [0.0, 0.85],
        yaw_deg: 25.0,
        pitch_deg: 2.0,
        fov_deg: 40.0,
    };
    let (view, eye_pos) = build_view_matrix(&camera);
    let projection = build_projection_matrix(camera.fov_deg, 1.0, 0.05, 20.0);

    let mut mesh_instances = Vec::new();
    for mesh in &asset.meshes {
        for prim in &mesh.primitives {
            let material_asset = asset.materials.iter().find(|m| m.id == prim.material_id);
            let mut material_binding = material_asset
                .map(MaterialUploadRequest::from_asset_material)
                .unwrap_or_else(MaterialUploadRequest::default_material);
            material_binding.mode = MaterialShaderMode::ToonLike;
            let alpha_mode = match material_binding.alpha_mode {
                vulvatar_lib::asset::AlphaMode::Opaque => RenderAlphaMode::Opaque,
                vulvatar_lib::asset::AlphaMode::Mask(_) => RenderAlphaMode::Cutout,
                vulvatar_lib::asset::AlphaMode::Blend => RenderAlphaMode::Blend,
            };
            let cull_mode = if material_binding.double_sided {
                RenderCullMode::DoubleSided
            } else {
                RenderCullMode::BackFace
            };
            mesh_instances.push(RenderMeshInstance {
                mesh_id: mesh.id,
                primitive_id: prim.id,
                material_binding,
                bounds: prim.bounds,
                alpha_mode,
                cull_mode,
                outline: Default::default(),
                primitive_data: Some(Arc::clone(prim)),
                morph_weights: Vec::new(),
            });
        }
    }

    let frame_input = RenderFrameInput {
        camera: CameraState {
            view,
            projection,
            position_ws: eye_pos,
            viewport_extent: [width, height],
        },
        lighting: LightingState::default(),
        instances: vec![RenderAvatarInstance {
            instance_id: avatar.id,
            world_transform: avatar.world_transform.clone(),
            mesh_instances,
            skinning_matrices: avatar.pose.skinning_matrices.clone(),
            cloth_deforms: Vec::new(),
body_sdf: None,
            debug_flags: Default::default(),
        }],
        output_request: OutputTargetRequest {
            preview_enabled: true,
            output_enabled: true,
            extent: [width, height],
            color_space: RenderColorSpace::Srgb,
            alpha_mode: vulvatar_lib::renderer::frame_input::RenderOutputAlpha::Opaque,
            export_mode: RenderExportMode::CpuReadback,
            msaa: MsaaMode::Off,
        },
        background_image_path: None,
        show_ground_grid: false,
        background_color: [0.12, 0.12, 0.14],
        transparent_background: false,
        avatar_opacity: 1.0,
        bloom: Default::default(),
        generative_background: Default::default(),
        background_tracking: Default::default(),
        time_seconds: 0.0,
    };

    // Double render for the pipelined CPU readback.
    let _ = renderer.render(&frame_input).map_err(|e| format!("warm-up render failed: {e}"))?;
    let result = renderer.render(&frame_input).map_err(|e| format!("render failed: {e}"))?;

    if let Some(exported) = result.exported_frame.as_ref() {
        if let Some(pixels) = exported.cpu_pixel_data() {
            let path = out.join("tpose_fullbody.png");
            let img: ImageBuffer<image::Rgba<u8>, _> =
                ImageBuffer::from_raw(width, height, pixels.to_vec())
                    .ok_or("png buffer construction failed")?;
            img.save(&path).map_err(|e| format!("save {}: {e}", path.display()))?;
            println!("wrote {}", path.display());
        }
    }

    // ---- posed variant: lower both arms ~65deg in world space -------
    // Bind pose hides every skinning/anchor issue (matrices are
    // identity). The live bug shows under tracking, i.e. any deviation
    // from bind — so rotate the upper arms down and re-render.
    avatar.build_base_pose();
    avatar.compute_global_pose();
    let mut arm_report = String::new();
    for (bone, sign) in [
        (vulvatar_lib::asset::HumanoidBone::LeftUpperArm, -1.0f32),
        (vulvatar_lib::asset::HumanoidBone::RightUpperArm, 1.0),
    ] {
        let Some(node_idx) = asset
            .skeleton
            .nodes
            .iter()
            .position(|n| n.humanoid_bone == Some(bone))
        else {
            continue;
        };
        let parent_world = match asset.skeleton.nodes[node_idx].parent {
            Some(p) => vulvatar_lib::math_utils::mat4_rotation_to_quat(
                &avatar.pose.global_transforms[p.0 as usize],
            ),
            None => [0.0, 0.0, 0.0, 1.0],
        };
        let world_rest = vulvatar_lib::math_utils::mat4_rotation_to_quat(
            &avatar.pose.global_transforms[node_idx],
        );
        let ang = sign * 65.0f32.to_radians();
        let delta: vulvatar_lib::math_utils::Quat =
            [0.0, 0.0, (ang * 0.5).sin(), (ang * 0.5).cos()];
        let world_new = vulvatar_lib::math_utils::quat_normalize(
            &vulvatar_lib::math_utils::quat_mul(&delta, &world_rest),
        );
        let local_new = vulvatar_lib::math_utils::quat_normalize(
            &vulvatar_lib::math_utils::quat_mul(
                &vulvatar_lib::math_utils::quat_conjugate(&parent_world),
                &world_new,
            ),
        );
        avatar.pose.local_transforms[node_idx].rotation = local_new;
        arm_report.push_str(&format!(
            "{}->node{} ",
            format!("{bone:?}").trim_start_matches("Some("),
            node_idx
        ));
    }
    println!("posed arms: {}", arm_report.trim_end());
    avatar.compute_global_pose();
    avatar.build_skinning_matrices();

    // CPU sanity: skirt skinned bbox in the posed state.
    if let Some(skirt) = asset
        .meshes
        .iter()
        .find(|m| m.name.eq_ignore_ascii_case("circle.056"))
        .and_then(|m| m.primitives.iter().find(|p| p.vertices.is_some()))
    {
        let vd = skirt.vertices.as_ref().unwrap();
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for (i, &pos) in vd.positions.iter().enumerate() {
            let mut world = [0.0f32; 3];
            let mut total_w = 0.0;
            for k in 0..4 {
                let w = vd.joint_weights[i][k];
                if w > 0.0001 {
                    let j = vd.joint_indices[i][k] as usize;
                    if j < avatar.pose.skinning_matrices.len() {
                        let sm = &avatar.pose.skinning_matrices[j];
                        for c in 0..3 {
                            world[c] += w
                                * (sm[0][c] * pos[0] + sm[1][c] * pos[1] + sm[2][c] * pos[2]
                                    + sm[3][c]);
                        }
                        total_w += w;
                    }
                }
            }
            if total_w > 0.001 {
                for c in 0..3 {
                    world[c] /= total_w;
                }
            }
            for c in 0..3 {
                lo[c] = lo[c].min(world[c]);
                hi[c] = hi[c].max(world[c]);
            }
        }
        println!(
            "Circle.056 CPU-skinned bbox (arms down): [{:.3},{:.3},{:.3}] .. [{:.3},{:.3},{:.3}]",
            lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]
        );
    }

    let mut frame_input2 = frame_input;
    frame_input2.instances[0].skinning_matrices = avatar.pose.skinning_matrices.clone();
    let _ = renderer.render(&frame_input2).map_err(|e| format!("warm-up render 2 failed: {e}"))?;
    let result2 = renderer
        .render(&frame_input2)
        .map_err(|e| format!("render 2 failed: {e}"))?;
    if let Some(exported) = result2.exported_frame.as_ref() {
        if let Some(pixels) = exported.cpu_pixel_data() {
            let path = out.join("tpose_armsdown.png");
            let img: ImageBuffer<image::Rgba<u8>, _> =
                ImageBuffer::from_raw(width, height, pixels.to_vec())
                    .ok_or("png buffer construction failed")?;
            img.save(&path).map_err(|e| format!("save {}: {e}", path.display()))?;
            println!("wrote {}", path.display());
        }
    }

    // ---- spring excitation test -------------------------------------
    // The app steps the (spring-driven) skirt chains every frame with
    // the user's tuning (sway 2.0, gravity_offset 1.0, natural gravity).
    // Excite with body sway + arm waves, then freeze; a healthy solver
    // relaxes the skirt back to a hang. Report chain-tip world radii
    // and capture renders along the way.
    let tuning = vulvatar_lib::simulation::spring::SpringTuning {
        sway_scale: 2.0,
        gravity_offset: 1.0,
        natural_gravity: true,
    };
    let hips_idx = asset
        .skeleton
        .nodes
        .iter()
        .position(|n| n.name.eq_ignore_ascii_case("hips"))
        .unwrap_or(0);
    let arm_l = asset
        .skeleton
        .nodes
        .iter()
        .position(|n| n.humanoid_bone == Some(vulvatar_lib::asset::HumanoidBone::LeftUpperArm));
    let arm_r = asset
        .skeleton
        .nodes
        .iter()
        .position(|n| n.humanoid_bone == Some(vulvatar_lib::asset::HumanoidBone::RightUpperArm));
    let skirt_tips: Vec<usize> = asset
        .skeleton
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| {
            let l = n.name.to_lowercase();
            l.starts_with("skirt_") && l.ends_with(".003")
        })
        .map(|(i, _)| i)
        .collect();
    let rest_hips = avatar.pose.local_transforms[hips_idx].clone();

    let fixed_dt = 1.0f32 / 60.0;
    let total_steps = 480usize;
    let capture_at: Vec<usize> = vec![60, 180, 300, 479];

    for step in 0..total_steps {
        avatar.build_base_pose();
        let t = step as f32 * fixed_dt;
        if step < 240 {
            let sway = (t * 3.0).sin() * 0.025;
            let mut cur = rest_hips.clone();
            cur.translation[0] += sway;
            let roll = (t * 3.0).sin() * 0.04;
            let q_roll = [0.0, 0.0, (roll * 0.5).sin(), (roll * 0.5).cos()];
            cur.rotation = vulvatar_lib::math_utils::quat_mul(&rest_hips.rotation, &q_roll);
            avatar.pose.local_transforms[hips_idx] = cur;
            let wave = (t * 5.0).sin() * 0.35;
            for (idx, sign) in [(arm_l, -1.0f32), (arm_r, 1.0)] {
                if let Some(i) = idx {
                    let rest = avatar.pose.local_transforms[i].rotation;
                    let dq = [0.0, 0.0, (sign * wave * 0.5).sin(), (sign * wave * 0.5).cos()];
                    avatar.pose.local_transforms[i].rotation =
                        vulvatar_lib::math_utils::quat_mul(&rest, &dq);
                }
            }
        }
        avatar.compute_global_pose();
        vulvatar_lib::simulation::spring::step_spring_bones(
            fixed_dt,
            &mut avatar,
            &[],
            &tuning,
            [0.0, -1.0, 0.0],
            1.0,
            None,
        );
        if capture_at.contains(&step) {
            avatar.compute_global_pose();
            avatar.build_skinning_matrices();
            let phase = if step < 240 { "moving" } else { "frozen" };
            let mut worst = (0.0f32, String::new());
            let mut min_y = f32::INFINITY;
            for &tip in &skirt_tips {
                let m = &avatar.pose.global_transforms[tip];
                let (x, y, z) = (m[3][0], m[3][1], m[3][2]);
                let rad = (x * x + z * z).sqrt();
                min_y = min_y.min(y);
                if rad > worst.0 {
                    worst = (
                        rad,
                        format!("{}=({:.2},{:.2},{:.2})", asset.skeleton.nodes[tip].name, x, y, z),
                    );
                }
            }
            println!(
                "spring step {:>3} ({}): worst skirt tip radius {:.3}m (min tip y {:.2})  {}",
                step, phase, worst.0, min_y, worst.1
            );
            let mut fi = frame_input2.clone();
            fi.instances[0].skinning_matrices = avatar.pose.skinning_matrices.clone();
            let _ = renderer.render(&fi);
            if let Ok(res) = renderer.render(&fi) {
                if let Some(exported) = res.exported_frame.as_ref() {
                    if let Some(pixels) = exported.cpu_pixel_data() {
                        let path = out.join(format!("spring_step_{:03}.png", step));
                        let img: ImageBuffer<image::Rgba<u8>, _> =
                            ImageBuffer::from_raw(width, height, pixels.to_vec())
                                .ok_or("png buffer construction failed")?;
                        let _ = img.save(&path);
                    }
                }
            }
        }
    }
    // ---- skirt mesh topology census ----------------------------------
    {
        let skirt = asset
            .meshes
            .iter()
            .find(|m| m.name.eq_ignore_ascii_case("circle.056"))
            .and_then(|m| m.primitives.iter().find(|p| p.vertices.is_some()));
        if let Some(skirt) = skirt {
            let vd = skirt.vertices.as_ref().unwrap();
            let idx = skirt.indices.as_ref().unwrap();
            let mut edges = std::collections::HashSet::new();
            for t in idx.chunks_exact(3) {
                for pair in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    edges.insert(if pair.0 < pair.1 { pair } else { (pair.1, pair.0) });
                }
            }
            let mut degree = vec![0u32; vd.positions.len()];
            for &(a, b) in &edges {
                degree[a as usize] += 1;
                degree[b as usize] += 1;
            }
            let mut hist = std::collections::BTreeMap::new();
            for d in &degree {
                *hist.entry(*d).or_insert(0usize) += 1;
            }
            println!(
                "skirt topology: verts={} tris={} unique_edges={} degree_hist={:?}",
                vd.positions.len(),
                idx.len() / 3,
                edges.len(),
                hist
            );
        }
    }

    // ---- CPU-solver comparison on the same auto asset ----------------
    {
        for slot in avatar.cloth_overlays.iter_mut() {
            slot.state.solver_backend =
                vulvatar_lib::simulation::cloth_gpu_boundary::ClothSolverBackend::Cpu;
        }
        for fidx in 0..12usize {
            avatar.build_base_pose();
            avatar.compute_global_pose();
            vulvatar_lib::simulation::cloth_solver::step_cloth(1.0 / 60.0, &mut avatar, &[]);
            if fidx == 0 {
                if let Some(slot) = avatar.cloth_overlays.first() {
                    let mut worst: Vec<(f32, usize)> = slot
                        .state
                        .sim_positions
                        .iter()
                        .enumerate()
                        .map(|(i, p)| (p[0].abs().max(p[1].abs()).max(p[2].abs()), i))
                        .collect();
                    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                    for (mag, i) in worst.iter().take(5) {
                        let pin = slot.sim.particles.get(*i).map(|p| p.pinned);
                        let n_c = slot
                            .sim
                            .distance_constraints
                            .iter()
                            .filter(|c| c.a as usize == *i || c.b as usize == *i)
                            .count();
                        println!(
                            "cpu f0 worst particle i={i} |p|={mag:.3e} pinned={pin:?} degree={n_c}"
                        );
                    }
                }
            }
            if fidx % 3 == 0 {
                if let Some(slot) = avatar.cloth_overlays.first() {
                    let mut lo = [f32::MAX; 3];
                    let mut hi = [f32::MIN; 3];
                    for q in &slot.state.sim_positions {
                        for c in 0..3 {
                            lo[c] = lo[c].min(q[c]);
                            hi[c] = hi[c].max(q[c]);
                        }
                    }
                    println!(
                        "cpu cloth frame {fidx:>2}: bbox=[{:.2},{:.2},{:.2}]..[{:.2},{:.2},{:.2}]",
                        lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]
                    );
                }
            }
        }
    }

    // ---- auto-cloth live test ----------------------------------------
    // Reproduce the live bug offline: attach the auto-generated GPU
    // cloth exactly like the app, render frames, and read back the
    // solver's positions (the same readback the app folds every
    // frame). A healthy skirt keeps its bbox near the hips.
    {
        let attached = vulvatar_lib::simulation::auto_cloth::attach_auto_cloth(&mut avatar);
        println!("auto-cloth attached: {attached}");
        let mut mesh_instances2 = frame_input2.instances[0].mesh_instances.clone();
        for fidx in 0..12usize {
            avatar.build_base_pose();
            avatar.compute_global_pose();
            avatar.build_skinning_matrices();
            // collect cloth deforms like the app does (app/render.rs)
            let mut cloth_deforms = Vec::new();
            for slot in avatar.cloth_overlays.iter().filter(|s| s.enabled) {
                let Some(target) = slot.state.target_primitive_id else { continue };
                use vulvatar_lib::renderer::frame_input::{
                    ClothDeformSnapshot, ClothGpuAttachData, ClothGpuDispatchControl,
                };
                let (gpu_control, gpu_attach) =
                    if slot.state.solver_backend
                        == vulvatar_lib::simulation::cloth_gpu_boundary::ClothSolverBackend::Gpu
                    {
                        let sim = &slot.sim;
                        let wind = vulvatar_lib::math_utils::vec3_scale(
                            &sim.wind_direction,
                            sim.wind_response,
                        );
                        let pins = (0..sim.particles.len())
                            .map(|_| [0.0f32; 3])
                            .collect::<Vec<_>>();
                        let _ = pins;
                        let colliders: Vec<vulvatar_lib::renderer::frame_input::ClothGpuCollider> =
                            Vec::new();
                        (
                            Some(ClothGpuDispatchControl {
                                dt: 1.0 / 60.0,
                                substeps: 1,
                                damping: sim.damping,
                                gravity: sim.gravity,
                                wind_force: wind,
                                solver_iterations: sim.solver_iterations as u32,
                                pin_positions: gpu_pins(sim, &avatar.pose.global_transforms),
                                collision_margin: sim.collision_margin,
                                colliders,
                                self_collision: sim.self_collision,
                                self_collision_radius: sim.self_collision_radius,
                            }),
                            Some(ClothGpuAttachData {
                                constraints: sim
                                    .distance_constraints
                                    .iter()
                                    .map(|c| (c.a as u32, c.b as u32, c.rest_length, c.stiffness))
                                    .collect(),
                                triangle_indices: sim.triangle_indices.clone(),
                                inv_masses: sim.particles.iter().map(|p| p.inv_mass).collect(),
                                pinned: sim.particles.iter().map(|p| p.pinned).collect(),
                            }),
                        )
                    } else {
                        (None, None)
                    };
                cloth_deforms.push(ClothDeformSnapshot {
                    target_primitive_id: target,
                    target_mesh_id: slot.state.target_mesh_id,
                    vertex_offset: slot.state.target_vertex_offset,
                    vertex_count: slot.state.target_vertex_count,
                    deformed_positions: slot.state.deform_output.deformed_positions.clone(),
                    deformed_normals: slot.state.deform_output.deformed_normals.clone(),
                    version: slot.state.deform_output.version,
                    solver_backend: slot.state.solver_backend,
                    gpu_control,
                    gpu_attach,
                });
            }
            let mut fi = frame_input2.clone();
            fi.instances[0].mesh_instances = mesh_instances2.clone();
            fi.instances[0].skinning_matrices = avatar.pose.skinning_matrices.clone();
            fi.instances[0].cloth_deforms = cloth_deforms;
            let _ = renderer.render(&fi);
            if let Ok(res) = renderer.render(&fi) {
                if let Some(entry) = res.cloth_readback.first() {
                    let mut lo = [f32::MAX; 3];
                    let mut hi = [f32::MIN; 3];
                    let mut nan = 0usize;
                    for q in &entry.positions {
                        for c in 0..3 {
                            if q[c].is_nan() {
                                nan += 1;
                            } else {
                                lo[c] = lo[c].min(q[c]);
                                hi[c] = hi[c].max(q[c]);
                            }
                        }
                    }
                    println!(
                        "cloth frame {fidx:>2}: n={} nan={nan} bbox=[{:.2},{:.2},{:.2}]..[{:.2},{:.2},{:.2}] v={}",
                        entry.positions.len(), lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], entry.version
                    );
                }
                if fidx == 11 {
                    if let Some(exported) = res.exported_frame.as_ref() {
                        if let Some(pixels) = exported.cpu_pixel_data() {
                            let path = out.join("tpose_autocloth.png");
                            let img: ImageBuffer<image::Rgba<u8>, _> =
                                ImageBuffer::from_raw(width, height, pixels.to_vec())
                                    .ok_or("png buffer construction failed")?;
                            let _ = img.save(&path);
                            println!("wrote {}", path.display());
                        }
                    }
                }
            }
        }
    }

    // ---- expression/morph test --------------------------------------
    // Face tracking + lipsync drive expression morphs continuously —
    // even in T-pose. If morph weights leak onto garment primitives,
    // the costume breaks pose-independently. Print morph-target
    // carriage per primitive, then render with expressions fully on.
    println!("--- morph targets per primitive ---");
    for mesh in &asset.meshes {
        for prim in &mesh.primitives {
            if !prim.morph_targets.is_empty() {
                println!(
                    "  {:<16} prim{:?} targets={} verts={}",
                    mesh.name,
                    prim.id.0,
                    prim.morph_targets.len(),
                    prim.vertex_count
                );
            }
        }
    }
    let expr_names: Vec<String> = asset
        .default_expressions
        .expressions
        .iter()
        .map(|e| e.name.clone())
        .collect();
    println!("expressions available: {:?}", &expr_names);
    avatar.build_base_pose();
    avatar.compute_global_pose();
    avatar.build_skinning_matrices();
    avatar.expression_weights = expr_names
        .iter()
        .take(8)
        .map(|n| vulvatar_lib::avatar::expressions::ResolvedExpressionWeight {
            name: n.clone(),
            weight: 1.0,
        })
        .collect();
    println!(
        "driving expressions ON: {:?}",
        &expr_names.iter().take(8).map(|s| s.as_str()).collect::<Vec<_>>()
    );
    let mut fi3 = frame_input2.clone();
    fi3.instances[0].skinning_matrices = avatar.pose.skinning_matrices.clone();
    for mi in fi3.instances[0].mesh_instances.iter_mut() {
        if let Some(prim) = mi.primitive_data.as_ref() {
            mi.morph_weights = avatar.morph_weights_for_prim(prim);
            if !mi.morph_weights.is_empty() {
                println!(
                    "  prim {:?} gets {} morph weights (max {:.2})",
                    mi.primitive_id.0,
                    mi.morph_weights.len(),
                    mi.morph_weights.iter().cloned().fold(0.0f32, f32::max)
                );
            }
        }
    }
    let _ = renderer.render(&fi3);
    if let Ok(res) = renderer.render(&fi3) {
        if let Some(exported) = res.exported_frame.as_ref() {
            if let Some(pixels) = exported.cpu_pixel_data() {
                let path = out.join("tpose_expr_on.png");
                let img: ImageBuffer<image::Rgba<u8>, _> =
                    ImageBuffer::from_raw(width, height, pixels.to_vec())
                        .ok_or("png buffer construction failed")?;
                let _ = img.save(&path);
                println!("wrote {}", path.display());
            }
        }
    }
    println!("done");
    Ok(())
}

fn gpu_pins(
    sim: &vulvatar_lib::simulation::cloth::ClothSimState,
    global_transforms: &[vulvatar_lib::asset::Mat4],
) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0f32; 3]; sim.particles.len()];
    for pin in &sim.pin_targets {
        let Some(mat) = global_transforms.get(pin.node_index) else {
            continue;
        };
        let [ox, oy, oz] = pin.offset;
        let world = [
            mat[0][0] * ox + mat[1][0] * oy + mat[2][0] * oz + mat[3][0],
            mat[0][1] * ox + mat[1][1] * oy + mat[2][1] * oz + mat[3][1],
            mat[0][2] * ox + mat[1][2] * oy + mat[2][2] * oz + mat[3][2],
        ];
        for &pi in &pin.particle_indices {
            if pi < out.len() {
                out[pi] = world;
            }
        }
    }
    out
}

fn build_view_matrix(cam: &ViewportCamera) -> (vulvatar_lib::asset::Mat4, [f32; 3]) {
    let yaw = cam.yaw_deg.to_radians();
    let pitch = cam.pitch_deg.to_radians();
    let (sy, cy) = (yaw.sin(), yaw.cos());
    let (sp, cp) = (pitch.sin(), pitch.cos());

    let right = [cy, 0.0, -sy];
    let up = [-sy * sp, cp, -cy * sp];

    let wx = cam.pan[0] * right[0] + cam.pan[1] * up[0];
    let wy = cam.pan[0] * right[1] + cam.pan[1] * up[1];
    let wz = cam.pan[0] * right[2] + cam.pan[1] * up[2];

    let eye_x = cam.distance * cp * sy + wx;
    let eye_y = cam.distance * sp + wy;
    let eye_z = cam.distance * cp * cy + wz;

    let target = [wx, wy, wz];
    let fwd = [
        target[0] - eye_x,
        target[1] - eye_y,
        target[2] - eye_z,
    ];
    let len = (fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2])
        .sqrt()
        .max(1e-6);
    let f = [fwd[0] / len, fwd[1] / len, fwd[2] / len];

    let world_up = [0.0f32, 1.0, 0.0];
    let r = [
        f[1] * world_up[2] - f[2] * world_up[1],
        f[2] * world_up[0] - f[0] * world_up[2],
        f[0] * world_up[1] - f[1] * world_up[0],
    ];
    let rlen = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt().max(1e-6);
    let r = [r[0] / rlen, r[1] / rlen, r[2] / rlen];
    let u = [
        r[1] * f[2] - r[2] * f[1],
        r[2] * f[0] - r[0] * f[2],
        r[0] * f[1] - r[1] * f[0],
    ];

    (
        [
            [r[0], r[1], r[2], -(r[0] * eye_x + r[1] * eye_y + r[2] * eye_z)],
            [u[0], u[1], u[2], -(u[0] * eye_x + u[1] * eye_y + u[2] * eye_z)],
            [-f[0], -f[1], -f[2], (f[0] * eye_x + f[1] * eye_y + f[2] * eye_z)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [eye_x, eye_y, eye_z],
    )
}

fn build_projection_matrix(
    fov_deg: f32,
    aspect: f32,
    near: f32,
    far: f32,
) -> vulvatar_lib::asset::Mat4 {
    let fov_rad = fov_deg.to_radians();
    let f = 1.0 / (fov_rad * 0.5).tan();
    let a = far / (near - far);
    let b = far * near / (near - far);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0],
        [0.0, 0.0, a, b],
        [0.0, 0.0, -1.0, 0.0],
    ]
}
