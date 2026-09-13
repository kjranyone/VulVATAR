//! Auto-cloth: turn skirt-classified primitives into simulated garments
//! without authoring.
//!
//! The clearance-anchor system keeps statically-skinned garments apart,
//! but a skirt is fundamentally a swinging surface — anchors approximate
//! that with rest-derived targets. This module derives a `ClothAsset`
//! straight from the loaded avatar at attach time: particles from the
//! rest-pose skinned positions, a pin ring at the waist band bound to
//! the skirt root (falling back to the hips), distance constraints from
//! the mesh edges. The generated overlay is runtime-only (no
//! `source_path`, never round-trips through projects) and regenerates
//! on every load, so avatar re-exports self-heal.
//!
//! The builder mirrors `diagnose_cloth::build_skirt_cloth_asset` (the
//! verification bin's Yumeka recipe), generalized: the target primitive
//! is found by the Phase-1 skirt classifier (mesh/material name, or a
//! majority of vertices weighted to skirt-named bones), and the pin
//! band is the primitive's own top ring rather than a fixed world Y.

use std::collections::HashSet;

use log::{info, warn};

use crate::asset::{
    AvatarAsset, ClothAsset, ClothConstraintSet, ClothMappingMode, ClothMeshMapping, ClothOverlayId,
    ClothOverlayMetadata, ClothPin, ClothRegionTag, ClothRenderRegionBinding, ClothSimVertex,
    ClothSimulationMesh, ClothSolverParams, ClothStableRefSet, DistanceConstraint, MeshRef,
    NodeRef, PrimitiveRef, VertexSubsetRef,
};
use crate::avatar::AvatarInstance;

/// How far below the primitive's top the pin band reaches. 3.5 cm
/// covers a waistband ring on human-scale avatars without pinning the
/// flare below it.
const PIN_BAND: f32 = 0.035;
/// Skirt classifier bone-weight threshold (Phase 1 parity).
const SKIRT_WEIGHT_RATIO: f32 = 0.4;

/// Build one `ClothAsset` per skirt-classified primitive on the avatar.
/// The instance must have its rest pose built (base pose → global pose
/// → skinning matrices); callers in the load path do that before us.
pub fn build_auto_cloth_assets(avatar: &AvatarInstance, first_id: usize) -> Vec<ClothAsset> {
    let asset = &avatar.asset;
    let mut out = Vec::new();
    for mesh in &asset.meshes {
        for prim in &mesh.primitives {
            let Some(vd) = prim.vertices.as_ref() else {
                continue;
            };
            let Some(indices) = prim.indices.as_ref() else {
                continue;
            };
            if vd.positions.len() < 500 || indices.len() < 3 {
                continue;
            }
            if !is_skirt_prim(asset, mesh.name.as_str(), prim.material_id, vd) {
                continue;
            }
            let mat_name = asset
                .materials
                .iter()
                .find(|m| m.id == prim.material_id)
                .map(|m| m.name.clone())
                .unwrap_or_else(|| mesh.name.clone());
            match build_cloth_for_prim(
                avatar,
                mesh.id,
                mesh.name.clone(),
                prim.id,
                vd,
                indices,
            ) {
                Some(built) => {
                    info!(
                        "auto-cloth: '{}' (prim {:?}, mat '{}'): {} particles, {} pins, {} constraints",
                        mesh.name,
                        prim.id.0,
                        mat_name,
                        built.simulation_mesh.vertices.len(),
                        built.pins.len(),
                        built.constraints.distance_constraints.len()
                    );
                    out.push(built);
                }
                None => {
                    warn!(
                        "auto-cloth: '{}' (prim {:?}) classified as a skirt but no pin node or pins were found; skipping",
                        mesh.name, prim.id.0
                    );
                }
            }
        }
    }
    // Stamp sequential ids so multiple garments coexist.
    out.into_iter()
        .enumerate()
        .map(|(i, mut a)| {
            a.id = ClothOverlayId((first_id + i) as u64);
            a
        })
        .collect()
}

/// Phase-1 skirt classifier (name-based, or a majority of vertices
/// weighted to skirt-named bones), minus the upper-body exclusion list.
fn is_skirt_prim(
    asset: &AvatarAsset,
    mesh_name: &str,
    material_id: crate::asset::MaterialId,
    vd: &crate::asset::VertexData,
) -> bool {
    let m_name = mesh_name.to_lowercase();
    let mat_name = asset
        .materials
        .iter()
        .find(|m| m.id == material_id)
        .map(|m| m.name.to_lowercase())
        .unwrap_or_default();

    let is_upper_body = m_name.contains("shirt")
        || m_name.contains("blouse")
        || m_name.contains("jacket")
        || m_name.contains("sleeve")
        || m_name.contains("arm")
        || m_name.contains("cuff")
        || m_name.contains("collar")
        || m_name.contains("ribbon")
        || m_name.contains("tie")
        || m_name.contains("wing")
        || m_name.contains("acc")
        || m_name.contains("hair")
        || m_name.contains("shoe")
        || m_name.contains("boot")
        || m_name.contains("sock")
        || mat_name.contains("ribbon")
        || mat_name.contains("tie")
        || mat_name.contains("wing")
        || mat_name.contains("shoe");
    if is_upper_body {
        return false;
    }
    let has_skirt_name = m_name.contains("skirt")
        || m_name.contains("bottom")
        || m_name.contains("pants")
        || mat_name.contains("skirt")
        || mat_name.contains("bottom")
        || mat_name.contains("pants");
    if has_skirt_name {
        return true;
    }

    if vd.positions.is_empty() {
        return false;
    }
    let mut skirt_vert_count = 0usize;
    for (vi, joints) in vd.joint_indices.iter().enumerate() {
        let is_skirt_vert = joints.iter().enumerate().any(|(slot, &ji)| {
            if (ji as usize) < asset.skeleton.nodes.len() {
                let node_name = asset.skeleton.nodes[ji as usize].name.to_lowercase();
                node_name.contains("skirt")
                    && vd.joint_weights
                        .get(vi)
                        .map_or(false, |w| w[slot] > 0.1)
            } else {
                false
            }
        });
        if is_skirt_vert {
            skirt_vert_count += 1;
        }
    }
    skirt_vert_count as f32 / vd.positions.len() as f32 >= SKIRT_WEIGHT_RATIO
}

/// One garment: particles, pins, constraints, render binding.
#[allow(clippy::too_many_arguments)]
fn build_cloth_for_prim(
    avatar: &AvatarInstance,
    mesh_id: crate::asset::MeshId,
    mesh_name: String,
    prim_id: crate::asset::PrimitiveId,
    vd: &crate::asset::VertexData,
    indices: &[u32],
) -> Option<ClothAsset> {
    let asset = &avatar.asset;
    let skinning = &avatar.pose.skinning_matrices;

    // Pin node: skirt_root, else hips, else the first skinned joint.
    let pin_node_idx = asset
        .skeleton
        .nodes
        .iter()
        .position(|n| n.name.to_lowercase().contains("skirt_root"))
        .or_else(|| {
            asset
                .skeleton
                .nodes
                .iter()
                .position(|n| n.name.eq_ignore_ascii_case("hips"))
        })
        .or_else(|| {
            vd.joint_indices
                .first()
                .and_then(|j| j.first().map(|&j| j as usize))
                .filter(|&j| j < asset.skeleton.nodes.len())
        })?;
    let pin_node = &asset.skeleton.nodes[pin_node_idx];

    // Rest-frame inverse of the pin node's global transform (rotation
    // rows + translation), for expressing pin offsets in bone space.
    let m = &avatar.pose.global_transforms[pin_node_idx];
    let r0 = [m[0][0], m[1][0], m[2][0]];
    let r1 = [m[0][1], m[1][1], m[2][1]];
    let r2 = [m[0][2], m[1][2], m[2][2]];
    let t = [m[3][0], m[3][1], m[3][2]];

    // Skinned rest positions (4-weight blend, weight-normalized — same
    // recipe as the clearance pass and the diagnose bin).
    let world: Vec<[f32; 3]> = vd
        .positions
        .iter()
        .enumerate()
        .map(|(i, &pos)| {
            let mut w_pos = [0.0f32; 3];
            let mut total_w = 0.0;
            if i < vd.joint_weights.len() && i < vd.joint_indices.len() {
                for k in 0..4 {
                    let w = vd.joint_weights[i][k];
                    if w > 0.0001 {
                        let j = vd.joint_indices[i][k] as usize;
                        if let Some(sm) = skinning.get(j) {
                            for c in 0..3 {
                                w_pos[c] += w
                                    * (sm[0][c] * pos[0] + sm[1][c] * pos[1]
                                        + sm[2][c] * pos[2] + sm[3][c]);
                            }
                            total_w += w;
                        }
                    }
                }
            }
            if total_w > 0.001 {
                for c in 0..3 {
                    w_pos[c] /= total_w;
                }
                w_pos
            } else {
                pos
            }
        })
        .collect();

    // Pin band: the primitive's own top ring.
    let top = world.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
    let pin_y = top - PIN_BAND;

    let mut sim_vertices = Vec::with_capacity(world.len());
    let mut pins = Vec::new();
    for (i, &w_pos) in world.iter().enumerate() {
        let is_pin = w_pos[1] >= pin_y;
        if is_pin {
            let d = [
                w_pos[0] - t[0],
                w_pos[1] - t[1],
                w_pos[2] - t[2],
            ];
            let offset = [
                r0[0] * d[0] + r0[1] * d[1] + r0[2] * d[2],
                r1[0] * d[0] + r1[1] * d[1] + r1[2] * d[2],
                r2[0] * d[0] + r2[1] * d[1] + r2[2] * d[2],
            ];
            pins.push(ClothPin {
                sim_vertex_indices: vec![i as u32],
                binding_node: NodeRef {
                    id: pin_node.id,
                    name: pin_node.name.clone(),
                    humanoid_bone: pin_node.humanoid_bone,
                    parent_path: None,
                },
                offset,
            });
        }
        sim_vertices.push(ClothSimVertex {
            position: w_pos,
            normal: vd.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
            uv: vd.uvs.get(i).copied().unwrap_or([0.0, 0.0]),
            pinned: is_pin,
        });
    }
    if pins.is_empty() {
        return None;
    }

    // Unique undirected edges → distance constraints at rest lengths.
    //
    // The edge set is built over WELDED positions, not raw vertex ids:
    // flat-shaded exports (Yumeka's Circle.056 measures 2,460 verts /
    // 820 disjoint triangles — every vertex used by exactly one
    // triangle) have no shared topology, so naive per-index edges give
    // every particle degree 2 inside its own triangle and the "cloth"
    // free-falls as confetti on BOTH solvers. Welding by quantized
    // position recovers the surface topology; each welded edge then
    // becomes one constraint per duplicate pair so every render vertex
    // (the particle, in this 1:1 sim↔render architecture) carries the
    // structural edges of its welded position.
    let weld = |p: [f32; 3]| -> (i64, i64, i64) {
        const Q: f32 = 2.0e4; // 0.05 mm quantization
        ((p[0] * Q) as i64, (p[1] * Q) as i64, (p[2] * Q) as i64)
    };
    let mut weld_ids: std::collections::HashMap<(i64, i64, i64), u32> = std::collections::HashMap::new();
    let mut welded_of: Vec<u32> = Vec::with_capacity(world.len());
    let mut welded_pos: Vec<[f32; 3]> = Vec::new();
    for &p in &world {
        let key = weld(p);
        let next = welded_pos.len() as u32;
        let id = *weld_ids.entry(key).or_insert(next);
        if id == next {
            welded_pos.push(p);
        }
        welded_of.push(id);
    }
    let welded_count = welded_pos.len();
    let mut dup_lists: Vec<Vec<u32>> = vec![Vec::new(); welded_count];
    for (vi, &w) in welded_of.iter().enumerate() {
        dup_lists[w as usize].push(vi as u32);
    }

    let mut edge_set: HashSet<(u32, u32)> = HashSet::new();
    let mut tri_indices = Vec::with_capacity(indices.len());
    for tri in indices.chunks_exact(3) {
        tri_indices.extend_from_slice(tri);
        let w: Vec<u32> = tri.iter().map(|&vi| welded_of[vi as usize]).collect();
        for pair in [(w[0], w[1]), (w[1], w[2]), (w[2], w[0])] {
            let edge = if pair.0 < pair.1 { pair } else { (pair.1, pair.0) };
            edge_set.insert(edge);
        }
    }
    let mut distance_constraints = Vec::new();
    let mut rest_lengths = Vec::new();
    for &(wa, wb) in &edge_set {
        let pa = welded_pos[wa as usize];
        let pb = welded_pos[wb as usize];
        let d = ((pb[0] - pa[0]).powi(2) + (pb[1] - pa[1]).powi(2) + (pb[2] - pa[2]).powi(2)).sqrt();
        // Round-robin the duplicate lists: every PARTICLE carries each
        // of its welded edges exactly once. A full cross product gives
        // coincident copies of the same correction to one particle and
        // the Jacobi XPBD sum overshoots into an explosion on the
        // first step (measured: ±1e8).
        let us = &dup_lists[wa as usize];
        let vs = &dup_lists[wb as usize];
        for k in 0..us.len().max(vs.len()) {
            let du = us[k % us.len()];
            let dv = vs[k % vs.len()];
            rest_lengths.push(d);
            distance_constraints.push(DistanceConstraint {
                indices: [du, dv],
                rest_length: d,
                stiffness: 0.98,
            });
        }
    }

    let vert_count = sim_vertices.len() as u32;
    let sim_mesh = ClothSimulationMesh {
        vertices: sim_vertices,
        indices: tri_indices,
        rest_lengths,
        attachment_classes: Vec::new(),
        region_tags: vec![ClothRegionTag(1)],
    };

    Some(ClothAsset {
        id: ClothOverlayId(0), // stamped by the caller
        target_avatar: asset.id,
        target_avatar_hash: asset.source_hash.clone(),
        metadata: ClothOverlayMetadata {
            name: format!("Auto: {mesh_name}"),
            format_version: 1,
            created_with: format!("VulVATAR {}", env!("CARGO_PKG_VERSION")),
            last_saved_with: format!("VulVATAR {}", env!("CARGO_PKG_VERSION")),
        },
        stable_refs: ClothStableRefSet {
            node_refs: vec![NodeRef {
                id: pin_node.id,
                name: pin_node.name.clone(),
                humanoid_bone: pin_node.humanoid_bone,
                parent_path: None,
            }],
            mesh_refs: vec![MeshRef {
                id: mesh_id,
                name: mesh_name.clone(),
            }],
            primitive_refs: vec![PrimitiveRef {
                id: prim_id,
                name: "auto_skirt".to_string(),
            }],
        },
        simulation_mesh: sim_mesh,
        render_bindings: vec![ClothRenderRegionBinding {
            primitive: PrimitiveRef {
                id: prim_id,
                name: "auto_skirt".to_string(),
            },
            vertex_subset: VertexSubsetRef {
                offset: 0,
                count: vert_count,
            },
            mapping_region: ClothRegionTag(1),
            mesh: Some(MeshRef {
                id: mesh_id,
                name: mesh_name,
            }),
        }],
        mesh_mapping: ClothMeshMapping {
            mapping_mode: ClothMappingMode::Nearest,
            entries: Vec::new(),
        },
        pins,
        constraints: ClothConstraintSet {
            distance_constraints,
            bend_constraints: Vec::new(),
        },
        collision_bindings: Vec::new(),
        lods: Vec::new(),
        solver_params: ClothSolverParams {
            substeps: 4,
            iterations: 8,
            gravity_scale: 1.0,
            damping: 0.035,
            self_collision: true,
            collision_margin: 0.015,
            wind_response: 0.35,
        },
    })
}

/// Attach auto-generated cloth for every skirt-classified primitive,
/// running on the GPU backend. Returns how many garments were attached.
///
/// The backend is assigned directly on the slot (not via the process-
/// wide request) so the choice is deterministic regardless of what
/// attached earlier in the process lifetime. Wind is a light idle
/// breeze; the asset's response (0.35) keeps it a sway, not a flag.
pub fn attach_auto_cloth(avatar: &mut AvatarInstance) -> usize {
    let first_id = avatar.cloth_overlays.len() + 2;
    avatar.build_base_pose();
    avatar.compute_global_pose();
    avatar.build_skinning_matrices();
    let assets = build_auto_cloth_assets(avatar, first_id);
    let mut attached = 0usize;
    for cloth_asset in assets {
        let slot_idx = avatar.attach_cloth_overlay(cloth_asset.id);
        avatar.init_cloth_overlay(slot_idx, &cloth_asset);
        if let Some(slot) = avatar.cloth_overlays.get_mut(slot_idx) {
            slot.sim.wind_direction = [0.3, 0.0, 0.15];
            slot.sim.wind_response = 0.35;
            slot.sim.self_collision = true;
            slot.sim.self_collision_radius = 0.012;
            slot.state.solver_backend =
                crate::simulation::cloth_gpu_boundary::ClothSolverBackend::Gpu;
        }
        attached += 1;
    }
    if attached > 0 {
        info!("auto-cloth: attached {attached} GPU garment(s)");
    }
    attached
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avatar::{AvatarInstance, AvatarInstanceId};

    fn yumeka_fbx_path() -> Option<String> {
        let pinned = "sample_data/YUMEKA_v1.0.1/FBX/Yumeka_v1.0.fbx";
        if std::path::Path::new(pinned).exists() {
            return Some(pinned.to_string());
        }
        let mut best: Option<(String, String)> = None;
        let Ok(entries) = std::fs::read_dir("sample_data") else {
            return None;
        };
        for entry in entries.flatten() {
            let dir_name = entry.file_name().to_string_lossy().to_string();
            if !dir_name.starts_with("YUMEKA_v") {
                continue;
            }
            let Ok(fbx_dir) = std::fs::read_dir(entry.path().join("FBX")) else {
                continue;
            };
            for f in fbx_dir.flatten() {
                if f.path().extension().and_then(|e| e.to_str()) != Some("fbx") {
                    continue;
                }
                let is_newer = match &best {
                    Some((cur, _)) => dir_name > *cur,
                    None => true,
                };
                if is_newer {
                    if let Some(p) = f.path().to_str() {
                        best = Some((dir_name.clone(), p.to_string()));
                    }
                }
            }
        }
        best.map(|(_, p)| p)
    }

    /// The whole auto-cloth contract on the real sample: the skirt is
    /// detected, the pin ring binds to a skirt/hips node, the attach
    /// helper lands GPU-backed overlays with finite rest positions.
    #[test]
    fn auto_cloth_builds_and_attaches_skirt_on_yumeka() {
        let Some(path) = yumeka_fbx_path() else {
            return;
        };
        let loader = crate::asset::fbx::FbxAssetLoader::new();
        let asset = loader.load(&path).expect("load Yumeka");
        let mut avatar = AvatarInstance::new(AvatarInstanceId(1), asset);
        avatar.build_base_pose();
        avatar.compute_global_pose();
        avatar.build_skinning_matrices();

        let assets = build_auto_cloth_assets(&avatar, 2);
        assert!(!assets.is_empty(), "Yumeka's skirt must classify for auto-cloth");
        for a in &assets {
            assert!(
                a.pins.len() >= 50,
                "expected a waist pin ring, got {} pins",
                a.pins.len()
            );
            assert!(
                a.pins.len() < a.simulation_mesh.vertices.len() / 3,
                "pin band must not swallow the garment ({} of {})",
                a.pins.len(),
                a.simulation_mesh.vertices.len()
            );
            assert!(
                !a.constraints.distance_constraints.is_empty(),
                "edge constraints expected"
            );
            let pin_node = &a.stable_refs.node_refs[0];
            let n = pin_node.name.to_lowercase();
            assert!(
                n.contains("skirt") || n.contains("hips"),
                "pins must bind to the skirt root or hips, got '{}'",
                pin_node.name
            );
            for v in &a.simulation_mesh.vertices {
                assert!(v.position[0].is_finite() && v.position[1].is_finite() && v.position[2].is_finite());
            }
        }

        let attached = attach_auto_cloth(&mut avatar);
        assert_eq!(attached, assets.len());
        assert!(avatar.cloth_enabled);
        for slot in &avatar.cloth_overlays {
            assert!(
                matches!(
                    slot.state.solver_backend,
                    crate::simulation::cloth_gpu_boundary::ClothSolverBackend::Gpu
                ),
                "auto-cloth must attach GPU-backed"
            );
            assert!(slot.state.target_primitive_id.is_some());
            assert!(slot.source_path.is_none(), "auto slots are runtime-only");
        }
    }

    /// The classifier must NOT swallow upper-body garments even when
    /// their names are absent (Yumeka's meshes are all "Circle.xxx" —
    /// the jacket is excluded by its lack of skirt-bone weights).
    #[test]
    fn auto_cloth_classifier_rejects_jacket_mesh() {
        let Some(path) = yumeka_fbx_path() else {
            return;
        };
        let loader = crate::asset::fbx::FbxAssetLoader::new();
        let asset = loader.load(&path).expect("load Yumeka");
        let jacket = asset
            .meshes
            .iter()
            .find(|m| m.name.eq_ignore_ascii_case("circle.051"))
            .and_then(|m| m.primitives.iter().find(|p| p.vertices.is_some()))
            .expect("jacket mesh present");
        let vd = jacket.vertices.as_ref().unwrap();
        assert!(
            !is_skirt_prim(&asset, "Circle.051", jacket.material_id, vd),
            "the jacket must not classify as a skirt"
        );
    }
}
