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

/// How far below the primitive's top the pin band reaches. A DEEP pin
/// band is the anti-buckling contract for distance-only XPBD (zero
/// bending stiffness): a shallow 3.5 cm band left most of the skirt as
/// free cloth, and hip sway buckled it at the sides — the hem folded up
/// to the waist exposing the hips/butt (measured, `diagnostics/
/// pinband_ab.png`). Pinning ~40% of the garment height keeps the
/// authored A-line while the free lower half still sways. Scaled by the
/// garment's own height and clamped so small and long garments stay
/// sane.
const PIN_BAND_MIN: f32 = 0.035;
const PIN_BAND_MAX: f32 = 0.12;
const PIN_BAND_HEIGHT_FRACTION: f32 = 0.4;
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

    // Pin band: the primitive's own top ring, deepened by the garment's
    // own height (see the PIN_BAND contract above).
    let top = world.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
    let bottom = world.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
    let pin_band = (PIN_BAND_HEIGHT_FRACTION * (top - bottom))
        .clamp(PIN_BAND_MIN, PIN_BAND_MAX);
    let pin_y = top - pin_band;

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
    let weld = weld_by_quantized_position(&world);

    let mut edge_set: HashSet<(u32, u32)> = HashSet::new();
    let mut tri_indices = Vec::with_capacity(indices.len());
    for tri in indices.chunks_exact(3) {
        tri_indices.extend_from_slice(tri);
        let w: Vec<u32> = tri.iter().map(|&vi| weld.weld_of[vi as usize]).collect();
        for pair in [(w[0], w[1]), (w[1], w[2]), (w[2], w[0])] {
            let edge = if pair.0 < pair.1 { pair } else { (pair.1, pair.0) };
            edge_set.insert(edge);
        }
    }
    let (mut distance_constraints, mut rest_lengths) =
        round_robin_weld_constraints(&weld, &edge_set, 0.98);
    // Hold each weld group's copies at their shared position (R6) —
    // without these the copies drift apart and self-collision tears
    // the seams apart (see `intra_weld_group_constraints` doc).
    let (intra, intra_rest) = intra_weld_group_constraints(&weld, 0.98);
    distance_constraints.extend(intra);
    rest_lengths.extend(intra_rest);

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

/// Quantized-position weld of skinned rest positions, extracted for the
/// R6 audit: THIS is the topology-recovery contract auto-cloth rests on.
///
/// Known limits (documented by `weld_merges_coincident_layers`, and the
/// reason R6 exists): the weld groups purely by POSITION, so
/// (a) UV/seam split copies of one surface vertex are welded into one
/// group and held together only by round-robin distance constraints —
/// there is no shared-particle guarantee, so a seam copy CAN drift away
/// from its group under asymmetric constraint/collision loads; and
/// (b) two truly separate garment layers resting at identical rest
/// positions merge into the same groups. The structural fix — separate
/// simulation particles from render vertices with explicit
/// shared-particle seams — is the Phase C design backlog; do not paper
/// over either case with stiffer springs.
pub struct WeldGroups {
    /// Original vertex index → weld group id.
    pub weld_of: Vec<u32>,
    /// One representative position per group (first occurrence).
    pub group_pos: Vec<[f32; 3]>,
    /// Per group, the original vertex indices resting at that position.
    pub dup_lists: Vec<Vec<u32>>,
}

/// Build weld groups from world-space rest positions (0.05 mm
/// quantization — tolerance is sub-millimetre so distinct garment
/// detail must not merge).
pub fn weld_by_quantized_position(world: &[[f32; 3]]) -> WeldGroups {
    let weld = |p: [f32; 3]| -> (i64, i64, i64) {
        const Q: f32 = 2.0e4; // 0.05 mm quantization
        ((p[0] * Q) as i64, (p[1] * Q) as i64, (p[2] * Q) as i64)
    };
    let mut weld_ids: std::collections::HashMap<(i64, i64, i64), u32> =
        std::collections::HashMap::new();
    let mut weld_of: Vec<u32> = Vec::with_capacity(world.len());
    let mut group_pos: Vec<[f32; 3]> = Vec::new();
    for &p in world {
        let key = weld(p);
        let next = group_pos.len() as u32;
        let id = *weld_ids.entry(key).or_insert(next);
        if id == next {
            group_pos.push(p);
        }
        weld_of.push(id);
    }
    let mut dup_lists: Vec<Vec<u32>> = vec![Vec::new(); group_pos.len()];
    for (vi, &w) in weld_of.iter().enumerate() {
        dup_lists[w as usize].push(vi as u32);
    }
    WeldGroups {
        weld_of,
        group_pos,
        dup_lists,
    }
}

/// Distance constraints for a welded edge set: round-robin the duplicate
/// lists so every PARTICLE carries each of its welded edges (once per
/// opposite copy when the opposite group has more copies). A full cross
/// product of the two duplicate lists would give coincident copies of
/// the same correction to one particle and the Jacobi XPBD sum
/// overshoots into an explosion on the first step (measured: ±1e8).
/// Returns `(constraints, rest_lengths)` in matching order.
pub fn round_robin_weld_constraints(
    weld: &WeldGroups,
    edges: &HashSet<(u32, u32)>,
    stiffness: f32,
) -> (Vec<DistanceConstraint>, Vec<f32>) {
    let mut distance_constraints = Vec::new();
    let mut rest_lengths = Vec::new();
    for &(wa, wb) in edges {
        let pa = weld.group_pos[wa as usize];
        let pb = weld.group_pos[wb as usize];
        let d = ((pb[0] - pa[0]).powi(2) + (pb[1] - pa[1]).powi(2) + (pb[2] - pa[2]).powi(2))
            .sqrt();
        let us = &weld.dup_lists[wa as usize];
        let vs = &weld.dup_lists[wb as usize];
        for k in 0..us.len().max(vs.len()) {
            let du = us[k % us.len()];
            let dv = vs[k % vs.len()];
            rest_lengths.push(d);
            distance_constraints.push(DistanceConstraint {
                indices: [du, dv],
                rest_length: d,
                stiffness,
            });
        }
    }
    (distance_constraints, rest_lengths)
}

/// Intra-weld-group constraints: one `rest_length = 0` distance
/// constraint per pair of particles resting at the same weld position.
///
/// R6 contract fix (measured on the GPU smoke): position welding alone
/// gives the copies of one group NOTHING that holds them together —
/// every distance constraint spans two DIFFERENT groups, so the copies
/// drift apart under asymmetric Jacobi averaging / collision loads, and
/// the moment their spacing exceeds the self-collision coincident
/// epsilon (0.5 mm) the self-collision pass reads them as penetrations
/// and blasts the seam apart into surface-wide spikes (both solvers,
/// `diagnostics/cloth_gpu_every_frame/` vs `cloth_gpu_noselfcol/`).
/// A zero-rest constraint per copy pair is the 1:1 sim↔render
/// architecture's version of "physically continuous vertices share a
/// particle": it re-glues drifted copies, and because
/// `connected_pairs` / the GPU selfcol CSR are built from the FULL
/// constraint list, it also excludes group-internal pairs from
/// self-collision. The XPBD zero-length guard (1 nm) makes exact
/// coincident pairs a no-op, so the constraint only spends effort when
/// a copy has actually drifted.
pub fn intra_weld_group_constraints(
    weld: &WeldGroups,
    stiffness: f32,
) -> (Vec<DistanceConstraint>, Vec<f32>) {
    let mut constraints = Vec::new();
    let mut rest_lengths = Vec::new();
    for dups in &weld.dup_lists {
        if dups.len() < 2 {
            continue;
        }
        for a in 0..dups.len() {
            for b in (a + 1)..dups.len() {
                constraints.push(DistanceConstraint {
                    indices: [dups[a], dups[b]],
                    rest_length: 0.0,
                    stiffness,
                });
                rest_lengths.push(0.0);
            }
        }
    }
    (constraints, rest_lengths)
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
    use std::collections::HashMap;

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
            // The deep anti-buckling pin band (40% of garment height,
            // clamped) legitimately pins a larger share than the old
            // 3.5 cm band; the hard bound only guards against the band
            // swallowing the whole garment.
            assert!(
                a.pins.len() * 2 < a.simulation_mesh.vertices.len(),
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

    // ---- R6 weld-contract tests --------------------------------------

    /// Quad corners on a 1 cm grid.
    const P0: [f32; 3] = [0.0, 0.0, 0.0];
    const P1: [f32; 3] = [0.01, 0.0, 0.0];
    const P2: [f32; 3] = [0.0, 0.01, 0.0];
    const P3: [f32; 3] = [0.01, 0.01, 0.0];

    /// A split-vertex quad: two triangles with per-corner duplicates
    /// (flat-shaded export style). The weld must recover the 4 shared
    /// corners, and the round-robin constraints must connect the two
    /// copies of each shared corner exactly once — no duplicate
    /// corrections, no particle left unwelded.
    #[test]
    fn weld_reconnects_split_vertex_quad() {
        // tri1 (P0,P1,P2), tri2 (P1,P3,P2) → indices 0..6 with copies
        // of P1 (indices 1, 3) and P2 (indices 2, 5).
        let world = vec![P0, P1, P2, P1, P3, P2];
        let weld = weld_by_quantized_position(&world);

        assert_eq!(weld.group_pos.len(), 4, "6 corner copies → 4 weld groups");
        assert_eq!(weld.dup_lists.iter().map(Vec::len).collect::<Vec<_>>(), vec![1, 2, 2, 1]);

        // Welded edge set of the two triangles: quad rim + diagonal.
        let mut edges: HashSet<(u32, u32)> = HashSet::new();
        for tri in [[0u32, 1, 2], [3, 4, 5]] {
            let w: Vec<u32> = tri.iter().map(|&vi| weld.weld_of[vi as usize]).collect();
            for pair in [(w[0], w[1]), (w[1], w[2]), (w[2], w[0])] {
                edges.insert(if pair.0 < pair.1 { pair } else { (pair.1, pair.0) });
            }
        }
        assert_eq!(edges.len(), 5, "quad rim (4) + diagonal (1)");

        let (constraints, rest_lengths) = round_robin_weld_constraints(&weld, &edges, 0.98);
        assert_eq!(constraints.len(), rest_lengths.len());
        // Every constraint is finite and (being from the 1 cm quad) short.
        for (c, d) in constraints.iter().zip(&rest_lengths) {
            assert!((*d > 0.0) && d.is_finite());
            assert_eq!(c.rest_length, *d);
            assert!(c.indices[0] != c.indices[1], "degenerate self-constraint");
        }

        // Degree contract: each welded edge yields exactly
        // max(|dup_a|, |dup_b|) constraints, and every particle's
        // constraint degree is AT LEAST its group's welded-edge degree
        // (connectivity guarantee). A singleton corner against a
        // multi-copy group carries the edge once per opposite copy —
        // e.g. this quad's rim corner P0 (group 0, degree 2) ends up
        // with 4 constraints, one per copy of P1 and P2, because
        // dropping any would strand that opposite copy from P0's
        // particle.
        let degree_of_group: HashMap<u32, usize> = {
            let mut m: HashMap<u32, usize> = HashMap::new();
            for &(a, b) in &edges {
                *m.entry(a).or_default() += 1;
                *m.entry(b).or_default() += 1;
            }
            m
        };
        for (gi, dups) in weld.dup_lists.iter().enumerate() {
            let want = degree_of_group.get(&(gi as u32)).copied().unwrap_or(0);
            for &vi in dups {
                let got = constraints
                    .iter()
                    .filter(|c| c.indices[0] == vi || c.indices[1] == vi)
                    .count();
                assert!(
                    got >= want,
                    "particle {} (group {}) under-connected: {} < {}",
                    vi,
                    gi,
                    got,
                    want
                );
            }
        }
        // Per-edge constraint count is exactly the round-robin max.
        let mut per_edge: HashMap<(u32, u32), usize> = HashMap::new();
        for c in &constraints {
            let (g0, g1) = (
                weld.weld_of[c.indices[0] as usize],
                weld.weld_of[c.indices[1] as usize],
            );
            let key = if g0 < g1 { (g0, g1) } else { (g1, g0) };
            *per_edge.entry(key).or_default() += 1;
        }
        for &(a, b) in &edges {
            let want = weld.dup_lists[a as usize]
                .len()
                .max(weld.dup_lists[b as usize].len());
            assert_eq!(
                per_edge.get(&(a, b)).copied(),
                Some(want),
                "welded edge ({a},{b}) constraint count"
            );
        }
    }

    /// The R6 RISK, pinned as observable behaviour: two garment layers
    /// resting at IDENTICAL positions merge into the same weld groups —
    /// position-only welding cannot distinguish layers. Whether their
    /// distance constraints couple the layers then depends purely on
    /// VERTEX ORDER: with both layers enumerating their copies in
    /// matching order the round-robin pairs copy k with copy k (no
    /// coupling); reorder one layer and the modular wrap pairs across
    /// layers. This order-fragility is why true layer separation needs
    /// the Phase C sim-particle/render-vertex split, not stiffer
    /// springs.
    #[test]
    fn weld_merges_coincident_layers_documents_the_risk() {
        // Layer A: tri (P0,P1,P2). Layer B: the same tri, same positions.
        let layer_a = vec![P0, P1, P2];

        // Matching enumeration order → copy k pairs with copy k: the
        // layers weld into shared groups but stay constraint-
        // independent (accidentally benign).
        let mut world = layer_a.clone();
        world.extend_from_slice(&layer_a);
        let weld = weld_by_quantized_position(&world);

        assert_eq!(
            weld.group_pos.len(),
            3,
            "coincident layers merge into shared weld groups"
        );
        assert!(weld.dup_lists.iter().all(|d| d.len() == 2));

        let mut edges: HashSet<(u32, u32)> = HashSet::new();
        for tri in [[0u32, 1, 2], [3, 4, 5]] {
            let w: Vec<u32> = tri.iter().map(|&vi| weld.weld_of[vi as usize]).collect();
            for pair in [(w[0], w[1]), (w[1], w[2]), (w[2], w[0])] {
                edges.insert(if pair.0 < pair.1 { pair } else { (pair.1, pair.0) });
            }
        }
        let (constraints, _) = round_robin_weld_constraints(&weld, &edges, 0.98);
        assert_eq!(constraints.len(), 6, "3 welded edges × max(2,2) copies");
        let cross_layer = constraints
            .iter()
            .any(|c| (c.indices[0] < 3) != (c.indices[1] < 3));
        assert!(
            !cross_layer,
            "matching copy order pairs copy k with copy k within each layer"
        );

        // Interleave the two layers so one layer's copy is NOT at the
        // same duplicate-list position in every group: layer A =
        // indices {0, 2, 5}, layer B = {1, 3, 4} — in group 2 the B
        // copy enumerates first, so A's duplicate-list position
        // differs between groups. The round-robin then pairs
        // list-position k with list-position k ACROSS layers — the
        // layers become structurally coupled purely because of vertex
        // ORDER.
        let world = vec![P0, P0, P1, P1, P2, P2]; // A,B, A,B, B,A
        let layer_a: HashSet<usize> = [0usize, 2, 5].into_iter().collect();
        let weld = weld_by_quantized_position(&world);
        // Sanity: tri A = (0,2,5), tri B = (1,3,4) weld to the same
        // three groups.
        let mut edges: HashSet<(u32, u32)> = HashSet::new();
        for tri in [[0u32, 2, 5], [1, 3, 4]] {
            let w: Vec<u32> = tri.iter().map(|&vi| weld.weld_of[vi as usize]).collect();
            for pair in [(w[0], w[1]), (w[1], w[2]), (w[2], w[0])] {
                edges.insert(if pair.0 < pair.1 { pair } else { (pair.1, pair.0) });
            }
        }
        assert_eq!(edges.len(), 3);
        let (constraints, _) = round_robin_weld_constraints(&weld, &edges, 0.98);
        let cross_layer = constraints
            .iter()
            .any(|c| {
                let a_is_a = layer_a.contains(&(c.indices[0] as usize));
                let b_is_a = layer_a.contains(&(c.indices[1] as usize));
                a_is_a != b_is_a
            });
        assert!(
            cross_layer,
            "position-mismatched enumeration couples the layers through round-robin"
        );
    }

    /// The intra-group constraints must connect EVERY pair of copies
    /// within each weld group at rest length 0 — that is the R6 fix
    /// that keeps seam copies coincident and excludes them from
    /// self-collision (via connected_pairs / the GPU CSR).
    #[test]
    fn intra_weld_group_constraints_fully_connect_groups() {
        let world = vec![P0, P1, P2, P0, P1, P2, P0]; // group sizes 3/2/2
        let weld = weld_by_quantized_position(&world);
        assert_eq!(weld.group_pos.len(), 3);
        assert_eq!(
            weld.dup_lists.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![3, 2, 2]
        );

        let (constraints, rest_lengths) = intra_weld_group_constraints(&weld, 0.98);
        // C(3,2) + C(2,2) + C(2,2) = 3 + 1 + 1.
        assert_eq!(constraints.len(), 5);
        assert!(rest_lengths.iter().all(|r| *r == 0.0));
        for (gi, dups) in weld.dup_lists.iter().enumerate() {
            for a in 0..dups.len() {
                for b in (a + 1)..dups.len() {
                    let (lo, hi) = (dups[a].min(dups[b]), dups[a].max(dups[b]));
                    assert!(
                        constraints
                            .iter()
                            .any(|c| c.indices == [lo, hi] && c.rest_length == 0.0),
                        "group {} pair ({},{}) missing",
                        gi,
                        lo,
                        hi
                    );
                }
            }
        }
    }

    /// Whole-asset R6 invariant on the real sample: every weld group's
    /// particles rest at the SAME position (max intra-group spread
    /// under the 0.05 mm quantum), and merging actually happened for
    /// the flat-shaded skirt (groups ≪ vertices).
    #[test]
    fn auto_cloth_weld_groups_hold_identical_positions_on_yumeka() {
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
        assert!(!assets.is_empty());
        for a in &assets {
            // Recompute the skinned rest positions with the same LBS
            // recipe the builder used, then re-run the weld on them.
            let prim = avatar
                .asset
                .meshes
                .iter()
                .flat_map(|m| m.primitives.iter())
                .find(|p| p.vertices.is_some() && p.id.0 == a.render_bindings[0].primitive.id.0)
                .expect("binding primitive present");
        let pvd = prim.vertices.as_ref().unwrap();
        let skinning = &avatar.pose.skinning_matrices;
        let world: Vec<[f32; 3]> = pvd
            .positions
            .iter()
            .enumerate()
            .map(|(i, &pos)| {
                let mut w_pos = [0.0f32; 3];
                let mut total_w = 0.0;
                if i < pvd.joint_weights.len() && i < pvd.joint_indices.len() {
                    for k in 0..4 {
                        let w = pvd.joint_weights[i][k];
                        if w > 0.0001 {
                            let j = pvd.joint_indices[i][k] as usize;
                            if let Some(sm) = skinning.get(j) {
                                for c in 0..3 {
                                    w_pos[c] += w
                                        * (sm[0][c] * pos[0]
                                            + sm[1][c] * pos[1]
                                            + sm[2][c] * pos[2]
                                            + sm[3][c]);
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
                }
                w_pos
            })
            .collect();
            let weld = weld_by_quantized_position(&world);
            assert!(
                weld.group_pos.len() < world.len(),
                "flat-shaded garment must merge ({} groups of {} verts)",
                weld.group_pos.len(),
                world.len()
            );
            // Particle↔group position identity: the weld tolerance is the
            // 0.05 mm quantum; any drift larger than that between a
            // particle and its group position means the quantization
            // collapsed distinct positions.
            for (vi, &g) in weld.weld_of.iter().enumerate() {
                let p = world[vi];
                let q = weld.group_pos[g as usize];
                let spread = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2))
                    .sqrt();
                assert!(
                    spread < 1.0e-4,
                    "particle {} sits {} m from its weld group",
                    vi,
                    spread
                );
            }
            // Constraint sanity: finite rest lengths, no self-loops.
            for c in &a.constraints.distance_constraints {
                assert!(c.rest_length.is_finite() && c.rest_length >= 0.0);
                assert_ne!(c.indices[0], c.indices[1]);
            }
        }
    }
}
