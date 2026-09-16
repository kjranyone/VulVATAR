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
use std::sync::Arc;

use log::{info, warn};

use crate::asset::{
    AvatarAsset, ClothAsset, ClothConstraintSet, ColliderAsset, ColliderId, ColliderShape,
    ClothMappingMode, ClothMeshMapping, ClothOverlayId, ClothOverlayMetadata, ClothPin,
    ClothRegionTag, ClothRenderRegionBinding, ClothSimVertex, ClothSimulationMesh,
    ClothSolverParams, ClothStableRefSet, DistanceConstraint, HumanoidBone, Mat4, MeshRef, NodeRef,
    PrimitiveRef, VertexSubsetRef,
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
/// Default body-SDF contact band for auto-cloth garments (metres).
/// Measured sweet spot of the `skirt_*_sdfbend` A/B set (2026-09-16).
const AUTO_CLOTH_SDF_CONTACT_M: f32 = 0.004;

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
    // `VULVATAR_AUTO_PIN_FRACTION` overrides the depth fraction for the
    // band A/B (seated drape vs sway buckling); unset keeps 0.4.
    let pin_fraction = std::env::var("VULVATAR_AUTO_PIN_FRACTION")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(PIN_BAND_HEIGHT_FRACTION);
    let top = world.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
    let bottom = world.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
    let pin_band = (pin_fraction * (top - bottom)).clamp(PIN_BAND_MIN, PIN_BAND_MAX);
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

    // Edge-angle bend constraints over the welded topology (see
    // `edge_angle_bend_constraints`): keeps the authored pleat folds
    // and the A-line without deepening the pin band — the anti-buckling
    // job the deep band was doing alone. `VULVATAR_AUTO_NO_BEND=1`
    // disables for A/B.
    let bend_stiffness = std::env::var("VULVATAR_AUTO_BEND_STIFFNESS")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.9);
    let bend_constraints: Vec<crate::asset::BendConstraint> =
        if std::env::var_os("VULVATAR_AUTO_NO_BEND").is_some() {
            Vec::new()
        } else {
            edge_angle_bend_constraints(&weld, indices, bend_stiffness)
        };

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
            bend_constraints,
        },
        collision_bindings: Vec::new(),
        lods: Vec::new(),
        solver_params: ClothSolverParams {
            substeps: 4,
            iterations: std::env::var("VULVATAR_AUTO_ITERATIONS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(32),
            gravity_scale: 1.0,
            // VULVATAR_AUTO_DAMPING / _ITERATIONS: churn A/B knobs.
            // Damping 0.15 + 32 iterations is the churn-campaign
            // outcome (2026-09-16, desk-pose A/B on the GPU backend):
            // the shipped 0.035/8 rang at ~5.3 mm p95 per step forever —
            // a limit cycle that kept the settle gate from ever going
            // quiet — while this pair settles to ~0.22 mm p95 with
            // visually identical drape (skirt_frame_075 A/B). ~0.22 mm
            // still sits above the 100 µm quiet threshold, so live
            // settle behaviour decides whether SETTLE_SLEEP_EPS needs a
            // follow-up raise.
            damping: std::env::var("VULVATAR_AUTO_DAMPING")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.15),
            self_collision: true,
            collision_margin: 0.015,
            // Constant wind on a constrained skirt sustains permanent
            // particle motion — the settle gate can then never go quiet
            // and the GPU cloth never stops (live FPS campaign
            // 2026-09-16). Default OFF; the cloth inspector's wind
            // slider opts back in.
            wind_response: 0.0,
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

/// Edge-angle bend constraints over the welded topology.
///
/// For every interior welded edge — shared by exactly two triangles —
/// the two opposite vertices fold across that edge. This model's
/// contract (T09, see `cloth_solver::constraints` — the hinge is never
/// moved) is the angle AT a vertex, so each shared edge yields TWO
/// constraints, hinging at each endpoint between the two opposite
/// vertices: `(a; c1, c2)` and `(b; c1, c2)`. All three indices use the
/// weld group's FIRST particle as its representative — the hinge is
/// never moved anyway, and the wings' duplicates follow through the
/// intra-weld glue constraints. `from_asset` derives `rest_angle` from
/// the rest positions, so only indices + stiffness travel here.
///
/// Boundary edges (one triangle) and non-manifold seams (3+) produce
/// nothing: there is nothing to fold across, or the direction is
/// ambiguous.
pub fn edge_angle_bend_constraints(
    weld: &WeldGroups,
    indices: &[u32],
    stiffness: f32,
) -> Vec<crate::asset::BendConstraint> {
    use std::collections::HashMap;

    if stiffness <= 0.0 {
        return Vec::new();
    }
    let rep = |g: u32| weld.dup_lists[g as usize][0];

    // Welded edge → opposite vertices.
    let mut opposites: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for tri in indices.chunks_exact(3) {
        let w = [
            weld.weld_of[tri[0] as usize],
            weld.weld_of[tri[1] as usize],
            weld.weld_of[tri[2] as usize],
        ];
        for k in 0..3 {
            let (a, b, opp) = (w[k], w[(k + 1) % 3], w[(k + 2) % 3]);
            let key = if a < b { (a, b) } else { (b, a) };
            opposites.entry(key).or_default().push(opp);
        }
    }

    let mut out = Vec::new();
    for ((a, b), opps) in opposites {
        if opps.len() != 2 {
            continue;
        }
        let (c1, c2) = (opps[0], opps[1]);
        if c1 == c2 {
            continue;
        }
        out.push(crate::asset::BendConstraint {
            indices: [rep(a), rep(c1), rep(c2)],
            stiffness,
        });
        out.push(crate::asset::BendConstraint {
            indices: [rep(b), rep(c1), rep(c2)],
            stiffness,
        });
    }
    out
}

/// Attach auto-generated cloth for every skirt-classified primitive,
/// running on the GPU backend. Returns how many garments were attached.
///
/// The backend is assigned directly on the slot (not via the process-
/// wide request) so the choice is deterministic regardless of what
/// attached earlier in the process lifetime. Wind is a light idle
/// breeze; the asset's response (0.35) keeps it a sway, not a flag.
///
/// Also re-measures the body colliders against the body mesh when
/// `VULVATAR_AUTO_CONFORMAL_COLLIDERS=1` (see
/// [`ensure_body_conformal_cloth_colliders`] — EXPERIMENTAL). The
/// VRChat-derived default keeps the whole skirt floating clear of the
/// body: intact silhouette, but it floats 3–8 cm off the belly/thighs
/// (the front-top "shelf") and ejects deep-overlap particles on the
/// first simulated frames in folded poses. The conformal set hugs the
/// body and fixes both, but on a pleated skirt the hip capsules then
/// push on every sway frame and spread the pleats open (holes at the
/// hips, `diagnostics/skirt_rest_conformal/` + `skirt_rest_capped/` vs
/// the intact `skirt_rest_vrc/`) — shipping that needs the solver-grade
/// follow-ups (SDF-based smooth push + GPU bend), not harder capsules.
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
            // No default wind (see the solver_params note above).
            slot.sim.wind_direction = [0.0, 0.0, 0.0];
            slot.sim.wind_response = 0.0;
            // Self-collision on by default; `VULVATAR_AUTO_NO_SELFCOL=1`
            // disables it and `VULVATAR_AUTO_SELFCOL_RADIUS=<m>` tunes
            // the exclusion radius (12 mm ⇒ 24 mm min spacing — already
            // past this skirt's pleat pitch, see the R6 notes).
            if std::env::var_os("VULVATAR_AUTO_NO_SELFCOL").is_none() {
                slot.sim.self_collision = true;
                if let Some(r) = std::env::var("VULVATAR_AUTO_SELFCOL_RADIUS")
                    .ok()
                    .and_then(|v| v.parse::<f32>().ok())
                {
                    slot.sim.self_collision_radius = r;
                }
            }
            // Body-SDF contact band (metres; 0 = capsules only). The
            // smooth gradient projection engages on garments whenever the
            // app supplies the posed body field. ON by default
            // (2026-09-16): the humanoid TORSO capsules (Spine r 90 mm /
            // Chest r 114 mm on Yumeka) protrude through the skirt front
            // panel in the live desk pose (measured: spine capsule front
            // z +0.119..+0.122 vs skirt panel z +0.106) and domed it into
            // a phallic bulge the stiff bend constraints then held — so
            // upper-body humanoid capsules are masked off for this
            // avatar. Thigh/hips capsules STAY: without them the free
            // front panel sags into the leg gap and the SDF gradient
            // ridge between the thighs shreds it into strips
            // (`diagnostics/skirt_tent_rest_sdfdefault`; the validated
            // `skirt_rest_sdf` A/B ran with the thigh capsules active).
            // Scene colliders (props) stay. Instance-level mask —
            // cloth-only by construction, springs never see
            // `asset.colliders`. `VULVATAR_AUTO_SDF_CONTACT=<m>` retunes
            // the band, `=0` falls back to capsules-only. The app-side
            // splat runs with the spring solver; without it the contact
            // stage is inert (kernel gate `has_sdf != 0 &&
            // sdf_contact > 0`).
            let sdf_contact = match std::env::var("VULVATAR_AUTO_SDF_CONTACT")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
            {
                Some(r) => r.max(0.0),
                None => AUTO_CLOTH_SDF_CONTACT_M,
            };
            if sdf_contact > 0.0 {
                slot.sim.sdf_contact = sdf_contact;
                let humanoid_nodes: std::collections::HashSet<_> = avatar
                    .asset
                    .humanoid
                    .as_ref()
                    .map(|h| h.bone_map.values().copied().collect())
                    .unwrap_or_default();
                for (i, c) in avatar.asset.colliders.iter().enumerate() {
                    let node_name = avatar
                        .asset
                        .skeleton
                        .nodes
                        .get(c.node.0 as usize)
                        .map(|n| n.name.to_lowercase())
                        .unwrap_or_default();
                    let lower_body = node_name.contains("leg")
                        || node_name.contains("thigh")
                        || node_name.contains("hips");
                    if humanoid_nodes.contains(&c.node) && !lower_body {
                        if let Some(flag) = avatar.collider_enabled.get_mut(i) {
                            *flag = false;
                        }
                    }
                }
                info!(
                    "auto-cloth: SDF contact {sdf_contact} m — upper-body capsules masked off for avatar {}",
                    avatar.id.0
                );
            }
            slot.state.solver_backend =
                crate::simulation::cloth_gpu_boundary::ClothSolverBackend::Gpu;
        }
        attached += 1;
    }
    let colliders = if std::env::var_os("VULVATAR_AUTO_CONFORMAL_COLLIDERS").is_some() {
        ensure_body_conformal_cloth_colliders(avatar)
    } else {
        0
    };
    if colliders > 0 {
        info!(
            "auto-cloth: re-measured {colliders} body collider(s) against the body mesh"
        );
    }
    if attached > 0 {
        info!("auto-cloth: attached {attached} GPU garment(s)");
    }
    attached
}

/// Humanoid bone chains the cloth colliders are measured over: the
/// capsule binds to the first bone of the pair, whose local Y points
/// down the chain (true for humanoid skeletons). Ids are reused by
/// node wherever a collider already binds the chain head (the VRChat
/// one on the first pass, ours on re-runs).
const COLLIDER_CHAINS: &[(HumanoidBone, HumanoidBone)] = &[
    (HumanoidBone::Hips, HumanoidBone::Spine), // pelvis
    (HumanoidBone::Spine, HumanoidBone::Chest),
    (HumanoidBone::Chest, HumanoidBone::Neck),
    (HumanoidBone::LeftUpperLeg, HumanoidBone::LeftLowerLeg),
    (HumanoidBone::RightUpperLeg, HumanoidBone::RightLowerLeg),
    (HumanoidBone::LeftLowerLeg, HumanoidBone::LeftFoot), // shin
    (HumanoidBone::RightLowerLeg, HumanoidBone::RightFoot),
    (HumanoidBone::LeftUpperArm, HumanoidBone::LeftLowerArm),
    (HumanoidBone::RightUpperArm, HumanoidBone::RightLowerArm),
    (HumanoidBone::LeftLowerArm, HumanoidBone::LeftHand),
    (HumanoidBone::RightLowerArm, HumanoidBone::RightHand),
];

/// Re-measure the body colliders against the body mesh at attach time.
///
/// The VRChat-derived collider set is authored for dynamic-bone
/// pushback and is far fatter than the mesh (Yumeka: UpperLeg r=74 mm
/// vs ~57 mm measured; Chest r=114 mm). Cloth resolves these every
/// substep, and at rest hundreds of hem particles sit INSIDE the
/// oversized volumes — the first simulated frames eject them onto the
/// capsule surfaces (measured: 93 mm max displacement within 33 ms,
/// `diagnostics/skirt_fit_baseline/metrics.csv`), which scatters the
/// skirt's front panel behind the legs in seated poses.
///
/// This pass replaces each humanoid-bound collider with a capsule (or
/// head sphere) measured from the body primitive's rest surface: the
/// segment runs between the chain's two joints, and the radius is the
/// 85th-percentile distance of nearby body vertices to that segment
/// (p85 ≈ limb radius; the remaining tail is clothing/contact noise).
/// Segments the VRChat set lacks (pelvis, shins) are appended — the
/// pelvis capsule closes the gap between the spine and thigh colliders
/// that the pinned band region previously floated over. Non-humanoid
/// colliders (props) are left untouched. Idempotent: the measurement
/// depends only on the rest mesh and skeleton, never on the current
/// collider list.
///
/// Cloth-only by construction: the spring solver receives scene
/// colliders + the body SDF, never `asset.colliders`.
pub fn ensure_body_conformal_cloth_colliders(avatar: &mut AvatarInstance) -> usize {
    use crate::asset::clearance::compute_rest_world_vertices;

    const CUTOFF: f32 = 0.14;
    const RADIUS_MIN: f32 = 0.018;
    const RADIUS_MAX: f32 = 0.16;
    const HEIGHT_MIN: f32 = 0.02;

    let humanoid = match avatar.asset.humanoid.as_ref() {
        Some(h) => h.bone_map.clone(),
        None => return 0,
    };
    let node_of = |b: HumanoidBone| humanoid.get(&b).map(|id| id.0 as usize);
    let skinning: Vec<Mat4> = avatar.pose.skinning_matrices.clone();

    // Body-mesh rest world vertices for the radius measurement.
    let body_verts: Vec<[f32; 3]> = avatar
        .asset
        .body_primitive_id
        .and_then(|pid| {
            avatar
                .asset
                .meshes
                .iter()
                .flat_map(|m| m.primitives.iter())
                .find(|p| p.id == pid)
        })
        .map(|p| {
            compute_rest_world_vertices(p, &skinning)
                .into_iter()
                .map(|(pos, _)| pos)
                .collect()
        })
        .unwrap_or_default();
    if body_verts.is_empty() {
        return 0;
    }

    // Rest inner envelope of the cloth garments: a capsule fatter than
    // this overlaps the resting skirt and shoves its pleats apart on
    // every sway frame (measured: hip-height tears exposing skin,
    // `diagnostics/skirt_rest_conformal/` vs the intact VRC reference).
    // Each measured radius is capped so the capsule at rest stays just
    // INSIDE the skirt's inner surface; body motion still sweeps the
    // capsule through the cloth exactly as before.
    let skirt_verts: Vec<[f32; 3]> = avatar
        .cloth_overlays
        .iter()
        .filter_map(|s| s.state.target_primitive_id)
        .filter_map(|pid| {
            avatar
                .asset
                .meshes
                .iter()
                .flat_map(|m| m.primitives.iter())
                .find(|p| p.id == pid)
        })
        .flat_map(|p| {
            compute_rest_world_vertices(p, &skinning)
                .into_iter()
                .map(|(pos, _)| pos)
                .collect::<Vec<_>>()
        })
        .collect();

    // Limb-radius measurement: walk K stations down the segment and take
    // the distance to the NEAREST body vertex at each station — the
    // closest surface to a bone axis is the limb's own skin, so the
    // per-station minimum is the local limb radius, and neighbouring
    // parts (the other thigh, the pelvis beside a forearm) never win
    // because they are farther. Radius = median over stations; reject
    // the chain when no surface lies within the cutoff anywhere.
    let measured_radius = |a: [f32; 3], b: [f32; 3]| -> Option<f32> {
        const STATIONS: usize = 8;
        let mut station_min: Vec<f32> = Vec::with_capacity(STATIONS - 1);
        for i in 1..STATIONS {
            let t = i as f32 / STATIONS as f32;
            let s = [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            ];
            let mut min_d2 = f32::MAX;
            for p in &body_verts {
                let d2 = (p[0] - s[0]).powi(2) + (p[1] - s[1]).powi(2) + (p[2] - s[2]).powi(2);
                if d2 < min_d2 {
                    min_d2 = d2;
                }
            }
            if min_d2 > CUTOFF * CUTOFF {
                return None;
            }
            station_min.push(min_d2.sqrt());
        }
        station_min.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut r = station_min[station_min.len() / 2];
        // Cap by the garments' rest inner envelope (see skirt_verts):
        // never fatter than skirt_inner − 5 mm, so the resting skirt is
        // never overlapped (and never pushed by an idle sway).
        if !skirt_verts.is_empty() {
            let mut skirt_min = f32::MAX;
            for p in &skirt_verts {
                let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
                let ab2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
                let t = if ab2 > 1e-12 {
                    ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / ab2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let dx = p[0] - (a[0] + ab[0] * t);
                let dy = p[1] - (a[1] + ab[1] * t);
                let dz = p[2] - (a[2] + ab[2] * t);
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < skirt_min {
                    skirt_min = d2;
                }
            }
            let skirt_min = skirt_min.sqrt();
            if skirt_min > 0.025 {
                r = r.min(skirt_min - 0.005);
            }
        }
        Some(r.clamp(RADIUS_MIN, RADIUS_MAX))
    };

    // World-space joint positions at rest.
    let world_pos = |b: HumanoidBone| -> Option<[f32; 3]> {
        let i = node_of(b)?;
        let m = avatar.pose.global_transforms.get(i)?;
        Some([m[3][0], m[3][1], m[3][2]])
    };
    // World→local of node A at rest (rigid: inverse rotation = transpose).
    let to_local = |a: usize, w: [f32; 3]| -> [f32; 3] {
        let m = &avatar.pose.global_transforms[a];
        let d = [
            w[0] - m[3][0],
            w[1] - m[3][1],
            w[2] - m[3][2],
        ];
        [
            m[0][0] * d[0] + m[0][1] * d[1] + m[0][2] * d[2],
            m[1][0] * d[0] + m[1][1] * d[1] + m[1][2] * d[2],
            m[2][0] * d[0] + m[2][1] * d[1] + m[2][2] * d[2],
        ]
    };

    // Build the measured collider set.
    let mut measured: Vec<ColliderAsset> = Vec::new();
    let mut next_id = avatar
        .asset
        .colliders
        .iter()
        .map(|c| c.id.0)
        .max()
        .unwrap_or(0)
        + 1;
    for &(head, tail) in COLLIDER_CHAINS {
        // Yumeka-class rigs often merge the neck into the head — fall
        // back to the head joint so the chest collider survives.
        let tail = if node_of(tail).is_none() && head == HumanoidBone::Chest {
            HumanoidBone::Head
        } else {
            tail
        };
        let (Some(a_idx), Some(_b_idx)) = (node_of(head), node_of(tail)) else {
            continue;
        };
        let (Some(wa), Some(wb)) = (world_pos(head), world_pos(tail)) else {
            continue;
        };
        let Some(radius) = measured_radius(wa, wb) else {
            continue;
        };
        let height = ((wb[0] - wa[0]).powi(2)
            + (wb[1] - wa[1]).powi(2)
            + (wb[2] - wa[2]).powi(2))
        .sqrt()
        .max(HEIGHT_MIN);
        let mid_world = [
            (wa[0] + wb[0]) * 0.5,
            (wa[1] + wb[1]) * 0.5,
            (wa[2] + wb[2]) * 0.5,
        ];
        let offset = to_local(a_idx, mid_world);
        let node = avatar.asset.skeleton.nodes[a_idx].id;
        // Reuse the id of whatever collider already binds this node (a
        // VRChat one on the first run, our own on re-runs — the pass
        // must be idempotent), else mint a fresh one.
        let id = avatar
            .asset
            .colliders
            .iter()
            .find(|c| c.node == node)
            .map(|c| c.id)
            .unwrap_or_else(|| {
                let id = ColliderId(next_id);
                next_id += 1;
                id
            });
        measured.push(ColliderAsset {
            id,
            node,
            shape: ColliderShape::Capsule { radius, height },
            offset,
        });
    }
    // Head sphere (the VRChat set has one; keep the coverage).
    if let (Some(h_idx), Some(hw)) = (node_of(HumanoidBone::Head), world_pos(HumanoidBone::Head))
    {
        let node = avatar.asset.skeleton.nodes[h_idx].id;
        if !measured.iter().any(|c| c.node == node) {
            if let Some(radius) = measured_radius(hw, hw) {
                let id = avatar
                    .asset
                    .colliders
                    .iter()
                    .find(|c| c.node == node)
                    .map(|c| c.id)
                    .unwrap_or_else(|| {
                        let id = ColliderId(next_id);
                        next_id += 1;
                        id
                    });
                measured.push(ColliderAsset {
                    id,
                    node,
                    shape: ColliderShape::Sphere { radius },
                    offset: [0.0, 0.0, 0.0],
                });
            }
        }
    }

    if measured.is_empty() {
        return 0;
    }

    // CoW swap: keep any non-humanoid colliders (props) the asset
    // carried, drop the replaced body ones, extend the enable mask for
    // appended entries.
    let humanoid_nodes: std::collections::HashSet<_> = humanoid.values().copied().collect();
    let asset = Arc::make_mut(&mut avatar.asset);
    let mut kept: Vec<ColliderAsset> = asset
        .colliders
        .iter()
        .filter(|c| !humanoid_nodes.contains(&c.node))
        .cloned()
        .collect();
    kept.extend(measured);
    let new_len = kept.len();
    asset.colliders = kept;
    if avatar.collider_enabled.len() < new_len {
        avatar.collider_enabled.resize(new_len, true);
    }
    new_len
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

        // The simulated garment must NOT keep its load-time clearance
        // anchors: the solver owns its body interaction, render-only
        // anchors fight the simulated state (`init_cloth_overlay`
        // strips them; measured p95 15–18 mm drawn-vs-simulated skew,
        // diagnostics/cloth_vboaudit_20260915).
        for slot in &avatar.cloth_overlays {
            let Some(pid) = slot.state.target_primitive_id else {
                continue;
            };
            let prim = avatar
                .asset
                .meshes
                .iter()
                .flat_map(|m| m.primitives.iter())
                .find(|p| p.id == pid)
                .expect("cloth target prim present");
            assert!(
                prim.skin_anchors.is_none(),
                "simulated garment must not carry render-side clearance anchors"
            );
        }
    }

    /// The conformal-collider re-measure (experimental,
    /// `VULVATAR_AUTO_CONFORMAL_COLLIDERS=1`): every humanoid-bound
    /// collider gets a mesh-measured radius, and the pelvis capsule the
    /// VRChat set lacks is appended. Radii must be body-conformal —
    /// thinner than the VRChat thigh capsule (74 mm) — and finite.
    #[test]
    fn auto_cloth_conformal_colliders_measure_body() {
        let Some(path) = yumeka_fbx_path() else {
            return;
        };
        let loader = crate::asset::fbx::FbxAssetLoader::new();
        let asset = loader.load(&path).expect("load Yumeka");
        let mut avatar = AvatarInstance::new(AvatarInstanceId(1), asset);
        avatar.build_base_pose();
        avatar.compute_global_pose();
        avatar.build_skinning_matrices();

        let before = avatar.asset.colliders.len();
        let n = ensure_body_conformal_cloth_colliders(&mut avatar);
        assert!(n > 0, "humanoid colliders must be re-measured");
        assert!(n >= before, "appended {n} of {before}");

        let humanoid = &avatar.asset.humanoid.as_ref().expect("humanoid map").bone_map;
        let hips_node = humanoid.get(&HumanoidBone::Hips).expect("hips node").0;
        assert!(
            avatar.asset.colliders.iter().any(|c| c.node.0 == hips_node),
            "pelvis capsule must exist (the VRChat set has none)"
        );
        for c in &avatar.asset.colliders {
            match c.shape {
                ColliderShape::Sphere { radius } | ColliderShape::Capsule { radius, .. } => {
                    assert!(
                        radius.is_finite() && (0.015..=0.2).contains(&radius),
                        "measured radius out of range: {radius}"
                    );
                }
            }
        }
        // Thigh capsules must be thinner than the VRChat 74 mm pushback
        // spheres — the whole point of the re-measure.
        for bone in [HumanoidBone::LeftUpperLeg, HumanoidBone::RightUpperLeg] {
            let node = humanoid.get(&bone).expect("upper leg node").0;
            let col = avatar
                .asset
                .colliders
                .iter()
                .find(|c| c.node.0 == node)
                .expect("thigh collider");
            let r = match col.shape {
                ColliderShape::Capsule { radius, .. } => radius,
                ColliderShape::Sphere { radius } => radius,
            };
            assert!(
                r < 0.074,
                "measured thigh radius {r} must be thinner than the VRChat 74 mm"
            );
        }
    }

    /// The conformal re-measure must be idempotent — running it twice
    /// (re-attach scenarios) keeps the same collider count and radii,
    /// because the measurement depends only on the rest mesh.
    #[test]
    fn auto_cloth_conformal_colliders_are_idempotent() {
        let Some(path) = yumeka_fbx_path() else {
            return;
        };
        let loader = crate::asset::fbx::FbxAssetLoader::new();
        let asset = loader.load(&path).expect("load Yumeka");
        let mut avatar = AvatarInstance::new(AvatarInstanceId(1), asset);
        avatar.build_base_pose();
        avatar.compute_global_pose();
        avatar.build_skinning_matrices();

        ensure_body_conformal_cloth_colliders(&mut avatar);
        let snapshot: Vec<_> = avatar
            .asset
            .colliders
            .iter()
            .map(|c| (c.id, c.node, c.shape.clone(), c.offset))
            .collect();
        ensure_body_conformal_cloth_colliders(&mut avatar);
        let after: Vec<_> = avatar
            .asset
            .colliders
            .iter()
            .map(|c| (c.id, c.node, c.shape.clone(), c.offset))
            .collect();
        assert_eq!(snapshot.len(), after.len());
        for (a, b) in snapshot.iter().zip(after.iter()) {
            assert_eq!(a.0, b.0, "collider ids must be stable");
            assert_eq!(a.1, b.1, "collider nodes must be stable");
            assert_eq!(a.3, b.3, "collider offsets must be stable");
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

#[cfg(test)]
mod bend_generation_tests {
    use super::*;
    use crate::avatar::{AvatarInstance, AvatarInstanceId};

    // The sibling `tests` module's helper is private to it — a local
    // copy for the integration case below.
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

    // 1 cm quad corners (local copies — the R6 test module's consts are
    // private to it).
    const Q0: [f32; 3] = [0.0, 0.0, 0.0];
    const Q1: [f32; 3] = [0.01, 0.0, 0.0];
    const Q2: [f32; 3] = [0.0, 0.01, 0.0];
    const Q3: [f32; 3] = [0.01, 0.01, 0.0];

    /// Two triangles sharing one welded edge → exactly TWO edge-angle
    /// constraints (hinge at each endpoint, wings = the two opposite
    /// vertices); boundary edges produce nothing.
    #[test]
    fn bend_generation_hinges_each_endpoint_of_interior_edges() {
        // Welded quad: P0-P1 shared edge, wings P2 and P3.
        // tri1 = (P0,P1,P2), tri2 = (P1,P0,P3) — reversed winding so
        // the shared edge appears from both sides.
        let world = vec![Q0, Q1, Q2, Q3];
        let weld = weld_by_quantized_position(&world);
        let indices: Vec<u32> = vec![0, 1, 2, 1, 0, 3];

        let bends = edge_angle_bend_constraints(&weld, &indices, 0.9);
        assert_eq!(bends.len(), 2, "one interior edge → two hinges");

        let groups: Vec<[u32; 3]> = bends
            .iter()
            .map(|b| [
                weld.weld_of[b.indices[0] as usize],
                weld.weld_of[b.indices[1] as usize],
                weld.weld_of[b.indices[2] as usize],
            ])
            .collect();
        // Each hinge weld is an endpoint of the shared edge (groups 0
        // or 1), with wings = the two opposite welds (2 and 3).
        for g in &groups {
            assert!(
                (g[0] == 0 || g[0] == 1) && g[1] != g[2],
                "hinge on a shared-edge endpoint, distinct wings: {g:?}"
            );
            assert!(
                [g[1], g[2]].iter().all(|w| *w == 2 || *w == 3),
                "wings are the two opposite vertices: {g:?}"
            );
        }
        // Both endpoints hinge, i.e. the two constraints differ in p0.
        assert_ne!(groups[0][0], groups[1][0]);
        for b in &bends {
            assert_eq!(b.stiffness, 0.9);
        }
    }

    /// A lone triangle (all boundary edges) generates nothing.
    #[test]
    fn bend_generation_ignores_boundary_edges() {
        let world = vec![Q0, Q1, Q2];
        let weld = weld_by_quantized_position(&world);
        let bends = edge_angle_bend_constraints(&weld, &[0, 1, 2], 0.9);
        assert!(bends.is_empty());
    }

    /// Yumeka integration: the auto skirt carries a healthy bend set —
    /// one in each direction around every interior welded edge, never
    /// exceeding two per edge.
    #[test]
    fn auto_cloth_yumeka_skirt_generates_bend_constraints() {
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
            assert!(
                !a.constraints.bend_constraints.is_empty(),
                "pleated skirt must generate bend constraints"
            );
            for bc in &a.constraints.bend_constraints {
                assert!(
                    (bc.indices[0] < a.simulation_mesh.vertices.len() as u32)
                        && (bc.indices[1] < a.simulation_mesh.vertices.len() as u32)
                        && (bc.indices[2] < a.simulation_mesh.vertices.len() as u32),
                    "bend indices in particle range"
                );
            }
        }
    }
}
