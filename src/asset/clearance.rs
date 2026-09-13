//! Skin-Anchor Clearance Field
//!
//! Provides geometric anti-penetration constraints between clothing/skirts
//! and the avatar's underlying body mesh, and between layered garments.
//!
//! At asset import time:
//! 1. Identifies the avatar's primary body mesh/primitive (e.g. skin, torso + legs).
//! 2. Evaluates the rest-pose world positions of both body and clothing primitives.
//! 3. For each vertex in a clothing primitive within proximity of the body, locates
//!    the nearest surface vertex on the body mesh with compatible outward normals.
//! 4. Computes the rest clearance `c = dot(V_cloth - V_body, N_body)` and records a
//!    [`SkinAnchor`].
//! 5. Layered garment pairs (shirt under blazer, …) get anchors in both
//!    directions: clearance anchors on the outer layer and containment
//!    anchors on the inner layer.
//!
//! At runtime:
//! The GPU compute shader (`transform_cs`) applies each anchor according
//! to its mode. Clearance anchors (skirts over legs, outer garments over
//! inner ones) project the anchored vertex outward along the parent
//! normal until it is at least `min_clearance` away. Containment anchors
//! (inner garments) clamp the anchored vertex back inside the parent
//! surface when its clearance along the parent normal exceeds
//! `min_clearance`. Together the two modes keep the outer surface
//! outside the inner one at bent joints.

use std::collections::HashMap;
use std::sync::Arc;

use log::{debug, info, warn};
use serde::{Deserialize, Serialize};

use crate::asset::{AvatarAsset, Mat4, MeshId, MeshPrimitiveAsset, PrimitiveId};

/// Anchor mode: push this vertex OUTWARD off the parent surface until it
/// is at least `min_clearance` along the parent vertex's outward normal.
/// Used by outer garments (skirt over legs, blazer over shirt).
pub const SKIN_ANCHOR_CLEARANCE: u32 = 0;

/// Anchor mode: clamp this vertex back INSIDE the parent surface when
/// its clearance along the parent vertex's outward normal exceeds
/// `min_clearance` (typically rest-derived and negative). Used by inner
/// garments (shirt under blazer) — the outer-surface clearance pass
/// alone cannot stop an inner vertex sliding between two anchored
/// outer vertices and poking through at bent joints.
pub const SKIN_ANCHOR_CONTAINMENT: u32 = 1;

/// Per-vertex skin anchor constraint.
///
/// std430 alignment: 16 bytes (`uvec4`-aligned).
#[derive(
    Clone, Copy, Debug, PartialEq, Serialize, Deserialize, bytemuck::Pod, bytemuck::Zeroable,
)]
#[repr(C)]
pub struct SkinAnchor {
    /// Index of the nearest body vertex in the body primitive's vertex buffer.
    /// `u32::MAX` indicates that this vertex is unconstrained.
    pub body_vertex_idx: u32,
    /// Clearance target in meters along the parent vertex's outward
    /// normal. Semantics depend on `mode`: minimum distance (clearance)
    /// or maximum distance — usually negative, i.e. how far inside the
    /// parent surface the vertex may sit at most (containment).
    pub min_clearance: f32,
    /// Influence weight (0.0 to 1.0).
    pub weight: f32,
    /// Anchor mode — see [`SKIN_ANCHOR_CLEARANCE`] /
    /// [`SKIN_ANCHOR_CONTAINMENT`]. Occupies the fourth `uvec4` slot to
    /// keep std430 16-byte alignment.
    pub mode: u32,
}

impl Default for SkinAnchor {
    fn default() -> Self {
        Self {
            body_vertex_idx: u32::MAX,
            min_clearance: 0.0,
            weight: 0.0,
            mode: SKIN_ANCHOR_CLEARANCE,
        }
    }
}

/// Identifies the avatar's primary body primitive.
///
/// Prioritizes primitives whose mesh or material name contains "body" or "skin",
/// excluding clothing/hair/accessories, and chooses the candidate with the
/// largest vertex count.
pub fn find_body_primitive(asset: &AvatarAsset) -> Option<(MeshId, PrimitiveId)> {
    let mut best_match: Option<(MeshId, PrimitiveId, usize)> = None;

    for mesh in &asset.meshes {
        let m_name = mesh.name.to_lowercase();
        for prim in &mesh.primitives {
            let mat_name = asset
                .materials
                .iter()
                .find(|m| m.id == prim.material_id)
                .map(|m| m.name.to_lowercase())
                .unwrap_or_default();

            let vert_count = prim.vertex_count as usize;

            let is_excluded = m_name.contains("cloth")
                || m_name.contains("skirt")
                || m_name.contains("hair")
                || m_name.contains("dress")
                || m_name.contains("costume")
                || m_name.contains("acc")
                || m_name.contains("shoe")
                || m_name.contains("transparent")
                || mat_name.contains("cloth")
                || mat_name.contains("skirt")
                || mat_name.contains("hair")
                || mat_name.contains("dress")
                || mat_name.contains("costume")
                || mat_name.contains("acc")
                || mat_name.contains("shoe")
                || mat_name.contains("transparent");

            if is_excluded {
                continue;
            }

            let is_body = m_name.contains("body")
                || m_name.contains("skin")
                || mat_name.contains("body")
                || mat_name.contains("skin");

            if is_body {
                if best_match.map(|(_, _, vc)| vert_count > vc).unwrap_or(true) {
                    best_match = Some((mesh.id, prim.id, vert_count));
                }
            }
        }
    }

    if let Some((m_id, p_id, _)) = best_match {
        return Some((m_id, p_id));
    }

    // Fallback: pick the largest primitive not excluded
    for mesh in &asset.meshes {
        let m_name = mesh.name.to_lowercase();
        for prim in &mesh.primitives {
            let mat_name = asset
                .materials
                .iter()
                .find(|m| m.id == prim.material_id)
                .map(|m| m.name.to_lowercase())
                .unwrap_or_default();

            let vert_count = prim.vertex_count as usize;
            let is_excluded = m_name.contains("cloth")
                || m_name.contains("skirt")
                || m_name.contains("hair")
                || mat_name.contains("cloth")
                || mat_name.contains("skirt")
                || mat_name.contains("hair");
            if !is_excluded && vert_count > 1000 {
                if best_match.map(|(_, _, vc)| vert_count > vc).unwrap_or(true) {
                    best_match = Some((mesh.id, prim.id, vert_count));
                }
            }
        }
    }

    best_match.map(|(m_id, p_id, _)| (m_id, p_id))
}

/// Helper to compute rest world-space positions and unit normals for a primitive.
pub fn compute_rest_world_vertices(
    prim: &MeshPrimitiveAsset,
    skinning: &[Mat4],
) -> Vec<([f32; 3], [f32; 3])> {
    let Some(ref vd) = prim.vertices else {
        return Vec::new();
    };
    let count = vd.positions.len();
    let mut out = Vec::with_capacity(count);

    for i in 0..count {
        let pos = vd.positions[i];
        let norm = vd.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]);
        let mut wp = [0.0f32; 3];
        let mut wn = [0.0f32; 3];
        let mut total_w = 0.0f32;

        if i < vd.joint_weights.len() && i < vd.joint_indices.len() {
            for k in 0..4 {
                let w = vd.joint_weights[i][k];
                if w > 1e-4 {
                    let j = vd.joint_indices[i][k] as usize;
                    if j < skinning.len() {
                        let sm = &skinning[j];
                        let tx =
                            sm[0][0] * pos[0] + sm[1][0] * pos[1] + sm[2][0] * pos[2] + sm[3][0];
                        let ty =
                            sm[0][1] * pos[0] + sm[1][1] * pos[1] + sm[2][1] * pos[2] + sm[3][1];
                        let tz =
                            sm[0][2] * pos[0] + sm[1][2] * pos[1] + sm[2][2] * pos[2] + sm[3][2];
                        wp[0] += w * tx;
                        wp[1] += w * ty;
                        wp[2] += w * tz;

                        let nx = sm[0][0] * norm[0] + sm[1][0] * norm[1] + sm[2][0] * norm[2];
                        let ny = sm[0][1] * norm[0] + sm[1][1] * norm[1] + sm[2][1] * norm[2];
                        let nz = sm[0][2] * norm[0] + sm[1][2] * norm[1] + sm[2][2] * norm[2];
                        wn[0] += w * nx;
                        wn[1] += w * ny;
                        wn[2] += w * nz;

                        total_w += w;
                    }
                }
            }
        }

        if total_w > 1e-4 {
            wp[0] /= total_w;
            wp[1] /= total_w;
            wp[2] /= total_w;
            let nlen = (wn[0] * wn[0] + wn[1] * wn[1] + wn[2] * wn[2]).sqrt();
            if nlen > 1e-4 {
                wn[0] /= nlen;
                wn[1] /= nlen;
                wn[2] /= nlen;
            } else {
                wn = norm;
            }
        } else {
            wp = pos;
            wn = norm;
        }

        out.push((wp, wn));
    }
    out
}

/// 3D Spatial Grid for fast nearest-neighbor queries within an influence radius.
struct SpatialGrid {
    inv_cell: f32,
    cells: HashMap<(i32, i32, i32), Vec<u32>>,
}

impl SpatialGrid {
    fn new(cell_size: f32) -> Self {
        Self {
            inv_cell: 1.0 / cell_size.max(1e-4),
            cells: HashMap::new(),
        }
    }

    fn key(&self, p: [f32; 3]) -> (i32, i32, i32) {
        (
            (p[0] * self.inv_cell).floor() as i32,
            (p[1] * self.inv_cell).floor() as i32,
            (p[2] * self.inv_cell).floor() as i32,
        )
    }

    fn insert(&mut self, idx: u32, p: [f32; 3]) {
        let k = self.key(p);
        self.cells.entry(k).or_default().push(idx);
    }

    fn query_nearest(
        &self,
        p: [f32; 3],
        normal: [f32; 3],
        body_verts: &[([f32; 3], [f32; 3])],
        max_radius: f32,
    ) -> Option<(u32, f32)> {
        let max_r2 = max_radius * max_radius;
        let cell_radius = (max_radius * self.inv_cell).ceil() as i32;
        let center_key = self.key(p);

        let mut best_idx = None;
        let mut best_d2 = max_r2;

        for dx in -cell_radius..=cell_radius {
            for dy in -cell_radius..=cell_radius {
                for dz in -cell_radius..=cell_radius {
                    let k = (center_key.0 + dx, center_key.1 + dy, center_key.2 + dz);
                    if let Some(indices) = self.cells.get(&k) {
                        for &b_idx in indices {
                            let (bp, bn) = body_verts[b_idx as usize];
                            // Normal compatibility: clothing normal and body normal should point outward together
                            let dot_n = normal[0] * bn[0] + normal[1] * bn[1] + normal[2] * bn[2];
                            if dot_n < 0.1 {
                                continue;
                            }

                            let diff = [p[0] - bp[0], p[1] - bp[1], p[2] - bp[2]];
                            let d2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
                            if d2 < best_d2 {
                                best_d2 = d2;
                                best_idx = Some(b_idx);
                            }
                        }
                    }
                }
            }
        }

        best_idx.map(|idx| (idx, best_d2.sqrt()))
    }
}

/// Generates skin anchors for all clothing / skirt primitives in the avatar.
///
/// Runs once at asset import and caches the anchors inside the `.vvtcache` file.
pub fn generate_skin_anchors(asset: &mut AvatarAsset) {
    let Some((body_mesh_id, body_prim_id)) = find_body_primitive(asset) else {
        debug!("clearance: no body primitive identified; skipping skin anchors");
        return;
    };

    asset.body_primitive_id = Some(body_prim_id);

    // Compute rest-pose skinning matrices across skeleton nodes
    let node_count = asset.skeleton.nodes.len();
    let locals: Vec<_> = asset
        .skeleton
        .nodes
        .iter()
        .map(|n| n.rest_local.clone())
        .collect();
    let mut globals = vec![crate::asset::identity_matrix(); node_count];
    crate::avatar::pose::compute_global_transforms(&asset.skeleton, &locals, &mut globals);

    let mut skinning = vec![crate::asset::identity_matrix(); node_count];
    crate::avatar::pose::build_skinning_matrices(&asset.skeleton, &globals, &mut skinning);

    // Find body primitive vertex data in rest pose
    let body_prim = asset
        .meshes
        .iter()
        .find(|m| m.id == body_mesh_id)
        .and_then(|m| m.primitives.iter().find(|p| p.id == body_prim_id));

    let Some(body_prim) = body_prim else {
        return;
    };

    let body_world = compute_rest_world_vertices(body_prim, &skinning);
    if body_world.is_empty() {
        return;
    }
    // Cloned out of `asset` so the reference doesn't outlive the
    // Phase-1 mutable mesh loop below (~4 MB for a 79k-vert body).
    let (body_joint_indices, body_joint_weights) = {
        let vd = body_prim.vertices.as_ref().unwrap();
        (vd.joint_indices.clone(), vd.joint_weights.clone())
    };

    info!(
        "clearance: indexed body primitive {:?} ({} vertices) for skin anchors",
        body_prim_id,
        body_world.len()
    );

    // Build spatial grid with 4 cm voxel cells
    let cell_size = 0.04f32;
    let mut grid = SpatialGrid::new(cell_size);
    for (idx, &(bp, _)) in body_world.iter().enumerate() {
        grid.insert(idx as u32, bp);
    }

    let max_influence_radius = 0.12f32; // 12 cm maximum distance to body surface at rest
    // Layer-style grid over the body for anchor-target refinement.
    let body_layer_grid = build_layer_grid(&body_world, 0.03);
    let body_inv_cell = 1.0f32 / 0.03;

    // Process all other primitives in the avatar
    let mut anchors_generated_count = 0usize;
    // Phase-1 bottom garments (skirts), consumed by the Phase-3
    // cross-region pass below.
    let mut skirt_prim_ids: Vec<PrimitiveId> = Vec::new();

    for mesh in &mut asset.meshes {
        let m_name = mesh.name.to_lowercase();
        for prim_arc in &mut mesh.primitives {
            if prim_arc.id == body_prim_id {
                continue;
            }

            let mat_name = asset
                .materials
                .iter()
                .find(|m| m.id == prim_arc.material_id)
                .map(|m| m.name.to_lowercase())
                .unwrap_or_default();

            // Target bottom clothing: skirts, pants, dresses (lower portion) that
            // collide with thighs and legs.
            // Strictly exclude upper-body clothing (shirt, blouse, jacket, sleeve, collar, ribbon, tie, etc.)
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
                continue;
            }

            let has_skirt_name = m_name.contains("skirt")
                || m_name.contains("bottom")
                || m_name.contains("pants")
                || mat_name.contains("skirt")
                || mat_name.contains("bottom")
                || mat_name.contains("pants");

            // Check how many vertices are weighted to skirt bones
            let skirt_bone_weighted_ratio = prim_arc.vertices.as_ref().map_or(0.0f32, |vd| {
                if vd.positions.is_empty() {
                    return 0.0;
                }
                let mut skirt_vert_count = 0usize;
                for (vi, indices) in vd.joint_indices.iter().enumerate() {
                    let is_skirt_vert = indices.iter().enumerate().any(|(slot, &ji)| {
                        if (ji as usize) < asset.skeleton.nodes.len() {
                            let node_name = asset.skeleton.nodes[ji as usize].name.to_lowercase();
                            node_name.contains("skirt")
                                && vd.joint_weights.get(vi).map_or(false, |w| w[slot] > 0.1)
                        } else {
                            false
                        }
                    });
                    if is_skirt_vert {
                        skirt_vert_count += 1;
                    }
                }
                skirt_vert_count as f32 / vd.positions.len() as f32
            });

            // Must either have explicit skirt name, or >= 40% of vertices weighted to skirt bones
            let is_skirt = has_skirt_name || skirt_bone_weighted_ratio >= 0.4;
            if !is_skirt {
                continue;
            }

            let cloth_world = compute_rest_world_vertices(prim_arc, &skinning);
            if cloth_world.is_empty() {
                continue;
            }

            let mut anchors = Vec::with_capacity(cloth_world.len());
            let mut bound_count = 0usize;

            for &(cp, cn) in &cloth_world {
                if let Some((b_idx, _dist)) =
                    grid.query_nearest(cp, cn, &body_world, max_influence_radius)
                {
                    let (bp, bn) = body_world[b_idx as usize];
                    let diff = [cp[0] - bp[0], cp[1] - bp[1], cp[2] - bp[2]];
                    let raw_clearance = diff[0] * bn[0] + diff[1] * bn[1] + diff[2] * bn[2];

                    // Enforce at least 2mm clearance to prevent z-fighting and resting penetrations
                    let min_clearance = raw_clearance.max(0.002);
                    let min_clearance = refine_clearance_target(
                        cp,
                        bn,
                        raw_clearance,
                        min_clearance,
                        &body_world,
                        &body_layer_grid,
                        body_inv_cell,
                        0.06,
                    );
                    let weight = 1.0f32;

                    anchors.push(SkinAnchor {
                        body_vertex_idx: b_idx,
                        min_clearance,
                        weight,
                        mode: SKIN_ANCHOR_CLEARANCE,
                    });
                    bound_count += 1;
                } else {
                    anchors.push(SkinAnchor::default());
                }
            }

            if bound_count > 0 {
                info!(
                    "clearance: generated {} skin anchors for primitive {:?} ('{}', mat='{}')",
                    bound_count, prim_arc.id, mesh.name, mat_name
                );
                skirt_prim_ids.push(prim_arc.id);
                if let Some(ref idx) = prim_arc.indices {
                    smooth_anchor_targets(&mut anchors, idx, 3);
                }
                let prim_mut = Arc::make_mut(prim_arc);
                prim_mut.skin_anchors = Some(anchors);
                prim_mut.body_primitive_id = Some(body_prim_id);
                anchors_generated_count += bound_count;
            }
        }
    }

    info!(
        "clearance: total {} skin anchors established across avatar (Phase 1)",
        anchors_generated_count
    );

    // Phase 2: Hierarchical Layered Clothing Clearance (inner -> outer pairings)
    generate_layered_clothing_anchors(
        asset,
        &globals,
        &skinning,
        body_prim_id,
        &body_world,
        &body_joint_indices,
        &body_joint_weights,
    );

    // Phase 3: cross-region (upper-outer vs bottom-inner) clearance.
    generate_cross_region_anchors(asset, &skinning, body_prim_id, &skirt_prim_ids);
}

/// Phase 3 — cross-region clearance between the layered-clothing stack
/// and Phase-1 bottom garments.
///
/// Phase 1 anchors bottom garments (skirts, pants) against the body;
/// Phase 2 pairs the UPPER layered stack (body ← shirt ← jacket) but
/// excludes already-anchored prims from its candidate set — so the two
/// branches never meet and nothing constrains an upper garment's hem
/// against a bottom garment it drapes over. Yumeka measures ~340 jacket
/// vertices resting inside the skirt volume at bind (worst −28 mm),
/// which reads as "the jacket sinks into the skirt" the moment the
/// spring-driven skirt swings.
///
/// The pass binds each layered OUTER (a prim whose clearance parent is
/// another garment, not the body) to the overlapping bottom garment
/// with clearance-mode anchors stored in the outer's containment slot
/// (`containment_anchors` + `containment_primitive_id`). The transform
/// shader branches per anchor `mode`, so the slot carries either
/// semantic; outers whose containment slot is already occupied (middle
/// layers of a 3+ stack) are skipped — the single-slot limitation is
/// logged, not silently ignored. Skirt-in-jacket pokes in the other
/// direction stay unconstrained on purpose: a skirt surface hidden
/// inside the jacket volume is occluded and visually correct, while a
/// jacket hem inside the skirt cone visibly swallows the hem.
pub fn generate_cross_region_anchors(
    asset: &mut AvatarAsset,
    skinning: &[Mat4],
    body_pid: PrimitiveId,
    skirt_prim_ids: &[PrimitiveId],
) {
    if skirt_prim_ids.is_empty() {
        return;
    }

    struct InnerSurface {
        prim_id: PrimitiveId,
        mesh_name: String,
        verts: Vec<([f32; 3], [f32; 3])>,
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
    }
    let mut inners: Vec<InnerSurface> = Vec::new();
    for mesh in asset.meshes.iter() {
        for prim in mesh.primitives.iter() {
            if !skirt_prim_ids.contains(&prim.id) {
                continue;
            }
            let verts = compute_rest_world_vertices(prim, skinning);
            if verts.is_empty() {
                continue;
            }
            let mut aabb_min = [f32::MAX; 3];
            let mut aabb_max = [f32::MIN; 3];
            for &(p, _) in &verts {
                for c in 0..3 {
                    aabb_min[c] = aabb_min[c].min(p[c]);
                    aabb_max[c] = aabb_max[c].max(p[c]);
                }
            }
            inners.push(InnerSurface {
                prim_id: prim.id,
                mesh_name: mesh.name.clone(),
                verts,
                aabb_min,
                aabb_max,
            });
        }
    }
    if inners.is_empty() {
        return;
    }

    for mesh in &mut asset.meshes {
        for prim_arc in &mut mesh.primitives {
            if prim_arc.id == body_pid
                || skirt_prim_ids.contains(&prim_arc.id)
                || prim_arc.containment_anchors.is_some()
            {
                continue;
            }
            // Layered outers only: clearance parent is a garment, not
            // the body (Phase-1 bottoms and middle layers anchor
            // against the body and are skipped).
            let is_layered_outer = prim_arc.skin_anchors.is_some()
                && prim_arc.body_primitive_id != Some(body_pid)
                && prim_arc.body_primitive_id.is_some();
            if !is_layered_outer {
                continue;
            }

            let outer_verts = compute_rest_world_vertices(prim_arc, skinning);
            if outer_verts.is_empty() {
                continue;
            }
            let mut o_min = [f32::MAX; 3];
            let mut o_max = [f32::MIN; 3];
            for &(p, _) in &outer_verts {
                for c in 0..3 {
                    o_min[c] = o_min[c].min(p[c]);
                    o_max[c] = o_max[c].max(p[c]);
                }
            }

            for inner in &inners {
                if !aabb_intersects(o_min, o_max, inner.aabb_min, inner.aabb_max) {
                    continue;
                }
                let cell_size = 0.03f32;
                let inv_cell = 1.0 / cell_size;
                let inner_grid = build_layer_grid(&inner.verts, cell_size);
                let min_bound = (outer_verts.len() / 50).max(32);

                let mut anchors = Vec::with_capacity(outer_verts.len());
                let mut bound_count = 0usize;
                for &(op, _) in &outer_verts {
                    // Nearest inner vertex within 4 cm. No normal
                    // filter: a hem vertex resting inside the skirt
                    // cone lies BEHIND every nearby skirt vertex, so a
                    // "front-facing only" filter would drop exactly the
                    // penetrating anchors this pass exists to create.
                    // Pushing along the nearest vertex's outward normal
                    // is the escape direction either way.
                    let ck = (
                        (op[0] * inv_cell).floor() as i32,
                        (op[1] * inv_cell).floor() as i32,
                        (op[2] * inv_cell).floor() as i32,
                    );
                    let mut best: Option<(f32, usize)> = None;
                    let max_r2 = 0.04f32 * 0.04f32;
                    // ±2 cells of 3 cm cover the full 4 cm radius.
                    for dx in -2..=2 {
                        for dy in -2..=2 {
                            for dz in -2..=2 {
                                if let Some(list) =
                                    inner_grid.get(&(ck.0 + dx, ck.1 + dy, ck.2 + dz))
                                {
                                    for &idx in list {
                                        let (ip, _) = inner.verts[idx as usize];
                                        let d2 = (op[0] - ip[0]).powi(2)
                                            + (op[1] - ip[1]).powi(2)
                                            + (op[2] - ip[2]).powi(2);
                                        if d2 < max_r2
                                            && best.map_or(true, |(b, _)| d2 < b)
                                        {
                                            best = Some((d2, idx as usize));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if let Some((_, idx)) = best {
                        let (ip, inrm) = inner.verts[idx];
                        let raw = (op[0] - ip[0]) * inrm[0]
                            + (op[1] - ip[1]) * inrm[1]
                            + (op[2] - ip[2]) * inrm[2];
                        let min_clearance = refine_clearance_target(
                            op,
                            inrm,
                            raw,
                            raw.max(0.006),
                            &inner.verts,
                            &inner_grid,
                            inv_cell,
                            0.04,
                        );
                        anchors.push(SkinAnchor {
                            body_vertex_idx: idx as u32,
                            min_clearance,
                            weight: 1.0,
                            mode: SKIN_ANCHOR_CLEARANCE,
                        });
                        bound_count += 1;
                    } else {
                        anchors.push(SkinAnchor::default());
                    }
                }

                if bound_count >= min_bound {
                    info!(
                        "clearance: cross-region pair: outer '{}' (prim {:?}) -> bottom '{}' (prim {:?}), {} / {} verts bound",
                        mesh.name,
                        prim_arc.id,
                        inner.mesh_name,
                        inner.prim_id,
                        bound_count,
                        outer_verts.len()
                    );
                    if let Some(ref idx) = prim_arc.indices {
                        smooth_anchor_targets(&mut anchors, idx, 3);
                    }
                    let prim_mut = Arc::make_mut(prim_arc);
                    prim_mut.containment_anchors = Some(anchors);
                    prim_mut.containment_primitive_id = Some(inner.prim_id);
                    // One bottom parent per outer: stop at the first
                    // (currently the only) overlapping bottom garment.
                    break;
                } else {
                    info!(
                        "clearance: bottom candidate '{}' bound only {}/{} verts on outer '{}' (min {}); skipping pair",
                        inner.mesh_name,
                        bound_count,
                        outer_verts.len(),
                        mesh.name,
                        min_bound
                    );
                }
            }
        }
    }

    // Direction B — the same gap seen from the bottom: an UNANCHORED
    // inner layer the bottom should wrap outside (Yumeka's underwear,
    // garter straps) pokes through the bottom's surface. Phase 1 keeps
    // the bottom off the BODY, but the underwear sits a few mm ABOVE
    // the body, so the body-anchored bottom still swallows it — the
    // visible "underwear through the skirt" clip. Bind the bottom's
    // (free) containment slot with clearance-mode anchors against the
    // worst rest penetrator.
    for bottom in &inners {
        // The bottom prim's own mesh slot.
        let mut bottom_loc: Option<(usize, usize)> = None;
        for (m_idx, mesh) in asset.meshes.iter().enumerate() {
            for (p_idx, prim) in mesh.primitives.iter().enumerate() {
                if prim.id == bottom.prim_id {
                    bottom_loc = Some((m_idx, p_idx));
                }
            }
        }
        let Some((b_mesh, b_prim)) = bottom_loc else {
            continue;
        };
        if asset.meshes[b_mesh].primitives[b_prim]
            .containment_anchors
            .is_some()
        {
            continue;
        }

        // Candidate inner layers: unanchored, not the body, not
        // bottoms, not face/hair classes, overlapping the bottom, and
        // mostly INSIDE it (a majority-outside prim is a belt worn over
        // the bottom and must not receive this constraint).
        struct UnderCandidate {
            prim_id: PrimitiveId,
            mesh_name: String,
            verts: Vec<([f32; 3], [f32; 3])>,
            pokes: usize,
        }
        let mut cands: Vec<UnderCandidate> = Vec::new();
        let b_grid = build_layer_grid(&bottom.verts, 0.04);
        for mesh in asset.meshes.iter() {
            let m_name = mesh.name.to_lowercase();
            if m_name.contains("hair") || m_name.contains("face") || m_name.contains("eye") {
                continue;
            }
            for prim in mesh.primitives.iter() {
                if prim.id == body_pid
                    || prim.id == bottom.prim_id
                    || skirt_prim_ids.contains(&prim.id)
                    || prim.skin_anchors.is_some()
                    || prim.vertex_count < 100
                {
                    continue;
                }
                let mat_name = asset
                    .materials
                    .iter()
                    .find(|m| m.id == prim.material_id)
                    .map(|m| m.name.to_lowercase())
                    .unwrap_or_default();
                if mat_name.contains("hair") || mat_name.contains("face") {
                    continue;
                }
                let verts = compute_rest_world_vertices(prim, skinning);
                if verts.is_empty() {
                    continue;
                }
                let mut c_min = [f32::MAX; 3];
                let mut c_max = [f32::MIN; 3];
                for &(p, _) in &verts {
                    for c in 0..3 {
                        c_min[c] = c_min[c].min(p[c]);
                        c_max[c] = c_max[c].max(p[c]);
                    }
                }
                if !aabb_intersects(c_min, c_max, bottom.aabb_min, bottom.aabb_max) {
                    continue;
                }

                // Sample: nearest bottom vertex within 6 cm; count
                // rest pokes (beyond the bottom surface) and near pairs.
                let inv_cell = 1.0f32 / 0.04;
                let step = (verts.len() / 500).max(1);
                let mut near = 0usize;
                let mut pokes = 0usize;
                for vi in (0..verts.len()).step_by(step) {
                    let (cp, _) = verts[vi];
                    let ck = (
                        (cp[0] * inv_cell).floor() as i32,
                        (cp[1] * inv_cell).floor() as i32,
                        (cp[2] * inv_cell).floor() as i32,
                    );
                    let mut best: Option<(f32, usize)> = None;
                    let max_r2 = 0.06f32 * 0.06f32;
                    for dx in -2..=2 {
                        for dy in -2..=2 {
                            for dz in -2..=2 {
                                if let Some(list) =
                                    b_grid.get(&(ck.0 + dx, ck.1 + dy, ck.2 + dz))
                                {
                                    for &bi in list {
                                        let (bp, _) = bottom.verts[bi as usize];
                                        let d2 = (cp[0] - bp[0]).powi(2)
                                            + (cp[1] - bp[1]).powi(2)
                                            + (cp[2] - bp[2]).powi(2);
                                        if d2 < max_r2 && best.map_or(true, |(b, _)| d2 < b) {
                                            best = Some((d2, bi as usize));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if let Some((_, bi)) = best {
                        let (bp, bn) = bottom.verts[bi];
                        let d = (cp[0] - bp[0]) * bn[0]
                            + (cp[1] - bp[1]) * bn[1]
                            + (cp[2] - bp[2]) * bn[2];
                        near += 1;
                        if d > 0.002 {
                            pokes += 1;
                        }
                    }
                }
                // Inner-layer gate: enough overlap, real penetration,
                // and a majority still inside (an over-belt reads as
                // mostly pokes and is excluded).
                if pokes >= 40 && near >= 80 && (near - pokes) * 2 >= near {
                    cands.push(UnderCandidate {
                        prim_id: prim.id,
                        mesh_name: mesh.name.clone(),
                        verts,
                        pokes,
                    });
                }
            }
        }
        cands.sort_by(|a, b| b.pokes.cmp(&a.pokes));

        for cand in &cands {
            let c_grid = build_layer_grid(&cand.verts, 0.03);
            let inv_cell = 1.0f32 / 0.03f32;
            let min_bound = (bottom.verts.len() / 50).max(32);
            let mut anchors = Vec::with_capacity(bottom.verts.len());
            let mut bound_count = 0usize;
            for &(bp, _) in &bottom.verts {
                let ck = (
                    (bp[0] * inv_cell).floor() as i32,
                    (bp[1] * inv_cell).floor() as i32,
                    (bp[2] * inv_cell).floor() as i32,
                );
                let mut best: Option<(f32, usize)> = None;
                let max_r2 = 0.04f32 * 0.04f32;
                for dx in -2..=2 {
                    for dy in -2..=2 {
                        for dz in -2..=2 {
                            if let Some(list) = c_grid.get(&(ck.0 + dx, ck.1 + dy, ck.2 + dz)) {
                                for &ci in list {
                                    let (cp, _) = cand.verts[ci as usize];
                                    let d2 = (bp[0] - cp[0]).powi(2)
                                        + (bp[1] - cp[1]).powi(2)
                                        + (bp[2] - cp[2]).powi(2);
                                    if d2 < max_r2 && best.map_or(true, |(b, _)| d2 < b) {
                                        best = Some((d2, ci as usize));
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some((_, ci)) = best {
                    let (cp, cn) = cand.verts[ci];
                    let raw = (bp[0] - cp[0]) * cn[0]
                        + (bp[1] - cp[1]) * cn[1]
                        + (bp[2] - cp[2]) * cn[2];
                    let min_clearance = refine_clearance_target(
                        bp,
                        cn,
                        raw,
                        raw.max(0.006),
                        &cand.verts,
                        &c_grid,
                        inv_cell,
                        0.04,
                    );
                    anchors.push(SkinAnchor {
                        body_vertex_idx: ci as u32,
                        min_clearance,
                        weight: 1.0,
                        mode: SKIN_ANCHOR_CLEARANCE,
                    });
                    bound_count += 1;
                } else {
                    anchors.push(SkinAnchor::default());
                }
            }
            if bound_count >= min_bound {
                info!(
                    "clearance: cross-region pair (direction B): bottom '{}' (prim {:?}) -> inner '{}' (prim {:?}), {} / {} verts bound ({} rest pokes)",
                    bottom.mesh_name,
                    bottom.prim_id,
                    cand.mesh_name,
                    cand.prim_id,
                    bound_count,
                    bottom.verts.len(),
                    cand.pokes
                );
                if let Some(ref idx) =
                    asset.meshes[b_mesh].primitives[b_prim].indices.clone()
                {
                    smooth_anchor_targets(&mut anchors, &idx, 3);
                }
                let prim_mut = Arc::make_mut(&mut asset.meshes[b_mesh].primitives[b_prim]);
                prim_mut.containment_anchors = Some(anchors);
                prim_mut.containment_primitive_id = Some(cand.prim_id);
                break;
            }
        }
    }
}

/// Bump a clearance anchor's target until ONE push along the anchor
/// normal lands the vertex outside every same-facing plane nearby.
///
/// A naive `(raw, floor)` target only clears the anchor's own plane:
/// convex bulges (buttocks vs a skirt authored flat) and stacked
/// sheets (frills) between anchor vertices still swallow the pushed
/// vertex, and the rendered surface interpolates through them. The
/// refinement simulates the push and raises the target until the
/// landing point clears every nearby plane whose normal roughly
/// agrees with the push direction — back-facing sheets are excluded,
/// because a point outside a closed-ish surface is always behind some
/// opposite-facing sheet and demanding clearance from those never
/// converges (measured: over-pushed hems punching through far sheets).
fn refine_clearance_target(
    origin: [f32; 3],
    push_normal: [f32; 3],
    raw: f32,
    mut min_clearance: f32,
    parent_verts: &[([f32; 3], [f32; 3])],
    parent_grid: &HashMap<(i32, i32, i32), Vec<u32>>,
    inv_cell: f32,
    max_target: f32,
) -> f32 {
    for _ in 0..4 {
        if min_clearance >= max_target {
            return max_target;
        }
        let t = min_clearance - raw;
        let land = [
            origin[0] + push_normal[0] * t,
            origin[1] + push_normal[1] * t,
            origin[2] + push_normal[2] * t,
        ];
        let lk = (
            (land[0] * inv_cell).floor() as i32,
            (land[1] * inv_cell).floor() as i32,
            (land[2] * inv_cell).floor() as i32,
        );
        let mut worst = 0.0f32;
        for dx in -2..=2 {
            for dy in -2..=2 {
                for dz in -2..=2 {
                    if let Some(list) = parent_grid.get(&(lk.0 + dx, lk.1 + dy, lk.2 + dz)) {
                        for &j in list {
                            let (jp, jn) = parent_verts[j as usize];
                            // Same-facing planes only.
                            if jn[0] * push_normal[0] + jn[1] * push_normal[1]
                                + jn[2] * push_normal[2]
                                < 0.3
                            {
                                continue;
                            }
                            let c = (land[0] - jp[0]) * jn[0]
                                + (land[1] - jp[1]) * jn[1]
                                + (land[2] - jp[2]) * jn[2];
                            worst = worst.min(c);
                        }
                    }
                }
            }
        }
        if worst >= 0.002 {
            return min_clearance;
        }
        min_clearance += 0.002 - worst;
    }
    min_clearance.min(max_target)
}

/// Smooth the per-vertex clearance-target field over the owning mesh.
///
/// Refined targets vary sharply where the parent surface bulges (a
/// skirt authored flat across the buttocks needs ~50 mm pushes at the
/// peak and ~0 next to it); the rendered surface interpolates linearly
/// between vertices, so unsmoothed targets leave deep dips between a
/// pushed vertex and its neighbours — measured as worse residual
/// penetration than no refinement at all. A few Jacobi iterations of
/// neighbour averaging spread the peak over the region: the garment
/// comes to REST on the bulge instead of tenting over it. Unbound
/// vertices (default anchors) stay fixed — smoothing into them would
/// invent pushes their anchor data can't express.
fn smooth_anchor_targets(anchors: &mut [SkinAnchor], indices: &[u32], iterations: usize) {
    if indices.is_empty() || anchors.is_empty() {
        return;
    }
    let n = anchors.len();
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for t in (0..indices.len()).step_by(3) {
        let tri = [&indices[t], &indices[t + 1], &indices[t + 2]];
        for a in 0..3 {
            for b in 0..3 {
                if a != b {
                    let (i, j) = (*tri[a] as usize, *tri[b] as usize);
                    if i < n && j < n {
                        adj[i].push(j as u32);
                    }
                }
            }
        }
    }
    for _ in 0..iterations {
        let old: Vec<f32> = anchors
            .iter()
            .map(|a| {
                if a.body_vertex_idx == u32::MAX {
                    f32::NAN
                } else {
                    a.min_clearance
                }
            })
            .collect();
        for (i, a) in anchors.iter_mut().enumerate() {
            if a.body_vertex_idx == u32::MAX {
                continue;
            }
            let neighbors: Vec<f32> = adj[i]
                .iter()
                .filter_map(|j| {
                    let v = old[*j as usize];
                    if v.is_nan() {
                        None
                    } else {
                        Some(v)
                    }
                })
                .collect();
            if neighbors.is_empty() {
                continue;
            }
            let avg = neighbors.iter().sum::<f32>() / neighbors.len() as f32;
            a.min_clearance = 0.5 * old[i] + 0.5 * avg;
        }
    }
}

/// Helper: computes cosine similarity between two vertex 4-bone weight sets.
pub fn compute_bone_weight_similarity(
    indices_a: &[u16; 4],
    weights_a: &[f32; 4],
    indices_b: &[u16; 4],
    weights_b: &[f32; 4],
) -> f32 {
    let mut dot = 0.0f32;
    let mut len_a2 = 0.0f32;
    let mut len_b2 = 0.0f32;
    for i in 0..4 {
        let wa = weights_a[i];
        len_a2 += wa * wa;
        for j in 0..4 {
            let wb = weights_b[j];
            if indices_a[i] == indices_b[j] {
                dot += wa * wb;
            }
        }
    }
    for j in 0..4 {
        let wb = weights_b[j];
        len_b2 += wb * wb;
    }
    if len_a2 > 1e-4 && len_b2 > 1e-4 {
        dot / (len_a2.sqrt() * len_b2.sqrt())
    } else {
        0.0
    }
}

/// Helper: checks whether two 3D Axis-Aligned Bounding Boxes intersect.
fn aabb_intersects(min_a: [f32; 3], max_a: [f32; 3], min_b: [f32; 3], max_b: [f32; 3]) -> bool {
    min_a[0] <= max_b[0]
        && max_a[0] >= min_b[0]
        && min_a[1] <= max_b[1]
        && max_a[1] >= min_b[1]
        && min_a[2] <= max_b[2]
        && max_a[2] >= min_b[2]
}

/// Spatial grid over a layer's rest-pose world vertices for the layered
/// anchor searches. 3 cm cells (an anchor search window of a few cells
/// covers the 4 cm binding radius).
fn build_layer_grid(
    verts: &[([f32; 3], [f32; 3])],
    cell_size: f32,
) -> HashMap<(i32, i32, i32), Vec<u32>> {
    let inv_cell = 1.0 / cell_size.max(1e-4);
    let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
    for (idx, &(p, _)) in verts.iter().enumerate() {
        let k = (
            (p[0] * inv_cell).floor() as i32,
            (p[1] * inv_cell).floor() as i32,
            (p[2] * inv_cell).floor() as i32,
        );
        grid.entry(k).or_default().push(idx as u32);
    }
    grid
}

/// Nearest compatible vertex on a target layer for one query vertex,
/// using the layered-clothing compatibility filters: outward-normal
/// agreement (`dot >= 0`), bone-weight similarity `>= 0.2`, and within
/// `max_radius` at rest. The score prefers near neighbours and high
/// bone-weight similarity. Shared by the forward (outer→inner
/// clearance) and reverse (inner→outer containment) anchor passes.
#[allow(clippy::too_many_arguments)]
fn query_layered_nearest(
    query_pos: [f32; 3],
    query_nrm: [f32; 3],
    query_joint_indices: &[u16; 4],
    query_joint_weights: &[f32; 4],
    target_grid: &HashMap<(i32, i32, i32), Vec<u32>>,
    target_verts: &[([f32; 3], [f32; 3])],
    target_joint_indices: &[[u16; 4]],
    target_joint_weights: &[[f32; 4]],
    inv_cell: f32,
    max_radius: f32,
    min_sim: f32,
) -> Option<u32> {
    let max_r2 = max_radius * max_radius;
    let ck = (
        (query_pos[0] * inv_cell).floor() as i32,
        (query_pos[1] * inv_cell).floor() as i32,
        (query_pos[2] * inv_cell).floor() as i32,
    );
    let cell_radius = (max_radius * inv_cell).ceil() as i32;
    let mut best_i = None;
    let mut best_score = max_r2;

    for dx in -cell_radius..=cell_radius {
        for dy in -cell_radius..=cell_radius {
            for dz in -cell_radius..=cell_radius {
                if let Some(list) = target_grid.get(&(ck.0 + dx, ck.1 + dy, ck.2 + dz)) {
                    for &idx in list {
                        let (tp, tn) = target_verts[idx as usize];
                        // Normal compatibility: the two surfaces must
                        // face the same way at the pairing point.
                        let dot_n =
                            query_nrm[0] * tn[0] + query_nrm[1] * tn[1] + query_nrm[2] * tn[2];
                        if dot_n < 0.0 {
                            continue;
                        }

                        let w_sim = compute_bone_weight_similarity(
                            query_joint_indices,
                            query_joint_weights,
                            &target_joint_indices[idx as usize],
                            &target_joint_weights[idx as usize],
                        );
                        if w_sim < min_sim {
                            continue;
                        }

                        let diff = [
                            query_pos[0] - tp[0],
                            query_pos[1] - tp[1],
                            query_pos[2] - tp[2],
                        ];
                        let d2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
                        let score = d2 + 0.0003 * (1.0 - w_sim);
                        if score < best_score {
                            best_score = score;
                            best_i = Some(idx);
                        }
                    }
                }
            }
        }
    }
    best_i
}

/// Generate body-surface clearance anchors for one garment's rest world
/// vertices — the Phase-2 body-parenting rules: middle layers of a
/// layered stack and innermost layers with a free clearance slot anchor
/// against the BODY so sleeves stay off the skin at bent joints (a
/// garment-garment clearance parent never covers sleeves — the inner
/// garment has none).
///
/// The pairing is BONE-WEIGHT-AWARE (unlike the Phase-1 skirt query):
/// near joints the garment and the skin are often weighted differently
/// across the joint boundary, and a pure nearest-neighbour pairing
/// matches vertices that do not co-move under bending — the anchored
/// plane then describes a different part of the arm than the vertex
/// pressing through it (measured: the elbow bulge poked 19.8mm past a
/// sleeve held 16.8mm off a skin vertex 17mm away). Weight-similar
/// pairings co-move and the rest-derived clearance stays meaningful.
#[allow(clippy::too_many_arguments)]
fn anchor_prim_against_body(
    world_verts: &[([f32; 3], [f32; 3])],
    own_joint_indices: &[[u16; 4]],
    own_joint_weights: &[[f32; 4]],
    body_world: &[([f32; 3], [f32; 3])],
    body_joint_indices: &[[u16; 4]],
    body_joint_weights: &[[f32; 4]],
    body_grid_3cm: &HashMap<(i32, i32, i32), Vec<u32>>,
) -> (Vec<SkinAnchor>, usize) {
    let inv_cell = 1.0 / 0.03f32;
    let mut anchors = Vec::with_capacity(world_verts.len());
    let mut bound_count = 0usize;
    for (vi, &(cp, cn)) in world_verts.iter().enumerate() {
        let b_idx = query_layered_nearest(
            cp,
            cn,
            &own_joint_indices[vi],
            &own_joint_weights[vi],
            body_grid_3cm,
            body_world,
            body_joint_indices,
            body_joint_weights,
            inv_cell,
            0.12,
            0.25,
        );
        if let Some(b_idx) = b_idx {
            let (bp, bn) = body_world[b_idx as usize];
            let diff = [cp[0] - bp[0], cp[1] - bp[1], cp[2] - bp[2]];
            let raw_clearance = diff[0] * bn[0] + diff[1] * bn[1] + diff[2] * bn[2];
            anchors.push(SkinAnchor {
                body_vertex_idx: b_idx,
                min_clearance: raw_clearance.max(0.002),
                weight: 1.0,
                mode: SKIN_ANCHOR_CLEARANCE,
            });
            bound_count += 1;
        } else {
            anchors.push(SkinAnchor::default());
        }
    }
    (anchors, bound_count)
}

/// Phase 2: Hierarchical Layered Clothing Clearance.
///
/// Automatically discovers multi-layer clothing pairings (e.g. shirt -> blazer,
/// inner blouse -> outer jacket, underwear -> outerwear) using rest-pose geometry
/// and bone kinematics. Each pair gets anchors in BOTH directions:
///
/// - forward (clearance) on the outer layer: the transform shader pushes the
///   outer surface off the inner one so the outer garment never sinks into it;
/// - reverse (containment) on the inner layer: the shader clamps inner vertices
///   back inside the outer surface when they poke through it — the outer-side
///   clearance alone cannot stop an inner vertex sliding between two anchored
///   outer vertices at bent joints.
///
/// The two directions live in separate slots (`skin_anchors` /
/// `body_primitive_id` vs `containment_anchors` / `containment_primitive_id`),
/// so the middle layer of a 3+ layer stack carries both at once.
pub fn generate_layered_clothing_anchors(
    asset: &mut AvatarAsset,
    globals: &[Mat4],
    skinning: &[Mat4],
    body_pid: PrimitiveId,
    body_world: &[([f32; 3], [f32; 3])],
    body_joint_indices: &[[u16; 4]],
    body_joint_weights: &[[f32; 4]],
) {
    struct PrimCandidate {
        mesh_idx: usize,
        prim_idx: usize,
        prim_id: PrimitiveId,
        mesh_name: String,
        world_verts: Vec<([f32; 3], [f32; 3])>,
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
    }

    let mut candidates = Vec::new();

    for (m_idx, mesh) in asset.meshes.iter().enumerate() {
        let m_name = mesh.name.to_lowercase();
        if m_name.contains("hair")
            || m_name.contains("face")
            || m_name.contains("eye")
            || m_name.contains("brow")
        {
            continue;
        }

        for (p_idx, prim) in mesh.primitives.iter().enumerate() {
            if Some(prim.id) == asset.body_primitive_id || prim.body_primitive_id.is_some() {
                continue;
            }

            let mat_name = asset
                .materials
                .iter()
                .find(|m| m.id == prim.material_id)
                .map(|m| m.name.to_lowercase())
                .unwrap_or_default();

            if mat_name.contains("hair") || mat_name.contains("face") || mat_name.contains("eye") {
                continue;
            }

            let vert_count = prim.vertex_count as usize;
            if vert_count < 100 {
                continue;
            }

            let world_verts = compute_rest_world_vertices(prim, skinning);
            if world_verts.is_empty() {
                continue;
            }

            let mut aabb_min = [f32::MAX; 3];
            let mut aabb_max = [f32::MIN; 3];
            for &(p, _) in &world_verts {
                for c in 0..3 {
                    if p[c] < aabb_min[c] {
                        aabb_min[c] = p[c];
                    }
                    if p[c] > aabb_max[c] {
                        aabb_max[c] = p[c];
                    }
                }
            }

            candidates.push(PrimCandidate {
                mesh_idx: m_idx,
                prim_idx: p_idx,
                prim_id: prim.id,
                mesh_name: mesh.name.clone(),
                world_verts,
                aabb_min,
                aabb_max,
            });
        }
    }

    if candidates.len() < 2 {
        return;
    }

    let body_grid_3cm = build_layer_grid(body_world, 0.03);

    // Pairwise geometric layer analysis: determine which primitive is inner vs outer.
    // Map of outer_candidate_index -> every accepted inner candidate
    // (inner_candidate_index, avg_clearance, paired_count). All
    // candidates are kept — the forward pass ranks them and falls
    // through to the next when one fails to bind enough anchors, so a
    // stray accessory that wins the radial analysis by tightness cannot
    // silently leave the outer layer unconstrained.
    let mut inner_candidates: HashMap<usize, Vec<(usize, f32, usize)>> = HashMap::new();

    let candidate_count = candidates.len();
    for i in 0..candidate_count {
        for j in (i + 1)..candidate_count {
            if !aabb_intersects(
                candidates[i].aabb_min,
                candidates[i].aabb_max,
                candidates[j].aabb_min,
                candidates[j].aabb_max,
            ) {
                continue;
            }

            let prim_a = &candidates[i];
            let prim_b = &candidates[j];

            let a_vd = asset.meshes[prim_a.mesh_idx].primitives[prim_a.prim_idx]
                .vertices
                .as_ref()
                .unwrap();
            let b_vd = asset.meshes[prim_b.mesh_idx].primitives[prim_b.prim_idx]
                .vertices
                .as_ref()
                .unwrap();

            // Build spatial grid for candidate A
            let cell_size = 0.04f32;
            let inv_cell = 1.0 / cell_size;
            let mut grid_a: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
            for (idx, &(p, _)) in prim_a.world_verts.iter().enumerate() {
                let k = (
                    (p[0] * inv_cell).floor() as i32,
                    (p[1] * inv_cell).floor() as i32,
                    (p[2] * inv_cell).floor() as i32,
                );
                grid_a.entry(k).or_default().push(idx as u32);
            }

            // Query sample of vertices from B to A within 3.5 cm
            let max_r2 = 0.035 * 0.035;
            let mut overlap_pairs = Vec::new();

            // Stride to keep import time under 15 ms
            let step_b = (prim_b.world_verts.len() / 500).max(1);
            for b_idx in (0..prim_b.world_verts.len()).step_by(step_b) {
                let (bp, bn) = prim_b.world_verts[b_idx];
                let ck = (
                    (bp[0] * inv_cell).floor() as i32,
                    (bp[1] * inv_cell).floor() as i32,
                    (bp[2] * inv_cell).floor() as i32,
                );
                let mut best_a = None;
                let mut best_d2 = max_r2;

                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            if let Some(list) = grid_a.get(&(ck.0 + dx, ck.1 + dy, ck.2 + dz)) {
                                for &a_idx in list {
                                    let (ap, an) = prim_a.world_verts[a_idx as usize];
                                    let dot_n = bn[0] * an[0] + bn[1] * an[1] + bn[2] * an[2];
                                    if dot_n < 0.0 {
                                        continue;
                                    }

                                    let sim = compute_bone_weight_similarity(
                                        &b_vd.joint_indices[b_idx],
                                        &b_vd.joint_weights[b_idx],
                                        &a_vd.joint_indices[a_idx as usize],
                                        &a_vd.joint_weights[a_idx as usize],
                                    );
                                    if sim < 0.2 {
                                        continue;
                                    }

                                    let diff = [bp[0] - ap[0], bp[1] - ap[1], bp[2] - ap[2]];
                                    let d2 =
                                        diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
                                    if d2 < best_d2 {
                                        best_d2 = d2;
                                        best_a = Some(a_idx as usize);
                                    }
                                }
                            }
                        }
                    }
                }

                if let Some(a_idx) = best_a {
                    let (ap, _an) = prim_a.world_verts[a_idx];
                    let prim_bone_a = a_vd.joint_indices[a_idx][0] as usize;
                    let bone_pos = if prim_bone_a < globals.len() {
                        [
                            globals[prim_bone_a][3][0],
                            globals[prim_bone_a][3][1],
                            globals[prim_bone_a][3][2],
                        ]
                    } else {
                        [0.0, 0.0, 0.0]
                    };
                    let dist_b = crate::math_utils::vec3_length(&crate::math_utils::vec3_sub(
                        &bp, &bone_pos,
                    ));
                    let dist_a = crate::math_utils::vec3_length(&crate::math_utils::vec3_sub(
                        &ap, &bone_pos,
                    ));
                    let radial_diff = dist_b - dist_a;
                    overlap_pairs.push(radial_diff);
                }
            }

            if overlap_pairs.len() >= 40 {
                let outer_b_count = overlap_pairs.iter().filter(|&&r| r > 0.0).count();
                let outer_b_ratio = outer_b_count as f32 / overlap_pairs.len() as f32;
                let avg_r: f32 = overlap_pairs.iter().sum::<f32>() / overlap_pairs.len() as f32;

                // If B is outer of A (B is further from bone than A)
                if outer_b_ratio >= 0.65 && avg_r > 0.001 {
                    inner_candidates
                        .entry(j)
                        .or_default()
                        .push((i, avg_r, overlap_pairs.len()));
                }
                // If A is outer of B (A is further from bone than B)
                else if outer_b_ratio <= 0.35 && avg_r < -0.001 {
                    inner_candidates
                        .entry(i)
                        .or_default()
                        .push((j, -avg_r, overlap_pairs.len()));
                }
            }
        }
    }

    // Forward pass: clearance anchors on the outer primitives. Inner
    // candidates are ranked — major overlap first (the "bigger mesh is
    // the real garment" heuristic), tightest average clearance second —
    // and tried in order. A candidate must bind at least 2% of the
    // outer surface (floor of 32 verts) to be accepted; a stray
    // accessory that won the radial analysis by tightness binds almost
    // nothing and falls through to the next candidate instead of
    // silently leaving the outer layer unconstrained.
    let mut chosen: Vec<(usize, usize, usize, Vec<SkinAnchor>)> = Vec::new();
    for (&outer_idx, cands) in &inner_candidates {
        let outer_cand = &candidates[outer_idx];
        let min_bound = (outer_cand.world_verts.len() / 50).max(32);
        let mut ranked = cands.clone();
        ranked.sort_by(|a, b| b.2.cmp(&a.2).then(a.1.partial_cmp(&b.1).unwrap()));

        let mut accepted: Option<(usize, usize, Vec<SkinAnchor>)> = None;
        for &(inner_idx, avg_c, n_overlap) in &ranked {
            let inner_cand = &candidates[inner_idx];

            let cell_size = 0.03f32;
            let inv_cell = 1.0 / cell_size;
            let inner_grid = build_layer_grid(&inner_cand.world_verts, cell_size);

            let o_vd = asset.meshes[outer_cand.mesh_idx].primitives[outer_cand.prim_idx]
                .vertices
                .as_ref()
                .unwrap();
            let i_vd = asset.meshes[inner_cand.mesh_idx].primitives[inner_cand.prim_idx]
                .vertices
                .as_ref()
                .unwrap();

            let mut anchors = Vec::with_capacity(outer_cand.world_verts.len());
            let mut bound_count = 0usize;

            for (oi, &(op, on)) in outer_cand.world_verts.iter().enumerate() {
                let best_i = query_layered_nearest(
                    op,
                    on,
                    &o_vd.joint_indices[oi],
                    &o_vd.joint_weights[oi],
                    &inner_grid,
                    &inner_cand.world_verts,
                    &i_vd.joint_indices,
                    &i_vd.joint_weights,
                    inv_cell,
                    0.04,
                    0.2,
                );

                if let Some(idx) = best_i {
                    let (ip, inrm) = inner_cand.world_verts[idx as usize];
                    let diff = [op[0] - ip[0], op[1] - ip[1], op[2] - ip[2]];
                    let raw_c = diff[0] * inrm[0] + diff[1] * inrm[1] + diff[2] * inrm[2];
                    let min_clearance = raw_c.max(0.006);
                    anchors.push(SkinAnchor {
                        body_vertex_idx: idx,
                        min_clearance,
                        weight: 1.0,
                        mode: SKIN_ANCHOR_CLEARANCE,
                    });
                    bound_count += 1;
                } else {
                    anchors.push(SkinAnchor::default());
                }
            }

            if bound_count >= min_bound {
                info!(
                    "clearance: paired layered clothing: outer='{}' (prim {:?}) -> inner='{}' (prim {:?}), avg clearance={:.2}mm, {} overlap samples, {}/{} verts bound",
                    outer_cand.mesh_name,
                    outer_cand.prim_id,
                    inner_cand.mesh_name,
                    inner_cand.prim_id,
                    avg_c * 1000.0,
                    n_overlap,
                    bound_count,
                    outer_cand.world_verts.len()
                );
                accepted = Some((inner_idx, n_overlap, anchors));
                break;
            } else {
                info!(
                    "clearance: inner candidate '{}' bound only {}/{} verts on outer '{}' (min {}); trying next candidate",
                    inner_cand.mesh_name,
                    bound_count,
                    outer_cand.world_verts.len(),
                    outer_cand.mesh_name,
                    min_bound
                );
            }
        }

        if let Some((inner_idx, n_overlap, anchors)) = accepted {
            chosen.push((outer_idx, inner_idx, n_overlap, anchors));
        } else if !ranked.is_empty() {
            warn!(
                "clearance: no inner candidate produced usable anchors for outer '{}' (prim {:?}); layer left unconstrained",
                outer_cand.mesh_name, outer_cand.prim_id
            );
        }
    }

    // Parent selection + write. A garment that is itself the inner of
    // another accepted pair (a MIDDLE layer of the stack) anchors its
    // clearance against the BODY instead of its garment-inner: the
    // garment-inner is sleeveless or differently-cut, so its anchors
    // only cover the torso and leave the sleeve region unconstrained —
    // exactly where the arm presses through at bent joints. Body
    // anchoring uses the Phase-1 semantics (12 cm radius, 2 mm floor).
    // Outermost layers keep the garment pairing (their clearance
    // follows the inner garment surface, which itself now follows the
    // body — a transitive chain body -> middle -> outer).
    let middles: std::collections::HashSet<usize> =
        chosen.iter().map(|(_, inner, _, _)| *inner).collect();
    let mut accepted_pairs: Vec<(usize, usize, usize)> = Vec::new();
    for (outer_idx, inner_idx, n_overlap, garment_anchors) in chosen {
        let outer_cand = &candidates[outer_idx];
        let inner_cand = &candidates[inner_idx];
        let use_body = middles.contains(&outer_idx);
        let (anchors, parent_pid) = if use_body {
            let (own_idx, own_w) = {
                let vd = asset.meshes[outer_cand.mesh_idx].primitives[outer_cand.prim_idx]
                    .vertices
                    .as_ref()
                    .unwrap();
                (&vd.joint_indices, &vd.joint_weights)
            };
            let (body_anchors, body_bound) = anchor_prim_against_body(
                &outer_cand.world_verts,
                own_idx,
                own_w,
                body_world,
                body_joint_indices,
                body_joint_weights,
                &body_grid_3cm,
            );
            if body_bound > 0 {
                info!(
                    "clearance: middle layer '{}' (prim {:?}) anchors against the body ({} bound) — garment-inner '{}' has no sleeve coverage",
                    outer_cand.mesh_name, outer_cand.prim_id, body_bound, inner_cand.mesh_name
                );
                (body_anchors, body_pid)
            } else {
                (garment_anchors, inner_cand.prim_id)
            }
        } else {
            (garment_anchors, inner_cand.prim_id)
        };
        let prim_mut = Arc::make_mut(
            &mut asset.meshes[outer_cand.mesh_idx].primitives[outer_cand.prim_idx],
        );
        prim_mut.skin_anchors = Some(anchors);
        prim_mut.body_primitive_id = Some(parent_pid);
        accepted_pairs.push((outer_idx, inner_idx, n_overlap));
    }

    // Reverse pass: containment anchors on the inner layers.
    //
    // The forward pass keeps the OUTER surface clear of the inner one at
    // each anchor correspondence, but a *different* inner vertex can
    // still slide between two anchored outer vertices and poke through
    // at bent joints — the shirt-through-blazer failure. Each pair's
    // inner layer therefore also gets containment anchors referencing
    // the outer surface: the transform shader clamps those vertices
    // back inside when their clearance along the outer normal exceeds
    // the rest-derived target (`rest_clearance + CONTAINMENT_SLACK`).
    // The target sits ABOVE the rest clearance so the constraint never
    // fires on the rest pose (however tight the layer gap) and only
    // pulls a vertex back once it drifts more than the slack beyond
    // its rest offset — preserving the authored layer separation.
    //
    // Containment anchors live in their own `containment_anchors` /
    // `containment_primitive_id` slots, so the middle layer of a 3+
    // layer stack carries clearance anchors against its inner
    // neighbour AND containment anchors against its outer neighbour
    // simultaneously. A primitive serves as the inner of at most one
    // pair — when two outer layers share an inner candidate, the pair
    // with the larger overlap count wins (sorted for determinism).
    const CONTAINMENT_SLACK: f32 = 0.002;

    let mut pairs: Vec<(usize, usize, usize)> = accepted_pairs.clone();
    pairs.sort_by(|a, b| b.2.cmp(&a.2));

    for &(outer_idx, inner_idx, _n_overlap) in &pairs {
        {
            let inner_check = &asset.meshes[candidates[inner_idx].mesh_idx].primitives
                [candidates[inner_idx].prim_idx];
            if inner_check.containment_anchors.is_some()
                || inner_check.containment_primitive_id.is_some()
            {
                continue;
            }
        }

        let outer_cand = &candidates[outer_idx];
        let inner_cand = &candidates[inner_idx];

        let cell_size = 0.03f32;
        let inv_cell = 1.0 / cell_size;
        let outer_grid = build_layer_grid(&outer_cand.world_verts, cell_size);

        let o_vd = asset.meshes[outer_cand.mesh_idx].primitives[outer_cand.prim_idx]
            .vertices
            .as_ref()
            .unwrap();
        let i_vd = asset.meshes[inner_cand.mesh_idx].primitives[inner_cand.prim_idx]
            .vertices
            .as_ref()
            .unwrap();

        let mut anchors = Vec::with_capacity(inner_cand.world_verts.len());
        let mut bound_count = 0usize;

        for (ii, &(ip, inrm)) in inner_cand.world_verts.iter().enumerate() {
            let best = query_layered_nearest(
                ip,
                inrm,
                &i_vd.joint_indices[ii],
                &i_vd.joint_weights[ii],
                &outer_grid,
                &outer_cand.world_verts,
                &o_vd.joint_indices,
                &o_vd.joint_weights,
                inv_cell,
                0.08,
                0.1,
            );

            if let Some(idx) = best {
                let (op, on) = outer_cand.world_verts[idx as usize];
                let diff = [ip[0] - op[0], ip[1] - op[1], ip[2] - op[2]];
                let rest_c = diff[0] * on[0] + diff[1] * on[1] + diff[2] * on[2];
                anchors.push(SkinAnchor {
                    body_vertex_idx: idx,
                    min_clearance: rest_c + CONTAINMENT_SLACK,
                    weight: 1.0,
                    mode: SKIN_ANCHOR_CONTAINMENT,
                });
                bound_count += 1;
            } else {
                anchors.push(SkinAnchor::default());
            }
        }

        if bound_count > 0 {
            info!(
                "clearance: established {} containment anchors on inner '{}' (prim {:?}) -> outer '{}'",
                bound_count, inner_cand.mesh_name, inner_cand.prim_id, outer_cand.mesh_name
            );
            let prim_mut = Arc::make_mut(
                &mut asset.meshes[candidates[inner_idx].mesh_idx].primitives
                    [candidates[inner_idx].prim_idx],
            );
            prim_mut.containment_anchors = Some(anchors);
            prim_mut.containment_primitive_id = Some(outer_cand.prim_id);
        }
    }

    // Phase 1b: innermost layers — containment-carrying prims whose
    // clearance slot is still free — get body-surface clearance as
    // well. The vest under the shirt hugs the skin, and nothing else
    // would otherwise keep it (or the skin beneath it) separated from
    // the body at bent joints.
    for cand in &candidates {
        let needs_body = {
            let prim = &asset.meshes[cand.mesh_idx].primitives[cand.prim_idx];
            prim.containment_anchors.is_some() && prim.skin_anchors.is_none()
        };
        if !needs_body {
            continue;
        }
        let (own_idx, own_w) = {
            let vd = asset.meshes[cand.mesh_idx].primitives[cand.prim_idx]
                .vertices
                .as_ref()
                .unwrap();
            (&vd.joint_indices, &vd.joint_weights)
        };
        let (anchors, bound_count) = anchor_prim_against_body(
            &cand.world_verts,
            own_idx,
            own_w,
            body_world,
            body_joint_indices,
            body_joint_weights,
            &body_grid_3cm,
        );
        if bound_count > 0 {
            info!(
                "clearance: generated {} body anchors for innermost layer '{}' (prim {:?})",
                bound_count, cand.mesh_name, cand.prim_id
            );
            let prim_mut =
                Arc::make_mut(&mut asset.meshes[cand.mesh_idx].primitives[cand.prim_idx]);
            prim_mut.skin_anchors = Some(anchors);
            prim_mut.body_primitive_id = Some(body_pid);
        }
    }
}
