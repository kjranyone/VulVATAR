use std::collections::{BTreeSet, HashMap, HashSet};

use crate::asset::{
    AvatarAsset, ClothSimVertex, ClothSimulationMesh, DistanceConstraint, MaterialId, PrimitiveId,
};

#[derive(Clone, Debug)]
pub struct RegionSelection {
    pub target_primitive: PrimitiveId,
    pub selected_vertex_range: (u32, u32),
    /// Explicit set of selected vertex indices within the target primitive.
    /// When non-empty this takes precedence over `selected_vertex_range`.
    pub selected_vertices: BTreeSet<usize>,
}

#[derive(Clone, Debug)]
pub enum AuthoringValidationError {
    UnresolvedNodeRef { node_name: String },
    EmptySimulationMesh,
}

// ---------------------------------------------------------------------------
// Task 3: Simulation mesh generation
// ---------------------------------------------------------------------------

/// Generate a cloth simulation mesh from a subset of vertices on a mesh
/// primitive.
///
/// The algorithm:
/// 1. Extract selected vertices from the primitive's vertex data.
/// 2. Build triangle connectivity -- only triangles where all 3 vertices are
///    in `selected_vertices`.
/// 3. Compute rest-length distance constraints for every edge.
/// 4. Mark boundary vertices (connected to at least one non-selected vertex
///    in the original mesh) as pinned.
pub fn generate_sim_mesh(
    avatar: &AvatarAsset,
    primitive_id: PrimitiveId,
    selected_vertices: &[usize],
) -> Option<ClothSimulationMesh> {
    // Locate the primitive.
    let primitive = avatar
        .meshes
        .iter()
        .flat_map(|m| m.primitives.iter())
        .find(|p| p.id == primitive_id)?;

    let vertex_data = primitive.vertices.as_ref()?;
    let indices = primitive.indices.as_ref()?;

    let selected_set: HashSet<usize> = selected_vertices.iter().copied().collect();

    // Build old-index -> new-index mapping for selected vertices.
    let mut old_to_new: HashMap<usize, usize> = HashMap::new();
    let mut sim_vertices: Vec<ClothSimVertex> = Vec::new();

    for (new_idx, &old_idx) in selected_vertices.iter().enumerate() {
        if old_idx >= vertex_data.positions.len() {
            continue;
        }
        old_to_new.insert(old_idx, new_idx);
        sim_vertices.push(ClothSimVertex {
            position: vertex_data.positions[old_idx],
            normal: if old_idx < vertex_data.normals.len() {
                vertex_data.normals[old_idx]
            } else {
                [0.0, 1.0, 0.0]
            },
            uv: if old_idx < vertex_data.uvs.len() {
                vertex_data.uvs[old_idx]
            } else {
                [0.0, 0.0]
            },
            pinned: false, // will be set below
        });
    }

    if sim_vertices.is_empty() {
        return None;
    }

    // Build triangle list: only triangles with all 3 verts selected.
    let tri_count = indices.len() / 3;
    let mut sim_indices: Vec<u32> = Vec::new();
    // Track edges for constraint generation (using new indices).
    let mut edge_set: HashSet<(u32, u32)> = HashSet::new();

    for t in 0..tri_count {
        let i0 = indices[t * 3] as usize;
        let i1 = indices[t * 3 + 1] as usize;
        let i2 = indices[t * 3 + 2] as usize;

        if let (Some(&n0), Some(&n1), Some(&n2)) = (
            old_to_new.get(&i0),
            old_to_new.get(&i1),
            old_to_new.get(&i2),
        ) {
            sim_indices.push(n0 as u32);
            sim_indices.push(n1 as u32);
            sim_indices.push(n2 as u32);

            // Record edges (canonical order: smaller index first).
            for (a, b) in [(n0, n1), (n1, n2), (n2, n0)] {
                let edge = if a < b {
                    (a as u32, b as u32)
                } else {
                    (b as u32, a as u32)
                };
                edge_set.insert(edge);
            }
        }
    }

    // Compute rest lengths and build distance constraints.
    let mut rest_lengths: Vec<f32> = Vec::new();
    let mut distance_constraints: Vec<DistanceConstraint> = Vec::new();

    for (a, b) in &edge_set {
        let pa = sim_vertices[*a as usize].position;
        let pb = sim_vertices[*b as usize].position;
        let dx = pb[0] - pa[0];
        let dy = pb[1] - pa[1];
        let dz = pb[2] - pa[2];
        let rest = (dx * dx + dy * dy + dz * dz).sqrt();
        rest_lengths.push(rest);
        distance_constraints.push(DistanceConstraint {
            indices: [*a, *b],
            rest_length: rest,
            stiffness: 1.0,
        });
    }

    // Determine boundary vertices: selected vertices that share an edge with
    // at least one non-selected vertex in the original mesh.
    let mut boundary_old: HashSet<usize> = HashSet::new();
    for t in 0..tri_count {
        let tri_verts = [
            indices[t * 3] as usize,
            indices[t * 3 + 1] as usize,
            indices[t * 3 + 2] as usize,
        ];
        let any_in = tri_verts.iter().any(|v| selected_set.contains(v));
        let any_out = tri_verts.iter().any(|v| !selected_set.contains(v));
        if any_in && any_out {
            for v in &tri_verts {
                if selected_set.contains(v) {
                    boundary_old.insert(*v);
                }
            }
        }
    }

    // Mark boundary vertices as pinned.
    for &old_idx in &boundary_old {
        if let Some(&new_idx) = old_to_new.get(&old_idx) {
            sim_vertices[new_idx].pinned = true;
        }
    }

    Some(ClothSimulationMesh {
        vertices: sim_vertices,
        indices: sim_indices,
        rest_lengths,
        attachment_classes: vec![],
        region_tags: vec![],
    })
}

// ---------------------------------------------------------------------------
// Task 4: Region selection basics
// ---------------------------------------------------------------------------

impl RegionSelection {
    /// Select all vertices of primitives that use the given material.
    pub fn select_by_material(
        avatar: &AvatarAsset,
        material_id: MaterialId,
    ) -> Option<RegionSelection> {
        // Find the first primitive using this material and select all its
        // vertices.  In a multi-primitive scenario the caller could iterate,
        // but for the POC we target one primitive at a time.
        for mesh in &avatar.meshes {
            for prim in &mesh.primitives {
                if prim.material_id == material_id {
                    let vert_count = prim.vertex_count as usize;
                    let selected: BTreeSet<usize> = (0..vert_count).collect();
                    return Some(RegionSelection {
                        target_primitive: prim.id,
                        selected_vertex_range: (0, prim.vertex_count),
                        selected_vertices: selected,
                    });
                }
            }
        }
        None
    }

    /// Expand the selection to include all vertices adjacent to the current
    /// selection via shared edges in the mesh topology.
    pub fn grow_selection(&mut self, avatar: &AvatarAsset) {
        let primitive = avatar
            .meshes
            .iter()
            .flat_map(|m| m.primitives.iter())
            .find(|p| p.id == self.target_primitive);

        let primitive = match primitive {
            Some(p) => p,
            None => return,
        };

        let indices = match &primitive.indices {
            Some(i) => i,
            None => return,
        };

        // Ensure selected_vertices is populated from the range if empty.
        if self.selected_vertices.is_empty() {
            let (start, end) = self.selected_vertex_range;
            self.selected_vertices = (start as usize..end as usize).collect();
        }

        // Build adjacency: for each selected vertex, find neighbors.
        let mut new_selection = self.selected_vertices.clone();
        let tri_count = indices.len() / 3;

        for t in 0..tri_count {
            let tri_verts = [
                indices[t * 3] as usize,
                indices[t * 3 + 1] as usize,
                indices[t * 3 + 2] as usize,
            ];
            let has_selected = tri_verts.iter().any(|v| self.selected_vertices.contains(v));
            if has_selected {
                // Add all triangle vertices to the new selection.
                for v in &tri_verts {
                    if *v < primitive.vertex_count as usize {
                        new_selection.insert(*v);
                    }
                }
            }
        }

        self.selected_vertices = new_selection;
        // Update range to span the full set.
        if let (Some(&first), Some(&last)) = (
            self.selected_vertices.iter().next(),
            self.selected_vertices.iter().next_back(),
        ) {
            self.selected_vertex_range = (first as u32, (last + 1) as u32);
        }
    }

    /// Shrink the selection by removing boundary vertices (vertices that are
    /// adjacent to at least one non-selected vertex).
    pub fn shrink_selection(&mut self, avatar: &AvatarAsset) {
        let primitive = avatar
            .meshes
            .iter()
            .flat_map(|m| m.primitives.iter())
            .find(|p| p.id == self.target_primitive);

        let primitive = match primitive {
            Some(p) => p,
            None => return,
        };

        let indices = match &primitive.indices {
            Some(i) => i,
            None => return,
        };

        // Ensure selected_vertices is populated.
        if self.selected_vertices.is_empty() {
            let (start, end) = self.selected_vertex_range;
            self.selected_vertices = (start as usize..end as usize).collect();
        }

        // Find boundary vertices: selected vertices adjacent to non-selected.
        let mut boundary: HashSet<usize> = HashSet::new();
        let tri_count = indices.len() / 3;

        for t in 0..tri_count {
            let tri_verts = [
                indices[t * 3] as usize,
                indices[t * 3 + 1] as usize,
                indices[t * 3 + 2] as usize,
            ];
            let any_in = tri_verts.iter().any(|v| self.selected_vertices.contains(v));
            let any_out = tri_verts
                .iter()
                .any(|v| !self.selected_vertices.contains(v));
            if any_in && any_out {
                for v in &tri_verts {
                    if self.selected_vertices.contains(v) {
                        boundary.insert(*v);
                    }
                }
            }
        }

        // Remove boundary vertices.
        for v in &boundary {
            self.selected_vertices.remove(v);
        }

        // Update range.
        if let (Some(&first), Some(&last)) = (
            self.selected_vertices.iter().next(),
            self.selected_vertices.iter().next_back(),
        ) {
            self.selected_vertex_range = (first as u32, (last + 1) as u32);
        } else {
            self.selected_vertex_range = (0, 0);
        }
    }
}
