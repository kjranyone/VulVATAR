#![allow(dead_code)]
use crate::asset::{ClothSimulationMesh, ColliderAsset, ColliderShape, Mat4, SkeletonAsset, Vec3};

/// A single colored line segment in world space.
#[derive(Clone, Debug)]
pub struct DebugLine {
    pub start: [f32; 3],
    pub end: [f32; 3],
    pub color: [f32; 4],
}

/// A debug sphere (rendered as a circle in the CPU fallback path).
#[derive(Clone, Debug)]
pub struct DebugSphere {
    pub center: [f32; 3],
    pub radius: f32,
    pub color: [f32; 4],
}

/// Accumulated debug draw primitives for one frame.
#[derive(Clone, Debug, Default)]
pub struct DebugDrawList {
    pub lines: Vec<DebugLine>,
    pub spheres: Vec<DebugSphere>,
}

impl DebugDrawList {
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge another draw list into this one.
    pub fn append(&mut self, other: &DebugDrawList) {
        self.lines.extend_from_slice(&other.lines);
        self.spheres.extend_from_slice(&other.spheres);
    }
}

// ---------------------------------------------------------------------------
// Skeleton debug
// ---------------------------------------------------------------------------

/// Generate lines between parent-child bone pairs, using the provided skinning
/// matrices (one per node, in world space) to determine bone positions.
///
/// `pose_matrices` should contain a world-space transform per skeleton node
/// (index corresponds to position in `skeleton.nodes`).  If the pose slice is
/// shorter than the node list, the missing bones are skipped.
pub fn build_skeleton_debug(skeleton: &SkeletonAsset, pose_matrices: &[Mat4]) -> DebugDrawList {
    let mut list = DebugDrawList::new();

    // Bone color: cyan-ish.
    let bone_color: [f32; 4] = [0.0, 0.9, 0.9, 1.0];
    // Joint indicator color: yellow.
    let joint_color: [f32; 4] = [1.0, 0.9, 0.1, 1.0];

    // Build a quick index map: NodeId -> index in `skeleton.nodes`.
    let node_index: std::collections::HashMap<crate::asset::NodeId, usize> = skeleton
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id, i))
        .collect();

    for (child_idx, node) in skeleton.nodes.iter().enumerate() {
        if child_idx >= pose_matrices.len() {
            break;
        }
        let child_pos = translation_from_mat4(&pose_matrices[child_idx]);

        // Draw a small cross at the joint position.
        let arm = 0.005;
        list.lines.push(DebugLine {
            start: [child_pos[0] - arm, child_pos[1], child_pos[2]],
            end: [child_pos[0] + arm, child_pos[1], child_pos[2]],
            color: joint_color,
        });
        list.lines.push(DebugLine {
            start: [child_pos[0], child_pos[1] - arm, child_pos[2]],
            end: [child_pos[0], child_pos[1] + arm, child_pos[2]],
            color: joint_color,
        });

        // Draw a line from parent to this child.
        if let Some(parent_id) = node.parent {
            if let Some(&parent_idx) = node_index.get(&parent_id) {
                if parent_idx < pose_matrices.len() {
                    let parent_pos = translation_from_mat4(&pose_matrices[parent_idx]);
                    list.lines.push(DebugLine {
                        start: parent_pos,
                        end: child_pos,
                        color: bone_color,
                    });
                }
            }
        }
    }

    list
}

// ---------------------------------------------------------------------------
// Collider debug
// ---------------------------------------------------------------------------

/// Generate debug spheres (and capsule endpoints) from collider assets.
///
/// `pose_matrices` are the world-space joint transforms, indexed by skeleton
/// node order.  We look up each collider's node to find its world position.
pub fn build_collider_debug(
    colliders: &[ColliderAsset],
    skeleton: &SkeletonAsset,
    pose_matrices: &[Mat4],
) -> DebugDrawList {
    let mut list = DebugDrawList::new();

    let collider_color: [f32; 4] = [0.1, 1.0, 0.3, 0.6];
    let capsule_line_color: [f32; 4] = [0.1, 1.0, 0.3, 0.4];

    let node_index: std::collections::HashMap<crate::asset::NodeId, usize> = skeleton
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id, i))
        .collect();

    for collider in colliders {
        let idx = match node_index.get(&collider.node) {
            Some(&i) if i < pose_matrices.len() => i,
            _ => continue,
        };

        let mat = &pose_matrices[idx];
        // Transform the collider offset through the node's global transform.
        let center = transform_point(mat, &collider.offset);

        match &collider.shape {
            ColliderShape::Sphere { radius } => {
                list.spheres.push(DebugSphere {
                    center,
                    radius: *radius,
                    color: collider_color,
                });
            }
            ColliderShape::Capsule { radius, height } => {
                // Transform capsule endpoints using the node's global
                // rotation, not just a Y-axis offset.
                let half_h = height * 0.5;
                let local_top = [
                    collider.offset[0],
                    collider.offset[1] + half_h,
                    collider.offset[2],
                ];
                let local_bottom = [
                    collider.offset[0],
                    collider.offset[1] - half_h,
                    collider.offset[2],
                ];
                let top = transform_point(mat, &local_top);
                let bottom = transform_point(mat, &local_bottom);
                list.spheres.push(DebugSphere {
                    center: top,
                    radius: *radius,
                    color: collider_color,
                });
                list.spheres.push(DebugSphere {
                    center: bottom,
                    radius: *radius,
                    color: collider_color,
                });
                list.lines.push(DebugLine {
                    start: top,
                    end: bottom,
                    color: capsule_line_color,
                });
            }
        }
    }

    list
}

// ---------------------------------------------------------------------------
// Cloth simulation mesh debug (wireframe)
// ---------------------------------------------------------------------------

/// Generate wireframe lines for a cloth simulation mesh.
///
/// If `sim_positions` is non-empty it is used instead of the rest positions
/// stored in `cloth_mesh.vertices`, allowing display of the deformed state.
pub fn build_cloth_mesh_debug(
    cloth_mesh: &ClothSimulationMesh,
    sim_positions: &[Vec3],
) -> DebugDrawList {
    let mut list = DebugDrawList::new();

    let wire_color: [f32; 4] = [1.0, 0.5, 0.1, 0.8];
    let pinned_color: [f32; 4] = [1.0, 0.1, 0.1, 1.0];

    let positions: Vec<Vec3> = if !sim_positions.is_empty() {
        sim_positions.to_vec()
    } else {
        cloth_mesh.vertices.iter().map(|v| v.position).collect()
    };

    // Draw triangles as wireframe edges.
    let tri_count = cloth_mesh.indices.len() / 3;
    for tri in 0..tri_count {
        let i0 = cloth_mesh.indices[tri * 3] as usize;
        let i1 = cloth_mesh.indices[tri * 3 + 1] as usize;
        let i2 = cloth_mesh.indices[tri * 3 + 2] as usize;

        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
            continue;
        }

        let edges = [(i0, i1), (i1, i2), (i2, i0)];
        for (a, b) in edges {
            list.lines.push(DebugLine {
                start: positions[a],
                end: positions[b],
                color: wire_color,
            });
        }
    }

    // Highlight pinned vertices with small spheres.
    for (i, vert) in cloth_mesh.vertices.iter().enumerate() {
        if vert.pinned && i < positions.len() {
            list.spheres.push(DebugSphere {
                center: positions[i],
                radius: 0.003,
                color: pinned_color,
            });
        }
    }

    list
}

// ---------------------------------------------------------------------------
// Normal debug
// ---------------------------------------------------------------------------

/// Draw short lines from each vertex in the direction of its normal.
pub fn build_normal_debug(positions: &[Vec3], normals: &[Vec3]) -> DebugDrawList {
    let mut list = DebugDrawList::new();

    let normal_color: [f32; 4] = [0.3, 0.3, 1.0, 0.8];
    let normal_length: f32 = 0.01;

    let count = positions.len().min(normals.len());
    for i in 0..count {
        let p = positions[i];
        let n = normals[i];
        let end = [
            p[0] + n[0] * normal_length,
            p[1] + n[1] * normal_length,
            p[2] + n[2] * normal_length,
        ];
        list.lines.push(DebugLine {
            start: p,
            end,
            color: normal_color,
        });
    }

    list
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract translation from a row-major 4x4 matrix.
fn translation_from_mat4(m: &Mat4) -> Vec3 {
    [m[3][0], m[3][1], m[3][2]]
}

/// Transform a point by the full 4x4 matrix (rotation + translation).
/// Assumes row-major storage where row 3 is translation.
fn transform_point(m: &Mat4, p: &Vec3) -> Vec3 {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}
