#![allow(dead_code)]
use crate::asset::{AvatarAsset, PrimitiveId};

#[derive(Clone, Debug)]
pub struct PickingResult {
    pub primitive_id: PrimitiveId,
    pub triangle_index: u32,
    pub hit_point: [f32; 3],
    pub distance: f32,
    pub vertex_indices: [usize; 3],
}

#[derive(Clone, Copy, Debug)]
struct BvhAabb {
    min: [f32; 3],
    max: [f32; 3],
}

impl BvhAabb {
    fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
        }
    }

    fn from_point(p: [f32; 3]) -> Self {
        Self { min: p, max: p }
    }

    fn merge(&self, other: &Self) -> Self {
        Self {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    fn ray_intersect(&self, origin: [f32; 3], inv_dir: [f32; 3], max_dist: f32) -> bool {
        let mut tmin = 0.0f32;
        let mut tmax = max_dist;

        for i in 0..3 {
            if inv_dir[i].is_infinite() {
                if origin[i] < self.min[i] || origin[i] > self.max[i] {
                    return false;
                }
            } else {
                let t1 = (self.min[i] - origin[i]) * inv_dir[i];
                let t2 = (self.max[i] - origin[i]) * inv_dir[i];
                let (t_near, t_far) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
                tmin = tmin.max(t_near);
                tmax = tmax.min(t_far);
                if tmin > tmax {
                    return false;
                }
            }
        }

        tmax >= 0.0
    }
}

struct TriRecord {
    primitive_index: usize,
    triangle_index: u32,
    i0: usize,
    i1: usize,
    i2: usize,
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    prim_id: PrimitiveId,
}

enum BvhNode {
    Leaf {
        aabb: BvhAabb,
        start: usize,
        count: usize,
    },
    Inner {
        aabb: BvhAabb,
        left: Box<BvhNode>,
        right: Box<BvhNode>,
    },
}

pub struct Bvh {
    root: BvhNode,
    tris: Vec<TriRecord>,
}

impl Bvh {
    fn build(mut tris: Vec<TriRecord>) -> Self {
        let root = Self::build_recursive(&mut tris, 0, 0);
        Self { root, tris }
    }

    fn build_recursive(tris: &mut [TriRecord], base: usize, depth: usize) -> BvhNode {
        let aabb = tris.iter().fold(BvhAabb::empty(), |acc, t| {
            let ta = BvhAabb::from_point(t.v0)
                .merge(&BvhAabb::from_point(t.v1))
                .merge(&BvhAabb::from_point(t.v2));
            acc.merge(&ta)
        });

        if tris.len() <= 4 || depth >= 32 {
            return BvhNode::Leaf {
                aabb,
                start: base,
                count: tris.len(),
            };
        }

        let extent = [
            aabb.max[0] - aabb.min[0],
            aabb.max[1] - aabb.min[1],
            aabb.max[2] - aabb.min[2],
        ];
        let axis = if extent[0] >= extent[1] && extent[0] >= extent[2] {
            0usize
        } else if extent[1] >= extent[2] {
            1
        } else {
            2
        };

        let mid = tris.len() / 2;
        tris.select_nth_unstable_by(mid, |a, b| {
            let ca = (a.v0[axis] + a.v1[axis] + a.v2[axis])
                .partial_cmp(&(b.v0[axis] + b.v1[axis] + b.v2[axis]))
                .unwrap_or(std::cmp::Ordering::Equal);
            ca
        });

        let (left_slice, right_slice) = tris.split_at_mut(mid);
        let left = Self::build_recursive(left_slice, base, depth + 1);
        let right = Self::build_recursive(right_slice, base + mid, depth + 1);

        BvhNode::Inner {
            aabb,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    fn query(&self, origin: [f32; 3], dir: [f32; 3], max_dist: f32) -> Option<PickingResult> {
        let inv_dir = [
            if dir[0].abs() > 1e-12 {
                1.0 / dir[0]
            } else {
                f32::INFINITY
            },
            if dir[1].abs() > 1e-12 {
                1.0 / dir[1]
            } else {
                f32::INFINITY
            },
            if dir[2].abs() > 1e-12 {
                1.0 / dir[2]
            } else {
                f32::INFINITY
            },
        ];

        let mut best: Option<PickingResult> = None;
        let best_dist = self.query_node(
            &self.root, origin, dir, &inv_dir, max_dist, &self.tris, &mut best,
        );
        if best_dist < f32::MAX {
            best
        } else {
            None
        }
    }

    fn query_node(
        &self,
        node: &BvhNode,
        origin: [f32; 3],
        dir: [f32; 3],
        inv_dir: &[f32; 3],
        max_dist: f32,
        tris: &[TriRecord],
        best: &mut Option<PickingResult>,
    ) -> f32 {
        match node {
            BvhNode::Leaf { aabb, start, count } => {
                if !aabb.ray_intersect(origin, *inv_dir, max_dist) {
                    return f32::MAX;
                }
                let mut local_best = max_dist;
                for i in *start..(*start + *count) {
                    let t = &tris[i];
                    if let Some((t_dist, _u, _v)) =
                        ray_triangle_intersect(origin, dir, t.v0, t.v1, t.v2)
                    {
                        if t_dist >= 0.0 && t_dist < local_best {
                            let is_better = match best {
                                None => true,
                                Some(b) => t_dist < b.distance,
                            };
                            if is_better {
                                *best = Some(PickingResult {
                                    primitive_id: t.prim_id,
                                    triangle_index: t.triangle_index,
                                    hit_point: [
                                        origin[0] + dir[0] * t_dist,
                                        origin[1] + dir[1] * t_dist,
                                        origin[2] + dir[2] * t_dist,
                                    ],
                                    distance: t_dist,
                                    vertex_indices: [t.i0, t.i1, t.i2],
                                });
                                local_best = t_dist;
                            }
                        }
                    }
                }
                local_best
            }
            BvhNode::Inner { aabb, left, right } => {
                if !aabb.ray_intersect(origin, *inv_dir, max_dist) {
                    return f32::MAX;
                }
                let d1 = self.query_node(left, origin, dir, inv_dir, max_dist, tris, best);
                let current_max = max_dist.min(d1);
                let d2 = self.query_node(right, origin, dir, inv_dir, current_max, tris, best);
                d1.min(d2)
            }
        }
    }
}

fn build_bvh(avatar: &AvatarAsset) -> Bvh {
    let mut tris = Vec::new();
    for (pi, mesh) in avatar.meshes.iter().enumerate() {
        for prim in &mesh.primitives {
            let vertices = match &prim.vertices {
                Some(v) => v,
                None => continue,
            };
            let indices = match &prim.indices {
                Some(i) => i,
                None => continue,
            };
            let tri_count = indices.len() / 3;
            for t in 0..tri_count {
                let i0 = indices[t * 3] as usize;
                let i1 = indices[t * 3 + 1] as usize;
                let i2 = indices[t * 3 + 2] as usize;
                if i0 >= vertices.positions.len()
                    || i1 >= vertices.positions.len()
                    || i2 >= vertices.positions.len()
                {
                    continue;
                }
                tris.push(TriRecord {
                    primitive_index: pi,
                    triangle_index: t as u32,
                    i0,
                    i1,
                    i2,
                    v0: vertices.positions[i0],
                    v1: vertices.positions[i1],
                    v2: vertices.positions[i2],
                    prim_id: prim.id,
                });
            }
        }
    }
    Bvh::build(tris)
}

pub fn pick_mesh(
    avatar: &AvatarAsset,
    ray_origin: [f32; 3],
    ray_direction: [f32; 3],
    max_distance: f32,
) -> Option<PickingResult> {
    let dir_len = (ray_direction[0] * ray_direction[0]
        + ray_direction[1] * ray_direction[1]
        + ray_direction[2] * ray_direction[2])
        .sqrt();
    if dir_len < 1e-8 {
        return None;
    }
    let dir = [
        ray_direction[0] / dir_len,
        ray_direction[1] / dir_len,
        ray_direction[2] / dir_len,
    ];

    let bvh = build_bvh(avatar);
    bvh.query(ray_origin, dir, max_distance)
}

pub fn pick_mesh_with_bvh(
    bvh: &Bvh,
    ray_origin: [f32; 3],
    ray_direction: [f32; 3],
    max_distance: f32,
) -> Option<PickingResult> {
    let dir_len = (ray_direction[0] * ray_direction[0]
        + ray_direction[1] * ray_direction[1]
        + ray_direction[2] * ray_direction[2])
        .sqrt();
    if dir_len < 1e-8 {
        return None;
    }
    let dir = [
        ray_direction[0] / dir_len,
        ray_direction[1] / dir_len,
        ray_direction[2] / dir_len,
    ];
    bvh.query(ray_origin, dir, max_distance)
}

pub fn build_picking_bvh(avatar: &AvatarAsset) -> Bvh {
    build_bvh(avatar)
}

fn ray_triangle_intersect(
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
) -> Option<(f32, f32, f32)> {
    let epsilon = 1e-7;

    let edge1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let edge2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

    let h = cross(ray_dir, edge2);
    let a = dot(edge1, h);

    if a > -epsilon && a < epsilon {
        return None;
    }

    let f = 1.0 / a;
    let s = [
        ray_origin[0] - v0[0],
        ray_origin[1] - v0[1],
        ray_origin[2] - v0[2],
    ];
    let u = f * dot(s, h);
    if u < 0.0 || u > 1.0 {
        return None;
    }

    let q = cross(s, edge1);
    let v = f * dot(ray_dir, q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = f * dot(edge2, q);
    if t > epsilon {
        Some((t, u, v))
    } else {
        None
    }
}

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bvh_single_triangle() {
        let aabb = BvhAabb {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        let inv_dir = [1.0f32, 1.0f32, 1.0f32];
        assert!(aabb.ray_intersect([0.0, 0.0, 0.0], inv_dir, 10.0));
        assert!(!aabb.ray_intersect([2.0, 2.0, -1.0], inv_dir, 10.0));
    }

    #[test]
    fn test_bvh_ray_intersect_behind() {
        let aabb = BvhAabb {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        let inv_dir = [0.0, 0.0, -1.0];
        assert!(!aabb.ray_intersect([0.5, 0.5, 2.0], inv_dir, 1.0));
    }
}
