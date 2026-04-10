use std::collections::{HashMap, HashSet};

use crate::asset::{ClothAsset, ColliderShape, Mat4, Vec3};

// ---------------------------------------------------------------------------
// Spatial hash grid for self-collision broadphase
// ---------------------------------------------------------------------------

/// Uniform-grid spatial hash for fast neighbour queries among cloth particles.
#[derive(Clone, Debug)]
pub struct SpatialHashGrid {
    cell_size: f32,
    inv_cell_size: f32,
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SpatialHashGrid {
    pub fn new(cell_size: f32) -> Self {
        let cell_size = cell_size.max(1e-6);
        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            cells: HashMap::new(),
        }
    }

    /// Remove all entries and release the underlying bucket storage.
    ///
    /// This fully resets the HashMap rather than just clearing individual
    /// buckets, which prevents unbounded growth when particles move through
    /// many different cells over time.
    pub fn clear(&mut self) {
        self.cells.clear();
    }

    /// Insert a particle index at the given world-space position.
    /// Skips particles with NaN or Infinity positions to avoid corrupting the grid.
    pub fn insert(&mut self, index: usize, position: Vec3) {
        if position.iter().any(|v| v.is_nan() || v.is_infinite()) {
            return;
        }
        let key = self.cell_key(position);
        self.cells.entry(key).or_insert_with(Vec::new).push(index);
    }

    /// Return indices of all particles whose cell overlaps the query sphere.
    ///
    /// The caller must still perform exact distance checks; this is a
    /// broadphase query that may return false positives but no false negatives.
    pub fn query_neighbors(&self, position: Vec3, radius: f32) -> Vec<usize> {
        let mut result = Vec::new();
        let min_cell = self.cell_key([
            position[0] - radius,
            position[1] - radius,
            position[2] - radius,
        ]);
        let max_cell = self.cell_key([
            position[0] + radius,
            position[1] + radius,
            position[2] + radius,
        ]);

        for cx in min_cell.0..=max_cell.0 {
            for cy in min_cell.1..=max_cell.1 {
                for cz in min_cell.2..=max_cell.2 {
                    if let Some(bucket) = self.cells.get(&(cx, cy, cz)) {
                        result.extend_from_slice(bucket);
                    }
                }
            }
        }
        result
    }

    #[inline]
    fn cell_key(&self, position: Vec3) -> (i32, i32, i32) {
        (
            (position[0] * self.inv_cell_size).floor() as i32,
            (position[1] * self.inv_cell_size).floor() as i32,
            (position[2] * self.inv_cell_size).floor() as i32,
        )
    }
}

// ---------------------------------------------------------------------------
// Particle
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ClothParticle {
    pub position: Vec3,
    pub prev_position: Vec3,
    pub velocity: Vec3,
    pub inv_mass: f32, // 0.0 = pinned / infinite mass
    pub pinned: bool,
    /// If pinned, index into `ClothSimState::pin_targets`.
    pub pin_target_index: Option<usize>,
}

impl ClothParticle {
    pub fn new(pos: Vec3, pinned: bool) -> Self {
        Self {
            position: pos,
            prev_position: pos,
            velocity: [0.0, 0.0, 0.0],
            inv_mass: if pinned { 0.0 } else { 1.0 },
            pinned,
            pin_target_index: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Distance constraint (edge)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ClothDistanceConstraint {
    pub a: usize,
    pub b: usize,
    pub rest_length: f32,
    pub stiffness: f32,
}

// ---------------------------------------------------------------------------
// Bend constraint (runtime)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ClothBendConstraint {
    /// Three vertex indices: p0 is the shared-edge vertex, p1 and p2 are the
    /// wing vertices, forming two triangles sharing the edge (p0, p1) and
    /// (p0, p2).  In the isometric bending model with 3 indices the angle is
    /// measured at p0 between edges p0->p1 and p0->p2.
    pub p0: usize,
    pub p1: usize,
    pub p2: usize,
    pub rest_angle: f32,
    pub stiffness: f32,
}

// ---------------------------------------------------------------------------
// Collider snapshot (world-space, resolved each frame)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ResolvedSphereCollider {
    pub center: Vec3,
    pub radius: f32,
}

/// A resolved world-space collider that supports both sphere and capsule shapes.
#[derive(Clone, Debug)]
pub enum ResolvedCollider {
    Sphere {
        center: Vec3,
        radius: f32,
    },
    Capsule {
        center: Vec3,
        radius: f32,
        half_height: f32,
        /// Normalized axis direction of the capsule (world-space).
        axis: Vec3,
    },
}

// ---------------------------------------------------------------------------
// Pin target: world-space position that a pinned vertex should follow
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PinTarget {
    /// Skeleton node index the pin is attached to.
    pub node_index: usize,
    /// Local offset from that node.
    pub offset: Vec3,
    /// Indices into the particle array that this pin drives.
    pub particle_indices: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Runtime simulation state
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ClothSimState {
    pub particles: Vec<ClothParticle>,
    pub distance_constraints: Vec<ClothDistanceConstraint>,
    pub bend_constraints: Vec<ClothBendConstraint>,
    pub pin_targets: Vec<PinTarget>,
    pub solver_iterations: u32,
    pub gravity: Vec3,
    pub damping: f32,
    pub collision_margin: f32,
    pub wind_response: f32,
    pub wind_direction: Vec3,
    /// Triangle indices for normal computation (triplets of vertex indices).
    pub triangle_indices: Vec<u32>,
    /// Per-vertex normals computed after constraint solving (M2).
    pub computed_normals: Vec<Vec3>,
    pub initialized: bool,

    // -- Self-collision fields --
    /// Whether self-collision is enabled for this cloth (from ClothSolverParams).
    pub self_collision: bool,
    /// Particle radius used for self-collision detection.
    /// Two particles collide when their distance is less than `2 * self_collision_radius`.
    pub self_collision_radius: f32,
    /// Set of particle-index pairs that are connected by a distance constraint.
    /// These pairs are expected to be close and should be skipped during
    /// self-collision resolution.  Stored as `(min, max)` for canonical ordering.
    pub connected_pairs: HashSet<(usize, usize)>,
    /// Reusable spatial hash grid (allocated once, cleared each frame).
    pub spatial_hash: SpatialHashGrid,
}

impl ClothSimState {
    /// Build simulation state from a `ClothAsset`.
    pub fn from_asset(asset: &ClothAsset) -> Self {
        let sim_mesh = &asset.simulation_mesh;

        // --- particles ---
        let mut particles: Vec<ClothParticle> = sim_mesh
            .vertices
            .iter()
            .map(|v| ClothParticle::new(v.position, v.pinned))
            .collect();

        // --- distance constraints ---
        let distance_constraints: Vec<ClothDistanceConstraint> = asset
            .constraints
            .distance_constraints
            .iter()
            .map(|dc| ClothDistanceConstraint {
                a: dc.indices[0] as usize,
                b: dc.indices[1] as usize,
                rest_length: dc.rest_length,
                stiffness: dc.stiffness,
            })
            .collect();

        // --- pin targets ---
        let mut pin_targets: Vec<PinTarget> = Vec::new();
        for pin in &asset.pins {
            let node_idx = pin.binding_node.id.0 as usize;
            let pt_idx = pin_targets.len();
            let particle_indices: Vec<usize> =
                pin.sim_vertex_indices.iter().map(|&i| i as usize).collect();
            // Mark particles as pinned and link them to this target.
            for &pi in &particle_indices {
                if pi < particles.len() {
                    particles[pi].pinned = true;
                    particles[pi].inv_mass = 0.0;
                    particles[pi].pin_target_index = Some(pt_idx);
                }
            }
            pin_targets.push(PinTarget {
                node_index: node_idx,
                offset: pin.offset,
                particle_indices,
            });
        }

        // --- bend constraints ---
        let bend_constraints: Vec<ClothBendConstraint> = asset
            .constraints
            .bend_constraints
            .iter()
            .map(|bc| {
                let p0 = bc.indices[0] as usize;
                let p1 = bc.indices[1] as usize;
                let p2 = bc.indices[2] as usize;
                // Compute rest angle between edges p0->p1 and p0->p2
                let rest_angle = if p0 < particles.len()
                    && p1 < particles.len()
                    && p2 < particles.len()
                {
                    let v0 = particles[p0].position;
                    let v1 = particles[p1].position;
                    let v2 = particles[p2].position;
                    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
                    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
                    let dot = e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2];
                    let cross = [
                        e1[1] * e2[2] - e1[2] * e2[1],
                        e1[2] * e2[0] - e1[0] * e2[2],
                        e1[0] * e2[1] - e1[1] * e2[0],
                    ];
                    let cross_len =
                        (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
                    cross_len.atan2(dot)
                } else {
                    std::f32::consts::PI
                };
                ClothBendConstraint {
                    p0,
                    p1,
                    p2,
                    rest_angle,
                    stiffness: bc.stiffness,
                }
            })
            .collect();

        // --- triangle indices (copy from simulation mesh for normal computation) ---
        let triangle_indices = sim_mesh.indices.clone();

        let params = &asset.solver_params;

        let particle_count = particles.len();

        // Build connected-pair lookup from distance constraints so that
        // self-collision can skip pairs that share an edge.
        let mut connected_pairs = HashSet::new();
        for dc in &distance_constraints {
            let lo = dc.a.min(dc.b);
            let hi = dc.a.max(dc.b);
            connected_pairs.insert((lo, hi));
        }

        let self_collision_radius: f32 = 0.01;
        let spatial_hash = SpatialHashGrid::new(self_collision_radius * 4.0);

        Self {
            particles,
            distance_constraints,
            bend_constraints,
            pin_targets,
            solver_iterations: params.iterations.max(1),
            gravity: [0.0, -9.81 * params.gravity_scale, 0.0],
            damping: params.damping,
            collision_margin: params.collision_margin,
            wind_response: params.wind_response,
            wind_direction: [1.0, 0.0, 0.0], // default wind direction
            triangle_indices,
            computed_normals: vec![[0.0; 3]; particle_count],
            initialized: true,
            self_collision: params.self_collision,
            self_collision_radius,
            connected_pairs,
            spatial_hash,
        }
    }

    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }
}

// ---------------------------------------------------------------------------
// Temporary per-frame buffers (reusable scratch space)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ClothSimTempBuffers {
    pub temp_positions: Vec<Vec3>,
    pub correction_accumulator: Vec<Vec3>,
    pub correction_counts: Vec<u32>,
}

impl ClothSimTempBuffers {
    pub fn new(particle_count: usize) -> Self {
        Self {
            temp_positions: vec![[0.0; 3]; particle_count],
            correction_accumulator: vec![[0.0; 3]; particle_count],
            correction_counts: vec![0; particle_count],
        }
    }

    pub fn resize(&mut self, particle_count: usize) {
        self.temp_positions.resize(particle_count, [0.0; 3]);
        self.correction_accumulator
            .resize(particle_count, [0.0; 3]);
        self.correction_counts.resize(particle_count, 0);
    }

    pub fn clear(&mut self) {
        for v in self.correction_accumulator.iter_mut() {
            *v = [0.0; 3];
        }
        for c in self.correction_counts.iter_mut() {
            *c = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: resolve colliders to world-space spheres
// ---------------------------------------------------------------------------

pub fn resolve_colliders(
    colliders: &[crate::asset::ColliderAsset],
    global_transforms: &[Mat4],
) -> Vec<ResolvedCollider> {
    let mut out = Vec::with_capacity(colliders.len());
    for collider in colliders {
        let node_idx = collider.node.0 as usize;
        if node_idx >= global_transforms.len() {
            continue;
        }
        let mat = &global_transforms[node_idx];
        // Transform offset into world space: world_pos = mat * [ox, oy, oz, 1]
        let [ox, oy, oz] = collider.offset;
        let wx = mat[0][0] * ox + mat[1][0] * oy + mat[2][0] * oz + mat[3][0];
        let wy = mat[0][1] * ox + mat[1][1] * oy + mat[2][1] * oz + mat[3][1];
        let wz = mat[0][2] * ox + mat[1][2] * oy + mat[2][2] * oz + mat[3][2];
        let center = [wx, wy, wz];

        match collider.shape {
            ColliderShape::Sphere { radius } => {
                out.push(ResolvedCollider::Sphere { center, radius });
            }
            ColliderShape::Capsule { radius, height } => {
                // Capsule axis runs along local Y of the collider node.
                let up = [
                    mat[0][0] * 0.0 + mat[1][0] * 1.0 + mat[2][0] * 0.0,
                    mat[0][1] * 0.0 + mat[1][1] * 1.0 + mat[2][1] * 0.0,
                    mat[0][2] * 0.0 + mat[1][2] * 1.0 + mat[2][2] * 0.0,
                ];
                let up_len = (up[0] * up[0] + up[1] * up[1] + up[2] * up[2]).sqrt();
                let axis = if up_len > 1e-8 {
                    [up[0] / up_len, up[1] / up_len, up[2] / up_len]
                } else {
                    [0.0, 1.0, 0.0]
                };
                out.push(ResolvedCollider::Capsule {
                    center,
                    radius,
                    half_height: height * 0.5,
                    axis,
                });
            }
        }
    }
    out
}
