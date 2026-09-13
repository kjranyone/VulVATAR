//! Body-surface distance field for spring-bone (hair) collision.
//!
//! Replaces the heuristic capsule colliders on spring chains with the
//! actual posed body surface: the renderer voxel-splats the freshly
//! skinned body primitive into a distance field every frame (see
//! `renderer::sdf_field`), ships the field back through
//! [`crate::renderer::RenderResult`], and the spring solver resolves
//! strand joints against it here.
//!
//! The field is **unsigned** and truncated: cell values hold the exact
//! distance to the nearest splatted triangle within
//! [`SHELL_METRES`] of the surface, and [`SENTINEL`] beyond. Hair only
//! ever needs the near-surface band — a strand joint's collision
//! radius is millimetres to centimetres — so signing is unnecessary:
//! a joint that ends up inside the body projects to the nearest
//! surface either way.
//!
//! Spaces: the grid is defined in **avatar-root space**, the same space
//! the renderer draws skinned vertices in (`frame_input.rs`,
//! `RenderAvatarInstance::world_transform` is not applied) and the same
//! space the spring solver runs in (`pose.global_transforms`). If the
//! renderer ever starts folding the instance world transform into
//! skinned vertices, both the splat shader and this sampler need the
//! same correction.

use std::sync::Arc;

use crate::asset::Aabb;

/// Cells at or beyond this value hold no splatted triangle within the
/// shell — treated as "far from the body, no collision".
pub const SENTINEL: f32 = f32::MAX;

/// Distance band (metres) the splat pass represents exactly. Must stay
/// comfortably above the largest strand radius + margin the solvers
/// use; beyond it collision is a no-op anyway.
pub const SHELL_METRES: f32 = 0.06;

/// Value substituted for unsplatted cells during interpolation. Large
/// enough that no real contact radius ever triggers, small enough that
/// gradients stay finite and sane at the shell boundary (a raw
/// [`SENTINEL`] corner would zero out central differences).
const OUTSIDE_VALUE: f32 = SHELL_METRES * 2.0;

/// Hard cap on total cells so a huge avatar AABB cannot balloon the
/// per-frame readback. ~1.05 M cells × 4 B ≈ 4.2 MB — the same order
/// as the existing full-frame pixel readback.
const MAX_CELLS: usize = 1_050_000;

/// Candidate voxel sizes (metres), coarse-to-fine search order.
const VOXEL_CANDIDATES: [f32; 6] = [0.020, 0.016, 0.0125, 0.010, 0.008, 0.006];

/// Grid geometry in avatar-root space. Deterministic from the asset's
/// rest AABB, so the CPU and the renderer agree without any handshake.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfGrid {
    /// World position of cell (0,0,0)'s lower corner.
    pub origin: [f32; 3],
    /// Metres per cell.
    pub voxel: f32,
    /// Cells per axis.
    pub dims: [u32; 3],
}

impl SdfGrid {
    /// Build a grid covering `aabb` padded by [`SHELL_METRES`] on every
    /// side, choosing the finest voxel size whose cell count fits
    /// [`MAX_CELLS`]. Falls back to the coarsest candidate when even
    /// that overflows (degenerate oversized assets).
    pub fn for_aabb(aabb: &Aabb) -> Self {
        let pad = SHELL_METRES;
        let min = [aabb.min[0] - pad, aabb.min[1] - pad, aabb.min[2] - pad];
        let max = [aabb.max[0] + pad, aabb.max[1] + pad, aabb.max[2] + pad];
        // Defensive: degenerate/empty AABB still yields a sliver grid
        // instead of a division by zero below.
        let size = [
            (max[0] - min[0]).max(0.01),
            (max[1] - min[1]).max(0.01),
            (max[2] - min[2]).max(0.01),
        ];
        // Finest candidate whose cell count fits the budget. Candidates
        // are coarse-to-fine; walk fine-to-coarse and take the first
        // that fits, falling back to the coarsest when none do.
        let mut chosen = VOXEL_CANDIDATES[0];
        for &voxel in VOXEL_CANDIDATES.iter().rev() {
            let cells = (size[0] / voxel).ceil() as f64
                * (size[1] / voxel).ceil() as f64
                * (size[2] / voxel).ceil() as f64;
            if cells <= MAX_CELLS as f64 {
                chosen = voxel;
                break;
            }
        }
        let dims = [
            ((size[0] / chosen).ceil() as u32).max(1),
            ((size[1] / chosen).ceil() as u32).max(1),
            ((size[2] / chosen).ceil() as u32).max(1),
        ];
        Self {
            origin: min,
            voxel: chosen,
            dims,
        }
    }

    pub fn cell_count(&self) -> usize {
        self.dims[0] as usize * self.dims[1] as usize * self.dims[2] as usize
    }

    /// Linear cell index — MUST match the splat shader's layout
    /// (`x + dims.x * (y + dims.y * z)`).
    #[inline]
    pub fn index(&self, x: u32, y: u32, z: u32) -> usize {
        (x + self.dims[0] * (y + self.dims[1] * z)) as usize
    }
}

/// A distance field sampled from the renderer's readback.
#[derive(Clone, Debug)]
pub struct SdfField {
    pub grid: SdfGrid,
    pub data: Arc<Vec<f32>>,
}

impl SdfField {
    pub fn new(grid: SdfGrid, data: Arc<Vec<f32>>) -> Self {
        Self { grid, data }
    }

    /// Trilinear sample at `p` (avatar-root space). Points outside the
    /// grid return [`SENTINEL`]; unsplatted corners interpolate as
    /// [`OUTSIDE_VALUE`], so the field stays continuous across the
    /// shell boundary and gradients stay well-defined.
    pub fn sample(&self, p: [f32; 3]) -> f32 {
        let g = self.grid;
        let fx = (p[0] - g.origin[0]) / g.voxel;
        let fy = (p[1] - g.origin[1]) / g.voxel;
        let fz = (p[2] - g.origin[2]) / g.voxel;
        if fx < 0.0 || fy < 0.0 || fz < 0.0 {
            return SENTINEL;
        }
        let x0 = fx.floor() as i64;
        let y0 = fy.floor() as i64;
        let z0 = fz.floor() as i64;
        if x0 + 1 >= g.dims[0] as i64
            || y0 + 1 >= g.dims[1] as i64
            || z0 + 1 >= g.dims[2] as i64
        {
            return SENTINEL;
        }
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let tz = fz - z0 as f32;
        let mut acc = 0.0f32;
        for (dz, wz) in [(0u32, 1.0 - tz), (1u32, tz)] {
            for (dy, wy) in [(0u32, 1.0 - ty), (1u32, ty)] {
                for (dx, wx) in [(0u32, 1.0 - tx), (1u32, tx)] {
                    let idx = g.index(
                        (x0 as u32) + dx,
                        (y0 as u32) + dy,
                        (z0 as u32) + dz,
                    );
                    let raw = self.data[idx];
                    let v = if raw >= SENTINEL { OUTSIDE_VALUE } else { raw };
                    acc += v * wx * wy * wz;
                }
            }
        }
        acc
    }

    /// Central-difference gradient of the sampled field, normalized.
    /// Returns None inside the sentinel region or at a gradient null
    /// (field minimum — ambiguous projection direction).
    pub fn gradient(&self, p: [f32; 3]) -> Option<[f32; 3]> {
        let h = self.grid.voxel;
        let dx = self.sample([p[0] + h, p[1], p[2]]) - self.sample([p[0] - h, p[1], p[2]]);
        let dy = self.sample([p[0], p[1] + h, p[2]]) - self.sample([p[0], p[1] - h, p[2]]);
        let dz = self.sample([p[0], p[1], p[2] + h]) - self.sample([p[0], p[1], p[2] - h]);
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len < 1e-6 || !len.is_finite() {
            return None;
        }
        Some([dx / len, dy / len, dz / len])
    }

    /// Project `p` out to at least `radius` from the body surface.
    /// Returns `Some((corrected, applied_correction))` when the point
    /// was inside the collision band, `None` when no contact.
    pub fn resolve(&self, p: [f32; 3], radius: f32) -> Option<([f32; 3], [f32; 3])> {
        let d = self.sample(p);
        if d >= radius || d >= SENTINEL {
            return None;
        }
        let n = self.gradient(p)?;
        let target = radius.min(SHELL_METRES * 0.9);
        let push = target - d;
        Some((
            [p[0] + n[0] * push, p[1] + n[1] * push, p[2] + n[2] * push],
            [n[0] * push, n[1] * push, n[2] * push],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Splats a single distance sample into an otherwise-sentinel grid.
    fn field_with(values: Vec<(u32, u32, u32, f32)>, dims: [u32; 3], voxel: f32) -> SdfField {
        let grid = SdfGrid {
            origin: [0.0, 0.0, 0.0],
            voxel,
            dims,
        };
        let mut data = vec![SENTINEL; grid.cell_count()];
        for (x, y, z, v) in values {
            data[grid.index(x, y, z)] = v;
        }
        SdfField::new(grid, Arc::new(data))
    }

    #[test]
    fn grid_for_aabb_pads_and_fits_budget() {
        let aabb = Aabb {
            min: [-0.2, 0.0, -0.15],
            max: [0.2, 1.34, 0.15],
        };
        let g = SdfGrid::for_aabb(&aabb);
        assert!(g.voxel <= VOXEL_CANDIDATES[0]);
        assert!(g.cell_count() <= MAX_CELLS);
        // Padded bounds contain the padded AABB.
        assert!(g.origin[0] <= aabb.min[0] - SHELL_METRES + 1e-4);
        let extent_x = g.origin[0] + g.dims[0] as f32 * g.voxel;
        assert!(extent_x >= aabb.max[0] + SHELL_METRES - 1e-4);
    }

    #[test]
    fn sample_interpolates_inside_band_and_sentinel_outside() {
        // One splatted node with distance 0.01 at grid position
        // (5,5,5), 1 cm voxels — values live at node positions.
        let f = field_with(vec![(5, 5, 5, 0.01)], [16, 16, 16], 0.01);
        let node = [0.05, 0.05, 0.05];
        assert!((f.sample(node) - 0.01).abs() < 1e-6);
        // Halfway to the neighbouring (unsplatted) nodes: 1/8 of the
        // splatted value, 7/8 OUTSIDE_VALUE.
        let mid = [0.055, 0.055, 0.055];
        let expected = 0.125 * 0.01 + 0.875 * OUTSIDE_VALUE;
        assert!((f.sample(mid) - expected).abs() < 1e-5);
        // Far corner: no support → outside value (no contact band).
        assert!((f.sample([0.001, 0.001, 0.001]) - OUTSIDE_VALUE).abs() < 1e-5);
        // Outside the grid entirely.
        assert_eq!(f.sample([9.9, 9.9, 9.9]), SENTINEL);
    }

    #[test]
    fn resolve_pushes_point_out_to_radius() {
        // A surface along z = 0.08 with nodes splatted on BOTH sides
        // (the real splat pass covers ±shell around every triangle),
        // 1 cm voxels.
        let mut values = Vec::new();
        for x in 4..12 {
            for y in 4..12 {
                values.push((x, y, 7, 0.01));
                values.push((x, y, 8, 0.0));
                values.push((x, y, 9, 0.01));
            }
        }
        let f = field_with(values, [16, 16, 16], 0.01);
        // 4 mm above the zero slab → inside the 2 cm collision radius.
        let p = [0.08, 0.08, 0.084];
        let (corrected, corr) = f.resolve(p, 0.02).expect("contact expected");
        // Push is along +z (away from the surface).
        assert!(corr[2] > 0.015, "pushed along +z, corr={corr:?}");
        assert!((corrected[2] - p[2]).abs() > 0.014);
        // After correction the sampled distance clears the radius.
        assert!(f.sample(corrected) >= 0.02 - 1e-3);
        // A point well clear of the band stays untouched.
        assert!(f.resolve([0.08, 0.08, 0.2], 0.02).is_none());
    }

    #[test]
    fn resolve_skips_ambiguous_minimum() {
        // A single isolated cell surrounded by unsplatted space: the
        // interpolated value at its centre is dominated by
        // OUTSIDE_VALUE, so no contact triggers and nothing is pushed.
        let f = field_with(vec![(8, 8, 8, 0.0)], [16, 16, 16], 0.01);
        let p = [8.5 * 0.01, 8.5 * 0.01, 8.5 * 0.01];
        assert!(f.resolve(p, 0.02).is_none());
    }

    #[test]
    fn gradient_finite_across_shell_boundary() {
        // A linear ramp along z: gradient exists everywhere inside the
        // splatted block.
        let mut values = Vec::new();
        for x in 4..12 {
            for y in 4..12 {
                for z in 4..12 {
                    values.push((x, y, z, (z as i32 - 8) as f32 * 0.001));
                }
            }
        }
        let f = field_with(values, [16, 16, 16], 0.01);
        // Inside the splatted block: gradient is +z, unit length.
        let g = f.gradient([0.08, 0.08, 0.08]).expect("gradient inside block");
        assert!(g[2] > 0.99, "ramp gradient along +z, got {g:?}");
        // Just outside the block but inside the interpolation support:
        // still finite (OUTSIDE_VALUE substitution), no NaN.
        if let Some(g) = f.gradient([0.125, 0.08, 0.08]) {
            assert!(g.iter().all(|c| c.is_finite()));
        }
    }
}
