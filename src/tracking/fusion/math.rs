//! Small f64 linear-algebra kit for the fusion estimator: 3-vectors,
//! 3×3 rotations with SO(3) exp/log and their left Jacobians, a dense
//! symmetric solver (LDLᵀ with diagonal damping) sized for the ~100-DoF
//! normal equations, and the robust kernels used by the residual blocks.
//!
//! Everything here is allocation-free except `Dense`, which is reused
//! across iterations by the estimator.

pub type V3 = [f64; 3];
pub type M3 = [[f64; 3]; 3];

#[inline]
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
#[inline]
pub fn normalize(a: V3) -> V3 {
    let n = norm(a);
    if n > 1e-12 {
        scale(a, 1.0 / n)
    } else {
        [0.0, 0.0, 0.0]
    }
}

pub const I3: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

#[inline]
pub fn mat_mul(a: &M3, b: &M3) -> M3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}
#[inline]
pub fn mat_vec(a: &M3, v: V3) -> V3 {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}
#[inline]
pub fn transpose(a: &M3) -> M3 {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}
/// Column `k` of a rotation matrix = image of the k-th basis vector.
#[inline]
pub fn col(a: &M3, k: usize) -> V3 {
    [a[0][k], a[1][k], a[2][k]]
}
#[inline]
pub fn skew(w: V3) -> M3 {
    [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]]
}
#[inline]
pub fn mat_add(a: &M3, b: &M3) -> M3 {
    let mut r = *a;
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] += b[i][j];
        }
    }
    r
}
#[inline]
pub fn mat_scale(a: &M3, s: f64) -> M3 {
    let mut r = *a;
    for row in r.iter_mut() {
        for v in row.iter_mut() {
            *v *= s;
        }
    }
    r
}

/// Rodrigues: exp([w]×).
pub fn so3_exp(w: V3) -> M3 {
    let th2 = dot(w, w);
    let th = th2.sqrt();
    let k = skew(w);
    let k2 = mat_mul(&k, &k);
    let (a, b) = if th < 1e-6 {
        (1.0 - th2 / 6.0, 0.5 - th2 / 24.0)
    } else {
        (th.sin() / th, (1.0 - th.cos()) / th2)
    };
    mat_add(&mat_add(&I3, &mat_scale(&k, a)), &mat_scale(&k2, b))
}

/// Logarithm: rotation vector of `r` (angle in [0, π]).
pub fn so3_log(r: &M3) -> V3 {
    let tr = (r[0][0] + r[1][1] + r[2][2]).clamp(-1.0, 3.0);
    let cos_th = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0);
    let th = cos_th.acos();
    let v = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
    if th < 1e-6 {
        return scale(v, 0.5);
    }
    if th > std::f64::consts::PI - 1e-4 {
        // Near π: use the symmetric part to recover the axis robustly.
        let mut axis = [
            (r[0][0] + 1.0).max(0.0).sqrt(),
            (r[1][1] + 1.0).max(0.0).sqrt(),
            (r[2][2] + 1.0).max(0.0).sqrt(),
        ];
        // Fix signs from the off-diagonals.
        let i = if axis[0] >= axis[1] && axis[0] >= axis[2] {
            0
        } else if axis[1] >= axis[2] {
            1
        } else {
            2
        };
        let s = axis[i] / std::f64::consts::SQRT_2;
        axis[i] = s;
        let (j, k) = ((i + 1) % 3, (i + 2) % 3);
        axis[j] = (r[i][j] + r[j][i]) / (4.0 * s);
        axis[k] = (r[i][k] + r[k][i]) / (4.0 * s);
        return scale(normalize(axis), th);
    }
    scale(v, th / (2.0 * th.sin()))
}

/// Left Jacobian of SO(3): exp(w + δ) ≈ exp(J_l(w) δ) exp(w).
pub fn so3_left_jacobian(w: V3) -> M3 {
    let th2 = dot(w, w);
    let th = th2.sqrt();
    let k = skew(w);
    let k2 = mat_mul(&k, &k);
    let (a, b) = if th < 1e-6 {
        (0.5 - th2 / 24.0, 1.0 / 6.0 - th2 / 120.0)
    } else {
        ((1.0 - th.cos()) / th2, (th - th.sin()) / (th2 * th))
    };
    mat_add(&mat_add(&I3, &mat_scale(&k, a)), &mat_scale(&k2, b))
}

/// Inverse left Jacobian: d log(exp(δ) R) / dδ at δ=0 equals J_l⁻¹(log R).
pub fn so3_left_jacobian_inv(w: V3) -> M3 {
    let th2 = dot(w, w);
    let th = th2.sqrt();
    let k = skew(w);
    let k2 = mat_mul(&k, &k);
    let b = if th < 1e-6 {
        1.0 / 12.0 + th2 / 720.0
    } else {
        1.0 / th2 - (1.0 + th.cos()) / (2.0 * th * th.sin())
    };
    mat_add(&mat_add(&I3, &mat_scale(&k, -0.5)), &mat_scale(&k2, b))
}

/// Quaternion `[x, y, z, w]` → rotation matrix (f32 in, f64 out).
pub fn quat_to_mat(q: [f32; 4]) -> M3 {
    let (x, y, z, w) = (q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64);
    let n = (x * x + y * y + z * z + w * w).sqrt().max(1e-12);
    let (x, y, z, w) = (x / n, y / n, z / n, w / n);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// Rotation matrix → quaternion `[x, y, z, w]` (f32).
pub fn mat_to_quat(m: &M3) -> [f32; 4] {
    let tr = m[0][0] + m[1][1] + m[2][2];
    let q = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        [
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
            0.25 * s,
        ]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        [
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[2][1] - m[1][2]) / s,
        ]
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        [
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
            (m[0][2] - m[2][0]) / s,
        ]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        [
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
            (m[1][0] - m[0][1]) / s,
        ]
    };
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [
        (q[0] / n) as f32,
        (q[1] / n) as f32,
        (q[2] / n) as f32,
        (q[3] / n) as f32,
    ]
}

/// Re-orthonormalise a drifting rotation matrix (Gram–Schmidt on columns).
pub fn orthonormalize(r: &M3) -> M3 {
    let c0 = normalize(col(r, 0));
    let c1 = col(r, 1);
    let c1 = normalize(sub(c1, scale(c0, dot(c0, c1))));
    let c2 = cross(c0, c1);
    [
        [c0[0], c1[0], c2[0]],
        [c0[1], c1[1], c2[1]],
        [c0[2], c1[2], c2[2]],
    ]
}

// ---------------------------------------------------------------------------
// Robust kernels
// ---------------------------------------------------------------------------

/// Robust loss ρ(s) on the squared, whitened residual `s = r²/σ²`.
/// Returns `(ρ, ρ')` — the IRLS weight is `ρ'`.
#[derive(Clone, Copy, Debug)]
pub enum Kernel {
    /// Plain least squares.
    L2,
    /// Cauchy: ρ = c² log(1 + s/c²).
    Cauchy(f64),
    /// Geman–McClure: ρ = s / (1 + s/c²) — outliers saturate at c².
    GemanMcClure(f64),
    /// Tukey's biweight: redescending M-estimator where outliers beyond c
    /// receive exactly 0 weight and cost saturates at c²/3.
    Tukey(f64),
}

impl Kernel {
    #[inline]
    pub fn eval(self, s: f64) -> (f64, f64) {
        match self {
            Kernel::L2 => (s, 1.0),
            Kernel::Cauchy(c) => {
                let c2 = c * c;
                let u = 1.0 + s / c2;
                (c2 * u.ln(), 1.0 / u)
            }
            Kernel::GemanMcClure(c) => {
                let c2 = c * c;
                let u = 1.0 + s / c2;
                (s / u, 1.0 / (u * u))
            }
            Kernel::Tukey(c) => {
                let c2 = c * c;
                if s >= c2 {
                    (c2 / 3.0, 0.0)
                } else {
                    let u = 1.0 - s / c2;
                    (c2 / 3.0 * (1.0 - u * u * u), u * u)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dense symmetric system
// ---------------------------------------------------------------------------

/// Dense symmetric matrix + gradient accumulator for the normal equations
/// `(H + λD) δ = -g`. Rows are stored fully (not just a triangle) so the
/// accumulate loop is branch-free; only the upper triangle is read by the
/// solver.
pub struct Dense {
    pub n: usize,
    pub h: Vec<f64>,
    pub g: Vec<f64>,
    /// Scratch for the factorisation.
    l: Vec<f64>,
    d: Vec<f64>,
}

impl Dense {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            h: vec![0.0; n * n],
            g: vec![0.0; n],
            l: vec![0.0; n * n],
            d: vec![0.0; n],
        }
    }

    pub fn resize(&mut self, n: usize) {
        if n != self.n {
            *self = Self::new(n);
        }
    }

    pub fn clear(&mut self) {
        self.h.iter_mut().for_each(|v| *v = 0.0);
        self.g.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Accumulate one scalar residual `r` with sparse Jacobian
    /// `(index, value)` pairs, information weight `w` (already including
    /// 1/σ² and the robust weight): `H += w JᵀJ`, `g += w Jᵀ r`.
    #[inline]
    pub fn add_residual(&mut self, jac: &[(usize, f64)], r: f64, w: f64) {
        let n = self.n;
        for &(i, ji) in jac {
            let wi = w * ji;
            self.g[i] += wi * r;
            let row = i * n;
            for &(j, jj) in jac {
                self.h[row + j] += wi * jj;
            }
        }
    }

    /// Accumulate a 3-vector residual whose Jacobian is given per parameter
    /// as a 3-vector column: `jac[k] = (index, ∂r/∂x_index)`.
    #[inline]
    pub fn add_residual3(&mut self, jac: &[(usize, V3)], r: V3, w: f64) {
        let n = self.n;
        for &(i, ji) in jac {
            let gi = w * dot(ji, r);
            self.g[i] += gi;
            let row = i * n;
            for &(j, jj) in jac {
                self.h[row + j] += w * dot(ji, jj);
            }
        }
    }

    /// Solve `(H + λ·diag(H) + ε I) δ = -g` via LDLᵀ. Returns `None` if the
    /// factorisation hits a non-positive pivot (caller raises λ).
    pub fn solve_damped(&mut self, lambda: f64, eps: f64, out: &mut [f64]) -> Option<()> {
        let n = self.n;
        // Copy H (upper) into L working storage with damping on the diagonal.
        for i in 0..n {
            for j in 0..n {
                self.l[i * n + j] = self.h[i * n + j];
            }
            let dii = self.h[i * n + i];
            self.l[i * n + i] = dii + lambda * dii + eps;
        }
        // In-place LDLᵀ (row-oriented, lower triangle of `l`).
        for j in 0..n {
            let mut djj = self.l[j * n + j];
            for k in 0..j {
                let ljk = self.l[j * n + k];
                djj -= ljk * ljk * self.d[k];
            }
            if !(djj > 1e-300) || !djj.is_finite() {
                return None;
            }
            self.d[j] = djj;
            for i in (j + 1)..n {
                let mut v = self.l[i * n + j];
                for k in 0..j {
                    v -= self.l[i * n + k] * self.l[j * n + k] * self.d[k];
                }
                self.l[i * n + j] = v / djj;
            }
        }
        // Forward: L y = -g
        for i in 0..n {
            let mut y = -self.g[i];
            for k in 0..i {
                y -= self.l[i * n + k] * out[k];
            }
            out[i] = y;
        }
        // Diagonal
        for i in 0..n {
            out[i] /= self.d[i];
        }
        // Backward: Lᵀ x = z
        for i in (0..n).rev() {
            let mut x = out[i];
            for k in (i + 1)..n {
                x -= self.l[k * n + i] * out[k];
            }
            out[i] = x;
        }
        Some(())
    }

    /// Diagonal of the inverse of `H + eps I` (marginal variances), via
    /// the same LDLᵀ. O(n³) but n ≈ 100 → ~1 ms; called once per solve.
    pub fn marginal_variances(&mut self, eps: f64, out: &mut [f64]) -> Option<()> {
        self.inverse_impl(eps, out, None)
    }

    /// Full inverse of `H + eps I` (row-major n×n into `full`) plus its
    /// diagonal into `out`.
    pub fn full_inverse(&mut self, eps: f64, out: &mut [f64], full: &mut Vec<f64>) -> Option<()> {
        let n = self.n;
        if full.len() != n * n {
            full.resize(n * n, 0.0);
        }
        self.inverse_impl(eps, out, Some(full))
    }

    fn inverse_impl(
        &mut self,
        eps: f64,
        out: &mut [f64],
        mut full: Option<&mut Vec<f64>>,
    ) -> Option<()> {
        let n = self.n;
        // Factor once (λ = 0).
        for i in 0..n {
            for j in 0..n {
                self.l[i * n + j] = self.h[i * n + j];
            }
            self.l[i * n + i] += eps;
        }
        for j in 0..n {
            let mut djj = self.l[j * n + j];
            for k in 0..j {
                let ljk = self.l[j * n + k];
                djj -= ljk * ljk * self.d[k];
            }
            if !(djj > 1e-300) || !djj.is_finite() {
                return None;
            }
            self.d[j] = djj;
            for i in (j + 1)..n {
                let mut v = self.l[i * n + j];
                for k in 0..j {
                    v -= self.l[i * n + k] * self.l[j * n + k] * self.d[k];
                }
                self.l[i * n + j] = v / djj;
            }
        }
        // Solve for each unit vector, keep only the diagonal entry.
        let mut col = vec![0.0; n];
        for c in 0..n {
            for v in col.iter_mut() {
                *v = 0.0;
            }
            // Forward L y = e_c
            for i in c..n {
                let mut y = if i == c { 1.0 } else { 0.0 };
                for k in c..i {
                    y -= self.l[i * n + k] * col[k];
                }
                col[i] = y;
            }
            for i in c..n {
                col[i] /= self.d[i];
            }
            // Backward, only need x[c]: x_c = z_c - Σ_{k>c} L[k][c] x_k
            for i in (c..n).rev() {
                let mut x = col[i];
                for k in (i + 1)..n {
                    x -= self.l[k * n + i] * col[k];
                }
                col[i] = x;
            }
            out[c] = col[c].max(0.0);
            if let Some(f) = full.as_deref_mut() {
                // Backward pass for rows < c too (full column of the inverse).
                for i in (0..c).rev() {
                    let mut x = 0.0;
                    for k in (i + 1)..n {
                        x -= self.l[k * n + i] * col[k];
                    }
                    col[i] = x;
                }
                for i in 0..n {
                    f[i * n + c] = col[i];
                }
            }
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_log_roundtrip() {
        for w in [
            [0.1, -0.2, 0.3],
            [1.5, 0.2, -0.4],
            [0.0, 0.0, 3.0],
            [1e-9, 0.0, 0.0],
        ] {
            let r = so3_exp(w);
            let w2 = so3_log(&r);
            for k in 0..3 {
                assert!((w[k] - w2[k]).abs() < 1e-9, "{w:?} vs {w2:?}");
            }
        }
    }

    #[test]
    fn left_jacobian_matches_finite_difference() {
        let w = [0.4, -0.7, 0.2];
        let r0 = so3_exp(w);
        let jl = so3_left_jacobian(w);
        let eps = 1e-6;
        for k in 0..3 {
            let mut d = [0.0; 3];
            d[k] = eps;
            let r1 = so3_exp(add(w, d));
            // exp(w+d) ≈ exp(J d) exp(w)  →  log(r1 r0ᵀ) ≈ J d
            let dr = so3_log(&mat_mul(&r1, &transpose(&r0)));
            let jd = mat_vec(&jl, d);
            for i in 0..3 {
                assert!(
                    (dr[i] - jd[i]).abs() < 1e-8,
                    "k={k} i={i}: {} vs {}",
                    dr[i],
                    jd[i]
                );
            }
        }
        // inverse consistency
        let ji = so3_left_jacobian_inv(w);
        let p = mat_mul(&jl, &ji);
        for i in 0..3 {
            for j in 0..3 {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((p[i][j] - e).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn quat_mat_roundtrip() {
        let w = [0.3, 0.9, -1.1];
        let r = so3_exp(w);
        let q = mat_to_quat(&r);
        let r2 = quat_to_mat(q);
        for i in 0..3 {
            for j in 0..3 {
                assert!((r[i][j] - r2[i][j]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn dense_solve_and_marginals() {
        // H = A Aᵀ + I, known solution.
        let n = 5;
        let mut d = Dense::new(n);
        let a: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| ((i * 7 + j * 3) % 5) as f64 - 2.0).collect())
            .collect();
        let x_true: Vec<f64> = (0..n).map(|i| i as f64 * 0.5 - 1.0).collect();
        // Build H, g such that H x = -g  → g = -H x
        for i in 0..n {
            for j in 0..n {
                let mut s = if i == j { 1.0 } else { 0.0 };
                for k in 0..n {
                    s += a[i][k] * a[j][k];
                }
                d.h[i * n + j] = s;
            }
        }
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += d.h[i * n + j] * x_true[j];
            }
            d.g[i] = -s;
        }
        let mut x = vec![0.0; n];
        d.solve_damped(0.0, 0.0, &mut x).unwrap();
        for i in 0..n {
            assert!((x[i] - x_true[i]).abs() < 1e-9, "{x:?} vs {x_true:?}");
        }
        // marginals: compare with explicit inverse via solving unit vectors
        let mut var = vec![0.0; n];
        d.marginal_variances(0.0, &mut var).unwrap();
        for c in 0..n {
            let mut d2 = Dense::new(n);
            d2.h.copy_from_slice(&d.h);
            for i in 0..n {
                d2.g[i] = if i == c { -1.0 } else { 0.0 };
            }
            let mut e = vec![0.0; n];
            d2.solve_damped(0.0, 0.0, &mut e).unwrap();
            assert!((e[c] - var[c]).abs() < 1e-9);
        }
    }

    #[test]
    fn test_tukey_kernel() {
        let k = Kernel::Tukey(4.0);
        // at s=0: rho=0, w=1
        let (rho0, w0) = k.eval(0.0);
        assert_eq!(rho0, 0.0);
        assert_eq!(w0, 1.0);

        // at s = c^2 = 16: rho = c^2/3 = 16/3, w = 0
        let (rhoc, wc) = k.eval(16.0);
        assert!((rhoc - 16.0 / 3.0).abs() < 1e-9);
        assert_eq!(wc, 0.0);

        // at s > 16: rho stays 16/3, w stays 0
        let (rho_out, w_out) = k.eval(25.0);
        assert!((rho_out - 16.0 / 3.0).abs() < 1e-9);
        assert_eq!(w_out, 0.0);

        // at s = 8: 0 < w < 1
        let (_, w_mid) = k.eval(8.0);
        assert!((w_mid - 0.25).abs() < 1e-9);
    }

    #[test]
    fn kernels_have_unit_weight_near_zero() {
        for k in [Kernel::L2, Kernel::Cauchy(1.0), Kernel::GemanMcClure(1.0)] {
            let (rho, w) = k.eval(1e-9);
            assert!(rho.abs() < 1e-8);
            assert!((w - 1.0).abs() < 1e-6);
        }
        let (_, w) = Kernel::GemanMcClure(1.0).eval(100.0);
        assert!(w < 1e-3);
    }
}
