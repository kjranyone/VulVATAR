/// Shared math utilities used across simulation, avatar, and other subsystems.
///
/// These replace the previously duplicated per-module vec3/quat/geometry helpers.
pub type Vec3 = [f32; 3];
pub type Quat = [f32; 4];

// ---------------------------------------------------------------------------
// Vec3 helpers
// ---------------------------------------------------------------------------

#[inline]
pub fn vec3_add(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn vec3_sub(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn vec3_scale(a: &Vec3, s: f32) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub fn vec3_dot(a: &Vec3, b: &Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn vec3_cross(a: &Vec3, b: &Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn vec3_length(a: &Vec3) -> f32 {
    vec3_length_sq(a).sqrt()
}

#[inline]
pub fn vec3_length_sq(a: &Vec3) -> f32 {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
}

#[inline]
pub fn vec3_normalize(a: &Vec3) -> Vec3 {
    let len = vec3_length(a);
    if len < 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        vec3_scale(a, 1.0 / len)
    }
}

// ---------------------------------------------------------------------------
// Quaternion helpers (xyzw layout)
// ---------------------------------------------------------------------------

/// Normalize a quaternion. Returns identity if magnitude is near zero.
#[inline]
pub fn quat_normalize(q: &Quat) -> Quat {
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if len < 1e-12 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let inv = 1.0 / len;
    [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
}

/// Multiply two quaternions: result = a * b (Hamilton product).
#[inline]
pub fn quat_mul(a: &Quat, b: &Quat) -> Quat {
    let [ax, ay, az, aw] = *a;
    let [bx, by, bz, bw] = *b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

/// Conjugate (inverse, for unit quaternions).
#[inline]
pub fn quat_conjugate(q: &Quat) -> Quat {
    [-q[0], -q[1], -q[2], q[3]]
}

/// Rotate a vector by a unit quaternion: `v' = q * v * q^*`.
pub fn quat_rotate_vec3(q: &Quat, v: &Vec3) -> Vec3 {
    // Optimized form: v' = v + 2 * q.xyz × (q.xyz × v + q.w * v)
    let qv = [q[0], q[1], q[2]];
    let t = vec3_scale(&vec3_cross(&qv, v), 2.0);
    let t2 = vec3_cross(&qv, &t);
    [
        v[0] + q[3] * t[0] + t2[0],
        v[1] + q[3] * t[1] + t2[1],
        v[2] + q[3] * t[2] + t2[2],
    ]
}

/// Shortest-arc rotation that takes unit vector `from` to unit vector `to`.
/// Both inputs should be normalised; the result is a unit quaternion. Handles
/// the 180° degenerate case by choosing an arbitrary perpendicular axis.
pub fn quat_from_vectors(from: &Vec3, to: &Vec3) -> Quat {
    let dot = vec3_dot(from, to).clamp(-1.0, 1.0);
    if dot > 0.9999999 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    if dot < -0.9999999 {
        // 180° rotation — pick any axis perpendicular to `from`.
        let axis = if from[0].abs() < 0.9 {
            vec3_normalize(&vec3_cross(from, &[1.0, 0.0, 0.0]))
        } else {
            vec3_normalize(&vec3_cross(from, &[0.0, 1.0, 0.0]))
        };
        return [axis[0], axis[1], axis[2], 0.0];
    }
    let axis = vec3_cross(from, to);
    let w = 1.0 + dot;
    quat_normalize(&[axis[0], axis[1], axis[2], w])
}

/// Build a quaternion from Euler angles (radians) using Y-up graphics
/// convention: `pitch` around X, `yaw` around Y, `roll` around Z. Rotations
/// compose as `R_yaw * R_pitch * R_roll` (yaw applied last in world frame).
pub fn quat_from_euler_ypr(pitch: f32, yaw: f32, roll: f32) -> Quat {
    let qx = [(pitch * 0.5).sin(), 0.0, 0.0, (pitch * 0.5).cos()];
    let qy = [0.0, (yaw * 0.5).sin(), 0.0, (yaw * 0.5).cos()];
    let qz = [0.0, 0.0, (roll * 0.5).sin(), (roll * 0.5).cos()];
    quat_mul(&quat_mul(&qy, &qx), &qz)
}

// ---------------------------------------------------------------------------
// Mat4 helpers (column-major)
// ---------------------------------------------------------------------------

/// Column-major 4×4 layout matching `crate::asset::Mat4`. Defined here
/// rather than imported from `asset` to keep `math_utils` free of
/// upstream dependencies — the type is just `[[f32; 4]; 4]`, which the
/// compiler treats as identical to the `asset` alias.
pub type Mat4 = [[f32; 4]; 4];

/// Multiply two column-major 4×4 matrices: `result = a * b`.
#[inline]
pub fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = a[0][row] * b[col][0]
                + a[1][row] * b[col][1]
                + a[2][row] * b[col][2]
                + a[3][row] * b[col][3];
        }
    }
    out
}

/// Extract the translation column (column 3) of a column-major 4×4
/// affine matrix.
#[inline]
pub fn mat4_translation(m: &Mat4) -> Vec3 {
    [m[3][0], m[3][1], m[3][2]]
}

/// Extract the rotation component of a column-major affine 4×4 as a
/// unit quaternion using Shoemake's branch-stable algorithm.
///
/// Each basis column is normalised before extraction so a uniform
/// scale on the matrix doesn't bleed into the resulting quaternion's
/// magnitude. Non-uniform scale will produce a slightly off-axis
/// result — VRM rigs don't use it on humanoid bones, which is the
/// only place this is currently called from.
pub fn mat4_rotation_to_quat(m: &Mat4) -> Quat {
    let col0 = vec3_normalize(&[m[0][0], m[0][1], m[0][2]]);
    let col1 = vec3_normalize(&[m[1][0], m[1][1], m[1][2]]);
    let col2 = vec3_normalize(&[m[2][0], m[2][1], m[2][2]]);
    let m00 = col0[0];
    let m10 = col0[1];
    let m20 = col0[2];
    let m01 = col1[0];
    let m11 = col1[1];
    let m21 = col1[2];
    let m02 = col2[0];
    let m12 = col2[1];
    let m22 = col2[2];

    let trace = m00 + m11 + m22;
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        let inv = 1.0 / s;
        quat_normalize(&[
            (m21 - m12) * inv,
            (m02 - m20) * inv,
            (m10 - m01) * inv,
            0.25 * s,
        ])
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        let inv = 1.0 / s;
        quat_normalize(&[
            0.25 * s,
            (m01 + m10) * inv,
            (m02 + m20) * inv,
            (m21 - m12) * inv,
        ])
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        let inv = 1.0 / s;
        quat_normalize(&[
            (m01 + m10) * inv,
            0.25 * s,
            (m12 + m21) * inv,
            (m02 - m20) * inv,
        ])
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        let inv = 1.0 / s;
        quat_normalize(&[
            (m02 + m20) * inv,
            (m12 + m21) * inv,
            0.25 * s,
            (m10 - m01) * inv,
        ])
    }
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Closest point on line segment (a, b) to point p.
pub fn closest_point_on_segment(a: &Vec3, b: &Vec3, p: &Vec3) -> Vec3 {
    let ab = vec3_sub(b, a);
    let ap = vec3_sub(p, a);
    let ab_len_sq = vec3_length_sq(&ab);
    if ab_len_sq < 1e-12 {
        return *a;
    }
    let t = (vec3_dot(&ap, &ab) / ab_len_sq).clamp(0.0, 1.0);
    vec3_add(a, &vec3_scale(&ab, t))
}
