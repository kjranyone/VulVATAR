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
