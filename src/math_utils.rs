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

/// Rotation that maps an orthonormal basis pair `(rest_fwd, rest_up)`
/// onto a target pair `(src_fwd, src_up)`. Use this when a single
/// `quat_from_vectors` would leave the bone's twist around its
/// length axis ambiguous — e.g. the wrist, where finger orientation
/// hinges on the palm plane and not just the wrist→fingers vector.
///
/// All four inputs should be unit-length. Inputs need not be exactly
/// orthogonal: the function projects `up` onto the plane perpendicular
/// to `fwd` internally, so a slightly off-axis `up` (typical of the
/// noisy MCP-derived palm normal) still gives a well-defined result.
pub fn quat_from_basis_pair(
    rest_fwd: &Vec3,
    rest_up: &Vec3,
    src_fwd: &Vec3,
    src_up: &Vec3,
) -> Quat {
    // Step 1: shortest-arc to align the forward vectors. After this
    // rotation the rest frame's forward already points where we want;
    // only the twist around `src_fwd` remains to be fixed.
    let q1 = quat_from_vectors(rest_fwd, src_fwd);
    let rest_up_after = quat_rotate_vec3(&q1, rest_up);

    // Step 2: project both ups onto the plane perpendicular to
    // src_fwd, then find the rotation around src_fwd that takes
    // the rotated rest-up onto src_up. This second rotation has
    // axis exactly along src_fwd so it preserves the alignment we
    // achieved in step 1.
    let dot_a = vec3_dot(&rest_up_after, src_fwd);
    let proj_a = vec3_normalize(&[
        rest_up_after[0] - src_fwd[0] * dot_a,
        rest_up_after[1] - src_fwd[1] * dot_a,
        rest_up_after[2] - src_fwd[2] * dot_a,
    ]);
    let dot_b = vec3_dot(src_up, src_fwd);
    let proj_b = vec3_normalize(&[
        src_up[0] - src_fwd[0] * dot_b,
        src_up[1] - src_fwd[1] * dot_b,
        src_up[2] - src_fwd[2] * dot_b,
    ]);
    if proj_a == [0.0, 0.0, 0.0] || proj_b == [0.0, 0.0, 0.0] {
        // up is parallel to fwd on either side — twist undefined,
        // fall back to forward-only alignment.
        return q1;
    }
    let q2 = quat_from_vectors(&proj_a, &proj_b);
    quat_normalize(&quat_mul(&q2, &q1))
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
