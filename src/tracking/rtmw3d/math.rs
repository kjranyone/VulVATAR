//! Tiny vector helpers shared between [`super::skeleton`] and
//! [`super::wrist`]. Splitting them into their own file (rather than
//! defining them in either consumer) avoids a circular import between
//! the two and keeps both call sites symmetrical.

#[inline]
pub(super) fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(super) fn length3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}
