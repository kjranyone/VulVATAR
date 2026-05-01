use crate::asset::{
    AnimationChannel, AnimationClip, AnimationClipId, AnimationProperty, InterpolationMode,
    Keyframe, Transform, Vec4,
};
use crate::math_utils::quat_normalize;

#[derive(Clone, Debug)]
pub struct AnimationState {
    pub active_clip: Option<AnimationClipId>,
    pub playhead_seconds: f32,
    pub speed: f32,
    pub looping: bool,
}

impl Default for AnimationState {
    fn default() -> Self {
        Self {
            active_clip: None,
            playhead_seconds: 0.0,
            speed: 1.0,
            looping: true,
        }
    }
}

impl AnimationState {
    /// Advance the animation playhead by `dt` seconds.
    ///
    /// `clip_duration` is the length of the currently active clip in seconds.
    /// If no clip is active the call is a no-op.
    pub fn advance(&mut self, dt: f32, clip_duration: f32) {
        if self.active_clip.is_none() || clip_duration <= 0.0 {
            return;
        }
        self.playhead_seconds += dt * self.speed;
        if self.looping {
            self.playhead_seconds = self.playhead_seconds.rem_euclid(clip_duration);
        } else {
            self.playhead_seconds = self.playhead_seconds.min(clip_duration);
        }
    }

    /// Reset the playhead to the start of the clip.
    pub fn reset(&mut self) {
        self.playhead_seconds = 0.0;
    }

    /// Returns a normalized sample position in `[0, 1]` given the clip
    /// duration.  Useful for sampling keyframes.
    pub fn normalized_time(&self, clip_duration: f32) -> f32 {
        if clip_duration <= 0.0 {
            return 0.0;
        }
        (self.playhead_seconds / clip_duration).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Keyframe sampling
// ---------------------------------------------------------------------------

/// Linearly interpolate two `Vec4` values (component-wise).
fn lerp_vec4(a: &Vec4, b: &Vec4, t: f32) -> Vec4 {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

/// Spherical linear interpolation for unit quaternions stored as `[x, y, z, w]`.
fn slerp(a: &Vec4, b: &Vec4, t: f32) -> Vec4 {
    let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];

    // If the dot product is negative, negate one quaternion to take the
    // shorter arc.
    let mut b_adj = *b;
    if dot < 0.0 {
        b_adj = [-b[0], -b[1], -b[2], -b[3]];
        dot = -dot;
    }

    // Fall back to lerp + normalize for nearly-parallel quaternions.
    if dot > 0.9995 {
        let out = lerp_vec4(a, &b_adj, t);
        return quat_normalize(&out);
    }

    let theta = dot.clamp(-1.0, 1.0).acos();
    let sin_theta = theta.sin();
    let wa = ((1.0 - t) * theta).sin() / sin_theta;
    let wb = (t * theta).sin() / sin_theta;

    [
        a[0] * wa + b_adj[0] * wb,
        a[1] * wa + b_adj[1] * wb,
        a[2] * wa + b_adj[2] * wb,
        a[3] * wa + b_adj[3] * wb,
    ]
}

/// Cubic hermite interpolation for a single `Vec4` value.
///
/// `p0` / `p1` are the endpoint values, `m0` is the out-tangent of the first
/// keyframe and `m1` the in-tangent of the second, both already scaled by
/// the segment duration `dt`.
fn cubic_hermite_vec4(p0: &Vec4, m0: &Vec4, p1: &Vec4, m1: &Vec4, t: f32) -> Vec4 {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    [
        h00 * p0[0] + h10 * m0[0] + h01 * p1[0] + h11 * m1[0],
        h00 * p0[1] + h10 * m0[1] + h01 * p1[1] + h11 * m1[1],
        h00 * p0[2] + h10 * m0[2] + h01 * p1[2] + h11 * m1[2],
        h00 * p0[3] + h10 * m0[3] + h01 * p1[3] + h11 * m1[3],
    ]
}

/// Find the index of the last keyframe whose time is <= `time`.
/// Returns `None` if all keyframes are after `time`.
fn find_lower_keyframe(keyframes: &[Keyframe], time: f32) -> Option<usize> {
    keyframes
        .partition_point(|kf| kf.time <= time)
        .checked_sub(1)
}

/// Interpolate a single channel at the given `time` and return the resulting `Vec4`.
fn sample_channel(channel: &AnimationChannel, time: f32) -> Option<Vec4> {
    let kfs = &channel.keyframes;
    if kfs.is_empty() {
        return None;
    }

    // Before the first keyframe -> use first value.
    if time <= kfs[0].time {
        return Some(kfs[0].value);
    }

    // After the last keyframe -> use last value.
    let last = kfs.len() - 1;
    if time >= kfs[last].time {
        return Some(kfs[last].value);
    }

    // Find surrounding keyframes.
    let lo = find_lower_keyframe(kfs, time).unwrap_or(0);
    let hi = (lo + 1).min(last);

    if lo == hi {
        return Some(kfs[lo].value);
    }

    let kf0 = &kfs[lo];
    let kf1 = &kfs[hi];
    let dt_seg = kf1.time - kf0.time;
    let t = if dt_seg > 1e-9 {
        ((time - kf0.time) / dt_seg).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let interpolation = kf0.interpolation;

    match interpolation {
        InterpolationMode::Step => Some(kf0.value),
        InterpolationMode::Linear => {
            if channel.property == AnimationProperty::Rotation {
                Some(slerp(&kf0.value, &kf1.value, t))
            } else {
                Some(lerp_vec4(&kf0.value, &kf1.value, t))
            }
        }
        InterpolationMode::CubicSpline => {
            let (m0, m1) = match (&kf0.tangents, &kf1.tangents) {
                (Some([_in0, out0]), Some([in1, _out1])) => {
                    // Scale tangents by the segment duration.
                    let m0 = [
                        out0[0] * dt_seg,
                        out0[1] * dt_seg,
                        out0[2] * dt_seg,
                        out0[3] * dt_seg,
                    ];
                    let m1 = [
                        in1[0] * dt_seg,
                        in1[1] * dt_seg,
                        in1[2] * dt_seg,
                        in1[3] * dt_seg,
                    ];
                    (m0, m1)
                }
                _ => {
                    // Fallback to linear when tangent data is missing.
                    if channel.property == AnimationProperty::Rotation {
                        return Some(slerp(&kf0.value, &kf1.value, t));
                    } else {
                        return Some(lerp_vec4(&kf0.value, &kf1.value, t));
                    }
                }
            };

            let result = cubic_hermite_vec4(&kf0.value, &m0, &kf1.value, &m1, t);

            // Normalize if this is a rotation channel.
            if channel.property == AnimationProperty::Rotation {
                Some(quat_normalize(&result))
            } else {
                Some(result)
            }
        }
    }
}

/// Apply a `Vec4` sample to a `Transform` depending on the target property.
fn apply_sample(transform: &mut Transform, property: AnimationProperty, value: Vec4) {
    match property {
        AnimationProperty::Translation => {
            transform.translation = [value[0], value[1], value[2]];
        }
        AnimationProperty::Rotation => {
            transform.rotation = value;
        }
        AnimationProperty::Scale => {
            transform.scale = [value[0], value[1], value[2]];
        }
    }
}

/// Sample an entire animation clip at `time` and write results into `local_transforms`.
///
/// Each channel looks up its `target_node` index (via `NodeId.0`) and writes
/// the interpolated value into the corresponding element of `local_transforms`.
pub fn sample_clip(clip: &AnimationClip, time: f32, local_transforms: &mut [Transform]) {
    let clamped_time = time.clamp(0.0, clip.duration);

    for channel in &clip.channels {
        let node_idx = channel.target_node.0 as usize;
        if node_idx >= local_transforms.len() {
            continue;
        }

        if let Some(value) = sample_channel(channel, clamped_time) {
            apply_sample(&mut local_transforms[node_idx], channel.property, value);
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers for looking up clip duration from the asset
// ---------------------------------------------------------------------------

/// Given an `AnimationClipId` and the clips stored on the asset, return the
/// matching clip's duration. Returns `None` if no clip matches.
pub fn clip_duration(clip_id: &AnimationClipId, clips: &[AnimationClip]) -> Option<f32> {
    clips.iter().find(|c| c.id == *clip_id).map(|c| c.duration)
}

/// Find a clip reference by id.
pub fn find_clip<'a>(
    clip_id: &AnimationClipId,
    clips: &'a [AnimationClip],
) -> Option<&'a AnimationClip> {
    clips.iter().find(|c| c.id == *clip_id)
}
