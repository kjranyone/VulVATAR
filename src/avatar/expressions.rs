//! Expression (blend-shape) resolution: FaceMesh blendshape weights from
//! the tracking sample → the avatar's named VRM expressions, with the
//! mouth-viseme source policy (camera / audio lip-sync / both) and a rest
//! deadband against blink micro-flutter.
//!
//! Extracted intact from the retired v1 `pose_solver` — this channel was
//! never part of the position-based body solve.

use std::collections::HashMap;

use crate::tracking::source_skeleton::SourceSkeleton;
use crate::tracking::MouthSource;

/// Rest deadband on expression weights (eye/brow channels): per-frame
/// deltas below this are shrunk toward zero so micro-flutter freezes
/// while a real blink (Δ≈1.0) passes almost unshrunk.
const EXPR_REST_DEAD: f32 = 0.06;

/// Soft-threshold retention factor: 0 at/below `dead`, ramping to 1 as
/// `mag` exceeds it via `(mag - dead) / mag` (continuous at the boundary).
#[inline]
fn soft_threshold_keep(mag: f32, dead: f32) -> f32 {
    if mag <= dead {
        0.0
    } else {
        (mag - dead) / mag
    }
}

/// Resolved expression weight (API unchanged across the v1→v2 move so
/// rendering code does not need a new type name).
#[derive(Clone, Debug)]
pub struct ResolvedExpressionWeight {
    pub name: String,
    pub weight: f32,
}

/// Per-avatar temporal state for the expression solve.
#[derive(Clone, Debug, Default)]
pub struct ExpressionState {
    /// Camera-viseme EMA per mouth shape (the eye/brow path is smoothed
    /// by the blended delta; the mouth policy bypasses it).
    mouth_viseme_ema: HashMap<String, f32>,
}

impl ExpressionState {
    pub fn reset(&mut self) {
        self.mouth_viseme_ema.clear();
    }
}

/// Resolve tracking expression weights against the avatar's named blend
/// shapes. Expression names must match the VRM 1.0 canonical identifiers.
pub fn solve_expressions(
    source: &SourceSkeleton,
    avatar_expressions: &crate::asset::ExpressionAssetSet,
    previous: Option<&[ResolvedExpressionWeight]>,
    expression_blend: f32,
    face_confidence_threshold: f32,
    mouth_source: MouthSource,
    state: &mut ExpressionState,
) -> Vec<ResolvedExpressionWeight> {
    // VRM mouth visemes whose driver (audio lip-sync vs camera) is
    // selectable. Everything else (eyes / brows / emotions) always comes
    // from the camera.
    const MOUTH_VISEMES: [&str; 5] = ["aa", "ih", "ou", "ee", "oh"];
    // Gate on the face mesh's own confidence (the FaceMesh model outputs
    // an "is this a face" sigmoid). When the gate fails, return the
    // previous weights verbatim — the avatar's expression freezes at the
    // last good value rather than snapping to neutral when the face
    // briefly turns away from camera.
    let face_conf = source.face_mesh_confidence.unwrap_or(0.0);
    if face_conf < face_confidence_threshold {
        return previous.map(|p| p.to_vec()).unwrap_or_default();
    }

    let prev_map: HashMap<&str, f32> = previous
        .map(|p| p.iter().map(|w| (w.name.as_str(), w.weight)).collect())
        .unwrap_or_default();

    avatar_expressions
        .expressions
        .iter()
        .filter_map(|expr_def| {
            let tracking = match source.expressions.iter().find(|e| e.name == expr_def.name) {
                Some(t) => t,
                None => {
                    // No tracking counterpart (e.g. an FBX body-size
                    // shape key): carry the previous weight through so
                    // manual slider edits survive the per-frame
                    // overwrite instead of being dropped to zero.
                    let weight = prev_map.get(expr_def.name.as_str()).copied().unwrap_or(0.0);
                    return Some(ResolvedExpressionWeight {
                        name: expr_def.name.clone(),
                        weight: weight.clamp(0.0, 1.0),
                    });
                }
            };
            let raw = tracking.weight.clamp(0.0, 1.0);
            let prev_w = prev_map.get(expr_def.name.as_str()).copied().unwrap_or(raw);
            // Rest deadband (eye/brow path), soft-thresholded so crossing
            // the boundary is continuous.
            let delta = raw - prev_w;
            let deadbanded = prev_w + delta * soft_threshold_keep(delta.abs(), EXPR_REST_DEAD);
            let blended = prev_w + expression_blend * (deadbanded - prev_w);
            let weight = if MOUTH_VISEMES.contains(&expr_def.name.as_str()) {
                // `prev_w` carries the audio lip-sync value: `step_lipsync`
                // runs just before the face solve each frame and writes the
                // mouth visemes into the weights `previous` points at. So
                // the source policy mixes camera against audio (`prev_w`).
                let cam = {
                    let ema = state
                        .mouth_viseme_ema
                        .entry(expr_def.name.clone())
                        .or_insert(raw);
                    *ema += expression_blend * (raw - *ema);
                    *ema
                };
                match mouth_source {
                    MouthSource::Audio => prev_w,
                    MouthSource::Image => cam,
                    MouthSource::Both => cam.max(prev_w),
                }
            } else {
                blended
            };
            Some(ResolvedExpressionWeight {
                name: expr_def.name.clone(),
                weight: weight.clamp(0.0, 1.0),
            })
        })
        .collect()
}
