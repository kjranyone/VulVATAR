//! Source skeleton: the avatar-agnostic intermediate representation emitted
//! by the tracking backends; the fusion estimator publishes it as a
//! compatibility projection for the GUI overlay / expression solve.
//!
//! A `SourceSkeleton` carries sparse 3D joint positions (image-space x/y
//! plus a depth z relative to the body anchor), per-side hand landmark
//! arrays, and either a face pose or ARKit-compatible blendshape weights.
//! It deliberately does not attempt to compute per-bone rotations — that
//! is the solver's job and depends on the target avatar's rest pose.
//!
//! ## Coordinate convention
//!
//! * `x` ∈ `[-aspect, +aspect]` (camera-space, aspect pre-applied)
//! * `y` ∈ `[-1, +1]`, Y-up (image bottom is `-1`, top is `+1`)
//! * `z` ∈ depth in the same units as x/y, with `+z` pointing *toward the
//!   camera*. The body's hip midpoint is the depth origin (`z = 0`); a
//!   joint reaching toward the lens has `z > 0`.
//!
//! With a 3D-native backend (RTMW3D whole-body) the solver consumes
//! the depth directly — no foreshortening reconstruction needed.
//! Backends that only output 2D must populate `z = 0` and accept that
//! the solver cannot disambiguate forward / backward limbs.

use std::collections::HashMap;

use crate::asset::HumanoidBone;

/// One tracked joint in image-space, with depth.
///
/// * `position` — `[x, y, z]` in camera coords (see module docs for
///   axis conventions). Fusion's compatibility projection fills this
///   from the articulated-body FK.
/// * `confidence` — detector-reported confidence in `[0, 1]`. For
///   SimCC-style outputs (RTMW3D) this is the sigmoid of the heatmap
///   peak.
/// * `metric_depth_m` — absolute camera-space depth in metres when a
///   D435 sample exists for this joint. `None` when the joint was
///   not lifted from the point cloud.
#[derive(Clone, Copy, Debug, Default)]
pub struct SourceJoint {
    pub position: [f32; 3],
    pub confidence: f32,
    pub metric_depth_m: Option<f32>,
}

/// Which estimator produced a [`FacePose`]. The two sources have
/// *different* systematic residuals (different landmark sets, different
/// hardcoded anatomical neutrals — body ≈ 1.25, mesh ≈ 0.49 pitch
/// signal), so their raw angle spaces are not comparable. Consumers
/// that must not mix them (source selection / crossfade) key on this tag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FaceSource {
    /// RTMW3D body face keypoints (nose / eyes / ears, COCO 0..=4).
    #[default]
    Body,
    /// MediaPipe FaceMesh dense 478-landmark pose.
    Mesh,
}

/// Head orientation derived from facial keypoints.
///
/// Angles are in radians with neutral at `yaw = pitch = roll = 0`. Sign
/// conventions are chosen so the values feed straight into
/// [`crate::math_utils::quat_from_euler_ypr`] without a sign flip:
///
/// * **`yaw`** — positive turns the head around +Y toward camera-space +X.
/// * **`pitch`** — positive tilts the head around +X so the top of the
///   head moves toward +Z (i.e. the chin drops — "looking down").
/// * **`roll`** — positive rolls around +Z so the head leans toward +X.
///
/// The face track is filled by two producers: the body-derived
/// fallback (`derive_face_pose_from_body`) and the RTMW3D 68-landmark
/// decoder (`derive_face_pose_from_landmarks`, FaceMesh path), with
/// the producing source stamped on [`FacePose::source`]. Phase C
/// (SMIRK) will additionally drive ARKit blendshape weights.
#[derive(Clone, Copy, Debug, Default)]
pub struct FacePose {
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub confidence: f32,
    /// Which estimator produced these angles — see [`FaceSource`].
    /// Stamped by the producer (`derive_face_pose_from_body` → `Body`,
    /// `derive_face_pose_from_landmarks` → `Mesh`) and preserved
    /// through source selection / crossfade so downstream consumers can
    /// tell which estimator's angle space they are reading.
    pub source: FaceSource,
    /// `Some((from_source, t))` while the source-switch crossfade is
    /// easing this pose between two estimators (`t ∈ (0, 1)`, the
    /// blend fraction toward [`Self::source`]). The raw angles are a
    /// lerp of the two sources' raw spaces, so a consumer that needs a
    /// source-pure value must blend per source with the same `t`.
    ///
    /// `None` on every steady-state frame.
    pub blend: Option<(FaceSource, f32)>,
}

/// A named expression blend-shape weight (e.g. `"blink" → 0.6`). Names use the
/// VRM 1.0 canonical identifier space; the avatar-side retarget maps them to
/// matching `ExpressionDef` entries at solve time.
#[derive(Clone, Debug)]
pub struct SourceExpression {
    pub name: String,
    pub weight: f32,
}

/// Pinhole camera intrinsics captured from the depth sensor. Plain fields
/// (a copy of the realsense `CamIntrinsics`) so this avatar-agnostic module
/// stays free of backend-specific types. Consumed by the 1:1 sensor-matched
/// render to reproduce the exact projection the physical camera saw.
#[derive(Clone, Copy, Debug)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
}

/// Present on a [`SourceSkeleton`] iff it was built from a true metric-depth
/// backend (RealSense D435). Its presence is THE signal that downstream
/// consumers (solver orientation/root, sensor-matched render) should treat
/// the joint positions as faithful camera-space 3D rather than monocular
/// heuristics: `metric_frame_info.is_some()` == "3D-native / metric path".
///
/// It carries the pieces the metric path needs beyond the joint positions:
/// the raw camera-space anchor (for 1:1 metric root placement), whether it
/// came from the hip or shoulder pair, the metres-per-source-unit scale used
/// to normalise the positions into the source frame, and the camera
/// intrinsics (for the mirror render).
#[derive(Clone, Copy, Debug)]
pub struct MetricFrameInfo {
    /// Anchor origin in RAW camera metres (x-right, y-down, z-forward) —
    /// i.e. the torso-fit anchor (`TorsoFit.anchor_cam`) before the
    /// source-frame axis flip.
    pub anchor_cam_m: [f32; 3],
    /// `true` when the anchor came from the hip pair, `false` for shoulders.
    pub anchor_is_hip: bool,
    /// Metres per source unit used to normalise `SourceJoint.position` into
    /// the source frame. `source_units = metres / mpsu`.
    pub mpsu: f32,
    /// The subject's real shoulder span in metres used as the normalisation
    /// reference (this frame's measured span, else an anatomical mean).
    /// The solver derives the metres→avatar-unit scale for
    /// 1:1 root placement from this: `avatar_rest_shoulder_span / reference_span_m`.
    pub reference_span_m: f32,
    /// Colour-image pinhole intrinsics for the frame this skeleton came from.
    pub intrinsics: CameraIntrinsics,
}

/// One tracker sample.
///
/// Body joints are sparse: only the humanoid bones for which the detector
/// produced (or the tracker derived) a 3D position are populated. Face pose,
/// if `Some`, takes precedence over any rotation that might otherwise be
/// computed from the Head/Neck joint pair — the head is driven by facial
/// keypoint geometry, not by the nose-to-shoulder direction.
///
/// Hand finger joints (thumb/index/middle/ring/little × proximal/
/// intermediate/distal) live in `joints` alongside body bones once a
/// hand landmarker has run; the auxiliary `fingertips` map carries the
/// tip beyond each distal bone, since the tip itself does not have a
/// humanoid bone slot.
#[derive(Clone, Debug, Default)]
pub struct SourceSkeleton {
    pub source_timestamp: u64,
    /// Device (hardware-clock) capture time of the camera frame this
    /// sample was estimated from, in milliseconds. `Some` only on the
    /// real capture path (D435) — synthetic / bench / test producers
    /// leave it `None`.
    ///
    /// This is the pipeline's single time base: consecutive samples'
    /// deltas give the true inter-*capture* interval, which is what the
    /// solver's 1€ filters and rest gates must use as `dt`. Wall-clock
    /// deltas measured at the consumer conflate capture cadence with
    /// scheduling (render rate, inference stalls) and were the root
    /// cause of the render-fps-dependent filter behaviour. Also the
    /// dedup key half that lets the solver skip re-filtering when the
    /// same sample is observed on multiple render frames.
    pub capture_timestamp_ms: Option<f64>,
    pub joints: HashMap<HumanoidBone, SourceJoint>,
    /// Auxiliary positions for finger *tips* (the keypoint beyond the
    /// `*Distal` bone). Keyed by the distal bone whose tip it represents —
    /// e.g. `fingertips[LeftIndexDistal]` is the 3D position of the left
    /// index fingertip. Used by the solver to drive the distal bone's
    /// orientation; the tip itself is not a humanoid bone. The same slot
    /// is reused for foot toe tips (`fingertips[LeftFoot]`).
    pub fingertips: HashMap<HumanoidBone, SourceJoint>,
    pub face: Option<FacePose>,
    pub expressions: Vec<SourceExpression>,
    /// Optional FaceMesh-model "is this a face" confidence (post sigmoid)
    /// for the frame's face crop. Distinct from `face.confidence`, which
    /// is body-derived (min over the 5 COCO face landmarks): this one
    /// is the dedicated face-mesh model's own output and is what the
    /// solver gates `expressions` on via `face_confidence_threshold`.
    /// `None` when no face mesh ran (no face crop, no model).
    pub face_mesh_confidence: Option<f32>,
    /// Overall scalar detection confidence (used for stale/quality gating
    /// upstream). `0.0` means "no person detected".
    pub overall_confidence: f32,
    /// Body-anchor offset from a neutral camera-frame reference
    /// (image centre for 2D-only providers; the torso-fit anchor's
    /// metric position for depth-aware providers). Used by
    /// the retarget to translate the avatar's `Hips`
    /// bone so the avatar follows the subject's side-step / lean-in /
    /// crouch motion instead of just spinning in place.
    ///
    /// `None` when *neither* the hip pair *nor* the shoulder pair was
    /// detected — solver leaves the avatar at its rest position.
    ///
    /// Units depend on the path. Metric (D435,
    /// `metric_frame_info.is_some()`): all three components are RAW
    /// camera **metres** with the axes flipped to the source
    /// orientation (selfie-mirror x, y-up, z toward camera → negative
    /// for a subject in front of the lens); the solver's "Metric
    /// translation" 1:1 contract builds on this. Legacy 2D path
    /// (SimCC-era detectors): `x ∈ [-aspect, +aspect]`, `y ∈ [-1, +1]`,
    /// `z = 0`.
    ///
    /// Use [`Self::root_anchor_is_hip`] to disambiguate hip vs
    /// shoulder anchor for downstream consumers (EMA seed selection).
    pub root_offset: Option<[f32; 3]>,
    /// `true` when [`Self::root_offset`] was derived from the hip
    /// pair (COCO 11/12); `false` when it fell back to the shoulder
    /// pair (COCO 5/6) — the `Upper Body Only` framing where the
    /// pelvis is below the camera frame. Consumers gate the runtime
    /// EMA seeding on this (a hip-anchored session shouldn't drive
    /// translation off a shoulder-anchor frame and vice-versa).
    pub root_anchor_is_hip: bool,
    /// `Some` iff this skeleton was built from a true metric-depth backend
    /// (RealSense D435). See [`MetricFrameInfo`] — its presence is the
    /// signal that the sample carries true camera metres / intrinsics.
    /// `None` for synthetic / 2D-only producers.
    pub metric_frame_info: Option<MetricFrameInfo>,
    /// Tracking-v2 output: joint rotations + root + per-joint σ from the
    /// fusion estimator (`tracking::fusion`). `Some` ⇒ the avatar is
    /// driven by `avatar::retarget` from these rotations; the joint
    /// positions above are then a compatibility projection for the GUI.
    pub rig: Option<std::sync::Arc<crate::tracking::fusion::output::RigPose>>,
}

impl SourceSkeleton {
    pub fn empty(source_timestamp: u64) -> Self {
        Self {
            source_timestamp,
            capture_timestamp_ms: None,
            joints: HashMap::new(),
            fingertips: HashMap::new(),
            face: None,
            expressions: Vec::new(),
            face_mesh_confidence: None,
            overall_confidence: 0.0,
            root_offset: None,
            root_anchor_is_hip: false,
            metric_frame_info: None,
            rig: None,
        }
    }

    /// Bench/test helper — stamp a synthetic [`MetricFrameInfo`] so this
    /// skeleton drives the solver's metric path.
    ///
    /// Since the D435-exclusive rebuild, the pose consumer
    /// reads `metric_frame_info` only for the root-translation scale
    /// (`avatar_span / reference_span_m`); every other solver behaviour is
    /// identical with or without it. Image-only benches have no depth to feed
    /// through `set_external_depth`, so this lets them exercise the same
    /// `Some(..)` path the shipping D435 pipeline takes rather than the `None`
    /// 1:1 fallback. `reference_span_m` is the skeleton's own measured L/R
    /// `UpperArm` span so the scale lands as it would on a real metric frame;
    /// the anchor / intrinsics fields the solver never reads carry
    /// placeholders. Not for production use — the real path sets this in
    /// the metric depth path.
    pub fn stamp_synthetic_metric_frame(&mut self) {
        // Fallback when the shoulders are absent (~adult shoulder span).
        const FALLBACK_SPAN_M: f32 = 0.75;
        let reference_span_m = match (
            self.joints.get(&HumanoidBone::LeftUpperArm),
            self.joints.get(&HumanoidBone::RightUpperArm),
        ) {
            (Some(l), Some(r)) => {
                let dx = l.position[0] - r.position[0];
                let dy = l.position[1] - r.position[1];
                let dz = l.position[2] - r.position[2];
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                if d > 0.05 {
                    d
                } else {
                    FALLBACK_SPAN_M
                }
            }
            _ => FALLBACK_SPAN_M,
        };
        self.metric_frame_info = Some(MetricFrameInfo {
            anchor_cam_m: self.root_offset.unwrap_or([0.0, 0.0, 0.0]),
            anchor_is_hip: self.root_anchor_is_hip,
            mpsu: 1.0,
            reference_span_m,
            intrinsics: CameraIntrinsics {
                fx: 600.0,
                fy: 600.0,
                cx: 320.0,
                cy: 240.0,
                width: 640,
                height: 480,
            },
        });
    }

    /// Multiply every confidence channel by `scale`, clamped to
    /// `[0.0, 1.0]`. Used by the tracking hold/fade policy in
    /// [`crate::app::Application::run_frame`]: when the mailbox is
    /// stale but still inside the hold window, the last known sample
    /// is reused with confidence decayed linearly so that joints fall
    /// below the solver's threshold gradually rather than the avatar
    /// snapping back to its base pose.
    ///
    /// Positions are left untouched — the avatar should *freeze* and
    /// then fade, not blend toward an arbitrary intermediate position
    /// while the tracker is briefly silent. Expression weights are
    /// also left untouched: `solve_expressions` reads
    /// `face_mesh_confidence` against the user threshold and skips
    /// updates when the face is below confidence, so the smoothing
    /// path takes care of the expression decay implicitly.
    pub fn scale_confidence(&mut self, scale: f32) {
        let scale = scale.clamp(0.0, 1.0);
        for joint in self.joints.values_mut() {
            joint.confidence *= scale;
        }
        for tip in self.fingertips.values_mut() {
            tip.confidence *= scale;
        }
        if let Some(face) = self.face.as_mut() {
            face.confidence *= scale;
        }
        if let Some(conf) = self.face_mesh_confidence.as_mut() {
            *conf *= scale;
        }
        self.overall_confidence *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::HumanoidBone;

    fn populated() -> SourceSkeleton {
        let mut s = SourceSkeleton::empty(0);
        s.overall_confidence = 0.8;
        s.face_mesh_confidence = Some(0.9);
        s.joints.insert(
            HumanoidBone::Hips,
            SourceJoint {
                position: [0.0, 0.0, 0.0],
                confidence: 0.6,
                metric_depth_m: None,
            },
        );
        s.fingertips.insert(
            HumanoidBone::LeftIndexDistal,
            SourceJoint {
                position: [0.0, 0.0, 0.0],
                confidence: 0.5,
                metric_depth_m: None,
            },
        );
        s.face = Some(FacePose {
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            confidence: 0.7,
            source: FaceSource::Body,
            ..Default::default()
        });
        s
    }

    #[test]
    fn scale_confidence_halves_every_channel() {
        let mut s = populated();
        let pos = s.joints[&HumanoidBone::Hips].position;
        s.scale_confidence(0.5);
        assert!((s.overall_confidence - 0.4).abs() < 1e-6);
        assert!((s.face_mesh_confidence.unwrap() - 0.45).abs() < 1e-6);
        assert!((s.joints[&HumanoidBone::Hips].confidence - 0.3).abs() < 1e-6);
        assert!((s.fingertips[&HumanoidBone::LeftIndexDistal].confidence - 0.25).abs() < 1e-6,);
        assert!((s.face.unwrap().confidence - 0.35).abs() < 1e-6);
        assert_eq!(
            s.joints[&HumanoidBone::Hips].position,
            pos,
            "scaling confidence must not perturb joint positions",
        );
    }

    #[test]
    fn scale_confidence_zero_drives_all_signals_to_zero() {
        let mut s = populated();
        s.scale_confidence(0.0);
        assert_eq!(s.overall_confidence, 0.0);
        assert_eq!(s.face_mesh_confidence, Some(0.0));
        assert_eq!(s.joints[&HumanoidBone::Hips].confidence, 0.0);
        assert_eq!(s.face.unwrap().confidence, 0.0);
    }

    #[test]
    fn scale_confidence_clamps_out_of_range_input() {
        let mut s = populated();
        s.scale_confidence(2.0);
        // Clamped to 1.0 — original values preserved.
        assert!((s.overall_confidence - 0.8).abs() < 1e-6);
        let mut t = populated();
        t.scale_confidence(-1.0);
        // Clamped to 0.0.
        assert_eq!(t.overall_confidence, 0.0);
    }
}
