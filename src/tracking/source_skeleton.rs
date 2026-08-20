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
/// * `position` — `[x, y, z]` in normalised camera coords (see module
///   docs for axis conventions). For backends that only produce 2D
///   keypoints, set `z = 0`. **The `z` component is intended for
///   *rotation* / direction-vector consumers** (retarget,
///   `compute_body_yaw_3d`, etc): for the depth-aware providers it
///   carries the unbiased relative Z signal (RTMW3D's hip-mid-anchored
///   nz), not the metric depth value. Metric depth lives on the
///   separate `metric_depth_m` field below to keep the rotation path
///   free of metric-depth-model bias.
/// * `confidence` — detector-reported confidence in `[0, 1]`. For
///   SimCC-style outputs (RTMW3D) this is the sigmoid of the heatmap
///   peak, so a single threshold gates "joint actually in frame and
///   well-localised".
/// * `metric_depth_m` — absolute camera-space depth in metres, when a
///   metric depth model (DAv2 / MoGe-2) sampled this keypoint. `None`
///   for purely 2D providers and for joints that the depth path
///   couldn't sample (off-frame, low-confidence depth window). Used
///   by *metric* consumers — root translation Z, bone-length
///   reconstruction, torso template — that genuinely need
///   real-world distances. Decoupling this from `position[2]` is
///   what lets the rotation path use the unbiased relative-Z signal
///   while metric consumers keep the high-fidelity DAv2 / MoGe-2
///   measurement.
#[derive(Clone, Copy, Debug, Default)]
pub struct SourceJoint {
    pub position: [f32; 3],
    pub confidence: f32,
    pub metric_depth_m: Option<f32>,
}

/// Provenance of a joint sample: whether the position was actually
/// measured by the detector / depth sampler, or manufactured somewhere
/// along the pipeline. Consumers that feed *statistics* (scale EMAs,
/// anchor stabilisers, calibration accumulators, depth fusion) must only
/// ingest [`JointOrigin::Observed`] samples — fabricated positions carry
/// detector-grade confidence but constant/heuristic geometry, and letting
/// them into an estimator drags it toward the fabrication constant (e.g.
/// the canonical-frontal shoulder pair polluting the torso span EMA).
///
/// Stored sparsely in [`SourceSkeleton::joint_origins`]; an absent entry
/// means `Observed` so the many existing observation sites need no change.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum JointOrigin {
    /// Position comes from the detector (and, when present, the depth
    /// window sample) for this very frame.
    #[default]
    Observed,
    /// Derived from real observations of *other* joints or *other*
    /// frames: anatomical offset from an observed neighbour, held-last
    /// value, bone-length re-pin of an observed direction.
    Extrapolated,
    /// Manufactured from constants or canonical-pose assumptions with no
    /// per-frame measurement behind the geometry (collapsed-pair
    /// canonical reconstruction, fixed anatomical fallbacks).
    Synthesized,
}

/// Which estimator produced a [`FacePose`]. The two sources have
/// *different* systematic residuals (different landmark sets, different
/// hardcoded anatomical neutrals — body ≈ 1.25, mesh ≈ 0.49 pitch
/// signal), so the per-session neutral captured during calibration is
/// only valid for the source it was measured from. Consumers that
/// subtract a calibrated neutral (`TrackingCalibration::apply_calibration`)
/// and the calibration accumulator itself must key on this tag.
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
/// The face track is currently empty — RTMW3D emits 68 face landmarks
/// (indices 23-90) but they are not yet decoded into yaw/pitch/roll;
/// the head bone falls back to spine direction. Phase C (SMIRK) will
/// drive this field along with ARKit blendshape weights.
#[derive(Clone, Copy, Debug, Default)]
pub struct FacePose {
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub confidence: f32,
    /// Which estimator produced these angles — see [`FaceSource`].
    /// Stamped by the producer (`derive_face_pose_from_body` → `Body`,
    /// `derive_face_pose_from_landmarks` → `Mesh`) and preserved
    /// through source selection / crossfade so calibration subtraction
    /// picks the neutral measured for the *same* estimator.
    pub source: FaceSource,
    /// `Some((from_source, t))` while the source-switch crossfade is
    /// easing this pose between two estimators (`t ∈ (0, 1)`, the
    /// blend fraction toward [`Self::source`]). Two consumers key on
    /// it:
    ///
    /// * `TrackingCalibration::apply_calibration` interpolates the
    ///   per-source neutrals with the *same* `t` — the raw angles are
    ///   a lerp of the two sources' raw spaces, so subtracting the
    ///   target's neutral outright would re-introduce the
    ///   inter-source neutral gap as a calibrated-space head step on
    ///   the very frame the crossfade exists to smooth (lerp is
    ///   linear, so `lerp(raw) − lerp(neutral) =
    ///   lerp(raw − neutral)`: exactly a calibrated-space blend).
    /// * The calibration hold skips these frames entirely — a mixed
    ///   pose under the target's tag would pollute that source's
    ///   neutral accumulator.
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

/// Full orientation of a tracked hand. `forward` is the wrist→fingers
/// axis (typically wrist → middle MCP) and `up` is the palm normal
/// (away from the back of the hand, toward where the fingernails
/// face when curled up). Both are unit vectors in source-skeleton
/// coords. The pair is computed by the tracker from the four MCP
/// landmarks and used by the solver to drive the wrist bone's full
/// 3-DoF rotation — without `up`, finger MCPs end up with arbitrary
/// 90° twist because the wrist→tip direction alone leaves the
/// twist axis under-constrained.
#[derive(Clone, Copy, Debug)]
pub struct HandOrientation {
    pub forward: [f32; 3],
    pub up: [f32; 3],
    pub confidence: f32,
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
    /// reference (calibration, else this frame's measured span, else an
    /// anatomical mean). The solver derives the metres→avatar-unit scale for
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
    /// Sparse provenance map for `joints` (and `fingertips`, keyed the
    /// same way). Absent entry ⇒ [`JointOrigin::Observed`]. Producers
    /// that fabricate or extrapolate a joint must record it here via
    /// [`Self::mark_origin`]; statistical consumers filter with
    /// [`Self::origin`].
    pub joint_origins: HashMap<HumanoidBone, JointOrigin>,
    /// Raw body-derived face pose for this frame, published *alongside*
    /// the selected [`Self::face`] so the calibration hold can
    /// accumulate a per-source neutral for BOTH estimators in one
    /// capture (during a frontal hold the mesh wins selection nearly
    /// every frame, so the body neutral would otherwise never be
    /// measured). Never calibration-subtracted and never consumed by
    /// the solver — calibration capture only. `None` when the body
    /// face keypoints were below the visibility floor.
    pub face_body_raw: Option<FacePose>,
    /// Auxiliary positions for finger *tips* (the keypoint beyond the
    /// `*Distal` bone). Keyed by the distal bone whose tip it represents —
    /// e.g. `fingertips[LeftIndexDistal]` is the 3D position of the left
    /// index fingertip. Used by the solver to drive the distal bone's
    /// orientation; the tip itself is not a humanoid bone. The same slot
    /// is reused for foot toe tips (`fingertips[LeftFoot]`).
    pub fingertips: HashMap<HumanoidBone, SourceJoint>,
    pub face: Option<FacePose>,
    pub expressions: Vec<SourceExpression>,
    /// Full 3-DoF orientation of the left hand (wrist) when the hand
    /// track produced enough MCP landmarks to define a palm plane.
    /// `None` when the hand is not tracked or the MCPs are too
    /// degenerate to extract a palm normal.
    pub left_hand_orientation: Option<HandOrientation>,
    pub right_hand_orientation: Option<HandOrientation>,
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
    /// (image centre for 2D-only providers; the calibration-anchor
    /// metric position for depth-aware providers). Used by
    /// the retarget to translate the avatar's `Hips`
    /// bone so the avatar follows the subject's side-step / lean-in /
    /// crouch motion instead of just spinning in place, and read by
    /// the pose-calibration modal as the per-frame anchor sample.
    ///
    /// `None` when *neither* the hip pair *nor* the shoulder pair was
    /// detected — solver leaves the avatar at its rest position.
    ///
    /// Units depend on the path. Metric (D435,
    /// `metric_frame_info.is_some()`): all three components are RAW
    /// camera **metres** with the axes flipped to the source
    /// orientation (selfie-mirror x, y-up, z toward camera → negative
    /// for a subject in front of the lens); the solver's "Metric
    /// translation" 1:1 contract and the calibration anchor fields
    /// both build on this. Legacy 2D path (`rtmw3d`-only): `x ∈
    /// [-aspect, +aspect]`, `y ∈ [-1, +1]`, `z = 0`.
    ///
    /// Use [`Self::root_anchor_is_hip`] to disambiguate hip vs
    /// shoulder anchor for downstream consumers (calibration mode
    /// matching, EMA seed selection).
    pub root_offset: Option<[f32; 3]>,
    /// `true` when [`Self::root_offset`] was derived from the hip
    /// pair (COCO 11/12); `false` when it fell back to the shoulder
    /// pair (COCO 5/6) — the `Upper Body Only` framing where the
    /// pelvis is below the camera frame. Pose calibration uses this
    /// to verify the captured anchor matches the user's chosen mode
    /// and to gate the runtime EMA seeding (a hip-calibrated session
    /// shouldn't drive translation off a shoulder-anchor frame and
    /// vice-versa).
    pub root_anchor_is_hip: bool,
    /// `Some` iff this skeleton was built from a true metric-depth backend
    /// (RealSense D435). See [`MetricFrameInfo`] — its presence is the single
    /// signal that switches the solver and render onto the metric-3D-direct
    /// path (faithful camera-space projection) instead of the monocular
    /// heuristics kept for the 2D webcam fallback. `None` for 2D providers.
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
            joint_origins: HashMap::new(),
            face_body_raw: None,
            fingertips: HashMap::new(),
            face: None,
            expressions: Vec::new(),
            left_hand_orientation: None,
            right_hand_orientation: None,
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
        // Mirrors `skeleton_from_depth::TARGET_SRC_SHOULDER_SPAN` (the source
        // normalisation target) when the shoulders are absent.
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

    /// Insert a joint only if its confidence clears `min_conf`.
    pub fn put_joint(&mut self, bone: HumanoidBone, joint: SourceJoint, min_conf: f32) {
        if joint.confidence >= min_conf {
            self.joints.insert(bone, joint);
        }
    }

    /// Provenance for `bone`. Absent entry ⇒ `Observed`.
    pub fn origin(&self, bone: HumanoidBone) -> JointOrigin {
        self.joint_origins
            .get(&bone)
            .copied()
            .unwrap_or(JointOrigin::Observed)
    }

    /// Record provenance for `bone`. `Observed` clears any prior mark so
    /// the map stays sparse and a re-observed joint doesn't keep a stale
    /// fabrication flag from an earlier fallback frame.
    pub fn mark_origin(&mut self, bone: HumanoidBone, origin: JointOrigin) {
        if origin == JointOrigin::Observed {
            self.joint_origins.remove(&bone);
        } else {
            self.joint_origins.insert(bone, origin);
        }
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
        if let Some(hand) = self.left_hand_orientation.as_mut() {
            hand.confidence *= scale;
        }
        if let Some(hand) = self.right_hand_orientation.as_mut() {
            hand.confidence *= scale;
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
        s.left_hand_orientation = Some(HandOrientation {
            forward: [1.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            confidence: 0.4,
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
        assert!(
            (s.fingertips[&HumanoidBone::LeftIndexDistal].confidence - 0.25).abs() < 1e-6,
        );
        assert!((s.face.unwrap().confidence - 0.35).abs() < 1e-6);
        assert!((s.left_hand_orientation.unwrap().confidence - 0.2).abs() < 1e-6);
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
