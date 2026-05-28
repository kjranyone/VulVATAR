//! ViTPose-Wholebody (xy) + RTMW3D (z) source-level hybrid provider.
//!
//! ViTPose-L wins on 2D keypoint accuracy (esp. deskcrop / off-frame
//! hands) but is 2D-only, so subtle body twist with no shoulder
//! foreshortening signal silently collapses to a frontal source
//! skeleton (the 2026-05-25 `frilly_chest_small_pose003_three_quarter_
//! torso_twist` case had source z=0 throughout, avatar rendered fully
//! frontal). RTMW3D's z output captures that twist (z=-0.485 on the
//! same shoulder_line) but its 2D xy is noticeably worse in
//! deskcrop / arm-forward scenarios (L_upper_arm max 18.7° vs ViTPose
//! 0.0° in our 327-image benchmark).
//!
//! This provider runs both inferences and merges at the SourceSkeleton
//! joint level: ViTPose owns `position.x` and `position.y`, RTMW3D
//! provides `position.z`. Confidence is the min of the two (so a joint
//! is gated whenever either side is uncertain).
//!
//! ## Cost
//!
//! Two YOLOX crops + two pose inferences per frame — roughly 2× the
//! cost of a single-provider path. Intended for offline `validate_pipeline`
//! and best-quality offline rendering, NOT live GUI use yet (we'd need
//! a shared YOLOX worker and an async second-inference path to hit
//! realtime).

use std::path::Path;

use super::provider::PoseProvider;
use super::PoseEstimate;

#[cfg(feature = "inference")]
use super::rtmw3d::Rtmw3dInference;
#[cfg(feature = "inference")]
use super::vitpose_provider::VitposeWholebodyProvider;
#[cfg(feature = "inference")]
use log::{debug, info};

pub struct VitposeRtmw3dHybridProvider {
    #[cfg(feature = "inference")]
    vitpose: VitposeWholebodyProvider,
    #[cfg(feature = "inference")]
    rtmw3d: Rtmw3dInference,
    load_warnings: Vec<String>,
}

impl VitposeRtmw3dHybridProvider {
    #[cfg(feature = "inference")]
    pub(super) fn from_models_dir(models_dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = models_dir.as_ref();
        let mut vitpose = VitposeWholebodyProvider::from_models_dir(dir)?;
        let rtmw3d = Rtmw3dInference::from_models_dir(dir)?;

        // Surface any warnings from the inner ViTPose provider so the
        // top-level loader can report them once instead of swallowing
        // them inside the hybrid wrapper.
        let load_warnings = vitpose.take_load_warnings();

        info!("ViTPose+RTMW3D hybrid provider ready (xy=ViTPose, z=RTMW3D)");

        Ok(Self {
            vitpose,
            rtmw3d,
            load_warnings,
        })
    }

    #[cfg(not(feature = "inference"))]
    pub(super) fn from_models_dir(_: impl AsRef<Path>) -> Result<Self, String> {
        Err("ViTPose+RTMW3D hybrid provider requires the `inference` cargo feature".to_string())
    }
}

impl PoseProvider for VitposeRtmw3dHybridProvider {
    fn label(&self) -> String {
        #[cfg(feature = "inference")]
        {
            format!(
                "ViTPose+RTMW3D-z hybrid / vp={}, rtmw3d={}",
                self.vitpose.label(),
                self.rtmw3d.backend().label()
            )
        }
        #[cfg(not(feature = "inference"))]
        {
            "ViTPose+RTMW3D hybrid (inference disabled)".to_string()
        }
    }

    fn take_load_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.load_warnings)
    }

    fn set_calibration(&mut self, calibration: Option<crate::tracking::PoseCalibration>) {
        #[cfg(feature = "inference")]
        {
            self.vitpose.set_calibration(calibration);
        }
        #[cfg(not(feature = "inference"))]
        let _ = calibration;
    }

    fn set_calibration_mode_hint(&mut self, hint: Option<crate::tracking::CalibrationMode>) {
        #[cfg(feature = "inference")]
        {
            self.vitpose.set_calibration_mode_hint(hint);
        }
        #[cfg(not(feature = "inference"))]
        let _ = hint;
    }

    fn estimate_pose(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> PoseEstimate {
        #[cfg(feature = "inference")]
        {
            let t_total = std::time::Instant::now();

            let vp_est = self.vitpose.estimate_pose(rgb_data, width, height, frame_index);
            let dt_vp = t_total.elapsed();

            let t_rm = std::time::Instant::now();
            let rm_est = self.rtmw3d.estimate_pose(rgb_data, width, height, frame_index);
            let dt_rm = t_rm.elapsed();

            // Merge: keep ViTPose's xy + everything else, but inject
            // RTMW3D's z into joints where both inferences agree it's
            // confidently observed. RTMW3D's z degrades when the
            // subject is partially out of frame (deskcrop) — the model
            // assumes a full-body context for its z classifier and a
            // missing hip / leg signal pushes the upper-body z toward
            // noise (depth_006 hand-pointing case showed L_upper_arm
            // c=0.6 with z=+0.068 that contradicted the actual 2D
            // arm-down posture, producing 18° avatar deflection).
            //
            // Gate at 0.7 for z injection while keeping the lower
            // 0.30 score-metric floor independent — the z signal is
            // noisier than the keypoint position itself, so it needs
            // stricter qualification. Joints below the threshold fall
            // back to ViTPose's planar z=0, which is the same source
            // skeleton the proven `vitpose-wholebody` provider uses.
            // Torso-only z injection. Limb z signals from RTMW3D
            // contaminate the avatar in deskcrop framing (depth_006
            // L_upper_arm c=0.59 z=+0.068 hallucinated forward-extend
            // when the photo showed arm at the side, producing 5.9°
            // regression). The body-twist signal lives in torso /
            // spine / clavicle z values; once those are set, FK + the
            // solver's per-bone driving propagates twist down the
            // limbs naturally, without inheriting RTMW3D's noisier
            // per-limb depth.
            //
            // Decision matrix (RTMW3D conf at twist case ≈ 0.70,
            // deskcrop case ≈ 0.60):
            //   * gate 0.7 + all bones: twist blocked (boundary 0.70)
            //   * gate 0.5 + all bones: deskcrop regresses (limbs noisy)
            //   * gate 0.5 + torso-only (this branch): both fixed
            use crate::asset::HumanoidBone;
            const TORSO_Z_CONF_FLOOR: f32 = 0.5;
            let is_torso = |b: &HumanoidBone| -> bool {
                matches!(
                    b,
                    HumanoidBone::Hips
                        | HumanoidBone::Spine
                        | HumanoidBone::Chest
                        | HumanoidBone::UpperChest
                        | HumanoidBone::Neck
                        | HumanoidBone::Head
                        | HumanoidBone::LeftShoulder
                        | HumanoidBone::RightShoulder
                )
            };
            let mut merged = vp_est.skeleton;
            for (bone, vp_joint) in merged.joints.iter_mut() {
                if let Some(rm_joint) = rm_est.skeleton.joints.get(bone) {
                    if is_torso(bone) && rm_joint.confidence >= TORSO_Z_CONF_FLOOR {
                        vp_joint.position[2] = rm_joint.position[2];
                    }
                    vp_joint.confidence = vp_joint.confidence.min(rm_joint.confidence);
                }
            }
            // Fingertips: never inject z. Their depth is meaningless
            // for body twist and propagates noise into hand-bone IK.
            // root_offset z: ViTPose emits 0 (planar), RTMW3D carries
            // the hip-anchor depth offset relative to the image-centre
            // ray. Always preferred over ViTPose's 0 when present
            // (root translation needs *some* depth signal and ViTPose
            // gives none — even noisy RTMW3D z is more informative
            // than zero for the root anchor).
            if let (Some(vp_root), Some(rm_root)) =
                (merged.root_offset.as_mut(), rm_est.skeleton.root_offset)
            {
                vp_root[2] = rm_root[2];
            }

            let dt_total = t_total.elapsed();
            debug!(
                "Hybrid timing: total={:>5.1}ms vp={:>5.1}ms rm={:>5.1}ms",
                dt_total.as_secs_f32() * 1000.0,
                dt_vp.as_secs_f32() * 1000.0,
                dt_rm.as_secs_f32() * 1000.0,
            );

            PoseEstimate {
                annotation: vp_est.annotation,
                skeleton: merged,
            }
        }

        #[cfg(not(feature = "inference"))]
        {
            let _ = (rgb_data, width, height);
            empty_estimate(frame_index)
        }
    }
}

#[cfg(not(feature = "inference"))]
fn empty_estimate(frame_index: u64) -> PoseEstimate {
    PoseEstimate {
        annotation: DetectionAnnotation::default(),
        skeleton: SourceSkeleton::empty(frame_index),
    }
}
