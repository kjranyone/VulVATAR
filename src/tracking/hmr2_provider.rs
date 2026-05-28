//! HMR2 (4D-Humans) SMPL-parametric pose provider + MediaPipe Hands.
//!
//! Two-engine architecture chosen to avoid GPU-command stacking on
//! Intel Arc B570 (see [[dav2-directml-tdr]]):
//!   * **Body**: HMR2 ONNX on DirectML → 13 ms / frame
//!   * **Hands**: MediaPipe Hand Landmarker ONNX on CPU → ~1.3 ms / hand
//!
//! HMR2 emits SMPL joint rotations from a single-image SMPL fit; the
//! handle for finger pose comes from a separate CPU model so the DML
//! command queue stays uncontested.

use std::path::Path;

use super::provider::PoseProvider;
use super::{DetectionAnnotation, PoseEstimate, SourceSkeleton};

#[cfg(feature = "inference")]
use super::hand_mediapipe::HandLandmarker;
#[cfg(feature = "inference")]
use super::hmr2::{smpl, Hmr2Inference, IMAGE_SIZE};
#[cfg(feature = "inference")]
use super::palm_detector::{PalmDetection, PalmDetector};
#[cfg(feature = "inference")]
use super::rtmw3d::{crop_rgb, pad_and_clamp_bbox};
#[cfg(feature = "inference")]
use super::source_skeleton::SourceJoint;
#[cfg(feature = "inference")]
use super::yolox::YoloxPersonDetector;
#[cfg(feature = "inference")]
use super::face_mediapipe::{FaceBbox, FaceMeshEp, FaceMeshInference};
#[cfg(feature = "inference")]
use crate::asset::HumanoidBone;
#[cfg(feature = "inference")]
use log::{debug, info, warn};

pub struct Hmr2Provider {
    #[cfg(feature = "inference")]
    hmr2: Hmr2Inference,
    #[cfg(feature = "inference")]
    hand_landmarker: Option<HandLandmarker>,
    #[cfg(feature = "inference")]
    palm_detector: Option<PalmDetector>,
    #[cfg(feature = "inference")]
    face_mesh: Option<FaceMeshInference>,
    #[cfg(feature = "inference")]
    person_detector: Option<YoloxPersonDetector>,
    load_warnings: Vec<String>,
}

impl Hmr2Provider {
    #[cfg(feature = "inference")]
    pub(super) fn from_models_dir(models_dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = models_dir.as_ref();
        let fp16 = dir.join("hmr2_fp16.onnx");
        let fp32 = dir.join("hmr2.onnx");
        let model_path = if fp16.is_file() {
            fp16
        } else if fp32.is_file() {
            fp32
        } else {
            return Err(format!(
                "HMR2 model not found at {} or {}. Run hmr2_export/export_onnx.py + fp16.py.",
                fp16.display(),
                fp32.display()
            ));
        };
        let hmr2 = Hmr2Inference::from_model_path(&model_path)?;

        let mut load_warnings = Vec::new();
        let hand_model_path = dir.join("mediapipe_hand_landmark.onnx");
        let hand_landmarker = if hand_model_path.is_file() {
            match HandLandmarker::from_model_path(&hand_model_path) {
                Ok(h) => Some(h),
                Err(e) => {
                    warn!("MediaPipe hand landmarker failed to load: {}", e);
                    load_warnings.push(format!("hand landmarker load: {}", e));
                    None
                }
            }
        } else {
            let msg = format!(
                "MediaPipe hand landmarker not found at {}; fingers will stay in rest pose",
                hand_model_path.display()
            );
            warn!("{}", msg);
            load_warnings.push(msg);
            None
        };

        let palm_model_path = dir.join("mediapipe_palm_detection.onnx");
        let palm_detector = if palm_model_path.is_file() {
            match PalmDetector::from_model_path(&palm_model_path) {
                Ok(p) => Some(p),
                Err(e) => {
                    warn!("MediaPipe palm detector failed to load: {}", e);
                    load_warnings.push(format!("palm detector load: {}", e));
                    None
                }
            }
        } else {
            let msg = format!(
                "MediaPipe palm detector not found at {}; hand crops will fall back to HMR2 wrist projection",
                palm_model_path.display()
            );
            warn!("{}", msg);
            load_warnings.push(msg);
            None
        };

        // MediaPipe FaceMesh for facial expressions (blendshapes). HMR2 is a
        // body+hands model with no facial landmarks, so without this the
        // avatar's face stays neutral even with Face Tracking enabled. CPU EP
        // (ForceCpu) keeps DirectML clear for the HMR2 body model + Vulkan
        // render — same rationale as the CPU-pinned hand stage.
        let face_mesh = match FaceMeshInference::try_from_models_dir(dir, FaceMeshEp::ForceCpu) {
            Ok(Some(f)) => Some(f),
            Ok(None) => {
                let msg = "MediaPipe FaceMesh not found (face_landmark.onnx / face_blendshapes.onnx); facial expressions will stay neutral".to_string();
                warn!("{}", msg);
                load_warnings.push(msg);
                None
            }
            Err(e) => {
                warn!("MediaPipe FaceMesh failed to load: {}", e);
                load_warnings.push(format!("face mesh load: {}", e));
                None
            }
        };

        let person_detector = match YoloxPersonDetector::try_from_models_dir(dir) {
            Ok(Some(d)) => {
                info!("YOLOX person detector loaded ({})", d.backend().label());
                Some(d)
            }
            Ok(None) => {
                let msg = "YOLOX person detector not found; HMR2 will run whole-frame and is likely to produce empty heatmaps. Install via dev.ps1.".to_string();
                warn!("{}", msg);
                load_warnings.push(msg);
                None
            }
            Err(e) => {
                let msg = format!(
                    "YOLOX person detector failed to load: {}. HMR2 will run whole-frame.",
                    e
                );
                warn!("{}", msg);
                load_warnings.push(msg);
                None
            }
        };

        info!(
            "HMR2 provider ready (HMR2: {}, model: {}, YOLOX: {}, Hands: {}, Face: {})",
            hmr2.backend().label(),
            model_path.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
            if person_detector.is_some() { "on" } else { "off" },
            if hand_landmarker.is_some() { "on" } else { "off" },
            if face_mesh.is_some() { "on" } else { "off" }
        );

        Ok(Self {
            hmr2,
            hand_landmarker,
            palm_detector,
            face_mesh,
            person_detector,
            load_warnings,
        })
    }

    #[cfg(not(feature = "inference"))]
    pub(super) fn from_models_dir(_: impl AsRef<Path>) -> Result<Self, String> {
        Err("HMR2 provider requires the `inference` cargo feature".to_string())
    }
}

impl PoseProvider for Hmr2Provider {
    fn label(&self) -> String {
        #[cfg(feature = "inference")]
        {
            format!(
                "HMR2 / {} + MediaPipe Hands (CPU){}",
                self.hmr2.backend().label(),
                if self.hand_landmarker.is_some() { "" } else { " [hands off]" }
            )
        }
        #[cfg(not(feature = "inference"))]
        {
            "HMR2 (inference disabled)".to_string()
        }
    }

    fn take_load_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.load_warnings)
    }

    fn estimate_pose(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> PoseEstimate {
        #[cfg(feature = "inference")]
        return self.estimate_pose_internal(rgb_data, width, height, frame_index);

        #[cfg(not(feature = "inference"))]
        {
            let _ = (rgb_data, width, height);
            PoseEstimate {
                annotation: DetectionAnnotation::default(),
                skeleton: SourceSkeleton::empty(frame_index),
            }
        }
    }
}

#[cfg(feature = "inference")]
impl Hmr2Provider {
    fn estimate_pose_internal(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> PoseEstimate {
        let expected = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(3);
        if rgb_data.len() < expected || width == 0 || height == 0 {
            return empty_estimate(frame_index);
        }

        let t_total = std::time::Instant::now();

        // YOLOX person crop.
        let t_yolox = std::time::Instant::now();
        let (cropped, crop_w, crop_h, crop_origin, annotation_bbox) = {
            let bbox_opt = self
                .person_detector
                .as_mut()
                .and_then(|d| d.detect_largest_person(rgb_data, width, height));
            match bbox_opt {
                Some(bbox) => {
                    let (cx1, cy1, cx2, cy2) =
                        pad_and_clamp_bbox(&bbox, width, height, 0.25);
                    let cw = cx2 - cx1;
                    let ch = cy2 - cy1;
                    if cw < 32 || ch < 32 {
                        (None, width, height, (0, 0), None)
                    } else {
                        let crop = crop_rgb(rgb_data, width, height, cx1, cy1, cw, ch);
                        let ann = Some((
                            bbox.x1 / width as f32,
                            bbox.y1 / height as f32,
                            bbox.x2 / width as f32,
                            bbox.y2 / height as f32,
                        ));
                        (Some(crop), cw, ch, (cx1, cy1), ann)
                    }
                }
                None => (None, width, height, (0, 0), None),
            }
        };
        let infer_rgb: &[u8] = match &cropped {
            Some(buf) => buf.as_slice(),
            None => rgb_data,
        };
        let dt_yolox = t_yolox.elapsed();

        let t_infer = std::time::Instant::now();
        let pose = match self.hmr2.estimate(infer_rgb, crop_w, crop_h) {
            Some(p) => p,
            None => return empty_estimate(frame_index),
        };
        let dt_infer = t_infer.elapsed();

        // FK once in source frame for body keypoints, once in camera
        // frame for hand bbox projection.
        let world_joints = smpl::forward_kinematics(&pose.global_orient, &pose.body_pose);
        let camera_joints = smpl::forward_kinematics_camera_frame(
            &pose.global_orient,
            &pose.body_pose,
        );

        let mut skeleton = SourceSkeleton::empty(frame_index);
        skeleton.overall_confidence = 1.0;
        skeleton.root_anchor_is_hip = true;
        for (smpl_idx, joint_pos) in world_joints.iter().enumerate() {
            if let Some(vrm_bone) = smpl::smpl_index_to_vrm(smpl_idx) {
                let source_pos = [-joint_pos[0], joint_pos[1], -joint_pos[2]];
                skeleton.joints.insert(
                    vrm_bone,
                    SourceJoint {
                        position: source_pos,
                        confidence: 1.0,
                        metric_depth_m: None,
                    },
                );
            }
        }

        // Hand landmarker pass. Palm detector (CPU) finds up to 4
        // hand bboxes in the YOLOX-cropped image; for each we run the
        // hand landmarker. Falls back to HMR2 wrist projection when
        // the palm detector is missing — that path is known-shaky
        // (HMR2 trained on full-body, deskcrop pred_cam underflows the
        // wrist into the thigh area) so it's only used as last resort.
        let dt_hands = if let Some(hand) = self.hand_landmarker.as_mut() {
            let t_hands = std::time::Instant::now();
            let infer_w = crop_w as i32;
            let infer_h = crop_h as i32;

            // Build hand-region list either from palm detection (preferred)
            // or HMR2 wrist projection (fallback).
            #[derive(Clone)]
            struct HandRegion {
                hcx1: i32,
                hcy1: i32,
                hw: u32,
                hh: u32,
                avatar_left_hint: Option<bool>,
            }
            let mut regions: Vec<HandRegion> = Vec::new();
            if let Some(palm) = self.palm_detector.as_mut() {
                let detections: Vec<PalmDetection> =
                    palm.detect(infer_rgb, crop_w, crop_h);
                // HMR2 wrist projections in YOLOX crop pixel coords,
                // used to disambiguate L/R for each palm bbox. The
                // pred_cam projection may have absolute-position bias
                // (deskcrop offsets the body downward) but the
                // RELATIVE wrist positions (left vs right side of the
                // person) remain reliable, so distance-to-nearest-
                // projected-wrist gives a clean L/R assignment that's
                // independent of whether the photo is selfie-mirrored.
                let scale_x = infer_w as f32 / IMAGE_SIZE as f32;
                let scale_y = infer_h as f32 / IMAGE_SIZE as f32;
                let wrist_xy = |smpl_idx: usize| -> (f32, f32) {
                    let (u_px, v_px) = project_to_crop_px(
                        &camera_joints[smpl_idx],
                        &pose.pred_cam,
                        IMAGE_SIZE as i32,
                    );
                    (u_px * scale_x, v_px * scale_y)
                };
                let l_wrist_xy = wrist_xy(20); // SMPL 20 → AVATAR's RIGHT (mirror)
                let r_wrist_xy = wrist_xy(21); // SMPL 21 → AVATAR's LEFT
                // Single-assignment: highest-scoring palm claims the
                // closer wrist; the next palm gets the remaining one.
                let mut sorted = detections.clone();
                sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                let mut claim_l = false;
                let mut claim_r = false;
                // Distance gate: a palm detection more than 50 % of the
                // crop's longer side from BOTH wrist projections is
                // probably a false positive (background object, knee,
                // etc.) — skip it rather than misassign.
                let max_distance_sq = {
                    let l = (infer_w.max(infer_h) as f32) * 0.5;
                    l * l
                };
                for det in &sorted {
                    let cx = (det.x1 + det.x2) * 0.5;
                    let cy = (det.y1 + det.y2) * 0.5;
                    let dist_l = (cx - l_wrist_xy.0).powi(2) + (cy - l_wrist_xy.1).powi(2);
                    let dist_r = (cx - r_wrist_xy.0).powi(2) + (cy - r_wrist_xy.1).powi(2);
                    if dist_l.min(dist_r) > max_distance_sq {
                        continue;
                    }
                    let prefer_l = dist_l < dist_r;
                    let avatar_left = if prefer_l && !claim_l {
                        claim_l = true;
                        false // SMPL 20 (L_Wrist) → avatar's RIGHT
                    } else if !claim_r {
                        claim_r = true;
                        true // SMPL 21 (R_Wrist) → avatar's LEFT
                    } else if !claim_l {
                        claim_l = true;
                        false
                    } else {
                        continue;
                    };
                    let side = ((det.x2 - det.x1).max(det.y2 - det.y1) * 1.7)
                        .max(96.0) as i32;
                    let hcx1 = ((cx - side as f32 * 0.5) as i32).clamp(0, infer_w - 1);
                    let hcy1 = ((cy - side as f32 * 0.5) as i32).clamp(0, infer_h - 1);
                    let hcx2 = ((cx + side as f32 * 0.5) as i32).clamp(1, infer_w);
                    let hcy2 = ((cy + side as f32 * 0.5) as i32).clamp(1, infer_h);
                    regions.push(HandRegion {
                        hcx1,
                        hcy1,
                        hw: (hcx2 - hcx1) as u32,
                        hh: (hcy2 - hcy1) as u32,
                        avatar_left_hint: Some(avatar_left),
                    });
                    if claim_l && claim_r {
                        break;
                    }
                }
            }
            // Fallback when palm detector unavailable or no palms found:
            // project SMPL wrists, accepting the known deskcrop bias.
            if regions.is_empty() {
                for (smpl_wrist_idx, avatar_left) in &[(20usize, false), (21usize, true)] {
                    let wrist_cam = &camera_joints[*smpl_wrist_idx];
                    let (u_px, v_px) = project_to_crop_px(
                        wrist_cam,
                        &pose.pred_cam,
                        IMAGE_SIZE as i32,
                    );
                    let scale_x = infer_w as f32 / IMAGE_SIZE as f32;
                    let scale_y = infer_h as f32 / IMAGE_SIZE as f32;
                    let wrist_crop_x = (u_px * scale_x) as i32;
                    let wrist_crop_y = (v_px * scale_y) as i32;
                    let hand_side = (infer_w.max(infer_h) * 30 / 100).max(96);
                    let hcx1 = (wrist_crop_x - hand_side / 2).clamp(0, infer_w - 1);
                    let hcy1 = (wrist_crop_y - hand_side / 2).clamp(0, infer_h - 1);
                    let hcx2 = (wrist_crop_x + hand_side / 2).clamp(1, infer_w);
                    let hcy2 = (wrist_crop_y + hand_side / 2).clamp(1, infer_h);
                    regions.push(HandRegion {
                        hcx1,
                        hcy1,
                        hw: (hcx2 - hcx1) as u32,
                        hh: (hcy2 - hcy1) as u32,
                        avatar_left_hint: Some(*avatar_left),
                    });
                }
            }

            for region in &regions {
                let hcx1 = region.hcx1;
                let hcy1 = region.hcy1;
                let hw = region.hw;
                let hh = region.hh;
                if hw < 16 || hh < 16 {
                    continue;
                }
                let hand_crop = crop_rgb(
                    infer_rgb,
                    crop_w,
                    crop_h,
                    hcx1 as u32,
                    hcy1 as u32,
                    hw,
                    hh,
                );
                let resized = resize_rgb_to_224(&hand_crop, hw, hh);
                let Some(landmarks) = hand.estimate(&resized) else {
                    continue;
                };
                if landmarks.presence < 0.30 {
                    continue;
                }
                // Convert hand-crop+YOLOX coords to original-image
                // pixel coords for ViTPose-style source mapping.
                // Required to put fingers in the SAME source frame the
                // existing solver hand pipeline uses; the MediaPipe
                // canonical-world frame is hand-relative and was
                // experimentally observed to produce upside-down
                // fingers (kp[12] is at wrist-Y < kp[9] = MCP-Y in
                // canon, but in image space tip is *further* from
                // wrist, so canon Y needs flipping — easier to just
                // use the screen output directly).
                let img_to_source = |orig_x: f32, orig_y: f32| -> [f32; 2] {
                    let aspect = width as f32 / height.max(1) as f32;
                    let nx = orig_x / width as f32;
                    let ny = orig_y / height as f32;
                    [-(nx - 0.5) * 2.0 * aspect, -(ny - 0.5) * 2.0]
                };
                // avatar_left from the wrist-distance assignment we
                // computed above when building the regions list. Far
                // more reliable than the landmarker's handedness
                // output, which is sensitive to selfie-mirror state.
                let avatar_left = region.avatar_left_hint.unwrap_or(false);
                let wrist_bone = if avatar_left {
                    HumanoidBone::LeftHand
                } else {
                    HumanoidBone::RightHand
                };
                let wrist_anchor = skeleton
                    .joints
                    .get(&wrist_bone)
                    .map(|j| j.position)
                    .unwrap_or([0.0, 0.0, 0.0]);

                inject_hand_landmarks_from_screen(
                    &mut skeleton,
                    &landmarks,
                    avatar_left,
                    crop_origin.0 as f32 + hcx1 as f32,
                    crop_origin.1 as f32 + hcy1 as f32,
                    hw as f32,
                    hh as f32,
                    &img_to_source,
                    wrist_anchor,
                );
                // 3-DoF wrist rotation derived from the SOURCE-FRAME
                // finger keypoints we just injected. Computing forward
                // and up vectors here keeps us in the same image frame
                // the rest of the pipeline uses (avoid the canonical-
                // world frame's hand-relative axes which were wrong).
                let screen_to_src = |kp: super::hand_mediapipe::HandKp| -> [f32; 3] {
                    let s = landmarks.screen[kp as usize];
                    let orig_x = crop_origin.0 as f32 + hcx1 as f32 + (s[0] / 224.0) * hw as f32;
                    let orig_y = crop_origin.1 as f32 + hcy1 as f32 + (s[1] / 224.0) * hh as f32;
                    let xy = img_to_source(orig_x, orig_y);
                    let z_per_pixel = 2.0 / 224.0 * (hh as f32 / 224.0).max(hw as f32 / 224.0);
                    [xy[0], xy[1], -s[2] * z_per_pixel]
                };
                let _ = screen_to_src; // kept for diagnostic use; orientation uses world below
                // Use MediaPipe world coords with sign flip on all
                // three axes to get true 3D forward/up in source frame.
                // The screen-based path was depth-blind (forward had
                // ~zero z), so palm-camera gestures came out
                // palm-down on the avatar. World coords give us the
                // metacarpus's actual 3D direction (slightly +z =
                // toward camera for forward-extended hands) and a
                // proper palm normal via cross product.
                let world_to_src = |kp: super::hand_mediapipe::HandKp| -> [f32; 3] {
                    let w = landmarks.world[kp as usize];
                    [-w[0], -w[1], -w[2]]
                };
                let wrist_src = world_to_src(super::hand_mediapipe::HandKp::Wrist);
                let mid_src = world_to_src(super::hand_mediapipe::HandKp::MiddleMcp);
                let idx_src = world_to_src(super::hand_mediapipe::HandKp::IndexMcp);
                let pky_src = world_to_src(super::hand_mediapipe::HandKp::PinkyMcp);
                let forward = vec3_sub(&mid_src, &wrist_src);
                let v_idx = vec3_sub(&idx_src, &wrist_src);
                let v_pky = vec3_sub(&pky_src, &wrist_src);
                // Palm normal: cross(v_idx, v_pky) for both hands.
                // Empirically verified: this points toward camera (+z
                // source = palm-face direction) for palm-camera poses
                // on both LEFT and RIGHT hands.
                let up = vec3_cross(&v_idx, &v_pky);
                let orient = super::source_skeleton::HandOrientation {
                    forward: vec3_normalize(&forward),
                    up: vec3_normalize(&up),
                    confidence: landmarks.presence,
                };
                if avatar_left {
                    skeleton.left_hand_orientation = Some(orient);
                } else {
                    skeleton.right_hand_orientation = Some(orient);
                }
            }
            t_hands.elapsed()
        } else {
            std::time::Duration::ZERO
        };

        let dt_total = t_total.elapsed();
        debug!(
            "HMR2 timing: total={:>5.1}ms yolox={:>5.1}ms infer={:>5.1}ms hands={:>5.1}ms",
            dt_total.as_secs_f32() * 1000.0,
            dt_yolox.as_secs_f32() * 1000.0,
            dt_infer.as_secs_f32() * 1000.0,
            dt_hands.as_secs_f32() * 1000.0,
        );
        // Face mesh pass: HMR2 has no facial landmarks, so derive a face bbox
        // from the projected SMPL head / neck joints and run the MediaPipe
        // FaceMesh cascade. Populates `skeleton.expressions` +
        // `face_mesh_confidence`, which the solver's `solve_expressions`
        // consumes when Face Tracking is enabled — without it the avatar's
        // face stays neutral on the HMR2 provider. Head rotation remains
        // HMR2/SMPL-driven; this only adds blendshape expressions. Placement
        // inherits pred_cam's deskcrop bias, mitigated by a generous bbox.
        if let Some(face_mesh) = self.face_mesh.as_mut() {
            let crop_sx = crop_w as f32 / IMAGE_SIZE as f32;
            let crop_sy = crop_h as f32 / IMAGE_SIZE as f32;
            let project_full = |idx: usize| -> (f32, f32) {
                let (u, v) =
                    project_to_crop_px(&camera_joints[idx], &pose.pred_cam, IMAGE_SIZE as i32);
                (
                    crop_origin.0 as f32 + u * crop_sx,
                    crop_origin.1 as f32 + v * crop_sy,
                )
            };
            let (hx, hy) = project_full(15); // SMPL 15 = Head
            let (nx, ny) = project_full(12); // SMPL 12 = Neck
            let head_len = ((hx - nx).powi(2) + (hy - ny).powi(2)).sqrt().max(24.0);
            // The SMPL Head joint sits at the atlas (≈ ear level); the mouth /
            // jaw is *below* it, toward the neck. Nudge the centre down the
            // head→neck direction so the lower face is captured (an earlier
            // up-nudge cropped the mouth out, leaving jawOpen ≈ 0 → no image
            // lip-sync), and size the box generously to keep forehead→chin in
            // frame.
            let size = head_len * 2.8;
            let cx = hx + (nx - hx) * 0.2;
            let cy = hy + (ny - hy) * 0.3;
            let bbox = FaceBbox {
                x: cx - size * 0.5,
                y: cy - size * 0.5,
                size,
            };
            if let Some((exprs, mesh_conf, _face_pose)) =
                face_mesh.estimate(rgb_data, width, height, &bbox)
            {
                skeleton.expressions = exprs;
                skeleton.face_mesh_confidence = Some(mesh_conf);
            }
        }

        // Project the SMPL body joints to normalised full-image coords so the
        // webcam-preview wipe can draw HMR2's detected pose (keypoints +
        // skeleton), matching the RTMW3D / ViTPose providers. HMR2 is a
        // parametric model with no native 2D keypoints, so reuse the same
        // weak-perspective `pred_cam` projection the hand-wrist disambiguation
        // uses. The crop is stretched to IMAGE_SIZE for inference, so invert
        // that (`* crop_w / IMAGE_SIZE`) then offset by the crop origin. Joint
        // placement inherits pred_cam's known deskcrop bias, but the wipe is a
        // debug overlay — an approximate skeleton is more useful than none.
        // Confidence is 1.0: the model always emits a complete pose.
        let inv_w = 1.0 / width as f32;
        let inv_h = 1.0 / height as f32;
        let crop_sx = crop_w as f32 / IMAGE_SIZE as f32;
        let crop_sy = crop_h as f32 / IMAGE_SIZE as f32;
        let ann_keypoints: Vec<(f32, f32, f32)> = camera_joints
            .iter()
            .map(|j| {
                let (u_px, v_px) = project_to_crop_px(j, &pose.pred_cam, IMAGE_SIZE as i32);
                let full_x = crop_origin.0 as f32 + u_px * crop_sx;
                let full_y = crop_origin.1 as f32 + v_px * crop_sy;
                (full_x * inv_w, full_y * inv_h, 1.0)
            })
            .collect();
        let ann_skeleton: Vec<(usize, usize)> = smpl::PARENTS
            .iter()
            .enumerate()
            .filter_map(|(i, &p)| (p >= 0).then_some((i, p as usize)))
            .collect();

        PoseEstimate {
            annotation: DetectionAnnotation {
                keypoints: ann_keypoints,
                skeleton: ann_skeleton,
                bounding_box: annotation_bbox,
            },
            skeleton,
        }
    }
}

/// Project an SMPL camera-frame joint to pixel coords inside HMR2's
/// 256×256 input. Mirrors the projection PyTorch HMR2 does internally
/// (weak perspective + 32-pixel pad correction).
#[cfg(feature = "inference")]
fn project_to_crop_px(
    wrist_cam: &[f32; 3],
    pred_cam: &[f32; 3],
    image_size: i32,
) -> (f32, f32) {
    let scale = pred_cam[0];
    let tx = pred_cam[1];
    let ty = pred_cam[2];
    // Weak perspective projects to a 192-wide / 256-tall crop space
    // because HMR2 internally trims 32 px off each horizontal edge.
    let u_norm = scale * wrist_cam[0] + tx; // ∈ roughly [-1, 1]
    let v_norm = scale * wrist_cam[1] + ty;
    let inner_w = image_size as f32 - 64.0; // 192
    let u_inner = (u_norm + 1.0) * 0.5 * inner_w;
    let v_inner = (v_norm + 1.0) * 0.5 * image_size as f32;
    let u_px = u_inner + 32.0; // re-add the 32-px left pad
    (u_px, v_inner)
}

/// Bilinear resize RGB → 224×224. Plain CPU loop — hand crops are
/// usually 200-400 px on a side, so a SIMD or image-crate version
/// isn't worth the dependency churn.
#[cfg(feature = "inference")]
fn resize_rgb_to_224(rgb: &[u8], src_w: u32, src_h: u32) -> Vec<u8> {
    let mut out = vec![0u8; 224 * 224 * 3];
    if src_w == 0 || src_h == 0 {
        return out;
    }
    let scale_x = src_w as f32 / 224.0;
    let scale_y = src_h as f32 / 224.0;
    let max_x = src_w as i32 - 1;
    let max_y = src_h as i32 - 1;
    let stride = src_w as usize * 3;
    for dy in 0..224 {
        let fy = ((dy as f32) + 0.5) * scale_y - 0.5;
        let y0 = fy.floor() as i32;
        let wy1 = (fy - y0 as f32).clamp(0.0, 1.0);
        let wy0 = 1.0 - wy1;
        let y0c = y0.clamp(0, max_y) as usize;
        let y1c = (y0 + 1).clamp(0, max_y) as usize;
        for dx in 0..224 {
            let fx = ((dx as f32) + 0.5) * scale_x - 0.5;
            let x0 = fx.floor() as i32;
            let wx1 = (fx - x0 as f32).clamp(0.0, 1.0);
            let wx0 = 1.0 - wx1;
            let x0c = x0.clamp(0, max_x) as usize;
            let x1c = (x0 + 1).clamp(0, max_x) as usize;
            for c in 0..3 {
                let r00 = rgb[y0c * stride + x0c * 3 + c] as f32;
                let r01 = rgb[y0c * stride + x1c * 3 + c] as f32;
                let r10 = rgb[y1c * stride + x0c * 3 + c] as f32;
                let r11 = rgb[y1c * stride + x1c * 3 + c] as f32;
                let v = r00 * wy0 * wx0 + r01 * wy0 * wx1 + r10 * wy1 * wx0 + r11 * wy1 * wx1;
                out[(dy * 224 + dx) * 3 + c] = v.clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

#[cfg(feature = "inference")]
fn insert_joint(skeleton: &mut SourceSkeleton, bone: HumanoidBone, pos: [f32; 3]) {
    skeleton.joints.insert(bone, joint(pos));
}

#[cfg(feature = "inference")]
fn joint(pos: [f32; 3]) -> SourceJoint {
    SourceJoint {
        position: pos,
        confidence: 1.0,
        metric_depth_m: None,
    }
}

/// Inject finger keypoints into the source skeleton using the
/// MediaPipe `screen` output (224-pixel hand crop). The screen path
/// puts fingers in the same image-pixel-derived source frame the
/// existing ViTPose hand pipeline uses, so the solver's
/// Tip::Joint chain math (rest 3D direction vs source 2D direction)
/// works the same way it does for ViTPose fingers — finger curl in
/// the image plane translates to finger curl on the avatar.
///
/// Tradeoff (vs a canonical-world-frame mapping):
/// finger Z is planar (no depth from MediaPipe screen output beyond
/// a relative-depth field we currently discard). Palm depth still
/// works via `HandOrientation`. For curl ("fingers open vs closed"),
/// image-plane 2D is enough.
#[cfg(feature = "inference")]
#[allow(clippy::too_many_arguments)]
fn inject_hand_landmarks_from_screen(
    skeleton: &mut SourceSkeleton,
    lm: &super::hand_mediapipe::HandLandmarks,
    avatar_left: bool,
    hand_origin_x: f32,
    hand_origin_y: f32,
    hand_w: f32,
    hand_h: f32,
    img_to_source: &impl Fn(f32, f32) -> [f32; 2],
    wrist_anchor: [f32; 3],
) {
    use super::hand_mediapipe::HandKp;
    // MediaPipe `world` is image-aligned 3D in meters relative to the
    // hand's geometric center. Axes (per MediaPipe docs and empirical
    // verification):
    //   +X: image right (matches image pixel +X)
    //   +Y: image down
    //   +Z: AWAY from camera (positive depth)
    //
    // Our source frame (avatar world):
    //   +X: image left  (img_to_source flips: source_x = -nx)
    //   +Y: image up    (img_to_source flips: source_y = -ny)
    //   +Z: toward camera (avatar facing +z, camera at +z looking back)
    //
    // So world→source is a simple sign flip on all three axes:
    //   source_x = -world_x, source_y = -world_y, source_z = -world_z
    //
    // This preserves the full 3D finger geometry MediaPipe predicts,
    // including depth (fingers extending toward camera get proper +z
    // source instead of just image-plane drift like with screen kpts).
    //
    // The unit scale is meters from MediaPipe — small relative to the
    // hand bbox in image-source units, so we apply a scale factor
    // matching the screen-derived hand size to keep the rendered
    // finger length proportional to the avatar's hand mesh.
    let wrist_world = lm.world[HandKp::Wrist as usize];
    let mid_world = lm.world[HandKp::MiddleMcp as usize];
    let world_hand_len = ((mid_world[0] - wrist_world[0]).powi(2)
        + (mid_world[1] - wrist_world[1]).powi(2)
        + (mid_world[2] - wrist_world[2]).powi(2))
        .sqrt()
        .max(1e-6);
    // Match the screen-projected hand length (wrist→MiddleMcp in
    // source units) so absolute scale is correct in avatar world.
    let mp_screen = |kp: HandKp| -> [f32; 2] {
        let s = lm.screen[kp as usize];
        let ox = hand_origin_x + (s[0] / 224.0) * hand_w;
        let oy = hand_origin_y + (s[1] / 224.0) * hand_h;
        img_to_source(ox, oy)
    };
    let screen_wrist = mp_screen(HandKp::Wrist);
    let screen_mid = mp_screen(HandKp::MiddleMcp);
    let screen_hand_len = ((screen_mid[0] - screen_wrist[0]).powi(2)
        + (screen_mid[1] - screen_wrist[1]).powi(2))
        .sqrt()
        .max(1e-6);
    let scale = screen_hand_len / world_hand_len;

    let kp_to_source = |kp: HandKp| -> [f32; 3] {
        let w = lm.world[kp as usize];
        let dx = (w[0] - wrist_world[0]) * scale;
        let dy = (w[1] - wrist_world[1]) * scale;
        let dz = (w[2] - wrist_world[2]) * scale;
        [
            wrist_anchor[0] - dx,
            wrist_anchor[1] - dy,
            wrist_anchor[2] - dz,
        ]
    };


    // VRM bone enum per side, same mapping as the canonical-world path.
    let (thumb_p, thumb_i, thumb_d, index_p, index_i, index_d,
         middle_p, middle_i, middle_d, ring_p, ring_i, ring_d,
         little_p, little_i, little_d): (HumanoidBone, _, _, _, _, _, _, _, _, _, _, _, _, _, _) =
        if avatar_left {
            (
                HumanoidBone::LeftThumbProximal, HumanoidBone::LeftThumbIntermediate, HumanoidBone::LeftThumbDistal,
                HumanoidBone::LeftIndexProximal, HumanoidBone::LeftIndexIntermediate, HumanoidBone::LeftIndexDistal,
                HumanoidBone::LeftMiddleProximal, HumanoidBone::LeftMiddleIntermediate, HumanoidBone::LeftMiddleDistal,
                HumanoidBone::LeftRingProximal, HumanoidBone::LeftRingIntermediate, HumanoidBone::LeftRingDistal,
                HumanoidBone::LeftLittleProximal, HumanoidBone::LeftLittleIntermediate, HumanoidBone::LeftLittleDistal,
            )
        } else {
            (
                HumanoidBone::RightThumbProximal, HumanoidBone::RightThumbIntermediate, HumanoidBone::RightThumbDistal,
                HumanoidBone::RightIndexProximal, HumanoidBone::RightIndexIntermediate, HumanoidBone::RightIndexDistal,
                HumanoidBone::RightMiddleProximal, HumanoidBone::RightMiddleIntermediate, HumanoidBone::RightMiddleDistal,
                HumanoidBone::RightRingProximal, HumanoidBone::RightRingIntermediate, HumanoidBone::RightRingDistal,
                HumanoidBone::RightLittleProximal, HumanoidBone::RightLittleIntermediate, HumanoidBone::RightLittleDistal,
            )
        };

    // Thumb mapping. VRM's ThumbProximal bone HEAD sits at the CMC
    // (where the thumb meets the palm), not the MCP — so anatomically
    // the bone segment ThumbProximal→ThumbIntermediate spans CMC→MCP,
    // not MCP→IP. Earlier we dropped MediaPipe's CMC entirely and
    // shifted everything one joint, which placed ThumbProximal at the
    // MCP and pointed the bone toward IP — leaving the thumb invisible
    // because the avatar mesh weighted near the actual CMC region had
    // no bone to drive it.
    //
    // Correct map (shift one keypoint earlier):
    //   CMC (kp 1) → ThumbProximal      (bone HEAD)
    //   MCP (kp 2) → ThumbIntermediate
    //   IP  (kp 3) → ThumbDistal
    //   Tip (kp 4) → fingertips[ThumbDistal]
    insert_joint(skeleton, thumb_p, kp_to_source(HandKp::ThumbCmc));
    insert_joint(skeleton, thumb_i, kp_to_source(HandKp::ThumbMcp));
    let thumb_dip = kp_to_source(HandKp::ThumbIp);
    insert_joint(skeleton, thumb_d, thumb_dip);
    skeleton.fingertips.insert(thumb_d, joint(kp_to_source(HandKp::ThumbTip)));

    // Fingers (Index/Middle/Ring/Pinky): MCP→Proximal, PIP→Intermediate,
    // DIP→Distal, Tip→fingertips[Distal].
    let fingers: [(_, _, _, _, _, _, _); 4] = [
        (HandKp::IndexMcp, HandKp::IndexPip, HandKp::IndexDip, HandKp::IndexTip, index_p, index_i, index_d),
        (HandKp::MiddleMcp, HandKp::MiddlePip, HandKp::MiddleDip, HandKp::MiddleTip, middle_p, middle_i, middle_d),
        (HandKp::RingMcp, HandKp::RingPip, HandKp::RingDip, HandKp::RingTip, ring_p, ring_i, ring_d),
        (HandKp::PinkyMcp, HandKp::PinkyPip, HandKp::PinkyDip, HandKp::PinkyTip, little_p, little_i, little_d),
    ];
    for (mcp, pip, dip, tip, b_p, b_i, b_d) in fingers {
        insert_joint(skeleton, b_p, kp_to_source(mcp));
        insert_joint(skeleton, b_i, kp_to_source(pip));
        let dip_pos = kp_to_source(dip);
        insert_joint(skeleton, b_d, dip_pos);
        skeleton.fingertips.insert(b_d, joint(kp_to_source(tip)));
    }
}

#[cfg(feature = "inference")]
fn vec3_sub(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[cfg(feature = "inference")]
fn vec3_cross(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(feature = "inference")]
fn vec3_normalize(v: &[f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-6 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

#[cfg(feature = "inference")]
fn empty_estimate(frame_index: u64) -> PoseEstimate {
    PoseEstimate {
        annotation: DetectionAnnotation::default(),
        skeleton: SourceSkeleton::empty(frame_index),
    }
}
