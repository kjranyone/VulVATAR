//! Head orientation derivation and tracking from dense FaceMesh pose and depth.
//!
//! Evaluates hand-face occlusion, verifies large claimed yaw turns against
//! the depth cheek slope profile, and enforces temporal continuity / model
//! consistency.

use crate::tracking::metric_frame::MetricDepthFrame;
use crate::tracking::PoseEstimate;

use super::estimator::{Intrinsics, OriObs};
use super::math::*;
use super::model::FACING_CAMERA;
use super::observe::window_point;

#[derive(Clone, Copy, Debug)]
pub struct FaceOcclusionStatus {
    pub face_occluded: bool,
    pub face_overlaps_hand: bool,
    pub nose_in_hand: bool,
}

pub struct HeadOriTracker {
    /// Frames remaining in which head-orientation observations stay
    /// suppressed after a hand-over-face episode.
    pub cooldown: u8,
    /// Last ACCEPTED head-orientation target (camera frame).
    pub last: Option<M3>,
    /// Consecutive rejections since the last accepted target.
    pub reject_run: u8,
    /// Frames since depth cheek-profile last confirmed large-yaw pose.
    pub depth_grace: u8,
}

impl HeadOriTracker {
    pub fn new() -> Self {
        Self {
            cooldown: 0,
            last: None,
            reject_run: 0,
            depth_grace: 0,
        }
    }

    pub fn reset(&mut self) {
        self.cooldown = 0;
        self.last = None;
        self.reject_run = 0;
        self.depth_grace = 0;
    }

    /// Check if hand crops overlap the face box or cover the nose.
    pub fn check_occlusion(
        &mut self,
        kps: &[(f32, f32, f32)],
        width: u32,
        height: u32,
        hand_rects: &[(f32, f32, f32, f32)],
        last_hands: &[Option<super::hands::HandResult>; 2],
    ) -> FaceOcclusionStatus {
        let in_hand_rect = |u: f64, v: f64| -> bool {
            hand_rects.iter().any(|&(x0, y0, x1, y1)| {
                u >= x0 as f64 && u <= x1 as f64 && v >= y0 as f64 && v <= y1 as f64
            })
        };
        let nose_in_hand = {
            let n = kps.first().copied().unwrap_or((0.5, 0.5, 0.0));
            in_hand_rect(n.0 as f64 * width as f64, n.1 as f64 * height as f64)
        };

        // Face-box × hand-box overlap: a palm anywhere over the face makes
        // the FaceMesh pose (and its confidence) untrustworthy well before
        // the nose itself is covered — the mesh happily locks onto a
        // saturated yaw while half the face is occluded. Face box from the
        // annotation's eye/ear keypoints, padded to cover the jaw.
        let face_overlaps_hand = {
            let mut x0 = f32::MAX;
            let mut y0 = f32::MAX;
            let mut x1 = f32::MIN;
            let mut y1 = f32::MIN;
            let mut n = 0;
            for k in kps.iter().take(5) {
                if k.2 < 0.3 {
                    continue;
                }
                let (u, v) = (k.0 * width as f32, k.1 * height as f32);
                x0 = x0.min(u);
                y0 = y0.min(v);
                x1 = x1.max(u);
                y1 = y1.max(v);
                n += 1;
            }
            if n >= 3 {
                let pad = 0.6 * (x1 - x0).max(24.0);
                let (fx0, fy0, fx1, fy1) = (x0 - pad, y0 - pad, x1 + pad, y1 + pad * 1.6);
                last_hands
                    .iter()
                    .flatten()
                    .filter(|hr| hr.presence >= 0.5)
                    .map(|hr| {
                        let (x, y, sz) = hr.crop;
                        (x, y, x + sz, y + sz)
                    })
                    .any(|(hx0, hy0, hx1, hy1)| fx0 < hx1 && hx0 < fx1 && fy0 < hy1 && hy0 < fy1)
            } else {
                false
            }
        };

        if nose_in_hand {
            self.cooldown = 5;
        } else {
            self.cooldown = self.cooldown.saturating_sub(1);
        }
        let face_occluded = self.cooldown > 0;

        FaceOcclusionStatus {
            face_occluded,
            face_overlaps_hand,
            nose_in_hand,
        }
    }

    /// Estimate head orientation observation from FaceMesh pose if accepted.
    pub fn estimate_orientation(
        &mut self,
        base: &PoseEstimate,
        occl: &FaceOcclusionStatus,
        depth: Option<&MetricDepthFrame>,
        head_center_pred: V3,
        head_depth_pred: f64,
        fk_pred_head_r: &M3,
        head_j: usize,
        t: f64,
        last_t: Option<f64>,
        width: u32,
        height: u32,
        intr: &Intrinsics,
        arm_capsules: &[(V3, V3, f64)],
        hand_rects: &[(f32, f32, f32, f32)],
    ) -> Option<OriObs> {
        let (Some(f), Some(c)) = (base.skeleton.face, base.skeleton.face_mesh_confidence) else {
            return None;
        };

        if !matches!(f.source, crate::tracking::FaceSource::Mesh)
            || std::env::var_os("VULVATAR_FUSION_NO_ORI").is_some()
            || c < 0.2
            // Hard envelope 1.35 rad (~77°), NOT 1.05: the user's habitual
            // desk pose (oblique camera) sits at |yaw| 0.96-1.13 rad, and at
            // the old 1.05 cut-off the whole orientation channel went dark
            // exactly there — measured live (2026-09-13): with the head
            // held at a constant −13° roll the avatar head roll tracked the
            // face channel while |yaw| < ~1.0 rad and collapsed to 0-3°
            // beyond it ("neck tilt dead while face direction still works"
            // — yaw survives via keypoints). The channel itself is
            // faithful end-to-end when it fires (replay: face→rig→avatar
            // all ≈ 1:1 on the three axes), so the envelope, not the
            // estimator, was the killer. 1.35 keeps a margin short of the
            // ±90° region where yaw/roll alias; landmark quality beyond it
            // is still guarded by step_ok / agrees_pred / depth_support.
            || f.yaw.abs() >= 1.35
            || self.cooldown != 0
        {
            if std::env::var_os("VULVATAR_ORI_DUMP").is_some() {
                let ann = &base.annotation.keypoints;
                let fc: f32 = ann.iter().take(5).map(|k| k.2).sum::<f32>() / 5.0;
                eprintln!(
                    "ORI gated t{:.3} src {:?} c {:.2} yaw {:.2} nose_in_hand {} overlap {} ann_face_c {:.2}",
                    t, f.source, c, f.yaw, occl.nose_in_hand, occl.face_overlaps_hand, fc
                );
            }
            return None;
        }

        // Source-frame angles → camera-frame head rotation:
        // R_head_cam = Rx(180) · R_view, with the view-frame head
        // delta composed yaw (about +Y, sign-flipped by the selfie
        // mirror), then pitch (about +X), then roll (about +Z).
        let (yw, pt, rl) = (-f.yaw as f64, f.pitch as f64, -f.roll as f64);
        let r_view = mat_mul(
            &so3_exp([0.0, yw, 0.0]),
            &mat_mul(&so3_exp([pt, 0.0, 0.0]), &so3_exp([0.0, 0.0, rl])),
        );
        let target = mat_mul(&FACING_CAMERA, &r_view);
        let dt_s = last_t.map(|lt| (t - lt).clamp(0.02, 0.25)).unwrap_or(0.033);
        let step_ok = match &self.last {
            Some(prev) => norm(so3_log(&mat_mul(&target, &transpose(prev)))) <= 3.0 * dt_s + 0.05,
            None => false,
        };
        let agrees_pred = norm(so3_log(&mat_mul(&target, &transpose(fk_pred_head_r)))) <= 0.4;

        let in_hand_rect = |u: f64, v: f64| -> bool {
            hand_rects.iter().any(|&(x0, y0, x1, y1)| {
                u >= x0 as f64 && u <= x1 as f64 && v >= y0 as f64 && v <= y1 as f64
            })
        };

        let depth_at = |u: f64, v: f64| -> Option<V3> {
            let d = depth?;
            if in_hand_rect(u, v) {
                return None;
            }
            let p = window_point(
                &d.points_m,
                d.width,
                d.height,
                u,
                v,
                1,
                (head_center_pred[2] - 0.20) as f32,
                (head_center_pred[2] + 0.20) as f32,
            )?;
            let t_obs = norm(p);
            if t_obs > 1e-6 {
                let dir = scale(p, 1.0 / t_obs);
                for &(a, b, r) in arm_capsules {
                    if let Some(t_hit) = super::estimator::ray_capsule_entry(dir, a, b, r) {
                        if t_hit < head_depth_pred - 0.05 {
                            return None;
                        }
                    }
                }
            }
            Some(p)
        };

        let depth_supports = if f.yaw.abs() <= 0.35 {
            Some(true)
        } else {
            face_depth_slope(
                &base.annotation.keypoints,
                width,
                height,
                &depth_at,
                f.yaw,
                occl.face_overlaps_hand,
                intr.fx,
                head_center_pred[2],
            )
        };

        if std::env::var_os("VULVATAR_ORI_DUMP").is_some() && f.yaw.abs() > 0.35 {
            eprintln!("ORI depth_support {:?} yaw {:.2}", depth_supports, f.yaw);
        }

        if depth_supports == Some(true) {
            self.depth_grace = 20;
        } else {
            self.depth_grace = self.depth_grace.saturating_sub(1);
        }
        let depth_ok = depth_supports == Some(true) || (self.depth_grace > 0 && step_ok);
        let accept = (step_ok || agrees_pred) && (f.yaw.abs() <= 0.35 || depth_ok);

        if !accept {
            self.reject_run = self.reject_run.saturating_add(1);
            if self.reject_run >= 15 {
                self.last = None;
            }
            return None;
        }

        self.last = Some(target);
        self.reject_run = 0;
        let sigma = std::env::var("VULVATAR_ORI_SIGMA")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or_else(|| 0.03 + 0.10 * (1.0 - c as f64))
            * if occl.face_overlaps_hand { 2.0 } else { 1.0 };

        if std::env::var_os("VULVATAR_ORI_DUMP").is_some() {
            let ann = &base.annotation.keypoints;
            let fc: f32 = ann.iter().take(5).map(|k| k.2).sum::<f32>() / 5.0;
            eprintln!(
                "ORI fired t{:.3} yaw {:.2} c {:.2} depth {:?} ann_face_c {:.2}",
                t, f.yaw, c, depth_supports, fc
            );
        }

        Some(OriObs {
            joint: head_j,
            target,
            sigma,
            chain_depth: usize::MAX,
        })
    }

    /// Roll-only head observation from the FaceMesh channel's eye-line
    /// tilt.
    ///
    /// Opt-in (`VULVATAR_FUSION_ROLLORI=1`, default OFF): the 2026-09-16
    /// bench measured no roll benefit on the 12-recording set — the mesh
    /// channel reaches this code on ~10/782 desk frames (conf ≥ 0.2 +
    /// hand cooldown), and where it fires the correction loses to the
    /// eye kp + priors (roll trace identical at σ down to 0.005) while
    /// perturbing the solve (+3–8 wrist snaps on knife-edge sessions).
    /// Kept for the frontal, well-lit, hands-away case where the channel
    /// is actually live.
    ///
    /// The full `estimate_orientation` is heavily gated (depth cheek
    /// slope, step/agreement, |yaw| envelope) and fires on a small
    /// minority of desk frames — but its ROLL component is just the
    /// image tilt of the eye line, the most robust quantity the channel
    /// produces (2026-09-16 pixel-level audit: within ±3° of the true
    /// eye-line tilt while the estimator rests −3..−6° low). The body
    /// detector's eye keypoints cannot supply absolute roll at the desk
    /// pose: the model's projected eye separator carries a pitch×yaw
    /// perspective cross-term (+7° at yaw 29° / pitch 14°, frame-dump
    /// measured) that the roll DOF absorbs as a rest bias. This obs
    /// injects ONLY the roll difference between the channel and the
    /// current prediction — a rotation about the bone's forward axis
    /// leaves yaw/pitch residuals exactly zero.
    pub fn estimate_roll_ori(
        &self,
        base: &PoseEstimate,
        occl: &FaceOcclusionStatus,
        fk_pred_head_r: &M3,
        head_j: usize,
    ) -> Option<OriObs> {
        if std::env::var_os("VULVATAR_FUSION_ROLLORI").is_none() {
            return None;
        }
        let (Some(f), Some(c)) = (base.skeleton.face, base.skeleton.face_mesh_confidence) else {
            return None;
        };
        roll_ori_obs(f, c, occl, self.cooldown == 0, fk_pred_head_r, head_j)
    }
}

/// Gate-free core of [`HeadOriTracker::estimate_roll_ori`] (the env
/// enable/cooldown gates live in the caller so tests can exercise the
/// geometry directly).
fn roll_ori_obs(
    f: crate::tracking::FacePose,
    c: f32,
    occl: &FaceOcclusionStatus,
    cooldown_ok: bool,
    fk_pred_head_r: &M3,
    head_j: usize,
) -> Option<OriObs> {
    if !matches!(f.source, crate::tracking::FaceSource::Mesh)
        || c < 0.2
        || f.yaw.abs() >= 1.35
        || !cooldown_ok
    {
        return None;
    }
    // Geometric roll of the predicted head bone in the viewer frame —
    // the same decomposition `diagnose_fusion_replay::ypr_deg` uses.
    let rc = mat_mul(&super::model::FACING_CAMERA, fk_pred_head_r);
    let fwd = col(&rc, 2);
    let up = col(&rc, 1);
    let right = cross(up, fwd);
    let roll_pred = right[1].atan2(up[1].abs().max(1e-6));
    // Channel roll, mirrored into the viewer frame (see the
    // composition in `estimate_orientation`).
    let roll_ch = -(f.roll as f64);
    let delta = roll_ch - roll_pred;
    // Trim-sized corrections only: this obs exists to remove the
    // measured −4..−6° rest bias (0.07–0.10 rad), not to rescue large
    // disagreements — a >0.15 rad gap means occlusion artifacts or a
    // broken prediction, and step-applying it yanks the head through
    // the solver (measured: 10 firings, δ up to 0.28 rad, +6 wrist
    // snaps on s1789246660; 22 without).
    if !delta.is_finite() || delta.abs() > 0.15 {
        return None;
    }
    if std::env::var_os("VULVATAR_ROLLORI_DUMP").is_some() {
        eprintln!(
            "ROLLORI cand roll_pred {roll_pred:+.3} ch {roll_ch:+.3} Δ{delta:+.3} c {c:.2} cd_ok {cooldown_ok} yaw {:.2}",
            f.yaw
        );
    }
    let sigma = std::env::var("VULVATAR_ROLLORI_SIGMA")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.03 + 0.08 * (1.0 - c as f64))
        * if occl.face_overlaps_hand { 2.0 } else { 1.0 };
    if std::env::var_os("VULVATAR_ROLLORI_DUMP").is_some() {
        eprintln!(
            "ROLLORI fired roll_pred {roll_pred:+.2} ch {roll_ch:+.2} Δ{delta:+.2} σ{sigma:.3} c {c:.2}"
        );
    }
    Some(OriObs {
        joint: head_j,
        // A rotation about the bone's LOCAL forward axis (+Z = face
        // forward) by `delta`: the world residual is then exactly a
        // forward-axis rotation, i.e. pure roll in the readout above.
        target: mat_mul(fk_pred_head_r, &so3_exp([0.0, 0.0, delta])),
        sigma,
        chain_depth: usize::MAX,
    })
}

fn face_depth_slope(
    kps: &[(f32, f32, f32)],
    width: u32,
    height: u32,
    depth_at: &dyn Fn(f64, f64) -> Option<V3>,
    claimed_yaw: f32,
    face_overlaps_hand: bool,
    intr_fx: f64,
    head_z_pred: f64,
) -> Option<bool> {
    let mut x0 = f32::MAX;
    let mut y0 = f32::MAX;
    let mut x1 = f32::MIN;
    let mut y1 = f32::MIN;
    let mut nk = 0;
    for k in kps.iter().take(5) {
        if k.2 < 0.3 {
            continue;
        }
        x0 = x0.min(k.0 * width as f32);
        y0 = y0.min(k.1 * height as f32);
        x1 = x1.max(k.0 * width as f32);
        y1 = y1.max(k.1 * height as f32);
        nk += 1;
    }
    if nk < 3 || x1 - x0 <= 30.0 {
        return None;
    }
    let cx = 0.5 * (x0 + x1);
    let cy = 0.5 * (y0 + y1);
    let half_h = 0.6 * (x1 - x0);
    let mut cols: Vec<(f64, f64)> = Vec::new();
    for iu in 0..14 {
        let u = x0 as f64 + (x1 - x0) as f64 * (iu as f64 + 0.5) / 14.0;
        let mut zs: Vec<f64> = Vec::new();
        for iv in 0..7 {
            let v = (cy - half_h) as f64 + 2.0 * half_h as f64 * (iv as f64 + 0.5) / 7.0;
            if let Some(p) = depth_at(u, v) {
                zs.push(p[2]);
            }
        }
        if zs.len() >= 3 {
            zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            cols.push((u, zs[zs.len() / 2]));
        }
    }
    if cols.len() < 8 {
        return None;
    }
    let n = cols.len() as f64;
    let mu = cols.iter().map(|c| c.0).sum::<f64>() / n;
    let mz = cols.iter().map(|c| c.1).sum::<f64>() / n;
    let (mut num, mut den) = (0.0, 0.0);
    for (u, z) in &cols {
        num += (u - mu) * (z - mz);
        den += (u - mu) * (u - mu);
    }
    let dz_full = num / den.max(1e-9) * (x1 - x0) as f64;
    if std::env::var_os("VULVATAR_ORI_CAL").is_some() {
        let w_m = (x1 - x0) as f64 * head_z_pred / intr_fx.max(1.0) as f64;
        eprintln!(
            "ORICAL yaw {:.3} dz {:+.4} w_m {:.3} expect {:+.4} ncol {}",
            claimed_yaw,
            dz_full,
            w_m,
            -w_m * (claimed_yaw as f64).tan(),
            cols.len()
        );
    }
    if std::env::var_os("VULVATAR_ORI_DUMP").is_some() {
        let nose = kps[0];
        let no = if nose.2 >= 0.3 && x1 - x0 > 1.0 {
            (nose.0 * width as f32 - cx) / (0.5 * (x1 - x0))
        } else {
            f32::NAN
        };
        eprintln!(
            "ORI dz {:+.3} ncol {} yaw {:.2} nose_off {:+.2}",
            dz_full,
            cols.len(),
            claimed_yaw,
            no
        );
    }
    let nose = kps[0];
    let nose_turned = if nose.2 >= 0.3 && x1 - x0 > 1.0 {
        let no = (nose.0 * width as f32 - cx) / (0.5 * (x1 - x0));
        // `claimed_yaw` is the MIRRORED source-frame yaw (the camera-frame
        // yaw is −claimed_yaw, see the composition in
        // `estimate_orientation`), while `no` is in unmirrored image
        // space: a genuine turn puts the nose on the side OPPOSITE the
        // source-frame yaw sign. Comparing them for EQUALITY (the old
        // check) passed only inconsistent geometries, so at large real
        // yaws this escape hatch never fired and the cheek-slope
        // magnitude path (damped to 0.011–0.025 m by hair/holes vs the
        // 0.025 m threshold) rejected the honest FaceMesh orientation —
        // measured on the s1789219959 desk replay: nose_off +0.75..+0.86
        // with yaw −0.8..−0.9 rejected for 214 frames while the estimator
        // parked the head 140° off the mesh channel.
        no.abs() > 0.40 && (no < 0.0) != (claimed_yaw < 0.0)
    } else {
        false
    };
    let need = if face_overlaps_hand { 0.040 } else { 0.025 };
    Some(nose_turned || (dz_full.abs() >= need && (dz_full < 0.0) == (claimed_yaw < 0.0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::source_skeleton::{FacePose, SourceSkeleton};
    use crate::tracking::FaceSource;
    use crate::tracking::PoseEstimate;

    fn base_with(mesh_roll_rad: f32, conf: f32) -> PoseEstimate {
        let mut sk = SourceSkeleton::empty(0);
        sk.face = Some(FacePose {
            yaw: 0.0,
            pitch: 0.0,
            roll: mesh_roll_rad,
            confidence: conf,
            source: FaceSource::Mesh,
            ..Default::default()
        });
        sk.face_mesh_confidence = Some(conf);
        PoseEstimate {
            skeleton: sk,
            annotation: Default::default(),
        }
    }

    /// The roll-only target must differ from the prediction by exactly a
    /// forward-axis rotation of the channel-vs-prediction roll gap — yaw
    /// and pitch residuals stay zero, and a zero gap returns the
    /// prediction unchanged. Exercises the gate-free core (the env
    /// opt-in lives in the wrapper).
    #[test]
    fn roll_ori_is_pure_forward_axis_correction() {
        let occl = FaceOcclusionStatus {
            face_occluded: false,
            face_overlaps_hand: false,
            nose_in_hand: false,
        };
        // Viewer-frame readout (same decomposition as the replay binary).
        let ypr = |r: &M3| {
            let rc = mat_mul(&super::super::model::FACING_CAMERA, r);
            let fwd = col(&rc, 2);
            let up = col(&rc, 1);
            let right = cross(up, fwd);
            (
                fwd[0].atan2(fwd[2]),
                (-fwd[1]).asin(),
                right[1].atan2(up[1].abs().max(1e-6)),
            )
        };
        // A desk-like prediction: yaw 0.5 rad, pitch 0.24 rad, roll 0.
        let pred = mat_mul(
            &super::super::model::FACING_CAMERA,
            &mat_mul(
                &so3_exp([0.0, 0.5, 0.0]),
                &mat_mul(&so3_exp([0.24, 0.0, 0.0]), &so3_exp([0.0, 0.0, 0.0])),
            ),
        );
        let (y0, p0, r0) = ypr(&pred);
        // Channel reads the head +6° more rolled than the prediction.
        let base = base_with(-6f64.to_radians() as f32, 0.5);
        let ori = roll_ori_obs(
            base.skeleton.face.unwrap(),
            base.skeleton.face_mesh_confidence.unwrap(),
            &occl,
            true,
            &pred,
            7,
        )
        .expect("roll ori should fire");
        let (y1, p1, r1) = ypr(&ori.target);
        assert!(
            (r1 - r0 - 6f64.to_radians()).abs() < 1e-6,
            "roll {r0:.6}→{r1:.6} Δ={} (want +6°)",
            r1 - r0
        );
        assert!(
            (y1 - y0).abs() < 1e-6 && (p1 - p0).abs() < 1e-6,
            "yaw/pitch moved {y0:.4}→{y1:.4} {p0:.4}→{p1:.4}"
        );
        // Zero gap → target == prediction.
        let base0 = base_with(-r0 as f32, 0.5);
        let ori0 = roll_ori_obs(
            base0.skeleton.face.unwrap(),
            base0.skeleton.face_mesh_confidence.unwrap(),
            &occl,
            true,
            &pred,
            7,
        )
        .unwrap();
        let e0 = so3_log(&mat_mul(&pred, &transpose(&ori0.target)));
        assert!(norm(e0) < 1e-9, "zero-gap residual {e0:?}");
    }
}
