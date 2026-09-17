//! State-driven hand crops → hand landmark model (21 keypoints) →
//! estimator observations.
//!
//! The body detector's hand block is coarse (≈ 20 px σ on a 1280-wide
//! frame); a dedicated crop around the *predicted* hand runs the hand
//! model at native resolution, giving finger keypoints good to a couple
//! of pixels. Crops are cut from the prediction, so no palm detector is
//! needed; a hand whose prediction leaves the frame is simply not
//! cropped (the estimator then relaxes it).
//!
//! The backend is `rtmpose` — `models/rtmpose-m-hand_256.onnx`
//! (RTMPose-m hand, SimCC heads), input `(1,3,256,256)` NCHW
//! ImageNet-normalised RGB, outputs `simcc_x`/`simcc_y` `(1,21,512)`;
//! presence is a distilled calibrated classifier (see [`PresenceNet`])
//! and handedness is chirality-geometric (see [`RtmposeHand`]).
//! Acquisition is two-stage: a full-frame palm-proposal heatmap (see
//! [`PalmNet`]) feeds candidate crops into the crop chain. The old
//! MediaPipe hand-landmark backend was removed on 2026-09-18 after the
//! six-recording A/B gate passed (clasp 780/781 duty 1.00/1.00 snaps
//! 0/0 vs MediaPipe 782/782/0; chin duty 0.90 vs 0.88; palms 0.89 vs
//! 0.26; namaste 0.97/0.75 vs 0.66/0.72; nohands FPs ≈ MediaPipe's
//! own) — git history has the reference implementation.

use std::path::Path;

use log::{error, info, warn};
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;

use crate::tracking::detector::session::{build_session, build_session_cpu_only};

use super::estimator::{AngleObs, Intrinsics, Kp2d, Kp3d, ModelPoint};
use super::math::*;
use super::model::*;

pub const HAND_KP: usize = 21;

#[derive(Clone, Debug, Default)]
pub struct HandResult {
    /// Frame-pixel x, y and crop-scale relative z per landmark.
    pub px: [[f32; 3]; HAND_KP],
    /// Wrist-relative metres (camera-aligned).
    pub world: [[f32; 3]; HAND_KP],
    pub presence: f32,
    pub handedness: f32,
    /// Crop used `(x, y, size)` in frame pixels.
    pub crop: (f32, f32, f32),
    /// Which crop candidate produced this result: 0 = previous frame's
    /// locked hand re-cropped, 1 = the body detector's hand block,
    /// 2 = the model prediction. Provenance for wrist-obs noise
    /// attribution (a prediction crop over desk pixels can hallucinate
    /// a coherent hand); the provider sets it.
    pub src: u8,
    /// Distilled presence-classifier score on the wrist-centred window,
    /// when the net ran for this candidate (the veto path or the
    /// provider's palm re-score). Unlike the SimCC sharpness in
    /// `presence`, this is calibrated and crop-size independent.
    pub net_presence: Option<f32>,
}

/// Crop-local RTMPose-m hand model (21 keypoints, InterHand2.6M-style
/// canonical order) — the sole hand backend via [`HandBackend`] since
/// the six-recording A/B gate passed and the MediaPipe hand-landmarker
/// was removed (2026-09-18).
///
/// Mapping from the model's outputs to the [`HandResult`] contract:
/// * `simcc_x` / `simcc_y` (`(1, 21, 512)` each — SimCC classification
///   heatmaps at 2× input resolution) → landmark pixels via per-keypoint
///   argmax + 3-point sub-bin refinement. There is NO detection score:
///   presence is the median SimCC peak sharpness (softmax mass vs
///   absolute-mass ratio) across the keypoints — a confidence proxy for
///   the provider's 0.5/0.6 lock gates, NOT a hand/no-hand probability.
///   The provider's wrist-move tracking stays the face-crop guard.
/// * `handedness` is NOT a model output — RTMPose hand heads have no
///   classification side head. It is reconstructed geometrically from
///   the landmark chirality (see `chirality_handedness`), which cannot
///   separate palm-up from palm-down views; the provider's FK-based L/R
///   assignment and duplicate-lock vote are the authority, the veto
///   bands only demote.
/// * `world` is zeros. It is diagnostics-only today (`wrist_abs` is
///   `None` in the provider — measured ±30–60 % proportion noise), so
///   nothing consumes it.
pub struct RtmposeHand {
    session: Session,
    input_name: String,
    /// Square model input side (from the `_NNN` filename convention).
    size: u32,
    tensor: Array4<f32>,
    /// Optional distilled presence classifier; absent → presence falls
    /// back to the SimCC sharpness proxy (unreliable — see PresenceNet).
    presence_net: Option<PresenceNet>,
    /// Full-frame wrist proposals (optional export, see [`PalmNet`]).
    palm_net: Option<PalmNet>,
    /// Inverts the chirality→handedness mapping (`VULVATAR_HAND_CHIRALITY_FLIP=1`).
    chirality_flip: bool,
}

/// Export filename candidates, fastest first. The `_NNN` suffix is the
/// square input side (same contract as the body export). This export
/// family's SimCC bins run at 2× the input side.
const RTMOSE_HAND_CANDIDATES: [&str; 1] = ["rtmpose-m-hand_256.onnx"];

/// ImageNet normalisation mmpose applies on top of `[0, 1]` RGB
/// (divided by 255 to match `fill_crop_tensor`'s output range).
const RTMOSE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const RTMOSE_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Tiny hand-vs-not classifier distilled OFFLINE from MediaPipe's
/// bimodal presence labels on this camera's own recordings — the runtime
/// stays MP-free. This is the presence authority for the RTMPose backend:
/// SimCC sharpness cannot separate a genuine hand from a confident
/// cuff/face answer (measured: sharpness 0.80 on hallucinations vs 0.52
/// on true hands, i.e. anti-correlated with correctness). Input contract
/// `(1,3,64,64)` float RGB in [0,1] — `fill_crop_tensor`'s output as-is.
struct PresenceNet {
    session: Session,
    input_name: String,
    tensor: Array4<f32>,
}

const PRESENCE_NET_FILE: &str = "rtmpose-hand-presence_64.onnx";

/// Full-frame palm proposal net: a 16×16 wrist heatmap distilled from the
/// same MP labels as the presence classifier, run on the letterboxed full
/// frame (the `yolo_pose::letterbox` contract — grey 114, centred pads).
/// This is what turns acquisition from heuristic crops (wrist / face-below
/// / predicted) into actual detections: the case it exists for is the hand
/// the body detector never sees (chin pose, hands entering the frame
/// edge). Optional — without the export the chain keeps heuristics only.
struct PalmNet {
    session: Session,
    input_name: String,
    tensor: Array4<f32>,
}

const PALM_NET_FILE: &str = "rtmpose-hand-palm_256.onnx";
const PALM_GRID: usize = 16;
const PALM_CELL: f32 = 256.0 / PALM_GRID as f32;

impl PalmNet {
    fn try_from_models_dir(dir: &Path) -> Option<Self> {
        let path = dir.join(PALM_NET_FILE);
        if !path.is_file() {
            return None;
        }
        let (session, _backend) = build_session(&path.to_string_lossy(), 1, "RTMPose-hand-palm").ok()?;
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input".to_string());
        Some(Self {
            session,
            input_name,
            tensor: Array4::zeros((1, 3, 256, 256)),
        })
    }

    /// Wrist proposals in frame pixels: `[score, x, y]`, best first, top 3,
    /// 3×3 local-max + parabolic sub-cell refine — the same decode the
    /// trainer validates (`scratchpad/train_hand_palm.py::peaks`).
    fn detect(&mut self, rgb: &[u8], width: u32, height: u32) -> Vec<[f32; 3]> {
        use crate::tracking::detector::yolo_pose::letterbox;
        let thresh = std::env::var("VULVATAR_HAND_PALM_THRESH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5);
        self.tensor = letterbox(rgb, width, height, 256);
        let input = match TensorRef::from_array_view(&self.tensor) {
            Ok(v) => v,
            Err(e) => {
                error!("hand-palm: tensor ref failed: {e}");
                return Vec::new();
            }
        };
        let Some(outputs) = self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
            .ok()
        else {
            return Vec::new();
        };
        let Some(out) = outputs.get("heat") else {
            return Vec::new();
        };
        let Ok((_, data)) = out.try_extract_tensor::<f32>() else {
            return Vec::new();
        };
        if data.len() < PALM_GRID * PALM_GRID {
            return Vec::new();
        }
        let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
        let mut hm = [[0.0f32; PALM_GRID]; PALM_GRID];
        for y in 0..PALM_GRID {
            for x in 0..PALM_GRID {
                hm[y][x] = sig(data[y * PALM_GRID + x]);
            }
        }
        // 3×3 local maxima above threshold, NMS'd at 2 cells, top 3.
        let mut cand: Vec<(f32, usize, usize)> = Vec::new();
        for y in 0..PALM_GRID {
            for x in 0..PALM_GRID {
                let v = hm[y][x];
                if v < thresh {
                    continue;
                }
                let mut is_max = true;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let (ny, nx) = (y as i32 + dy, x as i32 + dx);
                        if ny < 0 || nx < 0 || ny >= PALM_GRID as i32 || nx >= PALM_GRID as i32 {
                            continue;
                        }
                        if hm[ny as usize][nx as usize] > v {
                            is_max = false;
                        }
                    }
                }
                if is_max {
                    cand.push((v, y, x));
                }
            }
        }
        cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut picked: Vec<[f32; 3]> = Vec::new();
        for (v, y, x) in cand {
            if picked
                .iter()
                .any(|p| (p[2] - y as f32).powi(2) + (p[1] - x as f32).powi(2) < 4.0)
            {
                continue;
            }
            // parabolic sub-cell refine (clamped ±0.5), mirroring the trainer
            let refine = |l: f32, m: f32, r: f32| {
                let den = l - 2.0 * m + r;
                if den.abs() < 1e-6 {
                    0.0
                } else {
                    (0.5 * (l - r) / den).clamp(-0.5, 0.5)
                }
            };
            let dx = if x > 0 && x + 1 < PALM_GRID {
                refine(hm[y][x - 1], hm[y][x], hm[y][x + 1])
            } else {
                0.0
            };
            let dy = if y > 0 && y + 1 < PALM_GRID {
                refine(hm[y - 1][x], hm[y][x], hm[y + 1][x])
            } else {
                0.0
            };
            let tx = (x as f32 + dx + 0.5) * PALM_CELL;
            let ty = (y as f32 + dy + 0.5) * PALM_CELL;
            // un-letterbox (same geometry letterbox() applied)
            let r = (256.0 / width as f32).min(256.0 / height as f32);
            let nw = (width as f32 * r).round();
            let nh = (height as f32 * r).round();
            let pad_x = ((256.0 - nw) / 2.0).max(0.0);
            let pad_y = ((256.0 - nh) / 2.0).max(0.0);
            let fx = (tx - pad_x) / r;
            let fy = (ty - pad_y) / r;
            if !(0.0..width as f32).contains(&fx) || !(0.0..height as f32).contains(&fy) {
                continue;
            }
            picked.push([v, fx, fy]);
            if picked.len() >= 3 {
                break;
            }
        }
        picked
    }
}

impl PresenceNet {
    fn try_from_models_dir(dir: &Path) -> Option<Self> {
        let path = dir.join(PRESENCE_NET_FILE);
        if !path.is_file() {
            return None;
        }
        let (session, _backend) =
            build_session_cpu_only(&path.to_string_lossy(), 1, "RTMPose-hand-presence").ok()?;
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input".to_string());
        Some(Self {
            session,
            input_name,
            tensor: Array4::zeros((1, 3, 64, 64)),
        })
    }

    /// Score a wrist-centred sub-window (the training modality): a small
    /// window around the decoded wrist contains fingers vs sleeve/skin and
    /// stays free of the scene context that dominates the full crop.
    fn score(&mut self, rgb: &[u8], width: u32, height: u32, wrist: [f32; 2], base: f32) -> Option<f32> {
        use crate::tracking::detector::yolo_pose::fill_crop_tensor;
        let sub = (base * 0.35).max(48.0);
        let crop = (wrist[0] - sub / 2.0, wrist[1] - sub / 2.0, sub);
        if crop.0 > width as f32 || crop.1 > height as f32 {
            return None;
        }
        fill_crop_tensor(rgb, width, height, crop, &mut self.tensor);
        let input = match TensorRef::from_array_view(&self.tensor) {
            Ok(v) => v,
            Err(e) => {
                error!("hand-presence: tensor ref failed: {e}");
                return None;
            }
        };
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
            .ok()?;
        let out = outputs.get("logit")?;
        let (_, data) = out.try_extract_tensor::<f32>().ok()?;
        let logit = *data.first()?;
        let p = 1.0 / (1.0 + (-logit).exp());
        if std::env::var_os("VULVATAR_HAND_PRESENCE_DEBUG").is_some() {
            static COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 40 {
                info!(
                    "hand-presence: logit {logit:.3} presence {p:.3} at ({:.0},{:.0}) size {:.0}",
                    crop.0, crop.1, crop.2
                );
            }
        }
        Some(p)
    }
}

impl RtmposeHand {
    pub fn try_from_models_dir(dir: impl AsRef<Path>) -> Result<Option<Self>, String> {
        let dir = dir.as_ref();
        let Some(path) = RTMOSE_HAND_CANDIDATES
            .iter()
            .map(|c| dir.join(c))
            .find(|p| p.is_file())
        else {
            return Ok(None);
        };
        let (session, _backend) = build_session(&path.to_string_lossy(), 2, "RTMPose-hand")?;
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input".to_string());
        let size: u32 = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.rsplit('_').next())
            .and_then(|s| s.parse().ok())
            .unwrap_or(256);
        let chirality_flip = std::env::var("VULVATAR_HAND_CHIRALITY_FLIP")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let presence_net = PresenceNet::try_from_models_dir(dir);
        if presence_net.is_none() {
            warn!(
                "RTMPose hand presence model missing (models/{PRESENCE_NET_FILE}); \
                 falling back to the SimCC sharpness proxy"
            );
        }
        let palm_net = PalmNet::try_from_models_dir(dir);
        info!(
            "RTMPose hand model ready ({}, input '{}x{}', chirality_flip {})",
            path.display(),
            input_name,
            size,
            chirality_flip
        );
        Ok(Some(Self {
            session,
            input_name,
            size,
            tensor: Array4::<f32>::zeros((1, 3, size as usize, size as usize)),
            presence_net,
            palm_net,
            chirality_flip,
        }))
    }

    /// Run on a square crop `(x0, y0, size)` of the RGB frame, return
    /// landmarks in frame pixels.
    pub fn estimate(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
        crop: (f32, f32, f32),
    ) -> Option<HandResult> {
        use crate::tracking::detector::yolo_pose::fill_crop_tensor;
        let (x0, _y0, size) = crop;
        if size < 8.0 || rgb.len() < (width as usize) * (height as usize) * 3 {
            return None;
        }
        fill_crop_tensor(rgb, width, height, crop, &mut self.tensor);
        {
            let mut view = self.tensor.view_mut();
            for c in 0..3 {
                let (m, s) = (RTMOSE_MEAN[c], RTMOSE_STD[c]);
                view.slice_mut(ndarray::s![0, c, .., ..])
                    .mapv_inplace(|v| (v - m) / s);
            }
        }
        let input = match TensorRef::from_array_view(&self.tensor) {
            Ok(v) => v,
            Err(e) => {
                error!("rtmpose-hand: tensor ref failed: {e}");
                return None;
            }
        };
        let outputs = match self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
        {
            Ok(o) => o,
            Err(e) => {
                error!("rtmpose-hand: run failed: {e}");
                return None;
            }
        };
        let Some(out_x) = outputs.get("simcc_x") else {
            error!("rtmpose-hand: output 'simcc_x' missing");
            return None;
        };
        let Some(out_y) = outputs.get("simcc_y") else {
            error!("rtmpose-hand: output 'simcc_y' missing");
            return None;
        };
        let Ok((shape_x, data_x)) = out_x.try_extract_tensor::<f32>() else {
            error!("rtmpose-hand: simcc_x not f32");
            return None;
        };
        let Ok((_, data_y)) = out_y.try_extract_tensor::<f32>() else {
            error!("rtmpose-hand: simcc_y not f32");
            return None;
        };
        if shape_x.len() != 3 || shape_x[1] != HAND_KP as i64 {
            error!(
                "rtmpose-hand: unexpected simcc_x shape {:?} (expected [1, {} bins])",
                shape_x, HAND_KP
            );
            return None;
        }
        let bins = shape_x[2] as usize;
        let mut px = [[0.0f32; 3]; HAND_KP];
        let mut sharp: Vec<f32> = Vec::with_capacity(HAND_KP * 2);
        for j in 0..HAND_KP {
            let xs = &data_x[j * bins..(j + 1) * bins];
            let ys = &data_y[j * bins..(j + 1) * bins];
            let (ix, qx) = peak_and_sharpness(xs);
            let (iy, qy) = peak_and_sharpness(ys);
            sharp.push(qx);
            sharp.push(qy);
            px[j][0] = refine_peak(xs, ix) / bins as f32;
            px[j][1] = refine_peak(ys, iy) / bins as f32;
        }
        // Presence = SimCC sharpness median (crop-size dependent, and
        // ANTI-correlated with correctness on wrong-window decodes). The
        // provider re-scores every accepted candidate with the distilled
        // presence net — peak-centred for palm proposals, decoded-wrist
        // centred for the rest — which is calibrated and separates real
        // hands (~0.5-0.99) from cuff/face decodes (~0.00) with a wide
        // margin. Keeping the net call in the provider avoids scoring it
        // twice per candidate and lets palm candidates use the peak as the
        // window centre (the decoded wrist sits ~40 px off it on the chin
        // pose, which put the window on forearm fabric).
        let mut out = HandResult {
            presence: median(&mut sharp).clamp(0.0, 1.0),
            crop,
            ..Default::default()
        };
        // SimCC coords are normalised over the model input ([0,1] after the
        // bin division), so the frame mapping is x0 + n * crop_size — NOT
        // x0 + n * (crop_size / model_size): that collapses all 21 points
        // into a ~1 px blob at the crop origin.
        for (slot, k) in out.px.iter_mut().zip(px.iter()) {
            slot[0] = x0 + k[0] * size;
            slot[1] = _y0 + k[1] * size;
        }
        // Geometry gate: SimCC decodes each keypoint independently, so a
        // pattern-match on a face/forearm often collapses (palm ~0) or
        // stretches to an impossible skeleton. A real hand keeps the
        // knuckle-width / palm-length ratio in a narrow band — measured
        // 75% of true-hand windows in [0.15, 0.9] vs 22% of hallucinations.
        // `VULVATAR_HAND_NO_GEOM_GATE=1` disables.
        if std::env::var_os("VULVATAR_HAND_NO_GEOM_GATE").is_none() {
            let palm = ((out.px[9][0] - out.px[0][0]).powi(2)
                + (out.px[9][1] - out.px[0][1]).powi(2))
            .sqrt();
            let knuckle = ((out.px[5][0] - out.px[17][0]).powi(2)
                + (out.px[5][1] - out.px[17][1]).powi(2))
            .sqrt();
            let ratio = if palm > 1e-3 { knuckle / palm } else { 0.0 };
            if palm < 8.0 || !(0.15..=0.9).contains(&ratio) {
                static R: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let n = R.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 30 {
                    info!(
                        "hand-geom reject: palm {palm:.1}px knuckle {knuckle:.1}px ratio {ratio:.2} presence {:.2}",
                        out.presence
                    );
                }
                return None;
            }
        }
        if std::env::var_os("VULVATAR_HAND_GEOM_DEBUG").is_some() {
            // Palm-length / knuckle-width: a real hand sits in a narrow
            // ratio band; a SimCC pattern-match on a face/forearm does not.
            let palm = ((out.px[9][0] - out.px[0][0]).powi(2)
                + (out.px[9][1] - out.px[0][1]).powi(2))
            .sqrt();
            let knuckle = ((out.px[5][0] - out.px[17][0]).powi(2)
                + (out.px[5][1] - out.px[17][1]).powi(2))
            .sqrt();
            let ratio = if palm > 1e-3 { knuckle / palm } else { 0.0 };
            static G: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = G.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 60 {
                info!(
                    "hand-geom: palm {palm:.1}px knuckle {knuckle:.1}px ratio {ratio:.2} presence {:.2}",
                    out.presence
                );
            }
        }
        out.handedness = chirality_handedness(&out.px, self.chirality_flip);
        Some(out)
    }

    /// Re-score a palm-proposal result with the distilled presence net and
    /// OVERWRITE its presence with the calibrated score. The window is
    /// centred on the PALM HEATMAP PEAK (the detection), not the decoded
    /// wrist: SimCC decodes the chin-pose wrist ~40 px off the peak, and a
    /// window there sees forearm fabric — measured 0.00 net vs 0.89
    /// peak-centred on the same frames. SimCC sharpness is additionally
    /// crop-size dependent, so for palm proposals the net is the presence
    /// authority, full stop. `false` leaves the result untouched.
    pub fn rescore_presence(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
        res: &mut HandResult,
        centre: [f32; 2],
        base: f32,
    ) -> bool {
        let Some(net) = self.presence_net.as_mut() else {
            return false;
        };
        let Some(s) = net.score(rgb, width, height, centre, base) else {
            return false;
        };
        res.net_presence = Some(s);
        res.presence = s;
        true
    }
}

/// Argmax bin and its SimCC peak sharpness: softmax mass over absolute
/// deviation from the peak (1.0 = a delta spike, ~0.5 = a broad bump).
fn peak_and_sharpness(bins: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    let mut max = f32::NEG_INFINITY;
    for (i, v) in bins.iter().enumerate() {
        if *v > max {
            max = *v;
            best = i;
        }
    }
    let mut mass = 0.0f32;
    let mut abs = 0.0f32;
    for v in bins.iter() {
        let m = v - max;
        mass += m.exp();
        abs += m.abs();
    }
    (best, mass / (abs + mass))
}

/// 3-point parabolic sub-bin offset around `peak`, clamped to ±0.5 —
/// the same refinement `decode_simcc` applies to RTMW3D's SimCC.
fn refine_peak(bins: &[f32], peak: usize) -> f32 {
    if peak == 0 || peak + 1 >= bins.len() {
        return peak as f32;
    }
    let (l, m, r) = (bins[peak - 1], bins[peak], bins[peak + 1]);
    let denom = l - 2.0 * m + r;
    if denom.abs() < 1e-9 {
        return peak as f32;
    }
    peak as f32 + (0.5 * (l - r) / denom).clamp(-0.5, 0.5)
}

fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// 2-D landmark chirality → the pipeline's handedness convention
/// (`≈0.15` = left hand, `≈0.85` = right hand — the values MediaPipe's
/// head reads on this camera's frames, see the provider's veto bands).
///
/// Sign of the (index-MCP − wrist) × (pinky-MCP − wrist) cross product:
/// a RIGHT hand with the palm toward the camera has its index MCP on
/// the image LEFT of the pinky MCP, giving a positive cross in image
/// coordinates (x right, y down). Palm-away views invert the sign —
/// this heuristic cannot tell palm from back, which is the one thing
/// MediaPipe's learned head does better. The veto bands demote (σ×4)
/// rather than drop, and the provider's FK-based slot assignment stays
/// the L/R authority, so a wrong chirality degrades gracefully.
fn chirality_handedness(px: &[[f32; 3]; HAND_KP], flip: bool) -> f32 {
    let (w, index, pinky) = (px[0], px[5], px[17]);
    let cross =
        (index[0] - w[0]) * (pinky[1] - w[1]) - (index[1] - w[1]) * (pinky[0] - w[0]);
    let reads_right = if flip { cross < 0.0 } else { cross > 0.0 };
    if reads_right {
        0.85
    } else {
        0.15
    }
}

/// The hand crop-chain backend: RTMPose-m hand + distilled presence/palm
/// nets, the sole backend since the MediaPipe hand-landmarker was
/// removed on 2026-09-18 (the six-recording A/B gate passed: clasp
/// 780/781 duty 1.00/1.00 snaps 0/0 vs MediaPipe 782/782/0, chin duty
/// 0.90 vs 0.88, palms 0.89 vs 0.26, namaste 0.97/0.75 vs 0.66/0.72,
/// nohands FPs ≈ MediaPipe's own). Exports live in `models/` (gitignored
/// — see `RTMOSE_HAND_CANDIDATES` and the distilled-net filenames);
/// without them tracking fails loudly instead of silently losing hands.
pub enum HandBackend {
    Rtmpose(RtmposeHand),
}

impl HandBackend {
    pub fn try_from_models_dir(dir: impl AsRef<Path>) -> Result<Self, String> {
        match RtmposeHand::try_from_models_dir(&dir) {
            Ok(Some(b)) => Ok(Self::Rtmpose(b)),
            Ok(None) => Err(format!(
                "no RTMPose hand export in models/ (expected {}); the hand \
                 chain cannot start",
                RTMOSE_HAND_CANDIDATES[0]
            )),
            Err(e) => Err(format!("RTMPose hand model failed to load: {e}")),
        }
    }

    /// Run the hand model on a square frame crop; see
    /// [`RtmposeHand::estimate`] for the contract.
    pub fn estimate(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
        crop: (f32, f32, f32),
    ) -> Option<HandResult> {
        let Self::Rtmpose(r) = self;
        r.estimate(rgb, width, height, crop)
    }

    /// Full-frame wrist proposals `[score, x, y]` from the palm heatmap,
    /// best first. `None` when the export is missing — callers fall back
    /// to heuristic crops.
    pub fn palm_peaks(&mut self, rgb: &[u8], width: u32, height: u32) -> Option<Vec<[f32; 3]>> {
        let Self::Rtmpose(r) = self;
        r.palm_net.as_mut().map(|p| p.detect(rgb, width, height))
    }

    /// Calibrated presence re-score for a palm-proposal result, window
    /// centred on `centre` (the palm peak; see
    /// [`RtmposeHand::rescore_presence`]). `base` is the crop size the
    /// window scale derives from (see `PresenceNet::score`). No-op
    /// `false` when the distilled net is absent.
    pub fn rescore_presence(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
        res: &mut super::hands::HandResult,
        centre: [f32; 2],
        base: f32,
    ) -> bool {
        let Self::Rtmpose(r) = self;
        r.rescore_presence(rgb, width, height, res, centre, base)
    }

    /// Which backend was selected (GUI / debug surface).
    pub fn label(&self) -> &'static str {
        match self {
            Self::Rtmpose(_) => "rtmpose",
        }
    }
}

/// Predicted square crop for one hand from the model prediction: centred
/// on the wrist/MCP centroid, sized to the projected hand span × 2.4 (min
/// `min_px`). `None` when the hand is outside the frame or too small.
pub fn predicted_hand_crop(
    h: &Humanoid,
    fk: &Fk,
    hand: usize,
    intr: &Intrinsics,
    min_px: f64,
) -> Option<(f32, f32, f32)> {
    let wrist = if hand == 0 { h.j.l_wrist } else { h.j.r_wrist };
    let mut pts = vec![fk.t[wrist]];
    for f in 0..5 {
        pts.push(fk.t[h.j.finger[hand][f][0]]);
        pts.push(fk.site[h.s.tip[hand][f]]);
    }
    let mut umin = f64::INFINITY;
    let mut umax = f64::NEG_INFINITY;
    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    for p in &pts {
        let uv = intr.project(*p)?;
        umin = umin.min(uv[0]);
        umax = umax.max(uv[0]);
        vmin = vmin.min(uv[1]);
        vmax = vmax.max(uv[1]);
    }
    let cx = 0.5 * (umin + umax);
    let cy = 0.5 * (vmin + vmax);
    let span = (umax - umin).max(vmax - vmin);
    let size = (span * 1.6).max(min_px);
    // Require the centre inside the frame with some margin.
    if cx < -size * 0.2
        || cy < -size * 0.2
        || cx > intr.width + size * 0.2
        || cy > intr.height + size * 0.2
    {
        return None;
    }
    Some((
        (cx - size * 0.5) as f32,
        (cy - size * 0.5) as f32,
        size as f32,
    ))
}

/// Crop from the body detector's hand block (wrist + 20 landmarks, COCO
/// 91.. / 112..): bbox of the confident landmarks, padded. `None` when too
/// few landmarks are confident or the box is tiny.
pub fn detector_hand_crop(
    kps: &[(f32, f32, f32)],
    hand: usize,
    width: u32,
    height: u32,
    min_score: f32,
    min_px: f64,
) -> Option<(f32, f32, f32)> {
    let base = if hand == 0 { 91 } else { 112 };
    let mut umin = f64::INFINITY;
    let mut umax = f64::NEG_INFINITY;
    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    let mut n = 0;
    for i in base..(base + HAND_KP).min(kps.len()) {
        let (nx, ny, sc) = kps[i];
        if sc < min_score || !(0.0..=1.0).contains(&nx) || !(0.0..=1.0).contains(&ny) {
            continue;
        }
        let u = nx as f64 * width as f64;
        let v = ny as f64 * height as f64;
        umin = umin.min(u);
        umax = umax.max(u);
        vmin = vmin.min(v);
        vmax = vmax.max(v);
        n += 1;
    }
    if n < 6 {
        return None;
    }
    let cx = 0.5 * (umin + umax);
    let cy = 0.5 * (vmin + vmax);
    let span = (umax - umin).max(vmax - vmin);
    if span < 12.0 {
        return None;
    }
    let size = (span * 1.8).max(min_px);
    Some((
        (cx - size * 0.5) as f32,
        (cy - size * 0.5) as f32,
        size as f32,
    ))
}

/// Crop centred just BELOW the face — the streaming "hand at mouth/chin"
/// pose, where the body wrist keypoint is unreliable and the FK prior
/// lags behind a raising arm. Placement from the nose + eye keypoints;
/// `None` when they are not confident.
pub fn face_below_hand_crop(
    kps: &[(f32, f32, f32)],
    width: u32,
    height: u32,
    min_score: f32,
) -> Option<(f32, f32, f32)> {
    let nose = kps.get(0)?;
    let le = kps.get(1)?;
    let re = kps.get(2)?;
    if nose.2 < min_score || le.2 < min_score || re.2 < min_score {
        return None;
    }
    let ear_span = ((le.0 - re.0).abs() * width as f32)
        .max((le.1 - re.1).abs() * height as f32)
        .max(40.0);
    let nx = nose.0 * width as f32;
    let ny = nose.1 * height as f32;
    let down = ear_span * 1.4;
    let size = ear_span * 2.2;
    let cx = nx;
    let cy = ny + down;
    Some((cx - size / 2.0, cy - size / 2.0, size))
}

/// Crop seeded from the BODY wrist/elbow keypoints (COCO 7/9 left,
/// 8/10 right) — the wholebody hand blocks [`detector_hand_crop`]
/// relies on are gone with YOLO26, so this is the only detector-side
/// seed left for hand acquisition. The centre extrapolates past the
/// wrist along the elbow→wrist direction (the hand extends beyond the
/// wrist joint), sized from the projected forearm length; without a
/// confident elbow it falls back to a fixed-size crop centred on the
/// wrist. `None` when the wrist is unconfident / out of frame.
pub fn wrist_hand_crop(
    kps: &[(f32, f32, f32)],
    hand: usize,
    width: u32,
    height: u32,
    min_score: f32,
    min_px: f64,
) -> Option<(f32, f32, f32)> {
    let (el, wr) = if hand == 0 { (7usize, 9usize) } else { (8, 10) };
    let px = |i: usize| -> Option<(f64, f64)> {
        let (nx, ny, sc) = kps.get(i)?;
        if *sc < min_score || !(0.0..=1.0).contains(nx) || !(0.0..=1.0).contains(ny) {
            return None;
        }
        Some((*nx as f64 * width as f64, *ny as f64 * height as f64))
    };
    let (wu, wv) = px(wr)?;
    let (cu, cv, size) = match px(el) {
        Some((eu, ev)) => {
            let du = wu - eu;
            let dv = wv - ev;
            let fore = (du * du + dv * dv).sqrt();
            let size = (fore * 1.2).clamp(min_px, 320.0);
            if fore < 1.0 {
                (wu, wv, size)
            } else {
                // Centre a quarter crop-size past the wrist along the
                // forearm direction.
                (
                    wu + du / fore * size * 0.25,
                    wv + dv / fore * size * 0.25,
                    size,
                )
            }
        }
        // No confident elbow (bent arms hide it on desk/palms sessions):
        // enlarge generously instead of extrapolating — the hand may
        // extend in any direction from the wrist joint, and a wrist-tight
        // crop only ever sees the cuff.
        None => (wu, wv, min_px * 2.4),
    };
    if cu < -size * 0.2
        || cv < -size * 0.2
        || cu > width as f64 + size * 0.2
        || cv > height as f64 + size * 0.2
    {
        return None;
    }
    Some((
        (cu - size * 0.5) as f32,
        (cv - size * 0.5) as f32,
        size as f32,
    ))
}

/// Turn a hand result into estimator observations: 21 2-D keypoints
/// (σ from crop scale) and, when an absolute wrist position is known,
/// absolute 3-D points for the finger joints from the wrist-relative
/// world layout. `hand` = 0 left, 1 right (model side the crop was for).
#[allow(clippy::too_many_arguments)]
pub fn hand_observations(
    h: &Humanoid,
    hand: usize,
    res: &HandResult,
    wrist_abs: Option<V3>,
    width: u32,
    height: u32,
    // `sigma_scale` > 1 when the result is suspect (e.g. its handedness
    // contradicts the slot it was cropped for).
    sigma_scale: f64,
    out2d: &mut Vec<Kp2d>,
    out3d: &mut Vec<Kp3d>,
    out_ang: &mut Vec<AngleObs>,
) {
    // Denominator of the calibrated landmark-σ base, retained from the
    // MediaPipe era where it was the model input side (224). The σ was
    // empirically calibrated through it and the six-recording A/B gate
    // (2026-09-18) passed with this value; changing it re-scales every
    // hand observation σ.
    const HAND_SIGMA_CROP_DIVISOR: f64 = 224.0;
    let scale = res.crop.2 as f64 / HAND_SIGMA_CROP_DIVISOR;
    // Landmark noise measured live (2026-09-14, MCP keypoints on a held
    // desk hand) is 15–19 px rms frame-to-frame at 640w — far above this
    // σ — but inflating it destabilises the torso basin (same replay:
    // ×4 flips torso yaw mean −23.7°→+59.5°; ×8 → −20.5°). The hand
    // terms are load-bearing beyond the hand, so the calibrated crop-space
    // base stays; `VULVATAR_HAND_SIGMA_SCALE` multiplies it for future
    // work (the right fix is per-finger angle observations, not global
    // inflation).
    let sigma_scale_env = std::env::var("VULVATAR_HAND_SIGMA_SCALE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(1.0);
    // Fingers-only multiplier (VULVATAR_FINGER_SIGMA_SCALE): inflating the
    // WRIST landmark σ destabilises the torso basin (it anchors the arm
    // chain → clavicle → spine yaw; measured −23.7°→+59.5° flip), so the
    // wrist keeps the calibrated σ while the finger landmarks — measured
    // 15–33 px rms in poor-visibility sessions vs ~1 px σ — loosen alone.
    let finger_scale_env = std::env::var("VULVATAR_FINGER_SIGMA_SCALE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(1.0);
    let sigma_px = (2.0 * scale).max(1.0)
        * (1.0 + 2.0 * (1.0 - res.presence as f64))
        * sigma_scale
        * sigma_scale_env;
    let wrist = if hand == 0 { h.j.l_wrist } else { h.j.r_wrist };
    let mut points: [Option<ModelPoint>; HAND_KP] = [None; HAND_KP];
    points[0] = Some(ModelPoint::Joint(wrist));
    for f in 0..5 {
        let fj = h.j.finger[hand][f];
        points[1 + f * 4] = Some(ModelPoint::Joint(fj[0]));
        points[2 + f * 4] = Some(ModelPoint::Joint(fj[2]));
        points[3 + f * 4] = Some(ModelPoint::Joint(fj[3]));
        points[4 + f * 4] = Some(ModelPoint::Site(h.s.tip[hand][f]));
    }
    for i in 0..HAND_KP {
        let Some(pt) = points[i] else { continue };
        let (u, v) = (res.px[i][0] as f64, res.px[i][1] as f64);
        if !u.is_finite()
            || !v.is_finite()
            || u < 0.0
            || v < 0.0
            || u >= width as f64
            || v >= height as f64
        {
            continue;
        }
        out2d.push(Kp2d {
            point: pt,
            u,
            v,
            sigma: if i == 0 { sigma_px } else { sigma_px * finger_scale_env },
        });
        if let Some(w) = wrist_abs {
            if i > 0 {
                // NOTE: currently unreachable from the provider — the
                // converted checkpoint's "world" output is not metric and
                // its proportions proved too noisy for Cartesian terms
                // even after per-frame re-scaling (measured wrist snap
                // 1→3–4, max jump 0.48 m). The intended revival is as
                // finger-joint ANGLE observations from landmark triples
                // (invariant to scale and to the wrist anchor), not as
                // these absolute points.
                let rel = [
                    res.world[i][0] as f64,
                    res.world[i][1] as f64,
                    res.world[i][2] as f64,
                ];
                out3d.push(Kp3d {
                    point: pt,
                    p: add(w, rel),
                    sigma: 0.05 * (1.0 + 2.0 * (1.0 - res.presence as f64)) * sigma_scale,
                    lat_scale: 1.0,
                });
            }
        }
    }
    // Inter-phalanx angle observations from landmark triples — the
    // scale/wrist-invariant core of finger curl. Per finger: the angle at
    // the MCP (wrist–MCP–PIP), at the PIP (MCP–PIP–DIP) and at the DIP
    // (PIP–DIP–tip). σ propagates the landmark σ through the angle
    // (dθ ≈ σ·(1/d₁ + 1/d₂)), floored so a near-degenerate triple can't
    // become an infinitely sharp constraint. The thumb's CMC angle is
    // skipped (its model triple is not co-planar with the landmark one).
    let push_angle = |out_ang: &mut Vec<AngleObs>,
                      (ia, iv, ib): (usize, usize, usize),
                      (pa, pv, pb): (ModelPoint, ModelPoint, ModelPoint)| {
        let valid = |p: [f32; 3]| {
            p[0].is_finite()
                && p[1].is_finite()
                && p[0] >= 0.0
                && p[1] >= 0.0
                && (p[0] as f64) < width as f64
                && (p[1] as f64) < height as f64
        };
        let (a, v, b) = (res.px[ia], res.px[iv], res.px[ib]);
        if !valid(a) || !valid(v) || !valid(b) {
            return;
        }
        let u = [a[0] as f64 - v[0] as f64, a[1] as f64 - v[1] as f64];
        let w = [b[0] as f64 - v[0] as f64, b[1] as f64 - v[1] as f64];
        let d1 = (u[0] * u[0] + u[1] * u[1]).sqrt();
        let d2 = (w[0] * w[0] + w[1] * w[1]).sqrt();
        if d1 < 4.0 || d2 < 4.0 {
            return; // foreshortened segment: the angle is pure noise
        }
        let cross = u[0] * w[1] - u[1] * w[0];
        let dot = u[0] * w[0] + u[1] * w[1];
        let sigma = (sigma_px * (1.0 / d1 + 1.0 / d2)).max(0.06);
        out_ang.push(AngleObs {
            a: pa,
            vertex: pv,
            b: pb,
            angle: cross.atan2(dot),
            sigma: sigma * sigma_scale,
        });
    };
    for f in 0..5 {
        let (mcp, pip, dip, tip) = (
            points[1 + f * 4].unwrap(),
            points[2 + f * 4].unwrap(),
            points[3 + f * 4].unwrap(),
            points[4 + f * 4].unwrap(),
        );
        let wrist_mp = ModelPoint::Joint(wrist);
        push_angle(out_ang, (0, 1 + f * 4, 2 + f * 4), (wrist_mp, mcp, pip));
        push_angle(out_ang, (1 + f * 4, 2 + f * 4, 3 + f * 4), (mcp, pip, dip));
        push_angle(out_ang, (2 + f * 4, 3 + f * 4, 4 + f * 4), (pip, dip, tip));
    }
}

#[cfg(test)]
mod chirality_tests {
    use super::*;

    fn layout(index: [f32; 2], pinky: [f32; 2]) -> [[f32; 3]; HAND_KP] {
        let mut px = [[0.0f32; 3]; HAND_KP];
        px[0] = [100.0, 100.0, 0.0];
        px[5] = [index[0], index[1], 0.0];
        px[17] = [pinky[0], pinky[1], 0.0];
        px
    }

    #[test]
    fn right_palm_reads_right_left_palm_reads_left() {
        // Palm toward the camera, fingers up: the RIGHT hand's index MCP
        // sits on the image LEFT of the pinky MCP.
        let right = layout([80.0, 70.0], [120.0, 65.0]);
        assert!((chirality_handedness(&right, false) - 0.85).abs() < 1e-6);
        let left = layout([120.0, 70.0], [80.0, 65.0]);
        assert!((chirality_handedness(&left, false) - 0.15).abs() < 1e-6);
    }

    #[test]
    fn flip_env_inverts_the_mapping() {
        let right = layout([80.0, 70.0], [120.0, 65.0]);
        assert!((chirality_handedness(&right, true) - 0.15).abs() < 1e-6);
    }

    #[test]
    fn back_of_hand_inverts_like_a_flipped_palm() {
        // Palm away from the camera: the chirality sign inverts, which is
        // the documented limit of the geometric heuristic (the veto bands
        // demote, they don't drop — see chirality_handedness).
        let right_back = layout([120.0, 70.0], [80.0, 65.0]);
        assert!((chirality_handedness(&right_back, false) - 0.15).abs() < 1e-6);
    }
}

#[cfg(test)]
mod wrist_crop_tests {
    use super::*;

    fn kps_133() -> Vec<(f32, f32, f32)> {
        vec![(0.0, 0.0, 0.0); 133]
    }

    #[test]
    fn wrist_crop_centres_past_the_wrist_along_the_forearm() {
        let mut kps = kps_133();
        // Left elbow at (0.3, 0.5), wrist at (0.5, 0.5) on a 640x480 frame:
        // forearm points +x, 128 px long → size = 153.6, centre past the
        // wrist by a quarter size.
        kps[7] = (0.30, 0.50, 0.9);
        kps[9] = (0.50, 0.50, 0.9);
        let (x, y, size) = wrist_hand_crop(&kps, 0, 640, 480, 0.35, 96.0).unwrap();
        assert!((size - 153.6).abs() < 0.1, "size {size}");
        let cx = x as f64 + size as f64 * 0.5;
        let cy = y as f64 + size as f64 * 0.5;
        assert!((cx - (320.0 + 153.6 * 0.25)).abs() < 0.1, "cx {cx}");
        assert!((cy - 240.0).abs() < 0.1, "cy {cy}");
    }

    #[test]
    fn wrist_crop_falls_back_to_the_wrist_without_an_elbow() {
        let mut kps = kps_133();
        kps[10] = (0.50, 0.60, 0.9);
        let (x, y, size) = wrist_hand_crop(&kps, 1, 640, 480, 0.35, 96.0).unwrap();
        assert!((size - 96.0 * 2.4).abs() < 0.01, "elbow-less fallback enlarges the crop");
        assert!((x as f64 + size as f64 * 0.5 - 320.0).abs() < 0.1);
        assert!((y as f64 + size as f64 * 0.5 - 288.0).abs() < 0.1);
    }

    #[test]
    fn wrist_crop_rejects_unconfident_or_out_of_frame_wrists() {
        let mut kps = kps_133();
        kps[9] = (0.50, 0.60, 0.2);
        assert!(wrist_hand_crop(&kps, 0, 640, 480, 0.35, 96.0).is_none(), "low score");
        let mut kps = kps_133();
        kps[9] = (1.20, 0.60, 0.9);
        assert!(wrist_hand_crop(&kps, 0, 640, 480, 0.35, 96.0).is_none(), "out of frame");
    }
}
