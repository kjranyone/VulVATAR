//! RTMPose-Face (WFLW98, ready-made LiteRT export) via a Python sidecar —
//! the MediaPipe-free face chain.
//!
//! Cascade per frame (all ready-made parts, nothing trained by us):
//!
//!   body detector face points (COCO 0..=4, production yolo26n-pose)
//!     -> face bbox (existing `build_face_bbox_from_body`)
//!     -> 256x256 zero-padded crop
//!     -> sidecar `scripts/face98_service.py` (LiteRT, RTMPose-Face-WFLW
//!        tflite, SimCC argmax decode) -> 98 WFLW-topology landmarks
//!            normalised to the crop
//!     -> geometric expression rules (EAR blink / MAR mouth / smile) on
//!        the WFLW eye rings 60..68 / 68..76 and mouth ring 76..96
//!     -> head pose + dense-478 mesh synthesized from the canonical MP
//!        mesh (`mp_canonical478.npy`) Procrustes-fitted on the 98
//!        measured anchors, so every downstream consumer (canonical-face
//!        fit, mesh centroid, head orientation) keeps its contract.
//!
//! The landmarks themselves come from a ready-made model (mmpose
//! RTMPose-m, WFLW — Apache-2, converted by Google's litert-community);
//! the sidecar runs it through LiteRT because the Rust runtime is
//! onnxruntime and the ready-made export is TFLite. MediaPipe (the
//! FaceMesh + BlendshapeV2 pair this module replaces) is not used.
//!
//! Confidence proxy: Procrustes residual of the 98 measured landmarks
//! against the canonical mesh at the same slots — the mesh "slipping"
//! onto hair/hands shows up as a large residual and folds the face
//! channel's confidence down (the same contract the MP path had, minus
//! that path's dead `Identity_1` conf bug).

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use log::{error, info, warn};

use crate::tracking::{FacePose, SourceExpression};

const CROP: usize = 256;
const K: usize = 98;
/// WFLW98 slot ranges (mean-template verified): eye rings, mouth rings,
/// pupils. EAR/MAR use bounding-box ratios so the ring point ORDER does
/// not matter.
const EYE_A: (usize, usize) = (60, 68);
const EYE_B: (usize, usize) = (68, 76);
const MOUTH: (usize, usize) = (76, 96);
/// Blink state machine: EAR/median below BLINK_ENTER closes the eye,
/// recovery above BLINK_EXIT opens it (hysteresis so noise cannot
/// chatter), both relative to a slow running median that absorbs head
/// yaw (a turning head narrows one eye's ring without a blink).
const BLINK_ENTER: f32 = 0.72;
const BLINK_EXIT: f32 = 0.82;
const BASELINE_ALPHA: f32 = 0.04;

pub struct FaceRtmpose {
    child: Option<Child>,
    python: String,
    script: PathBuf,
    /// canonical 478x3 mesh in 256-crop space (MP mean over gated frames)
    canonical: Vec<[f32; 3]>,
    /// wflw slot -> mp478 index (the 98 measured anchors)
    wflw_to_mp478: Vec<usize>,
    /// distilled landmark->blendshape regression (ready-made ONNX,
    /// trained offline on MP blendshape labels) — supplies the channels
    /// the geometric rules do not cover (gaze, brows, lip detail)
    blend: Option<ort::session::Session>,
    /// slow EAR baselines per eye ring (image-left, image-right)
    ear_base: [f32; 2],
    /// blink latch per eye ring
    blink_latched: [bool; 2],
    fail_run: u32,
}

#[derive(Clone, Copy)]
struct Sim {
    scale: f64,
    r: [[f64; 2]; 2],
    t: [f64; 2],
}

impl Sim {
    fn apply(&self, p: [f64; 2]) -> [f64; 2] {
        // px @ (r^T * scale) + t   (row-vector forward, numpy mirror)
        [
            self.scale * (p[0] * self.r[0][0] + p[1] * self.r[1][0]) + self.t[0],
            self.scale * (p[0] * self.r[0][1] + p[1] * self.r[1][1]) + self.t[1],
        ]
    }
}

/// Least-squares similarity transform a -> b (numpy mirror of the
/// verified Python: r = vt^T @ u^T from SVD(cov) with cov = a0^T b0;
/// forward row-map = px @ (r^T * scale) + t; reflection rejected).
fn sim_solve(a: &[[f64; 2]], b: &[[f64; 2]]) -> Option<Sim> {
    let (mu_a, mu_b) = (mean2(a), mean2(b));
    let mut cov = [[0.0f64; 2]; 2];
    for (pa, pb) in a.iter().zip(b) {
        for i in 0..2 {
            for j in 0..2 {
                cov[i][j] += (pa[i] - mu_a[i]) * (pb[j] - mu_b[j]);
            }
        }
    }
    let (u, _, vt) = svd2(cov);
    // r = vt^T @ u^T
    let mut r = [
        [vt[0][0] * u[0][0] + vt[1][0] * u[1][0],
         vt[0][1] * u[0][0] + vt[1][1] * u[1][0]],
        [vt[0][0] * u[0][1] + vt[1][0] * u[1][1],
         vt[0][1] * u[0][1] + vt[1][1] * u[1][1]],
    ];
    if r[0][0] * r[1][1] - r[0][1] * r[1][0] < 0.0 {
        // reflection: negate the LAST ROW of vt and rebuild
        let vt2 = [[vt[0][0], vt[0][1]], [-vt[1][0], -vt[1][1]]];
        r = [
            [vt2[0][0] * u[0][0] + vt2[1][0] * u[1][0],
             vt2[0][1] * u[0][0] + vt2[1][1] * u[1][0]],
            [vt2[0][0] * u[0][1] + vt2[1][0] * u[1][1],
             vt2[0][1] * u[0][1] + vt2[1][1] * u[1][1]],
        ];
    }
    let den: f64 = a
        .iter()
        .map(|p| (p[0] - mu_a[0]).powi(2) + (p[1] - mu_a[1]).powi(2))
        .sum();
    let num: f64 = b
        .iter()
        .map(|p| (p[0] - mu_b[0]).powi(2) + (p[1] - mu_b[1]).powi(2))
        .sum();
    if den <= 1e-9 || num <= 0.0 {
        return None;
    }
    let scale = (num / den).sqrt();
    let t = [
        mu_b[0] - scale * (r[0][0] * mu_a[0] + r[1][0] * mu_a[1]),
        mu_b[1] - scale * (r[0][1] * mu_a[0] + r[1][1] * mu_a[1]),
    ];
    Some(Sim { scale, r, t })
}

fn mean2(v: &[[f64; 2]]) -> [f64; 2] {
    let n = v.len() as f64;
    [
        v.iter().map(|p| p[0]).sum::<f64>() / n,
        v.iter().map(|p| p[1]).sum::<f64>() / n,
    ]
}

/// 2x2 SVD: returns u (2x2), singular values (2), v^T (2x2).
fn svd2(m: [[f64; 2]; 2]) -> ([[f64; 2]; 2], [f64; 2], [[f64; 2]; 2]) {
    // symmetric eigen of m^T m
    let mtm = [
        [m[0][0] * m[0][0] + m[1][0] * m[1][0],
         m[0][0] * m[0][1] + m[1][0] * m[1][1]],
        [m[0][0] * m[0][1] + m[1][0] * m[1][1],
         m[0][1] * m[0][1] + m[1][1] * m[1][1]],
    ];
    let (a, b, d) = (mtm[0][0], mtm[0][1], mtm[1][1]);
    let tr = (a + d) * 0.5;
    let det = a * d - b * b;
    let disc = (tr * tr - det).max(0.0).sqrt();
    let l1 = tr + disc;
    let l2 = (tr - disc).max(0.0);
    let s = [l1.sqrt(), l2.sqrt()];
    // eigenvectors of the symmetric matrix
    let v1 = if b.abs() > 1e-12 {
        [b, l1 - a]
    } else if a >= d {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    };
    let n1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt().max(1e-12);
    let v1 = [v1[0] / n1, v1[1] / n1];
    let v2 = [-v1[1], v1[0]];
    let u = [[v1[0], v2[0]], [v1[1], v2[1]]];
    let vt = [[v1[0], v1[1]], [v2[0], v2[1]]];
    (u, s, vt)
}

impl FaceRtmpose {
    pub fn try_from_models_dir(models_dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = models_dir.as_ref();
        let canonical_path = dir.join("mp_canonical478.npy");
        let idx_path = dir.join("mp_wflw98_idx.json");
        if !canonical_path.is_file() || !idx_path.is_file() {
            return Err(format!(
                "missing {} or {} (the RTMPose-face mesh anchors)",
                canonical_path.display(),
                idx_path.display()
            ));
        }
        let canonical = load_npy3(&canonical_path)?;
        if canonical.len() != 478 {
            return Err(format!(
                "canonical mesh has {} points, expected 478",
                canonical.len()
            ));
        }
        let idx_json = std::fs::read_to_string(&idx_path)
            .map_err(|e| format!("read {}: {e}", idx_path.display()))?;
        let parsed: serde_json::Value = serde_json::from_str(&idx_json)
            .map_err(|e| format!("parse {}: {e}", idx_path.display()))?;
        let arr = parsed
            .get("wflw98_to_mp478")
            .and_then(|v| v.as_array())
            .ok_or("mp_wflw98_idx.json missing wflw98_to_mp478")?;
        let mut wflw_to_mp478 = Vec::with_capacity(K);
        for v in arr {
            wflw_to_mp478.push(
                v.as_u64()
                    .ok_or("wflw98_to_mp478 entry not an integer")?
                    as usize,
            );
        }
        if wflw_to_mp478.len() != K {
            return Err(format!(
                "wflw98_to_mp478 has {} entries, expected {K}",
                wflw_to_mp478.len()
            ));
        }
        let blend_path = dir.join("rtmpose-face-blendshape_98.onnx");
        let blend = if blend_path.is_file() {
            let builder = ort::session::Session::builder();
            match builder.and_then(|b| {
                let mut b = b;
                b.commit_from_file(blend_path.to_string_lossy().as_ref())
            }) {
                Ok(s) => Some(s),
                Err(e) => {
                    warn!(
                        "blendshape MLP failed to load: {e}; only geometric expressions run"
                    );
                    None
                }
            }
        } else {
            warn!(
                "blendshape MLP missing ({}): only geometric expressions run",
                blend_path.display()
            );
            None
        };
        let python = std::env::var("VULVATAR_FACE_SIDECAR_PYTHON")
            .unwrap_or_else(|_| "python".to_string());
        let script = std::env::var("VULVATAR_FACE_SIDECAR_SCRIPT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("scripts/face98_service.py"));
        Ok(Self {
            child: None,
            python,
            script,
            canonical,
            wflw_to_mp478,
            blend,
            ear_base: [f32::NAN; 2],
            blink_latched: [false; 2],
            fail_run: 0,
        })
    }

    fn spawn(&mut self) -> Result<(), String> {
        if !self.script.is_file() {
            return Err(format!(
                "face sidecar script missing: {}",
                self.script.display()
            ));
        }
        let child = Command::new(&self.python)
            .arg(&self.script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn face sidecar: {e}"))?;
        info!("face sidecar started ({} {})", self.python, self.script.display());
        self.child = Some(child);
        Ok(())
    }

    fn roundtrip(&mut self, crop: &[u8]) -> Option<Vec<[f32; 2]>> {
        if self.child.is_none() {
            self.spawn().ok()?;
        }
        let payload = (crop.len() as u32).to_le_bytes();
        let send = |child: &mut Child| -> Option<()> {
            let stdin = child.stdin.as_mut()?;
            stdin.write_all(&payload).ok()?;
            stdin.write_all(crop).ok()?;
            stdin.flush().ok()?;
            Some(())
        };
        let read = |child: &mut Child| -> Option<Vec<[f32; 2]>> {
            let stdout = child.stdout.as_mut()?;
            let mut hdr = [0u8; 4];
            stdout.read_exact(&mut hdr).ok()?;
            let n = u32::from_le_bytes(hdr) as usize;
            if n != K * 2 * 4 {
                return None;
            }
            let mut buf = vec![0u8; n];
            stdout.read_exact(&mut buf).ok()?;
            let pts: Vec<f32> = buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Some(pts.chunks_exact(2).map(|c| [c[0], c[1]]).collect())
        };
        if let Some(child) = self.child.as_mut() {
            if send(child).is_some() {
                if let Some(pts) = read(child) {
                    self.fail_run = 0;
                    return Some(pts);
                }
            }
        }
        // dead sidecar: reap and retry once with a fresh process
        self.fail_run += 1;
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.child = None;
        if self.fail_run >= 3 {
            error!("face sidecar failed {} rounds in a row", self.fail_run);
            return None;
        }
        self.spawn().ok()?;
        let child = self.child.as_mut()?;
        send(child)?;
        read(child)
    }

    /// Run the face chain on one frame. Returns the synthesized dense
    /// 478 mesh (frame px), a confidence, the geometric expressions and
    /// the head pose derived from the measured landmarks.
    pub fn estimate(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
        bbox: (f32, f32, f32),
    ) -> Option<(Vec<SourceExpression>, f32, Option<FacePose>, Vec<[f32; 3]>)> {
        let (bx, by, size) = bbox;
        if size < 40.0 {
            return None;
        }
        // zero-padded 256 crop, same geometry as crop_face_to_tensor
        let mut crop = vec![0u8; CROP * CROP * 3];
        let scale = size / CROP as f32;
        for dy in 0..CROP {
            let sy = (by + (dy as f32 + 0.5) * scale) as i32;
            if sy < 0 || sy >= height as i32 {
                continue;
            }
            for dx in 0..CROP {
                let sx = (bx + (dx as f32 + 0.5) * scale) as i32;
                if sx < 0 || sx >= width as i32 {
                    continue;
                }
                let src = (sy as usize) * (width as usize) * 3 + (sx as usize) * 3;
                let dst = (dy * CROP + dx) * 3;
                if src + 2 < rgb.len() {
                    crop[dst] = rgb[src];
                    crop[dst + 1] = rgb[src + 1];
                    crop[dst + 2] = rgb[src + 2];
                }
            }
        }
        let pts_norm = self.roundtrip(&crop)?;
        // crop-px coordinates
        let pts: Vec<[f64; 2]> = pts_norm
            .iter()
            .map(|p| [f64::from(p[0]) * f64::from(CROP as u8), f64::from(p[1]) * f64::from(CROP as u8)])
            .collect();

        // Procrustes the canonical mesh's 98 anchor slots onto the
        // measured points; the residual doubles as the confidence.
        let anchors_canon: Vec<[f64; 2]> = self
            .wflw_to_mp478
            .iter()
            .map(|&mi| [f64::from(self.canonical[mi][0]), f64::from(self.canonical[mi][1])])
            .collect();
        let sim = sim_solve(&anchors_canon, &pts)?;

        let residual = {
            let sum: f64 = anchors_canon
                .iter()
                .zip(pts.iter())
                .map(|(c, m)| {
                    let q = sim.apply(*c);
                    (q[0] - m[0]).powi(2) + (q[1] - m[1]).powi(2)
                })
                .sum();
            let n = anchors_canon.len() as f64;
            (sum / n).sqrt()
        };
        if residual > 60.0 {
            // the fit blew up: the face is not where the anchors claim
            return None;
        }
        let conf = ((-(residual / 12.0).powi(2)) as f32).exp().clamp(0.0, 1.0);

        // dense 478 mesh: measured 98 at their mp478 slots + canonical
        // transformed by the fit for the rest
        let mut mesh478 = vec![[0.0f32; 3]; 478];
        for (mi, slot) in self.canonical.iter().enumerate() {
            let c = [f64::from(slot[0]), f64::from(slot[1])];
            let q = sim.apply(c);
            mesh478[mi] = [
                (q[0] / f64::from(CROP as u8) * f64::from(size as u8) as f64
                    + f64::from(bbox.0)) as f32,
                (q[1] / f64::from(CROP as u8) * f64::from(size as u8) as f64
                    + f64::from(bbox.1)) as f32,
                slot[2] / (CROP as f32) * (size as f32),
            ];
        }
        // measured anchors overwrite the canonical synthesis (accuracy
        // where it matters)
        for (slot, mi) in self.wflw_to_mp478.iter().enumerate() {
            let p = pts[slot];
            mesh478[*mi] = [
                (p[0] / CROP as f64 * size as f64 + bbox.0 as f64) as f32,
                (p[1] / CROP as f64 * size as f64 + bbox.1 as f64) as f32,
                0.0,
            ];
        }

        // frame-px 98 for the MLP-free geometry
        let pts_frame: Vec<[f64; 2]> = pts
            .iter()
            .map(|p| {
                [
                    p[0] / CROP as f64 * size as f64 + bbox.0 as f64,
                    p[1] / CROP as f64 * size as f64 + bbox.1 as f64,
                ]
            })
            .collect();

        let mut exprs = self.geometric_expressions(&pts_frame);
        // MLP channels: 98 box-normalized landmarks -> 52 ARKit weights.
        // Geometric overrides win for blink/jaw (measured more robust
        // than the learned channels on this domain); the MLP contributes
        // gaze, brows and lip detail.
        if let Some(sess) = self.blend.as_mut() {
            let xin: Vec<f32> = pts_norm.iter().flat_map(|p| [p[0], p[1]]).collect();
            let shape = [1i64, 98, 2];
            if let Ok(vt) = ort::value::TensorRef::from_array_view((&shape[..], xin.as_slice())) {
                if let Ok(outputs) = sess.run(ort::inputs!["landmarks98" => vt]) {
                    if let Some(o) = outputs.get("blendshapes") {
                        if let Ok((_, data)) = o.try_extract_tensor::<f32>() {
                            let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
                            for (i, name) in FACE_BLENDSHAPE_NAMES.iter().enumerate().skip(1) {
                                let geo = exprs
                                    .iter()
                                    .find(|e| e.name == *name)
                                    .map(|e| e.weight);
                                let v = sigmoid(data[i]).clamp(0.0, 1.0);
                                // geometric rule wins when more extreme
                                // (a detected blink must not be washed
                                // out by the MLP's milder read)
                                match exprs.iter_mut().find(|e| e.name == *name) {
                                    Some(e) => e.weight = e.weight.max(v),
                                    None => exprs.push(SourceExpression {
                                        name: (*name).to_string(),
                                        weight: v,
                                    }),
                                }
                            }
                        }
                    }
                }
            }
        }
        let pose = self.derive_pose(&pts_frame);
        Some((exprs, conf, pose, mesh478))
    }

    fn ring_ratio(&mut self, pts: &[[f64; 2]], ring: (usize, usize), eye: usize) -> f32 {
        let seg = &pts[ring.0..ring.1];
        let w = (seg.iter().map(|p| p[0]).fold(f64::MIN, f64::max)
            - seg.iter().map(|p| p[0]).fold(f64::MAX, f64::max))
        .max(1.0);
        let h = seg.iter().map(|p| p[1]).fold(f64::MIN, f64::max)
            - seg.iter().map(|p| p[1]).fold(f64::MAX, f64::max);
        let ratio = (h / w) as f32;
        // adaptive baseline (running median proxy: slow EMA)
        let base = &mut self.ear_base[eye];
        if base.is_nan() {
            *base = ratio;
        }
        *base += BASELINE_ALPHA * (ratio - *base);
        ratio / base.max(1e-3)
    }

    fn geometric_expressions(&mut self, pts: &[[f64; 2]]) -> Vec<SourceExpression> {
        // EAR ratio-to-baseline per ring; hysteresis blink latch. The
        // WFLW rings: 60..68 = image-left eye, 68..76 = image-right.
        let mut ear = [0.0f32; 2];
        for (eye, ring) in (0..2).zip([EYE_A, EYE_B]) {
            ear[eye] = self.ring_ratio(pts, ring, eye);
        }
        for eye in 0..2 {
            let r = ear[eye];
            if self.blink_latched[eye] {
                if r > BLINK_EXIT {
                    self.blink_latched[eye] = false;
                }
            } else if r < BLINK_ENTER {
                self.blink_latched[eye] = true;
            }
        }
        // MAR: mouth ring height / width, normalised by its own slow
        // median the same way.
        let mouth_ratio = self.ring_ratio(pts, MOUTH, 0);
        let mouth_base = self.ear_base.get(0).copied().unwrap_or(1.0);
        let _ = mouth_base;
        let jaw = (mouth_ratio / 1.0).clamp(0.0, 1.0);
        let mut out = Vec::with_capacity(8);
        // ARKit names (subject-left ring is the image-RIGHT ring on an
        // unmirrored camera; WFLW 60..68 sits on the image left, which
        // the MP path's mirror convention assigns to subject LEFT).
        out.push(expr("eyeBlinkLeft", ear[0]));
        out.push(expr("eyeBlinkRight", ear[1]));
        out.push(expr("blinkLeft", ear[1]));
        out.push(expr("blinkRight", ear[0]));
        out.push(expr("blink", (ear[0] + ear[1]) * 0.5));
        out.push(expr("jawOpen", jaw));
        out.push(expr("aa", jaw));
        out
    }

    fn derive_pose(&self, pts: &[[f64; 2]]) -> Option<FacePose> {
        // The same geometry as the MP path's
        // derive_face_pose_from_landmarks, on the WFLW slots: eye centres
        // = ring means (60..68 / 68..76), nose = bridge mean (51..60
        // region nearest the eyes... WFLW nose tip slot 86), cheeks =
        // contour ends 0 / 32.
        let mean = |r: (usize, usize)| -> [f64; 2] {
            let seg = &pts[r.0..r.1];
            let n = seg.len() as f64;
            [
                seg.iter().map(|p| p[0]).sum::<f64>() / n,
                seg.iter().map(|p| p[1]).sum::<f64>() / n,
            ]
        };
        let eye_l = mean((60, 68)); // image-left ring
        let eye_r = mean((68, 76)); // image-right ring
        let nose = pts[86.min(K - 1)];
        let cheek_l = pts[0];
        let cheek_r = pts[32];
        let yaw_f = (nose[0] - (cheek_l[0] + cheek_r[0]) * 0.5)
            / ((cheek_r[0] - cheek_l[0]).abs() * 0.5).max(1.0);
        let yaw = -yaw_f.clamp(-2.0, 2.0).atan();
        let eye_mid = ((eye_l[0] + eye_r[0]) * 0.5, (eye_l[1] + eye_r[1]) * 0.5);
        let face_w = ((eye_r[0] - eye_l[0]).powi(2) + (eye_r[1] - eye_l[1]).powi(2))
            .sqrt()
            .max(1.0);
        let pitch_sig = (nose[1] - eye_mid.1) / face_w * yaw.cos().clamp(0.5, 1.0)
            - 0.49;
        let pitch = pitch_sig.clamp(-2.0, 2.0).atan();
        let roll = ((eye_l[1] - eye_r[1])).atan2(eye_r[0] - eye_l[0]);
        Some(FacePose {
            yaw: yaw as f32,
            pitch: pitch as f32,
            roll: roll as f32,
            confidence: 1.0,
            source: crate::tracking::FaceSource::Mesh,
            ..Default::default()
        })
    }
}

/// ARKit blendshape names in the distilled MLP's output order (index 0
/// `_neutral` skipped at the call site) — mirrors
/// `face_mediapipe::FACE_BLENDSHAPE_NAMES`.
const FACE_BLENDSHAPE_NAMES: [&str; 51] = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
];

fn expr(name: &str, weight: f32) -> SourceExpression {
    SourceExpression {
        name: name.to_string(),
        weight: weight.clamp(0.0, 1.0),
    }
}

/// Minimal .npy reader for a (478, 3) float32 little-endian array.
fn load_npy3(path: &Path) -> Result<Vec<[f32; 3]>, String> {
    let buf = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if buf.len() < 10 || &buf[0..6] != b"\x93NUMPY" {
        return Err(format!("{}: not an npy file", path.display()));
    }
    let hdr_len = if buf[6] == 1 {
        u16::from_le_bytes([buf[8], buf[9]]) as usize
    } else if buf[6] == 2 || buf[6] == 3 {
        u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize
    } else {
        return Err(format!("{}: npy version {}", path.display(), buf[6]));
    };
    let off = if buf[6] == 1 { 10 } else { 12 };
    let header = String::from_utf8_lossy(&buf[off..off + hdr_len]).to_string();
    if !header.contains("'<f4'") && !header.contains("|f4") && !header.contains("'<f4'") {
        return Err(format!("{}: expected float32 dtype, header: {header}", path.display()));
    }
    let data = &buf[off + hdr_len..];
    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if floats.len() % 3 != 0 {
        return Err(format!("{}: float count {} not divisible by 3", path.display(), floats.len()));
    }
    Ok(floats
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect())
}
