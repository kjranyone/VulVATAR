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
/// Consecutive sidecar failures (spawn or roundtrip) before the chain
/// latches off for the session instead of respawning at frame rate.
const MAX_FAIL_ROUNDS: u32 = 3;
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
    /// slow EAR baselines (image-left eye, image-right eye, mouth)
    ear_base: [f32; 3],
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
        // out = scale * r @ p + t (numpy mirror: r = vt^T @ u^T applied
        // non-transposed; the transposed form rotated by -theta and was
        // masked while svd2 returned V as u — r was symmetric identity
        // then, so both bugs cancelled into "no rotation at all")
        [
            self.scale * (self.r[0][0] * p[0] + self.r[0][1] * p[1]) + self.t[0],
            self.scale * (self.r[1][0] * p[0] + self.r[1][1] * p[1]) + self.t[1],
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
    // r = vt^T @ u^T, i.e. r[i][j] = Σ_k vt[k][i] * u[j][k] (numpy:
    // R = Vt.T @ U.T). The former vt^T @ u form applied the wrong
    // transpose of u.
    let mut r = [
        [vt[0][0] * u[0][0] + vt[1][0] * u[0][1],
         vt[0][0] * u[1][0] + vt[1][0] * u[1][1]],
        [vt[0][1] * u[0][0] + vt[1][1] * u[0][1],
         vt[0][1] * u[1][0] + vt[1][1] * u[1][1]],
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
        mu_b[0] - scale * (r[0][0] * mu_a[0] + r[0][1] * mu_a[1]),
        mu_b[1] - scale * (r[1][0] * mu_a[0] + r[1][1] * mu_a[1]),
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
///
/// V comes from the symmetric eigen of m^T m; U = m V Σ⁺ (the singular
/// vectors actually scaled by m) — NOT V itself. Feeding V as U made
/// r = V V^T = identity, so the Procrustes fit degenerated to
/// scale + translation with no rotation (measured: residual ~82 px on a
/// ~20°-yawed face vs ~11 px with a proper rotation; the > 60 px
/// rejection then dropped every frame and the face chain never fired).
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
    let vt = [[v1[0], v1[1]], [v2[0], v2[1]]];
    // U = m V Σ⁺, one singular vector at a time; rank-deficient axes get
    // the orthogonal complement of the previous column.
    let mut u = [[0.0f64; 2]; 2];
    for (k, (&sv, v)) in s.iter().zip([v1, v2].iter()).enumerate() {
        if sv > 1e-12 {
            u[0][k] = (m[0][0] * v[0] + m[0][1] * v[1]) / sv;
            u[1][k] = (m[1][0] * v[0] + m[1][1] * v[1]) / sv;
        } else {
            u[0][k] = -u[1][0];
            u[1][k] = u[0][0];
        }
    }
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
            ear_base: [f32::NAN; 3],
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
        // Latch: a dead sidecar (missing litert / python / script) fails
        // deterministically, so after MAX_FAIL_ROUNDS stop respawning —
        // the alternative is a respawn + error log at frame rate for the
        // rest of the session. Tracking restart re-arms.
        if self.fail_run >= MAX_FAIL_ROUNDS {
            return None;
        }
        if self.child.is_none() {
            if let Err(e) = self.spawn() {
                self.fail_run += 1;
                if self.fail_run >= MAX_FAIL_ROUNDS {
                    error!(
                        "face sidecar failed to start {} rounds in a row ({e}) — \
                         expressions disabled for this session",
                        self.fail_run
                    );
                }
                return None;
            }
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
        if self.fail_run >= MAX_FAIL_ROUNDS {
            error!(
                "face sidecar failed {} rounds in a row — expressions disabled \
                 for this session (tracking restart re-arms)",
                self.fail_run
            );
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
            if std::env::var_os("VULVATAR_FACE_DEBUG").is_some() {
                info!("face debug: bbox too small ({size:.0})");
            }
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
        let face_dbg = std::env::var_os("VULVATAR_FACE_DEBUG").is_some();
        let Some(pts_norm) = self.roundtrip(&crop) else {
            if face_dbg {
                info!("face debug: roundtrip returned None");
            }
            return None;
        };
        // crop-px coordinates
        let pts: Vec<[f64; 2]> = pts_norm
            .iter()
            .map(|p| [f64::from(p[0]) * CROP as f64, f64::from(p[1]) * CROP as f64])
            .collect();

        // Procrustes the canonical mesh's 98 anchor slots onto the
        // measured points; the residual doubles as the confidence.
        let anchors_canon: Vec<[f64; 2]> = self
            .wflw_to_mp478
            .iter()
            .map(|&mi| [f64::from(self.canonical[mi][0]), f64::from(self.canonical[mi][1])])
            .collect();
        let Some(sim) = sim_solve(&anchors_canon, &pts) else {
            if face_dbg {
                info!("face debug: sim_solve degenerate");
            }
            return None;
        };

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
        if face_dbg {
            info!(
                "face debug: residual {residual:.1} px (thr 60), sidecar span x [{:.0},{:.0}] y [{:.0},{:.0}], canon span x [{:.0},{:.0}] y [{:.0},{:.0}]",
                pts.iter().map(|p| p[0]).fold(f64::MAX, f64::min),
                pts.iter().map(|p| p[0]).fold(f64::MIN, f64::max),
                pts.iter().map(|p| p[1]).fold(f64::MAX, f64::min),
                pts.iter().map(|p| p[1]).fold(f64::MIN, f64::max),
                anchors_canon.iter().map(|p| p[0]).fold(f64::MAX, f64::min),
                anchors_canon.iter().map(|p| p[0]).fold(f64::MIN, f64::max),
                anchors_canon.iter().map(|p| p[1]).fold(f64::MAX, f64::min),
                anchors_canon.iter().map(|p| p[1]).fold(f64::MIN, f64::max),
            );
        }
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
                (q[0] / CROP as f64 * size as f64 + f64::from(bbox.0)) as f32,
                (q[1] / CROP as f64 * size as f64 + f64::from(bbox.1)) as f32,
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
        // min via fold(.., f64::min) — the former f64::max fold made
        // w = max − f64::MAX (a huge negative), every ratio went
        // negative and the [0,1] clamp zeroed every expression channel.
        let w = (seg.iter().map(|p| p[0]).fold(f64::MIN, f64::max)
            - seg.iter().map(|p| p[0]).fold(f64::MAX, f64::min))
        .max(1.0);
        let h = seg.iter().map(|p| p[1]).fold(f64::MIN, f64::max)
            - seg.iter().map(|p| p[1]).fold(f64::MAX, f64::min);
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
        // ARKit semantics: 0 = open, 1 = closed — the raw ratio reads
        // the opposite way (≈1 at rest), so map closedness through the
        // latch threshold instead of publishing the ratio itself (the
        // former raw-ratio channels sat at 1.0 rest / 0.5 blink, fully
        // inverted).
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
        let closedness = |r: f32, latched: bool| -> f32 {
            let c = ((1.0 - r) / (1.0 - BLINK_ENTER)).clamp(0.0, 1.0);
            if latched { 1.0 } else { c }
        };
        // Mouth ring gets its OWN baseline slot — routing it through an
        // eye slot let the mouth's much larger ratio corrupt that eye's
        // baseline. jawOpen: rest ratio ≈ baseline → 0; opening raises
        // the ring's h/w well above it.
        let mar = self.ring_ratio(pts, MOUTH, 2);
        let jaw = ((mar - 1.0) / 0.8).clamp(0.0, 1.0);
        let blink_l = closedness(ear[0], self.blink_latched[0]);
        let blink_r = closedness(ear[1], self.blink_latched[1]);
        let mut out = Vec::with_capacity(8);
        // ARKit names (subject-left ring is the image-RIGHT ring on an
        // unmirrored camera; WFLW 60..68 sits on the image left, which
        // the MP path's mirror convention assigns to subject LEFT).
        out.push(expr("eyeBlinkLeft", blink_l));
        out.push(expr("eyeBlinkRight", blink_r));
        out.push(expr("blinkLeft", blink_r));
        out.push(expr("blinkRight", blink_l));
        out.push(expr("blink", (blink_l + blink_r) * 0.5));
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

#[cfg(test)]
impl FaceRtmpose {
    fn new_for_tests() -> Self {
        Self {
            child: None,
            python: String::new(),
            script: std::path::PathBuf::new(),
            canonical: Vec::new(),
            wflw_to_mp478: Vec::new(),
            blend: None,
            ear_base: [f32::NAN; 3],
            blink_latched: [false; 2],
            fail_run: 0,
        }
    }
}

/// ARKit blendshape names in the distilled MLP's output order (index 0
/// `_neutral` skipped at the call site) — the same ordering the retired
/// MediaPipe BlendshapeV2 head used, kept so distilled labels and the
/// downstream ARKit aggregation stay aligned.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sim_solve_recovers_rotation_scale_translation() {
        // The Procrustes must recover a ~20° rotated, scaled, translated
        // target (the desk head-yaw regime) — the identity-rotation bug
        // in svd2 made this residual ~80 px, tripping the > 60 px gate
        // and killing the whole face chain.
        let theta = 20.0f64.to_radians();
        let r = [[theta.cos(), -theta.sin()], [theta.sin(), theta.cos()]];
        let scale = 1.17f64;
        let t = [42.0, -17.0];
        let src: Vec<[f64; 2]> = (0..98)
            .map(|i| {
                let a = i as f64 * 0.618;
                [(a * 7.0).sin() * 90.0 + 130.0, (a * 11.0).cos() * 60.0 + 140.0]
            })
            .collect();
        let dst: Vec<[f64; 2]> = src
            .iter()
            .map(|p| {
                [
                    scale * (r[0][0] * p[0] + r[1][0] * p[1]) + t[0],
                    scale * (r[0][1] * p[0] + r[1][1] * p[1]) + t[1],
                ]
            })
            .collect();
        let sim = sim_solve(&src, &dst).expect("sim must solve");
        let err: f64 = src
            .iter()
            .zip(&dst)
            .map(|(s, d)| {
                let q = sim.apply(*s);
                ((q[0] - d[0]).powi(2) + (q[1] - d[1]).powi(2)).sqrt()
            })
            .sum::<f64>()
            / src.len() as f64;
        assert!(err.sqrt() < 1e-6, "residual {err}");
    }

    #[test]
    fn ear_distinguishes_open_and_closed_eye() {
        // WFLW98 eye ring at slots 60..68. The baseline EMA seeds from
        // the first (open) frame, so feeding a closed ring afterwards
        // must drop eyeBlinkLeft well below the open baseline.
        let mut face = FaceRtmpose::new_for_tests();
        let open: Vec<[f64; 2]> = (0..98)
            .map(|i| {
                let (x, y) = match i {
                    60..=67 => ((i - 60) as f64 * 10.0, ((i - 60) as f64 * 3.0).sin() * 8.0 + 100.0),
                    68..=75 => (200.0 + (i - 68) as f64 * 10.0, ((i - 68) as f64 * 3.0).sin() * 8.0 + 100.0),
                    76..=95 => (100.0 + (i - 76) as f64 * 5.0, 160.0),
                    _ => (50.0 + i as f64, 50.0 + i as f64),
                };
                [x, y]
            })
            .collect();
        let _ = face.geometric_expressions(&open);
        let closed: Vec<[f64; 2]> = open
            .iter()
            .enumerate()
            .map(|(i, p)| if (60..=67).contains(&i) { [p[0], 100.0] } else { *p })
            .collect();
        let exprs = face.geometric_expressions(&closed);
        let blink_l = exprs.iter().find(|e| e.name == "eyeBlinkLeft").unwrap().weight;
        assert!(
            blink_l > 0.5,
            "closed eye should read as blink (weight {blink_l})"
        );
        // latch releases and the channel returns to ~0 once the ring
        // re-opens past BLINK_EXIT
        let open_exprs = face.geometric_expressions(&open);
        let open_v = open_exprs.iter().find(|e| e.name == "eyeBlinkLeft").unwrap().weight;
        assert!(open_v < 0.2, "open eye {open_v} should read as not-blinking");

        // jaw: a flat mouth line reads closed; a tall ring reads open
        let mut shut = face.geometric_expressions(&open);
        let jaw_shut = shut.iter().find(|e| e.name == "jawOpen").unwrap().weight;
        assert!(jaw_shut < 0.2, "flat mouth jawOpen {jaw_shut}");
        let open_mouth: Vec<[f64; 2]> = open
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if (76..=88).contains(&i) {
                    [p[0], p[1] - 30.0]
                } else if (88..=95).contains(&i) {
                    [p[0], p[1] + 30.0]
                } else {
                    *p
                }
            })
            .collect();
        let talking = face.geometric_expressions(&open_mouth);
        let jaw_open = talking.iter().find(|e| e.name == "jawOpen").unwrap().weight;
        assert!(jaw_open > 0.5, "open mouth jawOpen {jaw_open}");
    }

    #[test]
    fn svd2_returns_u_not_v() {
        // An anisotropic asymmetric matrix: U must differ from V and
        // reconstruct m = U Σ V^T.
        let m = [[3.0, 1.0], [0.5, 2.0]];
        let (u, s, vt) = svd2(m);
        let mut rec = [[0.0f64; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                rec[i][j] = u[i][0] * s[0] * vt[0][j] + u[i][1] * s[1] * vt[1][j];
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((rec[i][j] - m[i][j]).abs() < 1e-9);
            }
        }
        // proper rotations: det ≈ +1 for both factors
        let det_u = u[0][0] * u[1][1] - u[0][1] * u[1][0];
        let det_vt = vt[0][0] * vt[1][1] - vt[0][1] * vt[1][0];
        assert!((det_u - 1.0).abs() < 1e-9, "det u {det_u}");
        assert!((det_vt - 1.0).abs() < 1e-9, "det vt {det_vt}");
    }
}
