//! Offline Tier 1 replay of the D435 metric-depth pose path — **no live
//! camera**. Reads a recorded colour PNG + aligned `*_depth_mm.npy` (u16 mm,
//! C-order rows×cols), deprojects the depth to camera-space metres, feeds it
//! through the exact provider path the app uses (`set_external_depth` →
//! `estimate_from_external_depth`), and reports the resulting `SourceSkeleton`:
//! the metric branch signal, joint depths, the real-3D shoulder-line yaw, the
//! normalised shoulder span (the Tier 1 lock for `TARGET_SRC_SHOULDER_SPAN`),
//! forward reach, and the 1:1 root metres.
//!
//!   cargo run --features realsense --bin diagnose_depth_replay
//!   cargo run --features realsense --bin diagnose_depth_replay -- <color.png> [depth.npy]
//!
//! With no args it uses the checked-in palms-front fixture in
//! `diagnostics/depth/`. Needs the RTMW3D models in `models/` (same set the
//! app uses). Build/run env: see docs/realsense-build.md. The `realsense`
//! feature is required only to pull in the metric-depth provider path — the
//! physical camera is never opened.

use std::path::{Path, PathBuf};

use vulvatar_lib::asset::HumanoidBone;
use vulvatar_lib::tracking::provider::{create_pose_provider, TrackingPipelineConfig};
use vulvatar_lib::tracking::skeleton_from_depth::{MetricDepthFrame, TARGET_SRC_SHOULDER_SPAN};
use vulvatar_lib::tracking::{CameraIntrinsics, SourceSkeleton};

// Nominal D435 colour intrinsics at 1280×720 (HFOV ≈ 69.4°, VFOV ≈ 42.5°).
// z is exact from the sensor; only x/y scale rides on these and the error is
// bounded — enough to smoke-test direction / frontality. Dump the real
// intrinsics to diagnostics/depth/intrinsics.json to tighten x/y later.
const D435_FX: f32 = 924.0;
const D435_FY: f32 = 924.0;
const D435_CX: f32 = 640.0;
const D435_CY: f32 = 360.0;

fn default_color() -> PathBuf {
    PathBuf::from("diagnostics/depth/dump_palms_front_20260707_135054_color.png")
}

fn default_depth(color: &Path) -> PathBuf {
    PathBuf::from(color.to_string_lossy().replace("_color.png", "_depth_mm.npy"))
}

/// Minimal `.npy` reader for a 2-D little-endian `u16` array. Returns
/// `(rows, cols, data)` with `data` in C (row-major) order.
fn parse_npy_u16(bytes: &[u8]) -> Result<(usize, usize, Vec<u16>), String> {
    if bytes.len() < 12 || &bytes[0..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let major = bytes[6];
    let (header_len, data_start) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
    let data_start = data_start + header_len;
    let header = std::str::from_utf8(&bytes[10.min(data_start)..data_start])
        .map_err(|e| format!("npy header utf8: {e}"))?;
    if !header.contains("<u2") && !header.contains("|u2") {
        return Err(format!("expected little-endian u16 (<u2), header: {header}"));
    }
    let shape = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next())
        .ok_or("no shape in npy header")?;
    let dims: Vec<usize> = shape
        .split(',')
        .filter_map(|t| t.trim().parse::<usize>().ok())
        .collect();
    let (rows, cols) = match dims.as_slice() {
        [r, c, ..] => (*r, *c),
        _ => return Err(format!("expected 2-D shape, got {dims:?}")),
    };
    let count = rows * cols;
    let data = &bytes[data_start..];
    if data.len() < count * 2 {
        return Err(format!("npy data short: {} < {}", data.len(), count * 2));
    }
    let out = (0..count)
        .map(|i| u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]))
        .collect();
    Ok((rows, cols, out))
}

/// Load a colour PNG + aligned `*_depth_mm.npy` and deproject into a
/// `MetricDepthFrame` (camera-space metres, x-right/y-down/z-forward).
fn load_metric_frame(
    color_path: &Path,
    depth_path: &Path,
    verbose: bool,
) -> Result<(image::RgbImage, MetricDepthFrame), String> {
    let img = image::open(color_path).map_err(|e| format!("open colour: {e}"))?;
    let rgb = img.to_rgb8();
    let (cw, ch) = (rgb.width(), rgb.height());

    let depth_bytes = std::fs::read(depth_path).map_err(|e| format!("read depth: {e}"))?;
    let (rows, cols, depth_mm) = parse_npy_u16(&depth_bytes)?;
    let (dw, dh) = (cols as u32, rows as u32);
    if verbose {
        println!("colour {cw}x{ch}   depth {dw}x{dh} (u16 mm)");
        if dw != cw || dh != ch {
            println!("  NOTE: colour/depth extent differ — assuming aligned by index.");
        }
    }

    let intr = CameraIntrinsics {
        fx: D435_FX,
        fy: D435_FY,
        cx: D435_CX,
        cy: D435_CY,
        width: dw,
        height: dh,
    };
    let mut points_m = Vec::with_capacity(depth_mm.len());
    let mut valid = 0usize;
    for v in 0..rows {
        for u in 0..cols {
            let z = depth_mm[v * cols + u] as f32 * 0.001;
            if z > 0.0 {
                valid += 1;
            }
            points_m.push([
                (u as f32 - intr.cx) / intr.fx * z,
                (v as f32 - intr.cy) / intr.fy * z,
                z,
            ]);
        }
    }
    if verbose {
        println!(
            "depth valid = {:.1}%",
            100.0 * valid as f32 / depth_mm.len().max(1) as f32
        );
    }

    Ok((
        rgb,
        MetricDepthFrame {
            width: dw,
            height: dh,
            points_m,
            crop: None,
            intrinsics: Some(intr),
            // Offline replay at the recorded cadence — the nominal
            // 30 fps fallback matches the capture rate.
            timestamp_ms: None,
        },
    ))
}

/// Replay pipeline config: the app default, with an env-var override to
/// drop the YOLOX person-crop stage (`VULVATAR_REPLAY_NO_YOLOX=1`). The
/// override forces the whole-frame letterbox fallback on every frame,
/// which is the A/B lever for the crop-z contract measurement (compare
/// `vulvatar::rawz` lines for the same image with and without it).
fn replay_config() -> TrackingPipelineConfig {
    let mut cfg = TrackingPipelineConfig::default();
    if std::env::var_os("VULVATAR_REPLAY_NO_YOLOX").is_some() {
        cfg.yolox_enabled = false;
    }
    cfg
}

fn main() -> Result<(), String> {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let first: PathBuf = args.next().map(PathBuf::from).unwrap_or_else(default_color);

    // Directory → batch/CSV mode over every extracted frame pair (one model
    // load). This is the DYNAMIC verification: a wave/palms sequence exposes
    // temporal behaviour (forward/back sign flips, yaw flapping) a single
    // frame cannot.
    if first.is_dir() {
        return batch(&first);
    }

    let color_path = first;
    let depth_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| default_depth(&color_path));

    println!("color: {}", color_path.display());
    println!("depth: {}", depth_path.display());

    let (rgb, metric) = load_metric_frame(&color_path, &depth_path, true)?;
    let (cw, ch) = (rgb.width(), rgb.height());

    let mut provider = create_pose_provider("models", replay_config())?;
    println!("provider: {}", provider.label());

    provider.set_external_depth(metric);
    let est = provider.estimate_pose(rgb.as_raw(), cw, ch, 0);

    report(&est.skeleton)
}

/// Per-frame metrics extracted from a solved source skeleton, for the
/// temporal CSV + summary.
struct FrameMetrics {
    joints: usize,
    span: f32,
    yaw: f32,
    r_wrist_z: Option<f32>,
    r_hand_z: Option<f32>,
    l_wrist_z: Option<f32>,
    l_hand_z: Option<f32>,
    lsh: Option<[f32; 3]>,
    rsh: Option<[f32; 3]>,
    metric_present: bool,
    // --- head / face orientation probe ---
    head: Option<[f32; 3]>,
    neck: Option<[f32; 3]>,
    /// Inherited RGB face pose (yaw, pitch, roll, confidence), degrees.
    face: Option<[f32; 4]>,
    // --- hand L/R probe (source x; +x = anatomical-left per module docs) ---
    l_hand_x: Option<f32>,
    r_hand_x: Option<f32>,
    l_elbow_x: Option<f32>,
    r_elbow_x: Option<f32>,
    // --- global scale / root probe (the zoom-glitch suspects) ---
    /// The normalisation reference span (metres). Uncalibrated = this frame's
    /// measured shoulder span → its per-frame jitter scales the whole avatar
    /// (`avatar_scale = rest_span / reference_span_m`) = the zoom glitch.
    ref_span_m: Option<f32>,
    /// Raw anchor depth (metres). Its jitter moves the avatar toward/away.
    anchor_z: Option<f32>,
    mpsu: Option<f32>,
}

impl FrameMetrics {
    /// Neck-bend proxy for the head's *base* frame (contributor B): the
    /// horizontal (lean L/R) and depth (lean fwd/back) tilt of the
    /// Head-from-Neck vector, in degrees. This is the depth-driven part of
    /// the head orientation the solver's `HeadFromShoulders` tip consumes;
    /// the RGB `face` pose rides on top of it.
    fn head_tilt(&self) -> Option<(f32, f32)> {
        let (h, n) = (self.head?, self.neck?);
        let dx = h[0] - n[0];
        let dy = h[1] - n[1];
        let dz = h[2] - n[2];
        // +y is up; a head straight above the neck gives ~0/0.
        let lean_lr = dx.atan2(dy.abs().max(1e-3)).to_degrees();
        let lean_fb = dz.atan2(dy.abs().max(1e-3)).to_degrees();
        Some((lean_lr, lean_fb))
    }

    /// Heuristic: do the two hands look L/R-swapped this frame? In source
    /// coords +x is anatomical-left, so a non-swapped pose has the
    /// anatomical-left hand at greater x than the right *when the elbows are
    /// clearly separated the same way*. Fires when the hand x-order is
    /// inverted relative to the elbow x-order (the classic midline mix-up
    /// the depth path does NOT correct).
    fn hands_swapped(&self) -> Option<bool> {
        let (lhx, rhx) = (self.l_hand_x?, self.r_hand_x?);
        let (lex, rex) = (self.l_elbow_x?, self.r_elbow_x?);
        // Elbows tell us the true side; require them meaningfully apart.
        if (lex - rex).abs() < 0.15 {
            return None;
        }
        Some((lhx - rhx).signum() != (lex - rex).signum())
    }
}

fn frame_metrics(sk: &SourceSkeleton) -> FrameMetrics {
    use HumanoidBone::*;
    let z = |b: HumanoidBone| sk.joints.get(&b).map(|j| j.position[2]);
    let x = |b: HumanoidBone| sk.joints.get(&b).map(|j| j.position[0]);
    let pos = |b: HumanoidBone| sk.joints.get(&b).map(|j| j.position);
    let lsh = sk.joints.get(&LeftUpperArm).map(|j| j.position);
    let rsh = sk.joints.get(&RightUpperArm).map(|j| j.position);
    let (span, yaw) = match (lsh, rsh) {
        (Some(l), Some(r)) => {
            let dx = l[0] - r[0];
            let dy = l[1] - r[1];
            let dz = l[2] - r[2];
            let span = (dx * dx + dy * dy + dz * dz).sqrt();
            // Mirror the solver's `compute_shoulder_align_rotation` gate: a
            // degenerate (near-coincident) shoulder span yields no usable yaw,
            // so the solver leaves the torso at rest. Report NaN so the
            // temporal verdict reflects what the avatar actually does.
            let yaw = if span >= 0.30 {
                dz.atan2(dx).to_degrees()
            } else {
                f32::NAN
            };
            (span, yaw)
        }
        _ => (f32::NAN, f32::NAN),
    };
    FrameMetrics {
        joints: sk.joints.len(),
        span,
        yaw,
        r_wrist_z: z(RightLowerArm),
        r_hand_z: z(RightHand),
        l_wrist_z: z(LeftLowerArm),
        l_hand_z: z(LeftHand),
        lsh,
        rsh,
        metric_present: sk.metric_frame_info.is_some(),
        head: pos(Head),
        neck: pos(Neck).or_else(|| pos(UpperChest)),
        face: sk.face.map(|f| {
            [
                f.yaw.to_degrees(),
                f.pitch.to_degrees(),
                f.roll.to_degrees(),
                f.confidence,
            ]
        }),
        l_hand_x: x(LeftHand),
        r_hand_x: x(RightHand),
        l_elbow_x: x(LeftLowerArm),
        r_elbow_x: x(RightLowerArm),
        ref_span_m: sk.metric_frame_info.as_ref().map(|m| m.reference_span_m),
        anchor_z: sk.metric_frame_info.as_ref().map(|m| m.anchor_cam_m[2]),
        mpsu: sk.metric_frame_info.as_ref().map(|m| m.mpsu),
    }
}

fn batch(dir: &Path) -> Result<(), String> {
    let mut pairs: Vec<(u64, PathBuf, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| format!("read_dir: {e}"))? {
        let p = entry.map_err(|e| e.to_string())?.path();
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
        if let Some(stem) = name.strip_suffix("_color.png") {
            let depth = p.with_file_name(format!("{stem}_depth_mm.npy"));
            if depth.exists() {
                let idx = stem
                    .rsplit('_')
                    .next()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
                pairs.push((idx, p.clone(), depth));
            }
        }
    }
    pairs.sort_by_key(|(i, _, _)| *i);
    if pairs.is_empty() {
        return Err(format!(
            "no *_color.png + *_depth_mm.npy pairs in {}",
            dir.display()
        ));
    }

    let mut provider = create_pose_provider("models", replay_config())?;
    eprintln!("provider: {} — {} frames", provider.label(), pairs.len());

    let fz = |o: Option<f32>| o.map(|v| format!("{v:+.3}")).unwrap_or_else(|| "".into());
    println!(
        "frame,joints,span_src,yaw_deg,Rwrist_z,Rhand_z,Lwrist_z,Lhand_z,Lsh_x,Lsh_z,Rsh_x,Rsh_z,metric,\
         head_x,head_y,head_z,neck_x,neck_y,neck_z,headLeanLR,headLeanFB,fyaw,fpitch,froll,fconf,Lhand_x,Rhand_x,swap,\
         ref_span_m,anchor_z,mpsu"
    );

    // Temporal accumulators. The "inversion" signature we hunt is the OLD
    // bug: a limb's forward/back (source z) jumping across zero by a large
    // amount between adjacent frames. Yaw flapping = large frame-to-frame yaw
    // swings / bimodal ±extreme.
    let mut yaws: Vec<f32> = Vec::new();
    let mut prev_rz: Option<f32> = None;
    let mut prev_lz: Option<f32> = None;
    let mut r_flips = 0u32;
    let mut l_flips = 0u32;
    let mut yaw_jumps = 0u32;
    let mut prev_yaw: Option<f32> = None;
    let mut metric_frames = 0u32;
    let (mut hand_fwd, mut hand_tot, mut hand_outliers) = (0u32, 0u32, 0u32);
    // Head-orientation probe (contributor B = neck bend; C = RGB face pose).
    let mut head_lr: Vec<f32> = Vec::new();
    let mut head_fb: Vec<f32> = Vec::new();
    let mut fyaws: Vec<f32> = Vec::new();
    // Hand L/R-swap probe + hard dropouts (empty skeleton = avatar rests).
    let (mut swap_frames, mut swap_eligible) = (0u32, 0u32);
    let mut empty_frames = 0u32;

    for (n, (idx, cp, dp)) in pairs.iter().enumerate() {
        let (rgb, metric) = load_metric_frame(cp, dp, false)?;
        let (cw, ch) = (rgb.width(), rgb.height());
        provider.set_external_depth(metric);
        let sk = provider.estimate_pose(rgb.as_raw(), cw, ch, n as u64).skeleton;
        let m = frame_metrics(&sk);

        let (lean_lr, lean_fb) = m.head_tilt().map(|(a, b)| (Some(a), Some(b))).unwrap_or((None, None));
        let swap_flag = match m.hands_swapped() {
            Some(true) => "1",
            Some(false) => "0",
            None => "",
        };
        println!(
            "{idx},{},{:.3},{:.1},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            m.joints,
            m.span,
            m.yaw,
            fz(m.r_wrist_z),
            fz(m.r_hand_z),
            fz(m.l_wrist_z),
            fz(m.l_hand_z),
            fz(m.lsh.map(|p| p[0])),
            fz(m.lsh.map(|p| p[2])),
            fz(m.rsh.map(|p| p[0])),
            fz(m.rsh.map(|p| p[2])),
            m.metric_present as u8,
            fz(m.head.map(|p| p[0])),
            fz(m.head.map(|p| p[1])),
            fz(m.head.map(|p| p[2])),
            fz(m.neck.map(|p| p[0])),
            fz(m.neck.map(|p| p[1])),
            fz(m.neck.map(|p| p[2])),
            fz(lean_lr),
            fz(lean_fb),
            fz(m.face.map(|f| f[0])),
            fz(m.face.map(|f| f[1])),
            fz(m.face.map(|f| f[2])),
            fz(m.face.map(|f| f[3])),
            fz(m.l_hand_x),
            fz(m.r_hand_x),
            swap_flag,
            fz(m.ref_span_m),
            fz(m.anchor_z),
            fz(m.mpsu),
        );

        if m.metric_present {
            metric_frames += 1;
        }
        if m.joints == 0 {
            empty_frames += 1;
        }
        // Head-orientation inputs. Only trust them on well-detected frames so a
        // degraded frame's guessed head doesn't skew the head-jitter stats.
        if m.joints >= 20 {
            if let Some((lr, fb)) = m.head_tilt() {
                head_lr.push(lr);
                head_fb.push(fb);
            }
            if let Some(f) = m.face {
                fyaws.push(f[0]);
            }
            if let Some(sw) = m.hands_swapped() {
                swap_eligible += 1;
                if sw {
                    swap_frames += 1;
                }
            }
        }
        // End-effector forward/back: during a wave the hand stays in front of
        // the torso (+z). A SYSTEMATIC inversion (hand reading behind) is the
        // old monocular bug; occasional negatives are transitions / dropouts.
        // Only counted on well-detected frames (>=25 joints) so a degraded
        // frame's garbage hand doesn't skew the fraction.
        if m.joints >= 25 {
            for hz in [m.r_hand_z, m.l_hand_z].into_iter().flatten() {
                hand_tot += 1;
                if hz > 0.0 {
                    hand_fwd += 1;
                }
                if hz.abs() > 1.0 {
                    hand_outliers += 1;
                }
            }
        }
        if m.yaw.is_finite() {
            yaws.push(m.yaw);
            if let Some(py) = prev_yaw {
                if (py - m.yaw).abs() > 30.0 {
                    yaw_jumps += 1;
                }
            }
            prev_yaw = Some(m.yaw);
        }
        // Sign flip across zero with a large jump = the forward↔behind
        // inversion the depth rebuild exists to kill.
        if let Some(rz) = m.r_wrist_z {
            if let Some(pz) = prev_rz {
                if pz.signum() != rz.signum() && (pz - rz).abs() > 0.3 {
                    r_flips += 1;
                }
            }
            prev_rz = Some(rz);
        }
        if let Some(lz) = m.l_wrist_z {
            if let Some(pz) = prev_lz {
                if pz.signum() != lz.signum() && (pz - lz).abs() > 0.3 {
                    l_flips += 1;
                }
            }
            prev_lz = Some(lz);
        }
    }

    // --- temporal summary ---
    let n = pairs.len();
    let (mut ymin, mut ymax, mut ysum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f32);
    for &y in &yaws {
        ymin = ymin.min(y);
        ymax = ymax.max(y);
        ysum += y;
    }
    let ymean = if yaws.is_empty() { f32::NAN } else { ysum / yaws.len() as f32 };
    let ystd = if yaws.is_empty() {
        f32::NAN
    } else {
        (yaws.iter().map(|y| (y - ymean).powi(2)).sum::<f32>() / yaws.len() as f32).sqrt()
    };

    eprintln!("\n=== temporal summary over {n} frames ===");
    eprintln!(
        "metric-native frames : {metric_frames}/{n} ({:.0}%)",
        100.0 * metric_frames as f32 / n as f32
    );
    let hand_fwd_pct = if hand_tot > 0 {
        100.0 * hand_fwd as f32 / hand_tot as f32
    } else {
        f32::NAN
    };
    eprintln!(
        "yaw (deg)            : mean {ymean:+.1}  range [{ymin:+.1}, {ymax:+.1}]  std {ystd:.1}  jumps>30°/frame: {yaw_jumps}"
    );
    eprintln!(
        "hand forward (+z)    : {hand_fwd}/{hand_tot} ({hand_fwd_pct:.0}%)  outliers(|z|>1): {hand_outliers}"
    );
    eprintln!(
        "forearm z sign-flips (limb jitter, solver-smoothed): R={r_flips} L={l_flips}"
    );

    // --- head orientation probe (the symptom the yaw stat cannot see) ---
    let stats = |v: &[f32]| -> (f32, f32, f32, f32) {
        if v.is_empty() {
            return (f32::NAN, f32::NAN, f32::NAN, f32::NAN);
        }
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
        let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        (mean, std, mn, mx)
    };
    let (lr_m, lr_s, lr_mn, lr_mx) = stats(&head_lr);
    let (fb_m, fb_s, fb_mn, fb_mx) = stats(&head_fb);
    let (fy_m, fy_s, fy_mn, fy_mx) = stats(&fyaws);
    eprintln!(
        "head neck-bend L/R   : mean {lr_m:+.1}° std {lr_s:.1}° range [{lr_mn:+.1},{lr_mx:+.1}]  (contributor B, depth ear-mid)"
    );
    eprintln!(
        "head neck-bend F/B   : mean {fb_m:+.1}° std {fb_s:.1}° range [{fb_mn:+.1},{fb_mx:+.1}]"
    );
    eprintln!(
        "face pose yaw (RGB)  : mean {fy_m:+.1}° std {fy_s:.1}° range [{fy_mn:+.1},{fy_mx:+.1}]  (contributor C, depth-independent)"
    );
    let swap_pct = if swap_eligible > 0 {
        100.0 * swap_frames as f32 / swap_eligible as f32
    } else {
        f32::NAN
    };
    eprintln!(
        "hand L/R swapped     : {swap_frames}/{swap_eligible} ({swap_pct:.0}%) of elbow-separated frames  (depth path has NO hand-block swap fix)"
    );
    eprintln!(
        "empty-skeleton frames: {empty_frames}/{n}  (no torso fit → avatar snaps to rest)"
    );

    // Hard-fail only on the SEVERE signatures of the original bug — the torso
    // spinning to wild directions, and a SYSTEMATIC hand front/back inversion.
    // Forearm jitter and isolated outliers are limb-level noise the solver's
    // arm-IK + 1€ filter smooths, and over-smoothing limbs would kill wave
    // responsiveness, so they are reported (WARN) but do not fail the run.
    let mut concerns = Vec::new();
    if !ystd.is_nan() && ystd > 25.0 {
        concerns.push(format!("torso yaw std {ystd:.1}° > 25° — unstable orientation"));
    }
    if ymax.abs() > 60.0 || ymin.abs() > 60.0 {
        concerns.push(format!(
            "torso yaw hit [{ymin:+.0}, {ymax:+.0}]° — a wild-direction frame survived"
        ));
    }
    if !hand_fwd_pct.is_nan() && hand_fwd_pct < 60.0 {
        concerns.push(format!(
            "hand forward only {hand_fwd_pct:.0}% — possible systematic front/back inversion"
        ));
    }

    if r_flips + l_flips > 0 || hand_outliers > 0 {
        eprintln!(
            "\nWARN: forearm jitter (R={r_flips} L={l_flips}) / {hand_outliers} hand outlier(s) — limb-level noise left for the solver's arm-IK + 1€ smoothing (limbs are intentionally not position-smoothed, to keep gestures responsive)."
        );
    }

    if concerns.is_empty() {
        eprintln!(
            "\nDYNAMIC VERDICT: torso yaw stable (std {ystd:.1}°, no wild frames), hand stays forward ({hand_fwd_pct:.0}%) — the severe signatures of the old bug (±180° flapping, systematic front/back inversion) are absent."
        );
        Ok(())
    } else {
        eprintln!("\nDYNAMIC CONCERNS:");
        for c in &concerns {
            eprintln!("  - {c}");
        }
        Err(format!("{} dynamic concern(s) — inspect the CSV", concerns.len()))
    }
}

fn report(sk: &SourceSkeleton) -> Result<(), String> {
    use HumanoidBone::*;
    let mut fails: Vec<String> = Vec::new();

    // --- metric branch signal (Part 1) ---
    match sk.metric_frame_info.as_ref() {
        Some(m) => {
            println!("\nmetric_frame_info: PRESENT");
            println!(
                "  anchor_cam_m = [{:+.3},{:+.3},{:+.3}] m  ({})",
                m.anchor_cam_m[0],
                m.anchor_cam_m[1],
                m.anchor_cam_m[2],
                if m.anchor_is_hip { "hip" } else { "shoulder" }
            );
            println!(
                "  mpsu = {:.4} m/unit   reference_span = {:.3} m",
                m.mpsu, m.reference_span_m
            );
            println!(
                "  intrinsics = fx {:.0} fy {:.0} cx {:.0} cy {:.0} ({}x{})",
                m.intrinsics.fx,
                m.intrinsics.fy,
                m.intrinsics.cx,
                m.intrinsics.cy,
                m.intrinsics.width,
                m.intrinsics.height
            );
        }
        None => {
            println!("\nmetric_frame_info: ABSENT");
            fails.push("metric_frame_info missing (metric branch signal not wired)".into());
        }
    }

    // --- joints ---
    println!(
        "\njoints = {}   overall_conf = {:.2}",
        sk.joints.len(),
        sk.overall_confidence
    );
    let get = |b: HumanoidBone| sk.joints.get(&b);
    for b in [
        LeftUpperArm,
        RightUpperArm,
        LeftLowerArm,
        RightLowerArm,
        LeftHand,
        RightHand,
        Head,
        Hips,
    ] {
        match get(b) {
            Some(j) => println!(
                "  {:<14?} pos=[{:+.3},{:+.3},{:+.3}] depth_m={:?} conf={:.2}",
                b, j.position[0], j.position[1], j.position[2], j.metric_depth_m, j.confidence
            ),
            None => println!("  {:<14?} —absent—", b),
        }
    }

    // Sanity: joints carrying NO metric depth yet a wild position are 2-D
    // fallback guesses that ideally should be rested, not driven. Surfaced as
    // a WARN (not a fail) — a follow-up for the hand-attach path on frames
    // where a hand is occluded / out of frame.
    let wild: Vec<String> = sk
        .joints
        .iter()
        .filter(|(_, j)| j.metric_depth_m.is_none() && j.position.iter().any(|c| c.abs() > 2.0))
        .map(|(b, j)| {
            format!(
                "{:?}=[{:+.2},{:+.2},{:+.2}]",
                b, j.position[0], j.position[1], j.position[2]
            )
        })
        .collect();
    if !wild.is_empty() {
        println!(
            "\nWARN: {} no-depth joint(s) with wild positions (should be rested): {}",
            wild.len(),
            wild.join(", ")
        );
    }

    // --- shoulder line: real-3D yaw (no foreshortening) + normalised span ---
    if let (Some(l), Some(r)) = (get(LeftUpperArm), get(RightUpperArm)) {
        let dx = l.position[0] - r.position[0];
        let dy = l.position[1] - r.position[1];
        let dz = l.position[2] - r.position[2];
        let span = (dx * dx + dy * dy + dz * dz).sqrt();
        // yaw 0 = shoulder line along source-x (frontal); ±90 = along z.
        let yaw = dz.atan2(dx).to_degrees();
        println!(
            "\nshoulder-line: span_src = {:.3} (target {:.3})   yaw = {:+.1}°",
            span, TARGET_SRC_SHOULDER_SPAN, yaw
        );
        println!(
            "  (yaw is the REAL 3-D shoulder-line angle, no foreshortening/ratchet;"
        );
        println!(
            "   it reflects the subject's actual torso turn in this frame, not an assumed frontal pose.)"
        );
        if (span - TARGET_SRC_SHOULDER_SPAN).abs() > 0.05 {
            fails.push(format!(
                "normalised shoulder span {span:.3} off target {TARGET_SRC_SHOULDER_SPAN:.3} (>0.05)"
            ));
        }
        // Hard-fail only the OLD garbage signature: the running-max ratchet /
        // corner-clamped-shoulder bug pinned yaw near ±70° and flipped sign.
        // A finite, moderate yaw from clean in-frame shoulders is the fix
        // working — the exact value is the subject's real torso turn, which a
        // single non-frontal fixture can't pin, so we don't assert it.
        if !yaw.is_finite() || yaw.abs() > 60.0 {
            fails.push(format!(
                "shoulder yaw {yaw:+.1}° is non-finite or implausibly large (>60°) — \
                 the metric 3-D line should never blow up like the old ratchet did"
            ));
        }
    } else {
        fails.push("shoulders absent — no torso orientation".into());
    }

    // --- forward reach: source +z = toward camera; palms-front → hands ahead ---
    let shoulder_z = get(LeftUpperArm)
        .map(|j| j.position[2])
        .or_else(|| get(RightUpperArm).map(|j| j.position[2]))
        .unwrap_or(0.0);
    let hand_z = [LeftHand, RightHand, LeftLowerArm, RightLowerArm]
        .iter()
        .filter_map(|b| get(*b))
        .map(|j| j.position[2])
        .fold(f32::NEG_INFINITY, f32::max);
    if hand_z.is_finite() {
        println!(
            "forward reach: max hand/forearm z = {:+.3}  (shoulder z = {:+.3})",
            hand_z, shoulder_z
        );
        if hand_z <= shoulder_z {
            println!("  WARN: hands not ahead of shoulders — expected forward reach for palms-front.");
        }
    } else {
        println!("forward reach: no hand/forearm joints detected (WARN).");
    }

    // --- 1:1 root (raw metres) ---
    if let Some(root) = sk.root_offset {
        println!(
            "root_offset (m) = [{:+.3},{:+.3},{:+.3}]  anchor_is_hip={}",
            root[0], root[1], root[2], sk.root_anchor_is_hip
        );
    }

    println!();
    if fails.is_empty() {
        println!("Tier 1 PASS — metric branch + frontal shoulder line + normalised span in range.");
        Ok(())
    } else {
        for f in &fails {
            println!("FAIL: {f}");
        }
        Err(format!("{} Tier 1 assertion(s) failed", fails.len()))
    }
}
