//! Per-image dump of `rtmw3d-with-depth` (or any other provider)
//! key signals — face pose, shoulder/wrist positions, hand
//! orientations, calibration anchor — into JSON Lines for batch
//! analysis of the depth pipeline output across a validation set.
//!
//! Usage:
//!   cargo run --bin analyze_depth_provider --features inference -- \
//!       <provider> <image_or_dir> [--out <file.jsonl>]
//!     <provider> = rtmw3d | rtmw3d-with-depth | cigpose-metric-depth
//!
//! Defaults to rtmw3d-with-depth + the basic_pose_samples directory if
//! no args are given. Output is line-delimited JSON for grep / jq.

use std::fs;
use std::path::{Path, PathBuf};

use vulvatar_lib::asset::HumanoidBone;
use vulvatar_lib::tracking::provider::{create_pose_provider, PoseProviderKind};
use vulvatar_lib::tracking::source_skeleton::{HandOrientation, SourceJoint, SourceSkeleton};

fn main() -> Result<(), String> {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let provider_arg = args
        .next()
        .unwrap_or_else(|| "rtmw3d-with-depth".to_string());
    let target = args
        .next()
        .unwrap_or_else(|| "validation_images/basic_pose_samples/photorealistic".to_string());
    let mut out_path: Option<PathBuf> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out_path = args.next().map(PathBuf::from),
            other => return Err(format!("unknown arg '{other}'")),
        }
    }

    let kind = match provider_arg.as_str() {
        "rtmw3d" => PoseProviderKind::Rtmw3d,
        "rtmw3d-with-depth" => PoseProviderKind::Rtmw3dWithDepth,
        "cigpose-metric-depth" => PoseProviderKind::CigposeMetricDepth,
        other => return Err(format!("unknown provider '{other}'")),
    };

    let images = collect_images(Path::new(&target))?;
    eprintln!(
        "analyze_depth_provider: provider={:?}, images={}",
        kind,
        images.len()
    );

    let mut provider =
        create_pose_provider(kind, "models").map_err(|e| format!("provider load: {e}"))?;
    eprintln!("provider label: {}", provider.label());
    for w in provider.take_load_warnings() {
        eprintln!("warning: {w}");
    }

    let mut out_lines: Vec<String> = Vec::with_capacity(images.len());
    for (i, image_path) in images.iter().enumerate() {
        let img = match image::open(image_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("[{i:>3}] open failed: {} — skipping", e);
                continue;
            }
        };
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());

        // Two warm-up frames so the provider's async workers + DAv2
        // sticky outbox have steady-state values. Then one timed
        // measurement frame whose output we record.
        let _ = provider.estimate_pose(rgb.as_raw(), w, h, 0);
        let _ = provider.estimate_pose(rgb.as_raw(), w, h, 1);
        let est = provider.estimate_pose(rgb.as_raw(), w, h, 2);

        let row = build_row(image_path, w, h, &est.skeleton);
        let line = serde_json_line(&row);
        let root_str = est
            .skeleton
            .root_offset
            .map(|r| format!("({:.2},{:.2},{:.2})", r[0], r[1], r[2]))
            .unwrap_or_else(|| "none".to_string());
        eprintln!(
            "[{i:>3}] {} → face={} hand_l={} hand_r={} shoulders={} root={}",
            image_path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
            row.face_summary(),
            yes_no(est.skeleton.left_hand_orientation.is_some()),
            yes_no(est.skeleton.right_hand_orientation.is_some()),
            row.shoulder_summary(),
            root_str,
        );
        out_lines.push(line);
    }

    let output = out_lines.join("\n") + "\n";
    if let Some(path) = out_path {
        fs::write(&path, &output).map_err(|e| format!("write {}: {e}", path.display()))?;
        eprintln!("wrote {}", path.display());
    } else {
        print!("{}", output);
    }
    Ok(())
}

fn collect_images(target: &Path) -> Result<Vec<PathBuf>, String> {
    if target.is_file() {
        return Ok(vec![target.to_path_buf()]);
    }
    if !target.is_dir() {
        return Err(format!("target not found: {}", target.display()));
    }
    let mut out: Vec<PathBuf> = Vec::new();
    fn walk(dir: &Path, acc: &mut Vec<PathBuf>) -> Result<(), String> {
        for entry in
            fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?
        {
            let entry = entry.map_err(|e| format!("entry: {e}"))?;
            let path = entry.path();
            if path.is_dir() {
                walk(&path, acc)?;
            } else if matches!(
                path.extension().and_then(|s| s.to_str()),
                Some("png" | "jpg" | "jpeg")
            ) {
                acc.push(path);
            }
        }
        Ok(())
    }
    walk(target, &mut out)?;
    out.sort();
    Ok(out)
}

#[derive(Default)]
struct Row {
    image: String,
    width: u32,
    height: u32,
    overall_confidence: f32,
    face: Option<FaceRow>,
    face_mesh_confidence: Option<f32>,
    expressions_count: usize,
    left_shoulder: Option<JointRow>,
    right_shoulder: Option<JointRow>,
    left_upper_arm: Option<JointRow>,
    right_upper_arm: Option<JointRow>,
    left_hand: Option<JointRow>,
    right_hand: Option<JointRow>,
    left_hand_orient: Option<OrientRow>,
    right_hand_orient: Option<OrientRow>,
    hips: Option<JointRow>,
    joint_count: usize,
}

struct FaceRow {
    yaw_deg: f32,
    pitch_deg: f32,
    roll_deg: f32,
    confidence: f32,
}

struct JointRow {
    x: f32,
    y: f32,
    z: f32,
    confidence: f32,
}

struct OrientRow {
    forward: [f32; 3],
    up: [f32; 3],
    confidence: f32,
}

impl Row {
    fn face_summary(&self) -> String {
        match &self.face {
            Some(f) => format!(
                "yaw={:>5.1} pit={:>5.1} rol={:>5.1} c={:.2}",
                f.yaw_deg, f.pitch_deg, f.roll_deg, f.confidence
            ),
            None => "none".to_string(),
        }
    }
    fn shoulder_summary(&self) -> String {
        match (&self.left_shoulder, &self.right_shoulder) {
            (Some(l), Some(r)) => format!(
                "L({:.2},{:.2},{:.2}) R({:.2},{:.2},{:.2})",
                l.x, l.y, l.z, r.x, r.y, r.z
            ),
            _ => "incomplete".to_string(),
        }
    }
}

fn build_row(image_path: &Path, width: u32, height: u32, sk: &SourceSkeleton) -> Row {
    let face = sk.face.as_ref().map(|f| FaceRow {
        yaw_deg: f.yaw.to_degrees(),
        pitch_deg: f.pitch.to_degrees(),
        roll_deg: f.roll.to_degrees(),
        confidence: f.confidence,
    });
    let joint = |bone: HumanoidBone| -> Option<JointRow> {
        sk.joints.get(&bone).map(|j: &SourceJoint| JointRow {
            x: j.position[0],
            y: j.position[1],
            z: j.position[2],
            confidence: j.confidence,
        })
    };
    let orient = |o: Option<&HandOrientation>| -> Option<OrientRow> {
        o.map(|o| OrientRow {
            forward: o.forward,
            up: o.up,
            confidence: o.confidence,
        })
    };
    Row {
        image: image_path
            .strip_prefix(std::env::current_dir().unwrap_or_default())
            .unwrap_or(image_path)
            .display()
            .to_string()
            .replace('\\', "/"),
        width,
        height,
        overall_confidence: sk.overall_confidence,
        face,
        face_mesh_confidence: sk.face_mesh_confidence,
        expressions_count: sk.expressions.len(),
        left_shoulder: joint(HumanoidBone::LeftShoulder),
        right_shoulder: joint(HumanoidBone::RightShoulder),
        left_upper_arm: joint(HumanoidBone::LeftUpperArm),
        right_upper_arm: joint(HumanoidBone::RightUpperArm),
        left_hand: joint(HumanoidBone::LeftHand),
        right_hand: joint(HumanoidBone::RightHand),
        left_hand_orient: orient(sk.left_hand_orientation.as_ref()),
        right_hand_orient: orient(sk.right_hand_orientation.as_ref()),
        hips: joint(HumanoidBone::Hips),
        joint_count: sk.joints.len(),
    }
}

fn yes_no(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        " no"
    }
}

// Manual JSON serialization to avoid pulling in serde_json across the
// whole crate just for a diagnostic. NaN / inf rendered as null so
// the line stays valid JSON.
fn serde_json_line(r: &Row) -> String {
    let mut s = String::new();
    s.push('{');
    push_str_field(&mut s, "image", &r.image, true);
    push_num_field(&mut s, "width", r.width as f32, false);
    push_num_field(&mut s, "height", r.height as f32, false);
    push_num_field(&mut s, "overall_confidence", r.overall_confidence, false);
    push_num_field(&mut s, "joint_count", r.joint_count as f32, false);
    push_num_field(&mut s, "expressions_count", r.expressions_count as f32, false);
    push_opt_num_field(
        &mut s,
        "face_mesh_confidence",
        r.face_mesh_confidence,
        false,
    );
    push_face(&mut s, "face", r.face.as_ref());
    push_joint(&mut s, "left_shoulder", r.left_shoulder.as_ref());
    push_joint(&mut s, "right_shoulder", r.right_shoulder.as_ref());
    push_joint(&mut s, "left_upper_arm", r.left_upper_arm.as_ref());
    push_joint(&mut s, "right_upper_arm", r.right_upper_arm.as_ref());
    push_joint(&mut s, "left_hand", r.left_hand.as_ref());
    push_joint(&mut s, "right_hand", r.right_hand.as_ref());
    push_orient(&mut s, "left_hand_orient", r.left_hand_orient.as_ref());
    push_orient(&mut s, "right_hand_orient", r.right_hand_orient.as_ref());
    push_joint(&mut s, "hips", r.hips.as_ref());
    s.push('}');
    s
}

fn push_str_field(s: &mut String, name: &str, val: &str, is_first: bool) {
    if !is_first {
        s.push(',');
    }
    s.push('"');
    s.push_str(name);
    s.push_str("\":\"");
    for c in val.chars() {
        if c == '"' || c == '\\' {
            s.push('\\');
        }
        s.push(c);
    }
    s.push('"');
}

fn push_num_field(s: &mut String, name: &str, val: f32, is_first: bool) {
    if !is_first {
        s.push(',');
    }
    s.push('"');
    s.push_str(name);
    s.push_str("\":");
    if val.is_finite() {
        s.push_str(&format!("{:.4}", val));
    } else {
        s.push_str("null");
    }
}

fn push_opt_num_field(s: &mut String, name: &str, val: Option<f32>, is_first: bool) {
    if !is_first {
        s.push(',');
    }
    s.push('"');
    s.push_str(name);
    s.push_str("\":");
    match val {
        Some(v) if v.is_finite() => s.push_str(&format!("{:.4}", v)),
        _ => s.push_str("null"),
    }
}

fn push_face(s: &mut String, name: &str, f: Option<&FaceRow>) {
    s.push(',');
    s.push('"');
    s.push_str(name);
    s.push_str("\":");
    match f {
        Some(f) => s.push_str(&format!(
            "{{\"yaw_deg\":{:.2},\"pitch_deg\":{:.2},\"roll_deg\":{:.2},\"confidence\":{:.3}}}",
            f.yaw_deg, f.pitch_deg, f.roll_deg, f.confidence
        )),
        None => s.push_str("null"),
    }
}

fn push_joint(s: &mut String, name: &str, j: Option<&JointRow>) {
    s.push(',');
    s.push('"');
    s.push_str(name);
    s.push_str("\":");
    match j {
        Some(j) => s.push_str(&format!(
            "{{\"x\":{:.4},\"y\":{:.4},\"z\":{:.4},\"confidence\":{:.3}}}",
            j.x, j.y, j.z, j.confidence
        )),
        None => s.push_str("null"),
    }
}

fn push_orient(s: &mut String, name: &str, o: Option<&OrientRow>) {
    s.push(',');
    s.push('"');
    s.push_str(name);
    s.push_str("\":");
    match o {
        Some(o) => s.push_str(&format!(
            "{{\"forward\":[{:.3},{:.3},{:.3}],\"up\":[{:.3},{:.3},{:.3}],\"confidence\":{:.3}}}",
            o.forward[0], o.forward[1], o.forward[2], o.up[0], o.up[1], o.up[2], o.confidence
        )),
        None => s.push_str("null"),
    }
}
