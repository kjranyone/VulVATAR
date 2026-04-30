//! Experimental CIGPose + metric-depth tracking pipeline.
//!
//! The core idea is to keep CIGPose responsible for stable 2D whole-body
//! keypoints, then use UniDepthV2 / MoGe / Depth Pro style metric depth or a
//! point cloud to create 3D joint candidates by projection. Bone constraints
//! and temporal filtering should operate after this module has produced those
//! metric candidates.

use std::path::{Path, PathBuf};

use super::provider::{
    PoseProvider, CIGPOSE_MODEL_ENV, LEGACY_DEPTH_MODEL_ENV, METRIC_DEPTH_KIND_ENV,
    METRIC_DEPTH_MODEL_ENV,
};
use super::PoseEstimate;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricDepthKind {
    UniDepthV2,
    MoGe,
    DepthPro,
    Custom,
}

impl MetricDepthKind {
    pub fn from_env() -> Self {
        let raw = std::env::var(METRIC_DEPTH_KIND_ENV).unwrap_or_else(|_| "unidepth-v2".into());
        Self::parse(&raw).unwrap_or(Self::Custom)
    }

    fn parse(value: &str) -> Option<Self> {
        let normalized = value.trim().to_ascii_lowercase().replace('_', "-");
        match normalized.as_str() {
            "unidepth" | "unidepth-v2" | "unidepthv2" => Some(Self::UniDepthV2),
            "moge" => Some(Self::MoGe),
            "depth-pro" | "depthpro" => Some(Self::DepthPro),
            "custom" => Some(Self::Custom),
            _ => None,
        }
    }

    fn default_model_name(self) -> &'static str {
        match self {
            Self::UniDepthV2 => "unidepth_v2.onnx",
            Self::MoGe => "moge.onnx",
            Self::DepthPro => "depth_pro.onnx",
            Self::Custom => "metric_depth.onnx",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::UniDepthV2 => "UniDepthV2",
            Self::MoGe => "MoGe",
            Self::DepthPro => "Depth Pro",
            Self::Custom => "MetricDepth",
        }
    }
}

/// Pinhole camera intrinsics in pixel units.
#[derive(Clone, Copy, Debug)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

impl CameraIntrinsics {
    pub fn from_vertical_fov(width: u32, height: u32, fov_y_radians: f32) -> Option<Self> {
        if width == 0 || height == 0 || !fov_y_radians.is_finite() || fov_y_radians <= 0.0 {
            return None;
        }
        let fy = (height as f32 * 0.5) / (fov_y_radians * 0.5).tan();
        if !fy.is_finite() || fy <= 0.0 {
            return None;
        }
        Some(Self {
            fx: fy,
            fy,
            cx: (width as f32 - 1.0) * 0.5,
            cy: (height as f32 - 1.0) * 0.5,
        })
    }

    pub fn back_project(&self, x_px: f32, y_px: f32, depth_m: f32) -> Option<[f32; 3]> {
        if !depth_m.is_finite() || depth_m <= 0.0 || self.fx <= 0.0 || self.fy <= 0.0 {
            return None;
        }
        Some([
            (x_px - self.cx) / self.fx * depth_m,
            (y_px - self.cy) / self.fy * depth_m,
            depth_m,
        ])
    }
}

#[derive(Clone, Copy, Debug)]
pub struct JointObservation2d {
    pub x_px: f32,
    pub y_px: f32,
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Joint3dCandidate {
    pub position_m: [f32; 3],
    pub confidence: f32,
}

/// Metric depth output for one frame. `depth_m` is row-major metres. When a
/// model emits a point cloud directly, `point_cloud_m` should contain the same
/// number of pixels and is preferred over depth back-projection.
pub struct MetricDepthFrame<'a> {
    pub width: u32,
    pub height: u32,
    pub depth_m: &'a [f32],
    pub point_cloud_m: Option<&'a [[f32; 3]]>,
}

/// Project a 2D joint into a metric 3D candidate using a small local window.
///
/// The local median makes this robust against single-pixel depth spikes around
/// wrists/fingers while preserving the metric coordinate frame for shoulders,
/// hips, and torso yaw.
pub fn project_keypoint_to_3d(
    frame: &MetricDepthFrame<'_>,
    intrinsics: CameraIntrinsics,
    keypoint: JointObservation2d,
    sample_radius_px: u32,
) -> Option<Joint3dCandidate> {
    if keypoint.confidence <= 0.0 || !keypoint.x_px.is_finite() || !keypoint.y_px.is_finite() {
        return None;
    }
    let x = keypoint.x_px.round() as i32;
    let y = keypoint.y_px.round() as i32;
    let radius = sample_radius_px as i32;
    let width = frame.width as i32;
    let height = frame.height as i32;
    if width <= 0 || height <= 0 || x < 0 || y < 0 || x >= width || y >= height {
        return None;
    }
    let expected = frame.width as usize * frame.height as usize;
    if frame.depth_m.len() < expected {
        return None;
    }
    if let Some(points) = frame.point_cloud_m {
        if points.len() >= expected {
            if let Some(position_m) =
                sample_point_cloud(points, frame.width, frame.height, x, y, radius)
            {
                return Some(Joint3dCandidate {
                    position_m,
                    confidence: keypoint.confidence.clamp(0.0, 1.0),
                });
            }
        }
    }

    let depth_m = sample_depth(frame.depth_m, frame.width, frame.height, x, y, radius)?;
    let position_m = intrinsics.back_project(keypoint.x_px, keypoint.y_px, depth_m)?;
    Some(Joint3dCandidate {
        position_m,
        confidence: keypoint.confidence.clamp(0.0, 1.0),
    })
}

fn sample_depth(
    depth_m: &[f32],
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    radius: i32,
) -> Option<f32> {
    let mut values = Vec::new();
    for yy in (y - radius).max(0)..=(y + radius).min(height as i32 - 1) {
        for xx in (x - radius).max(0)..=(x + radius).min(width as i32 - 1) {
            let idx = yy as usize * width as usize + xx as usize;
            let depth = depth_m[idx];
            if depth.is_finite() && depth > 0.0 {
                values.push(depth);
            }
        }
    }
    median(values)
}

fn sample_point_cloud(
    points: &[[f32; 3]],
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    radius: i32,
) -> Option<[f32; 3]> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    for yy in (y - radius).max(0)..=(y + radius).min(height as i32 - 1) {
        for xx in (x - radius).max(0)..=(x + radius).min(width as i32 - 1) {
            let idx = yy as usize * width as usize + xx as usize;
            let [px, py, pz] = points[idx];
            if px.is_finite() && py.is_finite() && pz.is_finite() && pz > 0.0 {
                xs.push(px);
                ys.push(py);
                zs.push(pz);
            }
        }
    }
    Some([median(xs)?, median(ys)?, median(zs)?])
}

fn median(mut values: Vec<f32>) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    Some(values[values.len() / 2])
}

/// Tunables for converting 2D keypoints + metric depth into source-space 3D.
#[derive(Clone, Debug)]
pub struct CigposeMetricDepthParams {
    /// Half-size of the square depth/point-cloud sampling window.
    pub sample_radius_px: u32,
    /// Candidate blend per frame after bone constraints.
    pub temporal_blend: f32,
    /// Maximum accepted per-frame joint-depth jump in metres before filtering.
    pub max_depth_jump_m: f32,
}

impl Default for CigposeMetricDepthParams {
    fn default() -> Self {
        Self {
            sample_radius_px: 4,
            temporal_blend: 0.35,
            max_depth_jump_m: 0.35,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CigposeMetricDepthConfig {
    pub cigpose_model: PathBuf,
    pub metric_depth_model: PathBuf,
    pub metric_depth_kind: MetricDepthKind,
    pub params: CigposeMetricDepthParams,
}

impl CigposeMetricDepthConfig {
    pub fn from_models_dir(models_dir: impl AsRef<Path>) -> Self {
        let models_dir = models_dir.as_ref();
        let metric_depth_kind = MetricDepthKind::from_env();
        let cigpose_model = std::env::var_os(CIGPOSE_MODEL_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| models_dir.join("cigpose.onnx"));
        let metric_depth_model = std::env::var_os(METRIC_DEPTH_MODEL_ENV)
            .or_else(|| std::env::var_os(LEGACY_DEPTH_MODEL_ENV))
            .map(PathBuf::from)
            .unwrap_or_else(|| models_dir.join(metric_depth_kind.default_model_name()));
        Self {
            cigpose_model,
            metric_depth_model,
            metric_depth_kind,
            params: CigposeMetricDepthParams::default(),
        }
    }

    pub fn validate_models(&self) -> Result<(), String> {
        let mut missing = Vec::new();
        if !self.cigpose_model.is_file() {
            missing.push(format!(
                "CIGPose model not found at {}",
                self.cigpose_model.display()
            ));
        }
        if !self.metric_depth_model.is_file() {
            missing.push(format!(
                "{} metric depth model not found at {}",
                self.metric_depth_kind.label(),
                self.metric_depth_model.display()
            ));
        }
        if missing.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "{}. Run the dev.ps1 experimental pose-provider setup, provide ONNX exports via {} and {}, or place cigpose.onnx and the metric depth ONNX in models/.",
                missing.join("; "),
                CIGPOSE_MODEL_ENV,
                METRIC_DEPTH_MODEL_ENV
            ))
        }
    }
}

/// Provider implementation for the CIGPose 2D + metric-depth path.
///
/// It is intentionally not a silent fallback: until the CIGPose and metric
/// depth ONNX decoders are wired, selecting this provider returns a startup
/// error instead of producing synthetic or RTMW3D output under the new label.
pub struct CigposeMetricDepthProvider {
    config: CigposeMetricDepthConfig,
}

impl CigposeMetricDepthProvider {
    pub(super) fn from_models_dir(models_dir: impl AsRef<Path>) -> Result<Self, String> {
        let config = CigposeMetricDepthConfig::from_models_dir(models_dir);
        config.validate_models()?;
        Err(
            "CIGPose+MetricDepth provider is registered, but its ONNX runtime decoders are not implemented yet"
                .to_string(),
        )
    }
}

impl PoseProvider for CigposeMetricDepthProvider {
    fn label(&self) -> String {
        format!("CIGPose+{}", self.config.metric_depth_kind.label())
    }

    fn estimate_pose(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> PoseEstimate {
        super::pose_estimation::estimate_pose(rgb_data, width, height, frame_index)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        project_keypoint_to_3d, CameraIntrinsics, JointObservation2d, MetricDepthFrame,
        MetricDepthKind,
    };

    #[test]
    fn parses_metric_depth_kind_aliases() {
        assert_eq!(
            MetricDepthKind::parse("unidepth_v2"),
            Some(MetricDepthKind::UniDepthV2)
        );
        assert_eq!(MetricDepthKind::parse("moge"), Some(MetricDepthKind::MoGe));
        assert_eq!(
            MetricDepthKind::parse("depth-pro"),
            Some(MetricDepthKind::DepthPro)
        );
    }

    #[test]
    fn projects_keypoint_with_metric_depth() {
        let depth = vec![2.0; 9];
        let frame = MetricDepthFrame {
            width: 3,
            height: 3,
            depth_m: &depth,
            point_cloud_m: None,
        };
        let intrinsics = CameraIntrinsics {
            fx: 2.0,
            fy: 2.0,
            cx: 1.0,
            cy: 1.0,
        };
        let joint = project_keypoint_to_3d(
            &frame,
            intrinsics,
            JointObservation2d {
                x_px: 2.0,
                y_px: 1.0,
                confidence: 0.8,
            },
            0,
        )
        .unwrap();
        assert_eq!(joint.position_m, [1.0, 0.0, 2.0]);
        assert_eq!(joint.confidence, 0.8);
    }

    #[test]
    fn prefers_point_cloud_when_available() {
        let depth = vec![2.0; 4];
        let points = vec![
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [7.0, 8.0, 9.0],
        ];
        let frame = MetricDepthFrame {
            width: 2,
            height: 2,
            depth_m: &depth,
            point_cloud_m: Some(&points),
        };
        let intrinsics = CameraIntrinsics {
            fx: 1.0,
            fy: 1.0,
            cx: 0.0,
            cy: 0.0,
        };
        let joint = project_keypoint_to_3d(
            &frame,
            intrinsics,
            JointObservation2d {
                x_px: 1.0,
                y_px: 1.0,
                confidence: 1.0,
            },
            0,
        )
        .unwrap();
        assert_eq!(joint.position_m, [7.0, 8.0, 9.0]);
    }
}
