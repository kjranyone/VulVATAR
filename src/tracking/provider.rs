//! Pose provider boundary for swapping tracking pipelines.
//!
//! `TrackingWorker` talks to this module instead of binding directly to a
//! specific model. That keeps RTMW3D as the default path while making the
//! experimental CIGPose + metric-depth pipeline selectable and easy to
//! compare or roll back.

use std::path::Path;

use super::cigpose_metric_depth::CigposeMetricDepthProvider;
use super::rtmw3d_with_depth::Rtmw3dWithDepthProvider;
use super::PoseEstimate;

/// Selects which tracking pipeline runs. Set to one of:
///
/// * `rtmw3d` (default) — single-pass 3D body+hands+face, fast,
///   z is hip-relative synthetic from a body prior.
/// * `rtmw3d-with-depth` — RTMW3D 2D + Depth Anything V2 Small
///   relative depth + body-anchor calibration. Real measured z, sized
///   for ~30 fps streaming.
/// * `cigpose-metric-depth` — CIGPose 2D + MoGe-2 true metric
///   depth. Highest depth accuracy, slow (~3 fps) — reference path.
pub const POSE_PROVIDER_ENV: &str = "VULVATAR_POSE_PROVIDER";

/// Runtime pose-estimation implementation used by the tracking worker.
pub trait PoseProvider {
    /// User-visible provider/backend label.
    fn label(&self) -> String;

    /// Warnings collected during provider setup. They are shown once by the UI.
    fn take_load_warnings(&mut self) -> Vec<String> {
        Vec::new()
    }

    fn estimate_pose(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> PoseEstimate;

    /// Update the provider's view of the user's pose calibration. The
    /// tracking worker calls this each iteration with the latest value
    /// from `Application::tracking_calibration.pose` (forwarded via
    /// the calibration mailbox); providers that route the depth
    /// pipeline use it to switch anchor selection (`Upper Body` mode
    /// forces shoulder anchor) and — eventually — to clamp the
    /// metric calibration scale against the captured jitter range.
    /// The default no-op covers providers without a depth path.
    fn set_calibration(&mut self, _calibration: Option<crate::tracking::PoseCalibration>) {}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoseProviderKind {
    Rtmw3d,
    Rtmw3dWithDepth,
    CigposeMetricDepth,
}

impl PoseProviderKind {
    pub fn from_env() -> Result<Self, String> {
        let raw = std::env::var(POSE_PROVIDER_ENV).unwrap_or_else(|_| "rtmw3d".to_string());
        Self::parse(&raw).ok_or_else(|| {
            format!(
                "unknown pose provider '{}'. Set {} to one of: rtmw3d, rtmw3d-with-depth, cigpose-metric-depth",
                raw, POSE_PROVIDER_ENV
            )
        })
    }

    fn parse(value: &str) -> Option<Self> {
        let normalized = value.trim().to_ascii_lowercase().replace('_', "-");
        match normalized.as_str() {
            "" | "rtmw3d" | "rtmpose3d" => Some(Self::Rtmw3d),
            "rtmw3d-with-depth"
            | "rtmw3d-depth"
            | "rtmw3d-da"
            | "rtmw3d-dav2"
            | "rtmw3d+dav2" => Some(Self::Rtmw3dWithDepth),
            "cig-depth"
            | "cigpose-depth"
            | "cigpose-depth-anything"
            | "cigpose-metric-depth"
            | "cigpose-metric"
            | "cigpose-3d" => Some(Self::CigposeMetricDepth),
            _ => None,
        }
    }
}

pub fn create_pose_provider_from_env(
    models_dir: impl AsRef<Path>,
) -> Result<Box<dyn PoseProvider>, String> {
    create_pose_provider(PoseProviderKind::from_env()?, models_dir)
}

pub fn create_pose_provider(
    kind: PoseProviderKind,
    models_dir: impl AsRef<Path>,
) -> Result<Box<dyn PoseProvider>, String> {
    match kind {
        PoseProviderKind::Rtmw3d => create_rtmw3d_provider(models_dir),
        PoseProviderKind::Rtmw3dWithDepth => Rtmw3dWithDepthProvider::from_models_dir(models_dir)
            .map(|p| Box::new(p) as Box<dyn PoseProvider>),
        PoseProviderKind::CigposeMetricDepth => {
            CigposeMetricDepthProvider::from_models_dir(models_dir)
                .map(|p| Box::new(p) as Box<dyn PoseProvider>)
        }
    }
}

#[cfg(feature = "inference")]
fn create_rtmw3d_provider(models_dir: impl AsRef<Path>) -> Result<Box<dyn PoseProvider>, String> {
    Rtmw3dPoseProvider::from_models_dir(models_dir).map(|p| Box::new(p) as Box<dyn PoseProvider>)
}

#[cfg(not(feature = "inference"))]
fn create_rtmw3d_provider(_: impl AsRef<Path>) -> Result<Box<dyn PoseProvider>, String> {
    Err("RTMW3D provider requires the `inference` cargo feature".to_string())
}

#[cfg(feature = "inference")]
struct Rtmw3dPoseProvider {
    inner: super::rtmw3d::Rtmw3dInference,
}

#[cfg(feature = "inference")]
impl Rtmw3dPoseProvider {
    fn from_models_dir(models_dir: impl AsRef<Path>) -> Result<Self, String> {
        Ok(Self {
            inner: super::rtmw3d::Rtmw3dInference::from_models_dir(models_dir)?,
        })
    }
}

#[cfg(feature = "inference")]
impl PoseProvider for Rtmw3dPoseProvider {
    fn label(&self) -> String {
        format!("RTMW3D / {}", self.inner.backend().label())
    }

    fn take_load_warnings(&mut self) -> Vec<String> {
        self.inner.take_load_warnings()
    }

    fn estimate_pose(
        &mut self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> PoseEstimate {
        self.inner
            .estimate_pose(rgb_data, width, height, frame_index)
    }
}

#[cfg(test)]
mod tests {
    use super::{PoseProviderKind, POSE_PROVIDER_ENV};

    #[test]
    fn parses_pose_provider_aliases() {
        assert_eq!(
            PoseProviderKind::parse("rtmw3d"),
            Some(PoseProviderKind::Rtmw3d)
        );
        assert_eq!(
            PoseProviderKind::parse("rtmpose3d"),
            Some(PoseProviderKind::Rtmw3d)
        );
        assert_eq!(
            PoseProviderKind::parse("cigpose_depth"),
            Some(PoseProviderKind::CigposeMetricDepth)
        );
        assert_eq!(
            PoseProviderKind::parse("cigpose-metric-depth"),
            Some(PoseProviderKind::CigposeMetricDepth)
        );
        assert_eq!(
            PoseProviderKind::parse("rtmw3d-with-depth"),
            Some(PoseProviderKind::Rtmw3dWithDepth)
        );
        assert_eq!(
            PoseProviderKind::parse("rtmw3d_with_depth"),
            Some(PoseProviderKind::Rtmw3dWithDepth)
        );
        assert_eq!(
            PoseProviderKind::parse("rtmw3d-da"),
            Some(PoseProviderKind::Rtmw3dWithDepth)
        );
    }

    #[test]
    fn rejects_unknown_pose_provider() {
        let err = PoseProviderKind::parse("depth-only");
        assert_eq!(err, None);

        let message = PoseProviderKind::parse("depth-only")
            .ok_or_else(|| {
                format!(
                    "unknown pose provider 'depth-only'. Set {} to one of: rtmw3d, cigpose-metric-depth",
                    POSE_PROVIDER_ENV
                )
            })
            .unwrap_err();
        assert!(message.contains(POSE_PROVIDER_ENV));
    }
}
