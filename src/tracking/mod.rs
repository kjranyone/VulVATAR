type Vec3 = [f32; 3];
type Quat = [f32; 4];

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TrackingSourceId(pub u64);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TrackingImageFormat {
    Rgb8,
    Bgr8,
    Rgba8,
    Bgra8,
    Nv12,
    Yuyv,
}

impl Default for TrackingImageFormat {
    fn default() -> Self {
        Self::Rgba8
    }
}

#[derive(Clone, Debug)]
pub struct TrackingFrame {
    pub source_id: TrackingSourceId,
    pub capture_timestamp: u64,
    pub frame_size: [u32; 2],
    pub image_format: TrackingImageFormat,
    pub body_confidence: f32,
    pub face_confidence: f32,
}

#[derive(Clone, Debug)]
pub struct TrackingRigPose {
    pub source_timestamp: u64,
    pub head: Option<RigTarget>,
    pub neck: Option<RigTarget>,
    pub spine: Option<RigTarget>,
    pub shoulders: RigShoulderTargets,
    pub arms: RigArmTargets,
    pub hands: Option<RigHandTargets>,
    pub expressions: ExpressionWeightSet,
    pub confidence: TrackingConfidenceMap,
}

#[derive(Clone, Debug)]
pub struct RigTarget {
    pub position: Vec3,
    pub orientation: Quat,
    pub confidence: f32,
}

#[derive(Clone, Debug)]
pub struct RigShoulderTargets {
    pub left: Option<RigTarget>,
    pub right: Option<RigTarget>,
}

#[derive(Clone, Debug)]
pub struct RigArmTargets {
    pub left_upper: Option<RigTarget>,
    pub left_lower: Option<RigTarget>,
    pub right_upper: Option<RigTarget>,
    pub right_lower: Option<RigTarget>,
}

#[derive(Clone, Debug)]
pub struct RigHandTargets {
    pub left: Option<RigTarget>,
    pub right: Option<RigTarget>,
}

#[derive(Clone, Debug)]
pub struct ExpressionWeightSet {
    pub weights: Vec<ExpressionWeight>,
}

#[derive(Clone, Debug)]
pub struct ExpressionWeight {
    pub name: String,
    pub weight: f32,
}

#[derive(Clone, Debug)]
pub struct TrackingConfidenceMap {
    pub head_confidence: f32,
    pub torso_confidence: f32,
    pub left_arm_confidence: f32,
    pub right_arm_confidence: f32,
    pub face_confidence: f32,
}

#[derive(Clone, Debug)]
pub struct TrackingSmoothingParams {
    pub position_smoothing: f32,
    pub orientation_smoothing: f32,
    pub expression_smoothing: f32,
    pub confidence_threshold: f32,
    pub stale_timeout_nanos: u64,
}

impl Default for TrackingSmoothingParams {
    fn default() -> Self {
        Self {
            position_smoothing: 0.3,
            orientation_smoothing: 0.3,
            expression_smoothing: 0.2,
            confidence_threshold: 0.5,
            stale_timeout_nanos: 200_000_000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrackingCalibration {
    pub neutral_head_offset: Quat,
    pub scale_factor: f32,
    pub shoulder_width_override: Option<f32>,
    pub head_position_offset: Vec3,
}

impl Default for TrackingCalibration {
    fn default() -> Self {
        Self {
            neutral_head_offset: [0.0, 0.0, 0.0, 1.0],
            scale_factor: 1.0,
            shoulder_width_override: None,
            head_position_offset: [0.0, 0.0, 0.0],
        }
    }
}

impl Default for TrackingRigPose {
    fn default() -> Self {
        Self {
            source_timestamp: 0,
            head: None,
            neck: None,
            spine: None,
            shoulders: RigShoulderTargets {
                left: None,
                right: None,
            },
            arms: RigArmTargets {
                left_upper: None,
                left_lower: None,
                right_upper: None,
                right_lower: None,
            },
            hands: None,
            expressions: ExpressionWeightSet { weights: vec![] },
            confidence: TrackingConfidenceMap {
                head_confidence: 0.0,
                torso_confidence: 0.0,
                left_arm_confidence: 0.0,
                right_arm_confidence: 0.0,
                face_confidence: 0.0,
            },
        }
    }
}

pub struct TrackingMailbox {
    latest_pose: Option<TrackingRigPose>,
    sequence: u64,
}

impl TrackingMailbox {
    pub fn new() -> Self {
        Self {
            latest_pose: None,
            sequence: 0,
        }
    }

    pub fn publish(&mut self, pose: TrackingRigPose) {
        self.latest_pose = Some(pose);
        self.sequence += 1;
    }

    pub fn read(&self) -> Option<TrackingRigPose> {
        self.latest_pose.clone()
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }
}

pub struct TrackingSource {
    source_id: TrackingSourceId,
    frame_index: u64,
    mailbox: TrackingMailbox,
    smoothing: TrackingSmoothingParams,
    calibration: TrackingCalibration,
}

impl TrackingSource {
    pub fn new() -> Self {
        Self {
            source_id: TrackingSourceId(1),
            frame_index: 0,
            mailbox: TrackingMailbox::new(),
            smoothing: TrackingSmoothingParams::default(),
            calibration: TrackingCalibration::default(),
        }
    }

    pub fn sample(&mut self) -> TrackingFrame {
        let frame = TrackingFrame {
            source_id: self.source_id.clone(),
            capture_timestamp: self.frame_index,
            frame_size: [640, 480],
            image_format: TrackingImageFormat::default(),
            body_confidence: 0.9,
            face_confidence: 0.8,
        };

        let pose = TrackingRigPose {
            source_timestamp: self.frame_index,
            head: Some(RigTarget {
                position: [0.0, 1.6, 0.0],
                orientation: [0.0, 0.0, 0.0, 1.0],
                confidence: 0.95,
            }),
            neck: Some(RigTarget {
                position: [0.0, 1.5, 0.0],
                orientation: [0.0, 0.0, 0.0, 1.0],
                confidence: 0.9,
            }),
            spine: Some(RigTarget {
                position: [0.0, 1.0, 0.0],
                orientation: [0.0, 0.0, 0.0, 1.0],
                confidence: 0.85,
            }),
            shoulders: RigShoulderTargets {
                left: Some(RigTarget {
                    position: [-0.15, 1.4, 0.0],
                    orientation: [0.0, 0.0, 0.0, 1.0],
                    confidence: 0.88,
                }),
                right: Some(RigTarget {
                    position: [0.15, 1.4, 0.0],
                    orientation: [0.0, 0.0, 0.0, 1.0],
                    confidence: 0.88,
                }),
            },
            arms: RigArmTargets {
                left_upper: Some(RigTarget {
                    position: [-0.2, 1.2, 0.0],
                    orientation: [0.0, 0.0, 0.0, 1.0],
                    confidence: 0.8,
                }),
                left_lower: None,
                right_upper: Some(RigTarget {
                    position: [0.2, 1.2, 0.0],
                    orientation: [0.0, 0.0, 0.0, 1.0],
                    confidence: 0.8,
                }),
                right_lower: None,
            },
            hands: None,
            expressions: ExpressionWeightSet {
                weights: vec![ExpressionWeight {
                    name: "blink".to_string(),
                    weight: 0.1,
                }],
            },
            confidence: TrackingConfidenceMap {
                head_confidence: 0.95,
                torso_confidence: 0.85,
                left_arm_confidence: 0.8,
                right_arm_confidence: 0.8,
                face_confidence: 0.75,
            },
        };

        self.mailbox.publish(pose);
        self.frame_index += 1;
        frame
    }

    pub fn mailbox(&self) -> &TrackingMailbox {
        &self.mailbox
    }
}
