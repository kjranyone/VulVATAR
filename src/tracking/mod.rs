#[derive(Clone, Debug)]
pub struct TrackingFrame {
    pub frame_index: u64,
    pub body_confidence: f32,
    pub face_confidence: f32,
}

#[derive(Clone, Debug, Default)]
pub struct TrackingRigPose {
    pub head_orientation: [f32; 4],
    pub chest_orientation: [f32; 4],
    pub left_hand_target: [f32; 3],
    pub right_hand_target: [f32; 3],
    pub expression_weights: Vec<ExpressionWeight>,
}

#[derive(Clone, Debug)]
pub struct ExpressionWeight {
    pub name: String,
    pub weight: f32,
}

pub struct TrackingSource {
    frame_index: u64,
}

impl TrackingSource {
    pub fn new() -> Self {
        Self { frame_index: 0 }
    }

    pub fn sample(&mut self) -> (TrackingFrame, TrackingRigPose) {
        let frame = TrackingFrame {
            frame_index: self.frame_index,
            body_confidence: 0.9,
            face_confidence: 0.8,
        };
        let pose = TrackingRigPose {
            head_orientation: [0.0, 0.0, 0.0, 1.0],
            chest_orientation: [0.0, 0.0, 0.0, 1.0],
            left_hand_target: [-0.2, 1.2, 0.3],
            right_hand_target: [0.2, 1.2, 0.3],
            expression_weights: vec![ExpressionWeight {
                name: "blink".to_string(),
                weight: 0.1,
            }],
        };
        self.frame_index += 1;
        (frame, pose)
    }
}
