//! Audio-driven lip sync: microphone capture → viseme weights.
//!
//! Gated behind the `lipsync` feature flag (`cpal` + `rustfft` dependencies).
//! When the `inference` feature is also enabled, an ONNX model at
//! `models/viseme_nn.onnx` is used for classification; otherwise a simple
//! volume-based fallback drives the "aa" viseme.

pub mod audio_capture;
#[cfg(feature = "lipsync")]
pub mod mel;
#[cfg(feature = "lipsync")]
pub mod viseme_inference;

/// Five-vowel viseme weights matching VRM preset expressions.
#[derive(Clone, Debug, Default)]
pub struct VisemeWeights {
    pub aa: f32,
    pub ih: f32,
    pub ou: f32,
    pub ee: f32,
    pub oh: f32,
    /// Overall mouth openness (0 = closed, 1 = fully open).
    pub volume: f32,
}

impl VisemeWeights {
    /// Convert to a slice of named (expression_name, weight) pairs
    /// suitable for feeding into the expression weight pipeline.
    pub fn as_expression_pairs(&self) -> [(&'static str, f32); 5] {
        [
            ("aa", self.aa),
            ("ih", self.ih),
            ("ou", self.ou),
            ("ee", self.ee),
            ("oh", self.oh),
        ]
    }
}

/// Information about an available audio input device.
#[derive(Clone, Debug)]
pub struct AudioDeviceInfo {
    pub name: String,
    pub index: usize,
}

/// High-level lip sync processor: owns the audio stream and inferencer,
/// produces `VisemeWeights` each frame.
#[cfg(feature = "lipsync")]
pub struct LipSyncProcessor {
    stream: audio_capture::AudioCaptureStream,
    inferencer: viseme_inference::VisemeInferencer,
    audio_buf: Vec<f32>,
}

#[cfg(feature = "lipsync")]
impl LipSyncProcessor {
    pub fn start(device_index: Option<usize>) -> Result<Self, String> {
        let stream = audio_capture::AudioCaptureStream::open(device_index)?;
        let inferencer = viseme_inference::VisemeInferencer::new();
        Ok(Self {
            stream,
            inferencer,
            audio_buf: Vec::new(),
        })
    }

    /// Call once per frame. Reads the latest audio, runs inference, returns
    /// smoothed viseme weights.
    pub fn process_frame(&mut self, smoothing: f32) -> VisemeWeights {
        let window = viseme_inference::ANALYSIS_WINDOW_SAMPLES;
        self.stream.read_latest(&mut self.audio_buf, window);
        self.inferencer.infer(&self.audio_buf, smoothing)
    }

    /// Current RMS volume from the audio stream.
    pub fn rms_volume(&self) -> f32 {
        self.stream.rms_volume()
    }

    pub fn stop(&mut self) {
        self.stream.stop();
    }
}

#[cfg(feature = "lipsync")]
impl Drop for LipSyncProcessor {
    fn drop(&mut self) {
        self.stop();
    }
}
