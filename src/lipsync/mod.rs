//! Audio-driven lip sync using the uLipSync algorithm.
//!
//! Gated behind the `lipsync` feature flag (`cpal` + `rustfft` dependencies).
//!
//! Pipeline: microphone → MFCC extraction → profile distance matching → viseme weights.
//! No external model file required; uses pre-baked Japanese vowel profiles by default.

pub mod audio_capture;
#[cfg(feature = "lipsync")]
pub mod mfcc;
#[cfg(feature = "lipsync")]
pub mod profile;

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

/// High-level lip sync processor: owns the audio stream, MFCC extractor,
/// and phoneme profile. Produces `VisemeWeights` each frame.
#[cfg(feature = "lipsync")]
pub struct LipSyncProcessor {
    stream: audio_capture::AudioCaptureStream,
    mfcc: mfcc::MfccExtractor,
    profile: profile::LipSyncProfile,
    audio_buf: Vec<f32>,
    /// SmoothDamp state per viseme (aa, ih, ou, ee, oh).
    smoothed: [f32; 5],
    smooth_velocity: [f32; 5],
}

#[cfg(feature = "lipsync")]
impl LipSyncProcessor {
    pub fn start(device_index: Option<usize>) -> Result<Self, String> {
        let stream = audio_capture::AudioCaptureStream::open(device_index)?;
        let mfcc = mfcc::MfccExtractor::new(Default::default());
        let profile = profile::LipSyncProfile::default_japanese();
        Ok(Self {
            stream,
            mfcc,
            profile,
            audio_buf: Vec::new(),
            smoothed: [0.0; 5],
            smooth_velocity: [0.0; 5],
        })
    }

    /// Call once per frame. Reads the latest audio, computes MFCC,
    /// matches against the profile, and returns smoothed viseme weights.
    ///
    /// `smoothing` is in seconds (uLipSync default ~0.1). `dt` is frame delta time.
    pub fn process_frame(&mut self, smoothing: f32, dt: f32) -> VisemeWeights {
        // Read ~64ms of audio (1024 samples at 16kHz).
        self.stream.read_latest(&mut self.audio_buf, 1024);

        if self.audio_buf.is_empty() {
            return VisemeWeights::default();
        }

        // Extract MFCC from the audio window.
        let mfcc_vec = self.mfcc.extract(&self.audio_buf);

        // Match against phoneme profiles.
        let matches = self.profile.match_mfcc(&mfcc_vec);

        // Extract the 5 vowel weights.
        let mut raw = [0.0f32; 5];
        for (name, weight) in &matches {
            match name.as_str() {
                "aa" => raw[0] = *weight,
                "ih" => raw[1] = *weight,
                "ou" => raw[2] = *weight,
                "ee" => raw[3] = *weight,
                "oh" => raw[4] = *weight,
                _ => {}
            }
        }

        // SmoothDamp (matches Unity's Mathf.SmoothDamp).
        for ((smoothed, raw_val), velocity) in self.smoothed.iter_mut()
            .zip(raw.iter())
            .zip(self.smooth_velocity.iter_mut())
        {
            *smoothed = smooth_damp(
                *smoothed,
                *raw_val,
                velocity,
                smoothing.max(0.001),
                dt,
            );
        }

        let volume = self.smoothed.iter().sum();

        VisemeWeights {
            aa: self.smoothed[0],
            ih: self.smoothed[1],
            ou: self.smoothed[2],
            ee: self.smoothed[3],
            oh: self.smoothed[4],
            volume,
        }
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

/// Unity-compatible SmoothDamp.
#[cfg(feature = "lipsync")]
fn smooth_damp(current: f32, target: f32, velocity: &mut f32, smooth_time: f32, dt: f32) -> f32 {
    let omega = 2.0 / smooth_time;
    let x = omega * dt;
    let exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x);
    let change = current - target;
    let temp = (*velocity + omega * change) * dt;
    *velocity = (*velocity - omega * temp) * exp;
    target + (change + temp) * exp
}
