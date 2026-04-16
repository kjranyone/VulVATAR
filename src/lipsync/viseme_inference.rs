//! Viseme inference: audio → 5-vowel weights.
//!
//! Two backends:
//! - **ONNX NN**: loads a small classification model from `models/viseme_nn.onnx`.
//!   Input: `[1, n_mels, n_frames]` log-mel spectrogram.
//!   Output: `[1, 6]` softmax (aa, ih, ou, ee, oh, silent).
//! - **Volume fallback**: maps RMS volume to simple mouth-open ("aa") weight
//!   when no ONNX model is available.

#[cfg(feature = "lipsync")]
mod inner {
    use super::super::{mel::MelExtractor, VisemeWeights};
    use log::info;

    /// Number of audio samples to analyse per frame (~100 ms at 16 kHz).
    pub const ANALYSIS_WINDOW_SAMPLES: usize = 1600;

    pub struct VisemeInferencer {
        #[allow(dead_code)] // used only with the `inference` feature
        mel: MelExtractor,
        #[cfg(feature = "inference")]
        ort_session: Option<ort::Session>,
        /// Exponential smoothing state for each of the 6 classes.
        smoothed: [f32; 6],
    }

    impl VisemeInferencer {
        pub fn new() -> Self {
            let mel = MelExtractor::new(Default::default());

            #[cfg(feature = "inference")]
            let ort_session = Self::try_load_model();

            #[cfg(feature = "inference")]
            if ort_session.is_some() {
                info!("lipsync: ONNX viseme model loaded");
            } else {
                info!("lipsync: no ONNX model found, using volume fallback");
            }

            #[cfg(not(feature = "inference"))]
            info!("lipsync: inference feature disabled, using volume fallback");

            Self {
                mel,
                #[cfg(feature = "inference")]
                ort_session,
                smoothed: [0.0; 6],
            }
        }

        /// Run inference on a chunk of audio samples (mono, 16 kHz).
        /// `smoothing` is in [0, 1] where 0 = no smoothing, 1 = max smoothing.
        pub fn infer(&mut self, samples: &[f32], smoothing: f32) -> VisemeWeights {
            let raw = self.infer_raw(samples);

            // Exponential moving average.
            let alpha = 1.0 - smoothing.clamp(0.0, 0.99);
            for i in 0..6 {
                self.smoothed[i] = self.smoothed[i] * (1.0 - alpha) + raw[i] * alpha;
            }

            VisemeWeights {
                aa: self.smoothed[0],
                ih: self.smoothed[1],
                ou: self.smoothed[2],
                ee: self.smoothed[3],
                oh: self.smoothed[4],
                volume: self.smoothed[..5].iter().sum(),
            }
        }

        fn infer_raw(&self, samples: &[f32]) -> [f32; 6] {
            #[cfg(feature = "inference")]
            if let Some(ref session) = self.ort_session {
                if let Some(result) = self.infer_onnx(session, samples) {
                    return result;
                }
            }

            // Volume fallback: map RMS to a simple "aa" weight.
            self.volume_fallback(samples)
        }

        fn volume_fallback(&self, samples: &[f32]) -> [f32; 6] {
            if samples.is_empty() {
                return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]; // silent
            }
            let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
            // Map RMS (typically 0.0–0.3) to mouth openness.
            let open = (rms * 8.0).clamp(0.0, 1.0);
            let silent = 1.0 - open;
            // Simple heuristic: mostly "aa" when mouth is open.
            [open * 0.7, open * 0.05, open * 0.1, open * 0.05, open * 0.1, silent]
        }

        #[cfg(feature = "inference")]
        fn try_load_model() -> Option<ort::Session> {
            let model_path = std::path::Path::new("models/viseme_nn.onnx");
            if !model_path.exists() {
                return None;
            }
            match ort::Session::builder()
                .and_then(|b| b.with_model_from_file(model_path))
            {
                Ok(session) => Some(session),
                Err(e) => {
                    log::warn!("lipsync: failed to load ONNX model: {}", e);
                    None
                }
            }
        }

        #[cfg(feature = "inference")]
        fn infer_onnx(&self, session: &ort::Session, samples: &[f32]) -> Option<[f32; 6]> {
            let (mel_data, n_frames) = self.mel.extract(samples);
            let n_mels = self.mel.n_mels();

            // Reshape to [1, n_mels, n_frames] for the model.
            let shape = vec![1, n_mels as i64, n_frames as i64];

            // Transpose: mel_data is [n_frames, n_mels], model expects [n_mels, n_frames].
            let mut transposed = vec![0.0f32; n_mels * n_frames];
            for f in 0..n_frames {
                for m in 0..n_mels {
                    transposed[m * n_frames + f] = mel_data[f * n_mels + m];
                }
            }

            let input = ndarray::ArrayD::from_shape_vec(
                shape.iter().map(|&s| s as usize).collect::<Vec<_>>(),
                transposed,
            )
            .ok()?;

            let outputs = session.run(ort::inputs![input].ok()?).ok()?;
            let output_tensor = outputs.get(0)?;
            let view = output_tensor.try_extract_tensor::<f32>().ok()?;

            let mut result = [0.0f32; 6];
            let raw = view.as_slice()?;

            // Softmax.
            let max_val = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = raw.iter().map(|v| (v - max_val).exp()).sum();
            for (i, &v) in raw.iter().enumerate().take(6) {
                result[i] = (v - max_val).exp() / exp_sum;
            }

            Some(result)
        }
    }
}

#[cfg(feature = "lipsync")]
pub use inner::*;
