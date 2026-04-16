//! Mel spectrogram feature extraction from raw audio.
//!
//! Computes a log-mel spectrogram suitable as input to a small viseme
//! classification NN.  The implementation is minimal (no external DSP
//! crate) and uses `rustfft` for the FFT step.

#[cfg(feature = "lipsync")]
mod inner {
    use rustfft::{num_complex::Complex, FftPlanner};

    /// Parameters for mel spectrogram extraction.
    pub struct MelConfig {
        pub sample_rate: u32,
        pub n_fft: usize,
        pub hop_length: usize,
        pub n_mels: usize,
        pub fmin: f32,
        pub fmax: f32,
    }

    impl Default for MelConfig {
        fn default() -> Self {
            Self {
                sample_rate: 16_000,
                n_fft: 512,
                hop_length: 160, // 10 ms at 16 kHz
                n_mels: 40,
                fmin: 60.0,
                fmax: 7600.0,
            }
        }
    }

    /// Pre-computed mel filterbank + FFT planner for reuse across frames.
    pub struct MelExtractor {
        config: MelConfig,
        fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
        mel_bank: Vec<Vec<f32>>, // [n_mels][n_fft/2+1]
        hann_window: Vec<f32>,
    }

    impl MelExtractor {
        pub fn new(config: MelConfig) -> Self {
            let mut planner = FftPlanner::new();
            let fft = planner.plan_fft_forward(config.n_fft);
            let mel_bank = build_mel_filterbank(
                config.n_fft,
                config.sample_rate,
                config.n_mels,
                config.fmin,
                config.fmax,
            );
            let hann_window = hann(config.n_fft);
            Self {
                config,
                fft,
                mel_bank,
                hann_window,
            }
        }

        /// Compute a log-mel spectrogram from `samples` (mono, at `config.sample_rate`).
        ///
        /// Returns a flat `Vec<f32>` of shape `[n_frames, n_mels]` and the frame count.
        pub fn extract(&self, samples: &[f32]) -> (Vec<f32>, usize) {
            let n_fft = self.config.n_fft;
            let hop = self.config.hop_length;
            let n_bins = n_fft / 2 + 1;

            if samples.len() < n_fft {
                return (vec![0.0; self.config.n_mels], 1);
            }

            let n_frames = (samples.len() - n_fft) / hop + 1;
            let mut output = Vec::with_capacity(n_frames * self.config.n_mels);
            let mut fft_buf = vec![Complex::new(0.0f32, 0.0); n_fft];

            for frame_idx in 0..n_frames {
                let start = frame_idx * hop;
                // Apply Hann window and copy into FFT buffer.
                for i in 0..n_fft {
                    let s = if start + i < samples.len() {
                        samples[start + i]
                    } else {
                        0.0
                    };
                    fft_buf[i] = Complex::new(s * self.hann_window[i], 0.0);
                }

                self.fft.process(&mut fft_buf);

                // Power spectrum (only positive frequencies).
                let power: Vec<f32> = fft_buf[..n_bins]
                    .iter()
                    .map(|c| c.norm_sqr())
                    .collect();

                // Apply mel filterbank and log.
                for mel_filter in &self.mel_bank {
                    let energy: f32 = mel_filter
                        .iter()
                        .zip(power.iter())
                        .map(|(w, p)| w * p)
                        .sum();
                    // Log with floor to avoid -inf.
                    output.push((energy.max(1e-10)).ln());
                }
            }

            (output, n_frames)
        }

        pub fn n_mels(&self) -> usize {
            self.config.n_mels
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
    }

    fn build_mel_filterbank(
        n_fft: usize,
        sample_rate: u32,
        n_mels: usize,
        fmin: f32,
        fmax: f32,
    ) -> Vec<Vec<f32>> {
        let n_bins = n_fft / 2 + 1;
        let mel_min = hz_to_mel(fmin);
        let mel_max = hz_to_mel(fmax);

        // n_mels + 2 evenly spaced points in mel space.
        let mel_points: Vec<f32> = (0..n_mels + 2)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
        let bin_points: Vec<f32> = hz_points
            .iter()
            .map(|&hz| hz * n_fft as f32 / sample_rate as f32)
            .collect();

        (0..n_mels)
            .map(|m| {
                let left = bin_points[m];
                let center = bin_points[m + 1];
                let right = bin_points[m + 2];
                (0..n_bins)
                    .map(|k| {
                        let kf = k as f32;
                        if kf < left || kf > right {
                            0.0
                        } else if kf <= center {
                            (kf - left) / (center - left).max(1e-6)
                        } else {
                            (right - kf) / (right - center).max(1e-6)
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn hann(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
            })
            .collect()
    }
}

#[cfg(feature = "lipsync")]
pub use inner::*;
