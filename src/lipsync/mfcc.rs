//! MFCC (Mel-Frequency Cepstral Coefficients) extraction.
//!
//! Reimplements the uLipSync MFCC pipeline:
//! pre-emphasis → Hamming window → FFT → mel filterbank → log → DCT-II → coefficients 1..12

#[cfg(feature = "lipsync")]
mod inner {
    use rustfft::{num_complex::Complex, FftPlanner};

    /// MFCC extraction parameters matching uLipSync defaults.
    pub struct MfccConfig {
        pub sample_rate: u32,
        pub n_fft: usize,
        pub n_mels: usize,
        pub n_mfcc: usize, // output coefficients (excluding 0th)
        pub fmin: f32,
        pub fmax: f32,
        pub pre_emphasis: f32,
    }

    impl Default for MfccConfig {
        fn default() -> Self {
            Self {
                sample_rate: 16_000,
                n_fft: 1024,
                n_mels: 30,
                n_mfcc: 12,
                fmin: 0.0,
                fmax: 8000.0,
                pre_emphasis: 0.97,
            }
        }
    }

    /// Reusable MFCC extractor with pre-computed filterbank and FFT plan.
    pub struct MfccExtractor {
        config: MfccConfig,
        fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
        mel_bank: Vec<Vec<f32>>, // [n_mels][n_fft/2+1]
        hamming: Vec<f32>,
    }

    impl MfccExtractor {
        pub fn new(config: MfccConfig) -> Self {
            let mut planner = FftPlanner::new();
            let fft = planner.plan_fft_forward(config.n_fft);
            let mel_bank = build_mel_filterbank(
                config.n_fft,
                config.sample_rate,
                config.n_mels,
                config.fmin,
                config.fmax,
            );
            let hamming = hamming_window(config.n_fft);
            Self {
                config,
                fft,
                mel_bank,
                hamming,
            }
        }

        /// Extract MFCC vector (length = `n_mfcc`) from a single audio window.
        /// Input `samples` should be at `sample_rate`, length >= `n_fft`.
        /// Returns the MFCC coefficients (indices 1..n_mfcc, 0th skipped).
        pub fn extract(&self, samples: &[f32]) -> Vec<f32> {
            let n = self.config.n_fft;
            let n_bins = n / 2 + 1;

            // Pre-emphasis.
            let mut emphasized = vec![0.0f32; n];
            let len = samples.len().min(n);
            if len > 0 {
                emphasized[0] = samples[0];
            }
            for i in 1..len {
                emphasized[i] = samples[i] - self.config.pre_emphasis * samples[i - 1];
            }

            // Hamming window + FFT.
            let mut fft_buf: Vec<Complex<f32>> = emphasized
                .iter()
                .zip(self.hamming.iter())
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
                .collect();
            self.fft.process(&mut fft_buf);

            // Power spectrum.
            let power: Vec<f32> = fft_buf[..n_bins]
                .iter()
                .map(|c| c.norm_sqr())
                .collect();

            // Mel filterbank → dB.
            let mel_db: Vec<f32> = self
                .mel_bank
                .iter()
                .map(|filter| {
                    let energy: f32 = filter.iter().zip(power.iter()).map(|(w, p)| w * p).sum();
                    10.0 * (energy.max(1e-10)).log10()
                })
                .collect();

            // DCT-II (coefficients 1..n_mfcc, skip 0th).
            let n_mels = self.config.n_mels;
            let n_mfcc = self.config.n_mfcc;
            let mut mfcc = Vec::with_capacity(n_mfcc);
            for i in 1..=n_mfcc {
                let mut sum = 0.0f32;
                for j in 0..n_mels {
                    sum += mel_db[j]
                        * ((j as f32 + 0.5) * i as f32 * std::f32::consts::PI / n_mels as f32)
                            .cos();
                }
                mfcc.push(sum);
            }

            mfcc
        }

        pub fn n_mfcc(&self) -> usize {
            self.config.n_mfcc
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────

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

        let mel_points: Vec<f32> = (0..n_mels + 2)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();
        let bin_points: Vec<f32> = mel_points
            .iter()
            .map(|&m| mel_to_hz(m) * n_fft as f32 / sample_rate as f32)
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

    fn hamming_window(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
            .collect()
    }
}

#[cfg(feature = "lipsync")]
pub use inner::*;
