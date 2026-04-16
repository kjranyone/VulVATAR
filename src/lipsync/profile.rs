//! Phoneme profile: stored MFCC mean vectors for distance-based matching.
//!
//! Follows the uLipSync approach: each phoneme holds the mean MFCC vector
//! computed from calibration samples.  At runtime the live MFCC is compared
//! against all phoneme means using L2 distance → `10^(-d)` scoring →
//! softmax-like normalization.

#[cfg(feature = "lipsync")]
mod inner {
    use serde::{Deserialize, Serialize};

    /// A single phoneme's calibration data.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct PhonemeProfile {
        pub name: String,
        /// Mean MFCC vector (length = n_mfcc, typically 12).
        pub mfcc_mean: Vec<f32>,
    }

    /// Collection of phoneme profiles used for matching.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct LipSyncProfile {
        pub phonemes: Vec<PhonemeProfile>,
    }

    impl LipSyncProfile {
        /// Default Japanese 5-vowel + silent profile with pre-baked MFCC means.
        ///
        /// These values are approximate centroids derived from typical Japanese
        /// vowel formant patterns.  They work as a reasonable starting point;
        /// user calibration will improve accuracy.
        pub fn default_japanese() -> Self {
            // Pre-baked MFCC-12 mean vectors for Japanese vowels.
            // Generated from formant-based synthetic audio analysis.
            Self {
                phonemes: vec![
                    PhonemeProfile {
                        name: "aa".into(),
                        mfcc_mean: vec![
                            -2.5, 1.8, -0.5, 0.3, -0.8, 0.2, -0.1, 0.4, -0.3, 0.1, -0.2, 0.1,
                        ],
                    },
                    PhonemeProfile {
                        name: "ih".into(),
                        mfcc_mean: vec![
                            -3.0, -1.5, 1.2, -0.8, 0.5, -0.3, 0.6, -0.2, 0.3, -0.1, 0.2, -0.1,
                        ],
                    },
                    PhonemeProfile {
                        name: "ou".into(),
                        mfcc_mean: vec![
                            -2.8, 0.5, -1.0, 0.8, -0.2, 0.5, -0.4, 0.1, -0.2, 0.3, -0.1, 0.0,
                        ],
                    },
                    PhonemeProfile {
                        name: "ee".into(),
                        mfcc_mean: vec![
                            -2.7, -0.8, 0.9, -0.4, 0.7, -0.5, 0.3, -0.3, 0.1, -0.2, 0.3, -0.1,
                        ],
                    },
                    PhonemeProfile {
                        name: "oh".into(),
                        mfcc_mean: vec![
                            -2.6, 1.0, -0.8, 0.5, -0.5, 0.4, -0.2, 0.2, -0.1, 0.2, -0.1, 0.1,
                        ],
                    },
                    PhonemeProfile {
                        name: "silent".into(),
                        mfcc_mean: vec![
                            -8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        ],
                    },
                ],
            }
        }

        /// Compute normalized phoneme weights from a live MFCC vector.
        ///
        /// Uses L2 distance → `10^(-distance)` scoring, matching uLipSync's
        /// default `CompareMethod.L2Norm`.
        pub fn match_mfcc(&self, mfcc: &[f32]) -> Vec<(String, f32)> {
            let scores: Vec<f32> = self
                .phonemes
                .iter()
                .map(|p| {
                    let n = mfcc.len().min(p.mfcc_mean.len());
                    if n == 0 {
                        return 0.0;
                    }
                    let dist_sq: f32 = mfcc[..n]
                        .iter()
                        .zip(p.mfcc_mean[..n].iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    let dist = (dist_sq / n as f32).sqrt();
                    10.0f32.powf(-dist)
                })
                .collect();

            let total: f32 = scores.iter().sum();
            self.phonemes
                .iter()
                .zip(scores.iter())
                .map(|(p, &s)| {
                    let ratio = if total > 0.0 { s / total } else { 0.0 };
                    (p.name.clone(), ratio)
                })
                .collect()
        }
    }
}

#[cfg(feature = "lipsync")]
pub use inner::*;
