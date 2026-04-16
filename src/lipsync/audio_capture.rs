//! Microphone audio capture using `cpal`.
//!
//! Runs a dedicated input stream on a background thread. Audio samples
//! are written into a lock-free ring buffer that the lip-sync analyser
//! reads each frame.

#[cfg(feature = "lipsync")]
mod inner {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use log::{error, info};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use super::super::AudioDeviceInfo;

    /// Fixed analysis sample rate. Audio is captured at the device's native
    /// rate and resampled to this rate before being placed in the ring buffer.
    /// For MFCC / NN viseme analysis 16 kHz is sufficient and keeps buffers small.
    pub const ANALYSIS_SAMPLE_RATE: u32 = 16_000;

    /// Ring buffer capacity in samples (~0.5 s at 16 kHz).
    const RING_CAPACITY: usize = 8_000;

    fn device_display_name(dev: &cpal::Device, fallback_idx: usize) -> String {
        dev.description()
            .map(|desc| desc.name().to_string())
            .unwrap_or_else(|_| format!("Device {}", fallback_idx))
    }

    /// Lists available audio input devices.
    pub fn list_audio_devices() -> Vec<AudioDeviceInfo> {
        let host = cpal::default_host();
        host.input_devices()
            .map(|devices| {
                devices
                    .enumerate()
                    .map(|(i, dev)| {
                        let name = device_display_name(&dev, i);
                        AudioDeviceInfo { name, index: i }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Manages a cpal input stream that writes mono f32 samples into a ring buffer.
    pub struct AudioCaptureStream {
        ring: Arc<Mutex<RingBuffer>>,
        running: Arc<AtomicBool>,
        stream: Option<cpal::Stream>,
    }

    struct RingBuffer {
        data: Vec<f32>,
        write_pos: usize,
        /// Number of fresh samples since the last `drain()`.
        fresh: usize,
    }

    impl RingBuffer {
        fn new(capacity: usize) -> Self {
            Self {
                data: vec![0.0; capacity],
                write_pos: 0,
                fresh: 0,
            }
        }

        fn push(&mut self, sample: f32) {
            self.data[self.write_pos] = sample;
            self.write_pos = (self.write_pos + 1) % self.data.len();
            self.fresh = (self.fresh + 1).min(self.data.len());
        }

        /// Copy the most recent `count` samples into `dst`.
        /// Returns the actual number of samples copied (may be less than `count`
        /// if not enough fresh data is available).
        fn read_latest(&self, dst: &mut Vec<f32>, count: usize) -> usize {
            let available = self.fresh.min(count).min(self.data.len());
            dst.clear();
            if available == 0 {
                return 0;
            }
            let start = (self.write_pos + self.data.len() - available) % self.data.len();
            if start + available <= self.data.len() {
                dst.extend_from_slice(&self.data[start..start + available]);
            } else {
                let first = self.data.len() - start;
                dst.extend_from_slice(&self.data[start..]);
                dst.extend_from_slice(&self.data[..available - first]);
            }
            available
        }
    }

    impl AudioCaptureStream {
        /// Open the default input device (or the device at `device_index`) and
        /// start capturing audio.
        pub fn open(device_index: Option<usize>) -> Result<Self, String> {
            let host = cpal::default_host();

            let device = if let Some(idx) = device_index {
                host.input_devices()
                    .map_err(|e| format!("audio: failed to enumerate devices: {e}"))?
                    .nth(idx)
                    .ok_or_else(|| format!("audio: device index {idx} not found"))?
            } else {
                host.default_input_device()
                    .ok_or_else(|| "audio: no default input device".to_string())?
            };

            let dev_name = device_display_name(&device, device_index.unwrap_or(0));
            info!("audio: opening device '{}'", dev_name);

            let supported = device
                .supported_input_configs()
                .map_err(|e| format!("audio: no supported configs: {e}"))?
                .collect::<Vec<_>>();

            // Prefer a config close to our analysis rate, mono, f32.
            let config = Self::pick_config(&supported)?;
            let device_sample_rate = config.sample_rate();
            let device_channels = config.channels() as usize;

            info!(
                "audio: capture config: {}ch {}Hz",
                device_channels, device_sample_rate,
            );

            let ring = Arc::new(Mutex::new(RingBuffer::new(RING_CAPACITY)));
            let running = Arc::new(AtomicBool::new(true));

            let ring_clone = ring.clone();
            let running_clone = running.clone();

            // Simple sample-rate conversion state (linear decimation/interpolation).
            let ratio = ANALYSIS_SAMPLE_RATE as f64 / device_sample_rate as f64;
            let mut resample_accum: f64 = 0.0;

            let stream = device
                .build_input_stream(
                    &config.into(),
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        if !running_clone.load(Ordering::Relaxed) {
                            return;
                        }
                        let mut ring = match ring_clone.lock() {
                            Ok(r) => r,
                            Err(_) => return,
                        };
                        // Mix to mono and resample.
                        let frame_count = data.len() / device_channels;
                        for frame_idx in 0..frame_count {
                            let mono: f32 = (0..device_channels)
                                .map(|ch| data[frame_idx * device_channels + ch])
                                .sum::<f32>()
                                / device_channels as f32;

                            resample_accum += ratio;
                            while resample_accum >= 1.0 {
                                resample_accum -= 1.0;
                                ring.push(mono);
                            }
                        }
                    },
                    move |err| {
                        error!("audio: stream error: {}", err);
                    },
                    None,
                )
                .map_err(|e| format!("audio: failed to build stream: {e}"))?;

            stream
                .play()
                .map_err(|e| format!("audio: failed to start stream: {e}"))?;

            info!("audio: capture started");

            Ok(Self {
                ring,
                running,
                stream: Some(stream),
            })
        }

        /// Read the most recent `count` samples (at `ANALYSIS_SAMPLE_RATE`) into `dst`.
        pub fn read_latest(&self, dst: &mut Vec<f32>, count: usize) -> usize {
            match self.ring.lock() {
                Ok(ring) => ring.read_latest(dst, count),
                Err(_) => 0,
            }
        }

        /// Read the current RMS volume level (0.0 – 1.0 ish).
        pub fn rms_volume(&self) -> f32 {
            let mut buf = Vec::new();
            let n = self.read_latest(&mut buf, 1600); // ~100ms
            if n == 0 {
                return 0.0;
            }
            let sum_sq: f32 = buf.iter().map(|s| s * s).sum();
            (sum_sq / n as f32).sqrt()
        }

        pub fn stop(&mut self) {
            self.running.store(false, Ordering::Relaxed);
            self.stream.take(); // dropping the stream stops it
            info!("audio: capture stopped");
        }

        fn pick_config(
            supported: &[cpal::SupportedStreamConfigRange],
        ) -> Result<cpal::SupportedStreamConfig, String> {
            if supported.is_empty() {
                return Err("audio: no supported stream configs".into());
            }

            let range = &supported[0];
            let rate = ANALYSIS_SAMPLE_RATE
                .max(range.min_sample_rate())
                .min(range.max_sample_rate());
            Ok(range.clone().with_sample_rate(rate))
        }
    }

    impl Drop for AudioCaptureStream {
        fn drop(&mut self) {
            self.stop();
        }
    }
}

#[cfg(feature = "lipsync")]
pub use inner::*;

/// Stub when the `lipsync` feature is disabled.
#[cfg(not(feature = "lipsync"))]
pub fn list_audio_devices() -> Vec<super::AudioDeviceInfo> {
    Vec::new()
}
