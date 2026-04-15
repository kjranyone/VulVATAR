//! Webcam capture backend using the `nokhwa` crate.
//!
//! This module is only compiled when the `webcam` cargo feature is enabled.
//! It provides a thin wrapper around nokhwa's camera API that captures RGB
//! frames suitable for the pose estimation pipeline.

use super::CameraInfo;
use log::{error, info};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType, Resolution};
use nokhwa::Camera;

/// Wraps a nokhwa `Camera` handle and provides a simple frame-grab interface.
pub struct WebcamCapture {
    camera: Camera,
    width: u32,
    height: u32,
}

impl WebcamCapture {
    /// Open a camera device by index with the requested resolution and frame rate.
    ///
    /// Falls back to the camera's default format if the exact requested format
    /// is not supported.
    pub fn open(index: usize, width: u32, height: u32, fps: u32) -> Result<Self, String> {
        let camera_index = CameraIndex::Index(index as u32);

        let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::HighestResolution(
            Resolution::new(width, height),
        ));

        let mut camera = Camera::new(camera_index, requested)
            .map_err(|e| format!("nokhwa: failed to create camera: {}", e))?;

        camera
            .open_stream()
            .map_err(|e| format!("nokhwa: failed to open stream: {}", e))?;

        // Query the actual resolution the driver selected.
        let resolution = camera.resolution();
        let actual_width = resolution.width();
        let actual_height = resolution.height();

        info!(
            "webcam: opened camera {} at {}x{} (requested {}x{} @ {} fps)",
            index, actual_width, actual_height, width, height, fps,
        );

        Ok(Self {
            camera,
            width: actual_width,
            height: actual_height,
        })
    }

    /// Grab a single frame as an RGB byte buffer.
    ///
    /// The returned `Vec<u8>` has `width * height * 3` bytes in row-major RGB
    /// order.
    pub fn grab_frame(&mut self) -> Result<Vec<u8>, String> {
        let frame = self
            .camera
            .frame()
            .map_err(|e| format!("nokhwa: frame grab failed: {}", e))?;

        let decoded = frame
            .decode_image::<RgbFormat>()
            .map_err(|e| format!("nokhwa: frame decode failed: {}", e))?;

        // Update dimensions in case the driver changed resolution mid-stream.
        self.width = decoded.width();
        self.height = decoded.height();

        Ok(decoded.into_raw())
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

impl Drop for WebcamCapture {
    fn drop(&mut self) {
        let _ = self.camera.stop_stream();
    }
}

/// Enumerate cameras available on the system via nokhwa.
pub fn list_cameras_impl() -> Vec<CameraInfo> {
    match nokhwa::query(nokhwa::utils::ApiBackend::Auto) {
        Ok(devices) => devices
            .into_iter()
            .enumerate()
            .map(|(i, info)| CameraInfo {
                index: i,
                name: info.human_name().to_string(),
            })
            .collect(),
        Err(e) => {
            error!("webcam: failed to enumerate cameras: {}", e);
            Vec::new()
        }
    }
}
