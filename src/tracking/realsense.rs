//! RealSense D400-series (D435) depth-camera capture backend.
//!
//! This module is only compiled when the `realsense` cargo feature is enabled
//! (see `docs/realsense-build.md` for the native `librealsense2` build wiring).
//!
//! It streams synchronized color + depth and, each frame, aligns the depth
//! image into the color image so that a color pixel indexes the matching depth
//! sample directly. Each grabbed frame is handed to the pipeline as:
//!   - an RGB8 buffer (`w*h*3`, row-major, top-down) so the RTMW3D provider
//!     consumes it unchanged, plus
//!   - the aligned depth as raw Z16 + the metric scale + the color intrinsics,
//!     ready to be deprojected into a metric point cloud.
//!
//! Deliberately **decoupled from `feature = "inference"`**: it yields raw
//! metric depth, not a `MetricDepthFrame`, so the camera can be brought up and
//! validated without the ONNX stack. The raw-depth → `MetricDepthFrame`
//! deprojection lives with the other depth consumers (the provider), which is
//! where the coordinate frame / occlusion policy belongs.

use std::collections::HashSet;
use std::time::Duration;

use log::info;
use realsense_rust::{
    config::Config,
    context::Context,
    frame::{ColorFrame, DepthFrame, PixelKind},
    prelude::FrameEx,
    kind::{Rs2CameraInfo, Rs2Format, Rs2ProductLine, Rs2StreamKind},
    pipeline::{ActivePipeline, InactivePipeline},
    processing_blocks::align::Align,
};

/// Depth stream resolution requested from the device. 848x480 is a native
/// D435 depth mode; the align block resamples it into the color image, so the
/// depth handed downstream always ends up at the *color* resolution.
const DEPTH_W: usize = 848;
const DEPTH_H: usize = 480;

/// Default D435 metric depth scale (metres per raw Z16 unit = 1 mm). Used only
/// until the real value is read from the depth sensor on the first frame.
const DEFAULT_DEPTH_UNITS_M: f32 = 0.001;

/// Pinhole intrinsics of the color image (== the aligned depth image).
///
/// Lens distortion is ignored: the D435 color/aligned stream uses an
/// (Inverse-)Brown-Conrady model whose coefficients are effectively zero, so a
/// pinhole deprojection is accurate to well under the depth noise floor.
#[derive(Clone, Copy, Debug)]
pub struct CamIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
}

impl CamIntrinsics {
    /// Deproject a pixel `(u, v)` at metric depth `z_m` into a camera-space 3D
    /// point in metres. The output frame is x-right / y-down / z-forward — the
    /// RealSense convention, which is exactly `MetricDepthFrame`'s convention.
    #[inline]
    pub fn deproject(&self, u: f32, v: f32, z_m: f32) -> [f32; 3] {
        [
            (u - self.cx) / self.fx * z_m,
            (v - self.cy) / self.fy * z_m,
            z_m,
        ]
    }
}

/// One synchronized, color-aligned frame from the D435.
pub struct RealSenseFrame {
    /// Color image: `width*height*3` bytes, row-major RGB8, top-down.
    pub rgb: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Depth aligned into the color image: `width*height` raw Z16 samples,
    /// row-major, top-down. `0` marks an invalid / no-return pixel.
    pub depth_raw: Vec<u16>,
    /// Metres per raw depth unit (D435 default 0.001 = 1 mm).
    pub depth_units: f32,
    /// Intrinsics of the color image (== the aligned depth image).
    pub intrinsics: CamIntrinsics,
}

impl RealSenseFrame {
    /// Metric depth (metres) at an integer pixel, or `None` if invalid.
    #[inline]
    pub fn depth_m(&self, u: usize, v: usize) -> Option<f32> {
        let raw = *self.depth_raw.get(v * self.width as usize + u)?;
        if raw == 0 {
            None
        } else {
            Some(raw as f32 * self.depth_units)
        }
    }

    /// Deproject an integer pixel into a metric camera-space point, or `None`
    /// if the pixel has no valid depth.
    #[inline]
    pub fn point_m(&self, u: usize, v: usize) -> Option<[f32; 3]> {
        let z = self.depth_m(u, v)?;
        Some(self.intrinsics.deproject(u as f32, v as f32, z))
    }
}

/// Streams color + depth from a RealSense D400-series device, aligning depth
/// into the color image each frame.
///
/// Created and polled on the tracking-worker thread, so it does not need
/// to be `Send`.
pub struct RealSenseCapture {
    // Field order is also drop order: the pipeline must be torn down before the
    // context that produced it, so `_context` is declared last.
    pipeline: ActivePipeline,
    align: Align,
    width: u32,
    height: u32,
    depth_units: f32,
    depth_units_cached: bool,
    timeout: Duration,
    _context: Context,
}

impl RealSenseCapture {
    /// Open the first connected D400-series device and start streaming color
    /// (`width`x`height` @ `fps`) plus depth, with depth aligned to color.
    pub fn open(width: u32, height: u32, fps: u32) -> Result<Self, String> {
        let context = Context::new().map_err(|e| format!("realsense: context: {e}"))?;

        let mut product = HashSet::new();
        product.insert(Rs2ProductLine::D400);
        let devices = context.query_devices(product);
        let device = devices
            .first()
            .ok_or_else(|| "realsense: no D400-series device found".to_string())?;

        let serial = device
            .info(Rs2CameraInfo::SerialNumber)
            .ok_or_else(|| "realsense: device reports no serial number".to_string())?;
        let name = device
            .info(Rs2CameraInfo::Name)
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "RealSense".to_string());

        let inactive = InactivePipeline::try_from(&context)
            .map_err(|e| format!("realsense: create pipeline: {e}"))?;

        let mut config = Config::new();
        config
            .enable_device_from_serial(serial)
            .map_err(|e| format!("realsense: enable device: {e}"))?
            .disable_all_streams()
            .map_err(|e| format!("realsense: disable all streams: {e}"))?
            .enable_stream(
                Rs2StreamKind::Depth,
                None,
                DEPTH_W,
                DEPTH_H,
                Rs2Format::Z16,
                fps as usize,
            )
            .map_err(|e| format!("realsense: enable depth stream: {e}"))?
            .enable_stream(
                Rs2StreamKind::Color,
                None,
                width as usize,
                height as usize,
                Rs2Format::Rgb8,
                fps as usize,
            )
            .map_err(|e| format!("realsense: enable color stream: {e}"))?;

        let pipeline = inactive
            .start(Some(config))
            .map_err(|e| format!("realsense: start pipeline: {e}"))?;

        // Align depth INTO the color image so color pixels index depth directly.
        let align =
            Align::new(Rs2StreamKind::Color, 10).map_err(|e| format!("realsense: align: {e}"))?;

        info!(
            "realsense: opened {} (serial {}) — color {}x{} @ {} fps, depth {}x{} aligned to color",
            name,
            serial.to_string_lossy(),
            width,
            height,
            fps,
            DEPTH_W,
            DEPTH_H,
        );

        Ok(Self {
            pipeline,
            align,
            width,
            height,
            depth_units: DEFAULT_DEPTH_UNITS_M,
            depth_units_cached: false,
            // Ceiling, not a fixed delay: `wait` returns the instant a frame is
            // ready. Generous enough to cover the D435's multi-hundred-ms first
            // frame after `start` (sensor warm-up / USB negotiation).
            timeout: Duration::from_millis(5000),
            _context: context,
        })
    }

    /// Grab one synchronized, color-aligned frame (blocking, with timeout).
    pub fn grab_frame(&mut self) -> Result<RealSenseFrame, String> {
        let frames = self
            .pipeline
            .wait(Some(self.timeout))
            .map_err(|e| format!("realsense: wait for frames: {e}"))?;

        // Cache the metric depth scale once, from the *raw* depth sensor — the
        // aligned (synthetic) frame may not expose a sensor. Do this before
        // `align.queue` takes ownership of `frames`.
        if !self.depth_units_cached {
            let raw_depth: Vec<DepthFrame> = frames.frames_of_type();
            if let Some(d) = raw_depth.first() {
                if let Ok(units) = d.depth_units() {
                    if units > 0.0 {
                        self.depth_units = units;
                        self.depth_units_cached = true;
                    }
                }
            }
        }

        self.align
            .queue(frames)
            .map_err(|e| format!("realsense: align.queue: {e}"))?;
        let aligned = self
            .align
            .wait(self.timeout)
            .map_err(|e| format!("realsense: align.wait: {e}"))?;

        let depth_frames: Vec<DepthFrame> = aligned.frames_of_type();
        let color_frames: Vec<ColorFrame> = aligned.frames_of_type();
        let depth = depth_frames
            .first()
            .ok_or_else(|| "realsense: aligned set has no depth frame".to_string())?;
        let color = color_frames
            .first()
            .ok_or_else(|| "realsense: aligned set has no color frame".to_string())?;

        let w = color.width();
        let h = color.height();
        if depth.width() != w || depth.height() != h {
            return Err(format!(
                "realsense: aligned size mismatch — color {}x{}, depth {}x{}",
                w,
                h,
                depth.width(),
                depth.height()
            ));
        }

        // Color -> RGB8 (row-major, top-down). We request Rgb8, but stay robust
        // to a Bgr8 negotiation.
        let mut rgb = Vec::with_capacity(w * h * 3);
        for pixel in color.iter() {
            match pixel {
                PixelKind::Rgb8 { r, g, b } => {
                    rgb.push(*r);
                    rgb.push(*g);
                    rgb.push(*b);
                }
                PixelKind::Bgr8 { b, g, r } => {
                    rgb.push(*r);
                    rgb.push(*g);
                    rgb.push(*b);
                }
                _ => {
                    rgb.push(0);
                    rgb.push(0);
                    rgb.push(0);
                }
            }
        }

        // Aligned depth -> raw Z16 (row-major, top-down; 0 = invalid).
        let mut depth_raw = Vec::with_capacity(w * h);
        for pixel in depth.iter() {
            match pixel {
                PixelKind::Z16 { depth } => depth_raw.push(*depth),
                _ => depth_raw.push(0),
            }
        }

        // Intrinsics of the color image == the aligned depth image.
        let intr = color
            .stream_profile()
            .intrinsics()
            .map_err(|e| format!("realsense: color intrinsics: {e}"))?;
        let intrinsics = CamIntrinsics {
            fx: intr.fx(),
            fy: intr.fy(),
            cx: intr.ppx(),
            cy: intr.ppy(),
            width: w as u32,
            height: h as u32,
        };

        self.width = w as u32;
        self.height = h as u32;

        Ok(RealSenseFrame {
            rgb,
            width: w as u32,
            height: h as u32,
            depth_raw,
            depth_units: self.depth_units,
            intrinsics,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}
