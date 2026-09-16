//! RealSense D400-series (D435) depth-camera capture backend.
//!
//! This module is only compiled when the `realsense` cargo feature is enabled
//! (see `docs/realsense-build.md` for the native `librealsense2` build wiring).
//!
//! It streams synchronized color + depth and, each frame, aligns the depth
//! image into the color image so that a color pixel indexes the matching depth
//! sample directly. Each grabbed frame is handed to the pipeline as:
//!   - an RGB8 buffer (`w*h*3`, row-major, top-down) so the pose provider
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

use log::{info, warn};
use realsense_rust::{
    config::Config,
    context::Context,
    frame::{ColorFrame, DepthFrame},
    kind::{Rs2CameraInfo, Rs2Format, Rs2Option, Rs2ProductLine, Rs2StreamKind},
    pipeline::{ActivePipeline, InactivePipeline},
    prelude::FrameEx,
    processing_blocks::{
        align::Align, options::TemporalFilterOptions, temporal_filter::TemporalFilter,
    },
};

/// Depth stream resolution requested from the device. 848x480 is a native
/// D435 depth mode; the align block resamples it into the color image, so the
/// depth handed downstream always ends up at the *color* resolution.
const DEPTH_W: usize = 848;
const DEPTH_H: usize = 480;

/// Default D435 metric depth scale (metres per raw Z16 unit = 1 mm). Used only
/// until the real value is read from the depth sensor on the first frame.
const DEFAULT_DEPTH_UNITS_M: f32 = 0.001;

/// Temporal-filter EMA weight of the *current* frame (librealsense
/// `FILTER_SMOOTH_ALPHA`, valid 0..=1). Intel's recommended default: high
/// enough that a moving hand converges within ~2 frames (67 ms at 30 fps),
/// low enough to cut the D435's static temporal jitter (~1 % of z) by
/// roughly half. Raising it weakens smoothing; lowering it adds visible
/// lag to fast limbs.
const TEMPORAL_SMOOTH_ALPHA: f32 = 0.4;

/// Per-pixel step (in raw Z16 units — 20 = 2 cm at the 1 mm default scale)
/// beyond which the temporal EMA *resets* instead of blending
/// (librealsense `FILTER_SMOOTH_DELTA`, valid 1..=100, Intel default 20).
/// This is what keeps the filter honest on moving edges: a hand sweeping
/// across the background jumps far past 2 cm per frame, so those pixels
/// restart from the fresh measurement rather than smearing a ghost.
const TEMPORAL_SMOOTH_DELTA: f32 = 20.0;

/// Escape hatch for A/B measurement (`.db3` replay comparisons, live
/// diagnostics): set this env var to any value before `open` to stream
/// unfiltered depth. Default (unset) = filter enabled.
const TEMPORAL_BYPASS_ENV: &str = "VULVATAR_RS_TEMPORAL_OFF";

/// Pinhole intrinsics of the color image (== the aligned depth image).
///
/// Lens distortion is ignored: the D435 color/aligned stream uses an
/// (Inverse-)Brown-Conrady model whose coefficients are typically zero, so a
/// pinhole deprojection is accurate to well under the depth noise floor.
/// This assumption is *verified at runtime* on the first grabbed frame —
/// a device reporting non-zero coefficients logs a warning instead of
/// silently biasing edge-of-frame joints.
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
    /// Device (hardware-clock) capture timestamp of the color frame, in
    /// milliseconds. This is THE time base for the whole tracking
    /// pipeline: it is carried through `SourceSkeleton.capture_timestamp_ms`
    /// so downstream filters derive `dt` from when frames were *captured*,
    /// not when some thread happened to process them.
    pub timestamp_ms: f64,
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
    /// Temporal noise filter applied to the *aligned* depth each frame.
    /// `None` when bypassed via [`TEMPORAL_BYPASS_ENV`], when construction
    /// failed at `open` (capture must outlive a broken filter block), or
    /// after a mid-stream filter error disabled it for the session.
    ///
    /// Placement note: librealsense's recommended order is filter-then-
    /// align, but `Align::queue` only accepts a `CompositeFrame` and the
    /// crate exposes no frameset composer, so a pre-align filtered depth
    /// frame cannot be recombined with its color frame. Filtering the
    /// aligned depth is equivalent for this use: static pixels map to
    /// stable color-grid coordinates (coherent EMA), and moving edges
    /// exceed `TEMPORAL_SMOOTH_DELTA` so the filter resets there exactly
    /// as it would pre-align.
    ///
    /// Downstream effect when enabled: per-pixel depth jitter roughly
    /// halves and brief single-frame holes are back-filled by the block's
    /// default persistence ("valid in 2 of the last 4 frames" — index 3,
    /// inside the conservative 2–3 band; the crate cannot set the option
    /// explicitly, so the librealsense default is relied upon and
    /// documented here). Downstream gates are relative / hold-based since
    /// the Round-2 rework, so no threshold recalibration is required —
    /// the replay harness verifies this with [`TEMPORAL_BYPASS_ENV`].
    temporal: Option<TemporalFilter>,
    /// One-shot latch for the first-frame distortion verification log.
    distortion_logged: bool,
    width: u32,
    height: u32,
    depth_units: f32,
    depth_units_cached: bool,
    /// Frames grabbed while `depth_units` was still the compiled-in
    /// default. Warned once past `DEPTH_UNITS_WARN_FRAMES` — a device
    /// whose depth unit was reconfigured (e.g. 100 µm by another tool)
    /// would otherwise mis-scale the whole body silently.
    depth_units_fallback_frames: u32,
    /// Nominal frame period from the requested fps; the depth↔color
    /// timestamp-consistency gate is half of this.
    frame_period_ms: f64,
    /// `wait` ceiling for the *first* frame only: the D435 takes multi-
    /// hundred-ms to produce it after `start` (sensor warm-up / USB
    /// negotiation).
    first_timeout: Duration,
    /// Steady-state `wait` ceiling. Deliberately short: the capture loop
    /// must re-check its `running` flag often enough that `stop()` never
    /// has to wait multiple seconds for a wedged camera to time out.
    steady_timeout: Duration,
    got_first_frame: bool,
    _context: Context,
}

/// Frames the app grabs while `depth_units` still carries the default —
/// past this, warn (once) that the device never reported its depth scale.
const DEPTH_UNITS_WARN_FRAMES: u32 = 30;

/// Why a single `grab_frame` call yielded no frame. Split so the caller
/// can tell transient per-frame conditions from capture-stream failures:
///
/// * `SyncMismatch` — the frameset arrived but its depth and color
///   timestamps diverge more than half a frame period (single-stream
///   drop / auto-exposure fps sag). The *set* is unusable (stale depth
///   under fresh keypoints teleports limbs) but the stream is healthy:
///   drop the set, don't count it toward reconnect logic.
/// * `Capture` — `wait` timed out or the driver errored; counts toward
///   the consecutive-failure reconnect policy.
#[derive(Debug, Clone)]
pub enum GrabError {
    Capture(String),
    SyncMismatch { delta_ms: f64 },
}

impl std::fmt::Display for GrabError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GrabError::Capture(msg) => write!(f, "{msg}"),
            GrabError::SyncMismatch { delta_ms } => write!(
                f,
                "realsense: depth/color timestamp mismatch ({delta_ms:.1} ms) — frameset dropped"
            ),
        }
    }
}

// Lets diagnostic bins keep `grab_frame()?` inside `-> Result<_, String>`.
impl From<GrabError> for String {
    fn from(e: GrabError) -> String {
        e.to_string()
    }
}

/// Why opening the D435 stream failed. Typed (rather than a flat `String`)
/// so the GUI can show a root-cause-specific dialog — above all the USB-2
/// link-speed case, which is the usual reason a healthy, enumerable D435
/// still can't start the 30 fps depth+color profiles this backend needs.
#[derive(Debug, Clone)]
pub enum OpenFailure {
    /// No D400-series device is connected at all.
    NoDevice,
    /// A device is present but negotiated a USB-2 link, so the requested
    /// high-rate profiles don't exist. `detected` is the reported USB type
    /// descriptor, e.g. `"2.1"`.
    UsbLinkTooSlow { detected: String },
    /// Any other librealsense failure; carries the raw driver message.
    Other(String),
}

impl std::fmt::Display for OpenFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenFailure::NoDevice => write!(f, "realsense: no D400-series device found"),
            OpenFailure::UsbLinkTooSlow { detected } => write!(
                f,
                "realsense: USB link negotiated at USB {detected} (needs USB 3.0) — the requested \
                 depth+color profiles are unavailable at this link speed"
            ),
            OpenFailure::Other(msg) => write!(f, "{msg}"),
        }
    }
}

// Lets the diagnostic bins keep `open(..)?` inside a `-> Result<_, String>`
// `main` without change.
impl From<OpenFailure> for String {
    fn from(e: OpenFailure) -> String {
        e.to_string()
    }
}

/// Enumerate every connected RealSense device (all product lines) for the
/// Tracking panel's device list — see [`crate::tracking::CameraDeviceInfo`].
/// Unlike the D400-filtered query inside [`RealSenseCapture::open`], this
/// deliberately uses an empty product mask (= any product line) so a
/// plugged-in but unsupported device (e.g. an L515) still shows up as
/// "connected, not usable" instead of vanishing.
pub(crate) fn enumerate_devices() -> Result<Vec<crate::tracking::CameraDeviceInfo>, String> {
    let context = Context::new().map_err(|e| format!("realsense: context: {e}"))?;
    Ok(context
        .query_devices(HashSet::new())
        .iter()
        .map(|device| {
            let name = device
                .info(Rs2CameraInfo::Name)
                .map(|s| s.to_string_lossy().trim().to_string())
                .unwrap_or_else(|| "RealSense".to_string());
            let serial = device
                .info(Rs2CameraInfo::SerialNumber)
                .map(|s| s.to_string_lossy().trim().to_string())
                .unwrap_or_default();
            let usb_type = device
                .info(Rs2CameraInfo::UsbTypeDescriptor)
                .map(|s| s.to_string_lossy().trim().to_string());
            crate::tracking::CameraDeviceInfo {
                supported: crate::tracking::d400_product_name(&name),
                name,
                serial,
                usb_type,
            }
        })
        .collect())
}

impl RealSenseCapture {
    /// Open a connected D400-series device and start streaming color
    /// (`width`x`height` @ `fps`) plus depth, with depth aligned to color.
    ///
    /// `preferred_serial` picks WHICH device when several are connected
    /// (the Tracking panel's radio selection, persisted in
    /// `settings.json`): the device with that serial is used when
    /// present, otherwise the first enumerated D400 — a saved selection
    /// outliving the camera it named must not brick capture.
    pub fn open(
        width: u32,
        height: u32,
        fps: u32,
        preferred_serial: Option<&str>,
    ) -> Result<Self, OpenFailure> {
        let context =
            Context::new().map_err(|e| OpenFailure::Other(format!("realsense: context: {e}")))?;

        let mut product = HashSet::new();
        product.insert(Rs2ProductLine::D400);
        let devices = context.query_devices(product);
        let device = preferred_serial
            .and_then(|want| {
                devices.iter().find(|d| {
                    d.info(Rs2CameraInfo::SerialNumber)
                        .is_some_and(|s| s.to_string_lossy() == want)
                })
            })
            .or_else(|| devices.first())
            .ok_or(OpenFailure::NoDevice)?;
        let serial = device
            .info(Rs2CameraInfo::SerialNumber)
            .map(|s| s.to_string_lossy().into_owned())
            .ok_or_else(|| {
                OpenFailure::Other("realsense: device reports no serial number".to_string())
            })?;
        if let Some(want) = preferred_serial {
            if want != serial {
                warn!("realsense: preferred serial {want} not connected — using {serial} instead");
            }
        }
        info!("realsense: using D400 device S/N {serial}");
        let name = device
            .info(Rs2CameraInfo::Name)
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "RealSense".to_string());

        let inactive = InactivePipeline::try_from(&context)
            .map_err(|e| OpenFailure::Other(format!("realsense: create pipeline: {e}")))?;

        let mut config = Config::new();
        // `enable_device_from_serial` takes a CStr; serials are plain
        // ASCII so the CString rebuild can't realistically fail.
        let serial_c = std::ffi::CString::new(serial.clone())
            .map_err(|e| OpenFailure::Other(format!("realsense: serial contains NUL: {e}")))?;
        config
            .enable_device_from_serial(&serial_c)
            .map_err(|e| OpenFailure::Other(format!("realsense: enable device: {e}")))?
            .disable_all_streams()
            .map_err(|e| OpenFailure::Other(format!("realsense: disable all streams: {e}")))?
            .enable_stream(
                Rs2StreamKind::Depth,
                None,
                DEPTH_W,
                DEPTH_H,
                Rs2Format::Z16,
                fps as usize,
            )
            .map_err(|e| OpenFailure::Other(format!("realsense: enable depth stream: {e}")))?
            .enable_stream(
                Rs2StreamKind::Color,
                None,
                width as usize,
                height as usize,
                Rs2Format::Rgb8,
                fps as usize,
            )
            .map_err(|e| OpenFailure::Other(format!("realsense: enable color stream: {e}")))?;

        // The usual reason a present, healthy D435 refuses to start at the
        // requested profile is a USB-2 link: the high-rate depth+color modes
        // simply don't exist below USB 3.0. On failure, read the negotiated
        // link speed and, if it's USB 2.x, report *that* as the root cause
        // instead of the opaque "config cannot be resolved" driver string.
        let pipeline = match inactive.start(Some(config)) {
            Ok(p) => p,
            Err(e) => {
                let usb = device
                    .info(Rs2CameraInfo::UsbTypeDescriptor)
                    .map(|s| s.to_string_lossy().trim().to_string());
                return Err(match usb {
                    Some(u) if u.starts_with('2') => OpenFailure::UsbLinkTooSlow { detected: u },
                    _ => OpenFailure::Other(format!("realsense: start pipeline: {e}")),
                });
            }
        };

        // Align depth INTO the color image so color pixels index depth directly.
        let align = Align::new(Rs2StreamKind::Color, 10)
            .map_err(|e| OpenFailure::Other(format!("realsense: align: {e}")))?;

        // Pin the color sensor to the requested frame rate. The D435 ships
        // with auto-exposure *priority* enabled, which silently halves the
        // color fps in dim rooms while depth keeps streaming at the full
        // rate — the resulting framesets pair fresh depth with stale color
        // (or vice versa) and trip the timestamp-consistency gate on every
        // frame. Priority OFF tells the sensor to keep the fps and accept a
        // darker image instead. Best-effort: an option a sensor doesn't
        // support is skipped, and failure to set it never blocks capture.
        for mut sensor in pipeline.profile().device().sensors() {
            if !sensor.supports_option(Rs2Option::AutoExposurePriority) {
                continue;
            }
            match sensor.set_option(Rs2Option::AutoExposurePriority, 0.0) {
                Ok(()) => info!("realsense: color auto-exposure priority disabled (fps pinned)"),
                Err(e) => warn!("realsense: could not disable auto-exposure priority: {e}"),
            }
        }

        // Read the metric depth scale straight off the depth sensor at open.
        // The per-frame frameset peek (below, in `grab_frame`) stays as the
        // fallback, but a device whose depth unit was reconfigured by another
        // tool (e.g. 100 µm) mis-scales *everything*, so the earliest and
        // most direct read wins.
        let mut depth_units = DEFAULT_DEPTH_UNITS_M;
        let mut depth_units_cached = false;
        for sensor in pipeline.profile().device().sensors() {
            if let Some(units) = sensor.get_option(Rs2Option::DepthUnits) {
                if units > 0.0 {
                    if (units - DEFAULT_DEPTH_UNITS_M).abs() > f32::EPSILON {
                        warn!(
                            "realsense: device depth scale is {units} m/unit (not the usual \
                             {DEFAULT_DEPTH_UNITS_M}) — honouring it; another tool likely \
                             reconfigured this camera"
                        );
                    } else {
                        info!("realsense: depth scale {units} m/unit (read from sensor at open)");
                    }
                    depth_units = units;
                    depth_units_cached = true;
                    break;
                }
            }
        }

        // Temporal depth filter — see the `temporal` field docs for placement
        // and downstream-effect notes. A filter that cannot be constructed or
        // configured must never take the capture down with it: fall back to
        // unfiltered streaming with a warning.
        let temporal = if std::env::var_os(TEMPORAL_BYPASS_ENV).is_some() {
            info!("realsense: temporal depth filter bypassed ({TEMPORAL_BYPASS_ENV} set)");
            None
        } else {
            match TemporalFilter::new(10) {
                Ok(mut tf) => match tf.apply_options(&TemporalFilterOptions {
                    smooth_alpha: Some(TEMPORAL_SMOOTH_ALPHA),
                    smooth_delta: Some(TEMPORAL_SMOOTH_DELTA),
                    // Deliberately None: the crate cannot set the persistence
                    // option (and warns on stdout if asked to). librealsense's
                    // default persistency index 3 ("valid in 2 / last 4") is
                    // already the conservative setting this pipeline wants.
                    persistence_control: None,
                }) {
                    Ok(()) => {
                        info!(
                            "realsense: temporal depth filter on (alpha {TEMPORAL_SMOOTH_ALPHA}, \
                             delta {TEMPORAL_SMOOTH_DELTA}, persistence: librealsense default)"
                        );
                        Some(tf)
                    }
                    Err(e) => {
                        warn!("realsense: temporal filter options rejected ({e}) — running unfiltered");
                        None
                    }
                },
                Err(e) => {
                    warn!("realsense: temporal filter unavailable ({e}) — running unfiltered");
                    None
                }
            }
        };

        info!(
            "realsense: opened {} (serial {}) — color {}x{} @ {} fps, depth {}x{} aligned to color",
            name, serial, width, height, fps, DEPTH_W, DEPTH_H,
        );

        Ok(Self {
            pipeline,
            align,
            temporal,
            distortion_logged: false,
            width,
            height,
            depth_units,
            depth_units_cached,
            depth_units_fallback_frames: 0,
            frame_period_ms: 1000.0 / (fps.max(1) as f64),
            // Ceilings, not fixed delays: `wait` returns the instant a frame
            // is ready. The first frame needs a generous ceiling (sensor
            // warm-up / USB negotiation take multi-hundred ms); steady state
            // stays short so a wedged camera can't hold `stop()` hostage —
            // the capture loop re-checks `running` after every timeout.
            first_timeout: Duration::from_millis(5000),
            steady_timeout: Duration::from_millis(500),
            got_first_frame: false,
            _context: context,
        })
    }

    /// Grab one synchronized, color-aligned frame (blocking, with timeout).
    pub fn grab_frame(&mut self) -> Result<RealSenseFrame, GrabError> {
        let timeout = if self.got_first_frame {
            self.steady_timeout
        } else {
            self.first_timeout
        };
        let frames = self
            .pipeline
            .wait(Some(timeout))
            .map_err(|e| GrabError::Capture(format!("realsense: wait for frames: {e}")))?;

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
            if !self.depth_units_cached {
                self.depth_units_fallback_frames += 1;
                if self.depth_units_fallback_frames == DEPTH_UNITS_WARN_FRAMES {
                    warn!(
                        "realsense: depth sensor never reported its depth scale after {} frames — \
                         assuming the default {} m/unit. If this device's depth unit was \
                         reconfigured, every distance will be mis-scaled.",
                        DEPTH_UNITS_WARN_FRAMES, DEFAULT_DEPTH_UNITS_M
                    );
                }
            }
        }

        self.align
            .queue(frames)
            .map_err(|e| GrabError::Capture(format!("realsense: align.queue: {e}")))?;
        let aligned = self
            .align
            .wait(timeout)
            .map_err(|e| GrabError::Capture(format!("realsense: align.wait: {e}")))?;

        let depth_frames: Vec<DepthFrame> = aligned.frames_of_type();
        let color_frames: Vec<ColorFrame> = aligned.frames_of_type();
        let depth = depth_frames.into_iter().next().ok_or_else(|| {
            GrabError::Capture("realsense: aligned set has no depth frame".into())
        })?;
        let color = color_frames.first().ok_or_else(|| {
            GrabError::Capture("realsense: aligned set has no color frame".into())
        })?;

        // Depth ↔ color capture-time consistency. librealsense's frameset
        // matcher is best-effort: when one stream drops a frame it happily
        // pairs the other stream with the previous capture. Sampling
        // yesterday's depth under today's keypoints teleports fast-moving
        // limbs, so a set whose two clocks diverge more than half a frame
        // period is dropped as a unit.
        let ts_color = color.timestamp();
        let ts_depth = depth.timestamp();
        let delta_ms = (ts_color - ts_depth).abs();
        if delta_ms > self.frame_period_ms * 0.5 {
            return Err(GrabError::SyncMismatch { delta_ms });
        }

        // Temporal noise filter on the aligned depth. Runs *after* the sync
        // gate so a stale mismatched set never enters the filter's history,
        // and on this capture thread so the inference side pays nothing.
        // The block preserves frame metadata (timestamp, profile), so the
        // gate result above stays valid for the filtered frame. take/put-
        // back: a filter that errors mid-stream stays disabled (taken) —
        // one dropped frame, then permanent unfiltered streaming, never a
        // capture-thread crash loop.
        let depth = if let Some(mut tf) = self.temporal.take() {
            let filtered = tf
                .queue(depth)
                .map_err(|e| format!("realsense: temporal queue: {e}"))
                .and_then(|()| {
                    tf.wait(timeout)
                        .map_err(|e| format!("realsense: temporal wait: {e}"))
                });
            match filtered {
                Ok(f) => {
                    self.temporal = Some(tf);
                    f
                }
                Err(e) => {
                    warn!("{e} — disabling the temporal filter for this session");
                    return Err(GrabError::Capture(e));
                }
            }
        } else {
            depth
        };

        let w = color.width();
        let h = color.height();
        if depth.width() != w || depth.height() != h {
            return Err(GrabError::Capture(format!(
                "realsense: aligned size mismatch — color {}x{}, depth {}x{}",
                w,
                h,
                depth.width(),
                depth.height()
            )));
        }

        // Color -> RGB8 via bulk row copies. The format was requested as
        // Rgb8 at `open`; verify what was actually negotiated once per
        // frame and hard-error on anything else — a silent black-fill
        // (the old per-pixel match's `_` arm) feeds the pose model an
        // empty image and turns a format surprise into an undebuggable
        // "tracking doesn't work".
        let color_fmt = color.stream_profile().format();
        let stride = color.stride();
        let rgb = match color_fmt {
            Rs2Format::Rgb8 | Rs2Format::Bgr8 => {
                let data = unsafe {
                    std::slice::from_raw_parts(
                        color.get_data() as *const _ as *const u8,
                        color.get_data_size(),
                    )
                };
                let mut rgb = vec![0u8; w * h * 3];
                for row in 0..h {
                    let src = &data[row * stride..row * stride + w * 3];
                    let dst = &mut rgb[row * w * 3..(row + 1) * w * 3];
                    dst.copy_from_slice(src);
                }
                if color_fmt == Rs2Format::Bgr8 {
                    for px in rgb.chunks_exact_mut(3) {
                        px.swap(0, 2);
                    }
                }
                rgb
            }
            other => {
                return Err(GrabError::Capture(format!(
                    "realsense: unsupported color format {other:?} (expected Rgb8/Bgr8)"
                )))
            }
        };

        // Aligned depth -> raw Z16 (row-major, top-down; 0 = invalid),
        // same bulk-copy treatment.
        let depth_fmt = depth.stream_profile().format();
        if depth_fmt != Rs2Format::Z16 {
            return Err(GrabError::Capture(format!(
                "realsense: unsupported depth format {depth_fmt:?} (expected Z16)"
            )));
        }
        let depth_stride = depth.stride();
        let depth_data = unsafe {
            std::slice::from_raw_parts(
                depth.get_data() as *const _ as *const u8,
                depth.get_data_size(),
            )
        };
        let mut depth_raw = vec![0u16; w * h];
        for row in 0..h {
            let src = &depth_data[row * depth_stride..row * depth_stride + w * 2];
            let dst = &mut depth_raw[row * w..(row + 1) * w];
            for (d, s) in dst.iter_mut().zip(src.chunks_exact(2)) {
                *d = u16::from_le_bytes([s[0], s[1]]);
            }
        }

        // Intrinsics of the color image == the aligned depth image.
        let intr = color
            .stream_profile()
            .intrinsics()
            .map_err(|e| GrabError::Capture(format!("realsense: color intrinsics: {e}")))?;

        // Verify (once) the pinhole assumption that `CamIntrinsics` bakes in:
        // the docs claim the D435 color coefficients are effectively zero,
        // but that was never checked against the actual device until now.
        if !self.distortion_logged {
            self.distortion_logged = true;
            let dist = intr.distortion();
            let max_coeff = dist.coeffs.iter().fold(0.0f32, |m, c| m.max(c.abs()));
            if max_coeff > 1e-6 {
                warn!(
                    "realsense: color stream reports non-zero distortion (model {:?}, coeffs \
                     {:?}) — the pinhole deprojection ignores these, so joints near the frame \
                     edge carry a small systematic offset on this device",
                    dist.model, dist.coeffs,
                );
            } else {
                info!(
                    "realsense: color distortion model {:?}, all coefficients zero — pinhole \
                     deprojection is exact",
                    dist.model,
                );
            }
        }
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
        self.got_first_frame = true;

        Ok(RealSenseFrame {
            rgb,
            width: w as u32,
            height: h as u32,
            depth_raw,
            depth_units: self.depth_units,
            intrinsics,
            timestamp_ms: ts_color,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// librealsense clamps out-of-range block options to *silently wrong*
    /// values on some SDK builds instead of erroring — pin the constants to
    /// their documented valid ranges so a future edit can't drift out.
    #[test]
    fn temporal_filter_constants_are_within_librealsense_ranges() {
        // FILTER_SMOOTH_ALPHA: 0..=1.
        assert!((0.0..=1.0).contains(&TEMPORAL_SMOOTH_ALPHA));
        // FILTER_SMOOTH_DELTA: 1..=100 (raw Z16 units).
        assert!((1.0..=100.0).contains(&TEMPORAL_SMOOTH_DELTA));
        // Delta must stay well below the person-band half-width (~0.45 m =
        // 450 raw units) — a delta that large would blend the body into
        // background occluders instead of resetting on them.
        assert!(TEMPORAL_SMOOTH_DELTA * DEFAULT_DEPTH_UNITS_M < 0.1);
    }

    /// The bypass env var name is part of the measurement workflow's
    /// contract (replay A/B harness) — lock it against accidental rename.
    #[test]
    fn temporal_bypass_env_name_is_stable() {
        assert_eq!(TEMPORAL_BYPASS_ENV, "VULVATAR_RS_TEMPORAL_OFF");
    }
}
