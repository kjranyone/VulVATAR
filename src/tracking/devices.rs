//! UI-facing camera enumeration and device classification.
//!
//! Mirrors the lipsync `AudioDeviceInfo` + stub pattern: the struct and
//! the pure classification helpers live here ungated so the GUI compiles
//! without the `realsense` feature; only `enumerate_cameras` has a
//! backend. Enumeration is meant for on-demand UI scans (device list +
//! Rescan button) — the capture path never uses it and keeps its own
//! D400-filtered `query_devices` inside `RealSenseCapture::open`
//! (`super::realsense`).
//!
//! Everything here is re-exported at the `crate::tracking` root.

/// One librealsense-enumerable camera, as shown in the Tracking panel's
/// device list. `supported` is false for non-D400 RealSense devices —
/// they enumerate, but the capture backend only drives D400 hardware.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CameraDeviceInfo {
    /// Device-reported product name, e.g. "Intel RealSense D435".
    pub name: String,
    pub serial: String,
    /// Negotiated USB link descriptor, e.g. "3.2" or "2.1". `None` when
    /// the device doesn't report it (some legacy firmware).
    pub usb_type: Option<String>,
    /// True for D400-series hardware the capture backend can stream.
    pub supported: bool,
}

/// Enumerate every connected RealSense device across ALL product lines
/// (the GUI must show a plugged-in L500 as "connected but unsupported",
/// not swallow it like the D400-filtered capture query does). Returns
/// `Err` with the SDK/driver message when the context itself fails —
/// that usually means the librealsense driver stack is broken, which
/// the user needs to see verbatim.
pub fn enumerate_cameras() -> Result<Vec<CameraDeviceInfo>, String> {
    #[cfg(feature = "realsense")]
    return super::realsense::enumerate_devices();
    #[cfg(not(feature = "realsense"))]
    Ok(Vec::new())
}

/// A D4xx product name — the D400 family this backend drives. Matches
/// names like "Intel RealSense D435" / "D455" without coupling to the
/// exact vendor prefix librealsense reports. Public only so the
/// realsense-gated enumerator can stamp `supported`.
pub fn d400_product_name(name: &str) -> bool {
    name.split_whitespace().any(|tok| {
        let b = tok.as_bytes();
        b.len() >= 3 && b[0] == b'D' && b[1] == b'4' && b[2].is_ascii_digit()
    })
}

/// USB-2 link, per the negotiated descriptor? Same rule as the
/// post-pipeline-failure diagnosis in `realsense::RealSenseCapture::open`
/// (`UsbLinkTooSlow`): anything starting with '2' is too slow for the
/// depth+color profiles the capture backend needs.
pub fn usb_link_too_slow(usb_type: Option<&str>) -> bool {
    usb_type.is_some_and(|u| u.trim().starts_with('2'))
}

/// Is any enumerated device usable by the capture backend — i.e. can
/// Start Camera do anything? Drives the button's enabled state so the
/// "No RealSense D435 found" dialog stops being the first sign of
/// trouble. A device on a USB-2 link doesn't count: it enumerates but
/// can't stream the required profiles.
pub fn usable_capture_device(devices: &[CameraDeviceInfo]) -> bool {
    devices
        .iter()
        .any(|d| d.supported && !usb_link_too_slow(d.usb_type.as_deref()))
}

#[cfg(test)]
mod camera_enum_tests {
    use super::*;

    fn d435(usb: Option<&str>) -> CameraDeviceInfo {
        CameraDeviceInfo {
            name: "Intel RealSense D435".into(),
            serial: "0123456789".into(),
            usb_type: usb.map(Into::into),
            supported: true,
        }
    }

    #[test]
    fn product_name_classification() {
        assert!(d400_product_name("Intel RealSense D435"));
        assert!(d400_product_name("Intel RealSense D455"));
        assert!(d400_product_name("RealSense D415"));
        assert!(!d400_product_name("Intel RealSense L515"));
        assert!(!d400_product_name("Intel RealSense SR305"));
        // "D4" alone is not a product token; the trailing digit matters.
        assert!(!d400_product_name("D4"));
    }

    #[test]
    fn usb2_link_detection() {
        assert!(usb_link_too_slow(Some("2.1")));
        assert!(usb_link_too_slow(Some("2.0")));
        assert!(usb_link_too_slow(Some(" 2.1")));
        assert!(!usb_link_too_slow(Some("3.2")));
        assert!(!usb_link_too_slow(Some("3.1")));
        assert!(!usb_link_too_slow(None));
    }

    #[test]
    fn usable_requires_supported_and_fast_link() {
        assert!(usable_capture_device(&[d435(Some("3.1"))]));
        assert!(usable_capture_device(&[
            d435(Some("2.1")),
            CameraDeviceInfo {
                name: "Intel RealSense D455".into(),
                serial: "aaa".into(),
                usb_type: Some("3.2".into()),
                supported: true,
            },
        ]));
        // Present but USB-2 — the exact "healthy camera, dead link" trap.
        assert!(!usable_capture_device(&[d435(Some("2.1"))]));
        // Unsupported product line doesn't count even on USB 3.
        assert!(!usable_capture_device(&[CameraDeviceInfo {
            name: "Intel RealSense L515".into(),
            serial: "aaa".into(),
            usb_type: Some("3.2".into()),
            supported: false,
        }]));
        assert!(!usable_capture_device(&[]));
    }
}
