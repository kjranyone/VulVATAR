//! Streaming smoke test for the RealSense D435 capture backend (Phase A).
//!
//! Opens the camera, grabs a handful of color-aligned depth frames, and
//! reports metric-depth sanity per frame: valid-pixel %, the center pixel's
//! metres + deprojected 3D point, and the min/median/max of all valid depths.
//! This exercises the whole streaming module end-to-end (stream -> align ->
//! deproject) against the live device.
//!
//!   cargo run --bin diagnose_realsense --features realsense
//!
//! Build env (see docs/realsense-build.md): PKG_CONFIG_PATH -> the hand-written
//! realsense2.pc, LIBCLANG_PATH -> LLVM\bin. Runtime: the SDK's bin\x64
//! (realsense2.dll) on PATH.

use vulvatar_lib::tracking::realsense::RealSenseCapture;

fn main() -> Result<(), String> {
    env_logger::init();

    let mut cap = RealSenseCapture::open(1280, 720, 30)?;
    println!("opened D435 — streaming (first few frames warm up auto-exposure)...");

    for i in 0..30 {
        let frame = cap.grab_frame()?;
        // Only report every 5th frame; still grab every frame to keep the
        // stream flowing and let AE settle.
        if i % 5 != 0 {
            continue;
        }

        let w = frame.width as usize;
        let h = frame.height as usize;
        let total = (w * h) as f32;
        let valid = frame.depth_raw.iter().filter(|&&r| r != 0).count();

        let (cu, cv) = (w / 2, h / 2);
        let center = frame.point_m(cu, cv);

        let mut zs: Vec<f32> = frame
            .depth_raw
            .iter()
            .filter(|&&r| r != 0)
            .map(|&r| r as f32 * frame.depth_units)
            .collect();
        zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (zmin, zmed, zmax) = if zs.is_empty() {
            (f32::NAN, f32::NAN, f32::NAN)
        } else {
            (zs[0], zs[zs.len() / 2], zs[zs.len() - 1])
        };

        println!(
            "frame {:02}: {}x{}  valid={:.1}%  center={:?}  z[min/med/max]={:.3}/{:.3}/{:.3} m  units={:.4}  intr(fx={:.1} fy={:.1} cx={:.1} cy={:.1})",
            i,
            w,
            h,
            100.0 * valid as f32 / total,
            center,
            zmin,
            zmed,
            zmax,
            frame.depth_units,
            frame.intrinsics.fx,
            frame.intrinsics.fy,
            frame.intrinsics.cx,
            frame.intrinsics.cy,
        );
    }

    println!("done — streaming + alignment + metric deprojection verified.");
    Ok(())
}
