//! Bounded GPU fence waits.
//!
//! Every CPU-side wait on GPU work goes through [`wait_fence_bounded`]
//! instead of `FenceSignalFuture::wait(None)`. Rationale (2026-06-11
//! Intel Arc B570 freeze incident): an unbounded wait on a hung GPU
//! blocks the calling thread forever, freezing the app with zero
//! evidence of *which* GPU wait never returned — indistinguishable,
//! from the user's chair, from a whole-system hang.
//!
//! With the bound in place:
//! * a hung wait surfaces as an error after [`GPU_WAIT_TIMEOUT`]
//!   instead of blocking forever,
//! * the failure is stamped into the tracking stage log
//!   (`gpu_wait_timeout <label>` / `gpu_wait_device-lost <label>`)
//!   so post-incident forensics name the exact wait site,
//! * `VK_ERROR_DEVICE_LOST` is classified explicitly instead of being
//!   folded into a generic error string.
//!
//! ## Why the future is leaked on failure
//!
//! vulkano 0.35's `FenceSignalFuture::drop` runs
//! `fence.wait(None).unwrap()` (fence_signal.rs:538) when the future
//! is still in the `Flushed` state — on a hung device that re-blocks
//! forever, and on a lost device it panics. After a timeout the GPU
//! may also still be writing to buffers referenced by the submission,
//! so releasing them would be a use-after-free. `mem::forget` keeps
//! every referenced resource alive permanently and skips the blocking
//! drop: leaking a few MB against a dead device is the correct trade.

use std::time::Duration;

use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::GpuFuture;
use vulkano::{Validated, VulkanError};

/// Hard ceiling on any single CPU-side GPU wait. A healthy frame
/// completes in single-digit milliseconds; five seconds of silence
/// means the device is hung or lost, not slow.
pub(crate) const GPU_WAIT_TIMEOUT: Duration = Duration::from_secs(5);

/// Wait on `fence` with [`GPU_WAIT_TIMEOUT`], consuming the future.
///
/// On success the future is dropped normally (its state is already
/// `Cleaned`, so the drop is a no-op). On timeout / device loss /
/// other error the future is intentionally leaked (see module docs),
/// the failure is logged at ERROR and stamped into the tracking stage
/// log, and a descriptive error string is returned.
pub(crate) fn wait_fence_bounded<F>(
    fence: FenceSignalFuture<F>,
    label: &str,
) -> Result<(), String>
where
    F: GpuFuture,
{
    match fence.wait(Some(GPU_WAIT_TIMEOUT)) {
        Ok(()) => Ok(()),
        Err(e) => {
            let kind = match &e {
                Validated::Error(VulkanError::Timeout) => "timeout",
                Validated::Error(VulkanError::DeviceLost) => "device-lost",
                _ => "error",
            };
            // Leak before doing anything else so a panic in the
            // logging below can't reach the blocking drop.
            std::mem::forget(fence);
            crate::tracking::stagelog::mark(0, &format!("gpu_wait_{kind} {label}"));
            let msg = format!(
                "gpu wait '{label}' {kind} after {GPU_WAIT_TIMEOUT:?}: {e} — \
                 GPU driver hung or device lost; leaked the in-flight submission"
            );
            log::error!("{msg}");
            Err(msg)
        }
    }
}
