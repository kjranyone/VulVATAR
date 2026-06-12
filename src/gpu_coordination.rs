//! Cross-subsystem GPU exclusivity coordination.
//!
//! Evidence from the 2026-06-12 stage log (`logs/tracking_*.log`):
//! the Arc B570 system froze ~240 ms into `provider_load_begin`,
//! while the Vulkan render thread was still submitting at 60 fps —
//! `provider_load_end` never arrived and the final log line died
//! mid-flush. DirectML session initialisation (uploading the 370 MB
//! RTMW3D-x graph and compiling its kernels) is the single heaviest
//! GPU operation in the app, and racing it against an active Vulkan
//! submission stream on one device is what hung the driver.
//!
//! This module gives heavyweight GPU initialisation a way to ask for
//! exclusivity. It is *cooperative*, not enforced: the tracking
//! worker holds a [`GpuExclusiveGuard`] across DirectML session
//! creation + first-inference warm-up, and the render thread checks
//! [`is_exclusive_active`] before doing GPU work, skipping frames
//! while the guard is held. The viewport pauses for the load's
//! duration (~1–3 s once per tracking start) instead of the driver
//! arbitrating two unrelated heavy clients.
//!
//! The counter nests so multiple init phases (or a future second
//! client) compose without coordination.

use std::sync::atomic::{AtomicUsize, Ordering};

static EXCLUSIVE_COUNT: AtomicUsize = AtomicUsize::new(0);

/// True while at least one [`GpuExclusiveGuard`] is alive. GPU
/// clients that can tolerate skipping work (the render thread) check
/// this each iteration.
pub fn is_exclusive_active() -> bool {
    EXCLUSIVE_COUNT.load(Ordering::Acquire) > 0
}

/// RAII request for cooperative GPU exclusivity. Held by heavyweight
/// GPU initialisation (DirectML session creation + warm-up); released
/// on drop, including panic unwinds.
pub struct GpuExclusiveGuard;

impl GpuExclusiveGuard {
    pub fn acquire(reason: &str) -> Self {
        let prev = EXCLUSIVE_COUNT.fetch_add(1, Ordering::AcqRel);
        log::info!(
            "gpu-coordination: exclusive section begin ({reason}), depth {}",
            prev + 1
        );
        crate::tracking::stagelog::mark(0, &format!("gpu_exclusive_begin {reason}"));
        Self
    }
}

impl Drop for GpuExclusiveGuard {
    fn drop(&mut self) {
        let prev = EXCLUSIVE_COUNT.fetch_sub(1, Ordering::AcqRel);
        log::info!(
            "gpu-coordination: exclusive section end, depth {}",
            prev.saturating_sub(1)
        );
        crate::tracking::stagelog::mark(0, "gpu_exclusive_end");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_nests_and_releases() {
        assert!(!is_exclusive_active());
        {
            let _a = GpuExclusiveGuard::acquire("test-a");
            assert!(is_exclusive_active());
            {
                let _b = GpuExclusiveGuard::acquire("test-b");
                assert!(is_exclusive_active());
            }
            assert!(is_exclusive_active());
        }
        assert!(!is_exclusive_active());
    }
}
