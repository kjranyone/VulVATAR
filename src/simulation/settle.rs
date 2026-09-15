//! Settle-sleep gate shared by the spring-bone and cloth solvers.
//!
//! Both solvers keep residual per-step oscillation far below any
//! diagnostic's resolution when their driving inputs stop changing — a
//! float limit cycle, not real motion — yet that jitter is enough to
//! flip depth ties between nearly-coincident surfaces (the idle z-fight
//! flicker, 2026-09-15). A solver that has been quiet for
//! [`SETTLE_SLEEP_QUIET_FRAMES`] consecutive steps AND whose driving
//! inputs are bit-identical to the last stepped frame is skipped
//! entirely, freezing its vertex stream bit-stable. Any input change
//! (driver pose, gravity, tuning, colliders, the body-SDF field under
//! a spring chain) wakes it on the first frame.
//!
//! The "quiet" threshold is deliberately *not* zero: the point is not
//! "no motion" but "no motion the depth buffer can resolve differently
//! than last frame". Input changes that would move the solver by less
//! than [`SETTLE_SLEEP_EPS`] are below the scale the gate exists for.

/// Joint/particle motion per step below this counts as "quiet"
/// (metres). 100 µm is invisible motion — a fraction of a pixel — but
/// sits above the residual oscillation of a converged chain (measured
/// ~10 µm post-settle offline; the spring tail's sustained swing is
/// mm-scale and legitimately stays awake).
pub(crate) const SETTLE_SLEEP_EPS: f32 = 1e-4;
/// Consecutive quiet steps before a solver may sleep. With 1-2 substeps
/// per frame this is a few frames (~0.1 s) after the last real motion.
pub(crate) const SETTLE_SLEEP_QUIET_FRAMES: u32 = 5;

/// One quiet-frame update from the largest motion this step. Returns
/// `(quiet_frames, sleeping)` — `sleeping` is *re-evaluated*, never
/// latched: a chain woken by an input change (or a driver) falls back
/// to `false` as soon as it moves again, so a driver that stops
/// bit-stable mid-swing freezes nothing; the solver keeps stepping
/// until the swing decays below [`SETTLE_SLEEP_EPS`].
pub(crate) fn settle_bump(quiet_frames: u32, max_move: f32) -> (u32, bool) {
    let quiet_frames = if max_move < SETTLE_SLEEP_EPS {
        quiet_frames.saturating_add(1)
    } else {
        0
    };
    (quiet_frames, quiet_frames >= SETTLE_SLEEP_QUIET_FRAMES)
}

/// FNV-1a over u32 words — builds the input-identity fingerprints the
/// sleep gates compare. Full f32 bit precision: quantizing (the body-SDF
/// splat key hashes `to_bits() >> 10`) would hide real driver motion
/// below ~1 mm, exactly the band this gate must wake on.
pub(crate) struct SettleHasher(u64);

impl SettleHasher {
    pub(crate) fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    pub(crate) fn write_u32(&mut self, v: u32) {
        for byte in v.to_le_bytes() {
            self.0 ^= byte as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    pub(crate) fn write_f32(&mut self, v: f32) {
        self.write_u32(v.to_bits());
    }
    pub(crate) fn write_f32s(&mut self, vs: &[f32]) {
        for &v in vs {
            self.write_f32(v);
        }
    }
    pub(crate) fn write_bool(&mut self, v: bool) {
        self.write_u32(v as u32);
    }
    pub(crate) fn finish(self) -> u64 {
        self.0
    }
}

/// Per-cloth settle-sleep state (lives on `ClothState`). Same contract
/// as the spring-chain gate: sleep when quiet AND inputs unchanged,
/// wake on any input change. For GPU-backed cloth the sleep action is
/// `suppress_dispatch` — the snapshot's `substeps` is forced to 0 and
/// the renderer's existing frozen-frame contract (`compute_prepass`
/// skips every dispatch, pin write and version bump when substeps is
/// 0) keeps the persistent `cloth_pos_ssbo` bit-stable. For CPU-backed
/// cloth the CPU solver early-returns instead. `quiet_frames` /
/// `sleeping` / `last_max_delta` are fed from the one-frame-stale cloth
/// position readback (`apply_cloth_readback`) or, on the CPU backend,
/// from the solver's own per-step motion.
#[derive(Clone, Debug, Default)]
pub struct ClothSettleSleep {
    pub quiet_frames: u32,
    pub sleeping: bool,
    /// Fingerprint of every solver-relevant input of the last awake
    /// frame; `None` = never stepped.
    pub last_inputs: Option<u64>,
    /// Previous readback/step positions — the baseline the max-delta
    /// metric diffs against. GPU: the one-frame-stale position readback
    /// (`apply_cloth_readback`); CPU: implicit in `prev_position`.
    pub last_readback: Vec<crate::asset::Vec3>,
    /// Largest per-particle motion observed in the latest readback or
    /// CPU step (metres).
    pub last_max_delta: f32,
    /// Set by the app-side gate each frame; consumed by
    /// `collect_cloth_deforms` (GPU) and the CPU solver gate. Reset
    /// before every evaluation.
    pub suppress_dispatch: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settle_bump_counts_quiet_frames_and_resets_on_motion() {
        // Sub-eps motion accumulates toward sleep…
        let (q, sleep) = settle_bump(0, SETTLE_SLEEP_EPS * 0.5);
        assert_eq!((q, sleep), (1, false));
        let (q, sleep) = settle_bump(q, 0.0);
        assert_eq!((q, sleep), (2, false));
        let mut q = q;
        // Two more quiet bumps land on QUIET_FRAMES − 1 without sleeping;
        // the threshold itself is crossed by the bump after the loop.
        for _ in 2..SETTLE_SLEEP_QUIET_FRAMES - 1 {
            let (nq, sleep) = settle_bump(q, 0.0);
            assert!(!sleep);
            q = nq;
        }
        assert_eq!(q, SETTLE_SLEEP_QUIET_FRAMES - 1);
        // …sleeps exactly at the threshold…
        let (q, sleep) = settle_bump(SETTLE_SLEEP_QUIET_FRAMES - 1, 0.0);
        assert_eq!(q, SETTLE_SLEEP_QUIET_FRAMES);
        assert!(sleep);
        // …and any real motion resets both.
        let (q, sleep) = settle_bump(SETTLE_SLEEP_QUIET_FRAMES, SETTLE_SLEEP_EPS * 1.01);
        assert_eq!((q, sleep), (0, false));
    }

    #[test]
    fn settle_hasher_is_stable_and_input_sensitive() {
        let a = {
            let mut h = SettleHasher::new();
            h.write_f32(1.0 / 60.0);
            h.write_f32s(&[0.0, -9.81, 0.0]);
            h.write_bool(true);
            h.write_u32(4);
            h.finish()
        };
        let b = {
            let mut h = SettleHasher::new();
            h.write_f32(1.0 / 60.0);
            h.write_f32s(&[0.0, -9.81, 0.0]);
            h.write_bool(true);
            h.write_u32(4);
            h.finish()
        };
        assert_eq!(a, b);
        // One ULP on any input must change the hash — the wake key is
        // bit-precise by contract.
        let ulp = f32::from_bits(1.0_f32.to_bits() + 1);
        let mut h = SettleHasher::new();
        h.write_f32(ulp);
        h.write_f32s(&[0.0, -9.81, 0.0]);
        h.write_bool(true);
        h.write_u32(4);
        assert_ne!(h.finish(), a);
        let mut h = SettleHasher::new();
        h.write_f32(1.0 / 60.0);
        h.write_f32s(&[0.0, -9.81, 0.0]);
        h.write_bool(false);
        h.write_u32(4);
        assert_ne!(h.finish(), a);
    }
}
