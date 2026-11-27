//! Immutable per-device selection. Experimental paths are opt-in and only on an AMD wave32 PS5 profile.
use anyhow::{Context, ensure};
#[derive(Clone, Copy, Debug)]
pub(super) struct Kernels { pub(super) wg: usize, pub(super) small: bool }
impl Kernels {
    pub(super) fn read(ps5: bool, wave64: bool, dot_accel: bool) -> anyhow::Result<Self> {
        let wg = std::env::var("STRATA_PS5_GEMV_WG").ok().map(|v| v.parse::<usize>()).transpose()
            .context("STRATA_PS5_GEMV_WG must be 64, 128 or 256")?.unwrap_or(256);
        ensure!([64, 128, 256].contains(&wg), "STRATA_PS5_GEMV_WG must be 64, 128 or 256");
        let small = std::env::var("STRATA_PS5_SMALL_GEMM").is_ok_and(|v| v == "1");
        if wg == 256 && !small { return Ok(Self { wg, small }); }
        ensure!(cfg!(feature = "ps5-r2"), "PS5 R2 shaders not built: use --features ps5-r2");
        ensure!(ps5 && !wave64 && dot_accel, "PS5 R2 requires AMD PS5 profile, wave32 and accelerated signed packed INT8 dot products");
        Ok(Self { wg, small })
    }
    pub(super) fn gemv(self, nt: usize, moe: bool) -> (&'static str, usize) {
        // No change to high-NT register-heavy or multi-token MoE deduplication paths.
        let wg = if nt <= 4 && (!moe || nt == 1) { self.wg } else { 256 };
        let name = match (moe, wg) {
            (false, 64) => "gemv_q_ps5_64", (false, 128) => "gemv_q_ps5_128",
            (true, 64) => "moe_gemv_q_ps5_64", (true, 128) => "moe_gemv_q_ps5_128",
            (false, _) => "gemv_q", (true, _) => "moe_gemv_q",
        };
        (name, wg / 32)
    }
    pub(super) fn small_dense(self, coop_i8: u32, tiles: u64, nd: usize) -> bool {
        self.small && coop_i8 == 0 && tiles == 0 && (1..=64).contains(&nd)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn dispatch_contract() {
        for wg in [64, 128, 256] {
            let k = Kernels { wg, small: true };
            for nt in 1..=17 {
                assert_eq!(k.gemv(nt, false).1, if nt <= 4 { wg/32 } else {8});
                assert_eq!(k.gemv(nt, true).1, if nt == 1 { wg/32 } else {8});
            }
            assert!(k.small_dense(0, 0, 64));
            assert!(!k.small_dense(0, 0, 65));
            assert!(!k.small_dense(16, 0, 32));
            assert!(!k.small_dense(0, 8, 32));
        }
    }
}
