//! Opt-in tiled prefill variants; never infer PS5 capabilities from the device name.
use anyhow::{Context, ensure};
#[derive(Clone, Copy, Debug)]
pub(super) struct Attention { pub(super) wg: usize }
impl Attention {
    pub(super) fn read(ps5: bool, wave64: bool, coop: bool, llama: bool, max_wg: u32, max_smem: u32) -> anyhow::Result<Self> {
        let wg = std::env::var("STRATA_PS5_ATTN_WG").ok().map(|v| v.parse::<usize>()).transpose()
            .context("STRATA_PS5_ATTN_WG must be 0, 256, 512 or 1024")?.unwrap_or(0);
        ensure!([0, 256, 512, 1024].contains(&wg), "STRATA_PS5_ATTN_WG must be 0, 256, 512 or 1024");
        if wg != 0 {
            ensure!(cfg!(feature = "ps5-r2"), "R5 shaders require --features ps5-r2");
            ensure!(ps5 && !wave64 && !coop && !llama, "R5 requires PS5 AMD wave32, no cooperative matrices and default numerics");
            ensure!(std::env::var_os("STRATA_VK_DP_OLD").is_none(), "R5 conflicts with STRATA_VK_DP_OLD");
            ensure!(wg <= max_wg as usize, "R5 workgroup {wg} exceeds device limit {max_wg}");
            let (name, _) = Self { wg }.kernel();
            for hd in [64, 128, 256] {
                super::spv_smem::check_shared_memory(name, &[0, hd], crate::cuda::kv_mode(), max_smem)?;
            }
        }
        Ok(Self { wg })
    }
    pub(super) fn kernel(self) -> (&'static str, usize) {
        match self.wg {
            256 => ("attn_prefill_ps5_256", 8),
            512 => ("attn_prefill_ps5_512", 16),
            1024 => ("attn_prefill_ps5_1024", 32),
            _ => ("attn_prefill_tile", 32),
        }
    }
}
