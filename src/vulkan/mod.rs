//! Vulkan backend on a dynamically loaded libvulkan (`ash`). Every buffer has its own allocation and a device
//! address, so `Buf` addresses are real pointers that kernels take as push constants. Ops are recorded into an open
//! command buffer, which a host copy or `sync` submits and waits for. Every op ends with a full compute and transfer
//! barrier, which also orders it before the work of later submissions.

use crate::backend::{BATCHED, ATT_SPLITS, ATT_SPLITS_1, Backend, Buf, GpuTensor, SAME_X};
use crate::quant::GgmlType;
use anyhow::{Context, anyhow, ensure};
use ash::vk;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, MutexGuard};

pub mod preflight;
pub mod spv_smem;
mod ps5_r2;
mod ps5_r3;
mod ps5_r5;

/// Host-visible buffer that host copies go through, in chunks of this size.
const STAGING: usize = 64 << 20;
/// Staged copies are submitted once this many bytes are queued: a submit per expert matrix costs more than the copy.
const SUBMIT_BYTES: usize = 4 << 20;

/// SPIR-V of every kernel, from `tools/gen_spirv.sh`.
const SPV: &[(&str, &[u8])] = &[
    ("add", include_bytes!("spv/add.spv")),
    ("argmax_rows", include_bytes!("spv/argmax_rows.spv")),
    ("attn_decode", include_bytes!("spv/attn_decode.spv")),
    ("attn_decode_llama", include_bytes!("spv/attn_decode_llama.spv")),
    #[cfg(feature = "ps5-r2")]
    ("attn_merge", include_bytes!(concat!(env!("OUT_DIR"), "/attn_merge_r7.spv"))),
    #[cfg(not(feature = "ps5-r2"))]
    ("attn_merge", include_bytes!("spv/attn_merge.spv")),
    ("attn_prefill_64", include_bytes!("spv/attn_prefill_64.spv")),
    ("attn_prefill_dp", include_bytes!("spv/attn_prefill_dp.spv")),
    ("attn_prefill_llama", include_bytes!("spv/attn_prefill_llama.spv")),
    #[cfg(feature = "ps5-r2")]
    ("attn_prefill_ps5_256", include_bytes!(concat!(env!("OUT_DIR"), "/attn_prefill_ps5_256.spv"))),
    #[cfg(feature = "ps5-r2")]
    ("attn_prefill_ps5_512", include_bytes!(concat!(env!("OUT_DIR"), "/attn_prefill_ps5_512.spv"))),
    #[cfg(feature = "ps5-r2")]
    ("attn_prefill_ps5_1024", include_bytes!(concat!(env!("OUT_DIR"), "/attn_prefill_ps5_1024.spv"))),
    ("attn_prefill_tile", include_bytes!("spv/attn_prefill_tile.spv")),
    ("attn_prefill_128", include_bytes!("spv/attn_prefill_128.spv")),
    ("attn_prefill_256", include_bytes!("spv/attn_prefill_256.spv")),
    ("conv_silu", include_bytes!("spv/conv_silu.spv")),
    ("conv_seq", include_bytes!("spv/conv_seq.spv")),
    ("conv_state", include_bytes!("spv/conv_state.spv")),
    ("deltanet", include_bytes!("spv/deltanet.spv")),
    ("dequant", include_bytes!("spv/dequant.spv")),
    ("dflash2_conv", include_bytes!("spv/dflash2_conv.spv")),
    ("gemm_f", include_bytes!("spv/gemm_f.spv")),
    ("gemm_f_dp", include_bytes!("spv/gemm_f_dp.spv")),
    #[cfg(feature = "ps5-r2")]
    ("gemm_f_dp_ps5_iq32", include_bytes!(concat!(env!("OUT_DIR"), "/gemm_f_dp_ps5_iq32.spv"))),
    ("gemm_fs", include_bytes!("spv/gemm_fs.spv")),
    ("gemmi", include_bytes!("spv/gemmi.spv")),
    ("gemmi_16", include_bytes!("spv/gemmi_16.spv")),
    ("gemmi_dp", include_bytes!("spv/gemmi_dp.spv")),
    ("gemmi_s", include_bytes!("spv/gemmi_s.spv")),
    ("gemm_red", include_bytes!("spv/gemm_red.spv")),
    ("gemmi_s_16", include_bytes!("spv/gemmi_s_16.spv")),
    ("gemmi_s2", include_bytes!("spv/gemmi_s2.spv")),
    ("gemmi_s2_16", include_bytes!("spv/gemmi_s2_16.spv")),
    ("gemv_f", include_bytes!("spv/gemv_f.spv")),
    ("gemv_q", include_bytes!("spv/gemv_q.spv")),
    #[cfg(feature = "ps5-r2")]
    ("gemv_q_ps5_64", include_bytes!(concat!(env!("OUT_DIR"), "/gemv_q_ps5_64.spv"))),
    #[cfg(feature = "ps5-r2")]
    ("gemv_q_ps5_128", include_bytes!(concat!(env!("OUT_DIR"), "/gemv_q_ps5_128.spv"))),
    #[cfg(feature = "ps5-r2")]
    ("moe_gemv_q_ps5_64", include_bytes!(concat!(env!("OUT_DIR"), "/moe_gemv_q_ps5_64.spv"))),
    #[cfg(feature = "ps5-r2")]
    ("moe_gemv_q_ps5_128", include_bytes!(concat!(env!("OUT_DIR"), "/moe_gemv_q_ps5_128.spv"))),
    #[cfg(feature = "ps5-r2")]
    ("gemmi_dp_ps5_iq32", include_bytes!(concat!(env!("OUT_DIR"), "/gemmi_dp_ps5_iq32.spv"))),
    #[cfg(feature = "ps5-r2")]
    ("gemmi_dp_ps5_iq64", include_bytes!(concat!(env!("OUT_DIR"), "/gemmi_dp_ps5_iq64.spv"))),
    #[cfg(feature = "ps5-r2")]
    ("gemmi_dp_ps5_64", include_bytes!(concat!(env!("OUT_DIR"), "/gemmi_dp_ps5_64.spv"))),
    #[cfg(feature = "ps5-r2")]
    ("gemmi_dp_ps5_32", include_bytes!(concat!(env!("OUT_DIR"), "/gemmi_dp_ps5_32.spv"))),
    ("hadamard", include_bytes!("spv/hadamard.spv")),
    ("hc_mix", include_bytes!("spv/hc_mix.spv")),
    ("hc_silu", include_bytes!("spv/hc_silu.spv")),
    ("hc_write", include_bytes!("spv/hc_write.spv")),
    ("ink_attn", include_bytes!("spv/ink_attn.spv")),
    ("ink_attn_merge", include_bytes!("spv/ink_attn_merge.spv")),
    ("ink_attn_prefill", include_bytes!("spv/ink_attn_prefill.spv")),
    ("attn_sel", include_bytes!("spv/attn_sel.spv")),
    ("vit_attn", include_bytes!("spv/vit_attn.spv")),
    ("ink_clip", include_bytes!("spv/ink_clip.spv")),
    ("ink_conv", include_bytes!("spv/ink_conv.spv")),
    ("ink_conv_batch", include_bytes!("spv/ink_conv_batch.spv")),
    ("ink_ctrl_gate", include_bytes!("spv/ink_ctrl_gate.spv")),
    ("ink_ctrl_out", include_bytes!("spv/ink_ctrl_out.spv")),
    ("ink_ffn_sum", include_bytes!("spv/ink_ffn_sum.spv")),
    ("ink_mix", include_bytes!("spv/ink_mix.spv")),
    ("ink_route", include_bytes!("spv/ink_route.spv")),
    ("kv_store", include_bytes!("spv/kv_store.spv")),
    ("l2_norm", include_bytes!("spv/l2_norm.spv")),
    ("moe_combine", include_bytes!("spv/moe_combine.spv")),
    ("moe_down_combine", include_bytes!("spv/moe_down_combine.spv")),
    ("moe_combine_seq", include_bytes!("spv/moe_combine_seq.spv")),
    ("moe_gemv_f", include_bytes!("spv/moe_gemv_f.spv")),
    #[cfg(feature = "ps5-r2")]
    ("moe_gemv_q", include_bytes!(concat!(env!("OUT_DIR"), "/moe_gemv_q_audit.spv"))),
    #[cfg(not(feature = "ps5-r2"))]
    ("moe_gemv_q", include_bytes!("spv/moe_gemv_q.spv")),
    #[cfg(feature = "ps5-r2")]
    ("moe_route", include_bytes!(concat!(env!("OUT_DIR"), "/moe_route_r7.spv"))),
    #[cfg(not(feature = "ps5-r2"))]
    ("moe_route", include_bytes!("spv/moe_route.spv")),
    ("norm_gate", include_bytes!("spv/norm_gate.spv")),
    ("ple_conv", include_bytes!("spv/ple_conv.spv")),
    ("ple_gate", include_bytes!("spv/ple_gate.spv")),
    ("quant_q8", include_bytes!("spv/quant_q8.spv")),
    ("quant_xq", include_bytes!("spv/quant_xq.spv")),
    ("qsa_pool", include_bytes!("spv/qsa_pool.spv")),
    ("qsa_score", include_bytes!("spv/qsa_score.spv")),
    ("qsa_select", include_bytes!("spv/qsa_select.spv")),
    ("rms_norm", include_bytes!("spv/rms_norm.spv")),
    ("rope_neox", include_bytes!("spv/rope_neox.spv")),
    ("router_topk", include_bytes!("spv/router_topk.spv")),
    ("ssm_gates", include_bytes!("spv/ssm_gates.spv")),
    ("situ", include_bytes!("spv/situ.spv")),
    ("situ_q", include_bytes!("spv/situ_q.spv")),
    ("softcap", include_bytes!("spv/softcap.spv")),
    ("swiglu", include_bytes!("spv/swiglu.spv")),
    // quant_xq under another name, so that it gets no barrier inside a batch: its output is read after the batch.
    ("swiglu_q", include_bytes!("spv/quant_xq.spv")),
];

/// The device, a scratch buffer of XQ bytes for x quantized by `quant_xq` when an op does that itself, the IQ lookup
/// tables (`iq_tables`), the growable q8 rows for `gemmi` prefill, a fixed q8 buffer for `<= TMAX` graphed forwards,
/// and the split-K slices of batched-decode GEMMs (`SPLIT_SCR` bytes, made at first use).
pub struct Vk(Arc<Inner>, Buf, Buf, Mutex<(Buf, Option<[u64; 4]>)>, Mutex<(Buf, Option<[u64; 4]>)>, std::sync::OnceLock<Buf>);

/// Split-K scratch of a batched GEMM: ks * nd * rows floats, ks chosen so ks * workgroups <= SPLIT_WG (so it fits:
/// rows <= 64 g0 and nd <= 64 g1). 224, 448 and 896 measured alike at 16 rows; 448 was best at 64.
const SPLIT_WG: usize = 448;
const SPLIT_SCR: usize = SPLIT_WG * 64 * 64 * 4;

/// The IQ lookup tables in one buffer, and the `tables.glsl` that names each at its offset from the buffer's address
/// (spec constants 3 and 4). As GLSL const arrays the tables went to per-thread local memory on NVIDIA, which made
/// IQ2_XS expert GEMVs about 200 times slower than on CUDA.
pub fn iq_tables() -> (Vec<u8>, String) {
    use crate::quant::tables::*;
    let (mut b, mut g) = (Vec::new(), String::from("\
// Generated by vulkan::iq_tables via tests/backend.rs glsl_tables_are_fresh (STRATA_WRITE_TABLES=1).
layout(constant_id = 3) const uint TAB_LO = 0;
layout(constant_id = 4) const uint TAB_HI = 0;
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer Tab64 { uint64_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer Tab32 { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer TabI { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer TabF { float v[]; };
#define TAB (uint64_t(TAB_LO) | uint64_t(TAB_HI) << 32)
"));
    let mut add = |name: &str, ty: &str, v: Vec<u8>| { g += &format!("#define {name} {ty}(TAB + {}ul).v\n", b.len()); b.extend(v) };
    let le64 = |v: &[u64]| v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let le32 = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect();
    add("iq2xs_grid", "Tab64", le64(&IQ2XS_GRID));
    add("iq2xxs_grid", "Tab64", le64(&IQ2XXS_GRID));
    add("iq2s_grid", "Tab64", le64(&IQ2S_GRID));
    add("iq1s_grid", "Tab64", le64(&IQ1S_GRID));
    add("iq3xxs_grid", "Tab32", le32(&IQ3XXS_GRID));
    add("iq3s_grid", "Tab32", le32(&IQ3S_GRID));
    add("ksigns_iq2xs", "Tab32", le32(&KSIGNS_IQ2XS.map(u32::from)));
    add("kvalues_iq4nl", "TabI", KVALUES_IQ4NL.iter().flat_map(|&x| i32::from(x).to_le_bytes()).collect());
    add("tq_lut", "TabF", crate::quant::TQ_LUT.iter().flatten().flat_map(|x| x.to_le_bytes()).collect());
    (b, g)
}

/// Memory types to allocate from, in order of preference (`Inner::buffer`). Flags: 1 device-local, 2 host-visible,
/// 4 host-coherent, 8 host-cached.
const DEVICE: &[vk::MemoryPropertyFlags] = &[vk::MemoryPropertyFlags::from_raw(1)];
// Fresh immutable weights only: device local, host visible and coherent; no AMD uncached flags.
const MAPPED_DEVICE: &[vk::MemoryPropertyFlags] = &[vk::MemoryPropertyFlags::from_raw(15), vk::MemoryPropertyFlags::from_raw(7)];
/// Host memory: cached if possible, as CUDA's pinned memory (host reads of uncached memory are slow).
const HOST: &[vk::MemoryPropertyFlags] = &[vk::MemoryPropertyFlags::from_raw(0xe), vk::MemoryPropertyFlags::from_raw(6)];
/// Device memory the host maps through the PCIe BAR, for the CPU expert mailbox. A kernel polling host memory for the
/// worker's answer can keep reading a stale copy (in L2 on NVIDIA) for hundreds of milliseconds, even with an acquire
/// load; device memory written through the BAR is seen at once. Host reads of it are uncached and slow.
const BAR: &[vk::MemoryPropertyFlags] = &[vk::MemoryPropertyFlags::from_raw(7), vk::MemoryPropertyFlags::from_raw(0xe),
    vk::MemoryPropertyFlags::from_raw(6)];
/// `BAR` with VK_AMD_device_coherent_memory: AMD's GPU caches device memory without seeing host writes through the
/// BAR (and holds its own writes from the host), so the mailbox is memory the GPU does not cache (flags 0x40, 0x80).
const BAR_AMD: &[vk::MemoryPropertyFlags] = &[vk::MemoryPropertyFlags::from_raw(0xc7), vk::MemoryPropertyFlags::from_raw(0xc6),
    vk::MemoryPropertyFlags::from_raw(7), vk::MemoryPropertyFlags::from_raw(0xe), vk::MemoryPropertyFlags::from_raw(6)];

/// How long a waiting submit polls its fence before it sleeps in the driver.
const SPIN: std::time::Duration = std::time::Duration::from_micros(300);
/// How long the host waits for a submission before it tells MoE combines still waiting for the CPU worker to give up
/// (err[2], see `Inner::submit`).
const GIVE_UP: std::time::Duration = std::time::Duration::from_secs(1);

/// Bytes of the quantized-x scratch buffer.
const XQ: usize = 4 << 20;

struct Inner {
    _entry: ash::Entry,
    instance: ash::Instance,
    pd: vk::PhysicalDevice,
    dev: ash::Device,
    queue: vk::Queue,
    /// A transfer-only queue (copy engines) for the copies inside `beside`, unless STRATA_VK_NO_TQ; its family and
    /// the compute family, which every buffer is shared between.
    tq: Option<vk::Queue>,
    qfs: [u32; 2],
    /// Timeline semaphores signalled by each compute submit (`State::cval`) and transfer submit (`State::tval`).
    csem: vk::Semaphore,
    tsem: vk::Semaphore,
    mem: vk::PhysicalDeviceMemoryProperties,
    budget: bool,
    /// Integrated GPU (APU sharing unified system LPDDR memory between device and host heaps).
    igpu: bool,
    ps5: bool,
    r2: ps5_r2::Kernels,
    r3: Option<ps5_r3::HostImport>,
    ps5_gtt: bool,
    ps5_direct: bool,
    ps5_host_reserve: u64,
    ps5_upload_bytes: [std::sync::atomic::AtomicU64; 2],
    // R6 records only first actual dispatch/allocation per class, never pipeline creation.
    r6_trace: bool,
    r6_seen: std::sync::atomic::AtomicU32,
    /// AMD RDNA3 / RDNA3.5 (gfx11xx) or iGPU whose small L2 cache hurts from next-layer MoE prefetch.
    gfx11: bool,
    /// The device runs fixed 64-lane subgroups, split into logical 32-lane warps.
    wave64: bool,
    /// Use double-precision quantizer reciprocals (`RCP_D`, spec constant 13) on AMD devices even in wave32.
    rcp_d: bool,
    /// VK_KHR_cooperative_matrix with the 16 x 16 x 16 f16 / f32 shape is enabled (`gemm_f`, `attn_prefill`).
    coop: bool,
    /// Without cooperative matrices: run the tiled `attn_prefill_tile` (one KV-head tile shared by 32 query/head tasks)
    /// when the device allows it. STRATA_VK_DP_OLD=1 keeps the older per-query `attn_prefill_dp`.
    dp_tile: bool,
    r5: ps5_r5::Attention,
    r5_logged: std::sync::atomic::AtomicBool,
    /// Int8 cooperative matrix K-dimension (`32` for NVIDIA `16x16x32`, `16` for AMD RDNA3/3.5 `16x16x16`, `0` = off).
    coop_i8: u32,
    /// VK_AMD_device_coherent_memory is enabled (the mailbox uses `BAR_AMD`).
    amd_coherent: bool,
    /// An AMD GPU without it: the CPU expert mailbox cannot work (the GPU caches it), so a timeout names the fix.
    amd_plain: bool,
    /// VK_EXT_conditional_rendering, unless STRATA_VK_NO_COND or its self-test failed: a MoE combine whose wait for
    /// the CPU worker gave up skips the rest of the submission (`skip_rest`), and the host finishes the forward.
    cr: Option<ash::ext::conditional_rendering::Device>,
    /// Specialization constant 0 of kernels that quantize x: llama.cpp q8_K numerics (`--llama-numerics`).
    llama: u32,
    /// Specialization constants 15 and 16 of every kernel: independent K and V cache modes
    /// (0 = f16, 1 = q8, 2 = q4, 3 = q2, 4 = turbo4, 5 = turbo3, 6 = turbo2).
    kv_k_mode: u32,
    kv_v_mode: u32,
    /// `maxComputeSharedMemorySize` of the device (or STRATA_VK_SMEM_LIMIT, a test hook): no pipeline may need more
    /// `Workgroup` bytes than this (`Vk::pipeline`), or the driver accepts it and faults at dispatch (Xid 13 on Volta).
    smem_limit: u32,
    max_dispatch: [u32; 3],
    /// Nanoseconds per GPU timestamp tick (`VkPhysicalDeviceLimits::timestampPeriod`).
    ts_period: f32,
    r7_trace: bool,
    r7_commands: R7Commands,
    /// STRATA_VK_STATS: print the driver's statistics (registers, ...) of each pipeline as it is created.
    stats: bool,
    /// The one pipeline layout: a 128-byte push constant range (the Vulkan minimum), no descriptor sets.
    layout: vk::PipelineLayout,
    st: Mutex<State>,
}

struct State {
    /// Command buffers for ops outside graphs, recycled after each wait.
    pool: vk::CommandPool,
    spare: Vec<vk::CommandBuffer>,
    /// Submitted since the last wait.
    used: Vec<vk::CommandBuffer>,
    /// Closed command buffers and graph replays queued in execution order for the next `submit`.
    queued: Vec<vk::CommandBuffer>,
    /// The open command buffer.
    cur: Option<vk::CommandBuffer>,
    /// The command buffer with conditional rendering on (`cond_on`), ended before the buffer is.
    cond: Option<vk::CommandBuffer>,
    /// Conditional rendering failed the self-test in `Vk::new`: not used.
    no_cond: bool,
    /// Set by `skip_rest` when a non-fast-path MoE combine (`mb != 0`) uses conditional rendering.
    need_cond: bool,
    /// A MoE combine gave up waiting for the CPU worker since the last `stalled` (see `Backend::stalled`).
    stalled: bool,
    /// Work was submitted since the last wait.
    busy: bool,
    /// Graph command buffers, kept for the backend's life.
    gpool: vk::CommandPool,
    graphs: HashMap<u64, vk::CommandBuffer>,
    /// The graph being recorded; ops go there instead of `cur`.
    rec: Option<vk::CommandBuffer>,
    /// Inside `batch`: dispatches get no barrier after them (except quant_xq, read by the GEMV after it). quant_xq
    /// output has one scratch buffer, so at most one GEMV in a batch may quantize its own input.
    batch: bool,
    /// Inside `beside` (with a transfer queue): staged copies go to `tcur` on the transfer queue, after the compute
    /// submit `bwait` (the work before `beside`).
    beside: bool,
    tpool: vk::CommandPool,
    tcur: Option<vk::CommandBuffer>,
    tspare: Vec<vk::CommandBuffer>,
    tused: Vec<vk::CommandBuffer>,
    /// Bytes copied in `tcur`, from `staging` at `tstart` on.
    tbytes: usize,
    tstart: usize,
    /// Submitted transfer command buffers that may still read `staging`: (start, end, `tsem` value when done). The
    /// copies inside `beside` use `staging` as a ring, waiting only for these; `cstaged`: a compute-queue copy reads
    /// it too (until the next wait).
    tring: std::collections::VecDeque<(usize, usize, u64)>,
    cstaged: bool,
    /// Last values signalled on `csem` and `tsem`; `bwait` as above; the next compute submit waits for `tsem` to
    /// reach `twait` (0: no wait).
    cval: u64,
    tval: u64,
    bwait: u64,
    twait: u64,
    /// `tval` after the last `beside`: the compute submit after the next `beside` waits for it.
    tlag: u64,
    pipes: HashMap<(&'static str, Vec<u32>), vk::Pipeline>,
    fence: vk::Fence,
    staging: (vk::Buffer, vk::DeviceMemory),
    /// Mapped `staging` (a pointer kept as an integer so the state is `Send`).
    map: usize,
    /// Bytes of `staging` that queued host-to-device copies still read; freed by the next wait.
    staged: usize,
    /// `open_cb` calls; `copied` holds its value after the last staged copy and the staged bytes in `cur` not yet
    /// submitted, so `htod` knows when `cur` holds nothing but its own copies.
    ops: usize,
    copied: (usize, usize),
    /// Live buffers by start address: (buffer, bytes).
    bufs: BTreeMap<u64, (vk::Buffer, usize)>,
    /// Host-mapped words: the error that `moe_combine` sets when it gives up waiting for the CPU worker, then the
    /// polls the combines spent waiting since the last wait here (see mb_wait.glsl). Buffer, memory, host pointer,
    /// device address.
    err: (vk::Buffer, vk::DeviceMemory, usize, u64),
    /// STRATA_VK_PROF: timestamps before and after each dispatch outside graphs (its own time, without the copies and
    /// waits before it), names in order, and per-name totals (ns, calls)
    /// printed on drop.
    prof: Option<(vk::QueryPool, Vec<&'static str>, HashMap<&'static str, (u64, usize)>)>,
    /// The prefill position last read from a `step` buffer, and that buffer's address. A host copy into the buffer
    /// clears it, so a chunk reads it once, not once per layer.
    step_pos: Option<(u64, usize)>,
}

#[derive(Default)]
struct R7Commands {
    dispatch: std::sync::atomic::AtomicU64,
    barrier: std::sync::atomic::AtomicU64,
    conditional_barrier: std::sync::atomic::AtomicU64,
    graph_create: std::sync::atomic::AtomicU64,
    graph_reuse: std::sync::atomic::AtomicU64,
    transfer_submit: std::sync::atomic::AtomicU64,
    buffer_alloc: std::sync::atomic::AtomicU64,
    buffer_drop: std::sync::atomic::AtomicU64,
    upload_bytes: std::sync::atomic::AtomicU64,
    download_bytes: std::sync::atomic::AtomicU64,
    device_copy_bytes: std::sync::atomic::AtomicU64,
}

/// Owner of one `Buf`: the buffer and its memory, freed on drop.
struct Mem { vk: Arc<Inner>, buf: vk::Buffer, mem: vk::DeviceMemory, addr: u64, imported: Option<(Arc<crate::gguf::Map>, u64, crate::weights::ImportedRange)> }

impl Drop for Mem {
    fn drop(&mut self) {
        let mut st = self.vk.st();
        let flushed = self.vk.flush(&mut st); // queued work may still use the buffer
        if self.imported.is_some() && flushed.is_err() {
            // Never release host pages while GPU access may still be in flight. On an
            // exceptional wait failure retain the mapping for the remaining process lifetime.
            if unsafe { self.vk.dev.device_wait_idle() }.is_err() {
                if let Some(imported) = self.imported.take() { std::mem::forget(imported); }
                std::mem::forget(self.vk.clone());
                return;
            }
        }
        st.bufs.remove(&self.addr);
        if self.vk.r7_trace { self.vk.r7_commands.buffer_drop.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
        unsafe { self.vk.dev.destroy_buffer(self.buf, None); self.vk.dev.free_memory(self.mem, None) }
        if let Some((_, bytes, _)) = &self.imported {
            if let Some(r3) = &self.vk.r3 { r3.release(*bytes); }
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let st = self.st.get_mut().unwrap_or_else(|e| e.into_inner());
        if let Some(r3) = &self.r3 { r3.report(); }
        if self.ps5 {
            let [mapped, staged] = self.ps5_upload_bytes.each_ref().map(|v| v.load(std::sync::atomic::Ordering::Relaxed));
            eprintln!("PS5 fresh weights: mapped {mapped} bytes, staged {staged} bytes");
        }
        if self.r7_trace {
            use std::sync::atomic::Ordering::Relaxed;
            let c = &self.r7_commands;
            for (name, counter) in [
                ("dispatch_recorded", &c.dispatch), ("barrier_recorded", &c.barrier),
                ("conditional_barrier_recorded", &c.conditional_barrier), ("graph_created", &c.graph_create),
                ("graph_reused", &c.graph_reuse), ("transfer_submit", &c.transfer_submit),
                ("buffer_alloc", &c.buffer_alloc), ("buffer_drop", &c.buffer_drop),
                ("upload_bytes", &c.upload_bytes), ("download_bytes", &c.download_bytes),
                ("device_copy_bytes", &c.device_copy_bytes),
            ] {
                eprintln!("r7_count,{name},{}", counter.load(Relaxed));
            }
        }
        if let Some((_, _, tot)) = &st.prof {
            let scale = self.ts_period as f64;
            let mut v: Vec<_> = tot.iter().collect(); v.sort_by_key(|x| std::cmp::Reverse(x.1.0));
            let all: f64 = v.iter().filter(|(k, _)| **k != "_barrier_gap").map(|x| x.1.0 as f64 * scale).sum();
            eprintln!("vulkan prof (timestampPeriod = {:.2} ns, active kernel sum = {:.2} ms):", self.ts_period, all / 1e6);
            for (n, (ticks, c)) in v {
                let ns = *ticks as f64 * scale;
                eprintln!("prof {n:24} {:9.2} ms {:5.1}% {c:7} calls {:8.2} us", ns / 1e6, ns * 100.0 / all.max(1.0), ns / 1e3 / (*c).max(1) as f64);
            }
        }
        unsafe {
            let _ = self.dev.device_wait_idle();
            self.dev.destroy_buffer(st.staging.0, None);
            self.dev.free_memory(st.staging.1, None);
            self.dev.destroy_buffer(st.err.0, None);
            self.dev.free_memory(st.err.1, None);
            if let Some((qp, ..)) = &st.prof { self.dev.destroy_query_pool(*qp, None) }
            self.dev.destroy_fence(st.fence, None);
            self.dev.destroy_command_pool(st.pool, None);
            self.dev.destroy_command_pool(st.gpool, None);
            self.dev.destroy_command_pool(st.tpool, None);
            self.dev.destroy_semaphore(self.csem, None);
            self.dev.destroy_semaphore(self.tsem, None);
            for &p in st.pipes.values() { self.dev.destroy_pipeline(p, None) }
            self.dev.destroy_pipeline_layout(self.layout, None);
            self.dev.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Why `pd` cannot run the kernels, if it cannot.
fn unusable(instance: &ash::Instance, pd: vk::PhysicalDevice) -> Option<String> {
    let mut sg = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
    let mut p11 = vk::PhysicalDeviceVulkan11Properties::default();
    let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sg).push_next(&mut p11);
    unsafe { instance.get_physical_device_properties2(pd, &mut p2) };
    let p = p2.properties;
    if p.device_type == vk::PhysicalDeviceType::CPU { return Some("a CPU device".into()); }
    if p.api_version < vk::API_VERSION_1_3 { return Some("Vulkan 1.3 is needed".into()); }
    if !((sg.min_subgroup_size..=sg.max_subgroup_size).contains(&32) && sg.required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE))
        && !(sg.min_subgroup_size == 64 && sg.max_subgroup_size == 64 && p11.subgroup_size == 64) {
        return Some(format!("subgroup size 32 or fixed size 64 is needed (it has {}-{})", sg.min_subgroup_size, sg.max_subgroup_size));
    }
    let ops = vk::SubgroupFeatureFlags::ARITHMETIC | vk::SubgroupFeatureFlags::SHUFFLE;
    if !p11.subgroup_supported_operations.contains(ops) || !p11.subgroup_supported_stages.contains(vk::ShaderStageFlags::COMPUTE) {
        return Some("subgroup arithmetic and shuffle in compute shaders are needed".into());
    }
    let (mut f11, mut f12, mut f13) = Default::default();
    let mut f = features(&mut f11, &mut f12, &mut f13);
    unsafe { instance.get_physical_device_features2(pd, &mut f) };
    let (int16, int64, f64) = (f.features.shader_int16, f.features.shader_int64, f.features.shader_float64);
    let missing: Vec<_> = [
        ("shaderInt16", int16), ("shaderInt64", int64), ("shaderFloat64", f64), ("storageBuffer16BitAccess", f11.storage_buffer16_bit_access),
        ("bufferDeviceAddress", f12.buffer_device_address), ("storageBuffer8BitAccess", f12.storage_buffer8_bit_access),
        ("shaderInt8", f12.shader_int8), ("shaderFloat16", f12.shader_float16), ("vulkanMemoryModel", f12.vulkan_memory_model),
        ("vulkanMemoryModelDeviceScope", f12.vulkan_memory_model_device_scope),
        ("subgroupSizeControl", f13.subgroup_size_control), ("computeFullSubgroups", f13.compute_full_subgroups),
        ("shaderIntegerDotProduct", f13.shader_integer_dot_product),
    ].into_iter().filter(|f| f.1 == vk::FALSE).map(|f| f.0).collect();
    (!missing.is_empty()).then(|| format!("missing {}", missing.join(", ")))
}

/// Feature chain for querying (all false) or enabling (filled by the caller).
fn features<'a>(f11: &'a mut vk::PhysicalDeviceVulkan11Features<'static>, f12: &'a mut vk::PhysicalDeviceVulkan12Features<'static>,
                f13: &'a mut vk::PhysicalDeviceVulkan13Features<'static>) -> vk::PhysicalDeviceFeatures2<'a> {
    vk::PhysicalDeviceFeatures2::default().push_next(f11).push_next(f12).push_next(f13)
}

impl Vk {
    /// The usable GPU with index `device` (`backend::device_order`). Fails with "Vulkan init" (and the reasons) when there is none.
    pub fn new(device: usize) -> anyhow::Result<Vk> {
        let (kv_k_mode, kv_v_mode) = (crate::cuda::kv_k_mode(), crate::cuda::kv_v_mode());
        // ponytail: Vulkan turbo gave wrong text on Qwen3-4B (RTX 3060) and its hd 128 prefill needs 64000 B of shared
        // memory; refused until fixed, `STRATA_VK_TURBO=1` opens it for tests.
        ensure!((kv_k_mode < 4 && kv_v_mode < 4) || std::env::var_os("STRATA_VK_TURBO").is_some(),
            "--kv turbo* is not ready on Vulkan yet; use the CUDA backend or --kv q8/q4");
        ensure!(!(crate::cuda::kv_mode() > 0 && crate::cuda::llama_numerics()), "--kv q8/q4/turbo* does not combine with --llama-numerics (its attention reads f16 KV)");
        // RADV hides VK_AMD_device_coherent_memory on RDNA4 (RX 9000) unless this driconf option is set, and without
        // it the GPU caches the CPU expert mailbox: MoE combines miss the worker's answer and wait until they give up.
        // Mesa reads the option from the environment when the device is created; other drivers ignore it.
        if std::env::var_os("radv_device_coherent_memory").is_none() { unsafe { std::env::set_var("radv_device_coherent_memory", "true") } }
        let entry = unsafe { ash::Entry::load() }.context("Vulkan init")?;
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = unsafe { entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None) }
            .context("Vulkan init")?;
        let mut usable = vec![];
        let mut why = vec![];
        for pd in unsafe { instance.enumerate_physical_devices() }.unwrap_or_default() {
            let p = unsafe { instance.get_physical_device_properties(pd) };
            let name = p.device_name_as_c_str().map_or("?".into(), |s| s.to_string_lossy().into_owned());
            match unusable(&instance, pd) { None => usable.push((pd, name, p.device_type)), Some(w) => why.push(format!("{name}: {w}")) }
        }
        // Discrete GPUs first (stable), so device 0 is the dedicated card, not an integrated GPU that comes first in the driver's list.
        usable.sort_by_key(|u| match u.2 { vk::PhysicalDeviceType::DISCRETE_GPU => 0, vk::PhysicalDeviceType::INTEGRATED_GPU => 1, _ => 2 });
        let usable: Vec<_> = usable.into_iter().map(|u| (u.0, u.1)).collect();
        // `device` counts as CUDA does where CUDA knows the cards, so that an index is the same card on both backends:
        // CUDA puts the fastest card first and Vulkan does not, so device 1 of a run that fell back from CUDA to
        // Vulkan for that device only was the card device 0 already ran on.
        let ids: Vec<_> = usable.iter().map(|u| {
            let mut id = vk::PhysicalDeviceIDProperties::default();
            unsafe { instance.get_physical_device_properties2(u.0, &mut vk::PhysicalDeviceProperties2::default().push_next(&mut id)) };
            id.device_uuid
        }).collect();
        let pick = crate::backend::device_order(&crate::cuda::device_uuids(), &ids).get(device).copied().flatten();
        let Some((pd, name)) = pick.map(|i| usable[i].clone()) else {
            unsafe { instance.destroy_instance(None) };
            return Err(anyhow!("no usable Vulkan device {device} ({} usable; {})", usable.len(), why.join("; "))).context("Vulkan init");
        };
        crate::say!("vulkan device {device}: {name}");
        // ponytail: the instance leaks if device setup below fails; it is a one-time startup error.
        let mut sg = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        unsafe { instance.get_physical_device_properties2(pd, &mut vk::PhysicalDeviceProperties2::default().push_next(&mut sg)) };
        let wave64 = sg.min_subgroup_size == 64;
        let qf = unsafe { instance.get_physical_device_queue_family_properties(pd) }.iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE)).context("no compute queue")? as u32;
        let dexts = unsafe { instance.enumerate_device_extension_properties(pd) }?;
        let has = |n: &std::ffi::CStr| dexts.iter().any(|e| e.extension_name_as_c_str() == Ok(n));
        let budget = has(ash::ext::memory_budget::NAME);
        let mut exts = if budget { vec![ash::ext::memory_budget::NAME.as_ptr()] } else { vec![] };
        // 16 x 16 x 16 f16 products summed in f32 and 16 x 16 x 32 int8 ones summed in int32, per subgroup: the
        // mma.sync shapes of the CUDA kernels.
        // AMD RDNA has the f16 shape but int8 only as 16 x 16 x 16, so its gemmi runs without them.
        let cm = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
        let props = if has(ash::khr::cooperative_matrix::NAME) { unsafe { cm.get_physical_device_cooperative_matrix_properties(pd) }
            .unwrap_or_default() } else { vec![] };
        let shape = |k, ab, cr| props.iter().any(|c| (c.m_size, c.n_size, c.k_size) == (16, 16, k) && c.scope == vk::ScopeKHR::SUBGROUP
            && c.a_type == ab && c.b_type == ab && c.c_type == cr && c.result_type == cr);
        let coop = !wave64 && shape(16, vk::ComponentTypeKHR::FLOAT16, vk::ComponentTypeKHR::FLOAT32)
            && std::env::var_os("STRATA_VK_NO_COOP").is_none();  // STRATA_VK_NO_COOP: run the kernels for devices without them
        let coop_i8 = if !coop || std::env::var_os("STRATA_VK_NO_COOP_I8").is_some() { 0 }
            else if shape(32, vk::ComponentTypeKHR::SINT8, vk::ComponentTypeKHR::SINT32) { 32 }
            else if shape(16, vk::ComponentTypeKHR::SINT8, vk::ComponentTypeKHR::SINT32) { 16 }
            else { 0 };
        // The tiled fallback needs a 1024-thread workgroup and 32 KB of shared memory (see attn_prefill_tile.comp).
        let dev_props = unsafe { instance.get_physical_device_properties(pd) };
        let limits = dev_props.limits;
        let ps5 = dev_props.vendor_id == 0x1002 && std::env::var("STRATA_PS5").is_ok_and(|v| v == "1");
        // Upstream 1.4.0 TurboQuant Vulkan is experimental (and its tiled prefill rotates uninitialized Q).
        // The PS5 profile intentionally refuses it pending rebuilt shaders and numerical/driver validation.
        ensure!(!ps5 || (kv_k_mode < 4 && kv_v_mode < 4), "PS5: TurboQuant Vulkan is not validated; use F16/Q8/Q4/Q2 K/V");
        // Vulkan does not override Backend::alloc_host/attn_host; enabling this CUDA experiment here only adds staging.
        ensure!(!ps5 || std::env::var("STRATA_KV_HOST").unwrap_or_default().is_empty(),
            "PS5 Vulkan: STRATA_KV_HOST is a CUDA host-attention experiment; unset it");
        let mut dot = vk::PhysicalDeviceShaderIntegerDotProductProperties::default();
        unsafe { instance.get_physical_device_properties2(pd, &mut vk::PhysicalDeviceProperties2::default().push_next(&mut dot)) };
        let r2 = ps5_r2::Kernels::read(ps5, wave64, dot.integer_dot_product4x8_bit_packed_signed_accelerated == vk::TRUE)?;
        if ps5 { crate::say!("PS5 R2: GEMV workgroup {}, small GEMM {}, packed signed dot accelerated {}", r2.wg, r2.small,
            dot.integer_dot_product4x8_bit_packed_signed_accelerated == vk::TRUE); }
        let ps5_gtt = ps5 && budget && std::env::var("STRATA_PS5_GTT").is_ok_and(|v| v == "1");
        let r3_cfg = ps5_r3::probe(&instance, pd, ps5 && ps5_gtt, has(ash::ext::external_memory_host::NAME))?;
        if r3_cfg.is_some() { exts.push(ash::ext::external_memory_host::NAME.as_ptr()); }
        let ps5_direct = ps5 && std::env::var("STRATA_PS5_DIRECT").is_ok_and(|v| v == "1");
        let ps5_host_reserve = if ps5 {
            std::env::var("STRATA_PS5_HOST_RESERVE_MB").ok().map(|v| v.parse::<u64>()).transpose()
                .context("STRATA_PS5_HOST_RESERVE_MB")?.unwrap_or(1536).checked_mul(1 << 20)
                .context("STRATA_PS5_HOST_RESERVE_MB overflow")?
        } else { 0 };
        let igpu = dev_props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU || ps5;
        if ps5 { crate::say!("PS5 profile: GTT {}, mapped upload {}, host reserve {} MiB (driver budget {})",
            ps5_gtt, ps5_direct, ps5_host_reserve >> 20, budget); }
        let name_up = name.to_ascii_uppercase();
        let gfx11 = dev_props.vendor_id == 0x1002 && (igpu || (coop && coop_i8 != 32)
            || ["GFX11", "STRIX", "PHOENIX", "KRACKAN", "NAVI3", "RADEON 8", "RADEON 7", "RX 7"]
                .iter().any(|k| name_up.contains(k)));
        let dp_tile = !coop && limits.max_compute_work_group_invocations >= 1024 && limits.max_compute_shared_memory_size >= 32 << 10
            && std::env::var_os("STRATA_VK_DP_OLD").is_none();
        // Validate requested R5 variant before allocating logical-device resources. Respect a lowered test limit.
        let r5_smem = std::env::var("STRATA_VK_SMEM_LIMIT").ok().map(|v| v.parse::<u32>()).transpose()
            .context("STRATA_VK_SMEM_LIMIT")?.unwrap_or(limits.max_compute_shared_memory_size)
            .min(limits.max_compute_shared_memory_size);
        let r5 = ps5_r5::Attention::read(ps5, wave64, coop, crate::cuda::llama_numerics(),
            limits.max_compute_work_group_invocations.min(limits.max_compute_work_group_size[0]), r5_smem)?;
        if coop { exts.push(ash::khr::cooperative_matrix::NAME.as_ptr()) }
        let mut fdc = vk::PhysicalDeviceCoherentMemoryFeaturesAMD::default();
        let amd_coherent = has(ash::amd::device_coherent_memory::NAME) && {
            unsafe { instance.get_physical_device_features2(pd, &mut vk::PhysicalDeviceFeatures2::default().push_next(&mut fdc)) };
            fdc.device_coherent_memory == vk::TRUE
        };
        if amd_coherent { exts.push(ash::amd::device_coherent_memory::NAME.as_ptr()) }
        let amd_plain = !amd_coherent && unsafe { instance.get_physical_device_properties(pd) }.vendor_id == 0x1002;
        if amd_plain { crate::say!("warning: this AMD GPU has no VK_AMD_device_coherent_memory; experts on the CPU will make decode slow (the GPU \
            does not see the CPU worker's answers in time)"); }
        let mut fcr = vk::PhysicalDeviceConditionalRenderingFeaturesEXT::default();
        let cond = has(ash::ext::conditional_rendering::NAME) && std::env::var_os("STRATA_VK_NO_COND").is_none() && {
            unsafe { instance.get_physical_device_features2(pd, &mut vk::PhysicalDeviceFeatures2::default().push_next(&mut fcr)) };
            fcr.conditional_rendering == vk::TRUE
        };
        if cond { exts.push(ash::ext::conditional_rendering::NAME.as_ptr()) }
        let mut fcr = vk::PhysicalDeviceConditionalRenderingFeaturesEXT::default().conditional_rendering(true);
        let stats = std::env::var_os("STRATA_VK_STATS").is_some() && has(ash::khr::pipeline_executable_properties::NAME);
        if stats { exts.push(ash::khr::pipeline_executable_properties::NAME.as_ptr()) }
        let mut fstats = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default().pipeline_executable_info(true);
        let mut fcoop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let mut f11 = vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true)
            .storage_buffer8_bit_access(true).timeline_semaphore(true).shader_int8(true).shader_float16(true).vulkan_memory_model(true).vulkan_memory_model_device_scope(true);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default().subgroup_size_control(true).compute_full_subgroups(true)
            .shader_integer_dot_product(true);
        let mut f = features(&mut f11, &mut f12, &mut f13);
        f.features.shader_int16 = vk::TRUE;
        f.features.shader_int64 = vk::TRUE;
        f.features.shader_float64 = vk::TRUE;
        if coop { f = f.push_next(&mut fcoop) }
        if stats { f = f.push_next(&mut fstats) }
        if amd_coherent { f = f.push_next(&mut fdc) }
        if cond { f = f.push_next(&mut fcr) }
        let prio = [1.0];
        let tf = unsafe { instance.get_physical_device_queue_family_properties(pd) }.iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::TRANSFER) && !q.queue_flags.intersects(vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS))
            .filter(|_| std::env::var_os("STRATA_VK_NO_TQ").is_none()).map(|i| i as u32);
        let qi: Vec<_> = [qf].into_iter().chain(tf).map(|f| vk::DeviceQueueCreateInfo::default().queue_family_index(f).queue_priorities(&prio)).collect();
        let dev = unsafe { instance.create_device(pd, &vk::DeviceCreateInfo::default().queue_create_infos(&qi)
            .enabled_extension_names(&exts).push_next(&mut f), None) }.with_context(|| format!("create device {name}"))?;
        let queue = unsafe { dev.get_device_queue(qf, 0) };
        let tq = tf.map(|f| unsafe { dev.get_device_queue(f, 0) });
        let tpool = unsafe { dev.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(tf.unwrap_or(qf)), None) }?;
        let mut sti = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let csem = unsafe { dev.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut sti), None) }?;
        let tsem = unsafe { dev.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut sti), None) }?;
        let mem = unsafe { instance.get_physical_device_memory_properties(pd) };
        // Workgroup shared memory of a pipeline may not exceed this. Volta has 48 KB: gemmi_dp's KS = 16
        // specialization needs exactly 49152 (the 49168-byte request of the pre-fix gemmi_dp crossed it and lost the
        // device with Xid 13). STRATA_VK_SMEM_LIMIT pretends the device reports another value, so a test can exercise
        // the refusal without a doctored shader.
        let mut smem_limit = unsafe { instance.get_physical_device_properties(pd) }.limits.max_compute_shared_memory_size;
        // The AMD proprietary driver on Windows reports 32768 on RDNA3/4 cards (cooperative matrices), which really have
        // 64 KB (issue #20; vulkaninfo agrees with the driver, Linux RADV reports 65536). No kernel needs more than 49152.
        let mut drv = vk::PhysicalDeviceDriverProperties::default();
        unsafe { instance.get_physical_device_properties2(pd, &mut vk::PhysicalDeviceProperties2::default().push_next(&mut drv)) };
        if drv.driver_id == vk::DriverId::AMD_PROPRIETARY && coop && smem_limit == 32768 { smem_limit = 49152; }
        if let Ok(v) = std::env::var("STRATA_VK_SMEM_LIMIT") { smem_limit = smem_limit.min(v.parse().context("STRATA_VK_SMEM_LIMIT")?); }
        let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(qf);
        let pool = unsafe { dev.create_command_pool(&pool_info, None) }?;
        let gpool = unsafe { dev.create_command_pool(&pool_info, None) }?;
        let fence = unsafe { dev.create_fence(&vk::FenceCreateInfo::default(), None) }?;
        let pc = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(128)];
        let layout = unsafe { dev.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&pc), None) }?;
        let st = State { pool, spare: vec![], used: vec![], queued: vec![], cur: None, cond: None, no_cond: false, need_cond: false, stalled: false, busy: false, gpool, graphs: HashMap::new(), rec: None, batch: false, beside: false, tpool, tcur: None,
            tspare: vec![], tused: vec![], tbytes: 0, tstart: 0, tring: Default::default(), cstaged: false, cval: 0, tval: 0, bwait: 0, twait: 0, tlag: 0,
            pipes: HashMap::new(), fence, staging: Default::default(), map: 0, staged: 0, ops: 0, copied: (0, 0), bufs: BTreeMap::new(), err: Default::default(),
            prof: if std::env::var_os("STRATA_VK_PROF").is_some() { Some((unsafe { dev.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(65536), None) }?, vec![], HashMap::new())) } else { None }, step_pos: None };
        let cr = cond.then(|| ash::ext::conditional_rendering::Device::new(&instance, &dev));
        let ts_period = limits.timestamp_period;
        let rcp_d = dev_props.vendor_id == 0x1002;
        let r3 = r3_cfg.map(|cfg| ps5_r3::HostImport::new(cfg, &instance, &dev));
        let inner = Inner { _entry: entry, instance, pd, dev, queue, tq, qfs: [qf, tf.unwrap_or(qf)], csem, tsem, mem, budget, igpu, ps5, r2, r3, ps5_gtt, ps5_direct, ps5_host_reserve, ps5_upload_bytes: Default::default(), r6_trace: ps5 && std::env::var("STRATA_PS5_R6").is_ok_and(|v| v == "1"), r6_seen: Default::default(), gfx11, wave64, rcp_d, coop, dp_tile, r5, r5_logged: Default::default(), coop_i8, amd_coherent, amd_plain, cr, llama: crate::cuda::llama_numerics() as u32, kv_k_mode, kv_v_mode, smem_limit, max_dispatch: limits.max_compute_work_group_count, ts_period, r7_trace: std::env::var("STRATA_R7_VK_TRACE").is_ok_and(|v| v == "1"), r7_commands: Default::default(), stats, layout, st: Mutex::new(st) };
        let staging = inner.buffer(STAGING, HOST)?;
        let map = unsafe { inner.dev.map_memory(staging.1, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) };
        let mut st = inner.st();
        st.staging = staging;
        st.map = map? as usize;
        // In mailbox memory: the GPU must see the host's give-up word (err[2]) at once, as it sees the worker's answers.
        let (b, m) = inner.buffer(12, if inner.amd_coherent { BAR_AMD } else { BAR })?;
        let p = unsafe { inner.dev.map_memory(m, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }?;
        unsafe { (p as *mut [u32; 3]).write_volatile([0, 0, 0]) };
        st.err = (b, m, p as usize, unsafe { inner.dev.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(b)) });
        drop(st);
        let mut v = Vk(
            Arc::new(inner),
            Buf::new(0, 0, ()),
            Buf::new(0, 0, ()),
            Mutex::new((Buf::new(0, 0, ()), None)),
            Mutex::new((Buf::new(0, 0, ()), None)),
            Default::default(),
        );
        v.1 = v.buf(XQ, DEVICE)?.0;
        v.4.lock().unwrap().0 = v.alloc(1 << 20)?;
        let t = iq_tables().0;
        v.2 = v.alloc(t.len())?;
        v.htod(&t, &v.2, 0)?;
        if v.0.cr.is_some() && !v.cond_test()? {
            v.0.st().no_cond = true;
            crate::say!("note: Vulkan conditional rendering failed its self-test; a CPU expert worker that is slow on \
                a layer will be an error");
        }
        // One line with what the device has, so a pasted bug report shows it.
        let mut drv = vk::PhysicalDeviceDriverProperties::default();
        unsafe { v.0.instance.get_physical_device_properties2(v.0.pd, &mut vk::PhysicalDeviceProperties2::default().push_next(&mut drv)) };
        let s = |c: &[std::ffi::c_char]| unsafe { std::ffi::CStr::from_ptr(c.as_ptr()) }.to_string_lossy().into_owned();
        let yn = |b: bool| if b { "yes" } else { "no" };
        let amd = if v.0.amd_plain || v.0.amd_coherent { format!(", device-coherent memory {}", yn(v.0.amd_coherent)) } else { String::new() };
        let i8_s = match v.0.coop_i8 { 32 => "yes (16x16x32)", 16 => "yes (16x16x16)", _ => "no" };
        crate::say!("vulkan: {name} ({} {}){amd}, cooperative matrix f16 {} int8 {i8_s}, conditional rendering {}, maxComputeSharedMemorySize {}",
            s(&drv.driver_name), s(&drv.driver_info), yn(v.0.coop), yn(v.0.cr.is_some() && !v.0.st().no_cond), v.0.smem_limit);
        if !v.0.coop { crate::say!("vulkan: prefill attention: {} (no cooperative matrices)", if v.0.dp_tile { "tiled" } else { "per-query" }); }
        Ok(v)
    }

    /// Whether conditional rendering skips dispatches after a nonzero error word and runs them after a zero one.
    fn cond_test(&self) -> anyhow::Result<bool> {
        let (mk, p) = self.buf(16, HOST)?;
        let p = p.cast::<f32>();
        unsafe { p.write_volatile(1.0) };
        let (one, err) = (mk.addr(), self.0.st().err.3);
        self.add(err, one, 1)?;  // err = bits of 1.0, nonzero
        self.0.skip_rest(&mut self.0.st())?;
        self.add(one + 4, one, 1)?;
        self.sync()?;
        let stalled = self.stalled();
        let skipped = unsafe { p.add(1).read_volatile() } == 0.0 && stalled;
        self.0.skip_rest(&mut self.0.st())?;
        self.add(one + 8, one, 1)?;
        self.sync()?;
        Ok(skipped && unsafe { p.add(2).read_volatile() } == 1.0 && !self.stalled())
    }
}

impl Inner {
    fn st(&self) -> MutexGuard<'_, State> { self.st.lock().unwrap_or_else(|e| e.into_inner()) }

    fn heap_budget(&self) -> vk::PhysicalDeviceMemoryBudgetPropertiesEXT<'static> {
        let mut b = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        {
            let mut p = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut b);
            unsafe { self.instance.get_physical_device_memory_properties2(self.pd, &mut p) };
        }
        b
    }

    fn ps5_host_room(&self) -> u64 {
        // Unknown host availability is not permission to borrow from system RAM.
        crate::memory::available_ram().unwrap_or(0).saturating_sub(self.ps5_host_reserve)
    }

    fn heap_free(&self, b: &vk::PhysicalDeviceMemoryBudgetPropertiesEXT<'_>, h: usize) -> u64 {
        b.heap_budget[h].min(self.mem.memory_heaps[h].size).saturating_sub(b.heap_usage[h])
    }

    /// A buffer usable by kernels (device address) and copies, of at least `bytes`, in the first memory of `wants`
    /// (`DEVICE`, `HOST` or `BAR`) there is.
    fn buffer(&self, bytes: usize, wants: &[vk::MemoryPropertyFlags]) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | if self.cr.is_some() { vk::BufferUsageFlags::CONDITIONAL_RENDERING_EXT } else { vk::BufferUsageFlags::empty() };
        let size = bytes.max(1).next_multiple_of(4) as u64;
        let mut bi = vk::BufferCreateInfo::default().size(size).usage(usage);
        if self.tq.is_some() { bi = bi.sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&self.qfs); }
        let buf = unsafe { self.dev.create_buffer(&bi, None) }?;
        let req = unsafe { self.dev.get_buffer_memory_requirements(buf) };
        // Memory the GPU does not cache (AMD flags 0xc0) only when asked for: it would be slow for anything else.
        let ok = |f: vk::MemoryPropertyFlags, want| f.contains(want) && (f & vk::MemoryPropertyFlags::from_raw(0xc0) & !want).is_empty();
        // On an iGPU, device-local and GTT heaps live in the same LPDDR: fall back to HOST if DEVICE heap is full.
        let candidates: Vec<vk::MemoryPropertyFlags> = if wants == DEVICE && ((self.igpu && !self.ps5) || self.ps5_gtt) {
            wants.iter().chain(HOST.iter()).copied().collect()
        } else {
            wants.to_vec()
        };
        // Do not let RADV silently oversubscribe a heap on PS5. Unlike a discrete GPU,
        // GTT borrows from the same physical RAM the CPU experts and Linux need.
        let budget = (self.ps5 && self.budget && (wants == DEVICE || wants == MAPPED_DEVICE)).then(|| self.heap_budget());
        let host_room = if budget.is_some() { self.ps5_host_room() } else { 0 };
        let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut last_err = anyhow::anyhow!("no matching memory type for {bytes} bytes");
        let mut mem_ok = None;
        let mut nonlocal = false;
        'outer: for want in candidates {
            for i in 0..self.mem.memory_type_count {
                if req.memory_type_bits & (1 << i) != 0 && ok(self.mem.memory_types[i as usize].property_flags, want) {
                    let h = self.mem.memory_types[i as usize].heap_index as usize;
                    if let Some(b) = &budget {
                        let local = self.mem.memory_heaps[h].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL);
                        let free = self.heap_free(b, h);
                        let room = if local { free } else if self.ps5_gtt { free.min(host_room) } else { 0 };
                        if req.size > room {
                            last_err = anyhow::anyhow!("PS5 heap {h}: need {} MiB, budget {} MiB", req.size >> 20, room >> 20);
                            continue;
                        }
                    }
                    match unsafe {
                        self.dev.allocate_memory(
                            &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(i).push_next(&mut flags),
                            None,
                        )
                    } {
                        Ok(m) => {
                            nonlocal = wants == DEVICE && !self.mem.memory_heaps[h].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL);
                            mem_ok = Some(m); break 'outer;
                        }
                        Err(e) => last_err = anyhow::Error::from(e).context(format!("allocate {bytes} bytes")),
                    }
                }
            }
        }
        let mem = match mem_ok { Some(m) => m, None => { unsafe { self.dev.destroy_buffer(buf, None) }; return Err(last_err); } };
        if let Err(e) = unsafe { self.dev.bind_buffer_memory(buf, mem, 0) } {
            unsafe { self.dev.destroy_buffer(buf, None); self.dev.free_memory(mem, None); }
            return Err(e.into());
        }
        if nonlocal { self.r6_mark(1 << 7, "gtt_nonlocal"); }
        Ok((buf, mem))
    }

    fn r6_mark(&self, bit: u32, name: &str) {
        if self.r6_trace && self.r6_seen.fetch_or(bit, std::sync::atomic::Ordering::Relaxed) & bit == 0 {
            eprintln!("PS5 R6 exercised: {name}");
        }
    }

    /// Where ops record: the graph being recorded, else the open command buffer.
    fn cb(&self, st: &mut State) -> anyhow::Result<vk::CommandBuffer> {
        match st.rec { Some(c) => Ok(c), None => self.open_cb(st) }
    }

    /// The open command buffer, begun if needed.
    fn open_cb(&self, st: &mut State) -> anyhow::Result<vk::CommandBuffer> {
        st.ops += 1;
        if let Some(c) = st.cur { return Ok(c); }
        let c = match st.spare.pop() { Some(c) => c, None => unsafe { self.dev.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default().command_pool(st.pool).command_buffer_count(1)) }?[0] };
        unsafe { self.dev.begin_command_buffer(c, &vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)) }?;
        st.cur = Some(c);
        if st.need_cond { self.cond_on(st, c); }
        Ok(c)
    }

    /// Orders everything recorded so far before what follows (`HOST`: also before host reads).
    fn barrier(&self, cb: vk::CommandBuffer, host: bool) {
        if self.r7_trace { self.r7_commands.barrier.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
        let all = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
        let (dst, dst_access) = if host { (vk::PipelineStageFlags::HOST, vk::AccessFlags::HOST_READ) } else {
            (all, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE)
        };
        let m = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(dst_access);
        unsafe { self.dev.cmd_pipeline_barrier(cb, all, dst, vk::DependencyFlags::empty(), &[m], &[], &[]) }
    }

    /// After a MoE combine that may give up waiting for the CPU worker: the dispatches `cb` records next run only
    /// while the error word is 0, so nothing after a combine that gave up runs until the host has seen it (no-op
    /// without conditional rendering). Every compute command buffer starts this way too, for the submissions after.
    fn skip_rest(&self, st: &mut State) -> anyhow::Result<()> {
        st.need_cond = true;
        let cb = self.cb(st)?;
        self.end_cond(st, cb);
        self.cond_on(st, cb);
        Ok(())
    }

    fn cond_on(&self, st: &mut State, cb: vk::CommandBuffer) {
        let Some(cr) = self.cr.as_ref().filter(|_| !st.no_cond) else { return };
        if self.r7_trace { self.r7_commands.conditional_barrier.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
        let m = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::CONDITIONAL_RENDERING_READ_EXT);
        unsafe {
            self.dev.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::CONDITIONAL_RENDERING_EXT,
                vk::DependencyFlags::empty(), &[m], &[], &[]);
            (cr.fp().cmd_begin_conditional_rendering_ext)(cb, &vk::ConditionalRenderingBeginInfoEXT::default().buffer(st.err.0)
                .flags(vk::ConditionalRenderingFlagsEXT::INVERTED));
        }
        st.cond = Some(cb);
    }

    /// End conditional rendering in `cb` if it is on there (before ending `cb`).
    fn end_cond(&self, st: &mut State, cb: vk::CommandBuffer) {
        if st.cond == Some(cb) {
            unsafe { (self.cr.as_ref().unwrap().fp().cmd_end_conditional_rendering_ext)(cb) };
            st.cond = None;
        }
    }

    /// Close `st.cur` if open and append it (plus `graph` if given) to `st.queued` in order.
    fn enqueue_cb(&self, st: &mut State, graph: Option<vk::CommandBuffer>) -> anyhow::Result<()> {
        if let Some(c) = st.cur.take() {
            st.copied.1 = 0;
            self.end_cond(st, c);
            unsafe { self.dev.end_command_buffer(c) }?;
            st.used.push(c);
            st.queued.push(c);
        }
        st.queued.extend(graph);
        Ok(())
    }

    /// Submit the open command buffer, then `graph`; with `wait`, wait for all submitted work.
    fn submit(&self, st: &mut State, graph: Option<vk::CommandBuffer>, wait: bool) -> anyhow::Result<()> {
        self.enqueue_cb(st, graph)?;
        let cbs = std::mem::take(&mut st.queued);
        if cbs.is_empty() && !(wait && st.busy) { return Ok(()); }
        let fence = if wait { st.fence } else { vk::Fence::null() };
        st.cval += 1;
        let (wv, sv, ws, ss, wst) = ([st.twait], [st.cval], [self.tsem], [self.csem], [vk::PipelineStageFlags::ALL_COMMANDS]);
        let n = (st.twait > 0) as usize;
        let mut tl = vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&wv[..n]).signal_semaphore_values(&sv);
        let si = vk::SubmitInfo::default().command_buffers(&cbs).wait_semaphores(&ws[..n]).wait_dst_stage_mask(&wst[..n])
            .signal_semaphores(&ss).push_next(&mut tl);
        let submit_start = self.r7_trace.then(std::time::Instant::now);
        unsafe { self.dev.queue_submit(self.queue, &[si], fence) }?;
        if let Some(start) = submit_start {
            eprintln!("r7,host,compute_submit,{},{},{}", st.cval, cbs.len(), start.elapsed().as_nanos());
        }
        st.twait = 0;
        st.busy = true;
        if wait {
            // A fence also covers every earlier submission on the queue; the transfer queue's copies are waited too.
            // A MoE combine waiting for the CPU worker gives up after SPINS polls, but a poll can be slow enough (RADV)
            // for that to outlast the driver's job timeout (amdgpu: 10 s, then the device is lost). So after GIVE_UP
            // the host raises err[2] and a waiting combine gives up within 256 polls. Without conditional rendering
            // giving up is an error, so only near the timeout.
            let e = st.err.2 as *mut u32;
            let give_up = if self.cr.is_some() && !st.no_cond { GIVE_UP } else { 8 * GIVE_UP };
            let t0 = std::time::Instant::now();
            // A short round trip (a matvec of one token) ends before a sleeping thread would wake: poll first.
            while t0.elapsed() < SPIN && !unsafe { self.dev.get_fence_status(st.fence) }? { std::hint::spin_loop(); }
            loop {
                match unsafe { self.dev.wait_for_fences(&[st.fence], true, 20_000_000) } {
                    Err(vk::Result::TIMEOUT) => if t0.elapsed() >= give_up && !e.is_null() { unsafe { e.add(2).write_volatile(1) } },
                    r => break r?,
                }
            }
            if self.r7_trace {
                eprintln!("r7,host,compute_fence_wait,{},1,{}", st.cval, t0.elapsed().as_nanos());
            }
            if !e.is_null() { unsafe { e.add(2).write_volatile(0) } }
            unsafe {
                self.dev.reset_fences(&[st.fence])?;
                self.dev.reset_command_pool(st.pool, vk::CommandPoolResetFlags::empty())?;
                if st.tval > 0 {
                    self.dev.wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&[self.tsem]).values(&[st.tval]), u64::MAX)?;
                    self.dev.reset_command_pool(st.tpool, vk::CommandPoolResetFlags::empty())?;
                    let t = std::mem::take(&mut st.tused);
                    st.tspare.extend(t);
                }
                st.tring.clear();
                st.cstaged = false;
            }
            if let Some((qp, names, tot)) = &mut st.prof { if !names.is_empty() {
                let mut t = vec![0u64; 2 * names.len()];
                unsafe { self.dev.get_query_pool_results(*qp, 0, &mut t, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT) }?;
                for (i, n) in names.drain(..).enumerate() {
                    let e = tot.entry(n).or_default();
                    let ticks = t[2 * i + 1].saturating_sub(t[2 * i]);
                    e.0 += ticks;
                    if self.r7_trace {
                        eprintln!("r7,gpu,{n},{},{},{}", st.cval, i, ticks as f64 * self.ts_period as f64);
                    }
                    e.1 += 1;
                    if i > 0 && t[2 * i] >= t[2 * i - 1] {
                        let g = tot.entry("_barrier_gap").or_default();
                        g.0 += t[2 * i] - t[2 * i - 1];
                        g.1 += 1;
                    }
                }
            } }
            let used = std::mem::take(&mut st.used);
            st.spare.extend(used);
            st.busy = false;
            st.need_cond = false;
            st.staged = 0;
            if !e.is_null() { unsafe { e.add(1).write_volatile(0) }; }
            if !e.is_null() && unsafe { e.read_volatile() } != 0 {
                unsafe { e.write_volatile(0) };
                if self.cr.is_some() && !st.no_cond { st.stalled = true; return Ok(()); }
                anyhow::ensure!(!self.amd_plain, "moe_combine: the CPU expert worker did not answer in time. This AMD GPU \
                    has no VK_AMD_device_coherent_memory, which the CPU expert path needs; RADV hides it on RDNA4 \
                    (RX 9000) unless you set radv_device_coherent_memory=true in the environment");
                // Without conditional rendering the rest of the submission would run on a half-done layer.
                anyhow::bail!("moe_combine: the CPU expert worker did not answer in time, and without Vulkan conditional \
                    rendering the forward cannot resume");
            }
        }
        Ok(())
    }

    /// Submit and wait for everything queued.
    fn flush(&self, st: &mut State) -> anyhow::Result<()> {
        ensure!(st.rec.is_none(), "a host copy or sync while recording a graph");
        self.tsubmit(st)?;
        self.submit(st, None, true)
    }

    /// Submit the open transfer command buffer, after compute submit `bwait`.
    fn tsubmit(&self, st: &mut State) -> anyhow::Result<()> {
        let (Some(c), Some(q)) = (st.tcur.take(), self.tq) else { return Ok(()) };
        unsafe { self.dev.end_command_buffer(c) }?;
        st.tused.push(c);
        st.tval += 1;
        let (a, b, v) = (st.tstart, st.staged, st.tval);
        st.tring.push_back((a, b, v));
        st.tbytes = 0;
        let (wv, sv, ws, ss, wst) = ([st.bwait], [st.tval], [self.csem], [self.tsem], [vk::PipelineStageFlags::TRANSFER]);
        let mut tl = vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&wv).signal_semaphore_values(&sv);
        let cs = [c];
        let si = vk::SubmitInfo::default().command_buffers(&cs).wait_semaphores(&ws).wait_dst_stage_mask(&wst)
            .signal_semaphores(&ss).push_next(&mut tl);
        unsafe { self.dev.queue_submit(q, &[si], vk::Fence::null()) }?;
        if self.r7_trace { self.r7_commands.transfer_submit.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
        st.busy = true;
        Ok(())
    }

    /// Wait for the transfer submits that read `staging` bytes `at .. at + n`.
    fn ring_wait(&self, st: &mut State, at: usize, n: usize) -> anyhow::Result<()> {
        if let Some(v) = st.tring.iter().filter(|r| r.0 < at + n && at < r.1).map(|r| r.2).max() {
            unsafe { self.dev.wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&[self.tsem]).values(&[v]), u64::MAX) }?;
            st.tring.retain(|r| r.2 > v);
        }
        Ok(())
    }

    /// The open transfer command buffer, begun if needed.
    fn topen(&self, st: &mut State) -> anyhow::Result<vk::CommandBuffer> {
        if let Some(c) = st.tcur { return Ok(c); }
        st.tstart = st.staged;
        let c = match st.tspare.pop() { Some(c) => c, None => unsafe { self.dev.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default().command_pool(st.tpool).command_buffer_count(1)) }?[0] };
        unsafe { self.dev.begin_command_buffer(c, &vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)) }?;
        st.tcur = Some(c);
        Ok(c)
    }

    /// The pipeline of kernel `name` with specialization constants `spec` (ids 0, 1, ...), created on first use.
    fn pipeline(&self, st: &mut State, name: &'static str, spec: &[u32]) -> anyhow::Result<vk::Pipeline> {
        if let Some(&p) = st.pipes.get(&(name, spec.to_vec())) { return Ok(p); }
        let code = SPV.iter().find(|s| s.0 == name).with_context(|| format!("no Vulkan kernel {name}"))?.1;
        // One choke point for Workgroup shared memory: the driver creates such a pipeline and faults at dispatch.
        spv_smem::check_shared_memory(name, spec, self.kv_k_mode, self.kv_v_mode, self.smem_limit)?;
        let words = ash::util::read_spv(&mut std::io::Cursor::new(code))?;
        let m = unsafe { self.dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None) }?;
        let n = spec.len() as u32;
        let entries: Vec<_> = (0..n).map(|i| vk::SpecializationMapEntry::default().constant_id(i).offset(4 * i).size(4))
            .chain([vk::SpecializationMapEntry::default().constant_id(15).offset(4 * n).size(4),
                    vk::SpecializationMapEntry::default().constant_id(16).offset(4 * n + 4).size(4),
                    vk::SpecializationMapEntry::default().constant_id(14).offset(4 * n + 8).size(4),
                    vk::SpecializationMapEntry::default().constant_id(13).offset(4 * n + 12).size(4)]).collect();
        let data: Vec<u8> = spec.iter().chain([&self.kv_k_mode, &self.kv_v_mode, &(self.wave64 as u32), &(self.rcp_d as u32)]).flat_map(|v| v.to_le_bytes()).collect();
        let si = vk::SpecializationInfo::default().map_entries(&entries).data(&data);
        let mut sub = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(32);
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(m).name(c"main").specialization_info(&si);
        let stage = if self.wave64 { stage } else { stage.flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS).push_next(&mut sub) };
        let r = unsafe { self.dev.create_compute_pipelines(vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(self.layout)
                .flags(if self.stats { vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR } else { vk::PipelineCreateFlags::empty() })], None) };
        unsafe { self.dev.destroy_shader_module(m, None) };
        let p = r.map_err(|e| e.1).with_context(|| format!("create pipeline {name}"))?[0];
        if self.stats {
            let pe = ash::khr::pipeline_executable_properties::Device::new(&self.instance, &self.dev);
            let s = unsafe { pe.get_pipeline_executable_statistics(&vk::PipelineExecutableInfoKHR::default().pipeline(p)) }?;
            let v = s.iter().map(|s| format!("{} {}", s.name_as_c_str().map_or("?".into(), |n| n.to_string_lossy()), unsafe {
                match s.format { vk::PipelineExecutableStatisticFormatKHR::INT64 => s.value.i64.to_string(),
                    vk::PipelineExecutableStatisticFormatKHR::FLOAT64 => s.value.f64.to_string(), _ => s.value.u64.to_string() } }));
            eprintln!("vk stats {name} {spec:?}: {}", v.collect::<Vec<_>>().join(", "));
        }
        st.pipes.insert((name, spec.to_vec()), p);
        Ok(p)
    }
}

impl Vk {
    /// `step[0]` on the host, cached until a host copy writes that step buffer. Prefill reads it once per chunk to
    /// choose how much attention work one submit may hold.
    fn step_pos(&self, step: u64) -> anyhow::Result<usize> {
        let st = self.0.st();
        if let Some((a, p)) = st.step_pos { if a == step { return Ok(p); } }
        drop(st);
        let mut b = [0u8; 4];
        self.dtoh(&Buf::new(step, 4, ()), 0, &mut b)?;
        let p = i32::from_le_bytes(b).max(0) as usize;
        self.0.st().step_pos = Some((step, p));
        Ok(p)
    }
}

/// A push constant argument: pointers are u64, the rest 32-bit, each at its natural alignment (std430).
trait Arg { fn put(&self, v: &mut Vec<u8>); }
impl Arg for u64 { fn put(&self, v: &mut Vec<u8>) { v.resize(v.len().next_multiple_of(8), 0); v.extend(self.to_le_bytes()) } }
impl Arg for f64 { fn put(&self, v: &mut Vec<u8>) { v.resize(v.len().next_multiple_of(8), 0); v.extend(self.to_le_bytes()) } }
impl Arg for u32 { fn put(&self, v: &mut Vec<u8>) { v.extend(self.to_le_bytes()) } }
impl Arg for i32 { fn put(&self, v: &mut Vec<u8>) { v.extend(self.to_le_bytes()) } }
impl Arg for f32 { fn put(&self, v: &mut Vec<u8>) { v.extend(self.to_le_bytes()) } }

impl Vk {
    /// As the CUDA one: both kernels run, the step decides which one works, then attn_merge combines the splits (and
    /// writes out quantized into xq when not null).
    fn attn(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize, scale: f32, gq_min: i32, nt: usize, xq: u64, sel: u64, sst: usize, sink: u64, win: usize, kvt: u64, splits: usize) -> anyhow::Result<()> {
        ensure!([64, 128, 256, 512].contains(&hd), "attention head_dim {hd} unsupported (64, 128, 256 or 512)");
        let (per, pz) = ((hd / 32) as u32, (nh * splits * (hd + 2)) as u32);  // splits: of attn_decode_gqa
        if self.0.llama != 0 {  // one head per workgroup, merged as the attn_decode splits (empty ones weigh 0)
            self.run("attn_decode_llama", &[0, hd as u32], [nh * nt, ATT_SPLITS_1, 1], &[&q, &kc, &vc, &part, &step, &sel, &(sst as u32), &(nh as u32), &(nkv as u32), &scale, &pz, &(nt as u32), &(win as i32)])?;
            return self.run("attn_merge", &[], [nh, 1, nt], &[&part, &qg, &out, &step, &(hd as u32), &(ATT_SPLITS as u32), &(ATT_SPLITS_1 as u32), &i32::MAX, &0u64, &sink]);
        }
        let g = [8, 4, 2, 1].into_iter().find(|&g| (nh / nkv) % g == 0 && g * hd <= 2048).unwrap();
        let args: [&dyn Arg; 15] = [&q, &kc, &vc, &part, &step, &sel, &(sst as u32), &(nh as u32), &(nkv as u32), &scale, &gq_min, &pz, &(nt as u32), &(win as i32), &kvt];
        // gq_min i32::MAX or 0 fixes the regime, so the dispatch that would exit at once is skipped.
        let wide = (nt >= 2) as u32;
        if gq_min != 0 { self.run("attn_decode", &[0, per, 1, 0, wide], [nh * nt, ATT_SPLITS_1, 1], &args)?; }
        if gq_min != i32::MAX { self.run("attn_decode", &[0, per, g as u32, 1, wide], [nh / g * nt, splits, 1], &args)?; }
        self.run("attn_merge", &[], [nh, 1, nt], &[&part, &qg, &out, &step, &(hd as u32), &(splits as u32), &(ATT_SPLITS_1 as u32), &gq_min, &xq, &sink])
    }
    /// rms_norm with rope and (kc, or kvt with v) the K/V store; see rms_norm.comp.
    fn rope_rows(&self, x: u64, x_stride: usize, w: u64, out: u64, n: usize, rows: usize, eps: f32, per: usize, n_rot: usize, step: u64, base: f32, v: u64, kc: u64, vc: u64, kvt: u64) -> anyhow::Result<()> {
        if kc != 0 || (kvt != 0 && v != 0) {
            let turbo = self.0.kv_k_mode >= 4 || self.0.kv_v_mode >= 4;
            ensure!(!turbo || n % 128 == 0, "turbo KV cache: head_dim {n} is not a multiple of 128");
        }
        let ts = (base as f64).powf(-2.0 / n_rot as f64);
        self.run("rms_norm", &[self.0.llama], [rows, 1, 1], &[&x, &w, &out, &0u64, &(x_stride as u32), &(n as u32), &(n as u32), &eps, &step, &ts, &(per as u32), &(n_rot as u32), &0u64, &v, &kc, &vc, &kvt])
    }
    /// Record kernel `name` over `groups` workgroups with push constants `args`.
    fn run(&self, name: &'static str, spec: &[u32], groups: [usize; 3], args: &[&dyn Arg]) -> anyhow::Result<()> {
        ensure!(groups.iter().zip(self.0.max_dispatch).all(|(&n, max)| n <= max as usize),
            "{name}: dispatch {groups:?} exceeds device limits {:?}", self.0.max_dispatch);
        let mut pc = Vec::with_capacity(128);
        for a in args { a.put(&mut pc) }
        assert!(pc.len() <= 128, "{name}: {} bytes of push constants", pc.len());
        let mut st = self.0.st();
        let p = self.0.pipeline(&mut st, name, spec)?;
        let cb = self.0.cb(&mut st)?;
        let d = &self.0.dev;
        unsafe {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, p);
            d.cmd_push_constants(cb, self.0.layout, vk::ShaderStageFlags::COMPUTE, 0, &pc);
            if st.rec.is_none() { if let Some((qp, names, _)) = &mut st.prof { if names.len() < 32767 {
                if names.is_empty() { d.cmd_reset_query_pool(cb, *qp, 0, 65536); }
                d.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, *qp, 2 * names.len() as u32); } } }
            d.cmd_dispatch(cb, groups[0] as u32, groups[1] as u32, groups[2] as u32);
        }
        if self.0.r7_trace { self.0.r7_commands.dispatch.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
        if self.0.r6_trace {
            let bit = match name {
                "gemv_q_ps5_64" | "moe_gemv_q_ps5_64" => 1,
                "gemv_q_ps5_128" | "moe_gemv_q_ps5_128" => 2,
                "gemmi_dp_ps5_32" => 4,
                "gemmi_dp_ps5_64" => 1 << 8,
                "gemm_f_dp_ps5_iq32" => 1 << 9,
                "gemmi_dp_ps5_iq32" => 1 << 12,
                "gemmi_dp_ps5_iq64" => 1 << 13,
                "attn_prefill_ps5_256" => 8,
                "attn_prefill_ps5_512" => 16,
                "attn_prefill_ps5_1024" => 32,
                _ => 0,
            };
            if bit != 0 { self.0.r6_mark(bit, name); }
        }
        if !st.batch || name == "quant_xq" { self.0.barrier(cb, false); }
        if st.rec.is_none() { if let Some((qp, names, _)) = &mut st.prof { if names.len() < 32767 {
            let label = match (name, spec.get(1).copied()) {
                ("gemv_q", Some(11)) => "gemv_q:Q3_K",
                ("gemv_q", Some(13)) => "gemv_q:Q5_K",
                ("gemv_q", Some(14)) => "gemv_q:Q6_K",
                ("gemv_q", Some(8)) => "gemv_q:Q8_0",
                ("gemv_q", Some(18)) => "gemv_q:IQ3_XXS",
                ("gemm_f_dp", Some(17)) => "gemm_f_dp:IQ2_XS",
                ("gemmi_dp", Some(18)) => "gemmi_dp:IQ3_XXS",
                ("gemmi_dp", Some(23)) => "gemmi_dp:IQ4_XS",
                ("moe_gemv_q", Some(17)) => "moe_gemv_q:IQ2_XS",
                ("moe_gemv_q", Some(23)) => "moe_gemv_q:IQ4_XS",
                ("moe_down_combine", Some(23)) => "moe_down_combine:IQ4_XS",
                ("moe_gemv_q", Some(18)) => "moe_gemv_q:IQ3_XXS",
                ("moe_gemv_q", Some(11)) => "moe_gemv_q:Q3_K",
                ("moe_down_combine", Some(18)) => "moe_down_combine:IQ3_XXS",
                ("moe_down_combine", Some(11)) => "moe_down_combine:Q3_K",
                _ => name,
            };
            unsafe { d.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, *qp, 2 * names.len() as u32 + 1) }; names.push(label); } } }
        Ok(())
    }

    /// A zeroed buffer of `bytes` in memory `wants`; unless `DEVICE`, mapped, with its host pointer.
    fn buf(&self, bytes: usize, wants: &[vk::MemoryPropertyFlags]) -> anyhow::Result<(Buf, *mut u8)> {
        let (buf, mem) = self.0.buffer(bytes, wants)?;
        if self.0.r7_trace { self.0.r7_commands.buffer_alloc.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
        let host = wants != DEVICE;
        let addr = unsafe { self.0.dev.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf)) };
        let owner = Mem { vk: self.0.clone(), buf, mem, addr, imported: None };
        let mut st = self.0.st();
        st.bufs.insert(addr, (buf, bytes));
        let mut ptr = std::ptr::null_mut();
        if host {
            ptr = unsafe { self.0.dev.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }?.cast::<u8>();
            unsafe { ptr.write_bytes(0, bytes) };
        } else {
            // Outside any graph being recorded: zeroing is done once, now.
            let cb = self.0.open_cb(&mut st)?;
            unsafe { self.0.dev.cmd_fill_buffer(cb, buf, 0, vk::WHOLE_SIZE, 0) };
            self.0.barrier(cb, false);
        }
        drop(st);
        Ok((Buf::new(addr, bytes, owner), ptr))
    }
}

impl State {
    /// The buffer holding `bytes` from device address `addr`, and the offset of `addr` in it.
    fn find(&self, addr: u64, bytes: usize) -> anyhow::Result<(vk::Buffer, u64)> {
        let (&a, &(b, len)) = self.bufs.range(..=addr).next_back().with_context(|| format!("{addr:#x} is not a device buffer"))?;
        ensure!((addr - a).checked_add(bytes as u64).is_some_and(|end| end <= len as u64), "{bytes} bytes at {addr:#x} pass the end of its buffer");
        Ok((b, addr - a))
    }
}

/// An op the Vulkan backend does not have yet.

impl Vk {
    /// Specialization constants of the weight kernels (deq.glsl): llama numerics, the ggml type id, the bytes per
    /// 256-element super-block and the IQ tables' address (low, high).
    fn wspec(&self, ty: GgmlType) -> [u32; 5] {
        let id = (0..256).find(|&i| GgmlType::from_u32(i).is_ok_and(|t| t == ty)).unwrap();
        let t = self.2.addr();
        [self.0.llama, id, (ty.block_bytes() * 256 / ty.block_elems()) as u32, t as u32, (t >> 32) as u32]
    }
}

impl Vk {
    // Shape-only tuning; the shader's per-row reduction and combine order do not change.
    fn r11_rpw(&self, ty: GgmlType, key: &str, baseline: usize) -> anyhow::Result<usize> {
        if !self.0.ps5 || self.0.wave64 || !matches!(ty, GgmlType::IQ2XS | GgmlType::IQ3XXS | GgmlType::IQ4XS) { return Ok(baseline); }
        static GEMV: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        static DOWN: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        let setting = if key == "STRATA_PS5_R11_GEMV_RPW" { &GEMV } else { &DOWN };
        match setting.get_or_init(|| std::env::var(key).ok()).as_deref().unwrap_or("0") {
            "0" => Ok(baseline), "1" => Ok(1), "2" => Ok(2), "4" => Ok(4), "8" => Ok(8),
            v => anyhow::bail!("{key}: expected 0, 1, 2, 4 or 8, got {v}"),
        }
    }

    fn r11_iq32(&self, ty: GgmlType) -> bool {
        static ENABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        cfg!(feature = "ps5-r2") && self.0.ps5 && !self.0.wave64 && !self.0.coop && ty == GgmlType::IQ2XS
            && *ENABLE.get_or_init(|| std::env::var("STRATA_PS5_R11_IQ32").as_deref() == Ok("1"))
    }

    // R12: independent routed INT8 tiles, no cooperative matrices or changed arithmetic.
    fn r12_iq_tile(&self, ty: GgmlType) -> usize {
        static TILE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        if !cfg!(feature = "ps5-r2") || !self.0.ps5 || self.0.wave64 || self.0.coop_i8 != 0
            || !matches!(ty, GgmlType::IQ3XXS | GgmlType::IQ4XS) { return 0; }
        *TILE.get_or_init(|| match std::env::var("STRATA_PS5_R12_IQ_INT_TILE").as_deref() {
            Ok("32") => 32, Ok("64") => 64, _ => 0,
        })
    }

    fn q8_mutex(&self, n_x: usize, cols: usize) -> &Mutex<(Buf, Option<[u64; 4]>)> {
        let need = n_x.saturating_mul(cols.saturating_add(cols / 8).saturating_add(cols / 4));
        if n_x <= crate::cpu_worker::TMAX && need <= (1 << 20) { &self.4 } else { &self.3 }
    }

    /// Quantize `x` for the int8 GEMM, unless the cache already holds the same rows; `None` when `ty` or `cols`
    /// selects the f16 path. Shared by `gemm_raw` and `prequant_gemm_x`.
    fn q8_x(&self, q8: &mut (Buf, Option<[u64; 4]>), ty: GgmlType, cols: usize, n_x: usize, x: u64, x_stride: usize) -> anyhow::Result<Option<u64>> {
        if !crate::cuda::Gpu::mmq(ty) || !cols.is_multiple_of(128) { return Ok(None); }
        let need = q8_gemm_bytes(n_x, cols, x_stride)?;
        let key = Some([x, x_stride as u64, n_x as u64, cols as u64]);
        if q8.1 == key { return Ok(Some(q8.0.addr())); }
        if q8.0.len() < need { q8.0 = self.alloc(need)?; }  // dropping the old buffer waits for work that uses it
        let xq = q8.0.addr();
        self.run("quant_q8", &[self.0.llama], [n_x, 1, 1], &[&x, &xq, &(x_stride as u32), &(cols as u32)])?;
        q8.1 = key;
        Ok(Some(xq))
    }

    /// `gemm` over a tile list, or (tiles = 0) the dense matrix `wd` over positions 0 .. nd; as `gemm_raw` in
    /// src/cuda/mod.rs, with the same choice of f16 or int8 products (`Gpu::mmq`).
    fn gemm_raw(&self, ty: GgmlType, rows: usize, cols: usize, tiles: u64, n_tiles: usize, wd: u64, nd: usize, x: u64, x_stride: usize, n_x: usize, xrow: u64, y: u64, y_stride: usize, act: i32, w_scale: f32) -> anyhow::Result<()> {
        // Partial super-blocks (Q2_0's 640 columns, BF16's 320) are fine: K steps are 64 deep.
        if n_tiles == 0 { return Ok(()); }
        ensure!(rows > 0 && cols > 0 && cols % 64 == 0 && cols % ty.block_elems() == 0,
            "{} matrix {rows}x{cols}: GEMM needs nonempty complete blocks and a multiple of 64 columns", ty.name());
        ensure!(n_x > 0 && x_stride >= cols && y_stride >= rows && [rows, cols, n_x, nd, x_stride, y_stride].iter().all(|&v| u32::try_from(v).is_ok()),
            "GEMM invalid dimensions or strides");
        ensure!(rows.checked_mul(ty.row_bytes(cols as u64)).is_some_and(|v| v <= u32::MAX as usize),
            "GEMM weight offsets exceed shader uint range");
        ensure!(n_x.checked_mul(x_stride).is_some_and(|v| v <= u32::MAX as usize), "GEMM input offsets exceed shader uint range");
        // Routed output rows are indexed by the tile list, not n_x. Only dense output is bounded by nd here.
        ensure!(tiles != 0 || nd.checked_mul(y_stride).is_some_and(|v| v <= u32::MAX as usize), "GEMM output offsets exceed shader uint range");
        let (spec, grid, nk) = (self.wspec(ty), [rows.div_ceil(64), n_tiles, 1], (cols / 64) as u32);
        let (same, batched, act) = (act & SAME_X != 0, act & BATCHED != 0, act & 3);
        let mut q8 = self.q8_mutex(n_x, cols).lock().unwrap();
        let key = Some([x, x_stride as u64, n_x as u64, cols as u64]);  // as on CUDA
        let reuse = same && q8.1 == key;
        // A gemm inside a batch cannot quantize x (the quantize and the gemm would not be ordered): when the cached
        // q8 rows are not x's (or, without SAME_X, may not be), use the f16 kernel (as on CUDA).
        let int8 = crate::cuda::Gpu::mmq(ty) && cols.is_multiple_of(128);
        let f = !int8 || (!reuse && self.0.st().batch);
        static R11_PROFILE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *R11_PROFILE.get_or_init(|| std::env::var("STRATA_PS5_R11_PROFILE").as_deref() == Ok("1")) {
            // Nominal compressed weight bytes per listed tile, not measured bus traffic (short tiles may re-read).
            eprintln!("r11_gemm_shape,{},rows={},cols={},tiles={},routed={},x_rows={},contract={},weight_bytes_per_tile={}",
                ty.name(),rows,cols,n_tiles,tiles!=0,n_x,if f {"f16"} else {"q8"},rows.saturating_mul(ty.row_bytes(cols as u64)));
        }
        // The R8 oracle selects a contract from type and columns. Reject any routed fallback caused by batching.
        static R8_PROBE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if tiles != 0 && *R8_PROBE.get_or_init(|| std::env::var("STRATA_PS5_R8_CONTRACT").as_deref() == Ok("1")
            || std::env::var("STRATA_R7_R4_STAGE_PROBE").as_deref() == Ok("1")) {
            ensure!(!self.0.st().batch, "R8 routed numerical probe excludes batched GEMM");
        }
        // A dense GEMM of few workgroups runs as gemm_fs/gemmi_s (cooperative matrices only), 32 positions a
        // workgroup (as on CUDA).
        let sm = tiles == 0 && grid[0] * grid[1] < 112 && if f { self.0.coop } else { self.0.coop_i8 != 0 };
        let grid = if sm { [grid[0], nd.div_ceil(32), 1] } else { grid };
        let tail = [&(rows as u32) as &dyn Arg, &nk, &act, &w_scale];
        // gemmi loads q8 rows 16 bytes at a time: their stride (cols * 11 / 8) is aligned only for cols % 128 == 0.
        if f {
            if !same { q8.1 = None; }
            let args: [&dyn Arg; 12] = [&tiles, &wd, &x, &xrow, &y, &(nd as u32), &(x_stride as u32), &(y_stride as u32), tail[0], tail[1], tail[2], tail[3]];
            if tiles != 0 && self.r11_iq32(ty) {
                return self.run("gemm_f_dp_ps5_iq32", &spec, grid, &args);
            }
            self.run(if !self.0.coop { "gemm_f_dp" } else if sm { "gemm_fs" } else { "gemm_f" }, &spec, grid, &args)?;
            // Tile lists: gemm_f skips the tiles of at most 32 positions, which gemm_fs takes (as gemms_* on CUDA).
            if self.0.coop && tiles != 0 { self.run("gemm_fs", &spec, grid, &args)?; }
            return Ok(());
        }
        // Only SAME_X promises that x still holds what was quantized (as on CUDA).
        if !same { q8.1 = None; }
        let xq = self.q8_x(&mut q8, ty, cols, n_x, x, x_stride)?.expect("int8 GEMM without a quantized x");
        // A step of a batch of requests (at most 64 rows of x): gemmi_s, K split over ks workgroups so that the few
        // weight-row tiles fill the device, each slice into the scratch, then gemm_red sums them (as gemmd on CUDA).
        if batched && tiles == 0 && xrow == 0 && nd <= 64 && self.0.coop_i8 != 0 && !self.0.st().batch {
            // More than 32 positions: gemmi_s2, one tile of 64, so each weight is decoded once.
            let s2 = nd > 32;
            let g = [rows.div_ceil(64), nd.div_ceil(if s2 { 64 } else { 32 }), 1];
            let ks = (SPLIT_WG / (g[0] * g[1])).clamp(1, 8).min(nk as usize / 4).max(1);
            let (sk, scr) = if ks > 1 { (ks, self.5.get_or_init(|| self.alloc(SPLIT_SCR).expect("split-K scratch")).addr()) } else { (1, y) };
            let (ys, a) = if ks > 1 { (rows, 0) } else { (y_stride, act) };
            let args: [&dyn Arg; 11] = [&tiles, &wd, &xq, &xrow, &scr, &(nd as u32), &(ys as u32), tail[0], tail[1], &a, tail[3]];
            self.run(match (s2, self.0.coop_i8 == 32) { (false, true) => "gemmi_s", (false, false) => "gemmi_s_16", (true, true) => "gemmi_s2", (true, false) => "gemmi_s2_16" }, &spec, [g[0], g[1], sk], &args)?;
            if ks > 1 {
                self.run("gemm_red", &[], [(rows * nd).div_ceil(256), 1, 1],
                    &[&scr, &y, &(rows as u32), &(nd as u32), &(y_stride as u32), &(ks as u32), &act])?;
            }
            return Ok(());
        }
        let args: [&dyn Arg; 11] = [&tiles, &wd, &xq, &xrow, &y, &(nd as u32), &(y_stride as u32), tail[0], tail[1], tail[2], tail[3]];
        let (gk, gks) = match self.0.coop_i8 { 32 => ("gemmi", "gemmi_s"), 16 => ("gemmi_16", "gemmi_s_16"), _ => ("gemmi_dp", "gemmi_dp") };
        if tiles != 0 {
            let tile = self.r12_iq_tile(ty);
            if tile != 0 {
                // Original routed tile list is bounded to 128 positions. z selects
                // a disjoint subtile; each output still reduces the entire K dimension.
                return self.run(if tile == 32 { "gemmi_dp_ps5_iq32" } else { "gemmi_dp_ps5_iq64" },
                    &spec, [grid[0], grid[1], crate::backend::GEMM_BM / tile], &args);
            }
        }
        if self.0.r2.tiled64(self.0.coop_i8, tiles, nd) {
            // SM=2 uses two 16-column fragments per logical wave and 64 dense positions.
            // No split-K (z=1); original tile-list and cooperative routes remain untouched.
            return self.run("gemmi_dp_ps5_64", &spec, [grid[0], nd.div_ceil(64), 1], &args);
        }
        if self.0.r2.small_dense(self.0.coop_i8, tiles, nd) {
            return self.run("gemmi_dp_ps5_32", &spec, [grid[0], nd.div_ceil(32), 1], &args);
        }
        self.run(if self.0.coop_i8 == 0 { "gemmi_dp" } else if sm { gks } else { gk }, &spec, grid, &args)?;
        if self.0.coop_i8 != 0 && tiles != 0 { self.run(gks, &spec, grid, &args)?; }
        Ok(())
    }
}

// quant_q8 and gemmi use uint products before dividing by four; reject truncation before allocating or dispatching.
fn q8_gemm_bytes(n: usize, cols: usize, stride: usize) -> anyhow::Result<usize> {
    ensure!(n > 0 && cols > 0 && cols % 128 == 0 && stride >= cols, "invalid Q8 GEMM geometry");
    let limit = u32::MAX as usize;
    ensure!(n.checked_mul(stride).is_some_and(|v| v <= limit), "Q8 input offsets exceed shader uint range");
    let bytes = cols.checked_add(cols / 8).and_then(|v| v.checked_add(cols / 4)).and_then(|v| v.checked_mul(n))
        .filter(|&v| v <= limit).context("Q8 GEMM byte offsets exceed shader uint range")?;
    Ok(bytes)
}

#[test]
fn r12_audit_q8_geometry() {
    assert_eq!(q8_gemm_bytes(512, 2048, 2048).unwrap(), 1441792);
    for (n, c, s) in [(0,128,128),(1,0,0),(1,64,64),(1,128,127),(usize::MAX,128,128), (1<<24,256,256), (1<<24,128,512)] {
        assert!(q8_gemm_bytes(n,c,s).is_err());
    }
}

/// GEMV column count: a multiple of 32 and of the type's block (a partial last super-block is read by the lanes
/// inside it). llama numerics take one x scale per 256 values, so they need whole super-blocks.
fn gemv_cols(llama: u32, ty: GgmlType, cols: usize) -> anyhow::Result<u32> {
    let b = if llama != 0 { 256 } else { ty.block_elems().max(32) };
    ensure!(cols % b == 0, "{} tensor with {cols} columns: GEMV needs a multiple of {b}", ty.name());
    Ok(cols as u32)
}

#[allow(unused_variables)]
impl Backend for Vk {
    fn name(&self) -> &'static str { "vulkan" }
    fn ps5_preset(&self) -> bool { self.0.ps5 }
    fn device(&self) -> String {
        let p = unsafe { self.0.instance.get_physical_device_properties(self.0.pd) };
        p.device_name_as_c_str().map_or(String::new(), |s| s.to_string_lossy().into_owned())
    }
    fn share(&self) -> anyhow::Result<Box<dyn Backend>> {
        let v = Vk(
            self.0.clone(),
            self.1.clone(),
            self.2.clone(),
            Mutex::new((Buf::new(0, 0, ()), None)),
            Mutex::new((Buf::new(0, 0, ()), None)),
            Default::default(),
        );
        v.4.lock().unwrap().0 = v.alloc(1 << 20)?;
        Ok(Box::new(v))
    }

    fn alloc(&self, bytes: usize) -> anyhow::Result<Buf> { Ok(self.buf(bytes, DEVICE)?.0) }

    fn upload_immutable(&self, t: &crate::weights::Tensor, source: &crate::gguf::Gguf) -> anyhow::Result<GpuTensor> {
        if let Some(b) = self.try_import_weight(t, source)? {
            return Ok(GpuTensor::new(t.ty, t.rows, t.cols, b).with_scale(t.scale));
        }
        self.upload(t)
    }

    fn upload_bytes(&self, src: &[u8]) -> anyhow::Result<Buf> {
        // New buffer, never used by a submitted or recorded command: the next queue
        // submission makes coherent host writes available to device operations.
        // Do not use this path for live expert slots, KV, scratch or mailbox updates.
        if self.0.ps5_direct && src.len() > 65536 {
            if let Ok((buf, mem)) = self.0.buffer(src.len(), MAPPED_DEVICE) {
                match unsafe { self.0.dev.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) } {
                    Ok(p) => {
                        unsafe {
                            let p = p.cast::<u8>();
                            std::ptr::copy_nonoverlapping(src.as_ptr(), p, src.len());
                            // Match the old zeroed allocation's tail, up to Vulkan's 4-byte alignment.
                            p.add(src.len()).write_bytes(0, src.len().next_multiple_of(4) - src.len());
                            self.0.dev.unmap_memory(mem);
                        }
                        let addr = unsafe { self.0.dev.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf)) };
                        self.0.st().bufs.insert(addr, (buf, src.len()));
                        self.0.ps5_upload_bytes[0].fetch_add(src.len() as u64, std::sync::atomic::Ordering::Relaxed);
                        return Ok(Buf::new(addr, src.len(), Mem { vk: self.0.clone(), buf, mem, addr, imported: None }));
                    }
                    Err(_) => unsafe { self.0.dev.destroy_buffer(buf, None); self.0.dev.free_memory(mem, None); },
                }
            }
        }
        let b = self.alloc(src.len())?;
        self.htod(src, &b, 0)?;
        if self.0.ps5 { self.0.ps5_upload_bytes[1].fetch_add(src.len() as u64, std::sync::atomic::Ordering::Relaxed); }
        Ok(b)
    }

    fn htod(&self, src: &[u8], dst: &Buf, off: usize) -> anyhow::Result<()> {
        if self.0.r7_trace { self.0.r7_commands.upload_bytes.fetch_add(src.len() as u64, std::sync::atomic::Ordering::Relaxed); }
        ensure!(!dst.is_read_only(), "cannot write imported immutable GGUF weights");
        ensure!(off.checked_add(src.len()).is_some_and(|end| end <= dst.len()), "htod: {} bytes at {off} into {}", src.len(), dst.len());
        let mut st = self.0.st();
        if st.step_pos.is_some_and(|(a, _)| dst.addr() + off as u64 == a) { st.step_pos = None; }
        // A small copy goes inline into the open command buffer, with no staging round trip and no wait: the next
        // submit (a graph replay or a flush) carries it.
        if !src.is_empty() && src.len() <= 65536 && src.len() % 4 == 0 && st.rec.is_none() {
            let (b, o) = st.find(dst.addr() + off as u64, src.len())?;
            if o % 4 == 0 {
                let cb = self.0.open_cb(&mut st)?;
                self.0.barrier(cb, false);
                unsafe { self.0.dev.cmd_update_buffer(cb, b, o, src) };
                self.0.barrier(cb, false);
                return Ok(());
            }
        }
        // Larger copies fill `staging` front to back and wait only when it is full, so a run of uploads (experts
        // streamed in prefill) overlaps each copy on the GPU with the host filling the next.
        ensure!(st.rec.is_none(), "a host copy while recording a graph");
        if st.beside {
            for (i, c) in src.chunks(STAGING).enumerate() {
                if st.staged + c.len() > STAGING {
                    // Wrap to the start of the ring, past the copies of the open command buffer.
                    if st.cstaged { self.0.flush(&mut st)?; } else { self.0.tsubmit(&mut st)?; st.staged = 0; }
                }
                let at = st.staged;
                self.0.ring_wait(&mut st, at, c.len())?;
                let (b, o) = st.find(dst.addr() + (off + i * STAGING) as u64, c.len())?;
                unsafe { std::ptr::copy_nonoverlapping(c.as_ptr(), (st.map + at) as *mut u8, c.len()) };
                let cb = self.0.topen(&mut st)?;
                let r = vk::BufferCopy::default().src_offset(at as u64).dst_offset(o).size(c.len() as u64);
                unsafe { self.0.dev.cmd_copy_buffer(cb, st.staging.0, b, &[r]) };
                st.staged = (at + c.len()).next_multiple_of(256);
                st.tbytes += c.len();
                if st.tbytes >= SUBMIT_BYTES { self.0.tsubmit(&mut st)?; }
            }
            return Ok(());
        }
        for (i, c) in src.chunks(STAGING).enumerate() {
            if st.staged + c.len() > STAGING { self.0.flush(&mut st)?; }
            // Submit the work recorded so far first, so the GPU runs it while the host fills staging.
            if st.cur.is_some() && st.ops != st.copied.0 { self.0.submit(&mut st, None, false)?; }
            let (b, o) = st.find(dst.addr() + (off + i * STAGING) as u64, c.len())?;
            let at = st.staged;
            self.0.ring_wait(&mut st, at, c.len())?;
            unsafe { std::ptr::copy_nonoverlapping(c.as_ptr(), (st.map + at) as *mut u8, c.len()) };
            let cb = self.0.open_cb(&mut st)?;
            let r = vk::BufferCopy::default().src_offset(at as u64).dst_offset(o).size(c.len() as u64);
            unsafe { self.0.dev.cmd_copy_buffer(cb, st.staging.0, b, &[r]) };
            // Inside `batch` the copy may overlap the dispatches after it; the batch ends with a barrier.
            if !st.batch { self.0.barrier(cb, false); }
            st.staged = (at + c.len()).next_multiple_of(256);
            st.cstaged = true;
            st.copied = (st.ops, st.copied.1 + c.len());
            if st.copied.1 >= SUBMIT_BYTES { self.0.submit(&mut st, None, false)?; }
        }
        Ok(())
    }

    fn dtoh(&self, src: &Buf, off: usize, dst: &mut [u8]) -> anyhow::Result<()> {
        if self.0.r7_trace { self.0.r7_commands.download_bytes.fetch_add(dst.len() as u64, std::sync::atomic::Ordering::Relaxed); }
        ensure!(off.checked_add(dst.len()).is_some_and(|end| end <= src.len()), "dtoh: {} bytes at {off} from {}", dst.len(), src.len());
        let mut st = self.0.st();
        for (i, c) in dst.chunks_mut(STAGING).enumerate() {
            let (b, o) = st.find(src.addr() + (off + i * STAGING) as u64, c.len())?;
            self.0.ring_wait(&mut st, 0, c.len())?;  // transfer-queue copies may still read staging
            let cb = self.0.cb(&mut st)?;
            let r = vk::BufferCopy::default().src_offset(o).size(c.len() as u64);
            unsafe { self.0.dev.cmd_copy_buffer(cb, b, st.staging.0, &[r]) };
            self.0.barrier(cb, true);
            self.0.flush(&mut st)?;
            unsafe { std::ptr::copy_nonoverlapping(st.map as *const u8, c.as_mut_ptr(), c.len()) };
        }
        Ok(())
    }

    fn dtod(&self, src: &Buf, src_off: usize, dst: &Buf, dst_off: usize, bytes: usize) -> anyhow::Result<()> {
        if self.0.r7_trace { self.0.r7_commands.device_copy_bytes.fetch_add(bytes as u64, std::sync::atomic::Ordering::Relaxed); }
        ensure!(!dst.is_read_only(), "cannot write imported immutable GGUF weights");
        ensure!(src_off.checked_add(bytes).is_some_and(|end| end <= src.len())
            && dst_off.checked_add(bytes).is_some_and(|end| end <= dst.len()), "dtod: {bytes} bytes out of range");
        if bytes == 0 { return Ok(()); }
        let mut st = self.0.st();
        let (s, so) = st.find(src.addr() + src_off as u64, bytes)?;
        let (d, d_o) = st.find(dst.addr() + dst_off as u64, bytes)?;
        let cb = self.0.cb(&mut st)?;
        unsafe { self.0.dev.cmd_copy_buffer(cb, s, d, &[vk::BufferCopy::default().src_offset(so).dst_offset(d_o).size(bytes as u64)]) };
        if !st.batch { self.0.barrier(cb, false); }
        Ok(())
    }

    fn memset(&self, dst: &Buf) -> anyhow::Result<()> {
        ensure!(!dst.is_read_only(), "cannot clear imported immutable GGUF weights");
        let mut st = self.0.st();
        let (b, _) = st.find(dst.addr(), dst.len())?;
        let cb = self.0.cb(&mut st)?;
        unsafe { self.0.dev.cmd_fill_buffer(cb, b, 0, vk::WHOLE_SIZE, 0) };
        self.0.barrier(cb, false);
        Ok(())
    }

    /// Free from `VK_EXT_memory_budget` when the driver has it, else the heap size less this backend's buffers.
    fn mem_info(&self) -> anyhow::Result<(usize, usize)> {
        let m = &self.0.mem;
        let h = (0..m.memory_heap_count as usize).find(|&h| m.memory_heaps[h].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .context("no device-local heap")?;
        let total = m.memory_heaps[h].size as usize;
        if !self.0.budget { return Ok((total.saturating_sub(self.0.st().bufs.values().map(|b| b.1).sum()), total)); }
        let b = self.0.heap_budget();
        if !self.0.ps5_gtt {
            return Ok((self.0.heap_free(&b, h) as usize, total));
        }
        // Count each eligible non-local heap once, even if it has several HOST types.
        // Do not add a second copy of a unified/device-local heap to the total.
        let mut seen = [false; 16];
        let (mut host_free, mut host_total) = (0u64, 0u64);
        for t in &m.memory_types[..m.memory_type_count as usize] {
            let i = t.heap_index as usize;
            if seen[i] || m.memory_heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                || !HOST.iter().any(|&want| t.property_flags.contains(want))
                || !(t.property_flags & vk::MemoryPropertyFlags::from_raw(0xc0)).is_empty() { continue; }
            seen[i] = true;
            host_free = host_free.saturating_add(self.0.heap_free(&b, i));
            host_total = host_total.saturating_add(m.memory_heaps[i].size);
        }
        let free = self.0.heap_free(&b, h).saturating_add(host_free.min(self.0.ps5_host_room()));
        let total = (total as u64).saturating_add(host_total);
        Ok((free.min(total) as usize, total as usize))
    }

    fn sync(&self) -> anyhow::Result<()> { self.0.flush(&mut self.0.st()) }

    /// A reusable command buffer per key, submitted without waiting. `STRATA_NO_GRAPH` runs `f` every time instead.
    fn batch(&self, f: &mut dyn FnMut() -> anyhow::Result<()>) -> anyhow::Result<()> {
        self.0.st().batch = true;
        let r = f();
        let mut st = self.0.st();
        st.batch = false;
        let cb = self.0.cb(&mut st)?;
        self.0.barrier(cb, false);
        r
    }
    fn beside_overlaps(&self) -> bool { self.0.tq.is_some() }
    fn stalled(&self) -> bool { std::mem::take(&mut self.0.st().stalled) }
    /// With a transfer queue: the work so far is submitted, the copies of `f` go to the transfer queue after it, and
    /// `f`'s dispatches are submitted without waiting for them; the compute submit after the next `beside` waits for them.
    fn beside(&self, f: &mut dyn FnMut() -> anyhow::Result<()>) -> anyhow::Result<()> {
        if self.0.tq.is_none() { return f(); }
        {
            let mut st = self.0.st();
            ensure!(st.rec.is_none() && !st.batch, "beside inside a graph or batch");
            self.0.submit(&mut st, None, false)?;
            st.bwait = st.cval;
            st.beside = true;
        }
        let r = f();
        let mut st = self.0.st();
        st.beside = false;
        self.0.tsubmit(&mut st)?;
        self.0.submit(&mut st, None, false)?;
        let tv = st.tval;
        st.twait = std::mem::replace(&mut st.tlag, tv);
        r
    }
    fn kick(&self) -> anyhow::Result<()> {
        let mut st = self.0.st();
        if st.rec.is_some() || st.batch || st.beside { return Ok(()); }
        self.0.submit(&mut st, None, false)
    }
    fn beside_join(&self) -> anyhow::Result<()> {
        if self.0.tq.is_none() { return Ok(()); }
        let mut st = self.0.st();
        st.twait = st.tval;
        st.tlag = 0;
        Ok(())
    }
    fn graph(&self, key: u64, f: &mut dyn FnMut() -> anyhow::Result<()>) -> anyhow::Result<()> {
        let mut st = self.0.st();
        if let Some(&g) = st.graphs.get(&key) {
            if self.0.r7_trace { self.0.r7_commands.graph_reuse.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
            return self.0.enqueue_cb(&mut st, Some(g));
        }
        if std::env::var_os("STRATA_NO_GRAPH").is_some() { drop(st); return f(); }
        let c = unsafe { self.0.dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(st.gpool)
            .command_buffer_count(1)) }?[0];
        unsafe { self.0.dev.begin_command_buffer(c, &vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)) }?;
        st.rec = Some(c);
        if st.need_cond { self.0.cond_on(&mut st, c); }
        drop(st);
        let r = f();
        let mut st = self.0.st();
        st.rec = None;
        self.0.end_cond(&mut st, c);
        unsafe { self.0.dev.end_command_buffer(c) }?;
        if let Err(e) = r { unsafe { self.0.dev.free_command_buffers(st.gpool, &[c]) }; return Err(e); }
        st.graphs.insert(key, c);
        if self.0.r7_trace { self.0.r7_commands.graph_create.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
        self.0.enqueue_cb(&mut st, Some(c))
    }
    fn graph_drop(&self, key: u64) -> anyhow::Result<()> {
        let mut st = self.0.st();
        let Some(c) = st.graphs.remove(&key) else { return Ok(()) };
        // The buffer may still be queued or running.
        self.0.flush(&mut st)?;
        unsafe { self.0.dev.free_command_buffers(st.gpool, &[c]) };
        Ok(())
    }
    fn graph_clear(&self) -> anyhow::Result<()> {
        let mut st = self.0.st();
        if st.graphs.is_empty() { return Ok(()) }
        self.0.flush(&mut st)?;
        let cs: Vec<_> = st.graphs.drain().map(|(_, c)| c).collect();
        unsafe { self.0.dev.free_command_buffers(st.gpool, &cs) };
        Ok(())
    }

    fn moe_mailbox_optional(&self) -> bool { true }
    fn verify_rows(&self) -> usize { crate::cpu_worker::TMAX }

    /// Device memory mapped through the BAR (`BAR`, or `BAR_AMD`).
    fn host_mailbox(&self, words: usize) -> anyhow::Result<(Buf, *mut u32)> {
        let (b, p) = self.buf(4 * words.max(1), if self.0.amd_coherent { BAR_AMD } else { BAR })?;
        Ok((b, p.cast()))
    }
    fn preflight(&self, types: &[GgmlType], verbose: bool) -> anyhow::Result<()> { Vk::preflight(self, types, verbose) }
    fn kv_words(&self, n: usize) -> usize {
        match self.0.kv_k_mode { 0 => n, 1 => n * 17 / 32, 2 | 4..=6 => n * 9 / 32, 3 => n * 3 / 16, _ => unreachable!() }
    }
    fn kv_words_v(&self, n: usize) -> usize {
        match self.0.kv_v_mode { 0 => n, 1 => n * 17 / 32, 2 | 4..=6 => n * 9 / 32, 3 => n * 3 / 16, _ => unreachable!() }
    }

    fn gemm(&self, ty: GgmlType, rows: usize, cols: usize, tiles: u64, n_tiles: usize, x: u64, x_stride: usize, n_x: usize, xrow: u64, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()> {
        self.gemm_raw(ty, rows, cols, tiles, n_tiles, 0, 0, x, x_stride, n_x, xrow, y, y_stride, act, 1.0f32)
    }
    fn gemm_dense(&self, w: &GpuTensor, n: usize, x: u64, x_stride: usize, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()> {
        self.gemm_raw(w.ty, w.rows, w.cols, 0, n.div_ceil(128), w.ptr, n, x, x_stride, n, 0, y, y_stride, act, w.scale)
    }
    fn int8_gemm(&self, ty: GgmlType) -> bool { crate::cuda::Gpu::mmq(ty) }

    fn prequant_gemm_x(&self, w: &GpuTensor, n: usize, x: u64, x_stride: usize) -> anyhow::Result<()> {
        let mut q8 = self.q8_mutex(n, w.cols).lock().unwrap();
        self.q8_x(&mut q8, w.ty, w.cols, n, x, x_stride)?;
        Ok(())
    }
    fn gemv(&self, w: &GpuTensor, x: u64, y: u64) -> anyhow::Result<()> { self.gemvq(w, x, 0, 0, y, 0, 1) }
    fn gemvn(&self, w: &GpuTensor, x: u64, x_stride: usize, y: u64, y_stride: usize, nt: usize) -> anyhow::Result<()> {
        self.gemvq(w, x, x_stride, 0, y, y_stride, nt)
    }
    fn gemvq(&self, w: &GpuTensor, x: u64, x_stride: usize, xq: u64, y: u64, y_stride: usize, nt: usize) -> anyhow::Result<()> {
        ensure!((1..=crate::cpu_worker::TMAX).contains(&nt), "gemvn: {nt} inputs (1 to {})", crate::cpu_worker::TMAX);
        let (nc, spec) = (gemv_cols(self.0.llama, w.ty, w.cols)?, [&self.wspec(w.ty)[..], &[nt as u32]].concat());
        if w.ty.float_x() {
            // Rows per workgroup as CUDA gemv_cfg: 8 for tall matrices, else 1 (hc_attn_up 10240 x 320).
            let wpb = if w.rows >= 4096 { 8 } else { 1 };
            let spec = [&spec[..], &[32 * wpb as u32]].concat();
            return self.run("gemv_f", &spec, [w.rows.div_ceil(wpb), 1, 1], &[&w.ptr, &x, &y, &(x_stride as u32), &(y_stride as u32), &nc, &(w.rows as u32), &w.scale]);
        }
        let xq = if xq != 0 { xq } else {
            ensure!(nt * w.cols * 9 / 8 <= XQ, "gemv: {nt} x {} inputs do not fit the quantized-x buffer", w.cols);
            self.quant_xq(x, x_stride, w.cols, nt, self.1.addr())?;
            self.1.addr()
        };
        let (kernel, wpb) = self.0.r2.gemv(nt, false);
        self.run(kernel, &spec, [w.rows.div_ceil(wpb), 1, 1], &[&w.ptr, &xq, &y, &0u64, &0u64, &0u64, &(y_stride as u32), &(w.rows as u32), &0u32, &nc, &w.scale, &1.0f32])
    }
    fn gemvq2(&self, a: &GpuTensor, b: &GpuTensor, x: u64, x_stride: usize, xq: u64, ya: u64, yb: u64, nt: usize) -> anyhow::Result<()> {
        if a.ty != b.ty || a.cols != b.cols || a.ty.float_x() {
            self.gemvq(a, x, x_stride, xq, ya, a.rows, nt)?;
            return self.gemvq(b, x, x_stride, xq, yb, b.rows, nt);
        }
        ensure!((1..=crate::cpu_worker::TMAX).contains(&nt), "gemvq2: {nt} inputs (1 to {})", crate::cpu_worker::TMAX);
        let (nc, spec) = (gemv_cols(self.0.llama, a.ty, a.cols)?, [&self.wspec(a.ty)[..], &[nt as u32]].concat());
        let xq = if xq != 0 { xq } else {
            ensure!(nt * a.cols * 9 / 8 <= XQ, "gemvq2: {nt} x {} inputs do not fit the quantized-x buffer", a.cols);
            self.quant_xq(x, x_stride, a.cols, nt, self.1.addr())?;
            self.1.addr()
        };
        let (kernel, wpb) = self.0.r2.gemv(nt, false);
        self.run(kernel, &spec, [(a.rows + b.rows).div_ceil(wpb), 1, 1],
            &[&a.ptr, &xq, &ya, &b.ptr, &yb, &0u64, &(a.rows as u32), &(a.rows as u32), &(b.rows as u32), &nc, &a.scale, &b.scale])
    }
    fn supports_gemv_rows(&self) -> bool { true }
    fn gemv_rows(&self, w: &GpuTensor, x: u64, ids: u64, n_rows: usize, y: u64) -> anyhow::Result<()> {
        self.gemv_rows_n(w, x, w.cols, 0, ids, n_rows, y, 1)
    }
    fn gemv_rows_n(&self, w: &GpuTensor, x: u64, x_stride: usize, xq: u64, ids: u64, n_rows: usize, y: u64, nt: usize) -> anyhow::Result<()> {
        ensure!(!w.ty.float_x(), "gemv_rows: float weights unsupported");
        ensure!((1..=crate::cpu_worker::TMAX).contains(&nt), "gemv_rows_n: {nt} inputs (1 to {})", crate::cpu_worker::TMAX);
        let (nc, spec) = (gemv_cols(self.0.llama, w.ty, w.cols)?, [&self.wspec(w.ty)[..], &[nt as u32]].concat());
        let xq = if xq != 0 { xq } else {
            ensure!(nt * w.cols * 9 / 8 <= XQ, "gemv_rows: {nt} x {} do not fit quantized-x buffer", w.cols);
            self.quant_xq(x, x_stride, w.cols, nt, self.1.addr())?;
            self.1.addr()
        };
        let (kernel, wpb) = self.0.r2.gemv(nt, false);
        self.run(kernel, &spec, [n_rows.div_ceil(wpb), 1, 1], &[&w.ptr, &xq, &y, &0u64, &0u64, &ids, &(n_rows as u32), &(n_rows as u32), &0u32, &nc, &w.scale, &1.0f32])
    }
    fn moe_gemv(&self, ty: GgmlType, rows: usize, cols: usize, w: u64, stride: usize, ids: u64, k: usize, x: u64, x_stride: usize, xg: usize, y: u64, pair: Option<(i64, i64)>, xq: u64) -> anyhow::Result<()> {
        self.moe_gemv_nm(ty, rows, cols, w, stride, ids, k, x, x_stride, xg, y, pair, xq, k / xg.max(1))
    }
    fn dequant(&self, w: &GpuTensor, rows: usize, out: u64) -> anyhow::Result<()> {
        self.run("dequant", &self.wspec(w.ty), [rows, 1, 1], &[&w.ptr, &out, &gemv_cols(0, w.ty, w.cols)?, &w.scale])
    }
    fn quant_xq(&self, x: u64, x_stride: usize, n: usize, nt: usize, out: u64) -> anyhow::Result<()> {
        self.run("quant_xq", &[self.0.llama], [n.div_ceil(256), nt, 1], &[&x, &out, &0u64, &(x_stride as u32), &(n as u32)])
    }
    // The SwiGLU computed while quantizing x for the GEMV: one dispatch fewer (on CUDA a separate quantization lost).
    fn swiglu_q(&self, g: u64, u: u64, n: usize, rows: usize, xq: u64) -> anyhow::Result<u64> {
        self.run("swiglu_q", &[self.0.llama], [n.div_ceil(256), rows, 1], &[&g, &xq, &u, &(n as u32), &(n as u32)])?;
        Ok(xq)
    }
    fn rms_norm(&self, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, n: usize, rows: usize, eps: f32) -> anyhow::Result<()> {
        self.rms_norm_q(x, x_stride, w, out, out_stride, n, rows, eps, 0, 0)
    }
    fn rms_norm_q(&self, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, n: usize, rows: usize, eps: f32, xq: u64, res: u64) -> anyhow::Result<()> {
        self.run("rms_norm", &[self.0.llama], [rows, 1, 1], &[&x, &w, &out, &xq, &(x_stride as u32), &(out_stride as u32), &(n as u32), &eps, &0u64, &0f64, &0u32, &0u32, &res, &0u64, &0u64, &0u64, &0u64])
    }
    fn rms_norm_rope(&self, x: u64, x_stride: usize, w: u64, out: u64, n: usize, rows: usize, eps: f32, per: usize, n_rot: usize, step: u64, base: f32) -> anyhow::Result<()> {
        self.rms_norm_rope_kv(x, x_stride, w, out, n, rows, eps, per, n_rot, step, base, 0, 0, 0)
    }
    fn rms_norm_rope_kv(&self, x: u64, x_stride: usize, w: u64, out: u64, n: usize, rows: usize, eps: f32, per: usize, n_rot: usize, step: u64, base: f32, v: u64, kc: u64, vc: u64) -> anyhow::Result<()> {
        self.rope_rows(x, x_stride, w, out, n, rows, eps, per, n_rot, step, base, v, kc, vc, 0)
    }
    fn kv_rows(&self) -> bool { self.0.llama == 0 }  // attn_decode_llama has no kvt
    fn rms_norm_rope_qk_rows(&self, q: u64, q_stride: usize, qw: u64, qo: u64, nh: usize, k: u64, kw: u64, ko: u64, nkv: usize, hd: usize, eps: f32, n_rot: usize,
                             step: u64, base: f32, v: u64, kvt: u64, nt: usize) -> anyhow::Result<()> {
        self.rope_rows(q, q_stride, qw, qo, hd, nt * nh, eps, nh, n_rot, step, base, 0, 0, 0, kvt)?;
        self.rope_rows(k, hd, kw, ko, hd, nt * nkv, eps, nkv, n_rot, step, base, v, 0, 0, kvt)
    }
    fn attn_rows(&self, q: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize, scale: f32, nt: usize, sink: u64, win: usize, kvt: u64, splits: usize) -> anyhow::Result<()> {
        self.attn(q, 0, 0, qg, out, part, step, nh, nkv, hd, scale, 0, nt, 0, 0, 0, sink, win, kvt, splits)
    }
    fn argmax_rows(&self, x: u64, n: usize, rows: usize, out: u64) -> anyhow::Result<()> {
        // One workgroup per row reads 1 MB (a 248320-token vocabulary) in ~25 us; 64 per row share it. The partials go
        // to the quantized-x scratch buffer, which no op reads before writing it.
        ensure!(rows * 64 * 8 <= XQ, "argmax_rows: {rows} rows");
        let part = self.1.addr();
        self.run("argmax_rows", &[0, 0], [64, rows, 1], &[&x, &part, &out, &(n as u32)])?;
        self.run("argmax_rows", &[0, 1], [rows, 1, 1], &[&x, &part, &out, &(n as u32)])
    }
    fn l2_norm(&self, x: u64, n: usize, per: usize, stride: usize, rows: usize, eps: f32) -> anyhow::Result<()> {
        self.run("l2_norm", &[], [rows, 1, 1], &[&x, &(n as u32), &(per as u32), &(stride as u32), &eps])
    }
    fn rope_pairs(&self, x: u64, stride: usize, heads: usize, per: usize, n_rot: usize, step: u64, base: f32, lo: usize, hi: usize) -> anyhow::Result<()> {
        let ts = (base as f64).powf(-2.0 / n_rot as f64);
        self.run("rope_neox", &[], [heads / per, 1, 1], &[&x, &step, &ts, &(stride as u32), &(per as u32), &(n_rot as u32), &(lo as u32), &(hi as u32)])
    }
    fn add(&self, x: u64, y: u64, n: usize) -> anyhow::Result<()> { self.run("add", &[], [n.div_ceil(256), 1, 1], &[&x, &y, &(n as u32)]) }
    fn swiglu(&self, g: u64, u: u64, n: usize) -> anyhow::Result<()> { self.run("swiglu", &[], [n.div_ceil(256), 1, 1], &[&g, &u, &(n as u32)]) }
    fn situ(&self, y: u64, n: usize, rows: usize, scale: u64, beta: f32, lb: f32) -> anyhow::Result<()> {
        self.run("situ", &[], [(n * rows).div_ceil(256), 1, 1], &[&y, &scale, &(n as u32), &(rows as u32), &beta, &lb])
    }
    fn situ_q(&self, y: u64, n: usize, rows: usize, s: u64, beta: f32, lb: f32, xq: u64) -> anyhow::Result<()> {
        ensure!(n % 32 == 0, "situ_q: n ({n}) must be a multiple of 32");
        self.run("situ_q", &[self.0.llama], [n.div_ceil(256), rows, 1], &[&y, &xq, &s, &(n as u32), &beta, &lb])
    }
    fn softcap(&self, x: u64, n: usize, cap: f32) -> anyhow::Result<()> {
        self.run("softcap", &[], [n.div_ceil(256), 1, 1], &[&x, &(n as u32), &cap])
    }
    fn ink_mix_n(&self, blocks: u64, nb: usize, cur: u64, q: u64, nq: usize, out: u64, n: usize, nt: usize) -> anyhow::Result<()> {
        self.run("ink_mix", &[], [nq, nt, 1], &[&blocks, &cur, &q, &out, &(nb as u32), &(n as u32), &(nt as u32)])
    }
    fn ink_ctrl_n(&self, down: u64, gain: u64, forget: u64, cur: u64, nins: usize, msg: u64, gate: u64, out: u64, n: usize, rank: usize, count: usize, nt: usize) -> anyhow::Result<()> {
        self.run("ink_ctrl_gate", &[], [rank, count, nt], &[&down, &cur, &msg, &gate, &(nins as u32), &(n as u32), &(rank as u32), &(nt as u32)])?;
        self.run("ink_ctrl_out", &[], [n.div_ceil(256), count, nt], &[&gain, &forget, &gate, &cur, &msg, &out, &(nins as u32), &(n as u32), &(rank as u32), &(nt as u32)])
    }
    fn ink_conv(&self, cur: u64, w: u64, taps: u64, hist: u64, out: u64, n: usize, n_conv: usize, per: usize) -> anyhow::Result<()> {
        self.run("ink_conv", &[], [n.div_ceil(256), 1, 1], &[&cur, &w, &taps, &hist, &out, &(n as u32), &(n_conv as u32), &(per as u32)])
    }
    fn ink_conv_batch(&self, cur: u64, w: u64, taps: u64, hist: u64, out: u64, n: usize, n_conv: usize, per: usize, nt: usize) -> anyhow::Result<()> {
        ensure!(cur != out, "ink_conv_batch: out must not alias cur");
        ensure!(n_conv <= 8, "ink_conv_batch: n_conv={n_conv} > 8");
        self.run("ink_conv_batch", &[], [n.div_ceil(256), nt, 1], &[&cur, &w, &taps, &hist, &out, &(n as u32), &(n_conv as u32), &(per as u32), &(nt as u32), &0u32])?;
        self.run("ink_conv_batch", &[], [n.div_ceil(256), 1, 1], &[&cur, &w, &taps, &hist, &out, &(n as u32), &(n_conv as u32), &(per as u32), &(nt as u32), &1u32])
    }
    fn ink_clip_n(
        &self, x: u64, scale: u64, w: usize, rows: usize, nt: usize,
        thr: f32, eps: f32, mul: f32,
        dst: u64, dst_at0: usize, dst_ring: usize,
        pos0: usize, log_alpha: f32, log_floor: f32,
    ) -> anyhow::Result<()> {
        ensure!(w <= 128, "ink_clip_n: w={w} > 128");
        let quant = |m: u32| m > 0;
        ensure!(!(quant(self.0.kv_k_mode) || quant(self.0.kv_v_mode)) || dst == 0 || w % 32 == 0, "ink_clip_n: quantized KV requires w % 32 == 0 (got {w})");
        let ring = if dst != 0 { dst_ring.max(1) } else { 1 };
        let fl = if log_alpha == 0.0 && log_floor == 0.0 { 1.0 } else { log_floor };
        self.run("ink_clip", &[], [rows, nt, 1], &[
            &x, &scale, &dst, &0u64, &(w as u32), &thr, &eps, &mul,
            &(dst_at0 as u32), &(ring as u32), &(pos0 as u32), &log_alpha, &fl, &0u32,
        ])
    }
    fn ink_clip_pos(
        &self, x: u64, scale: u64, w: usize, rows: usize, nt: usize,
        thr: f32, eps: f32, mul: f32,
        dst: u64, dst_ring: usize,
        pos_ptr: u64, log_alpha: f32, log_floor: f32,
    ) -> anyhow::Result<()> {
        ensure!(w <= 128, "ink_clip_pos: w={w} > 128");
        ensure!(!(self.0.kv_k_mode > 0 || self.0.kv_v_mode > 0) || dst == 0 || w % 32 == 0, "ink_clip_pos: quantized KV requires w % 32 == 0 (got {w})");
        let ring = if dst != 0 { dst_ring.max(1) } else { 1 };
        let fl = if log_alpha == 0.0 && log_floor == 0.0 { 1.0 } else { log_floor };
        self.run("ink_clip", &[], [rows, nt, 1], &[
            &x, &scale, &dst, &pos_ptr, &(w as u32), &thr, &eps, &mul,
            &0u32, &(ring as u32), &0u32, &log_alpha, &fl, &0u32,
        ])
    }
    fn ink_clip_n_v(
        &self, x: u64, scale: u64, w: usize, rows: usize, nt: usize,
        thr: f32, eps: f32, mul: f32,
        dst: u64, dst_at0: usize, dst_ring: usize,
        pos0: usize, log_alpha: f32, log_floor: f32,
    ) -> anyhow::Result<()> {
        ensure!(w <= 128, "ink_clip_n: w={w} > 128");
        let quant = self.0.kv_k_mode > 0 || self.0.kv_v_mode > 0;
        ensure!(!quant || dst == 0 || w % 32 == 0, "ink_clip_n: quantized KV requires w % 32 == 0 (got {w})");
        let ring = if dst != 0 { dst_ring.max(1) } else { 1 };
        let fl = if log_alpha == 0.0 && log_floor == 0.0 { 1.0 } else { log_floor };
        self.run("ink_clip", &[], [rows, nt, 1], &[
            &x, &scale, &dst, &0u64, &(w as u32), &thr, &eps, &mul,
            &(dst_at0 as u32), &(ring as u32), &(pos0 as u32), &log_alpha, &fl, &1u32,
        ])
    }
    fn ink_clip_pos_v(
        &self, x: u64, scale: u64, w: usize, rows: usize, nt: usize,
        thr: f32, eps: f32, mul: f32,
        dst: u64, dst_ring: usize,
        pos_ptr: u64, log_alpha: f32, log_floor: f32,
    ) -> anyhow::Result<()> {
        ensure!(w <= 128, "ink_clip_pos: w={w} > 128");
        let quant = self.0.kv_k_mode > 0 || self.0.kv_v_mode > 0;
        ensure!(!quant || dst == 0 || w % 32 == 0, "ink_clip_pos: quantized KV requires w % 32 == 0 (got {w})");
        let ring = if dst != 0 { dst_ring.max(1) } else { 1 };
        let fl = if log_alpha == 0.0 && log_floor == 0.0 { 1.0 } else { log_floor };
        self.run("ink_clip", &[], [rows, nt, 1], &[
            &x, &scale, &dst, &pos_ptr, &(w as u32), &thr, &eps, &mul,
            &0u32, &(ring as u32), &0u32, &log_alpha, &fl, &1u32,
        ])
    }
    fn ink_attn(&self, q: u64, r: u64, sink: u64, gate: u64, k: u64, v: u64, bias: u64, sc: u64, out: u64, rows: usize, at: usize, nh: usize, nkv: usize, hd: usize, nr: usize, ext: usize) -> anyhow::Result<()> {
        let chunks = rows.div_ceil(64);
        self.run("ink_attn", &[], [nh, chunks, 1], &[&q, &r, &sink, &k, &v, &bias, &sc, &0u64, &(rows as u32), &(at as u32), &1u32, &(nh as u32), &(nkv as u32), &(hd as u32), &(nr as u32), &(ext as u32)])?;
        self.run("ink_attn_merge", &[], [nh, 1, 1], &[&sc, &gate, &out, &(chunks as u32), &(hd as u32)])
    }
    fn ink_attn_pos(&self, q: u64, r: u64, sink: u64, gate: u64, k: u64, v: u64, bias: u64, sc: u64, out: u64, max_rows: usize, ring: usize, pos_ptr: u64, nh: usize, nkv: usize, hd: usize, nr: usize, ext: usize) -> anyhow::Result<()> {
        let chunks = max_rows.div_ceil(64);
        self.run("ink_attn", &[], [nh, chunks, 1], &[&q, &r, &sink, &k, &v, &bias, &sc, &pos_ptr, &0u32, &0u32, &(ring.max(1) as u32), &(nh as u32), &(nkv as u32), &(hd as u32), &(nr as u32), &(ext as u32)])?;
        self.run("ink_attn_merge", &[], [nh, 1, 1], &[&sc, &gate, &out, &(chunks as u32), &(hd as u32)])
    }
    fn ink_attn_prefill(
        &self, q: u64, r: u64, sink: u64, gate: u64, kc: u64, vc: u64, kc_new: u64, vc_new: u64, bias: u64, out: u64,
        pos: usize, nt: usize, nh: usize, nkv: usize, hd: usize, nr: usize, ext: usize, ring: usize, win: usize,
    ) -> anyhow::Result<()> {
        ensure!(hd > 0 && hd <= 128 && hd % 32 == 0, "ink_attn_prefill: unsupported hd={hd}");
        self.run("ink_attn_prefill", &[0, (hd / 32) as u32], [nt.div_ceil(4), nh, 1], &[
            &q, &r, &sink, &gate, &kc, &vc, &kc_new, &vc_new, &bias, &out,
            &(pos as u32), &(nt as u32), &(nh as u32), &(nkv as u32), &(nr as u32), &(ext as u32), &(ring.max(1) as u32), &(win as u32),
        ])
    }
    fn attn_sel(&self, q: u64, kc: u64, vc: u64, idx: u64, cnt: u64, out: u64, nt: usize, nh: usize, nkv: usize, hd: usize, ns: usize, scale: f32) -> anyhow::Result<()> {
        let per = nh / nkv.max(1);
        ensure!(hd % 32 == 0 && hd <= 256 && nkv > 0 && nh % nkv == 0 && (1..=8).contains(&per), "attn_sel: unsupported shape nh={nh} nkv={nkv} hd={hd}");
        self.run("attn_sel", &[0, hd as u32, per as u32, (32 * per) as u32], [nt, nkv, 1], &[&q, &kc, &vc, &idx, &cnt, &out, &(nh as u32), &(nkv as u32), &(ns as u32), &scale])
    }
    fn vit_attn(&self, q: u64, k: u64, v: u64, out: u64, n: usize, nh: usize, hd: usize, scale: f32) -> anyhow::Result<()> {
        ensure!(hd > 0 && hd <= 128, "vit_attn: head dim {hd} not supported");
        self.run("vit_attn", &[0, hd as u32], [n.div_ceil(32), nh, 1], &[&q, &k, &v, &out, &(n as u32), &(nh as u32), &scale])
    }
    fn ink_ffn_sum_n(&self, acc: u64, d: u64, wts: u64, ne: usize, n: usize, nt: usize, cpu: u64) -> anyhow::Result<()> {
        self.run("ink_ffn_sum", &[], [n.div_ceil(256), nt, 1], &[&acc, &d, &wts, &cpu, &(ne as u32), &(n as u32)])
    }
    fn ink_route(&self, logits: u64, bias: u64, ids: u64, wts: u64, gu_sc: u64, dn_sc: u64, gsc_out: u64, n_expert: usize, k: usize, exp_scale: f32, rows: usize) -> anyhow::Result<()> {
        ensure!(n_expert <= 512 && k <= 32, "ink_route: n_expert={n_expert} k={k}");
        self.run("ink_route", &[], [rows, 1, 1], &[&logits, &bias, &ids, &wts, &gu_sc, &dn_sc, &gsc_out, &(n_expert as u32), &(k as u32), &exp_scale])
    }
    fn hadamard(&self, x: u64, u: u64, s: u64, out: u64, n: usize, block: usize, rows: usize) -> anyhow::Result<()> {
        // ponytail: a block is one workgroup, and 128 invocations is what every Vulkan device takes; a larger block
        // needs maxComputeWorkGroupInvocations checked.
        ensure!(block <= 128, "prism.hadamard.block_size {block}: at most 128 on Vulkan");
        self.run("hadamard", &[self.0.llama, block as u32], [n / block, rows, 1], &[&x, &u, &s, &out, &(n as u32), &(1.0 / (block as f32).sqrt())])
    }
    fn ssm_gates(&self, beta: u64, g: u64, dt_bias: u64, a: u64, n: usize, nv: usize) -> anyhow::Result<()> {
        self.run("ssm_gates", &[], [n.div_ceil(256), 1, 1], &[&beta, &g, &dt_bias, &a, &(n as u32), &(nv as u32)])
    }
    fn conv_silu(&self, qkv: u64, state: u64, w: u64, y: u64, ch: usize, kw: usize, nn: usize, eps: f32) -> anyhow::Result<()> {
        self.run("conv_silu", &[], [ch.div_ceil(128), 1, 1], &[&qkv, &state, &w, &y, &(ch as u32), &(kw as u32), &(nn as u32), &eps, &0u64])
    }
    fn conv_silu_rows(&self, qkv: u64, tab: u64, states: &[u64], w: u64, y: u64, ch: usize, kw: usize, eps: f32) -> anyhow::Result<()> {
        self.run("conv_silu", &[], [ch.div_ceil(128), states.len(), 1], &[&qkv, &0u64, &w, &y, &(ch as u32), &(kw as u32), &0u32, &eps, &tab])
    }
    fn deltanet_rows(&self, y: u64, tab: u64, states: &[u64], beta: u64, g: u64, o: u64, nk: usize, nv: usize, interleave: bool, dk: usize, dv: usize, scale: f32, ych: usize) -> anyhow::Result<()> {
        let rows = 512 / dk.max(1);
        ensure!(matches!(dk, 32 | 64 | 128 | 256) && dv % rows == 0, "deltanet: head dims {dk}/{dv} unsupported");
        self.run("deltanet", &[], [nv, dv.div_ceil(8 * rows), states.len()], &[&y, &0u64, &beta, &g, &o, &0u64, &0u64, &0u64, &tab, &(nk as u32), &(nv as u32),
            &(interleave as u32), &(dk as u32), &(dv as u32), &scale, &1u32, &(ych as u32), &0u32])
    }
    fn deltanet(&self, y: u64, s: u64, beta: u64, g: u64, dt_bias: u64, a: u64, o: u64, nk: usize, nv: usize, interleave: bool, dk: usize, dv: usize, scale: f32, t: usize, ych: usize, snap: u64, nsnap: usize) -> anyhow::Result<()> {
        self.deltanet_strided(y, s, beta, g, dt_bias, a, o, nk, nv, interleave, dk, dv, scale, t, ych, snap, nsnap, 1)
    }
    fn delta_snap_stride(&self) -> usize { crate::gpu_model::SNAP_STRIDE }
    fn deltanet_strided(&self, y: u64, s: u64, beta: u64, g: u64, dt_bias: u64, a: u64, o: u64, nk: usize, nv: usize, interleave: bool, dk: usize, dv: usize, scale: f32, t: usize, ych: usize, snap: u64, nsnap: usize, snap_stride: usize) -> anyhow::Result<()> {
        let rows = 512 / dk.max(1);  // state rows per subgroup
        ensure!(matches!(dk, 32 | 64 | 128 | 256) && dv % rows == 0, "deltanet: head dims {dk}/{dv} unsupported");
        let spec: &[u32] = if snap_stride > 1 { &[0, snap_stride as u32] } else { &[] };
        self.run("deltanet", spec, [nv, dv.div_ceil(8 * rows), 1], &[&y, &s, &beta, &g, &o, &snap, &dt_bias, &a, &0u64, &(nk as u32), &(nv as u32),
            &(interleave as u32), &(dk as u32), &(dv as u32), &scale, &(t as u32), &(ych as u32), &(nsnap as u32)])
    }
    fn conv_seq(&self, qkv: u64, state: u64, w: u64, y: u64, ch: usize, kw: usize, t: usize, snap: u64, nsnap: usize) -> anyhow::Result<()> {
        self.run("conv_seq", &[], [(t * ch).div_ceil(256), 1, 1], &[&qkv, &state, &w, &y, &(ch as u32), &(kw as u32), &(t as u32)])?;
        self.run("conv_state", &[], [ch.div_ceil(256), 1, 1], &[&qkv, &state, &snap, &(ch as u32), &(kw as u32), &(t as u32), &(nsnap as u32)])
    }
    fn dflash2_conv(&self, x: u64, dyn_w: u64, base: u64, out: u64, h: usize, gs: usize, k: usize, t: usize, part: usize) -> anyhow::Result<()> {
        self.run("dflash2_conv", &[], [h.div_ceil(256), t, 1], &[&x, &dyn_w, &base, &out, &(h as u32), &(gs as u32), &(k as u32), &(t as u32), &(part as u32)])
    }
    fn attn_prefill_m(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, step: u64, t: usize, nh: usize, nkv: usize, hd: usize, scale: f32, qm: u64, mw: usize, sink: u64, win: usize) -> anyhow::Result<()> {
        let name = match hd { 64 => "attn_prefill_64", 128 => "attn_prefill_128", 256 => "attn_prefill_256",
            _ => anyhow::bail!("attention prefill: head_dim {hd} unsupported (64, 128 or 256)") };
        if t == 0 { return Ok(()); }
        ensure!(nkv > 0 && nh >= nkv && nh % nkv == 0, "attention prefill: invalid head geometry {nh}/{nkv}");
        // A chunk's prefill attention is t * (pos + t) query-key pairs, which can grow past amdgpu's 10 s job timeout
        // at a long context. Cut it into query slices, each its own submit, so no one job runs that long. The budget
        // is query-key pairs per slice; STRATA_VK_ATTN_BUDGET overrides it (0 turns slicing off). 2^30 does not split
        // the reported chunk 4096 at position 131072, so the default is 2^28: about 0.45 s on the 3060, and about
        // 3.5 s at the 8x slower AMD pace the report implies. That is the cooperative-matrix kernel. Without
        // cooperative matrices (tiled, dp, llama numerics) a pair costs about 8x more: measured 13 ns on the 3060
        // (attn_prefill_tile, Flash-Next, q8 KV), so 2^28 did not split chunk 4096 below position 61440 while one
        // job there already ran 2.5 s on the 3060 and past 10 s on RDNA2 (#42). Their default is 2^25, 0.44 s on
        // the 3060.
        // ponytail: fixed budgets from one card; time the first slice and adapt if a slower device still resets.
        let pos = self.step_pos(step)?;
        let default = if self.0.coop && self.0.llama == 0 { 1 << 28 } else { 1 << 25 };
        let budget = std::env::var("STRATA_VK_ATTN_BUDGET").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(default);
        let per_slice = if budget == 0 { t } else { (budget / pos.saturating_add(t).max(1)).max(1).min(t) };
        let sliced = per_slice < t;
        let run = |q_off: usize, n: usize| -> anyhow::Result<()> {
            let q = q + (4 * nh * hd) as u64 * q_off as u64;
            let out = out + (4 * nh * hd) as u64 * q_off as u64;
            let qg = if qg != 0 { qg + (4 * nh * 2 * hd) as u64 * q_off as u64 } else { 0 };
            if self.0.llama != 0 {
                return self.run("attn_prefill_llama", &[0, hd as u32], [n, nh, 1], &[&q, &kc, &vc, &qg, &out, &step, &(n as u32), &(nh as u32), &(nkv as u32), &scale, &qm, &(mw as u32), &(q_off as u32), &sink, &(win as i32)]);
            }
            if !self.0.coop {
                if self.0.r5.wg != 0 || self.0.dp_tile {
                    // One workgroup: 8/16/32 (query, head) tasks of one KV head, sharing its K/V tile.
                    // hper heads, bq queries each; separate head groups when GQA exceeds the task count.
                    let g = nh / nkv;
                    let (kernel, tasks) = self.0.r5.kernel();
                    let hper = g.min(tasks);
                    let bq = tasks / hper;
                    let hgroups = g.div_ceil(hper);
                    if self.0.r5.wg != 0 && !self.0.r5_logged.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        eprintln!("PS5 R5 attention dispatch: {kernel}");
                    }
                    return self.run(kernel, &[0, hd as u32], [n.div_ceil(bq), nkv * hgroups, 1],
                        &[&q, &kc, &vc, &qg, &out, &step, &(n as u32), &(nh as u32), &(nkv as u32), &scale, &qm, &(mw as u32),
                          &(bq as u32), &(hper as u32), &(hgroups as u32), &(q_off as u32), &sink, &(win as i32)]);
                }
                return self.run("attn_prefill_dp", &[0, (hd / 32) as u32], [n.div_ceil(4), nh, 1], &[&q, &kc, &vc, &qg, &out, &step, &(n as u32), &(nh as u32), &(nkv as u32), &scale, &qm, &(mw as u32), &(q_off as u32), &sink, &(win as i32)]);
            }
            self.run(name, &[], [n.div_ceil(64), nh, 1], &[&q, &kc, &vc, &qg, &out, &step, &(n as u32), &(nh as u32), &(nkv as u32), &scale, &qm, &(mw as u32), &(q_off as u32), &sink, &(win as i32)])
        };
        for q_off in (0..t).step_by(per_slice) {
            run(q_off, per_slice.min(t - q_off))?;
            if sliced {
                let mut st = self.0.st();
                self.0.submit(&mut st, None, false)?;
            }
        }
        Ok(())
    }
    fn moe_combine_seq(&self, x: u64, ye: u64, wts: u64, inv: u64, k: usize, sh: u64, sg: u64, n: usize, t: usize) -> anyhow::Result<()> {
        self.run("moe_combine_seq", &[], [(t * n).div_ceil(256), 1, 1], &[&x, &ye, &wts, &inv, &sh, &sg, &(k as u32), &(n as u32), &(t as u32)])
    }
    fn norm_gate(&self, o: u64, w: u64, z: u64, n: usize, heads: usize, eps: f32) -> anyhow::Result<()> {
        self.run("norm_gate", &[], [heads, 1, 1], &[&o, &w, &z, &(n as u32), &eps, &0u64, &0u32])
    }
    fn norm_gate_q(&self, o: u64, w: u64, z: u64, n: usize, heads: usize, eps: f32, xq: u64) -> anyhow::Result<u64> {
        // llama numerics take one scale per 256 values, more than a head.
        if self.0.llama != 0 || n % 32 != 0 { self.norm_gate(o, w, z, n, heads, eps)?; return Ok(0); }
        self.run("norm_gate", &[], [heads, 1, 1], &[&o, &w, &z, &(n as u32), &eps, &xq, &0u32])?;
        Ok(xq)
    }
    fn norm_gate_sig(&self, o: u64, w: u64, z: u64, n: usize, heads: usize, eps: f32, xq: u64) -> anyhow::Result<u64> {
        let xq = if self.0.llama != 0 || n % 32 != 0 { 0 } else { xq };
        self.run("norm_gate", &[], [heads, 1, 1], &[&o, &w, &z, &(n as u32), &eps, &xq, &1u32])?;
        Ok(xq)
    }
    fn hc_silu(&self, x: u64, sc: f32, n: usize) -> anyhow::Result<()> { self.run("hc_silu", &[], [n.div_ceil(256), 1, 1], &[&x, &sc, &(n as u32)]) }
    fn hc_mix(&self, xn: u64, g: u64, out: u64, n: usize, hc: usize, nt: usize) -> anyhow::Result<()> {
        self.run("hc_mix", &[], [n.div_ceil(256), nt, 1], &[&xn, &g, &out, &(n as u32), &(hc as u32)])
    }
    fn hc_write(&self, r: u64, out: u64, inj: u64, n: usize, hc: usize, nt: usize, clear: bool) -> anyhow::Result<()> {
        self.run("hc_write", &[], [n.div_ceil(256), nt, 1], &[&r, &out, &inj, &(n as u32), &(hc as u32), &(clear as u32)])
    }
    fn ple_gate(&self, key: u64, query: u64, value: u64, gated: u64, n: usize, hc: usize, nt: usize) -> anyhow::Result<()> {
        self.run("ple_gate", &[], [hc, nt, 1], &[&key, &query, &value, &gated, &(n as u32)])
    }
    fn ple_conv(&self, r: u64, gated: u64, norm: u64, kw: u64, ring: u64, step: u64, d: usize, nt: usize) -> anyhow::Result<()> {
        self.run("ple_conv", &[], [d.div_ceil(256), 1, 1], &[&r, &gated, &norm, &kw, &ring, &step, &(d as u32), &(nt as u32)])
    }
    fn kv_store(&self, k: u64, v: u64, kc: u64, vc: u64, n: usize, t: usize, step: u64, hd: usize) -> anyhow::Result<()> {
        // q8/q4: each subgroup quantizes one 32-value block of one row.
        ensure!((self.0.kv_k_mode == 0 && self.0.kv_v_mode == 0) || (hd % 32 == 0 && n % hd == 0), "quantized KV cache: head_dim {hd} is not a multiple of 32");
        let turbo = self.0.kv_k_mode >= 4 || self.0.kv_v_mode >= 4;
        ensure!(!turbo || hd % 128 == 0, "turbo KV cache: head_dim {hd} is not a multiple of 128");
        self.run("kv_store", &[], [(n * t).div_ceil(256), 1, 1], &[&k, &v, &kc, &vc, &step, &(n as u32), &((n * t) as u32), &(hd as u32)])
    }
    fn attn_decode_gq(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize, scale: f32, gq_min: i32, nt: usize, sink: u64, win: usize) -> anyhow::Result<()> {
        self.attn(q, kc, vc, qg, out, part, step, nh, nkv, hd, scale, gq_min, nt, 0, 0, 0, sink, win, 0, ATT_SPLITS)
    }
    fn attn_decode_q(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize, scale: f32, gq_min: i32, nt: usize, xq: u64, sel: u64, sst: usize, sink: u64, win: usize) -> anyhow::Result<u64> {
        let xq = if self.0.llama != 0 { 0 } else { xq };
        self.attn(q, kc, vc, qg, out, part, step, nh, nkv, hd, scale, gq_min, nt, xq, sel, sst, sink, win, 0, ATT_SPLITS)?;
        Ok(xq)
    }
    fn qsa_pool(&self, ik: u64, ring: u64, pool: u64, w: u64, step: u64, t: usize, n_rot: usize, base: f32, eps: f32) -> anyhow::Result<()> {
        let ts = (base as f64).powf(-2.0 / n_rot as f64);
        self.run("qsa_pool", &[], [t / 4 + 1, 1, 1], &[&ik, &ring, &pool, &w, &step, &ts, &(t as u32), &(n_rot as u32), &eps])
    }
    fn qsa_select(&self, q: u64, pool: u64, score: u64, sel: u64, mask: u64, step: u64, r0: usize, rows: usize, nbs: usize, top: usize, sst: usize, mw: usize) -> anyhow::Result<()> {
        self.run("qsa_score", &[], [64, rows.div_ceil(8), 1], &[&q, &pool, &score, &step, &(r0 as i32), &(rows as u32), &(nbs as u32), &(top as u32)])?;
        self.run("qsa_select", &[], [rows, 1, 1], &[&score, &sel, &mask, &step, &(r0 as i32), &(nbs as u32), &(top as u32), &(sst as u32), &(mw as u32)])
    }
    fn router_topk(&self, p: u64, n: usize, k: usize, extra_scale: f32, ids: u64, wts: u64, t: usize, sigmoid: bool) -> anyhow::Result<()> {
        ensure!(n <= 512 && k <= 32, "router_topk: {n} experts, top {k}");
        self.run("router_topk", &[], [t, 1, 1], &[&p, &ids, &wts, &(n as u32), &(k as u32), &extra_scale, &(sigmoid as u32)])
    }
    fn moe_route(&self, p: u64, n: usize, k: usize, extra_scale: f32, table: u64, ids: u64, wts: u64, slots: u64, wh: u64, h: u64, mb: u64, nh: usize, h_off: usize, step: u64, t: usize, pid: u64, sigmoid: bool) -> anyhow::Result<()> {
        debug_assert_eq!(pid, 0, "the Vulkan backend has no next-layer prefetch");
        ensure!(n <= 512 && k <= 32, "moe_route: {n} experts, top {k}");
        self.run("moe_route", &[], [1, 1, 1], &[&p, &table, &ids, &wts, &slots, &wh, &h, &mb, &step, &(n as u32), &(k as u32), &extra_scale, &(nh as u32), &(h_off as u32), &(t as u32), &(sigmoid as u32)])
    }
    fn moe_prefetch_supported(&self) -> bool { !self.0.gfx11 && !self.0.igpu }
    fn moe_combine(&self, x: u64, ye: u64, w: u64, k: usize, sh: u64, sg: u64, mb: u64, y_off: usize, step: u64, slots: u64, n: usize, t: usize) -> anyhow::Result<()> {
        let err = self.0.st().err.3;
        self.run("moe_combine", &[], [n.div_ceil(256), t, 1], &[&x, &ye, &w, &sh, &sg, &mb, &step, &slots, &err, &(k as u32), &(y_off as u32), &(n as u32)])?;
        if mb != 0 { self.0.skip_rest(&mut self.0.st())?; }
        Ok(())
    }
    fn moe_down_combine(&self, ty: GgmlType, rows: usize, cols: usize, w: u64, stride: usize, slots: u64, k: usize, h: u64, ye: u64, xq: u64,
                        x: u64, wh: u64, sh: u64, sg: u64, mb: u64, y_off: usize, step: u64, t: usize) -> anyhow::Result<()> {
        if t > 1 || xq == 0 || cols % 256 != 0 || k > 32 {  // t > 1 deduplicates shared experts across tokens in moe_gemv
            self.moe_gemv_nm(ty, rows, cols, w, stride, slots, t * k, h, cols, 1, ye, None, xq, t)?;
            return self.moe_combine(x, ye, wh, k, sh, sg, mb, y_off, step, slots, rows, t);
        }
        // Rows per workgroup as moe_gemv's rows per subgroup: each subgroup takes a slot.
        let original = (2048 / cols).clamp(1, 8);
        let (rpw, err) = (self.r11_rpw(ty, "STRATA_PS5_R11_DOWN_RPW", original)?, self.0.st().err.3);
        if rpw != original { self.0.r6_mark(1 << 10, "r11_down_rpw"); }
        self.run("moe_down_combine", &self.wspec(ty), [rows.div_ceil(rpw), t, 1], &[&w, &slots, &xq, &(stride as u64), &x, &wh, &sh, &sg, &mb, &step,
            &err, &(rows as u32), &((cols / 256) as u32), &((t * k) as u32), &(k as u32), &(y_off as u32), &(rpw as u32)])?;
        if mb != 0 { self.0.skip_rest(&mut self.0.st())?; }
        Ok(())
    }
}

impl Vk {
    fn moe_gemv_nm(&self, ty: GgmlType, rows: usize, cols: usize, w: u64, stride: usize, ids: u64, k: usize, x: u64, x_stride: usize, xg: usize, y: u64, pair: Option<(i64, i64)>, xq: u64, max_dup: usize) -> anyhow::Result<()> {
        ensure!(xg > 0 && k % xg == 0, "moe_gemv: {k} slots in groups of {xg}");
        let (nc, ws, ntok) = (gemv_cols(self.0.llama, ty, cols)?, self.wspec(ty), k / xg);
        let (w2, y2) = pair.unwrap_or((0, 0));
        if ty.float_x() {
            let z = if pair.is_some() { 2 } else { 1 };
            return self.run("moe_gemv_f", &ws, [rows, k, z], &[&w, &ids, &x, &y, &(stride as u64), &(w2 as u64), &(y2 as u64),
                &(x_stride as u32), &(rows as u32), &nc, &(xg as u32)]);
        }
        let xq = if xq != 0 { xq } else {
            ensure!(ntok * cols * 9 / 8 <= XQ, "moe_gemv: {ntok} x {cols} inputs do not fit the quantized-x buffer");
            self.quant_xq(x, x_stride, cols, ntok, self.1.addr())?;
            self.1.addr()
        };
        // Without rebuilt shaders the checked-in SPIR-V still contains the old shared
        // flag race and truncated dedup list. Keep its correct single-slot path.
        let max_nm = if cfg!(feature = "ps5-r2") {
            match max_dup.min(17) {
                0 | 1 => 1u32,
                2..=4 => 4,
                5..=8 => 8,
                _ => 17,
            }
        } else { 1 };
        let spec = [&ws[..], &[max_nm]].concat();
        let original = if max_nm > 1 { 1 } else { (4096 / cols).max(1) };
        let rpw = if max_nm == 1 && ntok == 1 { self.r11_rpw(ty, "STRATA_PS5_R11_GEMV_RPW", original)? } else { original };
        if rpw != original { self.0.r6_mark(1 << 11, "r11_gemv_rpw"); }
        let z = if pair.is_some() { 2 } else { 1 };
        let (kernel, wpb) = self.0.r2.gemv(if max_nm == 1 && ntok == 1 { 1 } else { 2 }, true);
        self.run(kernel, &spec, [rows.div_ceil(wpb * rpw), k, z], &[&w, &ids, &xq, &y, &(stride as u64), &(w2 as u64), &(y2 as u64),
            &(rows as u32), &nc, &(xg as u32), &(ntok as u32), &(rpw as u32)])
    }
}
