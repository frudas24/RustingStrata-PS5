# RustingStrata — PS5 fork

**Experimental fork of [RustingStrata](https://github.com/RustingStudio/RustingStrata) for the PlayStation 5 under
Linux.** Original author Vasyl Trefilov, same [LICENSE.md](LICENSE.md), no endorsement implied. The upstream text,
docs and numbers further down are unchanged and still describe upstream; this fork keeps an opt-in PS5 path on top of
them.

Current tip: **R9** — R6U1 (R1–R6 merged with upstream 1.4.0) plus five later deltas, all of them research or
diagnostic work whose flags are off by default: R7 (two shader fixes ported from a physical R6 campaign, opt-in Vulkan
counters, experimental dense64 GEMM), R7-V1 and R7-V2 ("Venus": opt-in CPU F16/BF16 kernels and a one-row GPU MoE
trace), R8 (routed-arithmetic contract oracle) and R9 (matched single-expert causal replay). **None of them is a
measured speedup and none changes a default**; the only tok/s this fork owns are still the R1 rows below. Upstream has
not moved past 1.4.0 (`260a11c9`), so nothing of upstream is missing here. Details:
[docs/PS5-R7.md](docs/PS5-R7.md), [docs/PS5-R7-VENUS.md](docs/PS5-R7-VENUS.md), [docs/PS5-R7-V2.md](docs/PS5-R7-V2.md),
[docs/PS5-R8.md](docs/PS5-R8.md), [docs/PS5-R9.md](docs/PS5-R9.md).

## This fork is meant for the PS5

One purpose: running the engine on a console under Linux — Zen 2 x86-64 with AVX2/FMA/F16C, AMD GFX1013 through the
RADV Vulkan driver, and **one pool of shared GDDR6** instead of separate RAM and VRAM. Nothing here targets a desktop:
the desktop paths are upstream's, and the PS5 behaviour is opt-in (`STRATA_PS5`), so a build without it behaves as
upstream.

What the fork adds, all of it in `main`:

- **`STRATA_PS5` Vulkan profile** (`src/vulkan/mod.rs`): the device is treated as sharing memory, with separate
  local-device and GTT (host-visible) heap budgets, a per-allocation budget check, and a reserve borrowed from Linux
  `MemAvailable` (`src/memory.rs`, `STRATA_PS5_HOST_RESERVE_MB`), so expert pools are sized from what the console
  actually has instead of from a desktop's assumptions.
- **Direct uploads** (`src/vulkan/mod.rs`, `src/backend.rs`): fresh uploads above 64 KiB go straight into
  `HOST_VISIBLE` + `HOST_COHERENT` device-local memory, without a staging copy and without a pre-clear, and F16 weights
  stream to F32 through a bounded 4 MiB scratch buffer (`upload_bytes`) instead of a whole-tensor F32 host allocation.
- **CPU side** (`src/cpu_moe.rs`): Q8_K block sums fused into the existing AVX2 packing, and per-batch deduplication of
  activation quantization; upstream's own CPU changes from 1.4.0 are kept next to them. These apply to every build,
  not only to the PS5 one.
- **PS5 versions R2–R6**, each one behind its own default-off switch and with its own document: wave32 quantized
  kernels (R2), immutable GGUF host-pointer import (R3), adaptive heterogeneous MoE prefill (R4), tiled long-context
  attention prefill with independent K/V pair loads (R5), and the bounded autotuner, dispatch telemetry and PGO
  toolchain (R6). Table below.
- **Five later deltas, R7 to R9**, also default-off: the two shader fixes ported from a physical R6 campaign plus
  optional Vulkan counters and an experimental dense64 GEMM (R7); opt-in F16C and BF16 CPU kernels with a one-row GPU
  MoE trace (R7-V1, R7-V2); a deliberately slow routed-arithmetic oracle that tests what Vulkan actually computes (R8);
  and a matched single-expert causal replay (R9). None of them promotes anything: R4's hybrid scheduler and the
  file-backed import stay quarantined behind an explicit research switch, and every number they report comes from their
  authors' environment, not from this console.
- **Tooling** (`tools/ps5/`): `build.sh` (explicit `znver2`, fat LTO, one codegen unit, target-feature check),
  `env.sh`, the `run.sh` preset, `diagnose.sh`, `test.sh`, `test-merge.sh`, `test-r2.sh`–`test-r7.sh`, `bench.sh`,
  `bench-r2.sh`–`bench-r5.sh`, `check-r2.py`–`check-r6.py`, `summarize*.py`, `kv-budget.py`, `parity-r4.py`,
  `pgo-r6.sh`, `autotune-r6.py` and `probe-r6.sh`, plus the later gates: `test-r7-gemm.sh`, `check-r7-gemm.py` and
  `bench-r7-gemm.py` (dense64 fingerprint and A/B); `test-venus.sh` with `check-venus-cpu.py`,
  `check-venus-trace.py`, `venus-f16-check.c`, `venus-real-model.py` and `venus-real-weights.c` (F16/BF16 parity);
  `test-r8-contract.sh` with `r8-run-guarded.py`, `check-r8.py` and `compare-r8.py` (four-arm contract campaign); and
  `test-r9.sh` with `check-r9.py` and `compare-r9.py` (single-expert replay). None of it modifies the console or the
  system.

### Measured on one console (R1, historical)

PS5 under Linux 7.3.3 with `amdgpu.gttsize=12888` (13.09 GiB combined GTT+VRAM ceiling), `Qwen3.6-35B-A3B-UD-Q2_K_XL`
Vulkan, `--ctx 8192 --chunk 512 --kv q8 --threads 6 --vram-margin 2048 --ram-margin 1536 --cache-ram 256 --mtp 0`,
`STRATA_FIXED_EXPERTS=1`, 3 repetitions, one session, serial:

| run | pp512 prefill / decode tok/s | pp4096 prefill / decode | experts resident |
|---|---:|---:|---:|
| this fork, PS5 profile on (`STRATA_PS5=1`) | **294.5 / 86.8** | **347.0 / 83.5** | **9703/10240** (94.2% hit) |
| same binary, profile off (`STRATA_PS5=0`) | 144.9 / 33.2 | 300.1 / 41.5 | 5344/10240 (51.4%) |
| 1.3.7 + prefill-protect only (`b7b832b`), for reference | 185.7 / 41.1 | 278.0 / 42.0 | — |

That run reported `PS5 fresh weights: mapped 1 401 266 176 bytes, staged 1 034 752 bytes` (0.07% still through the
staging path) and `worker busy` falling from 13 706 to 1 833 µs/token. It is one session in one order with 3 samples
per cell, so order and thermals can bias small deltas; the method and its limits are in [docs/PS5.md](docs/PS5.md).

The model behind every number here is the GGUF `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` (~11.7 GiB) with the optional MTP
head `mtp-Qwen3.6-35B-A3B-Q8_0.gguf`. A BF16 (~70 GB) or FP8 (~35 GB) checkpoint neither fits the 13.09 GiB ceiling
nor is a format this engine loads, which reads GGUF.

**Those three rows belong to R1 and to the pre-merge prefill-protect build.** Later evidence lives in the documents,
not in this table: [docs/PS5-R7.md](docs/PS5-R7.md) summarizes the supplied physical R6 campaign (one exploratory run per
depth, 65024 prompt tokens at 139.5 prefill / 53.0 decode) and [docs/PS5-R8.md](docs/PS5-R8.md) a single-run R7
comparison whose own text says the 8K decode delta comes from a slow baseline arm and is not a causal gain. No R6U1,
R7, R8 or R9 benchmark is claimed here. Building a tree is not a measurement, and the per-version documents state
themselves that their authors had no console, no toolchain and no weights when they were written. Treat R2–R9 as
candidates until a benchmark says otherwise.

### Build and run on the console

No sudo, and nothing installed outside the source tree:

```sh
bash tools/ps5/diagnose.sh        # what the console gives the engine (GTT, VRAM, kernel, weights)
bash tools/ps5/build.sh           # needs cargo/rustc; glslc + spirv-val for the R2 shader variants
bash tools/ps5/test-merge.sh      # compile every test target, CPU-only gate
bash tools/ps5/test.sh --gpu      # Vulkan tests on the console
bash tools/ps5/test-r2.sh         # per-version gates; R5 also runs symmetric and mixed K/V pairs
bash tools/ps5/test-r5.sh
bash tools/ps5/test-r7.sh         # R7: empty attention and MoE mailbox gates
bash tools/ps5/test-r7-gemm.sh    # R7: dense128/dense64 fingerprint comparison
bash tools/ps5/test-venus.sh --cpu   # R7-V1/V2 CPU kernels; no GPU and no shader compiler needed
bash tools/ps5/run.sh chat  --model "$MODEL" --ctx 8192 --k q8 --v q4   # mixed KV (R6U1)
bash tools/ps5/run.sh serve --model "$MODEL" --ctx 65536 --kv q8
```

`build.sh` sources `env.sh`, so the build always gets `-C target-cpu=znver2`, fat LTO, one codegen unit and a
`PS5_TARGET` of `x86_64-unknown-linux-gnu`; it verifies that avx2/fma/f16c really are in the target before compiling.
The `ps5-r2` Cargo feature (five extra shader variants built with `glslc` and checked with `spirv-val`) is on by
default in that script; `PS5_R2_BUILD=0` falls back to the R1 shaders on a console without the shader tools.
`PS5_R5_ALL_PAIRS=1` widens the R5 test from 8 to 16 K/V pairs. R7's repaired shaders need a recent shader
toolchain: the campaign used Shaderc 2026.4 and SPIRV-Tools 2025.1, and glslc 2023.8 fails the packed-dot compile.
The R8 and R9 research campaigns are run separately, with the MoE model and a directory that must not exist yet:
`bash tools/ps5/test-r8-contract.sh "$MODEL_MOE" results/r8-contract` and
`bash tools/ps5/test-r9.sh "$MODEL_MOE" results/r9-layer0`.

### Rollback

One variable at a time, no rebuild:

```sh
STRATA_PS5=0                     # the whole Vulkan profile; the CPU and F16 changes stay
STRATA_PS5_GTT=0                 # only the GTT budget split
STRATA_PS5_DIRECT=0              # only the direct upload path
STRATA_PREFILL_PROTECT=1|strict  # opt back into the fork's old prompt handling (now OFF by default, see R6U1)
```

Every R2–R9 switch is off by default — the later ones are read only as the literal `1`, except `STRATA_PS5_R9_MODE`,
which takes `shadow`, `legacy` or `contract` and is off when empty — so a plain build is R1 behaviour plus upstream.
`STRATA_SCALAR` overrides the new F16/BF16 CPU kernels, and `STRATA_PS5_R4_RESEARCH=1` is still required to touch the
quarantined R4 hybrid scheduler and the file-backed import. The tags below rebuild any earlier point
(`git switch --detach ps5-r3`, `bash tools/ps5/build.sh`, measure).

### Branches and tags

| ref | what it is |
|---|---|
| `main` | the tip: R9 = R1–R6 + upstream 1.4.0 (R6U1) + the R7, R7-V1, R7-V2, R8 and R9 deltas + this README |
| `ps5-r1` … `ps5-r6` | one tag per PS5 version; each checkpoint stays reconstructible without being re-derived |
| `ps5-r6u1` | the R6U1 merge |
| `ps5-r7`, `ps5-r7-venus`, `ps5-r7-venus-v2`, `ps5-r8`, `ps5-r9` | one tag per later delta, each with its own branch (`ps5/r7` … `ps5/r9`) and its own document; they arrived as git bundles and were merged fast-forward, so their commit hashes are the authors' |
| `feat/expert-cache-prefill-protect` | `b7b832b`, the prefill-protect change alone (the upstream PR branch, pushed as `upstream/pr-108`), kept free of the PS5 port so the PR does not carry it |
| `upstream/main` | `260a11c9`, upstream 1.4.0 — the merge base R6U1 was built on; R1–R6 were cut from `8df1c22` (1.3.7) |

### PS5 versions R1–R9

| version | tag | what it adds | opt-in switches |
|---|---|---|---|
| R1 | `ps5-r1` | the base port: opt-in Vulkan PS5 profile with separate local/GTT budgets and a `MemAvailable` reserve, direct `HOST_VISIBLE`/`HOST_COHERENT` uploads, F16→F32 streaming through 4 MiB scratch, AVX2 Q8_K fusion, `tools/ps5/` | `STRATA_PS5`, `STRATA_PS5_GTT`, `STRATA_PS5_DIRECT`, `STRATA_PS5_HOST_RESERVE_MB` |
| R2 | `ps5-r2` | wave32 quantized kernels: optional 64/128-thread `gemv_q` and `moe_gemv_q` workgroups, 32×64 `gemmi_dp_ps5_32` small INT8 GEMM tile; Cargo feature `ps5-r2` builds five extra shader variants with `glslc` and validates them with `spirv-val` | `STRATA_PS5_GEMV_WG`, `STRATA_PS5_SMALL`, `STRATA_PS5_SMALL_GEMM`, `PS5_R2_BUILD` |
| R3 | `ps5-r3` | experimental immutable GGUF host-pointer import over `VK_EXT_external_memory_host`, with `weights::cold` skipping imported ranges | `STRATA_PS5_ZERO_COPY`, `STRATA_PS5_ZERO_COPY_MB` (512) |
| R4 | `ps5-r4` | adaptive heterogeneous MoE prefill: each layer's missed experts split between GPU, half and CPU arms by a bounded EWMA policy instead of staging every miss | `STRATA_PS5_SCHED`, `_SCHED_LOG`, `_SCHED_MAX_ROWS`, `_SCHED_SCRATCH_MB` |
| R5 | `ps5-r5` | tiled long-context attention prefill (`attn_prefill_ps5.comp`, 1/2/4 KiB K/V tiles, sliding-window start bound) and paired KV loads | `STRATA_PS5_ATTN_WG`, `STRATA_R5_WG` |
| R6 | `ps5-r6` | bounded autotuner (`autotune-r6.py`), dispatch telemetry and the PGO/probe toolchain | `STRATA_PS5_R6` |
| R6U1 | `ps5-r6u1` | 1.4.0 merged onto R6: upstream's independent K/V formats, split-K GEMM, dynamic KV growth, new architectures; PS5 path kept and gated | same as R1–R6; `STRATA_PREFILL_PROTECT` now off |
| R7 | `ps5-r7` | evidence-driven delta from a physical R6 campaign: merge guard for empty attention and an atomic OR on the MoE miss flag in the mixed-KV shaders, R4 hybrid and the file import forced behind an explicit research switch, optional Vulkan counters and token timing, experimental dense64 GEMM (dense128 stays the default) | `STRATA_PS5_R4_RESEARCH`, `STRATA_PS5_R7_GEMM_TILE`, `STRATA_R7_VK_TRACE`, `STRATA_VK_PROF`, `STRATA_R7_TOKEN_TIMING` |
| R7-V1 | `ps5-r7-venus` | "Venus" CPU kernels: F16C+FMA 3x4 GEMM and single-token matvec, plus a diagnostic trace of exactly one real GPU-routed MoE row with a CPU cross-feed | `STRATA_PS5_VENUS_F16`, `STRATA_PS5_VENUS_MATVEC`, `STRATA_PS5_VENUS_LOG`, `STRATA_PS5_VENUS_TRACE=layer:position:rank`, `STRATA_PS5_VENUS_TRACE_VALUES` |
| R7-V2 | `ps5-r7-venus-v2` | native BF16 CPU GEMV for exact 2-byte rows, and an independent BF16/F16 parity harness over real Qwen3-0.6B weights; needs a GGUF converted with `--outtype bf16` | `STRATA_PS5_VENUS_BF16` |
| R8 | `ps5-r8` | routed-arithmetic contract investigation: a deliberately slow CPU oracle that follows Vulkan's own type/column selection, route-aware gate/up/down probes, and a guarded four-arm campaign; an accuracy experiment, never a production path | `STRATA_PS5_R8_CONTRACT`, `STRATA_R7_R4_STAGE_PROBE`, `PS5_R8_TIMEOUT` |
| R9 | `ps5-r9` | matched single-expert causal replay: replaces exactly one GPU-routed expert row just before the Vulkan combine, reads the combined row back, and fails closed on a missing target or a mismatch | `STRATA_PS5_R9_MODE` (`shadow`, `legacy`, `contract`), `PS5_R9_LAYER`, `PS5_R9_POS`, `PS5_R9_TOKEN`, `PS5_R9_RANK` |

**`main` is the version to run**, with two nuances. The code line is cumulative: R6 contains R1–R5, R6U1 adds upstream
1.4.0, and R7–R9 keep every earlier file and every earlier switch, so moving forward drops nothing — across R1→R6,
R6→R6U1 and R6U1→R9 the count of deleted files is zero. But cumulative in code is not cumulative in evidence: R7–R9 are
research deltas whose flags default to off and whose own documents report no console and no toolchain, so the tags are
kept to compare each successor against its predecessor A/B/B/A and to keep every checkpoint reconstructible without
being re-derived.

Per-version documentation: [docs/PS5.md](docs/PS5.md) (R1), [docs/PS5-R2.md](docs/PS5-R2.md),
[docs/PS5-R3.md](docs/PS5-R3.md), [docs/PS5-R4.md](docs/PS5-R4.md), [docs/PS5-R5.md](docs/PS5-R5.md),
[docs/PS5-R6.md](docs/PS5-R6.md), [docs/PS5-R6U1.md](docs/PS5-R6U1.md), [docs/PS5-R7.md](docs/PS5-R7.md),
[docs/PS5-R7-VENUS.md](docs/PS5-R7-VENUS.md), [docs/PS5-R7-V2.md](docs/PS5-R7-V2.md), [docs/PS5-R8.md](docs/PS5-R8.md)
and [docs/PS5-R9.md](docs/PS5-R9.md), each with its contracts, its flags and its own limits.

### R6U1: what the upstream merge changed

R6U1 is a three-way **content** merge of upstream `260a11c9` (1.4.0) onto the fork `612c3c9`, not another round of
hardware optimisation: 171 files changed, no source file removed, and the seven conflicts (`expert_cache`,
`gpu_model`, `lib`, `main`, `vulkan/mod`, `gemmi.glsl`, `tests/backend`) resolved in source. What it changes in
behaviour:

- **`STRATA_PREFILL_PROTECT` now defaults to OFF** (upstream policy; the fork had it on). Upstream reports that with
  protection on, the reply's expert hit rate on Qwen3.6-35B-A3B fell from 83% to 76% on an 8k prompt with a 128-token
  reply. `1` and the stricter `strict` remain available, R6's autotuner pins `0`, and the next-layer read-ahead and
  cold-expert release stay on regardless. The binary measured in the R1 table above was the pre-merge, protection-on
  one: reproduce it with `STRATA_PREFILL_PROTECT=1`.
- **Independent K/V formats adopted**: `--k`/`--v` override each half of `--kv`, with Vulkan specialization IDs 15/16
  and R5's pair loads harmonised. R6's calibration profile is still symmetric and now rejects a stale binary
  identity, so old profiles must be recalibrated rather than reused.
- **Two experiments are refused on the PS5 rather than mis-run**: TurboQuant Vulkan
  (`src/vulkan/mod.rs`, "PS5: TurboQuant Vulkan is not validated") and `STRATA_KV_HOST`, which is a CUDA
  host-attention experiment. Upstream's code for both is kept byte-for-byte; keeping it is not a claim that it works
  on the console.
- **Upstream's new architectures are adopted as upstream** (Raon and the Inkling 2, Gemma, DeepSeek and GLM changes,
  plus tokenizer, grammar, vision and serving work). Carrying them is not evidence that they run on the PS5.

### Validation status

Honest summary, because the distinction matters:

- **R6U1 was not compiled, not GLSL/SPIR-V validated, not run and not benchmarked** in the environment that prepared
  it. Its own logs (`docs/validation-r6u1/`) record PASS for the independent C SIMD check, the R2–R5 Python
  geometry/model checks, three upstream KLD tests, syntax checks, the mixed-KV budget smoke and the first nine R6
  controller tests, and **NOT_RUN_TO_COMPLETION** for the tenth (child timeout/failure). The Rust build, the new
  shader build, GPU parity and any tok/s were BLOCKED there.
- **R6 (the tree before this merge) was compiled on the console** with `tools/ps5/build.sh`, shader variants included,
  in 3m11s. That is compilation only.
- **R1 was compiled and measured** on the console; those are the numbers in the table. **R2–R6 were never
  benchmarked**, and neither were R6U1 or R7–R9; none of them inherits the R1 figures as a measurement of its own.
- **R7–R9 were prepared without a toolchain and without a console**, as their own documents state: the Rust build, the
  new shader build, GPU parity and any tok/s are pending for all five deltas. What they do ship is evidence from their
  authors' environment — `docs/validation-r8/` copies the R6/R7 campaign tables (four-arm parity, dense64 retries,
  revision comparison) and `docs/validation-venus-v2/` an independent BF16/F16 run on an EPYC host with real
  Qwen3-0.6B weights (maximum causal logit drift 3.3e-5, identical top-1 and identical six greedy tokens). None of it
  was produced on this console, and R9's own document says outright that R9 has no physical parity and that R8's
  physical results are not R9 results.

What is still open is written down rather than implied: the A/B/B/A plan, the acceptance criteria, the mixed-KV
quality question and the unresolved `attn_merge` / TurboQuant / MoE shared-flag risks are in
[docs/PS5-R6U1.md](docs/PS5-R6U1.md); the physical build of R7 and R7's dense64 gate are in
[docs/PS5-R7.md](docs/PS5-R7.md); the Rust/parity gates of the Venus kernels are in
[docs/PS5-R7-VENUS.md](docs/PS5-R7-VENUS.md) and [docs/PS5-R7-V2.md](docs/PS5-R7-V2.md); the R4 hybrid failure and the
four-arm race are in [docs/PS5-R8.md](docs/PS5-R8.md), and the replay is in [docs/PS5-R9.md](docs/PS5-R9.md). Those
documents, not this README, are the acceptance criteria. The historical fork report is preserved verbatim in
[docs/archive/PS5-fork-612c3c9-README.md](docs/archive/PS5-fork-612c3c9-README.md), with its R1 measurements, and its
own report in `docs/archive/PS5-fork-612c3c9-R1-report.md`.

The licence and the credit are unchanged: original RustingStrata by Vasyl Trefilov, distributed under the same
[LICENSE.md](LICENSE.md), no endorsement implied. **Upstream's own README follows, verbatim.**

---

# RustingStrata

[![Reviewed by Aletheore](https://img.shields.io/badge/Reviewed%20by-Aletheore-16140F)](https://aletheore.com)

A pure-Rust inference engine for GGUF models, built for MoE models that do not fit in VRAM. The CUDA kernels are
compiled at run time with NVRTC, and the same kernels run on Vulkan: no C++ build step, no ggml. Experts that do not
fit in VRAM run on the CPU at the same time as the GPU work, and the expert placement adapts to the routing it sees.

Supported models: Qwen3, Qwen3-Next, Qwen3.5/3.6, Qwen3.8-Flash-Next, Recursive Flint, Gemma 4, DeepSeek-V4.1-Flash,
GLM-5.3-Flash, Hy4-flash, Inkling 2 Mini and Kolibri 1.1. The architectures, the GGUF types and the notes for each model are in
[docs/MODELS.md](docs/MODELS.md).

Version 1.4.0.

## Numbers

Qwen3.6-35B-A3B on an RTX 3060 12 GB (Ryzen 5 7600X, 32 GB RAM), CUDA, greedy, against llama.cpp tuned for the same
file. Neither file fits in VRAM.

| | RustingStrata | llama.cpp |
|---|---|---|
| UD-Q5_K_M (25.2 GiB), decode after a 4K prompt | **80.5 tok/s** | 64.4 |
| UD-Q5_K_M, prefill of a 4K prompt | **1530 tok/s** | 1118-1122 |
| UD-Q2_K_XL (11.7 GiB), decode after a 4K prompt | **113.4 tok/s** | 66.5 |
| UD-Q2_K_XL, prefill of a 4K prompt | 2216 tok/s | ~2236 |

### Many users at once

`serve --parallel N` decodes the requests together in one batched forward (CUDA). Same machine, same load script
(`tools/loadtest.py`: N streamed chats at once, 128 reply tokens each, greedy), total decode tok/s:

| Clients | Qwen3-4B, RustingStrata (Q4_K_M) | Qwen3-4B, vLLM 0.33 (bf16) | Qwen3.6-35B-A3B UD-Q5_K_M, RustingStrata |
|---:|---:|---:|---:|
| 1 | 91.3 | 36.7 | 49.0 |
| 4 | 228.1 | 139.9 | 88.9 |
| 16 | 799.0 | 520.1 | 100.7 |
| 64 | 1698 | 1688-1765 | 129.3 |

vLLM runs bf16 weights and RustingStrata Q4_K_M, so the dense columns are not like for like. The 27 GB MoE file does
not fit on the 12 GB card, so vLLM was not run on it. RustingStrata reads each expert once for all requests of a step,
runs the experts that are not in VRAM on the CPU, and keeps slot state that does not fit in VRAM in RAM. The MoE
column moves by about 30% between runs on this machine (the 27 GB file is read through a 30 GB page cache).

The method, the ranges, Vulkan, the other models and the quality checks are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Install

Windows (PowerShell; installs to `%LOCALAPPDATA%\Programs\RustingStrata`, adds it to PATH and makes a Start menu
entry that opens the chat page):

```powershell
irm https://raw.githubusercontent.com/RustingStudio/RustingStrata/main/install.ps1 | iex
```

Linux x86_64 (installs to `~/.local/bin`):

```sh
curl -fsSL https://raw.githubusercontent.com/RustingStudio/RustingStrata/main/install.sh | sh
```

Arch Linux: `yay -S rustingstrata-bin` (AUR). Or download a binary from the
[releases page](https://github.com/RustingStudio/RustingStrata/releases), or build it (Rust 2024 edition; on Windows
install the Visual Studio C++ Build Tools first):

```sh
cargo build --release    # the binary is target/release/rustingstrata
```

Then `rustingstrata serve --gpu --open` opens the chat page, which downloads models, and `rustingstrata interface`
asks a few questions in the terminal. `rustingstrata update` replaces the binary with the newest release (`--check`
only says whether one is out); `serve` prints a line at start when a newer release is out. A binary installed by a
package manager (the AUR package) is updated with that package manager.

A GPU run needs one of:

- an NVIDIA driver with the CUDA NVRTC library (loaded at run time). Volta (sm_70), Turing (sm_75) and Ampere or newer
  (sm_80+) work; CUDA 13 dropped Volta, so a V100 needs the 12.x toolkit. Pascal and older use Vulkan.
  The library is found by name (`libnvrtc.so`; on Windows `nvrtc64_130_0.dll` or `nvrtc64_120_0.dll`, on `PATH` or
  under `%CUDA_PATH%\bin`); `STRATA_NVRTC` names another file. Without it the run falls back to Vulkan.
- a Vulkan 1.3 driver with shaderFloat64, int8/16/64 and subgroup size 32 or a fixed 64 (tested on RTX 3060, Radeon VII
  and V100; Intel Arc lacks shaderFloat64).
- for `--backend hip`: a ROCm runtime with hiprtc (`libamdhip64.so`, `libhiprtc.so`), also loaded at run time; the
  kernels are compiled for the card's `gfx` target on first use. Experimental — status, numbers and limits are in
  [docs/HIP.md](docs/HIP.md).

`--gpu` picks CUDA when it loads, else Vulkan; `--backend` chooses (`cuda`, `vulkan` or `hip`). The first CUDA run compiles the kernels (~15 s) and
caches them. Every GPU tried so far is in [docs/SUPPORTED_GPUS.md](docs/SUPPORTED_GPUS.md).

## Quick start

Run `rustingstrata` with no arguments (double-click `rustingstrata.exe` on Windows). It starts a server on
http://127.0.0.1:8080 and opens a chat page that downloads models from Hugging Face and switches between the ones in
your models folder. Any app that takes an OpenAI-compatible address (Open WebUI, Jan, SillyTavern, code editors) can
use `http://127.0.0.1:8080/v1`.

`rustingstrata interface` asks for the model, the mode (terminal chat, chat page, API server, speed test), GPU or CPU,
the context length and the address, prints the command these make and runs it.

With flags:

```sh
M=path/to/model.gguf

rustingstrata calibrate --model $M --dry-run  # preview the workload matrix and knob space
rustingstrata calibrate --model $M     # once per model and machine: best chunk, MTP drafts, short threshold
rustingstrata chat --model $M --gpu    # --think for thinking mode, --temp 0.7 to sample
rustingstrata generate --model $M --gpu --prompt "Write a haiku about rivers." -n 128
rustingstrata serve --model $M --gpu --addr 127.0.0.1:8080
rustingstrata bench --model $M --gpu --pp 512,4096 --tg 128
```

Without `--gpu` the model runs on the CPU reference, which is slow and meant for checking. DeepSeek, GLM, Hy4 and
Inkling 2 open the GPU by themselves.

## Docs

- [USAGE.md](docs/USAGE.md): flags, the chat page and MCP tools, the OpenAI and Anthropic APIs, prompt caching,
  several GPUs.
- [CALIBRATION.md](docs/CALIBRATION.md): the budgeted search, task corpora, the `spec` sweep and the JSON file.
- [MODELS.md](docs/MODELS.md): supported architectures and GGUF types, notes and caveats for each model.
- [BENCHMARKS.md](docs/BENCHMARKS.md): every model against llama.cpp; older runs are in the
  [benchmark history](docs/README.md#benchmark-history).
- [HOSTING.md](docs/HOSTING.md): running a server for other people.
- [SUPPORTED_GPUS.md](docs/SUPPORTED_GPUS.md) and [AMD.md](docs/AMD.md): the GPUs tried so far and AMD notes.
- [VISION.md](docs/VISION.md): image input. [LORA.md](docs/LORA.md): LoRA adapters.
- [PS5.md](docs/PS5.md), [PS5-R2.md](docs/PS5-R2.md), [PS5-R3.md](docs/PS5-R3.md), [PS5-R4.md](docs/PS5-R4.md),
  [PS5-R5.md](docs/PS5-R5.md), [PS5-R6.md](docs/PS5-R6.md), [PS5-R6U1.md](docs/PS5-R6U1.md), [PS5-R7.md](docs/PS5-R7.md),
  [PS5-R7-VENUS.md](docs/PS5-R7-VENUS.md), [PS5-R7-V2.md](docs/PS5-R7-V2.md), [PS5-R8.md](docs/PS5-R8.md) and
  [PS5-R9.md](docs/PS5-R9.md): this fork's PS5 port, one version per document, with its contracts, its switches and its
  limits.
- [ROADMAP.md](ROADMAP.md): what is planned next, by section; ideas from users are welcome.

## Development

```sh
cargo test --release                      # GPU tests need the model: STRATA_MODEL=path/to/model.gguf
python -I tools/parity.py $M --extra "--gpu" --tag gpu   # agreement with llama.cpp (needs llama-server once)
```

CI only builds, so run the tests before a pull request.

Source map: `src/gguf.rs` (file reader), `src/quant/` (dequantization), `src/model.rs` (CPU reference forward),
`src/cuda/` and `src/vulkan/` (GPU kernels, behind `src/backend.rs`), `src/gpu_model.rs` (GPU forward),
`src/expert_cache.rs` (expert placement), `src/deepseek/`, `src/glm/`, `src/hy4/`, `src/inkling2/` and `src/gemma4/`
(the models with their own forward), `src/serve.rs` (HTTP), `src/chat.rs` (chat templates and tool calls),
`src/main.rs` (CLI).

## License

RustingStrata is released under the [RustingStrata License 1.0](LICENSE.md). You may use, modify and distribute it,
including for commercial services, as long as you keep the license and credit Vasyl Trefilov as the original author.
You may not sell RustingStrata itself.
