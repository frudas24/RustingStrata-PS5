# RustingStrata

[![Reviewed by Aletheore](https://img.shields.io/badge/Reviewed%20by-Aletheore-16140F)](https://aletheore.com)

A pure-Rust inference engine for GGUF models, built for MoE models that do not fit in VRAM. The CUDA kernels are
compiled at run time with NVRTC, and the same kernels run on Vulkan: no C++ build step, no ggml. Experts that do not
fit in VRAM run on the CPU at the same time as the GPU work, and the expert placement adapts to the routing it sees.

Supported models: Qwen3, Qwen3-Next, Qwen3.5/3.6, Qwen3.8-Flash-Next, Recursive Flint, Gemma 4, DeepSeek-V4.1-Flash,
GLM-5.3-Flash, Hy4-flash, Inkling 2 Mini and Kolibri 1.1. The architectures, the GGUF types and the notes for each model are in
[docs/MODELS.md](docs/MODELS.md).

Version 1.3.7.

## This fork is meant for the PS5

This is a fork of [RustingStrata](https://github.com/RustingStudio/RustingStrata) with one purpose: running the engine
on a **PlayStation 5 under Linux** — Zen 2 x86-64 with AVX2/FMA/F16C, AMD GFX1013 through the RADV Vulkan driver, and
one pool of shared GDDR6 instead of separate RAM and VRAM. Nothing here targets a desktop. The upstream engine, its
docs and its numbers below are unchanged and still describe upstream behaviour; this fork adds an opt-in PS5 path on
top of them.

What the fork adds (all of it in `main`):

- **`STRATA_PS5` Vulkan profile** (`src/vulkan/mod.rs`): the device is treated as sharing memory, with separate
  local-device and GTT (host-visible) heap budgets, a per-allocation budget check, and a reserve borrowed from Linux
  `MemAvailable` (`src/memory.rs`), so expert pools are sized from what the console actually has instead of from a
  desktop's assumption.
- **Direct uploads** (`src/vulkan/mod.rs`, `src/backend.rs`): fresh uploads above 64 KiB go straight into
  `HOST_VISIBLE` + `HOST_COHERENT` device-local memory with no staging copy and no pre-clear, and F16 weights stream
  to F32 through a bounded 4 MiB scratch buffer (`upload_bytes`) instead of a whole-tensor F32 host allocation.
- **CPU side** (`src/cpu_moe.rs`): Q8_K block sums fused into the existing AVX2 packing, and per-batch deduplication
  of activation quantization. These apply to every build, not only to the PS5 one.
- **Tooling** (`tools/ps5/`): `build.sh` (explicit `znver2`, fat LTO, target feature check), a `run.sh` preset,
  `diagnose.sh`, `test.sh`, `bench.sh` (A/B over the memory paths) and `summarize.py`. [docs/PS5.md](docs/PS5.md)
  documents every change, the assumptions about the console, the rollback and the limits.

Measured on one console (PS5 under Linux 7.3.3, `amdgpu.gttsize=12888`, Qwen3.6-35B-A3B UD-Q2_K_XL, Vulkan,
`--ctx 8192 --chunk 512 --kv q8 --threads 6 --mtp 0`, `STRATA_FIXED_EXPERTS=1`, 3 repetitions, one session, serial):

| run | pp512 prefill / decode tok/s | pp4096 prefill / decode | experts resident |
|---|---:|---:|---:|
| this fork, PS5 profile on (`STRATA_PS5=1`) | **294.5 / 86.8** | **347.0 / 83.5** | **9703/10240** (94.2% hit) |
| same binary, profile off (`STRATA_PS5=0`) | 144.9 / 33.2 | 300.1 / 41.5 | 5344/10240 (51.4%) |
| 1.3.7 + prefill-protect only (`b7b832b`), for reference | 185.7 / 41.1 | 278.0 / 42.0 | — |

That run reported `PS5 fresh weights: mapped 1 401 266 176 bytes, staged 1 034 752 bytes` (0.07% still through the
staging path), and `worker busy` fell from 13 706 to 1 833 µs/token. It is one session in one order with 3 samples per
cell, so order and thermals can bias small deltas; the method and its limits are in [docs/PS5.md](docs/PS5.md).

The first two rows are the PS5 port alone and the third is the prefill-protect build from before the merge; `main`
now carries both. The merged tree compiles with `tools/ps5/build.sh` on the console, but the combination has not been
re-measured yet.

Build and run on the console (no sudo):

```sh
bash tools/ps5/diagnose.sh
bash tools/ps5/build.sh
bash tools/ps5/test.sh --gpu
bash tools/ps5/run.sh chat --model /path/to/model.gguf
```

Rollback is one variable: `STRATA_PS5=0` disables the Vulkan profile (the CPU and F16 changes stay), while
`STRATA_PS5_GTT=0` and `STRATA_PS5_DIRECT=0` switch off the GTT budget and the direct uploads on their own.

Branches of this fork:

- `main` — upstream 1.3.7, this PS5 port, the `STRATA_PREFILL_PROTECT` change described below, and this README.
  Experimental: what is still unmeasured is listed in [docs/PS5.md](docs/PS5.md).
- `feat/expert-cache-prefill-protect` — the prefill-protect change alone (`b7b832b`, on upstream `8df1c22`), kept
  clean as the upstream PR branch: `STRATA_PREFILL_PROTECT` (default on) stops a long prompt from flushing the hot
  decode cache, overlaps the next layer's cold NVMe read-ahead and releases cold prefill experts. It is merged into
  `main` here, and the branch must stay free of the PS5 port or the PR would carry it.

### PS5 versions R1–R6

`main` carries the whole PS5 line, one commit and one tag per version, so every checkpoint stays reconstructible while
the tip is the version to run:

| version | tag | what it adds | opt-in switches |
|---|---|---|---|
| R1 | `ps5-r1` (`71d4a92`, tip before R2) | the base port: opt-in Vulkan PS5 profile with separate local/GTT budgets and a `MemAvailable` reserve, direct `HOST_VISIBLE`/`HOST_COHERENT` uploads, F16→F32 streaming through 4 MiB scratch, AVX2 Q8_K fusion, `tools/ps5/` | `STRATA_PS5`, `STRATA_PS5_GTT`, `STRATA_PS5_DIRECT` |
| R2 | `ps5-r2` | wave32 quantized kernels: optional 64/128-thread `gemv_q` and `moe_gemv_q` workgroups, 32×64 `gemmi_dp_ps5_32` small INT8 GEMM tile; Cargo feature `ps5-r2` builds five extra shader variants with `glslc` and validates them with `spirv-val` | `STRATA_PS5_GEMV_WG`, `STRATA_PS5_SMALL`, `STRATA_PS5_SMALL_GEMM`, `PS5_R2_BUILD` |
| R3 | `ps5-r3` | experimental immutable GGUF host-pointer import over `VK_EXT_external_memory_host`, with `weights::cold` skipping imported ranges | `STRATA_PS5_ZERO_COPY`, `STRATA_PS5_ZERO_COPY_MB` (512) |
| R4 | `ps5-r4` | adaptive heterogeneous MoE prefill: each layer's missed experts split between GPU, half and CPU arms by a bounded EWMA policy instead of staging every miss | `STRATA_PS5_SCHED`, `_SCHED_LOG`, `_SCHED_MAX_ROWS`, `_SCHED_SCRATCH_MB` |
| R5 | `ps5-r5` | tiled long-context attention prefill (`attn_prefill_ps5.comp`, 1/2/4 KiB K/V tiles, sliding-window start bound) and paired KV loads | `STRATA_PS5_ATTN_WG`, `STRATA_R5_WG` |
| R6 | `ps5-r6` (tip) | bounded autotuner (`autotune-r6.py`), dispatch telemetry and the PGO/probe toolchain | `STRATA_PS5_R6` |

**Yes, the last one is enough.** The line is cumulative: R6 contains R1–R5, and `main` at the tip is R6 plus the
`STRATA_PREFILL_PROTECT` change. The earlier versions are kept as commits and tags because each one is then
reconstructible without being re-derived (`git switch --detach ps5-r3`, rebuild, measure). No version removes a file
of the one before it, and none of them touches `src/expert_cache.rs`; R4–R6 modify `src/gpu_model/prefill.rs`, so the
two `STRATA_PREFILL_PROTECT` hunks were reapplied on top of their version of that file.

Per-version documentation: [docs/PS5.md](docs/PS5.md) (R1), [docs/PS5-R2.md](docs/PS5-R2.md),
[docs/PS5-R3.md](docs/PS5-R3.md), [docs/PS5-R4.md](docs/PS5-R4.md), [docs/PS5-R5.md](docs/PS5-R5.md) and
[docs/PS5-R6.md](docs/PS5-R6.md), each with its contracts, its flags and its own limits.

What is still unmeasured: **R2–R6 were never benchmarked on this console.** The numbers above belong to the R1 port
(rows 1 and 2) and to the pre-merge R1 + prefill-protect build (row 3); no tok/s has been taken from R2, R3, R4, R5 or
R6. Building a tree is not a measurement, and the R4, R5 and R6 documents state themselves that their authors had no
console, no toolchain and no weights when they were written. Treat R2–R6 as candidates until a benchmark says
otherwise; `STRATA_PS5=0`, each version's default-off switches and the tags are the rollback.

The licence and the credit are unchanged: original RustingStrata by Vasyl Trefilov, distributed under the same
[LICENSE.md](LICENSE.md), no endorsement implied. Upstream's own README follows.

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
- [PS5.md](docs/PS5.md): this fork's PS5 port, what changed, the build, the rollback and the benchmark method.
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
