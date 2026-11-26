# RustingStrata — PS5 Linux experimental port, R1

Original RustingStrata 1.3.7 by Vasyl Trefilov, archive commit `8df1c22e24645d4e9ec5980548ddabe11251548e`.
This modified edition is distributed under the original LICENSE.md. Uses RustingStrata by Vasyl Trefilov. No endorsement is implied.

## Target and validation status

Base PS5 / Slim, x86-64 Zen 2 with AVX2/FMA/F16C, AMD custom GPU through a working Linux amdgpu/Mesa RADV Vulkan driver. The 16 GB GDDR6 is shared, not 16 GB RAM plus 16 GB VRAM; the Linux memory split depends on the image and boot configuration. No assumptions about accessible capacity, GPU clocks, CU unlocks, native INT8 acceleration, or cooperative matrix support. No firmware, internal SSD, boot configuration, GPU power limits, or kernel settings are modified by this package. Not a native PS5 OS executable. No model weights are bundled.

**Experimental source release; not compiled or run in Rust in the preparation environment** (no Rust toolchain, no PS5 GPU, no dependency network, no GGUF). No tokens/s measurements or claim of PS5 compatibility yet. A standalone C test of the new AVX2 group-sum primitive passed 1,000,000 vectors (2,000,000 signed 16-element sums), on AMD EPYC x86-64, compiled with `-march=znver2`; this is not a Rust test or an end-to-end benchmark. Shell syntax, CLI argument construction with a fake binary, and Python summary syntax were checked. Rust tests supplied below remain to be executed on the target. The engine still requires all Vulkan features checked by upstream `unusable()`; do not fake features or change GPU target IDs to bypass that gate.

## What changed

| File | Change | Expected effect / risk |
|---|---|---|
| `src/vulkan/mod.rs` | Opt-in AMD PS5 profile, separate local and GTT heap budgets, Linux `MemAvailable` reserve, per-allocation budget check, fallback only when explicitly enabled and VK_EXT_memory_budget available | May fit more GPU-resident experts, avoiding CPU work and USB page faults. Estimates are not guarantees: driver allocations, GTT ceilings and concurrent processes can still fail. Individual buffers cannot span heaps. |
| `src/vulkan/mod.rs` | Fresh, >64 KiB uploads into device-local HOST_VISIBLE and HOST_COHERENT memory, without an intermediate staging transfer or pre-clear; preserves padded tail; falls back to original staged path | Lower load traffic; CPU writes to a write-combined mapping may be slower on some drivers; test both. Not zero-copy from GGUF; not used to mutate live expert slots, KV or mailbox. |
| `src/backend.rs` | `upload_bytes` backend seam and F16-to-F32 streaming with bounded 4 MiB scratch | Avoids whole-tensor F32 host allocation plus serialized F32 byte vector. A 1 GiB F16 tensor used to require roughly 4 GiB of additional transient host buffers; now scratch <=4 MiB, still needs 2 GiB F32 GPU allocation. Does not change weight precision. |
| `src/cpu_moe.rs` | Fuse Q8_K block sums into existing AVX2 packing; deduplicate activation quantization by immutable slice identity during a batch | Less redundant work when a token routes to multiple experts; unchanged scale, packing and dot products. Benefit unmeasured. |
| `src/gpu_model.rs` | Explicit expert cache size cannot exceed reported free budget minus margin in PS5 mode | Avoids deliberately oversizing expert pools. |
| `src/memory.rs` | Available RAM reader | Conservative bound for borrowing GTT. |
| `tools/ps5/` | Explicit Zen 2 build, launch preset, diagnostics, tests, A/B benchmark scripts | Reproducible deployment and tuning. |

The upstream SPIR-V binaries, all attention/quantization formulas, model architectures and CUDA/HIP kernel sources are untouched. Existing wave32/wave64 and hardware-feature detection remains. The PS5 profile marks the GPU as sharing memory so the existing next-layer expert prefetch is disabled (avoids extra pressure on the shared bus); no GPU shader was retuned or rewritten. The CPU and F16 changes apply to all builds; the Vulkan PS5 paths are opt-in and independently switchable.

## Build / run

From a terminal on the console running Linux, or a compatible x86-64 Linux machine with a sufficiently old compatible glibc and the Rust target installed. Requires stable Rust with edition 2024 support and upstream dependencies, C compiler, Vulkan loader + PS5 RADV driver. Build on the console if unsure about glibc portability. No sudo is used by these scripts.

```bash
bash tools/ps5/diagnose.sh
bash tools/ps5/build.sh
bash tools/ps5/test.sh --gpu
bash tools/ps5/run.sh chat --model /path/to/model.gguf
```

Build uses `target-cpu=znver2` (not `native`, which could introduce AVX-512 on a build machine) and fat LTO. `PS5_LTO=thin` is a lower-memory build alternative. Start with a small GGUF known to the upstream engine, then increase. The preset is an initial engineering hypothesis, **not** measured optimum. Defaults: context 4096, chunk 512, KV q8, 6 CPU threads, 768 MiB free device margin, 1536 MiB host reserve, 256 MiB conversation cache, conservative Vulkan attention budget 16,777,216. Existing user flags override preset defaults. The `--ctx` value applies to the configured model path; special architectures may implement KV formats or chunks differently. Existing calibrated settings may influence options not pinned by the wrapper.

For 64K, only after a smaller context passes:

```bash
bash tools/ps5/run.sh chat --model /path/to/model.gguf --ctx 65536 --kv q8 --chunk 256
```

If memory is insufficient, try 8K/16K/32K first, then q4 KV if supported by that model path, with quality checks. Never infer 64K fit from GGUF file size alone. MoE models may load bigger than RAM through mmap, but repeated reads from a USB SSD can make decode very slow. A dense model must fit its resident weights and working state; MoE's few active parameters do not remove the storage requirement for total expert weights. Qwen 27B at 4-bit has a theoretical 13.5 GB weight payload before metadata/mixed quantization/state, so a full GPU-resident 27B Q4 with 64K context is not a reasonable default for base PS5. Start with dense 4–8B Q4 or small-active MoE models whose quantized files fit the measured budget. Actual model support and memory vary.

## Rollback and quality

```bash
STRATA_PS5_DIRECT=0 bash tools/ps5/run.sh chat --model "$MODEL"
STRATA_PS5_GTT=0 bash tools/ps5/run.sh chat --model "$MODEL"
STRATA_PS5=0 bash tools/ps5/run.sh chat --model "$MODEL"
```

`STRATA_PS5=0` disables the new Vulkan profile, not the common CPU/F16 changes. If a mapped type is not available, mapping fails, or the local budget is insufficient, the staged path is used. On exit the line `PS5 fresh weights: mapped ... bytes, staged ... bytes` tells whether the fast path was actually taken (only fresh raw uploads, not streamed F16 conversion). GPU tests include first-dispatch visibility and a F16 tensor crossing the chunk boundary. An inability to create a real Vulkan device fails the explicit PS5 tests; it is not reported as a successful skip. For the actual original version, compile the unmodified archive with the same `target-cpu=znver2` and flags; optionally point `BASELINE_BIN` at it.

## Benchmark / tuning

```bash
bash tools/ps5/bench.sh /path/to/model.gguf /path/to/results
PS5_THREADS=4 PS5_CHUNK=256 bash tools/ps5/bench.sh "$MODEL" results-t4-c256
PS5_THREADS=8 PS5_CHUNK=1024 bash tools/ps5/bench.sh "$MODEL" results-t8-c1024
```

A/B compares control (profile disabled), guarded local only, GTT, mapped, and combined. Optional `BASELINE_BIN` supplies a true upstream baseline. Uses matching prompt lengths, chunk, KV, threads, greedy decoding, MTP=0 and fixed expert placement (`STRATA_FIXED_EXPERTS=1`) to reduce confounders. A model too big for a case may fail to load; this is recorded as failure, never zero tok/s. Run small model first to compare all paths, then a larger one to test GTT fit. Repeat in reversed order if measuring small deltas (the provided script uses a fixed order; disk cache and thermal state can bias results). Compare min/max as well as mean in JSON. Capture the actual mapped/staged counts and plan/hit rates in logs. Choose defaults based on evidence, not the largest cache. Separately test long context and quality (`perplexity` against an identical corpus, greedy output and/or logits, compare same KV type, do not compare q8 to f16 as if exact). Use a corpus longer than the context and identical weights, prompt, and sampling. No benchmark numbers are prefilled.

Recommended next frontier after the measurements: GPU timestamp profiling (`STRATA_VK_PROF=1`), identify GEMV/GEMMI/attention hot kernels, check actual packed dot acceleration and subgroup properties, then write a targeted RDNA kernel variant with SPIR-V validation and per-kernel numerical parity. Try shader variants and chunk size without weakening the feature checks or forcing fictitious hardware capabilities. A separate research path is direct GPU import of GGUF host pages, only if the driver supports a correct external-host-memory import with lifetime/alignment guarantees. This release does not implement zero-copy file-backed weights, a new native GFX ISA backend, PGO, or 64K validation.

Hardware reference: Sony PS5 specifications https://blog.es.playstation.com/2020/03/18/revelamos-nuevos-detalles-de-la-playstation-5-especificaciones-tecnicas-del-hardware/ ; Linux memory notes https://wiki.ps5-linux.eu/post-install/all-distros/ and https://wiki.ps5-linux.eu/hardware/gpu/ . The latter are community documentation, not proof of any particular console's driver feature set.
