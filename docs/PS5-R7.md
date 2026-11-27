# PS5 R7 — evidence-driven delta

Base: `7657445c42f239ecdbc4065a8d12707441feb39d` (R6U1 plus fork README).
The supplied R7 research archive is a physical campaign on **R6**, not this tree.
The original checkout is not overwritten; the commits port the findings onto the actual R6U1 tip.

## Evidence and decisions

| Research observation (R6, physical PS5) | R7 decision |
|---|---|
| RADV GFX1013, 36 CU, wave32 usable, 64 KiB LDS; integer dot available but signed packed INT8 acceleration false | Keep R2 accelerated gate; skip these candidates in tuner when baseline explicitly reports no acceleration. |
| R3 file-backed import rejected at allocation (`ERROR_INVALID_EXTERNAL_HANDLE`); fallback preserved parity | Keep manual research route and existing per-device rejection handling; no profile promotion. Anonymous allocation acceptance is not evidence for GGUF zero-copy. |
| R4 hybrid max relative logit drift up to 42.32%; GPU-only bitwise passed 12/12 | Require explicit `STRATA_PS5_R4_RESEARCH=1` for hybrid modes; exclude hybrid from deployable profiles. Do not disable the original decode CPU worker. |
| Empty attention: 64/64 outputs NaN without sink | Port merge guard, preserving independent K/V and upstream V rotation. |
| Non-atomic shared MoE miss flag (static race); fixture passed both old and fixed shader in 80 trials | Port atomic OR + existing publication ordering, add regression fixture. No claim that the race was reproduced on hardware. |
| R2/R5 test allocators obsolete; R5 bitwise passed after R6 test fix; attention512 score +2.58%, below 3% gate | Fix tests for R6U1 independent K/V sizes; leave R5 default off. |
| GEMM ~80.5% of captured prefill GPU intervals on 0.6B control, attention ~14.4% | Test 64-position dense non-cooperative GEMM, separate from accelerated R2. |
| Command graph reuses recordings; GPU-copy/CPU interference up to -31.4% measured payload | Port optional counters and timing; do not delete barriers or force CPU/GPU parallelism without dependency/parity evidence. |
| KV Q4/Q2 changed greedy output relative to F16; Q8 matched 32 tokens at 16K but logits differed | Keep Q8 as working reference; no quantization change hidden in speed trials. |
| MTP short-prompt regression and changed expert residence | Leave MTP off in comparison; investigate separately. |

Sources: supplied `R7-RESEARCH/EXECUTIVE_SUMMARY.md`, `AUDIT.md`, `BUILD_REPORT.md`, `HARDWARE.md`, `BOTTLENECKS.md`, `RECOMMENDATIONS.md`, associated raw logs and patches 0001–0006. No research dataset/log archive is added to the repository.

Qwen3.6-35B-A3B UD-Q2_K_XL, ctx65536, Q8, MTP off: physical R6 campaign reported 9,214/10,240 experts resident, 680 MiB KV, and the following single exploratory run at each prompt depth (32 generated tokens):

| Prompt tokens | Prefill tok/s | Decode tok/s |
|---:|---:|---:|
| 8192 | 284.8 | 70.7 |
| 32768 | 202.8 | 61.4 |
| 65024 | 139.5 | 53.0 |

These are not R7 measurements, nor enough to attribute the slowdown to attention alone.

## Build and gates

Use the default `bash tools/ps5/build.sh` with `ps5-r2`. Research used Shaderc 2026.4 and SPIRV-Tools 2025.1; older glslc 2023.8 failed packed-dot compilation. The two repaired shaders are built from the **merged source**, not old R6 SPIR-V binaries. A non-feature legacy build intentionally retains upstream embedded binaries and does **not** contain the new fixes. Do not deploy that build as R7.

```bash
bash tools/ps5/build.sh
bash tools/ps5/test-r7.sh
bash tools/ps5/test-r2.sh
bash tools/ps5/test-r5.sh
PS5_R5_ALL_PAIRS=1 bash tools/ps5/test-r5.sh
bash tools/ps5/test-r7-gemm.sh results/r7-gemm-parity
```

Keep all previous results keyed by binary/model/driver. New binary invalidates previous R6 profiles. R3 and R4 remain isolated research; do not use an old profile for deployment. The R4 parity failure is not repaired in this delta: localize gate/up/down and scatter differences layer-by-layer before restoring it. `bench-r4.sh` explicitly enters research mode, not production.

## Dense64 experiment

Default `STRATA_PS5_R7_GEMM_TILE=0` retains dense128. Setting `64` enables a candidate for generic, dense, noncooperative integer GEMM with more than 64 positions, independent of hardware packed-dot acceleration. Uses existing shared upstream GEMM source with `SM=2`, no split-K, and no change to routed tile lists, cooperative matrix routes, small GEMM, quantized weight layout or KV formats. This is a tile/LDS experiment, not a new integer-dot implementation. Static LDS Q4_K 40 -> 27 KiB, Q2_K 48 -> 35 KiB. No speed claim: fewer positions per tile can reduce reuse and increase workgroups.

First run the physical fingerprint test above (requires actual dense64 marker, compares complete output bits). Then, with supported generic dense 0.6B and 4B controls separately:

```bash
python3 tools/ps5/check-r7-gemm.py
python3 tools/ps5/bench-r7-gemm.py --model "$MODEL" --out results/r7-gemm-control --ctx 8192 --pp 128,512,2048,4096 --tg 32 --reps 3
```

The latter uses fixed Q8, 6 CPU threads, chunk512 and MTP0 by default; prose + Python holdout, ABBA for each, matching greedy and top5, candidate recording marker, 15% drift limit, geometric gain >=3% and no individual paired workload below 95% of baseline. Model/bin/driver identity is checked. It writes research evidence, **never** an approved deployment profile. Do not run instrumented timing simultaneously with performance comparison. On large MoE, the candidate only applies to dense operations, not routed expert tiles; judge its benefit separately.

## Instrumentation

`STRATA_R7_VK_TRACE=1` enables host submit/wait and recording/copy counts. Add `STRATA_VK_PROF=1` for GPU timestamp samples where the engine can collect them. Recorded dispatch/barrier counts are not execution counts for graph replay. Bytes are requested payload, not hardware traffic. Allocation count is not a complete lifetime ledger. `STRATA_R7_TOKEN_TIMING=1` records `r7_load` and `r7_tokens` for generic bench; token times are observations after prefill, not independent kernel latencies (MTP can bunch tokens). Use separate diagnostic runs. Never sum overlapping CPU/GPU intervals as wall time.

## Validation and next gate

Local Python tests and independent output geometry are not Rust/shader compilation or physical Vulkan parity. The integration environment has no cargo/rustc/glslc/spirv-val or PS5. The physical R6 patches are evidence, not certification of their port to R6U1. Validate all mixed-format paths, including empty attention under additional KV combinations if extending the correction. Keep R4 hybrid disabled pending layerwise parity; then measure executor persistence and resource contention. Prioritize dense GEMM only if the physical candidate passes; do not remove synchronization without profiling and a dependency proof.
