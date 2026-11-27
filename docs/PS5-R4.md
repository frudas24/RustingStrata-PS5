# PS5 R4 — adaptive heterogeneous MoE prefill

Status: source implementation and independent CPU models validated; **Rust build, shader compilation, actual PS5 execution, logits comparison and throughput are not validated**. No Rust toolchain, GLSL compiler, model weights or PS5 GPU were present in the development environment. This checkpoint contains R1–R3 unchanged except for the explicitly listed integration changes. Do not promote R4 to a production default before the physical gates pass.

## Why this frontier

The original generic `GpuModel` already overlaps CPU misses and GPU cache hits during decode/verify via `CpuWorker` and the mailbox. Prompt prefill instead stages missed experts into a three-bank GPU staging slab, with resident grouped GEMMs and optional transfer-queue overlap. The existing `rows.is_some()` batch-step CPU route remains unchanged. R4 experiments with the prompt route, where a low-occupancy expert's staging cost may exceed CPU compute. It does not add another mailbox, change routing/top-k, rewrite decode, promote/evict cache slots or change quantization formats.

## Policy

* Opt-in, Linux, AMD Vulkan with the R1 PS5 preset active. Off by default, including `tools/ps5/run.sh`.
* Scope: generic `GpuModel` prompt prefill (`rows == None`), main layers only; no specialized DeepSeek/Gemma/GLM/Hy4/Kolibri/Inkling paths, no Hadamard, LoRA or active NaN trace. Requires a CPU integer dot kernel for all three expert tensors. Use local inference, not multi-tenant serving: policy history is per model part, not per request owner.
* Keep all resident experts on GPU. Only nonresident selected experts with 1..MAX_ROWS routed positions qualify. Candidate order is `(row count, expert id)`. Move whole expert groups, never individual partial rows, and preserve sorted expert order in both paths. The unchanged `inv` and `moe_combine_seq` apply original router weights.
* Arms: 0 = all misses staged to GPU; 1 = half (rounded up) of eligible experts to CPU; 2 = all eligible experts to CPU. Ineligible misses always stage. If only one candidate, arms 1 and 2 coincide, so auto uses two arms. Even `cpu` mode leaves resident/ineligible experts on GPU.
* Per-model/per-layer timing keyed by power-of-two buckets of token count, resident rows, missed rows and eligible expert count. Bounded to 4096 keys per model part. First sample each available arm; EWMA alpha=0.2; prefer the less aggressive arm unless the next arm is at least ~10% faster. Every 16th visit explore in rotation. No on-disk history. Distinct routing patterns in one bucket may differ; short sessions may never converge.
* Start clock after router readback; measure through GPU staging and GEMMs, optional host activation readback, CPU expert compute, CPU row upload and explicit GPU synchronization. This is wall-clock layer-fragment latency, **not** a GPU kernel timestamp. Measures contention indirectly. `gpu` isolates measurement overhead; `off` is R3 baseline. GPU dispatch stays on the original thread; scoped CPU thread uses the existing Rayon CPU integer implementation. `kick()` submits queued GPU work even without a transfer queue. The scope joins before CPU results are uploaded, even if GPU dispatch errors. A mixed layer is evidence of both routes, not proof that GPU/CPU physically overlapped for the full duration.
* Budget uses `min(SCRATCH_MB, MemAvailable - HOST_RESERVE_MB)`, with fail-closed zero if MemAvailable is missing. Estimated input readback 8 * tokens * hidden; per pair 64 * (hidden + FF) + 512 bytes, checked arithmetic. This is a conservative estimate, not a hard limit on allocator, Rayon, file-cache or Vulkan memory. Budget is not a reservation. No external memory imports are enabled by R4.
* Correctness: original gate/up/down and combine math; CPU and GPU have different accumulation/quantization rounding, so **hybrid output is not necessarily bit-identical**. On GPU-only timing control the gate requires identical f32 bits. Hybrid gate uses a provisional max absolute drift <= 5% of baseline logit range and verifies finite values and near-tie argmax bounds. This is not proof of quality across models; test perplexity/quality before deployment.
* Failure: absent/ineligible policy or no budget retains baseline GPU staging. GPU/device or CPU-task failures propagate errors; no silent retry of a partially executed forward. Off is the rollback. No firmware, clock or OS modifications.

## Configuration

| Environment | Default | Meaning |
|---|---|---|
| STRATA_PS5_SCHED | off | off / gpu / half / cpu / auto |
| STRATA_PS5_SCHED_MAX_ROWS | 4 | eligible rows per missed expert, 1..32 |
| STRATA_PS5_SCHED_SCRATCH_MB | 64 | scratch estimate cap, 1..512 MiB |
| STRATA_PS5_HOST_RESERVE_MB | 1536 | retain host margin; R4 requires 1..65536 MiB |
| STRATA_PS5_SCHED_LOG | unset | log each completed trial if 1 |
| STRATA_EXPERT_MB | original auto | explicit cache slot budget, 0 forces no slots |

## Reproduction

```sh
bash tools/ps5/build.sh
bash tools/ps5/test-r4.sh /path/to/generic-moe.gguf
bash tools/ps5/bench-r4.sh /path/to/generic-moe.gguf results-r4
STRATA_PS5_SCHED=auto STRATA_PS5_SCHED_LOG=1 \
  bash tools/ps5/run.sh chat --model /path/to/generic-moe.gguf --ctx 8192 --chunk 128 --threads 6 --kv q8
```

First run existing R1/R2 regression gates, and R3 only if host import is supported. `test-r4.sh` uses separate processes and identical tokens, disables zero-copy and adaptive placement, forces an empty cache and compares 12 complete logit vectors per mode (two prompts, three repetitions, prefill and one following fixed decode token). GPU control must be bitwise identical; hybrid modes must actually execute CPU rows; half must show mixed rows. It deliberately fails rather than skip unsupported hardware/models. A 5% logit-range criterion is provisional, not a universal quality threshold.

`bench-r4.sh` fixes cache capacity (default 512 MiB), quantization, R2 settings and disables R3 import, so that one variable changes. It runs off1/gpu/half/auto1/auto2/off2, with 5 reps per prompt length by default. The summarizer requires actual mixed trials in half/auto and compares prefill and decode throughput, including timing overhead and logging overhead. `STRATA_EXPERT_MB=0` or `STRATA_PS5_SCHED_MAX_ROWS=8` may be needed to exercise this path on a model with few eligible misses; use the same settings in all arms. Use separate `--threads`/chunk sweeps, compare warm and cold runs, check peak RAM and storage faults; do not assume 6 threads or 128 chunk are optimal. A change in decode rate may be due to altered prompt states and different routed experts, not a decode optimization.

## Executed here (independent, not engine execution)

- `check-r4.py`: 20,000 generated router/cache/budget scenarios, 60,000 arm partitions/scatter checks, 1,024 synthetic EWMA observations.
- `check-r2.py`: 4,080 GEMV/MoE geometry cases, 780 GEMM tail cases.
- `check-r3.py`: 100,000 window cases and 5 boundary cases.
- `simd_check.c` built with GCC `-O3 -march=znver2 -Wall -Wextra -Werror`: 1,000,000 AVX2 vectors, 2,000,000 group sums.
- All PS5 bash scripts syntax checked. R4 parity/benchmark analyzers tested on valid synthetic data and rejected synthetic inactive trials.
- Full Rust test/compile, Vulkan GPU parity and speed remain pending.

## Continuity / next frontier

R1 memory/unified allocation; R2 specialized GPU kernels; R3 optional import; R4 prefill scheduler. R5 investigate KV and attention for long context (32K/64K), actual memory accounting, Q8/Q4 quality and prefill/decode throughput. R6 physical autotuning/PGO. A further R4b may optimize cache promotion cost and decode policy, use GPU timestamps and bandwidth counters, or replace the per-layer host thread creation with a bounded persistent executor, but only after R4 has physical results. Do not move resident expert decode placement or import dynamic buffers in the next round without separate tests.
