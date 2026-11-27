# R7-V1: UI-Venus transfer (experimental, incremental)

Base: `044674ba93af6eef93a38b9c4cede42c846a06ef` (R7). This branch is separate from the physical R7 measurements currently in progress. Do not merge before completing the baseline capture. No changes to R1–R7 defaults, R4 production gate, Vulkan shaders, firmware, or OS. No new dependencies.

## Adopted and flagged

| Flag (default off) | Mechanism | Scope |
|---|---|---|
| `STRATA_PS5_VENUS_F16=1` | 3 weight rows x 4 tokens, F16C+FMA in registers | `Tensor::matmul`, currently used by vision CPU fallback; not generic CPU text prefill |
| `STRATA_PS5_VENUS_MATVEC=1` | F16C+FMA single token, four vector accumulators | `Tensor::matvec`, including generic CPU text inference |
| `STRATA_PS5_VENUS_LOG=1` | one dispatch evidence marker per kernel | use to prove the path ran, not for timing |
| `STRATA_PS5_VENUS_TRACE=layer:absolute_position:router_rank` | one selected real GPU routed row per matching prefill, CPU shadow and cross-feed | generic all-GPU Vulkan MoE, diagnostic only |
| `STRATA_PS5_VENUS_TRACE_VALUES=1` | full selected vectors in diagnostic JSON stderr | opt-in; log can include model-derived activations |

The CPU code is a Rust intrinsics adaptation of the strategies in Venus `internal/kernels/gemm3_mem_amd64.s` and `simd_amd64.s` (`dotF16AVX`). It does not copy Go ABI or change quantized integer experts. It requires runtime AVX/FMA/F16C, exact F16 bytes, scale=1, 8-aligned column count. For GEMM only complete 3x4 tiles use the new accumulation order, all edges use the old dot. `STRATA_SCALAR` prevents both dispatches. New results are not promised bit-identical (FMA and reduction order). GPU F16 weight upload remains unchanged (FP32 conversion), and the GEMM still allocates the existing output transpose. No model-wide weight expansion or new worker pool. Large batches may reread/reconvert weights more than the reference; do not assume faster.

The trace follows the idea of Venus `internal/decoder/ffn_capture.go`. It reads actual `h`, fused SwiGLU `ge` and down output `ye` using router rank, inverse permutation and expert range checks, before combination. It never substitutes CPU results. It reports CPU single-pair and grouped-singleton output, CPU down on the measured GPU activation, and scalar FP32-dequant reference. GPU gate/up are fused, so only CPU gate/up and the actual GPU fused activation are available, not separate GPU gate/up or actual quantized GPU buffers. A large difference in activation with smaller cross-feed difference suggests an upstream gate/up/quantization divergence, not definitive causality. Grouped singleton cannot exclude multi-row grouping or scatter bugs. R4 remains blocked by R7; no attempt to harmonize Q8 contracts without measured evidence. Trace explicitly rejects R4, non-Vulkan, LoRA, Hadamard, nonstandard batches and oversized rows. Repeat in separate diagnostic runs for different ranks/layers/positions. Keep logs local if the model is sensitive.

Deferred: SIMD attention/exp, scratch and session changes, procedural experts, Q8/LLAMA_Q8 harmonization. Do not copy rejected FP32-expansion variants. No new autotuner choices: R6 continues to clear STRATA_* variables and does not promote these candidates.

## Reproduce

Independent C arithmetic/layout mirror on an x86-64 AVX/FMA/F16C host:

```sh
cc -O3 -march=znver2 -Wall -Wextra -Werror tools/ps5/venus-f16-check.c -o /tmp/venus-f16-check
/tmp/venus-f16-check
```

Rust CPU tests (no GPU/shader toolchain needed for these commands):

```sh
bash tools/ps5/test-venus.sh --cpu
bash tools/ps5/test-venus.sh --cpu /absolute/path/small-f16.gguf
bash tools/ps5/test-venus.sh --model /absolute/path/small-f16.gguf /absolute/path/venus-cpu-results
```

The GGUF microbenchmark selects a 2D F16 weight tensor (override `VENUS_F16_TENSOR` with a name), limits to 384 rows and uses deterministic synthetic activations. A/B/B/A timing samples are emitted but not automatically promoted. The model test uses identical teacher tokens from prose and Python, full finite logits, top1, and 8 greedy tokens per prompt; max absolute logit drift gate 0.05 (provisional, not quality proof). A candidate must emit dispatch evidence. Separate processes keep the environment immutable. For the model choose a small generic dense Qwen F16, not Q4-only; Q4 is a negative control for these flags. Vision performance requires a compatible F16 mmproj and actual image control.

Actual Vulkan MoE capture (first finish R7, then compile feature ps5-r2):

```sh
bash tools/ps5/test-venus.sh --trace /absolute/path/moe.gguf 0:16:0 /absolute/path/venus-trace.log
```

Select a valid routed MoE layer for the model instead of 0 if it starts with dense layers. Position must be <128 for this fixture; rank < experts used. Script requires exactly one actual record and all eight comparisons finite. This is a diagnostic gate, not approval of R4. To test the alternate quantization contract, use a separate manually controlled invocation of the Rust trace test with the existing `STRATA_LLAMA_NUMERICS` setting after verifying dispatch; script uses the default contract. Do not time trace runs because readbacks synchronize GPU.

## Evidence in this delivery

Host AMD EPYC 9V74, not PS5. C mirror passed 5000 randomized tiles with odd-byte weight address; 60000 GEMM outputs and 60000 GEMV outputs against FP64, bound `2e-6 * sum(abs(products)) + 1e-6`. Maximum observed GEMM absolute error 3.15010548e-05. This is not execution of Rust code. Independent R2/R3/R4/R5/R7 geometry scripts also passed. Synthetic trace checker rejects missing, duplicated, wrong-target and nonfinite records. Bash syntax checked. No Rust compiler/cargo/glslc or physical PS5 available in this environment; Rust build, actual GPU trace, model parity, performance and codegen remain pending. No tokens/s gain claimed. Verify assembly on the actual toolchain (register spills, memory-form VCVTPH2PS) before promoting.

Next: after physical R7 tests, use the trace to locate first divergence and choose a minimal numerical fix. CPU F16 flags are independent of this work. Recalibrate profiles only after acceptance; binary identity has changed even with flags off.
