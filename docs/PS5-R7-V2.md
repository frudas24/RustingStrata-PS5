# R7-V2 — native BF16 and real-weight laboratory

Base: `30b4a26292f8ea365c5d260f2085b24228e3f725` (R7-V1). No Vulkan, MoE scheduler, or default changes. R3/R4 remain isolated. R6 autotuning strips STRATA_* overrides and does not promote this flag.

## New opt-in candidate

`STRATA_PS5_VENUS_BF16=1` enables BF16 CPU GEMV (only exact contiguous 2-byte rows, unit scale, positive dimensions, cols multiple of 8, AVX2+FMA; respects STRATA_SCALAR). BF16 expansion is a zero extension and left shift of 16 bits, not a BF16-to-F16 conversion. Four independent accumulators. Existing Rayon output partitions; no new pool or dependencies. `STRATA_PS5_VENUS_LOG=1` reports `bf16_matvec` on actual execution. F16 remains separately gated. CPU matmul, quantized experts and Vulkan are unchanged. BF16 compute is FP32 FMA, not hardware BF16 dot instructions.

## Reproducible independent real-model test

The supplied model is a HuggingFace Qwen3-0.6B directory, BF16 SafeTensors, 311 stored tensors; 751632384 stored elements (includes both embedding and output matrices), SHA-256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`. No model weights distributed in this delta.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python3 tools/ps5/venus-real-model.py /path/Qwen3-0.6B /path/results.json
```

Requires Linux x86-64, AVX2/FMA/F16C, GCC, NumPy, PyTorch CPU, safetensors. Uses local model files only, no remote code or Transformers. Independent Python implementation of dense Qwen3 with identical FP32 non-linear operations for both paths, comparing PyTorch FP32 matmul over original BF16 weights against an independent C mirror of the new AVX2 kernel. Full causal recomputation (not KV cache). Fixed vocabulary IDs for two four-token sequences; three greedy continuations each. Not a test of tokenization, actual Rust engine, quality benchmark, or PS5 performance. Real weights and activations from layers 0, 14, 27, plus output head; 22 sampled matrices, first 384 rows, four causal inputs, 33792 outputs per tested kernel; FP64 references on identical weights. F16 samples explicitly convert BF16; conversion drift measured separately. Eight alternating microbenchmark samples, three invocations per sample, single-thread C loops compiled -march=znver2; no Rust/Rayon overhead. PyTorch runs with four threads. Host AMD EPYC 9V74; no Rust toolchain or PS5 GPU.

Results in `docs/validation-venus-v2/qwen3-06b-host.json`: no nonfinite source/converted weights; 120378 changed by BF16->F16 (max 2.98023224e-8). Max causal logit absolute drift 3.3378601e-5, identical top-1 at every compared position, identical six greedy tokens. FP64 max errors: BF16 GEMV and F16 GEMV 1.76053404e-4, F16 GEMM 6.64334654e-4, all pass scale-sensitive tolerances. F16 3x4 median speedup 2.28x against four independent F16 GEMV calls, 22/22 sampled matrices faster. BF16 AVX2 median 15.82x against deliberately sequential FP32 scalar C loop, NOT against Strata or llama.cpp; BF16 is roughly 9% slower than F16 on this host median, no extrapolation. Raw times retained; do not promote flags from this evidence.

## Pending Rust / physical acceptance

Rust remains uncompiled here. On a host with the toolchain, first run `bash tools/ps5/test-venus.sh --cpu`. For full Rust model parity, convert the original HF directory using the normal llama.cpp GGUF converter with `--outtype bf16` (preserve original weights) and run `bash tools/ps5/test-venus.sh --bf16-model /path/model-bf16.gguf results/bf16`. Existing `--model` remains F16, and tests disable other Venus flags explicitly. Only after parity, run actual A/B measurements on PS5. No conversion tool bundled. Model-specific BF16 GGUF conversion and native Rust inference remain pending. No claim of improved MoE Q2_K or Q4_K inference.
