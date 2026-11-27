# PS5 R12 — defensive audit (before physical run)

Base: `1bea8f66b8edae44fc25446a61b76078dcf84e30`. No performance promotion, no shader arithmetic change, no R4 enablement. The R12 candidate flags and the IQ32 comparison baseline remain unchanged.

## Findings and fixes

1. **Dispatch and range validation:** `Vk::run` previously narrowed `usize` group counts to `u32` without checking the device's three `maxComputeWorkGroupCount` limits. `State::find`, Vulkan copies and typed backend copies used unchecked offset/length arithmetic. The typed `get` allocated before rejecting invalid ranges. Check dimensions, strides, complete blocks, shader `uint` offsets (including `rowb` intermediate and Q8 byte stride), and ranges before allocation/dispatch. Routed output positions remain supplied by the trusted tile producer: these checks do not prove all GPU pointer/index accesses. The shared-memory test override can now lower, not raise, the effective limit; upstream's specific proprietary-driver workaround is retained.
2. **Incomplete quantizer subgroup:** `quant_q8.comp` with `LLAMA_Q8` uses a 16-lane reduction over 256 elements, but the host selected Q8 GEMM for columns divisible only by 128. A trailing half-block can read inactive lanes. Use the existing F16 route for that shape, consistently in prequant and GEMM. PS5 preflight checks F16 fallback even for MMQ types; a deliberately limited device can therefore refuse earlier. Default Qwen PS5 Q8-32 arithmetic is unchanged.
3. **False negative shell gates:** `! grep` is exempt from Bash `errexit`, including in the R12 `else` branch. Use explicit failing branches for leaked markers in R7/R11/R12. Check the exact R12 fingerprint size (37,981,440 bytes per arm). Add a dense non-routed reference for each case with the same weight and gathered input, output offsets and activation. Compare every bit including padding and untouched gaps; this catches common routed output omissions that A/B equality alone cannot catch. The dense reference uses the existing Vulkan arithmetic, not an independent mathematical implementation; its bitwise agreement is still to be validated on PS5. It adds two dense launches per case, so run separately from performance measurements.
4. **Causal evidence:** R12 full-chunk boundary data is now tied to the independently logged tracked row of the FFN input, output and intervention combine. Report every changed token and preserve a signed-zero-only mismatch. R9 filters irrelevant multi-megabyte R12 log lines before retaining them. Scope stays single-device no-HC/no-MTP R9 research, and no proof of absence of a GPU race is inferred.
5. **KLD payload and NLL:** check actual u16 code range, odd-vocab padding, consistency of zero scale and maximum reconstructed log-probability, plus PPL >= 1. KLD encoding remains unchanged. Compute NLL by subtracting the common logit offset before adding normalization, in f64, avoiding catastrophic cancellation for large finite logits. PPL may change slightly as a result; do not require bitwise PPL identity across builds. Encoded dump identity remains the quality gate.
6. **Telemetry:** `/proc/status` contains empty fields on some kernels; old parsing indexed the empty split even for unrequested keys. Handle empty and missing fields, fail closed on unavailable memory readings, handle a process exiting while being stopped, and normalize signal exit codes.

## Validation

Executed host-only: `bash tools/ps5/test-r12-audit.sh --offline` (57 test invocations, some inherited/repeated), including 2,322 R12 tile geometries, 20,000 independent deduplication scenarios and 16 invalid KLD fixtures rejected. The shell tests use a fake cargo to exercise actual script exit paths, not Vulkan. All four archived R11 quality dumps (1,020 or 2,040 scored positions, 248,320 vocabulary entries) passed the updated production Python parser. Logs are in `docs/validation-r12-audit/`.

Pending: Rust build/unit tests, shader compile/validation and physical dense-vs-routed parity. No Rust compiler, glslc, spirv-val or PS5 is available in this environment. No new measured performance claim. This is a targeted audit, not full verification of every architecture, unsafe block, synchronization path or backend.

## Agent order

1. Build (`bash tools/ps5/build.sh`).
2. `bash tools/ps5/test-r12-audit.sh --offline`, then `--cpu`.
3. `bash tools/ps5/test-r12-audit.sh --gpu results/r12-audit-iq` (new directory).
4. Continue the original R12 mission, including batching isolation, IQ32 baseline reproduction, benchmarks, independent corpus, and causal controls. If dense-vs-routed fails, preserve both vectors/geometry and investigate before promotion; do not relax bitwise tolerances to pass. Do not enable R4 for production. Keep R12 flags off until the physical gates pass.
