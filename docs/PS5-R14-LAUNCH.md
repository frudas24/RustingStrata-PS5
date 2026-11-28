# R14-Launch — final defensive E2E audit with real offline toolchain

Base: `1884ffa9bf9c9f3c9f38d3d6eeceb8c13390e14c` (R14-Audit).
Code commit: `46ca126738668fd8384b13545cd18cce86cc662e`.
No changes to shaders, arithmetic, selectors, default flags or the R4 quarantine.

## Fixes

1. Dynamic KV resize was publishing new buffers layer by layer before all copies and the final wait, and releasing old addresses before graph retirement. The new order is: validate/allocate all replacements, sync, clear graphs while old buffers are alive, copy all, sync, then publish all replacements and capacity. The old buffers and capacity remain intact on a returned allocation/copy/wait error before publication. Graph invalidation itself is allowed even if a subsequent copy fails. This is not recovery from device loss, nor an atomic transaction across split devices. The existing backend destructor behavior on severe wait failure remains outside this patch. The Qwen35B MoE does not normally use dynamic KV.
2. Previously the benchmark's GPU argmax and quality's FP32 readback could each pass their own control/candidate comparisons without matching each other. The optional `--bench-reference DIR` in quality mode verifies the original A/B/B/A records, log dispatch markers, request settings, hardware/model/bin identity, holdout/corpus identity, prompt, prescribed tokens, predictions and top-5. It records hashes for referenced JSON/log/request files and guards file identity during verification. The summary sets `greedy_cross_validated=true` only after the comparison has passed. ROI rejection is allowed as a reference if `validated=true`; a failed validation is not. This is not cryptographic attestation of the original hardware logs.
3. Added a physical dense two-slot KV growth test: verify old bytes remain intact, other slot remains intact, rejected resize is inert, growth at greedy decode, shrink/reset and reuse. Explicit opt-in, compiled but not executed here.

## Real local verification

The supplied Linux offline toolchain had 7134 matching manifest entries. Rust/Cargo 1.100.0, shaderc 2026.1-1, SPIRV-Tools 2026.1. The lock matches after CRLF normalization. Native compiler/linker supplied by this Linux host. No network used. No usable Vulkan ICD/device in this container.

Commands (from repository root, after sourcing the supplied activate.sh):

```bash
export PS5_LTO=thin CARGO_BUILD_JOBS=3
PS5_R2_BUILD=1 bash tools/ps5/build.sh
source tools/ps5/env.sh
cargo test --locked --release --target "$PS5_TARGET" --features ps5-r2 \
  --lib --bin rustingstrata --test r14_dense --test r14_context --test gguf -- --test-threads=2
bash tools/ps5/test-r14-audit.sh --cpu
bash tools/ps5/test-r13-audit.sh --cpu
python3 tools/ps5/check-r14.py
bash tools/ps5/test-r13-audit.sh --offline
```

Baseline R14-Audit release build succeeded. Final code release build and selected test targets compiled; actual Rust test summaries: lib 189 pass/6 ignored, bin 47 pass, GGUF 11 pass, R14 context 1 ignored, R14 dense 1 pass/1 ignored. Total 248 passed, 0 failed, 8 explicitly ignored; some inherited tests also conditionally return without external fixtures. CPU gates rerun successfully, not counted twice. R14 Python 20/20 and inherited Python 104/104 passed, including the end-to-end mock CLI R6. 71 checked-in + 19 newly compiled SPIR-V modules independently passed spirv-val Vulkan 1.3. No GLSL source changes. Actual compiled LDS byte counts:

| Format | Base | Tile32 | Tile64 |
|---|---:|---:|---:|
| Q5_K | 40960 | 20992 | 27648 |
| Q6_K | 49152 | 29184 | 35840 |

Static SPIR-V size, not driver-reported runtime allocation or measured occupancy. The actual LDS log passed the production check_lds gate. Build uses thin LTO rather than PS5's default fat LTO. The local final binary SHA256 is `edd4390e35359e5af3dca7b4b3936090970190ab5d35c5db06beb8938da6c025`. It is not a physical PS5 benchmark binary recommendation. Logs in `docs/validation-r14-launch/`.

## Physical launch

Follow R14-Audit and R14 mission. Build the exact final HEAD on PS5. Run the same offline/CPU/physical context, inherited GPU and R14 dense gates. For a small dense attention GGUF (e.g. Qwen3 0.6B), add:

```bash
bash tools/ps5/test-r14-audit.sh --kv-growth "$MODEL_DENSE" results/r14-launch-kv-growth.log
```

Requires both slots to fit and dynamic KV active; do not count a missing dense model as pass. It is separate from the MoE performance campaign. No physical test executed here. For each matched deep context, run fixed bench and quality with exactly the same pp/ctx/tg/reps, corpus, flags and binary. Example for R12 revalidation at 64K:

```bash
python3 tools/ps5/r14-campaign.py --model "$MODEL_MOE" --tile 0 \
  --base-iq-tile 0 --iq-tile 64 --ctx 65536 --pp 65024 --tg 32 --reps 2 \
  --mode bench --out results/r14-r12-64k-bench
# If bench exits 1, inspect summary.json; continue only on validated=true (ROI rejection).
python3 tools/ps5/r14-campaign.py --model "$MODEL_MOE" --tile 0 \
  --base-iq-tile 0 --iq-tile 64 --ctx 65536 --pp 65024 --tg 32 --reps 2 \
  --mode quality --bench-reference results/r14-r12-64k-bench \
  --out results/r14-r12-64k-quality
```

Use the same pattern for R14 finalists and 8K/32K/64K. Pass the same optional `--corpus` to both. Require `greedy_cross_validated=true` for the combined quality/performance conclusion. Natural generation remains a separate EOS-aware test, not scored. Never relax gates or promote automatically. Preserve GGUF, driver, swap, and flags. A hardware/device loss is a stop condition. Remaining unverified: physical parity/ROI, dynamic growth on actual Vulkan, fat LTO build, full external model tests, and exhaustive recovery/concurrency.
