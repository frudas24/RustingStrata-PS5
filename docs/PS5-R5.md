# PS5 R5 — long-context attention (experimental)

Base: full R4. No format or model changes. Default `STRATA_PS5_ATTN_WG=0` preserves the original kernel selection; the only unconditional prefill change is validation of head geometry. R5 is restricted to AMD PS5-profile wave32 devices, no cooperative matrices, no llama numerics, no `STRATA_VK_DP_OLD`. Explicit unsupported selections fail before logical-device allocation. The existing pipeline shared-memory validator is also retained. The build feature remains named `ps5-r2` for compatibility; it now compiles the three R5 shaders as well, with `glslc` and `spirv-val`.

## Audit and implementation

The existing code already fuses KV dequantization and online softmax; Q8/Q4/Q2 are already available and **not new R5 formats**. R4's non-cooperative tiled prefill has 1024 threads, 32 query/head tasks and two 8192-half tiles (32 KiB total). R5 variants use 256/512/1024 invocations, 8/16/32 tasks and 2048/4096/8192 half values per K or V tile (8/16/32 KiB total, plus any common unused/shared scratch retained by the compiler). At head_dim 128 that is 16/32/64 keys per tile. Workgroups share K/V across GQA and queries. Each loader invocation reads two adjacent elements at a time using the existing `ldkv2` for Q8/Q4/Q2: scales/mins are reused, no pair crosses a row or a 32-value block, and cache values are still rounded to f16. F16 mode preserves the stored half bits. No new global scratch or cache allocations. The online softmax and order of valid keys are unchanged. Output parity on real hardware is a gate, not an assumption.

For sliding-window layers, start at `floor(max(0, earliest_query - window + 1)/BK)*BK`; the remaining per-query mask stays in place, along with causal and QSA masks. No eligible key can be skipped. This avoids loading tiles that all queries in the workgroup would reject. Example: 65,536 positions and a 4,096-token window, one query at position 65,535: at most 4,096 key rows loaded instead of 65,536 for the supported tile sizes. This is a 93.75% reduction in rows visited in that particular case, not a throughput measurement and not applicable to global attention. Smaller tiles may lose cross-query reuse and increase barriers/traffic; do not assume 256 beats 512 or 1024. The 1024 candidate isolates packed loads and sliding-window start from the group-size reduction.

R5 changes only `attn_prefill_m`'s non-cooperative tiled Vulkan path. Decode, MLA/other specialized attention, KV allocation, cache eviction, attention semantics, R2/R3/R4 and CUDA/HIP are not redesigned. The existing prefill submission budget is intentionally kept unchanged for A/B isolation. 64K context still requires supported model context/RoPE, cache, scratch and weights to fit physically. It does not create a ring buffer or reduce the KV allocation for sliding layers. A separate source-audit concern remains: decode `attn_merge` can encounter all empty splits (`M=-inf`, no sink), producing a nonfinite subtraction/division; whether valid QSA routing can reach that state is unverified. Do not claim decode is fixed by R5.

## Execution

```bash
bash tools/ps5/build.sh
bash tools/ps5/test-r5.sh
PS5_CTX=8192 PS5_PP=128,512,2048,4096 bash tools/ps5/bench-r5.sh /path/model.gguf results-r5-8k
PS5_CTX=65536 bash tools/ps5/bench-r5.sh /path/model.gguf results-r5-64k
STRATA_VK_NO_COOP=1 STRATA_PS5_ATTN_WG=512 bash tools/ps5/run.sh chat --model /path/model.gguf --ctx 65536 --chunk 256 --threads 6 --kv q8
python3 tools/ps5/kv-budget.py --ctx 65536 --layers 32 --kv-heads 8 --head-dim 128
```

Start at 8K and scale to 16K/32K/64K only when memory and model limits allow. Benchmark repeats 0/256/512/1024 in forward and reverse order; checks equivalent JSON settings and prompt lengths, finite rates and candidate dispatch log. Full 64K prefill can take substantial time; benchmark does not modify the driver or firmware. Candidate selection remains manual. The physical test fingerprints seven workloads per KV format per variant (including 64K synthetic key positions, head tails, QSA, sliding masks, gate and sink, sliced queries); any difference from R4 fails, as does absent Vulkan or candidate dispatch. `shared_memory_bounds` checks compiled SPIR-V accounting. The test does not prove full-model 64K inference or model quality across different KV formats.

## Validation status

Independent Python reference check: 6,000 dispatch geometries, 100,000 sliding-window bounds, all nine pair-load geometries, 60,000 reference quantized pairs. These tests do not execute the Rust or GLSL. Rust compiler, shader compiler/validator and real PS5 Vulkan execution unavailable in authoring environment. Exact shader parity, GPU occupancy, performance, stability and long-context quality are pending. No tokens/s claims. See `validation-r5.txt` for actual local checks. Next: R6 autotuning with a physical capability probe and strict gates, and PGO once a verified model workload is available; investigate decode split scheduling/all-empty mask separately.
