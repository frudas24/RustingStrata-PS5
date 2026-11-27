# R11U1-Audit — defensive hardening

Base: `4d8c153fd333a811d3046348906764350cae8392` (upstream 1.4.1 integration). Keep the running R11 physical campaign unchanged. This branch adds no performance candidate or new numerical tolerance; it does not promote R4 or R3.

## Findings and fixes

| Priority | Finding | Fix / acceptance |
|---|---|---|
| High | `moe_gemv_q.comp`: non-atomic concurrent writes to shared dedup flag. A duplicate group can also exceed the specialized capacity (17) and suppress slots that were never written. | Atomic first-slot election and count; fall back to one computation per slot on overflow. PS5 feature rebuilds and validates the shader at build time. The non-feature build uses embedded SPIR-V and disables its unsafe multi-slot dedup; potential performance cost outside PS5. Physical fixture compares 5 shapes with independent slot computations, including >17 duplicates and negative IDs. |
| High | `gguf.rs`: incomplete quantization blocks and overflowing size arithmetic could pass early checks or panic; unsupported auxiliary offsets and raw range arithmetic not checked in all cases. | Check rank 1..4, nonzero dimensions, uniqueness, block alignment, checked sizes and offsets; raw auxiliary range returns errors. Eight malformed layouts, valid layout, and invalid raw ranges covered by prepared Rust tests. |
| High | `gguf.rs`: unbounded recursive array parsing; unknown element type accepted for empty arrays. | Bound metadata recursion to 32 levels and reject unknown array element types. This intentionally rejects deeply nested nonstandard inputs; normal flat tokenizer arrays are unchanged. Two additional fixtures. |
| High (research integrity) | `compare-r9.py` counted stages without binding coordinates/widths to the actual intervention; R10 did not tie the fixed intervention directly to the reference routing tape. | Require exact expert, token, rank, slot, stage widths and IDs/weights; new negative cases. Capture combine from the actual `fx` destination, not assumed slab start. |
| High (quality integrity) | `perplexity` divides by zero with ctx-len=0; nonfinite logits can reach lossy KLD encoding; invalid targets can panic. The quality tool checked shape/size and hash but could accept identical malformed rows or an unrelated PPL count. | Check context before division, target/finite values, u32 header dimensions and final PPL. Validate KLD header, exact size, token IDs, finite nonnegative scale, finite nonpositive offset, and unique PPL summary with exact coverage. Normal valid encoding unchanged. |

## Executed here

- `bash tools/ps5/test-r11u1-audit.sh --offline`: PASS. 3 new Python tests (20,000 independent dedup coverage scenarios, 11 malformed KLD fixtures), R9 10/10, R10 10/10, R11 6/6, KLD 3/3. R11 includes independent C AVX2 checks (8,000 FP16 FMA results and 100,036 signed dot products). See `docs/validation-r11u1-audit/offline.log`.
- Historical R9 evidence, q8-layer0, q8-layer0-rank3, q8-layer1: `read_arm` accepts shadow/legacy/contract, six cases per arm (54 arm-cases total). This checks actual archived stage identity, not full new engine execution.
- Shell syntax and Git whitespace checks.

## Required physical gates

```bash
bash tools/ps5/build.sh
bash tools/ps5/test-r11u1-audit.sh --offline
bash tools/ps5/test-r11u1-audit.sh --cpu
bash tools/ps5/test-r11u1-audit.sh --gpu
```

The `--gpu` fixture must produce `AUDIT_MOE_DUPLICATES PASS` and cannot silently skip Vulkan. Then run the existing R7/R5/R11 parity tests, R11U1 batch quarantine, and rerun quality for any promoted candidate. Compare uninstrumented baseline vs audit at 8K and 64K, separately with MTP off and on, with identical model and cache configuration; do not promote based on dedup simulation. The new default shader may change outputs where old dedup lost results. For ordinary valid paths bitwise parity is the goal, not a demonstrated result yet.

## Boundaries

No Rust compiler, glslc, spirv-val or PS5 device available here: all new Rust/GLSL fixtures remain uncompiled/unexecuted. Independent Python scheduling tests are not GPU tests. Old SPIR-V is not regenerated for feature-off builds; host dispatch prevents its faulty dedup path, possibly reducing throughput in those builds. GGUF validation does not prove model authenticity or arbitrary metadata safety. This is a targeted audit, not exhaustive verification. The existing CPU expert fallback during individual decode is unchanged; GPU-exclusive prefill does not establish GPU-exclusive decode. No claim that these fixes solve R9 causal divergence or improve the 35B's throughput. No changes to firmware, model weights, KV precision or R11 experimental flags.
