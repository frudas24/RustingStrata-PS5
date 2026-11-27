# R13-Audit — defensive launch gate for R12/R13

Base: `edd694ad27ef6e0abc93031540f36109b9a9d1dc`. R12/R13 shader arithmetic, LUTs, experimental defaults and R4 quarantine unchanged. Focused audit, not a complete proof of Vulkan memory safety.

## Findings and changes

- FP16 GEMM reads V4 references and divides scalar offsets by four. Require 16-byte input alignment and a stride divisible by four; reject instead of silently reading a different row. General GEMM validates F32 addresses, tile alignment, dense geometry and checked allocation spans on PS5. Routed indices stored in device tiles remain the planner's responsibility: the host checks the tile-list span and first output/xrow, not every device-provided index.
- Invalidate the old Q8 cache key before replacing the backing allocation or attempting a new quantization. A miss must be outside a batch with aligned input and a checked PS5 allocation span. Hits retain existing semantics; this is not a proof of all SAME_X callers or cache generations.
- Poison all active outputs with NaN for act=0 in the IQ2 and integer parity tests. Compare complete buffers against the existing dense Vulkan oracle; require all active values finite and preserve guard regions. Common omitted stores can no longer pass merely because both arms share an initial sentinel. The dense oracle is independent of routing, not independent shader arithmetic.
- Require per-format routed markers for IQ3_XXS and IQ4_XS, and reject mixed integer geometries/FP16 fallback. These markers identify recorded dispatches; physical completion still depends on the Rust tests and checked outputs. Strictly tailored to the target Qwen GGUF with both formats.
- LDS preflight now requires seven uniquely identified specialization records, checks expected minimum deltas and reported limits, and requires exactly one non-skipped Rust test. Parity scripts require exact output lengths and test counts; compare the full files. No shader source changes.
- Performance gates require the pooled improvement and each paired geometric mean >=3%, no phase loss >5%, drift <=15%, existing greedy/top5 checks. Rejection of noisy marginal winners is intentional. Holdout identity checked across the campaign. Profile-only measurements are not scoring runs.
- Fingerprint swap devices and swappiness; collect system-wide start/end/peak sampled swap and page-in/out counters, plus completion RAM guard. Global counters cannot attribute swap exclusively to the engine. Do not change the swap configuration. Existing R6 fingerprint files must be regenerated for comparisons using the new schema.

## Executed offline

Logs in `validation-r13-audit/`: 14 new unittest cases (7 gate tests + 7 mocked parity campaign cases), 65 inherited R9/R10/R11/R12/R13/audit/KLD cases and 10 R6 tests: 89 passed. The seven mocked campaign cases consist of a positive control and six rejections (duplicate LDS, missing format, mixed fallback, wrong arm, bad file size, zero tests). Gate tests reject 756 marker mutations. The mock uses synthetic Cargo output/files; no Vulkan execution. Inherited geometry/lookup/product-dot and dedup tests are independent models, not GPU results. The combined mock run exceeded the execution limit; the cases were rerun separately successfully. The full R6 CLI mock end-to-end case exceeded the limit and remains pending. Cargo, rustc, glslc and spirv-val are absent here. New Rust span/alignment tests and physical GPU regressions not executed. Additional allocation lookups may have a performance cost; measure it before claiming no regressions.

## Physical launch sequence (new binary mandatory)

```bash
PS5_R2_BUILD=1 bash tools/ps5/build.sh
bash tools/ps5/test-r13-audit.sh --offline
bash tools/ps5/test-r13-audit.sh --cpu
mkdir -p results
bash tools/ps5/test-r13-audit.sh --gpu results/r13-audit-parity
bash tools/ps5/test-r11u1.sh --compile
bash tools/ps5/test-r11u1.sh --batch "$MODEL_MOE" "$PWD/results/r13-audit-batch.log"
```

Stop on compilation, correctness, missing-test, wrong-dispatch or memory-guard failure. Preserve logs and diagnose, do not relax tolerances. The R13 parity run covers 648 cases x 2 IQ2 arms and 1296 cases x 6 integer arms (9072 cases in total). The legacy R11 winner must also be remeasured under the audited binary. Continue with the R13 campaign in `PS5-R13.md`: R12 alone, R13 alone, combined, quality and long-context tests, profiling separately. Compare against R11 IQ32 on the same binary and machine state. Do not promote any new flags without physical evidence; R4 remains blocked.
