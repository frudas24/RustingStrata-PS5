# PS5 R2 — wave32 quantized kernels and small dense prefill

Base: RustingStrata-PS5-R1-full.zip, original engine 1.3.7. R1 remains the control. See `PS5.md` for memory planning, upload path, target hardware, original attribution and limitations. No firmware, storage configuration, clocks or kernel modifications. The engine remains a Linux Vulkan application.

## What changed

- `gemv_q` and the single-token `moe_gemv_q`: optional 64- and 128-thread workgroups (2 or 4 logical wave32s instead of 8). The packed signed INT8 dot product, quantization, butterfly reduction order, output scale and buffer layout are unchanged. Ordinary, dual-matrix and indexed GEMV use matching grid geometry. For multi-token MoE and dense GEMV NT > 4, retain the R1 path. MoE row-stride and group size use the same wave count; all table initialization remains before row-dependent exits.
- `gemmi_dp_ps5_32`: 32 token positions x 64 output rows instead of 128 x 64, 256 threads, same signed dot-product algorithm and order of accumulation per output. Dense INT8 GEMM only, nd 1..64, no cooperative matrix support; larger batches, float weights, incompatible column alignment and expert tile lists use existing kernels. The new variant guards scale loads so the small tile never reads unused second subtiles. No change to expert tile encoding or output activation semantics.
- Device selection: explicit PS5 AMD profile, wave32 support and actual `integerDotProduct4x8BitPackedSignedAccelerated`. Feature build and runtime switches are both required; unsupported experimental requests fail with a diagnostic, rather than silently reporting a benchmark of a different kernel.
- Optional Cargo feature `ps5-r2` compiles five new shader variants in OUT_DIR using glslc and validates them with spirv-val. Existing committed SPIR-V is unchanged. Baseline shader preprocessing is intended to be semantically identical; the upstream byte-for-byte `spirv_is_fresh` test could not be executed here.
- Preflight accounts for selected variants, including small GEMM; it still checks the full original GEMM because other batches/tile lists need it. The new tile does not remove the original device capability requirements.

## Static resource analysis (not a performance measurement)

The small GEMM changes shared arrays for X from 128 to 32 positions. Weights retain the same 64-row double buffer. From the declarations, Q4_K (KS32, PW20) goes from 40960 to 20992 bytes (48.75% reduction); Q2_K (KS16, PW36) from 49152 to 29184 bytes (40.625%). A compiled SPIR-V accounting test verifies those sizes on target. Lower LDS usage may improve occupancy but scheduling, VGPR pressure, CU utilization and memory throughput must be measured. More GEMV workgroups can help small row counts, but can also increase scheduling and shared lookup-table initialization costs; never assume 64 or 128 beats 256.

## Build and validate

Requires stable Rust (edition 2024), C compiler, `glslc`, `spirv-val` (spirv-tools), upstream dependencies and functional Vulkan 1.3 driver. Run:

```bash
bash tools/ps5/diagnose.sh
bash tools/ps5/build.sh
python3 tools/ps5/check-r2.py
bash tools/ps5/test-r2.sh results/ps5-r2-parity
bash tools/ps5/bench-r2.sh /path/model.gguf results/ps5-r2-bench
```

`test-r2.sh` requires a real Vulkan device and compares complete binary outputs (not rounded decimal summaries) for R1 versus all five experimental combinations. Covers 5 fixture types, widths 256/512 and Q8_0 width 320, odd row counts, NT 1/2/3/4/5/17, ordinary/dual/indexed GEMV, MoE single/multiple tokens and skipped experts, and GEMM n 1/15/16/17/31/32/33/63/64/65 with output overwrite, SwiGLU and add. Uses default quantization numerics and forces non-coop for A/B. A failed comparison stops; no automatic promotion. This is a relative parity test, not a substitute for independent reference-quality tests or end-to-end model evaluation.

`bench-r2.sh` uses the same binary, GGUF, expert-placement policy, R1 memory flags, context, chunk, KV, CPU threads and MTP=0 for all six configurations. Inspect raw JSON/logs and repeat in reversed order or alternating runs to rule out thermal and page-cache bias. Use `STRATA_VK_PROF=1` separately for GPU profiling. Profile names of new kernels are distinct. Select the winning geometry only after tests pass and measured ROI is positive. Defaults intentionally remain 256 threads / original GEMM until that evidence exists.

```bash
# Experimental configuration; only after the parity gate passes:
STRATA_PS5_GEMV_WG=128 STRATA_PS5_SMALL_GEMM=1 \
  bash tools/ps5/run.sh chat --model "$MODEL" --ctx 8192 --kv q8
# Return to the R1 kernel path (even when feature is compiled):
STRATA_PS5_GEMV_WG=256 STRATA_PS5_SMALL_GEMM=0 \
  bash tools/ps5/run.sh chat --model "$MODEL"
# Build without the optional shader compiler or validator:
PS5_R2_BUILD=0 bash tools/ps5/build.sh
```

## Validation actually performed in preparation environment

- Python CPU geometry model: 4,080 GEMV/MoE group/split cases, 780 small/full GEMM boundary cases, exact coverage, no duplicates; LDS size arithmetic checked.
- R1 standalone C AVX2 check repeated: 1,000,000 vectors and 2,000,000 signed group sums passed on AMD EPYC x86_64 (not the PS5).
- Shell syntax and Python syntax checked. Archive integrity checked. No Rust, glslc, spirv-val, PS5 GPU, Vulkan ICD or GGUF in preparation environment. Neither Rust build nor shader compiler/validator nor GPU inference ran. No measured tok/s and no verified PS5 compatibility. R2 is experimental source, not a tested binary.

## Remaining audit / next frontier

Multi-token MoE deduplication is intentionally not retuned. Its shared boolean duplicate flag can receive concurrent writes; its cap relies on correct max_dup from callers. Investigate with dedicated GPU validation before modifying the multi-token path. R3: study mapped/host-import GGUF weights only when driver external-host-memory support, alignment, padding, synchronization and lifetime contracts are established. Do not call ordinary mapped uploads zero-copy. Keep R1/R2 controls and measure load time, memory residency, page faults, throughput and quality; fallback to explicit staging if unsupported.
