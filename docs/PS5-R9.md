# R9 — matched single-expert causal replay

Physical R8 (HEAD b9f1395) passed Venus F16/BF16 after the agent's Rust 2024 keyword fix. All 2880 paired projection samples improved; 1872 IQ2XS gate/up samples were bit identical for the first 16 coordinates. The end-to-end diagnostic nevertheless failed (legacy vs contract: max/range 39.7492% vs 34.2595%, max KL 0.73692978 vs 3.3162744). Only four unique reference logits among twelve. The two hybrid arms had 50511 and 50340 CPU rows, so their causal partitions are not controlled. No promotion of hybrid R4.

Both agent patches are independent children of R8. This delta preserves their exact original commits and merges them before R9.

## Experiment

Run `bash tools/ps5/test-r9.sh "$MODEL_MOE" results/r9-layer0`. The output directory must not already exist. Optional `PS5_R9_LAYER`, `PS5_R9_POS` (chunk start), `PS5_R9_TOKEN` (within chunk), `PS5_R9_RANK` default to 0. Start at layer 0, chunk 0, token 0, rank 0; subsequently repeat with other routed layers/ranks in new directories. Six prompt invocations (two prompts repeated three times); not six independent workloads.

All arms use GPU-only R4 scheduling with zero resident experts, no MTP, no cooperative matrices and the existing R8 guarded runner. GPU is uninstrumented; shadow instruments exactly one routed row at the same causal location; legacy/contract do the same but replace that one GPU-computed row just before the actual Vulkan combine. All other experts use GPU. The comparison checks full fixed hidden input, down row, weights, route IDs, slot and coordinates in each arm, bit-identical shadow logits, complete stage coverage, replacement readback and full combined row. The decomposed contract must equal the actual R8 CPU expert function bit for bit. The reported combine differences and final logits are experimental results, not promotion gates. Missing target or mismatch fails closed in the analyzer. Do not benchmark instrumentation or infer performance from it. No wholesale hybrid replay; the purpose is isolating local causes before changing later routes.

Stage comparisons: full gate/up; CPU SiLU with GPU gate/up versus GPU activation; CPU SiLU with own gate/up versus GPU activation; down with GPU activation; down with CPU SiLU on GPU gate/up; down with own activation; actual contract vs decomposition. If activation-only is the first residual, inspect `common.glsl`'s custom expf_/fdiv versus CPU exp/div; otherwise inspect Q8 scale rounding and full-row dequantization. Keep R4 production blocked.

## Validation

Source archive evidence manifest: 277/277 entries verified. `python3 tools/ps5/check-r9.py` exercises seven synthetic acceptance/rejection cases. Bash syntax and Git patch integrity verified. Rust/Vulkan toolchain unavailable here; no claim that R9 compiles or has physical parity. R8 physical results are not R9 results. No production defaults changed.
