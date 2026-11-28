# R14-Audit — end-to-end launch hardening

Base: `e7cb2978b762ab05d80f1d3bdfaa264c7255b9e2`. No shader arithmetic, selected tiles, GGUF, driver, swap, KV format or production defaults changed. R14 remains opt-in; R4 excluded, R13 LUT off, MTP off for campaigns.

## Findings and corrections

- The shared greedy forward path lacked the context bound and `kv_need` reservation present in `step` and `verify_run`. Guard nonempty supported token count and checked position, reserve KV before recording any command. Relevant to fixed-work decode and inherited greedy/speculative callers; growing KV is used by dense multi-slot models, not the current single-slot MoE. No prior physical corruption demonstrated. New physical exact-limit/reset test compares fast argmax to full-readback step and rejects one extra token without advancing position. Growth itself still needs a real dense multi-slot test.
- Full FP32 comparator formerly checked only that claimed token was a maximum; it accepted an incorrect lower top-5 entry, logprob or tied argmax. Reconstruct all five IDs and logprobs from the raw prefill row with FP64 summation; enforce Rust `total_cmp` tie ordering and finite complete rows, including each subsequent raw argmax. Strict raw byte equality remains mandatory. Tests exercise malformed identities and ties. The GPU greedy path's signed-zero tie semantics differ from Rust total_cmp; raw quality uses CPU `ops::argmax`. No claim of universal numerical equivalence.
- Natural generation previously lacked prompt IDs and vocabulary bounds. R6 now emits actual input IDs and vocab for natural as well as fixed mode; existing readers ignore the additional fields. R14 requires the same prompt hash across repetitions/arms, complete prescribed continuation separately from executed tokens, valid IDs, plausible top5 probability mass, symmetric Q8 K/V and zero draft counters. Natural EOS remains valid; no natural timing score. Fixed mode excludes ngram speculation. The two modes are still distinct campaigns, not an automatic cross-mode equivalence proof.
- Fixed output alias risk: reserve raw dump and JSON output exclusively before inference repetitions; no overwriting a pre-existing JSON or a raw dump through a different spelling of the same path. Failure can leave an empty/partial new output; child failure prevents its acceptance. Never reuse output directories.
- Additional opt-in `STRATA_PS5_R14_TRACE=1` (set automatically by the R14 executors) records original dense Q5/Q6 and FP16 fallback markers using a separate per-device atomic bitset. Require both format-specific reference/selected markers, reject mixed baseline/candidate/FP16 dispatch on the selected type. Routed oracle is intentionally excluded from dense reference markers. Conservative: even a legitimate dense FP16 fallback makes this particular campaign inconclusive, inspect it rather than relaxing the gate. Old R6/R11/R12/R13 campaigns do not enable these markers. A marker is evidence of recording, not completion; full buffers and successful process status remain necessary.
- Model/shard/binary stat identities are checked around each child and around campaign hash boundaries (inode, ctime, mtime, size, canonical path). Detects ordinary replacement or write-and-restore without hashing a 35B model every arm; not an immutable filesystem snapshot. Hardware/driver and file SHA-256 identity remain checked at campaign boundaries.
- Dense parity shell formerly accepted duplicate/contradictory Cargo results and duplicate or malformed LDS lines. Require one successful result, both unique LDS records within 64 KiB, exact unique physical marker, expected bytes and matching fingerprints. Handle libtest's first stderr line prefix. Dense Rust test asserts expected total values and nontrivial active accumulation rows; no shader change.

## Launch (prepend to R14 mission)

```
set -euo pipefail
mkdir -p results
PS5_R2_BUILD=1 bash tools/ps5/build.sh
bash tools/ps5/test-r14-audit.sh --offline
bash tools/ps5/test-r13-audit.sh --offline
bash tools/ps5/test-r13-audit.sh --cpu
bash tools/ps5/test-r14-audit.sh --cpu
bash tools/ps5/test-r13-audit.sh --gpu results/r14-audit-inherited
bash tools/ps5/test-r11u1.sh --batch "$MODEL_MOE" "$PWD/results/r14-audit-batch.log"
bash tools/ps5/test-r14-audit.sh --context "$MODEL_MOE" results/r14-audit-context.log
bash tools/ps5/test-r14-dense.sh results/r14-audit-dense
```

The context test is a separate small, model-backed boundary regression (64-token context), not a performance measurement. `--cpu` compiles/runs the selector contract and raw logit writer tests, neither was wired to the previous launch. Only after all critical gates pass, proceed with `docs/PS5-R14.md` using the new binary; do not use old evidence (new required fields/markers). A pure ROI rejection is not a numerical failure. Keep all logs and do not relax gates.

## Executed host evidence

- R14: 18/18 (7 additions to 11 earlier tests); 14,648 independent geometric checks; 936 rejected dispatch mutations. Real shell gate exercised using mock Cargo (one valid case and five negative scenarios: duplicate LDS, mixed dispatch, truncation, differing bytes, contradictory results). No shader/GPU execution in these mocks.
- Inherited `test-r13-audit.sh --offline`: 104/104 across 12 suites, including full R6 CLI mock and KLD comparator; exit 0. Total 122 distinct host test cases including R14. Three KLD and three selected R14 tests were rerun separately and not counted twice.
- A full R14 attempt was interrupted by the environment's execution limit; independent repeat completed 18/18. Logs for successful runs in `docs/validation-r14-audit/`.
- Python syntax, Bash syntax, `git diff --check` pass. Rust/Cargo/glslc/spirv-val and PS5 not available. No Rust unit tests or Vulkan kernels executed here, including 3,486 physical dense cases and context boundary. No physical performance, quality or regressions asserted. General device loss, allocation failure recovery, dense slot growth and concurrent model callers are not exhaustively validated. No changes to the shaders in R14-Audit.
