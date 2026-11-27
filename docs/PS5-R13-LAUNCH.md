# R13-Launch — final defensive audit

Base: `7fbdc6dc4e7ec5e334ee6e2ce9a9c37ab5cd4e47` (R13-Audit). No shader, quantization arithmetic, experimental default, model or R4 policy change. No new acceleration candidate. Freeze after this audit and run the physical campaign.

## Findings

1. Q8 cache keys are CPU-side state. A recorded Vulkan graph may quantize scratch on replay without updating that state; a discarded recording can publish a key for work that never executes. Invalidate both keys at graph boundaries, including replay and failed recording. Keep reuse within the ordered recording/sequence. Reject nested graph/batch capture. Scratch growth with an existing allocation clears recorded graphs before replacing the allocation; reject growth during recording rather than freeing a still-referenced buffer. The first allocation is permitted. This is a targeted Q8 lifetime fix, not a general proof of all graph resource lifetimes. Clearing all graphs on growth is deliberately conservative; unusual larger graph shapes may now return a diagnostic instead of running. Two additional mutex acquisitions per graph boundary and graph re-recording on scratch growth may affect performance; measure them. The physical `q8_graph_cache_lifetime` regression covers replay, discarded recording, growth rejection during recording and re-recording after growth outside recording. No physical execution here.
2. R13's orchestrator inferred baseline/candidate from LUT string. It could not validate a winning R12 tile when all LUT candidates lost. `--lut off --base-tile 0 --tile 32|64` now works for benchmark and quality, with distinct artifacts; reject a no-op scored comparison. The existing arms and default flags are unchanged. Preserve/validate holdout and corpus identity before and after each arm (also in R11/R12 runners). Reject malformed dispatch records instead of skipping them. Enforce timeout at observed completion; a completion very near the deadline may be rejected conservatively because the host samples process state.
3. Batch isolation compared identical nonempty byte files and relied on Python asserts (disabled by optimization). The physical test now emits its vocabulary size; `verify-r11u1-batch.py` checks test count, both arm markers, staged/CPU decisions, exact six-row byte count, token bounds, finite logits and exact equality, with explicit errors. Old batch logs lack the new vocabulary record; rebuild before running. The independent verifier accepts a valid synthetic fixture and rejects six invalid fixtures even with asserts compiled out.
4. Inherited R7, R11U1 and R12 launch scripts could accept successful Cargo exits without a test actually running. Require positive exact test counts for targeted CPU/GPU checks. Run R7 attention/mailbox regressions and the new Q8 graph test in `test-r13-audit.sh --gpu` before the 9,072 routed parity cases. Clear ambient engine flags for the R7 and graph checks. R12 mock fixtures emit the actual expected test-count lines. Increase the R13 synthetic shell fixture timeout (30 to 120 seconds; no effect on production or physical benchmarks) after a valid mock took 36 seconds on this host. The R6 exit-code fixture uses a shell and a wider time limit to avoid confusing a slow Python startup with the intended nonzero exit test.

## Validation actually run

Logs in `validation-r13-launch/`. R9 10, R10 10, R11 6, R11U1 audit 3, R12 19, R12 audit 6, R13 8, R13 audit 14 (13 passed initially; positive shell mock timed out then passed alone), R13 Launch 8, R6 11 (including the complete CLI mock) and KLD 3: 98 distinct passed. The R12 mocks initially failed because the new exact test-count gate rejected their incomplete fake output; after updating the fixtures all six passed. R6's initial exit-code fixture hit a timeout; its replacement and the other selected tests passed. New tests include actual orchestrator calls with mocked inference, malformed marker rejection, six batch artifact mutations, no-op rejection, corpus/holdout mutation, and a real process timeout. These are not GPU measurements. The complete R6 CLI suite subsequently passed (11 tests, 80.142 s); the earlier selected run also passed after the exit-code fixture adjustment. Shell syntax and Python AST checked. No Cargo/rustc/glslc/spirv-val or PS5 present, so compilation, physical regression and throughput remain open. No exhaustive guarantee of absence of bugs.

## Mandatory physical launch gate

```bash
PS5_R2_BUILD=1 bash tools/ps5/build.sh
bash tools/ps5/test-r13-audit.sh --offline
bash tools/ps5/test-r13-audit.sh --cpu
mkdir -p results
bash tools/ps5/test-r13-audit.sh --gpu results/r13-launch-parity
bash tools/ps5/test-r11u1.sh --compile
bash tools/ps5/test-r11u1.sh --batch "$MODEL_MOE" "$PWD/results/r13-launch-batch.log"
```

The GPU wrapper runs the deduplication check, R7 empty-attention and mailbox regressions, Q8 graph replay/discard/growth and full R12/R13 LUT parity. A nonzero status, missing test, failed parity, invalid memory guard or unsupported graph size is a stop condition; do not loosen tolerances. Repeat R11 IQ32 on the new binary before measuring candidate improvement. Follow `PS5-R13.md`, retaining R4 quarantine. If R12 wins and every R13 LUT loses, run the final comparison with `--base-tile 0 --tile "$TILE" --lut off`, with the same quality and long-context gates. If both lose, retain R11 IQ32 rather than forcing a candidate. Do not change drivers or system swap settings. No R14 until results are received.
