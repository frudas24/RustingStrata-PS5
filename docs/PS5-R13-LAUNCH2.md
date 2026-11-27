# R13-Launch2 — narrow state-transition audit

Base `2c57ece0b4cd899b4f62bc7b89cf5ba3417e8615` (R13-Launch). This patch adds no performance candidate, shader arithmetic, experimental default, quantization change or R4 change. Review on a fresh Git worktree of the exact base; no changes to the supplied repository or driver.

## Findings and fixes

- `graph_drop` previously removed the graph map entry before `flush`. During an enclosing recording `flush` rejects; the failed drop still evicted the graph and lost its handle. Reject graph drop/clear inside recording or batch, and remove the map entry only after a successful flush. A failed driver submission is not a guarantee that the backend can resume; this fixes map ordering only.
- `dtoh` previously recorded a staging copy into the active recording before `flush` rejected host readback. If the caller handled the error, a future replay could execute the unexpected copy. Reject readback before any command recording in a graph or batch. Keep the existing diagnostic readbacks in `beside` (used by R8).
- Nested batch previously cleared the outer batch flag on return. Reject nesting before changing it. Reject nested graphs before invalidating the enclosing graph's Q8 keys; recheck state after taking the Q8 locks. A graph inside a transfer overlap (`beside`) is now explicitly rejected. Release allocated graph command buffers on begin/end failure and closure error; failed recordings are not submitted.
- `dtod` allowed partially overlapping regions of the same Vulkan buffer, which are invalid for vkCmdCopyBuffer. Reject before recording, preserve nonoverlapping intra-buffer copies and treat identical regions as no-ops. This is not a memmove implementation. No evidence that ordinary Qwen35B calls overlap; other models require regression tests too.

The physical `rejected_graph_operations_are_inert` test checks that a rejected graph drop does not evict a reusable graph, host readback preserves its host sentinel, nested closures are not executed, enclosing batch and graph still produce the expected outputs, and same-buffer copy rules hold. The existing `q8_graph_cache_lifetime` now additionally checks that rejecting a nested graph does not force the enclosing ordered batch off the Q8 contract. The physical gate requires both tests to actually execute once with exactly one success marker each. New offline checks inspect source order and drive that gate with mock Cargo (one valid run, five rejected variants); they do not execute Rust/Vulkan.

## Evidence from this pass

Logs in `validation-r13-final/`. 96 tests completed successfully in complete suites/selections: R9 10, R10 10, R11 6, R11U1 audit 3, R12 19, R12 audit 6, R13 8, R13 audit Gates 7, R13 Launch 8, R13 Launch2 6, KLD 3, R6 selected 10. R13 MockCampaign was interrupted at the environment timeout after four successful individual negative tests; not counted in the 96. R6 full CLI mock was not rerun this pass (it passed in the prior Launch report). Bash syntax, Python AST and Git whitespace checks passed. All shaders unchanged. No Rust/Cargo/glslc/spirv-val available, no physical PS5. None of the new physical tests were run. Existing 9,072 GPU parity cases remain mandatory.

## Remaining risks / scope

This is a targeted review, not proof of all Vulkan resource lifetimes. General buffer ownership across graph captures and recovery from device loss/failed waits (including Mem::drop), arbitrary device-resident routed indices, and potential concurrency outside the expected single ordered caller require separate design and physical validation; do not infer they are fixed by the Q8 changes. In particular, never treat a failed GPU submission as recoverable without a clean restart. The new guards may reject previously accepted unsupported calls; measure R11 throughput and exercise other models before a general release. No actual Qwen35B regression attributed to the findings so far.

## Launch

Use the original R13-Launch mission on this new HEAD. The wrapper `test-r13-audit.sh --gpu` now invokes both graph checks automatically before LUT parity. Capture the entire command output, for example `set -o pipefail; bash tools/ps5/test-r13-audit.sh --gpu results/r13-launch2-parity 2>&1 | tee results/r13-launch2-gpu.log`. Stop on a failure; preserve evidence. First compile, offline checks, CPU checks, GPU checks, batch quarantine, then R11 IQ32 revalidation and R12/R13 A/B tests. Keep R4 off, MTP off, driver and swap configuration unchanged. Do not treat offline tests as permission to promote experimental candidates.
