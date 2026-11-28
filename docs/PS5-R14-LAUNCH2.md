# R14-Launch2 — defensive audit and independent AnyPS5 research review

Base: `0531e8fd3a5370199e4eaac560c2eefc753e22d0` (R14-Launch).
Code: `925566f092498456b7c155bb186fa717720bd55c`.
No shader, GEMM selector, default flags, production arithmetic or R4 quarantine changes.

## Corrections

- Vulkan `Mem::drop`: failed flush is not evidence of safe retirement. Retain the Vulkan allocation, device reference and imported host pages on failure; report `Vulkan buffer retirement failed; retaining allocation until process exit`. This is an intentional exceptional leak, not a recovery mechanism. The campaign must stop and restart the process on this marker. This does not implement general deferred destruction, resource tracking of all cached graphs, device poisoning or recovery from device loss.
- Free never-submitted command buffers on failed begin/end in both compute and transfer paths. No claim of recovery after failed submit/wait. Reject host writes during graph recording before changing `step_pos`; make an empty write a no-op after validation. The existing physical rejected-operation test now exercises htod too.
- Validate KV slot index and reject shrinking with `keep=true`; the current callers already use `keep=false` for reset/shrink. Extend the ignored physical two-slot test to require inert rejection. No atomic transaction across split devices.
- R14 raw-logits verification hashes the bytes actually checked, with before/after file identity. Pin holdout, corpus and referenced benchmark evidence identity during quality validation. This is a local mutation detector, not an immutable filesystem snapshot or cryptographic hardware attestation. Reject resource retirement marker in R6-derived evidence and R14/graph gates.
- Test reference `f16tz`: allow correction from smallest binary16 subnormal to signed zero (`mag > 0` rather than `mag > 1`). The new Rust test checks 31,743 adjacent finite positive pairs with both signs (317,430 assertions), plus overflow/infinity for both signs. Standalone extracted exact test and helper passed with the real `half` crate, and the previous predicate failed at `bits=0x0000 input=2.9802326e-8`. This is a CPU reference test, not a GPU conversion measurement.

## AnyPS5 review (separate project; no source copied)

Input archive SHA256: `0319b4bd965d583b0607d4e988328a9fa124e53cf3ca5b3f299eea3cb02480a5`.
Reviewed README, `docs/dev/ARCHITECTURE.md`, `docs/dev/HW_ORACLE.md`, `tools/hw-oracle/probes.catalog`, `tools/hw-oracle/probe_catalog.py`, shader recompiler, graphics resource-management and execution-profile files. Its README states GPLv2-only; RustingStrata has its own license. Keep implementations separate; apply independent ideas, not copied code.

- HW Oracle: 33 probes, 131 input rows, wave32. Useful differential methodology, not measurements on our console. The HSA guard expects chip ID 0x13fc and rejects 0x13fe; our previous Vulkan log shows PCI ID 1002:13db and gfx1013. Do not equate identifiers across APIs or disable the guard. No HSA/driver/firmware installation in this audit.
- Float16ConversionProbe and ShaderDeviceProfile: inspire independent Vulkan numeric probes for binary16 RTNE/RTZ, signed zero, denormals, FMA and reduction contracts. Use fixed input bytes and bit/ULP oracles; correlate the first changed boundary in R4/DeltaNet; keep R4 off. Do not treat other AMD GPUs as gfx1013 measurements.
- DeferredDestruction, BarrierBatch, MapImportGate: principles for a later resource-retirement and synchronization audit, with explicit resource generation and last-use tracking. Do not remove barriers merely because they appear repetitive.
- AdaptiveStaging and PerformanceTrace: possible controlled study of model load/staging phase (previous profile had about 25 s cumulative staging write); measure warm/cold copies and memory pressure separately. Do not change the scored R14 campaign.
- No evidence that this archive exposes independent ACE/SDMA queues or provides VCN LLM acceleration; no promotion of those ideas.

## Verification

Toolchain: supplied offline Rust/Cargo 1.100.0, shaderc and SPIRV-Tools. Thin LTO, target x86_64-unknown-linux-gnu, feature ps5-r2. No Vulkan physical device here. `cargo check --release --tests` succeeded before adding the isolated FP16 test; the subsequent real release test build succeeded in 3m24s: library 189 passed/6 ignored, executable 47 passed, GGUF 11 passed, R13 graph tests 2 ignored, R14 context 1 ignored, R14 dense 1 passed/1 ignored. The separate kernels integration executable compiled and its exact FP16 test passed (42 filtered). Total 249 passed, 0 failed, 10 explicitly ignored. The standalone FP16 positive control is not counted twice. Existing unrelated unused-import/parentheses warnings remain. 90 SPIR-V modules passed Vulkan 1.3 validation. Python: R14 23/23 and inherited 104/104 (127 total); 14,648 independent geometries, 936 rejected marker mutations; mock graph gate repeated after the retirement negative was added, not counted twice. Logs normalize trailing whitespace only, except where specifically noted. No external GGUF/device-dependent early returns count as evidence of physical inference.

## Launch

Follow R14 and R14-Launch docs, using this final HEAD and the same model, driver, swap, flags and corpus. Stop immediately on retirement failure, device loss, synchronization failure, or invalid evidence. Do not try to recover in the same process. Build locally on PS5, run inherited regression, context, KV growth and dense parity gates, then original paired campaigns. Additional CPU test:

```bash
source tools/ps5/env.sh
cargo test --locked --release --target "$PS5_TARGET" --features ps5-r2 --test kernels f16_trunc_reference_boundaries -- --exact
```

No physical GPU tests or measured performance improvements claimed here. No full checkpoint. AnyPS5-inspired research is independent and deferred until after R14 measurements.
