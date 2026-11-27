# PS5 R6 — empirical calibration and PGO (experimental)

R6 extends the complete R5 checkpoint. The existing `rustingstrata calibrate` remains intact. R6 deliberately uses the same public `bench` path and introduces a separate, fail-closed launcher, `tools/ps5/autotune-r6.py`, rather than inventing a second inference engine. It targets the generic single-device GPU engine (e.g. supported Qwen models); dedicated DeepSeek, GLM, Gemma, Hy4, Kolibri and Inkling forward paths are rejected for R6 promotion because they do not yet produce the same fingerprint. Normal R1–R5 commands remain available for these architectures. This is source code, not a PS5-tested binary.

## Quick start

Requirements: working x86-64 Linux on the console, AMD Vulkan driver, Python 3.10+, `vulkaninfo`, Rust/Cargo, `glslc`, `spirv-val`, a C compiler and a supported local GGUF. No firmware, boot, kernel, clock or SSD configuration is changed.

```sh
M=/path/model.gguf
bash tools/ps5/build.sh
bash tools/ps5/test-r6.sh --cpu
bash tools/ps5/probe-r6.sh --gpu
# Validate earlier physical kernel parity before using the new tuning workflow:
bash tools/ps5/test-r2.sh
bash tools/ps5/test-r5.sh
python3 tools/ps5/autotune-r6.py tune --model "$M" --profile profiles/model-8k.json --out results/r6-8k --ctx 8192 --pp 32,64,512,2048,4096 --dry-run
python3 tools/ps5/autotune-r6.py tune --model "$M" --profile profiles/model-8k.json --out results/r6-8k --ctx 8192 --pp 32,64,512,2048,4096
python3 tools/ps5/autotune-r6.py run --model "$M" --profile profiles/model-8k.json chat -n 128
python3 tools/ps5/autotune-r6.py verify --model "$M" --profile profiles/model-8k.json --out results/recheck-8k
```

For a 64K experiment (only if the model and memory budget support it):

```sh
python3 tools/ps5/autotune-r6.py tune --model "$M" --profile profiles/model-64k-q8.json \
  --out results/r6-64k-q8 --ctx 65536 --kv q8 --pp 32,64,512,8192,32768,65024 --tg 32 --reps 2 --timeout 3600 --budget 12
python3 tools/ps5/autotune-r6.py run --model "$M" --profile profiles/model-64k-q8.json chat -n 128
```

Use a distinct directory for each run; the tuner will not overwrite an existing calibration. `--bin PATH` selects a specific binary on all subcommands, including the tuned launcher. The tuning script preserves explicit model/context/KV selection. `run` accepts only chat and text generate (`--prompt`), reply length and temperature; there is no automatic profile application to `serve` or arbitrary flags. The standard `run.sh` stays unchanged as a manual fallback. `--dry-run` does not open a model, run Vulkan, or write files. No model weights included.

## Knobs and scope

| Axis | Baseline (R1–R5 default preset) | Candidate values |
| --- | --- | --- |
| GEMV group | 256 | 64, 128, 256 |
| Small INT8 GEMM | off | off, on |
| Attention | original | original, 256, 512, 1024 |
| Rayon threads | 6 | 4, 6, 8, 16 |
| Prefill chunk | 512 | 64, 128, 256, 512, 1024 |
| GTT | requested | disabled, requested |
| Direct mapped upload | on | off, on |
| R3 import | off | off, on (only `--experimental`) |
| R4 prefill scheduler | off | off, half, auto (only `--experimental`) |

Always fixed: single GPU index 0, Vulkan, no cooperative matrices, short threshold 32, MTP 0, fixed expert placement, no LoRA, 768 MiB VRAM margin, 1536 MiB RAM reserve/margin, 256 MiB host cache, 16,777,216 attention budget. The same settings are used during tuned chat/generation; expert caching is intentionally fixed to avoid confounding the benchmark. R3 import is bounded to 512 MiB; R4 uses max 4 rows per expert and 64 MiB scratch. The search does **not** modify KV format, context length, model or runtime memory reserves. If R1 reports no driver memory-budget support, requested GTT may be unavailable; a GTT change cannot be scored unless a nonlocal device allocation is actually observed. The tuner will not silently use CPU instead of Vulkan. `--experimental` should be used only after the corresponding physical R3/R4 gates pass; unsupported candidates are rejected. For R4 see `test-r4.sh MODEL` on generic MoE; for R3 see `test-r3.sh` (it can fail legitimately on drivers rejecting file-backed import).

## Search and validation

1. Hash all GGUF shards and the executable (full SHA-256), record Vulkan enumeration/driver, Linux kernel and boot command line, CPU identity/flags, CPU affinity/governors, total memory and relevant loader environment. Recheck before committing a profile. This is deliberately expensive on a large model/slow external SSD; no sampled hash shortcut is used. Treat model files as immutable during calibration and execution. No hardware identification system guarantees detection of every local library replacement.
2. Run an R1–R5 baseline on the built-in prose corpus, with repeated benchmark measurements. A deterministic coordinate sweep considers at most `--budget` candidates (default 24), updating its incumbent only for a gain of at least 3% with no phase losing more than 5% against both the incumbent and the original baseline. This is a bounded heuristic, not an exhaustive search; knob interactions can hide a better combination.
3. Re-test a proposed winner and the baseline using A/B/B/A, separately on prose and a held-out Python code-review prompt. Each `bench` includes all requested prompt depths. Verify metadata, model, context, device, thread count, chunk, repeats, full requested generation length, valid rates and fingerprints. Require identical greedy token IDs and top-5 prefill token ordering in every repetition, finite prefill logits, and log-probability differences at most 0.05 nats for those five entries. These tests are stringent but do not prove full numerical equality, quality on all tasks or finite decode logits outside sampled outputs.
4. Objective: geometric mean of candidate/baseline prefill and decode rates across the prompt lengths. A winner needs >=3% aggregate gain on **both** corpora and >=95% of baseline on **every** metric and paired replicate. Replicate rates differing by more than 15% by min/max ratio fail closed. No speedup figure is claimed without data. A failed finalist leaves the baseline only after fresh baseline repetition on both corpora. If baseline cannot be validated, do not publish. Short EOS rejects a trial rather than timing fewer tokens; lower `--tg` or change workload if a model consistently terminates early.
5. Dispatch telemetry is opt-in (`STRATA_PS5_R6=1`). New kernel markers are emitted only after recording a real `cmd_dispatch`, not during pipeline preflight. Successful benchmarks additionally synchronize model work; this is not a GPU instruction-level trace. A GTT marker follows successful allocation and binding on a nonlocal heap. R3 must report actual imported bytes, and R4 must complete a mixed CPU/GPU prefill. A direct-mapped comparison needs nonzero mapped bytes. The controller rejects missing or no-op evidence. Existing R5 paths and timings are unchanged by default, except for the small opt-in branch in the recording path.
6. Each child has a timeout, process-group cleanup, and a sampled `/proc/meminfo` minimum-available guard (default 512 MiB); Linux `VmHWM` is sampled every 200 ms. These do not guarantee protection against sudden kernel OOM or measure GPU physical residency. Actual available memory is checked before launch. Normal tuned execution inherits the R1/R4 budgets, but does not run the calibration subprocess watchdog. Logs/results are kept per arm; profiles are written atomically with an advisory writer lock. Stale profiles fail closed when model/binary/driver/environment identity differs. `verify` runs a fresh A/B/B/A; failure revokes the profile rather than silently running it. There is no background revalidation or continuous runtime autotuning. A tuned chunk is static per profile, and different context/KV settings require separate profiles.

## Diagnostic microbenchmarks

`probe-r6.sh --cpu`: a single-thread AVX2 streaming read and randomized dependent load (16 KiB, 256 KiB, 4 MiB, 32 MiB, 64 MiB), with independent checksum and full-cycle validation. `--gpu` adds a 64 MiB transfer test: synchronized host-to-device and device-to-host, device-to-device payload bandwidth, and 4-byte synchronized-copy latency. These are end-to-end transfer diagnostics, not a measure of theoretical shader DRAM bandwidth. They do not choose the profile: the real model benchmark does. The GPU probe is an ignored Rust integration test that fails if the AMD PS5 profile cannot be opened. It does not silently skip.

## PGO (experimental)

```sh
rustup component add llvm-tools-preview
bash tools/ps5/pgo-r6.sh "$M" profiles/model-8k.json results/r6-pgo
```

The workflow follows the Rust compiler's documented instrumentation process (https://doc.rust-lang.org/rustc/profile-guided-optimization.html). It builds a separate `instrumented` target directory for Zen 2 with `profile-generate`, runs the selected model's prose workload, merges nonempty raw profiles using the LLVM tool bundled with the same Rust toolchain, then builds a separate `optimized` target directory with `profile-use`. The optimized binary never replaces the original. Compare at identical knobs with A/B/B/A on both corpora; the same output, dispatch and regression gates apply. The held-out code prompt is not used in PGO training. If the optimized candidate passes, retune using its `--bin` to produce a new profile because the binary hash changes. PGO affects Rust host machine code, not GLSL shader compilation. Toolchain and profile format must match. Fat LTO is inherited from R1. No PGO result or speedup has been measured on a PS5 yet.

## Validation performed in authoring environment

See `validation-r6.txt`. The simulated child-engine test deliberately manufactures timings and output signatures only to exercise the control path; its numbers have no hardware meaning. C/Python and shell checks do not replace Rust compilation, shader compilation or physical model validation. R6 introduces no new shader or dependency in Cargo. Physical gates remain required for R1–R5. Next: compile and run on actual PS5 with a real GGUF; collect and inspect failures, tune measured bottlenecks, then design any further R7 changes on evidence (e.g. per-layer dispatch, decode attention and a persistent CPU executor).
