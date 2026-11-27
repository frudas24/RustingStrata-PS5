# PS5 R3 — experimental immutable GGUF host import

Base: full PS5 R2 checkpoint. No firmware changes, kernel flags, clock changes, or writes to GGUF files. No new Rust dependencies or shaders. Default behavior remains R2 (R3 is disabled unless explicitly enabled).

## Why
On Linux AMD Vulkan, CPU and GPU may access the same physical GDDR6, but a normal Vulkan upload creates a separate device allocation. R3 tries `VK_EXT_external_memory_host` with a `HOST_ALLOCATION_EXT` pointer to an existing read-only GGUF mapping. This is a candidate for sharing the original pages, not proof of physical residency or performance. The extension does not guarantee that a file-backed pointer will be accepted. Some amdgpu userptr configurations reject file-backed or read-only ranges; even accepted host access can be slower than device-local memory. Do not force this feature or modify kernel safeguards.

Sources: https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_external_memory_host.html ; https://docs.vulkan.org/refpages/latest/refpages/source/VkImportMemoryHostPointerInfoEXT.html ; https://docs.vulkan.org/spec/latest/chapters/memory.html

## Contract
- Only Linux + AMD PS5 profile + GTT + memory budget + host import extension + importable storage/transfer-source/BDA buffer properties. Reject dedicated-only handles for now.
- Only explicitly immutable tensors on the generic `GpuModel::load_part` path, backed by a live original GGUF mapping. No derived heap tensors, no expert cache slots, no KV or scratch, no F16 conversion. DeepSeek/Gemma-specific loaders and DFlash are unchanged.
- Five formats: Q2_K, Q4_K, Q8_0, IQ3_XXS, IQ4_XS. Require >=64 KiB, byte length multiple of 4 and tensor address aligned to 16. Use aligned enclosing window wholly inside the file; reject partial last pages and arithmetic overflow. Allow original subviews only when they pass these checks.
- Query host pointer memory types and intersect with buffer requirements. Require host-visible, coherent, cached memory, no AMD uncached flags; verify heap budget, host headroom, maximum allocation size, live import cap. Default live span cap 512 MiB; tune only after passing tests. Count full enclosing windows conservatively, even if overlapping physical pages.
- GPU buffer lacks TRANSFER_DST; host upload, device copy destinations and memset reject read-only Buf. Shader BDA remains a trusted internal API; there is no hardware sandbox on arbitrary shader stores. Never use imported tensors as an output or mutate the model file while in use.
- A retained Arc keeps the source mmap alive until GPU completion and Vulkan destruction. `weights::cold` skips overlapping virtual ranges while an import is pending or live. On exceptional flush/idle failure retain the imported mapping and device for process lifetime rather than risk unmapping in-flight data. This is an intentional exceptional-path leak, not a recovery mechanism.
- Reject BDA zero, overflow, misalignment or overlap. On ordinary driver rejection disable further import attempts for that device, then use existing R1 mapped/staged upload; propagate device lost. Other rejected eligibility checks fall back without attempting the import. No hidden anonymous copy.

## Run
First build and pass R1/R2 gates. For R3-only builds with committed R1 shaders: `PS5_R2_BUILD=0 bash tools/ps5/build.sh`. Otherwise the default build includes R2 variants and needs glslc/spirv-val. `bash tools/ps5/test-r3.sh` runs baseline, permissive and strict real AMD Vulkan checks. Strict mode requires actual import; fallback alone is not a pass. Synthetic Q4_K fixture checks exact GEMV bits, mapping survival after Gguf drop, cleanup, source file integrity and destination write guards. Other formats require model-level parity before promotion.

`STRATA_PS5_ZERO_COPY=1 STRATA_PS5_ZERO_COPY_MB=512 bash tools/ps5/run.sh chat --model /path/model.gguf`

`bash tools/ps5/bench-r3.sh /path/model.gguf results-r3` runs baseline/import/import/baseline with the same R2 kernels and fixed experts. The parser requires a nonzero imported payload in both import runs and reports prefill/decode means; measure peak RSS, actual physical memory and GTT using the device's telemetry separately. Imported payload is cumulative, not peak physical savings. If the hardware refuses import, leave the flag disabled; R1 remains the fallback. A/B order only partly mitigates page cache/thermal effects; repeat with larger prompts and representative models, and verify model-level outputs, before drawing conclusions.

## Executed in the authoring environment
- `python3 tools/ps5/check-r3.py`: 100,000 modeled window cases, 23,220 eligible, plus 5 boundaries. Independent arithmetic oracle, NOT Rust/Vulkan execution.
- `python3 tools/ps5/check-r2.py`: 4,080 dispatch and 780 GEMM tail cases, pass.
- `cc -O3 -march=znver2 -Wall -Wextra -Werror tools/ps5/simd_check.c ...`: 1,000,000 vectors, 2,000,000 group sums, pass.
- Shell syntax and Python compilation, pass. Rust compiler and shader tools unavailable; full compile, Vulkan parity, actual file-backed import and physical measurements NOT executed.

## Next frontier R4
Use telemetry to decide per layer/expert whether local, GTT, or CPU is faster. Start with static calibration and a bounded placement policy, compare against fixed placement; do not assume an accepted zero-copy route is faster or change expert semantics. R3 remains opt-in pending device validation.
