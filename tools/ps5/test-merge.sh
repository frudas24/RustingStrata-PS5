#!/usr/bin/env bash
# R6U1: compile all test targets with real shaders, then run CPU-only checks. No physical performance claims.
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
command -v cargo >/dev/null
command -v "${GLSLC:-glslc}" >/dev/null
command -v "${SPIRV_VAL:-spirv-val}" >/dev/null
args=(--locked --release --target "$PS5_TARGET" --features ps5-r2)
cargo test "${args[@]}" --no-run
cargo test "${args[@]}" --lib cpu_moe::tests
cargo test "${args[@]}" --lib memory::tests
cargo test "${args[@]}" --lib expert_cache::tests
cargo test "${args[@]}" --lib vulkan::spv_smem::tests
cargo test "${args[@]}" --test ps5_r5 shared_memory_bounds
python3 tools/test_kld_compare.py
bash tools/ps5/test-r6.sh --cpu
printf '%s\n' 'PASS: compile and CPU checks. Physical Vulkan tests are separate.'
