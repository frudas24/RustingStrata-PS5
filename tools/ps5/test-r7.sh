#!/usr/bin/env bash
# Physical regression gates; never report success on a skipped test.
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
export STRATA_PS5=1 STRATA_TEST_BACKENDS=vulkan STRATA_PS5_ZERO_COPY=0 STRATA_PS5_SCHED=off
export STRATA_PS5_GEMV_WG=256 STRATA_PS5_SMALL_GEMM=0 STRATA_PS5_ATTN_WG=0
unset STRATA_LLAMA_NUMERICS STRATA_KV_Q8 STRATA_KV STRATA_KV_K STRATA_KV_V STRATA_VK_TURBO STRATA_KV_HOST
args=(--locked --release --target "$PS5_TARGET" --features ps5-r2)
for numerics in default llama; do
  if [[ $numerics == llama ]]; then export STRATA_LLAMA_NUMERICS=1; else unset STRATA_LLAMA_NUMERICS; fi
  cargo test "${args[@]}" --test r7_attention_empty masked_decode_has_defined_output -- --ignored --exact --nocapture --test-threads=1
done
unset STRATA_LLAMA_NUMERICS
cargo test "${args[@]}" --test r7_moe_mailbox publishes_activations_when_any_route_misses -- --ignored --exact --nocapture --test-threads=1
