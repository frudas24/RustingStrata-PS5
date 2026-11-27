#!/usr/bin/env bash
# Physical regression gates; never report success on a skipped test.
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
while IFS= read -r name; do unset "$name"; done < <(compgen -A variable STRATA_)
export STRATA_PS5_R7_GEMM_TILE=0 STRATA_VK_NO_COOP=1
export STRATA_PS5=1 STRATA_TEST_BACKENDS=vulkan STRATA_PS5_ZERO_COPY=0 STRATA_PS5_SCHED=off
export STRATA_PS5_GEMV_WG=256 STRATA_PS5_SMALL_GEMM=0 STRATA_PS5_ATTN_WG=0
unset STRATA_LLAMA_NUMERICS STRATA_KV_Q8 STRATA_KV STRATA_KV_K STRATA_KV_V STRATA_VK_TURBO STRATA_KV_HOST
args=(--locked --release --target "$PS5_TARGET" --features ps5-r2)
log=$(mktemp); trap 'rm -f "$log"' EXIT
checked_one() {
  cargo test "${args[@]}" "$@" -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$log"
  grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored;' "$log"
}
for numerics in default llama; do
  if [[ $numerics == llama ]]; then export STRATA_LLAMA_NUMERICS=1; else unset STRATA_LLAMA_NUMERICS; fi
  checked_one --test r7_attention_empty masked_decode_has_defined_output
done
unset STRATA_LLAMA_NUMERICS
checked_one --test r7_moe_mailbox publishes_activations_when_any_route_misses
