#!/usr/bin/env bash
# Compare exact output bits for each KV mode (including 64K synthetic attention).
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
out=${1:-$PS5_ROOT/results/ps5-r5-parity}
mkdir -p "$out"; out=$(cd "$out" && pwd)
export STRATA_PS5=1 STRATA_VK_NO_COOP=1 STRATA_PS5_ZERO_COPY=0 STRATA_PS5_SCHED=off
export STRATA_PS5_GEMV_WG=256 STRATA_PS5_SMALL_GEMM=0
# Force query slicing on long cases: exercise q_off and the last partial slice.
export STRATA_VK_ATTN_BUDGET=262144
unset STRATA_VK_DP_OLD STRATA_LLAMA_NUMERICS STRATA_KV_Q8
args=(--locked --release --target "$PS5_TARGET" --features ps5-r2 --test ps5_r5)
cargo test "${args[@]}" shared_memory_bounds
for kv in f16 q8 q4 q2; do
  for wg in 0 256 512 1024; do
    name=$kv-$wg; rm -f "$out/$name.bin"
    STRATA_KV=$kv STRATA_PS5_ATTN_WG=$wg PS5_R5_PARITY_FILE="$out/$name.bin" \
      cargo test "${args[@]}" fingerprint -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$out/$name.log"
    test -s "$out/$name.bin"
    if ((wg)); then
      grep -Fq "PS5 R5 attention dispatch: attn_prefill_ps5_$wg" "$out/$name.log"
      cmp "$out/$kv-0.bin" "$out/$name.bin"
    fi
  done
done
sha256sum "$out"/*.bin > "$out/sha256.txt"
echo 'PASS: all R5 outputs match R4 bit for bit on this device/driver.'
