#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
out=${1:-$PS5_ROOT/results/ps5-r7-gemm-parity}
mkdir -p "$out"; out=$(cd "$out" && pwd)
export STRATA_PS5=1 STRATA_VK_NO_COOP=1 STRATA_PS5_GEMV_WG=256 STRATA_PS5_SMALL_GEMM=0
export STRATA_PS5_ATTN_WG=0 STRATA_PS5_ZERO_COPY=0 STRATA_PS5_SCHED=off STRATA_PS5_R6=1
unset STRATA_LLAMA_NUMERICS STRATA_MMQ STRATA_KV_Q8 STRATA_KV STRATA_KV_K STRATA_KV_V STRATA_VK_TURBO STRATA_KV_HOST
args=(--locked --release --target "$PS5_TARGET" --features ps5-r2 --test r7_gemm)
cargo test "${args[@]}" shared_memory_bounds
for tile in 0 64; do
  rm -f "$out/$tile.bin"
  STRATA_PS5_R7_GEMM_TILE=$tile PS5_R7_GEMM_FILE="$out/$tile.bin" \
    cargo test "${args[@]}" fingerprint -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$out/$tile.log"
  test -s "$out/$tile.bin"
done
! grep -Fq 'PS5 R6 exercised: gemmi_dp_ps5_64' "$out/0.log"
grep -Fq 'PS5 R6 exercised: gemmi_dp_ps5_64' "$out/64.log"
cmp "$out/0.bin" "$out/64.bin"
sha256sum "$out/0.bin" "$out/64.bin" > "$out/sha256.txt"
echo 'PASS: dense64 physically exercised and bit-identical to dense128.'
