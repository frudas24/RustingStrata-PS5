#!/usr/bin/env bash
# Fail closed: no GPU, failed build, or even one differing output bit fails this gate.
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
out=${1:-$PS5_ROOT/results/ps5-r2-parity}
mkdir -p "$out"; out=$(cd "$out" && pwd)
export STRATA_PS5=1 STRATA_PS5_GTT=1 STRATA_PS5_DIRECT=1 STRATA_VK_NO_COOP=1
export STRATA_PS5_ZERO_COPY=0
export STRATA_TEST_BACKENDS=vulkan
# Default numerics; use the same quantization mode for all cases.
unset STRATA_LLAMA_NUMERICS STRATA_MMQ
args=(--locked --release --target "$PS5_TARGET" --features ps5-r2)
cargo test "${args[@]}" --lib ps5_r2::tests
cargo test "${args[@]}" --test ps5_r2 small_gemm_shared_memory
for case in r1 r2-64 r2-128 r2-small r2-combined-64 r2-combined-128; do
  wg=256; small=0
  case "$case" in r2-64) wg=64;; r2-128) wg=128;; r2-small) small=1;; r2-combined-64) wg=64; small=1;; r2-combined-128) wg=128; small=1;; esac
  rm -f "$out/$case.bin"
  STRATA_PS5_GEMV_WG=$wg STRATA_PS5_SMALL_GEMM=$small PS5_R2_PARITY_FILE="$out/$case.bin" \
    cargo test "${args[@]}" --test ps5_r2 parity_fingerprint -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$out/$case.log"
  test -s "$out/$case.bin"
  cmp "$out/r1.bin" "$out/$case.bin"
done
sha256sum "$out"/*.bin | tee "$out/sha256.txt"
echo 'PASS: all R2 output bits match R1 on this device and driver.'
