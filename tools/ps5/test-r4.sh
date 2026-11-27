#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 && -r $1 ]] || { echo 'usage: test-r4.sh GENERIC-MOE.gguf' >&2; exit 2; }
export STRATA_MODEL=$(realpath "$1")
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
export STRATA_PS5=1 STRATA_PS5_GTT=1 STRATA_PS5_DIRECT=1 STRATA_TEST_BACKENDS=vulkan
export STRATA_PS5_GEMV_WG=256 STRATA_PS5_SMALL_GEMM=0 STRATA_PS5_ZERO_COPY=0
export STRATA_PS5_SCHED_SCRATCH_MB=64 STRATA_PS5_SCHED_MAX_ROWS=32 STRATA_FIXED_EXPERTS=1 STRATA_EXPERT_MB=0
export RAYON_NUM_THREADS=${PS5_THREADS:-6}
unset STRATA_LLAMA_NUMERICS STRATA_MMQ STRATA_SCALAR STRATA_NAN_TRACE
args=(--locked --release --target "$PS5_TARGET" --features ps5-r2)
cargo test "${args[@]}" --lib ps5_r4::tests
tmp=$(mktemp -d); trap 'rm -rf -- "$tmp"' EXIT
for mode in off gpu half cpu auto; do
  STRATA_PS5_SCHED=$mode PS5_R4_PARITY_FILE="$tmp/$mode.json" cargo test "${args[@]}" --test ps5_r4 prefill_scheduler_parity -- --ignored --exact --nocapture --test-threads=1
  test -s "$tmp/$mode.json"
done
python3 tools/ps5/parity-r4.py "$tmp"
