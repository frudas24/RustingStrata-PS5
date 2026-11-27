#!/usr/bin/env bash
set -euo pipefail
export STRATA_PS5_R7_GEMM_TILE=0
export STRATA_PS5_ZERO_COPY=0
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
mode=${1:---cpu}
tmp=$(mktemp -d); trap 'rm -rf -- "$tmp"' EXIT
cc -O3 -march=znver2 -Wall -Wextra -Werror tools/ps5/simd_check.c -o "$tmp/simd_check"
"$tmp/simd_check"
cargo test --locked --release --target "$PS5_TARGET" --lib cpu_moe::tests
cargo test --locked --release --target "$PS5_TARGET" --lib memory::tests
if [[ $mode == --gpu ]]; then
  export STRATA_TEST_BACKENDS=vulkan STRATA_PS5=1 STRATA_PS5_GTT=1 STRATA_PS5_DIRECT=1
  cargo test --locked --release --target "$PS5_TARGET" --features ps5-r2 --test backend ps5_ -- --ignored --nocapture --test-threads=1
  cargo test --locked --release --target "$PS5_TARGET" --features ps5-r2 --test backend memory_ops_round_trip -- --nocapture
elif [[ $mode != --cpu ]]; then echo 'use --cpu or --gpu' >&2; exit 2; fi
