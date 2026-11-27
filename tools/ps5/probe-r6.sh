#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
mode=${1:---cpu}
[[ $mode == --cpu || $mode == --gpu ]] || { echo 'usage: probe-r6.sh [--cpu|--gpu]' >&2; exit 2; }
tmp=$(mktemp -d); trap 'rm -rf -- "$tmp"' EXIT
cc -O3 -march=znver2 -Wall -Wextra -Werror tools/ps5/probe-r6.c -o "$tmp/probe"
"$tmp/probe"
if [[ $mode == --gpu ]]; then
  export STRATA_PS5=1 STRATA_PS5_GTT=1 STRATA_PS5_DIRECT=1 STRATA_PS5_ZERO_COPY=0 STRATA_PS5_SCHED=off STRATA_PS5_ATTN_WG=0 STRATA_PS5_GEMV_WG=256 STRATA_PS5_SMALL_GEMM=0 STRATA_VK_NO_COOP=1
  cargo test --locked --release --target "$PS5_TARGET" --features ps5-r2 --test ps5_r6 transfer_probe -- --ignored --exact --nocapture --test-threads=1
fi
