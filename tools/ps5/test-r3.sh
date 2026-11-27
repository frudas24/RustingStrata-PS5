#!/usr/bin/env bash
# Strict physical test; without import the zero-copy gate must not pass.
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
export STRATA_PS5=1 STRATA_PS5_GTT=1 STRATA_PS5_DIRECT=1 STRATA_TEST_BACKENDS=vulkan
export STRATA_PS5_ZERO_COPY_MB=512
args=(--locked --release --target "$PS5_TARGET" --lib)
cargo test "${args[@]}" ps5_r3::tests::windows
tmp=$(mktemp -d); trap 'rm -rf -- "$tmp"' EXIT
for case in baseline fallback strict; do
  flag=1; required=0
  case "$case" in baseline) flag=0;; strict) required=1;; esac
  STRATA_PS5_ZERO_COPY=$flag PS5_R3_REQUIRE_IMPORT=$required cargo test "${args[@]}" vulkan::ps5_r3::tests::gpu_import_parity -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$tmp/$case.log"
  grep -q 'test result: ok. 1 passed;' "$tmp/$case.log"
  if [[ $case == strict ]]; then grep -q 'PS5 R3 test: imported=true' "$tmp/$case.log"; fi
done
