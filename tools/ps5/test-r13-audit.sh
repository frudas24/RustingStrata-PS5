#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
export PYTHONDONTWRITEBYTECODE=1
args=(--locked --release --target "$PS5_TARGET" --features ps5-r2)
case "${1:---offline}" in
  --offline)
    python3 tools/ps5/check-r13-launch.py
    python3 tools/ps5/check-r13-audit.py
    python3 tools/ps5/check-r13.py
    bash tools/ps5/test-r12-audit.sh --offline
    python3 tools/ps5/check-r6.py
    ;;
  --cpu)
    log=$(mktemp); trap 'rm -f "$log"' EXIT
    cargo test "${args[@]}" --lib r13_audit_ -- --nocapture 2>&1 | tee "$log"
    grep -Eq '^test result: ok\. 2 passed; 0 failed; 0 ignored;' "$log"
    bash tools/ps5/test-r12-audit.sh --cpu
    ;;
  --gpu)
    [[ $# == 2 ]] || { echo 'usage: test-r13-audit.sh --gpu NEW_OUTPUT_DIR' >&2; exit 2; }
    bash tools/ps5/test-r11u1-audit.sh --gpu
    bash tools/ps5/test-r7.sh
    # Run the explicit physical Q8 regression before the longer campaign.
    while IFS= read -r name; do unset "$name"; done < <(compgen -A variable STRATA_)
    export STRATA_PS5=1 STRATA_VK_NO_COOP=1 STRATA_TEST_BACKENDS=vulkan
    log=$(mktemp); trap 'rm -f "$log"' EXIT
    cargo test "${args[@]}" --test r13_launch q8_graph_cache_lifetime -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$log"
    grep -Fqx 'R13 LAUNCH Q8 GRAPH PASS: replay, discard and scratch growth' "$log"
    grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored;' "$log"
    bash tools/ps5/test-r13-lut.sh "$2"
    ;;
  *) echo 'usage: test-r13-audit.sh [--offline|--cpu|--gpu NEW_OUTPUT_DIR]' >&2; exit 2;;
esac
